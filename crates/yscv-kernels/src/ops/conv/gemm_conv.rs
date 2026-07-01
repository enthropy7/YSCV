//! GEMM-based general conv paths: im2col + BLAS GEMM, indirect (padding-free)
//! convolution, Winograd F(2x2,3x3), and the zero-padded NHWC conv.

use crate::core::scope_ctx::par_chunks_mut_dispatch;

use super::*;

// ---------------------------------------------------------------------------
// im2col + BLAS GEMM fast path for conv2d
// ---------------------------------------------------------------------------

/// Flatten each [kH, kW, C_in] input patch into a row of the im2col matrix.
///
/// Output `col` has shape [out_h * out_w, kH * kW * C_in] (row-major).
/// The input is NHWC layout (batch dimension already stripped by caller).
/// Indirect convolution: handles padding without allocating padded tensor.
/// Uses a zero buffer for out-of-bounds input positions.
/// This replaces pad_nhwc() + conv2d_nhwc_row() for padded group=1 Conv.
#[allow(unsafe_code)]
pub fn conv2d_nhwc_indirect_padded(
    input: &Tensor,
    kernel: &Tensor,
    bias: Option<&Tensor>,
    stride_h: usize,
    stride_w: usize,
    pad_top: usize,
    pad_left: usize,
    pad_bottom: usize,
    pad_right: usize,
    activation: Activation,
) -> Result<Tensor, KernelError> {
    let in_shape = input.shape();
    if in_shape.len() != 4 || kernel.shape().len() != 4 {
        return Err(KernelError::InvalidConvRank {
            input_rank: in_shape.len(),
            kernel_rank: kernel.shape().len(),
        });
    }
    let (batch, in_h, in_w, c_in) = (in_shape[0], in_shape[1], in_shape[2], in_shape[3]);
    let k_shape = kernel.shape();
    let (kh, kw, _k_cin, c_out) = (k_shape[0], k_shape[1], k_shape[2], k_shape[3]);

    let out_h = (in_h + pad_top + pad_bottom - kh) / stride_h + 1;
    let out_w = (in_w + pad_left + pad_right - kw) / stride_w + 1;
    let output_len = batch * out_h * out_w * c_out;
    let out_row_len = out_w * c_out;

    let mut output = AlignedVec::<f32>::uninitialized(output_len);
    let in_data = input.data();
    let ker_data = kernel.data();
    let bias_data = bias.map(Tensor::data);

    // Zero buffer for padded positions
    let zero_pixel = vec![0.0f32; c_in];

    let config = ParallelMatmulConfig::default();
    let process_row = |row_idx: usize, out_row: &mut [f32]| {
        let b = row_idx / out_h;
        let oy = row_idx % out_h;

        let batch_in_base = b * in_h * in_w * c_in;

        for ox in 0..out_w {
            let out_cell = &mut out_row[ox * c_out..(ox + 1) * c_out];

            // Init with bias
            if let Some(bv) = bias_data {
                out_cell.copy_from_slice(&bv[..c_out]);
            } else {
                out_cell.fill(0.0);
            }

            // Accumulate kernel positions with inline padding check
            for ky in 0..kh {
                let iy = oy * stride_h + ky;
                let in_y = iy as isize - pad_top as isize;

                for kx in 0..kw {
                    let ix = ox * stride_w + kx;
                    let in_x = ix as isize - pad_left as isize;

                    let input_pixel = if in_y >= 0
                        && (in_y as usize) < in_h
                        && in_x >= 0
                        && (in_x as usize) < in_w
                    {
                        let offset = batch_in_base + (in_y as usize * in_w + in_x as usize) * c_in;
                        &in_data[offset..offset + c_in]
                    } else {
                        &zero_pixel
                    };

                    let k_base = (ky * kw + kx) * c_in * c_out;
                    for ic in 0..c_in {
                        let iv = input_pixel[ic];
                        let kb = k_base + ic * c_out;
                        conv_fma_row(out_cell, &ker_data[kb..kb + c_out], iv);
                    }
                }
            }
        }

        apply_conv_activation_inplace(out_row, activation);
    };

    if should_parallelize_len(output_len, config.min_parallel_output_elements, None) {
        par_chunks_mut_dispatch(&mut output, out_row_len, process_row);
    } else {
        for (row_idx, out_row) in output.chunks_mut(out_row_len).enumerate() {
            process_row(row_idx, out_row);
        }
    }

    Tensor::from_aligned(vec![batch, out_h, out_w, c_out], output).map_err(Into::into)
}

/// im2col for NHWC input without padding.
/// Uses unsafe pointer arithmetic to avoid per-element bounds checks.
#[cfg(feature = "blas")]
#[allow(unsafe_code)]
fn im2col_nhwc(
    input: &[f32],
    in_w: usize,
    c: usize,
    kh: usize,
    kw: usize,
    stride_h: usize,
    stride_w: usize,
    out_h: usize,
    out_w: usize,
    col: &mut [f32],
) {
    let k = kh * kw * c;
    let in_row_stride = in_w * c;
    // SAFETY: all (oy*stride_h+ky, ox*stride_w+kx) are guaranteed in-bounds
    // because the output dimensions were computed from valid convolution params.
    unsafe {
        let inp = input.as_ptr();
        let mut dst = col.as_mut_ptr();
        for oy in 0..out_h {
            for ox in 0..out_w {
                for ky in 0..kh {
                    let src_row = inp.add((oy * stride_h + ky) * in_row_stride + ox * stride_w * c);
                    for kx in 0..kw {
                        std::ptr::copy_nonoverlapping(src_row.add(kx * c), dst, c);
                        dst = dst.add(c);
                    }
                }
            }
        }
        debug_assert_eq!(dst.offset_from(col.as_ptr()) as usize, out_h * out_w * k);
    }
}

/// im2col for a tile of output rows `[row_start .. row_start + tile_rows]`.
/// Uses unsafe pointer arithmetic for tight inner loops.
#[cfg(feature = "blas")]
#[allow(unsafe_code)]
fn im2col_nhwc_tile(
    input: &[f32],
    in_w: usize,
    c: usize,
    kh: usize,
    kw: usize,
    stride_h: usize,
    stride_w: usize,
    out_w: usize,
    row_start: usize,
    tile_rows: usize,
    col: &mut [f32],
) {
    let k = kh * kw * c;
    let in_row_stride = in_w * c;
    // SAFETY: all input indices are in-bounds (no padding case).
    unsafe {
        let inp = input.as_ptr();
        let mut dst = col.as_mut_ptr();
        for local_row in 0..tile_rows {
            let global_row = row_start + local_row;
            let oy = global_row / out_w;
            let ox = global_row % out_w;
            for ky in 0..kh {
                let src_row = inp.add((oy * stride_h + ky) * in_row_stride + ox * stride_w * c);
                for kx in 0..kw {
                    std::ptr::copy_nonoverlapping(src_row.add(kx * c), dst, c);
                    dst = dst.add(c);
                }
            }
        }
        debug_assert_eq!(dst.offset_from(col.as_ptr()) as usize, tile_rows * k);
    }
}

/// Conv2d via im2col + BLAS sgemm.
///
/// im2col matrix: [M, K] where M = out_h*out_w, K = kH*kW*C_in
/// kernel (already contiguous in NHWC): [K, N] where N = C_out
/// im2col + GEMM with fused bias+activation epilogue.
/// Works without BLAS — uses our custom blocked GEMM when BLAS is unavailable.
#[cfg(feature = "blas")]
pub(super) fn conv2d_im2col_gemm_fused(
    plan: &Conv2dPlan,
    input: &[f32],
    kernel: &[f32],
    bias: Option<&[f32]>,
    activation: Activation,
) -> Result<Tensor, KernelError> {
    let out_h = plan.out_h;
    let out_w = plan.out_w;
    let k = plan.kernel_h * plan.kernel_w * plan.in_channels;
    let m = out_h * out_w;
    let n = plan.out_channels;

    let epilogue = super::super::matmul::GemmEpilogue {
        bias: bias.map(|b| b.as_ptr()),
        activation,
        residual: None,
    };

    // For 1×1 conv with stride 1, input IS the im2col matrix — zero-copy.
    if plan.kernel_h == 1 && plan.kernel_w == 1 && plan.stride_h == 1 && plan.stride_w == 1 {
        #[allow(unsafe_code)]
        let mut output = AlignedVec::<f32>::uninitialized(m * n);
        super::super::matmul::blas_sgemm_fused(
            &input[..m * k],
            kernel,
            &mut output,
            m,
            k,
            n,
            epilogue,
        );
        return Tensor::from_aligned(vec![1, out_h, out_w, n], output).map_err(Into::into);
    }

    // Tile size: keep im2col_tile + output_tile in ~2 MB per thread.
    let bytes_per_row = (k + n) * std::mem::size_of::<f32>();
    let tile_m = (2usize * 1024 * 1024)
        .checked_div(bytes_per_row)
        .map(|rows| rows.max(1).min(m))
        .unwrap_or(m);

    #[allow(unsafe_code)]
    let mut output = AlignedVec::<f32>::uninitialized(m * n);

    if m > tile_m * 2 {
        super::super::super::scope_ctx::par_chunks_mut_dispatch(
            output.as_mut_slice(),
            tile_m * n,
            |tile_idx, out_chunk| {
                let row_start = tile_idx * tile_m;
                let actual_m = out_chunk.len() / n;
                if actual_m == 0 {
                    return;
                }
                thread_local! {
                    static COL_BUF: std::cell::RefCell<Vec<f32>> = const { std::cell::RefCell::new(Vec::new()) };
                }
                COL_BUF.with(|cell| {
                    let mut col_buf = cell.borrow_mut();
                    let needed = actual_m * k;
                    if col_buf.len() < needed {
                        col_buf.resize(needed, 0.0);
                    }
                    im2col_nhwc_tile(
                        input,
                        plan.in_w,
                        plan.in_channels,
                        plan.kernel_h,
                        plan.kernel_w,
                        plan.stride_h,
                        plan.stride_w,
                        out_w,
                        row_start,
                        actual_m,
                        &mut col_buf[..needed],
                    );
                    super::super::matmul::blas_sgemm_fused(
                        &col_buf[..needed],
                        kernel,
                        out_chunk,
                        actual_m,
                        k,
                        n,
                        epilogue,
                    );
                });
            },
        );
    } else {
        thread_local! {
            static MAIN_COL_BUF: std::cell::RefCell<Vec<f32>> = const { std::cell::RefCell::new(Vec::new()) };
        }
        MAIN_COL_BUF.with(|cell| {
            let mut col_buf = cell.borrow_mut();
            let needed = m * k;
            if col_buf.len() < needed {
                col_buf.resize(needed, 0.0);
            }
            im2col_nhwc(
                input,
                plan.in_w,
                plan.in_channels,
                plan.kernel_h,
                plan.kernel_w,
                plan.stride_h,
                plan.stride_w,
                out_h,
                out_w,
                &mut col_buf[..needed],
            );
            super::super::matmul::blas_sgemm_fused(
                &col_buf[..needed],
                kernel,
                &mut output,
                m,
                k,
                n,
                epilogue,
            );
        });
    }

    Tensor::from_aligned(vec![1, out_h, out_w, n], output).map_err(Into::into)
}

// ---------------------------------------------------------------------------
// Winograd F(2×2, 3×3) for non-Apple platforms
// ---------------------------------------------------------------------------
//
// On macOS, Apple Accelerate's AMX-backed sgemm is fast enough that Winograd's
// 16 smaller GEMMs lose more in BLAS efficiency than they gain in FLOPs.
// On other platforms (OpenBLAS, MKL, etc.) the arithmetic saving wins.

/// Transform 3×3 NHWC weights for Winograd F(2,3): G * g * G^T.
///
/// Input `kernel` is `[kH=3, kW=3, c_in, c_out]` (NHWC / HWIO).
/// Output is `[16, c_in, c_out]` (alpha-major, then c_in, then c_out).
#[cfg(not(target_os = "macos"))]
fn winograd_transform_weights_f32(kernel: &[f32], c_in: usize, c_out: usize) -> Vec<f32> {
    // G = [[1,0,0],[0.5,0.5,0.5],[0.5,-0.5,0.5],[0,0,1]]
    let mut out = vec![0.0f32; 16 * c_in * c_out];
    for ci in 0..c_in {
        for co in 0..c_out {
            // HWIO layout: kernel[ h*kw*c_in*c_out + w*c_in*c_out + ci*c_out + co ]
            let g = |r: usize, s: usize| {
                kernel[r * 3 * c_in * c_out + s * c_in * c_out + ci * c_out + co]
            };

            // G * g → 4×3
            let mut gg = [0.0f32; 12];
            for s in 0..3 {
                gg[s] = g(0, s);
                gg[3 + s] = 0.5 * (g(0, s) + g(1, s) + g(2, s));
                gg[6 + s] = 0.5 * (g(0, s) - g(1, s) + g(2, s));
                gg[9 + s] = g(2, s);
            }

            // (G * g) * G^T → 4×4
            let mut u = [0.0f32; 16];
            for r in 0..4 {
                let row = &gg[r * 3..r * 3 + 3];
                u[r * 4] = row[0];
                u[r * 4 + 1] = 0.5 * (row[0] + row[1] + row[2]);
                u[r * 4 + 2] = 0.5 * (row[0] - row[1] + row[2]);
                u[r * 4 + 3] = row[2];
            }

            // Scatter to [alpha, c_in, c_out]
            for a in 0..16 {
                out[a * c_in * c_out + ci * c_out + co] = u[a];
            }
        }
    }
    out
}

// Channel-vectorised elementwise combiners for the Winograd transforms. Plain
// slice loops so the autovectoriser emits the target SIMD (AVX/SSE/NEON) and
// falls back to scalar — one source, every architecture.
#[cfg(not(target_os = "macos"))]
#[inline]
fn v_sub(dst: &mut [f32], a: &[f32], b: &[f32]) {
    for ((o, &x), &y) in dst.iter_mut().zip(a).zip(b) {
        *o = x - y;
    }
}
#[cfg(not(target_os = "macos"))]
#[inline]
fn v_add(dst: &mut [f32], a: &[f32], b: &[f32]) {
    for ((o, &x), &y) in dst.iter_mut().zip(a).zip(b) {
        *o = x + y;
    }
}
#[cfg(not(target_os = "macos"))]
#[inline]
fn v_add3(dst: &mut [f32], a: &[f32], b: &[f32], c: &[f32]) {
    for (((o, &x), &y), &z) in dst.iter_mut().zip(a).zip(b).zip(c) {
        *o = x + y + z;
    }
}
#[cfg(not(target_os = "macos"))]
#[inline]
fn v_sub2(dst: &mut [f32], a: &[f32], b: &[f32], c: &[f32]) {
    for (((o, &x), &y), &z) in dst.iter_mut().zip(a).zip(b).zip(c) {
        *o = x - y - z;
    }
}
#[cfg(not(target_os = "macos"))]
#[inline]
fn v_add3_bias(dst: &mut [f32], a: &[f32], b: &[f32], c: &[f32], bias: &[f32]) {
    for ((((o, &x), &y), &z), &bb) in dst.iter_mut().zip(a).zip(b).zip(c).zip(bias) {
        *o = x + y + z + bb;
    }
}
#[cfg(not(target_os = "macos"))]
#[inline]
fn v_sub2_bias(dst: &mut [f32], a: &[f32], b: &[f32], c: &[f32], bias: &[f32]) {
    for ((((o, &x), &y), &z), &bb) in dst.iter_mut().zip(a).zip(b).zip(c).zip(bias) {
        *o = x - y - z + bb;
    }
}

/// Transformed + packed Winograd weights, memoised across inferences.
#[cfg(not(target_os = "macos"))]
struct WinogradCachedWeights {
    u: Vec<f32>,
    packed: Vec<std::sync::Arc<crate::PackedB>>,
}

/// Memoise the weight transform `U = GgGᵀ` and the per-α GEMM packing by kernel
/// pointer. Both depend only on the (constant) conv weights, so this turns
/// per-inference work into a one-time cost. Keyed on the weight data pointer
/// plus shape — fine for ONNX initializers, which outlive the session.
#[cfg(not(target_os = "macos"))]
fn winograd_cached_weights(
    kernel: &[f32],
    c_in: usize,
    c_out: usize,
) -> std::sync::Arc<WinogradCachedWeights> {
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex, OnceLock};

    static CACHE: OnceLock<
        Mutex<HashMap<(usize, usize, usize, usize), Arc<WinogradCachedWeights>>>,
    > = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let key = (kernel.as_ptr() as usize, kernel.len(), c_in, c_out);
    let mut guard = cache.lock().expect("winograd weight cache poisoned");
    Arc::clone(guard.entry(key).or_insert_with(|| {
        let u = winograd_transform_weights_f32(kernel, c_in, c_out);
        let packed = (0..16)
            .map(|a| {
                let u_slice = &u[a * c_in * c_out..(a + 1) * c_in * c_out];
                crate::pack_b_for_session(u_slice, c_in, c_out)
            })
            .collect();
        Arc::new(WinogradCachedWeights { u, packed })
    }))
}

/// Full Winograd F(2×2, 3×3) convolution for NHWC layout.
///
/// Only valid for 3×3 kernels with stride=1.
/// `input` NHWC `[batch, H, W, c_in]` (unpadded), `kernel` `[3, 3, c_in, c_out]`.
#[cfg(not(target_os = "macos"))]
fn winograd_conv2d_nhwc(
    input: &[f32],
    kernel: &[f32],
    bias: Option<&[f32]>,
    batch: usize,
    in_h: usize,
    in_w: usize,
    c_in: usize,
    c_out: usize,
    pad_top: usize,
    pad_left: usize,
    pad_bottom: usize,
    pad_right: usize,
    activation: Activation,
) -> Result<Tensor, KernelError> {
    use rayon::iter::{IntoParallelIterator, ParallelIterator};

    let padded_h = in_h + pad_top + pad_bottom;
    let padded_w = in_w + pad_left + pad_right;
    let out_h = padded_h - 2; // (padded_h - 3) / 1 + 1
    let out_w = padded_w - 2;

    // Number of 2×2 output tiles
    let tile_h = out_h.div_ceil(2);
    let tile_w = out_w.div_ceil(2);
    let n_tiles = tile_h * tile_w;

    // 1. Transformed + packed weights, memoised by kernel pointer. Both the
    //    weight transform and the GEMM packing depend only on the constant
    //    kernel, so they run once per weight instead of once per inference.
    let weights = winograd_cached_weights(kernel, c_in, c_out);
    let u: &[f32] = &weights.u;

    // SAFETY: every element written by the GEMM + output transform.
    #[allow(unsafe_code)]
    let mut output = AlignedVec::<f32>::uninitialized(batch * out_h * out_w * c_out);

    for b in 0..batch {
        use crate::GemmEpilogue;

        let in_batch = &input[b * in_h * in_w * c_in..(b + 1) * in_h * in_w * c_in];

        // 2. Input transform Bᵀ·d·B into layout [16, n_tiles, c_in]. All c_in
        //    channels of a tile are transformed together so the inner loops
        //    vectorise over the contiguous channel dimension (NHWC); tiles are
        //    spread across the rayon pool.
        // SAFETY: every element is written by the input transform below.
        #[allow(unsafe_code)]
        let mut v = AlignedVec::<f32>::uninitialized(16 * n_tiles * c_in);
        let zero_row = vec![0.0f32; c_in];
        let v_base = v.as_mut_ptr() as usize;
        let process_in_tile = |tile_idx: usize, t_scratch: &mut [f32]| {
            let th = tile_idx / tile_w;
            let tw = tile_idx % tile_w;
            let py0 = th * 2;
            let px0 = tw * 2;

            // Gather the 16 rows of the 4×4 input tile (zero row for OOB).
            let mut d_pos: [&[f32]; 16] = [zero_row.as_slice(); 16];
            for dy in 0..4 {
                for dx in 0..4 {
                    let iy = (py0 + dy).wrapping_sub(pad_top);
                    let ix = (px0 + dx).wrapping_sub(pad_left);
                    if iy < in_h && ix < in_w {
                        let off = iy * in_w * c_in + ix * c_in;
                        d_pos[dy * 4 + dx] = &in_batch[off..off + c_in];
                    }
                }
            }

            // Pass 1 — column transform M1 = Bᵀ·d, rows i=0..4 per column dx.
            for dx in 0..4 {
                let d0 = d_pos[dx];
                let d1 = d_pos[4 + dx];
                let d2 = d_pos[8 + dx];
                let d3 = d_pos[12 + dx];
                let base = |i: usize| (i * 4 + dx) * c_in;
                v_sub(&mut t_scratch[base(0)..base(0) + c_in], d0, d2);
                v_add(&mut t_scratch[base(1)..base(1) + c_in], d1, d2);
                v_sub(&mut t_scratch[base(2)..base(2) + c_in], d2, d1);
                v_sub(&mut t_scratch[base(3)..base(3) + c_in], d1, d3);
            }

            // Pass 2 — row transform V = M1·B, scattering the 16 αs into v.
            // SAFETY: each tile writes disjoint [tile_idx*c_in .. +c_in] runs
            // within each of the 16 α-blocks, so parallel tiles never overlap.
            for ty in 0..4 {
                let tb = ty * 4 * c_in;
                let (m0, m1, m2, m3) = (
                    &t_scratch[tb..tb + c_in],
                    &t_scratch[tb + c_in..tb + 2 * c_in],
                    &t_scratch[tb + 2 * c_in..tb + 3 * c_in],
                    &t_scratch[tb + 3 * c_in..tb + 4 * c_in],
                );
                for tx in 0..4 {
                    let voff = (ty * 4 + tx) * n_tiles * c_in + tile_idx * c_in;
                    #[allow(unsafe_code)]
                    let dst = unsafe {
                        std::slice::from_raw_parts_mut((v_base as *mut f32).add(voff), c_in)
                    };
                    match tx {
                        0 => v_sub(dst, m0, m2),
                        1 => v_add(dst, m1, m2),
                        2 => v_sub(dst, m2, m1),
                        _ => v_sub(dst, m1, m3),
                    }
                }
            }
        };
        (0..n_tiles).into_par_iter().for_each_init(
            || vec![0.0f32; 16 * c_in],
            |t_scratch, tile_idx| process_in_tile(tile_idx, t_scratch),
        );

        // 3. Batched GEMM: M[α] = V[α] · U[α], with V[α] [n_tiles, c_in],
        //    U[α] [c_in, c_out], M[α] [n_tiles, c_out].
        // SAFETY: every element is written by the 16 GEMMs (beta=0) below.
        #[allow(unsafe_code)]
        let mut m_buf = AlignedVec::<f32>::uninitialized(16 * n_tiles * c_out);
        let epilogue = GemmEpilogue::IDENTITY;
        let config = ParallelMatmulConfig::default();
        for a in 0..16 {
            use crate::matmul_2d_slices_fused_maybe_packed;

            let v_slice = &v[a * n_tiles * c_in..(a + 1) * n_tiles * c_in];
            let u_slice = &u[a * c_in * c_out..(a + 1) * c_in * c_out];
            let m_slice = &mut m_buf[a * n_tiles * c_out..(a + 1) * n_tiles * c_out];

            matmul_2d_slices_fused_maybe_packed(
                v_slice,
                n_tiles,
                c_in,
                u_slice,
                c_out,
                m_slice,
                Some(weights.packed[a].as_ref()),
                epilogue,
                config,
                None,
            );
        }

        // 4. Output transform: Aᵀ·M·A → 2×2 per tile, channel-vectorised, bias
        //    folded in. Activation is applied to the whole batch afterwards.
        let out_batch = &mut output[b * out_h * out_w * c_out..(b + 1) * out_h * out_w * c_out];
        let zero_bias = vec![0.0f32; c_out];
        let bias_slice: &[f32] = bias.unwrap_or(&zero_bias);
        let out_base = out_batch.as_mut_ptr() as usize;
        let process_out_tile = |tile_idx: usize, am_scratch: &mut [f32]| {
            let th = tile_idx / tile_w;
            let tw = tile_idx % tile_w;
            let oy0 = th * 2;
            let ox0 = tw * 2;
            // Clamp: last tile row/col may produce fewer than 2 valid outputs
            let valid_h = (out_h - oy0).min(2);
            let valid_w = (out_w - ox0).min(2);

            // Pass A — Aᵀ·M: am rows i=0..2 per column j=0..4.
            for j in 0..4 {
                let mb = |a: usize| {
                    let o = a * n_tiles * c_out + tile_idx * c_out;
                    &m_buf[o..o + c_out]
                };
                let (mm0, mm1, mm2, mm3) = (mb(j), mb(4 + j), mb(8 + j), mb(12 + j));
                let a0 = j * c_out;
                let a1 = (4 + j) * c_out;
                v_add3(&mut am_scratch[a0..a0 + c_out], mm0, mm1, mm2);
                v_sub2(&mut am_scratch[a1..a1 + c_out], mm1, mm2, mm3);
            }

            // Pass B — (Aᵀ·M)·A + bias, writing the valid 2×2 outputs.
            // SAFETY: each tile owns disjoint output pixels, so parallel tiles
            // never write the same location.
            let cs = c_out;
            let (am00, am01, am02, am03) = (
                &am_scratch[0..cs],
                &am_scratch[cs..2 * cs],
                &am_scratch[2 * cs..3 * cs],
                &am_scratch[3 * cs..4 * cs],
            );
            let (am10, am11, am12, am13) = (
                &am_scratch[4 * cs..5 * cs],
                &am_scratch[5 * cs..6 * cs],
                &am_scratch[6 * cs..7 * cs],
                &am_scratch[7 * cs..8 * cs],
            );
            let owrite = |oy: usize, ox: usize| {
                let o = (oy * out_w + ox) * c_out;
                #[allow(unsafe_code)]
                unsafe {
                    std::slice::from_raw_parts_mut((out_base as *mut f32).add(o), cs)
                }
            };
            v_add3_bias(owrite(oy0, ox0), am00, am01, am02, bias_slice);
            if valid_w == 2 {
                v_sub2_bias(owrite(oy0, ox0 + 1), am01, am02, am03, bias_slice);
            }
            if valid_h == 2 {
                v_add3_bias(owrite(oy0 + 1, ox0), am10, am11, am12, bias_slice);
                if valid_w == 2 {
                    v_sub2_bias(owrite(oy0 + 1, ox0 + 1), am11, am12, am13, bias_slice);
                }
            }
        };
        (0..n_tiles).into_par_iter().for_each_init(
            || vec![0.0f32; 8 * c_out],
            |am_scratch, tile_idx| process_out_tile(tile_idx, am_scratch),
        );

        // Apply activation on the whole batch output
        match activation {
            Activation::Silu => silu_slice_inplace(out_batch),
            Activation::Relu => relu_slice_inplace(out_batch),
            Activation::None => {}
        }
    }

    Tensor::from_aligned(vec![batch, out_h, out_w, c_out], output).map_err(Into::into)
}

/// Conv2D with implicit zero-padding — avoids separate padded tensor allocation.
///
/// `input` is NHWC `[batch, H, W, C_in]` (unpadded).
/// `kernel` is `[KH, KW, C_in, C_out]`.
/// Padding is applied virtually during im2col: out-of-bounds reads yield 0.
pub fn conv2d_nhwc_padded(
    input: &Tensor,
    kernel: &Tensor,
    bias: Option<&Tensor>,
    stride_h: usize,
    stride_w: usize,
    pad_top: usize,
    pad_left: usize,
    pad_bottom: usize,
    pad_right: usize,
    activation: Activation,
) -> Result<Tensor, KernelError> {
    let in_shape = input.shape();
    if in_shape.len() != 4 || kernel.shape().len() != 4 {
        return Err(KernelError::InvalidConvRank {
            input_rank: in_shape.len(),
            kernel_rank: kernel.shape().len(),
        });
    }
    let (batch, in_h, in_w, c_in) = (in_shape[0], in_shape[1], in_shape[2], in_shape[3]);
    let k_shape = kernel.shape();
    let (kh, kw, _k_cin, c_out) = (k_shape[0], k_shape[1], k_shape[2], k_shape[3]);

    let padded_h = in_h + pad_top + pad_bottom;
    let padded_w = in_w + pad_left + pad_right;
    let out_h = (padded_h - kh) / stride_h + 1;
    let out_w = (padded_w - kw) / stride_w + 1;
    let m = out_h * out_w;
    let k = kh * kw * c_in;
    let n = c_out;

    let in_data = input.data();
    let kernel_data = kernel.data();
    let bias_data = bias.map(|b| b.data());

    // Dedicated first-layer 3×3 stride-2 RGB microkernel. Shape matches
    // the opening Conv of the Siamese tracker (and most CV models that
    // take raw RGB input). Generic im2col+BLAS here has k=27 which kills
    // blocked-GEMM efficiency; the specialised kernel skips im2col and
    // packs the unroll directly into SIMD FMAs.
    if kh == 3
        && kw == 3
        && stride_h == 2
        && stride_w == 2
        && c_in == 3
        && (matches!(activation, Activation::None) || matches!(activation, Activation::Relu))
    {
        let output_len = batch * out_h * out_w * c_out;
        #[allow(unsafe_code)]
        let mut output = AlignedVec::<f32>::uninitialized(output_len);
        super::super::first_layer_3x3::conv2d_nhwc_3ch_3x3_s2_padded(
            in_data,
            kernel_data,
            bias_data,
            &mut output,
            batch,
            in_h,
            in_w,
            c_out,
            pad_top,
            pad_left,
            pad_bottom,
            pad_right,
            activation,
            None,
        );
        super::note_conv_path(super::ConvKernelPath::FirstLayerRgb3x3);
        return Tensor::from_aligned(vec![batch, out_h, out_w, c_out], output).map_err(Into::into);
    }

    // Winograd F(2×2,3×3) for 3×3 stride-1. Its 2.25× smaller GEMM is paid for
    // with per-tile transforms whose cost scales with (c_in + c_out), so it only
    // wins once both channel counts are large enough and there is enough spatial
    // output. The channel floor is per-arch: x86 wins broadly (floor 1), aarch64
    // shows no end-to-end gain over the blocked-GEMM im2col so Winograd is off by
    // default; override with YSCV_WINO_MIN_CH. macOS uses Accelerate's AMX sgemm.
    #[cfg(all(not(target_os = "macos"), target_arch = "aarch64"))]
    let default_min_ch = usize::MAX;
    #[cfg(all(not(target_os = "macos"), not(target_arch = "aarch64")))]
    let default_min_ch = 1usize;
    #[cfg(not(target_os = "macos"))]
    let wino_min_ch: usize = std::env::var("YSCV_WINO_MIN_CH")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default_min_ch);
    #[cfg(not(target_os = "macos"))]
    if kh == 3
        && kw == 3
        && stride_h == 1
        && stride_w == 1
        && out_h * out_w >= 256
        && c_in >= wino_min_ch
        && c_out >= wino_min_ch
    {
        super::note_conv_path(super::ConvKernelPath::Winograd3x3);
        return winograd_conv2d_nhwc(
            in_data,
            kernel_data,
            bias_data,
            batch,
            in_h,
            in_w,
            c_in,
            c_out,
            pad_top,
            pad_left,
            pad_bottom,
            pad_right,
            activation,
        );
    }

    super::note_conv_path(super::ConvKernelPath::Im2colGemm);
    // Tile size: keep im2col_tile + output_tile in ~2 MB per thread.
    let bytes_per_row = (k + n) * std::mem::size_of::<f32>();
    let tile_m = (2usize * 1024 * 1024)
        .checked_div(bytes_per_row)
        .map(|rows| rows.max(1).min(m))
        .unwrap_or(m);

    // SAFETY: every element is written by blas_sgemm (beta=0) + bias add.
    #[allow(unsafe_code)]
    let mut output = AlignedVec::<f32>::uninitialized(batch * m * n);

    for b in 0..batch {
        let in_slice = &in_data[b * in_h * in_w * c_in..(b + 1) * in_h * in_w * c_in];
        let out_batch = &mut output[b * m * n..(b + 1) * m * n];

        if m > tile_m * 2 {
            // Parallel tiled im2col + GEMM
            super::super::super::scope_ctx::par_chunks_mut_dispatch(
                out_batch,
                tile_m * n,
                |tile_idx, out_chunk| {
                    let row_start = tile_idx * tile_m;
                    let actual_m = out_chunk.len() / n;
                    if actual_m == 0 {
                        return;
                    }
                    thread_local! {
                        static PAD_COL_BUF: std::cell::RefCell<Vec<f32>> = const { std::cell::RefCell::new(Vec::new()) };
                    }
                    PAD_COL_BUF.with(|cell| {
                        let mut col_buf = cell.borrow_mut();
                        let needed = actual_m * k;
                        if col_buf.len() < needed {
                            col_buf.resize(needed, 0.0);
                        }
                        im2col_nhwc_padded_tile(
                            in_slice,
                            in_h,
                            in_w,
                            c_in,
                            kh,
                            kw,
                            stride_h,
                            stride_w,
                            pad_top,
                            pad_left,
                            out_w,
                            row_start,
                            actual_m,
                            &mut col_buf[..needed],
                        );
                        let epilogue = super::super::matmul::GemmEpilogue {
                            bias: bias_data.map(|b| b.as_ptr()),
                            activation,
                            residual: None,
                        };
                        super::super::matmul::blas_sgemm_fused(
                            &col_buf[..needed],
                            kernel_data,
                            out_chunk,
                            actual_m,
                            k,
                            n,
                            epilogue,
                        );
                    });
                },
            );
        } else {
            // Single tile
            thread_local! {
                static MAIN_PAD_COL_BUF: std::cell::RefCell<Vec<f32>> = const { std::cell::RefCell::new(Vec::new()) };
            }
            MAIN_PAD_COL_BUF.with(|cell| {
                let mut col_buf = cell.borrow_mut();
                let needed = m * k;
                if col_buf.len() < needed {
                    col_buf.resize(needed, 0.0);
                }
                im2col_nhwc_padded(
                    in_slice,
                    in_h,
                    in_w,
                    c_in,
                    kh,
                    kw,
                    stride_h,
                    stride_w,
                    pad_top,
                    pad_left,
                    out_h,
                    out_w,
                    &mut col_buf[..needed],
                );
                let epilogue = super::super::matmul::GemmEpilogue {
                    bias: bias_data.map(|b| b.as_ptr()),
                    activation,
                    residual: None,
                };
                super::super::matmul::blas_sgemm_fused(
                    &col_buf[..needed],
                    kernel_data,
                    out_batch,
                    m,
                    k,
                    n,
                    epilogue,
                );
            });
        }
    }

    Tensor::from_aligned(vec![batch, out_h, out_w, n], output).map_err(Into::into)
}

/// im2col with implicit zero-padding.  Out-of-bounds input reads are zero.
///
/// Interior optimization: for output rows where ALL kernel positions fall
/// within the valid input region, we skip per-element bounds checks entirely.
/// This covers ~90%+ of output positions for typical 3×3 pad=1 convolutions.
#[allow(unsafe_code)]
fn im2col_nhwc_padded(
    input: &[f32],
    in_h: usize,
    in_w: usize,
    c: usize,
    kh: usize,
    kw: usize,
    stride_h: usize,
    stride_w: usize,
    pad_top: usize,
    pad_left: usize,
    out_h: usize,
    out_w: usize,
    col: &mut [f32],
) {
    let k = kh * kw * c;
    // Compute the range of output rows/cols where ALL kernel taps are valid.
    // Interior: oy where oy*stride_h >= pad_top and oy*stride_h + kh - 1 < in_h + pad_top
    //   → oy >= ceil(pad_top / stride_h) and oy <= (in_h + pad_top - kh) / stride_h
    let oy_start = if pad_top > 0 {
        pad_top.div_ceil(stride_h)
    } else {
        0
    };
    let oy_end = if in_h + pad_top >= kh {
        (in_h + pad_top - kh) / stride_h + 1
    } else {
        0
    }
    .min(out_h);
    let ox_start = if pad_left > 0 {
        pad_left.div_ceil(stride_w)
    } else {
        0
    };
    let ox_end = if in_w + pad_left >= kw {
        (in_w + pad_left - kw) / stride_w + 1
    } else {
        0
    }
    .min(out_w);

    let in_row_stride = in_w * c;

    for oy in 0..out_h {
        let base_iy = oy * stride_h;
        let is_interior_y = oy >= oy_start && oy < oy_end;

        for ox in 0..out_w {
            let row_off = (oy * out_w + ox) * k;

            if is_interior_y && ox >= ox_start && ox < ox_end {
                // Interior: all kernel taps are valid — no bounds checks.
                let base_ix = ox * stride_w - pad_left;
                let base_iy_val = base_iy - pad_top;
                // SAFETY: we verified all (iy, ix) are in-bounds above.
                unsafe {
                    let mut dst = col.as_mut_ptr().add(row_off);
                    if stride_w == 1 {
                        // When stride_w==1, kernel taps along x are contiguous in NHWC
                        // layout, so we copy kw*c floats per kernel row instead of kw
                        // separate copies. For 3×3: 3 memcpys instead of 9.
                        let row_bytes = kw * c;
                        for ky in 0..kh {
                            let src_row = input
                                .as_ptr()
                                .add((base_iy_val + ky) * in_row_stride + base_ix * c);
                            std::ptr::copy_nonoverlapping(src_row, dst, row_bytes);
                            dst = dst.add(row_bytes);
                        }
                    } else {
                        for ky in 0..kh {
                            let src_row = input
                                .as_ptr()
                                .add((base_iy_val + ky) * in_row_stride + base_ix * c);
                            for kx in 0..kw {
                                std::ptr::copy_nonoverlapping(src_row.add(kx * c), dst, c);
                                dst = dst.add(c);
                            }
                        }
                    }
                }
            } else {
                // Border: some taps may be out of bounds.
                for ky in 0..kh {
                    let iy = (base_iy + ky) as isize - pad_top as isize;
                    for kx in 0..kw {
                        let ix = (ox * stride_w + kx) as isize - pad_left as isize;
                        let dst_off = row_off + (ky * kw + kx) * c;
                        if iy >= 0 && (iy as usize) < in_h && ix >= 0 && (ix as usize) < in_w {
                            let src_off = (iy as usize * in_w + ix as usize) * c;
                            col[dst_off..dst_off + c].copy_from_slice(&input[src_off..src_off + c]);
                        } else {
                            col[dst_off..dst_off + c].fill(0.0);
                        }
                    }
                }
            }
        }
    }
}

/// Tiled im2col with implicit padding — handles out-of-bounds as zero.
/// Same interface as `im2col_nhwc_tile` but with padding parameters.
///
/// Uses interior/border split: for output positions where all kernel taps
/// fall within valid input, skips bounds checks entirely via unsafe ptrs.
#[allow(unsafe_code)]
fn im2col_nhwc_padded_tile(
    input: &[f32],
    in_h: usize,
    in_w: usize,
    c: usize,
    kh: usize,
    kw: usize,
    stride_h: usize,
    stride_w: usize,
    pad_top: usize,
    pad_left: usize,
    out_w: usize,
    row_start: usize,
    tile_rows: usize,
    col: &mut [f32],
) {
    let k = kh * kw * c;
    let in_row_stride = in_w * c;

    // Interior bounds (same computation as non-tiled version).
    let oy_start = if pad_top > 0 {
        pad_top.div_ceil(stride_h)
    } else {
        0
    };
    let oy_end_val = if in_h + pad_top >= kh {
        (in_h + pad_top - kh) / stride_h + 1
    } else {
        0
    };
    let ox_start = if pad_left > 0 {
        pad_left.div_ceil(stride_w)
    } else {
        0
    };
    let ox_end = if in_w + pad_left >= kw {
        (in_w + pad_left - kw) / stride_w + 1
    } else {
        0
    }
    .min(out_w);

    for local_row in 0..tile_rows {
        let global_row = row_start + local_row;
        let oy = global_row / out_w;
        let ox = global_row % out_w;
        let row_off = local_row * k;

        let is_interior = oy >= oy_start && oy < oy_end_val && ox >= ox_start && ox < ox_end;

        if is_interior {
            let base_iy = oy * stride_h - pad_top;
            let base_ix = ox * stride_w - pad_left;
            // SAFETY: interior guarantees all (iy, ix) are in-bounds.
            unsafe {
                let mut dst = col.as_mut_ptr().add(row_off);
                if stride_w == 1 {
                    let row_bytes = kw * c;
                    for ky in 0..kh {
                        let src_row = input
                            .as_ptr()
                            .add((base_iy + ky) * in_row_stride + base_ix * c);
                        std::ptr::copy_nonoverlapping(src_row, dst, row_bytes);
                        dst = dst.add(row_bytes);
                    }
                } else {
                    for ky in 0..kh {
                        let src_row = input
                            .as_ptr()
                            .add((base_iy + ky) * in_row_stride + base_ix * c);
                        for kx in 0..kw {
                            std::ptr::copy_nonoverlapping(src_row.add(kx * c), dst, c);
                            dst = dst.add(c);
                        }
                    }
                }
            }
        } else {
            let base_iy = oy * stride_h;
            for ky in 0..kh {
                let iy = (base_iy + ky) as isize - pad_top as isize;
                for kx in 0..kw {
                    let ix = (ox * stride_w + kx) as isize - pad_left as isize;
                    let dst_off = row_off + (ky * kw + kx) * c;
                    if iy >= 0 && (iy as usize) < in_h && ix >= 0 && (ix as usize) < in_w {
                        let src_off = (iy as usize * in_w + ix as usize) * c;
                        col[dst_off..dst_off + c].copy_from_slice(&input[src_off..src_off + c]);
                    } else {
                        col[dst_off..dst_off + c].fill(0.0);
                    }
                }
            }
        }
    }
}

#[cfg(all(
    test,
    not(target_os = "macos"),
    any(target_arch = "aarch64", target_arch = "x86_64")
))]
mod tests {
    use super::*;

    fn assert_winograd_matches_indirect_padded(with_bias: bool, activation: Activation) {
        let batch = 2;
        let in_h = 8;
        let in_w = 9;
        let c_in = 3;
        let c_out = 5;
        let pad_top = 1;
        let pad_left = 1;
        let pad_bottom = 1;
        let pad_right = 1;

        let input_data: Vec<f32> = (0..batch * in_h * in_w * c_in)
            .map(|i| (i as f32 % 17.0) * 0.05 - 0.4)
            .collect();
        let kernel_data: Vec<f32> = (0..3 * 3 * c_in * c_out)
            .map(|i| (i as f32 % 11.0) * 0.03 - 0.15)
            .collect();

        let input = Tensor::from_vec(vec![batch, in_h, in_w, c_in], input_data).unwrap();
        let kernel = Tensor::from_vec(vec![3, 3, c_in, c_out], kernel_data).unwrap();
        let bias = with_bias.then(|| {
            let bias_data: Vec<f32> = (0..c_out).map(|i| i as f32 * 0.07 - 0.13).collect();
            Tensor::from_vec(vec![c_out], bias_data).unwrap()
        });

        let winograd = winograd_conv2d_nhwc(
            input.data(),
            kernel.data(),
            bias.as_ref().map(Tensor::data),
            batch,
            in_h,
            in_w,
            c_in,
            c_out,
            pad_top,
            pad_left,
            pad_bottom,
            pad_right,
            activation,
        )
        .unwrap();
        let indirect = conv2d_nhwc_indirect_padded(
            &input,
            &kernel,
            bias.as_ref(),
            1,
            1,
            pad_top,
            pad_left,
            pad_bottom,
            pad_right,
            activation,
        )
        .unwrap();

        assert_eq!(winograd.shape(), indirect.shape());
        for (idx, (&actual, &expected)) in winograd.data().iter().zip(indirect.data()).enumerate() {
            let diff = (actual - expected).abs();
            assert!(
                diff <= 1.0e-4,
                "with_bias={with_bias} activation={activation:?} idx={idx} actual={actual} expected={expected} diff={diff}"
            );
        }
    }

    #[test]
    fn winograd_conv2d_matches_indirect_padded() {
        assert_winograd_matches_indirect_padded(false, Activation::None);
    }

    #[test]
    fn winograd_conv2d_matches_indirect_padded_bias_only() {
        assert_winograd_matches_indirect_padded(true, Activation::None);
    }

    #[test]
    fn winograd_conv2d_matches_indirect_padded_activation_only() {
        for activation in [Activation::Relu, Activation::Silu] {
            assert_winograd_matches_indirect_padded(false, activation);
        }
    }

    #[test]
    fn winograd_conv2d_matches_indirect_padded_with_bias_and_activation() {
        for activation in [Activation::Relu, Activation::Silu] {
            assert_winograd_matches_indirect_padded(true, activation);
        }
    }
}
