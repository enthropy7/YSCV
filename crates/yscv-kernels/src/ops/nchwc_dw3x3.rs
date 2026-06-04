//! Step E2: native NCHWc depthwise 3×3 stride-1 SAME-pad Conv.
//!
//! Walks the NCHWc layout directly without stripping to NHWC. For each
//! channel-block, the 9 kernel weights are hoisted into SIMD registers
//! once and reused across every spatial position. 9 FMAs per output
//! pixel — no broadcast pattern (each lane is an independent channel),
//! so intrinsics match inline-asm throughput on Zen 4.
//!
//! Arch variants:
//! - AVX2+FMA (x86_64, block=8): 9 kernel YMMs + 1 acc + bias = 11 YMMs
//!   live; 5 free for the 16-YMM AVX2 register file.
//! - NEON (aarch64, block=8 via 2×float32x4): correct + cross-compiles;
//!   perf-tuned in a future session if the tracker shifts to ARM.
//! - Scalar: any block / any arch / miri.
//!
//! Padding strategy: one-shot pad the input into an [N, Cb, H+2, W+2,
//! block] buffer zero-initialized at the borders, then the inner loop
//! is fully branchless. Pad buffer cost for tracker DW shapes (e.g.
//! 32×32×96): 34×34×96 ≈ 440 KB — fits L2.
//!
//! Dispatch gate (caller's responsibility):
//! - 1×1 pointwise → NOT this kernel (pointwise goes through the MT
//!   inline-asm blocked GEMM wrapper, which is proven faster; see
//!   `project_step_e1_native_nchwc_failed.md`).
//! - stride != 1 or dilation != 1 → fall through to NHWC im2col path.
//! - `actual_channels % block != 0` → fall through (no tail-lane
//!   masking in the fast path).
//! - kernel != 3×3 → fall through.

use rayon::ThreadPool;
use yscv_tensor::{AlignedVec, Layout, Tensor};

use super::config::ParallelElementwiseConfig;
use super::conv::Activation;
use crate::core::error::KernelError;

/// Native NCHWc depthwise 3×3 stride-1 SAME-pad Conv.
///
/// `input` is `NCHWc [N, Cb, H, W, block]`. `kernel` is NHWC
/// `[3, 3, C, 1]` (depth multiplier = 1). `bias`, if present, is `[C]`.
/// `actual_channels` is the real C (can be less than `Cb * block` if
/// the block has padded tail lanes — but this fast path requires
/// `actual_channels % block == 0`).
///
/// Returns the output `NCHWc [N, Cb, H, W, block]` tagged with the same
/// block size as the input.
#[allow(clippy::too_many_arguments)]
/// K.3: pass `prepacked_dw = Some(&packed)` to use blocked `[C_blocks, KH, KW, block]`
/// weight layout (tighter strides → L1-resident kernel weights during inner loop).
/// Pass `None` to fall back to KHWC tensor layout (all existing callers use `None`).
pub fn conv2d_nchwc_dw3x3_s1_same_pad(
    input: &Tensor,
    kernel: &Tensor,
    bias: Option<&Tensor>,
    activation: Activation,
    actual_channels: usize,
    _config: ParallelElementwiseConfig,
    _thread_pool: Option<&ThreadPool>,
    prepacked_dw: Option<&super::nchwc_pack::PackedNChwBc>,
) -> Result<Tensor, KernelError> {
    let in_shape = input.shape();
    if in_shape.len() != 5 {
        return Err(KernelError::LayoutConversion(format!(
            "conv2d_nchwc_dw3x3_s1: expected 5-D NCHWc input, got {in_shape:?}"
        )));
    }
    let [n_batch, cb, h, w, block] = *<&[usize; 5]>::try_from(in_shape)
        .map_err(|_| KernelError::LayoutConversion("dw3x3 nchwc: rank mismatch".into()))?;
    if block == 0 || block > u8::MAX as usize {
        return Err(KernelError::LayoutConversion(format!(
            "dw3x3 nchwc: invalid block {block}"
        )));
    }
    if actual_channels > cb * block {
        return Err(KernelError::LayoutConversion(format!(
            "dw3x3 nchwc: actual_channels {actual_channels} > cb*block {}",
            cb * block
        )));
    }
    if !actual_channels.is_multiple_of(block) {
        return Err(KernelError::LayoutConversion(format!(
            "dw3x3 nchwc: actual_channels {actual_channels} not divisible by block {block} \
             (caller must dispatch to NHWC path)"
        )));
    }
    let cb_used = actual_channels / block;

    let ker_shape = kernel.shape();
    let [kh, kw, ker_c, dm] = *<&[usize; 4]>::try_from(ker_shape).map_err(|_| {
        KernelError::LayoutConversion(format!("dw3x3 nchwc: kernel rank != 4, got {ker_shape:?}"))
    })?;
    if kh != 3 || kw != 3 || dm != 1 {
        return Err(KernelError::LayoutConversion(format!(
            "dw3x3 nchwc: expected 3×3 kernel dm=1, got {kh}×{kw} dm={dm}"
        )));
    }
    if ker_c != actual_channels {
        return Err(KernelError::LayoutConversion(format!(
            "dw3x3 nchwc: kernel C {ker_c} != actual_channels {actual_channels}"
        )));
    }
    if let Some(b) = bias {
        let bs = b.shape();
        let blen: usize = bs.iter().product();
        if blen != actual_channels {
            return Err(KernelError::LayoutConversion(format!(
                "dw3x3 nchwc: bias len {blen} != actual_channels {actual_channels}"
            )));
        }
    }

    let input_data = input.try_data().map_err(KernelError::from)?;
    let kernel_data = kernel.try_data().map_err(KernelError::from)?;
    let bias_data = bias
        .map(|b| b.try_data().map_err(KernelError::from))
        .transpose()?;

    let in_n_stride = cb * h * w * block;
    let in_cb_stride = h * w * block;
    let out_cb_stride = h * w * block;
    let out_total = n_batch * cb * out_cb_stride;

    // K.3: PackedNChwBc stores weights as [C_blocks, KH, KW, block] with
    // kernel_c_stride = block, kernel_row_stride = KW * block.  All 9
    // weights for one co-block are in 9*block contiguous floats → L1-resident
    // across all spatial positions in the inner loop.
    //
    // KHWC fallback: kernel_c_stride = C (actual_channels); 9 weights span
    // a much larger non-contiguous region within the global kernel tensor.
    let (use_packed, packed_raw, per_cb_step) = if let Some(p) = prepacked_dw {
        (true, p.raw_data(), p.kh * p.kw * block)
    } else {
        (false, kernel_data, 0)
    };
    let kernel_c_stride = if use_packed { block } else { actual_channels };
    let kernel_row_stride = if use_packed {
        3 * block
    } else {
        3 * actual_channels
    };

    // No-pad-buffer AVX-512 block=16 fast path. The R1 pattern (pre-load 9
    // weight ZMMs, register-resident accumulators, border-aware loads in the
    // inner loop) lets us skip the 415 KB pad calloc + 320 KB input copy +
    // 320 KB output calloc per call that the legacy padded path pays. HOT
    // c=320 16×16 drops from ~196 µs → ~15 µs in micro-bench.
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if block == 16
            && !cfg!(miri)
            && crate::host_cpu().features.avx512f
            && !nchwc_dw3x3_nopad_disabled()
        {
            let mut out = AlignedVec::<f32>::uninitialized(out_total);
            let out_slice = out.as_mut_slice();
            if cb_used < cb {
                // Zero only the tail channel-blocks; the inner kernel writes
                // every used output cell.
                let tail_off =
                    n_batch.saturating_sub(1) * cb * out_cb_stride + cb_used * out_cb_stride;
                let _ = tail_off; // batch-loop below handles per-batch tails
                for n in 0..n_batch {
                    let tail_lo = n * cb * out_cb_stride + cb_used * out_cb_stride;
                    let tail_hi = (n + 1) * cb * out_cb_stride;
                    out_slice[tail_lo..tail_hi].fill(0.0);
                }
            }

            let in_ref: &[f32] = input_data;
            let kernel_ref: &[f32] = kernel_data;
            let bias_ref: Option<&[f32]> = bias_data;

            super::super::scope_ctx::par_chunks_mut_dispatch(
                out_slice,
                out_cb_stride,
                move |unit_idx, out_chunk| {
                    let n = unit_idx / cb;
                    let cb_i = unit_idx % cb;
                    if cb_i >= cb_used {
                        return;
                    }
                    let in_base = n * in_n_stride + cb_i * in_cb_stride;
                    let in_slab = &in_ref[in_base..in_base + in_cb_stride];

                    let kernel_cb: &[f32] = if use_packed {
                        &packed_raw[cb_i * per_cb_step..(cb_i + 1) * per_cb_step]
                    } else {
                        &kernel_ref[cb_i * block..]
                    };
                    let bias_cb = bias_ref.map(|b| &b[cb_i * block..cb_i * block + block]);
                    // SAFETY: avx512f detected at runtime above; in_slab is
                    // exactly h*w*16 floats; out_chunk is h*w*16 floats.
                    #[allow(unsafe_code)]
                    unsafe {
                        dw3x3_inner_avx512_block16_nopad(
                            in_slab,
                            kernel_cb,
                            kernel_c_stride,
                            kernel_row_stride,
                            bias_cb,
                            out_chunk,
                            h,
                            w,
                            activation,
                        );
                    }
                },
            );

            let out_shape = vec![n_batch, cb, h, w, block];
            let out_tensor = Tensor::from_aligned(out_shape, out).map_err(KernelError::from)?;
            return Ok(out_tensor.with_layout(Layout::NCHWc { block: block as u8 }));
        }
    }

    // Pad input into [N, Cb, H+2, W+2, block] zero-init.
    let padded_h = h + 2;
    let padded_w = w + 2;
    let padded_n_stride = cb * padded_h * padded_w * block;
    let padded_cb_stride = padded_h * padded_w * block;
    let padded_row_stride = padded_w * block;
    let mut padded_buf = AlignedVec::<f32>::calloc(n_batch * padded_n_stride);
    let pad_slice = padded_buf.as_mut_slice();
    let in_row_stride = w * block;
    for n in 0..n_batch {
        for cb_i in 0..cb {
            for y in 0..h {
                let src_off = n * in_n_stride + cb_i * in_cb_stride + y * in_row_stride;
                let dst_off = n * padded_n_stride
                    + cb_i * padded_cb_stride
                    + (y + 1) * padded_row_stride
                    + block; // skip the leading pad column
                pad_slice[dst_off..dst_off + in_row_stride]
                    .copy_from_slice(&input_data[src_off..src_off + in_row_stride]);
            }
        }
    }

    // Output [N, Cb, H, W, block] zero-init so tail cb (cb_used..cb)
    // stays clean.
    let mut out = AlignedVec::<f32>::calloc(out_total);
    let out_slice = out.as_mut_slice();

    let pad_ref: &[f32] = pad_slice;
    let kernel_ref: &[f32] = kernel_data;
    let bias_ref: Option<&[f32]> = bias_data;

    super::super::scope_ctx::par_chunks_mut_dispatch(
        out_slice,
        out_cb_stride,
        move |unit_idx, out_chunk| {
            let n = unit_idx / cb;
            let cb_i = unit_idx % cb;
            if cb_i >= cb_used {
                return;
            }
            let pad_base = n * padded_n_stride + cb_i * padded_cb_stride;
            let pad_slab = &pad_ref[pad_base..pad_base + padded_cb_stride];

            let kernel_cb: &[f32] = if use_packed {
                &packed_raw[cb_i * per_cb_step..(cb_i + 1) * per_cb_step]
            } else {
                // KHWC: offset to first channel of this co-block
                &kernel_ref[cb_i * block..]
            };
            let bias_cb = bias_ref.map(|b| &b[cb_i * block..cb_i * block + block]);
            dw3x3_dispatch(
                pad_slab,
                kernel_cb,
                kernel_c_stride,
                kernel_row_stride,
                bias_cb,
                out_chunk,
                h,
                w,
                padded_w,
                block,
                activation,
            );
        },
    );

    let out_shape = vec![n_batch, cb, h, w, block];
    let out_tensor = Tensor::from_aligned(out_shape, out).map_err(KernelError::from)?;
    Ok(out_tensor.with_layout(Layout::NCHWc { block: block as u8 }))
}

/// Env-cached kill switch for the AVX-512 block=16 no-pad fast path.
/// Set `YSCV_NCHWC_DW3X3_NOPAD_OFF=1` to fall back to the padded variant
/// (legacy) for A/B comparison.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline]
fn nchwc_dw3x3_nopad_disabled() -> bool {
    use std::sync::OnceLock;
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| std::env::var_os("YSCV_NCHWC_DW3X3_NOPAD_OFF").is_some())
}

/// Arch dispatch for the per-(n, cb_i) inner kernel.
///
/// `pad_slab` layout: `[(h+2), padded_w, block]` row-major.
#[allow(clippy::too_many_arguments)]
#[inline]
fn dw3x3_dispatch(
    pad_slab: &[f32],
    kernel_cb: &[f32],
    kernel_c_stride: usize,
    kernel_row_stride: usize,
    bias_cb: Option<&[f32]>,
    out_chunk: &mut [f32],
    h: usize,
    w: usize,
    padded_w: usize,
    block: usize,
    activation: Activation,
) {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if block == 16 && crate::host_cpu().features.avx512f {
            #[allow(unsafe_code)]
            unsafe {
                dw3x3_inner_avx512_block16(
                    pad_slab,
                    kernel_cb,
                    kernel_c_stride,
                    kernel_row_stride,
                    bias_cb,
                    out_chunk,
                    h,
                    w,
                    padded_w,
                    activation,
                );
            }
            return;
        }
        if block == 8 && crate::host_cpu().features.avx && crate::host_cpu().features.fma {
            // SAFETY: AVX+FMA feature-gated at runtime above. Slice
            // bounds verified by caller layout (padded = (h+2)*(w+2)*8,
            // output = h*w*8, kernel covers all 9 (ky,kx) positions
            // within kernel_cb).
            #[allow(unsafe_code)]
            unsafe {
                dw3x3_inner_avx2_fma_block8(
                    pad_slab,
                    kernel_cb,
                    kernel_c_stride,
                    kernel_row_stride,
                    bias_cb,
                    out_chunk,
                    h,
                    w,
                    padded_w,
                    activation,
                );
            }
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if block == 8 && crate::host_cpu().features.neon {
            // SAFETY: NEON feature-gated above; bounds identical to
            // the scalar path below.
            #[allow(unsafe_code)]
            unsafe {
                dw3x3_inner_neon_block8(
                    pad_slab,
                    kernel_cb,
                    kernel_c_stride,
                    kernel_row_stride,
                    bias_cb,
                    out_chunk,
                    h,
                    w,
                    padded_w,
                    activation,
                );
            }
            return;
        }
    }
    dw3x3_inner_scalar(
        pad_slab,
        kernel_cb,
        kernel_c_stride,
        kernel_row_stride,
        bias_cb,
        out_chunk,
        h,
        w,
        padded_w,
        block,
        activation,
    );
}

/// AVX2+FMA inner kernel, block=8. 9 kernel YMMs preloaded once per
/// (n, cb_i), 9 input loads + 9 FMAs per output pixel. Epilogue does
/// bias + activation inline.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx,fma")]
#[allow(unsafe_code)]
unsafe fn dw3x3_inner_avx2_fma_block8(
    pad_slab: &[f32],
    kernel_cb: &[f32],
    kernel_c_stride: usize,
    kernel_row_stride: usize,
    bias_cb: Option<&[f32]>,
    out_chunk: &mut [f32],
    h: usize,
    w: usize,
    padded_w: usize,
    activation: Activation,
) {
    unsafe {
        #[cfg(target_arch = "x86")]
        use std::arch::x86::{
            _mm256_add_ps, _mm256_fmadd_ps, _mm256_loadu_ps, _mm256_max_ps, _mm256_setzero_ps,
            _mm256_storeu_ps,
        };
        #[cfg(target_arch = "x86_64")]
        use std::arch::x86_64::{
            _mm256_add_ps, _mm256_fmadd_ps, _mm256_loadu_ps, _mm256_max_ps, _mm256_setzero_ps,
            _mm256_storeu_ps,
        };

        const BLOCK: usize = 8;

        // SAFETY: load 9 kernel YMMs from kernel_cb + (ky*row + kx*c). Each
        // YMM covers `block=8` contiguous f32 — fits entirely within the
        // kernel_cb slice (caller-sized).
        let kp = kernel_cb.as_ptr();
        let k00 = _mm256_loadu_ps(kp);
        let k01 = _mm256_loadu_ps(kp.add(kernel_c_stride));
        let k02 = _mm256_loadu_ps(kp.add(2 * kernel_c_stride));
        let k10 = _mm256_loadu_ps(kp.add(kernel_row_stride));
        let k11 = _mm256_loadu_ps(kp.add(kernel_row_stride + kernel_c_stride));
        let k12 = _mm256_loadu_ps(kp.add(kernel_row_stride + 2 * kernel_c_stride));
        let k20 = _mm256_loadu_ps(kp.add(2 * kernel_row_stride));
        let k21 = _mm256_loadu_ps(kp.add(2 * kernel_row_stride + kernel_c_stride));
        let k22 = _mm256_loadu_ps(kp.add(2 * kernel_row_stride + 2 * kernel_c_stride));

        let bias_v = match bias_cb {
            Some(b) => _mm256_loadu_ps(b.as_ptr()),
            None => _mm256_setzero_ps(),
        };

        let pad_ptr = pad_slab.as_ptr();
        let out_ptr = out_chunk.as_mut_ptr();
        let pad_row_block = padded_w * BLOCK;
        let out_row_block = w * BLOCK;

        for y in 0..h {
            let p_row0 = pad_ptr.add(y * pad_row_block);
            let p_row1 = pad_ptr.add((y + 1) * pad_row_block);
            let p_row2 = pad_ptr.add((y + 2) * pad_row_block);
            let o_row = out_ptr.add(y * out_row_block);

            for x in 0..w {
                let xo = x * BLOCK;
                let i00 = _mm256_loadu_ps(p_row0.add(xo));
                let i01 = _mm256_loadu_ps(p_row0.add(xo + BLOCK));
                let i02 = _mm256_loadu_ps(p_row0.add(xo + 2 * BLOCK));
                let i10 = _mm256_loadu_ps(p_row1.add(xo));
                let i11 = _mm256_loadu_ps(p_row1.add(xo + BLOCK));
                let i12 = _mm256_loadu_ps(p_row1.add(xo + 2 * BLOCK));
                let i20 = _mm256_loadu_ps(p_row2.add(xo));
                let i21 = _mm256_loadu_ps(p_row2.add(xo + BLOCK));
                let i22 = _mm256_loadu_ps(p_row2.add(xo + 2 * BLOCK));

                let mut acc = _mm256_fmadd_ps(k00, i00, bias_v);
                acc = _mm256_fmadd_ps(k01, i01, acc);
                acc = _mm256_fmadd_ps(k02, i02, acc);
                acc = _mm256_fmadd_ps(k10, i10, acc);
                acc = _mm256_fmadd_ps(k11, i11, acc);
                acc = _mm256_fmadd_ps(k12, i12, acc);
                acc = _mm256_fmadd_ps(k20, i20, acc);
                acc = _mm256_fmadd_ps(k21, i21, acc);
                acc = _mm256_fmadd_ps(k22, i22, acc);

                match activation {
                    Activation::Relu => {
                        let z = _mm256_setzero_ps();
                        acc = _mm256_max_ps(acc, z);
                    }
                    Activation::Silu => {
                        // Scalar SiLU over 8 lanes (x * sigmoid(x)). The
                        // SIMD sigmoid in `simd::*` is applied per-slice,
                        // not per-YMM, so we drop to stack temp here. DW
                        // SiLU shapes are rare in the tracker — this tail
                        // is not a measured hot path.
                        let mut tmp = [0.0f32; 8];
                        _mm256_storeu_ps(tmp.as_mut_ptr(), acc);
                        for v in tmp.iter_mut() {
                            let s = 1.0 / (1.0 + (-*v).exp());
                            *v *= s;
                        }
                        acc = _mm256_loadu_ps(tmp.as_ptr());
                    }
                    Activation::None => {}
                }
                let _ = _mm256_add_ps; // reserved for future bias path splits
                _mm256_storeu_ps(o_row.add(xo), acc);
            }
        }
    }
}

/// AVX-512 inner kernel for block=16 (16 channels per block).
///
/// Processes 4 spatial positions per inner iteration via 4 independent ZMM
/// accumulators — the 4-way unrolling saturates the 2-FMA/cycle Zen 4
/// pipeline and matches NHWC AVX-512 DW throughput (~4.5 cyc/pixel).
///
/// ZMM budget at peak: 9 weights + 6 inputs (loaded per row) + 4 acc +
/// 1 bias = 20 ZMMs live. Well within the 32-ZMM AVX-512 register file.
///
/// Tail handling: pixels beyond the last W=4 tile are processed one at
/// a time with 9 FMAs (still ZMM-wide, 16 channels × 1 pixel).
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx512f")]
#[allow(unsafe_code)]
unsafe fn dw3x3_inner_avx512_block16(
    pad_slab: &[f32],
    kernel_cb: &[f32],
    kernel_c_stride: usize,
    kernel_row_stride: usize,
    bias_cb: Option<&[f32]>,
    out_chunk: &mut [f32],
    h: usize,
    w: usize,
    padded_w: usize,
    activation: Activation,
) {
    unsafe {
        #[cfg(target_arch = "x86")]
        use std::arch::x86::*;
        #[cfg(target_arch = "x86_64")]
        use std::arch::x86_64::*;

        const B: usize = 16;

        let kp = kernel_cb.as_ptr();
        let k00 = _mm512_loadu_ps(kp);
        let k01 = _mm512_loadu_ps(kp.add(kernel_c_stride));
        let k02 = _mm512_loadu_ps(kp.add(2 * kernel_c_stride));
        let k10 = _mm512_loadu_ps(kp.add(kernel_row_stride));
        let k11 = _mm512_loadu_ps(kp.add(kernel_row_stride + kernel_c_stride));
        let k12 = _mm512_loadu_ps(kp.add(kernel_row_stride + 2 * kernel_c_stride));
        let k20 = _mm512_loadu_ps(kp.add(2 * kernel_row_stride));
        let k21 = _mm512_loadu_ps(kp.add(2 * kernel_row_stride + kernel_c_stride));
        let k22 = _mm512_loadu_ps(kp.add(2 * kernel_row_stride + 2 * kernel_c_stride));

        let bias_v = match bias_cb {
            Some(b) => _mm512_loadu_ps(b.as_ptr()),
            None => _mm512_setzero_ps(),
        };

        let pad_ptr = pad_slab.as_ptr();
        let out_ptr = out_chunk.as_mut_ptr();
        let pad_row_b = padded_w * B;
        let out_row_b = w * B;
        let do_relu = matches!(activation, Activation::Relu);

        for y in 0..h {
            let p0 = pad_ptr.add(y * pad_row_b);
            let p1 = pad_ptr.add((y + 1) * pad_row_b);
            let p2 = pad_ptr.add((y + 2) * pad_row_b);
            let o_row = out_ptr.add(y * out_row_b);

            let mut x = 0usize;
            // 4-pixel tile: processes output positions x, x+1, x+2, x+3 together.
            // Loads 6 input ZMMs per input row (covering kx=0..2 for each output).
            while x + 4 <= w {
                let xb = x * B;
                // Row 0
                let i0_0 = _mm512_loadu_ps(p0.add(xb));
                let i0_1 = _mm512_loadu_ps(p0.add(xb + B));
                let i0_2 = _mm512_loadu_ps(p0.add(xb + 2 * B));
                let i0_3 = _mm512_loadu_ps(p0.add(xb + 3 * B));
                let i0_4 = _mm512_loadu_ps(p0.add(xb + 4 * B));
                let i0_5 = _mm512_loadu_ps(p0.add(xb + 5 * B));
                // Row 1
                let i1_0 = _mm512_loadu_ps(p1.add(xb));
                let i1_1 = _mm512_loadu_ps(p1.add(xb + B));
                let i1_2 = _mm512_loadu_ps(p1.add(xb + 2 * B));
                let i1_3 = _mm512_loadu_ps(p1.add(xb + 3 * B));
                let i1_4 = _mm512_loadu_ps(p1.add(xb + 4 * B));
                let i1_5 = _mm512_loadu_ps(p1.add(xb + 5 * B));
                // Row 2
                let i2_0 = _mm512_loadu_ps(p2.add(xb));
                let i2_1 = _mm512_loadu_ps(p2.add(xb + B));
                let i2_2 = _mm512_loadu_ps(p2.add(xb + 2 * B));
                let i2_3 = _mm512_loadu_ps(p2.add(xb + 3 * B));
                let i2_4 = _mm512_loadu_ps(p2.add(xb + 4 * B));
                let i2_5 = _mm512_loadu_ps(p2.add(xb + 5 * B));

                // Accumulators initialized with bias. Each acc covers one output pixel.
                // acc[j] = bias + sum_ky sum_kx input[ky][x+j+kx] * weight[ky][kx]
                let mut acc0 = _mm512_fmadd_ps(i0_0, k00, bias_v);
                let mut acc1 = _mm512_fmadd_ps(i0_1, k00, bias_v);
                let mut acc2 = _mm512_fmadd_ps(i0_2, k00, bias_v);
                let mut acc3 = _mm512_fmadd_ps(i0_3, k00, bias_v);
                // row 0, kx=1
                acc0 = _mm512_fmadd_ps(i0_1, k01, acc0);
                acc1 = _mm512_fmadd_ps(i0_2, k01, acc1);
                acc2 = _mm512_fmadd_ps(i0_3, k01, acc2);
                acc3 = _mm512_fmadd_ps(i0_4, k01, acc3);
                // row 0, kx=2
                acc0 = _mm512_fmadd_ps(i0_2, k02, acc0);
                acc1 = _mm512_fmadd_ps(i0_3, k02, acc1);
                acc2 = _mm512_fmadd_ps(i0_4, k02, acc2);
                acc3 = _mm512_fmadd_ps(i0_5, k02, acc3);
                // row 1, kx=0
                acc0 = _mm512_fmadd_ps(i1_0, k10, acc0);
                acc1 = _mm512_fmadd_ps(i1_1, k10, acc1);
                acc2 = _mm512_fmadd_ps(i1_2, k10, acc2);
                acc3 = _mm512_fmadd_ps(i1_3, k10, acc3);
                // row 1, kx=1
                acc0 = _mm512_fmadd_ps(i1_1, k11, acc0);
                acc1 = _mm512_fmadd_ps(i1_2, k11, acc1);
                acc2 = _mm512_fmadd_ps(i1_3, k11, acc2);
                acc3 = _mm512_fmadd_ps(i1_4, k11, acc3);
                // row 1, kx=2
                acc0 = _mm512_fmadd_ps(i1_2, k12, acc0);
                acc1 = _mm512_fmadd_ps(i1_3, k12, acc1);
                acc2 = _mm512_fmadd_ps(i1_4, k12, acc2);
                acc3 = _mm512_fmadd_ps(i1_5, k12, acc3);
                // row 2, kx=0
                acc0 = _mm512_fmadd_ps(i2_0, k20, acc0);
                acc1 = _mm512_fmadd_ps(i2_1, k20, acc1);
                acc2 = _mm512_fmadd_ps(i2_2, k20, acc2);
                acc3 = _mm512_fmadd_ps(i2_3, k20, acc3);
                // row 2, kx=1
                acc0 = _mm512_fmadd_ps(i2_1, k21, acc0);
                acc1 = _mm512_fmadd_ps(i2_2, k21, acc1);
                acc2 = _mm512_fmadd_ps(i2_3, k21, acc2);
                acc3 = _mm512_fmadd_ps(i2_4, k21, acc3);
                // row 2, kx=2
                acc0 = _mm512_fmadd_ps(i2_2, k22, acc0);
                acc1 = _mm512_fmadd_ps(i2_3, k22, acc1);
                acc2 = _mm512_fmadd_ps(i2_4, k22, acc2);
                acc3 = _mm512_fmadd_ps(i2_5, k22, acc3);

                if do_relu {
                    let z = _mm512_setzero_ps();
                    acc0 = _mm512_max_ps(acc0, z);
                    acc1 = _mm512_max_ps(acc1, z);
                    acc2 = _mm512_max_ps(acc2, z);
                    acc3 = _mm512_max_ps(acc3, z);
                } else if matches!(activation, Activation::Silu) {
                    // Per-ZMM scalar SiLU — rare in tracker DW chains.
                    macro_rules! silu_zmm {
                        ($v:expr) => {{
                            let mut tmp = [0.0f32; 16];
                            _mm512_storeu_ps(tmp.as_mut_ptr(), $v);
                            for f in tmp.iter_mut() {
                                *f *= 1.0 / (1.0 + (-*f).exp());
                            }
                            _mm512_loadu_ps(tmp.as_ptr())
                        }};
                    }
                    acc0 = silu_zmm!(acc0);
                    acc1 = silu_zmm!(acc1);
                    acc2 = silu_zmm!(acc2);
                    acc3 = silu_zmm!(acc3);
                }

                _mm512_storeu_ps(o_row.add(xb), acc0);
                _mm512_storeu_ps(o_row.add(xb + B), acc1);
                _mm512_storeu_ps(o_row.add(xb + 2 * B), acc2);
                _mm512_storeu_ps(o_row.add(xb + 3 * B), acc3);
                x += 4;
            }

            // 1-pixel tail using 9 ZMM FMAs.
            while x < w {
                let xb = x * B;
                let i00 = _mm512_loadu_ps(p0.add(xb));
                let i01 = _mm512_loadu_ps(p0.add(xb + B));
                let i02 = _mm512_loadu_ps(p0.add(xb + 2 * B));
                let i10 = _mm512_loadu_ps(p1.add(xb));
                let i11 = _mm512_loadu_ps(p1.add(xb + B));
                let i12 = _mm512_loadu_ps(p1.add(xb + 2 * B));
                let i20 = _mm512_loadu_ps(p2.add(xb));
                let i21 = _mm512_loadu_ps(p2.add(xb + B));
                let i22 = _mm512_loadu_ps(p2.add(xb + 2 * B));
                let mut acc = _mm512_fmadd_ps(i00, k00, bias_v);
                acc = _mm512_fmadd_ps(i01, k01, acc);
                acc = _mm512_fmadd_ps(i02, k02, acc);
                acc = _mm512_fmadd_ps(i10, k10, acc);
                acc = _mm512_fmadd_ps(i11, k11, acc);
                acc = _mm512_fmadd_ps(i12, k12, acc);
                acc = _mm512_fmadd_ps(i20, k20, acc);
                acc = _mm512_fmadd_ps(i21, k21, acc);
                acc = _mm512_fmadd_ps(i22, k22, acc);
                if do_relu {
                    acc = _mm512_max_ps(acc, _mm512_setzero_ps());
                } else if matches!(activation, Activation::Silu) {
                    let mut tmp = [0.0f32; 16];
                    _mm512_storeu_ps(tmp.as_mut_ptr(), acc);
                    for f in tmp.iter_mut() {
                        *f *= 1.0 / (1.0 + (-*f).exp());
                    }
                    acc = _mm512_loadu_ps(tmp.as_ptr());
                }
                _mm512_storeu_ps(o_row.add(xb), acc);
                x += 1;
            }
        }
    } // end unsafe
}

/// AVX-512 block=16 NCHWc DW kernel with NO pad buffer. Reads from the
/// raw NCHWc input slab `[h, w, 16]` directly and applies the 3×3 SAME
/// pad inline via row/col validity flags. This eliminates the per-call
/// pad-buffer calloc + input-copy (~415 KB at c=320 16×16) that the
/// legacy padded variant pays.
///
/// Hot interior rows (1 ≤ out_y ≤ h-2) use a 4-pixel tile (4 acc ZMMs +
/// 18 input ZMM loads + 9 weight ZMMs + 1 bias = 32 ZMMs, exactly the
/// AVX-512 register file). Top/bottom border rows fall back to a single-
/// pixel loop with full bounds checks.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx512f")]
#[allow(unsafe_code)]
unsafe fn dw3x3_inner_avx512_block16_nopad(
    in_slab: &[f32],
    kernel_cb: &[f32],
    kernel_c_stride: usize,
    kernel_row_stride: usize,
    bias_cb: Option<&[f32]>,
    out_chunk: &mut [f32],
    h: usize,
    w: usize,
    activation: Activation,
) {
    unsafe {
        #[cfg(target_arch = "x86")]
        use std::arch::x86::*;
        #[cfg(target_arch = "x86_64")]
        use std::arch::x86_64::*;

        const B: usize = 16;
        let kp = kernel_cb.as_ptr();
        let w00 = _mm512_loadu_ps(kp);
        let w01 = _mm512_loadu_ps(kp.add(kernel_c_stride));
        let w02 = _mm512_loadu_ps(kp.add(2 * kernel_c_stride));
        let w10 = _mm512_loadu_ps(kp.add(kernel_row_stride));
        let w11 = _mm512_loadu_ps(kp.add(kernel_row_stride + kernel_c_stride));
        let w12 = _mm512_loadu_ps(kp.add(kernel_row_stride + 2 * kernel_c_stride));
        let w20 = _mm512_loadu_ps(kp.add(2 * kernel_row_stride));
        let w21 = _mm512_loadu_ps(kp.add(2 * kernel_row_stride + kernel_c_stride));
        let w22 = _mm512_loadu_ps(kp.add(2 * kernel_row_stride + 2 * kernel_c_stride));

        let bias_v = match bias_cb {
            Some(b) => _mm512_loadu_ps(b.as_ptr()),
            None => _mm512_setzero_ps(),
        };
        let zero = _mm512_setzero_ps();
        let do_relu = matches!(activation, Activation::Relu);
        let do_silu = matches!(activation, Activation::Silu);

        let in_row = w * B;
        let in_ptr = in_slab.as_ptr();
        let out_ptr = out_chunk.as_mut_ptr();

        // SiLU epilog (rare in DW chains) — scalar fallback.
        let silu_epi = |acc: __m512| -> __m512 {
            let mut tmp = [0.0f32; 16];
            _mm512_storeu_ps(tmp.as_mut_ptr(), acc);
            for f in tmp.iter_mut() {
                *f *= 1.0 / (1.0 + (-*f).exp());
            }
            _mm512_loadu_ps(tmp.as_ptr())
        };

        for out_y in 0..h {
            let top_ok = out_y > 0;
            let bot_ok = out_y + 1 < h;

            let p_mid = in_ptr.add(out_y * in_row);
            let p_top = if top_ok {
                in_ptr.add((out_y - 1) * in_row)
            } else {
                core::ptr::null()
            };
            let p_bot = if bot_ok {
                in_ptr.add((out_y + 1) * in_row)
            } else {
                core::ptr::null()
            };
            let o_row = out_ptr.add(out_y * in_row);

            if top_ok && bot_ok && w >= 2 {
                // Interior row: 1 left-edge pixel + 4-tile interior + tail + 1 right-edge pixel.

                // Left edge: out_x = 0 (no kx=0 column).
                {
                    let pt1 = _mm512_loadu_ps(p_top);
                    let pt2 = _mm512_loadu_ps(p_top.add(B));
                    let pm1 = _mm512_loadu_ps(p_mid);
                    let pm2 = _mm512_loadu_ps(p_mid.add(B));
                    let pb1 = _mm512_loadu_ps(p_bot);
                    let pb2 = _mm512_loadu_ps(p_bot.add(B));
                    let mut acc = _mm512_fmadd_ps(pt1, w01, bias_v);
                    acc = _mm512_fmadd_ps(pt2, w02, acc);
                    acc = _mm512_fmadd_ps(pm1, w11, acc);
                    acc = _mm512_fmadd_ps(pm2, w12, acc);
                    acc = _mm512_fmadd_ps(pb1, w21, acc);
                    acc = _mm512_fmadd_ps(pb2, w22, acc);
                    if do_relu {
                        acc = _mm512_max_ps(acc, zero);
                    } else if do_silu {
                        acc = silu_epi(acc);
                    }
                    _mm512_storeu_ps(o_row, acc);
                }

                // 4-tile interior: out_x in [1, w-5] (each tile covers x..x+3,
                // all fully interior). Each pixel needs 9 FMAs from cols x-1..x+1
                // (for out_x=x), so the tile reads cols x-1..x+4 = 6 ZMMs per row.
                let mut x = 1usize;
                while x + 4 < w {
                    let xb = x * B;
                    let i0_0 = _mm512_loadu_ps(p_top.add(xb - B));
                    let i0_1 = _mm512_loadu_ps(p_top.add(xb));
                    let i0_2 = _mm512_loadu_ps(p_top.add(xb + B));
                    let i0_3 = _mm512_loadu_ps(p_top.add(xb + 2 * B));
                    let i0_4 = _mm512_loadu_ps(p_top.add(xb + 3 * B));
                    let i0_5 = _mm512_loadu_ps(p_top.add(xb + 4 * B));
                    let i1_0 = _mm512_loadu_ps(p_mid.add(xb - B));
                    let i1_1 = _mm512_loadu_ps(p_mid.add(xb));
                    let i1_2 = _mm512_loadu_ps(p_mid.add(xb + B));
                    let i1_3 = _mm512_loadu_ps(p_mid.add(xb + 2 * B));
                    let i1_4 = _mm512_loadu_ps(p_mid.add(xb + 3 * B));
                    let i1_5 = _mm512_loadu_ps(p_mid.add(xb + 4 * B));
                    let i2_0 = _mm512_loadu_ps(p_bot.add(xb - B));
                    let i2_1 = _mm512_loadu_ps(p_bot.add(xb));
                    let i2_2 = _mm512_loadu_ps(p_bot.add(xb + B));
                    let i2_3 = _mm512_loadu_ps(p_bot.add(xb + 2 * B));
                    let i2_4 = _mm512_loadu_ps(p_bot.add(xb + 3 * B));
                    let i2_5 = _mm512_loadu_ps(p_bot.add(xb + 4 * B));

                    let mut acc0 = _mm512_fmadd_ps(i0_0, w00, bias_v);
                    let mut acc1 = _mm512_fmadd_ps(i0_1, w00, bias_v);
                    let mut acc2 = _mm512_fmadd_ps(i0_2, w00, bias_v);
                    let mut acc3 = _mm512_fmadd_ps(i0_3, w00, bias_v);

                    acc0 = _mm512_fmadd_ps(i0_1, w01, acc0);
                    acc1 = _mm512_fmadd_ps(i0_2, w01, acc1);
                    acc2 = _mm512_fmadd_ps(i0_3, w01, acc2);
                    acc3 = _mm512_fmadd_ps(i0_4, w01, acc3);

                    acc0 = _mm512_fmadd_ps(i0_2, w02, acc0);
                    acc1 = _mm512_fmadd_ps(i0_3, w02, acc1);
                    acc2 = _mm512_fmadd_ps(i0_4, w02, acc2);
                    acc3 = _mm512_fmadd_ps(i0_5, w02, acc3);

                    acc0 = _mm512_fmadd_ps(i1_0, w10, acc0);
                    acc1 = _mm512_fmadd_ps(i1_1, w10, acc1);
                    acc2 = _mm512_fmadd_ps(i1_2, w10, acc2);
                    acc3 = _mm512_fmadd_ps(i1_3, w10, acc3);

                    acc0 = _mm512_fmadd_ps(i1_1, w11, acc0);
                    acc1 = _mm512_fmadd_ps(i1_2, w11, acc1);
                    acc2 = _mm512_fmadd_ps(i1_3, w11, acc2);
                    acc3 = _mm512_fmadd_ps(i1_4, w11, acc3);

                    acc0 = _mm512_fmadd_ps(i1_2, w12, acc0);
                    acc1 = _mm512_fmadd_ps(i1_3, w12, acc1);
                    acc2 = _mm512_fmadd_ps(i1_4, w12, acc2);
                    acc3 = _mm512_fmadd_ps(i1_5, w12, acc3);

                    acc0 = _mm512_fmadd_ps(i2_0, w20, acc0);
                    acc1 = _mm512_fmadd_ps(i2_1, w20, acc1);
                    acc2 = _mm512_fmadd_ps(i2_2, w20, acc2);
                    acc3 = _mm512_fmadd_ps(i2_3, w20, acc3);

                    acc0 = _mm512_fmadd_ps(i2_1, w21, acc0);
                    acc1 = _mm512_fmadd_ps(i2_2, w21, acc1);
                    acc2 = _mm512_fmadd_ps(i2_3, w21, acc2);
                    acc3 = _mm512_fmadd_ps(i2_4, w21, acc3);

                    acc0 = _mm512_fmadd_ps(i2_2, w22, acc0);
                    acc1 = _mm512_fmadd_ps(i2_3, w22, acc1);
                    acc2 = _mm512_fmadd_ps(i2_4, w22, acc2);
                    acc3 = _mm512_fmadd_ps(i2_5, w22, acc3);

                    if do_relu {
                        acc0 = _mm512_max_ps(acc0, zero);
                        acc1 = _mm512_max_ps(acc1, zero);
                        acc2 = _mm512_max_ps(acc2, zero);
                        acc3 = _mm512_max_ps(acc3, zero);
                    } else if do_silu {
                        acc0 = silu_epi(acc0);
                        acc1 = silu_epi(acc1);
                        acc2 = silu_epi(acc2);
                        acc3 = silu_epi(acc3);
                    }
                    _mm512_storeu_ps(o_row.add(xb), acc0);
                    _mm512_storeu_ps(o_row.add(xb + B), acc1);
                    _mm512_storeu_ps(o_row.add(xb + 2 * B), acc2);
                    _mm512_storeu_ps(o_row.add(xb + 3 * B), acc3);
                    x += 4;
                }

                // 1-pixel interior tail: out_x in [x, w-2].
                while x < w - 1 {
                    let xb = x * B;
                    let i0_l = _mm512_loadu_ps(p_top.add(xb - B));
                    let i0_m = _mm512_loadu_ps(p_top.add(xb));
                    let i0_r = _mm512_loadu_ps(p_top.add(xb + B));
                    let i1_l = _mm512_loadu_ps(p_mid.add(xb - B));
                    let i1_m = _mm512_loadu_ps(p_mid.add(xb));
                    let i1_r = _mm512_loadu_ps(p_mid.add(xb + B));
                    let i2_l = _mm512_loadu_ps(p_bot.add(xb - B));
                    let i2_m = _mm512_loadu_ps(p_bot.add(xb));
                    let i2_r = _mm512_loadu_ps(p_bot.add(xb + B));
                    let mut acc = _mm512_fmadd_ps(i0_l, w00, bias_v);
                    acc = _mm512_fmadd_ps(i0_m, w01, acc);
                    acc = _mm512_fmadd_ps(i0_r, w02, acc);
                    acc = _mm512_fmadd_ps(i1_l, w10, acc);
                    acc = _mm512_fmadd_ps(i1_m, w11, acc);
                    acc = _mm512_fmadd_ps(i1_r, w12, acc);
                    acc = _mm512_fmadd_ps(i2_l, w20, acc);
                    acc = _mm512_fmadd_ps(i2_m, w21, acc);
                    acc = _mm512_fmadd_ps(i2_r, w22, acc);
                    if do_relu {
                        acc = _mm512_max_ps(acc, zero);
                    } else if do_silu {
                        acc = silu_epi(acc);
                    }
                    _mm512_storeu_ps(o_row.add(xb), acc);
                    x += 1;
                }

                // Right edge: out_x = w-1 (no kx=2 column).
                {
                    let xb = (w - 1) * B;
                    let i0_l = _mm512_loadu_ps(p_top.add(xb - B));
                    let i0_m = _mm512_loadu_ps(p_top.add(xb));
                    let i1_l = _mm512_loadu_ps(p_mid.add(xb - B));
                    let i1_m = _mm512_loadu_ps(p_mid.add(xb));
                    let i2_l = _mm512_loadu_ps(p_bot.add(xb - B));
                    let i2_m = _mm512_loadu_ps(p_bot.add(xb));
                    let mut acc = _mm512_fmadd_ps(i0_l, w00, bias_v);
                    acc = _mm512_fmadd_ps(i0_m, w01, acc);
                    acc = _mm512_fmadd_ps(i1_l, w10, acc);
                    acc = _mm512_fmadd_ps(i1_m, w11, acc);
                    acc = _mm512_fmadd_ps(i2_l, w20, acc);
                    acc = _mm512_fmadd_ps(i2_m, w21, acc);
                    if do_relu {
                        acc = _mm512_max_ps(acc, zero);
                    } else if do_silu {
                        acc = silu_epi(acc);
                    }
                    _mm512_storeu_ps(o_row.add(xb), acc);
                }
            } else {
                // Border row (out_y=0 or out_y=h-1) — also covers degenerate
                // w=1 case. Per-pixel with full bounds checks.
                for out_x in 0..w {
                    let xb = out_x * B;
                    let left_ok = out_x > 0;
                    let right_ok = out_x + 1 < w;
                    let mut acc = bias_v;
                    if top_ok {
                        if left_ok {
                            let p = p_top.add(xb - B);
                            acc = _mm512_fmadd_ps(_mm512_loadu_ps(p), w00, acc);
                        }
                        let p = p_top.add(xb);
                        acc = _mm512_fmadd_ps(_mm512_loadu_ps(p), w01, acc);
                        if right_ok {
                            let p = p_top.add(xb + B);
                            acc = _mm512_fmadd_ps(_mm512_loadu_ps(p), w02, acc);
                        }
                    }
                    if left_ok {
                        let p = p_mid.add(xb - B);
                        acc = _mm512_fmadd_ps(_mm512_loadu_ps(p), w10, acc);
                    }
                    let pmid = p_mid.add(xb);
                    acc = _mm512_fmadd_ps(_mm512_loadu_ps(pmid), w11, acc);
                    if right_ok {
                        let p = p_mid.add(xb + B);
                        acc = _mm512_fmadd_ps(_mm512_loadu_ps(p), w12, acc);
                    }
                    if bot_ok {
                        if left_ok {
                            let p = p_bot.add(xb - B);
                            acc = _mm512_fmadd_ps(_mm512_loadu_ps(p), w20, acc);
                        }
                        let p = p_bot.add(xb);
                        acc = _mm512_fmadd_ps(_mm512_loadu_ps(p), w21, acc);
                        if right_ok {
                            let p = p_bot.add(xb + B);
                            acc = _mm512_fmadd_ps(_mm512_loadu_ps(p), w22, acc);
                        }
                    }
                    if do_relu {
                        acc = _mm512_max_ps(acc, zero);
                    } else if do_silu {
                        acc = silu_epi(acc);
                    }
                    _mm512_storeu_ps(o_row.add(xb), acc);
                }
            }
        }
    }
}

/// NEON inner kernel, block=8 via two float32x4 passes per pixel. Not
/// perf-tuned (tracker runs on x86_64); kept for cross-compile
/// correctness on aarch64.
#[cfg(target_arch = "aarch64")]
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "neon")]
#[allow(unsafe_code)]
unsafe fn dw3x3_inner_neon_block8(
    pad_slab: &[f32],
    kernel_cb: &[f32],
    kernel_c_stride: usize,
    kernel_row_stride: usize,
    bias_cb: Option<&[f32]>,
    out_chunk: &mut [f32],
    h: usize,
    w: usize,
    padded_w: usize,
    activation: Activation,
) {
    unsafe {
        use std::arch::aarch64::{
            float32x4_t, vaddq_f32, vdupq_n_f32, vfmaq_f32, vld1q_f32, vmaxq_f32, vst1q_f32,
        };

        const BLOCK: usize = 8;
        let kp = kernel_cb.as_ptr();

        // Load 9 kernel pairs (lower = lanes 0..4, upper = lanes 4..8).
        let load_pair = |off: usize| -> (float32x4_t, float32x4_t) {
            // SAFETY: kernel_cb sized by caller to include up to
            // 2*row + 2*c + BLOCK - 1.
            (vld1q_f32(kp.add(off)), vld1q_f32(kp.add(off + 4)))
        };
        let (k00a, k00b) = load_pair(0);
        let (k01a, k01b) = load_pair(kernel_c_stride);
        let (k02a, k02b) = load_pair(2 * kernel_c_stride);
        let (k10a, k10b) = load_pair(kernel_row_stride);
        let (k11a, k11b) = load_pair(kernel_row_stride + kernel_c_stride);
        let (k12a, k12b) = load_pair(kernel_row_stride + 2 * kernel_c_stride);
        let (k20a, k20b) = load_pair(2 * kernel_row_stride);
        let (k21a, k21b) = load_pair(2 * kernel_row_stride + kernel_c_stride);
        let (k22a, k22b) = load_pair(2 * kernel_row_stride + 2 * kernel_c_stride);

        let (bias_a, bias_b) = match bias_cb {
            Some(b) => (vld1q_f32(b.as_ptr()), vld1q_f32(b.as_ptr().add(4))),
            None => (vdupq_n_f32(0.0), vdupq_n_f32(0.0)),
        };

        let pad_ptr = pad_slab.as_ptr();
        let out_ptr = out_chunk.as_mut_ptr();
        let pad_row_block = padded_w * BLOCK;
        let out_row_block = w * BLOCK;

        for y in 0..h {
            let p_row0 = pad_ptr.add(y * pad_row_block);
            let p_row1 = pad_ptr.add((y + 1) * pad_row_block);
            let p_row2 = pad_ptr.add((y + 2) * pad_row_block);
            let o_row = out_ptr.add(y * out_row_block);

            for x in 0..w {
                let xo = x * BLOCK;
                // Lower half (lanes 0..4).
                let i00a = vld1q_f32(p_row0.add(xo));
                let i01a = vld1q_f32(p_row0.add(xo + BLOCK));
                let i02a = vld1q_f32(p_row0.add(xo + 2 * BLOCK));
                let i10a = vld1q_f32(p_row1.add(xo));
                let i11a = vld1q_f32(p_row1.add(xo + BLOCK));
                let i12a = vld1q_f32(p_row1.add(xo + 2 * BLOCK));
                let i20a = vld1q_f32(p_row2.add(xo));
                let i21a = vld1q_f32(p_row2.add(xo + BLOCK));
                let i22a = vld1q_f32(p_row2.add(xo + 2 * BLOCK));
                let mut aa = vfmaq_f32(bias_a, i00a, k00a);
                aa = vfmaq_f32(aa, i01a, k01a);
                aa = vfmaq_f32(aa, i02a, k02a);
                aa = vfmaq_f32(aa, i10a, k10a);
                aa = vfmaq_f32(aa, i11a, k11a);
                aa = vfmaq_f32(aa, i12a, k12a);
                aa = vfmaq_f32(aa, i20a, k20a);
                aa = vfmaq_f32(aa, i21a, k21a);
                aa = vfmaq_f32(aa, i22a, k22a);

                // Upper half (lanes 4..8).
                let i00b = vld1q_f32(p_row0.add(xo + 4));
                let i01b = vld1q_f32(p_row0.add(xo + BLOCK + 4));
                let i02b = vld1q_f32(p_row0.add(xo + 2 * BLOCK + 4));
                let i10b = vld1q_f32(p_row1.add(xo + 4));
                let i11b = vld1q_f32(p_row1.add(xo + BLOCK + 4));
                let i12b = vld1q_f32(p_row1.add(xo + 2 * BLOCK + 4));
                let i20b = vld1q_f32(p_row2.add(xo + 4));
                let i21b = vld1q_f32(p_row2.add(xo + BLOCK + 4));
                let i22b = vld1q_f32(p_row2.add(xo + 2 * BLOCK + 4));
                let mut ab = vfmaq_f32(bias_b, i00b, k00b);
                ab = vfmaq_f32(ab, i01b, k01b);
                ab = vfmaq_f32(ab, i02b, k02b);
                ab = vfmaq_f32(ab, i10b, k10b);
                ab = vfmaq_f32(ab, i11b, k11b);
                ab = vfmaq_f32(ab, i12b, k12b);
                ab = vfmaq_f32(ab, i20b, k20b);
                ab = vfmaq_f32(ab, i21b, k21b);
                ab = vfmaq_f32(ab, i22b, k22b);

                match activation {
                    Activation::Relu => {
                        let z = vdupq_n_f32(0.0);
                        aa = vmaxq_f32(aa, z);
                        ab = vmaxq_f32(ab, z);
                    }
                    Activation::Silu => {
                        let mut tmp = [0.0f32; 8];
                        vst1q_f32(tmp.as_mut_ptr(), aa);
                        vst1q_f32(tmp.as_mut_ptr().add(4), ab);
                        for v in tmp.iter_mut() {
                            let s = 1.0 / (1.0 + (-*v).exp());
                            *v *= s;
                        }
                        aa = vld1q_f32(tmp.as_ptr());
                        ab = vld1q_f32(tmp.as_ptr().add(4));
                    }
                    Activation::None => {}
                }
                let _ = vaddq_f32; // reserved
                vst1q_f32(o_row.add(xo), aa);
                vst1q_f32(o_row.add(xo + 4), ab);
            }
        }
    }
}

/// Scalar fallback, supports any block size. Used on non-SIMD arches,
/// miri, or when the runtime feature gate fails.
#[allow(clippy::too_many_arguments)]
fn dw3x3_inner_scalar(
    pad_slab: &[f32],
    kernel_cb: &[f32],
    kernel_c_stride: usize,
    kernel_row_stride: usize,
    bias_cb: Option<&[f32]>,
    out_chunk: &mut [f32],
    h: usize,
    w: usize,
    padded_w: usize,
    block: usize,
    activation: Activation,
) {
    let pad_row_block = padded_w * block;
    let out_row_block = w * block;
    let bias_slice = bias_cb;

    for y in 0..h {
        let p0 = y * pad_row_block;
        let p1 = (y + 1) * pad_row_block;
        let p2 = (y + 2) * pad_row_block;
        let o_row = y * out_row_block;
        for x in 0..w {
            let xo = x * block;
            for lane in 0..block {
                let mut acc = match bias_slice {
                    Some(b) => b[lane],
                    None => 0.0,
                };
                for ky in 0..3 {
                    let p_row = match ky {
                        0 => p0,
                        1 => p1,
                        _ => p2,
                    };
                    for kx in 0..3 {
                        let in_val = pad_slab[p_row + (x + kx) * block + lane];
                        let k_off = ky * kernel_row_stride + kx * kernel_c_stride + lane;
                        acc += in_val * kernel_cb[k_off];
                    }
                }
                acc = match activation {
                    Activation::Relu => acc.max(0.0),
                    Activation::Silu => acc * (1.0 / (1.0 + (-acc).exp())),
                    Activation::None => acc,
                };
                out_chunk[o_row + xo + lane] = acc;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::config::{DepthwiseConv2dSpec, ParallelElementwiseConfig};
    use super::super::conv::depthwise_conv2d_nhwc_padded_with_activation_with_config_and_pool;
    use super::super::layout::{nchwc_to_nhwc, nhwc_to_nchwc};
    use super::*;

    fn fill_ramp(buf: &mut [f32], scale: f32, base: f32) {
        for (i, v) in buf.iter_mut().enumerate() {
            *v = ((i % 101) as f32) * scale + base;
        }
    }

    fn run_case(h: usize, w: usize, c: usize, block: usize, act: Activation, with_bias: bool) {
        let n = 1usize;
        let mut input_nhwc = vec![0.0f32; n * h * w * c];
        fill_ramp(&mut input_nhwc, 0.011, -0.4);
        let input_t = Tensor::from_vec(vec![n, h, w, c], input_nhwc.clone()).unwrap();
        let input_nchwc = nhwc_to_nchwc(&input_t, block).unwrap();

        let mut ker = vec![0.0f32; 3 * 3 * c];
        fill_ramp(&mut ker, 0.017, 0.2);
        let kernel = Tensor::from_vec(vec![3, 3, c, 1], ker).unwrap();
        let bias_t = if with_bias {
            let mut b = vec![0.0f32; c];
            fill_ramp(&mut b, 0.3, -1.2);
            Some(Tensor::from_vec(vec![c], b).unwrap())
        } else {
            None
        };

        let native_out = conv2d_nchwc_dw3x3_s1_same_pad(
            &input_nchwc,
            &kernel,
            bias_t.as_ref(),
            act,
            c,
            ParallelElementwiseConfig::default(),
            None,
            None,
        )
        .unwrap();
        let native_nhwc = nchwc_to_nhwc(&native_out, c).unwrap();

        let ref_out = depthwise_conv2d_nhwc_padded_with_activation_with_config_and_pool(
            &input_t,
            &kernel,
            bias_t.as_ref(),
            DepthwiseConv2dSpec {
                stride_h: 1,
                stride_w: 1,
            },
            1,
            1,
            1,
            1,
            act,
            ParallelElementwiseConfig::default(),
            None,
        )
        .unwrap();

        let a = native_nhwc.data();
        let r = ref_out.data();
        assert_eq!(a.len(), r.len(), "len mismatch h={h} w={w} c={c}");
        let mut max_diff = 0.0f32;
        let mut max_i = 0usize;
        for i in 0..a.len() {
            let d = (a[i] - r[i]).abs();
            if d > max_diff {
                max_diff = d;
                max_i = i;
            }
        }
        assert!(
            max_diff < 5e-3,
            "native dw3x3 nchwc diverges h={h} w={w} c={c} act={act:?} bias={with_bias}: \
             max diff {max_diff} at {max_i}: native={} ref={}",
            a[max_i],
            r[max_i],
        );
    }

    #[test]
    fn dw3x3_nchwc_matches_nhwc_16x16_c96_none_nobias() {
        run_case(16, 16, 96, 8, Activation::None, false);
    }

    #[test]
    fn dw3x3_nchwc_matches_nhwc_16x16_c96_relu_bias() {
        run_case(16, 16, 96, 8, Activation::Relu, true);
    }

    #[test]
    fn dw3x3_nchwc_matches_nhwc_32x32_c144_relu_bias() {
        run_case(32, 32, 144, 8, Activation::Relu, true);
    }

    #[test]
    fn dw3x3_nchwc_matches_nhwc_32x32_c192_silu_bias() {
        run_case(32, 32, 192, 8, Activation::Silu, true);
    }

    #[test]
    fn dw3x3_nchwc_matches_nhwc_64x64_c96_relu_nobias() {
        run_case(64, 64, 96, 8, Activation::Relu, false);
    }

    #[test]
    fn dw3x3_nchwc_matches_nhwc_128x128_c16_none_bias() {
        run_case(128, 128, 16, 8, Activation::None, true);
    }

    #[test]
    fn dw3x3_nchwc_matches_nhwc_8x8_c8_relu_bias() {
        run_case(8, 8, 8, 8, Activation::Relu, true);
    }

    #[test]
    fn dw3x3_nchwc_rejects_channel_tail() {
        // 24 % 8 == 0 → OK; 20 % 8 != 0 → must error.
        let input_t = Tensor::from_vec(vec![1, 8, 8, 20], vec![0.0; 8 * 8 * 20]).unwrap();
        let input_nchwc = nhwc_to_nchwc(&input_t, 8).unwrap();
        let kernel = Tensor::from_vec(vec![3, 3, 20, 1], vec![0.0; 3 * 3 * 20]).unwrap();
        let r = conv2d_nchwc_dw3x3_s1_same_pad(
            &input_nchwc,
            &kernel,
            None,
            Activation::None,
            20,
            ParallelElementwiseConfig::default(),
            None,
            None,
        );
        assert!(r.is_err(), "expected error on tail block");
    }

    #[test]
    fn dw3x3_nchwc_scalar_path_matches_avx() {
        // On AVX-512 this now hits the fast AVX-512 block=16 path; on
        // other archs it falls to scalar. Both must match NHWC reference.
        run_case(16, 16, 16, 16, Activation::Relu, true);
    }

    #[test]
    fn dw3x3_nchwc_block16_avx512_relu_bias() {
        run_case(32, 32, 96, 16, Activation::Relu, true);
    }

    #[test]
    fn dw3x3_nchwc_block16_avx512_none_nobias() {
        run_case(16, 16, 96, 16, Activation::None, false);
    }

    #[test]
    fn dw3x3_nchwc_block16_avx512_c320_relu() {
        run_case(16, 16, 320, 16, Activation::Relu, true);
    }

    #[test]
    fn dw3x3_nchwc_block16_avx512_tail_w() {
        // w=13 is not a multiple of 4 — exercises the 1-pixel tail path.
        run_case(13, 13, 32, 16, Activation::Relu, false);
    }

    /// Micro-bench native NCHWc DW (including NHWC↔NCHWc conversion
    /// round-trip, which is what the runtime actually pays per op in
    /// the all-NHWC graph) vs the existing NHWC DW path. Gated on
    /// `YSCV_DW_NCHWC_BENCH=1` to stay out of the default test run.
    ///
    /// Reports per-shape µs for 3 configurations:
    /// - `nhwc`: baseline, matches the current exec_conv_with_params
    ///   DW path.
    /// - `nchwc_native` (raw): just the native kernel, no conversion
    ///   (best-case if we could keep NCHWc layout across ops).
    /// - `nchwc_roundtrip`: full nhwc→nchwc + native + nchwc→nhwc, the
    ///   per-op dispatch cost when caller is NHWC.
    #[test]
    #[ignore = "perf bench; run with YSCV_DW_NCHWC_BENCH=1 --nocapture"]
    fn dw3x3_nchwc_perf_micro_bench() {
        if std::env::var("YSCV_DW_NCHWC_BENCH").is_err() {
            return;
        }
        let shapes: &[(usize, usize, usize, &str)] = &[
            (16, 16, 96, "16x16 c=96"),
            (16, 16, 144, "16x16 c=144"),
            (32, 32, 96, "32x32 c=96"),
            (32, 32, 144, "32x32 c=144"),
            (32, 32, 192, "32x32 c=192"),
            (64, 64, 96, "64x64 c=96"),
        ];
        const ITERS: usize = 500;
        const BLOCK: usize = 8;
        println!();
        println!(
            "{:20} {:>12} {:>14} {:>16}",
            "shape", "nhwc µs", "nchwc_raw µs", "nchwc_round µs"
        );
        for &(h, w, c, label) in shapes {
            let n = 1usize;
            let mut input_nhwc = vec![0.0f32; n * h * w * c];
            fill_ramp(&mut input_nhwc, 0.011, -0.4);
            let input_t = Tensor::from_vec(vec![n, h, w, c], input_nhwc.clone()).unwrap();
            let input_nchwc = nhwc_to_nchwc(&input_t, BLOCK).unwrap();
            let mut ker = vec![0.0f32; 3 * 3 * c];
            fill_ramp(&mut ker, 0.017, 0.2);
            let kernel = Tensor::from_vec(vec![3, 3, c, 1], ker).unwrap();
            let mut b = vec![0.0f32; c];
            fill_ramp(&mut b, 0.3, -1.2);
            let bias = Tensor::from_vec(vec![c], b).unwrap();

            // Warm up all paths.
            for _ in 0..50 {
                let _ = depthwise_conv2d_nhwc_padded_with_activation_with_config_and_pool(
                    &input_t,
                    &kernel,
                    Some(&bias),
                    DepthwiseConv2dSpec {
                        stride_h: 1,
                        stride_w: 1,
                    },
                    1,
                    1,
                    1,
                    1,
                    Activation::Relu,
                    ParallelElementwiseConfig::default(),
                    None,
                )
                .unwrap();
                let _ = conv2d_nchwc_dw3x3_s1_same_pad(
                    &input_nchwc,
                    &kernel,
                    Some(&bias),
                    Activation::Relu,
                    c,
                    ParallelElementwiseConfig::default(),
                    None,
                    None,
                )
                .unwrap();
            }

            let t0 = std::time::Instant::now();
            for _ in 0..ITERS {
                let r = depthwise_conv2d_nhwc_padded_with_activation_with_config_and_pool(
                    &input_t,
                    &kernel,
                    Some(&bias),
                    DepthwiseConv2dSpec {
                        stride_h: 1,
                        stride_w: 1,
                    },
                    1,
                    1,
                    1,
                    1,
                    Activation::Relu,
                    ParallelElementwiseConfig::default(),
                    None,
                )
                .unwrap();
                std::hint::black_box(r);
            }
            let nhwc_us = t0.elapsed().as_secs_f64() * 1e6 / ITERS as f64;

            let t0 = std::time::Instant::now();
            for _ in 0..ITERS {
                let r = conv2d_nchwc_dw3x3_s1_same_pad(
                    &input_nchwc,
                    &kernel,
                    Some(&bias),
                    Activation::Relu,
                    c,
                    ParallelElementwiseConfig::default(),
                    None,
                    None,
                )
                .unwrap();
                std::hint::black_box(r);
            }
            let raw_us = t0.elapsed().as_secs_f64() * 1e6 / ITERS as f64;

            let t0 = std::time::Instant::now();
            for _ in 0..ITERS {
                let blk = nhwc_to_nchwc(&input_t, BLOCK).unwrap();
                let mid = conv2d_nchwc_dw3x3_s1_same_pad(
                    &blk,
                    &kernel,
                    Some(&bias),
                    Activation::Relu,
                    c,
                    ParallelElementwiseConfig::default(),
                    None,
                    None,
                )
                .unwrap();
                let out = nchwc_to_nhwc(&mid, c).unwrap();
                std::hint::black_box(out);
            }
            let round_us = t0.elapsed().as_secs_f64() * 1e6 / ITERS as f64;

            println!(
                "{:20} {:>12.2} {:>14.2} {:>16.2}",
                label, nhwc_us, raw_us, round_us
            );
        }
        println!();
    }
}
