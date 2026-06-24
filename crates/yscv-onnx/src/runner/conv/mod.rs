use super::*;
use rayon::prelude::*;
use std::sync::OnceLock;

type DotFn = fn(&[f32], &[f32]) -> f32;

#[inline]
fn dot_scalar(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..a.len() {
        sum += a[i] * b[i];
    }
    sum
}

#[inline]
fn dot_dispatch() -> DotFn {
    static DOT_FN: OnceLock<DotFn> = OnceLock::new();
    *DOT_FN.get_or_init(|| {
        #[cfg(target_arch = "aarch64")]
        {
            if yscv_cpu::host_cpu().features.neon {
                return dot_neon;
            }
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if yscv_cpu::host_cpu().features.avx && yscv_cpu::host_cpu().features.fma {
                return dot_avx;
            }
            if yscv_cpu::host_cpu().features.sse {
                return dot_sse;
            }
        }
        dot_scalar
    })
}

#[inline]
fn apply_conv_activation(
    tensor: &mut Tensor,
    activation: yscv_kernels::Activation,
    activation_fused: bool,
) {
    if activation_fused {
        return;
    }
    match activation {
        yscv_kernels::Activation::Silu => yscv_kernels::silu_inplace(tensor),
        yscv_kernels::Activation::Relu => yscv_kernels::relu_inplace(tensor),
        yscv_kernels::Activation::None => {}
    }
}

#[inline]
fn nchw_i8_to_nhwc(src: &[i8], dst: &mut [i8], n_n: usize, c_in: usize, h: usize, w: usize) {
    let spatial = h * w;
    if !cfg!(miri) && dst.len() >= 16_384 && rayon::current_num_threads() > 1 {
        dst.par_chunks_mut(c_in)
            .enumerate()
            .for_each(|(pixel, dst_pixel)| {
                let n = pixel / spatial;
                let rem = pixel % spatial;
                let y = rem / w;
                let x = rem % w;
                for (c, d) in dst_pixel.iter_mut().enumerate() {
                    *d = src[((n * c_in + c) * h + y) * w + x];
                }
            });
    } else {
        for n in 0..n_n {
            for c in 0..c_in {
                for y in 0..h {
                    for x in 0..w {
                        let src_idx = ((n * c_in + c) * h + y) * w + x;
                        let dst_idx = ((n * h + y) * w + x) * c_in + c;
                        dst[dst_idx] = src[src_idx];
                    }
                }
            }
        }
    }
}

#[inline]
fn nhwc_i8_to_nchw(src: &[i8], dst: &mut [i8], n_n: usize, c_out: usize, h: usize, w: usize) {
    let plane = h * w;
    if !cfg!(miri) && dst.len() >= 16_384 && rayon::current_num_threads() > 1 {
        dst.par_chunks_mut(plane)
            .enumerate()
            .for_each(|(plane_idx, dst_plane)| {
                let n = plane_idx / c_out;
                let c = plane_idx % c_out;
                for y in 0..h {
                    for x in 0..w {
                        let src_idx = ((n * h + y) * w + x) * c_out + c;
                        dst_plane[y * w + x] = src[src_idx];
                    }
                }
            });
    } else {
        for n in 0..n_n {
            for c in 0..c_out {
                for y in 0..h {
                    for x in 0..w {
                        let src_idx = ((n * h + y) * w + x) * c_out + c;
                        let dst_idx = ((n * c_out + c) * h + y) * w + x;
                        dst[dst_idx] = src[src_idx];
                    }
                }
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
unsafe fn dot_neon_impl(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64::*;
    let n = a.len();
    let ap = a.as_ptr();
    let bp = b.as_ptr();
    let mut acc0 = vdupq_n_f32(0.0);
    let mut acc1 = vdupq_n_f32(0.0);
    let mut i = 0usize;
    while i + 8 <= n {
        let va0 = vld1q_f32(ap.add(i));
        let vb0 = vld1q_f32(bp.add(i));
        acc0 = vfmaq_f32(acc0, va0, vb0);
        let va1 = vld1q_f32(ap.add(i + 4));
        let vb1 = vld1q_f32(bp.add(i + 4));
        acc1 = vfmaq_f32(acc1, va1, vb1);
        i += 8;
    }
    while i + 4 <= n {
        let va = vld1q_f32(ap.add(i));
        let vb = vld1q_f32(bp.add(i));
        acc0 = vfmaq_f32(acc0, va, vb);
        i += 4;
    }
    acc0 = vaddq_f32(acc0, acc1);
    let mut sum = vaddvq_f32(acc0);
    while i < n {
        sum += *ap.add(i) * *bp.add(i);
        i += 1;
    }
    sum
}

#[cfg(target_arch = "aarch64")]
#[inline]
fn dot_neon(a: &[f32], b: &[f32]) -> f32 {
    #[allow(unsafe_code)]
    unsafe {
        dot_neon_impl(a, b)
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx", enable = "fma")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
unsafe fn dot_avx_impl(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let n = a.len();
    let ap = a.as_ptr();
    let bp = b.as_ptr();
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut i = 0usize;
    while i + 16 <= n {
        let va0 = _mm256_loadu_ps(ap.add(i));
        let vb0 = _mm256_loadu_ps(bp.add(i));
        acc0 = _mm256_fmadd_ps(va0, vb0, acc0);
        let va1 = _mm256_loadu_ps(ap.add(i + 8));
        let vb1 = _mm256_loadu_ps(bp.add(i + 8));
        acc1 = _mm256_fmadd_ps(va1, vb1, acc1);
        i += 16;
    }
    while i + 8 <= n {
        let va = _mm256_loadu_ps(ap.add(i));
        let vb = _mm256_loadu_ps(bp.add(i));
        acc0 = _mm256_fmadd_ps(va, vb, acc0);
        i += 8;
    }
    acc0 = _mm256_add_ps(acc0, acc1);
    let hi = _mm256_extractf128_ps(acc0, 1);
    let lo = _mm256_castps256_ps128(acc0);
    let sum128 = _mm_add_ps(lo, hi);
    let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
    let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 1));
    let mut sum = _mm_cvtss_f32(sum32);
    while i < n {
        sum += *ap.add(i) * *bp.add(i);
        i += 1;
    }
    sum
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline]
fn dot_avx(a: &[f32], b: &[f32]) -> f32 {
    #[allow(unsafe_code)]
    unsafe {
        dot_avx_impl(a, b)
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
unsafe fn dot_sse_impl(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let n = a.len();
    let ap = a.as_ptr();
    let bp = b.as_ptr();
    let mut acc0 = _mm_setzero_ps();
    let mut acc1 = _mm_setzero_ps();
    let mut i = 0usize;
    while i + 8 <= n {
        let va0 = _mm_loadu_ps(ap.add(i));
        let vb0 = _mm_loadu_ps(bp.add(i));
        acc0 = _mm_add_ps(acc0, _mm_mul_ps(va0, vb0));
        let va1 = _mm_loadu_ps(ap.add(i + 4));
        let vb1 = _mm_loadu_ps(bp.add(i + 4));
        acc1 = _mm_add_ps(acc1, _mm_mul_ps(va1, vb1));
        i += 8;
    }
    while i + 4 <= n {
        let va = _mm_loadu_ps(ap.add(i));
        let vb = _mm_loadu_ps(bp.add(i));
        acc0 = _mm_add_ps(acc0, _mm_mul_ps(va, vb));
        i += 4;
    }
    acc0 = _mm_add_ps(acc0, acc1);
    let shuf = _mm_movehl_ps(acc0, acc0);
    let sum64 = _mm_add_ps(acc0, shuf);
    let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 1));
    let mut sum = _mm_cvtss_f32(sum32);
    while i < n {
        sum += *ap.add(i) * *bp.add(i);
        i += 1;
    }
    sum
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline]
fn dot_sse(a: &[f32], b: &[f32]) -> f32 {
    #[allow(unsafe_code)]
    unsafe {
        dot_sse_impl(a, b)
    }
}

pub(super) fn repack_depthwise_kernel_once(
    weight: &Tensor,
    o_ch: usize,
    i_per_g: usize,
    kh: usize,
    kw: usize,
    channels: usize,
    depth_mult: usize,
) -> Result<Tensor, OnnxError> {
    let w_data = weight.data();
    let mut dw_data = vec![0.0f32; kh * kw * channels * depth_mult];
    for oc in 0..o_ch {
        let g = oc / depth_mult;
        let dm = oc % depth_mult;
        for ki in 0..kh {
            for kj in 0..kw {
                let src = ((oc * i_per_g) * kh + ki) * kw + kj;
                let dst = ((ki * kw + kj) * channels + g) * depth_mult + dm;
                dw_data[dst] = w_data[src];
            }
        }
    }
    Tensor::from_vec(vec![kh, kw, channels, depth_mult], dw_data).map_err(|e| {
        OnnxError::DecodeFailed {
            message: e.to_string(),
        }
    })
}

mod fused;
pub(crate) use fused::*;

pub(super) fn exec_conv(
    node: &OnnxNode,
    env: &mut TensorEnv,
    activation: yscv_kernels::Activation,
) -> Result<(), OnnxError> {
    exec_conv_with_params(node, env, activation, None, None)
}

/// Resolves `(stride, group, pads, has_padding)` from either precomputed
/// `ConvParams` or the ONNX attribute HashMap. Shared between the thin
/// env-binding wrapper `exec_conv_with_params` and the fused-pair path
/// `exec_fused_dw_pw`.
#[inline]
fn resolve_conv_params(
    node: &OnnxNode,
    precomputed: Option<&crate::loader::ConvParams>,
) -> (usize, usize, usize, usize, usize, usize, usize, bool) {
    if let Some(p) = precomputed {
        (
            p.stride_h,
            p.stride_w,
            p.group,
            p.pad_top,
            p.pad_left,
            p.pad_bottom,
            p.pad_right,
            p.has_padding,
        )
    } else {
        let strides = get_attr_ints(node, "strides").unwrap_or_else(|| vec![1, 1]);
        let pads = get_attr_ints(node, "pads").unwrap_or_else(|| vec![0, 0, 0, 0]);
        let group = get_attr_int(node, "group").unwrap_or(1) as usize;
        let (pt, pl) = (pads[0] as usize, pads[1] as usize);
        let (pb, pr) = (
            pads.get(2).copied().unwrap_or(0) as usize,
            pads.get(3).copied().unwrap_or(0) as usize,
        );
        let sh = strides[0] as usize;
        let sw = strides.get(1).copied().unwrap_or(1) as usize;
        let has_padding = pads.iter().any(|&p| p > 0);
        (sh, sw, group, pt, pl, pb, pr, has_padding)
    }
}

/// Conv with optional pre-computed params (skips HashMap attr lookups).
///
/// Thin wrapper around [`conv_compute_nhwc`]: resolves input/weight/bias
/// from `env`, converts input to NHWC if needed, calls the pure-compute
/// path, then binds the result back into `env`. The split lets the
/// DW+PW fused path (`exec_fused_dw_pw`) chain two conv computes with
/// the DW intermediate kept as a local `Tensor` — never touching the
/// env HashMap.
pub(super) fn exec_conv_with_params(
    node: &OnnxNode,
    env: &mut TensorEnv,
    activation: yscv_kernels::Activation,
    precomputed: Option<&crate::loader::ConvParams>,
    prepacked_weight: Option<&yscv_kernels::PackedB>,
) -> Result<(), OnnxError> {
    let input_is_nhwc = env.is_nhwc(&node.inputs[0]);

    // BNNS NCHW fast path: when input is already NCHW, use Apple Accelerate
    // directly without any layout conversion. Opt-in via YSCV_BNNS=1.
    #[cfg(all(target_os = "macos", feature = "blas"))]
    if !input_is_nhwc
        && std::env::var("YSCV_BNNS").is_ok()
        && let Some(result) = exec_conv_bnns_nchw(node, env, activation)?
    {
        note_conv_kernel(ConvKernel::BnnsNchw);
        env.insert(node.outputs[0].clone(), result);
        // Do NOT mark_nhwc — output stays NCHW
        return Ok(());
    }

    let input = get_tensor(env, &node.name, &node.inputs[0])?;
    let weight = get_tensor(env, &node.name, &node.inputs[1])?;
    let bias = if node.inputs.len() > 2 && !node.inputs[2].is_empty() {
        Some(get_tensor(env, &node.name, &node.inputs[2])?)
    } else {
        None
    };

    let (sh, sw, group, pt, pl, pb, pr, has_padding) = resolve_conv_params(node, precomputed);

    // Skip NCHW→NHWC if input is already NHWC
    let input_nhwc_owned;
    let input_nhwc: &Tensor = if input_is_nhwc {
        input
    } else {
        input_nhwc_owned = nchw_to_nhwc(input)?;
        &input_nhwc_owned
    };

    let output = conv_compute_nhwc(
        node,
        input_nhwc,
        weight,
        bias,
        env,
        prepacked_weight,
        activation,
        sh,
        sw,
        group,
        pt,
        pl,
        pb,
        pr,
        has_padding,
    )?;
    env.insert(node.outputs[0].clone(), output);
    env.mark_nhwc(&node.outputs[0]);
    Ok(())
}

/// Pure-compute variant of the Conv dispatch. Takes an already-NHWC
/// input tensor, the raw weight and bias tensors, plus all resolved
/// shape/stride/pad params, and returns the NHWC output tensor. Does
/// **not** mutate `env` — the caller is responsible for binding the
/// result. `env` is read only for weight-layout flags (KHWC variants)
/// and the prepacked-B lookup.
///
/// Mirrors the three-branch dispatch (group == 1 → NHWC Conv, group
/// == C → depthwise, general grouped) of the original inline body
/// with every `env.insert` / `env.mark_nhwc` replaced by a returned
/// tensor.
#[allow(clippy::too_many_arguments)]
fn conv_compute_nhwc(
    node: &OnnxNode,
    input_nhwc: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    env: &TensorEnv,
    prepacked_weight: Option<&yscv_kernels::PackedB>,
    activation: yscv_kernels::Activation,
    sh: usize,
    sw: usize,
    group: usize,
    pt: usize,
    pl: usize,
    pb: usize,
    pr: usize,
    has_padding: bool,
) -> Result<Tensor, OnnxError> {
    // Weight: ONNX [O, I/group, KH, KW]; pre-permuted group=1 is [KH, KW, I, O].
    let w_shape = weight.shape();
    let is_dw_khwc = group > 1 && env.is_dw_khwc_weight(&node.inputs[1]);
    let is_group_khwc = group > 1 && env.is_group_khwc_weight(&node.inputs[1]);
    let (o_ch, i_per_g, kh, kw) = if env.is_khwc_weight(&node.inputs[1]) {
        (w_shape[3], w_shape[2], w_shape[0], w_shape[1])
    } else if is_dw_khwc {
        (
            w_shape[2].saturating_mul(w_shape[3]),
            1,
            w_shape[0],
            w_shape[1],
        )
    } else if is_group_khwc {
        (w_shape[0], w_shape[3], w_shape[1], w_shape[2])
    } else {
        (w_shape[0], w_shape[1], w_shape[2], w_shape[3])
    };

    if group == 1 {
        // Use pre-permuted weight if available (OIHW→KHWC done once upfront).
        let w_nhwc_owned;
        let w_nhwc: &Tensor = if env.is_khwc_weight(&node.inputs[1]) {
            weight
        } else {
            w_nhwc_owned = oihw_to_khwc_cout(weight)?;
            &w_nhwc_owned
        };
        // aarch64 3×3 non-DW via indirect convolution — avoids
        // the separate pad_nhwc allocation AND the im2col step by walking
        // output positions with inline kernel-tap padding checks. Benefits
        // RAM-bandwidth-constrained ARM SoCs (RK3588, Graviton). Only 3×3
        // since the current indirect implementation assumes KH=KW with
        // simple strided indexing.
        #[cfg(target_arch = "aarch64")]
        {
            // The first-layer 3-channel 3×3 stride-2 Conv routes to its
            // dedicated microkernel (via conv2d_nhwc_padded below), which holds
            // the c_out accumulators in registers across the 27 taps. The
            // generic indirect path round-trips the accumulator through memory
            // once per (tap, in-channel) — fine when c_in is wide enough to
            // amortise it, but ~27× off peak at c_in = 3.
            let is_first_layer_3ch = input_nhwc.shape()[3] == 3 && sh == 2 && sw == 2;
            if kh == 3 && kw == 3 && group == 1 && !cfg!(miri) && !is_first_layer_3ch {
                let t = yscv_kernels::conv2d_nhwc_indirect_padded(
                    input_nhwc, w_nhwc, bias, sh, sw, pt, pl, pb, pr, activation,
                )
                .map_err(|e| OnnxError::DecodeFailed {
                    message: e.to_string(),
                })?;
                note_conv_kernel(ConvKernel::IndirectNhwc3x3);
                return Ok(t);
            }
        }

        // Look up a load-time pre-packed B for this weight, if any. Only
        // pointwise Convs with KHWC layout get a prepack (see build_runtime_index).
        let prepacked = prepacked_weight.or_else(|| env.prepacked_b(&node.inputs[1]));

        // Always use conv2d_nhwc_padded for the padded branch: it contains the
        // dedicated first-layer 3×3 stride-2 RGB microkernel (C_in=3 fast path)
        // and falls through to the same blocked-GEMM as the no-blas path for
        // all other shapes.  The old split (#[cfg(feature = "blas")] →
        // conv2d_nhwc_padded, else → pad_nhwc + generic) caused the first-layer
        // kernel to be silently bypassed when blas is disabled.
        let (mut out_nhwc, activation_fused) = if has_padding {
            let t = yscv_kernels::conv2d_nhwc_padded(
                input_nhwc, w_nhwc, bias, sh, sw, pt, pl, pb, pr, activation,
            )
            .map_err(|e| OnnxError::DecodeFailed {
                message: e.to_string(),
            })?;
            (t, true)
        } else {
            let t = yscv_kernels::conv2d_nhwc_with_activation_prepacked_default(
                input_nhwc, w_nhwc, bias, sh, sw, activation, prepacked,
            )
            .map_err(|e| OnnxError::DecodeFailed {
                message: e.to_string(),
            })?;
            (t, true)
        };
        note_conv_kernel(if has_padding {
            ConvKernel::NhwcPadded
        } else if prepacked.is_some() {
            ConvKernel::NhwcGemmPrepacked
        } else {
            ConvKernel::NhwcGemm
        });
        apply_conv_activation(&mut out_nhwc, activation, activation_fused);
        Ok(out_nhwc)
    } else if group == o_ch && group == input_nhwc.shape()[3] {
        let c = group;
        let depth_mult = o_ch / c;

        // Native NCHWc DW 3×3 stride-1 SAME-pad kernel, opt-in via
        // `YSCV_NCHWC_DW=1`, default OFF: the current intrinsics path loses to
        // the mature NHWC im2col+SIMD+MT path. Kept wired so a future
        // inline-asm tuning pass can flip the default.
        if kh == 3
            && kw == 3
            && sh == 1
            && sw == 1
            && pt == 1
            && pl == 1
            && pb == 1
            && pr == 1
            && depth_mult == 1
            && c.is_multiple_of(8)
            && std::env::var("YSCV_NCHWC_DW").is_ok()
        {
            let dw_kernel_owned;
            let dw_kernel: &Tensor = if is_dw_khwc {
                weight
            } else {
                dw_kernel_owned =
                    repack_depthwise_kernel_once(weight, o_ch, i_per_g, kh, kw, c, depth_mult)?;
                &dw_kernel_owned
            };
            let input_nchwc = yscv_kernels::nhwc_to_nchwc(input_nhwc, 8).map_err(|e| {
                OnnxError::DecodeFailed {
                    message: e.to_string(),
                }
            })?;
            let out_nchwc = yscv_kernels::conv2d_nchwc_dw3x3_s1_same_pad(
                &input_nchwc,
                dw_kernel,
                bias,
                activation,
                c,
                yscv_kernels::ParallelElementwiseConfig::default(),
                None,
                None,
            )
            .map_err(|e| OnnxError::DecodeFailed {
                message: e.to_string(),
            })?;
            let mut out_nhwc = yscv_kernels::nchwc_to_nhwc(&out_nchwc, c).map_err(|e| {
                OnnxError::DecodeFailed {
                    message: e.to_string(),
                }
            })?;
            note_conv_kernel(ConvKernel::DepthwiseNchwc3x3);
            apply_conv_activation(&mut out_nhwc, activation, true);
            return Ok(out_nhwc);
        }

        let activation_fused = true;
        let mut out_nhwc = if has_padding {
            if is_dw_khwc {
                yscv_kernels::depthwise_conv2d_nhwc_padded_with_activation(
                    input_nhwc, weight, bias, sh, sw, pt, pl, pb, pr, activation,
                )
                .map_err(|e| OnnxError::DecodeFailed {
                    message: e.to_string(),
                })
            } else {
                let dw_kernel =
                    repack_depthwise_kernel_once(weight, o_ch, i_per_g, kh, kw, c, depth_mult)?;
                yscv_kernels::depthwise_conv2d_nhwc_padded_with_activation(
                    input_nhwc, &dw_kernel, bias, sh, sw, pt, pl, pb, pr, activation,
                )
                .map_err(|e| OnnxError::DecodeFailed {
                    message: e.to_string(),
                })
            }
        } else if is_dw_khwc {
            yscv_kernels::depthwise_conv2d_nhwc_with_activation(
                input_nhwc, weight, bias, sh, sw, activation,
            )
            .map_err(|e| OnnxError::DecodeFailed {
                message: e.to_string(),
            })
        } else {
            let dw_kernel =
                repack_depthwise_kernel_once(weight, o_ch, i_per_g, kh, kw, c, depth_mult)?;
            yscv_kernels::depthwise_conv2d_nhwc_with_activation(
                input_nhwc, &dw_kernel, bias, sh, sw, activation,
            )
            .map_err(|e| OnnxError::DecodeFailed {
                message: e.to_string(),
            })
        }?;
        note_conv_kernel(if has_padding {
            ConvKernel::DepthwiseNhwcPadded
        } else {
            ConvKernel::DepthwiseNhwc
        });
        apply_conv_activation(&mut out_nhwc, activation, activation_fused);
        Ok(out_nhwc)
    } else {
        // Grouped convolution with virtual padding (no explicit padded tensor).
        let in_shape = input_nhwc.shape();
        let (n, ih, iw, total_ic) = (in_shape[0], in_shape[1], in_shape[2], in_shape[3]);
        if total_ic % group != 0 || o_ch % group != 0 {
            return Err(OnnxError::DecodeFailed {
                message: format!(
                    "Grouped Conv channel mismatch: IC={total_ic}, OC={o_ch}, group={group}"
                ),
            });
        }
        let ic_per_group = total_ic / group;
        let oc_per_group = o_ch / group;
        if i_per_g != ic_per_group {
            return Err(OnnxError::DecodeFailed {
                message: format!(
                    "Grouped Conv weight/input mismatch: weight I/G={i_per_g}, input I/G={ic_per_group}"
                ),
            });
        }
        let padded_h = ih + pt + pb;
        let padded_w = iw + pl + pr;
        if kh > padded_h || kw > padded_w {
            return Err(OnnxError::DecodeFailed {
                message: format!(
                    "Grouped Conv kernel larger than padded input: input=({ih},{iw}), pads=({pt},{pl},{pb},{pr}), kernel=({kh},{kw})"
                ),
            });
        }
        let oh = (padded_h - kh) / sh + 1;
        let ow = (padded_w - kw) / sw + 1;
        let mut out_data = vec![0.0f32; n * oh * ow * o_ch];

        let in_data = input_nhwc.data();
        let w_data = weight.data();

        let w_khwc_stride = kh * kw * ic_per_group;
        let w_reordered: std::borrow::Cow<'_, [f32]> = if is_group_khwc {
            // Already pre-packed at model-load time: [O, KH, KW, I/G].
            std::borrow::Cow::Borrowed(w_data)
        } else {
            // Fallback for non-prepacked models: OIHW -> [O, KH, KW, I/G].
            let mut reordered = vec![0.0f32; o_ch * w_khwc_stride];
            for oc in 0..o_ch {
                for ki in 0..kh {
                    for kj in 0..kw {
                        let dst_base = oc * w_khwc_stride + (ki * kw + kj) * ic_per_group;
                        for ci in 0..ic_per_group {
                            reordered[dst_base + ci] =
                                w_data[((oc * ic_per_group + ci) * kh + ki) * kw + kj];
                        }
                    }
                }
            }
            std::borrow::Cow::Owned(reordered)
        };

        let bias_data: &[f32] = match &bias {
            Some(b) => b.data(),
            None => &[],
        };
        let dot = dot_dispatch();
        let relu_fused = activation == yscv_kernels::Activation::Relu;

        for batch in 0..n {
            for g in 0..group {
                let ic_start = g * ic_per_group;
                let oc_start = g * oc_per_group;
                for orow in 0..oh {
                    for ocol in 0..ow {
                        let out_base = ((batch * oh + orow) * ow + ocol) * o_ch + oc_start;
                        for oc in 0..oc_per_group {
                            let abs_oc = oc_start + oc;
                            let mut val = if !bias_data.is_empty() {
                                bias_data[abs_oc]
                            } else {
                                0.0
                            };
                            let w_oc_base = abs_oc * w_khwc_stride;
                            if has_padding {
                                for ki in 0..kh {
                                    let ir_raw = orow * sh + ki;
                                    if ir_raw < pt || ir_raw >= pt + ih {
                                        continue;
                                    }
                                    let ir = ir_raw - pt;
                                    for kj in 0..kw {
                                        let ic_raw = ocol * sw + kj;
                                        if ic_raw < pl || ic_raw >= pl + iw {
                                            continue;
                                        }
                                        let ic_pos = ic_raw - pl;
                                        let in_base =
                                            ((batch * ih + ir) * iw + ic_pos) * total_ic + ic_start;
                                        let w_base = w_oc_base + (ki * kw + kj) * ic_per_group;
                                        let in_slice = &in_data[in_base..in_base + ic_per_group];
                                        let w_slice = &w_reordered[w_base..w_base + ic_per_group];
                                        val += dot(in_slice, w_slice);
                                    }
                                }
                            } else {
                                for ki in 0..kh {
                                    let ir = orow * sh + ki;
                                    for kj in 0..kw {
                                        let ic_pos = ocol * sw + kj;
                                        let in_base =
                                            ((batch * ih + ir) * iw + ic_pos) * total_ic + ic_start;
                                        let w_base = w_oc_base + (ki * kw + kj) * ic_per_group;
                                        let in_slice = &in_data[in_base..in_base + ic_per_group];
                                        let w_slice = &w_reordered[w_base..w_base + ic_per_group];
                                        val += dot(in_slice, w_slice);
                                    }
                                }
                            }
                            out_data[out_base + oc] = if relu_fused { val.max(0.0) } else { val };
                        }
                    }
                }
            }
        }
        let mut out_nhwc = Tensor::from_vec(vec![n, oh, ow, o_ch], out_data).map_err(|e| {
            OnnxError::DecodeFailed {
                message: e.to_string(),
            }
        })?;
        note_conv_kernel(ConvKernel::Grouped);
        apply_conv_activation(&mut out_nhwc, activation, relu_fused);
        Ok(out_nhwc)
    }
}

pub(super) fn exec_conv_transpose(node: &OnnxNode, env: &mut TensorEnv) -> Result<(), OnnxError> {
    let input_is_nhwc = env.is_nhwc(&node.inputs[0]);
    let input = get_tensor(env, &node.name, &node.inputs[0])?;
    let weight = get_tensor(env, &node.name, &node.inputs[1])?;
    let bias = if node.inputs.len() > 2 && !node.inputs[2].is_empty() {
        Some(get_tensor(env, &node.name, &node.inputs[2])?)
    } else {
        None
    };

    let strides = get_attr_ints(node, "strides").unwrap_or_else(|| vec![1, 1]);
    let sh = strides[0] as usize;
    let sw = strides[1] as usize;

    let input_nhwc = if input_is_nhwc {
        input.clone()
    } else {
        nchw_to_nhwc(input)?
    };
    // ONNX ConvTranspose weight: [C_in, C_out, KH, KW] → [KH, KW, C_in, C_out]
    let w_t = weight
        .permute(&[2, 3, 0, 1])
        .map_err(|e| OnnxError::DecodeFailed {
            message: e.to_string(),
        })?;

    let out_nhwc =
        yscv_kernels::transpose_conv2d_nhwc(&input_nhwc, &w_t, bias, sh, sw).map_err(|e| {
            OnnxError::DecodeFailed {
                message: e.to_string(),
            }
        })?;
    env.insert(node.outputs[0].clone(), out_nhwc);
    env.mark_nhwc(&node.outputs[0]);
    Ok(())
}

mod quantized;
pub(crate) use quantized::*;

/// ONNX DeformConv: deformable convolution with learned offsets.
///
/// Inputs: X (NCHW), offset, W (OIHW), [bias]
/// Attributes: strides, pads, group (only group=1 supported currently)
///
/// Converts NCHW inputs to NHWC, permutes weight from OIHW to [KH,KW,C_in,C_out],
/// and delegates to the `deformable_conv2d_nhwc` kernel.
pub(super) fn exec_deform_conv(node: &OnnxNode, env: &mut TensorEnv) -> Result<(), OnnxError> {
    let input_is_nhwc = env.is_nhwc(&node.inputs[0]);

    let input = get_tensor(env, &node.name, &node.inputs[0])?;
    let offset = get_tensor(env, &node.name, &node.inputs[1])?;
    let weight = get_tensor(env, &node.name, &node.inputs[2])?;
    let bias = if node.inputs.len() > 3 && !node.inputs[3].is_empty() {
        Some(get_tensor(env, &node.name, &node.inputs[3])?)
    } else {
        None
    };

    let strides = get_attr_ints(node, "strides").unwrap_or_else(|| vec![1, 1]);
    let pads = get_attr_ints(node, "pads").unwrap_or_else(|| vec![0, 0, 0, 0]);

    let stride = strides[0] as usize;
    // Symmetric padding only — use first pad value
    let padding = pads[0] as usize;

    // Convert input to NHWC if needed
    let input_nhwc = if input_is_nhwc {
        input.clone()
    } else {
        nchw_to_nhwc(input)?
    };

    // Convert offset from NCHW [N, kH*kW*2, out_H, out_W] to NHWC [N, out_H, out_W, kH*kW*2]
    let offset_nhwc = if offset.rank() == 4 {
        offset
            .permute(&[0, 2, 3, 1])
            .map_err(|e| OnnxError::DecodeFailed {
                message: e.to_string(),
            })?
    } else {
        offset.clone()
    };

    // Weight: ONNX [O, I, KH, KW] → kernel expects [KH, KW, I, O]
    let w_nhwc = weight
        .permute(&[2, 3, 1, 0])
        .map_err(|e| OnnxError::DecodeFailed {
            message: e.to_string(),
        })?;

    let out_nhwc = yscv_kernels::deformable_conv2d_nhwc(
        &input_nhwc,
        &w_nhwc,
        &offset_nhwc,
        bias,
        stride,
        padding,
    )
    .map_err(|e| OnnxError::DecodeFailed {
        message: e.to_string(),
    })?;

    env.insert(node.outputs[0].clone(), out_nhwc);
    env.mark_nhwc(&node.outputs[0]);
    Ok(())
}

// ── layout conversion helpers ──────────────────────────────────────

pub(super) fn nchw_to_nhwc(input: &Tensor) -> Result<Tensor, OnnxError> {
    // Uses the specialized AVX 8×8 block transpose when c%8==0 and hw%8==0,
    // falling back to the generic `Tensor::permute` for non-aligned shapes
    // (e.g. first-layer RGB c=3).
    yscv_kernels::nchw_to_nhwc_fast(input).map_err(|e| OnnxError::DecodeFailed {
        message: e.to_string(),
    })
}

/// Convert ONNX Conv weight [O, I, KH, KW] to yscv [KH, KW, I, O]
pub(super) fn oihw_to_khwc_cout(weight: &Tensor) -> Result<Tensor, OnnxError> {
    weight
        .permute(&[2, 3, 1, 0])
        .map_err(|e| OnnxError::DecodeFailed {
            message: e.to_string(),
        })
}

/// Zero-pad an NHWC tensor on H/W dimensions.
pub(super) fn pad_nhwc(
    input: &Tensor,
    top: usize,
    left: usize,
    bottom: usize,
    right: usize,
) -> Result<Tensor, OnnxError> {
    pad_nhwc_val(input, top, left, bottom, right, 0.0)
}

pub(super) fn pad_nhwc_val(
    input: &Tensor,
    top: usize,
    left: usize,
    bottom: usize,
    right: usize,
    val: f32,
) -> Result<Tensor, OnnxError> {
    let shape = input.shape();
    let (n, h, w, c) = (shape[0], shape[1], shape[2], shape[3]);
    let oh = h + top + bottom;
    let ow = w + left + right;
    let mut out = vec![val; n * oh * ow * c];
    let in_data = input.data();
    let row_bytes = w * c;
    for batch in 0..n {
        for row in 0..h {
            let src_start = (batch * h + row) * w * c;
            let dst_start = ((batch * oh + row + top) * ow + left) * c;
            out[dst_start..dst_start + row_bytes]
                .copy_from_slice(&in_data[src_start..src_start + row_bytes]);
        }
    }
    Tensor::from_vec(vec![n, oh, ow, c], out).map_err(|e| OnnxError::DecodeFailed {
        message: e.to_string(),
    })
}

// ── BNNS NCHW fast path ────────────────────────────────────────────

/// Try to execute conv via Apple BNNS on NCHW data.
/// Returns `Ok(Some(tensor))` on success, `Ok(None)` if BNNS can't handle this op.
#[cfg(all(target_os = "macos", feature = "blas"))]
fn exec_conv_bnns_nchw(
    node: &OnnxNode,
    env: &mut TensorEnv,
    activation: yscv_kernels::Activation,
) -> Result<Option<Tensor>, OnnxError> {
    use yscv_kernels::bnns_conv::{BnnsActivation, BnnsConvParams, conv2d_nchw_bnns};

    let input = get_tensor(env, &node.name, &node.inputs[0])?;
    let weight = get_tensor(env, &node.name, &node.inputs[1])?;
    let bias = if node.inputs.len() > 2 && !node.inputs[2].is_empty() {
        Some(get_tensor(env, &node.name, &node.inputs[2])?)
    } else {
        None
    };

    let strides = get_attr_ints(node, "strides").unwrap_or_else(|| vec![1, 1]);
    let pads = get_attr_ints(node, "pads").unwrap_or_else(|| vec![0, 0, 0, 0]);
    let group = get_attr_int(node, "group").unwrap_or(1) as usize;
    let sh = strides[0] as usize;
    let sw = strides[1] as usize;
    let (pt, pl, pb, pr) = (
        pads[0] as usize,
        pads[1] as usize,
        pads[2] as usize,
        pads[3] as usize,
    );

    // Weight must be OIHW for BNNS. group=1 weights are pre-permuted to KHWC —
    // reverse them back. Depthwise/grouped weights are already OIHW.
    let w_oihw_owned;
    let w_oihw: &Tensor = if env.is_khwc_weight(&node.inputs[1]) {
        // KHWC [KH, KW, I, O] → OIHW [O, I, KH, KW]
        w_oihw_owned = weight
            .permute(&[3, 2, 0, 1])
            .map_err(|e| OnnxError::DecodeFailed {
                message: e.to_string(),
            })?;
        &w_oihw_owned
    } else {
        weight
    };

    let in_shape = input.shape();
    if in_shape.len() != 4 {
        return Ok(None);
    }
    let (batch, in_c, in_h, in_w) = (in_shape[0], in_shape[1], in_shape[2], in_shape[3]);

    let w_shape = w_oihw.shape();
    let (out_c, _ic_per_g, kh, kw) = (w_shape[0], w_shape[1], w_shape[2], w_shape[3]);

    let out_h = (in_h + pt + pb - kh) / sh + 1;
    let out_w = (in_w + pl + pr - kw) / sw + 1;

    let bnns_act = match activation {
        yscv_kernels::Activation::Silu => BnnsActivation::Silu,
        yscv_kernels::Activation::Relu => BnnsActivation::Relu,
        yscv_kernels::Activation::None => BnnsActivation::None,
    };

    let params = BnnsConvParams {
        batch,
        in_c,
        in_h,
        in_w,
        out_c,
        out_h,
        out_w,
        kh,
        kw,
        stride_h: sh,
        stride_w: sw,
        pad_top: pt,
        pad_left: pl,
        pad_bottom: pb,
        pad_right: pr,
        groups: group,
        activation: bnns_act,
    };

    Ok(conv2d_nchw_bnns(input, w_oihw, bias, &params))
}
