//! NHWC depthwise conv kernels: the per-tap FMA primitives, the per-arch
//! row kernels (AVX-512/AVX2/NEON/SSE/scalar), the channels=16 fast path,
//! and the depth-multiplier accumulator.

use super::*;

/// Element-wise FMA across channels for padded DW border pixels: `out[i] += inp[i] * ker[i]`.
/// Replaces the scalar `for ch in 0..channels` inner loop with platform SIMD.
#[inline(always)]
pub(super) fn depthwise_tap_fma(out: &mut [f32], inp: &[f32], ker: &[f32]) {
    debug_assert_eq!(out.len(), inp.len());
    debug_assert_eq!(out.len(), ker.len());
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if crate::host_cpu().features.avx512f {
            #[allow(unsafe_code)]
            unsafe {
                depthwise_tap_fma_avx512(out, inp, ker);
            }
            return;
        }
        if crate::host_cpu().features.avx && crate::host_cpu().features.fma {
            #[allow(unsafe_code)]
            unsafe {
                depthwise_tap_fma_avx_fma(out, inp, ker);
            }
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        #[allow(unsafe_code)]
        unsafe {
            depthwise_tap_fma_neon(out, inp, ker);
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    #[allow(clippy::needless_range_loop)]
    for i in 0..out.len() {
        out[i] += inp[i] * ker[i];
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
unsafe fn depthwise_tap_fma_avx512(out: &mut [f32], inp: &[f32], ker: &[f32]) {
    #[cfg(target_arch = "x86")]
    use core::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::*;
    let n = out.len();
    let op = out.as_mut_ptr();
    let ip = inp.as_ptr();
    let kp = ker.as_ptr();
    let mut i = 0usize;
    while i + 128 <= n {
        _mm512_storeu_ps(
            op.add(i),
            _mm512_fmadd_ps(
                _mm512_loadu_ps(ip.add(i)),
                _mm512_loadu_ps(kp.add(i)),
                _mm512_loadu_ps(op.add(i)),
            ),
        );
        _mm512_storeu_ps(
            op.add(i + 16),
            _mm512_fmadd_ps(
                _mm512_loadu_ps(ip.add(i + 16)),
                _mm512_loadu_ps(kp.add(i + 16)),
                _mm512_loadu_ps(op.add(i + 16)),
            ),
        );
        _mm512_storeu_ps(
            op.add(i + 32),
            _mm512_fmadd_ps(
                _mm512_loadu_ps(ip.add(i + 32)),
                _mm512_loadu_ps(kp.add(i + 32)),
                _mm512_loadu_ps(op.add(i + 32)),
            ),
        );
        _mm512_storeu_ps(
            op.add(i + 48),
            _mm512_fmadd_ps(
                _mm512_loadu_ps(ip.add(i + 48)),
                _mm512_loadu_ps(kp.add(i + 48)),
                _mm512_loadu_ps(op.add(i + 48)),
            ),
        );
        _mm512_storeu_ps(
            op.add(i + 64),
            _mm512_fmadd_ps(
                _mm512_loadu_ps(ip.add(i + 64)),
                _mm512_loadu_ps(kp.add(i + 64)),
                _mm512_loadu_ps(op.add(i + 64)),
            ),
        );
        _mm512_storeu_ps(
            op.add(i + 80),
            _mm512_fmadd_ps(
                _mm512_loadu_ps(ip.add(i + 80)),
                _mm512_loadu_ps(kp.add(i + 80)),
                _mm512_loadu_ps(op.add(i + 80)),
            ),
        );
        _mm512_storeu_ps(
            op.add(i + 96),
            _mm512_fmadd_ps(
                _mm512_loadu_ps(ip.add(i + 96)),
                _mm512_loadu_ps(kp.add(i + 96)),
                _mm512_loadu_ps(op.add(i + 96)),
            ),
        );
        _mm512_storeu_ps(
            op.add(i + 112),
            _mm512_fmadd_ps(
                _mm512_loadu_ps(ip.add(i + 112)),
                _mm512_loadu_ps(kp.add(i + 112)),
                _mm512_loadu_ps(op.add(i + 112)),
            ),
        );
        i += 128;
    }
    while i + 16 <= n {
        _mm512_storeu_ps(
            op.add(i),
            _mm512_fmadd_ps(
                _mm512_loadu_ps(ip.add(i)),
                _mm512_loadu_ps(kp.add(i)),
                _mm512_loadu_ps(op.add(i)),
            ),
        );
        i += 16;
    }
    while i < n {
        *op.add(i) += *ip.add(i) * *kp.add(i);
        i += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx,fma")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
unsafe fn depthwise_tap_fma_avx_fma(out: &mut [f32], inp: &[f32], ker: &[f32]) {
    #[cfg(target_arch = "x86")]
    use core::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::*;
    let n = out.len();
    let op = out.as_mut_ptr();
    let ip = inp.as_ptr();
    let kp = ker.as_ptr();
    let mut i = 0usize;
    while i + 8 <= n {
        _mm256_storeu_ps(
            op.add(i),
            _mm256_fmadd_ps(
                _mm256_loadu_ps(ip.add(i)),
                _mm256_loadu_ps(kp.add(i)),
                _mm256_loadu_ps(op.add(i)),
            ),
        );
        i += 8;
    }
    while i < n {
        *op.add(i) += *ip.add(i) * *kp.add(i);
        i += 1;
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
unsafe fn depthwise_tap_fma_neon(out: &mut [f32], inp: &[f32], ker: &[f32]) {
    use core::arch::aarch64::*;
    let n = out.len();
    let op = out.as_mut_ptr();
    let ip = inp.as_ptr();
    let kp = ker.as_ptr();
    let mut i = 0usize;
    while i + 4 <= n {
        vst1q_f32(
            op.add(i),
            vfmaq_f32(
                vld1q_f32(op.add(i)),
                vld1q_f32(ip.add(i)),
                vld1q_f32(kp.add(i)),
            ),
        );
        i += 4;
    }
    while i < n {
        *op.add(i) += *ip.add(i) * *kp.add(i);
        i += 1;
    }
}

#[inline]
pub(super) fn depthwise_accumulate_dm(
    out_cell: &mut [f32],
    out_ch_base: usize,
    kernel_data: &[f32],
    kernel_ch_base: usize,
    input_val: f32,
    depth_multiplier: usize,
) {
    match depth_multiplier {
        2 => {
            out_cell[out_ch_base] += input_val * kernel_data[kernel_ch_base];
            out_cell[out_ch_base + 1] += input_val * kernel_data[kernel_ch_base + 1];
        }
        4 => {
            out_cell[out_ch_base] += input_val * kernel_data[kernel_ch_base];
            out_cell[out_ch_base + 1] += input_val * kernel_data[kernel_ch_base + 1];
            out_cell[out_ch_base + 2] += input_val * kernel_data[kernel_ch_base + 2];
            out_cell[out_ch_base + 3] += input_val * kernel_data[kernel_ch_base + 3];
        }
        _ => {
            let mut dm = 0usize;
            while dm + 8 <= depth_multiplier {
                conv_fma_row(
                    &mut out_cell[out_ch_base + dm..out_ch_base + dm + 8],
                    &kernel_data[kernel_ch_base + dm..kernel_ch_base + dm + 8],
                    input_val,
                );
                dm += 8;
            }
            while dm < depth_multiplier {
                out_cell[out_ch_base + dm] += input_val * kernel_data[kernel_ch_base + dm];
                dm += 1;
            }
        }
    }
}

/// FMA: out[i] += kernel[i] * input_val, SIMD-accelerated
#[allow(unsafe_code)]
pub(super) fn conv_fma_row(out: &mut [f32], kernel: &[f32], input_val: f32) {
    let len = out.len();
    debug_assert_eq!(len, kernel.len());

    if cfg!(miri) || len < 4 {
        conv_fma_scalar(out, kernel, input_val);
        return;
    }

    conv_fma_impl()(out, kernel, input_val);
}

type ConvFmaFn = fn(&mut [f32], &[f32], f32);

#[inline]
fn conv_fma_impl() -> ConvFmaFn {
    static IMPL: OnceLock<ConvFmaFn> = OnceLock::new();
    *IMPL.get_or_init(|| {
        #[cfg(target_arch = "aarch64")]
        {
            if crate::host_cpu().features.neon {
                return conv_fma_neon_dispatch;
            }
        }

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if crate::host_cpu().features.avx {
                return conv_fma_avx_dispatch;
            }
            if crate::host_cpu().features.sse {
                return conv_fma_sse_dispatch;
            }
        }

        conv_fma_scalar
    })
}

#[inline]
fn conv_fma_scalar(out: &mut [f32], kernel: &[f32], input_val: f32) {
    for i in 0..out.len() {
        out[i] += kernel[i] * input_val;
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
fn conv_fma_neon_dispatch(out: &mut [f32], kernel: &[f32], input_val: f32) {
    // SAFETY: selected only after runtime NEON feature detection.
    #[allow(unsafe_code)]
    unsafe {
        conv_fma_neon(out, kernel, input_val);
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline]
fn conv_fma_sse_dispatch(out: &mut [f32], kernel: &[f32], input_val: f32) {
    // SAFETY: selected only after runtime SSE feature detection.
    #[allow(unsafe_code)]
    unsafe {
        conv_fma_sse(out, kernel, input_val);
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline]
fn conv_fma_avx_dispatch(out: &mut [f32], kernel: &[f32], input_val: f32) {
    // SAFETY: selected only after runtime AVX feature detection.
    #[allow(unsafe_code)]
    unsafe {
        conv_fma_avx(out, kernel, input_val);
    }
}

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "neon")]
unsafe fn conv_fma_neon(out: &mut [f32], kernel: &[f32], input_val: f32) {
    use std::arch::aarch64::*;
    let len = out.len();
    let op = out.as_mut_ptr();
    let kp = kernel.as_ptr();
    let v_input = vdupq_n_f32(input_val);
    let mut i = 0usize;
    while i + 4 <= len {
        let o = vld1q_f32(op.add(i));
        let k = vld1q_f32(kp.add(i));
        vst1q_f32(op.add(i), vfmaq_f32(o, k, v_input));
        i += 4;
    }
    while i < len {
        *op.add(i) += *kp.add(i) * input_val;
        i += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "sse")]
unsafe fn conv_fma_sse(out: &mut [f32], kernel: &[f32], input_val: f32) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;
    let len = out.len();
    let op = out.as_mut_ptr();
    let kp = kernel.as_ptr();
    let v_input = _mm_set1_ps(input_val);
    let mut i = 0usize;
    while i + 4 <= len {
        let o = _mm_loadu_ps(op.add(i));
        let k = _mm_loadu_ps(kp.add(i));
        _mm_storeu_ps(op.add(i), _mm_add_ps(o, _mm_mul_ps(k, v_input)));
        i += 4;
    }
    while i < len {
        *op.add(i) += *kp.add(i) * input_val;
        i += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx")]
unsafe fn conv_fma_avx(out: &mut [f32], kernel: &[f32], input_val: f32) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;
    let len = out.len();
    let op = out.as_mut_ptr();
    let kp = kernel.as_ptr();
    let v_input = _mm256_set1_ps(input_val);
    let mut i = 0usize;
    while i + 8 <= len {
        let o = _mm256_loadu_ps(op.add(i));
        let k = _mm256_loadu_ps(kp.add(i));
        _mm256_storeu_ps(op.add(i), _mm256_add_ps(o, _mm256_mul_ps(k, v_input)));
        i += 8;
    }
    if i < len {
        conv_fma_sse(&mut out[i..], &kernel[i..], input_val);
    }
}

/// 3D convolution: input [B, D, H, W, C_in], kernel [KD, KH, KW, C_in, C_out], output [B, OD, OH, OW, C_out]
/// Supports padding and stride in all 3 dimensions.
pub fn conv3d(
    input: &[f32],
    input_shape: &[usize], // [B, D, H, W, C_in]
    kernel: &[f32],
    kernel_shape: &[usize],         // [KD, KH, KW, C_in, C_out]
    stride: (usize, usize, usize),  // (d, h, w)
    padding: (usize, usize, usize), // (d, h, w)
) -> (Vec<f32>, Vec<usize>) {
    assert_eq!(
        input_shape.len(),
        5,
        "input_shape must be [B, D, H, W, C_in]"
    );
    assert_eq!(
        kernel_shape.len(),
        5,
        "kernel_shape must be [KD, KH, KW, C_in, C_out]"
    );

    let (batch, in_d, in_h, in_w, c_in) = (
        input_shape[0],
        input_shape[1],
        input_shape[2],
        input_shape[3],
        input_shape[4],
    );
    let (kd, kh, kw, k_cin, c_out) = (
        kernel_shape[0],
        kernel_shape[1],
        kernel_shape[2],
        kernel_shape[3],
        kernel_shape[4],
    );
    let (stride_d, stride_h, stride_w) = stride;
    let (pad_d, pad_h, pad_w) = padding;

    assert_eq!(c_in, k_cin, "input C_in must match kernel C_in");
    assert!(
        stride_d > 0 && stride_h > 0 && stride_w > 0,
        "strides must be positive"
    );
    assert_eq!(input.len(), batch * in_d * in_h * in_w * c_in);
    assert_eq!(kernel.len(), kd * kh * kw * c_in * c_out);

    let out_d = (in_d + 2 * pad_d - kd) / stride_d + 1;
    let out_h = (in_h + 2 * pad_h - kh) / stride_h + 1;
    let out_w = (in_w + 2 * pad_w - kw) / stride_w + 1;

    let output_shape = vec![batch, out_d, out_h, out_w, c_out];
    let out_spatial = out_d * out_h * out_w;
    let output_len = batch * out_spatial * c_out;

    // im2col + BLAS path: reshape 3D conv into matrix multiply
    // im2col: [out_spatial, kd*kh*kw*c_in]
    // kernel reshaped: [kd*kh*kw*c_in, c_out]
    // output = im2col @ kernel_2d → [out_spatial, c_out]
    #[cfg(feature = "blas")]
    if !cfg!(miri) && batch == 1 {
        let k_spatial = kd * kh * kw;
        let col_k = k_spatial * c_in; // im2col column length
        let mut output = vec![0.0f32; output_len];
        let in_hwc = in_h * in_w * c_in;
        let in_wc = in_w * c_in;

        for b in 0..batch {
            let b_in = b * in_d * in_hwc;
            // Build im2col matrix
            let mut col = vec![0.0f32; out_spatial * col_k];
            let mut row = 0;
            for od in 0..out_d {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let mut col_idx = 0;
                        for fd in 0..kd {
                            let id_raw = od * stride_d + fd;
                            for fh in 0..kh {
                                let ih_raw = oh * stride_h + fh;
                                for fw in 0..kw {
                                    let iw_raw = ow * stride_w + fw;
                                    let in_bounds = id_raw >= pad_d
                                        && id_raw - pad_d < in_d
                                        && ih_raw >= pad_h
                                        && ih_raw - pad_h < in_h
                                        && iw_raw >= pad_w
                                        && iw_raw - pad_w < in_w;
                                    if in_bounds {
                                        let id = id_raw - pad_d;
                                        let ih = ih_raw - pad_h;
                                        let iw = iw_raw - pad_w;
                                        let base = b_in + id * in_hwc + ih * in_wc + iw * c_in;
                                        col[row * col_k + col_idx..row * col_k + col_idx + c_in]
                                            .copy_from_slice(&input[base..base + c_in]);
                                    }
                                    // else: padding zeros (already zeroed)
                                    col_idx += c_in;
                                }
                            }
                        }
                        row += 1;
                    }
                }
            }

            // BLAS: output[b] = col @ kernel_2d
            let b_out = b * out_spatial * c_out;
            super::super::matmul::blas_sgemm(
                &col,
                kernel,
                &mut output[b_out..b_out + out_spatial * c_out],
                out_spatial,
                col_k,
                c_out,
            );
        }
        return (output, output_shape);
    }

    // Fallback: naive 7-nested-loop implementation
    let mut output = vec![0.0f32; output_len];
    let in_dhwc = in_d * in_h * in_w * c_in;
    let in_hwc = in_h * in_w * c_in;
    let in_wc = in_w * c_in;
    let k_hwcico = kh * kw * c_in * c_out;
    let k_wcico = kw * c_in * c_out;
    let k_cico = c_in * c_out;
    let out_dhwco = out_d * out_h * out_w * c_out;
    let out_hwco = out_h * out_w * c_out;
    let out_wco = out_w * c_out;

    for b in 0..batch {
        let b_in = b * in_dhwc;
        let b_out = b * out_dhwco;
        for od in 0..out_d {
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let out_base = b_out + od * out_hwco + oh * out_wco + ow * c_out;
                    for fd in 0..kd {
                        let id = od * stride_d + fd;
                        if id < pad_d || id - pad_d >= in_d {
                            continue;
                        }
                        let id = id - pad_d;
                        for fh in 0..kh {
                            let ih = oh * stride_h + fh;
                            if ih < pad_h || ih - pad_h >= in_h {
                                continue;
                            }
                            let ih = ih - pad_h;
                            for fw in 0..kw {
                                let iw = ow * stride_w + fw;
                                if iw < pad_w || iw - pad_w >= in_w {
                                    continue;
                                }
                                let iw = iw - pad_w;
                                let in_base = b_in + id * in_hwc + ih * in_wc + iw * c_in;
                                let k_base = fd * k_hwcico + fh * k_wcico + fw * k_cico;
                                for ci in 0..c_in {
                                    let input_val = input[in_base + ci];
                                    let k_offset = k_base + ci * c_out;
                                    for co in 0..c_out {
                                        output[out_base + co] += input_val * kernel[k_offset + co];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    (output, output_shape)
}

// ---------------------------------------------------------------------------
// SIMD depthwise conv2d kernels (depth_multiplier == 1 fast path)
// ---------------------------------------------------------------------------

/// NEON-accelerated depthwise conv row for `depth_multiplier == 1`.
/// Vectorizes across the channel dimension (4 channels per `float32x4_t`).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(unsafe_code)]
unsafe fn depthwise_conv2d_nhwc_row_neon(
    input: &[f32],
    kernel: &[f32],
    bias: Option<&[f32]>,
    plan: DepthwiseConv2dPlan,
    row_idx: usize,
    out_row: &mut [f32],
    activation: Activation,
) {
    use core::arch::aarch64::*;

    let batch_idx = row_idx / plan.out_h;
    let out_y = row_idx % plan.out_h;
    let in_y0 = out_y * plan.stride_h;
    let batch_input_base = batch_idx * plan.in_h * plan.in_w * plan.channels;
    let channels = plan.channels;
    let kh = plan.kernel_h;
    let kw = plan.kernel_w;

    let inp_ptr = input.as_ptr();
    let ker_ptr = kernel.as_ptr();
    let out_ptr = out_row.as_mut_ptr();
    let bias_ptr = bias.map(|b| b.as_ptr());
    let do_relu = matches!(activation, Activation::Relu);
    let zero = vdupq_n_f32(0.0);

    let stride_w = plan.stride_w;
    let ch16_end = (channels / 16) * 16;
    let tile_end = (plan.out_w / 4) * 4;

    // Spatial register blocking: process 4 output columns per pass over the
    // kernel taps. The depthwise weight is independent of output position, so
    // one weight quad feeds all 4 pixels (4× fewer weight loads), and the 16
    // independent accumulators give the in-order core the ILP to hide per-tap
    // input-load latency — the dominant cost when the DW input is L2/DRAM
    // resident. Channels below the 16-multiple and columns below the 4-multiple
    // fall through to the per-pixel loop, which starts at `ch_start`.
    let mut tx = 0;
    while tx < tile_end {
        let in_x0 = tx * stride_w;
        let out_base = tx * channels;
        let mut ch = 0;
        while ch < ch16_end {
            unsafe {
                let (b0, b1, b2, b3) = if let Some(bp) = bias_ptr {
                    (
                        vld1q_f32(bp.add(ch)),
                        vld1q_f32(bp.add(ch + 4)),
                        vld1q_f32(bp.add(ch + 8)),
                        vld1q_f32(bp.add(ch + 12)),
                    )
                } else {
                    (zero, zero, zero, zero)
                };
                let (mut a00, mut a01, mut a02, mut a03) = (b0, b1, b2, b3);
                let (mut a10, mut a11, mut a12, mut a13) = (b0, b1, b2, b3);
                let (mut a20, mut a21, mut a22, mut a23) = (b0, b1, b2, b3);
                let (mut a30, mut a31, mut a32, mut a33) = (b0, b1, b2, b3);
                let col = stride_w * channels;
                for ky in 0..kh {
                    let in_y = in_y0 + ky;
                    let input_row_base = batch_input_base + (in_y * plan.in_w + in_x0) * channels;
                    let kernel_row_base = (ky * kw) * channels;
                    for kx in 0..kw {
                        let kb = kernel_row_base + kx * channels + ch;
                        let w0 = vld1q_f32(ker_ptr.add(kb));
                        let w1 = vld1q_f32(ker_ptr.add(kb + 4));
                        let w2 = vld1q_f32(ker_ptr.add(kb + 8));
                        let w3 = vld1q_f32(ker_ptr.add(kb + 12));
                        let q0 = input_row_base + kx * channels + ch;
                        a00 = vfmaq_f32(a00, vld1q_f32(inp_ptr.add(q0)), w0);
                        a01 = vfmaq_f32(a01, vld1q_f32(inp_ptr.add(q0 + 4)), w1);
                        a02 = vfmaq_f32(a02, vld1q_f32(inp_ptr.add(q0 + 8)), w2);
                        a03 = vfmaq_f32(a03, vld1q_f32(inp_ptr.add(q0 + 12)), w3);
                        let q1 = q0 + col;
                        a10 = vfmaq_f32(a10, vld1q_f32(inp_ptr.add(q1)), w0);
                        a11 = vfmaq_f32(a11, vld1q_f32(inp_ptr.add(q1 + 4)), w1);
                        a12 = vfmaq_f32(a12, vld1q_f32(inp_ptr.add(q1 + 8)), w2);
                        a13 = vfmaq_f32(a13, vld1q_f32(inp_ptr.add(q1 + 12)), w3);
                        let q2 = q1 + col;
                        a20 = vfmaq_f32(a20, vld1q_f32(inp_ptr.add(q2)), w0);
                        a21 = vfmaq_f32(a21, vld1q_f32(inp_ptr.add(q2 + 4)), w1);
                        a22 = vfmaq_f32(a22, vld1q_f32(inp_ptr.add(q2 + 8)), w2);
                        a23 = vfmaq_f32(a23, vld1q_f32(inp_ptr.add(q2 + 12)), w3);
                        let q3 = q2 + col;
                        a30 = vfmaq_f32(a30, vld1q_f32(inp_ptr.add(q3)), w0);
                        a31 = vfmaq_f32(a31, vld1q_f32(inp_ptr.add(q3 + 4)), w1);
                        a32 = vfmaq_f32(a32, vld1q_f32(inp_ptr.add(q3 + 8)), w2);
                        a33 = vfmaq_f32(a33, vld1q_f32(inp_ptr.add(q3 + 12)), w3);
                    }
                }
                if do_relu {
                    a00 = vmaxq_f32(a00, zero);
                    a01 = vmaxq_f32(a01, zero);
                    a02 = vmaxq_f32(a02, zero);
                    a03 = vmaxq_f32(a03, zero);
                    a10 = vmaxq_f32(a10, zero);
                    a11 = vmaxq_f32(a11, zero);
                    a12 = vmaxq_f32(a12, zero);
                    a13 = vmaxq_f32(a13, zero);
                    a20 = vmaxq_f32(a20, zero);
                    a21 = vmaxq_f32(a21, zero);
                    a22 = vmaxq_f32(a22, zero);
                    a23 = vmaxq_f32(a23, zero);
                    a30 = vmaxq_f32(a30, zero);
                    a31 = vmaxq_f32(a31, zero);
                    a32 = vmaxq_f32(a32, zero);
                    a33 = vmaxq_f32(a33, zero);
                }
                let o0 = out_base + ch;
                vst1q_f32(out_ptr.add(o0), a00);
                vst1q_f32(out_ptr.add(o0 + 4), a01);
                vst1q_f32(out_ptr.add(o0 + 8), a02);
                vst1q_f32(out_ptr.add(o0 + 12), a03);
                let o1 = o0 + channels;
                vst1q_f32(out_ptr.add(o1), a10);
                vst1q_f32(out_ptr.add(o1 + 4), a11);
                vst1q_f32(out_ptr.add(o1 + 8), a12);
                vst1q_f32(out_ptr.add(o1 + 12), a13);
                let o2 = o1 + channels;
                vst1q_f32(out_ptr.add(o2), a20);
                vst1q_f32(out_ptr.add(o2 + 4), a21);
                vst1q_f32(out_ptr.add(o2 + 8), a22);
                vst1q_f32(out_ptr.add(o2 + 12), a23);
                let o3 = o2 + channels;
                vst1q_f32(out_ptr.add(o3), a30);
                vst1q_f32(out_ptr.add(o3 + 4), a31);
                vst1q_f32(out_ptr.add(o3 + 8), a32);
                vst1q_f32(out_ptr.add(o3 + 12), a33);
            }
            ch += 16;
        }
        tx += 4;
    }

    for out_x in 0..plan.out_w {
        let in_x0 = out_x * plan.stride_w;
        let out_base = out_x * channels;
        // Tiled columns already wrote their 16-aligned channels above.
        let mut ch = if out_x < tile_end { ch16_end } else { 0 };

        while ch + 16 <= channels {
            unsafe {
                let (mut a0, mut a1, mut a2, mut a3) = if let Some(bp) = bias_ptr {
                    (
                        vld1q_f32(bp.add(ch)),
                        vld1q_f32(bp.add(ch + 4)),
                        vld1q_f32(bp.add(ch + 8)),
                        vld1q_f32(bp.add(ch + 12)),
                    )
                } else {
                    (zero, zero, zero, zero)
                };
                for ky in 0..kh {
                    let in_y = in_y0 + ky;
                    let input_row_base = batch_input_base + (in_y * plan.in_w + in_x0) * channels;
                    let kernel_row_base = (ky * kw) * channels;
                    for kx in 0..kw {
                        let ib = input_row_base + kx * channels + ch;
                        let kb = kernel_row_base + kx * channels + ch;
                        a0 = vfmaq_f32(a0, vld1q_f32(inp_ptr.add(ib)), vld1q_f32(ker_ptr.add(kb)));
                        a1 = vfmaq_f32(
                            a1,
                            vld1q_f32(inp_ptr.add(ib + 4)),
                            vld1q_f32(ker_ptr.add(kb + 4)),
                        );
                        a2 = vfmaq_f32(
                            a2,
                            vld1q_f32(inp_ptr.add(ib + 8)),
                            vld1q_f32(ker_ptr.add(kb + 8)),
                        );
                        a3 = vfmaq_f32(
                            a3,
                            vld1q_f32(inp_ptr.add(ib + 12)),
                            vld1q_f32(ker_ptr.add(kb + 12)),
                        );
                    }
                }
                if do_relu {
                    a0 = vmaxq_f32(a0, zero);
                    a1 = vmaxq_f32(a1, zero);
                    a2 = vmaxq_f32(a2, zero);
                    a3 = vmaxq_f32(a3, zero);
                }
                let ob = out_base + ch;
                vst1q_f32(out_ptr.add(ob), a0);
                vst1q_f32(out_ptr.add(ob + 4), a1);
                vst1q_f32(out_ptr.add(ob + 8), a2);
                vst1q_f32(out_ptr.add(ob + 12), a3);
            }
            ch += 16;
        }
        while ch + 4 <= channels {
            unsafe {
                let mut acc = if let Some(bp) = bias_ptr {
                    vld1q_f32(bp.add(ch))
                } else {
                    vdupq_n_f32(0.0)
                };
                for ky in 0..kh {
                    let in_y = in_y0 + ky;
                    let input_row_base = batch_input_base + (in_y * plan.in_w + in_x0) * channels;
                    let kernel_row_base = (ky * kw) * channels;
                    for kx in 0..kw {
                        let in_off = input_row_base + kx * channels + ch;
                        let k_off = kernel_row_base + kx * channels + ch;
                        acc = vfmaq_f32(
                            acc,
                            vld1q_f32(inp_ptr.add(in_off)),
                            vld1q_f32(ker_ptr.add(k_off)),
                        );
                    }
                }
                if do_relu {
                    acc = vmaxq_f32(acc, zero);
                }
                vst1q_f32(out_ptr.add(out_base + ch), acc);
            }
            ch += 4;
        }
        while ch < channels {
            let mut acc = bias.map_or(0.0, |b| b[ch]);
            for ky in 0..kh {
                let in_y = in_y0 + ky;
                let input_row_base = batch_input_base + (in_y * plan.in_w + in_x0) * channels;
                let kernel_row_base = (ky * kw) * channels;
                for kx in 0..kw {
                    acc += input[input_row_base + kx * channels + ch]
                        * kernel[kernel_row_base + kx * channels + ch];
                }
            }
            if do_relu && acc < 0.0 {
                acc = 0.0;
            }
            out_row[out_base + ch] = acc;
            ch += 1;
        }
    }
    // SiLU cannot be fused cheaply in SIMD (needs exp); apply as post-pass.
    if matches!(activation, Activation::Silu) {
        silu_slice_inplace(out_row);
    }
}

/// AVX+FMA depthwise conv row for `depth_multiplier == 1`.
/// Uses `_mm256_fmadd_ps` for fused multiply-add (Haswell+ / all modern x86).
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx", enable = "fma")]
#[allow(unsafe_code)]
unsafe fn depthwise_conv2d_nhwc_row_avx_fma(
    input: &[f32],
    kernel: &[f32],
    bias: Option<&[f32]>,
    plan: DepthwiseConv2dPlan,
    row_idx: usize,
    out_row: &mut [f32],
    activation: Activation,
) {
    #[cfg(target_arch = "x86")]
    use core::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::*;

    let batch_idx = row_idx / plan.out_h;
    let out_y = row_idx % plan.out_h;
    let in_y0 = out_y * plan.stride_h;
    let batch_input_base = batch_idx * plan.in_h * plan.in_w * plan.channels;
    let channels = plan.channels;
    let kh = plan.kernel_h;
    let kw = plan.kernel_w;

    let inp_ptr = input.as_ptr();
    let ker_ptr = kernel.as_ptr();
    let out_ptr = out_row.as_mut_ptr();

    let bias_ptr = bias.map(|b| b.as_ptr());
    let do_relu = matches!(activation, Activation::Relu);
    let zero = _mm256_setzero_ps();

    for out_x in 0..plan.out_w {
        let in_x0 = out_x * plan.stride_w;
        let out_base = out_x * channels;
        let mut ch = 0;

        // 32-wide: 4 independent accumulators to saturate both FMA ports.
        while ch + 32 <= channels {
            unsafe {
                let (mut a0, mut a1, mut a2, mut a3) = if let Some(bp) = bias_ptr {
                    (
                        _mm256_loadu_ps(bp.add(ch)),
                        _mm256_loadu_ps(bp.add(ch + 8)),
                        _mm256_loadu_ps(bp.add(ch + 16)),
                        _mm256_loadu_ps(bp.add(ch + 24)),
                    )
                } else {
                    (zero, zero, zero, zero)
                };
                let in_row_stride = plan.in_w * channels;
                let kw_stride = channels;
                let mut ky_inp =
                    inp_ptr.add(batch_input_base + (in_y0 * plan.in_w + in_x0) * channels + ch);
                let mut ky_ker = ker_ptr.add(ch);
                for _ky in 0..kh {
                    let mut kx_inp = ky_inp;
                    let mut kx_ker = ky_ker;
                    for _kx in 0..kw {
                        a0 = _mm256_fmadd_ps(_mm256_loadu_ps(kx_inp), _mm256_loadu_ps(kx_ker), a0);
                        a1 = _mm256_fmadd_ps(
                            _mm256_loadu_ps(kx_inp.add(8)),
                            _mm256_loadu_ps(kx_ker.add(8)),
                            a1,
                        );
                        a2 = _mm256_fmadd_ps(
                            _mm256_loadu_ps(kx_inp.add(16)),
                            _mm256_loadu_ps(kx_ker.add(16)),
                            a2,
                        );
                        a3 = _mm256_fmadd_ps(
                            _mm256_loadu_ps(kx_inp.add(24)),
                            _mm256_loadu_ps(kx_ker.add(24)),
                            a3,
                        );
                        kx_inp = kx_inp.add(kw_stride);
                        kx_ker = kx_ker.add(kw_stride);
                    }
                    ky_inp = ky_inp.add(in_row_stride);
                    ky_ker = ky_ker.add(kw * kw_stride);
                }
                if do_relu {
                    a0 = _mm256_max_ps(a0, zero);
                    a1 = _mm256_max_ps(a1, zero);
                    a2 = _mm256_max_ps(a2, zero);
                    a3 = _mm256_max_ps(a3, zero);
                }
                let ob = out_base + ch;
                _mm256_storeu_ps(out_ptr.add(ob), a0);
                _mm256_storeu_ps(out_ptr.add(ob + 8), a1);
                _mm256_storeu_ps(out_ptr.add(ob + 16), a2);
                _mm256_storeu_ps(out_ptr.add(ob + 24), a3);
            }
            ch += 32;
        }
        // 16-wide: 2 accumulators.
        while ch + 16 <= channels {
            unsafe {
                let (mut a0, mut a1) = if let Some(bp) = bias_ptr {
                    (_mm256_loadu_ps(bp.add(ch)), _mm256_loadu_ps(bp.add(ch + 8)))
                } else {
                    (zero, zero)
                };
                let in_row_stride = plan.in_w * channels;
                let kw_stride = channels;
                let mut ky_inp =
                    inp_ptr.add(batch_input_base + (in_y0 * plan.in_w + in_x0) * channels + ch);
                let mut ky_ker = ker_ptr.add(ch);
                for _ky in 0..kh {
                    let mut kx_inp = ky_inp;
                    let mut kx_ker = ky_ker;
                    for _kx in 0..kw {
                        a0 = _mm256_fmadd_ps(_mm256_loadu_ps(kx_inp), _mm256_loadu_ps(kx_ker), a0);
                        a1 = _mm256_fmadd_ps(
                            _mm256_loadu_ps(kx_inp.add(8)),
                            _mm256_loadu_ps(kx_ker.add(8)),
                            a1,
                        );
                        kx_inp = kx_inp.add(kw_stride);
                        kx_ker = kx_ker.add(kw_stride);
                    }
                    ky_inp = ky_inp.add(in_row_stride);
                    ky_ker = ky_ker.add(kw * kw_stride);
                }
                if do_relu {
                    a0 = _mm256_max_ps(a0, zero);
                    a1 = _mm256_max_ps(a1, zero);
                }
                _mm256_storeu_ps(out_ptr.add(out_base + ch), a0);
                _mm256_storeu_ps(out_ptr.add(out_base + ch + 8), a1);
            }
            ch += 16;
        }
        // 8-wide: single accumulator.
        while ch + 8 <= channels {
            unsafe {
                let mut acc = if let Some(bp) = bias_ptr {
                    _mm256_loadu_ps(bp.add(ch))
                } else {
                    _mm256_setzero_ps()
                };
                let in_row_stride = plan.in_w * channels;
                let kw_stride = channels;
                let mut ky_inp =
                    inp_ptr.add(batch_input_base + (in_y0 * plan.in_w + in_x0) * channels + ch);
                let mut ky_ker = ker_ptr.add(ch);
                for _ky in 0..kh {
                    let mut kx_inp = ky_inp;
                    let mut kx_ker = ky_ker;
                    for _kx in 0..kw {
                        acc =
                            _mm256_fmadd_ps(_mm256_loadu_ps(kx_inp), _mm256_loadu_ps(kx_ker), acc);
                        kx_inp = kx_inp.add(kw_stride);
                        kx_ker = kx_ker.add(kw_stride);
                    }
                    ky_inp = ky_inp.add(in_row_stride);
                    ky_ker = ky_ker.add(kw * kw_stride);
                }
                if do_relu {
                    acc = _mm256_max_ps(acc, zero);
                }
                _mm256_storeu_ps(out_ptr.add(out_base + ch), acc);
            }
            ch += 8;
        }
        // Scalar tail.
        while ch < channels {
            let mut acc = bias.map_or(0.0, |b| b[ch]);
            for ky in 0..kh {
                let in_y = in_y0 + ky;
                let input_row_base = batch_input_base + (in_y * plan.in_w + in_x0) * channels;
                let kernel_row_base = (ky * kw) * channels;
                for kx in 0..kw {
                    acc += input[input_row_base + kx * channels + ch]
                        * kernel[kernel_row_base + kx * channels + ch];
                }
            }
            if do_relu && acc < 0.0 {
                acc = 0.0;
            }
            out_row[out_base + ch] = acc;
            ch += 1;
        }
    }
    // SiLU cannot be fused cheaply in SIMD (needs exp); apply as post-pass.
    if matches!(activation, Activation::Silu) {
        silu_slice_inplace(out_row);
    }
}

/// AVX-512 depthwise conv row for `depth_multiplier == 1`.
/// Vectorizes across the channel dimension (16 channels per `__m512`).
/// Wider tiles than AVX+FMA: 128-wide (8 ZMM), 64-wide (4 ZMM),
/// 32-wide (2 ZMM), 16-wide (1 ZMM), then scalar tail.
///
/// Zen 4 AVX-512 is double-pumped (2× 256-bit FMA internally) so peak
/// FLOPS equal to AVX+FMA, but uop count halves — same work in fewer
/// front-end slots. DW has no broadcast bottleneck (each lane is an
/// independent channel), so intrinsics saturate the pipeline without
/// needing inline asm.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
unsafe fn depthwise_conv2d_nhwc_row_avx512(
    input: &[f32],
    kernel: &[f32],
    bias: Option<&[f32]>,
    plan: DepthwiseConv2dPlan,
    row_idx: usize,
    out_row: &mut [f32],
    activation: Activation,
) {
    #[cfg(target_arch = "x86")]
    use core::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::*;

    let batch_idx = row_idx / plan.out_h;
    let out_y = row_idx % plan.out_h;
    let in_y0 = out_y * plan.stride_h;
    let batch_input_base = batch_idx * plan.in_h * plan.in_w * plan.channels;
    let channels = plan.channels;
    let kh = plan.kernel_h;
    let kw = plan.kernel_w;

    let inp_ptr = input.as_ptr();
    let ker_ptr = kernel.as_ptr();
    let out_ptr = out_row.as_mut_ptr();

    let bias_ptr = bias.map(|b| b.as_ptr());
    let do_relu = matches!(activation, Activation::Relu);
    let zero = _mm512_setzero_ps();

    for out_x in 0..plan.out_w {
        let in_x0 = out_x * plan.stride_w;
        let out_base = out_x * channels;
        let mut ch = 0;

        // 128-wide (8 ZMM) — saturate deep channel counts (tracker c=320,672).
        while ch + 128 <= channels {
            let (mut a0, mut a1, mut a2, mut a3, mut a4, mut a5, mut a6, mut a7) =
                if let Some(bp) = bias_ptr {
                    (
                        _mm512_loadu_ps(bp.add(ch)),
                        _mm512_loadu_ps(bp.add(ch + 16)),
                        _mm512_loadu_ps(bp.add(ch + 32)),
                        _mm512_loadu_ps(bp.add(ch + 48)),
                        _mm512_loadu_ps(bp.add(ch + 64)),
                        _mm512_loadu_ps(bp.add(ch + 80)),
                        _mm512_loadu_ps(bp.add(ch + 96)),
                        _mm512_loadu_ps(bp.add(ch + 112)),
                    )
                } else {
                    (zero, zero, zero, zero, zero, zero, zero, zero)
                };
            let in_row_stride = plan.in_w * channels;
            let kw_stride = channels;
            let mut ky_inp =
                inp_ptr.add(batch_input_base + (in_y0 * plan.in_w + in_x0) * channels + ch);
            let mut ky_ker = ker_ptr.add(ch);
            for _ky in 0..kh {
                let mut kx_inp = ky_inp;
                let mut kx_ker = ky_ker;
                for _kx in 0..kw {
                    a0 = _mm512_fmadd_ps(_mm512_loadu_ps(kx_inp), _mm512_loadu_ps(kx_ker), a0);
                    a1 = _mm512_fmadd_ps(
                        _mm512_loadu_ps(kx_inp.add(16)),
                        _mm512_loadu_ps(kx_ker.add(16)),
                        a1,
                    );
                    a2 = _mm512_fmadd_ps(
                        _mm512_loadu_ps(kx_inp.add(32)),
                        _mm512_loadu_ps(kx_ker.add(32)),
                        a2,
                    );
                    a3 = _mm512_fmadd_ps(
                        _mm512_loadu_ps(kx_inp.add(48)),
                        _mm512_loadu_ps(kx_ker.add(48)),
                        a3,
                    );
                    a4 = _mm512_fmadd_ps(
                        _mm512_loadu_ps(kx_inp.add(64)),
                        _mm512_loadu_ps(kx_ker.add(64)),
                        a4,
                    );
                    a5 = _mm512_fmadd_ps(
                        _mm512_loadu_ps(kx_inp.add(80)),
                        _mm512_loadu_ps(kx_ker.add(80)),
                        a5,
                    );
                    a6 = _mm512_fmadd_ps(
                        _mm512_loadu_ps(kx_inp.add(96)),
                        _mm512_loadu_ps(kx_ker.add(96)),
                        a6,
                    );
                    a7 = _mm512_fmadd_ps(
                        _mm512_loadu_ps(kx_inp.add(112)),
                        _mm512_loadu_ps(kx_ker.add(112)),
                        a7,
                    );
                    kx_inp = kx_inp.add(kw_stride);
                    kx_ker = kx_ker.add(kw_stride);
                }
                ky_inp = ky_inp.add(in_row_stride);
                ky_ker = ky_ker.add(kw * kw_stride);
            }
            if do_relu {
                a0 = _mm512_max_ps(a0, zero);
                a1 = _mm512_max_ps(a1, zero);
                a2 = _mm512_max_ps(a2, zero);
                a3 = _mm512_max_ps(a3, zero);
                a4 = _mm512_max_ps(a4, zero);
                a5 = _mm512_max_ps(a5, zero);
                a6 = _mm512_max_ps(a6, zero);
                a7 = _mm512_max_ps(a7, zero);
            }
            let ob = out_base + ch;
            _mm512_storeu_ps(out_ptr.add(ob), a0);
            _mm512_storeu_ps(out_ptr.add(ob + 16), a1);
            _mm512_storeu_ps(out_ptr.add(ob + 32), a2);
            _mm512_storeu_ps(out_ptr.add(ob + 48), a3);
            _mm512_storeu_ps(out_ptr.add(ob + 64), a4);
            _mm512_storeu_ps(out_ptr.add(ob + 80), a5);
            _mm512_storeu_ps(out_ptr.add(ob + 96), a6);
            _mm512_storeu_ps(out_ptr.add(ob + 112), a7);
            ch += 128;
        }

        // 64-wide (4 ZMM).
        while ch + 64 <= channels {
            let (mut a0, mut a1, mut a2, mut a3) = if let Some(bp) = bias_ptr {
                (
                    _mm512_loadu_ps(bp.add(ch)),
                    _mm512_loadu_ps(bp.add(ch + 16)),
                    _mm512_loadu_ps(bp.add(ch + 32)),
                    _mm512_loadu_ps(bp.add(ch + 48)),
                )
            } else {
                (zero, zero, zero, zero)
            };
            let in_row_stride = plan.in_w * channels;
            let kw_stride = channels;
            let mut ky_inp =
                inp_ptr.add(batch_input_base + (in_y0 * plan.in_w + in_x0) * channels + ch);
            let mut ky_ker = ker_ptr.add(ch);
            for _ky in 0..kh {
                let mut kx_inp = ky_inp;
                let mut kx_ker = ky_ker;
                for _kx in 0..kw {
                    a0 = _mm512_fmadd_ps(_mm512_loadu_ps(kx_inp), _mm512_loadu_ps(kx_ker), a0);
                    a1 = _mm512_fmadd_ps(
                        _mm512_loadu_ps(kx_inp.add(16)),
                        _mm512_loadu_ps(kx_ker.add(16)),
                        a1,
                    );
                    a2 = _mm512_fmadd_ps(
                        _mm512_loadu_ps(kx_inp.add(32)),
                        _mm512_loadu_ps(kx_ker.add(32)),
                        a2,
                    );
                    a3 = _mm512_fmadd_ps(
                        _mm512_loadu_ps(kx_inp.add(48)),
                        _mm512_loadu_ps(kx_ker.add(48)),
                        a3,
                    );
                    kx_inp = kx_inp.add(kw_stride);
                    kx_ker = kx_ker.add(kw_stride);
                }
                ky_inp = ky_inp.add(in_row_stride);
                ky_ker = ky_ker.add(kw * kw_stride);
            }
            if do_relu {
                a0 = _mm512_max_ps(a0, zero);
                a1 = _mm512_max_ps(a1, zero);
                a2 = _mm512_max_ps(a2, zero);
                a3 = _mm512_max_ps(a3, zero);
            }
            let ob = out_base + ch;
            _mm512_storeu_ps(out_ptr.add(ob), a0);
            _mm512_storeu_ps(out_ptr.add(ob + 16), a1);
            _mm512_storeu_ps(out_ptr.add(ob + 32), a2);
            _mm512_storeu_ps(out_ptr.add(ob + 48), a3);
            ch += 64;
        }

        // 32-wide (2 ZMM).
        while ch + 32 <= channels {
            let (mut a0, mut a1) = if let Some(bp) = bias_ptr {
                (
                    _mm512_loadu_ps(bp.add(ch)),
                    _mm512_loadu_ps(bp.add(ch + 16)),
                )
            } else {
                (zero, zero)
            };
            let in_row_stride = plan.in_w * channels;
            let kw_stride = channels;
            let mut ky_inp =
                inp_ptr.add(batch_input_base + (in_y0 * plan.in_w + in_x0) * channels + ch);
            let mut ky_ker = ker_ptr.add(ch);
            for _ky in 0..kh {
                let mut kx_inp = ky_inp;
                let mut kx_ker = ky_ker;
                for _kx in 0..kw {
                    a0 = _mm512_fmadd_ps(_mm512_loadu_ps(kx_inp), _mm512_loadu_ps(kx_ker), a0);
                    a1 = _mm512_fmadd_ps(
                        _mm512_loadu_ps(kx_inp.add(16)),
                        _mm512_loadu_ps(kx_ker.add(16)),
                        a1,
                    );
                    kx_inp = kx_inp.add(kw_stride);
                    kx_ker = kx_ker.add(kw_stride);
                }
                ky_inp = ky_inp.add(in_row_stride);
                ky_ker = ky_ker.add(kw * kw_stride);
            }
            if do_relu {
                a0 = _mm512_max_ps(a0, zero);
                a1 = _mm512_max_ps(a1, zero);
            }
            let ob = out_base + ch;
            _mm512_storeu_ps(out_ptr.add(ob), a0);
            _mm512_storeu_ps(out_ptr.add(ob + 16), a1);
            ch += 32;
        }

        // 16-wide (1 ZMM).
        while ch + 16 <= channels {
            let mut acc = if let Some(bp) = bias_ptr {
                _mm512_loadu_ps(bp.add(ch))
            } else {
                zero
            };
            let in_row_stride = plan.in_w * channels;
            let kw_stride = channels;
            let mut ky_inp =
                inp_ptr.add(batch_input_base + (in_y0 * plan.in_w + in_x0) * channels + ch);
            let mut ky_ker = ker_ptr.add(ch);
            for _ky in 0..kh {
                let mut kx_inp = ky_inp;
                let mut kx_ker = ky_ker;
                for _kx in 0..kw {
                    acc = _mm512_fmadd_ps(_mm512_loadu_ps(kx_inp), _mm512_loadu_ps(kx_ker), acc);
                    kx_inp = kx_inp.add(kw_stride);
                    kx_ker = kx_ker.add(kw_stride);
                }
                ky_inp = ky_inp.add(in_row_stride);
                ky_ker = ky_ker.add(kw * kw_stride);
            }
            if do_relu {
                acc = _mm512_max_ps(acc, zero);
            }
            _mm512_storeu_ps(out_ptr.add(out_base + ch), acc);
            ch += 16;
        }

        // 8-wide tail via AVX (when channels % 16 != 0 but %8==0).
        {
            use core::arch::x86_64::{
                _mm256_fmadd_ps, _mm256_loadu_ps, _mm256_max_ps, _mm256_setzero_ps,
                _mm256_storeu_ps,
            };
            let zy = _mm256_setzero_ps();
            while ch + 8 <= channels {
                let mut acc = if let Some(bp) = bias_ptr {
                    _mm256_loadu_ps(bp.add(ch))
                } else {
                    zy
                };
                let in_row_stride = plan.in_w * channels;
                let kw_stride = channels;
                let mut ky_inp =
                    inp_ptr.add(batch_input_base + (in_y0 * plan.in_w + in_x0) * channels + ch);
                let mut ky_ker = ker_ptr.add(ch);
                for _ky in 0..kh {
                    let mut kx_inp = ky_inp;
                    let mut kx_ker = ky_ker;
                    for _kx in 0..kw {
                        acc =
                            _mm256_fmadd_ps(_mm256_loadu_ps(kx_inp), _mm256_loadu_ps(kx_ker), acc);
                        kx_inp = kx_inp.add(kw_stride);
                        kx_ker = kx_ker.add(kw_stride);
                    }
                    ky_inp = ky_inp.add(in_row_stride);
                    ky_ker = ky_ker.add(kw * kw_stride);
                }
                if do_relu {
                    acc = _mm256_max_ps(acc, zy);
                }
                _mm256_storeu_ps(out_ptr.add(out_base + ch), acc);
                ch += 8;
            }
        }

        // Scalar tail.
        while ch < channels {
            let mut acc = bias.map_or(0.0, |b| b[ch]);
            for ky in 0..kh {
                let in_y = in_y0 + ky;
                let input_row_base = batch_input_base + (in_y * plan.in_w + in_x0) * channels;
                let kernel_row_base = (ky * kw) * channels;
                for kx in 0..kw {
                    acc += input[input_row_base + kx * channels + ch]
                        * kernel[kernel_row_base + kx * channels + ch];
                }
            }
            if do_relu && acc < 0.0 {
                acc = 0.0;
            }
            out_row[out_base + ch] = acc;
            ch += 1;
        }
    }
    // SiLU cannot be fused cheaply in SIMD (needs exp); apply as post-pass.
    if matches!(activation, Activation::Silu) {
        silu_slice_inplace(out_row);
    }
}

/// Specialised AVX-512 depthwise 3×3 NHWC kernel for `channels == 16`,
/// `depth_multiplier == 1`. Pre-loads the 9 weight ZMMs once per output
/// row, keeps a single ZMM accumulator across the 9 taps, stores once
/// per output pixel. Compared to the generic per-tap kernel this drops
/// from ~45 mem ops per pixel to ~11.
///
/// Per-row workload (out_w=128, c=16, stride=1, pad=1):
///   * 9 weight ZMM loads (cache-line-hot after first row).
///   * Interior pixels (out_w - 2): 9 input loads + 9 FMAs + 1 store.
///   * Border pixels (2): conditional FMAs based on tap in-bounds checks.
///
/// Wired from `depthwise_conv2d_nhwc_padded_with_activation_with_config_and_pool`
/// when the gate fires. Output buffer is allocated by the caller.
#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code, clippy::too_many_arguments)]
pub(super) fn depthwise3x3_nhwc_c16_avx512(
    input: &[f32],
    kernel: &[f32],
    bias: Option<&[f32]>,
    batch: usize,
    in_h: usize,
    in_w: usize,
    pad_top: usize,
    pad_left: usize,
    _pad_bottom: usize,
    _pad_right: usize,
    stride_h: usize,
    stride_w: usize,
    out_h: usize,
    out_w: usize,
    output: &mut [f32],
    activation: Activation,
    thread_pool: Option<&ThreadPool>,
) {
    let c: usize = 16;
    let out_row_len = out_w * c;
    let in_row_len = in_w * c;
    let batch_in_len = in_h * in_row_len;
    let batch_out_len = out_h * out_row_len;
    let do_relu = matches!(activation, Activation::Relu);

    // Per-output-row closure. Captures the small kernel/bias slices and
    // walks one output row. Border-aware for both ky (top/bottom rows)
    // and kx (left/right columns).
    let process_row = |batch_idx: usize, out_y: usize, out_row: &mut [f32]| {
        // SAFETY: avx512f detected by caller; all addresses computed within
        // the validated batch/in_h/in_w/out_w bounds. Each output element
        // is written exactly once.
        unsafe {
            depthwise3x3_c16_avx512_one_row(
                input,
                kernel,
                bias,
                batch_idx,
                in_h,
                in_w,
                in_row_len,
                batch_in_len,
                pad_top,
                pad_left,
                stride_h,
                stride_w,
                out_y,
                out_w,
                out_row,
                do_relu,
            )
        };
    };

    // Parallelisation: walk `batch × out_h` rows. Each thread processes a
    // contiguous chunk of output rows. Mirrors the generic DW kernel's
    // rayon dispatch pattern.
    let total_rows = batch * out_h;
    let parallel_ok = total_rows >= 4 && !cfg!(miri);
    if !parallel_ok {
        for batch_idx in 0..batch {
            let out_batch = &mut output[batch_idx * batch_out_len..(batch_idx + 1) * batch_out_len];
            for out_y in 0..out_h {
                let row = &mut out_batch[out_y * out_row_len..(out_y + 1) * out_row_len];
                process_row(batch_idx, out_y, row);
            }
        }
        return;
    }

    // Per-thread chunk: split total_rows by current pool size. Mirrors
    // `par_chunks_mut_dispatch` but lets us share the kernel/bias slices
    // by reference (the closure captures them).
    let nthreads = thread_pool
        .map(|p| p.current_num_threads().max(1))
        .unwrap_or_else(|| rayon::current_num_threads().max(1));
    let rows_per_chunk = total_rows.div_ceil(nthreads).max(1);
    let bytes_per_chunk = rows_per_chunk * out_row_len;
    let run = |chunk_idx: usize, chunk: &mut [f32]| {
        let row_start = chunk_idx * rows_per_chunk;
        let row_end = (row_start + rows_per_chunk).min(total_rows);
        for (off, global_row) in (row_start..row_end).enumerate() {
            let batch_idx = global_row / out_h;
            let out_y = global_row % out_h;
            let row = &mut chunk[off * out_row_len..(off + 1) * out_row_len];
            process_row(batch_idx, out_y, row);
        }
    };
    if let Some(pool) = thread_pool {
        pool.install(|| {
            use rayon::prelude::*;
            output
                .par_chunks_mut(bytes_per_chunk)
                .enumerate()
                .for_each(|(idx, chunk)| run(idx, chunk));
        });
    } else {
        use rayon::prelude::*;
        output
            .par_chunks_mut(bytes_per_chunk)
            .enumerate()
            .for_each(|(idx, chunk)| run(idx, chunk));
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn, clippy::too_many_arguments)]
unsafe fn depthwise3x3_c16_avx512_one_row(
    input: &[f32],
    kernel: &[f32],
    bias: Option<&[f32]>,
    batch_idx: usize,
    in_h: usize,
    in_w: usize,
    in_row_len: usize,
    batch_in_len: usize,
    pad_top: usize,
    pad_left: usize,
    stride_h: usize,
    stride_w: usize,
    out_y: usize,
    out_w: usize,
    out_row: &mut [f32],
    do_relu: bool,
) {
    use core::arch::x86_64::*;

    // Pre-load 9 weight ZMMs into stable variables. LLVM with target_feature
    // keeps these in registers across the inner loop; the inner FMA chain
    // only touches input + accumulator memory.
    let kp = kernel.as_ptr();
    let w00 = _mm512_loadu_ps(kp);
    let w01 = _mm512_loadu_ps(kp.add(16));
    let w02 = _mm512_loadu_ps(kp.add(32));
    let w10 = _mm512_loadu_ps(kp.add(48));
    let w11 = _mm512_loadu_ps(kp.add(64));
    let w12 = _mm512_loadu_ps(kp.add(80));
    let w20 = _mm512_loadu_ps(kp.add(96));
    let w21 = _mm512_loadu_ps(kp.add(112));
    let w22 = _mm512_loadu_ps(kp.add(128));

    let bias_vec = match bias {
        Some(b) => _mm512_loadu_ps(b.as_ptr()),
        None => _mm512_setzero_ps(),
    };
    let zero = _mm512_setzero_ps();

    let in_y_base = (out_y * stride_h) as isize - pad_top as isize;
    let in_y0 = in_y_base; // ky = 0
    let in_y1 = in_y_base + 1; // ky = 1
    let in_y2 = in_y_base + 2; // ky = 2
    let row0_ok = in_y0 >= 0 && (in_y0 as usize) < in_h;
    let row1_ok = in_y1 >= 0 && (in_y1 as usize) < in_h;
    let row2_ok = in_y2 >= 0 && (in_y2 as usize) < in_h;

    let batch_base = batch_idx * batch_in_len;
    // Row pointers for ky=0/1/2 (only valid when row*_ok is true).
    let in_ptr0 = if row0_ok {
        input
            .as_ptr()
            .add(batch_base + (in_y0 as usize) * in_row_len)
    } else {
        std::ptr::null()
    };
    let in_ptr1 = if row1_ok {
        input
            .as_ptr()
            .add(batch_base + (in_y1 as usize) * in_row_len)
    } else {
        std::ptr::null()
    };
    let in_ptr2 = if row2_ok {
        input
            .as_ptr()
            .add(batch_base + (in_y2 as usize) * in_row_len)
    } else {
        std::ptr::null()
    };

    let out_ptr = out_row.as_mut_ptr();

    for out_x in 0..out_w {
        let in_x_base = (out_x * stride_w) as isize - pad_left as isize;
        let in_x0 = in_x_base;
        let in_x1 = in_x_base + 1;
        let in_x2 = in_x_base + 2;
        let col0_ok = in_x0 >= 0 && (in_x0 as usize) < in_w;
        let col1_ok = in_x1 >= 0 && (in_x1 as usize) < in_w;
        let col2_ok = in_x2 >= 0 && (in_x2 as usize) < in_w;

        let mut acc = bias_vec;

        // ky = 0
        if row0_ok {
            if col0_ok {
                let p = in_ptr0.add((in_x0 as usize) * 16);
                acc = _mm512_fmadd_ps(_mm512_loadu_ps(p), w00, acc);
            }
            if col1_ok {
                let p = in_ptr0.add((in_x1 as usize) * 16);
                acc = _mm512_fmadd_ps(_mm512_loadu_ps(p), w01, acc);
            }
            if col2_ok {
                let p = in_ptr0.add((in_x2 as usize) * 16);
                acc = _mm512_fmadd_ps(_mm512_loadu_ps(p), w02, acc);
            }
        }
        // ky = 1
        if row1_ok {
            if col0_ok {
                let p = in_ptr1.add((in_x0 as usize) * 16);
                acc = _mm512_fmadd_ps(_mm512_loadu_ps(p), w10, acc);
            }
            if col1_ok {
                let p = in_ptr1.add((in_x1 as usize) * 16);
                acc = _mm512_fmadd_ps(_mm512_loadu_ps(p), w11, acc);
            }
            if col2_ok {
                let p = in_ptr1.add((in_x2 as usize) * 16);
                acc = _mm512_fmadd_ps(_mm512_loadu_ps(p), w12, acc);
            }
        }
        // ky = 2
        if row2_ok {
            if col0_ok {
                let p = in_ptr2.add((in_x0 as usize) * 16);
                acc = _mm512_fmadd_ps(_mm512_loadu_ps(p), w20, acc);
            }
            if col1_ok {
                let p = in_ptr2.add((in_x1 as usize) * 16);
                acc = _mm512_fmadd_ps(_mm512_loadu_ps(p), w21, acc);
            }
            if col2_ok {
                let p = in_ptr2.add((in_x2 as usize) * 16);
                acc = _mm512_fmadd_ps(_mm512_loadu_ps(p), w22, acc);
            }
        }

        if do_relu {
            acc = _mm512_max_ps(acc, zero);
        }
        _mm512_storeu_ps(out_ptr.add(out_x * 16), acc);
    }
}

/// AVX-accelerated depthwise conv row for `depth_multiplier == 1` (no FMA fallback).
/// Vectorizes across the channel dimension (8 channels per `__m256`).
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx")]
#[allow(unsafe_code)]
unsafe fn depthwise_conv2d_nhwc_row_avx(
    input: &[f32],
    kernel: &[f32],
    bias: Option<&[f32]>,
    plan: DepthwiseConv2dPlan,
    row_idx: usize,
    out_row: &mut [f32],
    activation: Activation,
) {
    #[cfg(target_arch = "x86")]
    use core::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::*;

    let batch_idx = row_idx / plan.out_h;
    let out_y = row_idx % plan.out_h;
    let in_y0 = out_y * plan.stride_h;
    let batch_input_base = batch_idx * plan.in_h * plan.in_w * plan.channels;
    let channels = plan.channels;
    let kh = plan.kernel_h;
    let kw = plan.kernel_w;

    let inp_ptr = input.as_ptr();
    let ker_ptr = kernel.as_ptr();
    let out_ptr = out_row.as_mut_ptr();

    let bias_ptr = bias.map(|b| b.as_ptr());
    let do_relu = matches!(activation, Activation::Relu);
    let zero = _mm256_setzero_ps();

    for out_x in 0..plan.out_w {
        let in_x0 = out_x * plan.stride_w;
        let out_base = out_x * channels;
        let mut ch = 0;

        while ch + 32 <= channels {
            unsafe {
                let (mut a0, mut a1, mut a2, mut a3) = if let Some(bp) = bias_ptr {
                    (
                        _mm256_loadu_ps(bp.add(ch)),
                        _mm256_loadu_ps(bp.add(ch + 8)),
                        _mm256_loadu_ps(bp.add(ch + 16)),
                        _mm256_loadu_ps(bp.add(ch + 24)),
                    )
                } else {
                    (zero, zero, zero, zero)
                };
                for ky in 0..kh {
                    let in_y = in_y0 + ky;
                    let input_row_base = batch_input_base + (in_y * plan.in_w + in_x0) * channels;
                    let kernel_row_base = (ky * kw) * channels;
                    for kx in 0..kw {
                        let ib = input_row_base + kx * channels + ch;
                        let kb = kernel_row_base + kx * channels + ch;
                        a0 = _mm256_add_ps(
                            a0,
                            _mm256_mul_ps(
                                _mm256_loadu_ps(inp_ptr.add(ib)),
                                _mm256_loadu_ps(ker_ptr.add(kb)),
                            ),
                        );
                        a1 = _mm256_add_ps(
                            a1,
                            _mm256_mul_ps(
                                _mm256_loadu_ps(inp_ptr.add(ib + 8)),
                                _mm256_loadu_ps(ker_ptr.add(kb + 8)),
                            ),
                        );
                        a2 = _mm256_add_ps(
                            a2,
                            _mm256_mul_ps(
                                _mm256_loadu_ps(inp_ptr.add(ib + 16)),
                                _mm256_loadu_ps(ker_ptr.add(kb + 16)),
                            ),
                        );
                        a3 = _mm256_add_ps(
                            a3,
                            _mm256_mul_ps(
                                _mm256_loadu_ps(inp_ptr.add(ib + 24)),
                                _mm256_loadu_ps(ker_ptr.add(kb + 24)),
                            ),
                        );
                    }
                }
                if do_relu {
                    a0 = _mm256_max_ps(a0, zero);
                    a1 = _mm256_max_ps(a1, zero);
                    a2 = _mm256_max_ps(a2, zero);
                    a3 = _mm256_max_ps(a3, zero);
                }
                let ob = out_base + ch;
                _mm256_storeu_ps(out_ptr.add(ob), a0);
                _mm256_storeu_ps(out_ptr.add(ob + 8), a1);
                _mm256_storeu_ps(out_ptr.add(ob + 16), a2);
                _mm256_storeu_ps(out_ptr.add(ob + 24), a3);
            }
            ch += 32;
        }
        while ch + 16 <= channels {
            unsafe {
                let (mut a0, mut a1) = if let Some(bp) = bias_ptr {
                    (_mm256_loadu_ps(bp.add(ch)), _mm256_loadu_ps(bp.add(ch + 8)))
                } else {
                    (zero, zero)
                };
                for ky in 0..kh {
                    let in_y = in_y0 + ky;
                    let input_row_base = batch_input_base + (in_y * plan.in_w + in_x0) * channels;
                    let kernel_row_base = (ky * kw) * channels;
                    for kx in 0..kw {
                        let ib = input_row_base + kx * channels + ch;
                        let kb = kernel_row_base + kx * channels + ch;
                        a0 = _mm256_add_ps(
                            a0,
                            _mm256_mul_ps(
                                _mm256_loadu_ps(inp_ptr.add(ib)),
                                _mm256_loadu_ps(ker_ptr.add(kb)),
                            ),
                        );
                        a1 = _mm256_add_ps(
                            a1,
                            _mm256_mul_ps(
                                _mm256_loadu_ps(inp_ptr.add(ib + 8)),
                                _mm256_loadu_ps(ker_ptr.add(kb + 8)),
                            ),
                        );
                    }
                }
                if do_relu {
                    a0 = _mm256_max_ps(a0, zero);
                    a1 = _mm256_max_ps(a1, zero);
                }
                _mm256_storeu_ps(out_ptr.add(out_base + ch), a0);
                _mm256_storeu_ps(out_ptr.add(out_base + ch + 8), a1);
            }
            ch += 16;
        }
        while ch + 8 <= channels {
            unsafe {
                let mut acc = if let Some(bp) = bias_ptr {
                    _mm256_loadu_ps(bp.add(ch))
                } else {
                    _mm256_setzero_ps()
                };
                for ky in 0..kh {
                    let in_y = in_y0 + ky;
                    let input_row_base = batch_input_base + (in_y * plan.in_w + in_x0) * channels;
                    let kernel_row_base = (ky * kw) * channels;
                    for kx in 0..kw {
                        let in_off = input_row_base + kx * channels + ch;
                        let k_off = kernel_row_base + kx * channels + ch;
                        acc = _mm256_add_ps(
                            acc,
                            _mm256_mul_ps(
                                _mm256_loadu_ps(inp_ptr.add(in_off)),
                                _mm256_loadu_ps(ker_ptr.add(k_off)),
                            ),
                        );
                    }
                }
                if do_relu {
                    acc = _mm256_max_ps(acc, zero);
                }
                _mm256_storeu_ps(out_ptr.add(out_base + ch), acc);
            }
            ch += 8;
        }
        while ch < channels {
            let mut acc = bias.map_or(0.0, |b| b[ch]);
            for ky in 0..kh {
                let in_y = in_y0 + ky;
                let input_row_base = batch_input_base + (in_y * plan.in_w + in_x0) * channels;
                let kernel_row_base = (ky * kw) * channels;
                for kx in 0..kw {
                    acc += input[input_row_base + kx * channels + ch]
                        * kernel[kernel_row_base + kx * channels + ch];
                }
            }
            if do_relu && acc < 0.0 {
                acc = 0.0;
            }
            out_row[out_base + ch] = acc;
            ch += 1;
        }
    }
    // SiLU cannot be fused cheaply in SIMD (needs exp); apply as post-pass.
    if matches!(activation, Activation::Silu) {
        silu_slice_inplace(out_row);
    }
}

/// SSE-accelerated depthwise conv row for `depth_multiplier == 1`.
/// Vectorizes across the channel dimension (4 channels per `__m128`).
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse")]
#[allow(unsafe_code)]
unsafe fn depthwise_conv2d_nhwc_row_sse(
    input: &[f32],
    kernel: &[f32],
    bias: Option<&[f32]>,
    plan: DepthwiseConv2dPlan,
    row_idx: usize,
    out_row: &mut [f32],
    activation: Activation,
) {
    #[cfg(target_arch = "x86")]
    use core::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::*;

    let batch_idx = row_idx / plan.out_h;
    let out_y = row_idx % plan.out_h;
    let in_y0 = out_y * plan.stride_h;
    let batch_input_base = batch_idx * plan.in_h * plan.in_w * plan.channels;
    let channels = plan.channels;
    let simd_end = channels & !3; // round down to multiple of 4
    let kh = plan.kernel_h;
    let kw = plan.kernel_w;

    let inp_ptr = input.as_ptr();
    let ker_ptr = kernel.as_ptr();
    let out_ptr = out_row.as_mut_ptr();

    let do_relu = matches!(activation, Activation::Relu);
    let zero = _mm_setzero_ps();

    for out_x in 0..plan.out_w {
        let in_x0 = out_x * plan.stride_w;
        let out_base = out_x * channels;

        // Process 4 channels at a time with accumulator in register.
        let mut ch = 0;
        while ch + 4 <= simd_end {
            // SAFETY: ch + 4 <= channels, all offsets bounded by plan dims.
            unsafe {
                let mut acc = if let Some(b) = bias {
                    _mm_loadu_ps(b.as_ptr().add(ch))
                } else {
                    zero
                };

                for ky in 0..kh {
                    let in_y = in_y0 + ky;
                    let input_row_base = batch_input_base + (in_y * plan.in_w + in_x0) * channels;
                    let kernel_row_base = (ky * kw) * channels;

                    for kx in 0..kw {
                        let in_off = input_row_base + kx * channels + ch;
                        let k_off = kernel_row_base + kx * channels + ch;
                        let inp = _mm_loadu_ps(inp_ptr.add(in_off));
                        let ker = _mm_loadu_ps(ker_ptr.add(k_off));
                        acc = _mm_add_ps(acc, _mm_mul_ps(inp, ker));
                    }
                }

                if do_relu {
                    acc = _mm_max_ps(acc, zero);
                }
                _mm_storeu_ps(out_ptr.add(out_base + ch), acc);
            }
            ch += 4;
        }
        // Scalar tail.
        while ch < channels {
            let mut acc = bias.map_or(0.0, |b| b[ch]);
            for ky in 0..kh {
                let in_y = in_y0 + ky;
                let input_row_base = batch_input_base + (in_y * plan.in_w + in_x0) * channels;
                let kernel_row_base = (ky * kw) * channels;
                for kx in 0..kw {
                    acc += input[input_row_base + kx * channels + ch]
                        * kernel[kernel_row_base + kx * channels + ch];
                }
            }
            if do_relu && acc < 0.0 {
                acc = 0.0;
            }
            out_row[out_base + ch] = acc;
            ch += 1;
        }
    }
    // SiLU cannot be fused cheaply in SIMD (needs exp); apply as post-pass.
    if matches!(activation, Activation::Silu) {
        silu_slice_inplace(out_row);
    }
}

pub(super) fn depthwise_conv2d_nhwc_row(
    input: &[f32],
    kernel: &[f32],
    bias: Option<&[f32]>,
    plan: DepthwiseConv2dPlan,
    row_idx: usize,
    out_row: &mut [f32],
    activation: Activation,
) {
    // SIMD fast path for depth_multiplier == 1 (standard depthwise conv).
    // When dm=1, out_channels == channels and the kernel layout simplifies to
    // [KH, KW, C] — contiguous channel data enables vectorization.
    if plan.depth_multiplier == 1 && plan.out_channels >= 4 && !cfg!(miri) {
        #[cfg(target_arch = "aarch64")]
        if crate::host_cpu().features.neon {
            // SAFETY: NEON detected, pointers bounded by plan dimensions validated at
            // function entry. Each output element written exactly once.
            #[allow(unsafe_code)]
            unsafe {
                depthwise_conv2d_nhwc_row_neon(
                    input, kernel, bias, plan, row_idx, out_row, activation,
                );
            }
            return;
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            // AVX-512 DW row kernel when available. Kill-switch
            // `YSCV_AVX512_DW_OFF=1`, cached via OnceLock so the env read
            // doesn't repeat per DW row.
            if crate::host_cpu().features.avx512f && !avx512_dw_disabled() {
                // SAFETY: AVX-512F detected, bounds guaranteed.
                #[allow(unsafe_code)]
                unsafe {
                    depthwise_conv2d_nhwc_row_avx512(
                        input, kernel, bias, plan, row_idx, out_row, activation,
                    );
                }
                return;
            }
            if crate::host_cpu().features.avx && crate::host_cpu().features.fma {
                // SAFETY: AVX+FMA detected, same bounds guarantees.
                #[allow(unsafe_code)]
                unsafe {
                    depthwise_conv2d_nhwc_row_avx_fma(
                        input, kernel, bias, plan, row_idx, out_row, activation,
                    );
                }
                return;
            }
            if crate::host_cpu().features.avx {
                // SAFETY: AVX detected (no FMA), same bounds guarantees.
                #[allow(unsafe_code)]
                unsafe {
                    depthwise_conv2d_nhwc_row_avx(
                        input, kernel, bias, plan, row_idx, out_row, activation,
                    );
                }
                return;
            }
            if crate::host_cpu().features.sse {
                // SAFETY: SSE detected, same bounds guarantees.
                #[allow(unsafe_code)]
                unsafe {
                    depthwise_conv2d_nhwc_row_sse(
                        input, kernel, bias, plan, row_idx, out_row, activation,
                    );
                }
                return;
            }
        }
    }

    // Scalar fallback (handles depth_multiplier > 1 and all other cases).
    let batch_idx = row_idx / plan.out_h;
    let out_y = row_idx % plan.out_h;
    let in_y0 = out_y * plan.stride_h;
    let batch_input_base = batch_idx * plan.in_h * plan.in_w * plan.channels;
    let do_relu = matches!(activation, Activation::Relu);

    for out_x in 0..plan.out_w {
        let in_x0 = out_x * plan.stride_w;
        let out_cell_base = out_x * plan.out_channels;

        for out_channel in 0..plan.out_channels {
            let mut acc = bias.map_or(0.0, |bias_values| bias_values[out_channel]);
            let in_channel = out_channel / plan.depth_multiplier;
            let depth_index = out_channel % plan.depth_multiplier;

            for ky in 0..plan.kernel_h {
                let in_y = in_y0 + ky;
                let input_row_base = batch_input_base + (in_y * plan.in_w + in_x0) * plan.channels;
                let kernel_row_base = ky * plan.kernel_w * plan.channels * plan.depth_multiplier;

                for kx in 0..plan.kernel_w {
                    let input_value = input[input_row_base + kx * plan.channels + in_channel];
                    let kernel_index = kernel_row_base
                        + kx * plan.channels * plan.depth_multiplier
                        + in_channel * plan.depth_multiplier
                        + depth_index;
                    acc += input_value * kernel[kernel_index];
                }
            }

            if do_relu && acc < 0.0 {
                acc = 0.0;
            }
            out_row[out_cell_base + out_channel] = acc;
        }
    }
    // SiLU cannot be fused cheaply (needs exp); apply as post-pass.
    if matches!(activation, Activation::Silu) {
        silu_slice_inplace(out_row);
    }
}
