//! Direct NHWC convolution kernels: the generic per-row im2col-free path
//! and the specialized 3×3 direct kernels (NEON/AVX/SSE).

use super::*;

pub(super) fn conv2d_nhwc_row(
    input: &[f32],
    kernel: &[f32],
    bias: Option<&[f32]>,
    plan: Conv2dPlan,
    row_idx: usize,
    out_row: &mut [f32],
) {
    // For small C_out, use inlined SIMD to avoid 27+ function-pointer dispatches
    // per output pixel (3ns each × 110K calls = 330µs overhead for stem conv).
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if plan.out_channels >= 8
        && plan.out_channels <= 32
        && plan.out_channels.is_multiple_of(8)
        && !cfg!(miri)
        && std::is_x86_feature_detected!("avx")
        && std::is_x86_feature_detected!("fma")
    {
        #[allow(unsafe_code)]
        unsafe {
            conv2d_nhwc_row_inline_avx_fma(input, kernel, bias, plan, row_idx, out_row);
        }
        return;
    }

    conv2d_nhwc_row_generic(input, kernel, bias, plan, row_idx, out_row);
}

fn conv2d_nhwc_row_generic(
    input: &[f32],
    kernel: &[f32],
    bias: Option<&[f32]>,
    plan: Conv2dPlan,
    row_idx: usize,
    out_row: &mut [f32],
) {
    let batch_idx = row_idx / plan.out_h;
    let out_y = row_idx % plan.out_h;
    let in_y0 = out_y * plan.stride_h;
    let batch_input_base = batch_idx * plan.in_h * plan.in_w * plan.in_channels;

    for out_x in 0..plan.out_w {
        let in_x0 = out_x * plan.stride_w;
        let out_cell_base = out_x * plan.out_channels;
        let out_slice = &mut out_row[out_cell_base..out_cell_base + plan.out_channels];

        if let Some(bias_values) = bias {
            out_slice.copy_from_slice(&bias_values[..plan.out_channels]);
        } else {
            out_slice.fill(0.0);
        }

        for ky in 0..plan.kernel_h {
            let in_y = in_y0 + ky;
            let input_row_base = batch_input_base + (in_y * plan.in_w + in_x0) * plan.in_channels;
            let kernel_row_base = ky * plan.kernel_w * plan.in_channels * plan.out_channels;

            for kx in 0..plan.kernel_w {
                let input_pixel_base = input_row_base + kx * plan.in_channels;
                let kernel_pixel_base = kernel_row_base + kx * plan.in_channels * plan.out_channels;

                for in_channel in 0..plan.in_channels {
                    let input_val = input[input_pixel_base + in_channel];
                    let k_base = kernel_pixel_base + in_channel * plan.out_channels;
                    conv_fma_row(
                        out_slice,
                        &kernel[k_base..k_base + plan.out_channels],
                        input_val,
                    );
                }
            }
        }
    }
}

/// Inlined AVX+FMA conv row for small C_out (≤32).
/// Eliminates function-pointer dispatch overhead by keeping accumulators in
/// AVX registers across all kernel positions.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx,fma")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
unsafe fn conv2d_nhwc_row_inline_avx_fma(
    input: &[f32],
    kernel: &[f32],
    bias: Option<&[f32]>,
    plan: Conv2dPlan,
    row_idx: usize,
    out_row: &mut [f32],
) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let n = plan.out_channels;
    let batch_idx = row_idx / plan.out_h;
    let out_y = row_idx % plan.out_h;
    let in_y0 = out_y * plan.stride_h;
    let batch_input_base = batch_idx * plan.in_h * plan.in_w * plan.in_channels;
    let inp = input.as_ptr();
    let ker = kernel.as_ptr();
    let bias_ptr = bias.map(|b| b.as_ptr());
    let out_ptr = out_row.as_mut_ptr();

    for out_x in 0..plan.out_w {
        let in_x0 = out_x * plan.stride_w;
        let ob = out_x * n;

        // Initialize accumulators from bias (or zero).
        // Use exact multiples of 8 to avoid OOB reads/writes.
        let (mut a0, mut a1, mut a2, mut a3);
        if let Some(bp) = bias_ptr {
            a0 = _mm256_loadu_ps(bp);
            a1 = if n >= 16 {
                _mm256_loadu_ps(bp.add(8))
            } else {
                _mm256_setzero_ps()
            };
            a2 = if n >= 24 {
                _mm256_loadu_ps(bp.add(16))
            } else {
                _mm256_setzero_ps()
            };
            a3 = if n >= 32 {
                _mm256_loadu_ps(bp.add(24))
            } else {
                _mm256_setzero_ps()
            };
        } else {
            a0 = _mm256_setzero_ps();
            a1 = _mm256_setzero_ps();
            a2 = _mm256_setzero_ps();
            a3 = _mm256_setzero_ps();
        }

        // Accumulate all kernel positions inline — no function pointer dispatch
        for ky in 0..plan.kernel_h {
            let in_y = in_y0 + ky;
            let input_row_base = batch_input_base + (in_y * plan.in_w + in_x0) * plan.in_channels;
            let kernel_row_base = ky * plan.kernel_w * plan.in_channels * n;

            for kx in 0..plan.kernel_w {
                let input_pixel_base = input_row_base + kx * plan.in_channels;
                let kernel_pixel_base = kernel_row_base + kx * plan.in_channels * n;

                for ic in 0..plan.in_channels {
                    let iv = _mm256_set1_ps(*inp.add(input_pixel_base + ic));
                    let kb = kernel_pixel_base + ic * n;
                    a0 = _mm256_fmadd_ps(iv, _mm256_loadu_ps(ker.add(kb)), a0);
                    if n >= 16 {
                        a1 = _mm256_fmadd_ps(iv, _mm256_loadu_ps(ker.add(kb + 8)), a1);
                    }
                    if n >= 24 {
                        a2 = _mm256_fmadd_ps(iv, _mm256_loadu_ps(ker.add(kb + 16)), a2);
                    }
                    if n >= 32 {
                        a3 = _mm256_fmadd_ps(iv, _mm256_loadu_ps(ker.add(kb + 24)), a3);
                    }
                }
            }
        }

        // Store
        _mm256_storeu_ps(out_ptr.add(ob), a0);
        if n >= 16 {
            _mm256_storeu_ps(out_ptr.add(ob + 8), a1);
        }
        if n >= 24 {
            _mm256_storeu_ps(out_ptr.add(ob + 16), a2);
        }
        if n >= 32 {
            _mm256_storeu_ps(out_ptr.add(ob + 24), a3);
        }
    }
}

/// Direct 3×3 convolution microkernel — no im2col overhead.
/// For each output pixel, load 3×3×C_in input values and multiply with kernel.
/// Accumulate C_out output channels using SIMD FMA.
///
/// When stride_w == 1, processes two adjacent output pixels at a time.
/// Adjacent pixels at (ox, ox+1) share input columns: pixel ox uses columns
/// [ix, ix+1, ix+2] and pixel ox+1 uses [ix+1, ix+2, ix+3]. The middle two
/// columns are shared, saving ~33% of input loads.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
pub(super) unsafe fn conv2d_3x3_direct_neon(
    input: &[f32],      // [H, W, C_in] NHWC (batch-dim already stripped)
    kernel: &[f32],     // [3, 3, C_in, C_out]
    output: &mut [f32], // [out_H, out_W, C_out]
    w: usize,
    c_in: usize,
    c_out: usize,
    out_h: usize,
    out_w: usize,
    stride_h: usize,
    stride_w: usize,
) {
    use std::arch::aarch64::*;

    // Bounds proof: max input index = ((out_h-1)*stride_h + 2) * w * c_in + (out_w-1)*stride_w + 2) * c_in + c_in - 1
    debug_assert!(
        input.len() >= ((out_h.saturating_sub(1)) * stride_h + 3) * w * c_in,
        "conv2d_3x3_direct_neon: input too small"
    );
    debug_assert!(
        output.len() >= out_h * out_w * c_out,
        "conv2d_3x3_direct_neon: output too small"
    );
    debug_assert!(
        kernel.len() >= 3 * 3 * c_in * c_out,
        "conv2d_3x3_direct_neon: kernel too small"
    );

    for oy in 0..out_h {
        let iy_base = oy * stride_h;

        // When stride_w == 1 and we have at least 2 output pixels remaining,
        // process pairs of adjacent ox positions. For kernel column kx:
        //   pixel at ox   reads input column (ix_base + kx)
        //   pixel at ox+1 reads input column (ix_base + 1 + kx) = (ix_base + kx + 1)
        // So across kx=0,1,2 the pair reads columns ix_base..ix_base+3,
        // and columns ix_base+1 and ix_base+2 are shared.
        let mut ox = 0usize;
        if stride_w == 1 {
            while ox + 2 <= out_w {
                let ix_base = ox; // stride_w == 1
                let out_off_a = (oy * out_w + ox) * c_out;
                let out_off_b = out_off_a + c_out;

                let mut co = 0;
                while co + 8 <= c_out {
                    let mut acc_a0 = vdupq_n_f32(0.0);
                    let mut acc_a1 = vdupq_n_f32(0.0);
                    let mut acc_b0 = vdupq_n_f32(0.0);
                    let mut acc_b1 = vdupq_n_f32(0.0);

                    for ky in 0..3 {
                        let iy = iy_base + ky;
                        let row_base = iy * w;
                        // Load input values for 4 adjacent columns: ix_base..ix_base+3
                        // pixel A uses cols 0,1,2; pixel B uses cols 1,2,3
                        for ci in 0..c_in {
                            let in0 = *input.get_unchecked((row_base + ix_base) * c_in + ci);
                            let in1 = *input.get_unchecked((row_base + ix_base + 1) * c_in + ci);
                            let in2 = *input.get_unchecked((row_base + ix_base + 2) * c_in + ci);
                            let in3 = *input.get_unchecked((row_base + ix_base + 3) * c_in + ci);

                            // kernel weights for kx=0,1,2
                            let k0_off = ky * 3 * c_in * c_out + ci * c_out + co;
                            let k1_off = (ky * 3 + 1) * c_in * c_out + ci * c_out + co;
                            let k2_off = (ky * 3 + 2) * c_in * c_out + ci * c_out + co;
                            let kw0_lo = vld1q_f32(kernel.as_ptr().add(k0_off));
                            let kw0_hi = vld1q_f32(kernel.as_ptr().add(k0_off + 4));
                            let kw1_lo = vld1q_f32(kernel.as_ptr().add(k1_off));
                            let kw1_hi = vld1q_f32(kernel.as_ptr().add(k1_off + 4));
                            let kw2_lo = vld1q_f32(kernel.as_ptr().add(k2_off));
                            let kw2_hi = vld1q_f32(kernel.as_ptr().add(k2_off + 4));

                            // Pixel A: in0*k0 + in1*k1 + in2*k2
                            let va0 = vdupq_n_f32(in0);
                            let va1 = vdupq_n_f32(in1);
                            let va2 = vdupq_n_f32(in2);
                            acc_a0 = vfmaq_f32(acc_a0, va0, kw0_lo);
                            acc_a1 = vfmaq_f32(acc_a1, va0, kw0_hi);
                            acc_a0 = vfmaq_f32(acc_a0, va1, kw1_lo);
                            acc_a1 = vfmaq_f32(acc_a1, va1, kw1_hi);
                            acc_a0 = vfmaq_f32(acc_a0, va2, kw2_lo);
                            acc_a1 = vfmaq_f32(acc_a1, va2, kw2_hi);

                            // Pixel B: in1*k0 + in2*k1 + in3*k2
                            let vb3 = vdupq_n_f32(in3);
                            acc_b0 = vfmaq_f32(acc_b0, va1, kw0_lo);
                            acc_b1 = vfmaq_f32(acc_b1, va1, kw0_hi);
                            acc_b0 = vfmaq_f32(acc_b0, va2, kw1_lo);
                            acc_b1 = vfmaq_f32(acc_b1, va2, kw1_hi);
                            acc_b0 = vfmaq_f32(acc_b0, vb3, kw2_lo);
                            acc_b1 = vfmaq_f32(acc_b1, vb3, kw2_hi);
                        }
                    }

                    vst1q_f32(output.as_mut_ptr().add(out_off_a + co), acc_a0);
                    vst1q_f32(output.as_mut_ptr().add(out_off_a + co + 4), acc_a1);
                    vst1q_f32(output.as_mut_ptr().add(out_off_b + co), acc_b0);
                    vst1q_f32(output.as_mut_ptr().add(out_off_b + co + 4), acc_b1);
                    co += 8;
                }

                while co + 4 <= c_out {
                    let mut acc_a = vdupq_n_f32(0.0);
                    let mut acc_b = vdupq_n_f32(0.0);

                    for ky in 0..3 {
                        let iy = iy_base + ky;
                        let row_base = iy * w;
                        for ci in 0..c_in {
                            let in0 = *input.get_unchecked((row_base + ix_base) * c_in + ci);
                            let in1 = *input.get_unchecked((row_base + ix_base + 1) * c_in + ci);
                            let in2 = *input.get_unchecked((row_base + ix_base + 2) * c_in + ci);
                            let in3 = *input.get_unchecked((row_base + ix_base + 3) * c_in + ci);

                            let k0_off = ky * 3 * c_in * c_out + ci * c_out + co;
                            let k1_off = (ky * 3 + 1) * c_in * c_out + ci * c_out + co;
                            let k2_off = (ky * 3 + 2) * c_in * c_out + ci * c_out + co;
                            let kw0 = vld1q_f32(kernel.as_ptr().add(k0_off));
                            let kw1 = vld1q_f32(kernel.as_ptr().add(k1_off));
                            let kw2 = vld1q_f32(kernel.as_ptr().add(k2_off));

                            let va0 = vdupq_n_f32(in0);
                            let va1 = vdupq_n_f32(in1);
                            let va2 = vdupq_n_f32(in2);
                            acc_a = vfmaq_f32(acc_a, va0, kw0);
                            acc_a = vfmaq_f32(acc_a, va1, kw1);
                            acc_a = vfmaq_f32(acc_a, va2, kw2);

                            let vb3 = vdupq_n_f32(in3);
                            acc_b = vfmaq_f32(acc_b, va1, kw0);
                            acc_b = vfmaq_f32(acc_b, va2, kw1);
                            acc_b = vfmaq_f32(acc_b, vb3, kw2);
                        }
                    }

                    vst1q_f32(output.as_mut_ptr().add(out_off_a + co), acc_a);
                    vst1q_f32(output.as_mut_ptr().add(out_off_b + co), acc_b);
                    co += 4;
                }

                while co < c_out {
                    let mut acc_a = 0.0f32;
                    let mut acc_b = 0.0f32;
                    for ky in 0..3 {
                        let iy = iy_base + ky;
                        let row_base = iy * w;
                        for ci in 0..c_in {
                            let in0 = input[(row_base + ix_base) * c_in + ci];
                            let in1 = input[(row_base + ix_base + 1) * c_in + ci];
                            let in2 = input[(row_base + ix_base + 2) * c_in + ci];
                            let in3 = input[(row_base + ix_base + 3) * c_in + ci];
                            let k0 = kernel[ky * 3 * c_in * c_out + ci * c_out + co];
                            let k1 = kernel[(ky * 3 + 1) * c_in * c_out + ci * c_out + co];
                            let k2 = kernel[(ky * 3 + 2) * c_in * c_out + ci * c_out + co];
                            acc_a += in0 * k0 + in1 * k1 + in2 * k2;
                            acc_b += in1 * k0 + in2 * k1 + in3 * k2;
                        }
                    }
                    *output.get_unchecked_mut(out_off_a + co) = acc_a;
                    *output.get_unchecked_mut(out_off_b + co) = acc_b;
                    co += 1;
                }

                ox += 2;
            }
        }

        // Handle remaining single pixels (odd out_w or stride_w != 1).
        while ox < out_w {
            let ix_base = ox * stride_w;
            let out_off = (oy * out_w + ox) * c_out;

            let mut co = 0;
            while co + 8 <= c_out {
                let mut acc0 = vdupq_n_f32(0.0);
                let mut acc1 = vdupq_n_f32(0.0);

                for ky in 0..3 {
                    for kx in 0..3 {
                        let iy = iy_base + ky;
                        let ix = ix_base + kx;
                        let in_off = (iy * w + ix) * c_in;
                        let k_base = (ky * 3 + kx) * c_in * c_out;

                        for ci in 0..c_in {
                            let iv = vdupq_n_f32(*input.get_unchecked(in_off + ci));
                            let koff = k_base + ci * c_out + co;
                            acc0 = vfmaq_f32(acc0, iv, vld1q_f32(kernel.as_ptr().add(koff)));
                            acc1 = vfmaq_f32(acc1, iv, vld1q_f32(kernel.as_ptr().add(koff + 4)));
                        }
                    }
                }

                vst1q_f32(output.as_mut_ptr().add(out_off + co), acc0);
                vst1q_f32(output.as_mut_ptr().add(out_off + co + 4), acc1);
                co += 8;
            }

            while co + 4 <= c_out {
                let mut acc = vdupq_n_f32(0.0);
                for ky in 0..3 {
                    for kx in 0..3 {
                        let iy = iy_base + ky;
                        let ix = ix_base + kx;
                        let in_off = (iy * w + ix) * c_in;
                        for ci in 0..c_in {
                            let iv = vdupq_n_f32(*input.get_unchecked(in_off + ci));
                            acc = vfmaq_f32(
                                acc,
                                iv,
                                vld1q_f32(
                                    kernel
                                        .as_ptr()
                                        .add((ky * 3 + kx) * c_in * c_out + ci * c_out + co),
                                ),
                            );
                        }
                    }
                }
                vst1q_f32(output.as_mut_ptr().add(out_off + co), acc);
                co += 4;
            }

            // Handle remaining channels scalar
            while co < c_out {
                let mut acc = 0.0f32;
                for ky in 0..3 {
                    for kx in 0..3 {
                        let iy = iy_base + ky;
                        let ix = ix_base + kx;
                        for ci in 0..c_in {
                            acc += input[(iy * w + ix) * c_in + ci]
                                * kernel[(ky * 3 + kx) * c_in * c_out + ci * c_out + co];
                        }
                    }
                }
                *output.get_unchecked_mut(out_off + co) = acc;
                co += 1;
            }

            ox += 1;
        }
    }
}

/// Direct 3×3 convolution microkernel for x86_64 with AVX-256 + FMA.
/// Processes 8 output channels per iteration (vs 4 for SSE), doubling
/// throughput on the inner c_out loop. Falls back to scalar for tail channels.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx", enable = "fma")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
pub(super) unsafe fn conv2d_3x3_direct_avx(
    input: &[f32],      // [H, W, C_in] NHWC (batch-dim already stripped)
    kernel: &[f32],     // [3, 3, C_in, C_out]
    output: &mut [f32], // [out_H, out_W, C_out]
    w: usize,
    c_in: usize,
    c_out: usize,
    out_h: usize,
    out_w: usize,
    stride_h: usize,
    stride_w: usize,
) {
    use std::arch::x86_64::*;

    for oy in 0..out_h {
        let iy_base = oy * stride_h;

        let mut ox = 0usize;
        if stride_w == 1 {
            while ox + 2 <= out_w {
                let ix_base = ox; // stride_w == 1
                let out_off_a = (oy * out_w + ox) * c_out;
                let out_off_b = out_off_a + c_out;

                // Process 16 output channels per iteration (2x AVX-256 registers)
                let mut co = 0;
                while co + 16 <= c_out {
                    let mut acc_a0 = _mm256_setzero_ps();
                    let mut acc_a1 = _mm256_setzero_ps();
                    let mut acc_b0 = _mm256_setzero_ps();
                    let mut acc_b1 = _mm256_setzero_ps();

                    for ky in 0..3 {
                        let iy = iy_base + ky;
                        let row_base = iy * w;
                        for ci in 0..c_in {
                            let in0 = *input.get_unchecked((row_base + ix_base) * c_in + ci);
                            let in1 = *input.get_unchecked((row_base + ix_base + 1) * c_in + ci);
                            let in2 = *input.get_unchecked((row_base + ix_base + 2) * c_in + ci);
                            let in3 = *input.get_unchecked((row_base + ix_base + 3) * c_in + ci);

                            let k0_off = ky * 3 * c_in * c_out + ci * c_out + co;
                            let k1_off = (ky * 3 + 1) * c_in * c_out + ci * c_out + co;
                            let k2_off = (ky * 3 + 2) * c_in * c_out + ci * c_out + co;
                            let kw0_lo = _mm256_loadu_ps(kernel.as_ptr().add(k0_off));
                            let kw0_hi = _mm256_loadu_ps(kernel.as_ptr().add(k0_off + 8));
                            let kw1_lo = _mm256_loadu_ps(kernel.as_ptr().add(k1_off));
                            let kw1_hi = _mm256_loadu_ps(kernel.as_ptr().add(k1_off + 8));
                            let kw2_lo = _mm256_loadu_ps(kernel.as_ptr().add(k2_off));
                            let kw2_hi = _mm256_loadu_ps(kernel.as_ptr().add(k2_off + 8));

                            // Pixel A: in0*k0 + in1*k1 + in2*k2
                            let va0 = _mm256_set1_ps(in0);
                            let va1 = _mm256_set1_ps(in1);
                            let va2 = _mm256_set1_ps(in2);
                            acc_a0 = _mm256_fmadd_ps(va0, kw0_lo, acc_a0);
                            acc_a1 = _mm256_fmadd_ps(va0, kw0_hi, acc_a1);
                            acc_a0 = _mm256_fmadd_ps(va1, kw1_lo, acc_a0);
                            acc_a1 = _mm256_fmadd_ps(va1, kw1_hi, acc_a1);
                            acc_a0 = _mm256_fmadd_ps(va2, kw2_lo, acc_a0);
                            acc_a1 = _mm256_fmadd_ps(va2, kw2_hi, acc_a1);

                            // Pixel B: in1*k0 + in2*k1 + in3*k2
                            let vb3 = _mm256_set1_ps(in3);
                            acc_b0 = _mm256_fmadd_ps(va1, kw0_lo, acc_b0);
                            acc_b1 = _mm256_fmadd_ps(va1, kw0_hi, acc_b1);
                            acc_b0 = _mm256_fmadd_ps(va2, kw1_lo, acc_b0);
                            acc_b1 = _mm256_fmadd_ps(va2, kw1_hi, acc_b1);
                            acc_b0 = _mm256_fmadd_ps(vb3, kw2_lo, acc_b0);
                            acc_b1 = _mm256_fmadd_ps(vb3, kw2_hi, acc_b1);
                        }
                    }

                    _mm256_storeu_ps(output.as_mut_ptr().add(out_off_a + co), acc_a0);
                    _mm256_storeu_ps(output.as_mut_ptr().add(out_off_a + co + 8), acc_a1);
                    _mm256_storeu_ps(output.as_mut_ptr().add(out_off_b + co), acc_b0);
                    _mm256_storeu_ps(output.as_mut_ptr().add(out_off_b + co + 8), acc_b1);
                    co += 16;
                }

                // Process 8 output channels with single AVX register pair
                while co + 8 <= c_out {
                    let mut acc_a = _mm256_setzero_ps();
                    let mut acc_b = _mm256_setzero_ps();

                    for ky in 0..3 {
                        let iy = iy_base + ky;
                        let row_base = iy * w;
                        for ci in 0..c_in {
                            let in0 = *input.get_unchecked((row_base + ix_base) * c_in + ci);
                            let in1 = *input.get_unchecked((row_base + ix_base + 1) * c_in + ci);
                            let in2 = *input.get_unchecked((row_base + ix_base + 2) * c_in + ci);
                            let in3 = *input.get_unchecked((row_base + ix_base + 3) * c_in + ci);

                            let k0_off = ky * 3 * c_in * c_out + ci * c_out + co;
                            let k1_off = (ky * 3 + 1) * c_in * c_out + ci * c_out + co;
                            let k2_off = (ky * 3 + 2) * c_in * c_out + ci * c_out + co;
                            let kw0 = _mm256_loadu_ps(kernel.as_ptr().add(k0_off));
                            let kw1 = _mm256_loadu_ps(kernel.as_ptr().add(k1_off));
                            let kw2 = _mm256_loadu_ps(kernel.as_ptr().add(k2_off));

                            let va0 = _mm256_set1_ps(in0);
                            let va1 = _mm256_set1_ps(in1);
                            let va2 = _mm256_set1_ps(in2);
                            acc_a = _mm256_fmadd_ps(va0, kw0, acc_a);
                            acc_a = _mm256_fmadd_ps(va1, kw1, acc_a);
                            acc_a = _mm256_fmadd_ps(va2, kw2, acc_a);

                            let vb3 = _mm256_set1_ps(in3);
                            acc_b = _mm256_fmadd_ps(va1, kw0, acc_b);
                            acc_b = _mm256_fmadd_ps(va2, kw1, acc_b);
                            acc_b = _mm256_fmadd_ps(vb3, kw2, acc_b);
                        }
                    }

                    _mm256_storeu_ps(output.as_mut_ptr().add(out_off_a + co), acc_a);
                    _mm256_storeu_ps(output.as_mut_ptr().add(out_off_b + co), acc_b);
                    co += 8;
                }

                // Scalar tail for remaining channels
                while co < c_out {
                    let mut acc_a = 0.0f32;
                    let mut acc_b = 0.0f32;
                    for ky in 0..3 {
                        let iy = iy_base + ky;
                        let row_base = iy * w;
                        for ci in 0..c_in {
                            let in0 = input[(row_base + ix_base) * c_in + ci];
                            let in1 = input[(row_base + ix_base + 1) * c_in + ci];
                            let in2 = input[(row_base + ix_base + 2) * c_in + ci];
                            let in3 = input[(row_base + ix_base + 3) * c_in + ci];
                            let k0 = kernel[ky * 3 * c_in * c_out + ci * c_out + co];
                            let k1 = kernel[(ky * 3 + 1) * c_in * c_out + ci * c_out + co];
                            let k2 = kernel[(ky * 3 + 2) * c_in * c_out + ci * c_out + co];
                            acc_a += in0 * k0 + in1 * k1 + in2 * k2;
                            acc_b += in1 * k0 + in2 * k1 + in3 * k2;
                        }
                    }
                    *output.get_unchecked_mut(out_off_a + co) = acc_a;
                    *output.get_unchecked_mut(out_off_b + co) = acc_b;
                    co += 1;
                }

                ox += 2;
            }
        }

        // Handle remaining single pixels (odd out_w or stride_w != 1).
        while ox < out_w {
            let ix_base = ox * stride_w;
            let out_off = (oy * out_w + ox) * c_out;

            let mut co = 0;
            while co + 8 <= c_out {
                let mut acc = _mm256_setzero_ps();

                for ky in 0..3 {
                    for kx in 0..3 {
                        let iy = iy_base + ky;
                        let ix = ix_base + kx;
                        let in_off = (iy * w + ix) * c_in;
                        let k_base = (ky * 3 + kx) * c_in * c_out;

                        for ci in 0..c_in {
                            let iv = _mm256_set1_ps(*input.get_unchecked(in_off + ci));
                            let koff = k_base + ci * c_out + co;
                            acc = _mm256_fmadd_ps(
                                iv,
                                _mm256_loadu_ps(kernel.as_ptr().add(koff)),
                                acc,
                            );
                        }
                    }
                }

                _mm256_storeu_ps(output.as_mut_ptr().add(out_off + co), acc);
                co += 8;
            }

            // Handle remaining channels scalar
            while co < c_out {
                let mut acc = 0.0f32;
                for ky in 0..3 {
                    for kx in 0..3 {
                        let iy = iy_base + ky;
                        let ix = ix_base + kx;
                        for ci in 0..c_in {
                            acc += input[(iy * w + ix) * c_in + ci]
                                * kernel[(ky * 3 + kx) * c_in * c_out + ci * c_out + co];
                        }
                    }
                }
                *output.get_unchecked_mut(out_off + co) = acc;
                co += 1;
            }

            ox += 1;
        }
    }
}

/// Direct 3×3 convolution microkernel for x86_64 with SSE + FMA.
/// Mirrors the NEON implementation: processes two adjacent output pixels at a
/// time when stride_w == 1, sharing overlapping input columns to save loads.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse", enable = "fma")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
pub(super) unsafe fn conv2d_3x3_direct_sse(
    input: &[f32],      // [H, W, C_in] NHWC (batch-dim already stripped)
    kernel: &[f32],     // [3, 3, C_in, C_out]
    output: &mut [f32], // [out_H, out_W, C_out]
    w: usize,
    c_in: usize,
    c_out: usize,
    out_h: usize,
    out_w: usize,
    stride_h: usize,
    stride_w: usize,
) {
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    for oy in 0..out_h {
        let iy_base = oy * stride_h;

        let mut ox = 0usize;
        if stride_w == 1 {
            while ox + 2 <= out_w {
                let ix_base = ox; // stride_w == 1
                let out_off_a = (oy * out_w + ox) * c_out;
                let out_off_b = out_off_a + c_out;

                let mut co = 0;
                while co + 8 <= c_out {
                    let mut acc_a0 = _mm_setzero_ps();
                    let mut acc_a1 = _mm_setzero_ps();
                    let mut acc_b0 = _mm_setzero_ps();
                    let mut acc_b1 = _mm_setzero_ps();

                    for ky in 0..3 {
                        let iy = iy_base + ky;
                        let row_base = iy * w;
                        for ci in 0..c_in {
                            let in0 = *input.get_unchecked((row_base + ix_base) * c_in + ci);
                            let in1 = *input.get_unchecked((row_base + ix_base + 1) * c_in + ci);
                            let in2 = *input.get_unchecked((row_base + ix_base + 2) * c_in + ci);
                            let in3 = *input.get_unchecked((row_base + ix_base + 3) * c_in + ci);

                            let k0_off = ky * 3 * c_in * c_out + ci * c_out + co;
                            let k1_off = (ky * 3 + 1) * c_in * c_out + ci * c_out + co;
                            let k2_off = (ky * 3 + 2) * c_in * c_out + ci * c_out + co;
                            let kw0_lo = _mm_loadu_ps(kernel.as_ptr().add(k0_off));
                            let kw0_hi = _mm_loadu_ps(kernel.as_ptr().add(k0_off + 4));
                            let kw1_lo = _mm_loadu_ps(kernel.as_ptr().add(k1_off));
                            let kw1_hi = _mm_loadu_ps(kernel.as_ptr().add(k1_off + 4));
                            let kw2_lo = _mm_loadu_ps(kernel.as_ptr().add(k2_off));
                            let kw2_hi = _mm_loadu_ps(kernel.as_ptr().add(k2_off + 4));

                            // Pixel A: in0*k0 + in1*k1 + in2*k2
                            let va0 = _mm_set1_ps(in0);
                            let va1 = _mm_set1_ps(in1);
                            let va2 = _mm_set1_ps(in2);
                            acc_a0 = _mm_fmadd_ps(va0, kw0_lo, acc_a0);
                            acc_a1 = _mm_fmadd_ps(va0, kw0_hi, acc_a1);
                            acc_a0 = _mm_fmadd_ps(va1, kw1_lo, acc_a0);
                            acc_a1 = _mm_fmadd_ps(va1, kw1_hi, acc_a1);
                            acc_a0 = _mm_fmadd_ps(va2, kw2_lo, acc_a0);
                            acc_a1 = _mm_fmadd_ps(va2, kw2_hi, acc_a1);

                            // Pixel B: in1*k0 + in2*k1 + in3*k2
                            let vb3 = _mm_set1_ps(in3);
                            acc_b0 = _mm_fmadd_ps(va1, kw0_lo, acc_b0);
                            acc_b1 = _mm_fmadd_ps(va1, kw0_hi, acc_b1);
                            acc_b0 = _mm_fmadd_ps(va2, kw1_lo, acc_b0);
                            acc_b1 = _mm_fmadd_ps(va2, kw1_hi, acc_b1);
                            acc_b0 = _mm_fmadd_ps(vb3, kw2_lo, acc_b0);
                            acc_b1 = _mm_fmadd_ps(vb3, kw2_hi, acc_b1);
                        }
                    }

                    _mm_storeu_ps(output.as_mut_ptr().add(out_off_a + co), acc_a0);
                    _mm_storeu_ps(output.as_mut_ptr().add(out_off_a + co + 4), acc_a1);
                    _mm_storeu_ps(output.as_mut_ptr().add(out_off_b + co), acc_b0);
                    _mm_storeu_ps(output.as_mut_ptr().add(out_off_b + co + 4), acc_b1);
                    co += 8;
                }

                while co + 4 <= c_out {
                    let mut acc_a = _mm_setzero_ps();
                    let mut acc_b = _mm_setzero_ps();

                    for ky in 0..3 {
                        let iy = iy_base + ky;
                        let row_base = iy * w;
                        for ci in 0..c_in {
                            let in0 = *input.get_unchecked((row_base + ix_base) * c_in + ci);
                            let in1 = *input.get_unchecked((row_base + ix_base + 1) * c_in + ci);
                            let in2 = *input.get_unchecked((row_base + ix_base + 2) * c_in + ci);
                            let in3 = *input.get_unchecked((row_base + ix_base + 3) * c_in + ci);

                            let k0_off = ky * 3 * c_in * c_out + ci * c_out + co;
                            let k1_off = (ky * 3 + 1) * c_in * c_out + ci * c_out + co;
                            let k2_off = (ky * 3 + 2) * c_in * c_out + ci * c_out + co;
                            let kw0 = _mm_loadu_ps(kernel.as_ptr().add(k0_off));
                            let kw1 = _mm_loadu_ps(kernel.as_ptr().add(k1_off));
                            let kw2 = _mm_loadu_ps(kernel.as_ptr().add(k2_off));

                            let va0 = _mm_set1_ps(in0);
                            let va1 = _mm_set1_ps(in1);
                            let va2 = _mm_set1_ps(in2);
                            acc_a = _mm_fmadd_ps(va0, kw0, acc_a);
                            acc_a = _mm_fmadd_ps(va1, kw1, acc_a);
                            acc_a = _mm_fmadd_ps(va2, kw2, acc_a);

                            let vb3 = _mm_set1_ps(in3);
                            acc_b = _mm_fmadd_ps(va1, kw0, acc_b);
                            acc_b = _mm_fmadd_ps(va2, kw1, acc_b);
                            acc_b = _mm_fmadd_ps(vb3, kw2, acc_b);
                        }
                    }

                    _mm_storeu_ps(output.as_mut_ptr().add(out_off_a + co), acc_a);
                    _mm_storeu_ps(output.as_mut_ptr().add(out_off_b + co), acc_b);
                    co += 4;
                }

                while co < c_out {
                    let mut acc_a = 0.0f32;
                    let mut acc_b = 0.0f32;
                    for ky in 0..3 {
                        let iy = iy_base + ky;
                        let row_base = iy * w;
                        for ci in 0..c_in {
                            let in0 = input[(row_base + ix_base) * c_in + ci];
                            let in1 = input[(row_base + ix_base + 1) * c_in + ci];
                            let in2 = input[(row_base + ix_base + 2) * c_in + ci];
                            let in3 = input[(row_base + ix_base + 3) * c_in + ci];
                            let k0 = kernel[ky * 3 * c_in * c_out + ci * c_out + co];
                            let k1 = kernel[(ky * 3 + 1) * c_in * c_out + ci * c_out + co];
                            let k2 = kernel[(ky * 3 + 2) * c_in * c_out + ci * c_out + co];
                            acc_a += in0 * k0 + in1 * k1 + in2 * k2;
                            acc_b += in1 * k0 + in2 * k1 + in3 * k2;
                        }
                    }
                    *output.get_unchecked_mut(out_off_a + co) = acc_a;
                    *output.get_unchecked_mut(out_off_b + co) = acc_b;
                    co += 1;
                }

                ox += 2;
            }
        }

        // Handle remaining single pixels (odd out_w or stride_w != 1).
        while ox < out_w {
            let ix_base = ox * stride_w;
            let out_off = (oy * out_w + ox) * c_out;

            let mut co = 0;
            while co + 8 <= c_out {
                let mut acc0 = _mm_setzero_ps();
                let mut acc1 = _mm_setzero_ps();

                for ky in 0..3 {
                    for kx in 0..3 {
                        let iy = iy_base + ky;
                        let ix = ix_base + kx;
                        let in_off = (iy * w + ix) * c_in;
                        let k_base = (ky * 3 + kx) * c_in * c_out;

                        for ci in 0..c_in {
                            let iv = _mm_set1_ps(*input.get_unchecked(in_off + ci));
                            let koff = k_base + ci * c_out + co;
                            acc0 = _mm_fmadd_ps(iv, _mm_loadu_ps(kernel.as_ptr().add(koff)), acc0);
                            acc1 =
                                _mm_fmadd_ps(iv, _mm_loadu_ps(kernel.as_ptr().add(koff + 4)), acc1);
                        }
                    }
                }

                _mm_storeu_ps(output.as_mut_ptr().add(out_off + co), acc0);
                _mm_storeu_ps(output.as_mut_ptr().add(out_off + co + 4), acc1);
                co += 8;
            }

            while co + 4 <= c_out {
                let mut acc = _mm_setzero_ps();
                for ky in 0..3 {
                    for kx in 0..3 {
                        let iy = iy_base + ky;
                        let ix = ix_base + kx;
                        let in_off = (iy * w + ix) * c_in;
                        for ci in 0..c_in {
                            let iv = _mm_set1_ps(*input.get_unchecked(in_off + ci));
                            acc = _mm_fmadd_ps(
                                iv,
                                _mm_loadu_ps(
                                    kernel
                                        .as_ptr()
                                        .add((ky * 3 + kx) * c_in * c_out + ci * c_out + co),
                                ),
                                acc,
                            );
                        }
                    }
                }
                _mm_storeu_ps(output.as_mut_ptr().add(out_off + co), acc);
                co += 4;
            }

            // Handle remaining channels scalar
            while co < c_out {
                let mut acc = 0.0f32;
                for ky in 0..3 {
                    for kx in 0..3 {
                        let iy = iy_base + ky;
                        let ix = ix_base + kx;
                        for ci in 0..c_in {
                            acc += input[(iy * w + ix) * c_in + ci]
                                * kernel[(ky * 3 + kx) * c_in * c_out + ci * c_out + co];
                        }
                    }
                }
                *output.get_unchecked_mut(out_off + co) = acc;
                co += 1;
            }

            ox += 1;
        }
    }
}
