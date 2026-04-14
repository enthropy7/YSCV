use rayon::{ThreadPool, prelude::*};
use yscv_tensor::{AlignedVec, Tensor, TensorError};

use super::super::error::KernelError;
use super::config::{
    Conv2dPlan, Conv2dSpec, DepthwiseConv2dPlan, DepthwiseConv2dSpec, ParallelElementwiseConfig,
    SeparableConv2dKernels, SeparableConv2dSpec, should_parallelize_len,
};

/// Post-convolution fused activation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Activation {
    None,
    Silu,
}

/// Apply SiLU in-place on a mutable f32 slice.
#[inline]
fn silu_slice_inplace(data: &mut [f32]) {
    super::simd::silu_inplace(data);
}

pub fn conv2d_nhwc_with_config_and_pool(
    input: &Tensor,
    kernel: &Tensor,
    bias: Option<&Tensor>,
    spec: Conv2dSpec,
    config: ParallelElementwiseConfig,
    thread_pool: Option<&ThreadPool>,
) -> Result<Tensor, KernelError> {
    let plan = build_conv2d_plan(input, kernel, bias, spec)?;

    // Direct 3×3 microkernel for small inputs (no im2col overhead).
    // For small spatial sizes the im2col copy + BLAS call setup dominates; a
    // direct SIMD kernel that walks the input in-place is significantly faster.
    // However, for deep channels (large C_in * C_out) the BLAS GEMM has better
    // cache tiling, so we use a total-work threshold instead of spatial-only.
    // 2M FLOPs ≈ 0.08ms at 25 GFLOPS — below this the direct kernel wins.
    // Note: NEON kernel is single-threaded, so BLAS GEMM wins for large spatial
    // even with few channels due to multi-core parallelism.
    #[cfg(target_arch = "aarch64")]
    if plan.kernel_h == 3
        && plan.kernel_w == 3
        && plan.batch == 1
        && !cfg!(miri)
        && std::arch::is_aarch64_feature_detected!("neon")
        && (plan.out_h * plan.out_w)
            .saturating_mul(plan.in_channels)
            .saturating_mul(plan.out_channels)
            < 2_000_000
    {
        #[allow(unsafe_code)]
        let mut output = AlignedVec::<f32>::uninitialized(plan.output_len);
        #[allow(unsafe_code)]
        unsafe {
            conv2d_3x3_direct_neon(
                input.data(),
                kernel.data(),
                &mut output,
                plan.in_w,
                plan.in_channels,
                plan.out_channels,
                plan.out_h,
                plan.out_w,
                plan.stride_h,
                plan.stride_w,
            );
        }
        if let Some(b) = bias {
            let bd = b.data();
            for i in 0..plan.out_h * plan.out_w {
                for c in 0..plan.out_channels {
                    output[i * plan.out_channels + c] += bd[c];
                }
            }
        }
        return Tensor::from_aligned(vec![1, plan.out_h, plan.out_w, plan.out_channels], output)
            .map_err(Into::into);
    }

    #[cfg(target_arch = "x86_64")]
    if plan.kernel_h == 3
        && plan.kernel_w == 3
        && plan.batch == 1
        && !cfg!(miri)
        && is_x86_feature_detected!("avx")
        && is_x86_feature_detected!("fma")
        && (plan.out_h * plan.out_w)
            .saturating_mul(plan.in_channels)
            .saturating_mul(plan.out_channels)
            < 2_000_000
    {
        #[allow(unsafe_code)]
        let mut output = AlignedVec::<f32>::uninitialized(plan.output_len);
        #[allow(unsafe_code)]
        unsafe {
            conv2d_3x3_direct_avx(
                input.data(),
                kernel.data(),
                &mut output,
                plan.in_w,
                plan.in_channels,
                plan.out_channels,
                plan.out_h,
                plan.out_w,
                plan.stride_h,
                plan.stride_w,
            );
        }
        if let Some(b) = bias {
            let bd = b.data();
            for i in 0..plan.out_h * plan.out_w {
                for c in 0..plan.out_channels {
                    output[i * plan.out_channels + c] += bd[c];
                }
            }
        }
        return Tensor::from_aligned(vec![1, plan.out_h, plan.out_w, plan.out_channels], output)
            .map_err(Into::into);
    }

    #[cfg(target_arch = "x86_64")]
    if plan.kernel_h == 3
        && plan.kernel_w == 3
        && plan.batch == 1
        && !cfg!(miri)
        && is_x86_feature_detected!("fma")
        && (plan.out_h * plan.out_w)
            .saturating_mul(plan.in_channels)
            .saturating_mul(plan.out_channels)
            < 2_000_000
    {
        #[allow(unsafe_code)]
        let mut output = AlignedVec::<f32>::uninitialized(plan.output_len);
        #[allow(unsafe_code)]
        unsafe {
            conv2d_3x3_direct_sse(
                input.data(),
                kernel.data(),
                &mut output,
                plan.in_w,
                plan.in_channels,
                plan.out_channels,
                plan.out_h,
                plan.out_w,
                plan.stride_h,
                plan.stride_w,
            );
        }
        if let Some(b) = bias {
            let bd = b.data();
            for i in 0..plan.out_h * plan.out_w {
                for c in 0..plan.out_channels {
                    output[i * plan.out_channels + c] += bd[c];
                }
            }
        }
        return Tensor::from_aligned(vec![1, plan.out_h, plan.out_w, plan.out_channels], output)
            .map_err(Into::into);
    }

    // Fast path: im2col + BLAS sgemm for single-batch convolutions with enough output
    // positions to amortise BLAS/im2col overhead.
    #[cfg(feature = "blas")]
    if !cfg!(miri) && plan.batch == 1 {
        return conv2d_im2col_gemm(&plan, input.data(), kernel.data(), bias.map(Tensor::data));
    }

    let input_data = input.data();
    let kernel_data = kernel.data();
    let bias_data = bias.map(Tensor::data);
    let out_row_len = plan.out_w * plan.out_channels;
    if plan.output_len == 0 || out_row_len == 0 {
        return Tensor::from_vec(
            vec![plan.batch, plan.out_h, plan.out_w, plan.out_channels],
            vec![],
        )
        .map_err(Into::into);
    }

    // SAFETY: `conv2d_nhwc_row` writes every element in each output row.
    #[allow(unsafe_code)]
    let mut output = AlignedVec::<f32>::uninitialized(plan.output_len);

    if should_parallelize_len(plan.output_len, config.min_parallel_elements, thread_pool) {
        let mut work = || {
            output
                .par_chunks_mut(out_row_len)
                .enumerate()
                .for_each(|(row_idx, out_row)| {
                    conv2d_nhwc_row(input_data, kernel_data, bias_data, plan, row_idx, out_row);
                });
        };
        if let Some(pool) = thread_pool {
            pool.install(work);
        } else {
            work();
        }
    } else {
        for (row_idx, out_row) in output.chunks_mut(out_row_len).enumerate() {
            conv2d_nhwc_row(input_data, kernel_data, bias_data, plan, row_idx, out_row);
        }
    }

    Tensor::from_aligned(
        vec![plan.batch, plan.out_h, plan.out_w, plan.out_channels],
        output,
    )
    .map_err(Into::into)
}

pub fn depthwise_conv2d_nhwc_with_config_and_pool(
    input: &Tensor,
    kernel: &Tensor,
    bias: Option<&Tensor>,
    spec: DepthwiseConv2dSpec,
    config: ParallelElementwiseConfig,
    thread_pool: Option<&ThreadPool>,
) -> Result<Tensor, KernelError> {
    let plan = build_depthwise_conv2d_plan(input, kernel, bias, spec)?;
    let input_data = input.data();
    let kernel_data = kernel.data();
    let bias_data = bias.map(Tensor::data);
    let out_row_len = plan.out_w * plan.out_channels;
    if plan.output_len == 0 || out_row_len == 0 {
        return Tensor::from_aligned(
            vec![plan.batch, plan.out_h, plan.out_w, plan.out_channels],
            AlignedVec::<f32>::calloc(plan.output_len),
        )
        .map_err(Into::into);
    }

    let mut output = AlignedVec::<f32>::uninitialized(plan.output_len);

    if should_parallelize_len(plan.output_len, config.min_parallel_elements, thread_pool) {
        let mut work = || {
            output
                .par_chunks_mut(out_row_len)
                .enumerate()
                .for_each(|(row_idx, out_row)| {
                    depthwise_conv2d_nhwc_row(
                        input_data,
                        kernel_data,
                        bias_data,
                        plan,
                        row_idx,
                        out_row,
                    );
                });
        };
        if let Some(pool) = thread_pool {
            pool.install(work);
        } else {
            work();
        }
    } else {
        for (row_idx, out_row) in output.chunks_mut(out_row_len).enumerate() {
            depthwise_conv2d_nhwc_row(input_data, kernel_data, bias_data, plan, row_idx, out_row);
        }
    }

    Tensor::from_aligned(
        vec![plan.batch, plan.out_h, plan.out_w, plan.out_channels],
        output,
    )
    .map_err(Into::into)
}

pub fn separable_conv2d_nhwc_with_config_and_pool(
    input: &Tensor,
    kernels: SeparableConv2dKernels<'_>,
    spec: SeparableConv2dSpec,
    config: ParallelElementwiseConfig,
    thread_pool: Option<&ThreadPool>,
) -> Result<Tensor, KernelError> {
    if kernels.pointwise_kernel.rank() != 4
        || kernels.pointwise_kernel.shape()[0] != 1
        || kernels.pointwise_kernel.shape()[1] != 1
    {
        return Err(KernelError::InvalidSeparablePointwiseKernelShape {
            pointwise_shape: kernels.pointwise_kernel.shape().to_vec(),
        });
    }

    let depthwise_out = depthwise_conv2d_nhwc_with_config_and_pool(
        input,
        kernels.depthwise_kernel,
        kernels.depthwise_bias,
        DepthwiseConv2dSpec {
            stride_h: spec.stride_h,
            stride_w: spec.stride_w,
        },
        config,
        thread_pool,
    )?;

    conv2d_nhwc_with_config_and_pool(
        &depthwise_out,
        kernels.pointwise_kernel,
        kernels.pointwise_bias,
        Conv2dSpec {
            stride_h: 1,
            stride_w: 1,
        },
        config,
        thread_pool,
    )
}

fn build_conv2d_plan(
    input: &Tensor,
    kernel: &Tensor,
    bias: Option<&Tensor>,
    spec: Conv2dSpec,
) -> Result<Conv2dPlan, KernelError> {
    let stride_h = spec.stride_h;
    let stride_w = spec.stride_w;
    if input.rank() != 4 || kernel.rank() != 4 {
        return Err(KernelError::InvalidConvRank {
            input_rank: input.rank(),
            kernel_rank: kernel.rank(),
        });
    }
    if stride_h == 0 || stride_w == 0 {
        return Err(KernelError::InvalidConvParameters {
            kernel_h: kernel.shape()[0],
            kernel_w: kernel.shape()[1],
            stride_h,
            stride_w,
        });
    }

    let batch = input.shape()[0];
    let in_h = input.shape()[1];
    let in_w = input.shape()[2];
    let in_channels = input.shape()[3];
    let kernel_h = kernel.shape()[0];
    let kernel_w = kernel.shape()[1];
    let kernel_in_channels = kernel.shape()[2];
    let out_channels = kernel.shape()[3];

    if kernel_h == 0 || kernel_w == 0 {
        return Err(KernelError::InvalidConvParameters {
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
        });
    }
    if kernel_in_channels != in_channels {
        return Err(KernelError::ConvChannelMismatch {
            input_channels: in_channels,
            kernel_in_channels,
        });
    }
    if kernel_h > in_h || kernel_w > in_w {
        return Err(KernelError::ConvKernelLargerThanInput {
            input_h: in_h,
            input_w: in_w,
            kernel_h,
            kernel_w,
        });
    }
    if let Some(bias_tensor) = bias
        && (bias_tensor.rank() != 1 || bias_tensor.shape()[0] != out_channels)
    {
        return Err(KernelError::ConvBiasShapeMismatch {
            bias_shape: bias_tensor.shape().to_vec(),
            out_channels,
        });
    }

    let out_h = (in_h - kernel_h) / stride_h + 1;
    let out_w = (in_w - kernel_w) / stride_w + 1;
    let output_len = batch
        .checked_mul(out_h)
        .and_then(|v| v.checked_mul(out_w))
        .and_then(|v| v.checked_mul(out_channels))
        .ok_or_else(|| {
            KernelError::Tensor(TensorError::SizeOverflow {
                shape: vec![batch, out_h, out_w, out_channels],
            })
        })?;

    Ok(Conv2dPlan {
        batch,
        in_h,
        in_w,
        in_channels,
        out_h,
        out_w,
        out_channels,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        output_len,
    })
}

fn build_depthwise_conv2d_plan(
    input: &Tensor,
    kernel: &Tensor,
    bias: Option<&Tensor>,
    spec: DepthwiseConv2dSpec,
) -> Result<DepthwiseConv2dPlan, KernelError> {
    let stride_h = spec.stride_h;
    let stride_w = spec.stride_w;
    if input.rank() != 4 || kernel.rank() != 4 {
        return Err(KernelError::InvalidDepthwiseConvRank {
            input_rank: input.rank(),
            kernel_rank: kernel.rank(),
        });
    }
    if stride_h == 0 || stride_w == 0 {
        return Err(KernelError::InvalidDepthwiseConvParameters {
            kernel_h: kernel.shape()[0],
            kernel_w: kernel.shape()[1],
            stride_h,
            stride_w,
        });
    }

    let batch = input.shape()[0];
    let in_h = input.shape()[1];
    let in_w = input.shape()[2];
    let channels = input.shape()[3];
    let kernel_h = kernel.shape()[0];
    let kernel_w = kernel.shape()[1];
    let kernel_channels = kernel.shape()[2];
    let depth_multiplier = kernel.shape()[3];

    if kernel_h == 0 || kernel_w == 0 || depth_multiplier == 0 {
        return Err(KernelError::InvalidDepthwiseConvParameters {
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
        });
    }
    if kernel_channels != channels {
        return Err(KernelError::DepthwiseConvChannelMismatch {
            input_channels: channels,
            kernel_channels,
        });
    }
    if kernel_h > in_h || kernel_w > in_w {
        return Err(KernelError::DepthwiseConvKernelLargerThanInput {
            input_h: in_h,
            input_w: in_w,
            kernel_h,
            kernel_w,
        });
    }

    let out_channels = channels.checked_mul(depth_multiplier).ok_or_else(|| {
        KernelError::Tensor(TensorError::SizeOverflow {
            shape: vec![channels, depth_multiplier],
        })
    })?;
    if let Some(bias_tensor) = bias
        && (bias_tensor.rank() != 1 || bias_tensor.shape()[0] != out_channels)
    {
        return Err(KernelError::DepthwiseConvBiasShapeMismatch {
            bias_shape: bias_tensor.shape().to_vec(),
            out_channels,
        });
    }

    let out_h = (in_h - kernel_h) / stride_h + 1;
    let out_w = (in_w - kernel_w) / stride_w + 1;
    let output_len = batch
        .checked_mul(out_h)
        .and_then(|v| v.checked_mul(out_w))
        .and_then(|v| v.checked_mul(out_channels))
        .ok_or_else(|| {
            KernelError::Tensor(TensorError::SizeOverflow {
                shape: vec![batch, out_h, out_w, out_channels],
            })
        })?;

    Ok(DepthwiseConv2dPlan {
        batch,
        in_h,
        in_w,
        channels,
        depth_multiplier,
        out_h,
        out_w,
        out_channels,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        output_len,
    })
}

fn conv2d_nhwc_row(
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

        // Initialize with bias
        if let Some(bias_values) = bias {
            out_slice.copy_from_slice(&bias_values[..plan.out_channels]);
        } else {
            out_slice.fill(0.0);
        }

        // Accumulate: iterate over kernel window, broadcast input, FMA across out_channels
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
                    // SIMD: broadcast input_val, multiply-add across out_channels
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
unsafe fn conv2d_3x3_direct_neon(
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
unsafe fn conv2d_3x3_direct_avx(
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
unsafe fn conv2d_3x3_direct_sse(
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

/// FMA: out[i] += kernel[i] * input_val, SIMD-accelerated
#[allow(unsafe_code)]
fn conv_fma_row(out: &mut [f32], kernel: &[f32], input_val: f32) {
    let len = out.len();
    debug_assert_eq!(len, kernel.len());

    if cfg!(miri) || len < 4 {
        for i in 0..len {
            out[i] += kernel[i] * input_val;
        }
        return;
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe { conv_fma_neon(out, kernel, input_val) };
            return;
        }
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("avx") {
            unsafe { conv_fma_avx(out, kernel, input_val) };
            return;
        }
        if std::is_x86_feature_detected!("sse") {
            unsafe { conv_fma_sse(out, kernel, input_val) };
            return;
        }
    }

    for i in 0..len {
        out[i] += kernel[i] * input_val;
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
    let k_spatial = kd * kh * kw;
    let col_k = k_spatial * c_in; // im2col column length

    // im2col + BLAS path: reshape 3D conv into matrix multiply
    // im2col: [out_spatial, kd*kh*kw*c_in]
    // kernel reshaped: [kd*kh*kw*c_in, c_out]
    // output = im2col @ kernel_2d → [out_spatial, c_out]
    #[cfg(feature = "blas")]
    let use_blas = !cfg!(miri) && batch == 1;
    #[cfg(not(feature = "blas"))]
    let use_blas = false;

    if use_blas {
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
            super::matmul::blas_sgemm(
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
) {
    use core::arch::aarch64::*;

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

    for out_x in 0..plan.out_w {
        let in_x0 = out_x * plan.stride_w;
        let out_base = out_x * channels;

        // Process 4 channels at a time with accumulator kept in a register.
        // This avoids the load-modify-store per kernel position — the
        // accumulator stays in a NEON register across all kh×kw positions.
        let mut ch = 0;
        while ch + 4 <= simd_end {
            // SAFETY: ch + 4 <= channels, all offsets bounded by plan dims.
            unsafe {
                let mut acc = if let Some(b) = bias {
                    vld1q_f32(b.as_ptr().add(ch))
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
                        let inp = vld1q_f32(inp_ptr.add(in_off));
                        let ker = vld1q_f32(ker_ptr.add(k_off));
                        acc = vfmaq_f32(acc, inp, ker);
                    }
                }

                vst1q_f32(out_ptr.add(out_base + ch), acc);
            }
            ch += 4;
        }
        // Scalar tail for remaining channels.
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
            out_row[out_base + ch] = acc;
            ch += 1;
        }
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
    let simd_end = channels & !7;
    let kh = plan.kernel_h;
    let kw = plan.kernel_w;

    let inp_ptr = input.as_ptr();
    let ker_ptr = kernel.as_ptr();
    let out_ptr = out_row.as_mut_ptr();

    for out_x in 0..plan.out_w {
        let in_x0 = out_x * plan.stride_w;
        let out_base = out_x * channels;

        // Process 8 channels at a time with accumulator in register.
        let mut ch = 0;
        while ch + 8 <= simd_end {
            // SAFETY: ch + 8 <= channels, all offsets bounded by plan dims.
            unsafe {
                let mut acc = if let Some(b) = bias {
                    _mm256_loadu_ps(b.as_ptr().add(ch))
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
                        let inp = _mm256_loadu_ps(inp_ptr.add(in_off));
                        let ker = _mm256_loadu_ps(ker_ptr.add(k_off));
                        acc = _mm256_fmadd_ps(inp, ker, acc);
                    }
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
            out_row[out_base + ch] = acc;
            ch += 1;
        }
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
    let simd_end = channels & !7; // round down to multiple of 8
    let kh = plan.kernel_h;
    let kw = plan.kernel_w;

    let inp_ptr = input.as_ptr();
    let ker_ptr = kernel.as_ptr();
    let out_ptr = out_row.as_mut_ptr();

    for out_x in 0..plan.out_w {
        let in_x0 = out_x * plan.stride_w;
        let out_base = out_x * channels;

        // Process 8 channels at a time with accumulator in register.
        let mut ch = 0;
        while ch + 8 <= simd_end {
            // SAFETY: ch + 8 <= channels, all offsets bounded by plan dims.
            unsafe {
                let mut acc = if let Some(b) = bias {
                    _mm256_loadu_ps(b.as_ptr().add(ch))
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
                        let inp = _mm256_loadu_ps(inp_ptr.add(in_off));
                        let ker = _mm256_loadu_ps(ker_ptr.add(k_off));
                        acc = _mm256_add_ps(acc, _mm256_mul_ps(inp, ker));
                    }
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
            out_row[out_base + ch] = acc;
            ch += 1;
        }
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
                    _mm_setzero_ps()
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
            out_row[out_base + ch] = acc;
            ch += 1;
        }
    }
}

fn depthwise_conv2d_nhwc_row(
    input: &[f32],
    kernel: &[f32],
    bias: Option<&[f32]>,
    plan: DepthwiseConv2dPlan,
    row_idx: usize,
    out_row: &mut [f32],
) {
    // SIMD fast path for depth_multiplier == 1 (standard depthwise conv).
    // When dm=1, out_channels == channels and the kernel layout simplifies to
    // [KH, KW, C] — contiguous channel data enables vectorization.
    if plan.depth_multiplier == 1 && plan.out_channels >= 4 && !cfg!(miri) {
        #[cfg(target_arch = "aarch64")]
        if std::arch::is_aarch64_feature_detected!("neon") {
            // SAFETY: NEON detected, pointers bounded by plan dimensions validated at
            // function entry. Each output element written exactly once.
            #[allow(unsafe_code)]
            unsafe {
                depthwise_conv2d_nhwc_row_neon(input, kernel, bias, plan, row_idx, out_row);
            }
            return;
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if is_x86_feature_detected!("avx") && is_x86_feature_detected!("fma") {
                // SAFETY: AVX+FMA detected, same bounds guarantees.
                #[allow(unsafe_code)]
                unsafe {
                    depthwise_conv2d_nhwc_row_avx_fma(input, kernel, bias, plan, row_idx, out_row);
                }
                return;
            }
            if is_x86_feature_detected!("avx") {
                // SAFETY: AVX detected (no FMA), same bounds guarantees.
                #[allow(unsafe_code)]
                unsafe {
                    depthwise_conv2d_nhwc_row_avx(input, kernel, bias, plan, row_idx, out_row);
                }
                return;
            }
            if is_x86_feature_detected!("sse") {
                // SAFETY: SSE detected, same bounds guarantees.
                #[allow(unsafe_code)]
                unsafe {
                    depthwise_conv2d_nhwc_row_sse(input, kernel, bias, plan, row_idx, out_row);
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

            out_row[out_cell_base + out_channel] = acc;
        }
    }
}

// ---------------------------------------------------------------------------
// im2col + BLAS GEMM fast path for conv2d
// ---------------------------------------------------------------------------

/// Flatten each [kH, kW, C_in] input patch into a row of the im2col matrix.
///
/// Output `col` has shape [out_h * out_w, kH * kW * C_in] (row-major).
/// The input is NHWC layout (batch dimension already stripped by caller).
#[cfg(feature = "blas")]
/// im2col for NHWC input without padding.
/// Uses unsafe pointer arithmetic to avoid per-element bounds checks.
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
/// output: [M, N] which maps directly to [1, out_h, out_w, C_out]
///
/// For large spatial outputs, tiles along M and runs tiles in parallel via
/// rayon.  Each tile materialises only its own im2col slice, keeping the
/// working set in L2 cache (~2 MB per thread).
#[cfg(feature = "blas")]
fn conv2d_im2col_gemm(
    plan: &Conv2dPlan,
    input: &[f32],
    kernel: &[f32],
    bias: Option<&[f32]>,
) -> Result<Tensor, KernelError> {
    let out_h = plan.out_h;
    let out_w = plan.out_w;
    let k = plan.kernel_h * plan.kernel_w * plan.in_channels;
    let m = out_h * out_w;
    let n = plan.out_channels;

    // For 1×1 conv with stride 1, input IS the im2col matrix — zero-copy.
    if plan.kernel_h == 1 && plan.kernel_w == 1 && plan.stride_h == 1 && plan.stride_w == 1 {
        // SAFETY: blas_sgemm with beta=0 writes every element of `output`.
        #[allow(unsafe_code)]
        let mut output = AlignedVec::<f32>::uninitialized(m * n);
        super::matmul::blas_sgemm(&input[..m * k], kernel, &mut output, m, k, n);
        if let Some(bias) = bias {
            add_bias_nhwc(&mut output, bias, m, n);
        }
        return Tensor::from_aligned(vec![1, out_h, out_w, n], output).map_err(Into::into);
    }

    // Tile size: keep im2col_tile + output_tile in ~2 MB per thread.
    // M1: 12 MB shared L2 across 4 P-cores, ~3 MB effective per core.
    let bytes_per_row = (k + n) * std::mem::size_of::<f32>();
    let tile_m = if bytes_per_row > 0 {
        ((2 * 1024 * 1024) / bytes_per_row).max(1).min(m)
    } else {
        m
    };

    // SAFETY: every element is written by blas_sgemm (beta=0) + bias add.
    #[allow(unsafe_code)]
    let mut output = AlignedVec::<f32>::uninitialized(m * n);

    if m > tile_m * 2 {
        // ── Parallel tiled im2col + GEMM ──────────────────────────────
        // Each rayon thread reuses a thread-local im2col buffer across
        // Conv calls to avoid repeated allocation/deallocation.
        let out_slice: &mut [f32] = &mut output;
        out_slice
            .par_chunks_mut(tile_m * n)
            .enumerate()
            .for_each(|(tile_idx, out_chunk)| {
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
                    super::matmul::blas_sgemm(&col_buf[..needed], kernel, out_chunk, actual_m, k, n);
                    if let Some(bias) = bias {
                        add_bias_nhwc(out_chunk, bias, actual_m, n);
                    }
                });
            });
    } else {
        // ── Single tile — small spatial ───────────────────────────────
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
            super::matmul::blas_sgemm(&col_buf[..needed], kernel, &mut output, m, k, n);
            if let Some(bias) = bias {
                add_bias_nhwc(&mut output, bias, m, n);
            }
        });
    }

    Tensor::from_aligned(vec![1, out_h, out_w, n], output).map_err(Into::into)
}

/// Add per-channel bias to NHWC output: output[row, c] += bias[c].
/// Uses unsafe pointer arithmetic to avoid bounds-check overhead in the inner loop.
#[allow(unsafe_code)]
fn add_bias_nhwc(output: &mut [f32], bias: &[f32], m: usize, n: usize) {
    debug_assert!(output.len() >= m * n);
    debug_assert!(bias.len() >= n);
    // SAFETY: m*n <= output.len(), n <= bias.len() verified by callers.
    unsafe {
        let out_ptr = output.as_mut_ptr();
        let bias_ptr = bias.as_ptr();
        for row in 0..m {
            let row_ptr = out_ptr.add(row * n);
            for c in 0..n {
                *row_ptr.add(c) += *bias_ptr.add(c);
            }
        }
    }
}

/// Fused bias + SiLU in a single SIMD pass: output[i] = silu(output[i] + bias[i%n]).
/// Saves one full read+write pass over the output tile compared to separate
/// add_bias_nhwc + silu_slice_inplace calls.
#[inline]
fn add_bias_silu_nhwc(output: &mut [f32], bias: &[f32], m: usize, n: usize) {
    super::simd::bias_silu_nhwc_dispatch(output, bias, m, n);
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
#[cfg(all(feature = "blas", not(target_os = "macos")))]
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

/// Winograd input transform: B^T * d * B for one 4×4 tile.
///
/// B^T = [[1,0,-1,0],[0,1,1,0],[0,-1,1,0],[0,1,0,-1]]
#[cfg(all(feature = "blas", not(target_os = "macos")))]
#[inline]
fn winograd_input_tile(d: &[f32; 16], out: &mut [f32; 16]) {
    // B^T * d → 4×4 intermediate (rows transformed)
    let mut bd = [0.0f32; 16];
    for col in 0..4 {
        bd[col] = d[col] - d[2 * 4 + col];
        bd[4 + col] = d[4 + col] + d[2 * 4 + col];
        bd[8 + col] = -d[4 + col] + d[2 * 4 + col];
        bd[12 + col] = d[4 + col] - d[3 * 4 + col];
    }
    // (B^T * d) * B → 4×4 (columns transformed)
    for row in 0..4 {
        let r = row * 4;
        out[r] = bd[r] - bd[r + 2];
        out[r + 1] = bd[r + 1] + bd[r + 2];
        out[r + 2] = -bd[r + 1] + bd[r + 2];
        out[r + 3] = bd[r + 1] - bd[r + 3];
    }
}

/// Winograd output transform: A^T * m * A, yielding 2×2 output from 4×4 product.
///
/// A^T = [[1,1,1,0],[0,1,-1,-1]]
#[cfg(all(feature = "blas", not(target_os = "macos")))]
#[inline]
fn winograd_output_tile(m: &[f32; 16], out: &mut [f32; 4]) {
    // A^T * m → 2×4 intermediate (rows transformed)
    let mut am = [0.0f32; 8];
    for col in 0..4 {
        am[col] = m[col] + m[4 + col] + m[8 + col];
        am[4 + col] = m[4 + col] - m[8 + col] - m[12 + col];
    }
    // (A^T * m) * A → 2×2 (columns transformed)
    out[0] = am[0] + am[1] + am[2];
    out[1] = am[1] - am[2] - am[3];
    out[2] = am[4] + am[5] + am[6];
    out[3] = am[5] - am[6] - am[7];
}

/// Full Winograd F(2×2, 3×3) convolution for NHWC layout.
///
/// Only valid for 3×3 kernels with stride=1.
/// `input` NHWC `[batch, H, W, c_in]` (unpadded), `kernel` `[3, 3, c_in, c_out]`.
#[cfg(all(feature = "blas", not(target_os = "macos")))]
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
    let padded_h = in_h + pad_top + pad_bottom;
    let padded_w = in_w + pad_left + pad_right;
    let out_h = padded_h - 2; // (padded_h - 3) / 1 + 1
    let out_w = padded_w - 2;

    // Number of 2×2 output tiles
    let tile_h = out_h.div_ceil(2);
    let tile_w = out_w.div_ceil(2);
    let n_tiles = tile_h * tile_w;

    // 1. Transform weights: [16, c_in, c_out]
    let u = winograd_transform_weights_f32(kernel, c_in, c_out);

    // SAFETY: every element written by the GEMM + output transform.
    #[allow(unsafe_code)]
    let mut output = AlignedVec::<f32>::uninitialized(batch * out_h * out_w * c_out);

    for b in 0..batch {
        let in_batch = &input[b * in_h * in_w * c_in..(b + 1) * in_h * in_w * c_in];

        // 2. Input transform: for each tile, for each channel, compute B^T * d * B
        //    Result layout: [16, n_tiles, c_in]
        let mut v = vec![0.0f32; 16 * n_tiles * c_in];
        for th in 0..tile_h {
            for tw in 0..tile_w {
                let tile_idx = th * tile_w + tw;
                // Top-left corner of the 4×4 input tile in padded coords
                let py0 = th * 2;
                let px0 = tw * 2;

                for ci in 0..c_in {
                    // Load 4×4 tile with implicit zero-padding
                    let mut d = [0.0f32; 16];
                    for dy in 0..4 {
                        for dx in 0..4 {
                            let iy = (py0 + dy).wrapping_sub(pad_top);
                            let ix = (px0 + dx).wrapping_sub(pad_left);
                            if iy < in_h && ix < in_w {
                                d[dy * 4 + dx] = in_batch[iy * in_w * c_in + ix * c_in + ci];
                            }
                        }
                    }
                    let mut vt = [0.0f32; 16];
                    winograd_input_tile(&d, &mut vt);
                    for a in 0..16 {
                        v[a * n_tiles * c_in + tile_idx * c_in + ci] = vt[a];
                    }
                }
            }
        }

        // 3. Batched GEMM: for each alpha ∈ 0..16,
        //    M[alpha] = V[alpha] * U[alpha]
        //    V[alpha]: [n_tiles, c_in], U[alpha]: [c_in, c_out]
        //    M[alpha]: [n_tiles, c_out]
        let mut m_buf = vec![0.0f32; 16 * n_tiles * c_out];
        for a in 0..16 {
            let v_slice = &v[a * n_tiles * c_in..(a + 1) * n_tiles * c_in];
            let u_slice = &u[a * c_in * c_out..(a + 1) * c_in * c_out];
            let m_slice = &mut m_buf[a * n_tiles * c_out..(a + 1) * n_tiles * c_out];
            super::matmul::blas_sgemm(v_slice, u_slice, m_slice, n_tiles, c_in, c_out);
        }

        // 4. Output transform: A^T * M * A → 2×2 output per tile, with bias + activation
        let out_batch = &mut output[b * out_h * out_w * c_out..(b + 1) * out_h * out_w * c_out];
        for th in 0..tile_h {
            for tw in 0..tile_w {
                let tile_idx = th * tile_w + tw;
                let oy0 = th * 2;
                let ox0 = tw * 2;
                // Clamp: last tile row/col may produce fewer than 2 valid outputs
                let valid_h = (out_h - oy0).min(2);
                let valid_w = (out_w - ox0).min(2);

                for co in 0..c_out {
                    // Gather the 4×4 product elements for this (tile, channel)
                    let mut mt = [0.0f32; 16];
                    for a in 0..16 {
                        mt[a] = m_buf[a * n_tiles * c_out + tile_idx * c_out + co];
                    }
                    let mut otile = [0.0f32; 4];
                    winograd_output_tile(&mt, &mut otile);

                    // Add bias
                    let bias_val = bias.map_or(0.0, |bd| bd[co]);
                    for dy in 0..valid_h {
                        for dx in 0..valid_w {
                            let idx = (oy0 + dy) * out_w * c_out + (ox0 + dx) * c_out + co;
                            out_batch[idx] = otile[dy * 2 + dx] + bias_val;
                        }
                    }
                }
            }
        }

        // Apply activation on the whole batch output
        match activation {
            Activation::Silu => silu_slice_inplace(out_batch),
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
#[cfg(feature = "blas")]
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

    // Winograd F(2×2,3×3): on non-Apple platforms, use Winograd for 3×3 stride-1
    // convolutions with enough spatial output to amortise the transform overhead.
    // On macOS, Apple Accelerate's AMX-backed sgemm makes im2col+GEMM faster.
    #[cfg(not(target_os = "macos"))]
    if kh == 3 && kw == 3 && stride_h == 1 && stride_w == 1 && out_h * out_w >= 64 {
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

    // Tile size: keep im2col_tile + output_tile in ~2 MB per thread.
    let bytes_per_row = (k + n) * std::mem::size_of::<f32>();
    let tile_m = if bytes_per_row > 0 {
        ((2 * 1024 * 1024) / bytes_per_row).max(1).min(m)
    } else {
        m
    };

    // SAFETY: every element is written by blas_sgemm (beta=0) + bias add.
    #[allow(unsafe_code)]
    let mut output = AlignedVec::<f32>::uninitialized(batch * m * n);

    for b in 0..batch {
        let in_slice = &in_data[b * in_h * in_w * c_in..(b + 1) * in_h * in_w * c_in];
        let out_batch = &mut output[b * m * n..(b + 1) * m * n];

        if m > tile_m * 2 {
            // Parallel tiled im2col + GEMM
            out_batch
                .par_chunks_mut(tile_m * n)
                .enumerate()
                .for_each(|(tile_idx, out_chunk)| {
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
                        super::matmul::blas_sgemm(
                            &col_buf[..needed],
                            kernel_data,
                            out_chunk,
                            actual_m,
                            k,
                            n,
                        );
                        match (bias_data, activation) {
                            (Some(bd), Activation::Silu) => {
                                add_bias_silu_nhwc(out_chunk, bd, actual_m, n);
                            }
                            (Some(bd), _) => add_bias_nhwc(out_chunk, bd, actual_m, n),
                            (None, Activation::Silu) => silu_slice_inplace(out_chunk),
                            (None, _) => {}
                        }
                    });
                });
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
                super::matmul::blas_sgemm(&col_buf[..needed], kernel_data, out_batch, m, k, n);
                match (bias_data, activation) {
                    (Some(bd), Activation::Silu) => {
                        add_bias_silu_nhwc(out_batch, bd, m, n);
                    }
                    (Some(bd), _) => add_bias_nhwc(out_batch, bd, m, n),
                    (None, Activation::Silu) => silu_slice_inplace(out_batch),
                    (None, _) => {}
                }
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
