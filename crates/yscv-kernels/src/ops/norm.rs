use rayon::ThreadPool;
use yscv_tensor::{AlignedVec, Tensor, TensorError};

use super::super::error::KernelError;
use super::config::{
    BatchNorm2dPlan, BatchNorm2dTensors, GroupNorm2dPlan, GroupNorm2dTensors,
    LayerNormLastDimTensors, LayerNormPlan, LogSumExpPlan, ParallelElementwiseConfig,
    RmsNormLastDimTensors, RmsNormPlan, SoftmaxPlan, should_parallelize_len,
};
use super::simd::{log_softmax_row_fused_dispatch, softmax_row_fused_dispatch};

pub fn batch_norm2d_nhwc_with_config_and_pool(
    input: &Tensor,
    params: BatchNorm2dTensors<'_>,
    config: ParallelElementwiseConfig,
    thread_pool: Option<&ThreadPool>,
) -> Result<Tensor, KernelError> {
    let plan = build_batch_norm2d_plan(input, params)?;
    let row_len = plan.width * plan.channels;
    if plan.output_len == 0 || row_len == 0 {
        return Tensor::from_vec(
            vec![plan.batch, plan.height, plan.width, plan.channels],
            vec![],
        )
        .map_err(Into::into);
    }
    // SAFETY: batch_norm loop writes every element before read.
    #[allow(unsafe_code)]
    let mut output = AlignedVec::<f32>::uninitialized(plan.output_len);

    let gamma_data = params.gamma.data();
    let beta_data = params.beta.data();
    let mean_data = params.mean.data();
    let variance_data = params.variance.data();
    // WHY precompute scale/shift: reduces per-pixel work to a single FMA (out = x*scale + shift)
    // instead of 5 ops (sub mean, div sqrt, mul gamma, add beta).
    let mut scale = vec![0.0f32; plan.channels];
    let mut shift = vec![0.0f32; plan.channels];
    for channel in 0..plan.channels {
        let denom = variance_data[channel] + params.epsilon;
        if denom <= 0.0 {
            return Err(KernelError::InvalidBatchNormVariance { channel });
        }
        let inv_std = denom.sqrt().recip();
        // scale = gamma / sqrt(var + eps), shift = beta - mean * scale
        let channel_scale = gamma_data[channel] * inv_std;
        scale[channel] = channel_scale;
        shift[channel] = beta_data[channel] - mean_data[channel] * channel_scale;
    }

    let input_data = input.data();
    if should_parallelize_len(plan.output_len, config.min_parallel_elements, thread_pool) {
        super::super::scope_ctx::par_chunks_mut_dispatch(
            output.as_mut_slice(),
            row_len,
            |row_idx, out_row| {
                batch_norm2d_nhwc_row(input_data, plan, row_idx, out_row, &scale, &shift);
            },
        );
    } else {
        for (row_idx, out_row) in output.chunks_mut(row_len).enumerate() {
            batch_norm2d_nhwc_row(input_data, plan, row_idx, out_row, &scale, &shift);
        }
    }

    Tensor::from_aligned(
        vec![plan.batch, plan.height, plan.width, plan.channels],
        output,
    )
    .map_err(Into::into)
}

/// # Safety
/// `AlignedVec::uninitialized` allocates without zeroing. `softmax_last_dim_row`
/// writes every element before anything reads from the buffer.
#[allow(unsafe_code)]
pub fn softmax_last_dim_with_config_and_pool(
    input: &Tensor,
    config: ParallelElementwiseConfig,
    thread_pool: Option<&ThreadPool>,
) -> Result<Tensor, KernelError> {
    let plan = build_softmax_plan(input)?;
    if plan.output_len == 0 || plan.row_len == 0 {
        let output = vec![0.0f32; plan.output_len];
        return Tensor::from_vec(input.shape().to_vec(), output).map_err(Into::into);
    }

    // SAFETY: `softmax_last_dim_row` writes every element before we read.
    let mut output = AlignedVec::<f32>::uninitialized(plan.output_len);

    let input_data = input.data();
    if should_parallelize_len(plan.output_len, config.min_parallel_elements, thread_pool) {
        super::super::scope_ctx::par_chunks_mut_dispatch(
            output.as_mut_slice(),
            plan.row_len,
            |row_idx, out_row| {
                softmax_last_dim_row(input_data, row_idx, out_row);
            },
        );
    } else {
        for (row_idx, out_row) in output.chunks_mut(plan.row_len).enumerate() {
            softmax_last_dim_row(input_data, row_idx, out_row);
        }
    }

    Tensor::from_aligned(input.shape().to_vec(), output).map_err(Into::into)
}

/// # Safety
/// `AlignedVec::uninitialized` allocates without zeroing. `log_softmax_last_dim_row`
/// writes every element before anything reads from the buffer.
#[allow(unsafe_code)]
pub fn log_softmax_last_dim_with_config_and_pool(
    input: &Tensor,
    config: ParallelElementwiseConfig,
    thread_pool: Option<&ThreadPool>,
) -> Result<Tensor, KernelError> {
    let plan = build_log_softmax_plan(input)?;
    if plan.output_len == 0 || plan.row_len == 0 {
        let output = vec![0.0f32; plan.output_len];
        return Tensor::from_vec(input.shape().to_vec(), output).map_err(Into::into);
    }

    // SAFETY: `log_softmax_last_dim_row` writes every element before we read.
    let mut output = AlignedVec::<f32>::uninitialized(plan.output_len);

    let input_data = input.data();
    if should_parallelize_len(plan.output_len, config.min_parallel_elements, thread_pool) {
        super::super::scope_ctx::par_chunks_mut_dispatch(
            output.as_mut_slice(),
            plan.row_len,
            |row_idx, out_row| {
                log_softmax_last_dim_row(input_data, row_idx, out_row);
            },
        );
    } else {
        for (row_idx, out_row) in output.chunks_mut(plan.row_len).enumerate() {
            log_softmax_last_dim_row(input_data, row_idx, out_row);
        }
    }

    Tensor::from_aligned(input.shape().to_vec(), output).map_err(Into::into)
}

pub fn logsumexp_last_dim_with_config_and_pool(
    input: &Tensor,
    config: ParallelElementwiseConfig,
    thread_pool: Option<&ThreadPool>,
) -> Result<Tensor, KernelError> {
    let plan = build_logsumexp_plan(input)?;
    let mut output = vec![0.0f32; plan.output_len];
    if output.is_empty() {
        return Tensor::from_vec(plan.output_shape.clone(), output).map_err(Into::into);
    }
    if plan.row_len == 0 {
        output.fill(f32::NEG_INFINITY);
        return Tensor::from_vec(plan.output_shape.clone(), output).map_err(Into::into);
    }

    let input_data = input.data();
    if should_parallelize_len(plan.output_len, config.min_parallel_elements, thread_pool) {
        // par_iter_mut over `&mut [f32]` → `par_chunks_mut_dispatch` with
        // chunk size 1: each chunk is `&mut [f32; 1]`, giving the same
        // one-element-per-worker semantics.
        super::super::scope_ctx::par_chunks_mut_dispatch(
            output.as_mut_slice(),
            1,
            |row_idx, out| {
                out[0] = logsumexp_last_dim_row(input_data, row_idx, plan.row_len);
            },
        );
    } else {
        for (row_idx, out) in output.iter_mut().enumerate() {
            *out = logsumexp_last_dim_row(input_data, row_idx, plan.row_len);
        }
    }

    Tensor::from_vec(plan.output_shape, output).map_err(Into::into)
}

/// # Safety
/// `AlignedVec::uninitialized` allocates without zeroing. `layer_norm_last_dim_row`
/// writes every element before anything reads from the buffer.
#[allow(unsafe_code)]
pub fn layer_norm_last_dim_with_config_and_pool(
    input: &Tensor,
    params: LayerNormLastDimTensors<'_>,
    config: ParallelElementwiseConfig,
    thread_pool: Option<&ThreadPool>,
) -> Result<Tensor, KernelError> {
    let plan = build_layer_norm_plan(input, params)?;
    if plan.output_len == 0 || plan.row_len == 0 {
        let output = vec![0.0f32; plan.output_len];
        return Tensor::from_vec(input.shape().to_vec(), output).map_err(Into::into);
    }

    // SAFETY: `layer_norm_last_dim_row` writes every element before we read.
    let mut output = AlignedVec::<f32>::uninitialized(plan.output_len);

    let input_data = input.data();
    let gamma_data = params.gamma.data();
    let beta_data = params.beta.data();

    // For small inputs (< 16K elements), skip threading — rayon overhead dominates.
    let use_parallel = plan.output_len >= 16384
        && should_parallelize_len(plan.output_len, config.min_parallel_elements, thread_pool);

    if use_parallel {
        super::super::scope_ctx::par_chunks_mut_dispatch(
            output.as_mut_slice(),
            plan.row_len,
            |row_idx, out_row| {
                layer_norm_last_dim_row(
                    input_data,
                    row_idx,
                    out_row,
                    gamma_data,
                    beta_data,
                    params.epsilon,
                );
            },
        );
    } else {
        for (row_idx, out_row) in output.chunks_mut(plan.row_len).enumerate() {
            layer_norm_last_dim_row(
                input_data,
                row_idx,
                out_row,
                gamma_data,
                beta_data,
                params.epsilon,
            );
        }
    }

    Tensor::from_aligned(input.shape().to_vec(), output).map_err(Into::into)
}

fn build_batch_norm2d_plan(
    input: &Tensor,
    params: BatchNorm2dTensors<'_>,
) -> Result<BatchNorm2dPlan, KernelError> {
    if input.rank() != 4 {
        return Err(KernelError::InvalidBatchNormRank {
            got_rank: input.rank(),
        });
    }
    if !params.epsilon.is_finite() || params.epsilon <= 0.0 {
        return Err(KernelError::InvalidBatchNormEpsilon);
    }

    let batch = input.shape()[0];
    let height = input.shape()[1];
    let width = input.shape()[2];
    let channels = input.shape()[3];
    validate_batch_norm_parameter("gamma", params.gamma, channels)?;
    validate_batch_norm_parameter("beta", params.beta, channels)?;
    validate_batch_norm_parameter("mean", params.mean, channels)?;
    validate_batch_norm_parameter("variance", params.variance, channels)?;

    let output_len = batch
        .checked_mul(height)
        .and_then(|value| value.checked_mul(width))
        .and_then(|value| value.checked_mul(channels))
        .ok_or_else(|| {
            KernelError::Tensor(TensorError::SizeOverflow {
                shape: vec![batch, height, width, channels],
            })
        })?;

    Ok(BatchNorm2dPlan {
        batch,
        height,
        width,
        channels,
        output_len,
    })
}

fn build_softmax_plan(input: &Tensor) -> Result<SoftmaxPlan, KernelError> {
    if input.rank() == 0 {
        return Err(KernelError::InvalidSoftmaxRank {
            got_rank: input.rank(),
        });
    }
    let row_len = input.shape()[input.rank() - 1];
    let output_len = input.len();
    Ok(SoftmaxPlan {
        row_len,
        output_len,
    })
}

fn build_log_softmax_plan(input: &Tensor) -> Result<SoftmaxPlan, KernelError> {
    if input.rank() == 0 {
        return Err(KernelError::InvalidLogSoftmaxRank {
            got_rank: input.rank(),
        });
    }
    let row_len = input.shape()[input.rank() - 1];
    let output_len = input.len();
    Ok(SoftmaxPlan {
        row_len,
        output_len,
    })
}

fn build_logsumexp_plan(input: &Tensor) -> Result<LogSumExpPlan, KernelError> {
    if input.rank() == 0 {
        return Err(KernelError::InvalidLogSumExpRank {
            got_rank: input.rank(),
        });
    }

    let row_len = input.shape()[input.rank() - 1];
    let mut output_shape = input.shape().to_vec();
    output_shape[input.rank() - 1] = 1;

    let mut output_len = 1usize;
    for &dim in &output_shape {
        output_len = output_len.checked_mul(dim).ok_or_else(|| {
            KernelError::Tensor(TensorError::SizeOverflow {
                shape: output_shape.clone(),
            })
        })?;
    }

    Ok(LogSumExpPlan {
        row_len,
        output_shape,
        output_len,
    })
}

fn build_layer_norm_plan(
    input: &Tensor,
    params: LayerNormLastDimTensors<'_>,
) -> Result<LayerNormPlan, KernelError> {
    if input.rank() == 0 {
        return Err(KernelError::InvalidLayerNormRank {
            got_rank: input.rank(),
        });
    }
    if !params.epsilon.is_finite() || params.epsilon <= 0.0 {
        return Err(KernelError::InvalidLayerNormEpsilon);
    }
    let row_len = input.shape()[input.rank() - 1];
    validate_layer_norm_parameter("gamma", params.gamma, row_len)?;
    validate_layer_norm_parameter("beta", params.beta, row_len)?;
    Ok(LayerNormPlan {
        row_len,
        output_len: input.len(),
    })
}

fn validate_batch_norm_parameter(
    parameter: &'static str,
    tensor: &Tensor,
    expected_channels: usize,
) -> Result<(), KernelError> {
    if tensor.rank() != 1 || tensor.shape()[0] != expected_channels {
        return Err(KernelError::BatchNormParameterShapeMismatch {
            parameter,
            shape: tensor.shape().to_vec(),
            expected_channels,
        });
    }
    Ok(())
}

fn validate_layer_norm_parameter(
    parameter: &'static str,
    tensor: &Tensor,
    expected_features: usize,
) -> Result<(), KernelError> {
    if tensor.rank() != 1 || tensor.shape()[0] != expected_features {
        return Err(KernelError::LayerNormParameterShapeMismatch {
            parameter,
            shape: tensor.shape().to_vec(),
            expected_features,
        });
    }
    Ok(())
}

#[allow(unsafe_code)]
fn batch_norm2d_nhwc_row(
    input: &[f32],
    plan: BatchNorm2dPlan,
    row_idx: usize,
    out_row: &mut [f32],
    scale: &[f32],
    shift: &[f32],
) {
    let channels = plan.channels;
    let row_len = plan.width * channels;
    let input_row = &input[row_idx * row_len..row_idx * row_len + row_len];

    // SIMD: process pixel-by-pixel, each pixel has `channels` values
    // For channels divisible by 4 (common: 16, 32, 64, 128), use NEON/AVX
    #[cfg(target_arch = "aarch64")]
    if channels >= 4 && std::arch::is_aarch64_feature_detected!("neon") {
        unsafe { batch_norm_row_neon(input_row, out_row, scale, shift, channels) };
        return;
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if channels >= 8 && std::is_x86_feature_detected!("avx") {
        unsafe { batch_norm_row_avx(input_row, out_row, scale, shift, channels) };
        return;
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if channels >= 4 && std::is_x86_feature_detected!("sse2") {
        unsafe { batch_norm_row_sse(input_row, out_row, scale, shift, channels) };
        return;
    }

    // Scalar fallback — iterate per pixel to avoid modulo
    for px in 0..plan.width {
        let base = px * channels;
        for c in 0..channels {
            out_row[base + c] = input_row[base + c] * scale[c] + shift[c];
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
unsafe fn batch_norm_row_neon(
    input: &[f32],
    output: &mut [f32],
    scale: &[f32],
    shift: &[f32],
    channels: usize,
) {
    use std::arch::aarch64::*;
    let total = input.len();
    let inp = input.as_ptr();
    let outp = output.as_mut_ptr();
    let mut i = 0usize;
    // Process 4 channels at a time within each pixel
    while i + 4 <= total {
        let c = i % channels;
        let v = vld1q_f32(inp.add(i));
        let s = vld1q_f32(scale.as_ptr().add(c));
        let sh = vld1q_f32(shift.as_ptr().add(c));
        let r = vfmaq_f32(sh, v, s); // sh + v * s
        vst1q_f32(outp.add(i), r);
        i += 4;
    }
    while i < total {
        let c = i % channels;
        *outp.add(i) = *inp.add(i) * *scale.as_ptr().add(c) + *shift.as_ptr().add(c);
        i += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
unsafe fn batch_norm_row_avx(
    input: &[f32],
    output: &mut [f32],
    scale: &[f32],
    shift: &[f32],
    channels: usize,
) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;
    let total = input.len();
    let inp = input.as_ptr();
    let outp = output.as_mut_ptr();
    let mut i = 0usize;
    while i + 8 <= total {
        let c = i % channels;
        let v = _mm256_loadu_ps(inp.add(i));
        let s = _mm256_loadu_ps(scale.as_ptr().add(c));
        let sh = _mm256_loadu_ps(shift.as_ptr().add(c));
        let r = _mm256_add_ps(_mm256_mul_ps(v, s), sh);
        _mm256_storeu_ps(outp.add(i), r);
        i += 8;
    }
    while i < total {
        let c = i % channels;
        *outp.add(i) = *inp.add(i) * *scale.as_ptr().add(c) + *shift.as_ptr().add(c);
        i += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
unsafe fn batch_norm_row_sse(
    input: &[f32],
    output: &mut [f32],
    scale: &[f32],
    shift: &[f32],
    channels: usize,
) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;
    let total = input.len();
    let inp = input.as_ptr();
    let outp = output.as_mut_ptr();
    let mut i = 0usize;
    while i + 4 <= total {
        let c = i % channels;
        let v = _mm_loadu_ps(inp.add(i));
        let s = _mm_loadu_ps(scale.as_ptr().add(c));
        let sh = _mm_loadu_ps(shift.as_ptr().add(c));
        let r = _mm_add_ps(_mm_mul_ps(v, s), sh);
        _mm_storeu_ps(outp.add(i), r);
        i += 4;
    }
    while i < total {
        let c = i % channels;
        *outp.add(i) = *inp.add(i) * *scale.as_ptr().add(c) + *shift.as_ptr().add(c);
        i += 1;
    }
}

fn softmax_last_dim_row(input: &[f32], row_idx: usize, out_row: &mut [f32]) {
    let row_start = row_idx * out_row.len();
    let row_input = &input[row_start..row_start + out_row.len()];
    // Fused: max + sub-exp + sum + divide in one function, keeping data in L1 cache
    // and eliminating 4 separate dispatch calls.
    softmax_row_fused_dispatch(row_input, out_row);
}

fn log_softmax_last_dim_row(input: &[f32], row_idx: usize, out_row: &mut [f32]) {
    let row_start = row_idx * out_row.len();
    let row_input = &input[row_start..row_start + out_row.len()];
    // Fused: max + sum(exp(x-max)) + subtract in one dispatch, using SIMD where available.
    log_softmax_row_fused_dispatch(row_input, out_row);
}

fn logsumexp_last_dim_row(input: &[f32], row_idx: usize, row_len: usize) -> f32 {
    let row_start = row_idx * row_len;
    let row_input = &input[row_start..row_start + row_len];
    let mut max_value = f32::NEG_INFINITY;
    for &value in row_input {
        max_value = max_value.max(value);
    }

    let mut sum_exp = 0.0f32;
    for &value in row_input {
        sum_exp += (value - max_value).exp();
    }
    max_value + sum_exp.ln()
}

#[allow(unsafe_code)]
fn layer_norm_last_dim_row(
    input: &[f32],
    row_idx: usize,
    out_row: &mut [f32],
    gamma: &[f32],
    beta: &[f32],
    epsilon: f32,
) {
    let row_start = row_idx * out_row.len();
    let row_input = &input[row_start..row_start + out_row.len()];
    let n = row_input.len();
    let row_len = n as f32;

    // SIMD Pass 1: sum and sum-of-squares
    let (sum, sum_sq) = layer_norm_stats(row_input);
    let mean = sum / row_len;
    let variance = sum_sq / row_len - mean * mean;
    let inv_std = (variance + epsilon).sqrt().recip();

    // SIMD Pass 2: normalize + scale + shift
    layer_norm_apply(row_input, out_row, gamma, beta, mean, inv_std);
}

#[allow(unsafe_code)]
fn layer_norm_stats(data: &[f32]) -> (f32, f32) {
    let n = data.len();

    #[cfg(target_arch = "aarch64")]
    if n >= 4 && std::arch::is_aarch64_feature_detected!("neon") {
        return unsafe { layer_norm_stats_neon(data) };
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if n >= 8 && std::is_x86_feature_detected!("avx") {
        return unsafe { layer_norm_stats_avx(data) };
    }

    let mut sum = 0.0f32;
    let mut sum_sq = 0.0f32;
    for &v in data {
        sum += v;
        sum_sq += v * v;
    }
    (sum, sum_sq)
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
unsafe fn layer_norm_stats_neon(data: &[f32]) -> (f32, f32) {
    use std::arch::aarch64::*;
    let mut vsum = vdupq_n_f32(0.0);
    let mut vsq = vdupq_n_f32(0.0);
    let ptr = data.as_ptr();
    let mut i = 0usize;
    let end4 = data.len() & !3;
    while i < end4 {
        let v = vld1q_f32(ptr.add(i));
        vsum = vaddq_f32(vsum, v);
        vsq = vfmaq_f32(vsq, v, v);
        i += 4;
    }
    let mut sum = vaddvq_f32(vsum);
    let mut sum_sq = vaddvq_f32(vsq);
    while i < data.len() {
        let v = *ptr.add(i);
        sum += v;
        sum_sq += v * v;
        i += 1;
    }
    (sum, sum_sq)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
unsafe fn layer_norm_stats_avx(data: &[f32]) -> (f32, f32) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;
    let mut vsum = _mm256_setzero_ps();
    let mut vsq = _mm256_setzero_ps();
    let ptr = data.as_ptr();
    let mut i = 0usize;
    let end8 = data.len() & !7;
    while i < end8 {
        let v = _mm256_loadu_ps(ptr.add(i));
        vsum = _mm256_add_ps(vsum, v);
        vsq = _mm256_add_ps(vsq, _mm256_mul_ps(v, v));
        i += 8;
    }
    // horizontal sum
    let hi = _mm256_extractf128_ps(vsum, 1);
    let lo = _mm256_castps256_ps128(vsum);
    let s4 = _mm_add_ps(lo, hi);
    let s2 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
    let s1 = _mm_add_ss(s2, _mm_movehdup_ps(s2));
    let mut sum = _mm_cvtss_f32(s1);
    let hi2 = _mm256_extractf128_ps(vsq, 1);
    let lo2 = _mm256_castps256_ps128(vsq);
    let q4 = _mm_add_ps(lo2, hi2);
    let q2 = _mm_add_ps(q4, _mm_movehl_ps(q4, q4));
    let q1 = _mm_add_ss(q2, _mm_movehdup_ps(q2));
    let mut sum_sq = _mm_cvtss_f32(q1);
    while i < data.len() {
        let v = *ptr.add(i);
        sum += v;
        sum_sq += v * v;
        i += 1;
    }
    (sum, sum_sq)
}

#[allow(unsafe_code)]
fn layer_norm_apply(
    input: &[f32],
    out: &mut [f32],
    gamma: &[f32],
    beta: &[f32],
    mean: f32,
    inv_std: f32,
) {
    let n = input.len();

    #[cfg(target_arch = "aarch64")]
    if n >= 4 && std::arch::is_aarch64_feature_detected!("neon") {
        unsafe { layer_norm_apply_neon(input, out, gamma, beta, mean, inv_std) };
        return;
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if n >= 8 && std::is_x86_feature_detected!("avx") {
        unsafe { layer_norm_apply_avx(input, out, gamma, beta, mean, inv_std) };
        return;
    }

    for i in 0..n {
        out[i] = (input[i] - mean) * inv_std * gamma[i] + beta[i];
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
unsafe fn layer_norm_apply_neon(
    input: &[f32],
    out: &mut [f32],
    gamma: &[f32],
    beta: &[f32],
    mean: f32,
    inv_std: f32,
) {
    use std::arch::aarch64::*;
    let vmean = vdupq_n_f32(mean);
    let vinv = vdupq_n_f32(inv_std);
    let mut i = 0usize;
    let end4 = input.len() & !3;
    while i < end4 {
        let v = vld1q_f32(input.as_ptr().add(i));
        let g = vld1q_f32(gamma.as_ptr().add(i));
        let b = vld1q_f32(beta.as_ptr().add(i));
        let norm = vmulq_f32(vsubq_f32(v, vmean), vinv);
        let r = vfmaq_f32(b, norm, g);
        vst1q_f32(out.as_mut_ptr().add(i), r);
        i += 4;
    }
    while i < input.len() {
        out[i] = (input[i] - mean) * inv_std * gamma[i] + beta[i];
        i += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
unsafe fn layer_norm_apply_avx(
    input: &[f32],
    out: &mut [f32],
    gamma: &[f32],
    beta: &[f32],
    mean: f32,
    inv_std: f32,
) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;
    let vmean = _mm256_set1_ps(mean);
    let vinv = _mm256_set1_ps(inv_std);
    let mut i = 0usize;
    let end8 = input.len() & !7;
    while i < end8 {
        let v = _mm256_loadu_ps(input.as_ptr().add(i));
        let g = _mm256_loadu_ps(gamma.as_ptr().add(i));
        let b = _mm256_loadu_ps(beta.as_ptr().add(i));
        let norm = _mm256_mul_ps(_mm256_sub_ps(v, vmean), vinv);
        let r = _mm256_add_ps(_mm256_mul_ps(norm, g), b);
        _mm256_storeu_ps(out.as_mut_ptr().add(i), r);
        i += 8;
    }
    while i < input.len() {
        out[i] = (input[i] - mean) * inv_std * gamma[i] + beta[i];
        i += 1;
    }
}

// ---------------------------------------------------------------------------
// GroupNorm NHWC
// ---------------------------------------------------------------------------

pub fn group_norm_nhwc_with_config_and_pool(
    input: &Tensor,
    params: GroupNorm2dTensors<'_>,
    config: ParallelElementwiseConfig,
    thread_pool: Option<&ThreadPool>,
) -> Result<Tensor, KernelError> {
    let plan = build_group_norm_plan(input, params)?;
    let mut output = vec![0.0f32; plan.output_len];
    if output.is_empty() {
        return Tensor::from_vec(
            vec![plan.batch, plan.height, plan.width, plan.channels],
            output,
        )
        .map_err(Into::into);
    }

    let input_data = input.data();
    let gamma_data = params.gamma.data();
    let beta_data = params.beta.data();

    // Process per (sample, group) pair
    let spatial = plan.height * plan.width;
    let cpg = plan.channels_per_group;
    let total_pairs = plan.batch * plan.num_groups;

    let compute_pair = |pair_idx: usize, out: &mut [f32]| {
        let sample = pair_idx / plan.num_groups;
        let group = pair_idx % plan.num_groups;
        let group_channel_start = group * cpg;

        // Compute mean and variance over spatial * cpg elements
        let count = (spatial * cpg) as f32;
        let mut mean = 0.0f32;
        for pos in 0..spatial {
            let base = (sample * spatial + pos) * plan.channels + group_channel_start;
            for gc in 0..cpg {
                mean += input_data[base + gc];
            }
        }
        mean /= count;

        let mut variance = 0.0f32;
        for pos in 0..spatial {
            let base = (sample * spatial + pos) * plan.channels + group_channel_start;
            for gc in 0..cpg {
                let diff = input_data[base + gc] - mean;
                variance += diff * diff;
            }
        }
        variance /= count;
        let inv_std = (variance + params.epsilon).sqrt().recip();

        // Write normalized output
        for pos in 0..spatial {
            let in_base = (sample * spatial + pos) * plan.channels + group_channel_start;
            let out_base = (sample * spatial + pos) * plan.channels + group_channel_start;
            for gc in 0..cpg {
                let c = group_channel_start + gc;
                let normalized = (input_data[in_base + gc] - mean) * inv_std;
                out[out_base + gc] = normalized * gamma_data[c] + beta_data[c];
            }
        }
    };

    if should_parallelize_len(plan.output_len, config.min_parallel_elements, thread_pool) {
        let sample_len = spatial * plan.channels;
        super::super::scope_ctx::par_chunks_mut_dispatch(
            output.as_mut_slice(),
            sample_len,
            |sample, out_sample| {
                for group in 0..plan.num_groups {
                    let group_channel_start = group * cpg;
                    let count = (spatial * cpg) as f32;
                    let mut mean = 0.0f32;
                    for pos in 0..spatial {
                        let base = (sample * spatial + pos) * plan.channels + group_channel_start;
                        for gc in 0..cpg {
                            mean += input_data[base + gc];
                        }
                    }
                    mean /= count;

                    let mut variance = 0.0f32;
                    for pos in 0..spatial {
                        let base = (sample * spatial + pos) * plan.channels + group_channel_start;
                        for gc in 0..cpg {
                            let diff = input_data[base + gc] - mean;
                            variance += diff * diff;
                        }
                    }
                    variance /= count;
                    let inv_std = (variance + params.epsilon).sqrt().recip();

                    for pos in 0..spatial {
                        let in_base =
                            (sample * spatial + pos) * plan.channels + group_channel_start;
                        let out_off = pos * plan.channels + group_channel_start;
                        for gc in 0..cpg {
                            let c = group_channel_start + gc;
                            let normalized = (input_data[in_base + gc] - mean) * inv_std;
                            out_sample[out_off + gc] = normalized * gamma_data[c] + beta_data[c];
                        }
                    }
                }
            },
        );
    } else {
        for pair_idx in 0..total_pairs {
            compute_pair(pair_idx, &mut output);
        }
    }

    Tensor::from_vec(
        vec![plan.batch, plan.height, plan.width, plan.channels],
        output,
    )
    .map_err(Into::into)
}

fn build_group_norm_plan(
    input: &Tensor,
    params: GroupNorm2dTensors<'_>,
) -> Result<GroupNorm2dPlan, KernelError> {
    if input.rank() != 4 {
        return Err(KernelError::InvalidGroupNormRank {
            got_rank: input.rank(),
        });
    }
    if !params.epsilon.is_finite() || params.epsilon <= 0.0 {
        return Err(KernelError::InvalidGroupNormEpsilon);
    }
    if params.num_groups == 0 {
        return Err(KernelError::InvalidGroupNormNumGroups);
    }

    let batch = input.shape()[0];
    let height = input.shape()[1];
    let width = input.shape()[2];
    let channels = input.shape()[3];

    if !channels.is_multiple_of(params.num_groups) {
        return Err(KernelError::GroupNormChannelGroupMismatch {
            channels,
            num_groups: params.num_groups,
        });
    }
    let channels_per_group = channels / params.num_groups;

    validate_group_norm_parameter("gamma", params.gamma, channels)?;
    validate_group_norm_parameter("beta", params.beta, channels)?;

    let output_len = batch
        .checked_mul(height)
        .and_then(|v| v.checked_mul(width))
        .and_then(|v| v.checked_mul(channels))
        .ok_or_else(|| {
            KernelError::Tensor(TensorError::SizeOverflow {
                shape: vec![batch, height, width, channels],
            })
        })?;

    Ok(GroupNorm2dPlan {
        batch,
        height,
        width,
        channels,
        num_groups: params.num_groups,
        channels_per_group,
        output_len,
    })
}

fn validate_group_norm_parameter(
    parameter: &'static str,
    tensor: &Tensor,
    expected_channels: usize,
) -> Result<(), KernelError> {
    if tensor.rank() != 1 || tensor.shape()[0] != expected_channels {
        return Err(KernelError::GroupNormParameterShapeMismatch {
            parameter,
            shape: tensor.shape().to_vec(),
            expected_channels,
        });
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// RMSNorm (last dimension)
// ---------------------------------------------------------------------------

pub fn rms_norm_last_dim_with_config_and_pool(
    input: &Tensor,
    params: RmsNormLastDimTensors<'_>,
    config: ParallelElementwiseConfig,
    thread_pool: Option<&ThreadPool>,
) -> Result<Tensor, KernelError> {
    let plan = build_rms_norm_plan(input, params)?;
    let mut output = vec![0.0f32; plan.output_len];
    if output.is_empty() || plan.row_len == 0 {
        return Tensor::from_vec(input.shape().to_vec(), output).map_err(Into::into);
    }

    let input_data = input.data();
    let gamma_data = params.gamma.data();
    if should_parallelize_len(plan.output_len, config.min_parallel_elements, thread_pool) {
        super::super::scope_ctx::par_chunks_mut_dispatch(
            output.as_mut_slice(),
            plan.row_len,
            |row_idx, out_row| {
                rms_norm_last_dim_row(input_data, row_idx, out_row, gamma_data, params.epsilon);
            },
        );
    } else {
        for (row_idx, out_row) in output.chunks_mut(plan.row_len).enumerate() {
            rms_norm_last_dim_row(input_data, row_idx, out_row, gamma_data, params.epsilon);
        }
    }

    Tensor::from_vec(input.shape().to_vec(), output).map_err(Into::into)
}

fn build_rms_norm_plan(
    input: &Tensor,
    params: RmsNormLastDimTensors<'_>,
) -> Result<RmsNormPlan, KernelError> {
    if input.rank() == 0 {
        return Err(KernelError::InvalidRmsNormRank {
            got_rank: input.rank(),
        });
    }
    if !params.epsilon.is_finite() || params.epsilon <= 0.0 {
        return Err(KernelError::InvalidRmsNormEpsilon);
    }
    let row_len = input.shape()[input.rank() - 1];
    validate_rms_norm_parameter("gamma", params.gamma, row_len)?;
    Ok(RmsNormPlan {
        row_len,
        output_len: input.len(),
    })
}

fn validate_rms_norm_parameter(
    parameter: &'static str,
    tensor: &Tensor,
    expected_features: usize,
) -> Result<(), KernelError> {
    if tensor.rank() != 1 || tensor.shape()[0] != expected_features {
        return Err(KernelError::RmsNormParameterShapeMismatch {
            parameter,
            shape: tensor.shape().to_vec(),
            expected_features,
        });
    }
    Ok(())
}

fn rms_norm_last_dim_row(
    input: &[f32],
    row_idx: usize,
    out_row: &mut [f32],
    gamma: &[f32],
    epsilon: f32,
) {
    let row_start = row_idx * out_row.len();
    let row_input = &input[row_start..row_start + out_row.len()];
    let row_len = row_input.len() as f32;

    let mut sum_sq = 0.0f32;
    for &value in row_input {
        sum_sq += value * value;
    }
    let rms = (sum_sq / row_len + epsilon).sqrt();
    let inv_rms = rms.recip();

    for offset in 0..row_input.len() {
        out_row[offset] = row_input[offset] * inv_rms * gamma[offset];
    }
}
