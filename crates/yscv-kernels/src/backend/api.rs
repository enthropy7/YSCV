//! Backend-agnostic free-function op API: parallel-by-default convenience
//! wrappers over the `ops` kernels, plus the public conv/attention entries.

use yscv_tensor::Tensor;

use super::*;
use crate::core::error::KernelError;
use crate::core::ops::{
    self, BatchNorm2dTensors, Conv2dSpec, DepthwiseConv2dSpec, GroupNorm2dTensors,
    LayerNormLastDimTensors, ParallelElementwiseConfig, ParallelMatmulConfig, Pool2dSpec,
    RmsNormLastDimTensors, SeparableConv2dKernels, SeparableConv2dSpec,
};

/// Backend-agnostic convenience call for add (parallel by default).
pub fn add(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor, KernelError> {
    ops::add_with_config(lhs, rhs, ParallelElementwiseConfig::default())
}

/// Backend-agnostic add with explicit elementwise parallelization heuristics.
pub fn add_with_config(
    lhs: &Tensor,
    rhs: &Tensor,
    config: ParallelElementwiseConfig,
) -> Result<Tensor, KernelError> {
    ops::add_with_config(lhs, rhs, config)
}

/// Backend-agnostic convenience call for sub (parallel by default).
pub fn sub(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor, KernelError> {
    ops::sub_with_config(lhs, rhs, ParallelElementwiseConfig::default())
}

/// Backend-agnostic subtract with explicit elementwise parallelization heuristics.
pub fn sub_with_config(
    lhs: &Tensor,
    rhs: &Tensor,
    config: ParallelElementwiseConfig,
) -> Result<Tensor, KernelError> {
    ops::sub_with_config(lhs, rhs, config)
}

/// Backend-agnostic convenience call for mul (parallel by default).
pub fn mul(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor, KernelError> {
    ops::mul_with_config(lhs, rhs, ParallelElementwiseConfig::default())
}

/// Backend-agnostic multiply with explicit elementwise parallelization heuristics.
pub fn mul_with_config(
    lhs: &Tensor,
    rhs: &Tensor,
    config: ParallelElementwiseConfig,
) -> Result<Tensor, KernelError> {
    ops::mul_with_config(lhs, rhs, config)
}

/// Elementwise ReLU activation (parallel by default).
pub fn relu(input: &Tensor) -> Tensor {
    ops::relu_with_config(input, ParallelElementwiseConfig::default())
}

/// In-place ReLU activation: clamps negative values to zero.
pub fn relu_inplace(tensor: &mut Tensor) {
    ops::relu_inplace(tensor);
}

/// In-place element-wise add: `lhs += rhs`. Same-shape fast path.
pub fn add_inplace(lhs: &mut Tensor, rhs: &Tensor) {
    ops::add_inplace(lhs, rhs);
}

/// Fused in-place add + ReLU: `lhs[i] = max(lhs[i] + rhs[i], 0)`. Single SIMD pass.
pub fn add_relu_inplace(lhs: &mut Tensor, rhs: &Tensor) {
    ops::add_relu_inplace(lhs, rhs);
}

/// Elementwise ReLU with explicit elementwise parallelization heuristics.
pub fn relu_with_config(input: &Tensor, config: ParallelElementwiseConfig) -> Tensor {
    ops::relu_with_config(input, config)
}

/// Elementwise sigmoid activation (parallel by default).
pub fn sigmoid(input: &Tensor) -> Tensor {
    ops::sigmoid_with_config(input, ParallelElementwiseConfig::default())
}

/// Elementwise sigmoid with explicit elementwise parallelization heuristics.
pub fn sigmoid_with_config(input: &Tensor, config: ParallelElementwiseConfig) -> Tensor {
    ops::sigmoid_with_config(input, config)
}

/// Elementwise exp activation (parallel by default).
pub fn exp(input: &Tensor) -> Tensor {
    ops::exp_with_config(input, ParallelElementwiseConfig::default())
}

/// Elementwise exp with explicit elementwise parallelization heuristics.
pub fn exp_with_config(input: &Tensor, config: ParallelElementwiseConfig) -> Tensor {
    ops::exp_with_config(input, config)
}

/// Elementwise tanh activation (parallel by default).
pub fn tanh_act(input: &Tensor) -> Tensor {
    ops::tanh_act_with_config(input, ParallelElementwiseConfig::default())
}

/// Elementwise tanh with explicit elementwise parallelization heuristics.
pub fn tanh_act_with_config(input: &Tensor, config: ParallelElementwiseConfig) -> Tensor {
    ops::tanh_act_with_config(input, config)
}

/// Elementwise GELU activation (fast approximation): `x * sigmoid(1.702 * x)`.
pub fn gelu(input: &Tensor) -> Tensor {
    ops::gelu(input)
}

/// Elementwise SiLU (Swish) activation: `x * sigmoid(x)`.
pub fn silu(input: &Tensor) -> Tensor {
    ops::silu(input)
}

/// In-place SiLU: applies SiLU to a mutable tensor, avoiding allocation.
pub fn silu_inplace(tensor: &mut Tensor) {
    ops::silu_inplace(tensor)
}

/// Elementwise Mish activation: `x * tanh(ln(1 + exp(x)))`.
pub fn mish(input: &Tensor) -> Tensor {
    ops::mish(input)
}

/// Softmax along the last tensor dimension (parallel by default).
pub fn softmax_last_dim(input: &Tensor) -> Result<Tensor, KernelError> {
    ops::softmax_last_dim_with_config_and_pool(input, ParallelElementwiseConfig::default(), None)
}

/// Softmax along the last tensor dimension with explicit elementwise parallelization heuristics.
pub fn softmax_last_dim_with_config(
    input: &Tensor,
    config: ParallelElementwiseConfig,
) -> Result<Tensor, KernelError> {
    ops::softmax_last_dim_with_config_and_pool(input, config, None)
}

/// Log-softmax along the last tensor dimension (parallel by default).
pub fn log_softmax_last_dim(input: &Tensor) -> Result<Tensor, KernelError> {
    ops::log_softmax_last_dim_with_config_and_pool(
        input,
        ParallelElementwiseConfig::default(),
        None,
    )
}

/// Log-softmax along the last tensor dimension with explicit elementwise parallelization heuristics.
pub fn log_softmax_last_dim_with_config(
    input: &Tensor,
    config: ParallelElementwiseConfig,
) -> Result<Tensor, KernelError> {
    ops::log_softmax_last_dim_with_config_and_pool(input, config, None)
}

/// Log-sum-exp reduction along the last tensor dimension.
///
/// Returns shape equal to input with the last dimension set to `1`.
pub fn logsumexp_last_dim(input: &Tensor) -> Result<Tensor, KernelError> {
    ops::logsumexp_last_dim_with_config_and_pool(input, ParallelElementwiseConfig::default(), None)
}

/// Log-sum-exp reduction along the last tensor dimension with explicit elementwise parallelization heuristics.
pub fn logsumexp_last_dim_with_config(
    input: &Tensor,
    config: ParallelElementwiseConfig,
) -> Result<Tensor, KernelError> {
    ops::logsumexp_last_dim_with_config_and_pool(input, config, None)
}

/// Layer normalization over the last tensor dimension (parallel by default).
pub fn layer_norm_last_dim(
    input: &Tensor,
    params: LayerNormLastDimParams<'_>,
) -> Result<Tensor, KernelError> {
    ops::layer_norm_last_dim_with_config_and_pool(
        input,
        LayerNormLastDimTensors {
            gamma: params.gamma,
            beta: params.beta,
            epsilon: params.epsilon,
        },
        ParallelElementwiseConfig::default(),
        None,
    )
}

/// Layer normalization over the last tensor dimension with explicit elementwise parallelization heuristics.
pub fn layer_norm_last_dim_with_config(
    input: &Tensor,
    params: LayerNormLastDimParams<'_>,
    config: ParallelElementwiseConfig,
) -> Result<Tensor, KernelError> {
    ops::layer_norm_last_dim_with_config_and_pool(
        input,
        LayerNormLastDimTensors {
            gamma: params.gamma,
            beta: params.beta,
            epsilon: params.epsilon,
        },
        config,
        None,
    )
}

/// NHWC max-pooling without padding (parallel by default).
pub fn max_pool2d_nhwc(
    input: &Tensor,
    kernel_h: usize,
    kernel_w: usize,
    stride_h: usize,
    stride_w: usize,
) -> Result<Tensor, KernelError> {
    ops::max_pool2d_nhwc_with_config_and_pool(
        input,
        Pool2dSpec {
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
        },
        ParallelElementwiseConfig::default(),
        None,
    )
}

/// NHWC max-pooling without padding with explicit parallelization heuristics.
pub fn max_pool2d_nhwc_with_config(
    input: &Tensor,
    kernel_h: usize,
    kernel_w: usize,
    stride_h: usize,
    stride_w: usize,
    config: ParallelElementwiseConfig,
) -> Result<Tensor, KernelError> {
    ops::max_pool2d_nhwc_with_config_and_pool(
        input,
        Pool2dSpec {
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
        },
        config,
        None,
    )
}

/// NHWC average-pooling without padding (parallel by default).
pub fn avg_pool2d_nhwc(
    input: &Tensor,
    kernel_h: usize,
    kernel_w: usize,
    stride_h: usize,
    stride_w: usize,
) -> Result<Tensor, KernelError> {
    ops::avg_pool2d_nhwc_with_config_and_pool(
        input,
        Pool2dSpec {
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
        },
        ParallelElementwiseConfig::default(),
        None,
    )
}

/// NHWC average-pooling without padding with explicit parallelization heuristics.
pub fn avg_pool2d_nhwc_with_config(
    input: &Tensor,
    kernel_h: usize,
    kernel_w: usize,
    stride_h: usize,
    stride_w: usize,
    config: ParallelElementwiseConfig,
) -> Result<Tensor, KernelError> {
    ops::avg_pool2d_nhwc_with_config_and_pool(
        input,
        Pool2dSpec {
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
        },
        config,
        None,
    )
}

/// NHWC convolution without padding using kernel shape `[KH, KW, C_in, C_out]` (parallel by default).
pub fn conv2d_nhwc(
    input: &Tensor,
    kernel: &Tensor,
    bias: Option<&Tensor>,
    stride_h: usize,
    stride_w: usize,
) -> Result<Tensor, KernelError> {
    conv2d_nhwc_with_activation(
        input,
        kernel,
        bias,
        stride_h,
        stride_w,
        ops::Activation::None,
    )
}

/// NHWC convolution without padding with optional fused activation (parallel by default).
pub fn conv2d_nhwc_with_activation(
    input: &Tensor,
    kernel: &Tensor,
    bias: Option<&Tensor>,
    stride_h: usize,
    stride_w: usize,
    activation: ops::Activation,
) -> Result<Tensor, KernelError> {
    ops::conv2d_nhwc_with_activation_with_config_and_pool(
        input,
        kernel,
        bias,
        Conv2dSpec { stride_h, stride_w },
        activation,
        conv_parallel_config(),
        None,
    )
}

/// NHWC convolution without padding with optional fused activation and an
/// optional pre-packed kernel B (built at model load via
/// `pack_b_for_session`). When `prepacked_b` is `Some` and this is a 1×1
/// pointwise conv, the GEMM layer skips both the fingerprint-cache lookup
/// and the weight-pack itself — saving ~5–15% on tracker-like models where
/// most Convs are static-weight pointwise.
#[allow(clippy::too_many_arguments)]
pub fn conv2d_nhwc_with_activation_prepacked_default(
    input: &Tensor,
    kernel: &Tensor,
    bias: Option<&Tensor>,
    stride_h: usize,
    stride_w: usize,
    activation: ops::Activation,
    prepacked_b: Option<&ops::PackedB>,
) -> Result<Tensor, KernelError> {
    ops::conv2d_nhwc_with_activation_prepacked(
        input,
        kernel,
        bias,
        Conv2dSpec { stride_h, stride_w },
        activation,
        conv_parallel_config(),
        None,
        prepacked_b,
    )
}

/// NHWC convolution with implicit zero-padding (avoids separate padded tensor allocation).
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
    activation: ops::Activation,
) -> Result<Tensor, KernelError> {
    ops::conv2d_nhwc_padded(
        input, kernel, bias, stride_h, stride_w, pad_top, pad_left, pad_bottom, pad_right,
        activation,
    )
}

/// NHWC convolution without padding with explicit parallelization heuristics.
pub fn conv2d_nhwc_with_config(
    input: &Tensor,
    kernel: &Tensor,
    bias: Option<&Tensor>,
    stride_h: usize,
    stride_w: usize,
    config: ParallelElementwiseConfig,
) -> Result<Tensor, KernelError> {
    ops::conv2d_nhwc_with_config_and_pool(
        input,
        kernel,
        bias,
        Conv2dSpec { stride_h, stride_w },
        config,
        None,
    )
}

/// NHWC deformable convolution with learned offsets.
///
/// Input: `[N, H, W, C_in]`, Weight: `[kH, kW, C_in, C_out]`,
/// Offsets: `[N, H_out, W_out, kH*kW*2]`, Bias: `[C_out]` (optional).
pub fn deformable_conv2d_nhwc(
    input: &Tensor,
    weight: &Tensor,
    offsets: &Tensor,
    bias: Option<&Tensor>,
    stride: usize,
    padding: usize,
) -> Result<Tensor, KernelError> {
    ops::deformable_conv2d_nhwc(input, weight, offsets, bias, stride, padding)
}

/// NHWC depthwise convolution without padding using kernel shape `[KH, KW, C, depth_multiplier]` (parallel by default).
pub fn depthwise_conv2d_nhwc(
    input: &Tensor,
    kernel: &Tensor,
    bias: Option<&Tensor>,
    stride_h: usize,
    stride_w: usize,
) -> Result<Tensor, KernelError> {
    depthwise_conv2d_nhwc_with_activation(
        input,
        kernel,
        bias,
        stride_h,
        stride_w,
        ops::Activation::None,
    )
}

/// NHWC depthwise convolution without padding with optional fused activation.
pub fn depthwise_conv2d_nhwc_with_activation(
    input: &Tensor,
    kernel: &Tensor,
    bias: Option<&Tensor>,
    stride_h: usize,
    stride_w: usize,
    activation: ops::Activation,
) -> Result<Tensor, KernelError> {
    ops::depthwise_conv2d_nhwc_with_activation_with_config_and_pool(
        input,
        kernel,
        bias,
        DepthwiseConv2dSpec { stride_h, stride_w },
        activation,
        depthwise_parallel_config(),
        None,
    )
}

/// NHWC depthwise convolution with implicit zero-padding.
///
/// This applies padding virtually (no separate padded input allocation).
pub fn depthwise_conv2d_nhwc_padded(
    input: &Tensor,
    kernel: &Tensor,
    bias: Option<&Tensor>,
    stride_h: usize,
    stride_w: usize,
    pad_top: usize,
    pad_left: usize,
    pad_bottom: usize,
    pad_right: usize,
) -> Result<Tensor, KernelError> {
    depthwise_conv2d_nhwc_padded_with_activation(
        input,
        kernel,
        bias,
        stride_h,
        stride_w,
        pad_top,
        pad_left,
        pad_bottom,
        pad_right,
        ops::Activation::None,
    )
}

/// NHWC depthwise convolution with implicit zero-padding and optional fused activation.
pub fn depthwise_conv2d_nhwc_padded_with_activation(
    input: &Tensor,
    kernel: &Tensor,
    bias: Option<&Tensor>,
    stride_h: usize,
    stride_w: usize,
    pad_top: usize,
    pad_left: usize,
    pad_bottom: usize,
    pad_right: usize,
    activation: ops::Activation,
) -> Result<Tensor, KernelError> {
    ops::depthwise_conv2d_nhwc_padded_with_activation_with_config_and_pool(
        input,
        kernel,
        bias,
        DepthwiseConv2dSpec { stride_h, stride_w },
        pad_top,
        pad_left,
        pad_bottom,
        pad_right,
        activation,
        depthwise_parallel_config(),
        None,
    )
}

/// NHWC depthwise convolution without padding with explicit parallelization heuristics.
pub fn depthwise_conv2d_nhwc_with_config(
    input: &Tensor,
    kernel: &Tensor,
    bias: Option<&Tensor>,
    stride_h: usize,
    stride_w: usize,
    config: ParallelElementwiseConfig,
) -> Result<Tensor, KernelError> {
    ops::depthwise_conv2d_nhwc_with_config_and_pool(
        input,
        kernel,
        bias,
        DepthwiseConv2dSpec { stride_h, stride_w },
        config,
        None,
    )
}

/// NHWC depthwise convolution with implicit zero-padding and explicit
/// parallelization heuristics.
pub fn depthwise_conv2d_nhwc_padded_with_config(
    input: &Tensor,
    kernel: &Tensor,
    bias: Option<&Tensor>,
    stride_h: usize,
    stride_w: usize,
    pad_top: usize,
    pad_left: usize,
    pad_bottom: usize,
    pad_right: usize,
    config: ParallelElementwiseConfig,
) -> Result<Tensor, KernelError> {
    ops::depthwise_conv2d_nhwc_padded_with_config_and_pool(
        input,
        kernel,
        bias,
        DepthwiseConv2dSpec { stride_h, stride_w },
        pad_top,
        pad_left,
        pad_bottom,
        pad_right,
        config,
        None,
    )
}

/// NHWC separable convolution without padding (parallel by default):
/// depthwise (`[KH, KW, C, depth_multiplier]`) then pointwise (`[1, 1, C*depth_multiplier, C_out]`).
pub fn separable_conv2d_nhwc(
    input: &Tensor,
    params: SeparableConv2dParams<'_>,
    stride_h: usize,
    stride_w: usize,
) -> Result<Tensor, KernelError> {
    ops::separable_conv2d_nhwc_with_config_and_pool(
        input,
        SeparableConv2dKernels {
            depthwise_kernel: params.depthwise_kernel,
            depthwise_bias: params.depthwise_bias,
            pointwise_kernel: params.pointwise_kernel,
            pointwise_bias: params.pointwise_bias,
        },
        SeparableConv2dSpec { stride_h, stride_w },
        separable_parallel_config(),
        None,
    )
}

/// NHWC separable convolution without padding with explicit parallelization heuristics.
pub fn separable_conv2d_nhwc_with_config(
    input: &Tensor,
    params: SeparableConv2dParams<'_>,
    stride_h: usize,
    stride_w: usize,
    config: ParallelElementwiseConfig,
) -> Result<Tensor, KernelError> {
    ops::separable_conv2d_nhwc_with_config_and_pool(
        input,
        SeparableConv2dKernels {
            depthwise_kernel: params.depthwise_kernel,
            depthwise_bias: params.depthwise_bias,
            pointwise_kernel: params.pointwise_kernel,
            pointwise_bias: params.pointwise_bias,
        },
        SeparableConv2dSpec { stride_h, stride_w },
        config,
        None,
    )
}

/// NHWC per-channel batch normalization inference (parallel by default):
/// `out = ((x - mean) / sqrt(variance + epsilon)) * gamma + beta`.
pub fn batch_norm2d_nhwc(
    input: &Tensor,
    params: BatchNorm2dParams<'_>,
) -> Result<Tensor, KernelError> {
    ops::batch_norm2d_nhwc_with_config_and_pool(
        input,
        BatchNorm2dTensors {
            gamma: params.gamma,
            beta: params.beta,
            mean: params.mean,
            variance: params.variance,
            epsilon: params.epsilon,
        },
        ParallelElementwiseConfig::default(),
        None,
    )
}

/// NHWC per-channel batch normalization inference with explicit parallelization heuristics.
pub fn batch_norm2d_nhwc_with_config(
    input: &Tensor,
    params: BatchNorm2dParams<'_>,
    config: ParallelElementwiseConfig,
) -> Result<Tensor, KernelError> {
    ops::batch_norm2d_nhwc_with_config_and_pool(
        input,
        BatchNorm2dTensors {
            gamma: params.gamma,
            beta: params.beta,
            mean: params.mean,
            variance: params.variance,
            epsilon: params.epsilon,
        },
        config,
        None,
    )
}

/// NHWC group normalization: normalize within groups of channels (parallel by default).
pub fn group_norm_nhwc(
    input: &Tensor,
    params: GroupNormNhwcParams<'_>,
) -> Result<Tensor, KernelError> {
    ops::group_norm_nhwc_with_config_and_pool(
        input,
        GroupNorm2dTensors {
            gamma: params.gamma,
            beta: params.beta,
            num_groups: params.num_groups,
            epsilon: params.epsilon,
        },
        ParallelElementwiseConfig::default(),
        None,
    )
}

/// NHWC group normalization with explicit parallelization heuristics.
pub fn group_norm_nhwc_with_config(
    input: &Tensor,
    params: GroupNormNhwcParams<'_>,
    config: ParallelElementwiseConfig,
) -> Result<Tensor, KernelError> {
    ops::group_norm_nhwc_with_config_and_pool(
        input,
        GroupNorm2dTensors {
            gamma: params.gamma,
            beta: params.beta,
            num_groups: params.num_groups,
            epsilon: params.epsilon,
        },
        config,
        None,
    )
}

/// RMS normalization over the last tensor dimension (parallel by default).
pub fn rms_norm_last_dim(
    input: &Tensor,
    params: RmsNormLastDimParams<'_>,
) -> Result<Tensor, KernelError> {
    ops::rms_norm_last_dim_with_config_and_pool(
        input,
        RmsNormLastDimTensors {
            gamma: params.gamma,
            epsilon: params.epsilon,
        },
        ParallelElementwiseConfig::default(),
        None,
    )
}

/// RMS normalization over the last tensor dimension with explicit parallelization heuristics.
pub fn rms_norm_last_dim_with_config(
    input: &Tensor,
    params: RmsNormLastDimParams<'_>,
    config: ParallelElementwiseConfig,
) -> Result<Tensor, KernelError> {
    ops::rms_norm_last_dim_with_config_and_pool(
        input,
        RmsNormLastDimTensors {
            gamma: params.gamma,
            epsilon: params.epsilon,
        },
        config,
        None,
    )
}

/// Deterministic rank-2 matrix multiplication: `(m x k) * (k x n) -> (m x n)`.
pub fn matmul_2d(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor, KernelError> {
    ops::matmul_2d(lhs, rhs)
}

/// Zero-copy slice-based matmul: C[m×n] = A[m×k] × B[k×n].
/// Writes directly into `out` without any intermediate allocation.
pub fn matmul_2d_slices(a: &[f32], m: usize, k: usize, b: &[f32], n: usize, out: &mut [f32]) {
    ops::matmul_2d_slices(a, m, k, b, n, out);
}

/// Slice matmul with transposed left operand: `a_kt` is physically
/// `[K, M]` (pre-transpose); computes `out[m, n] = Σ_k a_kt[k, m] · b[k, n]`.
/// Drives the Transpose→MatMul graph fusion without materialising the
/// transpose. Uses BLAS `CblasTrans` under `--features blas`, else
/// falls back to a scratch-buffer transpose (see `ops::`).
pub fn matmul_2d_slices_trans_a(
    a_kt: &[f32],
    m: usize,
    k: usize,
    b: &[f32],
    n: usize,
    out: &mut [f32],
) {
    ops::matmul_2d_slices_trans_a(a_kt, m, k, b, n, out);
}

/// Parallel variant of [`matmul_2d_slices`]: row-parallel over M when
/// a `ParallelScope` is installed. Stream A1 FTMM NHWC fast path.
pub fn matmul_2d_slices_parallel(
    a: &[f32],
    m: usize,
    k: usize,
    b: &[f32],
    n: usize,
    out: &mut [f32],
) {
    ops::matmul_2d_slices_parallel(a, m, k, b, n, out);
}

/// Single-thread deterministic rank-2 matrix multiplication.
pub fn matmul_2d_sequential(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor, KernelError> {
    ops::matmul_2d_sequential(lhs, rhs)
}

/// Rank-2 matrix multiplication with explicit parallelization heuristics.
pub fn matmul_2d_with_config(
    lhs: &Tensor,
    rhs: &Tensor,
    config: ParallelMatmulConfig,
) -> Result<Tensor, KernelError> {
    ops::matmul_2d_with_config(lhs, rhs, config)
}

/// Rank-2 matrix multiplication executed through a dedicated thread pool.
pub fn matmul_2d_with_threads(
    lhs: &Tensor,
    rhs: &Tensor,
    num_threads: NonZeroUsize,
    config: ParallelMatmulConfig,
) -> Result<Tensor, KernelError> {
    let backend = ThreadedCpuBackend::with_config(num_threads, config)?;
    backend.matmul_2d(lhs, rhs)
}

/// Looks up embeddings from a weight matrix.
///
/// `weight`: `[vocab_size, embed_dim]`
/// `indices`: `[*]` — flat tensor of integer indices (stored as f32)
///
/// Returns: `[*indices_shape, embed_dim]`
pub fn embedding_lookup(weight: &Tensor, indices: &Tensor) -> Result<Tensor, KernelError> {
    ops::embedding_lookup(weight, indices)
}

/// Applies dropout: randomly zeroes elements with probability `p`.
///
/// During inference (`training=false`), returns input unchanged.
/// Uses xorshift64 PRNG with given seed for deterministic masking.
pub fn dropout(input: &Tensor, p: f32, seed: u64, training: bool) -> Result<Tensor, KernelError> {
    ops::dropout(input, p, seed, training)
}

/// Scaled dot-product attention for 2-D (unbatched) inputs.
///
/// `Attention(Q, K, V, mask?) = softmax(Q @ K^T / sqrt(d_k) + mask) @ V`
///
/// * `query`:  `[seq_q, d_k]`
/// * `key`:    `[seq_k, d_k]`
/// * `value`:  `[seq_k, d_v]`
/// * `mask`:   optional `[seq_q, seq_k]` additive mask
///
/// Returns `[seq_q, d_v]`.
pub fn scaled_dot_product_attention(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    mask: Option<&Tensor>,
) -> Result<Tensor, KernelError> {
    ops::attention::scaled_dot_product_attention(query, key, value, mask)
}

/// Memory-efficient (flash) attention — same result as `scaled_dot_product_attention`
/// but uses O(Br×Bc) peak memory instead of O(seq_q×seq_k).
pub fn flash_attention(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    mask: Option<&Tensor>,
) -> Result<Tensor, KernelError> {
    ops::attention::flash_attention(query, key, value, mask)
}

/// CPU transposed convolution (deconvolution) in NHWC layout.
///
/// Input: `[N,H,W,C_in]`, kernel: `[KH,KW,C_in,C_out]`, bias: optional `[C_out]`.
/// Output: `[N, (H-1)*stride_h + KH, (W-1)*stride_w + KW, C_out]`.
#[allow(clippy::too_many_arguments)]
pub fn transpose_conv2d_nhwc(
    input: &Tensor,
    kernel: &Tensor,
    bias: Option<&Tensor>,
    stride_h: usize,
    stride_w: usize,
) -> Result<Tensor, KernelError> {
    let is = input.shape();
    let ks = kernel.shape();
    if is.len() != 4 || ks.len() != 4 {
        return Err(KernelError::InvalidConvRank {
            input_rank: is.len(),
            kernel_rank: ks.len(),
        });
    }
    let (n, ih, iw, ic) = (is[0], is[1], is[2], is[3]);
    let (kh, kw, _kc, oc) = (ks[0], ks[1], ks[2], ks[3]);
    let oh = (ih - 1) * stride_h + kh;
    let ow = (iw - 1) * stride_w + kw;

    let in_d = input.data();
    let k_d = kernel.data();
    let bias_d: Vec<f32> = bias.map_or_else(|| vec![0.0f32; oc], |b| b.data().to_vec());

    let mut out = vec![0.0f32; n * oh * ow * oc];

    // Flatten the 7-deep naive loop into (b, iy, ix, ky, kx) outer loops
    // with a vectorised inner body: for each spatial scatter position we
    // compute  out[oy,ox,:] += input[iy,ix,:] · kernel[ky,kx,:,:]
    // i.e. a vector-matrix multiply of [ic] x [ic, oc] → [oc].
    #[allow(unsafe_code)]
    for b in 0..n {
        let in_batch = b * ih * iw * ic;
        let out_batch = b * oh * ow * oc;
        for iy in 0..ih {
            for ix in 0..iw {
                let in_base = in_batch + (iy * iw + ix) * ic;
                for ky in 0..kh {
                    let oy = iy * stride_h + ky;
                    for kx in 0..kw {
                        let ox = ix * stride_w + kx;
                        let out_base = out_batch + (oy * ow + ox) * oc;
                        let k_spatial = (ky * kw + kx) * ic * oc;
                        // SAFETY: all indices are within bounds because:
                        // - in_base + ci < n*ih*iw*ic (input dims)
                        // - out_base + co < n*oh*ow*oc (output dims)
                        // - k_spatial + ci*oc + co < kh*kw*ic*oc (kernel dims)
                        unsafe {
                            let in_ptr = in_d.as_ptr().add(in_base);
                            let out_ptr = out.as_mut_ptr().add(out_base);
                            let k_ptr = k_d.as_ptr().add(k_spatial);
                            for ci in 0..ic {
                                let in_val = *in_ptr.add(ci);
                                let k_row = k_ptr.add(ci * oc);
                                for co in 0..oc {
                                    *out_ptr.add(co) += in_val * *k_row.add(co);
                                }
                            }
                        }
                    }
                }
            }
        }
        // Add bias
        for idx in 0..(oh * ow) {
            let base = out_batch + idx * oc;
            for co in 0..oc {
                out[base + co] += bias_d[co];
            }
        }
    }

    Tensor::from_vec(vec![n, oh, ow, oc], out).map_err(Into::into)
}
