use std::num::NonZeroUsize;

use rayon::{ThreadPool, ThreadPoolBuilder};
use yscv_tensor::Tensor;

use super::{
    error::KernelError,
    ops::{
        self, BatchNorm2dTensors, Conv2dSpec, DepthwiseConv2dSpec, GroupNorm2dTensors,
        LayerNormLastDimTensors, ParallelElementwiseConfig, ParallelMatmulConfig, Pool2dSpec,
        RmsNormLastDimTensors, SeparableConv2dKernels, SeparableConv2dSpec,
    },
};

/// Tensor parameter bundle for NHWC separable convolution:
/// depthwise (`[KH, KW, C, depth_multiplier]`) then pointwise (`[1, 1, C*depth_multiplier, C_out]`).
#[derive(Debug, Clone, Copy)]
pub struct SeparableConv2dParams<'a> {
    pub depthwise_kernel: &'a Tensor,
    pub depthwise_bias: Option<&'a Tensor>,
    pub pointwise_kernel: &'a Tensor,
    pub pointwise_bias: Option<&'a Tensor>,
}

/// Tensor parameter bundle for NHWC batch-normalization inference.
#[derive(Debug, Clone, Copy)]
pub struct BatchNorm2dParams<'a> {
    pub gamma: &'a Tensor,
    pub beta: &'a Tensor,
    pub mean: &'a Tensor,
    pub variance: &'a Tensor,
    pub epsilon: f32,
}

/// Tensor parameter bundle for layer normalization over the last tensor dimension.
#[derive(Debug, Clone, Copy)]
pub struct LayerNormLastDimParams<'a> {
    pub gamma: &'a Tensor,
    pub beta: &'a Tensor,
    pub epsilon: f32,
}

/// Tensor parameter bundle for NHWC group normalization.
#[derive(Debug, Clone, Copy)]
pub struct GroupNormNhwcParams<'a> {
    pub gamma: &'a Tensor,
    pub beta: &'a Tensor,
    pub num_groups: usize,
    pub epsilon: f32,
}

/// Tensor parameter bundle for RMS normalization over the last tensor dimension.
#[derive(Debug, Clone, Copy)]
pub struct RmsNormLastDimParams<'a> {
    pub gamma: &'a Tensor,
    pub epsilon: f32,
}

/// Runtime backend contract for core deterministic kernels.
pub trait Backend {
    fn add(&self, lhs: &Tensor, rhs: &Tensor) -> Result<Tensor, KernelError>;
    fn sub(&self, lhs: &Tensor, rhs: &Tensor) -> Result<Tensor, KernelError>;
    fn mul(&self, lhs: &Tensor, rhs: &Tensor) -> Result<Tensor, KernelError>;
    fn relu(&self, input: &Tensor) -> Tensor;
    fn sigmoid(&self, input: &Tensor) -> Tensor;
    fn exp(&self, input: &Tensor) -> Tensor;
    fn tanh_act(&self, input: &Tensor) -> Tensor;
    fn softmax_last_dim(&self, input: &Tensor) -> Result<Tensor, KernelError>;
    fn log_softmax_last_dim(&self, input: &Tensor) -> Result<Tensor, KernelError>;
    fn logsumexp_last_dim(&self, input: &Tensor) -> Result<Tensor, KernelError>;
    fn layer_norm_last_dim(
        &self,
        input: &Tensor,
        params: LayerNormLastDimParams<'_>,
    ) -> Result<Tensor, KernelError>;
    fn max_pool2d_nhwc(
        &self,
        input: &Tensor,
        kernel_h: usize,
        kernel_w: usize,
        stride_h: usize,
        stride_w: usize,
    ) -> Result<Tensor, KernelError>;
    fn avg_pool2d_nhwc(
        &self,
        input: &Tensor,
        kernel_h: usize,
        kernel_w: usize,
        stride_h: usize,
        stride_w: usize,
    ) -> Result<Tensor, KernelError>;
    fn conv2d_nhwc(
        &self,
        input: &Tensor,
        kernel: &Tensor,
        bias: Option<&Tensor>,
        stride_h: usize,
        stride_w: usize,
    ) -> Result<Tensor, KernelError>;
    fn depthwise_conv2d_nhwc(
        &self,
        input: &Tensor,
        kernel: &Tensor,
        bias: Option<&Tensor>,
        stride_h: usize,
        stride_w: usize,
    ) -> Result<Tensor, KernelError>;
    fn separable_conv2d_nhwc(
        &self,
        input: &Tensor,
        params: SeparableConv2dParams<'_>,
        stride_h: usize,
        stride_w: usize,
    ) -> Result<Tensor, KernelError>;
    fn batch_norm2d_nhwc(
        &self,
        input: &Tensor,
        params: BatchNorm2dParams<'_>,
    ) -> Result<Tensor, KernelError>;
    fn group_norm_nhwc(
        &self,
        input: &Tensor,
        params: GroupNormNhwcParams<'_>,
    ) -> Result<Tensor, KernelError>;
    fn rms_norm_last_dim(
        &self,
        input: &Tensor,
        params: RmsNormLastDimParams<'_>,
    ) -> Result<Tensor, KernelError>;
    fn matmul_2d(&self, lhs: &Tensor, rhs: &Tensor) -> Result<Tensor, KernelError>;

    // ── Backward-relevant ops with default CPU implementations ──────

    /// Element-wise negation.
    fn neg(&self, input: &Tensor) -> Tensor {
        input.neg()
    }

    /// Element-wise division with broadcast.
    fn div(&self, lhs: &Tensor, rhs: &Tensor) -> Result<Tensor, KernelError> {
        lhs.div(rhs).map_err(Into::into)
    }

    /// Element-wise square root.
    fn sqrt(&self, input: &Tensor) -> Tensor {
        input.sqrt()
    }

    /// Transpose a 2-D matrix.
    fn transpose_2d(&self, input: &Tensor) -> Result<Tensor, KernelError> {
        input.transpose_2d().map_err(Into::into)
    }

    /// Scalar sum of all elements (returns a scalar tensor).
    fn sum_all(&self, input: &Tensor) -> Tensor {
        Tensor::scalar(input.sum())
    }

    /// Multiply every element by a scalar.
    fn mul_scalar(&self, input: &Tensor, scalar: f32) -> Tensor {
        input.scale(scalar)
    }

    /// Element-wise reciprocal (1/x).
    fn reciprocal(&self, input: &Tensor) -> Tensor {
        input.reciprocal()
    }
}

/// Extension trait for backward-pass operations.
///
/// Separated from [`Backend`] so that forward-only consumers (e.g. ONNX inference)
/// need not depend on backward-related method signatures.  All methods have default
/// CPU implementations, so `impl BackwardOps for MyBackend {}` is sufficient.
pub trait BackwardOps: Backend {
    /// ReLU backward: `grad_input[i] = upstream[i] * (forward_input[i] > 0 ? 1 : 0)`.
    fn relu_backward(
        &self,
        upstream: &Tensor,
        forward_input: &Tensor,
    ) -> Result<Tensor, KernelError> {
        let u = upstream.data();
        let f = forward_input.data();
        let out: Vec<f32> = u
            .iter()
            .zip(f.iter())
            .map(|(&u, &x)| if x > 0.0 { u } else { 0.0 })
            .collect();
        Tensor::from_vec(upstream.shape().to_vec(), out).map_err(Into::into)
    }

    /// Sigmoid backward: `grad_input[i] = upstream[i] * s[i] * (1 - s[i])` where `s` = forward output.
    fn sigmoid_backward(
        &self,
        upstream: &Tensor,
        forward_output: &Tensor,
    ) -> Result<Tensor, KernelError> {
        let u = upstream.data();
        let s = forward_output.data();
        let out: Vec<f32> = u
            .iter()
            .zip(s.iter())
            .map(|(&u, &s)| u * s * (1.0 - s))
            .collect();
        Tensor::from_vec(upstream.shape().to_vec(), out).map_err(Into::into)
    }

    /// Tanh backward: `grad_input[i] = upstream[i] * (1 - t[i]^2)` where `t` = forward output.
    fn tanh_backward(
        &self,
        upstream: &Tensor,
        forward_output: &Tensor,
    ) -> Result<Tensor, KernelError> {
        let u = upstream.data();
        let t = forward_output.data();
        let out: Vec<f32> = u
            .iter()
            .zip(t.iter())
            .map(|(&u, &t)| u * (1.0 - t * t))
            .collect();
        Tensor::from_vec(upstream.shape().to_vec(), out).map_err(Into::into)
    }

    /// Exp backward: `grad_input[i] = upstream[i] * e[i]` where `e` = forward output.
    fn exp_backward(
        &self,
        upstream: &Tensor,
        forward_output: &Tensor,
    ) -> Result<Tensor, KernelError> {
        let u = upstream.data();
        let e = forward_output.data();
        let out: Vec<f32> = u.iter().zip(e.iter()).map(|(&u, &e)| u * e).collect();
        Tensor::from_vec(upstream.shape().to_vec(), out).map_err(Into::into)
    }

    /// Reduce-sum backward: broadcast scalar gradient to all elements of `original_shape`.
    fn reduce_sum_backward(
        &self,
        upstream: &Tensor,
        original_shape: &[usize],
    ) -> Result<Tensor, KernelError> {
        let grad_val = upstream.data()[0];
        let len: usize = original_shape.iter().product();
        let out = vec![grad_val; len];
        Tensor::from_vec(original_shape.to_vec(), out).map_err(Into::into)
    }

    /// MatMul backward: `grad_lhs = upstream @ rhs^T`, `grad_rhs = lhs^T @ upstream`.
    fn matmul_backward(
        &self,
        upstream: &Tensor,
        lhs: &Tensor,
        rhs: &Tensor,
    ) -> Result<(Tensor, Tensor), KernelError> {
        let rt = self.transpose_2d(rhs)?;
        let lt = self.transpose_2d(lhs)?;
        let grad_lhs = self.matmul_2d(upstream, &rt)?;
        let grad_rhs = self.matmul_2d(&lt, upstream)?;
        Ok((grad_lhs, grad_rhs))
    }

    /// Add backward: gradient passes through unchanged to both operands.
    fn add_backward(
        &self,
        upstream: &Tensor,
        _lhs: &Tensor,
        _rhs: &Tensor,
    ) -> Result<(Tensor, Tensor), KernelError> {
        Ok((upstream.clone(), upstream.clone()))
    }

    /// Sub backward: `grad_lhs = upstream`, `grad_rhs = -upstream`.
    fn sub_backward(
        &self,
        upstream: &Tensor,
        _lhs: &Tensor,
        _rhs: &Tensor,
    ) -> Result<(Tensor, Tensor), KernelError> {
        Ok((upstream.clone(), self.neg(upstream)))
    }

    /// Mul backward: `grad_lhs = upstream * rhs`, `grad_rhs = upstream * lhs`.
    fn mul_backward(
        &self,
        upstream: &Tensor,
        lhs: &Tensor,
        rhs: &Tensor,
    ) -> Result<(Tensor, Tensor), KernelError> {
        let grad_lhs = self.mul(upstream, rhs)?;
        let grad_rhs = self.mul(upstream, lhs)?;
        Ok((grad_lhs, grad_rhs))
    }

    /// Conv2d backward (input gradient): compute dL/dInput from dL/dOutput and weights.
    ///
    /// Default CPU implementation via full convolution with flipped kernel.
    fn conv2d_input_backward(
        &self,
        upstream: &Tensor,
        kernel: &Tensor,
        input_shape: &[usize],
        stride_h: usize,
        stride_w: usize,
    ) -> Result<Tensor, KernelError> {
        // upstream: [N, OH, OW, OC], kernel: [KH, KW, IC, OC], output: [N, IH, IW, IC]
        let us = upstream.shape();
        let ks = kernel.shape();
        if us.len() != 4 || ks.len() != 4 || input_shape.len() != 4 {
            return Err(KernelError::InvalidConvRank {
                input_rank: input_shape.len(),
                kernel_rank: ks.len(),
            });
        }
        let (n, ih, iw, ic) = (
            input_shape[0],
            input_shape[1],
            input_shape[2],
            input_shape[3],
        );
        let (_n, oh, ow, oc) = (us[0], us[1], us[2], us[3]);
        let (kh, kw) = (ks[0], ks[1]);

        let u_data = upstream.data();
        let k_data = kernel.data();
        let mut grad_input = vec![0.0f32; n * ih * iw * ic];

        for b in 0..n {
            for oy in 0..oh {
                for ox in 0..ow {
                    for co in 0..oc {
                        let g = u_data[((b * oh + oy) * ow + ox) * oc + co];
                        if g == 0.0 {
                            continue;
                        }
                        for ky in 0..kh {
                            for kx in 0..kw {
                                let iy = oy * stride_h + ky;
                                let ix = ox * stride_w + kx;
                                if iy < ih && ix < iw {
                                    for ci in 0..ic {
                                        let k_val = k_data[((ky * kw + kx) * ic + ci) * oc + co];
                                        grad_input[((b * ih + iy) * iw + ix) * ic + ci] +=
                                            g * k_val;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        Tensor::from_vec(input_shape.to_vec(), grad_input).map_err(Into::into)
    }
}

// Explicit BackwardOps implementations using default methods (CPU fallback).
// GpuBackend provides GPU-accelerated overrides in gpu_backend.rs.

/// Deterministic CPU backend with fixed operation order.
#[derive(Debug, Clone, Copy, Default)]
pub struct CpuBackend;

impl BackwardOps for CpuBackend {}

impl Backend for CpuBackend {
    fn add(&self, lhs: &Tensor, rhs: &Tensor) -> Result<Tensor, KernelError> {
        ops::add_with_config(lhs, rhs, ParallelElementwiseConfig::disabled())
    }

    fn sub(&self, lhs: &Tensor, rhs: &Tensor) -> Result<Tensor, KernelError> {
        ops::sub_with_config(lhs, rhs, ParallelElementwiseConfig::disabled())
    }

    fn mul(&self, lhs: &Tensor, rhs: &Tensor) -> Result<Tensor, KernelError> {
        ops::mul_with_config(lhs, rhs, ParallelElementwiseConfig::disabled())
    }

    fn relu(&self, input: &Tensor) -> Tensor {
        ops::relu(input)
    }

    fn sigmoid(&self, input: &Tensor) -> Tensor {
        ops::sigmoid(input)
    }

    fn exp(&self, input: &Tensor) -> Tensor {
        ops::exp(input)
    }

    fn tanh_act(&self, input: &Tensor) -> Tensor {
        ops::tanh_act(input)
    }

    fn softmax_last_dim(&self, input: &Tensor) -> Result<Tensor, KernelError> {
        ops::softmax_last_dim_with_config_and_pool(
            input,
            ParallelElementwiseConfig::disabled(),
            None,
        )
    }

    fn log_softmax_last_dim(&self, input: &Tensor) -> Result<Tensor, KernelError> {
        ops::log_softmax_last_dim_with_config_and_pool(
            input,
            ParallelElementwiseConfig::disabled(),
            None,
        )
    }

    fn logsumexp_last_dim(&self, input: &Tensor) -> Result<Tensor, KernelError> {
        ops::logsumexp_last_dim_with_config_and_pool(
            input,
            ParallelElementwiseConfig::disabled(),
            None,
        )
    }

    fn layer_norm_last_dim(
        &self,
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
            ParallelElementwiseConfig::disabled(),
            None,
        )
    }

    fn max_pool2d_nhwc(
        &self,
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
            ParallelElementwiseConfig::disabled(),
            None,
        )
    }

    fn avg_pool2d_nhwc(
        &self,
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
            ParallelElementwiseConfig::disabled(),
            None,
        )
    }

    fn conv2d_nhwc(
        &self,
        input: &Tensor,
        kernel: &Tensor,
        bias: Option<&Tensor>,
        stride_h: usize,
        stride_w: usize,
    ) -> Result<Tensor, KernelError> {
        ops::conv2d_nhwc_with_config_and_pool(
            input,
            kernel,
            bias,
            Conv2dSpec { stride_h, stride_w },
            ParallelElementwiseConfig::disabled(),
            None,
        )
    }

    fn depthwise_conv2d_nhwc(
        &self,
        input: &Tensor,
        kernel: &Tensor,
        bias: Option<&Tensor>,
        stride_h: usize,
        stride_w: usize,
    ) -> Result<Tensor, KernelError> {
        ops::depthwise_conv2d_nhwc_with_config_and_pool(
            input,
            kernel,
            bias,
            DepthwiseConv2dSpec { stride_h, stride_w },
            ParallelElementwiseConfig::disabled(),
            None,
        )
    }

    fn separable_conv2d_nhwc(
        &self,
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
            ParallelElementwiseConfig::disabled(),
            None,
        )
    }

    fn batch_norm2d_nhwc(
        &self,
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
            ParallelElementwiseConfig::disabled(),
            None,
        )
    }

    fn group_norm_nhwc(
        &self,
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
            ParallelElementwiseConfig::disabled(),
            None,
        )
    }

    fn rms_norm_last_dim(
        &self,
        input: &Tensor,
        params: RmsNormLastDimParams<'_>,
    ) -> Result<Tensor, KernelError> {
        ops::rms_norm_last_dim_with_config_and_pool(
            input,
            RmsNormLastDimTensors {
                gamma: params.gamma,
                epsilon: params.epsilon,
            },
            ParallelElementwiseConfig::disabled(),
            None,
        )
    }

    fn matmul_2d(&self, lhs: &Tensor, rhs: &Tensor) -> Result<Tensor, KernelError> {
        ops::matmul_2d(lhs, rhs)
    }
}

/// CPU backend with a dedicated rayon thread pool for predictable kernel threading depth.
#[derive(Debug)]
pub struct ThreadedCpuBackend {
    matmul_config: ParallelMatmulConfig,
    elementwise_config: ParallelElementwiseConfig,
    thread_pool: ThreadPool,
}

/// Runtime knobs for threaded CPU backend execution behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ThreadedCpuBackendConfig {
    pub matmul: ParallelMatmulConfig,
    pub elementwise: ParallelElementwiseConfig,
}

impl ThreadedCpuBackend {
    /// Build a threaded backend with default parallel matmul heuristics.
    pub fn new(num_threads: NonZeroUsize) -> Result<Self, KernelError> {
        Self::with_full_config(num_threads, ThreadedCpuBackendConfig::default())
    }

    /// Build a threaded backend with explicit parallel-matmul configuration.
    pub fn with_config(
        num_threads: NonZeroUsize,
        matmul_config: ParallelMatmulConfig,
    ) -> Result<Self, KernelError> {
        Self::with_full_config(
            num_threads,
            ThreadedCpuBackendConfig {
                matmul: matmul_config,
                elementwise: ParallelElementwiseConfig::default(),
            },
        )
    }

    /// Build a threaded backend with explicit matmul and elementwise configuration.
    pub fn with_full_config(
        num_threads: NonZeroUsize,
        config: ThreadedCpuBackendConfig,
    ) -> Result<Self, KernelError> {
        let thread_pool = ThreadPoolBuilder::new()
            .num_threads(num_threads.get())
            .build()
            .map_err(|error| KernelError::ThreadPoolBuild {
                message: error.to_string(),
            })?;
        Ok(Self {
            matmul_config: config.matmul,
            elementwise_config: config.elementwise,
            thread_pool,
        })
    }

    /// Matmul parallelism knobs used by this backend.
    pub const fn matmul_config(&self) -> ParallelMatmulConfig {
        self.matmul_config
    }

    /// Elementwise parallelism knobs used by this backend.
    pub const fn elementwise_config(&self) -> ParallelElementwiseConfig {
        self.elementwise_config
    }
}

impl BackwardOps for ThreadedCpuBackend {}

impl Backend for ThreadedCpuBackend {
    fn add(&self, lhs: &Tensor, rhs: &Tensor) -> Result<Tensor, KernelError> {
        ops::add_with_config_and_pool(lhs, rhs, self.elementwise_config, Some(&self.thread_pool))
    }

    fn sub(&self, lhs: &Tensor, rhs: &Tensor) -> Result<Tensor, KernelError> {
        ops::sub_with_config_and_pool(lhs, rhs, self.elementwise_config, Some(&self.thread_pool))
    }

    fn mul(&self, lhs: &Tensor, rhs: &Tensor) -> Result<Tensor, KernelError> {
        ops::mul_with_config_and_pool(lhs, rhs, self.elementwise_config, Some(&self.thread_pool))
    }

    fn relu(&self, input: &Tensor) -> Tensor {
        ops::relu_with_config_and_pool(input, self.elementwise_config, Some(&self.thread_pool))
    }

    fn sigmoid(&self, input: &Tensor) -> Tensor {
        ops::sigmoid_with_config_and_pool(input, self.elementwise_config, Some(&self.thread_pool))
    }

    fn exp(&self, input: &Tensor) -> Tensor {
        ops::exp_with_config_and_pool(input, self.elementwise_config, Some(&self.thread_pool))
    }

    fn tanh_act(&self, input: &Tensor) -> Tensor {
        ops::tanh_act_with_config_and_pool(input, self.elementwise_config, Some(&self.thread_pool))
    }

    fn softmax_last_dim(&self, input: &Tensor) -> Result<Tensor, KernelError> {
        ops::softmax_last_dim_with_config_and_pool(
            input,
            self.elementwise_config,
            Some(&self.thread_pool),
        )
    }

    fn log_softmax_last_dim(&self, input: &Tensor) -> Result<Tensor, KernelError> {
        ops::log_softmax_last_dim_with_config_and_pool(
            input,
            self.elementwise_config,
            Some(&self.thread_pool),
        )
    }

    fn logsumexp_last_dim(&self, input: &Tensor) -> Result<Tensor, KernelError> {
        ops::logsumexp_last_dim_with_config_and_pool(
            input,
            self.elementwise_config,
            Some(&self.thread_pool),
        )
    }

    fn layer_norm_last_dim(
        &self,
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
            self.elementwise_config,
            Some(&self.thread_pool),
        )
    }

    fn max_pool2d_nhwc(
        &self,
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
            self.elementwise_config,
            Some(&self.thread_pool),
        )
    }

    fn avg_pool2d_nhwc(
        &self,
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
            self.elementwise_config,
            Some(&self.thread_pool),
        )
    }

    fn conv2d_nhwc(
        &self,
        input: &Tensor,
        kernel: &Tensor,
        bias: Option<&Tensor>,
        stride_h: usize,
        stride_w: usize,
    ) -> Result<Tensor, KernelError> {
        ops::conv2d_nhwc_with_config_and_pool(
            input,
            kernel,
            bias,
            Conv2dSpec { stride_h, stride_w },
            self.elementwise_config,
            Some(&self.thread_pool),
        )
    }

    fn depthwise_conv2d_nhwc(
        &self,
        input: &Tensor,
        kernel: &Tensor,
        bias: Option<&Tensor>,
        stride_h: usize,
        stride_w: usize,
    ) -> Result<Tensor, KernelError> {
        ops::depthwise_conv2d_nhwc_with_config_and_pool(
            input,
            kernel,
            bias,
            DepthwiseConv2dSpec { stride_h, stride_w },
            self.elementwise_config,
            Some(&self.thread_pool),
        )
    }

    fn separable_conv2d_nhwc(
        &self,
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
            self.elementwise_config,
            Some(&self.thread_pool),
        )
    }

    fn batch_norm2d_nhwc(
        &self,
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
            self.elementwise_config,
            Some(&self.thread_pool),
        )
    }

    fn group_norm_nhwc(
        &self,
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
            self.elementwise_config,
            Some(&self.thread_pool),
        )
    }

    fn rms_norm_last_dim(
        &self,
        input: &Tensor,
        params: RmsNormLastDimParams<'_>,
    ) -> Result<Tensor, KernelError> {
        ops::rms_norm_last_dim_with_config_and_pool(
            input,
            RmsNormLastDimTensors {
                gamma: params.gamma,
                epsilon: params.epsilon,
            },
            self.elementwise_config,
            Some(&self.thread_pool),
        )
    }

    fn matmul_2d(&self, lhs: &Tensor, rhs: &Tensor) -> Result<Tensor, KernelError> {
        ops::matmul_2d_with_config_and_pool(lhs, rhs, self.matmul_config, Some(&self.thread_pool))
    }
}

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
    ops::conv2d_nhwc_with_config_and_pool(
        input,
        kernel,
        bias,
        Conv2dSpec { stride_h, stride_w },
        ParallelElementwiseConfig::default(),
        None,
    )
}

/// NHWC convolution with implicit zero-padding (avoids separate padded tensor allocation).
/// Only available when the `blas` feature is enabled.
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
    ops::depthwise_conv2d_nhwc_with_config_and_pool(
        input,
        kernel,
        bias,
        DepthwiseConv2dSpec { stride_h, stride_w },
        ParallelElementwiseConfig::default(),
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
        ParallelElementwiseConfig::default(),
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
