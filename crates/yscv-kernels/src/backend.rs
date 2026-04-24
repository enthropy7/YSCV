use std::num::NonZeroUsize;
use std::sync::OnceLock;

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

const DEFAULT_CONV_MIN_PARALLEL_ELEMENTS: usize = 4_096;
const DEFAULT_DEPTHWISE_MIN_PARALLEL_ELEMENTS: usize = 4_096;
const DEFAULT_SEPARABLE_MIN_PARALLEL_ELEMENTS: usize = 4_096;

#[inline]
fn tuned_parallel_config(env_name: &str, default_value: usize) -> ParallelElementwiseConfig {
    let min_parallel_elements = std::env::var(env_name)
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&v| v > 0)
        .unwrap_or(default_value);
    ParallelElementwiseConfig {
        min_parallel_elements,
    }
}

#[inline]
fn conv_parallel_config() -> ParallelElementwiseConfig {
    static CONFIG: OnceLock<ParallelElementwiseConfig> = OnceLock::new();
    *CONFIG.get_or_init(|| {
        tuned_parallel_config("YSCV_CONV_MIN_PARALLEL", DEFAULT_CONV_MIN_PARALLEL_ELEMENTS)
    })
}

#[inline]
fn depthwise_parallel_config() -> ParallelElementwiseConfig {
    static CONFIG: OnceLock<ParallelElementwiseConfig> = OnceLock::new();
    *CONFIG.get_or_init(|| {
        tuned_parallel_config(
            "YSCV_DEPTHWISE_MIN_PARALLEL",
            DEFAULT_DEPTHWISE_MIN_PARALLEL_ELEMENTS,
        )
    })
}

#[inline]
fn separable_parallel_config() -> ParallelElementwiseConfig {
    static CONFIG: OnceLock<ParallelElementwiseConfig> = OnceLock::new();
    *CONFIG.get_or_init(|| {
        tuned_parallel_config(
            "YSCV_SEPARABLE_MIN_PARALLEL",
            DEFAULT_SEPARABLE_MIN_PARALLEL_ELEMENTS,
        )
    })
}

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

    /// Conv2d weight backward: compute dL/dWeights from upstream gradient and forward input.
    fn conv2d_weight_backward(
        &self,
        upstream: &Tensor,
        forward_input: &Tensor,
        kernel_shape: &[usize],
        stride_h: usize,
        stride_w: usize,
    ) -> Result<Tensor, KernelError> {
        // upstream: [N, OH, OW, OC], forward_input: [N, IH, IW, IC]
        // kernel_shape: [KH, KW, IC, OC]
        let us = upstream.shape();
        let fs = forward_input.shape();
        if us.len() != 4 || fs.len() != 4 || kernel_shape.len() != 4 {
            return Err(KernelError::InvalidConvRank {
                input_rank: fs.len(),
                kernel_rank: kernel_shape.len(),
            });
        }
        let (n, oh, ow, oc) = (us[0], us[1], us[2], us[3]);
        let (kh, kw, ic, _) = (
            kernel_shape[0],
            kernel_shape[1],
            kernel_shape[2],
            kernel_shape[3],
        );

        let u_data = upstream.data();
        let f_data = forward_input.data();
        let ih = fs[1];
        let iw = fs[2];
        let mut grad_kernel = vec![0.0f32; kh * kw * ic * oc];

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
                                        let input_val = f_data[((b * ih + iy) * iw + ix) * ic + ci];
                                        grad_kernel[((ky * kw + kx) * ic + ci) * oc + co] +=
                                            g * input_val;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        Tensor::from_vec(kernel_shape.to_vec(), grad_kernel).map_err(Into::into)
    }

    /// Conv2d bias backward: sum upstream gradient over all spatial/batch dims.
    fn conv2d_bias_backward(&self, upstream: &Tensor, c_out: usize) -> Result<Tensor, KernelError> {
        // upstream: [N, OH, OW, OC] — sum over N, OH, OW
        let us = upstream.shape();
        if us.len() != 4 || us[3] != c_out {
            return Err(KernelError::ConvBiasShapeMismatch {
                bias_shape: vec![us.get(3).copied().unwrap_or(0)],
                out_channels: c_out,
            });
        }
        let u_data = upstream.data();
        let spatial = us[0] * us[1] * us[2];
        let mut grad_bias = vec![0.0f32; c_out];
        for i in 0..spatial {
            let base = i * c_out;
            for c in 0..c_out {
                grad_bias[c] += u_data[base + c];
            }
        }
        Tensor::from_vec(vec![c_out], grad_bias).map_err(Into::into)
    }

    /// BatchNorm2d input backward: scale upstream by gamma / sqrt(var + eps).
    fn batch_norm2d_input_backward(
        &self,
        upstream: &Tensor,
        gamma: &Tensor,
        running_var: &Tensor,
        epsilon: f32,
    ) -> Result<Tensor, KernelError> {
        let us = upstream.shape();
        if us.len() != 4 {
            return Err(KernelError::InvalidBatchNormRank { got_rank: us.len() });
        }
        let c = us[3];
        let g_data = gamma.data();
        let v_data = running_var.data();
        let u_data = upstream.data();
        let total = us.iter().product::<usize>();
        let mut out = vec![0.0f32; total];

        for i in 0..total {
            let ci = i % c;
            let scale = g_data[ci] / (v_data[ci] + epsilon).sqrt();
            out[i] = u_data[i] * scale;
        }
        Tensor::from_vec(us.to_vec(), out).map_err(Into::into)
    }

    /// LayerNorm input backward (simplified inference-mode approximation).
    fn layer_norm_input_backward(
        &self,
        upstream: &Tensor,
        forward_input: &Tensor,
        gamma: &Tensor,
        epsilon: f32,
    ) -> Result<Tensor, KernelError> {
        let shape = upstream.shape();
        if shape.is_empty() {
            return Err(KernelError::InvalidLayerNormRank { got_rank: 0 });
        }
        let d = *shape.last().expect("checked non-empty");
        let n = d as f32;
        let u_data = upstream.data();
        let f_data = forward_input.data();
        let g_data = gamma.data();
        let total = shape.iter().product::<usize>();
        let num_vecs = total / d;
        let mut out = vec![0.0f32; total];

        for v in 0..num_vecs {
            let base = v * d;
            let mean: f32 = f_data[base..base + d].iter().sum::<f32>() / n;
            let var: f32 = f_data[base..base + d]
                .iter()
                .map(|&x| (x - mean) * (x - mean))
                .sum::<f32>()
                / n;
            let inv_std = 1.0 / (var + epsilon).sqrt();

            // Compute dy * gamma
            let mut dy_gamma = vec![0.0f32; d];
            let mut x_hat = vec![0.0f32; d];
            for i in 0..d {
                x_hat[i] = (f_data[base + i] - mean) * inv_std;
                dy_gamma[i] = u_data[base + i] * g_data[i];
            }
            let mean_dy_gamma = dy_gamma.iter().sum::<f32>() / n;
            let mean_xhat_dy_gamma: f32 = x_hat
                .iter()
                .zip(dy_gamma.iter())
                .map(|(x, d)| x * d)
                .sum::<f32>()
                / n;

            // Full layer norm backward: inv_std * (dy*gamma - mean(dy*gamma) - x_hat * mean(x_hat * dy*gamma))
            for i in 0..d {
                out[base + i] =
                    inv_std * (dy_gamma[i] - mean_dy_gamma - x_hat[i] * mean_xhat_dy_gamma);
            }
        }
        Tensor::from_vec(shape.to_vec(), out).map_err(Into::into)
    }

    /// MaxPool2d backward: route gradient to argmax positions.
    fn max_pool2d_backward(
        &self,
        upstream: &Tensor,
        forward_input: &Tensor,
        input_shape: &[usize],
        kernel_h: usize,
        kernel_w: usize,
        stride_h: usize,
        stride_w: usize,
    ) -> Result<Tensor, KernelError> {
        let us = upstream.shape();
        if us.len() != 4 || input_shape.len() != 4 {
            return Err(KernelError::InvalidPoolRank {
                got_rank: input_shape.len(),
            });
        }
        let (n, ih, iw, c) = (
            input_shape[0],
            input_shape[1],
            input_shape[2],
            input_shape[3],
        );
        let (_n, oh, ow, _c) = (us[0], us[1], us[2], us[3]);
        let u_data = upstream.data();
        let f_data = forward_input.data();
        let mut grad_input = vec![0.0f32; n * ih * iw * c];

        for b in 0..n {
            for oy in 0..oh {
                for ox in 0..ow {
                    for ch in 0..c {
                        let g = u_data[((b * oh + oy) * ow + ox) * c + ch];
                        // Find argmax in the window
                        let mut max_val = f32::NEG_INFINITY;
                        let mut max_iy = 0usize;
                        let mut max_ix = 0usize;
                        for ky in 0..kernel_h {
                            for kx in 0..kernel_w {
                                let iy = oy * stride_h + ky;
                                let ix = ox * stride_w + kx;
                                if iy < ih && ix < iw {
                                    let val = f_data[((b * ih + iy) * iw + ix) * c + ch];
                                    if val > max_val {
                                        max_val = val;
                                        max_iy = iy;
                                        max_ix = ix;
                                    }
                                }
                            }
                        }
                        grad_input[((b * ih + max_iy) * iw + max_ix) * c + ch] += g;
                    }
                }
            }
        }
        Tensor::from_vec(input_shape.to_vec(), grad_input).map_err(Into::into)
    }

    /// AvgPool2d backward: distribute gradient equally across the pooling window.
    fn avg_pool2d_backward(
        &self,
        upstream: &Tensor,
        input_shape: &[usize],
        kernel_h: usize,
        kernel_w: usize,
        stride_h: usize,
        stride_w: usize,
    ) -> Result<Tensor, KernelError> {
        let us = upstream.shape();
        if us.len() != 4 || input_shape.len() != 4 {
            return Err(KernelError::InvalidPoolRank {
                got_rank: input_shape.len(),
            });
        }
        let (n, ih, iw, c) = (
            input_shape[0],
            input_shape[1],
            input_shape[2],
            input_shape[3],
        );
        let (_n, oh, ow, _c) = (us[0], us[1], us[2], us[3]);
        let u_data = upstream.data();
        let inv_area = 1.0 / (kernel_h * kernel_w) as f32;
        let mut grad_input = vec![0.0f32; n * ih * iw * c];

        for b in 0..n {
            for oy in 0..oh {
                for ox in 0..ow {
                    for ch in 0..c {
                        let g = u_data[((b * oh + oy) * ow + ox) * c + ch] * inv_area;
                        for ky in 0..kernel_h {
                            for kx in 0..kernel_w {
                                let iy = oy * stride_h + ky;
                                let ix = ox * stride_w + kx;
                                if iy < ih && ix < iw {
                                    grad_input[((b * ih + iy) * iw + ix) * c + ch] += g;
                                }
                            }
                        }
                    }
                }
            }
        }
        Tensor::from_vec(input_shape.to_vec(), grad_input).map_err(Into::into)
    }

    /// Softmax backward: `grad_input = softmax_output * (upstream - sum(upstream * softmax_output))`.
    fn softmax_backward(
        &self,
        upstream: &Tensor,
        forward_output: &Tensor,
    ) -> Result<Tensor, KernelError> {
        let shape = forward_output.shape();
        if shape.is_empty() {
            return Err(KernelError::InvalidSoftmaxRank { got_rank: 0 });
        }
        let d = *shape.last().expect("checked non-empty");
        let u_data = upstream.data();
        let s_data = forward_output.data();
        let total = shape.iter().product::<usize>();
        let num_vecs = total / d;
        let mut out = vec![0.0f32; total];

        for v in 0..num_vecs {
            let base = v * d;
            let dot: f32 = (0..d).map(|i| u_data[base + i] * s_data[base + i]).sum();
            for i in 0..d {
                out[base + i] = s_data[base + i] * (u_data[base + i] - dot);
            }
        }
        Tensor::from_vec(shape.to_vec(), out).map_err(Into::into)
    }

    /// Embedding backward: scatter upstream gradients back to the embedding table.
    fn embedding_backward(
        &self,
        upstream: &Tensor,
        indices: &[usize],
        num_embeddings: usize,
        embed_dim: usize,
    ) -> Result<Tensor, KernelError> {
        let u_data = upstream.data();
        let mut grad_table = vec![0.0f32; num_embeddings * embed_dim];
        for (i, &idx) in indices.iter().enumerate() {
            if idx >= num_embeddings {
                return Err(KernelError::EmbeddingIndexOutOfBounds {
                    index: idx,
                    vocab_size: num_embeddings,
                });
            }
            let src_base = i * embed_dim;
            let dst_base = idx * embed_dim;
            for d in 0..embed_dim {
                grad_table[dst_base + d] += u_data[src_base + d];
            }
        }
        Tensor::from_vec(vec![num_embeddings, embed_dim], grad_table).map_err(Into::into)
    }

    /// Attention backward: compute gradients for query, key, and value.
    fn attention_backward(
        &self,
        upstream: &Tensor,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        attn_weights: &Tensor,
    ) -> Result<(Tensor, Tensor, Tensor), KernelError> {
        // upstream: [seq_q, d_v], attn_weights: [seq_q, seq_k]
        // grad_value = attn_weights^T @ upstream  [seq_k, d_v]
        let at = self.transpose_2d(attn_weights)?;
        let grad_value = self.matmul_2d(&at, upstream)?;

        // grad_attn = upstream @ value^T  [seq_q, seq_k]
        let vt = self.transpose_2d(value)?;
        let grad_attn = self.matmul_2d(upstream, &vt)?;

        // softmax backward on attn_weights
        let grad_scores = self.softmax_backward(&grad_attn, attn_weights)?;

        // scale by 1/sqrt(d_k)
        let d_k = query.shape().get(1).copied().unwrap_or(1);
        let scale = 1.0 / (d_k as f32).sqrt();
        let grad_scores_scaled = self.mul_scalar(&grad_scores, scale);

        // grad_query = grad_scores_scaled @ key  [seq_q, d_k]
        let grad_query = self.matmul_2d(&grad_scores_scaled, key)?;

        // grad_key = grad_scores_scaled^T @ query  [seq_k, d_k]
        let gst = self.transpose_2d(&grad_scores_scaled)?;
        let grad_key = self.matmul_2d(&gst, query)?;

        Ok((grad_query, grad_key, grad_value))
    }

    /// RNN backward (BPTT with tanh activation).
    ///
    /// Returns the gradient tensor for the input with shape `[seq_len, input_size]`.
    /// `upstream` has shape `[seq_len, hidden_size]`.
    /// `forward_input` has shape `[seq_len, input_size]`.
    /// `hidden` stores the concatenated hidden states `[(seq_len+1) * hidden_size]`
    /// where entry 0 is h_0 and entry t+1 is h_t (post-tanh).
    /// `weights_ih` has shape `[input_size, hidden_size]`.
    /// `weights_hh` has shape `[hidden_size, hidden_size]`.
    fn rnn_backward(
        &self,
        upstream: &Tensor,
        forward_input: &Tensor,
        hidden: &Tensor,
        weights_ih: &Tensor,
        weights_hh: &Tensor,
    ) -> Result<Tensor, KernelError> {
        let in_shape = forward_input.shape();
        if in_shape.len() != 2 {
            return Err(KernelError::UnsupportedOperation(format!(
                "rnn/lstm/gru backward requires rank-2 input, got rank {}",
                in_shape.len()
            )));
        }
        let seq_len = in_shape[0];
        let input_size = in_shape[1];
        let hidden_size = weights_ih.shape().get(1).copied().unwrap_or(0);
        if hidden_size == 0 {
            return Err(KernelError::UnsupportedOperation(format!(
                "rnn_backward: weights_ih second dim must be > 0, got shape {:?}",
                weights_ih.shape()
            )));
        }

        let _in_data = forward_input.data();
        let wih_data = weights_ih.data();
        let whh_data = weights_hh.data();
        let up_data = upstream.data();
        let h_data = hidden.data();

        let mut grad_input = vec![0.0f32; seq_len * input_size];
        let mut dh_next = vec![0.0f32; hidden_size];

        for t in (0..seq_len).rev() {
            // h_t = hidden[(t+1)*hidden_size .. (t+2)*hidden_size]
            let h_t_base = (t + 1) * hidden_size;

            // dh = upstream[t] + dh_next
            // d_raw = dh * (1 - h_t^2)  (tanh derivative)
            let mut d_raw = vec![0.0f32; hidden_size];
            for j in 0..hidden_size {
                let dh_j = up_data[t * hidden_size + j] + dh_next[j];
                let h_val = h_data[h_t_base + j];
                d_raw[j] = dh_j * (1.0 - h_val * h_val);
            }

            // grad_input[t] = d_raw @ W_ih^T
            let x_base = t * input_size;
            for i in 0..input_size {
                let mut s = 0.0f32;
                for j in 0..hidden_size {
                    s += d_raw[j] * wih_data[i * hidden_size + j];
                }
                grad_input[x_base + i] = s;
            }

            // dh_next = d_raw @ W_hh^T
            dh_next.fill(0.0);
            for i in 0..hidden_size {
                for j in 0..hidden_size {
                    dh_next[i] += d_raw[j] * whh_data[i * hidden_size + j];
                }
            }
        }

        Ok(Tensor::from_vec(in_shape.to_vec(), grad_input)?)
    }

    /// LSTM backward (BPTT through forget/input/output/cell gates).
    ///
    /// Returns the gradient tensor for the input with shape `[seq_len, input_size]`.
    /// `upstream` has shape `[seq_len, hidden_size]`.
    /// `forward_input` has shape `[seq_len, input_size]`.
    /// `hidden` stores concatenated hidden states `[(seq_len+1) * hidden_size]`.
    /// `cell` stores concatenated cell states `[(seq_len+1) * hidden_size]`.
    /// `weights_ih` has shape `[input_size, 4*hidden_size]`.
    /// `weights_hh` has shape `[hidden_size, 4*hidden_size]`.
    fn lstm_backward(
        &self,
        upstream: &Tensor,
        forward_input: &Tensor,
        hidden: &Tensor,
        cell: &Tensor,
        weights_ih: &Tensor,
        weights_hh: &Tensor,
    ) -> Result<Tensor, KernelError> {
        let in_shape = forward_input.shape();
        if in_shape.len() != 2 {
            return Err(KernelError::UnsupportedOperation(format!(
                "rnn/lstm/gru backward requires rank-2 input, got rank {}",
                in_shape.len()
            )));
        }
        let seq_len = in_shape[0];
        let input_size = in_shape[1];
        let h4 = weights_ih.shape().get(1).copied().unwrap_or(0);
        let hidden_size = h4 / 4;
        if hidden_size == 0 || h4 % 4 != 0 {
            return Err(KernelError::UnsupportedOperation(format!(
                "lstm_backward: weights_ih second dim must be divisible by 4, got shape {:?}",
                weights_ih.shape()
            )));
        }

        let in_data = forward_input.data();
        let wih_data = weights_ih.data();
        let whh_data = weights_hh.data();
        let up_data = upstream.data();
        let h_data = hidden.data();
        let c_data = cell.data();

        let mut grad_input = vec![0.0f32; seq_len * input_size];
        let mut dh_next = vec![0.0f32; hidden_size];
        let mut dc_next = vec![0.0f32; hidden_size];

        // Recompute gates for each timestep during backward pass.
        // Gate layout in weights: [i, f, g, o] each of size hidden_size.
        for t in (0..seq_len).rev() {
            let h_prev_base = t * hidden_size;
            let c_prev_base = t * hidden_size;
            let c_t_base = (t + 1) * hidden_size;
            let x_base = t * input_size;

            // Recompute gate pre-activations and activations
            let mut i_gate = vec![0.0f32; hidden_size];
            let mut f_gate = vec![0.0f32; hidden_size];
            let mut g_gate = vec![0.0f32; hidden_size];
            let mut o_gate = vec![0.0f32; hidden_size];

            for j in 0..hidden_size {
                let mut sum_i = 0.0f32;
                let mut sum_f = 0.0f32;
                let mut sum_g = 0.0f32;
                let mut sum_o = 0.0f32;
                for i in 0..input_size {
                    sum_i += in_data[x_base + i] * wih_data[i * h4 + j];
                    sum_f += in_data[x_base + i] * wih_data[i * h4 + hidden_size + j];
                    sum_g += in_data[x_base + i] * wih_data[i * h4 + 2 * hidden_size + j];
                    sum_o += in_data[x_base + i] * wih_data[i * h4 + 3 * hidden_size + j];
                }
                for i in 0..hidden_size {
                    sum_i += h_data[h_prev_base + i] * whh_data[i * h4 + j];
                    sum_f += h_data[h_prev_base + i] * whh_data[i * h4 + hidden_size + j];
                    sum_g += h_data[h_prev_base + i] * whh_data[i * h4 + 2 * hidden_size + j];
                    sum_o += h_data[h_prev_base + i] * whh_data[i * h4 + 3 * hidden_size + j];
                }
                // sigmoid for i, f, o; tanh for g
                i_gate[j] = 1.0 / (1.0 + (-sum_i).exp());
                f_gate[j] = 1.0 / (1.0 + (-sum_f).exp());
                g_gate[j] = sum_g.tanh();
                o_gate[j] = 1.0 / (1.0 + (-sum_o).exp());
            }

            // dh = upstream[t] + dh_next
            let mut dh = vec![0.0f32; hidden_size];
            for j in 0..hidden_size {
                dh[j] = up_data[t * hidden_size + j] + dh_next[j];
            }

            // d_o = dh * tanh(c_t) * o * (1 - o)
            let mut d_o = vec![0.0f32; hidden_size];
            for j in 0..hidden_size {
                let tanh_c = c_data[c_t_base + j].tanh();
                d_o[j] = dh[j] * tanh_c * o_gate[j] * (1.0 - o_gate[j]);
            }

            // dc = dh * o * (1 - tanh(c_t)^2) + dc_next
            let mut dc = vec![0.0f32; hidden_size];
            for j in 0..hidden_size {
                let tanh_c = c_data[c_t_base + j].tanh();
                dc[j] = dh[j] * o_gate[j] * (1.0 - tanh_c * tanh_c) + dc_next[j];
            }

            // d_f = dc * c_prev * f * (1 - f)
            let mut d_f = vec![0.0f32; hidden_size];
            for j in 0..hidden_size {
                d_f[j] = dc[j] * c_data[c_prev_base + j] * f_gate[j] * (1.0 - f_gate[j]);
            }

            // d_i = dc * g * i * (1 - i)
            let mut d_i = vec![0.0f32; hidden_size];
            for j in 0..hidden_size {
                d_i[j] = dc[j] * g_gate[j] * i_gate[j] * (1.0 - i_gate[j]);
            }

            // d_g = dc * i * (1 - g^2)
            let mut d_g = vec![0.0f32; hidden_size];
            for j in 0..hidden_size {
                d_g[j] = dc[j] * i_gate[j] * (1.0 - g_gate[j] * g_gate[j]);
            }

            // Concatenated gate gradients [d_i, d_f, d_g, d_o]
            let mut d_gates = vec![0.0f32; h4];
            for j in 0..hidden_size {
                d_gates[j] = d_i[j];
                d_gates[hidden_size + j] = d_f[j];
                d_gates[2 * hidden_size + j] = d_g[j];
                d_gates[3 * hidden_size + j] = d_o[j];
            }

            // grad_input[t] = d_gates @ W_ih^T
            for i in 0..input_size {
                let mut s = 0.0f32;
                for j in 0..h4 {
                    s += d_gates[j] * wih_data[i * h4 + j];
                }
                grad_input[x_base + i] = s;
            }

            // dh_next = d_gates @ W_hh^T
            dh_next.fill(0.0);
            for i in 0..hidden_size {
                for j in 0..h4 {
                    dh_next[i] += d_gates[j] * whh_data[i * h4 + j];
                }
            }

            // dc_next = dc * f
            for j in 0..hidden_size {
                dc_next[j] = dc[j] * f_gate[j];
            }
        }

        Ok(Tensor::from_vec(in_shape.to_vec(), grad_input)?)
    }

    /// GRU backward (BPTT through reset/update gates).
    ///
    /// Returns the gradient tensor for the input with shape `[seq_len, input_size]`.
    /// `upstream` has shape `[seq_len, hidden_size]`.
    /// `forward_input` has shape `[seq_len, input_size]`.
    /// `hidden` stores concatenated hidden states `[(seq_len+1) * hidden_size]`.
    /// `weights_ih` has shape `[input_size, 3*hidden_size]`.
    /// `weights_hh` has shape `[hidden_size, 3*hidden_size]`.
    fn gru_backward(
        &self,
        upstream: &Tensor,
        forward_input: &Tensor,
        hidden: &Tensor,
        weights_ih: &Tensor,
        weights_hh: &Tensor,
    ) -> Result<Tensor, KernelError> {
        let in_shape = forward_input.shape();
        if in_shape.len() != 2 {
            return Err(KernelError::UnsupportedOperation(format!(
                "rnn/lstm/gru backward requires rank-2 input, got rank {}",
                in_shape.len()
            )));
        }
        let seq_len = in_shape[0];
        let input_size = in_shape[1];
        let h3 = weights_ih.shape().get(1).copied().unwrap_or(0);
        let hidden_size = h3 / 3;
        if hidden_size == 0 || h3 % 3 != 0 {
            return Err(KernelError::UnsupportedOperation(format!(
                "gru_backward: weights_ih second dim must be divisible by 3, got shape {:?}",
                weights_ih.shape()
            )));
        }

        let in_data = forward_input.data();
        let wih_data = weights_ih.data();
        let whh_data = weights_hh.data();
        let up_data = upstream.data();
        let h_data = hidden.data();

        let mut grad_input = vec![0.0f32; seq_len * input_size];
        let mut dh_next = vec![0.0f32; hidden_size];

        for t in (0..seq_len).rev() {
            let h_prev_base = t * hidden_size;
            let x_base = t * input_size;

            // Recompute gates: r (reset), z (update), n (candidate)
            let mut r_gate = vec![0.0f32; hidden_size];
            let mut z_gate = vec![0.0f32; hidden_size];
            let mut n_cand = vec![0.0f32; hidden_size];

            // r and z: sigmoid(x @ W_ih[:, :2H] + h_prev @ W_hh[:, :2H])
            for j in 0..hidden_size {
                let mut sum_r = 0.0f32;
                let mut sum_z = 0.0f32;
                for i in 0..input_size {
                    sum_r += in_data[x_base + i] * wih_data[i * h3 + j];
                    sum_z += in_data[x_base + i] * wih_data[i * h3 + hidden_size + j];
                }
                for i in 0..hidden_size {
                    sum_r += h_data[h_prev_base + i] * whh_data[i * h3 + j];
                    sum_z += h_data[h_prev_base + i] * whh_data[i * h3 + hidden_size + j];
                }
                r_gate[j] = 1.0 / (1.0 + (-sum_r).exp());
                z_gate[j] = 1.0 / (1.0 + (-sum_z).exp());
            }

            // n: tanh(x @ W_ih[:, 2H:] + r * (h_prev @ W_hh[:, 2H:]))
            // First compute h_proj_n = h_prev @ W_hh[:, 2H:]
            let mut h_proj_n = vec![0.0f32; hidden_size];
            for j in 0..hidden_size {
                let mut sum = 0.0f32;
                for i in 0..hidden_size {
                    sum += h_data[h_prev_base + i] * whh_data[i * h3 + 2 * hidden_size + j];
                }
                h_proj_n[j] = sum;
            }
            for j in 0..hidden_size {
                let mut sum_n = 0.0f32;
                for i in 0..input_size {
                    sum_n += in_data[x_base + i] * wih_data[i * h3 + 2 * hidden_size + j];
                }
                sum_n += r_gate[j] * h_proj_n[j];
                n_cand[j] = sum_n.tanh();
            }

            // dh = upstream[t] + dh_next
            let mut dh = vec![0.0f32; hidden_size];
            for j in 0..hidden_size {
                dh[j] = up_data[t * hidden_size + j] + dh_next[j];
            }

            // dn = dh * (1 - z)
            let mut dn = vec![0.0f32; hidden_size];
            for j in 0..hidden_size {
                dn[j] = dh[j] * (1.0 - z_gate[j]);
            }

            // dz = dh * (h_prev - n) * z * (1 - z)
            let mut dz = vec![0.0f32; hidden_size];
            for j in 0..hidden_size {
                dz[j] =
                    dh[j] * (h_data[h_prev_base + j] - n_cand[j]) * z_gate[j] * (1.0 - z_gate[j]);
            }

            // dn_raw = dn * (1 - n^2)  (tanh derivative)
            let mut dn_raw = vec![0.0f32; hidden_size];
            for j in 0..hidden_size {
                dn_raw[j] = dn[j] * (1.0 - n_cand[j] * n_cand[j]);
            }

            // dr = dn_raw * h_proj_n * r * (1 - r)
            let mut dr = vec![0.0f32; hidden_size];
            for j in 0..hidden_size {
                dr[j] = dn_raw[j] * h_proj_n[j] * r_gate[j] * (1.0 - r_gate[j]);
            }

            // Input projection gradients: [dr, dz, dn_raw]
            let mut d_x_proj = vec![0.0f32; h3];
            for j in 0..hidden_size {
                d_x_proj[j] = dr[j];
                d_x_proj[hidden_size + j] = dz[j];
                d_x_proj[2 * hidden_size + j] = dn_raw[j];
            }

            // grad_input[t] = d_x_proj @ W_ih^T
            for i in 0..input_size {
                let mut s = 0.0f32;
                for j in 0..h3 {
                    s += d_x_proj[j] * wih_data[i * h3 + j];
                }
                grad_input[x_base + i] = s;
            }

            // Hidden projection gradients: [dr, dz, dn_raw * r]
            let mut d_h_proj = vec![0.0f32; h3];
            for j in 0..hidden_size {
                d_h_proj[j] = dr[j];
                d_h_proj[hidden_size + j] = dz[j];
                d_h_proj[2 * hidden_size + j] = dn_raw[j] * r_gate[j];
            }

            // dh_next = d_h_proj @ W_hh^T + dh * z
            dh_next.fill(0.0);
            for i in 0..hidden_size {
                let mut s = 0.0f32;
                for j in 0..h3 {
                    s += d_h_proj[j] * whh_data[i * h3 + j];
                }
                dh_next[i] = s + dh[i] * z_gate[i];
            }
        }

        Ok(Tensor::from_vec(in_shape.to_vec(), grad_input)?)
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
