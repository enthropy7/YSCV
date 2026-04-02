use yscv_autograd::{Graph, NodeId};
use yscv_kernels::BatchNorm2dParams;
use yscv_tensor::Tensor;

use crate::ModelError;

/// 2D batch normalization layer (NHWC layout).
///
/// Supports both inference-mode (raw tensor) and graph-mode (autograd training).
/// Stores learned gamma/beta and running mean/variance.
#[derive(Debug, Clone, PartialEq)]
pub struct BatchNorm2dLayer {
    num_features: usize,
    epsilon: f32,
    gamma: Tensor,
    beta: Tensor,
    running_mean: Tensor,
    running_var: Tensor,
    gamma_node: Option<NodeId>,
    beta_node: Option<NodeId>,
    mean_node: Option<NodeId>,
    var_node: Option<NodeId>,
}

impl BatchNorm2dLayer {
    pub fn new(
        num_features: usize,
        epsilon: f32,
        gamma: Tensor,
        beta: Tensor,
        running_mean: Tensor,
        running_var: Tensor,
    ) -> Result<Self, ModelError> {
        let expected = vec![num_features];
        for (name, t) in [
            ("gamma", &gamma),
            ("beta", &beta),
            ("running_mean", &running_mean),
            ("running_var", &running_var),
        ] {
            if t.shape() != expected {
                return Err(ModelError::InvalidParameterShape {
                    parameter: name,
                    expected: expected.clone(),
                    got: t.shape().to_vec(),
                });
            }
        }
        if !epsilon.is_finite() || epsilon <= 0.0 {
            return Err(ModelError::InvalidBatchNormEpsilon { epsilon });
        }
        Ok(Self {
            num_features,
            epsilon,
            gamma,
            beta,
            running_mean,
            running_var,
            gamma_node: None,
            beta_node: None,
            mean_node: None,
            var_node: None,
        })
    }

    /// Unit-scale/zero-shift initialization with zero running statistics.
    pub fn identity_init(num_features: usize, epsilon: f32) -> Result<Self, ModelError> {
        let gamma = Tensor::filled(vec![num_features], 1.0)?;
        let beta = Tensor::zeros(vec![num_features])?;
        let running_mean = Tensor::zeros(vec![num_features])?;
        let running_var = Tensor::filled(vec![num_features], 1.0)?;
        Self::new(
            num_features,
            epsilon,
            gamma,
            beta,
            running_mean,
            running_var,
        )
    }

    /// Registers gamma/beta as graph variables (trainable), running stats as constants.
    pub fn register_params(&mut self, graph: &mut Graph) {
        self.gamma_node = Some(graph.variable(self.gamma.clone()));
        self.beta_node = Some(graph.variable(self.beta.clone()));
        self.mean_node = Some(graph.constant(self.running_mean.clone()));
        self.var_node = Some(graph.constant(self.running_var.clone()));
    }

    pub fn sync_from_graph(&mut self, graph: &Graph) -> Result<(), ModelError> {
        if let Some(g_id) = self.gamma_node {
            self.gamma = graph.value(g_id)?.clone();
        }
        if let Some(b_id) = self.beta_node {
            self.beta = graph.value(b_id)?.clone();
        }
        Ok(())
    }

    pub fn num_features(&self) -> usize {
        self.num_features
    }
    pub fn epsilon(&self) -> f32 {
        self.epsilon
    }
    pub fn gamma(&self) -> &Tensor {
        &self.gamma
    }
    pub fn beta(&self) -> &Tensor {
        &self.beta
    }
    pub fn running_mean(&self) -> &Tensor {
        &self.running_mean
    }
    pub fn running_var(&self) -> &Tensor {
        &self.running_var
    }
    pub fn gamma_mut(&mut self) -> &mut Tensor {
        &mut self.gamma
    }
    pub fn beta_mut(&mut self) -> &mut Tensor {
        &mut self.beta
    }
    pub fn running_mean_mut(&mut self) -> &mut Tensor {
        &mut self.running_mean
    }
    pub fn running_var_mut(&mut self) -> &mut Tensor {
        &mut self.running_var
    }
    pub fn gamma_node(&self) -> Option<NodeId> {
        self.gamma_node
    }
    pub fn beta_node(&self) -> Option<NodeId> {
        self.beta_node
    }

    pub fn forward(&self, graph: &mut Graph, input: NodeId) -> Result<NodeId, ModelError> {
        let g_id = self.gamma_node.ok_or(ModelError::ParamsNotRegistered {
            layer: "BatchNorm2d",
        })?;
        let b_id = self.beta_node.ok_or(ModelError::ParamsNotRegistered {
            layer: "BatchNorm2d",
        })?;
        let m_id = self.mean_node.ok_or(ModelError::ParamsNotRegistered {
            layer: "BatchNorm2d",
        })?;
        let v_id = self.var_node.ok_or(ModelError::ParamsNotRegistered {
            layer: "BatchNorm2d",
        })?;
        graph
            .batch_norm2d_nhwc(input, g_id, b_id, m_id, v_id, self.epsilon)
            .map_err(Into::into)
    }

    pub fn forward_inference(&self, input: &Tensor) -> Result<Tensor, ModelError> {
        yscv_kernels::batch_norm2d_nhwc(
            input,
            BatchNorm2dParams {
                gamma: &self.gamma,
                beta: &self.beta,
                mean: &self.running_mean,
                variance: &self.running_var,
                epsilon: self.epsilon,
            },
        )
        .map_err(Into::into)
    }
}

/// Layer normalization over the last dimension.
#[derive(Debug, Clone)]
pub struct LayerNormLayer {
    normalized_shape: usize,
    eps: f32,
    gamma: NodeId,
    beta: NodeId,
}

impl LayerNormLayer {
    pub fn new(graph: &mut Graph, normalized_shape: usize, eps: f32) -> Result<Self, ModelError> {
        let gamma = graph.variable(Tensor::ones(vec![normalized_shape])?);
        let beta = graph.variable(Tensor::zeros(vec![normalized_shape])?);
        Ok(Self {
            normalized_shape,
            eps,
            gamma,
            beta,
        })
    }

    pub fn normalized_shape(&self) -> usize {
        self.normalized_shape
    }
    pub fn gamma_node(&self) -> NodeId {
        self.gamma
    }
    pub fn beta_node(&self) -> NodeId {
        self.beta
    }

    pub fn forward(&self, graph: &mut Graph, input: NodeId) -> Result<NodeId, ModelError> {
        graph
            .layer_norm(input, self.gamma, self.beta, self.eps)
            .map_err(Into::into)
    }

    pub fn forward_inference(&self, graph: &Graph, input: &Tensor) -> Result<Tensor, ModelError> {
        let shape = input.shape();
        let last_dim = *shape.last().ok_or(ModelError::InvalidInputShape {
            expected_features: self.normalized_shape,
            got: shape.to_vec(),
        })?;
        if last_dim != self.normalized_shape {
            return Err(ModelError::InvalidInputShape {
                expected_features: self.normalized_shape,
                got: shape.to_vec(),
            });
        }
        let data = input.data();
        let gamma = graph.value(self.gamma)?.data().to_vec();
        let beta = graph.value(self.beta)?.data().to_vec();
        let num_groups = data.len() / last_dim;
        let mut out = vec![0.0f32; data.len()];
        for g in 0..num_groups {
            let base = g * last_dim;
            let slice = &data[base..base + last_dim];
            let mean = slice.iter().sum::<f32>() / last_dim as f32;
            let var = slice.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / last_dim as f32;
            let inv_std = 1.0 / (var + self.eps).sqrt();
            for i in 0..last_dim {
                out[base + i] = (slice[i] - mean) * inv_std * gamma[i] + beta[i];
            }
        }
        Ok(Tensor::from_vec(shape.to_vec(), out)?)
    }
}

/// Group normalization: divides channels into groups and normalizes within each group.
#[derive(Debug, Clone)]
pub struct GroupNormLayer {
    num_groups: usize,
    num_channels: usize,
    eps: f32,
    gamma: NodeId,
    beta: NodeId,
}

impl GroupNormLayer {
    pub fn new(
        graph: &mut Graph,
        num_groups: usize,
        num_channels: usize,
        eps: f32,
    ) -> Result<Self, ModelError> {
        if !num_channels.is_multiple_of(num_groups) {
            return Err(ModelError::InvalidInputShape {
                expected_features: num_groups,
                got: vec![num_channels],
            });
        }
        let gamma = graph.variable(Tensor::ones(vec![num_channels])?);
        let beta = graph.variable(Tensor::zeros(vec![num_channels])?);
        Ok(Self {
            num_groups,
            num_channels,
            eps,
            gamma,
            beta,
        })
    }

    pub fn num_groups(&self) -> usize {
        self.num_groups
    }
    pub fn num_channels(&self) -> usize {
        self.num_channels
    }
    pub fn gamma_node(&self) -> NodeId {
        self.gamma
    }
    pub fn beta_node(&self) -> NodeId {
        self.beta
    }

    pub fn forward(&self, graph: &mut Graph, input: NodeId) -> Result<NodeId, ModelError> {
        graph
            .group_norm(input, self.gamma, self.beta, self.num_groups, self.eps)
            .map_err(Into::into)
    }

    /// Forward inference on NHWC input `[N, H, W, C]`.
    pub fn forward_inference(&self, graph: &Graph, input: &Tensor) -> Result<Tensor, ModelError> {
        let shape = input.shape();
        if shape.len() != 4 || shape[3] != self.num_channels {
            return Err(ModelError::InvalidInputShape {
                expected_features: self.num_channels,
                got: shape.to_vec(),
            });
        }
        let (n, h, w, c) = (shape[0], shape[1], shape[2], shape[3]);
        let channels_per_group = c / self.num_groups;
        let data = input.data();
        let gamma = graph.value(self.gamma)?.data().to_vec();
        let beta = graph.value(self.beta)?.data().to_vec();
        let mut out = vec![0.0f32; data.len()];

        // Two-pass GroupNorm: pass 1 uses Welford's online algorithm to compute
        // mean and variance in a single sweep, pass 2 normalises.
        for ni in 0..n {
            for gi in 0..self.num_groups {
                let c_start = gi * channels_per_group;
                let c_end = c_start + channels_per_group;

                // Pass 1: Welford online mean + variance
                let mut mean = 0.0f32;
                let mut m2 = 0.0f32;
                let mut count = 0u32;
                for hi in 0..h {
                    for wi in 0..w {
                        let base = ((ni * h + hi) * w + wi) * c;
                        for ci in c_start..c_end {
                            count += 1;
                            let val = data[base + ci];
                            let delta = val - mean;
                            mean += delta / count as f32;
                            let delta2 = val - mean;
                            m2 += delta * delta2;
                        }
                    }
                }
                let variance = m2 / count as f32;
                let inv_std = 1.0 / (variance + self.eps).sqrt();

                // Pass 2: normalise
                for hi in 0..h {
                    for wi in 0..w {
                        let base = ((ni * h + hi) * w + wi) * c;
                        for ci in c_start..c_end {
                            out[base + ci] =
                                (data[base + ci] - mean) * inv_std * gamma[ci] + beta[ci];
                        }
                    }
                }
            }
        }
        Ok(Tensor::from_vec(shape.to_vec(), out)?)
    }
}

/// Instance normalization (normalizes per-sample per-channel).
///
/// NHWC layout: `[batch, H, W, C]`.
#[derive(Debug, Clone, PartialEq)]
pub struct InstanceNormLayer {
    num_features: usize,
    eps: f32,
    gamma: Tensor,
    beta: Tensor,
    gamma_node: Option<NodeId>,
    beta_node: Option<NodeId>,
}

impl InstanceNormLayer {
    pub fn new(num_features: usize, eps: f32) -> Result<Self, ModelError> {
        Ok(Self {
            num_features,
            eps,
            gamma: Tensor::from_vec(vec![num_features], vec![1.0; num_features])?,
            beta: Tensor::zeros(vec![num_features])?,
            gamma_node: None,
            beta_node: None,
        })
    }

    pub fn register_params(&mut self, graph: &mut Graph) {
        self.gamma_node = Some(graph.variable(self.gamma.clone()));
        self.beta_node = Some(graph.variable(self.beta.clone()));
    }

    pub fn sync_from_graph(&mut self, graph: &Graph) -> Result<(), ModelError> {
        if let Some(g_id) = self.gamma_node {
            self.gamma = graph.value(g_id)?.clone();
        }
        if let Some(b_id) = self.beta_node {
            self.beta = graph.value(b_id)?.clone();
        }
        Ok(())
    }

    pub fn gamma_node(&self) -> Option<NodeId> {
        self.gamma_node
    }
    pub fn beta_node(&self) -> Option<NodeId> {
        self.beta_node
    }

    pub fn forward(&self, graph: &mut Graph, input: NodeId) -> Result<NodeId, ModelError> {
        let g_id = self.gamma_node.ok_or(ModelError::ParamsNotRegistered {
            layer: "InstanceNorm",
        })?;
        let b_id = self.beta_node.ok_or(ModelError::ParamsNotRegistered {
            layer: "InstanceNorm",
        })?;
        graph
            .instance_norm_nhwc(input, g_id, b_id, self.eps)
            .map_err(Into::into)
    }

    pub fn forward_inference(&self, input: &Tensor) -> Result<Tensor, ModelError> {
        let shape = input.shape();
        if shape.len() != 4 || shape[3] != self.num_features {
            return Err(ModelError::InvalidInputShape {
                expected_features: self.num_features,
                got: shape.to_vec(),
            });
        }
        let (batch, h, w, c) = (shape[0], shape[1], shape[2], shape[3]);
        let data = input.data();
        let g = self.gamma.data();
        let b = self.beta.data();
        let spatial = h * w;
        let mut out = vec![0.0f32; data.len()];

        for n in 0..batch {
            for ch in 0..c {
                let mut sum = 0.0f32;
                for s in 0..spatial {
                    let idx = (n * h * w + s) * c + ch;
                    sum += data[idx];
                }
                let mean = sum / spatial as f32;
                let mut var_sum = 0.0f32;
                for s in 0..spatial {
                    let idx = (n * h * w + s) * c + ch;
                    let d = data[idx] - mean;
                    var_sum += d * d;
                }
                let inv_std = 1.0 / (var_sum / spatial as f32 + self.eps).sqrt();
                for s in 0..spatial {
                    let idx = (n * h * w + s) * c + ch;
                    out[idx] = (data[idx] - mean) * inv_std * g[ch] + b[ch];
                }
            }
        }
        Ok(Tensor::from_vec(shape.to_vec(), out)?)
    }
}
