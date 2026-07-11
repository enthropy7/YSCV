use rustc_hash::FxHashMap;
use std::collections::hash_map::Entry;

use yscv_autograd::{Graph, NodeId};
use yscv_tensor::Tensor;

use super::validate::{validate_lr, validate_momentum};
use super::{LearningRate, OptimError};

/// Layer-wise Adaptive Rate Scaling (LARS) optimizer.
///
/// Scales the learning rate per layer using the ratio of parameter norm to
/// gradient norm, enabling stable training with very large batch sizes.
#[derive(Debug, Clone)]
pub struct Lars {
    base_lr: f32,
    momentum: f32,
    weight_decay: f32,
    trust_coefficient: f32,
    velocity: FxHashMap<u64, Tensor>,
}

impl Lars {
    /// Creates LARS with required base learning rate.
    pub fn new(base_lr: f32) -> Result<Self, OptimError> {
        validate_lr(base_lr)?;
        Ok(Self {
            base_lr,
            momentum: 0.0,
            weight_decay: 0.0,
            trust_coefficient: 0.001,
            velocity: FxHashMap::default(),
        })
    }

    /// Sets momentum factor in `[0, 1)`.
    pub fn with_momentum(mut self, momentum: f32) -> Result<Self, OptimError> {
        validate_momentum(momentum)?;
        self.momentum = momentum;
        Ok(self)
    }

    /// Sets L2 weight decay factor in `[0, +inf)`.
    pub fn with_weight_decay(mut self, weight_decay: f32) -> Result<Self, OptimError> {
        if !weight_decay.is_finite() || weight_decay < 0.0 {
            return Err(OptimError::InvalidWeightDecay { weight_decay });
        }
        self.weight_decay = weight_decay;
        Ok(self)
    }

    /// Sets trust coefficient for the local learning rate scaling.
    pub fn with_trust_coefficient(mut self, trust_coefficient: f32) -> Result<Self, OptimError> {
        if !trust_coefficient.is_finite() || trust_coefficient <= 0.0 {
            return Err(OptimError::InvalidEpsilon {
                epsilon: trust_coefficient,
            });
        }
        self.trust_coefficient = trust_coefficient;
        Ok(self)
    }

    /// Drops optimizer state (for example when restarting training).
    pub fn clear_state(&mut self) {
        self.velocity.clear();
    }

    /// Returns current learning rate.
    pub fn learning_rate(&self) -> f32 {
        self.base_lr
    }

    /// Overrides current learning rate.
    pub fn set_learning_rate(&mut self, lr: f32) -> Result<(), OptimError> {
        validate_lr(lr)?;
        self.base_lr = lr;
        Ok(())
    }

    /// Applies one update to raw tensor weights.
    pub fn step(
        &mut self,
        parameter_id: u64,
        weights: &mut Tensor,
        grad: &Tensor,
    ) -> Result<(), OptimError> {
        if weights.shape() != grad.shape() {
            return Err(OptimError::ShapeMismatch {
                weights: weights.shape().to_vec(),
                grad: grad.shape().to_vec(),
            });
        }

        // Compute weight norm and gradient norm.
        let w_data = weights.data();
        let g_data = grad.data();

        let w_norm = w_data.iter().map(|x| x * x).sum::<f32>().sqrt();
        let g_norm = g_data.iter().map(|x| x * x).sum::<f32>().sqrt();

        // Compute local learning rate.
        let local_lr = if w_norm > 0.0 && g_norm > 0.0 {
            self.trust_coefficient * w_norm / (g_norm + self.weight_decay * w_norm)
        } else {
            1.0
        };

        // Compute gradient with weight decay: g_with_wd = g + weight_decay * w
        let mut g_with_wd = g_data.to_vec();
        if self.weight_decay != 0.0 {
            for (gv, wv) in g_with_wd.iter_mut().zip(w_data.iter()) {
                *gv += self.weight_decay * *wv;
            }
        }

        let effective_lr = local_lr * self.base_lr;

        // Update velocity and weights.
        let velocity = match self.velocity.entry(parameter_id) {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => entry.insert(Tensor::zeros(weights.shape().to_vec())?),
        };

        if velocity.shape() != weights.shape() {
            *velocity = Tensor::zeros(weights.shape().to_vec())?;
        }

        let v_data = velocity.data_mut();
        let weights_data = weights.data_mut();

        for i in 0..weights_data.len() {
            v_data[i] = self.momentum * v_data[i] + effective_lr * g_with_wd[i];
            weights_data[i] -= v_data[i];
        }

        Ok(())
    }

    /// Applies one update to a trainable graph node by its `NodeId`.
    pub fn step_graph_node(&mut self, graph: &mut Graph, node: NodeId) -> Result<(), OptimError> {
        if !graph.requires_grad(node)? {
            return Ok(());
        }

        let grad = match graph.grad(node)? {
            Some(grad) => grad.clone(),
            None => return Err(OptimError::MissingGradient { node: node.0 }),
        };
        let weights = graph.value_mut(node)?;
        self.step(node.0 as u64, weights, &grad)
    }
}

impl LearningRate for Lars {
    fn learning_rate(&self) -> f32 {
        Lars::learning_rate(self)
    }

    fn set_learning_rate(&mut self, lr: f32) -> Result<(), OptimError> {
        Lars::set_learning_rate(self, lr)
    }
}
