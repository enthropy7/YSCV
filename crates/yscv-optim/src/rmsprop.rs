use rustc_hash::FxHashMap;
use std::collections::hash_map::Entry;

use yscv_autograd::{Graph, NodeId};
use yscv_tensor::Tensor;

use super::validate::{validate_epsilon, validate_lr, validate_momentum, validate_rmsprop_alpha};
use super::{LearningRate, OptimError};

#[derive(Debug, Clone)]
struct RmsPropState {
    square_avg: Tensor,
    grad_avg: Tensor,
    momentum_buffer: Tensor,
}

impl RmsPropState {
    fn new(shape: &[usize]) -> Result<Self, OptimError> {
        Ok(Self {
            square_avg: Tensor::zeros(shape.to_vec())?,
            grad_avg: Tensor::zeros(shape.to_vec())?,
            momentum_buffer: Tensor::zeros(shape.to_vec())?,
        })
    }

    fn reset(&mut self, shape: &[usize]) -> Result<(), OptimError> {
        *self = Self::new(shape)?;
        Ok(())
    }
}

/// RMSProp optimizer with optional momentum, weight decay, and centered variance.
#[derive(Debug, Clone)]
pub struct RmsProp {
    lr: f32,
    alpha: f32,
    epsilon: f32,
    weight_decay: f32,
    momentum: f32,
    centered: bool,
    state: FxHashMap<u64, RmsPropState>,
}

impl RmsProp {
    /// Creates RMSProp with required learning rate.
    pub fn new(lr: f32) -> Result<Self, OptimError> {
        validate_lr(lr)?;
        Ok(Self {
            lr,
            alpha: 0.99,
            epsilon: 1e-8,
            weight_decay: 0.0,
            momentum: 0.0,
            centered: false,
            state: FxHashMap::default(),
        })
    }

    /// Sets RMSProp smoothing factor in `[0, 1)`.
    pub fn with_alpha(mut self, alpha: f32) -> Result<Self, OptimError> {
        validate_rmsprop_alpha(alpha)?;
        self.alpha = alpha;
        Ok(self)
    }

    /// Sets epsilon value, must be finite and `> 0`.
    pub fn with_epsilon(mut self, epsilon: f32) -> Result<Self, OptimError> {
        validate_epsilon(epsilon)?;
        self.epsilon = epsilon;
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

    /// Sets momentum factor in `[0, 1)`.
    pub fn with_momentum(mut self, momentum: f32) -> Result<Self, OptimError> {
        validate_momentum(momentum)?;
        self.momentum = momentum;
        Ok(self)
    }

    /// Enables/disables centered RMSProp variant.
    pub fn with_centered(mut self, centered: bool) -> Self {
        self.centered = centered;
        self
    }

    /// Drops optimizer state (for example when restarting training).
    pub fn clear_state(&mut self) {
        self.state.clear();
    }

    /// Returns current learning rate.
    pub fn learning_rate(&self) -> f32 {
        self.lr
    }

    /// Overrides current learning rate.
    pub fn set_learning_rate(&mut self, lr: f32) -> Result<(), OptimError> {
        validate_lr(lr)?;
        self.lr = lr;
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

        let state = match self.state.entry(parameter_id) {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => entry.insert(RmsPropState::new(weights.shape())?),
        };
        if state.square_avg.shape() != weights.shape() {
            state.reset(weights.shape())?;
        }

        let grad_values = grad.data();
        let weights_data = weights.data_mut();
        let square_avg = state.square_avg.data_mut();
        let grad_avg = state.grad_avg.data_mut();
        let momentum_buffer = state.momentum_buffer.data_mut();

        let alpha = self.alpha;
        let one_minus_alpha = 1.0 - self.alpha;

        for index in 0..weights_data.len() {
            let grad_value = grad_values[index] + self.weight_decay * weights_data[index];
            square_avg[index] =
                alpha * square_avg[index] + one_minus_alpha * grad_value * grad_value;

            let avg = if self.centered {
                grad_avg[index] = alpha * grad_avg[index] + one_minus_alpha * grad_value;
                (square_avg[index] - grad_avg[index] * grad_avg[index]).max(0.0)
            } else {
                square_avg[index]
            };

            let denom = avg.sqrt() + self.epsilon;
            let normalized = grad_value / denom;
            let update = if self.momentum != 0.0 {
                let next = self.momentum * momentum_buffer[index] + normalized;
                momentum_buffer[index] = next;
                next
            } else {
                normalized
            };
            weights_data[index] -= self.lr * update;
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

impl LearningRate for RmsProp {
    fn learning_rate(&self) -> f32 {
        RmsProp::learning_rate(self)
    }

    fn set_learning_rate(&mut self, lr: f32) -> Result<(), OptimError> {
        RmsProp::set_learning_rate(self, lr)
    }
}
