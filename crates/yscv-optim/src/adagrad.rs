use rustc_hash::FxHashMap;
use std::collections::hash_map::Entry;

use yscv_autograd::{Graph, NodeId};
use yscv_tensor::Tensor;

use super::validate::{validate_epsilon, validate_lr};
use super::{LearningRate, OptimError};

#[derive(Debug, Clone)]
struct AdagradState {
    sum_sq: Tensor,
}

impl AdagradState {
    fn new(shape: &[usize]) -> Result<Self, OptimError> {
        Ok(Self {
            sum_sq: Tensor::zeros(shape.to_vec())?,
        })
    }

    fn reset(&mut self, shape: &[usize]) -> Result<(), OptimError> {
        *self = Self::new(shape)?;
        Ok(())
    }
}

/// Adagrad optimizer with optional L2 weight decay.
#[derive(Debug, Clone)]
pub struct Adagrad {
    lr: f32,
    epsilon: f32,
    weight_decay: f32,
    state: FxHashMap<u64, AdagradState>,
}

impl Adagrad {
    /// Creates Adagrad with required learning rate.
    pub fn new(lr: f32) -> Result<Self, OptimError> {
        validate_lr(lr)?;
        Ok(Self {
            lr,
            epsilon: 1e-10,
            weight_decay: 0.0,
            state: FxHashMap::default(),
        })
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
            Entry::Vacant(entry) => entry.insert(AdagradState::new(weights.shape())?),
        };
        if state.sum_sq.shape() != weights.shape() {
            state.reset(weights.shape())?;
        }

        let sum_sq = state.sum_sq.data_mut();
        let grad_values = grad.data();
        let weights_data = weights.data_mut();

        for index in 0..weights_data.len() {
            let grad_value = grad_values[index] + self.weight_decay * weights_data[index];
            sum_sq[index] += grad_value * grad_value;
            weights_data[index] -= self.lr * grad_value / (sum_sq[index].sqrt() + self.epsilon);
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

impl LearningRate for Adagrad {
    fn learning_rate(&self) -> f32 {
        Adagrad::learning_rate(self)
    }

    fn set_learning_rate(&mut self, lr: f32) -> Result<(), OptimError> {
        Adagrad::set_learning_rate(self, lr)
    }
}
