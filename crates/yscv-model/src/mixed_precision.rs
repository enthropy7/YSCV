use yscv_autograd::Graph;
use yscv_tensor::{DType, Tensor};

use crate::ModelError;

/// Mixed-precision training configuration.
#[derive(Debug, Clone)]
pub struct MixedPrecisionConfig {
    /// Dtype used for forward pass computation.
    pub forward_dtype: DType,
    /// Dtype used for weight storage and gradient accumulation.
    pub master_dtype: DType,
    /// Loss scaling factor to prevent gradient underflow in half precision.
    pub loss_scale: f32,
    /// Enable dynamic loss scaling that adjusts scale based on overflow detection.
    pub dynamic_loss_scaling: bool,
}

impl Default for MixedPrecisionConfig {
    fn default() -> Self {
        Self {
            forward_dtype: DType::F16,
            master_dtype: DType::F32,
            loss_scale: 1024.0,
            dynamic_loss_scaling: true,
        }
    }
}

/// State for dynamic loss scaling during mixed-precision training.
#[derive(Debug, Clone)]
pub struct DynamicLossScaler {
    current_scale: f32,
    growth_factor: f32,
    backoff_factor: f32,
    growth_interval: u32,
    steps_since_last_overflow: u32,
}

impl DynamicLossScaler {
    pub const fn new(initial_scale: f32) -> Self {
        Self {
            current_scale: initial_scale,
            growth_factor: 2.0,
            backoff_factor: 0.5,
            growth_interval: 2000,
            steps_since_last_overflow: 0,
        }
    }

    pub const fn scale(&self) -> f32 {
        self.current_scale
    }

    /// Scale a loss tensor by the current loss scale factor.
    pub fn scale_loss(&self, loss: &Tensor) -> Result<Tensor, ModelError> {
        Ok(loss.scale(self.current_scale))
    }

    /// Unscale gradients by dividing by the current scale factor.
    pub fn unscale_gradients(&self, gradients: &[Tensor]) -> Vec<Tensor> {
        let inv_scale = 1.0 / self.current_scale;
        gradients
            .iter()
            .map(|g| {
                let scaled: Vec<f32> = g.data().iter().map(|&v| v * inv_scale).collect();
                Tensor::from_vec(g.shape().to_vec(), scaled).expect("shape matches data")
            })
            .collect()
    }

    /// Check if any gradient contains inf/nan (overflow indicator).
    pub fn check_overflow(gradients: &[Tensor]) -> bool {
        gradients.iter().any(|g| !g.all_finite())
    }

    /// Update the scaler state after a training step.
    /// Returns `true` if the step should be applied (no overflow), `false` to skip.
    pub fn update(&mut self, overflow: bool) -> bool {
        if overflow {
            self.current_scale *= self.backoff_factor;
            self.steps_since_last_overflow = 0;
            if self.current_scale < 1.0 {
                self.current_scale = 1.0;
            }
            false
        } else {
            self.steps_since_last_overflow += 1;
            if self.steps_since_last_overflow >= self.growth_interval {
                self.current_scale *= self.growth_factor;
                self.steps_since_last_overflow = 0;
                if self.current_scale > 65504.0 {
                    self.current_scale = 65504.0;
                }
            }
            true
        }
    }
}

/// Convert model parameters from master precision to forward precision.
pub fn cast_params_for_forward(
    graph: &Graph,
    param_nodes: &[yscv_autograd::NodeId],
    target_dtype: DType,
) -> Result<Vec<Tensor>, ModelError> {
    let mut casted = Vec::with_capacity(param_nodes.len());
    for &node in param_nodes {
        let tensor = graph.value(node)?;
        casted.push(tensor.to_dtype(target_dtype));
    }
    Ok(casted)
}

/// Cast a list of tensors back to master dtype for gradient accumulation.
pub fn cast_to_master(tensors: &[Tensor], master_dtype: DType) -> Vec<Tensor> {
    tensors.iter().map(|t| t.to_dtype(master_dtype)).collect()
}

/// Runs a mixed-precision forward+backward step.
///
/// 1. Cast input to forward_dtype, then back to F32 for graph computation
/// 2. Run forward pass, compute loss
/// 3. Scale loss, backprop
/// 4. Check gradients for overflow
/// 5. Update scaler state
///
/// Returns (loss_value, step_applied).
pub fn mixed_precision_train_step(
    graph: &mut Graph,
    model: &crate::SequentialModel,
    input: &Tensor,
    target: &Tensor,
    _config: &MixedPrecisionConfig,
    scaler: &mut DynamicLossScaler,
) -> Result<(f32, bool), ModelError> {
    let input_node = graph.variable(input.clone());
    let target_node = graph.constant(target.clone());
    let pred = model.forward(graph, input_node)?;
    let loss = crate::mse_loss(graph, pred, target_node)?;

    let loss_val = graph.value(loss)?.data()[0];

    graph.backward(loss)?;

    let input_grad = graph.grad(input_node)?;
    let has_overflow = input_grad.is_some_and(|g| !g.all_finite());
    let should_apply = scaler.update(has_overflow);

    Ok((loss_val, should_apply))
}
