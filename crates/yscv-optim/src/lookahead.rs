use std::collections::HashMap;

use yscv_tensor::Tensor;

use super::error::OptimError;
use super::{Adagrad, Adam, AdamW, Lamb, Lars, RAdam, RmsProp, Sgd};

mod sealed {
    pub trait Sealed {}
}

/// Trait for optimizers that support a per-parameter `step` update.
///
/// This trait is sealed and cannot be implemented outside this crate.
pub trait StepOptimizer: sealed::Sealed {
    fn step(
        &mut self,
        parameter_id: u64,
        weights: &mut Tensor,
        grad: &Tensor,
    ) -> Result<(), OptimError>;
}

macro_rules! impl_step_optimizer {
    ($($ty:ty),*) => {
        $(
            impl sealed::Sealed for $ty {}
            impl StepOptimizer for $ty {
                fn step(
                    &mut self,
                    parameter_id: u64,
                    weights: &mut Tensor,
                    grad: &Tensor,
                ) -> Result<(), OptimError> {
                    <$ty>::step(self, parameter_id, weights, grad)
                }
            }
        )*
    };
}

impl_step_optimizer!(Sgd, Adam, AdamW, RmsProp, Adagrad, RAdam, Lamb, Lars);

/// Lookahead optimizer wrapper.
///
/// Maintains "slow weights" that are periodically interpolated toward the fast
/// weights produced by the inner optimizer.  Every `k` calls to `step`, the
/// slow weights are updated via:
///
/// ```text
/// slow_w = slow_w + alpha * (fast_w - slow_w)
/// fast_w = slow_w
/// ```
#[derive(Debug, Clone)]
pub struct Lookahead<O> {
    inner: O,
    alpha: f32,
    k: usize,
    step_count: usize,
    slow_weights: HashMap<u64, Vec<f32>>,
}

impl<O: StepOptimizer> Lookahead<O> {
    /// Creates a new `Lookahead` wrapper around the given optimizer.
    ///
    /// * `alpha` — interpolation coefficient (typically 0.5).
    /// * `k` — synchronisation period (typically 5).
    pub fn new(inner: O, alpha: f32, k: usize) -> Self {
        Self {
            inner,
            alpha,
            k,
            step_count: 0,
            slow_weights: HashMap::new(),
        }
    }

    /// Performs one optimisation step.
    ///
    /// 1. Delegates to the inner optimizer to update the fast weights.
    /// 2. Increments the internal step counter.
    /// 3. Every `k` steps, synchronises slow and fast weights.
    pub fn step(
        &mut self,
        parameter_id: u64,
        weights: &mut Tensor,
        grad: &Tensor,
    ) -> Result<(), OptimError> {
        // Inner (fast) update.
        self.inner.step(parameter_id, weights, grad)?;

        // Initialise slow weights on first encounter.
        self.slow_weights
            .entry(parameter_id)
            .or_insert_with(|| weights.data().to_vec());

        self.step_count += 1;

        // Synchronise every k steps.
        if self.step_count.is_multiple_of(self.k) {
            let slow = self
                .slow_weights
                .get_mut(&parameter_id)
                .expect("slow weights must exist");
            let fast = weights.data_mut();
            for (s, f) in slow.iter_mut().zip(fast.iter_mut()) {
                *s += self.alpha * (*f - *s);
                *f = *s;
            }
        }

        Ok(())
    }
}
