use yscv_tensor::Tensor;

/// Exponential Moving Average of model parameters.
///
/// Maintains shadow copies of parameters that are updated as a weighted
/// running average: `shadow = decay * shadow + (1 - decay) * param`.
/// This is commonly used to produce a smoothed version of model weights
/// that often generalises better at inference time.
pub struct ExponentialMovingAverage {
    decay: f32,
    shadow_params: Vec<Tensor>,
    num_updates: usize,
}

impl ExponentialMovingAverage {
    /// Creates a new EMA tracker with the given decay factor (e.g. 0.999).
    pub const fn new(decay: f32) -> Self {
        Self {
            decay,
            shadow_params: Vec::new(),
            num_updates: 0,
        }
    }

    /// Registers initial parameter values as shadow copies.
    pub fn register(&mut self, params: &[Tensor]) {
        self.shadow_params = params.to_vec();
    }

    /// Updates shadow parameters: `shadow = decay * shadow + (1 - decay) * param`.
    ///
    /// Panics if the number of tensors does not match the registered count or
    /// if any tensor length differs from its shadow counterpart.
    pub fn update(&mut self, params: &[Tensor]) {
        assert_eq!(
            params.len(),
            self.shadow_params.len(),
            "param count mismatch: expected {} but got {}",
            self.shadow_params.len(),
            params.len(),
        );
        let decay = self.decay;
        let one_minus_decay = 1.0 - decay;
        for (shadow, param) in self.shadow_params.iter_mut().zip(params.iter()) {
            let s = shadow.data_mut();
            let p = param.data();
            assert_eq!(s.len(), p.len(), "tensor length mismatch in EMA update");
            let len = s.len();
            for i in 0..len {
                s[i] = decay * s[i] + one_minus_decay * p[i];
            }
        }
        self.num_updates += 1;
    }

    /// Returns a reference to the shadow parameters.
    pub fn shadow_params(&self) -> &[Tensor] {
        &self.shadow_params
    }

    /// Copies shadow parameter values into the provided mutable slice.
    ///
    /// Panics if the slice length does not match the shadow parameter count or
    /// if any tensor length differs.
    pub fn apply_shadow(&self, params: &mut [Tensor]) {
        assert_eq!(
            params.len(),
            self.shadow_params.len(),
            "param count mismatch in apply_shadow",
        );
        for (dst, src) in params.iter_mut().zip(self.shadow_params.iter()) {
            let d = dst.data_mut();
            let s = src.data();
            assert_eq!(d.len(), s.len(), "tensor length mismatch in apply_shadow");
            d.copy_from_slice(s);
        }
    }

    /// Returns the number of update steps performed so far.
    pub const fn num_updates(&self) -> usize {
        self.num_updates
    }
}
