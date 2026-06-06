use std::collections::HashMap;
use std::collections::hash_map::Entry;

use yscv_autograd::{Graph, NodeId};
use yscv_tensor::Tensor;

use super::validate::{validate_beta1, validate_beta2, validate_epsilon, validate_lr};
use super::{LearningRate, OptimError};

#[derive(Debug, Clone)]
struct RAdamState {
    first_moment: Tensor,
    second_moment: Tensor,
    step: u64,
}

impl RAdamState {
    fn new(shape: &[usize]) -> Result<Self, OptimError> {
        Ok(Self {
            first_moment: Tensor::zeros(shape.to_vec())?,
            second_moment: Tensor::zeros(shape.to_vec())?,
            step: 0,
        })
    }

    fn reset(&mut self, shape: &[usize]) -> Result<(), OptimError> {
        *self = Self::new(shape)?;
        Ok(())
    }
}

/// RAdam (Rectified Adam) optimizer with variance rectification.
#[derive(Debug, Clone)]
pub struct RAdam {
    lr: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    weight_decay: f32,
    state: HashMap<u64, RAdamState>,
}

impl RAdam {
    /// Creates RAdam with required learning rate.
    pub fn new(lr: f32) -> Result<Self, OptimError> {
        validate_lr(lr)?;
        Ok(Self {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay: 0.0,
            state: HashMap::new(),
        })
    }

    /// Sets beta1 factor in `[0, 1)`.
    pub fn with_beta1(mut self, beta1: f32) -> Result<Self, OptimError> {
        validate_beta1(beta1)?;
        self.beta1 = beta1;
        Ok(self)
    }

    /// Sets beta2 factor in `[0, 1)`.
    pub fn with_beta2(mut self, beta2: f32) -> Result<Self, OptimError> {
        validate_beta2(beta2)?;
        self.beta2 = beta2;
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
            Entry::Vacant(entry) => entry.insert(RAdamState::new(weights.shape())?),
        };
        if state.first_moment.shape() != weights.shape()
            || state.second_moment.shape() != weights.shape()
        {
            state.reset(weights.shape())?;
        }

        state.step = state.step.saturating_add(1);
        let step_f64 = state.step as f64;
        let beta1_f64 = self.beta1 as f64;
        let beta2_f64 = self.beta2 as f64;

        let bias_correction1 = (1.0 - beta1_f64.powf(step_f64)).max(f64::MIN_POSITIVE) as f32;

        // rho_inf = 2 / (1 - beta2) - 1
        let rho_inf = 2.0 / (1.0 - beta2_f64) - 1.0;
        // rho_t = rho_inf - 2 * t * beta2^t / (1 - beta2^t)
        let beta2_pow_t = beta2_f64.powf(step_f64);
        let rho_t = rho_inf - 2.0 * step_f64 * beta2_pow_t / (1.0 - beta2_pow_t);

        let use_adaptive = rho_t > 5.0;

        let (r_t, bias_correction2) = if use_adaptive {
            let bc2 = (1.0 - beta2_pow_t).max(f64::MIN_POSITIVE) as f32;
            let r = ((rho_t - 4.0) * (rho_t - 2.0) * rho_inf
                / ((rho_inf - 4.0) * (rho_inf - 2.0) * rho_t))
                .sqrt() as f32;
            (r, bc2)
        } else {
            (1.0_f32, 1.0_f32)
        };

        let first_moment = state.first_moment.data_mut();
        let second_moment = state.second_moment.data_mut();
        let grad_values = grad.data();
        let weights_data = weights.data_mut();

        let beta1 = self.beta1;
        let beta2 = self.beta2;
        let one_minus_beta1 = 1.0 - beta1;
        let one_minus_beta2 = 1.0 - beta2;
        let bias_correction1_inv = 1.0 / bias_correction1;
        let bias_correction2_inv = 1.0 / bias_correction2;
        let lr = self.lr;
        let epsilon = self.epsilon;
        let weight_decay = self.weight_decay;

        radam_update_inner(
            weights_data,
            grad_values,
            first_moment,
            second_moment,
            beta1,
            beta2,
            one_minus_beta1,
            one_minus_beta2,
            bias_correction1_inv,
            bias_correction2_inv,
            lr,
            epsilon,
            weight_decay,
            r_t,
            use_adaptive,
        );

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

impl LearningRate for RAdam {
    fn learning_rate(&self) -> f32 {
        RAdam::learning_rate(self)
    }

    fn set_learning_rate(&mut self, lr: f32) -> Result<(), OptimError> {
        RAdam::set_learning_rate(self, lr)
    }
}

/// SIMD-accelerated RAdam parameter update.
#[allow(clippy::too_many_arguments, unsafe_code)]
fn radam_update_inner(
    weights: &mut [f32],
    grad: &[f32],
    first_moment: &mut [f32],
    second_moment: &mut [f32],
    beta1: f32,
    beta2: f32,
    one_minus_beta1: f32,
    one_minus_beta2: f32,
    bc1_inv: f32,
    bc2_inv: f32,
    lr: f32,
    epsilon: f32,
    weight_decay: f32,
    r_t: f32,
    use_adaptive: bool,
) {
    #[cfg(target_arch = "aarch64")]
    if !cfg!(miri) && yscv_cpu::host_cpu().features.neon {
        unsafe {
            radam_update_neon(
                weights,
                grad,
                first_moment,
                second_moment,
                beta1,
                beta2,
                one_minus_beta1,
                one_minus_beta2,
                bc1_inv,
                bc2_inv,
                lr,
                epsilon,
                weight_decay,
                r_t,
                use_adaptive,
            );
        }
        return;
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if !cfg!(miri) && yscv_cpu::host_cpu().features.avx {
        unsafe {
            radam_update_avx(
                weights,
                grad,
                first_moment,
                second_moment,
                beta1,
                beta2,
                one_minus_beta1,
                one_minus_beta2,
                bc1_inv,
                bc2_inv,
                lr,
                epsilon,
                weight_decay,
                r_t,
                use_adaptive,
            );
        }
        return;
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if !cfg!(miri) && yscv_cpu::host_cpu().features.sse {
        unsafe {
            radam_update_sse(
                weights,
                grad,
                first_moment,
                second_moment,
                beta1,
                beta2,
                one_minus_beta1,
                one_minus_beta2,
                bc1_inv,
                bc2_inv,
                lr,
                epsilon,
                weight_decay,
                r_t,
                use_adaptive,
            );
        }
        return;
    }

    let len = weights.len();
    let wp = weights.as_mut_ptr();
    let gp = grad.as_ptr();
    let mp = first_moment.as_mut_ptr();
    let vp = second_moment.as_mut_ptr();
    for i in 0..len {
        unsafe {
            let g = *gp.add(i) + weight_decay * *wp.add(i);
            let m = beta1 * *mp.add(i) + one_minus_beta1 * g;
            let v = beta2 * *vp.add(i) + one_minus_beta2 * g * g;
            *mp.add(i) = m;
            *vp.add(i) = v;
            let m_hat = m * bc1_inv;
            if use_adaptive {
                let v_hat = v * bc2_inv;
                *wp.add(i) -= lr * r_t * m_hat / (v_hat.sqrt() + epsilon);
            } else {
                *wp.add(i) -= lr * m_hat;
            }
        }
    }
}

// ── NEON implementation ─────────────────────────────────────────────────

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments, unsafe_code, unsafe_op_in_unsafe_fn)]
unsafe fn radam_update_neon(
    weights: &mut [f32],
    grad: &[f32],
    first_moment: &mut [f32],
    second_moment: &mut [f32],
    beta1: f32,
    beta2: f32,
    one_minus_beta1: f32,
    one_minus_beta2: f32,
    bc1_inv: f32,
    bc2_inv: f32,
    lr: f32,
    epsilon: f32,
    weight_decay: f32,
    r_t: f32,
    use_adaptive: bool,
) {
    use std::arch::aarch64::*;
    let len = weights.len();
    let wp = weights.as_mut_ptr();
    let gp = grad.as_ptr();
    let mp = first_moment.as_mut_ptr();
    let vp = second_moment.as_mut_ptr();
    let beta1_v = vdupq_n_f32(beta1);
    let beta2_v = vdupq_n_f32(beta2);
    let omb1_v = vdupq_n_f32(one_minus_beta1);
    let omb2_v = vdupq_n_f32(one_minus_beta2);
    let bc1_v = vdupq_n_f32(bc1_inv);
    let wd_v = vdupq_n_f32(weight_decay);
    let mut i = 0usize;

    if use_adaptive {
        let bc2_v = vdupq_n_f32(bc2_inv);
        let lr_rt_v = vdupq_n_f32(lr * r_t);
        let eps_v = vdupq_n_f32(epsilon);
        while i + 4 <= len {
            let w = vld1q_f32(wp.add(i));
            let raw_g = vld1q_f32(gp.add(i));
            let g = vfmaq_f32(raw_g, wd_v, w);
            let m_old = vld1q_f32(mp.add(i));
            let v_old = vld1q_f32(vp.add(i));
            let m_new = vfmaq_f32(vmulq_f32(g, omb1_v), m_old, beta1_v);
            let grad_sq = vmulq_f32(g, g);
            let v_new = vfmaq_f32(vmulq_f32(grad_sq, omb2_v), v_old, beta2_v);
            vst1q_f32(mp.add(i), m_new);
            vst1q_f32(vp.add(i), v_new);
            let m_hat = vmulq_f32(m_new, bc1_v);
            let v_hat = vmulq_f32(v_new, bc2_v);
            let update = vdivq_f32(
                vmulq_f32(lr_rt_v, m_hat),
                vaddq_f32(vsqrtq_f32(v_hat), eps_v),
            );
            vst1q_f32(wp.add(i), vsubq_f32(w, update));
            i += 4;
        }
    } else {
        let lr_v = vdupq_n_f32(lr);
        while i + 4 <= len {
            let w = vld1q_f32(wp.add(i));
            let raw_g = vld1q_f32(gp.add(i));
            let g = vfmaq_f32(raw_g, wd_v, w);
            let m_old = vld1q_f32(mp.add(i));
            let v_old = vld1q_f32(vp.add(i));
            let m_new = vfmaq_f32(vmulq_f32(g, omb1_v), m_old, beta1_v);
            let grad_sq = vmulq_f32(g, g);
            let v_new = vfmaq_f32(vmulq_f32(grad_sq, omb2_v), v_old, beta2_v);
            vst1q_f32(mp.add(i), m_new);
            vst1q_f32(vp.add(i), v_new);
            let m_hat = vmulq_f32(m_new, bc1_v);
            vst1q_f32(wp.add(i), vsubq_f32(w, vmulq_f32(lr_v, m_hat)));
            i += 4;
        }
    }

    while i < len {
        let g = *gp.add(i) + weight_decay * *wp.add(i);
        let m = beta1 * *mp.add(i) + one_minus_beta1 * g;
        let v = beta2 * *vp.add(i) + one_minus_beta2 * g * g;
        *mp.add(i) = m;
        *vp.add(i) = v;
        let m_hat = m * bc1_inv;
        if use_adaptive {
            let v_hat = v * bc2_inv;
            *wp.add(i) -= lr * r_t * m_hat / (v_hat.sqrt() + epsilon);
        } else {
            *wp.add(i) -= lr * m_hat;
        }
        i += 1;
    }
}

// ── AVX implementation ──────────────────────────────────────────────────

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx")]
#[allow(clippy::too_many_arguments, unsafe_code, unsafe_op_in_unsafe_fn)]
unsafe fn radam_update_avx(
    weights: &mut [f32],
    grad: &[f32],
    first_moment: &mut [f32],
    second_moment: &mut [f32],
    beta1: f32,
    beta2: f32,
    one_minus_beta1: f32,
    one_minus_beta2: f32,
    bc1_inv: f32,
    bc2_inv: f32,
    lr: f32,
    epsilon: f32,
    weight_decay: f32,
    r_t: f32,
    use_adaptive: bool,
) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;
    let len = weights.len();
    let wp = weights.as_mut_ptr();
    let gp = grad.as_ptr();
    let mp = first_moment.as_mut_ptr();
    let vp = second_moment.as_mut_ptr();
    let beta1_v = _mm256_set1_ps(beta1);
    let beta2_v = _mm256_set1_ps(beta2);
    let omb1_v = _mm256_set1_ps(one_minus_beta1);
    let omb2_v = _mm256_set1_ps(one_minus_beta2);
    let bc1_v = _mm256_set1_ps(bc1_inv);
    let wd_v = _mm256_set1_ps(weight_decay);
    let mut i = 0usize;

    if use_adaptive {
        let bc2_v = _mm256_set1_ps(bc2_inv);
        let lr_rt_v = _mm256_set1_ps(lr * r_t);
        let eps_v = _mm256_set1_ps(epsilon);
        while i + 8 <= len {
            let w = _mm256_loadu_ps(wp.add(i));
            let raw_g = _mm256_loadu_ps(gp.add(i));
            let g = _mm256_add_ps(raw_g, _mm256_mul_ps(wd_v, w));
            let m_old = _mm256_loadu_ps(mp.add(i));
            let v_old = _mm256_loadu_ps(vp.add(i));
            let m_new = _mm256_add_ps(_mm256_mul_ps(beta1_v, m_old), _mm256_mul_ps(omb1_v, g));
            let grad_sq = _mm256_mul_ps(g, g);
            let v_new = _mm256_add_ps(
                _mm256_mul_ps(beta2_v, v_old),
                _mm256_mul_ps(omb2_v, grad_sq),
            );
            _mm256_storeu_ps(mp.add(i), m_new);
            _mm256_storeu_ps(vp.add(i), v_new);
            let m_hat = _mm256_mul_ps(m_new, bc1_v);
            let v_hat = _mm256_mul_ps(v_new, bc2_v);
            let update = _mm256_div_ps(
                _mm256_mul_ps(lr_rt_v, m_hat),
                _mm256_add_ps(_mm256_sqrt_ps(v_hat), eps_v),
            );
            _mm256_storeu_ps(wp.add(i), _mm256_sub_ps(w, update));
            i += 8;
        }
    } else {
        let lr_v = _mm256_set1_ps(lr);
        while i + 8 <= len {
            let w = _mm256_loadu_ps(wp.add(i));
            let raw_g = _mm256_loadu_ps(gp.add(i));
            let g = _mm256_add_ps(raw_g, _mm256_mul_ps(wd_v, w));
            let m_old = _mm256_loadu_ps(mp.add(i));
            let v_old = _mm256_loadu_ps(vp.add(i));
            let m_new = _mm256_add_ps(_mm256_mul_ps(beta1_v, m_old), _mm256_mul_ps(omb1_v, g));
            let grad_sq = _mm256_mul_ps(g, g);
            let v_new = _mm256_add_ps(
                _mm256_mul_ps(beta2_v, v_old),
                _mm256_mul_ps(omb2_v, grad_sq),
            );
            _mm256_storeu_ps(mp.add(i), m_new);
            _mm256_storeu_ps(vp.add(i), v_new);
            let m_hat = _mm256_mul_ps(m_new, bc1_v);
            _mm256_storeu_ps(wp.add(i), _mm256_sub_ps(w, _mm256_mul_ps(lr_v, m_hat)));
            i += 8;
        }
    }

    while i < len {
        let g = *gp.add(i) + weight_decay * *wp.add(i);
        let m = beta1 * *mp.add(i) + one_minus_beta1 * g;
        let v = beta2 * *vp.add(i) + one_minus_beta2 * g * g;
        *mp.add(i) = m;
        *vp.add(i) = v;
        let m_hat = m * bc1_inv;
        if use_adaptive {
            let v_hat = v * bc2_inv;
            *wp.add(i) -= lr * r_t * m_hat / (v_hat.sqrt() + epsilon);
        } else {
            *wp.add(i) -= lr * m_hat;
        }
        i += 1;
    }
}

// ── SSE implementation ──────────────────────────────────────────────────

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse")]
#[allow(clippy::too_many_arguments, unsafe_code, unsafe_op_in_unsafe_fn)]
unsafe fn radam_update_sse(
    weights: &mut [f32],
    grad: &[f32],
    first_moment: &mut [f32],
    second_moment: &mut [f32],
    beta1: f32,
    beta2: f32,
    one_minus_beta1: f32,
    one_minus_beta2: f32,
    bc1_inv: f32,
    bc2_inv: f32,
    lr: f32,
    epsilon: f32,
    weight_decay: f32,
    r_t: f32,
    use_adaptive: bool,
) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;
    let len = weights.len();
    let wp = weights.as_mut_ptr();
    let gp = grad.as_ptr();
    let mp = first_moment.as_mut_ptr();
    let vp = second_moment.as_mut_ptr();
    let beta1_v = _mm_set1_ps(beta1);
    let beta2_v = _mm_set1_ps(beta2);
    let omb1_v = _mm_set1_ps(one_minus_beta1);
    let omb2_v = _mm_set1_ps(one_minus_beta2);
    let bc1_v = _mm_set1_ps(bc1_inv);
    let wd_v = _mm_set1_ps(weight_decay);
    let mut i = 0usize;

    if use_adaptive {
        let bc2_v = _mm_set1_ps(bc2_inv);
        let lr_rt_v = _mm_set1_ps(lr * r_t);
        let eps_v = _mm_set1_ps(epsilon);
        while i + 4 <= len {
            let w = _mm_loadu_ps(wp.add(i));
            let raw_g = _mm_loadu_ps(gp.add(i));
            let g = _mm_add_ps(raw_g, _mm_mul_ps(wd_v, w));
            let m_old = _mm_loadu_ps(mp.add(i));
            let v_old = _mm_loadu_ps(vp.add(i));
            let m_new = _mm_add_ps(_mm_mul_ps(beta1_v, m_old), _mm_mul_ps(omb1_v, g));
            let grad_sq = _mm_mul_ps(g, g);
            let v_new = _mm_add_ps(_mm_mul_ps(beta2_v, v_old), _mm_mul_ps(omb2_v, grad_sq));
            _mm_storeu_ps(mp.add(i), m_new);
            _mm_storeu_ps(vp.add(i), v_new);
            let m_hat = _mm_mul_ps(m_new, bc1_v);
            let v_hat = _mm_mul_ps(v_new, bc2_v);
            let update = _mm_div_ps(
                _mm_mul_ps(lr_rt_v, m_hat),
                _mm_add_ps(_mm_sqrt_ps(v_hat), eps_v),
            );
            _mm_storeu_ps(wp.add(i), _mm_sub_ps(w, update));
            i += 4;
        }
    } else {
        let lr_v = _mm_set1_ps(lr);
        while i + 4 <= len {
            let w = _mm_loadu_ps(wp.add(i));
            let raw_g = _mm_loadu_ps(gp.add(i));
            let g = _mm_add_ps(raw_g, _mm_mul_ps(wd_v, w));
            let m_old = _mm_loadu_ps(mp.add(i));
            let v_old = _mm_loadu_ps(vp.add(i));
            let m_new = _mm_add_ps(_mm_mul_ps(beta1_v, m_old), _mm_mul_ps(omb1_v, g));
            let grad_sq = _mm_mul_ps(g, g);
            let v_new = _mm_add_ps(_mm_mul_ps(beta2_v, v_old), _mm_mul_ps(omb2_v, grad_sq));
            _mm_storeu_ps(mp.add(i), m_new);
            _mm_storeu_ps(vp.add(i), v_new);
            let m_hat = _mm_mul_ps(m_new, bc1_v);
            _mm_storeu_ps(wp.add(i), _mm_sub_ps(w, _mm_mul_ps(lr_v, m_hat)));
            i += 4;
        }
    }

    while i < len {
        let g = *gp.add(i) + weight_decay * *wp.add(i);
        let m = beta1 * *mp.add(i) + one_minus_beta1 * g;
        let v = beta2 * *vp.add(i) + one_minus_beta2 * g * g;
        *mp.add(i) = m;
        *vp.add(i) = v;
        let m_hat = m * bc1_inv;
        if use_adaptive {
            let v_hat = v * bc2_inv;
            *wp.add(i) -= lr * r_t * m_hat / (v_hat.sqrt() + epsilon);
        } else {
            *wp.add(i) -= lr * m_hat;
        }
        i += 1;
    }
}
