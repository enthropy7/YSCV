use std::collections::HashMap;
use std::collections::hash_map::Entry;

use yscv_autograd::{Graph, NodeId};
use yscv_tensor::Tensor;

use super::validate::{validate_beta1, validate_beta2, validate_epsilon, validate_lr};
use super::{LearningRate, OptimError};

#[derive(Debug, Clone)]
struct LambState {
    first_moment: Tensor,
    second_moment: Tensor,
    step: u64,
}

impl LambState {
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

/// Layer-wise Adaptive Moments optimizer for Batch training (LAMB).
///
/// Combines Adam-style adaptive moment estimation with layer-wise trust ratio
/// scaling for stable large-batch training.
#[derive(Debug, Clone)]
pub struct Lamb {
    lr: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    weight_decay: f32,
    state: HashMap<u64, LambState>,
}

impl Lamb {
    /// Creates LAMB with required learning rate.
    pub fn new(lr: f32) -> Result<Self, OptimError> {
        validate_lr(lr)?;
        Ok(Self {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-6,
            weight_decay: 0.0,
            state: HashMap::new(),
        })
    }

    /// Sets beta coefficients `(beta1, beta2)` used for computing running averages.
    pub fn with_betas(mut self, beta1: f32, beta2: f32) -> Result<Self, OptimError> {
        validate_beta1(beta1)?;
        validate_beta2(beta2)?;
        self.beta1 = beta1;
        self.beta2 = beta2;
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

    /// Sets epsilon value, must be finite and `> 0`.
    pub fn with_epsilon(mut self, epsilon: f32) -> Result<Self, OptimError> {
        validate_epsilon(epsilon)?;
        self.epsilon = epsilon;
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
            Entry::Vacant(entry) => entry.insert(LambState::new(weights.shape())?),
        };
        if state.first_moment.shape() != weights.shape()
            || state.second_moment.shape() != weights.shape()
        {
            state.reset(weights.shape())?;
        }

        state.step = state.step.saturating_add(1);
        let step_f64 = state.step as f64;
        let bias_correction1 =
            (1.0 - (self.beta1 as f64).powf(step_f64)).max(f64::MIN_POSITIVE) as f32;
        let bias_correction2 =
            (1.0 - (self.beta2 as f64).powf(step_f64)).max(f64::MIN_POSITIVE) as f32;

        let first_moment = state.first_moment.data_mut();
        let second_moment = state.second_moment.data_mut();
        let grad_data = grad.data();
        let weights_data = weights.data_mut();

        let beta1 = self.beta1;
        let beta2 = self.beta2;
        let one_minus_beta1 = 1.0 - beta1;
        let one_minus_beta2 = 1.0 - beta2;
        let bias_correction1_inv = 1.0 / bias_correction1;
        let bias_correction2_inv = 1.0 / bias_correction2;
        let epsilon = self.epsilon;
        let weight_decay = self.weight_decay;

        let (w_norm_sq, step_norm_sq) = lamb_pass1_inner(
            weights_data,
            grad_data,
            first_moment,
            second_moment,
            beta1,
            beta2,
            one_minus_beta1,
            one_minus_beta2,
            bias_correction1_inv,
            bias_correction2_inv,
            epsilon,
            weight_decay,
        );

        let w_norm = w_norm_sq.sqrt();
        let step_norm = step_norm_sq.sqrt();
        let trust_ratio = if w_norm > 0.0 && step_norm > 0.0 {
            w_norm / step_norm
        } else {
            1.0
        };
        let scaled_lr = self.lr * trust_ratio;

        lamb_pass2_inner(
            weights_data,
            first_moment,
            second_moment,
            bias_correction1_inv,
            bias_correction2_inv,
            scaled_lr,
            epsilon,
            weight_decay,
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

impl LearningRate for Lamb {
    fn learning_rate(&self) -> f32 {
        Lamb::learning_rate(self)
    }

    fn set_learning_rate(&mut self, lr: f32) -> Result<(), OptimError> {
        Lamb::set_learning_rate(self, lr)
    }
}

// ── SIMD-accelerated LAMB pass 1: update moments + compute norms ────────

/// Pass 1: update moments, return `(w_norm_sq, step_norm_sq)`.
#[allow(clippy::too_many_arguments, unsafe_code)]
fn lamb_pass1_inner(
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
    epsilon: f32,
    weight_decay: f32,
) -> (f32, f32) {
    #[cfg(target_arch = "aarch64")]
    if !cfg!(miri) && yscv_cpu::host_cpu().features.neon {
        return unsafe {
            lamb_pass1_neon(
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
                epsilon,
                weight_decay,
            )
        };
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if !cfg!(miri) && yscv_cpu::host_cpu().features.avx {
        return unsafe {
            lamb_pass1_avx(
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
                epsilon,
                weight_decay,
            )
        };
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if !cfg!(miri) && yscv_cpu::host_cpu().features.sse {
        return unsafe {
            lamb_pass1_sse(
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
                epsilon,
                weight_decay,
            )
        };
    }

    let len = weights.len();
    let wp = weights.as_mut_ptr();
    let gp = grad.as_ptr();
    let mp = first_moment.as_mut_ptr();
    let vp = second_moment.as_mut_ptr();
    let mut w_norm_sq: f32 = 0.0;
    let mut step_norm_sq: f32 = 0.0;
    for i in 0..len {
        unsafe {
            let w = *wp.add(i);
            let g = *gp.add(i);
            let m = beta1 * *mp.add(i) + one_minus_beta1 * g;
            let v = beta2 * *vp.add(i) + one_minus_beta2 * g * g;
            *mp.add(i) = m;
            *vp.add(i) = v;
            let m_hat = m * bc1_inv;
            let v_hat = v * bc2_inv;
            let s = m_hat / (v_hat.sqrt() + epsilon) + weight_decay * w;
            w_norm_sq += w * w;
            step_norm_sq += s * s;
        }
    }
    (w_norm_sq, step_norm_sq)
}

/// Pass 2: apply trust-ratio-scaled update from already-updated moments.
#[allow(clippy::too_many_arguments, unsafe_code)]
fn lamb_pass2_inner(
    weights: &mut [f32],
    first_moment: &[f32],
    second_moment: &[f32],
    bc1_inv: f32,
    bc2_inv: f32,
    scaled_lr: f32,
    epsilon: f32,
    weight_decay: f32,
) {
    #[cfg(target_arch = "aarch64")]
    if !cfg!(miri) && yscv_cpu::host_cpu().features.neon {
        unsafe {
            lamb_pass2_neon(
                weights,
                first_moment,
                second_moment,
                bc1_inv,
                bc2_inv,
                scaled_lr,
                epsilon,
                weight_decay,
            );
        }
        return;
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if !cfg!(miri) && yscv_cpu::host_cpu().features.avx {
        unsafe {
            lamb_pass2_avx(
                weights,
                first_moment,
                second_moment,
                bc1_inv,
                bc2_inv,
                scaled_lr,
                epsilon,
                weight_decay,
            );
        }
        return;
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if !cfg!(miri) && yscv_cpu::host_cpu().features.sse {
        unsafe {
            lamb_pass2_sse(
                weights,
                first_moment,
                second_moment,
                bc1_inv,
                bc2_inv,
                scaled_lr,
                epsilon,
                weight_decay,
            );
        }
        return;
    }

    let len = weights.len();
    let wp = weights.as_mut_ptr();
    let mp = first_moment.as_ptr();
    let vp = second_moment.as_ptr();
    for i in 0..len {
        unsafe {
            let m_hat = *mp.add(i) * bc1_inv;
            let v_hat = *vp.add(i) * bc2_inv;
            let s = m_hat / (v_hat.sqrt() + epsilon) + weight_decay * *wp.add(i);
            *wp.add(i) -= scaled_lr * s;
        }
    }
}

// ── NEON implementations ────────────────────────────────────────────────

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments, unsafe_code, unsafe_op_in_unsafe_fn)]
unsafe fn lamb_pass1_neon(
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
    epsilon: f32,
    weight_decay: f32,
) -> (f32, f32) {
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
    let bc2_v = vdupq_n_f32(bc2_inv);
    let eps_v = vdupq_n_f32(epsilon);
    let wd_v = vdupq_n_f32(weight_decay);
    let mut w_norm_acc = vdupq_n_f32(0.0);
    let mut s_norm_acc = vdupq_n_f32(0.0);
    let mut i = 0usize;
    while i + 4 <= len {
        let w = vld1q_f32(wp.add(i));
        let g = vld1q_f32(gp.add(i));
        let m_old = vld1q_f32(mp.add(i));
        let v_old = vld1q_f32(vp.add(i));
        let m_new = vfmaq_f32(vmulq_f32(g, omb1_v), m_old, beta1_v);
        let grad_sq = vmulq_f32(g, g);
        let v_new = vfmaq_f32(vmulq_f32(grad_sq, omb2_v), v_old, beta2_v);
        vst1q_f32(mp.add(i), m_new);
        vst1q_f32(vp.add(i), v_new);
        let m_hat = vmulq_f32(m_new, bc1_v);
        let v_hat = vmulq_f32(v_new, bc2_v);
        let s = vfmaq_f32(
            vdivq_f32(m_hat, vaddq_f32(vsqrtq_f32(v_hat), eps_v)),
            wd_v,
            w,
        );
        w_norm_acc = vfmaq_f32(w_norm_acc, w, w);
        s_norm_acc = vfmaq_f32(s_norm_acc, s, s);
        i += 4;
    }
    let mut w_norm_sq = vaddvq_f32(w_norm_acc);
    let mut step_norm_sq = vaddvq_f32(s_norm_acc);
    while i < len {
        let w = *wp.add(i);
        let g = *gp.add(i);
        let m = beta1 * *mp.add(i) + one_minus_beta1 * g;
        let v = beta2 * *vp.add(i) + one_minus_beta2 * g * g;
        *mp.add(i) = m;
        *vp.add(i) = v;
        let m_hat = m * bc1_inv;
        let v_hat = v * bc2_inv;
        let s = m_hat / (v_hat.sqrt() + epsilon) + weight_decay * w;
        w_norm_sq += w * w;
        step_norm_sq += s * s;
        i += 1;
    }
    (w_norm_sq, step_norm_sq)
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments, unsafe_code, unsafe_op_in_unsafe_fn)]
unsafe fn lamb_pass2_neon(
    weights: &mut [f32],
    first_moment: &[f32],
    second_moment: &[f32],
    bc1_inv: f32,
    bc2_inv: f32,
    scaled_lr: f32,
    epsilon: f32,
    weight_decay: f32,
) {
    use std::arch::aarch64::*;
    let len = weights.len();
    let wp = weights.as_mut_ptr();
    let mp = first_moment.as_ptr();
    let vp = second_moment.as_ptr();
    let bc1_v = vdupq_n_f32(bc1_inv);
    let bc2_v = vdupq_n_f32(bc2_inv);
    let lr_v = vdupq_n_f32(scaled_lr);
    let eps_v = vdupq_n_f32(epsilon);
    let wd_v = vdupq_n_f32(weight_decay);
    let mut i = 0usize;
    while i + 4 <= len {
        let w = vld1q_f32(wp.add(i));
        let m_hat = vmulq_f32(vld1q_f32(mp.add(i)), bc1_v);
        let v_hat = vmulq_f32(vld1q_f32(vp.add(i)), bc2_v);
        let s = vfmaq_f32(
            vdivq_f32(m_hat, vaddq_f32(vsqrtq_f32(v_hat), eps_v)),
            wd_v,
            w,
        );
        vst1q_f32(wp.add(i), vsubq_f32(w, vmulq_f32(lr_v, s)));
        i += 4;
    }
    while i < len {
        let m_hat = *mp.add(i) * bc1_inv;
        let v_hat = *vp.add(i) * bc2_inv;
        let s = m_hat / (v_hat.sqrt() + epsilon) + weight_decay * *wp.add(i);
        *wp.add(i) -= scaled_lr * s;
        i += 1;
    }
}

// ── AVX implementations ─────────────────────────────────────────────────

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx")]
#[allow(clippy::too_many_arguments, unsafe_code, unsafe_op_in_unsafe_fn)]
unsafe fn lamb_pass1_avx(
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
    epsilon: f32,
    weight_decay: f32,
) -> (f32, f32) {
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
    let bc2_v = _mm256_set1_ps(bc2_inv);
    let eps_v = _mm256_set1_ps(epsilon);
    let wd_v = _mm256_set1_ps(weight_decay);
    let mut w_norm_acc = _mm256_setzero_ps();
    let mut s_norm_acc = _mm256_setzero_ps();
    let mut i = 0usize;
    while i + 8 <= len {
        let w = _mm256_loadu_ps(wp.add(i));
        let g = _mm256_loadu_ps(gp.add(i));
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
        let s = _mm256_add_ps(
            _mm256_div_ps(m_hat, _mm256_add_ps(_mm256_sqrt_ps(v_hat), eps_v)),
            _mm256_mul_ps(wd_v, w),
        );
        w_norm_acc = _mm256_add_ps(w_norm_acc, _mm256_mul_ps(w, w));
        s_norm_acc = _mm256_add_ps(s_norm_acc, _mm256_mul_ps(s, s));
        i += 8;
    }
    // Horizontal sum of 8-wide accumulators
    let w_lo = _mm256_castps256_ps128(w_norm_acc);
    let w_hi = _mm256_extractf128_ps(w_norm_acc, 1);
    let w_sum4 = _mm_add_ps(w_lo, w_hi);
    let w_shuf = _mm_movehdup_ps(w_sum4);
    let w_sum2 = _mm_add_ps(w_sum4, w_shuf);
    let w_shuf2 = _mm_movehl_ps(w_sum2, w_sum2);
    let mut w_norm_sq = _mm_cvtss_f32(_mm_add_ss(w_sum2, w_shuf2));

    let s_lo = _mm256_castps256_ps128(s_norm_acc);
    let s_hi = _mm256_extractf128_ps(s_norm_acc, 1);
    let s_sum4 = _mm_add_ps(s_lo, s_hi);
    let s_shuf = _mm_movehdup_ps(s_sum4);
    let s_sum2 = _mm_add_ps(s_sum4, s_shuf);
    let s_shuf2 = _mm_movehl_ps(s_sum2, s_sum2);
    let mut step_norm_sq = _mm_cvtss_f32(_mm_add_ss(s_sum2, s_shuf2));

    while i < len {
        let w = *wp.add(i);
        let g = *gp.add(i);
        let m = beta1 * *mp.add(i) + one_minus_beta1 * g;
        let v = beta2 * *vp.add(i) + one_minus_beta2 * g * g;
        *mp.add(i) = m;
        *vp.add(i) = v;
        let m_hat = m * bc1_inv;
        let v_hat = v * bc2_inv;
        let s = m_hat / (v_hat.sqrt() + epsilon) + weight_decay * w;
        w_norm_sq += w * w;
        step_norm_sq += s * s;
        i += 1;
    }
    (w_norm_sq, step_norm_sq)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx")]
#[allow(clippy::too_many_arguments, unsafe_code, unsafe_op_in_unsafe_fn)]
unsafe fn lamb_pass2_avx(
    weights: &mut [f32],
    first_moment: &[f32],
    second_moment: &[f32],
    bc1_inv: f32,
    bc2_inv: f32,
    scaled_lr: f32,
    epsilon: f32,
    weight_decay: f32,
) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;
    let len = weights.len();
    let wp = weights.as_mut_ptr();
    let mp = first_moment.as_ptr();
    let vp = second_moment.as_ptr();
    let bc1_v = _mm256_set1_ps(bc1_inv);
    let bc2_v = _mm256_set1_ps(bc2_inv);
    let lr_v = _mm256_set1_ps(scaled_lr);
    let eps_v = _mm256_set1_ps(epsilon);
    let wd_v = _mm256_set1_ps(weight_decay);
    let mut i = 0usize;
    while i + 8 <= len {
        let w = _mm256_loadu_ps(wp.add(i));
        let m_hat = _mm256_mul_ps(_mm256_loadu_ps(mp.add(i)), bc1_v);
        let v_hat = _mm256_mul_ps(_mm256_loadu_ps(vp.add(i)), bc2_v);
        let s = _mm256_add_ps(
            _mm256_div_ps(m_hat, _mm256_add_ps(_mm256_sqrt_ps(v_hat), eps_v)),
            _mm256_mul_ps(wd_v, w),
        );
        _mm256_storeu_ps(wp.add(i), _mm256_sub_ps(w, _mm256_mul_ps(lr_v, s)));
        i += 8;
    }
    while i < len {
        let m_hat = *mp.add(i) * bc1_inv;
        let v_hat = *vp.add(i) * bc2_inv;
        let s = m_hat / (v_hat.sqrt() + epsilon) + weight_decay * *wp.add(i);
        *wp.add(i) -= scaled_lr * s;
        i += 1;
    }
}

// ── SSE implementations ─────────────────────────────────────────────────

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse")]
#[allow(clippy::too_many_arguments, unsafe_code, unsafe_op_in_unsafe_fn)]
unsafe fn lamb_pass1_sse(
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
    epsilon: f32,
    weight_decay: f32,
) -> (f32, f32) {
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
    let bc2_v = _mm_set1_ps(bc2_inv);
    let eps_v = _mm_set1_ps(epsilon);
    let wd_v = _mm_set1_ps(weight_decay);
    let mut w_norm_acc = _mm_setzero_ps();
    let mut s_norm_acc = _mm_setzero_ps();
    let mut i = 0usize;
    while i + 4 <= len {
        let w = _mm_loadu_ps(wp.add(i));
        let g = _mm_loadu_ps(gp.add(i));
        let m_old = _mm_loadu_ps(mp.add(i));
        let v_old = _mm_loadu_ps(vp.add(i));
        let m_new = _mm_add_ps(_mm_mul_ps(beta1_v, m_old), _mm_mul_ps(omb1_v, g));
        let grad_sq = _mm_mul_ps(g, g);
        let v_new = _mm_add_ps(_mm_mul_ps(beta2_v, v_old), _mm_mul_ps(omb2_v, grad_sq));
        _mm_storeu_ps(mp.add(i), m_new);
        _mm_storeu_ps(vp.add(i), v_new);
        let m_hat = _mm_mul_ps(m_new, bc1_v);
        let v_hat = _mm_mul_ps(v_new, bc2_v);
        let s = _mm_add_ps(
            _mm_div_ps(m_hat, _mm_add_ps(_mm_sqrt_ps(v_hat), eps_v)),
            _mm_mul_ps(wd_v, w),
        );
        w_norm_acc = _mm_add_ps(w_norm_acc, _mm_mul_ps(w, w));
        s_norm_acc = _mm_add_ps(s_norm_acc, _mm_mul_ps(s, s));
        i += 4;
    }
    // Horizontal sum of 4-wide accumulators
    let w_shuf = _mm_movehdup_ps(w_norm_acc);
    let w_sum2 = _mm_add_ps(w_norm_acc, w_shuf);
    let w_shuf2 = _mm_movehl_ps(w_sum2, w_sum2);
    let mut w_norm_sq = _mm_cvtss_f32(_mm_add_ss(w_sum2, w_shuf2));

    let s_shuf = _mm_movehdup_ps(s_norm_acc);
    let s_sum2 = _mm_add_ps(s_norm_acc, s_shuf);
    let s_shuf2 = _mm_movehl_ps(s_sum2, s_sum2);
    let mut step_norm_sq = _mm_cvtss_f32(_mm_add_ss(s_sum2, s_shuf2));

    while i < len {
        let w = *wp.add(i);
        let g = *gp.add(i);
        let m = beta1 * *mp.add(i) + one_minus_beta1 * g;
        let v = beta2 * *vp.add(i) + one_minus_beta2 * g * g;
        *mp.add(i) = m;
        *vp.add(i) = v;
        let m_hat = m * bc1_inv;
        let v_hat = v * bc2_inv;
        let s = m_hat / (v_hat.sqrt() + epsilon) + weight_decay * w;
        w_norm_sq += w * w;
        step_norm_sq += s * s;
        i += 1;
    }
    (w_norm_sq, step_norm_sq)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse")]
#[allow(clippy::too_many_arguments, unsafe_code, unsafe_op_in_unsafe_fn)]
unsafe fn lamb_pass2_sse(
    weights: &mut [f32],
    first_moment: &[f32],
    second_moment: &[f32],
    bc1_inv: f32,
    bc2_inv: f32,
    scaled_lr: f32,
    epsilon: f32,
    weight_decay: f32,
) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;
    let len = weights.len();
    let wp = weights.as_mut_ptr();
    let mp = first_moment.as_ptr();
    let vp = second_moment.as_ptr();
    let bc1_v = _mm_set1_ps(bc1_inv);
    let bc2_v = _mm_set1_ps(bc2_inv);
    let lr_v = _mm_set1_ps(scaled_lr);
    let eps_v = _mm_set1_ps(epsilon);
    let wd_v = _mm_set1_ps(weight_decay);
    let mut i = 0usize;
    while i + 4 <= len {
        let w = _mm_loadu_ps(wp.add(i));
        let m_hat = _mm_mul_ps(_mm_loadu_ps(mp.add(i)), bc1_v);
        let v_hat = _mm_mul_ps(_mm_loadu_ps(vp.add(i)), bc2_v);
        let s = _mm_add_ps(
            _mm_div_ps(m_hat, _mm_add_ps(_mm_sqrt_ps(v_hat), eps_v)),
            _mm_mul_ps(wd_v, w),
        );
        _mm_storeu_ps(wp.add(i), _mm_sub_ps(w, _mm_mul_ps(lr_v, s)));
        i += 4;
    }
    while i < len {
        let m_hat = *mp.add(i) * bc1_inv;
        let v_hat = *vp.add(i) * bc2_inv;
        let s = m_hat / (v_hat.sqrt() + epsilon) + weight_decay * *wp.add(i);
        *wp.add(i) -= scaled_lr * s;
        i += 1;
    }
}
