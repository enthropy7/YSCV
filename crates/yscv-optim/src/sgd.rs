use std::collections::HashMap;
use std::collections::hash_map::Entry;

use yscv_autograd::{Graph, NodeId};
use yscv_tensor::Tensor;

use super::validate::{validate_dampening, validate_lr, validate_momentum};
use super::{LearningRate, OptimError};

/// Stochastic gradient descent optimizer with optional momentum and weight decay.
#[derive(Debug, Clone)]
pub struct Sgd {
    lr: f32,
    momentum: f32,
    dampening: f32,
    weight_decay: f32,
    nesterov: bool,
    velocity: HashMap<u64, Tensor>,
}

impl Sgd {
    /// Creates SGD with required learning rate.
    pub fn new(lr: f32) -> Result<Self, OptimError> {
        validate_lr(lr)?;
        Ok(Self {
            lr,
            momentum: 0.0,
            dampening: 0.0,
            weight_decay: 0.0,
            nesterov: false,
            velocity: HashMap::new(),
        })
    }

    /// Sets momentum factor in `[0, 1)`.
    pub fn with_momentum(mut self, momentum: f32) -> Result<Self, OptimError> {
        validate_momentum(momentum)?;
        self.momentum = momentum;
        self.validate_nesterov_constraints()?;
        Ok(self)
    }

    /// Sets dampening factor in `[0, 1]`.
    pub fn with_dampening(mut self, dampening: f32) -> Result<Self, OptimError> {
        validate_dampening(dampening)?;
        self.dampening = dampening;
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

    /// Enables/disables Nesterov update rule.
    pub fn with_nesterov(mut self, nesterov: bool) -> Result<Self, OptimError> {
        self.nesterov = nesterov;
        self.validate_nesterov_constraints()?;
        Ok(self)
    }

    /// Drops optimizer state (for example when restarting training).
    pub fn clear_state(&mut self) {
        self.velocity.clear();
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

        // Fast path: no weight decay, no momentum — just axpy in-place.
        if self.weight_decay == 0.0 && self.momentum == 0.0 {
            axpy_neg(weights.data_mut(), grad.data(), self.lr);
            return Ok(());
        }

        // When weight_decay != 0 we need adjusted gradients.
        let has_wd = self.weight_decay != 0.0;
        // Build adjusted_grad only when weight_decay is non-zero; otherwise
        // we can reference grad.data() directly.
        let adjusted_grad_buf: Vec<f32>;
        let grad_slice: &[f32] = if has_wd {
            let mut buf = grad.data().to_vec();
            let wd = self.weight_decay;
            fma_inplace(&mut buf, weights.data(), wd);
            adjusted_grad_buf = buf;
            &adjusted_grad_buf
        } else {
            grad.data()
        };

        if self.momentum != 0.0 {
            let velocity = match self.velocity.entry(parameter_id) {
                Entry::Occupied(entry) => entry.into_mut(),
                Entry::Vacant(entry) => {
                    let initial = Tensor::zeros(weights.shape().to_vec())?;
                    entry.insert(initial)
                }
            };

            if velocity.shape() != weights.shape() {
                *velocity = Tensor::zeros(weights.shape().to_vec())?;
            }

            // velocity = momentum * velocity + (1 - dampening) * grad
            // Done in-place on velocity's buffer to avoid allocation.
            let mom = self.momentum;
            let grad_scale = 1.0 - self.dampening;
            momentum_update(velocity.data_mut(), grad_slice, mom, grad_scale);

            if self.nesterov {
                // update = grad + momentum * velocity
                // weights -= lr * update
                // => weights -= lr * grad + lr * momentum * velocity
                axpy_neg(weights.data_mut(), grad_slice, self.lr);
                axpy_neg(weights.data_mut(), velocity.data(), self.lr * mom);
            } else {
                // weights -= lr * velocity
                axpy_neg(weights.data_mut(), velocity.data(), self.lr);
            }
        } else {
            axpy_neg(weights.data_mut(), grad_slice, self.lr);
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

    fn validate_nesterov_constraints(&self) -> Result<(), OptimError> {
        if self.nesterov && self.momentum == 0.0 {
            return Err(OptimError::NesterovRequiresMomentum);
        }
        Ok(())
    }
}

impl LearningRate for Sgd {
    fn learning_rate(&self) -> f32 {
        Sgd::learning_rate(self)
    }

    fn set_learning_rate(&mut self, lr: f32) -> Result<(), OptimError> {
        Sgd::set_learning_rate(self, lr)
    }
}

/// weights[i] -= lr * grads[i]  — SIMD-accelerated axpy(negative).
#[allow(unsafe_code)]
fn axpy_neg(weights: &mut [f32], grads: &[f32], lr: f32) {
    debug_assert_eq!(weights.len(), grads.len());
    let len = weights.len();

    #[cfg(target_arch = "aarch64")]
    if !cfg!(miri) && yscv_cpu::host_cpu().features.neon {
        unsafe { axpy_neg_neon(weights, grads, lr) };
        return;
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if !cfg!(miri) && yscv_cpu::host_cpu().features.avx {
        unsafe { axpy_neg_avx(weights, grads, lr) };
        return;
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if !cfg!(miri) && yscv_cpu::host_cpu().features.sse {
        unsafe { axpy_neg_sse(weights, grads, lr) };
        return;
    }

    let w_ptr = weights.as_mut_ptr();
    let g_ptr = grads.as_ptr();
    unsafe {
        let mut i = 0usize;
        while i + 4 <= len {
            *w_ptr.add(i) -= lr * *g_ptr.add(i);
            *w_ptr.add(i + 1) -= lr * *g_ptr.add(i + 1);
            *w_ptr.add(i + 2) -= lr * *g_ptr.add(i + 2);
            *w_ptr.add(i + 3) -= lr * *g_ptr.add(i + 3);
            i += 4;
        }
        while i < len {
            *w_ptr.add(i) -= lr * *g_ptr.add(i);
            i += 1;
        }
    }
}

/// dst[i] += src[i] * scale  — SIMD-accelerated fused multiply-add.
#[allow(unsafe_code)]
fn fma_inplace(dst: &mut [f32], src: &[f32], scale: f32) {
    debug_assert_eq!(dst.len(), src.len());
    let len = dst.len();

    #[cfg(target_arch = "aarch64")]
    if !cfg!(miri) && yscv_cpu::host_cpu().features.neon {
        unsafe { fma_inplace_neon(dst, src, scale) };
        return;
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if !cfg!(miri) && yscv_cpu::host_cpu().features.avx {
        unsafe { fma_inplace_avx(dst, src, scale) };
        return;
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if !cfg!(miri) && yscv_cpu::host_cpu().features.sse {
        unsafe { fma_inplace_sse(dst, src, scale) };
        return;
    }

    for i in 0..len {
        dst[i] += src[i] * scale;
    }
}

/// velocity[i] = momentum * velocity[i] + grad_scale * grad[i]  — in-place.
#[allow(unsafe_code)]
fn momentum_update(velocity: &mut [f32], grad: &[f32], momentum: f32, grad_scale: f32) {
    debug_assert_eq!(velocity.len(), grad.len());
    let len = velocity.len();

    #[cfg(target_arch = "aarch64")]
    if !cfg!(miri) && yscv_cpu::host_cpu().features.neon {
        unsafe { momentum_update_neon(velocity, grad, momentum, grad_scale) };
        return;
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if !cfg!(miri) && yscv_cpu::host_cpu().features.avx {
        unsafe { momentum_update_avx(velocity, grad, momentum, grad_scale) };
        return;
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if !cfg!(miri) && yscv_cpu::host_cpu().features.sse {
        unsafe { momentum_update_sse(velocity, grad, momentum, grad_scale) };
        return;
    }

    for i in 0..len {
        velocity[i] = momentum * velocity[i] + grad_scale * grad[i];
    }
}

// ── NEON implementations ────────────────────────────────────────────────

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
unsafe fn axpy_neg_neon(weights: &mut [f32], grads: &[f32], lr: f32) {
    use std::arch::aarch64::*;
    let len = weights.len();
    let wp = weights.as_mut_ptr();
    let gp = grads.as_ptr();
    let vlr = vdupq_n_f32(lr);
    let mut i = 0usize;
    while i + 4 <= len {
        let w = vld1q_f32(wp.add(i));
        let g = vld1q_f32(gp.add(i));
        vst1q_f32(wp.add(i), vfmsq_f32(w, g, vlr));
        i += 4;
    }
    while i < len {
        *wp.add(i) -= lr * *gp.add(i);
        i += 1;
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
unsafe fn fma_inplace_neon(dst: &mut [f32], src: &[f32], scale: f32) {
    use std::arch::aarch64::*;
    let len = dst.len();
    let dp = dst.as_mut_ptr();
    let sp = src.as_ptr();
    let vs = vdupq_n_f32(scale);
    let mut i = 0usize;
    while i + 4 <= len {
        let d = vld1q_f32(dp.add(i));
        let s = vld1q_f32(sp.add(i));
        vst1q_f32(dp.add(i), vfmaq_f32(d, s, vs));
        i += 4;
    }
    while i < len {
        *dp.add(i) += *sp.add(i) * scale;
        i += 1;
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
unsafe fn momentum_update_neon(velocity: &mut [f32], grad: &[f32], momentum: f32, grad_scale: f32) {
    use std::arch::aarch64::*;
    let len = velocity.len();
    let vp = velocity.as_mut_ptr();
    let gp = grad.as_ptr();
    let vmom = vdupq_n_f32(momentum);
    let vgs = vdupq_n_f32(grad_scale);
    let mut i = 0usize;
    while i + 4 <= len {
        let v = vld1q_f32(vp.add(i));
        let g = vld1q_f32(gp.add(i));
        // momentum * v + grad_scale * g
        let result = vfmaq_f32(vmulq_f32(vmom, v), g, vgs);
        vst1q_f32(vp.add(i), result);
        i += 4;
    }
    while i < len {
        *vp.add(i) = momentum * *vp.add(i) + grad_scale * *gp.add(i);
        i += 1;
    }
}

// ── AVX implementations ─────────────────────────────────────────────────

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
unsafe fn axpy_neg_avx(weights: &mut [f32], grads: &[f32], lr: f32) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;
    let len = weights.len();
    let wp = weights.as_mut_ptr();
    let gp = grads.as_ptr();
    let vlr = _mm256_set1_ps(lr);
    let mut i = 0usize;
    while i + 8 <= len {
        let w = _mm256_loadu_ps(wp.add(i));
        let g = _mm256_loadu_ps(gp.add(i));
        _mm256_storeu_ps(wp.add(i), _mm256_sub_ps(w, _mm256_mul_ps(g, vlr)));
        i += 8;
    }
    while i < len {
        *wp.add(i) -= lr * *gp.add(i);
        i += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
unsafe fn fma_inplace_avx(dst: &mut [f32], src: &[f32], scale: f32) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;
    let len = dst.len();
    let dp = dst.as_mut_ptr();
    let sp = src.as_ptr();
    let vs = _mm256_set1_ps(scale);
    let mut i = 0usize;
    while i + 8 <= len {
        let d = _mm256_loadu_ps(dp.add(i));
        let s = _mm256_loadu_ps(sp.add(i));
        _mm256_storeu_ps(dp.add(i), _mm256_add_ps(d, _mm256_mul_ps(s, vs)));
        i += 8;
    }
    while i < len {
        *dp.add(i) += *sp.add(i) * scale;
        i += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
unsafe fn momentum_update_avx(velocity: &mut [f32], grad: &[f32], momentum: f32, grad_scale: f32) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;
    let len = velocity.len();
    let vp = velocity.as_mut_ptr();
    let gp = grad.as_ptr();
    let vmom = _mm256_set1_ps(momentum);
    let vgs = _mm256_set1_ps(grad_scale);
    let mut i = 0usize;
    while i + 8 <= len {
        let v = _mm256_loadu_ps(vp.add(i));
        let g = _mm256_loadu_ps(gp.add(i));
        let result = _mm256_add_ps(_mm256_mul_ps(vmom, v), _mm256_mul_ps(g, vgs));
        _mm256_storeu_ps(vp.add(i), result);
        i += 8;
    }
    while i < len {
        *vp.add(i) = momentum * *vp.add(i) + grad_scale * *gp.add(i);
        i += 1;
    }
}

// ── SSE implementations ─────────────────────────────────────────────────

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
unsafe fn axpy_neg_sse(weights: &mut [f32], grads: &[f32], lr: f32) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;
    let len = weights.len();
    let wp = weights.as_mut_ptr();
    let gp = grads.as_ptr();
    let vlr = _mm_set1_ps(lr);
    let mut i = 0usize;
    while i + 4 <= len {
        let w = _mm_loadu_ps(wp.add(i));
        let g = _mm_loadu_ps(gp.add(i));
        _mm_storeu_ps(wp.add(i), _mm_sub_ps(w, _mm_mul_ps(g, vlr)));
        i += 4;
    }
    while i < len {
        *wp.add(i) -= lr * *gp.add(i);
        i += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
unsafe fn fma_inplace_sse(dst: &mut [f32], src: &[f32], scale: f32) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;
    let len = dst.len();
    let dp = dst.as_mut_ptr();
    let sp = src.as_ptr();
    let vs = _mm_set1_ps(scale);
    let mut i = 0usize;
    while i + 4 <= len {
        let d = _mm_loadu_ps(dp.add(i));
        let s = _mm_loadu_ps(sp.add(i));
        _mm_storeu_ps(dp.add(i), _mm_add_ps(d, _mm_mul_ps(s, vs)));
        i += 4;
    }
    while i < len {
        *dp.add(i) += *sp.add(i) * scale;
        i += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
unsafe fn momentum_update_sse(velocity: &mut [f32], grad: &[f32], momentum: f32, grad_scale: f32) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;
    let len = velocity.len();
    let vp = velocity.as_mut_ptr();
    let gp = grad.as_ptr();
    let vmom = _mm_set1_ps(momentum);
    let vgs = _mm_set1_ps(grad_scale);
    let mut i = 0usize;
    while i + 4 <= len {
        let v = _mm_loadu_ps(vp.add(i));
        let g = _mm_loadu_ps(gp.add(i));
        let result = _mm_add_ps(_mm_mul_ps(vmom, v), _mm_mul_ps(g, vgs));
        _mm_storeu_ps(vp.add(i), result);
        i += 4;
    }
    while i < len {
        *vp.add(i) = momentum * *vp.add(i) + grad_scale * *gp.add(i);
        i += 1;
    }
}
