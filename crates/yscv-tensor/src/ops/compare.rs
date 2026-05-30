//! Comparison + logical ops (eq/lt/gt/…, all_finite).

use super::super::error::TensorError;
use super::super::simd;
use super::super::tensor::Tensor;

impl Tensor {
    // ── Comparison ──────────────────────────────────────────────────────

    /// Element-wise equality check: 1.0 where equal, 0.0 otherwise.
    pub fn eq_tensor(&self, rhs: &Self) -> Result<Self, TensorError> {
        if self.shape() == rhs.shape() {
            return self.binary_same_shape(rhs, |l, r| {
                if (l - r).abs() < f32::EPSILON {
                    1.0
                } else {
                    0.0
                }
            });
        }
        self.binary_broadcast_op(rhs, |l, r| {
            if (l - r).abs() < f32::EPSILON {
                1.0
            } else {
                0.0
            }
        })
    }

    /// Element-wise greater-than: 1.0 where `self > rhs`, 0.0 otherwise.
    pub fn gt_tensor(&self, rhs: &Self) -> Result<Self, TensorError> {
        if self.shape() == rhs.shape() {
            return self.binary_same_shape(rhs, |l, r| if l > r { 1.0 } else { 0.0 });
        }
        self.binary_broadcast_op(rhs, |l, r| if l > r { 1.0 } else { 0.0 })
    }

    /// Element-wise less-than: 1.0 where `self < rhs`, 0.0 otherwise.
    pub fn lt_tensor(&self, rhs: &Self) -> Result<Self, TensorError> {
        if self.shape() == rhs.shape() {
            return self.binary_same_shape(rhs, |l, r| if l < r { 1.0 } else { 0.0 });
        }
        self.binary_broadcast_op(rhs, |l, r| if l < r { 1.0 } else { 0.0 })
    }

    /// Element-wise greater-than writing into a pre-allocated output tensor.
    /// `self`, `rhs`, and `output` must all have the same shape.
    pub fn gt_tensor_into(&self, rhs: &Self, output: &mut Self) {
        debug_assert_eq!(self.shape(), rhs.shape());
        debug_assert_eq!(self.shape(), output.shape());
        simd::cmp_dispatch(
            self.data(),
            rhs.data(),
            output.data_mut(),
            simd::CmpKind::Gt,
        );
    }

    /// Element-wise equality check writing into a pre-allocated output tensor.
    /// `self`, `rhs`, and `output` must all have the same shape.
    pub fn eq_tensor_into(&self, rhs: &Self, output: &mut Self) {
        debug_assert_eq!(self.shape(), rhs.shape());
        debug_assert_eq!(self.shape(), output.shape());
        simd::cmp_dispatch(
            self.data(),
            rhs.data(),
            output.data_mut(),
            simd::CmpKind::Eq,
        );
    }

    /// Element-wise less-than writing into a pre-allocated output tensor.
    /// `self`, `rhs`, and `output` must all have the same shape.
    pub fn lt_tensor_into(&self, rhs: &Self, output: &mut Self) {
        debug_assert_eq!(self.shape(), rhs.shape());
        debug_assert_eq!(self.shape(), output.shape());
        simd::cmp_dispatch(
            self.data(),
            rhs.data(),
            output.data_mut(),
            simd::CmpKind::Lt,
        );
    }

    /// Element-wise not-equal: 1.0 where not equal (diff.abs() >= 1e-7), 0.0 otherwise.
    pub fn ne_tensor(&self, rhs: &Self) -> Result<Self, TensorError> {
        if self.shape() == rhs.shape() {
            return self.binary_same_shape(
                rhs,
                |l, r| {
                    if (l - r).abs() >= 1e-7 { 1.0 } else { 0.0 }
                },
            );
        }
        self.binary_broadcast_op(rhs, |l, r| if (l - r).abs() >= 1e-7 { 1.0 } else { 0.0 })
    }

    /// Element-wise less-than-or-equal: 1.0 where `self <= rhs`, 0.0 otherwise.
    pub fn le_tensor(&self, rhs: &Self) -> Result<Self, TensorError> {
        if self.shape() == rhs.shape() {
            return self.binary_same_shape(rhs, |l, r| if l <= r { 1.0 } else { 0.0 });
        }
        self.binary_broadcast_op(rhs, |l, r| if l <= r { 1.0 } else { 0.0 })
    }

    /// Element-wise greater-than-or-equal: 1.0 where `self >= rhs`, 0.0 otherwise.
    pub fn ge_tensor(&self, rhs: &Self) -> Result<Self, TensorError> {
        if self.shape() == rhs.shape() {
            return self.binary_same_shape(rhs, |l, r| if l >= r { 1.0 } else { 0.0 });
        }
        self.binary_broadcast_op(rhs, |l, r| if l >= r { 1.0 } else { 0.0 })
    }

    /// Returns true if all elements are finite (no NaN or Inf).
    pub fn all_finite(&self) -> bool {
        self.data().iter().all(|v| v.is_finite())
    }
}
