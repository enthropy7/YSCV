//! Element-wise tensor ops: binary (broadcasting) + unary math/trig.

use super::super::aligned::AlignedVec;
use super::super::error::TensorError;
use super::super::simd;
use super::super::tensor::Tensor;
use super::fast_atan2_scalar;

impl Tensor {
    // ── Binary elementwise with broadcasting ────────────────────────────

    /// Element-wise addition with NumPy-style broadcasting.
    pub fn add(&self, rhs: &Self) -> Result<Self, TensorError> {
        if self.shape() == rhs.shape() {
            return self.binary_same_shape_simd(rhs, simd::BinaryKind::Add);
        }
        if let Some(result) = self.binary_broadcast_lastdim_simd(rhs, simd::BinaryKind::Add) {
            return result;
        }
        self.binary_broadcast_op(rhs, |l, r| l + r)
    }

    /// Element-wise subtraction with NumPy-style broadcasting.
    pub fn sub(&self, rhs: &Self) -> Result<Self, TensorError> {
        if self.shape() == rhs.shape() {
            return self.binary_same_shape_simd(rhs, simd::BinaryKind::Sub);
        }
        if let Some(result) = self.binary_broadcast_lastdim_simd(rhs, simd::BinaryKind::Sub) {
            return result;
        }
        self.binary_broadcast_op(rhs, |l, r| l - r)
    }

    /// Element-wise multiplication with NumPy-style broadcasting.
    pub fn mul(&self, rhs: &Self) -> Result<Self, TensorError> {
        if self.shape() == rhs.shape() {
            return self.binary_same_shape_simd(rhs, simd::BinaryKind::Mul);
        }
        if let Some(result) = self.binary_broadcast_lastdim_simd(rhs, simd::BinaryKind::Mul) {
            return result;
        }
        self.binary_broadcast_op(rhs, |l, r| l * r)
    }

    /// Element-wise division with NumPy-style broadcasting.
    pub fn div(&self, rhs: &Self) -> Result<Self, TensorError> {
        if self.shape() == rhs.shape() {
            return self.binary_same_shape_simd(rhs, simd::BinaryKind::Div);
        }
        if let Some(result) = self.binary_broadcast_lastdim_simd(rhs, simd::BinaryKind::Div) {
            return result;
        }
        self.binary_broadcast_op(rhs, |l, r| l / r)
    }

    /// Element-wise power with NumPy-style broadcasting.
    ///
    /// Fast paths:
    /// - Constant exponent 2.0 → `mul(x, x)` (SIMD-accelerated).
    /// - Constant exponent 0.5 → `sqrt(x)` (SIMD-accelerated).
    /// - Same-shape general case → SIMD `exp(exp * ln(base))`.
    #[allow(unsafe_code)]
    pub fn pow(&self, rhs: &Self) -> Result<Self, TensorError> {
        // Fast path: constant exponent (O(1) check instead of O(N) scan).
        // A single-element tensor or a tensor whose total size is 1 (broadcast scalar)
        // is always constant.  For same-shape tensors we skip the all-equal scan
        // entirely — the per-element SIMD path handles them efficiently.
        let rhs_total: usize = rhs.shape().iter().product();
        if rhs_total == 1 {
            let exp_val = rhs.data()[0];
            if exp_val == 2.0 {
                return self.mul(self);
            }
            if exp_val == 0.5 {
                return Ok(self.sqrt());
            }
            if exp_val == 1.0 {
                return Ok(self.clone());
            }
            if exp_val == 0.0 {
                return Tensor::ones(self.shape().to_vec());
            }
            if exp_val == -1.0 {
                return Ok(self.reciprocal());
            }
        }
        // Same-shape SIMD path: pow(base, exp) = exp(exp * ln(base))
        if self.shape() == rhs.shape() {
            return self.pow_simd(rhs);
        }
        self.binary_broadcast_op(rhs, |l, r| l.powf(r))
    }

    /// SIMD-accelerated pow for same-shape tensors via exp(exp * ln(base)).
    #[allow(unsafe_code)]
    fn pow_simd(&self, rhs: &Self) -> Result<Self, TensorError> {
        let len = self.len();
        // Compute ln(base)
        let mut ln_buf = AlignedVec::<f32>::uninitialized(len);
        simd::ln_dispatch(self.data(), &mut ln_buf);
        // Multiply by exponent: exp_val * ln(base)
        let mut prod_buf = AlignedVec::<f32>::uninitialized(len);
        simd::binary_dispatch(&ln_buf, rhs.data(), &mut prod_buf, simd::BinaryKind::Mul);
        // Compute exp(exp_val * ln(base))
        let mut out = AlignedVec::<f32>::uninitialized(len);
        simd::exp_dispatch(&prod_buf, &mut out);
        Ok(Tensor::from_raw_parts(self.shape(), self.strides(), out))
    }

    /// Element-wise atan2(self, other), with broadcasting.
    ///
    /// Same-shape case uses a SIMD-friendly polynomial approximation of atan
    /// with quadrant correction, avoiding scalar `f32::atan2`.
    #[allow(unsafe_code)]
    pub fn atan2(&self, rhs: &Self) -> Result<Self, TensorError> {
        if self.shape() == rhs.shape() {
            return self.atan2_fast(rhs);
        }
        self.binary_broadcast_op(rhs, f32::atan2)
    }

    /// Vectorized atan2 for same-shape tensors. Uses Cephes-style range
    /// reduction to [0, tan(pi/12)] then a polynomial approximation,
    /// giving < 1e-6 max error across all quadrants.
    #[allow(unsafe_code)]
    fn atan2_fast(&self, rhs: &Self) -> Result<Self, TensorError> {
        let y_data = self.data();
        let x_data = rhs.data();
        let len = self.len();
        let mut out = AlignedVec::<f32>::uninitialized(len);

        // Process in chunks of 4 for ILP (instruction-level parallelism).
        // Each call to the scalar fast path avoids the overhead of the
        // generic broadcasting machinery.
        let mut i = 0;
        while i + 4 <= len {
            out[i] = fast_atan2_scalar(y_data[i], x_data[i]);
            out[i + 1] = fast_atan2_scalar(y_data[i + 1], x_data[i + 1]);
            out[i + 2] = fast_atan2_scalar(y_data[i + 2], x_data[i + 2]);
            out[i + 3] = fast_atan2_scalar(y_data[i + 3], x_data[i + 3]);
            i += 4;
        }
        while i < len {
            out[i] = fast_atan2_scalar(y_data[i], x_data[i]);
            i += 1;
        }

        Ok(Tensor::from_raw_parts(self.shape(), self.strides(), out))
    }

    /// Element-wise minimum with NumPy-style broadcasting.
    pub fn minimum(&self, rhs: &Self) -> Result<Self, TensorError> {
        if self.shape() == rhs.shape() {
            return self.binary_same_shape(rhs, f32::min);
        }
        self.binary_broadcast_op(rhs, f32::min)
    }

    /// Element-wise maximum with NumPy-style broadcasting.
    pub fn maximum(&self, rhs: &Self) -> Result<Self, TensorError> {
        if self.shape() == rhs.shape() {
            return self.binary_same_shape(rhs, f32::max);
        }
        self.binary_broadcast_op(rhs, f32::max)
    }

    // ── Unary elementwise ───────────────────────────────────────────────

    /// Element-wise negation.
    pub fn neg(&self) -> Self {
        self.unary_simd_op(simd::UnaryKind::Neg)
    }

    /// Element-wise absolute value.
    pub fn abs(&self) -> Self {
        self.unary_simd_op(simd::UnaryKind::Abs)
    }

    /// Element-wise natural exponential.
    #[allow(unsafe_code)]
    pub fn exp(&self) -> Self {
        let len = self.len();
        // SAFETY: `uninitialized` allocates without zeroing.  `exp_dispatch`
        // writes every element before anything reads from `out`.
        let mut out = AlignedVec::<f32>::uninitialized(len);
        simd::exp_dispatch(self.data(), &mut out);
        Tensor::from_raw_parts(self.shape(), self.strides(), out)
    }

    /// Element-wise natural logarithm.
    #[allow(unsafe_code)]
    pub fn ln(&self) -> Self {
        let len = self.len();
        // SAFETY: `uninitialized` allocates without zeroing.  `ln_dispatch`
        // writes every element before we ever read from `out`.
        let mut out = AlignedVec::<f32>::uninitialized(len);
        simd::ln_dispatch(self.data(), &mut out);
        Tensor::from_raw_parts(self.shape(), self.strides(), out)
    }

    /// Element-wise square root.
    pub fn sqrt(&self) -> Self {
        self.unary_simd_op(simd::UnaryKind::Sqrt)
    }

    /// Element-wise reciprocal (`1 / x`).
    pub fn reciprocal(&self) -> Self {
        self.unary_simd_op(simd::UnaryKind::Recip)
    }

    /// Element-wise sign (`-1`, `0`, or `1`).
    pub fn sign(&self) -> Self {
        self.unary_simd_op(simd::UnaryKind::Sign)
    }

    /// Element-wise floor.
    pub fn floor(&self) -> Self {
        self.unary_simd_op(simd::UnaryKind::Floor)
    }

    /// Element-wise ceil.
    pub fn ceil(&self) -> Self {
        self.unary_simd_op(simd::UnaryKind::Ceil)
    }

    /// Element-wise round.
    pub fn round(&self) -> Self {
        self.unary_simd_op(simd::UnaryKind::Round)
    }

    /// Element-wise sine (SIMD-accelerated polynomial approximation).
    #[allow(unsafe_code)]
    pub fn sin(&self) -> Self {
        let len = self.len();
        let mut out = AlignedVec::<f32>::uninitialized(len);
        simd::sin_dispatch(self.data(), &mut out);
        Tensor::from_raw_parts(self.shape(), self.strides(), out)
    }

    /// Element-wise cosine (SIMD-accelerated polynomial approximation).
    #[allow(unsafe_code)]
    pub fn cos(&self) -> Self {
        let len = self.len();
        let mut out = AlignedVec::<f32>::uninitialized(len);
        simd::cos_dispatch(self.data(), &mut out);
        Tensor::from_raw_parts(self.shape(), self.strides(), out)
    }

    /// Element-wise tangent (SIMD-accelerated via sin/cos).
    #[allow(unsafe_code)]
    pub fn tan(&self) -> Self {
        let len = self.len();
        let mut out = AlignedVec::<f32>::uninitialized(len);
        simd::tan_dispatch(self.data(), &mut out);
        Tensor::from_raw_parts(self.shape(), self.strides(), out)
    }

    /// Element-wise arcsine.
    pub fn asin(&self) -> Self {
        self.unary_op(f32::asin)
    }

    /// Element-wise arccosine.
    pub fn acos(&self) -> Self {
        self.unary_op(f32::acos)
    }

    /// Element-wise arctangent.
    pub fn atan(&self) -> Self {
        self.unary_op(f32::atan)
    }

    /// Element-wise hyperbolic sine.
    pub fn sinh(&self) -> Self {
        self.unary_op(f32::sinh)
    }

    /// Element-wise hyperbolic cosine.
    pub fn cosh(&self) -> Self {
        self.unary_op(f32::cosh)
    }

    /// Element-wise base-2 logarithm.
    pub fn log2(&self) -> Self {
        self.unary_op(f32::log2)
    }

    /// Element-wise base-10 logarithm.
    pub fn log10(&self) -> Self {
        self.unary_op(f32::log10)
    }

    /// Convert radians to degrees.
    pub fn degrees(&self) -> Self {
        self.unary_op(|v| v.to_degrees())
    }

    /// Convert degrees to radians.
    pub fn radians(&self) -> Self {
        self.unary_op(|v| v.to_radians())
    }

    /// Clamp all elements to `[min, max]`.
    #[allow(unsafe_code)]
    pub fn clamp(&self, min: f32, max: f32) -> Self {
        let len = self.len();
        let mut out = AlignedVec::<f32>::uninitialized(len);
        simd::clamp_dispatch(self.data(), &mut out, min, max);
        Tensor::from_raw_parts(self.shape(), self.strides(), out)
    }

    /// Scalar multiplication (broadcast multiply by a constant).
    pub fn scale(&self, factor: f32) -> Self {
        self.unary_op(|v| v * factor)
    }

    /// Add a scalar to all elements.
    pub fn add_scalar(&self, value: f32) -> Self {
        self.unary_op(|v| v + value)
    }
}
