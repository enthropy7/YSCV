use super::aligned::AlignedVec;
use super::error::TensorError;
use super::shape::{
    broadcast_offset, broadcast_shape, compute_strides, increment_coords, shape_element_count,
};
use super::simd;
use super::tensor::Tensor;

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

    // ── Reductions ──────────────────────────────────────────────────────

    /// Sum reduction over all elements.
    pub fn sum(&self) -> f32 {
        simd::sum_dispatch(self.data())
    }

    /// Mean reduction over all elements.
    pub fn mean(&self) -> f32 {
        if self.is_empty() {
            return f32::NAN;
        }
        self.sum() / self.len() as f32
    }

    /// Global max reduction. Returns `f32::NEG_INFINITY` for empty tensors.
    pub fn max_value(&self) -> f32 {
        simd::max_dispatch(self.data())
    }

    /// Global min reduction. Returns `f32::INFINITY` for empty tensors.
    pub fn min_value(&self) -> f32 {
        simd::min_dispatch(self.data())
    }

    /// Global argmax (flat index of maximum value).
    pub fn argmax(&self) -> Option<usize> {
        if self.is_empty() {
            return None;
        }
        let mut best = 0;
        let mut best_val = self.data()[0];
        for (i, &v) in self.data().iter().enumerate().skip(1) {
            if v > best_val {
                best_val = v;
                best = i;
            }
        }
        Some(best)
    }

    /// Global argmin (flat index of minimum value).
    pub fn argmin(&self) -> Option<usize> {
        if self.is_empty() {
            return None;
        }
        let mut best = 0;
        let mut best_val = self.data()[0];
        for (i, &v) in self.data().iter().enumerate().skip(1) {
            if v < best_val {
                best_val = v;
                best = i;
            }
        }
        Some(best)
    }

    /// Variance over all elements (population variance).
    pub fn var(&self) -> f32 {
        if self.is_empty() {
            return f32::NAN;
        }
        let m = self.mean();
        self.data().iter().map(|&v| (v - m) * (v - m)).sum::<f32>() / self.len() as f32
    }

    /// Standard deviation over all elements (population).
    pub fn std_dev(&self) -> f32 {
        self.var().sqrt()
    }

    /// Sum reduction over one axis. Reduced axis is removed from output shape.
    pub fn sum_axis(&self, axis: usize) -> Result<Self, TensorError> {
        let shape = self.shape();
        let rank = shape.len();
        if axis >= rank {
            return Err(TensorError::InvalidAxis { axis, rank });
        }

        // Fast path: 2D contiguous tensor, axis 0 → accumulate rows with SIMD
        if rank == 2 && axis == 0 {
            let (rows, cols) = (shape[0], shape[1]);
            let data = self.data();
            let mut out = vec![0.0f32; cols];
            for row in 0..rows {
                let row_start = row * cols;
                simd::add_inplace_dispatch(&mut out, &data[row_start..row_start + cols]);
            }
            return Self::from_vec(vec![cols], out);
        }

        // Fast path: 2D contiguous tensor, axis 1 → sum each row with SIMD
        if rank == 2 && axis == 1 {
            let (rows, cols) = (shape[0], shape[1]);
            let data = self.data();
            let mut out = Vec::with_capacity(rows);
            for row in 0..rows {
                out.push(simd::sum_dispatch(&data[row * cols..(row + 1) * cols]));
            }
            return Self::from_vec(vec![rows], out);
        }

        self.reduce_axis(axis, 0.0, |acc, v| acc + v)
    }

    /// Mean reduction over one axis. Reduced axis is removed from output shape.
    pub fn mean_axis(&self, axis: usize) -> Result<Self, TensorError> {
        if axis >= self.rank() {
            return Err(TensorError::InvalidAxis {
                axis,
                rank: self.rank(),
            });
        }
        let axis_len = self.shape()[axis] as f32;
        let sum = self.sum_axis(axis)?;
        Ok(sum.scale(1.0 / axis_len))
    }

    /// Max reduction over one axis. Reduced axis is removed from output shape.
    pub fn max_axis(&self, axis: usize) -> Result<Self, TensorError> {
        let shape = self.shape();
        let rank = shape.len();
        if axis >= rank {
            return Err(TensorError::InvalidAxis { axis, rank });
        }

        // Fast path: 2D contiguous tensor, axis 0 → accumulate rows with SIMD max
        if rank == 2 && axis == 0 {
            let (rows, cols) = (shape[0], shape[1]);
            let data = self.data();
            let mut out = data[..cols].to_vec();
            for row in 1..rows {
                let row_start = row * cols;
                simd::max_inplace_dispatch(&mut out, &data[row_start..row_start + cols]);
            }
            return Self::from_vec(vec![cols], out);
        }

        // Fast path: 2D contiguous tensor, axis 1 → max each row with SIMD
        if rank == 2 && axis == 1 {
            let (rows, cols) = (shape[0], shape[1]);
            let data = self.data();
            let mut out = Vec::with_capacity(rows);
            for row in 0..rows {
                out.push(simd::max_dispatch(&data[row * cols..(row + 1) * cols]));
            }
            return Self::from_vec(vec![rows], out);
        }

        self.reduce_axis(axis, f32::NEG_INFINITY, f32::max)
    }

    /// Min reduction over one axis. Reduced axis is removed from output shape.
    pub fn min_axis(&self, axis: usize) -> Result<Self, TensorError> {
        self.reduce_axis(axis, f32::INFINITY, f32::min)
    }

    /// Variance reduction over one axis (population variance).
    pub fn var_axis(&self, axis: usize) -> Result<Self, TensorError> {
        let m = self.mean_axis(axis)?;
        let diff = self.sub(&m.unsqueeze(axis)?)?;
        let sq = diff.mul(&diff)?;
        sq.mean_axis(axis)
    }

    /// Global median of all elements.
    pub fn median(&self) -> f32 {
        let mut sorted = self.data().to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = sorted.len();
        if n == 0 {
            return 0.0;
        }
        if n % 2 == 1 {
            sorted[n / 2]
        } else {
            (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
        }
    }

    /// Median along a given axis.
    pub fn median_axis(&self, axis: usize) -> Result<Self, TensorError> {
        let shape = self.shape();
        let rank = shape.len();
        if axis >= rank {
            return Err(TensorError::InvalidAxis { axis, rank });
        }
        let axis_len = shape[axis];
        let outer: usize = shape[..axis].iter().product();
        let inner: usize = shape[axis + 1..].iter().product();
        let mut new_shape = shape.to_vec();
        new_shape.remove(axis);
        if new_shape.is_empty() {
            new_shape.push(1);
        }
        let data = self.data();
        let mut result = Vec::with_capacity(outer * inner);
        for o in 0..outer {
            for i in 0..inner {
                let mut vals: Vec<f32> = (0..axis_len)
                    .map(|a| data[o * axis_len * inner + a * inner + i])
                    .collect();
                vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let n = vals.len();
                let med = if n % 2 == 1 {
                    vals[n / 2]
                } else {
                    (vals[n / 2 - 1] + vals[n / 2]) / 2.0
                };
                result.push(med);
            }
        }
        Self::from_vec(new_shape, result)
    }

    /// Returns true if any element is non-zero.
    pub fn any(&self) -> bool {
        self.data().iter().any(|&v| v != 0.0)
    }

    /// Returns true if all elements are non-zero.
    pub fn all(&self) -> bool {
        self.data().iter().all(|&v| v != 0.0)
    }

    /// Quantile of all elements. q must be in [0, 1].
    pub fn quantile(&self, q: f32) -> f32 {
        let mut sorted = self.data().to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = sorted.len();
        if n == 0 {
            return 0.0;
        }
        let idx = q * (n - 1) as f32;
        let lo = idx.floor() as usize;
        let hi = idx.ceil() as usize;
        if lo == hi || hi >= n {
            sorted[lo.min(n - 1)]
        } else {
            let frac = idx - lo as f32;
            sorted[lo] * (1.0 - frac) + sorted[hi] * frac
        }
    }

    // ── Shape manipulation ──────────────────────────────────────────────

    /// 2D matrix transpose. Requires rank-2 input.
    ///
    /// # Safety
    /// `AlignedVec::uninitialized` allocates without zeroing. The tiled loop
    /// writes every element before anything reads from the buffer.
    #[allow(unsafe_code)]
    pub fn transpose_2d(&self) -> Result<Self, TensorError> {
        if self.rank() != 2 {
            return Err(TensorError::InvalidAxis {
                axis: 1,
                rank: self.rank(),
            });
        }
        let rows = self.shape()[0];
        let cols = self.shape()[1];
        // SAFETY: every element is written by the tiled loop below before we read.
        let mut out_data = AlignedVec::<f32>::uninitialized(rows * cols);
        let src = self.data();

        // Tiled transpose with 8x8 blocks for cache efficiency.
        const TILE: usize = 8;
        let rr = rows / TILE * TILE;
        let cc = cols / TILE * TILE;

        for ii in (0..rr).step_by(TILE) {
            for jj in (0..cc).step_by(TILE) {
                for r in ii..ii + TILE {
                    for c in jj..jj + TILE {
                        out_data[c * rows + r] = src[r * cols + c];
                    }
                }
            }
            // Right edge (columns beyond cc)
            for r in ii..ii + TILE {
                for c in cc..cols {
                    out_data[c * rows + r] = src[r * cols + c];
                }
            }
        }
        // Bottom edge (rows beyond rr)
        for r in rr..rows {
            for c in 0..cols {
                out_data[c * rows + r] = src[r * cols + c];
            }
        }

        Tensor::from_aligned(vec![cols, rows], out_data)
    }

    /// General axis permutation (like NumPy `transpose`/`permute`).
    pub fn permute(&self, axes: &[usize]) -> Result<Self, TensorError> {
        if axes.len() != self.rank() {
            return Err(TensorError::InvalidIndexRank {
                expected: self.rank(),
                got: axes.len(),
            });
        }
        let rank = self.rank();
        let mut seen = vec![false; rank];
        for &a in axes {
            if a >= rank {
                return Err(TensorError::InvalidAxis { axis: a, rank });
            }
            seen[a] = true;
        }
        if seen.iter().any(|&s| !s) {
            return Err(TensorError::InvalidAxis { axis: 0, rank });
        }

        let src_shape = self.shape();
        let mut out_shape = vec![0usize; rank];
        for (dst, &src_axis) in axes.iter().enumerate() {
            out_shape[dst] = src_shape[src_axis];
        }
        let out_count =
            shape_element_count(&out_shape).ok_or_else(|| TensorError::SizeOverflow {
                shape: out_shape.clone(),
            })?;

        // ── Fast path: tiled 2D transpose for common 4D permutations ──
        // Uses unsafe pointer arithmetic to eliminate bounds checks in hot inner loops.
        // NHWC→NCHW [0,3,1,2]: transpose inner [H*W, C] → [C, H*W]
        if rank == 4 && axes == [0, 3, 1, 2] {
            let (n, h, w, c) = (src_shape[0], src_shape[1], src_shape[2], src_shape[3]);
            let hw = h * w;
            let src = self.data();
            let mut dst = AlignedVec::<f32>::uninitialized(out_count);
            const TILE: usize = 32;
            #[allow(unsafe_code)]
            unsafe {
                let src_ptr = src.as_ptr();
                let dst_ptr = dst.as_mut_ptr();
                for batch in 0..n {
                    let s_base = src_ptr.add(batch * hw * c);
                    let d_base = dst_ptr.add(batch * c * hw);
                    for i0 in (0..hw).step_by(TILE) {
                        let ie = (i0 + TILE).min(hw);
                        for j0 in (0..c).step_by(TILE) {
                            let je = (j0 + TILE).min(c);
                            for i in i0..ie {
                                let s_row = s_base.add(i * c);
                                for j in j0..je {
                                    *d_base.add(j * hw + i) = *s_row.add(j);
                                }
                            }
                        }
                    }
                }
            }
            return Tensor::from_aligned(out_shape, dst);
        }
        // NCHW→NHWC [0,2,3,1]: transpose inner [C, H*W] → [H*W, C]
        if rank == 4 && axes == [0, 2, 3, 1] {
            let (n, c, h, w) = (src_shape[0], src_shape[1], src_shape[2], src_shape[3]);
            let hw = h * w;
            let src = self.data();
            let mut dst = AlignedVec::<f32>::uninitialized(out_count);
            const TILE: usize = 32;
            #[allow(unsafe_code)]
            unsafe {
                let src_ptr = src.as_ptr();
                let dst_ptr = dst.as_mut_ptr();
                for batch in 0..n {
                    let s_base = src_ptr.add(batch * c * hw);
                    let d_base = dst_ptr.add(batch * hw * c);
                    for i0 in (0..c).step_by(TILE) {
                        let ie = (i0 + TILE).min(c);
                        for j0 in (0..hw).step_by(TILE) {
                            let je = (j0 + TILE).min(hw);
                            for i in i0..ie {
                                let s_row = s_base.add(i * hw);
                                for j in j0..je {
                                    *d_base.add(j * c + i) = *s_row.add(j);
                                }
                            }
                        }
                    }
                }
            }
            return Tensor::from_aligned(out_shape, dst);
        }
        // 3D swap last two dims [0,2,1]: transpose [A, B, C] → [A, C, B]
        if rank == 3 && axes == [0, 2, 1] {
            let (a, b, c) = (src_shape[0], src_shape[1], src_shape[2]);
            let src = self.data();
            let mut dst = AlignedVec::<f32>::uninitialized(out_count);
            const TILE: usize = 32;
            #[allow(unsafe_code)]
            unsafe {
                let src_ptr = src.as_ptr();
                let dst_ptr = dst.as_mut_ptr();
                for batch in 0..a {
                    let s_base = src_ptr.add(batch * b * c);
                    let d_base = dst_ptr.add(batch * c * b);
                    for i0 in (0..b).step_by(TILE) {
                        let ie = (i0 + TILE).min(b);
                        for j0 in (0..c).step_by(TILE) {
                            let je = (j0 + TILE).min(c);
                            for i in i0..ie {
                                let s_row = s_base.add(i * c);
                                for j in j0..je {
                                    *d_base.add(j * b + i) = *s_row.add(j);
                                }
                            }
                        }
                    }
                }
            }
            return Tensor::from_aligned(out_shape, dst);
        }

        // [0,1,3,2]: swap last two dims in 4D → [N, A, C, B]
        // For each (n, a), tiled 2D transpose of [B, C] → [C, B].
        if rank == 4 && axes == [0, 1, 3, 2] {
            let (nn, a, b, c) = (src_shape[0], src_shape[1], src_shape[2], src_shape[3]);
            let src = self.data();
            let mut dst = AlignedVec::<f32>::uninitialized(out_count);
            const TILE: usize = 32;
            #[allow(unsafe_code)]
            unsafe {
                let src_ptr = src.as_ptr();
                let dst_ptr = dst.as_mut_ptr();
                for n in 0..nn {
                    for aa in 0..a {
                        let base = (n * a + aa) * b * c;
                        let s_base = src_ptr.add(base);
                        let d_base = dst_ptr.add(base); // same offset, different shape
                        for i0 in (0..b).step_by(TILE) {
                            let ie = (i0 + TILE).min(b);
                            for j0 in (0..c).step_by(TILE) {
                                let je = (j0 + TILE).min(c);
                                for i in i0..ie {
                                    let s_row = s_base.add(i * c);
                                    for j in j0..je {
                                        *d_base.add(j * b + i) = *s_row.add(j);
                                    }
                                }
                            }
                        }
                    }
                }
            }
            return Tensor::from_aligned(out_shape, dst);
        }
        // [0,2,1,3]: swap dims 1↔2 in 4D → [N, B, A, C]
        // Each element in the swap is a contiguous block of C floats — use memcpy.
        if rank == 4 && axes == [0, 2, 1, 3] {
            let (nn, a, b, c) = (src_shape[0], src_shape[1], src_shape[2], src_shape[3]);
            let src = self.data();
            let mut dst = AlignedVec::<f32>::uninitialized(out_count);
            #[allow(unsafe_code)]
            unsafe {
                let src_ptr = src.as_ptr();
                let dst_ptr = dst.as_mut_ptr();
                for n in 0..nn {
                    let s_batch = src_ptr.add(n * a * b * c);
                    let d_batch = dst_ptr.add(n * b * a * c);
                    for aa in 0..a {
                        for bb in 0..b {
                            std::ptr::copy_nonoverlapping(
                                s_batch.add(aa * b * c + bb * c),
                                d_batch.add(bb * a * c + aa * c),
                                c,
                            );
                        }
                    }
                }
            }
            return Tensor::from_aligned(out_shape, dst);
        }
        // [0,3,2,1]: swap dims 1↔3 in 4D → [N, D, B, A]
        // For each (n, b), tiled 2D transpose of [A, D] → [D, A] with strides.
        if rank == 4 && axes == [0, 3, 2, 1] {
            let (nn, a, b, d) = (src_shape[0], src_shape[1], src_shape[2], src_shape[3]);
            let src = self.data();
            let mut dst = AlignedVec::<f32>::uninitialized(out_count);
            let src_a_stride = b * d;
            let dst_d_stride = b * a;
            const TILE: usize = 32;
            #[allow(unsafe_code)]
            unsafe {
                let src_ptr = src.as_ptr();
                let dst_ptr = dst.as_mut_ptr();
                for n in 0..nn {
                    for bb in 0..b {
                        let s_base = src_ptr.add(n * a * b * d + bb * d);
                        let d_base = dst_ptr.add(n * d * b * a + bb * a);
                        for i0 in (0..a).step_by(TILE) {
                            let ie = (i0 + TILE).min(a);
                            for j0 in (0..d).step_by(TILE) {
                                let je = (j0 + TILE).min(d);
                                for i in i0..ie {
                                    for j in j0..je {
                                        *d_base.add(j * dst_d_stride + i) =
                                            *s_base.add(i * src_a_stride + j);
                                    }
                                }
                            }
                        }
                    }
                }
            }
            return Tensor::from_aligned(out_shape, dst);
        }
        // 2D transpose [1,0]: swap rows and cols
        if rank == 2 && axes == [1, 0] {
            let (rows, cols) = (src_shape[0], src_shape[1]);
            let src = self.data();
            let mut dst = AlignedVec::<f32>::uninitialized(out_count);
            const TILE: usize = 32;
            #[allow(unsafe_code)]
            unsafe {
                let src_ptr = src.as_ptr();
                let dst_ptr = dst.as_mut_ptr();
                for i0 in (0..rows).step_by(TILE) {
                    let ie = (i0 + TILE).min(rows);
                    for j0 in (0..cols).step_by(TILE) {
                        let je = (j0 + TILE).min(cols);
                        for i in i0..ie {
                            let s_row = src_ptr.add(i * cols);
                            for j in j0..je {
                                *dst_ptr.add(j * rows + i) = *s_row.add(j);
                            }
                        }
                    }
                }
            }
            return Tensor::from_aligned(out_shape, dst);
        }

        // ── General fallback: coordinate-based scatter ──
        let out_strides = compute_strides(&out_shape).ok_or_else(|| TensorError::SizeOverflow {
            shape: out_shape.clone(),
        })?;
        let mut out_data = vec![0.0f32; out_count];

        let mut in_coords = vec![0usize; rank];
        for &val in self.data().iter() {
            let mut out_offset = 0usize;
            for (dst_axis, &src_axis) in axes.iter().enumerate() {
                out_offset += in_coords[src_axis] * out_strides[dst_axis];
            }
            out_data[out_offset] = val;
            increment_coords(&mut in_coords, src_shape);
        }

        Tensor::from_vec(out_shape, out_data)
    }

    /// Insert a length-1 dimension at the given axis.
    pub fn unsqueeze(&self, axis: usize) -> Result<Self, TensorError> {
        if axis > self.rank() {
            return Err(TensorError::InvalidAxis {
                axis,
                rank: self.rank() + 1,
            });
        }
        let mut new_shape = self.shape().to_vec();
        new_shape.insert(axis, 1);
        self.reshape(new_shape)
    }

    /// Remove a length-1 dimension at the given axis.
    pub fn squeeze(&self, axis: usize) -> Result<Self, TensorError> {
        if axis >= self.rank() {
            return Err(TensorError::InvalidAxis {
                axis,
                rank: self.rank(),
            });
        }
        if self.shape()[axis] != 1 {
            return Err(TensorError::InvalidAxis {
                axis,
                rank: self.rank(),
            });
        }
        let mut new_shape = self.shape().to_vec();
        new_shape.remove(axis);
        self.reshape(new_shape)
    }

    /// Concatenate tensors along an axis. All tensors must have the same
    /// shape except along the concatenation axis.
    pub fn cat(tensors: &[&Self], axis: usize) -> Result<Self, TensorError> {
        if tensors.is_empty() {
            return Err(TensorError::SizeMismatch {
                shape: vec![],
                data_len: 0,
            });
        }
        let rank = tensors[0].rank();
        if axis >= rank {
            return Err(TensorError::InvalidAxis { axis, rank });
        }
        for t in &tensors[1..] {
            if t.rank() != rank {
                return Err(TensorError::ShapeMismatch {
                    left: tensors[0].shape().to_vec(),
                    right: t.shape().to_vec(),
                });
            }
            for (a, (&d0, &di)) in tensors[0].shape().iter().zip(t.shape().iter()).enumerate() {
                if a != axis && d0 != di {
                    return Err(TensorError::ShapeMismatch {
                        left: tensors[0].shape().to_vec(),
                        right: t.shape().to_vec(),
                    });
                }
            }
        }

        let mut out_shape = tensors[0].shape().to_vec();
        out_shape[axis] = tensors.iter().map(|t| t.shape()[axis]).sum();
        let out_count =
            shape_element_count(&out_shape).ok_or_else(|| TensorError::SizeOverflow {
                shape: out_shape.clone(),
            })?;

        let outer: usize = out_shape[..axis].iter().product();
        let inner: usize = out_shape[axis + 1..].iter().product();

        // Write directly into AlignedVec to avoid the double-copy through
        // Vec -> AlignedVec::from_vec.
        let mut out_data = AlignedVec::<f32>::uninitialized(out_count);

        if inner == 1 && tensors.len() <= 8 {
            // Last-axis concat: write entire output in outer-major order.
            let axis_lens: Vec<usize> = tensors.iter().map(|t| t.shape()[axis]).collect();
            let dst = out_data.as_mut_slice();
            let mut dst_off = 0;
            for o in 0..outer {
                for (ti, t) in tensors.iter().enumerate() {
                    let al = axis_lens[ti];
                    let src_off = o * al;
                    dst[dst_off..dst_off + al].copy_from_slice(&t.data()[src_off..src_off + al]);
                    dst_off += al;
                }
            }
        } else {
            let dst = out_data.as_mut_slice();
            let mut dst_off = 0;
            for o in 0..outer {
                for t in tensors {
                    let t_axis_len = t.shape()[axis];
                    let chunk_len = t_axis_len * inner;
                    let chunk_start = o * chunk_len;
                    dst[dst_off..dst_off + chunk_len]
                        .copy_from_slice(&t.data()[chunk_start..chunk_start + chunk_len]);
                    dst_off += chunk_len;
                }
            }
        }

        Tensor::from_aligned(out_shape, out_data)
    }

    /// Stack tensors along a new axis. All tensors must have identical shapes.
    pub fn stack(tensors: &[&Self], axis: usize) -> Result<Self, TensorError> {
        if tensors.is_empty() {
            return Err(TensorError::SizeMismatch {
                shape: vec![],
                data_len: 0,
            });
        }
        if axis > tensors[0].rank() {
            return Err(TensorError::InvalidAxis {
                axis,
                rank: tensors[0].rank() + 1,
            });
        }
        let expanded: Vec<Self> = tensors
            .iter()
            .map(|t| t.unsqueeze(axis))
            .collect::<Result<_, _>>()?;
        let refs: Vec<&Self> = expanded.iter().collect();
        Self::cat(&refs, axis)
    }

    /// Select a single slice along an axis, removing that axis from the output.
    pub fn select(&self, axis: usize, index: usize) -> Result<Self, TensorError> {
        if axis >= self.rank() {
            return Err(TensorError::InvalidAxis {
                axis,
                rank: self.rank(),
            });
        }
        if index >= self.shape()[axis] {
            return Err(TensorError::IndexOutOfBounds {
                axis,
                index,
                dim: self.shape()[axis],
            });
        }
        let outer: usize = self.shape()[..axis].iter().product();
        let axis_len = self.shape()[axis];
        let inner: usize = self.shape()[axis + 1..].iter().product();

        let mut out_data = Vec::with_capacity(outer * inner);
        for o in 0..outer {
            let base = o * axis_len * inner + index * inner;
            out_data.extend_from_slice(&self.data()[base..base + inner]);
        }

        let mut out_shape = self.shape().to_vec();
        out_shape.remove(axis);
        Tensor::from_vec(out_shape, out_data)
    }

    /// Narrow (slice) along an axis: extract elements `start..start+length`.
    pub fn narrow(&self, axis: usize, start: usize, length: usize) -> Result<Self, TensorError> {
        if axis >= self.rank() {
            return Err(TensorError::InvalidAxis {
                axis,
                rank: self.rank(),
            });
        }
        if start + length > self.shape()[axis] {
            return Err(TensorError::IndexOutOfBounds {
                axis,
                index: start + length,
                dim: self.shape()[axis],
            });
        }
        let outer: usize = self.shape()[..axis].iter().product();
        let axis_len = self.shape()[axis];
        let inner: usize = self.shape()[axis + 1..].iter().product();

        let mut out_data = Vec::with_capacity(outer * length * inner);
        for o in 0..outer {
            let base = o * axis_len * inner + start * inner;
            out_data.extend_from_slice(&self.data()[base..base + length * inner]);
        }

        let mut out_shape = self.shape().to_vec();
        out_shape[axis] = length;
        Tensor::from_vec(out_shape, out_data)
    }

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

    // ── Advanced indexing/selection ─────────────────────────────────────

    /// Element-wise where: `condition ? self : other`.
    /// `condition` has 1.0 for true, 0.0 for false.
    pub fn where_cond(&self, condition: &Self, other: &Self) -> Result<Self, TensorError> {
        if self.shape() != condition.shape() || self.shape() != other.shape() {
            return Err(TensorError::ShapeMismatch {
                left: self.shape().to_vec(),
                right: condition.shape().to_vec(),
            });
        }
        let data: Vec<f32> = condition
            .data()
            .iter()
            .zip(self.data().iter())
            .zip(other.data().iter())
            .map(|((&c, &t), &f)| if c != 0.0 { t } else { f })
            .collect();
        Tensor::from_vec(self.shape().to_vec(), data)
    }

    /// Replace elements where `mask != 0` with `value`.
    pub fn masked_fill(&self, mask: &Self, value: f32) -> Result<Self, TensorError> {
        if self.shape() != mask.shape() {
            return Err(TensorError::ShapeMismatch {
                left: self.shape().to_vec(),
                right: mask.shape().to_vec(),
            });
        }
        let data: Vec<f32> = self
            .data()
            .iter()
            .zip(mask.data().iter())
            .map(|(&v, &m)| if m != 0.0 { value } else { v })
            .collect();
        Tensor::from_vec(self.shape().to_vec(), data)
    }

    /// Scatter values into self along `axis` at positions given by `index`.
    /// `src` provides the values. `index` has same shape as `src`.
    pub fn scatter(&self, axis: usize, index: &Self, src: &Self) -> Result<Self, TensorError> {
        if index.shape() != src.shape() {
            return Err(TensorError::ShapeMismatch {
                left: index.shape().to_vec(),
                right: src.shape().to_vec(),
            });
        }
        if axis >= self.rank() {
            return Err(TensorError::InvalidAxis {
                axis,
                rank: self.rank(),
            });
        }
        let mut out = self.data().to_vec();
        let shape = index.shape();
        let outer: usize = shape[..axis].iter().product();
        let dim = shape[axis];
        let inner: usize = shape[axis + 1..].iter().product();
        let self_dim = self.shape()[axis];
        let self_inner: usize = self.shape()[axis + 1..].iter().product();

        for o in 0..outer {
            for d in 0..dim {
                for i in 0..inner {
                    let src_idx = (o * dim + d) * inner + i;
                    let target_d = index.data()[src_idx] as usize;
                    if target_d < self_dim {
                        let out_idx = (o * self_dim + target_d) * self_inner + i;
                        if out_idx < out.len() {
                            out[out_idx] = src.data()[src_idx];
                        }
                    }
                }
            }
        }
        Tensor::from_vec(self.shape().to_vec(), out)
    }

    /// Gather elements along `axis` at positions given by `index`.
    pub fn gather(&self, axis: usize, index: &Self) -> Result<Self, TensorError> {
        if axis >= self.rank() {
            return Err(TensorError::InvalidAxis {
                axis,
                rank: self.rank(),
            });
        }
        let shape = index.shape();
        let outer: usize = shape[..axis].iter().product();
        let dim = shape[axis];
        let inner: usize = shape[axis + 1..].iter().product();
        let self_dim = self.shape()[axis];
        let self_inner: usize = self.shape()[axis + 1..].iter().product();

        let mut out = vec![0.0f32; index.len()];
        for o in 0..outer {
            for d in 0..dim {
                for i in 0..inner {
                    let idx_pos = (o * dim + d) * inner + i;
                    let src_d = index.data()[idx_pos] as usize;
                    if src_d < self_dim {
                        let src_pos = (o * self_dim + src_d) * self_inner + i;
                        if src_pos < self.len() {
                            out[idx_pos] = self.data()[src_pos];
                        }
                    }
                }
            }
        }
        Tensor::from_vec(shape.to_vec(), out)
    }

    /// Returns the top-k values and their indices along the last axis.
    pub fn topk(&self, k: usize) -> Result<(Self, Self), TensorError> {
        if self.rank() == 0 {
            return Err(TensorError::InvalidAxis { axis: 0, rank: 0 });
        }
        let last_dim = *self.shape().last().expect("non-empty shape");
        let k = k.min(last_dim);
        let outer: usize = self.len() / last_dim;

        let mut values = Vec::with_capacity(outer * k);
        let mut indices = Vec::with_capacity(outer * k);

        for o in 0..outer {
            let start = o * last_dim;
            let slice = &self.data()[start..start + last_dim];
            let mut idx_vec: Vec<usize> = (0..last_dim).collect();
            idx_vec.sort_unstable_by(|&a, &b| {
                slice[b]
                    .partial_cmp(&slice[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            for &i in &idx_vec[..k] {
                values.push(slice[i]);
                indices.push(i as f32);
            }
        }

        let mut out_shape = self.shape().to_vec();
        *out_shape.last_mut().expect("non-empty shape") = k;
        let val_t = Tensor::from_vec(out_shape.clone(), values)?;
        let idx_t = Tensor::from_vec(out_shape, indices)?;
        Ok((val_t, idx_t))
    }

    /// Upper triangular mask: zero below diagonal, keep above.
    /// `diagonal` shifts: 0 = main, positive = above, negative = below.
    pub fn triu(&self, diagonal: i64) -> Result<Self, TensorError> {
        if self.rank() < 2 {
            return Err(TensorError::InvalidAxis {
                axis: 0,
                rank: self.rank(),
            });
        }
        let shape = self.shape();
        let rows = shape[shape.len() - 2];
        let cols = shape[shape.len() - 1];
        let batch: usize = shape[..shape.len() - 2].iter().product();
        let mut out = self.data().to_vec();
        for b in 0..batch {
            for r in 0..rows {
                for c in 0..cols {
                    if (c as i64) < (r as i64) + diagonal {
                        out[b * rows * cols + r * cols + c] = 0.0;
                    }
                }
            }
        }
        Tensor::from_vec(shape.to_vec(), out)
    }

    /// Lower triangular mask: zero above diagonal, keep below.
    pub fn tril(&self, diagonal: i64) -> Result<Self, TensorError> {
        if self.rank() < 2 {
            return Err(TensorError::InvalidAxis {
                axis: 0,
                rank: self.rank(),
            });
        }
        let shape = self.shape();
        let rows = shape[shape.len() - 2];
        let cols = shape[shape.len() - 1];
        let batch: usize = shape[..shape.len() - 2].iter().product();
        let mut out = self.data().to_vec();
        for b in 0..batch {
            for r in 0..rows {
                for c in 0..cols {
                    if (c as i64) > (r as i64) + diagonal {
                        out[b * rows * cols + r * cols + c] = 0.0;
                    }
                }
            }
        }
        Tensor::from_vec(shape.to_vec(), out)
    }

    /// Identity matrix `[n, n]`.
    pub fn eye(n: usize) -> Result<Self, TensorError> {
        let mut data = vec![0.0f32; n * n];
        for i in 0..n {
            data[i * n + i] = 1.0;
        }
        Tensor::from_vec(vec![n, n], data)
    }

    /// Create a diagonal matrix from a 1D vector.
    pub fn diag(vector: &Tensor) -> Result<Self, TensorError> {
        let shape = vector.shape();
        if shape.len() != 1 {
            return Err(TensorError::UnsupportedOperation {
                msg: format!("diag requires a 1D tensor, got shape {:?}", shape),
            });
        }
        let n = shape[0];
        let mut data = vec![0.0f32; n * n];
        for i in 0..n {
            data[i * n + i] = vector.data()[i];
        }
        Self::from_vec(vec![n, n], data)
    }

    /// Extract the diagonal of a 2D matrix as a 1D vector.
    pub fn diag_extract(&self) -> Result<Self, TensorError> {
        let shape = self.shape();
        if shape.len() != 2 {
            return Err(TensorError::UnsupportedOperation {
                msg: format!("diag_extract requires a 2D tensor, got shape {:?}", shape),
            });
        }
        let n = shape[0].min(shape[1]);
        let cols = shape[1];
        let data: Vec<f32> = (0..n).map(|i| self.data()[i * cols + i]).collect();
        Self::from_vec(vec![n], data)
    }

    /// Pad the tensor with a constant value. `padding` is a slice of (before, after) per dimension.
    pub fn pad(&self, padding: &[(usize, usize)], value: f32) -> Result<Self, TensorError> {
        let shape = self.shape();
        if padding.len() != shape.len() {
            return Err(TensorError::InvalidIndexRank {
                expected: shape.len(),
                got: padding.len(),
            });
        }
        let new_shape: Vec<usize> = shape
            .iter()
            .zip(padding)
            .map(|(&s, &(b, a))| s + b + a)
            .collect();
        let new_size: usize = new_shape.iter().product();
        let mut result = vec![value; new_size];
        let ndim = shape.len();

        // Compute strides for both old and new shapes
        let mut old_strides = vec![1usize; ndim];
        for i in (0..ndim.saturating_sub(1)).rev() {
            old_strides[i] = old_strides[i + 1] * shape[i + 1];
        }
        let mut new_strides = vec![1usize; ndim];
        for i in (0..ndim.saturating_sub(1)).rev() {
            new_strides[i] = new_strides[i + 1] * new_shape[i + 1];
        }

        let old_size: usize = shape.iter().product();
        let data = self.data();
        for flat_idx in 0..old_size {
            let mut remaining = flat_idx;
            let mut new_flat = 0;
            for d in 0..ndim {
                let coord = remaining / old_strides[d];
                remaining %= old_strides[d];
                new_flat += (coord + padding[d].0) * new_strides[d];
            }
            result[new_flat] = data[flat_idx];
        }

        Self::from_vec(new_shape, result)
    }

    /// Repeat tensor along each axis by the given counts.
    pub fn repeat(&self, counts: &[usize]) -> Result<Self, TensorError> {
        if counts.len() != self.rank() {
            return Err(TensorError::InvalidIndexRank {
                expected: self.rank(),
                got: counts.len(),
            });
        }
        let mut out = self.clone();
        for (axis, &count) in counts.iter().enumerate() {
            if count > 1 {
                let refs: Vec<&Tensor> = std::iter::repeat_n(&out, count).collect();
                out = Tensor::cat(&refs, axis)?;
            }
        }
        Ok(out)
    }

    // ── Cumulative operations ──────────────────────────────────────────

    /// Cumulative sum along an axis.
    pub fn cumsum(&self, axis: usize) -> Result<Self, TensorError> {
        if axis >= self.rank() {
            return Err(TensorError::InvalidAxis {
                axis,
                rank: self.rank(),
            });
        }
        let shape = self.shape();
        let outer: usize = shape[..axis].iter().product();
        let axis_len = shape[axis];
        let inner: usize = shape[axis + 1..].iter().product();
        let mut out = self.data().to_vec();

        for o in 0..outer {
            for i in 0..inner {
                let mut acc = 0.0f32;
                for d in 0..axis_len {
                    let idx = (o * axis_len + d) * inner + i;
                    acc += out[idx];
                    out[idx] = acc;
                }
            }
        }
        Tensor::from_vec(shape.to_vec(), out)
    }

    /// Cumulative product along an axis.
    pub fn cumprod(&self, axis: usize) -> Result<Self, TensorError> {
        if axis >= self.rank() {
            return Err(TensorError::InvalidAxis {
                axis,
                rank: self.rank(),
            });
        }
        let shape = self.shape();
        let outer: usize = shape[..axis].iter().product();
        let axis_len = shape[axis];
        let inner: usize = shape[axis + 1..].iter().product();
        let mut out = self.data().to_vec();

        for o in 0..outer {
            for i in 0..inner {
                let mut acc = 1.0f32;
                for d in 0..axis_len {
                    let idx = (o * axis_len + d) * inner + i;
                    acc *= out[idx];
                    out[idx] = acc;
                }
            }
        }
        Tensor::from_vec(shape.to_vec(), out)
    }

    // ── FP16 conversion ────────────────────────────────────────────────

    /// Convert all elements to IEEE 754 half-precision (FP16) bytes.
    /// Returns `Vec<u16>` where each u16 is an FP16 bit pattern.
    pub fn to_fp16(&self) -> Vec<u16> {
        self.data().iter().map(|&v| f32_to_fp16(v)).collect()
    }

    /// Create a tensor from FP16 bit patterns.
    pub fn from_fp16(shape: Vec<usize>, fp16_data: &[u16]) -> Result<Self, TensorError> {
        let data: Vec<f32> = fp16_data.iter().map(|&v| fp16_to_f32(v)).collect();
        Tensor::from_vec(shape, data)
    }

    // ── Internal helpers ────────────────────────────────────────────────

    #[allow(unsafe_code)]
    fn unary_op<F>(&self, op: F) -> Self
    where
        F: Fn(f32) -> f32,
    {
        let src = self.data();
        let len = src.len();
        // SAFETY: `uninitialized` allocates without zeroing.  The loop below
        // writes every element before we ever read from `out_data`.
        let mut out_data = AlignedVec::<f32>::uninitialized(len);
        let inp = src.as_ptr();
        let outp = out_data.as_mut_ptr();
        unsafe {
            for i in 0..len {
                *outp.add(i) = op(*inp.add(i));
            }
        }
        Tensor::from_raw_parts(self.shape(), self.strides(), out_data)
    }

    #[allow(unsafe_code)]
    fn unary_simd_op(&self, kind: simd::UnaryKind) -> Self {
        let len = self.len();
        // SAFETY: `uninitialized` allocates without zeroing.  `unary_dispatch`
        // writes every element before we ever read from `out`.
        let mut out = AlignedVec::<f32>::uninitialized(len);
        simd::unary_dispatch(self.data(), &mut out, kind);
        Tensor::from_raw_parts(self.shape(), self.strides(), out)
    }

    fn reduce_axis<F>(&self, axis: usize, init: f32, combine: F) -> Result<Self, TensorError>
    where
        F: Fn(f32, f32) -> f32,
    {
        if axis >= self.rank() {
            return Err(TensorError::InvalidAxis {
                axis,
                rank: self.rank(),
            });
        }

        let mut out_shape = self.shape().to_vec();
        out_shape.remove(axis);
        let out_count =
            shape_element_count(&out_shape).ok_or_else(|| TensorError::SizeOverflow {
                shape: out_shape.clone(),
            })?;
        let out_strides = compute_strides(&out_shape).ok_or_else(|| TensorError::SizeOverflow {
            shape: out_shape.clone(),
        })?;
        let mut out_data = vec![init; out_count];

        let mut in_coords = vec![0usize; self.rank()];
        for input in self.data().iter().copied() {
            let mut out_offset = 0usize;
            for (src_axis, coord) in in_coords.iter().copied().enumerate() {
                if src_axis == axis {
                    continue;
                }
                let dst_axis = if src_axis < axis {
                    src_axis
                } else {
                    src_axis - 1
                };
                if !out_shape.is_empty() {
                    out_offset += coord * out_strides[dst_axis];
                }
            }
            out_data[out_offset] = combine(out_data[out_offset], input);
            increment_coords(&mut in_coords, self.shape());
        }

        Tensor::from_vec(out_shape, out_data)
    }

    /// SIMD fast path for broadcasting a last-dim vector across all rows.
    ///
    /// Matches patterns like `[N, C] + [C]`, `[N, H, W, C] + [1, C]`, etc.
    /// Returns `None` if the pattern doesn't match, so the caller falls through
    /// to the generic broadcast loop.
    #[allow(unsafe_code)]
    fn binary_broadcast_lastdim_simd(
        &self,
        rhs: &Self,
        kind: simd::BinaryKind,
    ) -> Option<Result<Self, TensorError>> {
        let lhs_shape = self.shape();
        let rhs_shape = rhs.shape();

        // Detect: rhs is a 1-D vector whose length equals lhs's last dim,
        // or rhs shape is all-1 except the last dim which matches lhs's last dim.
        let lhs_last = *lhs_shape.last()?;
        if lhs_last == 0 {
            return None;
        }

        let rhs_last = *rhs_shape.last()?;

        // Check rhs is effectively a 1-D vector of size lhs_last:
        // either shape [C] or shape [1, 1, ..., C]
        let rhs_is_lastdim_vec =
            rhs_last == lhs_last && rhs_shape.iter().rev().skip(1).all(|&d| d == 1);
        // Also handle the symmetric case: lhs is the vector, rhs is the big tensor
        let lhs_is_lastdim_vec =
            lhs_last == rhs_last && lhs_shape.iter().rev().skip(1).all(|&d| d == 1);

        // The fast path is safe only when the "row vector" side (all-1
        // except possibly the last dim) ALSO has rank ≤ the other side
        // and its 1-padding doesn't introduce extra dims. Otherwise the
        // broadcast result has a higher rank than the big tensor, and
        // we fall through to the general path so the output shape
        // matches numpy semantics.
        if rhs_is_lastdim_vec && !lhs_is_lastdim_vec {
            // rhs is the row-vector. Safe iff rhs_shape is rank-1 OR
            // rhs's rank is ≤ lhs's rank (otherwise rhs's leading 1s
            // would still expand the broadcast rank past lhs's).
            if rhs_shape.len() > lhs_shape.len() {
                return None;
            }
            let lhs_data = self.data();
            let rhs_data = rhs.data();
            let row_len = lhs_last;
            let num_rows = lhs_data.len() / row_len;
            let mut out_data = AlignedVec::<f32>::uninitialized(lhs_data.len());

            for i in 0..num_rows {
                let start = i * row_len;
                let end = start + row_len;
                simd::binary_dispatch(
                    &lhs_data[start..end],
                    &rhs_data[..row_len],
                    &mut out_data[start..end],
                    kind,
                );
            }

            let out_strides = compute_strides(lhs_shape).expect("valid shape for strides");
            Some(Ok(Tensor::from_raw_parts(
                lhs_shape,
                &out_strides,
                out_data,
            )))
        } else if lhs_is_lastdim_vec && !rhs_is_lastdim_vec {
            if lhs_shape.len() > rhs_shape.len() {
                return None;
            }
            let lhs_data = self.data();
            let rhs_data = rhs.data();
            let row_len = rhs_last;
            let num_rows = rhs_data.len() / row_len;
            let mut out_data = AlignedVec::<f32>::uninitialized(rhs_data.len());

            for i in 0..num_rows {
                let start = i * row_len;
                let end = start + row_len;
                simd::binary_dispatch(
                    &lhs_data[..row_len],
                    &rhs_data[start..end],
                    &mut out_data[start..end],
                    kind,
                );
            }

            let out_strides = compute_strides(rhs_shape).expect("valid shape for strides");
            Some(Ok(Tensor::from_raw_parts(
                rhs_shape,
                &out_strides,
                out_data,
            )))
        } else {
            None
        }
    }

    /// Ternary `where` with NumPy-style broadcasting:
    /// `out[i] = if cond[i] != 0 { x[i] } else { y[i] }`, where each
    /// operand is broadcast to the common shape derived from all three.
    /// Mirrors ONNX's `Where` op (and `np.where` semantics).
    pub fn where_select(cond: &Self, x: &Self, y: &Self) -> Result<Self, TensorError> {
        // Same-shape fast path — no broadcasting work needed.
        if cond.shape() == x.shape() && cond.shape() == y.shape() {
            let cd = cond.data();
            let xd = x.data();
            let yd = y.data();
            let data: Vec<f32> = cd
                .iter()
                .zip(xd.iter().zip(yd.iter()))
                .map(|(&c, (&xv, &yv))| if c != 0.0 { xv } else { yv })
                .collect();
            return Tensor::from_vec(cond.shape().to_vec(), data);
        }
        let cx_shape = broadcast_shape(cond.shape(), x.shape()).ok_or_else(|| {
            TensorError::BroadcastIncompatible {
                left: cond.shape().to_vec(),
                right: x.shape().to_vec(),
            }
        })?;
        let out_shape = broadcast_shape(&cx_shape, y.shape()).ok_or_else(|| {
            TensorError::BroadcastIncompatible {
                left: cx_shape.clone(),
                right: y.shape().to_vec(),
            }
        })?;
        let out_count =
            shape_element_count(&out_shape).ok_or_else(|| TensorError::SizeOverflow {
                shape: out_shape.clone(),
            })?;
        let mut out_data = vec![0.0_f32; out_count];
        let mut coords = vec![0_usize; out_shape.len()];
        let cd = cond.data();
        let xd = x.data();
        let yd = y.data();
        for value in &mut out_data {
            let c_off = broadcast_offset(cond.shape(), cond.strides(), &coords);
            let x_off = broadcast_offset(x.shape(), x.strides(), &coords);
            let y_off = broadcast_offset(y.shape(), y.strides(), &coords);
            *value = if cd[c_off] != 0.0 {
                xd[x_off]
            } else {
                yd[y_off]
            };
            increment_coords(&mut coords, &out_shape);
        }
        Tensor::from_vec(out_shape, out_data)
    }

    fn binary_broadcast_op<F>(&self, rhs: &Self, op: F) -> Result<Self, TensorError>
    where
        F: Fn(f32, f32) -> f32,
    {
        let out_shape = broadcast_shape(self.shape(), rhs.shape()).ok_or_else(|| {
            TensorError::BroadcastIncompatible {
                left: self.shape().to_vec(),
                right: rhs.shape().to_vec(),
            }
        })?;

        let out_count =
            shape_element_count(&out_shape).ok_or_else(|| TensorError::SizeOverflow {
                shape: out_shape.clone(),
            })?;
        let mut out_data = vec![0.0; out_count];
        let mut coords = vec![0usize; out_shape.len()];

        for value in &mut out_data {
            let left_offset = broadcast_offset(self.shape(), self.strides(), &coords);
            let right_offset = broadcast_offset(rhs.shape(), rhs.strides(), &coords);
            *value = op(self.data()[left_offset], rhs.data()[right_offset]);
            increment_coords(&mut coords, &out_shape);
        }

        Tensor::from_vec(out_shape, out_data)
    }

    /// SIMD-accelerated binary op for same-shape tensors (add/sub/mul/div).
    #[allow(unsafe_code)]
    #[allow(unsafe_code)]
    fn binary_same_shape_simd(
        &self,
        rhs: &Self,
        kind: simd::BinaryKind,
    ) -> Result<Self, TensorError> {
        let len = self.len();
        let mut out_data = AlignedVec::<f32>::uninitialized(len);

        // Multi-threaded for large tensors. Cross-platform:
        // macOS → GCD dispatch_apply, others → rayon thread pool.
        if len >= 100_000 {
            let n = std::thread::available_parallelism()
                .map(|p| p.get())
                .unwrap_or(4);
            let chunk = len.div_ceil(n);
            let lp = self.data().as_ptr() as usize;
            let rp = rhs.data().as_ptr() as usize;
            let op = out_data.as_mut_ptr() as usize;

            #[cfg(target_os = "macos")]
            {
                use std::ffi::c_void;
                #[allow(unsafe_code)]
                unsafe extern "C" {
                    fn dispatch_get_global_queue(id: isize, flags: usize) -> *const c_void;
                    fn dispatch_apply_f(
                        n: usize,
                        q: *const c_void,
                        ctx: *mut c_void,
                        work: unsafe extern "C" fn(*mut c_void, usize),
                    );
                }
                struct Ctx {
                    lp: usize,
                    rp: usize,
                    op: usize,
                    chunk: usize,
                    len: usize,
                    kind: simd::BinaryKind,
                }
                let ctx = Ctx {
                    lp,
                    rp,
                    op,
                    chunk,
                    len,
                    kind,
                };
                unsafe extern "C" fn work(raw: *mut c_void, t: usize) {
                    let c = unsafe { &*(raw as *const Ctx) };
                    let start = t * c.chunk;
                    let end = (start + c.chunk).min(c.len);
                    if start >= end {
                        return;
                    }
                    let l = unsafe {
                        std::slice::from_raw_parts((c.lp as *const f32).add(start), end - start)
                    };
                    let r = unsafe {
                        std::slice::from_raw_parts((c.rp as *const f32).add(start), end - start)
                    };
                    let o = unsafe {
                        std::slice::from_raw_parts_mut((c.op as *mut f32).add(start), end - start)
                    };
                    simd::binary_dispatch(l, r, o, c.kind);
                }
                let q = unsafe { dispatch_get_global_queue(0, 0) };
                unsafe {
                    dispatch_apply_f(n, q, &ctx as *const Ctx as *mut c_void, work);
                }
            }

            #[cfg(not(target_os = "macos"))]
            {
                // Rayon global thread pool — threads pre-spawned, ~0.5µs dispatch.
                use rayon::prelude::*;
                (0..n).into_par_iter().for_each(|t| {
                    let start = t * chunk;
                    let end = (start + chunk).min(len);
                    if start >= end {
                        return;
                    }
                    let l = unsafe {
                        std::slice::from_raw_parts((lp as *const f32).add(start), end - start)
                    };
                    let r = unsafe {
                        std::slice::from_raw_parts((rp as *const f32).add(start), end - start)
                    };
                    let o = unsafe {
                        std::slice::from_raw_parts_mut((op as *mut f32).add(start), end - start)
                    };
                    simd::binary_dispatch(l, r, o, kind);
                });
            }

            return Ok(Tensor::from_raw_parts(
                self.shape(),
                self.strides(),
                out_data,
            ));
        }

        simd::binary_dispatch(self.data(), rhs.data(), &mut out_data, kind);
        Ok(Tensor::from_raw_parts(
            self.shape(),
            self.strides(),
            out_data,
        ))
    }

    #[allow(unsafe_code)]
    fn binary_same_shape<F>(&self, rhs: &Self, op: F) -> Result<Self, TensorError>
    where
        F: Fn(f32, f32) -> f32,
    {
        let len = self.len();
        // SAFETY: `uninitialized` allocates without zeroing.  The loop below
        // writes every element before we ever read from `out_data`.
        let mut out_data = AlignedVec::<f32>::uninitialized(len);

        let lhs_ptr = self.data().as_ptr();
        let rhs_ptr = rhs.data().as_ptr();
        let out_ptr = out_data.as_mut_ptr();

        // SAFETY:
        // - Pointers originate from valid slices/vectors of length `len`.
        // - Loop bounds guarantee all pointer arithmetic remains in-bounds.
        // - `out_data` is uniquely mutable and does not alias with input buffers.
        unsafe {
            let mut index = 0usize;
            while index + 4 <= len {
                *out_ptr.add(index) = op(*lhs_ptr.add(index), *rhs_ptr.add(index));
                *out_ptr.add(index + 1) = op(*lhs_ptr.add(index + 1), *rhs_ptr.add(index + 1));
                *out_ptr.add(index + 2) = op(*lhs_ptr.add(index + 2), *rhs_ptr.add(index + 2));
                *out_ptr.add(index + 3) = op(*lhs_ptr.add(index + 3), *rhs_ptr.add(index + 3));
                index += 4;
            }
            while index < len {
                *out_ptr.add(index) = op(*lhs_ptr.add(index), *rhs_ptr.add(index));
                index += 1;
            }
        }

        Ok(Tensor::from_raw_parts(
            self.shape(),
            self.strides(),
            out_data,
        ))
    }

    // ── In-place operations ──────────────────────────────────────────────

    /// In-place ReLU: clamp negative values to zero.
    pub fn relu_inplace(&mut self) {
        simd::relu_inplace_dispatch(self.data_mut());
    }

    /// In-place element-wise add from another tensor (must have same shape).
    pub fn add_inplace(&mut self, rhs: &Self) {
        debug_assert_eq!(self.len(), rhs.len());
        simd::add_inplace_dispatch(self.data_mut(), rhs.data());
    }

    /// In-place add scalar to all elements.
    pub fn add_scalar_inplace(&mut self, s: f32) {
        simd::add_scalar_inplace_dispatch(self.data_mut(), s);
    }

    /// In-place multiply all elements by scalar.
    pub fn mul_scalar_inplace(&mut self, s: f32) {
        simd::mul_scalar_inplace_dispatch(self.data_mut(), s);
    }

    // ── Binary-into operations (write into pre-allocated output) ─────

    /// Element-wise addition writing into a pre-allocated output tensor.
    /// `output`, `lhs`, and `rhs` must all have the same shape.
    pub fn add_into(output: &mut Self, lhs: &Self, rhs: &Self) {
        debug_assert_eq!(lhs.shape(), rhs.shape());
        debug_assert_eq!(lhs.shape(), output.shape());
        simd::binary_dispatch(
            lhs.data(),
            rhs.data(),
            output.data_mut(),
            simd::BinaryKind::Add,
        );
    }

    /// Element-wise subtraction writing into a pre-allocated output tensor.
    /// `output`, `lhs`, and `rhs` must all have the same shape.
    pub fn sub_into(output: &mut Self, lhs: &Self, rhs: &Self) {
        debug_assert_eq!(lhs.shape(), rhs.shape());
        debug_assert_eq!(lhs.shape(), output.shape());
        simd::binary_dispatch(
            lhs.data(),
            rhs.data(),
            output.data_mut(),
            simd::BinaryKind::Sub,
        );
    }

    /// Element-wise multiplication writing into a pre-allocated output tensor.
    /// `output`, `lhs`, and `rhs` must all have the same shape.
    pub fn mul_into(output: &mut Self, lhs: &Self, rhs: &Self) {
        debug_assert_eq!(lhs.shape(), rhs.shape());
        debug_assert_eq!(lhs.shape(), output.shape());
        simd::binary_dispatch(
            lhs.data(),
            rhs.data(),
            output.data_mut(),
            simd::BinaryKind::Mul,
        );
    }
}

// ── atan2 helper ───────────────────────────────────────────────────

/// Scalar atan2(y, x) using Cephes-style range reduction of atan.
///
/// Range-reduces the argument to [0, tan(pi/12)] ≈ [0, 0.268] using:
///   - If z > tan(3*pi/8) ≈ 2.414: atan(z) = pi/2 - atan(1/z)
///   - If z > tan(pi/8) ≈ 0.414: atan(z) = pi/4 + atan((z-1)/(z+1))
///
/// Then uses a degree-7 polynomial on the reduced argument.
/// Max error < 4e-7 across all inputs.
#[allow(clippy::excessive_precision)]
#[inline(always)]
fn fast_atan2_scalar(y: f32, x: f32) -> f32 {
    const PI: f32 = std::f32::consts::PI;
    const FRAC_PI_2: f32 = std::f32::consts::FRAC_PI_2;
    const FRAC_PI_4: f32 = std::f32::consts::FRAC_PI_4;
    const TAN_3PI_8: f32 = 2.414_213_6; // tan(3*pi/8) = 1 + sqrt(2)
    const TAN_PI_8: f32 = 0.414_213_57; // tan(pi/8) = sqrt(2) - 1

    let ax = x.abs();
    let ay = y.abs();

    // Compute atan(|y|/|x|) with range reduction
    let (num, den, swap) = if ax >= ay {
        (ay, ax, false)
    } else {
        (ax, ay, true)
    };
    let z = if den > 0.0 { num / den } else { 0.0 };

    // Range reduction for atan(z), z >= 0
    let (z_red, bias) = if z > TAN_3PI_8 {
        // Should not happen since z <= 1 from our swap, but just in case
        (-1.0 / z, FRAC_PI_2)
    } else if z > TAN_PI_8 {
        ((z - 1.0) / (z + 1.0), FRAC_PI_4)
    } else {
        (z, 0.0)
    };

    // Polynomial: atan(z) ≈ z + z³·P(z²) for small z
    // Coefficients from Cephes atanf.c (S. Moshier), reordered for Horner.
    let z2 = z_red * z_red;
    let p: f32 = 8.054_666e-02;
    let p = p.mul_add(z2, -1.384_895_1e-01);
    let p = p.mul_add(z2, 1.997_075_8e-01);
    let p = p.mul_add(z2, -3.333_129_8e-01);
    let atan_z = z_red.mul_add(z2 * p, z_red) + bias;

    // If we swapped: atan(|y|/|x|) = pi/2 - atan(|x|/|y|)
    let mut result = if swap { FRAC_PI_2 - atan_z } else { atan_z };

    // Quadrant correction
    if x < 0.0 {
        result = PI - result;
    }
    if y < 0.0 {
        result = -result;
    }

    result
}

// ── FP16 conversion utilities ──────────────────────────────────────

/// Convert f32 to IEEE 754 half-precision (FP16) bit pattern.
fn f32_to_fp16(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xFF) as i32;
    let mantissa = bits & 0x007F_FFFF;

    if exponent == 255 {
        // Inf or NaN
        return sign | 0x7C00 | if mantissa != 0 { 0x0200 } else { 0 };
    }

    let unbiased = exponent - 127;
    if unbiased > 15 {
        return sign | 0x7C00; // overflow -> Inf
    }
    if unbiased < -24 {
        return sign; // underflow -> zero
    }
    if unbiased < -14 {
        // subnormal FP16
        let shift = -1 - unbiased;
        let m = (mantissa | 0x0080_0000) >> (shift + 13);
        return sign | m as u16;
    }

    let fp16_exp = ((unbiased + 15) as u16) << 10;
    let fp16_man = (mantissa >> 13) as u16;
    sign | fp16_exp | fp16_man
}

impl Tensor {
    // ── Sort / argsort / unique / nonzero ──────────────────────────────

    /// Sort along `dim`. Returns `(sorted_values, sorted_indices)`.
    ///
    /// If `descending` is true, values are sorted largest-first.
    pub fn sort(&self, dim: usize, descending: bool) -> Result<(Self, Self), TensorError> {
        if dim >= self.rank() {
            return Err(TensorError::InvalidAxis {
                axis: dim,
                rank: self.rank(),
            });
        }
        let shape = self.shape();
        let outer: usize = shape[..dim].iter().product();
        let dim_len = shape[dim];
        let inner: usize = shape[dim + 1..].iter().product();
        let data = self.data();

        let mut out_vals = vec![0.0f32; data.len()];
        let mut out_idxs = vec![0.0f32; data.len()];

        for o in 0..outer {
            for i in 0..inner {
                let mut idx_vec: Vec<usize> = (0..dim_len).collect();
                idx_vec.sort_unstable_by(|&a, &b| {
                    let va = data[(o * dim_len + a) * inner + i];
                    let vb = data[(o * dim_len + b) * inner + i];
                    if descending {
                        vb.partial_cmp(&va).unwrap_or(std::cmp::Ordering::Equal)
                    } else {
                        va.partial_cmp(&vb).unwrap_or(std::cmp::Ordering::Equal)
                    }
                });
                for (rank, &src) in idx_vec.iter().enumerate() {
                    let dst = (o * dim_len + rank) * inner + i;
                    let src_pos = (o * dim_len + src) * inner + i;
                    out_vals[dst] = data[src_pos];
                    out_idxs[dst] = src as f32;
                }
            }
        }

        let v = Tensor::from_vec(shape.to_vec(), out_vals)?;
        let idx = Tensor::from_vec(shape.to_vec(), out_idxs)?;
        Ok((v, idx))
    }

    /// Return indices that would sort along `dim`.
    pub fn argsort(&self, dim: usize, descending: bool) -> Result<Self, TensorError> {
        let (_, indices) = self.sort(dim, descending)?;
        Ok(indices)
    }

    /// Return unique elements (sorted), inverse indices, and counts.
    pub fn unique(&self) -> (Self, Self, Self) {
        let data = self.data();
        let mut sorted: Vec<f32> = data.to_vec();
        sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        sorted.dedup();

        let mut inverse = vec![0.0f32; data.len()];
        let mut counts = vec![0.0f32; sorted.len()];
        for (i, &v) in data.iter().enumerate() {
            let pos = sorted
                .binary_search_by(|probe| {
                    probe.partial_cmp(&v).unwrap_or(std::cmp::Ordering::Equal)
                })
                .expect("value exists in sorted list");
            inverse[i] = pos as f32;
            counts[pos] += 1.0;
        }

        let vals = Tensor::from_vec(vec![sorted.len()], sorted).expect("unique vals");
        let inv = Tensor::from_vec(self.shape().to_vec(), inverse).expect("unique inv");
        let cnt = Tensor::from_vec(vec![counts.len()], counts).expect("unique counts");
        (vals, inv, cnt)
    }

    /// Return coordinates of nonzero elements as a 2D tensor `[N, rank]`.
    pub fn nonzero(&self) -> Self {
        let shape = self.shape();
        let rank = shape.len().max(1);
        let data = self.data();
        let mut coords: Vec<Vec<usize>> = Vec::new();

        if shape.is_empty() {
            // scalar
            if data[0] != 0.0 {
                coords.push(vec![0]);
            }
        } else {
            let mut idx = vec![0usize; shape.len()];
            for pos in 0..data.len() {
                if data[pos] != 0.0 {
                    coords.push(idx.clone());
                }
                // increment multi-dim index
                for d in (0..shape.len()).rev() {
                    idx[d] += 1;
                    if idx[d] < shape[d] {
                        break;
                    }
                    idx[d] = 0;
                }
            }
        }

        let n = coords.len();
        let mut flat = Vec::with_capacity(n * rank);
        for c in &coords {
            for &v in c {
                flat.push(v as f32);
            }
        }
        if n == 0 {
            Tensor::from_vec(vec![0, rank], flat).expect("nonzero empty")
        } else {
            Tensor::from_vec(vec![n, rank], flat).expect("nonzero")
        }
    }

    // ── Flip / roll ────────────────────────────────────────────────────

    /// Reverse elements along the given dimensions.
    pub fn flip(&self, dims: &[usize]) -> Result<Self, TensorError> {
        for &d in dims {
            if d >= self.rank() {
                return Err(TensorError::InvalidAxis {
                    axis: d,
                    rank: self.rank(),
                });
            }
        }
        let shape = self.shape();
        let data = self.data();
        let total = data.len();
        let mut out = vec![0.0f32; total];
        let rank = shape.len();

        let mut src_idx = vec![0usize; rank];
        for pos in 0..total {
            // compute destination index by flipping specified dims
            let mut dst_idx = src_idx.clone();
            for &d in dims {
                dst_idx[d] = shape[d] - 1 - src_idx[d];
            }
            // linear offset
            let mut dst_pos = 0;
            let mut stride = 1;
            for d in (0..rank).rev() {
                dst_pos += dst_idx[d] * stride;
                stride *= shape[d];
            }
            out[dst_pos] = data[pos];

            // increment src_idx
            for d in (0..rank).rev() {
                src_idx[d] += 1;
                if src_idx[d] < shape[d] {
                    break;
                }
                src_idx[d] = 0;
            }
        }
        Tensor::from_vec(shape.to_vec(), out)
    }

    /// Circular shift elements along `dim` by `shift` positions.
    pub fn roll(&self, shift: i64, dim: usize) -> Result<Self, TensorError> {
        if dim >= self.rank() {
            return Err(TensorError::InvalidAxis {
                axis: dim,
                rank: self.rank(),
            });
        }
        let shape = self.shape();
        let outer: usize = shape[..dim].iter().product();
        let dim_len = shape[dim];
        let inner: usize = shape[dim + 1..].iter().product();
        let data = self.data();

        let mut out = vec![0.0f32; data.len()];
        for o in 0..outer {
            for d in 0..dim_len {
                let dst_d = ((d as i64 + shift).rem_euclid(dim_len as i64)) as usize;
                for i in 0..inner {
                    out[(o * dim_len + dst_d) * inner + i] = data[(o * dim_len + d) * inner + i];
                }
            }
        }
        Tensor::from_vec(shape.to_vec(), out)
    }

    // ── Factory: linspace / arange / meshgrid ──────────────────────────

    /// Create a 1-D tensor of `steps` evenly spaced values from `start` to `end` (inclusive).
    pub fn linspace(start: f32, end: f32, steps: usize) -> Result<Self, TensorError> {
        if steps == 0 {
            return Tensor::from_vec(vec![0], vec![]);
        }
        if steps == 1 {
            return Tensor::from_vec(vec![1], vec![start]);
        }
        let step = (end - start) / (steps - 1) as f32;
        let data: Vec<f32> = (0..steps).map(|i| start + step * i as f32).collect();
        Tensor::from_vec(vec![steps], data)
    }

    /// Create a 1-D tensor with values in `[start, end)` with given `step`.
    pub fn arange(start: f32, end: f32, step: f32) -> Result<Self, TensorError> {
        if step == 0.0 {
            return Err(TensorError::ShapeMismatch {
                left: vec![],
                right: vec![],
            });
        }
        let mut data = Vec::new();
        let mut v = start;
        if step > 0.0 {
            while v < end {
                data.push(v);
                v += step;
            }
        } else {
            while v > end {
                data.push(v);
                v += step;
            }
        }
        let n = data.len();
        Tensor::from_vec(vec![n], data)
    }

    /// Create coordinate grids from 1-D tensors (numpy-style `meshgrid` with `indexing='ij'`).
    pub fn meshgrid(tensors: &[Self]) -> Result<Vec<Self>, TensorError> {
        let shape: Vec<usize> = tensors.iter().map(|t| t.len()).collect();
        let total: usize = shape.iter().product();
        let n = tensors.len();
        let mut result = Vec::with_capacity(n);

        for (idx, t) in tensors.iter().enumerate() {
            let t_data = t.data();
            let mut out = vec![0.0f32; total];
            // stride pattern: product of dims after idx
            let inner: usize = shape[idx + 1..].iter().product();
            let outer: usize = shape[..idx].iter().product();
            let dim_len = shape[idx];
            for o in 0..outer {
                for d in 0..dim_len {
                    for i in 0..inner {
                        out[(o * dim_len + d) * inner + i] = t_data[d];
                    }
                }
            }
            result.push(Tensor::from_vec(shape.clone(), out)?);
        }
        Ok(result)
    }

    // ── Advanced indexing extras ────────────────────────────────────────

    /// Select elements where `mask` (f32, nonzero = true) is true, returned as 1-D.
    pub fn boolean_mask(&self, mask: &Self) -> Result<Self, TensorError> {
        if self.shape() != mask.shape() {
            return Err(TensorError::ShapeMismatch {
                left: self.shape().to_vec(),
                right: mask.shape().to_vec(),
            });
        }
        let data = self.data();
        let m = mask.data();
        let out: Vec<f32> = data
            .iter()
            .zip(m.iter())
            .filter(|(_, mv)| **mv != 0.0)
            .map(|(v, _)| *v)
            .collect();
        let n = out.len();
        Tensor::from_vec(vec![n], out)
    }

    /// Select slices along `dim` using integer `indices` tensor (1-D).
    pub fn index_select(&self, dim: usize, indices: &Self) -> Result<Self, TensorError> {
        if dim >= self.rank() {
            return Err(TensorError::InvalidAxis {
                axis: dim,
                rank: self.rank(),
            });
        }
        let shape = self.shape();
        let idx_data = indices.data();
        let n_idx = idx_data.len();
        let outer: usize = shape[..dim].iter().product();
        let dim_len = shape[dim];
        let inner: usize = shape[dim + 1..].iter().product();
        let data = self.data();

        let mut out = Vec::with_capacity(outer * n_idx * inner);
        for o in 0..outer {
            for &idx_f in idx_data {
                let idx = idx_f as usize;
                if idx >= dim_len {
                    return Err(TensorError::IndexOutOfBounds {
                        axis: dim,
                        index: idx,
                        dim: dim_len,
                    });
                }
                let src_start = (o * dim_len + idx) * inner;
                out.extend_from_slice(&data[src_start..src_start + inner]);
            }
        }

        let mut out_shape = shape.to_vec();
        out_shape[dim] = n_idx;
        Tensor::from_vec(out_shape, out)
    }

    // ── Random tensor creation ──────────────────────────────────────────

    /// Create a tensor filled with uniform random values in [0, 1).
    pub fn rand(shape: Vec<usize>, seed: u64) -> Result<Self, TensorError> {
        let n: usize = shape.iter().product();
        let mut rng = seed;
        let data: Vec<f32> = (0..n)
            .map(|_| {
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                (rng as f32) / (u64::MAX as f32)
            })
            .collect();
        Self::from_vec(shape, data)
    }

    /// Create a tensor filled with normally distributed random values (mean=0, std=1).
    /// Uses Box-Muller transform.
    pub fn randn(shape: Vec<usize>, seed: u64) -> Result<Self, TensorError> {
        let n: usize = shape.iter().product();
        let mut rng = seed;
        let mut next_rng = || -> f32 {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            // Map to (0, 1) exclusive to avoid log(0)
            ((rng as f64) / (u64::MAX as f64)).clamp(1e-15, 1.0 - 1e-15) as f32
        };
        let mut data = Vec::with_capacity(n);
        let mut i = 0;
        while i < n {
            let u1 = next_rng();
            let u2 = next_rng();
            let r = (-2.0 * (u1 as f64).ln()).sqrt();
            let theta = 2.0 * std::f64::consts::PI * u2 as f64;
            data.push((r * theta.cos()) as f32);
            i += 1;
            if i < n {
                data.push((r * theta.sin()) as f32);
                i += 1;
            }
        }
        Self::from_vec(shape, data)
    }

    /// Create a tensor filled with random integers in [low, high).
    pub fn randint(shape: Vec<usize>, low: i64, high: i64, seed: u64) -> Result<Self, TensorError> {
        if high <= low {
            return Err(TensorError::UnsupportedOperation {
                msg: format!("randint requires high > low, got low={low}, high={high}"),
            });
        }
        let range = (high - low) as u64;
        let n: usize = shape.iter().product();
        let mut rng = seed;
        let data: Vec<f32> = (0..n)
            .map(|_| {
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                (low + (rng % range) as i64) as f32
            })
            .collect();
        Self::from_vec(shape, data)
    }

    /// Create a random permutation of integers [0, n).
    pub fn randperm(n: usize, seed: u64) -> Result<Self, TensorError> {
        let mut perm: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let mut rng = seed;
        for i in (1..n).rev() {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            let j = (rng as usize) % (i + 1);
            perm.swap(i, j);
        }
        Self::from_vec(vec![n], perm)
    }
}

// ── Advanced tensor operations ──────────────────────────────────────────

impl Tensor {
    /// Slice with step: extract every `step`-th element along `dim` from `start` to `end`.
    pub fn step_slice(
        &self,
        dim: usize,
        start: usize,
        end: usize,
        step: usize,
    ) -> Result<Self, TensorError> {
        let rank = self.rank();
        if dim >= rank {
            return Err(TensorError::InvalidAxis { axis: dim, rank });
        }
        if step == 0 {
            return Err(TensorError::UnsupportedOperation {
                msg: "step must be > 0".to_string(),
            });
        }
        let shape = self.shape();
        let dim_len = shape[dim];
        let end = end.min(dim_len);
        if start >= end {
            // empty along this dim
            let mut out_shape = shape.to_vec();
            out_shape[dim] = 0;
            return Tensor::from_vec(out_shape, vec![]);
        }

        let selected_indices: Vec<usize> = (start..end).step_by(step).collect();
        let new_dim = selected_indices.len();

        let outer: usize = shape[..dim].iter().product();
        let inner: usize = shape[dim + 1..].iter().product();
        let data = self.data();

        let mut out = Vec::with_capacity(outer * new_dim * inner);
        for o in 0..outer {
            for &idx in &selected_indices {
                let src_start = (o * dim_len + idx) * inner;
                out.extend_from_slice(&data[src_start..src_start + inner]);
            }
        }

        let mut out_shape = shape.to_vec();
        out_shape[dim] = new_dim;
        Tensor::from_vec(out_shape, out)
    }

    /// Einstein summation for common patterns.
    ///
    /// Supported equations:
    /// - `"ij,jk->ik"` — matrix multiply
    /// - `"ij->ji"` — transpose
    /// - `"ii->i"` — diagonal
    /// - `"ij->i"` — row sum
    /// - `"ij->j"` — column sum
    /// - `"ij->"` — total sum
    /// - `"i,i->"` — dot product
    /// - `"ij,ij->"` — Frobenius inner product
    pub fn einsum(equation: &str, tensors: &[&Tensor]) -> Result<Tensor, TensorError> {
        let equation = equation.replace(' ', "");
        let parts: Vec<&str> = equation.split("->").collect();
        if parts.len() != 2 {
            return Err(TensorError::UnsupportedOperation {
                msg: format!("invalid einsum equation: {equation}"),
            });
        }
        let inputs_str = parts[0];
        let output_str = parts[1];
        let input_parts: Vec<&str> = inputs_str.split(',').collect();

        if input_parts.len() != tensors.len() {
            return Err(TensorError::UnsupportedOperation {
                msg: format!(
                    "einsum equation has {} inputs but {} tensors provided",
                    input_parts.len(),
                    tensors.len()
                ),
            });
        }

        // Match known patterns
        let pattern = format!(
            "{}->{}",
            input_parts
                .iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>()
                .join(","),
            output_str
        );

        match pattern.as_str() {
            // matrix multiply: ij,jk->ik
            "ij,jk->ik" => {
                let a = tensors[0];
                let b = tensors[1];
                if a.rank() != 2 || b.rank() != 2 {
                    return Err(TensorError::UnsupportedOperation {
                        msg: "ij,jk->ik requires 2D tensors".to_string(),
                    });
                }
                let (m, k1) = (a.shape()[0], a.shape()[1]);
                let (k2, n) = (b.shape()[0], b.shape()[1]);
                if k1 != k2 {
                    return Err(TensorError::ShapeMismatch {
                        left: a.shape().to_vec(),
                        right: b.shape().to_vec(),
                    });
                }
                let ad = a.data();
                let bd = b.data();
                let mut out = vec![0.0f32; m * n];
                for i in 0..m {
                    for j in 0..n {
                        let mut sum = 0.0f32;
                        for p in 0..k1 {
                            sum += ad[i * k1 + p] * bd[p * n + j];
                        }
                        out[i * n + j] = sum;
                    }
                }
                Tensor::from_vec(vec![m, n], out)
            }
            // transpose: ij->ji
            "ij->ji" => {
                let a = tensors[0];
                if a.rank() != 2 {
                    return Err(TensorError::UnsupportedOperation {
                        msg: "ij->ji requires a 2D tensor".to_string(),
                    });
                }
                let (rows, cols) = (a.shape()[0], a.shape()[1]);
                let ad = a.data();
                let mut out = vec![0.0f32; rows * cols];
                for i in 0..rows {
                    for j in 0..cols {
                        out[j * rows + i] = ad[i * cols + j];
                    }
                }
                Tensor::from_vec(vec![cols, rows], out)
            }
            // diagonal: ii->i
            "ii->i" => {
                let a = tensors[0];
                if a.rank() != 2 || a.shape()[0] != a.shape()[1] {
                    return Err(TensorError::UnsupportedOperation {
                        msg: "ii->i requires a square 2D tensor".to_string(),
                    });
                }
                let n = a.shape()[0];
                let ad = a.data();
                let out: Vec<f32> = (0..n).map(|i| ad[i * n + i]).collect();
                Tensor::from_vec(vec![n], out)
            }
            // row sum: ij->i
            "ij->i" => {
                let a = tensors[0];
                if a.rank() != 2 {
                    return Err(TensorError::UnsupportedOperation {
                        msg: "ij->i requires a 2D tensor".to_string(),
                    });
                }
                let (rows, cols) = (a.shape()[0], a.shape()[1]);
                let ad = a.data();
                let out: Vec<f32> = (0..rows)
                    .map(|i| ad[i * cols..(i + 1) * cols].iter().sum())
                    .collect();
                Tensor::from_vec(vec![rows], out)
            }
            // column sum: ij->j
            "ij->j" => {
                let a = tensors[0];
                if a.rank() != 2 {
                    return Err(TensorError::UnsupportedOperation {
                        msg: "ij->j requires a 2D tensor".to_string(),
                    });
                }
                let (rows, cols) = (a.shape()[0], a.shape()[1]);
                let ad = a.data();
                let mut out = vec![0.0f32; cols];
                for i in 0..rows {
                    for j in 0..cols {
                        out[j] += ad[i * cols + j];
                    }
                }
                Tensor::from_vec(vec![cols], out)
            }
            // total sum: ij->
            "ij->" => {
                let a = tensors[0];
                if a.rank() != 2 {
                    return Err(TensorError::UnsupportedOperation {
                        msg: "ij-> requires a 2D tensor".to_string(),
                    });
                }
                let sum: f32 = a.data().iter().sum();
                Ok(Tensor::scalar(sum))
            }
            // dot product: i,i->
            "i,i->" => {
                let a = tensors[0];
                let b = tensors[1];
                if a.rank() != 1 || b.rank() != 1 {
                    return Err(TensorError::UnsupportedOperation {
                        msg: "i,i-> requires 1D tensors".to_string(),
                    });
                }
                if a.shape()[0] != b.shape()[0] {
                    return Err(TensorError::ShapeMismatch {
                        left: a.shape().to_vec(),
                        right: b.shape().to_vec(),
                    });
                }
                let sum: f32 = a
                    .data()
                    .iter()
                    .zip(b.data().iter())
                    .map(|(x, y)| x * y)
                    .sum();
                Ok(Tensor::scalar(sum))
            }
            // Frobenius inner product: ij,ij->
            "ij,ij->" => {
                let a = tensors[0];
                let b = tensors[1];
                if a.rank() != 2 || b.rank() != 2 {
                    return Err(TensorError::UnsupportedOperation {
                        msg: "ij,ij-> requires 2D tensors".to_string(),
                    });
                }
                if a.shape() != b.shape() {
                    return Err(TensorError::ShapeMismatch {
                        left: a.shape().to_vec(),
                        right: b.shape().to_vec(),
                    });
                }
                let sum: f32 = a
                    .data()
                    .iter()
                    .zip(b.data().iter())
                    .map(|(x, y)| x * y)
                    .sum();
                Ok(Tensor::scalar(sum))
            }
            _ => Err(TensorError::UnsupportedOperation {
                msg: format!("unsupported einsum pattern: {pattern}"),
            }),
        }
    }

    // ── Chunk ───────────────────────────────────────────────────────────

    /// Split tensor into `n_chunks` pieces along `axis`. Last chunk may be smaller.
    pub fn chunk(&self, n_chunks: usize, axis: usize) -> Result<Vec<Self>, TensorError> {
        if axis >= self.rank() {
            return Err(TensorError::InvalidAxis {
                axis,
                rank: self.rank(),
            });
        }
        if n_chunks == 0 {
            return Err(TensorError::UnsupportedOperation {
                msg: "n_chunks must be > 0".to_string(),
            });
        }
        let dim = self.shape()[axis];
        let chunk_size = dim.div_ceil(n_chunks); // ceil division
        let mut chunks = Vec::new();
        let mut start = 0;
        while start < dim {
            let length = chunk_size.min(dim - start);
            chunks.push(self.narrow(axis, start, length)?);
            start += length;
        }
        Ok(chunks)
    }

    // ── Histogram ───────────────────────────────────────────────────────

    /// Counts elements in each bin, returns 1D tensor of shape `[bins]`.
    /// Bins are evenly spaced between `min` and `max`.
    pub fn histogram(&self, bins: usize, min: f32, max: f32) -> Result<Self, TensorError> {
        let mut counts = vec![0.0f32; bins];
        let range = max - min;
        for &v in self.data() {
            if v >= min && v <= max {
                let idx = if v == max {
                    bins - 1
                } else {
                    ((v - min) / range * bins as f32) as usize
                };
                counts[idx] += 1.0;
            }
        }
        Tensor::from_vec(vec![bins], counts)
    }

    // ── Bincount ────────────────────────────────────────────────────────

    /// Treats values as integer indices, counts occurrences.
    /// Returns 1D tensor of shape `[num_bins]`.
    pub fn bincount(&self, num_bins: usize) -> Result<Self, TensorError> {
        let mut counts = vec![0.0f32; num_bins];
        for &v in self.data() {
            let idx = v as usize;
            if idx < num_bins {
                counts[idx] += 1.0;
            }
        }
        Tensor::from_vec(vec![num_bins], counts)
    }

    // ── Scalar convenience ──────────────────────────────────────────────

    /// Returns the single scalar value if tensor has exactly one element.
    /// Errors if tensor has more than one element.
    pub fn item(&self) -> Result<f32, TensorError> {
        if self.len() != 1 {
            return Err(TensorError::ShapeMismatch {
                left: vec![1],
                right: self.shape().to_vec(),
            });
        }
        Ok(self.data()[0])
    }

    /// Returns true if tensor has exactly one element.
    pub fn is_scalar(&self) -> bool {
        self.len() == 1
    }

    // ── Scatter Add ──────────────────────────────────────────────────

    /// Like `scatter` but adds instead of replacing values.
    ///
    /// For `dim=1`: `self[i][index[i][j][k]][k] += src[i][j][k]`
    pub fn scatter_add(&self, dim: usize, index: &Self, src: &Self) -> Result<Self, TensorError> {
        if dim >= self.rank() {
            return Err(TensorError::InvalidAxis {
                axis: dim,
                rank: self.rank(),
            });
        }
        if index.rank() != self.rank() {
            return Err(TensorError::InvalidIndexRank {
                expected: self.rank(),
                got: index.rank(),
            });
        }
        if src.shape() != index.shape() {
            return Err(TensorError::ShapeMismatch {
                left: src.shape().to_vec(),
                right: index.shape().to_vec(),
            });
        }

        let self_shape = self.shape();
        let idx_shape = index.shape();
        let idx_data = index.data();
        let src_data = src.data();
        let ndim = self.rank();

        let mut out = self.data().to_vec();
        let mut coords = vec![0usize; ndim];

        for pos in 0..index.len() {
            let idx_val = idx_data[pos] as usize;
            if idx_val >= self_shape[dim] {
                return Err(TensorError::IndexOutOfBounds {
                    axis: dim,
                    index: idx_val,
                    dim: self_shape[dim],
                });
            }

            let mut dst_offset = 0;
            for d in 0..ndim {
                let c = if d == dim { idx_val } else { coords[d] };
                dst_offset += c * self.strides()[d];
            }
            out[dst_offset] += src_data[pos];

            increment_coords(&mut coords, idx_shape);
        }

        Tensor::from_vec(self_shape.to_vec(), out)
    }
}

/// Convert IEEE 754 half-precision (FP16) bit pattern to f32.
fn fp16_to_f32(half: u16) -> f32 {
    let sign = ((half & 0x8000) as u32) << 16;
    let exponent = (half >> 10) & 0x1F;
    let mantissa = (half & 0x03FF) as u32;

    if exponent == 0 {
        if mantissa == 0 {
            return f32::from_bits(sign); // zero
        }
        // subnormal
        let mut m = mantissa;
        let mut e = 0i32;
        while m & 0x0400 == 0 {
            m <<= 1;
            e += 1;
        }
        m &= 0x03FF;
        let f32_exp = ((127 - 15 - e) as u32) << 23;
        let f32_man = m << 13;
        return f32::from_bits(sign | f32_exp | f32_man);
    }
    if exponent == 31 {
        let f32_exp = 0xFF << 23;
        let f32_man = mantissa << 13;
        return f32::from_bits(sign | f32_exp | f32_man);
    }

    let f32_exp = ((exponent as i32 - 15 + 127) as u32 & 0xFF) << 23;
    let f32_man = mantissa << 13;
    f32::from_bits(sign | f32_exp | f32_man)
}
