use super::aligned::AlignedVec;
use super::error::TensorError;
use super::shape::{
    broadcast_offset, broadcast_shape, compute_strides, increment_coords, shape_element_count,
};
use super::simd;
use super::tensor::Tensor;

impl Tensor {
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
pub(super) fn fast_atan2_scalar(y: f32, x: f32) -> f32 {
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

mod compare;
mod create;
mod elementwise;
mod indexing;
mod reduce;
mod shape;

mod advanced;

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
