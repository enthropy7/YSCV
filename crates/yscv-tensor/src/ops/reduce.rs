//! Reduction ops: sum/mean/min/max/argmin/argmax/var/std, axis + quantile.

use super::super::error::TensorError;
use super::super::simd;
use super::super::tensor::Tensor;

impl Tensor {
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
}
