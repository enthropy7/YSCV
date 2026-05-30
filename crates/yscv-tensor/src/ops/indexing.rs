//! Advanced indexing/selection: where/masked_fill/scatter/gather/topk/tri/pad.

use super::super::error::TensorError;
use super::super::tensor::Tensor;

impl Tensor {
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
}
