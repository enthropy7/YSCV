//! Advanced Tensor methods: einsum, sort-free reshapes, histogram/bincount,
//! chunking, scatter-add, and scalar extraction.

use super::super::error::TensorError;
use super::super::shape::increment_coords;
use super::super::tensor::Tensor;

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
