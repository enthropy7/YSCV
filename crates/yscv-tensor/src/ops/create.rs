//! Tensor creation + ordering ops: sort/argsort/unique/nonzero, flip/roll,
//! linspace/arange/meshgrid, boolean_mask/index_select, and the RNG factories.

use super::super::error::TensorError;
use super::super::tensor::Tensor;

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
