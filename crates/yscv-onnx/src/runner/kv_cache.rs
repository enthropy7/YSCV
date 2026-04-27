use yscv_tensor::Tensor;

use crate::error::OnnxError;

/// Storage dtype for the KV cache.
///
/// `F32` keeps full fp32 fidelity (default; matches every existing call
/// site). `I8` stores keys and values as per-row symmetric int8 with
/// one fp32 scale per row (token × layer); the cache pays `kv_dim` bytes
/// instead of `kv_dim * 4` plus a small per-row scale-table overhead.
/// On long contexts that's a 4× memory and bandwidth saving on the
/// dominant footprint.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum KvDtype {
    #[default]
    F32,
    I8,
}

enum LayerStorage {
    F32(Vec<f32>),
    I8 {
        data: Vec<i8>,
        /// One fp32 scale per row (= per token).
        scales: Vec<f32>,
    },
}

impl LayerStorage {
    fn new(dtype: KvDtype, kv_dim: usize, max_seq_len: usize) -> Self {
        match dtype {
            KvDtype::F32 => LayerStorage::F32(Vec::with_capacity(max_seq_len * kv_dim)),
            KvDtype::I8 => LayerStorage::I8 {
                data: Vec::with_capacity(max_seq_len * kv_dim),
                scales: Vec::with_capacity(max_seq_len),
            },
        }
    }

    /// Append `rows` worth of new tokens. `data` length must be
    /// `rows * kv_dim` fp32 values, row-major.
    fn append(&mut self, src: &[f32], rows: usize, kv_dim: usize) {
        match self {
            LayerStorage::F32(v) => v.extend_from_slice(src),
            LayerStorage::I8 { data, scales } => {
                for r in 0..rows {
                    let row = &src[r * kv_dim..(r + 1) * kv_dim];
                    let abs_max = row.iter().fold(0.0_f32, |m, &v| m.max(v.abs()));
                    let scale = if abs_max <= f32::EPSILON {
                        1.0
                    } else {
                        abs_max / 127.0
                    };
                    let inv = 1.0 / scale;
                    scales.push(scale);
                    for &v in row {
                        let q = (v * inv).round().clamp(-127.0, 127.0) as i8;
                        data.push(q);
                    }
                }
            }
        }
    }

    fn dequantise_to_vec(&self, kv_dim: usize) -> Vec<f32> {
        match self {
            LayerStorage::F32(v) => v.clone(),
            LayerStorage::I8 { data, scales } => {
                let rows = scales.len();
                let mut out = Vec::with_capacity(rows * kv_dim);
                for r in 0..rows {
                    let scale = scales[r];
                    let row = &data[r * kv_dim..(r + 1) * kv_dim];
                    out.extend(row.iter().map(|&q| (q as f32) * scale));
                }
                out
            }
        }
    }

    fn clear(&mut self) {
        match self {
            LayerStorage::F32(v) => v.clear(),
            LayerStorage::I8 { data, scales } => {
                data.clear();
                scales.clear();
            }
        }
    }
}

/// KV-cache for efficient autoregressive transformer inference.
///
/// Stores the key and value projections for each layer so they don't need
/// to be recomputed when generating subsequent tokens. On each step, only
/// the new token's K/V are appended; the full K/V history is read for
/// attention.
///
/// # Storage dtype
///
/// `KvDtype::F32` (default) keeps fp32 fidelity. `KvDtype::I8` stores
/// per-row symmetric int8 with a fp32 scale per token — 4× less memory
/// and bandwidth on long contexts. Reads dequantise on the fly.
///
/// # Usage
///
/// ```ignore
/// let mut cache = KvCache::new(num_layers, max_seq_len, kv_dim);
/// // or, opt in to int8 storage:
/// let mut cache = KvCache::with_dtype(num_layers, max_seq_len, kv_dim, KvDtype::I8);
/// for token in tokens {
///     let (new_k, new_v) = model.forward_layer(token, &cache);
///     cache.append(layer, &new_k, &new_v)?;
///     let (full_k, full_v) = cache.get(layer);
///     // use full_k, full_v for attention
/// }
/// ```
pub struct KvCache {
    /// Per-layer key cache. Each layer holds `current_seq_len * kv_dim`
    /// values (plus per-row scales when dtype is I8).
    k: Vec<LayerStorage>,
    /// Per-layer value cache. Same shape as keys.
    v: Vec<LayerStorage>,
    /// Number of tokens currently cached.
    seq_len: usize,
    /// Maximum sequence length (pre-allocated capacity).
    max_seq_len: usize,
    /// Number of layers.
    num_layers: usize,
    /// KV dimension per token (num_kv_heads * d_head).
    kv_dim: usize,
    dtype: KvDtype,
}

impl KvCache {
    /// Create a new empty KV-cache with the default fp32 storage.
    ///
    /// * `num_layers` — number of transformer layers
    /// * `max_seq_len` — maximum sequence length to pre-allocate
    /// * `kv_dim` — total KV dimension per token (num_kv_heads * d_head)
    pub fn new(num_layers: usize, max_seq_len: usize, kv_dim: usize) -> Self {
        Self::with_dtype(num_layers, max_seq_len, kv_dim, KvDtype::F32)
    }

    /// Create a new empty KV-cache with the given storage dtype.
    pub fn with_dtype(
        num_layers: usize,
        max_seq_len: usize,
        kv_dim: usize,
        dtype: KvDtype,
    ) -> Self {
        let k = (0..num_layers)
            .map(|_| LayerStorage::new(dtype, kv_dim, max_seq_len))
            .collect();
        let v = (0..num_layers)
            .map(|_| LayerStorage::new(dtype, kv_dim, max_seq_len))
            .collect();
        Self {
            k,
            v,
            seq_len: 0,
            max_seq_len,
            num_layers,
            kv_dim,
            dtype,
        }
    }

    /// Active storage dtype.
    pub fn dtype(&self) -> KvDtype {
        self.dtype
    }

    /// Current cached sequence length.
    pub fn seq_len(&self) -> usize {
        self.seq_len
    }

    /// Append new key/value projections for one or more tokens at a given layer.
    ///
    /// * `layer` — transformer layer index
    /// * `new_k` — key tensor for new tokens, shape `[num_new_tokens, kv_dim]`
    /// * `new_v` — value tensor for new tokens, same shape
    pub fn append(
        &mut self,
        layer: usize,
        new_k: &Tensor,
        new_v: &Tensor,
    ) -> Result<(), OnnxError> {
        if layer >= self.num_layers {
            return Err(OnnxError::ShapeMismatch {
                detail: format!(
                    "KV-cache layer {layer} out of range (max {})",
                    self.num_layers
                ),
            });
        }
        let new_tokens = new_k.shape()[0];
        if self.seq_len + new_tokens > self.max_seq_len {
            return Err(OnnxError::ShapeMismatch {
                detail: format!(
                    "KV-cache overflow: {} + {} > {}",
                    self.seq_len, new_tokens, self.max_seq_len
                ),
            });
        }

        self.k[layer].append(new_k.data(), new_tokens, self.kv_dim);
        self.v[layer].append(new_v.data(), new_tokens, self.kv_dim);

        // Update seq_len only from layer 0 to avoid double-counting
        if layer == 0 {
            self.seq_len += new_tokens;
        }

        Ok(())
    }

    /// Get the full cached key and value tensors for a layer.
    ///
    /// Returns `(key, value)` tensors with shape `[seq_len, kv_dim]`. For
    /// int8 storage the values are dequantised on the fly.
    pub fn get(&self, layer: usize) -> Result<(Tensor, Tensor), OnnxError> {
        if layer >= self.num_layers {
            return Err(OnnxError::ShapeMismatch {
                detail: format!("KV-cache layer {layer} out of range"),
            });
        }
        let k_data = self.k[layer].dequantise_to_vec(self.kv_dim);
        let v_data = self.v[layer].dequantise_to_vec(self.kv_dim);
        let k = Tensor::from_vec(vec![self.seq_len, self.kv_dim], k_data).map_err(|e| {
            OnnxError::ShapeMismatch {
                detail: format!("KV-cache key tensor: {e}"),
            }
        })?;
        let v = Tensor::from_vec(vec![self.seq_len, self.kv_dim], v_data).map_err(|e| {
            OnnxError::ShapeMismatch {
                detail: format!("KV-cache value tensor: {e}"),
            }
        })?;
        Ok((k, v))
    }

    /// Reset the cache (for a new generation session).
    pub fn clear(&mut self) {
        for layer_k in &mut self.k {
            layer_k.clear();
        }
        for layer_v in &mut self.v {
            layer_v.clear();
        }
        self.seq_len = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kv_cache_append_and_get() {
        let mut cache = KvCache::new(2, 10, 4);
        assert_eq!(cache.seq_len(), 0);
        assert_eq!(cache.dtype(), KvDtype::F32);

        // Append 3 tokens to layer 0
        let k0 = Tensor::from_vec(vec![3, 4], vec![1.0; 12]).expect("valid");
        let v0 = Tensor::from_vec(vec![3, 4], vec![2.0; 12]).expect("valid");
        cache.append(0, &k0, &v0).expect("append ok");
        assert_eq!(cache.seq_len(), 3);

        // Get from layer 0
        let (got_k, got_v) = cache.get(0).expect("get ok");
        assert_eq!(got_k.shape(), &[3, 4]);
        assert_eq!(got_v.shape(), &[3, 4]);
        assert!((got_k.data()[0] - 1.0).abs() < 1e-6);
        assert!((got_v.data()[0] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn kv_cache_incremental_append() {
        let mut cache = KvCache::new(1, 10, 2);

        // Append 2 tokens
        let k1 = Tensor::from_vec(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).expect("valid");
        let v1 = Tensor::from_vec(vec![2, 2], vec![5.0, 6.0, 7.0, 8.0]).expect("valid");
        cache.append(0, &k1, &v1).expect("ok");

        // Append 1 more token
        let k2 = Tensor::from_vec(vec![1, 2], vec![9.0, 10.0]).expect("valid");
        let v2 = Tensor::from_vec(vec![1, 2], vec![11.0, 12.0]).expect("valid");
        cache.append(0, &k2, &v2).expect("ok");

        assert_eq!(cache.seq_len(), 3);
        let (k, v) = cache.get(0).expect("get ok");
        assert_eq!(k.shape(), &[3, 2]);
        assert!((k.data()[4] - 9.0).abs() < 1e-6);
        assert!((v.data()[4] - 11.0).abs() < 1e-6);
    }

    #[test]
    fn kv_cache_overflow_error() {
        let mut cache = KvCache::new(1, 2, 2);
        let k = Tensor::from_vec(vec![3, 2], vec![0.0; 6]).expect("valid");
        let v = Tensor::from_vec(vec![3, 2], vec![0.0; 6]).expect("valid");
        assert!(cache.append(0, &k, &v).is_err());
    }

    #[test]
    fn kv_cache_clear() {
        let mut cache = KvCache::new(1, 10, 2);
        let k = Tensor::from_vec(vec![2, 2], vec![1.0; 4]).expect("valid");
        let v = Tensor::from_vec(vec![2, 2], vec![1.0; 4]).expect("valid");
        cache.append(0, &k, &v).expect("ok");
        assert_eq!(cache.seq_len(), 2);
        cache.clear();
        assert_eq!(cache.seq_len(), 0);
    }

    #[test]
    fn kv_cache_int8_round_trip_within_quantisation_step() {
        let mut cache = KvCache::with_dtype(1, 10, 8, KvDtype::I8);
        assert_eq!(cache.dtype(), KvDtype::I8);
        let k_data = vec![
            -3.0, -2.5, -1.0, 0.0, 0.5, 1.5, 2.0, 2.8, // row 0
            0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, // row 1
        ];
        let v_data = vec![
            10.0, 20.0, -30.0, 40.0, -50.0, 60.0, -70.0, 80.0, // row 0
            1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 4.0, -4.0, // row 1
        ];
        let k = Tensor::from_vec(vec![2, 8], k_data.clone()).unwrap();
        let v = Tensor::from_vec(vec![2, 8], v_data.clone()).unwrap();
        cache.append(0, &k, &v).unwrap();
        assert_eq!(cache.seq_len(), 2);

        let (got_k, got_v) = cache.get(0).unwrap();
        assert_eq!(got_k.shape(), &[2, 8]);
        // Per-row max-abs / 127 ≈ 0.022 (row 0 K), 0.0063 (row 1 K),
        // 0.63 (row 0 V), 0.031 (row 1 V). Per-element error bounded by
        // scale/2 since rounding is to nearest.
        for (i, (g, e)) in got_k.data().iter().zip(k_data.iter()).enumerate() {
            let row = i / 8;
            let scale = if row == 0 { 3.0 / 127.0 } else { 0.8 / 127.0 };
            assert!(
                (g - e).abs() <= scale,
                "k idx {i}: got {g} expected {e} (scale {scale})"
            );
        }
        for (i, (g, e)) in got_v.data().iter().zip(v_data.iter()).enumerate() {
            let row = i / 8;
            let scale = if row == 0 { 80.0 / 127.0 } else { 4.0 / 127.0 };
            assert!(
                (g - e).abs() <= scale,
                "v idx {i}: got {g} expected {e} (scale {scale})"
            );
        }
    }

    #[test]
    fn kv_cache_int8_clear_resets_scales() {
        let mut cache = KvCache::with_dtype(1, 10, 4, KvDtype::I8);
        let k = Tensor::from_vec(vec![2, 4], vec![1.0; 8]).unwrap();
        let v = Tensor::from_vec(vec![2, 4], vec![1.0; 8]).unwrap();
        cache.append(0, &k, &v).unwrap();
        cache.clear();
        assert_eq!(cache.seq_len(), 0);
        // Re-append after clear — must work without stale scale rows.
        let k2 = Tensor::from_vec(vec![1, 4], vec![5.0, 5.0, 5.0, 5.0]).unwrap();
        let v2 = Tensor::from_vec(vec![1, 4], vec![5.0, 5.0, 5.0, 5.0]).unwrap();
        cache.append(0, &k2, &v2).unwrap();
        let (got_k, _) = cache.get(0).unwrap();
        assert_eq!(got_k.shape(), &[1, 4]);
        assert!((got_k.data()[0] - 5.0).abs() < 0.05);
    }
}
