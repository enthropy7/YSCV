use yscv_tensor::Tensor;

use crate::error::OnnxError;

/// KV-cache for efficient autoregressive transformer inference.
///
/// Stores the key and value projections for each layer so they don't need
/// to be recomputed when generating subsequent tokens. On each step, only
/// the new token's K/V are appended; the full K/V history is read for
/// attention.
///
/// # Usage
///
/// ```ignore
/// let mut cache = KvCache::new(num_layers, max_seq_len);
/// for token in tokens {
///     let (new_k, new_v) = model.forward_layer(token, &cache);
///     cache.append(layer, &new_k, &new_v)?;
///     let (full_k, full_v) = cache.get(layer);
///     // use full_k, full_v for attention
/// }
/// ```
pub struct KvCache {
    /// Per-layer key cache. Shape of each: `[current_seq_len, num_kv_heads * d_head]`.
    k: Vec<Vec<f32>>,
    /// Per-layer value cache. Same shape as keys.
    v: Vec<Vec<f32>>,
    /// Number of tokens currently cached.
    seq_len: usize,
    /// Maximum sequence length (pre-allocated capacity).
    max_seq_len: usize,
    /// Number of layers.
    num_layers: usize,
    /// KV dimension per token (num_kv_heads * d_head).
    kv_dim: usize,
}

impl KvCache {
    /// Create a new empty KV-cache.
    ///
    /// * `num_layers` — number of transformer layers
    /// * `max_seq_len` — maximum sequence length to pre-allocate
    /// * `kv_dim` — total KV dimension per token (num_kv_heads * d_head)
    pub fn new(num_layers: usize, max_seq_len: usize, kv_dim: usize) -> Self {
        let k = (0..num_layers)
            .map(|_| Vec::with_capacity(max_seq_len * kv_dim))
            .collect();
        let v = (0..num_layers)
            .map(|_| Vec::with_capacity(max_seq_len * kv_dim))
            .collect();
        Self {
            k,
            v,
            seq_len: 0,
            max_seq_len,
            num_layers,
            kv_dim,
        }
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

        self.k[layer].extend_from_slice(new_k.data());
        self.v[layer].extend_from_slice(new_v.data());

        // Update seq_len only from layer 0 to avoid double-counting
        if layer == 0 {
            self.seq_len += new_tokens;
        }

        Ok(())
    }

    /// Get the full cached key and value tensors for a layer.
    ///
    /// Returns `(key, value)` tensors with shape `[seq_len, kv_dim]`.
    pub fn get(&self, layer: usize) -> Result<(Tensor, Tensor), OnnxError> {
        if layer >= self.num_layers {
            return Err(OnnxError::ShapeMismatch {
                detail: format!("KV-cache layer {layer} out of range"),
            });
        }
        let k = Tensor::from_vec(vec![self.seq_len, self.kv_dim], self.k[layer].clone()).map_err(
            |e| OnnxError::ShapeMismatch {
                detail: format!("KV-cache key tensor: {e}"),
            },
        )?;
        let v = Tensor::from_vec(vec![self.seq_len, self.kv_dim], self.v[layer].clone()).map_err(
            |e| OnnxError::ShapeMismatch {
                detail: format!("KV-cache value tensor: {e}"),
            },
        )?;
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
}
