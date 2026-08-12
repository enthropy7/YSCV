use yscv_tensor::{AlignedVec, Tensor};

use super::super::error::KernelError;

/// Looks up embeddings from a weight matrix.
///
/// `weight`: `[vocab_size, embed_dim]`
/// `indices`: `[*]` — flat tensor of integer indices (stored as f32)
///
/// Returns: `[*indices_shape, embed_dim]`
#[allow(unsafe_code)]
pub fn embedding_lookup(weight: &Tensor, indices: &Tensor) -> Result<Tensor, KernelError> {
    let w_shape = weight.shape();
    if w_shape.len() != 2 {
        return Err(KernelError::InvalidEmbeddingWeightRank {
            got_rank: w_shape.len(),
        });
    }
    let vocab_size = w_shape[0];
    let embed_dim = w_shape[1];
    let w_data = weight.data();
    let idx_data = indices.data();

    let num_indices = idx_data.len();
    // SAFETY: embedding lookup writes every output element before use.
    let mut output = unsafe { AlignedVec::<f32>::uninitialized(num_indices * embed_dim) };

    for (i, &idx_f) in idx_data.iter().enumerate() {
        let idx = idx_f as usize;
        if idx >= vocab_size {
            return Err(KernelError::EmbeddingIndexOutOfBounds {
                index: idx,
                vocab_size,
            });
        }
        let src_start = idx * embed_dim;
        let dst_start = i * embed_dim;
        output[dst_start..dst_start + embed_dim]
            .copy_from_slice(&w_data[src_start..src_start + embed_dim]);
    }

    let mut out_shape = indices.shape().to_vec();
    out_shape.push(embed_dim);
    Tensor::from_aligned(out_shape, output).map_err(Into::into)
}

/// Applies dropout: randomly zeroes elements with probability `p`.
///
/// During inference (`training=false`), returns input unchanged.
/// Uses xorshift64 PRNG with given seed for deterministic masking.
/// Surviving elements are scaled by `1 / (1 - p)` (inverted dropout).
#[allow(unsafe_code)]
pub fn dropout(input: &Tensor, p: f32, seed: u64, training: bool) -> Result<Tensor, KernelError> {
    if !training {
        return Ok(input.clone());
    }
    if p <= 0.0 {
        return Ok(input.clone());
    }
    if p >= 1.0 {
        let zeros = AlignedVec::<f32>::calloc(input.len());
        return Tensor::from_aligned(input.shape().to_vec(), zeros).map_err(Into::into);
    }

    let scale = 1.0 / (1.0 - p);
    let in_data = input.data();
    // SAFETY: embedding lookup writes every output element before use.
    let mut output = unsafe { AlignedVec::<f32>::uninitialized(in_data.len()) };

    // Threshold in u64 space: values below this are dropped.
    let threshold = (p as f64 * u64::MAX as f64) as u64;
    let mut state = if seed == 0 { 1 } else { seed };

    for (i, &val) in in_data.iter().enumerate() {
        // xorshift64
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        output[i] = if state < threshold { 0.0 } else { val * scale };
    }

    Tensor::from_aligned(input.shape().to_vec(), output).map_err(Into::into)
}
