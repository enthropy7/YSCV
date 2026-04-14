//! Autoregressive text generation for decoder-only transformer models.
//!
//! Provides a token-by-token generation loop with KV-cache, temperature
//! scaling, and top-k / top-p sampling.

use crate::error::OnnxError;

/// Configuration for autoregressive generation.
pub struct GenerateConfig {
    /// Maximum number of tokens to generate.
    pub max_tokens: usize,
    /// Temperature for logit scaling (1.0 = no change, <1.0 = sharper, >1.0 = flatter).
    pub temperature: f32,
    /// Top-k sampling: keep only the k most probable tokens (0 = disabled).
    pub top_k: usize,
    /// Top-p (nucleus) sampling: keep tokens until cumulative probability exceeds p (1.0 = disabled).
    pub top_p: f32,
    /// End-of-sequence token ID. Generation stops when this token is produced.
    pub eos_token_id: Option<u32>,
    /// Repetition penalty (1.0 = disabled, >1.0 = penalize repeated tokens).
    pub repetition_penalty: f32,
}

impl Default for GenerateConfig {
    fn default() -> Self {
        Self {
            max_tokens: 256,
            temperature: 1.0,
            top_k: 50,
            top_p: 0.9,
            eos_token_id: None,
            repetition_penalty: 1.0,
        }
    }
}

/// Generate tokens autoregressively from a decoder-only transformer ONNX model.
///
/// The model must have:
/// - Input: `"input_ids"` with shape `[1, seq_len]` (i64 or i32)
/// - Output: `"logits"` with shape `[1, seq_len, vocab_size]`
///
/// KV-caching is not yet wired into the ONNX graph (requires model-specific
/// plumbing). This implementation re-runs the full model for each token,
/// which is correct but not optimally efficient. KV-cache integration is a
/// follow-up when ONNX models export explicit cache tensors.
///
/// # Arguments
///
/// * `model` — loaded ONNX model
/// * `input_ids` — initial prompt token IDs
/// * `config` — generation parameters
/// * `run_fn` — function that runs the ONNX model and returns outputs
///
/// # Returns
///
/// Vector of generated token IDs (not including the prompt).
pub fn generate<F>(
    input_ids: &[u32],
    config: &GenerateConfig,
    mut run_fn: F,
) -> Result<Vec<u32>, OnnxError>
where
    F: FnMut(&[u32]) -> Result<Vec<f32>, OnnxError>,
{
    let mut tokens: Vec<u32> = input_ids.to_vec();
    let mut generated: Vec<u32> = Vec::with_capacity(config.max_tokens);
    let mut rng_state: u64 = 0xdeadbeef_u64;

    for _ in 0..config.max_tokens {
        // Run model on current token sequence
        let logits = run_fn(&tokens)?;

        // Take logits for the last position
        let vocab_size = logits.len() / tokens.len();
        let last_logits_start = (tokens.len() - 1) * vocab_size;
        let mut logits_slice = logits[last_logits_start..last_logits_start + vocab_size].to_vec();

        // Apply repetition penalty
        if config.repetition_penalty != 1.0 {
            for &token in &tokens {
                let idx = token as usize;
                if idx < logits_slice.len() {
                    if logits_slice[idx] > 0.0 {
                        logits_slice[idx] /= config.repetition_penalty;
                    } else {
                        logits_slice[idx] *= config.repetition_penalty;
                    }
                }
            }
        }

        // Temperature scaling
        if config.temperature != 1.0 && config.temperature > 0.0 {
            let inv_temp = 1.0 / config.temperature;
            for v in &mut logits_slice {
                *v *= inv_temp;
            }
        }

        // Sample next token
        let next_token = sample_token(
            &mut logits_slice,
            config.top_k,
            config.top_p,
            &mut rng_state,
        );

        // Check for EOS
        if config.eos_token_id == Some(next_token) {
            break;
        }

        generated.push(next_token);
        tokens.push(next_token);
    }

    Ok(generated)
}

/// Sample a token from logits using top-k + top-p filtering.
fn sample_token(logits: &mut [f32], top_k: usize, top_p: f32, rng: &mut u64) -> u32 {
    let vocab_size = logits.len();

    // Top-k: keep only the k highest logits
    if top_k > 0 && top_k < vocab_size {
        // Find the k-th largest value
        let mut sorted_indices: Vec<usize> = (0..vocab_size).collect();
        sorted_indices.sort_unstable_by(|&a, &b| {
            logits[b]
                .partial_cmp(&logits[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let threshold = logits[sorted_indices[top_k - 1]];
        for v in logits.iter_mut() {
            if *v < threshold {
                *v = f32::NEG_INFINITY;
            }
        }
    }

    // Softmax
    let max_val = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let mut sum = 0.0f32;
    for v in logits.iter_mut() {
        *v = (*v - max_val).exp();
        sum += *v;
    }
    if sum > 0.0 {
        let inv = 1.0 / sum;
        for v in logits.iter_mut() {
            *v *= inv;
        }
    }

    // Top-p (nucleus): keep tokens until cumulative probability exceeds p
    if top_p < 1.0 {
        let mut sorted_indices: Vec<usize> = (0..vocab_size).collect();
        sorted_indices.sort_unstable_by(|&a, &b| {
            logits[b]
                .partial_cmp(&logits[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut cumsum = 0.0f32;
        for &idx in &sorted_indices {
            cumsum += logits[idx];
            if cumsum > top_p {
                // Zero out everything after this point
                for &remaining_idx in &sorted_indices
                    [sorted_indices.iter().position(|&x| x == idx).unwrap_or(0) + 1..]
                {
                    logits[remaining_idx] = 0.0;
                }
                break;
            }
        }
        // Re-normalize
        let new_sum: f32 = logits.iter().sum();
        if new_sum > 0.0 {
            let inv = 1.0 / new_sum;
            for v in logits.iter_mut() {
                *v *= inv;
            }
        }
    }

    // Weighted random sampling using xorshift64
    *rng ^= *rng << 13;
    *rng ^= *rng >> 7;
    *rng ^= *rng << 17;
    let u = (*rng as f32) / (u64::MAX as f32);

    let mut cumsum = 0.0f32;
    for (i, &prob) in logits.iter().enumerate() {
        cumsum += prob;
        if cumsum > u {
            return i as u32;
        }
    }

    // Fallback: return the most probable token
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generate_basic() {
        let config = GenerateConfig {
            max_tokens: 5,
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            eos_token_id: None,
            repetition_penalty: 1.0,
        };

        // Mock model: always returns highest logit for token 42
        let mut call_count = 0;
        let result = generate(&[1, 2, 3], &config, |tokens| {
            call_count += 1;
            let vocab_size = 100;
            let seq_len = tokens.len();
            let mut logits = vec![0.0f32; seq_len * vocab_size];
            // Set token 42 as the most probable for the last position
            logits[(seq_len - 1) * vocab_size + 42] = 10.0;
            Ok(logits)
        })
        .expect("generation ok");

        assert_eq!(result.len(), 5);
        // With very peaked logits at token 42, all generated tokens should be 42
        assert!(result.iter().all(|&t| t == 42));
    }

    #[test]
    fn generate_eos_stops() {
        let config = GenerateConfig {
            max_tokens: 100,
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            eos_token_id: Some(0),
            repetition_penalty: 1.0,
        };

        let result = generate(&[1], &config, |tokens| {
            let vocab_size = 10;
            let seq_len = tokens.len();
            let mut logits = vec![0.0f32; seq_len * vocab_size];
            // Return EOS token (0) with high logit
            logits[(seq_len - 1) * vocab_size] = 10.0;
            Ok(logits)
        })
        .expect("generation ok");

        // Should stop immediately at EOS
        assert!(result.is_empty());
    }

    #[test]
    fn sample_token_greedy() {
        // top_k=1 forces greedy selection of the highest logit
        let mut logits = vec![0.0, 0.0, 10.0, 0.0, 0.0];
        let mut rng = 42u64;
        let token = sample_token(&mut logits, 1, 1.0, &mut rng);
        assert_eq!(token, 2); // token 2 has the highest logit
    }
}
