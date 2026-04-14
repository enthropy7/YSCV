// # Safety contract
//
// All `unsafe` blocks use SIMD intrinsics gated on target-feature
// detection (`is_aarch64_feature_detected!("neon")` /
// `is_x86_feature_detected!("sse2")`).

#![allow(unsafe_code)]

/// Apply Rotary Position Embedding (RoPE) in-place to query and key tensors.
///
/// Each head dimension is treated as pairs of floats `(x0, x1)` which are
/// rotated by `θ_i * pos` where `θ_i = 1 / theta^(2i / d_head)` and `pos`
/// is the sequence position. This is the standard GPT-NeoX / LLaMA RoPE.
///
/// # Arguments
///
/// * `q` — query tensor, shape `[seq_len, num_q_heads, d_head]`, flattened
/// * `k` — key tensor, shape `[seq_len, num_kv_heads, d_head]`, flattened
/// * `num_q_heads` — number of query heads
/// * `num_kv_heads` — number of key/value heads
/// * `d_head` — dimension per head (must be even)
/// * `seq_len` — sequence length
/// * `seq_offset` — position offset for KV-cache continuation
/// * `theta` — base frequency (10000.0 for LLaMA, 500000.0 for LLaMA 3.1)
pub fn apply_rotary_embedding(
    q: &mut [f32],
    k: &mut [f32],
    num_q_heads: usize,
    num_kv_heads: usize,
    d_head: usize,
    seq_len: usize,
    seq_offset: usize,
    theta: f32,
) {
    debug_assert_eq!(d_head % 2, 0, "d_head must be even for RoPE");
    debug_assert_eq!(q.len(), seq_len * num_q_heads * d_head);
    debug_assert_eq!(k.len(), seq_len * num_kv_heads * d_head);

    // Pre-compute inverse frequency table: θ_i = 1 / theta^(2i / d_head)
    let half = d_head / 2;
    let inv_freq: Vec<f32> = (0..half)
        .map(|i| 1.0 / theta.powf(2.0 * i as f32 / d_head as f32))
        .collect();

    // Apply rotation to Q
    for pos in 0..seq_len {
        let abs_pos = (pos + seq_offset) as f32;
        let q_base = pos * num_q_heads * d_head;
        for head in 0..num_q_heads {
            let offset = q_base + head * d_head;
            rotate_half_inplace(&mut q[offset..offset + d_head], &inv_freq, abs_pos);
        }
    }

    // Apply rotation to K
    for pos in 0..seq_len {
        let abs_pos = (pos + seq_offset) as f32;
        let k_base = pos * num_kv_heads * d_head;
        for head in 0..num_kv_heads {
            let offset = k_base + head * d_head;
            rotate_half_inplace(&mut k[offset..offset + d_head], &inv_freq, abs_pos);
        }
    }
}

/// Rotate pairs `(x0, x1)` by angle `freq * pos` for each frequency.
#[inline]
fn rotate_half_inplace(data: &mut [f32], inv_freq: &[f32], pos: f32) {
    let half = inv_freq.len();

    #[cfg(target_arch = "aarch64")]
    if std::arch::is_aarch64_feature_detected!("neon") {
        // SAFETY: NEON is available (feature detection guard).
        let done = unsafe { rotate_half_neon(data, inv_freq, pos, half) };
        if done >= half {
            return;
        }
        // Scalar tail
        for i in done..half {
            let angle = inv_freq[i] * pos;
            let (sin_a, cos_a) = angle.sin_cos();
            let x0 = data[i];
            let x1 = data[i + half];
            data[i] = x0 * cos_a - x1 * sin_a;
            data[i + half] = x0 * sin_a + x1 * cos_a;
        }
        return;
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("sse2") {
        // SAFETY: SSE2 is available (feature detection guard).
        let done = unsafe { rotate_half_sse2(data, inv_freq, pos, half) };
        if done >= half {
            return;
        }
        for i in done..half {
            let angle = inv_freq[i] * pos;
            let (sin_a, cos_a) = angle.sin_cos();
            let x0 = data[i];
            let x1 = data[i + half];
            data[i] = x0 * cos_a - x1 * sin_a;
            data[i + half] = x0 * sin_a + x1 * cos_a;
        }
        return;
    }

    // Scalar fallback
    for i in 0..half {
        let angle = inv_freq[i] * pos;
        let (sin_a, cos_a) = angle.sin_cos();
        let x0 = data[i];
        let x1 = data[i + half];
        data[i] = x0 * cos_a - x1 * sin_a;
        data[i + half] = x0 * sin_a + x1 * cos_a;
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn rotate_half_neon(data: &mut [f32], inv_freq: &[f32], pos: f32, half: usize) -> usize {
    use std::arch::aarch64::*;
    // SAFETY: NEON guaranteed by caller's feature detection. Pointer
    // arithmetic stays within `data[0..d_head]` and `inv_freq[0..half]`.
    unsafe {
        let mut i = 0;
        while i + 4 <= half {
            let mut sin_arr = [0.0f32; 4];
            let mut cos_arr = [0.0f32; 4];
            for j in 0..4 {
                let (s, c) = (inv_freq[i + j] * pos).sin_cos();
                sin_arr[j] = s;
                cos_arr[j] = c;
            }
            let sin_v = vld1q_f32(sin_arr.as_ptr());
            let cos_v = vld1q_f32(cos_arr.as_ptr());
            let x0 = vld1q_f32(data.as_ptr().add(i));
            let x1 = vld1q_f32(data.as_ptr().add(i + half));
            let out0 = vsubq_f32(vmulq_f32(x0, cos_v), vmulq_f32(x1, sin_v));
            let out1 = vaddq_f32(vmulq_f32(x0, sin_v), vmulq_f32(x1, cos_v));
            vst1q_f32(data.as_mut_ptr().add(i), out0);
            vst1q_f32(data.as_mut_ptr().add(i + half), out1);
            i += 4;
        }
        i
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn rotate_half_sse2(data: &mut [f32], inv_freq: &[f32], pos: f32, half: usize) -> usize {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    // SAFETY: SSE2 guaranteed by caller's feature detection.
    unsafe {
        let mut i = 0;
        while i + 4 <= half {
            let mut sin_arr = [0.0f32; 4];
            let mut cos_arr = [0.0f32; 4];
            for j in 0..4 {
                let (s, c) = (inv_freq[i + j] * pos).sin_cos();
                sin_arr[j] = s;
                cos_arr[j] = c;
            }
            let sin_v = _mm_loadu_ps(sin_arr.as_ptr());
            let cos_v = _mm_loadu_ps(cos_arr.as_ptr());
            let x0 = _mm_loadu_ps(data.as_ptr().add(i));
            let x1 = _mm_loadu_ps(data.as_ptr().add(i + half));
            let out0 = _mm_sub_ps(_mm_mul_ps(x0, cos_v), _mm_mul_ps(x1, sin_v));
            let out1 = _mm_add_ps(_mm_mul_ps(x0, sin_v), _mm_mul_ps(x1, cos_v));
            _mm_storeu_ps(data.as_mut_ptr().add(i), out0);
            _mm_storeu_ps(data.as_mut_ptr().add(i + half), out1);
            i += 4;
        }
        i
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rope_basic_rotation() {
        let d_head = 8;
        let num_heads = 2;
        let seq_len = 3;
        let theta = 10000.0;

        let mut q = vec![1.0f32; seq_len * num_heads * d_head];
        let mut k = vec![1.0f32; seq_len * num_heads * d_head];

        apply_rotary_embedding(
            &mut q, &mut k, num_heads, num_heads, d_head, seq_len, 0, theta,
        );

        // Position 0 should have angle=0, so cos=1, sin=0 → values unchanged for low freq
        // Just verify no NaN/Inf and values are reasonable
        assert!(q.iter().all(|&v| v.is_finite()));
        assert!(k.iter().all(|&v| v.is_finite()));

        // Position 0 with angle 0 for the first freq should preserve x0
        // First freq: inv_freq[0] = 1/10000^0 = 1.0, angle = 1.0 * 0 = 0
        // cos(0) = 1, sin(0) = 0 → x0 stays 1.0
        assert!((q[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn rope_kv_cache_offset() {
        let d_head = 4;
        let num_heads = 1;
        let theta = 10000.0;

        // Generate full sequence
        let mut q_full = vec![1.0f32; 4 * d_head];
        let mut k_full = vec![1.0f32; 4 * d_head];
        apply_rotary_embedding(
            &mut q_full,
            &mut k_full,
            num_heads,
            num_heads,
            d_head,
            4,
            0,
            theta,
        );

        // Generate last 2 tokens with offset=2 (simulating KV-cache append)
        let mut q_offset = vec![1.0f32; 2 * d_head];
        let mut k_offset = vec![1.0f32; 2 * d_head];
        apply_rotary_embedding(
            &mut q_offset,
            &mut k_offset,
            num_heads,
            num_heads,
            d_head,
            2,
            2,
            theta,
        );

        // Positions 2,3 should match between full and offset
        let stride = num_heads * d_head;
        for i in 0..stride {
            assert!(
                (q_full[2 * stride + i] - q_offset[i]).abs() < 1e-5,
                "q mismatch at pos 2, elem {i}"
            );
            assert!(
                (q_full[3 * stride + i] - q_offset[stride + i]).abs() < 1e-5,
                "q mismatch at pos 3, elem {i}"
            );
        }
    }

    #[test]
    fn rope_orthogonality() {
        // Rotation should preserve vector norm
        let d_head = 16;
        let num_heads = 1;
        let theta = 10000.0;

        let mut q = vec![1.0f32; d_head];
        let original_norm: f32 = q.iter().map(|x| x * x).sum::<f32>().sqrt();

        apply_rotary_embedding(
            &mut q,
            &mut vec![0.0; d_head],
            num_heads,
            num_heads,
            d_head,
            1,
            5,
            theta,
        );

        let rotated_norm: f32 = q.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (original_norm - rotated_norm).abs() < 1e-4,
            "RoPE should preserve norm: {original_norm} vs {rotated_norm}"
        );
    }
}
