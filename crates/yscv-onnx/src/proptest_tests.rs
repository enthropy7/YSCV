use proptest::prelude::*;
use yscv_kernels::{dequantize_int4_to_f32, quantize_f32_to_int4, scaled_dot_product_attention};
use yscv_tensor::Tensor;

use super::KvCache;

proptest! {
    // ── INT4 pack-unpack round-trip ────────────────────────────────────
    #[test]
    fn int4_pack_unpack_roundtrip(
        nibbles in proptest::collection::vec(-8i8..=7, 2..=64usize)
    ) {
        // Ensure even length for clean packing
        let even_len = nibbles.len() & !1;
        let nibbles = &nibbles[..even_len];

        // Pack nibbles manually
        let packed_len = even_len / 2;
        let mut packed = vec![0u8; packed_len];
        nibbles.chunks(2).zip(packed.iter_mut()).for_each(|(pair, out)| {
            let lo = (pair[0] as u8) & 0x0F;
            let hi = (pair[1] as u8) & 0x0F;
            *out = lo | (hi << 4);
        });

        // Unpack with dequant(scale=1, zp=0) to get integer values back
        let mut recovered = vec![0.0f32; even_len];
        dequantize_int4_to_f32(&packed, 1.0, 0, &mut recovered);

        for (i, (&orig, &rec)) in nibbles.iter().zip(recovered.iter()).enumerate() {
            prop_assert!(
                (orig as f32 - rec).abs() < 1e-6,
                "int4 pack-unpack mismatch at {}: orig={}, recovered={}", i, orig, rec
            );
        }
    }

    // ── Quantize-dequant bounded error ─────────────────────────────────
    #[test]
    fn quantize_dequant_bounded_error(
        data in proptest::collection::vec(-3.5f32..3.5, 2..=64),
        scale in 0.05f32..2.0,
    ) {
        let even_len = data.len() & !1;
        let data = &data[..even_len];
        let packed_len = even_len / 2;
        let mut packed = vec![0u8; packed_len];
        quantize_f32_to_int4(data, scale, 0, &mut packed);
        let mut recovered = vec![0.0f32; even_len];
        dequantize_int4_to_f32(&packed, scale, 0, &mut recovered);

        for (i, (&orig, &rec)) in data.iter().zip(recovered.iter()).enumerate() {
            let err = (orig - rec).abs();
            // For values within representable range, error <= scale/2
            // For values outside [-8*scale, 7*scale], clamping adds extra error
            let quantized = (orig / scale).round().clamp(-8.0, 7.0);
            let ideal_rec = quantized * scale;
            let ideal_err = (orig - ideal_rec).abs();
            prop_assert!(
                err <= ideal_err + scale * 0.01 + 1e-5,
                "quantize-dequant error too large at {i}: orig={orig}, rec={rec}, err={err}, ideal_err={ideal_err}"
            );
        }
    }

    // ── (j) GQA output shape via scaled_dot_product_attention ─────────
    #[test]
    fn gqa_output_shape(
        num_q_heads_idx in 0usize..3,   // index into [2, 4, 8]
        kv_ratio_idx in 0usize..3,       // index into [1, 2, 4]
        d_head_idx in 0usize..3,         // index into [4, 8, 16]
        seq_q in 1usize..=8,
        seq_k in 1usize..=8,
    ) {
        let num_q_heads_opts = [2usize, 4, 8];
        let kv_ratio_opts = [1usize, 2, 4];
        let d_head_opts = [4usize, 8, 16];

        let num_q_heads = num_q_heads_opts[num_q_heads_idx];
        let ratio = kv_ratio_opts[kv_ratio_idx];
        let d_head = d_head_opts[d_head_idx];
        let num_kv_heads = num_q_heads / ratio;

        // Must divide evenly
        if !num_q_heads.is_multiple_of(ratio) || num_kv_heads == 0 {
            return Ok(());
        }

        let groups = num_q_heads / num_kv_heads;

        // Simulate GQA by running per-head scaled_dot_product_attention
        // and concatenating results
        let q_total = seq_q * num_q_heads * d_head;
        let k_total = seq_k * num_kv_heads * d_head;
        let v_total = seq_k * num_kv_heads * d_head;

        let q_data: Vec<f32> = (0..q_total).map(|i| (i as f32 * 0.01) % 1.0).collect();
        let k_data: Vec<f32> = (0..k_total).map(|i| (i as f32 * 0.02) % 1.0).collect();
        let v_data: Vec<f32> = (0..v_total).map(|i| (i as f32 * 0.03) % 1.0).collect();

        // Run one head at a time through scaled_dot_product_attention
        // Each head: Q[seq_q, d_head], K[seq_k, d_head], V[seq_k, d_head]
        let mut out_parts: Vec<Vec<f32>> = Vec::new();

        for qh in 0..num_q_heads {
            let kv_h = qh / groups;

            // Extract Q slice for this head
            let mut q_head = Vec::with_capacity(seq_q * d_head);
            for sq in 0..seq_q {
                let base = sq * num_q_heads * d_head + qh * d_head;
                q_head.extend_from_slice(&q_data[base..base + d_head]);
            }

            // Extract K slice for this KV head
            let mut k_head = Vec::with_capacity(seq_k * d_head);
            for sk in 0..seq_k {
                let base = sk * num_kv_heads * d_head + kv_h * d_head;
                k_head.extend_from_slice(&k_data[base..base + d_head]);
            }

            // Extract V slice for this KV head
            let mut v_head = Vec::with_capacity(seq_k * d_head);
            for sk in 0..seq_k {
                let base = sk * num_kv_heads * d_head + kv_h * d_head;
                v_head.extend_from_slice(&v_data[base..base + d_head]);
            }

            let q_t = Tensor::from_vec(vec![seq_q, d_head], q_head).expect("Q tensor");
            let k_t = Tensor::from_vec(vec![seq_k, d_head], k_head).expect("K tensor");
            let v_t = Tensor::from_vec(vec![seq_k, d_head], v_head).expect("V tensor");

            let attn = scaled_dot_product_attention(&q_t, &k_t, &v_t, None)
                .expect("attention");

            prop_assert_eq!(
                attn.shape(),
                &[seq_q, d_head],
                "per-head attention output shape wrong for head {}", qh
            );
            out_parts.push(attn.data().to_vec());
        }

        // Concatenated output shape: [seq_q, num_q_heads * d_head]
        let total_out = out_parts.iter().map(|p| p.len()).sum::<usize>();
        prop_assert_eq!(
            total_out,
            seq_q * num_q_heads * d_head,
            "total GQA output size mismatch"
        );
    }

    // ── (k) KV-cache roundtrip ────────────────────────────────────────
    #[test]
    fn kv_cache_roundtrip(
        data in proptest::collection::vec(-10.0f32..10.0, 12) // 3 tokens * 4 dim
    ) {
        let kv_dim = 4;
        let num_tokens = 3;
        let mut cache = KvCache::new(2, 100, kv_dim);

        let k_data = data[..num_tokens * kv_dim].to_vec();
        let v_data = data[..num_tokens * kv_dim].to_vec();

        let k_tensor = Tensor::from_vec(
            vec![num_tokens, kv_dim], k_data.clone()
        ).expect("valid k");
        let v_tensor = Tensor::from_vec(
            vec![num_tokens, kv_dim], v_data.clone()
        ).expect("valid v");

        cache.append(0, &k_tensor, &v_tensor).expect("append ok");
        prop_assert_eq!(cache.seq_len(), num_tokens);

        let (got_k, got_v) = cache.get(0).expect("get ok");
        prop_assert_eq!(got_k.shape(), &[num_tokens, kv_dim]);
        prop_assert_eq!(got_v.shape(), &[num_tokens, kv_dim]);

        for (i, (&expected, &actual)) in k_data.iter().zip(got_k.data().iter()).enumerate() {
            prop_assert!(
                (expected - actual).abs() < 1e-6,
                "KV-cache key data mismatch at {i}: expected={expected}, got={actual}"
            );
        }
        for (i, (&expected, &actual)) in v_data.iter().zip(got_v.data().iter()).enumerate() {
            prop_assert!(
                (expected - actual).abs() < 1e-6,
                "KV-cache value data mismatch at {i}: expected={expected}, got={actual}"
            );
        }
    }
}
