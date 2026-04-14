use proptest::prelude::*;
use yscv_tensor::Tensor;

use super::ops::rope::apply_rotary_embedding;
use super::ops::sigmoid_slice_dispatch;
use super::ops::simd::sigmoid_scalar;
use super::ops::{exp_slice_dispatch, quantize, relu_slice_dispatch, tanh_slice_dispatch};

// Verify SIMD-dispatched sigmoid matches the scalar reference for arbitrary f32 slices.
proptest! {
    #[test]
    fn sigmoid_dispatch_matches_scalar(
        data in proptest::collection::vec(-20.0f32..20.0, 1..=256)
    ) {
        let mut simd_out = vec![0.0f32; data.len()];
        sigmoid_slice_dispatch(&data, &mut simd_out);

        for (i, &v) in data.iter().enumerate() {
            let expected = sigmoid_scalar(v);
            let actual = simd_out[i];
            // SIMD fast-exp approximations have ~5% relative error in tails
            let tol = expected.abs() * 0.06 + 1e-6;
            prop_assert!(
                (expected - actual).abs() < tol,
                "mismatch at index {}: input={}, expected={}, got={}", i, v, expected, actual
            );
        }
    }

    #[test]
    fn sigmoid_output_in_zero_one_range(
        data in proptest::collection::vec(prop::num::f32::NORMAL, 1..=128)
    ) {
        let mut out = vec![0.0f32; data.len()];
        sigmoid_slice_dispatch(&data, &mut out);

        for (i, &val) in out.iter().enumerate() {
            prop_assert!(
                (0.0..=1.0).contains(&val),
                "sigmoid output out of [0,1] at index {i}: got {val} for input {}",
                data[i]
            );
        }
    }

    // ── ReLU idempotent ────────────────────────────────────────────────
    #[test]
    fn relu_is_idempotent(
        data in proptest::collection::vec(-100.0f32..100.0, 1..=256)
    ) {
        let mut first = data.clone();
        relu_slice_dispatch(&mut first);
        let mut second = first.clone();
        relu_slice_dispatch(&mut second);
        for (i, (&a, &b)) in first.iter().zip(second.iter()).enumerate() {
            prop_assert_eq!(a, b, "relu idempotent violation at index {}", i);
        }
    }

    // ── Tanh range ─────────────────────────────────────────────────────
    #[test]
    fn tanh_output_in_range(
        data in proptest::collection::vec(-50.0f32..50.0, 1..=256)
    ) {
        let mut out = vec![0.0f32; data.len()];
        tanh_slice_dispatch(&data, &mut out);
        for (i, &val) in out.iter().enumerate() {
            prop_assert!(
                (-1.0..=1.0).contains(&val),
                "tanh output out of [-1,1] at index {i}: got {val} for input {}",
                data[i]
            );
        }
    }

    // ── Exp positive ───────────────────────────────────────────────────
    #[test]
    fn exp_output_is_positive(
        data in proptest::collection::vec(-20.0f32..20.0, 1..=256)
    ) {
        let mut out = vec![0.0f32; data.len()];
        exp_slice_dispatch(&data, &mut out);
        for (i, &val) in out.iter().enumerate() {
            prop_assert!(
                val > 0.0,
                "exp output not positive at index {i}: got {val} for input {}",
                data[i]
            );
        }
    }

    // ── Matmul dimensions ──────────────────────────────────────────────
    #[test]
    fn matmul_output_dimensions(
        m in 1usize..=16,
        k in 1usize..=16,
        n in 1usize..=16,
    ) {
        let a_data: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.01).collect();
        let b_data: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.01).collect();
        let a = Tensor::from_vec(vec![m, k], a_data).expect("valid tensor");
        let b = Tensor::from_vec(vec![k, n], b_data).expect("valid tensor");
        let c = super::matmul_2d_sequential(&a, &b).expect("matmul");
        prop_assert_eq!(c.shape(), &[m, n], "matmul output shape mismatch");
    }

    // ── INT4 quantize-dequant error bound ──────────────────────────────
    #[test]
    fn int4_quantize_dequant_error_bound(
        data in proptest::collection::vec(-3.0f32..3.0, 2..=64),
        scale in 0.01f32..2.0,
    ) {
        let even_len = data.len() & !1; // ensure even length
        let data = &data[..even_len];
        let packed_len = even_len / 2;
        let mut packed = vec![0u8; packed_len];
        quantize::quantize_f32_to_int4(data, scale, 0, &mut packed);
        let mut recovered = vec![0.0f32; packed_len * 2];
        quantize::dequantize_int4_to_f32(&packed, scale, 0, &mut recovered);
        for (i, (&orig, &rec)) in data.iter().zip(recovered.iter()).enumerate() {
            let err = (orig - rec).abs();
            // Error bound: quantization rounds to nearest int4, max error = scale * 7.5
            // (clamping to [-8,7]), but within representable range error <= scale/2 + epsilon
            let bound = scale * 0.5 + scale * 0.01;
            // Values outside representable range can have larger error from clamping
            let clamped_orig = (orig / scale).round().clamp(-8.0, 7.0) * scale;
            let clamp_err = (orig - clamped_orig).abs();
            let total_bound = bound + clamp_err;
            prop_assert!(
                err <= total_bound + 1e-5,
                "int4 roundtrip error too large at {i}: orig={orig}, rec={rec}, err={err}, bound={total_bound}"
            );
        }
    }

    // ── (h) RoPE preserves norm ────────────────────────────────────────
    #[test]
    fn rope_preserves_norm(
        data in proptest::collection::vec(-10.0f32..10.0, 16)
    ) {
        // 8 pairs, d_head=16, 1 head for Q and K, seq_len=1
        let mut q = data.clone();
        let mut k = data.clone();

        let norm_before_q: f32 = q.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_before_k: f32 = k.iter().map(|x| x * x).sum::<f32>().sqrt();

        apply_rotary_embedding(
            &mut q,
            &mut k,
            1,     // num_q_heads
            1,     // num_kv_heads
            16,    // d_head
            1,     // seq_len
            0,     // seq_offset
            10000.0, // theta
        );

        let norm_after_q: f32 = q.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_after_k: f32 = k.iter().map(|x| x * x).sum::<f32>().sqrt();

        // RoPE applies rotation, which preserves L2 norm
        prop_assert!(
            (norm_before_q - norm_after_q).abs() < 1e-4,
            "RoPE changed Q norm: before={norm_before_q}, after={norm_after_q}"
        );
        prop_assert!(
            (norm_before_k - norm_after_k).abs() < 1e-4,
            "RoPE changed K norm: before={norm_before_k}, after={norm_after_k}"
        );
    }

    // ── (i) Matmul associative with scalar ─────────────────────────────
    #[test]
    fn matmul_associative_with_scalar(
        a_data in proptest::collection::vec(-5.0f32..5.0, 6),  // 2x3
        b_data in proptest::collection::vec(-5.0f32..5.0, 6),  // 3x2
        s in -5.0f32..5.0,
    ) {
        let a = Tensor::from_vec(vec![2, 3], a_data.clone()).expect("valid A");
        let b = Tensor::from_vec(vec![3, 2], b_data.clone()).expect("valid B");

        // LHS: matmul(scale(A, s), B)
        let a_scaled = a.scale(s);
        let lhs = super::matmul_2d_sequential(&a_scaled, &b).expect("matmul lhs");

        // RHS: scale(matmul(A, B), s)
        let ab = super::matmul_2d_sequential(&a, &b).expect("matmul rhs");
        let rhs = ab.scale(s);

        prop_assert_eq!(lhs.shape(), rhs.shape(), "shape mismatch");
        for (i, (&l, &r)) in lhs.data().iter().zip(rhs.data().iter()).enumerate() {
            let tol = l.abs().max(r.abs()) * 1e-4 + 1e-5;
            prop_assert!(
                (l - r).abs() < tol,
                "matmul scalar associativity violation at {i}: lhs={l}, rhs={r}"
            );
        }
    }
}
