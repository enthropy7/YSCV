use yscv_tensor::{Tensor, TensorError};

use crate::{
    Backend, CpuBackend, KernelError, ParallelElementwiseConfig, add, add_out, add_with_config,
    exp, exp_with_config, gelu, mish, mul, mul_out, mul_with_config, relu, relu_out_with_config,
    relu_with_config, sigmoid, sigmoid_with_config, silu, silu_with_config, sub, sub_out,
    sub_with_config, tanh_act, tanh_act_with_config,
};

use super::{assert_slice_close, build_tensor};

#[test]
fn relu_clamps_negative_values() {
    let input = Tensor::from_vec(vec![4], vec![-1.0, 0.0, 1.5, -3.5]).unwrap();
    let out = relu(&input);
    assert_eq!(out.shape(), &[4]);
    assert_eq!(out.data(), &[0.0, 0.0, 1.5, 0.0]);
}

#[test]
fn relu_with_config_disabled_matches_relu() {
    let input = build_tensor(&[128, 64, 3], 0.41);
    let baseline = relu(&input);
    let disabled = relu_with_config(&input, ParallelElementwiseConfig::disabled());
    assert_eq!(baseline, disabled);
}

#[test]
fn sigmoid_produces_expected_values() {
    let input = Tensor::from_vec(vec![5], vec![-6.0, -1.0, 0.0, 1.0, 6.0]).unwrap();
    let out = sigmoid(&input);
    assert_eq!(out.shape(), &[5]);
    assert_slice_close(
        out.data(),
        &[0.002472623, 0.26894143, 0.5, 0.7310586, 0.99752736],
        0.035, // Schraudolph bit-trick exp has ~4% max error
    );
}

#[test]
// Miri's software FP emulation is non-deterministic: two calls to f32::exp()
// with the same input can return different bit patterns. Confirmed by standalone
// test: `let a = 0.33f32.exp(); let b = 0.33f32.exp(); assert_eq!(a.to_bits(), b.to_bits())`
// fails under Miri. This is a known Miri limitation, not a bug in yscv.
// See: https://github.com/rust-lang/miri — FP semantics are approximate.
#[cfg_attr(miri, ignore)]
fn sigmoid_with_config_disabled_matches_sigmoid() {
    let input = build_tensor(&[128, 64, 3], 0.19);
    let baseline = sigmoid(&input);
    let disabled = sigmoid_with_config(&input, ParallelElementwiseConfig::disabled());
    assert_eq!(baseline, disabled);
}

#[test]
fn elementwise_with_config_matches_fallback_math() {
    let lhs = build_tensor(&[64, 32], 0.11);
    let rhs = build_tensor(&[64, 32], 0.52);

    let add_reference = add(&lhs, &rhs).unwrap();
    let add_config = add_with_config(&lhs, &rhs, ParallelElementwiseConfig::disabled()).unwrap();
    assert_eq!(add_reference, add_config);

    let sub_reference = sub(&lhs, &rhs).unwrap();
    let sub_config = sub_with_config(&lhs, &rhs, ParallelElementwiseConfig::disabled()).unwrap();
    assert_eq!(sub_reference, sub_config);

    let mul_reference = mul(&lhs, &rhs).unwrap();
    let mul_config = mul_with_config(&lhs, &rhs, ParallelElementwiseConfig::disabled()).unwrap();
    assert_eq!(mul_reference, mul_config);
}

#[test]
fn binary_out_matches_allocating_path() {
    let lhs = build_tensor(&[64, 32], 0.11);
    let rhs = build_tensor(&[64, 32], 0.52);
    let mut output = Tensor::zeros(vec![64, 32]).unwrap();

    add_out(&lhs, &rhs, &mut output).unwrap();
    assert_eq!(output, add(&lhs, &rhs).unwrap());

    sub_out(&lhs, &rhs, &mut output).unwrap();
    assert_eq!(output, sub(&lhs, &rhs).unwrap());

    mul_out(&lhs, &rhs, &mut output).unwrap();
    assert_eq!(output, mul(&lhs, &rhs).unwrap());
}

#[test]
fn binary_out_rejects_shape_mismatch() {
    let lhs = build_tensor(&[2, 3], 0.11);
    let rhs = build_tensor(&[2, 3], 0.52);
    let mut output = Tensor::zeros(vec![3, 2]).unwrap();
    let err = add_out(&lhs, &rhs, &mut output).unwrap_err();
    assert!(matches!(
        err,
        KernelError::Tensor(TensorError::ShapeMismatch { .. })
    ));
}

#[test]
fn relu_out_with_config_matches_allocating_path() {
    let input = build_tensor(&[128, 64, 3], 0.41);
    let mut output = Tensor::zeros(input.shape().to_vec()).unwrap();
    relu_out_with_config(&input, &mut output, ParallelElementwiseConfig::default()).unwrap();
    assert_eq!(output, relu(&input));
}

#[test]
fn backend_add_supports_broadcasting() {
    let backend = CpuBackend;
    let lhs = Tensor::from_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let rhs = Tensor::from_vec(vec![3], vec![10.0, 20.0, 30.0]).unwrap();

    let out = backend.add(&lhs, &rhs).unwrap();
    assert_eq!(out.shape(), &[2, 3]);
    assert_eq!(out.data(), &[11.0, 22.0, 33.0, 14.0, 25.0, 36.0]);
}

#[test]
fn elementwise_lastdim_broadcast_uses_fast_path_shapes() {
    let lhs = Tensor::from_vec(
        vec![2, 2, 3],
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
    )
    .unwrap();
    let rhs = Tensor::from_vec(vec![1, 3], vec![10.0, 20.0, 30.0]).unwrap();
    let parallel = ParallelElementwiseConfig {
        min_parallel_elements: 1,
    };

    let out = add_with_config(&lhs, &rhs, parallel).unwrap();
    assert_eq!(out.shape(), &[2, 2, 3]);
    assert_eq!(
        out.data(),
        &[
            11.0, 22.0, 33.0, 14.0, 25.0, 36.0, 17.0, 28.0, 39.0, 20.0, 31.0, 42.0
        ]
    );
}

#[test]
fn elementwise_lastdim_broadcast_preserves_lhs_rhs_order_for_sub() {
    let lhs = Tensor::from_vec(vec![3], vec![10.0, 20.0, 30.0]).unwrap();
    let rhs = Tensor::from_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let out = sub_with_config(&lhs, &rhs, ParallelElementwiseConfig::disabled()).unwrap();

    assert_eq!(out.shape(), &[2, 3]);
    assert_eq!(out.data(), &[9.0, 18.0, 27.0, 6.0, 15.0, 24.0]);
}

#[test]
fn free_add_matches_backend_add() {
    let backend = CpuBackend;
    let lhs = Tensor::from_vec(vec![2], vec![1.0, 2.0]).unwrap();
    let rhs = Tensor::from_vec(vec![2], vec![3.0, 4.0]).unwrap();

    let direct = add(&lhs, &rhs).unwrap();
    let via_backend = backend.add(&lhs, &rhs).unwrap();
    assert_eq!(direct, via_backend);
}

#[test]
fn free_sub_and_mul_match_backend() {
    let backend = CpuBackend;
    let lhs = Tensor::from_vec(vec![2], vec![10.0, 5.0]).unwrap();
    let rhs = Tensor::from_vec(vec![2], vec![3.0, 4.0]).unwrap();

    let sub_direct = sub(&lhs, &rhs).unwrap();
    let sub_backend = backend.sub(&lhs, &rhs).unwrap();
    assert_eq!(sub_direct, sub_backend);
    assert_eq!(sub_direct.data(), &[7.0, 1.0]);

    let mul_direct = mul(&lhs, &rhs).unwrap();
    let mul_backend = backend.mul(&lhs, &rhs).unwrap();
    assert_eq!(mul_direct, mul_backend);
    assert_eq!(mul_direct.data(), &[30.0, 20.0]);
}

#[test]
fn gelu_produces_expected_values() {
    let input = Tensor::from_vec(vec![5], vec![-2.0, -1.0, 0.0, 1.0, 2.0]).unwrap();
    let out = gelu(&input);
    assert_eq!(out.shape(), &[5]);
    // GELU(0) = 0
    assert!((out.data()[2] - 0.0).abs() < 1e-6);
    // GELU(x) approx x for large positive x
    assert!((out.data()[4] - 2.0).abs() < 0.1);
    // GELU(x) approx 0 for large negative x
    assert!(out.data()[0].abs() < 0.1);
    // GELU is monotonically near-increasing for positive x
    assert!(out.data()[3] > out.data()[2]);
    assert!(out.data()[4] > out.data()[3]);
}

#[test]
fn silu_produces_expected_values() {
    let input = Tensor::from_vec(vec![5], vec![-2.0, -1.0, 0.0, 1.0, 2.0]).unwrap();
    let out = silu(&input);
    assert_eq!(out.shape(), &[5]);
    // SiLU(0) = 0 * sigmoid(0) = 0
    assert!((out.data()[2] - 0.0).abs() < 1e-6);
    // SiLU(1) = 1 * sigmoid(1) = 0.7310586
    // Uses fast 3-term exp polynomial in SIMD path (~2e-4 max error).
    assert_slice_close(&[out.data()[3]], &[0.7310586], 2e-4);
    // SiLU(-1) = -1 * sigmoid(-1) = -0.26894143
    assert_slice_close(&[out.data()[1]], &[-0.26894143], 2e-4);
}

#[test]
fn mish_produces_expected_values() {
    let input = Tensor::from_vec(vec![5], vec![-2.0, -1.0, 0.0, 1.0, 2.0]).unwrap();
    let out = mish(&input);
    assert_eq!(out.shape(), &[5]);
    // Mish(0) = 0 * tanh(ln(2)) = 0
    assert!((out.data()[2] - 0.0).abs() < 1e-6);
    // Mish(x) > 0 for positive x
    assert!(out.data()[3] > 0.0);
    assert!(out.data()[4] > 0.0);
    // Mish(x) is slightly negative for negative x
    assert!(out.data()[0] < 0.0);
}

#[test]
fn exp_simd_matches_scalar() {
    let data: Vec<f32> = (0..100).map(|i| (i as f32 - 50.0) * 0.1).collect();
    let input = Tensor::from_vec(vec![100], data.clone()).unwrap();
    let out = exp(&input);
    let expected: Vec<f32> = data.iter().map(|v| v.exp()).collect();
    assert_slice_close(out.data(), &expected, 1e-3);
}

#[test]
// Miri's software FP emulation is non-deterministic: two calls to f32::exp()
// with the same input can return different bit patterns. Confirmed by standalone
// test: `let a = 0.33f32.exp(); let b = 0.33f32.exp(); assert_eq!(a.to_bits(), b.to_bits())`
// fails under Miri. This is a known Miri limitation, not a bug in yscv.
// See: https://github.com/rust-lang/miri — FP semantics are approximate.
#[cfg_attr(miri, ignore)]
fn exp_with_config_disabled_matches_exp() {
    let input = build_tensor(&[128, 64, 3], 0.33);
    let baseline = exp(&input);
    let disabled = exp_with_config(&input, ParallelElementwiseConfig::disabled());
    assert_eq!(baseline, disabled);
}

#[test]
fn tanh_simd_matches_scalar() {
    let data: Vec<f32> = (0..100).map(|i| (i as f32 - 50.0) * 0.1).collect();
    let input = Tensor::from_vec(vec![100], data.clone()).unwrap();
    let out = tanh_act(&input);
    let expected: Vec<f32> = data.iter().map(|v| v.tanh()).collect();
    assert_slice_close(out.data(), &expected, 1e-3);
}

#[test]
// Miri's software FP emulation is non-deterministic: two calls to f32::exp()
// with the same input can return different bit patterns. Confirmed by standalone
// test: `let a = 0.33f32.exp(); let b = 0.33f32.exp(); assert_eq!(a.to_bits(), b.to_bits())`
// fails under Miri. This is a known Miri limitation, not a bug in yscv.
// See: https://github.com/rust-lang/miri — FP semantics are approximate.
#[cfg_attr(miri, ignore)]
fn tanh_with_config_disabled_matches_tanh() {
    let input = build_tensor(&[128, 64, 3], 0.27);
    let baseline = tanh_act(&input);
    let disabled = tanh_act_with_config(&input, ParallelElementwiseConfig::disabled());
    assert_eq!(baseline, disabled);
}

#[test]
#[cfg_attr(miri, ignore)]
fn parallel_activation_configs_match_disabled() {
    let input = build_tensor(&[1024, 1024], 0.23);
    let parallel = ParallelElementwiseConfig {
        min_parallel_elements: 1,
    };

    let sigmoid_parallel = sigmoid_with_config(&input, parallel);
    let sigmoid_disabled = sigmoid_with_config(&input, ParallelElementwiseConfig::disabled());
    assert_eq!(sigmoid_parallel, sigmoid_disabled);

    let tanh_parallel = tanh_act_with_config(&input, parallel);
    let tanh_disabled = tanh_act_with_config(&input, ParallelElementwiseConfig::disabled());
    assert_eq!(tanh_parallel, tanh_disabled);

    let silu_parallel = silu_with_config(&input, parallel);
    let silu_disabled = silu_with_config(&input, ParallelElementwiseConfig::disabled());
    assert_eq!(silu_parallel, silu_disabled);
}

#[test]
fn sigmoid_simd_matches_scalar() {
    let data: Vec<f32> = (0..100).map(|i| (i as f32 - 50.0) * 0.1).collect();
    let input = Tensor::from_vec(vec![100], data.clone()).unwrap();
    let out = sigmoid(&input);
    let expected: Vec<f32> = data.iter().map(|v| 1.0 / (1.0 + (-v).exp())).collect();
    assert_slice_close(out.data(), &expected, 0.035);
}

#[test]
fn backend_exp_and_tanh() {
    let backend = CpuBackend;
    let input = Tensor::from_vec(vec![4], vec![-1.0, 0.0, 1.0, 2.0]).unwrap();

    let exp_out = backend.exp(&input);
    let expected_exp: Vec<f32> = [-1.0f32, 0.0, 1.0, 2.0].iter().map(|v| v.exp()).collect();
    assert_slice_close(exp_out.data(), &expected_exp, 1e-3);

    let tanh_out = backend.tanh_act(&input);
    let expected_tanh: Vec<f32> = [-1.0f32, 0.0, 1.0, 2.0].iter().map(|v| v.tanh()).collect();
    assert_slice_close(tanh_out.data(), &expected_tanh, 1e-3);
}

// ── SIMD edge-case tests ──────────────────────────────────────
// Verify scalar tail paths for lengths not aligned to SIMD width.

const EDGE_LENGTHS: &[usize] = &[1, 2, 3, 4, 5, 7, 8, 15, 16, 17, 31, 32, 33];

#[test]
fn sigmoid_edge_lengths() {
    for &len in EDGE_LENGTHS {
        let data: Vec<f32> = (0..len).map(|i| i as f32 * 0.5 - 3.0).collect();
        let input = Tensor::from_vec(vec![len], data.clone()).unwrap();
        let out = sigmoid(&input);
        for (i, &v) in out.data().iter().enumerate() {
            let expected = 1.0 / (1.0 + (-data[i]).exp());
            assert!(
                (v - expected).abs() < 0.035,
                "sigmoid len={len} i={i}: got {v}, expected {expected}"
            );
        }
    }
}

#[test]
fn tanh_edge_lengths() {
    for &len in EDGE_LENGTHS {
        let data: Vec<f32> = (0..len).map(|i| i as f32 * 0.3 - 2.0).collect();
        let input = Tensor::from_vec(vec![len], data.clone()).unwrap();
        let out = tanh_act(&input);
        for (i, &v) in out.data().iter().enumerate() {
            let expected = data[i].tanh();
            assert!(
                (v - expected).abs() < 1e-3,
                "tanh len={len} i={i}: got {v}, expected {expected}"
            );
        }
    }
}

#[test]
fn exp_edge_lengths() {
    for &len in EDGE_LENGTHS {
        let data: Vec<f32> = (0..len)
            .map(|i| (i as f32 * 0.2 - 2.0).clamp(-10.0, 10.0))
            .collect();
        let input = Tensor::from_vec(vec![len], data.clone()).unwrap();
        let out = exp(&input);
        for (i, &v) in out.data().iter().enumerate() {
            let expected = data[i].exp();
            assert!(
                (v - expected).abs() < expected.abs() * 1e-4 + 1e-6,
                "exp len={len} i={i}: got {v}, expected {expected}"
            );
        }
    }
}

#[test]
fn gelu_edge_lengths() {
    for &len in EDGE_LENGTHS {
        let data: Vec<f32> = (0..len).map(|i| i as f32 * 0.4 - 2.5).collect();
        let input = Tensor::from_vec(vec![len], data.clone()).unwrap();
        let out = gelu(&input);
        for (i, &v) in out.data().iter().enumerate() {
            let x = data[i];
            let s = 1.0 / (1.0 + (-1.702 * x).exp());
            let expected = x * s;
            assert!(
                (v - expected).abs() < 1e-3,
                "gelu len={len} i={i}: got {v}, expected {expected}"
            );
        }
    }
}

#[test]
fn relu_edge_lengths() {
    for &len in EDGE_LENGTHS {
        let data: Vec<f32> = (0..len).map(|i| i as f32 * 0.5 - 3.0).collect();
        let input = Tensor::from_vec(vec![len], data.clone()).unwrap();
        let out = relu(&input);
        for (i, &v) in out.data().iter().enumerate() {
            let expected = data[i].max(0.0);
            assert_eq!(v, expected, "relu len={len} i={i}");
        }
    }
}
