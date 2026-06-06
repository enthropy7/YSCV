// ===========================================================================
// Tests
// ===========================================================================

use super::*;

fn assert_close(a: &[f32], b: &[f32], tol: f32) {
    assert_eq!(a.len(), b.len(), "length mismatch");
    for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        let d = (x - y).abs();
        assert!(d <= tol, "index {i}: {x} vs {y}, diff={d}, tolerance={tol}");
    }
}

fn assert_path_matches_features(path: SimdDispatchPath, report: CpuDispatchReport) {
    let features = report.cpu.features;
    match path {
        SimdDispatchPath::Avx512 => assert!(features.avx512f, "{path:?} without AVX-512F"),
        SimdDispatchPath::Avx => assert!(features.avx, "{path:?} without AVX"),
        SimdDispatchPath::Sse2 => assert!(features.sse2, "{path:?} without SSE2"),
        SimdDispatchPath::Sse => assert!(features.sse, "{path:?} without SSE"),
        SimdDispatchPath::Neon => assert!(features.neon, "{path:?} without NEON"),
        SimdDispatchPath::Accelerate => {
            #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
            panic!("Accelerate path outside macOS aarch64");
        }
        SimdDispatchPath::Mkl => {
            #[cfg(not(all(feature = "mkl", any(target_arch = "x86", target_arch = "x86_64"))))]
            panic!("MKL path without the x86 mkl feature");
        }
        SimdDispatchPath::Armpl => {
            #[cfg(not(all(feature = "armpl", target_arch = "aarch64", not(target_os = "macos"))))]
            panic!("ARMPL path without the non-macOS aarch64 armpl feature");
        }
        SimdDispatchPath::Scalar => {}
    }
}

#[test]
fn cpu_dispatch_report_matches_host_features() {
    let report = cpu_dispatch_report();
    println!("cpu dispatch report: {report:#?}");

    assert_path_matches_features(report.relu, report);
    assert_path_matches_features(report.sigmoid, report);
    assert_path_matches_features(report.exp, report);
    assert_path_matches_features(report.binary, report);
    assert_path_matches_features(report.fma, report);
    assert_path_matches_features(report.reduce, report);
    assert_path_matches_features(report.softmax, report);
    assert_path_matches_features(report.batch_norm, report);
    assert_path_matches_features(report.layer_norm, report);
}

#[test]
fn exp_matches_scalar() {
    let input: Vec<f32> = (-20..=20).map(|i| i as f32 * 0.5).collect();
    let mut simd_out = vec![0.0f32; input.len()];
    let mut scalar_out = vec![0.0f32; input.len()];

    exp_slice_dispatch(&input, &mut simd_out);
    exp::exp_slice_scalar(&input, &mut scalar_out);

    // Degree-6 Taylor polynomial is accurate to roughly 1e-6 relative error
    for (i, (&s, &r)) in simd_out.iter().zip(scalar_out.iter()).enumerate() {
        let rel = if r.abs() > 1e-10 {
            (s - r).abs() / r.abs()
        } else {
            (s - r).abs()
        };
        assert!(
            rel < 1e-5,
            "exp mismatch at index {i}: simd={s}, scalar={r}, rel_err={rel}"
        );
    }
}

#[test]
fn sigmoid_dispatch_matches_scalar() {
    let input: Vec<f32> = (-30..=30).map(|i| i as f32 * 0.3).collect();
    let mut simd_out = vec![0.0f32; input.len()];
    let mut scalar_out = vec![0.0f32; input.len()];

    sigmoid_slice_dispatch(&input, &mut simd_out);
    // Use scalar sigmoid for reference
    for (o, &v) in scalar_out.iter_mut().zip(input.iter()) {
        *o = activation::sigmoid_scalar(v);
    }

    // Sigmoid uses Schraudolph bit-trick exp (~4% max error on exp,
    // but sigmoid squashes error near 0/1, practical max ~0.03).
    assert_close(&simd_out, &scalar_out, 0.035);
}

#[test]
fn tanh_dispatch_matches_scalar() {
    let input: Vec<f32> = (-30..=30).map(|i| i as f32 * 0.3).collect();
    let mut simd_out = vec![0.0f32; input.len()];
    let mut scalar_out = vec![0.0f32; input.len()];

    tanh_slice_dispatch(&input, &mut simd_out);
    exp::tanh_slice_dispatch_scalar(&input, &mut scalar_out);

    // Uses fast 3-term exp polynomial for sigmoid path (~2e-3 max error vs scalar tanh).
    assert_close(&simd_out, &scalar_out, 2e-3);
}

#[test]
fn max_reduce_matches_scalar() {
    let data: Vec<f32> = (0..37).map(|i| (i as f32 * 0.7 - 12.0).sin()).collect();
    let simd_result = max_reduce_dispatch(&data);
    let scalar_result = reduce::max_reduce_scalar(&data);
    assert!((simd_result - scalar_result).abs() < 1e-6);
}

#[test]
fn max_reduce_empty() {
    assert_eq!(max_reduce_dispatch(&[]), f32::NEG_INFINITY);
}

#[test]
fn add_reduce_matches_scalar() {
    let data: Vec<f32> = (0..37).map(|i| i as f32 * 0.1).collect();
    let simd_result = add_reduce_dispatch(&data);
    let scalar_result = reduce::add_reduce_scalar(&data);
    assert!(
        (simd_result - scalar_result).abs() < 1e-3,
        "simd={simd_result}, scalar={scalar_result}"
    );
}

#[test]
fn add_reduce_empty() {
    assert_eq!(add_reduce_dispatch(&[]), 0.0);
}

#[test]
#[allow(unsafe_code)]
fn fma_matches_scalar() {
    let a: Vec<f32> = (0..33).map(|i| i as f32 * 0.3).collect();
    let b: Vec<f32> = (0..33).map(|i| (i as f32 * 0.7).sin()).collect();
    let mut simd_acc = vec![1.0f32; 33];
    let mut scalar_acc = vec![1.0f32; 33];

    fma_slice_dispatch(&a, &b, &mut simd_acc);
    unsafe { fma::fma_slice_scalar(&a, &b, &mut scalar_acc) };

    assert_close(&simd_acc, &scalar_acc, 1e-5);
}

#[test]
fn sigmoid_dispatch_boundary_values() {
    // Verify sigmoid at key points
    let input = vec![-100.0, -10.0, 0.0, 10.0, 100.0];
    let mut output = vec![0.0f32; 5];
    sigmoid_slice_dispatch(&input, &mut output);

    // sigmoid(-100) ~ 0, sigmoid(0) = 0.5, sigmoid(100) ~ 1
    assert!(
        output[0] < 0.01,
        "sigmoid(-100) should be near 0: {}",
        output[0]
    );
    assert!(
        (output[2] - 0.5).abs() < 0.01,
        "sigmoid(0) should be near 0.5: {}",
        output[2]
    );
    assert!(
        output[4] > 0.99,
        "sigmoid(100) should be near 1: {}",
        output[4]
    );
}

#[test]
fn tanh_dispatch_boundary_values() {
    let input = vec![-100.0, -1.0, 0.0, 1.0, 100.0];
    let mut output = vec![0.0f32; 5];
    tanh_slice_dispatch(&input, &mut output);

    assert!(
        output[0] < -0.99,
        "tanh(-100) should be near -1: {}",
        output[0]
    );
    assert!(
        (output[2]).abs() < 0.01,
        "tanh(0) should be near 0: {}",
        output[2]
    );
    assert!(
        output[4] > 0.99,
        "tanh(100) should be near 1: {}",
        output[4]
    );
}
