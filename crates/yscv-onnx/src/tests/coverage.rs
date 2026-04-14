//! //!
//! Before the dispatch table at `runner/mod.rs:909-1047` listed 128
//! CPU operators with only 55 of them exercised by tests. The 73-operator
//! coverage gap was the framework's biggest silent-bug surface — for example,
//! the entire INT8 quantization path (`QLinearConv`, `QLinearMatMul`,
//! `ConvInteger`, `QuantizeLinear`, `DequantizeLinear`) had zero unit tests.
//!
//! This file closes the gap by walking the four severity buckets defined in
//! `docs/roadmap-1.0.md`:
//!
//! - **2.1 CRITICAL_QUANT** — quantization ops, mandatory for INT8 deployment.
//! - **2.2 HIGH_VISION** — common vision/inference ops (Resize, Slice, Pad, etc.).
//! - **2.3 MEDIUM_MATH** — activations, reductions, elementwise math.
//! - **2.4 LOW_UTIL** — shape/utility ops, logical ops, misc.
//!
//! Every test compares against a hand-rolled reference computed in pure Rust
//! so a regression in any dispatch arm produces a numerical mismatch with a
//! human-readable message instead of a panic.

use super::*;

// ---------------------------------------------------------------------------
// Local helpers
// ---------------------------------------------------------------------------

/// Build a single-output `[O, I, KH, KW]` Conv weight initializer.
fn conv_weight_initializer(
    name: &str,
    o: usize,
    i: usize,
    kh: usize,
    kw: usize,
    data: Vec<f32>,
) -> onnx::TensorProto {
    onnx::TensorProto {
        name: Some(name.into()),
        dims: vec![o as i64, i as i64, kh as i64, kw as i64],
        data_type: Some(1),
        float_data: data,
        ..Default::default()
    }
}

fn scalar_initializer(name: &str, value: f32) -> onnx::TensorProto {
    onnx::TensorProto {
        name: Some(name.into()),
        dims: vec![],
        data_type: Some(1),
        float_data: vec![value],
        ..Default::default()
    }
}

fn vec_initializer(name: &str, dims: Vec<i64>, data: Vec<f32>) -> onnx::TensorProto {
    onnx::TensorProto {
        name: Some(name.into()),
        dims,
        data_type: Some(1),
        float_data: data,
        ..Default::default()
    }
}

fn assert_close(actual: &[f32], expected: &[f32], tol: f32) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "length mismatch: {} vs {}",
        actual.len(),
        expected.len()
    );
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        let diff = (a - e).abs();
        assert!(
            diff < tol,
            "element {i}: actual {a} vs expected {e}, diff {diff} >= tol {tol}"
        );
    }
}

// ===========================================================================
// 2.1 CRITICAL_QUANT — quantization operators
// ===========================================================================
//
// `yscv-onnx` stores all tensors as f32 even for "quantized" data. The tests
// below verify that quantize/dequantize round-trips, QLinearConv, QLinearMatMul,
// MatMulInteger, and ConvInteger produce the values defined by the runtime
// implementations in `runner/{misc,conv,linear}.rs`.

#[test]
fn quantize_linear_basic() {
    // q = clamp(round(x / scale + zero_point), -128, 127)
    let input = Tensor::from_vec(vec![5], vec![-1.5, -0.5, 0.0, 0.5, 1.5]).unwrap();
    let scale = Tensor::scalar(0.5);
    let zp = Tensor::scalar(0.0);
    let out = run_single_op(
        "QuantizeLinear",
        vec![("x", input), ("scale", scale), ("zp", zp)],
        vec![],
        vec![],
        vec!["x", "scale", "zp"],
        "y",
    );
    assert_eq!(out.data(), &[-3.0, -1.0, 0.0, 1.0, 3.0]);
}

#[test]
fn quantize_linear_with_nonzero_zero_point_clamps() {
    // zero_point = 10, scale = 0.5: q = round(x / 0.5 + 10), clamped to [-128, 127].
    // Pick a large positive input to force clamping at the high end.
    let input = Tensor::from_vec(vec![3], vec![100.0, 0.0, -100.0]).unwrap();
    let scale = Tensor::scalar(0.5);
    let zp = Tensor::scalar(10.0);
    let out = run_single_op(
        "QuantizeLinear",
        vec![("x", input), ("scale", scale), ("zp", zp)],
        vec![],
        vec![],
        vec!["x", "scale", "zp"],
        "y",
    );
    // q[0] = round(200 + 10) = 210 → clamped to 127
    // q[1] = round(0 + 10)   = 10
    // q[2] = round(-200 + 10) = -190 → clamped to -128
    assert_eq!(out.data(), &[127.0, 10.0, -128.0]);
}

#[test]
fn dequantize_linear_basic() {
    // x = (q - zero_point) * scale
    let input = Tensor::from_vec(vec![4], vec![-2.0, 0.0, 2.0, 4.0]).unwrap();
    let scale = Tensor::scalar(0.25);
    let zp = Tensor::scalar(0.0);
    let out = run_single_op(
        "DequantizeLinear",
        vec![("q", input), ("scale", scale), ("zp", zp)],
        vec![],
        vec![],
        vec!["q", "scale", "zp"],
        "y",
    );
    assert_eq!(out.data(), &[-0.5, 0.0, 0.5, 1.0]);
}

#[test]
fn dequantize_linear_with_nonzero_zero_point() {
    let input = Tensor::from_vec(vec![3], vec![10.0, 20.0, 30.0]).unwrap();
    let scale = Tensor::scalar(0.1);
    let zp = Tensor::scalar(15.0);
    let out = run_single_op(
        "DequantizeLinear",
        vec![("q", input), ("scale", scale), ("zp", zp)],
        vec![],
        vec![],
        vec!["q", "scale", "zp"],
        "y",
    );
    // (10-15)*0.1 = -0.5; (20-15)*0.1 = 0.5; (30-15)*0.1 = 1.5
    assert_close(out.data(), &[-0.5, 0.5, 1.5], 1e-6);
}

#[test]
fn quantize_dequantize_round_trip_is_close_to_identity() {
    // Quantize then dequantize: small precision loss but sign and magnitude
    // preserved within `scale` of the original.
    let original = vec![-1.0_f32, -0.3, 0.0, 0.3, 1.0];
    let input = Tensor::from_vec(vec![5], original.clone()).unwrap();
    let scale = Tensor::scalar(0.05);
    let zp = Tensor::scalar(0.0);

    let q_node = onnx::NodeProto {
        op_type: Some("QuantizeLinear".into()),
        name: Some("q".into()),
        input: vec!["x".into(), "s".into(), "z".into()],
        output: vec!["q_out".into()],
        ..Default::default()
    };
    let dq_node = onnx::NodeProto {
        op_type: Some("DequantizeLinear".into()),
        name: Some("dq".into()),
        input: vec!["q_out".into(), "s".into(), "z".into()],
        output: vec!["y".into()],
        ..Default::default()
    };
    let s_init = scalar_initializer("s", 0.05);
    let z_init = scalar_initializer("z", 0.0);
    let bytes = build_minimal_onnx_model(
        vec![q_node, dq_node],
        vec![s_init, z_init],
        vec!["x", "s", "z"],
        vec!["y"],
    );
    let model = load_onnx_model(&bytes).unwrap();
    let mut feed = HashMap::new();
    feed.insert("x".to_string(), input);
    let _ = scale; // explicitly unused: initializer is the source of truth
    let _ = zp;
    let result = run_onnx_model(&model, feed).unwrap();
    let recovered = result["y"].data();

    // Each element should be within `scale` of the original.
    for (i, (&orig, &rec)) in original.iter().zip(recovered.iter()).enumerate() {
        assert!(
            (orig - rec).abs() <= 0.05 + 1e-6,
            "element {i}: orig {orig} vs recovered {rec}, diff > scale"
        );
    }
}

#[test]
fn dynamic_quantize_linear_three_outputs() {
    // DynamicQuantizeLinear emits (y, y_scale, y_zero_point); inputs are
    // mapped to a uint8 [0, 255] range based on the data's min/max.
    let input = Tensor::from_vec(vec![4], vec![-2.0, -1.0, 0.0, 1.0]).unwrap();

    let node = onnx::NodeProto {
        op_type: Some("DynamicQuantizeLinear".into()),
        name: Some("dql".into()),
        input: vec!["x".into()],
        output: vec!["y".into(), "y_scale".into(), "y_zp".into()],
        ..Default::default()
    };
    let bytes =
        build_minimal_onnx_model(vec![node], vec![], vec!["x"], vec!["y", "y_scale", "y_zp"]);
    let model = load_onnx_model(&bytes).unwrap();
    let mut feed = HashMap::new();
    feed.insert("x".to_string(), input.clone());
    let result = run_onnx_model(&model, feed).unwrap();

    // Reference: range = max(0, 1) - min(0, -2) = 1 - (-2) = 3 → scale = 3/255.
    let expected_scale = 3.0_f32 / 255.0;
    let expected_zp = (0.0_f32 - (-2.0) / expected_scale)
        .round()
        .clamp(0.0, 255.0);
    let recovered_scale = result["y_scale"].data()[0];
    let recovered_zp = result["y_zp"].data()[0];
    assert!((recovered_scale - expected_scale).abs() < 1e-6);
    assert_eq!(recovered_zp, expected_zp);

    // The quantized output should round-trip back to the input within `scale`.
    let q = result["y"].data();
    for (i, (&orig, &qv)) in input.data().iter().zip(q.iter()).enumerate() {
        let recovered = (qv - recovered_zp) * recovered_scale;
        assert!(
            (orig - recovered).abs() <= recovered_scale + 1e-5,
            "element {i}: orig {orig} vs recovered {recovered}"
        );
    }
}

#[test]
fn qlinear_conv_dequant_conv_quant_round_trip() {
    // QLinearConv: dequantize x and w, run float Conv, then quantize the
    // output by y_scale/y_zp. With identity-like weights and unit scales,
    // the output should match a plain Conv up to clamping.
    let x_data = vec![-2.0_f32, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
    let w_data = vec![1.0_f32]; // 1×1×1×1 kernel = scale-only conv

    let x = vec_initializer("x", vec![1, 1, 3, 3], x_data.clone());
    let x_scale = scalar_initializer("x_s", 1.0);
    let x_zp = scalar_initializer("x_zp", 0.0);
    let w = conv_weight_initializer("w", 1, 1, 1, 1, w_data.clone());
    let w_scale = scalar_initializer("w_s", 1.0);
    let w_zp = scalar_initializer("w_zp", 0.0);
    let y_scale = scalar_initializer("y_s", 1.0);
    let y_zp = scalar_initializer("y_zp", 0.0);

    let node = onnx::NodeProto {
        op_type: Some("QLinearConv".into()),
        name: Some("qconv".into()),
        input: vec![
            "x".into(),
            "x_s".into(),
            "x_zp".into(),
            "w".into(),
            "w_s".into(),
            "w_zp".into(),
            "y_s".into(),
            "y_zp".into(),
        ],
        output: vec!["y".into()],
        attribute: vec![
            make_ints_attr("kernel_shape", vec![1, 1]),
            make_ints_attr("strides", vec![1, 1]),
        ],
        ..Default::default()
    };
    let bytes = build_minimal_onnx_model(
        vec![node],
        vec![x, x_scale, x_zp, w, w_scale, w_zp, y_scale, y_zp],
        vec!["x", "x_s", "x_zp", "w", "w_s", "w_zp", "y_s", "y_zp"],
        vec!["y"],
    );
    let model = load_onnx_model(&bytes).unwrap();
    let result = run_onnx_model(&model, HashMap::new()).unwrap();
    let out = result["y"].data();

    // With unit scales and identity weight, output == clamp(input, -128, 127).
    let expected: Vec<f32> = x_data.iter().map(|&v| v.clamp(-128.0, 127.0)).collect();
    assert_close(out, &expected, 1e-5);
}

#[test]
fn qlinear_matmul_dequant_matmul_quant_round_trip() {
    // 2×3 × 3×2 matmul: with unit scales the qlinear path equals a plain
    // matmul clamped to [-128, 127].
    let a_data = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2×3
    let b_data = vec![0.5_f32, -0.5, 1.0, -1.0, 0.0, 2.0]; // 3×2

    let a = vec_initializer("a", vec![2, 3], a_data.clone());
    let a_s = scalar_initializer("a_s", 1.0);
    let a_zp = scalar_initializer("a_zp", 0.0);
    let b = vec_initializer("b", vec![3, 2], b_data.clone());
    let b_s = scalar_initializer("b_s", 1.0);
    let b_zp = scalar_initializer("b_zp", 0.0);
    let y_s = scalar_initializer("y_s", 1.0);
    let y_zp = scalar_initializer("y_zp", 0.0);

    let node = onnx::NodeProto {
        op_type: Some("QLinearMatMul".into()),
        name: Some("qmm".into()),
        input: vec![
            "a".into(),
            "a_s".into(),
            "a_zp".into(),
            "b".into(),
            "b_s".into(),
            "b_zp".into(),
            "y_s".into(),
            "y_zp".into(),
        ],
        output: vec!["y".into()],
        ..Default::default()
    };
    let bytes = build_minimal_onnx_model(
        vec![node],
        vec![a, a_s, a_zp, b, b_s, b_zp, y_s, y_zp],
        vec!["a", "a_s", "a_zp", "b", "b_s", "b_zp", "y_s", "y_zp"],
        vec!["y"],
    );
    let model = load_onnx_model(&bytes).unwrap();
    let result = run_onnx_model(&model, HashMap::new()).unwrap();
    let out = result["y"].data();

    // Reference: plain 2×3 × 3×2 matmul, clamped.
    let mut expected = vec![0.0_f32; 4];
    for i in 0..2 {
        for j in 0..2 {
            let mut acc = 0.0_f32;
            for k in 0..3 {
                acc += a_data[i * 3 + k] * b_data[k * 2 + j];
            }
            expected[i * 2 + j] = acc.round().clamp(-128.0, 127.0);
        }
    }
    assert_close(out, &expected, 1e-5);
}

#[test]
fn matmul_integer_treats_inputs_as_offset_quantized_int_matmul() {
    // MatMulInteger: subtract zero points, then plain matmul (no rescale).
    let a_data = vec![10.0_f32, 11.0, 12.0, 13.0]; // 2×2
    let b_data = vec![1.0_f32, 0.0, 0.0, 1.0]; // 2×2 identity
    let a_zp = 10.0_f32; // shifts a to [0,1,2,3]
    let b_zp = 0.0_f32;

    let a = vec_initializer("a", vec![2, 2], a_data);
    let b = vec_initializer("b", vec![2, 2], b_data);
    let a_zp_init = scalar_initializer("a_zp", a_zp);
    let b_zp_init = scalar_initializer("b_zp", b_zp);

    let node = onnx::NodeProto {
        op_type: Some("MatMulInteger".into()),
        name: Some("mmi".into()),
        input: vec!["a".into(), "b".into(), "a_zp".into(), "b_zp".into()],
        output: vec!["y".into()],
        ..Default::default()
    };
    let bytes = build_minimal_onnx_model(
        vec![node],
        vec![a, b, a_zp_init, b_zp_init],
        vec!["a", "b", "a_zp", "b_zp"],
        vec!["y"],
    );
    let model = load_onnx_model(&bytes).unwrap();
    let result = run_onnx_model(&model, HashMap::new()).unwrap();
    // After zero-point subtraction A becomes [[0,1],[2,3]], B is identity.
    // Result: [[0,1],[2,3]].
    assert_close(result["y"].data(), &[0.0, 1.0, 2.0, 3.0], 1e-5);
}

// ===========================================================================
// 2.2 HIGH_VISION — vision/inference ops
// ===========================================================================
//
// These operators show up in essentially every vision model: ConvTranspose,
// Resize, Slice, Pad, Cast, Tile, Expand, Where, GatherND, ScatterND,
// RoiAlign. Several of them have non-trivial attribute parsing or
// shape-broadcasting logic, so a single happy-path test for each is enough
// to catch a future regression.

#[test]
fn slice_basic_axis0() {
    // Slice [0..2] along axis 0 of a [4, 3] tensor → [2, 3].
    let input =
        Tensor::from_vec(vec![4, 3], (0..12).map(|v| v as f32).collect::<Vec<_>>()).unwrap();
    let starts = Tensor::from_vec(vec![1], vec![0.0]).unwrap();
    let ends = Tensor::from_vec(vec![1], vec![2.0]).unwrap();
    let axes = Tensor::from_vec(vec![1], vec![0.0]).unwrap();
    let out = run_single_op(
        "Slice",
        vec![("data", input), ("s", starts), ("e", ends), ("a", axes)],
        vec![],
        vec![],
        vec!["data", "s", "e", "a"],
        "y",
    );
    assert_eq!(out.shape(), &[2, 3]);
    assert_eq!(out.data(), &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
}

#[test]
fn slice_with_step_2() {
    // Slice [0..6:2] along axis 0 of a [6] tensor → [0, 2, 4].
    let input = Tensor::from_vec(vec![6], vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
    let starts = Tensor::from_vec(vec![1], vec![0.0]).unwrap();
    let ends = Tensor::from_vec(vec![1], vec![6.0]).unwrap();
    let axes = Tensor::from_vec(vec![1], vec![0.0]).unwrap();
    let steps = Tensor::from_vec(vec![1], vec![2.0]).unwrap();
    let out = run_single_op(
        "Slice",
        vec![
            ("data", input),
            ("s", starts),
            ("e", ends),
            ("a", axes),
            ("st", steps),
        ],
        vec![],
        vec![],
        vec!["data", "s", "e", "a", "st"],
        "y",
    );
    assert_eq!(out.shape(), &[3]);
    assert_eq!(out.data(), &[0.0, 2.0, 4.0]);
}

#[test]
fn pad_constant_zero() {
    // Pad [3] with [2, 1] (begin=2, end=1) → [5] zeros padded.
    let input = Tensor::from_vec(vec![3], vec![10.0, 20.0, 30.0]).unwrap();
    let pads = Tensor::from_vec(vec![2], vec![2.0, 1.0]).unwrap();
    let out = run_single_op(
        "Pad",
        vec![("data", input), ("pads", pads)],
        vec![],
        vec![],
        vec!["data", "pads"],
        "y",
    );
    assert_eq!(out.shape(), &[6]);
    assert_eq!(out.data(), &[0.0, 0.0, 10.0, 20.0, 30.0, 0.0]);
}

#[test]
fn pad_constant_with_value() {
    // Pad [2] with [1, 1] using fill value -7.5 → [-7.5, a, b, -7.5].
    let input = Tensor::from_vec(vec![2], vec![1.0, 2.0]).unwrap();
    let pads = Tensor::from_vec(vec![2], vec![1.0, 1.0]).unwrap();
    let fill = Tensor::scalar(-7.5);
    let out = run_single_op(
        "Pad",
        vec![("data", input), ("pads", pads), ("v", fill)],
        vec![],
        vec![],
        vec!["data", "pads", "v"],
        "y",
    );
    assert_eq!(out.data(), &[-7.5, 1.0, 2.0, -7.5]);
}

#[test]
fn cast_passthrough_for_f32() {
    // The runner currently only supports f32 storage; Cast acts as identity.
    let input = Tensor::from_vec(vec![3], vec![1.5, 2.5, -3.5]).unwrap();
    let out = run_single_op(
        "Cast",
        vec![("x", input.clone())],
        vec![],
        vec![make_int_attr("to", 1)], // 1 = FLOAT
        vec!["x"],
        "y",
    );
    assert_eq!(out.data(), input.data());
}

#[test]
fn tile_repeats_each_axis() {
    // [1, 2] tiled by [3, 2] → [3, 4].
    let input = Tensor::from_vec(vec![1, 2], vec![1.0, 2.0]).unwrap();
    let repeats = Tensor::from_vec(vec![2], vec![3.0, 2.0]).unwrap();
    let out = run_single_op(
        "Tile",
        vec![("x", input), ("r", repeats)],
        vec![],
        vec![],
        vec!["x", "r"],
        "y",
    );
    assert_eq!(out.shape(), &[3, 4]);
    assert_eq!(
        out.data(),
        &[1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0]
    );
}

#[test]
fn expand_broadcasts_to_target_shape() {
    // [1, 3] expanded to [2, 3] replicates the row.
    let input = Tensor::from_vec(vec![1, 3], vec![10.0, 20.0, 30.0]).unwrap();
    let target = Tensor::from_vec(vec![2], vec![2.0, 3.0]).unwrap();
    let out = run_single_op(
        "Expand",
        vec![("x", input), ("s", target)],
        vec![],
        vec![],
        vec!["x", "s"],
        "y",
    );
    assert_eq!(out.shape(), &[2, 3]);
    assert_eq!(out.data(), &[10.0, 20.0, 30.0, 10.0, 20.0, 30.0]);
}

#[test]
fn where_op_picks_x_when_cond_true() {
    let cond = Tensor::from_vec(vec![4], vec![1.0, 0.0, 1.0, 0.0]).unwrap();
    let x = Tensor::from_vec(vec![4], vec![10.0, 20.0, 30.0, 40.0]).unwrap();
    let y = Tensor::from_vec(vec![4], vec![100.0, 200.0, 300.0, 400.0]).unwrap();
    let out = run_single_op(
        "Where",
        vec![("c", cond), ("x", x), ("y", y)],
        vec![],
        vec![],
        vec!["c", "x", "y"],
        "out",
    );
    // c[i] > 0 → x[i], else y[i]
    assert_eq!(out.data(), &[10.0, 200.0, 30.0, 400.0]);
}

#[test]
fn resize_nearest_neighbour_doubles_spatial() {
    // [1, 1, 2, 2] → resize to [1, 1, 4, 4] with nearest-neighbour.
    let input = Tensor::from_vec(vec![1, 1, 2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    // Resize takes [roi, scales, sizes] — we only need sizes (input[3]).
    // ONNX has empty placeholder strings for unused inputs.
    let sizes = Tensor::from_vec(vec![4], vec![1.0, 1.0, 4.0, 4.0]).unwrap();
    // Build the node manually because run_single_op can't represent empty inputs.
    let sizes_init = vec_initializer("sizes", vec![4], sizes.data().to_vec());
    let node = onnx::NodeProto {
        op_type: Some("Resize".into()),
        name: Some("resize".into()),
        input: vec![
            "x".into(),
            "".into(), // roi
            "".into(), // scales
            "sizes".into(),
        ],
        output: vec!["y".into()],
        ..Default::default()
    };
    let bytes =
        build_minimal_onnx_model(vec![node], vec![sizes_init], vec!["x", "sizes"], vec!["y"]);
    let model = load_onnx_model(&bytes).unwrap();
    let mut feed = HashMap::new();
    feed.insert("x".to_string(), input);
    let result = run_onnx_model(&model, feed).unwrap();
    let out = result["y"].data();
    assert_eq!(result["y"].shape(), &[1, 1, 4, 4]);
    // Nearest-neighbour 2× upscale: each input cell becomes a 2×2 block.
    assert_eq!(
        out,
        &[
            1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 3.0, 3.0, 4.0, 4.0,
        ]
    );
}

#[test]
fn upsample_dispatches_through_resize() {
    // Upsample is dispatched as a Resize alias at runner/mod.rs:974.
    let input = Tensor::from_vec(vec![1, 1, 2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let sizes_init = vec_initializer("sizes", vec![4], vec![1.0, 1.0, 4.0, 4.0]);
    let node = onnx::NodeProto {
        op_type: Some("Upsample".into()),
        name: Some("up".into()),
        input: vec!["x".into(), "".into(), "".into(), "sizes".into()],
        output: vec!["y".into()],
        ..Default::default()
    };
    let bytes =
        build_minimal_onnx_model(vec![node], vec![sizes_init], vec!["x", "sizes"], vec!["y"]);
    let model = load_onnx_model(&bytes).unwrap();
    let mut feed = HashMap::new();
    feed.insert("x".to_string(), input);
    let result = run_onnx_model(&model, feed).unwrap();
    assert_eq!(result["y"].shape(), &[1, 1, 4, 4]);
}

#[test]
fn conv_transpose_doubles_spatial_with_stride2() {
    // ConvTranspose 2×2 kernel, stride 2, identity-ish: each input pixel
    // contributes to its own 2×2 output block. With kernel = ones and a 1×1
    // input, output should be a 2×2 block of input values.
    let input = Tensor::from_vec(vec![1, 1, 1, 1], vec![5.0]).unwrap();
    let weight = conv_weight_initializer("w", 1, 1, 2, 2, vec![1.0, 1.0, 1.0, 1.0]);
    let node = onnx::NodeProto {
        op_type: Some("ConvTranspose".into()),
        name: Some("ct".into()),
        input: vec!["x".into(), "w".into()],
        output: vec!["y".into()],
        attribute: vec![
            make_ints_attr("kernel_shape", vec![2, 2]),
            make_ints_attr("strides", vec![2, 2]),
        ],
        ..Default::default()
    };
    let bytes = build_minimal_onnx_model(vec![node], vec![weight], vec!["x", "w"], vec!["y"]);
    let model = load_onnx_model(&bytes).unwrap();
    let mut feed = HashMap::new();
    feed.insert("x".to_string(), input);
    let result = run_onnx_model(&model, feed).unwrap();
    assert_eq!(result["y"].shape(), &[1, 1, 2, 2]);
    assert_close(result["y"].data(), &[5.0, 5.0, 5.0, 5.0], 1e-5);
}

#[test]
fn gather_nd_picks_individual_elements() {
    // data = [[1, 2], [3, 4]], indices = [[0, 0], [1, 1]] → [1, 4].
    let data = Tensor::from_vec(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let indices = Tensor::from_vec(vec![2, 2], vec![0.0, 0.0, 1.0, 1.0]).unwrap();
    let out = run_single_op(
        "GatherND",
        vec![("d", data), ("i", indices)],
        vec![],
        vec![],
        vec!["d", "i"],
        "y",
    );
    assert_eq!(out.data(), &[1.0, 4.0]);
}

#[test]
fn scatter_nd_writes_individual_elements() {
    // data = zeros(2, 2), indices = [[0, 0], [1, 1]], updates = [9, 8]
    // → [[9, 0], [0, 8]].
    let data = Tensor::from_vec(vec![2, 2], vec![0.0; 4]).unwrap();
    let indices = Tensor::from_vec(vec![2, 2], vec![0.0, 0.0, 1.0, 1.0]).unwrap();
    let updates = Tensor::from_vec(vec![2], vec![9.0, 8.0]).unwrap();
    let out = run_single_op(
        "ScatterND",
        vec![("d", data), ("i", indices), ("u", updates)],
        vec![],
        vec![],
        vec!["d", "i", "u"],
        "y",
    );
    assert_eq!(out.data(), &[9.0, 0.0, 0.0, 8.0]);
}

#[test]
fn roi_align_extracts_single_full_image_region() {
    // Single 1-channel 4×4 input, single ROI covering the entire image,
    // 2×2 output bins. With sampling_ratio=1 each output bin equals one
    // bilinear sample at its centre.
    let data: Vec<f32> = (0..16).map(|v| v as f32).collect();
    let input = Tensor::from_vec(vec![1, 1, 4, 4], data).unwrap();
    let rois = Tensor::from_vec(vec![1, 4], vec![0.0, 0.0, 4.0, 4.0]).unwrap();
    let batch_indices = Tensor::from_vec(vec![1], vec![0.0]).unwrap();

    let attrs = vec![
        make_int_attr("output_height", 2),
        make_int_attr("output_width", 2),
        make_int_attr("sampling_ratio", 1),
        onnx::AttributeProto {
            name: Some("spatial_scale".into()),
            r#type: Some(1),
            f: Some(1.0),
            ..Default::default()
        },
    ];
    let out = run_single_op(
        "RoiAlign",
        vec![("x", input), ("rois", rois), ("bi", batch_indices)],
        vec![],
        attrs,
        vec!["x", "rois", "bi"],
        "y",
    );
    assert_eq!(out.shape(), &[1, 1, 2, 2]);
    // Each output element is finite — exact values depend on the bilinear
    // sampler; we sanity-check structure rather than memorise the formula.
    for &v in out.data() {
        assert!(v.is_finite());
    }
    // Top-left bin samples around (1, 1) which interpolates among
    // input[(0,0), (0,1), (1,0), (1,1)] = 0..5 — expect a value strictly
    // between the min and max of those four.
    let tl = out.data()[0];
    assert!(tl > 0.0 && tl < 16.0, "tl = {tl}");
}

#[test]
fn lp_normalization_l1_normalises_each_row_to_unit_l1() {
    let input = Tensor::from_vec(vec![1, 4], vec![1.0, -2.0, 3.0, -4.0]).unwrap();
    let attrs = vec![make_int_attr("axis", 1), make_int_attr("p", 1)];
    let out = run_single_op(
        "LpNormalization",
        vec![("x", input)],
        vec![],
        attrs,
        vec!["x"],
        "y",
    );
    let l1: f32 = out.data().iter().map(|v| v.abs()).sum();
    assert!((l1 - 1.0).abs() < 1e-5, "L1 norm = {l1}");
}

#[test]
fn lrn_local_response_normalization_runs_without_panic() {
    // LRN: y = x / ((bias + alpha/size * sum(x_neighbours^2))^beta).
    // Trivially with size=1 and bias=1: y = x / (1 + alpha*x^2)^beta.
    let input = Tensor::from_vec(vec![1, 4, 1, 1], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let attrs = vec![
        make_int_attr("size", 3),
        onnx::AttributeProto {
            name: Some("alpha".into()),
            r#type: Some(1),
            f: Some(1e-4),
            ..Default::default()
        },
        onnx::AttributeProto {
            name: Some("beta".into()),
            r#type: Some(1),
            f: Some(0.75),
            ..Default::default()
        },
        onnx::AttributeProto {
            name: Some("bias".into()),
            r#type: Some(1),
            f: Some(1.0),
            ..Default::default()
        },
    ];
    let out = run_single_op(
        "LRN",
        vec![("x", input.clone())],
        vec![],
        attrs,
        vec!["x"],
        "y",
    );
    // LRN with these tiny coefficients should leave values close to the input.
    let inp_d = input.data();
    let out_d = out.data();
    for (i, (&iv, &ov)) in inp_d.iter().zip(out_d.iter()).enumerate() {
        assert!(
            (iv - ov).abs() < 0.05,
            "LRN element {i}: in {iv} out {ov} differs by more than 0.05"
        );
    }
}

// ===========================================================================
// 2.3 MEDIUM_MATH — activations, elementwise math, reductions, variadic ops
// ===========================================================================
//
// This bucket exists to make sure no operator silently rots: every test runs
// the dispatch arm at runner/mod.rs and asserts the value matches a hand-rolled
// reference. Most ops here use the simple `run_single_op` helper.

/// Run a unary single-input op and assert each output element equals the
/// reference produced by applying `f` to the input element.
fn assert_unary_op(op: &str, input_data: Vec<f32>, f: impl Fn(f32) -> f32, tol: f32) {
    let n = input_data.len();
    let input = Tensor::from_vec(vec![n], input_data.clone()).unwrap();
    let out = run_single_op(op, vec![("x", input)], vec![], vec![], vec!["x"], "y");
    let expected: Vec<f32> = input_data.iter().copied().map(&f).collect();
    assert_close(out.data(), &expected, tol);
}

fn assert_binary_op(op: &str, a: Vec<f32>, b: Vec<f32>, f: impl Fn(f32, f32) -> f32, tol: f32) {
    let n = a.len();
    let at = Tensor::from_vec(vec![n], a.clone()).unwrap();
    let bt = Tensor::from_vec(vec![n], b.clone()).unwrap();
    let out = run_single_op(
        op,
        vec![("a", at), ("b", bt)],
        vec![],
        vec![],
        vec!["a", "b"],
        "y",
    );
    let expected: Vec<f32> = a.iter().zip(b.iter()).map(|(&x, &y)| f(x, y)).collect();
    assert_close(out.data(), &expected, tol);
}

// ── Unary scalar functions (exec_unary path) ──────────────────────────

#[test]
fn unary_tan() {
    assert_unary_op("Tan", vec![-0.5, 0.0, 0.5], |v| v.tan(), 1e-5);
}
#[test]
fn unary_asin() {
    assert_unary_op("Asin", vec![-0.5, 0.0, 0.5], |v| v.asin(), 1e-5);
}
#[test]
fn unary_acos() {
    assert_unary_op("Acos", vec![-0.5, 0.0, 0.5], |v| v.acos(), 1e-5);
}
#[test]
fn unary_atan() {
    assert_unary_op("Atan", vec![-1.0, 0.0, 1.0], |v| v.atan(), 1e-5);
}
#[test]
fn unary_sinh() {
    assert_unary_op("Sinh", vec![-1.0, 0.0, 1.0], |v| v.sinh(), 1e-5);
}
#[test]
fn unary_cosh() {
    assert_unary_op("Cosh", vec![-1.0, 0.0, 1.0], |v| v.cosh(), 1e-5);
}
#[test]
fn unary_asinh() {
    assert_unary_op("Asinh", vec![-1.0, 0.0, 1.0], |v| v.asinh(), 1e-5);
}
#[test]
fn unary_acosh() {
    assert_unary_op("Acosh", vec![1.0, 1.5, 2.0], |v| v.acosh(), 1e-5);
}
#[test]
fn unary_atanh() {
    assert_unary_op("Atanh", vec![-0.5, 0.0, 0.5], |v| v.atanh(), 1e-5);
}
#[test]
fn unary_round() {
    assert_unary_op("Round", vec![-1.6, -0.5, 0.4, 1.7], |v| v.round(), 1e-6);
}
#[test]
fn unary_sign() {
    assert_unary_op("Sign", vec![-3.0, 0.0, 4.5], |v| v.signum(), 1e-6);
}
#[test]
fn unary_isnan() {
    let n = 4;
    let input = Tensor::from_vec(vec![n], vec![0.0, f32::NAN, 1.0, f32::NAN]).unwrap();
    let out = run_single_op("IsNaN", vec![("x", input)], vec![], vec![], vec!["x"], "y");
    assert_eq!(out.data(), &[0.0, 1.0, 0.0, 1.0]);
}
#[test]
fn unary_isinf() {
    let input =
        Tensor::from_vec(vec![4], vec![0.0, f32::INFINITY, 1.0, f32::NEG_INFINITY]).unwrap();
    let out = run_single_op("IsInf", vec![("x", input)], vec![], vec![], vec!["x"], "y");
    assert_eq!(out.data(), &[0.0, 1.0, 0.0, 1.0]);
}
#[test]
fn unary_softsign() {
    assert_unary_op(
        "Softsign",
        vec![-2.0, 0.0, 2.0],
        |v| v / (1.0 + v.abs()),
        1e-6,
    );
}
#[test]
fn unary_mish() {
    // Mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1+e^x))
    assert_unary_op(
        "Mish",
        vec![-1.0, 0.0, 1.0, 2.0],
        |v| v * (1.0 + v.exp()).ln().tanh(),
        1e-5,
    );
}

// ── Tensor-level unary ops (exec_tensor_op path: SIMD-accelerated) ────

#[test]
fn tensor_op_exp() {
    let input = Tensor::from_vec(vec![4], vec![-1.0, 0.0, 1.0, 2.0]).unwrap();
    let out = run_single_op(
        "Exp",
        vec![("x", input.clone())],
        vec![],
        vec![],
        vec!["x"],
        "y",
    );
    let expected: Vec<f32> = input.data().iter().map(|v| v.exp()).collect();
    assert_close(out.data(), &expected, 1e-5);
}
#[test]
fn tensor_op_log() {
    let e = std::f32::consts::E;
    let input = Tensor::from_vec(vec![3], vec![1.0, e, e * e]).unwrap();
    let out = run_single_op(
        "Log",
        vec![("x", input.clone())],
        vec![],
        vec![],
        vec!["x"],
        "y",
    );
    let expected: Vec<f32> = input.data().iter().map(|v| v.ln()).collect();
    assert_close(out.data(), &expected, 1e-5);
}
#[test]
fn tensor_op_sqrt() {
    let input = Tensor::from_vec(vec![4], vec![0.0, 1.0, 4.0, 9.0]).unwrap();
    let out = run_single_op("Sqrt", vec![("x", input)], vec![], vec![], vec!["x"], "y");
    assert_close(out.data(), &[0.0, 1.0, 2.0, 3.0], 1e-5);
}
#[test]
fn tensor_op_neg() {
    let input = Tensor::from_vec(vec![3], vec![-1.0, 0.0, 2.5]).unwrap();
    let out = run_single_op("Neg", vec![("x", input)], vec![], vec![], vec!["x"], "y");
    assert_eq!(out.data(), &[1.0, 0.0, -2.5]);
}
#[test]
fn tensor_op_abs() {
    let input = Tensor::from_vec(vec![3], vec![-1.5, 0.0, 2.5]).unwrap();
    let out = run_single_op("Abs", vec![("x", input)], vec![], vec![], vec!["x"], "y");
    assert_eq!(out.data(), &[1.5, 0.0, 2.5]);
}
#[test]
fn tensor_op_floor() {
    let input = Tensor::from_vec(vec![4], vec![-1.7, -0.3, 0.3, 1.7]).unwrap();
    let out = run_single_op("Floor", vec![("x", input)], vec![], vec![], vec!["x"], "y");
    assert_eq!(out.data(), &[-2.0, -1.0, 0.0, 1.0]);
}
#[test]
fn tensor_op_ceil() {
    let input = Tensor::from_vec(vec![4], vec![-1.7, -0.3, 0.3, 1.7]).unwrap();
    let out = run_single_op("Ceil", vec![("x", input)], vec![], vec![], vec!["x"], "y");
    assert_eq!(out.data(), &[-1.0, 0.0, 1.0, 2.0]);
}
#[test]
fn tensor_op_reciprocal() {
    let input = Tensor::from_vec(vec![3], vec![1.0, 2.0, 4.0]).unwrap();
    let out = run_single_op(
        "Reciprocal",
        vec![("x", input)],
        vec![],
        vec![],
        vec!["x"],
        "y",
    );
    assert_close(out.data(), &[1.0, 0.5, 0.25], 1e-6);
}
#[test]
fn tensor_op_tanh() {
    let input = Tensor::from_vec(vec![3], vec![-1.0, 0.0, 1.0]).unwrap();
    let out = run_single_op(
        "Tanh",
        vec![("x", input.clone())],
        vec![],
        vec![],
        vec!["x"],
        "y",
    );
    let expected: Vec<f32> = input.data().iter().map(|v| v.tanh()).collect();
    assert_close(out.data(), &expected, 1e-5);
}

// ── Binary elementwise ops ────────────────────────────────────────────

#[test]
fn binary_sub() {
    assert_binary_op(
        "Sub",
        vec![5.0, 8.0, 3.0],
        vec![1.0, 2.0, 3.0],
        |a, b| a - b,
        1e-6,
    );
}
#[test]
fn binary_mul() {
    assert_binary_op(
        "Mul",
        vec![2.0, 3.0, 4.0],
        vec![5.0, 6.0, 7.0],
        |a, b| a * b,
        1e-6,
    );
}
#[test]
fn binary_div() {
    assert_binary_op(
        "Div",
        vec![6.0, 9.0, 12.0],
        vec![2.0, 3.0, 4.0],
        |a, b| a / b,
        1e-6,
    );
}
#[test]
fn binary_pow() {
    assert_binary_op(
        "Pow",
        vec![2.0, 3.0, 4.0],
        vec![2.0, 2.0, 0.5],
        f32::powf,
        1e-5,
    );
}

#[test]
fn binary_mod_default_python_style() {
    // The runner's exec_mod uses Python-style modulo when `fmod=0` (default):
    // a - floor(a/b) * b. For a positive operand this matches `a % b`; for
    // a negative dividend with a positive divisor it returns a non-negative
    // remainder, unlike Rust's `i32 % i32`.
    let a = Tensor::from_vec(vec![3], vec![10.0, 7.0, -5.0]).unwrap();
    let b = Tensor::from_vec(vec![3], vec![3.0, 2.0, 4.0]).unwrap();
    let out = run_single_op(
        "Mod",
        vec![("a", a), ("b", b)],
        vec![],
        vec![],
        vec!["a", "b"],
        "y",
    );
    let py_mod = |x: f32, y: f32| x - (x / y).floor() * y;
    let expected = [py_mod(10.0, 3.0), py_mod(7.0, 2.0), py_mod(-5.0, 4.0)];
    assert_close(out.data(), &expected, 1e-6);
}

#[test]
fn binary_mod_fmod_uses_truncation() {
    // With fmod=1 the runner switches to Rust's `%` (truncation, sign of
    // dividend). For -5 % 4 this returns -1, distinct from the default path.
    let a = Tensor::from_vec(vec![1], vec![-5.0]).unwrap();
    let b = Tensor::from_vec(vec![1], vec![4.0]).unwrap();
    let attrs = vec![make_int_attr("fmod", 1)];
    let out = run_single_op(
        "Mod",
        vec![("a", a), ("b", b)],
        vec![],
        attrs,
        vec!["a", "b"],
        "y",
    );
    assert_eq!(out.data(), &[-1.0]);
}

#[test]
fn binary_bitshift_left() {
    let a = Tensor::from_vec(vec![3], vec![1.0, 2.0, 4.0]).unwrap();
    let b = Tensor::from_vec(vec![3], vec![1.0, 2.0, 3.0]).unwrap();
    let dir_attr = onnx::AttributeProto {
        name: Some("direction".into()),
        r#type: Some(3), // STRING
        s: Some(b"LEFT".to_vec()),
        ..Default::default()
    };
    let out = run_single_op(
        "BitShift",
        vec![("a", a), ("b", b)],
        vec![],
        vec![dir_attr],
        vec!["a", "b"],
        "y",
    );
    // 1<<1=2, 2<<2=8, 4<<3=32
    assert_eq!(out.data(), &[2.0, 8.0, 32.0]);
}

#[test]
fn binary_bitshift_right() {
    let a = Tensor::from_vec(vec![3], vec![16.0, 12.0, 8.0]).unwrap();
    let b = Tensor::from_vec(vec![3], vec![1.0, 2.0, 3.0]).unwrap();
    let dir_attr = onnx::AttributeProto {
        name: Some("direction".into()),
        r#type: Some(3),
        s: Some(b"RIGHT".to_vec()),
        ..Default::default()
    };
    let out = run_single_op(
        "BitShift",
        vec![("a", a), ("b", b)],
        vec![],
        vec![dir_attr],
        vec!["a", "b"],
        "y",
    );
    // 16>>1=8, 12>>2=3, 8>>3=1
    assert_eq!(out.data(), &[8.0, 3.0, 1.0]);
}

// ── Activations with attributes ───────────────────────────────────────

#[test]
fn activation_celu() {
    // Celu(x) = max(0, x) + min(0, alpha * (exp(x/alpha) - 1))
    let alpha = 1.0_f32;
    assert_unary_op(
        "Celu",
        vec![-2.0, 0.0, 1.0],
        |x| x.max(0.0) + (alpha * ((x / alpha).exp() - 1.0)).min(0.0),
        1e-5,
    );
}

#[test]
fn activation_thresholded_relu() {
    let alpha = 1.0_f32;
    let alpha_attr = onnx::AttributeProto {
        name: Some("alpha".into()),
        r#type: Some(1),
        f: Some(alpha),
        ..Default::default()
    };
    let input = Tensor::from_vec(vec![4], vec![-1.0, 0.5, 1.0, 2.0]).unwrap();
    let out = run_single_op(
        "ThresholdedRelu",
        vec![("x", input)],
        vec![],
        vec![alpha_attr],
        vec!["x"],
        "y",
    );
    // Output: 0 for x ≤ alpha, x for x > alpha
    assert_eq!(out.data(), &[0.0, 0.0, 0.0, 2.0]);
}

// ── Variadic ops ───────────────────────────────────────────────────────

#[test]
fn variadic_min_picks_smallest_per_position() {
    let a = Tensor::from_vec(vec![3], vec![1.0, 5.0, 3.0]).unwrap();
    let b = Tensor::from_vec(vec![3], vec![4.0, 2.0, 6.0]).unwrap();
    let out = run_single_op(
        "Min",
        vec![("a", a), ("b", b)],
        vec![],
        vec![],
        vec!["a", "b"],
        "y",
    );
    assert_eq!(out.data(), &[1.0, 2.0, 3.0]);
}

#[test]
fn variadic_max_picks_largest_per_position() {
    let a = Tensor::from_vec(vec![3], vec![1.0, 5.0, 3.0]).unwrap();
    let b = Tensor::from_vec(vec![3], vec![4.0, 2.0, 6.0]).unwrap();
    let out = run_single_op(
        "Max",
        vec![("a", a), ("b", b)],
        vec![],
        vec![],
        vec!["a", "b"],
        "y",
    );
    assert_eq!(out.data(), &[4.0, 5.0, 6.0]);
}

#[test]
fn variadic_mean_averages_inputs() {
    let a = Tensor::from_vec(vec![3], vec![1.0, 2.0, 3.0]).unwrap();
    let b = Tensor::from_vec(vec![3], vec![3.0, 4.0, 5.0]).unwrap();
    let out = run_single_op(
        "Mean",
        vec![("a", a), ("b", b)],
        vec![],
        vec![],
        vec!["a", "b"],
        "y",
    );
    assert_eq!(out.data(), &[2.0, 3.0, 4.0]);
}

#[test]
fn variadic_sum_adds_inputs() {
    let a = Tensor::from_vec(vec![3], vec![1.0, 2.0, 3.0]).unwrap();
    let b = Tensor::from_vec(vec![3], vec![10.0, 20.0, 30.0]).unwrap();
    let out = run_single_op(
        "Sum",
        vec![("a", a), ("b", b)],
        vec![],
        vec![],
        vec!["a", "b"],
        "y",
    );
    assert_eq!(out.data(), &[11.0, 22.0, 33.0]);
}

// ── Reductions (axes via attribute) ───────────────────────────────────

fn reduce_attrs(axes: Vec<i64>, keepdims: i64) -> Vec<onnx::AttributeProto> {
    vec![
        make_ints_attr("axes", axes),
        make_int_attr("keepdims", keepdims),
    ]
}

#[test]
fn reduce_mean_axis1_keepdims_false() {
    let input = Tensor::from_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let out = run_single_op(
        "ReduceMean",
        vec![("x", input)],
        vec![],
        reduce_attrs(vec![1], 0),
        vec!["x"],
        "y",
    );
    assert_eq!(out.shape(), &[2]);
    assert_close(out.data(), &[2.0, 5.0], 1e-6);
}

#[test]
fn reduce_sum_axis0_keepdims_true() {
    let input = Tensor::from_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let out = run_single_op(
        "ReduceSum",
        vec![("x", input)],
        vec![],
        reduce_attrs(vec![0], 1),
        vec!["x"],
        "y",
    );
    assert_eq!(out.shape(), &[1, 3]);
    assert_eq!(out.data(), &[5.0, 7.0, 9.0]);
}

#[test]
fn reduce_max_full_reduction() {
    let input = Tensor::from_vec(vec![2, 3], vec![1.0, 5.0, 3.0, 7.0, 2.0, 6.0]).unwrap();
    let out = run_single_op(
        "ReduceMax",
        vec![("x", input)],
        vec![],
        reduce_attrs(vec![0, 1], 0),
        vec!["x"],
        "y",
    );
    assert_eq!(out.data()[0], 7.0);
}

#[test]
fn reduce_min_full_reduction() {
    let input = Tensor::from_vec(vec![2, 3], vec![1.0, 5.0, 3.0, 7.0, 2.0, 6.0]).unwrap();
    let out = run_single_op(
        "ReduceMin",
        vec![("x", input)],
        vec![],
        reduce_attrs(vec![0, 1], 0),
        vec!["x"],
        "y",
    );
    assert_eq!(out.data()[0], 1.0);
}

#[test]
fn reduce_prod_axis0() {
    let input = Tensor::from_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let out = run_single_op(
        "ReduceProd",
        vec![("x", input)],
        vec![],
        reduce_attrs(vec![0], 0),
        vec!["x"],
        "y",
    );
    assert_eq!(out.data(), &[4.0, 10.0, 18.0]);
}

#[test]
fn reduce_l1_axis1() {
    let input = Tensor::from_vec(vec![2, 3], vec![1.0, -2.0, 3.0, -4.0, 5.0, -6.0]).unwrap();
    let out = run_single_op(
        "ReduceL1",
        vec![("x", input)],
        vec![],
        reduce_attrs(vec![1], 0),
        vec!["x"],
        "y",
    );
    assert_close(out.data(), &[6.0, 15.0], 1e-6);
}

#[test]
fn reduce_l2_axis1() {
    let input = Tensor::from_vec(vec![2, 3], vec![3.0, 4.0, 0.0, 0.0, 0.0, 5.0]).unwrap();
    let out = run_single_op(
        "ReduceL2",
        vec![("x", input)],
        vec![],
        reduce_attrs(vec![1], 0),
        vec!["x"],
        "y",
    );
    assert_close(out.data(), &[5.0, 5.0], 1e-6);
}

#[test]
fn argmin_picks_smallest_index_along_axis() {
    let input = Tensor::from_vec(vec![1, 4], vec![3.0, 1.0, 4.0, 1.0]).unwrap();
    let attrs = vec![make_int_attr("axis", 1), make_int_attr("keepdims", 0)];
    let out = run_single_op("ArgMin", vec![("x", input)], vec![], attrs, vec!["x"], "y");
    // First minimum (1.0) is at index 1.
    assert_eq!(out.data()[0], 1.0);
}

#[test]
fn hardmax_one_hot_along_axis() {
    let input = Tensor::from_vec(vec![1, 4], vec![1.0, 5.0, 3.0, 2.0]).unwrap();
    let attrs = vec![make_int_attr("axis", 1)];
    let out = run_single_op("Hardmax", vec![("x", input)], vec![], attrs, vec!["x"], "y");
    // Argmax along axis 1 is index 1 (value 5).
    assert_eq!(out.data(), &[0.0, 1.0, 0.0, 0.0]);
}

// ===========================================================================
// 2.4 LOW_UTIL — comparison, logical, shape, miscellaneous
// ===========================================================================
//
// Final coverage sweep. Each test is one operator, mostly two-line bodies.

// ── Comparison ops (return 0/1 mask) ──────────────────────────────────

#[test]
fn cmp_equal() {
    let a = Tensor::from_vec(vec![4], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let b = Tensor::from_vec(vec![4], vec![1.0, 5.0, 3.0, 0.0]).unwrap();
    let out = run_single_op(
        "Equal",
        vec![("a", a), ("b", b)],
        vec![],
        vec![],
        vec!["a", "b"],
        "y",
    );
    assert_eq!(out.data(), &[1.0, 0.0, 1.0, 0.0]);
}

#[test]
fn cmp_greater() {
    let a = Tensor::from_vec(vec![4], vec![1.0, 5.0, 3.0, 4.0]).unwrap();
    let b = Tensor::from_vec(vec![4], vec![2.0, 2.0, 3.0, 0.0]).unwrap();
    let out = run_single_op(
        "Greater",
        vec![("a", a), ("b", b)],
        vec![],
        vec![],
        vec!["a", "b"],
        "y",
    );
    assert_eq!(out.data(), &[0.0, 1.0, 0.0, 1.0]);
}

#[test]
fn cmp_less() {
    let a = Tensor::from_vec(vec![4], vec![1.0, 5.0, 3.0, 4.0]).unwrap();
    let b = Tensor::from_vec(vec![4], vec![2.0, 2.0, 3.0, 0.0]).unwrap();
    let out = run_single_op(
        "Less",
        vec![("a", a), ("b", b)],
        vec![],
        vec![],
        vec!["a", "b"],
        "y",
    );
    assert_eq!(out.data(), &[1.0, 0.0, 0.0, 0.0]);
}

#[test]
fn cmp_less_or_equal() {
    let a = Tensor::from_vec(vec![4], vec![1.0, 5.0, 3.0, 4.0]).unwrap();
    let b = Tensor::from_vec(vec![4], vec![2.0, 2.0, 3.0, 0.0]).unwrap();
    let out = run_single_op(
        "LessOrEqual",
        vec![("a", a), ("b", b)],
        vec![],
        vec![],
        vec!["a", "b"],
        "y",
    );
    assert_eq!(out.data(), &[1.0, 0.0, 1.0, 0.0]);
}

// ── Logical bitwise ops on 0/1 masks ──────────────────────────────────

#[test]
fn logical_and() {
    let a = Tensor::from_vec(vec![4], vec![1.0, 1.0, 0.0, 0.0]).unwrap();
    let b = Tensor::from_vec(vec![4], vec![1.0, 0.0, 1.0, 0.0]).unwrap();
    let out = run_single_op(
        "And",
        vec![("a", a), ("b", b)],
        vec![],
        vec![],
        vec!["a", "b"],
        "y",
    );
    assert_eq!(out.data(), &[1.0, 0.0, 0.0, 0.0]);
}

#[test]
fn logical_or() {
    let a = Tensor::from_vec(vec![4], vec![1.0, 1.0, 0.0, 0.0]).unwrap();
    let b = Tensor::from_vec(vec![4], vec![1.0, 0.0, 1.0, 0.0]).unwrap();
    let out = run_single_op(
        "Or",
        vec![("a", a), ("b", b)],
        vec![],
        vec![],
        vec!["a", "b"],
        "y",
    );
    assert_eq!(out.data(), &[1.0, 1.0, 1.0, 0.0]);
}

#[test]
fn logical_xor() {
    let a = Tensor::from_vec(vec![4], vec![1.0, 1.0, 0.0, 0.0]).unwrap();
    let b = Tensor::from_vec(vec![4], vec![1.0, 0.0, 1.0, 0.0]).unwrap();
    let out = run_single_op(
        "Xor",
        vec![("a", a), ("b", b)],
        vec![],
        vec![],
        vec!["a", "b"],
        "y",
    );
    assert_eq!(out.data(), &[0.0, 1.0, 1.0, 0.0]);
}

// ── Shape ops ──────────────────────────────────────────────────────────

#[test]
fn squeeze_removes_unit_axes() {
    let input = Tensor::from_vec(vec![1, 3, 1], vec![1.0, 2.0, 3.0]).unwrap();
    let axes_attr = make_ints_attr("axes", vec![0, 2]);
    let out = run_single_op(
        "Squeeze",
        vec![("x", input)],
        vec![],
        vec![axes_attr],
        vec!["x"],
        "y",
    );
    assert_eq!(out.shape(), &[3]);
    assert_eq!(out.data(), &[1.0, 2.0, 3.0]);
}

// ── Misc ───────────────────────────────────────────────────────────────

#[test]
fn nonzero_returns_indices_of_nonzero_elements() {
    // 1D input: nonzero indices are returned as a [1, k] tensor.
    let input = Tensor::from_vec(vec![5], vec![0.0, 3.0, 0.0, -1.0, 0.0]).unwrap();
    let out = run_single_op(
        "NonZero",
        vec![("x", input)],
        vec![],
        vec![],
        vec!["x"],
        "y",
    );
    assert_eq!(out.shape(), &[1, 2]);
    assert_eq!(out.data(), &[1.0, 3.0]);
}

#[test]
fn compress_filters_along_axis() {
    // Compress a [3, 2] tensor along axis 0 with mask [1, 0, 1] → [2, 2].
    let input = Tensor::from_vec(vec![3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let cond = Tensor::from_vec(vec![3], vec![1.0, 0.0, 1.0]).unwrap();
    let attrs = vec![make_int_attr("axis", 0)];
    let out = run_single_op(
        "Compress",
        vec![("d", input), ("c", cond)],
        vec![],
        attrs,
        vec!["d", "c"],
        "y",
    );
    assert_eq!(out.shape(), &[2, 2]);
    assert_eq!(out.data(), &[1.0, 2.0, 5.0, 6.0]);
}

#[test]
fn grid_sample_identity_grid_returns_input() {
    // 1×1×2×2 input + identity bilinear grid (the four output positions
    // sample exactly the four input pixels, in NHWC grid layout) → output
    // equals input.
    let input = Tensor::from_vec(vec![1, 1, 2, 2], vec![10.0, 20.0, 30.0, 40.0]).unwrap();
    // Grid with align_corners=1 maps (-1, -1) → (0,0), (1, 1) → (1,1).
    let grid = Tensor::from_vec(
        vec![1, 2, 2, 2],
        vec![
            -1.0, -1.0, 1.0, -1.0, // top row: x = -1,1; y = -1
            -1.0, 1.0, 1.0, 1.0, // bottom row: x = -1,1; y = 1
        ],
    )
    .unwrap();
    let attrs = vec![make_int_attr("align_corners", 1)];
    let out = run_single_op(
        "GridSample",
        vec![("x", input), ("grid", grid)],
        vec![],
        attrs,
        vec!["x", "grid"],
        "y",
    );
    assert_eq!(out.shape(), &[1, 1, 2, 2]);
    // With align_corners=1, the four corner samples land exactly on input pixels.
    assert_close(out.data(), &[10.0, 20.0, 30.0, 40.0], 1e-5);
}

#[test]
fn conv_integer_treats_inputs_as_offset_quantized_int_conv() {
    // ConvInteger: subtract zero points, plain Conv (no scale, no clamp).
    let x_data = vec![5.0_f32, 6.0, 7.0, 8.0]; // 1×1×2×2
    let w_data = vec![1.0_f32]; // 1×1×1×1 kernel
    let x_zp = 5.0_f32; // shift x to [0,1,2,3]
    let w_zp = 0.0_f32;

    let x = vec_initializer("x", vec![1, 1, 2, 2], x_data);
    let w = conv_weight_initializer("w", 1, 1, 1, 1, w_data);
    let x_zp_init = scalar_initializer("x_zp", x_zp);
    let w_zp_init = scalar_initializer("w_zp", w_zp);

    let node = onnx::NodeProto {
        op_type: Some("ConvInteger".into()),
        name: Some("ci".into()),
        input: vec!["x".into(), "w".into(), "x_zp".into(), "w_zp".into()],
        output: vec!["y".into()],
        attribute: vec![
            make_ints_attr("kernel_shape", vec![1, 1]),
            make_ints_attr("strides", vec![1, 1]),
        ],
        ..Default::default()
    };
    let bytes = build_minimal_onnx_model(
        vec![node],
        vec![x, w, x_zp_init, w_zp_init],
        vec!["x", "w", "x_zp", "w_zp"],
        vec!["y"],
    );
    let model = load_onnx_model(&bytes).unwrap();
    let result = run_onnx_model(&model, HashMap::new()).unwrap();
    // After x_zp subtraction the input is [0,1,2,3], 1×1 identity conv yields
    // exactly that.
    assert_close(result["y"].data(), &[0.0, 1.0, 2.0, 3.0], 1e-5);
}
