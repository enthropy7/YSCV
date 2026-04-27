use super::*;

use crate::quantize::{CalibrationCollector, calibrate::test_lock, rewrite_to_qdq};

/// Build a tiny Conv → Relu → Conv graph and run it inside a calibration
/// scope. Verify that the runner's `TensorEnv::insert*` hooks fire and the
/// collector observes activations for the input, the intermediate
/// `relu_out`, and the final output.
#[test]
fn collector_captures_runner_activations() {
    let _guard = test_lock().lock().unwrap_or_else(|e| e.into_inner());

    // Conv0: 1 -> 1 channel, 1x1 kernel, stride 1, no padding. Identity-ish
    // so we can predict the activation min/max from the input.
    let conv0_w = onnx::TensorProto {
        name: Some("conv0_w".into()),
        dims: vec![1, 1, 1, 1],
        data_type: Some(1),
        float_data: vec![2.0], // multiplies input by 2
        ..Default::default()
    };
    let conv1_w = onnx::TensorProto {
        name: Some("conv1_w".into()),
        dims: vec![1, 1, 1, 1],
        data_type: Some(1),
        float_data: vec![3.0], // multiplies relu_out by 3
        ..Default::default()
    };

    let nodes = vec![
        onnx::NodeProto {
            op_type: Some("Conv".into()),
            name: Some("conv0".into()),
            input: vec!["calib_test_input_xyz".into(), "conv0_w".into()],
            output: vec!["conv0_out".into()],
            attribute: vec![
                make_ints_attr("kernel_shape", vec![1, 1]),
                make_ints_attr("strides", vec![1, 1]),
            ],
            ..Default::default()
        },
        onnx::NodeProto {
            op_type: Some("Relu".into()),
            name: Some("relu0".into()),
            input: vec!["conv0_out".into()],
            output: vec!["calib_test_relu_out_xyz".into()],
            ..Default::default()
        },
        onnx::NodeProto {
            op_type: Some("Conv".into()),
            name: Some("conv1".into()),
            input: vec!["calib_test_relu_out_xyz".into(), "conv1_w".into()],
            output: vec!["calib_test_output_xyz".into()],
            attribute: vec![
                make_ints_attr("kernel_shape", vec![1, 1]),
                make_ints_attr("strides", vec![1, 1]),
            ],
            ..Default::default()
        },
    ];

    let bytes = build_minimal_onnx_model(
        nodes,
        vec![conv0_w, conv1_w],
        vec!["calib_test_input_xyz", "conv0_w", "conv1_w"],
        vec!["calib_test_output_xyz"],
    );
    let model = load_onnx_model(&bytes).unwrap();

    // Input: 4 values [-1, 0, 1, 2] in NCHW [1,1,2,2]. Pre-Relu Conv0 doubles
    // them: [-2, 0, 2, 4]. Relu clips negatives: [0, 0, 2, 4]. Conv1 triples:
    // [0, 0, 6, 12]. Expected min/max:
    //   input:     min=-1, max=2
    //   relu_out:  min=0,  max=4
    //   output:    min=0,  max=12
    let input = Tensor::from_vec(vec![1, 1, 2, 2], vec![-1.0, 0.0, 1.0, 2.0]).unwrap();
    let mut feed = HashMap::new();
    feed.insert("calib_test_input_xyz".to_string(), input);

    let collector = CalibrationCollector::new();
    {
        let _scope = collector.scope();
        let result = run_onnx_model(&model, feed).unwrap();
        let out = &result["calib_test_output_xyz"];
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        // Verify the forward pass before checking calibration data.
        let data = out.data();
        assert!((data[0] - 0.0).abs() < 1e-5);
        assert!((data[1] - 0.0).abs() < 1e-5);
        assert!((data[2] - 6.0).abs() < 1e-5);
        assert!((data[3] - 12.0).abs() < 1e-5);
    }

    let snap = collector.snapshot();
    assert!(
        !snap.is_empty(),
        "calibration snapshot should not be empty after a forward pass"
    );

    // Intermediate `relu_out` is the most reliable check: its name
    // appears in the static slot map and survives any layout passes.
    let relu = snap
        .get("calib_test_relu_out_xyz")
        .copied()
        .expect("relu_out activation should be captured");
    assert_eq!(relu.min, 0.0);
    assert_eq!(relu.max, 4.0);
    // Count may be a multiple of 4 if layout conversion (NHWC ↔ NCHW)
    // re-inserts the tensor under the same name. min/max are layout-
    // invariant, so they remain correct regardless.
    assert!(
        relu.count >= 4 && relu.count.is_multiple_of(4),
        "relu_out count = {}, expected positive multiple of 4",
        relu.count
    );

    let out = snap
        .get("calib_test_output_xyz")
        .copied()
        .expect("output activation should be captured");
    assert_eq!(out.min, 0.0);
    assert_eq!(out.max, 12.0);
    assert!(
        out.count >= 4 && out.count.is_multiple_of(4),
        "output count = {}, expected positive multiple of 4",
        out.count
    );
}

/// Verify that no recording happens when no scope is active, even if the
/// static collector is briefly bound by a different test.
#[test]
fn no_recording_outside_scope() {
    let _guard = test_lock().lock().unwrap_or_else(|e| e.into_inner());

    let conv_w = onnx::TensorProto {
        name: Some("conv_w".into()),
        dims: vec![1, 1, 1, 1],
        data_type: Some(1),
        float_data: vec![1.0],
        ..Default::default()
    };
    let nodes = vec![onnx::NodeProto {
        op_type: Some("Conv".into()),
        name: Some("conv0".into()),
        input: vec!["calib_test_input_xyz".into(), "conv_w".into()],
        output: vec!["calib_test_output_xyz".into()],
        attribute: vec![
            make_ints_attr("kernel_shape", vec![1, 1]),
            make_ints_attr("strides", vec![1, 1]),
        ],
        ..Default::default()
    }];
    let bytes = build_minimal_onnx_model(
        nodes,
        vec![conv_w],
        vec!["calib_test_input_xyz", "conv_w"],
        vec!["calib_test_output_xyz"],
    );
    let model = load_onnx_model(&bytes).unwrap();
    let input = Tensor::from_vec(vec![1, 1, 1, 1], vec![1.0_f32]).unwrap();
    let mut feed = HashMap::new();
    feed.insert("calib_test_input_xyz".to_string(), input);

    let collector = CalibrationCollector::new();
    // No scope() — runner should bypass the hook entirely.
    let _ = run_onnx_model(&model, feed).unwrap();
    assert!(
        collector.is_empty(),
        "no scope active -> snapshot must stay empty"
    );
}

/// End-to-end PTQ smoke test: build a small Conv graph with a 32-element
/// weight (above the rewriter's 16-element threshold), run it fp32 to
/// capture activation stats, rewrite the model to QDQ format, and re-run.
/// The rewritten model must execute and produce numerically close output
/// to the fp32 reference (loose tolerance — quantization is lossy).
#[test]
fn rewrite_to_qdq_round_trip_keeps_model_runnable() {
    let _guard = test_lock().lock().unwrap_or_else(|e| e.into_inner());

    // Conv: 4-in -> 8-out, 1x1 kernel. Weight = 8*4*1*1 = 32 elements,
    // > 16 so the rewriter quantizes it.
    let weight_data: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) * 0.05).collect();
    let conv_w = onnx::TensorProto {
        name: Some("w".into()),
        dims: vec![8, 4, 1, 1],
        data_type: Some(1),
        float_data: weight_data.clone(),
        ..Default::default()
    };
    let nodes = vec![onnx::NodeProto {
        op_type: Some("Conv".into()),
        name: Some("conv0".into()),
        input: vec!["x".into(), "w".into()],
        output: vec!["y".into()],
        attribute: vec![
            make_ints_attr("kernel_shape", vec![1, 1]),
            make_ints_attr("strides", vec![1, 1]),
        ],
        ..Default::default()
    }];
    let bytes = build_minimal_onnx_model(nodes, vec![conv_w], vec!["x", "w"], vec!["y"]);
    let model_fp32 = load_onnx_model(&bytes).unwrap();

    let input_data: Vec<f32> = (0..16).map(|i| (i as f32 - 8.0) * 0.25).collect();
    let input = Tensor::from_vec(vec![1, 4, 2, 2], input_data).unwrap();

    // 1) Reference fp32 run.
    let mut feed_fp32 = HashMap::new();
    feed_fp32.insert("x".to_string(), input.clone());
    let result_fp32 = run_onnx_model(&model_fp32, feed_fp32).unwrap();
    let y_fp32 = result_fp32["y"].clone();

    // 2) Calibrate on the same input (single-sample is enough for a smoke
    //    test — real calibration uses many).
    let collector = CalibrationCollector::new();
    {
        let _scope = collector.scope();
        let mut feed_cal = HashMap::new();
        feed_cal.insert("x".to_string(), input.clone());
        let _ = run_onnx_model(&model_fp32, feed_cal).unwrap();
    }
    let stats = collector.snapshot();
    assert!(
        stats.contains_key("x"),
        "calibration must capture input activation"
    );

    // 3) Rewrite the model with collected stats. Reload from bytes so we
    //    get a fresh, mutable copy.
    let mut model_qdq = load_onnx_model(&bytes).unwrap();
    rewrite_to_qdq(&mut model_qdq, &stats).unwrap();

    // Sanity: rewritten model has Q+DQ on x and DQ on w.
    assert!(
        model_qdq
            .nodes
            .iter()
            .any(|n| n.op_type == "QuantizeLinear"),
        "rewritten model must contain QuantizeLinear"
    );
    assert!(
        model_qdq.initializers.contains_key("w_q"),
        "rewritten model must contain quantized weight initializer"
    );

    // 4) Run rewritten model.
    let mut feed_qdq = HashMap::new();
    feed_qdq.insert("x".to_string(), input);
    let result_qdq = run_onnx_model(&model_qdq, feed_qdq).unwrap();
    let y_qdq = &result_qdq["y"];
    assert_eq!(y_qdq.shape(), y_fp32.shape());

    // 5) Compare. Loose tolerance: per-channel int8 quantization on a
    //    32-element weight + per-tensor int8 activation gives drift in
    //    the few-percent range. Check max abs diff stays below a
    //    fraction of the fp32 output's dynamic range, not bitwise.
    let fp32_data = y_fp32.data();
    let qdq_data = y_qdq.data();
    let abs_max_fp32 = fp32_data.iter().fold(0.0_f32, |m, &v| m.max(v.abs()));
    let max_abs_diff = fp32_data
        .iter()
        .zip(qdq_data.iter())
        .map(|(&a, &b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    let rel = if abs_max_fp32 > 0.0 {
        max_abs_diff / abs_max_fp32
    } else {
        0.0
    };
    assert!(
        rel < 0.10,
        "QDQ output drifted {rel:.3} (max_abs={max_abs_diff}, fp32_abs_max={abs_max_fp32})"
    );
}
