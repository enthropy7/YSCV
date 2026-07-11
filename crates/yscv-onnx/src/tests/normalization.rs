use super::*;

#[test]
fn exec_batch_norm_identity() {
    // BN with gamma=1, beta=0, mean=0, var=1 -> identity
    let input = Tensor::from_vec(
        vec![1, 2, 2, 2],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    )
    .unwrap();
    let gamma = onnx::TensorProto {
        name: Some("g".into()),
        dims: vec![2],
        data_type: Some(1),
        float_data: vec![1.0, 1.0],
        ..Default::default()
    };
    let beta = onnx::TensorProto {
        name: Some("b".into()),
        dims: vec![2],
        data_type: Some(1),
        float_data: vec![0.0, 0.0],
        ..Default::default()
    };
    let mean = onnx::TensorProto {
        name: Some("m".into()),
        dims: vec![2],
        data_type: Some(1),
        float_data: vec![0.0, 0.0],
        ..Default::default()
    };
    let var = onnx::TensorProto {
        name: Some("v".into()),
        dims: vec![2],
        data_type: Some(1),
        float_data: vec![1.0, 1.0],
        ..Default::default()
    };

    let node = onnx::NodeProto {
        op_type: Some("BatchNormalization".into()),
        name: Some("bn".into()),
        input: vec!["x".into(), "g".into(), "b".into(), "m".into(), "v".into()],
        output: vec!["y".into()],
        attribute: vec![make_float_attr("epsilon", 1e-5)],
        ..Default::default()
    };
    let bytes = build_minimal_onnx_model(
        vec![node],
        vec![gamma, beta, mean, var],
        vec!["x", "g", "b", "m", "v"],
        vec!["y"],
    );
    let model = load_onnx_model(&bytes).unwrap();
    let mut feed = FxHashMap::default();
    feed.insert("x".to_string(), input.clone());
    let result = run_onnx_model(&model, feed).unwrap();
    let output = &result["y"];
    assert_eq!(output.shape(), &[1, 2, 2, 2]);
    for (a, b) in output.data().iter().zip(input.data().iter()) {
        assert!((a - b).abs() < 1e-4, "BN identity mismatch: {a} vs {b}");
    }
}

#[test]
fn onnx_instance_norm() {
    let input = Tensor::from_vec(vec![1, 1, 2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let scale = Tensor::from_vec(vec![1], vec![1.0]).unwrap();
    let bias = Tensor::from_vec(vec![1], vec![0.0]).unwrap();
    let eps_attr = onnx::AttributeProto {
        name: Some("epsilon".into()),
        r#type: Some(1), // FLOAT
        f: Some(1e-5),
        ..Default::default()
    };
    let out = run_single_op(
        "InstanceNormalization",
        vec![("x", input), ("s", scale), ("b", bias)],
        vec![],
        vec![eps_attr],
        vec!["x", "s", "b"],
        "y",
    );
    assert_eq!(out.shape(), &[1, 1, 2, 2]);
    let d = out.data();
    let mean: f32 = d.iter().sum::<f32>() / d.len() as f32;
    assert!(mean.abs() < 0.01, "instance norm should center to ~0");
}

#[test]
fn onnx_lp_norm_l2() {
    let input = Tensor::from_vec(vec![1, 3], vec![3.0, 4.0, 0.0]).unwrap();
    let axis_attr = onnx::AttributeProto {
        name: Some("axis".into()),
        r#type: Some(2), // INT
        i: Some(1),
        ..Default::default()
    };
    let p_attr = onnx::AttributeProto {
        name: Some("p".into()),
        r#type: Some(2),
        i: Some(2),
        ..Default::default()
    };
    let out = run_single_op(
        "LpNormalization",
        vec![("x", input)],
        vec![],
        vec![axis_attr, p_attr],
        vec!["x"],
        "y",
    );
    let d = out.data();
    let norm: f32 = d.iter().map(|v| v * v).sum::<f32>().sqrt();
    assert!(
        (norm - 1.0).abs() < 1e-4,
        "L2-normalized vector should have unit norm"
    );
}

#[test]
fn onnx_layer_norm() {
    let input = Tensor::from_vec(vec![1, 4], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let scale = Tensor::from_vec(vec![4], vec![1.0, 1.0, 1.0, 1.0]).unwrap();
    let bias = Tensor::from_vec(vec![4], vec![0.0, 0.0, 0.0, 0.0]).unwrap();
    let eps_attr = onnx::AttributeProto {
        name: Some("epsilon".into()),
        r#type: Some(1),
        f: Some(1e-5),
        ..Default::default()
    };
    let axis_attr = onnx::AttributeProto {
        name: Some("axis".into()),
        r#type: Some(2),
        i: Some(-1),
        ..Default::default()
    };
    let out = run_single_op(
        "LayerNormalization",
        vec![("x", input), ("s", scale), ("b", bias)],
        vec![],
        vec![eps_attr, axis_attr],
        vec!["x", "s", "b"],
        "y",
    );
    let mean: f32 = out.data().iter().sum::<f32>() / out.data().len() as f32;
    assert!(mean.abs() < 0.01, "layer norm should center to ~0");
}
