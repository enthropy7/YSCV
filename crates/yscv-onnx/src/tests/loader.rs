use super::*;

#[test]
fn load_empty_graph() {
    let bytes = build_minimal_onnx_model(vec![], vec![], vec!["x"], vec!["y"]);
    let model = load_onnx_model(&bytes).unwrap();
    assert_eq!(model.graph_name, "test_graph");
    assert_eq!(model.ir_version, 8);
    assert_eq!(model.opset_version, 13);
    assert_eq!(model.inputs, vec!["x"]);
    assert_eq!(model.outputs, vec!["y"]);
    assert_eq!(model.node_count(), 0);
    assert!(model.initializers.is_empty());
}

#[test]
fn load_float_initializer_via_float_data() {
    let init = onnx::TensorProto {
        name: Some("weight".into()),
        dims: vec![2, 3],
        data_type: Some(1), // FLOAT
        float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        ..Default::default()
    };
    let bytes = build_minimal_onnx_model(vec![], vec![init], vec!["x"], vec!["y"]);
    let model = load_onnx_model(&bytes).unwrap();

    let w = model.get_initializer("weight").unwrap();
    assert_eq!(w.shape(), &[2, 3]);
    assert_eq!(w.data(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn load_float_initializer_via_raw_data() {
    let raw: Vec<u8> = [1.0f32, -2.0, 0.5]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let init = onnx::TensorProto {
        name: Some("bias".into()),
        dims: vec![3],
        data_type: Some(1),
        raw_data: Some(raw),
        ..Default::default()
    };
    let bytes = build_minimal_onnx_model(vec![], vec![init], vec!["x"], vec!["y"]);
    let model = load_onnx_model(&bytes).unwrap();

    let b = model.get_initializer("bias").unwrap();
    assert_eq!(b.shape(), &[3]);
    assert_eq!(b.data(), &[1.0, -2.0, 0.5]);
}

#[test]
fn load_int64_initializer() {
    let init = onnx::TensorProto {
        name: Some("indices".into()),
        dims: vec![4],
        data_type: Some(7), // INT64
        int64_data: vec![10, 20, 30, 40],
        ..Default::default()
    };
    let bytes = build_minimal_onnx_model(vec![], vec![init], vec!["x"], vec!["y"]);
    let model = load_onnx_model(&bytes).unwrap();

    let idx = model.get_initializer("indices").unwrap();
    assert_eq!(idx.data(), &[10.0, 20.0, 30.0, 40.0]);
}

#[test]
fn load_graph_with_relu_node() {
    let node = onnx::NodeProto {
        op_type: Some("Relu".into()),
        name: Some("relu0".into()),
        input: vec!["x".into()],
        output: vec!["y".into()],
        ..Default::default()
    };
    let bytes = build_minimal_onnx_model(vec![node], vec![], vec!["x"], vec!["y"]);
    let model = load_onnx_model(&bytes).unwrap();

    assert_eq!(model.node_count(), 1);
    assert_eq!(model.nodes[0].op_type, "Relu");
    assert_eq!(model.nodes[0].inputs, vec!["x"]);
    assert_eq!(model.nodes[0].outputs, vec!["y"]);
}

#[test]
fn load_node_with_attributes() {
    let node = onnx::NodeProto {
        op_type: Some("Conv".into()),
        name: Some("conv0".into()),
        input: vec!["x".into(), "w".into()],
        output: vec!["y".into()],
        attribute: vec![
            onnx::AttributeProto {
                name: Some("kernel_shape".into()),
                r#type: Some(7), // INTS
                ints: vec![3, 3],
                ..Default::default()
            },
            onnx::AttributeProto {
                name: Some("strides".into()),
                r#type: Some(7),
                ints: vec![1, 1],
                ..Default::default()
            },
            onnx::AttributeProto {
                name: Some("group".into()),
                r#type: Some(2), // INT
                i: Some(1),
                ..Default::default()
            },
        ],
        ..Default::default()
    };
    let bytes = build_minimal_onnx_model(vec![node], vec![], vec!["x", "w"], vec!["y"]);
    let model = load_onnx_model(&bytes).unwrap();

    let conv = &model.nodes[0];
    assert_eq!(conv.op_type, "Conv");
    match &conv.attributes[&Attr::KernelShape] {
        OnnxAttribute::Ints(v) => assert_eq!(v, &[3, 3]),
        _ => panic!("expected Ints attribute"),
    }
    match &conv.attributes[&Attr::Group] {
        OnnxAttribute::Int(v) => assert_eq!(*v, 1),
        _ => panic!("expected Int attribute"),
    }
}

#[test]
fn load_multi_node_graph() {
    let nodes = vec![
        onnx::NodeProto {
            op_type: Some("Conv".into()),
            name: Some("conv0".into()),
            input: vec!["input".into(), "conv_w".into()],
            output: vec!["conv_out".into()],
            ..Default::default()
        },
        onnx::NodeProto {
            op_type: Some("Relu".into()),
            name: Some("relu0".into()),
            input: vec!["conv_out".into()],
            output: vec!["relu_out".into()],
            ..Default::default()
        },
        onnx::NodeProto {
            op_type: Some("AveragePool".into()),
            name: Some("pool0".into()),
            input: vec!["relu_out".into()],
            output: vec!["output".into()],
            ..Default::default()
        },
    ];
    let init = onnx::TensorProto {
        name: Some("conv_w".into()),
        dims: vec![2, 1, 3, 3],
        data_type: Some(1),
        float_data: vec![0.1; 18],
        ..Default::default()
    };
    let bytes =
        build_minimal_onnx_model(nodes, vec![init], vec!["input", "conv_w"], vec!["output"]);
    let model = load_onnx_model(&bytes).unwrap();

    assert_eq!(model.node_count(), 3);
    assert_eq!(model.nodes[0].op_type, "Conv");
    assert_eq!(model.nodes[1].op_type, "Relu");
    assert_eq!(model.nodes[2].op_type, "AveragePool");
    assert!(model.get_initializer("conv_w").is_some());
}

#[test]
fn decode_error_on_invalid_bytes() {
    let result = load_onnx_model(&[0xFF, 0x00, 0x01]);
    assert!(result.is_err());
}

#[test]
fn shape_mismatch_error() {
    let init = onnx::TensorProto {
        name: Some("bad".into()),
        dims: vec![2, 3],
        data_type: Some(1),
        float_data: vec![1.0, 2.0], // 2 elements, expected 6
        ..Default::default()
    };
    let bytes = build_minimal_onnx_model(vec![], vec![init], vec!["x"], vec!["y"]);
    let result = load_onnx_model(&bytes);
    assert!(result.is_err());
}
