use super::*;

/// Build a small CNN graph: Conv -> Relu -> GlobalAvgPool -> Flatten -> Gemm
/// that accepts variable batch sizes (the graph itself is batch-agnostic).
fn build_dynamic_batch_cnn() -> Vec<u8> {
    // Conv weight [O=2, I=1, KH=3, KW=3]
    let conv_w = onnx::TensorProto {
        name: Some("conv_w".into()),
        dims: vec![2, 1, 3, 3],
        data_type: Some(1),
        float_data: vec![0.1f32; 18],
        ..Default::default()
    };
    // Gemm weight [num_classes=3, features=2]
    let fc_w = onnx::TensorProto {
        name: Some("fc_w".into()),
        dims: vec![3, 2],
        data_type: Some(1),
        float_data: vec![1.0, -1.0, 0.5, 0.5, -0.5, 1.0],
        ..Default::default()
    };
    // Gemm bias [3]
    let fc_b = onnx::TensorProto {
        name: Some("fc_b".into()),
        dims: vec![3],
        data_type: Some(1),
        float_data: vec![0.1, 0.2, 0.3],
        ..Default::default()
    };

    let nodes = vec![
        onnx::NodeProto {
            op_type: Some("Conv".into()),
            name: Some("conv0".into()),
            input: vec!["input".into(), "conv_w".into()],
            output: vec!["conv_out".into()],
            attribute: vec![
                make_ints_attr("kernel_shape", vec![3, 3]),
                make_ints_attr("strides", vec![1, 1]),
            ],
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
            op_type: Some("GlobalAveragePool".into()),
            name: Some("gap".into()),
            input: vec!["relu_out".into()],
            output: vec!["gap_out".into()],
            ..Default::default()
        },
        onnx::NodeProto {
            op_type: Some("Flatten".into()),
            name: Some("flat".into()),
            input: vec!["gap_out".into()],
            output: vec!["flat_out".into()],
            attribute: vec![make_int_attr("axis", 1)],
            ..Default::default()
        },
        onnx::NodeProto {
            op_type: Some("Gemm".into()),
            name: Some("fc".into()),
            input: vec!["flat_out".into(), "fc_w".into(), "fc_b".into()],
            output: vec!["output".into()],
            attribute: vec![make_int_attr("transB", 1)],
            ..Default::default()
        },
    ];

    build_minimal_onnx_model(
        nodes,
        vec![conv_w, fc_w, fc_b],
        vec!["input", "conv_w", "fc_w", "fc_b"],
        vec!["output"],
    )
}

/// Run the dynamic-batch CNN with a given batch size and return output tensor.
fn run_dynamic_cnn(model: &crate::loader::OnnxModel, batch: usize) -> Tensor {
    // Input: [batch, 1, 4, 4]
    let input_data: Vec<f32> = (0..batch * 16).map(|v| (v % 16) as f32 / 16.0).collect();
    let input = Tensor::from_vec(vec![batch, 1, 4, 4], input_data).unwrap();
    let mut feed = FxHashMap::default();
    feed.insert("input".to_string(), input);
    let result = run_onnx_model(model, feed).unwrap();
    result["output"].clone()
}

#[test]
fn dynamic_batch_cnn_batch1_and_batch4() {
    let bytes = build_dynamic_batch_cnn();
    let model = load_onnx_model(&bytes).unwrap();

    // Run with batch=1
    let out1 = run_dynamic_cnn(&model, 1);
    assert_eq!(out1.shape(), &[1, 3]);
    for v in out1.data() {
        assert!(v.is_finite(), "batch=1 output must be finite");
    }

    // Run with batch=4
    let out4 = run_dynamic_cnn(&model, 4);
    assert_eq!(out4.shape(), &[4, 3]);
    for v in out4.data() {
        assert!(v.is_finite(), "batch=4 output must be finite");
    }

    // Since each batch element uses the same input pattern,
    // every row in batch=4 output should match the batch=1 output.
    let ref_row = &out1.data()[..3];
    for b in 0..4 {
        let row = &out4.data()[b * 3..(b + 1) * 3];
        for (a, e) in row.iter().zip(ref_row.iter()) {
            assert!(
                (a - e).abs() < 1e-5,
                "batch element {b} mismatch: {a} vs {e}"
            );
        }
    }
}

#[test]
fn dynamic_batch_conv_bn_pool() {
    // Conv -> BatchNorm -> MaxPool with variable batch
    let conv_w = onnx::TensorProto {
        name: Some("conv_w".into()),
        dims: vec![2, 1, 3, 3],
        data_type: Some(1),
        float_data: vec![0.1f32; 18],
        ..Default::default()
    };
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

    let nodes = vec![
        onnx::NodeProto {
            op_type: Some("Conv".into()),
            name: Some("conv".into()),
            input: vec!["x".into(), "conv_w".into()],
            output: vec!["conv_out".into()],
            attribute: vec![
                make_ints_attr("kernel_shape", vec![3, 3]),
                make_ints_attr("strides", vec![1, 1]),
                make_ints_attr("pads", vec![1, 1, 1, 1]),
            ],
            ..Default::default()
        },
        onnx::NodeProto {
            op_type: Some("BatchNormalization".into()),
            name: Some("bn".into()),
            input: vec![
                "conv_out".into(),
                "g".into(),
                "b".into(),
                "m".into(),
                "v".into(),
            ],
            output: vec!["bn_out".into()],
            attribute: vec![make_float_attr("epsilon", 1e-5)],
            ..Default::default()
        },
        onnx::NodeProto {
            op_type: Some("MaxPool".into()),
            name: Some("pool".into()),
            input: vec!["bn_out".into()],
            output: vec!["y".into()],
            attribute: vec![
                make_ints_attr("kernel_shape", vec![2, 2]),
                make_ints_attr("strides", vec![2, 2]),
            ],
            ..Default::default()
        },
    ];

    let bytes = build_minimal_onnx_model(
        nodes,
        vec![conv_w, gamma, beta, mean, var],
        vec!["x", "conv_w", "g", "b", "m", "v"],
        vec!["y"],
    );
    let model = load_onnx_model(&bytes).unwrap();

    // batch=1: input [1,1,4,4] -> conv same-pad [1,2,4,4] -> bn [1,2,4,4] -> pool [1,2,2,2]
    let input1 = Tensor::from_vec(vec![1, 1, 4, 4], (0..16).map(|i| i as f32).collect()).unwrap();
    let mut feed1 = FxHashMap::default();
    feed1.insert("x".to_string(), input1);
    let r1 = run_onnx_model(&model, feed1).unwrap();
    let y1 = &r1["y"];
    assert_eq!(y1.shape(), &[1, 2, 2, 2]);

    // batch=3
    let input3 =
        Tensor::from_vec(vec![3, 1, 4, 4], (0..48).map(|i| (i % 16) as f32).collect()).unwrap();
    let mut feed3 = FxHashMap::default();
    feed3.insert("x".to_string(), input3);
    let r3 = run_onnx_model(&model, feed3).unwrap();
    let y3 = &r3["y"];
    assert_eq!(y3.shape(), &[3, 2, 2, 2]);

    // Each batch element should match batch=1 result
    let n_elem = 2 * 2 * 2;
    let ref_data = y1.data();
    for b in 0..3 {
        let slice = &y3.data()[b * n_elem..(b + 1) * n_elem];
        for (a, e) in slice.iter().zip(ref_data.iter()) {
            assert!(
                (a - e).abs() < 1e-5,
                "batch {b} conv-bn-pool mismatch: {a} vs {e}"
            );
        }
    }
}

#[test]
fn reshape_neg1_dynamic_batch() {
    // Reshape [N, 3, 2] -> [N, 6] using shape tensor [0, -1]
    // The 0 preserves the input dim, -1 infers the rest.
    for batch in [1, 2, 5] {
        let total = batch * 6;
        let input =
            Tensor::from_vec(vec![batch, 3, 2], (0..total).map(|i| i as f32).collect()).unwrap();
        let shape = Tensor::from_vec(vec![2], vec![0.0, -1.0]).unwrap();
        let out = run_single_op(
            "Reshape",
            vec![("x", input.clone())],
            vec![("s", shape)],
            vec![],
            vec!["x", "s"],
            "y",
        );
        assert_eq!(out.shape(), &[batch, 6]);
        assert_eq!(out.data(), input.data());
    }
}

#[test]
fn reshape_neg1_infer_batch_dim() {
    // Reshape [12] -> [-1, 3] should give [4, 3]
    let input = Tensor::from_vec(vec![12], (0..12).map(|i| i as f32).collect()).unwrap();
    let shape = Tensor::from_vec(vec![2], vec![-1.0, 3.0]).unwrap();
    let out = run_single_op(
        "Reshape",
        vec![("x", input)],
        vec![("s", shape)],
        vec![],
        vec!["x", "s"],
        "y",
    );
    assert_eq!(out.shape(), &[4, 3]);
}

#[test]
fn reshape_zero_preserves_dim() {
    // Reshape [2, 3, 4] with shape [0, -1] -> [2, 12]
    let input = Tensor::from_vec(vec![2, 3, 4], vec![0.0; 24]).unwrap();
    let shape = Tensor::from_vec(vec![2], vec![0.0, -1.0]).unwrap();
    let out = run_single_op(
        "Reshape",
        vec![("x", input)],
        vec![("s", shape)],
        vec![],
        vec!["x", "s"],
        "y",
    );
    assert_eq!(out.shape(), &[2, 12]);
}

#[test]
fn shape_op_with_dynamic_batch() {
    // Shape op should report the actual runtime shape
    for batch in [1, 7] {
        let input = Tensor::from_vec(vec![batch, 3, 4, 4], vec![0.0; batch * 48]).unwrap();
        let out = run_single_op("Shape", vec![("x", input)], vec![], vec![], vec!["x"], "y");
        assert_eq!(out.shape(), &[4]);
        assert_eq!(
            out.data(),
            &[batch as f32, 3.0, 4.0, 4.0],
            "Shape op must reflect actual batch size"
        );
    }
}

#[test]
fn flatten_dynamic_batch() {
    for batch in [1, 3, 8] {
        let input = Tensor::from_vec(vec![batch, 2, 3, 4], vec![1.0; batch * 24]).unwrap();
        let out = run_single_op(
            "Flatten",
            vec![("x", input)],
            vec![],
            vec![make_int_attr("axis", 1)],
            vec!["x"],
            "y",
        );
        assert_eq!(out.shape(), &[batch, 24]);
    }
}

#[test]
fn concat_dynamic_batch() {
    // Concat along axis=1 with varying batch
    for batch in [1, 4] {
        let a = Tensor::from_vec(vec![batch, 2], vec![1.0; batch * 2]).unwrap();
        let b = Tensor::from_vec(vec![batch, 3], vec![2.0; batch * 3]).unwrap();
        let out = run_single_op(
            "Concat",
            vec![("a", a), ("b", b)],
            vec![],
            vec![make_int_attr("axis", 1)],
            vec!["a", "b"],
            "y",
        );
        assert_eq!(out.shape(), &[batch, 5]);
    }
}
