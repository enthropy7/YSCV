use super::*;

#[test]
fn exec_relu_single_op() {
    let input = Tensor::from_vec(vec![1, 2, 2, 1], vec![-1.0, 2.0, -3.0, 4.0]).unwrap();
    let out = run_single_op("Relu", vec![("x", input)], vec![], vec![], vec!["x"], "y");
    assert_eq!(out.data(), &[0.0, 2.0, 0.0, 4.0]);
}

#[test]
fn exec_sigmoid_single_op() {
    let input = Tensor::from_vec(vec![2], vec![0.0, 100.0]).unwrap();
    let out = run_single_op(
        "Sigmoid",
        vec![("x", input)],
        vec![],
        vec![],
        vec!["x"],
        "y",
    );
    assert!((out.data()[0] - 0.5).abs() < 1e-5);
    assert!((out.data()[1] - 1.0).abs() < 1e-3);
}

#[test]
fn exec_where_same_shape() {
    let cond = Tensor::from_vec(vec![4], vec![1.0, 0.0, 1.0, 0.0]).unwrap();
    let x = Tensor::from_vec(vec![4], vec![10.0, 20.0, 30.0, 40.0]).unwrap();
    let y = Tensor::from_vec(vec![4], vec![-1.0, -2.0, -3.0, -4.0]).unwrap();
    let out = run_single_op(
        "Where",
        vec![("cond", cond), ("x", x), ("y", y)],
        vec![],
        vec![],
        vec!["cond", "x", "y"],
        "out",
    );
    assert_eq!(out.data(), &[10.0, -2.0, 30.0, -4.0]);
}

#[test]
fn exec_where_broadcast_mask_to_attention_shape() {
    // Causal-attention masking pattern from HuggingFace exports:
    //   Where(mask_true_on_kept_positions, scores [1,H,L,L], -inf [1])
    // The mask broadcasts from [1,1,L,L] to [1,H,L,L]; the scalar -inf
    // broadcasts from [1] to [1,H,L,L].
    let cond = Tensor::from_vec(
        vec![1, 1, 3, 3],
        // Lower-triangular keep (true = keep the score):
        //   1 0 0
        //   1 1 0
        //   1 1 1
        vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0],
    )
    .unwrap();
    // 2 heads × 3 × 3 attention scores.
    let x_data: Vec<f32> = (0..18).map(|i| i as f32).collect();
    let x = Tensor::from_vec(vec![1, 2, 3, 3], x_data.clone()).unwrap();
    let y = Tensor::from_vec(vec![1], vec![-1e9]).unwrap();
    let out = run_single_op(
        "Where",
        vec![("cond", cond), ("x", x), ("y", y)],
        vec![],
        vec![],
        vec!["cond", "x", "y"],
        "out",
    );
    assert_eq!(out.shape(), &[1, 2, 3, 3]);
    let expected: Vec<f32> = (0..18)
        .map(|i| {
            let pos_in_3x3 = i % 9;
            let row = pos_in_3x3 / 3;
            let col = pos_in_3x3 % 3;
            if col <= row { x_data[i] } else { -1e9 }
        })
        .collect();
    assert_eq!(out.data(), expected);
}

#[test]
fn exec_add_broadcast() {
    let a = Tensor::from_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let b = Tensor::from_vec(vec![3], vec![10.0, 20.0, 30.0]).unwrap();
    let out = run_single_op(
        "Add",
        vec![("a", a), ("b", b)],
        vec![],
        vec![],
        vec!["a", "b"],
        "y",
    );
    assert_eq!(out.data(), &[11.0, 22.0, 33.0, 14.0, 25.0, 36.0]);
}

#[test]
fn exec_clip_min_max() {
    let input = Tensor::from_vec(vec![4], vec![-2.0, 0.5, 3.0, 10.0]).unwrap();
    let min_t = Tensor::scalar(0.0);
    let max_t = Tensor::scalar(6.0);
    let out = run_single_op(
        "Clip",
        vec![("x", input), ("min", min_t), ("max", max_t)],
        vec![],
        vec![],
        vec!["x", "min", "max"],
        "y",
    );
    assert_eq!(out.data(), &[0.0, 0.5, 3.0, 6.0]);
}

#[test]
fn exec_dropout_passthrough() {
    let input = Tensor::from_vec(vec![3], vec![1.0, 2.0, 3.0]).unwrap();
    let out = run_single_op(
        "Dropout",
        vec![("x", input.clone())],
        vec![],
        vec![],
        vec!["x"],
        "y",
    );
    assert_eq!(out.data(), input.data());
}

#[test]
fn leaky_relu_op() {
    let alpha_attr = onnx::AttributeProto {
        name: Some("alpha".into()),
        r#type: Some(1), // FLOAT
        f: Some(0.1),
        ..Default::default()
    };
    let input = Tensor::from_vec(vec![4], vec![1.0, -2.0, 0.0, 3.0]).unwrap();
    let out = run_single_op(
        "LeakyRelu",
        vec![("x", input)],
        vec![],
        vec![alpha_attr],
        vec!["x"],
        "y",
    );
    let d = out.data();
    assert!((d[0] - 1.0).abs() < 1e-5);
    assert!((d[1] - (-0.2)).abs() < 1e-5);
    assert!((d[2] - 0.0).abs() < 1e-5);
    assert!((d[3] - 3.0).abs() < 1e-5);
}

#[test]
fn identity_op() {
    let input = Tensor::from_vec(vec![3], vec![1.0, 2.0, 3.0]).unwrap();
    let out = run_single_op(
        "Identity",
        vec![("x", input.clone())],
        vec![],
        vec![],
        vec!["x"],
        "y",
    );
    assert_eq!(out.data(), input.data());
}

#[test]
fn elu_op() {
    let input = Tensor::from_vec(vec![3], vec![1.0, -1.0, 0.0]).unwrap();
    let out = run_single_op("Elu", vec![("x", input)], vec![], vec![], vec!["x"], "y");
    let d = out.data();
    assert!((d[0] - 1.0).abs() < 1e-5);
    assert!((d[1] - ((-1.0f32).exp() - 1.0)).abs() < 1e-5);
    assert!((d[2] - 0.0).abs() < 1e-5);
}

#[test]
fn onnx_gelu() {
    let out = run_single_op(
        "Gelu",
        vec![(
            "x",
            Tensor::from_vec(vec![3], vec![0.0, 1.0, -1.0]).unwrap(),
        )],
        vec![],
        vec![],
        vec!["x"],
        "y",
    );
    let d = out.data();
    assert!(d[0].abs() < 1e-5, "gelu(0) ~ 0");
    assert!((d[1] - 0.8413).abs() < 0.01, "gelu(1) ~ 0.8413");
}

#[test]
fn onnx_erf() {
    let out = run_single_op(
        "Erf",
        vec![("x", Tensor::from_vec(vec![2], vec![0.0, 1.0]).unwrap())],
        vec![],
        vec![],
        vec!["x"],
        "y",
    );
    let d = out.data();
    assert!(d[0].abs() < 1e-5, "erf(0) = 0");
    assert!((d[1] - 0.8427).abs() < 0.01, "erf(1) ~ 0.8427");
}

#[test]
fn onnx_hard_sigmoid() {
    let out = run_single_op(
        "HardSigmoid",
        vec![(
            "x",
            Tensor::from_vec(vec![3], vec![-5.0, 0.0, 5.0]).unwrap(),
        )],
        vec![],
        vec![],
        vec!["x"],
        "y",
    );
    let d = out.data();
    assert!(d[0].abs() < 1e-5);
    assert!((d[1] - 0.5).abs() < 1e-5);
    assert!((d[2] - 1.0).abs() < 1e-5);
}

#[test]
fn onnx_selu() {
    let out = run_single_op(
        "Selu",
        vec![("x", Tensor::from_vec(vec![2], vec![1.0, -1.0]).unwrap())],
        vec![],
        vec![],
        vec!["x"],
        "y",
    );
    assert!(out.data()[0] > 0.0, "selu(1) > 0");
    assert!(out.data()[1] < 0.0, "selu(-1) < 0");
}

#[test]
fn onnx_not() {
    let input = Tensor::from_vec(vec![3], vec![1.0, 0.0, 1.0]).unwrap();
    let out = run_single_op("Not", vec![("x", input)], vec![], vec![], vec!["x"], "y");
    assert_eq!(out.data(), &[0.0, 1.0, 0.0]);
}

#[test]
fn onnx_sin_cos() {
    let input = Tensor::from_vec(vec![2], vec![0.0, std::f32::consts::PI / 2.0]).unwrap();
    let sin_out = run_single_op(
        "Sin",
        vec![("x", input.clone())],
        vec![],
        vec![],
        vec!["x"],
        "y",
    );
    assert!((sin_out.data()[0]).abs() < 0.01);
    assert!((sin_out.data()[1] - 1.0).abs() < 0.01);

    let cos_out = run_single_op("Cos", vec![("x", input)], vec![], vec![], vec!["x"], "y");
    assert!((cos_out.data()[0] - 1.0).abs() < 0.01);
    assert!((cos_out.data()[1]).abs() < 0.01);
}

#[test]
fn onnx_greater_or_equal() {
    let a = Tensor::from_vec(vec![3], vec![1.0, 2.0, 3.0]).unwrap();
    let b = Tensor::from_vec(vec![3], vec![2.0, 2.0, 1.0]).unwrap();
    let out = run_single_op(
        "GreaterOrEqual",
        vec![("a", a), ("b", b)],
        vec![],
        vec![],
        vec!["a", "b"],
        "y",
    );
    assert_eq!(out.data(), &[0.0, 1.0, 1.0]);
}

#[test]
fn onnx_constant_of_shape() {
    let shape = Tensor::from_vec(vec![2], vec![3.0, 4.0]).unwrap();
    let out = run_single_op(
        "ConstantOfShape",
        vec![("shape", shape)],
        vec![],
        vec![onnx::AttributeProto {
            name: Some("value".into()),
            r#type: Some(1),
            f: Some(5.0),
            ..Default::default()
        }],
        vec!["shape"],
        "y",
    );
    assert_eq!(out.shape(), &[3, 4]);
    assert!(out.data().iter().all(|&v| (v - 5.0).abs() < 0.01));
}

#[test]
fn onnx_softplus() {
    let input = Tensor::from_vec(vec![2], vec![0.0, 1.0]).unwrap();
    let out = run_single_op(
        "Softplus",
        vec![("x", input)],
        vec![],
        vec![],
        vec!["x"],
        "y",
    );
    assert!((out.data()[0] - std::f32::consts::LN_2).abs() < 0.01);
}

#[test]
fn onnx_hard_swish() {
    let input = Tensor::from_vec(vec![3], vec![-4.0, 0.0, 4.0]).unwrap();
    let out = run_single_op(
        "HardSwish",
        vec![("x", input)],
        vec![],
        vec![],
        vec!["x"],
        "y",
    );
    assert_eq!(out.data()[0], 0.0);
    assert_eq!(out.data()[1], 0.0);
    assert_eq!(out.data()[2], 4.0);
}
