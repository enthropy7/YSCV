use super::*;

#[test]
fn exec_conv_with_padding() {
    // Conv with pads=[1,1,1,1] to produce same-size output
    // Input: [1, 1, 3, 3], kernel 3x3 stride 1 pad 1 -> [1, 1, 3, 3]
    let conv_w_data = vec![0.0f32; 9];
    let mut cw = conv_w_data;
    cw[4] = 1.0; // center element = 1 -> identity conv
    let conv_w = onnx::TensorProto {
        name: Some("w".into()),
        dims: vec![1, 1, 3, 3],
        data_type: Some(1),
        float_data: cw,
        ..Default::default()
    };
    let node = onnx::NodeProto {
        op_type: Some("Conv".into()),
        name: Some("conv".into()),
        input: vec!["x".into(), "w".into()],
        output: vec!["y".into()],
        attribute: vec![
            make_ints_attr("kernel_shape", vec![3, 3]),
            make_ints_attr("strides", vec![1, 1]),
            make_ints_attr("pads", vec![1, 1, 1, 1]),
        ],
        ..Default::default()
    };
    let bytes = build_minimal_onnx_model(vec![node], vec![conv_w], vec!["x", "w"], vec!["y"]);
    let model = load_onnx_model(&bytes).unwrap();

    let input = Tensor::from_vec(
        vec![1, 1, 3, 3],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
    )
    .unwrap();
    let mut feed = HashMap::new();
    feed.insert("x".to_string(), input.clone());
    let result = run_onnx_model(&model, feed).unwrap();
    let output = &result["y"];
    assert_eq!(output.shape(), &[1, 1, 3, 3]);
    // center pixel should be preserved by identity conv
    assert!((output.data()[4] - 5.0).abs() < 1e-5);
}

#[test]
fn dispatch_records_depthwise_padded_kernel() {
    // group == channels, 3×3 with SAME padding: routes to the depthwise
    // NHWC padded path. Deterministic and arch-stable — the NCHWc-DW path
    // needs YSCV_NCHWC_DW + channels%8, and the aarch64 indirect path is
    // group=1 only, so this lands on `dw-nhwc-padded` on every target.
    let c = 2usize;
    let conv_w = onnx::TensorProto {
        name: Some("w".into()),
        dims: vec![c as i64, 1, 3, 3],
        data_type: Some(1),
        float_data: vec![0.1f32; c * 9],
        ..Default::default()
    };
    let node = onnx::NodeProto {
        op_type: Some("Conv".into()),
        name: Some("dw".into()),
        input: vec!["x".into(), "w".into()],
        output: vec!["y".into()],
        attribute: vec![
            make_ints_attr("kernel_shape", vec![3, 3]),
            make_ints_attr("strides", vec![1, 1]),
            make_ints_attr("pads", vec![1, 1, 1, 1]),
            make_int_attr("group", c as i64),
        ],
        ..Default::default()
    };
    let bytes = build_minimal_onnx_model(vec![node], vec![conv_w], vec!["x", "w"], vec!["y"]);
    let model = load_onnx_model(&bytes).unwrap();
    let input = Tensor::from_vec(vec![1, c, 3, 3], vec![1.0f32; c * 9]).unwrap();
    let mut feed = HashMap::new();
    feed.insert("x".to_string(), input);
    run_onnx_model(&model, feed).unwrap();

    // c=2 (< 4 channels) forces the depthwise scalar row kernel, so the label
    // combines the runner-level dispatch with the kernel-internal sub-path.
    assert_eq!(
        crate::runner::take_conv_kernel()
            .map(|k| k.label())
            .as_deref(),
        Some("dw-nhwc-padded/dw-scalar"),
    );
}

#[test]
fn dispatch_records_pointwise_gemm_kernel() {
    // 1×1 group=1 conv, no padding: a dense NHWC GEMM path. 1×1 never
    // routes to the aarch64 indirect 3×3 kernel, so the label is one of
    // the GEMM variants on every arch (prepacked depending on the index).
    let (o, i) = (3usize, 2usize);
    let conv_w = onnx::TensorProto {
        name: Some("w".into()),
        dims: vec![o as i64, i as i64, 1, 1],
        data_type: Some(1),
        float_data: vec![0.1f32; o * i],
        ..Default::default()
    };
    let node = onnx::NodeProto {
        op_type: Some("Conv".into()),
        name: Some("pw".into()),
        input: vec!["x".into(), "w".into()],
        output: vec!["y".into()],
        attribute: vec![
            make_ints_attr("kernel_shape", vec![1, 1]),
            make_ints_attr("strides", vec![1, 1]),
        ],
        ..Default::default()
    };
    let bytes = build_minimal_onnx_model(vec![node], vec![conv_w], vec!["x", "w"], vec!["y"]);
    let model = load_onnx_model(&bytes).unwrap();
    let input = Tensor::from_vec(vec![1, i, 2, 2], vec![1.0f32; i * 4]).unwrap();
    let mut feed = HashMap::new();
    feed.insert("x".to_string(), input);
    run_onnx_model(&model, feed).unwrap();

    // Runner-level dispatch is a dense GEMM (prepacked or not); the kernel
    // body reports the pointwise sub-path, so the label is `runner/pw-gemm`.
    let label = crate::runner::take_conv_kernel().map(|k| k.label());
    assert!(
        matches!(
            label.as_deref(),
            Some("nhwc-gemm/pw-gemm" | "nhwc-gemm-prepacked/pw-gemm")
        ),
        "expected a dense pointwise GEMM path, got {label:?}",
    );
}

#[test]
fn dispatch_records_matmul_kernel() {
    // A plain 2-D MatMul routes through matmul_2d_slices_fused_maybe_packed,
    // which records the GEMM family it dispatched to.
    let b_w = onnx::TensorProto {
        name: Some("w".into()),
        dims: vec![64, 48],
        data_type: Some(1),
        float_data: vec![0.02f32; 64 * 48],
        ..Default::default()
    };
    let node = onnx::NodeProto {
        op_type: Some("MatMul".into()),
        name: Some("mm".into()),
        input: vec!["x".into(), "w".into()],
        output: vec!["y".into()],
        ..Default::default()
    };
    let bytes = build_minimal_onnx_model(vec![node], vec![b_w], vec!["x", "w"], vec!["y"]);
    let model = load_onnx_model(&bytes).unwrap();
    let input = Tensor::from_vec(vec![32, 64], vec![0.1f32; 32 * 64]).unwrap();
    let mut feed = HashMap::new();
    feed.insert("x".to_string(), input);
    run_onnx_model(&model, feed).unwrap();

    let label = yscv_kernels::take_matmul_kernel().map(|k| k.label());
    assert!(
        label.is_some_and(|l| matches!(
            l,
            "blas-sgemm"
                | "blocked-mr8"
                | "blocked-mr6"
                | "blocked-mr12"
                | "low-k-tile"
                | "blocked-mr4"
                | "row-gemm"
        )),
        "expected a matmul family, got {label:?}"
    );
}

#[test]
fn dispatch_records_gemm_kernel() {
    // The Gemm op routes through matmul_2d -> *_with_plan, which records the
    // same GEMM family enum as the MatMul fused dispatcher.
    let b_w = onnx::TensorProto {
        name: Some("w".into()),
        dims: vec![64, 48],
        data_type: Some(1),
        float_data: vec![0.02f32; 64 * 48],
        ..Default::default()
    };
    let c_bias = onnx::TensorProto {
        name: Some("c".into()),
        dims: vec![48],
        data_type: Some(1),
        float_data: vec![0.1f32; 48],
        ..Default::default()
    };
    let node = onnx::NodeProto {
        op_type: Some("Gemm".into()),
        name: Some("gemm".into()),
        input: vec!["x".into(), "w".into(), "c".into()],
        output: vec!["y".into()],
        ..Default::default()
    };
    let bytes = build_minimal_onnx_model(
        vec![node],
        vec![b_w, c_bias],
        vec!["x", "w", "c"],
        vec!["y"],
    );
    let model = load_onnx_model(&bytes).unwrap();
    let input = Tensor::from_vec(vec![32, 64], vec![0.1f32; 32 * 64]).unwrap();
    let mut feed = HashMap::new();
    feed.insert("x".to_string(), input);
    run_onnx_model(&model, feed).unwrap();

    let label = yscv_kernels::take_matmul_kernel().map(|k| k.label());
    assert!(
        label.is_some_and(|l| matches!(l, "blas-sgemm" | "blocked-mr4" | "row-gemm")),
        "expected a Gemm-tree matmul family, got {label:?}"
    );
}
