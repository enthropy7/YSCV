use super::*;

#[test]
fn load_empty_graph() {
    let bytes = build_minimal_onnx_model(vec![], vec![], vec!["x"], vec!["y"]);
    let model = OnnxLoader::from_bytes(&bytes).decode().unwrap();
    assert_eq!(model.graph_name, "test_graph");
    assert_eq!(model.ir_version, 8);
    assert_eq!(model.opset_version, 13);
    assert_eq!(model.inputs, vec!["x"]);
    assert_eq!(model.outputs, vec!["y"]);
    assert_eq!(model.node_count(), 0);
    assert!(model.initializers.is_empty());
}

#[test]
fn load_rejects_a_value_defined_twice() {
    let relu = |input: &str, output: &str| onnx::NodeProto {
        op_type: Some("Relu".into()),
        input: vec![input.into()],
        output: vec![output.into()],
        ..Default::default()
    };
    let bias = onnx::TensorProto {
        name: Some("b".into()),
        dims: vec![1],
        data_type: Some(1),
        float_data: vec![0.0],
        ..Default::default()
    };
    for (nodes, inits) in [
        (vec![relu("x", "y"), relu("x", "y")], vec![]),
        (vec![relu("x", "x")], vec![]),
        (vec![relu("x", "b")], vec![bias.clone()]),
    ] {
        let bytes = build_minimal_onnx_model(nodes, inits, vec!["x"], vec!["y"]);
        assert!(matches!(
            OnnxLoader::from_bytes(&bytes).decode(),
            Err(crate::OnnxError::DuplicateDefinition { .. })
        ));
    }

    // An initializer that shares its name with a graph input is the older IR's
    // way of giving that input a default, not a second definition.
    let bytes = build_minimal_onnx_model(vec![relu("b", "y")], vec![bias], vec!["b"], vec!["y"]);
    assert!(OnnxLoader::from_bytes(&bytes).decode().is_ok());
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
    let model = OnnxLoader::from_bytes(&bytes).decode().unwrap();

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
    let model = OnnxLoader::from_bytes(&bytes).decode().unwrap();

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
    let model = OnnxLoader::from_bytes(&bytes).decode().unwrap();

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
    let model = OnnxLoader::from_bytes(&bytes).decode().unwrap();

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
    let model = OnnxLoader::from_bytes(&bytes).decode().unwrap();

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
    let model = OnnxLoader::from_bytes(&bytes).decode().unwrap();

    assert_eq!(model.node_count(), 3);
    assert_eq!(model.nodes[0].op_type, "Conv");
    assert_eq!(model.nodes[1].op_type, "Relu");
    assert_eq!(model.nodes[2].op_type, "AveragePool");
    assert!(model.get_initializer("conv_w").is_some());
}

#[test]
fn decode_error_on_invalid_bytes() {
    let result = OnnxLoader::from_bytes(&[0xFF, 0x00, 0x01]).decode();
    assert!(
        matches!(result, Err(crate::OnnxError::Decode { .. })),
        "{result:?}"
    );
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
    let result = OnnxLoader::from_bytes(&bytes).decode();
    assert!(
        matches!(
            &result,
            Err(crate::OnnxError::InitializerShapeMismatch { name, expected: 6, got: 2 })
                if name == "bad"
        ),
        "{result:?}"
    );
}

/// `x -> Relu -> Dropout -> Relu -> y`. Dropout is the identity at inference,
/// so the optimizer removes it — which makes its presence a cheap probe for
/// whether the optimizer ran.
fn dropout_probe_model() -> Vec<u8> {
    let nodes = vec![
        onnx::NodeProto {
            op_type: Some("Relu".into()),
            name: Some("relu0".into()),
            input: vec!["x".into()],
            output: vec!["relu_out".into()],
            ..Default::default()
        },
        onnx::NodeProto {
            op_type: Some("Dropout".into()),
            name: Some("drop0".into()),
            input: vec!["relu_out".into()],
            output: vec!["drop_out".into()],
            ..Default::default()
        },
        onnx::NodeProto {
            op_type: Some("Relu".into()),
            name: Some("relu1".into()),
            input: vec!["drop_out".into()],
            output: vec!["y".into()],
            ..Default::default()
        },
    ];
    build_minimal_onnx_model(nodes, vec![], vec!["x"], vec!["y"])
}

/// Executable construction optimizes by default, so a caller that only builds
/// still gets the fused, folded graph the runner is tuned for while the decoded
/// source remains unchanged.
#[test]
fn build_runs_the_optimizer_by_default() {
    // Held because `build` without an explicit level *reads*
    // `YSCV_ONNX_OPTIMIZE_OFF`, and the test below sets it. Without the lock
    // the two race and this one intermittently sees the optimizer switched off
    // under it.
    let _env = crate::tests::equivalence::lock_env();
    let bytes = dropout_probe_model();

    let model = OnnxLoader::from_bytes(&bytes).build().expect("load");
    assert_eq!(
        model.nodes.len(),
        2,
        "loading should have removed the Dropout, got {:?}",
        model.nodes.iter().map(|n| &n.op_type).collect::<Vec<_>>()
    );
    assert!(model.nodes.iter().all(|n| n.op_type == "Relu"));
    assert_eq!(model.nodes[1].inputs[0], "relu_out");
}

/// The opt-out for a caller that wants the file's own graph — an inspector, or
/// a test asserting against a fixture.
#[test]
fn decode_keeps_the_graph_as_written() {
    let bytes = dropout_probe_model();

    let model = OnnxLoader::from_bytes(&bytes).decode().expect("load");
    assert_eq!(model.node_count(), 3);
    assert_eq!(model.nodes[1].op_type, "Dropout");
}

/// `YSCV_ONNX_OPTIMIZE_OFF=1` turns IR optimization off process-wide when no
/// explicit level is selected, for bisecting a model the optimizer miscompiles.
/// An explicit `optimization_level` always wins over the variable.
#[test]
fn optimize_off_env_var_disables_ir_optimization() {
    // Held for the whole test: `build` without an explicit level reads this
    // variable, so a sibling test building a model concurrently would see the
    // toggle.
    let _env = crate::tests::equivalence::lock_env();
    let bytes = dropout_probe_model();

    // SAFETY: `set_var`/`remove_var` are not thread-safe and `cargo test` runs
    // tests concurrently. The `EnvGuard` above proves this thread has exclusive
    // use of the environment for the whole test. The build's result is held
    // unexamined until after `remove_var`, so a failing build clears the
    // variable rather than leaking it into every later test.
    #[allow(unsafe_code)]
    unsafe {
        std::env::set_var("YSCV_ONNX_OPTIMIZE_OFF", "1");
    }
    let disabled = OnnxLoader::from_bytes(&bytes).build();
    let explicit_enabled = OnnxLoader::from_bytes(&bytes)
        .optimization_level(OptimizationLevel::Enabled)
        .build();
    #[allow(unsafe_code)]
    unsafe {
        std::env::remove_var("YSCV_ONNX_OPTIMIZE_OFF");
    }
    let disabled = disabled.expect("build with the optimizer off");
    let explicit_enabled = explicit_enabled.expect("explicit enabled build");
    let enabled = OnnxLoader::from_bytes(&bytes).build().expect("build");
    let explicit_disabled = OnnxLoader::from_bytes(&bytes)
        .optimization_level(OptimizationLevel::Disabled)
        .build()
        .expect("explicit disabled build");

    assert_eq!(disabled.nodes.len(), 3, "the Dropout should have survived");
    assert_eq!(disabled.nodes[1].op_type, "Dropout");
    assert_eq!(explicit_enabled.nodes.len(), 2, "explicit Enabled must win");
    assert_eq!(enabled.nodes.len(), 2, "and be gone once the knob is unset");
    assert_eq!(explicit_disabled.nodes.len(), 3);

    #[allow(unsafe_code)]
    unsafe {
        std::env::set_var("YSCV_ONNX_OPTIMIZE_OFF", "0");
    }
    let zero = OnnxLoader::from_bytes(&bytes).build();
    #[allow(unsafe_code)]
    unsafe {
        std::env::remove_var("YSCV_ONNX_OPTIMIZE_OFF");
    }
    assert_eq!(zero.expect("build with opt-out set to zero").nodes.len(), 2);
}

#[test]
fn builder_path_bytes_and_model_sources_match() {
    let _env = crate::tests::equivalence::lock_env();
    let bytes = dropout_probe_model();
    let directory = tempfile::tempdir().expect("tempdir");
    let path = directory.path().join("model.onnx");
    std::fs::write(&path, &bytes).expect("write model");
    let source = OnnxLoader::from_bytes(&bytes).decode().expect("decode");

    let from_path = OnnxLoader::load(&path).build().expect("path build");
    let from_bytes = OnnxLoader::from_bytes(&bytes).build().expect("byte build");
    let from_model = OnnxLoader::from_model(&source)
        .build()
        .expect("model build");

    let op_types = |executable: &crate::plan::OnnxExecutable| {
        executable
            .nodes
            .iter()
            .map(|node| node.op_type.clone())
            .collect::<Vec<_>>()
    };
    assert_eq!(op_types(&from_path), op_types(&from_bytes));
    assert_eq!(op_types(&from_model), op_types(&from_bytes));
    assert_eq!(from_path.input_names(), from_bytes.input_names());
    assert_eq!(from_path.output_names(), from_bytes.output_names());
}

#[test]
fn executable_is_a_snapshot_of_its_source_model() {
    let _env = crate::tests::equivalence::lock_env();
    let bytes = dropout_probe_model();
    let mut source = OnnxLoader::from_bytes(&bytes).decode().expect("decode");
    let original = OnnxLoader::from_model(&source).build().expect("build");

    source.nodes.clear();
    let changed = OnnxLoader::from_model(&source).build().expect("rebuild");

    assert_eq!(original.nodes.len(), 2);
    assert!(changed.nodes.is_empty());
}

#[test]
fn optimized_build_owns_fusions_without_mutating_source() {
    let weight = onnx::TensorProto {
        name: Some("w".into()),
        dims: vec![1, 1, 1, 1],
        data_type: Some(1),
        float_data: vec![1.0],
        ..Default::default()
    };
    let nodes = vec![
        onnx::NodeProto {
            op_type: Some("Conv".into()),
            name: Some("conv".into()),
            input: vec!["x".into(), "w".into()],
            output: vec!["conv_out".into()],
            ..Default::default()
        },
        onnx::NodeProto {
            op_type: Some("Relu".into()),
            name: Some("relu".into()),
            input: vec!["conv_out".into()],
            output: vec!["y".into()],
            ..Default::default()
        },
    ];
    let bytes = build_minimal_onnx_model(nodes, vec![weight], vec!["x"], vec!["y"]);
    let source = OnnxLoader::from_bytes(&bytes).decode().unwrap();
    let optimized = OnnxLoader::from_model(&source)
        .optimization_level(OptimizationLevel::Enabled)
        .build()
        .unwrap();
    let disabled = OnnxLoader::from_model(&source)
        .optimization_level(OptimizationLevel::Disabled)
        .build()
        .unwrap();

    assert_eq!(
        source
            .nodes
            .iter()
            .map(|n| n.op_type.as_str())
            .collect::<Vec<_>>(),
        ["Conv", "Relu"]
    );
    assert_eq!(optimized.nodes.len(), 1);
    assert_eq!(optimized.nodes[0].op_type, "Conv_Relu");
    assert_eq!(disabled.nodes.len(), 2);
    assert!(!disabled.runtime_index.execution_plan.is_empty());
    assert!(disabled.runtime_index.conv_kernels[0].is_some());
    assert!(
        disabled
            .runtime_index
            .prepacked_conv_weights
            .iter()
            .any(Option::is_some)
    );

    let input = Tensor::from_vec(vec![1, 1, 1, 4], vec![-2.0, -0.5, 1.0, 3.0]).unwrap();
    let feed = || FxHashMap::from_iter([("x".to_string(), input.clone())]);
    let optimized_out = run_onnx_model(&optimized, feed()).unwrap();
    let disabled_out = run_onnx_model(&disabled, feed()).unwrap();
    assert_eq!(optimized_out["y"], disabled_out["y"]);
}

#[test]
fn optimized_build_owns_folded_constants() {
    let constant = |name: &str, value: f32| onnx::TensorProto {
        name: Some(name.into()),
        dims: vec![1],
        data_type: Some(1),
        float_data: vec![value],
        ..Default::default()
    };
    let nodes = vec![
        onnx::NodeProto {
            op_type: Some("Add".into()),
            name: Some("add".into()),
            input: vec!["a".into(), "b".into()],
            output: vec!["sum".into()],
            ..Default::default()
        },
        onnx::NodeProto {
            op_type: Some("Relu".into()),
            name: Some("relu".into()),
            input: vec!["sum".into()],
            output: vec!["y".into()],
            ..Default::default()
        },
    ];
    let bytes = build_minimal_onnx_model(
        nodes,
        vec![constant("a", -2.0), constant("b", 3.0)],
        vec![],
        vec!["y"],
    );
    let source = OnnxLoader::from_bytes(&bytes).decode().unwrap();
    let executable = OnnxLoader::from_model(&source)
        .optimization_level(OptimizationLevel::Enabled)
        .build()
        .unwrap();

    assert_eq!(source.nodes.len(), 2);
    assert_eq!(executable.nodes.len(), 1);
    assert_eq!(executable.nodes[0].op_type, "Relu");
    assert!(
        executable
            .initializers
            .keys()
            .any(|name| !source.initializers.contains_key(name))
    );
    let output = run_onnx_model(&executable, FxHashMap::default()).unwrap();
    assert_eq!(output["y"].data(), &[1.0]);
}
