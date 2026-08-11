use super::*;
use crate::exporter::{
    OnnxExportAttr, OnnxExportGraph, OnnxExportNode, OnnxExportValueInfo, export_onnx_model,
};
use crate::loader::{OnnxModel, OnnxNode};
use rustc_hash::FxHashMap;

#[test]
fn export_roundtrip_relu_graph() {
    let graph = OnnxExportGraph {
        nodes: vec![OnnxExportNode {
            op_type: "Relu".into(),
            name: "relu0".into(),
            inputs: vec!["x".into()],
            outputs: vec!["y".into()],
            attributes: vec![],
        }],
        initializers: vec![],
        inputs: vec![OnnxExportValueInfo {
            name: "x".into(),
            shape: vec![1, 4],
        }],
        outputs: vec![OnnxExportValueInfo {
            name: "y".into(),
            shape: vec![1, 4],
        }],
        opset_version: 13,
        int64_initializers: Vec::new(),
    };
    let bytes = export_onnx_model(&graph, "yscv-test", "relu_model").unwrap();
    let model = load_onnx_model(&bytes).unwrap();
    assert_eq!(model.node_count(), 1);
    assert_eq!(model.nodes[0].op_type, "Relu");

    let input = Tensor::from_vec(vec![1, 4], vec![-1.0, 2.0, -3.0, 4.0]).unwrap();
    let mut feed = FxHashMap::default();
    feed.insert("x".to_string(), input);
    let result = run_onnx_model(&model, feed).unwrap();
    assert_eq!(result["y"].data(), &[0.0, 2.0, 0.0, 4.0]);
}

#[test]
fn export_roundtrip_gemm_with_weights() {
    let weight = Tensor::from_vec(vec![2, 3], vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0]).unwrap();
    let bias = Tensor::from_vec(vec![2], vec![0.5, -0.5]).unwrap();

    let graph = OnnxExportGraph {
        nodes: vec![OnnxExportNode {
            op_type: "Gemm".into(),
            name: "fc".into(),
            inputs: vec!["x".into(), "w".into(), "b".into()],
            outputs: vec!["y".into()],
            attributes: vec![OnnxExportAttr::Int("transB".into(), 1)],
        }],
        initializers: vec![("w".into(), weight), ("b".into(), bias)],
        inputs: vec![OnnxExportValueInfo {
            name: "x".into(),
            shape: vec![1, 3],
        }],
        outputs: vec![OnnxExportValueInfo {
            name: "y".into(),
            shape: vec![1, 2],
        }],
        opset_version: 13,
        int64_initializers: Vec::new(),
    };
    let bytes = export_onnx_model(&graph, "yscv", "gemm_model").unwrap();
    let model = load_onnx_model(&bytes).unwrap();

    let input = Tensor::from_vec(vec![1, 3], vec![1.0, 2.0, 3.0]).unwrap();
    let mut feed = FxHashMap::default();
    feed.insert("x".to_string(), input);
    let result = run_onnx_model(&model, feed).unwrap();
    let out = &result["y"];
    assert_eq!(out.shape(), &[1, 2]);
    assert!((out.data()[0] - 1.5).abs() < 1e-5); // 1*1 + 0.5
    assert!((out.data()[1] - 1.5).abs() < 1e-5); // 2*1 + (-0.5)
}

#[test]
fn export_to_file_roundtrip() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.onnx");

    let graph = OnnxExportGraph {
        nodes: vec![OnnxExportNode {
            op_type: "Relu".into(),
            name: "r".into(),
            inputs: vec!["in".into()],
            outputs: vec!["out".into()],
            attributes: vec![],
        }],
        initializers: vec![],
        inputs: vec![OnnxExportValueInfo {
            name: "in".into(),
            shape: vec![2],
        }],
        outputs: vec![OnnxExportValueInfo {
            name: "out".into(),
            shape: vec![2],
        }],
        opset_version: 13,
        int64_initializers: Vec::new(),
    };

    crate::exporter::export_onnx_model_to_file(&graph, "test", "test", &path).unwrap();
    let model = crate::loader::load_onnx_model_from_file(&path).unwrap();
    assert_eq!(model.node_count(), 1);
}

/// Loading and exporting a model returns its Conv weights unchanged, byte for
/// byte.
///
/// The export used to have to undo three internal permutations, because
/// `load_onnx_model` rewrote the initializers in place. Now the permuted copies
/// belong to the plan and the initializers stay as ONNX wrote them, so this is
/// a pass-through — and a round trip is the direct way to say so. It fails the
/// moment anything starts writing packed bytes back into `initializers`, which
/// is what made the export need an inverse in the first place.
#[test]
fn onnx_model_export_round_trips_conv_weights_unchanged() {
    let conv = |name: &str, weight: &str, group: i64| onnx::NodeProto {
        op_type: Some("Conv".into()),
        name: Some(name.into()),
        input: vec!["x".into(), weight.into()],
        output: vec![format!("{name}_out")],
        attribute: vec![
            make_ints_attr("kernel_shape", vec![3, 3]),
            make_int_attr("group", group),
        ],
        ..Default::default()
    };
    let init = |name: &str, dims: Vec<i64>| onnx::TensorProto {
        name: Some(name.into()),
        dims: dims.clone(),
        data_type: Some(1),
        float_data: (0..dims.iter().product::<i64>())
            .map(|v| v as f32)
            .collect(),
        ..Default::default()
    };
    // One weight per layout the plan packs into: group-1, depthwise, grouped.
    let bytes = build_minimal_onnx_model(
        vec![
            conv("plain", "w_plain", 1),
            conv("dw", "w_dw", 8),
            conv("grouped", "w_group", 2),
        ],
        vec![
            init("w_plain", vec![8, 4, 3, 3]),
            init("w_dw", vec![8, 1, 3, 3]),
            init("w_group", vec![8, 4, 3, 3]),
        ],
        vec!["x"],
        vec!["plain_out"],
    );

    let model = load_onnx_model(&bytes).expect("load");
    let graph = crate::exporter::onnx_model_to_export_graph(&model);

    for name in ["w_plain", "w_dw", "w_group"] {
        let exported = graph
            .initializers
            .iter()
            .find(|(n, _)| n == name)
            .unwrap_or_else(|| panic!("`{name}` missing from the export"))
            .1
            .clone();
        let expected: Vec<f32> = (0..exported.data().len()).map(|v| v as f32).collect();
        assert_eq!(
            exported.shape(),
            &[8, if name == "w_dw" { 1 } else { 4 }, 3, 3],
            "`{name}` changed shape across the round trip"
        );
        assert_eq!(
            exported.data(),
            expected.as_slice(),
            "`{name}` changed values across the round trip"
        );
    }
}

#[test]
fn export_graph_defuses_relu_annotations_and_loads_int8_initializers() {
    let mut initializers = FxHashMap::default();
    initializers.insert(
        "w_q".to_string(),
        Tensor::from_vec(vec![4], vec![-128.0, -2.0, 3.0, 127.0]).unwrap(),
    );
    initializers.insert(
        "w_zp".to_string(),
        Tensor::from_vec(vec![1], vec![0.0]).unwrap(),
    );
    let model = OnnxModel {
        ir_version: 7,
        opset_version: 13,
        producer_name: "test".to_string(),
        graph_name: "g".to_string(),
        inputs: vec!["x".to_string()],
        outputs: vec!["y".to_string()],
        initializers,
        nodes: vec![OnnxNode {
            op_type: "Conv_Relu".to_string(),
            name: "conv_relu".to_string(),
            inputs: vec!["x".to_string(), "w_q".to_string()],
            outputs: vec!["y".to_string()],
            attributes: FxHashMap::default(),
        }],
        packed_int4_weights: Default::default(),
        runtime_index: Default::default(),
    };

    let graph = crate::exporter::onnx_model_to_export_graph(&model);
    assert_eq!(graph.nodes[0].op_type, "Conv");
    assert_eq!(graph.nodes[1].op_type, "Relu");
    assert_eq!(graph.nodes[1].outputs, vec!["y".to_string()]);

    let bytes = crate::exporter::export_onnx_model(&graph, "yscv", "q").unwrap();
    let loaded = crate::loader::load_onnx_model(&bytes).unwrap();
    assert_eq!(
        loaded.initializers["w_q"].data(),
        &[-128.0, -2.0, 3.0, 127.0]
    );
    assert_eq!(loaded.initializers["w_zp"].data(), &[0.0]);
}
