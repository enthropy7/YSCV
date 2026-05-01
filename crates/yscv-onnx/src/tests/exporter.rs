use super::*;
use crate::exporter::{
    OnnxExportAttr, OnnxExportGraph, OnnxExportNode, OnnxExportValueInfo, export_onnx_model,
};
use crate::loader::{OnnxModel, OnnxNode};
use std::collections::HashMap;

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
    let mut feed = HashMap::new();
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
    let mut feed = HashMap::new();
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

#[test]
fn onnx_model_export_unpermutes_internal_conv_layouts() {
    let mut initializers = HashMap::new();
    // Regular Conv internal KHWC [KH, KW, IC, OC] -> ONNX OIHW [OC, IC, KH, KW].
    initializers.insert(
        "regular".to_string(),
        Tensor::from_vec(vec![2, 3, 4, 5], (0..120).map(|v| v as f32).collect()).unwrap(),
    );
    // Depthwise internal [KH, KW, C, dm] -> ONNX [C*dm, 1, KH, KW].
    initializers.insert(
        "dw".to_string(),
        Tensor::from_vec(vec![3, 3, 7, 1], (0..63).map(|v| v as f32).collect()).unwrap(),
    );
    // Grouped internal [O, KH, KW, I/G] -> ONNX [O, I/G, KH, KW].
    initializers.insert(
        "group".to_string(),
        Tensor::from_vec(vec![6, 3, 3, 2], (0..108).map(|v| v as f32).collect()).unwrap(),
    );

    let mut model = OnnxModel {
        ir_version: 7,
        opset_version: 13,
        producer_name: "test".to_string(),
        graph_name: "g".to_string(),
        inputs: vec!["x".to_string()],
        outputs: vec!["y".to_string()],
        initializers,
        nodes: vec![OnnxNode {
            op_type: "Conv".to_string(),
            name: "conv".to_string(),
            inputs: vec!["x".to_string(), "regular".to_string()],
            outputs: vec!["y".to_string()],
            attributes: HashMap::new(),
        }],
        khwc_weights: Default::default(),
        dw_khwc_weights: Default::default(),
        group_khwc_weights: Default::default(),
        packed_int4_weights: Default::default(),
        runtime_index: Default::default(),
    };
    model.khwc_weights.insert("regular".to_string());
    model.dw_khwc_weights.insert("dw".to_string());
    model.group_khwc_weights.insert("group".to_string());

    let graph = crate::exporter::onnx_model_to_export_graph(&model);
    let regular = graph
        .initializers
        .iter()
        .find(|(name, _)| name == "regular")
        .unwrap()
        .1
        .clone();
    let dw = graph
        .initializers
        .iter()
        .find(|(name, _)| name == "dw")
        .unwrap()
        .1
        .clone();
    let group = graph
        .initializers
        .iter()
        .find(|(name, _)| name == "group")
        .unwrap()
        .1
        .clone();

    assert_eq!(regular.shape(), &[5, 4, 2, 3]);
    assert_eq!(dw.shape(), &[7, 1, 3, 3]);
    assert_eq!(group.shape(), &[6, 2, 3, 3]);

    let dw_src = model.initializers["dw"].data();
    let dw_exported = dw.data();
    // c=2, dm=0, kh=1, kw=2 maps to oc=2 in OIHW.
    let src_idx = ((3 + 2) * 7) + 2;
    let dst_idx = (2 * 3 + 1) * 3 + 2;
    assert_eq!(dw_exported[dst_idx], dw_src[src_idx]);
}

#[test]
fn export_graph_defuses_relu_annotations_and_loads_int8_initializers() {
    let mut initializers = HashMap::new();
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
            attributes: HashMap::new(),
        }],
        khwc_weights: Default::default(),
        dw_khwc_weights: Default::default(),
        group_khwc_weights: Default::default(),
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
