use super::*;
use crate::optimizer::{fuse_conv_relu, graph_stats, optimize_onnx_graph};

#[test]
fn optimize_removes_dropout_nodes() {
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
    let bytes = build_minimal_onnx_model(nodes, vec![], vec!["x"], vec!["y"]);
    let mut model = load_onnx_model(&bytes).unwrap();
    assert_eq!(model.node_count(), 3);

    optimize_onnx_graph(&mut model);
    assert_eq!(model.node_count(), 2, "dropout should be removed");
    // relu1 should now consume relu_out directly
    assert_eq!(model.nodes[1].inputs[0], "relu_out");
}

#[test]
fn optimize_eliminates_dead_nodes() {
    let nodes = vec![
        onnx::NodeProto {
            op_type: Some("Relu".into()),
            name: Some("used".into()),
            input: vec!["x".into()],
            output: vec!["y".into()],
            ..Default::default()
        },
        onnx::NodeProto {
            op_type: Some("Relu".into()),
            name: Some("dead".into()),
            input: vec!["x".into()],
            output: vec!["unused_output".into()],
            ..Default::default()
        },
    ];
    let bytes = build_minimal_onnx_model(nodes, vec![], vec!["x"], vec!["y"]);
    let mut model = load_onnx_model(&bytes).unwrap();
    optimize_onnx_graph(&mut model);
    assert_eq!(model.node_count(), 1, "dead node should be eliminated");
    assert_eq!(model.nodes[0].name, "used");
}

#[test]
fn fuse_conv_relu_merges_pair() {
    let nodes = vec![
        onnx::NodeProto {
            op_type: Some("Conv".into()),
            name: Some("conv0".into()),
            input: vec!["x".into(), "w".into()],
            output: vec!["conv_out".into()],
            ..Default::default()
        },
        onnx::NodeProto {
            op_type: Some("Relu".into()),
            name: Some("relu0".into()),
            input: vec!["conv_out".into()],
            output: vec!["y".into()],
            ..Default::default()
        },
    ];
    let bytes = build_minimal_onnx_model(nodes, vec![], vec!["x", "w"], vec!["y"]);
    let mut model = load_onnx_model(&bytes).unwrap();
    fuse_conv_relu(&mut model);
    assert_eq!(model.node_count(), 1);
    assert_eq!(model.nodes[0].op_type, "Conv_Relu");
    assert_eq!(model.nodes[0].outputs[0], "y");
}

#[test]
fn reorder_enables_fusion_on_interleaved_branches() {
    // Two branches exported interleaved (`convA, convB, reluA, reluB`),
    // the order multi-input models (e.g. a Siamese tracker) commonly get.
    // Positional `fuse_conv_relu` only inspects `nodes[i+1]`, so neither
    // Conv+Relu pair is adjacent and nothing fuses. `optimize_onnx_graph`
    // reorders into a depth-first topological order first, which walks each
    // branch to completion and restores producer/consumer adjacency.
    let nodes = vec![
        onnx::NodeProto {
            op_type: Some("Conv".into()),
            name: Some("conv_a".into()),
            input: vec!["xa".into(), "w".into()],
            output: vec!["conv_a_out".into()],
            ..Default::default()
        },
        onnx::NodeProto {
            op_type: Some("Conv".into()),
            name: Some("conv_b".into()),
            input: vec!["xb".into(), "w".into()],
            output: vec!["conv_b_out".into()],
            ..Default::default()
        },
        onnx::NodeProto {
            op_type: Some("Relu".into()),
            name: Some("relu_a".into()),
            input: vec!["conv_a_out".into()],
            output: vec!["ya".into()],
            ..Default::default()
        },
        onnx::NodeProto {
            op_type: Some("Relu".into()),
            name: Some("relu_b".into()),
            input: vec!["conv_b_out".into()],
            output: vec!["yb".into()],
            ..Default::default()
        },
    ];
    let bytes = build_minimal_onnx_model(nodes, vec![], vec!["xa", "xb", "w"], vec!["ya", "yb"]);
    let mut model = load_onnx_model(&bytes).unwrap();
    optimize_onnx_graph(&mut model);
    assert_eq!(
        model.node_count(),
        2,
        "both interleaved Conv+Relu pairs should fuse after reorder"
    );
    assert!(
        model.nodes.iter().all(|n| n.op_type == "Conv_Relu"),
        "every node should be a fused Conv_Relu, got {:?}",
        model
            .nodes
            .iter()
            .map(|n| n.op_type.clone())
            .collect::<Vec<_>>()
    );
}

#[test]
fn graph_stats_reports_op_counts() {
    let nodes = vec![
        onnx::NodeProto {
            op_type: Some("Conv".into()),
            name: Some("c1".into()),
            input: vec!["x".into()],
            output: vec!["a".into()],
            ..Default::default()
        },
        onnx::NodeProto {
            op_type: Some("Relu".into()),
            name: Some("r1".into()),
            input: vec!["a".into()],
            output: vec!["b".into()],
            ..Default::default()
        },
        onnx::NodeProto {
            op_type: Some("Conv".into()),
            name: Some("c2".into()),
            input: vec!["b".into()],
            output: vec!["y".into()],
            ..Default::default()
        },
    ];
    let bytes = build_minimal_onnx_model(nodes, vec![], vec!["x"], vec!["y"]);
    let model = load_onnx_model(&bytes).unwrap();
    let stats = graph_stats(&model);
    assert_eq!(stats.node_count, 3);
    assert_eq!(stats.op_types[0], ("Conv".to_string(), 2));
    assert_eq!(stats.op_types[1], ("Relu".to_string(), 1));
}
