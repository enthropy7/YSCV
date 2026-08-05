use super::*;
use crate::optimizer::{
    fuse_conv_relu, graph_cost, graph_cost_report, graph_stats, optimize_onnx_graph,
};
use crate::{TensorShape, infer_shapes};

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

/// `NodeProto.name` is optional. The pass used to delete Dropouts by name via
/// `retain`, so a single unnamed Dropout put the empty string in the delete set
/// and took every other unnamed node in the graph with it.
#[test]
fn remove_dropout_keeps_unnamed_siblings() {
    let nodes = vec![
        onnx::NodeProto {
            op_type: Some("Relu".into()),
            input: vec!["x".into()],
            output: vec!["relu_out".into()],
            ..Default::default()
        },
        onnx::NodeProto {
            op_type: Some("Dropout".into()),
            input: vec!["relu_out".into()],
            output: vec!["drop_out".into()],
            ..Default::default()
        },
        onnx::NodeProto {
            op_type: Some("Relu".into()),
            input: vec!["drop_out".into()],
            output: vec!["y".into()],
            ..Default::default()
        },
    ];
    let bytes = build_minimal_onnx_model(nodes, vec![], vec!["x"], vec!["y"]);
    let mut model = load_onnx_model(&bytes).unwrap();
    assert!(
        model.nodes.iter().all(|n| n.name.is_empty()),
        "fixture must have unnamed nodes for this to be a regression test"
    );

    optimize_onnx_graph(&mut model);

    assert_eq!(
        model.node_count(),
        2,
        "only the Dropout should go, not every unnamed node"
    );
    assert!(model.nodes.iter().all(|n| n.op_type == "Relu"));
    assert_eq!(model.nodes[1].inputs[0], "relu_out");
}

/// A Squeeze -> Unsqueeze -> Squeeze chain with matching axes matches the
/// inverse-pair predicate at both `i` and `i + 1`. Accepting both made the
/// reverse-removal loop delete already-shifted indices.
#[test]
fn eliminate_squeeze_unsqueeze_handles_overlapping_chain() {
    let axes_attr = || make_ints_attr("axes", vec![0]);
    let nodes = vec![
        onnx::NodeProto {
            op_type: Some("Squeeze".into()),
            name: Some("sq0".into()),
            input: vec!["x".into()],
            output: vec!["a".into()],
            attribute: vec![axes_attr()],
            ..Default::default()
        },
        onnx::NodeProto {
            op_type: Some("Unsqueeze".into()),
            name: Some("unsq0".into()),
            input: vec!["a".into()],
            output: vec!["b".into()],
            attribute: vec![axes_attr()],
            ..Default::default()
        },
        onnx::NodeProto {
            op_type: Some("Squeeze".into()),
            name: Some("sq1".into()),
            input: vec!["b".into()],
            output: vec!["c".into()],
            attribute: vec![axes_attr()],
            ..Default::default()
        },
        onnx::NodeProto {
            op_type: Some("Relu".into()),
            name: Some("sink".into()),
            input: vec!["c".into()],
            output: vec!["y".into()],
            ..Default::default()
        },
    ];
    let bytes = build_minimal_onnx_model(nodes, vec![], vec!["x"], vec!["y"]);
    let mut model = load_onnx_model(&bytes).unwrap();

    optimize_onnx_graph(&mut model);

    // The Relu is the only node that must survive; whichever inverse pair the
    // pass claims, it must not corrupt the graph around it.
    let sink = model
        .nodes
        .iter()
        .find(|n| n.op_type == "Relu")
        .expect("Relu sink must survive");
    assert_eq!(model.outputs, vec!["y".to_string()]);
    assert!(
        !sink.inputs[0].is_empty(),
        "sink input must stay wired to a real value"
    );
    for node in &model.nodes {
        assert!(
            matches!(node.op_type.as_str(), "Squeeze" | "Unsqueeze" | "Relu"),
            "unrelated node type {} appeared",
            node.op_type
        );
    }
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
fn graph_cost_report_is_deterministic_and_sorted_by_stable_key() {
    let nodes = vec![
        onnx::NodeProto {
            op_type: Some("Relu".into()),
            name: Some("relu_b".into()),
            input: vec!["mid_b".into()],
            output: vec!["out_b".into()],
            ..Default::default()
        },
        onnx::NodeProto {
            op_type: Some("Relu".into()),
            name: Some("relu_a".into()),
            input: vec!["mid_a".into()],
            output: vec!["out_a".into()],
            ..Default::default()
        },
    ];
    let bytes = build_minimal_onnx_model(
        nodes,
        vec![],
        vec!["mid_a", "mid_b"],
        vec!["out_a", "out_b"],
    );
    let model = load_onnx_model(&bytes).unwrap();
    let shapes = infer_shapes(
        &model,
        &rustc_hash::FxHashMap::from_iter([
            ("mid_a".to_string(), TensorShape::known(vec![1, 4])),
            ("mid_b".to_string(), TensorShape::known(vec![1, 4])),
        ]),
    );
    let cost = graph_cost(&model, &shapes);
    let report_a = graph_cost_report(&cost);
    let report_b = graph_cost_report(&cost);

    assert_eq!(
        report_a, report_b,
        "report formatting must be deterministic"
    );
    let pos_a = report_a.find("key=Relu|relu_a|out_a|mid_a").unwrap();
    let pos_b = report_a.find("key=Relu|relu_b|out_b|mid_b").unwrap();
    assert!(
        pos_a < pos_b,
        "nodes should be sorted by stable textual key"
    );
}

#[test]
fn graph_cost_report_shows_lighter_graph_after_optimization() {
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
            output: vec!["relu_out".into()],
            ..Default::default()
        },
        onnx::NodeProto {
            op_type: Some("Dropout".into()),
            name: Some("drop0".into()),
            input: vec!["relu_out".into()],
            output: vec!["y".into()],
            ..Default::default()
        },
    ];
    let w = onnx::TensorProto {
        name: Some("w".into()),
        dims: vec![8, 3, 3, 3],
        data_type: Some(1),
        float_data: vec![0.0; 8 * 3 * 3 * 3],
        ..Default::default()
    };
    let bytes = build_minimal_onnx_model(nodes, vec![w], vec!["x"], vec!["y"]);
    let mut model = load_onnx_model(&bytes).unwrap();
    let input_shapes =
        rustc_hash::FxHashMap::from_iter([("x".to_string(), TensorShape::known(vec![1, 3, 8, 8]))]);

    let before_shapes = infer_shapes(&model, &input_shapes);
    let before = graph_cost(&model, &before_shapes);
    let before_report = graph_cost_report(&before);

    optimize_onnx_graph(&mut model);

    let after_shapes = infer_shapes(&model, &input_shapes);
    let after = graph_cost(&model, &after_shapes);
    let after_report = graph_cost_report(&after);

    assert!(
        after.node_count < before.node_count,
        "optimizer should remove/fuse nodes"
    );
    assert!(
        after.score <= before.score,
        "optimized graph should not get heavier in this case"
    );
    assert!(before_report.contains("summary.node_count=3"));
    assert!(after_report.contains("summary.node_count=1"));
    assert!(after_report.contains("op_type=Conv_Relu"));
    assert!(!after_report.contains("op_type=Dropout"));
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

#[test]
fn rewrite_convtranspose_dts_is_numerically_identical() {
    use crate::optimizer::rewrite_convtranspose_dts;
    use crate::tests::equivalence::{Lcg, Tolerance, assert_transform_preserves_numerics};

    // ConvTranspose k=2, s=2: C_in=3, C_out=2, pseudo-random 1x3x4x4 input.
    let (c_in, c_out, k, ih, iw) = (3usize, 2usize, 2usize, 4usize, 4usize);
    let mut rng = Lcg::new(0xABCD_1234);
    let w: Vec<f32> = rng.vec(c_in * c_out * k * k);
    let b: Vec<f32> = rng.vec(c_out);
    let x: Vec<f32> = rng.vec(c_in * ih * iw);

    let weight = onnx::TensorProto {
        name: Some("w".into()),
        dims: vec![c_in as i64, c_out as i64, k as i64, k as i64],
        data_type: Some(1),
        float_data: w,
        ..Default::default()
    };
    let bias = onnx::TensorProto {
        name: Some("b".into()),
        dims: vec![c_out as i64],
        data_type: Some(1),
        float_data: b,
        ..Default::default()
    };
    let node = onnx::NodeProto {
        op_type: Some("ConvTranspose".into()),
        name: Some("up".into()),
        input: vec!["x".into(), "w".into(), "b".into()],
        output: vec!["y".into()],
        attribute: vec![
            make_ints_attr("kernel_shape", vec![k as i64, k as i64]),
            make_ints_attr("strides", vec![k as i64, k as i64]),
        ],
        ..Default::default()
    };
    let bytes = build_minimal_onnx_model(vec![node], vec![weight, bias], vec!["x"], vec!["y"]);

    let input = Tensor::from_vec(vec![1, c_in, ih, iw], x).unwrap();
    let mut feed = FxHashMap::default();
    feed.insert("x".to_string(), input);

    // The rewrite replaces the ConvTranspose kernel with Conv1x1 +
    // DepthToSpace(CRD), so the arithmetic is re-associated rather than
    // preserved bit-for-bit.
    assert_transform_preserves_numerics(
        "rewrite_convtranspose_dts",
        &bytes,
        &feed,
        Tolerance::Abs(1e-5),
        |model| {
            rewrite_convtranspose_dts(model);
            assert!(
                model.nodes.iter().all(|n| n.op_type != "ConvTranspose"),
                "pass must replace the eligible ConvTranspose"
            );
            assert!(model.nodes.iter().any(|n| n.op_type == "DepthToSpace"));
        },
    );

    // The harness checks shape equality between the two runs; pin the absolute
    // shape too, so a rewrite that shrinks both sides identically is caught.
    let model = load_onnx_model(&bytes).unwrap();
    let out = run_onnx_model(&model, feed).unwrap();
    assert_eq!(out["y"].shape(), &[1, c_out, ih * k, iw * k]);
}

#[test]
fn rewrite_convtranspose_dts_skips_unsafe_cases() {
    use crate::optimizer::rewrite_convtranspose_dts;
    use crate::tests::equivalence::{Lcg, assert_transform_is_noop};

    // k != s puts this outside the rewrite's safe subset, so the pass must
    // decline. Firing anyway would silently produce a differently-shaped graph.
    let weight = onnx::TensorProto {
        name: Some("w".into()),
        dims: vec![1, 1, 3, 3],
        data_type: Some(1),
        float_data: vec![0.1; 9],
        ..Default::default()
    };
    let node = onnx::NodeProto {
        op_type: Some("ConvTranspose".into()),
        name: Some("up".into()),
        input: vec!["x".into(), "w".into()],
        output: vec!["y".into()],
        attribute: vec![
            make_ints_attr("kernel_shape", vec![3, 3]),
            make_ints_attr("strides", vec![2, 2]),
        ],
        ..Default::default()
    };
    let bytes = build_minimal_onnx_model(vec![node], vec![weight], vec!["x"], vec!["y"]);

    let mut rng = Lcg::new(0x5EED_0001);
    let mut feed = FxHashMap::default();
    feed.insert("x".to_string(), rng.tensor(vec![1, 1, 4, 4]));

    assert_transform_is_noop(
        "rewrite_convtranspose_dts/k!=s",
        &bytes,
        &feed,
        rewrite_convtranspose_dts,
    );

    let mut model = load_onnx_model(&bytes).unwrap();
    rewrite_convtranspose_dts(&mut model);
    assert!(model.nodes.iter().any(|n| n.op_type == "ConvTranspose"));
}
