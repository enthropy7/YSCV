use super::*;
use crate::ir::Pass;
use crate::optimizer::{
    FuseActivation, graph_cost, graph_cost_report, graph_stats, optimize_onnx_graph,
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

    optimize_onnx_graph(&mut model).expect("optimize succeeds");
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

    let mut rng = crate::tests::equivalence::Lcg::new(0x0FF1_CE02);
    let mut feed = FxHashMap::default();
    feed.insert("x".to_string(), rng.tensor(vec![1, 3, 2, 2]));
    crate::tests::equivalence::assert_transform_preserves_numerics(
        "optimize/unnamed-dropout",
        &bytes,
        &feed,
        crate::tests::equivalence::Tolerance::Exact,
        |m| optimize_onnx_graph(m).expect("optimize succeeds"),
    );

    optimize_onnx_graph(&mut model).expect("optimize succeeds");

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

    // Structural survival is not enough — the chain has to still compute the
    // same thing after whichever inverse pair the pass claims.
    let mut rng = crate::tests::equivalence::Lcg::new(0x0FF1_CE01);
    let mut feed = FxHashMap::default();
    feed.insert("x".to_string(), rng.tensor(vec![1, 3, 2, 2]));
    crate::tests::equivalence::assert_transform_preserves_numerics(
        "optimize/squeeze-unsqueeze-chain",
        &bytes,
        &feed,
        crate::tests::equivalence::Tolerance::Exact,
        |m| optimize_onnx_graph(m).expect("optimize succeeds"),
    );

    let mut model = load_onnx_model(&bytes).unwrap();
    optimize_onnx_graph(&mut model).expect("optimize succeeds");

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

/// The positional matcher required the inverse pair to be adjacent in node
/// order, which is why `reorder_nodes_for_fusion` had to run before it. On the
/// def-use IR the pair is found through the use list, so this runs the pass
/// *directly on the unreordered graph* — no reordering step to rescue it — with
/// an unrelated node scheduled between the two halves.
#[test]
fn eliminate_squeeze_unsqueeze_matches_non_adjacent_pair() {
    use crate::ir::Pass;
    use crate::optimizer::EliminateSqueezeUnsqueezePairs;
    let axes_attr = || make_ints_attr("axes", vec![0]);
    let nodes = vec![
        onnx::NodeProto {
            op_type: Some("Squeeze".into()),
            name: Some("sq".into()),
            input: vec!["x".into()],
            output: vec!["a".into()],
            attribute: vec![axes_attr()],
            ..Default::default()
        },
        // Independent of the pair, but sitting between its two halves.
        onnx::NodeProto {
            op_type: Some("Relu".into()),
            name: Some("interloper".into()),
            input: vec!["x".into()],
            output: vec!["side".into()],
            ..Default::default()
        },
        onnx::NodeProto {
            op_type: Some("Unsqueeze".into()),
            name: Some("unsq".into()),
            input: vec!["a".into()],
            output: vec!["b".into()],
            attribute: vec![axes_attr()],
            ..Default::default()
        },
        onnx::NodeProto {
            op_type: Some("Add".into()),
            name: Some("sink".into()),
            input: vec!["b".into(), "side".into()],
            output: vec!["y".into()],
            ..Default::default()
        },
    ];
    let bytes = build_minimal_onnx_model(nodes, vec![], vec!["x"], vec!["y"]);

    let mut rng = crate::tests::equivalence::Lcg::new(0x0FF1_CE03);
    let mut feed = FxHashMap::default();
    feed.insert("x".to_string(), rng.tensor(vec![1, 3, 2, 2]));

    // Apply the pass on its own, so nothing has reordered the nodes first.
    crate::tests::equivalence::assert_transform_preserves_numerics(
        "eliminate_squeeze_unsqueeze/non-adjacent",
        &bytes,
        &feed,
        crate::tests::equivalence::Tolerance::Exact,
        |model| {
            let mut graph = model.to_ir();
            let fired = EliminateSqueezeUnsqueezePairs
                .run(&mut graph)
                .expect("pass runs");
            assert!(fired, "pass must match across the intervening node");
            model.apply_ir(&graph);
        },
    );

    let mut model = load_onnx_model(&bytes).unwrap();
    let mut graph = model.to_ir();
    EliminateSqueezeUnsqueezePairs
        .run(&mut graph)
        .expect("pass runs");
    model.apply_ir(&graph);

    assert!(
        model
            .nodes
            .iter()
            .all(|n| !matches!(n.op_type.as_str(), "Squeeze" | "Unsqueeze")),
        "the non-adjacent inverse pair should have been eliminated, got {:?}",
        model.nodes.iter().map(|n| &n.op_type).collect::<Vec<_>>()
    );
}

/// Builds `Conv -> BatchNormalization`, optionally with both Convs sharing one
/// weight initializer so the aliasing guard can be exercised.
fn conv_bn_bytes(shared_weight: bool) -> Vec<u8> {
    let bn_attrs = vec![make_float_attr("epsilon", 1e-5)];
    let mut nodes = vec![
        onnx::NodeProto {
            op_type: Some("Conv".into()),
            name: Some("conv0".into()),
            input: vec!["x".into(), "w".into()],
            output: vec!["c0".into()],
            ..Default::default()
        },
        onnx::NodeProto {
            op_type: Some("BatchNormalization".into()),
            name: Some("bn0".into()),
            input: vec![
                "c0".into(),
                "gamma".into(),
                "beta".into(),
                "mean".into(),
                "var".into(),
            ],
            output: vec!["y".into()],
            attribute: bn_attrs.clone(),
            ..Default::default()
        },
    ];
    if shared_weight {
        nodes.push(onnx::NodeProto {
            op_type: Some("Conv".into()),
            name: Some("conv1".into()),
            input: vec!["x2".into(), "w".into()],
            output: vec!["c1".into()],
            ..Default::default()
        });
        nodes.push(onnx::NodeProto {
            op_type: Some("BatchNormalization".into()),
            name: Some("bn1".into()),
            input: vec![
                "c1".into(),
                "gamma".into(),
                "beta".into(),
                "mean".into(),
                "var".into(),
            ],
            output: vec!["y2".into()],
            attribute: bn_attrs,
            ..Default::default()
        });
    }

    let per_channel = |name: &str, data: Vec<f32>| onnx::TensorProto {
        name: Some(name.into()),
        dims: vec![2],
        data_type: Some(1),
        float_data: data,
        ..Default::default()
    };
    let inits = vec![
        onnx::TensorProto {
            name: Some("w".into()),
            dims: vec![2, 3, 1, 1],
            data_type: Some(1),
            float_data: vec![0.3, -0.7, 0.5, 0.2, 0.9, -0.4],
            ..Default::default()
        },
        per_channel("gamma", vec![1.4, 0.8]),
        per_channel("beta", vec![-0.2, 0.35]),
        per_channel("mean", vec![0.1, -0.25]),
        per_channel("var", vec![0.9, 1.3]),
    ];

    let (inputs, outputs) = if shared_weight {
        (vec!["x", "x2"], vec!["y", "y2"])
    } else {
        (vec!["x"], vec!["y"])
    };
    build_minimal_onnx_model(nodes, inits, inputs, outputs)
}

#[test]
fn fold_conv_bn_is_numerically_equivalent() {
    let bytes = conv_bn_bytes(false);
    let mut rng = crate::tests::equivalence::Lcg::new(0x0FF1_CE05);
    let mut feed = FxHashMap::default();
    feed.insert("x".to_string(), rng.tensor(vec![1, 3, 4, 4]));

    // Folding re-associates the arithmetic, so exact equality is not the bar.
    crate::tests::equivalence::assert_transform_preserves_numerics(
        "fold_conv_bn",
        &bytes,
        &feed,
        crate::tests::equivalence::Tolerance::Abs(1e-5),
        |m| optimize_onnx_graph(m).expect("optimize succeeds"),
    );

    let mut model = load_onnx_model(&bytes).unwrap();
    optimize_onnx_graph(&mut model).expect("optimize succeeds");
    assert_eq!(
        model.node_count(),
        1,
        "the BatchNormalization should fold in"
    );
    assert_eq!(model.nodes[0].op_type, "Conv");
    assert_eq!(
        model.nodes[0].inputs.len(),
        3,
        "a bias operand should have been synthesized"
    );
}

/// Folding rewrites the weight in place, so a weight shared by two Convs — the
/// two branches of a Siamese tracker, say — would get the second Conv's scale
/// applied on top of the first's. The pass must decline rather than corrupt it.
#[test]
fn fold_conv_bn_declines_on_a_shared_weight() {
    let bytes = conv_bn_bytes(true);
    let mut rng = crate::tests::equivalence::Lcg::new(0x0FF1_CE06);
    let mut feed = FxHashMap::default();
    feed.insert("x".to_string(), rng.tensor(vec![1, 3, 4, 4]));
    feed.insert("x2".to_string(), rng.tensor(vec![1, 3, 4, 4]));

    crate::tests::equivalence::assert_transform_preserves_numerics(
        "fold_conv_bn/shared-weight",
        &bytes,
        &feed,
        crate::tests::equivalence::Tolerance::Abs(1e-5),
        |m| optimize_onnx_graph(m).expect("optimize succeeds"),
    );

    let mut model = load_onnx_model(&bytes).unwrap();
    optimize_onnx_graph(&mut model).expect("optimize succeeds");
    assert_eq!(
        model
            .nodes
            .iter()
            .filter(|n| n.op_type == "BatchNormalization")
            .count(),
        2,
        "neither BatchNormalization should fold into the shared weight"
    );
}

/// `Conv -> Mul(const)` and `Conv -> Add(const)`, with the constant on the
/// given operand port so both orderings get exercised.
fn conv_const_binary_bytes(op: &str, const_on_port: usize) -> Vec<u8> {
    let mut binary_inputs = vec!["c0".to_string(), "k".to_string()];
    if const_on_port == 0 {
        binary_inputs.reverse();
    }
    let nodes = vec![
        onnx::NodeProto {
            op_type: Some("Conv".into()),
            name: Some("conv0".into()),
            input: vec!["x".into(), "w".into()],
            output: vec!["c0".into()],
            ..Default::default()
        },
        onnx::NodeProto {
            op_type: Some(op.into()),
            name: Some("binop".into()),
            input: binary_inputs,
            output: vec!["y".into()],
            ..Default::default()
        },
    ];
    let inits = vec![
        onnx::TensorProto {
            name: Some("w".into()),
            dims: vec![2, 3, 1, 1],
            data_type: Some(1),
            float_data: vec![0.3, -0.7, 0.5, 0.2, 0.9, -0.4],
            ..Default::default()
        },
        // `[1, OC, 1, 1]` — the only spelling that broadcasts per-channel
        // against an NCHW convolution output.
        onnx::TensorProto {
            name: Some("k".into()),
            dims: vec![1, 2, 1, 1],
            data_type: Some(1),
            float_data: vec![1.7, -0.6],
            ..Default::default()
        },
    ];
    build_minimal_onnx_model(nodes, inits, vec!["x"], vec!["y"])
}

/// Structural coverage for both operators and both operand orders.
///
/// No numerical check here: the runtime's elementwise kernels require equal
/// shapes, so the *unfolded* reference cannot execute a broadcast constant at
/// all. The scalar case below covers equivalence on a graph that both sides can
/// run.
#[test]
fn fold_conv_const_binary_absorbs_mul_and_add() {
    for op in ["Mul", "Add"] {
        for const_on_port in [0usize, 1] {
            let bytes = conv_const_binary_bytes(op, const_on_port);
            let mut model = load_onnx_model(&bytes).unwrap();
            optimize_onnx_graph(&mut model).expect("optimize succeeds");

            assert_eq!(
                model.node_count(),
                1,
                "{op} with the constant on port {const_on_port} should fold into the Conv"
            );
            assert_eq!(model.nodes[0].op_type, "Conv");
            assert_eq!(model.nodes[0].inputs.len(), 3, "bias should be synthesized");
        }
    }
}

/// A scalar constant broadcasts to everything, so both the folded and unfolded
/// graphs run and the arithmetic can be compared directly.
#[test]
fn fold_conv_scalar_mul_is_numerically_equivalent() {
    let nodes = vec![
        onnx::NodeProto {
            op_type: Some("Conv".into()),
            name: Some("conv0".into()),
            input: vec!["x".into(), "w".into()],
            output: vec!["c0".into()],
            ..Default::default()
        },
        onnx::NodeProto {
            op_type: Some("Mul".into()),
            name: Some("scale".into()),
            input: vec!["c0".into(), "k".into()],
            output: vec!["y".into()],
            ..Default::default()
        },
    ];
    let inits = vec![
        onnx::TensorProto {
            name: Some("w".into()),
            dims: vec![2, 3, 1, 1],
            data_type: Some(1),
            float_data: vec![0.3, -0.7, 0.5, 0.2, 0.9, -0.4],
            ..Default::default()
        },
        onnx::TensorProto {
            name: Some("k".into()),
            dims: vec![1],
            data_type: Some(1),
            float_data: vec![1.7],
            ..Default::default()
        },
    ];
    let bytes = build_minimal_onnx_model(nodes, inits, vec!["x"], vec!["y"]);

    let mut rng = crate::tests::equivalence::Lcg::new(0x0FF1_CE07);
    let mut feed = FxHashMap::default();
    feed.insert("x".to_string(), rng.tensor(vec![1, 3, 4, 4]));
    crate::tests::equivalence::assert_transform_preserves_numerics(
        "fold_conv_mul/scalar",
        &bytes,
        &feed,
        crate::tests::equivalence::Tolerance::Abs(1e-5),
        |m| optimize_onnx_graph(m).expect("optimize succeeds"),
    );

    let mut model = load_onnx_model(&bytes).unwrap();
    optimize_onnx_graph(&mut model).expect("optimize succeeds");
    assert_eq!(model.node_count(), 1, "the scalar Mul should fold in");
}

/// A rank-1 constant aligns against the *width* axis, not channels, so folding
/// it into the weights as a per-channel scale would compute something else.
/// The element count alone does not establish that a constant is per-channel.
#[test]
fn fold_conv_const_binary_declines_on_non_channel_constant() {
    let nodes = vec![
        onnx::NodeProto {
            op_type: Some("Conv".into()),
            name: Some("conv0".into()),
            input: vec!["x".into(), "w".into()],
            output: vec!["c0".into()],
            ..Default::default()
        },
        onnx::NodeProto {
            op_type: Some("Mul".into()),
            name: Some("binop".into()),
            input: vec!["c0".into(), "k".into()],
            output: vec!["y".into()],
            ..Default::default()
        },
    ];
    let inits = vec![
        onnx::TensorProto {
            name: Some("w".into()),
            dims: vec![2, 3, 1, 1],
            data_type: Some(1),
            float_data: vec![0.3, -0.7, 0.5, 0.2, 0.9, -0.4],
            ..Default::default()
        },
        // Rank-1 with one element per output channel. Tempting, and what the
        // string version accepted, but it broadcasts against width.
        onnx::TensorProto {
            name: Some("k".into()),
            dims: vec![2],
            data_type: Some(1),
            float_data: vec![1.7, -0.6],
            ..Default::default()
        },
    ];
    let bytes = build_minimal_onnx_model(nodes, inits, vec!["x"], vec!["y"]);

    let mut model = load_onnx_model(&bytes).unwrap();
    optimize_onnx_graph(&mut model).expect("optimize succeeds");
    assert_eq!(
        model.node_count(),
        2,
        "a non-per-channel constant must not fold"
    );
}

fn const_tensor(name: &str, dims: Vec<i64>, data: Vec<f32>) -> onnx::TensorProto {
    onnx::TensorProto {
        name: Some(name.into()),
        dims,
        data_type: Some(1),
        float_data: data,
        ..Default::default()
    }
}

/// A chain of constant nodes folds in a single sweep: each fold makes its
/// output constant, which can only enable nodes later in topological order.
#[test]
fn fold_constants_collapses_a_chain_in_one_sweep() {
    let nodes = vec![
        onnx::NodeProto {
            op_type: Some("Add".into()),
            name: Some("a".into()),
            input: vec!["k1".into(), "k2".into()],
            output: vec!["s".into()],
            ..Default::default()
        },
        onnx::NodeProto {
            op_type: Some("Mul".into()),
            name: Some("b".into()),
            input: vec!["s".into(), "k3".into()],
            output: vec!["p".into()],
            ..Default::default()
        },
        onnx::NodeProto {
            op_type: Some("Add".into()),
            name: Some("sink".into()),
            input: vec!["x".into(), "p".into()],
            output: vec!["y".into()],
            ..Default::default()
        },
    ];
    let inits = vec![
        const_tensor("k1", vec![4], vec![1.0, 2.0, 3.0, 4.0]),
        const_tensor("k2", vec![4], vec![0.5, 0.5, 0.5, 0.5]),
        const_tensor("k3", vec![4], vec![2.0, 2.0, 2.0, 2.0]),
    ];
    let bytes = build_minimal_onnx_model(nodes, inits, vec!["x"], vec!["y"]);

    let mut rng = crate::tests::equivalence::Lcg::new(0x0FF1_CE08);
    let mut feed = FxHashMap::default();
    feed.insert("x".to_string(), rng.tensor(vec![4]));
    crate::tests::equivalence::assert_transform_preserves_numerics(
        "fold_constants/chain",
        &bytes,
        &feed,
        crate::tests::equivalence::Tolerance::Abs(1e-6),
        |m| optimize_onnx_graph(m).expect("optimize succeeds"),
    );

    let mut model = load_onnx_model(&bytes).unwrap();
    optimize_onnx_graph(&mut model).expect("optimize succeeds");
    assert_eq!(
        model.node_count(),
        1,
        "both constant nodes should fold, leaving only the sink, got {:?}",
        model.nodes.iter().map(|n| &n.name).collect::<Vec<_>>()
    );
    assert_eq!(model.nodes[0].name, "sink");
}

/// The old implementation broke out of its loop on the first node it could not
/// evaluate, so one awkward operator stopped every later fold too. An
/// unevaluatable node is now simply skipped.
#[test]
fn fold_constants_skips_an_unfoldable_node_without_giving_up() {
    let nodes = vec![
        // Not a real operator, so the runner declines it.
        onnx::NodeProto {
            op_type: Some("NotAnOperator".into()),
            name: Some("awkward".into()),
            input: vec!["k1".into()],
            output: vec!["odd".into()],
            ..Default::default()
        },
        onnx::NodeProto {
            op_type: Some("Add".into()),
            name: Some("foldable".into()),
            input: vec!["k1".into(), "k2".into()],
            output: vec!["s".into()],
            ..Default::default()
        },
        onnx::NodeProto {
            op_type: Some("Add".into()),
            name: Some("sink".into()),
            input: vec!["x".into(), "s".into()],
            output: vec!["y".into()],
            ..Default::default()
        },
    ];
    let inits = vec![
        const_tensor("k1", vec![4], vec![1.0, 2.0, 3.0, 4.0]),
        const_tensor("k2", vec![4], vec![0.5, 0.5, 0.5, 0.5]),
    ];
    let bytes = build_minimal_onnx_model(nodes, inits, vec!["x"], vec!["y"]);

    let mut model = load_onnx_model(&bytes).unwrap();
    optimize_onnx_graph(&mut model).expect("optimize succeeds");

    assert!(
        !model.nodes.iter().any(|n| n.name == "foldable"),
        "the foldable node should still fold despite the earlier unfoldable one, got {:?}",
        model.nodes.iter().map(|n| &n.name).collect::<Vec<_>>()
    );
}

/// A constant node whose output feeds a graph output must keep its producer, or
/// lowering would drop the name the model promises.
#[test]
fn fold_constants_leaves_a_graph_output_alone() {
    let nodes = vec![onnx::NodeProto {
        op_type: Some("Add".into()),
        name: Some("a".into()),
        input: vec!["k1".into(), "k2".into()],
        output: vec!["y".into()],
        ..Default::default()
    }];
    let inits = vec![
        const_tensor("k1", vec![4], vec![1.0, 2.0, 3.0, 4.0]),
        const_tensor("k2", vec![4], vec![0.5, 0.5, 0.5, 0.5]),
    ];
    let bytes = build_minimal_onnx_model(nodes, inits, vec!["k1"], vec!["y"]);

    let mut model = load_onnx_model(&bytes).unwrap();
    optimize_onnx_graph(&mut model).expect("optimize succeeds");

    assert_eq!(
        model.node_count(),
        1,
        "the producer of a graph output stays"
    );
    assert_eq!(model.outputs, vec!["y".to_string()]);
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
    optimize_onnx_graph(&mut model).expect("optimize succeeds");
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
    let mut graph = model.to_ir();
    assert!(
        FuseActivation::conv_relu().run(&mut graph).unwrap(),
        "pass should fire"
    );
    model.apply_ir(&graph);

    assert_eq!(model.node_count(), 1);
    assert_eq!(model.nodes[0].op_type, "Conv_Relu");
    assert_eq!(
        model.nodes[0].outputs[0], "y",
        "the fused node must keep the Relu's output name, since it is a graph output"
    );
    assert_eq!(model.outputs, vec!["y".to_string()]);
}

/// The fusion rewrites the producer's output in place, so it must decline when
/// anything else reads that intermediate — otherwise the other reader would
/// observe post-activation values.
#[test]
fn fuse_conv_relu_declines_when_intermediate_has_another_reader() {
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
        onnx::NodeProto {
            op_type: Some("Sigmoid".into()),
            name: Some("other".into()),
            input: vec!["conv_out".into()],
            output: vec!["z".into()],
            ..Default::default()
        },
    ];
    let bytes = build_minimal_onnx_model(nodes, vec![], vec!["x", "w"], vec!["y", "z"]);
    let model = load_onnx_model(&bytes).unwrap();
    let mut graph = model.to_ir();

    assert!(
        !FuseActivation::conv_relu().run(&mut graph).unwrap(),
        "Conv output feeds a second consumer, so the pass must decline"
    );
}

/// Two branches exported interleaved (`convA, convB, reluA, reluB`), the order
/// multi-input models such as a Siamese tracker commonly get.
///
/// This used to be a test that `reorder_nodes_for_fusion` rescued positional
/// matching: the old `fuse_conv_relu` only inspected `nodes[i + 1]`, so neither
/// pair was adjacent and nothing fused until the reorder restored adjacency.
/// The fusion is now def-use based and fuses regardless of order, so what this
/// pins down is the end-to-end result — both pairs fused — rather than the
/// mechanism. Reordering still matters, but for the layer-3 plan builder, which
/// is still positional; see `reorder_restores_producer_consumer_adjacency`.
#[test]
fn interleaved_branches_both_fuse() {
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
    let weight = onnx::TensorProto {
        name: Some("w".into()),
        dims: vec![2, 3, 1, 1],
        data_type: Some(1),
        float_data: vec![0.3, -0.7, 0.5, 0.2, 0.9, -0.4],
        ..Default::default()
    };
    let bytes = build_minimal_onnx_model(nodes, vec![weight], vec!["xa", "xb"], vec!["ya", "yb"]);

    // Activation fusion moves no arithmetic, so it must be bitwise identical.
    let mut rng = crate::tests::equivalence::Lcg::new(0x0FF1_CE04);
    let mut feed = FxHashMap::default();
    feed.insert("xa".to_string(), rng.tensor(vec![1, 3, 4, 4]));
    feed.insert("xb".to_string(), rng.tensor(vec![1, 3, 4, 4]));
    crate::tests::equivalence::assert_transform_preserves_numerics(
        "optimize/interleaved-conv-relu",
        &bytes,
        &feed,
        crate::tests::equivalence::Tolerance::Exact,
        |m| optimize_onnx_graph(m).expect("optimize succeeds"),
    );

    let mut model = load_onnx_model(&bytes).unwrap();
    optimize_onnx_graph(&mut model).expect("optimize succeeds");
    assert_eq!(model.node_count(), 2, "both Conv+Relu pairs should fuse");
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

/// The def-use fusion no longer needs reordering, but the layer-3 plan builder
/// in `build_runtime_index` still matches on `nodes[i + 1]` / `nodes[i + 2]`,
/// so `reorder_nodes_for_fusion` remains load-bearing until that moves onto the
/// IR too. This pins the property the plan builder depends on.
#[test]
fn reorder_restores_producer_consumer_adjacency() {
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
            op_type: Some("Sigmoid".into()),
            name: Some("act_a".into()),
            input: vec!["conv_a_out".into()],
            output: vec!["ya".into()],
            ..Default::default()
        },
        onnx::NodeProto {
            op_type: Some("Sigmoid".into()),
            name: Some("act_b".into()),
            input: vec!["conv_b_out".into()],
            output: vec!["yb".into()],
            ..Default::default()
        },
    ];
    let bytes = build_minimal_onnx_model(nodes, vec![], vec!["xa", "xb", "w"], vec!["ya", "yb"]);
    let mut model = load_onnx_model(&bytes).unwrap();
    optimize_onnx_graph(&mut model).expect("optimize succeeds");

    // Sigmoid is not an activation any pass fuses, so all four nodes survive
    // and only the ordering is under test.
    assert_eq!(model.node_count(), 4);
    for (idx, node) in model.nodes.iter().enumerate() {
        if node.op_type != "Conv" {
            continue;
        }
        let consumer = &model.nodes[idx + 1];
        assert_eq!(
            consumer.inputs[0],
            node.outputs[0],
            "each Conv should be immediately followed by its consumer, got {:?}",
            model
                .nodes
                .iter()
                .map(|n| n.name.clone())
                .collect::<Vec<_>>()
        );
    }
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

    optimize_onnx_graph(&mut model).expect("optimize succeeds");

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
    use crate::optimizer::RewriteConvTransposeToDepthToSpace;
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
            let mut graph = model.to_ir();
            RewriteConvTransposeToDepthToSpace
                .run(&mut graph)
                .expect("pass runs");
            model.apply_ir(&graph);
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
    use crate::optimizer::RewriteConvTransposeToDepthToSpace;
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

    assert_transform_is_noop("rewrite_convtranspose_dts/k!=s", &bytes, &feed, |model| {
        let mut graph = model.to_ir();
        RewriteConvTransposeToDepthToSpace
            .run(&mut graph)
            .expect("pass runs");
        model.apply_ir(&graph);
    });

    let mut model = load_onnx_model(&bytes).unwrap();
    let mut graph = model.to_ir();
    RewriteConvTransposeToDepthToSpace
        .run(&mut graph)
        .expect("pass runs");
    model.apply_ir(&graph);
    assert!(model.nodes.iter().any(|n| n.op_type == "ConvTranspose"));
}

/// The DW+PW and PW+DW plan fusions used to find their partner by checking the
/// next non-skipped node, which is adjacency rather than dataflow: an unrelated
/// node scheduled between the pair silently cost the fusion. Both sites already
/// required the intermediate to have exactly one reader, so that reader is
/// unique and is now looked up directly.
///
/// This fixture puts a `Relu` on an independent branch between the depthwise
/// and pointwise convolutions. Node order is `dw, interloper, pw`, so the old
/// matcher saw `interloper` as the next node and gave up.
#[test]
fn plan_fuses_depthwise_pointwise_across_an_unrelated_node() {
    let dw_weight = onnx::TensorProto {
        name: Some("dw_w".into()),
        dims: vec![4, 1, 3, 3],
        data_type: Some(1),
        float_data: vec![0.05; 36],
        ..Default::default()
    };
    let pw_weight = onnx::TensorProto {
        name: Some("pw_w".into()),
        dims: vec![8, 4, 1, 1],
        data_type: Some(1),
        float_data: vec![0.1; 32],
        ..Default::default()
    };
    let nodes = vec![
        onnx::NodeProto {
            op_type: Some("Conv".into()),
            name: Some("dw".into()),
            input: vec!["x".into(), "dw_w".into()],
            output: vec!["dw_out".into()],
            attribute: vec![
                make_ints_attr("kernel_shape", vec![3, 3]),
                make_ints_attr("pads", vec![1, 1, 1, 1]),
                make_int_attr("group", 4),
            ],
            ..Default::default()
        },
        // Independent of the DW/PW pair, but scheduled between its halves.
        onnx::NodeProto {
            op_type: Some("Relu".into()),
            name: Some("interloper".into()),
            input: vec!["side_in".into()],
            output: vec!["side_out".into()],
            ..Default::default()
        },
        onnx::NodeProto {
            op_type: Some("Conv".into()),
            name: Some("pw".into()),
            input: vec!["dw_out".into(), "pw_w".into()],
            output: vec!["y".into()],
            attribute: vec![make_ints_attr("kernel_shape", vec![1, 1])],
            ..Default::default()
        },
    ];
    let bytes = build_minimal_onnx_model(
        nodes,
        vec![dw_weight, pw_weight],
        vec!["x", "side_in"],
        vec!["y", "side_out"],
    );
    let model = load_onnx_model(&bytes).unwrap();

    assert!(
        model
            .runtime_index
            .execution_plan
            .iter()
            .any(|a| matches!(a, crate::plan::NodeAction::FusedDwPw { .. })),
        "DW and PW should fuse despite the node between them, got {:?}",
        model.runtime_index.execution_plan
    );
}

/// `ConvAdd` matched `nodes[i + 1]` for the residual Add and `nodes[i + 2]` for
/// the optional Relu, so a residual block lost its in-place fusion whenever the
/// schedule put anything between the Conv and its Add — which a topological
/// sort of a branching graph legitimately can. Both steps already required a
/// single reader, so both are now looked up through the consumer index.
#[test]
fn plan_fuses_conv_add_across_an_unrelated_node() {
    let weight = onnx::TensorProto {
        name: Some("w".into()),
        dims: vec![4, 4, 1, 1],
        data_type: Some(1),
        float_data: vec![0.1; 16],
        ..Default::default()
    };
    let nodes = vec![
        onnx::NodeProto {
            op_type: Some("Conv".into()),
            name: Some("conv".into()),
            input: vec!["x".into(), "w".into()],
            output: vec!["conv_out".into()],
            attribute: vec![make_ints_attr("kernel_shape", vec![1, 1])],
            ..Default::default()
        },
        // Independent of the residual pair, but scheduled between its halves.
        onnx::NodeProto {
            op_type: Some("Relu".into()),
            name: Some("interloper".into()),
            input: vec!["side_in".into()],
            output: vec!["side_out".into()],
            ..Default::default()
        },
        onnx::NodeProto {
            op_type: Some("Add".into()),
            name: Some("residual".into()),
            input: vec!["conv_out".into(), "skip".into()],
            output: vec!["y".into()],
            ..Default::default()
        },
    ];
    let bytes = build_minimal_onnx_model(
        nodes,
        vec![weight],
        vec!["x", "skip", "side_in"],
        vec!["y", "side_out"],
    );
    let model = load_onnx_model(&bytes).unwrap();

    assert!(
        model
            .runtime_index
            .execution_plan
            .iter()
            .any(|a| matches!(a, crate::plan::NodeAction::ConvAdd { .. })),
        "Conv and its residual Add should fuse despite the node between them, got {:?}",
        model.runtime_index.execution_plan
    );
}

/// The DW+PW matcher backs off when the pointwise half would instead form the
/// stronger `ConvAdd`. That back-off used to `break` out of the inner search
/// loop; when the search became a lookup the `break` was left behind and now
/// exits the node loop itself, truncating the plan at the first inverted
/// bottleneck with a residual — i.e. on every MobileNet-shaped model. The plan
/// must describe every node.
#[test]
fn plan_covers_every_node_when_dw_pw_backs_off_to_conv_add() {
    let dw_weight = onnx::TensorProto {
        name: Some("dw_w".into()),
        dims: vec![4, 1, 3, 3],
        data_type: Some(1),
        float_data: vec![0.05; 36],
        ..Default::default()
    };
    let pw_weight = onnx::TensorProto {
        name: Some("pw_w".into()),
        dims: vec![4, 4, 1, 1],
        data_type: Some(1),
        float_data: vec![0.1; 16],
        ..Default::default()
    };
    let nodes = vec![
        onnx::NodeProto {
            op_type: Some("Conv".into()),
            name: Some("dw".into()),
            input: vec!["x".into(), "dw_w".into()],
            output: vec!["dw_out".into()],
            attribute: vec![
                make_ints_attr("kernel_shape", vec![3, 3]),
                make_ints_attr("pads", vec![1, 1, 1, 1]),
                make_int_attr("group", 4),
            ],
            ..Default::default()
        },
        onnx::NodeProto {
            op_type: Some("Conv".into()),
            name: Some("pw".into()),
            input: vec!["dw_out".into(), "pw_w".into()],
            output: vec!["pw_out".into()],
            attribute: vec![make_ints_attr("kernel_shape", vec![1, 1])],
            ..Default::default()
        },
        onnx::NodeProto {
            op_type: Some("Add".into()),
            name: Some("residual".into()),
            input: vec!["pw_out".into(), "skip".into()],
            output: vec!["sum".into()],
            ..Default::default()
        },
        // Trailing work, to make a truncated plan observable as a missing
        // output rather than only as a short plan.
        onnx::NodeProto {
            op_type: Some("Relu".into()),
            name: Some("tail".into()),
            input: vec!["sum".into()],
            output: vec!["y".into()],
            ..Default::default()
        },
    ];
    let bytes = build_minimal_onnx_model(
        nodes,
        vec![dw_weight, pw_weight],
        vec!["x", "skip"],
        vec!["y"],
    );
    let model = load_onnx_model(&bytes).unwrap();

    assert_eq!(
        model.runtime_index.execution_plan.len(),
        model.nodes.len(),
        "plan must cover every node, got {:?}",
        model.runtime_index.execution_plan
    );

    let mut feed = FxHashMap::default();
    feed.insert(
        "x".to_string(),
        Tensor::from_vec(vec![1, 4, 4, 4], vec![0.25_f32; 64]).unwrap(),
    );
    feed.insert(
        "skip".to_string(),
        Tensor::from_vec(vec![1, 4, 4, 4], vec![1.0_f32; 64]).unwrap(),
    );
    let out = run_onnx_model(&model, feed).unwrap();
    assert_eq!(out["y"].shape(), &[1, 4, 4, 4]);
}

/// A residual block gives its Add two Conv producers — the main path and the
/// shortcut — and each is the only reader of its own output, so each one on its
/// own looks like a `ConvAdd`. The fused action runs at the Conv's position and
/// reads the other branch from the environment, so only the *later* Conv may
/// claim the Add; fusing the earlier one schedules the Add before the other
/// branch has run and it reads a tensor nothing has written yet.
///
/// Adjacency hid this: the Add had to sit at `conv_idx + 1`, which only the
/// later Conv satisfies. Matching by dataflow does not, and resnet-18 stopped
/// loading with `missing input .../shortcut/convolution/Conv_output_0`.
#[test]
fn plan_fuses_conv_add_with_the_later_producer_of_a_residual() {
    let init = |name: &str, v: f32| onnx::TensorProto {
        name: Some(name.into()),
        dims: vec![1, 1, 1, 1],
        data_type: Some(1),
        float_data: vec![v],
        ..Default::default()
    };
    let conv = |name: &str, w: &str, out: &str| onnx::NodeProto {
        op_type: Some("Conv".into()),
        name: Some(name.into()),
        input: vec!["x".into(), w.into()],
        output: vec![out.into()],
        attribute: vec![make_ints_attr("kernel_shape", vec![1, 1])],
        ..Default::default()
    };
    let nodes = vec![
        conv("main", "w_main", "main_out"),
        conv("shortcut", "w_short", "short_out"),
        onnx::NodeProto {
            op_type: Some("Add".into()),
            name: Some("residual".into()),
            input: vec!["main_out".into(), "short_out".into()],
            output: vec!["y".into()],
            ..Default::default()
        },
    ];
    let bytes = build_minimal_onnx_model(
        nodes,
        vec![init("w_main", 2.0), init("w_short", 3.0)],
        vec!["x"],
        vec!["y"],
    );
    let model = load_onnx_model(&bytes).unwrap();

    // The shortcut Conv is the later producer, so it is the one that fuses.
    assert!(
        matches!(
            model.runtime_index.execution_plan.as_slice(),
            [
                crate::plan::NodeAction::Conv { node_idx: 0, .. },
                crate::plan::NodeAction::ConvAdd { conv_idx: 1, .. },
                crate::plan::NodeAction::Skip,
            ]
        ),
        "the later Conv should own the residual Add, got {:?}",
        model.runtime_index.execution_plan
    );

    let mut feed = FxHashMap::default();
    feed.insert(
        "x".to_string(),
        Tensor::from_vec(vec![1, 1, 2, 2], vec![1.0_f32; 4]).unwrap(),
    );
    let out = run_onnx_model(&model, feed).unwrap();
    assert_eq!(out["y"].data(), &[5.0_f32; 4]);
}

/// Two chained depthwise+pointwise blocks, both wide enough for the AVX-512
/// NCHWc16 kernel. The first should hand its output to the second in blocked
/// form; the second has no eligible consumer and must convert back.
///
/// `channels` picks whether the blocked kernel is eligible at all — it needs
/// both channel counts to be a multiple of 16.
// Depends on the loader pre-permuting conv weights: the fusion and
// handoff gates below match on the permuted shapes. GPU builds keep the
// ONNX-native OIHW layout, so the plan legitimately comes out different
// there and the shape this pins is not the one to expect.
#[cfg(not(any(feature = "metal-backend", feature = "gpu")))]
fn nchwc_handoff_for_chained_blocks(channels: usize) -> Vec<bool> {
    let c = channels as i64;
    let dw_w = |name: &str| onnx::TensorProto {
        name: Some(name.into()),
        dims: vec![c, 1, 3, 3],
        data_type: Some(1),
        float_data: vec![0.05; channels * 9],
        ..Default::default()
    };
    let pw_w = |name: &str| onnx::TensorProto {
        name: Some(name.into()),
        dims: vec![c, c, 1, 1],
        data_type: Some(1),
        float_data: vec![0.1; channels * channels],
        ..Default::default()
    };
    let dw = |name: &str, w: &str, inp: &str, out: &str| onnx::NodeProto {
        op_type: Some("Conv".into()),
        name: Some(name.into()),
        input: vec![inp.into(), w.into()],
        output: vec![out.into()],
        attribute: vec![
            make_ints_attr("kernel_shape", vec![3, 3]),
            make_ints_attr("pads", vec![1, 1, 1, 1]),
            make_int_attr("group", c),
        ],
        ..Default::default()
    };
    let pw = |name: &str, w: &str, inp: &str, out: &str| onnx::NodeProto {
        op_type: Some("Conv".into()),
        name: Some(name.into()),
        input: vec![inp.into(), w.into()],
        output: vec![out.into()],
        attribute: vec![make_ints_attr("kernel_shape", vec![1, 1])],
        ..Default::default()
    };
    let nodes = vec![
        dw("dw1", "dw1_w", "x", "dw1_out"),
        pw("pw1", "pw1_w", "dw1_out", "pw1_out"),
        dw("dw2", "dw2_w", "pw1_out", "dw2_out"),
        pw("pw2", "pw2_w", "dw2_out", "y"),
    ];
    let bytes = build_minimal_onnx_model(
        nodes,
        vec![dw_w("dw1_w"), pw_w("pw1_w"), dw_w("dw2_w"), pw_w("pw2_w")],
        vec!["x"],
        vec!["y"],
    );
    let model = load_onnx_model(&bytes).unwrap();
    assert!(
        matches!(
            model.runtime_index.execution_plan.as_slice(),
            [
                crate::plan::NodeAction::FusedDwPw { .. },
                crate::plan::NodeAction::Skip,
                crate::plan::NodeAction::FusedDwPw { .. },
                crate::plan::NodeAction::Skip,
            ]
        ),
        "fixture must produce two chained FusedDwPw actions, got {:?}",
        model.runtime_index.execution_plan
    );
    model.runtime_index.nchwc_handoff.clone()
}

/// The runner used to decide the NCHWc handoff per action per inference, by
/// walking forward through the plan and pulling weight shapes back out of the
/// tensor environment. Every input to that decision is fixed at load time.
///
/// Checked here rather than through a run because the kernel that consumes the
/// flag is gated on AVX-512, which the development host does not have — the
/// predicate itself is what changed, so the predicate is what gets pinned.
#[cfg(not(any(feature = "metal-backend", feature = "gpu")))]
#[test]
fn nchwc_handoff_is_resolved_at_plan_time() {
    // c = 16: both blocks clear the blocked kernel's channel gate, so the
    // first hands off to the second. The second has nothing after it.
    assert_eq!(
        nchwc_handoff_for_chained_blocks(16),
        vec![true, false, false, false]
    );

    // c = 8: the kernel would fall back to the NHWC path, and handing it a
    // blocked tensor there is a crash rather than a slowdown.
    assert_eq!(
        nchwc_handoff_for_chained_blocks(8),
        vec![false, false, false, false]
    );
}

/// `FusedPwDwPwReduce` merges a `PW_expand → DW 3×3 → PW_reduce` inverted
/// bottleneck into one streaming action. It found the PW reduce by taking the
/// first non-skipped node after the depthwise — adjacency, the pattern the rest
/// of the fusion scan moved off — so a node scheduled between the depthwise and
/// its consumer cost the merge and the block fell back to two actions.
///
/// The DW output already has to have exactly one reader for the merge to be
/// legal, so that reader is unique and is now looked up directly.
#[cfg(not(any(feature = "metal-backend", feature = "gpu")))]
#[test]
fn plan_merges_pw_dw_pw_reduce_across_an_unrelated_node() {
    // Held for the whole test: the plan assertion below reads the same switch
    // the harness toggles, so it races a sibling test without this.
    let env = crate::tests::equivalence::lock_env();
    let c_in = 16usize;
    let c_exp = 32usize;
    let w = |name: &str, dims: Vec<i64>| onnx::TensorProto {
        name: Some(name.into()),
        dims: dims.clone(),
        data_type: Some(1),
        float_data: vec![0.05; dims.iter().product::<i64>() as usize],
        ..Default::default()
    };
    let nodes = vec![
        onnx::NodeProto {
            op_type: Some("Conv".into()),
            name: Some("pw_expand".into()),
            input: vec!["x".into(), "exp_w".into()],
            output: vec!["exp".into()],
            attribute: vec![make_ints_attr("kernel_shape", vec![1, 1])],
            ..Default::default()
        },
        onnx::NodeProto {
            op_type: Some("Conv".into()),
            name: Some("dw".into()),
            input: vec!["exp".into(), "dw_w".into()],
            output: vec!["dwo".into()],
            attribute: vec![
                make_ints_attr("kernel_shape", vec![3, 3]),
                make_ints_attr("pads", vec![1, 1, 1, 1]),
                make_int_attr("group", c_exp as i64),
            ],
            ..Default::default()
        },
        // Independent of the bottleneck, but scheduled inside it.
        onnx::NodeProto {
            op_type: Some("Relu".into()),
            name: Some("interloper".into()),
            input: vec!["side_in".into()],
            output: vec!["side_out".into()],
            ..Default::default()
        },
        onnx::NodeProto {
            op_type: Some("Conv".into()),
            name: Some("pw_reduce".into()),
            input: vec!["dwo".into(), "red_w".into()],
            output: vec!["y".into()],
            attribute: vec![make_ints_attr("kernel_shape", vec![1, 1])],
            ..Default::default()
        },
    ];
    let bytes = build_minimal_onnx_model(
        nodes,
        vec![
            w("exp_w", vec![c_exp as i64, c_in as i64, 1, 1]),
            w("dw_w", vec![c_exp as i64, 1, 3, 3]),
            w("red_w", vec![c_in as i64, c_exp as i64, 1, 1]),
        ],
        vec!["x", "side_in"],
        vec!["y", "side_out"],
    );
    let model = load_onnx_model(&bytes).unwrap();

    assert!(
        model
            .runtime_index
            .execution_plan
            .iter()
            .any(|a| matches!(a, crate::plan::NodeAction::FusedPwDwPwReduce { .. })),
        "the bottleneck should merge despite the node between its halves, got {:?}",
        model.runtime_index.execution_plan
    );

    // The merge rewrites how the block is computed, so check it still computes
    // the same thing as the unmerged plan — every output of it, not just `y`.
    let mut feed = FxHashMap::default();
    feed.insert(
        "x".to_string(),
        Tensor::from_vec(vec![1, c_in, 8, 8], vec![0.25_f32; c_in * 64]).unwrap(),
    );
    feed.insert(
        "side_in".to_string(),
        Tensor::from_vec(vec![1, 2], vec![1.0_f32, -1.0]).unwrap(),
    );
    crate::tests::equivalence::assert_plan_fusion_preserves_numerics(
        &env,
        "FusedPwDwPwReduce/non-adjacent",
        &bytes,
        &feed,
        crate::tests::equivalence::Tolerance::Abs(1e-5),
        "YSCV_FUSED_PW_DW_PW_REDUCE_OFF",
    );
}

/// Builds `PW_expand → DW 3×3 → PW_reduce → Add(residual)` plus one unrelated
/// `Relu`, and returns the model bytes with a feed.
///
/// `residual_from_input` selects between the two ways the merge and `ConvAdd`
/// can disagree, which need the interloper in different places:
///
/// * `true` — residual is the graph input, interloper scheduled *between the PW
///   reduce and the Add*, so the Add is not at `pw_reduce_idx + 1`.
/// * `false` — residual is the interloper's output and the interloper is
///   scheduled *inside the block*, so it is computed after the merged action's
///   anchor but before the PW reduce.
fn pw_dw_pw_reduce_residual_fixture(
    residual_from_input: bool,
) -> (Vec<u8>, FxHashMap<String, Tensor>) {
    let c_in = 16usize;
    let c_exp = 32usize;
    let w = |name: &str, dims: Vec<i64>| onnx::TensorProto {
        name: Some(name.into()),
        dims: dims.clone(),
        data_type: Some(1),
        float_data: vec![0.05; dims.iter().product::<i64>() as usize],
        ..Default::default()
    };
    let conv = |name: &str, wn: &str, i: &str, o: &str, k: Vec<i64>, extra: Vec<_>| {
        let mut attribute = vec![make_ints_attr("kernel_shape", k)];
        attribute.extend(extra);
        onnx::NodeProto {
            op_type: Some("Conv".into()),
            name: Some(name.into()),
            input: vec![i.into(), wn.into()],
            output: vec![o.into()],
            attribute,
            ..Default::default()
        }
    };
    let interloper = onnx::NodeProto {
        op_type: Some("Relu".into()),
        name: Some("interloper".into()),
        input: vec!["x".into()],
        output: vec!["mid".into()],
        ..Default::default()
    };
    let dw = conv(
        "dw",
        "dw_w",
        "exp",
        "dwo",
        vec![3, 3],
        vec![
            make_ints_attr("pads", vec![1, 1, 1, 1]),
            make_int_attr("group", c_exp as i64),
        ],
    );
    let pw_expand = conv("pw_expand", "exp_w", "x", "exp", vec![1, 1], vec![]);
    let pw_reduce = conv("pw_reduce", "red_w", "dwo", "red", vec![1, 1], vec![]);
    let add = |operand: &str| onnx::NodeProto {
        op_type: Some("Add".into()),
        name: Some("residual".into()),
        input: vec!["red".into(), operand.into()],
        output: vec!["y".into()],
        ..Default::default()
    };
    let nodes = if residual_from_input {
        vec![pw_expand, dw, pw_reduce, interloper, add("x")]
    } else {
        vec![pw_expand, interloper, dw, pw_reduce, add("mid")]
    };
    // `mid` is a graph output in the first case so the interloper is not dead
    // code; in the second the Add already consumes it.
    let outputs: Vec<&str> = if residual_from_input {
        vec!["y", "mid"]
    } else {
        vec!["y"]
    };
    let bytes = build_minimal_onnx_model(
        nodes,
        vec![
            w("exp_w", vec![c_exp as i64, c_in as i64, 1, 1]),
            w("dw_w", vec![c_exp as i64, 1, 3, 3]),
            w("red_w", vec![c_in as i64, c_exp as i64, 1, 1]),
        ],
        vec!["x"],
        outputs,
    );
    let mut feed = FxHashMap::default();
    feed.insert(
        "x".to_string(),
        Tensor::from_vec(vec![1, c_in, 8, 8], vec![0.25_f32; c_in * 64]).unwrap(),
    );
    (bytes, feed)
}

/// `FusedPwDwPwReduce` looked for its residual `Add` at `pw_reduce_idx + 1`
/// after every other matcher had moved to dataflow, so the two disagreed.
///
/// When `ConvAdd` claimed a non-adjacent `Add`, the merge absorbed the PW
/// reduce without noticing the `Add`; the `retain` that drops actions whose
/// node was absorbed then deleted the whole `ConvAdd`, and the `Add`'s own plan
/// slot was already `Skip`. The addition vanished and the run returned `Ok`
/// with the graph output missing — no error anywhere.
///
/// The merge now takes over whatever `ConvAdd` resolved, or declines.
#[cfg(not(any(feature = "metal-backend", feature = "gpu")))]
#[test]
fn plan_merge_keeps_a_non_adjacent_residual_add() {
    let env = crate::tests::equivalence::lock_env();
    let (bytes, feed) = pw_dw_pw_reduce_residual_fixture(true);
    let model = load_onnx_model(&bytes).unwrap();
    assert!(
        model
            .runtime_index
            .execution_plan
            .iter()
            .any(|a| matches!(a, crate::plan::NodeAction::FusedPwDwPwReduce { .. })),
        "fixture must exercise the merge, got {:?}",
        model.runtime_index.execution_plan
    );

    crate::tests::equivalence::assert_plan_fusion_preserves_numerics(
        &env,
        "FusedPwDwPwReduce/non-adjacent-residual",
        &bytes,
        &feed,
        crate::tests::equivalence::Tolerance::Abs(1e-5),
        "YSCV_FUSED_PW_DW_PW_REDUCE_OFF",
    );
}

/// The mirror of the above: the merged action runs at the PW expand's position,
/// so a residual operand produced *inside* the block does not exist yet.
/// `ConvAdd` declines this through `available_at`; the merge used to absorb it
/// regardless and fail the run with a missing input.
/// No kill-switch comparison here: the merge declines either way, so toggling
/// it would prove nothing. What this pins is that it declines at all — before
/// the fix the merged action read a tensor nothing had written and the run
/// failed with `MissingInput { node: "residual", input: "mid" }`.
#[test]
fn plan_merge_declines_a_residual_produced_inside_the_block() {
    let _env = crate::tests::equivalence::lock_env();
    let (bytes, feed) = pw_dw_pw_reduce_residual_fixture(false);
    let model = load_onnx_model(&bytes).unwrap();

    assert!(
        !model
            .runtime_index
            .execution_plan
            .iter()
            .any(|a| matches!(a, crate::plan::NodeAction::FusedPwDwPwReduce { .. })),
        "merge must decline when the residual is produced inside the block, got {:?}",
        model.runtime_index.execution_plan
    );

    let out = run_onnx_model(&model, feed).expect("plan must be executable");
    assert!(out.contains_key("y"), "graph output dropped from the plan");
}

/// The plan's kernel choice has to agree with the branch the dispatch takes,
/// on real graphs — that is the whole basis for later replacing those branches
/// with a match on the plan. `note_kernel` asserts it per Conv in debug builds,
/// so running the models is the check.
///
/// Ignored by default: it needs the benchmark assets, which are downloaded
/// rather than committed. Run with `--ignored` when they are present.
#[test]
#[ignore]
fn planned_conv_kernels_match_the_dispatch_on_real_models() {
    for (file, shape) in [
        ("resnet-18.onnx", vec![1usize, 3, 224, 224]),
        ("mobilenet-v3-small.onnx", vec![1, 3, 224, 224]),
    ] {
        let path = format!(
            "{}/../../benchmarks/onnx-models/target/assets/{file}",
            env!("CARGO_MANIFEST_DIR")
        );
        let mut model = crate::load_onnx_model_from_file(&path).unwrap_or_else(|e| {
            panic!(
                "{file}: {e}. This test needs the benchmark assets; run \
                 benchmarks/onnx-models/download-assets.sh first."
            )
        });
        optimize_onnx_graph(&mut model).expect("optimize");
        let planned = model
            .runtime_index
            .conv_kernels
            .iter()
            .filter(|k| k.is_some())
            .count();
        assert!(planned > 0, "{file}: plan resolved no conv kernels");

        let n: usize = shape.iter().product();
        let mut feed = FxHashMap::default();
        feed.insert(
            model.inputs[0].clone(),
            Tensor::from_vec(shape, vec![0.25_f32; n]).unwrap(),
        );
        // The per-Conv debug assertion inside `note_kernel` is what this run
        // exercises; reaching the end means every dispatch matched the plan.
        run_onnx_model(&model, feed).expect("run");
    }
}

/// `FusedTransposeMatMul` elides a `Transpose(perm=[0,2,1])` feeding a MatMul's
/// left input by dispatching a `transA=1` GEMM instead. It had no test.
///
/// Two things need pinning. The fusion must fire, and — because the post-pass
/// only turns the Transpose into `Skip` when *every* consumer absorbed it — a
/// Transpose with a second, non-MatMul consumer must still be materialized.
#[test]
fn plan_fuses_transpose_into_matmul_and_keeps_a_shared_transpose() {
    let b_init = onnx::TensorProto {
        name: Some("b".into()),
        dims: vec![1, 2, 3],
        data_type: Some(1),
        float_data: (0..6).map(|v| v as f32 * 0.25).collect(),
        ..Default::default()
    };
    let build = |extra_consumer: bool| {
        let mut nodes = vec![
            onnx::NodeProto {
                op_type: Some("Transpose".into()),
                name: Some("t".into()),
                input: vec!["x".into()],
                output: vec!["xt".into()],
                attribute: vec![make_ints_attr("perm", vec![0, 2, 1])],
                ..Default::default()
            },
            onnx::NodeProto {
                op_type: Some("MatMul".into()),
                name: Some("mm".into()),
                input: vec!["xt".into(), "b".into()],
                output: vec!["y".into()],
                ..Default::default()
            },
        ];
        let mut outs = vec!["y"];
        if extra_consumer {
            nodes.push(onnx::NodeProto {
                op_type: Some("Relu".into()),
                name: Some("side".into()),
                input: vec!["xt".into()],
                output: vec!["side_out".into()],
                ..Default::default()
            });
            outs.push("side_out");
        }
        build_minimal_onnx_model(nodes, vec![b_init.clone()], vec!["x"], outs)
    };

    let mut rng = crate::tests::equivalence::Lcg::new(0x7EA5_0001);
    let feed_for = |rng: &mut crate::tests::equivalence::Lcg| {
        let mut f = FxHashMap::default();
        f.insert("x".to_string(), rng.tensor(vec![1, 2, 2]));
        f
    };

    // Sole consumer: the Transpose is absorbed and its own slot becomes Skip.
    let bytes = build(false);
    let model = load_onnx_model(&bytes).unwrap();
    assert!(
        matches!(
            model.runtime_index.execution_plan.as_slice(),
            [
                crate::plan::NodeAction::Skip,
                crate::plan::NodeAction::FusedTransposeMatMul { .. },
            ]
        ),
        "sole-consumer Transpose should be absorbed, got {:?}",
        model.runtime_index.execution_plan
    );
    let out = run_onnx_model(&model, feed_for(&mut rng)).unwrap();
    assert_eq!(out["y"].shape(), &[1, 2, 3]);

    // Shared: `side` also reads the transposed value, so eliding the Transpose
    // would leave it reading a tensor nothing wrote.
    let bytes = build(true);
    let model = load_onnx_model(&bytes).unwrap();
    assert!(
        !model
            .runtime_index
            .execution_plan
            .iter()
            .any(|a| matches!(a, crate::plan::NodeAction::FusedTransposeMatMul { .. })),
        "a Transpose with a non-MatMul consumer must not be fused away, got {:?}",
        model.runtime_index.execution_plan
    );
    let out = run_onnx_model(&model, feed_for(&mut rng)).unwrap();
    let mut names: Vec<&String> = out.keys().collect();
    names.sort();
    assert_eq!(names, vec!["side_out", "y"], "output names diverged");
}
