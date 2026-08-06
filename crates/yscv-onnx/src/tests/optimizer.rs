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
