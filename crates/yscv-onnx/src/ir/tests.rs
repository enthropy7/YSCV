//! IR unit tests.
//!
//! The def-use index is redundant state — it duplicates information already
//! present in the node list — so the risk is drift. Every mutation already
//! calls [`Graph::validate`] behind a debug assertion; these tests additionally
//! check that the index says the *right* thing, and that lowering round-trips.

use super::*;
use crate::loader::load_onnx_model;
use crate::proto::onnx;
use crate::tests::build_minimal_onnx_model;

/// `x -> Relu -> mid -> Relu -> y`.
fn two_relu_bytes() -> Vec<u8> {
    let nodes = vec![
        onnx::NodeProto {
            op_type: Some("Relu".into()),
            name: Some("r0".into()),
            input: vec!["x".into()],
            output: vec!["mid".into()],
            ..Default::default()
        },
        onnx::NodeProto {
            op_type: Some("Relu".into()),
            name: Some("r1".into()),
            input: vec!["mid".into()],
            output: vec!["y".into()],
            ..Default::default()
        },
    ];
    build_minimal_onnx_model(nodes, vec![], vec!["x"], vec!["y"])
}

/// A value read by two nodes, so fan-out queries have something to answer.
fn fanout_bytes() -> Vec<u8> {
    let nodes = vec![
        onnx::NodeProto {
            op_type: Some("Relu".into()),
            name: Some("src".into()),
            input: vec!["x".into()],
            output: vec!["mid".into()],
            ..Default::default()
        },
        onnx::NodeProto {
            op_type: Some("Sigmoid".into()),
            name: Some("a".into()),
            input: vec!["mid".into()],
            output: vec!["ya".into()],
            ..Default::default()
        },
        onnx::NodeProto {
            op_type: Some("Tanh".into()),
            name: Some("b".into()),
            input: vec!["mid".into()],
            output: vec!["yb".into()],
            ..Default::default()
        },
    ];
    build_minimal_onnx_model(nodes, vec![], vec!["x"], vec!["ya", "yb"])
}

fn graph_of(bytes: &[u8]) -> Graph {
    load_onnx_model(bytes).expect("fixture loads").to_ir()
}

#[test]
fn to_ir_builds_a_valid_index() {
    let graph = graph_of(&two_relu_bytes());
    assert_eq!(graph.validate(), Ok(()));
    assert_eq!(graph.node_count(), 2);
}

#[test]
fn def_and_uses_point_at_the_right_nodes() {
    let graph = graph_of(&two_relu_bytes());
    let ids: Vec<NodeId> = graph.node_ids().collect();
    let mid = graph.lookup_value("mid").expect("mid exists");

    assert_eq!(graph.value(mid).def, Some(ids[0]), "mid is produced by r0");
    assert_eq!(
        graph.value(mid).uses,
        vec![Use {
            node: ids[1],
            port: 0
        }],
        "mid is consumed by r1 at port 0"
    );
}

#[test]
fn sole_consumer_reports_single_use() {
    let graph = graph_of(&two_relu_bytes());
    let ids: Vec<NodeId> = graph.node_ids().collect();
    let mid = graph.lookup_value("mid").expect("mid exists");

    assert_eq!(
        graph.sole_consumer(mid),
        Some(Use {
            node: ids[1],
            port: 0
        })
    );
}

/// The query has to say "no" on fan-out, or the fold passes it replaces would
/// absorb a producer that another branch still reads.
#[test]
fn sole_consumer_declines_on_fanout() {
    let graph = graph_of(&fanout_bytes());
    let mid = graph.lookup_value("mid").expect("mid exists");

    assert_eq!(graph.use_count(mid), 2);
    assert_eq!(graph.sole_consumer(mid), None);
}

/// A value consumed once but also exposed as a graph output is not
/// exclusively owned by that consumer, so fusing into it would drop an output.
#[test]
fn sole_consumer_declines_on_graph_output() {
    let nodes = vec![
        onnx::NodeProto {
            op_type: Some("Relu".into()),
            name: Some("r0".into()),
            input: vec!["x".into()],
            output: vec!["mid".into()],
            ..Default::default()
        },
        onnx::NodeProto {
            op_type: Some("Relu".into()),
            name: Some("r1".into()),
            input: vec!["mid".into()],
            output: vec!["y".into()],
            ..Default::default()
        },
    ];
    let bytes = build_minimal_onnx_model(nodes, vec![], vec!["x"], vec!["mid", "y"]);
    let graph = graph_of(&bytes);
    let mid = graph.lookup_value("mid").expect("mid exists");

    assert_eq!(graph.use_count(mid), 1);
    assert_eq!(graph.sole_consumer(mid), None);
}

#[test]
fn replace_all_uses_rewires_every_consumer() {
    let mut graph = graph_of(&fanout_bytes());
    let mid = graph.lookup_value("mid").expect("mid exists");
    let x = graph.lookup_value("x").expect("x exists");

    graph.replace_all_uses_with(mid, x);

    assert_eq!(graph.validate(), Ok(()));
    assert_eq!(graph.use_count(mid), 0);
    assert_eq!(graph.use_count(x), 3, "x now feeds src, a and b");
}

#[test]
fn replace_all_uses_transfers_graph_output_status() {
    let mut graph = graph_of(&two_relu_bytes());
    let mid = graph.lookup_value("mid").expect("mid exists");
    let y = graph.lookup_value("y").expect("y exists");

    graph.replace_all_uses_with(y, mid);

    assert!(graph.is_graph_output(mid));
    assert!(!graph.is_graph_output(y));
}

/// Removing a node the way a pass does: rewire consumers, then drop it.
#[test]
fn remove_node_unlinks_cleanly() {
    let mut graph = graph_of(&two_relu_bytes());
    let ids: Vec<NodeId> = graph.node_ids().collect();
    let x = graph.lookup_value("x").expect("x exists");
    let mid = graph.lookup_value("mid").expect("mid exists");

    graph.replace_all_uses_with(mid, x);
    graph.remove_node(ids[0]);

    assert_eq!(graph.validate(), Ok(()));
    assert_eq!(graph.node_count(), 1);
    assert_eq!(graph.value(mid).def, None);
    assert_eq!(graph.use_count(x), 1, "only r1 reads x now");
}

/// Ids must survive removal, since passes collect candidates in one sweep and
/// rewrite them in another.
#[test]
fn ids_stay_valid_after_removal() {
    let mut graph = graph_of(&fanout_bytes());
    let ids: Vec<NodeId> = graph.node_ids().collect();
    let ya = graph.lookup_value("ya").expect("ya exists");

    graph.replace_all_uses_with(ya, graph.lookup_value("x").expect("x exists"));
    graph.remove_node(ids[1]);

    assert!(graph.node(ids[1]).is_none(), "removed slot is tombstoned");
    assert_eq!(
        graph.node(ids[2]).map(|n| n.name.as_str()),
        Some("b"),
        "later ids still resolve"
    );
}

#[test]
fn compact_drops_tombstoned_ids_from_order() {
    let mut graph = graph_of(&fanout_bytes());
    let ids: Vec<NodeId> = graph.node_ids().collect();
    graph.replace_all_uses_with(
        graph.lookup_value("ya").expect("ya exists"),
        graph.lookup_value("x").expect("x exists"),
    );
    graph.remove_node(ids[1]);

    assert_eq!(graph.node_count(), 2);
    graph.compact();
    assert_eq!(graph.node_count(), 2, "compaction changes no live node");
    assert_eq!(graph.validate(), Ok(()));
}

/// `Sub(x, x)` reads one value at two ports. The use list has to record both,
/// or a pass would conclude the operand had a single owner and fuse into it.
#[test]
fn repeated_operand_tracks_both_ports() {
    let nodes = vec![onnx::NodeProto {
        op_type: Some("Sub".into()),
        name: Some("s".into()),
        input: vec!["x".into(), "x".into()],
        output: vec!["y".into()],
        ..Default::default()
    }];
    let bytes = build_minimal_onnx_model(nodes, vec![], vec!["x"], vec!["y"]);
    let graph = graph_of(&bytes);
    let ids: Vec<NodeId> = graph.node_ids().collect();
    let x = graph.lookup_value("x").expect("x exists");

    assert_eq!(graph.use_count(x), 2, "both ports count as uses");
    assert_eq!(graph.sole_consumer(x), None);
    assert_eq!(
        graph.value(x).uses,
        vec![
            Use {
                node: ids[0],
                port: 0
            },
            Use {
                node: ids[0],
                port: 1
            }
        ],
        "each port is recorded separately"
    );
}

/// A Conv without bias has an empty third input name in ONNX; that must stay a
/// hole and not become a use of some value named "".
#[test]
fn omitted_optional_input_is_none() {
    let nodes = vec![onnx::NodeProto {
        op_type: Some("Conv".into()),
        name: Some("c".into()),
        input: vec!["x".into(), "w".into(), String::new()],
        output: vec!["y".into()],
        ..Default::default()
    }];
    let weight = onnx::TensorProto {
        name: Some("w".into()),
        dims: vec![1, 1, 1, 1],
        data_type: Some(1),
        float_data: vec![1.0],
        ..Default::default()
    };
    let bytes = build_minimal_onnx_model(nodes, vec![weight], vec!["x"], vec!["y"]);
    let graph = graph_of(&bytes);
    let ids: Vec<NodeId> = graph.node_ids().collect();

    let node = graph.node(ids[0]).expect("node is live");
    assert_eq!(node.inputs.len(), 3);
    assert!(node.inputs[2].is_none(), "empty name is an omitted input");
    assert!(
        graph.lookup_value("").is_none(),
        "no value was interned for it"
    );
}

#[test]
fn constants_are_visible_as_values() {
    let nodes = vec![onnx::NodeProto {
        op_type: Some("Conv".into()),
        name: Some("c".into()),
        input: vec!["x".into(), "w".into()],
        output: vec!["y".into()],
        ..Default::default()
    }];
    let weight = onnx::TensorProto {
        name: Some("w".into()),
        dims: vec![1, 1, 1, 1],
        data_type: Some(1),
        float_data: vec![0.5],
        ..Default::default()
    };
    let bytes = build_minimal_onnx_model(nodes, vec![weight], vec!["x"], vec!["y"]);
    let graph = graph_of(&bytes);

    let w = graph.lookup_value("w").expect("w exists");
    assert_eq!(graph.constant(w).map(|t| t.data()[0]), Some(0.5));
    assert!(
        graph
            .constant(graph.lookup_value("x").expect("x exists"))
            .is_none(),
        "graph inputs are not constants"
    );
}

/// A reorder that is not a permutation would drop or duplicate execution of a
/// node, so both shapes are rejected rather than applied.
#[test]
fn set_order_rejects_a_non_permutation() {
    let mut graph = graph_of(&fanout_bytes());
    let ids: Vec<NodeId> = graph.node_ids().collect();

    assert!(
        matches!(
            graph.set_order(vec![ids[0], ids[1]]),
            Err(IrError::NotAPermutation { .. })
        ),
        "a short order would drop a node"
    );
    assert!(
        matches!(
            graph.set_order(vec![ids[0], ids[1], ids[1]]),
            Err(IrError::NotAPermutation { .. })
        ),
        "a duplicate would run one node twice and drop another"
    );

    assert_eq!(graph.set_order(vec![ids[2], ids[1], ids[0]]), Ok(()));
    assert_eq!(
        graph.node_ids().collect::<Vec<_>>(),
        vec![ids[2], ids[1], ids[0]]
    );
}

/// `validate` is the safety net for every mutation, so it has to actually
/// detect a corrupt index rather than always returning `Ok`.
#[test]
fn validate_detects_a_corrupted_use_list() {
    let mut graph = graph_of(&two_relu_bytes());
    let mid = graph.lookup_value("mid").expect("mid exists");
    graph.values[mid.idx()].uses.clear();

    assert!(matches!(
        graph.validate(),
        Err(IrError::InconsistentUses { .. })
    ));
}

#[test]
fn validate_detects_a_stale_def() {
    let mut graph = graph_of(&two_relu_bytes());
    let mid = graph.lookup_value("mid").expect("mid exists");
    graph.values[mid.idx()].def = None;

    assert!(matches!(graph.validate(), Err(IrError::StaleDef { .. })));
}

// ── Lowering ─────────────────────────────────────────────────────────────

/// Snapshot of everything `apply_ir` is responsible for reproducing.
fn model_shape(
    model: &crate::loader::OnnxModel,
) -> (
    Vec<(String, String, Vec<String>, Vec<String>)>,
    Vec<String>,
    Vec<String>,
    Vec<String>,
) {
    let nodes = model
        .nodes
        .iter()
        .map(|n| {
            (
                n.op_type.clone(),
                n.name.clone(),
                n.inputs.clone(),
                n.outputs.clone(),
            )
        })
        .collect();
    let mut initializers: Vec<String> = model.initializers.keys().cloned().collect();
    initializers.sort();
    (
        nodes,
        initializers,
        model.inputs.clone(),
        model.outputs.clone(),
    )
}

#[test]
fn round_trip_preserves_graph_structure() {
    for bytes in [two_relu_bytes(), fanout_bytes()] {
        let mut model = load_onnx_model(&bytes).expect("fixture loads");
        let before = model_shape(&model);

        let graph = model.to_ir();
        model.apply_ir(&graph);

        assert_eq!(model_shape(&model), before, "round-trip changed the model");
    }
}

/// Weights and omitted optional inputs are the two things a naive round-trip
/// loses, so exercise a graph that has both.
#[test]
fn round_trip_preserves_weights_and_optional_inputs() {
    let nodes = vec![onnx::NodeProto {
        op_type: Some("Conv".into()),
        name: Some("c".into()),
        input: vec!["x".into(), "w".into(), String::new()],
        output: vec!["y".into()],
        ..Default::default()
    }];
    let weight = onnx::TensorProto {
        name: Some("w".into()),
        dims: vec![1, 1, 1, 1],
        data_type: Some(1),
        float_data: vec![0.25],
        ..Default::default()
    };
    let bytes = build_minimal_onnx_model(nodes, vec![weight], vec!["x"], vec!["y"]);

    let mut model = load_onnx_model(&bytes).expect("fixture loads");
    let before = model_shape(&model);
    let before_weight = model.initializers["w"].data().to_vec();

    let graph = model.to_ir();
    model.apply_ir(&graph);

    assert_eq!(model_shape(&model), before);
    assert_eq!(model.initializers["w"].data(), before_weight.as_slice());
    assert_eq!(model.nodes[0].inputs[2], "", "optional input stays omitted");
}

/// Operators the IR does not intern must survive lowering unchanged, or every
/// optimized model would lose its long-tail ops.
#[test]
fn round_trip_preserves_unknown_operators() {
    let nodes = vec![onnx::NodeProto {
        op_type: Some("ScatterElements".into()),
        name: Some("s".into()),
        input: vec!["x".into()],
        output: vec!["y".into()],
        ..Default::default()
    }];
    let bytes = build_minimal_onnx_model(nodes, vec![], vec!["x"], vec!["y"]);

    let mut model = load_onnx_model(&bytes).expect("fixture loads");
    let graph = model.to_ir();
    model.apply_ir(&graph);

    assert_eq!(model.nodes[0].op_type, "ScatterElements");
}

/// Tombstoned nodes must not reappear on the way out.
#[test]
fn apply_ir_drops_removed_nodes() {
    let bytes = two_relu_bytes();
    let mut model = load_onnx_model(&bytes).expect("fixture loads");
    let mut graph = model.to_ir();

    let ids: Vec<NodeId> = graph.node_ids().collect();
    let x = graph.lookup_value("x").expect("x exists");
    let mid = graph.lookup_value("mid").expect("mid exists");
    graph.replace_all_uses_with(mid, x);
    graph.remove_node(ids[0]);

    model.apply_ir(&graph);

    assert_eq!(model.nodes.len(), 1);
    assert_eq!(model.nodes[0].name, "r1");
    assert_eq!(model.nodes[0].inputs[0], "x", "consumer was rewired");
}
