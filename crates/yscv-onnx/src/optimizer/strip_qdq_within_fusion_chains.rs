use rustc_hash::FxHashMap;

use crate::loader::{OnnxModel, OnnxNode};

/// Fuses consecutive Conv + Relu into a single annotated node.
/// QDQ-aware: strips redundant `QuantizeLinear → DequantizeLinear` pairs
/// whose only purpose is to sit between two fusion-eligible Conv ops.
/// The rewriter inserts Q+DQ pairs around every Conv/MatMul/Gemm input
/// to simulate quantization at activation boundaries; on chained
/// Conv→Conv (depthwise→pointwise) sequences this also breaks DW+PW /
/// PW+DW runtime fusion because the inner ops are no longer adjacent.
///
/// Removing the inner Q+DQ pair reverts that boundary to fp32 — losing
/// the calibration simulation **at that one edge** but restoring the
/// fusion eligibility. Outer Q+DQ pairs (graph inputs / first Conv,
/// last Conv / graph outputs) stay intact, so end-to-end accuracy is
/// preserved within int8 tolerance for typical CV models.
///
/// Returns the number of Q+DQ pairs removed.
pub fn strip_qdq_within_fusion_chains(model: &mut OnnxModel) -> usize {
    let removed = run(model);
    if removed != 0 {
        model.rebuild_runtime_index();
    }
    removed
}

/// Strips without rebuilding the runtime index; returns the number of pairs
/// removed. This pass is not in the driver pipeline — the quantization tools
/// call the public wrapper above — but it follows the same contract.
pub(super) fn run(model: &mut OnnxModel) -> usize {
    let conv_like = |op: &str| matches!(op, "Conv" | "Conv_Relu" | "MatMul" | "Gemm");

    let mut to_remove: Vec<usize> = Vec::new();
    let mut renames: FxHashMap<String, String> = FxHashMap::default();

    for (i, dq) in model.nodes.iter().enumerate() {
        if dq.op_type != "DequantizeLinear" || dq.inputs.is_empty() || dq.outputs.is_empty() {
            continue;
        }
        let q_out = &dq.inputs[0];
        let q_idx = match model.nodes.iter().position(|n| {
            n.op_type == "QuantizeLinear" && !n.outputs.is_empty() && n.outputs[0] == *q_out
        }) {
            Some(j) => j,
            None => continue,
        };
        if to_remove.contains(&q_idx) {
            continue;
        }
        let q = &model.nodes[q_idx];
        if q.inputs.is_empty() {
            continue;
        }
        let q_consumers = model
            .nodes
            .iter()
            .filter(|n| n.inputs.iter().any(|s| s == q_out))
            .count();
        if q_consumers != 1 {
            continue;
        }
        let dq_out = &dq.outputs[0];
        let consumers: Vec<&OnnxNode> = model
            .nodes
            .iter()
            .filter(|n| n.inputs.iter().any(|s| s == dq_out))
            .collect();
        if consumers.len() != 1 {
            continue;
        }
        if !conv_like(&consumers[0].op_type) {
            continue;
        }
        let q_source = &q.inputs[0];
        let upstream = model
            .nodes
            .iter()
            .find(|n| n.outputs.iter().any(|o| o == q_source));
        let upstream = match upstream {
            Some(n) => n,
            None => continue,
        };
        if !conv_like(&upstream.op_type) {
            continue;
        }
        to_remove.push(q_idx);
        to_remove.push(i);
        renames.insert(dq_out.clone(), q_source.clone());
    }

    if to_remove.is_empty() {
        return 0;
    }

    for node in &mut model.nodes {
        for input in &mut node.inputs {
            if let Some(replacement) = renames.get(input) {
                *input = replacement.clone();
            }
        }
    }

    to_remove.sort_unstable();
    to_remove.dedup();
    let removed = to_remove.len() / 2;
    for &idx in to_remove.iter().rev() {
        model.nodes.remove(idx);
    }
    removed
}

#[cfg(test)]
mod strip_qdq_tests {

    use super::*;

    fn node(op: &str, name: &str, ins: &[&str], outs: &[&str]) -> OnnxNode {
        OnnxNode {
            op_type: op.to_string(),
            name: name.to_string(),
            inputs: ins.iter().map(|s| s.to_string()).collect(),
            outputs: outs.iter().map(|s| s.to_string()).collect(),
            attributes: FxHashMap::default(),
        }
    }

    fn empty_model(nodes: Vec<OnnxNode>) -> OnnxModel {
        OnnxModel {
            ir_version: 0,
            opset_version: 0,
            producer_name: String::new(),
            graph_name: String::new(),
            inputs: vec!["x".into()],
            outputs: vec!["y".into()],
            initializers: FxHashMap::default(),
            nodes,
            packed_int4_weights: Default::default(),
            runtime_index: Default::default(),
        }
    }

    #[test]
    fn strips_qdq_pair_between_two_convs() {
        let nodes = vec![
            node("Conv", "c1", &["x", "w1"], &["c1_out"]),
            node("QuantizeLinear", "q", &["c1_out", "s", "z"], &["q_out"]),
            node("DequantizeLinear", "dq", &["q_out", "s", "z"], &["dq_out"]),
            node("Conv", "c2", &["dq_out", "w2"], &["y"]),
        ];
        let mut model = empty_model(nodes);
        let removed = strip_qdq_within_fusion_chains(&mut model);
        assert_eq!(removed, 1);
        assert_eq!(model.nodes.len(), 2);
        assert_eq!(model.nodes[0].op_type, "Conv");
        assert_eq!(model.nodes[1].op_type, "Conv");
        assert_eq!(model.nodes[1].inputs[0], "c1_out");
    }

    #[test]
    fn keeps_qdq_when_source_is_graph_input() {
        let nodes = vec![
            node("QuantizeLinear", "q", &["x", "s", "z"], &["q_out"]),
            node("DequantizeLinear", "dq", &["q_out", "s", "z"], &["dq_out"]),
            node("Conv", "c1", &["dq_out", "w"], &["y"]),
        ];
        let mut model = empty_model(nodes);
        let removed = strip_qdq_within_fusion_chains(&mut model);
        assert_eq!(removed, 0);
        assert_eq!(model.nodes.len(), 3);
    }

    #[test]
    fn keeps_qdq_when_dq_has_multiple_consumers() {
        let nodes = vec![
            node("Conv", "c1", &["x", "w1"], &["c1_out"]),
            node("QuantizeLinear", "q", &["c1_out", "s", "z"], &["q_out"]),
            node("DequantizeLinear", "dq", &["q_out", "s", "z"], &["dq_out"]),
            node("Conv", "c2", &["dq_out", "w2"], &["y"]),
            node("Conv", "c3", &["dq_out", "w3"], &["y2"]),
        ];
        let mut model = empty_model(nodes);
        let removed = strip_qdq_within_fusion_chains(&mut model);
        assert_eq!(removed, 0);
    }
}
