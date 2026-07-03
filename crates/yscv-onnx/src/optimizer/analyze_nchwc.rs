use std::collections::HashMap;

use crate::loader::OnnxModel;

/// Summary statistics describing NCHWc-transformability of a loaded graph.
///
/// Produced by [`analyze_nchwc`] — inspects the graph for op patterns that
/// could stay in NCHWc layout (Conv, Pool, BN, elementwise Add/Relu/Sigmoid,
/// SiLU-as-Sigmoid-Mul) and counts maximal contiguous NCHWc-capable chains.
/// Used by the runner and by B.8 rollout decisions to judge whether
/// `YSCV_NCHWC=on` is worth flipping for a given model.
///
/// The metric that matters is `max_chain_len`: conversion cost at the
/// chain boundaries is paid once per chain, so longer chains amortize
/// better. Per the plan's cost model, chains of ≥3 NCHWc-capable ops
/// are worth converting; anything smaller burns more on layout reorders
/// than it saves on kernel speedup.
#[derive(Debug, Clone)]
pub struct NchwcStats {
    pub capable_nodes: usize,
    pub total_nodes: usize,
    pub chain_count: usize,
    pub max_chain_len: usize,
    pub mean_chain_len: f32,
    pub op_types_capable: Vec<(String, usize)>,
}

fn is_nchwc_capable_op(op_type: &str) -> bool {
    matches!(
        op_type,
        "Conv"
            | "MaxPool"
            | "AveragePool"
            | "GlobalAveragePool"
            | "BatchNormalization"
            | "Relu"
            | "Sigmoid"
            | "Add"
            | "Mul"
            | "Clip"
    )
}

pub fn analyze_nchwc(model: &OnnxModel) -> NchwcStats {
    let total_nodes = model.nodes.len();
    let mut op_counts: HashMap<String, usize> = HashMap::new();
    let mut capable_nodes = 0usize;
    let mut chain_lengths: Vec<usize> = Vec::new();
    let mut current_chain = 0usize;

    for node in &model.nodes {
        if is_nchwc_capable_op(&node.op_type) {
            capable_nodes += 1;
            current_chain += 1;
            *op_counts.entry(node.op_type.clone()).or_insert(0) += 1;
        } else if current_chain > 0 {
            chain_lengths.push(current_chain);
            current_chain = 0;
        }
    }
    if current_chain > 0 {
        chain_lengths.push(current_chain);
    }

    let chain_count = chain_lengths.len();
    let max_chain_len = chain_lengths.iter().copied().max().unwrap_or(0);
    let mean_chain_len = if chain_count == 0 {
        0.0
    } else {
        chain_lengths.iter().copied().sum::<usize>() as f32 / chain_count as f32
    };
    let mut op_types_capable: Vec<(String, usize)> = op_counts.into_iter().collect();
    op_types_capable.sort_by_key(|&(_, count)| std::cmp::Reverse(count));

    NchwcStats {
        capable_nodes,
        total_nodes,
        chain_count,
        max_chain_len,
        mean_chain_len,
        op_types_capable,
    }
}

#[cfg(test)]
mod nchwc_stats_tests {
    use std::collections::HashSet;

    use super::*;
    use crate::loader::OnnxNode;

    fn node(op: &str, name: &str) -> OnnxNode {
        OnnxNode {
            op_type: op.to_string(),
            name: name.to_string(),
            inputs: vec![],
            outputs: vec![],
            attributes: HashMap::new(),
        }
    }

    fn model_with(ops: &[&str]) -> OnnxModel {
        let mut m = OnnxModel {
            ir_version: 0,
            opset_version: 0,
            producer_name: String::new(),
            graph_name: String::new(),
            inputs: vec![],
            outputs: vec![],
            initializers: HashMap::new(),
            nodes: vec![],
            khwc_weights: HashSet::new(),
            dw_khwc_weights: HashSet::new(),
            group_khwc_weights: HashSet::new(),
            packed_int4_weights: Default::default(),
            runtime_index: Default::default(),
        };
        for (i, op) in ops.iter().enumerate() {
            m.nodes.push(node(op, &format!("n{i}")));
        }
        m
    }

    #[test]
    fn analyze_detects_no_chain() {
        let m = model_with(&["Reshape", "Gather"]);
        let s = analyze_nchwc(&m);
        assert_eq!(s.capable_nodes, 0);
        assert_eq!(s.chain_count, 0);
        assert_eq!(s.max_chain_len, 0);
    }

    #[test]
    fn analyze_counts_single_chain() {
        let m = model_with(&["Conv", "BatchNormalization", "Relu", "Conv", "Add"]);
        let s = analyze_nchwc(&m);
        assert_eq!(s.capable_nodes, 5);
        assert_eq!(s.chain_count, 1);
        assert_eq!(s.max_chain_len, 5);
    }

    #[test]
    fn analyze_splits_on_incapable_op() {
        let m = model_with(&["Conv", "Reshape", "Conv", "Relu", "Transpose", "Conv"]);
        let s = analyze_nchwc(&m);
        assert_eq!(s.capable_nodes, 4);
        assert_eq!(s.chain_count, 3);
        assert_eq!(s.max_chain_len, 2);
    }

    #[test]
    fn analyze_computes_mean_chain() {
        let m = model_with(&["Conv", "Conv", "Reshape", "Conv", "Conv", "Conv"]);
        let s = analyze_nchwc(&m);
        assert_eq!(s.chain_count, 2);
        assert_eq!(s.max_chain_len, 3);
        assert!((s.mean_chain_len - 2.5).abs() < 1e-6);
    }
}
