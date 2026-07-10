use rustc_hash::FxHashMap;

use crate::loader::OnnxModel;

/// Returns statistics about a model graph (for diagnostics).
#[derive(Debug, Clone)]
pub struct GraphStats {
    pub node_count: usize,
    pub initializer_count: usize,
    pub op_types: Vec<(String, usize)>,
}

/// Computes summary statistics for an ONNX model graph.
pub fn graph_stats(model: &OnnxModel) -> GraphStats {
    let mut op_counts = FxHashMap::default();
    for node in &model.nodes {
        *op_counts.entry(node.op_type.clone()).or_insert(0usize) += 1;
    }
    let mut op_types: Vec<(String, usize)> = op_counts.into_iter().collect();
    op_types.sort_by_key(|&(_, count)| std::cmp::Reverse(count));

    GraphStats {
        node_count: model.nodes.len(),
        initializer_count: model.initializers.len(),
        op_types,
    }
}
