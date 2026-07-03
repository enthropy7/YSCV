use std::collections::HashMap;

use crate::loader::OnnxModel;

/// Reorder graph nodes into a fusion-friendly topological order.
///
/// ONNX topological order is not unique. Multi-input models — e.g. a Siamese
/// tracker whose two backbone branches share weights — are often exported with
/// the branches interleaved: `convA, convB, reluA, reluB, …`. Every positional
/// fusion pass (`fuse_conv_relu` here, and the runtime planner's `FusedDwPw` /
/// `FusedPwDw` / `ConvAdd` matchers) inspects only the immediately-following
/// node, so an interleaved producer/consumer pair never fuses and the whole
/// inverted bottleneck runs unfused.
///
/// A depth-first topological sort (LIFO ready stack) walks each dependency
/// chain to its join before starting the next, placing every producer directly
/// ahead of its consumer. That restores positional adjacency for the fusion
/// passes regardless of export order — the same invariant ORT gets from its
/// dataflow graph optimizer. The reorder only permutes independent nodes, so
/// the computation and its outputs are unchanged.
pub fn reorder_nodes_for_fusion(model: &mut OnnxModel) {
    if std::env::var_os("YSCV_REORDER_FUSION_OFF").is_some() {
        return;
    }
    let n = model.nodes.len();
    if n < 2 {
        return;
    }

    let mut producer: HashMap<&str, usize> = HashMap::with_capacity(n);
    for (i, node) in model.nodes.iter().enumerate() {
        for out in &node.outputs {
            producer.insert(out.as_str(), i);
        }
    }

    let mut in_degree = vec![0usize; n];
    let mut consumers: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (i, node) in model.nodes.iter().enumerate() {
        let mut deps: Vec<usize> = Vec::new();
        for inp in &node.inputs {
            if let Some(&p) = producer.get(inp.as_str())
                && p != i
                && !deps.contains(&p)
            {
                deps.push(p);
            }
        }
        in_degree[i] = deps.len();
        for p in deps {
            consumers[p].push(i);
        }
    }

    let mut stack: Vec<usize> = Vec::new();
    for i in (0..n).rev() {
        if in_degree[i] == 0 {
            stack.push(i);
        }
    }

    let mut order: Vec<usize> = Vec::with_capacity(n);
    while let Some(node) = stack.pop() {
        order.push(node);
        let mut newly: Vec<usize> = Vec::new();
        for &c in &consumers[node] {
            in_degree[c] -= 1;
            if in_degree[c] == 0 {
                newly.push(c);
            }
        }
        for &c in newly.iter().rev() {
            stack.push(c);
        }
    }

    if order.len() != n {
        return;
    }

    let mut taken: Vec<_> = model.nodes.drain(..).map(Some).collect();
    model.nodes = order
        .iter()
        .map(|&idx| {
            taken[idx]
                .take()
                .expect("permutation visits each node once")
        })
        .collect();
}
