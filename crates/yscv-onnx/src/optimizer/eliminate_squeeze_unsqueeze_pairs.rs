use crate::loader::OnnxModel;
use crate::loader::OnnxNode;

/// Eliminates inverse Squeeze/Unsqueeze pairs left by ONNX export pipelines
/// (common after PyTorch → ONNX for broadcasting workarounds). Matches both
/// Squeeze(Unsqueeze(x, axes=A), axes=A) and Unsqueeze(Squeeze(x, axes=A), axes=A).
pub fn eliminate_squeeze_unsqueeze_pairs(model: &mut OnnxModel) {
    let mut remove_pairs: Vec<(usize, usize, String, String)> = Vec::new();
    // Index of the first node not already claimed by an accepted pair. A chain
    // like Squeeze -> Unsqueeze -> Squeeze matches at both `i` and `i + 1`;
    // accepting both would make the reverse-removal loop below delete shifted
    // indices, taking unrelated nodes with them.
    let mut next_free = 0usize;

    for i in 0..model.nodes.len().saturating_sub(1) {
        if i < next_free {
            continue;
        }
        let first = &model.nodes[i];
        let second = &model.nodes[i + 1];
        let is_pair = (first.op_type == "Squeeze" && second.op_type == "Unsqueeze")
            || (first.op_type == "Unsqueeze" && second.op_type == "Squeeze");
        if !is_pair || first.outputs.is_empty() || second.inputs.is_empty() {
            continue;
        }
        if second.inputs[0] != first.outputs[0] {
            continue;
        }
        let first_out = &first.outputs[0];
        let consumers: usize = model
            .nodes
            .iter()
            .enumerate()
            .filter(|&(j, n)| j != i + 1 && n.inputs.contains(first_out))
            .count();
        if consumers != 0 || model.outputs.contains(first_out) {
            continue;
        }
        let first_axes = node_axes(model, first);
        let second_axes = node_axes(model, second);
        if first_axes.is_none() || first_axes != second_axes {
            continue;
        }

        if first.inputs.is_empty() || second.outputs.is_empty() {
            continue;
        }
        remove_pairs.push((i, i + 1, first.inputs[0].clone(), second.outputs[0].clone()));
        next_free = i + 2;
    }

    for &(first_idx, second_idx, ref producer_input, ref consumer_output) in
        remove_pairs.iter().rev()
    {
        for node in &mut model.nodes {
            for inp in &mut node.inputs {
                if inp == consumer_output {
                    *inp = producer_input.clone();
                }
            }
        }
        for out in &mut model.outputs {
            if out == consumer_output {
                *out = producer_input.clone();
            }
        }
        model.nodes.remove(second_idx);
        model.nodes.remove(first_idx);
    }
}

fn node_axes(model: &OnnxModel, node: &OnnxNode) -> Option<Vec<i64>> {
    if node.inputs.len() > 1 && !node.inputs[1].is_empty() {
        let t = model.initializers.get(&node.inputs[1])?;
        let mut v: Vec<i64> = t.data().iter().map(|&x| x as i64).collect();
        v.sort_unstable();
        return Some(v);
    }
    if let Some(crate::loader::OnnxAttribute::Ints(axes)) = node.attributes.get("axes") {
        let mut v = axes.clone();
        v.sort_unstable();
        return Some(v);
    }
    None
}
