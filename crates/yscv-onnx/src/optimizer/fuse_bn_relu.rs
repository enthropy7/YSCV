use crate::loader::OnnxModel;

/// Fuses consecutive BatchNormalization + Relu into a single marked node
/// (annotation-only; execution still handles them separately, but this
/// reduces graph traversal overhead for large models).
pub fn fuse_bn_relu(model: &mut OnnxModel) {
    if run(model) {
        model.rebuild_runtime_index();
    }
}

/// Fuses without rebuilding the runtime index; returns whether the graph
/// changed. The driver calls this and rebuilds once for the whole pipeline.
pub(super) fn run(model: &mut OnnxModel) -> bool {
    let mut fuse_pairs: Vec<(usize, usize)> = Vec::new();
    for i in 0..model.nodes.len().saturating_sub(1) {
        if model.nodes[i].op_type == "BatchNormalization"
            && model.nodes[i + 1].op_type == "Relu"
            && model.nodes[i + 1].inputs.len() == 1
            && !model.nodes[i].outputs.is_empty()
            && model.nodes[i + 1].inputs[0] == model.nodes[i].outputs[0]
        {
            let bn_out = &model.nodes[i].outputs[0];
            let consumers: usize = model
                .nodes
                .iter()
                .enumerate()
                .filter(|&(j, n)| j != i + 1 && n.inputs.contains(bn_out))
                .count();
            if consumers == 0 && !model.outputs.contains(bn_out) {
                fuse_pairs.push((i, i + 1));
            }
        }
    }

    for &(bn_idx, relu_idx) in fuse_pairs.iter().rev() {
        let relu_output = model.nodes[relu_idx].outputs[0].clone();
        model.nodes[bn_idx].outputs[0] = relu_output;
        model.nodes[bn_idx].op_type = "BatchNormalization_Relu".to_string();
        model.nodes.remove(relu_idx);
    }
    !fuse_pairs.is_empty()
}
