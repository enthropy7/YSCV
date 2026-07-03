use crate::loader::OnnxModel;

/// Fuse `Conv` -> `Relu` node pairs into a single fused-activation Conv.
pub fn fuse_conv_relu(model: &mut OnnxModel) {
    let mut fuse_pairs: Vec<(usize, usize)> = Vec::new();
    for i in 0..model.nodes.len().saturating_sub(1) {
        if model.nodes[i].op_type == "Conv"
            && model.nodes[i + 1].op_type == "Relu"
            && model.nodes[i + 1].inputs.len() == 1
            && !model.nodes[i].outputs.is_empty()
            && model.nodes[i + 1].inputs[0] == model.nodes[i].outputs[0]
        {
            let conv_out = &model.nodes[i].outputs[0];
            let consumers: usize = model
                .nodes
                .iter()
                .enumerate()
                .filter(|&(j, n)| j != i + 1 && n.inputs.contains(conv_out))
                .count();
            if consumers == 0 && !model.outputs.contains(conv_out) {
                fuse_pairs.push((i, i + 1));
            }
        }
    }

    for &(conv_idx, relu_idx) in fuse_pairs.iter().rev() {
        let relu_output = model.nodes[relu_idx].outputs[0].clone();
        model.nodes[conv_idx].outputs[0] = relu_output;
        model.nodes[conv_idx].op_type = "Conv_Relu".to_string();
        model.nodes.remove(relu_idx);
    }
    model.rebuild_runtime_index();
}
