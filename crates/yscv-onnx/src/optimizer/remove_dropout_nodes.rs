use crate::loader::OnnxModel;

/// Removes Dropout nodes by rewiring their consumers to the Dropout's input.
pub fn remove_dropout_nodes(model: &mut OnnxModel) {
    let mut rewire: Vec<(String, String)> = Vec::new();
    // Indices of the Dropouts we actually rewired, ascending. Deleting by node
    // name instead would be wrong twice over: `NodeProto.name` is optional in
    // ONNX, so one unnamed Dropout would take every other unnamed node in the
    // graph with it, and a malformed Dropout that was skipped above would be
    // deleted without its consumers ever being rewired.
    let mut drop_indices: Vec<usize> = Vec::new();

    for (idx, node) in model.nodes.iter().enumerate() {
        if node.op_type == "Dropout" && !node.inputs.is_empty() && !node.outputs.is_empty() {
            rewire.push((node.outputs[0].clone(), node.inputs[0].clone()));
            drop_indices.push(idx);
        }
    }

    if rewire.is_empty() {
        return;
    }

    for (old_name, new_name) in &rewire {
        for node in &mut model.nodes {
            for inp in &mut node.inputs {
                if inp == old_name {
                    *inp = new_name.clone();
                }
            }
        }
        for out in &mut model.outputs {
            if out == old_name {
                *out = new_name.clone();
            }
        }
    }

    for &idx in drop_indices.iter().rev() {
        model.nodes.remove(idx);
    }
}
