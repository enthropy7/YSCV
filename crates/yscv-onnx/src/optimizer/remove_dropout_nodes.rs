use rustc_hash::FxHashSet;

use crate::loader::OnnxModel;

/// Removes Dropout nodes by rewiring their consumers to the Dropout's input.
pub fn remove_dropout_nodes(model: &mut OnnxModel) {
    let mut rewire: Vec<(String, String)> = Vec::new();

    for node in &model.nodes {
        if node.op_type == "Dropout" && !node.inputs.is_empty() && !node.outputs.is_empty() {
            rewire.push((node.outputs[0].clone(), node.inputs[0].clone()));
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

    let dropout_names: FxHashSet<String> = model
        .nodes
        .iter()
        .filter(|n| n.op_type == "Dropout")
        .map(|n| n.name.clone())
        .collect();
    model.nodes.retain(|n| !dropout_names.contains(&n.name));
}
