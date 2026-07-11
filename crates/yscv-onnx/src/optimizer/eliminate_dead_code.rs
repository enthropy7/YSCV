use rustc_hash::FxHashSet;

use crate::loader::OnnxModel;

/// Removes nodes whose outputs are never consumed by any other node or graph output.
pub fn eliminate_dead_code(model: &mut OnnxModel) {
    loop {
        let consumed: FxHashSet<String> = {
            let mut set: FxHashSet<String> = model.outputs.iter().cloned().collect();
            for node in &model.nodes {
                for inp in &node.inputs {
                    if !inp.is_empty() {
                        set.insert(inp.clone());
                    }
                }
            }
            set
        };

        let before = model.nodes.len();
        model
            .nodes
            .retain(|node| node.outputs.iter().any(|o| consumed.contains(o)));

        if model.nodes.len() == before {
            break;
        }
    }
}
