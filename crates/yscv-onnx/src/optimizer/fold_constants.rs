use rustc_hash::FxHashMap;
use rustc_hash::FxHashSet;

use yscv_tensor::Tensor;

use crate::loader::OnnxModel;
use crate::runner::run_onnx_model;

/// Folds constant sub-graphs by executing nodes whose inputs are all initializers.
///
/// Iterates until a fixed point is reached (no more foldable nodes).
pub fn fold_constants(model: &mut OnnxModel) {
    loop {
        let foldable = model.nodes.iter().enumerate().find(|(_, node)| {
            !node.inputs.is_empty()
                && node
                    .inputs
                    .iter()
                    .all(|inp| inp.is_empty() || model.initializers.contains_key(inp))
                && node.outputs.len() == 1
                && !matches!(
                    node.op_type.as_str(),
                    "Conv" | "ConvTranspose" | "DeformConv"
                )
        });

        let (idx, _) = match foldable {
            Some(pair) => pair,
            None => break,
        };

        let node = model.nodes[idx].clone();
        if node.op_type == "Identity"
            && node.inputs.len() == 1
            && node.outputs.len() == 1
            && !node.inputs[0].is_empty()
        {
            let src = &node.inputs[0];
            let dst = &node.outputs[0];
            if let Some(t) = model.initializers.get(src).cloned() {
                model.initializers.insert(dst.clone(), t);
                if model.khwc_weights.contains(src) {
                    model.khwc_weights.insert(dst.clone());
                }
                if model.dw_khwc_weights.contains(src) {
                    model.dw_khwc_weights.insert(dst.clone());
                }
                if model.group_khwc_weights.contains(src) {
                    model.group_khwc_weights.insert(dst.clone());
                }
            }
            model.nodes.remove(idx);
            continue;
        }

        let mut mini_model = OnnxModel {
            ir_version: model.ir_version,
            opset_version: model.opset_version,
            producer_name: String::new(),
            graph_name: String::new(),
            inputs: node.inputs.clone(),
            outputs: node.outputs.clone(),
            initializers: FxHashMap::default(),
            nodes: vec![node.clone()],
            khwc_weights: FxHashSet::default(),
            dw_khwc_weights: FxHashSet::default(),
            group_khwc_weights: FxHashSet::default(),
            packed_int4_weights: Default::default(),
            runtime_index: Default::default(),
        };
        mini_model.rebuild_runtime_index();

        let mut inputs: FxHashMap<String, Tensor> = FxHashMap::default();
        for inp in &node.inputs {
            if !inp.is_empty()
                && let Some(t) = model.initializers.get(inp)
            {
                inputs.insert(inp.clone(), t.clone());
            }
        }

        match run_onnx_model(&mini_model, inputs) {
            Ok(results) => {
                for (name, tensor) in results {
                    model.initializers.insert(name, tensor);
                }
                model.nodes.remove(idx);
            }
            Err(_) => {
                break;
            }
        }
    }
    model.rebuild_runtime_index();
}
