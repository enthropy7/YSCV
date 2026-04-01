use std::collections::{HashMap, HashSet};

use yscv_tensor::Tensor;

use crate::loader::OnnxModel;
use crate::runner::run_onnx_model;

/// Optimizes an ONNX model graph in-place for inference.
///
/// Currently applies:
/// - Dead-code elimination (remove nodes whose outputs are never consumed)
/// - Dropout removal (replace with identity in inference mode)
/// - Constant folding for trivial reshape/transpose of single-element tensors
pub fn optimize_onnx_graph(model: &mut OnnxModel) {
    remove_dropout_nodes(model);
    fold_conv_bn(model);
    fold_constants(model);
    eliminate_dead_code(model);
}

/// Removes Dropout nodes by rewiring their consumers to the Dropout's input.
fn remove_dropout_nodes(model: &mut OnnxModel) {
    let mut rewire: Vec<(String, String)> = Vec::new();

    for node in &model.nodes {
        if node.op_type == "Dropout" && !node.inputs.is_empty() && !node.outputs.is_empty() {
            rewire.push((node.outputs[0].clone(), node.inputs[0].clone()));
        }
    }

    if rewire.is_empty() {
        return;
    }

    // Apply rewiring to all downstream consumers
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

    // Remove the dropout nodes themselves
    let dropout_names: HashSet<String> = model
        .nodes
        .iter()
        .filter(|n| n.op_type == "Dropout")
        .map(|n| n.name.clone())
        .collect();
    model.nodes.retain(|n| !dropout_names.contains(&n.name));
}

/// Removes nodes whose outputs are never consumed by any other node or graph output.
fn eliminate_dead_code(model: &mut OnnxModel) {
    loop {
        let consumed: HashSet<String> = {
            let mut set: HashSet<String> = model.outputs.iter().cloned().collect();
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

/// Returns statistics about a model graph (for diagnostics).
#[derive(Debug, Clone)]
pub struct GraphStats {
    pub node_count: usize,
    pub initializer_count: usize,
    pub op_types: Vec<(String, usize)>,
}

/// Computes summary statistics for an ONNX model graph.
pub fn graph_stats(model: &OnnxModel) -> GraphStats {
    let mut op_counts = std::collections::HashMap::new();
    for node in &model.nodes {
        *op_counts.entry(node.op_type.clone()).or_insert(0usize) += 1;
    }
    let mut op_types: Vec<(String, usize)> = op_counts.into_iter().collect();
    op_types.sort_by(|a, b| b.1.cmp(&a.1));

    GraphStats {
        node_count: model.nodes.len(),
        initializer_count: model.initializers.len(),
        op_types,
    }
}

/// Fuses consecutive BatchNormalization + Relu into a single marked node
/// (annotation-only; execution still handles them separately, but this
/// reduces graph traversal overhead for large models).
pub fn fuse_bn_relu(model: &mut OnnxModel) {
    let mut fuse_pairs: Vec<(usize, usize)> = Vec::new();
    for i in 0..model.nodes.len().saturating_sub(1) {
        if model.nodes[i].op_type == "BatchNormalization"
            && model.nodes[i + 1].op_type == "Relu"
            && model.nodes[i + 1].inputs.len() == 1
            && !model.nodes[i].outputs.is_empty()
            && model.nodes[i + 1].inputs[0] == model.nodes[i].outputs[0]
        {
            // Check that the BN output is only consumed by this Relu
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

    // Apply fusions in reverse order to keep indices valid
    for &(bn_idx, relu_idx) in fuse_pairs.iter().rev() {
        let relu_output = model.nodes[relu_idx].outputs[0].clone();
        model.nodes[bn_idx].outputs[0] = relu_output;
        model.nodes[bn_idx].op_type = "BatchNormalization_Relu".to_string();
        model.nodes.remove(relu_idx);
    }
}

/// Fuses consecutive Conv + Relu into a single annotated node.
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
}

/// Folds Conv + BatchNormalization pairs by absorbing BN parameters into Conv weights.
///
/// For each Conv immediately followed by a BatchNormalization whose sole input is the
/// Conv output, we compute fused weights:
///   scale_c = gamma_c / sqrt(var_c + eps)
///   `W_fused[c] = W[c] * scale_c`
///   `b_fused[c] = (b[c] - mean_c) * scale_c + beta_c`
/// The Conv initializers are replaced and the BN node is removed.
pub fn fold_conv_bn(model: &mut OnnxModel) {
    let mut fuse_pairs: Vec<(usize, usize)> = Vec::new();

    for i in 0..model.nodes.len().saturating_sub(1) {
        if model.nodes[i].op_type == "Conv"
            && model.nodes[i + 1].op_type == "BatchNormalization"
            && !model.nodes[i].outputs.is_empty()
            && !model.nodes[i + 1].inputs.is_empty()
            && model.nodes[i + 1].inputs[0] == model.nodes[i].outputs[0]
        {
            // Ensure Conv output is only consumed by this BN
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

    for &(conv_idx, bn_idx) in fuse_pairs.iter().rev() {
        let conv_node = &model.nodes[conv_idx];
        let bn_node = &model.nodes[bn_idx];

        // Conv inputs: X, W, [B]
        // BN inputs: X, gamma (scale), beta (B), mean, var
        if conv_node.inputs.len() < 2 || bn_node.inputs.len() < 5 {
            continue;
        }

        let w_name = &conv_node.inputs[1];
        let gamma_name = &bn_node.inputs[1];
        let beta_name = &bn_node.inputs[2];
        let mean_name = &bn_node.inputs[3];
        let var_name = &bn_node.inputs[4];

        let epsilon = bn_node
            .attributes
            .get("epsilon")
            .and_then(|a| {
                if let crate::loader::OnnxAttribute::Float(v) = a {
                    Some(*v)
                } else {
                    None
                }
            })
            .unwrap_or(1e-5);

        // Clone tensors we need
        let w = match model.initializers.get(w_name) {
            Some(t) => t.clone(),
            None => continue,
        };
        let gamma = match model.initializers.get(gamma_name) {
            Some(t) => t.clone(),
            None => continue,
        };
        let beta = match model.initializers.get(beta_name) {
            Some(t) => t.clone(),
            None => continue,
        };
        let mean = match model.initializers.get(mean_name) {
            Some(t) => t.clone(),
            None => continue,
        };
        let var = match model.initializers.get(var_name) {
            Some(t) => t.clone(),
            None => continue,
        };

        let out_channels = w.shape()[0];
        let gamma_d = gamma.data();
        let beta_d = beta.data();
        let mean_d = mean.data();
        let var_d = var.data();

        if gamma_d.len() < out_channels
            || beta_d.len() < out_channels
            || mean_d.len() < out_channels
            || var_d.len() < out_channels
        {
            continue;
        }

        // Compute scale per output channel
        let scale: Vec<f32> = (0..out_channels)
            .map(|c| gamma_d[c] / (var_d[c] + epsilon).sqrt())
            .collect();

        // Fuse weights: W_fused[c, ...] = W[c, ...] * scale[c]
        let w_shape = w.shape().to_vec();
        let mut w_data = w.data().to_vec();
        let elems_per_channel = w_data.len() / out_channels;
        for c in 0..out_channels {
            let start = c * elems_per_channel;
            let end = start + elems_per_channel;
            for v in &mut w_data[start..end] {
                *v *= scale[c];
            }
        }
        let w_fused = Tensor::from_vec(w_shape, w_data).expect("fused weight shape matches data");

        // Fuse bias: b_fused[c] = (b[c] - mean[c]) * scale[c] + beta[c]
        let conv_has_bias = conv_node.inputs.len() >= 3 && !conv_node.inputs[2].is_empty();
        let old_bias: Vec<f32> = if conv_has_bias {
            model
                .initializers
                .get(&conv_node.inputs[2])
                .map(|t| t.data().to_vec())
                .unwrap_or_else(|| vec![0.0; out_channels])
        } else {
            vec![0.0; out_channels]
        };
        let b_fused_data: Vec<f32> = (0..out_channels)
            .map(|c| (old_bias[c] - mean_d[c]) * scale[c] + beta_d[c])
            .collect();
        let b_fused = Tensor::from_vec(vec![out_channels], b_fused_data)
            .expect("fused bias shape matches data");

        // Update initializers
        model.initializers.insert(w_name.clone(), w_fused);

        let bias_name = if conv_has_bias {
            conv_node.inputs[2].clone()
        } else {
            let name = format!("{}_fused_bias", conv_node.name);
            // Add bias input to Conv node
            model.nodes[conv_idx].inputs.push(name.clone());
            name
        };
        model.initializers.insert(bias_name, b_fused);

        // Rewire: Conv output takes BN output name
        let bn_output = model.nodes[bn_idx].outputs[0].clone();
        model.nodes[conv_idx].outputs[0] = bn_output;

        // Remove BN node
        model.nodes.remove(bn_idx);
    }
}

/// Folds constant sub-graphs by executing nodes whose inputs are all initializers.
///
/// Iterates until a fixed point is reached (no more foldable nodes).
pub fn fold_constants(model: &mut OnnxModel) {
    loop {
        // Find a node where ALL inputs are initializers (constants)
        let foldable = model.nodes.iter().enumerate().find(|(_, node)| {
            !node.inputs.is_empty()
                && node.inputs.iter().all(|inp| {
                    inp.is_empty() || model.initializers.contains_key(inp)
                })
                // Skip nodes that produce more than one output for simplicity
                && node.outputs.len() == 1
        });

        let (idx, _) = match foldable {
            Some(pair) => pair,
            None => break,
        };

        // Build a minimal model with just this node to execute it
        let node = model.nodes[idx].clone();
        let mini_model = OnnxModel {
            ir_version: model.ir_version,
            opset_version: model.opset_version,
            producer_name: String::new(),
            graph_name: String::new(),
            inputs: node.inputs.clone(),
            outputs: node.outputs.clone(),
            initializers: HashMap::new(),
            nodes: vec![node.clone()],
            khwc_weights: HashSet::new(),
            dw_khwc_weights: HashSet::new(),
        };

        // Gather inputs for the mini model
        let mut inputs: HashMap<String, Tensor> = HashMap::new();
        for inp in &node.inputs {
            if !inp.is_empty()
                && let Some(t) = model.initializers.get(inp)
            {
                inputs.insert(inp.clone(), t.clone());
            }
        }

        match run_onnx_model(&mini_model, inputs) {
            Ok(results) => {
                // Store outputs as initializers
                for (name, tensor) in results {
                    model.initializers.insert(name, tensor);
                }
                // Remove the folded node
                model.nodes.remove(idx);
            }
            Err(_) => {
                // Cannot fold this node; skip it by breaking to avoid infinite loop
                break;
            }
        }
    }
}
