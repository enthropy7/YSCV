use std::collections::{HashMap, HashSet};

use yscv_tensor::Tensor;

use crate::loader::OnnxModel;
use crate::runner::run_onnx_model;

/// Optimizes an ONNX model graph in-place for inference.
///
/// Applies load-time passes modeled after ORT's Level-1 optimizer:
/// - Dropout removal (inference-only, rewire consumers to Dropout input)
/// - Conv-BatchNormalization folding (absorb BN γ/β/μ/σ into Conv weights)
/// - Conv-Mul(const) scale absorption (absorb scalar/per-channel Mul into weights)
/// - Conv-Add(const) bias absorption (absorb per-channel Add into Conv bias)
/// - Constant folding (execute nodes with all-initializer inputs at load)
/// - Squeeze/Unsqueeze pair elimination (drop inverse pairs left by PyTorch export)
/// - Conv+Clip(0,max) fusion (ReLU6-style clamped activation)
/// - Conv+Relu / BN+Relu fusion (annotation-only; kernel dispatches on op_type)
/// - Dead code elimination (iterate to fixpoint)
///
/// Order matters: `fold_conv_bn` runs before Conv-Mul/Conv-Add because BN
/// usually absorbs into Conv already; only stray scale/bias left over fall
/// through. Constant folding runs before Relu/Clip fusions because folded
/// Relus may turn Clip-style patterns into plain activations.
pub fn optimize_onnx_graph(model: &mut OnnxModel) {
    remove_dropout_nodes(model);
    fold_conv_bn(model);
    fold_conv_mul(model);
    fold_conv_add_const(model);
    fold_constants(model);
    eliminate_squeeze_unsqueeze_pairs(model);
    fuse_conv_relu(model);
    fuse_bn_relu(model);
    eliminate_dead_code(model);
    model.rebuild_runtime_index();

    // NCHWc transformer (Part B of the backus plan) — opt-in feasibility
    // diagnostic. When `YSCV_NCHWC=on`, logs per-graph statistics so a
    // developer can decide whether to build out the runtime NCHWc dispatch
    // for this model. No graph mutation yet — the kernels exist (see
    // `yscv_kernels::conv2d_nchwc_with_activation_prepacked` and peers)
    // but wiring requires runner-side layout propagation which has not
    // landed. Activate via env and check stderr; default path is silent.
    if std::env::var("YSCV_NCHWC").as_deref() == Ok("on") {
        let stats = analyze_nchwc(model);
        eprintln!(
            "[yscv-onnx] NCHWc stats: capable={}/{} chains={} max_chain={} mean_chain={:.2}",
            stats.capable_nodes,
            stats.total_nodes,
            stats.chain_count,
            stats.max_chain_len,
            stats.mean_chain_len,
        );
        if !stats.op_types_capable.is_empty() {
            let top: Vec<String> = stats
                .op_types_capable
                .iter()
                .take(5)
                .map(|(op, n)| format!("{op}:{n}"))
                .collect();
            eprintln!("[yscv-onnx] NCHWc top ops: {}", top.join(" "));
        }
    }
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
    // Sort descending by count — `sort_by_key` with a negated key keeps
    // the hot ops first in the report.
    op_types.sort_by_key(|&(_, count)| std::cmp::Reverse(count));

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
    model.rebuild_runtime_index();
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
    model.rebuild_runtime_index();
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

        let is_khwc = model.khwc_weights.contains(w_name);
        let is_dw_khwc = model.dw_khwc_weights.contains(w_name);
        let is_group_khwc = model.group_khwc_weights.contains(w_name);

        let out_channels = if is_khwc {
            // Pre-permuted OIHW→KHWC [KH, KW, I, O]: O is last dim
            w.shape()[3]
        } else if is_dw_khwc {
            // Pre-packed depthwise [KH, KW, C, dm]: C is dim 2
            w.shape()[2]
        } else if is_group_khwc {
            // Pre-packed grouped [O, KH, KW, I/G]: O is first dim
            w.shape()[0]
        } else {
            // Original OIHW [O, I, KH, KW]: O is first dim
            w.shape()[0]
        };

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
        if is_khwc {
            // KHWC [KH, KW, I, O]: O is innermost dim, stride=1
            for (i, v) in w_data.iter_mut().enumerate() {
                *v *= scale[i % out_channels];
            }
        } else if is_dw_khwc {
            // Depthwise [KH, KW, C, dm]: C is dim 2, dm is innermost
            let dm = w.shape()[3];
            for (i, v) in w_data.iter_mut().enumerate() {
                let c = (i / dm) % out_channels;
                *v *= scale[c];
            }
        } else {
            // OIHW or grouped [O, KH, KW, I/G]: O is outermost dim
            let elems_per_channel = w_data.len() / out_channels;
            for c in 0..out_channels {
                let start = c * elems_per_channel;
                let end = start + elems_per_channel;
                for v in &mut w_data[start..end] {
                    *v *= scale[c];
                }
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
    model.rebuild_runtime_index();
}

/// Returns the per-output-channel scaling vector for a Conv weight layout.
/// Copies the slice out so callers can multiply in-place without borrowing
/// the initializer map. `None` means the scale tensor was not usable
/// (not an initializer, wrong shape, wrong element count).
fn broadcast_scale_to_oc(
    model: &OnnxModel,
    scale_name: &str,
    out_channels: usize,
) -> Option<Vec<f32>> {
    let t = model.initializers.get(scale_name)?;
    let data = t.data();
    // Accept: scalar [1], per-channel [OC], or [1,OC,1,1] (ONNX Conv output NCHW layout).
    let shape = t.shape();
    let numel: usize = shape.iter().product();
    if numel == 1 {
        // Broadcast scalar.
        Some(vec![data[0]; out_channels])
    } else if numel == out_channels {
        Some(data.to_vec())
    } else {
        None
    }
}

/// Scales flat weight data in-place by a per-OC vector, respecting the
/// pre-permuted layout tag on the weight initializer.
fn scale_weight_inplace(
    w_data: &mut [f32],
    w_shape: &[usize],
    out_channels: usize,
    scale: &[f32],
    is_khwc: bool,
    is_dw_khwc: bool,
) {
    if is_khwc {
        // KHWC [KH, KW, IC, OC] — OC is innermost, stride 1.
        for (i, v) in w_data.iter_mut().enumerate() {
            *v *= scale[i % out_channels];
        }
    } else if is_dw_khwc {
        // Depthwise [KH, KW, C, dm] — C is dim 2, dm innermost.
        let dm = w_shape[3];
        for (i, v) in w_data.iter_mut().enumerate() {
            let c = (i / dm) % out_channels;
            *v *= scale[c];
        }
    } else {
        // OIHW or pre-packed grouped [O, KH, KW, IC/G] — O is outermost.
        let elems_per_channel = w_data.len() / out_channels;
        for c in 0..out_channels {
            let start = c * elems_per_channel;
            let end = start + elems_per_channel;
            for v in &mut w_data[start..end] {
                *v *= scale[c];
            }
        }
    }
}

/// Folds Conv + Mul(conv_out, const_scale) pairs by absorbing the scale into
/// Conv weights and bias. Mirrors ORT's Level-1 ConvMulFusion.
///
/// Pattern: `Conv(x, W, b?) → Mul(conv_out, scale_init)` where `scale_init`
/// is a graph initializer with shape `[1]`, `[OC]`, or `[1,OC,1,1]` and the
/// Conv output is only consumed by the Mul. After fusion:
///   W'[c] = W[c] · s[c]
///   b'[c] = b[c] · s[c]
pub fn fold_conv_mul(model: &mut OnnxModel) {
    let mut fuse_pairs: Vec<(usize, usize, usize)> = Vec::new(); // (conv_idx, mul_idx, scale_input_idx)

    for i in 0..model.nodes.len().saturating_sub(1) {
        let node = &model.nodes[i];
        let next = &model.nodes[i + 1];
        if node.op_type != "Conv"
            || next.op_type != "Mul"
            || next.inputs.len() != 2
            || node.outputs.is_empty()
        {
            continue;
        }
        // One of Mul's inputs must be the conv output, the other a graph initializer.
        let (scale_idx, data_idx) = if next.inputs[0] == node.outputs[0] {
            (1, 0)
        } else if next.inputs[1] == node.outputs[0] {
            (0, 1)
        } else {
            continue;
        };
        if next.inputs[data_idx] != node.outputs[0] {
            continue;
        }
        let scale_name = &next.inputs[scale_idx];
        if !model.initializers.contains_key(scale_name) {
            continue;
        }
        // Conv output must be sole-consumer (this Mul, not a model output).
        let conv_out = &node.outputs[0];
        let consumers: usize = model
            .nodes
            .iter()
            .enumerate()
            .filter(|&(j, n)| j != i + 1 && n.inputs.contains(conv_out))
            .count();
        if consumers != 0 || model.outputs.contains(conv_out) {
            continue;
        }
        fuse_pairs.push((i, i + 1, scale_idx));
    }

    for &(conv_idx, mul_idx, scale_input_idx) in fuse_pairs.iter().rev() {
        let (w_name, bias_name_opt, scale_name, mul_out) = {
            let conv = &model.nodes[conv_idx];
            let mul = &model.nodes[mul_idx];
            if conv.inputs.len() < 2 {
                continue;
            }
            let bias = if conv.inputs.len() >= 3 && !conv.inputs[2].is_empty() {
                Some(conv.inputs[2].clone())
            } else {
                None
            };
            (
                conv.inputs[1].clone(),
                bias,
                mul.inputs[scale_input_idx].clone(),
                mul.outputs[0].clone(),
            )
        };

        // Determine out_channels from the weight layout.
        let w = match model.initializers.get(&w_name) {
            Some(t) => t.clone(),
            None => continue,
        };
        let is_khwc = model.khwc_weights.contains(&w_name);
        let is_dw_khwc = model.dw_khwc_weights.contains(&w_name);
        let is_group_khwc = model.group_khwc_weights.contains(&w_name);
        let out_channels = if is_khwc {
            w.shape()[3]
        } else if is_dw_khwc {
            w.shape()[2]
        } else if is_group_khwc {
            w.shape()[0]
        } else {
            w.shape()[0]
        };

        let scale = match broadcast_scale_to_oc(model, &scale_name, out_channels) {
            Some(s) => s,
            None => continue,
        };

        let mut w_data = w.data().to_vec();
        scale_weight_inplace(
            &mut w_data,
            w.shape(),
            out_channels,
            &scale,
            is_khwc,
            is_dw_khwc,
        );
        let w_fused =
            Tensor::from_vec(w.shape().to_vec(), w_data).expect("fused weight shape matches data");
        model.initializers.insert(w_name.clone(), w_fused);

        if let Some(bname) = &bias_name_opt
            && let Some(old_bias) = model.initializers.get(bname).cloned()
        {
            let bd = old_bias.data();
            let b_new: Vec<f32> = (0..out_channels).map(|c| bd[c] * scale[c]).collect();
            let b_fused =
                Tensor::from_vec(vec![out_channels], b_new).expect("fused bias shape matches data");
            model.initializers.insert(bname.clone(), b_fused);
        }

        // Rewire: Conv output takes Mul's output name; remove Mul node.
        model.nodes[conv_idx].outputs[0] = mul_out;
        model.nodes.remove(mul_idx);
    }
    model.rebuild_runtime_index();
}

/// Folds Conv + Add(conv_out, const_bias_init) by adding the constant into
/// the Conv bias (creating a fresh bias initializer if the Conv had none).
///
/// Residual (non-constant) Adds are handled at runtime by `NodeAction::ConvAdd`
/// and are NOT touched here — we only fold when the second Add operand is a
/// graph initializer broadcastable to OC.
pub fn fold_conv_add_const(model: &mut OnnxModel) {
    let mut fuse_pairs: Vec<(usize, usize, usize)> = Vec::new();

    for i in 0..model.nodes.len().saturating_sub(1) {
        let node = &model.nodes[i];
        let next = &model.nodes[i + 1];
        // Accept both plain Conv and Conv+Relu-marked nodes? No — the add
        // must come *before* the Relu to be absorbable into bias. Only plain Conv.
        if node.op_type != "Conv"
            || next.op_type != "Add"
            || next.inputs.len() != 2
            || node.outputs.is_empty()
        {
            continue;
        }
        let (bias_idx, data_idx) = if next.inputs[0] == node.outputs[0] {
            (1, 0)
        } else if next.inputs[1] == node.outputs[0] {
            (0, 1)
        } else {
            continue;
        };
        if next.inputs[data_idx] != node.outputs[0] {
            continue;
        }
        let bias_name = &next.inputs[bias_idx];
        if !model.initializers.contains_key(bias_name) {
            // Non-const second operand → this is a residual Add; leave for
            // runtime NodeAction::ConvAdd fusion.
            continue;
        }
        let conv_out = &node.outputs[0];
        let consumers: usize = model
            .nodes
            .iter()
            .enumerate()
            .filter(|&(j, n)| j != i + 1 && n.inputs.contains(conv_out))
            .count();
        if consumers != 0 || model.outputs.contains(conv_out) {
            continue;
        }
        fuse_pairs.push((i, i + 1, bias_idx));
    }

    for &(conv_idx, add_idx, bias_input_idx) in fuse_pairs.iter().rev() {
        let (w_name, existing_bias_opt, add_bias_name, add_out) = {
            let conv = &model.nodes[conv_idx];
            let add = &model.nodes[add_idx];
            if conv.inputs.len() < 2 {
                continue;
            }
            let existing_bias = if conv.inputs.len() >= 3 && !conv.inputs[2].is_empty() {
                Some(conv.inputs[2].clone())
            } else {
                None
            };
            (
                conv.inputs[1].clone(),
                existing_bias,
                add.inputs[bias_input_idx].clone(),
                add.outputs[0].clone(),
            )
        };

        let w = match model.initializers.get(&w_name) {
            Some(t) => t.clone(),
            None => continue,
        };
        let is_khwc = model.khwc_weights.contains(&w_name);
        let is_dw_khwc = model.dw_khwc_weights.contains(&w_name);
        let is_group_khwc = model.group_khwc_weights.contains(&w_name);
        let out_channels = if is_khwc {
            w.shape()[3]
        } else if is_dw_khwc {
            w.shape()[2]
        } else if is_group_khwc {
            w.shape()[0]
        } else {
            w.shape()[0]
        };

        let add_bias_vec = match broadcast_scale_to_oc(model, &add_bias_name, out_channels) {
            Some(v) => v,
            None => continue,
        };

        let old_bias: Vec<f32> = if let Some(bname) = &existing_bias_opt {
            model
                .initializers
                .get(bname)
                .map(|t| t.data().to_vec())
                .unwrap_or_else(|| vec![0.0; out_channels])
        } else {
            vec![0.0; out_channels]
        };

        let b_new: Vec<f32> = (0..out_channels)
            .map(|c| old_bias[c] + add_bias_vec[c])
            .collect();
        let b_fused =
            Tensor::from_vec(vec![out_channels], b_new).expect("fused bias shape matches data");

        let bias_name = if let Some(bname) = existing_bias_opt {
            bname
        } else {
            let name = format!("{}_fused_add_bias", model.nodes[conv_idx].name);
            model.nodes[conv_idx].inputs.push(name.clone());
            name
        };
        model.initializers.insert(bias_name, b_fused);

        model.nodes[conv_idx].outputs[0] = add_out;
        model.nodes.remove(add_idx);
    }
    model.rebuild_runtime_index();
}

/// Eliminates inverse Squeeze/Unsqueeze pairs left by ONNX export pipelines
/// (common after PyTorch → ONNX for broadcasting workarounds). Matches both
/// Squeeze(Unsqueeze(x, axes=A), axes=A) and Unsqueeze(Squeeze(x, axes=A), axes=A).
pub fn eliminate_squeeze_unsqueeze_pairs(model: &mut OnnxModel) {
    let mut remove_pairs: Vec<(usize, usize, String, String)> = Vec::new(); // (first, second, producer_input, second_output)

    for i in 0..model.nodes.len().saturating_sub(1) {
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
        // First op's output must be sole-consumed by the second.
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
        // Axes must match: both ops take their axes either from an attr or
        // from a second input initializer. Check both sources.
        let first_axes = node_axes(model, first);
        let second_axes = node_axes(model, second);
        if first_axes.is_none() || first_axes != second_axes {
            continue;
        }

        if first.inputs.is_empty() || second.outputs.is_empty() {
            continue;
        }
        remove_pairs.push((i, i + 1, first.inputs[0].clone(), second.outputs[0].clone()));
    }

    for &(first_idx, second_idx, ref producer_input, ref consumer_output) in
        remove_pairs.iter().rev()
    {
        // Rewire any downstream consumer of `consumer_output` to `producer_input`.
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
        // Remove the higher index first so indices stay valid.
        model.nodes.remove(second_idx);
        model.nodes.remove(first_idx);
    }
    model.rebuild_runtime_index();
}

/// Reads axes from a Squeeze/Unsqueeze node (opset ≥13 uses a second input
/// initializer, older opsets use the "axes" attribute). Returns a sorted
/// `Vec<i64>` for direct comparison between paired nodes.
fn node_axes(model: &OnnxModel, node: &crate::loader::OnnxNode) -> Option<Vec<i64>> {
    // Opset-13+: second input is an initializer with the axes values.
    if node.inputs.len() > 1 && !node.inputs[1].is_empty() {
        let t = model.initializers.get(&node.inputs[1])?;
        let mut v: Vec<i64> = t.data().iter().map(|&x| x as i64).collect();
        v.sort_unstable();
        return Some(v);
    }
    // Older opsets: "axes" attribute.
    if let Some(crate::loader::OnnxAttribute::Ints(axes)) = node.attributes.get("axes") {
        let mut v = axes.clone();
        v.sort_unstable();
        return Some(v);
    }
    None
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
                // Skip Conv/depthwise: weights may be pre-permuted (KHWC) and
                // the mini-model runner lacks layout tracking metadata.
                && !matches!(node.op_type.as_str(), "Conv" | "ConvTranspose" | "DeformConv")
        });

        let (idx, _) = match foldable {
            Some(pair) => pair,
            None => break,
        };

        let node = model.nodes[idx].clone();
        // Identity with a single initializer input: directly alias the output
        // to the same tensor. This preserves KHWC/dw_khwc/group_khwc weight
        // tracking that would otherwise be lost via mini-model execution.
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

        // Build a minimal model with just this node to execute it
        let mut mini_model = OnnxModel {
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
            group_khwc_weights: HashSet::new(),
            packed_int4_weights: Default::default(),
            runtime_index: Default::default(),
        };
        mini_model.rebuild_runtime_index();

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
    model.rebuild_runtime_index();
}

/// Summary statistics describing NCHWc-transformability of a loaded graph.
///
/// Produced by [`analyze_nchwc`] — inspects the graph for op patterns that
/// could stay in NCHWc layout (Conv, Pool, BN, elementwise Add/Relu/Sigmoid,
/// SiLU-as-Sigmoid-Mul) and counts maximal contiguous NCHWc-capable chains.
/// Used by the runner and by B.8 rollout decisions to judge whether
/// `YSCV_NCHWC=on` is worth flipping for a given model.
///
/// The metric that matters is `max_chain_len`: conversion cost at the
/// chain boundaries is paid once per chain, so longer chains amortize
/// better. Per the plan's cost model, chains of ≥3 NCHWc-capable ops
/// are worth converting; anything smaller burns more on layout reorders
/// than it saves on kernel speedup.
#[derive(Debug, Clone)]
pub struct NchwcStats {
    /// Total count of nodes whose op_type could run natively in NCHWc.
    pub capable_nodes: usize,
    /// Total count of all nodes in the graph.
    pub total_nodes: usize,
    /// Number of maximal NCHWc-capable chains (consecutive runs).
    pub chain_count: usize,
    /// Longest single chain of consecutive NCHWc-capable ops.
    pub max_chain_len: usize,
    /// Mean chain length across all chains (0 when no chains).
    pub mean_chain_len: f32,
    /// Set of NCHWc-capable op_types encountered, with counts.
    pub op_types_capable: Vec<(String, usize)>,
}

/// Returns `true` if this op_type has an NCHWc-native (or layout-agnostic)
/// path in yscv-kernels today. See [`yscv_kernels::conv2d_nchwc_with_activation_prepacked`],
/// [`yscv_kernels::max_pool2d_nchwc`], etc.
fn is_nchwc_capable_op(op_type: &str) -> bool {
    matches!(
        op_type,
        "Conv"
            | "MaxPool"
            | "AveragePool"
            | "GlobalAveragePool"
            | "BatchNormalization"
            | "Relu"
            | "Sigmoid"
            | "Add"
            | "Mul"
            | "Clip"
    )
}

/// Analyzes a loaded model for NCHWc-transformability. Does **not** mutate
/// the graph — returns a read-only report. Intended for diagnostics and
/// for runner / profiler logic that decides whether to activate the
/// NCHWc pipeline.
pub fn analyze_nchwc(model: &OnnxModel) -> NchwcStats {
    let total_nodes = model.nodes.len();
    let mut op_counts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    let mut capable_nodes = 0usize;
    let mut chain_lengths: Vec<usize> = Vec::new();
    let mut current_chain = 0usize;

    for node in &model.nodes {
        if is_nchwc_capable_op(&node.op_type) {
            capable_nodes += 1;
            current_chain += 1;
            *op_counts.entry(node.op_type.clone()).or_insert(0) += 1;
        } else if current_chain > 0 {
            chain_lengths.push(current_chain);
            current_chain = 0;
        }
    }
    if current_chain > 0 {
        chain_lengths.push(current_chain);
    }

    let chain_count = chain_lengths.len();
    let max_chain_len = chain_lengths.iter().copied().max().unwrap_or(0);
    let mean_chain_len = if chain_count == 0 {
        0.0
    } else {
        chain_lengths.iter().copied().sum::<usize>() as f32 / chain_count as f32
    };
    let mut op_types_capable: Vec<(String, usize)> = op_counts.into_iter().collect();
    op_types_capable.sort_by_key(|&(_, count)| std::cmp::Reverse(count));

    NchwcStats {
        capable_nodes,
        total_nodes,
        chain_count,
        max_chain_len,
        mean_chain_len,
        op_types_capable,
    }
}

#[cfg(test)]
mod nchwc_stats_tests {
    use super::*;
    use crate::loader::{OnnxModel, OnnxNode};

    fn node(op: &str, name: &str) -> OnnxNode {
        OnnxNode {
            op_type: op.to_string(),
            name: name.to_string(),
            inputs: vec![],
            outputs: vec![],
            attributes: HashMap::new(),
        }
    }

    fn model_with(ops: &[&str]) -> OnnxModel {
        let mut m = OnnxModel {
            ir_version: 0,
            opset_version: 0,
            producer_name: String::new(),
            graph_name: String::new(),
            inputs: vec![],
            outputs: vec![],
            initializers: HashMap::new(),
            nodes: vec![],
            khwc_weights: HashSet::new(),
            dw_khwc_weights: HashSet::new(),
            group_khwc_weights: HashSet::new(),
            packed_int4_weights: Default::default(),
            runtime_index: Default::default(),
        };
        for (i, op) in ops.iter().enumerate() {
            m.nodes.push(node(op, &format!("n{i}")));
        }
        m
    }

    #[test]
    fn analyze_detects_no_chain() {
        let m = model_with(&["Reshape", "Gather"]);
        let s = analyze_nchwc(&m);
        assert_eq!(s.capable_nodes, 0);
        assert_eq!(s.chain_count, 0);
        assert_eq!(s.max_chain_len, 0);
    }

    #[test]
    fn analyze_counts_single_chain() {
        let m = model_with(&["Conv", "BatchNormalization", "Relu", "Conv", "Add"]);
        let s = analyze_nchwc(&m);
        assert_eq!(s.capable_nodes, 5);
        assert_eq!(s.chain_count, 1);
        assert_eq!(s.max_chain_len, 5);
    }

    #[test]
    fn analyze_splits_on_incapable_op() {
        let m = model_with(&["Conv", "Reshape", "Conv", "Relu", "Transpose", "Conv"]);
        let s = analyze_nchwc(&m);
        assert_eq!(s.capable_nodes, 4);
        assert_eq!(s.chain_count, 3);
        assert_eq!(s.max_chain_len, 2);
    }

    #[test]
    fn analyze_computes_mean_chain() {
        let m = model_with(&["Conv", "Conv", "Reshape", "Conv", "Conv", "Conv"]);
        let s = analyze_nchwc(&m);
        assert_eq!(s.chain_count, 2);
        assert_eq!(s.max_chain_len, 3);
        assert!((s.mean_chain_len - 2.5).abs() < 1e-6);
    }
}
