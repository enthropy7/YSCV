use yscv_tensor::Tensor;

use crate::loader::OnnxModel;

use super::fold_conv_mul::broadcast_scale_to_oc;

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
