use yscv_tensor::Tensor;

use crate::loader::OnnxModel;

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
            w.shape()[3]
        } else if is_dw_khwc {
            w.shape()[2]
        } else if is_group_khwc {
            w.shape()[0]
        } else {
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

        let scale: Vec<f32> = (0..out_channels)
            .map(|c| gamma_d[c] / (var_d[c] + epsilon).sqrt())
            .collect();

        let w_shape = w.shape().to_vec();
        let mut w_data = w.data().to_vec();
        if is_khwc {
            for (i, v) in w_data.iter_mut().enumerate() {
                *v *= scale[i % out_channels];
            }
        } else if is_dw_khwc {
            let dm = w.shape()[3];
            for (i, v) in w_data.iter_mut().enumerate() {
                let c = (i / dm) % out_channels;
                *v *= scale[c];
            }
        } else {
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

        model.initializers.insert(w_name.clone(), w_fused);

        let bias_name = if conv_has_bias {
            conv_node.inputs[2].clone()
        } else {
            let name = format!("{}_fused_bias", conv_node.name);
            model.nodes[conv_idx].inputs.push(name.clone());
            name
        };
        model.initializers.insert(bias_name, b_fused);

        let bn_output = model.nodes[bn_idx].outputs[0].clone();
        model.nodes[conv_idx].outputs[0] = bn_output;

        model.nodes.remove(bn_idx);
    }
    model.rebuild_runtime_index();
}
