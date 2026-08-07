use yscv_tensor::Tensor;

use crate::loader::OnnxModel;

/// Returns the per-output-channel scaling vector for a Conv weight layout.
/// Copies the slice out so callers can multiply in-place without borrowing
/// the initializer map. `None` means the scale tensor was not usable
/// (not an initializer, wrong shape, wrong element count).
pub(crate) fn broadcast_scale_to_oc(
    model: &OnnxModel,
    scale_name: &str,
    out_channels: usize,
) -> Option<Vec<f32>> {
    let t = model.initializers.get(scale_name)?;
    let data = t.data();
    let shape = t.shape();
    let numel: usize = shape.iter().product();
    if numel == 1 {
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
        for (i, v) in w_data.iter_mut().enumerate() {
            *v *= scale[i % out_channels];
        }
    } else if is_dw_khwc {
        let dm = w_shape[3];
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
}

/// Folds Conv + Mul(conv_out, const_scale) pairs by absorbing the scale into
/// Conv weights and bias. Mirrors ORT's Level-1 ConvMulFusion.
///
/// Pattern: `Conv(x, W, b?) → Mul(conv_out, scale_init)` where `scale_init`
/// is a graph initializer with shape `[1]`, `[OC]`, or `[1,OC,1,1]` and the
/// Conv output is only consumed by the Mul. After fusion:
///   W'[c] = W[c] · s[c]
///   b'[c] = b[c] · s[c]
/// Folds without rebuilding the runtime index; returns whether the graph
/// changed. The driver rebuilds once for the whole pipeline.
pub(super) fn run(model: &mut OnnxModel) -> bool {
    let mut changed = false;
    let mut fuse_pairs: Vec<(usize, usize, usize)> = Vec::new();

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

        model.nodes[conv_idx].outputs[0] = mul_out;
        model.nodes.remove(mul_idx);
        changed = true;
    }
    changed
}
