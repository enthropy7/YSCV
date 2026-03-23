use super::reduce::compute_strides;
use super::*;

pub(super) fn exec_flatten(node: &OnnxNode, env: &mut TensorEnv) -> Result<(), OnnxError> {
    let input = get_tensor(env, &node.name, &node.inputs[0])?;
    let axis = get_attr_int(node, "axis").unwrap_or(1) as usize;
    let shape = input.shape();

    let outer: usize = shape[..axis].iter().product();
    let inner: usize = shape[axis..].iter().product();
    let out = input
        .reshape(vec![outer, inner])
        .map_err(|e| OnnxError::DecodeFailed {
            message: e.to_string(),
        })?;
    env.insert(node.outputs[0].clone(), out);
    Ok(())
}

pub(super) fn exec_reshape(node: &OnnxNode, env: &mut TensorEnv) -> Result<(), OnnxError> {
    let input = get_tensor(env, &node.name, &node.inputs[0])?;
    let shape_tensor = get_tensor(env, &node.name, &node.inputs[1])?;
    let total = input.len();
    let raw_dims: Vec<f32> = shape_tensor.data().to_vec();
    let in_shape = input.shape().to_vec();

    let mut neg_idx = None;
    let mut new_shape = Vec::with_capacity(raw_dims.len());
    for (i, &dim_f) in raw_dims.iter().enumerate() {
        let d = dim_f as i64;
        if d == -1 {
            neg_idx = Some(i);
            new_shape.push(1); // placeholder
        } else if d == 0 {
            new_shape.push(if i < in_shape.len() { in_shape[i] } else { 1 });
        } else {
            new_shape.push(d as usize);
        }
    }

    if let Some(idx) = neg_idx {
        let known: usize = new_shape
            .iter()
            .enumerate()
            .filter(|&(i, _)| i != idx)
            .map(|(_, &d)| d)
            .product();
        new_shape[idx] = if known > 0 { total / known } else { total };
    }

    let out = input
        .reshape(new_shape)
        .map_err(|e| OnnxError::DecodeFailed {
            message: e.to_string(),
        })?;
    env.insert(node.outputs[0].clone(), out);
    Ok(())
}

pub(super) fn exec_transpose(node: &OnnxNode, env: &mut TensorEnv) -> Result<(), OnnxError> {
    let input = get_tensor(env, &node.name, &node.inputs[0])?;
    let perm = get_attr_ints(node, "perm");

    let axes: Vec<usize> = match perm {
        Some(p) => p.iter().map(|&v| v as usize).collect(),
        None => (0..input.rank()).rev().collect(),
    };

    let out = input.permute(&axes).map_err(|e| OnnxError::DecodeFailed {
        message: e.to_string(),
    })?;
    env.insert(node.outputs[0].clone(), out);
    Ok(())
}

pub(super) fn exec_concat(node: &OnnxNode, env: &mut TensorEnv) -> Result<(), OnnxError> {
    let axis = get_attr_int(node, "axis").unwrap_or(0);
    let tensors: Vec<Tensor> = node
        .inputs
        .iter()
        .filter(|name| !name.is_empty())
        .map(|name| get_tensor(env, &node.name, name).cloned())
        .collect::<Result<Vec<_>, _>>()?;

    let refs: Vec<&Tensor> = tensors.iter().collect();
    let rank = refs[0].rank();
    let actual_axis = if axis < 0 {
        (rank as i64 + axis) as usize
    } else {
        axis as usize
    };

    let out = Tensor::cat(&refs, actual_axis).map_err(|e| OnnxError::DecodeFailed {
        message: e.to_string(),
    })?;
    env.insert(node.outputs[0].clone(), out);
    Ok(())
}

pub(super) fn exec_unsqueeze(node: &OnnxNode, env: &mut TensorEnv) -> Result<(), OnnxError> {
    let input = get_tensor(env, &node.name, &node.inputs[0])?;

    // ONNX >= opset 13: axes from second input tensor; older: from attribute
    let axes: Vec<i64> = if node.inputs.len() > 1 && !node.inputs[1].is_empty() {
        let axes_tensor = get_tensor(env, &node.name, &node.inputs[1])?;
        axes_tensor.data().iter().map(|&v| v as i64).collect()
    } else {
        get_attr_ints(node, "axes").unwrap_or_default()
    };

    let mut shape = input.shape().to_vec();
    let mut sorted_axes: Vec<usize> = axes
        .iter()
        .map(|&a| {
            if a < 0 {
                (shape.len() as i64 + 1 + a) as usize
            } else {
                a as usize
            }
        })
        .collect();
    sorted_axes.sort();
    for &ax in &sorted_axes {
        shape.insert(ax, 1);
    }
    let out = input.reshape(shape).map_err(|e| OnnxError::DecodeFailed {
        message: e.to_string(),
    })?;
    env.insert(node.outputs[0].clone(), out);
    Ok(())
}

pub(super) fn exec_squeeze(node: &OnnxNode, env: &mut TensorEnv) -> Result<(), OnnxError> {
    let input = get_tensor(env, &node.name, &node.inputs[0])?;

    let axes: Vec<i64> = if node.inputs.len() > 1 && !node.inputs[1].is_empty() {
        let axes_tensor = get_tensor(env, &node.name, &node.inputs[1])?;
        axes_tensor.data().iter().map(|&v| v as i64).collect()
    } else {
        get_attr_ints(node, "axes").unwrap_or_default()
    };

    let mut shape = input.shape().to_vec();
    if axes.is_empty() {
        shape.retain(|&d| d != 1);
    } else {
        let mut to_remove: Vec<usize> = axes
            .iter()
            .map(|&a| {
                if a < 0 {
                    (shape.len() as i64 + a) as usize
                } else {
                    a as usize
                }
            })
            .collect();
        to_remove.sort();
        for (i, &ax) in to_remove.iter().enumerate() {
            shape.remove(ax - i);
        }
    }
    if shape.is_empty() {
        shape.push(1);
    }
    let out = input.reshape(shape).map_err(|e| OnnxError::DecodeFailed {
        message: e.to_string(),
    })?;
    env.insert(node.outputs[0].clone(), out);
    Ok(())
}

pub(super) fn exec_shape(node: &OnnxNode, env: &mut TensorEnv) -> Result<(), OnnxError> {
    let input = get_tensor(env, &node.name, &node.inputs[0])?;
    let shape_data: Vec<f32> = input.shape().iter().map(|&d| d as f32).collect();
    let out = Tensor::from_vec(vec![shape_data.len()], shape_data).map_err(|e| {
        OnnxError::DecodeFailed {
            message: e.to_string(),
        }
    })?;
    env.insert(node.outputs[0].clone(), out);
    Ok(())
}

fn equal_split(dim: usize, num_outputs: usize) -> Vec<usize> {
    let base = dim / num_outputs;
    let rem = dim % num_outputs;
    (0..num_outputs)
        .map(|i| if i < rem { base + 1 } else { base })
        .collect()
}

pub(super) fn exec_split(node: &OnnxNode, env: &mut TensorEnv) -> Result<(), OnnxError> {
    let input = get_tensor(env, &node.name, &node.inputs[0])?.clone();
    let axis = get_attr_int(node, "axis").unwrap_or(0);
    let shape = input.shape();
    let rank = shape.len();
    let axis_usize = if axis < 0 {
        (rank as i64 + axis) as usize
    } else {
        axis as usize
    };
    let dim = shape[axis_usize];
    let num_outputs = node.outputs.len();

    // Opset 13+: split sizes come from the second input tensor;
    // older opsets use the "split" attribute.
    let split_sizes: Vec<usize> = if node.inputs.len() > 1 && !node.inputs[1].is_empty() {
        if let Ok(split_t) = get_tensor(env, &node.name, &node.inputs[1]) {
            split_t.data().iter().map(|&v| v as usize).collect()
        } else {
            equal_split(dim, num_outputs)
        }
    } else if let Some(s) = get_attr_ints(node, "split") {
        s.iter().map(|&v| v as usize).collect()
    } else {
        equal_split(dim, num_outputs)
    };

    let data = input.data();
    let strides = compute_strides(shape);
    let mut offset_along_axis = 0;
    for (out_idx, &sz) in split_sizes.iter().enumerate() {
        let mut out_shape = shape.to_vec();
        out_shape[axis_usize] = sz;
        let out_size: usize = out_shape.iter().product();
        let mut out_data = Vec::with_capacity(out_size);
        let out_strides = compute_strides(&out_shape);
        for flat in 0..out_size {
            let mut coords = vec![0usize; rank];
            let mut r = flat;
            for d in 0..rank {
                coords[d] = r / out_strides[d];
                r %= out_strides[d];
            }
            coords[axis_usize] += offset_along_axis;
            let src = coords
                .iter()
                .zip(strides.iter())
                .map(|(&c, &s)| c * s)
                .sum::<usize>();
            out_data.push(data[src]);
        }
        offset_along_axis += sz;
        if out_idx < node.outputs.len() {
            let t = Tensor::from_vec(out_shape, out_data).map_err(|e| OnnxError::DecodeFailed {
                message: e.to_string(),
            })?;
            env.insert(node.outputs[out_idx].clone(), t);
        }
    }
    Ok(())
}

pub(super) fn exec_slice(node: &OnnxNode, env: &mut TensorEnv) -> Result<(), OnnxError> {
    let input = get_tensor(env, &node.name, &node.inputs[0])?;
    let shape = input.shape();
    let rank = shape.len();

    let starts_t = get_tensor(env, &node.name, &node.inputs[1])?;
    let ends_t = get_tensor(env, &node.name, &node.inputs[2])?;
    let axes_t = if node.inputs.len() > 3 && !node.inputs[3].is_empty() {
        Some(get_tensor(env, &node.name, &node.inputs[3])?)
    } else {
        None
    };
    let steps_t = if node.inputs.len() > 4 && !node.inputs[4].is_empty() {
        Some(get_tensor(env, &node.name, &node.inputs[4])?)
    } else {
        None
    };

    let mut starts = vec![0i64; rank];
    let mut ends: Vec<i64> = shape.iter().map(|&d| d as i64).collect();
    let mut steps = vec![1i64; rank];

    let s_data = starts_t.data();
    let e_data = ends_t.data();
    let axes: Vec<usize> = if let Some(at) = axes_t {
        at.data()
            .iter()
            .map(|&v| {
                let i = v as i64;
                if i < 0 {
                    (rank as i64 + i) as usize
                } else {
                    i as usize
                }
            })
            .collect()
    } else {
        (0..s_data.len()).collect()
    };

    for (i, &ax) in axes.iter().enumerate() {
        starts[ax] = s_data[i] as i64;
        ends[ax] = e_data[i] as i64;
        if let Some(st) = steps_t {
            steps[ax] = st.data()[i] as i64;
        }
    }

    // Normalize negative indices and clamp
    for d in 0..rank {
        let dim = shape[d] as i64;
        if starts[d] < 0 {
            starts[d] += dim;
        }
        if ends[d] < 0 {
            ends[d] += dim;
        }
        starts[d] = starts[d].max(0).min(dim);
        ends[d] = ends[d].max(0).min(dim);
    }

    let mut out_shape = Vec::with_capacity(rank);
    for d in 0..rank {
        let s = ((ends[d] - starts[d]) as f64 / steps[d] as f64).ceil() as usize;
        out_shape.push(s.max(0));
    }

    let out_size: usize = out_shape.iter().product();
    let data = input.data();
    let in_strides = compute_strides(shape);
    let out_strides = compute_strides(&out_shape);
    let mut out_data = Vec::with_capacity(out_size);

    for flat in 0..out_size {
        let mut in_idx = 0;
        let mut r = flat;
        for d in 0..rank {
            let coord = r / out_strides[d];
            r %= out_strides[d];
            let src_coord = starts[d] as usize + coord * steps[d] as usize;
            in_idx += src_coord * in_strides[d];
        }
        out_data.push(data[in_idx]);
    }

    let out = Tensor::from_vec(out_shape, out_data).map_err(|e| OnnxError::DecodeFailed {
        message: e.to_string(),
    })?;
    env.insert(node.outputs[0].clone(), out);
    Ok(())
}

pub(super) fn exec_expand(node: &OnnxNode, env: &mut TensorEnv) -> Result<(), OnnxError> {
    let input = get_tensor(env, &node.name, &node.inputs[0])?;
    let shape_t = get_tensor(env, &node.name, &node.inputs[1])?;
    let target_shape: Vec<usize> = shape_t.data().iter().map(|&v| v as usize).collect();
    let in_shape = input.shape();

    let rank = target_shape.len().max(in_shape.len());
    let mut out_shape = vec![1; rank];
    for i in 0..rank {
        let in_d = if i < rank - in_shape.len() {
            1
        } else {
            in_shape[i - (rank - in_shape.len())]
        };
        let tgt_d = if i < rank - target_shape.len() {
            1
        } else {
            target_shape[i - (rank - target_shape.len())]
        };
        out_shape[i] = in_d.max(tgt_d);
    }

    let out_size: usize = out_shape.iter().product();
    let data = input.data();
    let in_strides = compute_strides(in_shape);
    let out_strides = compute_strides(&out_shape);
    let mut out_data = Vec::with_capacity(out_size);

    let in_padded: Vec<usize> = {
        let pad = rank - in_shape.len();
        let mut s = vec![1; pad];
        s.extend_from_slice(in_shape);
        s
    };

    for flat in 0..out_size {
        let mut src_idx = 0;
        let mut r = flat;
        for d in 0..rank {
            let coord = r / out_strides[d];
            r %= out_strides[d];
            let in_coord = if in_padded[d] == 1 { 0 } else { coord };
            if d >= rank - in_shape.len() {
                src_idx += in_coord * in_strides[d - (rank - in_shape.len())];
            }
        }
        out_data.push(data[src_idx]);
    }

    let out = Tensor::from_vec(out_shape, out_data).map_err(|e| OnnxError::DecodeFailed {
        message: e.to_string(),
    })?;
    env.insert(node.outputs[0].clone(), out);
    Ok(())
}

pub(super) fn exec_tile(node: &OnnxNode, env: &mut TensorEnv) -> Result<(), OnnxError> {
    let input = get_tensor(env, &node.name, &node.inputs[0])?;
    let repeats_t = get_tensor(env, &node.name, &node.inputs[1])?;
    let repeats: Vec<usize> = repeats_t.data().iter().map(|&v| v as usize).collect();
    let shape = input.shape();
    let rank = shape.len();

    let out_shape: Vec<usize> = shape
        .iter()
        .zip(repeats.iter())
        .map(|(&s, &r)| s * r)
        .collect();
    let out_size: usize = out_shape.iter().product();
    let data = input.data();
    let in_strides = compute_strides(shape);
    let out_strides = compute_strides(&out_shape);
    let mut out_data = Vec::with_capacity(out_size);

    for flat in 0..out_size {
        let mut src_idx = 0;
        let mut r = flat;
        for d in 0..rank {
            let coord = r / out_strides[d];
            r %= out_strides[d];
            src_idx += (coord % shape[d]) * in_strides[d];
        }
        out_data.push(data[src_idx]);
    }

    let out = Tensor::from_vec(out_shape, out_data).map_err(|e| OnnxError::DecodeFailed {
        message: e.to_string(),
    })?;
    env.insert(node.outputs[0].clone(), out);
    Ok(())
}

pub(super) fn exec_cast(node: &OnnxNode, env: &mut TensorEnv) -> Result<(), OnnxError> {
    // Passthrough for f32 -- we only support f32 tensors currently
    let input = get_tensor(env, &node.name, &node.inputs[0])?;
    env.insert(node.outputs[0].clone(), input.clone());
    Ok(())
}

pub(super) fn exec_pad(node: &OnnxNode, env: &mut TensorEnv) -> Result<(), OnnxError> {
    let input = get_tensor(env, &node.name, &node.inputs[0])?;
    let pads_tensor = if node.inputs.len() > 1 && !node.inputs[1].is_empty() {
        Some(get_tensor(env, &node.name, &node.inputs[1])?)
    } else {
        None
    };
    let constant_value = if node.inputs.len() > 2 && !node.inputs[2].is_empty() {
        get_tensor(env, &node.name, &node.inputs[2])?.data()[0]
    } else {
        0.0
    };

    let pads: Vec<i64> = if let Some(pt) = pads_tensor {
        pt.data().iter().map(|&v| v as i64).collect()
    } else {
        get_attr_ints(node, "pads").unwrap_or_default()
    };

    if pads.iter().all(|&p| p == 0) {
        env.insert(node.outputs[0].clone(), input.clone());
        return Ok(());
    }

    // ONNX pads format: [x1_begin, x2_begin, ..., x1_end, x2_end, ...]
    let rank = input.rank();
    let shape = input.shape();
    let mut new_shape = vec![0usize; rank];
    for i in 0..rank {
        let pad_begin = pads[i] as usize;
        let pad_end = pads[rank + i] as usize;
        new_shape[i] = shape[i] + pad_begin + pad_end;
    }
    let total: usize = new_shape.iter().product();
    let mut out_data = vec![constant_value; total];
    let in_data = input.data();

    // Copy input data into padded output
    let mut in_strides = vec![1usize; rank];
    for i in (0..rank - 1).rev() {
        in_strides[i] = in_strides[i + 1] * shape[i + 1];
    }
    let mut out_strides = vec![1usize; rank];
    for i in (0..rank - 1).rev() {
        out_strides[i] = out_strides[i + 1] * new_shape[i + 1];
    }

    for (flat_idx, &val) in in_data.iter().enumerate() {
        let mut remainder = flat_idx;
        let mut out_flat = 0;
        for d in 0..rank {
            let coord = remainder / in_strides[d];
            remainder %= in_strides[d];
            out_flat += (coord + pads[d] as usize) * out_strides[d];
        }
        out_data[out_flat] = val;
    }

    let out = Tensor::from_vec(new_shape, out_data).map_err(|e| OnnxError::DecodeFailed {
        message: e.to_string(),
    })?;
    env.insert(node.outputs[0].clone(), out);
    Ok(())
}

pub(super) fn exec_depth_to_space(node: &OnnxNode, env: &mut TensorEnv) -> Result<(), OnnxError> {
    let input = get_tensor(env, &node.name, &node.inputs[0])?;
    let blocksize = get_attr_int(node, "blocksize").ok_or_else(|| OnnxError::MissingAttribute {
        node: node.name.clone(),
        attr: "blocksize".into(),
    })? as usize;
    let mode = get_attr_string(node, "mode").unwrap_or_default();
    let shape = input.shape();
    if shape.len() != 4 {
        return Err(OnnxError::ShapeMismatch {
            detail: "DepthToSpace requires rank-4 NCHW".into(),
        });
    }
    let (n, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
    let bs2 = blocksize * blocksize;
    if c % bs2 != 0 {
        return Err(OnnxError::ShapeMismatch {
            detail: "C must be divisible by blocksize^2".into(),
        });
    }
    let new_c = c / bs2;
    let new_h = h * blocksize;
    let new_w = w * blocksize;
    let data = input.data();
    let mut out = vec![0.0f32; n * new_c * new_h * new_w];
    let dcr = mode != "CRD";

    for nn in 0..n {
        for oc in 0..new_c {
            for oh in 0..new_h {
                for ow in 0..new_w {
                    let bh = oh % blocksize;
                    let bw = ow % blocksize;
                    let ih = oh / blocksize;
                    let iw = ow / blocksize;
                    let ic = if dcr {
                        (bh * blocksize + bw) * new_c + oc
                    } else {
                        oc * bs2 + bh * blocksize + bw
                    };
                    let src_idx = ((nn * c + ic) * h + ih) * w + iw;
                    let dst_idx = ((nn * new_c + oc) * new_h + oh) * new_w + ow;
                    out[dst_idx] = data[src_idx];
                }
            }
        }
    }

    let t = Tensor::from_vec(vec![n, new_c, new_h, new_w], out).map_err(|e| {
        OnnxError::DecodeFailed {
            message: e.to_string(),
        }
    })?;
    env.insert(node.outputs[0].clone(), t);
    Ok(())
}

pub(super) fn exec_space_to_depth(node: &OnnxNode, env: &mut TensorEnv) -> Result<(), OnnxError> {
    let input = get_tensor(env, &node.name, &node.inputs[0])?;
    let blocksize = get_attr_int(node, "blocksize").ok_or_else(|| OnnxError::MissingAttribute {
        node: node.name.clone(),
        attr: "blocksize".into(),
    })? as usize;
    let shape = input.shape();
    if shape.len() != 4 {
        return Err(OnnxError::ShapeMismatch {
            detail: "SpaceToDepth requires rank-4 NCHW".into(),
        });
    }
    let (n, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
    if h % blocksize != 0 || w % blocksize != 0 {
        return Err(OnnxError::ShapeMismatch {
            detail: "H and W must be divisible by blocksize".into(),
        });
    }
    let new_c = c * blocksize * blocksize;
    let new_h = h / blocksize;
    let new_w = w / blocksize;
    let data = input.data();
    let mut out = vec![0.0f32; n * new_c * new_h * new_w];

    for nn in 0..n {
        for ic in 0..c {
            for ih in 0..h {
                for iw in 0..w {
                    let bh = ih % blocksize;
                    let bw = iw % blocksize;
                    let oh = ih / blocksize;
                    let ow = iw / blocksize;
                    let oc = (bh * blocksize + bw) * c + ic;
                    let src_idx = ((nn * c + ic) * h + ih) * w + iw;
                    let dst_idx = ((nn * new_c + oc) * new_h + oh) * new_w + ow;
                    out[dst_idx] = data[src_idx];
                }
            }
        }
    }

    let t = Tensor::from_vec(vec![n, new_c, new_h, new_w], out).map_err(|e| {
        OnnxError::DecodeFailed {
            message: e.to_string(),
        }
    })?;
    env.insert(node.outputs[0].clone(), t);
    Ok(())
}

pub(super) fn exec_constant_of_shape(
    node: &OnnxNode,
    env: &mut TensorEnv,
) -> Result<(), OnnxError> {
    let shape_t = get_tensor(env, &node.name, &node.inputs[0])?;
    let shape: Vec<usize> = shape_t.data().iter().map(|&v| v as usize).collect();
    let value = get_attr_float(node, "value").unwrap_or(0.0);
    let out = Tensor::filled(shape, value).map_err(|e| OnnxError::DecodeFailed {
        message: e.to_string(),
    })?;
    env.insert(node.outputs[0].clone(), out);
    Ok(())
}
