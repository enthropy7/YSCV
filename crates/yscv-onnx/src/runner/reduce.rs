use super::*;

pub(super) fn exec_reduce_mean(node: &OnnxNode, env: &mut TensorEnv) -> Result<(), OnnxError> {
    let input = get_tensor(env, &node.name, &node.inputs[0])?;
    let axes = get_attr_ints(node, Attr::Axes).unwrap_or_default();
    let keepdims = get_attr_int(node, Attr::KeepDims).unwrap_or(1) != 0;
    let out = reduce_op(input, &axes, keepdims, |vals| {
        vals.iter().sum::<f32>() / vals.len() as f32
    })?;
    env.insert(node.outputs[0].clone(), out);
    Ok(())
}

pub(super) fn exec_reduce_sum(node: &OnnxNode, env: &mut TensorEnv) -> Result<(), OnnxError> {
    let input = get_tensor(env, &node.name, &node.inputs[0])?;
    let axes = get_attr_ints(node, Attr::Axes).unwrap_or_default();
    let keepdims = get_attr_int(node, Attr::KeepDims).unwrap_or(1) != 0;
    let out = reduce_op(input, &axes, keepdims, |vals| vals.iter().sum::<f32>())?;
    env.insert(node.outputs[0].clone(), out);
    Ok(())
}

pub(super) fn exec_reduce_max(node: &OnnxNode, env: &mut TensorEnv) -> Result<(), OnnxError> {
    let input = get_tensor(env, &node.name, &node.inputs[0])?;
    let axes = get_attr_ints(node, Attr::Axes).unwrap_or_default();
    let keepdims = get_attr_int(node, Attr::KeepDims).unwrap_or(1) != 0;
    let out = reduce_op(input, &axes, keepdims, |vals| {
        vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
    })?;
    env.insert(node.outputs[0].clone(), out);
    Ok(())
}

pub(super) fn exec_reduce_min(node: &OnnxNode, env: &mut TensorEnv) -> Result<(), OnnxError> {
    let input = get_tensor(env, &node.name, &node.inputs[0])?;
    let axes = get_attr_ints(node, Attr::Axes).unwrap_or_default();
    let keepdims = get_attr_int(node, Attr::KeepDims).unwrap_or(1) != 0;
    let out = reduce_op(input, &axes, keepdims, |vals: &[f32]| {
        vals.iter().copied().fold(f32::INFINITY, f32::min)
    })?;
    env.insert(node.outputs[0].clone(), out);
    Ok(())
}

pub(super) fn exec_reduce_prod(node: &OnnxNode, env: &mut TensorEnv) -> Result<(), OnnxError> {
    let input = get_tensor(env, &node.name, &node.inputs[0])?;
    let axes = get_attr_ints(node, Attr::Axes).unwrap_or_default();
    let keepdims = get_attr_int(node, Attr::KeepDims).unwrap_or(1) != 0;
    let out = reduce_op(input, &axes, keepdims, |vals: &[f32]| vals.iter().product())?;
    env.insert(node.outputs[0].clone(), out);
    Ok(())
}

pub(super) fn exec_reduce_l2(node: &OnnxNode, env: &mut TensorEnv) -> Result<(), OnnxError> {
    let input = get_tensor(env, &node.name, &node.inputs[0])?;
    let keepdims = get_attr_int(node, Attr::KeepDims).unwrap_or(1) != 0;
    let axes = get_attr_ints(node, Attr::Axes).unwrap_or_default();
    let out = reduce_op(input, &axes, keepdims, |vals| {
        vals.iter().map(|v| v * v).sum::<f32>().sqrt()
    })?;
    env.insert(node.outputs[0].clone(), out);
    Ok(())
}

pub(super) fn exec_reduce_l1(node: &OnnxNode, env: &mut TensorEnv) -> Result<(), OnnxError> {
    let input = get_tensor(env, &node.name, &node.inputs[0])?;
    let keepdims = get_attr_int(node, Attr::KeepDims).unwrap_or(1) != 0;
    let axes = get_attr_ints(node, Attr::Axes).unwrap_or_default();
    let out = reduce_op(input, &axes, keepdims, |vals| {
        vals.iter().map(|v| v.abs()).sum::<f32>()
    })?;
    env.insert(node.outputs[0].clone(), out);
    Ok(())
}

fn reduce_op<F>(input: &Tensor, axes: &[i64], keepdims: bool, op: F) -> Result<Tensor, OnnxError>
where
    F: Fn(&[f32]) -> f32,
{
    let shape = input.shape();
    let rank = shape.len();
    if axes.is_empty() {
        let val = op(input.data());
        let out_shape = if keepdims { vec![1; rank] } else { vec![1] };
        return Tensor::from_vec(out_shape, vec![val]).map_err(|e| OnnxError::DecodeFailed {
            message: e.to_string(),
        });
    }

    let norm_axes: Vec<usize> = axes
        .iter()
        .map(|&a| {
            if a < 0 {
                (rank as i64 + a) as usize
            } else {
                a as usize
            }
        })
        .collect();

    let mut out_shape = Vec::new();
    for (i, &d) in shape.iter().enumerate() {
        if norm_axes.contains(&i) {
            if keepdims {
                out_shape.push(1);
            }
        } else {
            out_shape.push(d);
        }
    }
    if out_shape.is_empty() {
        out_shape.push(1);
    }

    let out_size: usize = out_shape.iter().product();
    let reduce_size: usize = norm_axes.iter().map(|&a| shape[a]).product();
    let data = input.data();
    let strides = compute_strides(shape);
    let mut out_data = Vec::with_capacity(out_size);

    let out_strides = compute_strides(&out_shape);
    for out_idx in 0..out_size {
        let mut out_coords = vec![0usize; out_shape.len()];
        let mut rem = out_idx;
        for d in 0..out_shape.len() {
            out_coords[d] = rem / out_strides[d];
            rem %= out_strides[d];
        }

        let mut vals = Vec::with_capacity(reduce_size);
        collect_reduce_vals(
            data,
            shape,
            &strides,
            &norm_axes,
            &out_coords,
            keepdims,
            0,
            0,
            &mut vals,
        );
        out_data.push(op(&vals));
    }

    Tensor::from_vec(out_shape, out_data).map_err(|e| OnnxError::DecodeFailed {
        message: e.to_string(),
    })
}

#[allow(clippy::too_many_arguments)]
fn collect_reduce_vals(
    data: &[f32],
    shape: &[usize],
    strides: &[usize],
    reduce_axes: &[usize],
    out_coords: &[usize],
    keepdims: bool,
    dim: usize,
    offset: usize,
    vals: &mut Vec<f32>,
) {
    if dim == shape.len() {
        vals.push(data[offset]);
        return;
    }
    if reduce_axes.contains(&dim) {
        for i in 0..shape[dim] {
            collect_reduce_vals(
                data,
                shape,
                strides,
                reduce_axes,
                out_coords,
                keepdims,
                dim + 1,
                offset + i * strides[dim],
                vals,
            );
        }
    } else {
        let mut oc_idx = 0;
        for d in 0..dim {
            if !reduce_axes.contains(&d) || keepdims {
                oc_idx += 1;
            }
        }
        let coord = if oc_idx < out_coords.len() {
            out_coords[oc_idx]
        } else {
            0
        };
        collect_reduce_vals(
            data,
            shape,
            strides,
            reduce_axes,
            out_coords,
            keepdims,
            dim + 1,
            offset + coord * strides[dim],
            vals,
        );
    }
}

pub(super) fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

pub(super) fn exec_argmax(node: &OnnxNode, env: &mut TensorEnv) -> Result<(), OnnxError> {
    let input = get_tensor(env, &node.name, &node.inputs[0])?;
    let axis = get_attr_int(node, Attr::Axis).unwrap_or(0);
    let keepdims = get_attr_int(node, Attr::KeepDims).unwrap_or(1) != 0;
    let shape = input.shape();
    let rank = shape.len() as i64;
    let ax = if axis < 0 {
        (rank + axis) as usize
    } else {
        axis as usize
    };
    let data = input.data();

    let outer: usize = shape[..ax].iter().product();
    let dim = shape[ax];
    let inner: usize = shape[ax + 1..].iter().product();

    let mut result = Vec::with_capacity(outer * inner);
    for o in 0..outer {
        for i in 0..inner {
            let mut best_idx = 0usize;
            let mut best_val = f32::NEG_INFINITY;
            for d in 0..dim {
                let v = data[(o * dim + d) * inner + i];
                if v > best_val {
                    best_val = v;
                    best_idx = d;
                }
            }
            result.push(best_idx as f32);
        }
    }

    let mut out_shape: Vec<usize> = shape.to_vec();
    if keepdims {
        out_shape[ax] = 1;
    } else {
        out_shape.remove(ax);
    }
    let t = Tensor::from_vec(out_shape, result).map_err(|e| OnnxError::DecodeFailed {
        message: e.to_string(),
    })?;
    env.insert(node.outputs[0].clone(), t);
    Ok(())
}

pub(super) fn exec_argmin(node: &OnnxNode, env: &mut TensorEnv) -> Result<(), OnnxError> {
    let input = get_tensor(env, &node.name, &node.inputs[0])?;
    let axis = get_attr_int(node, Attr::Axis).unwrap_or(0);
    let keepdims = get_attr_int(node, Attr::KeepDims).unwrap_or(1) != 0;
    let shape = input.shape();
    let rank = shape.len() as i64;
    let ax = if axis < 0 {
        (rank + axis) as usize
    } else {
        axis as usize
    };
    let data = input.data();

    let outer: usize = shape[..ax].iter().product();
    let dim = shape[ax];
    let inner: usize = shape[ax + 1..].iter().product();

    let mut result = Vec::with_capacity(outer * inner);
    for o in 0..outer {
        for i in 0..inner {
            let mut best_idx = 0usize;
            let mut best_val = f32::INFINITY;
            for d in 0..dim {
                let v = data[(o * dim + d) * inner + i];
                if v < best_val {
                    best_val = v;
                    best_idx = d;
                }
            }
            result.push(best_idx as f32);
        }
    }

    let mut out_shape: Vec<usize> = shape.to_vec();
    if keepdims {
        out_shape[ax] = 1;
    } else {
        out_shape.remove(ax);
    }
    let t = Tensor::from_vec(out_shape, result).map_err(|e| OnnxError::DecodeFailed {
        message: e.to_string(),
    })?;
    env.insert(node.outputs[0].clone(), t);
    Ok(())
}

pub(super) fn exec_topk(node: &OnnxNode, env: &mut TensorEnv) -> Result<(), OnnxError> {
    let input = get_tensor(env, &node.name, &node.inputs[0])?;
    let k_tensor = get_tensor(env, &node.name, &node.inputs[1])?;
    let k = k_tensor.data()[0] as usize;
    let axis = get_attr_int(node, Attr::Axis).unwrap_or(-1);
    let largest = get_attr_int(node, Attr::Largest).unwrap_or(1) != 0;
    let shape = input.shape();
    let rank = shape.len() as i64;
    let ax = if axis < 0 {
        (rank + axis) as usize
    } else {
        axis as usize
    };
    let data = input.data();

    let outer: usize = shape[..ax].iter().product();
    let dim = shape[ax];
    let inner: usize = shape[ax + 1..].iter().product();

    let mut values = Vec::with_capacity(outer * k * inner);
    let mut indices = Vec::with_capacity(outer * k * inner);

    for o in 0..outer {
        for i in 0..inner {
            let mut pairs: Vec<(f32, usize)> = (0..dim)
                .map(|d| (data[(o * dim + d) * inner + i], d))
                .collect();
            if largest {
                pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
            } else {
                pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            }
            for pair in pairs.iter().take(k) {
                values.push(pair.0);
                indices.push(pair.1 as f32);
            }
        }
    }

    let mut out_shape = shape.to_vec();
    out_shape[ax] = k;
    let val_tensor =
        Tensor::from_vec(out_shape.clone(), values).map_err(|e| OnnxError::DecodeFailed {
            message: e.to_string(),
        })?;
    let idx_tensor = Tensor::from_vec(out_shape, indices).map_err(|e| OnnxError::DecodeFailed {
        message: e.to_string(),
    })?;
    env.insert(node.outputs[0].clone(), val_tensor);
    if node.outputs.len() > 1 {
        env.insert(node.outputs[1].clone(), idx_tensor);
    }
    Ok(())
}
