use super::*;

pub(super) fn exec_gather(node: &OnnxNode, env: &mut TensorEnv) -> Result<(), OnnxError> {
    let input = get_tensor(env, &node.name, &node.inputs[0])?;
    let indices = get_tensor(env, &node.name, &node.inputs[1])?;
    let axis_raw = get_attr_int(node, "axis").unwrap_or(0);
    let rank = input.rank() as i64;
    let axis = if axis_raw < 0 {
        (rank + axis_raw) as usize
    } else {
        axis_raw as usize
    };
    if axis >= input.rank() {
        return Err(OnnxError::DecodeFailed {
            message: format!(
                "Gather axis {axis} out of range for rank-{} input",
                input.rank()
            ),
        });
    }

    // Fast path: axis=0, rank-1 input — straight 1-D embedding lookup
    // (the LLM token-embedding hot path; keep the bulk-copyless impl).
    if axis == 0 && input.rank() == 1 {
        let data = input.data();
        let idx_data = indices.data();
        let src_dim = data.len() as i64;
        let out_data: Vec<f32> = idx_data
            .iter()
            .map(|&i| {
                let mut k = i as i64;
                if k < 0 {
                    k += src_dim;
                }
                data[k as usize]
            })
            .collect();
        let out = Tensor::from_vec(indices.shape().to_vec(), out_data).map_err(|e| {
            OnnxError::DecodeFailed {
                message: e.to_string(),
            }
        })?;
        env.insert(node.outputs[0].clone(), out);
        return Ok(());
    }

    // General path. ONNX Gather:
    //   out[i_0..i_{a-1}, j_0..j_{q-1}, i_{a+1}..i_{r-1}]
    //     = input[i_0..i_{a-1}, indices[j], i_{a+1}..]
    // where j ranges over indices.shape and a == axis. Output shape is
    // input.shape[..axis] ++ indices.shape ++ input.shape[axis+1..].
    let in_shape = input.shape();
    let idx_shape = indices.shape();
    let mut out_shape: Vec<usize> = Vec::with_capacity(in_shape.len() + idx_shape.len() - 1);
    out_shape.extend_from_slice(&in_shape[..axis]);
    out_shape.extend_from_slice(idx_shape);
    out_shape.extend_from_slice(&in_shape[axis + 1..]);

    let outer: usize = in_shape[..axis].iter().product();
    let src_dim = in_shape[axis];
    let inner: usize = in_shape[axis + 1..].iter().product();
    let idx_count = indices.len().max(1);

    let src = input.data();
    let idx_data = indices.data();
    let mut out = vec![0.0_f32; outer * idx_count * inner];

    let src_outer_stride = src_dim * inner;
    for o in 0..outer {
        let src_o = o * src_outer_stride;
        let dst_o = o * idx_count * inner;
        for (q, &raw) in idx_data.iter().enumerate() {
            let mut k = raw as i64;
            if k < 0 {
                k += src_dim as i64;
            }
            if k < 0 || k as usize >= src_dim {
                return Err(OnnxError::DecodeFailed {
                    message: format!(
                        "Gather index {raw} out of bounds for axis {axis} (dim {src_dim})"
                    ),
                });
            }
            let src_off = src_o + (k as usize) * inner;
            let dst_off = dst_o + q * inner;
            out[dst_off..dst_off + inner].copy_from_slice(&src[src_off..src_off + inner]);
        }
    }

    let result = Tensor::from_vec(out_shape, out).map_err(|e| OnnxError::DecodeFailed {
        message: e.to_string(),
    })?;
    env.insert(node.outputs[0].clone(), result);
    Ok(())
}

pub(super) fn exec_gather_elements(node: &OnnxNode, env: &mut TensorEnv) -> Result<(), OnnxError> {
    let input = get_tensor(env, &node.name, &node.inputs[0])?;
    let indices = get_tensor(env, &node.name, &node.inputs[1])?;
    let axis = get_attr_int(node, "axis").unwrap_or(0);
    let shape = input.shape();
    let rank = shape.len() as i64;
    let ax = if axis < 0 {
        (rank + axis) as usize
    } else {
        axis as usize
    };

    let idx_shape = indices.shape();
    let outer: usize = idx_shape[..ax].iter().product();
    let dim = idx_shape[ax];
    let inner: usize = idx_shape[ax + 1..].iter().product();
    let src_dim = shape[ax];
    let src_inner: usize = shape[ax + 1..].iter().product();

    let mut out = vec![0.0f32; indices.len()];
    let idx_data = indices.data();
    let src_data = input.data();

    for o in 0..outer {
        for d in 0..dim {
            for i in 0..inner {
                let pos = (o * dim + d) * inner + i;
                let src_d = idx_data[pos] as i64;
                let src_d = if src_d < 0 {
                    (src_dim as i64 + src_d) as usize
                } else {
                    src_d as usize
                };
                if src_d < src_dim {
                    out[pos] = src_data[(o * src_dim + src_d) * src_inner + i];
                }
            }
        }
    }

    let result =
        Tensor::from_vec(idx_shape.to_vec(), out).map_err(|e| OnnxError::DecodeFailed {
            message: e.to_string(),
        })?;
    env.insert(node.outputs[0].clone(), result);
    Ok(())
}

pub(super) fn exec_scatter_elements(node: &OnnxNode, env: &mut TensorEnv) -> Result<(), OnnxError> {
    let input = get_tensor(env, &node.name, &node.inputs[0])?;
    let indices = get_tensor(env, &node.name, &node.inputs[1])?;
    let updates = get_tensor(env, &node.name, &node.inputs[2])?;
    let axis = get_attr_int(node, "axis").unwrap_or(0);
    let shape = input.shape();
    let rank = shape.len() as i64;
    let ax = if axis < 0 {
        (rank + axis) as usize
    } else {
        axis as usize
    };

    let mut out = input.data().to_vec();
    let idx_shape = indices.shape();
    let outer: usize = idx_shape[..ax].iter().product();
    let dim = idx_shape[ax];
    let inner: usize = idx_shape[ax + 1..].iter().product();
    let dst_dim = shape[ax];
    let dst_inner: usize = shape[ax + 1..].iter().product();

    for o in 0..outer {
        for d in 0..dim {
            for i in 0..inner {
                let pos = (o * dim + d) * inner + i;
                let dst_d = indices.data()[pos] as i64;
                let dst_d = if dst_d < 0 {
                    (dst_dim as i64 + dst_d) as usize
                } else {
                    dst_d as usize
                };
                if dst_d < dst_dim {
                    let out_pos = (o * dst_dim + dst_d) * dst_inner + i;
                    if out_pos < out.len() {
                        out[out_pos] = updates.data()[pos];
                    }
                }
            }
        }
    }

    let result = Tensor::from_vec(shape.to_vec(), out).map_err(|e| OnnxError::DecodeFailed {
        message: e.to_string(),
    })?;
    env.insert(node.outputs[0].clone(), result);
    Ok(())
}

pub(super) fn exec_scatter_nd(node: &OnnxNode, env: &mut TensorEnv) -> Result<(), OnnxError> {
    let data = get_tensor(env, &node.name, &node.inputs[0])?;
    let indices = get_tensor(env, &node.name, &node.inputs[1])?;
    let updates = get_tensor(env, &node.name, &node.inputs[2])?;
    let shape = data.shape();
    let idx_shape = indices.shape();
    let idx_data = indices.data();
    let upd_data = updates.data();

    let last_idx_dim = *idx_shape.last().unwrap_or(&1);
    let num_updates: usize = idx_shape[..idx_shape.len() - 1].iter().product();
    let slice_size: usize = shape[last_idx_dim..].iter().product();

    let mut out = data.data().to_vec();
    for u in 0..num_updates {
        let mut flat_offset = 0usize;
        let mut stride = 1usize;
        for d in (0..last_idx_dim).rev() {
            let idx_val = idx_data[u * last_idx_dim + d] as usize;
            flat_offset += idx_val * stride;
            stride *= shape[d];
        }
        for s in 0..slice_size {
            out[flat_offset + s] = upd_data[u * slice_size + s];
        }
    }

    let t = Tensor::from_vec(shape.to_vec(), out).map_err(|e| OnnxError::DecodeFailed {
        message: e.to_string(),
    })?;
    env.insert(node.outputs[0].clone(), t);
    Ok(())
}

pub(super) fn exec_gather_nd(node: &OnnxNode, env: &mut TensorEnv) -> Result<(), OnnxError> {
    let data = get_tensor(env, &node.name, &node.inputs[0])?;
    let indices = get_tensor(env, &node.name, &node.inputs[1])?;
    let batch_dims = get_attr_int(node, "batch_dims").unwrap_or(0) as usize;
    let shape = data.shape();
    let idx_shape = indices.shape();
    let idx_data = indices.data();
    let src = data.data();

    let last_idx_dim = *idx_shape.last().unwrap_or(&1);

    let batch_size: usize = shape[..batch_dims].iter().product();
    let data_batch_stride: usize = shape[batch_dims..].iter().product();
    let idx_batch_stride: usize = idx_shape[batch_dims..].iter().product();
    let num_lookups = idx_batch_stride / last_idx_dim;
    let slice_size: usize = shape[batch_dims + last_idx_dim..].iter().product();

    let mut result = Vec::with_capacity(batch_size * num_lookups * slice_size);
    for b in 0..batch_size {
        let data_base = b * data_batch_stride;
        let idx_base = b * idx_batch_stride;
        for l in 0..num_lookups {
            let mut flat_offset = 0usize;
            let mut stride = 1usize;
            for d in (0..last_idx_dim).rev() {
                let idx_val = idx_data[idx_base + l * last_idx_dim + d] as usize;
                flat_offset += idx_val * stride;
                stride *= shape[batch_dims + d];
            }
            for s in 0..slice_size {
                result.push(src[data_base + flat_offset + s]);
            }
        }
    }

    let mut out_shape = Vec::new();
    out_shape.extend_from_slice(&shape[..batch_dims]);
    out_shape.extend_from_slice(&idx_shape[batch_dims..idx_shape.len() - 1]);
    if slice_size > 1 {
        out_shape.extend_from_slice(&shape[batch_dims + last_idx_dim..]);
    }
    if out_shape.is_empty() {
        out_shape.push(1);
    }

    let t = Tensor::from_vec(out_shape, result).map_err(|e| OnnxError::DecodeFailed {
        message: e.to_string(),
    })?;
    env.insert(node.outputs[0].clone(), t);
    Ok(())
}
