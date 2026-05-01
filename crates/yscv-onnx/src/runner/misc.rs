use super::*;

pub(super) fn exec_constant(node: &OnnxNode, env: &mut TensorEnv) -> Result<(), OnnxError> {
    if let Some(OnnxAttribute::Tensor(t)) = node.attributes.get("value") {
        env.insert(node.outputs[0].clone(), t.clone());
    } else if let Some(OnnxAttribute::Float(v)) = node.attributes.get("value_float") {
        env.insert(node.outputs[0].clone(), Tensor::scalar(*v));
    } else if let Some(OnnxAttribute::Int(v)) = node.attributes.get("value_int") {
        env.insert(node.outputs[0].clone(), Tensor::scalar(*v as f32));
    } else if let Some(OnnxAttribute::Floats(v)) = node.attributes.get("value_floats") {
        let t =
            Tensor::from_vec(vec![v.len()], v.clone()).map_err(|e| OnnxError::DecodeFailed {
                message: e.to_string(),
            })?;
        env.insert(node.outputs[0].clone(), t);
    } else if let Some(OnnxAttribute::Ints(v)) = node.attributes.get("value_ints") {
        let data: Vec<f32> = v.iter().map(|&i| i as f32).collect();
        let t = Tensor::from_vec(vec![data.len()], data).map_err(|e| OnnxError::DecodeFailed {
            message: e.to_string(),
        })?;
        env.insert(node.outputs[0].clone(), t);
    } else {
        env.insert(node.outputs[0].clone(), Tensor::scalar(0.0));
    }
    Ok(())
}

pub(super) fn exec_identity(node: &OnnxNode, env: &mut TensorEnv) -> Result<(), OnnxError> {
    // Zero-copy: alias the output to the input (ONNX outputs are write-once).
    // Fallback to clone when alias can't resolve the input (e.g., runtime inputs
    // not yet materialized into slots).
    if let Some(input) = env.get(&node.inputs[0]) {
        let t = input.clone();
        env.insert(node.outputs[0].clone(), t);
    } else {
        env.alias(&node.outputs[0], &node.inputs[0]);
    }
    Ok(())
}

pub(super) fn exec_quantize_linear(node: &OnnxNode, env: &mut TensorEnv) -> Result<(), OnnxError> {
    let input = get_tensor(env, &node.name, &node.inputs[0])?;
    let scale = get_tensor(env, &node.name, &node.inputs[1])?;
    let zero_point = if node.inputs.len() > 2 && !node.inputs[2].is_empty() {
        Some(get_tensor(env, &node.name, &node.inputs[2])?)
    } else {
        None
    };

    let s = scale.data()[0];
    let zp = zero_point.map_or(0.0f32, |t| t.data()[0]);
    if !node
        .outputs
        .first()
        .map(|name| name.contains("__qlinear_x_q"))
        .unwrap_or(false)
    {
        let mut data = vec![0.0_f32; input.data().len()];
        yscv_kernels::quantize_linear_f32_to_f32_i8_dispatch(input.data(), s, zp, &mut data);
        let out = Tensor::from_vec(input.shape().to_vec(), data).map_err(|e| {
            OnnxError::DecodeFailed {
                message: e.to_string(),
            }
        })?;
        env.insert(node.outputs[0].clone(), out);
        return Ok(());
    }
    let mut data = vec![0_i8; input.data().len()];
    yscv_kernels::quantize_linear_f32_to_i8_dispatch(input.data(), s, zp, &mut data);
    env.insert_quant_i8(
        node.outputs[0].clone(),
        QuantTensor {
            data,
            shape: input.shape().to_vec(),
            scale: s,
            zero_point: zp,
            nhwc: env.is_nhwc(&node.inputs[0]),
        },
    );
    Ok(())
}

pub(super) fn exec_dequantize_linear(
    node: &OnnxNode,
    env: &mut TensorEnv,
) -> Result<(), OnnxError> {
    let scale = get_tensor(env, &node.name, &node.inputs[1])?;
    let zero_point = if node.inputs.len() > 2 && !node.inputs[2].is_empty() {
        Some(get_tensor(env, &node.name, &node.inputs[2])?)
    } else {
        None
    };

    if let Some(input) = env.get_quant_i8(&node.inputs[0]) {
        let in_shape = input.shape.clone();
        let input_nhwc = input.nhwc;
        let scale_data = scale.data();
        let data: Vec<f32> = if scale_data.len() == 1 {
            let s = scale_data[0];
            let zp = zero_point.map_or(0.0f32, |t| t.data()[0]);
            input.data.iter().map(|&v| ((v as f32) - zp) * s).collect()
        } else {
            let axis = match node.attributes.get("axis") {
                Some(crate::loader::OnnxAttribute::Int(a)) => *a,
                _ => 1,
            };
            let axis = if axis < 0 {
                (in_shape.len() as i64 + axis) as usize
            } else {
                axis as usize
            };
            let chan = in_shape[axis];
            if scale_data.len() != chan {
                return Err(OnnxError::DecodeFailed {
                    message: format!(
                        "DequantizeLinear: scale len={} but axis-{axis} dim={chan}",
                        scale_data.len()
                    ),
                });
            }
            let zp_storage: Vec<f32>;
            let zp_data: &[f32] = match zero_point {
                Some(t) => t.data(),
                None => {
                    zp_storage = vec![0.0_f32; chan];
                    &zp_storage
                }
            };
            let outer = in_shape[..axis].iter().product::<usize>();
            let inner = in_shape[axis + 1..].iter().product::<usize>();
            let mut out = vec![0.0_f32; input.data.len()];
            for o in 0..outer {
                for c in 0..chan {
                    let s = scale_data[c];
                    let zp = zp_data[c];
                    let base = (o * chan + c) * inner;
                    for i in 0..inner {
                        out[base + i] = ((input.data[base + i] as f32) - zp) * s;
                    }
                }
            }
            out
        };
        let out = Tensor::from_vec(in_shape, data).map_err(|e| OnnxError::DecodeFailed {
            message: e.to_string(),
        })?;
        env.insert(node.outputs[0].clone(), out);
        if input_nhwc {
            env.mark_nhwc(&node.outputs[0]);
        }
        return Ok(());
    }

    let input = get_tensor(env, &node.name, &node.inputs[0])?;
    let in_shape = input.shape();
    let scale_data = scale.data();
    let data: Vec<f32> = if scale_data.len() == 1 {
        // Per-tensor.
        let s = scale_data[0];
        let zp = zero_point.map_or(0.0f32, |t| t.data()[0]);
        input.data().iter().map(|&v| (v - zp) * s).collect()
    } else {
        // Per-channel along `axis` attribute (default 1 per ONNX spec, but
        // most exporters write the attribute explicitly).
        let axis = match node.attributes.get("axis") {
            Some(crate::loader::OnnxAttribute::Int(a)) => *a,
            _ => 1,
        };
        let axis = if axis < 0 {
            (in_shape.len() as i64 + axis) as usize
        } else {
            axis as usize
        };
        let chan = in_shape[axis];
        if scale_data.len() != chan {
            return Err(OnnxError::DecodeFailed {
                message: format!(
                    "DequantizeLinear: scale len={} but axis-{axis} dim={chan}",
                    scale_data.len()
                ),
            });
        }
        let zp_storage: Vec<f32>;
        let zp_data: &[f32] = match zero_point {
            Some(t) => t.data(),
            None => {
                zp_storage = vec![0.0_f32; chan];
                &zp_storage
            }
        };
        let outer = in_shape[..axis].iter().product::<usize>();
        let inner = in_shape[axis + 1..].iter().product::<usize>();
        let in_data = input.data();
        let mut out = vec![0.0_f32; in_data.len()];
        for o in 0..outer {
            for c in 0..chan {
                let s = scale_data[c];
                let zp = zp_data[c];
                let base = (o * chan + c) * inner;
                for i in 0..inner {
                    out[base + i] = (in_data[base + i] - zp) * s;
                }
            }
        }
        out
    };
    let out = Tensor::from_vec(in_shape.to_vec(), data).map_err(|e| OnnxError::DecodeFailed {
        message: e.to_string(),
    })?;
    env.insert(node.outputs[0].clone(), out);
    Ok(())
}

pub(super) fn exec_dynamic_quantize_linear(
    node: &OnnxNode,
    env: &mut TensorEnv,
) -> Result<(), OnnxError> {
    let input = get_tensor(env, &node.name, &node.inputs[0])?;
    let data = input.data();

    let mut min_val = f32::MAX;
    let mut max_val = f32::MIN;
    for &v in data {
        if v < min_val {
            min_val = v;
        }
        if v > max_val {
            max_val = v;
        }
    }
    let qmin = 0.0f32;
    let qmax = 255.0f32;
    min_val = min_val.min(0.0);
    max_val = max_val.max(0.0);
    let scale = if (max_val - min_val).abs() < f32::EPSILON {
        1.0
    } else {
        (max_val - min_val) / (qmax - qmin)
    };
    let zero_point = (qmin - min_val / scale).round().clamp(qmin, qmax);

    let quant: Vec<f32> = data
        .iter()
        .map(|&v| (v / scale + zero_point).round().clamp(qmin, qmax))
        .collect();

    let y =
        Tensor::from_vec(input.shape().to_vec(), quant).map_err(|e| OnnxError::DecodeFailed {
            message: e.to_string(),
        })?;
    let y_scale = Tensor::scalar(scale);
    let y_zp = Tensor::scalar(zero_point);

    env.insert(node.outputs[0].clone(), y);
    if node.outputs.len() > 1 {
        env.insert(node.outputs[1].clone(), y_scale);
    }
    if node.outputs.len() > 2 {
        env.insert(node.outputs[2].clone(), y_zp);
    }
    Ok(())
}

pub(super) fn exec_resize(node: &OnnxNode, env: &mut TensorEnv) -> Result<(), OnnxError> {
    let input_is_nhwc = env.is_nhwc(&node.inputs[0]);
    let input = get_tensor(env, &node.name, &node.inputs[0])?;
    let shape = input.shape();
    if shape.len() != 4 {
        return Err(OnnxError::DecodeFailed {
            message: "Resize expects 4D input".into(),
        });
    }

    // Determine spatial dims based on layout
    let (n, ih, iw, c) = if input_is_nhwc {
        (shape[0], shape[1], shape[2], shape[3])
    } else {
        (shape[0], shape[2], shape[3], shape[1])
    };

    // ONNX sizes/scales are always in NCHW order: [N, C, H, W]
    let (oh, ow) = if node.inputs.len() > 3 && !node.inputs[3].is_empty() {
        let sizes = get_tensor(env, &node.name, &node.inputs[3])?;
        let sd = sizes.data();
        (sd[2] as usize, sd[3] as usize)
    } else if node.inputs.len() > 2 && !node.inputs[2].is_empty() {
        let scales = get_tensor(env, &node.name, &node.inputs[2])?;
        let sd = scales.data();
        ((ih as f32 * sd[2]) as usize, (iw as f32 * sd[3]) as usize)
    } else {
        (ih, iw)
    };

    let data = input.data();
    let sy = ih as f32 / oh as f32;
    let sx = iw as f32 / ow as f32;

    if input_is_nhwc {
        // Nearest-neighbor resize in NHWC — use row-based memcpy for integer 2× upscale
        let mut out = vec![0.0f32; n * oh * ow * c];
        for b in 0..n {
            for y in 0..oh {
                let src_y = (((y as f32 + 0.5) * sy).floor() as usize).min(ih - 1);
                for x in 0..ow {
                    let src_x = (((x as f32 + 0.5) * sx).floor() as usize).min(iw - 1);
                    let src_off = ((b * ih + src_y) * iw + src_x) * c;
                    let dst_off = ((b * oh + y) * ow + x) * c;
                    out[dst_off..dst_off + c].copy_from_slice(&data[src_off..src_off + c]);
                }
            }
        }
        let out_t =
            Tensor::from_vec(vec![n, oh, ow, c], out).map_err(|e| OnnxError::DecodeFailed {
                message: e.to_string(),
            })?;
        env.insert(node.outputs[0].clone(), out_t);
        env.mark_nhwc(&node.outputs[0]);
    } else {
        // NCHW path
        let mut out = Vec::with_capacity(n * c * oh * ow);
        for b in 0..n {
            for ch in 0..c {
                for y in 0..oh {
                    let src_y = (((y as f32 + 0.5) * sy).floor() as usize).min(ih - 1);
                    for x in 0..ow {
                        let src_x = (((x as f32 + 0.5) * sx).floor() as usize).min(iw - 1);
                        out.push(data[((b * c + ch) * ih + src_y) * iw + src_x]);
                    }
                }
            }
        }
        let out_t =
            Tensor::from_vec(vec![n, c, oh, ow], out).map_err(|e| OnnxError::DecodeFailed {
                message: e.to_string(),
            })?;
        env.insert(node.outputs[0].clone(), out_t);
    }
    Ok(())
}

pub(super) fn exec_onehot(node: &OnnxNode, env: &mut TensorEnv) -> Result<(), OnnxError> {
    let indices = get_tensor(env, &node.name, &node.inputs[0])?;
    let depth_t = get_tensor(env, &node.name, &node.inputs[1])?;
    let values_t = get_tensor(env, &node.name, &node.inputs[2])?;
    let depth = depth_t.data()[0] as usize;
    let off_val = values_t.data()[0];
    let on_val = values_t.data()[1];

    let n = indices.len();
    let mut out = vec![off_val; n * depth];
    for (i, &idx_f) in indices.data().iter().enumerate() {
        let idx = idx_f as usize;
        if idx < depth {
            out[i * depth + idx] = on_val;
        }
    }

    let mut out_shape = indices.shape().to_vec();
    out_shape.push(depth);
    let result = Tensor::from_vec(out_shape, out).map_err(|e| OnnxError::DecodeFailed {
        message: e.to_string(),
    })?;
    env.insert(node.outputs[0].clone(), result);
    Ok(())
}

pub(super) fn exec_range(node: &OnnxNode, env: &mut TensorEnv) -> Result<(), OnnxError> {
    let start = get_tensor(env, &node.name, &node.inputs[0])?.data()[0];
    let limit = get_tensor(env, &node.name, &node.inputs[1])?.data()[0];
    let delta = get_tensor(env, &node.name, &node.inputs[2])?.data()[0];

    let mut vals = Vec::new();
    let mut v = start;
    if delta > 0.0 {
        while v < limit {
            vals.push(v);
            v += delta;
        }
    } else if delta < 0.0 {
        while v > limit {
            vals.push(v);
            v += delta;
        }
    }

    let len = vals.len();
    let out = Tensor::from_vec(
        vec![len.max(1)],
        if vals.is_empty() { vec![0.0] } else { vals },
    )
    .map_err(|e| OnnxError::DecodeFailed {
        message: e.to_string(),
    })?;
    env.insert(node.outputs[0].clone(), out);
    Ok(())
}

pub(super) fn exec_nonzero(node: &OnnxNode, env: &mut TensorEnv) -> Result<(), OnnxError> {
    let input = get_tensor(env, &node.name, &node.inputs[0])?;
    let shape = input.shape();
    let rank = shape.len();

    let mut coords_list: Vec<Vec<f32>> = vec![Vec::new(); rank];
    let mut coords = vec![0usize; rank];

    for &v in input.data() {
        if v != 0.0 {
            for (r, c) in coords.iter().enumerate() {
                coords_list[r].push(*c as f32);
            }
        }
        increment_multi_coords(&mut coords, shape);
    }

    let n = coords_list.first().map_or(0, |v| v.len()).max(1);
    let mut flat = Vec::with_capacity(rank * n);
    for row in &coords_list {
        flat.extend_from_slice(row);
        flat.extend(std::iter::repeat_n(0.0f32, n - row.len()));
    }

    let out = Tensor::from_vec(vec![rank, n], flat).map_err(|e| OnnxError::DecodeFailed {
        message: e.to_string(),
    })?;
    env.insert(node.outputs[0].clone(), out);
    Ok(())
}

fn increment_multi_coords(coords: &mut [usize], shape: &[usize]) {
    for i in (0..coords.len()).rev() {
        coords[i] += 1;
        if coords[i] < shape[i] {
            return;
        }
        coords[i] = 0;
    }
}

pub(super) fn exec_compress(node: &OnnxNode, env: &mut TensorEnv) -> Result<(), OnnxError> {
    let input = get_tensor(env, &node.name, &node.inputs[0])?;
    let condition = get_tensor(env, &node.name, &node.inputs[1])?;
    let axis = get_attr_int(node, "axis");
    let cond = condition.data();

    if let Some(ax) = axis {
        let shape = input.shape();
        let rank = shape.len() as i64;
        let ax = if ax < 0 {
            (rank + ax) as usize
        } else {
            ax as usize
        };
        let data = input.data();
        let outer: usize = shape[..ax].iter().product();
        let dim = shape[ax];
        let inner: usize = shape[ax + 1..].iter().product();

        let selected: Vec<usize> = (0..dim)
            .filter(|&d| d < cond.len() && cond[d] != 0.0)
            .collect();
        let mut result = Vec::new();
        for o in 0..outer {
            for &d in &selected {
                for i in 0..inner {
                    result.push(data[(o * dim + d) * inner + i]);
                }
            }
        }
        let mut out_shape = shape.to_vec();
        out_shape[ax] = selected.len();
        let t = Tensor::from_vec(out_shape, result).map_err(|e| OnnxError::DecodeFailed {
            message: e.to_string(),
        })?;
        env.insert(node.outputs[0].clone(), t);
    } else {
        let flat = input.data();
        let result: Vec<f32> = flat
            .iter()
            .zip(cond.iter().cycle())
            .filter(|&(_, c)| *c != 0.0)
            .map(|(&v, _)| v)
            .collect();
        let len = result.len();
        let t = Tensor::from_vec(vec![len], result).map_err(|e| OnnxError::DecodeFailed {
            message: e.to_string(),
        })?;
        env.insert(node.outputs[0].clone(), t);
    }
    Ok(())
}

pub(super) fn exec_cumsum(node: &OnnxNode, env: &mut TensorEnv) -> Result<(), OnnxError> {
    let input = get_tensor(env, &node.name, &node.inputs[0])?;
    let axis_tensor = get_tensor(env, &node.name, &node.inputs[1])?;
    let axis = axis_tensor.data()[0] as i64;
    let shape = input.shape();
    let rank = shape.len() as i64;
    let ax = if axis < 0 {
        (rank + axis) as usize
    } else {
        axis as usize
    };

    let outer: usize = shape[..ax].iter().product();
    let dim = shape[ax];
    let inner: usize = shape[ax + 1..].iter().product();

    let mut out = input.data().to_vec();
    for o in 0..outer {
        for i in 0..inner {
            let mut acc = 0.0f32;
            for d in 0..dim {
                let idx = (o * dim + d) * inner + i;
                acc += out[idx];
                out[idx] = acc;
            }
        }
    }

    let result = Tensor::from_vec(shape.to_vec(), out).map_err(|e| OnnxError::DecodeFailed {
        message: e.to_string(),
    })?;
    env.insert(node.outputs[0].clone(), result);
    Ok(())
}

pub(super) fn exec_grid_sample(node: &OnnxNode, env: &mut TensorEnv) -> Result<(), OnnxError> {
    let input = get_tensor(env, &node.name, &node.inputs[0])?;
    let grid = get_tensor(env, &node.name, &node.inputs[1])?;
    let mode = get_attr_string(node, "mode").unwrap_or_else(|| "bilinear".to_string());
    let padding_mode = get_attr_string(node, "padding_mode").unwrap_or_else(|| "zeros".to_string());
    let align_corners = get_attr_int(node, "align_corners").unwrap_or(0) != 0;
    let shape = input.shape();
    let grid_shape = grid.shape();
    if shape.len() != 4 || grid_shape.len() != 4 {
        return Err(OnnxError::ShapeMismatch {
            detail: "GridSample requires rank-4 NCHW input and NHWC grid".into(),
        });
    }
    let (n, c, h_in, w_in) = (shape[0], shape[1], shape[2], shape[3]);
    let h_out = grid_shape[1];
    let w_out = grid_shape[2];
    let data = input.data();
    let grid_data = grid.data();

    let denormalize = |coord: f32, size: usize| -> f32 {
        if align_corners {
            ((coord + 1.0) / 2.0) * (size as f32 - 1.0)
        } else {
            ((coord + 1.0) * size as f32 - 1.0) / 2.0
        }
    };

    let sample_pixel = |nn: usize, cc: usize, yy: i64, xx: i64| -> f32 {
        match padding_mode.as_str() {
            "zeros" => {
                if yy < 0 || yy >= h_in as i64 || xx < 0 || xx >= w_in as i64 {
                    0.0
                } else {
                    data[((nn * c + cc) * h_in + yy as usize) * w_in + xx as usize]
                }
            }
            "border" => {
                let cy = yy.clamp(0, h_in as i64 - 1) as usize;
                let cx = xx.clamp(0, w_in as i64 - 1) as usize;
                data[((nn * c + cc) * h_in + cy) * w_in + cx]
            }
            _ => {
                let reflect = |v: i64, size: usize| -> usize {
                    let s = size as i64;
                    let mut vv = v;
                    if vv < 0 {
                        vv = -vv;
                    }
                    vv %= 2 * s;
                    if vv >= s {
                        vv = 2 * s - 1 - vv;
                    }
                    vv.clamp(0, s - 1) as usize
                };
                let cy = reflect(yy, h_in);
                let cx = reflect(xx, w_in);
                data[((nn * c + cc) * h_in + cy) * w_in + cx]
            }
        }
    };

    let total = n * c * h_out * w_out;
    let mut out = vec![0.0f32; total];

    for nn in 0..n {
        for oh in 0..h_out {
            for ow in 0..w_out {
                let grid_idx = ((nn * h_out + oh) * w_out + ow) * 2;
                let gx = grid_data[grid_idx];
                let gy = grid_data[grid_idx + 1];
                let x = denormalize(gx, w_in);
                let y = denormalize(gy, h_in);

                for cc in 0..c {
                    let val = match mode.as_str() {
                        "nearest" => {
                            let nx = x.round() as i64;
                            let ny = y.round() as i64;
                            sample_pixel(nn, cc, ny, nx)
                        }
                        _ => {
                            let x0 = x.floor() as i64;
                            let y0 = y.floor() as i64;
                            let x1 = x0 + 1;
                            let y1 = y0 + 1;
                            let wa = (x1 as f32 - x) * (y1 as f32 - y);
                            let wb = (x - x0 as f32) * (y1 as f32 - y);
                            let wc = (x1 as f32 - x) * (y - y0 as f32);
                            let wd = (x - x0 as f32) * (y - y0 as f32);
                            wa * sample_pixel(nn, cc, y0, x0)
                                + wb * sample_pixel(nn, cc, y0, x1)
                                + wc * sample_pixel(nn, cc, y1, x0)
                                + wd * sample_pixel(nn, cc, y1, x1)
                        }
                    };
                    out[((nn * c + cc) * h_out + oh) * w_out + ow] = val;
                }
            }
        }
    }

    let t =
        Tensor::from_vec(vec![n, c, h_out, w_out], out).map_err(|e| OnnxError::DecodeFailed {
            message: e.to_string(),
        })?;
    env.insert(node.outputs[0].clone(), t);
    Ok(())
}

pub(super) fn exec_roi_align(node: &OnnxNode, env: &mut TensorEnv) -> Result<(), OnnxError> {
    let input = get_tensor(env, &node.name, &node.inputs[0])?;
    let rois = get_tensor(env, &node.name, &node.inputs[1])?;
    let batch_indices = get_tensor(env, &node.name, &node.inputs[2])?;
    let output_height = get_attr_int(node, "output_height").unwrap_or(1) as usize;
    let output_width = get_attr_int(node, "output_width").unwrap_or(1) as usize;
    let sampling_ratio = get_attr_int(node, "sampling_ratio").unwrap_or(0) as usize;
    let spatial_scale = get_attr_float(node, "spatial_scale").unwrap_or(1.0);
    let mode = get_attr_string(node, "mode").unwrap_or_else(|| "avg".to_string());
    let shape = input.shape();
    if shape.len() != 4 {
        return Err(OnnxError::ShapeMismatch {
            detail: "RoiAlign requires rank-4 NCHW".into(),
        });
    }
    let (_n, c, h_in, w_in) = (shape[0], shape[1], shape[2], shape[3]);
    let num_rois = rois.shape()[0];
    let data = input.data();
    let rois_data = rois.data();
    let batch_data = batch_indices.data();

    let bilinear_interp = |nn: usize, cc: usize, y: f32, x: f32| -> f32 {
        let x0 = x.floor() as i64;
        let y0 = y.floor() as i64;
        let x1 = x0 + 1;
        let y1 = y0 + 1;
        let get = |yy: i64, xx: i64| -> f32 {
            if yy < 0 || yy >= h_in as i64 || xx < 0 || xx >= w_in as i64 {
                0.0
            } else {
                data[((nn * c + cc) * h_in + yy as usize) * w_in + xx as usize]
            }
        };
        let wa = (x1 as f32 - x) * (y1 as f32 - y);
        let wb = (x - x0 as f32) * (y1 as f32 - y);
        let wc = (x1 as f32 - x) * (y - y0 as f32);
        let wd = (x - x0 as f32) * (y - y0 as f32);
        wa * get(y0, x0) + wb * get(y0, x1) + wc * get(y1, x0) + wd * get(y1, x1)
    };

    let mut out = vec![0.0f32; num_rois * c * output_height * output_width];
    for r in 0..num_rois {
        let batch_idx = batch_data[r] as usize;
        let roi_x1 = rois_data[r * 4] * spatial_scale;
        let roi_y1 = rois_data[r * 4 + 1] * spatial_scale;
        let roi_x2 = rois_data[r * 4 + 2] * spatial_scale;
        let roi_y2 = rois_data[r * 4 + 3] * spatial_scale;
        let roi_h = (roi_y2 - roi_y1).max(1e-6);
        let roi_w = (roi_x2 - roi_x1).max(1e-6);
        let bin_h = roi_h / output_height as f32;
        let bin_w = roi_w / output_width as f32;
        let sample_h = if sampling_ratio > 0 {
            sampling_ratio
        } else {
            (bin_h.ceil() as usize).max(1)
        };
        let sample_w = if sampling_ratio > 0 {
            sampling_ratio
        } else {
            (bin_w.ceil() as usize).max(1)
        };

        for cc in 0..c {
            for oh in 0..output_height {
                for ow in 0..output_width {
                    let mut acc = 0.0f32;
                    let count = sample_h * sample_w;
                    for sh in 0..sample_h {
                        for sw in 0..sample_w {
                            let y =
                                roi_y1 + bin_h * (oh as f32 + (sh as f32 + 0.5) / sample_h as f32);
                            let x =
                                roi_x1 + bin_w * (ow as f32 + (sw as f32 + 0.5) / sample_w as f32);
                            let v = bilinear_interp(batch_idx, cc, y, x);
                            if mode == "max" {
                                acc = if sh == 0 && sw == 0 { v } else { acc.max(v) };
                            } else {
                                acc += v;
                            }
                        }
                    }
                    if mode != "max" {
                        acc /= count as f32;
                    }
                    out[((r * c + cc) * output_height + oh) * output_width + ow] = acc;
                }
            }
        }
    }

    let t = Tensor::from_vec(vec![num_rois, c, output_height, output_width], out).map_err(|e| {
        OnnxError::DecodeFailed {
            message: e.to_string(),
        }
    })?;
    env.insert(node.outputs[0].clone(), t);
    Ok(())
}

pub(super) fn exec_variadic_mean(node: &OnnxNode, env: &mut TensorEnv) -> Result<(), OnnxError> {
    let first = get_tensor(env, &node.name, &node.inputs[0])?;
    let n = node.inputs.len() as f32;
    let mut acc = first.data().to_vec();
    for inp in &node.inputs[1..] {
        let t = get_tensor(env, &node.name, inp)?;
        acc.iter_mut()
            .zip(t.data().iter())
            .for_each(|(a, &b)| *a += b);
    }
    for v in &mut acc {
        *v /= n;
    }
    let out =
        Tensor::from_vec(first.shape().to_vec(), acc).map_err(|e| OnnxError::DecodeFailed {
            message: e.to_string(),
        })?;
    env.insert(node.outputs[0].clone(), out);
    Ok(())
}

pub(super) fn exec_variadic_sum(node: &OnnxNode, env: &mut TensorEnv) -> Result<(), OnnxError> {
    let first = get_tensor(env, &node.name, &node.inputs[0])?;
    let mut acc = first.data().to_vec();
    for inp in &node.inputs[1..] {
        let t = get_tensor(env, &node.name, inp)?;
        acc.iter_mut()
            .zip(t.data().iter())
            .for_each(|(a, &b)| *a += b);
    }
    let out =
        Tensor::from_vec(first.shape().to_vec(), acc).map_err(|e| OnnxError::DecodeFailed {
            message: e.to_string(),
        })?;
    env.insert(node.outputs[0].clone(), out);
    Ok(())
}

/// ONNX NonMaxSuppression op.
///
/// Inputs:
///   0: boxes `[num_batches, num_boxes, 4]` — each box as `[y1, x1, y2, x2]`
///   1: scores `[num_batches, num_classes, num_boxes]`
///   2: max_output_boxes_per_class (optional, scalar)
///   3: iou_threshold (optional, scalar)
///   4: score_threshold (optional, scalar)
///
/// Output: `[num_selected, 3]` — each row `[batch_index, class_index, box_index]`.
pub(super) fn exec_nms(node: &OnnxNode, env: &mut TensorEnv) -> Result<(), OnnxError> {
    let boxes = get_tensor(env, &node.name, &node.inputs[0])?;
    let scores = get_tensor(env, &node.name, &node.inputs[1])?;

    let max_per_class = if node.inputs.len() > 2 && !node.inputs[2].is_empty() {
        get_tensor(env, &node.name, &node.inputs[2])?.data()[0] as usize
    } else {
        0
    };
    let iou_thr = if node.inputs.len() > 3 && !node.inputs[3].is_empty() {
        get_tensor(env, &node.name, &node.inputs[3])?.data()[0]
    } else {
        0.0
    };
    let score_thr = if node.inputs.len() > 4 && !node.inputs[4].is_empty() {
        get_tensor(env, &node.name, &node.inputs[4])?.data()[0]
    } else {
        f32::NEG_INFINITY
    };

    let bs = boxes.shape();
    let ss = scores.shape();
    let (num_batches, num_boxes) = (bs[0], bs[1]);
    let num_classes = ss[1];
    let box_data = boxes.data();
    let score_data = scores.data();

    let mut selected: Vec<[usize; 3]> = Vec::new();

    for b in 0..num_batches {
        for c in 0..num_classes {
            let mut candidates: Vec<(f32, usize)> = Vec::new();
            for i in 0..num_boxes {
                let s = score_data[(b * num_classes + c) * num_boxes + i];
                if s > score_thr {
                    candidates.push((s, i));
                }
            }
            candidates.sort_by(|a, cb| cb.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

            let limit = if max_per_class > 0 {
                max_per_class
            } else {
                candidates.len()
            };
            let mut kept: Vec<usize> = Vec::new();

            for &(_, idx) in &candidates {
                if kept.len() >= limit {
                    break;
                }
                let bi = (b * num_boxes + idx) * 4;
                let (y1i, x1i, y2i, x2i) = (
                    box_data[bi],
                    box_data[bi + 1],
                    box_data[bi + 2],
                    box_data[bi + 3],
                );
                let area_i = (y2i - y1i).max(0.0) * (x2i - x1i).max(0.0);

                let mut suppress = false;
                for &k in &kept {
                    let bk = (b * num_boxes + k) * 4;
                    let (y1k, x1k, y2k, x2k) = (
                        box_data[bk],
                        box_data[bk + 1],
                        box_data[bk + 2],
                        box_data[bk + 3],
                    );
                    let area_k = (y2k - y1k).max(0.0) * (x2k - x1k).max(0.0);
                    let inter = (y2i.min(y2k) - y1i.max(y1k)).max(0.0)
                        * (x2i.min(x2k) - x1i.max(x1k)).max(0.0);
                    let union = area_i + area_k - inter;
                    if union > 0.0 && inter / union > iou_thr {
                        suppress = true;
                        break;
                    }
                }
                if !suppress {
                    kept.push(idx);
                    selected.push([b, c, idx]);
                }
            }
        }
    }

    let n = selected.len();
    let data: Vec<f32> = selected
        .iter()
        .flat_map(|r| r.iter().map(|&v| v as f32))
        .collect();
    let shape = if n > 0 { vec![n, 3] } else { vec![0, 3] };
    let out = Tensor::from_vec(shape, data).map_err(|e| OnnxError::DecodeFailed {
        message: e.to_string(),
    })?;
    env.insert(node.outputs[0].clone(), out);
    Ok(())
}
