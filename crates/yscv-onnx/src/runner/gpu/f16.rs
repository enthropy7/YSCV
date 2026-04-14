use super::*;
use yscv_kernels::{GpuBackend, GpuBuffer};

// ═══════════════════════════════════════════════════════════════════
// ██  f16 I/O Pipeline — halves memory traffic for all intermediates
// ═══════════════════════════════════════════════════════════════════

/// Upload a CPU tensor to GPU as f16 storage (converted CPU-side).
fn to_gpu_f16(gpu: &GpuBackend, name: &str, env: &TensorEnv, gc: &mut GpuCache) {
    if gc.contains_key(name) {
        return;
    }
    if let Some(tensor) = env.get(name) {
        let data = tensor.data();
        if data.len() < 4 {
            return; // Too small for GPU vec4 shaders — leave on CPU
        }
        let buf_raw = gpu.storage_buf_f16(data);
        let buf = GpuBuffer::from_raw_parts(buf_raw, data.len(), tensor.shape().to_vec());
        gc.insert(
            name.to_string(),
            GpuTensor {
                buf,
                nhwc: false,
                f16_io: true,
            },
        );
    }
}

/// Unary op on f16 buffers (relu, sigmoid, silu).
/// Falls back to f32 for very small tensors (< 4 elements for vec4).
fn unary_f16(
    gpu: &GpuBackend,
    node: &OnnxNode,
    gc: &mut GpuCache,
    env: &mut TensorEnv,
    op: impl Fn(&GpuBackend, &GpuBuffer) -> Result<GpuBuffer, yscv_kernels::KernelError>,
) -> Result<(), OnnxError> {
    let name = &node.inputs[0];
    to_gpu_f16(gpu, name, env, gc);

    let gt = gc.get(name).ok_or_else(|| OnnxError::MissingInput {
        node: node.name.clone(),
        input: name.clone(),
    })?;

    if gt.buf.len() < 4 {
        return dispatch_f16_with_f32_fallback(gpu, node, env, gc);
    }

    let out = op(gpu, &gt.buf)?;
    let nhwc = gt.nhwc;
    gc.insert(
        node.outputs[0].clone(),
        GpuTensor {
            buf: out,
            nhwc,
            f16_io: true,
        },
    );
    Ok(())
}

/// Binary op on f16 buffers (add, mul).
/// Falls back to f32 for broadcasting or very small tensors (< 4 elements for vec4).
fn binary_f16(
    gpu: &GpuBackend,
    node: &OnnxNode,
    gc: &mut GpuCache,
    env: &mut TensorEnv,
    op: impl Fn(&GpuBackend, &GpuBuffer, &GpuBuffer) -> Result<GpuBuffer, yscv_kernels::KernelError>,
) -> Result<(), OnnxError> {
    let a_name = &node.inputs[0];
    let b_name = &node.inputs[1];
    to_gpu_f16(gpu, a_name, env, gc);
    to_gpu_f16(gpu, b_name, env, gc);

    let result = {
        let a = gc.get(a_name);
        let b = gc.get(b_name);
        match (a, b) {
            (Some(a), Some(b)) if a.buf.len() == b.buf.len() && a.buf.len() >= 4 => {
                Some((op(gpu, &a.buf, &b.buf)?, a.nhwc))
            }
            _ => None,
        }
    };

    if let Some((out, nhwc)) = result {
        gc.insert(
            node.outputs[0].clone(),
            GpuTensor {
                buf: out,
                nhwc,
                f16_io: true,
            },
        );
        Ok(())
    } else {
        // Fall back to f32 path for broadcasting or small tensors
        dispatch_f16_with_f32_fallback(gpu, node, env, gc)
    }
}

/// Conv with f16 I/O — input and weight buffers are f16, bias stays f32.
fn exec_conv_f16(
    gpu: &GpuBackend,
    node: &OnnxNode,
    env: &mut TensorEnv,
    gc: &mut GpuCache,
    act: u32,
) -> Result<(), OnnxError> {
    let weight = get_tensor(env, &node.name, &node.inputs[1])?;
    let bias = if node.inputs.len() > 2 && !node.inputs[2].is_empty() {
        Some(get_tensor(env, &node.name, &node.inputs[2])?)
    } else {
        None
    };

    let strides = get_attr_ints(node, "strides").unwrap_or_else(|| vec![1, 1]);
    let pads = get_attr_ints(node, "pads").unwrap_or_else(|| vec![0, 0, 0, 0]);
    let group = get_attr_int(node, "group").unwrap_or(1) as usize;
    let sh = strides[0] as usize;
    let sw = strides[1] as usize;
    let pad4 = [
        pads[0] as usize,
        pads[1] as usize,
        pads[2] as usize,
        pads[3] as usize,
    ];

    let w_shape = weight.shape();
    let is_khwc = env.is_khwc_weight(&node.inputs[1]);
    let (o_ch, _i_per_g, _kh, _kw) = if is_khwc {
        (w_shape[3], w_shape[2], w_shape[0], w_shape[1])
    } else {
        (w_shape[0], w_shape[1], w_shape[2], w_shape[3])
    };

    // Ensure input is on GPU in NHWC
    let input_name = &node.inputs[0];
    to_gpu_f16(gpu, input_name, env, gc);
    ensure_nhwc(gpu, input_name, gc)?;

    let ic = gc.get(input_name).map(|gt| gt.buf.shape()[3]).unwrap_or(0);

    // Non-standard grouped conv → fall back to f32 CPU
    if group > 1 && !(group == o_ch && group == ic) {
        inputs_to_cpu(gpu, node, env, gc)?;
        return conv::exec_conv(node, env, yscv_kernels::Activation::None);
    }

    if group == 1 {
        // Weight key: f16 weights cached separately
        let w_key = format!("__w_f16_{}", node.inputs[1]);
        let b_key = format!("__bias_{}", node.inputs[1]); // bias stays f32
        if !gc.contains_key(&w_key) {
            let w_nhwc = if is_khwc {
                weight.clone()
            } else {
                oihw_to_khwc_cout(weight)?
            };
            // Upload weight as f16
            let w_data = w_nhwc.data();
            let w_raw = gpu.storage_buf_f16(w_data);
            let w_buf = GpuBuffer::from_raw_parts(w_raw, w_data.len(), w_nhwc.shape().to_vec());
            gc.insert(
                w_key.clone(),
                GpuTensor {
                    buf: w_buf,
                    nhwc: false,
                    f16_io: true,
                },
            );
            // Bias stays f32
            let bias_data = bias
                .map(|b| b.data().to_vec())
                .unwrap_or_else(|| vec![0.0f32; o_ch]);
            let bias_t =
                Tensor::from_vec(vec![o_ch], bias_data).map_err(|e| OnnxError::DecodeFailed {
                    message: e.to_string(),
                })?;
            gc.insert(
                b_key.clone(),
                GpuTensor {
                    buf: gpu.upload(&bias_t),
                    nhwc: false,
                    f16_io: false,
                },
            );
        }

        let out = {
            let w = &gc.get(&w_key).unwrap().buf;
            let b = &gc.get(&b_key).unwrap().buf;
            let input_buf = &gc
                .get(input_name)
                .unwrap_or_else(|| {
                    unreachable!(
                        "f16 conv: input '{}' not in gc for node '{}' (op {}). Bug in graph scheduling.",
                        input_name, node.name, node.op_type,
                    )
                })
                .buf;
            gpu.im2col_conv_f16_on_device(input_buf, w, b, sh, sw, pad4, act)?
        };
        gc.insert(
            node.outputs[0].clone(),
            GpuTensor {
                buf: out,
                nhwc: true,
                f16_io: true,
            },
        );
    } else {
        // Depthwise: fall back to f32 CPU path
        inputs_to_cpu(gpu, node, env, gc)?;
        let cpu_act = yscv_kernels::Activation::None;
        return conv::exec_conv(node, env, cpu_act);
    }

    Ok(())
}

/// Concat on f16 buffers — channel-axis.
fn exec_concat_f16_gpu(
    gpu: &GpuBackend,
    node: &OnnxNode,
    env: &mut TensorEnv,
    gc: &mut GpuCache,
) -> Result<(), OnnxError> {
    let axis = get_attr_int(node, "axis").unwrap_or(0);
    let input_names: Vec<&String> = node.inputs.iter().filter(|n| !n.is_empty()).collect();

    if input_names.is_empty() {
        return super::reshape::exec_concat(node, env);
    }

    let any_on_gpu = input_names.iter().any(|n| gc.contains_key(n.as_str()));
    if !any_on_gpu {
        return super::reshape::exec_concat(node, env);
    }

    for &name in &input_names {
        to_gpu_f16(gpu, name, env, gc);
    }
    if !input_names.iter().all(|n| gc.contains_key(n.as_str())) {
        inputs_to_cpu(gpu, node, env, gc)?;
        return super::reshape::exec_concat(node, env);
    }

    let first = gc.get(input_names[0].as_str()).unwrap();
    let rank = first.buf.shape().len();
    let is_nhwc = first.nhwc && rank == 4;

    let actual_axis = if is_nhwc {
        match if axis < 0 { axis + 4 } else { axis } {
            0 => 0,
            1 => 3,
            2 => 1,
            3 => 2,
            a => a as usize,
        }
    } else if axis < 0 {
        (rank as i64 + axis) as usize
    } else {
        axis as usize
    };

    let bufs: Vec<&GpuBuffer> = input_names
        .iter()
        .map(|n| &gc.get(n.as_str()).unwrap().buf)
        .collect();
    let any_f16 = input_names
        .iter()
        .any(|n| gc.get(n.as_str()).unwrap().f16_io);
    let (out, is_f16) = if actual_axis == rank - 1 && any_f16 {
        (gpu.channel_concat_f16_on_device(&bufs)?, true)
    } else if actual_axis == rank - 1 {
        (gpu.channel_concat_on_device(&bufs)?, false)
    } else if any_f16 {
        // Non-channel concat with f16 inputs: convert to f32, concat, convert back to f16
        let f32_bufs: Vec<GpuBuffer> = bufs
            .iter()
            .map(|b| gpu.convert_f16_to_f32_on_device(b))
            .collect::<Result<Vec<_>, _>>()?;
        let f32_refs: Vec<&GpuBuffer> = f32_bufs.iter().collect();
        let f32_out = gpu.general_concat_on_device(&f32_refs, actual_axis)?;
        let f16_out = gpu.convert_f32_to_f16_on_device(&f32_out)?;
        (f16_out, true)
    } else {
        (gpu.general_concat_on_device(&bufs, actual_axis)?, false)
    };
    gc.insert(
        node.outputs[0].clone(),
        GpuTensor {
            buf: out,
            nhwc: is_nhwc,
            f16_io: is_f16,
        },
    );
    Ok(())
}

/// Resize on f16 buffers.
fn exec_resize_f16_gpu(
    gpu: &GpuBackend,
    node: &OnnxNode,
    env: &mut TensorEnv,
    gc: &mut GpuCache,
) -> Result<(), OnnxError> {
    let name = &node.inputs[0];
    let on_gpu_nhwc = gc
        .get(name)
        .is_some_and(|gt| gt.nhwc && gt.buf.shape().len() == 4);

    if !on_gpu_nhwc {
        inputs_to_cpu(gpu, node, env, gc)?;
        return super::misc::exec_resize(node, env);
    }

    let input_shape = gc.get(name).unwrap().buf.shape().to_vec();
    let (ih, iw) = (input_shape[1], input_shape[2]);

    let (oh, ow) = if node.inputs.len() > 3 && !node.inputs[3].is_empty() {
        to_cpu(gpu, &node.inputs[3], env, gc)?;
        if let Some(sizes) = env.get(&node.inputs[3]) {
            let sd = sizes.data();
            (sd[2] as usize, sd[3] as usize)
        } else {
            (ih, iw)
        }
    } else if node.inputs.len() > 2 && !node.inputs[2].is_empty() {
        to_cpu(gpu, &node.inputs[2], env, gc)?;
        if let Some(scales) = env.get(&node.inputs[2]) {
            let sd = scales.data();
            if sd.len() >= 4 && sd.iter().any(|&v| v != 0.0) {
                ((ih as f32 * sd[2]) as usize, (iw as f32 * sd[3]) as usize)
            } else {
                (ih, iw)
            }
        } else {
            (ih, iw)
        }
    } else {
        (ih, iw)
    };

    if oh == ih && ow == iw {
        if let Some(gt) = gc.remove(name) {
            gc.insert(node.outputs[0].clone(), gt);
        }
        return Ok(());
    }

    let input_buf = &gc.get(name).unwrap().buf;
    let out = gpu.resize_nearest_f16_on_device(input_buf, oh, ow)?;
    gc.insert(
        node.outputs[0].clone(),
        GpuTensor {
            buf: out,
            nhwc: true,
            f16_io: true,
        },
    );
    Ok(())
}

/// Split on f16 buffers.
fn exec_split_f16_gpu(
    gpu: &GpuBackend,
    node: &OnnxNode,
    env: &mut TensorEnv,
    gc: &mut GpuCache,
) -> Result<(), OnnxError> {
    let axis = get_attr_int(node, "axis").unwrap_or(0);
    let name = &node.inputs[0];

    let on_gpu = gc.contains_key(name.as_str());
    if !on_gpu {
        return super::reshape::exec_split(node, env);
    }

    let gt = gc.get(name).unwrap();
    let rank = gt.buf.shape().len();
    let is_nhwc = gt.nhwc && rank == 4;

    let actual_axis = if is_nhwc {
        match if axis < 0 { axis + 4 } else { axis } {
            0 => 0,
            1 => 3,
            2 => 1,
            3 => 2,
            a => a as usize,
        }
    } else if axis < 0 {
        (rank as i64 + axis) as usize
    } else {
        axis as usize
    };

    let split_sizes: Vec<usize> = if node.inputs.len() > 1 && !node.inputs[1].is_empty() {
        if let Some(st) = env.get(&node.inputs[1]) {
            st.data().iter().map(|&v| v as usize).collect()
        } else {
            let dim = gt.buf.shape()[actual_axis];
            let n_out = node.outputs.len();
            vec![dim / n_out; n_out]
        }
    } else {
        let attr_split = get_attr_ints(node, "split");
        if let Some(s) = attr_split {
            s.iter().map(|&v| v as usize).collect()
        } else {
            let dim = gt.buf.shape()[actual_axis];
            let n_out = node.outputs.len();
            vec![dim / n_out; n_out]
        }
    };

    if actual_axis == rank - 1 {
        let results = gpu.channel_split_f16_on_device(&gc.get(name).unwrap().buf, &split_sizes)?;
        for (i, buf) in results.into_iter().enumerate() {
            if i < node.outputs.len() {
                gc.insert(
                    node.outputs[i].clone(),
                    GpuTensor {
                        buf,
                        nhwc: is_nhwc,
                        f16_io: true,
                    },
                );
            }
        }
    } else {
        // Fall back to f32 for non-channel split
        inputs_to_cpu(gpu, node, env, gc)?;
        return super::reshape::exec_split(node, env);
    }

    Ok(())
}

fn fuse_relu_f16(
    gpu: &GpuBackend,
    src: &str,
    dst: &str,
    env: &mut TensorEnv,
    gc: &mut GpuCache,
) -> Result<(), OnnxError> {
    if let Some(gt) = gc.remove(src) {
        let out = gpu.relu_f16_on_device(&gt.buf)?;
        gc.insert(
            dst.to_string(),
            GpuTensor {
                buf: out,
                nhwc: gt.nhwc,
                f16_io: true,
            },
        );
    } else if let Some(tensor) = env.get_mut(src) {
        for v in tensor.data_mut() {
            *v = v.max(0.0);
        }
        env.alias(dst, src);
    }
    Ok(())
}

/// f16 dispatch router for compilation.
/// Native f16 shaders for Conv and common ops; f32 fallback with conversion for the rest.
fn dispatch_f16(
    gpu: &GpuBackend,
    node: &OnnxNode,
    env: &mut TensorEnv,
    gc: &mut GpuCache,
) -> Result<(), OnnxError> {
    match node.op_type.as_str() {
        "Conv" => exec_conv_f16(gpu, node, env, gc, 0),
        "Relu" => unary_f16(gpu, node, gc, env, |g, b| g.relu_f16_on_device(b)),
        "Sigmoid" => unary_f16(gpu, node, gc, env, |g, b| g.sigmoid_f16_on_device(b)),
        "Add" => binary_f16(gpu, node, gc, env, |g, a, b| g.add_f16_on_device(a, b)),
        "Mul" => binary_f16(gpu, node, gc, env, |g, a, b| g.mul_f16_on_device(a, b)),
        "Concat" => exec_concat_f16_gpu(gpu, node, env, gc),
        "Split" => exec_split_f16_gpu(gpu, node, env, gc),
        "Resize" | "Upsample" => exec_resize_f16_gpu(gpu, node, env, gc),
        // Shape: reads metadata only, safe for f16
        "Shape" => dispatch_inner(gpu, node, env, gc),
        // Unsqueeze/Squeeze: only adds/removes dims of size 1, buffer data unchanged
        "Unsqueeze" | "Squeeze" => dispatch_inner(gpu, node, env, gc),
        // Everything else: convert f16→f32 inputs, run f32 op, convert outputs back to f16
        _ => dispatch_f16_with_f32_fallback(gpu, node, env, gc),
    }
}

/// Run an f32 dispatch on a node whose inputs may be f16.
/// Converts f16 inputs → f32 before dispatch, then converts f32 outputs → f16 after.
/// Restores converted inputs back to f16 after dispatch so later f16 ops can use them.
fn dispatch_f16_with_f32_fallback(
    gpu: &GpuBackend,
    node: &OnnxNode,
    env: &mut TensorEnv,
    gc: &mut GpuCache,
) -> Result<(), OnnxError> {
    // Remove f16 inputs from gc, save them, put f32 versions in gc
    let mut saved_f16: Vec<(String, GpuTensor)> = Vec::new();
    for input_name in &node.inputs {
        if input_name.is_empty() {
            continue;
        }
        if let Some(gt) = gc.get(input_name.as_str())
            && gt.f16_io
        {
            let gt = gc.remove(input_name).unwrap();
            let f32_buf = gpu.convert_f16_to_f32_on_device(&gt.buf)?;
            let nhwc = gt.nhwc;
            saved_f16.push((input_name.clone(), gt));
            gc.insert(
                input_name.clone(),
                GpuTensor {
                    buf: f32_buf,
                    nhwc,
                    f16_io: false,
                },
            );
        }
    }

    // Download tiny f32 GPU buffers to CPU — they can't be used in vec4 shaders
    for input_name in &node.inputs {
        if input_name.is_empty() {
            continue;
        }
        if gc
            .get(input_name.as_str())
            .is_some_and(|gt| !gt.f16_io && gt.buf.len() < 4)
        {
            to_cpu(gpu, input_name, env, gc)?;
        }
    }

    // Run the f32 dispatch
    dispatch_inner(gpu, node, env, gc)?;

    // Restore original f16 inputs (overwrite any f32 remnants)
    for (name, gt) in saved_f16 {
        gc.insert(name, gt);
    }

    // Convert f32 outputs back to f16, but download tiny outputs to CPU
    for output_name in &node.outputs {
        if output_name.is_empty() {
            continue;
        }
        if let Some(gt) = gc.get(output_name.as_str()) {
            if !gt.f16_io && gt.buf.len() >= 64 {
                let f16_buf = gpu.convert_f32_to_f16_on_device(&gt.buf)?;
                let nhwc = gt.nhwc;
                gc.insert(
                    output_name.clone(),
                    GpuTensor {
                        buf: f16_buf,
                        nhwc,
                        f16_io: true,
                    },
                );
            } else if !gt.f16_io && gt.buf.len() < 64 {
                // Tiny output: download to CPU so later ops handle it naturally
                to_cpu(gpu, output_name, env, gc)?;
            }
        }
    }

    Ok(())
}

/// Compile a GPU execution plan with f16 intermediate buffers.
/// All activation buffers use f16 storage, halving memory bandwidth.
/// Weights are converted to f16, bias stays f32.
pub fn compile_gpu_plan_f16(
    gpu: &GpuBackend,
    model: &OnnxModel,
    plan: &GpuExecPlan,
    weight_cache: &mut GpuWeightCache,
    input_name: &str,
    input_tensor: &Tensor,
) -> Result<CompiledGpuPlan, OnnxError> {
    gpu.start_recording();

    let mut env = TensorEnv::from_model(model);
    let mut gc: GpuCache = std::mem::take(&mut weight_cache.0);

    for (name, tensor) in &model.initializers {
        env.insert(name.clone(), tensor.clone());
    }
    env.insert(input_name.to_string(), input_tensor.clone());

    let nodes = &model.nodes;
    for (i, node) in nodes.iter().enumerate() {
        if matches!(plan.actions[i], GpuExecAction::Skip) {
            continue;
        }

        match &plan.actions[i] {
            GpuExecAction::Skip => unreachable!(),

            GpuExecAction::ConvSiLU { output_name, .. } => {
                exec_conv_f16(gpu, node, &mut env, &mut gc, 2)?;
                if let Some(gt) = gc.remove(&node.outputs[0]) {
                    gc.insert(output_name.clone(), gt);
                } else if let Some(conv_out) = env.remove(&node.outputs[0]) {
                    // Conv fell back to CPU (e.g., depthwise) — apply SiLU on CPU
                    let data: Vec<f32> = conv_out
                        .data()
                        .iter()
                        .map(|&v| v / (1.0 + (-v).exp()))
                        .collect();
                    let out = Tensor::from_vec(conv_out.shape().to_vec(), data).unwrap();
                    env.insert(output_name.clone(), out);
                }
            }

            GpuExecAction::SiLU {
                x_name,
                output_name,
                ..
            } => {
                to_gpu_f16(gpu, x_name, &env, &mut gc);
                if let Some(gt) = gc.get(x_name.as_str()) {
                    let out = gpu.silu_f16_on_device(&gt.buf)?;
                    let nhwc = gt.nhwc;
                    gc.insert(
                        output_name.clone(),
                        GpuTensor {
                            buf: out,
                            nhwc,
                            f16_io: true,
                        },
                    );
                } else if let Some(t) = env.get(x_name) {
                    let data: Vec<f32> = t.data().iter().map(|&v| v / (1.0 + (-v).exp())).collect();
                    let out = Tensor::from_vec(t.shape().to_vec(), data).unwrap();
                    env.insert(output_name.clone(), out);
                }
            }

            GpuExecAction::ConvBnRelu {
                bn_idx,
                bn_output,
                relu_output,
                ..
            } => {
                dispatch_f16(gpu, node, &mut env, &mut gc)?;
                dispatch_f16(gpu, &nodes[*bn_idx], &mut env, &mut gc)?;
                fuse_relu_f16(gpu, bn_output, relu_output, &mut env, &mut gc)?;
            }

            GpuExecAction::OpRelu {
                op_output,
                relu_output,
                ..
            } => {
                dispatch_f16(gpu, node, &mut env, &mut gc)?;
                fuse_relu_f16(gpu, op_output, relu_output, &mut env, &mut gc)?;
            }

            GpuExecAction::MatMulAdd { add_idx } => {
                dispatch_f16(gpu, node, &mut env, &mut gc)?;
                dispatch_f16(gpu, &nodes[*add_idx], &mut env, &mut gc)?;
            }

            GpuExecAction::Normal => {
                let is_shape_op = matches!(
                    node.op_type.as_str(),
                    "Reshape" | "Flatten" | "Unsqueeze" | "Squeeze"
                );
                if is_shape_op
                    && gc
                        .get(node.inputs[0].as_str())
                        .is_some_and(|gt| !gt.nhwc || gt.buf.shape().len() != 4)
                    && plan
                        .last_use
                        .get(node.inputs[0].as_str())
                        .is_some_and(|&lu| lu <= i)
                {
                    let gt = gc.remove(&node.inputs[0]).unwrap();
                    let shape = gt.buf.shape().to_vec();
                    let total = gt.buf.len();
                    let nhwc = gt.nhwc;
                    let f16_io = gt.f16_io;
                    let new_shape = match node.op_type.as_str() {
                        "Unsqueeze" => {
                            let axes: Vec<i64> =
                                if node.inputs.len() > 1 && !node.inputs[1].is_empty() {
                                    if let Some(at) = env.get(&node.inputs[1]) {
                                        at.data().iter().map(|&v| v as i64).collect()
                                    } else {
                                        get_attr_ints(node, "axes").unwrap_or_default()
                                    }
                                } else {
                                    get_attr_ints(node, "axes").unwrap_or_default()
                                };
                            let mut sorted: Vec<usize> = axes
                                .iter()
                                .map(|&a| {
                                    if a < 0 {
                                        (shape.len() as i64 + 1 + a) as usize
                                    } else {
                                        a as usize
                                    }
                                })
                                .collect();
                            sorted.sort();
                            let mut s = shape;
                            for &ax in &sorted {
                                s.insert(ax, 1);
                            }
                            s
                        }
                        "Squeeze" => {
                            let axes: Vec<i64> =
                                if node.inputs.len() > 1 && !node.inputs[1].is_empty() {
                                    if let Some(at) = env.get(&node.inputs[1]) {
                                        at.data().iter().map(|&v| v as i64).collect()
                                    } else {
                                        vec![]
                                    }
                                } else {
                                    get_attr_ints(node, "axes").unwrap_or_default()
                                };
                            let mut s = shape;
                            if axes.is_empty() {
                                s.retain(|&d| d != 1);
                            } else {
                                let mut to_remove: Vec<usize> = axes
                                    .iter()
                                    .map(|&a| {
                                        if a < 0 {
                                            (s.len() as i64 + a) as usize
                                        } else {
                                            a as usize
                                        }
                                    })
                                    .collect();
                                to_remove.sort_unstable_by(|a, b| b.cmp(a));
                                for ax in to_remove {
                                    if s[ax] == 1 {
                                        s.remove(ax);
                                    }
                                }
                            }
                            s
                        }
                        _ => get_reshape_shape(node, &mut env, gpu, &mut gc, &shape, total)?,
                    };
                    let out = gpu.reshape_on_device(gt.buf, new_shape);
                    gc.insert(
                        node.outputs[0].clone(),
                        GpuTensor {
                            buf: out,
                            nhwc,
                            f16_io,
                        },
                    );
                } else {
                    dispatch_f16(gpu, node, &mut env, &mut gc)?;
                }
            }
        }

        // Flush after each dispatch during compilation to catch GPU errors early
        gpu.flush();
    }

    gpu.flush();

    let ops = gpu.take_recording();

    // Extract input buffer
    let input_buf = gc
        .remove(input_name)
        .ok_or_else(|| OnnxError::MissingInput {
            node: "compiled_plan_f16".to_string(),
            input: input_name.to_string(),
        })?
        .buf;

    // Extract output buffers
    let output_names: Vec<String> = model.outputs.clone();
    let mut output_bufs = Vec::new();
    for name in &output_names {
        if let Some(gt) = gc.get(name.as_str()) {
            output_bufs.push((
                GpuBuffer::from_raw_parts(
                    gt.buf.raw_buffer().clone(),
                    gt.buf.len(),
                    gt.buf.shape().to_vec(),
                ),
                gt.nhwc,
                gt.f16_io,
            ));
        }
    }

    // Separate weight cache entries from activation entries
    let mut weights = GpuCache::new();
    let mut pinned = GpuCache::new();
    let keys: Vec<String> = gc.keys().cloned().collect();
    for k in keys {
        if let Some(v) = gc.remove(&k) {
            if k.starts_with("__") {
                weights.insert(k, v);
            } else {
                pinned.insert(k, v);
            }
        }
    }
    weight_cache.0 = weights;

    Ok(CompiledGpuPlan {
        ops,
        pinned,
        input_buf,
        output_names,
        output_bufs,
    })
}

/// Run f16-compiled plan with fused single-pass replay.
/// Input is written as f16. Outputs are read as f16 and converted to f32 NCHW.
pub fn run_compiled_gpu_f16_fused(
    gpu: &GpuBackend,
    compiled: &CompiledGpuPlan,
    input_data: &[f32],
) -> Result<HashMap<String, Tensor>, OnnxError> {
    // Write input data as f16
    gpu.write_buffer_f16(compiled.input_buf.raw_buffer(), input_data);

    // Replay in single pass (Metal)
    gpu.replay_recording_fused(&compiled.ops);

    // Download outputs — handle f16 and f32 buffers based on f16_io flag
    let mut result = HashMap::new();
    for (i, name) in compiled.output_names.iter().enumerate() {
        if let Some((out_buf, nhwc, f16_io)) = compiled.output_bufs.get(i) {
            let shape = out_buf.shape();
            if *f16_io {
                if *nhwc && shape.len() == 4 {
                    // Use GPU NHWC(f16)→NCHW(f32) permute shader
                    let nchw_buf = gpu.nhwc_to_nchw_f16_to_f32_on_device(out_buf)?;
                    gpu.flush();
                    let t = gpu.download(&nchw_buf)?;
                    result.insert(name.clone(), t);
                } else {
                    // Read f16 buffer as f32
                    gpu.flush();
                    let data = gpu.read_buf_f16(out_buf.raw_buffer(), out_buf.len())?;
                    let t = Tensor::from_vec(shape.to_vec(), data).map_err(|e| {
                        OnnxError::GpuKernel {
                            message: e.to_string(),
                        }
                    })?;
                    result.insert(name.clone(), t);
                }
            } else {
                // f32 output — standard download
                let t = gpu.download(out_buf)?;
                let shape = out_buf.shape();
                if *nhwc && shape.len() == 4 {
                    let (n, h, w, c) = (shape[0], shape[1], shape[2], shape[3]);
                    let data = t.data();
                    let mut nchw = vec![0.0f32; data.len()];
                    for ni in 0..n {
                        for hi in 0..h {
                            for wi in 0..w {
                                for ci in 0..c {
                                    nchw[((ni * c + ci) * h + hi) * w + wi] =
                                        data[((ni * h + hi) * w + wi) * c + ci];
                                }
                            }
                        }
                    }
                    let out = Tensor::from_vec(vec![n, c, h, w], nchw).unwrap();
                    result.insert(name.clone(), out);
                } else {
                    result.insert(name.clone(), t);
                }
            }
        }
    }

    Ok(result)
}
