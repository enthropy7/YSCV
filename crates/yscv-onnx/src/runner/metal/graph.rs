//! MPSGraph-based whole-model inference.
//! Builds an MPSGraph from the ONNX model, compiles it once,
//! then executes as a single GPU dispatch — no per-op encoder transitions.

use std::collections::HashMap;

use metal::*;

use yscv_kernels::metal_backend::mpsgraph::{
    Conv2dDesc, MpsGraph, MpsGraphExecutable, MpsGraphTensorRef, Pool2dDesc,
};
use yscv_tensor::Tensor;

use crate::error::OnnxError;
use crate::loader::{OnnxModel, OnnxNode};
use crate::runner::{get_attr_float, get_attr_int, get_attr_ints};

/// Compiled MPSGraph plan for inference.
pub struct MpsGraphPlan {
    graph: MpsGraph,
    executable: MpsGraphExecutable,
    #[allow(dead_code)]
    device: Device,
    queue: CommandQueue,
    input_placeholder: MpsGraphTensorRef,
    input_shape: Vec<usize>,
    input_buf: Buffer, // pre-allocated shared buffer for input
    output_tensors: Vec<(String, MpsGraphTensorRef)>,
    output_shapes: Vec<Vec<usize>>,
}

impl MpsGraphPlan {
    pub fn ops_count(&self) -> usize {
        // Graph ops are opaque; return 0 (unknown)
        0
    }
}

/// Compile an ONNX model into an MPSGraph execution plan.
/// The graph operates in NCHW layout with f16 precision throughout.
pub fn compile_mpsgraph_plan(
    model: &OnnxModel,
    input_name: &str,
    input_tensor: &Tensor,
) -> Result<MpsGraphPlan, OnnxError> {
    let graph = MpsGraph::new().ok_or_else(|| OnnxError::DecodeFailed {
        message: "Failed to create MPSGraph".to_string(),
    })?;

    let device = Device::system_default().ok_or_else(|| OnnxError::DecodeFailed {
        message: "No Metal device available".to_string(),
    })?;
    let queue = device.new_command_queue();

    let input_shape = input_tensor.shape().to_vec();

    // Create placeholder for the input tensor (NCHW, f32).
    // We accept f32 from the caller and let MPSGraph cast to f16 on GPU — this avoids
    // expensive CPU-side f32→f16 conversion (~3M elements for VballNet).
    let input_placeholder_f32 = graph.placeholder_f32(&input_shape, input_name);
    let input_f16 = graph.cast_to_f16(input_placeholder_f32);

    // Map tensor name → MpsGraphTensorRef as we walk the graph.
    // Use the f16-cast tensor so all downstream ops run in f16.
    let mut tensors: HashMap<String, MpsGraphTensorRef> = HashMap::new();
    tensors.insert(input_name.to_string(), input_f16);

    // Track shapes for output readback
    let mut cpu_shapes: HashMap<String, Vec<usize>> = HashMap::new();
    cpu_shapes.insert(input_name.to_string(), input_shape.clone());

    // Upload initializers as constants.
    // Conv weights may have been pre-permuted OIHW→KHWC by the loader.
    // MPSGraph needs OIHW, so we reverse-permute KHWC weights back.
    for (name, tensor) in &model.initializers {
        let shape = tensor.shape();
        let data = tensor.data();

        let (final_data, final_shape) = if model.khwc_weights.contains(name) && shape.len() == 4 {
            // KHWC [kh, kw, ic, oc] → OIHW [oc, ic, kh, kw]
            let (kh, kw, ic, oc) = (shape[0], shape[1], shape[2], shape[3]);
            let mut oihw = vec![0.0f32; data.len()];
            for o in 0..oc {
                for i in 0..ic {
                    for h in 0..kh {
                        for w in 0..kw {
                            oihw[((o * ic + i) * kh + h) * kw + w] =
                                data[((h * kw + w) * ic + i) * oc + o];
                        }
                    }
                }
            }
            (oihw, vec![oc, ic, kh, kw])
        } else {
            (data.to_vec(), shape.to_vec())
        };

        // MPSGraph requires at least rank-1 shapes for constants
        if final_shape.is_empty() {
            let final_shape = vec![1usize];
            let f16_data: Vec<u16> = final_data.iter().map(|&v| f32_to_f16_bits(v)).collect();
            let t = graph.constant_f16(&f16_data, &final_shape);
            tensors.insert(name.clone(), t);
            cpu_shapes.insert(name.clone(), final_shape);
            continue;
        }

        let f16_data: Vec<u16> = final_data.iter().map(|&v| f32_to_f16_bits(v)).collect();
        let t = graph.constant_f16(&f16_data, &final_shape);
        tensors.insert(name.clone(), t);
        cpu_shapes.insert(name.clone(), final_shape);
    }

    // Constant values map for compile-time shape ops (Shape, Gather, etc.)
    // These produce integer/metadata tensors that don't go into the GPU graph.
    let mut const_values: HashMap<String, Vec<f32>> = HashMap::new();
    // Seed with initializer data
    for (name, tensor) in &model.initializers {
        const_values.insert(name.clone(), tensor.data().to_vec());
    }

    // Walk ONNX nodes and build graph ops
    let debug = std::env::var("MPSGRAPH_DEBUG").is_ok();
    for (i, node) in model.nodes.iter().enumerate() {
        if debug {
            eprintln!(
                "  [mpsgraph] node {} {} '{}' inputs={:?} outputs={:?}",
                i, node.op_type, node.name, node.inputs, node.outputs
            );
        }

        // Handle compile-time shape ops that don't produce GPU tensors
        match node.op_type.as_str() {
            "Shape" => {
                let in_shape = cpu_shapes.get(&node.inputs[0]).cloned().unwrap_or_default();
                let values: Vec<f32> = in_shape.iter().map(|&d| d as f32).collect();
                let out_shape = vec![values.len()];
                const_values.insert(node.outputs[0].clone(), values);
                cpu_shapes.insert(node.outputs[0].clone(), out_shape);
                continue;
            }
            "Gather" if const_values.contains_key(&node.inputs[0]) => {
                let data = const_values
                    .get(&node.inputs[0])
                    .cloned()
                    .unwrap_or_default();
                let indices = const_values
                    .get(&node.inputs[1])
                    .cloned()
                    .unwrap_or_default();
                if indices.is_empty() {
                    // Scalar gather — single index from initializer
                    if let Some(idx_t) = model.initializers.get(&node.inputs[1]) {
                        let idx = idx_t.data()[0] as i64;
                        let idx = if idx < 0 {
                            data.len() as i64 + idx
                        } else {
                            idx
                        } as usize;
                        const_values.insert(node.outputs[0].clone(), vec![data[idx]]);
                        cpu_shapes.insert(node.outputs[0].clone(), vec![1]);
                    }
                } else {
                    let gathered: Vec<f32> = indices
                        .iter()
                        .map(|&idx| {
                            let i = idx as i64;
                            let i = if i < 0 { data.len() as i64 + i } else { i } as usize;
                            data.get(i).copied().unwrap_or(0.0)
                        })
                        .collect();
                    let out_shape = vec![gathered.len()];
                    const_values.insert(node.outputs[0].clone(), gathered);
                    cpu_shapes.insert(node.outputs[0].clone(), out_shape);
                }
                continue;
            }
            "Mul" if node.inputs.iter().all(|n| const_values.contains_key(n)) => {
                let a = const_values
                    .get(&node.inputs[0])
                    .cloned()
                    .unwrap_or_default();
                let b = const_values
                    .get(&node.inputs[1])
                    .cloned()
                    .unwrap_or_default();
                let values: Vec<f32> = a.iter().zip(b.iter()).map(|(x, y)| x * y).collect();
                let out_shape = vec![values.len().max(1)];
                const_values.insert(node.outputs[0].clone(), values);
                cpu_shapes.insert(node.outputs[0].clone(), out_shape);
                continue;
            }
            "Concat"
                if node
                    .inputs
                    .iter()
                    .all(|n| const_values.contains_key(n) || n.is_empty()) =>
            {
                let mut values = Vec::new();
                for inp in &node.inputs {
                    if let Some(v) = const_values.get(inp) {
                        values.extend_from_slice(v);
                    }
                }
                let out_shape = vec![values.len()];
                const_values.insert(node.outputs[0].clone(), values);
                cpu_shapes.insert(node.outputs[0].clone(), out_shape);
                continue;
            }
            "Unsqueeze" if const_values.contains_key(&node.inputs[0]) => {
                let values = const_values
                    .get(&node.inputs[0])
                    .cloned()
                    .unwrap_or_default();
                const_values.insert(node.outputs[0].clone(), values.clone());
                // Simplified: just keep the data, actual shape doesn't matter for constants
                cpu_shapes.insert(node.outputs[0].clone(), vec![values.len()]);
                continue;
            }
            "Cast" if const_values.contains_key(&node.inputs[0]) => {
                // Compile-time cast: just pass through the const values
                let values = const_values
                    .get(&node.inputs[0])
                    .cloned()
                    .unwrap_or_default();
                let out_shape = cpu_shapes
                    .get(&node.inputs[0])
                    .cloned()
                    .unwrap_or_else(|| vec![values.len()]);
                const_values.insert(node.outputs[0].clone(), values);
                cpu_shapes.insert(node.outputs[0].clone(), out_shape);
                continue;
            }
            "Range"
                if node.inputs.iter().all(|n| {
                    const_values.contains_key(n) || model.initializers.contains_key(n)
                }) =>
            {
                let get_val = |idx: usize| -> f32 {
                    if let Some(cv) = const_values.get(&node.inputs[idx]) {
                        cv[0]
                    } else if let Some(t) = model.initializers.get(&node.inputs[idx]) {
                        t.data()[0]
                    } else {
                        0.0
                    }
                };
                let start = get_val(0);
                let limit = get_val(1);
                let delta = get_val(2);
                let mut values = Vec::new();
                let mut v = start;
                while (delta > 0.0 && v < limit) || (delta < 0.0 && v > limit) {
                    values.push(v);
                    v += delta;
                }
                let out_shape = vec![values.len()];
                let f16_data: Vec<u16> = values.iter().map(|&v| f32_to_f16_bits(v)).collect();
                let t = graph.constant_f16(&f16_data, &out_shape);
                tensors.insert(node.outputs[0].clone(), t);
                cpu_shapes.insert(node.outputs[0].clone(), out_shape);
                const_values.insert(node.outputs[0].clone(), values);
                continue;
            }
            "ConstantOfShape" => {
                let shape_vals = const_values
                    .get(&node.inputs[0])
                    .cloned()
                    .unwrap_or_default();
                let out_shape: Vec<usize> = shape_vals.iter().map(|&v| v as usize).collect();
                let n: usize = out_shape.iter().product();
                // Default fill value is 0.0
                let fill = get_attr_float(node, "value").unwrap_or(0.0);
                let f16_data: Vec<u16> = vec![f32_to_f16_bits(fill); n];
                let t = graph.constant_f16(&f16_data, &out_shape);
                tensors.insert(node.outputs[0].clone(), t);
                cpu_shapes.insert(node.outputs[0].clone(), out_shape);
                continue;
            }
            _ => {}
        }

        // Auto-promote const_values to graph constants for any inputs that
        // are in const_values but not yet in the graph tensor map
        for inp_name in &node.inputs {
            if !inp_name.is_empty()
                && !tensors.contains_key(inp_name)
                && let Some(cv) = const_values.get(inp_name)
            {
                let shape = cpu_shapes
                    .get(inp_name)
                    .cloned()
                    .unwrap_or_else(|| vec![cv.len()]);
                let f16_data: Vec<u16> = cv.iter().map(|&v| f32_to_f16_bits(v)).collect();
                let t = graph.constant_f16(&f16_data, &shape);
                tensors.insert(inp_name.clone(), t);
            }
        }

        let result = build_graph_node(&graph, node, &tensors, &cpu_shapes, model, &const_values);

        match result {
            Some(outputs) => {
                for (name, tensor_ref, shape) in outputs {
                    tensors.insert(name.clone(), tensor_ref);
                    cpu_shapes.insert(name, shape);
                }
            }
            None => {
                if debug {
                    eprintln!(
                        "  [mpsgraph] SKIP unsupported op: {} '{}'",
                        node.op_type, node.name
                    );
                }
                return Err(OnnxError::DecodeFailed {
                    message: format!(
                        "MPSGraph: unsupported op '{}' (node '{}')",
                        node.op_type, node.name
                    ),
                });
            }
        }
    }

    // Collect output tensors
    let mut output_tensors = Vec::new();
    let mut output_shapes = Vec::new();
    for out_name in &model.outputs {
        if let Some(&t) = tensors.get(out_name) {
            // Cast output to f32 for readback
            let t_f32 = graph.cast_to_f32(t);
            output_tensors.push((out_name.clone(), t_f32));
            let shape = cpu_shapes.get(out_name).cloned().unwrap_or_default();
            output_shapes.push(shape);
        }
    }

    // Compile the graph — input is f32 (GPU casts to f16 internally)
    let feeds = vec![(
        input_placeholder_f32,
        input_shape.as_slice(),
        yscv_kernels::metal_backend::mpsgraph::MPS_DATA_TYPE_FLOAT32,
    )];
    let target_refs: Vec<MpsGraphTensorRef> = output_tensors.iter().map(|(_, t)| *t).collect();

    let executable = graph
        .compile(&device, &feeds, &target_refs)
        .ok_or_else(|| OnnxError::DecodeFailed {
            message: "Failed to compile MPSGraph".to_string(),
        })?;

    // Pre-allocate a shared input buffer (f32) to avoid per-run allocation.
    let input_elems: usize = input_shape.iter().product();
    let input_buf = device.new_buffer(
        (input_elems * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    Ok(MpsGraphPlan {
        graph,
        executable,
        device,
        queue,
        input_placeholder: input_placeholder_f32,
        input_shape,
        input_buf,
        output_tensors,
        output_shapes,
    })
}

/// Execute a compiled MPSGraph plan.
pub fn run_mpsgraph_plan(
    plan: &MpsGraphPlan,
    input_data: &[f32],
) -> Result<HashMap<String, Tensor>, OnnxError> {
    // Copy f32 input into pre-allocated shared buffer (memcpy, no alloc).
    // The graph casts f32→f16 on GPU (hardware path, essentially free).
    let input_bytes = input_data.len() * 4;
    unsafe {
        std::ptr::copy_nonoverlapping(
            input_data.as_ptr() as *const u8,
            plan.input_buf.contents() as *mut u8,
            input_bytes,
        );
    }
    let inputs = vec![(
        plan.input_placeholder,
        &plan.input_buf,
        plan.input_shape.as_slice(),
        yscv_kernels::metal_backend::mpsgraph::MPS_DATA_TYPE_FLOAT32,
    )];

    let out_bufs = plan
        .graph
        .run_with_buffers(&plan.executable, &plan.queue, &inputs);

    // Read output buffers
    let mut result = HashMap::new();
    for (idx, (name, _)) in plan.output_tensors.iter().enumerate() {
        if idx < out_bufs.len() {
            let buf = &out_bufs[idx];
            let shape = &plan.output_shapes[idx];
            let n: usize = shape.iter().product();
            let ptr = buf.contents() as *const f32;
            let data = unsafe { std::slice::from_raw_parts(ptr, n).to_vec() };
            let tensor =
                Tensor::from_vec(shape.clone(), data).map_err(|e| OnnxError::DecodeFailed {
                    message: e.to_string(),
                })?;
            result.insert(name.clone(), tensor);
        }
    }

    Ok(result)
}

/// Build a single ONNX node as MPSGraph operations.
/// Returns None if the op is unsupported.
/// Returns Some(vec of (output_name, tensor_ref, shape)) on success.
fn build_graph_node(
    graph: &MpsGraph,
    node: &OnnxNode,
    tensors: &HashMap<String, MpsGraphTensorRef>,
    shapes: &HashMap<String, Vec<usize>>,
    model: &OnnxModel,
    const_values: &HashMap<String, Vec<f32>>,
) -> Option<Vec<(String, MpsGraphTensorRef, Vec<usize>)>> {
    let get = |name: &str| -> Option<MpsGraphTensorRef> { tensors.get(name).copied() };
    let get_shape = |name: &str| -> Vec<usize> { shapes.get(name).cloned().unwrap_or_default() };

    match node.op_type.as_str() {
        "Conv" => {
            let input = get(&node.inputs[0])?;
            let weight = get(&node.inputs[1])?;
            let in_shape = get_shape(&node.inputs[0]);
            let w_shape = get_shape(&node.inputs[1]);

            // Weight is in OIHW format (or KHWC if pre-permuted).
            // For MPSGraph we need OIHW. Initializers are stored as-is (OIHW)
            // unless they were pre-permuted to KHWC by the loader.
            // Since we load weights as constants directly from initializer data,
            // they're in the original ONNX format (OIHW).

            let strides = get_attr_ints(node, "strides").unwrap_or_else(|| vec![1, 1]);
            let pads = get_attr_ints(node, "pads").unwrap_or_else(|| vec![0, 0, 0, 0]);
            let dilations = get_attr_ints(node, "dilations").unwrap_or_else(|| vec![1, 1]);
            let group = get_attr_int(node, "group").unwrap_or(1) as usize;

            let (o_ch, _i_per_g, kh, kw) = (w_shape[0], w_shape[1], w_shape[2], w_shape[3]);

            let desc = Conv2dDesc {
                stride_h: strides[0] as usize,
                stride_w: strides[1] as usize,
                dilation_h: dilations[0] as usize,
                dilation_w: dilations[1] as usize,
                pad_top: pads[0] as usize,
                pad_left: pads[1] as usize,
                pad_bottom: pads[2] as usize,
                pad_right: pads[3] as usize,
                groups: group,
            };

            let mut out = graph.conv2d(input, weight, &desc);

            // Add bias if present
            if node.inputs.len() > 2
                && !node.inputs[2].is_empty()
                && let Some(bias) = get(&node.inputs[2])
            {
                // Reshape bias [C] → [1, C, 1, 1] for NCHW broadcast
                let bias_reshaped = graph.reshape(bias, &[1, o_ch as i64, 1, 1]);
                out = graph.add(out, bias_reshaped);
            }

            // Compute output shape
            let (n, _c, ih, iw) = (in_shape[0], in_shape[1], in_shape[2], in_shape[3]);
            let oh =
                (ih + pads[0] as usize + pads[2] as usize - dilations[0] as usize * (kh - 1) - 1)
                    / strides[0] as usize
                    + 1;
            let ow =
                (iw + pads[1] as usize + pads[3] as usize - dilations[1] as usize * (kw - 1) - 1)
                    / strides[1] as usize
                    + 1;

            Some(vec![(node.outputs[0].clone(), out, vec![n, o_ch, oh, ow])])
        }

        "Relu" => {
            let input = get(&node.inputs[0])?;
            let shape = get_shape(&node.inputs[0]);
            let out = graph.relu(input);
            Some(vec![(node.outputs[0].clone(), out, shape)])
        }

        "Sigmoid" => {
            let input = get(&node.inputs[0])?;
            let shape = get_shape(&node.inputs[0]);
            let out = graph.sigmoid(input);
            Some(vec![(node.outputs[0].clone(), out, shape)])
        }

        "Add" | "Sub" | "Mul" | "Div" => {
            let a = get(&node.inputs[0])?;
            let b = get(&node.inputs[1])?;
            let a_shape = get_shape(&node.inputs[0]);
            let b_shape = get_shape(&node.inputs[1]);
            let out = match node.op_type.as_str() {
                "Add" => graph.add(a, b),
                "Sub" => graph.sub(a, b),
                "Mul" => graph.mul(a, b),
                "Div" => graph.div(a, b),
                _ => unreachable!(),
            };
            // Output shape = broadcast of a_shape and b_shape (use larger)
            let out_shape = if a_shape.len() >= b_shape.len() {
                a_shape
            } else {
                b_shape
            };
            Some(vec![(node.outputs[0].clone(), out, out_shape)])
        }

        "MaxPool" => {
            let input = get(&node.inputs[0])?;
            let in_shape = get_shape(&node.inputs[0]);
            let kernel_shape = get_attr_ints(node, "kernel_shape").unwrap_or_else(|| vec![2, 2]);
            let strides = get_attr_ints(node, "strides").unwrap_or_else(|| vec![1, 1]);
            let pads = get_attr_ints(node, "pads").unwrap_or_else(|| vec![0, 0, 0, 0]);

            let desc = Pool2dDesc {
                kernel_h: kernel_shape[0] as usize,
                kernel_w: kernel_shape[1] as usize,
                stride_h: strides[0] as usize,
                stride_w: strides[1] as usize,
                pad_top: pads[0] as usize,
                pad_left: pads[1] as usize,
                pad_bottom: pads[2] as usize,
                pad_right: pads[3] as usize,
            };
            let out = graph.max_pool2d(input, &desc);

            let (n, c, ih, iw) = (in_shape[0], in_shape[1], in_shape[2], in_shape[3]);
            let oh = (ih + pads[0] as usize + pads[2] as usize - kernel_shape[0] as usize)
                / strides[0] as usize
                + 1;
            let ow = (iw + pads[1] as usize + pads[3] as usize - kernel_shape[1] as usize)
                / strides[1] as usize
                + 1;

            Some(vec![(node.outputs[0].clone(), out, vec![n, c, oh, ow])])
        }

        "AveragePool" => {
            let input = get(&node.inputs[0])?;
            let in_shape = get_shape(&node.inputs[0]);
            let kernel_shape = get_attr_ints(node, "kernel_shape").unwrap_or_else(|| vec![2, 2]);
            let strides = get_attr_ints(node, "strides").unwrap_or_else(|| vec![1, 1]);
            let pads = get_attr_ints(node, "pads").unwrap_or_else(|| vec![0, 0, 0, 0]);

            let desc = Pool2dDesc {
                kernel_h: kernel_shape[0] as usize,
                kernel_w: kernel_shape[1] as usize,
                stride_h: strides[0] as usize,
                stride_w: strides[1] as usize,
                pad_top: pads[0] as usize,
                pad_left: pads[1] as usize,
                pad_bottom: pads[2] as usize,
                pad_right: pads[3] as usize,
            };
            let out = graph.avg_pool2d(input, &desc);

            let (n, c, ih, iw) = (in_shape[0], in_shape[1], in_shape[2], in_shape[3]);
            let oh = (ih + pads[0] as usize + pads[2] as usize - kernel_shape[0] as usize)
                / strides[0] as usize
                + 1;
            let ow = (iw + pads[1] as usize + pads[3] as usize - kernel_shape[1] as usize)
                / strides[1] as usize
                + 1;

            Some(vec![(node.outputs[0].clone(), out, vec![n, c, oh, ow])])
        }

        "GlobalAveragePool" => {
            let input = get(&node.inputs[0])?;
            let in_shape = get_shape(&node.inputs[0]);
            let out = graph.global_avg_pool(input);
            let (n, c) = (in_shape[0], in_shape[1]);
            Some(vec![(node.outputs[0].clone(), out, vec![n, c, 1, 1])])
        }

        "BatchNormalization" => {
            let input = get(&node.inputs[0])?;
            let gamma = get(&node.inputs[1])?;
            let beta = get(&node.inputs[2])?;
            let mean = get(&node.inputs[3])?;
            let variance = get(&node.inputs[4])?;
            let epsilon = get_attr_float(node, "epsilon").unwrap_or(1e-5);
            let in_shape = get_shape(&node.inputs[0]);

            // Reshape gamma/beta/mean/var from [C] → [1, C, 1, 1] for NCHW broadcast
            let c = in_shape.get(1).copied().unwrap_or(1) as i64;
            let gamma_r = graph.reshape(gamma, &[1, c, 1, 1]);
            let beta_r = graph.reshape(beta, &[1, c, 1, 1]);
            let mean_r = graph.reshape(mean, &[1, c, 1, 1]);
            let var_r = graph.reshape(variance, &[1, c, 1, 1]);

            let out = graph.batch_norm(input, mean_r, var_r, gamma_r, beta_r, epsilon);
            Some(vec![(node.outputs[0].clone(), out, in_shape)])
        }

        "Concat" => {
            let axis = get_attr_int(node, "axis").unwrap_or(1);
            let first_shape = get_shape(&node.inputs[0]);
            let ndim = first_shape.len() as i64;
            let ax = if axis < 0 { ndim + axis } else { axis } as usize;

            // Validate: all input shapes must match except at concat axis
            let mut input_refs = Vec::new();
            let mut shapes = Vec::new();
            for name in &node.inputs {
                if name.is_empty() {
                    continue;
                }
                input_refs.push(get(name)?);
                shapes.push(get_shape(name));
            }
            for s in &shapes[1..] {
                if s.len() != shapes[0].len() {
                    return None;
                }
                for (d, (&a, &b)) in shapes[0].iter().zip(s.iter()).enumerate() {
                    if d != ax && a != b {
                        return None;
                    }
                }
            }

            let out = graph.concat(&input_refs, axis);

            let mut out_shape = first_shape.clone();
            let total_axis: usize = shapes.iter().map(|s| s.get(ax).copied().unwrap_or(0)).sum();
            if ax < out_shape.len() {
                out_shape[ax] = total_axis;
            }

            Some(vec![(node.outputs[0].clone(), out, out_shape)])
        }

        "Reshape" | "Flatten" => {
            let input = get(&node.inputs[0])?;
            let in_shape = get_shape(&node.inputs[0]);

            // For Flatten, compute shape from axis
            let target = if node.op_type == "Flatten" {
                let axis = get_attr_int(node, "axis").unwrap_or(1) as usize;
                let before: usize = in_shape.iter().take(axis).product();
                let after: usize = in_shape.iter().skip(axis).product();
                vec![before as i64, after as i64]
            } else {
                // Reshape: read target shape from the shape initializer
                if node.inputs.len() < 2 || node.inputs[1].is_empty() {
                    return None;
                }
                let shape_name = &node.inputs[1];
                if let Some(shape_tensor) = model.initializers.get(shape_name) {
                    shape_tensor.data().iter().map(|&v| v as i64).collect()
                } else if let Some(cv) = const_values.get(shape_name) {
                    cv.iter().map(|&v| v as i64).collect()
                } else {
                    return None; // Dynamic shape not supported
                }
            };

            // Resolve -1 and 0 in target shape
            let total: usize = in_shape.iter().product();
            let mut resolved = target.clone();
            let mut neg_idx = None;
            let mut known_product: i64 = 1;
            for i in 0..resolved.len() {
                let d = resolved[i];
                if d == -1 {
                    neg_idx = Some(i);
                } else if d == 0 {
                    resolved[i] = in_shape.get(i).copied().unwrap_or(1) as i64;
                    known_product *= resolved[i];
                } else {
                    known_product *= d;
                }
            }
            if let Some(idx) = neg_idx
                && known_product != 0
            {
                resolved[idx] = total as i64 / known_product;
            }

            let out_shape: Vec<usize> = resolved.iter().map(|&d| d as usize).collect();
            let out_total: usize = out_shape.iter().product();
            if out_total != total {
                return None; // Shape mismatch — bail
            }
            let out = graph.reshape(input, &resolved);
            Some(vec![(node.outputs[0].clone(), out, out_shape)])
        }

        "Transpose" => {
            let input = get(&node.inputs[0])?;
            let in_shape = get_shape(&node.inputs[0]);
            let perm = get_attr_ints(node, "perm").unwrap_or_default();

            if perm.is_empty() {
                return None; // Need explicit perm
            }

            // MPSGraph transpose only swaps two dims at a time.
            // For arbitrary permutations, we chain transpose operations.
            let perm_usize: Vec<usize> = perm.iter().map(|&p| p as usize).collect();
            let mut current = input;
            let mut current_perm: Vec<usize> = (0..in_shape.len()).collect();

            for target_pos in 0..perm_usize.len() {
                let target_dim = perm_usize[target_pos];
                let current_pos = current_perm.iter().position(|&d| d == target_dim);
                if let Some(pos) = current_pos
                    && pos != target_pos
                {
                    current = graph.transpose(current, target_pos, pos);
                    current_perm.swap(target_pos, pos);
                }
            }

            // Compute output shape
            let out_shape: Vec<usize> = perm_usize.iter().map(|&p| in_shape[p]).collect();
            Some(vec![(node.outputs[0].clone(), current, out_shape)])
        }

        "Softmax" => {
            let input = get(&node.inputs[0])?;
            let in_shape = get_shape(&node.inputs[0]);
            let axis = get_attr_int(node, "axis").unwrap_or(-1);
            let ndim = in_shape.len() as i64;
            let ax = if axis < 0 { ndim + axis } else { axis };
            let out = graph.softmax(input, ax);
            Some(vec![(node.outputs[0].clone(), out, in_shape)])
        }

        "Resize" => {
            let input = get(&node.inputs[0])?;
            let in_shape = get_shape(&node.inputs[0]);
            if in_shape.len() != 4 {
                return None;
            }
            let (n, c, ih, iw) = (in_shape[0], in_shape[1], in_shape[2], in_shape[3]);

            // Helper to read f32 data from initializers or const_values
            let read_data = |name: &str| -> Option<Vec<f32>> {
                model
                    .initializers
                    .get(name)
                    .map(|t| t.data().to_vec())
                    .or_else(|| const_values.get(name).cloned())
            };

            // Try sizes first (input index 3), then scales (input index 2)
            let (oh, ow) = if node.inputs.len() > 3 && !node.inputs[3].is_empty() {
                if let Some(d) = read_data(&node.inputs[3]) {
                    if d.len() >= 4 {
                        (d[2] as usize, d[3] as usize)
                    } else {
                        return None;
                    }
                } else {
                    return None;
                }
            } else if node.inputs.len() > 2 && !node.inputs[2].is_empty() {
                if let Some(d) = read_data(&node.inputs[2]) {
                    if d.len() >= 4 {
                        ((ih as f32 * d[2]) as usize, (iw as f32 * d[3]) as usize)
                    } else {
                        return None;
                    }
                } else {
                    return None;
                }
            } else {
                return None;
            };

            let out = graph.resize_nearest(input, oh, ow);
            Some(vec![(node.outputs[0].clone(), out, vec![n, c, oh, ow])])
        }

        "MatMul" => {
            let a = get(&node.inputs[0])?;
            let b = get(&node.inputs[1])?;
            let a_shape = get_shape(&node.inputs[0]);
            let b_shape = get_shape(&node.inputs[1]);
            let out = graph.matmul(a, b);

            // Output shape: [..., M, N] from [..., M, K] x [..., K, N]
            let mut out_shape = a_shape.clone();
            if let Some(last) = b_shape.last()
                && let Some(s) = out_shape.last_mut()
            {
                *s = *last;
            }
            Some(vec![(node.outputs[0].clone(), out, out_shape)])
        }

        "Split" => {
            let input = get(&node.inputs[0])?;
            let in_shape = get_shape(&node.inputs[0]);
            let axis = get_attr_int(node, "axis").unwrap_or(0);
            let ndim = in_shape.len() as i64;
            let ax = if axis < 0 { ndim + axis } else { axis } as usize;

            let num_outputs = node.outputs.len();

            // Read split sizes: try attribute first (opset < 13), then input tensor (opset 13+)
            let split_sizes = get_attr_ints(node, "split").or_else(|| {
                if node.inputs.len() > 1 && !node.inputs[1].is_empty() {
                    let name = &node.inputs[1];
                    if let Some(t) = model.initializers.get(name) {
                        Some(t.data().iter().map(|&v| v as i64).collect())
                    } else {
                        const_values
                            .get(name)
                            .map(|cv| cv.iter().map(|&v| v as i64).collect())
                    }
                } else {
                    None
                }
            });

            let sizes: Vec<usize> = if let Some(splits) = split_sizes {
                splits.iter().map(|&s| s as usize).collect()
            } else {
                // Equal split
                let dim = in_shape.get(ax).copied().unwrap_or(0);
                let per = dim / num_outputs;
                vec![per; num_outputs]
            };

            let mut results = Vec::new();
            let mut offset = 0usize;
            for (out_idx, &size) in sizes.iter().enumerate() {
                if out_idx >= node.outputs.len() {
                    break;
                }
                // Build slice: starts and ends for each dim
                let mut starts: Vec<i64> = vec![0; in_shape.len()];
                let mut ends: Vec<i64> = in_shape.iter().map(|&d| d as i64).collect();
                let strides: Vec<i64> = vec![1; in_shape.len()];
                starts[ax] = offset as i64;
                ends[ax] = (offset + size) as i64;

                let sliced = graph.slice(input, &starts, &ends, &strides);
                let mut out_shape = in_shape.clone();
                out_shape[ax] = size;
                results.push((node.outputs[out_idx].clone(), sliced, out_shape));
                offset += size;
            }

            Some(results)
        }

        "Slice" => {
            let input = get(&node.inputs[0])?;
            let in_shape = get_shape(&node.inputs[0]);
            let rank = in_shape.len();

            // Read starts, ends, axes, steps from initializers or const_values
            let read_i64 = |idx: usize| -> Option<Vec<i64>> {
                if idx >= node.inputs.len() || node.inputs[idx].is_empty() {
                    return None;
                }
                let name = &node.inputs[idx];
                if let Some(t) = model.initializers.get(name) {
                    Some(t.data().iter().map(|&v| v as i64).collect())
                } else {
                    const_values
                        .get(name)
                        .map(|cv| cv.iter().map(|&v| v as i64).collect())
                }
            };

            let starts_raw = read_i64(1)?;
            let ends_raw = read_i64(2)?;
            let axes_raw = read_i64(3);
            let steps_raw = read_i64(4);

            let axes: Vec<usize> = if let Some(a) = axes_raw {
                a.iter()
                    .map(|&v| {
                        if v < 0 {
                            (rank as i64 + v) as usize
                        } else {
                            v as usize
                        }
                    })
                    .collect()
            } else {
                (0..starts_raw.len()).collect()
            };

            let mut starts = vec![0i64; rank];
            let mut ends: Vec<i64> = in_shape.iter().map(|&d| d as i64).collect();
            let mut steps = vec![1i64; rank];

            for (i, &ax) in axes.iter().enumerate() {
                starts[ax] = starts_raw[i];
                ends[ax] = ends_raw[i];
                if let Some(ref st) = steps_raw {
                    steps[ax] = st[i];
                }
            }

            // Normalize negative indices and clamp
            for d in 0..rank {
                let dim = in_shape[d] as i64;
                if starts[d] < 0 {
                    starts[d] += dim;
                }
                if ends[d] < 0 {
                    ends[d] += dim;
                }
                starts[d] = starts[d].max(0).min(dim);
                ends[d] = ends[d].max(0).min(dim);
            }

            let out_shape: Vec<usize> = (0..rank)
                .map(|d| ((ends[d] - starts[d]) as f64 / steps[d] as f64).ceil() as usize)
                .collect();

            let out = graph.slice(input, &starts, &ends, &steps);
            Some(vec![(node.outputs[0].clone(), out, out_shape)])
        }

        "Cast" => {
            // In our f16 graph, Cast is a no-op (everything is already f16).
            // Just pass through the tensor.
            let input = get(&node.inputs[0])?;
            let in_shape = get_shape(&node.inputs[0]);
            Some(vec![(node.outputs[0].clone(), input, in_shape)])
        }

        "Unsqueeze" => {
            let input = get(&node.inputs[0])?;
            let in_shape = get_shape(&node.inputs[0]);

            // Read axes from attribute (opset < 13) or input tensor (opset 13+)
            let axes = get_attr_ints(node, "axes")
                .or_else(|| {
                    if node.inputs.len() > 1 && !node.inputs[1].is_empty() {
                        let name = &node.inputs[1];
                        model
                            .initializers
                            .get(name)
                            .map(|t| t.data().iter().map(|&v| v as i64).collect())
                            .or_else(|| {
                                const_values
                                    .get(name)
                                    .map(|cv| cv.iter().map(|&v| v as i64).collect())
                            })
                    } else {
                        None
                    }
                })
                .unwrap_or_default();

            let mut out_shape = in_shape.clone();
            // Insert dimensions in sorted descending order to keep indices valid
            let mut sorted_axes: Vec<i64> = axes;
            let out_rank = in_shape.len() + sorted_axes.len();
            // Normalize negative axes
            for ax in sorted_axes.iter_mut() {
                if *ax < 0 {
                    *ax += out_rank as i64;
                }
            }
            sorted_axes.sort_unstable();
            for &ax in sorted_axes.iter().rev() {
                out_shape.insert(ax as usize, 1);
            }

            let shape_i64: Vec<i64> = out_shape.iter().map(|&d| d as i64).collect();
            let out = graph.reshape(input, &shape_i64);
            Some(vec![(node.outputs[0].clone(), out, out_shape)])
        }

        "Squeeze" => {
            let input = get(&node.inputs[0])?;
            let in_shape = get_shape(&node.inputs[0]);

            let axes = get_attr_ints(node, "axes").or_else(|| {
                if node.inputs.len() > 1 && !node.inputs[1].is_empty() {
                    let name = &node.inputs[1];
                    model
                        .initializers
                        .get(name)
                        .map(|t| t.data().iter().map(|&v| v as i64).collect())
                        .or_else(|| {
                            const_values
                                .get(name)
                                .map(|cv| cv.iter().map(|&v| v as i64).collect())
                        })
                } else {
                    None
                }
            });

            let mut out_shape = Vec::new();
            let rank = in_shape.len() as i64;
            if let Some(axes) = axes {
                let ax_set: std::collections::HashSet<usize> = axes
                    .iter()
                    .map(|&a| {
                        if a < 0 {
                            (rank + a) as usize
                        } else {
                            a as usize
                        }
                    })
                    .collect();
                for (i, &d) in in_shape.iter().enumerate() {
                    if !ax_set.contains(&i) || d != 1 {
                        out_shape.push(d);
                    }
                }
            } else {
                // Remove all dimensions of size 1
                for &d in &in_shape {
                    if d != 1 {
                        out_shape.push(d);
                    }
                }
            }

            let shape_i64: Vec<i64> = out_shape.iter().map(|&d| d as i64).collect();
            let out = graph.reshape(input, &shape_i64);
            Some(vec![(node.outputs[0].clone(), out, out_shape)])
        }

        "Expand" => {
            let input = get(&node.inputs[0])?;
            let in_shape = get_shape(&node.inputs[0]);

            // Read target shape from second input (initializer or const_values)
            let target_shape: Vec<usize> = if let Some(t) = model.initializers.get(&node.inputs[1])
            {
                t.data().iter().map(|&v| v as usize).collect()
            } else if let Some(cv) = const_values.get(&node.inputs[1]) {
                cv.iter().map(|&v| v as usize).collect()
            } else {
                // No static target shape available — pass through
                return Some(vec![(node.outputs[0].clone(), input, in_shape)]);
            };

            // Compute broadcast output shape (NumPy-style)
            let ndim = in_shape.len().max(target_shape.len());
            let mut out_shape = vec![1usize; ndim];
            for i in 0..ndim {
                let a = if i < ndim - in_shape.len() {
                    1
                } else {
                    in_shape[i - (ndim - in_shape.len())]
                };
                let b = if i < ndim - target_shape.len() {
                    1
                } else {
                    target_shape[i - (ndim - target_shape.len())]
                };
                out_shape[i] = a.max(b);
            }

            // Use MPSGraph broadcast: reshape input to match ndim, then broadcast via mul with ones
            let shape_i64: Vec<i64> = out_shape.iter().map(|&d| d as i64).collect();
            let ones_data: Vec<u16> = vec![f32_to_f16_bits(1.0); out_shape.iter().product()];
            let ones = graph.constant_f16(&ones_data, &out_shape);
            let out = graph.mul(input, ones);
            Some(vec![(node.outputs[0].clone(), out, out_shape)])
        }

        _ => None, // Unsupported op
    }
}

/// Convert f32 to f16 bits (same as metal_conv::f32_to_f16).
fn f32_to_f16_bits(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = (bits >> 16) & 0x8000;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let man = bits & 0x7FFFFF;
    if exp == 255 {
        return (sign | 0x7C00 | if man != 0 { 0x200 } else { 0 }) as u16;
    }
    let new_exp = exp - 127 + 15;
    if new_exp >= 31 {
        return (sign | 0x7C00) as u16;
    }
    if new_exp <= 0 {
        if new_exp < -10 {
            return sign as u16;
        }
        let man_with_hidden = man | 0x800000;
        let shift = (1 - new_exp) as u32;
        let half_man = man_with_hidden >> (13 + shift);
        let round_bit = (man_with_hidden >> (12 + shift)) & 1;
        let sticky = man_with_hidden & ((1 << (12 + shift)) - 1);
        let round_up = round_bit != 0 && (sticky != 0 || (half_man & 1) != 0);
        return (sign | (half_man + round_up as u32)) as u16;
    }
    let truncated = man >> 13;
    let round_bit = (man >> 12) & 1;
    let sticky = man & 0xFFF;
    let round_up = round_bit != 0 && (sticky != 0 || (truncated & 1) != 0);
    let result = sign | ((new_exp as u32) << 10) | truncated;
    (result + round_up as u32) as u16
}
