use ::metal::*;
use std::collections::{HashMap, HashSet};

use super::record::{record_conv, record_node};
use super::types::*;

use crate::runner::{TensorEnv, execute_node_with_layout};

use yscv_kernels::metal_backend::metal_conv::MetalInference;

use crate::error::OnnxError;
use crate::loader::{OnnxModel, OnnxNode};

/// Compile a Metal execution plan for the given ONNX model.
/// Runs a shape-inference pass on CPU, then pre-allocates Metal buffers
/// and records dispatch operations. Does its own fusion (Conv+SiLU, Op+Relu, etc.)
/// without depending on the wgpu-based GpuExecPlan.
pub fn compile_metal_plan(
    model: &OnnxModel,
    input_name: &str,
    input_tensor: &yscv_tensor::Tensor,
) -> Result<MetalPlan, OnnxError> {
    let inf = MetalInference::new().ok_or_else(|| OnnxError::DecodeFailed {
        message: "Metal device not available".to_string(),
    })?;

    // Run on CPU to get all tensor shapes and intermediate data
    #[cfg(feature = "profile")]
    let debug_metal = std::env::var("METAL_DEBUG").is_ok();
    #[cfg(not(feature = "profile"))]
    let debug_metal = false;
    let mut env = TensorEnv::from_model(model);
    env.insert(input_name.to_string(), input_tensor.clone());
    // We need tensor shapes AND data for fallback ops. Some ops (Split) consume
    // their inputs, so we snapshot shapes + data for fallback-eligible outputs
    // immediately after each node executes.
    let mut cpu_shapes: HashMap<String, Vec<usize>> = HashMap::new();
    let mut cpu_data: HashMap<String, Vec<f32>> = HashMap::new();
    for (ni, node) in model.nodes.iter().enumerate() {
        if let Err(e) = execute_node_with_layout(node, &mut env)
            && debug_metal
        {
            eprintln!(
                "  [metal] CPU pass node {} {} '{}' FAILED: {}",
                ni, node.op_type, node.name, e
            );
        }
        // Snapshot outputs that Metal will need for cpu_fallback
        for out_name in &node.outputs {
            if out_name.is_empty() {
                continue;
            }
            if let Some(t) = env.get(out_name) {
                cpu_shapes.insert(out_name.clone(), t.shape().to_vec());
                // Only save data for cpu_fallback-eligible ops (shape ops, etc.)
                // to avoid excessive memory usage.
                // Save data for any op that might need cpu_fallback
                // (shape ops, unknown ops, etc.) — limit to small tensors to save memory
                let n_elem = t.len();
                if n_elem <= 1_000_000 {
                    // ~4MB limit per tensor
                    cpu_data.insert(out_name.clone(), t.data().to_vec());
                }
            }
        }
    }

    // Now we know all tensor shapes. Build Metal buffers and ops.
    let mut bufs: HashMap<String, Buffer> = HashMap::new();
    let mut buf_shapes: HashMap<String, Vec<usize>> = HashMap::new();
    let mut buf_nhwc: HashMap<String, bool> = HashMap::new();
    let mut ops = Vec::new();
    // Track buffers that are in f32 format (for attention precision chain)
    let mut f32_bufs: HashSet<String> = HashSet::new();

    // Upload input: for 4D NCHW, do f32→f16 + NCHW→NHWC on CPU (avoids GPU cast op).
    let input_shape = input_tensor.shape();
    let input_n = input_tensor.data().len();
    let input_upload = if input_shape.len() == 4 {
        let (n_dim, c, h, w) = (
            input_shape[0],
            input_shape[1],
            input_shape[2],
            input_shape[3],
        );
        // Allocate f16 NHWC buffer — Shared because CPU writes at runtime
        let input_f16_buf = inf.output_buffer_f16_shared(input_n);
        bufs.insert(input_name.to_string(), input_f16_buf);
        buf_shapes.insert(input_name.to_string(), vec![n_dim, h, w, c]);
        buf_nhwc.insert(input_name.to_string(), true);
        // No GPU cast op needed — CPU handles it
        InputUploadMode::CpuCastNchwToNhwc {
            batch: n_dim,
            c,
            h,
            w,
        }
    } else {
        // Non-4D: keep f32 buffer + GPU cast
        let input_buf = inf.output_buffer(input_n);
        let input_f32_name = format!("{}_f32", input_name);
        bufs.insert(input_f32_name.clone(), input_buf);
        buf_shapes.insert(input_f32_name.clone(), input_shape.to_vec());
        buf_nhwc.insert(input_f32_name.clone(), false);
        let input_f16_buf = inf.output_buffer_f16(input_n);
        bufs.insert(input_name.to_string(), input_f16_buf);
        buf_shapes.insert(input_name.to_string(), input_shape.to_vec());
        buf_nhwc.insert(input_name.to_string(), false);
        ops.push(MetalOp::CastF32ToF16 {
            input: input_f32_name.clone(),
            out: input_name.to_string(),
            n: input_n as u32,
        });
        InputUploadMode::F32GpuCast
    };

    // Upload all initializer weights as f16 (halves bandwidth for non-conv weights)
    for (name, tensor) in &model.initializers {
        let data = tensor.data();
        let buf = inf.buffer_from_f32_as_f16(data);
        bufs.insert(name.clone(), buf);
        buf_shapes.insert(name.clone(), tensor.shape().to_vec());
        buf_nhwc.insert(name.clone(), false);
    }

    let nodes = &model.nodes;

    // Build a set of node indices to skip (fused into a previous node)
    let mut skip: std::collections::HashSet<usize> = std::collections::HashSet::new();

    #[cfg(feature = "profile")]
    let debug_metal = std::env::var("METAL_DEBUG").is_ok();
    #[cfg(not(feature = "profile"))]
    let debug_metal = false;

    // Walk graph and record Metal ops with inline fusion
    for (i, node) in nodes.iter().enumerate() {
        if skip.contains(&i) {
            if debug_metal {
                eprintln!("  [metal] skip node {} {} '{}'", i, node.op_type, node.name);
            }
            continue;
        }
        if debug_metal {
            eprintln!(
                "  [metal] node {} {} '{}' inputs={:?} outputs={:?}",
                i, node.op_type, node.name, node.inputs, node.outputs
            );
        }

        // --- Conv → Sigmoid → Mul  (SiLU fusion) ---
        // Widened look-ahead: Sigmoid may be up to 5 nodes after Conv (detection head
        // has parallel branches that interleave). Only skip the matched Sigmoid+Mul,
        // not intermediate nodes from other branches.
        if node.op_type == "Conv" {
            let conv_out = &node.outputs[0];
            let mut fused_silu = false;
            'sig_search: for sig_look in 0..=5 {
                let sig_idx = i + 1 + sig_look;
                if let Some(sig) = nodes.get(sig_idx)
                    && sig.op_type == "Sigmoid"
                    && sig.inputs.len() == 1
                    && sig.inputs[0] == *conv_out
                {
                    let sig_out = &sig.outputs[0];
                    for mul_look in 1..=5 {
                        let mul_idx = sig_idx + mul_look;
                        if let Some(mul) = nodes.get(mul_idx)
                            && mul.op_type == "Mul"
                            && mul.inputs.len() == 2
                            && ((mul.inputs[0] == *sig_out && mul.inputs[1] == *conv_out)
                                || (mul.inputs[1] == *sig_out && mul.inputs[0] == *conv_out))
                        {
                            let mul_out = &mul.outputs[0];
                            record_conv(
                                &inf,
                                node,
                                &env,
                                &mut bufs,
                                &mut buf_shapes,
                                &mut buf_nhwc,
                                &mut ops,
                                2,
                            )?;
                            if let Some(buf) = bufs.remove(conv_out) {
                                bufs.insert(mul_out.clone(), buf);
                                let shape = buf_shapes.remove(conv_out).unwrap_or_default();
                                buf_shapes.insert(mul_out.clone(), shape);
                                buf_nhwc.insert(mul_out.clone(), true);
                                match ops.last_mut() {
                                    Some(MetalOp::ConvGemm { output, .. })
                                    | Some(MetalOp::ConvDirect { output, .. })
                                    | Some(MetalOp::DepthwiseConv { output, .. })
                                    | Some(MetalOp::MpsConv { output, .. })
                                    | Some(MetalOp::ConvWinograd { output, .. }) => {
                                        *output = mul_out.clone();
                                    }
                                    _ => {}
                                }
                            }
                            // Only skip the specific Sigmoid and Mul nodes
                            skip.insert(sig_idx);
                            skip.insert(mul_idx);

                            // --- Check for Add after Mul (Conv+SiLU+Add residual) ---
                            for add_look in 1..=3 {
                                let add_idx = mul_idx + add_look;
                                if let Some(add_node) = nodes.get(add_idx)
                                    && add_node.op_type == "Add"
                                    && add_node.inputs.len() == 2
                                {
                                    let res_name = if add_node.inputs[0] == *mul_out {
                                        Some(&add_node.inputs[1])
                                    } else if add_node.inputs[1] == *mul_out {
                                        Some(&add_node.inputs[0])
                                    } else {
                                        None
                                    };
                                    if let Some(rn) = res_name {
                                        let conv_shape = buf_shapes.get(mul_out.as_str());
                                        let res_shape = buf_shapes.get(rn.as_str());
                                        let shapes_match = conv_shape == res_shape
                                            && bufs.contains_key(rn.as_str());
                                        if shapes_match {
                                            let fused_add = match ops.last_mut() {
                                                Some(MetalOp::ConvWinograd {
                                                    residual,
                                                    output,
                                                    ..
                                                }) => {
                                                    *residual = Some(rn.clone());
                                                    let add_out = &add_node.outputs[0];
                                                    if let Some(buf) = bufs.remove(mul_out.as_str())
                                                    {
                                                        bufs.insert(add_out.clone(), buf);
                                                        let shape = buf_shapes
                                                            .remove(mul_out.as_str())
                                                            .unwrap_or_default();
                                                        buf_shapes.insert(add_out.clone(), shape);
                                                        buf_nhwc.insert(add_out.clone(), true);
                                                        *output = add_out.clone();
                                                    }
                                                    true
                                                }
                                                Some(MetalOp::ConvGemm {
                                                    residual,
                                                    output,
                                                    params,
                                                    ..
                                                }) => {
                                                    params.has_residual = 1;
                                                    *residual = Some(rn.clone());
                                                    let add_out = &add_node.outputs[0];
                                                    if let Some(buf) = bufs.remove(mul_out.as_str())
                                                    {
                                                        bufs.insert(add_out.clone(), buf);
                                                        let shape = buf_shapes
                                                            .remove(mul_out.as_str())
                                                            .unwrap_or_default();
                                                        buf_shapes.insert(add_out.clone(), shape);
                                                        buf_nhwc.insert(add_out.clone(), true);
                                                        *output = add_out.clone();
                                                    }
                                                    true
                                                }
                                                _ => false,
                                            };
                                            if fused_add {
                                                skip.insert(add_idx);
                                                if debug_metal {
                                                    eprintln!(
                                                        "  [metal] Fused Conv+SiLU+Add residual: {} + {}",
                                                        mul_out, rn
                                                    );
                                                }
                                            }
                                        }
                                        break; // found Add match
                                    }
                                }
                            }

                            fused_silu = true;
                            break 'sig_search;
                        }
                    }
                }
            }
            if fused_silu {
                continue;
            }

            // --- Conv → Relu fusion ---
            if let Some(next) = nodes.get(i + 1)
                && next.op_type == "Relu"
                && next.inputs.len() == 1
                && next.inputs[0] == *conv_out
            {
                record_conv(
                    &inf,
                    node,
                    &env,
                    &mut bufs,
                    &mut buf_shapes,
                    &mut buf_nhwc,
                    &mut ops,
                    1,
                )?;
                let relu_out = &next.outputs[0];
                if let Some(buf) = bufs.remove(conv_out) {
                    bufs.insert(relu_out.clone(), buf);
                    let shape = buf_shapes.remove(conv_out).unwrap_or_default();
                    buf_shapes.insert(relu_out.clone(), shape);
                    buf_nhwc.insert(relu_out.clone(), true);
                    match ops.last_mut() {
                        Some(MetalOp::ConvGemm { output, .. })
                        | Some(MetalOp::ConvDirect { output, .. })
                        | Some(MetalOp::DepthwiseConv { output, .. })
                        | Some(MetalOp::MpsConv { output, .. })
                        | Some(MetalOp::ConvWinograd { output, .. }) => {
                            *output = relu_out.clone();
                        }
                        _ => {}
                    }
                }
                skip.insert(i + 1);
                continue;
            }

            // --- Conv → Add (residual connection, Winograd only) ---
            if let Some(next) = nodes.get(i + 1)
                && next.op_type == "Add"
                && next.inputs.len() == 2
            {
                let conv_out = &node.outputs[0];
                let residual_name = if next.inputs[0] == *conv_out {
                    Some(&next.inputs[1])
                } else if next.inputs[1] == *conv_out {
                    Some(&next.inputs[0])
                } else {
                    None
                };
                if let Some(res_name) = residual_name {
                    // Record the conv, then try to fuse the Add into it
                    record_conv(
                        &inf,
                        node,
                        &env,
                        &mut bufs,
                        &mut buf_shapes,
                        &mut buf_nhwc,
                        &mut ops,
                        0,
                    )?;
                    let fused = {
                        let conv_shape = buf_shapes.get(conv_out.as_str());
                        let res_shape = buf_shapes.get(res_name.as_str());
                        if debug_metal {
                            eprintln!(
                                "  [metal] Conv+Add candidate: conv_out={} res={} conv_shape={:?} res_shape={:?} res_buf={}",
                                conv_out,
                                res_name,
                                conv_shape,
                                res_shape,
                                bufs.contains_key(res_name.as_str())
                            );
                        }
                        let shapes_match =
                            conv_shape == res_shape && bufs.contains_key(res_name.as_str());
                        if shapes_match {
                            match ops.last_mut() {
                                Some(MetalOp::ConvWinograd {
                                    residual, output, ..
                                }) => {
                                    *residual = Some(res_name.clone());
                                    let add_out = &next.outputs[0];
                                    if let Some(buf) = bufs.remove(conv_out.as_str()) {
                                        bufs.insert(add_out.clone(), buf);
                                        let shape = buf_shapes
                                            .remove(conv_out.as_str())
                                            .unwrap_or_default();
                                        buf_shapes.insert(add_out.clone(), shape);
                                        buf_nhwc.insert(add_out.clone(), true);
                                        *output = add_out.clone();
                                    }
                                    true
                                }
                                Some(MetalOp::ConvGemm {
                                    residual,
                                    output,
                                    params,
                                    ..
                                }) => {
                                    params.has_residual = 1;
                                    *residual = Some(res_name.clone());
                                    let add_out = &next.outputs[0];
                                    if let Some(buf) = bufs.remove(conv_out.as_str()) {
                                        bufs.insert(add_out.clone(), buf);
                                        let shape = buf_shapes
                                            .remove(conv_out.as_str())
                                            .unwrap_or_default();
                                        buf_shapes.insert(add_out.clone(), shape);
                                        buf_nhwc.insert(add_out.clone(), true);
                                        *output = add_out.clone();
                                    }
                                    true
                                }
                                _ => false,
                            }
                        } else {
                            false
                        }
                    };
                    if fused {
                        skip.insert(i + 1);
                        if debug_metal {
                            eprintln!(
                                "  [metal] Fused Conv+Add residual: {} + {}",
                                conv_out, res_name
                            );
                        }
                    }
                    continue; // conv already recorded
                }
            }

            // Plain conv
            record_conv(
                &inf,
                node,
                &env,
                &mut bufs,
                &mut buf_shapes,
                &mut buf_nhwc,
                &mut ops,
                0,
            )?;
            continue;
        }

        // --- Sigmoid → Mul (standalone SiLU) ---
        if node.op_type == "Sigmoid" && node.inputs.len() == 1 {
            let sig_in = &node.inputs[0];
            let sig_out = &node.outputs[0];
            let mut fused = false;
            for look in 1..=3 {
                if let Some(mul) = nodes.get(i + look)
                    && mul.op_type == "Mul"
                    && mul.inputs.len() == 2
                    && ((mul.inputs[0] == *sig_out && mul.inputs[1] == *sig_in)
                        || (mul.inputs[1] == *sig_out && mul.inputs[0] == *sig_in))
                {
                    let mul_out = &mul.outputs[0];
                    ensure_on_metal(
                        &inf,
                        sig_in,
                        &env,
                        &mut bufs,
                        &mut buf_shapes,
                        &mut buf_nhwc,
                    );
                    if let Some(shape) = buf_shapes.get(sig_in) {
                        let n: usize = shape.iter().product();
                        bufs.insert(mul_out.clone(), inf.output_buffer_f16(n));
                        buf_shapes.insert(mul_out.clone(), shape.clone());
                        let nhwc_flag = *buf_nhwc.get(sig_in.as_str()).unwrap_or(&false);
                        buf_nhwc.insert(mul_out.clone(), nhwc_flag);
                        ops.push(MetalOp::SiLU {
                            input: sig_in.clone(),
                            out: mul_out.clone(),
                            n: n as u32,
                            f16: true,
                        });
                    }
                    // Only skip the Mul node, not intermediate nodes from other branches
                    skip.insert(i + look);
                    fused = true;
                    break;
                }
            }
            if fused {
                continue;
            }
        }

        // --- Normal node ---
        record_node(
            &inf,
            node,
            &env,
            &cpu_data,
            &cpu_shapes,
            &mut bufs,
            &mut buf_shapes,
            &mut buf_nhwc,
            &mut ops,
            &mut f32_bufs,
        )?;
    }

    let output_names = model.outputs.clone();

    // Add CastF16ToF32 for each output so the host can read f32
    // (skip if the output is already f32 from the attention chain)
    for out_name in &output_names {
        if let Some(shape) = buf_shapes.get(out_name) {
            let n: usize = shape.iter().product();
            if n > 0 {
                if f32_bufs.contains(out_name) {
                    // Output is already f32 — just alias to the f32out name
                    let f32_name = format!("{}_f32out", out_name);
                    let existing = bufs.get(out_name).unwrap().clone();
                    bufs.insert(f32_name.clone(), existing);
                    buf_shapes.insert(f32_name.clone(), shape.clone());
                    buf_nhwc.insert(f32_name.clone(), *buf_nhwc.get(out_name).unwrap_or(&false));
                } else {
                    let f32_name = format!("{}_f32out", out_name);
                    bufs.insert(f32_name.clone(), inf.output_buffer(n));
                    buf_shapes.insert(f32_name.clone(), shape.clone());
                    buf_nhwc.insert(f32_name.clone(), *buf_nhwc.get(out_name).unwrap_or(&false));
                    ops.push(MetalOp::CastF16ToF32 {
                        input: out_name.clone(),
                        out: f32_name,
                        n: n as u32,
                    });
                }
            }
        }
    }

    // ── Concat fusion: redirect conv outputs to write directly into concat buffer ──
    // For NHWC last-dim concat where ALL inputs come from ConvGemm/ConvDirect
    // with no other consumers, we set out_stride/out_offset on each conv so
    // it writes interleaved into the pre-allocated concat buffer, eliminating
    // the concat copy entirely.
    if std::env::var("METAL_NO_CONV_CONCAT").is_err() {
        // Map conv output name → op index (only for kernels with strided output support)
        let mut conv_out_idx: HashMap<&str, usize> = HashMap::new();
        for (i, op) in ops.iter().enumerate() {
            match op {
                MetalOp::ConvGemm { output, .. }
                | MetalOp::ConvDirect { output, .. }
                | MetalOp::ConvWinograd { output, .. } => {
                    conv_out_idx.insert(output.as_str(), i);
                }
                _ => {}
            }
        }

        // Count how many ops read each buffer
        let mut consumers: HashMap<&str, usize> = HashMap::new();
        for op in &ops {
            let inputs: Vec<&str> = match op {
                MetalOp::ConvGemm {
                    input,
                    weight,
                    bias,
                    residual,
                    ..
                } => {
                    let mut v: Vec<&str> = vec![input, weight, bias];
                    if let Some(r) = residual {
                        v.push(r.as_str());
                    }
                    v
                }
                MetalOp::ConvDirect {
                    input,
                    weight,
                    bias,
                    ..
                } => vec![input, weight, bias],
                MetalOp::DepthwiseConv {
                    input,
                    weight,
                    bias,
                    ..
                } => vec![input, weight, bias],
                MetalOp::Binary { a, b, .. } => vec![a, b],
                MetalOp::BroadcastBinary { a, b, .. } => vec![a, b],
                MetalOp::Unary { input, .. } | MetalOp::SiLU { input, .. } => vec![input],
                MetalOp::Concat { inputs, .. } => inputs.iter().map(|s| s.as_str()).collect(),
                MetalOp::Split { input, .. } | MetalOp::SplitFused { input, .. } => vec![input],
                MetalOp::MaxPool { input, .. } | MetalOp::Resize { input, .. } => vec![input],
                MetalOp::Softmax { input, .. } | MetalOp::Transpose2D { input, .. } => vec![input],
                MetalOp::CpuReshape { input, .. } | MetalOp::Permute0213 { input, .. } => {
                    vec![input]
                }
                MetalOp::SliceCopy { input, .. } => vec![input],
                MetalOp::FlatConcat { inputs, .. } => inputs.iter().map(|s| s.as_str()).collect(),
                MetalOp::MatMul { a, b, .. } => vec![a, b],
                MetalOp::CastF32ToF16 { input, .. } | MetalOp::CastF16ToF32 { input, .. } => {
                    vec![input]
                }
                MetalOp::ConvWinograd {
                    input,
                    weight,
                    bias,
                    residual,
                    ..
                } => {
                    let mut v: Vec<&str> = vec![input, weight, bias];
                    if let Some(r) = residual {
                        v.push(r.as_str());
                    }
                    v
                }
                MetalOp::MpsConv {
                    input,
                    weight,
                    bias,
                    ..
                } => vec![input, weight, bias],
                MetalOp::NhwcToFlatConcat { inputs, .. } => {
                    inputs.iter().map(|s| s.as_str()).collect()
                }
                MetalOp::ChannelScatter { input, .. } => vec![input],
            };
            for name in inputs {
                *consumers.entry(name).or_insert(0) += 1;
            }
        }

        // Collect fusable concats: (concat_op_idx, concat_out_name, vec<(conv_op_idx, ch_offset)>, total_c)
        let mut fusions: Vec<(usize, String, Vec<(usize, u32)>, u32)> = Vec::new();
        for (i, op) in ops.iter().enumerate() {
            if let MetalOp::Concat {
                inputs,
                channels,
                out,
                out_c,
                ..
            } = op
            {
                let all_fusable = !inputs.is_empty()
                    && inputs.iter().all(|name| {
                        conv_out_idx.contains_key(name.as_str())
                            && consumers.get(name.as_str()).copied().unwrap_or(0) == 1
                    });
                if !all_fusable {
                    continue;
                }
                let mut conv_ops = Vec::new();
                let mut offset = 0u32;
                for (j, in_name) in inputs.iter().enumerate() {
                    let cidx = conv_out_idx[in_name.as_str()];
                    conv_ops.push((cidx, offset));
                    offset += channels[j];
                }
                fusions.push((i, out.clone(), conv_ops, *out_c));
            }
        }

        // Apply fusions
        let mut removed_indices: HashSet<usize> = HashSet::new();
        for (concat_idx, concat_out, conv_ops, total_c) in &fusions {
            if let Some(concat_buf) = bufs.get(concat_out).cloned() {
                for &(conv_idx, ch_offset) in conv_ops {
                    match &mut ops[conv_idx] {
                        MetalOp::ConvGemm { output, params, .. }
                        | MetalOp::ConvDirect { output, params, .. } => {
                            params.out_stride = *total_c;
                            params.out_offset = ch_offset;
                            bufs.insert(output.clone(), concat_buf.clone());
                        }
                        MetalOp::ConvWinograd {
                            output,
                            wino_params,
                            ..
                        } => {
                            wino_params.out_stride = *total_c;
                            wino_params.out_offset = ch_offset;
                            bufs.insert(output.clone(), concat_buf.clone());
                        }
                        _ => unreachable!(),
                    }
                }
                removed_indices.insert(*concat_idx);
            }
        }

        if !removed_indices.is_empty() {
            let old_ops = std::mem::take(&mut ops);
            ops = old_ops
                .into_iter()
                .enumerate()
                .filter(|(i, _)| !removed_indices.contains(i))
                .map(|(_, op)| op)
                .collect();
            if debug_metal {
                eprintln!(
                    "  [metal] Fused {} concat ops into conv strided outputs",
                    removed_indices.len()
                );
            }
        }
    }

    // ── Detection head fusion: CpuReshape(NHWC→NCHW) + FlatConcat → NhwcToFlatConcat ──
    // Replaces separate NHWC→NCHW permutations + flat copy with a single fused kernel.
    if std::env::var("METAL_NO_FUSION").is_err() {
        // Map CpuReshape output name → (op_idx, nhwc_input_name, (n,h,w,c))
        let mut reshape_info: HashMap<&str, (usize, &str, (u32, u32, u32, u32))> = HashMap::new();
        for (i, op) in ops.iter().enumerate() {
            if let MetalOp::CpuReshape {
                input,
                out,
                nhwc_to_nchw: Some(nhwc_val),
                ..
            } = op
            {
                reshape_info.insert(out.as_str(), (i, input.as_str(), *nhwc_val));
            }
        }

        // Count consumers of each buffer
        let mut buf_consumers: HashMap<&str, usize> = HashMap::new();
        for op in &ops {
            let ins: Vec<&str> = match op {
                MetalOp::FlatConcat { inputs, .. } => inputs.iter().map(|s| s.as_str()).collect(),
                MetalOp::Concat { inputs, .. } => inputs.iter().map(|s| s.as_str()).collect(),
                MetalOp::NhwcToFlatConcat { inputs, .. } => {
                    inputs.iter().map(|s| s.as_str()).collect()
                }
                MetalOp::ConvGemm { input, .. } => vec![input.as_str()],
                MetalOp::ConvWinograd { input, .. } => vec![input.as_str()],
                MetalOp::Binary { a, b, .. } => vec![a.as_str(), b.as_str()],
                MetalOp::BroadcastBinary { a, b, .. } => vec![a.as_str(), b.as_str()],
                MetalOp::CpuReshape { input, .. } => vec![input.as_str()],
                MetalOp::SplitFused { input, .. } => vec![input.as_str()],
                MetalOp::Split { input, .. } => vec![input.as_str()],
                MetalOp::Unary { input, .. } | MetalOp::SiLU { input, .. } => {
                    vec![input.as_str()]
                }
                MetalOp::MatMul { a, b, .. } => vec![a.as_str(), b.as_str()],
                MetalOp::SliceCopy { input, .. } => vec![input.as_str()],
                MetalOp::MaxPool { input, .. } => vec![input.as_str()],
                MetalOp::Resize { input, .. } => vec![input.as_str()],
                MetalOp::Softmax { input, .. } => vec![input.as_str()],
                MetalOp::Transpose2D { input, .. } => vec![input.as_str()],
                MetalOp::Permute0213 { input, .. } => vec![input.as_str()],
                MetalOp::CastF32ToF16 { input, .. } | MetalOp::CastF16ToF32 { input, .. } => {
                    vec![input.as_str()]
                }
                MetalOp::ChannelScatter { input, .. } => vec![input.as_str()],
                _ => vec![],
            };
            for name in ins {
                *buf_consumers.entry(name).or_insert(0) += 1;
            }
        }

        let mut removed_indices: HashSet<usize> = HashSet::new();
        let mut replacements: Vec<(usize, MetalOp)> = Vec::new();

        for (i, op) in ops.iter().enumerate() {
            // Check both FlatConcat and regular Concat ops
            let (inputs, out, is_f16) = match op {
                MetalOp::FlatConcat {
                    inputs, out, f16, ..
                } => (inputs, out, *f16),
                MetalOp::Concat {
                    inputs, out, f16, ..
                } => (inputs, out, *f16),
                _ => continue,
            };
            {
                if !is_f16 || inputs.len() < 2 || inputs.len() > 3 {
                    continue;
                }
                // Check all inputs come from CpuReshape(NHWC→NCHW) with single consumer
                let mut all_fusable = true;
                let mut nhwc_inputs: Vec<String> = Vec::new();
                let mut hw_pairs: Vec<(u32, u32)> = Vec::new();
                let mut common_c: Option<u32> = None;
                let mut reshape_indices: Vec<usize> = Vec::new();

                for in_name in inputs {
                    if let Some(&(ridx, nhwc_input, (_n, h, w, c))) =
                        reshape_info.get(in_name.as_str())
                    {
                        // CpuReshape output must have single consumer (this concat)
                        if buf_consumers.get(in_name.as_str()).copied().unwrap_or(0) != 1 {
                            all_fusable = false;
                            break;
                        }
                        if let Some(cc) = common_c {
                            if cc != c {
                                all_fusable = false;
                                break;
                            }
                        } else {
                            common_c = Some(c);
                        }
                        nhwc_inputs.push(nhwc_input.to_string());
                        hw_pairs.push((h, w));
                        reshape_indices.push(ridx);
                    } else {
                        all_fusable = false;
                        break;
                    }
                }

                if !all_fusable || common_c.is_none() {
                    continue;
                }

                let c = common_c.unwrap();
                let total_spatial: u32 = hw_pairs.iter().map(|(h, w)| h * w).sum();

                // Mark FlatConcat and CpuReshape ops for removal
                removed_indices.insert(i);
                for ridx in &reshape_indices {
                    removed_indices.insert(*ridx);
                }

                // Allocate output buffer (reuse FlatConcat's output)
                let fused_op = MetalOp::NhwcToFlatConcat {
                    inputs: nhwc_inputs,
                    out: out.clone(),
                    c,
                    hw: hw_pairs.clone(),
                    total_spatial,
                };
                replacements.push((i, fused_op));

                if debug_metal {
                    eprintln!(
                        "  [metal] Fused {} CpuReshape + Concat into NhwcToFlatConcat (c={}, spatial={}, out={}, hw={:?})",
                        reshape_indices.len(),
                        c,
                        total_spatial,
                        out,
                        hw_pairs,
                    );
                }
            }
        }

        if !removed_indices.is_empty() {
            // Build new ops list: replace FlatConcat with fused op, remove CpuReshape ops
            let mut repl_map: HashMap<usize, MetalOp> = replacements.into_iter().collect();
            let old_ops = std::mem::take(&mut ops);
            ops = old_ops
                .into_iter()
                .enumerate()
                .filter_map(|(i, op)| {
                    if let Some(fused) = repl_map.remove(&i) {
                        Some(fused)
                    } else if removed_indices.contains(&i) {
                        None
                    } else {
                        Some(op)
                    }
                })
                .collect();
        }
    }

    // In-place optimization: for elementwise ops where an input buffer
    // is dead after this op, alias the output buffer to that input.
    // This saves both a buffer allocation and a full read+write round-trip.
    if std::env::var("METAL_NO_INPLACE").is_err() {
        // Compute last_use: buffer_name → last op index that reads it
        let mut last_use: HashMap<String, usize> = HashMap::new();
        for (i, op) in ops.iter().enumerate() {
            let inputs: Vec<&str> = match op {
                MetalOp::ConvGemm {
                    input,
                    weight,
                    bias,
                    residual,
                    ..
                } => {
                    let mut v: Vec<&str> = vec![input, weight, bias];
                    if let Some(r) = residual {
                        v.push(r.as_str());
                    }
                    v
                }
                MetalOp::ConvDirect {
                    input,
                    weight,
                    bias,
                    ..
                } => vec![input, weight, bias],
                MetalOp::DepthwiseConv {
                    input,
                    weight,
                    bias,
                    ..
                } => vec![input, weight, bias],
                MetalOp::Binary { a, b, .. } => vec![a, b],
                MetalOp::BroadcastBinary { a, b, .. } => vec![a, b],
                MetalOp::Unary { input, .. } => vec![input],
                MetalOp::SiLU { input, .. } => vec![input],
                MetalOp::Concat { inputs, .. } => inputs.iter().map(|s| s.as_str()).collect(),
                MetalOp::Split { input, .. } => vec![input],
                MetalOp::SplitFused { input, .. } => vec![input],
                MetalOp::MaxPool { input, .. } => vec![input],
                MetalOp::Resize { input, .. } => vec![input],
                MetalOp::Softmax { input, .. } => vec![input],
                MetalOp::Transpose2D { input, .. } => vec![input],
                MetalOp::CpuReshape { input, .. } => vec![input],
                MetalOp::Permute0213 { input, .. } => vec![input],
                MetalOp::SliceCopy { input, .. } => vec![input],
                MetalOp::FlatConcat { inputs, .. } => inputs.iter().map(|s| s.as_str()).collect(),
                MetalOp::MatMul { a, b, .. } => vec![a, b],
                MetalOp::CastF32ToF16 { input, .. } => vec![input],
                MetalOp::CastF16ToF32 { input, .. } => vec![input],
                MetalOp::ConvWinograd {
                    input,
                    weight,
                    bias,
                    residual,
                    ..
                } => {
                    let mut v: Vec<&str> = vec![input, weight, bias];
                    if let Some(r) = residual {
                        v.push(r.as_str());
                    }
                    v
                }
                MetalOp::MpsConv {
                    input,
                    weight,
                    bias,
                    ..
                } => vec![input, weight, bias],
                MetalOp::NhwcToFlatConcat { inputs, .. } => {
                    inputs.iter().map(|s| s.as_str()).collect()
                }
                MetalOp::ChannelScatter { input, .. } => vec![input],
            };
            for name in inputs {
                last_use.insert(name.to_string(), i);
            }
        }

        let mut aliased = 0usize;
        for (i, op) in ops.iter().enumerate() {
            match op {
                MetalOp::SiLU { input, out, .. } | MetalOp::Unary { input, out, .. } => {
                    let in_name = input.as_str();
                    if !in_name.starts_with("__mtl_wino")
                        && !in_name.contains("initializer")
                        && last_use.get(in_name) == Some(&i)
                        && let Some(buf) = bufs.get(in_name).cloned()
                    {
                        bufs.insert(out.clone(), buf);
                        aliased += 1;
                    }
                }
                _ => {}
            }
        }
        #[cfg(feature = "profile")]
        if aliased > 0 && std::env::var("METAL_DEBUG").is_ok() {
            eprintln!("  [metal] In-place aliased {} ops", aliased);
        }
        #[cfg(not(feature = "profile"))]
        let _ = aliased;
    }

    let input_buf_name = match &input_upload {
        InputUploadMode::CpuCastNchwToNhwc { .. } => input_name.to_string(),
        InputUploadMode::F32GpuCast => format!("{}_f32", input_name),
    };
    let cpu_ref = if std::env::var("METAL_COMPARE").is_ok() {
        cpu_data.clone()
    } else {
        HashMap::new()
    };
    Ok(MetalPlan {
        inf,
        ops,
        bufs,
        buf_shapes,
        buf_nhwc,
        input_buf_name,
        input_upload,
        output_names,
        cpu_ref,
        buf_f32: f32_bufs,
    })
}

// ── Helper: ensure tensor is on Metal ──

pub(crate) fn ensure_on_metal(
    inf: &MetalInference,
    name: &str,
    env: &TensorEnv,
    bufs: &mut HashMap<String, Buffer>,
    shapes: &mut HashMap<String, Vec<usize>>,
    nhwc: &mut HashMap<String, bool>,
) {
    if bufs.contains_key(name) {
        return;
    }
    if let Some(tensor) = env.get(name) {
        let data = tensor.data();
        bufs.insert(name.to_string(), inf.buffer_from_f32_as_f16(data));
        shapes.insert(name.to_string(), tensor.shape().to_vec());
        nhwc.insert(name.to_string(), false);
    }
}

// ── Ensure NHWC layout ──

/// Ensure tensor `name` is available in NHWC layout for Conv/Pool/Resize.
/// Returns the buffer name that contains NHWC data — callers must use this
/// name (not the original) for the op's input field.
///
/// If already NHWC, returns `name` unchanged.  Otherwise records a
/// NCHW→NHWC permutation into a *separate* buffer (`{name}_nhwc_perm`)
/// and returns that name.  The original `bufs[name]` is never aliased,
/// so earlier ops that write NCHW data to it are unaffected.
pub(crate) fn ensure_nhwc_metal(
    inf: &MetalInference,
    name: &str,
    bufs: &mut HashMap<String, Buffer>,
    shapes: &mut HashMap<String, Vec<usize>>,
    nhwc: &mut HashMap<String, bool>,
    ops: &mut Vec<MetalOp>,
) -> String {
    // Already NHWC natively (e.g. output of a previous Conv)
    if *nhwc.get(name).unwrap_or(&false) {
        return name.to_string();
    }

    // Already permuted by a previous ensure_nhwc_metal call (skip connection)
    let nhwc_name = format!("{}_nhwc_perm", name);
    if bufs.contains_key(&nhwc_name) {
        return nhwc_name;
    }

    let shape = match shapes.get(name) {
        Some(s) if s.len() == 4 => s.clone(),
        _ => return name.to_string(),
    };
    let (n, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
    let total = n * c * h * w;
    if !bufs.contains_key(name) {
        return name.to_string();
    }

    #[cfg(feature = "profile")]
    if std::env::var("METAL_PERM_DBG").is_ok() {
        eprintln!("  NHWC perm: '{}' shape={:?}", name, shape);
    }

    // Allocate a separate NHWC output buffer — do NOT alias bufs[name].
    bufs.insert(nhwc_name.clone(), inf.output_buffer_f16(total));
    shapes.insert(nhwc_name.clone(), vec![n, h, w, c]);

    // CpuReshape reads from bufs[name] (NCHW) and writes to bufs[nhwc_name] (NHWC).
    // At runtime both names resolve to distinct GPU buffers, so earlier ops that
    // write to bufs[name] are unaffected.
    ops.push(MetalOp::CpuReshape {
        input: name.to_string(),
        out: nhwc_name.clone(),
        n: total as u32,
        nhwc_to_nchw: None,
        nchw_to_nhwc: Some((n as u32, c as u32, h as u32, w as u32)),
        f16: true,
    });

    nhwc_name
}

/// Fallback: record a CPU op that will execute at runtime.
/// During compilation, allocate the output buffer using CPU-known shapes.
/// During execution, the CpuFallback op will download inputs, compute, and upload.
pub(crate) fn cpu_fallback(
    node: &OnnxNode,
    env: &TensorEnv,
    cpu_data: &HashMap<String, Vec<f32>>,
    cpu_shapes: &HashMap<String, Vec<usize>>,
    bufs: &mut HashMap<String, Buffer>,
    shapes: &mut HashMap<String, Vec<usize>>,
    nhwc: &mut HashMap<String, bool>,
    inf: &MetalInference,
) {
    for out_name in &node.outputs {
        if out_name.is_empty() {
            continue;
        }

        // Determine output shape from CPU snapshots
        let out_shape = if let Some(s) = cpu_shapes.get(out_name) {
            s.clone()
        } else if let Some(t) = env.get(out_name) {
            t.shape().to_vec()
        } else {
            continue;
        };

        let n: usize = out_shape.iter().product();
        if n == 0 {
            bufs.insert(out_name.clone(), inf.output_buffer_f16(1));
        } else {
            // For initializer data that doesn't depend on the runtime input,
            // we can use the pre-computed CPU data directly
            let is_runtime_dependent = node.inputs.iter().any(|inp| {
                !inp.is_empty()
                    && !env.get(inp).is_some_and(|_| {
                        // Check if input is a model initializer (constant)
                        false // conservative: assume all are runtime-dependent
                    })
            });

            if let Some(data) = cpu_data.get(out_name) {
                // Use the pre-computed data for now; runtime ops will overwrite
                bufs.insert(out_name.clone(), inf.buffer_from_f32_as_f16(data));
            } else {
                bufs.insert(out_name.clone(), inf.output_buffer_f16(n));
            }
        }
        shapes.insert(out_name.clone(), out_shape);
        nhwc.insert(out_name.clone(), false);
    }
}
