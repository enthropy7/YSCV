//! MPSGraph-based whole-model inference.
//! Builds an MPSGraph from the ONNX model, compiles it once,
//! then executes as a single GPU dispatch — no per-op encoder transitions.

use std::cell::{Cell, RefCell};
use std::collections::HashMap;

use metal::*;

use yscv_kernels::KernelError;
use yscv_kernels::metal_backend::mpsgraph::{
    Conv2dDesc, MpsGraph, MpsGraphExecutable, MpsGraphTensorRef, Pool2dDesc, PreparedInputs,
};
use yscv_tensor::Tensor;

use crate::error::OnnxError;
use crate::loader::{OnnxAttribute, OnnxModel, OnnxNode};
use crate::runner::{get_attr_float, get_attr_int, get_attr_ints};

/// Graph-level description of one input: name, f32 placeholder, shape,
/// element count. Shared across all pipeline slots — only the backing
/// buffer differs per slot.
struct InputDesc {
    name: String,
    placeholder: MpsGraphTensorRef,
    shape: Vec<usize>,
    elems: usize,
}

/// Per-slot backing buffer for one input.
struct InputSlot {
    buf: Buffer,
}

/// One pipeline slot: a complete, independent set of buffers + NSArrays
/// for a single in-flight inference. The plan holds N slots (default 3 for
/// triple-buffering); `submit` picks the next one round-robin. While the
/// GPU works on slot k, the CPU can marshal slot (k+1) — no buffer
/// aliasing, no stalls.
struct PipelineSlot {
    input_slots: Vec<InputSlot>,
    prepared: PreparedInputs,
    output_bufs: Vec<Buffer>,
    /// Most recently committed command buffer for this slot, if any.
    /// Reused as a completion fence: `submit` waits on it before reusing
    /// this slot's buffers; `wait` consumes it to return outputs.
    in_flight: RefCell<Option<CommandBuffer>>,
}

/// Compiled MPSGraph plan for inference — triple-buffered pipelined.
///
/// Every allocation that can be lifted out of the hot path is lifted:
/// all per-slot MTL buffers, retained `MPSGraphTensorData` NSArrays, and
/// compile-time graph state. In steady state each `submit` is only a
/// handful of `memcpy`s + an `encodeToCommandBuffer` + `commit`; each
/// `wait` is `waitUntilCompleted` + NEON-widened f16→f32 readback.
///
/// Outputs stay in f16 end-to-end (no final GPU `cast_to_f32`); the CPU
/// widens during `wait` via aarch64 `vcvt_f32_f16` — 4 halves per
/// instruction, effectively free relative to GPU latency.
pub struct MpsGraphPlan {
    graph: MpsGraph,
    executable: MpsGraphExecutable,
    #[allow(dead_code)]
    device: Device,
    queue: CommandQueue,
    input_descs: Vec<InputDesc>,
    slots: Vec<PipelineSlot>,
    next_slot: Cell<usize>,
    output_names: Vec<String>,
    output_shapes: Vec<Vec<usize>>,
    output_bytes: Vec<usize>, // f16 bytes per output
}

/// Opaque handle returned by `submit_mpsgraph_plan`. Identifies which
/// pipeline slot's buffers hold (or will hold) the result. Pass to
/// `wait_mpsgraph_plan` to block until the GPU finishes and widen the
/// outputs into a `HashMap<String, Tensor>`.
#[must_use = "an InferenceHandle represents in-flight GPU work; wait on it or the next submit will back-pressure"]
pub struct InferenceHandle {
    slot_idx: usize,
}

impl MpsGraphPlan {
    pub fn ops_count(&self) -> usize {
        // Graph ops are opaque; return 0 (unknown)
        0
    }
}

/// Compile an ONNX model into an MPSGraph execution plan.
///
/// The graph operates in NCHW layout with f16 precision throughout — each user
/// input is declared as an f32 placeholder, then an on-GPU cast to f16 feeds
/// the rest of the graph. This avoids CPU-side f32→f16 conversion (Apple
/// Silicon's ALU does the cast for free on read).
///
/// `inputs` is a slice of `(name, tensor)` pairs. One entry per ONNX graph
/// input. Order is preserved in the returned plan; callers must pass the same
/// names to `run_mpsgraph_plan` (order-independent — lookup is by name).
pub fn compile_mpsgraph_plan(
    model: &OnnxModel,
    inputs: &[(&str, &Tensor)],
) -> Result<MpsGraphPlan, OnnxError> {
    if inputs.is_empty() {
        return Err(OnnxError::DecodeFailed {
            message: "compile_mpsgraph_plan: at least one input required".to_string(),
        });
    }

    let graph = MpsGraph::new().ok_or_else(|| OnnxError::DecodeFailed {
        message: "Failed to create MPSGraph".to_string(),
    })?;

    let device = Device::system_default().ok_or_else(|| OnnxError::DecodeFailed {
        message: "No Metal device available".to_string(),
    })?;
    let queue = device.new_command_queue();

    let mut tensors: HashMap<String, MpsGraphTensorRef> = HashMap::new();
    let mut cpu_shapes: HashMap<String, Vec<usize>> = HashMap::new();

    // One f32 placeholder + f16-cast per user input. These go into the graph
    // once; per-pipeline-slot backing buffers are allocated after compile.
    let mut input_descs: Vec<InputDesc> = Vec::with_capacity(inputs.len());
    let mut input_placeholders_f32: Vec<(MpsGraphTensorRef, Vec<usize>)> =
        Vec::with_capacity(inputs.len());
    for (name, tensor) in inputs {
        let shape = tensor.shape().to_vec();
        let placeholder_f32 = graph.placeholder_f32(&shape, name)?;
        let input_f16 = graph.cast_to_f16(placeholder_f32)?;
        tensors.insert((*name).to_string(), input_f16);
        cpu_shapes.insert((*name).to_string(), shape.clone());

        let elems: usize = shape.iter().product();
        input_descs.push(InputDesc {
            name: (*name).to_string(),
            placeholder: placeholder_f32,
            shape: shape.clone(),
            elems,
        });
        input_placeholders_f32.push((placeholder_f32, shape));
    }

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
            let t = graph.constant_f16(&f16_data, &final_shape)?;
            tensors.insert(name.clone(), t);
            cpu_shapes.insert(name.clone(), final_shape);
            continue;
        }

        let f16_data: Vec<u16> = final_data.iter().map(|&v| f32_to_f16_bits(v)).collect();
        let t = graph.constant_f16(&f16_data, &final_shape)?;
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
    #[cfg(feature = "profile")]
    let debug = std::env::var("MPSGRAPH_DEBUG").is_ok();
    #[cfg(not(feature = "profile"))]
    let debug = false;
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
                let t = graph.constant_f16(&f16_data, &out_shape)?;
                tensors.insert(node.outputs[0].clone(), t);
                cpu_shapes.insert(node.outputs[0].clone(), out_shape);
                const_values.insert(node.outputs[0].clone(), values);
                continue;
            }
            "Constant" => {
                // Extract f32 data + shape from the value attribute. Populate
                // BOTH the graph-constant tensor map (for math consumers) and
                // const_values (for compile-time shape consumers like Reshape).
                let (data, shape) = if let Some(OnnxAttribute::Tensor(t)) =
                    node.attributes.get("value")
                {
                    (t.data().to_vec(), t.shape().to_vec())
                } else if let Some(OnnxAttribute::Float(v)) =
                    node.attributes.get("value_float")
                {
                    (vec![*v], vec![1usize])
                } else if let Some(OnnxAttribute::Int(v)) = node.attributes.get("value_int")
                {
                    (vec![*v as f32], vec![1usize])
                } else if let Some(OnnxAttribute::Floats(v)) =
                    node.attributes.get("value_floats")
                {
                    let n = v.len();
                    (v.clone(), vec![n])
                } else if let Some(OnnxAttribute::Ints(v)) = node.attributes.get("value_ints")
                {
                    let n = v.len();
                    (v.iter().map(|&i| i as f32).collect(), vec![n])
                } else {
                    return Err(OnnxError::DecodeFailed {
                        message: format!(
                            "Constant node '{}' has no recognized value attribute",
                            node.name
                        ),
                    });
                };
                const_values.insert(node.outputs[0].clone(), data.clone());
                let stored_shape = if shape.is_empty() { vec![1usize] } else { shape };
                cpu_shapes.insert(node.outputs[0].clone(), stored_shape.clone());
                let f16_data: Vec<u16> = data.iter().map(|&v| f32_to_f16_bits(v)).collect();
                let t = graph.constant_f16(&f16_data, &stored_shape)?;
                tensors.insert(node.outputs[0].clone(), t);
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
                let t = graph.constant_f16(&f16_data, &out_shape)?;
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
                let t = graph.constant_f16(&f16_data, &shape)?;
                tensors.insert(inp_name.clone(), t);
            }
        }

        let result = build_graph_node(&graph, node, &tensors, &cpu_shapes, model, &const_values)?;

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
    // Outputs stay in f16 all the way to the shared output buffer. The
    // final f16→f32 widening happens on the CPU during Tensor construction
    // — roughly 1 µs per kilo-element with NEON `vcvt_f32_f16`, and skips
    // one GPU kernel + halves output writeback bandwidth.
    let mut output_tensors = Vec::new();
    let mut output_shapes = Vec::new();
    for out_name in &model.outputs {
        if let Some(&t) = tensors.get(out_name) {
            output_tensors.push((out_name.clone(), t));
            let shape = cpu_shapes.get(out_name).cloned().unwrap_or_default();
            output_shapes.push(shape);
        }
    }

    // Compile the graph — all user inputs are f32 placeholders; the graph
    // casts to f16 internally on GPU.
    let feeds: Vec<(MpsGraphTensorRef, &[usize], u32)> = input_placeholders_f32
        .iter()
        .map(|(ph, shape)| {
            (
                *ph,
                shape.as_slice(),
                yscv_kernels::metal_backend::mpsgraph::MPS_DATA_TYPE_FLOAT32,
            )
        })
        .collect();
    let target_refs: Vec<MpsGraphTensorRef> = output_tensors.iter().map(|(_, t)| *t).collect();

    let executable = graph
        .compile(&device, &feeds, &target_refs)?
        .ok_or_else(|| OnnxError::DecodeFailed {
            message: "Failed to compile MPSGraph".to_string(),
        })?;

    // --- Build N pipeline slots ---
    //
    // N defaults to 3 (triple-buffering). Override via `YSCV_MPS_PIPELINE`
    // env var; values < 1 are clamped to 1 (sync), > 8 to 8 (diminishing
    // returns on Apple Silicon's in-flight limit).
    let pipeline_depth: usize = std::env::var("YSCV_MPS_PIPELINE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(3)
        .clamp(1, 8);

    let output_bytes: Vec<usize> = output_shapes
        .iter()
        .map(|s| s.iter().product::<usize>() * 2) // f16 = 2 bytes/elem
        .collect();

    let mut slots: Vec<PipelineSlot> = Vec::with_capacity(pipeline_depth);
    for _ in 0..pipeline_depth {
        // Per-slot input buffers — one StorageModeShared Buffer per graph input.
        let slot_input_slots: Vec<InputSlot> = input_descs
            .iter()
            .map(|d| InputSlot {
                buf: device.new_buffer(
                    (d.elems * 4) as u64,
                    MTLResourceOptions::StorageModeShared,
                ),
            })
            .collect();
        // Per-slot output buffers (f16).
        let slot_output_bufs: Vec<Buffer> = output_bytes
            .iter()
            .map(|&b| device.new_buffer(b as u64, MTLResourceOptions::StorageModeShared))
            .collect();
        // Retained NSArrays of MPSGraphTensorData, wrapping this slot's buffers.
        let prepared_feeds: Vec<(MpsGraphTensorRef, &Buffer, &[usize], u32)> = input_descs
            .iter()
            .zip(slot_input_slots.iter())
            .map(|(d, s)| {
                (
                    d.placeholder,
                    &s.buf,
                    d.shape.as_slice(),
                    yscv_kernels::metal_backend::mpsgraph::MPS_DATA_TYPE_FLOAT32,
                )
            })
            .collect();
        let prepared_outputs: Vec<(&Buffer, &[usize], u32)> = slot_output_bufs
            .iter()
            .zip(output_shapes.iter())
            .map(|(buf, shape)| {
                (
                    buf,
                    shape.as_slice(),
                    yscv_kernels::metal_backend::mpsgraph::MPS_DATA_TYPE_FLOAT16,
                )
            })
            .collect();
        let prepared = graph.prepare_inputs(&prepared_feeds, &prepared_outputs)?;

        slots.push(PipelineSlot {
            input_slots: slot_input_slots,
            prepared,
            output_bufs: slot_output_bufs,
            in_flight: RefCell::new(None),
        });
    }

    let output_names: Vec<String> = output_tensors.iter().map(|(n, _)| n.clone()).collect();

    Ok(MpsGraphPlan {
        graph,
        executable,
        device,
        queue,
        input_descs,
        slots,
        next_slot: Cell::new(0),
        output_names,
        output_shapes,
        output_bytes,
    })
}

/// Submit a frame to the pipeline without waiting for GPU completion.
///
/// Picks the next slot round-robin. If that slot still has an in-flight
/// command buffer from a previous submit (because the caller has more
/// outstanding frames than the pipeline depth), **blocks** on its
/// completion before reusing its buffers — this is the back-pressure
/// mechanism that keeps buffer ownership safe without copies.
///
/// Returns an `InferenceHandle` that must be passed to
/// `wait_mpsgraph_plan` to retrieve outputs. Steady-state work per call:
/// `k` memcpys into that slot's input buffers, an encode-into-command-buffer,
/// a `commit()`. CPU returns immediately; GPU runs in the background.
pub fn submit_mpsgraph_plan(
    plan: &MpsGraphPlan,
    inputs: &[(&str, &[f32])],
) -> Result<InferenceHandle, OnnxError> {
    if inputs.len() != plan.input_descs.len() {
        return Err(OnnxError::DecodeFailed {
            message: format!(
                "submit_mpsgraph_plan: expected {} inputs, got {}",
                plan.input_descs.len(),
                inputs.len()
            ),
        });
    }

    // 1. Pick the next slot; bump ring index.
    let slot_idx = plan.next_slot.get();
    plan.next_slot.set((slot_idx + 1) % plan.slots.len());
    let slot = &plan.slots[slot_idx];

    // 2. If this slot is still in flight (ring went all the way around),
    //    wait for it. In steady state with pipeline_depth ≥ 3 and one
    //    outstanding handle at a time, this is a no-op; if the user
    //    over-submits, this is the safety net.
    if let Some(prev_cb) = slot.in_flight.borrow_mut().take() {
        prev_cb.wait_until_completed();
    }

    // 3. Copy fresh user data into this slot's input buffers.
    for (desc, in_slot) in plan.input_descs.iter().zip(slot.input_slots.iter()) {
        let data = inputs
            .iter()
            .find(|(n, _)| *n == desc.name.as_str())
            .map(|(_, d)| *d)
            .ok_or_else(|| OnnxError::DecodeFailed {
                message: format!("submit_mpsgraph_plan: missing input '{}'", desc.name),
            })?;
        if data.len() != desc.elems {
            return Err(OnnxError::DecodeFailed {
                message: format!(
                    "submit_mpsgraph_plan: input '{}' expects {} f32s, got {}",
                    desc.name,
                    desc.elems,
                    data.len()
                ),
            });
        }
        // SAFETY: `in_slot.buf` is StorageModeShared, sized `desc.elems*4`;
        // length verified above. GPU casts f32→f16 on read.
        #[allow(unsafe_code)]
        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr() as *const u8,
                in_slot.buf.contents() as *mut u8,
                desc.elems * 4,
            );
        }
    }

    // 4. Encode + commit through the MPS command-buffer wrapper. Returns
    //    immediately — GPU work runs in the background until
    //    `wait_until_completed()` is later called on this CB.
    let cb = plan.queue.new_command_buffer();
    plan.graph
        .encode_and_commit_with_prepared(&plan.executable, cb, &slot.prepared)?;

    // 5. Stash the command buffer in the slot so next reuse waits on it.
    *slot.in_flight.borrow_mut() = Some(cb.to_owned());

    Ok(InferenceHandle { slot_idx })
}

/// Wait for the GPU work associated with `handle` and return its outputs.
///
/// Blocks on the slot's command buffer (`wait_until_completed` is a
/// Mach-semaphore sleep — CPU yields while GPU runs), then widens f16
/// output buffers into f32 `Tensor`s using aarch64 `vcvt_f32_f16`.
pub fn wait_mpsgraph_plan(
    plan: &MpsGraphPlan,
    handle: InferenceHandle,
) -> Result<HashMap<String, Tensor>, OnnxError> {
    let slot = &plan.slots[handle.slot_idx];

    // Drain the in-flight command buffer. `take` leaves `None` so a
    // subsequent `submit` to this slot won't double-wait. If a later
    // submit re-enters this slot, it allocates a fresh CB.
    if let Some(cb) = slot.in_flight.borrow_mut().take() {
        cb.wait_until_completed();
    }

    // Widen f16 outputs → f32 Tensors.
    let mut result = HashMap::with_capacity(plan.output_names.len());
    for (idx, name) in plan.output_names.iter().enumerate() {
        let buf = &slot.output_bufs[idx];
        let shape = &plan.output_shapes[idx];
        let n: usize = plan.output_bytes[idx] / 2;
        // SAFETY: buf is StorageModeShared, sized `output_bytes[idx]`;
        // MPSGraph wrote exactly `n` f16 (u16) values before the CB
        // completed (verified by wait above).
        #[allow(unsafe_code)]
        let f16_slice =
            unsafe { std::slice::from_raw_parts(buf.contents() as *const u16, n) };
        let data = f16_slice_to_f32_vec(f16_slice);
        let tensor =
            Tensor::from_vec(shape.clone(), data).map_err(|e| OnnxError::DecodeFailed {
                message: e.to_string(),
            })?;
        result.insert(name.clone(), tensor);
    }

    Ok(result)
}

/// Synchronous wrapper — submits, waits, returns outputs. Equivalent to
/// `let h = submit(...)?; wait(h)` but one line.
///
/// For highest throughput, callers who can tolerate latency-of-N-frames
/// should use `submit_mpsgraph_plan` + `wait_mpsgraph_plan` directly and
/// keep the pipeline saturated.
pub fn run_mpsgraph_plan(
    plan: &MpsGraphPlan,
    inputs: &[(&str, &[f32])],
) -> Result<HashMap<String, Tensor>, OnnxError> {
    let handle = submit_mpsgraph_plan(plan, inputs)?;
    wait_mpsgraph_plan(plan, handle)
}

/// Widen an f16 (IEEE-754 binary16) slice to f32. Uses aarch64 NEON
/// `vcvt_f32_f16` (4 halves → 4 floats per instruction) on Apple Silicon;
/// falls back to a scalar bit-pattern conversion elsewhere.
#[inline]
fn f16_slice_to_f32_vec(src: &[u16]) -> Vec<f32> {
    let n = src.len();
    let mut dst: Vec<f32> = Vec::with_capacity(n);
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    #[allow(unsafe_code)]
    unsafe {
        use std::arch::aarch64::{vcvt_f32_f16, vld1_u16, vreinterpret_f16_u16, vst1q_f32};
        let dst_ptr = dst.as_mut_ptr();
        let chunks = n / 4;
        for i in 0..chunks {
            let v = vld1_u16(src.as_ptr().add(i * 4));
            let f = vcvt_f32_f16(vreinterpret_f16_u16(v));
            vst1q_f32(dst_ptr.add(i * 4), f);
        }
        // Tail.
        for i in (chunks * 4)..n {
            *dst_ptr.add(i) = f16_bits_to_f32_scalar(*src.as_ptr().add(i));
        }
        dst.set_len(n);
        return dst;
    }
    #[allow(unreachable_code)]
    {
        dst.extend(src.iter().map(|&b| f16_bits_to_f32_scalar(b)));
        dst
    }
}

/// Scalar IEEE-754 binary16 → f32 conversion. Used for the NEON tail and
/// as the fallback on non-aarch64 hosts.
#[inline]
fn f16_bits_to_f32_scalar(bits: u16) -> f32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let exp = (bits >> 10) & 0x1F;
    let mant = (bits & 0x03FF) as u32;
    if exp == 0 {
        if mant == 0 {
            return f32::from_bits(sign);
        }
        // Subnormal: normalize into f32 range.
        let mut e = 0i32;
        let mut m = mant;
        while m & 0x0400 == 0 {
            m <<= 1;
            e += 1;
        }
        let f32_exp = ((127 - 15 - e) as u32) << 23;
        let f32_man = (m & 0x03FF) << 13;
        return f32::from_bits(sign | f32_exp | f32_man);
    }
    if exp == 31 {
        // Inf or NaN.
        let f32_bits = sign | 0x7F80_0000 | if mant != 0 { 0x0040_0000 } else { 0 };
        return f32::from_bits(f32_bits);
    }
    let f32_exp = ((exp as u32).wrapping_add(112)) << 23;
    let f32_man = mant << 13;
    f32::from_bits(sign | f32_exp | f32_man)
}

/// Build a single ONNX node as MPSGraph operations.
/// Returns Ok(None) if the op is unsupported.
/// Returns Ok(Some(vec of (output_name, tensor_ref, shape))) on success.
fn build_graph_node(
    graph: &MpsGraph,
    node: &OnnxNode,
    tensors: &HashMap<String, MpsGraphTensorRef>,
    shapes: &HashMap<String, Vec<usize>>,
    model: &OnnxModel,
    const_values: &HashMap<String, Vec<f32>>,
) -> Result<Option<Vec<(String, MpsGraphTensorRef, Vec<usize>)>>, KernelError> {
    // Helper: return Ok(None) when an input tensor is not found (unsupported path)
    macro_rules! try_get {
        ($name:expr) => {
            match tensors.get($name).copied() {
                Some(v) => v,
                None => return Ok(None),
            }
        };
    }
    let get_shape = |name: &str| -> Vec<usize> { shapes.get(name).cloned().unwrap_or_default() };

    match node.op_type.as_str() {
        "Conv" => {
            let input = try_get!(&node.inputs[0]);
            let weight = try_get!(&node.inputs[1]);
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

            let mut out = graph.conv2d(input, weight, &desc)?;

            // Add bias if present
            if node.inputs.len() > 2
                && !node.inputs[2].is_empty()
                && let Some(bias) = tensors.get(&node.inputs[2]).copied()
            {
                // Reshape bias [C] → [1, C, 1, 1] for NCHW broadcast
                let bias_reshaped = graph.reshape(bias, &[1, o_ch as i64, 1, 1])?;
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

            Ok(Some(vec![(
                node.outputs[0].clone(),
                out,
                vec![n, o_ch, oh, ow],
            )]))
        }

        "Relu" => {
            let input = try_get!(&node.inputs[0]);
            let shape = get_shape(&node.inputs[0]);
            let out = graph.relu(input);
            Ok(Some(vec![(node.outputs[0].clone(), out, shape)]))
        }

        "Identity" => {
            // Pass-through: output gets the same tensor ref as input.
            // No GPU op — MPSGraph tensor refs are SSA-style handles so
            // aliasing is semantically correct.
            let input = try_get!(&node.inputs[0]);
            let shape = get_shape(&node.inputs[0]);
            Ok(Some(vec![(node.outputs[0].clone(), input, shape)]))
        }


        "Sigmoid" => {
            let input = try_get!(&node.inputs[0]);
            let shape = get_shape(&node.inputs[0]);
            let out = graph.sigmoid(input);
            Ok(Some(vec![(node.outputs[0].clone(), out, shape)]))
        }

        "Exp" => {
            let input = try_get!(&node.inputs[0]);
            let shape = get_shape(&node.inputs[0]);
            let out = graph.exp(input);
            Ok(Some(vec![(node.outputs[0].clone(), out, shape)]))
        }

        "Add" | "Sub" | "Mul" | "Div" => {
            let a = try_get!(&node.inputs[0]);
            let b = try_get!(&node.inputs[1]);
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
            Ok(Some(vec![(node.outputs[0].clone(), out, out_shape)]))
        }

        "MaxPool" => {
            let input = try_get!(&node.inputs[0]);
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
            let out = graph.max_pool2d(input, &desc)?;

            let (n, c, ih, iw) = (in_shape[0], in_shape[1], in_shape[2], in_shape[3]);
            let oh = (ih + pads[0] as usize + pads[2] as usize - kernel_shape[0] as usize)
                / strides[0] as usize
                + 1;
            let ow = (iw + pads[1] as usize + pads[3] as usize - kernel_shape[1] as usize)
                / strides[1] as usize
                + 1;

            Ok(Some(vec![(
                node.outputs[0].clone(),
                out,
                vec![n, c, oh, ow],
            )]))
        }

        "AveragePool" => {
            let input = try_get!(&node.inputs[0]);
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
            let out = graph.avg_pool2d(input, &desc)?;

            let (n, c, ih, iw) = (in_shape[0], in_shape[1], in_shape[2], in_shape[3]);
            let oh = (ih + pads[0] as usize + pads[2] as usize - kernel_shape[0] as usize)
                / strides[0] as usize
                + 1;
            let ow = (iw + pads[1] as usize + pads[3] as usize - kernel_shape[1] as usize)
                / strides[1] as usize
                + 1;

            Ok(Some(vec![(
                node.outputs[0].clone(),
                out,
                vec![n, c, oh, ow],
            )]))
        }

        "GlobalAveragePool" => {
            let input = try_get!(&node.inputs[0]);
            let in_shape = get_shape(&node.inputs[0]);
            let out = graph.global_avg_pool(input)?;
            let (n, c) = (in_shape[0], in_shape[1]);
            Ok(Some(vec![(node.outputs[0].clone(), out, vec![n, c, 1, 1])]))
        }

        "BatchNormalization" => {
            let input = try_get!(&node.inputs[0]);
            let gamma = try_get!(&node.inputs[1]);
            let beta = try_get!(&node.inputs[2]);
            let mean = try_get!(&node.inputs[3]);
            let variance = try_get!(&node.inputs[4]);
            let epsilon = get_attr_float(node, "epsilon").unwrap_or(1e-5);
            let in_shape = get_shape(&node.inputs[0]);

            // Reshape gamma/beta/mean/var from [C] → [1, C, 1, 1] for NCHW broadcast
            let c = in_shape.get(1).copied().unwrap_or(1) as i64;
            let gamma_r = graph.reshape(gamma, &[1, c, 1, 1])?;
            let beta_r = graph.reshape(beta, &[1, c, 1, 1])?;
            let mean_r = graph.reshape(mean, &[1, c, 1, 1])?;
            let var_r = graph.reshape(variance, &[1, c, 1, 1])?;

            let out = graph.batch_norm(input, mean_r, var_r, gamma_r, beta_r, epsilon);
            Ok(Some(vec![(node.outputs[0].clone(), out, in_shape)]))
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
                input_refs.push(try_get!(name));
                shapes.push(get_shape(name));
            }
            for s in &shapes[1..] {
                if s.len() != shapes[0].len() {
                    return Ok(None);
                }
                for (d, (&a, &b)) in shapes[0].iter().zip(s.iter()).enumerate() {
                    if d != ax && a != b {
                        return Ok(None);
                    }
                }
            }

            let out = graph.concat(&input_refs, axis)?;

            let mut out_shape = first_shape.clone();
            let total_axis: usize = shapes.iter().map(|s| s.get(ax).copied().unwrap_or(0)).sum();
            if ax < out_shape.len() {
                out_shape[ax] = total_axis;
            }

            Ok(Some(vec![(node.outputs[0].clone(), out, out_shape)]))
        }

        "Reshape" | "Flatten" => {
            let input = try_get!(&node.inputs[0]);
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
                    return Ok(None);
                }
                let shape_name = &node.inputs[1];
                if let Some(shape_tensor) = model.initializers.get(shape_name) {
                    shape_tensor.data().iter().map(|&v| v as i64).collect()
                } else if let Some(cv) = const_values.get(shape_name) {
                    cv.iter().map(|&v| v as i64).collect()
                } else {
                    return Ok(None); // Dynamic shape not supported
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
                return Ok(None); // Shape mismatch — bail
            }
            let out = graph.reshape(input, &resolved)?;
            Ok(Some(vec![(node.outputs[0].clone(), out, out_shape)]))
        }

        "Transpose" => {
            let input = try_get!(&node.inputs[0]);
            let in_shape = get_shape(&node.inputs[0]);
            let perm = get_attr_ints(node, "perm").unwrap_or_default();

            if perm.is_empty() {
                return Ok(None); // Need explicit perm
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
            Ok(Some(vec![(node.outputs[0].clone(), current, out_shape)]))
        }

        "Softmax" => {
            let input = try_get!(&node.inputs[0]);
            let in_shape = get_shape(&node.inputs[0]);
            let axis = get_attr_int(node, "axis").unwrap_or(-1);
            let ndim = in_shape.len() as i64;
            let ax = if axis < 0 { ndim + axis } else { axis };

            // Run attention softmax in f32 to avoid f16 exp() underflow/overflow
            let is_attn = node.name.contains("/attn/");
            let out = if is_attn {
                let input_f32 = graph.cast_to_f32(input)?;
                let result_f32 = graph.softmax(input_f32, ax);
                graph.cast_to_f16(result_f32)?
            } else {
                graph.softmax(input, ax)
            };
            Ok(Some(vec![(node.outputs[0].clone(), out, in_shape)]))
        }

        "Resize" => {
            let input = try_get!(&node.inputs[0]);
            let in_shape = get_shape(&node.inputs[0]);
            if in_shape.len() != 4 {
                return Ok(None);
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
                        return Ok(None);
                    }
                } else {
                    return Ok(None);
                }
            } else if node.inputs.len() > 2 && !node.inputs[2].is_empty() {
                if let Some(d) = read_data(&node.inputs[2]) {
                    if d.len() >= 4 {
                        ((ih as f32 * d[2]) as usize, (iw as f32 * d[3]) as usize)
                    } else {
                        return Ok(None);
                    }
                } else {
                    return Ok(None);
                }
            } else {
                return Ok(None);
            };

            let out = graph.resize_nearest(input, oh, ow)?;
            Ok(Some(vec![(
                node.outputs[0].clone(),
                out,
                vec![n, c, oh, ow],
            )]))
        }

        "MatMul" => {
            let a = try_get!(&node.inputs[0]);
            let b = try_get!(&node.inputs[1]);
            let a_shape = get_shape(&node.inputs[0]);
            let b_shape = get_shape(&node.inputs[1]);

            // Run attention MatMul in f32 to avoid f16 accumulation errors
            let is_attn = node.name.contains("/attn/");
            let out = if is_attn {
                let a_f32 = graph.cast_to_f32(a)?;
                let b_f32 = graph.cast_to_f32(b)?;
                let result_f32 = graph.matmul(a_f32, b_f32);
                graph.cast_to_f16(result_f32)?
            } else {
                graph.matmul(a, b)
            };

            // Output shape: [..., M, N] from [..., M, K] x [..., K, N]
            let mut out_shape = a_shape.clone();
            if let Some(last) = b_shape.last()
                && let Some(s) = out_shape.last_mut()
            {
                *s = *last;
            }
            Ok(Some(vec![(node.outputs[0].clone(), out, out_shape)]))
        }

        "Split" => {
            let input = try_get!(&node.inputs[0]);
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

                let sliced = graph.slice(input, &starts, &ends, &strides)?;
                let mut out_shape = in_shape.clone();
                out_shape[ax] = size;
                results.push((node.outputs[out_idx].clone(), sliced, out_shape));
                offset += size;
            }

            Ok(Some(results))
        }

        "Slice" => {
            let input = try_get!(&node.inputs[0]);
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

            let starts_raw = match read_i64(1) {
                Some(v) => v,
                None => return Ok(None),
            };
            let ends_raw = match read_i64(2) {
                Some(v) => v,
                None => return Ok(None),
            };
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

            let out = graph.slice(input, &starts, &ends, &steps)?;
            Ok(Some(vec![(node.outputs[0].clone(), out, out_shape)]))
        }

        "Cast" => {
            // In our f16 graph, Cast is a no-op (everything is already f16).
            // Just pass through the tensor.
            let input = try_get!(&node.inputs[0]);
            let in_shape = get_shape(&node.inputs[0]);
            Ok(Some(vec![(node.outputs[0].clone(), input, in_shape)]))
        }

        "Unsqueeze" => {
            let input = try_get!(&node.inputs[0]);
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
            let out = graph.reshape(input, &shape_i64)?;
            Ok(Some(vec![(node.outputs[0].clone(), out, out_shape)]))
        }

        "Squeeze" => {
            let input = try_get!(&node.inputs[0]);
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
            let out = graph.reshape(input, &shape_i64)?;
            Ok(Some(vec![(node.outputs[0].clone(), out, out_shape)]))
        }

        "Expand" => {
            let input = try_get!(&node.inputs[0]);
            let in_shape = get_shape(&node.inputs[0]);

            // Read target shape from second input (initializer or const_values)
            let target_shape: Vec<usize> = if let Some(t) = model.initializers.get(&node.inputs[1])
            {
                t.data().iter().map(|&v| v as usize).collect()
            } else if let Some(cv) = const_values.get(&node.inputs[1]) {
                cv.iter().map(|&v| v as usize).collect()
            } else {
                // No static target shape available — pass through
                return Ok(Some(vec![(node.outputs[0].clone(), input, in_shape)]));
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
            let ones = graph.constant_f16(&ones_data, &out_shape)?;
            let out = graph.mul(input, ones);
            Ok(Some(vec![(node.outputs[0].clone(), out, out_shape)]))
        }

        _ => Ok(None), // Unsupported op
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
