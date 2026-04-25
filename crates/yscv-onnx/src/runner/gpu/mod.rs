//! GPU-accelerated ONNX inference — device-resident tensor chaining.
//!
//! Keeps intermediate tensors on GPU between operations (no per-op sync).
//! Only downloads at CPU-fallback boundaries and for the final output.
//! On Apple Silicon unified memory, uploads/downloads are near-instant memcpy.

pub(super) mod f16;
pub use f16::{compile_gpu_plan_f16, run_compiled_gpu_f16_fused};

use super::*;
use conv::oihw_to_khwc_cout;

use yscv_kernels::{GpuBackend, GpuBuffer};

// ── GPU tensor with layout metadata ──────────────────────────────

/// A tensor living on the GPU with layout metadata.
struct GpuTensor {
    buf: GpuBuffer,
    /// True when data is in NHWC layout (spatial ops produce NHWC).
    nhwc: bool,
    /// True when the buffer contains f16 data (2 bytes per element).
    f16_io: bool,
}

/// GPU tensor cache — maps tensor names to device-resident buffers.
type GpuCache = HashMap<String, GpuTensor>;

/// Opaque weight cache that persists transformed weights on GPU across
/// inference runs. Create with `GpuWeightCache::new()` and pass to
/// `run_onnx_model_gpu_cached()`.
pub struct GpuWeightCache(GpuCache);
impl GpuWeightCache {
    pub fn new() -> Self {
        Self(HashMap::new())
    }
}

// ── Precomputed execution plan ──────────────────────────────────

/// Describes what to do at each node index during execution.
/// Fusion patterns consume multiple consecutive nodes into a single action;
/// the consumed followers are recorded as `Skip`.
#[derive(Debug, Clone)]
pub enum GpuExecAction {
    /// Conv → Sigmoid → Mul  fused into Conv+SiLU (act=2).
    /// Nodes at `sig_idx` and `mul_idx` are skipped.
    ConvSiLU {
        sig_idx: usize,
        mul_idx: usize,
        /// Output name from the Mul node (final output of the fused pattern).
        output_name: String,
    },
    /// Sigmoid → Mul  fused into a standalone SiLU.
    /// Node at `mul_idx` is skipped.
    SiLU {
        mul_idx: usize,
        /// The original input to Sigmoid (= x), used to apply silu(x).
        x_name: String,
        /// Output name from the Mul node.
        output_name: String,
    },
    /// Conv → BatchNorm → Relu  (3-node fusion).
    /// Nodes at `bn_idx` and `relu_idx` are skipped.
    ConvBnRelu {
        bn_idx: usize,
        relu_idx: usize,
        /// BatchNorm output name (intermediate).
        bn_output: String,
        /// Relu output name (final output of the fused pattern).
        relu_output: String,
    },
    /// <Op> → Relu  fused pair (covers Conv+Relu, BN+Relu, Gemm+Relu, Add+Relu).
    /// Node at `relu_idx` is skipped.
    OpRelu {
        relu_idx: usize,
        /// Output name of the first op (source for relu).
        op_output: String,
        /// Output name from the Relu node.
        relu_output: String,
    },
    /// MatMul → Add  fused pair.
    /// Node at `add_idx` is skipped.
    MatMulAdd { add_idx: usize },
    /// Execute a single node normally (no fusion).
    Normal,
    /// This node index is consumed by an earlier fusion; skip it entirely.
    Skip,
}

/// Precomputed execution plan for GPU inference.
///
/// Caches the tensor-lifetime map (`last_use`), fusion decisions, and the
/// per-node action. Because the model graph never changes between inference
/// runs, this plan can be computed once and reused alongside `GpuWeightCache`.
pub struct GpuExecPlan {
    /// For each tensor name, the last node index that consumes it.
    /// Model outputs are mapped to `usize::MAX` so they are never recycled.
    pub last_use: HashMap<String, usize>,
    /// Precomputed action for every node index.
    pub actions: Vec<GpuExecAction>,
    /// Precomputed per-node recycle lists: recycle_at[i] = tensor names to
    /// return to the buffer pool after executing node i.
    /// Eliminates O(gc_size) scan per node in the hot loop.
    pub recycle_at: Vec<Vec<String>>,
}

/// Build a `GpuExecPlan` from a model graph.  This performs the exact same
/// fusion pattern-matching as the inline loop, but stores the results so
/// they don't need to be recomputed on every inference call.
pub fn plan_gpu_execution(model: &OnnxModel) -> GpuExecPlan {
    let nodes = &model.nodes;

    // ── Precompute tensor last-use index ────────────────────────────
    let mut last_use: HashMap<String, usize> = HashMap::new();
    for (i, node) in nodes.iter().enumerate() {
        for inp in &node.inputs {
            if !inp.is_empty() {
                last_use.insert(inp.clone(), i);
            }
        }
    }
    for name in &model.outputs {
        last_use.insert(name.clone(), usize::MAX);
    }

    // ── Build per-node action vector (mirrors the fusion block) ─────
    let mut actions: Vec<GpuExecAction> = vec![GpuExecAction::Normal; nodes.len()];

    let mut i = 0;
    while i < nodes.len() {
        let node = &nodes[i];

        // Conv → Sigmoid → Mul → Conv+SiLU
        if node.op_type == "Conv"
            && let Some(sig) = nodes.get(i + 1)
            && sig.op_type == "Sigmoid"
            && sig.inputs.len() == 1
            && sig.inputs[0] == node.outputs[0]
            && let Some(mul) = nodes.get(i + 2)
            && mul.op_type == "Mul"
            && mul.inputs.len() == 2
            && ((mul.inputs[0] == node.outputs[0] && mul.inputs[1] == sig.outputs[0])
                || (mul.inputs[1] == node.outputs[0] && mul.inputs[0] == sig.outputs[0]))
        {
            actions[i] = GpuExecAction::ConvSiLU {
                sig_idx: i + 1,
                mul_idx: i + 2,
                output_name: mul.outputs[0].clone(),
            };
            actions[i + 1] = GpuExecAction::Skip;
            actions[i + 2] = GpuExecAction::Skip;
            i += 3;
            continue;
        }

        // Sigmoid → Mul → SiLU (scan forward for matching Mul, skip claimed nodes)
        if node.op_type == "Sigmoid" && node.inputs.len() == 1 {
            let mut silu_matched = false;
            for offset in 1..12 {
                let mul_idx = i + offset;
                // Skip nodes already claimed by earlier fusions
                if matches!(actions.get(mul_idx), Some(GpuExecAction::Skip)) {
                    continue;
                }
                if let Some(next) = nodes.get(mul_idx)
                    && next.op_type == "Mul"
                    && next.inputs.len() == 2
                    && ((next.inputs[0] == node.inputs[0] && next.inputs[1] == node.outputs[0])
                        || (next.inputs[1] == node.inputs[0] && next.inputs[0] == node.outputs[0]))
                {
                    actions[i] = GpuExecAction::SiLU {
                        mul_idx,
                        x_name: node.inputs[0].clone(),
                        output_name: next.outputs[0].clone(),
                    };
                    actions[mul_idx] = GpuExecAction::Skip;
                    // Advance by 1 only — don't skip intermediate Sigmoid nodes
                    i += 1;
                    silu_matched = true;
                    break;
                }
            }
            if silu_matched {
                continue;
            }
        }

        // Conv → BatchNorm → Relu
        if node.op_type == "Conv"
            && let Some(next) = nodes.get(i + 1)
            && next.op_type == "BatchNormalization"
            && !next.inputs.is_empty()
            && next.inputs[0] == node.outputs[0]
            && let Some(next2) = nodes.get(i + 2)
            && next2.op_type == "Relu"
            && next2.inputs.len() == 1
            && next2.inputs[0] == next.outputs[0]
        {
            actions[i] = GpuExecAction::ConvBnRelu {
                bn_idx: i + 1,
                relu_idx: i + 2,
                bn_output: next.outputs[0].clone(),
                relu_output: next2.outputs[0].clone(),
            };
            actions[i + 1] = GpuExecAction::Skip;
            actions[i + 2] = GpuExecAction::Skip;
            i += 3;
            continue;
        }

        // Conv + Relu
        if node.op_type == "Conv"
            && let Some(next) = nodes.get(i + 1)
            && next.op_type == "Relu"
            && next.inputs.len() == 1
            && next.inputs[0] == node.outputs[0]
        {
            actions[i] = GpuExecAction::OpRelu {
                relu_idx: i + 1,
                op_output: node.outputs[0].clone(),
                relu_output: next.outputs[0].clone(),
            };
            actions[i + 1] = GpuExecAction::Skip;
            i += 2;
            continue;
        }

        // BatchNorm + Relu
        if node.op_type == "BatchNormalization"
            && let Some(next) = nodes.get(i + 1)
            && next.op_type == "Relu"
            && next.inputs.len() == 1
            && next.inputs[0] == node.outputs[0]
        {
            actions[i] = GpuExecAction::OpRelu {
                relu_idx: i + 1,
                op_output: node.outputs[0].clone(),
                relu_output: next.outputs[0].clone(),
            };
            actions[i + 1] = GpuExecAction::Skip;
            i += 2;
            continue;
        }

        // Gemm + Relu
        if node.op_type == "Gemm"
            && let Some(next) = nodes.get(i + 1)
            && next.op_type == "Relu"
            && next.inputs.len() == 1
            && next.inputs[0] == node.outputs[0]
        {
            actions[i] = GpuExecAction::OpRelu {
                relu_idx: i + 1,
                op_output: node.outputs[0].clone(),
                relu_output: next.outputs[0].clone(),
            };
            actions[i + 1] = GpuExecAction::Skip;
            i += 2;
            continue;
        }

        // Add + Relu
        if node.op_type == "Add"
            && let Some(next) = nodes.get(i + 1)
            && next.op_type == "Relu"
            && next.inputs.len() == 1
            && next.inputs[0] == node.outputs[0]
        {
            actions[i] = GpuExecAction::OpRelu {
                relu_idx: i + 1,
                op_output: node.outputs[0].clone(),
                relu_output: next.outputs[0].clone(),
            };
            actions[i + 1] = GpuExecAction::Skip;
            i += 2;
            continue;
        }

        // MatMul + Add
        if node.op_type == "MatMul"
            && let Some(next) = nodes.get(i + 1)
            && next.op_type == "Add"
            && next.inputs.len() == 2
            && (next.inputs[0] == node.outputs[0] || next.inputs[1] == node.outputs[0])
        {
            actions[i] = GpuExecAction::MatMulAdd { add_idx: i + 1 };
            actions[i + 1] = GpuExecAction::Skip;
            i += 2;
            continue;
        }

        // No fusion — Normal
        // actions[i] is already Normal
        i += 1;
    }

    // ── Diagnostic: print action breakdown ─────────────────────────
    #[cfg(feature = "profile")]
    {
        let mut op_counts: HashMap<String, u32> = HashMap::new();
        let mut fusion_counts: HashMap<&str, u32> = HashMap::new();
        for (i, action) in actions.iter().enumerate() {
            match action {
                GpuExecAction::Skip => {}
                GpuExecAction::ConvSiLU { .. } => {
                    *fusion_counts.entry("ConvSiLU").or_default() += 1;
                }
                GpuExecAction::SiLU { .. } => {
                    *fusion_counts.entry("SiLU").or_default() += 1;
                }
                GpuExecAction::ConvBnRelu { .. } => {
                    *fusion_counts.entry("ConvBnRelu").or_default() += 1;
                }
                GpuExecAction::OpRelu { .. } => {
                    *fusion_counts.entry("OpRelu").or_default() += 1;
                }
                GpuExecAction::MatMulAdd { .. } => {
                    *fusion_counts.entry("MatMulAdd").or_default() += 1;
                }
                GpuExecAction::Normal => {
                    *op_counts.entry(nodes[i].op_type.clone()).or_default() += 1;
                }
            }
        }
        eprintln!("  ── GPU Plan ──");
        let mut ops: Vec<_> = op_counts.into_iter().collect();
        ops.sort_by_key(|entry| std::cmp::Reverse(entry.1));
        for (op, n) in &ops {
            eprintln!("    {:>4}x  {}", n, op);
        }
        let mut fus: Vec<_> = fusion_counts.into_iter().collect();
        fus.sort_by_key(|entry| std::cmp::Reverse(entry.1));
        for (f, n) in &fus {
            eprintln!("    {:>4}x  {} (fused)", n, f);
        }
        let total_actions: u32 =
            ops.iter().map(|(_, n)| n).sum::<u32>() + fus.iter().map(|(_, n)| n).sum::<u32>();
        eprintln!("    {:>4}  total actions (excluding skips)", total_actions);
    }

    // ── Precompute per-node recycle lists ───────────────────────────
    // For each non-weight tensor with a finite last_use, find the first
    // non-skip node at or after that position.  Since effective_pos >= i
    // for every non-skip node, the tensor is always dead by then.
    let mut recycle_at: Vec<Vec<String>> = vec![Vec::new(); nodes.len()];
    for (name, &lu) in &last_use {
        if lu >= nodes.len() || name.starts_with("__") {
            continue;
        }
        for j in lu..nodes.len() {
            if !matches!(actions[j], GpuExecAction::Skip) {
                recycle_at[j].push(name.clone());
                break;
            }
        }
    }

    GpuExecPlan {
        last_use,
        actions,
        recycle_at,
    }
}

/// Recycle a GpuTensor's buffer back to the pool for reuse.
fn recycle(gpu: &GpuBackend, gt: GpuTensor) {
    let (raw, len) = gt.buf.into_raw();
    gpu.return_output_buf(len, raw);
}

/// Insert a tensor into the cache, recycling any previous buffer with the same name.
#[allow(dead_code)]
fn gc_insert(gpu: &GpuBackend, gc: &mut GpuCache, name: String, gt: GpuTensor) {
    if let Some(old) = gc.insert(name, gt) {
        recycle(gpu, old);
    }
}

// ── Public entry point ───────────────────────────────────────────

/// Runs ONNX model inference with GPU acceleration using device-resident
/// tensor chaining.  Intermediate results stay on GPU between operations,
/// eliminating the per-op device sync that dominates naive GPU dispatch.
pub fn run_onnx_model_gpu(
    model: &OnnxModel,
    inputs: HashMap<String, Tensor>,
) -> Result<HashMap<String, Tensor>, OnnxError> {
    let gpu = GpuBackend::new().map_err(|e| OnnxError::DecodeFailed {
        message: format!("GPU init: {e}"),
    })?;
    run_onnx_model_gpu_with(&gpu, model, inputs)
}

/// Runs ONNX model inference on a pre-created GPU backend.
/// Reusing the same `GpuBackend` across runs avoids device init and
/// pipeline compilation overhead (~20-40ms per run).
pub fn run_onnx_model_gpu_with(
    gpu: &GpuBackend,
    model: &OnnxModel,
    inputs: HashMap<String, Tensor>,
) -> Result<HashMap<String, Tensor>, OnnxError> {
    run_onnx_model_gpu_cached(gpu, model, inputs, &mut GpuWeightCache::new(), None)
}

/// Runs ONNX model inference with a persistent weight cache.
/// Pass the same `weight_cache` across runs to avoid re-uploading
/// transformed weights on every inference call.
///
/// When `plan` is `Some`, the precomputed execution plan is used instead of
/// re-analyzing fusion patterns and tensor lifetimes on every call.
/// Build a plan once with [`plan_gpu_execution`] and reuse it.
pub fn run_onnx_model_gpu_cached(
    gpu: &GpuBackend,
    model: &OnnxModel,
    inputs: HashMap<String, Tensor>,
    weight_cache: &mut GpuWeightCache,
    plan: Option<&GpuExecPlan>,
) -> Result<HashMap<String, Tensor>, OnnxError> {
    // If no plan was provided, compute one on the fly (original behavior).
    let owned_plan;
    let plan = match plan {
        Some(p) => p,
        None => {
            owned_plan = plan_gpu_execution(model);
            &owned_plan
        }
    };

    let mut env = TensorEnv::from_model(model);
    // Start with cached weights + fresh gc for activations
    let mut gc: GpuCache = std::mem::take(&mut weight_cache.0);

    for (name, tensor) in &model.initializers {
        env.insert(name.clone(), tensor.clone());
    }
    for (name, tensor) in inputs {
        env.insert(name, tensor);
    }

    let nodes = &model.nodes;

    // ── Execute graph using precomputed actions ──────────────────
    let mut dispatch_count = 0u32;
    for (i, node) in nodes.iter().enumerate() {
        if matches!(plan.actions[i], GpuExecAction::Skip) {
            continue;
        }

        match &plan.actions[i] {
            GpuExecAction::Skip => unreachable!(),

            GpuExecAction::ConvSiLU { output_name, .. } => {
                exec_conv_act(gpu, node, &mut env, &mut gc, 2)?;
                if let Some(gt) = gc.remove(&node.outputs[0]) {
                    gc.insert(output_name.clone(), gt);
                }
            }

            GpuExecAction::SiLU {
                x_name,
                output_name,
                ..
            } => {
                to_gpu(gpu, x_name, &env, &mut gc);
                if let Some(gt) = gc.get(x_name.as_str()) {
                    let out = gpu.silu_on_device(&gt.buf);
                    let nhwc = gt.nhwc;
                    gc.insert(
                        output_name.clone(),
                        GpuTensor {
                            buf: out,
                            nhwc,
                            f16_io: false,
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
                dispatch(gpu, node, &mut env, &mut gc)?;
                dispatch(gpu, &nodes[*bn_idx], &mut env, &mut gc)?;
                fuse_relu(gpu, bn_output, relu_output, &mut env, &mut gc);
            }

            GpuExecAction::OpRelu {
                op_output,
                relu_output,
                ..
            } => {
                dispatch(gpu, node, &mut env, &mut gc)?;
                fuse_relu(gpu, op_output, relu_output, &mut env, &mut gc);
            }

            GpuExecAction::MatMulAdd { add_idx } => {
                dispatch(gpu, node, &mut env, &mut gc)?;
                dispatch(gpu, &nodes[*add_idx], &mut env, &mut gc)?;
            }

            GpuExecAction::Normal => {
                // Zero-copy reshape/unsqueeze/squeeze: consume buffer when input is last used here
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
                    let mut shape = gt.buf.shape().to_vec();
                    let total = gt.buf.len();
                    let nhwc = gt.nhwc;
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
                            for &ax in &sorted {
                                shape.insert(ax, 1);
                            }
                            shape
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
                                to_remove.sort_unstable_by(|a, b| b.cmp(a));
                                for ax in to_remove {
                                    if shape[ax] == 1 {
                                        shape.remove(ax);
                                    }
                                }
                            }
                            shape
                        }
                        _ => get_reshape_shape(node, &mut env, gpu, &mut gc, &shape, total)?,
                    };
                    let out = gpu.reshape_on_device(gt.buf, new_shape);
                    gc.insert(
                        node.outputs[0].clone(),
                        GpuTensor {
                            buf: out,
                            nhwc,
                            f16_io: false,
                        },
                    );
                } else {
                    dispatch(gpu, node, &mut env, &mut gc)?;
                }
            }
        }

        // ── Recycle consumed GPU buffers back to pool ─────────────
        for name in &plan.recycle_at[i] {
            if let Some(gt) = gc.remove(name) {
                recycle(gpu, gt);
            }
        }

        dispatch_count += 1;
        if dispatch_count.is_multiple_of(20) {
            gpu.flush();
        }
    }

    // ── Collect outputs ──────────────────────────────────────────
    let mut result = HashMap::new();
    for name in &model.outputs {
        to_cpu(gpu, name, &mut env, &mut gc)?;
        if let Some(t) = env.get(name) {
            result.insert(name.clone(), t.clone());
        } else {
            eprintln!("warning: ONNX output '{}' not found", name);
        }
    }

    // Return cached weights (prefixed with "__") to the weight cache.
    // Recycle non-weight activation buffers.
    let activation_keys: Vec<String> = gc
        .keys()
        .filter(|k| !k.starts_with("__"))
        .cloned()
        .collect();
    for k in activation_keys {
        if let Some(gt) = gc.remove(&k) {
            recycle(gpu, gt);
        }
    }
    weight_cache.0 = gc;

    Ok(result)
}

// ── Compiled execution plan ──────────────────────────────────────

/// Pre-compiled GPU execution plan with all bind groups and buffers pre-created.
/// After warmup, compile this plan and use `run_compiled_gpu` for near-zero
/// per-dispatch CPU overhead.
pub struct CompiledGpuPlan {
    /// Recorded dispatch sequence (bind groups reference pinned buffers).
    ops: Vec<yscv_kernels::RecordedOp>,
    /// Pinned activation buffers — kept alive so bind groups remain valid.
    /// Maps tensor name → GpuTensor.
    #[allow(dead_code)]
    pinned: GpuCache,
    /// Input tensor buffer (for writing new input data each run).
    input_buf: GpuBuffer,
    /// Output tensor names.
    output_names: Vec<String>,
    /// Output GpuBuffers + nhwc + f16_io flags (for download).
    output_bufs: Vec<(GpuBuffer, bool, bool)>,
}

impl CompiledGpuPlan {
    /// Number of recorded GPU operations.
    pub fn ops_count(&self) -> usize {
        self.ops.len()
    }

    /// Input buffer handle (for manual timing breakdown).
    pub fn input_buf(&self) -> &GpuBuffer {
        &self.input_buf
    }

    /// Reference to recorded ops (for manual replay in profiling).
    pub fn ops_ref(&self) -> &[yscv_kernels::RecordedOp] {
        &self.ops
    }
}

/// Compile a GPU execution plan by doing a recording run.
/// The model is executed once (with recording enabled), capturing all dispatches.
/// The returned `CompiledGpuPlan` holds pre-created bind groups and pinned buffers
/// that enable near-zero-overhead replay on subsequent runs.
///
/// Requirements:
/// - `weight_cache` must be populated (run at least one warmup inference first).
/// - `plan` must be the precomputed execution plan for this model.
pub fn compile_gpu_plan(
    gpu: &GpuBackend,
    model: &OnnxModel,
    plan: &GpuExecPlan,
    weight_cache: &mut GpuWeightCache,
    input_name: &str,
    input_tensor: &Tensor,
) -> Result<CompiledGpuPlan, OnnxError> {
    // Start recording all dispatches
    gpu.start_recording();

    let mut env = TensorEnv::from_model(model);
    let mut gc: GpuCache = std::mem::take(&mut weight_cache.0);

    for (name, tensor) in &model.initializers {
        env.insert(name.clone(), tensor.clone());
    }
    env.insert(input_name.to_string(), input_tensor.clone());

    let nodes = &model.nodes;

    // Execute graph with recording — same logic as run_onnx_model_gpu_cached
    // but WITHOUT recycling (so all buffers stay alive for bind group validity).
    let mut dispatch_count = 0u32;
    for (i, node) in nodes.iter().enumerate() {
        if matches!(plan.actions[i], GpuExecAction::Skip) {
            continue;
        }

        match &plan.actions[i] {
            GpuExecAction::Skip => unreachable!(),

            GpuExecAction::ConvSiLU { output_name, .. } => {
                exec_conv_act(gpu, node, &mut env, &mut gc, 2)?;
                if let Some(gt) = gc.remove(&node.outputs[0]) {
                    gc.insert(output_name.clone(), gt);
                }
            }

            GpuExecAction::SiLU {
                x_name,
                output_name,
                ..
            } => {
                to_gpu(gpu, x_name, &env, &mut gc);
                if let Some(gt) = gc.get(x_name.as_str()) {
                    let out = gpu.silu_on_device(&gt.buf);
                    let nhwc = gt.nhwc;
                    gc.insert(
                        output_name.clone(),
                        GpuTensor {
                            buf: out,
                            nhwc,
                            f16_io: false,
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
                dispatch(gpu, node, &mut env, &mut gc)?;
                dispatch(gpu, &nodes[*bn_idx], &mut env, &mut gc)?;
                fuse_relu(gpu, bn_output, relu_output, &mut env, &mut gc);
            }

            GpuExecAction::OpRelu {
                op_output,
                relu_output,
                ..
            } => {
                dispatch(gpu, node, &mut env, &mut gc)?;
                fuse_relu(gpu, op_output, relu_output, &mut env, &mut gc);
            }

            GpuExecAction::MatMulAdd { add_idx } => {
                dispatch(gpu, node, &mut env, &mut gc)?;
                dispatch(gpu, &nodes[*add_idx], &mut env, &mut gc)?;
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
                    let mut shape = gt.buf.shape().to_vec();
                    let total = gt.buf.len();
                    let nhwc = gt.nhwc;
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
                            for &ax in &sorted {
                                shape.insert(ax, 1);
                            }
                            shape
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
                                to_remove.sort_unstable_by(|a, b| b.cmp(a));
                                for ax in to_remove {
                                    if shape[ax] == 1 {
                                        shape.remove(ax);
                                    }
                                }
                            }
                            shape
                        }
                        _ => get_reshape_shape(node, &mut env, gpu, &mut gc, &shape, total)?,
                    };
                    let out = gpu.reshape_on_device(gt.buf, new_shape);
                    gc.insert(
                        node.outputs[0].clone(),
                        GpuTensor {
                            buf: out,
                            nhwc,
                            f16_io: false,
                        },
                    );
                } else {
                    dispatch(gpu, node, &mut env, &mut gc)?;
                }
            }
        }

        // NO recycling during compilation — keep all buffers alive.
        dispatch_count += 1;
        if dispatch_count.is_multiple_of(20) {
            gpu.flush();
        }
    }

    // Flush remaining work
    gpu.flush();

    // Take the recorded dispatch sequence
    let ops = gpu.take_recording();

    // Extract input buffer
    let input_buf = gc
        .remove(input_name)
        .ok_or_else(|| OnnxError::MissingInput {
            node: "compiled_plan".to_string(),
            input: input_name.to_string(),
        })?
        .buf;

    // Extract output buffers (remove from gc so we own them separately)
    let output_names: Vec<String> = model.outputs.clone();
    let mut output_bufs = Vec::new();
    for name in &output_names {
        // Output tensors may have been converted to CPU by to_cpu.
        // Re-check gc for GPU-resident outputs.
        if let Some(gt) = gc.get(name.as_str()) {
            // Clone shape info but keep buffer in gc (will be moved to pinned)
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

/// Run inference using a pre-compiled GPU plan.
/// Only writes new input data and replays pre-created dispatches.
/// Near-zero per-dispatch CPU overhead.
pub fn run_compiled_gpu(
    gpu: &GpuBackend,
    compiled: &CompiledGpuPlan,
    input_data: &[f32],
) -> Result<HashMap<String, Tensor>, OnnxError> {
    // Write new input data to the pre-allocated input buffer
    gpu.write_buffer(compiled.input_buf.raw_buffer(), input_data);

    // Replay all pre-created dispatches
    gpu.replay_recording(&compiled.ops, 20);

    // Flush remaining work
    gpu.flush();

    // Download outputs
    let mut result = HashMap::new();
    for (i, name) in compiled.output_names.iter().enumerate() {
        if let Some((out_buf, nhwc, _f16_io)) = compiled.output_bufs.get(i) {
            let t = gpu.download(out_buf)?;
            let shape = out_buf.shape();
            // If NHWC 4D, permute to NCHW for output
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

    Ok(result)
}

/// Run inference using a pre-compiled GPU plan with fused single-pass replay.
///
/// All dispatches are encoded into ONE compute pass, relying on Metal's
/// implicit memory ordering within a single `MTLComputeCommandEncoder`.
/// This eliminates per-dispatch pass creation overhead (~0.3-0.5ms × N).
///
/// **Only correct on Apple Silicon / Metal.** For Vulkan/DX12, use
/// `run_compiled_gpu` which creates separate passes with barriers.
pub fn run_compiled_gpu_fused(
    gpu: &GpuBackend,
    compiled: &CompiledGpuPlan,
    input_data: &[f32],
) -> Result<HashMap<String, Tensor>, OnnxError> {
    // Write new input data to the pre-allocated input buffer
    gpu.write_buffer(compiled.input_buf.raw_buffer(), input_data);

    // Replay all dispatches in a SINGLE compute pass (Metal-only optimization)
    gpu.replay_recording_fused(&compiled.ops);

    // Download outputs
    let mut result = HashMap::new();
    for (i, name) in compiled.output_names.iter().enumerate() {
        if let Some((out_buf, nhwc, _f16_io)) = compiled.output_bufs.get(i) {
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

    Ok(result)
}

/// Run compiled GPU plan (fused) with detailed timing breakdown.
/// Returns (result, upload_ms, encode_ms, gpu_ms, download_ms).
pub fn run_compiled_gpu_fused_timed(
    gpu: &GpuBackend,
    compiled: &CompiledGpuPlan,
    input_data: &[f32],
) -> Result<(HashMap<String, Tensor>, f64, f64, f64, f64), OnnxError> {
    let t0 = std::time::Instant::now();
    gpu.write_buffer(compiled.input_buf.raw_buffer(), input_data);
    let t1 = std::time::Instant::now();

    gpu.replay_recording_fused(&compiled.ops);
    let t2 = std::time::Instant::now();

    // Sync to measure pure GPU time
    gpu.sync();
    let t3 = std::time::Instant::now();

    // Download outputs
    let mut result = HashMap::new();
    for (i, name) in compiled.output_names.iter().enumerate() {
        if let Some((out_buf, nhwc, _f16_io)) = compiled.output_bufs.get(i) {
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
    let t4 = std::time::Instant::now();

    let upload_ms = (t1 - t0).as_secs_f64() * 1000.0;
    let encode_ms = (t2 - t1).as_secs_f64() * 1000.0;
    let gpu_ms = (t3 - t2).as_secs_f64() * 1000.0;
    let download_ms = (t4 - t3).as_secs_f64() * 1000.0;
    Ok((result, upload_ms, encode_ms, gpu_ms, download_ms))
}

// ── Helpers ──────────────────────────────────────────────────────

/// Apply Relu to the result of a previous op (GPU or CPU) and store
/// under `dst_name`.
fn fuse_relu(gpu: &GpuBackend, src: &str, dst: &str, env: &mut TensorEnv, gc: &mut GpuCache) {
    if let Some(gt) = gc.remove(src) {
        let out = gpu.relu_on_device(&gt.buf);
        gc.insert(
            dst.to_string(),
            GpuTensor {
                buf: out,
                nhwc: gt.nhwc,
                f16_io: false,
            },
        );
    } else if let Some(tensor) = env.get_mut(src) {
        for v in tensor.data_mut() {
            *v = v.max(0.0);
        }
        env.alias(dst, src);
    }
}

/// Download a GPU tensor to CPU env in NCHW format.  No-op if already on CPU.
/// If the tensor is NHWC 4D, permutes NHWC→NCHW on GPU before download
/// (faster than CPU permute for large tensors).
fn to_cpu(
    gpu: &GpuBackend,
    name: &str,
    env: &mut TensorEnv,
    gc: &mut GpuCache,
) -> Result<(), OnnxError> {
    if env.get(name).is_some() {
        return Ok(());
    }
    if let Some(gt) = gc.remove(name) {
        if gt.f16_io {
            // f16 buffer: convert to f32 first, then download
            if gt.nhwc && gt.buf.shape().len() == 4 {
                let nchw_buf = gpu.nhwc_to_nchw_f16_to_f32_on_device(&gt.buf)?;
                gpu.flush();
                let tensor = gpu.download(&nchw_buf)?;
                env.insert(name.to_string(), tensor);
            } else {
                gpu.flush();
                let data = gpu.read_buf_f16(gt.buf.raw_buffer(), gt.buf.len())?;
                let tensor = Tensor::from_vec(gt.buf.shape().to_vec(), data).map_err(|e| {
                    OnnxError::GpuKernel {
                        message: e.to_string(),
                    }
                })?;
                env.insert(name.to_string(), tensor);
            }
        } else if gt.nhwc && gt.buf.shape().len() == 4 {
            // Permute NHWC → NCHW on GPU, then download
            let nchw_buf = gpu.nhwc_to_nchw_on_device(&gt.buf);
            let tensor = gpu.download(&nchw_buf)?;
            env.insert(name.to_string(), tensor);
        } else {
            let tensor = gpu.download(&gt.buf)?;
            env.insert(name.to_string(), tensor);
        }
    }
    Ok(())
}

/// Ensure a tensor exists on GPU.  If only on CPU, upload it (NCHW, nhwc=false).
fn to_gpu(gpu: &GpuBackend, name: &str, env: &TensorEnv, gc: &mut GpuCache) {
    if gc.contains_key(name) {
        return;
    }
    if let Some(tensor) = env.get(name) {
        let buf = gpu.upload(tensor);
        gc.insert(
            name.to_string(),
            GpuTensor {
                buf,
                nhwc: false,
                f16_io: false,
            },
        );
    }
}

/// Ensure the tensor in `gc[name]` is in NHWC layout.
/// If it's NCHW 4D, permutes on GPU using [0,2,3,1] — no download.
fn ensure_nhwc(
    gpu: &GpuBackend,
    name: &str,
    gc: &mut GpuCache,
) -> Result<(), yscv_kernels::KernelError> {
    let needs = gc
        .get(name)
        .is_some_and(|gt| !gt.nhwc && gt.buf.shape().len() == 4);
    if needs {
        let gt = gc.get(name).unwrap();
        if gt.f16_io {
            // f16 buffer: convert to f32, permute, convert back to f16
            let f32_buf = gpu.convert_f16_to_f32_on_device(&gt.buf)?;
            let nhwc_f32 = gpu.permute_on_device(&f32_buf, &[0, 2, 3, 1]);
            let nhwc_f16 = gpu.convert_f32_to_f16_on_device(&nhwc_f32)?;
            gc.insert(
                name.to_string(),
                GpuTensor {
                    buf: nhwc_f16,
                    nhwc: true,
                    f16_io: true,
                },
            );
        } else {
            let out = gpu.permute_on_device(&gt.buf, &[0, 2, 3, 1]);
            gc.insert(
                name.to_string(),
                GpuTensor {
                    buf: out,
                    nhwc: true,
                    f16_io: false,
                },
            );
        }
    }
    Ok(())
}

/// Materialize all inputs of a node to CPU for fallback execution.
fn inputs_to_cpu(
    gpu: &GpuBackend,
    node: &OnnxNode,
    env: &mut TensorEnv,
    gc: &mut GpuCache,
) -> Result<(), OnnxError> {
    for name in &node.inputs {
        if !name.is_empty() {
            to_cpu(gpu, name, env, gc)?;
        }
    }
    Ok(())
}

// ── GPU op dispatch ──────────────────────────────────────────────

fn dispatch(
    gpu: &GpuBackend,
    node: &OnnxNode,
    env: &mut TensorEnv,
    gc: &mut GpuCache,
) -> Result<(), OnnxError> {
    dispatch_inner(gpu, node, env, gc)
}

/// Profile one complete inference with GPU sync after each fused action.
/// Uses cached weights and execution plan for realistic timings.
/// Returns map of action_label → (total_ms, count).
pub fn profile_onnx_model_gpu(
    model: &OnnxModel,
    inputs: HashMap<String, Tensor>,
) -> Result<HashMap<String, (f64, usize)>, OnnxError> {
    let gpu = GpuBackend::new().map_err(|e| OnnxError::DecodeFailed {
        message: format!("GPU init: {e}"),
    })?;
    let plan = plan_gpu_execution(model);
    let mut wc = GpuWeightCache::new();

    // Warm-up run to populate weight cache
    {
        let mut inputs_warm = HashMap::new();
        for (k, v) in &inputs {
            inputs_warm.insert(k.clone(), v.clone());
        }
        let _ = run_onnx_model_gpu_cached(&gpu, model, inputs_warm, &mut wc, Some(&plan))?;
    }

    // Now profile with cached weights
    let mut env = TensorEnv::from_model(model);
    let mut gc: GpuCache = std::mem::take(&mut wc.0);
    let mut stats: HashMap<String, (f64, usize)> = HashMap::new();

    for (name, tensor) in &model.initializers {
        env.insert(name.clone(), tensor.clone());
    }
    for (name, tensor) in inputs {
        env.insert(name, tensor);
    }

    let nodes = &model.nodes;
    for (i, node) in nodes.iter().enumerate() {
        let action = &plan.actions[i];
        if matches!(action, GpuExecAction::Skip) {
            continue;
        }

        let label = match action {
            GpuExecAction::ConvSiLU { .. } => "ConvSiLU",
            GpuExecAction::SiLU { .. } => "SiLU",
            GpuExecAction::ConvBnRelu { .. } => "ConvBnRelu",
            GpuExecAction::OpRelu { .. } => format!("{}+Relu", node.op_type).leak() as &str,
            GpuExecAction::MatMulAdd { .. } => "MatMul+Add",
            GpuExecAction::Normal => node.op_type.as_str(),
            GpuExecAction::Skip => unreachable!(),
        };

        gpu.flush();
        gpu.sync();
        let t0 = std::time::Instant::now();

        match action {
            GpuExecAction::ConvSiLU { output_name, .. } => {
                exec_conv_act(&gpu, node, &mut env, &mut gc, 2)?;
                if let Some(gt) = gc.remove(&node.outputs[0]) {
                    gc.insert(output_name.clone(), gt);
                }
            }
            GpuExecAction::SiLU {
                x_name,
                output_name,
                ..
            } => {
                to_gpu(&gpu, x_name, &env, &mut gc);
                if let Some(gt) = gc.get(x_name.as_str()) {
                    let out = gpu.silu_on_device(&gt.buf);
                    let nhwc = gt.nhwc;
                    gc.insert(
                        output_name.clone(),
                        GpuTensor {
                            buf: out,
                            nhwc,
                            f16_io: false,
                        },
                    );
                }
            }
            GpuExecAction::ConvBnRelu {
                bn_idx,
                bn_output,
                relu_output,
                ..
            } => {
                dispatch(&gpu, node, &mut env, &mut gc)?;
                dispatch(&gpu, &nodes[*bn_idx], &mut env, &mut gc)?;
                fuse_relu(&gpu, bn_output, relu_output, &mut env, &mut gc);
            }
            GpuExecAction::OpRelu {
                op_output,
                relu_output,
                ..
            } => {
                dispatch(&gpu, node, &mut env, &mut gc)?;
                fuse_relu(&gpu, op_output, relu_output, &mut env, &mut gc);
            }
            GpuExecAction::MatMulAdd { add_idx } => {
                dispatch(&gpu, node, &mut env, &mut gc)?;
                dispatch(&gpu, &nodes[*add_idx], &mut env, &mut gc)?;
            }
            GpuExecAction::Normal => {
                dispatch(&gpu, node, &mut env, &mut gc)?;
            }
            GpuExecAction::Skip => unreachable!(),
        }

        gpu.flush();
        gpu.sync();
        let elapsed = t0.elapsed().as_secs_f64() * 1000.0;

        let e = stats.entry(label.to_string()).or_insert((0.0, 0));
        e.0 += elapsed;
        e.1 += 1;

        // Recycle
        for name in &plan.recycle_at[i] {
            if let Some(gt) = gc.remove(name) {
                recycle(&gpu, gt);
            }
        }
    }

    #[cfg(feature = "profile")]
    {
        let mut sorted: Vec<_> = stats.iter().collect();
        sorted.sort_by(|a, b| b.1.0.partial_cmp(&a.1.0).unwrap());
        eprintln!("\n  ── GPU Profile (fused, cached, sync per action) ──");
        for (op, (ms, count)) in &sorted {
            eprintln!("  {:>8.2}ms  {:>4}x  {}", ms, count, op);
        }
        let total: f64 = stats.values().map(|v| v.0).sum();
        eprintln!("  {:>8.2}ms  total", total);
    }

    Ok(stats)
}

fn dispatch_inner(
    gpu: &GpuBackend,
    node: &OnnxNode,
    env: &mut TensorEnv,
    gc: &mut GpuCache,
) -> Result<(), OnnxError> {
    match node.op_type.as_str() {
        // ── Spatial ops (need NHWC) ──────────────────────────────
        "Conv" => exec_conv(gpu, node, env, gc),
        "BatchNormalization" => exec_bn(gpu, node, env, gc),
        "MaxPool" => exec_pool(gpu, node, env, gc, 0),
        "AveragePool" => exec_pool(gpu, node, env, gc, 1),
        "GlobalAveragePool" => exec_gap(gpu, node, env, gc),

        // ── Element-wise unary (layout-agnostic) ─────────────────
        "Relu" => unary(gpu, node, gc, env, |g, b| g.relu_on_device(b)),
        "Sigmoid" => unary(gpu, node, gc, env, |g, b| g.sigmoid_on_device(b)),
        "Exp" => unary(gpu, node, gc, env, |g, b| g.exp_on_device(b)),
        "Tanh" => unary(gpu, node, gc, env, |g, b| g.tanh_on_device(b)),

        // ── Element-wise binary (layout-agnostic) ────────────────
        "Add" => binary(gpu, node, gc, env, |g, a, b| g.add_on_device(a, b)),
        "Sub" => binary(gpu, node, gc, env, |g, a, b| g.sub_on_device(a, b)),
        "Mul" => binary(gpu, node, gc, env, |g, a, b| g.mul_on_device(a, b)),
        "Div" => binary(gpu, node, gc, env, |g, a, b| g.div_on_device(a, b)),

        // ── Linear algebra ───────────────────────────────────────
        "MatMul" => exec_matmul(gpu, node, env, gc),
        "Gemm" => exec_gemm(gpu, node, env, gc),
        "Softmax" => exec_softmax(gpu, node, env, gc),

        // ── Fused patterns ───────────────────────────────────────
        "Conv_Relu" => {
            exec_conv(gpu, node, env, gc)?;
            fuse_relu(gpu, &node.outputs[0], &node.outputs[0], env, gc);
            Ok(())
        }
        "BatchNormalization_Relu" => {
            exec_bn(gpu, node, env, gc)?;
            fuse_relu(gpu, &node.outputs[0], &node.outputs[0], env, gc);
            Ok(())
        }

        // ── Reshape / data movement (GPU-accelerated) ────────────
        "Transpose" => exec_transpose_gpu(gpu, node, env, gc),
        "Concat" => exec_concat_gpu(gpu, node, env, gc),
        "Split" => exec_split_gpu(gpu, node, env, gc),
        "Resize" | "Upsample" => exec_resize_gpu(gpu, node, env, gc),
        "Shape" => exec_shape_gpu(gpu, node, env, gc),
        "Reshape" | "Flatten" => exec_reshape_gpu(gpu, node, env, gc),
        "Unsqueeze" | "Squeeze" => exec_unsqueeze_gpu(gpu, node, env, gc),
        "Slice" => exec_slice_gpu(gpu, node, env, gc),
        "ConstantOfShape" => exec_constant_of_shape_gpu(gpu, node, env, gc),
        "Expand" => exec_expand_gpu(gpu, node, env, gc),

        // ── CPU fallback ─────────────────────────────────────────
        _ => {
            inputs_to_cpu(gpu, node, env, gc)?;
            super::execute_node_cpu_fallback(node, env)
        }
    }
}

// ── Unary / Binary helpers ───────────────────────────────────────

fn unary(
    gpu: &GpuBackend,
    node: &OnnxNode,
    gc: &mut GpuCache,
    env: &mut TensorEnv,
    op: impl Fn(&GpuBackend, &GpuBuffer) -> GpuBuffer,
) -> Result<(), OnnxError> {
    let name = &node.inputs[0];

    // Small CPU-only tensors: keep on CPU to avoid GPU sync on shape metadata
    if !gc.contains_key(name)
        && let Some(t) = env.get(name)
        && t.data().len() < 64
    {
        return super::execute_node_cpu_fallback(node, env);
    }

    to_gpu(gpu, name, env, gc);

    let (out, nhwc) = {
        let gt = gc.get(name).ok_or_else(|| OnnxError::MissingInput {
            node: node.name.clone(),
            input: name.clone(),
        })?;
        (op(gpu, &gt.buf), gt.nhwc)
    };

    gc.insert(
        node.outputs[0].clone(),
        GpuTensor {
            buf: out,
            nhwc,
            f16_io: false,
        },
    );
    Ok(())
}

fn binary(
    gpu: &GpuBackend,
    node: &OnnxNode,
    gc: &mut GpuCache,
    env: &mut TensorEnv,
    op: impl Fn(&GpuBackend, &GpuBuffer, &GpuBuffer) -> GpuBuffer,
) -> Result<(), OnnxError> {
    let a_name = &node.inputs[0];
    let b_name = &node.inputs[1];

    // Small CPU-only tensors: keep on CPU to avoid GPU sync on shape metadata
    let a_cpu_small =
        !gc.contains_key(a_name) && env.get(a_name).is_some_and(|t| t.data().len() < 64);
    let b_cpu_small =
        !gc.contains_key(b_name) && env.get(b_name).is_some_and(|t| t.data().len() < 64);
    // Tensors < 4 elements can't be used in vec4 GPU shaders at all
    let a_too_small =
        !gc.contains_key(a_name) && env.get(a_name).is_some_and(|t| t.data().len() < 4);
    let b_too_small =
        !gc.contains_key(b_name) && env.get(b_name).is_some_and(|t| t.data().len() < 4);
    if (a_cpu_small && b_cpu_small) || a_too_small || b_too_small {
        inputs_to_cpu(gpu, node, env, gc)?;
        return super::execute_node_cpu_fallback(node, env);
    }

    to_gpu(gpu, a_name, env, gc);
    to_gpu(gpu, b_name, env, gc);

    let result = {
        let a = gc.get(a_name);
        let b = gc.get(b_name);
        match (a, b) {
            (Some(a), Some(b)) if a.buf.len() == b.buf.len() => {
                Some((op(gpu, &a.buf, &b.buf), a.nhwc))
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
                f16_io: false,
            },
        );
        Ok(())
    } else {
        // Try broadcasting on GPU
        let bcast_result = {
            let a = gc.get(a_name);
            let b = gc.get(b_name);
            if let (Some(a), Some(b)) = (a, b) {
                let op_code = match node.op_type.as_str() {
                    "Add" => 0u32,
                    "Sub" => 1,
                    "Mul" => 2,
                    "Div" => 3,
                    _ => 255,
                };
                if op_code < 255 {
                    let out = gpu.broadcast_binary_on_device(&a.buf, &b.buf, op_code);
                    Some((out, a.nhwc))
                } else {
                    None
                }
            } else {
                None
            }
        };

        if let Some((out, nhwc)) = bcast_result {
            gc.insert(
                node.outputs[0].clone(),
                GpuTensor {
                    buf: out,
                    nhwc,
                    f16_io: false,
                },
            );
            Ok(())
        } else {
            inputs_to_cpu(gpu, node, env, gc)?;
            super::execute_node_cpu_fallback(node, env)
        }
    }
}

// ── Winograd F(2,3) weight transform ────────────────────────────
// G * g * G^T for each (oc, ic) pair.  g is 3x3, result is 4x4.
// G = [[1, 0, 0], [0.5, 0.5, 0.5], [0.5, -0.5, 0.5], [0, 0, 1]]
// Output layout: [16, IC, OC] (alpha-major, then IC, then OC).
fn winograd_transform_weights(weight: &Tensor) -> Tensor {
    let (oc, ic, _kh, _kw) = (
        weight.shape()[0],
        weight.shape()[1],
        weight.shape()[2],
        weight.shape()[3],
    );
    let w = weight.data();
    let mut out = vec![0.0f32; 16 * ic * oc];

    for o in 0..oc {
        for c in 0..ic {
            // Load 3x3 kernel g[r][s]
            let base = ((o * ic + c) * 3) * 3;
            let g = |r: usize, s: usize| w[base + r * 3 + s];

            // Compute G * g (4x3)
            // G row0: [1, 0, 0]
            // G row1: [0.5, 0.5, 0.5]
            // G row2: [0.5, -0.5, 0.5]
            // G row3: [0, 0, 1]
            let mut gg = [0.0f32; 12]; // 4x3
            for s in 0..3 {
                gg[s] = g(0, s);
                gg[3 + s] = 0.5 * (g(0, s) + g(1, s) + g(2, s));
                gg[2 * 3 + s] = 0.5 * (g(0, s) - g(1, s) + g(2, s));
                gg[3 * 3 + s] = g(2, s);
            }

            // Compute (G * g) * G^T (4x4)
            // G^T col0: [1, 0.5, 0.5, 0]
            // G^T col1: [0, 0.5, -0.5, 0]
            // G^T col2: [0, 0.5, 0.5, 1]
            let mut u = [0.0f32; 16]; // 4x4
            for r in 0..4 {
                let row = &gg[r * 3..r * 3 + 3];
                u[r * 4] = row[0];
                u[r * 4 + 1] = 0.5 * (row[0] + row[1] + row[2]);
                u[r * 4 + 2] = 0.5 * (row[0] - row[1] + row[2]);
                u[r * 4 + 3] = row[2];
            }

            // Scatter to [alpha, IC, OC] layout
            for a in 0..16 {
                out[a * ic * oc + c * oc + o] = u[a];
            }
        }
    }

    Tensor::from_vec(vec![16, ic, oc], out).unwrap()
}

// ── Conv2D (standard + depthwise) ────────────────────────────────

fn exec_conv(
    gpu: &GpuBackend,
    node: &OnnxNode,
    env: &mut TensorEnv,
    gc: &mut GpuCache,
) -> Result<(), OnnxError> {
    exec_conv_act(gpu, node, env, gc, 0)
}

fn exec_conv_act(
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
    let (o_ch, i_per_g, kh, kw) = if is_khwc {
        // Pre-permuted KHWC: [kH, kW, C_in/group, C_out]
        (w_shape[3], w_shape[2], w_shape[0], w_shape[1])
    } else {
        // Standard OIHW: [C_out, C_in/group, kH, kW]
        (w_shape[0], w_shape[1], w_shape[2], w_shape[3])
    };

    // Ensure input is on GPU in NHWC layout — no download
    let input_name = &node.inputs[0];
    to_gpu(gpu, input_name, env, gc);
    ensure_nhwc(gpu, input_name, gc)?;

    let ic = gc.get(input_name).map(|gt| gt.buf.shape()[3]).unwrap_or(0);

    // General grouped conv (not depthwise) → CPU fallback
    if group > 1 && !(group == o_ch && group == ic) {
        inputs_to_cpu(gpu, node, env, gc)?;
        return conv::exec_conv(node, env, yscv_kernels::Activation::None);
    }

    if group == 1 {
        // Winograd F(2,3) needs 3 dispatches + intermediate buffers;
        // on bandwidth-bound M1 GPU the extra memory traffic outweighs
        // the 2.25x arithmetic savings. Disable for now (threshold=never).
        // Winograd F(2,3): disabled for now — extra memory traffic outweighs savings.
        let use_winograd = kh == 3 && kw == 3 && sh == 1 && sw == 1;

        if use_winograd {
            // Winograd F(2,3) path — cache transformed weights [16, IC, OC]
            let wino_key = format!("__w_wino_{}", node.inputs[1]);
            let b_key = format!("__bias_{}", node.inputs[1]);
            if !gc.contains_key(&wino_key) {
                let wt = winograd_transform_weights(weight);
                gc.insert(
                    wino_key.clone(),
                    GpuTensor {
                        buf: gpu.upload(&wt),
                        nhwc: false,
                        f16_io: false,
                    },
                );
                let bias_data = bias
                    .map(|b| b.data().to_vec())
                    .unwrap_or_else(|| vec![0.0f32; o_ch]);
                let bias_t = Tensor::from_vec(vec![o_ch], bias_data).map_err(|e| {
                    OnnxError::DecodeFailed {
                        message: e.to_string(),
                    }
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
                let ww = &gc.get(&wino_key).unwrap().buf;
                let b = &gc.get(&b_key).unwrap().buf;
                let input_buf = &gc.get(input_name).unwrap().buf;
                gpu.winograd_conv_on_device(input_buf, ww, b, pad4, act)
            };
            gc.insert(
                node.outputs[0].clone(),
                GpuTensor {
                    buf: out,
                    nhwc: true,
                    f16_io: false,
                },
            );
        } else {
            // Standard conv_gemm path — cache transformed weights on GPU
            let w_key = format!("__w_nhwc_{}", node.inputs[1]);
            let b_key = format!("__bias_{}", node.inputs[1]);
            if !gc.contains_key(&w_key) {
                let w_nhwc = if is_khwc {
                    weight.clone()
                } else {
                    oihw_to_khwc_cout(weight)?
                };
                let w_gpu = gpu.upload(&w_nhwc);
                gc.insert(
                    w_key.clone(),
                    GpuTensor {
                        buf: w_gpu,
                        nhwc: false,
                        f16_io: false,
                    },
                );
                let bias_data = bias
                    .map(|b| b.data().to_vec())
                    .unwrap_or_else(|| vec![0.0f32; o_ch]);
                let bias_t = Tensor::from_vec(vec![o_ch], bias_data).map_err(|e| {
                    OnnxError::DecodeFailed {
                        message: e.to_string(),
                    }
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
                let input_buf = &gc.get(input_name).unwrap().buf;
                gpu.im2col_conv_on_device(input_buf, w, b, sh, sw, pad4, act)
            };
            gc.insert(
                node.outputs[0].clone(),
                GpuTensor {
                    buf: out,
                    nhwc: true,
                    f16_io: false,
                },
            );
        }
    } else {
        // Depthwise convolution — cache transformed weights on GPU
        let dw_key = format!("__dw_{}", node.inputs[1]);
        let b_key = format!("__bias_dw_{}", node.inputs[1]);
        if !gc.contains_key(&dw_key) {
            let c = group;
            let depth_mult = o_ch / c;
            let mut dw_data = vec![0.0f32; kh * kw * c * depth_mult];
            let w_data = weight.data();
            for oc in 0..o_ch {
                let g = oc / depth_mult;
                let dm = oc % depth_mult;
                for ki in 0..kh {
                    for kj in 0..kw {
                        let src = ((oc * i_per_g) * kh + ki) * kw + kj;
                        let dst = ((ki * kw + kj) * c + g) * depth_mult + dm;
                        dw_data[dst] = w_data[src];
                    }
                }
            }
            let dw_kernel =
                Tensor::from_vec(vec![kh, kw, c, depth_mult], dw_data).map_err(|e| {
                    OnnxError::DecodeFailed {
                        message: e.to_string(),
                    }
                })?;
            gc.insert(
                dw_key.clone(),
                GpuTensor {
                    buf: gpu.upload(&dw_kernel),
                    nhwc: false,
                    f16_io: false,
                },
            );
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

        let c = group;
        let _depth_mult = o_ch / c;
        let out = {
            let dw = &gc.get(&dw_key).unwrap().buf;
            let b = &gc.get(&b_key).unwrap().buf;
            let input_buf = &gc.get(input_name).unwrap().buf;
            gpu.depthwise_conv2d_nhwc_on_device(input_buf, dw, b, sh, sw, pad4, act)
        };
        gc.insert(
            node.outputs[0].clone(),
            GpuTensor {
                buf: out,
                nhwc: true,
                f16_io: false,
            },
        );
    }

    Ok(())
}

// ── BatchNormalization ───────────────────────────────────────────

fn exec_bn(
    gpu: &GpuBackend,
    node: &OnnxNode,
    env: &mut TensorEnv,
    gc: &mut GpuCache,
) -> Result<(), OnnxError> {
    let name = &node.inputs[0];
    to_gpu(gpu, name, env, gc);
    ensure_nhwc(gpu, name, gc)?;

    let gamma = get_tensor(env, &node.name, &node.inputs[1])?;
    let beta = get_tensor(env, &node.name, &node.inputs[2])?;
    let mean = get_tensor(env, &node.name, &node.inputs[3])?;
    let var = get_tensor(env, &node.name, &node.inputs[4])?;
    let eps = get_attr_float(node, "epsilon").unwrap_or(1e-5);

    let gamma_gpu = gpu.upload(gamma);
    let beta_gpu = gpu.upload(beta);
    let mean_gpu = gpu.upload(mean);
    let var_gpu = gpu.upload(var);

    let out = {
        let input_buf = &gc
            .get(name)
            .ok_or_else(|| OnnxError::MissingInput {
                node: node.name.clone(),
                input: name.clone(),
            })?
            .buf;
        gpu.batch_norm2d_nhwc_on_device(input_buf, &gamma_gpu, &beta_gpu, &mean_gpu, &var_gpu, eps)
    };

    gc.insert(
        node.outputs[0].clone(),
        GpuTensor {
            buf: out,
            nhwc: true,
            f16_io: false,
        },
    );
    Ok(())
}

// ── Pooling ──────────────────────────────────────────────────────

fn exec_pool(
    gpu: &GpuBackend,
    node: &OnnxNode,
    env: &mut TensorEnv,
    gc: &mut GpuCache,
    mode: u32,
) -> Result<(), OnnxError> {
    let name = &node.inputs[0];
    let kernel_shape = get_attr_ints(node, "kernel_shape").unwrap_or_else(|| vec![2, 2]);
    let strides = get_attr_ints(node, "strides").unwrap_or_else(|| vec![1, 1]);
    let pads = get_attr_ints(node, "pads").unwrap_or_else(|| vec![0, 0, 0, 0]);
    let pad4 = [
        pads[0] as usize,
        pads[1] as usize,
        pads[2] as usize,
        pads[3] as usize,
    ];

    // Stay on GPU — padding handled in shader
    to_gpu(gpu, name, env, gc);
    ensure_nhwc(gpu, name, gc)?;

    let out = {
        let input_buf = &gc
            .get(name)
            .ok_or_else(|| OnnxError::MissingInput {
                node: node.name.clone(),
                input: name.clone(),
            })?
            .buf;
        gpu.pool2d_nhwc_on_device(
            input_buf,
            kernel_shape[0] as usize,
            kernel_shape[1] as usize,
            strides[0] as usize,
            strides[1] as usize,
            mode,
            pad4,
        )
    };
    gc.insert(
        node.outputs[0].clone(),
        GpuTensor {
            buf: out,
            nhwc: true,
            f16_io: false,
        },
    );

    Ok(())
}

fn exec_gap(
    gpu: &GpuBackend,
    node: &OnnxNode,
    env: &mut TensorEnv,
    gc: &mut GpuCache,
) -> Result<(), OnnxError> {
    let name = &node.inputs[0];
    to_gpu(gpu, name, env, gc);
    ensure_nhwc(gpu, name, gc)?;

    let (h, w) = {
        let gt = gc.get(name).ok_or_else(|| OnnxError::MissingInput {
            node: node.name.clone(),
            input: name.clone(),
        })?;
        (gt.buf.shape()[1], gt.buf.shape()[2])
    };

    let out = {
        let buf = &gc.get(name).unwrap().buf;
        gpu.pool2d_nhwc_on_device(buf, h, w, 1, 1, 1, [0, 0, 0, 0]) // avg mode
    };
    gc.insert(
        node.outputs[0].clone(),
        GpuTensor {
            buf: out,
            nhwc: true,
            f16_io: false,
        },
    );
    Ok(())
}

// ── Softmax ──────────────────────────────────────────────────────

fn exec_softmax(
    gpu: &GpuBackend,
    node: &OnnxNode,
    env: &TensorEnv,
    gc: &mut GpuCache,
) -> Result<(), OnnxError> {
    let name = &node.inputs[0];

    // Resolve axis attribute
    let ndim = gc
        .get(name)
        .map(|gt| gt.buf.shape().len())
        .or_else(|| env.get(name).map(|t| t.rank()))
        .unwrap_or(0);
    let raw_axis = get_attr_int(node, "axis").unwrap_or(-1);
    let axis = if raw_axis < 0 {
        (ndim as i64 + raw_axis) as usize
    } else {
        raw_axis as usize
    };

    if axis == ndim - 1 {
        // Fast path: softmax on last dim — run on GPU
        to_gpu(gpu, name, env, gc);
        let (out, nhwc) = {
            let gt = gc.get(name).ok_or_else(|| OnnxError::MissingInput {
                node: node.name.clone(),
                input: name.clone(),
            })?;
            (gpu.softmax_on_device(&gt.buf), gt.nhwc)
        };
        gc.insert(
            node.outputs[0].clone(),
            GpuTensor {
                buf: out,
                nhwc,
                f16_io: false,
            },
        );
    } else {
        // Non-last axis: fall back to CPU (transpose + softmax + transpose)
        to_gpu(gpu, name, env, gc);
        let input = {
            let gt = gc.get(name).ok_or_else(|| OnnxError::MissingInput {
                node: node.name.clone(),
                input: name.clone(),
            })?;
            gpu.download(&gt.buf)?
        };
        let mut perm: Vec<usize> = (0..ndim).collect();
        perm.swap(axis, ndim - 1);
        let transposed = input.permute(&perm).map_err(|e| OnnxError::DecodeFailed {
            message: e.to_string(),
        })?;
        let sm = softmax_last_dim(&transposed).map_err(|e| OnnxError::DecodeFailed {
            message: e.to_string(),
        })?;
        let result = sm.permute(&perm).map_err(|e| OnnxError::DecodeFailed {
            message: e.to_string(),
        })?;
        // Upload result back to GPU
        let gpu_buf = gpu.upload(&result);
        gc.insert(
            node.outputs[0].clone(),
            GpuTensor {
                buf: gpu_buf,
                nhwc: false,
                f16_io: false,
            },
        );
    }
    Ok(())
}

// ── MatMul ───────────────────────────────────────────────────────

fn exec_matmul(
    gpu: &GpuBackend,
    node: &OnnxNode,
    env: &mut TensorEnv,
    gc: &mut GpuCache,
) -> Result<(), OnnxError> {
    let a_name = &node.inputs[0];
    let b_name = &node.inputs[1];

    // Check rank — GPU matmul is 2D only; batched goes to CPU
    let a_rank = gc
        .get(a_name)
        .map(|gt| gt.buf.shape().len())
        .or_else(|| env.get(a_name).map(|t| t.rank()))
        .unwrap_or(0);
    let b_rank = gc
        .get(b_name)
        .map(|gt| gt.buf.shape().len())
        .or_else(|| env.get(b_name).map(|t| t.rank()))
        .unwrap_or(0);

    // Batched matmul for rank > 2 (attention etc.)
    if a_rank > 2 || b_rank > 2 {
        to_gpu(gpu, a_name, env, gc);
        to_gpu(gpu, b_name, env, gc);
        let a_buf = gc.get(a_name);
        let b_buf = gc.get(b_name);
        if let (Some(a), Some(b)) = (a_buf, b_buf) {
            let out = gpu.batched_matmul_on_device(&a.buf, &b.buf);
            gc.insert(
                node.outputs[0].clone(),
                GpuTensor {
                    buf: out,
                    nhwc: false,
                    f16_io: false,
                },
            );
            return Ok(());
        }
        // Fallback to CPU if inputs missing
        inputs_to_cpu(gpu, node, env, gc)?;
        return super::execute_node_cpu_fallback(node, env);
    }

    to_gpu(gpu, a_name, env, gc);
    to_gpu(gpu, b_name, env, gc);

    let out = {
        let a_buf = &gc
            .get(a_name)
            .ok_or_else(|| OnnxError::MissingInput {
                node: node.name.clone(),
                input: a_name.clone(),
            })?
            .buf;
        let b_buf = &gc
            .get(b_name)
            .ok_or_else(|| OnnxError::MissingInput {
                node: node.name.clone(),
                input: b_name.clone(),
            })?
            .buf;
        gpu.matmul_2d_on_device(a_buf, b_buf)
    };

    gc.insert(
        node.outputs[0].clone(),
        GpuTensor {
            buf: out,
            nhwc: false,
            f16_io: false,
        },
    );
    Ok(())
}

// ── Gemm ─────────────────────────────────────────────────────────

fn exec_gemm(
    gpu: &GpuBackend,
    node: &OnnxNode,
    env: &mut TensorEnv,
    gc: &mut GpuCache,
) -> Result<(), OnnxError> {
    // Gemm has alpha/beta/transA/transB — handled on CPU for simplicity
    inputs_to_cpu(gpu, node, env, gc)?;
    super::linear::exec_gemm(node, env)
}

// ── GPU-native Transpose ─────────────────────────────────────────

fn exec_transpose_gpu(
    gpu: &GpuBackend,
    node: &OnnxNode,
    env: &mut TensorEnv,
    gc: &mut GpuCache,
) -> Result<(), OnnxError> {
    let name = &node.inputs[0];
    let on_gpu = gc.contains_key(name.as_str());

    if !on_gpu {
        return super::reshape::exec_transpose(node, env);
    }

    let gt = gc.get(name.as_str()).unwrap();
    let rank = gt.buf.shape().len();

    // Get permutation (default: reverse axes)
    let perm: Vec<usize> = match get_attr_ints(node, "perm") {
        Some(p) => p.iter().map(|&v| v as usize).collect(),
        None => (0..rank).rev().collect(),
    };

    // For NHWC 4D tensors, remap the perm from NCHW→NHWC coordinate space
    let actual_perm = if gt.nhwc && rank == 4 {
        // ONNX perm is in NCHW space. Map: N=0→0, C=1→3, H=2→1, W=3→2
        let nchw_to_nhwc = [0usize, 3, 1, 2]; // NCHW dim i → NHWC dim
        let nhwc_to_nchw = [0usize, 2, 3, 1]; // NHWC dim i → NCHW dim
        // Compose: nhwc_out[nchw_to_nhwc[i]] = nhwc_in[nchw_to_nhwc[perm[i]]]
        // Actually: perm maps output NCHW dim → input NCHW dim
        // We need actual_perm mapping output NHWC dim → input NHWC dim
        // For each NHWC output dim j:
        //   NCHW output dim = nhwc_to_nchw[j]
        //   NCHW input dim = perm[nhwc_to_nchw[j]]
        //   NHWC input dim = nchw_to_nhwc[perm[nhwc_to_nchw[j]]]
        (0..4)
            .map(|j| nchw_to_nhwc[perm[nhwc_to_nchw[j]]])
            .collect::<Vec<_>>()
    } else {
        perm
    };

    // Check if permutation is identity
    let is_identity = actual_perm.iter().enumerate().all(|(i, &p)| p == i);
    if is_identity {
        // No-op: just alias the tensor
        let out = gpu.copy_reshape_on_device(&gt.buf, gt.buf.shape().to_vec());
        gc.insert(
            node.outputs[0].clone(),
            GpuTensor {
                buf: out,
                nhwc: gt.nhwc,
                f16_io: false,
            },
        );
        return Ok(());
    }

    let out = gpu.permute_on_device(&gt.buf, &actual_perm);
    // After a general permute, the tensor is no longer in NHWC layout
    let nhwc = false;
    gc.insert(
        node.outputs[0].clone(),
        GpuTensor {
            buf: out,
            nhwc,
            f16_io: false,
        },
    );
    Ok(())
}

// ── GPU-native Concat ────────────────────────────────────────────

fn exec_concat_gpu(
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

    // Small tensors all on CPU: keep on CPU to avoid GPU sync on shape metadata
    let all_small_cpu = input_names
        .iter()
        .all(|n| env.get(n.as_str()).is_some_and(|t| t.data().len() < 64));
    if all_small_cpu {
        return super::reshape::exec_concat(node, env);
    }

    // Check if any input is on GPU — if so, upload the rest to GPU too
    let any_on_gpu = input_names.iter().any(|n| gc.contains_key(n.as_str()));
    if !any_on_gpu {
        return super::reshape::exec_concat(node, env);
    }
    // Upload any CPU-only inputs to GPU
    for &name in &input_names {
        to_gpu(gpu, name, env, gc);
    }
    // Verify all on GPU (some might not exist at all)
    if !input_names.iter().all(|n| gc.contains_key(n.as_str())) {
        inputs_to_cpu(gpu, node, env, gc)?;
        return super::reshape::exec_concat(node, env);
    }

    // Layout normalisation. The first input's NHWC flag was used as the
    // sole signal of "are we doing channel-concat here", which silently
    // misbehaves when sibling inputs are still in NCHW: axis=1 (NCHW C)
    // gets remapped to axis=3 (NHWC C) for the whole call, but the
    // NCHW siblings have W at axis 3, so `channel_concat_on_device`
    // glues 256-NHWC channels with 16 NCHW W-positions and the output
    // tensor reports 272 channels instead of the expected 256+64=320.
    // Force every 4-D sibling into the NHWC layout the first input is
    // in (or vice versa) so axis 3 means C for all of them.
    let any_nhwc = input_names.iter().any(|n| {
        gc.get(n.as_str())
            .is_some_and(|g| g.nhwc && g.buf.shape().len() == 4)
    });
    if any_nhwc {
        for &name in &input_names {
            if gc
                .get(name.as_str())
                .is_some_and(|g| g.buf.shape().len() == 4 && !g.nhwc)
            {
                ensure_nhwc(gpu, name, gc).map_err(|e| OnnxError::DecodeFailed {
                    message: e.to_string(),
                })?;
            }
        }
    }

    let first = gc.get(input_names[0].as_str()).unwrap();
    let rank = first.buf.shape().len();
    let is_nhwc = first.nhwc && rank == 4;

    // Map ONNX axis (NCHW) to actual axis in our layout
    let actual_axis = if is_nhwc {
        // NCHW→NHWC: 0→0, 1→3, 2→1, 3→2
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

    // Last-axis: use the optimized channel_concat shader
    // Any other axis: use the general concat shader
    let bufs: Vec<&GpuBuffer> = input_names
        .iter()
        .map(|n| &gc.get(n.as_str()).unwrap().buf)
        .collect();
    let out = if actual_axis == rank - 1 {
        gpu.channel_concat_on_device(&bufs)?
    } else {
        gpu.general_concat_on_device(&bufs, actual_axis)?
    };
    gc.insert(
        node.outputs[0].clone(),
        GpuTensor {
            buf: out,
            nhwc: is_nhwc,
            f16_io: false,
        },
    );
    Ok(())
}

// ── GPU-native Split ─────────────────────────────────────────────

fn exec_split_gpu(
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

    let gt = gc.get(name.as_str()).unwrap();
    let rank = gt.buf.shape().len();
    let is_nhwc = gt.nhwc && rank == 4;

    // Map ONNX axis to actual axis
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

    let dim_size = gt.buf.shape()[actual_axis];
    let num_outputs = node.outputs.len();

    // Get split sizes
    let split_sizes: Vec<usize> = if node.inputs.len() > 1 && !node.inputs[1].is_empty() {
        if let Some(split_t) = env.get(&node.inputs[1]) {
            split_t.data().iter().map(|&v| v as usize).collect()
        } else if let Some(gt_split) = gc.get(node.inputs[1].as_str()) {
            let data = gpu.download(&gt_split.buf)?;
            data.data().iter().map(|&v| v as usize).collect()
        } else {
            equal_split(dim_size, num_outputs)
        }
    } else if let Some(s) = get_attr_ints(node, "split") {
        s.iter().map(|&v| v as usize).collect()
    } else {
        equal_split(dim_size, num_outputs)
    };

    // Use channel_split for last axis, general_split otherwise.
    // Both paths batch all outputs into a single compute pass.
    let outputs: Vec<(String, GpuBuffer)> = if actual_axis == rank - 1 {
        let input_buf = &gc.get(name.as_str()).unwrap().buf;
        let sizes: Vec<usize> = split_sizes
            .iter()
            .take(node.outputs.len())
            .copied()
            .collect();
        let bufs = gpu.channel_split_all_on_device(input_buf, &sizes)?;
        bufs.into_iter()
            .enumerate()
            .map(|(i, buf)| (node.outputs[i].clone(), buf))
            .collect()
    } else {
        let input_buf = &gc.get(name.as_str()).unwrap().buf;
        let bufs = gpu.general_split_on_device(input_buf, actual_axis, &split_sizes);
        bufs.into_iter()
            .enumerate()
            .filter(|(i, _)| *i < node.outputs.len())
            .map(|(i, buf)| (node.outputs[i].clone(), buf))
            .collect()
    };

    for (name, buf) in outputs {
        gc.insert(
            name,
            GpuTensor {
                buf,
                nhwc: is_nhwc,
                f16_io: false,
            },
        );
    }
    Ok(())
}

fn equal_split(dim: usize, n: usize) -> Vec<usize> {
    let base = dim / n;
    let rem = dim % n;
    (0..n)
        .map(|i| if i < rem { base + 1 } else { base })
        .collect()
}

// ── GPU-native Resize ────────────────────────────────────────────

fn exec_resize_gpu(
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

    let input_shape = gc.get(name).unwrap().buf.shape().to_vec(); // [N, H, W, C]
    let (ih, iw) = (input_shape[1], input_shape[2]);

    // Determine output size — sizes input or scales
    let (oh, ow) = if node.inputs.len() > 3 && !node.inputs[3].is_empty() {
        // sizes tensor (NCHW format: [n, c, oh, ow])
        to_cpu(gpu, &node.inputs[3], env, gc)?;
        if let Some(sizes) = env.get(&node.inputs[3]) {
            let sd = sizes.data();
            (sd[2] as usize, sd[3] as usize)
        } else {
            (ih, iw)
        }
    } else if node.inputs.len() > 2 && !node.inputs[2].is_empty() {
        // scales tensor
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
        // No-op resize
        if let Some(gt) = gc.remove(name) {
            gc.insert(node.outputs[0].clone(), gt);
        }
        return Ok(());
    }

    let input_buf = &gc.get(name).unwrap().buf;
    let out = gpu.resize_nearest_nhwc_on_device(input_buf, oh, ow);
    gc.insert(
        node.outputs[0].clone(),
        GpuTensor {
            buf: out,
            nhwc: true,
            f16_io: false,
        },
    );
    Ok(())
}

// ── GPU-native Shape ─────────────────────────────────────────────

fn exec_shape_gpu(
    _gpu: &GpuBackend,
    node: &OnnxNode,
    env: &mut TensorEnv,
    gc: &mut GpuCache,
) -> Result<(), OnnxError> {
    let name = &node.inputs[0];

    // If tensor is on GPU, read shape from metadata — no download needed
    if let Some(gt) = gc.get(name) {
        let shape = if gt.nhwc && gt.buf.shape().len() == 4 {
            // Map NHWC [N, H, W, C] → NCHW [N, C, H, W] for shape output
            let s = gt.buf.shape();
            vec![s[0] as f32, s[3] as f32, s[1] as f32, s[2] as f32]
        } else {
            gt.buf.shape().iter().map(|&d| d as f32).collect()
        };
        let out =
            Tensor::from_vec(vec![shape.len()], shape).map_err(|e| OnnxError::DecodeFailed {
                message: e.to_string(),
            })?;
        env.insert(node.outputs[0].clone(), out);
        Ok(())
    } else {
        // Already on CPU
        super::reshape::exec_shape(node, env)
    }
}

// ── GPU-native Reshape ───────────────────────────────────────────

fn exec_reshape_gpu(
    gpu: &GpuBackend,
    node: &OnnxNode,
    env: &mut TensorEnv,
    gc: &mut GpuCache,
) -> Result<(), OnnxError> {
    let name = &node.inputs[0];

    let on_gpu_nhwc_4d = gc
        .get(name)
        .is_some_and(|gt| gt.nhwc && gt.buf.shape().len() == 4);

    if on_gpu_nhwc_4d {
        // Permute NHWC → NCHW on GPU (borrows input, creates new buffer)
        let (nchw_buf, nchw_shape, total) = {
            let buf = &gc.get(name).unwrap().buf;
            let nchw = gpu.nhwc_to_nchw_on_device(buf);
            let shape = nchw.shape().to_vec();
            let total = nchw.len();
            (nchw, shape, total)
        };

        let new_shape = get_reshape_shape(node, env, gpu, gc, &nchw_shape, total)?;
        let out = gpu.reshape_on_device(nchw_buf, new_shape);
        gc.insert(
            node.outputs[0].clone(),
            GpuTensor {
                buf: out,
                nhwc: false,
                f16_io: false,
            },
        );
        Ok(())
    } else if gc.contains_key(name) {
        // On GPU but not NHWC 4D — device-side copy + reshape metadata
        let (shape, total, nhwc) = {
            let gt = gc.get(name).unwrap();
            (gt.buf.shape().to_vec(), gt.buf.len(), gt.nhwc)
        };
        let new_shape = get_reshape_shape(node, env, gpu, gc, &shape, total)?;
        let out = {
            let buf = &gc.get(name).unwrap().buf;
            gpu.copy_reshape_on_device(buf, new_shape)
        };
        gc.insert(
            node.outputs[0].clone(),
            GpuTensor {
                buf: out,
                nhwc,
                f16_io: false,
            },
        );
        Ok(())
    } else {
        // On CPU — ensure inputs are materialized from gc to env
        inputs_to_cpu(gpu, node, env, gc)?;
        if node.op_type == "Flatten" {
            super::reshape::exec_flatten(node, env)
        } else {
            super::reshape::exec_reshape(node, env)
        }
    }
}

fn get_reshape_shape(
    node: &OnnxNode,
    env: &mut TensorEnv,
    gpu: &GpuBackend,
    gc: &mut GpuCache,
    in_shape: &[usize],
    total: usize,
) -> Result<Vec<usize>, yscv_kernels::KernelError> {
    if node.op_type == "Flatten" {
        let axis = get_attr_int(node, "axis").unwrap_or(1) as usize;
        let d0: usize = in_shape[..axis].iter().product();
        let d1: usize = in_shape[axis..].iter().product();
        return Ok(vec![d0, d1]);
    }

    // Get shape tensor data — might need to download from GPU
    let shape_name = &node.inputs[1];
    let shape_data: Vec<f32> = if let Some(st) = env.get(shape_name) {
        st.data().to_vec()
    } else if let Some(gt) = gc.get(shape_name) {
        // Download shape tensor directly (it's small)
        let t = gpu.download(&gt.buf)?;
        t.data().to_vec()
    } else {
        return Ok(vec![]);
    };

    let mut new_s = Vec::with_capacity(shape_data.len());
    let mut neg_idx = None;
    for (i, &dim_f) in shape_data.iter().enumerate() {
        let d = dim_f as i64;
        if d == -1 {
            neg_idx = Some(i);
            new_s.push(1);
        } else if d == 0 {
            new_s.push(if i < in_shape.len() { in_shape[i] } else { 1 });
        } else {
            new_s.push(d as usize);
        }
    }
    if let Some(idx) = neg_idx {
        let known: usize = new_s
            .iter()
            .enumerate()
            .filter(|&(i, _)| i != idx)
            .map(|(_, &d)| d)
            .product();
        new_s[idx] = total.checked_div(known).unwrap_or(total);
    }
    Ok(new_s)
}

// ── GPU-native Unsqueeze / Squeeze ───────────────────────────────

fn exec_unsqueeze_gpu(
    gpu: &GpuBackend,
    node: &OnnxNode,
    env: &mut TensorEnv,
    gc: &mut GpuCache,
) -> Result<(), OnnxError> {
    let name = &node.inputs[0];
    let is_unsqueeze = node.op_type == "Unsqueeze";

    // Small CPU-only tensors: keep on CPU to avoid GPU sync on shape metadata
    if !gc.contains_key(name)
        && let Some(t) = env.get(name)
        && t.data().len() < 64
    {
        return if is_unsqueeze {
            super::reshape::exec_unsqueeze(node, env)
        } else {
            super::reshape::exec_squeeze(node, env)
        };
    }

    // For non-NHWC GPU tensors: device-side copy with new shape (no download)
    if let Some(gt) = gc.get(name) {
        if !gt.nhwc || gt.buf.shape().len() != 4 {
            let mut shape = gt.buf.shape().to_vec();
            let nhwc = gt.nhwc;

            if is_unsqueeze {
                // Get axes from input[1] or attribute
                let axes: Vec<i64> = if node.inputs.len() > 1 && !node.inputs[1].is_empty() {
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
                for &ax in &sorted {
                    shape.insert(ax, 1);
                }
            } else {
                // Squeeze
                let axes: Vec<i64> = if node.inputs.len() > 1 && !node.inputs[1].is_empty() {
                    if let Some(at) = env.get(&node.inputs[1]) {
                        at.data().iter().map(|&v| v as i64).collect()
                    } else {
                        vec![]
                    }
                } else {
                    get_attr_ints(node, "axes").unwrap_or_default()
                };
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
                    to_remove.sort_unstable_by(|a, b| b.cmp(a));
                    for ax in to_remove {
                        if shape[ax] == 1 {
                            shape.remove(ax);
                        }
                    }
                }
            }

            let out = {
                let buf = &gc.get(name).unwrap().buf;
                gpu.copy_reshape_on_device(buf, shape)
            };
            gc.insert(
                node.outputs[0].clone(),
                GpuTensor {
                    buf: out,
                    nhwc,
                    f16_io: false,
                },
            );
            return Ok(());
        }

        // NHWC 4D — complex layout, fall back to CPU
        inputs_to_cpu(gpu, node, env, gc)?;
    }

    if is_unsqueeze {
        super::reshape::exec_unsqueeze(node, env)
    } else {
        super::reshape::exec_squeeze(node, env)
    }
}

// ── GPU-native Slice ─────────────────────────────────────────────

fn exec_slice_gpu(
    gpu: &GpuBackend,
    node: &OnnxNode,
    env: &mut TensorEnv,
    gc: &mut GpuCache,
) -> Result<(), OnnxError> {
    let name = &node.inputs[0];
    if !gc.contains_key(name.as_str()) {
        return super::reshape::exec_slice(node, env);
    }

    // Read slice parameters from CPU env (starts, ends, axes, steps are small tensors)
    let starts_t = get_small_i64_vec(gpu, &node.inputs[1], env, gc)?;
    let ends_t = get_small_i64_vec(gpu, &node.inputs[2], env, gc)?;
    let axes_v = if node.inputs.len() > 3 && !node.inputs[3].is_empty() {
        Some(get_small_i64_vec(gpu, &node.inputs[3], env, gc)?)
    } else {
        None
    };
    let steps_v = if node.inputs.len() > 4 && !node.inputs[4].is_empty() {
        Some(get_small_i64_vec(gpu, &node.inputs[4], env, gc)?)
    } else {
        None
    };

    // Check all steps are 1 (we only handle step=1 on GPU)
    if let Some(ref sv) = steps_v
        && sv.iter().any(|&s| s != 1)
    {
        inputs_to_cpu(gpu, node, env, gc)?;
        return super::reshape::exec_slice(node, env);
    }

    let gt = gc.get(name.as_str()).unwrap();
    let shape = gt.buf.shape();
    let rank = shape.len();
    let is_nhwc = gt.nhwc && rank == 4;

    let mut starts = vec![0i64; rank];
    let mut ends: Vec<i64> = shape.iter().map(|&d| d as i64).collect();

    let axes: Vec<usize> = if let Some(av) = axes_v {
        av.iter()
            .map(|&v| {
                if v < 0 {
                    (rank as i64 + v) as usize
                } else {
                    v as usize
                }
            })
            .collect()
    } else {
        (0..starts_t.len()).collect()
    };

    for (i, &ax) in axes.iter().enumerate() {
        // For NHWC 4D, remap axis from NCHW space
        let actual_ax = if is_nhwc {
            match ax {
                0 => 0,
                1 => 3,
                2 => 1,
                3 => 2,
                a => a,
            }
        } else {
            ax
        };
        starts[actual_ax] = starts_t[i];
        ends[actual_ax] = ends_t[i];
    }

    // Normalize and clamp
    let actual_shape = gt.buf.shape();
    for d in 0..rank {
        let dim = actual_shape[d] as i64;
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
        .map(|d| (ends[d] - starts[d]).max(0) as usize)
        .collect();
    let starts_usize: Vec<usize> = starts.iter().map(|&s| s as usize).collect();

    let out = gpu.slice_on_device(&gt.buf, &starts_usize, &out_shape);
    gc.insert(
        node.outputs[0].clone(),
        GpuTensor {
            buf: out,
            nhwc: is_nhwc,
            f16_io: false,
        },
    );
    Ok(())
}

/// Read a small tensor as Vec<i64> — from env or by downloading from GPU.
fn get_small_i64_vec(
    gpu: &GpuBackend,
    name: &str,
    env: &TensorEnv,
    gc: &GpuCache,
) -> Result<Vec<i64>, yscv_kernels::KernelError> {
    if let Some(t) = env.get(name) {
        return Ok(t.data().iter().map(|&v| v as i64).collect());
    }
    if let Some(gt) = gc.get(name) {
        let t = gpu.download(&gt.buf)?;
        return Ok(t.data().iter().map(|&v| v as i64).collect());
    }
    Ok(vec![])
}

// ── GPU-aware ConstantOfShape ────────────────────────────────────

fn exec_constant_of_shape_gpu(
    gpu: &GpuBackend,
    node: &OnnxNode,
    env: &mut TensorEnv,
    gc: &mut GpuCache,
) -> Result<(), OnnxError> {
    // Read shape from env (preferred, no sync) or gc (small sync)
    let shape_name = &node.inputs[0];
    let shape: Vec<usize> = if let Some(t) = env.get(shape_name) {
        t.data().iter().map(|&v| v as usize).collect()
    } else if let Some(gt) = gc.get(shape_name) {
        let t = gpu.download(&gt.buf)?;
        t.data().iter().map(|&v| v as usize).collect()
    } else {
        return Err(OnnxError::DecodeFailed {
            message: format!(
                "missing shape input '{}' for ConstantOfShape '{}'",
                shape_name, node.name
            ),
        });
    };

    let value = get_attr_float(node, "value").unwrap_or(0.0);
    let total: usize = shape.iter().product();

    // Create filled tensor on CPU — no GPU needed for a constant
    let data = vec![value; total];
    let out = Tensor::from_vec(shape, data).map_err(|e| OnnxError::DecodeFailed {
        message: e.to_string(),
    })?;
    env.insert(node.outputs[0].clone(), out);
    Ok(())
}

// ── GPU-aware Expand ─────────────────────────────────────────────

fn exec_expand_gpu(
    gpu: &GpuBackend,
    node: &OnnxNode,
    env: &mut TensorEnv,
    gc: &mut GpuCache,
) -> Result<(), OnnxError> {
    let data_name = &node.inputs[0];
    let shape_name = &node.inputs[1];

    // If data is on GPU, use broadcast_binary (multiply by 1) to expand
    if gc.contains_key(data_name.as_str()) {
        let target_shape: Vec<usize> = if let Some(t) = env.get(shape_name) {
            t.data().iter().map(|&v| v as usize).collect()
        } else if let Some(gt) = gc.get(shape_name) {
            let t = gpu.download(&gt.buf)?;
            t.data().iter().map(|&v| v as usize).collect()
        } else {
            inputs_to_cpu(gpu, node, env, gc)?;
            return super::reshape::exec_expand(node, env);
        };

        // Create a scalar 1.0 buffer and use broadcast_binary mul
        let ones_t = Tensor::from_vec(
            target_shape.clone(),
            vec![1.0f32; target_shape.iter().product()],
        )
        .map_err(|e| OnnxError::DecodeFailed {
            message: e.to_string(),
        })?;
        let ones_buf = gpu.upload(&ones_t);
        let data_buf = &gc.get(data_name.as_str()).unwrap().buf;
        let nhwc = gc.get(data_name.as_str()).unwrap().nhwc;
        let out = gpu.broadcast_binary_on_device(data_buf, &ones_buf, 2); // mul by 1
        gc.insert(
            node.outputs[0].clone(),
            GpuTensor {
                buf: out,
                nhwc,
                f16_io: false,
            },
        );
        Ok(())
    } else {
        inputs_to_cpu(gpu, node, env, gc)?;
        super::reshape::exec_expand(node, env)
    }
}
