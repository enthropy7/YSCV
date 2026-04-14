pub(crate) use std::collections::{HashMap, HashSet};

pub(crate) use yscv_kernels::{
    BatchNorm2dParams, add as kernel_add, avg_pool2d_nhwc, batch_norm2d_nhwc, conv2d_nhwc,
    matmul_2d, matmul_2d_slices, max_pool2d_nhwc, mul as kernel_mul, relu, relu_inplace, sigmoid,
    softmax_last_dim, sub as kernel_sub,
};
pub(crate) use yscv_tensor::{DType, Tensor};

pub(crate) use crate::error::OnnxError;
pub(crate) use crate::loader::{OnnxAttribute, OnnxModel, OnnxNode};

mod compare;
mod conv;
mod elementwise;
mod gather_scatter;
#[cfg(feature = "gpu")]
pub(crate) mod gpu;
pub mod kv_cache;
mod linear;
#[cfg(feature = "metal-backend")]
#[allow(unsafe_code)]
#[path = "metal/mod.rs"]
pub mod metal_runner;
mod misc;
mod normalization;
mod pooling;
mod reduce;
mod reshape;

use compare::*;
use conv::*;
use elementwise::*;
use gather_scatter::*;
use linear::*;
use misc::*;
use normalization::*;
use pooling::*;
use reduce::*;
use reshape::*;

/// A tensor environment backed by a `Vec<Option<Tensor>>` for O(1) lookups
/// by integer index. Tensor names are mapped to dense integer IDs during
/// construction, eliminating string hashing in the hot execution loop.
///
/// Model initializers (weights) are referenced without cloning. Only when
/// mutation is needed (get_mut/remove) is a clone-on-write performed.
pub(crate) struct TensorEnv<'m> {
    name_to_id: HashMap<String, usize>,
    slots: Vec<Option<Tensor>>,
    /// Per-slot flag: true if the tensor is stored in NHWC layout.
    nhwc_flags: Vec<bool>,
    /// Slot IDs whose tensors have been pre-permuted from OIHW to KHWC.
    khwc_weights: HashSet<usize>,
    /// Slot IDs whose depthwise weights were pre-permuted [O,1,KH,KW] → [KH,KW,C,dm].
    /// Currently unused (pre-permutation reverted), kept for forward compatibility.
    #[allow(dead_code)]
    dw_khwc_weights: HashSet<usize>,
    /// Counter for dynamically allocated temporary names that were not in
    /// the pre-built mapping (e.g., "__qa", "__qb_mat").
    next_dynamic: usize,
    /// Reference to model initializers for zero-copy weight access.
    initializers: &'m HashMap<String, Tensor>,
}

impl<'m> TensorEnv<'m> {
    /// Build from the model, pre-allocating a slot for every known tensor name.
    /// Holds a reference to model initializers for zero-copy weight access.
    fn from_model(model: &'m OnnxModel) -> Self {
        let mut names: HashSet<&str> = HashSet::new();
        for name in &model.inputs {
            names.insert(name.as_str());
        }
        for name in &model.outputs {
            names.insert(name.as_str());
        }
        for name in model.initializers.keys() {
            names.insert(name.as_str());
        }
        for node in &model.nodes {
            for name in &node.inputs {
                names.insert(name.as_str());
            }
            for name in &node.outputs {
                names.insert(name.as_str());
            }
        }
        let name_to_id: HashMap<String, usize> = names
            .into_iter()
            .enumerate()
            .map(|(id, name)| (name.to_string(), id))
            .collect();
        let num_slots = name_to_id.len();
        let khwc_ids: HashSet<usize> = model
            .khwc_weights
            .iter()
            .filter_map(|name| name_to_id.get(name.as_str()).copied())
            .collect();
        let dw_khwc_ids: HashSet<usize> = model
            .dw_khwc_weights
            .iter()
            .filter_map(|name| name_to_id.get(name.as_str()).copied())
            .collect();
        TensorEnv {
            next_dynamic: num_slots,
            name_to_id,
            slots: vec![None; num_slots],
            nhwc_flags: vec![false; num_slots],
            khwc_weights: khwc_ids,
            dw_khwc_weights: dw_khwc_ids,
            initializers: &model.initializers,
        }
    }

    /// Look up a tensor by name. Falls back to model initializers if the
    /// slot is empty (zero-copy access to weights).
    #[inline]
    pub(crate) fn get(&self, name: &str) -> Option<&Tensor> {
        let id = self.name_to_id.get(name)?;
        self.slots[*id]
            .as_ref()
            .or_else(|| self.initializers.get(name))
    }

    /// Insert a tensor by name. If the name is unknown, a new slot is
    /// allocated dynamically (this handles temporary names created by
    /// quantization ops, etc.).
    #[inline]
    pub(crate) fn insert(&mut self, name: String, tensor: Tensor) {
        if let Some(&id) = self.name_to_id.get(&name) {
            self.slots[id] = Some(tensor);
            if id < self.nhwc_flags.len() {
                self.nhwc_flags[id] = false;
            }
        } else {
            let id = self.next_dynamic;
            self.next_dynamic += 1;
            self.name_to_id.insert(name, id);
            self.slots.push(Some(tensor));
            self.nhwc_flags.push(false);
        }
    }

    /// Get a mutable reference to a tensor by name.
    /// Clone-on-write: if the tensor is only in initializers, clone it into
    /// the slot first.
    #[inline]
    pub(crate) fn get_mut(&mut self, name: &str) -> Option<&mut Tensor> {
        let id = *self.name_to_id.get(name)?;
        if self.slots[id].is_none()
            && let Some(t) = self.initializers.get(name)
        {
            self.slots[id] = Some(t.clone());
        }
        self.slots[id].as_mut()
    }

    /// Remove a tensor by name (sets the slot to `None`).
    /// If the tensor is only in initializers, clone and return it.
    #[inline]
    pub(crate) fn remove(&mut self, name: &str) -> Option<Tensor> {
        let id = *self.name_to_id.get(name)?;
        if id < self.nhwc_flags.len() {
            self.nhwc_flags[id] = false;
        }
        self.slots[id]
            .take()
            .or_else(|| self.initializers.get(name).cloned())
    }

    /// Returns true if the tensor at `name` is stored in NHWC layout.
    #[inline]
    pub(crate) fn is_nhwc(&self, name: &str) -> bool {
        self.name_to_id
            .get(name)
            .map(|&id| self.nhwc_flags.get(id).copied().unwrap_or(false))
            .unwrap_or(false)
    }

    /// Mark the tensor at `name` as being in NHWC layout.
    #[inline]
    pub(crate) fn mark_nhwc(&mut self, name: &str) {
        if let Some(&id) = self.name_to_id.get(name)
            && id < self.nhwc_flags.len()
        {
            self.nhwc_flags[id] = true;
        }
    }

    /// Returns true if the tensor has been pre-permuted to KHWC format.
    #[inline]
    pub(crate) fn is_khwc_weight(&self, name: &str) -> bool {
        self.name_to_id
            .get(name)
            .is_some_and(|&id| self.khwc_weights.contains(&id))
    }

    /// Returns true if the depthwise conv weight has been pre-permuted to
    /// `[KH, KW, C, depth_multiplier]` format at load time.
    /// Currently unused (pre-permutation reverted), kept for forward compatibility.
    #[inline]
    #[allow(dead_code)]
    pub(crate) fn is_dw_khwc_weight(&self, name: &str) -> bool {
        self.name_to_id
            .get(name)
            .is_some_and(|&id| self.dw_khwc_weights.contains(&id))
    }

    /// Create a zero-copy alias: remap `alias_name` to the same slot as
    /// `target_name`. No tensor data is cloned — both names point to the
    /// identical storage. Safe because ONNX outputs are write-once.
    #[inline]
    pub(crate) fn alias(&mut self, alias_name: &str, target_name: &str) {
        let target_id = match self.name_to_id.get(target_name) {
            Some(&id) => id,
            None => return,
        };
        // If the target lives only in `initializers`, materialize it into the
        // slot so the alias name can resolve via `get()` — which otherwise
        // would fall back to `initializers.get(alias_name)` and miss.
        if self.slots[target_id].is_none()
            && let Some(t) = self.initializers.get(target_name)
        {
            self.slots[target_id] = Some(t.clone());
        }
        // Point alias_name to the same slot ID as target_name.
        self.name_to_id.insert(alias_name.to_string(), target_id);
    }
}

/// Convert NHWC tensor to NCHW in-place in the environment.
pub(crate) fn ensure_nchw(env: &mut TensorEnv, name: &str) -> Result<(), OnnxError> {
    if env.is_nhwc(name)
        && let Some(t) = env.remove(name)
    {
        let nchw = t
            .permute(&[0, 3, 1, 2])
            .map_err(|e| OnnxError::DecodeFailed {
                message: e.to_string(),
            })?;
        env.insert(name.to_string(), nchw);
    }
    Ok(())
}

/// Map an axis from NCHW to NHWC for 4D tensors.
pub(crate) fn nchw_axis_to_nhwc(axis: usize) -> usize {
    const MAP: [usize; 4] = [0, 3, 1, 2];
    if axis < 4 { MAP[axis] } else { axis }
}

fn is_nhwc_producer(op_type: &str) -> bool {
    matches!(
        op_type,
        "Conv"
            | "MaxPool"
            | "AveragePool"
            | "GlobalAveragePool"
            | "BatchNormalization"
            | "Conv_Relu"
            | "BatchNormalization_Relu"
            | "Resize"
            | "Upsample"
            | "DeformConv"
    )
}

fn is_passthrough_op(op_type: &str) -> bool {
    matches!(
        op_type,
        "Relu"
            | "Sigmoid"
            | "Tanh"
            | "Exp"
            | "Log"
            | "Neg"
            | "Abs"
            | "Sqrt"
            | "Pow"
            | "Clip"
            | "LeakyRelu"
            | "Elu"
            | "Selu"
            | "Gelu"
            | "Erf"
            | "HardSigmoid"
            | "Softplus"
            | "Softsign"
            | "HardSwish"
            | "Mish"
            | "ThresholdedRelu"
            | "Celu"
            | "Add"
            | "Sub"
            | "Mul"
            | "Div"
            | "Min"
            | "Max"
            | "Dropout"
            | "Identity"
    )
}

/// Execute a node with automatic NHWC layout management.
pub(crate) fn execute_node_with_layout(
    node: &OnnxNode,
    env: &mut TensorEnv,
) -> Result<(), OnnxError> {
    let op = node.op_type.as_str();

    // NHWC producers and adjusted ops handle layout internally
    if is_nhwc_producer(op) || op == "Concat" || op == "Split" || op == "Transpose" || op == "Shape"
    {
        return execute_node(node, env);
    }

    let propagate_nhwc = if is_passthrough_op(op) {
        // Check for mixed 4D layouts
        let has_nhwc = node.inputs.iter().any(|n| !n.is_empty() && env.is_nhwc(n));
        let has_nchw_4d = node
            .inputs
            .iter()
            .any(|n| !n.is_empty() && !env.is_nhwc(n) && env.get(n).is_some_and(|t| t.rank() == 4));
        if has_nhwc && has_nchw_4d {
            // Mixed 4D layouts: convert all to NCHW
            for name in &node.inputs {
                if !name.is_empty() {
                    ensure_nchw(env, name)?;
                }
            }
            false
        } else {
            has_nhwc
        }
    } else {
        // NCHW-required op: ensure all inputs are NCHW
        for name in &node.inputs {
            if !name.is_empty() {
                ensure_nchw(env, name)?;
            }
        }
        false
    };

    execute_node(node, env)?;

    if propagate_nhwc {
        for out in &node.outputs {
            if !out.is_empty() {
                env.mark_nhwc(out);
            }
        }
    }

    Ok(())
}

/// Runs inference on a loaded ONNX model with the given named inputs.
///
/// Returns a map of output-name -> tensor for the graph's declared outputs.
pub fn run_onnx_model(
    model: &OnnxModel,
    inputs: HashMap<String, Tensor>,
) -> Result<HashMap<String, Tensor>, OnnxError> {
    let mut env = TensorEnv::from_model(model);

    // Initializers (weights) are accessed via zero-copy fallback reference
    // in TensorEnv::get(). Only user inputs need to be inserted.
    for (name, tensor) in inputs {
        env.insert(name, tensor);
    }

    // --- Operator fusion: scan for fusible patterns ---
    // Build a set of node indices that should be skipped because they were
    // fused into the preceding node.  We also create synthetic "fused" nodes
    // that carry a combined op_type (e.g. "Conv_Relu").
    let nodes = &model.nodes;
    let mut skip: HashSet<usize> = HashSet::new();

    // Build reference counts: how many nodes consume each tensor as input.
    // Used by SiLU fusions to decide in-place vs allocating path.
    let use_counts: HashMap<&str, usize> = {
        let mut counts: HashMap<&str, usize> = HashMap::new();
        for n in nodes {
            for inp in &n.inputs {
                if !inp.is_empty() {
                    *counts.entry(inp.as_str()).or_insert(0) += 1;
                }
            }
        }
        counts
    };

    for (i, node) in nodes.iter().enumerate() {
        if skip.contains(&i) {
            continue;
        }
        // --- Conv → BatchNorm → Relu 3-node fusion ---
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
            execute_node_with_layout(node, &mut env)?;
            execute_node_with_layout(next, &mut env)?;
            if let Some(tensor) = env.get_mut(&next.outputs[0]) {
                relu_inplace(tensor);
            }
            env.alias(&next2.outputs[0], &next.outputs[0]);
            skip.insert(i + 1);
            skip.insert(i + 2);
            continue;
        }

        // --- Conv + Relu fusion ---
        if node.op_type == "Conv"
            && let Some(next) = nodes.get(i + 1)
            && next.op_type == "Relu"
            && next.inputs.len() == 1
            && next.inputs[0] == node.outputs[0]
        {
            execute_node_with_layout(node, &mut env)?;
            if let Some(tensor) = env.get_mut(&node.outputs[0]) {
                relu_inplace(tensor);
            }
            env.alias(&next.outputs[0], &node.outputs[0]);
            skip.insert(i + 1);
            continue;
        }

        // --- Conv + SiLU fusion (Conv → Sigmoid → Mul) ---
        // Detect Sigmoid at i+1 and Mul at i+2 that form SiLU on Conv output.
        if node.op_type == "Conv" {
            let conv_out = &node.outputs[0];
            // Look for Sigmoid(conv_out) → Mul(conv_out, sigmoid_out) pattern
            let mut silu_mul_idx = None;
            for sig_offset in 1..=2 {
                if let Some(sig) = nodes.get(i + sig_offset)
                    && sig.op_type == "Sigmoid"
                    && sig.inputs.len() == 1
                    && sig.inputs[0] == *conv_out
                {
                    let sig_out = &sig.outputs[0];
                    for mul_offset in (sig_offset + 1)..=(sig_offset + 2) {
                        if let Some(mul) = nodes.get(i + mul_offset)
                            && mul.op_type == "Mul"
                            && mul.inputs.len() == 2
                            && ((mul.inputs[0] == *sig_out && mul.inputs[1] == *conv_out)
                                || (mul.inputs[1] == *sig_out && mul.inputs[0] == *conv_out))
                        {
                            silu_mul_idx = Some((sig_offset, mul_offset, mul.outputs[0].clone()));
                            break;
                        }
                    }
                    if silu_mul_idx.is_some() {
                        break;
                    }
                }
            }
            if let Some((sig_off, mul_off, mul_out)) = silu_mul_idx {
                let conv_out_uses = use_counts.get(conv_out.as_str()).copied().unwrap_or(0);
                if conv_out_uses <= 2 {
                    // Fuse SiLU into Conv GEMM tiles (applied cache-hot after bias).
                    exec_conv(node, &mut env, yscv_kernels::Activation::Silu)?;
                    env.alias(&mul_out, conv_out);
                } else {
                    // Other consumers need raw conv_out — can't fuse.
                    execute_node_with_layout(node, &mut env)?;
                    if let Some(tensor) = env.get(conv_out) {
                        let result = yscv_kernels::silu(tensor);
                        env.insert(mul_out.clone(), result);
                    }
                }
                let is_nhwc = env.is_nhwc(conv_out);
                if is_nhwc {
                    env.mark_nhwc(&mul_out);
                }
                // Execute any intermediate nodes between Conv and Sigmoid,
                // then mark them as done so the main loop doesn't re-execute them.
                for mid in 1..sig_off {
                    if !skip.contains(&(i + mid)) {
                        execute_node_with_layout(&nodes[i + mid], &mut env)?;
                        skip.insert(i + mid);
                    }
                }
                skip.insert(i + sig_off);
                skip.insert(i + mul_off);
                continue;
            }
        }

        // --- BatchNormalization + Relu fusion ---
        if node.op_type == "BatchNormalization"
            && let Some(next) = nodes.get(i + 1)
            && next.op_type == "Relu"
            && next.inputs.len() == 1
            && next.inputs[0] == node.outputs[0]
        {
            execute_node_with_layout(node, &mut env)?;
            if let Some(tensor) = env.get_mut(&node.outputs[0]) {
                relu_inplace(tensor);
            }
            env.alias(&next.outputs[0], &node.outputs[0]);
            skip.insert(i + 1);
            continue;
        }

        // --- Gemm + Relu fusion ---
        if node.op_type == "Gemm"
            && let Some(next) = nodes.get(i + 1)
            && next.op_type == "Relu"
            && next.inputs.len() == 1
            && next.inputs[0] == node.outputs[0]
        {
            execute_node_with_layout(node, &mut env)?;
            if let Some(tensor) = env.get_mut(&node.outputs[0]) {
                relu_inplace(tensor);
            }
            env.alias(&next.outputs[0], &node.outputs[0]);
            skip.insert(i + 1);
            continue;
        }

        // --- Add + Relu fusion ---
        if node.op_type == "Add"
            && let Some(next) = nodes.get(i + 1)
            && next.op_type == "Relu"
            && next.inputs.len() == 1
            && next.inputs[0] == node.outputs[0]
        {
            execute_node_with_layout(node, &mut env)?;
            if let Some(tensor) = env.get_mut(&node.outputs[0]) {
                relu_inplace(tensor);
            }
            env.alias(&next.outputs[0], &node.outputs[0]);
            skip.insert(i + 1);
            continue;
        }

        // --- MatMul + Add fusion (Gemm-like) ---
        if node.op_type == "MatMul"
            && let Some(next) = nodes.get(i + 1)
            && next.op_type == "Add"
            && next.inputs.len() == 2
            && (next.inputs[0] == node.outputs[0] || next.inputs[1] == node.outputs[0])
        {
            execute_node_with_layout(node, &mut env)?;
            execute_node_with_layout(next, &mut env)?;
            skip.insert(i + 1);
            continue;
        }

        // --- Sigmoid + Mul → SiLU fusion ---
        // SiLU(x) = x * sigmoid(x).  Pattern: Sigmoid(x)->y, Mul(x,y)->z
        // Single-pass SIMD kernel avoids separate sigmoid allocation + Mul dispatch.
        if node.op_type == "Sigmoid" && node.inputs.len() == 1 {
            let sig_in = &node.inputs[0];
            let sig_out = &node.outputs[0];
            // Look ahead up to 3 positions for a matching Mul (SiLU pattern).
            let mut found_silu = false;
            for look in 1..=3 {
                if let Some(next) = nodes.get(i + look)
                    && next.op_type == "Mul"
                    && next.inputs.len() == 2
                {
                    let is_silu = (next.inputs[0] == *sig_out && next.inputs[1] == *sig_in)
                        || (next.inputs[1] == *sig_out && next.inputs[0] == *sig_in);
                    if is_silu {
                        let is_nhwc = env.is_nhwc(sig_in);
                        let mul_out = &next.outputs[0];
                        // sig_in is used by Sigmoid + Mul = 2 fused consumers.
                        // Only remove if no other node needs it.
                        let sig_in_uses = use_counts.get(sig_in.as_str()).copied().unwrap_or(0);
                        if sig_in_uses <= 2 {
                            if let Some(mut tensor) = env.remove(sig_in) {
                                yscv_kernels::silu_inplace(&mut tensor);
                                env.insert(mul_out.clone(), tensor);
                            }
                        } else if let Some(x_tensor) = env.get(sig_in) {
                            let result_tensor = yscv_kernels::silu(x_tensor);
                            env.insert(mul_out.clone(), result_tensor);
                        }
                        if is_nhwc {
                            env.mark_nhwc(mul_out);
                        }
                        // Execute any intermediate nodes, then mark them done
                        // so the main loop doesn't re-execute them.
                        for mid in 1..look {
                            if !skip.contains(&(i + mid)) {
                                execute_node_with_layout(&nodes[i + mid], &mut env)?;
                                skip.insert(i + mid);
                            }
                        }
                        skip.insert(i + look);
                        found_silu = true;
                        break;
                    }
                }
            }
            if found_silu {
                continue;
            }
        }

        // Zero-copy Reshape: avoid data clone when the data input has only
        // one consumer (this Reshape node).
        if node.op_type == "Reshape" {
            for name in &node.inputs {
                if !name.is_empty() {
                    ensure_nchw(&mut env, name)?;
                }
            }
            exec_reshape_zerocopy(node, &mut env, &use_counts)?;
            continue;
        }

        execute_node_with_layout(node, &mut env)?;
    }

    // Optional per-op trace for debugging inference divergence.
    if std::env::var("CPU_TRACE").is_ok() {
        for node in nodes {
            for out_name in &node.outputs {
                if let Some(t) = env.get(out_name) {
                    let d = t.data();
                    if d.is_empty() {
                        continue;
                    }
                    let max = d.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                    let min = d.iter().fold(f32::INFINITY, |a, &b| a.min(b));
                    let mean = d.iter().sum::<f32>() / d.len() as f32;
                    let nhwc = if env.is_nhwc(out_name) { " [NHWC]" } else { "" };
                    eprintln!(
                        "[{:>20}] {:60} shape={:?} min={:>10.4} max={:>10.4} mean={:>10.4}{}",
                        node.op_type,
                        out_name,
                        t.shape(),
                        min,
                        max,
                        mean,
                        nhwc,
                    );
                }
            }
        }
    }

    // Ensure all outputs are in NCHW (ONNX standard layout)
    for name in &model.outputs {
        ensure_nchw(&mut env, name)?;
    }

    let mut result = HashMap::new();
    for name in &model.outputs {
        if let Some(t) = env.remove(name) {
            result.insert(name.clone(), t);
        } else if let Some(t) = env.get(name) {
            result.insert(name.clone(), t.clone());
        } else {
            eprintln!("warning: ONNX output '{}' not found in environment", name);
        }
    }
    Ok(result)
}

/// Profile CPU inference: measure per-op-type timing.
pub fn profile_onnx_model_cpu(
    model: &OnnxModel,
    inputs: HashMap<String, Tensor>,
) -> Result<(), OnnxError> {
    use std::time::Instant;

    let mut env = TensorEnv::from_model(model);
    for (name, tensor) in inputs {
        env.insert(name, tensor);
    }

    let nodes = &model.nodes;
    let mut skip: HashSet<usize> = HashSet::new();
    let mut timings: HashMap<String, (f64, usize)> = HashMap::new();
    let mut conv_details: Vec<(String, f64, Vec<usize>, Vec<usize>)> = Vec::new();

    let prof_use_counts: HashMap<&str, usize> = {
        let mut counts: HashMap<&str, usize> = HashMap::new();
        for n in nodes {
            for inp in &n.inputs {
                if !inp.is_empty() {
                    *counts.entry(inp.as_str()).or_insert(0) += 1;
                }
            }
        }
        counts
    };

    for (i, node) in nodes.iter().enumerate() {
        if skip.contains(&i) {
            continue;
        }

        // SiLU fusion in profiler too (with look-ahead)
        if node.op_type == "Sigmoid" && node.inputs.len() == 1 {
            let sig_in = &node.inputs[0];
            let sig_out = &node.outputs[0];
            let mut found_silu = false;
            for look in 1..=3 {
                if let Some(next) = nodes.get(i + look)
                    && next.op_type == "Mul"
                    && next.inputs.len() == 2
                    && ((next.inputs[0] == *sig_out && next.inputs[1] == *sig_in)
                        || (next.inputs[1] == *sig_out && next.inputs[0] == *sig_in))
                {
                    let is_nhwc = env.is_nhwc(sig_in);
                    let mul_out = &next.outputs[0];
                    let start = Instant::now();
                    let sig_in_uses = prof_use_counts.get(sig_in.as_str()).copied().unwrap_or(0);
                    if sig_in_uses <= 2 {
                        if let Some(mut tensor) = env.remove(sig_in) {
                            yscv_kernels::silu_inplace(&mut tensor);
                            env.insert(mul_out.clone(), tensor);
                        }
                    } else if let Some(x_tensor) = env.get(sig_in) {
                        let result_tensor = yscv_kernels::silu(x_tensor);
                        env.insert(mul_out.clone(), result_tensor);
                    }
                    if is_nhwc {
                        env.mark_nhwc(mul_out);
                    }
                    let elapsed = start.elapsed().as_secs_f64() * 1000.0;
                    let entry = timings.entry("SiLU(fused)".to_string()).or_insert((0.0, 0));
                    entry.0 += elapsed;
                    entry.1 += 1;
                    // Execute intermediate nodes, mark done to prevent re-execution.
                    for mid in 1..look {
                        if !skip.contains(&(i + mid)) {
                            let mid_node = &nodes[i + mid];
                            let mid_start = Instant::now();
                            execute_node_with_layout(mid_node, &mut env)?;
                            let mid_elapsed = mid_start.elapsed().as_secs_f64() * 1000.0;
                            let mid_entry =
                                timings.entry(mid_node.op_type.clone()).or_insert((0.0, 0));
                            mid_entry.0 += mid_elapsed;
                            mid_entry.1 += 1;
                            skip.insert(i + mid);
                        }
                    }
                    skip.insert(i + look);
                    found_silu = true;
                    break;
                }
            }
            if found_silu {
                continue;
            }
        }

        let op_type = node.op_type.clone();
        let in_shape = env
            .get(&node.inputs[0])
            .map(|t| t.shape().to_vec())
            .unwrap_or_default();

        let start = Instant::now();
        execute_node_with_layout(node, &mut env)?;
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;

        if op_type == "Conv" {
            let out_shape = env
                .get(&node.outputs[0])
                .map(|t| t.shape().to_vec())
                .unwrap_or_default();
            conv_details.push((node.name.clone(), elapsed, in_shape, out_shape));
        }

        let entry = timings.entry(op_type).or_insert((0.0, 0));
        entry.0 += elapsed;
        entry.1 += 1;
    }

    for name in &model.outputs {
        ensure_nchw(&mut env, name)?;
    }

    println!("\n  ── CPU Profile (per-op timing) ──");
    let mut sorted: Vec<_> = timings.into_iter().collect();
    sorted.sort_by(|a, b| b.1.0.partial_cmp(&a.1.0).unwrap());
    let total: f64 = sorted.iter().map(|(_, (t, _))| t).sum();
    for (op, (time_ms, count)) in &sorted {
        println!("    {:>8.2}ms {:>5}x  {}", time_ms, count, op);
    }
    println!("    {:>8.2}ms  total", total);

    // Per-Conv detail: top 10 slowest
    if !conv_details.is_empty() {
        conv_details.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        println!("\n  ── Top Conv layers ──");
        for (name, ms, in_s, out_s) in conv_details.iter().take(10) {
            println!("    {:>6.2}ms  {:?} → {:?}  {}", ms, in_s, out_s, name);
        }
    }
    Ok(())
}

#[inline]
pub(crate) fn get_tensor<'a>(
    env: &'a TensorEnv,
    node_name: &str,
    input_name: &str,
) -> Result<&'a Tensor, OnnxError> {
    env.get(input_name).ok_or_else(|| OnnxError::MissingInput {
        node: node_name.to_string(),
        input: input_name.to_string(),
    })
}

#[inline]
pub(crate) fn get_attr_ints(node: &OnnxNode, name: &str) -> Option<Vec<i64>> {
    match node.attributes.get(name) {
        Some(OnnxAttribute::Ints(v)) => Some(v.clone()),
        _ => None,
    }
}

#[inline]
pub(crate) fn get_attr_int(node: &OnnxNode, name: &str) -> Option<i64> {
    match node.attributes.get(name) {
        Some(OnnxAttribute::Int(v)) => Some(*v),
        _ => None,
    }
}

#[inline]
pub(crate) fn get_attr_float(node: &OnnxNode, name: &str) -> Option<f32> {
    match node.attributes.get(name) {
        Some(OnnxAttribute::Float(v)) => Some(*v),
        _ => None,
    }
}

#[inline]
pub(crate) fn get_attr_string(node: &OnnxNode, name: &str) -> Option<String> {
    match node.attributes.get(name) {
        Some(OnnxAttribute::String(v)) => Some(v.clone()),
        _ => None,
    }
}

/// Converts inputs in the environment to f32 before executing a node, then converts
/// outputs back to the original dtype. Ops that handle dtypes themselves (Cast, Shape,
/// Identity, quantization ops) are exempt.
fn execute_node(node: &OnnxNode, env: &mut TensorEnv) -> Result<(), OnnxError> {
    // Ops that should NOT have automatic dtype conversion
    let dtype_exempt = matches!(
        node.op_type.as_str(),
        "Cast"
            | "Shape"
            | "Identity"
            | "Constant"
            | "ConstantOfShape"
            | "QuantizeLinear"
            | "DequantizeLinear"
            | "DynamicQuantizeLinear"
            | "QLinearConv"
            | "QLinearMatMul"
            | "MatMulInteger"
            | "ConvInteger"
    );

    // Detect original dtype from first input (if any) and convert inputs to f32
    let orig_dtype = if !dtype_exempt && !node.inputs.is_empty() {
        let first_dt = node
            .inputs
            .iter()
            .filter_map(|name| env.get(name))
            .map(|t| t.dtype())
            .find(|&dt| dt != DType::F32);

        if let Some(dt) = first_dt {
            // Convert all non-f32 inputs to f32 in-place
            for input_name in &node.inputs {
                if let Some(tensor) = env.get(input_name)
                    && tensor.dtype() != DType::F32
                {
                    let converted = tensor.to_dtype(DType::F32);
                    env.insert(input_name.clone(), converted);
                }
            }
            Some(dt)
        } else {
            None
        }
    } else {
        None
    };

    // Execute the actual op
    execute_node_inner(node, env)?;

    // Convert outputs back to original dtype if needed
    if let Some(dt) = orig_dtype {
        for output_name in &node.outputs {
            if let Some(tensor) = env.get(output_name)
                && tensor.dtype() == DType::F32
            {
                let converted = tensor.to_dtype(dt);
                env.insert(output_name.clone(), converted);
            }
        }
    }

    Ok(())
}

fn execute_node_inner(node: &OnnxNode, env: &mut TensorEnv) -> Result<(), OnnxError> {
    match node.op_type.as_str() {
        "Conv" => exec_conv(node, env, yscv_kernels::Activation::None),
        "Relu" => exec_relu(node, env),
        "MaxPool" => exec_max_pool(node, env),
        "AveragePool" => exec_avg_pool(node, env),
        "GlobalAveragePool" => exec_global_avg_pool(node, env),
        "BatchNormalization" => exec_batch_norm(node, env),
        "Flatten" => exec_flatten(node, env),
        "Gemm" => exec_gemm(node, env),
        "MatMul" => exec_matmul(node, env),
        "Add" => exec_add(node, env),
        "Sub" => exec_sub(node, env),
        "Mul" => exec_mul(node, env),
        "Softmax" => exec_softmax(node, env),
        "Sigmoid" => exec_sigmoid(node, env),
        "Reshape" => exec_reshape(node, env),
        "Transpose" => exec_transpose(node, env),
        "Concat" => exec_concat(node, env),
        "Unsqueeze" => exec_unsqueeze(node, env),
        "Squeeze" => exec_squeeze(node, env),
        "Clip" => exec_clip(node, env),
        "Shape" => exec_shape(node, env),
        "Gather" => exec_gather(node, env),
        "Constant" => exec_constant(node, env),
        "Dropout" => exec_dropout(node, env),
        "Pad" => exec_pad(node, env),
        "Pow" => exec_pow(node, env),
        "Sqrt" => exec_sqrt(node, env),
        "Exp" => exec_exp(node, env),
        "Log" => exec_log(node, env),
        "Neg" => exec_neg(node, env),
        "Abs" => exec_abs(node, env),
        "Reciprocal" => exec_reciprocal(node, env),
        "Tanh" => exec_tanh(node, env),
        "Floor" => exec_floor(node, env),
        "Ceil" => exec_ceil(node, env),
        "Equal" => exec_cmp(node, env, 0),
        "Greater" => exec_cmp(node, env, 1),
        "Less" => exec_cmp(node, env, 2),
        "Where" => exec_where(node, env),
        "ReduceMean" => exec_reduce_mean(node, env),
        "ReduceSum" => exec_reduce_sum(node, env),
        "Split" => exec_split(node, env),
        "Slice" => exec_slice(node, env),
        "Expand" => exec_expand(node, env),
        "Tile" => exec_tile(node, env),
        "Cast" => exec_cast(node, env),
        "Div" => exec_div(node, env),
        "Min" => exec_min_max(node, env, false),
        "Max" => exec_min_max(node, env, true),
        "ReduceMax" => exec_reduce_max(node, env),
        "ConvTranspose" => exec_conv_transpose(node, env),
        "DeformConv" => exec_deform_conv(node, env),
        "Resize" => exec_resize(node, env),
        "LeakyRelu" => exec_leaky_relu(node, env),
        "Elu" => exec_elu(node, env),
        "ReduceMin" => exec_reduce_min(node, env),
        "ReduceProd" => exec_reduce_prod(node, env),
        "Identity" => exec_identity(node, env),
        "QuantizeLinear" => exec_quantize_linear(node, env),
        "DequantizeLinear" => exec_dequantize_linear(node, env),
        "Gelu" => exec_gelu(node, env),
        "Erf" => exec_erf(node, env),
        "HardSigmoid" => exec_hard_sigmoid(node, env),
        "InstanceNormalization" => exec_instance_norm(node, env),
        "LpNormalization" => exec_lp_norm(node, env),
        "Upsample" => exec_resize(node, env),
        "Selu" => exec_selu(node, env),
        "Celu" => exec_celu(node, env),
        "ThresholdedRelu" => exec_thresholded_relu(node, env),
        "Hardmax" => exec_hardmax(node, env),
        "OneHot" => exec_onehot(node, env),
        "Range" => exec_range(node, env),
        "NonZero" => exec_nonzero(node, env),
        "LayerNormalization" => exec_layer_norm(node, env),
        "GatherElements" => exec_gather_elements(node, env),
        "ScatterElements" => exec_scatter_elements(node, env),
        "Einsum" => exec_einsum(node, env),
        "ReduceL2" => exec_reduce_l2(node, env),
        "ReduceL1" => exec_reduce_l1(node, env),
        "CumSum" => exec_cumsum(node, env),
        "ArgMax" => exec_argmax(node, env),
        "ArgMin" => exec_argmin(node, env),
        "TopK" => exec_topk(node, env),
        "ScatterND" => exec_scatter_nd(node, env),
        "GatherND" => exec_gather_nd(node, env),
        "DepthToSpace" => exec_depth_to_space(node, env),
        "SpaceToDepth" => exec_space_to_depth(node, env),
        "GridSample" => exec_grid_sample(node, env),
        "RoiAlign" => exec_roi_align(node, env),
        "Compress" => exec_compress(node, env),
        "QLinearConv" => exec_qlinear_conv(node, env),
        "QLinearMatMul" => exec_qlinear_matmul(node, env),
        "MatMulInteger" => exec_matmul_integer(node, env),
        "ConvInteger" => exec_conv_integer(node, env),
        "DynamicQuantizeLinear" => exec_dynamic_quantize_linear(node, env),
        "Not" => exec_not(node, env),
        "And" => exec_logical_bin(node, env, 0),
        "Or" => exec_logical_bin(node, env, 1),
        "Xor" => exec_logical_bin(node, env, 2),
        "Sin" => exec_tensor_op(node, env, |t| t.sin()),
        "Cos" => exec_tensor_op(node, env, |t| t.cos()),
        "Tan" => exec_unary(node, env, |v| v.tan()),
        "Asin" => exec_unary(node, env, |v| v.asin()),
        "Acos" => exec_unary(node, env, |v| v.acos()),
        "Atan" => exec_unary(node, env, |v| v.atan()),
        "Sinh" => exec_unary(node, env, |v| v.sinh()),
        "Cosh" => exec_unary(node, env, |v| v.cosh()),
        "Asinh" => exec_unary(node, env, |v| v.asinh()),
        "Acosh" => exec_unary(node, env, |v| v.acosh()),
        "Atanh" => exec_unary(node, env, |v| v.atanh()),
        "Round" => exec_unary(node, env, |v| v.round()),
        "Sign" => exec_unary(node, env, |v| v.signum()),
        "IsNaN" => exec_unary(node, env, |v| if v.is_nan() { 1.0 } else { 0.0 }),
        "IsInf" => exec_unary(node, env, |v| if v.is_infinite() { 1.0 } else { 0.0 }),
        "Mod" => exec_mod(node, env),
        "GreaterOrEqual" => exec_cmp(node, env, 3),
        "LessOrEqual" => exec_cmp(node, env, 4),
        "BitShift" => exec_bitshift(node, env),
        "Mean" => exec_variadic_mean(node, env),
        "Sum" => exec_variadic_sum(node, env),
        "ConstantOfShape" => exec_constant_of_shape(node, env),
        "LRN" => exec_lrn(node, env),
        "Softplus" => exec_unary(node, env, |v| (1.0 + v.exp()).ln()),
        "Softsign" => exec_unary(node, env, |v| v / (1.0 + v.abs())),
        "HardSwish" => exec_unary(node, env, |v| v * ((v + 3.0).clamp(0.0, 6.0) / 6.0)),
        "Mish" => exec_unary(node, env, |v| v * (1.0 + v.exp()).ln().tanh()),
        "NonMaxSuppression" => exec_nms(node, env),
        "GroupQueryAttention" => exec_grouped_query_attention(node, env),
        "Conv_Relu" => {
            exec_conv(node, env, yscv_kernels::Activation::None)?;
            exec_relu_inplace(node, env)
        }
        "BatchNormalization_Relu" => {
            exec_batch_norm(node, env)?;
            exec_relu_inplace(node, env)
        }
        other => Err(OnnxError::UnsupportedOpType {
            op_type: other.to_string(),
        }),
    }
}
