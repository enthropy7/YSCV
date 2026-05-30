//! Layout management around node execution: NCHW/NHWC coercion,
//! Reshape passthrough, and the kind-aware execute wrappers.

use super::*;

pub(crate) fn should_use_prepacked_i8_b(m: usize, k: usize, n: usize) -> bool {
    // For VNNI-friendly tracker pointwise Conv shapes the load-time packed
    // 4x16 RHS avoids per-inference B packing. The large MatMul gate keeps
    // the previous prepacked path for LLM/head-like regimes.
    (m >= 4 && k >= 4 && n.is_multiple_of(16)) || (m <= 64 && k >= 512 && n >= 1024)
}

/// Convert NHWC tensor to NCHW in-place in the environment.
pub(crate) fn ensure_nchw(env: &mut TensorEnv, name: &str) -> Result<(), OnnxError> {
    if env.is_nhwc(name)
        && let Some(t) = env.remove(name)
    {
        // SIMD fast path: `[N, H, W, C]` → `[N, C, H, W]` via 8×8 (x86 AVX) or
        // 4×4 (aarch64 NEON) block transpose. Falls back to the scalar tiled
        // permute internally when dims don't align or the host lacks the
        // required ISA. The SIMD path is much faster than the scalar permute
        // on the Reshape inputs.
        let nchw = if t.rank() == 4 {
            yscv_kernels::nhwc_to_nchw_fast(&t).map_err(|e| OnnxError::DecodeFailed {
                message: e.to_string(),
            })?
        } else {
            t.permute(&[0, 3, 1, 2])
                .map_err(|e| OnnxError::DecodeFailed {
                    message: e.to_string(),
                })?
        };
        env.insert(name.to_string(), nchw);
    }
    Ok(())
}

/// Map an axis from NCHW to NHWC for 4D tensors.
pub(crate) fn nchw_axis_to_nhwc(axis: usize) -> usize {
    const MAP: [usize; 4] = [0, 3, 1, 2];
    if axis < 4 { MAP[axis] } else { axis }
}

/// Env-cached `YSCV_RESHAPE_NHWC_PASSTHROUGH_OFF=1` kill switch for the
/// NHWC-passthrough fast path.
pub(crate) fn reshape_nhwc_passthrough_disabled() -> bool {
    use std::sync::OnceLock;
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| std::env::var_os("YSCV_RESHAPE_NHWC_PASSTHROUGH_OFF").is_some())
}

/// Fast path: avoid the `ensure_nchw` permute before a
/// `Reshape` when the input is NHWC physical rank-4 `[N,H,W,C]` and the
/// reshape merges spatial dims into `[N, C, H*W]` (the model's NCHW
/// logical view). The NHWC memory order is already `[N, H*W, C]` which
/// is what a downstream `Transpose(perm=[0,2,1])+MatMul` consumes as
/// its post-transpose A — `exec_fused_transpose_matmul` honours the
/// NHWC tag and switches to a non-transposed matmul kernel.
///
/// Returns `Ok(true)` when the fast path handled the reshape (caller
/// should skip the default ensure_nchw+reshape path); `Ok(false)`
/// otherwise.
pub(crate) fn try_reshape_nhwc_passthrough(
    node: &OnnxNode,
    env: &mut TensorEnv,
    use_counts: &HashMap<String, usize>,
) -> Result<bool, OnnxError> {
    if node.inputs.len() < 2 || node.inputs[0].is_empty() {
        return Ok(false);
    }
    if !env.is_nhwc(&node.inputs[0]) {
        return Ok(false);
    }
    let output_name = match node.outputs.first() {
        Some(n) if !n.is_empty() => n,
        _ => return Ok(false),
    };
    if !env.reshape_nhwc_passthrough_safe.contains(output_name) {
        return Ok(false);
    }
    let in_shape = match env.get(&node.inputs[0]) {
        Some(t) if t.rank() == 4 => t.shape().to_vec(),
        _ => return Ok(false),
    };
    let n = in_shape[0];
    let h = in_shape[1];
    let w = in_shape[2];
    let c = in_shape[3];
    let total: usize = in_shape.iter().product();
    let target_raw: Vec<i64> = match env.get(&node.inputs[1]) {
        Some(t) => t.data().iter().map(|&v| v as i64).collect(),
        None => return Ok(false),
    };
    let mut target: Vec<usize> = Vec::with_capacity(target_raw.len());
    let mut neg_idx: Option<usize> = None;
    for (i, &d) in target_raw.iter().enumerate() {
        if d == -1 {
            neg_idx = Some(i);
            target.push(1);
        } else if d == 0 {
            target.push(if i < in_shape.len() { in_shape[i] } else { 1 });
        } else {
            target.push(d as usize);
        }
    }
    if let Some(idx) = neg_idx {
        let known: usize = target
            .iter()
            .enumerate()
            .filter(|&(i, _)| i != idx)
            .map(|(_, &d)| d)
            .product();
        target[idx] = total.checked_div(known.max(1)).unwrap_or(total);
    }
    if target.len() != 3 {
        return Ok(false);
    }
    if target[0] != n || target[1] != c || target[2] != h * w {
        return Ok(false);
    }
    // Metadata-only reshape: keep the NHWC physical data, set the
    // model's NCHW logical shape, and keep the NHWC tag so the
    // FusedTransposeMatMul consumer can adjust.
    let sole_consumer = use_counts
        .get(node.inputs[0].as_str())
        .copied()
        .unwrap_or(0)
        <= 1;
    let new_shape = vec![n, c, h * w];
    let out = if sole_consumer {
        let input = env
            .remove(&node.inputs[0])
            .ok_or_else(|| OnnxError::MissingInput {
                node: node.name.clone(),
                input: node.inputs[0].clone(),
            })?;
        input
            .into_reshape(new_shape)
            .map_err(|e| OnnxError::DecodeFailed {
                message: e.to_string(),
            })?
    } else {
        let input = get_tensor(env, &node.name, &node.inputs[0])?;
        input
            .reshape(new_shape)
            .map_err(|e| OnnxError::DecodeFailed {
                message: e.to_string(),
            })?
    };
    env.insert(node.outputs[0].clone(), out);
    env.mark_nhwc(&node.outputs[0]);
    Ok(true)
}

/// Same as [`try_reshape_nhwc_passthrough`] but without a borrowed
/// `use_counts` table — falls back to `reshape` (CoW clone) rather
/// than the `remove`+`into_reshape` zero-copy path. Used by the
/// plan-based `execute_node_with_layout_kind_inner` dispatch.
fn try_reshape_nhwc_passthrough_inner(
    node: &OnnxNode,
    env: &mut TensorEnv,
) -> Result<bool, OnnxError> {
    if node.inputs.len() < 2 || node.inputs[0].is_empty() {
        return Ok(false);
    }
    if !env.is_nhwc(&node.inputs[0]) {
        return Ok(false);
    }
    // Safety gate: only fire when the loader has confirmed the output
    // is consumed exclusively by a `FusedTransposeMatMul` (the single op
    // that honours our NHWC tag).
    let output_name = match node.outputs.first() {
        Some(n) if !n.is_empty() => n,
        _ => return Ok(false),
    };
    if !env.reshape_nhwc_passthrough_safe.contains(output_name) {
        return Ok(false);
    }
    let in_shape = match env.get(&node.inputs[0]) {
        Some(t) if t.rank() == 4 => t.shape().to_vec(),
        _ => return Ok(false),
    };
    let n = in_shape[0];
    let h = in_shape[1];
    let w = in_shape[2];
    let c = in_shape[3];
    let total: usize = in_shape.iter().product();
    // Resolve the target shape via the ONNX Reshape spec: `-1` means
    // "infer from total", `0` means "preserve input dim at that index".
    let target_raw: Vec<i64> = match env.get(&node.inputs[1]) {
        Some(t) => t.data().iter().map(|&v| v as i64).collect(),
        None => return Ok(false),
    };
    let mut target: Vec<usize> = Vec::with_capacity(target_raw.len());
    let mut neg_idx: Option<usize> = None;
    for (i, &d) in target_raw.iter().enumerate() {
        if d == -1 {
            neg_idx = Some(i);
            target.push(1);
        } else if d == 0 {
            target.push(if i < in_shape.len() { in_shape[i] } else { 1 });
        } else {
            target.push(d as usize);
        }
    }
    if let Some(idx) = neg_idx {
        let known: usize = target
            .iter()
            .enumerate()
            .filter(|&(i, _)| i != idx)
            .map(|(_, &d)| d)
            .product();
        target[idx] = total.checked_div(known.max(1)).unwrap_or(total);
    }
    if target.len() != 3 || target[0] != n || target[1] != c || target[2] != h * w {
        return Ok(false);
    }
    let new_shape = vec![n, c, h * w];
    // Always CoW-reshape (don't `env.remove`): the FusedDwPw output may
    // have multiple consumers (e.g. cls_encode's m11s.0 output goes to
    // BOTH cls_encode/Reshape AND cls_dw/Concat). `Tensor::reshape` is
    // O(1) thanks to Arc-shared storage; the slot stays populated for
    // siblings.
    let input = get_tensor(env, &node.name, &node.inputs[0])?;
    let out = input
        .reshape(new_shape)
        .map_err(|e| OnnxError::DecodeFailed {
            message: e.to_string(),
        })?;
    env.insert(node.outputs[0].clone(), out);
    env.mark_nhwc(&node.outputs[0]);
    Ok(true)
}

#[inline]
pub(crate) fn node_kind(node_kinds: &[NodeKind], nodes: &[OnnxNode], idx: usize) -> NodeKind {
    if let Some(kind) = node_kinds.get(idx).copied() {
        kind
    } else {
        NodeKind::from_op_type(&nodes[idx].op_type)
    }
}

/// CPU-side fallback entry point for the wgpu GPU runner.
///
/// Why: when the wgpu backend hits an unsupported op or a degenerate
/// tensor (size < 4 for vec4 shaders, scalar shape metadata), it
/// rematerialises inputs on the CPU and delegates execution back here.
/// This wrapper hides `NodeKind` classification from the GPU module so
/// it doesn't need to know the dispatch taxonomy. Goes through
/// `execute_node_kind` (no layout management) — the wgpu path stages
/// its own NHWC/NCHW conversions in `inputs_to_cpu`, adding the layout
/// pass here would double-flip them.
#[cfg(feature = "gpu")]
#[inline]
pub(crate) fn execute_node_cpu_fallback(
    node: &OnnxNode,
    env: &mut TensorEnv,
) -> Result<(), OnnxError> {
    let kind = NodeKind::from_op_type(&node.op_type);
    execute_node_kind(node, env, kind)
}

/// CPU pre-pass entry point for the Metal `compile_metal_plan` walker.
///
/// Different from the wgpu fallback above: the Metal compile pass walks
/// every node on CPU to gather tensor shapes + intermediate data
/// before recording Metal commands. It is a full ONNX run, so it needs
/// the same automatic NHWC layout management that the regular CPU
/// runner does, not the bare `execute_node_kind`. Kept under its own
/// name (and not the `cpu_fallback` one used by wgpu) so that builds
/// with `--features "gpu metal-backend"` get both entry points without
/// symbol collision and without the wgpu fallback inheriting the
/// layout pass.
#[cfg(feature = "metal-backend")]
#[inline]
pub(crate) fn execute_node_cpu_for_metal_compile(
    node: &OnnxNode,
    env: &mut TensorEnv,
) -> Result<(), OnnxError> {
    let kind = NodeKind::from_op_type(&node.op_type);
    execute_node_with_layout_kind(node, env, kind)
}

#[inline]
fn is_nhwc_producer_with_kind(kind: NodeKind, op_type: &str) -> bool {
    matches!(
        kind,
        NodeKind::Conv | NodeKind::ConvRelu | NodeKind::ConvSilu | NodeKind::BatchNormalization
    ) || matches!(
        op_type,
        "MaxPool"
            | "AveragePool"
            | "GlobalAveragePool"
            | "BatchNormalization_Relu"
            | "Resize"
            | "Upsample"
            | "DeformConv"
    )
}

#[inline]
fn is_passthrough_op_with_kind(kind: NodeKind, op_type: &str) -> bool {
    if matches!(
        kind,
        NodeKind::Relu | NodeKind::Sigmoid | NodeKind::Add | NodeKind::Mul
    ) {
        return true;
    }
    matches!(
        op_type,
        "Tanh"
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
            | "Sub"
            | "Div"
            | "Min"
            | "Max"
            | "Dropout"
            | "Identity"
    )
}

/// Execute a node with automatic NHWC layout management.
#[inline]
pub(crate) fn execute_node_with_layout_kind(
    node: &OnnxNode,
    env: &mut TensorEnv,
    kind: NodeKind,
) -> Result<(), OnnxError> {
    let trace = std::env::var("YSCV_TRACE_SHAPES")
        .ok()
        .filter(|v| v != "0")
        .is_some();
    let result = execute_node_with_layout_kind_inner(node, env, kind);
    if trace && result.is_ok() {
        let shapes: Vec<String> = node
            .outputs
            .iter()
            .filter(|n| !n.is_empty())
            .map(|n| {
                env.get(n)
                    .map(|t| format!("{:?}", t.shape()))
                    .unwrap_or_else(|| "?".to_string())
            })
            .collect();
        eprintln!(
            "[trace] {:>20}  {} -> {}",
            node.op_type,
            node.name,
            shapes.join(", ")
        );
    }
    result.map_err(|e| match e {
        // Already tagged — pass through unchanged.
        OnnxError::DecodeFailed { message } if message.starts_with("node ") => {
            OnnxError::DecodeFailed { message }
        }
        other => OnnxError::DecodeFailed {
            message: format!(
                "node {} ({}): {other}",
                if node.name.is_empty() {
                    node.outputs.first().map(String::as_str).unwrap_or("?")
                } else {
                    node.name.as_str()
                },
                node.op_type
            ),
        },
    })
}

fn execute_node_with_layout_kind_inner(
    node: &OnnxNode,
    env: &mut TensorEnv,
    kind: NodeKind,
) -> Result<(), OnnxError> {
    let op = node.op_type.as_str();

    // NHWC-passthrough Reshape. When the input is NHWC physical
    // rank-4 `[N,H,W,C]` and the model reshapes to `[N, C, H*W]` (the
    // typical "merge spatial dims" pattern after a Conv head), skip the
    // `ensure_nchw` permute and keep the NHWC tag on the output.
    // The loader has confirmed the single consumer is a
    // `FusedTransposeMatMul(perm=[0,2,1])` which honours the tag and
    // switches to a non-transposed matmul kernel.
    if (kind == NodeKind::Reshape || op == "Reshape")
        && !reshape_nhwc_passthrough_disabled()
        && let Ok(true) = try_reshape_nhwc_passthrough_inner(node, env)
    {
        return Ok(());
    }

    // NHWC producers and adjusted ops handle layout internally
    if is_nhwc_producer_with_kind(kind, op)
        || op == "Concat"
        || op == "Split"
        || op == "Transpose"
        || op == "Shape"
    {
        return execute_node_kind(node, env, kind);
    }

    let propagate_nhwc = if is_passthrough_op_with_kind(kind, op) {
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

    execute_node_kind(node, env, kind)?;

    if propagate_nhwc {
        for out in &node.outputs {
            if !out.is_empty() {
                env.mark_nhwc(out);
            }
        }
    }

    Ok(())
}
