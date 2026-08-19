//! Layout decisions resolved at plan time.
//!
//! Layout is a property of a *value*, not of an execution. Which physical
//! arrangement a tensor is in — NCHW, NHWC, or the AVX-512 blocked NCHWc16 —
//! follows from the ops that produce and consume it, all of which are known
//! once the plan is built. Deciding it here is what lets the runner stop
//! re-deriving it from tensor shapes on every inference.

use super::NodeAction;
use crate::loader::OnnxNode;

/// For each plan position, whether the fused conv action there should leave its
/// output in blocked NCHWc16 instead of converting back to NHWC.
///
/// The three streaming fused kernels — `FusedDwPw`, `FusedPwDw` and
/// `FusedPwDwPwReduce` — work internally in NCHWc16 on AVX-512. When the next
/// action consumes their output and can take NCHWc16 directly, the pair can skip
/// a round trip through NHWC: one full output pass saved per handoff, on the
/// hottest part of a MobileNet-shaped backbone.
///
/// The runner used to answer this per action per inference, walking forward
/// through the plan and pulling weight shapes back out of the tensor
/// environment. Nothing in that walk is dynamic. The plan structure is fixed at
/// load time and the shapes it reads belong to initializers, so the whole thing
/// is a constant being recomputed a few hundred times a second.
///
/// Indexed by plan position, not node index: this runs after the
/// `FusedPwDwPwReduce` merge has dropped the actions it absorbed, so the two
/// have diverged by the time we get here.
pub(crate) fn resolve_nchwc_handoff(
    plan: &[NodeAction],
    nodes: &[OnnxNode],
    conv_weight: impl Fn(&str) -> Option<Vec<usize>>,
) -> Vec<bool> {
    // The next action that actually runs. Absorbed nodes are left in the plan
    // as `Skip`, so a genuine chain can have them sitting between its links.
    let next_live = |from: usize| -> Option<&NodeAction> {
        plan[from + 1..]
            .iter()
            .find(|a| !matches!(a, NodeAction::Skip))
    };

    // Reads a weight shape the way the runtime does — the packed copy, since
    // the gate below is written against depthwise KHWC `[kH, kW, C, dm]` and
    // the initializer keeps ONNX-native OIHW.
    let weight_shape = |node: &OnnxNode, port: usize| -> Vec<usize> {
        node.inputs
            .get(port)
            .and_then(|n| conv_weight(n))
            .unwrap_or_default()
    };

    plan.iter()
        .enumerate()
        .map(|(idx, action)| match action {
            NodeAction::FusedDwPw { pw_idx, .. } => {
                // Only another `FusedDwPw` reading this output can take the
                // blocked form, and only when it will itself take the AVX-512
                // path — same gate as `exec_fused_dw_pw` applies internally.
                // Handing NCHWc16 to a kernel that then falls into the NHWC
                // path is a crash, not a slowdown.
                let Some(NodeAction::FusedDwPw {
                    dw_idx: next_dw_idx,
                    pw_idx: next_pw_idx,
                    ..
                }) = next_live(idx)
                else {
                    return false;
                };
                let next_dw = &nodes[*next_dw_idx];
                let next_pw = &nodes[*next_pw_idx];
                if next_dw.inputs.first() != nodes[*pw_idx].outputs.first() {
                    return false;
                }
                let dw_ws = weight_shape(next_dw, 1);
                let pw_ws = weight_shape(next_pw, 1);
                dw_ws.len() == 4
                    && pw_ws.len() == 4
                    && dw_ws[0] == 3
                    && dw_ws[1] == 3
                    && dw_ws[3] == 1
                    && pw_ws[0] == 1
                    && pw_ws[1] == 1
                    && pw_ws[2] == dw_ws[2]
                    && dw_ws[2].is_multiple_of(16)
                    && pw_ws[3].is_multiple_of(16)
            }
            NodeAction::FusedPwDw { dw_idx, .. } => {
                consumes_blocked(next_live(idx), nodes, &nodes[*dw_idx].outputs[0])
            }
            NodeAction::FusedPwDwPwReduce {
                pw_reduce_idx,
                residual,
                ..
            } => {
                // The chain's last written value depends on how much of the
                // residual tail the action absorbed.
                let out = match residual {
                    Some(r) if r.post_activation == 1 => &nodes[r.relu_idx as usize].outputs[0],
                    Some(r) => &nodes[r.add_idx].outputs[0],
                    None => &nodes[*pw_reduce_idx].outputs[0],
                };
                consumes_blocked(next_live(idx), nodes, out)
            }
            _ => false,
        })
        .collect()
}

/// Whether `next` is one of the actions that opens with a pointwise conv able to
/// read blocked NCHWc16, and reads `value` as its first input.
fn consumes_blocked(next: Option<&NodeAction>, nodes: &[OnnxNode], value: &str) -> bool {
    let entry_idx = match next {
        Some(
            NodeAction::FusedPwDwPwReduce {
                pw_expand_idx: idx, ..
            }
            | NodeAction::FusedPwDw { pw_idx: idx, .. }
            | NodeAction::ConvAdd { conv_idx: idx, .. },
        ) => *idx,
        _ => return false,
    };
    nodes[entry_idx].inputs.first().is_some_and(|i| i == value)
}
