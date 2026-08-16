//! Phase 2: convolution parameters, resolved once.
//!
//! Strides, pads, group and the depthwise/pointwise classification come from
//! node attributes and the weight's shape, neither of which changes between
//! inferences. Resolving them here keeps attribute-map lookups out of the
//! per-inference Conv dispatch.
//!
//! Note the weight-shape reading has to account for the loader having
//! pre-permuted conv weights into one of four layouts — the same leak the IR's
//! `WeightLayout` works around, and which this stage is eventually meant to fix
//! by moving the permute into `prepack_weights`.

use rustc_hash::{FxHashMap, FxHashSet};
use yscv_tensor::Tensor;

use crate::attr::Attr;
use crate::loader::{OnnxAttribute, OnnxNode};

use super::*;

pub(super) fn resolve_conv_params(
    nodes: &[OnnxNode],
    node_kinds: &[NodeKind],
    initializers: &FxHashMap<String, Tensor>,
    khwc_weights: &FxHashSet<String>,
    dw_khwc_weights: &FxHashSet<String>,
    group_khwc_weights: &FxHashSet<String>,
) -> Vec<Option<ConvParams>> {
    // Pre-parse Conv attributes to avoid FxHashMap lookups in hot path.
    nodes
        .iter()
        .zip(node_kinds.iter())
        .map(|(node, kind)| {
            if !matches!(
                kind,
                NodeKind::Conv | NodeKind::ConvRelu | NodeKind::ConvSilu | NodeKind::ConvHardSwish
            ) {
                return None;
            }
            let strides = node
                .attributes
                .get(&Attr::Strides)
                .and_then(|a| {
                    if let OnnxAttribute::Ints(v) = a {
                        Some(v.clone())
                    } else {
                        None
                    }
                })
                .unwrap_or_else(|| vec![1, 1]);
            let pads = node
                .attributes
                .get(&Attr::Pads)
                .and_then(|a| {
                    if let OnnxAttribute::Ints(v) = a {
                        Some(v.clone())
                    } else {
                        None
                    }
                })
                .unwrap_or_else(|| vec![0, 0, 0, 0]);
            let group = node
                .attributes
                .get(&Attr::Group)
                .and_then(|a| {
                    if let OnnxAttribute::Int(v) = a {
                        Some(*v as usize)
                    } else {
                        None
                    }
                })
                .unwrap_or(1);
            let (pt, pl, pb, pr) = (
                pads[0] as usize,
                pads[1] as usize,
                pads.get(2).copied().unwrap_or(0) as usize,
                pads.get(3).copied().unwrap_or(0) as usize,
            );
            // Determine depthwise/pointwise from weight shape. Weights
            // may already be permuted to KHWC `[KH, KW, I, O]` by the
            // load-time normalization above (`khwc_weights` pass) for
            // group==1 Conv. Check both layouts and infer which applies.
            //
            // Must dispatch by layout: KHWC-permuted weights are [KH, KW, I, O]
            // (shape[2]=I, shape[3]=O), not OIHW. Reading shape[2]/shape[3] as
            // kernel dims on a KHWC 1×1 weight misclassifies it as non-pointwise
            // and the Conv_Add fast path never fires.
            let weight_name = node.inputs.get(1).map(|s| s.as_str()).unwrap_or("");
            let weight_shape = initializers
                .get(weight_name)
                .map(|t| t.shape().to_vec())
                .unwrap_or_default();
            let weight_is_khwc = khwc_weights.contains(weight_name);
            let weight_is_dw_khwc = dw_khwc_weights.contains(weight_name);
            let weight_is_group_khwc = group_khwc_weights.contains(weight_name);
            // the loader permutes three KHWC
            // variants. DW-permuted `[KH, KW, C, dm]` and grouped
            // `[O, KH, KW, I/G]` previously fell through to the OIHW
            // branch and produced garbage, wrongly setting
            // `is_depthwise = false` for every tracker DW conv. That
            // blocked `FusedDwPw` detection. With the pure-compute
            // `conv_compute_nhwc` split the fused path now keeps the
            // DW intermediate as a local `Tensor` (no env traffic),
            // so enabling this detection no longer regresses tracker.
            let (o_ch, kh_w, kw_w) = if weight_shape.len() == 4 {
                if weight_is_dw_khwc {
                    // Depthwise KHWC: `[KH, KW, C, depth_multiplier]`.
                    let dm = weight_shape[3];
                    (weight_shape[2] * dm, weight_shape[0], weight_shape[1])
                } else if weight_is_group_khwc {
                    // Grouped KHWC: `[O, KH, KW, I/G]`.
                    (weight_shape[0], weight_shape[1], weight_shape[2])
                } else if weight_is_khwc {
                    // Regular KHWC: `[KH, KW, I, O]`.
                    (weight_shape[3], weight_shape[0], weight_shape[1])
                } else {
                    // Plain OIHW: `[O, I, KH, KW]`.
                    (weight_shape[0], weight_shape[2], weight_shape[3])
                }
            } else {
                (0, 0, 0)
            };
            let is_depthwise = group > 1 && group == o_ch;
            let is_pointwise = kh_w == 1 && kw_w == 1 && group == 1;

            Some(ConvParams {
                stride_h: strides[0] as usize,
                stride_w: strides.get(1).copied().unwrap_or(1) as usize,
                pad_top: pt,
                pad_left: pl,
                pad_bottom: pb,
                pad_right: pr,
                group,
                has_padding: pt > 0 || pl > 0 || pb > 0 || pr > 0,
                is_depthwise,
                is_pointwise,
            })
        })
        .collect()
}
