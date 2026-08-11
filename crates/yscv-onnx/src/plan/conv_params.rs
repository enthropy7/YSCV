//! Phase 2: convolution parameters, resolved once.
//!
//! Strides, pads, group and the depthwise/pointwise classification come from
//! node attributes and the weight's shape, neither of which changes between
//! inferences. Resolving them here keeps attribute-map lookups out of the
//! per-inference Conv dispatch.
//!
//! Weight shapes are read as ONNX wrote them. Prepacking happens after this
//! phase and keeps its permuted copies to itself ([`super::prepack`]), so there
//! is one layout to interpret here rather than four.

use rustc_hash::FxHashMap;
use yscv_tensor::Tensor;

use crate::attr::Attr;
use crate::loader::{OnnxAttribute, OnnxNode};

use super::*;

pub(super) fn resolve_conv_params(
    nodes: &[OnnxNode],
    node_kinds: &[NodeKind],
    initializers: &FxHashMap<String, Tensor>,
) -> Vec<Option<ConvParams>> {
    // Pre-parse Conv attributes to avoid FxHashMap lookups in hot path.
    nodes
        .iter()
        .zip(node_kinds.iter())
        .map(|(node, kind)| {
            if !matches!(
                kind,
                NodeKind::Conv | NodeKind::ConvRelu | NodeKind::ConvSilu
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
            // Depthwise/pointwise follow from the weight's shape, read as
            // ONNX-native `[O, I/G, KH, KW]`.
            let weight_name = node.inputs.get(1).map(|s| s.as_str()).unwrap_or("");
            let weight_shape = initializers
                .get(weight_name)
                .map(|t| t.shape().to_vec())
                .unwrap_or_default();
            let (o_ch, kh_w, kw_w) = if weight_shape.len() == 4 {
                (weight_shape[0], weight_shape[2], weight_shape[3])
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
