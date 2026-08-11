use yscv_tensor::Tensor;

use crate::error::OnnxError;
use crate::ir::{Changed, Graph, NodeId, Op, Pass, ValueId};

/// Absorbs a constant `Mul` or `Add` that immediately follows a `Conv` into the
/// convolution's own weights and bias.
///
/// Both are the same rewrite over a per-output-channel constant `k`, differing
/// only in what they touch:
///
/// ```text
/// Mul:  W'[c] = W[c] * k[c],  b'[c] = b[c] * k[c]
/// Add:  W'    = W,            b'[c] = b[c] + k[c]
/// ```
///
/// Mirrors ORT's Level-1 ConvMulFusion and ConvAddFusion. `fold_conv_mul` and
/// `fold_conv_add_const` were separate files that shared a matcher, a layout
/// branch chain and an O(N) consumer rescan; they are one pass parameterized
/// twice.
///
/// Runs after `fold_conv_bn`, which normally absorbs the scale and shift
/// already — what reaches here is the stray scale or bias left over.
pub(crate) struct FoldConvConstBinary {
    name: &'static str,
    /// The binary operator being absorbed.
    op: Op,
}

impl FoldConvConstBinary {
    pub(crate) fn mul() -> Self {
        Self {
            name: "fold_conv_mul",
            op: Op::Mul,
        }
    }

    pub(crate) fn add() -> Self {
        Self {
            name: "fold_conv_add_const",
            op: Op::Add,
        }
    }

    /// Whether the constant also scales the weight, or only shifts the bias.
    fn scales_weight(&self) -> bool {
        self.op == Op::Mul
    }
}

impl Pass for FoldConvConstBinary {
    fn name(&self) -> &'static str {
        self.name
    }

    fn run(&self, graph: &mut Graph) -> Result<Changed, OnnxError> {
        let mut changed = false;

        for conv_id in graph.node_ids().collect::<Vec<_>>() {
            let Some(fold) = self.match_fold(graph, conv_id) else {
                continue;
            };

            if let Some(weight) = fold.fused_weight {
                graph.set_constant(fold.weight, weight);
            }
            match fold.bias {
                Some(bias) => graph.set_constant(bias, fold.fused_bias.clone()),
                None => {
                    let name = graph.fresh_value_name(conv_id, "fused_bias");
                    let bias = graph.value_by_name(&name);
                    graph.set_constant(bias, fold.fused_bias.clone());
                    graph.push_input(conv_id, bias);
                }
            }
            graph.absorb_consumer(conv_id, fold.binary_id);
            changed = true;
        }

        Ok(changed)
    }
}

/// Resolved rewrite, computed before anything is mutated so a rejection cannot
/// leave a half-folded weight behind.
struct Fold {
    binary_id: NodeId,
    weight: ValueId,
    bias: Option<ValueId>,
    /// `None` for `Add`, which leaves the weight alone.
    fused_weight: Option<Tensor>,
    fused_bias: Tensor,
}

impl FoldConvConstBinary {
    fn match_fold(&self, graph: &Graph, conv_id: NodeId) -> Option<Fold> {
        let conv = graph.node(conv_id)?;
        if conv.op != Op::Conv {
            return None;
        }

        // The binary op must be the only reader of the Conv's output, or the
        // unscaled values are still observable elsewhere.
        let use_site = graph.sole_consumer(*conv.outputs.first()?)?;
        let binary = graph.node(use_site.node)?;
        if binary.op != self.op || binary.inputs.len() != 2 {
            return None;
        }

        // The other operand has to be a load-time constant.
        let other_port = 1 - use_site.port as usize;
        let constant_value = (*binary.inputs.get(other_port)?)?;

        let weight = (*conv.inputs.get(1)?)?;
        let w = graph.constant(weight)?;
        // Same aliasing rule as fold_conv_bn: rewriting a weight another Conv
        // shares would apply this constant to that one too.
        if graph.use_count(weight) != 1 {
            return None;
        }

        let shape = w.shape().to_vec();
        let out_channels = *shape.first()?;
        if out_channels == 0 || shape.len() < 4 {
            return None;
        }
        let k = broadcast_to_channels(graph, constant_value, out_channels)?;

        let bias = conv.inputs.get(2).copied().flatten();
        if let Some(b) = bias
            && graph.use_count(b) != 1
        {
            return None;
        }
        let old_bias = match bias {
            Some(b) => graph.constant(b).map(|t| t.data().to_vec())?,
            None => vec![0.0; out_channels],
        };
        if old_bias.len() < out_channels {
            return None;
        }

        let (fused_weight, fused_bias_data) = if self.scales_weight() {
            let mut w_data = w.data().to_vec();
            let per_channel = w_data.len() / out_channels;
            for (i, v) in w_data.iter_mut().enumerate() {
                *v *= k[i / per_channel];
            }
            let bias = (0..out_channels).map(|c| old_bias[c] * k[c]).collect();
            (Some(Tensor::from_vec(shape, w_data).ok()?), bias)
        } else {
            let bias = (0..out_channels).map(|c| old_bias[c] + k[c]).collect();
            (None, bias)
        };

        Some(Fold {
            binary_id: use_site.node,
            weight,
            bias,
            fused_weight,
            fused_bias: Tensor::from_vec(vec![out_channels], fused_bias_data).ok()?,
        })
    }
}

/// Expands a constant operand to one value per output channel, or declines if
/// it is not a per-channel constant at all.
///
/// ONNX broadcasting aligns trailing axes, and a 2-D convolution's output is
/// NCHW, so the channel axis sits third from the right. A constant is
/// per-channel only if it is a scalar, or if its channel-aligned axis holds
/// `out_channels` and every other axis is 1 — `[1, OC, 1, 1]` or `[OC, 1, 1]`.
///
/// A bare `[OC]` is **not** per-channel: it aligns against width, so it scales
/// columns rather than channels, and only broadcasts at all when `OC` happens
/// to equal the output width. The string version tested element count alone and
/// so folded that case as if it were per-channel, silently computing something
/// else.
fn broadcast_to_channels(graph: &Graph, value: ValueId, out_channels: usize) -> Option<Vec<f32>> {
    const CHANNEL_AXIS_FROM_RIGHT: usize = 3;

    let tensor = graph.constant(value)?;
    let data = tensor.data();
    if data.len() == 1 {
        return Some(vec![data[0]; out_channels]);
    }

    let shape = tensor.shape();
    if shape.len() < CHANNEL_AXIS_FROM_RIGHT {
        return None;
    }
    let channel_axis = shape.len() - CHANNEL_AXIS_FROM_RIGHT;
    if shape[channel_axis] != out_channels {
        return None;
    }
    if shape
        .iter()
        .enumerate()
        .any(|(axis, &dim)| axis != channel_axis && dim != 1)
    {
        return None;
    }
    Some(data.to_vec())
}
