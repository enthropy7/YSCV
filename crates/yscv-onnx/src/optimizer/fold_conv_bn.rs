use crate::attr::Attr;
use yscv_tensor::Tensor;

use crate::error::OnnxError;
use crate::ir::{Changed, Graph, NodeId, Op, Pass, ValueId};
use crate::loader::OnnxAttribute;

/// Folds `BatchNormalization` into the weights of the `Conv` that feeds it.
///
/// At inference the batch statistics are frozen, so the normalization is an
/// affine per-channel transform and collapses into the convolution:
///
/// ```text
/// scale[c] = gamma[c] / sqrt(var[c] + epsilon)
/// W'[c]    = W[c] * scale[c]
/// b'[c]    = (b[c] - mean[c]) * scale[c] + beta[c]
/// ```
///
/// Reads the weight as ONNX-native `[O, I/G, KH, KW]`: output channel on axis
/// 0, so each owns one contiguous block. The three layout branches this used to
/// carry went away with the loader-side permute (see `plan::prepack`).
pub(crate) struct FoldConvBatchNorm;

impl Pass for FoldConvBatchNorm {
    fn name(&self) -> &'static str {
        "fold_conv_bn"
    }

    fn run(&self, graph: &mut Graph) -> Result<Changed, OnnxError> {
        let mut changed = false;

        for conv_id in graph.node_ids().collect::<Vec<_>>() {
            let Some(fold) = match_fold(graph, conv_id) else {
                continue;
            };
            apply(graph, conv_id, &fold);
            graph.absorb_consumer(conv_id, fold.bn_id);
            changed = true;
        }

        Ok(changed)
    }
}

/// Everything the rewrite needs, resolved before any mutation.
struct Fold {
    bn_id: NodeId,
    weight: ValueId,
    /// Existing bias operand, when the Conv already had one.
    bias: Option<ValueId>,
    fused_weight: Tensor,
    fused_bias: Tensor,
}

fn match_fold(graph: &Graph, conv_id: NodeId) -> Option<Fold> {
    let conv = graph.node(conv_id)?;
    if conv.op != Op::Conv {
        return None;
    }

    // The BatchNormalization must be the only thing reading the Conv's output,
    // or the unnormalized values are still observable somewhere.
    let use_site = graph.sole_consumer(*conv.outputs.first()?)?;
    if use_site.port != 0 {
        return None;
    }
    let bn = graph.node(use_site.node)?;
    if bn.op != Op::BatchNormalization || bn.inputs.len() < 5 {
        return None;
    }

    let weight = (*conv.inputs.get(1)?)?;
    let w = graph.constant(weight)?;

    // Folding rewrites the weight in place, so a weight shared with another
    // Conv — the two branches of a Siamese tracker, say — would have the second
    // Conv's scale applied on top of the first's. The string version rewrote
    // `initializers[name]` without checking, and silently corrupted such
    // models.
    if graph.use_count(weight) != 1 {
        return None;
    }

    let gamma = graph.constant((*bn.inputs.get(1)?)?)?.data();
    let beta = graph.constant((*bn.inputs.get(2)?)?)?.data();
    let mean = graph.constant((*bn.inputs.get(3)?)?)?.data();
    let var = graph.constant((*bn.inputs.get(4)?)?)?.data();

    let epsilon = match bn.attributes.get(&Attr::Epsilon) {
        Some(OnnxAttribute::Float(v)) => *v,
        _ => 1e-5,
    };

    let shape = w.shape().to_vec();
    let out_channels = *shape.first()?;
    if out_channels == 0 || shape.len() < 4 {
        return None;
    }
    if gamma.len() < out_channels
        || beta.len() < out_channels
        || mean.len() < out_channels
        || var.len() < out_channels
    {
        return None;
    }

    let scale: Vec<f32> = (0..out_channels)
        .map(|c| gamma[c] / (var[c] + epsilon).sqrt())
        .collect();

    let mut w_data = w.data().to_vec();
    let per_channel = w_data.len() / out_channels;
    for (i, v) in w_data.iter_mut().enumerate() {
        *v *= scale[i / per_channel];
    }
    let fused_weight = Tensor::from_vec(shape, w_data).ok()?;

    let bias = conv.inputs.get(2).copied().flatten();
    let old_bias = match bias {
        Some(b) => graph.constant(b).map(|t| t.data().to_vec()),
        None => Some(vec![0.0; out_channels]),
    }?;
    if old_bias.len() < out_channels {
        return None;
    }
    // A shared bias has the same aliasing problem as a shared weight.
    if let Some(b) = bias
        && graph.use_count(b) != 1
    {
        return None;
    }

    let fused_bias_data: Vec<f32> = (0..out_channels)
        .map(|c| (old_bias[c] - mean[c]) * scale[c] + beta[c])
        .collect();
    let fused_bias = Tensor::from_vec(vec![out_channels], fused_bias_data).ok()?;

    Some(Fold {
        bn_id: use_site.node,
        weight,
        bias,
        fused_weight,
        fused_bias,
    })
}

fn apply(graph: &mut Graph, conv_id: NodeId, fold: &Fold) {
    graph.set_constant(fold.weight, fold.fused_weight.clone());
    match fold.bias {
        Some(bias) => graph.set_constant(bias, fold.fused_bias.clone()),
        None => {
            let name = graph.fresh_value_name(conv_id, "fused_bias");
            let bias = graph.value_by_name(&name);
            graph.set_constant(bias, fold.fused_bias.clone());
            graph.push_input(conv_id, bias);
        }
    }
}
