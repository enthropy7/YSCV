use rustc_hash::FxHashMap;
use yscv_tensor::Tensor;

use crate::error::OnnxError;
use crate::ir::{Changed, Graph, Node, NodeId, Op, Pass, ValueId, WeightLayout};
use crate::loader::OnnxAttribute;

/// Rewrites `ConvTranspose` with `kernel == stride` into `Conv 1x1` followed by
/// `DepthToSpace(CRD)`.
///
/// When the kernel exactly tiles the stride, the transposed convolution is a
/// per-sub-position 1×1 convolution whose `k²` output groups are then
/// interleaved into the spatial grid — which is what `DepthToSpace` does. The
/// rewrite puts the work on the GEMM-backed Conv path and unlocks backends with
/// no `ConvTranspose` kernel at all.
///
/// Applies only to the safe subset: square kernel, `k == stride`, `group == 1`,
/// zero pads, zero output padding, no dilation, and a constant weight. Anything
/// else is left alone.
pub(crate) struct RewriteConvTransposeToDepthToSpace;

impl Pass for RewriteConvTransposeToDepthToSpace {
    fn name(&self) -> &'static str {
        "rewrite_convtranspose_dts"
    }

    fn run(&self, graph: &mut Graph) -> Result<Changed, OnnxError> {
        let mut changed = false;

        for node_id in graph.node_ids().collect::<Vec<_>>() {
            let Some(rewrite) = match_rewrite(graph, node_id) else {
                continue;
            };
            apply(graph, node_id, rewrite);
            changed = true;
        }

        Ok(changed)
    }
}

/// The resolved rewrite: new weights, and the pieces needed to name things.
struct Rewrite {
    data: ValueId,
    weight: Tensor,
    bias: Option<Tensor>,
    /// Kernel size, which is also the DepthToSpace block size.
    block: usize,
}

fn match_rewrite(graph: &Graph, node_id: NodeId) -> Option<Rewrite> {
    let node = graph.node(node_id)?;
    if node.op != Op::ConvTranspose || node.inputs.len() < 2 {
        return None;
    }

    let ints = |name: &str| match node.attributes.get(name) {
        Some(OnnxAttribute::Ints(v)) => Some(v.clone()),
        _ => None,
    };
    let strides = ints("strides").unwrap_or_else(|| vec![1, 1]);
    let pads = ints("pads").unwrap_or_else(|| vec![0, 0, 0, 0]);
    let output_padding = ints("output_padding").unwrap_or_else(|| vec![0, 0]);
    let dilations = ints("dilations").unwrap_or_else(|| vec![1, 1]);
    let group = match node.attributes.get("group") {
        Some(OnnxAttribute::Int(g)) => *g,
        _ => 1,
    };

    let data = (*node.inputs.first()?)?;
    let weight_value = (*node.inputs.get(1)?)?;
    let weight = graph.constant(weight_value)?;
    // The rewrite reinterprets the weight's axes, so a pre-permuted one would
    // be read against the wrong layout.
    if graph.weight_layout(weight_value) != WeightLayout::Oihw {
        return None;
    }

    let shape = weight.shape();
    if shape.len() != 4 {
        return None;
    }
    let (c_in, c_out, kh, kw) = (shape[0], shape[1], shape[2], shape[3]);
    let k = kh;
    let safe = kh == kw
        && k > 0
        && strides.len() == 2
        && strides.iter().all(|&s| s == k as i64)
        && group == 1
        && pads.iter().all(|&p| p == 0)
        && output_padding.iter().all(|&p| p == 0)
        && dilations.iter().all(|&d| d == 1);
    if !safe {
        return None;
    }

    // `W[C_in, C_out, k, k]` becomes a 1×1 Conv weight `[k²·C_out, C_in, 1, 1]`,
    // where output channel `(co·k + di)·k + dj` carries the contribution of
    // sub-position `(di, dj)` — the CRD ordering DepthToSpace then unpacks.
    let w = weight.data();
    let mut packed = vec![0.0f32; c_out * k * k * c_in];
    for ci in 0..c_in {
        for co in 0..c_out {
            for di in 0..k {
                for dj in 0..k {
                    let src = ((ci * c_out + co) * k + di) * k + dj;
                    let dst_channel = (co * k + di) * k + dj;
                    packed[dst_channel * c_in + ci] = w[src];
                }
            }
        }
    }
    let weight = Tensor::from_vec(vec![c_out * k * k, c_in, 1, 1], packed).ok()?;

    // Each original bias entry is one output channel; after the rewrite it
    // applies to all k² sub-positions of that channel.
    let bias = match node.inputs.get(2).copied().flatten() {
        Some(bias_value) => {
            let b = graph.constant(bias_value)?.data();
            if b.len() != c_out {
                return None;
            }
            let mut expanded = vec![0.0f32; c_out * k * k];
            for co in 0..c_out {
                for sub in 0..k * k {
                    expanded[co * k * k + sub] = b[co];
                }
            }
            Some(Tensor::from_vec(vec![c_out * k * k], expanded).ok()?)
        }
        None => None,
    };

    Some(Rewrite {
        data,
        weight,
        bias,
        block: k,
    })
}

fn apply(graph: &mut Graph, node_id: NodeId, rewrite: Rewrite) {
    let base = graph.fresh_value_name(node_id, "dts");
    let node_name = graph
        .node(node_id)
        .map(|n| n.name.clone())
        .unwrap_or_default();

    let weight = graph.value_by_name(&format!("{base}_w"));
    graph.set_constant(weight, rewrite.weight);

    let mut conv_inputs = vec![Some(rewrite.data), Some(weight)];
    if let Some(bias) = rewrite.bias {
        let bias_value = graph.value_by_name(&format!("{base}_b"));
        graph.set_constant(bias_value, bias);
        conv_inputs.push(Some(bias_value));
    }

    let mid = graph.value_by_name(&format!("{base}_conv"));
    let conv = Node {
        op: Op::Conv,
        name: format!("{node_name}_c1x1"),
        inputs: conv_inputs,
        outputs: vec![mid],
        attributes: attrs([
            ("strides", OnnxAttribute::Ints(vec![1, 1])),
            ("pads", OnnxAttribute::Ints(vec![0, 0, 0, 0])),
            ("kernel_shape", OnnxAttribute::Ints(vec![1, 1])),
        ]),
    };
    let depth_to_space = Node {
        op: Op::DepthToSpace,
        name: format!("{node_name}_dts"),
        inputs: vec![Some(mid)],
        // `replace_node` fills these in from the node being replaced.
        outputs: Vec::new(),
        attributes: attrs([
            ("blocksize", OnnxAttribute::Int(rewrite.block as i64)),
            ("mode", OnnxAttribute::String("CRD".to_owned())),
        ]),
    };

    graph.replace_node(node_id, vec![conv, depth_to_space]);
}

fn attrs<const N: usize>(pairs: [(&str, OnnxAttribute); N]) -> FxHashMap<String, OnnxAttribute> {
    pairs
        .into_iter()
        .map(|(name, value)| (name.to_owned(), value))
        .collect()
}
