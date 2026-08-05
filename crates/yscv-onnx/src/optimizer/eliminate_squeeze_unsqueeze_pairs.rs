use crate::attr::Attr;
use crate::error::OnnxError;
use crate::ir::{Changed, Graph, Node, Op, Pass};
use crate::loader::OnnxAttribute;

/// Drops inverse `Squeeze`/`Unsqueeze` pairs left by ONNX export pipelines —
/// common after PyTorch export, where they appear as broadcasting workarounds.
///
/// Matches `Unsqueeze(Squeeze(x, A), A)` and `Squeeze(Unsqueeze(x, A), A)`,
/// where the first op's result feeds only the second.
///
/// Two things change relative to the pre-IR version. It matched only on
/// *positional* adjacency (`nodes[i]`, `nodes[i + 1]`), so a pair separated by
/// an unrelated node was missed even when the dataflow was identical; here the
/// consumer is found through the use list, so position is irrelevant. And
/// because candidates are node ids rather than indices, the overlapping-match
/// bug — a `Squeeze -> Unsqueeze -> Squeeze` chain matching at both `i` and
/// `i + 1`, then removing already-shifted indices — cannot be expressed.
pub(crate) struct EliminateSqueezeUnsqueezePairs;

impl Pass for EliminateSqueezeUnsqueezePairs {
    fn name(&self) -> &'static str {
        "eliminate_squeeze_unsqueeze_pairs"
    }

    fn run(&self, graph: &mut Graph) -> Result<Changed, OnnxError> {
        let mut changed = false;

        for first_id in graph.node_ids().collect::<Vec<_>>() {
            let Some(first) = graph.node(first_id) else {
                continue;
            };
            if !matches!(first.op, Op::Squeeze | Op::Unsqueeze) {
                continue;
            }
            let (Some(Some(source)), Some(&produced)) =
                (first.inputs.first(), first.outputs.first())
            else {
                continue;
            };
            let source = *source;

            // The pair only cancels if nothing else observes the intermediate.
            let Some(use_site) = graph.sole_consumer(produced) else {
                continue;
            };
            if use_site.port != 0 {
                continue;
            }
            let Some(second) = graph.node(use_site.node) else {
                continue;
            };
            let inverse = matches!(
                (&first.op, &second.op),
                (Op::Squeeze, Op::Unsqueeze) | (Op::Unsqueeze, Op::Squeeze)
            );
            if !inverse {
                continue;
            }
            let Some(&second_out) = second.outputs.first() else {
                continue;
            };

            // Only exactly-inverse axes cancel; `Squeeze([0])` followed by
            // `Unsqueeze([1])` is a real reshape.
            let (Some(first_axes), Some(second_axes)) =
                (axes_of(graph, first), axes_of(graph, second))
            else {
                continue;
            };
            if first_axes != second_axes {
                continue;
            }

            graph.replace_all_uses_with(second_out, source);
            graph.remove_node(use_site.node);
            graph.remove_node(first_id);
            changed = true;
        }

        Ok(changed)
    }
}

/// Reads a Squeeze/Unsqueeze node's axes, sorted.
///
/// Opset 13 moved axes from an attribute to input 1, and both spellings occur
/// in models found in the wild. Returns `None` when the axes are dynamic, which
/// makes the pair unverifiable and therefore ineligible.
fn axes_of(graph: &Graph, node: &Node) -> Option<Vec<i64>> {
    if let Some(Some(axes_value)) = node.inputs.get(1) {
        let tensor = graph.constant(*axes_value)?;
        let mut axes: Vec<i64> = tensor.data().iter().map(|&v| v as i64).collect();
        axes.sort_unstable();
        return Some(axes);
    }
    if let Some(OnnxAttribute::Ints(axes)) = node.attributes.get(&Attr::Axes) {
        let mut axes = axes.clone();
        axes.sort_unstable();
        return Some(axes);
    }
    None
}
