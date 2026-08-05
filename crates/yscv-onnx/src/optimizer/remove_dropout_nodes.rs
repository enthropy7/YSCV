use crate::error::OnnxError;
use crate::ir::{Changed, Graph, Op, Pass};

/// Removes `Dropout` nodes, rewiring consumers to the Dropout's data input.
/// Dropout is the identity at inference time.
///
/// The pre-IR version rewired by string-replacing tensor names across every
/// node, then deleted by node *name* — which ONNX makes optional, so a single
/// unnamed Dropout took every other unnamed node in the graph with it, and a
/// malformed Dropout that was skipped during rewiring got deleted anyway.
/// Neither failure is expressible here: rewiring is `replace_all_uses_with` on
/// a value id, and removal targets exactly the node that was rewired.
pub(crate) struct RemoveDropout;

impl Pass for RemoveDropout {
    fn name(&self) -> &'static str {
        "remove_dropout_nodes"
    }

    fn run(&self, graph: &mut Graph) -> Result<Changed, OnnxError> {
        let mut changed = false;

        for id in graph.node_ids().collect::<Vec<_>>() {
            let Some(node) = graph.node(id) else { continue };
            if node.op != Op::Dropout {
                continue;
            }
            // A malformed Dropout — no data input or no output — is left in
            // place, since there is nothing to rewire its consumers onto.
            let (Some(Some(data)), Some(&output)) = (node.inputs.first(), node.outputs.first())
            else {
                continue;
            };
            let data = *data;

            // Dropout's optional second output is the training mask. Nothing
            // reads it at inference, but if something does, removing the node
            // would leave that consumer dangling.
            let mask_is_live = node.outputs[1..]
                .iter()
                .any(|&extra| graph.use_count(extra) > 0 || graph.is_graph_output(extra));
            if mask_is_live {
                continue;
            }

            graph.replace_all_uses_with(output, data);
            graph.remove_node(id);
            changed = true;
        }

        Ok(changed)
    }
}
