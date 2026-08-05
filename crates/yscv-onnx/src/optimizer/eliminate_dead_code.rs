use crate::error::OnnxError;
use crate::ir::{Changed, Graph, NodeId, Pass};

/// Removes nodes whose outputs nothing consumes.
///
/// A node is dead when none of its outputs are read by a live node and none are
/// graph outputs. Removing it can kill its producers in turn, so this is a
/// worklist: seed with every node, and when one is removed, re-examine the
/// nodes that defined its inputs.
///
/// The pre-IR version reached the same fixpoint by rebuilding a
/// `FxHashSet<String>` of every consumed tensor name and re-running `retain`
/// until the node count stopped falling — O(sweeps · N) with a string hash per
/// edge. Here the use counts are already maintained, so the whole thing is
/// O(N + E) with no hashing.
pub(crate) struct EliminateDeadCode;

impl Pass for EliminateDeadCode {
    fn name(&self) -> &'static str {
        "eliminate_dead_code"
    }

    fn run(&self, graph: &mut Graph) -> Result<Changed, OnnxError> {
        let mut worklist: Vec<NodeId> = graph.node_ids().collect();
        let mut changed = false;

        while let Some(id) = worklist.pop() {
            let Some(node) = graph.node(id) else {
                // Already removed via another path through the worklist.
                continue;
            };
            let live = node
                .outputs
                .iter()
                .any(|&out| graph.use_count(out) > 0 || graph.is_graph_output(out));
            if live {
                continue;
            }

            // Producers of this node's inputs may become dead once it stops
            // reading them, so queue them before the edges disappear.
            let producers: Vec<NodeId> = node
                .inputs
                .iter()
                .flatten()
                .filter_map(|&v| graph.value(v).def)
                .collect();

            graph.remove_node(id);
            changed = true;
            worklist.extend(producers);
        }

        Ok(changed)
    }
}
