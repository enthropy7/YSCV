use rustc_hash::FxHashMap;

use crate::error::OnnxError;
use crate::ir::{Changed, Graph, NodeId, Pass};

/// Reorders nodes into a fusion-friendly topological order.
///
/// ONNX topological order is not unique. Multi-input models — a Siamese tracker
/// whose two backbone branches share weights, say — are often exported with the
/// branches interleaved: `convA, convB, reluA, reluB, …`.
///
/// Neither the IR passes nor the plan builder need the repair any more: both
/// match through the use list, so a producer and consumer separated by
/// unrelated nodes still pair up. Fusion is no longer why this pass exists.
///
/// What it is for now is memory. The schedule *determines* peak working set:
/// the same nodes in a different topological order leave a different number of
/// live ranges overlapping, because the order is what fixes how many
/// intermediates coexist. That makes this register-pressure scheduling, with a
/// measurable objective rather than a matcher to appease.
///
/// The name is now wrong and should follow the purpose — `ScheduleForMemory` or
/// similar — once the buffer arena that consumes the property lands and there
/// is something to measure the rename against.
///
/// A depth-first topological sort — a LIFO ready stack — walks each dependency
/// chain to its join before starting the next, placing every producer directly
/// ahead of its consumer. Only independent nodes are permuted, so the
/// computation and its outputs are unchanged.
pub(crate) struct ReorderForFusion;

impl Pass for ReorderForFusion {
    fn name(&self) -> &'static str {
        "reorder_nodes_for_fusion"
    }

    fn run(&self, graph: &mut Graph) -> Result<Changed, OnnxError> {
        if std::env::var_os("YSCV_REORDER_FUSION_OFF").is_some() {
            return Ok(false);
        }
        let live: Vec<NodeId> = graph.node_ids().collect();
        if live.len() < 2 {
            return Ok(false);
        }

        // Dense positions, because node ids are sparse once anything has been
        // removed.
        let slot: FxHashMap<NodeId, usize> =
            live.iter().enumerate().map(|(i, &id)| (id, i)).collect();

        let mut in_degree = vec![0usize; live.len()];
        let mut consumers: Vec<Vec<usize>> = vec![Vec::new(); live.len()];
        for (i, &id) in live.iter().enumerate() {
            let Some(node) = graph.node(id) else { continue };
            // Distinct producers only: a node reading the same value twice, or
            // two values from one producer, still depends on it once.
            let mut deps: Vec<usize> = Vec::new();
            for value in node.inputs.iter().flatten() {
                let Some(def) = graph.value(*value).def else {
                    continue;
                };
                let Some(&p) = slot.get(&def) else { continue };
                if p != i && !deps.contains(&p) {
                    deps.push(p);
                }
            }
            in_degree[i] = deps.len();
            for p in deps {
                consumers[p].push(i);
            }
        }

        // Seed in reverse so the stack pops roots in their original relative
        // order, keeping the result stable for graphs that are already sorted.
        let mut stack: Vec<usize> = (0..live.len())
            .rev()
            .filter(|&i| in_degree[i] == 0)
            .collect();

        let mut order: Vec<NodeId> = Vec::with_capacity(live.len());
        while let Some(i) = stack.pop() {
            order.push(live[i]);
            let mut ready: Vec<usize> = Vec::new();
            for &c in &consumers[i] {
                in_degree[c] -= 1;
                if in_degree[c] == 0 {
                    ready.push(c);
                }
            }
            for &c in ready.iter().rev() {
                stack.push(c);
            }
        }

        // A short result means a dependency cycle, which is not something this
        // pass can repair — leave the graph alone rather than dropping nodes.
        if order.len() != live.len() || order == live {
            return Ok(false);
        }

        graph
            .set_order(order)
            .map_err(|e| OnnxError::DecodeFailed {
                message: format!("reorder_nodes_for_fusion produced an invalid order: {e}"),
            })?;
        Ok(true)
    }
}
