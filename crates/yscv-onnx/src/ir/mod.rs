//! Optimization IR: a def-use graph for load-time transformations.
//!
//! # Why this exists
//!
//! Passes used to mutate [`OnnxModel`](crate::loader::OnnxModel) directly,
//! whose edges are `String` tensor names with no index over them. Answering
//! "who consumes this value?" meant rescanning every node's input list, so six
//! passes carried a verbatim copy of the same O(N²) loop, and matching was
//! restricted to positional adjacency (`nodes[i]` / `nodes[i + 1]`) — which is
//! the sole reason `reorder_nodes_for_fusion` has to run first.
//!
//! Here, edges are [`ValueId`]s and every value carries its defining node and
//! its complete list of uses. Producer/consumer queries are O(1), so pattern
//! matching no longer depends on node ordering.
//!
//! # Invariants
//!
//! These hold at every public API boundary, and [`Graph::validate`] checks them
//! (wired into debug assertions on every mutation):
//!
//! 1. If `values[v].def == Some(n)` then `nodes[n]` is live and lists `v` in its
//!    outputs.
//! 2. `values[v].uses` contains exactly one entry per (live node, input port)
//!    referencing `v`, with no duplicates.
//! 3. Tombstoned nodes (`nodes[n] == None`) appear in no `def` and no `uses`.
//!
//! # Identifier stability
//!
//! [`NodeId`] and [`ValueId`] index into vectors that are only ever appended
//! to; removal tombstones the slot. Ids therefore stay valid across mutations,
//! so a pass can collect candidates in one sweep and rewrite them in another
//! without the index-invalidation dance that `Vec::remove` forces.

mod lower;
mod op;
mod pass;

use rustc_hash::FxHashMap;
use yscv_tensor::Tensor;

use crate::loader::OnnxAttribute;

pub(crate) use op::Op;
pub(crate) use pass::{Changed, Pass, PassManager};

/// Index of a node in [`Graph::nodes`]. Stable across mutations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct NodeId(pub(crate) u32);

/// Index of a value in [`Graph::values`]. Stable across mutations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct ValueId(pub(crate) u32);

impl NodeId {
    fn idx(self) -> usize {
        self.0 as usize
    }
}

impl ValueId {
    fn idx(self) -> usize {
        self.0 as usize
    }
}

/// A single consumer of a value: which node reads it, at which input port.
///
/// The port matters — `Sub(x, x)` uses the same value twice and the two uses
/// are not interchangeable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Use {
    pub(crate) node: NodeId,
    pub(crate) port: u32,
}

/// What a value is, from the graph's point of view.
#[derive(Debug, Clone)]
pub(crate) enum ValueKind {
    /// A runtime input fed at inference time.
    GraphInput,
    /// A weight known at load time.
    Constant(Tensor),
    /// Produced by a node in this graph.
    Intermediate,
}

/// An edge in the graph: a named tensor with its producer and consumers.
#[derive(Debug, Clone)]
pub(crate) struct Value {
    /// Original ONNX tensor name. Preserved so lowering round-trips and so
    /// error messages stay recognizable against the source model.
    pub(crate) name: String,
    /// Producing node, or `None` for graph inputs and constants.
    pub(crate) def: Option<NodeId>,
    /// Every (node, port) that reads this value. Maintained by the mutation
    /// API; see the module invariants.
    pub(crate) uses: Vec<Use>,
    pub(crate) kind: ValueKind,
}

/// An operator instance.
#[derive(Debug, Clone)]
pub(crate) struct Node {
    pub(crate) op: Op,
    /// ONNX node name. May be empty — the format makes it optional.
    pub(crate) name: String,
    /// Inputs by port. `None` is an omitted optional input, which ONNX encodes
    /// as an empty name — a Conv with no bias, say.
    pub(crate) inputs: Vec<Option<ValueId>>,
    pub(crate) outputs: Vec<ValueId>,
    pub(crate) attributes: FxHashMap<String, OnnxAttribute>,
}

/// A model graph in def-use form.
pub(crate) struct Graph {
    /// Node storage. `None` marks a tombstoned slot, keeping [`NodeId`] stable.
    nodes: Vec<Option<Node>>,
    /// Value storage. Values are never removed, only orphaned; lowering drops
    /// the unreachable ones.
    values: Vec<Value>,
    /// Execution order as a list of node ids. May contain tombstoned ids, which
    /// iteration skips and [`Graph::compact`] drops.
    order: Vec<NodeId>,
    /// Graph inputs, in declaration order.
    inputs: Vec<ValueId>,
    /// Graph outputs, in declaration order.
    outputs: Vec<ValueId>,
    /// Name lookup for values. Kept in sync so lowering and tests can address
    /// values the way the source model does.
    by_name: FxHashMap<String, ValueId>,
}

impl Graph {
    /// An empty graph.
    ///
    /// Model-level metadata (ir_version, opset, producer, graph name) stays on
    /// `OnnxModel` — no pass reads it, and duplicating it here would be one
    /// more thing for lowering to keep in sync.
    pub(crate) fn new() -> Self {
        Graph {
            nodes: Vec::new(),
            values: Vec::new(),
            order: Vec::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            by_name: FxHashMap::default(),
        }
    }

    // ── Accessors ────────────────────────────────────────────────────────

    pub(crate) fn node(&self, id: NodeId) -> Option<&Node> {
        self.nodes.get(id.idx())?.as_ref()
    }

    pub(crate) fn value(&self, id: ValueId) -> &Value {
        &self.values[id.idx()]
    }

    /// Total number of values, live or orphaned. Lowering walks this range to
    /// collect constants.
    pub(crate) fn value_count(&self) -> usize {
        self.values.len()
    }

    pub(crate) fn graph_inputs(&self) -> &[ValueId] {
        &self.inputs
    }

    pub(crate) fn graph_outputs(&self) -> &[ValueId] {
        &self.outputs
    }

    /// Live node ids in execution order.
    pub(crate) fn node_ids(&self) -> impl Iterator<Item = NodeId> + '_ {
        self.order
            .iter()
            .copied()
            .filter(|id| self.nodes[id.idx()].is_some())
    }

    /// Number of live nodes.
    pub(crate) fn node_count(&self) -> usize {
        self.node_ids().count()
    }

    /// The constant behind a value, if it is one.
    ///
    /// Replaces the `model.initializers.get(name)` lookups passes used to do,
    /// and answers the question directly rather than by name.
    pub(crate) fn constant(&self, id: ValueId) -> Option<&Tensor> {
        match &self.values[id.idx()].kind {
            ValueKind::Constant(t) => Some(t),
            _ => None,
        }
    }

    /// True when the value is a graph output, and so cannot be eliminated even
    /// if nothing inside the graph reads it.
    pub(crate) fn is_graph_output(&self, id: ValueId) -> bool {
        self.outputs.contains(&id)
    }

    /// The single consumer of a value, when it has exactly one and the value is
    /// not also a graph output.
    ///
    /// This is the query the fold and fuse passes actually want: "is this
    /// producer's output consumed only by the node I want to merge it into?"
    /// Each of them previously answered it with its own O(N) rescan.
    pub(crate) fn sole_consumer(&self, id: ValueId) -> Option<Use> {
        if self.is_graph_output(id) {
            return None;
        }
        match self.values[id.idx()].uses.as_slice() {
            [only] => Some(*only),
            _ => None,
        }
    }

    /// Number of live consumers of a value.
    pub(crate) fn use_count(&self, id: ValueId) -> usize {
        self.values[id.idx()].uses.len()
    }

    // ── Construction ─────────────────────────────────────────────────────

    /// Resolves a value by name without interning it.
    ///
    /// Passes address values by id, reached through the node they are wired to,
    /// so this exists for tests that need to name a specific edge in a fixture.
    #[cfg(test)]
    pub(crate) fn lookup_value(&self, name: &str) -> Option<ValueId> {
        self.by_name.get(name).copied()
    }

    /// Interns a value by name, creating it if new.
    ///
    /// ONNX identifies edges by name, so building a graph means resolving the
    /// same name from both producer and consumer side, in either order.
    pub(crate) fn value_by_name(&mut self, name: &str) -> ValueId {
        if let Some(&id) = self.by_name.get(name) {
            return id;
        }
        let id = ValueId(self.values.len() as u32);
        self.values.push(Value {
            name: name.to_string(),
            def: None,
            uses: Vec::new(),
            kind: ValueKind::Intermediate,
        });
        self.by_name.insert(name.to_string(), id);
        id
    }

    /// Marks a value as a load-time constant.
    pub(crate) fn set_constant(&mut self, id: ValueId, tensor: Tensor) {
        self.values[id.idx()].kind = ValueKind::Constant(tensor);
    }

    pub(crate) fn set_graph_inputs(&mut self, ids: Vec<ValueId>) {
        for &id in &ids {
            if matches!(self.values[id.idx()].kind, ValueKind::Intermediate) {
                self.values[id.idx()].kind = ValueKind::GraphInput;
            }
        }
        self.inputs = ids;
    }

    pub(crate) fn set_graph_outputs(&mut self, ids: Vec<ValueId>) {
        self.outputs = ids;
    }

    /// Appends a node to the end of the execution order, wiring its def-use
    /// edges.
    pub(crate) fn push_node(&mut self, node: Node) -> NodeId {
        let id = NodeId(self.nodes.len() as u32);
        self.link(id, &node);
        self.nodes.push(Some(node));
        self.order.push(id);
        id
    }

    /// Records `node`'s uses and defs. Called with the node not yet stored, so
    /// it takes the node by reference rather than reading it back by id.
    fn link(&mut self, id: NodeId, node: &Node) {
        for (port, input) in node.inputs.iter().enumerate() {
            if let Some(v) = input {
                self.values[v.idx()].uses.push(Use {
                    node: id,
                    port: port as u32,
                });
            }
        }
        for &out in &node.outputs {
            self.values[out.idx()].def = Some(id);
        }
    }

    // ── Mutation ─────────────────────────────────────────────────────────

    /// Redirects every consumer of `old` to `new`, and transfers graph-output
    /// status.
    ///
    /// This is the primitive behind node elimination: rewire the consumers,
    /// then drop the now-unused producer. Passes previously did it by
    /// string-replacing input names across the whole node list.
    pub(crate) fn replace_all_uses_with(&mut self, old: ValueId, new: ValueId) {
        if old == new {
            return;
        }
        let uses = std::mem::take(&mut self.values[old.idx()].uses);
        for use_site in uses {
            if let Some(slot) = self.nodes[use_site.node.idx()]
                .as_mut()
                .and_then(|n| n.inputs.get_mut(use_site.port as usize))
            {
                *slot = Some(new);
            }
            self.values[new.idx()].uses.push(use_site);
        }
        for out in &mut self.outputs {
            if *out == old {
                *out = new;
            }
        }
        debug_assert!(
            self.validate().is_ok(),
            "replace_all_uses_with broke an invariant"
        );
    }

    /// Tombstones a node, unlinking it from every value it touched.
    ///
    /// The node's outputs keep their names but lose their producer. Callers are
    /// expected to have rewired consumers first — removing a node whose outputs
    /// are still read leaves the graph unschedulable, and debug builds trip on
    /// it.
    pub(crate) fn remove_node(&mut self, id: NodeId) {
        let Some(node) = self.nodes[id.idx()].take() else {
            return;
        };
        for (port, input) in node.inputs.iter().enumerate() {
            if let Some(v) = input {
                self.remove_use(*v, id, port as u32);
            }
        }
        for &out in &node.outputs {
            debug_assert!(
                self.values[out.idx()].uses.is_empty() || self.is_graph_output(out),
                "removed a node whose output {} is still consumed",
                self.values[out.idx()].name
            );
            if self.values[out.idx()].def == Some(id) {
                self.values[out.idx()].def = None;
            }
        }
        debug_assert!(self.validate().is_ok(), "remove_node broke an invariant");
    }

    fn remove_use(&mut self, value: ValueId, node: NodeId, port: u32) {
        let uses = &mut self.values[value.idx()].uses;
        if let Some(pos) = uses.iter().position(|u| u.node == node && u.port == port) {
            uses.swap_remove(pos);
        } else {
            debug_assert!(false, "use list missing an entry it should have had");
        }
    }

    /// Drops tombstoned ids from the execution order.
    ///
    /// Removal is O(1) and leaves holes; this pays the compaction cost once
    /// rather than per removed node.
    pub(crate) fn compact(&mut self) {
        let nodes = &self.nodes;
        self.order.retain(|id| nodes[id.idx()].is_some());
    }

    // ── Validation ───────────────────────────────────────────────────────

    /// Recomputes the def-use index by brute force and compares it against the
    /// maintained one.
    ///
    /// Every mutation calls this behind `debug_assert!`, so a pass that corrupts
    /// the index fails in the test suite at the point of corruption instead of
    /// producing a wrong model later.
    pub(crate) fn validate(&self) -> Result<(), IrError> {
        let mut expected: Vec<Vec<Use>> = vec![Vec::new(); self.values.len()];
        for (idx, slot) in self.nodes.iter().enumerate() {
            let Some(node) = slot else { continue };
            let id = NodeId(idx as u32);
            for (port, input) in node.inputs.iter().enumerate() {
                if let Some(v) = input {
                    expected[v.idx()].push(Use {
                        node: id,
                        port: port as u32,
                    });
                }
            }
            for &out in &node.outputs {
                if self.values[out.idx()].def != Some(id) {
                    return Err(IrError::BadDef {
                        value: self.values[out.idx()].name.clone(),
                    });
                }
            }
        }

        for (idx, value) in self.values.iter().enumerate() {
            let mut actual = value.uses.clone();
            let mut want = std::mem::take(&mut expected[idx]);
            let key = |u: &Use| (u.node.0, u.port);
            actual.sort_by_key(key);
            want.sort_by_key(key);
            if actual != want {
                return Err(IrError::BadUses {
                    value: value.name.clone(),
                });
            }
            if let Some(def) = value.def
                && self.nodes[def.idx()].is_none()
            {
                return Err(IrError::BadDef {
                    value: value.name.clone(),
                });
            }
        }
        Ok(())
    }
}

/// Structural faults in the IR. These are bugs in a pass, not malformed input.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub(crate) enum IrError {
    #[error("value '{value}' has a stale or missing definition")]
    BadDef { value: String },
    #[error("value '{value}' has a use list that disagrees with the graph")]
    BadUses { value: String },
}

#[cfg(test)]
mod tests;
