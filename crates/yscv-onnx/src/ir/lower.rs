//! Lowering between [`OnnxModel`] and the optimization [`Graph`].
//!
//! `OnnxModel` is the canonical source graph; the IR is the internal working
//! form for passes and executable construction. Both source optimization and
//! executable builds use `to_ir` → optional passes → `lower`, so anything the IR
//! cannot represent must be preserved rather than silently dropped.
//!
//! Known gap: `OnnxNode` has no `domain` field — the loader discards operator
//! domains when it decodes the proto, so custom-domain ops are already
//! indistinguishable from standard ones before the IR sees them. Preserving it
//! means fixing the loader and the exporter together, which is not this
//! change's job.

use rustc_hash::FxHashMap;

use super::{Graph, Node, Op, ValueKind};
use crate::loader::{OnnxModel, OnnxNode};

pub(crate) struct LoweredGraph {
    pub(crate) inputs: Vec<String>,
    pub(crate) outputs: Vec<String>,
    pub(crate) initializers: FxHashMap<String, yscv_tensor::Tensor>,
    pub(crate) nodes: Vec<OnnxNode>,
}

impl Graph {
    pub(crate) fn lower(&self) -> LoweredGraph {
        let nodes = self
            .node_ids()
            .filter_map(|id| self.to_onnx_node(id))
            .collect();
        let mut initializers = FxHashMap::default();
        for idx in 0..self.value_count() {
            let value = self.value(super::ValueId(idx as u32));
            if let ValueKind::Constant(tensor) = &value.kind {
                initializers.insert(value.name.clone(), tensor.clone());
            }
        }
        LoweredGraph {
            inputs: self
                .graph_inputs()
                .iter()
                .map(|v| self.value(*v).name.clone())
                .collect(),
            outputs: self
                .graph_outputs()
                .iter()
                .map(|v| self.value(*v).name.clone())
                .collect(),
            initializers,
            nodes,
        }
    }
}

impl OnnxModel {
    /// Builds the def-use IR for this model.
    ///
    /// Initializers become [`ValueKind::Constant`] values rather than a
    /// separate side table, so passes ask `graph.constant(v)` instead of
    /// looking a name up in a map.
    pub(crate) fn to_ir(&self) -> Graph {
        let mut graph = Graph::new();

        // Constants first, so a node referencing a weight resolves to a value
        // that already knows it is constant.
        for (name, tensor) in &self.initializers {
            let id = graph.value_by_name(name);
            graph.set_constant(id, tensor.clone());
        }

        let input_ids: Vec<_> = self
            .inputs
            .iter()
            .map(|name| graph.value_by_name(name))
            .collect();
        graph.set_graph_inputs(input_ids);

        for node in &self.nodes {
            // ONNX spells an omitted optional input as an empty name.
            let inputs = node
                .inputs
                .iter()
                .map(|name| {
                    if name.is_empty() {
                        None
                    } else {
                        Some(graph.value_by_name(name))
                    }
                })
                .collect();
            let outputs = node
                .outputs
                .iter()
                .map(|name| graph.value_by_name(name))
                .collect();
            graph.push_node(Node {
                op: Op::from_op_type(&node.op_type),
                name: node.name.clone(),
                inputs,
                outputs,
                attributes: node.attributes.clone(),
            });
        }

        let output_ids: Vec<_> = self
            .outputs
            .iter()
            .map(|name| graph.value_by_name(name))
            .collect();
        graph.set_graph_outputs(output_ids);

        debug_assert!(graph.validate().is_ok(), "to_ir produced an invalid graph");
        graph
    }
}

impl Graph {
    /// Materializes one IR node as an `OnnxNode`.
    ///
    /// The runner's dispatch is written against `OnnxNode`, so constant folding
    /// needs a node in that shape to evaluate. Building one node is cheap; it is
    /// building a whole model around it that was not.
    pub(crate) fn to_onnx_node(&self, id: super::NodeId) -> Option<OnnxNode> {
        let node = self.node(id)?;
        Some(OnnxNode {
            op_type: node.op.as_str().to_string(),
            name: node.name.clone(),
            inputs: node
                .inputs
                .iter()
                .map(|slot| match slot {
                    Some(v) => self.value(*v).name.clone(),
                    None => String::new(),
                })
                .collect(),
            outputs: node
                .outputs
                .iter()
                .map(|v| self.value(*v).name.clone())
                .collect(),
            attributes: node.attributes.clone(),
        })
    }
}

impl OnnxModel {
    /// Writes an optimized graph back over this model.
    ///
    /// Rebuilds the source graph from the IR without constructing runtime state.
    pub(crate) fn apply_ir(&mut self, graph: &Graph) {
        let lowered = graph.lower();
        self.nodes = lowered.nodes;
        self.initializers = lowered.initializers;
        self.inputs = lowered.inputs;
        self.outputs = lowered.outputs;
    }
}
