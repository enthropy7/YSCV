//! Lowering between [`OnnxModel`] and the optimization [`Graph`].
//!
//! `OnnxModel` stays the crate's public model type and the runner's input;
//! the IR is an internal working form that passes operate on. So the flow is
//! `to_ir` → run passes → `apply_ir`, and the two directions have to agree
//! exactly: anything the IR cannot represent would be silently dropped from
//! every optimized model.
//!
//! Known gap: `OnnxNode` has no `domain` field — the loader discards operator
//! domains when it decodes the proto, so custom-domain ops are already
//! indistinguishable from standard ones before the IR sees them. Preserving it
//! means fixing the loader and the exporter together, which is not this
//! change's job.

use rustc_hash::{FxHashMap, FxHashSet};

use super::{Graph, Node, Op, ValueKind, WeightLayout};
use crate::loader::{OnnxModel, OnnxNode};

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

            // The loader pre-permutes conv weights and records which
            // permutation in three side sets; passes that rewrite a weight need
            // to know, because the output channel sits on a different axis in
            // each. Carried on the value so they ask once instead of consulting
            // three sets. See `ir::weight_layout`.
            let layout = if self.khwc_weights.contains(name) {
                WeightLayout::Khwc
            } else if self.dw_khwc_weights.contains(name) {
                WeightLayout::DepthwiseKhwc
            } else if self.group_khwc_weights.contains(name) {
                WeightLayout::GroupKhwc
            } else {
                WeightLayout::Oihw
            };
            if layout != WeightLayout::Oihw {
                graph.set_weight_layout(id, layout);
            }
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
    /// Rebuilds `nodes`, `initializers` and the loader's weight-layout side
    /// tables from the IR. The runtime index is left alone — the driver rebuilds
    /// it once after the whole pipeline.
    ///
    /// The side tables cannot be carried over by name. A pass that rewrites a
    /// weight is free to give the result a new one, and the runtime reads the
    /// tables by the name the Conv actually points at, so a renamed weight
    /// silently reverts to being read as ONNX-native OIHW — a pre-permuted
    /// depthwise `[KH, KW, C, 1]` then parses as `[O, I, KH, KW]`, and the
    /// output-channel count comes back as the kernel height. `weight_layouts`
    /// is keyed by value, survives the rename, and is the authority here.
    pub(crate) fn apply_ir(&mut self, graph: &Graph) {
        let nodes: Vec<OnnxNode> = graph
            .node_ids()
            .filter_map(|id| graph.to_onnx_node(id))
            .collect();

        let mut initializers = FxHashMap::default();
        let mut khwc_weights = FxHashSet::default();
        let mut dw_khwc_weights = FxHashSet::default();
        let mut group_khwc_weights = FxHashSet::default();
        for idx in 0..graph.value_count() {
            let id = super::ValueId(idx as u32);
            let value = graph.value(id);
            if let ValueKind::Constant(tensor) = &value.kind {
                initializers.insert(value.name.clone(), tensor.clone());
                match graph.weight_layout(id) {
                    WeightLayout::Khwc => khwc_weights.insert(value.name.clone()),
                    WeightLayout::DepthwiseKhwc => dw_khwc_weights.insert(value.name.clone()),
                    WeightLayout::GroupKhwc => group_khwc_weights.insert(value.name.clone()),
                    WeightLayout::Oihw => false,
                };
            }
        }

        self.nodes = nodes;
        self.initializers = initializers;
        self.khwc_weights = khwc_weights;
        self.dw_khwc_weights = dw_khwc_weights;
        self.group_khwc_weights = group_khwc_weights;
        self.inputs = graph
            .graph_inputs()
            .iter()
            .map(|v| graph.value(*v).name.clone())
            .collect();
        self.outputs = graph
            .graph_outputs()
            .iter()
            .map(|v| graph.value(*v).name.clone())
            .collect();
    }
}
