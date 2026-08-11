use rustc_hash::FxHashSet;

use prost::Message;
use rustc_hash::FxHashMap;
use yscv_tensor::Tensor;

use crate::attr::Attr;
use crate::error::OnnxError;
use crate::proto::onnx;

/// A named tensor extracted from an ONNX model initializer.
#[derive(Debug, Clone)]
pub struct OnnxTensor {
    pub name: String,
    pub tensor: Tensor,
}

/// An ONNX operator node with its type, inputs, outputs, and attributes.
#[derive(Debug, Clone)]
pub struct OnnxNode {
    pub op_type: String,
    pub name: String,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub attributes: FxHashMap<Attr, OnnxAttribute>,
}

/// Supported ONNX attribute value types.
#[derive(Debug, Clone)]
pub enum OnnxAttribute {
    Int(i64),
    Float(f32),
    String(String),
    Ints(Vec<i64>),
    Floats(Vec<f32>),
    Tensor(Tensor),
}

/// Parsed ONNX model containing graph topology and weight tensors.
#[derive(Debug, Clone)]
pub struct OnnxModel {
    pub ir_version: i64,
    pub opset_version: i64,
    pub producer_name: String,
    pub graph_name: String,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub initializers: FxHashMap<String, Tensor>,
    pub nodes: Vec<OnnxNode>,
    /// MatMul/Gemm weights packed to INT4 with per-group fp32 scales for
    /// the LLM decode hot path. Keyed by the original initializer name;
    /// the original `initializers` entry is removed when a weight is
    /// packed so dispatch routes through `packed_int4_gemv_dispatch`.
    pub(crate) packed_int4_weights: FxHashMap<String, crate::quantize::PackedInt4Weight>,
    /// Precomputed runtime metadata for fast per-inference environment setup.
    pub(crate) runtime_index: RuntimeModelIndex,
}

impl OnnxModel {
    /// Returns the weight tensor for a given initializer name, if present.
    pub fn get_initializer(&self, name: &str) -> Option<&Tensor> {
        self.initializers.get(name)
    }

    /// Returns the number of operator nodes in the graph.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Rebuilds runtime slot/id metadata after graph mutations.
    pub(crate) fn rebuild_runtime_index(&mut self) {
        self.runtime_index =
            build_runtime_index(&self.inputs, &self.outputs, &self.initializers, &self.nodes);
        // Execution now has no non-plan fallback: the runner walks the plan and
        // nothing else. A plan shorter than the node list would silently skip
        // the trailing nodes, so hold the one-action-per-node invariant here,
        // where it is established, rather than at the point it would be missed.
        debug_assert_eq!(
            self.runtime_index.execution_plan.len(),
            self.nodes.len(),
            "execution plan must carry one action per node"
        );
    }
}

/// Loads an ONNX model from raw protobuf bytes.
pub fn load_onnx_model(data: &[u8]) -> Result<OnnxModel, OnnxError> {
    let model_proto = onnx::ModelProto::decode(data).map_err(|e| OnnxError::DecodeFailed {
        message: e.to_string(),
    })?;

    let graph = model_proto.graph.ok_or(OnnxError::MissingGraph)?;

    let opset_version = model_proto
        .opset_import
        .first()
        .and_then(|o| o.version)
        .unwrap_or(0);

    let inputs: Vec<String> = graph
        .input
        .iter()
        .map(|v| v.name.clone().unwrap_or_default())
        .collect();

    let outputs: Vec<String> = graph
        .output
        .iter()
        .map(|v| v.name.clone().unwrap_or_default())
        .collect();

    let mut initializers = FxHashMap::default();
    for init in &graph.initializer {
        let name = init.name.clone().unwrap_or_default();
        let tensor = convert_tensor_proto(init)?;
        initializers.insert(name, tensor);
    }

    let mut nodes = Vec::new();
    for node_proto in &graph.node {
        let mut attributes = FxHashMap::default();
        for attr in &node_proto.attribute {
            let attr_name = attr.name.clone().unwrap_or_default();
            let value = convert_attribute(attr);
            if let Some(v) = value {
                attributes.insert(Attr::from_name(&attr_name), v);
            }
        }
        nodes.push(OnnxNode {
            op_type: node_proto.op_type.clone().unwrap_or_default(),
            name: node_proto.name.clone().unwrap_or_default(),
            inputs: node_proto.input.clone(),
            outputs: node_proto.output.clone(),
            attributes,
        });
    }

    let matmul_rhs_inputs: FxHashSet<String> = nodes
        .iter()
        .filter(|node| node.op_type == "MatMul")
        .filter_map(|node| node.inputs.get(1).cloned())
        .collect();
    let graph_outputs: FxHashSet<String> = outputs.iter().cloned().collect();
    let mut folded_nodes = Vec::with_capacity(nodes.len());
    for node in nodes {
        let can_fold_const_transpose = node.op_type == "Transpose"
            && node.inputs.len() == 1
            && node.outputs.len() == 1
            && matmul_rhs_inputs.contains(&node.outputs[0])
            && !graph_outputs.contains(&node.outputs[0]);
        if can_fold_const_transpose && let Some(input) = initializers.get(&node.inputs[0]) {
            let axes: Vec<usize> = match node.attributes.get(&Attr::Perm) {
                Some(OnnxAttribute::Ints(v)) if v.len() == input.rank() => {
                    v.iter().map(|&x| x as usize).collect()
                }
                _ => (0..input.rank()).rev().collect(),
            };
            if axes.iter().all(|&axis| axis < input.rank())
                && let Ok(permuted) = input.permute(&axes)
            {
                initializers.insert(node.outputs[0].clone(), permuted);
                continue;
            }
        }
        folded_nodes.push(node);
    }
    let nodes = folded_nodes;

    let runtime_index = build_runtime_index(&inputs, &outputs, &initializers, &nodes);

    Ok(OnnxModel {
        ir_version: model_proto.ir_version.unwrap_or(0),
        opset_version,
        producer_name: model_proto.producer_name.unwrap_or_default(),
        graph_name: graph.name.unwrap_or_default(),
        inputs,
        outputs,
        initializers,
        nodes,
        packed_int4_weights: FxHashMap::default(),
        runtime_index,
    })
}

/// Loads an ONNX model from a file path.
///
/// Accepts any path-like type (`&str`, `String`, `&Path`, `PathBuf`, etc.).
pub fn load_onnx_model_from_file(
    path: impl AsRef<std::path::Path>,
) -> Result<OnnxModel, OnnxError> {
    let path = path.as_ref();
    let data = std::fs::read(path).map_err(|e| OnnxError::Io {
        message: format!("{}: {e}", path.display()),
    })?;
    load_onnx_model(&data)
}

mod convert;
use convert::*;

use crate::plan::{RuntimeModelIndex, build_runtime_index};
