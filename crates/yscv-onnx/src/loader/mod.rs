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
    ///
    /// One action per node is *not* an invariant of the finished plan, and this
    /// used to assert it was: the `FusedPwDwPwReduce` merge deliberately drops
    /// the actions it absorbs, so a merged plan is shorter than the node list
    /// and `nchwc_handoff` is documented as indexed by plan position for
    /// exactly that reason. Nothing downstream reads a node through its plan
    /// position — every action carries its own `node_idx` — so a short plan
    /// executes correctly. The place the correspondence does have to hold is
    /// the build loop that establishes it, before the merge runs, and
    /// [`build_runtime_index`] asserts it there.
    pub(crate) fn rebuild_runtime_index(&mut self) {
        self.runtime_index =
            build_runtime_index(&self.inputs, &self.outputs, &self.initializers, &self.nodes);
    }
}

/// Loads an ONNX model from raw protobuf bytes, optimized for inference.
///
/// The graph optimizer runs as part of loading. It used to be a separate
/// [`optimize_onnx_graph`](crate::optimize_onnx_graph) call every caller had to
/// remember, and forgetting it did not produce an unoptimized model so much as
/// a slow one — or, on the Conv paths, a runtime shape error from a kernel
/// handed a weight the plan builder never got to permute.
///
/// Set `YSCV_ONNX_OPTIMIZE_OFF=1` to skip it process-wide, or call
/// [`load_onnx_model_unoptimized`] for the graph exactly as the file spells it.
/// The variable is read per load rather than cached, because loading is not a
/// hot path and a cached read would ignore anything set after the first model.
pub fn load_onnx_model(data: &[u8]) -> Result<OnnxModel, OnnxError> {
    let mut model = parse_onnx_model(data)?;
    if optimize_on_load() {
        crate::optimizer::optimize_onnx_graph(&mut model)?;
    } else {
        model.rebuild_runtime_index();
    }
    Ok(model)
}

/// Loads an ONNX model from raw protobuf bytes without optimizing it.
///
/// The graph comes back node-for-node as the file spells it. That is what an
/// inspection tool wants to report on, and what a test asserting the shape of a
/// fixture wants to assert against; it is not what an inference caller wants,
/// since none of the load-time fusions or weight folds have run.
pub fn load_onnx_model_unoptimized(data: &[u8]) -> Result<OnnxModel, OnnxError> {
    let mut model = parse_onnx_model(data)?;
    model.rebuild_runtime_index();
    Ok(model)
}

/// Whether [`load_onnx_model`] should run the optimizer.
fn optimize_on_load() -> bool {
    std::env::var_os("YSCV_ONNX_OPTIMIZE_OFF").is_none()
}

/// Decodes the protobuf into a model, leaving the runtime index empty.
///
/// The index is expensive to build — plan construction and weight prepacking —
/// and the optimizer invalidates it, so it is built exactly once, by whichever
/// entry point above finishes the load.
fn parse_onnx_model(data: &[u8]) -> Result<OnnxModel, OnnxError> {
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
        runtime_index: RuntimeModelIndex::default(),
    })
}

/// Loads an ONNX model from a file path, optimized for inference.
///
/// Accepts any path-like type (`&str`, `String`, `&Path`, `PathBuf`, etc.).
/// See [`load_onnx_model`] for the optimizer's role in loading and how to opt
/// out of it.
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
