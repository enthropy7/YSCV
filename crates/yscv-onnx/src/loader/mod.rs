use prost::Message;
use rustc_hash::{FxHashMap, FxHashSet};
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
}

impl OnnxModel {
    /// Returns the weight tensor for a given initializer name, if present.
    pub fn get_initializer(&self, name: &str) -> Option<&Tensor> {
        self.initializers.get(name)
    }

    /// Returns the number of operator nodes in the graph.
    pub const fn node_count(&self) -> usize {
        self.nodes.len()
    }
}

enum LoaderSource<'a> {
    Path(std::path::PathBuf),
    Bytes(&'a [u8]),
    Model(&'a OnnxModel),
}

/// Configures decoding and compilation of an ONNX executable.
pub struct OnnxLoader<'a> {
    source: LoaderSource<'a>,
    optimization_level: Option<crate::plan::OptimizationLevel>,
}

impl<'a> OnnxLoader<'a> {
    /// Loads an ONNX model from a file when [`Self::build`] or [`Self::decode`] is called.
    pub fn load(path: impl AsRef<std::path::Path>) -> Self {
        Self {
            source: LoaderSource::Path(path.as_ref().to_path_buf()),
            optimization_level: None,
        }
    }

    /// Loads an ONNX model from protobuf bytes.
    pub fn from_bytes(data: &'a [u8]) -> Self {
        Self {
            source: LoaderSource::Bytes(data),
            optimization_level: None,
        }
    }

    /// Builds an executable from an existing source model.
    pub fn from_model(model: &'a OnnxModel) -> Self {
        Self {
            source: LoaderSource::Model(model),
            optimization_level: None,
        }
    }

    /// Selects whether IR optimization passes run during [`Self::build`].
    pub fn optimization_level(mut self, level: crate::plan::OptimizationLevel) -> Self {
        self.optimization_level = Some(level);
        self
    }

    /// Decodes the canonical source model without constructing runtime state.
    pub fn decode(self) -> Result<OnnxModel, OnnxError> {
        match self.source {
            LoaderSource::Path(path) => {
                let data = std::fs::read(&path).map_err(|source| OnnxError::Io {
                    path,
                    source: std::sync::Arc::new(source),
                })?;
                parse_onnx_model(&data)
            }
            LoaderSource::Bytes(data) => parse_onnx_model(data),
            LoaderSource::Model(model) => Ok(model.clone()),
        }
    }

    /// Decodes, lowers, and compiles an immutable executable.
    pub fn build(self) -> Result<crate::plan::OnnxExecutable, OnnxError> {
        let optimization_level = self.optimization_level.unwrap_or_else(|| {
            if std::env::var("YSCV_ONNX_OPTIMIZE_OFF").as_deref() == Ok("1") {
                crate::plan::OptimizationLevel::Disabled
            } else {
                crate::plan::OptimizationLevel::Enabled
            }
        });
        match self.source {
            LoaderSource::Model(model) => crate::plan::build_executable(model, optimization_level),
            _ => {
                let model = self.decode()?;
                crate::plan::build_executable(&model, optimization_level)
            }
        }
    }
}

/// Decodes the protobuf into a canonical source model without runtime state.
fn parse_onnx_model(data: &[u8]) -> Result<OnnxModel, OnnxError> {
    let model_proto = onnx::ModelProto::decode(data)?;

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

    // ONNX graphs are in SSA form: a node output may not redefine a graph
    // input, an initializer, or another node's output. An initializer may
    // share its name with a graph input, which is how older IR versions give
    // an input a default.
    let mut defined: FxHashSet<&str> = inputs.iter().map(String::as_str).collect();
    defined.extend(initializers.keys().map(String::as_str));
    for name in nodes.iter().flat_map(|n| &n.outputs) {
        if !name.is_empty() && !defined.insert(name) {
            return Err(OnnxError::DuplicateDefinition { name: name.clone() });
        }
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
    })
}

mod convert;
use convert::*;
