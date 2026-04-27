use prost::Message;
use yscv_tensor::Tensor;

use crate::error::OnnxError;
use crate::proto::onnx;

/// Specification for an ONNX graph to export.
pub struct OnnxExportGraph {
    pub nodes: Vec<OnnxExportNode>,
    pub initializers: Vec<(String, Tensor)>,
    pub inputs: Vec<OnnxExportValueInfo>,
    pub outputs: Vec<OnnxExportValueInfo>,
}

/// A node in the export graph.
pub struct OnnxExportNode {
    pub op_type: String,
    pub name: String,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub attributes: Vec<OnnxExportAttr>,
}

/// An attribute value for export. Mirrors the variants of
/// `crate::loader::OnnxAttribute` so a loaded `OnnxModel` can be
/// serialised back without losing attribute fidelity.
pub enum OnnxExportAttr {
    Int(String, i64),
    Float(String, f32),
    Ints(String, Vec<i64>),
    String(String, String),
    Floats(String, Vec<f32>),
    Tensor(String, Tensor),
}

/// Shape + name info for graph I/O.
pub struct OnnxExportValueInfo {
    pub name: String,
    pub shape: Vec<i64>,
}

/// Exports an ONNX model graph to protobuf bytes.
pub fn export_onnx_model(
    graph: &OnnxExportGraph,
    producer_name: &str,
    model_name: &str,
) -> Result<Vec<u8>, OnnxError> {
    let mut initializer_protos = Vec::new();
    for (name, tensor) in &graph.initializers {
        initializer_protos.push(tensor_to_proto(name, tensor));
    }

    let mut node_protos = Vec::new();
    for node in &graph.nodes {
        let mut attrs = Vec::new();
        for attr in &node.attributes {
            match attr {
                OnnxExportAttr::Int(name, v) => attrs.push(onnx::AttributeProto {
                    name: Some(name.clone()),
                    r#type: Some(2), // INT
                    i: Some(*v),
                    ..Default::default()
                }),
                OnnxExportAttr::Float(name, v) => attrs.push(onnx::AttributeProto {
                    name: Some(name.clone()),
                    r#type: Some(1), // FLOAT
                    f: Some(*v),
                    ..Default::default()
                }),
                OnnxExportAttr::Ints(name, v) => attrs.push(onnx::AttributeProto {
                    name: Some(name.clone()),
                    r#type: Some(7), // INTS
                    ints: v.clone(),
                    ..Default::default()
                }),
                OnnxExportAttr::String(name, v) => attrs.push(onnx::AttributeProto {
                    name: Some(name.clone()),
                    r#type: Some(3), // STRING
                    s: Some(v.as_bytes().to_vec()),
                    ..Default::default()
                }),
                OnnxExportAttr::Floats(name, v) => attrs.push(onnx::AttributeProto {
                    name: Some(name.clone()),
                    r#type: Some(6), // FLOATS
                    floats: v.clone(),
                    ..Default::default()
                }),
                OnnxExportAttr::Tensor(name, t) => attrs.push(onnx::AttributeProto {
                    name: Some(name.clone()),
                    r#type: Some(4), // TENSOR
                    t: Some(tensor_to_proto(name, t)),
                    ..Default::default()
                }),
            }
        }
        node_protos.push(onnx::NodeProto {
            op_type: Some(node.op_type.clone()),
            name: Some(node.name.clone()),
            input: node.inputs.clone(),
            output: node.outputs.clone(),
            attribute: attrs,
            ..Default::default()
        });
    }

    let input_protos: Vec<onnx::ValueInfoProto> = graph
        .inputs
        .iter()
        .map(|vi| make_value_info(&vi.name, &vi.shape))
        .collect();

    let output_protos: Vec<onnx::ValueInfoProto> = graph
        .outputs
        .iter()
        .map(|vi| make_value_info(&vi.name, &vi.shape))
        .collect();

    let graph_proto = onnx::GraphProto {
        name: Some(model_name.to_string()),
        node: node_protos,
        initializer: initializer_protos,
        input: input_protos,
        output: output_protos,
        ..Default::default()
    };

    let model_proto = onnx::ModelProto {
        ir_version: Some(7),
        producer_name: Some(producer_name.to_string()),
        model_version: Some(1),
        graph: Some(graph_proto),
        opset_import: vec![onnx::OperatorSetIdProto {
            version: Some(13),
            ..Default::default()
        }],
        ..Default::default()
    };

    let mut buf = Vec::new();
    model_proto.encode(&mut buf).map_err(|e| OnnxError::Io {
        message: e.to_string(),
    })?;
    Ok(buf)
}

/// Exports an ONNX model graph to a file.
pub fn export_onnx_model_to_file(
    graph: &OnnxExportGraph,
    producer_name: &str,
    model_name: &str,
    path: &std::path::Path,
) -> Result<(), OnnxError> {
    let bytes = export_onnx_model(graph, producer_name, model_name)?;
    std::fs::write(path, bytes).map_err(|e| OnnxError::Io {
        message: format!("{}: {e}", path.display()),
    })
}

/// Convert a loaded [`crate::loader::OnnxModel`] into an
/// [`OnnxExportGraph`] that the exporter can serialise. Round-trip with
/// `load_onnx_model` should preserve nodes, attributes (all 6
/// [`crate::loader::OnnxAttribute`] variants), and initializer tensor
/// data; input/output shapes are written as scalar placeholders since
/// `OnnxModel` only tracks names — downstream tools re-infer shapes.
pub fn onnx_model_to_export_graph(model: &crate::loader::OnnxModel) -> OnnxExportGraph {
    use crate::loader::OnnxAttribute;
    let nodes: Vec<OnnxExportNode> = model
        .nodes
        .iter()
        .map(|n| OnnxExportNode {
            op_type: n.op_type.clone(),
            name: n.name.clone(),
            inputs: n.inputs.clone(),
            outputs: n.outputs.clone(),
            attributes: n
                .attributes
                .iter()
                .map(|(k, v)| match v {
                    OnnxAttribute::Int(x) => OnnxExportAttr::Int(k.clone(), *x),
                    OnnxAttribute::Float(x) => OnnxExportAttr::Float(k.clone(), *x),
                    OnnxAttribute::Ints(x) => OnnxExportAttr::Ints(k.clone(), x.clone()),
                    OnnxAttribute::Floats(x) => OnnxExportAttr::Floats(k.clone(), x.clone()),
                    OnnxAttribute::String(x) => OnnxExportAttr::String(k.clone(), x.clone()),
                    OnnxAttribute::Tensor(t) => OnnxExportAttr::Tensor(k.clone(), t.clone()),
                })
                .collect(),
        })
        .collect();

    let initializers: Vec<(String, Tensor)> = model
        .initializers
        .iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();

    // OnnxModel only carries names for graph I/O; emit empty shapes which
    // the loader treats as dynamic. Downstream consumers re-infer.
    let inputs = model
        .inputs
        .iter()
        .map(|n| OnnxExportValueInfo {
            name: n.clone(),
            shape: Vec::new(),
        })
        .collect();
    let outputs = model
        .outputs
        .iter()
        .map(|n| OnnxExportValueInfo {
            name: n.clone(),
            shape: Vec::new(),
        })
        .collect();

    OnnxExportGraph {
        nodes,
        initializers,
        inputs,
        outputs,
    }
}

/// Save an [`crate::loader::OnnxModel`] back to an ONNX file. Convenience
/// wrapper around [`onnx_model_to_export_graph`] + [`export_onnx_model_to_file`].
pub fn save_onnx_model_to_file(
    model: &crate::loader::OnnxModel,
    path: &std::path::Path,
) -> Result<(), OnnxError> {
    let graph = onnx_model_to_export_graph(model);
    export_onnx_model_to_file(&graph, "yscv", &model.graph_name, path)
}

fn tensor_to_proto(name: &str, tensor: &Tensor) -> onnx::TensorProto {
    onnx::TensorProto {
        name: Some(name.to_string()),
        dims: tensor.shape().iter().map(|&d| d as i64).collect(),
        data_type: Some(1), // FLOAT
        float_data: tensor.data().to_vec(),
        ..Default::default()
    }
}

fn make_value_info(name: &str, shape: &[i64]) -> onnx::ValueInfoProto {
    let dims: Vec<onnx::tensor_shape_proto::Dimension> = shape
        .iter()
        .map(|&d| onnx::tensor_shape_proto::Dimension {
            value: Some(onnx::tensor_shape_proto::dimension::Value::DimValue(d)),
            ..Default::default()
        })
        .collect();

    onnx::ValueInfoProto {
        name: Some(name.to_string()),
        r#type: Some(onnx::TypeProto {
            value: Some(onnx::type_proto::Value::TensorType(
                onnx::type_proto::Tensor {
                    elem_type: Some(1), // FLOAT
                    shape: Some(onnx::TensorShapeProto { dim: dims }),
                },
            )),
            ..Default::default()
        }),
        ..Default::default()
    }
}
