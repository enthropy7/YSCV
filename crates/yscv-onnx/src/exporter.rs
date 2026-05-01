use prost::Message;
use std::collections::HashSet;
use yscv_tensor::Tensor;

use crate::error::OnnxError;
use crate::proto::onnx;

/// Specification for an ONNX graph to export.
pub struct OnnxExportGraph {
    pub nodes: Vec<OnnxExportNode>,
    pub initializers: Vec<(String, Tensor)>,
    pub inputs: Vec<OnnxExportValueInfo>,
    pub outputs: Vec<OnnxExportValueInfo>,
    pub opset_version: i64,
    pub int64_initializers: Vec<String>,
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
    Int64Tensor(String, Tensor),
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
        let force_int64 = graph.int64_initializers.iter().any(|n| n == name);
        initializer_protos.push(tensor_to_proto(name, tensor, force_int64));
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
                    t: Some(tensor_to_proto(name, t, false)),
                    ..Default::default()
                }),
                OnnxExportAttr::Int64Tensor(name, t) => attrs.push(onnx::AttributeProto {
                    name: Some(name.clone()),
                    r#type: Some(4), // TENSOR
                    t: Some(tensor_to_proto_int64(name, t)),
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
            version: Some(graph.opset_version),
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
///
/// Loader/optimizer hot paths store Conv initializers in internal layouts
/// (KHWC variants) for runtime speed. ONNX files, however, must contain
/// Conv weights in OIHW. Convert those internal layouts back here so a
/// saved optimized model can be loaded again without double-packing its
/// weights.
pub fn onnx_model_to_export_graph(model: &crate::loader::OnnxModel) -> OnnxExportGraph {
    use crate::loader::OnnxAttribute;
    let int64_tensor_names = infer_int64_tensor_names(&model.nodes);
    let mut nodes: Vec<OnnxExportNode> = Vec::with_capacity(model.nodes.len());
    for n in &model.nodes {
        let attributes: Vec<OnnxExportAttr> = n
            .attributes
            .iter()
            .map(|(k, v)| match v {
                OnnxAttribute::Int(x) => OnnxExportAttr::Int(k.clone(), *x),
                OnnxAttribute::Float(x) => OnnxExportAttr::Float(k.clone(), *x),
                OnnxAttribute::Ints(x) => OnnxExportAttr::Ints(k.clone(), x.clone()),
                OnnxAttribute::Floats(x) => OnnxExportAttr::Floats(k.clone(), x.clone()),
                OnnxAttribute::String(x) => OnnxExportAttr::String(k.clone(), x.clone()),
                OnnxAttribute::Tensor(t)
                    if k == "value"
                        && n.outputs.iter().any(|out| int64_tensor_names.contains(out)) =>
                {
                    OnnxExportAttr::Int64Tensor(k.clone(), t.clone())
                }
                OnnxAttribute::Tensor(t) => OnnxExportAttr::Tensor(k.clone(), t.clone()),
            })
            .collect();
        if matches!(n.op_type.as_str(), "Conv_Relu" | "BatchNormalization_Relu")
            && n.outputs.len() == 1
        {
            let base_op = n.op_type.trim_end_matches("_Relu").to_string();
            let fused_out = n.outputs[0].clone();
            let inner_out = format!("{}__export_pre_relu", fused_out);
            nodes.push(OnnxExportNode {
                op_type: base_op,
                name: n.name.clone(),
                inputs: n.inputs.clone(),
                outputs: vec![inner_out.clone()],
                attributes,
            });
            nodes.push(OnnxExportNode {
                op_type: "Relu".to_string(),
                name: format!("{}__export_relu", n.name),
                inputs: vec![inner_out],
                outputs: vec![fused_out],
                attributes: Vec::new(),
            });
        } else {
            nodes.push(OnnxExportNode {
                op_type: n.op_type.clone(),
                name: n.name.clone(),
                inputs: n.inputs.clone(),
                outputs: n.outputs.clone(),
                attributes,
            });
        }
    }

    let initializers: Vec<(String, Tensor)> = model
        .initializers
        .iter()
        .map(|(k, v)| (k.clone(), initializer_for_onnx_export(model, k, v)))
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
        opset_version: model.opset_version,
        int64_initializers: model
            .initializers
            .keys()
            .filter(|name| int64_tensor_names.contains(*name))
            .cloned()
            .collect(),
    }
}

fn infer_int64_tensor_names(nodes: &[crate::loader::OnnxNode]) -> HashSet<String> {
    let mut names = HashSet::new();
    for node in nodes {
        match node.op_type.as_str() {
            "Reshape" | "Expand" | "Tile" | "Gather" | "GatherElements" | "GatherND" => {
                if let Some(name) = node.inputs.get(1) {
                    names.insert(name.clone());
                }
            }
            "Slice" => {
                for idx in 1..node.inputs.len().min(5) {
                    if let Some(name) = node.inputs.get(idx) {
                        names.insert(name.clone());
                    }
                }
            }
            "Unsqueeze" | "Squeeze" | "ReduceMean" | "ReduceSum" | "ReduceMax" | "ReduceMin"
            | "ReduceProd" | "ReduceL1" | "ReduceL2" => {
                if let Some(name) = node.inputs.get(1) {
                    names.insert(name.clone());
                }
            }
            _ => {}
        }
    }
    names
}

fn initializer_for_onnx_export(
    model: &crate::loader::OnnxModel,
    name: &str,
    tensor: &Tensor,
) -> Tensor {
    if model.khwc_weights.contains(name)
        && tensor.rank() == 4
        && let Ok(t) = tensor.permute(&[3, 2, 0, 1])
    {
        return t;
    }

    if model.group_khwc_weights.contains(name)
        && tensor.rank() == 4
        && let Ok(t) = tensor.permute(&[0, 3, 1, 2])
    {
        return t;
    }

    if model.dw_khwc_weights.contains(name)
        && let Some(t) = depthwise_khwc_to_oihw(tensor)
    {
        return t;
    }

    tensor.clone()
}

fn depthwise_khwc_to_oihw(tensor: &Tensor) -> Option<Tensor> {
    let shape = tensor.shape();
    if shape.len() != 4 {
        return None;
    }
    let (kh, kw, channels, depth_mult) = (shape[0], shape[1], shape[2], shape[3]);
    let out_channels = channels.checked_mul(depth_mult)?;
    let mut data = vec![0.0_f32; out_channels * kh * kw];
    let src = tensor.data();
    for c in 0..channels {
        for dm in 0..depth_mult {
            let oc = c * depth_mult + dm;
            for ki in 0..kh {
                for kj in 0..kw {
                    let src_idx = ((ki * kw + kj) * channels + c) * depth_mult + dm;
                    let dst_idx = (oc * kh + ki) * kw + kj;
                    data[dst_idx] = src[src_idx];
                }
            }
        }
    }
    Tensor::from_vec(vec![out_channels, 1, kh, kw], data).ok()
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

fn tensor_to_proto(name: &str, tensor: &Tensor, force_int64: bool) -> onnx::TensorProto {
    if force_int64 {
        return tensor_to_proto_int64(name, tensor);
    }

    if should_export_initializer_as_int8(name) {
        let raw_data: Vec<u8> = tensor
            .data()
            .iter()
            .map(|&v| v.round().clamp(-128.0, 127.0) as i8 as u8)
            .collect();
        return onnx::TensorProto {
            name: Some(name.to_string()),
            dims: tensor.shape().iter().map(|&d| d as i64).collect(),
            data_type: Some(3), // INT8
            raw_data: Some(raw_data),
            ..Default::default()
        };
    }

    if should_export_initializer_as_int32(name) {
        return onnx::TensorProto {
            name: Some(name.to_string()),
            dims: tensor.shape().iter().map(|&d| d as i64).collect(),
            data_type: Some(6), // INT32
            int32_data: tensor.data().iter().map(|&v| v.round() as i32).collect(),
            ..Default::default()
        };
    }

    onnx::TensorProto {
        name: Some(name.to_string()),
        dims: tensor.shape().iter().map(|&d| d as i64).collect(),
        data_type: Some(1), // FLOAT
        float_data: tensor.data().to_vec(),
        ..Default::default()
    }
}

fn tensor_to_proto_int64(name: &str, tensor: &Tensor) -> onnx::TensorProto {
    onnx::TensorProto {
        name: Some(name.to_string()),
        dims: tensor.shape().iter().map(|&d| d as i64).collect(),
        data_type: Some(7), // INT64
        int64_data: tensor.data().iter().map(|&v| v.round() as i64).collect(),
        ..Default::default()
    }
}

fn should_export_initializer_as_int8(name: &str) -> bool {
    name.ends_with("_q") || name.ends_with("_zp")
}

fn should_export_initializer_as_int32(name: &str) -> bool {
    name.ends_with("_bias_i32")
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
