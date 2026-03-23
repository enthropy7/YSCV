mod conv_pool;
mod dynamic_shapes;
mod elementwise;
mod exporter;
mod gather_scatter;
mod gemm_matmul;
mod integration;
mod loader;
mod normalization;
mod optimizer;
mod reshape;

use prost::Message;

use super::loader::{OnnxAttribute, load_onnx_model};
use super::proto::onnx;
use super::runner::run_onnx_model;
use std::collections::HashMap;
use yscv_tensor::Tensor;

pub(super) fn build_minimal_onnx_model(
    nodes: Vec<onnx::NodeProto>,
    initializers: Vec<onnx::TensorProto>,
    inputs: Vec<&str>,
    outputs: Vec<&str>,
) -> Vec<u8> {
    let graph = onnx::GraphProto {
        name: Some("test_graph".into()),
        node: nodes,
        initializer: initializers,
        input: inputs
            .into_iter()
            .map(|n| onnx::ValueInfoProto {
                name: Some(n.into()),
                ..Default::default()
            })
            .collect(),
        output: outputs
            .into_iter()
            .map(|n| onnx::ValueInfoProto {
                name: Some(n.into()),
                ..Default::default()
            })
            .collect(),
        ..Default::default()
    };
    let model = onnx::ModelProto {
        ir_version: Some(8),
        opset_import: vec![onnx::OperatorSetIdProto {
            domain: Some(String::new()),
            version: Some(13),
        }],
        producer_name: Some("test".into()),
        graph: Some(graph),
        ..Default::default()
    };
    model.encode_to_vec()
}

pub(super) fn run_single_op(
    op_type: &str,
    inputs: Vec<(&str, Tensor)>,
    initializers: Vec<(&str, Tensor)>,
    attrs: Vec<onnx::AttributeProto>,
    input_names: Vec<&str>,
    output_name: &str,
) -> Tensor {
    let node = onnx::NodeProto {
        op_type: Some(op_type.into()),
        name: Some("op0".into()),
        input: input_names.iter().map(|s| s.to_string()).collect(),
        output: vec![output_name.into()],
        attribute: attrs,
        ..Default::default()
    };
    let init_protos: Vec<onnx::TensorProto> = initializers
        .iter()
        .map(|(name, t)| onnx::TensorProto {
            name: Some(name.to_string()),
            dims: t.shape().iter().map(|&d| d as i64).collect(),
            data_type: Some(1),
            float_data: t.data().to_vec(),
            ..Default::default()
        })
        .collect();
    let all_input_names: Vec<&str> = input_names.clone();
    let bytes =
        build_minimal_onnx_model(vec![node], init_protos, all_input_names, vec![output_name]);
    let model = load_onnx_model(&bytes).unwrap();
    let mut feed = HashMap::new();
    for (name, tensor) in inputs {
        feed.insert(name.to_string(), tensor);
    }
    let result = run_onnx_model(&model, feed).unwrap();
    result[output_name].clone()
}

pub(super) fn make_ints_attr(name: &str, values: Vec<i64>) -> onnx::AttributeProto {
    onnx::AttributeProto {
        name: Some(name.into()),
        r#type: Some(7),
        ints: values,
        ..Default::default()
    }
}

pub(super) fn make_int_attr(name: &str, value: i64) -> onnx::AttributeProto {
    onnx::AttributeProto {
        name: Some(name.into()),
        r#type: Some(2),
        i: Some(value),
        ..Default::default()
    }
}

pub(super) fn make_float_attr(name: &str, value: f32) -> onnx::AttributeProto {
    onnx::AttributeProto {
        name: Some(name.into()),
        r#type: Some(1),
        f: Some(value),
        ..Default::default()
    }
}

/// Build an ONNX attribute of type TENSOR (type=4) containing a TensorProto.
pub(super) fn make_tensor_attr(name: &str, dims: Vec<i64>, data: Vec<f32>) -> onnx::AttributeProto {
    onnx::AttributeProto {
        name: Some(name.into()),
        r#type: Some(4), // TENSOR
        t: Some(onnx::TensorProto {
            dims,
            data_type: Some(1), // FLOAT
            float_data: data,
            ..Default::default()
        }),
        ..Default::default()
    }
}
