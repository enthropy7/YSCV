use rustc_hash::FxHashMap;

use yscv_tensor::Tensor;

use super::{build_minimal_onnx_model, make_ints_attr};
use crate::proto::onnx;
use crate::{Dim, TensorShape, graph_cost, infer_shapes, load_onnx_model};

fn tensor_proto(name: &str, shape: Vec<i64>, data: Vec<f32>) -> onnx::TensorProto {
    onnx::TensorProto {
        name: Some(name.to_string()),
        dims: shape,
        data_type: Some(1),
        float_data: data,
        ..Default::default()
    }
}

#[test]
fn infers_conv_shape_and_cost() {
    let node = onnx::NodeProto {
        op_type: Some("Conv".into()),
        name: Some("conv".into()),
        input: vec!["x".into(), "w".into()],
        output: vec!["y".into()],
        attribute: vec![
            make_ints_attr("strides", vec![2, 2]),
            make_ints_attr("pads", vec![1, 1, 1, 1]),
        ],
        ..Default::default()
    };
    let model = load_onnx_model(&build_minimal_onnx_model(
        vec![node],
        vec![tensor_proto(
            "w",
            vec![8, 3, 3, 3],
            vec![0.0; 8 * 3 * 3 * 3],
        )],
        vec!["x"],
        vec!["y"],
    ))
    .unwrap();
    let input_shapes =
        FxHashMap::from_iter([("x".to_string(), TensorShape::known(vec![1, 3, 32, 32]))]);

    let inferred = infer_shapes(&model, &input_shapes);
    assert!(
        inferred.diagnostics.is_empty(),
        "{:?}",
        inferred.diagnostics
    );
    assert_eq!(
        inferred.shapes["y"].dims,
        vec![Dim::Known(1), Dim::Known(8), Dim::Known(16), Dim::Known(16)]
    );

    let cost = graph_cost(&model, &inferred);
    assert_eq!(cost.unknown_nodes, 0);
    assert_eq!(cost.estimated_macs, 8 * 16 * 16 * 3 * 3 * 3);
}

#[test]
fn infers_reshape_and_matmul_shapes() {
    let reshape = onnx::NodeProto {
        op_type: Some("Reshape".into()),
        name: Some("reshape".into()),
        input: vec!["x".into(), "shape".into()],
        output: vec!["flat".into()],
        ..Default::default()
    };
    let matmul = onnx::NodeProto {
        op_type: Some("MatMul".into()),
        name: Some("matmul".into()),
        input: vec!["flat".into(), "w".into()],
        output: vec!["y".into()],
        ..Default::default()
    };
    let model = load_onnx_model(&build_minimal_onnx_model(
        vec![reshape, matmul],
        vec![
            tensor_proto("shape", vec![2], vec![1.0, 48.0]),
            tensor_proto("w", vec![48, 10], vec![0.0; 48 * 10]),
        ],
        vec!["x"],
        vec!["y"],
    ))
    .unwrap();
    let input_shapes =
        FxHashMap::from_iter([("x".to_string(), TensorShape::known(vec![1, 3, 4, 4]))]);

    let inferred = infer_shapes(&model, &input_shapes);
    assert!(
        inferred.diagnostics.is_empty(),
        "{:?}",
        inferred.diagnostics
    );
    assert_eq!(inferred.shapes["flat"].as_known_dims(), Some(vec![1, 48]));
    assert_eq!(inferred.shapes["y"].as_known_dims(), Some(vec![1, 10]));

    let cost = graph_cost(&model, &inferred);
    assert_eq!(cost.unknown_nodes, 0);
    assert_eq!(cost.estimated_macs, 480);
}

#[test]
fn reports_unknowns_without_guessing() {
    let node = onnx::NodeProto {
        op_type: Some("Resize".into()),
        name: Some("resize".into()),
        input: vec!["x".into()],
        output: vec!["y".into()],
        ..Default::default()
    };
    let model = load_onnx_model(&build_minimal_onnx_model(
        vec![node],
        vec![],
        vec!["x"],
        vec!["y"],
    ))
    .unwrap();
    let input_shapes =
        FxHashMap::from_iter([("x".to_string(), TensorShape::known(vec![1, 3, 8, 8]))]);

    let inferred = infer_shapes(&model, &input_shapes);
    assert_eq!(inferred.diagnostics.len(), 1);

    let cost = graph_cost(&model, &inferred);
    assert_eq!(cost.unknown_nodes, 1);
    assert_eq!(cost.known_nodes, 0);
}

#[test]
fn infers_shapes_from_initializer_inputs() {
    let node = onnx::NodeProto {
        op_type: Some("Relu".into()),
        name: Some("relu".into()),
        input: vec!["x".into()],
        output: vec!["y".into()],
        ..Default::default()
    };
    let init = Tensor::from_vec(vec![2], vec![1.0, -1.0]).unwrap();
    let model = load_onnx_model(&build_minimal_onnx_model(
        vec![node],
        vec![tensor_proto("x", vec![2], init.data().to_vec())],
        vec!["x"],
        vec!["y"],
    ))
    .unwrap();

    let inferred = infer_shapes(&model, &FxHashMap::default());
    assert_eq!(inferred.shapes["y"].as_known_dims(), Some(vec![2]));
}

#[test]
fn global_average_pool_cost_reads_spatial_from_inferred_shape() {
    // The pooled input is an activation, never an initializer, so the cost of
    // averaging every input pixel has to come from shape inference. A [1,8,4,4]
    // input pools to [1,8,1,1]: 8 output elements each averaging 16 pixels.
    let node = onnx::NodeProto {
        op_type: Some("GlobalAveragePool".into()),
        name: Some("gap".into()),
        input: vec!["x".into()],
        output: vec!["y".into()],
        ..Default::default()
    };
    let model = load_onnx_model(&build_minimal_onnx_model(
        vec![node],
        vec![],
        vec!["x"],
        vec!["y"],
    ))
    .unwrap();
    let input_shapes =
        FxHashMap::from_iter([("x".to_string(), TensorShape::known(vec![1, 8, 4, 4]))]);

    let inferred = infer_shapes(&model, &input_shapes);
    assert!(
        inferred.diagnostics.is_empty(),
        "{:?}",
        inferred.diagnostics
    );
    assert_eq!(inferred.shapes["y"].as_known_dims(), Some(vec![1, 8, 1, 1]));

    let cost = graph_cost(&model, &inferred);
    assert_eq!(cost.unknown_nodes, 0);
    assert_eq!(cost.nodes[0].element_ops, 8 * 16);
}
