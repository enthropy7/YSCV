use yscv_autograd::Graph;
use yscv_tensor::Tensor;

use crate::{
    LayerCheckpoint, ModelError, SequentialModel, checkpoint_from_json, checkpoint_to_json,
};

#[test]
fn sequential_checkpoint_roundtrip_preserves_outputs() {
    let mut graph = Graph::new();
    let mut model = SequentialModel::new(&graph);
    model
        .add_linear(
            &mut graph,
            2,
            2,
            Tensor::from_vec(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap(),
            Tensor::from_vec(vec![2], vec![0.5, -0.5]).unwrap(),
        )
        .unwrap();
    model.add_relu();

    let checkpoint = model.checkpoint(&graph).unwrap();
    let json = checkpoint_to_json(&checkpoint).unwrap();
    let loaded_checkpoint = checkpoint_from_json(&json).unwrap();

    let mut graph2 = Graph::new();
    let loaded_model = SequentialModel::from_checkpoint(&mut graph2, &loaded_checkpoint).unwrap();

    let input_a = graph.constant(Tensor::from_vec(vec![1, 2], vec![1.0, -1.0]).unwrap());
    let out_a = model.forward(&mut graph, input_a).unwrap();
    let input_b = graph2.constant(Tensor::from_vec(vec![1, 2], vec![1.0, -1.0]).unwrap());
    let out_b = loaded_model.forward(&mut graph2, input_b).unwrap();

    assert_eq!(graph.value(out_a).unwrap(), graph2.value(out_b).unwrap());
    assert_eq!(loaded_checkpoint.layers.len(), 2);
    assert!(matches!(loaded_checkpoint.layers[1], LayerCheckpoint::ReLU));
}

#[test]
fn sequential_checkpoint_roundtrip_preserves_leaky_relu_layer() {
    let mut graph = Graph::new();
    let mut model = SequentialModel::new(&graph);
    model
        .add_linear(
            &mut graph,
            2,
            2,
            Tensor::from_vec(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap(),
            Tensor::from_vec(vec![2], vec![0.5, -0.5]).unwrap(),
        )
        .unwrap();
    model.add_leaky_relu(0.2).unwrap();

    let checkpoint = model.checkpoint(&graph).unwrap();
    let json = checkpoint_to_json(&checkpoint).unwrap();
    let loaded_checkpoint = checkpoint_from_json(&json).unwrap();

    let mut graph2 = Graph::new();
    let loaded_model = SequentialModel::from_checkpoint(&mut graph2, &loaded_checkpoint).unwrap();

    let input_a = graph.constant(Tensor::from_vec(vec![1, 2], vec![1.0, -1.0]).unwrap());
    let out_a = model.forward(&mut graph, input_a).unwrap();
    let input_b = graph2.constant(Tensor::from_vec(vec![1, 2], vec![1.0, -1.0]).unwrap());
    let out_b = loaded_model.forward(&mut graph2, input_b).unwrap();

    assert_eq!(graph.value(out_a).unwrap(), graph2.value(out_b).unwrap());
    assert!(matches!(
        loaded_checkpoint.layers[1],
        LayerCheckpoint::LeakyReLU { negative_slope } if (negative_slope - 0.2).abs() <= f32::EPSILON
    ));
}

#[test]
fn checkpoint_parser_reports_invalid_json() {
    let err = checkpoint_from_json("{not-valid-json").unwrap_err();
    assert!(matches!(err, ModelError::CheckpointSerialization { .. }));
}

#[test]
fn checkpoint_roundtrip_cnn_layers() {
    let graph = Graph::new();
    let mut model = SequentialModel::new(&graph);
    model.add_conv2d_zero(3, 8, 3, 3, 1, 1, true).unwrap();
    model.add_batch_norm2d_identity(8, 1e-5).unwrap();
    model.add_max_pool2d(2, 2, 2, 2).unwrap();
    model.add_avg_pool2d(2, 2, 1, 1).unwrap();
    model.add_flatten();
    model.add_softmax();

    let checkpoint = model.checkpoint(&graph).unwrap();
    let json = checkpoint_to_json(&checkpoint).unwrap();
    let restored_checkpoint = checkpoint_from_json(&json).unwrap();
    assert_eq!(checkpoint, restored_checkpoint);

    let mut graph2 = Graph::new();
    let model2 = SequentialModel::from_checkpoint(&mut graph2, &restored_checkpoint).unwrap();
    assert_eq!(model.layers().len(), model2.layers().len());
}

#[test]
fn save_load_weights_roundtrip() {
    let mut tensors = std::collections::HashMap::default();
    tensors.insert(
        "layer1.weight".to_string(),
        Tensor::from_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
    );
    tensors.insert(
        "layer1.bias".to_string(),
        Tensor::from_vec(vec![3], vec![0.1, 0.2, 0.3]).unwrap(),
    );

    let dir = std::env::temp_dir().join("yscv_test_weights");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("test_weights.bin");

    crate::save_weights(&path, &tensors).unwrap();
    let loaded = crate::load_weights(&path).unwrap();

    assert_eq!(loaded.len(), 2);
    assert_eq!(loaded["layer1.weight"].shape(), &[2, 3]);
    assert_eq!(
        loaded["layer1.weight"].data(),
        tensors["layer1.weight"].data()
    );
    assert_eq!(loaded["layer1.bias"].data(), tensors["layer1.bias"].data());

    std::fs::remove_file(&path).ok();
}

#[test]
fn embedding_checkpoint_roundtrip() {
    let mut graph = Graph::new();
    let mut model = SequentialModel::new(&graph);
    let weight = Tensor::from_vec(vec![3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    model.add_embedding(&mut graph, 3, 2, weight).unwrap();
    let ckpt = model.checkpoint(&graph).unwrap();
    let json = checkpoint_to_json(&ckpt).unwrap();
    let restored_ckpt = checkpoint_from_json(&json).unwrap();
    assert_eq!(ckpt.layers.len(), restored_ckpt.layers.len());
}

#[test]
fn depthwise_conv2d_checkpoint_round_trip() {
    let mut g = yscv_autograd::Graph::new();
    let mut model = crate::SequentialModel::new(&g);
    model
        .add_depthwise_conv2d_zero(2, 3, 3, 1, 1, true)
        .unwrap();
    model.register_cnn_params(&mut g);
    let ckpt = model.checkpoint(&g).unwrap();
    let json = crate::checkpoint_to_json(&ckpt).unwrap();
    let restored_ckpt = crate::checkpoint_from_json(&json).unwrap();
    assert_eq!(ckpt, restored_ckpt);
}

#[test]
fn separable_conv2d_checkpoint_round_trip() {
    let mut g = yscv_autograd::Graph::new();
    let mut model = crate::SequentialModel::new(&g);
    model
        .add_separable_conv2d_zero(3, 8, 3, 3, 1, 1, true)
        .unwrap();
    model.register_cnn_params(&mut g);
    let ckpt = model.checkpoint(&g).unwrap();
    let json = crate::checkpoint_to_json(&ckpt).unwrap();
    let restored_ckpt = crate::checkpoint_from_json(&json).unwrap();
    assert_eq!(ckpt, restored_ckpt);
}
