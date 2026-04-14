#![doc = include_str!("../README.md")]

pub use yscv_autograd as autograd;
pub use yscv_cli as cli;
pub use yscv_detect as detect;
pub use yscv_eval as eval;
pub use yscv_imgproc as imgproc;
pub use yscv_kernels as kernels;
pub use yscv_model as model;
pub use yscv_onnx as onnx;
pub use yscv_optim as optim;
pub use yscv_recognize as recognize;
pub use yscv_tensor as tensor;
pub use yscv_track as track;
pub use yscv_video as video;

/// Commonly used types from across the yscv ecosystem.
pub mod prelude {
    // tensor
    pub use yscv_tensor::{DType, Device, Tensor, TensorError};

    // autograd
    pub use yscv_autograd::{Graph, NodeId};

    // model
    pub use yscv_model::{
        Compose, DataLoader, DataLoaderConfig, InferencePipeline, ModelLayer, SequentialModel,
        Trainer, TrainerConfig, Transform,
    };

    // losses
    pub use yscv_model::{
        bce_loss, cross_entropy_loss, dice_loss, focal_loss, mse_loss, smooth_l1_loss,
    };

    // callbacks
    pub use yscv_model::{BestModelCheckpoint, EarlyStopping};

    // transforms
    pub use yscv_model::{Normalize, PermuteDims, ScaleValues};

    // config types
    pub use yscv_model::{LossKind, OptimizerKind, SupervisedDataset};

    // optim
    pub use yscv_optim::{Adam, LearningRate, Sgd};

    // imgproc — module re-export plus essential functions and types
    pub use crate::imgproc;
    pub use yscv_imgproc::{
        ImageF32, ImageU8, ImgProcError, imread, imread_gray, imwrite, rgb_to_grayscale,
    };

    // onnx — model loading, inference, and error type
    pub use yscv_onnx::{
        OnnxError, OnnxModel, load_onnx_model, load_onnx_model_from_file, run_onnx_model,
    };

    // detect
    pub use yscv_detect::BoundingBox;

    // eval metrics
    pub use yscv_eval::{accuracy, confusion_matrix, f1_score};
}

#[cfg(test)]
mod tests {
    use super::prelude::*;

    #[test]
    fn smoke_create_tensor() {
        let t = Tensor::from_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        assert_eq!(t.shape(), &[2, 3]);
    }

    #[test]
    fn smoke_from_slice() {
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0]);
        assert_eq!(t.shape(), &[3]);
        assert_eq!(t.data(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn smoke_full() {
        let t = Tensor::full(vec![2, 3], 7.0).unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert!(t.data().iter().all(|&v| v == 7.0));
    }

    #[test]
    fn smoke_display() {
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0]);
        let s = format!("{t}");
        assert!(s.contains("Tensor("));
        assert!(s.contains("[3]"));
    }

    #[test]
    fn smoke_operator_add() {
        let a = Tensor::from_slice(&[1.0, 2.0]);
        let b = Tensor::from_slice(&[3.0, 4.0]);
        let c = &a + &b;
        assert_eq!(c.data(), &[4.0, 6.0]);
    }

    #[test]
    fn smoke_operator_sub() {
        let a = Tensor::from_slice(&[5.0, 3.0]);
        let b = Tensor::from_slice(&[1.0, 1.0]);
        let c = &a - &b;
        assert_eq!(c.data(), &[4.0, 2.0]);
    }

    #[test]
    fn smoke_operator_mul_scalar() {
        let a = Tensor::from_slice(&[2.0, 3.0]);
        let b = &a * 10.0;
        assert_eq!(b.data(), &[20.0, 30.0]);
    }

    #[test]
    fn smoke_operator_neg() {
        let a = Tensor::from_slice(&[1.0, -2.0]);
        let b = -&a;
        assert_eq!(b.data(), &[-1.0, 2.0]);
    }

    #[test]
    fn smoke_prelude_types_accessible() {
        // Verify new prelude types compile.
        let _: Option<DType> = None;
        let _: Option<Device> = None;
        let _: Option<ImgProcError> = None;
        let _: Option<OnnxError> = None;
        let _: Option<OnnxModel> = None;
    }

    #[test]
    fn smoke_build_sequential_model() {
        let graph = Graph::new();
        let model = SequentialModel::new(&graph);
        assert_eq!(model.layers().len(), 0);
    }

    #[test]
    fn smoke_inference_pipeline() {
        let graph = Graph::new();
        let model = SequentialModel::new(&graph);
        let pipeline = InferencePipeline::new(model);
        let input = Tensor::from_vec(vec![1, 4], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        // Empty model returns input unchanged.
        let output = pipeline.run(&input).unwrap();
        assert_eq!(output.shape(), &[1, 4]);
    }

    #[test]
    fn smoke_prelude_new_imports() {
        // Verify loss functions are callable via the prelude.
        let mut graph = Graph::new();
        let pred_t = Tensor::from_vec(vec![3], vec![0.9, 0.1, 0.0]).unwrap();
        let targ_t = Tensor::from_vec(vec![3], vec![1.0, 0.0, 0.0]).unwrap();
        let pred_node = graph.constant(pred_t);
        let targ_node = graph.constant(targ_t);
        let loss_node = mse_loss(&mut graph, pred_node, targ_node).unwrap();
        let loss_val = graph.value(loss_node).unwrap().data()[0];
        assert!(loss_val >= 0.0);

        // Verify eval metrics are callable via the prelude.
        let preds: Vec<usize> = vec![0, 1, 2, 0];
        let labels: Vec<usize> = vec![0, 1, 2, 1];
        let acc = accuracy(&preds, &labels).unwrap();
        assert!((acc - 0.75).abs() < 1e-5);

        // Verify transform types are accessible.
        let _scale = ScaleValues::new(1.0 / 255.0);
        let _norm = Normalize::new(vec![0.5], vec![0.5]);
        let _perm = PermuteDims::new(vec![2, 0, 1]);

        // Verify config types are accessible (just reference the types).
        let _: Option<OptimizerKind> = None;
        let _: Option<LossKind> = None;
    }

    #[test]
    fn smoke_compose_transform() {
        use yscv_model::{Normalize, ScaleValues};

        let compose = Compose::new()
            .add(ScaleValues::new(1.0 / 255.0))
            .add(Normalize::new(vec![0.5], vec![0.5]));
        let input = Tensor::from_vec(vec![2], vec![255.0, 0.0]).unwrap();
        let output = compose.apply(&input).unwrap();
        assert_eq!(output.shape(), &[2]);
        // 255 * (1/255) = 1.0, then (1.0 - 0.5) / 0.5 = 1.0
        let data = output.data();
        assert!((data[0] - 1.0).abs() < 1e-5);
        // 0 * (1/255) = 0.0, then (0.0 - 0.5) / 0.5 = -1.0
        assert!((data[1] - (-1.0)).abs() < 1e-5);
    }
}
