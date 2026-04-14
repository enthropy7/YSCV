use thiserror::Error;
use yscv_kernels::KernelError;
use yscv_tensor::TensorError;

/// Errors returned by autograd graph building and backpropagation.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum AutogradError {
    #[error("graph node not found: id={id}")]
    NodeNotFound { id: usize },
    #[error(transparent)]
    Tensor(#[from] TensorError),
    #[error(transparent)]
    Kernel(#[from] KernelError),
    #[error("backward target must be scalar, got shape={shape:?}")]
    NonScalarTarget { shape: Vec<usize> },
    #[error("invalid gradient shape for node {node}: expected {expected:?}, got {got:?}")]
    InvalidGradientShape {
        node: usize,
        expected: Vec<usize>,
        got: Vec<usize>,
    },
    #[error("invalid tensor rank for {op}: expected {expected}, got {got}")]
    InvalidRankForOperation {
        op: &'static str,
        expected: usize,
        got: usize,
    },
    #[error("cannot reduce broadcast gradient from shape {upstream:?} to {target:?}")]
    BroadcastGradientIncompatible {
        upstream: Vec<usize>,
        target: Vec<usize>,
    },
    #[error("invalid truncate request: requested={requested}, available={available}")]
    InvalidTruncate { requested: usize, available: usize },
    #[error("backend error: {0}")]
    BackendError(String),
}
