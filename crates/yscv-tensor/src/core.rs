//! Tensor and numeric primitives for the yscv framework.
#![allow(unsafe_code)]

pub const CRATE_ID: &str = "yscv-tensor";

#[path = "aligned.rs"]
mod aligned;
#[path = "error.rs"]
mod error;
#[path = "linalg.rs"]
mod linalg;
#[path = "ops.rs"]
mod ops;
#[path = "shape.rs"]
mod shape;
#[path = "simd/mod.rs"]
mod simd;
#[path = "tensor.rs"]
mod tensor;

pub use aligned::AlignedVec;
pub use error::{DType, TensorError};
pub use tensor::{Device, Tensor};

#[cfg(test)]
#[path = "proptest_tests.rs"]
mod proptest_tests;

#[path = "tests/mod.rs"]
#[cfg(test)]
mod tests;
