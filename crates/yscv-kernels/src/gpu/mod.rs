//! wgpu-based GPU backend — dispatches to Vulkan (Linux/Win), Metal (macOS), DX12 (Win).
//!
//! Module layout:
//!
//! - [`backend`] — main `GpuBackend` type, `Pipelines` cache, trait impls, public ops
//! - [`buffer`] — `GpuBuffer` (device-resident tensor) and internal `BufferPool`
//! - [`recorded`] — `RecordedOp` enum for compiled replay
//! - [`shaders`] — embedded WGSL shader source (`include_str!`)
//! - [`helpers`] — pure utility fns: f16 bit conversion, div-ceil, shape compare

pub(crate) mod backend;
pub(crate) mod buffer;
pub(crate) mod helpers;
pub(crate) mod recorded;
pub(crate) mod shaders;

pub use backend::{GpuBackend, gpu_batch_norm, gpu_layer_norm, gpu_transpose};
pub use buffer::GpuBuffer;
pub use recorded::RecordedOp;
