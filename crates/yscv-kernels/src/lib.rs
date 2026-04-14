#![doc = include_str!("../README.md")]
#![deny(unsafe_code)]

mod core;

#[cfg(all(target_os = "macos", feature = "metal-backend"))]
#[allow(unsafe_code)]
#[path = "metal/mod.rs"]
pub mod metal_backend;

#[cfg(feature = "rknn")]
#[allow(unsafe_code)]
pub mod rknn;

#[cfg(feature = "rknn")]
pub use rknn as rknn_backend;

pub use core::*;

#[cfg(all(target_os = "macos", feature = "metal-backend"))]
pub use metal_backend::metal_conv::{MetalConv, MetalInference};

#[cfg(feature = "rknn")]
pub use rknn::{
    AsyncFrame, ContextPool, CustomOp, CustomOpAttr, CustomOpContext, CustomOpHandler,
    CustomOpRegistration, CustomOpTarget, CustomOpTensor, InferenceBackend, MAX_CUSTOM_OP_SLOTS,
    MatmulQuantParams, MemAllocFlags, MemSize, MemSyncMode, NpuCoreMask, OpPerf, PerfDetail,
    RknnBackend, RknnCompileConfig, RknnInferenceHandle, RknnMatmul, RknnMatmulIoAttr,
    RknnMatmulLayout, RknnMatmulQuantType, RknnMatmulShape, RknnMatmulTensorAttr, RknnMatmulType,
    RknnMem, RknnPipelinedPool, RknnQuantType, RknnTensorAttr, RknnTensorFormat, RknnTensorType,
    compile_onnx_to_rknn, detect_backend, load_onnx_as_rknn, rknn_available,
    rknn_compiler_available,
};
