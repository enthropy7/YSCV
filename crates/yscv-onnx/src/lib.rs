#![doc = include_str!("../README.md")]

pub const CRATE_ID: &str = "yscv-onnx";

pub mod cpu_topology;
mod dtype;
#[path = "error.rs"]
mod error;
#[path = "exporter.rs"]
mod exporter;
pub mod generate;
#[path = "loader.rs"]
mod loader;
#[path = "optimizer.rs"]
mod optimizer;
mod proto;
pub mod quantize;
mod runner;

pub use dtype::{OnnxDtype, OnnxTensorData};
pub use error::OnnxError;
pub use exporter::{
    OnnxExportAttr, OnnxExportGraph, OnnxExportNode, OnnxExportValueInfo, export_onnx_model,
    export_onnx_model_to_file, onnx_model_to_export_graph, save_onnx_model_to_file,
};
pub use generate::{GenerateConfig, generate};
pub use loader::{
    OnnxAttribute, OnnxModel, OnnxNode, OnnxTensor, load_onnx_model, load_onnx_model_from_file,
};
pub use optimizer::{
    GraphStats, fold_constants, fold_conv_bn, fuse_bn_relu, fuse_conv_relu, graph_stats,
    optimize_onnx_graph, strip_qdq_within_fusion_chains,
};
pub use quantize::quantize_weights_int4;
pub use quantize::{CalibrationCollector, CalibrationScope, MinMax};
pub use quantize::{
    QuantParams, QuantTarget, derive_asymmetric, derive_symmetric, int4_symmetric_per_channel,
    int8_asymmetric_per_tensor, int8_symmetric_per_channel, int8_symmetric_per_tensor,
    rewrite_to_qdq,
};
pub use runner::OnnxRunner;
pub use runner::dump_runner_profile;
#[cfg(feature = "gpu")]
pub use runner::gpu::profile_onnx_model_gpu;
#[cfg(feature = "gpu")]
pub use runner::gpu::run_onnx_model_gpu;
#[cfg(feature = "gpu")]
pub use runner::gpu::run_onnx_model_gpu_with;
#[cfg(feature = "gpu")]
pub use runner::gpu::{
    CompiledGpuPlan, compile_gpu_plan, run_compiled_gpu, run_compiled_gpu_fused,
    run_compiled_gpu_fused_timed,
};
#[cfg(feature = "gpu")]
pub use runner::gpu::{GpuExecAction, GpuExecPlan, plan_gpu_execution};
#[cfg(feature = "gpu")]
pub use runner::gpu::{GpuWeightCache, run_onnx_model_gpu_cached};
#[cfg(feature = "gpu")]
pub use runner::gpu::{compile_gpu_plan_f16, run_compiled_gpu_f16_fused};
pub use runner::kv_cache::{KvCache, KvDtype};
pub use runner::profile_onnx_model_cpu;
pub use runner::run_onnx_model;
pub use runner::run_onnx_model_borrowed;
pub use runner::run_onnx_model_borrowed_slice;

#[cfg(feature = "metal-backend")]
pub use runner::metal_runner::{
    InferenceHandle, MpsGraphPlan, compile_mpsgraph_plan, run_mpsgraph_plan, submit_mpsgraph_plan,
    wait_mpsgraph_plan,
};
#[cfg(feature = "metal-backend")]
pub use runner::metal_runner::{MetalPlan, compile_metal_plan, run_metal_plan};

#[cfg(test)]
mod tests;

#[cfg(test)]
mod proptest_tests;
