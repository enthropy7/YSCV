//! Metal-native ONNX inference — bypasses wgpu/naga for ~2x faster GPU compute.
//! Uses native MSL shaders compiled directly by Apple's Metal compiler.

#![allow(unused_variables)]

mod compile;
mod dispatch;
pub mod graph;
mod record;
mod run;
mod types;

// Re-export everything so callers don't change
pub use compile::compile_metal_plan;
pub use graph::{
    InferenceHandle, MpsGraphPlan, compile_mpsgraph_plan, run_mpsgraph_plan, submit_mpsgraph_plan,
    wait_mpsgraph_plan,
};
pub use run::run_metal_plan;
pub use types::MetalPlan;
