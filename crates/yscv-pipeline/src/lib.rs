//! TOML-driven pipeline configuration with explicit multi-accelerator
//! dispatch. Designed for AI-FPV drone pipelines where the user owns the
//! mapping of inference tasks to accelerators (RKNN cores, GPU, CPU) and
//! the framework just executes that mapping deterministically.
//!
//! # Design philosophy
//!
//! - **Explicit, not magic.** The user names which accelerator each model
//!   runs on. No auto-tier-detection, no silent CPU fallback. If a TOML
//!   says "RKNN core 1" and `librknnrt.so` is unavailable, startup fails
//!   loud with a config error.
//!
//! - **One binary, many boards.** Build with `--features "rknn gpu
//!   metal-backend"`. At runtime, pick the right `boards/<name>.toml` and
//!   run.
//!
//! - **Topological scheduling.** Tasks declare input bindings (camera /
//!   another task's output). The scheduler builds a DAG, runs independent
//!   tasks in parallel, blocks dependents until inputs are ready.
//!
//! # Example
//!
//! ```no_run
//! use yscv_pipeline::{PipelineConfig, run_pipeline};
//!
//! let cfg = PipelineConfig::from_toml_path("boards/rock4d.toml")?;
//! // `validate()` runs inside `run_pipeline` too; explicit for clarity.
//! cfg.validate()?;
//! let handle = run_pipeline(cfg)?;
//!
//! // Hot loop: feed camera bytes, get back every task's outputs.
//! let camera_bytes: &[u8] = &[/* NV12 / RGB frame bytes */];
//! let outputs = handle.dispatch_frame(&[("images", camera_bytes)])?;
//! for (key, bytes) in outputs {
//!     // `key` is "<task>.<output>" for task outputs or
//!     // "camera.<name>" / "camera" for the ingress echo.
//!     println!("{key}: {} bytes", bytes.len());
//! }
//! # Ok::<(), yscv_pipeline::Error>(())
//! ```
//!
//! See `examples/src/edge_pipeline_v2.rs` for the full runtime
//! including latency reporting and recovery on transient faults.
//!
//! # Features
//!
//! - `rknn` — `RknnDispatcher` (Rockchip NPU via `RknnPipelinedPool`,
//!   with NPU-hang auto-recovery and on-startup ONNX→RKNN compile +
//!   cache when `model_path` ends in `.onnx`).
//! - `rknn-validate` — dry-run `.rknn` load inside `validate_models`.
//! - `metal-backend` — `MetalDispatcher` for Apple Silicon
//!   (lazily-compiled `MpsGraphPlan` behind a `Mutex`).
//! - `gpu` — `GpuDispatcher` for cross-platform wgpu (Vulkan / Metal /
//!   DX12) via `yscv_onnx::run_onnx_model_gpu`.
//! - `realtime` — wires SCHED_FIFO / affinity / mlockall from
//!   `[realtime]` in the TOML, plus
//!   `PipelineHandle::spawn_watchdog(stats, interval)` which polls
//!   `PipelineStats5::watchdog_alarm` and auto-invokes `recover_all`
//!   on overrun (adds `yscv-video` as a dep).

#![deny(unsafe_code)]

mod accelerator;
mod config;
mod dispatch;
mod error;
mod scheduler;

pub use accelerator::{Accelerator, AcceleratorAvailability, NpuCoreSpec, probe_accelerators};
pub use config::{
    CameraSpec, EncoderSpec, InferenceTask, OsdSpec, OutputSpec, PipelineConfig, RtSpec,
    TensorBinding,
};
pub use dispatch::{AcceleratorDispatcher, dispatcher_for};
pub use error::{ConfigError, Error};
#[cfg(feature = "realtime")]
pub use scheduler::Watchdog;
pub use scheduler::{PipelineHandle, TaskScheduler, run_pipeline};
