//! Rockchip NPU (RKNN) backend — full SDK 2.4.3a0 coverage.
//!
//! Provides a safe Rust wrapper around the RKNN C runtime, loaded dynamically
//! via `dlopen` so the crate compiles on any platform. Actual NPU inference
//! requires `librknnrt.so` present on the target Rockchip device.
//!
//! # Capabilities
//!
//! - Full runtime API: init / destroy / query / inputs / outputs / run
//! - Context duplication for multi-stream execution
//! - NPU core affinity (RK3588 has 3 cores; API supports pinning and pools)
//! - Zero-copy memory: allocate / wrap DMA-BUF fd / wrap physical address / SRAM
//! - External weight / internal scratch storage
//! - Cache synchronisation (to/from device, bidirectional)
//! - Async inference with frame IDs and non-blocking completion polling
//! - Dynamic input shapes (switch resolution at runtime without re-init)
//! - Performance profiling (per-op timing, memory footprint, SDK version)
//! - Dedicated matmul accelerator (int4/int8/fp16 GEMM independent of conv NPU)
//! - Custom OpenCL operator registration (kernel source + callbacks)
//! - On-device ONNX→RKNN compiler via `librknn_api.so`
//!
//! # Module layout
//!
//! - [`consts`] — SDK constants + public enums (`RknnTensorType`, `NpuCoreMask`, …)
//! - [`ffi`] — `#[repr(C)]` structs + function pointer types + `dlopen` loader
//! - [`backend`] — safe public API (`RknnBackend`, `RknnMem`, `ContextPool`,
//!   `RknnMatmul`, `CustomOp`, `AsyncFrame`, runtime detection)
//! - [`compile`] — on-device ONNX → RKNN compiler (`librknn_api.so`)
//!
//! # Safety contract
//!
//! All unsafe blocks fall into these categories:
//!
//! 1. **FFI calls to `librknnrt.so` / `librknn_api.so`** — function pointers
//!    obtained via `dlsym` are validated non-null before use. The RKNN API
//!    is a stable C ABI provided by Rockchip; struct layouts match the
//!    official SDK headers (verified by compile-time size/alignment
//!    assertions in the `tests` module).
//!
//! 2. **`dlopen` / `dlsym` / `dlclose`** — standard POSIX dynamic linking.
//!    Library handles are checked for null; symbol pointers are validated
//!    before transmute to typed function pointers.
//!
//! 3. **`std::slice::from_raw_parts`** on RKNN output buffers — the runtime
//!    allocates and owns the buffer; `size` field gives the byte length.
//!    Pointer is checked non-null; slice lifetime is bounded by the
//!    `rknn_outputs_release` call that follows the copy.
//!
//! 4. **Raw pointer field access** on `RknnTensorMem` — allocated/owned by
//!    the RKNN runtime via `rknn_create_mem*` and freed via
//!    `rknn_destroy_mem` in `Drop`. Pointer validity is guaranteed between
//!    creation and destruction.

pub mod backend;
pub mod compile;
pub mod consts;
pub mod custom_op;
pub mod ffi;
pub mod pipeline;

pub use backend::{
    AsyncFrame, ContextPool, CustomOp, CustomOpRegistration, CustomOpTarget, InferenceBackend,
    MatmulQuantParams, MemSize, OpPerf, PerfDetail, RknnBackend, RknnMatmul, RknnMem,
    detect_backend, rknn_available,
};
pub use compile::{
    RknnCompileConfig, compile_onnx_to_rknn, load_onnx_as_rknn, rknn_compiler_available,
};
pub use consts::{
    MemAllocFlags,
    MemSyncMode,
    NpuCoreMask,
    // init flag constants
    RKNN_FLAG_ASYNC_MASK,
    RKNN_FLAG_COLLECT_MODEL_INFO_ONLY,
    RKNN_FLAG_COLLECT_PERF_MASK,
    RKNN_FLAG_DISABLE_FLUSH_INPUT_MEM_CACHE,
    RKNN_FLAG_DISABLE_FLUSH_OUTPUT_MEM_CACHE,
    RKNN_FLAG_DISABLE_PROC_HIGH_PRIORITY,
    RKNN_FLAG_ENABLE_SRAM,
    RKNN_FLAG_EXECUTE_FALLBACK_PRIOR_DEVICE_GPU,
    RKNN_FLAG_FENCE_IN_OUTSIDE,
    RKNN_FLAG_FENCE_OUT_OUTSIDE,
    RKNN_FLAG_INTERNAL_ALLOC_OUTSIDE,
    RKNN_FLAG_MEM_ALLOC_OUTSIDE,
    RKNN_FLAG_MODEL_BUFFER_ZERO_COPY,
    RKNN_FLAG_PRIOR_HIGH,
    RKNN_FLAG_PRIOR_LOW,
    RKNN_FLAG_PRIOR_MEDIUM,
    RKNN_FLAG_SHARE_SRAM,
    RKNN_FLAG_SHARE_WEIGHT_MEM,
    RKNN_MEM_FLAG_ALLOC_NO_CONTEXT,
    RknnMatmulLayout,
    RknnMatmulQuantType,
    RknnMatmulType,
    RknnQuantType,
    RknnTensorFormat,
    RknnTensorType,
};
pub use custom_op::{
    CustomOpAttr, CustomOpContext, CustomOpHandler, CustomOpTensor, MAX_CUSTOM_OP_SLOTS,
};
pub use ffi::{RknnMatmulIoAttr, RknnMatmulShape, RknnMatmulTensorAttr, RknnTensorAttr};
pub use pipeline::{RknnInferenceHandle, RknnPipelinedPool};
