//! Direct Metal backend for GPU inference on Apple Silicon.
//! Bypasses wgpu/naga for ~2x faster compute shader execution.
//!
//! # Safety contract for unsafe code in this module
//!
//! All unsafe blocks fall into these categories:
//!
//! 1. **`from_raw_parts` on Metal buffer contents** — `Buffer::contents()` returns a
//!    raw pointer to GPU-shared memory. Safety: buffer length is validated via
//!    `debug_assert!` before each access; the buffer is `StorageModeShared` so the
//!    pointer is CPU-accessible for the buffer's lifetime.
//!
//! 2. **`copy_nonoverlapping` to Metal buffers** — used for f32/f16 upload.
//!    Safety: destination length checked via `debug_assert!`; source slice is valid.
//!
//! 3. **Inline assembly (NEON `fcvtn` / `st3`)** — f32-to-f16 conversion + NHWC
//!    interleave. Safety: `cfg(target_arch = "aarch64")` guarantees ISA availability;
//!    pointer offsets bounded by `batch * c * spatial` validated at entry.
//!
//! 4. **Objective-C `msg_send!` calls** — MPS/Metal framework interop. Safety: class
//!    existence is checked via `Class::get(...).ok_or_else()`; all ObjC objects are
//!    released after use; `autoreleasepool` drains factory objects.
//!
//! 5. **`std::mem::transmute` for COM-style vtable dispatch** — Metal encoder helper
//!    functions. Safety: vtable layout matches Apple's documented Metal API ABI.
//!
//! # Module layout
//!
//! - [`metal_conv`] — per-op Metal encoder dispatch (conv/GEMM/activations), with
//!   Winograd + MPS matmul fast paths
//! - [`mpsgraph`] — whole-model MetalPerformanceShadersGraph backend (single-dispatch
//!   graph compilation — fastest path on Apple Silicon)

#[cfg(all(target_os = "macos", feature = "metal-backend"))]
#[allow(unexpected_cfgs)]
pub mod metal_conv;

#[cfg(all(target_os = "macos", feature = "metal-backend"))]
#[allow(unexpected_cfgs)]
pub mod mpsgraph;
