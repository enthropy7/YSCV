//! Records which compute sub-path a dense NHWC Conv kernel took.
//!
//! A single `yscv-kernels` conv entry point (`conv2d_nhwc_padded`, the
//! prepacked body) is itself a dispatcher: depending on shape it routes to a
//! first-layer RGB microkernel, Winograd, a direct 3×3 kernel, im2col+GEMM,
//! pointwise GEMM, or the portable row fallback. The ONNX profiler records the
//! runner-level choice (which entry point) on its own; this slot exposes the
//! *kernel-internal* choice on top of it, so a `via nhwc-padded` row can be
//! refined to e.g. `via nhwc-padded/first-layer-rgb`.
//!
//! The slot is a single thread-local cell consumed once. The conv dispatch
//! runs synchronously on the caller's thread (only the inner compute loops fan
//! out), so the caller — `yscv-onnx`'s conv dispatch — reads back the sub-path
//! of the kernel it just invoked; consuming on read means an uninstrumented
//! path reports as unknown rather than leaking a stale label.

use std::cell::Cell;

/// The compute sub-path a dense Conv kernel took inside `yscv-kernels`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConvKernelPath {
    /// Dedicated first-layer 3×3 stride-2 RGB (`C_in == 3`) microkernel.
    FirstLayerRgb3x3,
    /// Winograd F(2×2, 3×3) for 3×3 stride-1 with enough spatial output.
    Winograd3x3,
    /// 1×1 pointwise via the K=16,N=16 direct microkernel.
    Pointwise16x16Direct,
    /// 1×1 pointwise via the N%16 direct microkernel.
    PointwiseNx16Direct,
    /// 1×1 pointwise via the general fused GEMM.
    PointwiseGemm,
    /// Direct 3×3 SIMD microkernel for small inputs (no im2col).
    Direct3x3,
    /// im2col + (BLAS or portable) SGEMM.
    Im2colGemm,
    /// Portable per-row FMA fallback.
    RowFma,
    /// Depthwise 3×3, channels=16, depth_multiplier=1 AVX-512 fast path.
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    DwC16Avx512,
    /// Depthwise per-row AVX-512 kernel (depth_multiplier=1).
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    DwAvx512,
    /// Depthwise per-row AVX+FMA kernel (depth_multiplier=1).
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    DwAvxFma,
    /// Depthwise per-row AVX kernel (depth_multiplier=1).
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    DwAvx,
    /// Depthwise per-row SSE kernel (depth_multiplier=1).
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    DwSse,
    /// Depthwise per-row NEON kernel (depth_multiplier=1).
    #[cfg(target_arch = "aarch64")]
    DwNeon,
    /// Depthwise scalar fallback (depth_multiplier>1, tiny channels, or no SIMD).
    DwScalar,
}

impl ConvKernelPath {
    /// Stable short label for profiler output.
    pub fn label(self) -> &'static str {
        match self {
            ConvKernelPath::FirstLayerRgb3x3 => "first-layer-rgb",
            ConvKernelPath::Winograd3x3 => "winograd-3x3",
            ConvKernelPath::Pointwise16x16Direct => "pw-16x16-direct",
            ConvKernelPath::PointwiseNx16Direct => "pw-nx16-direct",
            ConvKernelPath::PointwiseGemm => "pw-gemm",
            ConvKernelPath::Direct3x3 => "direct-3x3",
            ConvKernelPath::Im2colGemm => "im2col-gemm",
            ConvKernelPath::RowFma => "row-fma",
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            ConvKernelPath::DwC16Avx512 => "dw-c16-avx512",
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            ConvKernelPath::DwAvx512 => "dw-avx512",
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            ConvKernelPath::DwAvxFma => "dw-avx-fma",
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            ConvKernelPath::DwAvx => "dw-avx",
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            ConvKernelPath::DwSse => "dw-sse",
            #[cfg(target_arch = "aarch64")]
            ConvKernelPath::DwNeon => "dw-neon",
            ConvKernelPath::DwScalar => "dw-scalar",
        }
    }
}

thread_local! {
    /// Last dense-conv sub-path on this thread, consumed once by the caller.
    static LAST_CONV_PATH: Cell<Option<ConvKernelPath>> = const { Cell::new(None) };
}

/// Record the sub-path the current dense conv kernel took. Called from each
/// dispatch leaf inside the conv entry points.
#[inline]
pub(crate) fn note_conv_path(path: ConvKernelPath) {
    LAST_CONV_PATH.with(|slot| slot.set(Some(path)));
}

/// Take the last recorded dense-conv sub-path on this thread, clearing the
/// slot. Returns `None` when the most recent conv took an uninstrumented path
/// (e.g. depthwise, grouped, or the indirect kernel).
pub fn take_conv_path() -> Option<ConvKernelPath> {
    LAST_CONV_PATH.with(Cell::take)
}
