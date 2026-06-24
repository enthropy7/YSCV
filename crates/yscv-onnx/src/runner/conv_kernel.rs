//! Records which compute path each Conv node took, for the CPU profiler.
//!
//! A single ONNX `Conv` op resolves to one of several `yscv-kernels` entry
//! points depending on group/shape/stride/padding (indirect 3×3, blocked
//! GEMM, depthwise, grouped, BNNS …). That choice is invisible from the op
//! type alone, so the profiler can't tell a bandwidth-bound pointwise GEMM
//! from a depthwise pass. Each dispatch leaf in [`super::conv`] records the
//! path it took here; [`super::profile_onnx_model_cpu`] consumes it right
//! after running the node.
//!
//! Granularity is *runner-level*: this names the `yscv-kernels` entry point
//! the runner selected, not any sub-path chosen inside the kernel itself
//! (e.g. the first-layer RGB microkernel living inside `conv2d_nhwc_padded`).
//!
//! The slot is a single thread-local cell, consumed once. The CPU profiler
//! runs nodes sequentially on its own thread, so it always reads back the
//! dispatch from the Conv it just executed; consuming on read means an
//! unrecorded path reports as unknown rather than leaking a stale label.

use std::cell::Cell;

/// The compute path a single Conv dispatch took.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ConvKernel {
    /// aarch64 indirect convolution, 3×3 non-depthwise.
    #[cfg(target_arch = "aarch64")]
    IndirectNhwc3x3,
    /// group=1 with explicit padding (`conv2d_nhwc_padded`).
    NhwcPadded,
    /// group=1, no padding, load-time pre-packed weight.
    NhwcGemmPrepacked,
    /// group=1, no padding, blocked GEMM.
    NhwcGemm,
    /// Depthwise 3×3 stride-1 SAME via the NCHWc kernel (`YSCV_NCHWC_DW`).
    DepthwiseNchwc3x3,
    /// Depthwise with explicit padding.
    DepthwiseNhwcPadded,
    /// Depthwise without padding.
    DepthwiseNhwc,
    /// General grouped convolution (1 < group < channels).
    Grouped,
    /// Apple BNNS NCHW fast path (`YSCV_BNNS`).
    #[cfg(all(target_os = "macos", feature = "blas"))]
    BnnsNchw,
}

impl ConvKernel {
    /// Stable short label for profiler output.
    pub(crate) fn label(self) -> &'static str {
        match self {
            #[cfg(target_arch = "aarch64")]
            ConvKernel::IndirectNhwc3x3 => "indirect-nhwc-3x3",
            ConvKernel::NhwcPadded => "nhwc-padded",
            ConvKernel::NhwcGemmPrepacked => "nhwc-gemm-prepacked",
            ConvKernel::NhwcGemm => "nhwc-gemm",
            ConvKernel::DepthwiseNchwc3x3 => "dw-nchwc-3x3",
            ConvKernel::DepthwiseNhwcPadded => "dw-nhwc-padded",
            ConvKernel::DepthwiseNhwc => "dw-nhwc",
            ConvKernel::Grouped => "grouped",
            #[cfg(all(target_os = "macos", feature = "blas"))]
            ConvKernel::BnnsNchw => "bnns-nchw",
        }
    }
}

/// Runner-level Conv dispatch paired with the kernel-internal sub-path, when
/// the `yscv-kernels` entry point recorded one. Rendered as `runner` (e.g.
/// `dw-nhwc-padded`) or `runner/sub` (e.g. `nhwc-padded/first-layer-rgb`).
#[derive(Clone, Copy, Debug)]
pub(crate) struct ConvDispatch {
    pub(crate) kernel: ConvKernel,
    pub(crate) sub: Option<yscv_kernels::ConvKernelPath>,
}

impl ConvDispatch {
    pub(crate) fn label(self) -> String {
        match self.sub {
            Some(sub) => format!("{}/{}", self.kernel.label(), sub.label()),
            None => self.kernel.label().to_string(),
        }
    }
}

thread_local! {
    /// Last Conv dispatch on this thread, consumed once by the profiler.
    static LAST_CONV_KERNEL: Cell<Option<ConvDispatch>> = const { Cell::new(None) };
}

/// Record the compute path the current Conv dispatch took. Called from every
/// leaf of the Conv dispatch, right after the kernel returns; it pairs the
/// runner-level `kernel` with whatever kernel-internal sub-path the
/// `yscv-kernels` entry point recorded (consuming it). The profiler reads the
/// pair immediately after the node runs, so only the most recent matters.
#[inline]
pub(crate) fn note_conv_kernel(kernel: ConvKernel) {
    let sub = yscv_kernels::take_conv_path();
    LAST_CONV_KERNEL.with(|slot| slot.set(Some(ConvDispatch { kernel, sub })));
}

/// Take the last recorded Conv dispatch on this thread, clearing the slot.
/// Returns `None` when the most recent Conv took an unrecorded path.
pub(crate) fn take_conv_kernel() -> Option<ConvDispatch> {
    LAST_CONV_KERNEL.with(Cell::take)
}
