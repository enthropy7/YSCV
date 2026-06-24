//! Records which microkernel family a fused matmul dispatch took.
//!
//! `matmul_2d_slices_fused_maybe_packed` (the entry the ONNX `MatMul` op and
//! pointwise Conv both use) routes to one of several GEMM families depending on
//! shape, CPU features, and kill-switches: external BLAS, the AVX-512 MR=12,
//! AVX2 MR=6, NEON MR=8, MR=4 blocked kernels, the low-k tile, or the portable
//! per-row fallback. That choice is invisible from the op type, so the profiler
//! can't tell a packed blocked GEMM from the per-row path. The dispatch records
//! the family it took here; the CPU profiler reads it back per `MatMul` node.
//!
//! Like the conv recorder, the dispatch runs synchronously on the caller's
//! thread (only the inner GEMM loops fan out), so the value is observed on the
//! same thread; consuming on read means an uninstrumented path reports as
//! unknown rather than leaking a stale label.

use std::cell::Cell;

/// The GEMM microkernel family a fused matmul dispatch selected.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MatmulKernel {
    /// External BLAS `sgemm`.
    #[cfg(feature = "blas")]
    BlasSgemm,
    /// NEON MR=8 / 8×12 blocked kernel.
    #[cfg(target_arch = "aarch64")]
    BlockedMr8,
    /// AVX2 MR=6×16 blocked kernel.
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    BlockedMr6,
    /// AVX-512 MR=12×NR=32 blocked kernel.
    #[cfg(all(target_arch = "x86_64", any(target_os = "linux", target_os = "macos")))]
    BlockedMr12,
    /// AVX+FMA low-k 4×24 tile (k ∈ {16, 24}).
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    LowKTile,
    /// MR=4 blocked GEMM (the portable blocked path).
    BlockedMr4,
    /// Per-row GEMM fallback.
    RowGemm,
}

impl MatmulKernel {
    /// Stable short label for profiler output.
    pub fn label(self) -> &'static str {
        match self {
            #[cfg(feature = "blas")]
            MatmulKernel::BlasSgemm => "blas-sgemm",
            #[cfg(target_arch = "aarch64")]
            MatmulKernel::BlockedMr8 => "blocked-mr8",
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            MatmulKernel::BlockedMr6 => "blocked-mr6",
            #[cfg(all(target_arch = "x86_64", any(target_os = "linux", target_os = "macos")))]
            MatmulKernel::BlockedMr12 => "blocked-mr12",
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            MatmulKernel::LowKTile => "low-k-tile",
            MatmulKernel::BlockedMr4 => "blocked-mr4",
            MatmulKernel::RowGemm => "row-gemm",
        }
    }
}

thread_local! {
    /// Last fused-matmul family on this thread, consumed once by the profiler.
    static LAST_MATMUL_KERNEL: Cell<Option<MatmulKernel>> = const { Cell::new(None) };
}

/// Record the GEMM family the current fused matmul dispatch took. Called from
/// each dispatch leaf of `matmul_2d_slices_fused_maybe_packed`.
#[inline]
pub(crate) fn note_matmul_kernel(kernel: MatmulKernel) {
    LAST_MATMUL_KERNEL.with(|slot| slot.set(Some(kernel)));
}

/// Take the last recorded fused-matmul family on this thread, clearing the
/// slot. Returns `None` when the most recent matmul took an uninstrumented
/// entry point (e.g. the `Gemm`-op dispatcher tree).
pub fn take_matmul_kernel() -> Option<MatmulKernel> {
    LAST_MATMUL_KERNEL.with(Cell::take)
}
