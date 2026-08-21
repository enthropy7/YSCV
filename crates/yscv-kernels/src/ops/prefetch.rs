//! L1 prefetch hint shared by the kernels that walk a strided operand.
//!
//! A kernel that reads its weight or `B` operand down a column at stride
//! `n * 4` bytes leaves the range a hardware stride detector tracks once `n`
//! grows. On an in-order core the resulting miss stalls the FMA pipe outright,
//! so a hint issued several K-iterations ahead covers the latency.
//!
//! Two rules the measured results here depend on:
//!
//! * Hoist the decision out of the inner loop. Split the K-loop into a
//!   prefetching range and a plain tail rather than testing a flag per
//!   iteration — on the Cortex-A53 the per-iteration branch alone costs more
//!   than the hint saves.
//! * Only add a hint where the operand stride actually defeats the hardware
//!   prefetcher. Unit-stride streaming loops do not need one, and out-of-order
//!   cores cover strided GEMM operands on their own; both cases measure as
//!   neutral and only add uops.

/// Number of K-iterations to run ahead of the current one. Four covers the
/// L2 hit latency on both Zen 4 and Cortex-A53 without pushing the hinted line
/// out of L1 before the loop reaches it.
pub(crate) const PREFETCH_AHEAD: usize = 4;

/// Issue an L1 "keep" prefetch hint for `p`.
///
/// `p` is a hint only and is never dereferenced, so it may point at a line the
/// caller will not read. It must still be a pointer the caller formed without
/// running off the end of the allocation, since computing an out-of-bounds
/// pointer is already undefined behaviour regardless of this call.
#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "arm"
))]
#[inline(always)]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
pub(crate) unsafe fn prefetch_l1_keep(p: *const f32) {
    #[cfg(target_arch = "x86")]
    std::arch::x86::_mm_prefetch::<{ std::arch::x86::_MM_HINT_T0 }>(p as *const i8);
    #[cfg(target_arch = "x86_64")]
    std::arch::x86_64::_mm_prefetch::<{ std::arch::x86_64::_MM_HINT_T0 }>(p as *const i8);
    #[cfg(target_arch = "aarch64")]
    core::arch::asm!(
        "prfm pldl1keep, [{p}]",
        p = in(reg) p,
        options(nostack, preserves_flags, readonly),
    );
    // armv7 spells the same hint `pld`, and has no cache-level selector.
    #[cfg(target_arch = "arm")]
    core::arch::asm!(
        "pld [{p}]",
        p = in(reg) p,
        options(nostack, preserves_flags, readonly),
    );
}

/// Split point for a K-loop that prefetches `PREFETCH_AHEAD` iterations ahead.
///
/// Returns the iteration count to run with hints; the caller runs `sp..k`
/// plain. When `enabled` is false this is `0`, so the prefetching loop is
/// skipped entirely and the plain loop stays branch-free — that is the whole
/// point of returning a bound instead of a flag to test per iteration.
#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "arm"
))]
#[inline(always)]
pub(crate) fn prefetch_split(enabled: bool, k: usize) -> usize {
    if enabled {
        k.saturating_sub(PREFETCH_AHEAD)
    } else {
        0
    }
}

/// Kill switch for the strided-operand prefetch hints: `YSCV_MATMUL_PREFETCH_OFF`.
///
/// Presence-checked and cached. The hoisted split leaves the disabled path
/// branch-free, so an A/B costs one predictable load per kernel call rather
/// than a branch per K-iteration.
#[cfg(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "arm"
))]
pub(crate) fn matmul_prefetch_disabled() -> bool {
    static CACHED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| std::env::var_os("YSCV_MATMUL_PREFETCH_OFF").is_some())
}
