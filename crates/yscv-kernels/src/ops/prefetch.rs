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

/// Issue an L1 "keep" prefetch hint for `p`.
///
/// `p` is a hint only and is never dereferenced, so it may point at a line the
/// caller will not read. It must still be a pointer the caller formed without
/// running off the end of the allocation, since computing an out-of-bounds
/// pointer is already undefined behaviour regardless of this call.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
pub(crate) unsafe fn prefetch_l1_keep(p: *const f32) {
    core::arch::asm!(
        "prfm pldl1keep, [{p}]",
        p = in(reg) p,
        options(nostack, preserves_flags, readonly),
    );
}
