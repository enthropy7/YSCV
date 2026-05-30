//! Hand-written SGEMM/conv microkernels assembled from src/asm/*.S,
//! exposed as `extern "C"` declarations. See build.rs for the assembly step.

// ---------------------------------------------------------------------------
// Hand-tuned SGEMM microkernel (x86_64 AVX+FMA) — external assembly
// ---------------------------------------------------------------------------
//
// Source: `src/asm/x86_64_sysv.S` (Linux/macOS) or `src/asm/x86_64_win64.S`
// (Windows) — assembled by build.rs via the `cc` crate. Keeping the kernel
// in a real `.S` file (vs `global_asm!`) lets us use preprocessor macros for
// symbol decoration across ELF/Mach-O/COFF, and keeps the register schedule
// free of Rust's inline-asm syntax restrictions.
//
// 4×8 microkernel with separate SET / ACCUMULATE entry points. k-loop is
// 2-way unrolled and software-pipelines B[k+1] to hide load latency behind
// the FMA chain on B[k].
//
// Calling convention (System V AMD64 / Win64 — shim in Win64.S translates):
//   rdi = a_panel (packed, MR=4 stride per k)
//   rsi = b_panel (packed, NR=8 stride per k)
//   rdx = c (output pointer)
//   rcx = ldc (row stride in FLOATS, not bytes)
//   r8  = kc (k dimension)

// The 4×8 hand-tuned kernel has two ABI variants, selected by build.rs:
//   SysV AMD64 — Linux, macOS      (`src/asm/x86_64_sysv.S`)
//   Win64      — Windows GNU ABI   (`src/asm/x86_64_win64.S`)
// Both export the same two symbols, so the Rust call sites are identical.
//
// Windows-MSVC is excluded: `cl.exe` does not speak GAS, so build.rs skips
// the .S assembly step there. Without these `extern "C"` decls being cfg-gated
// to match, `link.exe` fails with LNK2019 on `yscv_sgemm_4x8_*` /
// `yscv_sgemm_4x24_avx2_*_fused`. The MSVC build falls through to the
// intrinsics 4×24 / 4×8 paths below.
#[cfg(all(
    target_arch = "x86_64",
    any(
        target_os = "linux",
        target_os = "macos",
        all(target_os = "windows", not(target_env = "msvc"))
    )
))]
#[allow(unsafe_code)]
pub(super) mod sgemm_asm {
    unsafe extern "C" {
        pub(crate) fn yscv_sgemm_4x8_set(
            a_panel: *const f32,
            b_panel: *const f32,
            c: *mut f32,
            ldc: usize,
            kc: usize,
        );
        pub(crate) fn yscv_sgemm_4x8_acc(
            a_panel: *const f32,
            b_panel: *const f32,
            c: *mut f32,
            ldc: usize,
            kc: usize,
        );
        // 4×24 AVX+FMA pure-.S with fused bias + residual + Relu epilogue.
        // Used on the hot pointwise Conv paths when activation is None/Relu and
        // the CPU has FMA+AVX; SiLU and the AVX-only (non-FMA) fallback go
        // through the `microkernel_4x24_avx_fma` intrinsics version instead.
        //
        // Args (SysV):
        //   rdi = a_panel, rsi/rdx/rcx = b_panel_{0,1,2},
        //   r8  = c, r9 = ldc (in floats),
        //   stack: kc, bias_ptr (nullable), residual_ptr (nullable; stride=ldc),
        //          activation (0/1), is_last_k (0/1).
        pub(crate) fn yscv_sgemm_4x24_avx2_set_fused(
            a_panel: *const f32,
            b_panel_0: *const f32,
            b_panel_1: *const f32,
            b_panel_2: *const f32,
            c: *mut f32,
            ldc: usize,
            kc: usize,
            bias: *const f32,
            residual: *const f32,
            activation: usize,
            is_last_k: usize,
        );
        pub(crate) fn yscv_sgemm_4x24_avx2_acc_fused(
            a_panel: *const f32,
            b_panel_0: *const f32,
            b_panel_1: *const f32,
            b_panel_2: *const f32,
            c: *mut f32,
            ldc: usize,
            kc: usize,
            bias: *const f32,
            residual: *const f32,
            activation: usize,
            is_last_k: usize,
        );
    }
} // mod sgemm_asm

// AVX-512 4×32 microkernel (SysV only — Win64 variant lives in Phase 6 if
// measurement justifies). Zen 4 double-pumps 512-bit FMAs so the throughput
// win over AVX2 4×24 is marginal; we keep the path gated on runtime detection
// and on the `YSCV_NO_AVX512` env override for A/B benchmarking.
#[cfg(all(target_arch = "x86_64", any(target_os = "linux", target_os = "macos")))]
#[allow(unsafe_code)]
pub(super) mod sgemm_asm_avx512 {
    unsafe extern "C" {
        pub(crate) fn yscv_sgemm_4x32_avx512_set(
            a_panel: *const f32,
            b_base: *const f32,
            c: *mut f32,
            ldc: usize,
            kc: usize,
            bias: *const f32,  // nullable
            activation: usize, // 0=None, 1=Relu (SiLU routes elsewhere)
            is_last_k: usize,  // 0 or 1
        );
        pub(crate) fn yscv_sgemm_4x32_avx512_acc(
            a_panel: *const f32,
            b_base: *const f32,
            c: *mut f32,
            ldc: usize,
            kc: usize,
            bias: *const f32,
            activation: usize,
            is_last_k: usize,
        );
        // MR=12 × NR=32 AVX-512 microkernel using the full 32-ZMM file:
        // 24 accumulators (12 rows × 2 ZMM) + 2 B + 1 A-broadcast + scratch.
        pub(crate) fn yscv_sgemm_12x32_avx512_set(
            a_panel: *const f32,
            b_panel: *const f32,
            c: *mut f32,
            ldc: usize,
            kc: usize,
            bias: *const f32,
            activation: usize,
            is_last_k: usize,
            residual: *const f32, // NULL = no residual; same row-stride as ldc
        );
        pub(crate) fn yscv_sgemm_12x32_avx512_acc(
            a_panel: *const f32,
            b_panel: *const f32,
            c: *mut f32,
            ldc: usize,
            kc: usize,
            bias: *const f32,
            activation: usize,
            is_last_k: usize,
            residual: *const f32, // NULL = no residual; same row-stride as ldc
        );
    }
}

// Hand-tuned aarch64 4×24 NEON microkernel (src/asm/aarch64.S). Covers the
// no-epilogue fast path (no bias, no activation); call sites that need
// bias+activation fall through to the `microkernel_4x24_neon` intrinsics
// version which applies the epilogue in-register.
#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code)]
pub(super) mod sgemm_asm_aarch64 {
    unsafe extern "C" {
        pub(crate) fn yscv_sgemm_4x24_neon_set(
            a_panel: *const f32,
            b_panel_0: *const f32,
            b_panel_1: *const f32,
            b_panel_2: *const f32,
            c: *mut f32,
            ldc: usize,
            kc: usize,
        );
        pub(crate) fn yscv_sgemm_4x24_neon_acc(
            a_panel: *const f32,
            b_panel_0: *const f32,
            b_panel_1: *const f32,
            b_panel_2: *const f32,
            c: *mut f32,
            ldc: usize,
            kc: usize,
        );
        // MR=8 × NR=12 with inline bias+Relu epilogue. Uses a dedicated
        // NR=12 packed-B layout (see `pack_b_panel_nr12` below); this is
        // NOT compatible with the generic NR=8 `pack_b_panel` used by the
        // MR=4 paths. 24 accumulators × 2× A-reuse = 2× throughput vs the
        // MR=4 intrinsic path at the cost of a second packer variant.
        pub(crate) fn yscv_sgemm_8x12_neon_set(
            a_panel: *const f32,
            b_panel: *const f32, // single NR=12 panel (kc rows × 12 f32)
            c: *mut f32,
            ldc: usize,
            kc: usize,
            bias: *const f32,  // nullable
            activation: usize, // 0=None, 1=Relu (SiLU routed to MR=4)
            is_last_k: usize,
        );
        pub(crate) fn yscv_sgemm_8x12_neon_acc(
            a_panel: *const f32,
            b_panel: *const f32,
            c: *mut f32,
            ldc: usize,
            kc: usize,
            bias: *const f32,
            activation: usize,
            is_last_k: usize,
        );
        // FEAT_FP16 half-precision 6×16 SGEMM kernel (fp16 in, fp16 out,
        // fp16 accumulate). 2× throughput vs fp32 NEON on Cortex-A76+/
        // Apple M1+/Neoverse-V1+. Pack layout: A is MR=6 × kc fp16 packed
        // as 8-lane v-regs (2 lanes unused). B is kc × NR=16 fp16.
        pub(crate) fn yscv_hgemm_6x16_neon_set(
            a_panel: *const u16,
            b_panel_0: *const u16,
            b_panel_1: *const u16,
            c: *mut u16,
            ldc: usize,
            kc: usize,
        );
        pub(crate) fn yscv_hgemm_6x16_neon_acc(
            a_panel: *const u16,
            b_panel_0: *const u16,
            b_panel_1: *const u16,
            c: *mut u16,
            ldc: usize,
            kc: usize,
        );
    }
} // mod sgemm_asm_aarch64
