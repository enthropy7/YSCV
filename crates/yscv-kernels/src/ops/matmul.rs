#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::{
    float32x4_t, vaddq_f32, vdivq_f32, vdupq_n_f32, vfmaq_f32, vld1q_f32, vmaxq_f32, vnegq_f32,
    vst1q_f32,
};
#[cfg(target_arch = "x86")]
use std::arch::x86::{
    __m128, __m256, _mm_add_ps, _mm_div_ps, _mm_loadu_ps, _mm_max_ps, _mm_mul_ps, _mm_set1_ps,
    _mm_setzero_ps, _mm_storeu_ps, _mm_sub_ps, _mm256_add_ps, _mm256_broadcast_ss, _mm256_div_ps,
    _mm256_fmadd_ps, _mm256_loadu_ps, _mm256_max_ps, _mm256_mul_ps, _mm256_set1_ps,
    _mm256_setzero_ps, _mm256_storeu_ps, _mm256_sub_ps,
};
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{
    __m128, __m256, _mm_add_ps, _mm_div_ps, _mm_loadu_ps, _mm_max_ps, _mm_mul_ps, _mm_set1_ps,
    _mm_setzero_ps, _mm_storeu_ps, _mm_sub_ps, _mm256_add_ps, _mm256_broadcast_ss, _mm256_div_ps,
    _mm256_fmadd_ps, _mm256_loadu_ps, _mm256_max_ps, _mm256_mul_ps, _mm256_set1_ps,
    _mm256_setzero_ps, _mm256_storeu_ps, _mm256_sub_ps,
};

use rayon::{ThreadPool, prelude::*};
use yscv_tensor::{AlignedVec, Tensor, TensorError};

use super::super::error::KernelError;
use super::config::{MatMulPlan, ParallelMatmulConfig, should_parallelize_len};
use super::conv::Activation;
use super::simd::matmul_row_set_dispatch;

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
mod sgemm_asm {
    unsafe extern "C" {
        pub(super) fn yscv_sgemm_4x8_set(
            a_panel: *const f32,
            b_panel: *const f32,
            c: *mut f32,
            ldc: usize,
            kc: usize,
        );
        pub(super) fn yscv_sgemm_4x8_acc(
            a_panel: *const f32,
            b_panel: *const f32,
            c: *mut f32,
            ldc: usize,
            kc: usize,
        );
        // 4×24 AVX+FMA pure-.S with fused bias + residual + Relu epilogue.
        // Phase B of the fp32 ORT-gap arc — replaces the intrinsics + inline
        // `asm!` `microkernel_4x24_avx_fma` on the hot pointwise Conv paths
        // whenever activation is None or Relu and CPU has FMA+AVX. SiLU and
        // the AVX-only (non-FMA) fallback continue through intrinsics.
        //
        // Args (SysV):
        //   rdi = a_panel, rsi/rdx/rcx = b_panel_{0,1,2},
        //   r8  = c, r9 = ldc (in floats),
        //   stack: kc, bias_ptr (nullable), residual_ptr (nullable; stride=ldc),
        //          activation (0/1), is_last_k (0/1).
        pub(super) fn yscv_sgemm_4x24_avx2_set_fused(
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
        pub(super) fn yscv_sgemm_4x24_avx2_acc_fused(
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
mod sgemm_asm_avx512 {
    unsafe extern "C" {
        pub(super) fn yscv_sgemm_4x32_avx512_set(
            a_panel: *const f32,
            b_base: *const f32,
            c: *mut f32,
            ldc: usize,
            kc: usize,
            bias: *const f32,  // nullable
            activation: usize, // 0=None, 1=Relu (SiLU routes elsewhere)
            is_last_k: usize,  // 0 or 1
        );
        pub(super) fn yscv_sgemm_4x32_avx512_acc(
            a_panel: *const f32,
            b_base: *const f32,
            c: *mut f32,
            ldc: usize,
            kc: usize,
            bias: *const f32,
            activation: usize,
            is_last_k: usize,
        );
        // Step A1.1: MR=12 × NR=32 AVX-512 microkernel. Uses full
        // 32-ZMM register file: 24 accumulators (12 rows × 2 ZMM) +
        // 2 B + 1 A-broadcast + 5 scratch. Target for Zen 4 AVX-512
        // front-end savings (same peak FLOPS as AVX2 4×24 but half the
        // uops per FMA pair).
        pub(super) fn yscv_sgemm_12x32_avx512_set(
            a_panel: *const f32,
            b_panel: *const f32,
            c: *mut f32,
            ldc: usize,
            kc: usize,
            bias: *const f32,
            activation: usize,
            is_last_k: usize,
        );
        pub(super) fn yscv_sgemm_12x32_avx512_acc(
            a_panel: *const f32,
            b_panel: *const f32,
            c: *mut f32,
            ldc: usize,
            kc: usize,
            bias: *const f32,
            activation: usize,
            is_last_k: usize,
        );
    }
}

// Hand-tuned aarch64 4×24 NEON microkernel (src/asm/aarch64.S). Covers the
// no-epilogue fast path (no bias, no activation); call sites that need
// bias+activation fall through to the `microkernel_4x24_neon` intrinsics
// version which applies the epilogue in-register.
#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code)]
mod sgemm_asm_aarch64 {
    unsafe extern "C" {
        pub(super) fn yscv_sgemm_4x24_neon_set(
            a_panel: *const f32,
            b_panel_0: *const f32,
            b_panel_1: *const f32,
            b_panel_2: *const f32,
            c: *mut f32,
            ldc: usize,
            kc: usize,
        );
        pub(super) fn yscv_sgemm_4x24_neon_acc(
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
        pub(super) fn yscv_sgemm_8x12_neon_set(
            a_panel: *const f32,
            b_panel: *const f32, // single NR=12 panel (kc rows × 12 f32)
            c: *mut f32,
            ldc: usize,
            kc: usize,
            bias: *const f32,  // nullable
            activation: usize, // 0=None, 1=Relu (SiLU routed to MR=4)
            is_last_k: usize,
        );
        pub(super) fn yscv_sgemm_8x12_neon_acc(
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
        pub(super) fn yscv_hgemm_6x16_neon_set(
            a_panel: *const u16,
            b_panel_0: *const u16,
            b_panel_1: *const u16,
            c: *mut u16,
            ldc: usize,
            kc: usize,
        );
        pub(super) fn yscv_hgemm_6x16_neon_acc(
            a_panel: *const u16,
            b_panel_0: *const u16,
            b_panel_1: *const u16,
            c: *mut u16,
            ldc: usize,
            kc: usize,
        );
    }
} // mod sgemm_asm_aarch64

// ---------------------------------------------------------------------------
// GEMM epilogue: fused bias + activation applied in the microkernel store phase
// ---------------------------------------------------------------------------

/// Describes fused post-GEMM operations applied in the microkernel store phase.
///
/// When the microkernel finishes its last k-block, the final result is still in
/// SIMD registers. Instead of storing to memory and re-reading for bias/activation,
/// the epilogue fuses bias addition and activation directly on those registers.
#[derive(Clone, Copy)]
pub struct GemmEpilogue {
    /// Pointer to the full bias vector (length = N output columns).
    /// The microkernel indexes `bias[col_offset..col_offset+NR]` for its tile.
    pub bias: Option<*const f32>,
    /// Activation to apply after bias addition.
    pub activation: Activation,
    /// Optional residual tensor added in-place before activation. When set,
    /// the epilogue computes `out = activation(acc + bias + residual)`,
    /// saving a separate `add_inplace` pass over the output buffer. Shape
    /// must match the output (`M * N` elements, row-major). Used by the
    /// runner's `Conv+Add+Relu` fusion for residual connections.
    pub residual: Option<*const f32>,
}

impl GemmEpilogue {
    /// Convenience: bias + activation with no residual (the common case).
    #[inline]
    pub fn new(bias: Option<*const f32>, activation: Activation) -> Self {
        Self {
            bias,
            activation,
            residual: None,
        }
    }
}

impl GemmEpilogue {
    pub(crate) const IDENTITY: Self = Self {
        bias: None,
        activation: Activation::None,
        residual: None,
    };
}

// SAFETY: bias points to a read-only shared buffer (conv weight bias).
// All threads read from disjoint column offsets of the same buffer.
#[allow(unsafe_code)]
unsafe impl Send for GemmEpilogue {}
#[allow(unsafe_code)]
unsafe impl Sync for GemmEpilogue {}

// ---------------------------------------------------------------------------
// Blocked matmul constants
// ---------------------------------------------------------------------------

/// Micro-kernel tile: 4 rows of A × 8 columns of B (default).
// WHY 4: saturates NEON/SSE register file without spilling (4 accumulator regs × NR columns).
const MR: usize = 4;
// WHY 8: 8 columns = 2 AVX registers or 2 NEON registers per row, fits L1 cache line (64 bytes).
const NR: usize = 8;
/// Step C1: MR=6 AVX2 microkernel tile — 6 rows × NR_MR6=16 columns.
/// Matches MLAS SgemmKernelFma3's 6×16 layout. 12 YMM accumulators
/// (6 rows × 2 NR panels) + 2 B-panels + 2 A-broadcast scratch =
/// 16 YMM exact fit. Reduces B-panel loads 55% vs MR=4 on m≥192 shapes.
/// Opt-in via `YSCV_MR6=1` during validation; enables by default once
/// sgemm A/B shows ≥10% pointwise win.
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
const MR6: usize = 6;
/// Cache blocking parameters.
///
/// KC controls the inner k-loop length per blocked pass. Empirically swept
/// on Zen 4 (Siamese tracker workload, 2026-04-19): KC=256 beats KC=128 by
/// 0.6–0.9% on 1T p50 despite L1D being 32 KB. The reason is our workload:
/// pointwise Conv with large IC (up to 672) means KC=256 captures K in
/// fewer outer passes, reducing pack_a overhead proportionally. MLAS picks
/// 128 — optimal for their shape mix, but ours rewards the larger pass.
/// Left as a single constant for all archs: ARM cores have even larger
/// L1D (Cortex-A76 64 KB, Apple M 192 KB), so 256 is comfortable there too.
// WHY 256: 256 floats × 4 bytes = 1 KB panel — fits in L1D on all targets.
const KC: usize = 256;
// WHY 128: 128 rows × KC columns = 128KB packed panel fits in L2 cache (typically 256KB-1MB).
const MC: usize = 128;
// WHY 256: balances work per thread with cache reuse across micro-kernel calls.
const NC: usize = 256;

/// MC used inside `blocked_gemm_parallel`: smaller than MC so that `m/MC_PARALLEL`
/// yields enough IC blocks to saturate 6–8 threads even at typical Conv M values.
/// Tracker's `M = 256` gives 2 blocks at MC=128 (starves 6 threads) but 8 blocks
/// at MC=32 (fully saturates). pack_a still reuses its buffer per-thread, and
/// gebp_kernel runs identically on smaller mc — cache reuse is traded for
/// better work distribution.
/// MC used inside `blocked_gemm_parallel` + at the dispatch threshold. Smaller
/// than MC pushes small-M shapes into the blocked path (saturating threads)
/// AND reduces each worker's mc step for finer parallelism within large-M.
const MC_PARALLEL: usize = 8;

const BLOCKED_THRESHOLD: usize = 32;

// ---------------------------------------------------------------------------
// Public API (unchanged signatures)
// ---------------------------------------------------------------------------

/// Rank-2 matrix multiplication with adaptive CPU row-level parallelization.
///
/// Uses sequential fallback for small matrices and for `miri`.
pub fn matmul_2d(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor, KernelError> {
    matmul_2d_with_config(lhs, rhs, ParallelMatmulConfig::default())
}

/// Zero-copy slice-based matmul: C[m×n] = A[m×k] × B[k×n].
/// Writes directly into `out` without any intermediate allocation.
#[allow(unsafe_code)]
pub fn matmul_2d_slices(a: &[f32], m: usize, k: usize, b: &[f32], n: usize, out: &mut [f32]) {
    debug_assert!(a.len() >= m * k);
    debug_assert!(b.len() >= k * n);
    debug_assert!(out.len() >= m * n);

    #[cfg(feature = "blas")]
    if use_blas() {
        blas_sgemm(a, b, out, m, k, n);
        return;
    }

    if use_blocked(m, k, n) {
        blocked_gemm_sequential(a, b, out, m, k, n, GemmEpilogue::IDENTITY, None);
    } else {
        row_gemm_set_sequential(a, b, out, m, k, n);
    }
}

/// Matmul with transposed left operand. `a_kt` is shape `[K, M]` in
/// row-major memory (i.e. NOT transposed — it's the original A before
/// the Transpose node we've fused into this call). Computes
/// `out[m, n] = sum_k a_kt[k, m] * b[k, n]` — equivalent to
/// `(a_kt^T) @ b` — without materialising the transpose.
///
/// When the `blas` feature is enabled this routes through BLAS's
/// `cblas_sgemm` with `TransA=CblasTrans`, which handles the
/// transposed-access pattern inside the library's tuned microkernels
/// (Accelerate/OpenBLAS/MKL all do this efficiently). The non-BLAS
/// fallback materialises the transpose into a stack scratch buffer
/// and calls the standard path — correct but less tuned; will pick
/// up a dedicated `transA` variant of `blocked_gemm_sequential` in a
/// follow-up if tracker shapes land there.
///
/// Used by the Transpose-perm[0,2,1] → MatMul fusion in the runner.
pub fn matmul_2d_slices_trans_a(
    a_kt: &[f32],
    m: usize,
    k: usize,
    b: &[f32],
    n: usize,
    out: &mut [f32],
) {
    debug_assert!(a_kt.len() >= k * m);
    debug_assert!(b.len() >= k * n);
    debug_assert!(out.len() >= m * n);

    #[cfg(feature = "blas")]
    if use_blas() {
        // SAFETY: bounds asserted above. `cblas_sgemm` is pure compute,
        // beta=0 so no aliasing concerns between output and inputs.
        use cblas_sys::{CBLAS_LAYOUT, CBLAS_TRANSPOSE, cblas_sgemm};
        #[allow(unsafe_code)]
        unsafe {
            cblas_sgemm(
                CBLAS_LAYOUT::CblasRowMajor,
                CBLAS_TRANSPOSE::CblasTrans,
                CBLAS_TRANSPOSE::CblasNoTrans,
                m as i32,
                n as i32,
                k as i32,
                1.0,
                a_kt.as_ptr(),
                // Physical leading dimension of A (row stride in row-major
                // memory). With layout (K, M), lda = M.
                m as i32,
                b.as_ptr(),
                n as i32,
                0.0,
                out.as_mut_ptr(),
                n as i32,
            );
        }
        return;
    }

    // Non-BLAS fallback: materialise the transposed view and call the
    // standard path. For a shape where BLAS would win we'd land here
    // only on builds with `--no-default-features` (no blas), which is
    // also the build tracker benchmarks exercise on NixOS — the
    // fallback is correct, just slower than it could be. TODO(r9): add
    // a `transA` variant of `blocked_gemm_sequential` that packs A
    // directly from the (K, M) layout to avoid this scratch pass.
    // zero-init + transpose in one pass. The zero-fill cost is a single
    // linear memset; caller is the BLAS-transA fallback path, not a hot
    // loop, so the extra write is negligible. Replaces a prior
    // `with_capacity + set_len` pattern flagged by `clippy::uninit_vec`.
    let mut a_transposed: Vec<f32> = vec![0.0; m * k];
    for mi in 0..m {
        for ki in 0..k {
            a_transposed[mi * k + ki] = a_kt[ki * m + mi];
        }
    }
    if use_blocked(m, k, n) {
        blocked_gemm_sequential(&a_transposed, b, out, m, k, n, GemmEpilogue::IDENTITY, None);
    } else {
        row_gemm_set_sequential(&a_transposed, b, out, m, k, n);
    }
}

/// Zero-copy slice-based matmul with fused bias+activation epilogue.
///
/// When `epilogue` is non-identity, bias and activation are applied directly
/// in the GEMM microkernel store phase — eliminating a separate read+write pass.
#[allow(unsafe_code)]
pub(super) fn matmul_2d_slices_fused(
    a: &[f32],
    m: usize,
    k: usize,
    b: &[f32],
    n: usize,
    out: &mut [f32],
    epilogue: GemmEpilogue,
    config: ParallelMatmulConfig,
    thread_pool: Option<&ThreadPool>,
) {
    matmul_2d_slices_fused_maybe_packed(a, m, k, b, n, out, None, epilogue, config, thread_pool);
}

/// Like `matmul_2d_slices_fused` but may accept a pre-packed B (e.g. built at
/// model load time). When provided, the blocked GEMM path skips `pack_b_panel`
/// entirely and streams directly from the shared packed storage — eliminates
/// B-packing from the hot path for static Conv kernel weights. Public so
/// higher-level crates (ONNX loader) can route the pre-packed handle in.
pub fn matmul_2d_slices_fused_maybe_packed(
    a: &[f32],
    m: usize,
    k: usize,
    b: &[f32],
    n: usize,
    out: &mut [f32],
    packed_b: Option<&PackedB>,
    epilogue: GemmEpilogue,
    config: ParallelMatmulConfig,
    thread_pool: Option<&ThreadPool>,
) {
    debug_assert!(a.len() >= m * k);
    debug_assert!(b.len() >= k * n || packed_b.is_some());
    debug_assert!(out.len() >= m * n);
    debug_assert!(packed_b.is_none_or(|p| p.matches(k, n)));

    #[cfg(feature = "blas")]
    if use_blas() {
        blas_sgemm(a, b, out, m, k, n);
        apply_epilogue_fallback(out, m, n, &epilogue);
        return;
    }

    let plan = MatMulPlan {
        m,
        k,
        n,
        output_len: m * n,
    };
    // aarch64 tracker hot shapes include many pointwise GEMMs with k=16/24.
    // The generic `use_blocked` gate requires k>=32, which pushes those
    // layers into row-GEMM despite large total work. On small ARM cores this
    // leaves throughput on the table versus blocked kernels. For static Conv
    // weights (prepacked B present), allow a low-k blocked route behind a
    // conservative work threshold + kill switch.
    #[cfg(target_arch = "aarch64")]
    let use_blocked_low_k = use_blocked_aarch64_low_k(m, k, n, packed_b.is_some());
    #[cfg(not(target_arch = "aarch64"))]
    let use_blocked_low_k = false;
    let use_blocked_path = use_blocked(m, k, n) || use_blocked_low_k;

    let has_residual = epilogue.residual.is_some();
    #[cfg(target_arch = "aarch64")]
    let residual_blocked_unsafe = has_residual
        && (!aarch64_residual_blocked_enabled() || blocked_residual_has_unsupported_tail(n));
    #[cfg(not(target_arch = "aarch64"))]
    let residual_blocked_unsafe = has_residual && blocked_residual_has_unsupported_tail(n);

    // aarch64 MR=8 / 8×12 NEON primary path. Inline bias+Relu epilogue in
    // the `.S` kernel. SiLU falls through to MR=4 intrinsic (which fuses
    // SiLU via silu_neon). Opt out: `YSCV_NO_MR8=1`. Gated on n >= 16 so
    // the 8×12 kernel always has room to read 2 NR=8 panels.
    //
    // Residual support for MR8 is currently EXPERIMENTAL and defaults OFF
    // (`YSCV_MR8_RESIDUAL=1` to enable): full 8×12 tiles run asm with bias,
    // then apply residual+activation in a compact post-pass at `is_last_k`.
    #[cfg(target_arch = "aarch64")]
    let allow_mr8_residual = !has_residual || mr8_residual_enabled();
    #[cfg(target_arch = "aarch64")]
    if mr8_enabled()
        && use_blocked_path
        && n >= 16
        && !matches!(epilogue.activation, Activation::Silu)
        && allow_mr8_residual
    {
        if should_parallelize(plan, config, thread_pool) {
            let nthreads = super::config::available_threads(thread_pool);
            let blocked_blocks = m.div_ceil(MC_PARALLEL_MR8);
            if blocked_blocks >= nthreads {
                blocked_gemm_parallel_mr8(a, b, out, m, k, n, epilogue, thread_pool, packed_b);
                return;
            }
        } else {
            blocked_gemm_sequential_mr8(a, b, out, m, k, n, epilogue, packed_b);
            return;
        }
    }

    // Step C1: MR=6×16 AVX2 fast path. On x86 with FMA+AVX, gated on
    // m ≥ MR6, n ≥ 2*NR (full-tile fit), AND n multiple of 2*NR so the
    // jr loop stays in full-tile mode. Scalar tail handlers now support
    // residual epilogue too, so no extra residual-based m-tail gate is
    // needed. SiLU falls through to MR=4 (activation fully supported in
    // 6×16 kernel, gated only to match MR=8 policy).
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    let n_fits_mr6_tiles = n.is_multiple_of(2 * NR);
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if mr6_enabled()
        && use_blocked(m, k, n)
        && m >= MR6
        && n >= 2 * NR
        && n_fits_mr6_tiles
        && !matches!(epilogue.activation, Activation::Silu)
    {
        if should_parallelize(plan, config, thread_pool) {
            let nthreads = super::config::available_threads(thread_pool);
            let blocked_blocks = m.div_ceil(MC_PARALLEL_MR6);
            if blocked_blocks >= nthreads {
                blocked_gemm_parallel_mr6(a, b, out, m, k, n, epilogue, thread_pool, packed_b);
                return;
            }
        } else {
            blocked_gemm_sequential_mr6(a, b, out, m, k, n, epilogue, packed_b);
            return;
        }
    }

    // Step A1.2: AVX-512 MR=12×NR=32 dispatch. Gated on AVX-512F + shape
    // alignment (m%12==0, n%32==0) + no residual + non-SiLU + env
    // `YSCV_AVX512_SGEMM=1`. Session 8 attempted m-tail handling via
    // recursion, regressed +1248 µs @ 6T due to duplicate pack-B cost
    // (AVX-512 path and fallback tail path each pack B independently)
    // plus minor FP ordering drift (bitwise-identical check failed).
    // Reverted to strict gate.
    #[cfg(all(target_arch = "x86_64", any(target_os = "linux", target_os = "macos")))]
    if use_avx512_mr12(m, k, n, has_residual, epilogue.activation) && m.is_multiple_of(MR12) {
        if should_parallelize(plan, config, thread_pool) {
            blocked_gemm_parallel_mr12(a, b, out, m, k, n, epilogue, thread_pool);
        } else {
            blocked_gemm_sequential_mr12(a, b, out, m, k, n, epilogue);
        }
        return;
    }

    // Step S.1′: specialized low-k tile path (k ∈ {16, 24}, n % 24 == 0,
    // m % 4 == 0, work ≥ 1M FMAs). Hot tracker shapes (m=16384 k=16 n=96,
    // m=4096 k=24 n=144, m=4096 k=16 n=96) miss `use_blocked` (requires
    // k ≥ 32) and fall into `row_gemm_set_parallel_fused` which processes
    // 1 row at a time via `matmul_row_set_avx_fma` (7.95% of 6T cycles).
    // The tile path processes 4 rows × 24 cols at once with a
    // fully-unrolled const-K k-loop.
    //
    // **Ships OPT-IN via `YSCV_LOW_K_TILE=1`, default OFF.** Measured
    // tracker A/B: 6T p50 delta +70 µs (noise), 1T p50 +230 µs (+2.1%
    // regression — row_gemm's mature per-row path is already well-tuned
    // for k<32 shapes). Kept wired so a future inline-asm rewrite can
    // flip the default. See `project_step_s1_prime_landed.md`.
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if use_low_k_tile_avx_fma(m, k, n)
        && low_k_tile_enabled()
        && std::is_x86_feature_detected!("avx")
        && std::is_x86_feature_detected!("fma")
    {
        // SAFETY: caller guarantees `a` has ≥ m*k floats, `b` has ≥ k*n,
        // `out` has ≥ m*n. `use_low_k_tile_avx_fma` already verified the
        // shape alignment (m % 4 == 0, n % 24 == 0, k ∈ {16, 24}).
        // AVX+FMA feature is runtime-checked immediately above.
        #[allow(unsafe_code)]
        unsafe {
            low_k_tile_4x24_parallel_fused(a, b, out, m, k, n, &epilogue);
        }
        return;
    }

    // Blocked GEMM's `gebp_kernel_raw` now supports residual in the hot
    // AVX2 4×24 and 4×16 tile paths (Phase 1.2). For shapes whose jr
    // iteration ends in a 4×8 or scalar tail (n not expressible as
    // 3·NR·k + {0, 16}·j · NR), residual isn't propagated there yet —
    // fall back to row_gemm for those. Common Conv_Add shapes in
    // inference (n ∈ {16, 48, 64, 96, 112, ...}) all hit 4×24 + 4×16
    // only, so the fast path kicks in.
    if should_parallelize(plan, config, thread_pool) {
        let nthreads = super::config::available_threads(thread_pool);
        let blocked_blocks = m.div_ceil(MC_PARALLEL);
        if !residual_blocked_unsafe && use_blocked_path && blocked_blocks >= nthreads {
            blocked_gemm_parallel(a, b, out, m, k, n, epilogue, thread_pool, packed_b);
        } else {
            // Small-k path: fuse bias+activation into the per-row loop
            // so each row's output is written once while cache-hot.
            // Previous code did `row_gemm_set_parallel` + a separate
            // `apply_epilogue_fallback` pass over m×n output — two
            // memory passes instead of one. On m=16384, n=96 that's
            // ~6 MB of extra DRAM traffic per call.
            row_gemm_set_parallel_fused(a, b, out, m, k, n, thread_pool, &epilogue);
        }
    } else if !residual_blocked_unsafe && use_blocked_path {
        blocked_gemm_sequential(a, b, out, m, k, n, epilogue, packed_b);
    } else if has_residual {
        // Sequential row-GEMM with residual fused in each row's epilogue.
        // Rare branch — small matrices with residual usually go through
        // the parallel path above.
        row_gemm_set_parallel_fused(a, b, out, m, k, n, thread_pool, &epilogue);
    } else {
        row_gemm_set_sequential(a, b, out, m, k, n);
        apply_epilogue_fallback(out, m, n, &epilogue);
    }
}

/// Parallel row-GEMM with bias + activation fused into each row's
/// store phase. Mirrors `row_gemm_set_parallel` but folds the epilogue
/// work so the output is touched exactly once while cache-hot.
#[allow(unsafe_code)]
fn row_gemm_set_parallel_fused(
    left: &[f32],
    right: &[f32],
    output: &mut [f32],
    _m: usize,
    k: usize,
    n: usize,
    thread_pool: Option<&ThreadPool>,
    epilogue: &GemmEpilogue,
) {
    let _ = thread_pool;
    // Wrap raw pointers in Send+Sync adapters so the `par_chunks_mut_dispatch`
    // closure can capture them. The caller blocks until the scope completes,
    // so the pointed-to data (weights B, bias, residual) outlives every
    // worker deref.
    #[derive(Clone, Copy)]
    struct RawPtrs {
        left: *const f32,
        right: *const f32,
        bias: Option<*const f32>,
        residual: Option<*const f32>,
    }
    impl RawPtrs {
        #[inline]
        fn left(&self) -> *const f32 {
            self.left
        }
        #[inline]
        fn right(&self) -> *const f32 {
            self.right
        }
        #[inline]
        fn bias(&self) -> Option<*const f32> {
            self.bias
        }
        #[inline]
        fn residual(&self) -> Option<*const f32> {
            self.residual
        }
    }
    // SAFETY: Used only inside the par_chunks_mut_dispatch scope, which
    // blocks until all workers return. The underlying buffers outlive
    // that scope.
    #[allow(unsafe_code)]
    unsafe impl Send for RawPtrs {}
    #[allow(unsafe_code)]
    unsafe impl Sync for RawPtrs {}
    let ptrs = RawPtrs {
        left: left.as_ptr(),
        right: right.as_ptr(),
        bias: epilogue.bias,
        residual: epilogue.residual,
    };
    let activation = epilogue.activation;
    let activation_id: u8 = match activation {
        Activation::None => 0,
        Activation::Relu => 1,
        Activation::Silu => 2,
    };
    super::super::scope_ctx::par_chunks_mut_dispatch(output, n, move |row, out_row| unsafe {
        let p = ptrs;
        // Compute the raw matmul row.
        super::simd::matmul_row_set_dispatch(
            p.left().add(row * k),
            p.right(),
            out_row.as_mut_ptr(),
            k,
            n,
        );
        // Step S.4: fused residual + bias + activation in a single SIMD
        // pass over `out_row`. Previously 2-3 separate passes (scalar
        // residual + bias_relu_nhwc_dispatch). Cuts post-matmul memory
        // traffic over out_row from ~3× to 1×.
        let residual_slice = p
            .residual()
            .map(|r| std::slice::from_raw_parts(r.add(row * n), n));
        let bias_slice = p.bias().map(|b| std::slice::from_raw_parts(b, n));
        if residual_slice.is_some() || bias_slice.is_some() || activation_id != 0 {
            super::simd::fused_row_epilogue_dispatch(
                out_row,
                residual_slice,
                bias_slice,
                activation_id,
                n,
            );
        }
    });
}

/// Fallback epilogue for paths that don't go through blocked GEMM microkernels
/// (row-GEMM, BLAS). Applied as a separate pass — acceptable because these paths
/// handle small matrices where data is still in L1.
#[allow(unsafe_code)]
fn apply_epilogue_fallback(out: &mut [f32], m: usize, n: usize, epilogue: &GemmEpilogue) {
    // Phase 1: bias addition (before residual and activation).
    if let Some(bias_ptr) = epilogue.bias {
        // SAFETY: bias points to n valid f32s for the lifetime of this call.
        let bias = unsafe { std::slice::from_raw_parts(bias_ptr, n) };
        super::simd::bias_add_nhwc_dispatch(out, bias, m, n);
    }

    // Phase 2: residual add. Must come after bias and before activation so
    // the fused order matches the blocked-GEMM microkernel epilogue:
    // out[i] = gemm[i] + bias[i%n] + residual[i].
    if let Some(res_ptr) = epilogue.residual {
        // SAFETY: residual points to m*n valid f32s for the lifetime of this call.
        let res = unsafe { std::slice::from_raw_parts(res_ptr, m * n) };
        for (o, r) in out.iter_mut().zip(res.iter()) {
            *o += r;
        }
    }

    // Phase 3: activation.
    match epilogue.activation {
        Activation::None => {}
        Activation::Relu => super::simd::relu_slice_dispatch(out),
        Activation::Silu => super::conv::silu_slice_inplace(out),
    }
}

/// Rank-2 matrix multiplication with explicit parallelization heuristics.
pub fn matmul_2d_with_config(
    lhs: &Tensor,
    rhs: &Tensor,
    config: ParallelMatmulConfig,
) -> Result<Tensor, KernelError> {
    matmul_2d_with_config_and_pool(lhs, rhs, config, None)
}

/// Rank-2 matrix multiplication with strict sequential execution.
pub fn matmul_2d_sequential(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor, KernelError> {
    let plan = build_matmul_plan(lhs, rhs)?;
    matmul_2d_sequential_with_plan(lhs, rhs, plan)
}

pub fn matmul_2d_with_config_and_pool(
    lhs: &Tensor,
    rhs: &Tensor,
    config: ParallelMatmulConfig,
    thread_pool: Option<&ThreadPool>,
) -> Result<Tensor, KernelError> {
    let plan = build_matmul_plan(lhs, rhs)?;
    if should_parallelize(plan, config, thread_pool) {
        return matmul_2d_parallel_with_plan(lhs, rhs, plan, thread_pool);
    }
    matmul_2d_sequential_with_plan(lhs, rhs, plan)
}

// ---------------------------------------------------------------------------
// Plan & heuristics
// ---------------------------------------------------------------------------

fn build_matmul_plan(lhs: &Tensor, rhs: &Tensor) -> Result<MatMulPlan, KernelError> {
    if lhs.rank() != 2 || rhs.rank() != 2 {
        return Err(KernelError::InvalidMatMulRank {
            left_rank: lhs.rank(),
            right_rank: rhs.rank(),
        });
    }

    let m = lhs.shape()[0];
    let k_left = lhs.shape()[1];
    let k_right = rhs.shape()[0];
    let n = rhs.shape()[1];
    if k_left != k_right {
        return Err(KernelError::MatMulShapeMismatch {
            left: lhs.shape().to_vec(),
            right: rhs.shape().to_vec(),
        });
    }

    let output_len = m
        .checked_mul(n)
        .ok_or_else(|| KernelError::Tensor(TensorError::SizeOverflow { shape: vec![m, n] }))?;

    Ok(MatMulPlan {
        m,
        k: k_left,
        n,
        output_len,
    })
}

fn should_parallelize(
    plan: MatMulPlan,
    config: ParallelMatmulConfig,
    thread_pool: Option<&ThreadPool>,
) -> bool {
    if plan.m < 2
        || plan.output_len < config.min_parallel_output_elements
        || plan.k < config.min_parallel_shared_dim
    {
        return false;
    }
    should_parallelize_len(
        plan.output_len,
        config.min_parallel_output_elements,
        thread_pool,
    )
}

/// Returns true if blocked matmul should be used for these dimensions.
fn use_blocked(m: usize, k: usize, n: usize) -> bool {
    !cfg!(miri) && m >= BLOCKED_THRESHOLD && k >= BLOCKED_THRESHOLD && n >= 2 * NR
}

/// aarch64-only low-k blocked gate for Conv-like matmul shapes.
///
/// Hot ARM tracker shapes often have `k ∈ {16,24}` and large `m*n`. The
/// generic `use_blocked` gate rejects these because `k < 32`, routing them
/// through row-GEMM. When weights are static (`has_prepacked_b=true`), blocked
/// kernels can still win by reusing B-pack and increasing tile reuse.
///
/// Kill switch: `YSCV_NO_AARCH64_LOW_K_BLOCKED=1`.
/// Work threshold override:
/// `YSCV_AARCH64_LOW_K_BLOCKED_MIN_WORK_FMAS=<N>` (default 1_048_576 FMAs).
#[cfg(target_arch = "aarch64")]
#[inline]
fn use_blocked_aarch64_low_k(m: usize, k: usize, n: usize, has_prepacked_b: bool) -> bool {
    use std::sync::OnceLock;
    static ENABLED: OnceLock<bool> = OnceLock::new();
    static MIN_WORK_FMAS: OnceLock<usize> = OnceLock::new();
    let enabled =
        *ENABLED.get_or_init(|| std::env::var_os("YSCV_NO_AARCH64_LOW_K_BLOCKED").is_none());
    let min_work_fmas = *MIN_WORK_FMAS.get_or_init(|| {
        std::env::var("YSCV_AARCH64_LOW_K_BLOCKED_MIN_WORK_FMAS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&v| v > 0)
            .unwrap_or(1_048_576)
    });
    enabled
        && !cfg!(miri)
        && has_prepacked_b
        && m >= BLOCKED_THRESHOLD
        && (16..BLOCKED_THRESHOLD).contains(&k)
        && n >= 2 * NR
        && m.saturating_mul(k).saturating_mul(n) >= min_work_fmas
}

/// Enables blocked residual epilogue on aarch64 NEON matmul kernels.
///
/// Kill switch: `YSCV_NO_AARCH64_RESIDUAL_BLOCKED=1`.
#[cfg(target_arch = "aarch64")]
#[inline]
fn aarch64_residual_blocked_enabled() -> bool {
    use std::sync::OnceLock;
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("YSCV_NO_AARCH64_RESIDUAL_BLOCKED").is_none())
}

/// Returns true when blocked-GEMM residual fusion must be disabled due to
/// micro-kernel limitations.
///
/// Current state: scalar tail kernels now fold residual correctly, so there
/// are no known tail-shape correctness blockers in the generic blocked path.
#[inline]
fn blocked_residual_has_unsupported_tail(_nc: usize) -> bool {
    false
}

// ---------------------------------------------------------------------------
// BLAS dispatch (Accelerate on macOS, skip on other platforms)
// ---------------------------------------------------------------------------

/// Use BLAS when available and not under miri.
#[cfg(feature = "blas")]
fn use_blas() -> bool {
    !cfg!(miri) && cfg!(feature = "blas") && std::env::var_os("YSCV_FORCE_NO_BLAS").is_none()
}

#[inline]
#[cfg(feature = "blas")]
// Parameters are only consumed in the x86 cfg block; other targets ignore them.
#[allow(unused_variables)]
fn prefer_custom_gemm(m: usize, k: usize, n: usize) -> bool {
    if std::env::var_os("YSCV_FORCE_BLAS").is_some() {
        return false;
    }
    if std::env::var_os("YSCV_FORCE_NO_BLAS").is_some() {
        return true;
    }

    // OpenBLAS on x86 can be dramatically slower than our SIMD row/blocked GEMM
    // for Conv-like shapes (moderate K/N, often small M). Keep BLAS for very
    // wide square-ish GEMMs where its packing/micro-kernel wins.
    #[cfg(all(
        not(target_os = "macos"),
        any(target_arch = "x86", target_arch = "x86_64")
    ))]
    {
        if k <= 704 && n <= 704 {
            return true;
        }
        if m <= 128 && k <= 1536 && n <= 256 {
            return true;
        }
    }

    false
}

#[cfg(feature = "blas")]
#[inline]
fn run_custom_gemm(left: &[f32], right: &[f32], output: &mut [f32], m: usize, k: usize, n: usize) {
    let in_rayon_worker = rayon::current_thread_index().is_some();
    let can_parallel = !in_rayon_worker
        && rayon::current_num_threads() > 1
        && m >= 64
        && n >= 64
        && m.saturating_mul(n) >= 65_536;

    if use_blocked(m, k, n) {
        if can_parallel {
            blocked_gemm_parallel(
                left,
                right,
                output,
                m,
                k,
                n,
                GemmEpilogue::IDENTITY,
                None,
                None,
            );
        } else {
            blocked_gemm_sequential(left, right, output, m, k, n, GemmEpilogue::IDENTITY, None);
        }
    } else if can_parallel {
        row_gemm_set_parallel(left, right, output, m, k, n, None);
    } else {
        row_gemm_set_sequential(left, right, output, m, k, n);
    }
}

#[cfg(feature = "blas")]
#[allow(unsafe_code)]
pub(crate) fn blas_sgemm(
    left: &[f32],
    right: &[f32],
    output: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) {
    debug_assert!(
        left.len() >= m * k,
        "blas_sgemm: left.len()={} < m*k={}",
        left.len(),
        m * k
    );
    debug_assert!(
        right.len() >= k * n,
        "blas_sgemm: right.len()={} < k*n={}",
        right.len(),
        k * n
    );
    debug_assert!(
        output.len() >= m * n,
        "blas_sgemm: output.len()={} < m*n={}",
        output.len(),
        m * n
    );

    #[cfg(feature = "blas")]
    {
        if use_blas() && !prefer_custom_gemm(m, k, n) {
            // Using `cblas-sys` gives us battle-tested FFI declarations with
            // proper `#[repr]`-ed enum types for CBLAS_LAYOUT / CBLAS_TRANSPOSE
            // instead of raw `i32`. Same symbol, same ABI — but eliminates the
            // tiny risk of enum-underlying-type mismatch between our hand-rolled
            // extern block and the C header on any given platform (notably
            // vcpkg OpenBLAS + MSVC).
            use cblas_sys::{CBLAS_LAYOUT, CBLAS_TRANSPOSE, cblas_sgemm};
            // SAFETY: pointers are valid for at least m*k / k*n / m*n f32s as
            // asserted above; `cblas_sgemm` is a pure computation with no
            // aliasing requirements beyond the buffers being distinct when
            // beta != 0 (we pass beta = 0 so C is pure output).
            unsafe {
                cblas_sgemm(
                    CBLAS_LAYOUT::CblasRowMajor,
                    CBLAS_TRANSPOSE::CblasNoTrans,
                    CBLAS_TRANSPOSE::CblasNoTrans,
                    m as i32,
                    n as i32,
                    k as i32,
                    1.0,
                    left.as_ptr(),
                    k as i32,
                    right.as_ptr(),
                    n as i32,
                    0.0,
                    output.as_mut_ptr(),
                    n as i32,
                );
            }
            return;
        }
    }

    // No BLAS at build/runtime, or explicit custom-GEMM preference.
    run_custom_gemm(left, right, output, m, k, n);
}

/// Like `blas_sgemm` but with fused bias+activation epilogue.
/// When using our custom GEMM (no BLAS feature), the epilogue is applied
/// in the microkernel store phase. When using actual BLAS, it falls back
/// to a separate pass (BLAS is a black box).
#[allow(unsafe_code)]
pub(crate) fn blas_sgemm_fused(
    left: &[f32],
    right: &[f32],
    output: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    epilogue: GemmEpilogue,
) {
    debug_assert!(left.len() >= m * k);
    debug_assert!(right.len() >= k * n);
    debug_assert!(output.len() >= m * n);

    #[cfg(feature = "blas")]
    {
        if use_blas() && !prefer_custom_gemm(m, k, n) {
            use cblas_sys::{CBLAS_LAYOUT, CBLAS_TRANSPOSE, cblas_sgemm};
            unsafe {
                cblas_sgemm(
                    CBLAS_LAYOUT::CblasRowMajor,
                    CBLAS_TRANSPOSE::CblasNoTrans,
                    CBLAS_TRANSPOSE::CblasNoTrans,
                    m as i32,
                    n as i32,
                    k as i32,
                    1.0,
                    left.as_ptr(),
                    k as i32,
                    right.as_ptr(),
                    n as i32,
                    0.0,
                    output.as_mut_ptr(),
                    n as i32,
                );
            }
            apply_epilogue_fallback(output, m, n, &epilogue);
            return;
        }
    }

    run_custom_gemm_fused(left, right, output, m, k, n, epilogue);
}

fn run_custom_gemm_fused(
    left: &[f32],
    right: &[f32],
    output: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    epilogue: GemmEpilogue,
) {
    let in_rayon_worker = rayon::current_thread_index().is_some();
    let can_parallel = !in_rayon_worker
        && rayon::current_num_threads() > 1
        && m >= 64
        && n >= 64
        && m.saturating_mul(n) >= 65_536;

    if use_blocked(m, k, n) {
        if can_parallel {
            blocked_gemm_parallel(left, right, output, m, k, n, epilogue, None, None);
        } else {
            blocked_gemm_sequential(left, right, output, m, k, n, epilogue, None);
        }
    } else if can_parallel {
        row_gemm_set_parallel(left, right, output, m, k, n, None);
        apply_epilogue_fallback(output, m, n, &epilogue);
    } else {
        row_gemm_set_sequential(left, right, output, m, k, n);
        apply_epilogue_fallback(output, m, n, &epilogue);
    }
}

// ---------------------------------------------------------------------------
// Sequential dispatch
// ---------------------------------------------------------------------------

#[allow(unsafe_code)]
fn matmul_2d_sequential_with_plan(
    lhs: &Tensor,
    rhs: &Tensor,
    plan: MatMulPlan,
) -> Result<Tensor, KernelError> {
    // SAFETY: Every element is written by BLAS / blocked GEMM / row GEMM
    // before the tensor is returned, so uninit memory is never exposed.
    let mut output = AlignedVec::<f32>::uninitialized(plan.output_len);
    let left = lhs.data();
    let right = rhs.data();

    #[cfg(feature = "blas")]
    if use_blas() {
        blas_sgemm(left, right, &mut output, plan.m, plan.k, plan.n);
        return Tensor::from_aligned(vec![plan.m, plan.n], output).map_err(Into::into);
    }

    if use_blocked(plan.m, plan.k, plan.n) {
        blocked_gemm_sequential(
            left,
            right,
            &mut output,
            plan.m,
            plan.k,
            plan.n,
            GemmEpilogue::IDENTITY,
            None,
        );
    } else {
        row_gemm_set_sequential(left, right, &mut output, plan.m, plan.k, plan.n);
    }

    Tensor::from_aligned(vec![plan.m, plan.n], output).map_err(Into::into)
}

#[allow(unsafe_code)]
fn matmul_2d_parallel_with_plan(
    lhs: &Tensor,
    rhs: &Tensor,
    plan: MatMulPlan,
    thread_pool: Option<&ThreadPool>,
) -> Result<Tensor, KernelError> {
    let mut output = AlignedVec::<f32>::uninitialized(plan.output_len);
    let left = lhs.data();
    let right = rhs.data();

    #[cfg(feature = "blas")]
    if use_blas() {
        blas_sgemm(left, right, &mut output, plan.m, plan.k, plan.n);
        return Tensor::from_aligned(vec![plan.m, plan.n], output).map_err(Into::into);
    }

    if use_blocked(plan.m, plan.k, plan.n) {
        blocked_gemm_parallel(
            left,
            right,
            &mut output,
            plan.m,
            plan.k,
            plan.n,
            GemmEpilogue::IDENTITY,
            thread_pool,
            None,
        );
    } else {
        row_gemm_set_parallel(
            left,
            right,
            &mut output,
            plan.m,
            plan.k,
            plan.n,
            thread_pool,
        );
    }

    Tensor::from_aligned(vec![plan.m, plan.n], output).map_err(Into::into)
}

// ---------------------------------------------------------------------------
// Legacy row-based GEMM (for small matrices)
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// "Set" row GEMM — writes directly, no zero-init needed
// ---------------------------------------------------------------------------

#[allow(unsafe_code)]
fn row_gemm_set_sequential(
    left: &[f32],
    right: &[f32],
    output: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) {
    debug_assert!(left.len() >= m * k);
    debug_assert!(right.len() >= k * n);
    debug_assert!(output.len() >= m * n);
    let left_ptr = left.as_ptr();
    let right_ptr = right.as_ptr();
    let out_ptr = output.as_mut_ptr();

    for row in 0..m {
        unsafe {
            matmul_row_set_dispatch(left_ptr.add(row * k), right_ptr, out_ptr.add(row * n), k, n);
        }
    }
}

#[allow(unsafe_code)]
fn row_gemm_set_parallel(
    left: &[f32],
    right: &[f32],
    output: &mut [f32],
    _m: usize,
    k: usize,
    n: usize,
    thread_pool: Option<&ThreadPool>,
) {
    let _ = thread_pool; // route through TLS scope_ctx
    super::super::scope_ctx::par_chunks_mut_dispatch(output, n, |row, out_row| unsafe {
        matmul_row_set_dispatch(
            left.as_ptr().add(row * k),
            right.as_ptr(),
            out_row.as_mut_ptr(),
            k,
            n,
        );
    });
}

// ---------------------------------------------------------------------------
// Blocked tiled GEMM
// ---------------------------------------------------------------------------

/// Pack A[ic..ic+mc, pc..pc+kc] into panel format: (mc/MR) panels × kc × MR.
///
/// Each panel stores MR rows for all kc columns contiguously:
/// `packed[(ir/MR)*kc*MR + p*MR + i] = A[(ic+ir+i), (pc+p)]`
#[allow(unsafe_code)]
fn pack_a_panel(
    a: &[f32],
    lda: usize,
    ic: usize,
    mc: usize,
    pc: usize,
    kc: usize,
    packed: &mut [f32],
) {
    let a_ptr = a.as_ptr();
    let p_ptr = packed.as_mut_ptr();
    let mut idx = 0usize;
    for ir in (0..mc).step_by(MR) {
        let mr = MR.min(mc - ir);
        if mr == MR {
            for p in 0..kc {
                unsafe {
                    for i in 0..MR {
                        *p_ptr.add(idx + i) = *a_ptr.add((ic + ir + i) * lda + pc + p);
                    }
                }
                idx += MR;
            }
        } else {
            for p in 0..kc {
                unsafe {
                    for i in 0..mr {
                        *p_ptr.add(idx + i) = *a_ptr.add((ic + ir + i) * lda + pc + p);
                    }
                    std::ptr::write_bytes(p_ptr.add(idx + mr), 0, MR - mr);
                }
                idx += MR;
            }
        }
    }
}

/// MR=8 variant of `pack_a_panel` for the aarch64 NEON 8×12 microkernel.
/// Groups rows in 8-row panels, tail rows zero-padded. aarch64 only —
/// x86/scalar stay on MR=4.
#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code)]
fn pack_a_panel_mr8(
    a: &[f32],
    lda: usize,
    ic: usize,
    mc: usize,
    pc: usize,
    kc: usize,
    packed: &mut [f32],
) {
    const MR8: usize = 8;
    let a_ptr = a.as_ptr();
    let p_ptr = packed.as_mut_ptr();
    let mut idx = 0usize;
    for ir in (0..mc).step_by(MR8) {
        let mr = MR8.min(mc - ir);
        if mr == MR8 {
            for p in 0..kc {
                unsafe {
                    for i in 0..MR8 {
                        *p_ptr.add(idx + i) = *a_ptr.add((ic + ir + i) * lda + pc + p);
                    }
                }
                idx += MR8;
            }
        } else {
            for p in 0..kc {
                unsafe {
                    for i in 0..mr {
                        *p_ptr.add(idx + i) = *a_ptr.add((ic + ir + i) * lda + pc + p);
                    }
                    std::ptr::write_bytes(p_ptr.add(idx + mr), 0, MR8 - mr);
                }
                idx += MR8;
            }
        }
    }
}

/// Step C1: MR=6 variant of `pack_a_panel` for the x86 MR=6×16 AVX2
/// microkernel. Groups rows in 6-row panels; tail rows zero-padded.
/// Pack layout matches `pack_a_panel` style: (mc/MR6) panels × kc × MR6.
///
/// Used only by the MR=6 fast path when `use_mr6_blocked()` returns true
/// (x86 with FMA+AVX, m≥192, k≥64). MR=4 path continues for tail and
/// small shapes.
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[allow(unsafe_code)]
fn pack_a_panel_mr6(
    a: &[f32],
    lda: usize,
    ic: usize,
    mc: usize,
    pc: usize,
    kc: usize,
    packed: &mut [f32],
) {
    let a_ptr = a.as_ptr();
    let p_ptr = packed.as_mut_ptr();
    let mut idx = 0usize;
    for ir in (0..mc).step_by(MR6) {
        let mr = MR6.min(mc - ir);
        if mr == MR6 {
            for p in 0..kc {
                // SAFETY: `idx + MR6` stays within `packed` (sized
                // `div_ceil(mc, MR6) * kc * MR6` by caller);
                // `(ic+ir+i)*lda + pc + p` is inside A's full panel.
                unsafe {
                    for i in 0..MR6 {
                        *p_ptr.add(idx + i) = *a_ptr.add((ic + ir + i) * lda + pc + p);
                    }
                }
                idx += MR6;
            }
        } else {
            for p in 0..kc {
                // SAFETY: partial row — zero-pad the tail to MR6 so the
                // microkernel can always load MR6 floats per k.
                unsafe {
                    for i in 0..mr {
                        *p_ptr.add(idx + i) = *a_ptr.add((ic + ir + i) * lda + pc + p);
                    }
                    std::ptr::write_bytes(p_ptr.add(idx + mr), 0, MR6 - mr);
                }
                idx += MR6;
            }
        }
    }
}

/// Pack B[pc..pc+kc, jc..jc+nc] into panel format: (nc/NR) panels × kc × NR.
///
/// Each panel stores NR columns for all kc rows contiguously:
/// `packed[(jr/NR)*kc*NR + p*NR + j] = B[(pc+p), (jc+jr+j)]`
#[allow(unsafe_code)]
fn pack_b_panel(
    b: &[f32],
    ldb: usize,
    pc: usize,
    kc: usize,
    jc: usize,
    nc: usize,
    packed: &mut [f32],
) {
    let b_ptr = b.as_ptr();
    let p_ptr = packed.as_mut_ptr();
    let mut idx = 0usize;
    for jr in (0..nc).step_by(NR) {
        let nr = NR.min(nc - jr);
        if nr == NR {
            for p in 0..kc {
                let src = (pc + p) * ldb + jc + jr;
                unsafe {
                    std::ptr::copy_nonoverlapping(b_ptr.add(src), p_ptr.add(idx), NR);
                }
                idx += NR;
            }
        } else {
            for p in 0..kc {
                let src = (pc + p) * ldb + jc + jr;
                unsafe {
                    std::ptr::copy_nonoverlapping(b_ptr.add(src), p_ptr.add(idx), nr);
                    std::ptr::write_bytes(p_ptr.add(idx + nr), 0, NR - nr);
                }
                idx += NR;
            }
        }
    }
}

fn div_ceil(a: usize, b: usize) -> usize {
    a.div_ceil(b)
}

/// aarch64-only: NR=12-aligned B packer for the MR=8×12 NEON kernel.
///
/// The generic `pack_b_panel` uses NR=8; the 8×12 kernel needs 12 cols
/// per k-step in one contiguous load (`ld1 {v24,v25,v26}, [x1]`) which
/// forces a different packed stride. Layout mirrors `pack_b_panel` but
/// with `NR_MR8=12` panels:
///   `packed[jr_panel * kc * 12 + p * 12 + j]` = `B[pc+p][jc+jr_panel*12+j]`.
/// Tail cols (nc not a multiple of 12) zero-padded so every panel is
/// exactly 12 floats per k-step.
#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code)]
fn pack_b_panel_nr12(
    b: &[f32],
    ldb: usize,
    pc: usize,
    kc: usize,
    jc: usize,
    nc: usize,
    packed: &mut [f32],
) {
    const NR12: usize = 12;
    let b_ptr = b.as_ptr();
    let p_ptr = packed.as_mut_ptr();
    let mut idx = 0usize;
    for jr in (0..nc).step_by(NR12) {
        let nr = NR12.min(nc - jr);
        if nr == NR12 {
            for p in 0..kc {
                let src = (pc + p) * ldb + jc + jr;
                unsafe {
                    std::ptr::copy_nonoverlapping(b_ptr.add(src), p_ptr.add(idx), NR12);
                }
                idx += NR12;
            }
        } else {
            for p in 0..kc {
                let src = (pc + p) * ldb + jc + jr;
                unsafe {
                    std::ptr::copy_nonoverlapping(b_ptr.add(src), p_ptr.add(idx), nr);
                    std::ptr::write_bytes(p_ptr.add(idx + nr), 0, NR12 - nr);
                }
                idx += NR12;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Packed-B cache: kernel weights are constant across inferences, so packing
// them every call is pure waste. Cache the fully-packed form keyed by the B
// buffer's pointer + (K, N) so repeat calls with the same kernel are free.
// ---------------------------------------------------------------------------

/// One packed B contains every (pc_idx, jc_idx) block of a (K × N) matrix,
/// stored contiguously. Block layout per block matches `pack_b_panel` output.
///
/// Publicly exposed so the ONNX loader can pre-pack constant Conv/MatMul
/// weights at session-create and hand them out as `Arc<PackedB>` — the
/// per-inference hot path then skips both the fingerprint lookup and the
/// pack itself. Pre-packed data is immutable after construction, so `PackedB`
/// is `Send + Sync` (the internal `Vec<f32>` is never mutated post-build).
pub struct PackedB {
    /// Contiguous storage: blocks laid out in (pc_idx, jc_idx) row-major order.
    data: Vec<f32>,
    /// `block_slots` floats per block — equals `div_ceil(NC, NR) * KC * NR`.
    block_slots: usize,
    /// Number of column-blocks (`div_ceil(N, NC)`).
    num_jc: usize,
    /// Dimensions of the source B matrix. Used to reject stale packs if shape
    /// changes (should never happen for initializer kernels, but cheap check).
    k: usize,
    n: usize,
}

impl PackedB {
    #[inline]
    fn block(&self, pc_idx: usize, jc_idx: usize) -> &[f32] {
        let off = (pc_idx * self.num_jc + jc_idx) * self.block_slots;
        &self.data[off..off + self.block_slots]
    }

    #[inline]
    fn matches(&self, k: usize, n: usize) -> bool {
        self.k == k && self.n == n
    }

    /// Dimensions of the source B matrix (for dispatch shape checks at the
    /// callsite before handing the pre-pack to the GEMM layer).
    #[inline]
    pub fn dims(&self) -> (usize, usize) {
        (self.k, self.n)
    }
}

// Auto-derived Send+Sync via `Vec<f32>` + `usize` fields. Explicit note for
// readers: nothing in PackedB is mutated after `full_pack_b` returns, so
// sharing the `Arc<PackedB>` across threads at inference time is safe.

impl std::fmt::Debug for PackedB {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Abbreviated — a full dump of thousands of floats would be useless.
        write!(
            f,
            "PackedB {{ k: {}, n: {}, data_len: {}, block_slots: {}, num_jc: {} }}",
            self.k,
            self.n,
            self.data.len(),
            self.block_slots,
            self.num_jc
        )
    }
}

/// Pre-packs a constant B matrix (e.g. a Conv kernel weight) into the
/// blocked-GEMM layout for later zero-overhead reuse. Intended for use at
/// model load time — returned `Arc<PackedB>` can be handed to every
/// subsequent `matmul_2d_slices_fused_prepacked` call without hashing,
/// locking, or re-packing.
pub fn pack_b_for_session(b: &[f32], k: usize, n: usize) -> std::sync::Arc<PackedB> {
    std::sync::Arc::new(full_pack_b(b, k, n))
}

/// FEAT_FP16 half-precision 6×16 GEMM for a single tile (6 rows × 16 cols).
/// `a_panel`, `b_panel_{0,1}` are packed fp16 bit-patterns (u16); `c` is
/// fp16 output. Requires aarch64 + FEAT_FP16 — caller verifies via
/// `std::arch::is_aarch64_feature_detected!("fp16")` and routes scalar
/// otherwise. Each matmul invocation tiles at the caller layer; this
/// function is the microkernel.
///
/// Opt-in via the yscv-onnx loader env `YSCV_FP16=1`. Not wired into the
/// default Conv dispatch yet because load-time weight casting to fp16 +
/// accuracy validation on real ARM hw are prerequisites (see Phase 3.J
/// in the roadmap).
#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code)]
pub fn hgemm_6x16_neon(
    accumulate: bool,
    a_panel: &[u16],
    b_panel_0: &[u16],
    b_panel_1: &[u16],
    c: &mut [u16],
    ldc: usize,
    kc: usize,
) {
    // Safety: caller must size panels correctly (6*kc, kc*8, kc*8) and
    // ensure `c` has at least 6*ldc valid u16 entries. We debug_assert
    // shape bounds to catch misuse in tests.
    debug_assert!(a_panel.len() >= 6 * kc);
    debug_assert!(b_panel_0.len() >= kc * 8);
    debug_assert!(b_panel_1.len() >= kc * 8);
    debug_assert!(c.len() >= 5 * ldc + 16);
    unsafe {
        if accumulate {
            sgemm_asm_aarch64::yscv_hgemm_6x16_neon_acc(
                a_panel.as_ptr(),
                b_panel_0.as_ptr(),
                b_panel_1.as_ptr(),
                c.as_mut_ptr(),
                ldc,
                kc,
            );
        } else {
            sgemm_asm_aarch64::yscv_hgemm_6x16_neon_set(
                a_panel.as_ptr(),
                b_panel_0.as_ptr(),
                b_panel_1.as_ptr(),
                c.as_mut_ptr(),
                ldc,
                kc,
            );
        }
    }
}

fn full_pack_b(b: &[f32], k: usize, n: usize) -> PackedB {
    let num_pc = div_ceil(k, KC);
    let num_jc = div_ceil(n, NC);
    let block_slots = div_ceil(NC, NR) * KC * NR;
    let total = num_pc * num_jc * block_slots;
    // Zero-filled allocation — `pack_b_panel` below overwrites every slot,
    // the pre-zero is just a safety pre-condition for `clippy::uninit_vec`.
    // Called once per (weight, K, N) tuple then cached, so the memset
    // cost is amortized to near-zero per inference.
    let mut data: Vec<f32> = vec![0.0; total];

    for pc_idx in 0..num_pc {
        let pc = pc_idx * KC;
        let kc = KC.min(k - pc);
        for jc_idx in 0..num_jc {
            let jc = jc_idx * NC;
            let nc = NC.min(n - jc);
            let block_off = (pc_idx * num_jc + jc_idx) * block_slots;
            // pack_b_panel writes `div_ceil(nc, NR) * kc * NR` floats — we give
            // it the full block_slots slice which is the max possible.
            pack_b_panel(
                b,
                n,
                pc,
                kc,
                jc,
                nc,
                &mut data[block_off..block_off + block_slots],
            );
        }
    }
    PackedB {
        data,
        block_slots,
        num_jc,
        k,
        n,
    }
}

thread_local! {
    /// Runtime cache for callers that don't pre-pack. Key includes a few
    /// fingerprint samples of the B buffer so we don't return a stale pack
    /// when a temporary buffer is reallocated at the same address.
    static PACKED_B_CACHE: std::cell::RefCell<
        std::collections::HashMap<(usize, usize, usize, u32, u32, u32), std::rc::Rc<PackedB>>,
    > = std::cell::RefCell::new(std::collections::HashMap::new());
}

/// aarch64-only NR=12 packed B companion to `PackedB`. Shares the
/// `(pc_idx, jc_idx)` block-layout idea but uses a 12-col stride so the
/// 8×12 NEON kernel can load three quads per k-step from one contiguous
/// panel (the generic NR=8 layout can't satisfy that).
#[cfg(target_arch = "aarch64")]
pub(crate) struct PackedBNr12 {
    data: Vec<f32>,
    block_slots: usize,
    num_jc: usize,
    k: usize,
    n: usize,
}

#[cfg(target_arch = "aarch64")]
impl PackedBNr12 {
    #[inline]
    fn block(&self, pc_idx: usize, jc_idx: usize) -> &[f32] {
        let off = (pc_idx * self.num_jc + jc_idx) * self.block_slots;
        &self.data[off..off + self.block_slots]
    }
}

#[cfg(target_arch = "aarch64")]
fn full_pack_b_nr12(b: &[f32], k: usize, n: usize) -> PackedBNr12 {
    const NR12: usize = 12;
    let num_pc = div_ceil(k, KC);
    let num_jc = div_ceil(n, NC);
    let block_slots = div_ceil(NC, NR12) * KC * NR12;
    let total = num_pc * num_jc * block_slots;
    // Zero-filled; `pack_b_panel_nr12` writes the active slots, tail
    // padding stays zero-pre-init. See `full_pack_b` for the same pattern.
    let mut data: Vec<f32> = vec![0.0; total];

    for pc_idx in 0..num_pc {
        let pc = pc_idx * KC;
        let kc = KC.min(k - pc);
        for jc_idx in 0..num_jc {
            let jc = jc_idx * NC;
            let nc = NC.min(n - jc);
            let block_off = (pc_idx * num_jc + jc_idx) * block_slots;
            pack_b_panel_nr12(
                b,
                n,
                pc,
                kc,
                jc,
                nc,
                &mut data[block_off..block_off + block_slots],
            );
        }
    }
    PackedBNr12 {
        data,
        block_slots,
        num_jc,
        k,
        n,
    }
}

#[cfg(target_arch = "aarch64")]
thread_local! {
    static PACKED_B_NR12_CACHE: std::cell::RefCell<
        std::collections::HashMap<
            (usize, usize, usize, u32, u32, u32),
            std::rc::Rc<PackedBNr12>,
        >,
    > = std::cell::RefCell::new(std::collections::HashMap::new());
}

#[cfg(target_arch = "aarch64")]
fn get_or_pack_b_nr12(b: &[f32], k: usize, n: usize) -> std::rc::Rc<PackedBNr12> {
    let ptr_key = b.as_ptr() as usize;
    // Fingerprint 3 evenly-spaced elements so same-address re-allocations
    // of different tensors don't silently alias. Matches the NR=8 cache.
    let len = b.len().max(1);
    let fp0 = if !b.is_empty() { b[0].to_bits() } else { 0 };
    let fp1 = b[len / 2].to_bits();
    let fp2 = b[len - 1].to_bits();
    let key = (ptr_key, k, n, fp0, fp1, fp2);

    PACKED_B_NR12_CACHE.with(|cache| {
        if let Some(hit) = cache.borrow().get(&key)
            && hit.k == k
            && hit.n == n
        {
            return std::rc::Rc::clone(hit);
        }
        let packed = std::rc::Rc::new(full_pack_b_nr12(b, k, n));
        cache.borrow_mut().insert(key, std::rc::Rc::clone(&packed));
        packed
    })
}

thread_local! {
    /// Step 2: Per-thread scratch for `pack_a_panel` output. Sized on-demand
    /// via `ensure_capacity`; reused across every blocked-GEMM call on this
    /// thread. Previously each `par_for_each_ic` closure did a fresh
    /// `Vec::<f32>::with_capacity(pa_size)` per `(jc, pc)` block per worker —
    /// for tracker's 132 Conv ops × 6 threads × 2-4 blocks = thousands of
    /// heap allocs per inference, amortized ~100-200 ns each. TLS reuse
    /// eliminates the allocator pressure entirely (one warmup alloc per
    /// thread, then constant thereafter).
    ///
    /// Initial capacity 64 KB (16K f32) covers typical MC×KC×MR = 128×256×4
    /// / MR = 32 768 floats / 128 KB for m<128 shapes. Auto-grows to the
    /// max observed size across the process lifetime.
    static PACKED_A_SCRATCH: std::cell::RefCell<Vec<f32>> =
        std::cell::RefCell::new(Vec::with_capacity(16 * 1024));
}

/// Run `f` with a thread-local `pa_size`-sized packed-A scratch buffer.
/// The buffer persists across calls — zero allocator traffic in the hot
/// path once warm.
#[inline]
fn with_packed_a_tls<R>(pa_size: usize, f: impl FnOnce(&mut [f32]) -> R) -> R {
    PACKED_A_SCRATCH.with(|cell| {
        let mut buf = cell.borrow_mut();
        let have = buf.len();
        if buf.capacity() < pa_size {
            buf.reserve(pa_size.saturating_sub(have));
        }
        // SAFETY: we just ensured `capacity >= pa_size`. `pack_a_panel`
        // writes all `pa_size` elements before the first read, so the
        // uninitialized data is never observed.
        #[allow(unsafe_code)]
        unsafe {
            buf.set_len(pa_size);
        }
        f(&mut buf[..pa_size])
    })
}

fn get_or_pack_b(b: &[f32], k: usize, n: usize) -> std::rc::Rc<PackedB> {
    // Cache key = (pointer, K, N) ALONE is unsound: if the caller hands us a
    // temporary buffer (e.g. a per-inference layout-converted kernel), the
    // allocator can reuse the same address across calls with different data.
    // Fingerprint a few sample f32 values so we detect aliasing and re-pack.
    let key_ptr = b.as_ptr() as usize;
    let sample0 = b.first().copied().unwrap_or(0.0).to_bits();
    let sample1 = b.get(b.len() / 2).copied().unwrap_or(0.0).to_bits();
    let sample2 = b.last().copied().unwrap_or(0.0).to_bits();
    let key = (key_ptr, k, n, sample0, sample1, sample2);
    PACKED_B_CACHE.with(|cache| {
        if let Some(rc) = cache.borrow().get(&key) {
            return rc.clone();
        }
        let packed = std::rc::Rc::new(full_pack_b(b, k, n));
        cache.borrow_mut().insert(key, packed.clone());
        packed
    })
}

/// Sequential blocked GEMM: 3-level cache blocking with MR×NR micro-kernel.
/// If `pre_packed_b` is `Some`, uses it directly (load-time pre-packed kernel
/// weights). Otherwise falls back to a thread-local runtime cache keyed by
/// B's pointer — constant kernels still pay the pack cost only once per thread.
#[allow(unsafe_code)]
fn blocked_gemm_sequential(
    left: &[f32],
    right: &[f32],
    output: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    epilogue: GemmEpilogue,
    pre_packed_b: Option<&PackedB>,
) {
    let a_size = div_ceil(MC, MR) * KC * MR;
    // `pack_a_panel` overwrites every slot before first read, but clippy
    // correctly flags the historical `with_capacity + set_len` pattern as
    // UB-adjacent. The zero-init is an inference-loop-negligible memset
    // (~a few µs per KC-block); real hot paths go through parallel which
    // uses the `PACKED_A_SCRATCH` TLS reuse pool and pays it once per
    // thread lifetime.
    let mut packed_a: Vec<f32> = vec![0.0; a_size];

    let rc_fallback;
    let packed_b_full: &PackedB = if let Some(pb) = pre_packed_b {
        pb
    } else {
        rc_fallback = get_or_pack_b(right, k, n);
        &rc_fallback
    };

    for jc in (0..n).step_by(NC) {
        let jc_idx = jc / NC;
        let nc = NC.min(n - jc);
        for pc in (0..k).step_by(KC) {
            let pc_idx = pc / KC;
            let kc = KC.min(k - pc);
            let accumulate = pc > 0;
            let is_last_k = pc + kc >= k;

            let packed_b = packed_b_full.block(pc_idx, jc_idx);

            for ic in (0..m).step_by(MC) {
                let mc = MC.min(m - ic);
                pack_a_panel(left, k, ic, mc, pc, kc, &mut packed_a);
                gebp_kernel(
                    &packed_a, packed_b, output, n, ic, jc, mc, nc, kc, accumulate, epilogue,
                    is_last_k,
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// MR=6 blocked-GEMM path (x86_64 Linux/macOS, AVX+FMA only) — INLINE EPILOGUE.
// ---------------------------------------------------------------------------
//
// MR=6 primary AVX2 path matching MLAS SgemmKernelFma3. 12/16 YMM acc
// occupancy = 75% vs our old 4×16 = 50%. The `.S` kernel fuses bias+Relu
// into the store phase so we get the occupancy win WITHOUT giving up the
// single-memory-pass store — unlike the earlier raw-kernel + post-pass
// attempt which regressed −18% (see dead-ends log).
//
// Not a universal replacement for MR=4:
//   - SiLU activation: caller routes to MR=4 path (SiLU approximation in
//     asm would need polynomial + sigmoid — too many insn for inline store).
//   - aarch64 / scalar: stay on MR=4 (kernels and pack layout unchanged).

// ---------------------------------------------------------------------------
// MR=8 / 8×12 aarch64 NEON blocked-GEMM path
// ---------------------------------------------------------------------------

/// MC for the aarch64 MR=8 sequential path. Must be multiple of 8. 128/8=16.
#[cfg(target_arch = "aarch64")]
const MC_MR8: usize = 128;

/// MC for the aarch64 MR=8 parallel path. Multiple of 8 for clean tiling.
#[cfg(target_arch = "aarch64")]
const MC_PARALLEL_MR8: usize = 16;

/// Step C1: MC for the x86 MR=6 sequential path. Multiple of 6 for clean
/// tiling. Matches MR=4's MC=192 in total work per block (32 MR=6 panels
/// × KC = 32×256×6 = 48 KB A-pack per block, fits L2).
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
const MC_MR6: usize = 192;

/// Step C1: MC for the x86 MR=6 parallel path. Multiple of 6 (18 = 3 × 6)
/// for clean tiling; smaller than sequential so thread count × blocks
/// >= nthreads across tracker shapes (m=64..1024).
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
const MC_PARALLEL_MR6: usize = 18;

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code)]
fn blocked_gemm_sequential_mr8(
    left: &[f32],
    right: &[f32],
    output: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    epilogue: GemmEpilogue,
    _pre_packed_b: Option<&PackedB>, // NR=8 pre-pack is incompatible with MR=8
) {
    const MR8: usize = 8;
    let a_size = div_ceil(MC_MR8, MR8) * KC * MR8;
    // Zero-filled A-pack scratch; `pack_a_panel_mr8` overwrites every slot
    // before the GEBP kernel reads. Same pattern as `blocked_gemm_sequential`.
    let mut packed_a: Vec<f32> = vec![0.0; a_size];

    // MR=8 path needs NR=12-aligned packed B (see `pack_b_panel_nr12`).
    // The generic `PackedB` handed in by `pre_packed_b` is NR=8 — ignore
    // it here and use the NR=12-specific cache. Callers that want to
    // avoid the runtime pack cost should keep `YSCV_NO_MR8=1` set; a
    // future pack_b_for_session variant can emit NR=12 for the MR=8
    // path if that tradeoff becomes worthwhile.
    let packed_b_full = get_or_pack_b_nr12(right, k, n);

    for jc in (0..n).step_by(NC) {
        let jc_idx = jc / NC;
        let nc = NC.min(n - jc);
        for pc in (0..k).step_by(KC) {
            let pc_idx = pc / KC;
            let kc = KC.min(k - pc);
            let accumulate = pc > 0;
            let is_last_k = pc + kc >= k;
            let packed_b = packed_b_full.block(pc_idx, jc_idx);

            for ic in (0..m).step_by(MC_MR8) {
                let mc = MC_MR8.min(m - ic);
                pack_a_panel_mr8(left, k, ic, mc, pc, kc, &mut packed_a);
                unsafe {
                    gebp_kernel_raw_mr8(
                        packed_a.as_ptr(),
                        packed_b.as_ptr(),
                        output.as_mut_ptr(),
                        n,
                        ic,
                        jc,
                        mc,
                        nc,
                        kc,
                        accumulate,
                        epilogue,
                        is_last_k,
                    );
                }
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code)]
fn blocked_gemm_parallel_mr8(
    left: &[f32],
    right: &[f32],
    output: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    epilogue: GemmEpilogue,
    thread_pool: Option<&ThreadPool>,
    _pre_packed_b: Option<&PackedB>, // NR=8 pre-pack incompatible, see _sequential
) {
    const MR8: usize = 8;
    let out_ptr = SendPtr(output.as_mut_ptr());

    // NR=12 packed B for the MR=8 kernel — see `blocked_gemm_sequential_mr8`.
    let packed_b_full = get_or_pack_b_nr12(right, k, n);
    let packed_b_ptr = packed_b_full.data.as_ptr() as usize;
    let block_slots = packed_b_full.block_slots;
    let num_jc = packed_b_full.num_jc;

    let ic_blocks: Vec<usize> = (0..m).step_by(MC_PARALLEL_MR8).collect();

    let work = || {
        for jc in (0..n).step_by(NC) {
            let jc_idx = jc / NC;
            let nc = NC.min(n - jc);
            for pc in (0..k).step_by(KC) {
                let pc_idx = pc / KC;
                let kc = KC.min(k - pc);
                let accumulate = pc > 0;
                let is_last_k = pc + kc >= k;
                let block_off = (pc_idx * num_jc + jc_idx) * block_slots;
                let out_p = &out_ptr;

                par_for_each_ic(&ic_blocks, thread_pool, |ic| {
                    let mc = MC_PARALLEL_MR8.min(m - ic);
                    let a_panels = div_ceil(mc, MR8);
                    let pa_size = a_panels * kc * MR8;
                    with_packed_a_tls(pa_size, |packed_a| {
                        pack_a_panel_mr8(left, k, ic, mc, pc, kc, packed_a);
                        unsafe {
                            gebp_kernel_raw_mr8(
                                packed_a.as_ptr(),
                                (packed_b_ptr as *const f32).add(block_off),
                                out_p.0,
                                n,
                                ic,
                                jc,
                                mc,
                                nc,
                                kc,
                                accumulate,
                                epilogue,
                                is_last_k,
                            );
                        }
                    });
                });
            }
        }
    };
    if let Some(pool) = thread_pool {
        pool.install(work);
    } else {
        work();
    }
}

/// MR=8 × NR=12 GEBP kernel for aarch64. Advances `jr += NR12 = 12` and
/// reads from a dedicated NR=12 packed-B layout (`pack_b_panel_nr12`),
/// so `b_off = (jr/NR12) * kc * NR12` always lands on a valid panel start.
///
/// Full 8×12 tile → `.S` kernel with inline bias+Relu. Tail cases
/// (`mr < 8` rows, `nr < NR12` cols) fall back to a scalar micro-kernel
/// that also consumes the NR=12 packed layout — each scalar call operates
/// within one packed panel, never crossing the 12-col boundary.
///
/// Phase 0.3: earlier code did 8×12 over NR=8 packing, which broke every
/// second iteration because `jr += 12` misaligned against 8-col panels.
/// Fixed by introducing NR=12 packing (in this file) and reusing the
/// original 8×12 `.S` kernel unchanged except for a single B-panel
/// argument.
#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "neon")]
unsafe fn gebp_kernel_raw_mr8(
    packed_a: *const f32,
    packed_b: *const f32, // NR=12-packed (see pack_b_panel_nr12)
    output: *mut f32,
    n: usize,
    ic: usize,
    jc: usize,
    mc: usize,
    nc: usize,
    kc: usize,
    accumulate: bool,
    epilogue: GemmEpilogue,
    is_last_k: bool,
) {
    const MR8: usize = 8;
    const NR12: usize = 12;
    let has_residual = epilogue.residual.is_some();

    let activation_id: usize = match epilogue.activation {
        Activation::None => 0,
        Activation::Relu => 1,
        Activation::Silu => 0, // caller routes Silu elsewhere; treat as None here
    };
    // Residual is applied after asm writes the tile, so asm itself must skip
    // activation when residual is present to preserve
    // activation(acc + bias + residual) order.
    let activation_id_asm = if has_residual { 0 } else { activation_id };
    let is_last_k_flag: usize = if is_last_k { 1 } else { 0 };

    let mut jr = 0usize;
    while jr < nc {
        let nr = NR12.min(nc - jr);
        let b_off = (jr / NR12) * kc * NR12;
        let col_offset = jc + jr;

        if nr == NR12 {
            // Full 8×12 fast path.
            for ir in (0..mc).step_by(MR8) {
                let mr = MR8.min(mc - ir);
                let a_off = (ir / MR8) * kc * MR8;
                let c_ptr = output.add((ic + ir) * n + col_offset);
                if mr == MR8 {
                    let bias_ptr: *const f32 = match epilogue.bias {
                        Some(b) => b.add(col_offset),
                        None => std::ptr::null(),
                    };
                    if accumulate {
                        sgemm_asm_aarch64::yscv_sgemm_8x12_neon_acc(
                            packed_a.add(a_off),
                            packed_b.add(b_off),
                            c_ptr,
                            n,
                            kc,
                            bias_ptr,
                            activation_id_asm,
                            is_last_k_flag,
                        );
                    } else {
                        sgemm_asm_aarch64::yscv_sgemm_8x12_neon_set(
                            packed_a.add(a_off),
                            packed_b.add(b_off),
                            c_ptr,
                            n,
                            kc,
                            bias_ptr,
                            activation_id_asm,
                            is_last_k_flag,
                        );
                    }
                    // Residual for full 8x12 tiles: apply as a compact row pass
                    // after asm writes acc(+bias), then apply activation.
                    if has_residual
                        && is_last_k
                        && let Some(residual_base) = epilogue.residual
                    {
                        let activation_id_row: u8 = match epilogue.activation {
                            Activation::None => 0,
                            Activation::Relu => 1,
                            Activation::Silu => 2,
                        };
                        for row in 0..MR8 {
                            let out_row = std::slice::from_raw_parts_mut(c_ptr.add(row * n), NR12);
                            let residual_row = std::slice::from_raw_parts(
                                residual_base.add((ic + ir + row) * n + col_offset),
                                NR12,
                            );
                            super::simd::fused_row_epilogue_dispatch(
                                out_row,
                                Some(residual_row),
                                None,
                                activation_id_row,
                                NR12,
                            );
                        }
                    }
                } else {
                    // Partial mr < 8: NR=12-layout-aware scalar.
                    microkernel_scalar_nr12_partial_mr8(
                        packed_a.add(a_off),
                        packed_b.add(b_off),
                        c_ptr,
                        n,
                        mr,
                        NR12,
                        kc,
                        accumulate,
                        epilogue,
                        is_last_k,
                        col_offset,
                        epilogue.residual.map(|r| r.add((ic + ir) * n + col_offset)),
                    );
                }
            }
            jr += NR12;
            continue;
        }

        // Col tail (nr < 12). Fast-path the common `nr == 8` case through
        // the 8x12 asm kernel + compact copy-back, otherwise fall back to
        // the NR=12-layout-aware scalar microkernel.
        if nr == NR && !has_residual && mr8_tail8_asm_enabled() {
            for ir in (0..mc).step_by(MR8) {
                let mr = MR8.min(mc - ir);
                let a_off = (ir / MR8) * kc * MR8;
                let c_ptr = output.add((ic + ir) * n + col_offset);
                if mr == MR8 {
                    let mut tmp = [0.0f32; MR8 * NR12];
                    if accumulate {
                        // Seed tmp with the current C values in the valid 8
                        // cols; the extra 4 cols stay zero-padded.
                        for row in 0..MR8 {
                            let src = std::slice::from_raw_parts(c_ptr.add(row * n), NR);
                            let dst = std::slice::from_raw_parts_mut(
                                tmp.as_mut_ptr().add(row * NR12),
                                NR12,
                            );
                            dst[..NR].copy_from_slice(src);
                        }
                    }
                    // Bias/activation are handled below over the real 8-cols
                    // slice so asm never reads past the model's bias tail.
                    if accumulate {
                        sgemm_asm_aarch64::yscv_sgemm_8x12_neon_acc(
                            packed_a.add(a_off),
                            packed_b.add(b_off),
                            tmp.as_mut_ptr(),
                            NR12,
                            kc,
                            std::ptr::null(),
                            0,
                            if is_last_k { 0 } else { 1 },
                        );
                    } else {
                        sgemm_asm_aarch64::yscv_sgemm_8x12_neon_set(
                            packed_a.add(a_off),
                            packed_b.add(b_off),
                            tmp.as_mut_ptr(),
                            NR12,
                            kc,
                            std::ptr::null(),
                            0,
                            if is_last_k { 0 } else { 1 },
                        );
                    }

                    let activation_id_row: u8 = match epilogue.activation {
                        Activation::None => 0,
                        Activation::Relu => 1,
                        Activation::Silu => 2,
                    };
                    for row in 0..MR8 {
                        let tmp_row =
                            std::slice::from_raw_parts_mut(tmp.as_mut_ptr().add(row * NR12), NR12);
                        let out_row = std::slice::from_raw_parts_mut(c_ptr.add(row * n), NR);
                        if is_last_k {
                            let bias_slice = epilogue
                                .bias
                                .map(|b| std::slice::from_raw_parts(b.add(col_offset), NR));
                            if bias_slice.is_some() || activation_id_row != 0 {
                                super::simd::fused_row_epilogue_dispatch(
                                    &mut tmp_row[..NR],
                                    None,
                                    bias_slice,
                                    activation_id_row,
                                    NR,
                                );
                            }
                        }
                        out_row.copy_from_slice(&tmp_row[..NR]);
                    }
                } else {
                    microkernel_scalar_nr12_partial_mr8(
                        packed_a.add(a_off),
                        packed_b.add(b_off),
                        c_ptr,
                        n,
                        mr,
                        nr,
                        kc,
                        accumulate,
                        epilogue,
                        is_last_k,
                        col_offset,
                        None,
                    );
                }
            }
            jr += nr;
            continue;
        }

        // Generic col tail.
        for ir in (0..mc).step_by(MR8) {
            let mr = MR8.min(mc - ir);
            let a_off = (ir / MR8) * kc * MR8;
            let c_ptr = output.add((ic + ir) * n + col_offset);
            microkernel_scalar_nr12_partial_mr8(
                packed_a.add(a_off),
                packed_b.add(b_off),
                c_ptr,
                n,
                mr,
                nr,
                kc,
                accumulate,
                epilogue,
                is_last_k,
                col_offset,
                epilogue.residual.map(|r| r.add((ic + ir) * n + col_offset)),
            );
        }
        jr += nr;
    }
}

// ---------------------------------------------------------------------------
// MR=6 / 6×16 x86 AVX+FMA blocked-GEMM path (Step C1)
// ---------------------------------------------------------------------------

/// Step C1: 6×16 GEBP kernel for x86 AVX2. Iterates `jr` in 2×NR=16 col
/// strides, calls `microkernel_6x16_avx_fma` for each MR=6 × 16 tile.
/// Tail handling:
/// - `mr < MR6`: partial-row tail uses MR=4 microkernel on the top
///   4 rows + scalar for the remaining 1-2 rows via
///   `microkernel_scalar_partial`.
/// - `nr < 2*NR`: fall back to single NR=8 microkernel or scalar.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx,fma")]
unsafe fn gebp_kernel_raw_mr6(
    packed_a: *const f32,
    packed_b: *const f32,
    output: *mut f32,
    n: usize,
    ic: usize,
    jc: usize,
    mc: usize,
    nc: usize,
    kc: usize,
    accumulate: bool,
    epilogue: GemmEpilogue,
    is_last_k: bool,
) {
    let mut jr = 0usize;
    while jr < nc {
        let nr = NR.min(nc - jr);
        let b_off = (jr / NR) * kc * NR;
        let col_offset = jc + jr;

        // 6×16 fast path: two full NR=8 panels, 6 full rows.
        if nr == NR && jr + 2 * NR <= nc {
            let b_off_1 = ((jr + NR) / NR) * kc * NR;
            for ir in (0..mc).step_by(MR6) {
                let mr = MR6.min(mc - ir);
                let a_off = (ir / MR6) * kc * MR6;
                let c_ptr = output.add((ic + ir) * n + col_offset);
                let residual_tile = epilogue.residual.map(|r| r.add((ic + ir) * n + col_offset));
                if mr == MR6 {
                    microkernel_6x16_avx_fma(
                        packed_a.add(a_off),
                        packed_b.add(b_off),
                        packed_b.add(b_off_1),
                        c_ptr,
                        n,
                        kc,
                        accumulate,
                        epilogue,
                        is_last_k,
                        col_offset,
                        residual_tile,
                    );
                } else {
                    // Partial rows (mr < 6): scalar with MR=6 stride,
                    // 16 cols (two NR=8 panels side by side).
                    microkernel_scalar_partial_stride(
                        packed_a.add(a_off),
                        packed_b.add(b_off),
                        c_ptr,
                        n,
                        mr,
                        NR,
                        kc,
                        MR6,
                        accumulate,
                        epilogue,
                        is_last_k,
                        col_offset,
                        residual_tile,
                    );
                    microkernel_scalar_partial_stride(
                        packed_a.add(a_off),
                        packed_b.add(b_off_1),
                        c_ptr.add(NR),
                        n,
                        mr,
                        NR,
                        kc,
                        MR6,
                        accumulate,
                        epilogue,
                        is_last_k,
                        col_offset + NR,
                        residual_tile.map(|r| r.add(NR)),
                    );
                }
            }
            jr += 2 * NR;
            continue;
        }

        // Tail path: scalar with MR=6 stride for remaining nr columns.
        for ir in (0..mc).step_by(MR6) {
            let mr = MR6.min(mc - ir);
            let a_off = (ir / MR6) * kc * MR6;
            let c_ptr = output.add((ic + ir) * n + col_offset);
            let residual_tile = epilogue.residual.map(|r| r.add((ic + ir) * n + col_offset));
            microkernel_scalar_partial_stride(
                packed_a.add(a_off),
                packed_b.add(b_off),
                c_ptr,
                n,
                mr,
                nr,
                kc,
                MR6,
                accumulate,
                epilogue,
                is_last_k,
                col_offset,
                residual_tile,
            );
        }
        jr += NR;
    }
}

/// Step C1: sequential blocked GEMM using MR=6 pack + 6×16 microkernel.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code)]
fn blocked_gemm_sequential_mr6(
    left: &[f32],
    right: &[f32],
    output: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    epilogue: GemmEpilogue,
    pre_packed_b: Option<&PackedB>,
) {
    let a_size = div_ceil(MC_MR6, MR6) * KC * MR6;
    // Zero-filled A-pack scratch; `pack_a_panel_mr6` overwrites every slot.
    let mut packed_a: Vec<f32> = vec![0.0; a_size];

    let rc_fallback;
    let packed_b_full: &PackedB = if let Some(pb) = pre_packed_b {
        pb
    } else {
        rc_fallback = get_or_pack_b(right, k, n);
        &rc_fallback
    };

    for jc in (0..n).step_by(NC) {
        let jc_idx = jc / NC;
        let nc = NC.min(n - jc);
        for pc in (0..k).step_by(KC) {
            let pc_idx = pc / KC;
            let kc = KC.min(k - pc);
            let accumulate = pc > 0;
            let is_last_k = pc + kc >= k;

            let block_off = (pc_idx * packed_b_full.num_jc + jc_idx) * packed_b_full.block_slots;

            for ic in (0..m).step_by(MC_MR6) {
                let mc = MC_MR6.min(m - ic);
                pack_a_panel_mr6(left, k, ic, mc, pc, kc, &mut packed_a);
                // SAFETY: packed_b_full owned across call; rows
                // ic..ic+mc of `output` are disjoint within this
                // sequential loop.
                #[allow(unsafe_code)]
                unsafe {
                    gebp_kernel_raw_mr6(
                        packed_a.as_ptr(),
                        packed_b_full.data.as_ptr().add(block_off),
                        output.as_mut_ptr(),
                        n,
                        ic,
                        jc,
                        mc,
                        nc,
                        kc,
                        accumulate,
                        epilogue,
                        is_last_k,
                    );
                }
            }
        }
    }
}

/// Step C1: parallel blocked GEMM using MR=6 pack + 6×16 microkernel.
/// Mirrors the existing `blocked_gemm_parallel` structure (MR=4) but
/// uses MR=6 stride throughout. Per-thread packed_a scratch reused via
/// `with_packed_a_tls`.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code)]
fn blocked_gemm_parallel_mr6(
    left: &[f32],
    right: &[f32],
    output: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    epilogue: GemmEpilogue,
    thread_pool: Option<&ThreadPool>,
    pre_packed_b: Option<&PackedB>,
) {
    let out_ptr = SendPtr(output.as_mut_ptr());

    let rc_fallback;
    let packed_b_full: &PackedB = if let Some(pb) = pre_packed_b {
        pb
    } else {
        rc_fallback = get_or_pack_b(right, k, n);
        &rc_fallback
    };
    // Hold raw pointer — `PackedB` is Send/Sync by construction, but
    // we need a fn-level reference independent of the `Rc` guard.
    let packed_b_ptr = packed_b_full.data.as_ptr() as usize;
    let block_slots = packed_b_full.block_slots;
    let num_jc = packed_b_full.num_jc;

    let ic_blocks: Vec<usize> = (0..m).step_by(MC_PARALLEL_MR6).collect();

    let work = || {
        for jc in (0..n).step_by(NC) {
            let jc_idx = jc / NC;
            let nc = NC.min(n - jc);
            for pc in (0..k).step_by(KC) {
                let pc_idx = pc / KC;
                let kc = KC.min(k - pc);
                let accumulate = pc > 0;
                let is_last_k = pc + kc >= k;
                let block_off = (pc_idx * num_jc + jc_idx) * block_slots;
                let out_p = &out_ptr;

                par_for_each_ic(&ic_blocks, thread_pool, |ic| {
                    let mc = MC_PARALLEL_MR6.min(m - ic);
                    let a_panels = div_ceil(mc, MR6);
                    let pa_size = a_panels * kc * MR6;
                    with_packed_a_tls(pa_size, |packed_a| {
                        pack_a_panel_mr6(left, k, ic, mc, pc, kc, packed_a);
                        // SAFETY: `packed_b_ptr` points to `packed_b_full.data`
                        // which outlives the parallel scope (owned by `rc_fallback`
                        // or caller); disjoint IC blocks → disjoint output rows.
                        #[allow(unsafe_code)]
                        unsafe {
                            gebp_kernel_raw_mr6(
                                packed_a.as_ptr(),
                                (packed_b_ptr as *const f32).add(block_off),
                                out_p.0,
                                n,
                                ic,
                                jc,
                                mc,
                                nc,
                                kc,
                                accumulate,
                                epilogue,
                                is_last_k,
                            );
                        }
                    });
                });
            }
        }
    };

    if let Some(pool) = thread_pool {
        pool.install(work);
    } else {
        work();
    }
}

/// Parallel blocked GEMM: parallelizes over IC (row) blocks.
///
/// Each thread gets its own packed_a buffer. Packed B is either the caller's
/// load-time pre-packed kernel or falls back to the thread-local runtime cache.
#[allow(unsafe_code)]
fn blocked_gemm_parallel(
    left: &[f32],
    right: &[f32],
    output: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    epilogue: GemmEpilogue,
    thread_pool: Option<&ThreadPool>,
    pre_packed_b: Option<&PackedB>,
) {
    // SAFETY: We split output into disjoint row ranges per IC block.
    // Different IC blocks write to non-overlapping rows.
    let out_ptr = SendPtr(output.as_mut_ptr());

    let rc_fallback;
    let packed_b_full: &PackedB = if let Some(pb) = pre_packed_b {
        pb
    } else {
        rc_fallback = get_or_pack_b(right, k, n);
        &rc_fallback
    };
    let packed_b_ptr = packed_b_full.data.as_ptr() as usize;
    let block_slots = packed_b_full.block_slots;
    let num_jc = packed_b_full.num_jc;

    // Use the smaller MC_PARALLEL for MT parallelism to get enough tiles to
    // saturate thread pools. Each worker still runs the same gebp_kernel_raw,
    // just with smaller mc per call — cache reuse of A trades for better
    // work distribution.
    let ic_blocks: Vec<usize> = (0..m).step_by(MC_PARALLEL).collect();

    let work = || {
        for jc in (0..n).step_by(NC) {
            let jc_idx = jc / NC;
            let nc = NC.min(n - jc);
            for pc in (0..k).step_by(KC) {
                let pc_idx = pc / KC;
                let kc = KC.min(k - pc);
                let accumulate = pc > 0;
                let is_last_k = pc + kc >= k;

                let block_off = (pc_idx * num_jc + jc_idx) * block_slots;
                let out_p = &out_ptr;

                par_for_each_ic(&ic_blocks, thread_pool, |ic| {
                    let mc = MC_PARALLEL.min(m - ic);
                    let a_panels = div_ceil(mc, MR);
                    let pa_size = a_panels * kc * MR;
                    with_packed_a_tls(pa_size, |packed_a| {
                        pack_a_panel(left, k, ic, mc, pc, kc, packed_a);
                        // SAFETY: packed_b_full outlives `work` (held via Rc);
                        // each IC block writes to rows ic..ic+mc (disjoint).
                        #[allow(unsafe_code)]
                        unsafe {
                            gebp_kernel_raw(
                                packed_a.as_ptr(),
                                (packed_b_ptr as *const f32).add(block_off),
                                out_p.0,
                                n,
                                ic,
                                jc,
                                mc,
                                nc,
                                kc,
                                accumulate,
                                epilogue,
                                is_last_k,
                            );
                        }
                    });
                });
            }
        }
    };

    if let Some(pool) = thread_pool {
        pool.install(work);
    } else {
        work();
    }
}

struct SendPtr(*mut f32);
// SAFETY: We ensure disjoint access per thread at the IC block level.
#[allow(unsafe_code)]
unsafe impl Send for SendPtr {}
#[allow(unsafe_code)]
unsafe impl Sync for SendPtr {}

// ---------------------------------------------------------------------------
// GEBP kernel: micro-kernel loop over one MC×NC tile
// ---------------------------------------------------------------------------

/// Process one MC×NC block using packed_a and packed_b.
#[allow(unsafe_code)]
fn gebp_kernel(
    packed_a: &[f32],
    packed_b: &[f32],
    output: &mut [f32],
    n: usize,
    ic: usize,
    jc: usize,
    mc: usize,
    nc: usize,
    kc: usize,
    accumulate: bool,
    epilogue: GemmEpilogue,
    is_last_k: bool,
) {
    // SAFETY: output has m*n elements, ic+mc <= m, jc+nc <= n.
    unsafe {
        gebp_kernel_raw(
            packed_a.as_ptr(),
            packed_b.as_ptr(),
            output.as_mut_ptr(),
            n,
            ic,
            jc,
            mc,
            nc,
            kc,
            accumulate,
            epilogue,
            is_last_k,
        );
    }
}

/// Singleton `YscvPool` activated via `YSCV_POOL=yscv`. Lazily constructed
/// with the same worker count as `rayon::current_num_threads()`; affinity
/// + spin defaults come from `yscv-threadpool`'s env vars. When unset or
///   `YSCV_POOL=rayon`, returns `None` and callers use rayon's par_iter.
///
/// This is the Phase A.5 integration path — minimum-invasive: only the
/// two hot-path `par_iter().for_each(|&ic| ...)` sites in the parallel
/// blocked-GEMM variants check this once and route accordingly. The 37
/// other `thread_pool: Option<&ThreadPool>` sites stay unchanged; they
/// are cold-path fallbacks (row_gemm, BLAS), where pool overhead is
/// drowned out by the actual kernel time anyway.
fn yscv_pool_singleton() -> Option<&'static yscv_threadpool::YscvPool> {
    use std::sync::OnceLock;
    static CACHED: OnceLock<Option<yscv_threadpool::YscvPool>> = OnceLock::new();
    CACHED
        .get_or_init(|| {
            if std::env::var("YSCV_POOL").as_deref() == Ok("yscv") {
                let n = rayon::current_num_threads().max(1);
                yscv_threadpool::YscvPool::new(n).ok()
            } else {
                None
            }
        })
        .as_ref()
}

/// Dispatch helper: run `f(ic)` for each entry of `ic_blocks` in parallel
/// through the currently-installed `ParallelScope` (see `scope_ctx`).
/// When no runner has installed a scope (benches, unit tests), falls back
/// to rayon's `par_iter` — zero overhead on the default path.
///
/// The legacy `yscv_pool_singleton` env-gated routing is kept as a
/// secondary dispatch so sites that operate outside the runner's scope
/// (e.g. when called from a non-ONNX consumer) can still pick up
/// `YSCV_POOL=yscv` — otherwise defeats the purpose of the env flag.
#[inline]
fn par_for_each_ic<F>(ic_blocks: &[usize], thread_pool: Option<&ThreadPool>, f: F)
where
    F: Fn(usize) + Send + Sync,
{
    // Preferred path: runner-installed scope (covers both rayon and yscv
    // backends depending on `YSCV_POOL`).
    let routed = super::super::scope_ctx::with_scope(|scope| {
        if let Some(s) = scope {
            s.par_for_each_index(ic_blocks.len(), &|idx| f(ic_blocks[idx]));
            true
        } else {
            false
        }
    });
    if routed {
        return;
    }
    // Fallback #1: process-global YscvPool singleton (for non-runner
    // consumers that still want the spin-idle pool).
    if let Some(yscv) = yscv_pool_singleton() {
        yscv.par_for_each_index(ic_blocks.len(), |idx| f(ic_blocks[idx]));
        return;
    }
    // Fallback #2: rayon — kernel unit tests and criterion benches.
    let work = || {
        ic_blocks.par_iter().for_each(|&ic| f(ic));
    };
    if let Some(pool) = thread_pool {
        pool.install(work);
    } else {
        work();
    }
}

/// Cached detector for the aarch64 NEON MR=8 / 8×12 fast path. NEON is
/// mandatory on aarch64, so this is `true` unless `YSCV_NO_MR8=1` is set.
#[cfg(target_arch = "aarch64")]
fn mr8_enabled() -> bool {
    use std::sync::OnceLock;
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| {
        if std::env::var_os("YSCV_NO_MR8").is_some() {
            return false;
        }
        std::arch::is_aarch64_feature_detected!("neon")
    })
}

/// Enable residual-aware MR8 dispatch on aarch64.
///
/// EXPERIMENTAL and opt-in: set `YSCV_MR8_RESIDUAL=1`.
/// The 8x12 asm kernels still don't consume residual directly, so residual is
/// applied as a compact post-pass on each full tile at `is_last_k`.
#[cfg(target_arch = "aarch64")]
#[inline]
fn mr8_residual_enabled() -> bool {
    use std::sync::OnceLock;
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| {
        matches!(
            std::env::var_os("YSCV_MR8_RESIDUAL").as_deref(),
            Some(v) if v == "1"
        )
    })
}

/// Enable MR8 tail-8 asm path on aarch64 (`nr == 8` in the NR=12 kernel).
/// This avoids the scalar tail for common channel counts (e.g. n=32).
#[cfg(target_arch = "aarch64")]
#[inline]
fn mr8_tail8_asm_enabled() -> bool {
    use std::sync::OnceLock;
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| std::env::var_os("YSCV_NO_MR8_TAIL8_ASM").is_none())
}

/// Step C1: Cached detector for the x86 AVX2 MR=6×16 fast path.
/// **Default OFF on Zen 4** — sgemm A/B (2026-04-19) showed the pure-
/// intrinsics 6×16 kernel regresses 1.7-2.3× vs the inline-asm 4×16/
/// 4×24 paths on hot pointwise shapes. The MR=4 kernels use inline
/// `asm!` with double-buffered B loads (preload B[k+1] during k's FMAs)
/// which LLVM can't replicate from pure intrinsics. Tracker 6T p50
/// regressed ~940 µs (3 450 → 4 390) at ON.
///
/// Kept as opt-in via `YSCV_MR6=1` for:
/// - Future inline-asm rewrite of the 6×16 k-loop with double-buffering
/// - AVX-512 12×32 extension that could still win on Intel SPR / Zen 5
/// - A/B measurement on microarchs with different register pressure
///
/// See `project_step_c1_mr6_kernel_landed.md` memory for findings.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn mr6_enabled() -> bool {
    use std::sync::OnceLock;
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| {
        let opt_in = matches!(
            std::env::var_os("YSCV_MR6").as_deref(),
            Some(v) if v == "1"
        );
        opt_in && std::is_x86_feature_detected!("fma") && std::is_x86_feature_detected!("avx")
    })
}

/// Opt-in gate for the AVX-512 Conv_Relu path. On Zen 4 the double-pumped
/// 512-bit FMA is actually slower than AVX2 4×24 for Conv_Relu, so the
/// relaxed gate regresses. On Intel Sapphire Rapids / Zen 5 (real 512-bit
/// FMA) this should win; set `YSCV_AVX512_RELU=1` to enable there.
#[cfg(all(target_arch = "x86_64", any(target_os = "linux", target_os = "macos")))]
fn avx512_relu_enabled() -> bool {
    use std::sync::OnceLock;
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| std::env::var_os("YSCV_AVX512_RELU").is_some())
}

/// Cached detector for the 4×24 AVX2 pure-.S kernel (Phase B). Default
/// OFF on Zen 4: sgemm A/B vs the intrinsics `microkernel_4x24_avx_fma`
/// showed the two paths are within ±17% per-shape noise, and tracker 6T
/// p50 lost 25 µs at ASM-ON — the intrinsics' inline `asm!` k-loop + LLVM
/// reg-alloc was already saturating the 2-FMA port pipeline without spills.
/// Opt-in via `YSCV_ASM_GEMM=1` for A/B measurement on other microarchs
/// (Intel Sapphire Rapids, Zen 5) where the ASM path may win.
#[cfg(all(
    target_arch = "x86_64",
    any(
        target_os = "linux",
        target_os = "macos",
        all(target_os = "windows", not(target_env = "msvc"))
    )
))]
fn asm_4x24_enabled() -> bool {
    use std::sync::OnceLock;
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| {
        let opt_in = matches!(
            std::env::var_os("YSCV_ASM_GEMM").as_deref(),
            Some(v) if v == "1"
        );
        opt_in && std::is_x86_feature_detected!("fma") && std::is_x86_feature_detected!("avx")
    })
}

/// Cached detector for the AVX-512F fast path. CPUID runs once and the
/// `YSCV_NO_AVX512` env var is consulted on first call — subsequent calls
/// hit a relaxed atomic load.
#[cfg(all(target_arch = "x86_64", any(target_os = "linux", target_os = "macos")))]
fn avx512_enabled() -> bool {
    use std::sync::OnceLock;
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| {
        if std::env::var_os("YSCV_NO_AVX512").is_some() {
            return false;
        }
        std::is_x86_feature_detected!("avx512f")
    })
}

/// Raw pointer version for use from parallel context.
///
/// # Safety
/// - `packed_a` must have at least `div_ceil(mc, MR) * kc * MR` elements.
/// - `packed_b` must have at least `div_ceil(nc, NR) * kc * NR` elements.
/// - `output` must point to a buffer with at least `(ic + mc) * n` elements.
/// - Caller must ensure no concurrent writes to the same output rows.
#[allow(unsafe_code)]
#[allow(unsafe_op_in_unsafe_fn)]
#[cfg_attr(
    any(target_arch = "x86", target_arch = "x86_64"),
    target_feature(enable = "avx,fma")
)]
#[cfg_attr(target_arch = "aarch64", target_feature(enable = "neon"))]
unsafe fn gebp_kernel_raw(
    packed_a: *const f32,
    packed_b: *const f32,
    output: *mut f32,
    n: usize,
    ic: usize,
    jc: usize,
    mc: usize,
    nc: usize,
    kc: usize,
    accumulate: bool,
    epilogue: GemmEpilogue,
    is_last_k: bool,
) {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    let use_4x16 = std::is_x86_feature_detected!("fma") && std::is_x86_feature_detected!("avx");
    #[cfg(target_arch = "aarch64")]
    let use_4x16 = std::arch::is_aarch64_feature_detected!("neon");
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
    let use_4x16 = false;

    // AVX-512 detection — cached (CPUID once, then atomic-load hot path).
    // Env override `YSCV_NO_AVX512=1` disables for A/B benchmarking.
    #[cfg(all(target_arch = "x86_64", any(target_os = "linux", target_os = "macos")))]
    let use_avx512 = avx512_enabled();
    #[cfg(not(all(target_arch = "x86_64", any(target_os = "linux", target_os = "macos"))))]
    let use_avx512 = false;
    let _ = use_avx512;

    let mut jr = 0usize;
    while jr < nc {
        let nr = NR.min(nc - jr);
        let b_off = (jr / NR) * kc * NR;
        let col_offset = jc + jr;

        // AVX-512 4×32 fast path. The `.S` kernel now has INLINE bias+Relu
        // epilogue (Phase 2.F) so it CAN handle Conv_Relu — but on Zen 4
        // the AVX-512 backend is double-pumped and measured 9% slower than
        // our 4×24 AVX2 intrinsic at Conv_Relu shapes. We keep the gate at
        // IDENTITY only; on true-AVX-512 hardware (Intel Sapphire Rapids,
        // Zen 5 with non-double-pumped 512-bit FMA) a relaxed gate
        // `!matches!(epilogue.activation, Activation::Silu)` would win.
        // Env override `YSCV_AVX512_RELU=1` forces the relaxed gate for
        // hand measurement / Intel validation.
        #[cfg(all(target_arch = "x86_64", any(target_os = "linux", target_os = "macos")))]
        if use_avx512
            && nr == NR
            && jr + 4 * NR <= nc
            && ((epilogue.bias.is_none() && matches!(epilogue.activation, Activation::None))
                || (avx512_relu_enabled() && !matches!(epilogue.activation, Activation::Silu)))
        {
            let b_off = (jr / NR) * kc * NR;
            let activation_id: usize = match epilogue.activation {
                Activation::None => 0,
                Activation::Relu => 1,
                Activation::Silu => 0, // unreachable (outer gate)
            };
            let is_last_k_flag: usize = if is_last_k { 1 } else { 0 };
            for ir in (0..mc).step_by(MR) {
                let mr = MR.min(mc - ir);
                let a_off = (ir / MR) * kc * MR;
                let c_ptr = output.add((ic + ir) * n + jc + jr);
                if mr == MR {
                    let bias_ptr: *const f32 = match epilogue.bias {
                        Some(b) => b.add(jc + jr),
                        None => std::ptr::null(),
                    };
                    if accumulate {
                        sgemm_asm_avx512::yscv_sgemm_4x32_avx512_acc(
                            packed_a.add(a_off),
                            packed_b.add(b_off),
                            c_ptr,
                            n,
                            kc,
                            bias_ptr,
                            activation_id,
                            is_last_k_flag,
                        );
                    } else {
                        sgemm_asm_avx512::yscv_sgemm_4x32_avx512_set(
                            packed_a.add(a_off),
                            packed_b.add(b_off),
                            c_ptr,
                            n,
                            kc,
                            bias_ptr,
                            activation_id,
                            is_last_k_flag,
                        );
                    }
                } else {
                    // Partial-row tail: fall back to four scalar NR-panel calls.
                    for panel in 0..4 {
                        let poff = ((jr + panel * NR) / NR) * kc * NR;
                        microkernel_scalar_partial(
                            packed_a.add(a_off),
                            packed_b.add(poff),
                            c_ptr.add(panel * NR),
                            n,
                            mr,
                            NR,
                            kc,
                            accumulate,
                            epilogue,
                            is_last_k,
                            jc + jr + panel * NR,
                            epilogue
                                .residual
                                .map(|r| r.add((ic + ir) * n + jc + jr + panel * NR)),
                        );
                    }
                }
            }
            jr += 4 * NR;
            continue;
        }

        // Try tripled 4×24 when three full NR=8 tiles are available.
        #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
        if use_4x16 && nr == NR && jr + 3 * NR <= nc {
            let b_off_1 = ((jr + NR) / NR) * kc * NR;
            let b_off_2 = ((jr + 2 * NR) / NR) * kc * NR;
            for ir in (0..mc).step_by(MR) {
                let mr = MR.min(mc - ir);
                let a_off = (ir / MR) * kc * MR;
                let c_ptr = output.add((ic + ir) * n + col_offset);
                // Per-tile residual pointer — same row/col math as output.
                // x86 4×24 tile uses `residual_tile` for in-kernel residual
                // fold; aarch64 NEON 4×24 still routes residual via the
                // matmul guard (falls back to row_gemm) — unused there.
                #[cfg_attr(not(target_arch = "x86_64"), allow(unused_variables))]
                let residual_tile = epilogue.residual.map(|r| r.add((ic + ir) * n + col_offset));
                if mr == MR {
                    // Windows-MSVC has no `.S` kernel linked (cl.exe can't
                    // speak GAS); take the intrinsics 4×24 path directly so
                    // `link.exe` doesn't hunt for `yscv_sgemm_4x24_avx2_*`.
                    #[cfg(all(target_arch = "x86_64", target_os = "windows", target_env = "msvc"))]
                    {
                        microkernel_4x24_avx_fma(
                            packed_a.add(a_off),
                            packed_b.add(b_off),
                            packed_b.add(b_off_1),
                            packed_b.add(b_off_2),
                            c_ptr,
                            n,
                            kc,
                            accumulate,
                            epilogue,
                            is_last_k,
                            col_offset,
                            residual_tile,
                        );
                    }

                    #[cfg(all(
                        target_arch = "x86_64",
                        not(all(target_os = "windows", target_env = "msvc"))
                    ))]
                    {
                        // Phase B pure-.S path: hand-scheduled 4×24 with fused
                        // bias + residual + Relu epilogue. SiLU still routes
                        // to intrinsics (bit-trick SiLU not yet in .S).
                        let use_asm =
                            asm_4x24_enabled() && !matches!(epilogue.activation, Activation::Silu);
                        if use_asm {
                            let activation_id: usize = match epilogue.activation {
                                Activation::Relu => 1,
                                _ => 0,
                            };
                            let is_last_k_flag: usize = if is_last_k { 1 } else { 0 };
                            let bias_ptr: *const f32 = match epilogue.bias {
                                Some(b) if is_last_k => b.add(col_offset),
                                _ => std::ptr::null(),
                            };
                            let residual_ptr: *const f32 = match residual_tile {
                                Some(r) if is_last_k => r,
                                _ => std::ptr::null(),
                            };
                            if accumulate {
                                sgemm_asm::yscv_sgemm_4x24_avx2_acc_fused(
                                    packed_a.add(a_off),
                                    packed_b.add(b_off),
                                    packed_b.add(b_off_1),
                                    packed_b.add(b_off_2),
                                    c_ptr,
                                    n,
                                    kc,
                                    bias_ptr,
                                    residual_ptr,
                                    activation_id,
                                    is_last_k_flag,
                                );
                            } else {
                                sgemm_asm::yscv_sgemm_4x24_avx2_set_fused(
                                    packed_a.add(a_off),
                                    packed_b.add(b_off),
                                    packed_b.add(b_off_1),
                                    packed_b.add(b_off_2),
                                    c_ptr,
                                    n,
                                    kc,
                                    bias_ptr,
                                    residual_ptr,
                                    activation_id,
                                    is_last_k_flag,
                                );
                            }
                        } else {
                            microkernel_4x24_avx_fma(
                                packed_a.add(a_off),
                                packed_b.add(b_off),
                                packed_b.add(b_off_1),
                                packed_b.add(b_off_2),
                                c_ptr,
                                n,
                                kc,
                                accumulate,
                                epilogue,
                                is_last_k,
                                col_offset,
                                residual_tile,
                            );
                        }
                    }
                    #[cfg(target_arch = "aarch64")]
                    {
                        // Fast path: hand-tuned `.S` kernel for pure GEMM (no
                        // bias/activation). Dispatch on epilogue shape, mirroring
                        // the x86_64 4×8 fast path.
                        if epilogue.bias.is_none()
                            && epilogue.residual.is_none()
                            && matches!(epilogue.activation, Activation::None)
                        {
                            if accumulate {
                                sgemm_asm_aarch64::yscv_sgemm_4x24_neon_acc(
                                    packed_a.add(a_off),
                                    packed_b.add(b_off),
                                    packed_b.add(b_off_1),
                                    packed_b.add(b_off_2),
                                    c_ptr,
                                    n,
                                    kc,
                                );
                            } else {
                                sgemm_asm_aarch64::yscv_sgemm_4x24_neon_set(
                                    packed_a.add(a_off),
                                    packed_b.add(b_off),
                                    packed_b.add(b_off_1),
                                    packed_b.add(b_off_2),
                                    c_ptr,
                                    n,
                                    kc,
                                );
                            }
                        } else {
                            microkernel_4x24_neon(
                                packed_a.add(a_off),
                                packed_b.add(b_off),
                                packed_b.add(b_off_1),
                                packed_b.add(b_off_2),
                                c_ptr,
                                n,
                                kc,
                                accumulate,
                                epilogue,
                                is_last_k,
                                col_offset,
                                residual_tile,
                            );
                        }
                    }
                } else {
                    for panel in 0..3 {
                        let poff = [b_off, b_off_1, b_off_2][panel];
                        microkernel_scalar_partial(
                            packed_a.add(a_off),
                            packed_b.add(poff),
                            c_ptr.add(panel * NR),
                            n,
                            mr,
                            NR,
                            kc,
                            accumulate,
                            epilogue,
                            is_last_k,
                            col_offset + panel * NR,
                            residual_tile.map(|r| r.add(panel * NR)),
                        );
                    }
                }
            }
            jr += 3 * NR;
            continue;
        }

        // Try paired 4×16 when two full NR=8 tiles are available.
        #[cfg(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64"))]
        if use_4x16 && nr == NR && jr + 2 * NR <= nc {
            let b_off_1 = ((jr + NR) / NR) * kc * NR;
            for ir in (0..mc).step_by(MR) {
                let mr = MR.min(mc - ir);
                let a_off = (ir / MR) * kc * MR;
                let c_ptr = output.add((ic + ir) * n + col_offset);
                // x86 4×16 tile uses `residual_tile` for in-kernel residual
                // fold; aarch64 NEON 4×16 still routes residual via the
                // matmul guard (falls back to row_gemm) — unused there.
                #[cfg_attr(
                    not(any(target_arch = "x86", target_arch = "x86_64")),
                    allow(unused_variables)
                )]
                let residual_tile = epilogue.residual.map(|r| r.add((ic + ir) * n + col_offset));
                if mr == MR {
                    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                    microkernel_4x16_avx_fma(
                        packed_a.add(a_off),
                        packed_b.add(b_off),
                        packed_b.add(b_off_1),
                        c_ptr,
                        n,
                        kc,
                        accumulate,
                        epilogue,
                        is_last_k,
                        col_offset,
                        residual_tile,
                    );
                    #[cfg(target_arch = "aarch64")]
                    microkernel_4x16_neon(
                        packed_a.add(a_off),
                        packed_b.add(b_off),
                        packed_b.add(b_off_1),
                        c_ptr,
                        n,
                        kc,
                        accumulate,
                        epilogue,
                        is_last_k,
                        col_offset,
                        residual_tile,
                    );
                } else {
                    microkernel_scalar_partial(
                        packed_a.add(a_off),
                        packed_b.add(b_off),
                        c_ptr,
                        n,
                        mr,
                        NR,
                        kc,
                        accumulate,
                        epilogue,
                        is_last_k,
                        col_offset,
                        residual_tile,
                    );
                    microkernel_scalar_partial(
                        packed_a.add(a_off),
                        packed_b.add(b_off_1),
                        c_ptr.add(NR),
                        n,
                        mr,
                        NR,
                        kc,
                        accumulate,
                        epilogue,
                        is_last_k,
                        col_offset + NR,
                        residual_tile.map(|r| r.add(NR)),
                    );
                }
            }
            jr += 2 * NR;
            continue;
        }
        let _ = use_4x16;

        for ir in (0..mc).step_by(MR) {
            let mr = MR.min(mc - ir);
            let a_off = (ir / MR) * kc * MR;
            let c_ptr = output.add((ic + ir) * n + col_offset);
            #[cfg_attr(
                not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")),
                allow(unused_variables)
            )]
            let residual_tile = epilogue.residual.map(|r| r.add((ic + ir) * n + col_offset));

            if mr == MR && nr == NR {
                microkernel_4x8_dispatch(
                    packed_a.add(a_off),
                    packed_b.add(b_off),
                    c_ptr,
                    n,
                    kc,
                    accumulate,
                    epilogue,
                    is_last_k,
                    col_offset,
                    residual_tile,
                );
            } else {
                microkernel_scalar_partial(
                    packed_a.add(a_off),
                    packed_b.add(b_off),
                    c_ptr,
                    n,
                    mr,
                    nr,
                    kc,
                    accumulate,
                    epilogue,
                    is_last_k,
                    col_offset,
                    residual_tile,
                );
            }
        }
        jr += NR;
    }
}

// Micro-kernel dispatch
// ---------------------------------------------------------------------------

/// Dispatch MR×NR micro-kernel to best SIMD path.
///
/// Computes C[MR×NR] += A_packed[MR×kc] * B_packed[kc×NR].
///
/// # Safety
/// - `a_panel`: at least `kc * MR` elements (packed column-of-MR format).
/// - `b_panel`: at least `kc * NR` elements (packed row-of-NR format).
/// - `c`: points to C[ir, jr]; next row at `c + ldc`.
/// - `ldc`: row stride of full C matrix (= N).
#[allow(unsafe_code)]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn microkernel_4x8_dispatch(
    a_panel: *const f32,
    b_panel: *const f32,
    c: *mut f32,
    ldc: usize,
    kc: usize,
    accumulate: bool,
    epilogue: GemmEpilogue,
    is_last_k: bool,
    col_offset: usize,
    residual_tile: Option<*const f32>,
) {
    // Residual is fused by `microkernel_4x24_avx_fma` / `microkernel_4x16_avx_fma`
    // and by the scalar/NEON 4×8 variants. The x86 SIMD/ASM 4×8 paths below
    // (`sgemm_4x8_set/acc`, `microkernel_4x8_avx_fma`, `microkernel_4x8_avx`,
    // `microkernel_4x8_sse`) do NOT read `residual_tile`, so when the caller
    // hands in a residual we must skip them and fall through to the scalar
    // kernel to avoid silent-dropping the add. Gated on `is_last_k` because
    // residual is only applied on the final k-block.
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    let needs_scalar_for_residual = residual_tile.is_some() && is_last_k;

    // Fast path: hand-tuned `.S` microkernel for no-epilogue case (pure GEMM).
    // Separate SET/ACC entry points → no branch in hot path. Covers SysV
    // (Linux, macOS) and Win64 (Windows); the Win64 variant translates args
    // internally so call sites are ABI-agnostic.
    #[cfg(all(
        target_arch = "x86_64",
        any(
            target_os = "linux",
            target_os = "macos",
            all(target_os = "windows", not(target_env = "msvc"))
        )
    ))]
    if !needs_scalar_for_residual
        && epilogue.bias.is_none()
        && matches!(epilogue.activation, Activation::None)
        && std::is_x86_feature_detected!("fma")
        && std::is_x86_feature_detected!("avx")
    {
        if accumulate {
            sgemm_asm::yscv_sgemm_4x8_acc(a_panel, b_panel, c, ldc, kc);
        } else {
            sgemm_asm::yscv_sgemm_4x8_set(a_panel, b_panel, c, ldc, kc);
        }
        return;
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if !needs_scalar_for_residual {
        if std::is_x86_feature_detected!("fma") && std::is_x86_feature_detected!("avx") {
            microkernel_4x8_avx_fma(
                a_panel, b_panel, c, ldc, kc, accumulate, epilogue, is_last_k, col_offset,
            );
            return;
        }
        if std::is_x86_feature_detected!("avx") {
            microkernel_4x8_avx(
                a_panel, b_panel, c, ldc, kc, accumulate, epilogue, is_last_k, col_offset,
            );
            return;
        }
        if std::is_x86_feature_detected!("sse") {
            microkernel_4x8_sse(
                a_panel, b_panel, c, ldc, kc, accumulate, epilogue, is_last_k, col_offset,
            );
            return;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            microkernel_4x8_neon(
                a_panel,
                b_panel,
                c,
                ldc,
                kc,
                accumulate,
                epilogue,
                is_last_k,
                col_offset,
                residual_tile,
            );
            return;
        }
    }

    microkernel_4x8_scalar(
        a_panel,
        b_panel,
        c,
        ldc,
        kc,
        accumulate,
        epilogue,
        is_last_k,
        col_offset,
        residual_tile,
    );
}

// ---------------------------------------------------------------------------
// Epilogue helpers: apply bias + activation on register values before store
// ---------------------------------------------------------------------------

/// Scalar epilogue: bias + activation on a row slice of accumulator values.
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[inline(always)]
unsafe fn apply_epilogue_scalar(
    row: &mut [f32],
    epilogue: &GemmEpilogue,
    col_offset: usize,
    residual_tile_row: Option<*const f32>,
) {
    if let Some(bias) = epilogue.bias {
        for (j, v) in row.iter_mut().enumerate() {
            *v += *bias.add(col_offset + j);
        }
    }
    if let Some(res) = residual_tile_row {
        for (j, v) in row.iter_mut().enumerate() {
            *v += *res.add(j);
        }
    }
    match epilogue.activation {
        Activation::Relu => {
            for v in row.iter_mut() {
                *v = v.max(0.0);
            }
        }
        Activation::Silu => {
            for v in row.iter_mut() {
                let sig = 1.0 / (1.0 + (-*v).exp());
                *v *= sig;
            }
        }
        Activation::None => {}
    }
}

// ---------------------------------------------------------------------------
// Scalar micro-kernel (fallback + miri)
// ---------------------------------------------------------------------------

#[allow(unsafe_code)]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn microkernel_4x8_scalar(
    a_panel: *const f32,
    b_panel: *const f32,
    c: *mut f32,
    ldc: usize,
    kc: usize,
    accumulate: bool,
    epilogue: GemmEpilogue,
    is_last_k: bool,
    col_offset: usize,
    residual_tile: Option<*const f32>,
) {
    let mut acc = [[0.0f32; NR]; MR];

    for p in 0..kc {
        let a_base = p * MR;
        let b_base = p * NR;
        for i in 0..MR {
            let a_val = *a_panel.add(a_base + i);
            for j in 0..NR {
                acc[i][j] += a_val * *b_panel.add(b_base + j);
            }
        }
    }

    for i in 0..MR {
        let row_ptr = c.add(i * ldc);
        if accumulate {
            for j in 0..NR {
                acc[i][j] += *row_ptr.add(j);
            }
        }
        if is_last_k {
            let residual_row = residual_tile.map(|r| r.add(i * ldc));
            apply_epilogue_scalar(&mut acc[i][..NR], &epilogue, col_offset, residual_row);
        }
        for j in 0..NR {
            *row_ptr.add(j) = acc[i][j];
        }
    }
}

/// aarch64-only: scalar fallback for the 8×12 MR=8 path when the row or
/// column count doesn't match the full tile. Unlike the general
/// `microkernel_scalar_partial_stride`, this one knows both packed
/// strides are larger than the generic NR=8 (`a_stride=8`, `b_stride=12`),
/// so indexing has to match the NR=12 packed-B layout instead of the
/// generic NR=8.
#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
unsafe fn microkernel_scalar_nr12_partial_mr8(
    a_panel: *const f32,
    b_panel: *const f32,
    c: *mut f32,
    ldc: usize,
    mr: usize,
    nr: usize,
    kc: usize,
    accumulate: bool,
    epilogue: GemmEpilogue,
    is_last_k: bool,
    col_offset: usize,
    residual_tile: Option<*const f32>,
) {
    const MR8: usize = 8;
    const NR12: usize = 12;
    debug_assert!(mr <= MR8);
    debug_assert!(nr <= NR12);

    let mut acc = [[0.0f32; NR12]; MR8];

    for p in 0..kc {
        let a_base = p * MR8;
        let b_base = p * NR12;
        for i in 0..mr {
            let a_val = *a_panel.add(a_base + i);
            for j in 0..nr {
                acc[i][j] += a_val * *b_panel.add(b_base + j);
            }
        }
    }

    for i in 0..mr {
        let row_ptr = c.add(i * ldc);
        if accumulate {
            for j in 0..nr {
                acc[i][j] += *row_ptr.add(j);
            }
        }
        if is_last_k {
            let residual_row = residual_tile.map(|r| r.add(i * ldc));
            apply_epilogue_scalar(&mut acc[i][..nr], &epilogue, col_offset, residual_row);
        }
        for j in 0..nr {
            *row_ptr.add(j) = acc[i][j];
        }
    }
}

/// Scalar micro-kernel for edge tiles where mr < MR or nr < NR.
#[allow(unsafe_code)]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn microkernel_scalar_partial(
    a_panel: *const f32,
    b_panel: *const f32,
    c: *mut f32,
    ldc: usize,
    mr: usize,
    nr: usize,
    kc: usize,
    accumulate: bool,
    epilogue: GemmEpilogue,
    is_last_k: bool,
    col_offset: usize,
    residual_tile: Option<*const f32>,
) {
    microkernel_scalar_partial_stride(
        a_panel,
        b_panel,
        c,
        ldc,
        mr,
        nr,
        kc,
        MR,
        accumulate,
        epilogue,
        is_last_k,
        col_offset,
        residual_tile,
    );
}

#[allow(unsafe_code)]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn microkernel_scalar_partial_stride(
    a_panel: *const f32,
    b_panel: *const f32,
    c: *mut f32,
    ldc: usize,
    mr: usize,
    nr: usize,
    kc: usize,
    a_stride: usize,
    accumulate: bool,
    epilogue: GemmEpilogue,
    is_last_k: bool,
    col_offset: usize,
    residual_tile: Option<*const f32>,
) {
    // Sized for max(MR=4 x86/scalar, MR=8 aarch64 mr8 path). Real rows bounded by `mr`.
    let mut acc = [[0.0f32; NR]; 8];

    for p in 0..kc {
        let a_base = p * a_stride;
        let b_base = p * NR;
        for i in 0..mr {
            let a_val = *a_panel.add(a_base + i);
            for j in 0..nr {
                acc[i][j] += a_val * *b_panel.add(b_base + j);
            }
        }
    }

    for i in 0..mr {
        let row_ptr = c.add(i * ldc);
        if accumulate {
            for j in 0..nr {
                acc[i][j] += *row_ptr.add(j);
            }
        }
        if is_last_k {
            let residual_row = residual_tile.map(|r| r.add(i * ldc));
            apply_epilogue_scalar(&mut acc[i][..nr], &epilogue, col_offset, residual_row);
        }
        for j in 0..nr {
            *row_ptr.add(j) = acc[i][j];
        }
    }
}

// ---------------------------------------------------------------------------
// NEON micro-kernel (aarch64)
// ---------------------------------------------------------------------------

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code)]
#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "neon")]
unsafe fn microkernel_4x8_neon(
    a_panel: *const f32,
    b_panel: *const f32,
    c: *mut f32,
    ldc: usize,
    kc: usize,
    accumulate: bool,
    epilogue: GemmEpilogue,
    is_last_k: bool,
    col_offset: usize,
    residual_tile: Option<*const f32>,
) {
    let mut c00: float32x4_t = vdupq_n_f32(0.0);
    let mut c01: float32x4_t = vdupq_n_f32(0.0);
    let mut c10: float32x4_t = vdupq_n_f32(0.0);
    let mut c11: float32x4_t = vdupq_n_f32(0.0);
    let mut c20: float32x4_t = vdupq_n_f32(0.0);
    let mut c21: float32x4_t = vdupq_n_f32(0.0);
    let mut c30: float32x4_t = vdupq_n_f32(0.0);
    let mut c31: float32x4_t = vdupq_n_f32(0.0);

    for p in 0..kc {
        let a_off = p * MR;
        let b_off = p * NR;

        let a0 = vdupq_n_f32(*a_panel.add(a_off));
        let a1 = vdupq_n_f32(*a_panel.add(a_off + 1));
        let a2 = vdupq_n_f32(*a_panel.add(a_off + 2));
        let a3 = vdupq_n_f32(*a_panel.add(a_off + 3));

        let b0 = vld1q_f32(b_panel.add(b_off));
        let b1 = vld1q_f32(b_panel.add(b_off + 4));

        c00 = vfmaq_f32(c00, a0, b0);
        c01 = vfmaq_f32(c01, a0, b1);
        c10 = vfmaq_f32(c10, a1, b0);
        c11 = vfmaq_f32(c11, a1, b1);
        c20 = vfmaq_f32(c20, a2, b0);
        c21 = vfmaq_f32(c21, a2, b1);
        c30 = vfmaq_f32(c30, a3, b0);
        c31 = vfmaq_f32(c31, a3, b1);
    }

    let cp0 = c;
    let cp1 = c.add(ldc);
    let cp2 = c.add(2 * ldc);
    let cp3 = c.add(3 * ldc);

    if accumulate {
        c00 = vaddq_f32(vld1q_f32(cp0), c00);
        c01 = vaddq_f32(vld1q_f32(cp0.add(4)), c01);
        c10 = vaddq_f32(vld1q_f32(cp1), c10);
        c11 = vaddq_f32(vld1q_f32(cp1.add(4)), c11);
        c20 = vaddq_f32(vld1q_f32(cp2), c20);
        c21 = vaddq_f32(vld1q_f32(cp2.add(4)), c21);
        c30 = vaddq_f32(vld1q_f32(cp3), c30);
        c31 = vaddq_f32(vld1q_f32(cp3.add(4)), c31);
    }

    if is_last_k {
        if let Some(bias) = epilogue.bias {
            let bv0 = vld1q_f32(bias.add(col_offset));
            let bv1 = vld1q_f32(bias.add(col_offset + 4));
            c00 = vaddq_f32(c00, bv0);
            c01 = vaddq_f32(c01, bv1);
            c10 = vaddq_f32(c10, bv0);
            c11 = vaddq_f32(c11, bv1);
            c20 = vaddq_f32(c20, bv0);
            c21 = vaddq_f32(c21, bv1);
            c30 = vaddq_f32(c30, bv0);
            c31 = vaddq_f32(c31, bv1);
        }
        if let Some(res_tile) = residual_tile {
            let rp0 = res_tile;
            let rp1 = res_tile.add(ldc);
            let rp2 = res_tile.add(2 * ldc);
            let rp3 = res_tile.add(3 * ldc);
            c00 = vaddq_f32(c00, vld1q_f32(rp0));
            c01 = vaddq_f32(c01, vld1q_f32(rp0.add(4)));
            c10 = vaddq_f32(c10, vld1q_f32(rp1));
            c11 = vaddq_f32(c11, vld1q_f32(rp1.add(4)));
            c20 = vaddq_f32(c20, vld1q_f32(rp2));
            c21 = vaddq_f32(c21, vld1q_f32(rp2.add(4)));
            c30 = vaddq_f32(c30, vld1q_f32(rp3));
            c31 = vaddq_f32(c31, vld1q_f32(rp3.add(4)));
        }
        match epilogue.activation {
            Activation::Relu => {
                let zero = vdupq_n_f32(0.0);
                c00 = vmaxq_f32(c00, zero);
                c01 = vmaxq_f32(c01, zero);
                c10 = vmaxq_f32(c10, zero);
                c11 = vmaxq_f32(c11, zero);
                c20 = vmaxq_f32(c20, zero);
                c21 = vmaxq_f32(c21, zero);
                c30 = vmaxq_f32(c30, zero);
                c31 = vmaxq_f32(c31, zero);
            }
            Activation::Silu => {
                c00 = silu_neon(c00);
                c01 = silu_neon(c01);
                c10 = silu_neon(c10);
                c11 = silu_neon(c11);
                c20 = silu_neon(c20);
                c21 = silu_neon(c21);
                c30 = silu_neon(c30);
                c31 = silu_neon(c31);
            }
            Activation::None => {}
        }
    }

    vst1q_f32(cp0, c00);
    vst1q_f32(cp0.add(4), c01);
    vst1q_f32(cp1, c10);
    vst1q_f32(cp1.add(4), c11);
    vst1q_f32(cp2, c20);
    vst1q_f32(cp2.add(4), c21);
    vst1q_f32(cp3, c30);
    vst1q_f32(cp3.add(4), c31);
}

// ---------------------------------------------------------------------------
// Paired 4×16 NEON micro-kernel: two adjacent NR=8 panels at once.
// 16 f32x4 accumulators (4 rows × 4 quarters) + 4 b loads + 1 a broadcast per row.
// aarch64 has 32 v-registers, so this fits comfortably.
// ---------------------------------------------------------------------------

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code)]
#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "neon")]
unsafe fn microkernel_4x16_neon(
    a_panel: *const f32,
    b_panel_0: *const f32,
    b_panel_1: *const f32,
    c: *mut f32,
    ldc: usize,
    kc: usize,
    accumulate: bool,
    epilogue: GemmEpilogue,
    is_last_k: bool,
    col_offset: usize,
    residual_tile: Option<*const f32>,
) {
    let mut c00: float32x4_t = vdupq_n_f32(0.0);
    let mut c01: float32x4_t = vdupq_n_f32(0.0);
    let mut c02: float32x4_t = vdupq_n_f32(0.0);
    let mut c03: float32x4_t = vdupq_n_f32(0.0);
    let mut c10: float32x4_t = vdupq_n_f32(0.0);
    let mut c11: float32x4_t = vdupq_n_f32(0.0);
    let mut c12: float32x4_t = vdupq_n_f32(0.0);
    let mut c13: float32x4_t = vdupq_n_f32(0.0);
    let mut c20: float32x4_t = vdupq_n_f32(0.0);
    let mut c21: float32x4_t = vdupq_n_f32(0.0);
    let mut c22: float32x4_t = vdupq_n_f32(0.0);
    let mut c23: float32x4_t = vdupq_n_f32(0.0);
    let mut c30: float32x4_t = vdupq_n_f32(0.0);
    let mut c31: float32x4_t = vdupq_n_f32(0.0);
    let mut c32: float32x4_t = vdupq_n_f32(0.0);
    let mut c33: float32x4_t = vdupq_n_f32(0.0);

    for p in 0..kc {
        let a_off = p * MR;
        let b_off = p * NR;

        let a0 = vdupq_n_f32(*a_panel.add(a_off));
        let a1 = vdupq_n_f32(*a_panel.add(a_off + 1));
        let a2 = vdupq_n_f32(*a_panel.add(a_off + 2));
        let a3 = vdupq_n_f32(*a_panel.add(a_off + 3));

        let b0 = vld1q_f32(b_panel_0.add(b_off));
        let b1 = vld1q_f32(b_panel_0.add(b_off + 4));
        let b2 = vld1q_f32(b_panel_1.add(b_off));
        let b3 = vld1q_f32(b_panel_1.add(b_off + 4));

        c00 = vfmaq_f32(c00, a0, b0);
        c01 = vfmaq_f32(c01, a0, b1);
        c02 = vfmaq_f32(c02, a0, b2);
        c03 = vfmaq_f32(c03, a0, b3);
        c10 = vfmaq_f32(c10, a1, b0);
        c11 = vfmaq_f32(c11, a1, b1);
        c12 = vfmaq_f32(c12, a1, b2);
        c13 = vfmaq_f32(c13, a1, b3);
        c20 = vfmaq_f32(c20, a2, b0);
        c21 = vfmaq_f32(c21, a2, b1);
        c22 = vfmaq_f32(c22, a2, b2);
        c23 = vfmaq_f32(c23, a2, b3);
        c30 = vfmaq_f32(c30, a3, b0);
        c31 = vfmaq_f32(c31, a3, b1);
        c32 = vfmaq_f32(c32, a3, b2);
        c33 = vfmaq_f32(c33, a3, b3);
    }

    let cp0 = c;
    let cp1 = c.add(ldc);
    let cp2 = c.add(2 * ldc);
    let cp3 = c.add(3 * ldc);

    if accumulate {
        c00 = vaddq_f32(vld1q_f32(cp0), c00);
        c01 = vaddq_f32(vld1q_f32(cp0.add(4)), c01);
        c02 = vaddq_f32(vld1q_f32(cp0.add(8)), c02);
        c03 = vaddq_f32(vld1q_f32(cp0.add(12)), c03);
        c10 = vaddq_f32(vld1q_f32(cp1), c10);
        c11 = vaddq_f32(vld1q_f32(cp1.add(4)), c11);
        c12 = vaddq_f32(vld1q_f32(cp1.add(8)), c12);
        c13 = vaddq_f32(vld1q_f32(cp1.add(12)), c13);
        c20 = vaddq_f32(vld1q_f32(cp2), c20);
        c21 = vaddq_f32(vld1q_f32(cp2.add(4)), c21);
        c22 = vaddq_f32(vld1q_f32(cp2.add(8)), c22);
        c23 = vaddq_f32(vld1q_f32(cp2.add(12)), c23);
        c30 = vaddq_f32(vld1q_f32(cp3), c30);
        c31 = vaddq_f32(vld1q_f32(cp3.add(4)), c31);
        c32 = vaddq_f32(vld1q_f32(cp3.add(8)), c32);
        c33 = vaddq_f32(vld1q_f32(cp3.add(12)), c33);
    }

    if is_last_k {
        if let Some(bias) = epilogue.bias {
            let bv0 = vld1q_f32(bias.add(col_offset));
            let bv1 = vld1q_f32(bias.add(col_offset + 4));
            let bv2 = vld1q_f32(bias.add(col_offset + 8));
            let bv3 = vld1q_f32(bias.add(col_offset + 12));
            c00 = vaddq_f32(c00, bv0);
            c01 = vaddq_f32(c01, bv1);
            c02 = vaddq_f32(c02, bv2);
            c03 = vaddq_f32(c03, bv3);
            c10 = vaddq_f32(c10, bv0);
            c11 = vaddq_f32(c11, bv1);
            c12 = vaddq_f32(c12, bv2);
            c13 = vaddq_f32(c13, bv3);
            c20 = vaddq_f32(c20, bv0);
            c21 = vaddq_f32(c21, bv1);
            c22 = vaddq_f32(c22, bv2);
            c23 = vaddq_f32(c23, bv3);
            c30 = vaddq_f32(c30, bv0);
            c31 = vaddq_f32(c31, bv1);
            c32 = vaddq_f32(c32, bv2);
            c33 = vaddq_f32(c33, bv3);
        }
        if let Some(res_tile) = residual_tile {
            let rp0 = res_tile;
            let rp1 = res_tile.add(ldc);
            let rp2 = res_tile.add(2 * ldc);
            let rp3 = res_tile.add(3 * ldc);
            c00 = vaddq_f32(c00, vld1q_f32(rp0));
            c01 = vaddq_f32(c01, vld1q_f32(rp0.add(4)));
            c02 = vaddq_f32(c02, vld1q_f32(rp0.add(8)));
            c03 = vaddq_f32(c03, vld1q_f32(rp0.add(12)));
            c10 = vaddq_f32(c10, vld1q_f32(rp1));
            c11 = vaddq_f32(c11, vld1q_f32(rp1.add(4)));
            c12 = vaddq_f32(c12, vld1q_f32(rp1.add(8)));
            c13 = vaddq_f32(c13, vld1q_f32(rp1.add(12)));
            c20 = vaddq_f32(c20, vld1q_f32(rp2));
            c21 = vaddq_f32(c21, vld1q_f32(rp2.add(4)));
            c22 = vaddq_f32(c22, vld1q_f32(rp2.add(8)));
            c23 = vaddq_f32(c23, vld1q_f32(rp2.add(12)));
            c30 = vaddq_f32(c30, vld1q_f32(rp3));
            c31 = vaddq_f32(c31, vld1q_f32(rp3.add(4)));
            c32 = vaddq_f32(c32, vld1q_f32(rp3.add(8)));
            c33 = vaddq_f32(c33, vld1q_f32(rp3.add(12)));
        }
        match epilogue.activation {
            Activation::Relu => {
                let zero = vdupq_n_f32(0.0);
                c00 = vmaxq_f32(c00, zero);
                c01 = vmaxq_f32(c01, zero);
                c02 = vmaxq_f32(c02, zero);
                c03 = vmaxq_f32(c03, zero);
                c10 = vmaxq_f32(c10, zero);
                c11 = vmaxq_f32(c11, zero);
                c12 = vmaxq_f32(c12, zero);
                c13 = vmaxq_f32(c13, zero);
                c20 = vmaxq_f32(c20, zero);
                c21 = vmaxq_f32(c21, zero);
                c22 = vmaxq_f32(c22, zero);
                c23 = vmaxq_f32(c23, zero);
                c30 = vmaxq_f32(c30, zero);
                c31 = vmaxq_f32(c31, zero);
                c32 = vmaxq_f32(c32, zero);
                c33 = vmaxq_f32(c33, zero);
            }
            Activation::Silu => {
                c00 = silu_neon(c00);
                c01 = silu_neon(c01);
                c02 = silu_neon(c02);
                c03 = silu_neon(c03);
                c10 = silu_neon(c10);
                c11 = silu_neon(c11);
                c12 = silu_neon(c12);
                c13 = silu_neon(c13);
                c20 = silu_neon(c20);
                c21 = silu_neon(c21);
                c22 = silu_neon(c22);
                c23 = silu_neon(c23);
                c30 = silu_neon(c30);
                c31 = silu_neon(c31);
                c32 = silu_neon(c32);
                c33 = silu_neon(c33);
            }
            Activation::None => {}
        }
    }

    vst1q_f32(cp0, c00);
    vst1q_f32(cp0.add(4), c01);
    vst1q_f32(cp0.add(8), c02);
    vst1q_f32(cp0.add(12), c03);
    vst1q_f32(cp1, c10);
    vst1q_f32(cp1.add(4), c11);
    vst1q_f32(cp1.add(8), c12);
    vst1q_f32(cp1.add(12), c13);
    vst1q_f32(cp2, c20);
    vst1q_f32(cp2.add(4), c21);
    vst1q_f32(cp2.add(8), c22);
    vst1q_f32(cp2.add(12), c23);
    vst1q_f32(cp3, c30);
    vst1q_f32(cp3.add(4), c31);
    vst1q_f32(cp3.add(8), c32);
    vst1q_f32(cp3.add(12), c33);
}

// ---------------------------------------------------------------------------
// Tripled 4×24 NEON micro-kernel: three adjacent NR=8 panels at once.
// 24 f32x4 accumulators (4 rows × 6 quarters) + 6 b loads + 1 a broadcast
// = 31 v-registers. aarch64 has exactly 32, so this is the widest NEON
// variant possible. Matches the x86 4×24 dispatch branch for the common
// "wide N" pointwise Conv case on ARM.
// ---------------------------------------------------------------------------

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code)]
#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "neon")]
unsafe fn microkernel_4x24_neon(
    a_panel: *const f32,
    b_panel_0: *const f32,
    b_panel_1: *const f32,
    b_panel_2: *const f32,
    c: *mut f32,
    ldc: usize,
    kc: usize,
    accumulate: bool,
    epilogue: GemmEpilogue,
    is_last_k: bool,
    col_offset: usize,
    residual_tile: Option<*const f32>,
) {
    // 4 rows × 6 f32x4 quarters = 24 accumulators.
    let mut c00 = vdupq_n_f32(0.0);
    let mut c01 = vdupq_n_f32(0.0);
    let mut c02 = vdupq_n_f32(0.0);
    let mut c03 = vdupq_n_f32(0.0);
    let mut c04 = vdupq_n_f32(0.0);
    let mut c05 = vdupq_n_f32(0.0);
    let mut c10 = vdupq_n_f32(0.0);
    let mut c11 = vdupq_n_f32(0.0);
    let mut c12 = vdupq_n_f32(0.0);
    let mut c13 = vdupq_n_f32(0.0);
    let mut c14 = vdupq_n_f32(0.0);
    let mut c15 = vdupq_n_f32(0.0);
    let mut c20 = vdupq_n_f32(0.0);
    let mut c21 = vdupq_n_f32(0.0);
    let mut c22 = vdupq_n_f32(0.0);
    let mut c23 = vdupq_n_f32(0.0);
    let mut c24 = vdupq_n_f32(0.0);
    let mut c25 = vdupq_n_f32(0.0);
    let mut c30 = vdupq_n_f32(0.0);
    let mut c31 = vdupq_n_f32(0.0);
    let mut c32 = vdupq_n_f32(0.0);
    let mut c33 = vdupq_n_f32(0.0);
    let mut c34 = vdupq_n_f32(0.0);
    let mut c35 = vdupq_n_f32(0.0);

    for p in 0..kc {
        let a_off = p * MR;
        let b_off = p * NR;

        let a0 = vdupq_n_f32(*a_panel.add(a_off));
        let a1 = vdupq_n_f32(*a_panel.add(a_off + 1));
        let a2 = vdupq_n_f32(*a_panel.add(a_off + 2));
        let a3 = vdupq_n_f32(*a_panel.add(a_off + 3));

        let b0 = vld1q_f32(b_panel_0.add(b_off));
        let b1 = vld1q_f32(b_panel_0.add(b_off + 4));
        let b2 = vld1q_f32(b_panel_1.add(b_off));
        let b3 = vld1q_f32(b_panel_1.add(b_off + 4));
        let b4 = vld1q_f32(b_panel_2.add(b_off));
        let b5 = vld1q_f32(b_panel_2.add(b_off + 4));

        c00 = vfmaq_f32(c00, a0, b0);
        c01 = vfmaq_f32(c01, a0, b1);
        c02 = vfmaq_f32(c02, a0, b2);
        c03 = vfmaq_f32(c03, a0, b3);
        c04 = vfmaq_f32(c04, a0, b4);
        c05 = vfmaq_f32(c05, a0, b5);
        c10 = vfmaq_f32(c10, a1, b0);
        c11 = vfmaq_f32(c11, a1, b1);
        c12 = vfmaq_f32(c12, a1, b2);
        c13 = vfmaq_f32(c13, a1, b3);
        c14 = vfmaq_f32(c14, a1, b4);
        c15 = vfmaq_f32(c15, a1, b5);
        c20 = vfmaq_f32(c20, a2, b0);
        c21 = vfmaq_f32(c21, a2, b1);
        c22 = vfmaq_f32(c22, a2, b2);
        c23 = vfmaq_f32(c23, a2, b3);
        c24 = vfmaq_f32(c24, a2, b4);
        c25 = vfmaq_f32(c25, a2, b5);
        c30 = vfmaq_f32(c30, a3, b0);
        c31 = vfmaq_f32(c31, a3, b1);
        c32 = vfmaq_f32(c32, a3, b2);
        c33 = vfmaq_f32(c33, a3, b3);
        c34 = vfmaq_f32(c34, a3, b4);
        c35 = vfmaq_f32(c35, a3, b5);
    }

    let cp0 = c;
    let cp1 = c.add(ldc);
    let cp2 = c.add(2 * ldc);
    let cp3 = c.add(3 * ldc);

    if accumulate {
        c00 = vaddq_f32(vld1q_f32(cp0), c00);
        c01 = vaddq_f32(vld1q_f32(cp0.add(4)), c01);
        c02 = vaddq_f32(vld1q_f32(cp0.add(8)), c02);
        c03 = vaddq_f32(vld1q_f32(cp0.add(12)), c03);
        c04 = vaddq_f32(vld1q_f32(cp0.add(16)), c04);
        c05 = vaddq_f32(vld1q_f32(cp0.add(20)), c05);
        c10 = vaddq_f32(vld1q_f32(cp1), c10);
        c11 = vaddq_f32(vld1q_f32(cp1.add(4)), c11);
        c12 = vaddq_f32(vld1q_f32(cp1.add(8)), c12);
        c13 = vaddq_f32(vld1q_f32(cp1.add(12)), c13);
        c14 = vaddq_f32(vld1q_f32(cp1.add(16)), c14);
        c15 = vaddq_f32(vld1q_f32(cp1.add(20)), c15);
        c20 = vaddq_f32(vld1q_f32(cp2), c20);
        c21 = vaddq_f32(vld1q_f32(cp2.add(4)), c21);
        c22 = vaddq_f32(vld1q_f32(cp2.add(8)), c22);
        c23 = vaddq_f32(vld1q_f32(cp2.add(12)), c23);
        c24 = vaddq_f32(vld1q_f32(cp2.add(16)), c24);
        c25 = vaddq_f32(vld1q_f32(cp2.add(20)), c25);
        c30 = vaddq_f32(vld1q_f32(cp3), c30);
        c31 = vaddq_f32(vld1q_f32(cp3.add(4)), c31);
        c32 = vaddq_f32(vld1q_f32(cp3.add(8)), c32);
        c33 = vaddq_f32(vld1q_f32(cp3.add(12)), c33);
        c34 = vaddq_f32(vld1q_f32(cp3.add(16)), c34);
        c35 = vaddq_f32(vld1q_f32(cp3.add(20)), c35);
    }

    if is_last_k {
        if let Some(bias) = epilogue.bias {
            let bv0 = vld1q_f32(bias.add(col_offset));
            let bv1 = vld1q_f32(bias.add(col_offset + 4));
            let bv2 = vld1q_f32(bias.add(col_offset + 8));
            let bv3 = vld1q_f32(bias.add(col_offset + 12));
            let bv4 = vld1q_f32(bias.add(col_offset + 16));
            let bv5 = vld1q_f32(bias.add(col_offset + 20));
            c00 = vaddq_f32(c00, bv0);
            c01 = vaddq_f32(c01, bv1);
            c02 = vaddq_f32(c02, bv2);
            c03 = vaddq_f32(c03, bv3);
            c04 = vaddq_f32(c04, bv4);
            c05 = vaddq_f32(c05, bv5);
            c10 = vaddq_f32(c10, bv0);
            c11 = vaddq_f32(c11, bv1);
            c12 = vaddq_f32(c12, bv2);
            c13 = vaddq_f32(c13, bv3);
            c14 = vaddq_f32(c14, bv4);
            c15 = vaddq_f32(c15, bv5);
            c20 = vaddq_f32(c20, bv0);
            c21 = vaddq_f32(c21, bv1);
            c22 = vaddq_f32(c22, bv2);
            c23 = vaddq_f32(c23, bv3);
            c24 = vaddq_f32(c24, bv4);
            c25 = vaddq_f32(c25, bv5);
            c30 = vaddq_f32(c30, bv0);
            c31 = vaddq_f32(c31, bv1);
            c32 = vaddq_f32(c32, bv2);
            c33 = vaddq_f32(c33, bv3);
            c34 = vaddq_f32(c34, bv4);
            c35 = vaddq_f32(c35, bv5);
        }
        if let Some(res_tile) = residual_tile {
            let rp0 = res_tile;
            let rp1 = res_tile.add(ldc);
            let rp2 = res_tile.add(2 * ldc);
            let rp3 = res_tile.add(3 * ldc);
            c00 = vaddq_f32(c00, vld1q_f32(rp0));
            c01 = vaddq_f32(c01, vld1q_f32(rp0.add(4)));
            c02 = vaddq_f32(c02, vld1q_f32(rp0.add(8)));
            c03 = vaddq_f32(c03, vld1q_f32(rp0.add(12)));
            c04 = vaddq_f32(c04, vld1q_f32(rp0.add(16)));
            c05 = vaddq_f32(c05, vld1q_f32(rp0.add(20)));
            c10 = vaddq_f32(c10, vld1q_f32(rp1));
            c11 = vaddq_f32(c11, vld1q_f32(rp1.add(4)));
            c12 = vaddq_f32(c12, vld1q_f32(rp1.add(8)));
            c13 = vaddq_f32(c13, vld1q_f32(rp1.add(12)));
            c14 = vaddq_f32(c14, vld1q_f32(rp1.add(16)));
            c15 = vaddq_f32(c15, vld1q_f32(rp1.add(20)));
            c20 = vaddq_f32(c20, vld1q_f32(rp2));
            c21 = vaddq_f32(c21, vld1q_f32(rp2.add(4)));
            c22 = vaddq_f32(c22, vld1q_f32(rp2.add(8)));
            c23 = vaddq_f32(c23, vld1q_f32(rp2.add(12)));
            c24 = vaddq_f32(c24, vld1q_f32(rp2.add(16)));
            c25 = vaddq_f32(c25, vld1q_f32(rp2.add(20)));
            c30 = vaddq_f32(c30, vld1q_f32(rp3));
            c31 = vaddq_f32(c31, vld1q_f32(rp3.add(4)));
            c32 = vaddq_f32(c32, vld1q_f32(rp3.add(8)));
            c33 = vaddq_f32(c33, vld1q_f32(rp3.add(12)));
            c34 = vaddq_f32(c34, vld1q_f32(rp3.add(16)));
            c35 = vaddq_f32(c35, vld1q_f32(rp3.add(20)));
        }
        match epilogue.activation {
            Activation::Relu => {
                let z = vdupq_n_f32(0.0);
                c00 = vmaxq_f32(c00, z);
                c01 = vmaxq_f32(c01, z);
                c02 = vmaxq_f32(c02, z);
                c03 = vmaxq_f32(c03, z);
                c04 = vmaxq_f32(c04, z);
                c05 = vmaxq_f32(c05, z);
                c10 = vmaxq_f32(c10, z);
                c11 = vmaxq_f32(c11, z);
                c12 = vmaxq_f32(c12, z);
                c13 = vmaxq_f32(c13, z);
                c14 = vmaxq_f32(c14, z);
                c15 = vmaxq_f32(c15, z);
                c20 = vmaxq_f32(c20, z);
                c21 = vmaxq_f32(c21, z);
                c22 = vmaxq_f32(c22, z);
                c23 = vmaxq_f32(c23, z);
                c24 = vmaxq_f32(c24, z);
                c25 = vmaxq_f32(c25, z);
                c30 = vmaxq_f32(c30, z);
                c31 = vmaxq_f32(c31, z);
                c32 = vmaxq_f32(c32, z);
                c33 = vmaxq_f32(c33, z);
                c34 = vmaxq_f32(c34, z);
                c35 = vmaxq_f32(c35, z);
            }
            Activation::Silu => {
                c00 = silu_neon(c00);
                c01 = silu_neon(c01);
                c02 = silu_neon(c02);
                c03 = silu_neon(c03);
                c04 = silu_neon(c04);
                c05 = silu_neon(c05);
                c10 = silu_neon(c10);
                c11 = silu_neon(c11);
                c12 = silu_neon(c12);
                c13 = silu_neon(c13);
                c14 = silu_neon(c14);
                c15 = silu_neon(c15);
                c20 = silu_neon(c20);
                c21 = silu_neon(c21);
                c22 = silu_neon(c22);
                c23 = silu_neon(c23);
                c24 = silu_neon(c24);
                c25 = silu_neon(c25);
                c30 = silu_neon(c30);
                c31 = silu_neon(c31);
                c32 = silu_neon(c32);
                c33 = silu_neon(c33);
                c34 = silu_neon(c34);
                c35 = silu_neon(c35);
            }
            Activation::None => {}
        }
    }

    vst1q_f32(cp0, c00);
    vst1q_f32(cp0.add(4), c01);
    vst1q_f32(cp0.add(8), c02);
    vst1q_f32(cp0.add(12), c03);
    vst1q_f32(cp0.add(16), c04);
    vst1q_f32(cp0.add(20), c05);
    vst1q_f32(cp1, c10);
    vst1q_f32(cp1.add(4), c11);
    vst1q_f32(cp1.add(8), c12);
    vst1q_f32(cp1.add(12), c13);
    vst1q_f32(cp1.add(16), c14);
    vst1q_f32(cp1.add(20), c15);
    vst1q_f32(cp2, c20);
    vst1q_f32(cp2.add(4), c21);
    vst1q_f32(cp2.add(8), c22);
    vst1q_f32(cp2.add(12), c23);
    vst1q_f32(cp2.add(16), c24);
    vst1q_f32(cp2.add(20), c25);
    vst1q_f32(cp3, c30);
    vst1q_f32(cp3.add(4), c31);
    vst1q_f32(cp3.add(8), c32);
    vst1q_f32(cp3.add(12), c33);
    vst1q_f32(cp3.add(16), c34);
    vst1q_f32(cp3.add(20), c35);
}

// ---------------------------------------------------------------------------
// Paired 4×16 AVX+FMA micro-kernel: two adjacent NR=8 tiles at once.
// Amortizes 4 a-broadcasts across 16 columns instead of 8, reaching
// theoretical peak FMA throughput on Zen 4 (2 FMA ports × 8-wide).
// ---------------------------------------------------------------------------

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code)]
#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx,fma")]
#[inline]
unsafe fn microkernel_4x16_avx_fma(
    a_panel: *const f32,
    b_panel_0: *const f32,
    b_panel_1: *const f32,
    c: *mut f32,
    ldc: usize,
    kc: usize,
    accumulate: bool,
    epilogue: GemmEpilogue,
    is_last_k: bool,
    col_offset: usize,
    residual_tile: Option<*const f32>,
) {
    // 8 accumulators: ymm0-ymm7 (c00,c01,c10,c11,c20,c21,c30,c31)
    // Temp: ymm8,ymm9 (b0,b1), ymm10 (a broadcast)
    let mut c00: __m256;
    let mut c01: __m256;
    let mut c10: __m256;
    let mut c11: __m256;
    let mut c20: __m256;
    let mut c21: __m256;
    let mut c30: __m256;
    let mut c31: __m256;

    #[cfg(target_arch = "x86_64")]
    std::arch::asm!(
        "vxorps {c00}, {c00}, {c00}",
        "vxorps {c01}, {c01}, {c01}",
        "vxorps {c10}, {c10}, {c10}",
        "vxorps {c11}, {c11}, {c11}",
        "vxorps {c20}, {c20}, {c20}",
        "vxorps {c21}, {c21}, {c21}",
        "vxorps {c30}, {c30}, {c30}",
        "vxorps {c31}, {c31}, {c31}",

        "mov {pairs}, {kc}",
        "shr {pairs}, 1",
        "test {pairs}, {pairs}",
        "jz 12f",

        ".p2align 4",
        "13:",
        // 2-way unrolled k-loop with DOUBLE-BUFFERED B loads.
        // b0a/b1a hold B[k]; we preload B[k+1] into b0b/b1b EARLY (interleaved
        // with FMAs on b0a/b1a) so k+1's FMAs can start immediately after k's
        // finish. 8 acc + 4 B + 1 A = 13 YMM, 3 free for OoO scheduling.
        //
        // --- k-step 0: load B[k] AND B[k+1] (into b0b, b1b) ---
        "vmovups {b0a}, [{bp0}]",
        "vmovups {b1a}, [{bp1}]",
        "vmovups {b0b}, [{bp0} + 32]",         // B[k+1] panel 0 — starts loading
        "vbroadcastss {av}, [{ap}]",
        "vfmadd231ps {c00}, {b0a}, {av}",
        "vfmadd231ps {c01}, {b1a}, {av}",
        "vmovups {b1b}, [{bp1} + 32]",         // B[k+1] panel 1 — interleaved with FMAs
        "vbroadcastss {av}, [{ap} + 4]",
        "vfmadd231ps {c10}, {b0a}, {av}",
        "vfmadd231ps {c11}, {b1a}, {av}",
        "vbroadcastss {av}, [{ap} + 8]",
        "vfmadd231ps {c20}, {b0a}, {av}",
        "vfmadd231ps {c21}, {b1a}, {av}",
        "prefetcht0 [{ap} + 64]",              // prefetch next A panel chunk
        "vbroadcastss {av}, [{ap} + 12]",
        "vfmadd231ps {c30}, {b0a}, {av}",
        "vfmadd231ps {c31}, {b1a}, {av}",
        // --- k-step 1: b0b, b1b already have B[k+1] (loaded above) ---
        "vbroadcastss {av}, [{ap} + 16]",
        "vfmadd231ps {c00}, {b0b}, {av}",
        "vfmadd231ps {c01}, {b1b}, {av}",
        "vbroadcastss {av}, [{ap} + 20]",
        "vfmadd231ps {c10}, {b0b}, {av}",
        "vfmadd231ps {c11}, {b1b}, {av}",
        "vbroadcastss {av}, [{ap} + 24]",
        "vfmadd231ps {c20}, {b0b}, {av}",
        "vfmadd231ps {c21}, {b1b}, {av}",
        "vbroadcastss {av}, [{ap} + 28]",
        "vfmadd231ps {c30}, {b0b}, {av}",
        "vfmadd231ps {c31}, {b1b}, {av}",
        "add {ap}, 32",
        "add {bp0}, 64",
        "add {bp1}, 64",
        "dec {pairs}",
        "jnz 13b",

        "12:",
        "test {kc}, 1",
        "jz 14f",
        "vmovups {b0a}, [{bp0}]",
        "vmovups {b1a}, [{bp1}]",
        "vbroadcastss {av}, [{ap}]",
        "vfmadd231ps {c00}, {b0a}, {av}",
        "vfmadd231ps {c01}, {b1a}, {av}",
        "vbroadcastss {av}, [{ap} + 4]",
        "vfmadd231ps {c10}, {b0a}, {av}",
        "vfmadd231ps {c11}, {b1a}, {av}",
        "vbroadcastss {av}, [{ap} + 8]",
        "vfmadd231ps {c20}, {b0a}, {av}",
        "vfmadd231ps {c21}, {b1a}, {av}",
        "vbroadcastss {av}, [{ap} + 12]",
        "vfmadd231ps {c30}, {b0a}, {av}",
        "vfmadd231ps {c31}, {b1a}, {av}",
        "14:",

        ap = inout(reg) a_panel => _,
        bp0 = inout(reg) b_panel_0 => _,
        bp1 = inout(reg) b_panel_1 => _,
        kc = in(reg) kc,
        pairs = out(reg) _,
        c00 = out(ymm_reg) c00,
        c01 = out(ymm_reg) c01,
        c10 = out(ymm_reg) c10,
        c11 = out(ymm_reg) c11,
        c20 = out(ymm_reg) c20,
        c21 = out(ymm_reg) c21,
        c30 = out(ymm_reg) c30,
        c31 = out(ymm_reg) c31,
        b0a = out(ymm_reg) _,
        b1a = out(ymm_reg) _,
        b0b = out(ymm_reg) _,
        b1b = out(ymm_reg) _,
        av = out(ymm_reg) _,
        options(nostack),
    );

    #[cfg(target_arch = "x86")]
    {
        // x86 32-bit fallback
        c00 = _mm256_setzero_ps();
        c01 = _mm256_setzero_ps();
        c10 = _mm256_setzero_ps();
        c11 = _mm256_setzero_ps();
        c20 = _mm256_setzero_ps();
        c21 = _mm256_setzero_ps();
        c30 = _mm256_setzero_ps();
        c31 = _mm256_setzero_ps();
        for p in 0..kc {
            let b0 = _mm256_loadu_ps(b_panel_0.add(p * NR));
            let b1 = _mm256_loadu_ps(b_panel_1.add(p * NR));
            let a0 = _mm256_set1_ps(*a_panel.add(p * MR));
            c00 = _mm256_fmadd_ps(a0, b0, c00);
            c01 = _mm256_fmadd_ps(a0, b1, c01);
            let a1 = _mm256_set1_ps(*a_panel.add(p * MR + 1));
            c10 = _mm256_fmadd_ps(a1, b0, c10);
            c11 = _mm256_fmadd_ps(a1, b1, c11);
            let a2 = _mm256_set1_ps(*a_panel.add(p * MR + 2));
            c20 = _mm256_fmadd_ps(a2, b0, c20);
            c21 = _mm256_fmadd_ps(a2, b1, c21);
            let a3 = _mm256_set1_ps(*a_panel.add(p * MR + 3));
            c30 = _mm256_fmadd_ps(a3, b0, c30);
            c31 = _mm256_fmadd_ps(a3, b1, c31);
        }
    }

    let cp0 = c;
    let cp1 = c.add(ldc);
    let cp2 = c.add(2 * ldc);
    let cp3 = c.add(3 * ldc);

    if accumulate {
        c00 = _mm256_add_ps(_mm256_loadu_ps(cp0), c00);
        c01 = _mm256_add_ps(_mm256_loadu_ps(cp0.add(8)), c01);
        c10 = _mm256_add_ps(_mm256_loadu_ps(cp1), c10);
        c11 = _mm256_add_ps(_mm256_loadu_ps(cp1.add(8)), c11);
        c20 = _mm256_add_ps(_mm256_loadu_ps(cp2), c20);
        c21 = _mm256_add_ps(_mm256_loadu_ps(cp2.add(8)), c21);
        c30 = _mm256_add_ps(_mm256_loadu_ps(cp3), c30);
        c31 = _mm256_add_ps(_mm256_loadu_ps(cp3.add(8)), c31);
    }

    if is_last_k {
        if let Some(bias) = epilogue.bias {
            let bv0 = _mm256_loadu_ps(bias.add(col_offset));
            let bv1 = _mm256_loadu_ps(bias.add(col_offset + 8));
            c00 = _mm256_add_ps(c00, bv0);
            c01 = _mm256_add_ps(c01, bv1);
            c10 = _mm256_add_ps(c10, bv0);
            c11 = _mm256_add_ps(c11, bv1);
            c20 = _mm256_add_ps(c20, bv0);
            c21 = _mm256_add_ps(c21, bv1);
            c30 = _mm256_add_ps(c30, bv0);
            c31 = _mm256_add_ps(c31, bv1);
        }
        // Residual add (Phase 1.2): 8 YMMs from per-tile residual base.
        if let Some(res_tile) = residual_tile {
            let rp0 = res_tile;
            let rp1 = res_tile.add(ldc);
            let rp2 = res_tile.add(2 * ldc);
            let rp3 = res_tile.add(3 * ldc);
            c00 = _mm256_add_ps(c00, _mm256_loadu_ps(rp0));
            c01 = _mm256_add_ps(c01, _mm256_loadu_ps(rp0.add(8)));
            c10 = _mm256_add_ps(c10, _mm256_loadu_ps(rp1));
            c11 = _mm256_add_ps(c11, _mm256_loadu_ps(rp1.add(8)));
            c20 = _mm256_add_ps(c20, _mm256_loadu_ps(rp2));
            c21 = _mm256_add_ps(c21, _mm256_loadu_ps(rp2.add(8)));
            c30 = _mm256_add_ps(c30, _mm256_loadu_ps(rp3));
            c31 = _mm256_add_ps(c31, _mm256_loadu_ps(rp3.add(8)));
        }
        match epilogue.activation {
            Activation::Relu => {
                let zero = _mm256_setzero_ps();
                c00 = _mm256_max_ps(c00, zero);
                c01 = _mm256_max_ps(c01, zero);
                c10 = _mm256_max_ps(c10, zero);
                c11 = _mm256_max_ps(c11, zero);
                c20 = _mm256_max_ps(c20, zero);
                c21 = _mm256_max_ps(c21, zero);
                c30 = _mm256_max_ps(c30, zero);
                c31 = _mm256_max_ps(c31, zero);
            }
            Activation::Silu => {
                c00 = silu_avx_fma(c00);
                c01 = silu_avx_fma(c01);
                c10 = silu_avx_fma(c10);
                c11 = silu_avx_fma(c11);
                c20 = silu_avx_fma(c20);
                c21 = silu_avx_fma(c21);
                c30 = silu_avx_fma(c30);
                c31 = silu_avx_fma(c31);
            }
            Activation::None => {}
        }
    }

    _mm256_storeu_ps(cp0, c00);
    _mm256_storeu_ps(cp0.add(8), c01);
    _mm256_storeu_ps(cp1, c10);
    _mm256_storeu_ps(cp1.add(8), c11);
    _mm256_storeu_ps(cp2, c20);
    _mm256_storeu_ps(cp2.add(8), c21);
    _mm256_storeu_ps(cp3, c30);
    _mm256_storeu_ps(cp3.add(8), c31);
}

// ---------------------------------------------------------------------------
// MR=6 × NR=16 AVX+FMA micro-kernel (Step C1)
// ---------------------------------------------------------------------------
// Matches MLAS SgemmKernelFma3's 6×16 tile. Consumes 6 rows of a MR=6-
// packed A panel and 2 NR=8 B panels, producing a 6×16 output tile.
// 12 YMM accumulators (6 rows × 2 panels) + 2 B + 1 A broadcast = 15 YMM.
// 1 YMM free — no room for B[k+1] double-buffering, so k-loop is
// single-step (LLVM is allowed to schedule; `#[inline]` + `target_feature`
// lets it unroll/schedule at its discretion).
//
// Register pressure justification: 6 rows × 2 panels × 8 cols = 96 floats
// of output per tile = 12 YMM at exactly AVX2's 16-YMM file. MR=8×16 would
// need 16 acc + B + A = 19 YMM (no fit). MR=6 is the sweet spot on AVX2.
//
// Compared to MR=4×NR=24 (12 acc × 3 panels = same 12 acc): MR=6 halves
// the number of B panels per tile-row (2 vs 3) and trades 50% more A
// broadcasts (6 vs 4 per k-step). On m≥192 shapes this nets **55% less
// B-panel traffic** across the outer loop. Expected: +10–20% throughput
// on pointwise Conv shapes.

/// 6×16 AVX+FMA microkernel. Pure intrinsics; compiler schedules.
/// Bias + optional residual + activation folded inline for single-pass
/// store. Mirrors `microkernel_4x16_avx_fma` structure scaled to 6 rows.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code)]
#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx,fma")]
#[inline]
unsafe fn microkernel_6x16_avx_fma(
    a_panel: *const f32,
    b_panel_0: *const f32,
    b_panel_1: *const f32,
    c: *mut f32,
    ldc: usize,
    kc: usize,
    accumulate: bool,
    epilogue: GemmEpilogue,
    is_last_k: bool,
    col_offset: usize,
    residual_tile: Option<*const f32>,
) {
    // 12 accumulators, 6 rows × 2 NR panels of 8 cols.
    let mut c00: __m256 = _mm256_setzero_ps();
    let mut c01: __m256 = _mm256_setzero_ps();
    let mut c10: __m256 = _mm256_setzero_ps();
    let mut c11: __m256 = _mm256_setzero_ps();
    let mut c20: __m256 = _mm256_setzero_ps();
    let mut c21: __m256 = _mm256_setzero_ps();
    let mut c30: __m256 = _mm256_setzero_ps();
    let mut c31: __m256 = _mm256_setzero_ps();
    let mut c40: __m256 = _mm256_setzero_ps();
    let mut c41: __m256 = _mm256_setzero_ps();
    let mut c50: __m256 = _mm256_setzero_ps();
    let mut c51: __m256 = _mm256_setzero_ps();

    // K-loop: 12 FMAs per k-step, 2 B-panel loads, 6 A broadcasts.
    // Pack-A stride is MR6=6 floats per k-step.
    for p in 0..kc {
        let a_off = p * MR6;
        let b_off = p * NR;
        let b0 = _mm256_loadu_ps(b_panel_0.add(b_off));
        let b1 = _mm256_loadu_ps(b_panel_1.add(b_off));

        let a0 = _mm256_broadcast_ss(&*a_panel.add(a_off));
        c00 = _mm256_fmadd_ps(b0, a0, c00);
        c01 = _mm256_fmadd_ps(b1, a0, c01);

        let a1 = _mm256_broadcast_ss(&*a_panel.add(a_off + 1));
        c10 = _mm256_fmadd_ps(b0, a1, c10);
        c11 = _mm256_fmadd_ps(b1, a1, c11);

        let a2 = _mm256_broadcast_ss(&*a_panel.add(a_off + 2));
        c20 = _mm256_fmadd_ps(b0, a2, c20);
        c21 = _mm256_fmadd_ps(b1, a2, c21);

        let a3 = _mm256_broadcast_ss(&*a_panel.add(a_off + 3));
        c30 = _mm256_fmadd_ps(b0, a3, c30);
        c31 = _mm256_fmadd_ps(b1, a3, c31);

        let a4 = _mm256_broadcast_ss(&*a_panel.add(a_off + 4));
        c40 = _mm256_fmadd_ps(b0, a4, c40);
        c41 = _mm256_fmadd_ps(b1, a4, c41);

        let a5 = _mm256_broadcast_ss(&*a_panel.add(a_off + 5));
        c50 = _mm256_fmadd_ps(b0, a5, c50);
        c51 = _mm256_fmadd_ps(b1, a5, c51);
    }

    // Row pointers for C.
    let cp0 = c;
    let cp1 = c.add(ldc);
    let cp2 = c.add(2 * ldc);
    let cp3 = c.add(3 * ldc);
    let cp4 = c.add(4 * ldc);
    let cp5 = c.add(5 * ldc);

    // Accumulate existing C when running non-first k-block.
    if accumulate {
        c00 = _mm256_add_ps(_mm256_loadu_ps(cp0), c00);
        c01 = _mm256_add_ps(_mm256_loadu_ps(cp0.add(8)), c01);
        c10 = _mm256_add_ps(_mm256_loadu_ps(cp1), c10);
        c11 = _mm256_add_ps(_mm256_loadu_ps(cp1.add(8)), c11);
        c20 = _mm256_add_ps(_mm256_loadu_ps(cp2), c20);
        c21 = _mm256_add_ps(_mm256_loadu_ps(cp2.add(8)), c21);
        c30 = _mm256_add_ps(_mm256_loadu_ps(cp3), c30);
        c31 = _mm256_add_ps(_mm256_loadu_ps(cp3.add(8)), c31);
        c40 = _mm256_add_ps(_mm256_loadu_ps(cp4), c40);
        c41 = _mm256_add_ps(_mm256_loadu_ps(cp4.add(8)), c41);
        c50 = _mm256_add_ps(_mm256_loadu_ps(cp5), c50);
        c51 = _mm256_add_ps(_mm256_loadu_ps(cp5.add(8)), c51);
    }

    // Epilogue (bias + residual + activation) only on last k-block.
    if is_last_k {
        if let Some(bias) = epilogue.bias {
            let bv0 = _mm256_loadu_ps(bias.add(col_offset));
            let bv1 = _mm256_loadu_ps(bias.add(col_offset + 8));
            c00 = _mm256_add_ps(c00, bv0);
            c01 = _mm256_add_ps(c01, bv1);
            c10 = _mm256_add_ps(c10, bv0);
            c11 = _mm256_add_ps(c11, bv1);
            c20 = _mm256_add_ps(c20, bv0);
            c21 = _mm256_add_ps(c21, bv1);
            c30 = _mm256_add_ps(c30, bv0);
            c31 = _mm256_add_ps(c31, bv1);
            c40 = _mm256_add_ps(c40, bv0);
            c41 = _mm256_add_ps(c41, bv1);
            c50 = _mm256_add_ps(c50, bv0);
            c51 = _mm256_add_ps(c51, bv1);
        }
        // Residual is already shifted by caller to this tile's base.
        // Stride matches ldc (= n).
        if let Some(res_tile) = residual_tile {
            let rp0 = res_tile;
            let rp1 = res_tile.add(ldc);
            let rp2 = res_tile.add(2 * ldc);
            let rp3 = res_tile.add(3 * ldc);
            let rp4 = res_tile.add(4 * ldc);
            let rp5 = res_tile.add(5 * ldc);
            c00 = _mm256_add_ps(c00, _mm256_loadu_ps(rp0));
            c01 = _mm256_add_ps(c01, _mm256_loadu_ps(rp0.add(8)));
            c10 = _mm256_add_ps(c10, _mm256_loadu_ps(rp1));
            c11 = _mm256_add_ps(c11, _mm256_loadu_ps(rp1.add(8)));
            c20 = _mm256_add_ps(c20, _mm256_loadu_ps(rp2));
            c21 = _mm256_add_ps(c21, _mm256_loadu_ps(rp2.add(8)));
            c30 = _mm256_add_ps(c30, _mm256_loadu_ps(rp3));
            c31 = _mm256_add_ps(c31, _mm256_loadu_ps(rp3.add(8)));
            c40 = _mm256_add_ps(c40, _mm256_loadu_ps(rp4));
            c41 = _mm256_add_ps(c41, _mm256_loadu_ps(rp4.add(8)));
            c50 = _mm256_add_ps(c50, _mm256_loadu_ps(rp5));
            c51 = _mm256_add_ps(c51, _mm256_loadu_ps(rp5.add(8)));
        }
        match epilogue.activation {
            Activation::Relu => {
                let zero = _mm256_setzero_ps();
                c00 = _mm256_max_ps(c00, zero);
                c01 = _mm256_max_ps(c01, zero);
                c10 = _mm256_max_ps(c10, zero);
                c11 = _mm256_max_ps(c11, zero);
                c20 = _mm256_max_ps(c20, zero);
                c21 = _mm256_max_ps(c21, zero);
                c30 = _mm256_max_ps(c30, zero);
                c31 = _mm256_max_ps(c31, zero);
                c40 = _mm256_max_ps(c40, zero);
                c41 = _mm256_max_ps(c41, zero);
                c50 = _mm256_max_ps(c50, zero);
                c51 = _mm256_max_ps(c51, zero);
            }
            Activation::Silu => {
                c00 = silu_avx_fma(c00);
                c01 = silu_avx_fma(c01);
                c10 = silu_avx_fma(c10);
                c11 = silu_avx_fma(c11);
                c20 = silu_avx_fma(c20);
                c21 = silu_avx_fma(c21);
                c30 = silu_avx_fma(c30);
                c31 = silu_avx_fma(c31);
                c40 = silu_avx_fma(c40);
                c41 = silu_avx_fma(c41);
                c50 = silu_avx_fma(c50);
                c51 = silu_avx_fma(c51);
            }
            Activation::None => {}
        }
    }

    _mm256_storeu_ps(cp0, c00);
    _mm256_storeu_ps(cp0.add(8), c01);
    _mm256_storeu_ps(cp1, c10);
    _mm256_storeu_ps(cp1.add(8), c11);
    _mm256_storeu_ps(cp2, c20);
    _mm256_storeu_ps(cp2.add(8), c21);
    _mm256_storeu_ps(cp3, c30);
    _mm256_storeu_ps(cp3.add(8), c31);
    _mm256_storeu_ps(cp4, c40);
    _mm256_storeu_ps(cp4.add(8), c41);
    _mm256_storeu_ps(cp5, c50);
    _mm256_storeu_ps(cp5.add(8), c51);
}

// ---------------------------------------------------------------------------
// AVX+FMA micro-kernel (x86/x86_64)
// ---------------------------------------------------------------------------

/// AVX+FMA 4×8 microkernel with inline asm k-loop.
///
/// The k-loop is hand-scheduled to maximize FMA throughput on Zen 4
/// (2 FMA ports, 4-cycle latency → need 8+ independent FMAs).
/// Uses `core::arch::asm!` to avoid LLVM stack frame overhead.
#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx,fma")]
unsafe fn microkernel_4x8_avx_fma(
    a_panel: *const f32,
    b_panel: *const f32,
    c: *mut f32,
    ldc: usize,
    kc: usize,
    accumulate: bool,
    epilogue: GemmEpilogue,
    is_last_k: bool,
    col_offset: usize,
) {
    // Accumulator registers: ymm0-ymm3 (c0-c3)
    // Temp registers: ymm4 (b_vec), ymm5-ymm8 (a broadcasts)
    let mut c0: std::arch::x86_64::__m256;
    let mut c1: std::arch::x86_64::__m256;
    let mut c2: std::arch::x86_64::__m256;
    let mut c3: std::arch::x86_64::__m256;

    // K-loop in inline asm — no stack frame, pure register computation.
    // a_panel: MR=4 packed floats per k (16 bytes stride)
    // b_panel: NR=8 packed floats per k (32 bytes stride)
    std::arch::asm!(
        // Zero accumulators
        "vxorps {c0}, {c0}, {c0}",
        "vxorps {c1}, {c1}, {c1}",
        "vxorps {c2}, {c2}, {c2}",
        "vxorps {c3}, {c3}, {c3}",

        // kc_pairs = kc >> 1
        "mov {pairs}, {kc}",
        "shr {pairs}, 1",
        "test {pairs}, {pairs}",
        "jz 2f",

        // Main loop: 2 k-steps per iteration
        ".p2align 4",
        "3:",
        // k-step 0
        "vmovups {bv}, [{bp}]",
        "vbroadcastss {t0}, [{ap}]",
        "vfmadd231ps {c0}, {bv}, {t0}",
        "vbroadcastss {t0}, [{ap} + 4]",
        "vfmadd231ps {c1}, {bv}, {t0}",
        "vbroadcastss {t0}, [{ap} + 8]",
        "vfmadd231ps {c2}, {bv}, {t0}",
        "vbroadcastss {t0}, [{ap} + 12]",
        "vfmadd231ps {c3}, {bv}, {t0}",
        // k-step 1 with prefetch
        "vmovups {bv}, [{bp} + 32]",
        "prefetcht0 [{ap} + 64]",
        "vbroadcastss {t0}, [{ap} + 16]",
        "vfmadd231ps {c0}, {bv}, {t0}",
        "vbroadcastss {t0}, [{ap} + 20]",
        "vfmadd231ps {c1}, {bv}, {t0}",
        "vbroadcastss {t0}, [{ap} + 24]",
        "vfmadd231ps {c2}, {bv}, {t0}",
        "vbroadcastss {t0}, [{ap} + 28]",
        "vfmadd231ps {c3}, {bv}, {t0}",
        // Advance pointers
        "add {ap}, 32",  // 2 * MR * 4 bytes
        "add {bp}, 64",  // 2 * NR * 4 bytes
        "dec {pairs}",
        "jnz 3b",

        "2:",
        // Handle odd remainder
        "test {kc}, 1",
        "jz 4f",
        "vmovups {bv}, [{bp}]",
        "vbroadcastss {t0}, [{ap}]",
        "vfmadd231ps {c0}, {bv}, {t0}",
        "vbroadcastss {t0}, [{ap} + 4]",
        "vfmadd231ps {c1}, {bv}, {t0}",
        "vbroadcastss {t0}, [{ap} + 8]",
        "vfmadd231ps {c2}, {bv}, {t0}",
        "vbroadcastss {t0}, [{ap} + 12]",
        "vfmadd231ps {c3}, {bv}, {t0}",
        "4:",

        ap = inout(reg) a_panel => _,
        bp = inout(reg) b_panel => _,
        kc = in(reg) kc,
        pairs = out(reg) _,
        c0 = out(ymm_reg) c0,
        c1 = out(ymm_reg) c1,
        c2 = out(ymm_reg) c2,
        c3 = out(ymm_reg) c3,
        bv = out(ymm_reg) _,
        t0 = out(ymm_reg) _,
        options(nostack),
    );

    // Store phase (Rust intrinsics — LLVM handles well)
    let cp0 = c;
    let cp1 = c.add(ldc);
    let cp2 = c.add(2 * ldc);
    let cp3 = c.add(3 * ldc);

    if accumulate {
        c0 = _mm256_add_ps(_mm256_loadu_ps(cp0), c0);
        c1 = _mm256_add_ps(_mm256_loadu_ps(cp1), c1);
        c2 = _mm256_add_ps(_mm256_loadu_ps(cp2), c2);
        c3 = _mm256_add_ps(_mm256_loadu_ps(cp3), c3);
    }

    if is_last_k {
        if let Some(bias) = epilogue.bias {
            let bv = _mm256_loadu_ps(bias.add(col_offset));
            c0 = _mm256_add_ps(c0, bv);
            c1 = _mm256_add_ps(c1, bv);
            c2 = _mm256_add_ps(c2, bv);
            c3 = _mm256_add_ps(c3, bv);
        }
        match epilogue.activation {
            Activation::Relu => {
                let zero = _mm256_setzero_ps();
                c0 = _mm256_max_ps(c0, zero);
                c1 = _mm256_max_ps(c1, zero);
                c2 = _mm256_max_ps(c2, zero);
                c3 = _mm256_max_ps(c3, zero);
            }
            Activation::Silu => {
                c0 = silu_avx_fma(c0);
                c1 = silu_avx_fma(c1);
                c2 = silu_avx_fma(c2);
                c3 = silu_avx_fma(c3);
            }
            Activation::None => {}
        }
    }

    _mm256_storeu_ps(cp0, c0);
    _mm256_storeu_ps(cp1, c1);
    _mm256_storeu_ps(cp2, c2);
    _mm256_storeu_ps(cp3, c3);
}

/// Fallback for x86 (32-bit) — uses Rust intrinsics.
#[cfg(target_arch = "x86")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx,fma")]
unsafe fn microkernel_4x8_avx_fma(
    a_panel: *const f32,
    b_panel: *const f32,
    c: *mut f32,
    ldc: usize,
    kc: usize,
    accumulate: bool,
    epilogue: GemmEpilogue,
    is_last_k: bool,
    col_offset: usize,
) {
    // x86 (32-bit) fallback — same logic, Rust intrinsics
    let mut c0: __m256 = _mm256_setzero_ps();
    let mut c1: __m256 = _mm256_setzero_ps();
    let mut c2: __m256 = _mm256_setzero_ps();
    let mut c3: __m256 = _mm256_setzero_ps();

    for p in 0..kc {
        let b_vec = _mm256_loadu_ps(b_panel.add(p * NR));
        c0 = _mm256_fmadd_ps(_mm256_set1_ps(*a_panel.add(p * MR)), b_vec, c0);
        c1 = _mm256_fmadd_ps(_mm256_set1_ps(*a_panel.add(p * MR + 1)), b_vec, c1);
        c2 = _mm256_fmadd_ps(_mm256_set1_ps(*a_panel.add(p * MR + 2)), b_vec, c2);
        c3 = _mm256_fmadd_ps(_mm256_set1_ps(*a_panel.add(p * MR + 3)), b_vec, c3);
    }

    let cp0 = c;
    let cp1 = c.add(ldc);
    let cp2 = c.add(2 * ldc);
    let cp3 = c.add(3 * ldc);

    if accumulate {
        c0 = _mm256_add_ps(_mm256_loadu_ps(cp0), c0);
        c1 = _mm256_add_ps(_mm256_loadu_ps(cp1), c1);
        c2 = _mm256_add_ps(_mm256_loadu_ps(cp2), c2);
        c3 = _mm256_add_ps(_mm256_loadu_ps(cp3), c3);
    }
    if is_last_k {
        if let Some(bias) = epilogue.bias {
            let bv = _mm256_loadu_ps(bias.add(col_offset));
            c0 = _mm256_add_ps(c0, bv);
            c1 = _mm256_add_ps(c1, bv);
            c2 = _mm256_add_ps(c2, bv);
            c3 = _mm256_add_ps(c3, bv);
        }
        match epilogue.activation {
            Activation::Relu => {
                let zero = _mm256_setzero_ps();
                c0 = _mm256_max_ps(c0, zero);
                c1 = _mm256_max_ps(c1, zero);
                c2 = _mm256_max_ps(c2, zero);
                c3 = _mm256_max_ps(c3, zero);
            }
            Activation::Silu => {
                c0 = silu_avx_fma(c0);
                c1 = silu_avx_fma(c1);
                c2 = silu_avx_fma(c2);
                c3 = silu_avx_fma(c3);
            }
            Activation::None => {}
        }
    }
    _mm256_storeu_ps(cp0, c0);
    _mm256_storeu_ps(cp1, c1);
    _mm256_storeu_ps(cp2, c2);
    _mm256_storeu_ps(cp3, c3);
}

// ---------------------------------------------------------------------------
// AVX micro-kernel (x86/x86_64)
// ---------------------------------------------------------------------------

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code)]
#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx")]
unsafe fn microkernel_4x8_avx(
    a_panel: *const f32,
    b_panel: *const f32,
    c: *mut f32,
    ldc: usize,
    kc: usize,
    accumulate: bool,
    epilogue: GemmEpilogue,
    is_last_k: bool,
    col_offset: usize,
) {
    let mut c0: __m256 = _mm256_setzero_ps();
    let mut c1: __m256 = _mm256_setzero_ps();
    let mut c2: __m256 = _mm256_setzero_ps();
    let mut c3: __m256 = _mm256_setzero_ps();

    for p in 0..kc {
        let b_vec = _mm256_loadu_ps(b_panel.add(p * NR));

        let a0 = _mm256_set1_ps(*a_panel.add(p * MR));
        c0 = _mm256_add_ps(c0, _mm256_mul_ps(a0, b_vec));

        let a1 = _mm256_set1_ps(*a_panel.add(p * MR + 1));
        c1 = _mm256_add_ps(c1, _mm256_mul_ps(a1, b_vec));

        let a2 = _mm256_set1_ps(*a_panel.add(p * MR + 2));
        c2 = _mm256_add_ps(c2, _mm256_mul_ps(a2, b_vec));

        let a3 = _mm256_set1_ps(*a_panel.add(p * MR + 3));
        c3 = _mm256_add_ps(c3, _mm256_mul_ps(a3, b_vec));
    }

    let cp0 = c;
    let cp1 = c.add(ldc);
    let cp2 = c.add(2 * ldc);
    let cp3 = c.add(3 * ldc);

    if accumulate {
        c0 = _mm256_add_ps(_mm256_loadu_ps(cp0), c0);
        c1 = _mm256_add_ps(_mm256_loadu_ps(cp1), c1);
        c2 = _mm256_add_ps(_mm256_loadu_ps(cp2), c2);
        c3 = _mm256_add_ps(_mm256_loadu_ps(cp3), c3);
    }

    if is_last_k {
        if let Some(bias) = epilogue.bias {
            let bv = _mm256_loadu_ps(bias.add(col_offset));
            c0 = _mm256_add_ps(c0, bv);
            c1 = _mm256_add_ps(c1, bv);
            c2 = _mm256_add_ps(c2, bv);
            c3 = _mm256_add_ps(c3, bv);
        }
        match epilogue.activation {
            Activation::Relu => {
                let zero = _mm256_setzero_ps();
                c0 = _mm256_max_ps(c0, zero);
                c1 = _mm256_max_ps(c1, zero);
                c2 = _mm256_max_ps(c2, zero);
                c3 = _mm256_max_ps(c3, zero);
            }
            Activation::Silu => {
                c0 = silu_avx(c0);
                c1 = silu_avx(c1);
                c2 = silu_avx(c2);
                c3 = silu_avx(c3);
            }
            Activation::None => {}
        }
    }

    _mm256_storeu_ps(cp0, c0);
    _mm256_storeu_ps(cp1, c1);
    _mm256_storeu_ps(cp2, c2);
    _mm256_storeu_ps(cp3, c3);
}

// ---------------------------------------------------------------------------
// SSE micro-kernel (x86/x86_64)
// ---------------------------------------------------------------------------

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code)]
#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "sse")]
unsafe fn microkernel_4x8_sse(
    a_panel: *const f32,
    b_panel: *const f32,
    c: *mut f32,
    ldc: usize,
    kc: usize,
    accumulate: bool,
    epilogue: GemmEpilogue,
    is_last_k: bool,
    col_offset: usize,
) {
    let mut c00: __m128 = _mm_setzero_ps();
    let mut c01: __m128 = _mm_setzero_ps();
    let mut c10: __m128 = _mm_setzero_ps();
    let mut c11: __m128 = _mm_setzero_ps();
    let mut c20: __m128 = _mm_setzero_ps();
    let mut c21: __m128 = _mm_setzero_ps();
    let mut c30: __m128 = _mm_setzero_ps();
    let mut c31: __m128 = _mm_setzero_ps();

    for p in 0..kc {
        let a_base = p * MR;
        let b_base = p * NR;

        let a0 = _mm_set1_ps(*a_panel.add(a_base));
        let a1 = _mm_set1_ps(*a_panel.add(a_base + 1));
        let a2 = _mm_set1_ps(*a_panel.add(a_base + 2));
        let a3 = _mm_set1_ps(*a_panel.add(a_base + 3));

        let b0 = _mm_loadu_ps(b_panel.add(b_base));
        let b1 = _mm_loadu_ps(b_panel.add(b_base + 4));

        c00 = _mm_add_ps(c00, _mm_mul_ps(a0, b0));
        c01 = _mm_add_ps(c01, _mm_mul_ps(a0, b1));
        c10 = _mm_add_ps(c10, _mm_mul_ps(a1, b0));
        c11 = _mm_add_ps(c11, _mm_mul_ps(a1, b1));
        c20 = _mm_add_ps(c20, _mm_mul_ps(a2, b0));
        c21 = _mm_add_ps(c21, _mm_mul_ps(a2, b1));
        c30 = _mm_add_ps(c30, _mm_mul_ps(a3, b0));
        c31 = _mm_add_ps(c31, _mm_mul_ps(a3, b1));
    }

    let cp0 = c;
    let cp1 = c.add(ldc);
    let cp2 = c.add(2 * ldc);
    let cp3 = c.add(3 * ldc);

    if accumulate {
        c00 = _mm_add_ps(_mm_loadu_ps(cp0), c00);
        c01 = _mm_add_ps(_mm_loadu_ps(cp0.add(4)), c01);
        c10 = _mm_add_ps(_mm_loadu_ps(cp1), c10);
        c11 = _mm_add_ps(_mm_loadu_ps(cp1.add(4)), c11);
        c20 = _mm_add_ps(_mm_loadu_ps(cp2), c20);
        c21 = _mm_add_ps(_mm_loadu_ps(cp2.add(4)), c21);
        c30 = _mm_add_ps(_mm_loadu_ps(cp3), c30);
        c31 = _mm_add_ps(_mm_loadu_ps(cp3.add(4)), c31);
    }

    if is_last_k {
        if let Some(bias) = epilogue.bias {
            let bv0 = _mm_loadu_ps(bias.add(col_offset));
            let bv1 = _mm_loadu_ps(bias.add(col_offset + 4));
            c00 = _mm_add_ps(c00, bv0);
            c01 = _mm_add_ps(c01, bv1);
            c10 = _mm_add_ps(c10, bv0);
            c11 = _mm_add_ps(c11, bv1);
            c20 = _mm_add_ps(c20, bv0);
            c21 = _mm_add_ps(c21, bv1);
            c30 = _mm_add_ps(c30, bv0);
            c31 = _mm_add_ps(c31, bv1);
        }
        match epilogue.activation {
            Activation::Relu => {
                let zero = _mm_setzero_ps();
                c00 = _mm_max_ps(c00, zero);
                c01 = _mm_max_ps(c01, zero);
                c10 = _mm_max_ps(c10, zero);
                c11 = _mm_max_ps(c11, zero);
                c20 = _mm_max_ps(c20, zero);
                c21 = _mm_max_ps(c21, zero);
                c30 = _mm_max_ps(c30, zero);
                c31 = _mm_max_ps(c31, zero);
            }
            Activation::Silu => {
                c00 = silu_sse(c00);
                c01 = silu_sse(c01);
                c10 = silu_sse(c10);
                c11 = silu_sse(c11);
                c20 = silu_sse(c20);
                c21 = silu_sse(c21);
                c30 = silu_sse(c30);
                c31 = silu_sse(c31);
            }
            Activation::None => {}
        }
    }

    _mm_storeu_ps(cp0, c00);
    _mm_storeu_ps(cp0.add(4), c01);
    _mm_storeu_ps(cp1, c10);
    _mm_storeu_ps(cp1.add(4), c11);
    _mm_storeu_ps(cp2, c20);
    _mm_storeu_ps(cp2.add(4), c21);
    _mm_storeu_ps(cp3, c30);
    _mm_storeu_ps(cp3.add(4), c31);
}

// ---------------------------------------------------------------------------
// 4×24 AVX+FMA micro-kernel: three adjacent NR=8 tiles at once.
// 12 ymm accumulators + 3 B loads + 1 A broadcast = 16 ymm registers (exact fit).
// Reduces microkernel call count by 50% vs 4×16 for wide N.
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx,fma")]
#[inline]
unsafe fn microkernel_4x24_avx_fma(
    a_panel: *const f32,
    b_panel_0: *const f32,
    b_panel_1: *const f32,
    b_panel_2: *const f32,
    c: *mut f32,
    ldc: usize,
    kc: usize,
    accumulate: bool,
    epilogue: GemmEpilogue,
    is_last_k: bool,
    col_offset: usize,
    // Per-tile residual pointer (already shifted by `(ic+ir)*n + col_offset`
    // at dispatch site). When `Some` and `is_last_k`, the epilogue adds
    // 12 residual YMMs to the accumulators BEFORE applying activation,
    // folding a separate `add_inplace` pass into the GEMM store. Stride
    // matches `ldc = n`.
    residual_tile: Option<*const f32>,
) {
    // 12 accumulators (4 rows × 3 panels of 8 cols)
    let mut c00: __m256;
    let mut c01: __m256;
    let mut c02: __m256;
    let mut c10: __m256;
    let mut c11: __m256;
    let mut c12: __m256;
    let mut c20: __m256;
    let mut c21: __m256;
    let mut c22: __m256;
    let mut c30: __m256;
    let mut c31: __m256;
    let mut c32: __m256;

    std::arch::asm!(
        // Zero 12 accumulators
        "vxorps {c00}, {c00}, {c00}",
        "vxorps {c01}, {c01}, {c01}",
        "vxorps {c02}, {c02}, {c02}",
        "vxorps {c10}, {c10}, {c10}",
        "vxorps {c11}, {c11}, {c11}",
        "vxorps {c12}, {c12}, {c12}",
        "vxorps {c20}, {c20}, {c20}",
        "vxorps {c21}, {c21}, {c21}",
        "vxorps {c22}, {c22}, {c22}",
        "vxorps {c30}, {c30}, {c30}",
        "vxorps {c31}, {c31}, {c31}",
        "vxorps {c32}, {c32}, {c32}",

        "mov {pairs}, {kc}",
        "shr {pairs}, 1",
        "test {pairs}, {pairs}",
        "jz 24f",

        ".p2align 4",
        "23:",
        // --- k-step 0 ---
        "vmovups {b0}, [{bp0}]",
        "vmovups {b1}, [{bp1}]",
        "vmovups {b2}, [{bp2}]",
        "vbroadcastss {av}, [{ap}]",
        "vfmadd231ps {c00}, {b0}, {av}",
        "vfmadd231ps {c01}, {b1}, {av}",
        "vfmadd231ps {c02}, {b2}, {av}",
        "vbroadcastss {av}, [{ap} + 4]",
        "vfmadd231ps {c10}, {b0}, {av}",
        "vfmadd231ps {c11}, {b1}, {av}",
        "vfmadd231ps {c12}, {b2}, {av}",
        "vbroadcastss {av}, [{ap} + 8]",
        "vfmadd231ps {c20}, {b0}, {av}",
        "vfmadd231ps {c21}, {b1}, {av}",
        "vfmadd231ps {c22}, {b2}, {av}",
        "vbroadcastss {av}, [{ap} + 12]",
        "vfmadd231ps {c30}, {b0}, {av}",
        "vfmadd231ps {c31}, {b1}, {av}",
        "vfmadd231ps {c32}, {b2}, {av}",
        // --- k-step 1 ---
        "vmovups {b0}, [{bp0} + 32]",
        "vmovups {b1}, [{bp1} + 32]",
        "vmovups {b2}, [{bp2} + 32]",
        "vbroadcastss {av}, [{ap} + 16]",
        "vfmadd231ps {c00}, {b0}, {av}",
        "vfmadd231ps {c01}, {b1}, {av}",
        "vfmadd231ps {c02}, {b2}, {av}",
        "vbroadcastss {av}, [{ap} + 20]",
        "vfmadd231ps {c10}, {b0}, {av}",
        "vfmadd231ps {c11}, {b1}, {av}",
        "vfmadd231ps {c12}, {b2}, {av}",
        "vbroadcastss {av}, [{ap} + 24]",
        "vfmadd231ps {c20}, {b0}, {av}",
        "vfmadd231ps {c21}, {b1}, {av}",
        "vfmadd231ps {c22}, {b2}, {av}",
        "vbroadcastss {av}, [{ap} + 28]",
        "vfmadd231ps {c30}, {b0}, {av}",
        "vfmadd231ps {c31}, {b1}, {av}",
        "vfmadd231ps {c32}, {b2}, {av}",
        // Advance both k-steps at once
        "add {ap}, 32",   // 2 * MR * 4 bytes
        "add {bp0}, 64",  // 2 * NR * 4 bytes
        "add {bp1}, 64",
        "add {bp2}, 64",
        "dec {pairs}",
        "jnz 23b",

        "24:",
        // Handle odd k-remainder (if kc was odd, one final k-step)
        "test {kc}, 1",
        "jz 25f",
        "vmovups {b0}, [{bp0}]",
        "vmovups {b1}, [{bp1}]",
        "vmovups {b2}, [{bp2}]",
        "vbroadcastss {av}, [{ap}]",
        "vfmadd231ps {c00}, {b0}, {av}",
        "vfmadd231ps {c01}, {b1}, {av}",
        "vfmadd231ps {c02}, {b2}, {av}",
        "vbroadcastss {av}, [{ap} + 4]",
        "vfmadd231ps {c10}, {b0}, {av}",
        "vfmadd231ps {c11}, {b1}, {av}",
        "vfmadd231ps {c12}, {b2}, {av}",
        "vbroadcastss {av}, [{ap} + 8]",
        "vfmadd231ps {c20}, {b0}, {av}",
        "vfmadd231ps {c21}, {b1}, {av}",
        "vfmadd231ps {c22}, {b2}, {av}",
        "vbroadcastss {av}, [{ap} + 12]",
        "vfmadd231ps {c30}, {b0}, {av}",
        "vfmadd231ps {c31}, {b1}, {av}",
        "vfmadd231ps {c32}, {b2}, {av}",
        "25:",

        ap = inout(reg) a_panel => _,
        bp0 = inout(reg) b_panel_0 => _,
        bp1 = inout(reg) b_panel_1 => _,
        bp2 = inout(reg) b_panel_2 => _,
        kc = in(reg) kc,
        pairs = out(reg) _,
        c00 = out(ymm_reg) c00, c01 = out(ymm_reg) c01, c02 = out(ymm_reg) c02,
        c10 = out(ymm_reg) c10, c11 = out(ymm_reg) c11, c12 = out(ymm_reg) c12,
        c20 = out(ymm_reg) c20, c21 = out(ymm_reg) c21, c22 = out(ymm_reg) c22,
        c30 = out(ymm_reg) c30, c31 = out(ymm_reg) c31, c32 = out(ymm_reg) c32,
        b0 = out(ymm_reg) _, b1 = out(ymm_reg) _, b2 = out(ymm_reg) _,
        av = out(ymm_reg) _,
        options(nostack),
    );

    // Store phase
    let cp0 = c;
    let cp1 = c.add(ldc);
    let cp2 = c.add(2 * ldc);
    let cp3 = c.add(3 * ldc);

    if accumulate {
        c00 = _mm256_add_ps(_mm256_loadu_ps(cp0), c00);
        c01 = _mm256_add_ps(_mm256_loadu_ps(cp0.add(8)), c01);
        c02 = _mm256_add_ps(_mm256_loadu_ps(cp0.add(16)), c02);
        c10 = _mm256_add_ps(_mm256_loadu_ps(cp1), c10);
        c11 = _mm256_add_ps(_mm256_loadu_ps(cp1.add(8)), c11);
        c12 = _mm256_add_ps(_mm256_loadu_ps(cp1.add(16)), c12);
        c20 = _mm256_add_ps(_mm256_loadu_ps(cp2), c20);
        c21 = _mm256_add_ps(_mm256_loadu_ps(cp2.add(8)), c21);
        c22 = _mm256_add_ps(_mm256_loadu_ps(cp2.add(16)), c22);
        c30 = _mm256_add_ps(_mm256_loadu_ps(cp3), c30);
        c31 = _mm256_add_ps(_mm256_loadu_ps(cp3.add(8)), c31);
        c32 = _mm256_add_ps(_mm256_loadu_ps(cp3.add(16)), c32);
    }

    if is_last_k {
        if let Some(bias) = epilogue.bias {
            let bv0 = _mm256_loadu_ps(bias.add(col_offset));
            let bv1 = _mm256_loadu_ps(bias.add(col_offset + 8));
            let bv2 = _mm256_loadu_ps(bias.add(col_offset + 16));
            c00 = _mm256_add_ps(c00, bv0);
            c01 = _mm256_add_ps(c01, bv1);
            c02 = _mm256_add_ps(c02, bv2);
            c10 = _mm256_add_ps(c10, bv0);
            c11 = _mm256_add_ps(c11, bv1);
            c12 = _mm256_add_ps(c12, bv2);
            c20 = _mm256_add_ps(c20, bv0);
            c21 = _mm256_add_ps(c21, bv1);
            c22 = _mm256_add_ps(c22, bv2);
            c30 = _mm256_add_ps(c30, bv0);
            c31 = _mm256_add_ps(c31, bv1);
            c32 = _mm256_add_ps(c32, bv2);
        }
        // Residual add: out = acc + bias + residual (Phase 1.2). Load 12
        // YMMs from the pre-computed per-tile residual base. Stride same
        // as `ldc` (row k = base + k*ldc). Only fires on is_last_k so
        // residual isn't double-added across accumulator k-blocks.
        if let Some(res_tile) = residual_tile {
            let rp0 = res_tile;
            let rp1 = res_tile.add(ldc);
            let rp2 = res_tile.add(2 * ldc);
            let rp3 = res_tile.add(3 * ldc);
            c00 = _mm256_add_ps(c00, _mm256_loadu_ps(rp0));
            c01 = _mm256_add_ps(c01, _mm256_loadu_ps(rp0.add(8)));
            c02 = _mm256_add_ps(c02, _mm256_loadu_ps(rp0.add(16)));
            c10 = _mm256_add_ps(c10, _mm256_loadu_ps(rp1));
            c11 = _mm256_add_ps(c11, _mm256_loadu_ps(rp1.add(8)));
            c12 = _mm256_add_ps(c12, _mm256_loadu_ps(rp1.add(16)));
            c20 = _mm256_add_ps(c20, _mm256_loadu_ps(rp2));
            c21 = _mm256_add_ps(c21, _mm256_loadu_ps(rp2.add(8)));
            c22 = _mm256_add_ps(c22, _mm256_loadu_ps(rp2.add(16)));
            c30 = _mm256_add_ps(c30, _mm256_loadu_ps(rp3));
            c31 = _mm256_add_ps(c31, _mm256_loadu_ps(rp3.add(8)));
            c32 = _mm256_add_ps(c32, _mm256_loadu_ps(rp3.add(16)));
        }
        match epilogue.activation {
            Activation::Relu => {
                let zero = _mm256_setzero_ps();
                c00 = _mm256_max_ps(c00, zero);
                c01 = _mm256_max_ps(c01, zero);
                c02 = _mm256_max_ps(c02, zero);
                c10 = _mm256_max_ps(c10, zero);
                c11 = _mm256_max_ps(c11, zero);
                c12 = _mm256_max_ps(c12, zero);
                c20 = _mm256_max_ps(c20, zero);
                c21 = _mm256_max_ps(c21, zero);
                c22 = _mm256_max_ps(c22, zero);
                c30 = _mm256_max_ps(c30, zero);
                c31 = _mm256_max_ps(c31, zero);
                c32 = _mm256_max_ps(c32, zero);
            }
            Activation::Silu => {
                c00 = silu_avx_fma(c00);
                c01 = silu_avx_fma(c01);
                c02 = silu_avx_fma(c02);
                c10 = silu_avx_fma(c10);
                c11 = silu_avx_fma(c11);
                c12 = silu_avx_fma(c12);
                c20 = silu_avx_fma(c20);
                c21 = silu_avx_fma(c21);
                c22 = silu_avx_fma(c22);
                c30 = silu_avx_fma(c30);
                c31 = silu_avx_fma(c31);
                c32 = silu_avx_fma(c32);
            }
            Activation::None => {}
        }
    }

    _mm256_storeu_ps(cp0, c00);
    _mm256_storeu_ps(cp0.add(8), c01);
    _mm256_storeu_ps(cp0.add(16), c02);
    _mm256_storeu_ps(cp1, c10);
    _mm256_storeu_ps(cp1.add(8), c11);
    _mm256_storeu_ps(cp1.add(16), c12);
    _mm256_storeu_ps(cp2, c20);
    _mm256_storeu_ps(cp2.add(8), c21);
    _mm256_storeu_ps(cp2.add(16), c22);
    _mm256_storeu_ps(cp3, c30);
    _mm256_storeu_ps(cp3.add(8), c31);
    _mm256_storeu_ps(cp3.add(16), c32);
}

// ---------------------------------------------------------------------------
/// SiLU using AVX+FMA: x / (1 + exp(-x)), uses bit-trick exp for speed.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx,fma")]
#[inline]
unsafe fn silu_avx_fma(x: __m256) -> __m256 {
    let one = _mm256_set1_ps(1.0);
    let neg_x = _mm256_sub_ps(_mm256_setzero_ps(), x);
    let exp_neg = super::simd::exp::fast_exp_bittrick_avx(neg_x);
    let denom = _mm256_add_ps(one, exp_neg);
    _mm256_div_ps(x, denom)
}

/// SiLU using AVX (no FMA): x / (1 + exp(-x)), uses bit-trick exp.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx")]
#[inline]
unsafe fn silu_avx(x: __m256) -> __m256 {
    let one = _mm256_set1_ps(1.0);
    let neg_x = _mm256_sub_ps(_mm256_setzero_ps(), x);
    let exp_neg = super::simd::exp::fast_exp_bittrick_avx(neg_x);
    let denom = _mm256_add_ps(one, exp_neg);
    _mm256_div_ps(x, denom)
}

/// SiLU using SSE: x / (1 + exp(-x)), uses bit-trick exp.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "sse")]
#[inline]
unsafe fn silu_sse(x: __m128) -> __m128 {
    let one = _mm_set1_ps(1.0);
    let neg_x = _mm_sub_ps(_mm_setzero_ps(), x);
    let exp_neg = super::simd::exp::fast_exp_bittrick_sse(neg_x);
    let denom = _mm_add_ps(one, exp_neg);
    _mm_div_ps(x, denom)
}

/// SiLU using NEON: x / (1 + exp(-x)), uses fast sigmoid helper.
#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn silu_neon(x: float32x4_t) -> float32x4_t {
    let one = vdupq_n_f32(1.0);
    let neg_x = vnegq_f32(x);
    let exp_neg = super::simd::exp::fast_exp_sigmoid_neon(neg_x);
    let denom = vaddq_f32(one, exp_neg);
    vdivq_f32(x, denom)
}

// ============================================================================
// Step S.1′ — specialized low-k pointwise tile (MR=4 × NR=24, const K)
// ============================================================================
//
// Replaces `matmul_row_set_avx_fma` on hot low-k pointwise Conv shapes
// (k ∈ {16, 24}, n % 24 == 0, m % 4 == 0). The blocked GEMM path rejects
// k < 32 (`use_blocked` gate), so low-k shapes land in the per-row
// `row_gemm_set_parallel_fused`. Per-row processing has 3× less A-row
// reuse than a 4-row tile.
//
// Register plan: 12 YMM accumulators (4 rows × 3 NR-panels) + 3 YMM B
// + 1 YMM A broadcast = 16 YMM exact. No room for double-buffer, so
// we fully unroll K at compile time (LLVM schedules 16 or 24 k-steps
// as one straight-line block, eliminating the k-loop branch).
//
// B is read **unpacked** directly from NHWC kernel weight: stride=n
// floats per k-row. No pack-B needed. A is also unpacked (stride=k
// for the 4 rows of this tile).

/// Shape gate for the Step S.1′ low-k tile fast path. Hard-coded
/// MR=4, NR=24 alignment plus minimum FMA count (avoids small-matrix
/// regression where rayon dispatch dominates).
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline]
fn use_low_k_tile_avx_fma(m: usize, k: usize, n: usize) -> bool {
    const MIN_WORK_FMAS: usize = 1_048_576;
    (k == 16 || k == 24)
        && m != 0
        && n != 0
        && m.is_multiple_of(4)
        && n.is_multiple_of(24)
        && m.saturating_mul(n).saturating_mul(k) >= MIN_WORK_FMAS
}

/// Step S.1′ inner tile: 4 rows × NR=24 cols, const-K k-loop, stride-n
/// B reads. Writes bias+residual+activation into C directly (no
/// accumulate path — we always overwrite).
///
/// # Safety
/// - `a` must point to ≥ 4*k contiguous floats laid out as 4 rows of
///   k stride=`k` floats each (i.e. `a.add(row*k + p)` = A[tile_row+r, p]).
/// - `b_panel` must point to the first k-row of the 24-col panel in B
///   (= `b_base.add(panel_col)`), and `b_stride` is the n stride (row
///   width of B in floats).
/// - `c` must point to the output tile top-left (4 rows of 24 cols,
///   stride = `c_stride` floats). `c_stride` equals n.
/// - Caller verified CPU supports AVX+FMA at runtime.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx,fma")]
#[allow(unsafe_code, clippy::too_many_arguments)]
unsafe fn low_k_tile_4x24_avx_fma<const K: usize>(
    a: *const f32,
    b_panel: *const f32,
    b_stride: usize,
    c: *mut f32,
    c_stride: usize,
    epilogue: &GemmEpilogue,
    col_offset: usize,
    residual_tile: Option<*const f32>,
) {
    unsafe {
        #[cfg(target_arch = "x86")]
        use std::arch::x86::{
            _mm256_add_ps, _mm256_broadcast_ss, _mm256_fmadd_ps, _mm256_loadu_ps, _mm256_max_ps,
            _mm256_setzero_ps, _mm256_storeu_ps,
        };
        #[cfg(target_arch = "x86_64")]
        use std::arch::x86_64::{
            _mm256_add_ps, _mm256_broadcast_ss, _mm256_fmadd_ps, _mm256_loadu_ps, _mm256_max_ps,
            _mm256_setzero_ps, _mm256_storeu_ps,
        };

        let mut c00 = _mm256_setzero_ps();
        let mut c01 = _mm256_setzero_ps();
        let mut c02 = _mm256_setzero_ps();
        let mut c10 = _mm256_setzero_ps();
        let mut c11 = _mm256_setzero_ps();
        let mut c12 = _mm256_setzero_ps();
        let mut c20 = _mm256_setzero_ps();
        let mut c21 = _mm256_setzero_ps();
        let mut c22 = _mm256_setzero_ps();
        let mut c30 = _mm256_setzero_ps();
        let mut c31 = _mm256_setzero_ps();
        let mut c32 = _mm256_setzero_ps();

        // const-K loop — LLVM fully unrolls at compile time for K ∈ {16, 24}.
        for p in 0..K {
            let b_row = b_panel.add(p * b_stride);
            let b0 = _mm256_loadu_ps(b_row);
            let b1 = _mm256_loadu_ps(b_row.add(8));
            let b2 = _mm256_loadu_ps(b_row.add(16));
            let a0 = _mm256_broadcast_ss(&*a.add(p));
            c00 = _mm256_fmadd_ps(a0, b0, c00);
            c01 = _mm256_fmadd_ps(a0, b1, c01);
            c02 = _mm256_fmadd_ps(a0, b2, c02);
            let a1 = _mm256_broadcast_ss(&*a.add(K + p));
            c10 = _mm256_fmadd_ps(a1, b0, c10);
            c11 = _mm256_fmadd_ps(a1, b1, c11);
            c12 = _mm256_fmadd_ps(a1, b2, c12);
            let a2 = _mm256_broadcast_ss(&*a.add(2 * K + p));
            c20 = _mm256_fmadd_ps(a2, b0, c20);
            c21 = _mm256_fmadd_ps(a2, b1, c21);
            c22 = _mm256_fmadd_ps(a2, b2, c22);
            let a3 = _mm256_broadcast_ss(&*a.add(3 * K + p));
            c30 = _mm256_fmadd_ps(a3, b0, c30);
            c31 = _mm256_fmadd_ps(a3, b1, c31);
            c32 = _mm256_fmadd_ps(a3, b2, c32);
        }

        // Epilogue — bias + residual + activation all inline while
        // accumulators are still hot in registers.
        if let Some(bias) = epilogue.bias {
            let bv0 = _mm256_loadu_ps(bias.add(col_offset));
            let bv1 = _mm256_loadu_ps(bias.add(col_offset + 8));
            let bv2 = _mm256_loadu_ps(bias.add(col_offset + 16));
            c00 = _mm256_add_ps(c00, bv0);
            c01 = _mm256_add_ps(c01, bv1);
            c02 = _mm256_add_ps(c02, bv2);
            c10 = _mm256_add_ps(c10, bv0);
            c11 = _mm256_add_ps(c11, bv1);
            c12 = _mm256_add_ps(c12, bv2);
            c20 = _mm256_add_ps(c20, bv0);
            c21 = _mm256_add_ps(c21, bv1);
            c22 = _mm256_add_ps(c22, bv2);
            c30 = _mm256_add_ps(c30, bv0);
            c31 = _mm256_add_ps(c31, bv1);
            c32 = _mm256_add_ps(c32, bv2);
        }
        if let Some(res) = residual_tile {
            let rp0 = res;
            let rp1 = res.add(c_stride);
            let rp2 = res.add(2 * c_stride);
            let rp3 = res.add(3 * c_stride);
            c00 = _mm256_add_ps(c00, _mm256_loadu_ps(rp0));
            c01 = _mm256_add_ps(c01, _mm256_loadu_ps(rp0.add(8)));
            c02 = _mm256_add_ps(c02, _mm256_loadu_ps(rp0.add(16)));
            c10 = _mm256_add_ps(c10, _mm256_loadu_ps(rp1));
            c11 = _mm256_add_ps(c11, _mm256_loadu_ps(rp1.add(8)));
            c12 = _mm256_add_ps(c12, _mm256_loadu_ps(rp1.add(16)));
            c20 = _mm256_add_ps(c20, _mm256_loadu_ps(rp2));
            c21 = _mm256_add_ps(c21, _mm256_loadu_ps(rp2.add(8)));
            c22 = _mm256_add_ps(c22, _mm256_loadu_ps(rp2.add(16)));
            c30 = _mm256_add_ps(c30, _mm256_loadu_ps(rp3));
            c31 = _mm256_add_ps(c31, _mm256_loadu_ps(rp3.add(8)));
            c32 = _mm256_add_ps(c32, _mm256_loadu_ps(rp3.add(16)));
        }
        match epilogue.activation {
            Activation::Relu => {
                let z = _mm256_setzero_ps();
                c00 = _mm256_max_ps(c00, z);
                c01 = _mm256_max_ps(c01, z);
                c02 = _mm256_max_ps(c02, z);
                c10 = _mm256_max_ps(c10, z);
                c11 = _mm256_max_ps(c11, z);
                c12 = _mm256_max_ps(c12, z);
                c20 = _mm256_max_ps(c20, z);
                c21 = _mm256_max_ps(c21, z);
                c22 = _mm256_max_ps(c22, z);
                c30 = _mm256_max_ps(c30, z);
                c31 = _mm256_max_ps(c31, z);
                c32 = _mm256_max_ps(c32, z);
            }
            Activation::Silu => {
                c00 = silu_avx_fma(c00);
                c01 = silu_avx_fma(c01);
                c02 = silu_avx_fma(c02);
                c10 = silu_avx_fma(c10);
                c11 = silu_avx_fma(c11);
                c12 = silu_avx_fma(c12);
                c20 = silu_avx_fma(c20);
                c21 = silu_avx_fma(c21);
                c22 = silu_avx_fma(c22);
                c30 = silu_avx_fma(c30);
                c31 = silu_avx_fma(c31);
                c32 = silu_avx_fma(c32);
            }
            Activation::None => {}
        }

        let cp0 = c;
        let cp1 = c.add(c_stride);
        let cp2 = c.add(2 * c_stride);
        let cp3 = c.add(3 * c_stride);
        _mm256_storeu_ps(cp0, c00);
        _mm256_storeu_ps(cp0.add(8), c01);
        _mm256_storeu_ps(cp0.add(16), c02);
        _mm256_storeu_ps(cp1, c10);
        _mm256_storeu_ps(cp1.add(8), c11);
        _mm256_storeu_ps(cp1.add(16), c12);
        _mm256_storeu_ps(cp2, c20);
        _mm256_storeu_ps(cp2.add(8), c21);
        _mm256_storeu_ps(cp2.add(16), c22);
        _mm256_storeu_ps(cp3, c30);
        _mm256_storeu_ps(cp3.add(8), c31);
        _mm256_storeu_ps(cp3.add(16), c32);
    }
}

/// Step S.1′ orchestrator. Chunks output by 4 rows × n cols, iterates
/// the n/24 panels inside each chunk. Routes MT dispatch through the
/// crate-wide `scope_ctx::par_chunks_mut_dispatch` — same infrastructure
/// as `row_gemm_set_parallel_fused`, so any pool the runner installed
/// (rayon, PersistentSection) stays consistent.
///
/// # Safety
/// - `a` length ≥ m*k; `b` length ≥ k*n; `out` length ≥ m*n.
/// - `use_low_k_tile_avx_fma(m, k, n)` returned true (m%4, n%24, k∈{16,24}).
/// - CPU supports AVX+FMA (feature-gated at caller).
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code)]
unsafe fn low_k_tile_4x24_parallel_fused(
    a: &[f32],
    b: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    epilogue: &GemmEpilogue,
) {
    unsafe {
        debug_assert_eq!(m % 4, 0);
        debug_assert_eq!(n % 24, 0);
        debug_assert!(k == 16 || k == 24);

        #[derive(Clone, Copy)]
        struct RawPtrs {
            a: *const f32,
            b: *const f32,
            bias: Option<*const f32>,
            residual: Option<*const f32>,
        }
        // SAFETY: Used only inside the par_chunks_mut_dispatch scope, which
        // blocks until all workers return. The buffers all outlive the scope.
        #[allow(unsafe_code)]
        unsafe impl Send for RawPtrs {}
        #[allow(unsafe_code)]
        unsafe impl Sync for RawPtrs {}

        let ptrs = RawPtrs {
            a: a.as_ptr(),
            b: b.as_ptr(),
            bias: epilogue.bias,
            residual: epilogue.residual,
        };
        let activation = epilogue.activation;
        let n_panels = n / 24;
        let row_block_out = 4 * n;
        let row_block_in = 4 * k;
        let n_blocks = m / 4;

        let run_one = |blk_idx: usize, out_chunk: &mut [f32]| {
            let p = ptrs;
            let a_tile = p.a.add(blk_idx * row_block_in);
            let res_tile_base = p.residual.map(|r| r.add(blk_idx * row_block_out));
            let local_epilogue = GemmEpilogue {
                bias: p.bias,
                residual: None, // residual applied via explicit tile ptr below
                activation,
            };
            for panel in 0..n_panels {
                let col_offset = panel * 24;
                let b_panel = p.b.add(col_offset);
                let c_tile = out_chunk.as_mut_ptr().add(col_offset);
                let res_tile = res_tile_base.map(|r| r.add(col_offset));
                // SAFETY: dimensions verified by caller shape gate. AVX+FMA
                // runtime-checked at dispatch. c_stride = n.
                match k {
                    16 => low_k_tile_4x24_avx_fma::<16>(
                        a_tile,
                        b_panel,
                        n,
                        c_tile,
                        n,
                        &local_epilogue,
                        col_offset,
                        res_tile,
                    ),
                    24 => low_k_tile_4x24_avx_fma::<24>(
                        a_tile,
                        b_panel,
                        n,
                        c_tile,
                        n,
                        &local_epilogue,
                        col_offset,
                        res_tile,
                    ),
                    _ => unreachable!("shape gate verifies k ∈ {{16, 24}}"),
                }
            }
        };

        // Avoid rayon fork/join overhead when running on a single thread.
        // With `RAYON_NUM_THREADS=1` the `par_chunks_mut` path still
        // constructs per-chunk task state which measurably regresses 1T
        // tracker (was +638 µs before this gate).
        let num_threads = rayon::current_num_threads();
        if num_threads <= 1 || n_blocks < 4 {
            for blk in 0..n_blocks {
                let start = blk * row_block_out;
                let end = start + row_block_out;
                run_one(blk, &mut out[start..end]);
            }
        } else {
            super::super::scope_ctx::par_chunks_mut_dispatch(
                out,
                row_block_out,
                move |blk_idx, out_chunk| {
                    run_one(blk_idx, out_chunk);
                },
            );
        }
    }
}

// ============================================================================
// Step A1.1 — AVX-512 MR=12×NR=32 microkernel (Stage 2, Session 5)
// ============================================================================
//
// Uses the full 32-ZMM register file on Zen 4 and Intel AVX-512 hosts.
// 24 accumulators (12 rows × 2 ZMM cols) + 2 B ZMM + 1 A-broadcast ZMM
// + 5 scratch. Linked via [`x86_64_sysv.S`].
//
// Benefits on Zen 4: double-pumped AVX-512 has the same peak FLOPS as
// AVX2 but halves the front-end overhead (1 ZMM uop instead of 2 YMM
// uops per FMA pair). ORT's FgemmKernelAvx512F uses the same MR=12×NR=32.
//
// Pack-A layout: MR=12 stride. Each k-column has 12 row-broadcast floats
// contiguous.
#[cfg(all(target_arch = "x86_64", any(target_os = "linux", target_os = "macos")))]
const MR12: usize = 12;

/// Step A1.1: pack-A variant for MR=12 AVX-512 microkernel. Mirrors
/// [`pack_a_panel`] (MR=4) structure; partial rows zero-padded so the
/// kernel can read full MR12 broadcasts per k-step.
#[cfg(all(target_arch = "x86_64", any(target_os = "linux", target_os = "macos")))]
#[allow(unsafe_code)]
fn pack_a_panel_mr12(
    a: &[f32],
    lda: usize,
    ic: usize,
    mc: usize,
    pc: usize,
    kc: usize,
    packed: &mut [f32],
) {
    let a_ptr = a.as_ptr();
    let p_ptr = packed.as_mut_ptr();
    let mut idx = 0usize;
    for ir in (0..mc).step_by(MR12) {
        let mr = MR12.min(mc - ir);
        if mr == MR12 {
            for p in 0..kc {
                // SAFETY: lda is the real stride; `(ic+ir+i)*lda + pc+p`
                // is bounded by the source matrix. `packed` sized by
                // caller to hold `div_ceil(mc, MR12) * kc * MR12`.
                unsafe {
                    for i in 0..MR12 {
                        *p_ptr.add(idx + i) = *a_ptr.add((ic + ir + i) * lda + pc + p);
                    }
                }
                idx += MR12;
            }
        } else {
            for p in 0..kc {
                // SAFETY: same bounds reasoning. Tail zero-fill ensures
                // the microkernel broadcast reads valid (zero-padded)
                // data for partial row groups.
                unsafe {
                    for i in 0..mr {
                        *p_ptr.add(idx + i) = *a_ptr.add((ic + ir + i) * lda + pc + p);
                    }
                    std::ptr::write_bytes(p_ptr.add(idx + mr), 0, MR12 - mr);
                }
                idx += MR12;
            }
        }
    }
}

#[cfg(all(target_arch = "x86_64", any(target_os = "linux", target_os = "macos")))]
#[allow(unsafe_code, clippy::too_many_arguments)]
#[inline]
unsafe fn microkernel_12x32_avx512_set(
    a_panel: *const f32,
    b_panel: *const f32,
    c: *mut f32,
    ldc: usize,
    kc: usize,
    bias: Option<*const f32>,
    activation: Activation,
    is_last_k: bool,
) {
    unsafe {
        let bias_ptr = bias.unwrap_or(std::ptr::null());
        let act_id: usize = match activation {
            Activation::Relu => 1,
            _ => 0,
        };
        let last_k: usize = if is_last_k { 1 } else { 0 };
        sgemm_asm_avx512::yscv_sgemm_12x32_avx512_set(
            a_panel, b_panel, c, ldc, kc, bias_ptr, act_id, last_k,
        );
    }
}

#[cfg(all(target_arch = "x86_64", any(target_os = "linux", target_os = "macos")))]
#[allow(unsafe_code, clippy::too_many_arguments)]
#[inline]
unsafe fn microkernel_12x32_avx512_acc(
    a_panel: *const f32,
    b_panel: *const f32,
    c: *mut f32,
    ldc: usize,
    kc: usize,
    bias: Option<*const f32>,
    activation: Activation,
    is_last_k: bool,
) {
    unsafe {
        let bias_ptr = bias.unwrap_or(std::ptr::null());
        let act_id: usize = match activation {
            Activation::Relu => 1,
            _ => 0,
        };
        let last_k: usize = if is_last_k { 1 } else { 0 };
        sgemm_asm_avx512::yscv_sgemm_12x32_avx512_acc(
            a_panel, b_panel, c, ldc, kc, bias_ptr, act_id, last_k,
        );
    }
}

// ---------------------------------------------------------------------------
// Step A1.2 — MR=12×NR=32 AVX-512 blocked GEMM orchestrator
// ---------------------------------------------------------------------------

#[cfg(all(target_arch = "x86_64", any(target_os = "linux", target_os = "macos")))]
const NR32: usize = 32;

/// MC for MR=12 blocked GEMM. 96 = 8 MR12 panels per block. Pack-A per
/// block = 96 × KC = 96 × 256 × 4 = 96 KB (fits L2 comfortably).
#[cfg(all(target_arch = "x86_64", any(target_os = "linux", target_os = "macos")))]
const MC_MR12: usize = 96;

/// NC for MR=12 path. 256 cols = 8 NR32 panels.
#[cfg(all(target_arch = "x86_64", any(target_os = "linux", target_os = "macos")))]
const NC_MR12: usize = 256;

/// Pack B[pc..pc+kc, jc..jc+nc] into `packed` with NR=32 stride. Layout
/// `packed[p * nc + j]` — row-major in-block; the microkernel reads
/// `[rsi + p*128]` ... so NR=32 cols contiguous per k-row.
#[cfg(all(target_arch = "x86_64", any(target_os = "linux", target_os = "macos")))]
#[allow(unsafe_code, clippy::too_many_arguments)]
fn pack_b_panel_nr32(
    b: &[f32],
    ldb: usize,
    pc: usize,
    kc: usize,
    jc: usize,
    nc: usize,
    packed: &mut [f32],
) {
    debug_assert_eq!(nc % NR32, 0);
    let b_ptr = b.as_ptr();
    let p_ptr = packed.as_mut_ptr();
    // Layout: NR32 columns at a time, kc row-interleaved.
    // For each jr (NR32 col panel), pack kc rows of 32 floats each
    // contiguously. Kernel reads `[panel_base + p*128]`.
    let n_panels = nc / NR32;
    for jr_idx in 0..n_panels {
        let jr = jc + jr_idx * NR32;
        let panel_base = jr_idx * kc * NR32;
        for p in 0..kc {
            // SAFETY: b sized for full K×N; packed sized by caller to
            // `n_panels * kc * NR32`.
            unsafe {
                let src_row = b_ptr.add((pc + p) * ldb + jr);
                let dst_row = p_ptr.add(panel_base + p * NR32);
                for c in 0..NR32 {
                    *dst_row.add(c) = *src_row.add(c);
                }
            }
        }
    }
}

/// Runtime gate for AVX-512 MR=12×NR=32 dispatch. Requires:
/// - AVX-512F feature
/// - n % 32 == 0 (no n-tail handling yet — shapes with n%32!=0 reject)
/// - m >= 12 (at least one full MR=12 row-group; if not, fall through)
/// - k >= 16 (smaller doesn't amortize dispatch)
/// - no residual (kernel doesn't support residual yet)
/// - non-SiLU activation (kernel only fuses None/Relu)
/// - Env gate `YSCV_AVX512_SGEMM=1` (default OFF pending tracker A/B)
///
/// Session 8 (A1.2 tail): removed `m % 12 == 0` requirement. Caller
/// splits m into m_full (MR12-aligned) AVX-512 + m_tail fallback.
#[cfg(all(target_arch = "x86_64", any(target_os = "linux", target_os = "macos")))]
fn use_avx512_mr12(
    m: usize,
    k: usize,
    n: usize,
    has_residual: bool,
    activation: Activation,
) -> bool {
    avx512_mr12_enabled()
        && m >= MR12
        && k >= 16
        && n > 0
        && n.is_multiple_of(NR32)
        && !has_residual
        && !matches!(activation, Activation::Silu)
        && std::is_x86_feature_detected!("avx512f")
}

#[cfg(all(target_arch = "x86_64", any(target_os = "linux", target_os = "macos")))]
fn avx512_mr12_enabled() -> bool {
    static CACHED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| std::env::var_os("YSCV_AVX512_SGEMM").is_some())
}

/// Cached `YSCV_LOW_K_TILE` kill-switch. The low-k tile dispatch fires
/// from `matmul_2d_slices_fused_maybe_packed` — called once per Conv,
/// ~100 Conv calls per tracker frame. Prior per-call `env::var_os` read
/// was ~0.1% cycles. Session 10 polish.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn low_k_tile_enabled() -> bool {
    static CACHED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| std::env::var_os("YSCV_LOW_K_TILE").is_some())
}

/// Sequential AVX-512 MR=12×NR=32 blocked GEMM. Mirrors
/// `blocked_gemm_sequential` structure. Assumes shape gate passed.
#[cfg(all(target_arch = "x86_64", any(target_os = "linux", target_os = "macos")))]
#[allow(unsafe_code)]
fn blocked_gemm_sequential_mr12(
    left: &[f32],
    right: &[f32],
    output: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    epilogue: GemmEpilogue,
) {
    debug_assert_eq!(m % MR12, 0);
    debug_assert_eq!(n % NR32, 0);

    let a_size = div_ceil(MC_MR12, MR12) * KC * MR12;
    let mut packed_a = vec![0.0f32; a_size];
    let b_size = (NC_MR12 / NR32) * KC * NR32;
    let mut packed_b = vec![0.0f32; b_size];

    for jc in (0..n).step_by(NC_MR12) {
        let nc = NC_MR12.min(n - jc);
        let nc_aligned = (nc / NR32) * NR32;
        if nc_aligned == 0 {
            continue;
        }
        for pc in (0..k).step_by(KC) {
            let kc = KC.min(k - pc);
            let accumulate = pc > 0;
            let is_last_k = pc + kc >= k;

            pack_b_panel_nr32(right, n, pc, kc, jc, nc_aligned, &mut packed_b);

            for ic in (0..m).step_by(MC_MR12) {
                let mc = MC_MR12.min(m - ic);
                pack_a_panel_mr12(left, k, ic, mc, pc, kc, &mut packed_a);
                // SAFETY: all packed buffers sized for mc × kc and
                // kc × nc_aligned; output row ic, col jc guaranteed
                // in-bounds by outer loop bounds; AVX-512F detected
                // at dispatch gate.
                unsafe {
                    gebp_kernel_mr12_avx512(
                        packed_a.as_ptr(),
                        packed_b.as_ptr(),
                        output.as_mut_ptr(),
                        n,
                        ic,
                        jc,
                        mc,
                        nc_aligned,
                        kc,
                        accumulate,
                        &epilogue,
                        is_last_k,
                    );
                }
            }
        }
    }
}

/// Inner-loop kernel: iterates MR=12 × NR=32 tiles inside the
/// (mc, nc, kc) block. `packed_a` has MR=12 stride; `packed_b` has
/// NR=32 stride (n_panels × kc × 32).
#[cfg(all(target_arch = "x86_64", any(target_os = "linux", target_os = "macos")))]
#[allow(unsafe_code, clippy::too_many_arguments)]
unsafe fn gebp_kernel_mr12_avx512(
    packed_a: *const f32,
    packed_b: *const f32,
    output: *mut f32,
    n: usize,  // total output stride (cols)
    ic: usize, // output row offset for this block
    jc: usize, // output col offset for this block
    mc: usize, // rows in this block (multiple of MR12)
    nc: usize, // cols in this block (multiple of NR32)
    kc: usize,
    accumulate: bool,
    epilogue: &GemmEpilogue,
    is_last_k: bool,
) {
    unsafe {
        let n_panels = nc / NR32;
        for jr_idx in 0..n_panels {
            let b_panel = packed_b.add(jr_idx * kc * NR32);
            let col = jc + jr_idx * NR32;
            let bias_for_tile: Option<*const f32> = match epilogue.bias {
                Some(b) if is_last_k => Some(b.add(col)),
                _ => None,
            };
            for ir in (0..mc).step_by(MR12) {
                let a_panel = packed_a.add((ir / MR12) * kc * MR12);
                let c_ptr = output.add((ic + ir) * n + col);
                if accumulate {
                    microkernel_12x32_avx512_acc(
                        a_panel,
                        b_panel,
                        c_ptr,
                        n,
                        kc,
                        bias_for_tile,
                        epilogue.activation,
                        is_last_k,
                    );
                } else {
                    microkernel_12x32_avx512_set(
                        a_panel,
                        b_panel,
                        c_ptr,
                        n,
                        kc,
                        bias_for_tile,
                        epilogue.activation,
                        is_last_k,
                    );
                }
            }
        }
    }
}

/// Parallel variant: rayon over ic blocks using MC_PARALLEL_MR12.
#[cfg(all(target_arch = "x86_64", any(target_os = "linux", target_os = "macos")))]
const MC_PARALLEL_MR12: usize = 48; // 4 MR12 panels per worker chunk

/// Parallel variant: same outer jc/pc blocking as sequential. B is
/// packed sequentially per (jc, pc); inner ic loop is parallelized.
/// Each worker packs its own A and reads the shared packed B.
#[cfg(all(target_arch = "x86_64", any(target_os = "linux", target_os = "macos")))]
#[allow(unsafe_code)]
fn blocked_gemm_parallel_mr12(
    left: &[f32],
    right: &[f32],
    output: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    epilogue: GemmEpilogue,
    thread_pool: Option<&ThreadPool>,
) {
    debug_assert_eq!(m % MR12, 0);
    debug_assert_eq!(n % NR32, 0);

    let out_ptr = SendPtr(output.as_mut_ptr());
    let b_size = (NC_MR12 / NR32) * KC * NR32;
    let mut packed_b = vec![0.0f32; b_size];

    let ic_blocks: Vec<usize> = (0..m).step_by(MC_PARALLEL_MR12).collect();

    let mut work = || {
        for jc in (0..n).step_by(NC_MR12) {
            let nc = NC_MR12.min(n - jc);
            let nc_aligned = (nc / NR32) * NR32;
            if nc_aligned == 0 {
                continue;
            }
            for pc in (0..k).step_by(KC) {
                let kc = KC.min(k - pc);
                let accumulate = pc > 0;
                let is_last_k = pc + kc >= k;

                pack_b_panel_nr32(right, n, pc, kc, jc, nc_aligned, &mut packed_b);
                let pb_ptr = packed_b.as_ptr() as usize;
                let out_p = &out_ptr;

                par_for_each_ic(&ic_blocks, thread_pool, |ic| {
                    let mc = MC_PARALLEL_MR12.min(m - ic);
                    let a_panels = div_ceil(mc, MR12);
                    let pa_size = a_panels * kc * MR12;
                    let mut packed_a = vec![0.0f32; pa_size];
                    pack_a_panel_mr12(left, k, ic, mc, pc, kc, &mut packed_a);
                    // SAFETY: output rows ic..ic+mc are disjoint per worker.
                    // `packed_b` outlives the `par_for_each_ic` scope.
                    unsafe {
                        gebp_kernel_mr12_avx512(
                            packed_a.as_ptr(),
                            pb_ptr as *const f32,
                            out_p.0,
                            n,
                            ic,
                            jc,
                            mc,
                            nc_aligned,
                            kc,
                            accumulate,
                            &epilogue,
                            is_last_k,
                        );
                    }
                });
            }
        }
    };

    if let Some(pool) = thread_pool {
        pool.install(work);
    } else {
        work();
    }
}

// ============================================================================
// Step C1 — MR=6×16 microkernel + pack_a_mr6 unit tests
// ============================================================================

#[cfg(all(test, any(target_arch = "x86", target_arch = "x86_64")))]
mod mr6_tests {
    use super::*;

    /// Reference matmul: out = A[m, k] × B[k, n], optionally accumulated.
    fn ref_matmul_accumulate(
        a: &[f32],
        b: &[f32],
        out: &mut [f32],
        m: usize,
        k: usize,
        n: usize,
        accumulate: bool,
    ) {
        for i in 0..m {
            for j in 0..n {
                let mut sum = if accumulate { out[i * n + j] } else { 0.0 };
                for p in 0..k {
                    sum += a[i * k + p] * b[p * n + j];
                }
                out[i * n + j] = sum;
            }
        }
    }

    fn fill_ramp(buf: &mut [f32], scale: f32) {
        for (i, v) in buf.iter_mut().enumerate() {
            *v = ((i % 97) as f32) * scale - 1.5;
        }
    }

    /// Verify `pack_a_panel_mr6` produces the expected (mc/MR6) × kc × MR6
    /// layout — first 6 elements per k are row 0..5 of A at col `pc+p`.
    #[test]
    fn pack_a_panel_mr6_layout_matches_spec() {
        const M: usize = 6;
        const K: usize = 4;
        let a: Vec<f32> = (0..M * K).map(|i| i as f32).collect();
        let mut packed = vec![0.0f32; div_ceil(M, MR6) * K * MR6];
        pack_a_panel_mr6(&a, K, 0, M, 0, K, &mut packed);
        // For each k column p, packed[p*MR6 + i] == a[i*K + p].
        for p in 0..K {
            for i in 0..MR6 {
                let expected = a[i * K + p];
                assert_eq!(packed[p * MR6 + i], expected, "k={p} row={i}");
            }
        }
    }

    /// Direct call into microkernel_6x16_avx_fma with a 6×8 tile from a
    /// 6×K × K×16 multiply. Verify bitwise-close match vs reference scalar.
    #[test]
    fn microkernel_6x16_matches_reference_no_epilogue() {
        if !std::is_x86_feature_detected!("fma") || !std::is_x86_feature_detected!("avx") {
            return;
        }
        const M: usize = 6;
        const K: usize = 16;
        const N: usize = 16;
        let mut a = vec![0.0f32; M * K];
        let mut b = vec![0.0f32; K * N];
        fill_ramp(&mut a, 0.13);
        fill_ramp(&mut b, 0.19);

        // Reference.
        let mut ref_out = vec![0.0f32; M * N];
        ref_matmul_accumulate(&a, &b, &mut ref_out, M, K, N, false);

        // Pack A MR=6 stride.
        let mut packed_a = vec![0.0f32; div_ceil(M, MR6) * K * MR6];
        pack_a_panel_mr6(&a, K, 0, M, 0, K, &mut packed_a);

        // Pack B as two NR=8 panels (panel_0 = cols 0..8, panel_1 = cols 8..16).
        let mut packed_b = vec![0.0f32; 2 * K * NR];
        for p in 0..K {
            for j in 0..NR {
                packed_b[p * NR + j] = b[p * N + j];
                packed_b[K * NR + p * NR + j] = b[p * N + NR + j];
            }
        }

        let mut out = vec![0.0f32; M * N];
        let epilogue = GemmEpilogue::new(None, Activation::None);
        // SAFETY: buffers sized to [M×N], packed_a to MR6 layout, packed_b
        // to two NR=8 panels. `microkernel_6x16_avx_fma` is gated on
        // fma+avx feature detection (checked above).
        #[allow(unsafe_code)]
        unsafe {
            microkernel_6x16_avx_fma(
                packed_a.as_ptr(),
                packed_b.as_ptr(),
                packed_b.as_ptr().add(K * NR),
                out.as_mut_ptr(),
                N,
                K,
                false,
                epilogue,
                true,
                0,
                None,
            );
        }

        for i in 0..M * N {
            let d = (out[i] - ref_out[i]).abs();
            assert!(d < 1e-3, "out[{i}]={} ref={} diff={d}", out[i], ref_out[i]);
        }
    }

    /// Same as above but with bias + Relu activation.
    #[test]
    fn microkernel_6x16_matches_reference_bias_relu() {
        if !std::is_x86_feature_detected!("fma") || !std::is_x86_feature_detected!("avx") {
            return;
        }
        const M: usize = 6;
        const K: usize = 8;
        const N: usize = 16;
        let mut a = vec![0.0f32; M * K];
        let mut b = vec![0.0f32; K * N];
        let mut bias = [0.0f32; N];
        fill_ramp(&mut a, 0.11);
        fill_ramp(&mut b, 0.17);
        for (i, v) in bias.iter_mut().enumerate() {
            *v = (i as f32) * 0.5 - 3.0;
        }

        // Reference with bias + Relu.
        let mut ref_out = vec![0.0f32; M * N];
        ref_matmul_accumulate(&a, &b, &mut ref_out, M, K, N, false);
        for i in 0..M {
            for j in 0..N {
                let v = ref_out[i * N + j] + bias[j];
                ref_out[i * N + j] = if v > 0.0 { v } else { 0.0 };
            }
        }

        let mut packed_a = vec![0.0f32; div_ceil(M, MR6) * K * MR6];
        pack_a_panel_mr6(&a, K, 0, M, 0, K, &mut packed_a);
        let mut packed_b = vec![0.0f32; 2 * K * NR];
        for p in 0..K {
            for j in 0..NR {
                packed_b[p * NR + j] = b[p * N + j];
                packed_b[K * NR + p * NR + j] = b[p * N + NR + j];
            }
        }

        let mut out = vec![0.0f32; M * N];
        let epilogue = GemmEpilogue::new(Some(bias.as_ptr()), Activation::Relu);
        // SAFETY: see preceding test; bias ptr stays valid for this call.
        #[allow(unsafe_code)]
        unsafe {
            microkernel_6x16_avx_fma(
                packed_a.as_ptr(),
                packed_b.as_ptr(),
                packed_b.as_ptr().add(K * NR),
                out.as_mut_ptr(),
                N,
                K,
                false,
                epilogue,
                true,
                0,
                None,
            );
        }

        for i in 0..M * N {
            let d = (out[i] - ref_out[i]).abs();
            assert!(
                d < 1e-3,
                "bias+relu out[{i}]={} ref={} diff={d}",
                out[i],
                ref_out[i]
            );
        }
    }
}

// ============================================================================
// Step S.1′ — low-k tile correctness tests
// ============================================================================

#[cfg(all(test, any(target_arch = "x86", target_arch = "x86_64")))]
mod low_k_tile_tests {
    use super::*;

    fn ref_matmul_fused(
        a: &[f32],
        b: &[f32],
        m: usize,
        k: usize,
        n: usize,
        bias: Option<&[f32]>,
        residual: Option<&[f32]>,
        act: Activation,
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut s = 0.0f32;
                for p in 0..k {
                    s += a[i * k + p] * b[p * n + j];
                }
                if let Some(bv) = bias {
                    s += bv[j];
                }
                if let Some(rv) = residual {
                    s += rv[i * n + j];
                }
                s = match act {
                    Activation::Relu => s.max(0.0),
                    Activation::Silu => s / (1.0 + (-s).exp()),
                    Activation::None => s,
                };
                out[i * n + j] = s;
            }
        }
        out
    }

    fn fill_ramp(buf: &mut [f32], scale: f32, base: f32) {
        for (i, v) in buf.iter_mut().enumerate() {
            *v = ((i % 131) as f32) * scale + base;
        }
    }

    fn run_case(
        m: usize,
        k: usize,
        n: usize,
        with_bias: bool,
        with_residual: bool,
        act: Activation,
    ) {
        if !std::is_x86_feature_detected!("fma") || !std::is_x86_feature_detected!("avx") {
            return;
        }
        assert_eq!(m % 4, 0);
        assert_eq!(n % 24, 0);
        assert!(k == 16 || k == 24);

        let mut a = vec![0.0f32; m * k];
        fill_ramp(&mut a, 0.013, -0.3);
        let mut b = vec![0.0f32; k * n];
        fill_ramp(&mut b, 0.017, 0.1);
        let bias = if with_bias {
            let mut bv = vec![0.0f32; n];
            fill_ramp(&mut bv, 0.3, -1.0);
            Some(bv)
        } else {
            None
        };
        let residual = if with_residual {
            let mut rv = vec![0.0f32; m * n];
            fill_ramp(&mut rv, 0.011, -0.2);
            Some(rv)
        } else {
            None
        };

        let mut out = vec![0.0f32; m * n];
        let epilogue = GemmEpilogue {
            bias: bias.as_ref().map(|v| v.as_ptr()),
            residual: residual.as_ref().map(|v| v.as_ptr()),
            activation: act,
        };
        // SAFETY: shape gate satisfied (asserted above), AVX+FMA
        // detected, buffers sized appropriately.
        #[allow(unsafe_code)]
        unsafe {
            low_k_tile_4x24_parallel_fused(&a, &b, &mut out, m, k, n, &epilogue);
        }

        let ref_out = ref_matmul_fused(&a, &b, m, k, n, bias.as_deref(), residual.as_deref(), act);
        let mut max_diff = 0.0f32;
        let mut max_i = 0;
        for i in 0..out.len() {
            let d = (out[i] - ref_out[i]).abs();
            if d > max_diff {
                max_diff = d;
                max_i = i;
            }
        }
        // SiLU uses `silu_avx_fma` with a fast bit-trick exp whose
        // approximation error is ~0.5-1% of the value; loosen tolerance
        // to 3% for SiLU cases, keep 1e-2 for None / Relu.
        let tol = match act {
            Activation::Silu => 3e-2,
            _ => 1e-2,
        };
        assert!(
            max_diff < tol,
            "low_k_tile diverges m={m} k={k} n={n} bias={with_bias} residual={with_residual} \
             act={act:?}: max diff {max_diff} at {max_i}: tile={} ref={}",
            out[max_i],
            ref_out[max_i],
        );
    }

    #[test]
    fn low_k_tile_k16_n96_small() {
        run_case(4, 16, 96, false, false, Activation::None);
    }

    #[test]
    fn low_k_tile_k16_n96_bias_relu() {
        run_case(256, 16, 96, true, false, Activation::Relu);
    }

    #[test]
    fn low_k_tile_k16_n96_residual() {
        run_case(256, 16, 96, true, true, Activation::Relu);
    }

    #[test]
    fn low_k_tile_k16_n96_large() {
        run_case(1024, 16, 96, true, false, Activation::None);
    }

    #[test]
    fn low_k_tile_k24_n144_bias_relu() {
        run_case(256, 24, 144, true, false, Activation::Relu);
    }

    #[test]
    fn low_k_tile_k24_n144_residual() {
        run_case(256, 24, 144, false, true, Activation::None);
    }

    #[test]
    fn low_k_tile_k16_n24_minimal() {
        run_case(64, 16, 24, false, false, Activation::None);
    }

    #[test]
    fn low_k_tile_k16_n48_silu() {
        run_case(64, 16, 48, true, false, Activation::Silu);
    }

    #[test]
    fn shape_gate_accepts_hot_tracker_shapes() {
        assert!(use_low_k_tile_avx_fma(16384, 16, 96));
        assert!(use_low_k_tile_avx_fma(4096, 24, 144));
        assert!(use_low_k_tile_avx_fma(4096, 16, 96));
    }

    #[test]
    fn shape_gate_rejects_small_or_misaligned() {
        assert!(!use_low_k_tile_avx_fma(256, 16, 96)); // work < 1M
        assert!(!use_low_k_tile_avx_fma(4097, 16, 96)); // m%4
        assert!(!use_low_k_tile_avx_fma(4096, 16, 25)); // n%24
        assert!(!use_low_k_tile_avx_fma(4096, 32, 96)); // k not 16/24
        assert!(!use_low_k_tile_avx_fma(4096, 8, 96)); // k not 16/24
    }
}

#[cfg(test)]
mod residual_tail_tests {
    use super::*;

    fn ref_matmul_fused(
        a: &[f32],
        b: &[f32],
        m: usize,
        k: usize,
        n: usize,
        bias: &[f32],
        residual: &[f32],
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut s = 0.0f32;
                for p in 0..k {
                    s += a[i * k + p] * b[p * n + j];
                }
                s += bias[j];
                s += residual[i * n + j];
                out[i * n + j] = s.max(0.0);
            }
        }
        out
    }

    fn fill_ramp(buf: &mut [f32], scale: f32, base: f32) {
        for (i, v) in buf.iter_mut().enumerate() {
            *v = ((i % 113) as f32) * scale + base;
        }
    }

    #[test]
    fn blocked_gemm_residual_handles_m_and_n_tails() {
        // Tail shape: m is not divisible by MR=4, n is not divisible by NR=8.
        // Still passes blocked gate (m>=32, k>=32, n>=16).
        let m = 35usize;
        let k = 40usize;
        let n = 18usize;

        let mut a = vec![0.0f32; m * k];
        let mut b = vec![0.0f32; k * n];
        let mut bias = vec![0.0f32; n];
        let mut residual = vec![0.0f32; m * n];
        fill_ramp(&mut a, 0.013, -0.4);
        fill_ramp(&mut b, 0.019, 0.2);
        fill_ramp(&mut bias, 0.11, -1.7);
        fill_ramp(&mut residual, 0.007, -0.3);

        let mut out = vec![0.0f32; m * n];
        let epilogue = GemmEpilogue {
            bias: Some(bias.as_ptr()),
            residual: Some(residual.as_ptr()),
            activation: Activation::Relu,
        };
        blocked_gemm_sequential(&a, &b, &mut out, m, k, n, epilogue, None);

        let reference = ref_matmul_fused(&a, &b, m, k, n, &bias, &residual);
        let mut max_diff = 0.0f32;
        let mut max_idx = 0usize;
        for i in 0..out.len() {
            let d = (out[i] - reference[i]).abs();
            if d > max_diff {
                max_diff = d;
                max_idx = i;
            }
        }
        assert!(
            max_diff < 1e-2,
            "blocked residual tail mismatch: max diff {max_diff} at {max_idx}, got={} ref={}",
            out[max_idx],
            reference[max_idx],
        );
    }
}

// ============================================================================
// Step A1.1 — AVX-512 MR=12×NR=32 correctness tests
// ============================================================================

#[cfg(all(
    test,
    target_arch = "x86_64",
    any(target_os = "linux", target_os = "macos")
))]
#[allow(unsafe_code)]
mod avx512_12x32_tests {
    use super::*;

    fn fill_ramp(buf: &mut [f32], scale: f32, base: f32) {
        for (i, v) in buf.iter_mut().enumerate() {
            *v = ((i % 131) as f32) * scale + base;
        }
    }

    fn reference_matmul(
        a: &[f32],
        b: &[f32],
        m: usize,
        k: usize,
        n: usize,
        bias: Option<&[f32]>,
        act: Activation,
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut s = 0.0f32;
                for p in 0..k {
                    s += a[i * k + p] * b[p * n + j];
                }
                if let Some(bv) = bias {
                    s += bv[j];
                }
                s = match act {
                    Activation::Relu => s.max(0.0),
                    Activation::None => s,
                    _ => s,
                };
                out[i * n + j] = s;
            }
        }
        out
    }

    /// Verify `pack_a_panel_mr12` produces `packed[p*MR12 + i] == a[i*k + p]`
    /// (row-major A, col-major panel layout with MR12=12 stride).
    #[test]
    fn pack_a_panel_mr12_layout_matches_spec() {
        const M: usize = 12;
        const K: usize = 32;
        let mut a = vec![0.0f32; M * K];
        fill_ramp(&mut a, 0.011, -0.2);
        let mut packed = vec![0.0f32; M * K];
        pack_a_panel_mr12(&a, K, 0, M, 0, K, &mut packed);
        for p in 0..K {
            for i in 0..M {
                let expected = a[i * K + p];
                assert_eq!(packed[p * 12 + i], expected, "p={p} i={i}");
            }
        }
    }

    /// Direct call into microkernel_12x32_avx512_set without epilogue
    /// (bias=None, activation=None), matching reference matmul bitwise.
    fn run_set_case(k: usize, with_bias: bool, act: Activation) {
        if !std::is_x86_feature_detected!("avx512f") {
            return;
        }
        const M: usize = 12;
        const N: usize = 32;
        let mut a = vec![0.0f32; M * k];
        fill_ramp(&mut a, 0.013, -0.3);
        let mut b = vec![0.0f32; k * N];
        fill_ramp(&mut b, 0.017, 0.1);

        let mut packed_a = vec![0.0f32; M * k];
        pack_a_panel_mr12(&a, k, 0, M, 0, k, &mut packed_a);

        let bias_vec = if with_bias {
            let mut bv = vec![0.0f32; N];
            fill_ramp(&mut bv, 0.3, -1.0);
            Some(bv)
        } else {
            None
        };
        let mut c = vec![0.0f32; M * N];

        // SAFETY: all buffers sized correctly above; AVX-512F detected.
        unsafe {
            microkernel_12x32_avx512_set(
                packed_a.as_ptr(),
                b.as_ptr(),
                c.as_mut_ptr(),
                N, // ldc in floats
                k,
                bias_vec.as_ref().map(|v| v.as_ptr()),
                act,
                true,
            );
        }

        let reference = reference_matmul(&a, &b, M, k, N, bias_vec.as_deref(), act);
        let mut max_diff = 0.0f32;
        let mut max_i = 0;
        for i in 0..c.len() {
            let d = (c[i] - reference[i]).abs();
            if d > max_diff {
                max_diff = d;
                max_i = i;
            }
        }
        assert!(
            max_diff < 1e-2,
            "12x32 kernel diverges k={k} bias={with_bias} act={act:?}: \
             max diff {max_diff} at {max_i}: kernel={} ref={}",
            c[max_i],
            reference[max_i],
        );
    }

    #[test]
    fn mr12_k16_no_epilogue() {
        run_set_case(16, false, Activation::None);
    }

    #[test]
    fn mr12_k32_no_epilogue() {
        run_set_case(32, false, Activation::None);
    }

    #[test]
    fn mr12_k64_bias_relu() {
        run_set_case(64, true, Activation::Relu);
    }

    #[test]
    fn mr12_k192_bias_relu() {
        // Tracker k=192 — common pointwise Conv k size.
        run_set_case(192, true, Activation::Relu);
    }

    #[test]
    fn mr12_k672_bias_relu() {
        // Tracker k=672 — hot pointwise Conv_Add/Conv_Relu.
        run_set_case(672, true, Activation::Relu);
    }

    /// Verify accumulate path: starts from existing C, adds kc FMAs.
    /// Integration test: `matmul_2d_slices_fused_maybe_packed` with
    /// AVX-512 path enabled → bitwise-close to the default MR=4×24 path.
    /// Covers several shapes at m%12==0 and n%32==0.
    fn run_integration_case(m: usize, k: usize, n: usize, with_bias: bool, act: Activation) {
        if !std::is_x86_feature_detected!("avx512f") {
            return;
        }
        // Safety of env mutation in tests: this test runs single-threaded in
        // its own process; `YSCV_AVX512_SGEMM` is re-read via OnceLock
        // inside `avx512_mr12_enabled`, so we must ensure the toggle is
        // cached before ANY matmul path reads it. For the test we just
        // flip it once and trust the OnceLock.
        unsafe {
            std::env::set_var("YSCV_AVX512_SGEMM", "1");
        }

        let mut a = vec![0.0f32; m * k];
        fill_ramp(&mut a, 0.009, -0.2);
        let mut b = vec![0.0f32; k * n];
        fill_ramp(&mut b, 0.013, 0.05);
        let bias = if with_bias {
            let mut bv = vec![0.0f32; n];
            fill_ramp(&mut bv, 0.2, -0.5);
            Some(bv)
        } else {
            None
        };

        let epilogue = GemmEpilogue {
            bias: bias.as_ref().map(|v| v.as_ptr()),
            residual: None,
            activation: act,
        };

        let mut out_avx512 = vec![0.0f32; m * n];
        matmul_2d_slices_fused_maybe_packed(
            &a,
            m,
            k,
            &b,
            n,
            &mut out_avx512,
            None,
            epilogue,
            ParallelMatmulConfig::default(),
            None,
        );

        // Reference: scalar ground truth.
        let reference = reference_matmul(&a, &b, m, k, n, bias.as_deref(), act);

        let mut max_diff = 0.0f32;
        let mut max_i = 0;
        for i in 0..out_avx512.len() {
            let d = (out_avx512[i] - reference[i]).abs();
            if d > max_diff {
                max_diff = d;
                max_i = i;
            }
        }
        assert!(
            max_diff < 1e-2,
            "AVX-512 integration diverges m={m} k={k} n={n} bias={with_bias} act={act:?}: \
             max diff {max_diff} at {max_i}: got={} ref={}",
            out_avx512[max_i],
            reference[max_i],
        );
    }

    #[test]
    fn integration_m12_k32_n32() {
        run_integration_case(12, 32, 32, false, Activation::None);
    }

    #[test]
    fn integration_m24_k64_n64_bias_relu() {
        run_integration_case(24, 64, 64, true, Activation::Relu);
    }

    #[test]
    fn integration_m96_k192_n96_bias_relu() {
        run_integration_case(96, 192, 96, true, Activation::Relu);
    }

    #[test]
    fn integration_m1008_k672_n96_bias_relu() {
        // Tracker-like: m = 1008 = 84×12, k=672, n=96.
        run_integration_case(1008, 672, 96, true, Activation::Relu);
    }

    // Session 8 tail tests removed — tail handling reverted due to
    // tracker regression (+1248 µs @ 6T from duplicate pack-B) and
    // FP ordering drift (bitwise-identical check failed). Strict
    // gate (m%12==0) restored.

    #[test]
    fn mr12_acc_path() {
        if !std::is_x86_feature_detected!("avx512f") {
            return;
        }
        const M: usize = 12;
        const N: usize = 32;
        const K: usize = 64;
        let mut a = vec![0.0f32; M * K];
        fill_ramp(&mut a, 0.013, -0.3);
        let mut b = vec![0.0f32; K * N];
        fill_ramp(&mut b, 0.017, 0.1);
        let mut packed_a = vec![0.0f32; M * K];
        pack_a_panel_mr12(&a, K, 0, M, 0, K, &mut packed_a);

        // Pre-fill c with known values.
        let mut c = vec![0.0f32; M * N];
        fill_ramp(&mut c, 0.5, -2.0);
        let c_initial = c.clone();

        unsafe {
            microkernel_12x32_avx512_acc(
                packed_a.as_ptr(),
                b.as_ptr(),
                c.as_mut_ptr(),
                N,
                K,
                None,
                Activation::None,
                false, // intermediate k-block, no epilogue
            );
        }

        // Expected: c_initial + A @ B
        let added = reference_matmul(&a, &b, M, K, N, None, Activation::None);
        for i in 0..M * N {
            let expected = c_initial[i] + added[i];
            let d = (c[i] - expected).abs();
            assert!(
                d < 1e-2,
                "acc path diverges at {i}: got={} expected={} (c_initial={} added={})",
                c[i],
                expected,
                c_initial[i],
                added[i],
            );
        }
    }

    /// Micro-bench MR=12×NR=32 single-tile cycle count vs MR=4×NR=24
    /// tripled. Gated on `YSCV_AVX512_BENCH=1`. Reports per-tile µs at
    /// k ∈ tracker-hot values, single-thread.
    #[test]
    #[ignore = "perf bench; run with YSCV_AVX512_BENCH=1"]
    fn mr12_vs_mr4_single_tile_bench() {
        if std::env::var("YSCV_AVX512_BENCH").is_err() {
            return;
        }
        if !std::is_x86_feature_detected!("avx512f") {
            println!("(AVX-512F not detected, skipping)");
            return;
        }
        const M12: usize = 12;
        const N12: usize = 32;

        println!();
        println!(
            "{:>6} {:>14} {:>14} {:>10}",
            "k", "12x32 µs", "4x24*3 µs", "speedup"
        );
        // For each tracker k, measure single-tile throughput.
        for &k in &[16usize, 24, 32, 64, 192, 384, 672] {
            // MR=12×NR=32 single tile
            let mut a = vec![0.0f32; M12 * k];
            fill_ramp(&mut a, 0.013, -0.3);
            let mut b = vec![0.0f32; k * N12];
            fill_ramp(&mut b, 0.017, 0.1);
            let mut packed_a = vec![0.0f32; M12 * k];
            pack_a_panel_mr12(&a, k, 0, M12, 0, k, &mut packed_a);
            let mut c = vec![0.0f32; M12 * N12];
            const ITERS: usize = 20000;
            // Warm up.
            for _ in 0..200 {
                unsafe {
                    microkernel_12x32_avx512_set(
                        packed_a.as_ptr(),
                        b.as_ptr(),
                        c.as_mut_ptr(),
                        N12,
                        k,
                        None,
                        Activation::None,
                        true,
                    );
                }
                std::hint::black_box(&c);
            }
            let t0 = std::time::Instant::now();
            for _ in 0..ITERS {
                unsafe {
                    microkernel_12x32_avx512_set(
                        packed_a.as_ptr(),
                        b.as_ptr(),
                        c.as_mut_ptr(),
                        N12,
                        k,
                        None,
                        Activation::None,
                        true,
                    );
                }
                std::hint::black_box(&c);
            }
            let us_12x32 = t0.elapsed().as_secs_f64() * 1e6 / ITERS as f64;

            // Compare to MR=4×NR=24 — reference count is "3 tiles" to
            // match output area (3 × 4×24 = 288 outputs = slightly less
            // than 12×32 = 384 but close). Use raw blocked GEMM over
            // a 12×24 region for a closer comparison.
            const M4: usize = 12;
            const N4: usize = 24;
            let mut a4 = vec![0.0f32; M4 * k];
            fill_ramp(&mut a4, 0.013, -0.3);
            let mut b4 = vec![0.0f32; k * N4];
            fill_ramp(&mut b4, 0.017, 0.1);
            let mut out4 = vec![0.0f32; M4 * N4];
            // Warm up.
            for _ in 0..200 {
                matmul_2d_slices_fused_maybe_packed(
                    &a4,
                    M4,
                    k,
                    &b4,
                    N4,
                    &mut out4,
                    None,
                    GemmEpilogue {
                        bias: None,
                        activation: Activation::None,
                        residual: None,
                    },
                    ParallelMatmulConfig::default(),
                    None,
                );
            }
            let t0 = std::time::Instant::now();
            for _ in 0..ITERS {
                matmul_2d_slices_fused_maybe_packed(
                    &a4,
                    M4,
                    k,
                    &b4,
                    N4,
                    &mut out4,
                    None,
                    GemmEpilogue {
                        bias: None,
                        activation: Activation::None,
                        residual: None,
                    },
                    ParallelMatmulConfig::default(),
                    None,
                );
            }
            let us_4x24 = t0.elapsed().as_secs_f64() * 1e6 / ITERS as f64;
            let speedup = us_4x24 / us_12x32;
            println!(
                "{:>6} {:>14.3} {:>14.3} {:>9.2}x",
                k, us_12x32, us_4x24, speedup
            );
        }
        println!();
    }
}

#[cfg(test)]
mod trans_a_tests {
    use super::*;

    fn ref_matmul(a_mk: &[f32], b_kn: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut s = 0.0f32;
                for p in 0..k {
                    s += a_mk[i * k + p] * b_kn[p * n + j];
                }
                out[i * n + j] = s;
            }
        }
        out
    }

    /// Feed `matmul_2d_slices_trans_a` a (K, M)-laid-out A matrix and
    /// check the output matches the reference where A is transposed to
    /// (M, K) first. Covers the tracker shapes the runner dispatches on.
    fn check_shape(m: usize, k: usize, n: usize) {
        let mut a_mk = vec![0.0f32; m * k];
        for (i, v) in a_mk.iter_mut().enumerate() {
            *v = ((i % 97) as f32) * 0.013 - 0.5;
        }
        // Build the (K, M) layout that the runner hands in.
        let mut a_kt = vec![0.0f32; k * m];
        for mi in 0..m {
            for ki in 0..k {
                a_kt[ki * m + mi] = a_mk[mi * k + ki];
            }
        }
        let mut b = vec![0.0f32; k * n];
        for (i, v) in b.iter_mut().enumerate() {
            *v = ((i % 113) as f32) * 0.009 + 0.25;
        }
        let expected = ref_matmul(&a_mk, &b, m, k, n);
        let mut out = vec![0.0f32; m * n];
        matmul_2d_slices_trans_a(&a_kt, m, k, &b, n, &mut out);
        for (idx, (&g, &e)) in out.iter().zip(expected.iter()).enumerate() {
            let tol = 1e-3 * e.abs().max(1.0);
            assert!(
                (g - e).abs() <= tol,
                "shape m={m} k={k} n={n} idx={idx}: got={g} expected={e}"
            );
        }
    }

    #[test]
    fn trans_a_small_square() {
        check_shape(4, 8, 6);
    }

    #[test]
    fn trans_a_tracker_cls_dw_shape() {
        // Tracker `cls_dw/MatMul` shape: A:[1,32,64]→[1,64,32], B:[1,32,64]
        // post-transpose the fused kernel sees m=64, k=32, n=64.
        check_shape(64, 32, 64);
    }

    #[test]
    fn trans_a_tall_thin() {
        check_shape(128, 96, 4);
    }

    #[test]
    fn trans_a_wide_short() {
        check_shape(4, 96, 128);
    }
}
