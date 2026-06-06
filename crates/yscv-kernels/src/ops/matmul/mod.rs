#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::{
    float32x4_t, vaddq_f32, vdivq_f32, vdupq_n_f32, vfmaq_f32, vfmaq_laneq_f32, vld1q_f32,
    vmaxq_f32, vnegq_f32, vst1q_f32,
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

mod asm;
#[cfg(any(
    target_arch = "aarch64",
    all(
        target_arch = "x86_64",
        any(
            target_os = "linux",
            target_os = "macos",
            all(target_os = "windows", not(target_env = "msvc"))
        )
    )
))]
use asm::*;

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

/// Shape-independent snapshot of matmul dispatch state for benchmark logs.
///
/// The final GEMM kernel remains shape-dependent, but these fields capture the
/// cached host ISA, BLAS availability, and opt-in/kill-switch gates that decide
/// the candidate set before shape checks run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MatmulDispatchReport {
    pub primary_isa: &'static str,
    pub blas: &'static str,
    pub avx512: bool,
    pub avx512_mr12: bool,
    pub avx512_relu: bool,
    pub asm_4x24: bool,
    pub mr6: bool,
    pub mr8: bool,
    pub low_k_tile: bool,
}

impl std::fmt::Display for MatmulDispatchReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "isa={}; blas={}; avx512={}; avx512_mr12={}; avx512_relu={}; asm_4x24={}; mr6={}; mr8={}; low_k_tile={}",
            self.primary_isa,
            self.blas,
            self.avx512,
            self.avx512_mr12,
            self.avx512_relu,
            self.asm_4x24,
            self.mr6,
            self.mr8,
            self.low_k_tile
        )
    }
}

pub fn matmul_dispatch_report() -> MatmulDispatchReport {
    let features = crate::host_cpu().features;
    MatmulDispatchReport {
        primary_isa: matmul_primary_isa(features),
        blas: matmul_blas_status(),
        avx512: matmul_avx512_enabled(),
        avx512_mr12: matmul_avx512_mr12_enabled(),
        avx512_relu: matmul_avx512_relu_enabled(),
        asm_4x24: matmul_asm_4x24_enabled(),
        mr6: matmul_mr6_enabled(),
        mr8: matmul_mr8_enabled(),
        low_k_tile: matmul_low_k_tile_enabled(),
    }
}

fn matmul_primary_isa(features: crate::CpuFeatures) -> &'static str {
    if features.avx512f && matmul_avx512_enabled() {
        "avx512"
    } else if features.x86_avx_fma() {
        "avx/fma"
    } else if features.avx {
        "avx"
    } else if features.sse2 {
        "sse2"
    } else if features.sse {
        "sse"
    } else if features.neon {
        "neon"
    } else {
        "scalar"
    }
}

#[cfg(feature = "blas")]
fn matmul_blas_status() -> &'static str {
    if use_blas() { "enabled" } else { "disabled" }
}

#[cfg(not(feature = "blas"))]
fn matmul_blas_status() -> &'static str {
    "not-compiled"
}

#[cfg(all(target_arch = "x86_64", any(target_os = "linux", target_os = "macos")))]
fn matmul_avx512_enabled() -> bool {
    kernels::avx512_enabled()
}

#[cfg(not(all(target_arch = "x86_64", any(target_os = "linux", target_os = "macos"))))]
fn matmul_avx512_enabled() -> bool {
    false
}

#[cfg(all(target_arch = "x86_64", any(target_os = "linux", target_os = "macos")))]
fn matmul_avx512_mr12_enabled() -> bool {
    avx512_mr12_enabled()
}

#[cfg(not(all(target_arch = "x86_64", any(target_os = "linux", target_os = "macos"))))]
fn matmul_avx512_mr12_enabled() -> bool {
    false
}

#[cfg(all(target_arch = "x86_64", any(target_os = "linux", target_os = "macos")))]
fn matmul_avx512_relu_enabled() -> bool {
    kernels::avx512_relu_enabled()
}

#[cfg(not(all(target_arch = "x86_64", any(target_os = "linux", target_os = "macos"))))]
fn matmul_avx512_relu_enabled() -> bool {
    false
}

#[cfg(all(
    target_arch = "x86_64",
    any(
        target_os = "linux",
        target_os = "macos",
        all(target_os = "windows", not(target_env = "msvc"))
    )
))]
fn matmul_asm_4x24_enabled() -> bool {
    kernels::asm_4x24_enabled()
}

#[cfg(not(all(
    target_arch = "x86_64",
    any(
        target_os = "linux",
        target_os = "macos",
        all(target_os = "windows", not(target_env = "msvc"))
    )
)))]
fn matmul_asm_4x24_enabled() -> bool {
    false
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn matmul_mr6_enabled() -> bool {
    kernels::mr6_enabled()
}

#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
fn matmul_mr6_enabled() -> bool {
    false
}

#[cfg(target_arch = "aarch64")]
fn matmul_mr8_enabled() -> bool {
    kernels::mr8_enabled()
}

#[cfg(not(target_arch = "aarch64"))]
fn matmul_mr8_enabled() -> bool {
    false
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn matmul_low_k_tile_enabled() -> bool {
    low_k::low_k_tile_enabled()
}

#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
fn matmul_low_k_tile_enabled() -> bool {
    false
}

// ---------------------------------------------------------------------------
// Blocked matmul constants
// ---------------------------------------------------------------------------

/// Micro-kernel tile: 4 rows of A × 8 columns of B (default).
// WHY 4: saturates NEON/SSE register file without spilling (4 accumulator regs × NR columns).
pub(super) const MR: usize = 4;
// WHY 8: 8 columns = 2 AVX registers or 2 NEON registers per row, fits L1 cache line (64 bytes).
pub(super) const NR: usize = 8;
/// MR=6 AVX2 microkernel tile — 6 rows × NR_MR6=16 columns, matching MLAS
/// SgemmKernelFma3's 6×16 layout: 12 YMM accumulators (6 rows × 2 NR panels)
/// + 2 B-panels + 2 A-broadcast scratch = exactly 16 YMM.
///
/// Opt-in via `YSCV_MR6=1`, default OFF: on Zen 4 the plain-intrinsics 6×16
/// loses to the inline-asm 2×-unrolled `microkernel_4x24_avx_fma` (which hides
/// B-load latency). Kept wired for a potential future asm rewrite.
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
pub(super) const MR6: usize = 6;
/// Cache blocking parameters.
///
/// KC controls the inner k-loop length per blocked pass. KC=256 (vs MLAS's
/// 128) suits this workload: pointwise Conv with large IC (up to 672) captures
/// K in fewer outer passes, reducing pack_a overhead proportionally. A single
/// constant for all archs — ARM L1D is as large or larger (Cortex-A76 64 KB,
/// Apple M 192 KB), so 256 is comfortable everywhere.
// WHY 256: 256 floats × 4 bytes = 1 KB panel — fits in L1D on all targets.
#[cfg(not(target_arch = "aarch64"))]
pub(super) const KC: usize = 256;
// WHY 128: 128 rows × KC columns = 128KB packed panel fits in L2 cache (typically 256KB-1MB).
#[cfg(not(target_arch = "aarch64"))]
pub(super) const MC: usize = 128;
// WHY 256: balances work per thread with cache reuse across micro-kernel calls.
#[cfg(not(target_arch = "aarch64"))]
pub(super) const NC: usize = 256;

// aarch64 SBCs (Cortex-A53/A55) are a single cluster sharing a small L2 (e.g.
// 512 KB across 4 cores = 128 KB/core under load). The x86 256×256 packed-B
// panel (256 KB) thrashes that at multi-thread, so use half-size blocks so the
// KC×NC panel (64 KB) fits per-core L2.
#[cfg(target_arch = "aarch64")]
pub(super) const KC: usize = 128;
#[cfg(target_arch = "aarch64")]
pub(super) const MC: usize = 128;
#[cfg(target_arch = "aarch64")]
pub(super) const NC: usize = 128;

/// Round `a` up to the next multiple of `b`.
pub(super) fn div_ceil(a: usize, b: usize) -> usize {
    a.div_ceil(b)
}

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

/// Hand-asm 5×5 depthwise microkernel: 4 output columns × 8 channels, interior
/// (no bounds), stride 1, column-reuse. See `src/asm/aarch64.S`.
///
/// # Safety
/// `rows` must point at 5 valid `*const f32`, each with at least 8 readable
/// columns from the tile's first input column; `weight`/`out`/`bias` valid for
/// the channel block; the 4 output columns must be interior (all taps in bounds).
/// `wstride` is the weight tap stride in BYTES (`c_exp*4` for the natural
/// KH·KW·C layout, `32` for a pre-packed `[25][8]` tile).
#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code)]
#[inline]
pub(crate) unsafe fn dw5_creuse_neon(
    rows: *const *const f32,
    weight: *const f32,
    out: *mut f32,
    c_exp: usize,
    bias: *const f32,
    relu: usize,
    wstride: usize,
) {
    unsafe {
        asm::sgemm_asm_aarch64::yscv_dw5_creuse_neon(rows, weight, out, c_exp, bias, relu, wstride)
    }
}

/// Hand-asm 3×3 stride-2 depthwise microkernel: 4 output columns × 8 channels,
/// interior, column-reuse. See `src/asm/aarch64.S`.
///
/// # Safety
/// `rows` must point at 3 valid `*const f32`, each readable for the 9 input
/// columns a stride-2 tile spans from its first; `weight`/`out`/`bias` valid
/// for the channel block; the 4 output columns must be interior. `wstride` is
/// the weight tap stride in BYTES (`c_exp*4` natural KH·KW·C layout).
#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code)]
#[inline]
pub(crate) unsafe fn dw3s2_creuse_neon(
    rows: *const *const f32,
    weight: *const f32,
    out: *mut f32,
    c_exp: usize,
    bias: *const f32,
    relu: usize,
    wstride: usize,
) {
    unsafe {
        asm::sgemm_asm_aarch64::yscv_dw3s2_creuse_neon(
            rows, weight, out, c_exp, bias, relu, wstride,
        )
    }
}

/// Sequential blocked GEMM with a bias/activation epilogue, routed straight to
/// `blocked_gemm_sequential` (8×12 NEON asm microkernel + cached B-pack). The
/// streaming PW-expand uses this — the broadcast kernel runs ~2.4× slower on
/// the same `[in_w, c_in] × [c_in, c_exp]` shape. B-pack is cached by pointer
/// via `get_or_pack_b`, so the per-row calls share one packed weight.
#[cfg(target_arch = "aarch64")]
pub fn matmul_2d_slices_blocked_fused(
    a: &[f32],
    m: usize,
    k: usize,
    b: &[f32],
    n: usize,
    out: &mut [f32],
    epilogue: GemmEpilogue,
) {
    // The pipelined 8×8 microkernel (full A+B double-buffer) runs ~10% over the
    // 8×12/4×16 kernels on these PW shapes by hiding the load-to-use latency.
    // Gated to no-residual, n%8==0 (the M tail falls back inside the driver).
    if !blocked_8x8_disabled() && n.is_multiple_of(8) && epilogue.residual.is_none() && m >= 1 {
        blocked::blocked_gemm_8x8(a, b, out, m, k, n, epilogue);
        return;
    }
    blocked_gemm_sequential(a, b, out, m, k, n, epilogue, None);
}

#[cfg(target_arch = "aarch64")]
#[inline]
fn blocked_8x8_disabled() -> bool {
    static C: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *C.get_or_init(|| std::env::var_os("YSCV_GEMM_8X8_OFF").is_some())
}

/// Parallel variant of [`matmul_2d_slices`]: parallelises over output
/// rows when a [`ParallelScope`] is installed, falls back to the
/// sequential blocked path otherwise. Used by the Stream A1 FTMM
/// non-trans fast path so the 6T scaling matches the trans-A variant.
#[allow(unsafe_code)]
pub fn matmul_2d_slices_parallel(
    a: &[f32],
    m: usize,
    k: usize,
    b: &[f32],
    n: usize,
    out: &mut [f32],
) {
    debug_assert!(a.len() >= m * k);
    debug_assert!(b.len() >= k * n);
    debug_assert!(out.len() >= m * n);

    #[cfg(feature = "blas")]
    if use_blas() {
        blas_sgemm(a, b, out, m, k, n);
        return;
    }

    // AVX-512 4-row × NR=16 outer-product tile (mirror of `trans_a_4row_avx512`
    // but for A in [M, K] row-major). Keeps 16 accumulator ZMMs in flight to
    // saturate the FMA pipe; without it this path falls through to the
    // single-row AVX2 `matmul_row_set_avx_fma`. Kill switch:
    // `YSCV_NON_TRANS_4ROW_OFF=1`.
    let m4 = m & !3usize;
    if !non_trans_4row_disabled() && m4 > 0 && trans_a_4row_supported(n) {
        let (head, tail) = out.split_at_mut(m4 * n);
        head.par_chunks_mut(4 * n)
            .enumerate()
            .for_each(|(group_idx, chunk_4rows)| {
                let mi_base = group_idx * 4;
                non_trans_a_4row_dispatch(a, mi_base, k, b, n, chunk_4rows);
            });
        if m4 < m {
            tail.par_chunks_mut(n)
                .enumerate()
                .for_each(|(off, out_row)| {
                    let mi = m4 + off;
                    #[allow(unsafe_code)]
                    unsafe {
                        matmul_row_set_dispatch(
                            a.as_ptr().add(mi * k),
                            b.as_ptr(),
                            out_row.as_mut_ptr(),
                            k,
                            n,
                        );
                    }
                });
        }
        return;
    }

    let _ = m;
    // Row-parallel matmul over the M dimension. Each chunk = one output
    // row of n floats; the inner `matmul_row_set_dispatch` runs the
    // best-available SIMD variant.
    super::super::scope_ctx::par_chunks_mut_dispatch(out, n, |row, out_row| unsafe {
        matmul_row_set_dispatch(
            a.as_ptr().add(row * k),
            b.as_ptr(),
            out_row.as_mut_ptr(),
            k,
            n,
        );
    });
}

mod trans_a;
pub use trans_a::matmul_2d_slices_trans_a;
use trans_a::*;

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

    // Only route to BLAS for shapes where our custom blocked GEMM defers to
    // it (large square-ish GEMMs where packing wins). When `prefer_custom_gemm`
    // returns true (most Conv-like tracker shapes: k≤704, n≤704), fall through
    // to our parallel blocked-GEMM path with fused epilogue. The old guard
    // `use_blas()` without this check caused all tracker matmul to exit here
    // even though `blas_sgemm` internally calls `run_custom_gemm` (single-
    // threaded, no tile packing, no rayon), bypassing blocked_gemm_parallel.
    #[cfg(feature = "blas")]
    if use_blas() && !prefer_custom_gemm(m, k, n) {
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
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    let use_blocked_low_k = use_blocked_x86_low_k(m, k, n, packed_b.is_some(), epilogue);
    #[cfg(not(target_arch = "aarch64"))]
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
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

    // MR=6×16 AVX2 fast path. On x86 with FMA+AVX, gated on
    // m ≥ MR6, n ≥ 2*NR (full-tile fit), AND n multiple of 2*NR so the
    // jr loop stays in full-tile mode. Scalar tail handlers now support
    // residual epilogue too, so no extra residual-based m-tail gate is
    // needed. SiLU falls through to MR=4 (activation fully supported in
    // 6×16 kernel, but MR=4 inline-asm 2×-unrolled k-loop wins on Zen 4).
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

    // P1.5: AVX-512 MR=12×NR=32 dispatch. Default ON for AVX-512F hosts
    // with n%32==0. All epilogues: None/Relu (asm), SiLU (ZMM post-store),
    // residual (per-tile ptr). Kill-switch: YSCV_AVX512_SGEMM=0.
    // `packed_b.data_nr32` holds session-prepacked NR=32 B when available,
    // eliminating per-inference B packing (the prior regression cause).
    #[cfg(all(target_arch = "x86_64", any(target_os = "linux", target_os = "macos")))]
    if use_avx512_mr12(m, k, n) {
        if should_parallelize(plan, config, thread_pool) {
            blocked_gemm_parallel_mr12(a, b, out, m, k, n, epilogue, thread_pool, packed_b);
        } else {
            blocked_gemm_sequential_mr12(a, b, out, m, k, n, epilogue, packed_b);
        }
        return;
    }

    // Specialized low-k tile path (k ∈ {16, 24}, n % 24 == 0, m % 4 == 0,
    // work ≥ 1M FMAs). These shapes miss `use_blocked` (which requires k ≥ 32)
    // and otherwise fall into `row_gemm_set_parallel_fused`'s 1-row-at-a-time
    // `matmul_row_set_avx_fma`. The tile path processes 4 rows × 24 cols at
    // once with a fully-unrolled const-K k-loop.
    //
    // Opt-in via `YSCV_LOW_K_TILE=1`, default OFF: on Zen 4 the mature per-row
    // path is already well-tuned for k<32 (plain-intrinsics tile regresses).
    // Kept wired so a future inline-asm rewrite can flip the default.
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if use_low_k_tile_avx_fma(m, k, n)
        && low_k_tile_enabled()
        && crate::host_cpu().features.avx
        && crate::host_cpu().features.fma
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
pub(super) fn row_gemm_set_parallel_fused(
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
        // Fused residual + bias + activation in a single SIMD pass over
        // `out_row` (vs 2-3 separate passes), cutting post-matmul memory
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
    // Order is bias → residual → activation, matching the blocked-GEMM
    // microkernel epilogue: out[i] = act(gemm[i] + bias[i%n] + residual[i]).
    if let Some(bias_ptr) = epilogue.bias {
        // SAFETY: bias points to n valid f32s for the lifetime of this call.
        let bias = unsafe { std::slice::from_raw_parts(bias_ptr, n) };
        super::simd::bias_add_nhwc_dispatch(out, bias, m, n);
    }

    if let Some(res_ptr) = epilogue.residual {
        // SAFETY: residual points to m*n valid f32s for the lifetime of this call.
        let res = unsafe { std::slice::from_raw_parts(res_ptr, m * n) };
        for (o, r) in out.iter_mut().zip(res.iter()) {
            *o += r;
        }
    }

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

/// Rank-2 matmul with an explicit parallel config and thread pool.
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

/// x86 opt-in low-k blocked route for Conv-like pointwise GEMMs.
///
/// The generic blocked gate requires `k >= 32`, so tracker pointwise
/// shapes with `k in {16, 24}` fall into row-GEMM even when B is already
/// prepacked at model load.  This gate lets those shapes reuse the existing
/// hand-written 4x24/4x16 AVX+FMA kernels.
///
/// Kill switch: `YSCV_NO_X86_LOW_K_BLOCKED=1`.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline]
fn use_blocked_x86_low_k(
    m: usize,
    k: usize,
    n: usize,
    has_prepacked_b: bool,
    epilogue: GemmEpilogue,
) -> bool {
    use std::sync::OnceLock;
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("YSCV_NO_X86_LOW_K_BLOCKED").is_none())
        && !cfg!(miri)
        && has_prepacked_b
        && m >= BLOCKED_THRESHOLD
        && (k == 16 || k == 24)
        && n >= 2 * NR
        && !matches!(epilogue.activation, Activation::Silu)
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
    !cfg!(miri)
        && cfg!(feature = "blas")
        && std::env::var_os("YSCV_FORCE_NO_BLAS").is_none()
        && !openblas_uses_64bit_int()
}

#[cfg(all(feature = "blas", any(target_os = "linux", target_os = "windows")))]
#[allow(unsafe_code)]
fn openblas_uses_64bit_int() -> bool {
    use std::{ffi::CStr, os::raw::c_char, sync::OnceLock};

    unsafe extern "C" {
        fn openblas_get_config() -> *const c_char;
    }

    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| {
        // SAFETY: OpenBLAS returns a process-static NUL-terminated string.
        let config_ptr = unsafe { openblas_get_config() };
        if config_ptr.is_null() {
            return false;
        }
        // SAFETY: `config_ptr` is non-null and owned by OpenBLAS for the
        // process lifetime.
        let config = unsafe { CStr::from_ptr(config_ptr) }.to_string_lossy();
        config.contains("USE64BITINT") || config.contains("INTERFACE64")
    })
}

#[cfg(all(feature = "blas", not(any(target_os = "linux", target_os = "windows"))))]
fn openblas_uses_64bit_int() -> bool {
    false
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

mod blocked;
pub(crate) use blocked::blocked_gemm_nchwc_a_parallel;
use blocked::*;

pub(super) struct SendPtr(pub(super) *mut f32);
// SAFETY: We ensure disjoint access per thread at the IC block level.
#[allow(unsafe_code)]
unsafe impl Send for SendPtr {}
#[allow(unsafe_code)]
unsafe impl Sync for SendPtr {}

mod kernels;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod low_k;
mod microkernels;
#[cfg(target_arch = "aarch64")]
mod neon;
use kernels::*;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use low_k::*;
use microkernels::*;

#[cfg(all(target_arch = "x86_64", any(target_os = "linux", target_os = "macos")))]
mod avx512_mr12;
#[cfg(all(target_arch = "x86_64", any(target_os = "linux", target_os = "macos")))]
use avx512_mr12::{
    NC_MR12, NR32, avx512_mr12_enabled, blocked_gemm_parallel_mr12, blocked_gemm_sequential_mr12,
    pack_b_panel_nr32, use_avx512_mr12,
};

#[cfg(test)]
mod tests;

mod pack;
#[cfg(target_arch = "aarch64")]
pub use pack::hgemm_6x16_neon;
pub use pack::{PackedB, pack_b_for_session};
