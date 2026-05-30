//! GEBP micro-kernel loop and the per-arch register-tiled microkernels
//! (AVX-512/AVX2/FMA/SSE/NEON/scalar), plus the low-k specialized tile.

use super::*;

// ---------------------------------------------------------------------------
// GEBP kernel: micro-kernel loop over one MC×NC tile
// ---------------------------------------------------------------------------

/// Process one MC×NC block using packed_a and packed_b.
#[allow(unsafe_code)]
pub(super) fn gebp_kernel(
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
pub(super) fn par_for_each_ic<F>(ic_blocks: &[usize], thread_pool: Option<&ThreadPool>, f: F)
where
    F: Fn(usize) + Send + Sync,
{
    // Preferred path: runner-installed scope (covers both rayon and yscv
    // backends depending on `YSCV_POOL`).
    let routed = super::super::super::scope_ctx::with_scope(|scope| {
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
pub(super) fn mr8_enabled() -> bool {
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
pub(super) fn mr8_residual_enabled() -> bool {
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
pub(super) fn mr8_tail8_asm_enabled() -> bool {
    use std::sync::OnceLock;
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| std::env::var_os("YSCV_NO_MR8_TAIL8_ASM").is_none())
}

/// Gate for the x86 AVX2 MR=6×16 fast path. Opt-in via `YSCV_MR6=1`,
/// default OFF on Zen 4: the pure-intrinsics 6×16 kernel loses to the
/// inline-asm 4×16/4×24 paths, whose double-buffered B loads (preload B[k+1]
/// during k's FMAs) LLVM can't replicate from intrinsics. Kept wired for a
/// future asm 6×16 rewrite and for microarchs with different register
/// pressure (Intel SPR, Zen 5).
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub(super) fn mr6_enabled() -> bool {
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
pub(super) fn avx512_relu_enabled() -> bool {
    use std::sync::OnceLock;
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| std::env::var_os("YSCV_AVX512_RELU").is_some())
}

/// Gate for the 4×24 AVX2 pure-.S kernel. Opt-in via `YSCV_ASM_GEMM=1`,
/// default OFF on Zen 4: it ties the intrinsics `microkernel_4x24_avx_fma`
/// (whose inline-`asm!` k-loop + LLVM reg-alloc already saturate the 2-FMA
/// port pipeline without spills). Kept for A/B on microarchs where the pure
/// asm path may win (Intel Sapphire Rapids, Zen 5).
#[cfg(all(
    target_arch = "x86_64",
    any(
        target_os = "linux",
        target_os = "macos",
        all(target_os = "windows", not(target_env = "msvc"))
    )
))]
pub(super) fn asm_4x24_enabled() -> bool {
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
pub(super) fn avx512_enabled() -> bool {
    use std::sync::OnceLock;
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| {
        if std::env::var_os("YSCV_NO_AVX512").is_some() {
            return false;
        }
        std::is_x86_feature_detected!("avx512f")
    })
}
