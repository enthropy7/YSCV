#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::{float32x4_t, vaddq_f32, vdupq_n_f32, vfmaq_f32, vld1q_f32, vst1q_f32};
#[cfg(target_arch = "x86")]
use std::arch::x86::{
    __m128, __m256, _mm_add_ps, _mm_loadu_ps, _mm_mul_ps, _mm_set1_ps, _mm_setzero_ps,
    _mm_storeu_ps, _mm256_add_ps, _mm256_fmadd_ps, _mm256_loadu_ps, _mm256_mul_ps, _mm256_set1_ps,
    _mm256_setzero_ps, _mm256_storeu_ps,
};
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{
    __m128, __m256, _mm_add_ps, _mm_loadu_ps, _mm_mul_ps, _mm_set1_ps, _mm_setzero_ps,
    _mm_storeu_ps, _mm256_add_ps, _mm256_fmadd_ps, _mm256_loadu_ps, _mm256_mul_ps, _mm256_set1_ps,
    _mm256_setzero_ps, _mm256_storeu_ps,
};

use rayon::{ThreadPool, prelude::*};
use yscv_tensor::{AlignedVec, Tensor, TensorError};

use super::super::error::KernelError;
use super::config::{MatMulPlan, ParallelMatmulConfig, should_parallelize_len};
use super::simd::matmul_row_dispatch;

// ---------------------------------------------------------------------------
// Blocked matmul constants
// ---------------------------------------------------------------------------

/// Micro-kernel tile: 4 rows of A × 8 columns of B.
// WHY 4: saturates NEON/SSE register file without spilling (4 accumulator regs × NR columns).
const MR: usize = 4;
// WHY 8: 8 columns = 2 AVX registers or 2 NEON registers per row, fits L1 cache line (64 bytes).
const NR: usize = 8;

/// Cache blocking parameters (tuned for 64KB L1, 4MB L2).
// WHY 256: 256 floats × 4 bytes = 1KB panel fits in L1 data cache (typically 32-64KB).
const KC: usize = 256;
// WHY 128: 128 rows × KC columns = 128KB packed panel fits in L2 cache (typically 256KB-1MB).
const MC: usize = 128;
// WHY 256: balances work per thread with cache reuse across micro-kernel calls.
const NC: usize = 256;

/// Use blocked matmul when all dimensions >= this threshold.
/// Lowered from 64 to 32 so that medium matrices (e.g. 64x64, 128x128)
/// benefit from cache-blocked GEMM with SIMD micro-kernels on x86.
// WHY 32: below this, naive row-by-row SIMD is faster because blocking setup overhead dominates.
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

    // Non-BLAS fallback: zero-init + blocked or row GEMM
    out[..m * n].fill(0.0);
    if use_blocked(m, k, n) {
        blocked_gemm_sequential(a, b, out, m, k, n);
    } else {
        row_gemm_sequential(a, b, out, m, k, n);
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
    !cfg!(miri) && m >= BLOCKED_THRESHOLD && k >= BLOCKED_THRESHOLD && n >= BLOCKED_THRESHOLD
}

// ---------------------------------------------------------------------------
// BLAS dispatch (Accelerate on macOS, skip on other platforms)
// ---------------------------------------------------------------------------

/// Use BLAS when available and not under miri.
fn use_blas() -> bool {
    !cfg!(miri) && cfg!(feature = "blas")
}

#[cfg(feature = "blas")]
#[allow(unsafe_code)]
unsafe extern "C" {
    /// cblas_sgemm from Accelerate.framework / OpenBLAS / MKL.
    /// C = alpha * A * B + beta * C
    fn cblas_sgemm(
        order: i32,  // CblasRowMajor = 101
        transa: i32, // CblasNoTrans = 111
        transb: i32, // CblasNoTrans = 111
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        a: *const f32,
        lda: i32,
        b: *const f32,
        ldb: i32,
        beta: f32,
        c: *mut f32,
        ldc: i32,
    );
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
    const CBLAS_ROW_MAJOR: i32 = 101;
    const CBLAS_NO_TRANS: i32 = 111;
    unsafe {
        cblas_sgemm(
            CBLAS_ROW_MAJOR,
            CBLAS_NO_TRANS,
            CBLAS_NO_TRANS,
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

    // Non-BLAS paths accumulate (+=) into output, so zero-init is required.
    output.as_mut_slice().fill(0.0);

    if use_blocked(plan.m, plan.k, plan.n) {
        blocked_gemm_sequential(left, right, &mut output, plan.m, plan.k, plan.n);
    } else {
        row_gemm_sequential(left, right, &mut output, plan.m, plan.k, plan.n);
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
    // SAFETY: Every element is written by BLAS / blocked GEMM / row GEMM
    // before the tensor is returned, so uninit memory is never exposed.
    let mut output = AlignedVec::<f32>::uninitialized(plan.output_len);
    let left = lhs.data();
    let right = rhs.data();

    // BLAS already uses multi-threaded internally
    #[cfg(feature = "blas")]
    if use_blas() {
        blas_sgemm(left, right, &mut output, plan.m, plan.k, plan.n);
        return Tensor::from_aligned(vec![plan.m, plan.n], output).map_err(Into::into);
    }

    // Non-BLAS paths accumulate (+=) into output, so zero-init is required.
    output.as_mut_slice().fill(0.0);

    if use_blocked(plan.m, plan.k, plan.n) {
        blocked_gemm_parallel(
            left,
            right,
            &mut output,
            plan.m,
            plan.k,
            plan.n,
            thread_pool,
        );
    } else {
        row_gemm_parallel(
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

#[allow(unsafe_code)]
fn row_gemm_sequential(
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
        // SAFETY: row < m, so row*k+k <= m*k <= left.len() and row*n+n <= m*n <= output.len().
        unsafe {
            matmul_row_dispatch(left_ptr.add(row * k), right_ptr, out_ptr.add(row * n), k, n);
        }
    }
}

#[allow(unsafe_code)]
fn row_gemm_parallel(
    left: &[f32],
    right: &[f32],
    output: &mut [f32],
    _m: usize,
    k: usize,
    n: usize,
    thread_pool: Option<&ThreadPool>,
) {
    let mut work = || {
        output
            .par_chunks_mut(n)
            .enumerate()
            .for_each(|(row, out_row)| {
                // SAFETY: left[row*k..row*k+k] and right[0..k*n] are valid.
                unsafe {
                    matmul_row_dispatch(
                        left.as_ptr().add(row * k),
                        right.as_ptr(),
                        out_row.as_mut_ptr(),
                        k,
                        n,
                    );
                }
            });
    };

    if let Some(pool) = thread_pool {
        pool.install(work);
    } else {
        work();
    }
}

// ---------------------------------------------------------------------------
// Blocked tiled GEMM
// ---------------------------------------------------------------------------

/// Pack A[ic..ic+mc, pc..pc+kc] into panel format: (mc/MR) panels × kc × MR.
///
/// Each panel stores MR rows for all kc columns contiguously:
/// `packed[(ir/MR)*kc*MR + p*MR + i] = A[(ic+ir+i), (pc+p)]`
fn pack_a_panel(
    a: &[f32],
    lda: usize,
    ic: usize,
    mc: usize,
    pc: usize,
    kc: usize,
    packed: &mut [f32],
) {
    let mut idx = 0;
    for ir in (0..mc).step_by(MR) {
        let mr = MR.min(mc - ir);
        for p in 0..kc {
            for i in 0..mr {
                packed[idx] = a[(ic + ir + i) * lda + pc + p];
                idx += 1;
            }
            // Zero-pad partial tiles.
            for _ in mr..MR {
                packed[idx] = 0.0;
                idx += 1;
            }
        }
    }
}

/// Pack B[pc..pc+kc, jc..jc+nc] into panel format: (nc/NR) panels × kc × NR.
///
/// Each panel stores NR columns for all kc rows contiguously:
/// `packed[(jr/NR)*kc*NR + p*NR + j] = B[(pc+p), (jc+jr+j)]`
fn pack_b_panel(
    b: &[f32],
    ldb: usize,
    pc: usize,
    kc: usize,
    jc: usize,
    nc: usize,
    packed: &mut [f32],
) {
    let mut idx = 0;
    for jr in (0..nc).step_by(NR) {
        let nr = NR.min(nc - jr);
        for p in 0..kc {
            for j in 0..nr {
                packed[idx] = b[(pc + p) * ldb + jc + jr + j];
                idx += 1;
            }
            for _ in nr..NR {
                packed[idx] = 0.0;
                idx += 1;
            }
        }
    }
}

fn div_ceil(a: usize, b: usize) -> usize {
    a.div_ceil(b)
}

/// Sequential blocked GEMM: 3-level cache blocking with MR×NR micro-kernel.
#[allow(unsafe_code)]
fn blocked_gemm_sequential(
    left: &[f32],
    right: &[f32],
    output: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) {
    let mut packed_b = vec![0.0f32; div_ceil(NC, NR) * KC * NR];
    let mut packed_a = vec![0.0f32; div_ceil(MC, MR) * KC * MR];

    for jc in (0..n).step_by(NC) {
        let nc = NC.min(n - jc);
        for pc in (0..k).step_by(KC) {
            let kc = KC.min(k - pc);

            pack_b_panel(right, n, pc, kc, jc, nc, &mut packed_b);

            for ic in (0..m).step_by(MC) {
                let mc = MC.min(m - ic);

                pack_a_panel(left, k, ic, mc, pc, kc, &mut packed_a);

                gebp_kernel(&packed_a, &packed_b, output, n, ic, jc, mc, nc, kc);
            }
        }
    }
}

/// Parallel blocked GEMM: parallelizes over IC (row) blocks.
///
/// Each thread gets its own packed_a buffer. Packed B is shared (read-only).
#[allow(unsafe_code)]
fn blocked_gemm_parallel(
    left: &[f32],
    right: &[f32],
    output: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    thread_pool: Option<&ThreadPool>,
) {
    // SAFETY: We split output into disjoint row ranges per IC block.
    // Different IC blocks write to non-overlapping rows.
    let out_ptr = SendPtr(output.as_mut_ptr());

    let mut packed_b = vec![0.0f32; div_ceil(NC, NR) * KC * NR];

    let ic_blocks: Vec<usize> = (0..m).step_by(MC).collect();

    let mut work = || {
        for jc in (0..n).step_by(NC) {
            let nc = NC.min(n - jc);
            for pc in (0..k).step_by(KC) {
                let kc = KC.min(k - pc);

                pack_b_panel(right, n, pc, kc, jc, nc, &mut packed_b);

                let packed_b_ref = &packed_b;
                let out_p = &out_ptr;

                ic_blocks.par_iter().for_each(|&ic| {
                    let mc = MC.min(m - ic);
                    let a_panels = div_ceil(mc, MR);
                    let mut packed_a = vec![0.0f32; a_panels * kc * MR];

                    pack_a_panel(left, k, ic, mc, pc, kc, &mut packed_a);

                    // SAFETY: Each IC block writes to rows ic..ic+mc of output.
                    // Different IC blocks are non-overlapping, so no data race.
                    unsafe {
                        gebp_kernel_raw(
                            packed_a.as_ptr(),
                            packed_b_ref.as_ptr(),
                            out_p.0,
                            n,
                            ic,
                            jc,
                            mc,
                            nc,
                            kc,
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
        );
    }
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
) {
    for jr in (0..nc).step_by(NR) {
        let nr = NR.min(nc - jr);
        let b_off = (jr / NR) * kc * NR;

        for ir in (0..mc).step_by(MR) {
            let mr = MR.min(mc - ir);
            let a_off = (ir / MR) * kc * MR;

            let c_ptr = output.add((ic + ir) * n + jc + jr);

            if mr == MR && nr == NR {
                microkernel_4x8_dispatch(packed_a.add(a_off), packed_b.add(b_off), c_ptr, n, kc);
            } else {
                microkernel_scalar_partial(
                    packed_a.add(a_off),
                    packed_b.add(b_off),
                    c_ptr,
                    n,
                    mr,
                    nr,
                    kc,
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
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
) {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("fma") && std::is_x86_feature_detected!("avx") {
            microkernel_4x8_avx_fma(a_panel, b_panel, c, ldc, kc);
            return;
        }
        if std::is_x86_feature_detected!("avx") {
            microkernel_4x8_avx(a_panel, b_panel, c, ldc, kc);
            return;
        }
        if std::is_x86_feature_detected!("sse") {
            microkernel_4x8_sse(a_panel, b_panel, c, ldc, kc);
            return;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            microkernel_4x8_neon(a_panel, b_panel, c, ldc, kc);
            return;
        }
    }

    microkernel_4x8_scalar(a_panel, b_panel, c, ldc, kc);
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
        for j in 0..NR {
            *row_ptr.add(j) += acc[i][j];
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
) {
    let mut acc = [[0.0f32; NR]; MR];

    for p in 0..kc {
        let a_base = p * MR;
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
        for j in 0..nr {
            *row_ptr.add(j) += acc[i][j];
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
) {
    // 8 accumulator registers: 4 rows × 2 halves (8 cols = 2 × float32x4).
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

    // Store: accumulate into C.
    let c0 = c;
    let c1 = c.add(ldc);
    let c2 = c.add(2 * ldc);
    let c3 = c.add(3 * ldc);

    vst1q_f32(c0, vaddq_f32(vld1q_f32(c0), c00));
    vst1q_f32(c0.add(4), vaddq_f32(vld1q_f32(c0.add(4)), c01));
    vst1q_f32(c1, vaddq_f32(vld1q_f32(c1), c10));
    vst1q_f32(c1.add(4), vaddq_f32(vld1q_f32(c1.add(4)), c11));
    vst1q_f32(c2, vaddq_f32(vld1q_f32(c2), c20));
    vst1q_f32(c2.add(4), vaddq_f32(vld1q_f32(c2.add(4)), c21));
    vst1q_f32(c3, vaddq_f32(vld1q_f32(c3), c30));
    vst1q_f32(c3.add(4), vaddq_f32(vld1q_f32(c3.add(4)), c31));
}

// ---------------------------------------------------------------------------
// AVX+FMA micro-kernel (x86/x86_64)
// ---------------------------------------------------------------------------

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code)]
#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx,fma")]
unsafe fn microkernel_4x8_avx_fma(
    a_panel: *const f32,
    b_panel: *const f32,
    c: *mut f32,
    ldc: usize,
    kc: usize,
) {
    let mut c0: __m256 = _mm256_setzero_ps();
    let mut c1: __m256 = _mm256_setzero_ps();
    let mut c2: __m256 = _mm256_setzero_ps();
    let mut c3: __m256 = _mm256_setzero_ps();

    // Unroll by 2 for better instruction-level parallelism on x86.
    // Each iteration issues 8 independent FMAs across 2 k-steps,
    // keeping the FMA pipeline saturated.
    let kc_pairs = kc / 2;
    let kc_rem = kc % 2;
    let mut p = 0usize;

    for _ in 0..kc_pairs {
        // First k-step
        let b_vec0 = _mm256_loadu_ps(b_panel.add(p * NR));
        let a00 = _mm256_set1_ps(*a_panel.add(p * MR));
        c0 = _mm256_fmadd_ps(a00, b_vec0, c0);
        let a10 = _mm256_set1_ps(*a_panel.add(p * MR + 1));
        c1 = _mm256_fmadd_ps(a10, b_vec0, c1);
        let a20 = _mm256_set1_ps(*a_panel.add(p * MR + 2));
        c2 = _mm256_fmadd_ps(a20, b_vec0, c2);
        let a30 = _mm256_set1_ps(*a_panel.add(p * MR + 3));
        c3 = _mm256_fmadd_ps(a30, b_vec0, c3);

        // Second k-step
        let b_vec1 = _mm256_loadu_ps(b_panel.add((p + 1) * NR));
        let a01 = _mm256_set1_ps(*a_panel.add((p + 1) * MR));
        c0 = _mm256_fmadd_ps(a01, b_vec1, c0);
        let a11 = _mm256_set1_ps(*a_panel.add((p + 1) * MR + 1));
        c1 = _mm256_fmadd_ps(a11, b_vec1, c1);
        let a21 = _mm256_set1_ps(*a_panel.add((p + 1) * MR + 2));
        c2 = _mm256_fmadd_ps(a21, b_vec1, c2);
        let a31 = _mm256_set1_ps(*a_panel.add((p + 1) * MR + 3));
        c3 = _mm256_fmadd_ps(a31, b_vec1, c3);

        p += 2;
    }

    // Handle odd remainder
    if kc_rem > 0 {
        let b_vec = _mm256_loadu_ps(b_panel.add(p * NR));
        let a0 = _mm256_set1_ps(*a_panel.add(p * MR));
        c0 = _mm256_fmadd_ps(a0, b_vec, c0);
        let a1 = _mm256_set1_ps(*a_panel.add(p * MR + 1));
        c1 = _mm256_fmadd_ps(a1, b_vec, c1);
        let a2 = _mm256_set1_ps(*a_panel.add(p * MR + 2));
        c2 = _mm256_fmadd_ps(a2, b_vec, c2);
        let a3 = _mm256_set1_ps(*a_panel.add(p * MR + 3));
        c3 = _mm256_fmadd_ps(a3, b_vec, c3);
    }

    // Accumulate into C.
    let cp0 = c;
    let cp1 = c.add(ldc);
    let cp2 = c.add(2 * ldc);
    let cp3 = c.add(3 * ldc);

    _mm256_storeu_ps(cp0, _mm256_add_ps(_mm256_loadu_ps(cp0), c0));
    _mm256_storeu_ps(cp1, _mm256_add_ps(_mm256_loadu_ps(cp1), c1));
    _mm256_storeu_ps(cp2, _mm256_add_ps(_mm256_loadu_ps(cp2), c2));
    _mm256_storeu_ps(cp3, _mm256_add_ps(_mm256_loadu_ps(cp3), c3));
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

    // Accumulate into C.
    let cp0 = c;
    let cp1 = c.add(ldc);
    let cp2 = c.add(2 * ldc);
    let cp3 = c.add(3 * ldc);

    _mm256_storeu_ps(cp0, _mm256_add_ps(_mm256_loadu_ps(cp0), c0));
    _mm256_storeu_ps(cp1, _mm256_add_ps(_mm256_loadu_ps(cp1), c1));
    _mm256_storeu_ps(cp2, _mm256_add_ps(_mm256_loadu_ps(cp2), c2));
    _mm256_storeu_ps(cp3, _mm256_add_ps(_mm256_loadu_ps(cp3), c3));
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
) {
    // 8 accumulators: 4 rows × 2 halves (8 cols = 2 × __m128).
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

    _mm_storeu_ps(cp0, _mm_add_ps(_mm_loadu_ps(cp0), c00));
    _mm_storeu_ps(cp0.add(4), _mm_add_ps(_mm_loadu_ps(cp0.add(4)), c01));
    _mm_storeu_ps(cp1, _mm_add_ps(_mm_loadu_ps(cp1), c10));
    _mm_storeu_ps(cp1.add(4), _mm_add_ps(_mm_loadu_ps(cp1.add(4)), c11));
    _mm_storeu_ps(cp2, _mm_add_ps(_mm_loadu_ps(cp2), c20));
    _mm_storeu_ps(cp2.add(4), _mm_add_ps(_mm_loadu_ps(cp2.add(4)), c21));
    _mm_storeu_ps(cp3, _mm_add_ps(_mm_loadu_ps(cp3), c30));
    _mm_storeu_ps(cp3.add(4), _mm_add_ps(_mm_loadu_ps(cp3.add(4)), c31));
}
