//! Specialized low-k pointwise tile (MR=4 x NR=24, const K) for hot
//! Conv shapes (k in {16,24}) the blocked GEMM gate rejects.

use super::*;

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

/// Shape gate for the low-k tile fast path. Hard-coded MR=4, NR=24
/// alignment plus minimum FMA count (avoids small-matrix regression
/// where rayon dispatch dominates).
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline]
pub(super) fn use_low_k_tile_avx_fma(m: usize, k: usize, n: usize) -> bool {
    const MIN_WORK_FMAS: usize = 1_048_576;
    (k == 16 || k == 24)
        && m != 0
        && n != 0
        && m.is_multiple_of(4)
        && n.is_multiple_of(24)
        && m.saturating_mul(n).saturating_mul(k) >= MIN_WORK_FMAS
}

/// inner tile: 4 rows × NR=24 cols, const-K k-loop, stride-n
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

/// orchestrator. Chunks output by 4 rows × n cols, iterates
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
pub(super) unsafe fn low_k_tile_4x24_parallel_fused(
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

        // Avoid rayon fork/join overhead on a single thread: even with
        // `RAYON_NUM_THREADS=1` the `par_chunks_mut` path constructs per-chunk
        // task state, which measurably regresses single-thread latency.
        let num_threads = rayon::current_num_threads();
        if num_threads <= 1 || n_blocks < 4 {
            for blk in 0..n_blocks {
                let start = blk * row_block_out;
                let end = start + row_block_out;
                run_one(blk, &mut out[start..end]);
            }
        } else {
            super::super::super::scope_ctx::par_chunks_mut_dispatch(
                out,
                row_block_out,
                move |blk_idx, out_chunk| {
                    run_one(blk_idx, out_chunk);
                },
            );
        }
    }
}

/// `YSCV_LOW_K_TILE` kill-switch for the specialized low-k pointwise tile.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub(super) fn low_k_tile_enabled() -> bool {
    static CACHED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| std::env::var_os("YSCV_LOW_K_TILE").is_some())
}
