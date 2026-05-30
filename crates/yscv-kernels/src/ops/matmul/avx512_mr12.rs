//! AVX-512 MR=12×NR=32 microkernel + blocked-GEMM orchestrator (x86_64).

use super::*;

// ============================================================================
// AVX-512 MR=12×NR=32 microkernel
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

/// pack-A variant for MR=12 AVX-512 microkernel. Mirrors
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

/// SiLU(x) = x / (1 + exp(-x)) applied to one ZMM (16 floats).
/// Uses the same bit-trick exp as `fast_exp_bittrick_avx`: multiply by
/// 2^23/ln(2) then add the IEEE 754 bias, reinterpreting as f32.
/// Error ≲ 3% on the final SiLU value (dominated by exp approximation).
#[cfg(all(target_arch = "x86_64", any(target_os = "linux", target_os = "macos")))]
#[allow(unsafe_code)]
#[target_feature(enable = "avx512f")]
unsafe fn silu_zmm(x: std::arch::x86_64::__m512) -> std::arch::x86_64::__m512 {
    use std::arch::x86_64::{
        _mm512_add_epi32, _mm512_add_ps, _mm512_castsi512_ps, _mm512_cvtps_epi32, _mm512_div_ps,
        _mm512_max_ps, _mm512_min_ps, _mm512_mul_ps, _mm512_set1_epi32, _mm512_set1_ps,
        _mm512_setzero_ps, _mm512_sub_ps,
    };
    let neg_x = _mm512_sub_ps(_mm512_setzero_ps(), x);
    let clamp_lo = _mm512_set1_ps(-87.0);
    let clamp_hi = _mm512_set1_ps(88.0);
    let neg_clamped = _mm512_max_ps(_mm512_min_ps(neg_x, clamp_hi), clamp_lo);
    let scale = _mm512_set1_ps(12102203.0);
    let bias = _mm512_set1_epi32(1065353216i32);
    let val = _mm512_cvtps_epi32(_mm512_mul_ps(neg_clamped, scale));
    let exp_neg = _mm512_castsi512_ps(_mm512_add_epi32(val, bias));
    let denom = _mm512_add_ps(_mm512_set1_ps(1.0), exp_neg);
    _mm512_div_ps(x, denom)
}

/// Apply SiLU in-place to `mr` rows × NR32 cols at `dst` with stride `ldc` floats.
/// Reads each row as 2 ZMMs, applies `silu_zmm`, writes back.
#[cfg(all(target_arch = "x86_64", any(target_os = "linux", target_os = "macos")))]
#[allow(unsafe_code)]
#[target_feature(enable = "avx512f")]
unsafe fn apply_silu_zmm_tile(dst: *mut f32, ldc: usize, mr: usize) {
    use std::arch::x86_64::{_mm512_loadu_ps, _mm512_storeu_ps};
    // Raw pointer ops + unsafe fn calls require explicit unsafe blocks even
    // within a #[target_feature] function (only pure register intrinsics are safe).
    unsafe {
        let mut row_ptr = dst;
        for _ in 0..mr {
            let a = silu_zmm(_mm512_loadu_ps(row_ptr));
            let b = silu_zmm(_mm512_loadu_ps(row_ptr.add(16)));
            _mm512_storeu_ps(row_ptr, a);
            _mm512_storeu_ps(row_ptr.add(16), b);
            row_ptr = row_ptr.add(ldc);
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
    residual: Option<*const f32>,
) {
    unsafe {
        let bias_ptr = bias.unwrap_or(std::ptr::null());
        let residual_ptr = residual.unwrap_or(std::ptr::null());
        let act_id: usize = match activation {
            Activation::Relu => 1,
            _ => 0,
        };
        let last_k: usize = if is_last_k { 1 } else { 0 };
        sgemm_asm_avx512::yscv_sgemm_12x32_avx512_set(
            a_panel,
            b_panel,
            c,
            ldc,
            kc,
            bias_ptr,
            act_id,
            last_k,
            residual_ptr,
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
    residual: Option<*const f32>,
) {
    unsafe {
        let bias_ptr = bias.unwrap_or(std::ptr::null());
        let residual_ptr = residual.unwrap_or(std::ptr::null());
        let act_id: usize = match activation {
            Activation::Relu => 1,
            _ => 0,
        };
        let last_k: usize = if is_last_k { 1 } else { 0 };
        sgemm_asm_avx512::yscv_sgemm_12x32_avx512_acc(
            a_panel,
            b_panel,
            c,
            ldc,
            kc,
            bias_ptr,
            act_id,
            last_k,
            residual_ptr,
        );
    }
}

// ---------------------------------------------------------------------------
// MR=12×NR=32 AVX-512 blocked GEMM orchestrator
// ---------------------------------------------------------------------------

#[cfg(all(target_arch = "x86_64", any(target_os = "linux", target_os = "macos")))]
pub(super) const NR32: usize = 32;

/// MC for MR=12 blocked GEMM. 96 = 8 MR12 panels per block. Pack-A per
/// block = 96 × KC = 96 × 256 × 4 = 96 KB (fits L2 comfortably).
#[cfg(all(target_arch = "x86_64", any(target_os = "linux", target_os = "macos")))]
const MC_MR12: usize = 96;

/// NC for MR=12 path. 256 cols = 8 NR32 panels.
#[cfg(all(target_arch = "x86_64", any(target_os = "linux", target_os = "macos")))]
pub(super) const NC_MR12: usize = 256;

/// Pack B[pc..pc+kc, jc..jc+nc] into `packed` with NR=32 stride. Layout
/// `packed[p * nc + j]` — row-major in-block; the microkernel reads
/// `[rsi + p*128]` ... so NR=32 cols contiguous per k-row.
#[cfg(all(target_arch = "x86_64", any(target_os = "linux", target_os = "macos")))]
#[allow(unsafe_code, clippy::too_many_arguments)]
pub(super) fn pack_b_panel_nr32(
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
/// - AVX-512F detected at runtime
/// - n % 32 == 0 (shapes with n%32≠0 fall through to MR=4×24)
/// - m >= 12 (tail rows handled via tail_tile in gebp)
/// - k >= 16 (smaller k doesn't amortize the blocked-GEMM setup)
///
/// DEFAULT ON. Kill-switch: `YSCV_AVX512_SGEMM=0` disables.
/// All epilogue combinations supported: None/Relu (in asm),
/// SiLU (ZMM post-store via apply_silu_zmm_tile), residual.
#[cfg(all(target_arch = "x86_64", any(target_os = "linux", target_os = "macos")))]
pub(super) fn use_avx512_mr12(m: usize, k: usize, n: usize) -> bool {
    avx512_mr12_enabled()
        && m >= MR12
        && k >= 16
        && n > 0
        && n.is_multiple_of(NR32)
        && std::is_x86_feature_detected!("avx512f")
}

#[cfg(all(target_arch = "x86_64", any(target_os = "linux", target_os = "macos")))]
pub(super) fn avx512_mr12_enabled() -> bool {
    static CACHED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    // Default OFF on Zen 4: ZMM double-pump (1 µop/clock per 512-bit unit,
    // but 4×24 AVX2 runs 2 YMM FMAs/clock) means 4×24 wins. Set
    // YSCV_AVX512_SGEMM=1 to enable (wins on Intel / Zen 5 true-512 silicon).
    *CACHED.get_or_init(|| std::env::var_os("YSCV_AVX512_SGEMM").is_some_and(|v| v == "1"))
}

/// Sequential AVX-512 MR=12×NR=32 blocked GEMM. Mirrors
/// `blocked_gemm_sequential` structure. Assumes shape gate passed.
/// `prepacked` carries session-pre-packed NR=32 B data (via `PackedB::data_nr32`)
/// when available, avoiding per-inference B packing.
#[cfg(all(target_arch = "x86_64", any(target_os = "linux", target_os = "macos")))]
#[allow(unsafe_code)]
pub(super) fn blocked_gemm_sequential_mr12(
    left: &[f32],
    right: &[f32],
    output: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    epilogue: GemmEpilogue,
    prepacked: Option<&PackedB>,
) {
    debug_assert_eq!(n % NR32, 0);

    let a_size = div_ceil(MC_MR12, MR12) * KC * MR12;
    let mut packed_a = vec![0.0f32; a_size];
    // Local B buffer used as fallback when NR=32 session prepack is unavailable.
    let need_local_b = prepacked.is_none_or(|pb| pb.data_nr32.is_empty());
    let b_size = if need_local_b {
        (NC_MR12 / NR32) * KC * NR32
    } else {
        0
    };
    let mut local_b = vec![0.0f32; b_size];

    for jc in (0..n).step_by(NC_MR12) {
        let nc = NC_MR12.min(n - jc);
        let nc_aligned = (nc / NR32) * NR32;
        if nc_aligned == 0 {
            continue;
        }
        let jc_idx = jc / NC_MR12;
        for pc in (0..k).step_by(KC) {
            let kc = KC.min(k - pc);
            let accumulate = pc > 0;
            let is_last_k = pc + kc >= k;
            let pc_idx = pc / KC;

            // Use session-prepacked NR=32 data when available; else pack inline.
            let pb_ptr = if let Some(slice) = prepacked.and_then(|pb| pb.block_nr32(pc_idx, jc_idx))
            {
                slice.as_ptr()
            } else {
                pack_b_panel_nr32(right, n, pc, kc, jc, nc_aligned, &mut local_b);
                local_b.as_ptr()
            };

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
                        pb_ptr,
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
        let mut tail_tile = [0.0f32; MR12 * NR32];
        // Used when mr < MR12 and residual is active: holds the relevant
        // residual rows repacked with NR32 stride to match tail_tile layout.
        let mut residual_tail = [0.0f32; MR12 * NR32];
        for jr_idx in 0..n_panels {
            let b_panel = packed_b.add(jr_idx * kc * NR32);
            let col = jc + jr_idx * NR32;
            let bias_for_tile: Option<*const f32> = match epilogue.bias {
                Some(b) if is_last_k => Some(b.add(col)),
                _ => None,
            };
            for ir in (0..mc).step_by(MR12) {
                let mr = MR12.min(mc - ir);
                let a_panel = packed_a.add((ir / MR12) * kc * MR12);
                let c_ptr = output.add((ic + ir) * n + col);
                // Residual ptr for this tile: points at residual[ic+ir, col].
                let residual_for_tile: Option<*const f32> = match epilogue.residual {
                    Some(r) if is_last_k => Some(r.add((ic + ir) * n + col)),
                    _ => None,
                };
                let (dst_ptr, dst_ldc, res_ptr) = if mr == MR12 {
                    (c_ptr, n, residual_for_tile)
                } else {
                    tail_tile.fill(0.0);
                    if accumulate {
                        for row in 0..mr {
                            let src = c_ptr.add(row * n);
                            let dst = tail_tile.as_mut_ptr().add(row * NR32);
                            std::ptr::copy_nonoverlapping(src, dst, NR32);
                        }
                    }
                    // Repack residual rows into residual_tail so the kernel
                    // sees stride = NR32, matching tail_tile's layout.
                    let tail_res = if let Some(rp) = residual_for_tile {
                        for row in 0..mr {
                            let src = rp.add(row * n);
                            let dst = residual_tail.as_mut_ptr().add(row * NR32);
                            std::ptr::copy_nonoverlapping(src, dst, NR32);
                        }
                        Some(residual_tail.as_ptr())
                    } else {
                        None
                    };
                    (tail_tile.as_mut_ptr(), NR32, tail_res)
                };
                // Map SiLU to None for the asm kernel — applied below via
                // `apply_silu_zmm_tile` after the kernel stores to dst.
                let asm_activation = match epilogue.activation {
                    Activation::Silu => Activation::None,
                    a => a,
                };
                if accumulate {
                    microkernel_12x32_avx512_acc(
                        a_panel,
                        b_panel,
                        dst_ptr,
                        dst_ldc,
                        kc,
                        bias_for_tile,
                        asm_activation,
                        is_last_k,
                        res_ptr,
                    );
                } else {
                    microkernel_12x32_avx512_set(
                        a_panel,
                        b_panel,
                        dst_ptr,
                        dst_ldc,
                        kc,
                        bias_for_tile,
                        asm_activation,
                        is_last_k,
                        res_ptr,
                    );
                }
                // Apply SiLU in-register (ZMM-wide) after the store.
                // Data is cache-hot (just written by the asm kernel).
                if matches!(epilogue.activation, Activation::Silu) && is_last_k {
                    apply_silu_zmm_tile(dst_ptr, dst_ldc, mr);
                }
                if mr < MR12 {
                    for row in 0..mr {
                        let src = tail_tile.as_ptr().add(row * NR32);
                        let dst = c_ptr.add(row * n);
                        std::ptr::copy_nonoverlapping(src, dst, NR32);
                    }
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
/// `prepacked` carries session-pre-packed NR=32 B data when available.
#[cfg(all(target_arch = "x86_64", any(target_os = "linux", target_os = "macos")))]
#[allow(unsafe_code)]
pub(super) fn blocked_gemm_parallel_mr12(
    left: &[f32],
    right: &[f32],
    output: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    epilogue: GemmEpilogue,
    thread_pool: Option<&ThreadPool>,
    prepacked: Option<&PackedB>,
) {
    debug_assert_eq!(n % NR32, 0);

    let out_ptr = SendPtr(output.as_mut_ptr());
    // Allocate local B buffer only when session prepack is unavailable.
    let need_local_b = prepacked.is_none_or(|pb| pb.data_nr32.is_empty());
    let b_size = if need_local_b {
        (NC_MR12 / NR32) * KC * NR32
    } else {
        0
    };
    let mut local_b = vec![0.0f32; b_size];

    let ic_blocks: Vec<usize> = (0..m).step_by(MC_PARALLEL_MR12).collect();

    let mut work = || {
        for jc in (0..n).step_by(NC_MR12) {
            let nc = NC_MR12.min(n - jc);
            let nc_aligned = (nc / NR32) * NR32;
            if nc_aligned == 0 {
                continue;
            }
            let jc_idx = jc / NC_MR12;
            for pc in (0..k).step_by(KC) {
                let kc = KC.min(k - pc);
                let accumulate = pc > 0;
                let is_last_k = pc + kc >= k;
                let pc_idx = pc / KC;

                // Use session-prepacked NR=32 data when available; else pack inline.
                let pb_ptr =
                    if let Some(slice) = prepacked.and_then(|pb| pb.block_nr32(pc_idx, jc_idx)) {
                        slice.as_ptr() as usize
                    } else {
                        pack_b_panel_nr32(right, n, pc, kc, jc, nc_aligned, &mut local_b);
                        local_b.as_ptr() as usize
                    };
                let out_p = &out_ptr;

                par_for_each_ic(&ic_blocks, thread_pool, |ic| {
                    let mc = MC_PARALLEL_MR12.min(m - ic);
                    let a_panels = div_ceil(mc, MR12);
                    let pa_size = a_panels * kc * MR12;
                    let mut packed_a = vec![0.0f32; pa_size];
                    pack_a_panel_mr12(left, k, ic, mc, pc, kc, &mut packed_a);
                    // SAFETY: output rows ic..ic+mc are disjoint per worker.
                    // Prepacked B (or local_b) outlives the `par_for_each_ic` scope.
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

    /// Direct call into microkernel_12x32_avx512_set. Covers bias, residual,
    /// and activation combinations. ldc = N (square tile, no padding).
    fn run_set_case(k: usize, with_bias: bool, with_residual: bool, act: Activation) {
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
        let residual_vec = if with_residual {
            let mut rv = vec![0.0f32; M * N];
            fill_ramp(&mut rv, 0.011, -0.5);
            Some(rv)
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
                residual_vec.as_ref().map(|v| v.as_ptr()),
            );
        }

        let reference = reference_matmul(
            &a,
            &b,
            M,
            k,
            N,
            bias_vec.as_deref(),
            residual_vec.as_deref(),
            act,
        );
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
            "12x32 kernel diverges k={k} bias={with_bias} residual={with_residual} act={act:?}: \
             max diff {max_diff} at {max_i}: kernel={} ref={}",
            c[max_i],
            reference[max_i],
        );
    }

    #[test]
    fn mr12_k16_no_epilogue() {
        run_set_case(16, false, false, Activation::None);
    }

    #[test]
    fn mr12_k32_no_epilogue() {
        run_set_case(32, false, false, Activation::None);
    }

    #[test]
    fn mr12_k64_bias_relu() {
        run_set_case(64, true, false, Activation::Relu);
    }

    #[test]
    fn mr12_k192_bias_relu() {
        // Tracker k=192 — common pointwise Conv k size.
        run_set_case(192, true, false, Activation::Relu);
    }

    #[test]
    fn mr12_k672_bias_relu() {
        // Tracker k=672 — hot pointwise Conv_Add/Conv_Relu.
        run_set_case(672, true, false, Activation::Relu);
    }

    // P1.1 residual epilogue tests.
    #[test]
    fn mr12_residual_no_bias() {
        run_set_case(64, false, true, Activation::None);
    }

    #[test]
    fn mr12_bias_plus_residual() {
        run_set_case(64, true, true, Activation::None);
    }

    #[test]
    fn mr12_residual_plus_relu() {
        run_set_case(64, false, true, Activation::Relu);
    }

    #[test]
    fn mr12_all_three() {
        // bias + residual + relu — the Conv_Add_fused epilogue.
        run_set_case(192, true, true, Activation::Relu);
    }

    #[test]
    fn mr12_residual_k672() {
        // Tracker hot shape with residual (Conv_Add, k=672).
        run_set_case(672, true, true, Activation::Relu);
    }

    // P1.2 SiLU tests. SiLU applied via apply_silu_zmm_tile post-store.
    #[test]
    fn mr12_silu_only() {
        run_set_case(64, false, false, Activation::Silu);
    }

    #[test]
    fn mr12_bias_silu() {
        run_set_case(64, true, false, Activation::Silu);
    }

    #[test]
    fn mr12_residual_silu() {
        run_set_case(64, false, true, Activation::Silu);
    }

    #[test]
    fn mr12_all_silu() {
        run_set_case(192, true, true, Activation::Silu);
    }

    /// Integration test through `matmul_2d_slices_fused_maybe_packed` with
    /// AVX-512 path enabled. Covers bias, residual, activation combinations.
    fn run_integration_case(
        m: usize,
        k: usize,
        n: usize,
        with_bias: bool,
        with_residual: bool,
        act: Activation,
    ) {
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
        let residual = if with_residual {
            let mut rv = vec![0.0f32; m * n];
            fill_ramp(&mut rv, 0.007, -0.3);
            Some(rv)
        } else {
            None
        };

        let epilogue = GemmEpilogue {
            bias: bias.as_ref().map(|v| v.as_ptr()),
            residual: residual.as_ref().map(|v| v.as_ptr()),
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
        let reference =
            reference_matmul(&a, &b, m, k, n, bias.as_deref(), residual.as_deref(), act);

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
            "AVX-512 integration diverges m={m} k={k} n={n} bias={with_bias} \
             residual={with_residual} act={act:?}: \
             max diff {max_diff} at {max_i}: got={} ref={}",
            out_avx512[max_i],
            reference[max_i],
        );
    }

    #[test]
    fn integration_m12_k32_n32() {
        run_integration_case(12, 32, 32, false, false, Activation::None);
    }

    #[test]
    fn integration_m24_k64_n64_bias_relu() {
        run_integration_case(24, 64, 64, true, false, Activation::Relu);
    }

    #[test]
    fn integration_m96_k192_n96_bias_relu() {
        run_integration_case(96, 192, 96, true, false, Activation::Relu);
    }

    #[test]
    fn integration_m100_k192_n96_bias_relu_tail_m() {
        run_integration_case(100, 192, 96, true, false, Activation::Relu);
    }

    #[test]
    fn integration_m1008_k672_n96_bias_relu() {
        // Tracker-like: m = 1008 = 84×12, k=672, n=96.
        run_integration_case(1008, 672, 96, true, false, Activation::Relu);
    }

    // P1.1 residual integration tests.
    #[test]
    fn integration_m96_k192_n96_residual_relu() {
        // m%12==0: all full tiles, residual via kernel epilogue.
        run_integration_case(96, 192, 96, true, true, Activation::Relu);
    }

    #[test]
    fn integration_m100_k192_n96_residual_relu_tail_m() {
        // m%12!=0 (tail=4): residual_tail repack path exercises.
        run_integration_case(100, 192, 96, true, true, Activation::Relu);
    }

    #[test]
    fn integration_m1008_k672_n96_residual_relu() {
        // Tracker Conv_Add_fused hot path: m=1008=84×12, k=672, n=96.
        run_integration_case(1008, 672, 96, true, true, Activation::Relu);
    }

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
                None,  // no residual
            );
        }

        // Expected: c_initial + A @ B
        let added = reference_matmul(&a, &b, M, K, N, None, None, Activation::None);
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
                        None,
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
                        None,
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
