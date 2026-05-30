//! Blocked (cache-tiled) GEMM drivers: the MC/KC/NC 3-level blocking that
//! feeds the GEBP microkernels, plus the MR=6/MR=8 and NCHWc-input variants.

use super::kernels::*;
use super::pack::*;
use super::*;

// ---------------------------------------------------------------------------
// Blocked tiled GEMM
// ---------------------------------------------------------------------------

/// Sequential blocked GEMM: 3-level cache blocking with MR×NR micro-kernel.
/// If `pre_packed_b` is `Some`, uses it directly (load-time pre-packed kernel
/// weights). Otherwise falls back to a thread-local runtime cache keyed by
/// B's pointer — constant kernels still pay the pack cost only once per thread.
#[allow(unsafe_code)]
pub(super) fn blocked_gemm_sequential(
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
// occupancy = 75% vs a 4×16 tile's 50%. The `.S` kernel fuses bias+Relu into
// the store phase, keeping the occupancy win without giving up the
// single-memory-pass store (a raw-kernel + separate post-pass loses to MR=4).
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
pub(super) const MC_MR8: usize = 128;

/// MC for the aarch64 MR=8 parallel path. Multiple of 8 for clean tiling.
#[cfg(target_arch = "aarch64")]
pub(super) const MC_PARALLEL_MR8: usize = 16;

/// MC for the x86 MR=6 sequential path. Multiple of 6 for clean
/// tiling. Matches MR=4's MC=192 in total work per block (32 MR=6 panels
/// × KC = 32×256×6 = 48 KB A-pack per block, fits L2).
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
const MC_MR6: usize = 192;

/// MC for the x86 MR=6 parallel path. Multiple of 6 (18 = 3 × 6)
/// for clean tiling; smaller than sequential so thread count × blocks
/// >= nthreads across tracker shapes (m=64..1024).
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub(super) const MC_PARALLEL_MR6: usize = 18;

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code)]
pub(super) fn blocked_gemm_sequential_mr8(
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
pub(super) fn blocked_gemm_parallel_mr8(
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

    let out_p = &out_ptr;
    let work = || {
        par_for_each_ic(&ic_blocks, thread_pool, |ic| {
            let mc = MC_PARALLEL_MR8.min(m - ic);
            let a_panels = div_ceil(mc, MR8);
            for pc in (0..k).step_by(KC) {
                let pc_idx = pc / KC;
                let kc = KC.min(k - pc);
                let accumulate = pc > 0;
                let is_last_k = pc + kc >= k;
                let pa_size = a_panels * kc * MR8;
                with_packed_a_tls(pa_size, |packed_a| {
                    pack_a_panel_mr8(left, k, ic, mc, pc, kc, packed_a);
                    for jc in (0..n).step_by(NC) {
                        let jc_idx = jc / NC;
                        let nc = NC.min(n - jc);
                        let block_off = (pc_idx * num_jc + jc_idx) * block_slots;
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
                    }
                });
            }
        });
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
/// The B side uses NR=12 packing (see `pack_b_panel_nr12`) so `jr += 12`
/// stays aligned to panel starts; the 8×12 `.S` kernel consumes it directly.
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
                            super::super::simd::fused_row_epilogue_dispatch(
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
                                super::super::simd::fused_row_epilogue_dispatch(
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

/// 6×16 GEBP kernel for x86 AVX2. Iterates `jr` in 2×NR=16 col
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

/// sequential blocked GEMM using MR=6 pack + 6×16 microkernel.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code)]
pub(super) fn blocked_gemm_sequential_mr6(
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

/// parallel blocked GEMM using MR=6 pack + 6×16 microkernel.
/// Mirrors the existing `blocked_gemm_parallel` structure (MR=4) but
/// uses MR=6 stride throughout. Per-thread packed_a scratch reused via
/// `with_packed_a_tls`.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code)]
pub(super) fn blocked_gemm_parallel_mr6(
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

    let out_p = &out_ptr;
    let work = || {
        par_for_each_ic(&ic_blocks, thread_pool, |ic| {
            let mc = MC_PARALLEL_MR6.min(m - ic);
            let a_panels = div_ceil(mc, MR6);
            for pc in (0..k).step_by(KC) {
                let pc_idx = pc / KC;
                let kc = KC.min(k - pc);
                let accumulate = pc > 0;
                let is_last_k = pc + kc >= k;
                let pa_size = a_panels * kc * MR6;
                with_packed_a_tls(pa_size, |packed_a| {
                    pack_a_panel_mr6(left, k, ic, mc, pc, kc, packed_a);
                    for jc in (0..n).step_by(NC) {
                        let jc_idx = jc / NC;
                        let nc = NC.min(n - jc);
                        let block_off = (pc_idx * num_jc + jc_idx) * block_slots;
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
                    }
                });
            }
        });
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
pub(super) fn blocked_gemm_parallel(
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

    let out_p = &out_ptr;
    let work = || {
        // ONE rayon barrier for the entire GEMM. Each thread owns its ic rows
        // exclusively — no need to synchronise between pc blocks or jc blocks.
        // Old loop order (jc > pc > par_ic) created num_jc × num_pc barriers
        // and repacked A num_jc times per (ic, pc). New order (par_ic > pc > jc)
        // packs A once per (ic, pc) and zero intermediate barriers.
        par_for_each_ic(&ic_blocks, thread_pool, |ic| {
            let mc = MC_PARALLEL.min(m - ic);
            let a_panels = div_ceil(mc, MR);
            for pc in (0..k).step_by(KC) {
                let pc_idx = pc / KC;
                let kc = KC.min(k - pc);
                let accumulate = pc > 0;
                let is_last_k = pc + kc >= k;
                let pa_size = a_panels * kc * MR;
                with_packed_a_tls(pa_size, |packed_a| {
                    pack_a_panel(left, k, ic, mc, pc, kc, packed_a);
                    for jc in (0..n).step_by(NC) {
                        let jc_idx = jc / NC;
                        let nc = NC.min(n - jc);
                        let block_off = (pc_idx * num_jc + jc_idx) * block_slots;
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
                    }
                });
            }
        });
    };

    if let Some(pool) = thread_pool {
        pool.install(work);
    } else {
        work();
    }
}

/// Parallel blocked GEMM with NCHWc-layout activations.
///
/// Mirrors `blocked_gemm_parallel` but reads A from `input_nchwc`
/// (one batch item: `[Cb * hw * block]` in NCHWc order) via
/// `pack_a_panel_nchwc` instead of the standard row-major pack.
/// M = `hw` (spatial positions), K = `actual_k` (IC channels),
/// N = `n_out` (OC channels). Output is flat `[hw, n_out]`.
///
/// Used by NCHWc chain pointwise Conv to skip the NHWC↔NCHWc roundtrip
/// that `conv2d_nchwc_pointwise_with_activation_impl` previously required.
#[allow(unsafe_code)]
pub(crate) fn blocked_gemm_nchwc_a_parallel(
    input_nchwc: &[f32],
    hw: usize,
    block: usize,
    actual_k: usize,
    b_raw: &[f32],
    output: &mut [f32],
    n_out: usize,
    epilogue: GemmEpilogue,
    thread_pool: Option<&ThreadPool>,
    pre_packed_b: Option<&PackedB>,
) {
    let m = hw;
    let k = actual_k;
    let n = n_out;
    let out_ptr = SendPtr(output.as_mut_ptr());

    let rc_fallback;
    let packed_b_full: &PackedB = if let Some(pb) = pre_packed_b {
        pb
    } else {
        rc_fallback = get_or_pack_b(b_raw, k, n);
        &rc_fallback
    };
    let packed_b_ptr = packed_b_full.data.as_ptr() as usize;
    let block_slots = packed_b_full.block_slots;
    let num_jc = packed_b_full.num_jc;

    let ic_blocks: Vec<usize> = (0..m).step_by(MC_PARALLEL).collect();

    let out_p = &out_ptr;
    let work = || {
        par_for_each_ic(&ic_blocks, thread_pool, |ic| {
            let mc = MC_PARALLEL.min(m - ic);
            let a_panels = div_ceil(mc, MR);
            for pc in (0..k).step_by(KC) {
                let pc_idx = pc / KC;
                let kc = KC.min(k - pc);
                let accumulate = pc > 0;
                let is_last_k = pc + kc >= k;
                let pa_size = a_panels * kc * MR;
                with_packed_a_tls(pa_size, |packed_a| {
                    pack_a_panel_nchwc(input_nchwc, hw, block, ic, mc, pc, kc, packed_a);
                    for jc in (0..n).step_by(NC) {
                        let jc_idx = jc / NC;
                        let nc = NC.min(n - jc);
                        let block_off = (pc_idx * num_jc + jc_idx) * block_slots;
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
                    }
                });
            }
        });
    };

    if let Some(pool) = thread_pool {
        pool.install(work);
    } else {
        work();
    }
}
