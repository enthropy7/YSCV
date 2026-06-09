//! AVX-512 + FMA register-blocked fused PW-expand→DW(→PW-reduce) kernels,
//! plus the NCHWc-streaming row tiles. Accumulators stay in ZMM across the
//! whole inner loop.

use super::{Dw5RowCtx, Dw5TileCtx, PwTileCtx};
use std::arch::x86_64::*;

/// `YSCV_FUSED_PW_DW_4X6_OFF=1` disables the 4×6 OC×OW PW tile and
/// forces the older 6-ZMM register-blocked path. Useful for A/B
/// benchmarking. Default: 4×6 enabled.
fn four_x_six_disabled() -> bool {
    static CACHED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| std::env::var_os("YSCV_FUSED_PW_DW_4X6_OFF").is_some())
}

/// Minimum OC-block count to consider the 4×6 tile. Below 4 we have
/// only an OC tail (≤3 chunks) so tile-widening gains nothing over
/// the 6-ZMM path.
const MIN_CHUNKS_FOR_4X6: usize = 4;

/// AVX-512 PW row: dispatches between the new MLAS-style 4 OC × 6 OW
/// tile (24 ZMM accumulators → higher broadcast reuse, FMA-bound
/// inner loop) and the legacy 6-ZMM register-blocked path for narrow
/// OW tiles or shapes that fall through the gate.
///
/// Gate (all must hold for 4×6):
/// - `c_exp % 16 == 0` (implied by dispatcher)
/// - `c_exp / 16 ≥ 4` (at least one full 4-chunk OC-outer tile)
/// - OW tile width `≥ 6` (at least one full 6-OW tile)
/// - `YSCV_FUSED_PW_DW_4X6_OFF` unset
pub(super) fn compute_pw_row_avx512(
    src_row: &[f32],
    pw_weight: &[f32],
    pw_bias: Option<&[f32]>,
    dst_row: &mut [f32],
    iw_start: usize,
    iw_end: usize,
    c_in: usize,
    c_exp: usize,
    relu: bool,
) {
    let use_4x6 = !four_x_six_disabled()
        && c_exp / 16 >= MIN_CHUNKS_FOR_4X6
        && iw_end.saturating_sub(iw_start) >= 6;
    // SAFETY: dispatcher confirmed `avx512f` and `c_exp / 16 ≤ 16`.
    #[allow(unsafe_code)]
    unsafe {
        if use_4x6 {
            compute_pw_row_avx512_4x6_inner(
                src_row, pw_weight, pw_bias, dst_row, iw_start, iw_end, c_in, c_exp, relu,
            )
        } else {
            compute_pw_row_avx512_inner(
                src_row, pw_weight, pw_bias, dst_row, iw_start, iw_end, c_in, c_exp, relu,
            )
        }
    }
}

#[target_feature(enable = "avx512f")]
#[allow(unsafe_code, clippy::too_many_arguments)]
unsafe fn compute_pw_row_avx512_inner(
    src_row: &[f32],
    pw_weight: &[f32],
    pw_bias: Option<&[f32]>,
    dst_row: &mut [f32],
    iw_start: usize,
    iw_end: usize,
    c_in: usize,
    c_exp: usize,
    relu: bool,
) {
    unsafe {
        let c_exp_chunks = c_exp / 16;
        debug_assert!(c_exp_chunks <= super::AVX512_REG_MAX_CHUNKS);
        let zero = _mm512_setzero_ps();

        // Fixed-size stack array of 16 ZMM slots — sized to
        // `AVX512_REG_MAX_CHUNKS` so LLVM can keep the live prefix in
        // registers and skip touching the untouched tail. Declaring as
        // a `[__m512; N]` (not Vec) is what lets the optimiser allocate
        // registers statically.
        let mut accs: [__m512; super::AVX512_REG_MAX_CHUNKS] =
            [_mm512_setzero_ps(); super::AVX512_REG_MAX_CHUNKS];

        for iw in iw_start..iw_end {
            let src_off = iw * c_in;
            let dst_off = iw * c_exp;

            // Seed accs from bias or zero.
            if let Some(b) = pw_bias {
                let bp = b.as_ptr();
                for ck in 0..c_exp_chunks {
                    accs[ck] = _mm512_loadu_ps(bp.add(ck * 16));
                }
            } else {
                for ck in 0..c_exp_chunks {
                    accs[ck] = zero;
                }
            }

            // Inner loop: broadcast one source lane, load `c_exp_chunks`
            // weight rows, FMA each into its acc. Accs stay in regs.
            // STREAM-PROBE prefetch: ci-stride is `c_exp * 4` bytes.
            const PF_AHEAD_PW_ROW: usize = 4;
            let pf_on_pw_row = pw_prefetch_enabled();
            for ci in 0..c_in {
                if pf_on_pw_row && ci + PF_AHEAD_PW_ROW < c_in {
                    let pf_ptr = pw_weight.as_ptr().add((ci + PF_AHEAD_PW_ROW) * c_exp);
                    _mm_prefetch(pf_ptr as *const i8, _MM_HINT_T0);
                }
                let x = _mm512_set1_ps(*src_row.as_ptr().add(src_off + ci));
                let w_row = pw_weight.as_ptr().add(ci * c_exp);
                for ck in 0..c_exp_chunks {
                    let w = _mm512_loadu_ps(w_row.add(ck * 16));
                    accs[ck] = _mm512_fmadd_ps(x, w, accs[ck]);
                }
            }

            // Optional Relu + store.
            let dst_ptr = dst_row.as_mut_ptr().add(dst_off);
            if relu {
                for ck in 0..c_exp_chunks {
                    let v = _mm512_max_ps(accs[ck], zero);
                    _mm512_storeu_ps(dst_ptr.add(ck * 16), v);
                }
            } else {
                for ck in 0..c_exp_chunks {
                    _mm512_storeu_ps(dst_ptr.add(ck * 16), accs[ck]);
                }
            }
        }
    }
}

/// MLAS-style 4×6 OC×OW tile PW row. Holds **24 ZMM accumulators**
/// (`4 OC-blocks × 6 OW cols`) live across the full `c_in` reduction
/// and FMAs 24 products per inner iteration — 4× more reuse per
/// scalar broadcast than the legacy 6-ZMM path. Register budget:
/// 24 acc + 6 broadcast-A + 1 weight + scratch ≈ 31 of 32 ZMM.
///
/// Non-aligned OW tail (< 6 cols remaining) falls through to the
/// 6-ZMM `compute_pw_row_avx512_inner` for the last 1..5 output cols.
/// Non-aligned OC tail (< 4 chunks remaining within the row) uses a
/// dedicated 1..3 OC × 6 OW inner pass inline.
///
/// Matches MLAS's `ProcessPointwiseFilterCountN` with FilterCount=4
/// in `SconvKernelCommon.h` — 4 OC-blocks per `vmovups` weight load,
/// 6 OW cols per `vbroadcastss` broadcast.
#[target_feature(enable = "avx512f")]
#[allow(unsafe_code, clippy::too_many_arguments)]
unsafe fn compute_pw_row_avx512_4x6_inner(
    src_row: &[f32],
    pw_weight: &[f32],
    pw_bias: Option<&[f32]>,
    dst_row: &mut [f32],
    iw_start: usize,
    iw_end: usize,
    c_in: usize,
    c_exp: usize,
    relu: bool,
) {
    unsafe {
        let c_exp_chunks = c_exp / 16;
        let num_oc4 = c_exp_chunks / 4;
        let oc_tail_chunks = c_exp_chunks % 4; // 0..=3
        let zero = _mm512_setzero_ps();

        let width = iw_end - iw_start;
        let main_ow_tiles = width / 6;
        let ow_tail_start = iw_start + main_ow_tiles * 6;

        for tile in 0..main_ow_tiles {
            let iw_base = iw_start + tile * 6;

            // Full 4-block OC-outer tiles: 24 ZMM live across c_in loop.
            for oc4_idx in 0..num_oc4 {
                let lane_off = oc4_idx * 64; // 4 chunks × 16 lanes

                // 24 accumulators laid out as acc[oc * 6 + ow].
                let mut acc: [__m512; 24] = [zero; 24];
                if let Some(b) = pw_bias {
                    let bp = b.as_ptr().add(lane_off);
                    for oc in 0..4 {
                        let bv = _mm512_loadu_ps(bp.add(oc * 16));
                        for ow in 0..6 {
                            acc[oc * 6 + ow] = bv;
                        }
                    }
                }

                // c_in reduction. For each ci:
                //   - broadcast 6 scalar inputs (one per OW col)
                //   - for each of 4 OC-blocks: load weight, 6 FMAs
                // 24 FMAs per ci step → load-port utilisation falls
                // below FMA throughput, leaves kernel FMA-bound.
                // STREAM-PROBE: prefetch next ci's weight cacheline (4
                // chunks; HW catches the 3 trailing 64 B lines).
                const PF_AHEAD_4X6: usize = 4;
                let pf_on_4x6 = pw_prefetch_enabled();
                for ci in 0..c_in {
                    if pf_on_4x6 && ci + PF_AHEAD_4X6 < c_in {
                        let pf_ptr = pw_weight
                            .as_ptr()
                            .add((ci + PF_AHEAD_4X6) * c_exp + lane_off);
                        _mm_prefetch(pf_ptr as *const i8, _MM_HINT_T0);
                    }
                    let src_ptr = src_row.as_ptr();
                    let mut bcast: [__m512; 6] = [zero; 6];
                    for ow in 0..6 {
                        let iw = iw_base + ow;
                        bcast[ow] = _mm512_set1_ps(*src_ptr.add(iw * c_in + ci));
                    }
                    let w_base = pw_weight.as_ptr().add(ci * c_exp + lane_off);
                    for oc in 0..4 {
                        let w = _mm512_loadu_ps(w_base.add(oc * 16));
                        for ow in 0..6 {
                            acc[oc * 6 + ow] = _mm512_fmadd_ps(w, bcast[ow], acc[oc * 6 + ow]);
                        }
                    }
                }

                // Relu + store (24 ZMMs).
                for oc in 0..4 {
                    for ow in 0..6 {
                        let iw = iw_base + ow;
                        let dst_ptr = dst_row.as_mut_ptr().add(iw * c_exp + lane_off + oc * 16);
                        let v = if relu {
                            _mm512_max_ps(acc[oc * 6 + ow], zero)
                        } else {
                            acc[oc * 6 + ow]
                        };
                        _mm512_storeu_ps(dst_ptr, v);
                    }
                }
            }

            // OC-outer tail: 1, 2, or 3 remaining OC chunks × 6 OW.
            if oc_tail_chunks > 0 {
                let lane_off = num_oc4 * 64;

                // Up to 18 accumulators (3 chunks × 6 OW). Fixed size
                // so LLVM allocates registers statically.
                let mut acc: [__m512; 18] = [zero; 18];
                if let Some(b) = pw_bias {
                    let bp = b.as_ptr().add(lane_off);
                    for oc in 0..oc_tail_chunks {
                        let bv = _mm512_loadu_ps(bp.add(oc * 16));
                        for ow in 0..6 {
                            acc[oc * 6 + ow] = bv;
                        }
                    }
                }

                const PF_AHEAD_4X6_TAIL: usize = 4;
                let pf_on_4x6_tail = pw_prefetch_enabled();
                for ci in 0..c_in {
                    if pf_on_4x6_tail && ci + PF_AHEAD_4X6_TAIL < c_in {
                        let pf_ptr = pw_weight
                            .as_ptr()
                            .add((ci + PF_AHEAD_4X6_TAIL) * c_exp + lane_off);
                        _mm_prefetch(pf_ptr as *const i8, _MM_HINT_T0);
                    }
                    let src_ptr = src_row.as_ptr();
                    let mut bcast: [__m512; 6] = [zero; 6];
                    for ow in 0..6 {
                        let iw = iw_base + ow;
                        bcast[ow] = _mm512_set1_ps(*src_ptr.add(iw * c_in + ci));
                    }
                    let w_base = pw_weight.as_ptr().add(ci * c_exp + lane_off);
                    for oc in 0..oc_tail_chunks {
                        let w = _mm512_loadu_ps(w_base.add(oc * 16));
                        for ow in 0..6 {
                            acc[oc * 6 + ow] = _mm512_fmadd_ps(w, bcast[ow], acc[oc * 6 + ow]);
                        }
                    }
                }

                for oc in 0..oc_tail_chunks {
                    for ow in 0..6 {
                        let iw = iw_base + ow;
                        let dst_ptr = dst_row.as_mut_ptr().add(iw * c_exp + lane_off + oc * 16);
                        let v = if relu {
                            _mm512_max_ps(acc[oc * 6 + ow], zero)
                        } else {
                            acc[oc * 6 + ow]
                        };
                        _mm512_storeu_ps(dst_ptr, v);
                    }
                }
            }
        }

        // OW tail (< 6 cols): fall through to the 6-ZMM path for the
        // remaining cols. Broadcast reuse is 6 FMAs/broadcast even
        // at 1 OW col, which is still the best available pattern
        // for the tail.
        if ow_tail_start < iw_end {
            compute_pw_row_avx512_inner(
                src_row,
                pw_weight,
                pw_bias,
                dst_row,
                ow_tail_start,
                iw_end,
                c_in,
                c_exp,
                relu,
            );
        }
    }
}

/// Tile width (in 16-lane ZMM chunks) for the tiled AVX-512 variant below.
/// 8 chunks = 128 `c_exp` lanes per tile: 8 accumulators + 2 register temps
/// (x broadcast + w load), well within the 32-ZMM file. Wider tiles (16)
/// lose: the dynamic residual-chunk loop LLVM leaves un-unrolled offsets
/// the fewer c_in re-reads.
pub(super) const TILED_CHUNKS: usize = 8;
const TILED4X6_CHUNKS: usize = 4;
// OW=4 eliminates the per-pixel tail for all power-of-2 widths (8,16,32,64)
// that tracker shapes use (16 % 6 = 4 tail pixels fell through to the slower
// compute_pw_row_avx512_tiled_packed_inner which doesn't amortize W-loads).
const TILED4X6_OW: usize = 4;

/// AVX-512 **tiled** PW row for `c_exp > 256`. Processes `c_exp`
/// in tiles of `TILED_CHUNKS * 16 = 128` lanes per output pixel:
/// inside each tile the 8 accumulators stay in ZMM registers
/// through the full `c_in` tap loop (same trick as the reg-block
/// path above). Outer tile loop re-initialises accumulators per
/// tile, so the memory traffic scales with `c_exp / 128` full
/// write-passes — still far fewer than the AVX2 fallback's
/// dst-through-memory accumulation on every `(c_in, chunk)` step.
///
/// Any residual of `c_exp_chunks % TILED_CHUNKS` (1..=7 chunks) is
/// handled by a trailing pass over a separate 7-slot stack array.
pub(super) fn compute_pw_row_avx512_tiled(
    src_row: &[f32],
    pw_weight: &[f32],
    pw_bias: Option<&[f32]>,
    dst_row: &mut [f32],
    iw_start: usize,
    iw_end: usize,
    c_in: usize,
    c_exp: usize,
    relu: bool,
) {
    // SAFETY: dispatcher confirmed `avx512f`.
    #[allow(unsafe_code)]
    unsafe {
        compute_pw_row_avx512_tiled_inner(
            src_row, pw_weight, pw_bias, dst_row, iw_start, iw_end, c_in, c_exp, relu,
        )
    }
}

#[target_feature(enable = "avx512f")]
#[allow(unsafe_code, clippy::too_many_arguments)]
unsafe fn compute_pw_row_avx512_tiled_inner(
    src_row: &[f32],
    pw_weight: &[f32],
    pw_bias: Option<&[f32]>,
    dst_row: &mut [f32],
    iw_start: usize,
    iw_end: usize,
    c_in: usize,
    c_exp: usize,
    relu: bool,
) {
    unsafe {
        let c_exp_chunks = c_exp / 16;
        let num_full_tiles = c_exp_chunks / TILED_CHUNKS;
        let residual_chunks = c_exp_chunks % TILED_CHUNKS;
        let zero = _mm512_setzero_ps();

        for iw in iw_start..iw_end {
            let src_off = iw * c_in;
            let dst_off = iw * c_exp;
            let dst_ptr = dst_row.as_mut_ptr().add(dst_off);

            // Full tiles: fixed 8-ZMM register-blocked body.
            for tile in 0..num_full_tiles {
                let chunk_base = tile * TILED_CHUNKS;
                let lane_off = chunk_base * 16;

                let mut acc: [__m512; TILED_CHUNKS] = [zero; TILED_CHUNKS];
                if let Some(b) = pw_bias {
                    let bp = b.as_ptr().add(lane_off);
                    for k in 0..TILED_CHUNKS {
                        acc[k] = _mm512_loadu_ps(bp.add(k * 16));
                    }
                }

                for ci in 0..c_in {
                    let x = _mm512_set1_ps(*src_row.as_ptr().add(src_off + ci));
                    let w_base = pw_weight.as_ptr().add(ci * c_exp + lane_off);
                    for k in 0..TILED_CHUNKS {
                        let w = _mm512_loadu_ps(w_base.add(k * 16));
                        acc[k] = _mm512_fmadd_ps(x, w, acc[k]);
                    }
                }

                if relu {
                    for k in 0..TILED_CHUNKS {
                        acc[k] = _mm512_max_ps(acc[k], zero);
                    }
                }
                for k in 0..TILED_CHUNKS {
                    _mm512_storeu_ps(dst_ptr.add(lane_off + k * 16), acc[k]);
                }
            }

            // Residual: 0..=7 chunks. LLVM may not unroll the dynamic
            // inner `0..residual_chunks` loop as cleanly as the fixed-8
            // case, but the total work is small enough that minor
            // inefficiency is acceptable (residual is a <12% tail at
            // worst on tracker shapes).
            if residual_chunks > 0 {
                let chunk_base = num_full_tiles * TILED_CHUNKS;
                let lane_off = chunk_base * 16;
                let mut acc: [__m512; TILED_CHUNKS] = [zero; TILED_CHUNKS];
                if let Some(b) = pw_bias {
                    let bp = b.as_ptr().add(lane_off);
                    for k in 0..residual_chunks {
                        acc[k] = _mm512_loadu_ps(bp.add(k * 16));
                    }
                }
                for ci in 0..c_in {
                    let x = _mm512_set1_ps(*src_row.as_ptr().add(src_off + ci));
                    let w_base = pw_weight.as_ptr().add(ci * c_exp + lane_off);
                    for k in 0..residual_chunks {
                        let w = _mm512_loadu_ps(w_base.add(k * 16));
                        acc[k] = _mm512_fmadd_ps(x, w, acc[k]);
                    }
                }
                if relu {
                    for k in 0..residual_chunks {
                        acc[k] = _mm512_max_ps(acc[k], zero);
                    }
                }
                for k in 0..residual_chunks {
                    _mm512_storeu_ps(dst_ptr.add(lane_off + k * 16), acc[k]);
                }
            }
        }
    }
}

/// Pack `pw_weight` from row-major `[c_in, c_exp]` into
/// `[num_tiles, c_in, TILED_CHUNKS * 16]` tile-major layout.
///
/// With the packed layout the inner CI loop in the tiled kernel reads
/// 128 bytes per CI step (stride 128 bytes) instead of the original
/// stride of `c_exp * 4` bytes (2688 bytes for C_exp=672).  The L1
/// hardware stream-prefetcher covers stride ≤ 512 bytes; the original
/// stride lands every read in L3.
pub(super) fn pack_pw_weight_tiled(
    weight: &[f32], // [c_in, c_exp] row-major
    c_in: usize,
    c_exp: usize,
    out: &mut Vec<f32>,
) {
    debug_assert!(c_exp.is_multiple_of(16));
    let tc = TILED_CHUNKS;
    let tc_lanes = tc * 16; // 128 for TILED_CHUNKS=8
    let c_exp_chunks = c_exp / 16;
    let num_full_tiles = c_exp_chunks / tc;
    let residual_chunks = c_exp_chunks % tc;
    out.resize(c_in * c_exp, 0.0);
    // Full tiles: packed[tile * c_in * tc_lanes + ci * tc_lanes .. +tc_lanes]
    for tile in 0..num_full_tiles {
        let src_lane_base = tile * tc_lanes;
        let dst_tile_base = tile * c_in * tc_lanes;
        for ci in 0..c_in {
            let src = &weight[ci * c_exp + src_lane_base..ci * c_exp + src_lane_base + tc_lanes];
            let dst =
                &mut out[dst_tile_base + ci * tc_lanes..dst_tile_base + ci * tc_lanes + tc_lanes];
            dst.copy_from_slice(src);
        }
    }
    // Residual tile: packed after full tiles, same ci-major layout.
    if residual_chunks > 0 {
        let res_lanes = residual_chunks * 16;
        let src_lane_base = num_full_tiles * tc_lanes;
        let dst_tile_base = num_full_tiles * c_in * tc_lanes;
        for ci in 0..c_in {
            let src = &weight[ci * c_exp + src_lane_base..ci * c_exp + src_lane_base + res_lanes];
            let dst = &mut out
                [dst_tile_base + ci * res_lanes..dst_tile_base + ci * res_lanes + res_lanes];
            dst.copy_from_slice(src);
        }
    }
}

/// Packed large-`c_exp` PW row with MLAS-style output-column reuse:
/// 4 OC-blocks × 6 OW columns = 24 ZMM accumulators.  This is the
/// high-channel analogue of `compute_pw_row_avx512_4x6_inner`, but
/// consumes the existing tile-major packed weights produced by
/// `pack_pw_weight_tiled`.
pub(super) fn compute_pw_row_avx512_tiled4x6_packed(
    src_row: &[f32],
    pw_weight: &[f32],
    pw_bias: Option<&[f32]>,
    dst_row: &mut [f32],
    iw_start: usize,
    iw_end: usize,
    c_in: usize,
    c_exp: usize,
    relu: bool,
) {
    #[allow(unsafe_code)]
    unsafe {
        compute_pw_row_avx512_tiled4x6_packed_inner(
            src_row, pw_weight, pw_bias, dst_row, iw_start, iw_end, c_in, c_exp, relu,
        )
    }
}

pub(super) fn compute_pw_row_avx512_tiled_packed(
    src_row: &[f32],
    pw_weight: &[f32],
    pw_bias: Option<&[f32]>,
    dst_row: &mut [f32],
    iw_start: usize,
    iw_end: usize,
    c_in: usize,
    c_exp: usize,
    relu: bool,
) {
    #[allow(unsafe_code)]
    unsafe {
        compute_pw_row_avx512_tiled_packed_inner(
            src_row, pw_weight, pw_bias, dst_row, iw_start, iw_end, c_in, c_exp, relu,
        )
    }
}

pub(super) fn tiled4x6_disabled() -> bool {
    static CACHED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| std::env::var_os("YSCV_FUSED_PW_DW_TILED4X6_OFF").is_some())
}

pub(super) fn fullrow_disabled() -> bool {
    static CACHED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| std::env::var_os("YSCV_FUSED_PW_DW_FULLROW_OFF").is_some())
}

/// **Stream-PROBE**: software prefetch of the next `ci`'s weight cacheline
/// inside `compute_pw_tile_avx512_inner`. ci-stride is `c_exp*4` B (1280 B
/// for c_exp=320) — beyond HW prefetcher. Manual `_mm_prefetch` hides the
/// L1 miss latency. Kill switch `YSCV_PW_PREFETCH_OFF=1` reverts.
pub(super) fn pw_prefetch_enabled() -> bool {
    static CACHED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| std::env::var_os("YSCV_PW_PREFETCH_OFF").is_none())
}

fn dw_interior_enabled() -> bool {
    static CACHED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| std::env::var_os("YSCV_FUSED_PW_DW_DW_INTERIOR").is_some())
}

/// DW 5×5 filter-tap reuse: process `DW5_REUSE_OW` interior output pixels per
/// inner block, loading each per-channel filter tap once and reusing it across
/// all of them. Drops the kernel from 2 loads/FMA (1-pixel tile) to
/// `(1 + OW)/OW ≈ 1.2` loads/FMA — the MLAS strategy. Kill switch
/// `YSCV_DW5_REUSE_OFF=1` falls back to the 1-pixel tiled body.
fn dw5_reuse_enabled() -> bool {
    static CACHED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| std::env::var_os("YSCV_DW5_REUSE_OFF").is_none())
}

/// Output pixels processed per reuse block. 6 mirrors MLAS's pixel tile and
/// keeps `DW5_REUSE_OW * DW5_REUSE_CHUNKS = 12` accumulator ZMMs plus 1 weight
/// + 1 input temp well within the 32-ZMM file.
const DW5_REUSE_OW: usize = 6;
/// Channel chunks (16-lane groups) accumulated together inside a reuse block.
const DW5_REUSE_CHUNKS: usize = 2;

fn interior_ow_range(
    in_w: usize,
    stride: usize,
    pad: usize,
    ow_start: usize,
    ow_end: usize,
) -> (usize, usize) {
    debug_assert!(stride > 0);
    if in_w + pad < 3 {
        return (ow_end, ow_end);
    }
    let start = pad.div_ceil(stride);
    let last = (in_w + pad - 3) / stride;
    (ow_start.max(start), ow_end.min(last + 1))
}

#[target_feature(enable = "avx512f")]
#[allow(unsafe_code, clippy::too_many_arguments)]
unsafe fn compute_pw_row_avx512_tiled4x6_packed_inner(
    src_row: &[f32],
    pw_weight: &[f32],
    pw_bias: Option<&[f32]>,
    dst_row: &mut [f32],
    iw_start: usize,
    iw_end: usize,
    c_in: usize,
    c_exp: usize,
    relu: bool,
) {
    unsafe {
        let chunks = c_exp / 16;
        let full_tiles = chunks / TILED_CHUNKS;
        let residual_chunks = chunks % TILED_CHUNKS;
        let tc_lanes = TILED_CHUNKS * 16;
        let zero = _mm512_setzero_ps();

        let width = iw_end - iw_start;
        let main_ow_tiles = width / TILED4X6_OW;
        let ow_tail_start = iw_start + main_ow_tiles * TILED4X6_OW;

        for ow_tile in 0..main_ow_tiles {
            let iw_base = iw_start + ow_tile * TILED4X6_OW;

            for tile in 0..full_tiles {
                let tile_base = tile * c_in * tc_lanes;
                for sub in 0..(TILED_CHUNKS / TILED4X6_CHUNKS) {
                    let lane_off = tile * tc_lanes + sub * TILED4X6_CHUNKS * 16;
                    let pack_sub_off = sub * TILED4X6_CHUNKS * 16;
                    let mut acc: [__m512; TILED4X6_CHUNKS * TILED4X6_OW] =
                        [zero; TILED4X6_CHUNKS * TILED4X6_OW];

                    if let Some(b) = pw_bias {
                        let bp = b.as_ptr().add(lane_off);
                        for oc in 0..TILED4X6_CHUNKS {
                            let bv = _mm512_loadu_ps(bp.add(oc * 16));
                            for ow in 0..TILED4X6_OW {
                                acc[oc * TILED4X6_OW + ow] = bv;
                            }
                        }
                    }

                    for ci in 0..c_in {
                        let src_ptr = src_row.as_ptr();
                        let mut x: [__m512; TILED4X6_OW] = [zero; TILED4X6_OW];
                        for ow in 0..TILED4X6_OW {
                            x[ow] = _mm512_set1_ps(*src_ptr.add((iw_base + ow) * c_in + ci));
                        }
                        let w_base = pw_weight
                            .as_ptr()
                            .add(tile_base + ci * tc_lanes + pack_sub_off);
                        for oc in 0..TILED4X6_CHUNKS {
                            let w = _mm512_loadu_ps(w_base.add(oc * 16));
                            for ow in 0..TILED4X6_OW {
                                let idx = oc * TILED4X6_OW + ow;
                                acc[idx] = _mm512_fmadd_ps(w, x[ow], acc[idx]);
                            }
                        }
                    }

                    for oc in 0..TILED4X6_CHUNKS {
                        for ow in 0..TILED4X6_OW {
                            let idx = oc * TILED4X6_OW + ow;
                            let v = if relu {
                                _mm512_max_ps(acc[idx], zero)
                            } else {
                                acc[idx]
                            };
                            let dst_ptr = dst_row
                                .as_mut_ptr()
                                .add((iw_base + ow) * c_exp + lane_off + oc * 16);
                            _mm512_storeu_ps(dst_ptr, v);
                        }
                    }
                }
            }

            if residual_chunks > 0 {
                let lane_off = full_tiles * tc_lanes;
                let res_lanes = residual_chunks * 16;
                let tile_base = full_tiles * c_in * tc_lanes;
                let mut acc: [__m512; TILED_CHUNKS * TILED4X6_OW] =
                    [zero; TILED_CHUNKS * TILED4X6_OW];

                if let Some(b) = pw_bias {
                    let bp = b.as_ptr().add(lane_off);
                    for oc in 0..residual_chunks {
                        let bv = _mm512_loadu_ps(bp.add(oc * 16));
                        for ow in 0..TILED4X6_OW {
                            acc[oc * TILED4X6_OW + ow] = bv;
                        }
                    }
                }

                for ci in 0..c_in {
                    let src_ptr = src_row.as_ptr();
                    let mut x: [__m512; TILED4X6_OW] = [zero; TILED4X6_OW];
                    for ow in 0..TILED4X6_OW {
                        x[ow] = _mm512_set1_ps(*src_ptr.add((iw_base + ow) * c_in + ci));
                    }
                    let w_base = pw_weight.as_ptr().add(tile_base + ci * res_lanes);
                    for oc in 0..residual_chunks {
                        let w = _mm512_loadu_ps(w_base.add(oc * 16));
                        for ow in 0..TILED4X6_OW {
                            let idx = oc * TILED4X6_OW + ow;
                            acc[idx] = _mm512_fmadd_ps(w, x[ow], acc[idx]);
                        }
                    }
                }

                for oc in 0..residual_chunks {
                    for ow in 0..TILED4X6_OW {
                        let idx = oc * TILED4X6_OW + ow;
                        let v = if relu {
                            _mm512_max_ps(acc[idx], zero)
                        } else {
                            acc[idx]
                        };
                        let dst_ptr = dst_row
                            .as_mut_ptr()
                            .add((iw_base + ow) * c_exp + lane_off + oc * 16);
                        _mm512_storeu_ps(dst_ptr, v);
                    }
                }
            }
        }

        if ow_tail_start < iw_end {
            compute_pw_row_avx512_tiled_packed_inner(
                src_row,
                pw_weight,
                pw_bias,
                dst_row,
                ow_tail_start,
                iw_end,
                c_in,
                c_exp,
                relu,
            );
        }
    }
}

#[target_feature(enable = "avx512f")]
#[allow(unsafe_code, clippy::too_many_arguments)]
unsafe fn compute_pw_row_avx512_tiled_packed_inner(
    src_row: &[f32],
    pw_weight: &[f32],
    pw_bias: Option<&[f32]>,
    dst_row: &mut [f32],
    iw_start: usize,
    iw_end: usize,
    c_in: usize,
    c_exp: usize,
    relu: bool,
) {
    unsafe {
        let c_exp_chunks = c_exp / 16;
        let num_full_tiles = c_exp_chunks / TILED_CHUNKS;
        let residual_chunks = c_exp_chunks % TILED_CHUNKS;
        let tc_lanes = TILED_CHUNKS * 16; // 128
        let zero = _mm512_setzero_ps();

        for iw in iw_start..iw_end {
            let src_off = iw * c_in;
            let dst_off = iw * c_exp;
            let dst_ptr = dst_row.as_mut_ptr().add(dst_off);

            // Full tiles — packed[tile * c_in * tc_lanes + ci * tc_lanes + k*16]
            for tile in 0..num_full_tiles {
                let lane_off = tile * tc_lanes; // offset in c_exp output dimension
                let tile_base = tile * c_in * tc_lanes;

                let mut acc: [__m512; TILED_CHUNKS] = [zero; TILED_CHUNKS];
                if let Some(b) = pw_bias {
                    let bp = b.as_ptr().add(lane_off);
                    for k in 0..TILED_CHUNKS {
                        acc[k] = _mm512_loadu_ps(bp.add(k * 16));
                    }
                }

                for ci in 0..c_in {
                    let x = _mm512_set1_ps(*src_row.as_ptr().add(src_off + ci));
                    // Sequential: tile_base + ci*tc_lanes, then +k*16 within block
                    let w_base = pw_weight.as_ptr().add(tile_base + ci * tc_lanes);
                    for k in 0..TILED_CHUNKS {
                        let w = _mm512_loadu_ps(w_base.add(k * 16));
                        acc[k] = _mm512_fmadd_ps(x, w, acc[k]);
                    }
                }

                if relu {
                    for k in 0..TILED_CHUNKS {
                        acc[k] = _mm512_max_ps(acc[k], zero);
                    }
                }
                for k in 0..TILED_CHUNKS {
                    _mm512_storeu_ps(dst_ptr.add(lane_off + k * 16), acc[k]);
                }
            }

            // Residual tile
            if residual_chunks > 0 {
                let lane_off = num_full_tiles * tc_lanes;
                let res_lanes = residual_chunks * 16;
                let tile_base = num_full_tiles * c_in * tc_lanes;
                let mut acc: [__m512; TILED_CHUNKS] = [zero; TILED_CHUNKS];
                if let Some(b) = pw_bias {
                    let bp = b.as_ptr().add(lane_off);
                    for k in 0..residual_chunks {
                        acc[k] = _mm512_loadu_ps(bp.add(k * 16));
                    }
                }
                for ci in 0..c_in {
                    let x = _mm512_set1_ps(*src_row.as_ptr().add(src_off + ci));
                    let w_base = pw_weight.as_ptr().add(tile_base + ci * res_lanes);
                    for k in 0..residual_chunks {
                        let w = _mm512_loadu_ps(w_base.add(k * 16));
                        acc[k] = _mm512_fmadd_ps(x, w, acc[k]);
                    }
                }
                if relu {
                    for k in 0..residual_chunks {
                        acc[k] = _mm512_max_ps(acc[k], zero);
                    }
                }
                for k in 0..residual_chunks {
                    _mm512_storeu_ps(dst_ptr.add(lane_off + k * 16), acc[k]);
                }
            }
        }
    }
}

/// Packed PW row kernel that reads each 16-float weight block ONCE
/// and applies it to all `P` input pixels simultaneously.
///
/// The tiled4x6 kernel's outermost `ow_tile` loop causes every
/// (tile, oc_sub, ci) weight to be loaded `in_w / TILED4X6_OW` = 4×
/// for the tracker's in_w=16 case.  This kernel eliminates the
/// redundancy: weight is loaded once per (tile, oc_sub, ci) and
/// FMA'd across all P pixel accumulators while live in a ZMM register.
///
/// ZMM budget: P acc + 1 w + 1 zero + ≤2 scratch = P+4 ≤ 32 → P ≤ 28.
/// Hot case P=16 → 19 ZMM.
pub(super) fn compute_pw_row_avx512_fullrow_packed(
    src_row: &[f32],
    pw_weight: &[f32],
    pw_bias: Option<&[f32]>,
    dst_row: &mut [f32],
    iw_start: usize,
    iw_end: usize,
    c_in: usize,
    c_exp: usize,
    relu: bool,
) {
    let width = iw_end - iw_start;
    #[allow(unsafe_code)]
    unsafe {
        match width {
            16 => compute_pw_row_avx512_fullrow_packed_inner::<16>(
                src_row, pw_weight, pw_bias, dst_row, iw_start, c_in, c_exp, relu,
            ),
            8 => compute_pw_row_avx512_fullrow_packed_inner::<8>(
                src_row, pw_weight, pw_bias, dst_row, iw_start, c_in, c_exp, relu,
            ),
            4 => compute_pw_row_avx512_fullrow_packed_inner::<4>(
                src_row, pw_weight, pw_bias, dst_row, iw_start, c_in, c_exp, relu,
            ),
            _ => compute_pw_row_avx512_tiled4x6_packed_inner(
                src_row, pw_weight, pw_bias, dst_row, iw_start, iw_end, c_in, c_exp, relu,
            ),
        }
    }
}

#[target_feature(enable = "avx512f")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn, clippy::too_many_arguments)]
unsafe fn compute_pw_row_avx512_fullrow_packed_inner<const P: usize>(
    src_row: &[f32],
    pw_weight: &[f32],
    pw_bias: Option<&[f32]>,
    dst_row: &mut [f32],
    iw_start: usize,
    c_in: usize,
    c_exp: usize,
    relu: bool,
) {
    debug_assert!(c_exp.is_multiple_of(16));
    let chunks = c_exp / 16;
    let full_tiles = chunks / TILED_CHUNKS;
    let residual_chunks = chunks % TILED_CHUNKS;
    let tc_lanes = TILED_CHUNKS * 16; // 128
    let zero = _mm512_setzero_ps();
    let sp = src_row.as_ptr().add(iw_start * c_in);

    for tile in 0..full_tiles {
        let tile_w_base = tile * c_in * tc_lanes;
        for oc_sub in 0..TILED_CHUNKS {
            let lane_off = tile * tc_lanes + oc_sub * 16;
            let w_base = tile_w_base + oc_sub * 16;

            let mut acc = [zero; P];
            if let Some(b) = pw_bias {
                let bv = _mm512_loadu_ps(b.as_ptr().add(lane_off));
                for p in 0..P {
                    acc[p] = bv;
                }
            }

            for ci in 0..c_in {
                // Weight is live in a ZMM register for the entire P-pixel loop.
                let w = _mm512_loadu_ps(pw_weight.as_ptr().add(w_base + ci * tc_lanes));
                for p in 0..P {
                    acc[p] = _mm512_fmadd_ps(_mm512_set1_ps(*sp.add(p * c_in + ci)), w, acc[p]);
                }
            }

            for p in 0..P {
                let v = if relu {
                    _mm512_max_ps(acc[p], zero)
                } else {
                    acc[p]
                };
                _mm512_storeu_ps(
                    dst_row.as_mut_ptr().add((iw_start + p) * c_exp + lane_off),
                    v,
                );
            }
        }
    }

    if residual_chunks > 0 {
        let res_lanes = residual_chunks * 16;
        let tile_w_base = full_tiles * c_in * tc_lanes;
        for oc_sub in 0..residual_chunks {
            let lane_off = full_tiles * tc_lanes + oc_sub * 16;
            let w_base = tile_w_base + oc_sub * 16;

            let mut acc = [zero; P];
            if let Some(b) = pw_bias {
                let bv = _mm512_loadu_ps(b.as_ptr().add(lane_off));
                for p in 0..P {
                    acc[p] = bv;
                }
            }

            for ci in 0..c_in {
                // Residual tile: ci stride is res_lanes, not tc_lanes.
                let w = _mm512_loadu_ps(pw_weight.as_ptr().add(w_base + ci * res_lanes));
                for p in 0..P {
                    acc[p] = _mm512_fmadd_ps(_mm512_set1_ps(*sp.add(p * c_in + ci)), w, acc[p]);
                }
            }

            for p in 0..P {
                let v = if relu {
                    _mm512_max_ps(acc[p], zero)
                } else {
                    acc[p]
                };
                _mm512_storeu_ps(
                    dst_row.as_mut_ptr().add((iw_start + p) * c_exp + lane_off),
                    v,
                );
            }
        }
    }
}

/// AVX-512 tiled DW 3×3 row for `c_exp > 256`. Mirror of the
/// PW tiled variant: 8-chunk tile with register-blocked
/// accumulators across 9 taps.
#[allow(clippy::too_many_arguments)]
pub(super) fn compute_dw_row_avx512_tiled(
    row0: Option<&[f32]>,
    row1: Option<&[f32]>,
    row2: Option<&[f32]>,
    dw_weight: &[f32],
    dw_bias: Option<&[f32]>,
    out_row: &mut [f32],
    in_w: usize,
    ow_start: usize,
    ow_end: usize,
    c_exp: usize,
    stride: usize,
    pad: usize,
    relu: bool,
) {
    // SAFETY: dispatcher confirmed `avx512f`.
    #[allow(unsafe_code)]
    unsafe {
        compute_dw_row_avx512_tiled_inner(
            row0, row1, row2, dw_weight, dw_bias, out_row, in_w, ow_start, ow_end, c_exp, stride,
            pad, relu,
        )
    }
}

#[target_feature(enable = "avx512f")]
#[allow(unsafe_code, clippy::too_many_arguments)]
unsafe fn compute_dw_row_avx512_tiled_inner(
    row0: Option<&[f32]>,
    row1: Option<&[f32]>,
    row2: Option<&[f32]>,
    dw_weight: &[f32],
    dw_bias: Option<&[f32]>,
    out_row: &mut [f32],
    in_w: usize,
    ow_start: usize,
    ow_end: usize,
    c_exp: usize,
    stride: usize,
    pad: usize,
    relu: bool,
) {
    unsafe {
        let c_exp_chunks = c_exp / 16;
        let num_full_tiles = c_exp_chunks / TILED_CHUNKS;
        let residual_chunks = c_exp_chunks % TILED_CHUNKS;
        let zero = _mm512_setzero_ps();
        let w_ky_stride = 3 * c_exp;
        let w_kx_stride = c_exp;
        let rows = [row0, row1, row2];
        let can_fast_interior = dw_interior_enabled()
            && pad == 1
            && (stride == 1 || stride == 2)
            && in_w >= 3
            && row0.is_some()
            && row1.is_some()
            && row2.is_some();

        if can_fast_interior && let (Some(r0), Some(r1), Some(r2)) = (row0, row1, row2) {
            let (interior_start, interior_end) =
                interior_ow_range(in_w, stride, pad, ow_start, ow_end);
            if interior_start < interior_end {
                compute_dw_row_avx512_tiled_interior_inner(
                    r0,
                    r1,
                    r2,
                    dw_weight,
                    dw_bias,
                    out_row,
                    interior_start,
                    interior_end,
                    c_exp,
                    stride,
                    pad,
                    relu,
                );
            }
        }

        for ow in ow_start..ow_end {
            let iw0 = (ow as i32) * (stride as i32) - (pad as i32);
            if can_fast_interior && iw0 >= 0 && (iw0 as usize) + 2 < in_w {
                continue;
            }
            let dst_off = ow * c_exp;
            let dst_ptr = out_row.as_mut_ptr().add(dst_off);

            // Full tiles.
            for tile in 0..num_full_tiles {
                let chunk_base = tile * TILED_CHUNKS;
                let lane_off = chunk_base * 16;

                let mut acc: [__m512; TILED_CHUNKS] = [zero; TILED_CHUNKS];
                if let Some(b) = dw_bias {
                    let bp = b.as_ptr().add(lane_off);
                    for k in 0..TILED_CHUNKS {
                        acc[k] = _mm512_loadu_ps(bp.add(k * 16));
                    }
                }

                for ky in 0..3usize {
                    let pw_row = match rows[ky] {
                        Some(r) => r,
                        None => continue,
                    };
                    for kx in 0..3usize {
                        let iw = iw0 + kx as i32;
                        if iw < 0 || (iw as usize) >= in_w {
                            continue;
                        }
                        let iw_u = iw as usize;
                        let pw_ptr = pw_row.as_ptr().add(iw_u * c_exp + lane_off);
                        let w_ptr = dw_weight
                            .as_ptr()
                            .add(ky * w_ky_stride + kx * w_kx_stride + lane_off);
                        for k in 0..TILED_CHUNKS {
                            let x = _mm512_loadu_ps(pw_ptr.add(k * 16));
                            let w = _mm512_loadu_ps(w_ptr.add(k * 16));
                            acc[k] = _mm512_fmadd_ps(x, w, acc[k]);
                        }
                    }
                }

                if relu {
                    for k in 0..TILED_CHUNKS {
                        acc[k] = _mm512_max_ps(acc[k], zero);
                    }
                }
                for k in 0..TILED_CHUNKS {
                    _mm512_storeu_ps(dst_ptr.add(lane_off + k * 16), acc[k]);
                }
            }

            // Residual.
            if residual_chunks > 0 {
                let chunk_base = num_full_tiles * TILED_CHUNKS;
                let lane_off = chunk_base * 16;
                let mut acc: [__m512; TILED_CHUNKS] = [zero; TILED_CHUNKS];
                if let Some(b) = dw_bias {
                    let bp = b.as_ptr().add(lane_off);
                    for k in 0..residual_chunks {
                        acc[k] = _mm512_loadu_ps(bp.add(k * 16));
                    }
                }
                for ky in 0..3usize {
                    let pw_row = match rows[ky] {
                        Some(r) => r,
                        None => continue,
                    };
                    for kx in 0..3usize {
                        let iw = iw0 + kx as i32;
                        if iw < 0 || (iw as usize) >= in_w {
                            continue;
                        }
                        let iw_u = iw as usize;
                        let pw_ptr = pw_row.as_ptr().add(iw_u * c_exp + lane_off);
                        let w_ptr = dw_weight
                            .as_ptr()
                            .add(ky * w_ky_stride + kx * w_kx_stride + lane_off);
                        for k in 0..residual_chunks {
                            let x = _mm512_loadu_ps(pw_ptr.add(k * 16));
                            let w = _mm512_loadu_ps(w_ptr.add(k * 16));
                            acc[k] = _mm512_fmadd_ps(x, w, acc[k]);
                        }
                    }
                }
                if relu {
                    for k in 0..residual_chunks {
                        acc[k] = _mm512_max_ps(acc[k], zero);
                    }
                }
                for k in 0..residual_chunks {
                    _mm512_storeu_ps(dst_ptr.add(lane_off + k * 16), acc[k]);
                }
            }
        }
    }
}

#[target_feature(enable = "avx512f")]
#[allow(unsafe_code)]
unsafe fn compute_dw_row_avx512_tiled_interior_inner(
    row0: &[f32],
    row1: &[f32],
    row2: &[f32],
    dw_weight: &[f32],
    dw_bias: Option<&[f32]>,
    out_row: &mut [f32],
    ow_start: usize,
    ow_end: usize,
    c_exp: usize,
    stride: usize,
    pad: usize,
    relu: bool,
) {
    unsafe {
        let c_exp_chunks = c_exp / 16;
        let num_full_tiles = c_exp_chunks / TILED_CHUNKS;
        let residual_chunks = c_exp_chunks % TILED_CHUNKS;
        let zero = _mm512_setzero_ps();
        let w_ky_stride = 3 * c_exp;
        let w_kx_stride = c_exp;
        let rows = [row0, row1, row2];

        for ow in ow_start..ow_end {
            let iw0 = ow * stride - pad;
            let dst_ptr = out_row.as_mut_ptr().add(ow * c_exp);

            for tile in 0..num_full_tiles {
                let chunk_base = tile * TILED_CHUNKS;
                let lane_off = chunk_base * 16;

                let mut acc: [__m512; TILED_CHUNKS] = [zero; TILED_CHUNKS];
                if let Some(b) = dw_bias {
                    let bp = b.as_ptr().add(lane_off);
                    for k in 0..TILED_CHUNKS {
                        acc[k] = _mm512_loadu_ps(bp.add(k * 16));
                    }
                }

                for ky in 0..3usize {
                    let pw_row = rows[ky];
                    for kx in 0..3usize {
                        let pw_ptr = pw_row.as_ptr().add((iw0 + kx) * c_exp + lane_off);
                        let w_ptr = dw_weight
                            .as_ptr()
                            .add(ky * w_ky_stride + kx * w_kx_stride + lane_off);
                        for k in 0..TILED_CHUNKS {
                            let x = _mm512_loadu_ps(pw_ptr.add(k * 16));
                            let w = _mm512_loadu_ps(w_ptr.add(k * 16));
                            acc[k] = _mm512_fmadd_ps(x, w, acc[k]);
                        }
                    }
                }

                if relu {
                    for k in 0..TILED_CHUNKS {
                        acc[k] = _mm512_max_ps(acc[k], zero);
                    }
                }
                for k in 0..TILED_CHUNKS {
                    _mm512_storeu_ps(dst_ptr.add(lane_off + k * 16), acc[k]);
                }
            }

            if residual_chunks > 0 {
                let chunk_base = num_full_tiles * TILED_CHUNKS;
                let lane_off = chunk_base * 16;
                let mut acc: [__m512; TILED_CHUNKS] = [zero; TILED_CHUNKS];
                if let Some(b) = dw_bias {
                    let bp = b.as_ptr().add(lane_off);
                    for k in 0..residual_chunks {
                        acc[k] = _mm512_loadu_ps(bp.add(k * 16));
                    }
                }
                for ky in 0..3usize {
                    let pw_row = rows[ky];
                    for kx in 0..3usize {
                        let pw_ptr = pw_row.as_ptr().add((iw0 + kx) * c_exp + lane_off);
                        let w_ptr = dw_weight
                            .as_ptr()
                            .add(ky * w_ky_stride + kx * w_kx_stride + lane_off);
                        for k in 0..residual_chunks {
                            let x = _mm512_loadu_ps(pw_ptr.add(k * 16));
                            let w = _mm512_loadu_ps(w_ptr.add(k * 16));
                            acc[k] = _mm512_fmadd_ps(x, w, acc[k]);
                        }
                    }
                }
                if relu {
                    for k in 0..residual_chunks {
                        acc[k] = _mm512_max_ps(acc[k], zero);
                    }
                }
                for k in 0..residual_chunks {
                    _mm512_storeu_ps(dst_ptr.add(lane_off + k * 16), acc[k]);
                }
            }
        }
    }
}

/// AVX-512 DW 3×3 row: same register-blocking idea. 9 taps
/// accumulate into the same `c_exp / 16` ZMM accumulators.
#[allow(clippy::too_many_arguments)]
pub(super) fn compute_dw_row_avx512(
    row0: Option<&[f32]>,
    row1: Option<&[f32]>,
    row2: Option<&[f32]>,
    dw_weight: &[f32],
    dw_bias: Option<&[f32]>,
    out_row: &mut [f32],
    in_w: usize,
    ow_start: usize,
    ow_end: usize,
    c_exp: usize,
    stride: usize,
    pad: usize,
    relu: bool,
) {
    // SAFETY: dispatcher confirmed `avx512f` and chunk budget.
    #[allow(unsafe_code)]
    unsafe {
        compute_dw_row_avx512_inner(
            row0, row1, row2, dw_weight, dw_bias, out_row, in_w, ow_start, ow_end, c_exp, stride,
            pad, relu,
        )
    }
}

#[target_feature(enable = "avx512f")]
#[allow(unsafe_code, clippy::too_many_arguments)]
unsafe fn compute_dw_row_avx512_inner(
    row0: Option<&[f32]>,
    row1: Option<&[f32]>,
    row2: Option<&[f32]>,
    dw_weight: &[f32],
    dw_bias: Option<&[f32]>,
    out_row: &mut [f32],
    in_w: usize,
    ow_start: usize,
    ow_end: usize,
    c_exp: usize,
    stride: usize,
    pad: usize,
    relu: bool,
) {
    unsafe {
        let c_exp_chunks = c_exp / 16;
        debug_assert!(c_exp_chunks <= super::AVX512_REG_MAX_CHUNKS);
        let zero = _mm512_setzero_ps();
        let w_ky_stride = 3 * c_exp;
        let w_kx_stride = c_exp;
        let rows = [row0, row1, row2];
        let full_rows =
            if dw_interior_enabled() && pad == 1 && (stride == 1 || stride == 2) && in_w >= 3 {
                match (row0, row1, row2) {
                    (Some(r0), Some(r1), Some(r2)) => Some([r0, r1, r2]),
                    _ => None,
                }
            } else {
                None
            };

        let mut accs: [__m512; super::AVX512_REG_MAX_CHUNKS] =
            [_mm512_setzero_ps(); super::AVX512_REG_MAX_CHUNKS];

        for ow in ow_start..ow_end {
            let iw0 = (ow as i32) * (stride as i32) - (pad as i32);
            let dst_off = ow * c_exp;

            // Seed accs from bias or zero.
            if let Some(b) = dw_bias {
                let bp = b.as_ptr();
                for ck in 0..c_exp_chunks {
                    accs[ck] = _mm512_loadu_ps(bp.add(ck * 16));
                }
            } else {
                for ck in 0..c_exp_chunks {
                    accs[ck] = zero;
                }
            }

            // 9 taps. Interior columns have all rows/columns valid,
            // so skip OOB checks in-place instead of doing a separate
            // interior pass and then a second border/skip pass.
            if iw0 >= 0
                && (iw0 as usize) + 2 < in_w
                && let Some(full_rows) = full_rows
            {
                let iw_base = iw0 as usize;
                for ky in 0..3usize {
                    let pw_row = full_rows[ky];
                    for kx in 0..3usize {
                        let pw_ptr = pw_row.as_ptr().add((iw_base + kx) * c_exp);
                        let w_ptr = dw_weight.as_ptr().add(ky * w_ky_stride + kx * w_kx_stride);
                        for ck in 0..c_exp_chunks {
                            let x = _mm512_loadu_ps(pw_ptr.add(ck * 16));
                            let w = _mm512_loadu_ps(w_ptr.add(ck * 16));
                            accs[ck] = _mm512_fmadd_ps(x, w, accs[ck]);
                        }
                    }
                }
            } else {
                for ky in 0..3usize {
                    let pw_row = match rows[ky] {
                        Some(r) => r,
                        None => continue,
                    };
                    for kx in 0..3usize {
                        let iw = iw0 + kx as i32;
                        if iw < 0 || (iw as usize) >= in_w {
                            continue;
                        }
                        let iw_u = iw as usize;
                        let pw_ptr = pw_row.as_ptr().add(iw_u * c_exp);
                        let w_ptr = dw_weight.as_ptr().add(ky * w_ky_stride + kx * w_kx_stride);
                        for ck in 0..c_exp_chunks {
                            let x = _mm512_loadu_ps(pw_ptr.add(ck * 16));
                            let w = _mm512_loadu_ps(w_ptr.add(ck * 16));
                            accs[ck] = _mm512_fmadd_ps(x, w, accs[ck]);
                        }
                    }
                }
            }

            // Optional Relu + store.
            let dst_ptr = out_row.as_mut_ptr().add(dst_off);
            if relu {
                for ck in 0..c_exp_chunks {
                    let v = _mm512_max_ps(accs[ck], zero);
                    _mm512_storeu_ps(dst_ptr.add(ck * 16), v);
                }
            } else {
                for ck in 0..c_exp_chunks {
                    _mm512_storeu_ps(dst_ptr.add(ck * 16), accs[ck]);
                }
            }
        }
    }
}

pub(super) fn compute_dw5_row_avx512(ctx: Dw5RowCtx<'_, '_>) {
    #[allow(unsafe_code)]
    unsafe {
        compute_dw5_row_avx512_inner(ctx)
    }
}

#[target_feature(enable = "avx512f")]
#[allow(unsafe_code)]
unsafe fn compute_dw5_row_avx512_inner(ctx: Dw5RowCtx<'_, '_>) {
    unsafe {
        let Dw5RowCtx {
            rows,
            dw_weight,
            dw_bias,
            out_row,
            in_w,
            ow_start,
            ow_end,
            c_exp,
            stride,
            pad,
            relu,
        } = ctx;
        let c_exp_chunks = c_exp / 16;
        let full_tiles = c_exp_chunks / TILED_CHUNKS;
        let residual_chunks = c_exp_chunks % TILED_CHUNKS;
        let zero = _mm512_setzero_ps();
        let w_ky_stride = 5 * c_exp;
        let w_kx_stride = c_exp;

        // Resolve 5 input-row pointers once (None → null). For interior
        // output rows all are non-null and the interior fast path runs
        // branch-free — see the matching comment in
        // `compute_dw5_tile_avx512_inner`.
        let row_ptrs: [*const f32; 5] = std::array::from_fn(|ky| match rows[ky] {
            Some(r) => r.as_ptr(),
            None => std::ptr::null(),
        });
        let all_rows = row_ptrs.iter().all(|p| !p.is_null());

        macro_rules! row_tile_body {
            ($ow:expr, $nchunks:expr, $lane_off:expr, $interior:expr) => {{
                let ow_v = $ow;
                let lane_off = $lane_off;
                let mut acc: [__m512; TILED_CHUNKS] = [zero; TILED_CHUNKS];
                if let Some(b) = dw_bias {
                    let bp = b.as_ptr().add(lane_off);
                    for k in 0..$nchunks {
                        acc[k] = _mm512_loadu_ps(bp.add(k * 16));
                    }
                }
                let iw0 = (ow_v as i32) * (stride as i32) - (pad as i32);
                for ky in 0..5usize {
                    let pw_row = row_ptrs[ky];
                    if pw_row.is_null() {
                        continue;
                    }
                    for kx in 0..5usize {
                        let iw = iw0 + kx as i32;
                        if !$interior && (iw < 0 || (iw as usize) >= in_w) {
                            continue;
                        }
                        let pw_ptr = pw_row.add(iw as usize * c_exp + lane_off);
                        let w_ptr = dw_weight
                            .as_ptr()
                            .add(ky * w_ky_stride + kx * w_kx_stride + lane_off);
                        for k in 0..$nchunks {
                            let x = _mm512_loadu_ps(pw_ptr.add(k * 16));
                            let w = _mm512_loadu_ps(w_ptr.add(k * 16));
                            acc[k] = _mm512_fmadd_ps(x, w, acc[k]);
                        }
                    }
                }
                if relu {
                    for k in 0..$nchunks {
                        acc[k] = _mm512_max_ps(acc[k], zero);
                    }
                }
                let dst_ptr = out_row.as_mut_ptr().add(ow_v * c_exp);
                for k in 0..$nchunks {
                    _mm512_storeu_ps(dst_ptr.add(lane_off + k * 16), acc[k]);
                }
            }};
        }

        // One pixel of the per-pixel path (full tiles + residual), interior or
        // border. Used for border pixels and the interior tail that does not
        // fill a reuse block.
        macro_rules! row_pixel {
            ($ow:expr, $interior:expr) => {{
                for tile in 0..full_tiles {
                    row_tile_body!($ow, TILED_CHUNKS, tile * TILED_CHUNKS * 16, $interior);
                }
                if residual_chunks > 0 {
                    row_tile_body!(
                        $ow,
                        residual_chunks,
                        full_tiles * TILED_CHUNKS * 16,
                        $interior
                    );
                }
            }};
        }

        // Contiguous interior range: every pixel has all 5 rows present and all
        // 5 kx taps in [0, in_w). `iw0 = ow*stride - pad` is monotone in `ow`,
        // so the interior pixels form one contiguous span.
        let (int_lo, int_hi) = if all_rows {
            dw5_interior_ow_range(in_w, stride, pad, ow_start, ow_end)
        } else {
            (ow_end, ow_end)
        };

        // Border pixels before the interior span.
        for ow in ow_start..int_lo {
            row_pixel!(ow, false);
        }

        if dw5_reuse_enabled() {
            // Interior: process DW5_REUSE_OW pixels per block, loading each
            // filter tap once per channel-chunk and reusing it across the
            // block's pixels. Per-pixel tap order (ky→kx) matches the per-pixel
            // body, so each accumulator's FMA chain is bit-identical.
            let ow_chunks = c_exp_chunks / DW5_REUSE_CHUNKS;
            let ow_res_chunks = c_exp_chunks % DW5_REUSE_CHUNKS;
            let mut ow = int_lo;
            while ow + DW5_REUSE_OW <= int_hi {
                for ct in 0..ow_chunks {
                    dw5_reuse_block(
                        &row_ptrs,
                        dw_weight,
                        dw_bias,
                        out_row,
                        ow,
                        ct * DW5_REUSE_CHUNKS * 16,
                        DW5_REUSE_CHUNKS,
                        c_exp,
                        w_ky_stride,
                        w_kx_stride,
                        stride,
                        pad,
                        relu,
                        zero,
                    );
                }
                if ow_res_chunks > 0 {
                    dw5_reuse_block(
                        &row_ptrs,
                        dw_weight,
                        dw_bias,
                        out_row,
                        ow,
                        ow_chunks * DW5_REUSE_CHUNKS * 16,
                        ow_res_chunks,
                        c_exp,
                        w_ky_stride,
                        w_kx_stride,
                        stride,
                        pad,
                        relu,
                        zero,
                    );
                }
                ow += DW5_REUSE_OW;
            }
            // Interior tail (fewer than DW5_REUSE_OW pixels).
            for ow in ow..int_hi {
                row_pixel!(ow, true);
            }
        } else {
            for ow in int_lo..int_hi {
                row_pixel!(ow, true);
            }
        }

        // Border pixels after the interior span.
        for ow in int_hi..ow_end {
            row_pixel!(ow, false);
        }
    }
}

/// Contiguous interior output-pixel range for a 5×5 DW: every `ow` in
/// `[lo, hi)` has `iw0 = ow*stride - pad >= 0` and `iw0 + 4 < in_w`, so all 5
/// horizontal taps land in `[0, in_w)`. Mirrors `interior_ow_range` (3×3) with
/// kernel size 5.
fn dw5_interior_ow_range(
    in_w: usize,
    stride: usize,
    pad: usize,
    ow_start: usize,
    ow_end: usize,
) -> (usize, usize) {
    debug_assert!(stride > 0);
    if in_w < 5 {
        return (ow_end, ow_end);
    }
    // iw0 >= 0  ⇒  ow >= ceil(pad / stride)
    let lo = pad.div_ceil(stride);
    // iw0 + 4 <= in_w - 1  ⇒  ow*stride <= in_w + pad - 5
    let last = (in_w + pad - 5) / stride;
    // Clamp into [ow_start, ow_end] and keep lo <= hi so the three caller
    // loops (border-before, interior, border-after) partition the output row
    // exactly once. An inverted or out-of-range span would double-process or
    // overrun pixels.
    let int_lo = ow_start.max(lo).min(ow_end);
    let int_hi = ow_end.min(last + 1).max(int_lo);
    (int_lo, int_hi)
}

/// One interior reuse block: `DW5_REUSE_OW` output pixels starting at `ow0`,
/// across `nchunks` channel chunks at lane offset `lane_off`. For each of the
/// 25 taps the weight chunk is loaded once and reused across all
/// `DW5_REUSE_OW` pixels; each pixel loads only its own shifted input chunk.
#[target_feature(enable = "avx512f")]
#[allow(unsafe_code, clippy::too_many_arguments)]
unsafe fn dw5_reuse_block(
    row_ptrs: &[*const f32; 5],
    dw_weight: &[f32],
    dw_bias: Option<&[f32]>,
    out_row: &mut [f32],
    ow0: usize,
    lane_off: usize,
    nchunks: usize,
    c_exp: usize,
    w_ky_stride: usize,
    w_kx_stride: usize,
    stride: usize,
    pad: usize,
    relu: bool,
    zero: __m512,
) {
    unsafe {
        // acc[p * nchunks + k]: pixel p, channel chunk k.
        let mut acc: [__m512; DW5_REUSE_OW * DW5_REUSE_CHUNKS] =
            [zero; DW5_REUSE_OW * DW5_REUSE_CHUNKS];
        if let Some(b) = dw_bias {
            let bp = b.as_ptr().add(lane_off);
            for k in 0..nchunks {
                let bv = _mm512_loadu_ps(bp.add(k * 16));
                for p in 0..DW5_REUSE_OW {
                    acc[p * DW5_REUSE_CHUNKS + k] = bv;
                }
            }
        }

        // Leftmost input column of pixel 0; pixel p starts at iw0 + p*stride.
        let iw0_base = (ow0 as i32) * (stride as i32) - (pad as i32);
        for ky in 0..5usize {
            let pw_row = row_ptrs[ky];
            for kx in 0..5usize {
                let w_ptr = dw_weight
                    .as_ptr()
                    .add(ky * w_ky_stride + kx * w_kx_stride + lane_off);
                for k in 0..nchunks {
                    let w = _mm512_loadu_ps(w_ptr.add(k * 16));
                    for p in 0..DW5_REUSE_OW {
                        let iw = iw0_base + (p * stride) as i32 + kx as i32;
                        let pw_ptr = pw_row.add(iw as usize * c_exp + lane_off + k * 16);
                        let x = _mm512_loadu_ps(pw_ptr);
                        let idx = p * DW5_REUSE_CHUNKS + k;
                        acc[idx] = _mm512_fmadd_ps(x, w, acc[idx]);
                    }
                }
            }
        }

        for p in 0..DW5_REUSE_OW {
            let dst_ptr = out_row.as_mut_ptr().add((ow0 + p) * c_exp + lane_off);
            for k in 0..nchunks {
                let idx = p * DW5_REUSE_CHUNKS + k;
                let v = if relu {
                    _mm512_max_ps(acc[idx], zero)
                } else {
                    acc[idx]
                };
                _mm512_storeu_ps(dst_ptr.add(k * 16), v);
            }
        }
    }
}

pub(super) fn compute_pw_tile_avx512(ctx: PwTileCtx<'_, '_>) {
    #[allow(unsafe_code)]
    unsafe {
        compute_pw_tile_avx512_inner(ctx)
    }
}

#[target_feature(enable = "avx512f")]
#[allow(unsafe_code)]
unsafe fn compute_pw_tile_avx512_inner(ctx: PwTileCtx<'_, '_>) {
    unsafe {
        let PwTileCtx {
            src_row,
            pw_weight,
            pw_bias,
            dst_row,
            iw_start,
            iw_end,
            c_in,
            c_exp,
            oc_start,
            c_tile,
            relu,
        } = ctx;
        let chunks = c_tile / 16;
        let full_tiles = chunks / TILED_CHUNKS;
        let residual_chunks = chunks % TILED_CHUNKS;
        let zero = _mm512_setzero_ps();
        let mut iw = iw_start;
        while iw + 1 < iw_end {
            let dst_ptr0 = dst_row.as_mut_ptr().add(iw * c_tile);
            let dst_ptr1 = dst_row.as_mut_ptr().add((iw + 1) * c_tile);
            for tile in 0..full_tiles {
                let tile_off = tile * TILED_CHUNKS * 16;
                let mut acc0: [__m512; TILED_CHUNKS] = [zero; TILED_CHUNKS];
                let mut acc1: [__m512; TILED_CHUNKS] = [zero; TILED_CHUNKS];
                if let Some(b) = pw_bias {
                    let bp = b.as_ptr().add(oc_start + tile_off);
                    for k in 0..TILED_CHUNKS {
                        let bv = _mm512_loadu_ps(bp.add(k * 16));
                        acc0[k] = bv;
                        acc1[k] = bv;
                    }
                }
                // STREAM-PROBE prefetch: ci-stride is `c_exp * 4` bytes
                // (e.g. 1280 B for xif4_5 c_exp=320), beyond HW prefetcher.
                // T0 prefetch on (ci + AHEAD)'s weight cacheline hides L1
                // miss latency behind the FMA pipe.
                const FPD_PF_AHEAD: usize = 4;
                let pf_on = pw_prefetch_enabled();
                for ci in 0..c_in {
                    if pf_on && ci + FPD_PF_AHEAD < c_in {
                        let pf_ptr = pw_weight
                            .as_ptr()
                            .add((ci + FPD_PF_AHEAD) * c_exp + oc_start + tile_off);
                        _mm_prefetch(pf_ptr as *const i8, _MM_HINT_T0);
                    }
                    let x0 = _mm512_set1_ps(*src_row.as_ptr().add(iw * c_in + ci));
                    let x1 = _mm512_set1_ps(*src_row.as_ptr().add((iw + 1) * c_in + ci));
                    let wp = pw_weight.as_ptr().add(ci * c_exp + oc_start + tile_off);
                    for k in 0..TILED_CHUNKS {
                        let w = _mm512_loadu_ps(wp.add(k * 16));
                        acc0[k] = _mm512_fmadd_ps(x0, w, acc0[k]);
                        acc1[k] = _mm512_fmadd_ps(x1, w, acc1[k]);
                    }
                }
                if relu {
                    for k in 0..TILED_CHUNKS {
                        acc0[k] = _mm512_max_ps(acc0[k], zero);
                        acc1[k] = _mm512_max_ps(acc1[k], zero);
                    }
                }
                for k in 0..TILED_CHUNKS {
                    let off = tile_off + k * 16;
                    _mm512_storeu_ps(dst_ptr0.add(off), acc0[k]);
                    _mm512_storeu_ps(dst_ptr1.add(off), acc1[k]);
                }
            }

            if residual_chunks > 0 {
                let tile_off = full_tiles * TILED_CHUNKS * 16;
                let mut acc0: [__m512; TILED_CHUNKS] = [zero; TILED_CHUNKS];
                let mut acc1: [__m512; TILED_CHUNKS] = [zero; TILED_CHUNKS];
                if let Some(b) = pw_bias {
                    let bp = b.as_ptr().add(oc_start + tile_off);
                    for k in 0..residual_chunks {
                        let bv = _mm512_loadu_ps(bp.add(k * 16));
                        acc0[k] = bv;
                        acc1[k] = bv;
                    }
                }
                const FPD_PF_AHEAD2: usize = 4;
                let pf_on2 = pw_prefetch_enabled();
                for ci in 0..c_in {
                    if pf_on2 && ci + FPD_PF_AHEAD2 < c_in {
                        let pf_ptr = pw_weight
                            .as_ptr()
                            .add((ci + FPD_PF_AHEAD2) * c_exp + oc_start + tile_off);
                        _mm_prefetch(pf_ptr as *const i8, _MM_HINT_T0);
                    }
                    let x0 = _mm512_set1_ps(*src_row.as_ptr().add(iw * c_in + ci));
                    let x1 = _mm512_set1_ps(*src_row.as_ptr().add((iw + 1) * c_in + ci));
                    let wp = pw_weight.as_ptr().add(ci * c_exp + oc_start + tile_off);
                    for k in 0..residual_chunks {
                        let w = _mm512_loadu_ps(wp.add(k * 16));
                        acc0[k] = _mm512_fmadd_ps(x0, w, acc0[k]);
                        acc1[k] = _mm512_fmadd_ps(x1, w, acc1[k]);
                    }
                }
                if relu {
                    for k in 0..residual_chunks {
                        acc0[k] = _mm512_max_ps(acc0[k], zero);
                        acc1[k] = _mm512_max_ps(acc1[k], zero);
                    }
                }
                for k in 0..residual_chunks {
                    let off = tile_off + k * 16;
                    _mm512_storeu_ps(dst_ptr0.add(off), acc0[k]);
                    _mm512_storeu_ps(dst_ptr1.add(off), acc1[k]);
                }
            }
            iw += 2;
        }
        while iw < iw_end {
            let dst_ptr = dst_row.as_mut_ptr().add(iw * c_tile);
            for tile in 0..full_tiles {
                let tile_off = tile * TILED_CHUNKS * 16;
                let mut acc: [__m512; TILED_CHUNKS] = [zero; TILED_CHUNKS];
                if let Some(b) = pw_bias {
                    let bp = b.as_ptr().add(oc_start + tile_off);
                    for k in 0..TILED_CHUNKS {
                        acc[k] = _mm512_loadu_ps(bp.add(k * 16));
                    }
                }
                const FPD_PF_AHEAD3: usize = 4;
                let pf_on3 = pw_prefetch_enabled();
                for ci in 0..c_in {
                    if pf_on3 && ci + FPD_PF_AHEAD3 < c_in {
                        let pf_ptr = pw_weight
                            .as_ptr()
                            .add((ci + FPD_PF_AHEAD3) * c_exp + oc_start + tile_off);
                        _mm_prefetch(pf_ptr as *const i8, _MM_HINT_T0);
                    }
                    let x = _mm512_set1_ps(*src_row.as_ptr().add(iw * c_in + ci));
                    let wp = pw_weight.as_ptr().add(ci * c_exp + oc_start + tile_off);
                    for k in 0..TILED_CHUNKS {
                        let w = _mm512_loadu_ps(wp.add(k * 16));
                        acc[k] = _mm512_fmadd_ps(x, w, acc[k]);
                    }
                }
                if relu {
                    for k in 0..TILED_CHUNKS {
                        acc[k] = _mm512_max_ps(acc[k], zero);
                    }
                }
                for k in 0..TILED_CHUNKS {
                    _mm512_storeu_ps(dst_ptr.add(tile_off + k * 16), acc[k]);
                }
            }

            if residual_chunks > 0 {
                let tile_off = full_tiles * TILED_CHUNKS * 16;
                let mut acc: [__m512; TILED_CHUNKS] = [zero; TILED_CHUNKS];
                if let Some(b) = pw_bias {
                    let bp = b.as_ptr().add(oc_start + tile_off);
                    for k in 0..residual_chunks {
                        acc[k] = _mm512_loadu_ps(bp.add(k * 16));
                    }
                }
                const FPD_PF_AHEAD4: usize = 4;
                let pf_on4 = pw_prefetch_enabled();
                for ci in 0..c_in {
                    if pf_on4 && ci + FPD_PF_AHEAD4 < c_in {
                        let pf_ptr = pw_weight
                            .as_ptr()
                            .add((ci + FPD_PF_AHEAD4) * c_exp + oc_start + tile_off);
                        _mm_prefetch(pf_ptr as *const i8, _MM_HINT_T0);
                    }
                    let x = _mm512_set1_ps(*src_row.as_ptr().add(iw * c_in + ci));
                    let wp = pw_weight.as_ptr().add(ci * c_exp + oc_start + tile_off);
                    for k in 0..residual_chunks {
                        let w = _mm512_loadu_ps(wp.add(k * 16));
                        acc[k] = _mm512_fmadd_ps(x, w, acc[k]);
                    }
                }
                if relu {
                    for k in 0..residual_chunks {
                        acc[k] = _mm512_max_ps(acc[k], zero);
                    }
                }
                for k in 0..residual_chunks {
                    _mm512_storeu_ps(dst_ptr.add(tile_off + k * 16), acc[k]);
                }
            }
            iw += 1;
        }
    }
}

pub(super) fn compute_dw5_tile_avx512(ctx: Dw5TileCtx<'_, '_>) {
    #[allow(unsafe_code)]
    unsafe {
        compute_dw5_tile_avx512_inner(ctx)
    }
}

#[target_feature(enable = "avx512f")]
#[allow(unsafe_code)]
unsafe fn compute_dw5_tile_avx512_inner(ctx: Dw5TileCtx<'_, '_>) {
    unsafe {
        let Dw5TileCtx {
            rows,
            dw_weight,
            dw_bias,
            out_row,
            in_w,
            ow_start,
            ow_end,
            c_exp,
            oc_start,
            c_tile,
            stride,
            pad,
            relu,
        } = ctx;
        let chunks = c_tile / 16;
        let full_tiles = chunks / TILED_CHUNKS;
        let residual_chunks = chunks % TILED_CHUNKS;
        let zero = _mm512_setzero_ps();
        let w_ky_stride = 5 * c_exp;
        let w_kx_stride = c_exp;

        // Resolve the 5 input-row pointers once. `None` (ih out of bounds)
        // → null, which the per-tile body skips. For interior output rows
        // all 5 are non-null, so the interior fast path runs branch-free.
        let row_ptrs: [*const f32; 5] = std::array::from_fn(|ky| match rows[ky] {
            Some(r) => r.as_ptr(),
            None => std::ptr::null(),
        });
        let all_rows = row_ptrs.iter().all(|p| !p.is_null());

        // One tile's worth of FMAs for `nchunks` chunks. `$interior`
        // selects the branch-free body (no per-kx bounds check) vs the
        // checked body. Hoisting the bounds check out of the inner k-loop
        // (the FMA pipe) is the whole point — the boundary `continue`
        // statements were preventing LLVM from scheduling the 5×5×nchunks
        // FMAs as a flat pipeline (DW5 ran at IPC ~1.3).
        macro_rules! tile_body {
            ($ow:expr, $nchunks:expr, $tile_off:expr, $interior:expr) => {{
                let ow_v = $ow;
                let tile_off = $tile_off;
                let mut acc: [__m512; TILED_CHUNKS] = [zero; TILED_CHUNKS];
                if let Some(b) = dw_bias {
                    let bp = b.as_ptr().add(oc_start + tile_off);
                    for k in 0..$nchunks {
                        acc[k] = _mm512_loadu_ps(bp.add(k * 16));
                    }
                }
                let iw0 = (ow_v as i32) * (stride as i32) - (pad as i32);
                for ky in 0..5usize {
                    let pw_row = row_ptrs[ky];
                    if pw_row.is_null() {
                        continue;
                    }
                    for kx in 0..5usize {
                        let iw = iw0 + kx as i32;
                        if !$interior && (iw < 0 || (iw as usize) >= in_w) {
                            continue;
                        }
                        let pw_ptr = pw_row.add(iw as usize * c_tile + tile_off);
                        let w_ptr = dw_weight
                            .as_ptr()
                            .add(ky * w_ky_stride + kx * w_kx_stride + oc_start + tile_off);
                        for k in 0..$nchunks {
                            let x = _mm512_loadu_ps(pw_ptr.add(k * 16));
                            let w = _mm512_loadu_ps(w_ptr.add(k * 16));
                            acc[k] = _mm512_fmadd_ps(x, w, acc[k]);
                        }
                    }
                }
                if relu {
                    for k in 0..$nchunks {
                        acc[k] = _mm512_max_ps(acc[k], zero);
                    }
                }
                let dst_ptr = out_row.as_mut_ptr().add(ow_v * c_exp + oc_start);
                for k in 0..$nchunks {
                    _mm512_storeu_ps(dst_ptr.add(tile_off + k * 16), acc[k]);
                }
            }};
        }

        macro_rules! tile_pixel {
            ($ow:expr, $interior:expr) => {{
                for tile in 0..full_tiles {
                    tile_body!($ow, TILED_CHUNKS, tile * TILED_CHUNKS * 16, $interior);
                }
                if residual_chunks > 0 {
                    tile_body!(
                        $ow,
                        residual_chunks,
                        full_tiles * TILED_CHUNKS * 16,
                        $interior
                    );
                }
            }};
        }

        let (int_lo, int_hi) = if all_rows {
            dw5_interior_ow_range(in_w, stride, pad, ow_start, ow_end)
        } else {
            (ow_end, ow_end)
        };

        for ow in ow_start..int_lo {
            tile_pixel!(ow, false);
        }

        if dw5_reuse_enabled() {
            let ow_chunks = chunks / DW5_REUSE_CHUNKS;
            let ow_res_chunks = chunks % DW5_REUSE_CHUNKS;
            let mut ow = int_lo;
            while ow + DW5_REUSE_OW <= int_hi {
                for ct in 0..ow_chunks {
                    dw5_reuse_block_tile(
                        &row_ptrs,
                        dw_weight,
                        dw_bias,
                        out_row,
                        ow,
                        ct * DW5_REUSE_CHUNKS * 16,
                        DW5_REUSE_CHUNKS,
                        c_exp,
                        c_tile,
                        oc_start,
                        w_ky_stride,
                        w_kx_stride,
                        stride,
                        pad,
                        relu,
                        zero,
                    );
                }
                if ow_res_chunks > 0 {
                    dw5_reuse_block_tile(
                        &row_ptrs,
                        dw_weight,
                        dw_bias,
                        out_row,
                        ow,
                        ow_chunks * DW5_REUSE_CHUNKS * 16,
                        ow_res_chunks,
                        c_exp,
                        c_tile,
                        oc_start,
                        w_ky_stride,
                        w_kx_stride,
                        stride,
                        pad,
                        relu,
                        zero,
                    );
                }
                ow += DW5_REUSE_OW;
            }
            for ow in ow..int_hi {
                tile_pixel!(ow, true);
            }
        } else {
            for ow in int_lo..int_hi {
                tile_pixel!(ow, true);
            }
        }

        for ow in int_hi..ow_end {
            tile_pixel!(ow, false);
        }
    }
}

/// `dw5_reuse_block` for the packed-tile layout: inputs are `c_tile`-strided
/// per row, weights and output carry the `oc_start` channel-tile offset.
#[target_feature(enable = "avx512f")]
#[allow(unsafe_code, clippy::too_many_arguments)]
unsafe fn dw5_reuse_block_tile(
    row_ptrs: &[*const f32; 5],
    dw_weight: &[f32],
    dw_bias: Option<&[f32]>,
    out_row: &mut [f32],
    ow0: usize,
    tile_off: usize,
    nchunks: usize,
    c_exp: usize,
    c_tile: usize,
    oc_start: usize,
    w_ky_stride: usize,
    w_kx_stride: usize,
    stride: usize,
    pad: usize,
    relu: bool,
    zero: __m512,
) {
    unsafe {
        let mut acc: [__m512; DW5_REUSE_OW * DW5_REUSE_CHUNKS] =
            [zero; DW5_REUSE_OW * DW5_REUSE_CHUNKS];
        if let Some(b) = dw_bias {
            let bp = b.as_ptr().add(oc_start + tile_off);
            for k in 0..nchunks {
                let bv = _mm512_loadu_ps(bp.add(k * 16));
                for p in 0..DW5_REUSE_OW {
                    acc[p * DW5_REUSE_CHUNKS + k] = bv;
                }
            }
        }

        let iw0_base = (ow0 as i32) * (stride as i32) - (pad as i32);
        for ky in 0..5usize {
            let pw_row = row_ptrs[ky];
            for kx in 0..5usize {
                let w_ptr = dw_weight
                    .as_ptr()
                    .add(ky * w_ky_stride + kx * w_kx_stride + oc_start + tile_off);
                for k in 0..nchunks {
                    let w = _mm512_loadu_ps(w_ptr.add(k * 16));
                    for p in 0..DW5_REUSE_OW {
                        let iw = iw0_base + (p * stride) as i32 + kx as i32;
                        let pw_ptr = pw_row.add(iw as usize * c_tile + tile_off + k * 16);
                        let x = _mm512_loadu_ps(pw_ptr);
                        let idx = p * DW5_REUSE_CHUNKS + k;
                        acc[idx] = _mm512_fmadd_ps(x, w, acc[idx]);
                    }
                }
            }
        }

        for p in 0..DW5_REUSE_OW {
            let dst_ptr = out_row
                .as_mut_ptr()
                .add((ow0 + p) * c_exp + oc_start + tile_off);
            for k in 0..nchunks {
                let idx = p * DW5_REUSE_CHUNKS + k;
                let v = if relu {
                    _mm512_max_ps(acc[idx], zero)
                } else {
                    acc[idx]
                };
                _mm512_storeu_ps(dst_ptr.add(k * 16), v);
            }
        }
    }
}
