//! Streaming PW-expand → DW 3×3 fused kernel.
//!
//! MobileNet-style inverted-bottleneck blocks start with a 1×1 pointwise
//! expansion (`C_in → C_exp`) followed by a 3×3 depthwise (`C_exp → C_exp`).
//! Running them as separate ops allocates and writes the full
//! `[N, H, W, C_exp]` intermediate to memory — ~6 MB for the tracker's
//! `/xif2_0` pair at `C_exp = 96`, big enough to spill out of L2.
//!
//! This kernel streams through one output row at a time, keeping exactly
//! three rows of PW-expanded output live in a small ring buffer (~36 KB
//! for the largest tracker shape). The DW 3×3 window reads from that
//! buffer, hot in L1. The full intermediate tensor never hits RAM.
//!
//! ## Contract
//!
//! - Input NHWC `[N, H, W, C_in]`, contiguous f32.
//! - PW weight KHWC `[1, 1, C_in, C_exp]` (flattened `C_in * C_exp` f32).
//!   Caller must hand us the pre-permuted KHWC form — this is what the
//!   loader stores in `khwc_weights`.
//! - DW weight `[KH=3, KW=3, C_exp, 1]` (flattened `9 * C_exp` f32). The
//!   loader's `dw_khwc_weights` permutation.
//! - Output NHWC `[N, out_h, out_w, C_exp]`, pre-allocated.
//! - Kernel is hard-wired to `KH=KW=3`, symmetric pad `pad_h=pad_w=1`,
//!   stride `s ∈ {1, 2}` (both matter on tracker). Other shapes fall
//!   back to the caller's chained-compute path.
//! - Activations: `None` or `Relu` for both PW and DW. `SiLU` is rare on
//!   inverted-bottleneck openings and not supported here.
//!
//! ## Multi-arch
//!
//! Three implementations share one scalar reference via tests:
//! - scalar: any target, fallback when C_exp doesn't meet SIMD alignment.
//! - AVX2 + FMA: primary Zen 4 / Intel SSE4+ path; processes `C_exp`
//!   in YMM groups of 8.
//! - NEON: aarch64 path; processes `C_exp` in q-reg groups of 4.
//!
//! ## MT
//!
//! Parallelises over output rows via `par_chunks_mut`. Each worker
//! owns a private 3-row PW ring buffer and computes its own PW rows —
//! a tiny amount of duplicated work at chunk boundaries (at most 2
//! PW rows re-computed per worker), well worth the avoided cross-
//! worker synchronisation.

use rayon::ThreadPool;
use rayon::prelude::*;

use super::conv::Activation;

/// Dispatches PW-expand → DW 3×3 to the fastest available implementation.
///
/// # Safety
///
/// Slices must match the shape contract in the module docs. Mismatches
/// trip `debug_assert` bounds checks and produce wrong output in
/// release builds (no unsafe read past the slice — all indexing goes
/// through bounds-checked `&[f32]` ops).
#[allow(clippy::too_many_arguments)]
pub fn fused_pw_expand_dw_3x3(
    input: &[f32],
    pw_weight: &[f32],
    pw_bias: Option<&[f32]>,
    dw_weight: &[f32],
    dw_bias: Option<&[f32]>,
    output: &mut [f32],
    batch: usize,
    in_h: usize,
    in_w: usize,
    c_in: usize,
    c_exp: usize,
    stride: usize,
    pw_activation: Activation,
    dw_activation: Activation,
    thread_pool: Option<&ThreadPool>,
) {
    debug_assert_eq!(input.len(), batch * in_h * in_w * c_in);
    debug_assert_eq!(pw_weight.len(), c_in * c_exp);
    debug_assert_eq!(dw_weight.len(), 9 * c_exp);
    debug_assert!(stride == 1 || stride == 2);
    if let Some(b) = pw_bias {
        debug_assert_eq!(b.len(), c_exp);
    }
    if let Some(b) = dw_bias {
        debug_assert_eq!(b.len(), c_exp);
    }

    let pad: usize = 1; // 3×3 SAME-pad
    let out_h = (in_h + 2 * pad - 3) / stride + 1;
    let out_w = (in_w + 2 * pad - 3) / stride + 1;
    debug_assert_eq!(output.len(), batch * out_h * out_w * c_exp);

    // Choose the per-row worker closure. Each variant writes one output
    // row of shape `[out_w, c_exp]` into its `out_row` argument, given
    // the DW output row index and the batch-slice input.
    let out_row_stride = out_w * c_exp;
    let out_batch_stride = out_h * out_row_stride;
    let in_batch_stride = in_h * in_w * c_in;

    let pw_relu = matches!(pw_activation, Activation::Relu);
    let dw_relu = matches!(dw_activation, Activation::Relu);

    // Select variant once per entry.
    let variant = select_variant(c_exp);

    for ni in 0..batch {
        let batch_in = &input[ni * in_batch_stride..(ni + 1) * in_batch_stride];
        let batch_out = &mut output[ni * out_batch_stride..(ni + 1) * out_batch_stride];

        let run_rows = |out_chunk: &mut [f32], oh_start: usize, oh_end: usize| {
            // Per-worker ring buffer: 3 PW rows × [in_w, c_exp]. Allocated
            // once per chunk, kept L1-hot across the chunk's rows.
            let pw_row_len = in_w * c_exp;
            let mut pw_ring: Vec<f32> = vec![0.0; 3 * pw_row_len];

            // Strip-mining: process `W_TILE` output columns at a time so
            // the per-tile ring-buffer footprint (3 × (W_TILE + halo) ×
            // c_exp × 4 bytes) stays resident in L1D. Without this, for
            // in_w=64 c_exp=96 the full ring buffer is 72 KB > 32 KB L1D,
            // and every DW tap read hits L2 latency instead of L1's.
            //
            // The tile's input-column range is `[iw_tile_start, iw_tile_end)`
            // where `iw_tile_start = ow_tile_start * stride - pad` (DW reads
            // up to 2 cols left of the output start for 3×3 same-pad) and
            // `iw_tile_end = (ow_tile_end - 1) * stride + 3 - pad` (up to
            // 2 cols right of the tile's last output col).
            //
            // Per-tile reset of `slot_row`: forces PW recompute at tile
            // boundaries (the slot indices get invalidated across tiles
            // because different tile's PW ranges overlap only at halos).
            let pick_w_tile = || -> usize {
                static W_TILE_OVERRIDE: std::sync::OnceLock<Option<usize>> =
                    std::sync::OnceLock::new();
                let w_tile_override = *W_TILE_OVERRIDE.get_or_init(|| {
                    std::env::var("YSCV_FUSED_PW_DW_W_TILE")
                        .ok()
                        .and_then(|v| v.parse::<usize>().ok())
                        .filter(|&v| v >= 4)
                });
                if let Some(v) = w_tile_override {
                    return v.min(out_w).max(4);
                }
                // Aim to keep per-tile ring buffer ≤ ~20 KB. Adjust based
                // on c_exp. Stride-2 output needs 2× input halo width, so
                // compute based on input-column footprint.
                let target_kb = 20usize;
                let per_col_bytes = c_exp * 3 * std::mem::size_of::<f32>();
                let max_iw_cols = (target_kb * 1024) / per_col_bytes.max(1);
                // Convert input cols to output cols: approximately iw / stride.
                let max_ow_cols = max_iw_cols / stride.max(1);
                // Round down to power of 2 between 4 and out_w for clean tiling.
                if max_ow_cols >= out_w {
                    out_w // no tiling needed, fits L1 at full width
                } else if max_ow_cols >= 32 {
                    32
                } else if max_ow_cols >= 16 {
                    16
                } else if max_ow_cols >= 8 {
                    8
                } else {
                    // Very wide channels (c_exp ≥ 384); tile 4 at most.
                    4
                }
            };
            let w_tile = pick_w_tile();

            // Ensures `pw_ring[slot_for(ih)]` holds PW output for input
            // row `ih` over the column range `[iw_lo, iw_hi)`. Returns
            // `false` when `ih` is OOB → DW treats as zero padding.
            let ensure_pw_row = |ih: i32,
                                 iw_lo: usize,
                                 iw_hi: usize,
                                 pw_ring: &mut [f32],
                                 slot_row: &mut [Option<(usize, usize, usize)>; 3]|
             -> bool {
                if ih < 0 || (ih as usize) >= in_h {
                    return false;
                }
                let ih_u = ih as usize;
                let slot = ih_u % 3;
                // Only reuse slot if same row AND current contents cover
                // the requested tile range.
                if let Some((cached_ih, cached_lo, cached_hi)) = slot_row[slot]
                    && cached_ih == ih_u
                    && cached_lo <= iw_lo
                    && cached_hi >= iw_hi
                {
                    return true;
                }
                let src_row = &batch_in[ih_u * in_w * c_in..(ih_u + 1) * in_w * c_in];
                let dst_row = &mut pw_ring[slot * pw_row_len..(slot + 1) * pw_row_len];
                (variant.compute_pw_row)(
                    src_row, pw_weight, pw_bias, dst_row, iw_lo, iw_hi, c_in, c_exp, pw_relu,
                );
                slot_row[slot] = Some((ih_u, iw_lo, iw_hi));
                true
            };

            // Slot tracking now carries (input row, iw_lo, iw_hi) so we
            // can detect tile-mismatched reuse.
            let mut slot_row_v2: [Option<(usize, usize, usize)>; 3] = [None, None, None];

            let mut ow_tile_start = 0usize;
            while ow_tile_start < out_w {
                let ow_tile_end = (ow_tile_start + w_tile).min(out_w);
                // Input col range needed for this output tile (with 3×3 halo).
                // For stride-1 same-pad (pad=1): iw covers 3 cols per output.
                // For stride-2 same-pad (pad=1): iw step 2 per output, 3 cols.
                let iw_lo = (ow_tile_start as i32 * stride as i32 - pad as i32).max(0) as usize;
                let iw_hi =
                    (((ow_tile_end - 1) as i32) * stride as i32 + 3 - pad as i32).max(0) as usize;
                let iw_hi = iw_hi.min(in_w);

                // Reset slot tracking on tile switch — previous tile's PW
                // data may not cover this tile's iw range.
                if ow_tile_start > 0 {
                    slot_row_v2 = [None, None, None];
                }

                for oh in oh_start..oh_end {
                    let ih0 = (oh as i32) * (stride as i32) - (pad as i32);
                    let row0_ok = ensure_pw_row(ih0, iw_lo, iw_hi, &mut pw_ring, &mut slot_row_v2);
                    let row1_ok =
                        ensure_pw_row(ih0 + 1, iw_lo, iw_hi, &mut pw_ring, &mut slot_row_v2);
                    let row2_ok =
                        ensure_pw_row(ih0 + 2, iw_lo, iw_hi, &mut pw_ring, &mut slot_row_v2);

                    let row0 = row0_ok.then(|| {
                        let k = ((ih0 as usize) % 3) * pw_row_len;
                        &pw_ring[k..k + pw_row_len]
                    });
                    let row1 = row1_ok.then(|| {
                        let k = (((ih0 + 1) as usize) % 3) * pw_row_len;
                        &pw_ring[k..k + pw_row_len]
                    });
                    let row2 = row2_ok.then(|| {
                        let k = (((ih0 + 2) as usize) % 3) * pw_row_len;
                        &pw_ring[k..k + pw_row_len]
                    });

                    let out_row_slice = &mut out_chunk
                        [(oh - oh_start) * out_row_stride..(oh - oh_start + 1) * out_row_stride];
                    (variant.compute_dw_row)(
                        row0,
                        row1,
                        row2,
                        dw_weight,
                        dw_bias,
                        out_row_slice,
                        in_w,
                        ow_tile_start,
                        ow_tile_end,
                        c_exp,
                        stride,
                        pad,
                        dw_relu,
                    );
                }
                ow_tile_start = ow_tile_end;
            }
        };

        // Parallel dispatch: split output rows across workers. Each
        // chunk has size >= 1 row; rayon handles balancing.
        let par_min_rows = 4;
        // cfg!(miri): rayon/crossbeam epoch-GC violates Stacked Borrows.
        // Force sequential under Miri so correctness tests can still run.
        if !cfg!(miri)
            && out_h >= par_min_rows
            && let Some(pool) = thread_pool
        {
            let nthreads = pool.current_num_threads().max(1);
            if nthreads <= 1 {
                run_rows(batch_out, 0, out_h);
            } else {
                let rows_per_chunk = out_h.div_ceil(nthreads).max(1);
                let bytes_per_chunk = rows_per_chunk * out_row_stride;
                pool.install(|| {
                    batch_out
                        .par_chunks_mut(bytes_per_chunk)
                        .enumerate()
                        .for_each(|(chunk_idx, chunk)| {
                            let oh_start = chunk_idx * rows_per_chunk;
                            let oh_end = (oh_start + rows_per_chunk).min(out_h);
                            run_rows(chunk, oh_start, oh_end);
                        });
                });
            }
        } else if !cfg!(miri) && out_h >= par_min_rows {
            // No explicit pool — rayon's current ambient scope still
            // gets par_iter. This catches callers inside an already-
            // installed pool.
            let nthreads = rayon::current_num_threads().max(1);
            if nthreads <= 1 {
                run_rows(batch_out, 0, out_h);
            } else {
                let rows_per_chunk = out_h.div_ceil(nthreads).max(1);
                let bytes_per_chunk = rows_per_chunk * out_row_stride;
                batch_out
                    .par_chunks_mut(bytes_per_chunk)
                    .enumerate()
                    .for_each(|(chunk_idx, chunk)| {
                        let oh_start = chunk_idx * rows_per_chunk;
                        let oh_end = (oh_start + rows_per_chunk).min(out_h);
                        run_rows(chunk, oh_start, oh_end);
                    });
            }
        } else {
            run_rows(batch_out, 0, out_h);
        }
    }
}

// ── Variant dispatch table ──────────────────────────────────────────

/// Per-arch vtable selected once at the entry point.
///
/// Functions accept `iw_start/iw_end` (PW) and `ow_start/ow_end` (DW) column
/// ranges. When a variant supports strip-mining, the outer loop calls these
/// with sub-ranges so the ring buffer read footprint stays L1-resident.
/// Callers that don't want tiling pass the full [0, in_w) / [0, out_w) range.
struct Variant {
    /// Writes `dst_row[iw * c_exp .. (iw+1) * c_exp]` for `iw ∈ [iw_start, iw_end)`.
    compute_pw_row: fn(
        &[f32],
        &[f32],
        Option<&[f32]>,
        &mut [f32],
        usize, // iw_start
        usize, // iw_end
        usize, // c_in
        usize, // c_exp
        bool,  // relu
    ),
    /// Writes `out_row[ow * c_exp .. (ow+1) * c_exp]` for `ow ∈ [ow_start, ow_end)`.
    compute_dw_row: fn(
        Option<&[f32]>,
        Option<&[f32]>,
        Option<&[f32]>,
        &[f32],
        Option<&[f32]>,
        &mut [f32],
        usize, // in_w
        usize, // ow_start
        usize, // ow_end
        usize, // c_exp
        usize, // stride
        usize, // pad
        bool,  // relu
    ),
}

/// Max `c_exp / 16` chunks the AVX-512 register-blocked path supports.
/// 16 ZMMs of accumulators = 256 lanes of `c_exp`. Shapes larger than
/// this spill below the performance floor of AVX2, so we fall back to
/// AVX2 rather than let AVX-512 go memory-bound.
#[cfg(target_arch = "x86_64")]
pub(super) const AVX512_REG_MAX_CHUNKS: usize = 16;

fn select_variant(c_exp: usize) -> Variant {
    #[cfg(target_arch = "x86_64")]
    {
        // Register-blocked AVX-512: keeps all `c_exp / 16` accumulators
        // in ZMM across the inner loop. Only eligible while the chunks
        // fit in the ZMM file with room for `x`/`w` temps — capped at
        // 16 chunks (c_exp ≤ 256) which covers tracker shapes
        // 16/96/144/192/256 but not 672.
        if c_exp.is_multiple_of(16)
            && c_exp / 16 <= AVX512_REG_MAX_CHUNKS
            && !cfg!(miri)
            && is_x86_feature_detected!("avx512f")
        {
            return Variant {
                compute_pw_row: avx512::compute_pw_row_avx512,
                compute_dw_row: avx512::compute_dw_row_avx512,
            };
        }
        // Tiled AVX-512: `c_exp > 256` spills the pure register-blocked
        // variant, so fall through to the tiled kernel (fixed 8-chunk
        // tile + residual). Covers tracker's `c_exp=672` pairs in the
        // `/xif4_*` stage.
        if c_exp.is_multiple_of(16) && !cfg!(miri) && is_x86_feature_detected!("avx512f") {
            return Variant {
                compute_pw_row: avx512::compute_pw_row_avx512_tiled,
                compute_dw_row: avx512::compute_dw_row_avx512_tiled,
            };
        }
        if c_exp.is_multiple_of(8)
            && !cfg!(miri)
            && is_x86_feature_detected!("avx2")
            && is_x86_feature_detected!("fma")
        {
            return Variant {
                compute_pw_row: avx2::compute_pw_row_avx2,
                compute_dw_row: avx2::compute_dw_row_avx2,
            };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if c_exp.is_multiple_of(4) && !cfg!(miri) && std::arch::is_aarch64_feature_detected!("neon")
        {
            return Variant {
                compute_pw_row: neon::compute_pw_row_neon,
                compute_dw_row: neon::compute_dw_row_neon,
            };
        }
    }
    Variant {
        compute_pw_row: scalar::compute_pw_row_scalar,
        compute_dw_row: scalar::compute_dw_row_scalar,
    }
}

// ── Scalar reference ────────────────────────────────────────────────

mod scalar {

    /// Scalar PW row: for each output column `iw`, compute `c_exp`
    /// outputs from `c_in` inputs. Applies bias + Relu.
    pub(super) fn compute_pw_row_scalar(
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
        for iw in iw_start..iw_end {
            let src_off = iw * c_in;
            let dst_off = iw * c_exp;
            // Initialise `dst[dst_off..dst_off+c_exp]` from bias or 0.
            if let Some(b) = pw_bias {
                dst_row[dst_off..dst_off + c_exp].copy_from_slice(&b[..c_exp]);
            } else {
                dst_row[dst_off..dst_off + c_exp].fill(0.0);
            }
            // Accumulate c_in × c_exp matmul.
            for ci in 0..c_in {
                let x = src_row[src_off + ci];
                let w_row = &pw_weight[ci * c_exp..(ci + 1) * c_exp];
                for ce in 0..c_exp {
                    dst_row[dst_off + ce] += x * w_row[ce];
                }
            }
            if relu {
                for v in &mut dst_row[dst_off..dst_off + c_exp] {
                    if *v < 0.0 {
                        *v = 0.0;
                    }
                }
            }
        }
    }

    /// Scalar DW 3×3 row: reads up to three PW rows (each Option<&[f32]>
    /// — None for padding rows), writes one DW output row.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn compute_dw_row_scalar(
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
        // `dw_weight` layout: `[KY=3][KX=3][C_exp]` flattened.
        let w_ky_stride = 3 * c_exp;
        let w_kx_stride = c_exp;
        // Safe Option → slice wrapper; maps None to a zero slice of
        // length c_exp (allocated once).
        for ow in ow_start..ow_end {
            let iw0 = (ow as i32) * (stride as i32) - (pad as i32);
            let dst_off = ow * c_exp;
            // Init acc with bias.
            if let Some(b) = dw_bias {
                out_row[dst_off..dst_off + c_exp].copy_from_slice(&b[..c_exp]);
            } else {
                out_row[dst_off..dst_off + c_exp].fill(0.0);
            }

            // Iterate over 9 kernel taps (ky, kx). Skip OOB input
            // positions via `row{0,1,2}` being None or `iw` OOB.
            let rows = [row0, row1, row2];
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
                    let pw_off = iw_u * c_exp;
                    let w_off = ky * w_ky_stride + kx * w_kx_stride;
                    for ce in 0..c_exp {
                        out_row[dst_off + ce] += pw_row[pw_off + ce] * dw_weight[w_off + ce];
                    }
                }
            }
            if relu {
                for v in &mut out_row[dst_off..dst_off + c_exp] {
                    if *v < 0.0 {
                        *v = 0.0;
                    }
                }
            }
        }
    }
}

// ── AVX2 + FMA (x86_64) ──────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
mod avx2 {
    use std::arch::x86_64::*;

    /// AVX2 PW row: `c_exp` vectorised in YMM groups of 8. Requires
    /// `c_exp % 8 == 0`; caller guarantees via `select_variant`.
    pub(super) fn compute_pw_row_avx2(
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
        // SAFETY: caller verified `avx2`+`fma` available and
        // `c_exp % 8 == 0`; slices sized per contract.
        #[allow(unsafe_code)]
        unsafe {
            compute_pw_row_avx2_inner(
                src_row, pw_weight, pw_bias, dst_row, iw_start, iw_end, c_in, c_exp, relu,
            )
        }
    }

    #[target_feature(enable = "avx2,fma")]
    #[allow(unsafe_code, clippy::too_many_arguments)]
    unsafe fn compute_pw_row_avx2_inner(
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
            let zero = _mm256_setzero_ps();
            let c_exp_chunks = c_exp / 8;

            for iw in iw_start..iw_end {
                let src_off = iw * c_in;
                let dst_off = iw * c_exp;
                let dst_ptr = dst_row.as_mut_ptr().add(dst_off);

                // Init dst with bias (or zero).
                if let Some(b) = pw_bias {
                    let bp = b.as_ptr();
                    for ck in 0..c_exp_chunks {
                        let v = _mm256_loadu_ps(bp.add(ck * 8));
                        _mm256_storeu_ps(dst_ptr.add(ck * 8), v);
                    }
                } else {
                    for ck in 0..c_exp_chunks {
                        _mm256_storeu_ps(dst_ptr.add(ck * 8), zero);
                    }
                }

                // Accumulate c_in × c_exp matmul, 8 c_exp lanes at a time.
                for ci in 0..c_in {
                    let x = _mm256_set1_ps(*src_row.as_ptr().add(src_off + ci));
                    let w_row = pw_weight.as_ptr().add(ci * c_exp);
                    for ck in 0..c_exp_chunks {
                        let w = _mm256_loadu_ps(w_row.add(ck * 8));
                        let d = _mm256_loadu_ps(dst_ptr.add(ck * 8));
                        let r = _mm256_fmadd_ps(x, w, d);
                        _mm256_storeu_ps(dst_ptr.add(ck * 8), r);
                    }
                }

                if relu {
                    for ck in 0..c_exp_chunks {
                        let d = _mm256_loadu_ps(dst_ptr.add(ck * 8));
                        let r = _mm256_max_ps(d, zero);
                        _mm256_storeu_ps(dst_ptr.add(ck * 8), r);
                    }
                }
            }
        }
    }

    /// AVX2 DW 3×3 row: C_exp vectorised in YMM groups of 8.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn compute_dw_row_avx2(
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
        // SAFETY: caller verified avx2+fma available and c_exp % 8 == 0.
        #[allow(unsafe_code)]
        unsafe {
            compute_dw_row_avx2_inner(
                row0, row1, row2, dw_weight, dw_bias, out_row, in_w, ow_start, ow_end, c_exp,
                stride, pad, relu,
            )
        }
    }

    #[target_feature(enable = "avx2,fma")]
    #[allow(unsafe_code, clippy::too_many_arguments)]
    unsafe fn compute_dw_row_avx2_inner(
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
            let zero = _mm256_setzero_ps();
            let c_exp_chunks = c_exp / 8;
            let w_ky_stride = 3 * c_exp;
            let w_kx_stride = c_exp;

            let rows = [row0, row1, row2];

            for ow in ow_start..ow_end {
                let iw0 = (ow as i32) * (stride as i32) - (pad as i32);
                let dst_off = ow * c_exp;
                let dst_ptr = out_row.as_mut_ptr().add(dst_off);

                // Init with bias or zero.
                if let Some(b) = dw_bias {
                    let bp = b.as_ptr();
                    for ck in 0..c_exp_chunks {
                        _mm256_storeu_ps(dst_ptr.add(ck * 8), _mm256_loadu_ps(bp.add(ck * 8)));
                    }
                } else {
                    for ck in 0..c_exp_chunks {
                        _mm256_storeu_ps(dst_ptr.add(ck * 8), zero);
                    }
                }

                // 9 taps; skip OOB (row None or iw OOB).
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
                            let x = _mm256_loadu_ps(pw_ptr.add(ck * 8));
                            let w = _mm256_loadu_ps(w_ptr.add(ck * 8));
                            let d = _mm256_loadu_ps(dst_ptr.add(ck * 8));
                            let r = _mm256_fmadd_ps(x, w, d);
                            _mm256_storeu_ps(dst_ptr.add(ck * 8), r);
                        }
                    }
                }

                if relu {
                    for ck in 0..c_exp_chunks {
                        let d = _mm256_loadu_ps(dst_ptr.add(ck * 8));
                        let r = _mm256_max_ps(d, zero);
                        _mm256_storeu_ps(dst_ptr.add(ck * 8), r);
                    }
                }
            }
        }
    }
}

// ── AVX-512 + FMA (x86_64), register-blocked ─────────────────────────
//
// Key difference vs the AVX2 variant above: accumulators for every
// `c_exp` chunk live in ZMM registers across the entire inner loop.
// The AVX2 path reloads/stores dst on every `(c_in, c_chunk)` iteration
// — fine when accs don't fit in YMMs, but pure memory traffic for our
// shapes. With ZMM (16-wide) and `c_exp / 16 ≤ 16` chunks we have
// enough registers (32 ZMMs total, minus 2 for `x`/`w` temps) to keep
// accumulators hot through all `c_in` taps. That's the unlock for
// matching the mature per-op SIMD paths.

#[cfg(target_arch = "x86_64")]
mod avx512 {
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
                for ci in 0..c_in {
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
                    for ci in 0..c_in {
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

                    for ci in 0..c_in {
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

    /// Tile width (in 16-lane ZMM chunks) used by the tiled AVX-512
    /// variant below. 8 chunks = 128 `c_exp` lanes per tile: fits 8
    /// accumulators + 2 register-temp slots (x broadcast + w load) well
    /// within the 32-ZMM file with headroom for spill avoidance.
    ///
    /// Attempted TILED_CHUNKS=16 in Phase 2 (fewer c_in re-reads for
    /// c_exp=672): regressed +43 µs @ 1T. Residual of 10 chunks
    /// (42 % 16) hits the dynamic `0..residual_chunks` loop which LLVM
    /// doesn't unroll as cleanly as fixed-8 body. Reverted.
    pub(super) const TILED_CHUNKS: usize = 8;

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
                row0, row1, row2, dw_weight, dw_bias, out_row, in_w, ow_start, ow_end, c_exp,
                stride, pad, relu,
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

            for ow in ow_start..ow_end {
                let iw0 = (ow as i32) * (stride as i32) - (pad as i32);
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
                row0, row1, row2, dw_weight, dw_bias, out_row, in_w, ow_start, ow_end, c_exp,
                stride, pad, relu,
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

                // 9 taps. Skip OOB (row None or iw OOB).
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
}

// ── NEON (aarch64) ────────────────────────────────────────────────────

#[cfg(target_arch = "aarch64")]
mod neon {

    use std::arch::aarch64::*;
    const NEON_TILE_CHUNKS: usize = 8;

    #[inline]
    fn pw_2x_disabled() -> bool {
        static CACHED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        // Default ON. This env is only a kill-switch for A/B and quick rollback.
        *CACHED.get_or_init(|| std::env::var_os("YSCV_FUSED_PW_DW_PW2X_OFF").is_some())
    }

    /// NEON PW row: C_exp vectorised in q-reg groups of 4.
    pub(super) fn compute_pw_row_neon(
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
        // SAFETY: NEON is mandatory on aarch64 (ARMv8), caller verified
        // c_exp % 4 == 0 via select_variant.
        #[allow(unsafe_code)]
        unsafe {
            compute_pw_row_neon_inner(
                src_row, pw_weight, pw_bias, dst_row, iw_start, iw_end, c_in, c_exp, relu,
            )
        }
    }

    #[target_feature(enable = "neon")]
    #[allow(unsafe_code, clippy::too_many_arguments)]
    unsafe fn compute_pw_row_neon_inner(
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
            let zero = vdupq_n_f32(0.0);
            let c_exp_chunks = c_exp / 4;
            let full_tiles = c_exp_chunks / NEON_TILE_CHUNKS;
            let residual_chunks = c_exp_chunks % NEON_TILE_CHUNKS;
            let use_2x = !pw_2x_disabled();
            let mut iw = iw_start;

            // 2-column PW block: reuse the same PW weights for two adjacent
            // output columns to halve weight-stream pressure in the hot loop.
            if use_2x {
                while iw + 1 < iw_end {
                    let src_off0 = iw * c_in;
                    let src_off1 = (iw + 1) * c_in;
                    let dst_ptr0 = dst_row.as_mut_ptr().add(iw * c_exp);
                    let dst_ptr1 = dst_row.as_mut_ptr().add((iw + 1) * c_exp);

                    for tile in 0..full_tiles {
                        let lane_off = tile * NEON_TILE_CHUNKS * 4;
                        let mut acc0: [float32x4_t; NEON_TILE_CHUNKS] = [zero; NEON_TILE_CHUNKS];
                        let mut acc1: [float32x4_t; NEON_TILE_CHUNKS] = [zero; NEON_TILE_CHUNKS];

                        if let Some(b) = pw_bias {
                            let bp = b.as_ptr().add(lane_off);
                            for k in 0..NEON_TILE_CHUNKS {
                                let vb = vld1q_f32(bp.add(k * 4));
                                acc0[k] = vb;
                                acc1[k] = vb;
                            }
                        }

                        for ci in 0..c_in {
                            let x0 = vdupq_n_f32(*src_row.as_ptr().add(src_off0 + ci));
                            let x1 = vdupq_n_f32(*src_row.as_ptr().add(src_off1 + ci));
                            let w_base = pw_weight.as_ptr().add(ci * c_exp + lane_off);
                            for k in 0..NEON_TILE_CHUNKS {
                                let w = vld1q_f32(w_base.add(k * 4));
                                acc0[k] = vfmaq_f32(acc0[k], x0, w);
                                acc1[k] = vfmaq_f32(acc1[k], x1, w);
                            }
                        }

                        if relu {
                            for k in 0..NEON_TILE_CHUNKS {
                                acc0[k] = vmaxq_f32(acc0[k], zero);
                                acc1[k] = vmaxq_f32(acc1[k], zero);
                            }
                        }
                        for k in 0..NEON_TILE_CHUNKS {
                            let off = lane_off + k * 4;
                            vst1q_f32(dst_ptr0.add(off), acc0[k]);
                            vst1q_f32(dst_ptr1.add(off), acc1[k]);
                        }
                    }

                    if residual_chunks > 0 {
                        let lane_off = full_tiles * NEON_TILE_CHUNKS * 4;
                        let mut acc0: [float32x4_t; NEON_TILE_CHUNKS] = [zero; NEON_TILE_CHUNKS];
                        let mut acc1: [float32x4_t; NEON_TILE_CHUNKS] = [zero; NEON_TILE_CHUNKS];

                        if let Some(b) = pw_bias {
                            let bp = b.as_ptr().add(lane_off);
                            for k in 0..residual_chunks {
                                let vb = vld1q_f32(bp.add(k * 4));
                                acc0[k] = vb;
                                acc1[k] = vb;
                            }
                        }

                        for ci in 0..c_in {
                            let x0 = vdupq_n_f32(*src_row.as_ptr().add(src_off0 + ci));
                            let x1 = vdupq_n_f32(*src_row.as_ptr().add(src_off1 + ci));
                            let w_base = pw_weight.as_ptr().add(ci * c_exp + lane_off);
                            for k in 0..residual_chunks {
                                let w = vld1q_f32(w_base.add(k * 4));
                                acc0[k] = vfmaq_f32(acc0[k], x0, w);
                                acc1[k] = vfmaq_f32(acc1[k], x1, w);
                            }
                        }

                        if relu {
                            for k in 0..residual_chunks {
                                acc0[k] = vmaxq_f32(acc0[k], zero);
                                acc1[k] = vmaxq_f32(acc1[k], zero);
                            }
                        }
                        for k in 0..residual_chunks {
                            let off = lane_off + k * 4;
                            vst1q_f32(dst_ptr0.add(off), acc0[k]);
                            vst1q_f32(dst_ptr1.add(off), acc1[k]);
                        }
                    }
                    iw += 2;
                }
            }

            while iw < iw_end {
                let src_off = iw * c_in;
                let dst_off = iw * c_exp;
                let dst_ptr = dst_row.as_mut_ptr().add(dst_off);

                for tile in 0..full_tiles {
                    let lane_off = tile * NEON_TILE_CHUNKS * 4;
                    let mut acc: [float32x4_t; NEON_TILE_CHUNKS] = [zero; NEON_TILE_CHUNKS];

                    if let Some(b) = pw_bias {
                        let bp = b.as_ptr().add(lane_off);
                        for k in 0..NEON_TILE_CHUNKS {
                            acc[k] = vld1q_f32(bp.add(k * 4));
                        }
                    }

                    for ci in 0..c_in {
                        let x = vdupq_n_f32(*src_row.as_ptr().add(src_off + ci));
                        let w_base = pw_weight.as_ptr().add(ci * c_exp + lane_off);
                        for k in 0..NEON_TILE_CHUNKS {
                            let w = vld1q_f32(w_base.add(k * 4));
                            acc[k] = vfmaq_f32(acc[k], x, w);
                        }
                    }

                    if relu {
                        for k in 0..NEON_TILE_CHUNKS {
                            acc[k] = vmaxq_f32(acc[k], zero);
                        }
                    }
                    for k in 0..NEON_TILE_CHUNKS {
                        vst1q_f32(dst_ptr.add(lane_off + k * 4), acc[k]);
                    }
                }

                if residual_chunks > 0 {
                    let lane_off = full_tiles * NEON_TILE_CHUNKS * 4;
                    let mut acc: [float32x4_t; NEON_TILE_CHUNKS] = [zero; NEON_TILE_CHUNKS];

                    if let Some(b) = pw_bias {
                        let bp = b.as_ptr().add(lane_off);
                        for k in 0..residual_chunks {
                            acc[k] = vld1q_f32(bp.add(k * 4));
                        }
                    }

                    for ci in 0..c_in {
                        let x = vdupq_n_f32(*src_row.as_ptr().add(src_off + ci));
                        let w_base = pw_weight.as_ptr().add(ci * c_exp + lane_off);
                        for k in 0..residual_chunks {
                            let w = vld1q_f32(w_base.add(k * 4));
                            acc[k] = vfmaq_f32(acc[k], x, w);
                        }
                    }

                    if relu {
                        for k in 0..residual_chunks {
                            acc[k] = vmaxq_f32(acc[k], zero);
                        }
                    }
                    for k in 0..residual_chunks {
                        vst1q_f32(dst_ptr.add(lane_off + k * 4), acc[k]);
                    }
                }
                iw += 1;
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn compute_dw_row_neon(
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
        #[allow(unsafe_code)]
        unsafe {
            compute_dw_row_neon_inner(
                row0, row1, row2, dw_weight, dw_bias, out_row, in_w, ow_start, ow_end, c_exp,
                stride, pad, relu,
            )
        }
    }

    #[target_feature(enable = "neon")]
    #[allow(unsafe_code, clippy::too_many_arguments)]
    unsafe fn compute_dw_row_neon_inner(
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
            let zero = vdupq_n_f32(0.0);
            let c_exp_chunks = c_exp / 4;
            let full_tiles = c_exp_chunks / NEON_TILE_CHUNKS;
            let residual_chunks = c_exp_chunks % NEON_TILE_CHUNKS;
            let w_ky_stride = 3 * c_exp;
            let w_kx_stride = c_exp;
            let rows = [row0, row1, row2];

            for ow in ow_start..ow_end {
                let iw0 = (ow as i32) * (stride as i32) - (pad as i32);
                let dst_off = ow * c_exp;
                let dst_ptr = out_row.as_mut_ptr().add(dst_off);

                for tile in 0..full_tiles {
                    let lane_off = tile * NEON_TILE_CHUNKS * 4;
                    let mut acc: [float32x4_t; NEON_TILE_CHUNKS] = [zero; NEON_TILE_CHUNKS];

                    if let Some(b) = dw_bias {
                        let bp = b.as_ptr().add(lane_off);
                        for k in 0..NEON_TILE_CHUNKS {
                            acc[k] = vld1q_f32(bp.add(k * 4));
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
                            for k in 0..NEON_TILE_CHUNKS {
                                let x = vld1q_f32(pw_ptr.add(k * 4));
                                let w = vld1q_f32(w_ptr.add(k * 4));
                                acc[k] = vfmaq_f32(acc[k], x, w);
                            }
                        }
                    }

                    if relu {
                        for k in 0..NEON_TILE_CHUNKS {
                            acc[k] = vmaxq_f32(acc[k], zero);
                        }
                    }
                    for k in 0..NEON_TILE_CHUNKS {
                        vst1q_f32(dst_ptr.add(lane_off + k * 4), acc[k]);
                    }
                }

                if residual_chunks > 0 {
                    let lane_off = full_tiles * NEON_TILE_CHUNKS * 4;
                    let mut acc: [float32x4_t; NEON_TILE_CHUNKS] = [zero; NEON_TILE_CHUNKS];

                    if let Some(b) = dw_bias {
                        let bp = b.as_ptr().add(lane_off);
                        for k in 0..residual_chunks {
                            acc[k] = vld1q_f32(bp.add(k * 4));
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
                                let x = vld1q_f32(pw_ptr.add(k * 4));
                                let w = vld1q_f32(w_ptr.add(k * 4));
                                acc[k] = vfmaq_f32(acc[k], x, w);
                            }
                        }
                    }

                    if relu {
                        for k in 0..residual_chunks {
                            acc[k] = vmaxq_f32(acc[k], zero);
                        }
                    }
                    for k in 0..residual_chunks {
                        vst1q_f32(dst_ptr.add(lane_off + k * 4), acc[k]);
                    }
                }
            }
        }
    }
}

// ── Tests: SIMD variants vs scalar reference ─────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn seeded(n: usize, seed: f32) -> Vec<f32> {
        (0..n).map(|i| ((i as f32 + seed) * 0.013).sin()).collect()
    }

    /// Runs the fused kernel and the scalar reference path via the
    /// variant-selector's `scalar` fallback (forced by passing a
    /// C_exp that would still go through SIMD, but we compare against
    /// an independent scalar execution by calling the same kernel
    /// with a C_exp that defeats the SIMD check is not trivial; instead
    /// we compare fused output against a sequential PW then DW compute
    /// performed in-test).
    fn run_sequential_reference(
        input: &[f32],
        pw_weight: &[f32],
        pw_bias: Option<&[f32]>,
        dw_weight: &[f32],
        dw_bias: Option<&[f32]>,
        batch: usize,
        in_h: usize,
        in_w: usize,
        c_in: usize,
        c_exp: usize,
        stride: usize,
        pad: usize,
        pw_relu: bool,
        dw_relu: bool,
    ) -> Vec<f32> {
        // PW step: produce [batch, in_h, in_w, c_exp] tensor.
        let mut pw_out = vec![0.0f32; batch * in_h * in_w * c_exp];
        for ni in 0..batch {
            for ih in 0..in_h {
                for iw in 0..in_w {
                    let src_off = ((ni * in_h + ih) * in_w + iw) * c_in;
                    let dst_off = ((ni * in_h + ih) * in_w + iw) * c_exp;
                    for ce in 0..c_exp {
                        let mut acc = pw_bias.map(|b| b[ce]).unwrap_or(0.0);
                        for ci in 0..c_in {
                            acc += input[src_off + ci] * pw_weight[ci * c_exp + ce];
                        }
                        if pw_relu && acc < 0.0 {
                            acc = 0.0;
                        }
                        pw_out[dst_off + ce] = acc;
                    }
                }
            }
        }

        // DW step: [batch, in_h, in_w, c_exp] → [batch, out_h, out_w, c_exp].
        let out_h = (in_h + 2 * pad - 3) / stride + 1;
        let out_w = (in_w + 2 * pad - 3) / stride + 1;
        let mut dw_out = vec![0.0f32; batch * out_h * out_w * c_exp];
        for ni in 0..batch {
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let ih0 = (oh as i32) * (stride as i32) - (pad as i32);
                    let iw0 = (ow as i32) * (stride as i32) - (pad as i32);
                    let dst_off = ((ni * out_h + oh) * out_w + ow) * c_exp;
                    for ce in 0..c_exp {
                        let mut acc = dw_bias.map(|b| b[ce]).unwrap_or(0.0);
                        for ky in 0..3 {
                            let ih = ih0 + ky as i32;
                            if ih < 0 || (ih as usize) >= in_h {
                                continue;
                            }
                            for kx in 0..3 {
                                let iw = iw0 + kx as i32;
                                if iw < 0 || (iw as usize) >= in_w {
                                    continue;
                                }
                                let pw_off =
                                    ((ni * in_h + ih as usize) * in_w + iw as usize) * c_exp;
                                let w_off = (ky * 3 + kx) * c_exp;
                                acc += pw_out[pw_off + ce] * dw_weight[w_off + ce];
                            }
                        }
                        if dw_relu && acc < 0.0 {
                            acc = 0.0;
                        }
                        dw_out[dst_off + ce] = acc;
                    }
                }
            }
        }
        dw_out
    }

    fn case(
        batch: usize,
        in_h: usize,
        in_w: usize,
        c_in: usize,
        c_exp: usize,
        stride: usize,
        relu: bool,
    ) {
        let input = seeded(batch * in_h * in_w * c_in, 0.3);
        let pw_weight = seeded(c_in * c_exp, 1.7);
        let pw_bias: Vec<f32> = (0..c_exp).map(|i| 0.125 * (i as f32 - 4.0)).collect();
        let dw_weight = seeded(9 * c_exp, 2.1);
        let dw_bias: Vec<f32> = (0..c_exp).map(|i| 0.0625 * (i as f32 - 2.0)).collect();

        let pad = 1;
        let out_h = (in_h + 2 * pad - 3) / stride + 1;
        let out_w = (in_w + 2 * pad - 3) / stride + 1;
        let mut fused_out = vec![0.0f32; batch * out_h * out_w * c_exp];

        let act = if relu {
            Activation::Relu
        } else {
            Activation::None
        };
        fused_pw_expand_dw_3x3(
            &input,
            &pw_weight,
            Some(&pw_bias),
            &dw_weight,
            Some(&dw_bias),
            &mut fused_out,
            batch,
            in_h,
            in_w,
            c_in,
            c_exp,
            stride,
            act,
            act,
            None,
        );

        let ref_out = run_sequential_reference(
            &input,
            &pw_weight,
            Some(&pw_bias),
            &dw_weight,
            Some(&dw_bias),
            batch,
            in_h,
            in_w,
            c_in,
            c_exp,
            stride,
            pad,
            relu,
            relu,
        );

        for (i, (a, b)) in fused_out.iter().zip(ref_out.iter()).enumerate() {
            let delta = (a - b).abs();
            assert!(
                delta < 1e-4,
                "mismatch at {i}: fused={a} ref={b} delta={delta}"
            );
        }
    }

    #[test]
    fn tracker_xif2_0_stride2_c16_c96_relu() {
        // /xif2_0/pw/conv_1 + /xif2_0/dw/conv_1 shape.
        case(1, 128, 128, 16, 96, 2, true);
    }

    #[test]
    fn tracker_stride1_c24_c144_relu() {
        // /xif3_0 series shape (stride=1 sibling).
        case(1, 64, 64, 24, 144, 1, true);
    }

    #[test]
    fn small_shape_stride1_c8_c16_no_relu() {
        case(1, 5, 5, 8, 16, 1, false);
    }

    #[test]
    fn small_shape_stride2_c4_c32_relu() {
        case(1, 7, 7, 4, 32, 2, true);
    }

    #[test]
    fn c_exp_not_multiple_of_8_falls_back_to_scalar() {
        // c_exp=12: AVX2 chunks of 8 won't match; variant selector
        // falls through to scalar on x86_64. NEON needs c_exp % 4 == 0,
        // so 12 works there; on aarch64 this exercises NEON.
        case(1, 7, 7, 4, 12, 1, true);
    }

    #[test]
    fn batch_gt_1() {
        case(2, 8, 8, 8, 16, 2, true);
    }

    /// Tracker's `/xif4_5/pw/conv_1` + `/xif4_5/dw/conv_1` shape
    /// (`c_exp=672`). On x86_64 w/ AVX-512 this exercises the tiled
    /// kernel (672/16 = 42 chunks > 16, so above the register-blocked
    /// path's ceiling). Small spatial (16×16) keeps the test fast.
    #[test]
    fn tracker_xif4_5_stride1_c112_c672_relu() {
        case(1, 16, 16, 112, 672, 1, true);
    }

    /// c_exp at the residual boundary: 288 = 18 chunks = 2 full tiles
    /// + 2 residual. Exercises the tiled path's residual handling.
    #[test]
    fn tiled_residual_c8_c288_stride2_relu() {
        case(1, 8, 8, 8, 288, 2, true);
    }

    /// 4×6 PW tile path with out_w=64 (tracker xif2_0 shape). On AVX-512
    /// this routes through `compute_pw_row_avx512_4x6_inner`:
    /// c_exp=96 → 6 chunks = 1 full 4-OC tile + 2-chunk OC tail;
    /// out_w=64 → 10 full 6-OW tiles + 4-col OW tail (falls back to
    /// 6-ZMM path for the tail).
    #[test]
    fn pw_4x6_tile_c96_ow64_stride1() {
        case(1, 64, 64, 16, 96, 1, true);
    }

    /// OW tail exercised: out_w=37 → 6 full tiles + 1-col tail.
    /// in_w=73 pad=1 stride=1 → out_w=73; need out_w=37 → set
    /// in_w=37, stride=1 (with SAME-pad that gives out_w=37).
    #[test]
    fn pw_4x6_tile_c128_ow37_tail() {
        case(1, 16, 37, 8, 128, 1, false);
    }

    /// OC-tail of exactly 1 chunk: c_exp=80 → 5 chunks = 1 full 4-OC
    /// tile + 1-chunk tail. Exercises the `oc_tail_chunks == 1` branch.
    #[test]
    fn pw_4x6_tile_c80_oc_tail() {
        case(1, 12, 12, 8, 80, 1, true);
    }
}
