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

/// Per-phase profiler for the streaming PW-expand→DW→PW-reduce kernel, gated
/// by `YSCV_DW_PROFILE=1`. Measures the real PW/DW/reduce split inside the
/// fused kernel in model context (isolated microbenches mislead). Zero
/// overhead when disabled.
pub mod dw_prof {
    use std::sync::OnceLock;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::Instant;

    pub(super) static PW_EXPAND_NS: AtomicU64 = AtomicU64::new(0);
    pub(super) static DW_NS: AtomicU64 = AtomicU64::new(0);
    pub(super) static PW_REDUCE_NS: AtomicU64 = AtomicU64::new(0);

    #[inline]
    pub(super) fn enabled() -> bool {
        static C: OnceLock<bool> = OnceLock::new();
        *C.get_or_init(|| std::env::var_os("YSCV_DW_PROFILE").is_some())
    }

    #[inline]
    pub(super) fn start() -> Option<Instant> {
        enabled().then(Instant::now)
    }

    #[inline]
    pub(super) fn add(slot: &AtomicU64, t: Option<Instant>) {
        if let Some(t) = t {
            slot.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
        }
    }

    /// `(pw_expand_ns, dw_ns, pw_reduce_ns)` accumulated since the last reset.
    pub fn snapshot() -> (u64, u64, u64) {
        (
            PW_EXPAND_NS.load(Ordering::Relaxed),
            DW_NS.load(Ordering::Relaxed),
            PW_REDUCE_NS.load(Ordering::Relaxed),
        )
    }

    /// Resets the phase counters.
    pub fn reset() {
        PW_EXPAND_NS.store(0, Ordering::Relaxed);
        DW_NS.store(0, Ordering::Relaxed);
        PW_REDUCE_NS.store(0, Ordering::Relaxed);
    }
}

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

    // For the tiled AVX-512 path (c_exp > 256), pre-pack pw_weight into
    // [tile, c_in, TILED_CHUNKS*16] tile-major layout. CI inner-loop stride
    // drops from 2688 B → 128 B so the L1 hardware prefetcher covers every
    // access. Packed once; all batch items and rayon workers share it via &[f32].
    //
    // Per-thread pack buffer reused across calls. Without this, each call
    // freshly allocates the packed weights (up to ~300 KB) on the hot path.
    // Mirrors `PACKED_B_CACHE` in `matmul.rs` and grows monotonically to the
    // largest size seen. We `take` the buffer at entry (leaving an empty Vec in
    // TLS) so the slice it owns stays valid across the body, then put it back
    // before return.
    thread_local! {
        static PACK_BUF: std::cell::RefCell<Vec<f32>> =
            const { std::cell::RefCell::new(Vec::new()) };
    }
    let mut _pw_pack_buf: Vec<f32> = PACK_BUF.with(|cell| std::mem::take(&mut *cell.borrow_mut()));
    let (eff_pw_weight, eff_variant): (&[f32], Variant) = {
        #[cfg(target_arch = "x86_64")]
        {
            if c_exp > AVX512_REG_MAX_CHUNKS * 16
                && c_exp.is_multiple_of(16)
                && !cfg!(miri)
                && crate::host_cpu().features.avx512f
            {
                avx512::pack_pw_weight_tiled(pw_weight, c_in, c_exp, &mut _pw_pack_buf);
                (
                    _pw_pack_buf.as_slice(),
                    Variant {
                        compute_pw_row: if !avx512::tiled4x6_disabled()
                            && !avx512::fullrow_disabled()
                            && in_w <= 16
                        {
                            // Full-row kernel: one weight load per (tile, oc_sub, ci)
                            // shared across all in_w pixels. 4× fewer weight loads than
                            // tiled4x6 for in_w=16 (the tracker hot shape).
                            avx512::compute_pw_row_avx512_fullrow_packed
                        } else if avx512::tiled4x6_disabled() {
                            avx512::compute_pw_row_avx512_tiled_packed
                        } else {
                            avx512::compute_pw_row_avx512_tiled4x6_packed
                        },
                        compute_dw_row: variant.compute_dw_row,
                    },
                )
            } else {
                (
                    pw_weight,
                    Variant {
                        compute_pw_row: variant.compute_pw_row,
                        compute_dw_row: variant.compute_dw_row,
                    },
                )
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            (
                pw_weight,
                Variant {
                    compute_pw_row: variant.compute_pw_row,
                    compute_dw_row: variant.compute_dw_row,
                },
            )
        }
    };

    for ni in 0..batch {
        let batch_in = &input[ni * in_batch_stride..(ni + 1) * in_batch_stride];
        let batch_out = &mut output[ni * out_batch_stride..(ni + 1) * out_batch_stride];

        let run_rows = |out_chunk: &mut [f32], oh_start: usize, oh_end: usize| {
            // Per-worker ring buffer: 3 PW rows × [in_w, c_exp]. Allocated
            // once per chunk, kept L1-hot across the chunk's rows.
            // `AlignedVec::uninitialized` skips the zero-init memset; the
            // ring is fully overwritten by `compute_pw_row` on every slot
            // we ever read (slot bookkeeping prevents reads of unwritten
            // slots), so the initial contents are irrelevant.
            let pw_row_len = in_w * c_exp;
            #[allow(unsafe_code)]
            let mut pw_ring_av = yscv_tensor::AlignedVec::<f32>::uninitialized(3 * pw_row_len);
            let pw_ring: &mut [f32] = pw_ring_av.as_mut_slice();

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
                // The original L1-sized budget over-tiled tracker shapes and
                // paid more in repeated PW rows / tile dispatch than it saved
                // in cache locality. Keep the guard for very wide shapes, but
                // let common tracker rows run full-width or with coarse tiles.
                let target_kb = 256usize;
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
                (eff_variant.compute_pw_row)(
                    src_row,
                    eff_pw_weight,
                    pw_bias,
                    dst_row,
                    iw_lo,
                    iw_hi,
                    c_in,
                    c_exp,
                    pw_relu,
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
                    let row0_ok = ensure_pw_row(ih0, iw_lo, iw_hi, &mut *pw_ring, &mut slot_row_v2);
                    let row1_ok =
                        ensure_pw_row(ih0 + 1, iw_lo, iw_hi, &mut *pw_ring, &mut slot_row_v2);
                    let row2_ok =
                        ensure_pw_row(ih0 + 2, iw_lo, iw_hi, &mut *pw_ring, &mut slot_row_v2);

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
                    (eff_variant.compute_dw_row)(
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
    // Return the (now grown) pack buffer to TLS so the next call on
    // this thread can reuse it. The slice `eff_pw_weight` has been
    // dropped by this point — the closures captured a reborrow that
    // ended when `run_rows` returned.
    PACK_BUF.with(|cell| *cell.borrow_mut() = _pw_pack_buf);
}

// ─── Full-block streaming: PW expand → DW 3×3 → PW reduce ────────────────────
//
// MobileNet-V2 inverted bottleneck. Eliminates the 1.5 MB DRAM intermediate
// between the PW-expand+DW step and the subsequent PW-reduce by streaming:
// each DW output row is written to a small TLS scratch (out_w × c_exp × 4 ≤
// 24 KB for all tracker shapes), then immediately consumed by the PW reduce
// micro-kernel while it is still L1-hot. The c_exp intermediate never spills
// L1 beyond a single row's worth.
//
// Multi-arch: the streaming wrapper itself is arch-agnostic. PW expand + DW
// reuse `fused_pw_expand_dw_3x3`'s existing per-row variants (AVX-512 / AVX2 /
// NEON / scalar). PW reduce reuses `super::conv::pointwise_nx16_direct_rows`
// which dispatches internally to the best arch implementation.
//
// Streaming (rather than a full-tensor PW+DW → PW-reduce wrapper) keeps the
// c_exp intermediate at out_w × c_exp × 4 bytes — one row, L1-resident —
// instead of materializing the whole expansion to DRAM.

use super::conv::pointwise_nx16_direct_rows;

/// Arguments for [`fused_pw_expand_dw_pw_reduce_3x3`].
pub struct FusedPwDwPwReduce<'a> {
    pub input: &'a [f32],
    /// PW expand weight, KHWC `[c_in * c_exp]` flat (loader's `khwc_weights` form).
    pub pw_expand_weight: &'a [f32],
    pub pw_expand_bias: Option<&'a [f32]>,
    /// DW weight, `[9 * c_exp]` flat (loader's `dw_khwc_weights` form).
    pub dw_weight: &'a [f32],
    pub dw_bias: Option<&'a [f32]>,
    /// PW reduce weight prepacked as `[c_exp, c_out_padded]` row-major with
    /// the `c_out_padded - c_out` tail lanes zero-filled. Build via
    /// [`pack_pw_reduce_weight_for_fusion`].
    pub pw_reduce_weight_packed: &'a [f32],
    /// PW reduce bias of length `c_out_padded`, tail zero-filled. Build via
    /// [`pack_pw_reduce_bias_for_fusion`].
    pub pw_reduce_bias: Option<&'a [f32]>,
    /// Optional residual `[batch * out_h * out_w * c_out]` added inside the
    /// PW reduce step (Conv_Add fusion). Slice indexing parallels `output`.
    pub residual: Option<&'a [f32]>,
    /// Output, `[batch * out_h * out_w * c_out]`.
    pub output: &'a mut [f32],
    pub batch: usize,
    pub in_h: usize,
    pub in_w: usize,
    pub c_in: usize,
    pub c_exp: usize,
    /// Logical output channel count.
    pub c_out: usize,
    /// `c_out` rounded up to the next multiple of 16 (kernel inner stride).
    pub c_out_padded: usize,
    pub stride: usize,
    pub pw_expand_activation: Activation,
    pub dw_activation: Activation,
    pub pw_reduce_activation: Activation,
    pub thread_pool: Option<&'a ThreadPool>,
}

/// Streaming MobileNet-V2 inverted bottleneck: PW expand → DW 3×3 → PW reduce.
/// See module-level comment above for the cache-locality argument.
#[allow(clippy::too_many_arguments)]
pub fn fused_pw_expand_dw_pw_reduce_3x3(args: FusedPwDwPwReduce<'_>) {
    let FusedPwDwPwReduce {
        input,
        pw_expand_weight,
        pw_expand_bias,
        dw_weight,
        dw_bias,
        pw_reduce_weight_packed,
        pw_reduce_bias,
        residual,
        output,
        batch,
        in_h,
        in_w,
        c_in,
        c_exp,
        c_out,
        c_out_padded,
        stride,
        pw_expand_activation,
        dw_activation,
        pw_reduce_activation,
        thread_pool,
    } = args;

    debug_assert_eq!(input.len(), batch * in_h * in_w * c_in);
    debug_assert_eq!(pw_expand_weight.len(), c_in * c_exp);
    debug_assert_eq!(dw_weight.len(), 9 * c_exp);
    debug_assert_eq!(pw_reduce_weight_packed.len(), c_exp * c_out_padded);
    debug_assert!(stride == 1 || stride == 2);
    debug_assert!(c_out_padded >= c_out);
    debug_assert!(c_out_padded.is_multiple_of(16));
    if let Some(b) = pw_expand_bias {
        debug_assert_eq!(b.len(), c_exp);
    }
    if let Some(b) = dw_bias {
        debug_assert_eq!(b.len(), c_exp);
    }
    if let Some(b) = pw_reduce_bias {
        debug_assert_eq!(b.len(), c_out_padded);
    }

    let pad: usize = 1;
    let out_h = (in_h + 2 * pad - 3) / stride + 1;
    let out_w = (in_w + 2 * pad - 3) / stride + 1;
    debug_assert_eq!(output.len(), batch * out_h * out_w * c_out);
    if let Some(res) = residual {
        debug_assert_eq!(res.len(), batch * out_h * out_w * c_out);
    }

    let out_row_stride_final = out_w * c_out;
    let out_batch_stride_final = out_h * out_row_stride_final;
    let in_batch_stride = in_h * in_w * c_in;

    let pw_relu = matches!(pw_expand_activation, Activation::Relu);
    let dw_relu = matches!(dw_activation, Activation::Relu);

    let variant = select_variant(c_exp);

    thread_local! {
        static PACK_BUF: std::cell::RefCell<Vec<f32>> =
            const { std::cell::RefCell::new(Vec::new()) };
    }
    let mut _pw_pack_buf: Vec<f32> = PACK_BUF.with(|cell| std::mem::take(&mut *cell.borrow_mut()));
    let (eff_pw_weight, eff_variant): (&[f32], Variant) = {
        #[cfg(target_arch = "x86_64")]
        {
            if c_exp > AVX512_REG_MAX_CHUNKS * 16
                && c_exp.is_multiple_of(16)
                && !cfg!(miri)
                && crate::host_cpu().features.avx512f
            {
                avx512::pack_pw_weight_tiled(pw_expand_weight, c_in, c_exp, &mut _pw_pack_buf);
                (
                    _pw_pack_buf.as_slice(),
                    Variant {
                        compute_pw_row: if !avx512::tiled4x6_disabled()
                            && !avx512::fullrow_disabled()
                            && in_w <= 16
                        {
                            avx512::compute_pw_row_avx512_fullrow_packed
                        } else if avx512::tiled4x6_disabled() {
                            avx512::compute_pw_row_avx512_tiled_packed
                        } else {
                            avx512::compute_pw_row_avx512_tiled4x6_packed
                        },
                        compute_dw_row: variant.compute_dw_row,
                    },
                )
            } else {
                (
                    pw_expand_weight,
                    Variant {
                        compute_pw_row: variant.compute_pw_row,
                        compute_dw_row: variant.compute_dw_row,
                    },
                )
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            (
                pw_expand_weight,
                Variant {
                    compute_pw_row: variant.compute_pw_row,
                    compute_dw_row: variant.compute_dw_row,
                },
            )
        }
    };

    let pw_reduce_act = pw_reduce_activation;
    let pad_needed = c_out_padded != c_out;

    for ni in 0..batch {
        let batch_in = &input[ni * in_batch_stride..(ni + 1) * in_batch_stride];
        let batch_out_final =
            &mut output[ni * out_batch_stride_final..(ni + 1) * out_batch_stride_final];
        let batch_res = residual
            .map(|res| &res[ni * out_batch_stride_final..(ni + 1) * out_batch_stride_final]);

        let run_rows = |out_chunk: &mut [f32], oh_start: usize, oh_end: usize| {
            let pw_row_len = in_w * c_exp;
            let pw_ring_size = 3 * pw_row_len;
            let dw_row_size = out_w * c_exp;
            let out_row_size = if pad_needed { out_w * c_out_padded } else { 0 };

            // `AlignedVec::uninitialized` skips the zero-init memset that
            // `vec![0.0; N]` pays. Inner kernels fully overwrite the bytes
            // they read later: PW writes its slot, DW writes the row, PW
            // reduce writes the row. Slot bookkeeping (`slot_row_v2`) keeps
            // us from reading an unwritten slot.
            #[allow(unsafe_code)]
            let mut pw_ring_av = yscv_tensor::AlignedVec::<f32>::uninitialized(pw_ring_size);
            #[allow(unsafe_code)]
            let mut dw_row_av = yscv_tensor::AlignedVec::<f32>::uninitialized(dw_row_size);
            #[allow(unsafe_code)]
            let mut out_row_av = yscv_tensor::AlignedVec::<f32>::uninitialized(out_row_size);
            let pw_ring: &mut [f32] = pw_ring_av.as_mut_slice();
            let dw_row_scratch: &mut [f32] = dw_row_av.as_mut_slice();
            let out_row_scratch: &mut [f32] = out_row_av.as_mut_slice();

            let mut slot_row_v2: [Option<(usize, usize, usize)>; 3] = [None, None, None];

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
                if let Some((cached_ih, cached_lo, cached_hi)) = slot_row[slot]
                    && cached_ih == ih_u
                    && cached_lo <= iw_lo
                    && cached_hi >= iw_hi
                {
                    return true;
                }
                let src_row = &batch_in[ih_u * in_w * c_in..(ih_u + 1) * in_w * c_in];
                let dst_row = &mut pw_ring[slot * pw_row_len..(slot + 1) * pw_row_len];
                (eff_variant.compute_pw_row)(
                    src_row,
                    eff_pw_weight,
                    pw_expand_bias,
                    dst_row,
                    iw_lo,
                    iw_hi,
                    c_in,
                    c_exp,
                    pw_relu,
                );
                slot_row[slot] = Some((ih_u, iw_lo, iw_hi));
                true
            };

            // Full-row PW expansion: streaming variant does NOT use w_tile.
            // Callers gate on (out_w × c_exp × 4 ≤ L1) so the per-row
            // scratch fits L1; no need to tile columns.
            let iw_lo = 0usize;
            let iw_hi = in_w;

            for oh in oh_start..oh_end {
                let ih0 = (oh as i32) * (stride as i32) - (pad as i32);
                let _t_pw = dw_prof::start();
                let row0_ok = ensure_pw_row(ih0, iw_lo, iw_hi, &mut *pw_ring, &mut slot_row_v2);
                let row1_ok = ensure_pw_row(ih0 + 1, iw_lo, iw_hi, &mut *pw_ring, &mut slot_row_v2);
                let row2_ok = ensure_pw_row(ih0 + 2, iw_lo, iw_hi, &mut *pw_ring, &mut slot_row_v2);
                dw_prof::add(&dw_prof::PW_EXPAND_NS, _t_pw);

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

                // DW writes c_exp lanes into the per-row scratch.
                let _t_dw = dw_prof::start();
                (eff_variant.compute_dw_row)(
                    row0,
                    row1,
                    row2,
                    dw_weight,
                    dw_bias,
                    &mut dw_row_scratch[..],
                    in_w,
                    0,
                    out_w,
                    c_exp,
                    stride,
                    pad,
                    dw_relu,
                );
                dw_prof::add(&dw_prof::DW_NS, _t_dw);

                // Immediately consume the L1-hot DW row with PW reduce.
                let _t_red = dw_prof::start();
                let final_row_start = (oh - oh_start) * out_row_stride_final;
                let final_row_end = final_row_start + out_row_stride_final;
                let final_row_slice = &mut out_chunk[final_row_start..final_row_end];
                let res_row_slice = batch_res
                    .as_ref()
                    .map(|res| &res[oh * out_row_stride_final..(oh + 1) * out_row_stride_final]);

                if pad_needed {
                    // PW reduce → scratch (c_out_padded-strided); residual
                    // can't be fed directly because its stride is c_out.
                    // We fold residual into the output AFTER the strided
                    // copy-out.
                    pointwise_nx16_direct_rows(
                        &dw_row_scratch[..],
                        pw_reduce_weight_packed,
                        pw_reduce_bias,
                        None,
                        &mut out_row_scratch[..],
                        out_w,
                        c_exp,
                        c_out_padded,
                        pw_reduce_act,
                    );
                    if let Some(res_row) = res_row_slice {
                        for ow in 0..out_w {
                            let src =
                                &out_row_scratch[ow * c_out_padded..ow * c_out_padded + c_out];
                            let dst_off = ow * c_out;
                            let res_off = ow * c_out;
                            for oc in 0..c_out {
                                final_row_slice[dst_off + oc] = src[oc] + res_row[res_off + oc];
                            }
                        }
                    } else {
                        for ow in 0..out_w {
                            let src =
                                &out_row_scratch[ow * c_out_padded..ow * c_out_padded + c_out];
                            let dst = &mut final_row_slice[ow * c_out..ow * c_out + c_out];
                            dst.copy_from_slice(src);
                        }
                    }
                } else {
                    // c_out aligned: write directly. The kernel honours
                    // `residual` natively (Conv_Add fusion inlined).
                    pointwise_nx16_direct_rows(
                        &dw_row_scratch[..],
                        pw_reduce_weight_packed,
                        pw_reduce_bias,
                        res_row_slice,
                        final_row_slice,
                        out_w,
                        c_exp,
                        c_out_padded,
                        pw_reduce_act,
                    );
                }
                dw_prof::add(&dw_prof::PW_REDUCE_NS, _t_red);
            }
        };

        let par_min_rows = 4;
        if !cfg!(miri)
            && out_h >= par_min_rows
            && let Some(pool) = thread_pool
        {
            let nthreads = pool.current_num_threads().max(1);
            if nthreads <= 1 {
                run_rows(batch_out_final, 0, out_h);
            } else {
                let rows_per_chunk = out_h.div_ceil(nthreads).max(1);
                let bytes_per_chunk = rows_per_chunk * out_row_stride_final;
                pool.install(|| {
                    batch_out_final
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
            let nthreads = rayon::current_num_threads().max(1);
            if nthreads <= 1 {
                run_rows(batch_out_final, 0, out_h);
            } else {
                let rows_per_chunk = out_h.div_ceil(nthreads).max(1);
                let bytes_per_chunk = rows_per_chunk * out_row_stride_final;
                batch_out_final
                    .par_chunks_mut(bytes_per_chunk)
                    .enumerate()
                    .for_each(|(chunk_idx, chunk)| {
                        let oh_start = chunk_idx * rows_per_chunk;
                        let oh_end = (oh_start + rows_per_chunk).min(out_h);
                        run_rows(chunk, oh_start, oh_end);
                    });
            }
        } else {
            run_rows(batch_out_final, 0, out_h);
        }
    }
    PACK_BUF.with(|cell| *cell.borrow_mut() = _pw_pack_buf);
}

/// NCHWc-aware streaming-fused PW-expand → DW3×3 → PW-reduce kernel
/// (experimental, AVX-512, 1T). The internal c_exp intermediate is held in a
/// NCHWc-blocked ring buffer `[3, c_blocks, w, 16]` instead of the existing
/// NHWC-interleaved `[3, w, c_exp]`. Goal: lift the row-by-row DW from the
/// NHWC strided-read floor and remove c_exp gather/scatter in PW-reduce.
///
/// Output is NHWC `[batch, out_h, out_w, c_out]` for now (downstream consumes
/// NHWC); leaving NCHWc-output would require a NHWC-write variant of
/// `compute_pw_reduce_row_nchwc_avx512`.
///
/// Stride 1 only, c_exp % 16 == 0, c_out_padded % 16 == 0. Single-thread.
#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
pub fn fused_pw_expand_dw_pw_reduce_3x3_nchwc_streaming(
    input: &[f32],            // NHWC [1, in_h, in_w, c_in]
    pw_expand_weight: &[f32], // KHWC [c_in * c_exp]
    pw_expand_bias: Option<&[f32]>,
    dw_weight_blocked: &[f32],       // [c_blocks][9][16]
    _dw_bias: Option<&[f32]>,        // [c_exp] (not used by blocked DW kernel yet)
    pw_reduce_weight_packed: &[f32], // [c_exp][c_out_padded]
    pw_reduce_bias: Option<&[f32]>,  // [c_out_padded]
    output: &mut [f32],              // NHWC [1, out_h, out_w, c_out]
    in_h: usize,
    in_w: usize,
    c_in: usize,
    c_exp: usize,
    c_out: usize,
    c_out_padded: usize,
    pw_expand_relu: bool,
    dw_relu: bool,
    pw_reduce_relu: bool,
) {
    debug_assert!(c_exp.is_multiple_of(16));
    debug_assert!(c_out_padded.is_multiple_of(16));
    debug_assert!(c_out_padded >= c_out);
    let pad: usize = 1;
    let out_h = in_h + 2 * pad - 3 + 1;
    let out_w = in_w + 2 * pad - 3 + 1;
    let c_blocks = c_exp / 16;
    let pw_blocked_row_len = c_blocks * in_w * 16;
    let dw_blocked_row_len = c_blocks * out_w * 16;
    let pw_reduce_row_len = out_w * c_out_padded;
    let out_row_stride = out_w * c_out;

    // NCHWc-blocked ring buffer (3 rows). Plus DW output row and PW-reduce
    // scratch row (padded c_out → trim on copy-out).
    #[allow(unsafe_code)]
    let mut pw_ring = yscv_tensor::AlignedVec::<f32>::uninitialized(3 * pw_blocked_row_len);
    #[allow(unsafe_code)]
    let mut dw_row = yscv_tensor::AlignedVec::<f32>::uninitialized(dw_blocked_row_len);
    #[allow(unsafe_code)]
    let mut pw_red_row = yscv_tensor::AlignedVec::<f32>::uninitialized(pw_reduce_row_len);
    let pw_ring_slice = pw_ring.as_mut_slice();
    let dw_row_slice = dw_row.as_mut_slice();
    let pw_red_row_slice = pw_red_row.as_mut_slice();
    let mut slot_filled = [false; 3];
    let mut slot_row_idx = [usize::MAX; 3];

    let ensure_pw_row = |ih: i32,
                         ring: &mut [f32],
                         filled: &mut [bool; 3],
                         slot_row_idx: &mut [usize; 3]|
     -> bool {
        if ih < 0 || (ih as usize) >= in_h {
            return false;
        }
        let ih_u = ih as usize;
        let slot = ih_u % 3;
        if filled[slot] && slot_row_idx[slot] == ih_u {
            return true;
        }
        let src_row = &input[ih_u * in_w * c_in..(ih_u + 1) * in_w * c_in];
        let dst_row = &mut ring[slot * pw_blocked_row_len..(slot + 1) * pw_blocked_row_len];
        compute_pw_row_nchwc_avx512(
            src_row,
            pw_expand_weight,
            pw_expand_bias,
            dst_row,
            0,
            in_w,
            c_in,
            c_exp,
            pw_expand_relu,
        );
        filled[slot] = true;
        slot_row_idx[slot] = ih_u;
        true
    };

    for oh in 0..out_h {
        let ih0 = oh as i32 - pad as i32;
        let r0_ok = ensure_pw_row(ih0, pw_ring_slice, &mut slot_filled, &mut slot_row_idx);
        let r1_ok = ensure_pw_row(ih0 + 1, pw_ring_slice, &mut slot_filled, &mut slot_row_idx);
        let r2_ok = ensure_pw_row(ih0 + 2, pw_ring_slice, &mut slot_filled, &mut slot_row_idx);
        let row0 = if r0_ok {
            let s = ((ih0 as usize) % 3) * pw_blocked_row_len;
            Some(&pw_ring_slice[s..s + pw_blocked_row_len])
        } else {
            None
        };
        let row1 = if r1_ok {
            let s = (((ih0 + 1) as usize) % 3) * pw_blocked_row_len;
            Some(&pw_ring_slice[s..s + pw_blocked_row_len])
        } else {
            None
        };
        let row2 = if r2_ok {
            let s = (((ih0 + 2) as usize) % 3) * pw_blocked_row_len;
            Some(&pw_ring_slice[s..s + pw_blocked_row_len])
        } else {
            None
        };
        compute_dw_row_nchwc_avx512(
            row0,
            row1,
            row2,
            dw_weight_blocked,
            dw_row_slice,
            out_w,
            c_blocks,
            dw_relu,
        );
        compute_pw_reduce_row_nchwc_avx512(
            dw_row_slice,
            pw_reduce_weight_packed,
            pw_reduce_bias,
            pw_red_row_slice,
            out_w,
            c_exp,
            c_out_padded,
            pw_reduce_relu,
        );
        // Copy-out to NHWC output (trim padded c_out_padded → c_out per pos).
        let out_row_off = oh * out_row_stride;
        if c_out_padded == c_out {
            output[out_row_off..out_row_off + out_row_stride]
                .copy_from_slice(&pw_red_row_slice[..out_row_stride]);
        } else {
            for ow in 0..out_w {
                let src = &pw_red_row_slice[ow * c_out_padded..ow * c_out_padded + c_out];
                let dst_off = out_row_off + ow * c_out;
                output[dst_off..dst_off + c_out].copy_from_slice(src);
            }
        }
    }
}

/// Pack DW weight `[3, 3, c, 1]` KHWC (loader's form) into the
/// NCHWc-blocked layout `[c_blocks][9][16]` expected by
/// [`compute_dw_row_nchwc_avx512`] and
/// [`fused_pw_expand_dw_pw_reduce_3x3_nchwc_streaming`]. x86_64-only: the
/// only consumers are the AVX-512 NCHWc streaming kernels.
#[cfg(target_arch = "x86_64")]
pub fn pack_dw_weight_nchwc_blocked(khwc: &[f32], c: usize) -> Vec<f32> {
    debug_assert_eq!(khwc.len(), 9 * c);
    let c_blocks = c.div_ceil(16);
    let mut out = vec![0.0f32; c_blocks * 9 * 16];
    for cb in 0..c_blocks {
        let c_start = cb * 16;
        let c_take = 16.min(c - c_start);
        for tap in 0..9usize {
            // khwc layout: index by [ky][kx][c] = tap*c + c_start
            let src_off = tap * c + c_start;
            let dst_off = cb * 9 * 16 + tap * 16;
            out[dst_off..dst_off + c_take].copy_from_slice(&khwc[src_off..src_off + c_take]);
        }
    }
    out
}

/// Pack KHWC PW-reduce weight `[c_out, 1, 1, c_exp]` (flat `c_out * c_exp` f32,
/// the loader's standard form for 1×1 conv) into the row-major
/// `[c_exp, c_out_padded]` layout expected by [`fused_pw_expand_dw_pw_reduce_3x3`].
/// The trailing `c_out_padded - c_out` lanes per row are zero-filled so the
/// AVX-512 inner kernel can write 16-lane chunks without explicit tail logic.
///
/// `c_out_padded` must equal the next multiple of 16 ≥ `c_out`.
pub fn pack_pw_reduce_weight_for_fusion(
    khwc_weight: &[f32],
    c_out: usize,
    c_exp: usize,
    c_out_padded: usize,
) -> yscv_tensor::AlignedVec<f32> {
    assert_eq!(khwc_weight.len(), c_out * c_exp);
    assert!(c_out_padded >= c_out);
    assert!(c_out_padded.is_multiple_of(16));
    let mut packed = yscv_tensor::AlignedVec::<f32>::calloc(c_exp * c_out_padded);
    let dst = packed.as_mut_slice();
    for cx in 0..c_exp {
        for oc in 0..c_out {
            dst[cx * c_out_padded + oc] = khwc_weight[oc * c_exp + cx];
        }
        // The remaining c_out_padded - c_out lanes are already 0 from calloc.
    }
    packed
}

/// Pad a `[c_out]` bias to `[c_out_padded]` with zeros in the tail. Returns
/// `None` when the input bias is `None`.
pub fn pack_pw_reduce_bias_for_fusion(
    bias: Option<&[f32]>,
    c_out: usize,
    c_out_padded: usize,
) -> Option<yscv_tensor::AlignedVec<f32>> {
    let bias = bias?;
    assert_eq!(bias.len(), c_out);
    assert!(c_out_padded >= c_out);
    let mut padded = yscv_tensor::AlignedVec::<f32>::calloc(c_out_padded);
    padded.as_mut_slice()[..c_out].copy_from_slice(bias);
    Some(padded)
}

/// 5×5 sibling of [`fused_pw_expand_dw_pw_reduce_3x3`] — same streaming
/// strategy with a 5-row PW ring buffer and the 5-tap DW row writer. Per-row
/// `out_w × c_exp × 4` scratch must fit L1 (caller gates on shape).
///
/// The `oc_tiled` path used by the regular `fused_pw_expand_dw_5x5` for very
/// wide `c_exp > 256` is NOT inherited — that path materialises full output
/// rows per tile, which defeats streaming. Callers that want full-block fusion
/// at wide `c_exp` must use chained `FusedPwDw + Conv` instead.
#[allow(clippy::too_many_arguments)]
pub fn fused_pw_expand_dw_pw_reduce_5x5(args: FusedPwDwPwReduce<'_>) {
    let FusedPwDwPwReduce {
        input,
        pw_expand_weight,
        pw_expand_bias,
        dw_weight,
        dw_bias,
        pw_reduce_weight_packed,
        pw_reduce_bias,
        residual,
        output,
        batch,
        in_h,
        in_w,
        c_in,
        c_exp,
        c_out,
        c_out_padded,
        stride,
        pw_expand_activation,
        dw_activation,
        pw_reduce_activation,
        thread_pool,
    } = args;

    debug_assert_eq!(input.len(), batch * in_h * in_w * c_in);
    debug_assert_eq!(pw_expand_weight.len(), c_in * c_exp);
    debug_assert_eq!(dw_weight.len(), 25 * c_exp);
    debug_assert_eq!(pw_reduce_weight_packed.len(), c_exp * c_out_padded);
    debug_assert!(stride == 1 || stride == 2);
    debug_assert!(c_out_padded >= c_out);
    debug_assert!(c_out_padded.is_multiple_of(16));
    if let Some(b) = pw_expand_bias {
        debug_assert_eq!(b.len(), c_exp);
    }
    if let Some(b) = dw_bias {
        debug_assert_eq!(b.len(), c_exp);
    }
    if let Some(b) = pw_reduce_bias {
        debug_assert_eq!(b.len(), c_out_padded);
    }

    let pad: usize = 2; // 5×5 SAME-pad
    let out_h = (in_h + 2 * pad - 5) / stride + 1;
    let out_w = (in_w + 2 * pad - 5) / stride + 1;
    debug_assert_eq!(output.len(), batch * out_h * out_w * c_out);
    if let Some(res) = residual {
        debug_assert_eq!(res.len(), batch * out_h * out_w * c_out);
    }

    let out_row_stride_final = out_w * c_out;
    let out_batch_stride_final = out_h * out_row_stride_final;
    let in_batch_stride = in_h * in_w * c_in;

    let pw_relu = matches!(pw_expand_activation, Activation::Relu);
    let dw_relu = matches!(dw_activation, Activation::Relu);

    let variant = select_variant(c_exp);
    let dw5_variant = select_dw5_variant(c_exp);

    let pw_reduce_act = pw_reduce_activation;
    let pad_needed = c_out_padded != c_out;

    for ni in 0..batch {
        let batch_in = &input[ni * in_batch_stride..(ni + 1) * in_batch_stride];
        let batch_out_final =
            &mut output[ni * out_batch_stride_final..(ni + 1) * out_batch_stride_final];
        let batch_res = residual
            .map(|res| &res[ni * out_batch_stride_final..(ni + 1) * out_batch_stride_final]);

        let run_rows = |out_chunk: &mut [f32], oh_start: usize, oh_end: usize| {
            let pw_row_len = in_w * c_exp;
            let pw_ring_size = 5 * pw_row_len;
            let dw_row_size = out_w * c_exp;
            let out_row_size = if pad_needed { out_w * c_out_padded } else { 0 };

            #[allow(unsafe_code)]
            let mut pw_ring_av = yscv_tensor::AlignedVec::<f32>::uninitialized(pw_ring_size);
            #[allow(unsafe_code)]
            let mut dw_row_av = yscv_tensor::AlignedVec::<f32>::uninitialized(dw_row_size);
            #[allow(unsafe_code)]
            let mut out_row_av = yscv_tensor::AlignedVec::<f32>::uninitialized(out_row_size);
            let pw_ring: &mut [f32] = pw_ring_av.as_mut_slice();
            let dw_row_scratch: &mut [f32] = dw_row_av.as_mut_slice();
            let out_row_scratch: &mut [f32] = out_row_av.as_mut_slice();

            let mut slot_row: [Option<(usize, usize, usize)>; 5] = [None; 5];

            let ensure_pw_row = |ih: i32,
                                 iw_lo: usize,
                                 iw_hi: usize,
                                 pw_ring: &mut [f32],
                                 slot_row: &mut [Option<(usize, usize, usize)>; 5]|
             -> bool {
                if ih < 0 || (ih as usize) >= in_h {
                    return false;
                }
                let ih_u = ih as usize;
                let slot = ih_u % 5;
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
                    src_row,
                    pw_expand_weight,
                    pw_expand_bias,
                    dst_row,
                    iw_lo,
                    iw_hi,
                    c_in,
                    c_exp,
                    pw_relu,
                );
                slot_row[slot] = Some((ih_u, iw_lo, iw_hi));
                true
            };

            // Full-row PW expansion (no column tile): caller gates on L1-fit.
            let iw_lo = 0usize;
            let iw_hi = in_w;

            for oh in oh_start..oh_end {
                let ih0 = (oh as i32) * (stride as i32) - (pad as i32);
                let rows_ok = [
                    ensure_pw_row(ih0, iw_lo, iw_hi, &mut *pw_ring, &mut slot_row),
                    ensure_pw_row(ih0 + 1, iw_lo, iw_hi, &mut *pw_ring, &mut slot_row),
                    ensure_pw_row(ih0 + 2, iw_lo, iw_hi, &mut *pw_ring, &mut slot_row),
                    ensure_pw_row(ih0 + 3, iw_lo, iw_hi, &mut *pw_ring, &mut slot_row),
                    ensure_pw_row(ih0 + 4, iw_lo, iw_hi, &mut *pw_ring, &mut slot_row),
                ];
                let row_refs: [Option<&[f32]>; 5] = std::array::from_fn(|ky| {
                    if !rows_ok[ky] {
                        return None;
                    }
                    let ih = ih0 + ky as i32;
                    let slot = (ih as usize) % 5;
                    let start = slot * pw_row_len;
                    Some(&pw_ring[start..start + pw_row_len])
                });

                // DW 5×5 writes c_exp lanes into per-row scratch.
                (dw5_variant.compute_dw_row)(Dw5RowCtx {
                    rows: row_refs,
                    dw_weight,
                    dw_bias,
                    out_row: &mut dw_row_scratch[..],
                    in_w,
                    ow_start: 0,
                    ow_end: out_w,
                    c_exp,
                    stride,
                    pad,
                    relu: dw_relu,
                });

                // Immediately consume the L1-hot DW row with PW reduce.
                let final_row_start = (oh - oh_start) * out_row_stride_final;
                let final_row_end = final_row_start + out_row_stride_final;
                let final_row_slice = &mut out_chunk[final_row_start..final_row_end];
                let res_row_slice = batch_res
                    .as_ref()
                    .map(|res| &res[oh * out_row_stride_final..(oh + 1) * out_row_stride_final]);

                if pad_needed {
                    pointwise_nx16_direct_rows(
                        &dw_row_scratch[..],
                        pw_reduce_weight_packed,
                        pw_reduce_bias,
                        None,
                        &mut out_row_scratch[..],
                        out_w,
                        c_exp,
                        c_out_padded,
                        pw_reduce_act,
                    );
                    if let Some(res_row) = res_row_slice {
                        for ow in 0..out_w {
                            let src =
                                &out_row_scratch[ow * c_out_padded..ow * c_out_padded + c_out];
                            let dst_off = ow * c_out;
                            for oc in 0..c_out {
                                final_row_slice[dst_off + oc] = src[oc] + res_row[dst_off + oc];
                            }
                        }
                    } else {
                        for ow in 0..out_w {
                            let src =
                                &out_row_scratch[ow * c_out_padded..ow * c_out_padded + c_out];
                            let dst = &mut final_row_slice[ow * c_out..ow * c_out + c_out];
                            dst.copy_from_slice(src);
                        }
                    }
                } else {
                    pointwise_nx16_direct_rows(
                        &dw_row_scratch[..],
                        pw_reduce_weight_packed,
                        pw_reduce_bias,
                        res_row_slice,
                        final_row_slice,
                        out_w,
                        c_exp,
                        c_out_padded,
                        pw_reduce_act,
                    );
                }
            }
        };

        let par_min_rows = 4;
        if !cfg!(miri)
            && out_h >= par_min_rows
            && let Some(pool) = thread_pool
        {
            let nthreads = pool.current_num_threads().max(1);
            if nthreads <= 1 {
                run_rows(batch_out_final, 0, out_h);
            } else {
                let rows_per_chunk = out_h.div_ceil(nthreads).max(1);
                let bytes_per_chunk = rows_per_chunk * out_row_stride_final;
                pool.install(|| {
                    batch_out_final
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
            let nthreads = rayon::current_num_threads().max(1);
            if nthreads <= 1 {
                run_rows(batch_out_final, 0, out_h);
            } else {
                let rows_per_chunk = out_h.div_ceil(nthreads).max(1);
                let bytes_per_chunk = rows_per_chunk * out_row_stride_final;
                batch_out_final
                    .par_chunks_mut(bytes_per_chunk)
                    .enumerate()
                    .for_each(|(chunk_idx, chunk)| {
                        let oh_start = chunk_idx * rows_per_chunk;
                        let oh_end = (oh_start + rows_per_chunk).min(out_h);
                        run_rows(chunk, oh_start, oh_end);
                    });
            }
        } else {
            run_rows(batch_out_final, 0, out_h);
        }
    }
}

/// Arguments for [`fused_pw_expand_dw_5x5`].
pub struct FusedPwDw5x5<'a> {
    pub input: &'a [f32],
    pub pw_weight: &'a [f32],
    pub pw_bias: Option<&'a [f32]>,
    pub dw_weight: &'a [f32],
    pub dw_bias: Option<&'a [f32]>,
    pub output: &'a mut [f32],
    pub batch: usize,
    pub in_h: usize,
    pub in_w: usize,
    pub c_in: usize,
    pub c_exp: usize,
    pub stride: usize,
    pub pw_activation: Activation,
    pub dw_activation: Activation,
    pub thread_pool: Option<&'a ThreadPool>,
}

/// Streaming PW-expand → DW 5×5 sibling of [`fused_pw_expand_dw_3x3`].
///
/// This covers the tracker's 5×5 inverted-bottleneck openings that otherwise
/// materialise the full PW-expanded tensor before depthwise. PW computation
/// reuses the same multi-arch row kernels as the 3×3 path; the DW stage has
/// scalar, AVX2/FMA, AVX-512, and NEON row kernels behind runtime dispatch.
pub fn fused_pw_expand_dw_5x5(args: FusedPwDw5x5<'_>) {
    let FusedPwDw5x5 {
        input,
        pw_weight,
        pw_bias,
        dw_weight,
        dw_bias,
        output,
        batch,
        in_h,
        in_w,
        c_in,
        c_exp,
        stride,
        pw_activation,
        dw_activation,
        thread_pool,
    } = args;
    debug_assert_eq!(input.len(), batch * in_h * in_w * c_in);
    debug_assert_eq!(pw_weight.len(), c_in * c_exp);
    debug_assert_eq!(dw_weight.len(), 25 * c_exp);
    debug_assert!(stride == 1 || stride == 2);
    if let Some(b) = pw_bias {
        debug_assert_eq!(b.len(), c_exp);
    }
    if let Some(b) = dw_bias {
        debug_assert_eq!(b.len(), c_exp);
    }

    let pad: usize = 2;
    let out_h = (in_h + 2 * pad - 5) / stride + 1;
    let out_w = (in_w + 2 * pad - 5) / stride + 1;
    debug_assert_eq!(output.len(), batch * out_h * out_w * c_exp);

    if c_exp > 256 {
        return fused_pw_expand_dw_5x5_oc_tiled(FusedPwDw5x5 {
            input,
            pw_weight,
            pw_bias,
            dw_weight,
            dw_bias,
            output,
            batch,
            in_h,
            in_w,
            c_in,
            c_exp,
            stride,
            pw_activation,
            dw_activation,
            thread_pool,
        });
    }

    let out_row_stride = out_w * c_exp;
    let out_batch_stride = out_h * out_row_stride;
    let in_batch_stride = in_h * in_w * c_in;
    let pw_relu = matches!(pw_activation, Activation::Relu);
    let dw_relu = matches!(dw_activation, Activation::Relu);
    let variant = select_variant(c_exp);
    let dw5_variant = select_dw5_variant(c_exp);

    for ni in 0..batch {
        let batch_in = &input[ni * in_batch_stride..(ni + 1) * in_batch_stride];
        let batch_out = &mut output[ni * out_batch_stride..(ni + 1) * out_batch_stride];

        let run_rows = |out_chunk: &mut [f32], oh_start: usize, oh_end: usize| {
            let pw_row_len = in_w * c_exp;
            #[allow(unsafe_code)]
            let mut pw_ring_av = yscv_tensor::AlignedVec::<f32>::uninitialized(5 * pw_row_len);
            let pw_ring: &mut [f32] = pw_ring_av.as_mut_slice();

            let pick_w_tile = || -> usize {
                static W_TILE_OVERRIDE: std::sync::OnceLock<Option<usize>> =
                    std::sync::OnceLock::new();
                let w_tile_override = *W_TILE_OVERRIDE.get_or_init(|| {
                    std::env::var("YSCV_FUSED_PW_DW_5X5_W_TILE")
                        .ok()
                        .and_then(|v| v.parse::<usize>().ok())
                        .filter(|&v| v >= 4)
                });
                if let Some(v) = w_tile_override {
                    return v.min(out_w).max(4);
                }
                let target_kb = 256usize;
                let per_col_bytes = c_exp * 5 * std::mem::size_of::<f32>();
                let max_iw_cols = (target_kb * 1024) / per_col_bytes.max(1);
                let max_ow_cols = max_iw_cols / stride.max(1);
                if max_ow_cols >= out_w {
                    out_w
                } else if max_ow_cols >= 32 {
                    32
                } else if max_ow_cols >= 16 {
                    16
                } else if max_ow_cols >= 8 {
                    8
                } else {
                    4
                }
            };
            let w_tile = pick_w_tile();

            let ensure_pw_row = |ih: i32,
                                 iw_lo: usize,
                                 iw_hi: usize,
                                 pw_ring: &mut [f32],
                                 slot_row: &mut [Option<(usize, usize, usize)>; 5]|
             -> bool {
                if ih < 0 || (ih as usize) >= in_h {
                    return false;
                }
                let ih_u = ih as usize;
                let slot = ih_u % 5;
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

            let mut slot_row: [Option<(usize, usize, usize)>; 5] = [None, None, None, None, None];
            let mut ow_tile_start = 0usize;
            while ow_tile_start < out_w {
                let ow_tile_end = (ow_tile_start + w_tile).min(out_w);
                let iw_lo = (ow_tile_start as i32 * stride as i32 - pad as i32).max(0) as usize;
                let iw_hi =
                    (((ow_tile_end - 1) as i32) * stride as i32 + 5 - pad as i32).max(0) as usize;
                let iw_hi = iw_hi.min(in_w);
                if ow_tile_start > 0 {
                    slot_row = [None, None, None, None, None];
                }

                for oh in oh_start..oh_end {
                    let ih0 = (oh as i32) * (stride as i32) - (pad as i32);
                    let rows_ok = [
                        ensure_pw_row(ih0, iw_lo, iw_hi, &mut *pw_ring, &mut slot_row),
                        ensure_pw_row(ih0 + 1, iw_lo, iw_hi, &mut *pw_ring, &mut slot_row),
                        ensure_pw_row(ih0 + 2, iw_lo, iw_hi, &mut *pw_ring, &mut slot_row),
                        ensure_pw_row(ih0 + 3, iw_lo, iw_hi, &mut *pw_ring, &mut slot_row),
                        ensure_pw_row(ih0 + 4, iw_lo, iw_hi, &mut *pw_ring, &mut slot_row),
                    ];
                    let row_refs: [Option<&[f32]>; 5] = std::array::from_fn(|ky| {
                        if !rows_ok[ky] {
                            return None;
                        }
                        let ih = ih0 + ky as i32;
                        let slot = (ih as usize) % 5;
                        let start = slot * pw_row_len;
                        Some(&pw_ring[start..start + pw_row_len])
                    });
                    let out_row_slice = &mut out_chunk
                        [(oh - oh_start) * out_row_stride..(oh - oh_start + 1) * out_row_stride];
                    (dw5_variant.compute_dw_row)(Dw5RowCtx {
                        rows: row_refs,
                        dw_weight,
                        dw_bias,
                        out_row: out_row_slice,
                        in_w,
                        ow_start: ow_tile_start,
                        ow_end: ow_tile_end,
                        c_exp,
                        stride,
                        pad,
                        relu: dw_relu,
                    });
                }
                ow_tile_start = ow_tile_end;
            }
        };

        let par_min_rows = 4;
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

fn fused_pw_expand_dw_5x5_oc_tiled(args: FusedPwDw5x5<'_>) {
    let FusedPwDw5x5 {
        input,
        pw_weight,
        pw_bias,
        dw_weight,
        dw_bias,
        output,
        batch,
        in_h,
        in_w,
        c_in,
        c_exp,
        stride,
        pw_activation,
        dw_activation,
        thread_pool,
    } = args;
    let pad: usize = 2;
    let out_h = (in_h + 2 * pad - 5) / stride + 1;
    let out_w = (in_w + 2 * pad - 5) / stride + 1;
    let out_row_stride = out_w * c_exp;
    let out_batch_stride = out_h * out_row_stride;
    let in_batch_stride = in_h * in_w * c_in;
    let pw_relu = matches!(pw_activation, Activation::Relu);
    let dw_relu = matches!(dw_activation, Activation::Relu);
    let tile_variant = select_tile_variant(c_exp);
    let oc_tile = pick_oc_tile(c_exp);

    for ni in 0..batch {
        let batch_in = &input[ni * in_batch_stride..(ni + 1) * in_batch_stride];
        let batch_out = &mut output[ni * out_batch_stride..(ni + 1) * out_batch_stride];

        let run_rows = |out_chunk: &mut [f32], oh_start: usize, oh_end: usize| {
            let mut oc_start = 0usize;
            while oc_start < c_exp {
                let c_tile = (c_exp - oc_start).min(oc_tile);
                let pw_row_len = in_w * c_tile;
                #[allow(unsafe_code)]
                let mut pw_ring_av = yscv_tensor::AlignedVec::<f32>::uninitialized(5 * pw_row_len);
                let pw_ring: &mut [f32] = pw_ring_av.as_mut_slice();
                let mut slot_row: [Option<usize>; 5] = [None, None, None, None, None];

                let ensure_pw_tile =
                    |ih: i32, pw_ring: &mut [f32], slot_row: &mut [Option<usize>; 5]| -> bool {
                        if ih < 0 || (ih as usize) >= in_h {
                            return false;
                        }
                        let ih_u = ih as usize;
                        let slot = ih_u % 5;
                        if slot_row[slot] == Some(ih_u) {
                            return true;
                        }
                        let src_row = &batch_in[ih_u * in_w * c_in..(ih_u + 1) * in_w * c_in];
                        let dst_row = &mut pw_ring[slot * pw_row_len..(slot + 1) * pw_row_len];
                        (tile_variant.compute_pw_tile)(PwTileCtx {
                            src_row,
                            pw_weight,
                            pw_bias,
                            dst_row,
                            iw_start: 0,
                            iw_end: in_w,
                            c_in,
                            c_exp,
                            oc_start,
                            c_tile,
                            relu: pw_relu,
                        });
                        slot_row[slot] = Some(ih_u);
                        true
                    };

                for oh in oh_start..oh_end {
                    let ih0 = (oh as i32) * (stride as i32) - (pad as i32);
                    let rows_ok = [
                        ensure_pw_tile(ih0, &mut *pw_ring, &mut slot_row),
                        ensure_pw_tile(ih0 + 1, &mut *pw_ring, &mut slot_row),
                        ensure_pw_tile(ih0 + 2, &mut *pw_ring, &mut slot_row),
                        ensure_pw_tile(ih0 + 3, &mut *pw_ring, &mut slot_row),
                        ensure_pw_tile(ih0 + 4, &mut *pw_ring, &mut slot_row),
                    ];
                    let row_refs: [Option<&[f32]>; 5] = std::array::from_fn(|ky| {
                        if !rows_ok[ky] {
                            return None;
                        }
                        let ih = ih0 + ky as i32;
                        let slot = (ih as usize) % 5;
                        let start = slot * pw_row_len;
                        Some(&pw_ring[start..start + pw_row_len])
                    });
                    let out_row = &mut out_chunk
                        [(oh - oh_start) * out_row_stride..(oh - oh_start + 1) * out_row_stride];
                    (tile_variant.compute_dw5_tile)(Dw5TileCtx {
                        rows: row_refs,
                        dw_weight,
                        dw_bias,
                        out_row,
                        in_w,
                        ow_start: 0,
                        ow_end: out_w,
                        c_exp,
                        oc_start,
                        c_tile,
                        stride,
                        pad,
                        relu: dw_relu,
                    });
                }

                oc_start += c_tile;
            }
        };

        let par_min_rows = 4;
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

fn pick_oc_tile(c_exp: usize) -> usize {
    static OVERRIDE: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    if let Some(v) = *OVERRIDE.get_or_init(|| {
        std::env::var("YSCV_FUSED_PW_DW_5X5_OC_TILE")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&v| v >= 16)
    }) {
        return v.min(c_exp);
    }
    if c_exp.is_multiple_of(16) {
        128
    } else if c_exp.is_multiple_of(8) {
        128
    } else if c_exp.is_multiple_of(4) {
        64
    } else {
        32
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

struct Dw5RowCtx<'a, 'b> {
    rows: [Option<&'a [f32]>; 5],
    dw_weight: &'a [f32],
    dw_bias: Option<&'a [f32]>,
    out_row: &'b mut [f32],
    in_w: usize,
    ow_start: usize,
    ow_end: usize,
    c_exp: usize,
    stride: usize,
    pad: usize,
    relu: bool,
}

struct Dw5Variant {
    compute_dw_row: for<'a, 'b> fn(Dw5RowCtx<'a, 'b>),
}

struct PwTileCtx<'a, 'b> {
    src_row: &'a [f32],
    pw_weight: &'a [f32],
    pw_bias: Option<&'a [f32]>,
    dst_row: &'b mut [f32],
    iw_start: usize,
    iw_end: usize,
    c_in: usize,
    c_exp: usize,
    oc_start: usize,
    c_tile: usize,
    relu: bool,
}

struct Dw5TileCtx<'a, 'b> {
    rows: [Option<&'a [f32]>; 5],
    dw_weight: &'a [f32],
    dw_bias: Option<&'a [f32]>,
    out_row: &'b mut [f32],
    in_w: usize,
    ow_start: usize,
    ow_end: usize,
    c_exp: usize,
    oc_start: usize,
    c_tile: usize,
    stride: usize,
    pad: usize,
    relu: bool,
}

struct TileVariant {
    compute_pw_tile: for<'a, 'b> fn(PwTileCtx<'a, 'b>),
    compute_dw5_tile: for<'a, 'b> fn(Dw5TileCtx<'a, 'b>),
}

/// Max `c_exp / 16` chunks the AVX-512 register-blocked path supports.
/// 16 ZMMs of accumulators = 256 lanes of `c_exp`. Shapes larger than
/// this spill below the performance floor of AVX2, so we fall back to
/// AVX2 rather than let AVX-512 go memory-bound.
#[cfg(target_arch = "x86_64")]
pub(super) const AVX512_REG_MAX_CHUNKS: usize = 16;

fn select_variant(c_exp: usize) -> Variant {
    // Dispatch reads the unified host CPU identity (`arch::host_cpu`) — the
    // single source of truth for ISA features (and, in later phases, microarch).
    // Selection is capability-first: features gate correctness, so the chosen
    // kernels are unchanged from the prior ad-hoc `is_*_feature_detected!` checks.
    #[cfg(target_arch = "x86_64")]
    {
        let cpu = crate::host_cpu();
        // Register-blocked AVX-512: keeps all `c_exp / 16` accumulators
        // in ZMM across the inner loop. Only eligible while the chunks
        // fit in the ZMM file with room for `x`/`w` temps — capped at
        // 16 chunks (c_exp ≤ 256) which covers tracker shapes
        // 16/96/144/192/256 but not 672.
        if c_exp.is_multiple_of(16)
            && c_exp / 16 <= AVX512_REG_MAX_CHUNKS
            && !cfg!(miri)
            && cpu.features.avx512f
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
        if c_exp.is_multiple_of(16) && !cfg!(miri) && cpu.features.avx512f {
            return Variant {
                compute_pw_row: avx512::compute_pw_row_avx512_tiled,
                compute_dw_row: avx512::compute_dw_row_avx512_tiled,
            };
        }
        if c_exp.is_multiple_of(8) && !cfg!(miri) && cpu.features.avx2 && cpu.features.fma {
            return Variant {
                compute_pw_row: avx2::compute_pw_row_avx2,
                compute_dw_row: avx2::compute_dw_row_avx2,
            };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        let cpu = crate::host_cpu();
        // Microarch refinement hook (later phase): the column-reuse DW asm
        // inside `compute_dw_row_neon` is an in-order optimisation, so
        // `cpu.uarch.is_in_order()` (Cortex-A53/A55) is where a future split
        // selects it vs a plain-NEON variant for out-of-order cores. For now
        // every NEON core takes the same kernel — the asm stays env-gated
        // inside it — so this is behaviourally identical to before.
        if c_exp.is_multiple_of(4) && !cfg!(miri) && cpu.features.neon {
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

/// DW 3×3 NHWC row kernel fn pointer (see `Variant::compute_dw_row`).
/// Exposed for headroom microbenching the L1-hot backbone DW.
pub type Dw3RowFn = fn(
    Option<&[f32]>,
    Option<&[f32]>,
    Option<&[f32]>,
    &[f32],
    Option<&[f32]>,
    &mut [f32],
    usize,
    usize,
    usize,
    usize,
    usize,
    usize,
    bool,
);

/// NCHWc-blocked DW 3×3 row kernel (stride 1, pad 1, block=16). Input/output
/// rows in `[c_block][w][16]` layout. AVX-512, per-channel-block weight-hoisted
/// — building block for the experimental NCHWc-aware streaming-fused kernel
/// [`fused_pw_expand_dw_pw_reduce_3x3_nchwc_streaming`].
///
/// Empirically equal-to-NHWC row kernel for the row-by-row streaming context
/// (no cross-row weight reuse possible). The win comes from removing layout
/// gather/scatter overhead in the PW-reduce step.
#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code)]
pub fn compute_dw_row_nchwc_avx512(
    row0: Option<&[f32]>,
    row1: Option<&[f32]>,
    row2: Option<&[f32]>,
    weight_blocked: &[f32],
    out_row: &mut [f32],
    w: usize,
    c_blocks: usize,
    relu: bool,
) {
    // SAFETY: caller checks avx512f.
    unsafe {
        compute_dw_row_nchwc_avx512_inner(
            row0,
            row1,
            row2,
            weight_blocked,
            out_row,
            w,
            c_blocks,
            relu,
        )
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn, clippy::too_many_arguments)]
unsafe fn compute_dw_row_nchwc_avx512_inner(
    row0: Option<&[f32]>,
    row1: Option<&[f32]>,
    row2: Option<&[f32]>,
    weight_blocked: &[f32],
    out_row: &mut [f32],
    w: usize,
    c_blocks: usize,
    relu: bool,
) {
    use std::arch::x86_64::*;
    debug_assert_eq!(weight_blocked.len(), c_blocks * 9 * 16);
    let present = [row0.is_some(), row1.is_some(), row2.is_some()];
    let rows = [row0, row1, row2];
    let zero = _mm512_setzero_ps();
    for cb in 0..c_blocks {
        let wbase = weight_blocked.as_ptr().add(cb * 9 * 16);
        // Hoist 9 weight ZMMs for this c-block across the whole row.
        let mut wv = [zero; 9];
        for t in 0..9 {
            wv[t] = _mm512_loadu_ps(wbase.add(t * 16));
        }
        let cb_off = cb * w * 16;
        let rp: [*const f32; 3] = [
            rows[0].map_or(std::ptr::null(), |r| r.as_ptr().add(cb_off)),
            rows[1].map_or(std::ptr::null(), |r| r.as_ptr().add(cb_off)),
            rows[2].map_or(std::ptr::null(), |r| r.as_ptr().add(cb_off)),
        ];
        let out_p = out_row.as_mut_ptr().add(cb_off);
        // Border columns (ow=0, ow=w-1) use the bounds-checked path.
        let col = |ow: usize| {
            let mut acc = zero;
            for ky in 0..3usize {
                if !present[ky] {
                    continue;
                }
                let base = rp[ky];
                for kx in 0..3usize {
                    let iw = ow as isize + kx as isize - 1;
                    if iw < 0 || iw as usize >= w {
                        continue;
                    }
                    let x = _mm512_loadu_ps(base.add(iw as usize * 16));
                    acc = _mm512_fmadd_ps(x, wv[ky * 3 + kx], acc);
                }
            }
            if relu {
                acc = _mm512_max_ps(acc, zero);
            }
            _mm512_storeu_ps(out_p.add(ow * 16), acc);
        };
        col(0);
        if w > 1 {
            col(w - 1);
        }
        // Interior columns [1, w-1): branchless 3×3 (skip absent rows).
        for ow in 1..w.saturating_sub(1) {
            let mut acc = zero;
            let ib = (ow - 1) * 16;
            for ky in 0..3usize {
                if !present[ky] {
                    continue;
                }
                let base = rp[ky].add(ib);
                let x0 = _mm512_loadu_ps(base);
                let x1 = _mm512_loadu_ps(base.add(16));
                let x2 = _mm512_loadu_ps(base.add(32));
                acc = _mm512_fmadd_ps(x0, wv[ky * 3], acc);
                acc = _mm512_fmadd_ps(x1, wv[ky * 3 + 1], acc);
                acc = _mm512_fmadd_ps(x2, wv[ky * 3 + 2], acc);
            }
            if relu {
                acc = _mm512_max_ps(acc, zero);
            }
            _mm512_storeu_ps(out_p.add(ow * 16), acc);
        }
    }
}

/// NCHWc-blocked PW-expand row kernel: reads NHWC input row `[w][c_in]`,
/// writes NCHWc-blocked output row `[c_exp_blocks][w][16]`. Weight is the
/// standard KHWC `[c_in][c_exp]` layout (no prepack needed).
///
/// Register-tiled with 8-row × 1-c-block accumulators (8 ZMMs in flight =
/// Zen4 FMA pipe full at 4-cycle latency × 2 ports), mirroring the existing
/// `pointwise_nx16_direct_rows_avx512` NHWC kernel. 4-row and 1-row tails
/// handle w not a multiple of 8.
#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code)]
pub fn compute_pw_row_nchwc_avx512(
    input_row: &[f32], // [w * c_in]
    weight: &[f32],    // [c_in * c_exp]
    bias: Option<&[f32]>,
    out_row: &mut [f32], // [c_exp_blocks * w * 16]
    iw_start: usize,
    iw_end: usize,
    c_in: usize,
    c_exp: usize,
    relu: bool,
) {
    unsafe {
        compute_pw_row_nchwc_avx512_inner(
            input_row, weight, bias, out_row, iw_start, iw_end, c_in, c_exp, relu,
        )
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn, clippy::too_many_arguments)]
unsafe fn compute_pw_row_nchwc_avx512_inner(
    input_row: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    out_row: &mut [f32],
    iw_start: usize,
    iw_end: usize,
    c_in: usize,
    c_exp: usize,
    relu: bool,
) {
    use std::arch::x86_64::*;
    debug_assert!(c_exp.is_multiple_of(16));
    let c_exp_blocks = c_exp / 16;
    let w_total = out_row.len() / (c_exp_blocks * 16);
    let zero = _mm512_setzero_ps();

    // Outer loop: c-block (output channel block). Each iteration produces
    // a contiguous block of the output `[cb, w_idx..w_idx+rows, 16]`.
    for cb in 0..c_exp_blocks {
        let coff = cb * 16;
        let bias_v = if let Some(b) = bias {
            _mm512_loadu_ps(b.as_ptr().add(coff))
        } else {
            zero
        };
        let dst_cb_base = out_row.as_mut_ptr().add(cb * w_total * 16);

        let mut w_idx = iw_start;
        // 8-row tile: 8 ZMM accumulators saturate the Zen4 FMA pipe.
        while w_idx + 8 <= iw_end {
            let mut a0 = bias_v;
            let mut a1 = bias_v;
            let mut a2 = bias_v;
            let mut a3 = bias_v;
            let mut a4 = bias_v;
            let mut a5 = bias_v;
            let mut a6 = bias_v;
            let mut a7 = bias_v;
            let ip = input_row.as_ptr().add(w_idx * c_in);
            for ic in 0..c_in {
                let wv = _mm512_loadu_ps(weight.as_ptr().add(ic * c_exp + coff));
                a0 = _mm512_fmadd_ps(_mm512_set1_ps(*ip.add(ic)), wv, a0);
                a1 = _mm512_fmadd_ps(_mm512_set1_ps(*ip.add(c_in + ic)), wv, a1);
                a2 = _mm512_fmadd_ps(_mm512_set1_ps(*ip.add(2 * c_in + ic)), wv, a2);
                a3 = _mm512_fmadd_ps(_mm512_set1_ps(*ip.add(3 * c_in + ic)), wv, a3);
                a4 = _mm512_fmadd_ps(_mm512_set1_ps(*ip.add(4 * c_in + ic)), wv, a4);
                a5 = _mm512_fmadd_ps(_mm512_set1_ps(*ip.add(5 * c_in + ic)), wv, a5);
                a6 = _mm512_fmadd_ps(_mm512_set1_ps(*ip.add(6 * c_in + ic)), wv, a6);
                a7 = _mm512_fmadd_ps(_mm512_set1_ps(*ip.add(7 * c_in + ic)), wv, a7);
            }
            if relu {
                a0 = _mm512_max_ps(a0, zero);
                a1 = _mm512_max_ps(a1, zero);
                a2 = _mm512_max_ps(a2, zero);
                a3 = _mm512_max_ps(a3, zero);
                a4 = _mm512_max_ps(a4, zero);
                a5 = _mm512_max_ps(a5, zero);
                a6 = _mm512_max_ps(a6, zero);
                a7 = _mm512_max_ps(a7, zero);
            }
            let dst = dst_cb_base.add(w_idx * 16);
            _mm512_storeu_ps(dst, a0);
            _mm512_storeu_ps(dst.add(16), a1);
            _mm512_storeu_ps(dst.add(32), a2);
            _mm512_storeu_ps(dst.add(48), a3);
            _mm512_storeu_ps(dst.add(64), a4);
            _mm512_storeu_ps(dst.add(80), a5);
            _mm512_storeu_ps(dst.add(96), a6);
            _mm512_storeu_ps(dst.add(112), a7);
            w_idx += 8;
        }
        // 4-row tile.
        while w_idx + 4 <= iw_end {
            let mut a0 = bias_v;
            let mut a1 = bias_v;
            let mut a2 = bias_v;
            let mut a3 = bias_v;
            let ip = input_row.as_ptr().add(w_idx * c_in);
            for ic in 0..c_in {
                let wv = _mm512_loadu_ps(weight.as_ptr().add(ic * c_exp + coff));
                a0 = _mm512_fmadd_ps(_mm512_set1_ps(*ip.add(ic)), wv, a0);
                a1 = _mm512_fmadd_ps(_mm512_set1_ps(*ip.add(c_in + ic)), wv, a1);
                a2 = _mm512_fmadd_ps(_mm512_set1_ps(*ip.add(2 * c_in + ic)), wv, a2);
                a3 = _mm512_fmadd_ps(_mm512_set1_ps(*ip.add(3 * c_in + ic)), wv, a3);
            }
            if relu {
                a0 = _mm512_max_ps(a0, zero);
                a1 = _mm512_max_ps(a1, zero);
                a2 = _mm512_max_ps(a2, zero);
                a3 = _mm512_max_ps(a3, zero);
            }
            let dst = dst_cb_base.add(w_idx * 16);
            _mm512_storeu_ps(dst, a0);
            _mm512_storeu_ps(dst.add(16), a1);
            _mm512_storeu_ps(dst.add(32), a2);
            _mm512_storeu_ps(dst.add(48), a3);
            w_idx += 4;
        }
        // 1-row tail.
        while w_idx < iw_end {
            let mut acc = bias_v;
            let ip = input_row.as_ptr().add(w_idx * c_in);
            for ic in 0..c_in {
                let wv = _mm512_loadu_ps(weight.as_ptr().add(ic * c_exp + coff));
                acc = _mm512_fmadd_ps(_mm512_set1_ps(*ip.add(ic)), wv, acc);
            }
            if relu {
                acc = _mm512_max_ps(acc, zero);
            }
            _mm512_storeu_ps(dst_cb_base.add(w_idx * 16), acc);
            w_idx += 1;
        }
    }
}

/// NCHWc-blocked PW-reduce row kernel: reads blocked DW output row
/// `[c_exp_blocks][w][16]`, writes NHWC output row `[w][c_out_padded]`.
/// Weight is `[c_exp][c_out_padded]` row-major (same layout as the existing
/// fused kernel's prepacked weight).
///
/// Register-tiled with 8-row × 1-oc-block accumulators, mirroring
/// `pointwise_nx16_direct_rows_avx512`. The inner reduction iterates over
/// `c_exp` as `(cb, k)` where the input lane is read from the NCHWc-blocked
/// `[cb, ow, k]` layout; weight is a contiguous row of `c_out_padded`.
#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code)]
pub fn compute_pw_reduce_row_nchwc_avx512(
    dw_blocked: &[f32],    // [c_exp_blocks * w * 16]
    weight_packed: &[f32], // [c_exp * c_out_padded]
    bias: Option<&[f32]>,  // [c_out_padded]
    out_row: &mut [f32],   // [w * c_out_padded]
    w: usize,
    c_exp: usize,
    c_out_padded: usize,
    relu: bool,
) {
    unsafe {
        compute_pw_reduce_row_nchwc_avx512_inner(
            dw_blocked,
            weight_packed,
            bias,
            out_row,
            w,
            c_exp,
            c_out_padded,
            relu,
        )
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn, clippy::too_many_arguments)]
unsafe fn compute_pw_reduce_row_nchwc_avx512_inner(
    dw_blocked: &[f32],
    weight_packed: &[f32],
    bias: Option<&[f32]>,
    out_row: &mut [f32],
    w: usize,
    c_exp: usize,
    c_out_padded: usize,
    relu: bool,
) {
    use std::arch::x86_64::*;
    debug_assert!(c_exp.is_multiple_of(16));
    debug_assert!(c_out_padded.is_multiple_of(16));
    let c_exp_blocks = c_exp / 16;
    let c_out_blocks = c_out_padded / 16;
    let zero = _mm512_setzero_ps();

    for ocb in 0..c_out_blocks {
        let oc_off = ocb * 16;
        let bias_v = if let Some(b) = bias {
            _mm512_loadu_ps(b.as_ptr().add(oc_off))
        } else {
            zero
        };
        let mut ow = 0usize;
        // 8-row tile.
        while ow + 8 <= w {
            let mut a0 = bias_v;
            let mut a1 = bias_v;
            let mut a2 = bias_v;
            let mut a3 = bias_v;
            let mut a4 = bias_v;
            let mut a5 = bias_v;
            let mut a6 = bias_v;
            let mut a7 = bias_v;
            for cb in 0..c_exp_blocks {
                let xp = dw_blocked.as_ptr().add(cb * w * 16);
                let wp_cb_base = weight_packed.as_ptr().add(cb * 16 * c_out_padded + oc_off);
                for k in 0..16usize {
                    let wv = _mm512_loadu_ps(wp_cb_base.add(k * c_out_padded));
                    a0 = _mm512_fmadd_ps(_mm512_set1_ps(*xp.add(ow * 16 + k)), wv, a0);
                    a1 = _mm512_fmadd_ps(_mm512_set1_ps(*xp.add((ow + 1) * 16 + k)), wv, a1);
                    a2 = _mm512_fmadd_ps(_mm512_set1_ps(*xp.add((ow + 2) * 16 + k)), wv, a2);
                    a3 = _mm512_fmadd_ps(_mm512_set1_ps(*xp.add((ow + 3) * 16 + k)), wv, a3);
                    a4 = _mm512_fmadd_ps(_mm512_set1_ps(*xp.add((ow + 4) * 16 + k)), wv, a4);
                    a5 = _mm512_fmadd_ps(_mm512_set1_ps(*xp.add((ow + 5) * 16 + k)), wv, a5);
                    a6 = _mm512_fmadd_ps(_mm512_set1_ps(*xp.add((ow + 6) * 16 + k)), wv, a6);
                    a7 = _mm512_fmadd_ps(_mm512_set1_ps(*xp.add((ow + 7) * 16 + k)), wv, a7);
                }
            }
            if relu {
                a0 = _mm512_max_ps(a0, zero);
                a1 = _mm512_max_ps(a1, zero);
                a2 = _mm512_max_ps(a2, zero);
                a3 = _mm512_max_ps(a3, zero);
                a4 = _mm512_max_ps(a4, zero);
                a5 = _mm512_max_ps(a5, zero);
                a6 = _mm512_max_ps(a6, zero);
                a7 = _mm512_max_ps(a7, zero);
            }
            let op = out_row.as_mut_ptr().add(ow * c_out_padded + oc_off);
            _mm512_storeu_ps(op, a0);
            _mm512_storeu_ps(op.add(c_out_padded), a1);
            _mm512_storeu_ps(op.add(2 * c_out_padded), a2);
            _mm512_storeu_ps(op.add(3 * c_out_padded), a3);
            _mm512_storeu_ps(op.add(4 * c_out_padded), a4);
            _mm512_storeu_ps(op.add(5 * c_out_padded), a5);
            _mm512_storeu_ps(op.add(6 * c_out_padded), a6);
            _mm512_storeu_ps(op.add(7 * c_out_padded), a7);
            ow += 8;
        }
        // 4-row tile.
        while ow + 4 <= w {
            let mut a0 = bias_v;
            let mut a1 = bias_v;
            let mut a2 = bias_v;
            let mut a3 = bias_v;
            for cb in 0..c_exp_blocks {
                let xp = dw_blocked.as_ptr().add(cb * w * 16);
                let wp_cb_base = weight_packed.as_ptr().add(cb * 16 * c_out_padded + oc_off);
                for k in 0..16usize {
                    let wv = _mm512_loadu_ps(wp_cb_base.add(k * c_out_padded));
                    a0 = _mm512_fmadd_ps(_mm512_set1_ps(*xp.add(ow * 16 + k)), wv, a0);
                    a1 = _mm512_fmadd_ps(_mm512_set1_ps(*xp.add((ow + 1) * 16 + k)), wv, a1);
                    a2 = _mm512_fmadd_ps(_mm512_set1_ps(*xp.add((ow + 2) * 16 + k)), wv, a2);
                    a3 = _mm512_fmadd_ps(_mm512_set1_ps(*xp.add((ow + 3) * 16 + k)), wv, a3);
                }
            }
            if relu {
                a0 = _mm512_max_ps(a0, zero);
                a1 = _mm512_max_ps(a1, zero);
                a2 = _mm512_max_ps(a2, zero);
                a3 = _mm512_max_ps(a3, zero);
            }
            let op = out_row.as_mut_ptr().add(ow * c_out_padded + oc_off);
            _mm512_storeu_ps(op, a0);
            _mm512_storeu_ps(op.add(c_out_padded), a1);
            _mm512_storeu_ps(op.add(2 * c_out_padded), a2);
            _mm512_storeu_ps(op.add(3 * c_out_padded), a3);
            ow += 4;
        }
        // 1-row tail.
        while ow < w {
            let mut acc = bias_v;
            for cb in 0..c_exp_blocks {
                let xp = dw_blocked.as_ptr().add(cb * w * 16 + ow * 16);
                let wp_cb_base = weight_packed.as_ptr().add(cb * 16 * c_out_padded + oc_off);
                for k in 0..16usize {
                    let wv = _mm512_loadu_ps(wp_cb_base.add(k * c_out_padded));
                    acc = _mm512_fmadd_ps(_mm512_set1_ps(*xp.add(k)), wv, acc);
                }
            }
            if relu {
                acc = _mm512_max_ps(acc, zero);
            }
            _mm512_storeu_ps(out_row.as_mut_ptr().add(ow * c_out_padded + oc_off), acc);
            ow += 1;
        }
    }
}

/// Returns the register-blocked DW 3×3 row kernel selected for `c` channels.
pub fn select_dw3_row_fn(c: usize) -> Dw3RowFn {
    select_variant(c).compute_dw_row
}

fn select_dw5_variant(c_exp: usize) -> Dw5Variant {
    #[cfg(target_arch = "x86_64")]
    {
        let cpu = crate::host_cpu();
        if c_exp.is_multiple_of(16) && !cfg!(miri) && cpu.features.avx512f {
            return Dw5Variant {
                compute_dw_row: avx512::compute_dw5_row_avx512,
            };
        }
        if c_exp.is_multiple_of(8) && !cfg!(miri) && cpu.features.avx2 && cpu.features.fma {
            return Dw5Variant {
                compute_dw_row: avx2::compute_dw5_row_avx2,
            };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        let cpu = crate::host_cpu();
        // In-order cores (cpu.uarch.is_in_order()) drive the column-reuse DW5
        // asm inside `compute_dw5_row_neon`; the per-uarch split lands here in a
        // later phase. Same kernel for every NEON core today.
        if c_exp.is_multiple_of(4) && !cfg!(miri) && cpu.features.neon {
            return Dw5Variant {
                compute_dw_row: neon::compute_dw5_row_neon,
            };
        }
    }
    Dw5Variant {
        compute_dw_row: scalar::compute_dw5_row_scalar,
    }
}

fn select_tile_variant(c_exp: usize) -> TileVariant {
    #[cfg(target_arch = "x86_64")]
    {
        let cpu = crate::host_cpu();
        if c_exp.is_multiple_of(16) && !cfg!(miri) && cpu.features.avx512f {
            return TileVariant {
                compute_pw_tile: avx512::compute_pw_tile_avx512,
                compute_dw5_tile: avx512::compute_dw5_tile_avx512,
            };
        }
        if c_exp.is_multiple_of(8) && !cfg!(miri) && cpu.features.avx2 && cpu.features.fma {
            return TileVariant {
                compute_pw_tile: avx2::compute_pw_tile_avx2,
                compute_dw5_tile: avx2::compute_dw5_tile_avx2,
            };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        let cpu = crate::host_cpu();
        if c_exp.is_multiple_of(4) && !cfg!(miri) && cpu.features.neon {
            return TileVariant {
                compute_pw_tile: neon::compute_pw_tile_neon,
                compute_dw5_tile: neon::compute_dw5_tile_neon,
            };
        }
    }
    TileVariant {
        compute_pw_tile: scalar::compute_pw_tile_scalar,
        compute_dw5_tile: scalar::compute_dw5_tile_scalar,
    }
}

// ── Scalar reference ────────────────────────────────────────────────

mod scalar;

// ── AVX2 + FMA (x86_64) ──────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
mod avx2;

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
mod avx512;

// ── NEON (aarch64) ────────────────────────────────────────────────────

#[cfg(target_arch = "aarch64")]
mod neon;

// ── Tests: SIMD variants vs scalar reference ─────────────────────────

#[cfg(test)]
mod tests;
