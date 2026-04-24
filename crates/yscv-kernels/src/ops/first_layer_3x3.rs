//! Dedicated microkernel for the model's first-layer 3×3 stride-2 RGB Conv.
//!
//! This shape (`kh=kw=3, stride=2, C_in=3`) shows up at the head of every
//! computer-vision model that takes raw RGB input, and it's the single
//! largest Conv in our Siamese tracker's fused-path profile — 0.75ms per
//! inference at 6T across the two first-layer Convs, 13% of the measured
//! time. See `scripts/gap_reports/2026-04-19_part_a_prime_and_b_profile.md`.
//!
//! Why the generic im2col + BLAS sgemm path is bad here: k = 3·3·3 = 27.
//! Blocked GEMM's `KC = 256` collapses to a single panel of 27 cols, so
//! all the packing/tiling overhead is paid with none of the cache-reuse
//! benefit. MLAS (what ORT uses) has a dedicated first-layer microkernel
//! — visible as `_nchwc` ops in the ORT profile.
//!
//! This module is that dedicated microkernel for yscv. Multi-arch per
//! workspace rule: x86 AVX2/AVX-512 + aarch64 NEON + scalar fallback.
//! All variants share one scalar reference implementation via tests, so
//! correctness is identical across targets.
//!
//! ## Contract
//!
//! - `input`  : `[N, H, W, 3]` NHWC, contiguous f32.
//! - `weight` : `[KH=3, KW=3, C_in=3, C_out]` KHWC layout (what yscv's
//!   runner pre-permutes Conv weights into at model load).
//! - `bias`   : optional `[C_out]`, added pre-activation.
//! - `output` : `[N, out_h, out_w, C_out]`, pre-allocated, contiguous.
//! - `stride_h = stride_w = 2`; `KH = KW = 3`.
//! - Padding: any `(pad_top, pad_left, pad_bottom, pad_right)`. Out-of-bounds
//!   reads are zero (implicit zero padding).
//! - `activation`: None or Relu (Silu is rare on first-layer Conv; if needed
//!   later, add the variant).
//!
//! ## Register plan (AVX2, C_out = 16 — the tracker's shape)
//!
//! Per output pixel:
//! - `ymm0`, `ymm1` : accumulators for lanes \[0..8\] and \[8..16\].
//! - Kernel tap loop over `(ky, kx, c_in) = 3×3×3 = 27`:
//!   broadcast `input[ih, iw, c_in]` → `ymm_x`,
//!   load `weight[ky, kx, c_in, 0..8]` → `ymm_w0`, `weight[..., 8..16]` → `ymm_w1`,
//!   `ymm0 = fma(ymm0, ymm_x, ymm_w0)`,
//!   `ymm1 = fma(ymm1, ymm_x, ymm_w1)`.
//! - Activation: `_mm256_max_ps` against zero when Relu.
//! - Store to `output[out_h, out_w, 0..16]`.
//!
//! 54 FMAs per output pixel. On Zen 4 with 2 × 8-wide FMA ports (16
//! FMAs/cycle peak), theoretical lower bound 3.5 cycles/pixel. With load
//! latencies + stores, expect 6–10 cycles/pixel real = ~32µs per
//! `256→128` Conv on a single thread.
//!
//! ## Boundary handling
//!
//! Per-pixel branch on ih/iw bounds (scalar path + all SIMD variants).
//! The first layer is only ~1% of output pixels on the boundary — not
//! hot enough to justify split peel loops. Inner loop stays predictable
//! for the bulk interior pixels.

use super::conv::Activation;

use rayon::ThreadPool;
use rayon::prelude::*;

/// Compute output spatial dim from input + kernel + stride + padding.
#[inline]
fn out_dim(in_dim: usize, pad_lo: usize, pad_hi: usize, k: usize, stride: usize) -> usize {
    (in_dim + pad_lo + pad_hi - k) / stride + 1
}

/// Session 14 R4: minimum out_h for parallel dispatch. Below this, the
/// threading overhead (~5-10 µs rayon pool wake-up) dominates the per-row
/// work. 32 rows × 128 cols × 16 channels × 27 MACs/pixel = ~1.7M FMAs
/// per chunk — large enough to amortise the wake-up, small enough that
/// batch=1 tracker shapes (out_h=128) still split into 6 chunks.
const PARALLEL_MIN_OUT_H: usize = 32;

/// Environment kill-switch for the parallel dispatch path. Leave unset
/// for default (parallel on when thread_pool provided + out_h large
/// enough). Set `YSCV_FIRST_LAYER_PAR_OFF=1` to force sequential (for
/// A/B measurement and worst-case profiling).
fn first_layer_par_disabled() -> bool {
    static CACHED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| std::env::var_os("YSCV_FIRST_LAYER_PAR_OFF").is_some())
}

/// Entry point. Dispatches to the fastest available SIMD implementation
/// at runtime. Falls back to the scalar reference when no SIMD feature
/// matches or when `miri` is detected (intrinsics don't run under Miri).
///
/// When `thread_pool` is `Some` and `out_h >= PARALLEL_MIN_OUT_H`, the
/// output is split row-wise across the pool's workers. Each worker
/// runs the same SIMD variant on a disjoint chunk of output rows;
/// input/weight slices are shared read-only.
///
/// See module docs for the `input`/`weight`/`output` layout contract.
#[allow(clippy::too_many_arguments)]
pub(crate) fn conv2d_nhwc_3ch_3x3_s2_padded(
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    output: &mut [f32],
    batch: usize,
    in_h: usize,
    in_w: usize,
    c_out: usize,
    pad_top: usize,
    pad_left: usize,
    pad_bottom: usize,
    pad_right: usize,
    activation: Activation,
    thread_pool: Option<&ThreadPool>,
) {
    // Resolve which variant to run once; the same callback is used for
    // sequential and per-chunk parallel dispatch. Each variant's
    // `run_rows` writes only to its `output_chunk` parameter using
    // chunk-relative indexing, so disjoint chunks across workers are
    // memory-safe.
    //
    // `VariantFn` captures no state — it's just the function pointer to
    // one of the SIMD `run_rows` implementations (or the scalar fallback).
    // Selected lazily outside the hot loop to avoid re-detecting CPU
    // features per chunk.
    type VariantFn = fn(
        &[f32],
        &[f32],
        Option<&[f32]>,
        &mut [f32],
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
        Activation,
    );
    let variant: VariantFn = select_variant(c_out);

    let out_h = out_dim(in_h, pad_top, pad_bottom, 3, 2);
    let out_w = out_dim(in_w, pad_left, pad_right, 3, 2);
    let out_row_stride = out_w * c_out;
    let out_batch_stride = out_h * out_row_stride;

    // Parallel dispatch fires when the work is large enough to amortise
    // rayon's fork cost. Uses the caller's pool (via `install`) when
    // provided, else rayon's thread-local default pool. The `_default`
    // wrapper callers in backend.rs pass `None`; that still benefits
    // from parallelism via the global pool since rayon's par_iter picks
    // up the caller's ambient scope. Kill-switch env
    // `YSCV_FIRST_LAYER_PAR_OFF=1` forces sequential for A/B measurement.
    // Disable parallel dispatch under Miri: rayon/crossbeam epoch-GC
    // violates Stacked Borrows — the parallelism itself is sound, but
    // Miri can't model crossbeam's lock-free GC. No perf impact in
    // production (cfg!(miri) folds to false at compile time).
    let use_par = !cfg!(miri) && !first_layer_par_disabled() && out_h >= PARALLEL_MIN_OUT_H;

    for ni in 0..batch {
        let batch_slice = &mut output[ni * out_batch_stride..(ni + 1) * out_batch_stride];
        if !use_par {
            variant(
                input,
                weight,
                bias,
                batch_slice,
                ni,
                0,
                out_h,
                in_h,
                in_w,
                c_out,
                pad_top,
                pad_left,
                pad_bottom,
                pad_right,
                activation,
            );
            continue;
        }
        // par_chunks_mut splits by row groups. Rayon distributes chunks
        // across whatever pool is active — an explicit `thread_pool`
        // when given, rayon's global pool otherwise.
        let nthreads = thread_pool
            .map(|p| p.current_num_threads())
            .unwrap_or_else(rayon::current_num_threads)
            .max(1);
        let rows_per_chunk = out_h.div_ceil(nthreads);
        let chunk_bytes = rows_per_chunk * out_row_stride;
        if nthreads <= 1 {
            variant(
                input,
                weight,
                bias,
                batch_slice,
                ni,
                0,
                out_h,
                in_h,
                in_w,
                c_out,
                pad_top,
                pad_left,
                pad_bottom,
                pad_right,
                activation,
            );
            continue;
        }
        // `batch_slice` is &mut [f32], so the closure is FnOnce after the
        // move. `pool.install` takes FnOnce -> R, so we just run the
        // closure inline in both branches.
        if let Some(pool) = thread_pool {
            pool.install(|| {
                batch_slice
                    .par_chunks_mut(chunk_bytes)
                    .enumerate()
                    .for_each(|(chunk_idx, chunk)| {
                        let oh_start = chunk_idx * rows_per_chunk;
                        let oh_end = (oh_start + rows_per_chunk).min(out_h);
                        variant(
                            input, weight, bias, chunk, ni, oh_start, oh_end, in_h, in_w, c_out,
                            pad_top, pad_left, pad_bottom, pad_right, activation,
                        );
                    });
            });
        } else {
            batch_slice
                .par_chunks_mut(chunk_bytes)
                .enumerate()
                .for_each(|(chunk_idx, chunk)| {
                    let oh_start = chunk_idx * rows_per_chunk;
                    let oh_end = (oh_start + rows_per_chunk).min(out_h);
                    variant(
                        input, weight, bias, chunk, ni, oh_start, oh_end, in_h, in_w, c_out,
                        pad_top, pad_left, pad_bottom, pad_right, activation,
                    );
                });
        }
    }
}

/// Variant selector: resolves the fastest SIMD `run_rows` available at
/// runtime. Returns a pointer to a function with the same `run_rows`
/// signature so both sequential and parallel dispatch paths can call it.
///
/// Cached CPU-feature detection matters less here than in a per-pixel
/// hot loop (we call this once per outer op invocation), but keeping it
/// deterministic simplifies testing.
fn select_variant(
    c_out: usize,
) -> fn(
    &[f32],
    &[f32],
    Option<&[f32]>,
    &mut [f32],
    usize,
    usize,
    usize,
    usize,
    usize,
    usize,
    usize,
    usize,
    usize,
    usize,
    Activation,
) {
    #[cfg(target_arch = "x86_64")]
    {
        // Step A3: AVX-512 first-layer default ON on Zen 4 (ZMM FMA latency
        // matches YMM at 4 cycles, and halved uop count amortises dispatch).
        // Kill-switch `YSCV_FIRST_LAYER_AVX512_OFF=1` falls back to AVX2.
        fn first_layer_avx512_disabled() -> bool {
            static CACHED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            *CACHED.get_or_init(|| std::env::var_os("YSCV_FIRST_LAYER_AVX512_OFF").is_some())
        }
        if !cfg!(miri)
            && !first_layer_avx512_disabled()
            && c_out.is_multiple_of(16)
            && is_x86_feature_detected!("avx512f")
        {
            return avx512_run_rows_entry;
        }
        if !cfg!(miri)
            && c_out.is_multiple_of(8)
            && is_x86_feature_detected!("avx2")
            && is_x86_feature_detected!("fma")
        {
            return avx2_run_rows_entry;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if !cfg!(miri) && c_out % 4 == 0 && std::arch::is_aarch64_feature_detected!("neon") {
            return neon_run_rows_entry;
        }
    }
    scalar_run_rows
}

// ── Safe `fn` wrappers around each variant's `unsafe run_rows` ────────
//
// Variants live inside `mod {avx2, avx512, neon}` and are gated by
// `target_feature`. The variant selector needs a plain `fn` pointer
// whose type doesn't carry target-feature attributes, so each wrapper
// below re-enters the variant under a matching runtime feature guard.
// Runtime detection already passed in `select_variant`; the wrappers
// are `#[inline]` so LLVM strips the extra call when it can.

#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
fn avx512_run_rows_entry(
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    output_chunk: &mut [f32],
    ni: usize,
    oh_start: usize,
    oh_end: usize,
    in_h: usize,
    in_w: usize,
    c_out: usize,
    pad_top: usize,
    pad_left: usize,
    pad_bottom: usize,
    pad_right: usize,
    activation: Activation,
) {
    // SAFETY: `select_variant` verified avx512f is available and
    // `c_out % 16 == 0`. Output chunk bounds are caller-guaranteed
    // (`par_chunks_mut` in the dispatcher yields disjoint slices).
    #[allow(unsafe_code)]
    unsafe {
        avx512::run_rows(
            input,
            weight,
            bias,
            output_chunk,
            ni,
            oh_start,
            oh_end,
            in_h,
            in_w,
            c_out,
            pad_top,
            pad_left,
            pad_bottom,
            pad_right,
            activation,
        );
    }
}

#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
fn avx2_run_rows_entry(
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    output_chunk: &mut [f32],
    ni: usize,
    oh_start: usize,
    oh_end: usize,
    in_h: usize,
    in_w: usize,
    c_out: usize,
    pad_top: usize,
    pad_left: usize,
    pad_bottom: usize,
    pad_right: usize,
    activation: Activation,
) {
    // SAFETY: `select_variant` verified avx2+fma and `c_out % 8 == 0`.
    #[allow(unsafe_code)]
    unsafe {
        avx2::run_rows(
            input,
            weight,
            bias,
            output_chunk,
            ni,
            oh_start,
            oh_end,
            in_h,
            in_w,
            c_out,
            pad_top,
            pad_left,
            pad_bottom,
            pad_right,
            activation,
        );
    }
}

#[cfg(target_arch = "aarch64")]
#[allow(clippy::too_many_arguments)]
fn neon_run_rows_entry(
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    output_chunk: &mut [f32],
    ni: usize,
    oh_start: usize,
    oh_end: usize,
    in_h: usize,
    in_w: usize,
    c_out: usize,
    pad_top: usize,
    pad_left: usize,
    pad_bottom: usize,
    pad_right: usize,
    activation: Activation,
) {
    // SAFETY: `select_variant` verified NEON and `c_out % 4 == 0`.
    #[allow(unsafe_code)]
    unsafe {
        neon::run_rows(
            input,
            weight,
            bias,
            output_chunk,
            ni,
            oh_start,
            oh_end,
            in_h,
            in_w,
            c_out,
            pad_top,
            pad_left,
            pad_bottom,
            pad_right,
            activation,
        );
    }
}

/// Reference scalar implementation (test oracle). Loops `ni` and defers
/// per-row work to `scalar_run_rows`.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn scalar_run(
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    output: &mut [f32],
    batch: usize,
    in_h: usize,
    in_w: usize,
    c_out: usize,
    pad_top: usize,
    pad_left: usize,
    pad_bottom: usize,
    pad_right: usize,
    activation: Activation,
) {
    let out_h = out_dim(in_h, pad_top, pad_bottom, 3, 2);
    let out_w = out_dim(in_w, pad_left, pad_right, 3, 2);
    let out_row_stride = out_w * c_out;
    let out_batch_stride = out_h * out_row_stride;
    for ni in 0..batch {
        let batch_slice = &mut output[ni * out_batch_stride..(ni + 1) * out_batch_stride];
        scalar_run_rows(
            input,
            weight,
            bias,
            batch_slice,
            ni,
            0,
            out_h,
            in_h,
            in_w,
            c_out,
            pad_top,
            pad_left,
            pad_bottom,
            pad_right,
            activation,
        );
    }
}

/// Scalar per-chunk worker: processes output rows in `oh_start..oh_end`
/// for the single batch `ni`. Input/weight reads use absolute NI/row
/// coordinates; output writes use chunk-relative `oh - oh_start` indexing
/// so disjoint chunks across parallel workers don't alias.
#[allow(clippy::too_many_arguments)]
fn scalar_run_rows(
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    output_chunk: &mut [f32],
    ni: usize,
    oh_start: usize,
    oh_end: usize,
    in_h: usize,
    in_w: usize,
    c_out: usize,
    pad_top: usize,
    pad_left: usize,
    _pad_bottom: usize,
    pad_right: usize,
    activation: Activation,
) {
    let _ = pad_right; // boundary handled via explicit bounds checks below
    let out_w_for_row = out_dim(in_w, pad_left, pad_right, 3, 2);
    let c_in: usize = 3;
    let in_row_stride = in_w * c_in;
    let in_batch_stride = in_h * in_row_stride;
    let out_row_stride = out_w_for_row * c_out;
    let w_c_stride = c_out;
    let w_kx_stride = c_in * c_out;
    let w_ky_stride = 3 * w_kx_stride;

    let mut acc = vec![0f32; c_out];
    for oh in oh_start..oh_end {
        for ow in 0..out_w_for_row {
            match bias {
                Some(b) => acc.copy_from_slice(&b[..c_out]),
                None => acc.fill(0.0),
            }
            let ih0 = (oh as isize) * 2 - pad_top as isize;
            let iw0 = (ow as isize) * 2 - pad_left as isize;
            for ky in 0..3isize {
                let ih = ih0 + ky;
                if ih < 0 || (ih as usize) >= in_h {
                    continue;
                }
                let w_ky_off = ky as usize * w_ky_stride;
                for kx in 0..3isize {
                    let iw = iw0 + kx;
                    if iw < 0 || (iw as usize) >= in_w {
                        continue;
                    }
                    let in_off =
                        ni * in_batch_stride + ih as usize * in_row_stride + iw as usize * c_in;
                    let w_kx_off = w_ky_off + kx as usize * w_kx_stride;
                    for c in 0..c_in {
                        let x = input[in_off + c];
                        let w_base = w_kx_off + c * w_c_stride;
                        for oc in 0..c_out {
                            acc[oc] += x * weight[w_base + oc];
                        }
                    }
                }
            }
            if matches!(activation, Activation::Relu) {
                for v in acc.iter_mut() {
                    if *v < 0.0 {
                        *v = 0.0;
                    }
                }
            }
            // Chunk-relative output index: subtract oh_start so row 0 of
            // the chunk maps to offset 0.
            let out_off = (oh - oh_start) * out_row_stride + ow * c_out;
            output_chunk[out_off..out_off + c_out].copy_from_slice(&acc);
        }
    }
}

// ── AVX2 ─────────────────────────────────────────────────────────────
#[cfg(target_arch = "x86_64")]
mod avx2 {
    use super::*;
    use std::arch::x86_64::*;

    /// AVX2 implementation of the first-layer 3×3 s=2 RGB Conv.
    ///
    /// Requires `c_out % 8 == 0`. Processes output channels in chunks of
    /// 8, fully unrolling the 3×3×3 = 27-tap inner loop. Interior pixels
    /// skip the bounds check via a fast path.
    ///
    /// # Safety
    /// Caller must have verified `avx2` + `fma` are available and
    /// `c_out % 8 == 0`. `input`, `weight`, `output` and `bias` slices
    /// are assumed to match the shape contract in the module docs.
    /// Per-chunk AVX2 worker. Processes `oh_start..oh_end` rows for the
    /// single batch index `ni`, writing into the chunk-relative slice
    /// `output_chunk` (length == `(oh_end - oh_start) * out_row_stride`).
    /// Input/weight reads use absolute ni/row coordinates; output writes
    /// use `(oh - oh_start) * out_row_stride + ow * c_out`.
    #[target_feature(enable = "avx2,fma")]
    #[allow(unsafe_code, clippy::too_many_arguments)]
    pub(super) unsafe fn run_rows(
        input: &[f32],
        weight: &[f32],
        bias: Option<&[f32]>,
        output_chunk: &mut [f32],
        ni: usize,
        oh_start: usize,
        oh_end: usize,
        in_h: usize,
        in_w: usize,
        c_out: usize,
        pad_top: usize,
        pad_left: usize,
        _pad_bottom: usize,
        pad_right: usize,
        activation: Activation,
    ) {
        unsafe {
            let out_w = out_dim(in_w, pad_left, pad_right, 3, 2);
            let c_in: usize = 3;
            let in_row_stride = in_w * c_in;
            let in_batch_stride = in_h * in_row_stride;
            let out_row_stride = out_w * c_out;
            let w_c_stride = c_out;
            let w_kx_stride = c_in * c_out;
            let w_ky_stride = 3 * w_kx_stride;

            // Interior stretch of output rows/cols where all 9 taps are
            // in-bounds — no per-tap branch. With stride=2 & kernel=3, an
            // output pixel at (oh, ow) reads input rows [2*oh-pad_top ..
            // 2*oh-pad_top+2]. Interior starts at oh where 2*oh-pad_top >= 0
            // and ends at oh where 2*oh-pad_top+2 < in_h.
            let oh_interior_lo = pad_top.div_ceil(2);
            let oh_interior_hi = if in_h + pad_top >= 3 {
                (in_h + pad_top - 3) / 2 + 1
            } else {
                0
            };
            let ow_interior_lo = pad_left.div_ceil(2);
            let ow_interior_hi = if in_w + pad_left >= 3 {
                (in_w + pad_left - 3) / 2 + 1
            } else {
                0
            };

            let zero = _mm256_setzero_ps();
            let relu = matches!(activation, Activation::Relu);
            // Specialised path: c_out = 16 is the tracker shape and by far
            // the most common first-layer output channel count. Two YMM
            // accumulators per output pixel, processed together every tap
            // so the FMAs on ymm_lo and ymm_hi run on both FMA ports
            // concurrently. Additionally we tile across 2 adjacent output
            // columns → 4 independent accumulator chains, hiding the 27-tap
            // FMA latency behind throughput.
            let spec16 = c_out == 16;

            for oh in oh_start..oh_end {
                let ih0 = (oh as isize) * 2 - pad_top as isize;
                let oh_is_interior = oh >= oh_interior_lo && oh < oh_interior_hi;

                if spec16 && oh_is_interior {
                    // Fully interior rows: iterate ow in pairs. `row_out_base`
                    // is chunk-relative: row 0 of the chunk is at offset 0.
                    let row_out_base = (oh - oh_start) * out_row_stride;
                    let bias_lo = match bias {
                        Some(b) => _mm256_loadu_ps(b.as_ptr()),
                        None => _mm256_setzero_ps(),
                    };
                    let bias_hi = match bias {
                        Some(b) => _mm256_loadu_ps(b.as_ptr().add(8)),
                        None => _mm256_setzero_ps(),
                    };
                    let mut ow = 0usize;
                    while ow + 1 < out_w && ow >= ow_interior_lo && (ow + 1) < ow_interior_hi {
                        let iw0_a = (ow as isize) * 2 - pad_left as isize;
                        let iw0_b = ((ow + 1) as isize) * 2 - pad_left as isize;
                        let mut acc_a_lo = bias_lo;
                        let mut acc_a_hi = bias_hi;
                        let mut acc_b_lo = bias_lo;
                        let mut acc_b_hi = bias_hi;
                        for ky in 0..3 {
                            let ih = (ih0 + ky as isize) as usize;
                            for kx in 0..3 {
                                let iw_a = (iw0_a + kx as isize) as usize;
                                let iw_b = (iw0_b + kx as isize) as usize;
                                let row_in_base = ni * in_batch_stride + ih * in_row_stride;
                                let in_off_a = row_in_base + iw_a * c_in;
                                let in_off_b = row_in_base + iw_b * c_in;
                                let w_off = ky * w_ky_stride + kx * w_kx_stride;
                                // c = 0, 1, 2 with both pixel columns at once.
                                // 6 FMAs per tap across a + b × (lo, hi).
                                let w_c0_lo = _mm256_loadu_ps(weight.as_ptr().add(w_off));
                                let w_c0_hi = _mm256_loadu_ps(weight.as_ptr().add(w_off + 8));
                                let w_c1_lo =
                                    _mm256_loadu_ps(weight.as_ptr().add(w_off + w_c_stride));
                                let w_c1_hi =
                                    _mm256_loadu_ps(weight.as_ptr().add(w_off + w_c_stride + 8));
                                let w_c2_lo =
                                    _mm256_loadu_ps(weight.as_ptr().add(w_off + 2 * w_c_stride));
                                let w_c2_hi = _mm256_loadu_ps(
                                    weight.as_ptr().add(w_off + 2 * w_c_stride + 8),
                                );
                                let xa0 = _mm256_broadcast_ss(&*input.as_ptr().add(in_off_a));
                                let xa1 = _mm256_broadcast_ss(&*input.as_ptr().add(in_off_a + 1));
                                let xa2 = _mm256_broadcast_ss(&*input.as_ptr().add(in_off_a + 2));
                                let xb0 = _mm256_broadcast_ss(&*input.as_ptr().add(in_off_b));
                                let xb1 = _mm256_broadcast_ss(&*input.as_ptr().add(in_off_b + 1));
                                let xb2 = _mm256_broadcast_ss(&*input.as_ptr().add(in_off_b + 2));
                                acc_a_lo = _mm256_fmadd_ps(xa0, w_c0_lo, acc_a_lo);
                                acc_a_hi = _mm256_fmadd_ps(xa0, w_c0_hi, acc_a_hi);
                                acc_b_lo = _mm256_fmadd_ps(xb0, w_c0_lo, acc_b_lo);
                                acc_b_hi = _mm256_fmadd_ps(xb0, w_c0_hi, acc_b_hi);
                                acc_a_lo = _mm256_fmadd_ps(xa1, w_c1_lo, acc_a_lo);
                                acc_a_hi = _mm256_fmadd_ps(xa1, w_c1_hi, acc_a_hi);
                                acc_b_lo = _mm256_fmadd_ps(xb1, w_c1_lo, acc_b_lo);
                                acc_b_hi = _mm256_fmadd_ps(xb1, w_c1_hi, acc_b_hi);
                                acc_a_lo = _mm256_fmadd_ps(xa2, w_c2_lo, acc_a_lo);
                                acc_a_hi = _mm256_fmadd_ps(xa2, w_c2_hi, acc_a_hi);
                                acc_b_lo = _mm256_fmadd_ps(xb2, w_c2_lo, acc_b_lo);
                                acc_b_hi = _mm256_fmadd_ps(xb2, w_c2_hi, acc_b_hi);
                            }
                        }
                        if relu {
                            acc_a_lo = _mm256_max_ps(acc_a_lo, zero);
                            acc_a_hi = _mm256_max_ps(acc_a_hi, zero);
                            acc_b_lo = _mm256_max_ps(acc_b_lo, zero);
                            acc_b_hi = _mm256_max_ps(acc_b_hi, zero);
                        }
                        let out_a = row_out_base + ow * 16;
                        let out_b = row_out_base + (ow + 1) * 16;
                        _mm256_storeu_ps(output_chunk.as_mut_ptr().add(out_a), acc_a_lo);
                        _mm256_storeu_ps(output_chunk.as_mut_ptr().add(out_a + 8), acc_a_hi);
                        _mm256_storeu_ps(output_chunk.as_mut_ptr().add(out_b), acc_b_lo);
                        _mm256_storeu_ps(output_chunk.as_mut_ptr().add(out_b + 8), acc_b_hi);
                        ow += 2;
                    }
                    // Tail: odd number of interior columns or edge pixels,
                    // fall through to the generic loop below for this `ow`.
                    while ow < out_w {
                        process_single_pixel_avx2(
                            input,
                            weight,
                            bias,
                            output_chunk,
                            ni,
                            ih0,
                            oh,
                            oh_start,
                            ow,
                            pad_left,
                            in_batch_stride,
                            in_row_stride,
                            out_row_stride,
                            c_out,
                            c_in,
                            in_h,
                            in_w,
                            w_ky_stride,
                            w_kx_stride,
                            w_c_stride,
                            ow_interior_lo,
                            ow_interior_hi,
                            oh_is_interior,
                            relu,
                            zero,
                        );
                        ow += 1;
                    }
                    continue;
                }

                // Generic path for non-c_out=16 shapes and edge rows.
                for ow in 0..out_w {
                    process_single_pixel_avx2(
                        input,
                        weight,
                        bias,
                        output_chunk,
                        ni,
                        ih0,
                        oh,
                        oh_start,
                        ow,
                        pad_left,
                        in_batch_stride,
                        in_row_stride,
                        out_row_stride,
                        c_out,
                        c_in,
                        in_h,
                        in_w,
                        w_ky_stride,
                        w_kx_stride,
                        w_c_stride,
                        ow_interior_lo,
                        ow_interior_hi,
                        oh_is_interior,
                        relu,
                        zero,
                    );
                }
            }
        }
    }

    /// Generic single-pixel AVX2 path — used for tail pixels, edge rows,
    /// and non-c_out=16 shapes. Kept as a separate fn to avoid cloning
    /// the loop body twice in the main function.
    ///
    /// # Safety
    /// Caller already under `#[target_feature = "avx2,fma"]`. Slice
    /// offsets are bounded by the shape contract of the parent fn.
    #[target_feature(enable = "avx2,fma")]
    #[allow(unsafe_code, clippy::too_many_arguments)]
    unsafe fn process_single_pixel_avx2(
        input: &[f32],
        weight: &[f32],
        bias: Option<&[f32]>,
        output_chunk: &mut [f32],
        ni: usize,
        ih0: isize,
        oh: usize,
        oh_start: usize,
        ow: usize,
        pad_left: usize,
        in_batch_stride: usize,
        in_row_stride: usize,
        out_row_stride: usize,
        c_out: usize,
        c_in: usize,
        in_h: usize,
        in_w: usize,
        w_ky_stride: usize,
        w_kx_stride: usize,
        w_c_stride: usize,
        ow_interior_lo: usize,
        ow_interior_hi: usize,
        oh_is_interior: bool,
        relu: bool,
        zero: __m256,
    ) {
        unsafe {
            let iw0 = (ow as isize) * 2 - pad_left as isize;
            let ow_is_interior = ow >= ow_interior_lo && ow < ow_interior_hi;
            // Chunk-relative: row 0 of chunk is at offset 0; no batch term.
            let out_off = (oh - oh_start) * out_row_stride + ow * c_out;
            let mut oc_chunk_start = 0usize;
            while oc_chunk_start < c_out {
                let mut acc = match bias {
                    Some(b) => _mm256_loadu_ps(b.as_ptr().add(oc_chunk_start)),
                    None => _mm256_setzero_ps(),
                };
                if oh_is_interior && ow_is_interior {
                    for ky in 0..3 {
                        let ih = (ih0 + ky as isize) as usize;
                        for kx in 0..3 {
                            let iw = (iw0 + kx as isize) as usize;
                            let in_off = ni * in_batch_stride + ih * in_row_stride + iw * c_in;
                            let w_off = ky * w_ky_stride + kx * w_kx_stride;
                            for c in 0..c_in {
                                let x = _mm256_broadcast_ss(&*input.as_ptr().add(in_off + c));
                                let w = _mm256_loadu_ps(
                                    weight.as_ptr().add(w_off + c * w_c_stride + oc_chunk_start),
                                );
                                acc = _mm256_fmadd_ps(x, w, acc);
                            }
                        }
                    }
                } else {
                    for ky in 0..3isize {
                        let ih = ih0 + ky;
                        if ih < 0 || (ih as usize) >= in_h {
                            continue;
                        }
                        let ih = ih as usize;
                        for kx in 0..3isize {
                            let iw = iw0 + kx;
                            if iw < 0 || (iw as usize) >= in_w {
                                continue;
                            }
                            let iw = iw as usize;
                            let in_off = ni * in_batch_stride + ih * in_row_stride + iw * c_in;
                            let w_off = ky as usize * w_ky_stride + kx as usize * w_kx_stride;
                            for c in 0..c_in {
                                let x = _mm256_broadcast_ss(&*input.as_ptr().add(in_off + c));
                                let w = _mm256_loadu_ps(
                                    weight.as_ptr().add(w_off + c * w_c_stride + oc_chunk_start),
                                );
                                acc = _mm256_fmadd_ps(x, w, acc);
                            }
                        }
                    }
                }
                if relu {
                    acc = _mm256_max_ps(acc, zero);
                }
                _mm256_storeu_ps(output_chunk.as_mut_ptr().add(out_off + oc_chunk_start), acc);
                oc_chunk_start += 8;
            }
        }
    }
}

// ── AVX-512 ──────────────────────────────────────────────────────────
#[cfg(target_arch = "x86_64")]
mod avx512 {
    use super::*;
    use std::arch::x86_64::*;

    /// AVX-512 implementation — 16-wide lanes. For the tracker's c_out=16
    /// this is one ZMM per output pixel.
    ///
    /// # Safety
    /// Caller verified `avx512f`, `c_out % 16 == 0`.
    /// Per-chunk AVX-512 worker. See avx2::run_rows for the contract —
    /// same shape (chunk-relative output, absolute ni/row for inputs).
    #[target_feature(enable = "avx512f")]
    #[allow(unsafe_code, clippy::too_many_arguments)]
    pub(super) unsafe fn run_rows(
        input: &[f32],
        weight: &[f32],
        bias: Option<&[f32]>,
        output_chunk: &mut [f32],
        ni: usize,
        oh_start: usize,
        oh_end: usize,
        in_h: usize,
        in_w: usize,
        c_out: usize,
        pad_top: usize,
        pad_left: usize,
        _pad_bottom: usize,
        pad_right: usize,
        activation: Activation,
    ) {
        unsafe {
            let out_w = out_dim(in_w, pad_left, pad_right, 3, 2);
            let c_in: usize = 3;
            let in_row_stride = in_w * c_in;
            let in_batch_stride = in_h * in_row_stride;
            let out_row_stride = out_w * c_out;
            let w_c_stride = c_out;
            let w_kx_stride = c_in * c_out;
            let w_ky_stride = 3 * w_kx_stride;

            let oh_interior_lo = pad_top.div_ceil(2);
            let oh_interior_hi = if in_h + pad_top >= 3 {
                (in_h + pad_top - 3) / 2 + 1
            } else {
                0
            };
            let ow_interior_lo = pad_left.div_ceil(2);
            let ow_interior_hi = if in_w + pad_left >= 3 {
                (in_w + pad_left - 3) / 2 + 1
            } else {
                0
            };

            let zero = _mm512_setzero_ps();
            let relu = matches!(activation, Activation::Relu);

            // Pre-load bias ZMM once (reused for all pixels).
            let bias_zmm = match bias {
                Some(b) => _mm512_loadu_ps(b.as_ptr()),
                None => zero,
            };

            for oh in oh_start..oh_end {
                let ih0 = (oh as isize) * 2 - pad_top as isize;
                let oh_is_interior = oh >= oh_interior_lo && oh < oh_interior_hi;
                let row_out_base = (oh - oh_start) * out_row_stride;

                let mut ow = 0usize;

                // Tile-8 fast path: for c_out=16 and interior rows, process
                // 8 adjacent output columns at a time with 8 independent ZMM
                // accumulators. Breaks the 27-tap FMA latency chain that the
                // per-pixel path below serialises — 8 parallel chains × 4
                // cycle FMA latency is ~fully hidden at Zen 4's 2-FMA-port
                // throughput. Projected: ~9× speedup over the per-pixel
                // path on tracker's `/xif0_0/conv_1/Conv` (256×256 → 128×128,
                // c_out=16), closing most of the 238 µs gap vs ORT.
                if c_out == 16 && oh_is_interior {
                    let ih_0 = ih0 as usize;
                    let ih_1 = (ih0 + 1) as usize;
                    let ih_2 = (ih0 + 2) as usize;
                    while ow + 8 <= out_w && ow >= ow_interior_lo && (ow + 7) < ow_interior_hi {
                        spec16_tile8_interior(
                            input,
                            weight,
                            bias_zmm,
                            output_chunk,
                            ni,
                            ow,
                            row_out_base,
                            ih_0,
                            ih_1,
                            ih_2,
                            pad_left,
                            in_batch_stride,
                            in_row_stride,
                            out_row_stride,
                            w_ky_stride,
                            w_kx_stride,
                            w_c_stride,
                            relu,
                            zero,
                        );
                        ow += 8;
                    }
                }

                // Fallback per-pixel path: edges, non-16 c_out, residual
                // interior columns past the tile boundary. Same logic as
                // before the tile-8 refactor.
                while ow < out_w {
                    let iw0 = (ow as isize) * 2 - pad_left as isize;
                    let ow_is_interior = ow >= ow_interior_lo && ow < ow_interior_hi;
                    let out_off = row_out_base + ow * c_out;
                    let mut oc_chunk_start = 0usize;
                    while oc_chunk_start < c_out {
                        let mut acc = match bias {
                            Some(b) => _mm512_loadu_ps(b.as_ptr().add(oc_chunk_start)),
                            None => _mm512_setzero_ps(),
                        };
                        if oh_is_interior && ow_is_interior {
                            for ky in 0..3 {
                                let ih = (ih0 + ky as isize) as usize;
                                for kx in 0..3 {
                                    let iw = (iw0 + kx as isize) as usize;
                                    let in_off =
                                        ni * in_batch_stride + ih * in_row_stride + iw * c_in;
                                    let w_off = ky * w_ky_stride + kx * w_kx_stride;
                                    let x0 = _mm512_set1_ps(*input.as_ptr().add(in_off));
                                    let w0 = _mm512_loadu_ps(
                                        weight.as_ptr().add(w_off + oc_chunk_start),
                                    );
                                    acc = _mm512_fmadd_ps(x0, w0, acc);
                                    let x1 = _mm512_set1_ps(*input.as_ptr().add(in_off + 1));
                                    let w1 = _mm512_loadu_ps(
                                        weight.as_ptr().add(w_off + w_c_stride + oc_chunk_start),
                                    );
                                    acc = _mm512_fmadd_ps(x1, w1, acc);
                                    let x2 = _mm512_set1_ps(*input.as_ptr().add(in_off + 2));
                                    let w2 = _mm512_loadu_ps(
                                        weight
                                            .as_ptr()
                                            .add(w_off + 2 * w_c_stride + oc_chunk_start),
                                    );
                                    acc = _mm512_fmadd_ps(x2, w2, acc);
                                }
                            }
                        } else {
                            for ky in 0..3isize {
                                let ih = ih0 + ky;
                                if ih < 0 || (ih as usize) >= in_h {
                                    continue;
                                }
                                let ih = ih as usize;
                                for kx in 0..3isize {
                                    let iw = iw0 + kx;
                                    if iw < 0 || (iw as usize) >= in_w {
                                        continue;
                                    }
                                    let iw = iw as usize;
                                    let in_off =
                                        ni * in_batch_stride + ih * in_row_stride + iw * c_in;
                                    let w_off =
                                        ky as usize * w_ky_stride + kx as usize * w_kx_stride;
                                    for c in 0..c_in {
                                        let x = _mm512_set1_ps(*input.as_ptr().add(in_off + c));
                                        let w = _mm512_loadu_ps(
                                            weight
                                                .as_ptr()
                                                .add(w_off + c * w_c_stride + oc_chunk_start),
                                        );
                                        acc = _mm512_fmadd_ps(x, w, acc);
                                    }
                                }
                            }
                        }
                        if relu {
                            acc = _mm512_max_ps(acc, zero);
                        }
                        _mm512_storeu_ps(
                            output_chunk.as_mut_ptr().add(out_off + oc_chunk_start),
                            acc,
                        );
                        oc_chunk_start += 16;
                    }
                    ow += 1;
                }
            }
        }
    }

    /// Tile-8 interior fast path: processes `ow..ow+8` output pixels
    /// concurrently with 8 independent ZMM accumulators. Caller
    /// guarantees all 8 pixels' input reads stay in-bounds (interior
    /// row + `ow >= ow_interior_lo && ow+7 < ow_interior_hi`), so no
    /// per-tap bounds checks.
    ///
    /// Register budget: 8 accumulators + 1 shared weight ZMM = 9 live
    /// ZMMs max. Plenty of room for the x-broadcast temp.
    ///
    /// FMA schedule (per tap — 27 total: ky ∈ 0..3, kx ∈ 0..3, c ∈ 0..3):
    ///   - 1 `_mm512_loadu_ps(weight)` (shared across 8 pixels)
    ///   - 8 `_mm512_set1_ps(input_scalar)` + 8 `_mm512_fmadd_ps(...)`
    #[target_feature(enable = "avx512f")]
    #[allow(unsafe_code, clippy::too_many_arguments)]
    unsafe fn spec16_tile8_interior(
        input: &[f32],
        weight: &[f32],
        bias_zmm: __m512,
        output_chunk: &mut [f32],
        ni: usize,
        ow: usize,
        row_out_base: usize,
        ih_0: usize,
        ih_1: usize,
        ih_2: usize,
        pad_left: usize,
        in_batch_stride: usize,
        in_row_stride: usize,
        out_row_stride: usize,
        w_ky_stride: usize,
        w_kx_stride: usize,
        w_c_stride: usize,
        relu: bool,
        zero: __m512,
    ) {
        unsafe {
            let _ = out_row_stride;
            // 8 independent accumulators, seeded from bias. Keeping them
            // as 8 separate local variables (not an array or tuple) gives
            // LLVM's register allocator the clearest signal that they
            // should live in ZMM registers for the entire function body.
            let mut acc0 = bias_zmm;
            let mut acc1 = bias_zmm;
            let mut acc2 = bias_zmm;
            let mut acc3 = bias_zmm;
            let mut acc4 = bias_zmm;
            let mut acc5 = bias_zmm;
            let mut acc6 = bias_zmm;
            let mut acc7 = bias_zmm;

            // Input NHWC stride per output-pixel is `2 * c_in = 6` (stride=2
            // and c_in=3). Pre-compute the base pointer for each of the
            // three input rows (ky = 0, 1, 2) so the inner tap loop only
            // adjusts by `(kx * c_in + c)`.
            let iw_base = 2 * ow - pad_left;
            let row_in_0 = input
                .as_ptr()
                .add(ni * in_batch_stride + ih_0 * in_row_stride + iw_base * 3);
            let row_in_1 = input
                .as_ptr()
                .add(ni * in_batch_stride + ih_1 * in_row_stride + iw_base * 3);
            let row_in_2 = input
                .as_ptr()
                .add(ni * in_batch_stride + ih_2 * in_row_stride + iw_base * 3);

            // 27 taps fully unrolled. Each tap:
            //   - One weight ZMM load (shared across 8 pixels).
            //   - 8 scalar loads + 8 broadcasts + 8 FMAs.
            //
            // FMA throughput: 8 independent accumulator chains × 2 FMA
            // ports at Zen 4's 4-cycle FMA latency → chains fully hide the
            // latency. Compute-bound at ~108 cycles per tile (216 FMAs / 2
            // ports), ~21 ns. Loads pipeline with compute.
            //
            // Unroll pattern: ky (row) outer, kx (col) + c (channel) inner.
            // For each tap, `x{t}` = scalar at (row, 2*(ow+t)+kx-pad, c)
            // with stride 6 (=2*c_in) between adjacent pixels.
            macro_rules! tap {
                ($row_ptr:expr, $ky:expr) => {
                    let row_ptr = $row_ptr;
                    // kx = 0, c = 0..=2
                    tap_inner!(row_ptr, $ky, 0, 0);
                    tap_inner!(row_ptr, $ky, 0, 1);
                    tap_inner!(row_ptr, $ky, 0, 2);
                    // kx = 1
                    tap_inner!(row_ptr, $ky, 1, 0);
                    tap_inner!(row_ptr, $ky, 1, 1);
                    tap_inner!(row_ptr, $ky, 1, 2);
                    // kx = 2
                    tap_inner!(row_ptr, $ky, 2, 0);
                    tap_inner!(row_ptr, $ky, 2, 1);
                    tap_inner!(row_ptr, $ky, 2, 2);
                };
            }
            macro_rules! tap_inner {
                ($row_ptr:expr, $ky:expr, $kx:expr, $c:expr) => {{
                    let w = _mm512_loadu_ps(
                        weight
                            .as_ptr()
                            .add($ky * w_ky_stride + $kx * w_kx_stride + $c * w_c_stride),
                    );
                    let base = $row_ptr.add($kx * 3 + $c);
                    let x0 = _mm512_set1_ps(*base);
                    let x1 = _mm512_set1_ps(*base.add(6));
                    let x2 = _mm512_set1_ps(*base.add(12));
                    let x3 = _mm512_set1_ps(*base.add(18));
                    let x4 = _mm512_set1_ps(*base.add(24));
                    let x5 = _mm512_set1_ps(*base.add(30));
                    let x6 = _mm512_set1_ps(*base.add(36));
                    let x7 = _mm512_set1_ps(*base.add(42));
                    acc0 = _mm512_fmadd_ps(x0, w, acc0);
                    acc1 = _mm512_fmadd_ps(x1, w, acc1);
                    acc2 = _mm512_fmadd_ps(x2, w, acc2);
                    acc3 = _mm512_fmadd_ps(x3, w, acc3);
                    acc4 = _mm512_fmadd_ps(x4, w, acc4);
                    acc5 = _mm512_fmadd_ps(x5, w, acc5);
                    acc6 = _mm512_fmadd_ps(x6, w, acc6);
                    acc7 = _mm512_fmadd_ps(x7, w, acc7);
                }};
            }

            tap!(row_in_0, 0);
            tap!(row_in_1, 1);
            tap!(row_in_2, 2);

            if relu {
                acc0 = _mm512_max_ps(acc0, zero);
                acc1 = _mm512_max_ps(acc1, zero);
                acc2 = _mm512_max_ps(acc2, zero);
                acc3 = _mm512_max_ps(acc3, zero);
                acc4 = _mm512_max_ps(acc4, zero);
                acc5 = _mm512_max_ps(acc5, zero);
                acc6 = _mm512_max_ps(acc6, zero);
                acc7 = _mm512_max_ps(acc7, zero);
            }

            let out_ptr = output_chunk.as_mut_ptr().add(row_out_base + ow * 16);
            _mm512_storeu_ps(out_ptr, acc0);
            _mm512_storeu_ps(out_ptr.add(16), acc1);
            _mm512_storeu_ps(out_ptr.add(32), acc2);
            _mm512_storeu_ps(out_ptr.add(48), acc3);
            _mm512_storeu_ps(out_ptr.add(64), acc4);
            _mm512_storeu_ps(out_ptr.add(80), acc5);
            _mm512_storeu_ps(out_ptr.add(96), acc6);
            _mm512_storeu_ps(out_ptr.add(112), acc7);
        }
    }
}

// ── NEON (aarch64) ───────────────────────────────────────────────────
#[cfg(target_arch = "aarch64")]
mod neon {
    use super::*;
    use std::arch::aarch64::*;

    /// NEON implementation — 4-wide lanes. For c_out=16 this is 4 q-regs
    /// per output pixel. Interior pixels skip bounds check.
    ///
    /// # Safety
    /// NEON is mandatory on ARMv8 aarch64; caller has checked
    /// `c_out % 4 == 0` and all slice layouts match the module contract.
    /// Per-chunk NEON worker. See avx2::run_rows for the contract.
    #[target_feature(enable = "neon")]
    #[allow(unsafe_code, clippy::too_many_arguments)]
    pub(super) unsafe fn run_rows(
        input: &[f32],
        weight: &[f32],
        bias: Option<&[f32]>,
        output_chunk: &mut [f32],
        ni: usize,
        oh_start: usize,
        oh_end: usize,
        in_h: usize,
        in_w: usize,
        c_out: usize,
        pad_top: usize,
        pad_left: usize,
        _pad_bottom: usize,
        pad_right: usize,
        activation: Activation,
    ) {
        unsafe {
            let out_w = out_dim(in_w, pad_left, pad_right, 3, 2);
            let c_in: usize = 3;
            let in_row_stride = in_w * c_in;
            let in_batch_stride = in_h * in_row_stride;
            let out_row_stride = out_w * c_out;
            let w_c_stride = c_out;
            let w_kx_stride = c_in * c_out;
            let w_ky_stride = 3 * w_kx_stride;

            let oh_interior_lo = pad_top.div_ceil(2);
            let oh_interior_hi = if in_h + pad_top >= 3 {
                (in_h + pad_top - 3) / 2 + 1
            } else {
                0
            };
            let ow_interior_lo = pad_left.div_ceil(2);
            let ow_interior_hi = if in_w + pad_left >= 3 {
                (in_w + pad_left - 3) / 2 + 1
            } else {
                0
            };

            let zero = vdupq_n_f32(0.0);
            let relu = matches!(activation, Activation::Relu);

            for oh in oh_start..oh_end {
                let ih0 = (oh as isize) * 2 - pad_top as isize;
                let oh_is_interior = oh >= oh_interior_lo && oh < oh_interior_hi;
                for ow in 0..out_w {
                    let iw0 = (ow as isize) * 2 - pad_left as isize;
                    let ow_is_interior = ow >= ow_interior_lo && ow < ow_interior_hi;
                    // Chunk-relative: row 0 of chunk is at offset 0.
                    let out_off = (oh - oh_start) * out_row_stride + ow * c_out;

                    let mut oc_chunk_start = 0usize;
                    while oc_chunk_start < c_out {
                        let mut acc = match bias {
                            Some(b) => vld1q_f32(b.as_ptr().add(oc_chunk_start)),
                            None => vdupq_n_f32(0.0),
                        };
                        if oh_is_interior && ow_is_interior {
                            for ky in 0..3 {
                                let ih = (ih0 + ky as isize) as usize;
                                for kx in 0..3 {
                                    let iw = (iw0 + kx as isize) as usize;
                                    let in_off =
                                        ni * in_batch_stride + ih * in_row_stride + iw * c_in;
                                    let w_off = ky * w_ky_stride + kx * w_kx_stride;
                                    for c in 0..c_in {
                                        let x = vdupq_n_f32(*input.as_ptr().add(in_off + c));
                                        let w = vld1q_f32(
                                            weight
                                                .as_ptr()
                                                .add(w_off + c * w_c_stride + oc_chunk_start),
                                        );
                                        acc = vfmaq_f32(acc, x, w);
                                    }
                                }
                            }
                        } else {
                            for ky in 0..3isize {
                                let ih = ih0 + ky;
                                if ih < 0 || (ih as usize) >= in_h {
                                    continue;
                                }
                                let ih = ih as usize;
                                for kx in 0..3isize {
                                    let iw = iw0 + kx;
                                    if iw < 0 || (iw as usize) >= in_w {
                                        continue;
                                    }
                                    let iw = iw as usize;
                                    let in_off =
                                        ni * in_batch_stride + ih * in_row_stride + iw * c_in;
                                    let w_off =
                                        ky as usize * w_ky_stride + kx as usize * w_kx_stride;
                                    for c in 0..c_in {
                                        let x = vdupq_n_f32(*input.as_ptr().add(in_off + c));
                                        let w = vld1q_f32(
                                            weight
                                                .as_ptr()
                                                .add(w_off + c * w_c_stride + oc_chunk_start),
                                        );
                                        acc = vfmaq_f32(acc, x, w);
                                    }
                                }
                            }
                        }
                        if relu {
                            acc = vmaxq_f32(acc, zero);
                        }
                        vst1q_f32(output_chunk.as_mut_ptr().add(out_off + oc_chunk_start), acc);
                        oc_chunk_start += 4;
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_seeded(len: usize, seed: f32) -> Vec<f32> {
        (0..len)
            .map(|i| ((i as f32 + seed) * 0.017).sin())
            .collect()
    }

    /// Cross-check each active SIMD variant against the scalar reference
    /// on a representative tracker shape plus a smaller shape that exercises
    /// edge pixels.
    fn run_case(
        batch: usize,
        in_h: usize,
        in_w: usize,
        c_out: usize,
        pad: usize,
        activation: Activation,
        with_bias: bool,
    ) {
        let input = make_seeded(batch * in_h * in_w * 3, 0.1);
        let weight = make_seeded(3 * 3 * 3 * c_out, 1.7);
        let bias: Vec<f32> = if with_bias {
            (0..c_out).map(|i| 0.125 * (i as f32 - 7.0)).collect()
        } else {
            vec![]
        };
        let bias_ref = if with_bias {
            Some(bias.as_slice())
        } else {
            None
        };

        let out_h = out_dim(in_h, pad, pad, 3, 2);
        let out_w = out_dim(in_w, pad, pad, 3, 2);
        let out_len = batch * out_h * out_w * c_out;
        let mut ref_out = vec![0f32; out_len];
        scalar_run(
            &input,
            &weight,
            bias_ref,
            &mut ref_out,
            batch,
            in_h,
            in_w,
            c_out,
            pad,
            pad,
            pad,
            pad,
            activation,
        );

        let mut dispatch_out = vec![0f32; out_len];
        conv2d_nhwc_3ch_3x3_s2_padded(
            &input,
            &weight,
            bias_ref,
            &mut dispatch_out,
            batch,
            in_h,
            in_w,
            c_out,
            pad,
            pad,
            pad,
            pad,
            activation,
            None,
        );

        for (i, (a, b)) in ref_out.iter().zip(dispatch_out.iter()).enumerate() {
            let delta = (a - b).abs();
            assert!(
                delta < 1e-4,
                "mismatch at {i}: scalar={a} simd={b} delta={delta}"
            );
        }
    }

    #[test]
    fn tracker_shape_128_pad1_cout16_relu_bias() {
        run_case(1, 128, 128, 16, 1, Activation::Relu, true);
    }

    #[test]
    fn tracker_shape_256_pad1_cout16_relu_bias() {
        run_case(1, 256, 256, 16, 1, Activation::Relu, true);
    }

    #[test]
    fn small_shape_pad0_cout16_none_nobias() {
        // Smallest non-trivial (5→2) to exercise edge pixels.
        run_case(1, 5, 5, 16, 0, Activation::None, false);
    }

    #[test]
    fn edge_padding_cout32_relu() {
        run_case(1, 7, 5, 32, 1, Activation::Relu, true);
    }

    #[test]
    fn batch_gt_1_cout16_none() {
        run_case(2, 12, 12, 16, 1, Activation::None, true);
    }

    /// Session 14 R4 parallel-dispatch correctness: the same kernel
    /// invoked with a `thread_pool` must produce bitwise-identical
    /// output to the sequential path. Input size chosen large enough
    /// to clear `PARALLEL_MIN_OUT_H = 32` so the par branch actually
    /// fires.
    #[cfg(not(miri))]
    #[test]
    fn parallel_matches_sequential_on_tracker_shape() {
        let (batch, in_h, in_w, c_out, pad) = (1usize, 256usize, 256usize, 16usize, 1usize);
        let input = make_seeded(batch * in_h * in_w * 3, 0.1);
        let weight = make_seeded(3 * 3 * 3 * c_out, 1.7);
        let bias: Vec<f32> = (0..c_out).map(|i| 0.125 * (i as f32 - 7.0)).collect();
        let bias_ref = Some(bias.as_slice());

        let out_h = out_dim(in_h, pad, pad, 3, 2);
        let out_w = out_dim(in_w, pad, pad, 3, 2);
        let out_len = batch * out_h * out_w * c_out;

        let mut seq = vec![0f32; out_len];
        conv2d_nhwc_3ch_3x3_s2_padded(
            &input,
            &weight,
            bias_ref,
            &mut seq,
            batch,
            in_h,
            in_w,
            c_out,
            pad,
            pad,
            pad,
            pad,
            Activation::Relu,
            None,
        );

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .expect("build rayon pool");
        let mut par = vec![0f32; out_len];
        conv2d_nhwc_3ch_3x3_s2_padded(
            &input,
            &weight,
            bias_ref,
            &mut par,
            batch,
            in_h,
            in_w,
            c_out,
            pad,
            pad,
            pad,
            pad,
            Activation::Relu,
            Some(&pool),
        );

        for (i, (a, b)) in seq.iter().zip(par.iter()).enumerate() {
            assert_eq!(a.to_bits(), b.to_bits(), "mismatch at {i}: seq={a} par={b}");
        }
    }

    /// Parallel path must be correct on a shape that produces an uneven
    /// chunk split (out_h % n_threads != 0 — the last chunk is shorter).
    /// Skipped under Miri: rayon/crossbeam's epoch GC violates Stacked
    /// Borrows — this is a known rayon+Miri limitation, not a bug here.
    #[cfg(not(miri))]
    #[test]
    fn parallel_handles_uneven_chunk_split() {
        let (batch, in_h, in_w, c_out, pad) = (1usize, 90usize, 90usize, 16usize, 1usize);
        let input = make_seeded(batch * in_h * in_w * 3, 0.2);
        let weight = make_seeded(3 * 3 * 3 * c_out, 0.4);

        let out_h = out_dim(in_h, pad, pad, 3, 2);
        let out_w = out_dim(in_w, pad, pad, 3, 2);
        let out_len = batch * out_h * out_w * c_out;

        let mut seq = vec![0f32; out_len];
        conv2d_nhwc_3ch_3x3_s2_padded(
            &input,
            &weight,
            None,
            &mut seq,
            batch,
            in_h,
            in_w,
            c_out,
            pad,
            pad,
            pad,
            pad,
            Activation::None,
            None,
        );

        // 7 threads, out_h = 45 → ceil(45/7) = 7 rows per chunk;
        // chunks 0-5 = 7 rows, chunk 6 = 3 rows (45 - 6*7 = 3).
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(7)
            .build()
            .unwrap();
        let mut par = vec![0f32; out_len];
        conv2d_nhwc_3ch_3x3_s2_padded(
            &input,
            &weight,
            None,
            &mut par,
            batch,
            in_h,
            in_w,
            c_out,
            pad,
            pad,
            pad,
            pad,
            Activation::None,
            Some(&pool),
        );

        for (i, (a, b)) in seq.iter().zip(par.iter()).enumerate() {
            assert_eq!(a.to_bits(), b.to_bits(), "mismatch at {i}: seq={a} par={b}");
        }
    }
}
