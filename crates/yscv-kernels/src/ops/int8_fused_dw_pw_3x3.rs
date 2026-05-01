//! Streaming INT8 DW (3×3 / 5×5) → PW (1×1) fused chain kernel.
//!
//! Mirror of [`super::int8_fused_pw_dw_3x3`] for the closing pair of the
//! inverted bottleneck. One kernel call covers the tracker chain
//! `QLinearConv(dw kxk) -> DequantizeLinear -> [Relu] -> QuantizeLinear ->
//! QLinearConv(pw 1×1)` with no fp32 fallback inside the window: DW reads
//! NHWC i8 input directly, requantises to i8 in registers, and the PW
//! reduction (matmul into `c_out`) consumes those i8 immediately.
//!
//! ## Contract
//!
//! - Input NHWC i8 `[N, in_h, in_w, c_in]`. Caller is responsible for any
//!   NCHW->NHWC transpose at chain entry. `c_in == c_dw`, the DW group
//!   count.
//! - DW weight KHWC `[kh, kw, c_in]` (the load-time `prepacked_i8_depthwise`
//!   layout — same one explicit `QLinearConv` already consumes via
//!   `env.prepacked_i8_depthwise`).
//! - PW weight: a [`PackedI8B`] for `[K=c_in, N=c_out]` (load-time
//!   prepacked VNNI 4×16 reused).
//! - DW bias / PW bias: optional `&[i32]` of length `c_in` / `c_out`. ONNX
//!   QLinearConv bias dtype is i32; the loader stores it as f32 but values
//!   are integral — the runner casts to i32 before calling us.
//! - Output: NCHW i8 `[N, c_out, out_h, out_w]`. The kernel writes its
//!   internal NHWC scratch and then transposes to NCHW so downstream
//!   consumers (which `exec_qlinear_conv` already feeds NCHW) see the same
//!   bytes the unfused per-op chain produces.
//! - All zero-points on inputs/weights are 0; DW `y_zp` is 0 (the boundary
//!   fold gate enforces that); PW `y_zp` is the chain output's zp and may
//!   be non-zero.
//! - `kh == kw ∈ {3, 5}`, DW `stride ∈ {1, 2}`, symmetric pad
//!   `pad = (kh - 1) / 2`. PW is always 1×1 stride 1, no pad.
//!
//! ## Multi-arch
//!
//! - DW per-row reduction: scalar reference; runtime dispatch picks
//!   AVX-512BW widen-mul (x86_64), AVX2 widen-mul (x86_64), NEON
//!   `vmull_s8` (aarch64), or scalar everywhere else and Miri. The SIMD
//!   variants read directly from the NHWC input slice; no ring buffer.
//! - PW per-row dot: delegated to [`int8_matmul_prepacked_dispatch`] which
//!   already covers AVX-512-VNNI / AVX-VNNI / AVX2-widen / NEON-SDOT /
//!   NEON-i8mm / scalar.
//! - Requant epilogue (i32 → i8): SIMD-dispatched
//!   ([`super::int8_requant::requant_i32_row_to_i8_dispatch`]) — AVX-512BW
//!   16-lane / AVX2 8-lane / NEON 4-lane / scalar tail. Bitwise-identical
//!   to the scalar reference.
//!
//! ## MT
//!
//! `par_chunks_mut` over NHWC output rows when `out_h >= 4` and the rayon
//! pool has more than one thread. Each worker owns private DW i32 / DW i8
//! / PW i32 row scratch — no shared mutable state. After the parallel
//! phase the NHWC scratch is transposed to NCHW into the caller-supplied
//! output.

#![allow(unsafe_code, unsafe_op_in_unsafe_fn)]

use rayon::ThreadPool;
use rayon::prelude::*;

use super::int8_matmul::{PackedI8B, int8_matmul_prepacked_dispatch};
use super::int8_requant::requant_i32_row_to_i8_dispatch;

/// Static parameters for one fused INT8 DW->PW chain call.
///
/// All shape / quant params are resolved by the runner from the underlying
/// `QLinearConv` nodes before dispatch.
#[derive(Clone, Copy, Debug)]
pub struct Int8FusedDwPwParams {
    pub batch: usize,
    pub in_h: usize,
    pub in_w: usize,
    pub c_in: usize,
    pub c_out: usize,
    pub kh: usize,
    pub stride: usize,
    pub pad: usize,
    pub out_h: usize,
    pub out_w: usize,
    /// Apply Relu to the DW i8 output before PW reads it. Equivalent to
    /// the `(*v).max(0)` fold in [`crate::loader::NodeAction::QuantizedQdq`].
    pub dw_relu: bool,
    /// `(dw_x_scale * dw_w_scale) / dw_y_scale`. DW `y_zp` is 0 (chain
    /// gate).
    pub dw_composite: f32,
    /// `(pw_x_scale * pw_w_scale) / pw_y_scale`. PW `x_scale == dw_y_scale`.
    pub pw_composite: f32,
    /// PW output zero-point — chain output `y_zp`, may be non-zero.
    pub pw_y_zp: f32,
}

impl Int8FusedDwPwParams {
    #[inline]
    pub fn input_len(&self) -> usize {
        self.batch * self.in_h * self.in_w * self.c_in
    }

    #[inline]
    pub fn output_len(&self) -> usize {
        self.batch * self.c_out * self.out_h * self.out_w
    }

    #[inline]
    pub fn dw_weight_len(&self) -> usize {
        self.kh * self.kh * self.c_in
    }
}

/// One DW output row — scalar reference. Reads up to `kh` NHWC input
/// rows directly from the batch slice (no ring buffer; the input is
/// already i8 so nothing has to be precomputed).
fn dw_row_from_input_scalar(
    batch_in: &[i8],
    n: usize,
    row_present: [bool; 5],
    ih_top: i64,
    dw_weight: &[i8],
    p: &Int8FusedDwPwParams,
    dw_acc_row: &mut [i32],
) {
    let pad = p.pad as i64;
    let stride = p.stride as i64;
    let row_pixels = p.in_w * p.c_in;
    for ow_i in 0..p.out_w {
        let iw_left = (ow_i as i64) * stride - pad;
        for c in 0..p.c_in {
            let mut acc = 0_i32;
            for ky in 0..p.kh {
                if !row_present[ky] {
                    continue;
                }
                let ih = (ih_top + ky as i64) as usize;
                let row_off = (n * p.in_h + ih) * row_pixels;
                for kx in 0..p.kh {
                    let iw = iw_left + kx as i64;
                    if iw < 0 || (iw as usize) >= p.in_w {
                        continue;
                    }
                    let pix = iw as usize;
                    let x = batch_in[row_off + pix * p.c_in + c] as i32;
                    let w = dw_weight[(ky * p.kh + kx) * p.c_in + c] as i32;
                    acc += x * w;
                }
            }
            dw_acc_row[ow_i * p.c_in + c] = acc;
        }
    }
}

/// AVX2 widen-mul DW row reducer that reads NHWC input directly.
///
/// # Safety
///
/// Caller must guarantee:
/// * `avx2` is runtime-available (gated by `dw_row_from_input_dispatch`).
/// * `p.c_in >= 8` so the 8-lane main loop has at least one full
///   iteration (the dispatcher only delegates here when that holds; the
///   tail handles the remainder in scalar).
/// * `batch_in.len() >= (n+1) * p.in_h * p.in_w * p.c_in`.
/// * `dw_weight.len() >= kh * kh * c_in`.
/// * `dw_acc_row.len() >= p.out_w * p.c_in`.
/// * `row_present[ky]` correctly reflects whether row `ih_top + ky` is
///   in-bounds (out-of-bounds rows are zero-pad).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dw_row_from_input_avx2(
    batch_in: &[i8],
    n: usize,
    row_present: [bool; 5],
    ih_top: i64,
    dw_weight: &[i8],
    p: &Int8FusedDwPwParams,
    dw_acc_row: &mut [i32],
) {
    use std::arch::x86_64::*;
    let pad = p.pad as i64;
    let stride = p.stride as i64;
    let row_pixels = p.in_w * p.c_in;
    let c8 = p.c_in & !7;
    for ow_i in 0..p.out_w {
        let iw_left = (ow_i as i64) * stride - pad;
        for c in (0..c8).step_by(8) {
            let mut acc = _mm256_setzero_si256();
            for ky in 0..p.kh {
                if !row_present[ky] {
                    continue;
                }
                let ih = (ih_top + ky as i64) as usize;
                let row_off = (n * p.in_h + ih) * row_pixels;
                for kx in 0..p.kh {
                    let iw = iw_left + kx as i64;
                    if iw < 0 || (iw as usize) >= p.in_w {
                        continue;
                    }
                    let pix = iw as usize;
                    let x_off = row_off + pix * p.c_in + c;
                    let w_off = (ky * p.kh + kx) * p.c_in + c;
                    let x8 = _mm_loadl_epi64(batch_in.as_ptr().add(x_off) as *const __m128i);
                    let w8 = _mm_loadl_epi64(dw_weight.as_ptr().add(w_off) as *const __m128i);
                    let x16 = _mm256_cvtepi8_epi16(x8);
                    let w16 = _mm256_cvtepi8_epi16(w8);
                    let prod16 = _mm256_mullo_epi16(x16, w16);
                    let prod32 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(prod16));
                    acc = _mm256_add_epi32(acc, prod32);
                }
            }
            _mm256_storeu_si256(
                dw_acc_row.as_mut_ptr().add(ow_i * p.c_in + c) as *mut __m256i,
                acc,
            );
        }
        for c in c8..p.c_in {
            let mut acc = 0_i32;
            for ky in 0..p.kh {
                if !row_present[ky] {
                    continue;
                }
                let ih = (ih_top + ky as i64) as usize;
                let row_off = (n * p.in_h + ih) * row_pixels;
                for kx in 0..p.kh {
                    let iw = iw_left + kx as i64;
                    if iw < 0 || (iw as usize) >= p.in_w {
                        continue;
                    }
                    let pix = iw as usize;
                    let x = batch_in[row_off + pix * p.c_in + c] as i32;
                    let w = dw_weight[(ky * p.kh + kx) * p.c_in + c] as i32;
                    acc += x * w;
                }
            }
            dw_acc_row[ow_i * p.c_in + c] = acc;
        }
    }
}

/// AVX-512BW widen-mul DW row reducer with direct NHWC input addressing.
/// Same input contract as [`dw_row_from_input_avx2`] with a 16-lane main
/// loop.
///
/// # Safety
///
/// Caller must guarantee:
/// * `avx512f` and `avx512bw` are runtime-available (gated by the
///   dispatcher).
/// * `p.c_in >= 16` so the 16-lane main loop has at least one full
///   iteration.
/// * Slice contracts on `batch_in`, `dw_weight`, `dw_acc_row` as in the
///   AVX2 variant.
/// * `row_present[ky]` correctly reflects in-bounds rows.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn dw_row_from_input_avx512(
    batch_in: &[i8],
    n: usize,
    row_present: [bool; 5],
    ih_top: i64,
    dw_weight: &[i8],
    p: &Int8FusedDwPwParams,
    dw_acc_row: &mut [i32],
) {
    use std::arch::x86_64::*;
    let pad = p.pad as i64;
    let stride = p.stride as i64;
    let row_pixels = p.in_w * p.c_in;
    let c16 = p.c_in & !15;
    for ow_i in 0..p.out_w {
        let iw_left = (ow_i as i64) * stride - pad;
        for c in (0..c16).step_by(16) {
            let mut acc = _mm512_setzero_si512();
            for ky in 0..p.kh {
                if !row_present[ky] {
                    continue;
                }
                let ih = (ih_top + ky as i64) as usize;
                let row_off = (n * p.in_h + ih) * row_pixels;
                for kx in 0..p.kh {
                    let iw = iw_left + kx as i64;
                    if iw < 0 || (iw as usize) >= p.in_w {
                        continue;
                    }
                    let pix = iw as usize;
                    let x_off = row_off + pix * p.c_in + c;
                    let w_off = (ky * p.kh + kx) * p.c_in + c;
                    let x = _mm512_cvtepi8_epi32(_mm_loadu_si128(
                        batch_in.as_ptr().add(x_off) as *const __m128i
                    ));
                    let w = _mm512_cvtepi8_epi32(_mm_loadu_si128(
                        dw_weight.as_ptr().add(w_off) as *const __m128i
                    ));
                    acc = _mm512_add_epi32(acc, _mm512_mullo_epi32(x, w));
                }
            }
            _mm512_storeu_si512(
                dw_acc_row.as_mut_ptr().add(ow_i * p.c_in + c) as *mut __m512i,
                acc,
            );
        }
        for c in c16..p.c_in {
            let mut acc = 0_i32;
            for ky in 0..p.kh {
                if !row_present[ky] {
                    continue;
                }
                let ih = (ih_top + ky as i64) as usize;
                let row_off = (n * p.in_h + ih) * row_pixels;
                for kx in 0..p.kh {
                    let iw = iw_left + kx as i64;
                    if iw < 0 || (iw as usize) >= p.in_w {
                        continue;
                    }
                    let pix = iw as usize;
                    let x = batch_in[row_off + pix * p.c_in + c] as i32;
                    let w = dw_weight[(ky * p.kh + kx) * p.c_in + c] as i32;
                    acc += x * w;
                }
            }
            dw_acc_row[ow_i * p.c_in + c] = acc;
        }
    }
}

/// NEON widen-mul DW row reducer with direct NHWC input addressing.
/// Same input contract as [`dw_row_from_input_avx2`].
///
/// # Safety
///
/// Caller must guarantee:
/// * `neon` is runtime-available (gated by the dispatcher).
/// * `p.c_in >= 8`.
/// * Slice contracts on `batch_in`, `dw_weight`, `dw_acc_row` as in the
///   x86 variants.
/// * `row_present[ky]` correctly reflects in-bounds rows.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn dw_row_from_input_neon(
    batch_in: &[i8],
    n: usize,
    row_present: [bool; 5],
    ih_top: i64,
    dw_weight: &[i8],
    p: &Int8FusedDwPwParams,
    dw_acc_row: &mut [i32],
) {
    use std::arch::aarch64::*;
    let pad = p.pad as i64;
    let stride = p.stride as i64;
    let row_pixels = p.in_w * p.c_in;
    let c8 = p.c_in & !7;
    for ow_i in 0..p.out_w {
        let iw_left = (ow_i as i64) * stride - pad;
        for c in (0..c8).step_by(8) {
            let mut acc_lo = vdupq_n_s32(0);
            let mut acc_hi = vdupq_n_s32(0);
            for ky in 0..p.kh {
                if !row_present[ky] {
                    continue;
                }
                let ih = (ih_top + ky as i64) as usize;
                let row_off = (n * p.in_h + ih) * row_pixels;
                for kx in 0..p.kh {
                    let iw = iw_left + kx as i64;
                    if iw < 0 || (iw as usize) >= p.in_w {
                        continue;
                    }
                    let pix = iw as usize;
                    let x_off = row_off + pix * p.c_in + c;
                    let w_off = (ky * p.kh + kx) * p.c_in + c;
                    let xv = vld1_s8(batch_in.as_ptr().add(x_off));
                    let wv = vld1_s8(dw_weight.as_ptr().add(w_off));
                    let prod = vmull_s8(xv, wv);
                    acc_lo = vaddq_s32(acc_lo, vmovl_s16(vget_low_s16(prod)));
                    acc_hi = vaddq_s32(acc_hi, vmovl_s16(vget_high_s16(prod)));
                }
            }
            vst1q_s32(dw_acc_row.as_mut_ptr().add(ow_i * p.c_in + c), acc_lo);
            vst1q_s32(dw_acc_row.as_mut_ptr().add(ow_i * p.c_in + c + 4), acc_hi);
        }
        for c in c8..p.c_in {
            let mut acc = 0_i32;
            for ky in 0..p.kh {
                if !row_present[ky] {
                    continue;
                }
                let ih = (ih_top + ky as i64) as usize;
                let row_off = (n * p.in_h + ih) * row_pixels;
                for kx in 0..p.kh {
                    let iw = iw_left + kx as i64;
                    if iw < 0 || (iw as usize) >= p.in_w {
                        continue;
                    }
                    let pix = iw as usize;
                    let x = batch_in[row_off + pix * p.c_in + c] as i32;
                    let w = dw_weight[(ky * p.kh + kx) * p.c_in + c] as i32;
                    acc += x * w;
                }
            }
            dw_acc_row[ow_i * p.c_in + c] = acc;
        }
    }
}

#[inline]
fn dw_row_from_input_dispatch(
    batch_in: &[i8],
    n: usize,
    row_present: [bool; 5],
    ih_top: i64,
    dw_weight: &[i8],
    p: &Int8FusedDwPwParams,
    dw_acc_row: &mut [i32],
) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx512f")
            && std::is_x86_feature_detected!("avx512bw")
            && p.c_in >= 16
        {
            // SAFETY: `is_x86_feature_detected!` confirmed avx512f+avx512bw
            // at runtime; `p.c_in >= 16` satisfies the 16-lane gate. Slice
            // length contracts on `batch_in`, `dw_weight`, and `dw_acc_row`
            // are upheld by `run_chunk_nhwc` (input is the full NHWC batch
            // slice; weight/acc allocations match `dw_weight_len()` and
            // `out_w * c_in`).
            unsafe {
                dw_row_from_input_avx512(
                    batch_in,
                    n,
                    row_present,
                    ih_top,
                    dw_weight,
                    p,
                    dw_acc_row,
                );
            }
            return;
        }
        if std::is_x86_feature_detected!("avx2") && p.c_in >= 8 {
            // SAFETY: `is_x86_feature_detected!` confirmed avx2 at runtime;
            // `p.c_in >= 8` satisfies the 8-lane gate. Slice contracts as
            // in the avx512 branch above.
            unsafe {
                dw_row_from_input_avx2(batch_in, n, row_present, ih_top, dw_weight, p, dw_acc_row);
            }
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") && p.c_in >= 8 {
            // SAFETY: `is_aarch64_feature_detected!` confirmed neon at
            // runtime; `p.c_in >= 8` satisfies the 8-lane gate. Slice
            // contracts as in the x86 branches above.
            unsafe {
                dw_row_from_input_neon(batch_in, n, row_present, ih_top, dw_weight, p, dw_acc_row);
            }
            return;
        }
    }
    dw_row_from_input_scalar(batch_in, n, row_present, ih_top, dw_weight, p, dw_acc_row);
}

/// One worker: compute a contiguous range `[oh_start, oh_end)` of NHWC i8
/// output rows for batch `n` into `chunk` (shape `[oh_end-oh_start, out_w,
/// c_out]`). Owns private DW i32 / DW i8 / PW i32 row scratch — no shared
/// mutable state.
#[allow(clippy::too_many_arguments)]
fn run_chunk_nhwc(
    batch_in: &[i8],
    dw_weight: &[i8],
    dw_bias: Option<&[i32]>,
    pw_weight_packed: &PackedI8B,
    pw_bias: Option<&[i32]>,
    p: Int8FusedDwPwParams,
    n: usize,
    chunk_nhwc: &mut [i8],
    oh_start: usize,
    oh_end: usize,
) {
    let dw_row_len = p.out_w * p.c_in;
    let mut dw_acc_row = vec![0_i32; dw_row_len];
    let mut dw_i8_row = vec![0_i8; dw_row_len];
    let pw_row_len = p.out_w * p.c_out;
    let mut pw_acc_row = vec![0_i32; pw_row_len];

    let pad = p.pad as i64;
    let stride = p.stride as i64;

    for oh in oh_start..oh_end {
        let ih_top = (oh as i64) * stride - pad;
        let mut row_present = [false; 5];
        for ky in 0..p.kh {
            let ih = ih_top + ky as i64;
            row_present[ky] = ih >= 0 && (ih as usize) < p.in_h;
        }

        dw_row_from_input_dispatch(
            batch_in,
            n,
            row_present,
            ih_top,
            dw_weight,
            &p,
            &mut dw_acc_row,
        );
        requant_i32_row_to_i8_dispatch(
            &dw_acc_row,
            dw_bias,
            p.dw_composite,
            0.0,
            p.dw_relu,
            &mut dw_i8_row,
            p.c_in,
        );

        // PW: matmul one DW row by [c_in, c_out] -> i32 row.
        int8_matmul_prepacked_dispatch(&dw_i8_row, pw_weight_packed, p.out_w, &mut pw_acc_row);

        let oh_off = oh - oh_start;
        let dst_row = &mut chunk_nhwc[oh_off * pw_row_len..(oh_off + 1) * pw_row_len];
        requant_i32_row_to_i8_dispatch(
            &pw_acc_row,
            pw_bias,
            p.pw_composite,
            p.pw_y_zp,
            false,
            dst_row,
            p.c_out,
        );
    }
}

/// In-place NHWC->NCHW transpose of a `[N, H, W, C]` i8 tensor into a
/// pre-allocated `[N, C, H, W]` slice.
fn nhwc_to_nchw_i8(nhwc: &[i8], nchw: &mut [i8], batch: usize, h: usize, w: usize, c: usize) {
    debug_assert_eq!(nhwc.len(), batch * h * w * c);
    debug_assert_eq!(nchw.len(), batch * c * h * w);
    for n in 0..batch {
        for ch in 0..c {
            for y in 0..h {
                for x in 0..w {
                    nchw[((n * c + ch) * h + y) * w + x] = nhwc[((n * h + y) * w + x) * c + ch];
                }
            }
        }
    }
}

/// Streaming INT8 DW->PW chain dispatch.
///
/// Caller contract: see module docs. `output_nchw` is pre-allocated to
/// `p.output_len()` bytes and gets fully written.
#[allow(clippy::too_many_arguments)]
pub fn int8_fused_dw_pw_dispatch(
    input_nhwc: &[i8],
    dw_weight: &[i8],
    dw_bias: Option<&[i32]>,
    pw_weight_packed: &PackedI8B,
    pw_bias: Option<&[i32]>,
    p: Int8FusedDwPwParams,
    output_nchw: &mut [i8],
    thread_pool: Option<&ThreadPool>,
) {
    debug_assert_eq!(input_nhwc.len(), p.input_len());
    debug_assert_eq!(output_nchw.len(), p.output_len());
    debug_assert_eq!(dw_weight.len(), p.dw_weight_len());
    debug_assert_eq!(pw_weight_packed.k(), p.c_in);
    debug_assert_eq!(pw_weight_packed.n(), p.c_out);
    debug_assert!(p.kh == 3 || p.kh == 5);
    debug_assert!(p.stride == 1 || p.stride == 2);
    debug_assert_eq!(p.pad, (p.kh - 1) / 2);
    if let Some(b) = dw_bias {
        debug_assert_eq!(b.len(), p.c_in);
    }
    if let Some(b) = pw_bias {
        debug_assert_eq!(b.len(), p.c_out);
    }

    let nhwc_batch_stride = p.out_h * p.out_w * p.c_out;
    let mut nhwc_out = vec![0_i8; p.batch * nhwc_batch_stride];

    for n in 0..p.batch {
        let batch_out_nhwc = &mut nhwc_out[n * nhwc_batch_stride..(n + 1) * nhwc_batch_stride];
        let row_bytes = p.out_w * p.c_out;

        let par_min_rows = 4;
        let nthreads = thread_pool
            .map(|p| p.current_num_threads().max(1))
            .unwrap_or_else(|| rayon::current_num_threads().max(1));

        if !cfg!(miri) && p.out_h >= par_min_rows && nthreads > 1 {
            let rows_per_chunk = p.out_h.div_ceil(nthreads).max(1);
            let bytes_per_chunk = rows_per_chunk * row_bytes;
            let par_iter = batch_out_nhwc
                .par_chunks_mut(bytes_per_chunk)
                .enumerate()
                .map(|(idx, chunk)| {
                    let oh_start = idx * rows_per_chunk;
                    let oh_end = (oh_start + rows_per_chunk).min(p.out_h);
                    (chunk, oh_start, oh_end)
                });

            if let Some(pool) = thread_pool {
                pool.install(|| {
                    par_iter.for_each(|(chunk, oh_start, oh_end)| {
                        run_chunk_nhwc(
                            input_nhwc,
                            dw_weight,
                            dw_bias,
                            pw_weight_packed,
                            pw_bias,
                            p,
                            n,
                            chunk,
                            oh_start,
                            oh_end,
                        );
                    });
                });
            } else {
                par_iter.for_each(|(chunk, oh_start, oh_end)| {
                    run_chunk_nhwc(
                        input_nhwc,
                        dw_weight,
                        dw_bias,
                        pw_weight_packed,
                        pw_bias,
                        p,
                        n,
                        chunk,
                        oh_start,
                        oh_end,
                    );
                });
            }
        } else {
            run_chunk_nhwc(
                input_nhwc,
                dw_weight,
                dw_bias,
                pw_weight_packed,
                pw_bias,
                p,
                n,
                batch_out_nhwc,
                0,
                p.out_h,
            );
        }
    }

    nhwc_to_nchw_i8(&nhwc_out, output_nchw, p.batch, p.out_h, p.out_w, p.c_out);
}

#[cfg(test)]
mod tests {
    use super::super::int8_matmul::pack_i8_b_for_matmul;
    use super::*;

    fn pseudo_i8(seed: u64, n: usize) -> Vec<i8> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((s >> 33) as i64 % 64 - 32) as i8
            })
            .collect()
    }

    fn pseudo_i32(seed: u64, n: usize, range: i32) -> Vec<i32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let v = (s >> 33) as i64 % (range as i64 * 2) - range as i64;
                v as i32
            })
            .collect()
    }

    /// Reference: per-op QLinearConv chain (DW i32 -> requant i8 -> PW
    /// i32 -> requant i8 NCHW). Mirrors the math the unfused runner
    /// executes; every unit test compares against this.
    #[allow(clippy::too_many_arguments)]
    fn reference_chain(
        input_nhwc: &[i8],
        dw_weight: &[i8],
        dw_bias: Option<&[i32]>,
        pw_weight_kn: &[i8],
        pw_bias: Option<&[i32]>,
        p: Int8FusedDwPwParams,
    ) -> Vec<i8> {
        let mut dw_i8 = vec![0_i8; p.batch * p.out_h * p.out_w * p.c_in];
        for n in 0..p.batch {
            for oh_i in 0..p.out_h {
                for ow_i in 0..p.out_w {
                    for c in 0..p.c_in {
                        let mut acc = 0_i32;
                        for ky in 0..p.kh {
                            let ih = (oh_i * p.stride) as i64 + ky as i64 - p.pad as i64;
                            if ih < 0 || (ih as usize) >= p.in_h {
                                continue;
                            }
                            for kx in 0..p.kh {
                                let iw = (ow_i * p.stride) as i64 + kx as i64 - p.pad as i64;
                                if iw < 0 || (iw as usize) >= p.in_w {
                                    continue;
                                }
                                let x = input_nhwc[((n * p.in_h + ih as usize) * p.in_w
                                    + iw as usize)
                                    * p.c_in
                                    + c] as i32;
                                let wv = dw_weight[(ky * p.kh + kx) * p.c_in + c] as i32;
                                acc += x * wv;
                            }
                        }
                        if let Some(b) = dw_bias {
                            acc += b[c];
                        }
                        let v = (acc as f32) * p.dw_composite + 0.0;
                        let mut q = v.round().clamp(-128.0, 127.0) as i8;
                        if p.dw_relu && q < 0 {
                            q = 0;
                        }
                        dw_i8[((n * p.out_h + oh_i) * p.out_w + ow_i) * p.c_in + c] = q;
                    }
                }
            }
        }
        let mut out = vec![0_i8; p.batch * p.c_out * p.out_h * p.out_w];
        for n in 0..p.batch {
            for oh_i in 0..p.out_h {
                for ow_i in 0..p.out_w {
                    for o in 0..p.c_out {
                        let mut acc = 0_i32;
                        for ci in 0..p.c_in {
                            let x =
                                dw_i8[((n * p.out_h + oh_i) * p.out_w + ow_i) * p.c_in + ci] as i32;
                            let wv = pw_weight_kn[ci * p.c_out + o] as i32;
                            acc += x * wv;
                        }
                        if let Some(b) = pw_bias {
                            acc += b[o];
                        }
                        let v = (acc as f32) * p.pw_composite + p.pw_y_zp;
                        let q = v.round().clamp(-128.0, 127.0) as i8;
                        out[((n * p.c_out + o) * p.out_h + oh_i) * p.out_w + ow_i] = q;
                    }
                }
            }
        }
        out
    }

    #[allow(clippy::too_many_arguments)]
    fn make_params(
        batch: usize,
        in_h: usize,
        in_w: usize,
        c_in: usize,
        c_out: usize,
        kh: usize,
        stride: usize,
        dw_relu: bool,
        dw_composite: f32,
        pw_composite: f32,
        pw_y_zp: f32,
    ) -> Int8FusedDwPwParams {
        let pad = (kh - 1) / 2;
        let out_h = (in_h + 2 * pad - kh) / stride + 1;
        let out_w = (in_w + 2 * pad - kh) / stride + 1;
        Int8FusedDwPwParams {
            batch,
            in_h,
            in_w,
            c_in,
            c_out,
            kh,
            stride,
            pad,
            out_h,
            out_w,
            dw_relu,
            dw_composite,
            pw_composite,
            pw_y_zp,
        }
    }

    fn check_shape(p: Int8FusedDwPwParams) {
        let input = pseudo_i8(0xA, p.input_len());
        let dw_weight = pseudo_i8(0xB, p.dw_weight_len());
        let pw_weight = pseudo_i8(0xC, p.c_in * p.c_out);
        let dw_bias = pseudo_i32(0xD, p.c_in, 1024);
        let pw_bias = pseudo_i32(0xE, p.c_out, 1024);

        let expected = reference_chain(
            &input,
            &dw_weight,
            Some(&dw_bias),
            &pw_weight,
            Some(&pw_bias),
            p,
        );

        let packed = pack_i8_b_for_matmul(&pw_weight, p.c_in, p.c_out);
        let mut got = vec![0_i8; p.output_len()];
        int8_fused_dw_pw_dispatch(
            &input,
            &dw_weight,
            Some(&dw_bias),
            &packed,
            Some(&pw_bias),
            p,
            &mut got,
            None,
        );
        assert_eq!(
            got, expected,
            "mismatch shape b={} ih={} iw={} c_in={} c_out={} kh={} s={} relu={}",
            p.batch, p.in_h, p.in_w, p.c_in, p.c_out, p.kh, p.stride, p.dw_relu
        );
    }

    /// Tracker `/xif2_0` closing pair: c_in=96 c_out=24 ih=64 stride=1
    /// (DW shape after PW expand reverts to original c_in for the
    /// reduce). Smaller shape used here to keep the test fast.
    #[test]
    fn xif2_0_close_3x3_s1_relu() {
        let p = make_params(1, 32, 32, 96, 24, 3, 1, true, 0.012, 0.0085, 0.0);
        check_shape(p);
    }

    #[test]
    fn xif3_0_close_3x3_s1_relu() {
        let p = make_params(1, 16, 16, 144, 32, 3, 1, true, 0.010, 0.0073, 0.0);
        check_shape(p);
    }

    #[test]
    fn xif4_0_close_3x3_s1_relu() {
        let p = make_params(1, 16, 16, 192, 64, 3, 1, true, 0.008, 0.0065, 0.0);
        check_shape(p);
    }

    #[test]
    fn no_relu_3x3_s1() {
        let p = make_params(1, 12, 12, 32, 16, 3, 1, false, 0.02, 0.015, 0.0);
        check_shape(p);
    }

    #[test]
    fn small_5x5_s1() {
        let p = make_params(1, 12, 12, 24, 8, 5, 1, true, 0.018, 0.011, 1.0);
        check_shape(p);
    }

    #[test]
    fn batch_3x3_s2_no_bias_offsets() {
        let p = make_params(2, 14, 16, 24, 8, 3, 2, false, 0.02, 0.013, -2.0);
        let input = pseudo_i8(0x11, p.input_len());
        let dw_weight = pseudo_i8(0x22, p.dw_weight_len());
        let pw_weight = pseudo_i8(0x33, p.c_in * p.c_out);

        let expected = reference_chain(&input, &dw_weight, None, &pw_weight, None, p);
        let packed = pack_i8_b_for_matmul(&pw_weight, p.c_in, p.c_out);
        let mut got = vec![0_i8; p.output_len()];
        int8_fused_dw_pw_dispatch(&input, &dw_weight, None, &packed, None, p, &mut got, None);
        assert_eq!(got, expected);
    }
}
