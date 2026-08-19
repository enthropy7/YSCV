//! Shared SIMD requant epilogue used by the fused INT8 chain kernels
//! ([`super::int8_fused_pw_dw_3x3`], [`super::int8_fused_dw_pw_3x3`]).
//!
//! Both fused kernels must turn an `i32` accumulator row into an `i8` row
//! via a single composite scale, optional per-channel bias, and an optional
//! Relu fold. The reference scalar form is:
//!
//! ```text
//! let mut a = acc[i];
//! if let Some(b) = bias { a += b[ch]; }
//! let v = (a as f32) * composite + y_zp;
//! let mut q = v.round_ties_even().clamp(-128.0, 127.0) as i8;
//! if relu && q < 0 { q = 0; }
//! ```
//!
//! The SIMD variants in this module match the scalar form bit-for-bit:
//!
//! * Rounding is round-half-to-even (IEEE default), matching the ONNX
//!   QuantizeLinear spec and ONNX Runtime. Each arch has a native
//!   nearest-even instruction — `_mm512_roundscale_ps` /
//!   `_mm256_round_ps` with `_MM_FROUND_TO_NEAREST_INT`, and NEON's
//!   `vrndnq_f32` — so the value goes straight through without the
//!   sign/abs/bias emulation the old round-half-away path needed. The
//!   scalar reference uses `f32::round_ties_even()`; SIMD and scalar
//!   agree byte-for-byte across `[-128, 127]` (verified by the
//!   `matches_scalar_realistic_chain_inputs_stress` parity test).
//! * Relu is folded as `max(v, 0)` BEFORE the round-clamp-cvt sequence.
//!   The scalar reference clamps the final i8 to `>= 0`, but for any input
//!   `v` the two formulations produce the same final i8: when scalar
//!   would emit a negative i8, `max(v, 0) = 0` rounds to 0; when scalar
//!   would emit a non-negative i8, `max(v, 0) = v` and the rest of the
//!   pipeline is unchanged.
//!
//! The dispatch picks the widest path the host supports at runtime
//! (`AVX-512BW` → `AVX2 + SSE4.1` → `NEON` → scalar). Each SIMD path falls
//! back to the scalar tail when the channel count isn't a clean multiple
//! of the lane width — closing pointwise layers in real models can have
//! `c_out` as small as 1, so the tail must be tight.

#[allow(unused_imports)]
use std::sync::atomic::{AtomicU8, Ordering};

#[cfg(target_arch = "x86_64")]
const PATH_UNCHECKED: u8 = 0;
#[cfg(target_arch = "x86_64")]
const PATH_AVX512: u8 = 1;
#[cfg(target_arch = "x86_64")]
const PATH_AVX2: u8 = 2;
#[cfg(target_arch = "x86_64")]
const PATH_SCALAR: u8 = 3;

#[cfg(target_arch = "x86_64")]
static X86_PATH: AtomicU8 = AtomicU8::new(PATH_UNCHECKED);

#[cfg(target_arch = "x86_64")]
fn x86_path() -> u8 {
    let cached = X86_PATH.load(Ordering::Relaxed);
    if cached != PATH_UNCHECKED {
        return cached;
    }
    let features = crate::host_cpu().features;
    let path = if features.x86_avx512_bw() {
        PATH_AVX512
    } else if features.x86_avx2_sse41() {
        PATH_AVX2
    } else {
        PATH_SCALAR
    };
    X86_PATH.store(path, Ordering::Relaxed);
    path
}

/// Quantize an `[pixels * c]` i32 accumulator row into an `[pixels * c]`
/// i8 row using a single composite scale, an optional per-channel `bias`
/// (length `c`), and an optional Relu fold.
///
/// The two fused INT8 chain kernels both call this exact dispatcher;
/// keeping the logic shared guarantees they agree byte-for-byte on the
/// epilogue and removes the mismatch risk that comes with two near-copies.
#[inline]
pub fn requant_i32_row_to_i8_dispatch(
    acc: &[i32],
    bias: Option<&[i32]>,
    composite: f32,
    y_zp: f32,
    relu: bool,
    out: &mut [i8],
    c: usize,
) {
    debug_assert_eq!(acc.len(), out.len());
    debug_assert!(c > 0);
    debug_assert_eq!(acc.len() % c, 0);
    if let Some(b) = bias {
        debug_assert_eq!(b.len(), c);
    }

    #[cfg(target_arch = "x86_64")]
    {
        match x86_path() {
            PATH_AVX512 => {
                // SAFETY: x86_path() returns AVX512 only when the host
                // reports avx512f + avx512bw via cpuid. The function is
                // gated with the matching `target_feature` annotations.
                #[allow(unsafe_code)]
                unsafe {
                    requant_avx512(acc, bias, composite, y_zp, relu, out, c);
                }
                return;
            }
            PATH_AVX2 => {
                // SAFETY: x86_path() returns AVX2 only when the host
                // reports avx2 + sse4.1 via cpuid; the inner function is
                // annotated `target_feature(enable = "avx2,sse4.1")`.
                #[allow(unsafe_code)]
                unsafe {
                    requant_avx2(acc, bias, composite, y_zp, relu, out, c);
                }
                return;
            }
            _ => {}
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if crate::host_cpu().features.neon {
            // SAFETY: NEON is feature-detected at runtime; the inner
            // function is annotated `target_feature(enable = "neon")`.
            #[allow(unsafe_code)]
            unsafe {
                requant_neon(acc, bias, composite, y_zp, relu, out, c);
            }
            return;
        }
    }

    requant_scalar(acc, bias, composite, y_zp, relu, out, c);
}

/// Scalar reference. Bitwise oracle for the SIMD paths; also the actual
/// path on hosts without AVX2/AVX-512/NEON.
#[inline]
pub(crate) fn requant_scalar(
    acc: &[i32],
    bias: Option<&[i32]>,
    composite: f32,
    y_zp: f32,
    relu: bool,
    out: &mut [i8],
    c: usize,
) {
    let pixels = acc.len() / c;
    for p in 0..pixels {
        let row_acc = &acc[p * c..(p + 1) * c];
        let row_out = &mut out[p * c..(p + 1) * c];
        for ch in 0..c {
            let mut a = row_acc[ch];
            if let Some(b) = bias {
                a = a.wrapping_add(b[ch]);
            }
            let v = (a as f32) * composite + y_zp;
            let mut q = v.round_ties_even().clamp(-128.0, 127.0) as i8;
            if relu && q < 0 {
                q = 0;
            }
            row_out[ch] = q;
        }
    }
}

// ---------------------------------------------------------------------------
// AVX-512 (16 lanes) — 1 KB host scratch per call, no allocation.
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
#[allow(unsafe_code, clippy::too_many_arguments)]
unsafe fn requant_avx512(
    acc: &[i32],
    bias: Option<&[i32]>,
    composite: f32,
    y_zp: f32,
    relu: bool,
    out: &mut [i8],
    c: usize,
) {
    use std::arch::x86_64::*;

    unsafe {
        let comp = _mm512_set1_ps(composite);
        let zp = _mm512_set1_ps(y_zp);
        let zero = _mm512_setzero_ps();
        let lo = _mm512_set1_ps(-128.0);
        let hi = _mm512_set1_ps(127.0);

        let pixels = acc.len() / c;
        let lane = 16;
        let main = (c / lane) * lane;
        for p in 0..pixels {
            let row_acc = &acc[p * c..(p + 1) * c];
            let row_out = &mut out[p * c..(p + 1) * c];
            let mut ch = 0;
            while ch < main {
                let mut ai = _mm512_loadu_si512(row_acc.as_ptr().add(ch).cast::<__m512i>());
                if let Some(b) = bias {
                    let bi = _mm512_loadu_si512(b.as_ptr().add(ch).cast::<__m512i>());
                    ai = _mm512_add_epi32(ai, bi);
                }
                let af = _mm512_cvtepi32_ps(ai);
                let v = _mm512_add_ps(_mm512_mul_ps(af, comp), zp);
                // Round half to even (IEEE default) to match ONNX / ORT.
                let rounded =
                    _mm512_roundscale_ps::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(v);
                let mut clamped = _mm512_min_ps(_mm512_max_ps(rounded, lo), hi);
                if relu {
                    clamped = _mm512_max_ps(clamped, zero);
                }
                let i32s = _mm512_cvtps_epi32(clamped);
                let i8s = _mm512_cvtsepi32_epi8(i32s);
                _mm_storeu_si128(row_out.as_mut_ptr().add(ch).cast::<__m128i>(), i8s);
                ch += lane;
            }
            // Channel tail: same arithmetic, scalar.
            while ch < c {
                let mut a = row_acc[ch];
                if let Some(b) = bias {
                    a = a.wrapping_add(b[ch]);
                }
                let v = (a as f32) * composite + y_zp;
                let mut q = v.round_ties_even().clamp(-128.0, 127.0) as i8;
                if relu && q < 0 {
                    q = 0;
                }
                row_out[ch] = q;
                ch += 1;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// AVX2 (8 lanes) — same arithmetic as AVX-512, narrower vectors.
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,sse4.1")]
#[allow(unsafe_code, clippy::too_many_arguments)]
unsafe fn requant_avx2(
    acc: &[i32],
    bias: Option<&[i32]>,
    composite: f32,
    y_zp: f32,
    relu: bool,
    out: &mut [i8],
    c: usize,
) {
    use std::arch::x86_64::*;

    unsafe {
        let comp = _mm256_set1_ps(composite);
        let zp = _mm256_set1_ps(y_zp);
        let zero = _mm256_setzero_ps();
        let lo = _mm256_set1_ps(-128.0);
        let hi = _mm256_set1_ps(127.0);

        let pixels = acc.len() / c;
        let lane = 8;
        let main = (c / lane) * lane;
        for p in 0..pixels {
            let row_acc = &acc[p * c..(p + 1) * c];
            let row_out = &mut out[p * c..(p + 1) * c];
            let mut ch = 0;
            while ch < main {
                let mut ai = _mm256_loadu_si256(row_acc.as_ptr().add(ch).cast::<__m256i>());
                if let Some(b) = bias {
                    let bi = _mm256_loadu_si256(b.as_ptr().add(ch).cast::<__m256i>());
                    ai = _mm256_add_epi32(ai, bi);
                }
                let af = _mm256_cvtepi32_ps(ai);
                let v = _mm256_add_ps(_mm256_mul_ps(af, comp), zp);
                // Round half to even (IEEE default) to match ONNX / ORT.
                let rounded =
                    _mm256_round_ps::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(v);
                let mut clamped = _mm256_min_ps(_mm256_max_ps(rounded, lo), hi);
                if relu {
                    clamped = _mm256_max_ps(clamped, zero);
                }
                let i32s = _mm256_cvtps_epi32(clamped);
                let lo128 = _mm256_castsi256_si128(i32s);
                let hi128 = _mm256_extracti128_si256::<1>(i32s);
                let i16s = _mm_packs_epi32(lo128, hi128);
                let i8s = _mm_packs_epi16(i16s, _mm_setzero_si128());
                _mm_storel_epi64(row_out.as_mut_ptr().add(ch).cast::<__m128i>(), i8s);
                ch += lane;
            }
            while ch < c {
                let mut a = row_acc[ch];
                if let Some(b) = bias {
                    a = a.wrapping_add(b[ch]);
                }
                let v = (a as f32) * composite + y_zp;
                let mut q = v.round_ties_even().clamp(-128.0, 127.0) as i8;
                if relu && q < 0 {
                    q = 0;
                }
                row_out[ch] = q;
                ch += 1;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// NEON (4 lanes) — aarch64.
// ---------------------------------------------------------------------------

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(unsafe_code, clippy::too_many_arguments)]
unsafe fn requant_neon(
    acc: &[i32],
    bias: Option<&[i32]>,
    composite: f32,
    y_zp: f32,
    relu: bool,
    out: &mut [i8],
    c: usize,
) {
    use std::arch::aarch64::*;

    unsafe {
        let comp = vdupq_n_f32(composite);
        let zp = vdupq_n_f32(y_zp);
        let zero = vdupq_n_f32(0.0);
        let lo = vdupq_n_f32(-128.0);
        let hi = vdupq_n_f32(127.0);

        let pixels = acc.len() / c;
        let lane = 4;
        let main = (c / lane) * lane;
        for p in 0..pixels {
            let row_acc = &acc[p * c..(p + 1) * c];
            let row_out = &mut out[p * c..(p + 1) * c];
            let mut ch = 0;
            while ch < main {
                let mut ai = vld1q_s32(row_acc.as_ptr().add(ch));
                if let Some(b) = bias {
                    let bi = vld1q_s32(b.as_ptr().add(ch));
                    ai = vaddq_s32(ai, bi);
                }
                let af = vcvtq_f32_s32(ai);
                let v = vaddq_f32(vmulq_f32(af, comp), zp);
                // Round half to even (IEEE default, `vrndnq_f32`) to match the
                // ONNX QuantizeLinear spec and ORT.
                let rounded = vrndnq_f32(v);
                let mut clamped = vminq_f32(vmaxq_f32(rounded, lo), hi);
                if relu {
                    clamped = vmaxq_f32(clamped, zero);
                }
                let i32s = vcvtq_s32_f32(clamped);
                // Saturating narrow 32→16, 16→8.
                let i16s = vqmovn_s32(i32s);
                let i16s_padded = vcombine_s16(i16s, vdup_n_s16(0));
                let i8s = vqmovn_s16(i16s_padded);
                // Store 4 bytes from the low half of the 8-byte result.
                let bytes = std::slice::from_raw_parts(&i8s as *const _ as *const u8, 4);
                row_out[ch..ch + 4].copy_from_slice(std::mem::transmute::<&[u8], &[i8]>(bytes));
                ch += lane;
            }
            while ch < c {
                let mut a = row_acc[ch];
                if let Some(b) = bias {
                    a = a.wrapping_add(b[ch]);
                }
                let v = (a as f32) * composite + y_zp;
                let mut q = v.round_ties_even().clamp(-128.0, 127.0) as i8;
                if relu && q < 0 {
                    q = 0;
                }
                row_out[ch] = q;
                ch += 1;
            }
        }
    }
}

// ===========================================================================
// Per-channel composite requantize for QLinearConv path1/im2col epilogues.
// `acc` is `[pixels, c]` row-major (NHWC). Unlike `requant_i32_row_to_i8`
// above (one composite scale, used by the fused DQ->Q boundaries), here each
// output channel carries its own `corr[ch]` (i32, the folded bias − x_zp·Σw)
// and `composite[ch]` (f32, x_scale·w_scale[ch]/y_scale). Bit-identical to the
// scalar epilogue it replaces:
//   ((acc + corr) as f32 * composite + y_zp).round_ties_even().clamp(-128,127)
// The vectorised path folds round-ties-even and clamp into FCVTNS + double
// saturating-narrow (SQXTN), which is exactly what the scalar oracle computes.
// ===========================================================================

/// Dispatch the per-channel-composite requantize to the widest host SIMD path.
#[inline]
pub fn requant_i8_per_channel_dispatch(
    acc: &[i32],
    corr: &[i32],
    composite: &[f32],
    y_zp: f32,
    out: &mut [i8],
    c: usize,
) {
    debug_assert_eq!(acc.len(), out.len());
    debug_assert!(c > 0);
    debug_assert_eq!(acc.len() % c, 0);
    debug_assert_eq!(corr.len(), c);
    debug_assert_eq!(composite.len(), c);

    #[cfg(target_arch = "aarch64")]
    {
        if crate::host_cpu().features.neon {
            // SAFETY: NEON feature-detected at runtime; the inner function is
            // annotated `target_feature(enable = "neon")`.
            #[allow(unsafe_code)]
            unsafe {
                requant_i8_per_channel_neon(acc, corr, composite, y_zp, out, c);
            }
            return;
        }
    }

    requant_i8_per_channel_scalar(acc, corr, composite, y_zp, out, c);
}

/// Fused Squeeze-Excite scale + quantize for an NHWC f32 feature:
/// `out[p,ch] = clamp(round(feat[p,ch]·gate[batch,ch] / y_scale + y_zp))` as i8.
/// `feat` is `[N·H·W, C]` row-major (channel-contiguous), `gate` is `[N, C]`.
/// Bit-identical to `Mul(feat, gate)` followed by ONNX `QuantizeLinear`.
#[inline]
pub fn se_mul_quantize_nhwc_dispatch(
    feat: &[f32],
    gate: &[f32],
    n: usize,
    c: usize,
    y_scale: f32,
    y_zp: f32,
    out: &mut [i8],
) {
    debug_assert_eq!(feat.len(), out.len());
    debug_assert!(c > 0 && n > 0);
    debug_assert_eq!(feat.len() % c, 0);
    #[cfg(target_arch = "aarch64")]
    {
        if crate::host_cpu().features.neon {
            // SAFETY: NEON feature-detected; inner fn is `target_feature=neon`.
            #[allow(unsafe_code)]
            unsafe {
                se_mul_quantize_nhwc_neon(feat, gate, n, c, y_scale, y_zp, out);
            }
            return;
        }
    }
    se_mul_quantize_nhwc_scalar(feat, gate, n, c, y_scale, y_zp, out);
}

#[inline]
fn se_mul_quantize_nhwc_scalar(
    feat: &[f32],
    gate: &[f32],
    n: usize,
    c: usize,
    y_scale: f32,
    y_zp: f32,
    out: &mut [i8],
) {
    let pixels = feat.len() / c;
    let ppb = (pixels / n).max(1);
    for p in 0..pixels {
        let g = &gate[(p / ppb) * c..];
        let frow = &feat[p * c..p * c + c];
        let orow = &mut out[p * c..p * c + c];
        for ch in 0..c {
            orow[ch] = (frow[ch] * g[ch] / y_scale + y_zp)
                .round_ties_even()
                .clamp(-128.0, 127.0) as i8;
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(unsafe_code, clippy::too_many_arguments)]
unsafe fn se_mul_quantize_nhwc_neon(
    feat: &[f32],
    gate: &[f32],
    n: usize,
    c: usize,
    y_scale: f32,
    y_zp: f32,
    out: &mut [i8],
) {
    use std::arch::aarch64::*;
    unsafe {
        let ys = vdupq_n_f32(y_scale);
        let yr = vdupq_n_f32(1.0 / y_scale);
        let zp = vdupq_n_f32(y_zp);
        let lo = vdupq_n_f32(-128.0);
        let hi = vdupq_n_f32(127.0);
        let pixels = feat.len() / c;
        let ppb = (pixels / n).max(1);
        let main = (c / 4) * 4;
        for p in 0..pixels {
            let g = &gate[(p / ppb) * c..];
            let frow = &feat[p * c..p * c + c];
            let orow = &mut out[p * c..p * c + c];
            let mut ch = 0;
            while ch < main {
                let f = vld1q_f32(frow.as_ptr().add(ch));
                let gg = vld1q_f32(g.as_ptr().add(ch));
                let v = vaddq_f32(super::div_invariant_neon(vmulq_f32(f, gg), ys, yr), zp);
                let r = vrndnq_f32(v);
                let clamped = vminq_f32(vmaxq_f32(r, lo), hi);
                let i32s = vcvtq_s32_f32(clamped);
                let i8s = vqmovn_s16(vcombine_s16(vqmovn_s32(i32s), vdup_n_s16(0)));
                let bytes = std::slice::from_raw_parts(&i8s as *const _ as *const u8, 4);
                orow[ch..ch + 4].copy_from_slice(std::mem::transmute::<&[u8], &[i8]>(bytes));
                ch += 4;
            }
            while ch < c {
                orow[ch] = (frow[ch] * g[ch] / y_scale + y_zp)
                    .round_ties_even()
                    .clamp(-128.0, 127.0) as i8;
                ch += 1;
            }
        }
    }
}

/// Scalar reference / bitwise oracle for the per-channel-composite requantize.
#[inline]
pub(crate) fn requant_i8_per_channel_scalar(
    acc: &[i32],
    corr: &[i32],
    composite: &[f32],
    y_zp: f32,
    out: &mut [i8],
    c: usize,
) {
    let pixels = acc.len() / c;
    for p in 0..pixels {
        let row_acc = &acc[p * c..(p + 1) * c];
        let row_out = &mut out[p * c..(p + 1) * c];
        for ch in 0..c {
            let a = row_acc[ch].wrapping_add(corr[ch]);
            let v = (a as f32) * composite[ch] + y_zp;
            row_out[ch] = v.round_ties_even().clamp(-128.0, 127.0) as i8;
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(unsafe_code)]
unsafe fn requant_i8_per_channel_neon(
    acc: &[i32],
    corr: &[i32],
    composite: &[f32],
    y_zp: f32,
    out: &mut [i8],
    c: usize,
) {
    use std::arch::aarch64::*;

    // One channel-lane of the pipeline: load acc+corr, `(acc+corr)·comp + zp`,
    // FCVTNS (round ties to even, i32-saturating) — matches `round_ties_even`.
    #[inline(always)]
    unsafe fn lane4(
        row_acc: *const i32,
        corr: *const i32,
        composite: *const f32,
        zp: float32x4_t,
        ch: usize,
    ) -> int32x4_t {
        unsafe {
            let ai = vld1q_s32(row_acc.add(ch));
            let ci = vld1q_s32(corr.add(ch));
            let comp = vld1q_f32(composite.add(ch));
            let af = vcvtq_f32_s32(vaddq_s32(ai, ci));
            vcvtnq_s32_f32(vaddq_f32(vmulq_f32(af, comp), zp))
        }
    }

    unsafe {
        let zp = vdupq_n_f32(y_zp);
        let pixels = acc.len() / c;
        for p in 0..pixels {
            let row_acc = acc.as_ptr().add(p * c);
            let row_out = out.as_mut_ptr().add(p * c);
            let cp = corr.as_ptr();
            let fp = composite.as_ptr();
            let mut ch = 0;
            // 16 channels/iter: 4 lanes → double saturating-narrow → one
            // 16-byte store (`vst1q_s8`), amortising the i32→i8 pack and
            // killing the per-4-byte `copy_from_slice` memcpy call.
            while ch + 16 <= c {
                let r0 = lane4(row_acc, cp, fp, zp, ch);
                let r1 = lane4(row_acc, cp, fp, zp, ch + 4);
                let r2 = lane4(row_acc, cp, fp, zp, ch + 8);
                let r3 = lane4(row_acc, cp, fp, zp, ch + 12);
                let s01 = vcombine_s16(vqmovn_s32(r0), vqmovn_s32(r1));
                let s23 = vcombine_s16(vqmovn_s32(r2), vqmovn_s32(r3));
                let b = vcombine_s8(vqmovn_s16(s01), vqmovn_s16(s23));
                vst1q_s8(row_out.add(ch), b);
                ch += 16;
            }
            // 8 channels/iter: 2 lanes → one 8-byte store.
            while ch + 8 <= c {
                let r0 = lane4(row_acc, cp, fp, zp, ch);
                let r1 = lane4(row_acc, cp, fp, zp, ch + 4);
                let b = vqmovn_s16(vcombine_s16(vqmovn_s32(r0), vqmovn_s32(r1)));
                vst1_s8(row_out.add(ch), b);
                ch += 8;
            }
            // Scalar tail (< 8 channels).
            while ch < c {
                let a = (*row_acc.add(ch)).wrapping_add(*cp.add(ch));
                let v = (a as f32) * *fp.add(ch) + y_zp;
                *row_out.add(ch) = v.round_ties_even().clamp(-128.0, 127.0) as i8;
                ch += 1;
            }
        }
    }
}

// ===========================================================================
// Fused DequantizeLinear -> [Relu | Clip] -> QuantizeLinear across an int8
// rescale boundary. Between two int8 convs the graph does i8 -> DQ -> f32 ->
// act -> f32 -> Q -> i8; folding it into one pass keeps the activation in i8
// (no intermediate f32 buffer, one pass instead of three). Bit-identical to the
// sequence: DQ `(v - zp_in) * scale_in`, the activation as `f.clamp(lo, hi)` on
// the real value (Relu is `(0, inf)`, no activation `(-inf, inf)`), Q
// `(f / scale_out + zp_out).round_ties_even().clamp(-128,127)`. The `/` (not
// reciprocal-multiply) and `round_ties_even` match the scalar ops exactly.
// ===========================================================================

/// Dispatch the fused i8->i8 rescale (`DQ -> clamp -> Q`) to the widest host
/// SIMD path. See module-level notes; `clamp` folds the activation into the
/// boundary — `(0.0, f32::INFINITY)` for Relu, `(-f32::INFINITY,
/// f32::INFINITY)` for none, the node's bounds for a Clip.
pub fn requant_i8_dq_relu_q_dispatch(
    input: &[i8],
    scale_in: f32,
    zp_in: f32,
    scale_out: f32,
    zp_out: f32,
    clamp: (f32, f32),
    out: &mut [i8],
) {
    debug_assert_eq!(input.len(), out.len());
    #[cfg(target_arch = "aarch64")]
    {
        if crate::host_cpu().features.neon {
            // SAFETY: NEON feature-detected at runtime.
            #[allow(unsafe_code)]
            unsafe {
                requant_i8_dq_relu_q_neon(input, scale_in, zp_in, scale_out, zp_out, clamp, out);
            }
            return;
        }
    }
    requant_i8_dq_relu_q_scalar(input, scale_in, zp_in, scale_out, zp_out, clamp, out);
}

/// Scalar reference / bitwise oracle.
#[inline]
pub(crate) fn requant_i8_dq_relu_q_scalar(
    input: &[i8],
    scale_in: f32,
    zp_in: f32,
    scale_out: f32,
    zp_out: f32,
    clamp: (f32, f32),
    out: &mut [i8],
) {
    for (o, &q) in out.iter_mut().zip(input) {
        let f = ((q as f32 - zp_in) * scale_in).clamp(clamp.0, clamp.1);
        *o = (f / scale_out + zp_out)
            .round_ties_even()
            .clamp(-128.0, 127.0) as i8;
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(unsafe_code)]
unsafe fn requant_i8_dq_relu_q_neon(
    input: &[i8],
    scale_in: f32,
    zp_in: f32,
    scale_out: f32,
    zp_out: f32,
    clamp: (f32, f32),
    out: &mut [i8],
) {
    use std::arch::aarch64::*;
    unsafe {
        let v_zp_in = vdupq_n_f32(zp_in);
        let v_sin = vdupq_n_f32(scale_in);
        let v_sout = vdupq_n_f32(scale_out);
        let v_rout = vdupq_n_f32(1.0 / scale_out);
        let v_zp_out = vdupq_n_f32(zp_out);
        // Infinities make the unclamped case a no-op without a branch.
        let act_lo = vdupq_n_f32(clamp.0);
        let act_hi = vdupq_n_f32(clamp.1);
        let lo = vdupq_n_f32(-128.0);
        let hi = vdupq_n_f32(127.0);
        let n = input.len();
        let main = n & !7;
        let mut i = 0;
        while i < main {
            let q8 = vld1_s8(input.as_ptr().add(i));
            let q16 = vmovl_s8(q8);
            let chunk = |f32x4: float32x4_t| -> int16x4_t {
                let f = vmulq_f32(vsubq_f32(f32x4, v_zp_in), v_sin);
                let f = vminq_f32(vmaxq_f32(f, act_lo), act_hi);
                let q = vaddq_f32(super::div_invariant_neon(f, v_sout, v_rout), v_zp_out);
                let q = vrndnq_f32(q);
                let q = vminq_f32(vmaxq_f32(q, lo), hi);
                vqmovn_s32(vcvtq_s32_f32(q))
            };
            let flo = vcvtq_f32_s32(vmovl_s16(vget_low_s16(q16)));
            let fhi = vcvtq_f32_s32(vmovl_s16(vget_high_s16(q16)));
            let i16lo = chunk(flo);
            let i16hi = chunk(fhi);
            let i8x8 = vqmovn_s16(vcombine_s16(i16lo, i16hi));
            vst1_s8(out.as_mut_ptr().add(i), i8x8);
            i += 8;
        }
        while i < n {
            let f = ((*input.get_unchecked(i) as f32 - zp_in) * scale_in).clamp(clamp.0, clamp.1);
            *out.get_unchecked_mut(i) = (f / scale_out + zp_out)
                .round_ties_even()
                .clamp(-128.0, 127.0) as i8;
            i += 1;
        }
    }
}

// ===========================================================================
// Fused DequantizeLinear -> HardSwish(HardSigmoid+Mul) -> QuantizeLinear.
// MobileNetV3 emits HardSwish as `x * HardSigmoid(x)` = DQ -> HardSigmoid ->
// Mul(dq, hs) -> Q. Fold it into one i8->i8 pass. Bit-identical to the
// sequence: DQ `(v-zp_in)*s_in`, HardSigmoid `(alpha*f+beta).clamp(0,1)`,
// Mul `f*hs`, Q `(m/s_out+zp_out).round_ties_even().clamp`.
// ===========================================================================

/// Dispatch the fused i8->i8 `DQ -> HardSwish -> Q` boundary. `alpha`/`beta`
/// are the HardSigmoid attributes (MobileNetV3 uses 1/6, 1/2).
#[allow(clippy::too_many_arguments)]
pub fn requant_i8_dq_hardswish_q_dispatch(
    input: &[i8],
    scale_in: f32,
    zp_in: f32,
    scale_out: f32,
    zp_out: f32,
    alpha: f32,
    beta: f32,
    out: &mut [i8],
) {
    debug_assert_eq!(input.len(), out.len());
    #[cfg(target_arch = "aarch64")]
    {
        if crate::host_cpu().features.neon {
            // SAFETY: NEON feature-detected at runtime.
            #[allow(unsafe_code)]
            unsafe {
                requant_i8_dq_hardswish_q_neon(
                    input, scale_in, zp_in, scale_out, zp_out, alpha, beta, out,
                );
            }
            return;
        }
    }
    requant_i8_dq_hardswish_q_scalar(input, scale_in, zp_in, scale_out, zp_out, alpha, beta, out);
}

/// Scalar reference / bitwise oracle.
#[inline]
#[allow(clippy::too_many_arguments)]
pub(crate) fn requant_i8_dq_hardswish_q_scalar(
    input: &[i8],
    scale_in: f32,
    zp_in: f32,
    scale_out: f32,
    zp_out: f32,
    alpha: f32,
    beta: f32,
    out: &mut [i8],
) {
    for (o, &q) in out.iter_mut().zip(input) {
        let f = (q as f32 - zp_in) * scale_in;
        let hs = (alpha * f + beta).clamp(0.0, 1.0);
        let m = f * hs;
        *o = (m / scale_out + zp_out)
            .round_ties_even()
            .clamp(-128.0, 127.0) as i8;
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(unsafe_code, clippy::too_many_arguments)]
unsafe fn requant_i8_dq_hardswish_q_neon(
    input: &[i8],
    scale_in: f32,
    zp_in: f32,
    scale_out: f32,
    zp_out: f32,
    alpha: f32,
    beta: f32,
    out: &mut [i8],
) {
    use std::arch::aarch64::*;
    unsafe {
        let v_zp_in = vdupq_n_f32(zp_in);
        let v_sin = vdupq_n_f32(scale_in);
        let v_sout = vdupq_n_f32(scale_out);
        let v_zp_out = vdupq_n_f32(zp_out);
        let v_rout = vdupq_n_f32(1.0 / scale_out);
        let v_alpha = vdupq_n_f32(alpha);
        let v_beta = vdupq_n_f32(beta);
        let zero = vdupq_n_f32(0.0);
        let one = vdupq_n_f32(1.0);
        let lo = vdupq_n_f32(-128.0);
        let hi = vdupq_n_f32(127.0);
        let n = input.len();
        let main = n & !7;
        let mut i = 0;
        while i < main {
            let q16 = vmovl_s8(vld1_s8(input.as_ptr().add(i)));
            let chunk = |f32x4: float32x4_t| -> int16x4_t {
                let f = vmulq_f32(vsubq_f32(f32x4, v_zp_in), v_sin);
                let hs = vminq_f32(
                    vmaxq_f32(vaddq_f32(vmulq_f32(v_alpha, f), v_beta), zero),
                    one,
                );
                let m = vmulq_f32(f, hs);
                let q = vaddq_f32(super::div_invariant_neon(m, v_sout, v_rout), v_zp_out);
                let q = vminq_f32(vmaxq_f32(vrndnq_f32(q), lo), hi);
                vqmovn_s32(vcvtq_s32_f32(q))
            };
            let flo = vcvtq_f32_s32(vmovl_s16(vget_low_s16(q16)));
            let fhi = vcvtq_f32_s32(vmovl_s16(vget_high_s16(q16)));
            let i8x8 = vqmovn_s16(vcombine_s16(chunk(flo), chunk(fhi)));
            vst1_s8(out.as_mut_ptr().add(i), i8x8);
            i += 8;
        }
        while i < n {
            let f = (*input.get_unchecked(i) as f32 - zp_in) * scale_in;
            let hs = (alpha * f + beta).clamp(0.0, 1.0);
            *out.get_unchecked_mut(i) = ((f * hs) / scale_out + zp_out)
                .round_ties_even()
                .clamp(-128.0, 127.0) as i8;
            i += 1;
        }
    }
}

// ===========================================================================
// Fused f32 HardSwish -> QuantizeLinear. MobileNetV3 has `x*HardSigmoid(x)`
// whose input is a plain f32 tensor (e.g. an SE Mul output), not a
// DequantizeLinear — so the DQ->HardSwish->Q fold above doesn't apply. This
// folds `HardSigmoid -> Mul(x, hs) -> Q` (3 f32 passes + 2 intermediate Vec
// allocations) into one f32->i8 pass. Bit-identical to the op sequence:
//   hs  = (alpha*x + beta).clamp(0, 1)   (HardSigmoid)
//   out = (x*hs / y_scale + y_zp).round_ties_even().clamp(-128, 127)  (Mul, Q)
// The vectorised path keeps mul+add separate (not FMA) and uses a real divide
// (not reciprocal-multiply) so it matches the scalar rounding bit-for-bit.
// ===========================================================================

/// Dispatch the fused f32 HardSwish + QuantizeLinear to the widest host path.
#[allow(clippy::too_many_arguments)]
pub fn hardswish_quantize_f32_to_i8_dispatch(
    x: &[f32],
    alpha: f32,
    beta: f32,
    y_scale: f32,
    y_zp: f32,
    out: &mut [i8],
) {
    debug_assert_eq!(x.len(), out.len());

    #[cfg(target_arch = "aarch64")]
    {
        if crate::host_cpu().features.neon {
            // SAFETY: NEON feature-detected at runtime; inner fn is
            // annotated `target_feature(enable = "neon")`.
            #[allow(unsafe_code)]
            unsafe {
                hardswish_quantize_f32_to_i8_neon(x, alpha, beta, y_scale, y_zp, out);
            }
            return;
        }
    }

    hardswish_quantize_f32_to_i8_scalar(x, alpha, beta, y_scale, y_zp, out);
}

/// Scalar reference / bitwise oracle.
pub(crate) fn hardswish_quantize_f32_to_i8_scalar(
    x: &[f32],
    alpha: f32,
    beta: f32,
    y_scale: f32,
    y_zp: f32,
    out: &mut [i8],
) {
    for (o, &v) in out.iter_mut().zip(x) {
        let hs = (alpha * v + beta).clamp(0.0, 1.0);
        *o = ((v * hs) / y_scale + y_zp)
            .round_ties_even()
            .clamp(-128.0, 127.0) as i8;
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(unsafe_code)]
unsafe fn hardswish_quantize_f32_to_i8_neon(
    x: &[f32],
    alpha: f32,
    beta: f32,
    y_scale: f32,
    y_zp: f32,
    out: &mut [i8],
) {
    use std::arch::aarch64::*;

    unsafe {
        let va = vdupq_n_f32(alpha);
        let vb = vdupq_n_f32(beta);
        let zero = vdupq_n_f32(0.0);
        let one = vdupq_n_f32(1.0);
        let vscale = vdupq_n_f32(y_scale);
        let vrecip = vdupq_n_f32(1.0 / y_scale);
        let vzp = vdupq_n_f32(y_zp);
        let n = x.len();
        let xp = x.as_ptr();
        let op = out.as_mut_ptr();
        let mut i = 0;
        // 16/iter: 4 lanes -> double saturating-narrow -> one 16-byte store.
        let lane = |i: usize| -> int32x4_t {
            let v = vld1q_f32(xp.add(i));
            // hs = clamp(alpha*v + beta, 0, 1); mul+add kept separate (not FMA).
            let hs = vminq_f32(vmaxq_f32(vaddq_f32(vmulq_f32(v, va), vb), zero), one);
            // (v*hs)/y_scale + y_zp, correctly-rounded divide, then FCVTNS.
            let q = vaddq_f32(
                super::div_invariant_neon(vmulq_f32(v, hs), vscale, vrecip),
                vzp,
            );
            vcvtnq_s32_f32(q)
        };
        while i + 16 <= n {
            let r0 = lane(i);
            let r1 = lane(i + 4);
            let r2 = lane(i + 8);
            let r3 = lane(i + 12);
            let s01 = vcombine_s16(vqmovn_s32(r0), vqmovn_s32(r1));
            let s23 = vcombine_s16(vqmovn_s32(r2), vqmovn_s32(r3));
            vst1q_s8(op.add(i), vcombine_s8(vqmovn_s16(s01), vqmovn_s16(s23)));
            i += 16;
        }
        while i + 8 <= n {
            let r0 = lane(i);
            let r1 = lane(i + 4);
            let b = vqmovn_s16(vcombine_s16(vqmovn_s32(r0), vqmovn_s32(r1)));
            vst1_s8(op.add(i), b);
            i += 8;
        }
        while i < n {
            let v = *xp.add(i);
            let hs = (alpha * v + beta).clamp(0.0, 1.0);
            *op.add(i) = ((v * hs) / y_scale + y_zp)
                .round_ties_even()
                .clamp(-128.0, 127.0) as i8;
            i += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run_oracle(
        acc: &[i32],
        bias: Option<&[i32]>,
        composite: f32,
        y_zp: f32,
        relu: bool,
        c: usize,
    ) -> Vec<i8> {
        let mut out_disp = vec![0_i8; acc.len()];
        let mut out_scalar = vec![0_i8; acc.len()];
        requant_i32_row_to_i8_dispatch(acc, bias, composite, y_zp, relu, &mut out_disp, c);
        requant_scalar(acc, bias, composite, y_zp, relu, &mut out_scalar, c);
        if out_disp != out_scalar {
            for (idx, (d, s)) in out_disp.iter().zip(out_scalar.iter()).enumerate() {
                if d != s {
                    let pixel = idx / c;
                    let ch = idx % c;
                    let a_raw = acc[idx];
                    let b_raw = bias.map(|b| b[ch]).unwrap_or(0);
                    let a = a_raw.wrapping_add(b_raw);
                    let v_f = (a as f32) * composite + y_zp;
                    panic!(
                        "first mismatch at idx={idx} (pixel={pixel}, ch={ch}): \
                         acc={a_raw} bias={b_raw} a={a} v={v_f:?} \
                         simd={d} scalar={s} composite={composite} y_zp={y_zp} relu={relu}"
                    );
                }
            }
        }
        out_disp
    }

    #[test]
    fn hardswish_quantize_f32_matches_scalar() {
        // dispatch (SIMD/scalar) must match the scalar oracle across a lane16 +
        // lane8 + scalar-tail length, with inputs that exercise both HardSigmoid
        // clamps (x very negative -> hs=0, x large -> hs=1) and i8 saturation.
        let alpha = 1.0 / 6.0;
        let beta = 0.5;
        for &(y_scale, y_zp) in &[(0.023_f32, 0.0_f32), (0.05, -12.0), (0.008, 7.0)] {
            for &len in &[27usize, 32, 40] {
                let x: Vec<f32> = (0..len)
                    .map(|i| ((i as f32) - len as f32 / 2.0) * 0.9)
                    .collect();
                let mut a = vec![0i8; len];
                let mut b = vec![0i8; len];
                hardswish_quantize_f32_to_i8_dispatch(&x, alpha, beta, y_scale, y_zp, &mut a);
                hardswish_quantize_f32_to_i8_scalar(&x, alpha, beta, y_scale, y_zp, &mut b);
                assert_eq!(a, b, "hardswish+quant mismatch len={len} scale={y_scale}");
            }
        }
    }

    #[test]
    fn per_channel_matches_scalar() {
        // Per-channel composite + corr epilogue: dispatch (SIMD/scalar) must
        // match the scalar oracle across channel counts that exercise the
        // 4-lane main path and the scalar tail (16 exact, 24=16+8, 40, and
        // the awkward 5/17 tails), with per-channel scales/corrections and a
        // y_zp, including inputs that force saturation at both ends.
        for &c in &[4usize, 5, 16, 17, 24, 40, 72] {
            let pixels = 37;
            let n = pixels * c;
            let mut acc = vec![0i32; n];
            let mut s: u32 = 0x9e3779b9 ^ (c as u32);
            for v in acc.iter_mut() {
                s ^= s << 13;
                s ^= s >> 17;
                s ^= s << 5;
                // Wide range so requant lands both inside and outside i8.
                *v = (s as i32) % 4000 - 2000;
            }
            let corr: Vec<i32> = (0..c).map(|ch| (ch as i32 * 13) % 97 - 48).collect();
            let composite: Vec<f32> = (0..c).map(|ch| 0.003 + (ch as f32) * 0.0011).collect();
            let y_zp = -6.0_f32;
            let mut out_disp = vec![0i8; n];
            let mut out_scalar = vec![0i8; n];
            requant_i8_per_channel_dispatch(&acc, &corr, &composite, y_zp, &mut out_disp, c);
            requant_i8_per_channel_scalar(&acc, &corr, &composite, y_zp, &mut out_scalar, c);
            assert_eq!(
                out_disp, out_scalar,
                "per-channel requant mismatch at c={c}"
            );
        }
    }

    #[test]
    fn dq_relu_q_matches_scalar_and_three_op_sequence() {
        // dispatch (SIMD/scalar) must match the scalar reference AND the
        // literal DQ->[act]->Q composition, across no activation, Relu and the
        // two-sided Clip the tracker head uses, various scale/zp, and a length
        // with an 8-tail.
        let acts = [
            (f32::NEG_INFINITY, f32::INFINITY),
            (0.0, f32::INFINITY),
            (0.0, 6.0),
            (-1.5, 1.5),
            (f32::NEG_INFINITY, 2.0),
        ];
        // The kernel's whole input domain is the 256 i8 values, so sweeping
        // scales over all of them is very nearly exhaustive — which is what the
        // reciprocal-multiply divide needs to be held to.
        let mut st = 12345u32;
        let mut rnd = || {
            st = st.wrapping_mul(1664525).wrapping_add(1013904223);
            (st >> 8) as f32 / 16_777_216.0
        };
        let mut combos = vec![
            (0.021_f32, 0.0_f32, 0.037_f32, 0.0_f32),
            (0.05, -7.0, 0.011, 12.0),
            (0.003, 5.0, 0.05, -20.0),
        ];
        for _ in 0..64 {
            combos.push((
                0.0001 + rnd() * 0.2,
                (rnd() * 60.0 - 30.0).round(),
                0.0001 + rnd() * 0.2,
                (rnd() * 60.0 - 30.0).round(),
            ));
        }
        for &clamp in &acts {
            for &(sin, zin, sout, zout) in &combos {
                let input: Vec<i8> = (-109..=108).map(|v| v as i8).collect(); // len 218 (tail)
                let mut got = vec![0i8; input.len()];
                let mut sc = vec![0i8; input.len()];
                requant_i8_dq_relu_q_dispatch(&input, sin, zin, sout, zout, clamp, &mut got);
                requant_i8_dq_relu_q_scalar(&input, sin, zin, sout, zout, clamp, &mut sc);
                assert_eq!(got, sc, "dispatch vs scalar clamp={clamp:?}");
                // Literal three-op reference.
                let three: Vec<i8> = input
                    .iter()
                    .map(|&q| {
                        let f = ((q as f32 - zin) * sin).clamp(clamp.0, clamp.1);
                        (f / sout + zout).round_ties_even().clamp(-128.0, 127.0) as i8
                    })
                    .collect();
                assert_eq!(got, three, "dispatch vs 3-op clamp={clamp:?}");
            }
        }
    }

    #[test]
    fn dq_hardswish_q_matches_scalar_and_four_op_sequence() {
        for &(sin, zin, sout, zout, alpha, beta) in &[
            (0.03_f32, 0.0_f32, 0.02_f32, 0.0_f32, 1.0 / 6.0, 0.5),
            (0.05, -7.0, 0.011, 12.0, 0.2, 0.5),
            (0.008, 4.0, 0.05, -20.0, 1.0 / 6.0, 0.5),
        ] {
            let input: Vec<i8> = (-109..=108).map(|v| v as i8).collect();
            let mut got = vec![0i8; input.len()];
            let mut sc = vec![0i8; input.len()];
            requant_i8_dq_hardswish_q_dispatch(&input, sin, zin, sout, zout, alpha, beta, &mut got);
            requant_i8_dq_hardswish_q_scalar(&input, sin, zin, sout, zout, alpha, beta, &mut sc);
            assert_eq!(got, sc, "hardswish dispatch vs scalar");
            let four: Vec<i8> = input
                .iter()
                .map(|&q| {
                    let f = (q as f32 - zin) * sin;
                    let hs = (alpha * f + beta).clamp(0.0, 1.0);
                    ((f * hs) / sout + zout)
                        .round_ties_even()
                        .clamp(-128.0, 127.0) as i8
                })
                .collect();
            assert_eq!(got, four, "hardswish dispatch vs 4-op");
        }
    }

    #[test]
    fn matches_scalar_no_bias_no_relu_lane_aligned() {
        let c = 32;
        let pixels = 4;
        let acc: Vec<i32> = (0..(pixels * c) as i32).map(|x| x - 64).collect();
        let got = run_oracle(&acc, None, 0.07, 3.5, false, c);
        assert_eq!(got.len(), pixels * c);
    }

    #[test]
    fn matches_scalar_with_bias_no_relu() {
        let c = 16;
        let pixels = 5;
        let acc: Vec<i32> = (0..(pixels * c) as i32).map(|x| x * 7 - 100).collect();
        let bias: Vec<i32> = (0..c as i32).map(|x| x - 8).collect();
        run_oracle(&acc, Some(&bias), 0.013, -2.0, false, c);
    }

    #[test]
    fn matches_scalar_with_bias_with_relu() {
        let c = 24;
        let pixels = 6;
        let acc: Vec<i32> = (0..(pixels * c) as i32).map(|x| x - 80).collect();
        let bias: Vec<i32> = (0..c as i32).map(|x| -x).collect();
        run_oracle(&acc, Some(&bias), 0.21, 0.0, true, c);
    }

    #[test]
    fn matches_scalar_relu_clips_negatives_to_zero() {
        // All negative inputs → relu must zero the entire row.
        let c = 8;
        let pixels = 3;
        let acc: Vec<i32> = vec![-100; pixels * c];
        let got = run_oracle(&acc, None, 1.0, 0.0, true, c);
        assert!(got.iter().all(|&q| q == 0));
    }

    #[test]
    fn matches_scalar_clamps_saturation_at_both_ends() {
        let c = 16;
        // Half above +127, half below -128 after composite scale.
        let mut acc = vec![0_i32; c];
        for ch in 0..c / 2 {
            acc[ch] = 1_000_000;
        }
        for ch in c / 2..c {
            acc[ch] = -1_000_000;
        }
        let got = run_oracle(&acc, None, 0.001, 0.0, false, c);
        for ch in 0..c / 2 {
            assert_eq!(got[ch], 127);
        }
        for ch in c / 2..c {
            assert_eq!(got[ch], -128);
        }
    }

    #[test]
    fn matches_scalar_round_half_to_even() {
        // Composite + y_zp tuned so each acc value lands on a half-integer.
        // composite = 0.5, y_zp = 0  →  v = a * 0.5, so every value is an
        // exact tie and rounds to the nearest even integer (ONNX / ORT):
        // 0.5→0, -0.5→0, 1.5→2, -1.5→-2, 2.5→2, -2.5→-2, 3.5→4, -3.5→-4.
        let c = 8;
        let acc: Vec<i32> = vec![1, -1, 3, -3, 5, -5, 7, -7];
        let got = run_oracle(&acc, None, 0.5, 0.0, false, c);
        assert_eq!(got, vec![0_i8, 0, 2, -2, 2, -2, 4, -4]);
    }

    #[test]
    fn matches_scalar_channel_tail_below_lane_width() {
        // c=1 (closing pointwise like /xif*/pwl with c_out=1). Forces
        // every iteration through the scalar tail.
        let c = 1;
        let pixels: i32 = 17;
        let acc: Vec<i32> = (0..pixels).map(|x| x * 11 - 30).collect();
        run_oracle(&acc, None, 0.04, 1.5, false, c as usize);
    }

    #[test]
    fn matches_scalar_channel_count_5_mixes_lane_and_tail() {
        // c=5 (e.g. /connect_model/cls_pred kind heads, c_out small).
        // Tests that the per-pixel loop correctly resets the SIMD tail.
        let c = 5;
        let pixels = 32;
        let acc: Vec<i32> = (0..(pixels * c) as i32).map(|x| (x * 13) - 200).collect();
        let bias: Vec<i32> = (0..c as i32).map(|x| -2 * x + 1).collect();
        run_oracle(&acc, Some(&bias), 0.017, -0.7, true, c);
    }

    #[test]
    fn matches_scalar_realistic_chain_inputs_stress() {
        // Sweep a realistic configuration: c=16, pixels=144 (matches the
        // kh=5 e2e bitwise case that flipped one element), composite +
        // y_zp pulled from the kh=5 chain (composite = (0.03 * 0.05)/0.09,
        // y_zp = 0.0). We feed the i32 accumulator with values plausible
        // for a tracker pw output (range ~ [-2000, 2000]) and exercise
        // every bias / relu combination.
        let c = 16;
        let pixels = 144;
        let composite = (0.03_f32 * 0.05_f32) / 0.09_f32;
        let y_zp = 0.0_f32;
        let mut state: u64 = 0xDEADBEEF;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 32) as i32
        };
        let acc: Vec<i32> = (0..(pixels * c)).map(|_| next() % 4001 - 2000).collect();
        let bias: Vec<i32> = (0..c).map(|_| next() % 21 - 10).collect();
        for relu in [false, true] {
            for bias_opt in [None, Some(bias.as_slice())] {
                run_oracle(&acc, bias_opt, composite, y_zp, relu, c);
            }
        }
    }

    #[test]
    fn matches_scalar_negative_only_inputs_relu_force_clamp() {
        // All-negative i32 inputs with relu — the slow path's per-op
        // QLinearConv would emit the clamped-but-not-relu i8, then the
        // QuantizedQdq fold would zero them; the fused requant must
        // produce the same zeros byte-for-byte.
        let c = 16;
        let pixels = 64;
        let acc: Vec<i32> = (0..(pixels * c) as i32).map(|x| -(x + 1)).collect();
        run_oracle(&acc, None, 0.05, 0.0, true, c);
        run_oracle(&acc, None, 0.5, 0.0, true, c);
    }

    #[test]
    fn matches_scalar_channel_count_17_one_lane_one_tail() {
        // c=17, lane=8/16 → main loop emits one full vector then the
        // channel tail handles the residual 1 (AVX-512) or 1 (AVX2).
        let c = 17;
        let pixels = 7;
        let acc: Vec<i32> = (0..(pixels * c) as i32)
            .map(|x| (x.wrapping_mul(31)) - 50)
            .collect();
        let bias: Vec<i32> = (0..c as i32).collect();
        run_oracle(&acc, Some(&bias), 0.029, 4.0, false, c);
    }
}
