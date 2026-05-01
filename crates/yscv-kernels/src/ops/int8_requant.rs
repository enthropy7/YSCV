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
//! let mut q = v.round().clamp(-128.0, 127.0) as i8;
//! if relu && q < 0 { q = 0; }
//! ```
//!
//! The SIMD variants in this module match the scalar form bit-for-bit:
//!
//! * `f32::round()` is round-half-away-from-zero (LLVM `llvm.round.f32`).
//!   We replicate it by extracting the sign, taking the absolute value,
//!   adding `0.5 - ULP(0.5)/2` (= `0.49999997`, the closest f32 below
//!   0.5), flooring, then OR-ing the sign back in.
//!
//!   The textbook `floor(abs(v) + 0.5)` trick has a precision pitfall: at
//!   `v = 0.49999997` (one ULP below 0.5), the f32 sum `v + 0.5` rounds
//!   up to exactly 1.0 due to round-to-nearest-even on the sub-ULP gap,
//!   giving `floor = 1` even though `v.round()` is 0. Biasing by
//!   `0.5 - ULP(0.5)` shifts the f32 rounding boundary just enough that
//!   sub-half values stay below 1.0 after the add (so they floor to 0)
//!   while exact ties at `±0.5, ±1.5, ±2.5, …` still round through to
//!   the next integer in f32 (so they floor to ±N+1). This matches
//!   `f32::round()` byte-for-byte across the whole `[-128, 127]` range
//!   we care about (verified by the `matches_scalar_realistic_chain_inputs_stress`
//!   parity test).
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
    let path =
        if std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("avx512bw") {
            PATH_AVX512
        } else if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("sse4.1") {
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
pub(crate) fn requant_i32_row_to_i8_dispatch(
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
        if std::arch::is_aarch64_feature_detected!("neon") {
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

/// `0.5 - ULP/2` at the magnitude of 0.5. Equal to `f32::from_bits(0x3EFFFFFF)`.
/// Used as the bias for the `floor(abs + bias)` round-half-away-from-zero
/// emulation; see the module-level docstring for the full derivation.
const ROUND_BIAS_HALF: f32 = 0.499_999_97_f32;

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
            let mut q = v.round().clamp(-128.0, 127.0) as i8;
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
        let half = _mm512_set1_ps(ROUND_BIAS_HALF);
        let lo = _mm512_set1_ps(-128.0);
        let hi = _mm512_set1_ps(127.0);
        let sign_mask = _mm512_castsi512_ps(_mm512_set1_epi32(i32::MIN));
        let abs_mask = _mm512_castsi512_ps(_mm512_set1_epi32(i32::MAX));

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
                let sign = _mm512_and_ps(v, sign_mask);
                let abs = _mm512_and_ps(v, abs_mask);
                let rounded_abs = _mm512_roundscale_ps::<
                    { _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC },
                >(_mm512_add_ps(abs, half));
                let rounded = _mm512_or_ps(rounded_abs, sign);
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
                let mut q = v.round().clamp(-128.0, 127.0) as i8;
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
        let half = _mm256_set1_ps(ROUND_BIAS_HALF);
        let lo = _mm256_set1_ps(-128.0);
        let hi = _mm256_set1_ps(127.0);
        let sign_mask = _mm256_castsi256_ps(_mm256_set1_epi32(i32::MIN));
        let abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(i32::MAX));

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
                let sign = _mm256_and_ps(v, sign_mask);
                let abs = _mm256_and_ps(v, abs_mask);
                let rounded_abs = _mm256_floor_ps(_mm256_add_ps(abs, half));
                let rounded = _mm256_or_ps(rounded_abs, sign);
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
                let mut q = v.round().clamp(-128.0, 127.0) as i8;
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
        let half = vdupq_n_f32(ROUND_BIAS_HALF);
        let lo = vdupq_n_f32(-128.0);
        let hi = vdupq_n_f32(127.0);
        let sign_mask = vreinterpretq_f32_u32(vdupq_n_u32(0x8000_0000));
        let abs_mask = vreinterpretq_f32_u32(vdupq_n_u32(0x7FFF_FFFF));

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
                let sign = vreinterpretq_f32_u32(vandq_u32(
                    vreinterpretq_u32_f32(v),
                    vreinterpretq_u32_f32(sign_mask),
                ));
                let abs = vreinterpretq_f32_u32(vandq_u32(
                    vreinterpretq_u32_f32(v),
                    vreinterpretq_u32_f32(abs_mask),
                ));
                // floor(abs + 0.5) — use vrndmq_f32 (round toward -inf,
                // i.e. floor). NEON v8.2-A gives us this directly; the
                // base v8 NEON `target_feature = "neon"` already covers
                // it on aarch64 since the rounding instructions are part
                // of the mandatory ARMv8 FP set.
                let rounded_abs = vrndmq_f32(vaddq_f32(abs, half));
                let rounded = vreinterpretq_f32_u32(vorrq_u32(
                    vreinterpretq_u32_f32(rounded_abs),
                    vreinterpretq_u32_f32(sign),
                ));
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
                let mut q = v.round().clamp(-128.0, 127.0) as i8;
                if relu && q < 0 {
                    q = 0;
                }
                row_out[ch] = q;
                ch += 1;
            }
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
    fn matches_scalar_round_half_away_from_zero() {
        // Composite + y_zp tuned so each acc value lands on a half-integer.
        // composite = 0.5, y_zp = 0  →  v = a * 0.5
        // acc = 1 → v = 0.5  → round = 1 (away from zero)
        // acc = -1 → v = -0.5 → round = -1 (away from zero)
        // acc = 3 → v = 1.5  → round = 2
        // acc = -3 → v = -1.5 → round = -2
        let c = 8;
        let acc: Vec<i32> = vec![1, -1, 3, -3, 5, -5, 7, -7];
        let got = run_oracle(&acc, None, 0.5, 0.0, false, c);
        assert_eq!(got, vec![1_i8, -1, 2, -2, 3, -3, 4, -4]);
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
