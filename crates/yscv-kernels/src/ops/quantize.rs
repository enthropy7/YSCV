// ===========================================================================
// INT4 quantize / dequantize kernels with SIMD acceleration
// ===========================================================================
//
// Packed INT4 layout: 2 signed nibbles per byte, low nibble first.
// Each nibble is a signed 4-bit value in range [-8, 7].

/// Dequantize packed INT4 data to f32.
///
/// Each byte contains 2 signed 4-bit values (low nibble first).
/// `output[i] = (nibble[i] - zero_point) * scale`
///
/// # Panics
///
/// Panics in debug mode if `output.len() < packed.len() * 2`.
#[allow(unsafe_code)]
#[inline]
pub fn dequantize_int4_to_f32(packed: &[u8], scale: f32, zero_point: i8, output: &mut [f32]) {
    debug_assert!(
        output.len() >= packed.len() * 2,
        "output buffer too small: need {}, got {}",
        packed.len() * 2,
        output.len()
    );

    if cfg!(miri) {
        dequantize_int4_to_f32_scalar(packed, scale, zero_point, output);
        return;
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            // SAFETY: guarded by runtime feature detection. All pointer
            // accesses stay within the bounds enforced by chunk iteration.
            unsafe {
                dequantize_int4_to_f32_neon(packed, scale, zero_point, output);
            }
            return;
        }
    }

    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("sse2") {
            // SAFETY: guarded by runtime feature detection. Pointer
            // accesses bounded by chunk iteration.
            unsafe {
                dequantize_int4_to_f32_sse2(packed, scale, zero_point, output);
            }
            return;
        }
    }

    dequantize_int4_to_f32_scalar(packed, scale, zero_point, output);
}

/// Quantize f32 values to packed INT4.
///
/// `quantized[i] = clamp(round(data[i] / scale) + zero_point, -8, 7)`
///
/// Output is packed nibbles (low first). If `data.len()` is odd the high
/// nibble of the last byte is zero-padded.
///
/// # Panics
///
/// Panics in debug mode if `output.len() < (data.len() + 1) / 2`.
#[inline]
pub fn quantize_f32_to_int4(data: &[f32], scale: f32, zero_point: i8, output: &mut [u8]) {
    let needed = data.len().div_ceil(2);
    debug_assert!(
        output.len() >= needed,
        "output buffer too small: need {needed}, got {}",
        output.len()
    );

    let inv_scale = if scale.abs() > f32::EPSILON {
        1.0 / scale
    } else {
        0.0
    };
    let zp = zero_point as i32;

    data.chunks(2)
        .zip(output.iter_mut())
        .for_each(|(pair, out)| {
            let q0 = ((pair[0] * inv_scale).round() as i32 + zp).clamp(-8, 7);
            let q1 = if pair.len() > 1 {
                ((pair[1] * inv_scale).round() as i32 + zp).clamp(-8, 7)
            } else {
                0
            };
            *out = (q0 as u8 & 0x0F) | (((q1 as u8) & 0x0F) << 4);
        });
}

// ---------------------------------------------------------------------------
// Scalar fallback
// ---------------------------------------------------------------------------

/// Extract low nibble from byte and sign-extend to i8.
#[inline(always)]
fn nibble_lo(byte: u8) -> i8 {
    let v = (byte & 0x0F) as i8;
    if v >= 8 { v - 16 } else { v }
}

/// Extract high nibble from byte and sign-extend to i8.
#[inline(always)]
fn nibble_hi(byte: u8) -> i8 {
    let v = ((byte >> 4) & 0x0F) as i8;
    if v >= 8 { v - 16 } else { v }
}

#[inline]
fn dequantize_int4_to_f32_scalar(packed: &[u8], scale: f32, zero_point: i8, output: &mut [f32]) {
    let zp = zero_point as f32;
    packed.iter().enumerate().for_each(|(i, &byte)| {
        output[i * 2] = (nibble_lo(byte) as f32 - zp) * scale;
        output[i * 2 + 1] = (nibble_hi(byte) as f32 - zp) * scale;
    });
}

// ---------------------------------------------------------------------------
// NEON (aarch64)
// ---------------------------------------------------------------------------

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(unsafe_code)]
/// SAFETY: caller must ensure NEON is available and output has space for
/// `packed.len() * 2` elements.
unsafe fn dequantize_int4_to_f32_neon(
    packed: &[u8],
    scale: f32,
    zero_point: i8,
    output: &mut [f32],
) {
    use std::arch::aarch64::{
        float32x2x2_t, float32x4_t, int8x8_t, int16x8_t, vand_s8, vandq_s16, vcombine_f32,
        vcvtq_f32_s32, vdup_n_s8, vdupq_n_f32, vdupq_n_s16, vget_high_f32, vget_high_s16,
        vget_low_f32, vget_low_s16, vld1_s8, vmovl_s8, vmovl_s16, vmulq_f32, vshlq_n_s16,
        vshrq_n_s16, vst1q_f32, vsubq_s16, vzip_f32,
    };

    // SAFETY: all intrinsics below require NEON, which is guaranteed
    // by the caller's `is_aarch64_feature_detected!("neon")` check.
    // Pointer arithmetic stays within bounds: chunk loop processes
    // 8 packed bytes → 16 f32 outputs at a time; tail handles remainder.
    unsafe {
        let zp_vec: int16x8_t = vdupq_n_s16(zero_point as i16);
        let scale_vec: float32x4_t = vdupq_n_f32(scale);
        let mask_0f: int8x8_t = vdup_n_s8(0x0F);

        let chunks = packed.len() / 8;
        let packed_ptr = packed.as_ptr();
        let out_ptr = output.as_mut_ptr();

        for c in 0..chunks {
            let base = c * 8;
            let raw: int8x8_t = vld1_s8(packed_ptr.add(base) as *const i8);
            let lo: int8x8_t = vand_s8(raw, mask_0f);
            let raw_wide: int16x8_t = vmovl_s8(raw);
            let hi_wide: int16x8_t = vandq_s16(vshrq_n_s16(raw_wide, 4), vdupq_n_s16(0x0F));

            let lo_wide: int16x8_t = vmovl_s8(lo);
            let lo_ext: int16x8_t = vshrq_n_s16(vshlq_n_s16(lo_wide, 12), 12);
            let hi_ext: int16x8_t = vshrq_n_s16(vshlq_n_s16(hi_wide, 12), 12);

            let lo_sub: int16x8_t = vsubq_s16(lo_ext, zp_vec);
            let hi_sub: int16x8_t = vsubq_s16(hi_ext, zp_vec);

            let lo_lo_f: float32x4_t =
                vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo_sub))), scale_vec);
            let lo_hi_f: float32x4_t =
                vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo_sub))), scale_vec);
            let hi_lo_f: float32x4_t =
                vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi_sub))), scale_vec);
            let hi_hi_f: float32x4_t =
                vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi_sub))), scale_vec);

            let dst = out_ptr.add(base * 2);
            let z1: float32x2x2_t = vzip_f32(vget_low_f32(lo_lo_f), vget_low_f32(hi_lo_f));
            let z2: float32x2x2_t = vzip_f32(vget_high_f32(lo_lo_f), vget_high_f32(hi_lo_f));
            let z3: float32x2x2_t = vzip_f32(vget_low_f32(lo_hi_f), vget_low_f32(hi_hi_f));
            let z4: float32x2x2_t = vzip_f32(vget_high_f32(lo_hi_f), vget_high_f32(hi_hi_f));

            vst1q_f32(dst, vcombine_f32(z1.0, z1.1));
            vst1q_f32(dst.add(4), vcombine_f32(z2.0, z2.1));
            vst1q_f32(dst.add(8), vcombine_f32(z3.0, z3.1));
            vst1q_f32(dst.add(12), vcombine_f32(z4.0, z4.1));
        }

        // Scalar tail for remaining bytes
        let start = chunks * 8;
        let zp = zero_point as f32;
        for i in start..packed.len() {
            let byte = packed[i];
            output[i * 2] = (nibble_lo(byte) as f32 - zp) * scale;
            output[i * 2 + 1] = (nibble_hi(byte) as f32 - zp) * scale;
        }
    }
}

// ---------------------------------------------------------------------------
// SSE2 (x86_64)
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
#[allow(unsafe_code)]
/// SAFETY: caller must ensure SSE2 is available and output has space for
/// `packed.len() * 2` elements.
unsafe fn dequantize_int4_to_f32_sse2(
    packed: &[u8],
    scale: f32,
    zero_point: i8,
    output: &mut [f32],
) {
    use std::arch::x86_64::{
        __m128i, _mm_and_si128, _mm_cvtepi32_ps, _mm_loadu_si128, _mm_mul_ps, _mm_set1_epi16,
        _mm_set1_ps, _mm_setzero_si128, _mm_slli_epi16, _mm_srai_epi16, _mm_srli_epi16,
        _mm_storeu_ps, _mm_sub_epi16, _mm_unpackhi_epi16, _mm_unpacklo_epi8, _mm_unpacklo_epi16,
    };

    // SAFETY: all intrinsics below require SSE2, which is guaranteed
    // by the caller's `is_x86_feature_detected!("sse2")` check.
    unsafe {
        let scale_vec = _mm_set1_ps(scale);
        let zp_vec = _mm_set1_epi16(zero_point as i16);
        let mask_0f = _mm_set1_epi16(0x000F);

        let chunks = packed.len() / 8;
        let packed_ptr = packed.as_ptr();
        let out_ptr = output.as_mut_ptr();

        for c in 0..chunks {
            let base = c * 8;

            let raw: __m128i = {
                let mut tmp = [0u8; 16];
                std::ptr::copy_nonoverlapping(packed_ptr.add(base), tmp.as_mut_ptr(), 8);
                _mm_loadu_si128(tmp.as_ptr() as *const __m128i)
            };

            let zero = _mm_setzero_si128();
            let words: __m128i = _mm_unpacklo_epi8(raw, zero);

            let lo = _mm_and_si128(words, mask_0f);
            let hi = _mm_and_si128(_mm_srli_epi16(words, 4), mask_0f);

            let lo_ext = _mm_srai_epi16(_mm_slli_epi16(lo, 12), 12);
            let hi_ext = _mm_srai_epi16(_mm_slli_epi16(hi, 12), 12);

            let lo_sub = _mm_sub_epi16(lo_ext, zp_vec);
            let hi_sub = _mm_sub_epi16(hi_ext, zp_vec);

            let interleaved_lo = _mm_unpacklo_epi16(lo_sub, hi_sub);
            let interleaved_hi = _mm_unpackhi_epi16(lo_sub, hi_sub);

            let dst = out_ptr.add(base * 2);

            let sign_lo = _mm_srai_epi16(interleaved_lo, 15);
            let i32_0 = _mm_unpacklo_epi16(interleaved_lo, sign_lo);
            let i32_1 = _mm_unpackhi_epi16(interleaved_lo, sign_lo);
            _mm_storeu_ps(dst, _mm_mul_ps(_mm_cvtepi32_ps(i32_0), scale_vec));
            _mm_storeu_ps(dst.add(4), _mm_mul_ps(_mm_cvtepi32_ps(i32_1), scale_vec));

            let sign_hi = _mm_srai_epi16(interleaved_hi, 15);
            let i32_2 = _mm_unpacklo_epi16(interleaved_hi, sign_hi);
            let i32_3 = _mm_unpackhi_epi16(interleaved_hi, sign_hi);
            _mm_storeu_ps(dst.add(8), _mm_mul_ps(_mm_cvtepi32_ps(i32_2), scale_vec));
            _mm_storeu_ps(dst.add(12), _mm_mul_ps(_mm_cvtepi32_ps(i32_3), scale_vec));
        }

        // Scalar tail
        let start = chunks * 8;
        let zp = zero_point as f32;
        for i in start..packed.len() {
            let byte = packed[i];
            output[i * 2] = (nibble_lo(byte) as f32 - zp) * scale;
            output[i * 2 + 1] = (nibble_hi(byte) as f32 - zp) * scale;
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference scalar implementation for comparison.
    fn dequantize_ref(packed: &[u8], scale: f32, zero_point: i8) -> Vec<f32> {
        let mut out = vec![0.0f32; packed.len() * 2];
        dequantize_int4_to_f32_scalar(packed, scale, zero_point, &mut out);
        out
    }

    #[test]
    fn dequantize_basic() {
        // Pack [3, -2] into one byte: lo=3 (0x03), hi=-2 (0x0E) -> 0xE3
        let packed = [0xE3u8];
        let mut out = [0.0f32; 2];
        dequantize_int4_to_f32(&packed, 0.5, 0, &mut out);
        assert!((out[0] - 1.5).abs() < 1e-6, "got {}", out[0]); // 3 * 0.5
        assert!((out[1] - (-1.0)).abs() < 1e-6, "got {}", out[1]); // -2 * 0.5
    }

    #[test]
    fn dequantize_with_zero_point() {
        // Value 5, zero_point 2: (5 - 2) * 0.1 = 0.3
        let packed = [0x05u8]; // lo=5, hi=0
        let mut out = [0.0f32; 2];
        dequantize_int4_to_f32(&packed, 0.1, 2, &mut out);
        assert!((out[0] - 0.3).abs() < 1e-6, "got {}", out[0]);
        assert!((out[1] - (-0.2)).abs() < 1e-6, "got {}", out[1]); // (0-2)*0.1
    }

    #[test]
    fn dequantize_simd_matches_scalar() {
        // Large enough buffer to exercise SIMD paths (>= 8 bytes for one SIMD chunk).
        let n = 64;
        let packed: Vec<u8> = (0..n).map(|i| (i * 17 + 3) as u8).collect();
        let scale = 0.25;
        let zp = 1i8;

        let reference = dequantize_ref(&packed, scale, zp);
        let mut simd_out = vec![0.0f32; packed.len() * 2];
        dequantize_int4_to_f32(&packed, scale, zp, &mut simd_out);

        for (i, (&r, &s)) in reference.iter().zip(simd_out.iter()).enumerate() {
            assert!(
                (r - s).abs() < 1e-6,
                "mismatch at index {i}: scalar={r}, simd={s}"
            );
        }
    }

    #[test]
    fn dequantize_non_aligned_length() {
        // Length not a multiple of 8 to test scalar tail.
        let packed: Vec<u8> = (0..11).map(|i| (i * 37 + 5) as u8).collect();
        let scale = 0.1;
        let zp = -3i8;

        let reference = dequantize_ref(&packed, scale, zp);
        let mut simd_out = vec![0.0f32; packed.len() * 2];
        dequantize_int4_to_f32(&packed, scale, zp, &mut simd_out);

        for (i, (&r, &s)) in reference.iter().zip(simd_out.iter()).enumerate() {
            assert!(
                (r - s).abs() < 1e-6,
                "mismatch at index {i}: scalar={r}, simd={s}"
            );
        }
    }

    #[test]
    fn quantize_dequantize_error_bounds() {
        let original: Vec<f32> = (-8..=7).map(|x| x as f32 * 0.5).collect();
        let scale = 0.5;
        let zp = 0i8;

        let mut packed = vec![0u8; original.len().div_ceil(2)];
        quantize_f32_to_int4(&original, scale, zp, &mut packed);

        let mut recovered = vec![0.0f32; packed.len() * 2];
        dequantize_int4_to_f32(&packed, scale, zp, &mut recovered);

        for (i, (&orig, &rec)) in original.iter().zip(recovered.iter()).enumerate() {
            assert!(
                (orig - rec).abs() <= scale + 1e-6,
                "error too large at {i}: original={orig}, recovered={rec}"
            );
        }
    }

    #[test]
    fn quantize_clamps_to_range() {
        let data = vec![100.0, -100.0, 0.0, 3.5];
        let scale = 1.0;
        let zp = 0i8;
        let mut packed = vec![0u8; 2];
        quantize_f32_to_int4(&data, scale, zp, &mut packed);
        let mut out = vec![0.0f32; 4];
        dequantize_int4_to_f32(&packed, scale, zp, &mut out);
        assert!((out[0] - 7.0).abs() < 1e-6); // clamped to 7
        assert!((out[1] - (-8.0)).abs() < 1e-6); // clamped to -8
        assert!((out[2] - 0.0).abs() < 1e-6);
        assert!((out[3] - 4.0).abs() < 1e-6); // round(3.5) = 4
    }

    #[test]
    fn quantize_odd_length() {
        let data = vec![1.0, 2.0, 3.0];
        let scale = 1.0;
        let zp = 0i8;
        let mut packed = vec![0u8; 2];
        quantize_f32_to_int4(&data, scale, zp, &mut packed);
        let mut out = vec![0.0f32; 4];
        dequantize_int4_to_f32(&packed, scale, zp, &mut out);
        assert!((out[0] - 1.0).abs() < 1e-6);
        assert!((out[1] - 2.0).abs() < 1e-6);
        assert!((out[2] - 3.0).abs() < 1e-6);
    }
}
