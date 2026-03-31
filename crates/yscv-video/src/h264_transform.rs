//! H.264 inverse transform and dequantization.

// ---------------------------------------------------------------------------
// Inverse 4x4 integer DCT (H.264 specification)
// ---------------------------------------------------------------------------

/// Performs the H.264 4x4 inverse integer transform in-place.
///
/// The transform uses the simplified butterfly operations specified in
/// ITU-T H.264 section 8.5.12. Coefficients should already be dequantized.
#[allow(unsafe_code)]
pub fn inverse_dct_4x4(coeffs: &mut [i32; 16]) {
    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            inverse_dct_4x4_neon(coeffs);
        }
        return;
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("sse2") {
            unsafe {
                inverse_dct_4x4_sse2(coeffs);
            }
            return;
        }
    }

    #[allow(unreachable_code)]
    inverse_dct_4x4_scalar(coeffs);
}

fn inverse_dct_4x4_scalar(coeffs: &mut [i32; 16]) {
    for i in 0..4 {
        let base = i * 4;
        let s0 = coeffs[base];
        let s1 = coeffs[base + 1];
        let s2 = coeffs[base + 2];
        let s3 = coeffs[base + 3];
        let e0 = s0 + s2;
        let e1 = s0 - s2;
        let e2 = (s1 >> 1) - s3;
        let e3 = s1 + (s3 >> 1);
        coeffs[base] = e0 + e3;
        coeffs[base + 1] = e1 + e2;
        coeffs[base + 2] = e1 - e2;
        coeffs[base + 3] = e0 - e3;
    }
    for j in 0..4 {
        let s0 = coeffs[j];
        let s1 = coeffs[4 + j];
        let s2 = coeffs[8 + j];
        let s3 = coeffs[12 + j];
        let e0 = s0 + s2;
        let e1 = s0 - s2;
        let e2 = (s1 >> 1) - s3;
        let e3 = s1 + (s3 >> 1);
        coeffs[j] = (e0 + e3 + 32) >> 6;
        coeffs[4 + j] = (e1 + e2 + 32) >> 6;
        coeffs[8 + j] = (e1 - e2 + 32) >> 6;
        coeffs[12 + j] = (e0 - e3 + 32) >> 6;
    }
}

/// NEON SIMD 4x4 inverse DCT — processes each row/column as a 4-wide i32 vector.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
unsafe fn inverse_dct_4x4_neon(coeffs: &mut [i32; 16]) {
    use std::arch::aarch64::*;

    let ptr = coeffs.as_mut_ptr();

    // Row pass: load 4 rows, butterfly in-place
    for i in 0..4 {
        let row = vld1q_s32(ptr.add(i * 4));
        let s0 = vgetq_lane_s32(row, 0);
        let s1 = vgetq_lane_s32(row, 1);
        let s2 = vgetq_lane_s32(row, 2);
        let s3 = vgetq_lane_s32(row, 3);
        let e0 = s0 + s2;
        let e1 = s0 - s2;
        let e2 = (s1 >> 1) - s3;
        let e3 = s1 + (s3 >> 1);
        let out = [e0 + e3, e1 + e2, e1 - e2, e0 - e3];
        vst1q_s32(ptr.add(i * 4), vld1q_s32(out.as_ptr()));
    }

    // Column pass: load 4 columns as rows of transposed matrix, process, transpose back
    // Load all 4 rows
    let r0 = vld1q_s32(ptr);
    let r1 = vld1q_s32(ptr.add(4));
    let r2 = vld1q_s32(ptr.add(8));
    let r3 = vld1q_s32(ptr.add(12));

    // Transpose 4x4: use NEON zip/unzip
    let t01_lo = vzipq_s32(r0, r2); // interleave r0,r2
    let t01_hi = vzipq_s32(r1, r3); // interleave r1,r3
    let col0 = vzipq_s32(t01_lo.0, t01_hi.0).0;
    let col1 = vzipq_s32(t01_lo.0, t01_hi.0).1;
    let col2 = vzipq_s32(t01_lo.1, t01_hi.1).0;
    let col3 = vzipq_s32(t01_lo.1, t01_hi.1).1;

    // Butterfly on each column (now in registers as rows)
    let _round = vdupq_n_s32(32);
    for (col_vec, j) in [(col0, 0), (col1, 1), (col2, 2), (col3, 3)] {
        let s0 = vgetq_lane_s32(col_vec, 0);
        let s1 = vgetq_lane_s32(col_vec, 1);
        let s2 = vgetq_lane_s32(col_vec, 2);
        let s3 = vgetq_lane_s32(col_vec, 3);
        let e0 = s0 + s2;
        let e1 = s0 - s2;
        let e2 = (s1 >> 1) - s3;
        let e3 = s1 + (s3 >> 1);
        *ptr.add(j) = (e0 + e3 + 32) >> 6;
        *ptr.add(4 + j) = (e1 + e2 + 32) >> 6;
        *ptr.add(8 + j) = (e1 - e2 + 32) >> 6;
        *ptr.add(12 + j) = (e0 - e3 + 32) >> 6;
    }
}

/// SSE2 SIMD 4x4 inverse DCT.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
unsafe fn inverse_dct_4x4_sse2(coeffs: &mut [i32; 16]) {
    use std::arch::x86_64::*;

    let ptr = coeffs.as_mut_ptr();

    // Row pass
    for i in 0..4 {
        let row = _mm_loadu_si128(ptr.add(i * 4) as *const __m128i);
        let s0 = _mm_extract_epi32::<0>(row);
        let s1 = _mm_extract_epi32::<1>(row);
        let s2 = _mm_extract_epi32::<2>(row);
        let s3 = _mm_extract_epi32::<3>(row);
        let e0 = s0 + s2;
        let e1 = s0 - s2;
        let e2 = (s1 >> 1) - s3;
        let e3 = s1 + (s3 >> 1);
        let out = _mm_set_epi32(e0 - e3, e1 - e2, e1 + e2, e0 + e3);
        _mm_storeu_si128(ptr.add(i * 4) as *mut __m128i, out);
    }

    // Column pass (scalar — SSE2 doesn't have efficient column extract)
    for j in 0..4 {
        let s0 = *ptr.add(j);
        let s1 = *ptr.add(4 + j);
        let s2 = *ptr.add(8 + j);
        let s3 = *ptr.add(12 + j);
        let e0 = s0 + s2;
        let e1 = s0 - s2;
        let e2 = (s1 >> 1) - s3;
        let e3 = s1 + (s3 >> 1);
        *ptr.add(j) = (e0 + e3 + 32) >> 6;
        *ptr.add(4 + j) = (e1 + e2 + 32) >> 6;
        *ptr.add(8 + j) = (e1 - e2 + 32) >> 6;
        *ptr.add(12 + j) = (e0 - e3 + 32) >> 6;
    }
}

// ---------------------------------------------------------------------------
// Inverse quantization (dequantization)
// ---------------------------------------------------------------------------

/// H.264 dequantization scale factors for qp%6, position-dependent.
/// LevelScale(m) values from the spec for flat scaling matrices.
const DEQUANT_SCALE: [[i32; 16]; 6] = [
    [
        10, 13, 10, 13, 13, 16, 13, 16, 10, 13, 10, 13, 13, 16, 13, 16,
    ],
    [
        11, 14, 11, 14, 14, 18, 14, 18, 11, 14, 11, 14, 14, 18, 14, 18,
    ],
    [
        13, 16, 13, 16, 16, 20, 16, 20, 13, 16, 13, 16, 16, 20, 16, 20,
    ],
    [
        14, 18, 14, 18, 18, 23, 18, 23, 14, 18, 14, 18, 18, 23, 18, 23,
    ],
    [
        16, 20, 16, 20, 20, 25, 20, 25, 16, 20, 16, 20, 20, 25, 20, 25,
    ],
    [
        18, 23, 18, 23, 23, 29, 23, 29, 18, 23, 18, 23, 23, 29, 23, 29,
    ],
];

/// Dequantizes a 4x4 block of transform coefficients in-place.
///
/// Applies H.264 inverse quantization: `level * scale[qp%6][pos] << (qp/6)`.
/// Clamps QP to the valid range [0, 51].
#[allow(unsafe_code)]
pub fn dequant_4x4(coeffs: &mut [i32; 16], qp: i32) {
    let qp = qp.clamp(0, 51);
    let qp_div6 = (qp / 6) as u32;
    let qp_mod6 = (qp % 6) as usize;
    let scale = &DEQUANT_SCALE[qp_mod6];

    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            use std::arch::aarch64::*;
            let shift = qp_div6 as i32;
            let shift_v = vdupq_n_s32(shift);
            let ptr = coeffs.as_mut_ptr();
            let sptr = scale.as_ptr();
            for i in (0..16).step_by(4) {
                let c = vld1q_s32(ptr.add(i));
                let s = vld1q_s32(sptr.add(i));
                let mul = vmulq_s32(c, s);
                let shifted = vshlq_s32(mul, shift_v);
                vst1q_s32(ptr.add(i), shifted);
            }
        }
        return;
    }

    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("sse2") {
        unsafe {
            use std::arch::x86_64::*;
            let ptr = coeffs.as_mut_ptr();
            let sptr = scale.as_ptr();
            for i in (0..16).step_by(4) {
                let c = _mm_loadu_si128(ptr.add(i) as *const __m128i);
                let s = _mm_loadu_si128(sptr.add(i) as *const __m128i);
                // SSE2 doesn't have _mm_mullo_epi32 (needs SSE4.1), use manual
                // multiply: split into lo/hi, multiply, recombine
                let c_lo = _mm_shuffle_epi32(c, 0b11_01_10_00);
                let s_lo = _mm_shuffle_epi32(s, 0b11_01_10_00);
                let mul_02 = _mm_mul_epu32(c, s);
                let mul_13 = _mm_mul_epu32(_mm_srli_si128(c, 4), _mm_srli_si128(s, 4));
                // Pack low 32 bits of each 64-bit product
                let r0 = _mm_cvtsi128_si32(mul_02);
                let r1 = _mm_cvtsi128_si32(mul_13);
                let r2 = _mm_cvtsi128_si32(_mm_srli_si128(mul_02, 8));
                let r3 = _mm_cvtsi128_si32(_mm_srli_si128(mul_13, 8));
                let result =
                    _mm_set_epi32(r3 << qp_div6, r2 << qp_div6, r1 << qp_div6, r0 << qp_div6);
                _mm_storeu_si128(ptr.add(i) as *mut __m128i, result);
            }
        }
        return;
    }

    #[allow(unreachable_code)]
    for i in 0..16 {
        coeffs[i] = (coeffs[i] * scale[i]) << qp_div6;
    }
}

// ---------------------------------------------------------------------------
// 4x4 block zigzag scan order
// ---------------------------------------------------------------------------

/// H.264 4x4 zigzag scan order: maps scan index to (row, col) position.
pub(crate) const ZIGZAG_4X4: [(usize, usize); 16] = [
    (0, 0),
    (0, 1),
    (1, 0),
    (2, 0),
    (1, 1),
    (0, 2),
    (0, 3),
    (1, 2),
    (2, 1),
    (3, 0),
    (3, 1),
    (2, 2),
    (1, 3),
    (2, 3),
    (3, 2),
    (3, 3),
];

// ---------------------------------------------------------------------------
// Inverse 8x8 integer DCT (H.264 specification, section 8.5.13)
// ---------------------------------------------------------------------------

/// Performs the H.264 8x8 inverse integer transform in-place.
///
/// Uses the simplified butterfly operations specified in ITU-T H.264
/// section 8.5.13 (Table 8-13). Coefficients should already be dequantized.
pub fn inverse_dct_8x8(coeffs: &mut [i32; 64]) {
    // Process rows (8 iterations)
    for i in 0..8 {
        let base = i * 8;
        let a0 = coeffs[base] + coeffs[base + 4];
        let a1 = -coeffs[base + 3] + coeffs[base + 5] - coeffs[base + 7] - (coeffs[base + 7] >> 1);
        let a2 = coeffs[base] - coeffs[base + 4];
        let a3 = coeffs[base + 1] + coeffs[base + 7] - coeffs[base + 3] - (coeffs[base + 3] >> 1);
        let a4 = (coeffs[base + 2] >> 1) - coeffs[base + 6];
        let a5 = -coeffs[base + 1] + coeffs[base + 7] + coeffs[base + 5] + (coeffs[base + 5] >> 1);
        let a6 = coeffs[base + 2] + (coeffs[base + 6] >> 1);
        let a7 = coeffs[base + 3] + coeffs[base + 5] + coeffs[base + 1] + (coeffs[base + 1] >> 1);

        let b0 = a0 + a6;
        let b1 = a2 + a4;
        let b2 = a2 - a4;
        let b3 = a0 - a6;

        coeffs[base] = b0 + a7;
        coeffs[base + 1] = b1 + a5;
        coeffs[base + 2] = b2 + a3;
        coeffs[base + 3] = b3 + a1;
        coeffs[base + 4] = b3 - a1;
        coeffs[base + 5] = b2 - a3;
        coeffs[base + 6] = b1 - a5;
        coeffs[base + 7] = b0 - a7;
    }
    // Process columns (same butterfly with stride 8 and final normalization >>6)
    for j in 0..8 {
        let a0 = coeffs[j] + coeffs[32 + j];
        let a1 = -coeffs[24 + j] + coeffs[40 + j] - coeffs[56 + j] - (coeffs[56 + j] >> 1);
        let a2 = coeffs[j] - coeffs[32 + j];
        let a3 = coeffs[8 + j] + coeffs[56 + j] - coeffs[24 + j] - (coeffs[24 + j] >> 1);
        let a4 = (coeffs[16 + j] >> 1) - coeffs[48 + j];
        let a5 = -coeffs[8 + j] + coeffs[56 + j] + coeffs[40 + j] + (coeffs[40 + j] >> 1);
        let a6 = coeffs[16 + j] + (coeffs[48 + j] >> 1);
        let a7 = coeffs[24 + j] + coeffs[40 + j] + coeffs[8 + j] + (coeffs[8 + j] >> 1);

        let b0 = a0 + a6;
        let b1 = a2 + a4;
        let b2 = a2 - a4;
        let b3 = a0 - a6;

        coeffs[j] = (b0 + a7 + 32) >> 6;
        coeffs[8 + j] = (b1 + a5 + 32) >> 6;
        coeffs[16 + j] = (b2 + a3 + 32) >> 6;
        coeffs[24 + j] = (b3 + a1 + 32) >> 6;
        coeffs[32 + j] = (b3 - a1 + 32) >> 6;
        coeffs[40 + j] = (b2 - a3 + 32) >> 6;
        coeffs[48 + j] = (b1 - a5 + 32) >> 6;
        coeffs[56 + j] = (b0 - a7 + 32) >> 6;
    }
}

// ---------------------------------------------------------------------------
// 8x8 inverse quantization (dequantization)
// ---------------------------------------------------------------------------

/// H.264 8x8 dequantization scale factors for qp%6.
/// LevelScale8x8(m) values from ITU-T H.264 Table 8-15 for flat scaling matrices.
/// Each sub-array has 64 entries in raster order.
const DEQUANT_SCALE_8X8: [[i32; 64]; 6] = [
    [
        20, 19, 25, 19, 20, 19, 25, 19, 19, 18, 24, 18, 19, 18, 24, 18, 25, 24, 32, 24, 25, 24, 32,
        24, 19, 18, 24, 18, 19, 18, 24, 18, 20, 19, 25, 19, 20, 19, 25, 19, 19, 18, 24, 18, 19, 18,
        24, 18, 25, 24, 32, 24, 25, 24, 32, 24, 19, 18, 24, 18, 19, 18, 24, 18,
    ],
    [
        22, 21, 28, 21, 22, 21, 28, 21, 21, 19, 26, 19, 21, 19, 26, 19, 28, 26, 35, 26, 28, 26, 35,
        26, 21, 19, 26, 19, 21, 19, 26, 19, 22, 21, 28, 21, 22, 21, 28, 21, 21, 19, 26, 19, 21, 19,
        26, 19, 28, 26, 35, 26, 28, 26, 35, 26, 21, 19, 26, 19, 21, 19, 26, 19,
    ],
    [
        26, 24, 33, 24, 26, 24, 33, 24, 24, 23, 31, 23, 24, 23, 31, 23, 33, 31, 42, 31, 33, 31, 42,
        31, 24, 23, 31, 23, 24, 23, 31, 23, 26, 24, 33, 24, 26, 24, 33, 24, 24, 23, 31, 23, 24, 23,
        31, 23, 33, 31, 42, 31, 33, 31, 42, 31, 24, 23, 31, 23, 24, 23, 31, 23,
    ],
    [
        28, 26, 35, 26, 28, 26, 35, 26, 26, 25, 33, 25, 26, 25, 33, 25, 35, 33, 45, 33, 35, 33, 45,
        33, 26, 25, 33, 25, 26, 25, 33, 25, 28, 26, 35, 26, 28, 26, 35, 26, 26, 25, 33, 25, 26, 25,
        33, 25, 35, 33, 45, 33, 35, 33, 45, 33, 26, 25, 33, 25, 26, 25, 33, 25,
    ],
    [
        32, 30, 40, 30, 32, 30, 40, 30, 30, 28, 38, 28, 30, 28, 38, 28, 40, 38, 51, 38, 40, 38, 51,
        38, 30, 28, 38, 28, 30, 28, 38, 28, 32, 30, 40, 30, 32, 30, 40, 30, 30, 28, 38, 28, 30, 28,
        38, 28, 40, 38, 51, 38, 40, 38, 51, 38, 30, 28, 38, 28, 30, 28, 38, 28,
    ],
    [
        36, 34, 46, 34, 36, 34, 46, 34, 34, 32, 43, 32, 34, 32, 43, 32, 46, 43, 58, 43, 46, 43, 58,
        43, 34, 32, 43, 32, 34, 32, 43, 32, 36, 34, 46, 34, 36, 34, 46, 34, 34, 32, 43, 32, 34, 32,
        43, 32, 46, 43, 58, 43, 46, 43, 58, 43, 34, 32, 43, 32, 34, 32, 43, 32,
    ],
];

/// Dequantizes an 8x8 block of transform coefficients in-place.
///
/// Applies H.264 inverse quantization for 8x8 blocks:
/// `level * scale8x8[qp%6][pos] << (qp/6 - 6)` when qp/6 >= 6,
/// `(level * scale8x8[qp%6][pos] + (1 << (5-qp/6))) >> (6 - qp/6)` otherwise.
/// Clamps QP to the valid range [0, 51].
pub fn dequant_8x8(coeffs: &mut [i32; 64], qp: i32) {
    let qp = qp.clamp(0, 51);
    let qp_div6 = qp / 6;
    let qp_mod6 = (qp % 6) as usize;
    let scale = &DEQUANT_SCALE_8X8[qp_mod6];

    if qp_div6 >= 6 {
        let shift = (qp_div6 - 6) as u32;
        for i in 0..64 {
            coeffs[i] = (coeffs[i] * scale[i]) << shift;
        }
    } else {
        let shift = (6 - qp_div6) as u32;
        let round = 1i32 << (shift - 1);
        for i in 0..64 {
            coeffs[i] = (coeffs[i] * scale[i] + round) >> shift;
        }
    }
}

// ---------------------------------------------------------------------------
// 8x8 block zigzag scan order
// ---------------------------------------------------------------------------

/// H.264 8x8 zigzag scan order: maps scan index to raster position.
pub(crate) const ZIGZAG_8X8: [usize; 64] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20,
    13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59,
    52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
];

/// Converts scan-order coefficients to 8x8 raster order.
pub(crate) fn unscan_8x8(scan_coeffs: &[i32], out: &mut [i32; 64]) {
    *out = [0i32; 64];
    for (scan_idx, &val) in scan_coeffs.iter().enumerate().take(64) {
        out[ZIGZAG_8X8[scan_idx]] = val;
    }
}

/// Converts scan-order coefficients to 4x4 raster order.
pub(crate) fn unscan_4x4(scan_coeffs: &[i32], out: &mut [i32; 16]) {
    *out = [0i32; 16];
    for (scan_idx, &val) in scan_coeffs.iter().enumerate().take(16) {
        let (r, c) = ZIGZAG_4X4[scan_idx];
        out[r * 4 + c] = val;
    }
}
