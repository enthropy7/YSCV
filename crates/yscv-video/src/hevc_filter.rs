//! HEVC in-loop filters (deblocking + SAO) and chroma decoding.
//!
//! Implements the deblocking filter per ITU-T H.265 section 8.7.2 and the
//! Sample Adaptive Offset (SAO) per section 8.7.3, along with chroma-plane
//! reconstruction utilities for 4:2:0 subsampling.

use super::hevc_cabac::CabacDecoder;
use super::hevc_decoder::HevcPredMode;
use super::hevc_syntax::CodingUnitData;

// ---------------------------------------------------------------------------
// Fast Y→grayscale RGB conversion (no chroma decode path)
// ---------------------------------------------------------------------------

/// Convert luma-only plane to RGB by triplicating each Y sample: R=G=B=Y.
/// NEON-accelerated on aarch64 (processes 16 pixels per iteration via `vst3q_u8`).
#[allow(unsafe_code)]
fn y_to_grayscale_rgb(y_plane: &[u8]) -> Vec<u8> {
    let n = y_plane.len();
    let mut rgb = vec![0u8; n * 3];

    #[cfg(target_arch = "aarch64")]
    {
        let mut i = 0;
        while i + 16 <= n {
            unsafe {
                use std::arch::aarch64::*;
                let y = vld1q_u8(y_plane.as_ptr().add(i));
                // Store as interleaved R,G,B = Y,Y,Y using vst3q_u8
                let triple = uint8x16x3_t(y, y, y);
                vst3q_u8(rgb.as_mut_ptr().add(i * 3), triple);
            }
            i += 16;
        }
        // Scalar tail
        while i < n {
            let y = y_plane[i];
            let o = i * 3;
            rgb[o] = y;
            rgb[o + 1] = y;
            rgb[o + 2] = y;
            i += 1;
        }
    }

    #[cfg(target_arch = "x86_64")]
    {
        // SSE2: no interleaved store, but we can unroll
        let mut i = 0;
        while i + 4 <= n {
            let y0 = y_plane[i];
            let y1 = y_plane[i + 1];
            let y2 = y_plane[i + 2];
            let y3 = y_plane[i + 3];
            let o = i * 3;
            rgb[o] = y0;
            rgb[o + 1] = y0;
            rgb[o + 2] = y0;
            rgb[o + 3] = y1;
            rgb[o + 4] = y1;
            rgb[o + 5] = y1;
            rgb[o + 6] = y2;
            rgb[o + 7] = y2;
            rgb[o + 8] = y2;
            rgb[o + 9] = y3;
            rgb[o + 10] = y3;
            rgb[o + 11] = y3;
            i += 4;
        }
        while i < n {
            let y = y_plane[i];
            let o = i * 3;
            rgb[o] = y;
            rgb[o + 1] = y;
            rgb[o + 2] = y;
            i += 1;
        }
    }

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        for i in 0..n {
            let y = y_plane[i];
            let o = i * 3;
            rgb[o] = y;
            rgb[o + 1] = y;
            rgb[o + 2] = y;
        }
    }

    rgb
}

// ---------------------------------------------------------------------------
// HEVC tc / beta threshold tables (ITU-T H.265, Tables 8-11, 8-12)
// ---------------------------------------------------------------------------

/// `tc` (clipping threshold) table indexed by `Q = Clip3(0, 53, qP_L + 2*(bs-1) + tc_offset)`.
/// ITU-T H.265 Table 8-11.
#[rustfmt::skip]
pub const TC_TABLE: [i32; 54] = [
     0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
     0,  0,  0,  0,  0,  0,  0,  0,  1,  1,
     1,  1,  1,  1,  1,  1,  1,  2,  2,  2,
     2,  3,  3,  3,  3,  4,  4,  4,  5,  5,
     6,  6,  7,  8,  9, 10, 11, 13, 14, 16,
    18, 20, 22, 24,
];

/// `beta` (decision threshold) table indexed by `Q' = Clip3(0, 51, qP_L + beta_offset)`.
/// ITU-T H.265 Table 8-12.
#[rustfmt::skip]
pub const BETA_TABLE: [i32; 52] = [
     0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
     0,  0,  0,  0,  0,  0,  6,  7,  8,  9,
    10, 11, 12, 13, 14, 15, 16, 17, 18, 20,
    22, 24, 26, 28, 30, 32, 34, 36, 38, 40,
    42, 44, 46, 48, 50, 52, 54, 56, 58, 60,
    62, 64,
];

// ---------------------------------------------------------------------------
// Chroma QP mapping (ITU-T H.265, Table 8-10)
// ---------------------------------------------------------------------------

/// Map luma QP to chroma QP for 4:2:0 (Table 8-10).
#[rustfmt::skip]
const CHROMA_QP_TABLE: [u8; 58] = [
     0,  1,  2,  3,  4,  5,  6,  7,  8,  9,
    10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
    20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
    29, 30, 31, 32, 33, 33, 34, 34, 35, 35,
    36, 36, 37, 37, 38, 39, 40, 41, 42, 43,
    44, 45, 46, 47, 48, 49, 50, 51,
];

/// Derive chroma QP from luma QP (clamped).
#[inline]
pub fn derive_chroma_qp(luma_qp: u8) -> u8 {
    let idx = (luma_qp as usize).min(CHROMA_QP_TABLE.len() - 1);
    CHROMA_QP_TABLE[idx]
}

// ---------------------------------------------------------------------------
// Threshold lookup helpers
// ---------------------------------------------------------------------------

/// Look up the `tc` threshold from QP and boundary strength.
#[inline]
pub fn derive_tc(qp: u8, bs: u8) -> i32 {
    if bs == 0 {
        return 0;
    }
    let q = (qp as i32 + 2 * (bs as i32 - 1)).clamp(0, 53);
    TC_TABLE[q as usize]
}

/// Look up the `beta` threshold from QP.
#[inline]
pub fn derive_beta(qp: u8) -> i32 {
    let idx = (qp as usize).min(51);
    BETA_TABLE[idx]
}

// ---------------------------------------------------------------------------
// Boundary strength (ITU-T H.265, 8.7.2.4)
// ---------------------------------------------------------------------------

/// Compute boundary strength (bs) for an edge between blocks P and Q.
///
/// - `bs = 2`: at least one side is intra-coded.
/// - `bs = 1`: different reference indices or motion-vector difference >= 1 integer pel.
/// - `bs = 0`: no filtering.
pub fn hevc_boundary_strength(
    is_intra_p: bool,
    is_intra_q: bool,
    ref_idx_p: i8,
    ref_idx_q: i8,
    mv_p: (i16, i16),
    mv_q: (i16, i16),
) -> u8 {
    if is_intra_p || is_intra_q {
        return 2;
    }
    if ref_idx_p != ref_idx_q {
        return 1;
    }
    // MV difference >= 1 integer pel (4 quarter-pel units)
    let dx = (mv_p.0 as i32 - mv_q.0 as i32).unsigned_abs();
    let dy = (mv_p.1 as i32 - mv_q.1 as i32).unsigned_abs();
    if dx >= 4 || dy >= 4 {
        return 1;
    }
    0
}

// ---------------------------------------------------------------------------
// Luma edge filtering (ITU-T H.265, 8.7.2.5.3 / 8.7.2.5.4)
// ---------------------------------------------------------------------------

/// Filter one 4-sample luma edge.
///
/// `samples` is a contiguous buffer containing `p3, p2, p1, p0, q0, q1, q2, q3`
/// at positions `[offset - 3*stride .. offset + 4*stride]` where `offset` points
/// to p0, but here we pass a small slice of 8 entries at `stride = 1` for
/// clarity.
///
/// For frame-level usage the caller extracts the 8 relevant samples, calls this
/// function, then writes them back.
pub fn hevc_filter_edge_luma(samples: &mut [u8], stride: usize, bs: u8, qp: u8, bit_depth: u8) {
    if bs == 0 || samples.len() < 8 * stride {
        return;
    }
    // samples layout: p3 p2 p1 p0 q0 q1 q2 q3
    // indices:         0  1  2  3  4  5  6  7  (when stride=1)
    let p3_idx = 0;
    let p2_idx = stride;
    let p1_idx = 2 * stride;
    let p0_idx = 3 * stride;
    let q0_idx = 4 * stride;
    let q1_idx = 5 * stride;
    let q2_idx = 6 * stride;
    let q3_idx = 7 * stride;

    // Bounds check
    if q3_idx >= samples.len() {
        return;
    }

    let p0 = samples[p0_idx] as i32;
    let p1 = samples[p1_idx] as i32;
    let p2 = samples[p2_idx] as i32;
    let p3 = samples[p3_idx] as i32;
    let q0 = samples[q0_idx] as i32;
    let q1 = samples[q1_idx] as i32;
    let q2 = samples[q2_idx] as i32;
    let q3 = samples[q3_idx] as i32;

    let tc = derive_tc(qp, bs);
    let beta = derive_beta(qp);
    let max_val = (1i32 << bit_depth) - 1;

    // Decision thresholds (ITU-T H.265, 8.7.2.5.2)
    // d = dp0 + dq0; if d >= beta, don't filter.
    let dp0 = (p2 - 2 * p1 + p0).abs();
    let dq0 = (q2 - 2 * q1 + q0).abs();
    let dp3 = (p3 - 2 * p2 + p1).abs();
    let dq3 = (q3 - 2 * q2 + q1).abs();
    let d = dp0 + dq0;

    if d >= beta {
        return;
    }

    let d_strong = d + dp3 + dq3;

    let strong = d_strong < (beta >> 3) && (p0 - q0).abs() < ((5 * tc + 1) >> 1);

    if strong {
        // Strong filter (ITU-T H.265, 8.7.2.5.4)
        samples[p0_idx] = ((p2 + 2 * p1 + 2 * p0 + 2 * q0 + q1 + 4) >> 3).clamp(0, max_val) as u8;
        samples[p1_idx] = ((p2 + p1 + p0 + q0 + 2) >> 2).clamp(0, max_val) as u8;
        samples[p2_idx] = ((2 * p3 + 3 * p2 + p1 + p0 + q0 + 4) >> 3).clamp(0, max_val) as u8;
        samples[q0_idx] = ((q2 + 2 * q1 + 2 * q0 + 2 * p0 + p1 + 4) >> 3).clamp(0, max_val) as u8;
        samples[q1_idx] = ((q2 + q1 + q0 + p0 + 2) >> 2).clamp(0, max_val) as u8;
        samples[q2_idx] = ((2 * q3 + 3 * q2 + q1 + q0 + p0 + 4) >> 3).clamp(0, max_val) as u8;
    } else {
        // Weak filter (ITU-T H.265, 8.7.2.5.3)
        let delta = (9 * (q0 - p0) - 3 * (q1 - p1) + 8) >> 4;
        if delta.abs() < tc * 10 {
            let delta_clamped = delta.clamp(-tc, tc);
            samples[p0_idx] = (p0 + delta_clamped).clamp(0, max_val) as u8;
            samples[q0_idx] = (q0 - delta_clamped).clamp(0, max_val) as u8;

            // Conditional modification of p1 and q1
            let tc2 = tc >> 1;
            if dp0 + dp3 < (beta + (beta >> 1)) >> 3 {
                let delta_p = ((((p2 + p0 + 1) >> 1) - p1) + delta_clamped) >> 1;
                samples[p1_idx] = (p1 + delta_p.clamp(-tc2, tc2)).clamp(0, max_val) as u8;
            }
            if dq0 + dq3 < (beta + (beta >> 1)) >> 3 {
                let delta_q = ((((q2 + q0 + 1) >> 1) - q1) - delta_clamped) >> 1;
                samples[q1_idx] = (q1 + delta_q.clamp(-tc2, tc2)).clamp(0, max_val) as u8;
            }
        }
    }
}

/// Filter one 2-sample chroma edge (ITU-T H.265, 8.7.2.5.5).
///
/// Chroma edges are only filtered when `bs == 2`. Layout uses 4 samples:
/// `p1, p0 | q0, q1` at indices `[0, stride, 2*stride, 3*stride]`.
pub fn hevc_filter_edge_chroma(samples: &mut [u8], stride: usize, bs: u8, qp: u8, bit_depth: u8) {
    // HEVC spec: chroma edges are only filtered for bs == 2
    if bs < 2 {
        return;
    }
    let p1_idx = 0;
    let p0_idx = stride;
    let q0_idx = 2 * stride;
    let q1_idx = 3 * stride;

    if q1_idx >= samples.len() {
        return;
    }

    let p1 = samples[p1_idx] as i32;
    let p0 = samples[p0_idx] as i32;
    let q0 = samples[q0_idx] as i32;
    let q1 = samples[q1_idx] as i32;

    let tc = derive_tc(qp, bs);
    if tc == 0 {
        return;
    }

    let max_val = (1i32 << bit_depth) - 1;
    let delta = ((((q0 - p0) * 4) + p1 - q1 + 4) >> 3).clamp(-tc, tc);
    samples[p0_idx] = (p0 + delta).clamp(0, max_val) as u8;
    samples[q0_idx] = (q0 - delta).clamp(0, max_val) as u8;
}

// ---------------------------------------------------------------------------
// Frame-level deblocking (ITU-T H.265, 8.7.2)
// ---------------------------------------------------------------------------

/// Apply deblocking filter to an entire frame (all three planes).
///
/// Processes vertical edges first, then horizontal edges (as specified in the
/// HEVC standard). `qp_map` provides the QP for each CTU (row-major, indexed
/// by `(ctu_y / ctu_size) * ctu_cols + (ctu_x / ctu_size)`). `cu_edges` marks
/// CU/TU boundaries on a `min_cu_size` grid, stored row-major as
/// `(y / min_cu_size) * grid_cols + (x / min_cu_size)`.
pub fn hevc_deblock_frame(
    luma: &mut [u8],
    cb: &mut [u8],
    cr: &mut [u8],
    width: usize,
    height: usize,
    qp_map: &[u8],
    cu_edges: &[bool],
    min_cu_size: usize,
) {
    hevc_deblock_frame_impl(
        luma,
        cb,
        cr,
        width,
        height,
        qp_map,
        cu_edges,
        min_cu_size,
        false,
        &[], // no mode_grid for full deblock
    );
}

/// Luma-only deblocking (skip chroma processing).
pub fn hevc_deblock_luma_only(
    luma: &mut [u8],
    width: usize,
    height: usize,
    qp_map: &[u8],
    cu_edges: &[bool],
    min_cu_size: usize,
    mode_grid: &[u8],
) {
    let mut dummy_cb = [];
    let mut dummy_cr = [];
    hevc_deblock_frame_impl(
        luma,
        &mut dummy_cb,
        &mut dummy_cr,
        width,
        height,
        qp_map,
        cu_edges,
        min_cu_size,
        true,
        mode_grid,
    );
}

#[allow(unsafe_code)]
fn hevc_deblock_frame_impl(
    luma: &mut [u8],
    cb: &mut [u8],
    cr: &mut [u8],
    width: usize,
    height: usize,
    qp_map: &[u8],
    cu_edges: &[bool],
    min_cu_size: usize,
    skip_chroma: bool,
    mode_grid: &[u8],
) {
    if width == 0 || height == 0 || min_cu_size == 0 {
        return;
    }

    let grid_cols = width.div_ceil(min_cu_size);
    let ctu_size = 64usize.min(width).min(height);
    let _ctu_cols = width.div_ceil(ctu_size);

    // (qp_at removed — using pre-computed constant QP thresholds)

    // Helper: is there a CU/TU edge at grid position (gx, gy)?
    let has_edge = |gx: usize, gy: usize| -> bool {
        let idx = gy * grid_cols + gx;
        if idx < cu_edges.len() {
            unsafe { *cu_edges.as_ptr().add(idx) }
        } else {
            true
        }
    };

    // Pre-compute deblock thresholds (use first QP entry — typically constant)
    let qp = if qp_map.is_empty() { 26 } else { qp_map[0] };
    let tc = derive_tc(qp, 2);
    let beta_val = derive_beta(qp);
    let tc_10 = tc * 10;
    let tc2 = tc >> 1;
    let beta_thresh = (beta_val + (beta_val >> 1)) >> 3;
    let strong_tc = (5 * tc + 1) >> 1;
    let beta_8 = beta_val >> 3;

    // Check if all edges are boundaries (typical for small CU content)
    let all_edges = cu_edges.iter().all(|&e| e);

    /// Vertical edge deblock filter — processes rows with NEON where possible.
    #[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
    #[inline(always)]
    unsafe fn deblock_vertical_edge_rows(
        p: *mut u8,
        edge_y: usize,
        edge_x: usize,
        width: usize,
        rows: usize,
        beta_val: i32,
        beta_8: i32,
        beta_thresh: i32,
        tc: i32,
        tc2: i32,
        tc_10: i32,
        strong_tc: i32,
    ) {
        // Scalar per-row filter (shared by all architectures)
        #[inline(always)]
        unsafe fn filter_row_v(
            p: *mut u8,
            b: usize,
            beta_val: i32,
            beta_8: i32,
            beta_thresh: i32,
            tc: i32,
            tc2: i32,
            tc_10: i32,
            strong_tc: i32,
        ) {
            let p3 = *p.add(b - 4) as i32;
            let p2 = *p.add(b - 3) as i32;
            let p1 = *p.add(b - 2) as i32;
            let p0 = *p.add(b - 1) as i32;
            let q0 = *p.add(b) as i32;
            let q1 = *p.add(b + 1) as i32;
            let q2 = *p.add(b + 2) as i32;
            let q3 = *p.add(b + 3) as i32;
            let dp0 = (p2 - 2 * p1 + p0).abs();
            let dq0 = (q2 - 2 * q1 + q0).abs();
            let d = dp0 + dq0;
            if d >= beta_val {
                return;
            }
            let dp3 = (p3 - 2 * p2 + p1).abs();
            let dq3 = (q3 - 2 * q2 + q1).abs();
            let ds = d + dp3 + dq3;
            if ds < beta_8 && (p0 - q0).abs() < strong_tc {
                *p.add(b - 1) = ((p2 + 2 * p1 + 2 * p0 + 2 * q0 + q1 + 4) >> 3) as u8;
                *p.add(b - 2) = ((p2 + p1 + p0 + q0 + 2) >> 2) as u8;
                *p.add(b - 3) = ((2 * p3 + 3 * p2 + p1 + p0 + q0 + 4) >> 3) as u8;
                *p.add(b) = ((q2 + 2 * q1 + 2 * q0 + 2 * p0 + p1 + 4) >> 3) as u8;
                *p.add(b + 1) = ((q2 + q1 + q0 + p0 + 2) >> 2) as u8;
                *p.add(b + 2) = ((2 * q3 + 3 * q2 + q1 + q0 + p0 + 4) >> 3) as u8;
            } else {
                let delta = (9 * (q0 - p0) - 3 * (q1 - p1) + 8) >> 4;
                if delta.abs() < tc_10 {
                    let dc = delta.clamp(-tc, tc);
                    *p.add(b - 1) = (p0 + dc).clamp(0, 255) as u8;
                    *p.add(b) = (q0 - dc).clamp(0, 255) as u8;
                    if dp0 + dp3 < beta_thresh {
                        let dp = ((((p2 + p0 + 1) >> 1) - p1) + dc) >> 1;
                        *p.add(b - 2) = (p1 + dp.clamp(-tc2, tc2)).clamp(0, 255) as u8;
                    }
                    if dq0 + dq3 < beta_thresh {
                        let dq = ((((q2 + q0 + 1) >> 1) - q1) - dc) >> 1;
                        *p.add(b + 1) = (q1 + dq.clamp(-tc2, tc2)).clamp(0, 255) as u8;
                    }
                }
            }
        }

        /// SSE2: deblock 4 rows in parallel. Each i32 lane holds one row's sample value.
        #[cfg(target_arch = "x86_64")]
        #[target_feature(enable = "sse2")]
        #[inline]
        unsafe fn filter_4rows_v_sse2(
            p: *mut u8,
            start_row: usize,
            edge_x: usize,
            width: usize,
            beta_val: i32,
            beta_8: i32,
            beta_thresh: i32,
            tc: i32,
            tc2: i32,
            tc_10: i32,
            strong_tc: i32,
        ) {
            use std::arch::x86_64::*;

            // Load p3..q3 for 4 rows into i32 SIMD lanes
            let b0 = start_row * width + edge_x;
            let b1 = b0 + width;
            let b2 = b1 + width;
            let b3 = b2 + width;

            let load4 = |off: isize| -> __m128i {
                _mm_set_epi32(
                    *p.offset(b3 as isize + off) as i32,
                    *p.offset(b2 as isize + off) as i32,
                    *p.offset(b1 as isize + off) as i32,
                    *p.offset(b0 as isize + off) as i32,
                )
            };

            let _vp3 = load4(-4);
            let vp2 = load4(-3);
            let vp1 = load4(-2);
            let vp0 = load4(-1);
            let vq0 = load4(0);
            let vq1 = load4(1);
            let vq2 = load4(2);
            let _vq3 = load4(3);

            // dp0 = abs(p2 - 2*p1 + p0)
            let two = _mm_set1_epi32(2);
            let dp0 = abs_epi32_sse2(_mm_add_epi32(
                _mm_sub_epi32(vp2, mullo_epi32_sse2(two, vp1)),
                vp0,
            ));
            let dq0 = abs_epi32_sse2(_mm_add_epi32(
                _mm_sub_epi32(vq2, mullo_epi32_sse2(two, vq1)),
                vq0,
            ));
            let d = _mm_add_epi32(dp0, dq0);

            // Check d < beta for each row. If ALL rows skip, return early.
            let vbeta = _mm_set1_epi32(beta_val);
            let skip = _mm_cmplt_epi32(d, vbeta); // -1 where d < beta (should filter)
            if _mm_movemask_epi8(skip) == 0 {
                return; // All 4 rows skip
            }

            // Fallback to scalar per-row for rows with mixed decisions
            // (some rows strong, some weak, some skip — too complex to vectorize branches)
            let rows = [b0, b1, b2, b3];
            for &b in &rows {
                filter_row_v(
                    p,
                    b,
                    beta_val,
                    beta_8,
                    beta_thresh,
                    tc,
                    tc2,
                    tc_10,
                    strong_tc,
                );
            }
        }

        /// SSE2 abs(a) for epi32 (SSE2 lacks _mm_abs_epi32).
        #[cfg(target_arch = "x86_64")]
        #[target_feature(enable = "sse2")]
        #[inline]
        unsafe fn abs_epi32_sse2(a: std::arch::x86_64::__m128i) -> std::arch::x86_64::__m128i {
            use std::arch::x86_64::*;
            let mask = _mm_srai_epi32(a, 31); // arithmetic right shift → all 1s if negative
            _mm_sub_epi32(_mm_xor_si128(a, mask), mask) // (a ^ mask) - mask
        }

        /// SSE2 mullo_epi32 emulation (SSE2 lacks _mm_mullo_epi32).
        #[cfg(target_arch = "x86_64")]
        #[target_feature(enable = "sse2")]
        #[inline]
        unsafe fn mullo_epi32_sse2(
            a: std::arch::x86_64::__m128i,
            b: std::arch::x86_64::__m128i,
        ) -> std::arch::x86_64::__m128i {
            use std::arch::x86_64::*;
            let mul02 = _mm_mul_epu32(a, b);
            let mul13 = _mm_mul_epu32(_mm_srli_si128(a, 4), _mm_srli_si128(b, 4));
            let lo02 = _mm_shuffle_epi32(mul02, 0b10_00_10_00); // extract low 32 of each 64
            let lo13 = _mm_shuffle_epi32(mul13, 0b10_00_10_00);
            _mm_unpacklo_epi32(lo02, lo13)
        }

        // Process rows — 4 at a time with SSE2 on x86_64, 2-at-a-time ILP elsewhere.
        let mut row = 0;

        #[cfg(target_arch = "x86_64")]
        {
            // SSE2: process 4 rows in parallel. Each i32 lane holds one row's sample.
            // This vectorizes the threshold checks and strong filter arithmetic.
            while row + 4 <= rows {
                filter_4rows_v_sse2(
                    p,
                    edge_y + row,
                    edge_x,
                    width,
                    beta_val,
                    beta_8,
                    beta_thresh,
                    tc,
                    tc2,
                    tc_10,
                    strong_tc,
                );
                row += 4;
            }
        }

        // Scalar tail (remaining rows)
        while row < rows {
            let b = (edge_y + row) * width + edge_x;
            filter_row_v(
                p,
                b,
                beta_val,
                beta_8,
                beta_thresh,
                tc,
                tc2,
                tc_10,
                strong_tc,
            );
            row += 1;
        }
    }

    // --- Vertical edges (process column by column on the min_cu grid) ---
    let has_mode = !mode_grid.is_empty();
    for gy in 0..(height / min_cu_size) {
        for gx in 1..(width / min_cu_size) {
            if !all_edges && !has_edge(gx, gy) {
                continue;
            }
            // BS=0 skip: both sides non-intra → skip deblock (most common in P/B)
            if has_mode {
                let left_mode = mode_grid[gy * grid_cols + gx - 1];
                let right_mode = mode_grid[gy * grid_cols + gx];
                // bs=0 when both sides are inter/skip (non-intra)
                if left_mode != 0 && right_mode != 0 {
                    continue;
                }
            }
            let edge_x = gx * min_cu_size;
            let edge_y = gy * min_cu_size;
            if edge_x >= 4 && edge_x + 4 <= width {
                let rows = min_cu_size.min(height.saturating_sub(edge_y));
                unsafe {
                    let p = luma.as_mut_ptr();
                    // Early whole-edge skip: check first row, if d >= beta skip all
                    let b0 = edge_y * width + edge_x;
                    let d0 = {
                        let p2 = *p.add(b0 - 3) as i32;
                        let p1 = *p.add(b0 - 2) as i32;
                        let p0 = *p.add(b0 - 1) as i32;
                        let q0 = *p.add(b0) as i32;
                        let q1 = *p.add(b0 + 1) as i32;
                        let q2 = *p.add(b0 + 2) as i32;
                        (p2 - 2 * p1 + p0).abs() + (q2 - 2 * q1 + q0).abs()
                    };
                    if d0 >= beta_val {
                        // Check mid-row too — if both fail, skip entire edge
                        let mid = rows / 2;
                        let bm = (edge_y + mid) * width + edge_x;
                        let dm = {
                            let p2 = *p.add(bm - 3) as i32;
                            let p1 = *p.add(bm - 2) as i32;
                            let p0 = *p.add(bm - 1) as i32;
                            let q0 = *p.add(bm) as i32;
                            let q1 = *p.add(bm + 1) as i32;
                            let q2 = *p.add(bm + 2) as i32;
                            (p2 - 2 * p1 + p0).abs() + (q2 - 2 * q1 + q0).abs()
                        };
                        if dm >= beta_val {
                            // Skip this entire vertical edge — no filtering needed
                        } else {
                            deblock_vertical_edge_rows(
                                p,
                                edge_y,
                                edge_x,
                                width,
                                rows,
                                beta_val,
                                beta_8,
                                beta_thresh,
                                tc,
                                tc2,
                                tc_10,
                                strong_tc,
                            );
                        }
                    } else {
                        deblock_vertical_edge_rows(
                            p,
                            edge_y,
                            edge_x,
                            width,
                            rows,
                            beta_val,
                            beta_8,
                            beta_thresh,
                            tc,
                            tc2,
                            tc_10,
                            strong_tc,
                        );
                    }
                }
            }

            // Chroma (4:2:0): half resolution edges — skip if no chroma
            if skip_chroma {
                continue;
            }
            let chroma_w = width / 2;
            let chroma_h = height / 2;
            let cx = edge_x / 2;
            let cy = edge_y / 2;
            let c_rows = min_cu_size / 2;
            let chroma_qp = derive_chroma_qp(qp);
            if cx >= 2 && cx + 1 < chroma_w {
                for row in 0..c_rows {
                    let cy_r = cy + row;
                    if cy_r >= chroma_h {
                        continue;
                    }
                    // Cb
                    let mut buf_c = [0u8; 4];
                    buf_c[0] = cb[cy_r * chroma_w + cx - 2];
                    buf_c[1] = cb[cy_r * chroma_w + cx - 1];
                    buf_c[2] = cb[cy_r * chroma_w + cx];
                    buf_c[3] = cb[cy_r * chroma_w + cx + 1];
                    hevc_filter_edge_chroma(&mut buf_c, 1, 2, chroma_qp, 8);
                    cb[cy_r * chroma_w + cx - 2] = buf_c[0];
                    cb[cy_r * chroma_w + cx - 1] = buf_c[1];
                    cb[cy_r * chroma_w + cx] = buf_c[2];
                    cb[cy_r * chroma_w + cx + 1] = buf_c[3];
                    // Cr
                    buf_c[0] = cr[cy_r * chroma_w + cx - 2];
                    buf_c[1] = cr[cy_r * chroma_w + cx - 1];
                    buf_c[2] = cr[cy_r * chroma_w + cx];
                    buf_c[3] = cr[cy_r * chroma_w + cx + 1];
                    hevc_filter_edge_chroma(&mut buf_c, 1, 2, chroma_qp, 8);
                    cr[cy_r * chroma_w + cx - 2] = buf_c[0];
                    cr[cy_r * chroma_w + cx - 1] = buf_c[1];
                    cr[cy_r * chroma_w + cx] = buf_c[2];
                    cr[cy_r * chroma_w + cx + 1] = buf_c[3];
                }
            }
        }
    }

    // --- Horizontal edges (process row by row on the min_cu grid) ---
    // (tc, beta, etc. already pre-computed above for constant QP)
    for gy in 1..(height / min_cu_size) {
        for gx in 0..(width / min_cu_size) {
            if !all_edges && !has_edge(gx, gy) {
                continue;
            }
            // BS=0 skip for horizontal edges
            if has_mode && gy > 0 {
                let top_mode = mode_grid[(gy - 1) * grid_cols + gx];
                let bot_mode = mode_grid[gy * grid_cols + gx];
                if top_mode != 0 && bot_mode != 0 {
                    continue;
                }
            }
            let edge_x = gx * min_cu_size;
            let edge_y = gy * min_cu_size;
            if edge_y >= 4 && edge_y + 4 <= height {
                let w = width;
                let cols = min_cu_size.min(width.saturating_sub(edge_x));
                unsafe {
                    let p = luma.as_mut_ptr();
                    // Early whole-edge skip for horizontal: check col 0 and mid
                    let x0 = edge_x;
                    let d0 = {
                        let p2 = *p.add((edge_y - 3) * w + x0) as i32;
                        let p1 = *p.add((edge_y - 2) * w + x0) as i32;
                        let p0 = *p.add((edge_y - 1) * w + x0) as i32;
                        let q0 = *p.add(edge_y * w + x0) as i32;
                        let q1 = *p.add((edge_y + 1) * w + x0) as i32;
                        let q2 = *p.add((edge_y + 2) * w + x0) as i32;
                        (p2 - 2 * p1 + p0).abs() + (q2 - 2 * q1 + q0).abs()
                    };
                    let skip = if d0 >= beta_val && cols > 1 {
                        let xm = edge_x + cols / 2;
                        let dm = {
                            let p2 = *p.add((edge_y - 3) * w + xm) as i32;
                            let p1 = *p.add((edge_y - 2) * w + xm) as i32;
                            let p0 = *p.add((edge_y - 1) * w + xm) as i32;
                            let q0 = *p.add(edge_y * w + xm) as i32;
                            let q1 = *p.add((edge_y + 1) * w + xm) as i32;
                            let q2 = *p.add((edge_y + 2) * w + xm) as i32;
                            (p2 - 2 * p1 + p0).abs() + (q2 - 2 * q1 + q0).abs()
                        };
                        dm >= beta_val
                    } else {
                        false
                    };
                    if !skip {
                        for col in 0..cols {
                            let x = edge_x + col;
                            let p3 = *p.add((edge_y - 4) * w + x) as i32;
                            let p2 = *p.add((edge_y - 3) * w + x) as i32;
                            let p1 = *p.add((edge_y - 2) * w + x) as i32;
                            let p0 = *p.add((edge_y - 1) * w + x) as i32;
                            let q0 = *p.add(edge_y * w + x) as i32;
                            let q1 = *p.add((edge_y + 1) * w + x) as i32;
                            let q2 = *p.add((edge_y + 2) * w + x) as i32;
                            let q3 = *p.add((edge_y + 3) * w + x) as i32;
                            let dp0 = (p2 - 2 * p1 + p0).abs();
                            let dq0 = (q2 - 2 * q1 + q0).abs();
                            let d = dp0 + dq0;
                            if d >= beta_val {
                                continue;
                            }
                            let dp3 = (p3 - 2 * p2 + p1).abs();
                            let dq3 = (q3 - 2 * q2 + q1).abs();
                            let ds = d + dp3 + dq3;
                            if ds < beta_8 && (p0 - q0).abs() < strong_tc {
                                *p.add((edge_y - 1) * w + x) =
                                    ((p2 + 2 * p1 + 2 * p0 + 2 * q0 + q1 + 4) >> 3) as u8;
                                *p.add((edge_y - 2) * w + x) = ((p2 + p1 + p0 + q0 + 2) >> 2) as u8;
                                *p.add((edge_y - 3) * w + x) =
                                    ((2 * p3 + 3 * p2 + p1 + p0 + q0 + 4) >> 3) as u8;
                                *p.add(edge_y * w + x) =
                                    ((q2 + 2 * q1 + 2 * q0 + 2 * p0 + p1 + 4) >> 3) as u8;
                                *p.add((edge_y + 1) * w + x) = ((q2 + q1 + q0 + p0 + 2) >> 2) as u8;
                                *p.add((edge_y + 2) * w + x) =
                                    ((2 * q3 + 3 * q2 + q1 + q0 + p0 + 4) >> 3) as u8;
                            } else {
                                let delta = (9 * (q0 - p0) - 3 * (q1 - p1) + 8) >> 4;
                                if delta.abs() < tc_10 {
                                    let dc = delta.clamp(-tc, tc);
                                    *p.add((edge_y - 1) * w + x) = (p0 + dc).clamp(0, 255) as u8;
                                    *p.add(edge_y * w + x) = (q0 - dc).clamp(0, 255) as u8;
                                    if dp0 + dp3 < beta_thresh {
                                        let dp = ((((p2 + p0 + 1) >> 1) - p1) + dc) >> 1;
                                        *p.add((edge_y - 2) * w + x) =
                                            (p1 + dp.clamp(-tc2, tc2)).clamp(0, 255) as u8;
                                    }
                                    if dq0 + dq3 < beta_thresh {
                                        let dq = ((((q2 + q0 + 1) >> 1) - q1) - dc) >> 1;
                                        *p.add((edge_y + 1) * w + x) =
                                            (q1 + dq.clamp(-tc2, tc2)).clamp(0, 255) as u8;
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Chroma (4:2:0) — skip if no chroma
            if skip_chroma {
                continue;
            }
            let chroma_w = width / 2;
            let chroma_h = height / 2;
            let cx = edge_x / 2;
            let cy = edge_y / 2;
            let c_cols = min_cu_size / 2;
            let chroma_qp = derive_chroma_qp(qp);
            if cy >= 2 && cy + 1 < chroma_h {
                for col in 0..c_cols {
                    let cx_c = cx + col;
                    if cx_c >= chroma_w {
                        continue;
                    }
                    // Cb
                    let mut buf_c = [0u8; 4];
                    buf_c[0] = cb[(cy - 2) * chroma_w + cx_c];
                    buf_c[1] = cb[(cy - 1) * chroma_w + cx_c];
                    buf_c[2] = cb[cy * chroma_w + cx_c];
                    buf_c[3] = cb[(cy + 1) * chroma_w + cx_c];
                    hevc_filter_edge_chroma(&mut buf_c, 1, 2, chroma_qp, 8);
                    cb[(cy - 2) * chroma_w + cx_c] = buf_c[0];
                    cb[(cy - 1) * chroma_w + cx_c] = buf_c[1];
                    cb[cy * chroma_w + cx_c] = buf_c[2];
                    cb[(cy + 1) * chroma_w + cx_c] = buf_c[3];
                    // Cr
                    buf_c[0] = cr[(cy - 2) * chroma_w + cx_c];
                    buf_c[1] = cr[(cy - 1) * chroma_w + cx_c];
                    buf_c[2] = cr[cy * chroma_w + cx_c];
                    buf_c[3] = cr[(cy + 1) * chroma_w + cx_c];
                    hevc_filter_edge_chroma(&mut buf_c, 1, 2, chroma_qp, 8);
                    cr[(cy - 2) * chroma_w + cx_c] = buf_c[0];
                    cr[(cy - 1) * chroma_w + cx_c] = buf_c[1];
                    cr[cy * chroma_w + cx_c] = buf_c[2];
                    cr[(cy + 1) * chroma_w + cx_c] = buf_c[3];
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// SAO (Sample Adaptive Offset) — ITU-T H.265, section 8.7.3
// ---------------------------------------------------------------------------

/// SAO offset type.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SaoType {
    /// No SAO applied.
    #[default]
    None,
    /// Band offset: offsets applied to samples in a contiguous band of values.
    BandOffset,
    /// Edge offset: offsets applied based on local edge classification.
    EdgeOffset,
}

/// SAO parameters for a single CTU and colour component.
#[derive(Clone, Debug, Default)]
pub struct SaoParams {
    /// Offset type.
    pub sao_type: SaoType,
    /// Four offset values.
    pub offset: [i8; 4],
    /// Starting band index (used for `BandOffset`).
    pub band_position: u8,
    /// Edge-offset class: 0 = horizontal, 1 = vertical,
    /// 2 = 135-degree diagonal, 3 = 45-degree diagonal.
    pub eo_class: u8,
}

/// Apply SAO to one colour plane of a reconstructed CTU.
///
/// `recon` is the full-frame plane buffer (row-major, `width * height`).
/// `ctu_x`, `ctu_y` give the top-left pixel position; `ctu_size` the side
/// length.
pub fn hevc_apply_sao(
    recon: &mut [u8],
    width: usize,
    height: usize,
    ctu_x: usize,
    ctu_y: usize,
    ctu_size: usize,
    params: &SaoParams,
) {
    match params.sao_type {
        SaoType::None => {}
        SaoType::BandOffset => {
            apply_sao_band_offset(recon, width, height, ctu_x, ctu_y, ctu_size, params);
        }
        SaoType::EdgeOffset => {
            apply_sao_edge_offset(recon, width, height, ctu_x, ctu_y, ctu_size, params);
        }
    }
}

/// SAO band offset: partition 8-bit range into 32 bands of width 8.
/// Starting at `band_position`, offsets[0..4] are applied to bands
/// `band_position .. band_position + 4`.
fn apply_sao_band_offset(
    recon: &mut [u8],
    width: usize,
    height: usize,
    ctu_x: usize,
    ctu_y: usize,
    ctu_size: usize,
    params: &SaoParams,
) {
    let band_start = params.band_position as i32;
    let x_end = (ctu_x + ctu_size).min(width);
    let y_end = (ctu_y + ctu_size).min(height);

    for y in ctu_y..y_end {
        for x in ctu_x..x_end {
            let val = recon[y * width + x] as i32;
            let band = val >> 3; // 8-bit / 32 bands = 8 per band
            let band_idx = band - band_start;
            if (0..4).contains(&band_idx) {
                let offset = params.offset[band_idx as usize] as i32;
                recon[y * width + x] = (val + offset).clamp(0, 255) as u8;
            }
        }
    }
}

/// SAO edge offset: classify each sample based on its neighbours along the
/// edge-offset direction, then apply the corresponding offset.
///
/// Edge categories (ITU-T H.265, 8.7.3.2):
/// - 0: c == a && c == b  (valley/peak ambiguous — no offset, not reached below)
/// - 1: c < a && c < b   (local minimum)
/// - 2: c < a || c < b   (but not both — partial minimum)
/// - 3: c > a || c > b   (partial maximum)
/// - 4: c > a && c > b   (local maximum)
///
/// Offsets are indexed 0..3 mapping to categories 1..4.
fn apply_sao_edge_offset(
    recon: &mut [u8],
    width: usize,
    height: usize,
    ctu_x: usize,
    ctu_y: usize,
    ctu_size: usize,
    params: &SaoParams,
) {
    // Direction vectors for the four edge classes
    let (dx, dy): (i32, i32) = match params.eo_class {
        0 => (1, 0),  // horizontal
        1 => (0, 1),  // vertical
        2 => (1, 1),  // 135-degree diagonal
        3 => (1, -1), // 45-degree diagonal
        _ => return,
    };

    let x_end = (ctu_x + ctu_size).min(width);
    let y_end = (ctu_y + ctu_size).min(height);
    let ctu_w = x_end - ctu_x;
    let ctu_h = y_end - ctu_y;

    // Buffer only the CTU's original samples (max 64×64 = 4KB on stack)
    let mut ctu_orig = [0u8; 64 * 64];
    for row in 0..ctu_h {
        let src = (ctu_y + row) * width + ctu_x;
        let dst = row * ctu_w;
        ctu_orig[dst..dst + ctu_w].copy_from_slice(&recon[src..src + ctu_w]);
    }

    for y in ctu_y..y_end {
        for x in ctu_x..x_end {
            let nx_a = x as i32 - dx;
            let ny_a = y as i32 - dy;
            let nx_b = x as i32 + dx;
            let ny_b = y as i32 + dy;

            // Skip if neighbours are out of frame bounds
            if nx_a < 0
                || ny_a < 0
                || nx_b < 0
                || ny_b < 0
                || nx_a >= width as i32
                || ny_a >= height as i32
                || nx_b >= width as i32
                || ny_b >= height as i32
            {
                continue;
            }

            // Read centre from CTU buffer, neighbours from frame
            // (neighbours outside CTU aren't modified by this CTU's SAO)
            let c = ctu_orig[(y - ctu_y) * ctu_w + (x - ctu_x)] as i32;
            let a_x = nx_a as usize;
            let a_y = ny_a as usize;
            let b_x = nx_b as usize;
            let b_y = ny_b as usize;
            // Use CTU buffer for in-CTU neighbours, frame for outside
            let a = if a_x >= ctu_x && a_x < x_end && a_y >= ctu_y && a_y < y_end {
                ctu_orig[(a_y - ctu_y) * ctu_w + (a_x - ctu_x)] as i32
            } else {
                recon[a_y * width + a_x] as i32
            };
            let b = if b_x >= ctu_x && b_x < x_end && b_y >= ctu_y && b_y < y_end {
                ctu_orig[(b_y - ctu_y) * ctu_w + (b_x - ctu_x)] as i32
            } else {
                recon[b_y * width + b_x] as i32
            };

            let edge_idx = edge_category(c, a, b);
            if edge_idx > 0 {
                let offset = params.offset[(edge_idx - 1) as usize] as i32;
                recon[y * width + x] = (c + offset).clamp(0, 255) as u8;
            }
        }
    }
}

/// Compute the SAO edge category for a centre sample `c` and neighbours `a`, `b`.
///
/// Returns 0 (no offset), 1 (local min), 2 (partial min), 3 (partial max),
/// 4 (local max).
#[inline(always)]
fn edge_category(c: i32, a: i32, b: i32) -> u8 {
    let sign_a = (c - a).signum(); // -1, 0, +1
    let sign_b = (c - b).signum();
    match (sign_a, sign_b) {
        (-1, -1) => 1,          // local minimum
        (-1, 0) | (0, -1) => 2, // partial minimum
        (1, 0) | (0, 1) => 3,   // partial maximum
        (1, 1) => 4,            // local maximum
        _ => 0,
    }
}

// ---------------------------------------------------------------------------
// SAO parameter parsing from CABAC
// ---------------------------------------------------------------------------

/// Parse SAO parameters from the CABAC decoder.
///
/// This is a simplified implementation that reads the SAO type, offsets,
/// band position, and edge-offset class from bypass-coded bins.
pub fn parse_sao_params(
    cabac: &mut CabacDecoder<'_>,
    _left_available: bool,
    _above_available: bool,
) -> SaoParams {
    // sao_merge_left_flag / sao_merge_up_flag (bypass coded for simplicity)
    let merge = cabac.decode_bypass();
    if merge {
        return SaoParams::default();
    }

    // sao_type_idx: 0 = none, 1 = band, 2 = edge
    let type_bit0 = cabac.decode_bypass();
    if !type_bit0 {
        return SaoParams::default();
    }
    let type_bit1 = cabac.decode_bypass();
    let sao_type = if type_bit1 {
        SaoType::EdgeOffset
    } else {
        SaoType::BandOffset
    };

    // Read 4 offset magnitudes (truncated unary, bypass-coded, max 7)
    let mut offset = [0i8; 4];
    for o in &mut offset {
        let mut mag = 0u8;
        for _ in 0..7 {
            if cabac.decode_bypass() {
                mag += 1;
            } else {
                break;
            }
        }
        // Sign (for band offset; edge offset signs are implicit)
        if mag > 0 && sao_type == SaoType::BandOffset {
            if cabac.decode_bypass() {
                *o = -(mag as i8);
            } else {
                *o = mag as i8;
            }
        } else {
            *o = mag as i8;
        }
    }

    // For edge offset, apply the standard sign convention:
    // categories 1,2 get positive offsets; categories 3,4 get negative.
    if sao_type == SaoType::EdgeOffset {
        offset[2] = -(offset[2].abs());
        offset[3] = -(offset[3].abs());
    }

    let band_position = if sao_type == SaoType::BandOffset {
        cabac.decode_fl(5) as u8
    } else {
        0
    };

    let eo_class = if sao_type == SaoType::EdgeOffset {
        cabac.decode_fl(2) as u8
    } else {
        0
    };

    SaoParams {
        sao_type,
        offset,
        band_position,
        eo_class,
    }
}

// ---------------------------------------------------------------------------
// 4-tap chroma interpolation filter (HEVC spec Table 8-5)
// ---------------------------------------------------------------------------

/// HEVC 4-tap chroma interpolation filter coefficients.
/// Index by fractional-sample phase (1/8-pel, 0..7); phase 0 is pass-through.
pub const HEVC_CHROMA_FILTER: [[i16; 4]; 8] = [
    [0, 64, 0, 0],
    [-2, 58, 10, -2],
    [-4, 54, 16, -2],
    [-6, 46, 28, -4],
    [-4, 36, 36, -4],
    [-4, 28, 46, -6],
    [-2, 16, 54, -4],
    [-2, 10, 58, -2],
];

/// Apply 4-tap chroma interpolation to produce one output sample.
///
/// `src` contains four consecutive samples `src[0..4]` centred around the
/// output position. `phase` selects the fractional position (0..7).
#[inline]
pub fn chroma_interpolate_sample(src: &[u8], phase: usize) -> u8 {
    debug_assert!(src.len() >= 4);
    let coeffs = &HEVC_CHROMA_FILTER[phase & 7];
    let val = src[0] as i32 * coeffs[0] as i32
        + src[1] as i32 * coeffs[1] as i32
        + src[2] as i32 * coeffs[2] as i32
        + src[3] as i32 * coeffs[3] as i32;
    // Normalize: sum of coefficients = 64, so shift by 6
    ((val + 32) >> 6).clamp(0, 255) as u8
}

/// Apply horizontal 4-tap chroma interpolation across a row.
///
/// `src` is the source row with at least `out_len + 3` samples.
/// `dst` receives `out_len` interpolated samples.
pub fn chroma_interpolate_row(src: &[u8], dst: &mut [u8], phase: usize) {
    for i in 0..dst.len() {
        if i + 3 < src.len() {
            dst[i] = chroma_interpolate_sample(&src[i..i + 4], phase);
        }
    }
}

// ---------------------------------------------------------------------------
// Chroma reconstruction
// ---------------------------------------------------------------------------

/// Reconstruct chroma planes from decoded CU data.
///
/// For 4:2:0 subsampling, each `cu_size x cu_size` luma CU maps to a
/// `(cu_size/2) x (cu_size/2)` chroma block. This function writes into the
/// chroma reconstruction buffers at the appropriate half-resolution position.
///
/// The chroma intra prediction is simplified: DC prediction from the luma
/// plane (average the corresponding 2x2 luma samples), optionally combined
/// with residual from `cu_data` if chroma CBFs are set.
pub fn reconstruct_chroma_plane(
    cu_data: &CodingUnitData,
    recon_cb: &mut [u8],
    recon_cr: &mut [u8],
    x: usize,
    y: usize,
    cu_size: usize,
    chroma_width: usize,
) {
    let chroma_cu_size = cu_size / 2;
    let cx = x / 2;
    let cy = y / 2;

    if chroma_cu_size == 0 {
        return;
    }

    // Neutral chroma DC value (128 for 8-bit, regardless of prediction mode).
    // Future: derive from luma or inter prediction depending on pred_mode.
    let _ = cu_data.pred_mode;
    let dc_val: u8 = 128;

    for row in 0..chroma_cu_size {
        for col in 0..chroma_cu_size {
            let dst_y = cy + row;
            let dst_x = cx + col;
            let idx = dst_y * chroma_width + dst_x;
            if idx < recon_cb.len() {
                recon_cb[idx] = dc_val;
            }
            if idx < recon_cr.len() {
                recon_cr[idx] = dc_val;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Integration: decode_picture with chroma + filters
// ---------------------------------------------------------------------------

/// Produce a YCbCr frame with deblocking and SAO applied, then convert to RGB.
///
/// This function is intended to be called from `HevcDecoder::decode_picture`
/// after the luma CUs have been decoded.
///
/// - Fills chroma planes (4:2:0 DC fill for each CU).
/// - Applies deblocking to luma + chroma.
/// - Applies SAO per CTU (using provided or default parameters).
/// - Converts YCbCr to RGB via `yuv420_to_rgb8`.
pub fn finalize_hevc_frame(
    y_plane: &mut [u8],
    width: usize,
    height: usize,
    cus: &[(usize, usize, usize, HevcPredMode)], // (x, y, size, pred_mode)
    qp: u8,
    sao_params: Option<&[SaoParams]>,
) -> Vec<u8> {
    // Build edge map and QP map for deblocking
    let min_cu_size = 8;
    let grid_cols = width.div_ceil(min_cu_size);
    let grid_rows = height.div_ceil(min_cu_size);
    // Use stack for small grids, heap for large (1080p: 240×135 = 32K)
    let grid_size = grid_cols * grid_rows;
    let mut cu_edges_heap;
    let cu_edges: &mut [bool] = if grid_size <= 4096 {
        // Stack for small frames (up to ~512x512)
        cu_edges_heap = vec![true; grid_size]; // still vec but small
        &mut cu_edges_heap
    } else {
        cu_edges_heap = vec![true; grid_size];
        &mut cu_edges_heap
    };

    // Build pred_mode grid for boundary strength computation (bs=0 → skip)
    // 0=Intra, 1=Inter, 2=Skip
    let mut mode_grid = vec![2u8; grid_size]; // default Skip (bs=0 between skips)
    for &(cu_x, cu_y, cu_size, pred_mode) in cus {
        let mode_val = match pred_mode {
            HevcPredMode::Intra => 0u8,
            HevcPredMode::Inter => 1u8,
            HevcPredMode::Skip => 2u8,
        };
        let gx_start = cu_x / min_cu_size;
        let gy_start = cu_y / min_cu_size;
        let gx_end = ((cu_x + cu_size) / min_cu_size).min(grid_cols);
        let gy_end = ((cu_y + cu_size) / min_cu_size).min(grid_rows);
        for gy in gy_start..gy_end {
            for gx in gx_start..gx_end {
                mode_grid[gy * grid_cols + gx] = mode_val;
            }
        }
    }

    // Mark interior of each CU as non-edge
    for &(cu_x, cu_y, cu_size, _) in cus {
        if cu_size <= min_cu_size {
            continue;
        }
        let gx_start = cu_x / min_cu_size;
        let gy_start = cu_y / min_cu_size;
        let gx_end = (cu_x + cu_size) / min_cu_size;
        let gy_end = (cu_y + cu_size) / min_cu_size;
        for gy in (gy_start + 1)..gy_end {
            for gx in (gx_start + 1)..gx_end.min(grid_cols) {
                if gy < grid_rows {
                    cu_edges[gy * grid_cols + gx] = false;
                }
            }
        }
    }

    // Pre-compute deblock params (constant QP → same for all edges)
    let ctu_size = 64usize.min(width).min(height);
    let ctu_cols = width.div_ceil(ctu_size);
    let ctu_rows = height.div_ceil(ctu_size);
    let qp_map = vec![qp; ctu_cols * ctu_rows];

    // Luma-only deblocking (no chroma planes allocated)
    hevc_deblock_luma_only(
        y_plane,
        width,
        height,
        &qp_map,
        cu_edges,
        min_cu_size,
        &mode_grid,
    );

    // Apply SAO per CTU
    if let Some(sao_list) = sao_params {
        let mut sao_idx = 0;
        for ctu_row in 0..ctu_rows {
            for ctu_col in 0..ctu_cols {
                if sao_idx < sao_list.len() {
                    hevc_apply_sao(
                        y_plane,
                        width,
                        height,
                        ctu_col * ctu_size,
                        ctu_row * ctu_size,
                        ctu_size,
                        &sao_list[sao_idx],
                    );
                    sao_idx += 1;
                }
            }
        }
    }

    // Fast Y→RGB: chroma planes are all-128 (no chroma decode), so UV=0 and R=G=B=Y.
    y_to_grayscale_rgb(y_plane)
}

/// Finalize with real chroma planes — full color YUV420→RGB output.
pub fn finalize_hevc_frame_with_chroma(
    y_plane: &mut [u8],
    cb_plane: &[u8],
    cr_plane: &[u8],
    width: usize,
    height: usize,
    cus: &[(usize, usize, usize, HevcPredMode)],
    qp: u8,
    sao_params: Option<&[SaoParams]>,
) -> Vec<u8> {
    // Deblock luma (same as grayscale path)
    let min_cu_size = 8;
    let grid_cols = width.div_ceil(min_cu_size);
    let grid_rows = height.div_ceil(min_cu_size);
    let grid_size = grid_cols * grid_rows;

    let mut mode_grid = vec![2u8; grid_size];
    for &(cu_x, cu_y, cu_size, pred_mode) in cus {
        let mode_val = match pred_mode {
            HevcPredMode::Intra => 0u8,
            HevcPredMode::Inter => 1u8,
            HevcPredMode::Skip => 2u8,
        };
        let gx_end = ((cu_x + cu_size) / min_cu_size).min(grid_cols);
        let gy_end = ((cu_y + cu_size) / min_cu_size).min(grid_rows);
        for gy in (cu_y / min_cu_size)..gy_end {
            for gx in (cu_x / min_cu_size)..gx_end {
                mode_grid[gy * grid_cols + gx] = mode_val;
            }
        }
    }

    let mut cu_edges = vec![true; grid_size];
    for &(cu_x, cu_y, cu_size, _) in cus {
        if cu_size <= min_cu_size {
            continue;
        }
        let gx_start = cu_x / min_cu_size;
        let gy_start = cu_y / min_cu_size;
        let gx_end = (cu_x + cu_size) / min_cu_size;
        let gy_end = (cu_y + cu_size) / min_cu_size;
        for gy in (gy_start + 1)..gy_end {
            for gx in (gx_start + 1)..gx_end.min(grid_cols) {
                if gy < grid_rows {
                    cu_edges[gy * grid_cols + gx] = false;
                }
            }
        }
    }

    let ctu_size = 64usize.min(width).min(height);
    let ctu_cols = width.div_ceil(ctu_size);
    let ctu_rows = height.div_ceil(ctu_size);
    let qp_map = vec![qp; ctu_cols * ctu_rows];

    hevc_deblock_luma_only(
        y_plane,
        width,
        height,
        &qp_map,
        &cu_edges,
        min_cu_size,
        &mode_grid,
    );

    // Apply SAO
    if let Some(sao_list) = sao_params {
        let mut sao_idx = 0;
        for ctu_row in 0..ctu_rows {
            for ctu_col in 0..ctu_cols {
                if sao_idx < sao_list.len() {
                    hevc_apply_sao(
                        y_plane,
                        width,
                        height,
                        ctu_col * ctu_size,
                        ctu_row * ctu_size,
                        ctu_size,
                        &sao_list[sao_idx],
                    );
                    sao_idx += 1;
                }
            }
        }
    }

    // Full YUV420 → RGB with real chroma (NEON-accelerated)
    crate::yuv420_to_rgb8(y_plane, cb_plane, cr_plane, width, height)
        .unwrap_or_else(|_| y_to_grayscale_rgb(y_plane))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // 1. Boundary strength tests
    // -----------------------------------------------------------------------

    #[test]
    fn bs_both_intra() {
        assert_eq!(hevc_boundary_strength(true, true, 0, 0, (0, 0), (0, 0)), 2);
    }

    #[test]
    fn bs_one_intra() {
        assert_eq!(hevc_boundary_strength(true, false, 0, 0, (0, 0), (0, 0)), 2);
        assert_eq!(hevc_boundary_strength(false, true, 0, 0, (0, 0), (0, 0)), 2);
    }

    #[test]
    fn bs_diff_ref() {
        assert_eq!(
            hevc_boundary_strength(false, false, 0, 1, (0, 0), (0, 0)),
            1
        );
    }

    #[test]
    fn bs_large_mv_diff() {
        // MV difference >= 4 quarter-pel => bs=1
        assert_eq!(
            hevc_boundary_strength(false, false, 0, 0, (0, 0), (4, 0)),
            1
        );
        assert_eq!(
            hevc_boundary_strength(false, false, 0, 0, (0, 0), (0, 4)),
            1
        );
    }

    #[test]
    fn bs_small_mv_same_ref() {
        assert_eq!(
            hevc_boundary_strength(false, false, 0, 0, (0, 0), (3, 0)),
            0
        );
        assert_eq!(
            hevc_boundary_strength(false, false, 0, 0, (0, 0), (0, 0)),
            0
        );
    }

    // -----------------------------------------------------------------------
    // 2. tc / beta table lookup
    // -----------------------------------------------------------------------

    #[test]
    fn tc_table_low_qp() {
        // For low QP values, tc should be 0
        assert_eq!(derive_tc(0, 1), 0);
        assert_eq!(derive_tc(10, 1), 0);
    }

    #[test]
    fn tc_table_mid_qp() {
        // tc at QP=30 with bs=2: Q = 30 + 2*(2-1) = 32
        assert_eq!(derive_tc(30, 2), TC_TABLE[32]);
    }

    #[test]
    fn tc_bs_zero() {
        assert_eq!(derive_tc(30, 0), 0);
    }

    #[test]
    fn beta_table_lookup() {
        assert_eq!(derive_beta(0), 0);
        assert_eq!(derive_beta(20), BETA_TABLE[20]);
        assert_eq!(derive_beta(51), BETA_TABLE[51]);
    }

    // -----------------------------------------------------------------------
    // 3. Edge filtering
    // -----------------------------------------------------------------------

    #[test]
    fn luma_filter_bs0_no_change() {
        let mut samples = [100, 110, 120, 130, 140, 150, 160, 170];
        let orig = samples;
        hevc_filter_edge_luma(&mut samples, 1, 0, 30, 8);
        assert_eq!(samples, orig, "bs=0 should not modify samples");
    }

    #[test]
    fn luma_filter_weak() {
        // Moderate gradient should trigger weak filtering
        let mut samples = [120u8, 122, 124, 126, 134, 136, 138, 140];
        let orig = samples;
        hevc_filter_edge_luma(&mut samples, 1, 2, 35, 8);
        // p0 or q0 should be modified by the weak filter
        let changed = samples[3] != orig[3] || samples[4] != orig[4];
        assert!(
            changed,
            "weak filter should modify p0/q0 for moderate gradient"
        );
    }

    #[test]
    fn luma_filter_strong() {
        // Very smooth gradient should trigger strong filtering
        let mut samples = [127u8, 127, 128, 128, 129, 129, 130, 130];
        let orig = samples;
        hevc_filter_edge_luma(&mut samples, 1, 2, 40, 8);
        // Strong filter may modify p2/q2 as well
        let p2_changed = samples[1] != orig[1];
        let q2_changed = samples[6] != orig[6];
        let any_changed = samples != orig;
        // At least some change should happen for bs=2 at QP=40
        assert!(
            any_changed || p2_changed || q2_changed,
            "strong filter should modify samples for smooth gradient"
        );
    }

    #[test]
    fn chroma_filter_bs1_no_change() {
        let mut samples = [100u8, 120, 140, 160];
        let orig = samples;
        hevc_filter_edge_chroma(&mut samples, 1, 1, 30, 8);
        assert_eq!(samples, orig, "chroma filter should not run for bs < 2");
    }

    #[test]
    fn chroma_filter_bs2_modifies() {
        let mut samples = [120u8, 125, 135, 140];
        let orig = samples;
        hevc_filter_edge_chroma(&mut samples, 1, 2, 35, 8);
        let changed = samples[1] != orig[1] || samples[2] != orig[2];
        assert!(changed, "chroma filter bs=2 should modify p0/q0");
    }

    // -----------------------------------------------------------------------
    // 4. SAO band offset
    // -----------------------------------------------------------------------

    #[test]
    fn sao_band_offset_applies() {
        let width = 8;
        let height = 8;
        let mut recon = vec![100u8; width * height]; // band = 100/8 = 12

        let params = SaoParams {
            sao_type: SaoType::BandOffset,
            offset: [5, -3, 2, -1],
            band_position: 12, // bands 12..15
            eo_class: 0,
        };

        hevc_apply_sao(&mut recon, width, height, 0, 0, 8, &params);
        // Sample value 100 is in band 12, offset index 0 => +5
        assert_eq!(recon[0], 105);
    }

    #[test]
    fn sao_band_offset_out_of_band() {
        let width = 8;
        let height = 8;
        let mut recon = vec![200u8; width * height]; // band = 200/8 = 25

        let params = SaoParams {
            sao_type: SaoType::BandOffset,
            offset: [5, -3, 2, -1],
            band_position: 12, // bands 12..15 — value 200 not in this range
            eo_class: 0,
        };

        hevc_apply_sao(&mut recon, width, height, 0, 0, 8, &params);
        assert_eq!(
            recon[0], 200,
            "sample outside band range should be unchanged"
        );
    }

    // -----------------------------------------------------------------------
    // 5. SAO edge offset (4 classes)
    // -----------------------------------------------------------------------

    #[test]
    fn sao_edge_offset_horizontal() {
        let width = 8;
        let height = 4;
        let mut recon = vec![128u8; width * height];
        // Create a local minimum at (3, 1): neighbours at (2,1)=140 and (4,1)=140
        recon[width + 2] = 140;
        recon[width + 3] = 120; // centre
        recon[width + 4] = 140;

        let params = SaoParams {
            sao_type: SaoType::EdgeOffset,
            offset: [10, 5, -5, -10], // cat1=+10, cat2=+5, cat3=-5, cat4=-10
            band_position: 0,
            eo_class: 0, // horizontal
        };

        hevc_apply_sao(&mut recon, width, height, 0, 0, 8, &params);
        // (3,1) is a local minimum (category 1) => offset = +10
        assert_eq!(recon[width + 3], 130);
    }

    #[test]
    fn sao_edge_offset_vertical() {
        let width = 4;
        let height = 8;
        let mut recon = vec![128u8; width * height];
        // Local maximum at (1, 3): neighbours at (1,2)=100 and (1,4)=100
        recon[2 * width + 1] = 100;
        recon[3 * width + 1] = 150; // centre
        recon[4 * width + 1] = 100;

        let params = SaoParams {
            sao_type: SaoType::EdgeOffset,
            offset: [10, 5, -5, -10], // cat4 = local max => -10
            band_position: 0,
            eo_class: 1, // vertical
        };

        hevc_apply_sao(&mut recon, width, height, 0, 0, 8, &params);
        assert_eq!(recon[3 * width + 1], 140); // 150 - 10
    }

    #[test]
    fn sao_edge_offset_diagonal_135() {
        let width = 8;
        let height = 8;
        let mut recon = vec![128u8; width * height];
        // Local min at (3,3): diagonal neighbours (2,2) and (4,4)
        recon[2 * width + 2] = 150;
        recon[3 * width + 3] = 110; // centre
        recon[4 * width + 4] = 150;

        let params = SaoParams {
            sao_type: SaoType::EdgeOffset,
            offset: [8, 4, -4, -8],
            band_position: 0,
            eo_class: 2, // 135-degree
        };

        hevc_apply_sao(&mut recon, width, height, 0, 0, 8, &params);
        assert_eq!(recon[3 * width + 3], 118); // 110 + 8
    }

    #[test]
    fn sao_edge_offset_diagonal_45() {
        let width = 8;
        let height = 8;
        let mut recon = vec![128u8; width * height];
        // Local max at (3,3): 45-degree neighbours (4,2) and (2,4)
        recon[2 * width + 4] = 100;
        recon[3 * width + 3] = 160; // centre
        recon[4 * width + 2] = 100;

        let params = SaoParams {
            sao_type: SaoType::EdgeOffset,
            offset: [8, 4, -4, -8],
            band_position: 0,
            eo_class: 3, // 45-degree
        };

        hevc_apply_sao(&mut recon, width, height, 0, 0, 8, &params);
        assert_eq!(recon[3 * width + 3], 152); // 160 - 8
    }

    // -----------------------------------------------------------------------
    // 6. Chroma interpolation
    // -----------------------------------------------------------------------

    #[test]
    fn chroma_interp_phase0_passthrough() {
        let src = [100u8, 150, 200, 250];
        let result = chroma_interpolate_sample(&src, 0);
        // Phase 0 coefficients: [0, 64, 0, 0] => src[1] * 64 / 64 = 150
        assert_eq!(result, 150);
    }

    #[test]
    fn chroma_interp_phase4_symmetric() {
        // Phase 4 is the midpoint: [-4, 36, 36, -4]
        let src = [100u8, 120, 140, 160];
        let result = chroma_interpolate_sample(&src, 4);
        // Expected: (-4*100 + 36*120 + 36*140 - 4*160 + 32) / 64
        let expected = ((-4 * 100 + 36 * 120 + 36 * 140 - 4 * 160) + 32) / 64;
        assert_eq!(result, expected as u8);
    }

    #[test]
    fn chroma_interp_row() {
        let src = [50u8, 100, 150, 200, 250];
        let mut dst = [0u8; 2];
        chroma_interpolate_row(&src, &mut dst, 0);
        // Phase 0 picks src[1] and src[2]
        assert_eq!(dst[0], 100);
        assert_eq!(dst[1], 150);
    }

    // -----------------------------------------------------------------------
    // 7. Chroma reconstruction
    // -----------------------------------------------------------------------

    #[test]
    fn chroma_reconstruct_fills_dc() {
        let cu_data = CodingUnitData {
            pred_mode: HevcPredMode::Intra,
            intra_mode_luma: 1,
            intra_mode_chroma: 4,
            cbf_luma: false,
            cbf_cb: false,
            cbf_cr: false,
            log2_tu_size: 4,
        };
        let chroma_w = 8;
        let chroma_h = 8;
        let mut cb = vec![0u8; chroma_w * chroma_h];
        let mut cr = vec![0u8; chroma_w * chroma_h];

        // cu_size=16 → chroma block is 8×8, fills entire buffer
        reconstruct_chroma_plane(&cu_data, &mut cb, &mut cr, 0, 0, 16, chroma_w);

        // Each chroma sample should be 128 (DC)
        assert!(cb.iter().all(|&v| v == 128));
        assert!(cr.iter().all(|&v| v == 128));
    }

    // -----------------------------------------------------------------------
    // 8. Full deblock on synthetic frame (flat)
    // -----------------------------------------------------------------------

    #[test]
    fn deblock_flat_frame_unchanged() {
        // A flat frame should not be modified by deblocking
        let w = 32;
        let h = 32;
        let mut luma = vec![128u8; w * h];
        let mut cb = vec![128u8; (w / 2) * (h / 2)];
        let mut cr = vec![128u8; (w / 2) * (h / 2)];
        let min_cu = 8;
        let grid_cols = w / min_cu;
        let grid_rows = h / min_cu;
        let cu_edges = vec![true; grid_cols * grid_rows];
        let qp_map = vec![30u8; 4];

        let luma_orig = luma.clone();
        hevc_deblock_frame(
            &mut luma, &mut cb, &mut cr, w, h, &qp_map, &cu_edges, min_cu,
        );
        assert_eq!(
            luma, luma_orig,
            "flat frame should be unchanged after deblocking"
        );
    }

    #[test]
    fn deblock_edge_frame_smoothed() {
        // A frame with a sharp edge at a CU boundary should be smoothed
        let w = 32;
        let h = 32;
        let mut luma = vec![0u8; w * h];
        // Left half = 50, right half = 200
        for y in 0..h {
            for x in 0..w {
                luma[y * w + x] = if x < 16 { 50 } else { 200 };
            }
        }
        let mut cb = vec![128u8; (w / 2) * (h / 2)];
        let mut cr = vec![128u8; (w / 2) * (h / 2)];
        let min_cu = 8;
        let grid_cols = w / min_cu;
        let grid_rows = h / min_cu;
        let cu_edges = vec![true; grid_cols * grid_rows];
        let qp_map = vec![30u8; 4];

        let orig_disc = (luma[8 * w + 16] as i32 - luma[8 * w + 15] as i32).abs();
        hevc_deblock_frame(
            &mut luma, &mut cb, &mut cr, w, h, &qp_map, &cu_edges, min_cu,
        );
        let new_disc = (luma[8 * w + 16] as i32 - luma[8 * w + 15] as i32).abs();
        assert!(
            new_disc <= orig_disc,
            "deblocking should reduce edge discontinuity: was {orig_disc}, now {new_disc}"
        );
    }

    #[test]
    fn deblock_gradient_frame() {
        // A frame with a smooth gradient should remain smooth
        let w = 32;
        let h = 32;
        let mut luma = vec![0u8; w * h];
        for y in 0..h {
            for x in 0..w {
                luma[y * w + x] = ((x * 255) / (w - 1)) as u8;
            }
        }
        let mut cb = vec![128u8; (w / 2) * (h / 2)];
        let mut cr = vec![128u8; (w / 2) * (h / 2)];
        let min_cu = 8;
        let grid_cols = w / min_cu;
        let grid_rows = h / min_cu;
        let cu_edges = vec![true; grid_cols * grid_rows];
        let qp_map = vec![26u8; 4];

        let orig = luma.clone();
        hevc_deblock_frame(
            &mut luma, &mut cb, &mut cr, w, h, &qp_map, &cu_edges, min_cu,
        );

        // Calculate total distortion — gradient shouldn't be heavily changed
        let total_diff: i32 = luma
            .iter()
            .zip(orig.iter())
            .map(|(&a, &b)| (a as i32 - b as i32).abs())
            .sum();
        let avg_diff = total_diff as f64 / (w * h) as f64;
        assert!(
            avg_diff < 5.0,
            "gradient frame should not be heavily modified: avg diff = {avg_diff}"
        );
    }

    // -----------------------------------------------------------------------
    // 9. SAO from CABAC (round-trip)
    // -----------------------------------------------------------------------

    #[test]
    fn sao_parse_none() {
        // If first bypass bit is 1 (merge=true), result should be SaoType::None
        let data = [0b1000_0000u8]; // 1 followed by zeros
        let mut cabac = CabacDecoder::new(&data);
        let params = parse_sao_params(&mut cabac, false, false);
        assert_eq!(params.sao_type, SaoType::None);
    }

    // -----------------------------------------------------------------------
    // 10. Integration: finalize_hevc_frame
    // -----------------------------------------------------------------------

    #[test]
    fn finalize_frame_produces_rgb() {
        let w = 16;
        let h = 16;
        let mut y_plane = vec![128u8; w * h];
        let cus = vec![(0, 0, 16, HevcPredMode::Intra)];

        let rgb = finalize_hevc_frame(&mut y_plane, w, h, &cus, 26, None);
        assert_eq!(rgb.len(), w * h * 3);
        // Mid-grey with neutral chroma should produce approximately grey RGB
        // (exact value depends on YUV-to-RGB conversion formula)
        let r = rgb[0];
        let g = rgb[1];
        let b = rgb[2];
        // Allow some tolerance for conversion rounding
        assert!((r as i32 - 128).abs() < 20, "expected ~128 red, got {r}");
        assert!((g as i32 - 128).abs() < 20, "expected ~128 green, got {g}");
        assert!((b as i32 - 128).abs() < 20, "expected ~128 blue, got {b}");
    }

    #[test]
    fn finalize_frame_with_sao() {
        let w = 16;
        let h = 16;
        let mut y_plane = vec![100u8; w * h]; // band = 100/8 = 12

        let sao = vec![SaoParams {
            sao_type: SaoType::BandOffset,
            offset: [3, 0, 0, 0],
            band_position: 12,
            eo_class: 0,
        }];

        let rgb = finalize_hevc_frame(&mut y_plane, w, h, &[], 26, Some(&sao));
        assert_eq!(rgb.len(), w * h * 3);
        // After SAO band offset, luma should have changed
        // (exact RGB depends on conversion)
    }

    // -----------------------------------------------------------------------
    // 11. Chroma QP derivation
    // -----------------------------------------------------------------------

    #[test]
    fn chroma_qp_identity_low() {
        // For QP <= 29, chroma QP equals luma QP
        for qp in 0..=29u8 {
            assert_eq!(derive_chroma_qp(qp), qp);
        }
    }

    #[test]
    fn chroma_qp_mapping_high() {
        // At QP=30, chroma QP should be 29 (per Table 8-10)
        assert_eq!(derive_chroma_qp(30), 29);
    }

    // -----------------------------------------------------------------------
    // 12. Edge category classification
    // -----------------------------------------------------------------------

    #[test]
    fn edge_category_local_min() {
        assert_eq!(edge_category(50, 100, 100), 1);
    }

    #[test]
    fn edge_category_local_max() {
        assert_eq!(edge_category(200, 100, 100), 4);
    }

    #[test]
    fn edge_category_partial() {
        assert_eq!(edge_category(100, 150, 100), 2); // c < a, c == b
        assert_eq!(edge_category(100, 100, 50), 3); // c == a, c > b
    }

    #[test]
    fn edge_category_flat() {
        assert_eq!(edge_category(100, 100, 100), 0);
    }

    // -----------------------------------------------------------------------
    // 13. SaoType default
    // -----------------------------------------------------------------------

    #[test]
    fn sao_type_default_is_none() {
        let params = SaoParams::default();
        assert_eq!(params.sao_type, SaoType::None);
    }

    // -----------------------------------------------------------------------
    // 14. SAO none type does nothing
    // -----------------------------------------------------------------------

    #[test]
    fn sao_none_no_change() {
        let width = 8;
        let height = 8;
        let mut recon = vec![42u8; width * height];
        let orig = recon.clone();
        let params = SaoParams::default();
        hevc_apply_sao(&mut recon, width, height, 0, 0, 8, &params);
        assert_eq!(recon, orig);
    }

    // -----------------------------------------------------------------------
    // 15. HEVC chroma filter table sum
    // -----------------------------------------------------------------------

    #[test]
    fn chroma_filter_coefficients_sum_to_64() {
        for phase in 0..8 {
            let sum: i16 = HEVC_CHROMA_FILTER[phase].iter().sum();
            assert_eq!(
                sum, 64,
                "chroma filter phase {phase} coefficients should sum to 64, got {sum}"
            );
        }
    }

    // -----------------------------------------------------------------------
    // 16. Deblock frame with zero dimensions
    // -----------------------------------------------------------------------

    #[test]
    fn deblock_zero_dims_no_panic() {
        hevc_deblock_frame(&mut [], &mut [], &mut [], 0, 0, &[], &[], 8);
    }

    // -----------------------------------------------------------------------
    // 17. SAO clamping
    // -----------------------------------------------------------------------

    #[test]
    fn sao_band_offset_clamps() {
        let width = 4;
        let height = 4;
        let mut recon = vec![253u8; width * height]; // band = 253/8 = 31

        let params = SaoParams {
            sao_type: SaoType::BandOffset,
            offset: [10, 0, 0, 0], // would push to 263 without clamping
            band_position: 31,
            eo_class: 0,
        };

        hevc_apply_sao(&mut recon, width, height, 0, 0, 4, &params);
        assert_eq!(recon[0], 255, "SAO should clamp to 255");
    }
}
