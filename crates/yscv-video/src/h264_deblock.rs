//! H.264 in-loop deblocking filter.
//!
//! Operates on the reconstructed frame to reduce blocking artifacts at
//! macroblock and 4x4 sub-block boundaries. Implements boundary strength
//! computation and adaptive edge filtering per the H.264/AVC specification.

use crate::h264_motion::MotionVector;

// ---------------------------------------------------------------------------
// Boundary strength
// ---------------------------------------------------------------------------

/// Compute boundary strength (bS) for a pair of adjacent blocks `p` and `q`.
///
/// Returns a value in 0..=4:
/// - 4: either block is intra-coded (strongest filtering)
/// - 2: either block contains coded residual coefficients
/// - 1: motion vectors differ by >= 1 integer sample (4 quarter-pel units)
/// - 0: no filtering needed
pub fn compute_boundary_strength(
    is_intra_p: bool,
    is_intra_q: bool,
    mv_p: MotionVector,
    mv_q: MotionVector,
    has_coded_residual_p: bool,
    has_coded_residual_q: bool,
) -> u8 {
    if is_intra_p || is_intra_q {
        return 4;
    }
    if has_coded_residual_p || has_coded_residual_q {
        return 2;
    }
    let mv_diff = (mv_p.dx - mv_q.dx).unsigned_abs() + (mv_p.dy - mv_q.dy).unsigned_abs();
    if mv_diff >= 4 {
        return 1;
    }
    0
}

// ---------------------------------------------------------------------------
// Threshold derivation
// ---------------------------------------------------------------------------

/// Alpha threshold lookup table indexed by indexA = clamp(QP + offset, 0, 51).
/// Values from Table 8-16 of the H.264 specification.
const ALPHA_TABLE: [i32; 52] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 5, 6, 7, 8, 9, 10, 12, 13, 15, 17, 20,
    22, 25, 28, 32, 36, 40, 45, 50, 56, 63, 71, 80, 90, 101, 113, 127, 144, 162, 182, 203, 226,
    255, 255,
];

/// Beta threshold lookup table indexed by indexB = clamp(QP + offset, 0, 51).
const BETA_TABLE: [i32; 52] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 6, 6, 7, 7, 8, 8,
    9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15, 16, 16, 17, 17, 18, 18,
];

/// Tc0 table indexed by [indexA][bS-1] for bS in 1..=3.
/// From Table 8-17 of the H.264 specification (subset for common QP range).
const TC0_TABLE: [[i32; 3]; 52] = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 1],
    [0, 0, 1],
    [0, 0, 1],
    [0, 0, 1],
    [0, 1, 1],
    [0, 1, 1],
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 2],
    [1, 1, 2],
    [1, 1, 2],
    [1, 1, 2],
    [1, 2, 3],
    [1, 2, 3],
    [2, 2, 3],
    [2, 2, 4],
    [2, 3, 4],
    [2, 3, 4],
    [3, 3, 5],
    [3, 4, 6],
    [3, 4, 6],
    [4, 5, 7],
    [4, 5, 8],
    [4, 6, 9],
    [5, 7, 10],
    [6, 8, 11],
    [6, 8, 13],
    [7, 10, 14],
    [8, 11, 16],
    [9, 12, 18],
    [10, 13, 20],
    [11, 15, 23],
    [13, 17, 25],
];

/// Derive the alpha threshold from a quantization parameter.
fn derive_alpha(qp: u8) -> i32 {
    let idx = (qp as usize).min(51);
    ALPHA_TABLE[idx]
}

/// Derive the beta threshold from a quantization parameter.
fn derive_beta(qp: u8) -> i32 {
    let idx = (qp as usize).min(51);
    BETA_TABLE[idx]
}

/// Derive tc0 from QP and boundary strength (bS in 1..=3).
fn derive_tc0(qp: u8, bs: u8) -> i32 {
    let idx = (qp as usize).min(51);
    if bs == 0 || bs > 3 {
        return 0;
    }
    TC0_TABLE[idx][(bs - 1) as usize]
}

// ---------------------------------------------------------------------------
// Edge filtering
// ---------------------------------------------------------------------------

/// Apply the deblocking filter to a single edge consisting of 4 pixel pairs.
///
/// `pixels` is the row-major frame buffer (single channel). `stride` is the
/// row stride. `offset` is the index of q0 (the first pixel on the "q" side
/// of the boundary). `step` is the distance between successive pixel pairs
/// along the edge (stride for vertical edges, 1 for horizontal edges).
///
/// For a **vertical** edge, p and q are horizontally adjacent:
///   p2 p1 p0 | q0 q1 q2   (step = stride, each pair one row apart)
///
/// For a **horizontal** edge, p and q are vertically adjacent:
///   p2 p1 p0 are in rows above, q0 q1 q2 in rows below.
pub fn deblock_edge_luma(
    pixels: &mut [u8],
    stride: usize,
    offset: usize,
    is_vertical: bool,
    bs: u8,
    alpha: i32,
    beta: i32,
    qp: u8,
) {
    if bs == 0 {
        return;
    }

    let (across, along) = if is_vertical {
        (1usize, stride)
    } else {
        (stride, 1usize)
    };

    let tc = if (1..=3).contains(&bs) {
        derive_tc0(qp, bs) + 1
    } else {
        0
    };
    let alpha_quarter_plus2 = (alpha >> 2) + 2;

    // Single bounds check for the entire 4-pixel group
    let max_idx = offset + 3 * along + 2 * across;
    let min_idx_needed = 3 * across;
    if max_idx >= pixels.len() || offset < min_idx_needed {
        return;
    }

    for i in 0..4 {
        let q0_idx = offset + i * along;
        let p0_idx = q0_idx - across;

        let p0 = pixels[p0_idx] as i32;
        let q0 = pixels[q0_idx] as i32;

        // Quick threshold reject
        let diff_pq = (p0 - q0).abs();
        if diff_pq >= alpha {
            continue;
        }

        let p1 = pixels[q0_idx - 2 * across] as i32;
        let q1 = pixels[q0_idx + across] as i32;

        if (p1 - p0).abs() >= beta || (q1 - q0).abs() >= beta {
            continue;
        }

        if bs == 4 {
            let p2 = pixels[q0_idx - 3 * across] as i32;
            let q2 = pixels[q0_idx + 2 * across] as i32;

            if (p2 - p0).abs() < beta && diff_pq < alpha_quarter_plus2 {
                pixels[p0_idx] = ((p2 + 2 * p1 + 2 * p0 + 2 * q0 + q1 + 4) >> 3) as u8;
                pixels[q0_idx - 2 * across] = ((p2 + p1 + p0 + q0 + 2) >> 2) as u8;
            } else {
                pixels[p0_idx] = ((2 * p1 + p0 + q1 + 2) >> 2) as u8;
            }

            if (q2 - q0).abs() < beta && diff_pq < alpha_quarter_plus2 {
                pixels[q0_idx] = ((q2 + 2 * q1 + 2 * q0 + 2 * p0 + p1 + 4) >> 3) as u8;
                pixels[q0_idx + across] = ((q2 + q1 + q0 + p0 + 2) >> 2) as u8;
            } else {
                pixels[q0_idx] = ((2 * q1 + q0 + p1 + 2) >> 2) as u8;
            }
        } else {
            let delta = ((4 * (q0 - p0) + (p1 - q1) + 4) >> 3).clamp(-tc, tc);
            pixels[p0_idx] = (p0 + delta).clamp(0, 255) as u8;
            pixels[q0_idx] = (q0 - delta).clamp(0, 255) as u8;
        }
    }
}

// ---------------------------------------------------------------------------
// Frame-level deblocking
// ---------------------------------------------------------------------------

/// Filter all macroblock boundaries in a frame.
///
/// Skip-aware deblocking: skips edges where both adjacent MBs are P-skip.
#[allow(unsafe_code)]
pub fn deblock_frame_skip_aware(
    frame: &mut [u8],
    width: usize,
    height: usize,
    stride: usize,
    channels: usize,
    qp: u8,
    mb_is_skip: &[bool],
    mb_w: usize,
) {
    use rayon::prelude::*;

    let alpha = derive_alpha(qp);
    let beta = derive_beta(qp);
    let mb_cols = width / 16;
    let mb_rows = height / 16;
    let bs: u8 = 2;
    let tc = if (1..=3).contains(&bs) {
        derive_tc0(qp, bs) + 1
    } else {
        0
    };

    let plane_size = stride * height;
    for ch in 0..channels {
        let plane_offset = ch * plane_size;
        if plane_offset + plane_size > frame.len() {
            break;
        }

        let plane = &mut frame[plane_offset..plane_offset + plane_size];

        // Vertical edges: parallel over MB rows, skip edges between two skip MBs
        if mb_rows >= 4 {
            plane
                .par_chunks_mut(16 * stride)
                .take(mb_rows)
                .enumerate()
                .for_each(|(mb_row, chunk)| {
                    let chunk_len = chunk.len();
                    for mb_col in 1..mb_cols {
                        // Skip if both left and right MBs are skip
                        let left_skip = mb_is_skip
                            .get(mb_row * mb_w + mb_col - 1)
                            .copied()
                            .unwrap_or(false);
                        let right_skip = mb_is_skip
                            .get(mb_row * mb_w + mb_col)
                            .copied()
                            .unwrap_or(false);
                        if left_skip && right_skip {
                            continue;
                        }
                        let edge_x = mb_col * 16;
                        for row in 0..16 {
                            let q0 = row * stride + edge_x;
                            if q0 < 2 || q0 + 2 > chunk_len {
                                continue;
                            }
                            let p0 = chunk[q0 - 1] as i32;
                            let q0v = chunk[q0] as i32;
                            if (p0 - q0v).abs() >= alpha {
                                continue;
                            }
                            let p1 = chunk[q0 - 2] as i32;
                            let q1 = chunk[q0 + 1] as i32;
                            if (p1 - p0).abs() >= beta || (q1 - q0v).abs() >= beta {
                                continue;
                            }
                            let delta = ((4 * (q0v - p0) + (p1 - q1) + 4) >> 3).clamp(-tc, tc);
                            chunk[q0 - 1] = (p0 + delta).clamp(0, 255) as u8;
                            chunk[q0] = (q0v - delta).clamp(0, 255) as u8;
                        }
                    }
                });
        } else {
            for mb_row in 0..mb_rows {
                for mb_col in 1..mb_cols {
                    let left_skip = mb_is_skip
                        .get(mb_row * mb_w + mb_col - 1)
                        .copied()
                        .unwrap_or(false);
                    let right_skip = mb_is_skip
                        .get(mb_row * mb_w + mb_col)
                        .copied()
                        .unwrap_or(false);
                    if left_skip && right_skip {
                        continue;
                    }
                    let edge_x = mb_col * 16;
                    let base_y = mb_row * 16;
                    for row in 0..16 {
                        let q0 = (base_y + row) * stride + edge_x;
                        if q0 < 2 || q0 + 2 > plane.len() {
                            continue;
                        }
                        let p0 = plane[q0 - 1] as i32;
                        let q0v = plane[q0] as i32;
                        if (p0 - q0v).abs() >= alpha {
                            continue;
                        }
                        let p1 = plane[q0 - 2] as i32;
                        let q1 = plane[q0 + 1] as i32;
                        if (p1 - p0).abs() >= beta || (q1 - q0v).abs() >= beta {
                            continue;
                        }
                        let delta = ((4 * (q0v - p0) + (p1 - q1) + 4) >> 3).clamp(-tc, tc);
                        plane[q0 - 1] = (p0 + delta).clamp(0, 255) as u8;
                        plane[q0] = (q0v - delta).clamp(0, 255) as u8;
                    }
                }
            }
        }

        // Horizontal edges (sequential, skip-aware)
        for mb_row in 1..mb_rows {
            for mb_col in 0..mb_cols {
                let top_skip = mb_is_skip
                    .get((mb_row - 1) * mb_w + mb_col)
                    .copied()
                    .unwrap_or(false);
                let bot_skip = mb_is_skip
                    .get(mb_row * mb_w + mb_col)
                    .copied()
                    .unwrap_or(false);
                if top_skip && bot_skip {
                    continue;
                }
                let edge_y = mb_row * 16;
                let base_x = mb_col * 16;
                for col in 0..16 {
                    let x = base_x + col;
                    let q0 = edge_y * stride + x;
                    if q0 < 3 * stride || q0 + 2 * stride >= plane.len() {
                        continue;
                    }
                    let p0 = plane[q0 - stride] as i32;
                    let q0v = plane[q0] as i32;
                    if (p0 - q0v).abs() >= alpha {
                        continue;
                    }
                    let p1 = plane[q0 - 2 * stride] as i32;
                    let q1 = plane[q0 + stride] as i32;
                    if (p1 - p0).abs() >= beta || (q1 - q0v).abs() >= beta {
                        continue;
                    }
                    let delta = ((4 * (q0v - p0) + (p1 - q1) + 4) >> 3).clamp(-tc, tc);
                    plane[q0 - stride] = (p0 + delta).clamp(0, 255) as u8;
                    plane[q0] = (q0v - delta).clamp(0, 255) as u8;
                }
            }
        }
    }
}

/// Iterates over every macroblock row and column, applying the deblocking
/// filter to both vertical and horizontal edges. The `qp` parameter is the
/// average slice quantization parameter used to derive filter thresholds.
///
/// For multi-channel frames the filter is applied independently to each
/// channel plane (the caller should ensure the frame is in planar or
/// interleaved format; this implementation assumes a single luma plane or
/// operates channel-by-channel).
#[allow(unsafe_code)]
pub fn deblock_frame(frame: &mut [u8], width: usize, height: usize, channels: usize, qp: u8) {
    use rayon::prelude::*;

    let alpha = derive_alpha(qp);
    let beta = derive_beta(qp);
    let mb_cols = width / 16;
    let mb_rows = height / 16;
    let bs: u8 = 2;
    let tc = if (1..=3).contains(&bs) {
        derive_tc0(qp, bs) + 1
    } else {
        0
    };

    let plane_size = width * height;
    for ch in 0..channels {
        let plane_offset = ch * plane_size;
        if plane_offset + plane_size > frame.len() {
            break;
        }

        // Vertical edges: parallelize over MB rows using chunks_mut.
        // Each MB row's vertical edges only touch pixels in rows [mb_row*16..(mb_row+1)*16].
        let plane = &mut frame[plane_offset..plane_offset + plane_size];
        if mb_rows >= 4 {
            plane.par_chunks_mut(16 * width).for_each(|mb_row_chunk| {
                // mb_row_chunk is exactly 16 rows of width pixels
                let chunk_len = mb_row_chunk.len();
                for mb_col in 1..mb_cols {
                    let edge_x = mb_col * 16;
                    for row_in_mb in 0..16 {
                        let q0 = row_in_mb * width + edge_x;
                        if q0 < 2 || q0 + 2 > chunk_len {
                            continue;
                        }
                        let p0 = mb_row_chunk[q0 - 1] as i32;
                        let q0v = mb_row_chunk[q0] as i32;
                        if (p0 - q0v).abs() >= alpha {
                            continue;
                        }
                        let p1 = mb_row_chunk[q0 - 2] as i32;
                        let q1 = mb_row_chunk[q0 + 1] as i32;
                        if (p1 - p0).abs() >= beta || (q1 - q0v).abs() >= beta {
                            continue;
                        }
                        let delta = ((4 * (q0v - p0) + (p1 - q1) + 4) >> 3).clamp(-tc, tc);
                        mb_row_chunk[q0 - 1] = (p0 + delta).clamp(0, 255) as u8;
                        mb_row_chunk[q0] = (q0v - delta).clamp(0, 255) as u8;
                    }
                }
            });
        } else {
            // Small frame: sequential fallback
            for mb_row in 0..mb_rows {
                for mb_col in 1..mb_cols {
                    let edge_x = mb_col * 16;
                    let base_y = mb_row * 16;
                    for row in 0..16 {
                        let y = base_y + row;
                        let q0 = plane_offset + y * width + edge_x;
                        if q0 < 3 || q0 + 2 >= frame.len() {
                            continue;
                        }
                        let p0 = frame[q0 - 1] as i32;
                        let q0v = frame[q0] as i32;
                        if (p0 - q0v).abs() >= alpha {
                            continue;
                        }
                        let p1 = frame[q0 - 2] as i32;
                        let q1 = frame[q0 + 1] as i32;
                        if (p1 - p0).abs() >= beta || (q1 - q0v).abs() >= beta {
                            continue;
                        }
                        let delta = ((4 * (q0v - p0) + (p1 - q1) + 4) >> 3).clamp(-tc, tc);
                        frame[q0 - 1] = (p0 + delta).clamp(0, 255) as u8;
                        frame[q0] = (q0v - delta).clamp(0, 255) as u8;
                    }
                }
            }
        }

        // Horizontal edges: parallelize over MB rows (each row accesses 2 adjacent pixel rows).
        // Horizontal edge at mb_row touches rows [mb_row*16-2..mb_row*16+2], so adjacent
        // MB row edges are 16 rows apart — no overlap if mb_rows > 1.
        // Horizontal edges: sequential (cross MB row boundaries).
        for mb_row in 1..mb_rows {
            for mb_col in 0..mb_cols {
                let edge_y = mb_row * 16;
                let base_x = mb_col * 16;
                for col in 0..16 {
                    let x = base_x + col;
                    let q0 = plane_offset + edge_y * width + x;
                    if q0 < 3 * width || q0 + 2 * width >= frame.len() {
                        continue;
                    }
                    let p0 = frame[q0 - width] as i32;
                    let q0v = frame[q0] as i32;
                    if (p0 - q0v).abs() >= alpha {
                        continue;
                    }
                    let p1 = frame[q0 - 2 * width] as i32;
                    let q1 = frame[q0 + width] as i32;
                    if (p1 - p0).abs() >= beta || (q1 - q0v).abs() >= beta {
                        continue;
                    }
                    let delta = ((4 * (q0v - p0) + (p1 - q1) + 4) >> 3).clamp(-tc, tc);
                    frame[q0 - width] = (p0 + delta).clamp(0, 255) as u8;
                    frame[q0] = (q0v - delta).clamp(0, 255) as u8;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Spec-compliant in-loop deblocking (clause 8.7)
// ---------------------------------------------------------------------------

/// Per-4x4-block reconstruction metadata needed to derive boundary strengths,
/// plus per-macroblock QP. `ref4 == -1` marks an intra (or unavailable) block.
pub struct DeblockInfo<'a> {
    pub nnz_y: &'a [u8],
    pub mvx4: &'a [i16],
    pub mvy4: &'a [i16],
    pub ref4: &'a [i8],
    /// L1 motion (B slices): per-4x4 picture identity and vector. `ref4_l1`
    /// is all `-1` outside B slices, so the boundary strength reduces to the
    /// uni-directional (L0-only) case.
    pub mvx4_l1: &'a [i16],
    pub mvy4_l1: &'a [i16],
    pub ref4_l1: &'a [i8],
    pub grid_w4: usize,
    pub mb_qp: &'a [i32],
    /// Per-macroblock 8x8-transform flag: internal 4-sample luma edges (at
    /// x/y = 4 and 12 within the MB) are not filtered when set (clause 8.7).
    pub tr8x8: &'a [bool],
    pub mb_w: usize,
    pub chroma_qp_index_offset: i32,
    /// FilterOffsetA / FilterOffsetB from the slice header (clause 8.7.2.2):
    /// added to the averaged QP before the alpha+tc0 / beta table lookups.
    pub alpha_c0_offset: i32,
    pub beta_offset: i32,
}

impl DeblockInfo<'_> {
    /// Boundary strength of one block pair (`q` current, `p` neighbour) from the
    /// two-list motion metadata (clause 8.7.2.1). For P/I slices the L1 grid is
    /// all `-1`, reducing this to the single-reference comparison.
    #[inline]
    fn bs_pair(&self, q: usize, p: usize, mb_edge: bool) -> u8 {
        let (rq0, rq1) = (self.ref4[q], self.ref4_l1[q]);
        let (rp0, rp1) = (self.ref4[p], self.ref4_l1[p]);
        // A block with no reference in either list is intra.
        if (rq0 < 0 && rq1 < 0) || (rp0 < 0 && rp1 < 0) {
            return if mb_edge { 4 } else { 3 };
        }
        if self.nnz_y[q] > 0 || self.nnz_y[p] > 0 {
            return 2;
        }
        // Both inter with no coded coefficients: compare reference pictures and
        // motion vectors. `diff` is the "≥ 1 integer sample apart" test.
        let diff = |a: (i32, i32), b: (i32, i32)| (a.0 - b.0).abs() >= 4 || (a.1 - b.1).abs() >= 4;
        // Fast path: both blocks uni-directional from L0 — every P-slice block
        // and most B-slice blocks. Avoids touching the L1 grid and the general
        // set-matching logic below.
        if rq1 < 0 && rp1 < 0 {
            let mvq0 = (self.mvx4[q] as i32, self.mvy4[q] as i32);
            let mvp0 = (self.mvx4[p] as i32, self.mvy4[p] as i32);
            return (rq0 != rp0 || diff(mvq0, mvp0)) as u8;
        }
        let mvq0 = (self.mvx4[q] as i32, self.mvy4[q] as i32);
        let mvq1 = (self.mvx4_l1[q] as i32, self.mvy4_l1[q] as i32);
        let mvp0 = (self.mvx4[p] as i32, self.mvy4[p] as i32);
        let mvp1 = (self.mvx4_l1[p] as i32, self.mvy4_l1[p] as i32);
        let qn = (rq0 >= 0) as u32 + (rq1 >= 0) as u32;
        let pn = (rp0 >= 0) as u32 + (rp1 >= 0) as u32;
        if qn != pn {
            return 1; // different number of motion vectors
        }
        if qn == 1 {
            let (rq, mvq) = if rq0 >= 0 { (rq0, mvq0) } else { (rq1, mvq1) };
            let (rp, mvp) = if rp0 >= 0 { (rp0, mvp0) } else { (rp1, mvp1) };
            return (rq != rp || diff(mvq, mvp)) as u8;
        }
        // Bi-predicted on both sides: the reference-picture *sets* must match.
        let same_set = (rq0 == rp0 && rq1 == rp1) || (rq0 == rp1 && rq1 == rp0);
        if !same_set {
            return 1;
        }
        if rq0 != rq1 {
            // Two distinct pictures: compare vectors for the matching reference.
            let (mp_q0, mp_q1) = if rq0 == rp0 {
                (mvp0, mvp1)
            } else {
                (mvp1, mvp0)
            };
            return (diff(mvq0, mp_q0) || diff(mvq1, mp_q1)) as u8;
        }
        // Same picture in both lists: either pairing may match.
        let pair_a = !diff(mvq0, mvp0) && !diff(mvq1, mvp1);
        let pair_b = !diff(mvq0, mvp1) && !diff(mvq1, mvp0);
        (!(pair_a || pair_b)) as u8
    }

    /// Boundary strengths of all four segments of one *horizontal* edge: the
    /// q blocks are the row at (`bx0`, `gy`), the p blocks the row above.
    #[inline]
    fn bs4_h(&self, bx0: usize, gy: usize, mb_edge: bool) -> [u8; 4] {
        let qo = gy * self.grid_w4 + bx0;
        let po = qo - self.grid_w4;
        let mut out = [0u8; 4];
        for (i, o) in out.iter_mut().enumerate() {
            *o = self.bs_pair(qo + i, po + i, mb_edge);
        }
        out
    }

    /// Boundary strengths of all four segments of one *vertical* edge: the q
    /// blocks are the column at (`qx`, `by0`..), the p blocks one to the left.
    #[inline]
    fn bs4_v(&self, qx: usize, by0: usize, mb_edge: bool) -> [u8; 4] {
        let mut out = [0u8; 4];
        for (i, o) in out.iter_mut().enumerate() {
            let qi = (by0 + i) * self.grid_w4 + qx;
            *o = self.bs_pair(qi, qi - 1, mb_edge);
        }
        out
    }
}

/// Chroma QP from a (already averaged) luma QP, honouring the PPS offset
/// (clause 8.5.8 QPc table).
fn chroma_qp(qp_av: i32, offset: i32) -> i32 {
    const QPC: [i32; 52] = [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
        25, 26, 27, 28, 29, 29, 30, 31, 32, 32, 33, 34, 34, 35, 35, 36, 36, 37, 37, 37, 38, 38, 38,
        39, 39, 39, 39,
    ];
    QPC[(qp_av + offset).clamp(0, 51) as usize]
}

/// Filters one 4-sample luma edge segment in-place (clause 8.7.2.3 / 8.7.2.4).
/// `across` steps from q0 to p0; `along` steps between the four sample pairs.
#[allow(clippy::too_many_arguments)]
fn filter_luma_segment(
    plane: &mut [u8],
    q0_base: usize,
    across: usize,
    along: usize,
    bs: u8,
    alpha: i32,
    beta: i32,
    tc0: i32,
) {
    if bs == 0 || q0_base < 3 * across {
        return;
    }
    let alpha_q = (alpha >> 2) + 2;
    for i in 0..4 {
        let q0i = q0_base + i * along;
        let p0i = q0i - across;
        if q0i + 2 * across >= plane.len() {
            break;
        }
        let p0 = plane[p0i] as i32;
        let q0 = plane[q0i] as i32;
        let p1 = plane[q0i - 2 * across] as i32;
        let q1 = plane[q0i + across] as i32;
        if (p0 - q0).abs() >= alpha || (p1 - p0).abs() >= beta || (q1 - q0).abs() >= beta {
            continue;
        }
        let p2 = plane[q0i - 3 * across] as i32;
        let q2 = plane[q0i + 2 * across] as i32;
        let ap = (p2 - p0).abs() < beta;
        let aq = (q2 - q0).abs() < beta;
        if bs == 4 {
            if ap && (p0 - q0).abs() < alpha_q {
                let p3 = plane[q0i - 4 * across] as i32;
                plane[p0i] = ((p2 + 2 * p1 + 2 * p0 + 2 * q0 + q1 + 4) >> 3) as u8;
                plane[q0i - 2 * across] = ((p2 + p1 + p0 + q0 + 2) >> 2) as u8;
                plane[q0i - 3 * across] = ((2 * p3 + 3 * p2 + p1 + p0 + q0 + 4) >> 3) as u8;
            } else {
                plane[p0i] = ((2 * p1 + p0 + q1 + 2) >> 2) as u8;
            }
            if aq && (p0 - q0).abs() < alpha_q {
                let q3 = plane[q0i + 3 * across] as i32;
                plane[q0i] = ((q2 + 2 * q1 + 2 * q0 + 2 * p0 + p1 + 4) >> 3) as u8;
                plane[q0i + across] = ((q2 + q1 + q0 + p0 + 2) >> 2) as u8;
                plane[q0i + 2 * across] = ((2 * q3 + 3 * q2 + q1 + q0 + p0 + 4) >> 3) as u8;
            } else {
                plane[q0i] = ((2 * q1 + q0 + p1 + 2) >> 2) as u8;
            }
        } else {
            let tc = tc0 + i32::from(ap) + i32::from(aq);
            let delta = (((q0 - p0) * 4 + (p1 - q1) + 4) >> 3).clamp(-tc, tc);
            plane[p0i] = (p0 + delta).clamp(0, 255) as u8;
            plane[q0i] = (q0 - delta).clamp(0, 255) as u8;
            let avg = (p0 + q0 + 1) >> 1;
            if ap {
                let dp = ((p2 + avg - 2 * p1) >> 1).clamp(-tc0, tc0);
                plane[q0i - 2 * across] = (p1 + dp).clamp(0, 255) as u8;
            }
            if aq {
                let dq = ((q2 + avg - 2 * q1) >> 1).clamp(-tc0, tc0);
                plane[q0i + across] = (q1 + dq).clamp(0, 255) as u8;
            }
        }
    }
}

/// Filters one full horizontal luma macroblock edge (16 samples spanning four
/// 4-sample segments) with per-segment boundary strength / tc0 and shared
/// alpha/beta. Dispatches to a vector kernel where the 16 samples are
/// contiguous lanes; the vertical-edge orientation keeps the scalar segment
/// filter (strided sample access).
#[allow(unsafe_code)]
#[allow(clippy::too_many_arguments)]
fn filter_luma_edge_h16(
    plane: &mut [u8],
    q0_base: usize,
    stride: usize,
    bs: &[u8; 4],
    alpha: i32,
    beta: i32,
    tc0: &[i32; 4],
) {
    if alpha == 0 || beta == 0 {
        // A zero threshold never passes the |p0-q0| < alpha / beta tests.
        return;
    }
    // The strong (bS 4) kernel reads p3/q3, so require a four-row apron.
    let in_bounds = q0_base >= 4 * stride && q0_base + 3 * stride + 16 <= plane.len();
    #[cfg(target_arch = "aarch64")]
    if in_bounds && yscv_cpu::host_cpu().features.neon {
        // SAFETY: NEON detected at runtime; the row apron was checked above.
        unsafe {
            deblock_luma_h16_neon(plane, q0_base, stride, bs, alpha, beta, tc0);
        }
        return;
    }
    #[cfg(target_arch = "x86_64")]
    {
        if in_bounds && yscv_cpu::host_cpu().features.avx2 {
            // SAFETY: AVX2 detected at runtime; bounds as above.
            unsafe {
                deblock_luma_h16_avx2(plane, q0_base, stride, bs, alpha, beta, tc0);
            }
            return;
        }
        if in_bounds && yscv_cpu::host_cpu().features.sse2 {
            // SAFETY: SSE2 detected at runtime; bounds as above.
            unsafe {
                deblock_luma_h16_sse2(plane, q0_base, stride, bs, alpha, beta, tc0);
            }
            return;
        }
    }
    for (seg, &b) in bs.iter().enumerate() {
        if b > 0 {
            filter_luma_segment(
                plane,
                q0_base + seg * 4,
                stride,
                1,
                b,
                alpha,
                beta,
                tc0[seg],
            );
        }
    }
}

/// Filters one full vertical luma macroblock edge (16 rows spanning four
/// 4-row segments) with per-segment boundary strength / tc0 and shared
/// alpha/beta. The vector kernels transpose the eight cross-edge columns
/// into lane vectors and reuse the horizontal filter core.
#[allow(unsafe_code)]
#[allow(clippy::too_many_arguments)]
fn filter_luma_edge_v16(
    plane: &mut [u8],
    q0_base: usize,
    stride: usize,
    bs: &[u8; 4],
    alpha: i32,
    beta: i32,
    tc0: &[i32; 4],
) {
    if alpha == 0 || beta == 0 {
        // A zero threshold never passes the |p0-q0| < alpha / beta tests.
        return;
    }
    // The kernels read/write the 8-byte span [q0-4, q0+4) on 16 rows.
    let in_bounds = q0_base >= 4 && q0_base + 15 * stride + 4 <= plane.len();
    #[cfg(target_arch = "aarch64")]
    if in_bounds && yscv_cpu::host_cpu().features.neon {
        // SAFETY: NEON detected at runtime; the column apron was checked above.
        unsafe {
            deblock_luma_v16_neon(plane, q0_base, stride, bs, alpha, beta, tc0);
        }
        return;
    }
    #[cfg(target_arch = "x86_64")]
    {
        if in_bounds && yscv_cpu::host_cpu().features.avx2 {
            // SAFETY: AVX2 detected at runtime; bounds as above.
            unsafe {
                deblock_luma_v16_avx2(plane, q0_base, stride, bs, alpha, beta, tc0);
            }
            return;
        }
        if in_bounds && yscv_cpu::host_cpu().features.sse2 {
            // SAFETY: SSE2 detected at runtime; bounds as above.
            unsafe {
                deblock_luma_v16_sse2(plane, q0_base, stride, bs, alpha, beta, tc0);
            }
            return;
        }
    }
    for (seg, &b) in bs.iter().enumerate() {
        if b > 0 {
            filter_luma_segment(
                plane,
                q0_base + seg * 4 * stride,
                1,
                stride,
                b,
                alpha,
                beta,
                tc0[seg],
            );
        }
    }
}

/// Splits u8 lanes into low/high i16x8 halves (values stay in 0..=255).
#[cfg(target_arch = "aarch64")]
#[inline]
#[allow(unsafe_code)]
unsafe fn wide_s16_neon(
    v: std::arch::aarch64::uint8x16_t,
) -> (std::arch::aarch64::int16x8_t, std::arch::aarch64::int16x8_t) {
    use std::arch::aarch64::*;
    // SAFETY: NEON is mandatory on aarch64.
    unsafe {
        (
            vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(v))),
            vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(v))),
        )
    }
}

/// Splits u8 lanes into low/high u16x8 halves.
#[cfg(target_arch = "aarch64")]
#[inline]
#[allow(unsafe_code)]
unsafe fn wide_u16_neon(
    v: std::arch::aarch64::uint8x16_t,
) -> (
    std::arch::aarch64::uint16x8_t,
    std::arch::aarch64::uint16x8_t,
) {
    use std::arch::aarch64::*;
    // SAFETY: NEON is mandatory on aarch64.
    unsafe { (vmovl_u8(vget_low_u8(v)), vmovl_u8(vget_high_u8(v))) }
}

/// Strong-filter (bS 4) output trio for one side of the edge plus the weak
/// two-tap fallback: e* are the own-side samples (p or q), o* the opposite
/// side. Returns (e0', e1', e2', e0_weak) per clause 8.7.2.4.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
unsafe fn strong_side_neon(
    e3: std::arch::aarch64::uint8x16_t,
    e2: std::arch::aarch64::uint8x16_t,
    e1: std::arch::aarch64::uint8x16_t,
    e0: std::arch::aarch64::uint8x16_t,
    o0: std::arch::aarch64::uint8x16_t,
    o1: std::arch::aarch64::uint8x16_t,
) -> (
    std::arch::aarch64::uint8x16_t,
    std::arch::aarch64::uint8x16_t,
    std::arch::aarch64::uint8x16_t,
    std::arch::aarch64::uint8x16_t,
) {
    use std::arch::aarch64::*;
    let (e3l, e3h) = wide_u16_neon(e3);
    let (e2l, e2h) = wide_u16_neon(e2);
    let (e1l, e1h) = wide_u16_neon(e1);
    let (e0l, e0h) = wide_u16_neon(e0);
    let (o0l, o0h) = wide_u16_neon(o0);
    let (o1l, o1h) = wide_u16_neon(o1);
    // e1' sum: e2 + e1 + e0 + o0 (also reused inside e0'/e2').
    let s1l = vaddq_u16(vaddq_u16(e2l, e1l), vaddq_u16(e0l, o0l));
    let s1h = vaddq_u16(vaddq_u16(e2h, e1h), vaddq_u16(e0h, o0h));
    // e0' sum: e2 + 2*(e1 + e0 + o0) + o1.
    let s0l = vaddq_u16(
        vaddq_u16(e2l, o1l),
        vshlq_n_u16::<1>(vaddq_u16(vaddq_u16(e1l, e0l), o0l)),
    );
    let s0h = vaddq_u16(
        vaddq_u16(e2h, o1h),
        vshlq_n_u16::<1>(vaddq_u16(vaddq_u16(e1h, e0h), o0h)),
    );
    // e2' sum: 2*(e3 + e2) + (e2 + e1 + e0 + o0).
    let s2l = vaddq_u16(vshlq_n_u16::<1>(vaddq_u16(e3l, e2l)), s1l);
    let s2h = vaddq_u16(vshlq_n_u16::<1>(vaddq_u16(e3h, e2h)), s1h);
    // Weak fallback: (2*e1 + e0 + o1 + 2) >> 2.
    let wl = vaddq_u16(vaddq_u16(vshlq_n_u16::<1>(e1l), e0l), o1l);
    let wh = vaddq_u16(vaddq_u16(vshlq_n_u16::<1>(e1h), e0h), o1h);
    (
        vcombine_u8(vrshrn_n_u16::<3>(s0l), vrshrn_n_u16::<3>(s0h)),
        vcombine_u8(vrshrn_n_u16::<2>(s1l), vrshrn_n_u16::<2>(s1h)),
        vcombine_u8(vrshrn_n_u16::<3>(s2l), vrshrn_n_u16::<3>(s2h)),
        vcombine_u8(vrshrn_n_u16::<2>(wl), vrshrn_n_u16::<2>(wh)),
    )
}

/// Normal-filter (bS < 4) p1/q1 update for one side:
/// `e1 + clip((e2 + avg - 2*e1) >> 1, ±tc0)`, unsigned-saturated.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
unsafe fn weak_p1_neon(
    e2: std::arch::aarch64::uint8x16_t,
    e1: std::arch::aarch64::uint8x16_t,
    avg: std::arch::aarch64::uint8x16_t,
    tc0: std::arch::aarch64::uint8x16_t,
) -> std::arch::aarch64::uint8x16_t {
    use std::arch::aarch64::*;
    let (e2l, e2h) = wide_s16_neon(e2);
    let (e1l, e1h) = wide_s16_neon(e1);
    let (avl, avh) = wide_s16_neon(avg);
    let (tl, th) = wide_s16_neon(tc0);
    let dl = vshrq_n_s16::<1>(vsubq_s16(vaddq_s16(e2l, avl), vshlq_n_s16::<1>(e1l)));
    let dh = vshrq_n_s16::<1>(vsubq_s16(vaddq_s16(e2h, avh), vshlq_n_s16::<1>(e1h)));
    let dl = vmaxq_s16(vminq_s16(dl, tl), vnegq_s16(tl));
    let dh = vmaxq_s16(vminq_s16(dh, th), vnegq_s16(th));
    vcombine_u8(
        vqmovun_s16(vaddq_s16(e1l, dl)),
        vqmovun_s16(vaddq_s16(e1h, dh)),
    )
}

/// Shared 16-lane luma filter core: takes the eight cross-edge sample vectors
/// `[p3, p2, p1, p0, q0, q1, q2, q3]` (lane i = position i along the edge,
/// segment i/4) and returns the six potentially modified ones `[p2', p1',
/// p0', q0', q1', q2']`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
unsafe fn luma_core_neon(
    v: &[std::arch::aarch64::uint8x16_t; 8],
    bs: &[u8; 4],
    alpha: i32,
    beta: i32,
    tc0: &[i32; 4],
) -> [std::arch::aarch64::uint8x16_t; 6] {
    use std::arch::aarch64::*;
    let [p3, p2, p1, p0, q0, q1, q2, q3] = *v;
    let av = vdupq_n_u8(alpha as u8);
    let bv = vdupq_n_u8(beta as u8);
    let mut mask = vandq_u8(
        vcltq_u8(vabdq_u8(p0, q0), av),
        vandq_u8(
            vcltq_u8(vabdq_u8(p1, p0), bv),
            vcltq_u8(vabdq_u8(q1, q0), bv),
        ),
    );
    // Per-segment bS>0 mask and tc0, expanded to 4 lanes each.
    let mut bsb = [0u8; 16];
    let mut tcb = [0u8; 16];
    for s in 0..4 {
        for k in 0..4 {
            bsb[s * 4 + k] = if bs[s] > 0 { 0xFF } else { 0 };
            tcb[s * 4 + k] = tc0[s] as u8;
        }
    }
    mask = vandq_u8(mask, vld1q_u8(bsb.as_ptr()));
    let ap = vcltq_u8(vabdq_u8(p2, p0), bv);
    let aq = vcltq_u8(vabdq_u8(q2, q0), bv);

    if bs[0] == 4 {
        // Strong filter: bS 4 is per-macroblock (intra at an MB edge), so the
        // whole edge takes this path.
        let alpha_q = vdupq_n_u8(((alpha >> 2) + 2) as u8);
        let cond = vcltq_u8(vabdq_u8(p0, q0), alpha_q);
        let (p0s, p1s, p2s, p0w) = strong_side_neon(p3, p2, p1, p0, q0, q1);
        let (q0s, q1s, q2s, q0w) = strong_side_neon(q3, q2, q1, q0, p0, p1);
        let sel_p = vandq_u8(ap, cond);
        let sel_q = vandq_u8(aq, cond);
        let smp = vandq_u8(mask, sel_p);
        let smq = vandq_u8(mask, sel_q);
        return [
            vbslq_u8(smp, p2s, p2),
            vbslq_u8(smp, p1s, p1),
            vbslq_u8(mask, vbslq_u8(sel_p, p0s, p0w), p0),
            vbslq_u8(mask, vbslq_u8(sel_q, q0s, q0w), q0),
            vbslq_u8(smq, q1s, q1),
            vbslq_u8(smq, q2s, q2),
        ];
    }

    // Normal filter (bS 1..3): tc = tc0 + ap + aq, delta clamped to ±tc.
    let one = vdupq_n_u8(1);
    let tc0v = vld1q_u8(tcb.as_ptr());
    let tc = vaddq_u8(tc0v, vaddq_u8(vandq_u8(ap, one), vandq_u8(aq, one)));
    let (p0l, p0h) = wide_s16_neon(p0);
    let (q0l, q0h) = wide_s16_neon(q0);
    let (p1l, p1h) = wide_s16_neon(p1);
    let (q1l, q1h) = wide_s16_neon(q1);
    let (tcl, tch) = wide_s16_neon(tc);
    let four = vdupq_n_s16(4);
    let dl = vshrq_n_s16::<3>(vaddq_s16(
        vaddq_s16(vshlq_n_s16::<2>(vsubq_s16(q0l, p0l)), vsubq_s16(p1l, q1l)),
        four,
    ));
    let dh = vshrq_n_s16::<3>(vaddq_s16(
        vaddq_s16(vshlq_n_s16::<2>(vsubq_s16(q0h, p0h)), vsubq_s16(p1h, q1h)),
        four,
    ));
    let dl = vmaxq_s16(vminq_s16(dl, tcl), vnegq_s16(tcl));
    let dh = vmaxq_s16(vminq_s16(dh, tch), vnegq_s16(tch));
    let np0 = vcombine_u8(
        vqmovun_s16(vaddq_s16(p0l, dl)),
        vqmovun_s16(vaddq_s16(p0h, dh)),
    );
    let nq0 = vcombine_u8(
        vqmovun_s16(vsubq_s16(q0l, dl)),
        vqmovun_s16(vsubq_s16(q0h, dh)),
    );
    let avg = vrhaddq_u8(p0, q0);
    let np1 = weak_p1_neon(p2, p1, avg, tc0v);
    let nq1 = weak_p1_neon(q2, q1, avg, tc0v);
    [
        p2,
        vbslq_u8(vandq_u8(mask, ap), np1, p1),
        vbslq_u8(mask, np0, p0),
        vbslq_u8(mask, nq0, q0),
        vbslq_u8(vandq_u8(mask, aq), nq1, q1),
        q2,
    ]
}

/// NEON kernel for one horizontal luma MB edge (16 contiguous samples).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[allow(clippy::too_many_arguments)]
unsafe fn deblock_luma_h16_neon(
    plane: &mut [u8],
    q0_base: usize,
    stride: usize,
    bs: &[u8; 4],
    alpha: i32,
    beta: i32,
    tc0: &[i32; 4],
) {
    use std::arch::aarch64::*;
    debug_assert!(q0_base >= 4 * stride && q0_base + 3 * stride + 16 <= plane.len());
    let base = plane.as_mut_ptr().add(q0_base);
    let v = [
        vld1q_u8(base.sub(4 * stride)),
        vld1q_u8(base.sub(3 * stride)),
        vld1q_u8(base.sub(2 * stride)),
        vld1q_u8(base.sub(stride)),
        vld1q_u8(base),
        vld1q_u8(base.add(stride)),
        vld1q_u8(base.add(2 * stride)),
        vld1q_u8(base.add(3 * stride)),
    ];
    let o = luma_core_neon(&v, bs, alpha, beta, tc0);
    vst1q_u8(base.sub(3 * stride), o[0]);
    vst1q_u8(base.sub(2 * stride), o[1]);
    vst1q_u8(base.sub(stride), o[2]);
    vst1q_u8(base, o[3]);
    vst1q_u8(base.add(stride), o[4]);
    vst1q_u8(base.add(2 * stride), o[5]);
}

/// Transposes an 8x8 byte tile held as eight row vectors.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
unsafe fn transpose_8x8_neon(
    r: [std::arch::aarch64::uint8x8_t; 8],
) -> [std::arch::aarch64::uint8x8_t; 8] {
    use std::arch::aarch64::*;
    let t01 = vtrn_u8(r[0], r[1]);
    let t23 = vtrn_u8(r[2], r[3]);
    let t45 = vtrn_u8(r[4], r[5]);
    let t67 = vtrn_u8(r[6], r[7]);
    let s02 = vtrn_u16(vreinterpret_u16_u8(t01.0), vreinterpret_u16_u8(t23.0));
    let s13 = vtrn_u16(vreinterpret_u16_u8(t01.1), vreinterpret_u16_u8(t23.1));
    let s46 = vtrn_u16(vreinterpret_u16_u8(t45.0), vreinterpret_u16_u8(t67.0));
    let s57 = vtrn_u16(vreinterpret_u16_u8(t45.1), vreinterpret_u16_u8(t67.1));
    let u04 = vtrn_u32(vreinterpret_u32_u16(s02.0), vreinterpret_u32_u16(s46.0));
    let u15 = vtrn_u32(vreinterpret_u32_u16(s13.0), vreinterpret_u32_u16(s57.0));
    let u26 = vtrn_u32(vreinterpret_u32_u16(s02.1), vreinterpret_u32_u16(s46.1));
    let u37 = vtrn_u32(vreinterpret_u32_u16(s13.1), vreinterpret_u32_u16(s57.1));
    [
        vreinterpret_u8_u32(u04.0),
        vreinterpret_u8_u32(u15.0),
        vreinterpret_u8_u32(u26.0),
        vreinterpret_u8_u32(u37.0),
        vreinterpret_u8_u32(u04.1),
        vreinterpret_u8_u32(u15.1),
        vreinterpret_u8_u32(u26.1),
        vreinterpret_u8_u32(u37.1),
    ]
}

/// NEON kernel for one vertical luma MB edge: 16 rows of the eight
/// cross-edge columns are transposed into lane vectors, run through the same
/// filter core as the horizontal edge, and transposed back.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[allow(clippy::too_many_arguments)]
unsafe fn deblock_luma_v16_neon(
    plane: &mut [u8],
    q0_base: usize,
    stride: usize,
    bs: &[u8; 4],
    alpha: i32,
    beta: i32,
    tc0: &[i32; 4],
) {
    use std::arch::aarch64::*;
    debug_assert!(q0_base >= 4 && q0_base + 15 * stride + 4 <= plane.len());
    let base = plane.as_mut_ptr().add(q0_base - 4);
    let row = |i: usize| vld1_u8(base.add(i * stride));
    let top = transpose_8x8_neon([
        row(0),
        row(1),
        row(2),
        row(3),
        row(4),
        row(5),
        row(6),
        row(7),
    ]);
    let bot = transpose_8x8_neon([
        row(8),
        row(9),
        row(10),
        row(11),
        row(12),
        row(13),
        row(14),
        row(15),
    ]);
    let v = [
        vcombine_u8(top[0], bot[0]),
        vcombine_u8(top[1], bot[1]),
        vcombine_u8(top[2], bot[2]),
        vcombine_u8(top[3], bot[3]),
        vcombine_u8(top[4], bot[4]),
        vcombine_u8(top[5], bot[5]),
        vcombine_u8(top[6], bot[6]),
        vcombine_u8(top[7], bot[7]),
    ];
    let o = luma_core_neon(&v, bs, alpha, beta, tc0);
    // p3/q3 are never modified; transpose the full eight columns back and
    // store whole 8-byte rows.
    let full = [v[0], o[0], o[1], o[2], o[3], o[4], o[5], v[7]];
    let rt = transpose_8x8_neon([
        vget_low_u8(full[0]),
        vget_low_u8(full[1]),
        vget_low_u8(full[2]),
        vget_low_u8(full[3]),
        vget_low_u8(full[4]),
        vget_low_u8(full[5]),
        vget_low_u8(full[6]),
        vget_low_u8(full[7]),
    ]);
    let rb = transpose_8x8_neon([
        vget_high_u8(full[0]),
        vget_high_u8(full[1]),
        vget_high_u8(full[2]),
        vget_high_u8(full[3]),
        vget_high_u8(full[4]),
        vget_high_u8(full[5]),
        vget_high_u8(full[6]),
        vget_high_u8(full[7]),
    ]);
    for (i, r) in rt.into_iter().enumerate() {
        vst1_u8(base.add(i * stride), r);
    }
    for (i, r) in rb.into_iter().enumerate() {
        vst1_u8(base.add((8 + i) * stride), r);
    }
}

/// `|a - b|` per u8 lane (SSE2 has no unsigned abs-diff).
#[cfg(target_arch = "x86_64")]
#[inline]
#[allow(unsafe_code)]
unsafe fn abd_u8_sse2(
    a: std::arch::x86_64::__m128i,
    b: std::arch::x86_64::__m128i,
) -> std::arch::x86_64::__m128i {
    use std::arch::x86_64::*;
    // SAFETY: SSE2 is part of the x86_64 baseline.
    unsafe { _mm_or_si128(_mm_subs_epu8(a, b), _mm_subs_epu8(b, a)) }
}

/// `x < t` per u8 lane for `t >= 1`: saturating-subtract t-1, test for zero.
#[cfg(target_arch = "x86_64")]
#[inline]
#[allow(unsafe_code)]
unsafe fn lt_u8_sse2(x: std::arch::x86_64::__m128i, t: i32) -> std::arch::x86_64::__m128i {
    use std::arch::x86_64::*;
    // SAFETY: SSE2 is part of the x86_64 baseline.
    unsafe {
        _mm_cmpeq_epi8(
            _mm_subs_epu8(x, _mm_set1_epi8((t - 1) as i8)),
            _mm_setzero_si128(),
        )
    }
}

/// Lane select: `mask ? a : b`.
#[cfg(target_arch = "x86_64")]
#[inline]
#[allow(unsafe_code)]
unsafe fn bsl_sse2(
    mask: std::arch::x86_64::__m128i,
    a: std::arch::x86_64::__m128i,
    b: std::arch::x86_64::__m128i,
) -> std::arch::x86_64::__m128i {
    use std::arch::x86_64::*;
    // SAFETY: SSE2 is part of the x86_64 baseline.
    unsafe { _mm_or_si128(_mm_and_si128(mask, a), _mm_andnot_si128(mask, b)) }
}

/// Zero-extends the low (`half == 0`) or high u8 lanes to u16.
#[cfg(target_arch = "x86_64")]
#[inline]
#[allow(unsafe_code)]
unsafe fn unpack_half_sse2(
    v: std::arch::x86_64::__m128i,
    half: usize,
) -> std::arch::x86_64::__m128i {
    use std::arch::x86_64::*;
    // SAFETY: SSE2 is part of the x86_64 baseline.
    unsafe {
        let zero = _mm_setzero_si128();
        if half == 0 {
            _mm_unpacklo_epi8(v, zero)
        } else {
            _mm_unpackhi_epi8(v, zero)
        }
    }
}

/// Strong-filter (bS 4) output trio for one side plus the weak two-tap
/// fallback (SSE2 analogue of [`strong_side_neon`]).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
unsafe fn strong_side_sse2(
    e3: std::arch::x86_64::__m128i,
    e2: std::arch::x86_64::__m128i,
    e1: std::arch::x86_64::__m128i,
    e0: std::arch::x86_64::__m128i,
    o0: std::arch::x86_64::__m128i,
    o1: std::arch::x86_64::__m128i,
) -> (
    std::arch::x86_64::__m128i,
    std::arch::x86_64::__m128i,
    std::arch::x86_64::__m128i,
    std::arch::x86_64::__m128i,
) {
    use std::arch::x86_64::*;
    let zero = _mm_setzero_si128();
    let two = _mm_set1_epi16(2);
    let four = _mm_set1_epi16(4);
    let mut out = [zero; 4];
    for half in 0..2 {
        let (e3w, e2w, e1w, e0w_, o0w, o1w) = (
            unpack_half_sse2(e3, half),
            unpack_half_sse2(e2, half),
            unpack_half_sse2(e1, half),
            unpack_half_sse2(e0, half),
            unpack_half_sse2(o0, half),
            unpack_half_sse2(o1, half),
        );
        // e1' sum: e2 + e1 + e0 + o0 (reused in e0'/e2').
        let s1 = _mm_add_epi16(_mm_add_epi16(e2w, e1w), _mm_add_epi16(e0w_, o0w));
        // e0' sum: e2 + 2*(e1 + e0 + o0) + o1.
        let s0 = _mm_add_epi16(
            _mm_add_epi16(e2w, o1w),
            _mm_slli_epi16(_mm_add_epi16(_mm_add_epi16(e1w, e0w_), o0w), 1),
        );
        // e2' sum: 2*(e3 + e2) + s1.
        let s2 = _mm_add_epi16(_mm_slli_epi16(_mm_add_epi16(e3w, e2w), 1), s1);
        // Weak fallback sum: 2*e1 + e0 + o1.
        let sw = _mm_add_epi16(_mm_add_epi16(_mm_slli_epi16(e1w, 1), e0w_), o1w);
        let r = [
            _mm_srli_epi16(_mm_add_epi16(s0, four), 3),
            _mm_srli_epi16(_mm_add_epi16(s1, two), 2),
            _mm_srli_epi16(_mm_add_epi16(s2, four), 3),
            _mm_srli_epi16(_mm_add_epi16(sw, two), 2),
        ];
        for (o, v) in out.iter_mut().zip(r) {
            *o = if half == 0 {
                v
            } else {
                _mm_packus_epi16(*o, v)
            };
        }
    }
    (out[0], out[1], out[2], out[3])
}

/// Shared 16-lane luma filter core (SSE2): takes `[p3, p2, p1, p0, q0, q1,
/// q2, q3]`, returns `[p2', p1', p0', q0', q1', q2']`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
unsafe fn luma_core_sse2(
    v: &[std::arch::x86_64::__m128i; 8],
    bs: &[u8; 4],
    alpha: i32,
    beta: i32,
    tc0: &[i32; 4],
) -> [std::arch::x86_64::__m128i; 6] {
    use std::arch::x86_64::*;
    let [p3, p2, p1, p0, q0, q1, q2, q3] = *v;

    let mut mask = _mm_and_si128(
        lt_u8_sse2(abd_u8_sse2(p0, q0), alpha),
        _mm_and_si128(
            lt_u8_sse2(abd_u8_sse2(p1, p0), beta),
            lt_u8_sse2(abd_u8_sse2(q1, q0), beta),
        ),
    );
    let mut bsb = [0u8; 16];
    let mut tcb = [0u8; 16];
    for seg in 0..4 {
        for k in 0..4 {
            bsb[seg * 4 + k] = if bs[seg] > 0 { 0xFF } else { 0 };
            tcb[seg * 4 + k] = tc0[seg] as u8;
        }
    }
    mask = _mm_and_si128(mask, _mm_loadu_si128(bsb.as_ptr() as *const __m128i));
    let ap = lt_u8_sse2(abd_u8_sse2(p2, p0), beta);
    let aq = lt_u8_sse2(abd_u8_sse2(q2, q0), beta);

    if bs[0] == 4 {
        let cond = lt_u8_sse2(abd_u8_sse2(p0, q0), (alpha >> 2) + 2);
        let (p0s, p1s, p2s, p0w) = strong_side_sse2(p3, p2, p1, p0, q0, q1);
        let (q0s, q1s, q2s, q0w) = strong_side_sse2(q3, q2, q1, q0, p0, p1);
        let sel_p = _mm_and_si128(ap, cond);
        let sel_q = _mm_and_si128(aq, cond);
        let smp = _mm_and_si128(mask, sel_p);
        let smq = _mm_and_si128(mask, sel_q);
        return [
            bsl_sse2(smp, p2s, p2),
            bsl_sse2(smp, p1s, p1),
            bsl_sse2(mask, bsl_sse2(sel_p, p0s, p0w), p0),
            bsl_sse2(mask, bsl_sse2(sel_q, q0s, q0w), q0),
            bsl_sse2(smq, q1s, q1),
            bsl_sse2(smq, q2s, q2),
        ];
    }

    // Normal filter (bS 1..3).
    let zero = _mm_setzero_si128();
    let one = _mm_set1_epi8(1);
    let tc0v = _mm_loadu_si128(tcb.as_ptr() as *const __m128i);
    let tc = _mm_add_epi8(
        tc0v,
        _mm_add_epi8(_mm_and_si128(ap, one), _mm_and_si128(aq, one)),
    );
    let avg = _mm_avg_epu8(p0, q0);
    let mut np0q0 = [zero; 2]; // packed p0', q0'
    let mut np1q1 = [zero; 2]; // packed p1', q1'
    for half in 0..2 {
        let (p0w, q0w, p1w, q1w) = (
            unpack_half_sse2(p0, half),
            unpack_half_sse2(q0, half),
            unpack_half_sse2(p1, half),
            unpack_half_sse2(q1, half),
        );
        let tcw = unpack_half_sse2(tc, half);
        let d = _mm_srai_epi16(
            _mm_add_epi16(
                _mm_add_epi16(
                    _mm_slli_epi16(_mm_sub_epi16(q0w, p0w), 2),
                    _mm_sub_epi16(p1w, q1w),
                ),
                _mm_set1_epi16(4),
            ),
            3,
        );
        let d = _mm_max_epi16(_mm_min_epi16(d, tcw), _mm_sub_epi16(zero, tcw));
        let rp0 = _mm_add_epi16(p0w, d);
        let rq0 = _mm_sub_epi16(q0w, d);
        // p1'/q1': e1 + clip((e2 + avg - 2*e1) >> 1, ±tc0).
        let t0w = unpack_half_sse2(tc0v, half);
        let neg_t0 = _mm_sub_epi16(zero, t0w);
        let avw = unpack_half_sse2(avg, half);
        let (p2w, q2w) = (unpack_half_sse2(p2, half), unpack_half_sse2(q2, half));
        let dp = _mm_srai_epi16(
            _mm_sub_epi16(_mm_add_epi16(p2w, avw), _mm_slli_epi16(p1w, 1)),
            1,
        );
        let dp = _mm_max_epi16(_mm_min_epi16(dp, t0w), neg_t0);
        let dq = _mm_srai_epi16(
            _mm_sub_epi16(_mm_add_epi16(q2w, avw), _mm_slli_epi16(q1w, 1)),
            1,
        );
        let dq = _mm_max_epi16(_mm_min_epi16(dq, t0w), neg_t0);
        let rp1 = _mm_add_epi16(p1w, dp);
        let rq1 = _mm_add_epi16(q1w, dq);
        if half == 0 {
            np0q0 = [rp0, rq0];
            np1q1 = [rp1, rq1];
        } else {
            np0q0 = [
                _mm_packus_epi16(np0q0[0], rp0),
                _mm_packus_epi16(np0q0[1], rq0),
            ];
            np1q1 = [
                _mm_packus_epi16(np1q1[0], rp1),
                _mm_packus_epi16(np1q1[1], rq1),
            ];
        }
    }
    [
        p2,
        bsl_sse2(_mm_and_si128(mask, ap), np1q1[0], p1),
        bsl_sse2(mask, np0q0[0], p0),
        bsl_sse2(mask, np0q0[1], q0),
        bsl_sse2(_mm_and_si128(mask, aq), np1q1[1], q1),
        q2,
    ]
}

/// SSE2 kernel for one horizontal luma MB edge (16 contiguous samples).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[allow(clippy::too_many_arguments)]
unsafe fn deblock_luma_h16_sse2(
    plane: &mut [u8],
    q0_base: usize,
    stride: usize,
    bs: &[u8; 4],
    alpha: i32,
    beta: i32,
    tc0: &[i32; 4],
) {
    use std::arch::x86_64::*;
    debug_assert!(q0_base >= 4 * stride && q0_base + 3 * stride + 16 <= plane.len());
    let base = plane.as_mut_ptr().add(q0_base);
    let s = stride as isize;
    // SAFETY (closures): offsets stay inside the four-row apron checked above.
    let ld = |off: isize| unsafe { _mm_loadu_si128(base.offset(off) as *const __m128i) };
    let st = |off: isize, v| unsafe { _mm_storeu_si128(base.offset(off) as *mut __m128i, v) };
    let v = [
        ld(-4 * s),
        ld(-3 * s),
        ld(-2 * s),
        ld(-s),
        ld(0),
        ld(s),
        ld(2 * s),
        ld(3 * s),
    ];
    let o = luma_core_sse2(&v, bs, alpha, beta, tc0);
    st(-3 * s, o[0]);
    st(-2 * s, o[1]);
    st(-s, o[2]);
    st(0, o[3]);
    st(s, o[4]);
    st(2 * s, o[5]);
}

/// Transposes 16 rows of 8 bytes (in the low half of each xmm) into eight
/// 16-lane column vectors (the punpck ladder).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
unsafe fn transpose_16x8_sse2(
    r: &[std::arch::x86_64::__m128i; 16],
) -> [std::arch::x86_64::__m128i; 8] {
    use std::arch::x86_64::*;
    // Level 1: byte-interleave row pairs → a[i] holds rows 2i/2i+1.
    let mut a = [_mm_setzero_si128(); 8];
    for (i, ai) in a.iter_mut().enumerate() {
        *ai = _mm_unpacklo_epi8(r[2 * i], r[2 * i + 1]);
    }
    // Level 2: word-interleave → w holds 4-row column groups.
    let w = [
        _mm_unpacklo_epi16(a[0], a[1]), // cols 0..3, rows 0..3
        _mm_unpackhi_epi16(a[0], a[1]), // cols 4..7, rows 0..3
        _mm_unpacklo_epi16(a[2], a[3]),
        _mm_unpackhi_epi16(a[2], a[3]),
        _mm_unpacklo_epi16(a[4], a[5]),
        _mm_unpackhi_epi16(a[4], a[5]),
        _mm_unpacklo_epi16(a[6], a[7]),
        _mm_unpackhi_epi16(a[6], a[7]),
    ];
    // Level 3: dword-interleave → d holds 8-row column pairs.
    let d = [
        _mm_unpacklo_epi32(w[0], w[2]), // cols 0,1 rows 0..7
        _mm_unpackhi_epi32(w[0], w[2]), // cols 2,3 rows 0..7
        _mm_unpacklo_epi32(w[1], w[3]), // cols 4,5 rows 0..7
        _mm_unpackhi_epi32(w[1], w[3]), // cols 6,7 rows 0..7
        _mm_unpacklo_epi32(w[4], w[6]), // cols 0,1 rows 8..15
        _mm_unpackhi_epi32(w[4], w[6]),
        _mm_unpacklo_epi32(w[5], w[7]),
        _mm_unpackhi_epi32(w[5], w[7]),
    ];
    // Level 4: qword-interleave → full 16-row columns.
    [
        _mm_unpacklo_epi64(d[0], d[4]),
        _mm_unpackhi_epi64(d[0], d[4]),
        _mm_unpacklo_epi64(d[1], d[5]),
        _mm_unpackhi_epi64(d[1], d[5]),
        _mm_unpacklo_epi64(d[2], d[6]),
        _mm_unpackhi_epi64(d[2], d[6]),
        _mm_unpacklo_epi64(d[3], d[7]),
        _mm_unpackhi_epi64(d[3], d[7]),
    ]
}

/// Transposes an 8x8 byte tile (rows in the low xmm half) — used to turn
/// eight 8-row column vectors back into eight rows.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
unsafe fn transpose_8x8_sse2(
    r: &[std::arch::x86_64::__m128i; 8],
) -> [std::arch::x86_64::__m128i; 8] {
    use std::arch::x86_64::*;
    let a = [
        _mm_unpacklo_epi8(r[0], r[1]),
        _mm_unpacklo_epi8(r[2], r[3]),
        _mm_unpacklo_epi8(r[4], r[5]),
        _mm_unpacklo_epi8(r[6], r[7]),
    ];
    let b = [
        _mm_unpacklo_epi16(a[0], a[1]), // cols 0..3, rows 0..3
        _mm_unpackhi_epi16(a[0], a[1]), // cols 4..7, rows 0..3
        _mm_unpacklo_epi16(a[2], a[3]), // cols 0..3, rows 4..7
        _mm_unpackhi_epi16(a[2], a[3]), // cols 4..7, rows 4..7
    ];
    let c = [
        _mm_unpacklo_epi32(b[0], b[2]), // cols 0,1
        _mm_unpackhi_epi32(b[0], b[2]), // cols 2,3
        _mm_unpacklo_epi32(b[1], b[3]), // cols 4,5
        _mm_unpackhi_epi32(b[1], b[3]), // cols 6,7
    ];
    [
        c[0],
        _mm_srli_si128(c[0], 8),
        c[1],
        _mm_srli_si128(c[1], 8),
        c[2],
        _mm_srli_si128(c[2], 8),
        c[3],
        _mm_srli_si128(c[3], 8),
    ]
}

/// Writes the filtered eight-column tile back as 16 rows of 8 bytes.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
unsafe fn store_v16_rows_sse2(
    full: &[std::arch::x86_64::__m128i; 8],
    base: *mut u8,
    stride: usize,
) {
    use std::arch::x86_64::*;
    let hi = [
        _mm_srli_si128(full[0], 8),
        _mm_srli_si128(full[1], 8),
        _mm_srli_si128(full[2], 8),
        _mm_srli_si128(full[3], 8),
        _mm_srli_si128(full[4], 8),
        _mm_srli_si128(full[5], 8),
        _mm_srli_si128(full[6], 8),
        _mm_srli_si128(full[7], 8),
    ];
    let rt = transpose_8x8_sse2(full);
    let rb = transpose_8x8_sse2(&hi);
    for (i, rowv) in rt.into_iter().enumerate() {
        _mm_storel_epi64(base.add(i * stride) as *mut __m128i, rowv);
    }
    for (i, rowv) in rb.into_iter().enumerate() {
        _mm_storel_epi64(base.add((8 + i) * stride) as *mut __m128i, rowv);
    }
}

/// SSE2 kernel for one vertical luma MB edge (transpose + shared core).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[allow(clippy::too_many_arguments)]
unsafe fn deblock_luma_v16_sse2(
    plane: &mut [u8],
    q0_base: usize,
    stride: usize,
    bs: &[u8; 4],
    alpha: i32,
    beta: i32,
    tc0: &[i32; 4],
) {
    use std::arch::x86_64::*;
    debug_assert!(q0_base >= 4 && q0_base + 15 * stride + 4 <= plane.len());
    let base = plane.as_mut_ptr().add(q0_base - 4);
    let mut rows = [_mm_setzero_si128(); 16];
    for (i, r) in rows.iter_mut().enumerate() {
        *r = _mm_loadl_epi64(base.add(i * stride) as *const __m128i);
    }
    let v = transpose_16x8_sse2(&rows);
    let o = luma_core_sse2(&v, bs, alpha, beta, tc0);
    let full = [v[0], o[0], o[1], o[2], o[3], o[4], o[5], v[7]];
    store_v16_rows_sse2(&full, base, stride);
}

/// Widens 16 u8 lanes to one ymm of u16 lanes.
#[cfg(target_arch = "x86_64")]
#[inline]
#[allow(unsafe_code)]
unsafe fn wide16_avx2(v: std::arch::x86_64::__m128i) -> std::arch::x86_64::__m256i {
    use std::arch::x86_64::*;
    // SAFETY: caller runs under the AVX2 target feature.
    unsafe { _mm256_cvtepu8_epi16(v) }
}

/// Narrows 16 i16 lanes back to u8 with unsigned saturation.
#[cfg(target_arch = "x86_64")]
#[inline]
#[allow(unsafe_code)]
unsafe fn narrow16_avx2(t: std::arch::x86_64::__m256i) -> std::arch::x86_64::__m128i {
    use std::arch::x86_64::*;
    // SAFETY: caller runs under the AVX2 target feature.
    unsafe {
        _mm256_castsi256_si128(_mm256_permute4x64_epi64(
            _mm256_packus_epi16(t, t),
            0b00_00_10_00,
        ))
    }
}

/// Strong-filter (bS 4) output trio for one side plus the weak two-tap
/// fallback (AVX2 analogue of [`strong_side_neon`], 16 u16 lanes per ymm).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
unsafe fn strong_side_avx2(
    e3: std::arch::x86_64::__m128i,
    e2: std::arch::x86_64::__m128i,
    e1: std::arch::x86_64::__m128i,
    e0: std::arch::x86_64::__m128i,
    o0: std::arch::x86_64::__m128i,
    o1: std::arch::x86_64::__m128i,
) -> (
    std::arch::x86_64::__m128i,
    std::arch::x86_64::__m128i,
    std::arch::x86_64::__m128i,
    std::arch::x86_64::__m128i,
) {
    use std::arch::x86_64::*;
    let (e3w, e2w, e1w) = (wide16_avx2(e3), wide16_avx2(e2), wide16_avx2(e1));
    let (e0w, o0w, o1w) = (wide16_avx2(e0), wide16_avx2(o0), wide16_avx2(o1));
    let two = _mm256_set1_epi16(2);
    let four = _mm256_set1_epi16(4);
    let s1 = _mm256_add_epi16(_mm256_add_epi16(e2w, e1w), _mm256_add_epi16(e0w, o0w));
    let s0 = _mm256_add_epi16(
        _mm256_add_epi16(e2w, o1w),
        _mm256_slli_epi16(_mm256_add_epi16(_mm256_add_epi16(e1w, e0w), o0w), 1),
    );
    let s2 = _mm256_add_epi16(_mm256_slli_epi16(_mm256_add_epi16(e3w, e2w), 1), s1);
    let sw = _mm256_add_epi16(_mm256_add_epi16(_mm256_slli_epi16(e1w, 1), e0w), o1w);
    (
        narrow16_avx2(_mm256_srli_epi16(_mm256_add_epi16(s0, four), 3)),
        narrow16_avx2(_mm256_srli_epi16(_mm256_add_epi16(s1, two), 2)),
        narrow16_avx2(_mm256_srli_epi16(_mm256_add_epi16(s2, four), 3)),
        narrow16_avx2(_mm256_srli_epi16(_mm256_add_epi16(sw, two), 2)),
    )
}

/// Shared 16-lane luma filter core (AVX2): the u8-domain masks stay in xmm,
/// the widened i16 arithmetic runs 16 lanes per ymm. Takes `[p3 .. q3]`,
/// returns `[p2' .. q2']`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
unsafe fn luma_core_avx2(
    v: &[std::arch::x86_64::__m128i; 8],
    bs: &[u8; 4],
    alpha: i32,
    beta: i32,
    tc0: &[i32; 4],
) -> [std::arch::x86_64::__m128i; 6] {
    use std::arch::x86_64::*;
    let [p3, p2, p1, p0, q0, q1, q2, q3] = *v;

    let mut mask = _mm_and_si128(
        lt_u8_sse2(abd_u8_sse2(p0, q0), alpha),
        _mm_and_si128(
            lt_u8_sse2(abd_u8_sse2(p1, p0), beta),
            lt_u8_sse2(abd_u8_sse2(q1, q0), beta),
        ),
    );
    let mut bsb = [0u8; 16];
    let mut tcb = [0u8; 16];
    for seg in 0..4 {
        for k in 0..4 {
            bsb[seg * 4 + k] = if bs[seg] > 0 { 0xFF } else { 0 };
            tcb[seg * 4 + k] = tc0[seg] as u8;
        }
    }
    mask = _mm_and_si128(mask, _mm_loadu_si128(bsb.as_ptr() as *const __m128i));
    let ap = lt_u8_sse2(abd_u8_sse2(p2, p0), beta);
    let aq = lt_u8_sse2(abd_u8_sse2(q2, q0), beta);

    if bs[0] == 4 {
        let cond = lt_u8_sse2(abd_u8_sse2(p0, q0), (alpha >> 2) + 2);
        let (p0s, p1s, p2s, p0w) = strong_side_avx2(p3, p2, p1, p0, q0, q1);
        let (q0s, q1s, q2s, q0w) = strong_side_avx2(q3, q2, q1, q0, p0, p1);
        let sel_p = _mm_and_si128(ap, cond);
        let sel_q = _mm_and_si128(aq, cond);
        let smp = _mm_and_si128(mask, sel_p);
        let smq = _mm_and_si128(mask, sel_q);
        return [
            bsl_sse2(smp, p2s, p2),
            bsl_sse2(smp, p1s, p1),
            bsl_sse2(mask, bsl_sse2(sel_p, p0s, p0w), p0),
            bsl_sse2(mask, bsl_sse2(sel_q, q0s, q0w), q0),
            bsl_sse2(smq, q1s, q1),
            bsl_sse2(smq, q2s, q2),
        ];
    }

    // Normal filter (bS 1..3), all 16 widened lanes in single ymm registers.
    let one = _mm_set1_epi8(1);
    let tc0v = _mm_loadu_si128(tcb.as_ptr() as *const __m128i);
    let tc = _mm_add_epi8(
        tc0v,
        _mm_add_epi8(_mm_and_si128(ap, one), _mm_and_si128(aq, one)),
    );
    let (p0w, q0w) = (wide16_avx2(p0), wide16_avx2(q0));
    let (p1w, q1w) = (wide16_avx2(p1), wide16_avx2(q1));
    let tcw = wide16_avx2(tc);
    let zero = _mm256_setzero_si256();
    let d = _mm256_srai_epi16(
        _mm256_add_epi16(
            _mm256_add_epi16(
                _mm256_slli_epi16(_mm256_sub_epi16(q0w, p0w), 2),
                _mm256_sub_epi16(p1w, q1w),
            ),
            _mm256_set1_epi16(4),
        ),
        3,
    );
    let d = _mm256_max_epi16(_mm256_min_epi16(d, tcw), _mm256_sub_epi16(zero, tcw));
    let np0 = narrow16_avx2(_mm256_add_epi16(p0w, d));
    let nq0 = narrow16_avx2(_mm256_sub_epi16(q0w, d));
    // p1'/q1': e1 + clip((e2 + avg - 2*e1) >> 1, ±tc0).
    let avgw = wide16_avx2(_mm_avg_epu8(p0, q0));
    let t0w = wide16_avx2(tc0v);
    let neg_t0 = _mm256_sub_epi16(zero, t0w);
    let (p2w, q2w) = (wide16_avx2(p2), wide16_avx2(q2));
    let dp = _mm256_srai_epi16(
        _mm256_sub_epi16(_mm256_add_epi16(p2w, avgw), _mm256_slli_epi16(p1w, 1)),
        1,
    );
    let dp = _mm256_max_epi16(_mm256_min_epi16(dp, t0w), neg_t0);
    let dq = _mm256_srai_epi16(
        _mm256_sub_epi16(_mm256_add_epi16(q2w, avgw), _mm256_slli_epi16(q1w, 1)),
        1,
    );
    let dq = _mm256_max_epi16(_mm256_min_epi16(dq, t0w), neg_t0);
    let np1 = narrow16_avx2(_mm256_add_epi16(p1w, dp));
    let nq1 = narrow16_avx2(_mm256_add_epi16(q1w, dq));
    [
        p2,
        bsl_sse2(_mm_and_si128(mask, ap), np1, p1),
        bsl_sse2(mask, np0, p0),
        bsl_sse2(mask, nq0, q0),
        bsl_sse2(_mm_and_si128(mask, aq), nq1, q1),
        q2,
    ]
}

/// AVX2 kernel for one horizontal luma MB edge (16 contiguous samples).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[allow(clippy::too_many_arguments)]
unsafe fn deblock_luma_h16_avx2(
    plane: &mut [u8],
    q0_base: usize,
    stride: usize,
    bs: &[u8; 4],
    alpha: i32,
    beta: i32,
    tc0: &[i32; 4],
) {
    use std::arch::x86_64::*;
    debug_assert!(q0_base >= 4 * stride && q0_base + 3 * stride + 16 <= plane.len());
    let base = plane.as_mut_ptr().add(q0_base);
    let s = stride as isize;
    // SAFETY (closures): offsets stay inside the four-row apron checked above.
    let ld = |off: isize| unsafe { _mm_loadu_si128(base.offset(off) as *const __m128i) };
    let st = |off: isize, v| unsafe { _mm_storeu_si128(base.offset(off) as *mut __m128i, v) };
    let v = [
        ld(-4 * s),
        ld(-3 * s),
        ld(-2 * s),
        ld(-s),
        ld(0),
        ld(s),
        ld(2 * s),
        ld(3 * s),
    ];
    let o = luma_core_avx2(&v, bs, alpha, beta, tc0);
    st(-3 * s, o[0]);
    st(-2 * s, o[1]);
    st(-s, o[2]);
    st(0, o[3]);
    st(s, o[4]);
    st(2 * s, o[5]);
}

/// AVX2 kernel for one vertical luma MB edge (SSE2 transpose + AVX2 core).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[allow(clippy::too_many_arguments)]
unsafe fn deblock_luma_v16_avx2(
    plane: &mut [u8],
    q0_base: usize,
    stride: usize,
    bs: &[u8; 4],
    alpha: i32,
    beta: i32,
    tc0: &[i32; 4],
) {
    use std::arch::x86_64::*;
    debug_assert!(q0_base >= 4 && q0_base + 15 * stride + 4 <= plane.len());
    let base = plane.as_mut_ptr().add(q0_base - 4);
    let mut rows = [_mm_setzero_si128(); 16];
    for (i, r) in rows.iter_mut().enumerate() {
        *r = _mm_loadl_epi64(base.add(i * stride) as *const __m128i);
    }
    let v = transpose_16x8_sse2(&rows);
    let o = luma_core_avx2(&v, bs, alpha, beta, tc0);
    let full = [v[0], o[0], o[1], o[2], o[3], o[4], o[5], v[7]];
    store_v16_rows_sse2(&full, base, stride);
}

/// Shared 16-lane chroma filter core (NEON): lanes 0..8 are the U samples,
/// 8..16 the V samples along the edge; per-pair bs/tc0 (`bs[i/2]`). Returns
/// (p0', q0') under the filter mask (clause 8.7.2.3/8.7.2.4 chroma).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[allow(clippy::too_many_arguments)]
unsafe fn chroma_core_neon(
    p1: std::arch::aarch64::uint8x16_t,
    p0: std::arch::aarch64::uint8x16_t,
    q0: std::arch::aarch64::uint8x16_t,
    q1: std::arch::aarch64::uint8x16_t,
    bs: &[u8; 4],
    alpha: i32,
    beta: i32,
    tc0: &[i32; 4],
) -> (
    std::arch::aarch64::uint8x16_t,
    std::arch::aarch64::uint8x16_t,
) {
    use std::arch::aarch64::*;
    let av = vdupq_n_u8(alpha as u8);
    let bv = vdupq_n_u8(beta as u8);
    let mut mask = vandq_u8(
        vcltq_u8(vabdq_u8(p0, q0), av),
        vandq_u8(
            vcltq_u8(vabdq_u8(p1, p0), bv),
            vcltq_u8(vabdq_u8(q1, q0), bv),
        ),
    );
    let mut bsb = [0u8; 16];
    let mut tcb = [0u8; 16];
    for i in 0..8 {
        let s = i / 2;
        bsb[i] = if bs[s] > 0 { 0xFF } else { 0 };
        bsb[8 + i] = bsb[i];
        // Chroma tc = tc0 + 1 (clause 8.7.2.4).
        tcb[i] = (tc0[s] + 1) as u8;
        tcb[8 + i] = tcb[i];
    }
    mask = vandq_u8(mask, vld1q_u8(bsb.as_ptr()));

    if bs[0] == 4 {
        // Strong chroma filter: two-tap averages.
        let (p1l, p1h) = wide_u16_neon(p1);
        let (p0l, p0h) = wide_u16_neon(p0);
        let (q0l, q0h) = wide_u16_neon(q0);
        let (q1l, q1h) = wide_u16_neon(q1);
        let npl = vaddq_u16(vaddq_u16(vshlq_n_u16::<1>(p1l), p0l), q1l);
        let nph = vaddq_u16(vaddq_u16(vshlq_n_u16::<1>(p1h), p0h), q1h);
        let nql = vaddq_u16(vaddq_u16(vshlq_n_u16::<1>(q1l), q0l), p1l);
        let nqh = vaddq_u16(vaddq_u16(vshlq_n_u16::<1>(q1h), q0h), p1h);
        let np0 = vcombine_u8(vrshrn_n_u16::<2>(npl), vrshrn_n_u16::<2>(nph));
        let nq0 = vcombine_u8(vrshrn_n_u16::<2>(nql), vrshrn_n_u16::<2>(nqh));
        return (vbslq_u8(mask, np0, p0), vbslq_u8(mask, nq0, q0));
    }

    // Normal chroma filter: delta clamped to ±(tc0 + 1), only p0/q0 move.
    let tc = vld1q_u8(tcb.as_ptr());
    let (p0l, p0h) = wide_s16_neon(p0);
    let (q0l, q0h) = wide_s16_neon(q0);
    let (p1l, p1h) = wide_s16_neon(p1);
    let (q1l, q1h) = wide_s16_neon(q1);
    let (tcl, tch) = wide_s16_neon(tc);
    let four = vdupq_n_s16(4);
    let dl = vshrq_n_s16::<3>(vaddq_s16(
        vaddq_s16(vshlq_n_s16::<2>(vsubq_s16(q0l, p0l)), vsubq_s16(p1l, q1l)),
        four,
    ));
    let dh = vshrq_n_s16::<3>(vaddq_s16(
        vaddq_s16(vshlq_n_s16::<2>(vsubq_s16(q0h, p0h)), vsubq_s16(p1h, q1h)),
        four,
    ));
    let dl = vmaxq_s16(vminq_s16(dl, tcl), vnegq_s16(tcl));
    let dh = vmaxq_s16(vminq_s16(dh, tch), vnegq_s16(tch));
    let np0 = vcombine_u8(
        vqmovun_s16(vaddq_s16(p0l, dl)),
        vqmovun_s16(vaddq_s16(p0h, dh)),
    );
    let nq0 = vcombine_u8(
        vqmovun_s16(vsubq_s16(q0l, dl)),
        vqmovun_s16(vsubq_s16(q0h, dh)),
    );
    (vbslq_u8(mask, np0, p0), vbslq_u8(mask, nq0, q0))
}

/// NEON kernel for one horizontal chroma MB edge: the 8 U and 8 V samples
/// are filtered together as one 16-lane vector.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[allow(clippy::too_many_arguments)]
unsafe fn deblock_chroma_h8_neon(
    u: &mut [u8],
    v: &mut [u8],
    q0_base: usize,
    stride: usize,
    bs: &[u8; 4],
    alpha: i32,
    beta: i32,
    tc0: &[i32; 4],
) {
    use std::arch::aarch64::*;
    debug_assert!(q0_base >= 2 * stride && q0_base + stride + 8 <= u.len().min(v.len()));
    let ub = u.as_mut_ptr().add(q0_base);
    let vb = v.as_mut_ptr().add(q0_base);
    let p1 = vcombine_u8(vld1_u8(ub.sub(2 * stride)), vld1_u8(vb.sub(2 * stride)));
    let p0 = vcombine_u8(vld1_u8(ub.sub(stride)), vld1_u8(vb.sub(stride)));
    let q0 = vcombine_u8(vld1_u8(ub), vld1_u8(vb));
    let q1 = vcombine_u8(vld1_u8(ub.add(stride)), vld1_u8(vb.add(stride)));
    let (np0, nq0) = chroma_core_neon(p1, p0, q0, q1, bs, alpha, beta, tc0);
    vst1_u8(ub.sub(stride), vget_low_u8(np0));
    vst1_u8(vb.sub(stride), vget_high_u8(np0));
    vst1_u8(ub, vget_low_u8(nq0));
    vst1_u8(vb, vget_high_u8(nq0));
}

/// NEON kernel for one vertical chroma MB edge: each of the 8 rows packs the
/// four U cross-edge samples and the four V ones into one 8-byte lane row,
/// one shared transpose feeds both planes through the 16-lane core.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[allow(clippy::too_many_arguments)]
unsafe fn deblock_chroma_v8_neon(
    u: &mut [u8],
    v: &mut [u8],
    q0_base: usize,
    stride: usize,
    bs: &[u8; 4],
    alpha: i32,
    beta: i32,
    tc0: &[i32; 4],
) {
    use std::arch::aarch64::*;
    debug_assert!(q0_base >= 2 && q0_base + 7 * stride + 2 <= u.len().min(v.len()));
    let ub = u.as_mut_ptr().add(q0_base - 2);
    let vb = v.as_mut_ptr().add(q0_base - 2);
    let row = |i: usize| {
        let uw = (ub.add(i * stride) as *const u32).read_unaligned() as u64;
        let vw = (vb.add(i * stride) as *const u32).read_unaligned() as u64;
        vcreate_u8(uw | (vw << 32))
    };
    let cols = transpose_8x8_neon([
        row(0),
        row(1),
        row(2),
        row(3),
        row(4),
        row(5),
        row(6),
        row(7),
    ]);
    let p1 = vcombine_u8(cols[0], cols[4]);
    let p0 = vcombine_u8(cols[1], cols[5]);
    let q0 = vcombine_u8(cols[2], cols[6]);
    let q1 = vcombine_u8(cols[3], cols[7]);
    let (np0, nq0) = chroma_core_neon(p1, p0, q0, q1, bs, alpha, beta, tc0);
    let back = transpose_8x8_neon([
        cols[0],
        vget_low_u8(np0),
        vget_low_u8(nq0),
        cols[3],
        cols[4],
        vget_high_u8(np0),
        vget_high_u8(nq0),
        cols[7],
    ]);
    for (i, r) in back.into_iter().enumerate() {
        let w = vreinterpret_u32_u8(r);
        (ub.add(i * stride) as *mut u32).write_unaligned(vget_lane_u32::<0>(w));
        (vb.add(i * stride) as *mut u32).write_unaligned(vget_lane_u32::<1>(w));
    }
}

/// Shared 16-lane chroma filter core (SSE2 analogue of [`chroma_core_neon`]).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[allow(clippy::too_many_arguments)]
unsafe fn chroma_core_sse2(
    p1: std::arch::x86_64::__m128i,
    p0: std::arch::x86_64::__m128i,
    q0: std::arch::x86_64::__m128i,
    q1: std::arch::x86_64::__m128i,
    bs: &[u8; 4],
    alpha: i32,
    beta: i32,
    tc0: &[i32; 4],
) -> (std::arch::x86_64::__m128i, std::arch::x86_64::__m128i) {
    use std::arch::x86_64::*;
    let mut mask = _mm_and_si128(
        lt_u8_sse2(abd_u8_sse2(p0, q0), alpha),
        _mm_and_si128(
            lt_u8_sse2(abd_u8_sse2(p1, p0), beta),
            lt_u8_sse2(abd_u8_sse2(q1, q0), beta),
        ),
    );
    let mut bsb = [0u8; 16];
    let mut tcb = [0u8; 16];
    for i in 0..8 {
        let s = i / 2;
        bsb[i] = if bs[s] > 0 { 0xFF } else { 0 };
        bsb[8 + i] = bsb[i];
        // Chroma tc = tc0 + 1 (clause 8.7.2.4).
        tcb[i] = (tc0[s] + 1) as u8;
        tcb[8 + i] = tcb[i];
    }
    mask = _mm_and_si128(mask, _mm_loadu_si128(bsb.as_ptr() as *const __m128i));
    let zero = _mm_setzero_si128();

    if bs[0] == 4 {
        // Strong chroma filter: two-tap averages in u16 halves.
        let two = _mm_set1_epi16(2);
        let mut np = [zero; 2];
        let mut nq = [zero; 2];
        for half in 0..2 {
            let p1w = unpack_half_sse2(p1, half);
            let p0w = unpack_half_sse2(p0, half);
            let q0w = unpack_half_sse2(q0, half);
            let q1w = unpack_half_sse2(q1, half);
            let rp = _mm_srli_epi16(
                _mm_add_epi16(
                    _mm_add_epi16(_mm_add_epi16(_mm_slli_epi16(p1w, 1), p0w), q1w),
                    two,
                ),
                2,
            );
            let rq = _mm_srli_epi16(
                _mm_add_epi16(
                    _mm_add_epi16(_mm_add_epi16(_mm_slli_epi16(q1w, 1), q0w), p1w),
                    two,
                ),
                2,
            );
            if half == 0 {
                np[0] = rp;
                nq[0] = rq;
            } else {
                np[0] = _mm_packus_epi16(np[0], rp);
                nq[0] = _mm_packus_epi16(nq[0], rq);
            }
        }
        return (bsl_sse2(mask, np[0], p0), bsl_sse2(mask, nq[0], q0));
    }

    // Normal chroma filter: delta clamped to ±(tc0 + 1), only p0/q0 move.
    let tc = _mm_loadu_si128(tcb.as_ptr() as *const __m128i);
    let mut np0 = zero;
    let mut nq0 = zero;
    for half in 0..2 {
        let p0w = unpack_half_sse2(p0, half);
        let q0w = unpack_half_sse2(q0, half);
        let p1w = unpack_half_sse2(p1, half);
        let q1w = unpack_half_sse2(q1, half);
        let tcw = unpack_half_sse2(tc, half);
        let d = _mm_srai_epi16(
            _mm_add_epi16(
                _mm_add_epi16(
                    _mm_slli_epi16(_mm_sub_epi16(q0w, p0w), 2),
                    _mm_sub_epi16(p1w, q1w),
                ),
                _mm_set1_epi16(4),
            ),
            3,
        );
        let d = _mm_max_epi16(_mm_min_epi16(d, tcw), _mm_sub_epi16(zero, tcw));
        let rp = _mm_add_epi16(p0w, d);
        let rq = _mm_sub_epi16(q0w, d);
        if half == 0 {
            np0 = rp;
            nq0 = rq;
        } else {
            np0 = _mm_packus_epi16(np0, rp);
            nq0 = _mm_packus_epi16(nq0, rq);
        }
    }
    (bsl_sse2(mask, np0, p0), bsl_sse2(mask, nq0, q0))
}

/// Shared 16-lane chroma filter core (AVX2: widened math in one ymm).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[allow(clippy::too_many_arguments)]
unsafe fn chroma_core_avx2(
    p1: std::arch::x86_64::__m128i,
    p0: std::arch::x86_64::__m128i,
    q0: std::arch::x86_64::__m128i,
    q1: std::arch::x86_64::__m128i,
    bs: &[u8; 4],
    alpha: i32,
    beta: i32,
    tc0: &[i32; 4],
) -> (std::arch::x86_64::__m128i, std::arch::x86_64::__m128i) {
    use std::arch::x86_64::*;
    let mut mask = _mm_and_si128(
        lt_u8_sse2(abd_u8_sse2(p0, q0), alpha),
        _mm_and_si128(
            lt_u8_sse2(abd_u8_sse2(p1, p0), beta),
            lt_u8_sse2(abd_u8_sse2(q1, q0), beta),
        ),
    );
    let mut bsb = [0u8; 16];
    let mut tcb = [0u8; 16];
    for i in 0..8 {
        let s = i / 2;
        bsb[i] = if bs[s] > 0 { 0xFF } else { 0 };
        bsb[8 + i] = bsb[i];
        // Chroma tc = tc0 + 1 (clause 8.7.2.4).
        tcb[i] = (tc0[s] + 1) as u8;
        tcb[8 + i] = tcb[i];
    }
    mask = _mm_and_si128(mask, _mm_loadu_si128(bsb.as_ptr() as *const __m128i));
    let (p1w, p0w) = (wide16_avx2(p1), wide16_avx2(p0));
    let (q0w, q1w) = (wide16_avx2(q0), wide16_avx2(q1));

    if bs[0] == 4 {
        let two = _mm256_set1_epi16(2);
        let np = _mm256_srli_epi16(
            _mm256_add_epi16(
                _mm256_add_epi16(_mm256_add_epi16(_mm256_slli_epi16(p1w, 1), p0w), q1w),
                two,
            ),
            2,
        );
        let nq = _mm256_srli_epi16(
            _mm256_add_epi16(
                _mm256_add_epi16(_mm256_add_epi16(_mm256_slli_epi16(q1w, 1), q0w), p1w),
                two,
            ),
            2,
        );
        return (
            bsl_sse2(mask, narrow16_avx2(np), p0),
            bsl_sse2(mask, narrow16_avx2(nq), q0),
        );
    }

    let zero = _mm256_setzero_si256();
    let tcw = wide16_avx2(_mm_loadu_si128(tcb.as_ptr() as *const __m128i));
    let d = _mm256_srai_epi16(
        _mm256_add_epi16(
            _mm256_add_epi16(
                _mm256_slli_epi16(_mm256_sub_epi16(q0w, p0w), 2),
                _mm256_sub_epi16(p1w, q1w),
            ),
            _mm256_set1_epi16(4),
        ),
        3,
    );
    let d = _mm256_max_epi16(_mm256_min_epi16(d, tcw), _mm256_sub_epi16(zero, tcw));
    (
        bsl_sse2(mask, narrow16_avx2(_mm256_add_epi16(p0w, d)), p0),
        bsl_sse2(mask, narrow16_avx2(_mm256_sub_epi16(q0w, d)), q0),
    )
}

/// Loads the four cross-edge sample vectors of a horizontal chroma edge with
/// U in the low and V in the high 8 lanes.
#[cfg(target_arch = "x86_64")]
#[inline]
#[allow(unsafe_code)]
unsafe fn load_chroma_h8_x86(
    ub: *const u8,
    vb: *const u8,
    stride: usize,
) -> [std::arch::x86_64::__m128i; 4] {
    use std::arch::x86_64::*;
    // SAFETY: caller checked the two-row apron around the edge.
    unsafe {
        let pair = |off: isize| {
            _mm_unpacklo_epi64(
                _mm_loadl_epi64(ub.offset(off) as *const __m128i),
                _mm_loadl_epi64(vb.offset(off) as *const __m128i),
            )
        };
        let s = stride as isize;
        [pair(-2 * s), pair(-s), pair(0), pair(s)]
    }
}

/// SSE2 kernel for one horizontal chroma MB edge (U and V as one vector).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[allow(clippy::too_many_arguments)]
unsafe fn deblock_chroma_h8_sse2(
    u: &mut [u8],
    v: &mut [u8],
    q0_base: usize,
    stride: usize,
    bs: &[u8; 4],
    alpha: i32,
    beta: i32,
    tc0: &[i32; 4],
) {
    use std::arch::x86_64::*;
    debug_assert!(q0_base >= 2 * stride && q0_base + stride + 8 <= u.len().min(v.len()));
    let ub = u.as_mut_ptr().add(q0_base);
    let vb = v.as_mut_ptr().add(q0_base);
    let [p1, p0, q0, q1] = load_chroma_h8_x86(ub, vb, stride);
    let (np0, nq0) = chroma_core_sse2(p1, p0, q0, q1, bs, alpha, beta, tc0);
    _mm_storel_epi64(ub.sub(stride) as *mut __m128i, np0);
    _mm_storel_epi64(vb.sub(stride) as *mut __m128i, _mm_srli_si128(np0, 8));
    _mm_storel_epi64(ub as *mut __m128i, nq0);
    _mm_storel_epi64(vb as *mut __m128i, _mm_srli_si128(nq0, 8));
}

/// AVX2 kernel for one horizontal chroma MB edge.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[allow(clippy::too_many_arguments)]
unsafe fn deblock_chroma_h8_avx2(
    u: &mut [u8],
    v: &mut [u8],
    q0_base: usize,
    stride: usize,
    bs: &[u8; 4],
    alpha: i32,
    beta: i32,
    tc0: &[i32; 4],
) {
    use std::arch::x86_64::*;
    debug_assert!(q0_base >= 2 * stride && q0_base + stride + 8 <= u.len().min(v.len()));
    let ub = u.as_mut_ptr().add(q0_base);
    let vb = v.as_mut_ptr().add(q0_base);
    let [p1, p0, q0, q1] = load_chroma_h8_x86(ub, vb, stride);
    let (np0, nq0) = chroma_core_avx2(p1, p0, q0, q1, bs, alpha, beta, tc0);
    _mm_storel_epi64(ub.sub(stride) as *mut __m128i, np0);
    _mm_storel_epi64(vb.sub(stride) as *mut __m128i, _mm_srli_si128(np0, 8));
    _mm_storel_epi64(ub as *mut __m128i, nq0);
    _mm_storel_epi64(vb as *mut __m128i, _mm_srli_si128(nq0, 8));
}

/// Loads and transposes the 8 combined U+V rows of a vertical chroma edge
/// into the four 16-lane cross-edge vectors plus the raw column set.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
unsafe fn load_chroma_v8_x86(
    ub: *const u8,
    vb: *const u8,
    stride: usize,
) -> [std::arch::x86_64::__m128i; 8] {
    use std::arch::x86_64::*;
    let mut rows = [_mm_setzero_si128(); 8];
    for (i, r) in rows.iter_mut().enumerate() {
        let uw = (ub.add(i * stride) as *const u32).read_unaligned() as u64;
        let vw = (vb.add(i * stride) as *const u32).read_unaligned() as u64;
        *r = _mm_cvtsi64_si128((uw | (vw << 32)) as i64);
    }
    transpose_8x8_sse2(&rows)
}

/// Writes the filtered vertical chroma columns back as 8 combined U+V rows.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
unsafe fn store_chroma_v8_x86(
    cols: &[std::arch::x86_64::__m128i; 8],
    np0: std::arch::x86_64::__m128i,
    nq0: std::arch::x86_64::__m128i,
    ub: *mut u8,
    vb: *mut u8,
    stride: usize,
) {
    use std::arch::x86_64::*;
    let back = transpose_8x8_sse2(&[
        cols[0],
        np0,
        nq0,
        cols[3],
        cols[4],
        _mm_srli_si128(np0, 8),
        _mm_srli_si128(nq0, 8),
        cols[7],
    ]);
    for (i, r) in back.into_iter().enumerate() {
        let uw = _mm_cvtsi128_si32(r) as u32;
        let vw = _mm_cvtsi128_si32(_mm_srli_si128(r, 4)) as u32;
        (ub.add(i * stride) as *mut u32).write_unaligned(uw);
        (vb.add(i * stride) as *mut u32).write_unaligned(vw);
    }
}

/// SSE2 kernel for one vertical chroma MB edge (shared U+V transpose).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[allow(clippy::too_many_arguments)]
unsafe fn deblock_chroma_v8_sse2(
    u: &mut [u8],
    v: &mut [u8],
    q0_base: usize,
    stride: usize,
    bs: &[u8; 4],
    alpha: i32,
    beta: i32,
    tc0: &[i32; 4],
) {
    use std::arch::x86_64::*;
    debug_assert!(q0_base >= 2 && q0_base + 7 * stride + 2 <= u.len().min(v.len()));
    let ub = u.as_mut_ptr().add(q0_base - 2);
    let vb = v.as_mut_ptr().add(q0_base - 2);
    let cols = load_chroma_v8_x86(ub, vb, stride);
    let p1 = _mm_unpacklo_epi64(cols[0], cols[4]);
    let p0 = _mm_unpacklo_epi64(cols[1], cols[5]);
    let q0 = _mm_unpacklo_epi64(cols[2], cols[6]);
    let q1 = _mm_unpacklo_epi64(cols[3], cols[7]);
    let (np0, nq0) = chroma_core_sse2(p1, p0, q0, q1, bs, alpha, beta, tc0);
    store_chroma_v8_x86(&cols, np0, nq0, ub, vb, stride);
}

/// AVX2 kernel for one vertical chroma MB edge.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[allow(clippy::too_many_arguments)]
unsafe fn deblock_chroma_v8_avx2(
    u: &mut [u8],
    v: &mut [u8],
    q0_base: usize,
    stride: usize,
    bs: &[u8; 4],
    alpha: i32,
    beta: i32,
    tc0: &[i32; 4],
) {
    use std::arch::x86_64::*;
    debug_assert!(q0_base >= 2 && q0_base + 7 * stride + 2 <= u.len().min(v.len()));
    let ub = u.as_mut_ptr().add(q0_base - 2);
    let vb = v.as_mut_ptr().add(q0_base - 2);
    let cols = load_chroma_v8_x86(ub, vb, stride);
    let p1 = _mm_unpacklo_epi64(cols[0], cols[4]);
    let p0 = _mm_unpacklo_epi64(cols[1], cols[5]);
    let q0 = _mm_unpacklo_epi64(cols[2], cols[6]);
    let q1 = _mm_unpacklo_epi64(cols[3], cols[7]);
    let (np0, nq0) = chroma_core_avx2(p1, p0, q0, q1, bs, alpha, beta, tc0);
    store_chroma_v8_x86(&cols, np0, nq0, ub, vb, stride);
}

/// Filters one full horizontal chroma MB edge (8 U + 8 V samples) with
/// per-pair boundary strength / tc0 and shared alpha/beta.
#[allow(unsafe_code)]
#[allow(clippy::too_many_arguments)]
fn filter_chroma_edge_h8(
    u: &mut [u8],
    v: &mut [u8],
    q0_base: usize,
    stride: usize,
    bs: &[u8; 4],
    alpha: i32,
    beta: i32,
    tc0: &[i32; 4],
) {
    if alpha == 0 || beta == 0 {
        return;
    }
    let in_bounds = q0_base >= 2 * stride && q0_base + stride + 8 <= u.len().min(v.len());
    #[cfg(target_arch = "aarch64")]
    if in_bounds && yscv_cpu::host_cpu().features.neon {
        // SAFETY: NEON detected at runtime; the row apron was checked above.
        unsafe {
            deblock_chroma_h8_neon(u, v, q0_base, stride, bs, alpha, beta, tc0);
        }
        return;
    }
    #[cfg(target_arch = "x86_64")]
    {
        if in_bounds && yscv_cpu::host_cpu().features.avx2 {
            // SAFETY: AVX2 detected at runtime; bounds as above.
            unsafe {
                deblock_chroma_h8_avx2(u, v, q0_base, stride, bs, alpha, beta, tc0);
            }
            return;
        }
        if in_bounds && yscv_cpu::host_cpu().features.sse2 {
            // SAFETY: SSE2 detected at runtime; bounds as above.
            unsafe {
                deblock_chroma_h8_sse2(u, v, q0_base, stride, bs, alpha, beta, tc0);
            }
            return;
        }
    }
    for cx in 0..8 {
        let b = bs[cx / 2];
        if b > 0 {
            filter_chroma_sample(u, q0_base + cx, stride, b, alpha, beta, tc0[cx / 2]);
            filter_chroma_sample(v, q0_base + cx, stride, b, alpha, beta, tc0[cx / 2]);
        }
    }
}

/// Filters one full vertical chroma MB edge (8 rows, U and V together) with
/// per-pair boundary strength / tc0 and shared alpha/beta.
#[allow(unsafe_code)]
#[allow(clippy::too_many_arguments)]
fn filter_chroma_edge_v8(
    u: &mut [u8],
    v: &mut [u8],
    q0_base: usize,
    stride: usize,
    bs: &[u8; 4],
    alpha: i32,
    beta: i32,
    tc0: &[i32; 4],
) {
    if alpha == 0 || beta == 0 {
        return;
    }
    let in_bounds = q0_base >= 2 && q0_base + 7 * stride + 2 <= u.len().min(v.len());
    #[cfg(target_arch = "aarch64")]
    if in_bounds && yscv_cpu::host_cpu().features.neon {
        // SAFETY: NEON detected at runtime; the column apron was checked above.
        unsafe {
            deblock_chroma_v8_neon(u, v, q0_base, stride, bs, alpha, beta, tc0);
        }
        return;
    }
    #[cfg(target_arch = "x86_64")]
    {
        if in_bounds && yscv_cpu::host_cpu().features.avx2 {
            // SAFETY: AVX2 detected at runtime; bounds as above.
            unsafe {
                deblock_chroma_v8_avx2(u, v, q0_base, stride, bs, alpha, beta, tc0);
            }
            return;
        }
        if in_bounds && yscv_cpu::host_cpu().features.sse2 {
            // SAFETY: SSE2 detected at runtime; bounds as above.
            unsafe {
                deblock_chroma_v8_sse2(u, v, q0_base, stride, bs, alpha, beta, tc0);
            }
            return;
        }
    }
    for cy in 0..8 {
        let b = bs[cy / 2];
        if b > 0 {
            filter_chroma_sample(u, q0_base + cy * stride, 1, b, alpha, beta, tc0[cy / 2]);
            filter_chroma_sample(v, q0_base + cy * stride, 1, b, alpha, beta, tc0[cy / 2]);
        }
    }
}

/// Filters one chroma edge sample pair in-place (clause 8.7.2.3/8.7.2.4 chroma:
/// only p0/q0 change; bS==4 uses the two-tap average).
#[allow(clippy::too_many_arguments)]
fn filter_chroma_sample(
    plane: &mut [u8],
    q0i: usize,
    across: usize,
    bs: u8,
    alpha: i32,
    beta: i32,
    tc0: i32,
) {
    if bs == 0 || q0i < 2 * across || q0i + across >= plane.len() {
        return;
    }
    let p0 = plane[q0i - across] as i32;
    let q0 = plane[q0i] as i32;
    let p1 = plane[q0i - 2 * across] as i32;
    let q1 = plane[q0i + across] as i32;
    if (p0 - q0).abs() >= alpha || (p1 - p0).abs() >= beta || (q1 - q0).abs() >= beta {
        return;
    }
    if bs == 4 {
        plane[q0i - across] = ((2 * p1 + p0 + q1 + 2) >> 2) as u8;
        plane[q0i] = ((2 * q1 + q0 + p1 + 2) >> 2) as u8;
    } else {
        let tc = tc0 + 1;
        let delta = (((q0 - p0) * 4 + (p1 - q1) + 4) >> 3).clamp(-tc, tc);
        plane[q0i - across] = (p0 + delta).clamp(0, 255) as u8;
        plane[q0i] = (q0 - delta).clamp(0, 255) as u8;
    }
}

/// Full-frame spec deblocking (clause 8.7): macroblocks in raster order, each
/// filtering its four vertical edges (left→right) then four horizontal edges
/// (top→bottom), for luma and 4:2:0 chroma, with per-edge boundary strength and
/// per-edge QP averaging. Assumes filter offsets of 0 and a single slice.
///
/// Boundary strengths are derived once per macroblock into two 4x4 tables
/// (vertical / horizontal edges); the chroma edges reuse the co-located luma
/// entries (columns/rows 0 and 2), so `bs` runs 32 times per MB instead of 96.
#[allow(clippy::too_many_arguments)]
/// Filters one macroblock row (all its MBs in raster order). Safe to call as
/// soon as the row *below* is fully decoded: the filter writes only rows up
/// to `mby*16 + 15`, and intra prediction of row `mby + 1` (which reads this
/// row's unfiltered bottom samples) has already run by then.
///
/// The planes are origin-based views of the padded buffers; `stride_y` /
/// `stride_c` are their row pitches.
pub fn deblock_mb_row(
    y: &mut [u8],
    u: &mut [u8],
    v: &mut [u8],
    stride_y: usize,
    stride_c: usize,
    info: &DeblockInfo,
    mby: usize,
) {
    let mb_w = info.mb_w;
    {
        for mbx in 0..mb_w {
            let bx0 = mbx * 4;
            let by0 = mby * 4;
            let qp_cur = info.mb_qp[mby * info.mb_w + mbx];

            // Fast screen: an inter macroblock with no coefficients and one
            // uniform motion vector (P_Skip and friends) has bS 0 on every
            // internal edge, so only the two MB-boundary edges need deriving.
            // Whole grid rows compare at once (4x u8 / 4x i16 per row).
            let uniform = {
                let i0 = by0 * info.grid_w4 + bx0;
                info.ref4[i0] >= 0
                    && (0..4).all(|r| {
                        let row = (by0 + r) * info.grid_w4 + bx0;
                        info.nnz_y[row..row + 4] == [0u8; 4]
                            && info.ref4[row..row + 4] == [info.ref4[i0]; 4]
                            && info.mvx4[row..row + 4] == [info.mvx4[i0]; 4]
                            && info.mvy4[row..row + 4] == [info.mvy4[i0]; 4]
                            // The L1 motion must be uniform too (B slices): a
                            // bi/uni prediction split within the MB produces
                            // internal edges even when L0 is uniform. For P/I the
                            // L1 grid is all `-1`/`0`, so this is always true.
                            && info.ref4_l1[row..row + 4] == [info.ref4_l1[i0]; 4]
                            && info.mvx4_l1[row..row + 4] == [info.mvx4_l1[i0]; 4]
                            && info.mvy4_l1[row..row + 4] == [info.mvy4_l1[i0]; 4]
                    })
            };

            // Boundary-strength tables: bs_v[e][seg] for vertical edge e (q
            // column bx0+e), bs_h[e][seg] for horizontal edge e (q row by0+e).
            let mut bs_v = [[0u8; 4]; 4];
            let mut bs_h = [[0u8; 4]; 4];
            let internal = if uniform { 1 } else { 4 };
            for e in 0..internal {
                if e > 0 || mbx > 0 {
                    bs_v[e] = info.bs4_v(bx0 + e, by0, e == 0);
                }
                if e > 0 || mby > 0 {
                    bs_h[e] = info.bs4_h(bx0, by0 + e, e == 0);
                }
            }

            // Per-edge averaged QP: only the two MB-boundary edges can differ.
            let qp_left = if mbx > 0 {
                (qp_cur + info.mb_qp[mby * info.mb_w + mbx - 1] + 1) >> 1
            } else {
                qp_cur
            };
            let qp_top = if mby > 0 {
                (qp_cur + info.mb_qp[(mby - 1) * info.mb_w + mbx] + 1) >> 1
            } else {
                qp_cur
            };

            // An 8x8-transform MB filters only its 8-aligned luma edges: the
            // internal 4-sample edges (e = 1 and 3) are skipped.
            let tr8x8 = info.tr8x8[mby * info.mb_w + mbx];
            // --- Luma vertical edges (e = 0 is the left MB boundary): the 16
            // edge rows are transposed and filtered as one vector edge.
            for e in 0..4 {
                if (e == 0 && mbx == 0) || (tr8x8 && (e == 1 || e == 3)) || bs_v[e] == [0u8; 4] {
                    continue;
                }
                let qp_av = if e == 0 { qp_left } else { qp_cur };
                let ia = (qp_av + info.alpha_c0_offset).clamp(0, 51) as u8;
                let ib = (qp_av + info.beta_offset).clamp(0, 51) as u8;
                let a = derive_alpha(ia);
                let b = derive_beta(ib);
                let tc0e = [
                    derive_tc0(ia, bs_v[e][0].min(3)),
                    derive_tc0(ia, bs_v[e][1].min(3)),
                    derive_tc0(ia, bs_v[e][2].min(3)),
                    derive_tc0(ia, bs_v[e][3].min(3)),
                ];
                let q0 = (mby * 16) * stride_y + mbx * 16 + e * 4;
                filter_luma_edge_v16(y, q0, stride_y, &bs_v[e], a, b, &tc0e);
            }
            // --- Luma horizontal edges (e = 0 is the top MB boundary):
            // the 16 edge samples are contiguous, filtered as one vector edge.
            for e in 0..4 {
                if (e == 0 && mby == 0) || (tr8x8 && (e == 1 || e == 3)) || bs_h[e] == [0u8; 4] {
                    continue;
                }
                let qp_av = if e == 0 { qp_top } else { qp_cur };
                let ia = (qp_av + info.alpha_c0_offset).clamp(0, 51) as u8;
                let ib = (qp_av + info.beta_offset).clamp(0, 51) as u8;
                let a = derive_alpha(ia);
                let b = derive_beta(ib);
                let tc0e = [
                    derive_tc0(ia, bs_h[e][0].min(3)),
                    derive_tc0(ia, bs_h[e][1].min(3)),
                    derive_tc0(ia, bs_h[e][2].min(3)),
                    derive_tc0(ia, bs_h[e][3].min(3)),
                ];
                let q0 = (mby * 16 + e * 4) * stride_y + mbx * 16;
                filter_luma_edge_h16(y, q0, stride_y, &bs_h[e], a, b, &tc0e);
            }

            // --- Chroma (4:2:0): edges at chroma x/y = 0 and 4 (luma 0 and 8),
            // boundary strengths from the co-located luma edge (column/row 2ce).
            // Chroma edge QP: clause 8.7.2.2 averages the *chroma* QPs of the
            // two macroblocks (the QPc mapping is non-linear, so mapping the
            // averaged luma QP would diverge whenever neighbours differ).
            let qpc_cur = chroma_qp(qp_cur, info.chroma_qp_index_offset);
            let qpc_left = if mbx > 0 {
                let qp_p = info.mb_qp[mby * info.mb_w + mbx - 1];
                (chroma_qp(qp_p, info.chroma_qp_index_offset) + qpc_cur + 1) >> 1
            } else {
                qpc_cur
            };
            let qpc_top = if mby > 0 {
                let qp_p = info.mb_qp[(mby - 1) * info.mb_w + mbx];
                (chroma_qp(qp_p, info.chroma_qp_index_offset) + qpc_cur + 1) >> 1
            } else {
                qpc_cur
            };

            // Both vertical chroma edges first, then both horizontal ones:
            // the horizontal filters must see the vertically-filtered samples.
            // U and V share one vector edge call (8 + 8 lanes).
            for ce in 0..2 {
                if (ce == 0 && mbx == 0) || bs_v[ce * 2] == [0u8; 4] {
                    continue;
                }
                let qpc = if ce == 0 { qpc_left } else { qpc_cur };
                let ia = (qpc + info.alpha_c0_offset).clamp(0, 51) as u8;
                let ib = (qpc + info.beta_offset).clamp(0, 51) as u8;
                let a = derive_alpha(ia);
                let b = derive_beta(ib);
                let tc0e = [
                    derive_tc0(ia, bs_v[ce * 2][0].min(3)),
                    derive_tc0(ia, bs_v[ce * 2][1].min(3)),
                    derive_tc0(ia, bs_v[ce * 2][2].min(3)),
                    derive_tc0(ia, bs_v[ce * 2][3].min(3)),
                ];
                let q0 = (mby * 8) * stride_c + mbx * 8 + ce * 4;
                filter_chroma_edge_v8(u, v, q0, stride_c, &bs_v[ce * 2], a, b, &tc0e);
            }
            for ce in 0..2 {
                if (ce == 0 && mby == 0) || bs_h[ce * 2] == [0u8; 4] {
                    continue;
                }
                let qpc = if ce == 0 { qpc_top } else { qpc_cur };
                let ia = (qpc + info.alpha_c0_offset).clamp(0, 51) as u8;
                let ib = (qpc + info.beta_offset).clamp(0, 51) as u8;
                let a = derive_alpha(ia);
                let b = derive_beta(ib);
                let tc0e = [
                    derive_tc0(ia, bs_h[ce * 2][0].min(3)),
                    derive_tc0(ia, bs_h[ce * 2][1].min(3)),
                    derive_tc0(ia, bs_h[ce * 2][2].min(3)),
                    derive_tc0(ia, bs_h[ce * 2][3].min(3)),
                ];
                let q0 = (mby * 8 + ce * 4) * stride_c + mbx * 8;
                filter_chroma_edge_h8(u, v, q0, stride_c, &bs_h[ce * 2], a, b, &tc0e);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn boundary_strength_intra() {
        let zero_mv = MotionVector::default();

        // Any intra block should yield bs=4.
        assert_eq!(
            compute_boundary_strength(true, false, zero_mv, zero_mv, false, false),
            4
        );
        assert_eq!(
            compute_boundary_strength(false, true, zero_mv, zero_mv, false, false),
            4
        );
        assert_eq!(
            compute_boundary_strength(true, true, zero_mv, zero_mv, true, true),
            4
        );

        // Coded residual but no intra -> bs=2.
        assert_eq!(
            compute_boundary_strength(false, false, zero_mv, zero_mv, true, false),
            2
        );
        assert_eq!(
            compute_boundary_strength(false, false, zero_mv, zero_mv, false, true),
            2
        );

        // Large MV difference, no residual -> bs=1.
        let mv_a = MotionVector {
            dx: 0,
            dy: 0,
            ref_idx: 0,
        };
        let mv_b = MotionVector {
            dx: 4,
            dy: 0,
            ref_idx: 0,
        };
        assert_eq!(
            compute_boundary_strength(false, false, mv_a, mv_b, false, false),
            1
        );

        // Small MV difference -> bs=0.
        let mv_c = MotionVector {
            dx: 1,
            dy: 0,
            ref_idx: 0,
        };
        assert_eq!(
            compute_boundary_strength(false, false, mv_a, mv_c, false, false),
            0
        );
    }

    #[test]
    fn deblock_edge_reduces_discontinuity() {
        // Create an artificial block boundary in a 32-pixel wide, single-row-
        // equivalent buffer. Left half = 50, right half = 200.
        let width = 32;
        let height = 8;
        let mut pixels = vec![0u8; width * height];
        for row in 0..height {
            for col in 0..width {
                pixels[row * width + col] = if col < 16 { 50 } else { 200 };
            }
        }

        // Record the original discontinuity at the boundary (col 15 vs 16).
        let orig_disc: i32 = (pixels[16] as i32 - pixels[15] as i32).abs();

        // Apply filtering at the vertical edge at column 16 for 4 rows.
        let alpha = 40;
        let beta = 20;
        let q0_offset = 3 * width + 16; // row 3 so we have p2..q2 room
        deblock_edge_luma(&mut pixels, width, q0_offset, true, 4, alpha, beta, 26);

        // After filtering, the discontinuity at the boundary should be
        // reduced (or at least not increased).
        let new_disc: i32 = (pixels[3 * width + 16] as i32 - pixels[3 * width + 15] as i32).abs();
        assert!(
            new_disc <= orig_disc,
            "deblocking should reduce discontinuity: was {orig_disc}, now {new_disc}"
        );
    }
}
