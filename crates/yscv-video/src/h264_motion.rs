//! H.264 P-slice motion compensation: motion vector parsing, prediction,
//! reference frame buffering, and inter-frame block copy.

use crate::{BitstreamReader, VideoError};

// ---------------------------------------------------------------------------
// Motion vector
// ---------------------------------------------------------------------------

/// Motion vector for a macroblock partition.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MotionVector {
    pub dx: i16,
    pub dy: i16,
    pub ref_idx: usize,
}

/// Parse motion vector difference from bitstream (Exp-Golomb coded).
pub fn parse_mvd(reader: &mut BitstreamReader) -> Result<(i16, i16), VideoError> {
    let mvd_x = reader.read_se()?;
    let mvd_y = reader.read_se()?;
    Ok((mvd_x as i16, mvd_y as i16))
}

/// Predict motion vector from neighboring blocks (median prediction).
pub fn predict_mv(left: MotionVector, top: MotionVector, top_right: MotionVector) -> MotionVector {
    MotionVector {
        dx: median_of_three(left.dx, top.dx, top_right.dx),
        dy: median_of_three(left.dy, top.dy, top_right.dy),
        ref_idx: 0,
    }
}

fn median_of_three(a: i16, b: i16, c: i16) -> i16 {
    let mut arr = [a, b, c];
    arr.sort();
    arr[1]
}

// ---------------------------------------------------------------------------
// Motion compensation
// ---------------------------------------------------------------------------

/// Apply motion compensation: copy a 16x16 block from the reference frame with
/// the given motion vector offset. Pixel coordinates that fall outside the
/// reference frame are clamped to the nearest edge sample.
#[allow(clippy::too_many_arguments)]
pub fn motion_compensate_16x16(
    reference: &[u8],
    ref_width: usize,
    ref_height: usize,
    channels: usize,
    mv: MotionVector,
    mb_x: usize,
    mb_y: usize,
    output: &mut [u8],
    out_width: usize,
) {
    let src_x = (mb_x * 16) as i32 + mv.dx as i32;
    let src_y = (mb_y * 16) as i32 + mv.dy as i32;
    let ref_w = ref_width as i32;
    let ref_h = ref_height as i32;

    // Fast path for single-channel (luma) when entire block is in-bounds
    if channels == 1 && src_x >= 0 && src_x + 16 <= ref_w && src_y >= 0 && src_y + 16 <= ref_h {
        let sx = src_x as usize;
        let sy = src_y as usize;
        for row in 0..16 {
            let dst_start = (mb_y * 16 + row) * out_width + mb_x * 16;
            let src_start = (sy + row) * ref_width + sx;
            if dst_start + 16 <= output.len() && src_start + 16 <= reference.len() {
                output[dst_start..dst_start + 16]
                    .copy_from_slice(&reference[src_start..src_start + 16]);
            }
        }
        return;
    }

    // Slow path: per-pixel with clamping (handles edge cases + multi-channel)
    for row in 0..16 {
        let sy = (src_y + row as i32).clamp(0, ref_h - 1) as usize;
        let dst_y = mb_y * 16 + row;
        for col in 0..16 {
            let sx = (src_x + col as i32).clamp(0, ref_w - 1) as usize;
            let dst_x = mb_x * 16 + col;
            for c in 0..channels {
                let dst_idx = (dst_y * out_width + dst_x) * channels + c;
                let src_idx = (sy * ref_width + sx) * channels + c;
                if dst_idx < output.len() && src_idx < reference.len() {
                    output[dst_idx] = reference[src_idx];
                }
            }
        }
    }
}

/// Generic block-size motion compensation for sub-partitions (16x8, 8x16, 8x8, etc.).
#[allow(clippy::too_many_arguments)]
pub fn motion_compensate_block(
    reference: &[u8],
    ref_width: usize,
    ref_height: usize,
    mv: MotionVector,
    block_x: usize,
    block_y: usize,
    block_w: usize,
    block_h: usize,
    output: &mut [u8],
    out_width: usize,
) {
    let src_x = block_x as i32 + mv.dx as i32;
    let src_y = block_y as i32 + mv.dy as i32;
    let ref_w = ref_width as i32;
    let ref_h = ref_height as i32;

    // Fast path: entire block in-bounds
    if src_x >= 0
        && src_x + block_w as i32 <= ref_w
        && src_y >= 0
        && src_y + block_h as i32 <= ref_h
    {
        let sx = src_x as usize;
        let sy = src_y as usize;
        for row in 0..block_h {
            let dst_start = (block_y + row) * out_width + block_x;
            let src_start = (sy + row) * ref_width + sx;
            if dst_start + block_w <= output.len() && src_start + block_w <= reference.len() {
                output[dst_start..dst_start + block_w]
                    .copy_from_slice(&reference[src_start..src_start + block_w]);
            }
        }
        return;
    }

    // Slow path: per-pixel with clamping
    for row in 0..block_h {
        let sy = (src_y + row as i32).clamp(0, ref_h - 1) as usize;
        for col in 0..block_w {
            let sx = (src_x + col as i32).clamp(0, ref_w - 1) as usize;
            let dst_idx = (block_y + row) * out_width + block_x + col;
            let src_idx = sy * ref_width + sx;
            if dst_idx < output.len() && src_idx < reference.len() {
                output[dst_idx] = reference[src_idx];
            }
        }
    }
}

/// Apply half-pel interpolation for sub-pixel motion vectors.
///
/// Motion vectors are in quarter-pel units. Integer-pel positions are copied
/// directly; fractional positions use bilinear interpolation between the two
/// nearest integer samples (half-pel approximation).
#[allow(clippy::too_many_arguments)]
pub fn motion_compensate_halfpel(
    reference: &[u8],
    ref_width: usize,
    ref_height: usize,
    channels: usize,
    mv: MotionVector,
    mb_x: usize,
    mb_y: usize,
    output: &mut [u8],
    out_width: usize,
) {
    let base_x = (mb_x * 16) as i32 * 4 + mv.dx as i32;
    let base_y = (mb_y * 16) as i32 * 4 + mv.dy as i32;

    for row in 0..16 {
        for col in 0..16 {
            let qx = base_x + col as i32 * 4;
            let qy = base_y + row as i32 * 4;

            // Integer sample position and fractional offset (0..3)
            let ix = qx >> 2;
            let iy = qy >> 2;
            let fx = (qx & 3) as u16;
            let fy = (qy & 3) as u16;

            // Round quarter-pel to half-pel grid (0, 2, or snap to nearest)
            let hx = fx.div_ceil(2); // 0->0, 1->1, 2->1, 3->2 but we only use 0 or 1
            let hy = fy.div_ceil(2);

            let x0 = ix.clamp(0, ref_width as i32 - 1) as usize;
            let y0 = iy.clamp(0, ref_height as i32 - 1) as usize;
            let x1 = (ix + 1).clamp(0, ref_width as i32 - 1) as usize;
            let y1 = (iy + 1).clamp(0, ref_height as i32 - 1) as usize;

            let dst_y = mb_y * 16 + row;
            let dst_x = mb_x * 16 + col;

            for c in 0..channels {
                let s00 = reference[(y0 * ref_width + x0) * channels + c] as u16;
                let s10 = reference[(y0 * ref_width + x1) * channels + c] as u16;
                let s01 = reference[(y1 * ref_width + x0) * channels + c] as u16;
                let s11 = reference[(y1 * ref_width + x1) * channels + c] as u16;

                // Bilinear blend: weight = hx, hy in {0, 1, 2} mapped to 0..2
                let val = if hx == 0 && hy == 0 {
                    s00
                } else if hx > 0 && hy == 0 {
                    (s00 + s10).div_ceil(2)
                } else if hx == 0 && hy > 0 {
                    (s00 + s01).div_ceil(2)
                } else {
                    (s00 + s10 + s01 + s11 + 2) / 4
                };

                let dst_idx = (dst_y * out_width + dst_x) * channels + c;
                if dst_idx < output.len() {
                    output[dst_idx] = val as u8;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Weighted prediction (H.264 7.4.3.2)
// ---------------------------------------------------------------------------

/// Apply uni-directional weighted prediction to a 16x16 block in-place.
///
/// For each sample: `out = clip((log2_denom == 0 ? 0 : (1 << (log2_denom-1))) + weight * pred + (offset << log2_denom)) >> log2_denom`
/// Simplified: `out = clip(((weight * pred + round) >> log2_denom) + offset)`
#[allow(clippy::too_many_arguments)]
pub fn apply_weighted_pred(
    plane: &mut [u8],
    stride: usize,
    bx: usize,
    by: usize,
    block_w: usize,
    block_h: usize,
    weight: i32,
    offset: i32,
    log2_denom: u32,
) {
    let round = if log2_denom > 0 {
        1i32 << (log2_denom - 1)
    } else {
        0
    };
    for row in 0..block_h {
        for col in 0..block_w {
            let idx = (by + row) * stride + bx + col;
            if idx < plane.len() {
                let pred = plane[idx] as i32;
                let val = ((weight * pred + round) >> log2_denom) + offset;
                plane[idx] = val.clamp(0, 255) as u8;
            }
        }
    }
}

/// Apply bi-directional weighted prediction to a 16x16 block.
///
/// `out = clip(((w0 * pred0 + w1 * pred1 + round) >> (log2_denom + 1)) + ((o0 + o1 + 1) >> 1))`
#[allow(clippy::too_many_arguments)]
pub fn apply_bipred_weighted(
    output: &mut [u8],
    stride: usize,
    bx: usize,
    by: usize,
    block_w: usize,
    block_h: usize,
    pred0: &[u8],
    pred1: &[u8],
    pred_stride: usize,
    w0: i32,
    o0: i32,
    w1: i32,
    o1: i32,
    log2_denom: u32,
) {
    let round = 1i32 << log2_denom;
    let offset = (o0 + o1 + 1) >> 1;
    let shift = log2_denom + 1;
    for row in 0..block_h {
        for col in 0..block_w {
            let p0 = pred0[row * pred_stride + col] as i32;
            let p1 = pred1[row * pred_stride + col] as i32;
            let val = ((w0 * p0 + w1 * p1 + round) >> shift) + offset;
            let idx = (by + row) * stride + bx + col;
            if idx < output.len() {
                output[idx] = val.clamp(0, 255) as u8;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// P-slice macroblock decoder
// ---------------------------------------------------------------------------

/// Decode a P-slice macroblock: parse mb_type, motion vectors, and apply
/// motion compensation from the reference frame.
///
/// Returns the decoded motion vector so the caller can store it for
/// neighboring-block prediction of subsequent macroblocks.
#[allow(clippy::too_many_arguments)]
pub fn decode_p_macroblock(
    reader: &mut BitstreamReader,
    reference_frame: &[u8],
    ref_width: usize,
    ref_height: usize,
    mb_x: usize,
    mb_y: usize,
    neighbor_mvs: &[MotionVector],
    output: &mut [u8],
    out_width: usize,
) -> Result<MotionVector, VideoError> {
    // 1. Parse mb_type (P_L0_16x16 = 0 for simplest case)
    let _mb_type = reader.read_ue()?;

    // 2. Parse motion vector difference
    let (mvd_x, mvd_y) = parse_mvd(reader)?;

    // 3. Predict MV from neighbors (left, top, top-right)
    let predicted = predict_mv(
        neighbor_mvs.first().copied().unwrap_or_default(),
        neighbor_mvs.get(1).copied().unwrap_or_default(),
        neighbor_mvs.get(2).copied().unwrap_or_default(),
    );

    // 4. Final MV = predicted + difference
    let mv = MotionVector {
        dx: predicted.dx + mvd_x,
        dy: predicted.dy + mvd_y,
        ref_idx: 0,
    };

    // 5. Motion compensate (integer-pel for P_L0_16x16)
    motion_compensate_16x16(
        reference_frame,
        ref_width,
        ref_height,
        3,
        mv,
        mb_x,
        mb_y,
        output,
        out_width,
    );

    Ok(mv)
}

// ---------------------------------------------------------------------------
// Reference frame buffer
// ---------------------------------------------------------------------------

/// Simple reference frame buffer for P-slice decoding.
///
/// Maintains a bounded FIFO of recent reconstructed frames so that P-slices
/// can reference them for motion compensation.
pub struct ReferenceFrameBuffer {
    frames: Vec<Vec<u8>>,
    max_refs: usize,
}

impl ReferenceFrameBuffer {
    /// Creates a new buffer that keeps at most `max_refs` reference frames.
    pub fn new(max_refs: usize) -> Self {
        Self {
            frames: Vec::new(),
            max_refs,
        }
    }

    /// Pushes a reconstructed frame into the buffer, evicting the oldest if
    /// the capacity is exceeded.
    pub fn push(&mut self, frame: Vec<u8>) {
        if self.frames.len() >= self.max_refs {
            self.frames.remove(0);
        }
        self.frames.push(frame);
    }

    /// Returns the reference frame at `idx` (0 = oldest retained frame).
    pub fn get(&self, idx: usize) -> Option<&[u8]> {
        self.frames.get(idx).map(|v| v.as_slice())
    }

    /// Returns the most recently pushed reference frame.
    pub fn latest(&self) -> Option<&[u8]> {
        self.frames.last().map(|v| v.as_slice())
    }

    /// Returns the number of reference frames currently stored.
    pub fn len(&self) -> usize {
        self.frames.len()
    }

    /// Returns `true` if the buffer contains no reference frames.
    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- Test helpers (same convention as h264_decoder.rs) -------------------

    fn push_exp_golomb(bits: &mut Vec<u8>, value: u32) {
        if value == 0 {
            bits.push(1);
            return;
        }
        let code = value + 1;
        let bit_len = 32 - code.leading_zeros();
        let leading_zeros = bit_len - 1;
        for _ in 0..leading_zeros {
            bits.push(0);
        }
        for i in (0..bit_len).rev() {
            bits.push(((code >> i) & 1) as u8);
        }
    }

    fn push_signed_exp_golomb(bits: &mut Vec<u8>, value: i32) {
        let code = if value > 0 {
            (2 * value - 1) as u32
        } else if value < 0 {
            (2 * (-value)) as u32
        } else {
            0
        };
        push_exp_golomb(bits, code);
    }

    fn bits_to_bytes(bits: &[u8]) -> Vec<u8> {
        let mut bytes = Vec::new();
        for chunk in bits.chunks(8) {
            let mut byte = 0u8;
            for (i, &bit) in chunk.iter().enumerate() {
                byte |= bit << (7 - i);
            }
            bytes.push(byte);
        }
        bytes
    }

    // -- 1. Median prediction -----------------------------------------------

    #[test]
    fn motion_vector_median_prediction() {
        let left = MotionVector {
            dx: 2,
            dy: -4,
            ref_idx: 0,
        };
        let top = MotionVector {
            dx: 6,
            dy: 1,
            ref_idx: 0,
        };
        let top_right = MotionVector {
            dx: -3,
            dy: 8,
            ref_idx: 0,
        };

        let pred = predict_mv(left, top, top_right);
        // median(2, 6, -3) = 2; median(-4, 1, 8) = 1
        assert_eq!(pred.dx, 2);
        assert_eq!(pred.dy, 1);

        // All-zero neighbors
        let zero = MotionVector::default();
        let pred_zero = predict_mv(zero, zero, zero);
        assert_eq!(pred_zero.dx, 0);
        assert_eq!(pred_zero.dy, 0);

        // Two equal, one different
        let a = MotionVector {
            dx: 5,
            dy: 5,
            ref_idx: 0,
        };
        let b = MotionVector {
            dx: 5,
            dy: 5,
            ref_idx: 0,
        };
        let c = MotionVector {
            dx: -10,
            dy: 20,
            ref_idx: 0,
        };
        let pred2 = predict_mv(a, b, c);
        assert_eq!(pred2.dx, 5);
        assert_eq!(pred2.dy, 5);
    }

    // -- 2. Motion compensation block copy ----------------------------------

    #[test]
    fn motion_compensate_copies_block() {
        // 32x32 reference, 1 channel, filled with row index as pixel value
        let ref_w = 32;
        let ref_h = 32;
        let channels = 1;
        let mut reference = vec![0u8; ref_w * ref_h * channels];
        for row in 0..ref_h {
            for col in 0..ref_w {
                reference[row * ref_w + col] = row as u8;
            }
        }

        // MB at (0, 0) with mv=(0, 0) should copy top-left 16x16
        let mut output = vec![0u8; ref_w * ref_h * channels];
        let mv = MotionVector {
            dx: 0,
            dy: 0,
            ref_idx: 0,
        };
        motion_compensate_16x16(
            &reference,
            ref_w,
            ref_h,
            channels,
            mv,
            0,
            0,
            &mut output,
            ref_w,
        );

        for row in 0..16 {
            for col in 0..16 {
                assert_eq!(
                    output[row * ref_w + col],
                    row as u8,
                    "mismatch at ({row}, {col})"
                );
            }
        }

        // MB at (0, 0) with mv=(4, 2) should copy from (4, 2)
        let mut output2 = vec![0u8; ref_w * ref_h * channels];
        let mv2 = MotionVector {
            dx: 4,
            dy: 2,
            ref_idx: 0,
        };
        motion_compensate_16x16(
            &reference,
            ref_w,
            ref_h,
            channels,
            mv2,
            0,
            0,
            &mut output2,
            ref_w,
        );

        for row in 0..16 {
            let expected_src_y = (row as i32 + 2).clamp(0, ref_h as i32 - 1) as u8;
            for col in 0..16 {
                assert_eq!(
                    output2[row * ref_w + col],
                    expected_src_y,
                    "offset mismatch at ({row}, {col})"
                );
            }
        }
    }

    // -- 3. Reference frame buffer FIFO -------------------------------------

    #[test]
    fn reference_frame_buffer_fifo() {
        let mut buf = ReferenceFrameBuffer::new(3);
        assert!(buf.is_empty());
        assert_eq!(buf.len(), 0);
        assert!(buf.latest().is_none());

        buf.push(vec![1, 2, 3]);
        buf.push(vec![4, 5, 6]);
        buf.push(vec![7, 8, 9]);
        assert_eq!(buf.len(), 3);
        assert_eq!(buf.get(0), Some([1u8, 2, 3].as_slice()));
        assert_eq!(buf.get(1), Some([4u8, 5, 6].as_slice()));
        assert_eq!(buf.get(2), Some([7u8, 8, 9].as_slice()));
        assert_eq!(buf.latest(), Some([7u8, 8, 9].as_slice()));

        // Pushing a 4th frame evicts the oldest
        buf.push(vec![10, 11, 12]);
        assert_eq!(buf.len(), 3);
        assert_eq!(buf.get(0), Some([4u8, 5, 6].as_slice()));
        assert_eq!(buf.latest(), Some([10u8, 11, 12].as_slice()));
        assert!(buf.get(3).is_none());
    }

    // -- 4. parse_mvd roundtrip ---------------------------------------------

    #[test]
    fn parse_mvd_roundtrip() {
        // Encode se(3) and se(-5) into a bitstream, then parse them back.
        let mut bits = Vec::new();
        push_signed_exp_golomb(&mut bits, 3);
        push_signed_exp_golomb(&mut bits, -5);
        // Pad to byte boundary
        while bits.len() % 8 != 0 {
            bits.push(0);
        }
        let bytes = bits_to_bytes(&bits);

        let mut reader = BitstreamReader::new(&bytes);
        let (mvd_x, mvd_y) = parse_mvd(&mut reader).unwrap();
        assert_eq!(mvd_x, 3);
        assert_eq!(mvd_y, -5);

        // Zero MVD
        let mut bits2 = Vec::new();
        push_signed_exp_golomb(&mut bits2, 0);
        push_signed_exp_golomb(&mut bits2, 0);
        while bits2.len() % 8 != 0 {
            bits2.push(0);
        }
        let bytes2 = bits_to_bytes(&bits2);

        let mut reader2 = BitstreamReader::new(&bytes2);
        let (mvd_x2, mvd_y2) = parse_mvd(&mut reader2).unwrap();
        assert_eq!(mvd_x2, 0);
        assert_eq!(mvd_y2, 0);
    }
}
