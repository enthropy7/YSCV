//! H.264 B-slice decoding: bi-directional motion vectors, prediction modes,
//! and bi-predictive motion compensation.

use crate::h264_motion::{MotionVector, motion_compensate_16x16, parse_mvd, predict_mv};
use crate::{BitstreamReader, VideoError};

// ---------------------------------------------------------------------------
// Bi-directional motion vector types
// ---------------------------------------------------------------------------

/// Bi-directional motion vectors for B-slice macroblocks.
#[derive(Debug, Clone, Copy, Default)]
pub struct BiMotionVector {
    /// Motion vector from past (forward) reference.
    pub forward: MotionVector,
    /// Motion vector from future (backward) reference.
    pub backward: MotionVector,
    /// Prediction mode for this macroblock.
    pub mode: BPredMode,
}

/// B-slice prediction mode.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub enum BPredMode {
    /// Use only forward (past) reference.
    #[default]
    Forward,
    /// Use only backward (future) reference.
    Backward,
    /// Average of forward and backward predictions.
    BiPred,
    /// Derive MVs from co-located MB in future reference.
    Direct,
}

// ---------------------------------------------------------------------------
// Bi-directional motion compensation
// ---------------------------------------------------------------------------

/// Bi-directional motion compensation: dispatches to the correct mode and
/// averages forward and backward predictions when needed.
#[allow(clippy::too_many_arguments)]
pub fn motion_compensate_bipred(
    ref_fwd: &[u8],
    ref_bwd: &[u8],
    width: usize,
    height: usize,
    channels: usize,
    bi_mv: &BiMotionVector,
    mb_x: usize,
    mb_y: usize,
    output: &mut [u8],
    out_width: usize,
) {
    match bi_mv.mode {
        BPredMode::Forward => {
            motion_compensate_16x16(
                ref_fwd,
                width,
                height,
                width,
                channels,
                bi_mv.forward,
                mb_x,
                mb_y,
                output,
                out_width,
            );
        }
        BPredMode::Backward => {
            motion_compensate_16x16(
                ref_bwd,
                width,
                height,
                width,
                channels,
                bi_mv.backward,
                mb_x,
                mb_y,
                output,
                out_width,
            );
        }
        BPredMode::BiPred | BPredMode::Direct => {
            // Allocate temporary buffers for each prediction direction.
            let mut fwd_block = vec![0u8; 16 * 16 * channels];
            let mut bwd_block = vec![0u8; 16 * 16 * channels];

            motion_compensate_16x16(
                ref_fwd,
                width,
                height,
                width,
                channels,
                bi_mv.forward,
                mb_x,
                mb_y,
                &mut fwd_block,
                16,
            );
            motion_compensate_16x16(
                ref_bwd,
                width,
                height,
                width,
                channels,
                bi_mv.backward,
                mb_x,
                mb_y,
                &mut bwd_block,
                16,
            );

            // Average the two predictions with rounding.
            for row in 0..16 {
                for col in 0..16 {
                    let dst_y = mb_y * 16 + row;
                    let dst_x = mb_x * 16 + col;
                    for c in 0..channels {
                        let f = fwd_block[(row * 16 + col) * channels + c] as u16;
                        let b = bwd_block[(row * 16 + col) * channels + c] as u16;
                        let dst_idx = (dst_y * out_width + dst_x) * channels + c;
                        if dst_idx < output.len() {
                            output[dst_idx] = (f + b).div_ceil(2) as u8;
                        }
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// B-slice macroblock decoder
// ---------------------------------------------------------------------------

/// Decode a B-slice macroblock: parse mb_type and motion vectors, then apply
/// motion compensation.
///
/// Returns the decoded `BiMotionVector` so the caller can store it for
/// neighboring-block prediction of subsequent macroblocks.
#[allow(clippy::too_many_arguments)]
pub fn decode_b_macroblock(
    reader: &mut BitstreamReader,
    ref_fwd: &[u8],
    ref_bwd: &[u8],
    width: usize,
    height: usize,
    mb_x: usize,
    mb_y: usize,
    neighbor_mvs_fwd: &[MotionVector],
    neighbor_mvs_bwd: &[MotionVector],
    output: &mut [u8],
    out_width: usize,
) -> Result<BiMotionVector, VideoError> {
    // 1. Parse mb_type to determine prediction mode.
    let mb_type = reader.read_ue()?;
    let mode = match mb_type {
        0 => BPredMode::Direct,
        1 => BPredMode::Forward,
        2 => BPredMode::Backward,
        _ => BPredMode::BiPred,
    };

    // 2. Parse motion vector differences and predict final MVs.
    let forward = if mode == BPredMode::Forward || mode == BPredMode::BiPred {
        let (mvd_x, mvd_y) = parse_mvd(reader)?;
        let predicted = predict_mv(
            neighbor_mvs_fwd.first().copied().unwrap_or_default(),
            neighbor_mvs_fwd.get(1).copied().unwrap_or_default(),
            neighbor_mvs_fwd.get(2).copied().unwrap_or_default(),
        );
        MotionVector {
            dx: predicted.dx + mvd_x,
            dy: predicted.dy + mvd_y,
            ref_idx: 0,
        }
    } else {
        MotionVector::default()
    };

    let backward = if mode == BPredMode::Backward || mode == BPredMode::BiPred {
        let (mvd_x, mvd_y) = parse_mvd(reader)?;
        let predicted = predict_mv(
            neighbor_mvs_bwd.first().copied().unwrap_or_default(),
            neighbor_mvs_bwd.get(1).copied().unwrap_or_default(),
            neighbor_mvs_bwd.get(2).copied().unwrap_or_default(),
        );
        MotionVector {
            dx: predicted.dx + mvd_x,
            dy: predicted.dy + mvd_y,
            ref_idx: 0,
        }
    } else {
        MotionVector::default()
    };

    let bi_mv = BiMotionVector {
        forward,
        backward,
        mode,
    };

    // 3. Apply motion compensation.
    let channels = 3;
    motion_compensate_bipred(
        ref_fwd, ref_bwd, width, height, channels, &bi_mv, mb_x, mb_y, output, out_width,
    );

    Ok(bi_mv)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bipred_averages_references() {
        // Two 32x32 single-channel reference frames: one filled with 100, one with 200.
        let width = 32;
        let height = 32;
        let channels = 1;
        let ref_fwd = vec![100u8; width * height * channels];
        let ref_bwd = vec![200u8; width * height * channels];

        let bi_mv = BiMotionVector {
            forward: MotionVector::default(),
            backward: MotionVector::default(),
            mode: BPredMode::BiPred,
        };

        let mut output = vec![0u8; width * height * channels];
        motion_compensate_bipred(
            &ref_fwd,
            &ref_bwd,
            width,
            height,
            channels,
            &bi_mv,
            0,
            0,
            &mut output,
            width,
        );

        // Average of 100 and 200 with rounding = (100 + 200 + 1) / 2 = 150.
        for row in 0..16 {
            for col in 0..16 {
                assert_eq!(
                    output[row * width + col],
                    150,
                    "bipred average mismatch at ({row}, {col})"
                );
            }
        }
    }

    #[test]
    fn bpred_forward_only() {
        let width = 32;
        let height = 32;
        let channels = 1;
        let ref_fwd = vec![42u8; width * height * channels];
        let ref_bwd = vec![200u8; width * height * channels];

        let bi_mv = BiMotionVector {
            forward: MotionVector::default(),
            backward: MotionVector::default(),
            mode: BPredMode::Forward,
        };

        let mut output = vec![0u8; width * height * channels];
        motion_compensate_bipred(
            &ref_fwd,
            &ref_bwd,
            width,
            height,
            channels,
            &bi_mv,
            0,
            0,
            &mut output,
            width,
        );

        // Forward-only should use ref_fwd exclusively.
        for row in 0..16 {
            for col in 0..16 {
                assert_eq!(
                    output[row * width + col],
                    42,
                    "forward-only mismatch at ({row}, {col})"
                );
            }
        }
    }

    #[test]
    fn bpred_backward_only() {
        let width = 32;
        let height = 32;
        let channels = 1;
        let ref_fwd = vec![42u8; width * height * channels];
        let ref_bwd = vec![77u8; width * height * channels];

        let bi_mv = BiMotionVector {
            forward: MotionVector::default(),
            backward: MotionVector::default(),
            mode: BPredMode::Backward,
        };

        let mut output = vec![0u8; width * height * channels];
        motion_compensate_bipred(
            &ref_fwd,
            &ref_bwd,
            width,
            height,
            channels,
            &bi_mv,
            0,
            0,
            &mut output,
            width,
        );

        // Backward-only should use ref_bwd exclusively.
        for row in 0..16 {
            for col in 0..16 {
                assert_eq!(
                    output[row * width + col],
                    77,
                    "backward-only mismatch at ({row}, {col})"
                );
            }
        }
    }
}
