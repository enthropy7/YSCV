//! H.264 integration tests against committed test fixtures.
//!
//! Runs end-to-end through `Mp4VideoReader::open` → `next_frame()` against
//! pre-generated Constrained Baseline (CAVLC) clips at `tests/fixtures/h264/`.
//! The clips are encoded with deblocking disabled, so the decoder's
//! reconstruction can be compared *bit-exactly* against an ffmpeg-decoded YUV
//! reference (converted through the same YUV→RGB path the decoder uses). This
//! locks in the intra and inter (P-slice) decode paths — the issue #20
//! regression.

use std::path::PathBuf;
use yscv_video::{yuv420_to_rgb8, Mp4VideoReader};

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/h264")
        .join(name)
}

/// Decodes every frame of `clip` and asserts each matches the corresponding
/// frame of the raw yuv420p reference bit-exactly (through the decoder's own
/// YUV→RGB conversion, so the comparison isolates entropy + reconstruction).
fn assert_clip_bit_exact(clip: &str, reference: &str, w: usize, h: usize, expect_frames: usize) {
    let path = fixture(clip);
    assert!(path.exists(), "fixture missing: {}", path.display());
    let ref_yuv = std::fs::read(fixture(reference)).expect("read reference yuv");
    let (cw, ch) = (w / 2, h / 2);
    let frame_bytes = w * h + 2 * cw * ch;

    let mut reader = Mp4VideoReader::open(&path).expect("open Baseline H.264 fixture");
    let mut decoded = 0usize;
    while let Ok(Some(frame)) = reader.next_frame() {
        let base = decoded * frame_bytes;
        assert!(
            base + frame_bytes <= ref_yuv.len(),
            "decoded more frames than the reference holds"
        );
        let (y, u, v) = (
            &ref_yuv[base..base + w * h],
            &ref_yuv[base + w * h..base + w * h + cw * ch],
            &ref_yuv[base + w * h + cw * ch..base + frame_bytes],
        );
        let ref_rgb = yuv420_to_rgb8(y, u, v, w, h).expect("convert reference yuv to rgb");

        assert_eq!((frame.width, frame.height), (w, h));
        assert_eq!(frame.rgb8_data.len(), ref_rgb.len(), "RGB size mismatch");
        let max_abs = frame
            .rgb8_data
            .iter()
            .zip(&ref_rgb)
            .map(|(&a, &b)| (a as i32 - b as i32).unsigned_abs())
            .max()
            .unwrap_or(0);
        assert_eq!(
            max_abs, 0,
            "frame {decoded} diverges from ffmpeg by up to {max_abs} (must be bit-exact)"
        );
        decoded += 1;
    }
    assert_eq!(
        decoded, expect_frames,
        "expected {expect_frames} decoded frames, got {decoded}"
    );
}

#[test]
fn h264_baseline_idr_decodes_bit_exact() {
    assert_clip_bit_exact(
        "idr_baseline_64x64_nodeblock.mp4",
        "idr_baseline_64x64_nodeblock.ref.yuv",
        64,
        64,
        1,
    );
}

#[test]
fn h264_baseline_pframes_decode_bit_exact() {
    // 5 frames: one IDR followed by four P-slices exercising motion
    // compensation (quarter-pel luma, 1/8-pel chroma), P_Skip and inter
    // residual against the previous reconstructed frame.
    assert_clip_bit_exact(
        "pframes_baseline_64x64_nodeblock.mp4",
        "pframes_baseline_64x64_nodeblock.ref.yuv",
        64,
        64,
        5,
    );
}

#[test]
fn h264_baseline_pframes_subpartitions_decode_bit_exact() {
    // 8 frames whose P-slices use P_8x8 / P_8x8ref0 sub-macroblock partitions
    // next to intra macroblocks — exercises sub-partition motion-vector
    // prediction and the availability-vs-intra distinction for neighbour C.
    assert_clip_bit_exact(
        "pframes_baseline_96x96_subparts.mp4",
        "pframes_baseline_96x96_subparts.ref.yuv",
        96,
        96,
        8,
    );
}

#[test]
fn h264_baseline_pframes_highqp_deblock_decode_bit_exact() {
    // High-QP (sparse) P-slices end in long trailing P_Skip runs, exercising the
    // more_rbsp_data() slice-termination and the deblocker's boundary-strength
    // metadata for those trailing macroblocks.
    assert_clip_bit_exact(
        "pframes_baseline_64x64_highqp_deblock.mp4",
        "pframes_baseline_64x64_highqp_deblock.ref.yuv",
        64,
        64,
        8,
    );
}

#[test]
fn h264_baseline_multislice_multiref_decodes_bit_exact() {
    // 8 frames encoded with three slices per picture, three reference frames,
    // adaptive per-MB QP (aq-mode=1) and non-zero deblock filter offsets
    // (deblock=1,-1): exercises first_mb_in_slice continuation with
    // slice-boundary prediction availability, the sliding-window DPB with
    // ref_idx selection, per-MB QP chroma-QP averaging in the deblocker and
    // the FilterOffsetA/B table indexing.
    assert_clip_bit_exact(
        "pframes_baseline_64x64_multislice_multiref.mp4",
        "pframes_baseline_64x64_multislice_multiref.ref.yuv",
        64,
        64,
        8,
    );
}

#[test]
fn h264_baseline_pframes_deblock_decode_bit_exact() {
    // 6 frames with the in-loop deblocking filter enabled: exercises the
    // spec deblocker (per-edge boundary strength, strong/normal luma filters,
    // chroma, per-edge QP) across I- and P-slices, with the deblocked frames
    // feeding the inter-prediction reference chain.
    assert_clip_bit_exact(
        "pframes_baseline_64x64_deblock.mp4",
        "pframes_baseline_64x64_deblock.ref.yuv",
        64,
        64,
        6,
    );
}
