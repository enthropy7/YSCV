//! HEVC integration tests against committed test fixtures.
//!
//! These tests run end-to-end through `Mp4VideoReader::open` →
//! `next_frame()` against pre-generated 320×240 HEVC clips at
//! `tests/fixtures/hevc/`. They are committed to the repo (~67 KB total)
//! so CI runs them on every push without needing ffmpeg.
//!
//! //! decoder gains coverage for weighted prediction, tiles, WPP, and
//! dependent slice segments.

use std::path::PathBuf;
use yscv_video::Mp4VideoReader;

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/hevc")
        .join(name)
}

fn assert_frame_dimensions(rgb_len: usize, width: usize, height: usize) {
    let expected = width * height * 3;
    assert_eq!(
        rgb_len, expected,
        "RGB buffer size mismatch: expected {expected} ({width}x{height}*3), got {rgb_len}"
    );
}

fn pixel_range(rgb: &[u8]) -> (u8, u8) {
    let min = rgb.iter().copied().min().unwrap_or(0);
    let max = rgb.iter().copied().max().unwrap_or(0);
    (min, max)
}

#[test]
fn hevc_main_ionly_decodes_without_panic() {
    let path = fixture("main_ionly_320x240.mp4");
    assert!(path.exists(), "fixture missing: {}", path.display());
    let mut reader = Mp4VideoReader::open(&path).expect("open Main I-only HEVC fixture");

    let mut decoded_count = 0usize;
    while let Ok(Some(frame)) = reader.next_frame() {
        assert_eq!(frame.width, 320);
        assert_eq!(frame.height, 240);
        assert_frame_dimensions(frame.rgb8_data.len(), 320, 240);
        // 8-bit Main profile should report bit_depth = 8.
        assert_eq!(
            frame.bit_depth, 8,
            "Main profile must report bit_depth=8, got {}",
            frame.bit_depth
        );
        let (min, max) = pixel_range(&frame.rgb8_data);
        assert!(
            max > min,
            "frame {decoded_count} is uniform ({min}=={max}); decoder probably failed silently"
        );
        decoded_count += 1;
    }
    assert!(
        decoded_count > 0,
        "Main I-only fixture should produce at least one decoded frame"
    );
}

#[test]
fn hevc_main_pb_decodes_without_panic() {
    // P/B frames without weighted prediction. This validates the existing
    // motion-compensation path under inter prediction.
    let path = fixture("main_pb_320x240.mp4");
    assert!(path.exists(), "fixture missing: {}", path.display());
    let mut reader = Mp4VideoReader::open(&path).expect("open Main P/B HEVC fixture");

    let mut decoded_count = 0usize;
    while let Ok(Some(frame)) = reader.next_frame() {
        assert_eq!(frame.width, 320);
        assert_eq!(frame.height, 240);
        assert_frame_dimensions(frame.rgb8_data.len(), 320, 240);
        assert_eq!(frame.bit_depth, 8);
        let (min, max) = pixel_range(&frame.rgb8_data);
        assert!(
            max > min,
            "P/B frame {decoded_count} is uniform; decoder may have stalled"
        );
        decoded_count += 1;
    }
    assert!(
        decoded_count > 0,
        "Main P/B fixture should produce at least one decoded frame"
    );
}

#[test]
fn hevc_main_pb_weighted_decodes_without_panic() {
    // P/B frames WITH weighted prediction (libx265 `weightp=2:weightb=1`).
    // added the slice-header full parser, L0/L1 reference list
    // construction, and the §8.5.3.3.4 weighted-prediction formulas in MC.
    // This test validates that the weighted-prediction code path runs end
    // to end without panicking and produces non-uniform output.
    let path = fixture("main_pb_weighted_320x240.mp4");
    assert!(path.exists(), "fixture missing: {}", path.display());
    let mut reader = Mp4VideoReader::open(&path).expect("open Main P/B weighted-pred HEVC fixture");

    let mut decoded_count = 0usize;
    while let Ok(Some(frame)) = reader.next_frame() {
        assert_eq!(frame.width, 320);
        assert_eq!(frame.height, 240);
        assert_frame_dimensions(frame.rgb8_data.len(), 320, 240);
        assert_eq!(frame.bit_depth, 8);
        let (min, max) = pixel_range(&frame.rgb8_data);
        assert!(
            max > min,
            "weighted-pred frame {decoded_count} is uniform; weighted MC path may be broken"
        );
        decoded_count += 1;
    }
    assert!(
        decoded_count > 0,
        "Main P/B weighted-pred fixture should produce at least one decoded frame"
    );
}

#[test]
fn hevc_main10_ionly_reports_bit_depth_10() {
    let path = fixture("main10_ionly_320x240.mp4");
    assert!(path.exists(), "fixture missing: {}", path.display());
    let mut reader = Mp4VideoReader::open(&path).expect("open Main10 I-only HEVC fixture");

    let mut decoded_count = 0usize;
    while let Ok(Some(frame)) = reader.next_frame() {
        assert_eq!(frame.width, 320);
        assert_eq!(frame.height, 240);
        assert_frame_dimensions(frame.rgb8_data.len(), 320, 240);
        // Main10 must propagate the source bit depth to the caller.
        assert_eq!(
            frame.bit_depth, 10,
            "Main10 fixture must report bit_depth=10, got {}",
            frame.bit_depth
        );
        let (min, max) = pixel_range(&frame.rgb8_data);
        assert!(
            max > min,
            "Main10 frame {decoded_count} is uniform ({min}=={max}); 10-bit pipeline broken"
        );
        decoded_count += 1;
    }
    assert!(
        decoded_count > 0,
        "Main10 I-only fixture should produce at least one decoded frame"
    );
}

// ---------------------------------------------------------------------------
// integration tests
// ---------------------------------------------------------------------------
//
// One test per fixture. Each test exercises the full
// `Mp4VideoReader::open(...) → next_frame()` path and asserts dimensions,
// RGB length, bit_depth, and that the pixel range is non-uniform. These are
// smoke tests that catch crashes and total decode failure; pixel-level
// correctness is validated by the existing per-feature unit tests in the
// hevc_decoder / hevc_syntax / hevc_filter modules.

/// Helper to drain a fixture and return (frame_count, bit_depth, min, max).
fn decode_fixture(name: &str, expected_bd: u8) -> (usize, u8, u8, u8) {
    let path = fixture(name);
    assert!(path.exists(), "fixture missing: {}", path.display());
    let mut reader = Mp4VideoReader::open(&path).unwrap_or_else(|e| panic!("open {name}: {e}"));
    let mut count = 0usize;
    let mut bd = 0u8;
    let mut min_v = 255u8;
    let mut max_v = 0u8;
    while let Some(frame) = reader
        .next_frame()
        .unwrap_or_else(|e| panic!("next_frame {name}: {e}"))
    {
        assert_eq!(frame.width, 320, "{name} width");
        assert_eq!(frame.height, 240, "{name} height");
        assert_frame_dimensions(frame.rgb8_data.len(), 320, 240);
        assert_eq!(frame.bit_depth, expected_bd, "{name} bit_depth");
        bd = frame.bit_depth;
        let (mn, mx) = pixel_range(&frame.rgb8_data);
        if mn < min_v {
            min_v = mn;
        }
        if mx > max_v {
            max_v = mx;
        }
        count += 1;
    }
    assert!(count > 0, "{name} produced 0 frames");
    assert!(
        max_v > min_v,
        "{name} produced uniform pixel range [{min_v},{max_v}]; decoder may have stalled"
    );
    (count, bd, min_v, max_v)
}

// 4c.1 — non-4:2:0 chroma format coverage

#[test]
fn hevc_yuv422_decodes_without_panic() {
    decode_fixture("main_yuv422_320x240.mp4", 8);
}

#[test]
fn hevc_yuv444_decodes_without_panic() {
    decode_fixture("main_yuv444_320x240.mp4", 8);
}

#[test]
fn hevc_monochrome_decodes_without_panic() {
    decode_fixture("main_mono_320x240.mp4", 8);
}

// 4c.4 / 4c.5 / 4c.6 — tiles, WPP, multi-segment slices

#[test]
fn hevc_tiles_2x2_decodes_without_panic() {
    decode_fixture("main_tiles_320x240.mp4", 8);
}

#[test]
fn hevc_wpp_decodes_without_panic() {
    decode_fixture("main_wpp_320x240.mp4", 8);
}

#[test]
fn hevc_multi_slice_decodes_without_panic() {
    decode_fixture("main_slices_320x240.mp4", 8);
}

// 4c.7 / 4c.8 — ref pic list modification + long-term references

#[test]
fn hevc_ref_modification_decodes_without_panic() {
    decode_fixture("main_ref_modification_320x240.mp4", 8);
}

#[test]
fn hevc_ltrp_decodes_without_panic() {
    decode_fixture("main_ltrp_320x240.mp4", 8);
}

// 4c.9 — separate colour planes (gbrp / chroma_format_idc=3)

#[test]
fn hevc_separate_colour_plane_decodes_without_panic() {
    decode_fixture("main_scp_320x240.mp4", 8);
}

// 4c.10 / 4c.11 — HEVC Range Extensions profiles

#[test]
fn hevc_main422_10_decodes_with_bit_depth_10() {
    decode_fixture("main422_10_320x240.mp4", 10);
}

#[test]
fn hevc_main444_10_decodes_with_bit_depth_10() {
    decode_fixture("main444_10_320x240.mp4", 10);
}

#[test]
fn hevc_main422_12_decodes_with_bit_depth_12() {
    decode_fixture("main422_12_320x240.mp4", 12);
}

// — Rext coding tools test fixtures (kvazaar-generated)

#[test]
fn hevc_rext_transform_skip_decodes_without_panic() {
    decode_fixture("rext_transform_skip_320x240.mp4", 8);
}

#[test]
fn hevc_rext_rdpcm_decodes_without_panic() {
    decode_fixture("rext_rdpcm_320x240.mp4", 8);
}
