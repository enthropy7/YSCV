use super::VideoError;
#[cfg(not(feature = "native-camera"))]
use super::camera::CameraFrameSource;
use super::camera::{CameraConfig, CameraDeviceInfo, filter_camera_devices, select_camera_device};
use super::convert::rgb8_bytes_to_frame;
use super::frame::{Frame, PixelFormat, Rgb8Frame};
use super::normalize_rgb8_to_f32_inplace;
use super::source::{FrameSource, InMemoryFrameSource};
use super::stream::FrameStream;
use bytes::Bytes;
use yscv_tensor::Tensor;

#[test]
fn frame_new_accepts_rgb_and_gray() {
    let rgb = Frame::new(
        0,
        0,
        Tensor::from_vec(vec![2, 2, 3], vec![0.0; 12]).unwrap(),
    )
    .unwrap();
    assert_eq!(rgb.pixel_format(), PixelFormat::RgbF32);

    let gray = Frame::new(
        1,
        1_000,
        Tensor::from_vec(vec![2, 2, 1], vec![0.0; 4]).unwrap(),
    )
    .unwrap();
    assert_eq!(gray.pixel_format(), PixelFormat::GrayF32);
}

#[test]
fn frame_new_rejects_invalid_shape() {
    let err = Frame::new(
        0,
        0,
        Tensor::from_vec(vec![4], vec![1.0, 2.0, 3.0, 4.0]).unwrap(),
    )
    .unwrap_err();
    assert_eq!(err, VideoError::InvalidFrameShape { got: vec![4] });
}

#[test]
fn in_memory_source_returns_frames_in_order() {
    let frames = vec![
        Frame::new(0, 0, Tensor::from_vec(vec![1, 1, 1], vec![1.0]).unwrap()).unwrap(),
        Frame::new(
            1,
            1_000,
            Tensor::from_vec(vec![1, 1, 1], vec![2.0]).unwrap(),
        )
        .unwrap(),
    ];
    let mut source = InMemoryFrameSource::new(frames);

    let a = source.next_frame().unwrap().unwrap();
    let b = source.next_frame().unwrap().unwrap();
    let c = source.next_frame().unwrap();

    assert_eq!(a.index(), 0);
    assert_eq!(b.index(), 1);
    assert!(c.is_none());
}

#[test]
fn frame_stream_respects_max_frames() {
    let frames = vec![
        Frame::new(0, 0, Tensor::from_vec(vec![1, 1, 1], vec![1.0]).unwrap()).unwrap(),
        Frame::new(
            1,
            1_000,
            Tensor::from_vec(vec![1, 1, 1], vec![2.0]).unwrap(),
        )
        .unwrap(),
    ];
    let source = InMemoryFrameSource::new(frames);
    let mut stream = FrameStream::new(source).with_max_frames(1);

    assert!(stream.try_next().unwrap().is_some());
    assert!(stream.try_next().unwrap().is_none());
}

#[derive(Debug)]
struct FailingSource;

impl FrameSource for FailingSource {
    fn next_frame(&mut self) -> Result<Option<Frame>, VideoError> {
        Err(VideoError::Source("boom".to_string()))
    }
}

#[test]
fn frame_stream_propagates_source_errors() {
    let mut stream = FrameStream::new(FailingSource);
    let err = stream.try_next().unwrap_err();
    assert_eq!(err, VideoError::Source("boom".to_string()));
}

#[test]
fn camera_config_validation_rejects_invalid_values() {
    let err = CameraConfig {
        device_index: 0,
        width: 0,
        height: 480,
        fps: 30,
    }
    .validate()
    .unwrap_err();
    assert_eq!(
        err,
        VideoError::InvalidCameraResolution {
            width: 0,
            height: 480
        }
    );

    let err = CameraConfig {
        device_index: 0,
        width: 640,
        height: 480,
        fps: 0,
    }
    .validate()
    .unwrap_err();
    assert_eq!(err, VideoError::InvalidCameraFps { fps: 0 });
}

#[test]
fn rgb8_bytes_to_frame_normalizes_values() {
    let frame = rgb8_bytes_to_frame(7, 99, 1, 1, &[255, 128, 0]).unwrap();
    assert_eq!(frame.index(), 7);
    assert_eq!(frame.timestamp_us(), 99);
    assert_eq!(frame.pixel_format(), PixelFormat::RgbF32);
    let pixels = frame.image().data();
    assert!((pixels[0] - 1.0).abs() < 1e-6);
    assert!((pixels[1] - (128.0 / 255.0)).abs() < 1e-6);
    assert!((pixels[2] - 0.0).abs() < 1e-6);
}

#[test]
fn rgb8_bytes_to_frame_rejects_wrong_buffer_size() {
    let err = rgb8_bytes_to_frame(0, 0, 2, 2, &[1, 2, 3]).unwrap_err();
    assert_eq!(
        err,
        VideoError::RawFrameSizeMismatch {
            expected: 12,
            got: 3
        }
    );
}

#[test]
fn rgb8_frame_new_validates_raw_buffer_size() {
    let frame = Rgb8Frame::new(3, 44, 2, 1, vec![10, 20, 30, 40, 50, 60]).unwrap();
    assert_eq!(frame.index(), 3);
    assert_eq!(frame.timestamp_us(), 44);
    assert_eq!(frame.width(), 2);
    assert_eq!(frame.height(), 1);
    assert_eq!(frame.data(), &[10, 20, 30, 40, 50, 60]);

    let err = Rgb8Frame::new(0, 0, 2, 2, vec![1, 2, 3]).unwrap_err();
    assert_eq!(
        err,
        VideoError::RawFrameSizeMismatch {
            expected: 12,
            got: 3
        }
    );
}

#[test]
fn rgb8_frame_from_bytes_roundtrip() {
    let bytes = Bytes::from_static(&[1, 2, 3, 4, 5, 6]);
    let frame = Rgb8Frame::from_bytes(8, 99, 2, 1, bytes.clone()).unwrap();
    assert_eq!(frame.index(), 8);
    assert_eq!(frame.timestamp_us(), 99);
    assert_eq!(frame.width(), 2);
    assert_eq!(frame.height(), 1);
    assert_eq!(frame.data(), &[1, 2, 3, 4, 5, 6]);
    assert_eq!(frame.into_bytes(), bytes);
}

#[test]
fn rgb8_bytes_to_frame_handles_vectorized_and_tail_segments() {
    let bytes = (0u8..18u8).collect::<Vec<_>>();
    let frame = rgb8_bytes_to_frame(9, 123, 3, 2, &bytes).unwrap();
    let data = frame.image().data();
    assert_eq!(data.len(), bytes.len());
    for (index, raw) in bytes.iter().copied().enumerate() {
        let expected = raw as f32 / 255.0;
        assert!((data[index] - expected).abs() < 1e-6);
    }
}

#[test]
fn normalize_rgb8_to_f32_inplace_rejects_buffer_size_mismatch() {
    let mut out = vec![0.0f32; 2];
    let err = normalize_rgb8_to_f32_inplace(&[1, 2, 3], &mut out).unwrap_err();
    assert_eq!(
        err,
        VideoError::NormalizedBufferSizeMismatch {
            expected: 3,
            got: 2,
        }
    );
}

#[test]
fn normalize_rgb8_to_f32_inplace_overwrites_reused_buffer() {
    let mut out = vec![0.0f32; 6];

    normalize_rgb8_to_f32_inplace(&[0, 64, 128, 192, 255, 32], &mut out).unwrap();
    let first = out.clone();
    assert!((first[0] - 0.0).abs() < 1e-6);
    assert!((first[1] - (64.0 / 255.0)).abs() < 1e-6);
    assert!((first[4] - 1.0).abs() < 1e-6);

    normalize_rgb8_to_f32_inplace(&[255, 0, 255, 0, 255, 0], &mut out).unwrap();
    assert!((out[0] - 1.0).abs() < 1e-6);
    assert!((out[1] - 0.0).abs() < 1e-6);
    assert!((out[2] - 1.0).abs() < 1e-6);
    assert!((out[3] - 0.0).abs() < 1e-6);
    assert!((out[4] - 1.0).abs() < 1e-6);
    assert!((out[5] - 0.0).abs() < 1e-6);
}

#[test]
fn select_camera_device_prefers_exact_match() {
    let devices = vec![
        CameraDeviceInfo {
            index: 0,
            label: "Laptop Camera".to_string(),
        },
        CameraDeviceInfo {
            index: 1,
            label: "USB Camera".to_string(),
        },
    ];
    let device = select_camera_device(&devices, "usb camera").unwrap();
    assert_eq!(device.index, 1);
}

#[test]
fn select_camera_device_supports_substring_match() {
    let devices = vec![
        CameraDeviceInfo {
            index: 2,
            label: "Front Sensor".to_string(),
        },
        CameraDeviceInfo {
            index: 3,
            label: "Studio Camera".to_string(),
        },
    ];
    let device = select_camera_device(&devices, "studio").unwrap();
    assert_eq!(device.index, 3);
}

#[test]
fn select_camera_device_rejects_ambiguous_query() {
    let devices = vec![
        CameraDeviceInfo {
            index: 2,
            label: "Front Sensor".to_string(),
        },
        CameraDeviceInfo {
            index: 3,
            label: "Studio Sensor".to_string(),
        },
    ];
    let err = select_camera_device(&devices, "sensor").unwrap_err();
    assert_eq!(
        err,
        VideoError::CameraDeviceAmbiguous {
            query: "sensor".to_string(),
            matches: vec![
                "2: Front Sensor".to_string(),
                "3: Studio Sensor".to_string(),
            ],
        }
    );
}

#[test]
fn select_camera_device_rejects_unknown_query() {
    let devices = vec![CameraDeviceInfo {
        index: 0,
        label: "Laptop Camera".to_string(),
    }];
    let err = select_camera_device(&devices, "external").unwrap_err();
    assert_eq!(
        err,
        VideoError::CameraDeviceNotFound {
            query: "external".to_string()
        }
    );
}

#[test]
fn select_camera_device_rejects_empty_query() {
    let devices = vec![CameraDeviceInfo {
        index: 0,
        label: "Laptop Camera".to_string(),
    }];
    let err = select_camera_device(&devices, "   ").unwrap_err();
    assert_eq!(
        err,
        VideoError::InvalidCameraDeviceQuery {
            query: "   ".to_string()
        }
    );
}

#[test]
fn select_camera_device_supports_numeric_query_by_index() {
    let devices = vec![
        CameraDeviceInfo {
            index: 3,
            label: "Studio Camera".to_string(),
        },
        CameraDeviceInfo {
            index: 7,
            label: "USB Camera".to_string(),
        },
    ];
    let device = select_camera_device(&devices, "7").unwrap();
    assert_eq!(device.index, 7);
    assert_eq!(device.label, "USB Camera");
}

#[test]
fn filter_camera_devices_supports_substring_query() {
    let devices = vec![
        CameraDeviceInfo {
            index: 0,
            label: "Laptop Camera".to_string(),
        },
        CameraDeviceInfo {
            index: 2,
            label: "Studio Camera".to_string(),
        },
        CameraDeviceInfo {
            index: 3,
            label: "USB Capture".to_string(),
        },
    ];
    let matches = filter_camera_devices(&devices, "studio").unwrap();
    assert_eq!(
        matches,
        vec![CameraDeviceInfo {
            index: 2,
            label: "Studio Camera".to_string(),
        }]
    );
}

#[test]
fn filter_camera_devices_supports_numeric_query() {
    let devices = vec![
        CameraDeviceInfo {
            index: 5,
            label: "Front Camera".to_string(),
        },
        CameraDeviceInfo {
            index: 7,
            label: "USB Camera".to_string(),
        },
    ];
    let matches = filter_camera_devices(&devices, "7").unwrap();
    assert_eq!(
        matches,
        vec![CameraDeviceInfo {
            index: 7,
            label: "USB Camera".to_string(),
        }]
    );
}

#[test]
fn filter_camera_devices_rejects_empty_query() {
    let devices = vec![CameraDeviceInfo {
        index: 0,
        label: "Laptop Camera".to_string(),
    }];
    let err = filter_camera_devices(&devices, "   ").unwrap_err();
    assert_eq!(
        err,
        VideoError::InvalidCameraDeviceQuery {
            query: "   ".to_string(),
        }
    );
}

#[test]
fn filter_camera_devices_returns_sorted_unique_matches() {
    let devices = vec![
        CameraDeviceInfo {
            index: 7,
            label: "USB Camera".to_string(),
        },
        CameraDeviceInfo {
            index: 2,
            label: "Front Camera".to_string(),
        },
        CameraDeviceInfo {
            index: 7,
            label: "USB Camera".to_string(),
        },
    ];
    let matches = filter_camera_devices(&devices, "camera").unwrap();
    assert_eq!(
        matches,
        vec![
            CameraDeviceInfo {
                index: 2,
                label: "Front Camera".to_string(),
            },
            CameraDeviceInfo {
                index: 7,
                label: "USB Camera".to_string(),
            },
        ]
    );
}

#[cfg(not(feature = "native-camera"))]
#[test]
fn camera_source_returns_disabled_error_without_feature() {
    let err = CameraFrameSource::open(CameraConfig::default()).unwrap_err();
    assert_eq!(err, VideoError::CameraBackendDisabled);

    let mut source = CameraFrameSource;
    let err = source.next_frame().unwrap_err();
    assert_eq!(err, VideoError::CameraBackendDisabled);
}

// ── Raw video I/O tests ────────────────────────────────────────────

use super::video_io::{RawVideoReader, RawVideoWriter};

#[test]
fn raw_video_roundtrip() {
    let dir = std::env::temp_dir().join("yscv_test_video_roundtrip");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("test.rcv");

    let mut writer = RawVideoWriter::new(2, 2, 30.0);
    let frame_data = vec![255u8; 12];
    writer.push_frame(&frame_data).unwrap();
    writer.push_frame(&frame_data).unwrap();
    writer.save(&path).unwrap();
    assert_eq!(writer.frame_count(), 2);

    let mut reader = RawVideoReader::open(&path).unwrap();
    assert_eq!(reader.frame_count(), 2);
    assert_eq!(reader.meta.width, 2);
    assert_eq!(reader.meta.height, 2);

    let f0 = reader.next_frame().unwrap();
    assert_eq!(f0.width(), 2);
    assert_eq!(f0.height(), 2);

    let f1 = reader.next_frame().unwrap();
    assert_eq!(f1.width(), 2);

    assert!(reader.next_frame().is_none());

    reader.seek_start();
    assert!(reader.next_frame().is_some());

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn raw_video_writer_rejects_wrong_size() {
    let mut writer = RawVideoWriter::new(2, 2, 30.0);
    let result = writer.push_frame(&[0u8; 11]);
    assert!(result.is_err());
}

#[test]
fn annex_b_parse_extracts_nal_units() {
    use super::codec::{NalUnitType, parse_annex_b};
    let mut stream = Vec::new();
    // SPS NAL
    stream.extend_from_slice(&[0x00, 0x00, 0x00, 0x01]);
    stream.push(0x67); // forbidden=0, ref_idc=3, type=7 (SPS)
    stream.extend_from_slice(&[0x42, 0x00, 0x1e]); // payload
    // PPS NAL
    stream.extend_from_slice(&[0x00, 0x00, 0x01]);
    stream.push(0x68); // forbidden=0, ref_idc=3, type=8 (PPS)
    stream.extend_from_slice(&[0xce, 0x38, 0x80]);
    // IDR NAL
    stream.extend_from_slice(&[0x00, 0x00, 0x00, 0x01]);
    stream.push(0x65); // forbidden=0, ref_idc=3, type=5 (IDR)
    stream.extend_from_slice(&[0x88, 0x84, 0x21]);

    let nals = parse_annex_b(&stream);
    assert_eq!(nals.len(), 3);
    assert_eq!(nals[0].nal_type, NalUnitType::Sps);
    assert_eq!(nals[1].nal_type, NalUnitType::Pps);
    assert_eq!(nals[2].nal_type, NalUnitType::Idr);
    assert!(nals[2].nal_type.is_vcl());
    assert!(!nals[0].nal_type.is_vcl());
}

#[test]
fn extract_sps_pps_from_nals() {
    use super::codec::{NalUnitType, extract_parameter_sets, parse_annex_b};
    let mut stream = Vec::new();
    stream.extend_from_slice(&[0x00, 0x00, 0x00, 0x01, 0x67, 0x42]);
    stream.extend_from_slice(&[0x00, 0x00, 0x01, 0x68, 0xce]);
    let nals = parse_annex_b(&stream);
    let (sps, pps) = extract_parameter_sets(&nals);
    assert!(sps.is_some());
    assert_eq!(sps.unwrap().nal_type, NalUnitType::Sps);
    assert!(pps.is_some());
    assert_eq!(pps.unwrap().nal_type, NalUnitType::Pps);
}

#[test]
fn mp4_box_parse_basic() {
    use super::codec::parse_mp4_boxes;
    let mut data = Vec::new();
    // ftyp box: size=12, type=ftyp, payload=[0x00; 4]
    data.extend_from_slice(&[0x00, 0x00, 0x00, 0x0C]); // size=12
    data.extend_from_slice(b"ftyp");
    data.extend_from_slice(&[0x00; 4]); // payload

    // moov box: size=8 (empty)
    data.extend_from_slice(&[0x00, 0x00, 0x00, 0x08]);
    data.extend_from_slice(b"moov");

    let boxes = parse_mp4_boxes(&data).unwrap();
    assert_eq!(boxes.len(), 2);
    assert_eq!(boxes[0].type_str(), "ftyp");
    assert_eq!(boxes[0].size, 12);
    assert_eq!(boxes[1].type_str(), "moov");
    assert_eq!(boxes[1].size, 8);
}

#[test]
fn extract_avcc_nals_parses_sps_pps() {
    // Build a synthetic moov chunk containing an avcC box
    let mut moov = Vec::new();

    // Some padding (like other boxes before avcC)
    moov.extend_from_slice(&[0x00; 20]);

    // "avcC" tag
    moov.extend_from_slice(b"avcC");

    // avcC config: version(1) + profile(66=baseline) + compat(0xC0) + level(30) + lengthSizeMinusOne(0xFF = 3+0xFC)
    moov.push(1); // configurationVersion
    moov.push(66); // AVCProfileIndication (Baseline)
    moov.push(0xC0); // profile_compatibility
    moov.push(30); // AVCLevelIndication (3.0)
    moov.push(0xFF); // lengthSizeMinusOne = 3 (lower 2 bits) + reserved bits

    // numSPS = 1 (lower 5 bits of 0xE1 = 1)
    moov.push(0xE1);

    // SPS: length=4, data=[0x67, 0x42, 0xC0, 0x1E] (typical SPS NAL header 0x67 = nal_type=7 SPS)
    let sps_data = [0x67, 0x42, 0xC0, 0x1E];
    moov.extend_from_slice(&(sps_data.len() as u16).to_be_bytes());
    moov.extend_from_slice(&sps_data);

    // numPPS = 1
    moov.push(1);

    // PPS: length=3, data=[0x68, 0xCE, 0x38] (typical PPS NAL header 0x68 = nal_type=8 PPS)
    let pps_data = [0x68, 0xCE, 0x38];
    moov.extend_from_slice(&(pps_data.len() as u16).to_be_bytes());
    moov.extend_from_slice(&pps_data);

    let nals = super::video_io::extract_avcc_nals(&moov);
    assert_eq!(nals.len(), 2, "should extract 1 SPS + 1 PPS");

    assert_eq!(nals[0].nal_type, super::codec::NalUnitType::Sps);
    assert_eq!(nals[0].data, sps_data);

    assert_eq!(nals[1].nal_type, super::codec::NalUnitType::Pps);
    assert_eq!(nals[1].data, pps_data);
}

#[test]
fn extract_avcc_nals_empty_on_no_avcc() {
    let moov = vec![0x00; 100]; // no avcC tag
    let nals = super::video_io::extract_avcc_nals(&moov);
    assert!(nals.is_empty());
}

#[test]
fn mp4_h264_decode_real_file() {
    // Integration test: decode a real H.264 MP4 if the test file exists
    let path = std::path::Path::new("/tmp/test_h264.mp4");
    if !path.exists() {
        // Skip test if file not present (created by ffmpeg in dev environment)
        return;
    }
    let mut reader =
        super::video_io::Mp4VideoReader::open(path).expect("should open H.264 MP4 without error");
    let nal_count = reader.nal_count();
    assert!(nal_count > 0, "should find NAL units, got 0");

    // Try to decode frames, collecting results
    let mut frames = Vec::new();
    let mut errors = Vec::new();
    for _ in 0..nal_count {
        match reader.next_frame() {
            Ok(Some(f)) => frames.push(f),
            Ok(None) => break,
            Err(e) => errors.push(format!("{e}")),
        }
    }

    // Debug: check what NAL types we got
    reader.seek_start();
    // We can't inspect NAL types directly but we know: 3 NALs = SPS + PPS + 1 video NAL
    // If no frames decoded, the video NAL is probably non-IDR Slice type

    assert!(
        !frames.is_empty(),
        "should decode at least one frame from {nal_count} NALs. Errors: {errors:?}"
    );
    let frame = &frames[0];

    assert!(frame.width > 0 && frame.height > 0, "valid dimensions");
    assert_eq!(frame.rgb8_data.len(), frame.width * frame.height * 3);

    // Verify it's not all-gray (which would mean decode failure)
    let min = frame.rgb8_data.iter().copied().min().unwrap_or(0);
    let max = frame.rgb8_data.iter().copied().max().unwrap_or(0);
    assert!(
        max > min,
        "frame should not be uniform gray — actual min={min} max={max}"
    );
}

#[test]
fn mp4_h264_high_profile_decode() {
    // Test H.264 High profile (CABAC) MP4
    let path = std::path::Path::new("/tmp/test_h264_high.mp4");
    if !path.exists() {
        return;
    }
    let mut reader =
        super::video_io::Mp4VideoReader::open(path).expect("should open H.264 High profile MP4");
    assert!(reader.nal_count() > 0);

    match reader.next_frame() {
        Ok(Some(frame)) => {
            assert!(frame.width > 0 && frame.height > 0);
            let min = frame.rgb8_data.iter().copied().min().unwrap_or(0);
            let max = frame.rgb8_data.iter().copied().max().unwrap_or(0);
            assert!(
                max > min,
                "CABAC frame should not be uniform — min={min} max={max}"
            );
        }
        Ok(None) => panic!("no frame decoded from High profile MP4"),
        Err(e) => panic!("decode error: {e}"),
    }
}

#[test]
fn video_codec_enum_basics() {
    use super::codec::VideoCodec;
    assert_eq!(VideoCodec::H264, VideoCodec::H264);
    assert_ne!(VideoCodec::H264, VideoCodec::H265);
}

// ── NV12 → RGB8 conversion tests ─────────────────────────────────────

#[test]
fn nv12_to_rgb8_pure_gray() {
    // Pure gray: Y=128, U=128, V=128 → R=G=B=128
    let w = 4;
    let h = 4;
    let y_plane = vec![128u8; w * h];
    // UV plane: interleaved U,V pairs, half height, same width
    let uv_plane = vec![128u8; w * (h / 2)];
    let mut out = vec![0u8; w * h * 3];

    super::h264_yuv::nv12_to_rgb8(&y_plane, &uv_plane, w, h, &mut out)
        .expect("NV12 conversion should succeed");

    for pixel in out.chunks_exact(3) {
        assert_eq!(pixel[0], 128, "R should be 128 for neutral gray");
        assert_eq!(pixel[1], 128, "G should be 128 for neutral gray");
        assert_eq!(pixel[2], 128, "B should be 128 for neutral gray");
    }
}

#[test]
fn nv12_to_rgb8_rejects_small_y_plane() {
    let mut out = vec![0u8; 4 * 4 * 3];
    let result = super::h264_yuv::nv12_to_rgb8(&[0; 8], &[128; 8], 4, 4, &mut out);
    assert!(result.is_err());
}

#[test]
fn nv12_to_rgb8_rejects_small_uv_plane() {
    let mut out = vec![0u8; 4 * 4 * 3];
    let result = super::h264_yuv::nv12_to_rgb8(&[128; 16], &[128; 2], 4, 4, &mut out);
    assert!(result.is_err());
}

#[test]
fn nv12_to_rgb8_rejects_small_output() {
    let result = super::h264_yuv::nv12_to_rgb8(&[128; 16], &[128; 8], 4, 4, &mut [0; 10]);
    assert!(result.is_err());
}

#[test]
fn nv12_to_rgb8_non_trivial_values() {
    // 2x2 image with specific values
    let w = 2;
    let h = 2;
    let y_plane = [200u8, 100, 150, 50];
    let uv_plane = [100u8, 200]; // one UV pair covers all 4 pixels
    let mut out = vec![0u8; w * h * 3];

    super::h264_yuv::nv12_to_rgb8(&y_plane, &uv_plane, w, h, &mut out)
        .expect("NV12 conversion should succeed");

    // Verify output is not all zeros (non-trivial conversion happened)
    let sum: u32 = out.iter().map(|&b| b as u32).sum();
    assert!(sum > 0, "output should not be all zeros");

    // Verify output contains distinct values (not all same, indicating real conversion)
    let distinct: std::collections::HashSet<u8> = out.iter().copied().collect();
    assert!(distinct.len() > 1, "output should contain distinct values");
}

// ── YUYV → RGB8 conversion tests ─────────────────────────────────────

#[test]
fn yuyv_to_rgb8_pure_gray() {
    // YUYV: Y0=128, U=128, Y1=128, V=128 → all pixels R=G=B=128
    let w = 4;
    let h = 2;
    let mut data = Vec::with_capacity(w * h * 2);
    for _ in 0..(w * h / 2) {
        data.extend_from_slice(&[128, 128, 128, 128]); // Y0, U, Y1, V
    }
    let mut out = vec![0u8; w * h * 3];

    super::h264_yuv::yuyv_to_rgb8(&data, w, h, &mut out).expect("YUYV conversion should succeed");

    for pixel in out.chunks_exact(3) {
        assert_eq!(pixel[0], 128, "R should be 128 for neutral gray");
        assert_eq!(pixel[1], 128, "G should be 128 for neutral gray");
        assert_eq!(pixel[2], 128, "B should be 128 for neutral gray");
    }
}

#[test]
fn yuyv_to_rgb8_rejects_odd_width() {
    let result = super::h264_yuv::yuyv_to_rgb8(&[0; 6], 3, 1, &mut [0; 9]);
    assert!(result.is_err());
}

#[test]
fn yuyv_to_rgb8_rejects_small_input() {
    let result = super::h264_yuv::yuyv_to_rgb8(&[0; 4], 4, 2, &mut [0; 24]);
    assert!(result.is_err());
}

#[test]
fn yuyv_to_rgb8_rejects_small_output() {
    let result = super::h264_yuv::yuyv_to_rgb8(&[128; 16], 4, 2, &mut [0; 10]);
    assert!(result.is_err());
}

#[test]
fn yuyv_to_rgb8_two_pixel_pair() {
    // Single macro-pixel: Y0=235, U=128, Y1=16, V=128 (neutral chroma)
    // Y=235 → R=G=B=235 (white-ish), Y=16 → R=G=B=16 (dark)
    let data = [235u8, 128, 16, 128];
    let mut out = vec![0u8; 6]; // 2 pixels * 3 channels

    super::h264_yuv::yuyv_to_rgb8(&data, 2, 1, &mut out).expect("YUYV conversion should succeed");

    // With neutral chroma (U=V=128), output should be close to Y values
    assert_eq!(out[0], out[1]); // R == G for pixel 0
    assert_eq!(out[1], out[2]); // G == B for pixel 0
    assert!(
        (out[0] as i16 - 235).abs() <= 1,
        "pixel 0 should be ~235, got {}",
        out[0]
    );

    assert_eq!(out[3], out[4]); // R == G for pixel 1
    assert_eq!(out[4], out[5]); // G == B for pixel 1
    assert!(
        (out[3] as i16 - 16).abs() <= 1,
        "pixel 1 should be ~16, got {}",
        out[3]
    );
}

// ── MJPEG decode tests ────────────────────────────────────────────────

#[test]
fn mjpeg_decode_rejects_non_jpeg() {
    let result = super::mjpeg::decode_mjpeg_to_rgb8(&[0x89, 0x50, 0x4E, 0x47], &mut Vec::new());
    assert!(
        result.is_err(),
        "should reject non-JPEG data (PNG signature)"
    );
}

// ── Edge pipeline end-to-end test ─────────────────────────────────────

#[test]
fn edge_pipeline_end_to_end() {
    use super::frame_pipeline::{FramePipeline, PipelineBBox, PipelineDetection, run_pipeline};
    use super::h264_encoder::{H264Encoder, rgb8_to_yuv420};
    use super::h264_yuv::yuv420_to_rgb8;
    use super::overlay::{TelemetryData, overlay_detections, overlay_telemetry};
    use std::sync::atomic::{AtomicUsize, Ordering};

    let width: usize = 320;
    let height: usize = 240;
    // Width and height must be multiples of 16 for H.264 — 320 and 240 already satisfy this.
    let rgb_bytes = width * height * 3;
    let total_frames = 3usize;

    // 1. Create FramePipeline
    let pipeline = FramePipeline::new(4, rgb_bytes);
    let output_count = AtomicUsize::new(0);
    let encoded_total_size = AtomicUsize::new(0);
    let annex_b_valid = AtomicUsize::new(0);

    run_pipeline(
        &pipeline,
        // Stage 1: Generate synthetic YUV420 frame, convert to RGB, write to slot
        |slot: &mut super::frame_pipeline::SlotMut<'_>| {
            let _ts = slot.timestamp_us(); // unused but available
            static COUNTER: AtomicUsize = AtomicUsize::new(0);
            let frame_idx = COUNTER.fetch_add(1, Ordering::Relaxed);
            if frame_idx >= total_frames {
                return false;
            }

            slot.set_width(width as u32);
            slot.set_height(height as u32);
            slot.set_pixel_format(2); // RGB8
            slot.set_timestamp_us(frame_idx as u64 * 33_333);

            // Generate synthetic YUV420 planar data
            let y_size = width * height;
            let uv_size = (width / 2) * (height / 2);
            let mut y_plane = vec![0u8; y_size];
            let u_plane = vec![128u8; uv_size];
            let v_plane = vec![128u8; uv_size];

            // Fill Y plane with gradient pattern
            for py in 0..height {
                for px in 0..width {
                    y_plane[py * width + px] =
                        ((px * 255 / width.max(1) + py * 255 / height.max(1)) / 2) as u8;
                }
            }

            // Convert YUV420 to RGB8
            let rgb = yuv420_to_rgb8(&y_plane, &u_plane, &v_plane, width, height)
                .expect("YUV420 to RGB conversion must succeed");

            // Write RGB data into the slot
            let data = slot.data_mut();
            let copy_len = rgb.len().min(data.len());
            data[..copy_len].copy_from_slice(&rgb[..copy_len]);

            true
        },
        // Stage 2: Create mock detections, overlay them
        |slot: &mut super::frame_pipeline::SlotMut<'_>| {
            let sw = slot.width() as usize;
            let sh = slot.height() as usize;

            // Mock detections
            slot.detections_mut().clear();
            slot.detections_mut().push(PipelineDetection {
                bbox: PipelineBBox {
                    x1: 50.0,
                    y1: 30.0,
                    x2: 130.0,
                    y2: 110.0,
                },
                score: 0.92,
                class_id: 0,
            });
            slot.detections_mut().push(PipelineDetection {
                bbox: PipelineBBox {
                    x1: 200.0,
                    y1: 100.0,
                    x2: 280.0,
                    y2: 200.0,
                },
                score: 0.87,
                class_id: 1,
            });

            // Build overlay data from detections before taking data_mut
            let det_labels: Vec<String> = slot
                .detections()
                .iter()
                .map(|d| format!("cls{}", d.class_id))
                .collect();
            let overlay_dets: Vec<(f32, f32, f32, f32, f32, String)> = slot
                .detections()
                .iter()
                .zip(det_labels.iter())
                .map(|(d, label)| {
                    (
                        d.bbox.x1,
                        d.bbox.y1,
                        d.bbox.x2 - d.bbox.x1,
                        d.bbox.y2 - d.bbox.y1,
                        d.score,
                        label.clone(),
                    )
                })
                .collect();
            let num_dets = slot.detections().len() as u32;

            // Now take mutable access for overlays
            let frame_size = sw * sh * 3;
            let data = slot.data_mut();
            if data.len() >= frame_size {
                let det_refs: Vec<(f32, f32, f32, f32, f32, &str)> = overlay_dets
                    .iter()
                    .map(|(x, y, w, h, s, l)| (*x, *y, *w, *h, *s, l.as_str()))
                    .collect();
                overlay_detections(&mut data[..frame_size], sw, sh, &det_refs);
            }

            // Overlay telemetry (mock)
            let telemetry = TelemetryData {
                battery_voltage: 12.4,
                battery_current: 1.2,
                altitude_m: 45.0,
                speed_ms: 5.3,
                lat: 55.7558,
                lon: 37.6173,
                heading_deg: 127.0,
                ai_detections: num_dets,
            };
            if data.len() >= frame_size {
                overlay_telemetry(&mut data[..frame_size], sw, sh, &telemetry);
            }
        },
        // Stage 3: Convert RGB to YUV420, encode H.264, verify output
        |slot: &super::frame_pipeline::SlotRef<'_>| {
            let sw = slot.width() as usize;
            let sh = slot.height() as usize;
            let frame_size = sw * sh * 3;

            // Copy frame data
            let src = slot.data();
            let mut rgb_frame = vec![0u8; frame_size];
            let copy_len = frame_size.min(src.len());
            rgb_frame[..copy_len].copy_from_slice(&src[..copy_len]);

            // Convert RGB to YUV420
            let yuv = rgb8_to_yuv420(&rgb_frame, sw, sh);
            assert_eq!(
                yuv.len(),
                sw * sh + (sw / 2) * (sh / 2) * 2,
                "YUV420 size must match expected"
            );

            // Encode with H264Encoder
            // Use a static-ish encoder: since output runs serially, this is safe
            let mut encoder = H264Encoder::new(sw as u32, sh as u32, 26);
            let nal_data = encoder.encode_frame(&yuv);

            // Verify output size > 0
            assert!(
                !nal_data.is_empty(),
                "H.264 encoded output must not be empty"
            );

            // Verify output starts with Annex B start code (0x00 0x00 0x00 0x01)
            if nal_data.len() >= 4
                && nal_data[0] == 0x00
                && nal_data[1] == 0x00
                && nal_data[2] == 0x00
                && nal_data[3] == 0x01
            {
                annex_b_valid.fetch_add(1, Ordering::Relaxed);
            }

            encoded_total_size.fetch_add(nal_data.len(), Ordering::Relaxed);
            output_count.fetch_add(1, Ordering::Relaxed);
        },
        total_frames,
    );

    // Final assertions
    assert_eq!(
        output_count.load(Ordering::Relaxed),
        total_frames,
        "all frames must be processed"
    );
    assert!(
        encoded_total_size.load(Ordering::Relaxed) > 0,
        "total encoded size must be > 0"
    );
    assert_eq!(
        annex_b_valid.load(Ordering::Relaxed),
        total_frames,
        "every encoded frame must start with Annex B start code"
    );
}
