use yscv_video::Frame;

use crate::heatmap::{HeatmapDetectScratch, detect_from_heatmap_data_with_scratch, map_shape};
use crate::nms::validate_nms_args;
use crate::{CLASS_ID_FACE, DetectError, Detection, non_max_suppression};

/// Reusable scratch storage for RGB8 people detection.
///
/// This allows callers with stable frame dimensions (for example camera loops)
/// to avoid reallocating grayscale heatmap buffers on each frame.
#[derive(Debug, Default, Clone, PartialEq)]
pub struct Rgb8PeopleDetectScratch {
    grayscale_heatmap: Vec<f32>,
    heatmap: HeatmapDetectScratch,
}

/// Reusable scratch storage for RGB8 face detection.
///
/// This allows callers with stable frame dimensions (for example camera loops)
/// to avoid reallocating skin-probability heatmap buffers on each frame.
#[derive(Debug, Default, Clone, PartialEq)]
pub struct Rgb8FaceDetectScratch {
    skin_heatmap: Vec<f32>,
    heatmap: HeatmapDetectScratch,
}

/// Reusable scratch storage for frame-based people detection.
///
/// This avoids per-frame grayscale heatmap and traversal-buffer allocations
/// when detection runs on [`Frame`] inputs.
#[derive(Debug, Default, Clone, PartialEq)]
pub struct FramePeopleDetectScratch {
    grayscale_heatmap: Vec<f32>,
    heatmap: HeatmapDetectScratch,
}

/// Reusable scratch storage for frame-based face detection.
///
/// This avoids per-frame skin heatmap and traversal-buffer allocations
/// when detection runs on [`Frame`] inputs.
#[derive(Debug, Default, Clone, PartialEq)]
pub struct FrameFaceDetectScratch {
    skin_heatmap: Vec<f32>,
    heatmap: HeatmapDetectScratch,
}

/// Convenience adapter from frame to heatmap-based detection.
pub fn detect_people_from_frame(
    frame: &Frame,
    score_threshold: f32,
    min_area: usize,
    iou_threshold: f32,
    max_detections: usize,
) -> Result<Vec<Detection>, DetectError> {
    let mut scratch = FramePeopleDetectScratch::default();
    detect_people_from_frame_with_scratch(
        frame,
        score_threshold,
        min_area,
        iou_threshold,
        max_detections,
        &mut scratch,
    )
}

/// Convenience adapter from frame to heatmap-based detection with reusable scratch storage.
pub fn detect_people_from_frame_with_scratch(
    frame: &Frame,
    score_threshold: f32,
    min_area: usize,
    iou_threshold: f32,
    max_detections: usize,
    scratch: &mut FramePeopleDetectScratch,
) -> Result<Vec<Detection>, DetectError> {
    let image = frame.image();
    let (h, w, c) = map_shape(image)?;
    match c {
        1 => detect_from_heatmap_data_with_scratch(
            (h, w),
            image.data(),
            score_threshold,
            min_area,
            iou_threshold,
            max_detections,
            &mut scratch.heatmap,
        ),
        3 => {
            fill_frame_rgb_grayscale_heatmap((h, w), image.data(), &mut scratch.grayscale_heatmap);
            detect_from_heatmap_data_with_scratch(
                (h, w),
                &scratch.grayscale_heatmap,
                score_threshold,
                min_area,
                iou_threshold,
                max_detections,
                &mut scratch.heatmap,
            )
        }
        other => Err(DetectError::InvalidChannelCount {
            expected: 1,
            got: other,
        }),
    }
}

/// People detector over raw RGB8 bytes.
///
/// This bypasses frame-tensor conversion and is useful for camera paths
/// where RGB8 bytes are available directly from the capture backend.
pub fn detect_people_from_rgb8(
    width: usize,
    height: usize,
    rgb8: &[u8],
    score_threshold: f32,
    min_area: usize,
    iou_threshold: f32,
    max_detections: usize,
) -> Result<Vec<Detection>, DetectError> {
    let mut scratch = Rgb8PeopleDetectScratch::default();
    detect_people_from_rgb8_with_scratch(
        (width, height),
        rgb8,
        score_threshold,
        min_area,
        iou_threshold,
        max_detections,
        &mut scratch,
    )
}

/// People detector over raw RGB8 bytes with reusable scratch storage.
pub fn detect_people_from_rgb8_with_scratch(
    shape: (usize, usize),
    rgb8: &[u8],
    score_threshold: f32,
    min_area: usize,
    iou_threshold: f32,
    max_detections: usize,
    scratch: &mut Rgb8PeopleDetectScratch,
) -> Result<Vec<Detection>, DetectError> {
    let (width, height) = shape;
    fill_rgb8_grayscale_heatmap(width, height, rgb8, &mut scratch.grayscale_heatmap)?;
    detect_from_heatmap_data_with_scratch(
        (height, width),
        &scratch.grayscale_heatmap,
        score_threshold,
        min_area,
        iou_threshold,
        max_detections,
        &mut scratch.heatmap,
    )
}

/// Heuristic face detector over RGB frames using a skin-probability heatmap.
///
/// This is a classical CV baseline that does not require a trained model and is
/// intended for camera demos where low-latency face regions are needed.
pub fn detect_faces_from_frame(
    frame: &Frame,
    score_threshold: f32,
    min_area: usize,
    iou_threshold: f32,
    max_detections: usize,
) -> Result<Vec<Detection>, DetectError> {
    let mut scratch = FrameFaceDetectScratch::default();
    detect_faces_from_frame_with_scratch(
        frame,
        score_threshold,
        min_area,
        iou_threshold,
        max_detections,
        &mut scratch,
    )
}

/// Heuristic face detector over RGB frames with reusable scratch storage.
pub fn detect_faces_from_frame_with_scratch(
    frame: &Frame,
    score_threshold: f32,
    min_area: usize,
    iou_threshold: f32,
    max_detections: usize,
    scratch: &mut FrameFaceDetectScratch,
) -> Result<Vec<Detection>, DetectError> {
    validate_nms_args(iou_threshold, max_detections)?;
    let image = frame.image();
    let (h, w, c) = map_shape(image)?;
    if c != 3 {
        return Err(DetectError::InvalidChannelCount {
            expected: 3,
            got: c,
        });
    }

    fill_frame_rgb_skin_heatmap((h, w), image.data(), &mut scratch.skin_heatmap);
    detect_faces_from_skin_heatmap_data_with_scratch(
        (h, w),
        &scratch.skin_heatmap,
        score_threshold,
        min_area,
        iou_threshold,
        max_detections,
        &mut scratch.heatmap,
    )
}

/// Heuristic face detector over raw RGB8 bytes.
///
/// This bypasses temporary tensor conversion and is useful for camera paths
/// where frame bytes are available directly from the capture backend.
pub fn detect_faces_from_rgb8(
    width: usize,
    height: usize,
    rgb8: &[u8],
    score_threshold: f32,
    min_area: usize,
    iou_threshold: f32,
    max_detections: usize,
) -> Result<Vec<Detection>, DetectError> {
    let mut scratch = Rgb8FaceDetectScratch::default();
    detect_faces_from_rgb8_with_scratch(
        (width, height),
        rgb8,
        score_threshold,
        min_area,
        iou_threshold,
        max_detections,
        &mut scratch,
    )
}

/// Heuristic face detector over raw RGB8 bytes with reusable scratch storage.
pub fn detect_faces_from_rgb8_with_scratch(
    shape: (usize, usize),
    rgb8: &[u8],
    score_threshold: f32,
    min_area: usize,
    iou_threshold: f32,
    max_detections: usize,
    scratch: &mut Rgb8FaceDetectScratch,
) -> Result<Vec<Detection>, DetectError> {
    let (width, height) = shape;
    validate_nms_args(iou_threshold, max_detections)?;
    fill_rgb8_skin_heatmap(width, height, rgb8, &mut scratch.skin_heatmap)?;
    detect_faces_from_skin_heatmap_data_with_scratch(
        (height, width),
        &scratch.skin_heatmap,
        score_threshold,
        min_area,
        iou_threshold,
        max_detections,
        &mut scratch.heatmap,
    )
}

fn detect_faces_from_skin_heatmap_data_with_scratch(
    shape: (usize, usize),
    skin_heatmap_data: &[f32],
    score_threshold: f32,
    min_area: usize,
    iou_threshold: f32,
    max_detections: usize,
    heatmap_scratch: &mut HeatmapDetectScratch,
) -> Result<Vec<Detection>, DetectError> {
    let candidate_limit = max_detections.saturating_mul(4).max(max_detections);
    let candidates = detect_from_heatmap_data_with_scratch(
        shape,
        skin_heatmap_data,
        score_threshold,
        min_area,
        iou_threshold,
        candidate_limit,
        heatmap_scratch,
    )?;

    let mut faces = Vec::with_capacity(candidates.len());
    for candidate in candidates {
        let height = candidate.bbox.height();
        if height <= 1.0e-6 {
            continue;
        }
        let aspect_ratio = candidate.bbox.width() / height;
        if !(0.65..=1.8).contains(&aspect_ratio) {
            continue;
        }

        let shape_score = triangular_score(aspect_ratio, 0.65, 1.8, 1.0);
        let score = clamp01(0.75 * candidate.score + 0.25 * shape_score);
        faces.push(Detection {
            bbox: candidate.bbox,
            score,
            class_id: CLASS_ID_FACE,
        });
    }

    Ok(non_max_suppression(&faces, iou_threshold, max_detections))
}

fn fill_frame_rgb_grayscale_heatmap(shape: (usize, usize), rgb: &[f32], out: &mut Vec<f32>) {
    let pixel_count = shape.0 * shape.1;
    if out.len() != pixel_count {
        out.resize(pixel_count, 0.0);
    }

    for (rgb, value) in rgb.as_chunks::<3>().0.iter().zip(out.iter_mut()) {
        *value = (rgb[0] + rgb[1] + rgb[2]) / 3.0;
    }
}

fn fill_frame_rgb_skin_heatmap(shape: (usize, usize), rgb: &[f32], out: &mut Vec<f32>) {
    let pixel_count = shape.0 * shape.1;
    if out.len() != pixel_count {
        out.resize(pixel_count, 0.0);
    }

    let max_value = rgb.iter().copied().fold(0.0f32, f32::max);
    let scale = if max_value > 1.5 { 1.0 / 255.0 } else { 1.0 };
    for (rgb, value) in rgb.as_chunks::<3>().0.iter().zip(out.iter_mut()) {
        let r = clamp01(rgb[0] * scale);
        let g = clamp01(rgb[1] * scale);
        let b = clamp01(rgb[2] * scale);
        *value = skin_probability(r, g, b);
    }
}

fn fill_rgb8_skin_heatmap(
    width: usize,
    height: usize,
    rgb8: &[u8],
    out: &mut Vec<f32>,
) -> Result<(), DetectError> {
    validate_rgb8_buffer_size(width, height, rgb8)?;
    let pixel_count = width
        .checked_mul(height)
        .ok_or(DetectError::Rgb8DimensionsOverflow { width, height })?;
    if out.len() != pixel_count {
        out.resize(pixel_count, 0.0);
    }

    const SCALE: f32 = 1.0 / 255.0;
    for (rgb, value) in rgb8.as_chunks::<3>().0.iter().zip(out.iter_mut()) {
        let r = rgb[0] as f32 * SCALE;
        let g = rgb[1] as f32 * SCALE;
        let b = rgb[2] as f32 * SCALE;
        *value = skin_probability(r, g, b);
    }
    Ok(())
}

fn fill_rgb8_grayscale_heatmap(
    width: usize,
    height: usize,
    rgb8: &[u8],
    out: &mut Vec<f32>,
) -> Result<(), DetectError> {
    validate_rgb8_buffer_size(width, height, rgb8)?;
    let pixel_count = width
        .checked_mul(height)
        .ok_or(DetectError::Rgb8DimensionsOverflow { width, height })?;
    if out.len() != pixel_count {
        out.resize(pixel_count, 0.0);
    }

    const SCALE: f32 = 1.0 / 255.0;
    for (rgb, value) in rgb8.as_chunks::<3>().0.iter().zip(out.iter_mut()) {
        *value = (rgb[0] as f32 + rgb[1] as f32 + rgb[2] as f32) * (SCALE / 3.0);
    }
    Ok(())
}

fn validate_rgb8_buffer_size(width: usize, height: usize, rgb8: &[u8]) -> Result<(), DetectError> {
    let expected = width
        .checked_mul(height)
        .and_then(|pixels| pixels.checked_mul(3))
        .ok_or(DetectError::Rgb8DimensionsOverflow { width, height })?;
    if rgb8.len() != expected {
        return Err(DetectError::InvalidRgb8BufferSize {
            expected,
            got: rgb8.len(),
        });
    }
    Ok(())
}

fn skin_probability(r: f32, g: f32, b: f32) -> f32 {
    let y = 0.299 * r + 0.587 * g + 0.114 * b;
    let cb = 0.5 + 0.564 * (b - y);
    let cr = 0.5 + 0.713 * (r - y);

    let cb_score = triangular_score(cb, 0.28, 0.57, 0.43);
    let cr_score = triangular_score(cr, 0.36, 0.76, 0.56);
    let luminance_score = triangular_score(y, 0.08, 0.95, 0.55);

    let rg_bias = clamp01((r - g + 0.15) / 0.35);
    let gb_bias = clamp01((g - b + 0.10) / 0.35);
    let chroma = ((r - g).abs() + (g - b).abs() + (r - b).abs()) / 3.0;
    let saturation_score = clamp01(chroma / 0.45);

    let score = 0.32 * cb_score
        + 0.32 * cr_score
        + 0.16 * luminance_score
        + 0.10 * rg_bias
        + 0.10 * gb_bias;
    clamp01(score * saturation_score.max(0.3))
}

fn triangular_score(value: f32, min: f32, max: f32, center: f32) -> f32 {
    if value < min || value > max {
        return 0.0;
    }
    if (value - center).abs() <= f32::EPSILON {
        return 1.0;
    }
    if value < center {
        return (value - min) / (center - min);
    }
    (max - value) / (max - center)
}

fn clamp01(value: f32) -> f32 {
    value.clamp(0.0, 1.0)
}
