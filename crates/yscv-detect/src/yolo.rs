//! YOLOv8 postprocessing pipeline.
//!
//! Decodes raw YOLOv8 model output tensors into [`Detection`]s. The actual
//! model inference is left to the caller (e.g. via `yscv-onnx`); this module
//! handles the coordinate decoding, confidence filtering, and NMS step.

use yscv_tensor::Tensor;

use crate::{BoundingBox, Detection, non_max_suppression};

/// YOLO model configuration.
#[derive(Debug, Clone)]
pub struct YoloConfig {
    /// Input image size (square).
    pub input_size: usize,
    /// Number of classes.
    pub num_classes: usize,
    /// Confidence threshold.
    pub conf_threshold: f32,
    /// IoU threshold for NMS.
    pub iou_threshold: f32,
    /// Class labels.
    pub class_labels: Vec<String>,
}

/// Returns the 80 COCO class labels.
#[rustfmt::skip]
pub fn coco_labels() -> Vec<String> {
    [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
        "truck", "boat", "traffic light", "fire hydrant", "stop sign",
        "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
        "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
        "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
        "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
        "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
        "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
        "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
        "couch", "potted plant", "bed", "dining table", "toilet", "tv",
        "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
        "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
        "scissors", "teddy bear", "hair drier", "toothbrush",
    ]
    .iter()
    .map(|s| (*s).to_string())
    .collect()
}

/// Returns the default YOLOv8 config for the COCO 80-class dataset.
pub fn yolov8_coco_config() -> YoloConfig {
    YoloConfig {
        input_size: 640,
        num_classes: 80,
        conf_threshold: 0.25,
        iou_threshold: 0.45,
        class_labels: coco_labels(),
    }
}

/// Decode YOLOv8 raw output tensor into detections.
///
/// YOLOv8 output format: `[1, 4 + num_classes, num_preds]` where the first
/// four rows are `(cx, cy, w, h)` and the remaining rows are per-class
/// confidence scores. A typical COCO model emits `[1, 84, 8400]`.
///
/// Coordinates in the output are relative to the **letterboxed** input image
/// (i.e. `input_size x input_size`). This function maps them back to the
/// original `(orig_width, orig_height)` frame.
///
/// Returns filtered detections after confidence thresholding and NMS.
pub fn decode_yolov8_output(
    output: &Tensor,
    config: &YoloConfig,
    orig_width: usize,
    orig_height: usize,
) -> Vec<Detection> {
    let shape = output.shape();
    // Expect [1, 4+num_classes, num_preds]
    if shape.len() != 3 || shape[0] != 1 {
        return Vec::new();
    }
    let rows = shape[1]; // 4 + num_classes
    let num_preds = shape[2];
    if rows < 5 {
        return Vec::new();
    }
    let num_classes = rows - 4;

    let data = output.data();

    // Compute letterbox scale and padding so we can map coords back.
    let scale = (config.input_size as f32 / orig_width as f32)
        .min(config.input_size as f32 / orig_height as f32);
    let new_w = orig_width as f32 * scale;
    let new_h = orig_height as f32 * scale;
    let pad_x = (config.input_size as f32 - new_w) / 2.0;
    let pad_y = (config.input_size as f32 - new_h) / 2.0;

    let mut candidates = Vec::new();

    for i in 0..num_preds {
        // Output is laid out row-major: data[row * num_preds + col]
        let cx = data[i];
        let cy = data[num_preds + i];
        let w = data[2 * num_preds + i];
        let h = data[3 * num_preds + i];

        // Find best class
        let mut best_score = f32::NEG_INFINITY;
        let mut best_class = 0usize;
        for c in 0..num_classes {
            let s = data[(4 + c) * num_preds + i];
            if s > best_score {
                best_score = s;
                best_class = c;
            }
        }

        if best_score < config.conf_threshold {
            continue;
        }

        // Convert from letterbox coordinates to original image coordinates.
        let x1 = ((cx - w / 2.0) - pad_x) / scale;
        let y1 = ((cy - h / 2.0) - pad_y) / scale;
        let x2 = ((cx + w / 2.0) - pad_x) / scale;
        let y2 = ((cy + h / 2.0) - pad_y) / scale;

        // Clamp to image bounds.
        let x1 = x1.max(0.0).min(orig_width as f32);
        let y1 = y1.max(0.0).min(orig_height as f32);
        let x2 = x2.max(0.0).min(orig_width as f32);
        let y2 = y2.max(0.0).min(orig_height as f32);

        candidates.push(Detection {
            bbox: BoundingBox { x1, y1, x2, y2 },
            score: best_score,
            class_id: best_class,
        });
    }

    non_max_suppression(&candidates, config.iou_threshold, candidates.len().max(1))
}

/// Default COCO config for YOLOv11 models.
///
/// Same as YOLOv8 COCO — 80 classes, 640 input, 0.25 conf, 0.45 IoU.
pub fn yolov11_coco_config() -> YoloConfig {
    yolov8_coco_config()
}

/// Decode YOLOv11 output tensor into detections.
///
/// YOLOv11 output shape: `[1, num_preds, 4 + num_classes]` (transposed vs YOLOv8).
/// Each row is `[cx, cy, w, h, class_0_score, class_1_score, ...]`.
/// Coordinates are in letterboxed image space; this function maps them back
/// to the original image dimensions.
pub fn decode_yolov11_output(
    output: &Tensor,
    config: &YoloConfig,
    orig_width: usize,
    orig_height: usize,
) -> Vec<Detection> {
    let shape = output.shape();
    // YOLOv11: [1, N, 4+C] or [N, 4+C]
    let (num_preds, cols) = if shape.len() == 3 {
        (shape[1], shape[2])
    } else if shape.len() == 2 {
        (shape[0], shape[1])
    } else {
        return Vec::new();
    };

    if cols < 5 {
        return Vec::new();
    }
    let num_classes = cols - 4;

    let data = output.data();

    let scale = (config.input_size as f32 / orig_width as f32)
        .min(config.input_size as f32 / orig_height as f32);
    let new_w = orig_width as f32 * scale;
    let new_h = orig_height as f32 * scale;
    let pad_x = (config.input_size as f32 - new_w) / 2.0;
    let pad_y = (config.input_size as f32 - new_h) / 2.0;

    let mut candidates = Vec::new();

    // Skip batch dimension offset if present
    let base = if shape.len() == 3 { 0 } else { 0 };

    for i in 0..num_preds {
        let row = base + i * cols;
        let cx = data[row];
        let cy = data[row + 1];
        let w = data[row + 2];
        let h = data[row + 3];

        let mut best_score = f32::NEG_INFINITY;
        let mut best_class = 0usize;
        for c in 0..num_classes {
            let s = data[row + 4 + c];
            if s > best_score {
                best_score = s;
                best_class = c;
            }
        }

        if best_score < config.conf_threshold {
            continue;
        }

        let x1 = ((cx - w / 2.0) - pad_x) / scale;
        let y1 = ((cy - h / 2.0) - pad_y) / scale;
        let x2 = ((cx + w / 2.0) - pad_x) / scale;
        let y2 = ((cy + h / 2.0) - pad_y) / scale;

        let x1 = x1.max(0.0).min(orig_width as f32);
        let y1 = y1.max(0.0).min(orig_height as f32);
        let x2 = x2.max(0.0).min(orig_width as f32);
        let y2 = y2.max(0.0).min(orig_height as f32);

        candidates.push(Detection {
            bbox: BoundingBox { x1, y1, x2, y2 },
            score: best_score,
            class_id: best_class,
        });
    }

    non_max_suppression(&candidates, config.iou_threshold, candidates.len().max(1))
}

/// Apply letterbox preprocessing: resize an image to a square with padding.
///
/// The input `image` is an `[H, W, 3]` f32 tensor (RGB, normalised to 0..1).
/// Returns `(padded, scale, pad_x, pad_y)` where `padded` has shape
/// `[target_size, target_size, 3]`.
pub fn letterbox_preprocess(image: &Tensor, target_size: usize) -> (Tensor, f32, f32, f32) {
    let shape = image.shape();
    assert!(
        shape.len() == 3 && shape[2] == 3,
        "expected [H, W, 3] tensor"
    );
    let src_h = shape[0];
    let src_w = shape[1];
    let data = image.data();

    let scale = (target_size as f32 / src_w as f32).min(target_size as f32 / src_h as f32);
    let new_w = (src_w as f32 * scale).round() as usize;
    let new_h = (src_h as f32 * scale).round() as usize;
    let pad_x = (target_size - new_w) as f32 / 2.0;
    let pad_y = (target_size - new_h) as f32 / 2.0;
    let pad_left = pad_x.floor() as usize;
    let pad_top = pad_y.floor() as usize;

    // Fill with 0.5 grey (common YOLO letterbox fill).
    let total = target_size * target_size * 3;
    let mut out = vec![0.5f32; total];

    // Nearest-neighbour resize into the padded region.
    let scale_x = src_w as f32 / new_w as f32;
    let scale_y = src_h as f32 / new_h as f32;

    for y in 0..new_h {
        let src_y = ((y as f32 * scale_y) as usize).min(src_h - 1);
        for x in 0..new_w {
            let src_x = ((x as f32 * scale_x) as usize).min(src_w - 1);
            let dst_idx = ((pad_top + y) * target_size + (pad_left + x)) * 3;
            let src_idx = (src_y * src_w + src_x) * 3;
            out[dst_idx] = data[src_idx];
            out[dst_idx + 1] = data[src_idx + 1];
            out[dst_idx + 2] = data[src_idx + 2];
        }
    }

    let tensor = Tensor::from_vec(vec![target_size, target_size, 3], out)
        .expect("letterbox output tensor creation");
    (tensor, scale, pad_x, pad_y)
}

/// Convert an `[H, W, 3]` HWC tensor to `[1, 3, H, W]` NCHW f32 data.
///
/// This is a pure layout transformation — no normalisation is applied
/// (the input is assumed to already be in `[0, 1]`).
#[allow(dead_code)]
fn hwc_to_nchw(hwc: &Tensor) -> Vec<f32> {
    let shape = hwc.shape();
    let h = shape[0];
    let w = shape[1];
    let data = hwc.data();
    let mut nchw = vec![0.0f32; 3 * h * w];
    for y in 0..h {
        for x in 0..w {
            let src = (y * w + x) * 3;
            for c in 0..3 {
                nchw[c * h * w + y * w + x] = data[src + c];
            }
        }
    }
    nchw
}

/// Run YOLOv8 inference using an ONNX model.
///
/// Takes an ONNX model, input image data (RGB, normalised to `[0,1]`) in
/// `[1, 3, H, W]` NCHW format, original image dimensions, and a
/// [`YoloConfig`].  Returns detected objects after NMS.
#[cfg(feature = "onnx")]
pub fn detect_yolov8_onnx(
    model: &yscv_onnx::OnnxModel,
    image_data: &[f32],
    img_height: usize,
    img_width: usize,
    config: &YoloConfig,
) -> Result<Vec<Detection>, crate::DetectError> {
    use std::collections::HashMap;

    let input_name = model
        .inputs
        .first()
        .cloned()
        .unwrap_or_else(|| "images".to_string());

    let tensor = Tensor::from_vec(
        vec![1, 3, config.input_size, config.input_size],
        image_data.to_vec(),
    )?;

    let mut inputs = HashMap::new();
    inputs.insert(input_name, tensor);

    let outputs = yscv_onnx::run_onnx_model(model, inputs)?;

    let output_name = model
        .outputs
        .first()
        .cloned()
        .unwrap_or_else(|| "output0".to_string());

    let output_tensor =
        outputs
            .get(&output_name)
            .ok_or_else(|| yscv_onnx::OnnxError::MissingInput {
                node: "model_output".to_string(),
                input: output_name,
            })?;

    Ok(decode_yolov8_output(
        output_tensor,
        config,
        img_width,
        img_height,
    ))
}

/// Run the full YOLOv8 detection pipeline on an HWC image.
///
/// Accepts raw `[H, W, 3]` RGB f32 pixel data (normalised to `[0,1]`),
/// applies letterbox preprocessing, runs ONNX inference, decodes the output,
/// and returns the final detections.
#[cfg(feature = "onnx")]
pub fn detect_yolov8_from_rgb(
    model: &yscv_onnx::OnnxModel,
    rgb_data: &[f32],
    height: usize,
    width: usize,
    config: &YoloConfig,
) -> Result<Vec<Detection>, crate::DetectError> {
    let image = Tensor::from_vec(vec![height, width, 3], rgb_data.to_vec())?;
    let (letterboxed, _scale, _pad_x, _pad_y) = letterbox_preprocess(&image, config.input_size);

    let nchw = hwc_to_nchw(&letterboxed);

    detect_yolov8_onnx(model, &nchw, height, width, config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coco_labels_count() {
        assert_eq!(coco_labels().len(), 80);
    }

    #[test]
    fn test_yolov8_coco_config_defaults() {
        let cfg = yolov8_coco_config();
        assert_eq!(cfg.input_size, 640);
        assert_eq!(cfg.num_classes, 80);
        assert!((cfg.conf_threshold - 0.25).abs() < 1e-6);
        assert!((cfg.iou_threshold - 0.45).abs() < 1e-6);
        assert_eq!(cfg.class_labels.len(), 80);
    }

    /// Build a synthetic [1, 84, 8400] tensor with exactly one strong
    /// prediction at index 0, class 5 with score 0.9.
    fn make_one_detection_tensor() -> Tensor {
        let num_classes = 80;
        let rows = 4 + num_classes;
        let num_preds = 8400;
        let mut data = vec![0.0f32; rows * num_preds];

        // Prediction at index 0: centre (320, 320), size 100x100 in 640x640.
        data[0] = 320.0; // cx
        data[num_preds] = 320.0; // cy
        data[2 * num_preds] = 100.0; // w
        data[3 * num_preds] = 100.0; // h

        // Class 5 has score 0.9; others stay at 0.
        data[(4 + 5) * num_preds] = 0.9;

        Tensor::from_vec(vec![1, rows, num_preds], data).unwrap()
    }

    #[test]
    fn test_decode_yolov8_output_basic() {
        let tensor = make_one_detection_tensor();
        let config = YoloConfig {
            input_size: 640,
            num_classes: 80,
            conf_threshold: 0.25,
            iou_threshold: 0.45,
            class_labels: coco_labels(),
        };

        // Original image is also 640x640 so no rescaling.
        let dets = decode_yolov8_output(&tensor, &config, 640, 640);
        assert_eq!(dets.len(), 1);
        assert_eq!(dets[0].class_id, 5);
        assert!((dets[0].score - 0.9).abs() < 1e-6);

        // Box should be (270, 270, 370, 370) in original coords.
        let b = &dets[0].bbox;
        assert!((b.x1 - 270.0).abs() < 1.0);
        assert!((b.y1 - 270.0).abs() < 1.0);
        assert!((b.x2 - 370.0).abs() < 1.0);
        assert!((b.y2 - 370.0).abs() < 1.0);
    }

    #[test]
    fn test_decode_yolov8_output_confidence_filter() {
        let tensor = make_one_detection_tensor();
        let config = YoloConfig {
            input_size: 640,
            num_classes: 80,
            conf_threshold: 0.95, // higher than our 0.9 score
            iou_threshold: 0.45,
            class_labels: coco_labels(),
        };
        let dets = decode_yolov8_output(&tensor, &config, 640, 640);
        assert!(dets.is_empty());
    }

    #[test]
    fn test_decode_yolov8_output_nms() {
        let num_classes = 80;
        let rows = 4 + num_classes;
        let num_preds = 8400;
        let mut data = vec![0.0f32; rows * num_preds];

        // Two highly overlapping boxes, same class (class 0).
        // Box 0: centre (320, 320), 100x100, score 0.9
        data[0] = 320.0;
        data[num_preds] = 320.0;
        data[2 * num_preds] = 100.0;
        data[3 * num_preds] = 100.0;
        data[4 * num_preds] = 0.9;

        // Box 1: centre (325, 325), 100x100, score 0.8 (heavily overlapping)
        data[1] = 325.0;
        data[num_preds + 1] = 325.0;
        data[2 * num_preds + 1] = 100.0;
        data[3 * num_preds + 1] = 100.0;
        data[4 * num_preds + 1] = 0.8;

        let tensor = Tensor::from_vec(vec![1, rows, num_preds], data).unwrap();
        let config = YoloConfig {
            input_size: 640,
            num_classes: 80,
            conf_threshold: 0.25,
            iou_threshold: 0.45,
            class_labels: coco_labels(),
        };

        let dets = decode_yolov8_output(&tensor, &config, 640, 640);
        // NMS should suppress the lower-scoring duplicate.
        assert_eq!(dets.len(), 1);
        assert!((dets[0].score - 0.9).abs() < 1e-6);
    }

    #[test]
    fn test_letterbox_preprocess_square() {
        // 100x100 image → 640x640 should have no padding.
        let img = Tensor::from_vec(vec![100, 100, 3], vec![0.5; 100 * 100 * 3]).unwrap();
        let (out, scale, pad_x, pad_y) = letterbox_preprocess(&img, 640);
        assert_eq!(out.shape(), &[640, 640, 3]);
        assert!((scale - 6.4).abs() < 0.01);
        assert!(pad_x.abs() < 1.0);
        assert!(pad_y.abs() < 1.0);
    }

    #[test]
    fn test_hwc_to_nchw_basic() {
        // 2x2 RGB image
        let data = vec![
            0.1, 0.2, 0.3, // (0,0) R G B
            0.4, 0.5, 0.6, // (0,1)
            0.7, 0.8, 0.9, // (1,0)
            1.0, 0.0, 0.5, // (1,1)
        ];
        let img = Tensor::from_vec(vec![2, 2, 3], data).unwrap();
        let nchw = hwc_to_nchw(&img);
        // Expected layout: [R-plane, G-plane, B-plane], each 2x2
        assert_eq!(nchw.len(), 12);
        // R plane
        assert!((nchw[0] - 0.1).abs() < 1e-6); // (0,0)
        assert!((nchw[1] - 0.4).abs() < 1e-6); // (0,1)
        assert!((nchw[2] - 0.7).abs() < 1e-6); // (1,0)
        assert!((nchw[3] - 1.0).abs() < 1e-6); // (1,1)
        // G plane
        assert!((nchw[4] - 0.2).abs() < 1e-6);
        assert!((nchw[5] - 0.5).abs() < 1e-6);
        assert!((nchw[6] - 0.8).abs() < 1e-6);
        assert!((nchw[7] - 0.0).abs() < 1e-6);
        // B plane
        assert!((nchw[8] - 0.3).abs() < 1e-6);
        assert!((nchw[9] - 0.6).abs() < 1e-6);
        assert!((nchw[10] - 0.9).abs() < 1e-6);
        assert!((nchw[11] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_letterbox_then_nchw_pipeline() {
        // Rectangular 100x200 image through full preprocess pipeline.
        let img = Tensor::from_vec(vec![100, 200, 3], vec![0.4; 100 * 200 * 3]).unwrap();
        let (letterboxed, _scale, _pad_x, _pad_y) = letterbox_preprocess(&img, 640);
        assert_eq!(letterboxed.shape(), &[640, 640, 3]);
        let nchw = hwc_to_nchw(&letterboxed);
        assert_eq!(nchw.len(), 3 * 640 * 640);
    }

    #[test]
    fn test_letterbox_preprocess_landscape() {
        // 200x100 image → scale limited by width: 640/200 = 3.2
        // new_w = 640, new_h = 320 → pad_y = (640-320)/2 = 160
        let img = Tensor::from_vec(vec![100, 200, 3], vec![0.4; 100 * 200 * 3]).unwrap();
        let (out, scale, pad_x, pad_y) = letterbox_preprocess(&img, 640);
        assert_eq!(out.shape(), &[640, 640, 3]);
        assert!((scale - 3.2).abs() < 0.01);
        assert!(pad_x.abs() < 1.0);
        assert!((pad_y - 160.0).abs() < 1.0);

        // Check that the padded (grey 0.5) region exists at top.
        let top_pixel = &out.data()[0..3];
        for &v in top_pixel {
            assert!((v - 0.5).abs() < 1e-6, "top padding should be 0.5 grey");
        }
    }
}
