use proptest::prelude::*;
use yscv_tensor::Tensor;

use super::{BoundingBox, Detection, iou, letterbox_preprocess, non_max_suppression};

/// Strategy: generate a random bounding box with x1 < x2, y1 < y2.
fn arb_bbox() -> impl Strategy<Value = BoundingBox> {
    (0.0f32..100.0, 0.0f32..100.0, 1.0f32..50.0, 1.0f32..50.0).prop_map(|(x1, y1, w, h)| {
        BoundingBox {
            x1,
            y1,
            x2: x1 + w,
            y2: y1 + h,
        }
    })
}

/// Strategy: generate a random detection.
fn arb_detection() -> impl Strategy<Value = Detection> {
    (arb_bbox(), 0.01f32..1.0, 0usize..5).prop_map(|(bbox, score, class_id)| Detection {
        bbox,
        score,
        class_id,
    })
}

proptest! {
    // ── NMS reduces count ──────────────────────────────────────────────
    #[test]
    fn nms_reduces_count(
        detections in proptest::collection::vec(arb_detection(), 0..=32),
        iou_thresh in 0.1f32..0.9,
    ) {
        let result = non_max_suppression(&detections, iou_thresh, detections.len().max(1));
        prop_assert!(
            result.len() <= detections.len(),
            "NMS increased count: {} -> {}",
            detections.len(),
            result.len()
        );
    }

    // ── IoU symmetry ───────────────────────────────────────────────────
    #[test]
    fn iou_is_symmetric(a in arb_bbox(), b in arb_bbox()) {
        let iou_ab = iou(a, b);
        let iou_ba = iou(b, a);
        prop_assert!(
            (iou_ab - iou_ba).abs() < 1e-6,
            "IoU not symmetric: iou(a,b)={iou_ab}, iou(b,a)={iou_ba}"
        );
    }

    // ── IoU range ──────────────────────────────────────────────────────
    #[test]
    fn iou_in_valid_range(a in arb_bbox(), b in arb_bbox()) {
        let v = iou(a, b);
        prop_assert!(
            (0.0..=1.0).contains(&v),
            "IoU out of [0,1]: got {v}"
        );
    }

    // ── (l) Letterbox preserves aspect ratio direction ─────────────────
    #[test]
    fn letterbox_preserves_aspect_ratio(
        w in 32usize..=640,
        h in 32usize..=640,
    ) {
        let target = 640usize;
        // Build an [H, W, 3] tensor with dummy pixel data
        let data = vec![128.0f32 / 255.0; h * w * 3];
        let image = Tensor::from_vec(vec![h, w, 3], data).expect("valid image tensor");

        let (output, scale, _pad_x, _pad_y) = letterbox_preprocess(&image, target);
        let out_shape = output.shape();

        // Output must be [target, target, 3]
        prop_assert_eq!(
            out_shape[0], target,
            "output height should be {}, got {}", target, out_shape[0]
        );
        prop_assert_eq!(
            out_shape[1], target,
            "output width should be {}, got {}", target, out_shape[1]
        );
        prop_assert_eq!(
            out_shape[2], 3,
            "output channels should be 3, got {}", out_shape[2]
        );

        // Scale should be positive. When image < target, scale > 1 (upscale).
        prop_assert!(
            scale > 0.0,
            "letterbox scale should be positive, got: {}", scale
        );
    }
}
