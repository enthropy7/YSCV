#![doc = include_str!("../README.md")]
#![forbid(unsafe_code)]

mod anchors;
mod error;
mod frame;
mod heatmap;
mod model_detector;
mod nms;
mod roi;
#[cfg(test)]
mod tests;

#[cfg(test)]
mod proptest_tests;
mod types;
mod yolo;

pub use anchors::generate_anchors;
pub use error::DetectError;
pub use frame::{
    FrameFaceDetectScratch, FramePeopleDetectScratch, Rgb8FaceDetectScratch,
    Rgb8PeopleDetectScratch, detect_faces_from_frame, detect_faces_from_frame_with_scratch,
    detect_faces_from_rgb8, detect_faces_from_rgb8_with_scratch, detect_people_from_frame,
    detect_people_from_frame_with_scratch, detect_people_from_rgb8,
    detect_people_from_rgb8_with_scratch,
};
pub use heatmap::{HeatmapDetectScratch, detect_from_heatmap, detect_from_heatmap_with_scratch};
pub use model_detector::{
    ModelDetector, ModelDetectorConfig, postprocess_detections, preprocess_rgb8_for_model,
};
pub use nms::{batched_nms, iou, non_max_suppression, soft_nms};
pub use roi::{roi_align, roi_pool};
pub use types::{BoundingBox, CLASS_ID_FACE, CLASS_ID_PERSON, CRATE_ID, Detection};
pub use yolo::{
    YoloConfig, coco_labels, decode_yolov8_output, decode_yolov11_output, letterbox_preprocess,
    yolov8_coco_config, yolov11_coco_config,
};
#[cfg(feature = "onnx")]
pub use yolo::{detect_yolov8_from_rgb, detect_yolov8_onnx};
