#![doc = include_str!("../README.md")]
#![forbid(unsafe_code)]

mod camera_diagnostics;
mod classification;
mod counting;
mod dataset;
mod detection;
mod error;
mod metrics;
mod pipeline;
mod regression;
#[cfg(test)]
mod tests;
mod timing;
mod tracking;
mod util;

pub const CRATE_ID: &str = "yscv-eval";

pub use camera_diagnostics::{
    CameraDiagnosticsCapture, CameraDiagnosticsDevice, CameraDiagnosticsFirstFrame,
    CameraDiagnosticsReport, CameraDiagnosticsRequested, CameraDiagnosticsThresholds,
    CameraDiagnosticsTiming, CameraDiagnosticsViolation, load_camera_diagnostics_report_json_file,
    parse_camera_diagnostics_report_json, validate_camera_diagnostics_report,
};
pub use classification::{
    F1Average, accuracy, average_precision, classification_report, cohens_kappa, confusion_matrix,
    f1_score, per_class_precision_recall, precision_recall_curve,
};
pub use counting::{CountingMetrics, evaluate_counts};
pub use dataset::{
    load_detection_dataset_coco_files, load_detection_dataset_jsonl_file,
    load_detection_dataset_kitti_label_dirs, load_detection_dataset_openimages_csv_files,
    load_detection_dataset_voc_xml_dirs, load_detection_dataset_widerface_files,
    load_detection_dataset_yolo_label_dirs, load_tracking_dataset_jsonl_file,
    load_tracking_dataset_mot_txt_files, parse_detection_dataset_coco,
    parse_detection_dataset_jsonl, parse_detection_dataset_openimages_csv,
    parse_detection_dataset_widerface, parse_tracking_dataset_jsonl, parse_tracking_dataset_mot,
};
pub use detection::{
    CocoMetrics, DetectionDatasetFrame, DetectionEvalConfig, DetectionFrame, DetectionMetrics,
    LabeledBox, detection_frames_as_view, evaluate_detections, evaluate_detections_coco,
    evaluate_detections_from_dataset,
};
pub use error::EvalError;
pub use metrics::{
    auc, dice_score, mean_iou, per_class_iou, psnr, roc_curve, ssim, top_k_accuracy,
};
pub use pipeline::{
    BenchmarkViolation, PipelineBenchmarkReport, PipelineBenchmarkThresholds, PipelineDurations,
    StageThresholds, parse_pipeline_benchmark_thresholds, summarize_pipeline_durations,
    validate_pipeline_benchmark_thresholds,
};
pub use regression::{mae, mape, r2_score, rmse};
pub use timing::{TimingStats, summarize_durations};
pub use tracking::{
    GroundTruthTrack, TrackingDatasetFrame, TrackingEvalConfig, TrackingFrame, TrackingMetrics,
    evaluate_tracking, evaluate_tracking_from_dataset, hota, idf1, tracking_frames_as_view,
};
