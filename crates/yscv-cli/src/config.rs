use std::env;
use std::path::PathBuf;

use thiserror::Error;
use yscv_detect::{CLASS_ID_FACE, CLASS_ID_PERSON};

use crate::util::face_min_area;

#[derive(Debug, Clone, PartialEq)]
pub struct CliConfig {
    pub list_cameras: bool,
    pub diagnose_camera: bool,
    pub diagnose_frames: usize,
    pub diagnose_report_path: Option<PathBuf>,
    pub camera: bool,
    pub benchmark: bool,
    pub detect_target: DetectTarget,
    pub detect_score_threshold: Option<f32>,
    pub detect_min_area: Option<usize>,
    pub detect_iou_threshold: Option<f32>,
    pub detect_max_detections: Option<usize>,
    pub device_index: u32,
    pub device_name_query: Option<String>,
    pub width: u32,
    pub height: u32,
    pub fps: u32,
    pub track_iou_threshold: f32,
    pub track_max_missed_frames: u32,
    pub track_max_tracks: usize,
    pub recognition_threshold: f32,
    pub max_frames: Option<usize>,
    pub identities_path: Option<PathBuf>,
    pub eval_detection_dataset_path: Option<PathBuf>,
    pub eval_detection_coco_gt_path: Option<PathBuf>,
    pub eval_detection_coco_pred_path: Option<PathBuf>,
    pub eval_detection_openimages_gt_path: Option<PathBuf>,
    pub eval_detection_openimages_pred_path: Option<PathBuf>,
    pub eval_detection_yolo_manifest_path: Option<PathBuf>,
    pub eval_detection_yolo_gt_dir_path: Option<PathBuf>,
    pub eval_detection_yolo_pred_dir_path: Option<PathBuf>,
    pub eval_detection_voc_manifest_path: Option<PathBuf>,
    pub eval_detection_voc_gt_dir_path: Option<PathBuf>,
    pub eval_detection_voc_pred_dir_path: Option<PathBuf>,
    pub eval_detection_kitti_manifest_path: Option<PathBuf>,
    pub eval_detection_kitti_gt_dir_path: Option<PathBuf>,
    pub eval_detection_kitti_pred_dir_path: Option<PathBuf>,
    pub eval_detection_widerface_gt_path: Option<PathBuf>,
    pub eval_detection_widerface_pred_path: Option<PathBuf>,
    pub eval_tracking_dataset_path: Option<PathBuf>,
    pub eval_tracking_mot_gt_path: Option<PathBuf>,
    pub eval_tracking_mot_pred_path: Option<PathBuf>,
    pub eval_iou_threshold: f32,
    pub eval_score_threshold: f32,
    pub validate_diagnostics_report_path: Option<PathBuf>,
    pub validate_diagnostics_min_frames: usize,
    pub validate_diagnostics_max_drift_pct: f64,
    pub validate_diagnostics_max_dropped_frames: u64,
    pub benchmark_report_path: Option<PathBuf>,
    pub benchmark_baseline_path: Option<PathBuf>,
    pub event_log_path: Option<PathBuf>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DetectTarget {
    People,
    Faces,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RuntimeDetectConfig {
    pub score_threshold: f32,
    pub min_area: usize,
    pub iou_threshold: f32,
    pub max_detections: usize,
}

impl DetectTarget {
    fn parse(raw: &str) -> Result<Self, CliError> {
        match raw {
            "people" => Ok(Self::People),
            "face" | "faces" => Ok(Self::Faces),
            _ => Err(CliError::Message(format!(
                "invalid --detect-target `{raw}`; expected one of: people, face"
            ))),
        }
    }

    pub const fn class_id(self) -> usize {
        match self {
            Self::People => CLASS_ID_PERSON,
            Self::Faces => CLASS_ID_FACE,
        }
    }

    pub const fn count_label(self) -> &'static str {
        match self {
            Self::People => "people",
            Self::Faces => "faces",
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::People => "people",
            Self::Faces => "face",
        }
    }

    fn default_config(self, frame_width: usize, frame_height: usize) -> RuntimeDetectConfig {
        match self {
            Self::People => RuntimeDetectConfig {
                score_threshold: 0.5,
                min_area: 2,
                iou_threshold: 0.4,
                max_detections: 16,
            },
            Self::Faces => RuntimeDetectConfig {
                score_threshold: 0.35,
                min_area: face_min_area(frame_width, frame_height),
                iou_threshold: 0.4,
                max_detections: 16,
            },
        }
    }
}

pub fn resolve_detect_config(
    cli: &CliConfig,
    frame_width: usize,
    frame_height: usize,
) -> RuntimeDetectConfig {
    let defaults = cli.detect_target.default_config(frame_width, frame_height);
    RuntimeDetectConfig {
        score_threshold: cli
            .detect_score_threshold
            .unwrap_or(defaults.score_threshold),
        min_area: cli.detect_min_area.unwrap_or(defaults.min_area),
        iou_threshold: cli.detect_iou_threshold.unwrap_or(defaults.iou_threshold),
        max_detections: cli.detect_max_detections.unwrap_or(defaults.max_detections),
    }
}

impl Default for CliConfig {
    fn default() -> Self {
        Self {
            list_cameras: false,
            diagnose_camera: false,
            diagnose_frames: 30,
            diagnose_report_path: None,
            camera: false,
            benchmark: false,
            detect_target: DetectTarget::People,
            detect_score_threshold: None,
            detect_min_area: None,
            detect_iou_threshold: None,
            detect_max_detections: None,
            device_index: 0,
            device_name_query: None,
            width: 640,
            height: 480,
            fps: 30,
            track_iou_threshold: 0.2,
            track_max_missed_frames: 2,
            track_max_tracks: 64,
            recognition_threshold: 0.92,
            max_frames: None,
            identities_path: None,
            eval_detection_dataset_path: None,
            eval_detection_coco_gt_path: None,
            eval_detection_coco_pred_path: None,
            eval_detection_openimages_gt_path: None,
            eval_detection_openimages_pred_path: None,
            eval_detection_yolo_manifest_path: None,
            eval_detection_yolo_gt_dir_path: None,
            eval_detection_yolo_pred_dir_path: None,
            eval_detection_voc_manifest_path: None,
            eval_detection_voc_gt_dir_path: None,
            eval_detection_voc_pred_dir_path: None,
            eval_detection_kitti_manifest_path: None,
            eval_detection_kitti_gt_dir_path: None,
            eval_detection_kitti_pred_dir_path: None,
            eval_detection_widerface_gt_path: None,
            eval_detection_widerface_pred_path: None,
            eval_tracking_dataset_path: None,
            eval_tracking_mot_gt_path: None,
            eval_tracking_mot_pred_path: None,
            eval_iou_threshold: 0.5,
            eval_score_threshold: 0.0,
            validate_diagnostics_report_path: None,
            validate_diagnostics_min_frames: 2,
            validate_diagnostics_max_drift_pct: 25.0,
            validate_diagnostics_max_dropped_frames: 0,
            benchmark_report_path: None,
            benchmark_baseline_path: None,
            event_log_path: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum CliError {
    #[error("help requested")]
    HelpRequested,
    #[error("{0}")]
    Message(String),
}

impl CliConfig {
    pub fn from_env() -> Result<Self, CliError> {
        Self::parse_from(env::args().skip(1))
    }

    pub fn parse_from<I>(args: I) -> Result<Self, CliError>
    where
        I: IntoIterator<Item = String>,
    {
        let args = args.into_iter().collect::<Vec<_>>();
        let mut config = Self::default();
        let mut index = 0usize;
        let mut has_device_index = false;
        let mut has_device_name = false;

        while index < args.len() {
            match args[index].as_str() {
                "--diagnose-camera" => {
                    config.diagnose_camera = true;
                }
                "--diagnose-frames" => {
                    let raw = next_value(&args, &mut index, "--diagnose-frames")?;
                    config.diagnose_frames = parse_usize("--diagnose-frames", raw)?;
                }
                "--diagnose-report" => {
                    let raw = next_value(&args, &mut index, "--diagnose-report")?;
                    config.diagnose_report_path = Some(parse_path(raw));
                }
                "--camera" => {
                    config.camera = true;
                }
                "--list-cameras" => {
                    config.list_cameras = true;
                }
                "--benchmark" => {
                    config.benchmark = true;
                }
                "--detect-target" => {
                    let raw = next_value(&args, &mut index, "--detect-target")?;
                    config.detect_target = DetectTarget::parse(&raw)?;
                }
                "--detect-score" => {
                    let raw = next_value(&args, &mut index, "--detect-score")?;
                    config.detect_score_threshold = Some(parse_f32("--detect-score", raw)?);
                }
                "--detect-min-area" => {
                    let raw = next_value(&args, &mut index, "--detect-min-area")?;
                    config.detect_min_area = Some(parse_usize("--detect-min-area", raw)?);
                }
                "--detect-iou" => {
                    let raw = next_value(&args, &mut index, "--detect-iou")?;
                    config.detect_iou_threshold = Some(parse_f32("--detect-iou", raw)?);
                }
                "--detect-max" => {
                    let raw = next_value(&args, &mut index, "--detect-max")?;
                    config.detect_max_detections = Some(parse_usize("--detect-max", raw)?);
                }
                "--device" => {
                    if has_device_name {
                        return Err(CliError::Message(
                            "`--device` cannot be used together with `--device-name`".to_string(),
                        ));
                    }
                    let raw = next_value(&args, &mut index, "--device")?;
                    config.device_index = parse_u32("--device", raw)?;
                    has_device_index = true;
                }
                "--device-name" => {
                    if has_device_index {
                        return Err(CliError::Message(
                            "`--device-name` cannot be used together with `--device`".to_string(),
                        ));
                    }
                    let raw = next_value(&args, &mut index, "--device-name")?;
                    config.device_name_query = Some(parse_non_empty("--device-name", raw)?);
                    has_device_name = true;
                }
                "--width" => {
                    let raw = next_value(&args, &mut index, "--width")?;
                    config.width = parse_u32("--width", raw)?;
                }
                "--height" => {
                    let raw = next_value(&args, &mut index, "--height")?;
                    config.height = parse_u32("--height", raw)?;
                }
                "--fps" => {
                    let raw = next_value(&args, &mut index, "--fps")?;
                    config.fps = parse_u32("--fps", raw)?;
                }
                "--track-iou" => {
                    let raw = next_value(&args, &mut index, "--track-iou")?;
                    config.track_iou_threshold = parse_f32("--track-iou", raw)?;
                }
                "--track-max-missed" => {
                    let raw = next_value(&args, &mut index, "--track-max-missed")?;
                    config.track_max_missed_frames = parse_u32("--track-max-missed", raw)?;
                }
                "--track-max" => {
                    let raw = next_value(&args, &mut index, "--track-max")?;
                    config.track_max_tracks = parse_usize("--track-max", raw)?;
                }
                "--recognition-threshold" => {
                    let raw = next_value(&args, &mut index, "--recognition-threshold")?;
                    config.recognition_threshold = parse_f32("--recognition-threshold", raw)?;
                }
                "--max-frames" => {
                    let raw = next_value(&args, &mut index, "--max-frames")?;
                    config.max_frames = Some(parse_usize("--max-frames", raw)?);
                }
                "--identities" => {
                    let raw = next_value(&args, &mut index, "--identities")?;
                    config.identities_path = Some(parse_path(raw));
                }
                "--eval-detection-jsonl" => {
                    let raw = next_value(&args, &mut index, "--eval-detection-jsonl")?;
                    config.eval_detection_dataset_path = Some(parse_path(raw));
                }
                "--eval-detection-coco-gt" => {
                    let raw = next_value(&args, &mut index, "--eval-detection-coco-gt")?;
                    config.eval_detection_coco_gt_path = Some(parse_path(raw));
                }
                "--eval-detection-coco-pred" => {
                    let raw = next_value(&args, &mut index, "--eval-detection-coco-pred")?;
                    config.eval_detection_coco_pred_path = Some(parse_path(raw));
                }
                "--eval-detection-openimages-gt" => {
                    let raw = next_value(&args, &mut index, "--eval-detection-openimages-gt")?;
                    config.eval_detection_openimages_gt_path = Some(parse_path(raw));
                }
                "--eval-detection-openimages-pred" => {
                    let raw = next_value(&args, &mut index, "--eval-detection-openimages-pred")?;
                    config.eval_detection_openimages_pred_path = Some(parse_path(raw));
                }
                "--eval-detection-yolo-manifest" => {
                    let raw = next_value(&args, &mut index, "--eval-detection-yolo-manifest")?;
                    config.eval_detection_yolo_manifest_path = Some(parse_path(raw));
                }
                "--eval-detection-yolo-gt-dir" => {
                    let raw = next_value(&args, &mut index, "--eval-detection-yolo-gt-dir")?;
                    config.eval_detection_yolo_gt_dir_path = Some(parse_path(raw));
                }
                "--eval-detection-yolo-pred-dir" => {
                    let raw = next_value(&args, &mut index, "--eval-detection-yolo-pred-dir")?;
                    config.eval_detection_yolo_pred_dir_path = Some(parse_path(raw));
                }
                "--eval-detection-voc-manifest" => {
                    let raw = next_value(&args, &mut index, "--eval-detection-voc-manifest")?;
                    config.eval_detection_voc_manifest_path = Some(parse_path(raw));
                }
                "--eval-detection-voc-gt-dir" => {
                    let raw = next_value(&args, &mut index, "--eval-detection-voc-gt-dir")?;
                    config.eval_detection_voc_gt_dir_path = Some(parse_path(raw));
                }
                "--eval-detection-voc-pred-dir" => {
                    let raw = next_value(&args, &mut index, "--eval-detection-voc-pred-dir")?;
                    config.eval_detection_voc_pred_dir_path = Some(parse_path(raw));
                }
                "--eval-detection-kitti-manifest" => {
                    let raw = next_value(&args, &mut index, "--eval-detection-kitti-manifest")?;
                    config.eval_detection_kitti_manifest_path = Some(parse_path(raw));
                }
                "--eval-detection-kitti-gt-dir" => {
                    let raw = next_value(&args, &mut index, "--eval-detection-kitti-gt-dir")?;
                    config.eval_detection_kitti_gt_dir_path = Some(parse_path(raw));
                }
                "--eval-detection-kitti-pred-dir" => {
                    let raw = next_value(&args, &mut index, "--eval-detection-kitti-pred-dir")?;
                    config.eval_detection_kitti_pred_dir_path = Some(parse_path(raw));
                }
                "--eval-detection-widerface-gt" => {
                    let raw = next_value(&args, &mut index, "--eval-detection-widerface-gt")?;
                    config.eval_detection_widerface_gt_path = Some(parse_path(raw));
                }
                "--eval-detection-widerface-pred" => {
                    let raw = next_value(&args, &mut index, "--eval-detection-widerface-pred")?;
                    config.eval_detection_widerface_pred_path = Some(parse_path(raw));
                }
                "--eval-tracking-jsonl" => {
                    let raw = next_value(&args, &mut index, "--eval-tracking-jsonl")?;
                    config.eval_tracking_dataset_path = Some(parse_path(raw));
                }
                "--eval-tracking-mot-gt" => {
                    let raw = next_value(&args, &mut index, "--eval-tracking-mot-gt")?;
                    config.eval_tracking_mot_gt_path = Some(parse_path(raw));
                }
                "--eval-tracking-mot-pred" => {
                    let raw = next_value(&args, &mut index, "--eval-tracking-mot-pred")?;
                    config.eval_tracking_mot_pred_path = Some(parse_path(raw));
                }
                "--eval-iou" => {
                    let raw = next_value(&args, &mut index, "--eval-iou")?;
                    config.eval_iou_threshold = parse_f32("--eval-iou", raw)?;
                }
                "--eval-score" => {
                    let raw = next_value(&args, &mut index, "--eval-score")?;
                    config.eval_score_threshold = parse_f32("--eval-score", raw)?;
                }
                "--validate-diagnostics-report" => {
                    let raw = next_value(&args, &mut index, "--validate-diagnostics-report")?;
                    config.validate_diagnostics_report_path = Some(parse_path(raw));
                }
                "--validate-diagnostics-min-frames" => {
                    let raw = next_value(&args, &mut index, "--validate-diagnostics-min-frames")?;
                    config.validate_diagnostics_min_frames =
                        parse_usize("--validate-diagnostics-min-frames", raw)?;
                }
                "--validate-diagnostics-max-drift-pct" => {
                    let raw =
                        next_value(&args, &mut index, "--validate-diagnostics-max-drift-pct")?;
                    config.validate_diagnostics_max_drift_pct =
                        parse_f64("--validate-diagnostics-max-drift-pct", raw)?;
                }
                "--validate-diagnostics-max-dropped" => {
                    let raw = next_value(&args, &mut index, "--validate-diagnostics-max-dropped")?;
                    config.validate_diagnostics_max_dropped_frames =
                        parse_u64("--validate-diagnostics-max-dropped", raw)?;
                }
                "--benchmark-report" => {
                    let raw = next_value(&args, &mut index, "--benchmark-report")?;
                    config.benchmark_report_path = Some(parse_path(raw));
                }
                "--benchmark-baseline" => {
                    let raw = next_value(&args, &mut index, "--benchmark-baseline")?;
                    config.benchmark_baseline_path = Some(parse_path(raw));
                }
                "--event-log" => {
                    let raw = next_value(&args, &mut index, "--event-log")?;
                    config.event_log_path = Some(parse_path(raw));
                }
                "--help" | "-h" => return Err(CliError::HelpRequested),
                unknown => {
                    return Err(CliError::Message(format!(
                        "unknown argument: {unknown}; run with --help for usage"
                    )));
                }
            }
            index += 1;
        }

        validate_positive("--width", config.width)?;
        validate_positive("--height", config.height)?;
        validate_positive("--fps", config.fps)?;
        validate_positive_usize("--diagnose-frames", config.diagnose_frames)?;
        if let Some(score_threshold) = config.detect_score_threshold {
            validate_in_unit_interval("--detect-score", score_threshold)?;
        }
        if let Some(min_area) = config.detect_min_area {
            validate_positive_usize("--detect-min-area", min_area)?;
        }
        if let Some(iou_threshold) = config.detect_iou_threshold {
            validate_in_unit_interval("--detect-iou", iou_threshold)?;
        }
        if let Some(max_detections) = config.detect_max_detections {
            validate_positive_usize("--detect-max", max_detections)?;
        }
        validate_in_unit_interval("--track-iou", config.track_iou_threshold)?;
        validate_positive_usize("--track-max", config.track_max_tracks)?;
        validate_in_closed_range(
            "--recognition-threshold",
            config.recognition_threshold,
            -1.0,
            1.0,
        )?;
        if let Some(max_frames) = config.max_frames
            && max_frames == 0
        {
            return Err(CliError::Message(
                "`--max-frames` must be greater than 0".to_string(),
            ));
        }
        if config.benchmark_baseline_path.is_some() {
            config.benchmark = true;
        }
        if config.device_name_query.is_some() && !config.camera && !config.list_cameras {
            return Err(CliError::Message(
                "`--device-name` requires `--camera` or `--list-cameras`".to_string(),
            ));
        }
        validate_in_unit_interval("--eval-iou", config.eval_iou_threshold)?;
        validate_in_unit_interval("--eval-score", config.eval_score_threshold)?;
        if config.eval_detection_coco_gt_path.is_some()
            != config.eval_detection_coco_pred_path.is_some()
        {
            return Err(CliError::Message(
                "`--eval-detection-coco-gt` and `--eval-detection-coco-pred` must be provided together".to_string(),
            ));
        }
        if config.eval_detection_openimages_gt_path.is_some()
            != config.eval_detection_openimages_pred_path.is_some()
        {
            return Err(CliError::Message(
                "`--eval-detection-openimages-gt` and `--eval-detection-openimages-pred` must be provided together".to_string(),
            ));
        }
        let yolo_eval_flags = [
            config.eval_detection_yolo_manifest_path.is_some(),
            config.eval_detection_yolo_gt_dir_path.is_some(),
            config.eval_detection_yolo_pred_dir_path.is_some(),
        ];
        if yolo_eval_flags.iter().any(|present| *present)
            && !yolo_eval_flags.iter().all(|present| *present)
        {
            return Err(CliError::Message(
                "`--eval-detection-yolo-manifest`, `--eval-detection-yolo-gt-dir`, and `--eval-detection-yolo-pred-dir` must be provided together".to_string(),
            ));
        }
        let voc_eval_flags = [
            config.eval_detection_voc_manifest_path.is_some(),
            config.eval_detection_voc_gt_dir_path.is_some(),
            config.eval_detection_voc_pred_dir_path.is_some(),
        ];
        if voc_eval_flags.iter().any(|present| *present)
            && !voc_eval_flags.iter().all(|present| *present)
        {
            return Err(CliError::Message(
                "`--eval-detection-voc-manifest`, `--eval-detection-voc-gt-dir`, and `--eval-detection-voc-pred-dir` must be provided together".to_string(),
            ));
        }
        let kitti_eval_flags = [
            config.eval_detection_kitti_manifest_path.is_some(),
            config.eval_detection_kitti_gt_dir_path.is_some(),
            config.eval_detection_kitti_pred_dir_path.is_some(),
        ];
        if kitti_eval_flags.iter().any(|present| *present)
            && !kitti_eval_flags.iter().all(|present| *present)
        {
            return Err(CliError::Message(
                "`--eval-detection-kitti-manifest`, `--eval-detection-kitti-gt-dir`, and `--eval-detection-kitti-pred-dir` must be provided together".to_string(),
            ));
        }
        if config.eval_detection_widerface_gt_path.is_some()
            != config.eval_detection_widerface_pred_path.is_some()
        {
            return Err(CliError::Message(
                "`--eval-detection-widerface-gt` and `--eval-detection-widerface-pred` must be provided together".to_string(),
            ));
        }
        if config.eval_tracking_mot_gt_path.is_some()
            != config.eval_tracking_mot_pred_path.is_some()
        {
            return Err(CliError::Message(
                "`--eval-tracking-mot-gt` and `--eval-tracking-mot-pred` must be provided together"
                    .to_string(),
            ));
        }
        validate_positive_usize(
            "--validate-diagnostics-min-frames",
            config.validate_diagnostics_min_frames,
        )?;
        validate_non_negative_finite(
            "--validate-diagnostics-max-drift-pct",
            config.validate_diagnostics_max_drift_pct,
        )?;
        let eval_mode = config.eval_detection_dataset_path.is_some()
            || config.eval_detection_coco_gt_path.is_some()
            || config.eval_detection_openimages_gt_path.is_some()
            || config.eval_detection_yolo_manifest_path.is_some()
            || config.eval_detection_voc_manifest_path.is_some()
            || config.eval_detection_kitti_manifest_path.is_some()
            || config.eval_detection_widerface_gt_path.is_some()
            || config.eval_tracking_dataset_path.is_some()
            || config.eval_tracking_mot_gt_path.is_some();
        let diagnostics_validation_mode = config.validate_diagnostics_report_path.is_some();
        if eval_mode && (config.camera || config.list_cameras) {
            return Err(CliError::Message(
                "evaluation dataset mode cannot be combined with camera/listing mode".to_string(),
            ));
        }
        if eval_mode && config.diagnose_camera {
            return Err(CliError::Message(
                "evaluation dataset mode cannot be combined with `--diagnose-camera`".to_string(),
            ));
        }
        if eval_mode && config.benchmark {
            return Err(CliError::Message(
                "evaluation dataset mode cannot be combined with benchmark mode".to_string(),
            ));
        }
        if config.event_log_path.is_some() && config.list_cameras {
            return Err(CliError::Message(
                "`--event-log` cannot be used together with `--list-cameras`".to_string(),
            ));
        }
        if eval_mode && config.event_log_path.is_some() {
            return Err(CliError::Message(
                "evaluation dataset mode cannot be combined with `--event-log`".to_string(),
            ));
        }
        if config.diagnose_camera && config.list_cameras {
            return Err(CliError::Message(
                "`--diagnose-camera` cannot be used together with `--list-cameras`".to_string(),
            ));
        }
        if config.diagnose_camera && config.benchmark {
            return Err(CliError::Message(
                "`--diagnose-camera` cannot be used together with `--benchmark`".to_string(),
            ));
        }
        if config.diagnose_report_path.is_some() && !config.diagnose_camera {
            return Err(CliError::Message(
                "`--diagnose-report` requires `--diagnose-camera`".to_string(),
            ));
        }
        if diagnostics_validation_mode
            && (config.camera
                || config.list_cameras
                || config.diagnose_camera
                || eval_mode
                || config.benchmark
                || config.event_log_path.is_some())
        {
            return Err(CliError::Message(
                "diagnostics report validation mode cannot be combined with camera/diagnostics/eval/benchmark/event-log modes".to_string(),
            ));
        }
        if !diagnostics_validation_mode
            && (config.validate_diagnostics_min_frames
                != CliConfig::default().validate_diagnostics_min_frames
                || (config.validate_diagnostics_max_drift_pct
                    - CliConfig::default().validate_diagnostics_max_drift_pct)
                    .abs()
                    > f64::EPSILON
                || config.validate_diagnostics_max_dropped_frames
                    != CliConfig::default().validate_diagnostics_max_dropped_frames)
        {
            return Err(CliError::Message(
                "diagnostics validation thresholds require `--validate-diagnostics-report`"
                    .to_string(),
            ));
        }

        Ok(config)
    }
}

pub fn print_usage() {
    println!("yscv-cli usage:");
    println!("  cargo run -p yscv-cli --bin yscv-cli -- [options]");
    println!();
    println!("options:");
    println!("  --list-cameras           list available camera devices and exit");
    println!("  --diagnose-camera        run camera environment/capture diagnostics and exit");
    println!(
        "  --diagnose-frames <n>    number of frames to sample in diagnostics mode (default: 30)"
    );
    println!("  --diagnose-report <path> write diagnostics summary JSON report");
    println!(
        "  --camera                 use native camera source instead of deterministic demo frames"
    );
    println!("  --detect-target <mode>  detection mode: people (default) or face");
    println!("  --detect-score <value>  detection score threshold in [0, 1]");
    println!("  --detect-min-area <n>   minimum connected-component area in pixels");
    println!("  --detect-iou <value>    NMS IoU threshold in [0, 1]");
    println!("  --detect-max <n>        maximum detections per frame");
    println!("  --device <index>         camera device index (default: 0)");
    println!(
        "  --device-name <query>    camera device query by label substring (also filters --list-cameras)"
    );
    println!("  --width <pixels>         camera frame width (default: 640)");
    println!("  --height <pixels>        camera frame height (default: 480)");
    println!("  --fps <value>            camera target FPS (default: 30)");
    println!("  --track-iou <value>      tracker IoU match threshold in [0, 1] (default: 0.2)");
    println!("  --track-max-missed <n>   tracker missed-frame budget before expiry (default: 2)");
    println!("  --track-max <n>          tracker max simultaneous tracks (default: 64)");
    println!("  --recognition-threshold <value> recognizer threshold in [-1, 1] (default: 0.92)");
    println!("  --max-frames <count>     stop after N emitted frames");
    println!("  --identities <path>      load recognizer identities from JSON snapshot");
    println!("  --eval-detection-jsonl <path> evaluate detection metrics from JSONL dataset");
    println!("  --eval-detection-coco-gt <path> COCO detection ground-truth JSON");
    println!("  --eval-detection-coco-pred <path> COCO detection predictions JSON");
    println!("  --eval-detection-openimages-gt <path> OpenImages detection ground-truth CSV");
    println!("  --eval-detection-openimages-pred <path> OpenImages detection predictions CSV");
    println!(
        "  --eval-detection-yolo-manifest <path> YOLO detection manifest (`<image_id> <width> <height>` per line)"
    );
    println!("  --eval-detection-yolo-gt-dir <path> YOLO ground-truth label directory");
    println!("  --eval-detection-yolo-pred-dir <path> YOLO prediction label directory");
    println!(
        "  --eval-detection-voc-manifest <path> VOC detection image-id manifest (one `<image_id>` per line)"
    );
    println!("  --eval-detection-voc-gt-dir <path> VOC ground-truth XML directory");
    println!("  --eval-detection-voc-pred-dir <path> VOC prediction XML directory");
    println!(
        "  --eval-detection-kitti-manifest <path> KITTI detection image-id manifest (one `<image_id>` per line)"
    );
    println!("  --eval-detection-kitti-gt-dir <path> KITTI ground-truth label directory");
    println!("  --eval-detection-kitti-pred-dir <path> KITTI prediction label directory");
    println!("  --eval-detection-widerface-gt <path> WIDER FACE detection ground-truth TXT");
    println!("  --eval-detection-widerface-pred <path> WIDER FACE detection predictions TXT");
    println!("  --eval-tracking-jsonl <path> evaluate tracking metrics from JSONL dataset");
    println!("  --eval-tracking-mot-gt <path> MOTChallenge tracking ground-truth TXT");
    println!("  --eval-tracking-mot-pred <path> MOTChallenge tracking predictions TXT");
    println!("  --eval-iou <value>       evaluation IoU threshold in [0, 1] (default: 0.5)");
    println!("  --eval-score <value>     evaluation score threshold in [0, 1] (default: 0.0)");
    println!("  --validate-diagnostics-report <path> validate camera diagnostics JSON report");
    println!(
        "  --validate-diagnostics-min-frames <n> min collected frames for diagnostics report validation (default: 2)"
    );
    println!(
        "  --validate-diagnostics-max-drift-pct <value> max absolute wall/sensor fps drift percent (default: 25.0)"
    );
    println!(
        "  --validate-diagnostics-max-dropped <n> max dropped frames allowed in diagnostics report (default: 0)"
    );
    println!("  --event-log <path>       write per-frame JSONL events for downstream tooling");
    println!("  --benchmark              collect and print per-stage timing report");
    println!("  --benchmark-report <path> write benchmark report to a file");
    println!("  --benchmark-baseline <path> check benchmark report against thresholds");
    println!("  -h, --help               print this help");
    println!();
    println!("To enable camera capture, run with feature flag:");
    println!("  cargo run -p yscv-cli --bin yscv-cli --features native-camera -- --camera");
}

fn next_value(args: &[String], index: &mut usize, flag: &str) -> Result<String, CliError> {
    *index += 1;
    if let Some(value) = args.get(*index) {
        Ok(value.clone())
    } else {
        Err(CliError::Message(format!(
            "missing value for {flag}; run with --help for usage"
        )))
    }
}

fn parse_u32(flag: &str, raw: String) -> Result<u32, CliError> {
    raw.parse::<u32>().map_err(|_| {
        CliError::Message(format!(
            "failed to parse {flag} value `{raw}` as unsigned integer"
        ))
    })
}

fn parse_usize(flag: &str, raw: String) -> Result<usize, CliError> {
    raw.parse::<usize>().map_err(|_| {
        CliError::Message(format!(
            "failed to parse {flag} value `{raw}` as unsigned integer"
        ))
    })
}

fn parse_f32(flag: &str, raw: String) -> Result<f32, CliError> {
    raw.parse::<f32>()
        .map_err(|_| CliError::Message(format!("failed to parse {flag} value `{raw}` as number")))
}

fn parse_f64(flag: &str, raw: String) -> Result<f64, CliError> {
    raw.parse::<f64>()
        .map_err(|_| CliError::Message(format!("failed to parse {flag} value `{raw}` as number")))
}

fn parse_u64(flag: &str, raw: String) -> Result<u64, CliError> {
    raw.parse::<u64>().map_err(|_| {
        CliError::Message(format!(
            "failed to parse {flag} value `{raw}` as unsigned integer"
        ))
    })
}

fn parse_non_empty(flag: &str, raw: String) -> Result<String, CliError> {
    let normalized = raw.trim();
    if normalized.is_empty() {
        return Err(CliError::Message(format!("{flag} must not be empty")));
    }
    Ok(normalized.to_string())
}

fn parse_path(raw: String) -> PathBuf {
    PathBuf::from(raw)
}

fn validate_positive(flag: &str, value: u32) -> Result<(), CliError> {
    if value == 0 {
        return Err(CliError::Message(format!("{flag} must be greater than 0")));
    }
    Ok(())
}

fn validate_positive_usize(flag: &str, value: usize) -> Result<(), CliError> {
    if value == 0 {
        return Err(CliError::Message(format!("{flag} must be greater than 0")));
    }
    Ok(())
}

fn validate_in_unit_interval(flag: &str, value: f32) -> Result<(), CliError> {
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        return Err(CliError::Message(format!(
            "{flag} must be finite and in [0, 1]"
        )));
    }
    Ok(())
}

fn validate_in_closed_range(flag: &str, value: f32, min: f32, max: f32) -> Result<(), CliError> {
    if !value.is_finite() || !(min..=max).contains(&value) {
        return Err(CliError::Message(format!(
            "{flag} must be finite and in [{min}, {max}]"
        )));
    }
    Ok(())
}

fn validate_non_negative_finite(flag: &str, value: f64) -> Result<(), CliError> {
    if !value.is_finite() || value < 0.0 {
        return Err(CliError::Message(format!("{flag} must be finite and >= 0")));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{CliConfig, CliError, DetectTarget};

    #[test]
    fn parse_defaults_without_args() {
        let config = CliConfig::parse_from(Vec::<String>::new()).unwrap();
        assert_eq!(config, CliConfig::default());
    }

    #[test]
    fn parse_list_cameras_flag() {
        let config = CliConfig::parse_from(vec!["--list-cameras".to_string()]).unwrap();
        assert!(config.list_cameras);
        assert!(!config.camera);
    }

    #[test]
    fn parse_diagnose_camera_flag() {
        let config = CliConfig::parse_from(vec!["--diagnose-camera".to_string()]).unwrap();
        assert!(config.diagnose_camera);
        assert!(!config.camera);
    }

    #[test]
    fn parse_diagnose_frames_override() {
        let config =
            CliConfig::parse_from(vec!["--diagnose-frames".to_string(), "12".to_string()]).unwrap();
        assert_eq!(config.diagnose_frames, 12);
    }

    #[test]
    fn parse_diagnose_report_path() {
        let config = CliConfig::parse_from(vec![
            "--diagnose-camera".to_string(),
            "--diagnose-report".to_string(),
            "diag.json".to_string(),
        ])
        .unwrap();
        assert_eq!(
            config.diagnose_report_path,
            Some(std::path::PathBuf::from("diag.json"))
        );
    }

    #[test]
    fn parse_rejects_zero_diagnose_frames() {
        let err = CliConfig::parse_from(vec!["--diagnose-frames".to_string(), "0".to_string()])
            .unwrap_err();
        assert_eq!(
            err,
            CliError::Message("--diagnose-frames must be greater than 0".to_string())
        );
    }

    #[test]
    fn parse_device_name_query() {
        let config = CliConfig::parse_from(vec![
            "--camera".to_string(),
            "--device-name".to_string(),
            "Logitech".to_string(),
        ])
        .unwrap();
        assert_eq!(config.device_name_query.as_deref(), Some("Logitech"));
    }

    #[test]
    fn parse_rejects_conflicting_device_flags() {
        let err = CliConfig::parse_from(vec![
            "--device".to_string(),
            "1".to_string(),
            "--device-name".to_string(),
            "cam".to_string(),
        ])
        .unwrap_err();
        assert_eq!(
            err,
            CliError::Message(
                "`--device-name` cannot be used together with `--device`".to_string()
            )
        );
    }

    #[test]
    fn parse_rejects_device_name_without_camera_mode() {
        let err = CliConfig::parse_from(vec!["--device-name".to_string(), "cam".to_string()])
            .unwrap_err();
        assert_eq!(
            err,
            CliError::Message(
                "`--device-name` requires `--camera` or `--list-cameras`".to_string()
            )
        );
    }

    #[test]
    fn parse_allows_device_name_with_list_cameras() {
        let config = CliConfig::parse_from(vec![
            "--list-cameras".to_string(),
            "--device-name".to_string(),
            "cam".to_string(),
        ])
        .unwrap();
        assert!(config.list_cameras);
        assert_eq!(config.device_name_query.as_deref(), Some("cam"));
    }

    #[test]
    fn parse_event_log_path() {
        let config =
            CliConfig::parse_from(vec!["--event-log".to_string(), "events.jsonl".to_string()])
                .unwrap();
        assert_eq!(
            config.event_log_path,
            Some(std::path::PathBuf::from("events.jsonl"))
        );
    }

    #[test]
    fn parse_rejects_event_log_with_list_cameras() {
        let err = CliConfig::parse_from(vec![
            "--list-cameras".to_string(),
            "--event-log".to_string(),
            "events.jsonl".to_string(),
        ])
        .unwrap_err();
        assert_eq!(
            err,
            CliError::Message(
                "`--event-log` cannot be used together with `--list-cameras`".to_string()
            )
        );
    }

    #[test]
    fn parse_eval_dataset_flags() {
        let config = CliConfig::parse_from(vec![
            "--eval-detection-jsonl".to_string(),
            "det.jsonl".to_string(),
            "--eval-detection-coco-gt".to_string(),
            "det-gt.json".to_string(),
            "--eval-detection-coco-pred".to_string(),
            "det-pred.json".to_string(),
            "--eval-detection-openimages-gt".to_string(),
            "det-openimages-gt.csv".to_string(),
            "--eval-detection-openimages-pred".to_string(),
            "det-openimages-pred.csv".to_string(),
            "--eval-detection-yolo-manifest".to_string(),
            "det-yolo-manifest.txt".to_string(),
            "--eval-detection-yolo-gt-dir".to_string(),
            "det-yolo-gt".to_string(),
            "--eval-detection-yolo-pred-dir".to_string(),
            "det-yolo-pred".to_string(),
            "--eval-detection-voc-manifest".to_string(),
            "det-voc-manifest.txt".to_string(),
            "--eval-detection-voc-gt-dir".to_string(),
            "det-voc-gt".to_string(),
            "--eval-detection-voc-pred-dir".to_string(),
            "det-voc-pred".to_string(),
            "--eval-detection-kitti-manifest".to_string(),
            "det-kitti-manifest.txt".to_string(),
            "--eval-detection-kitti-gt-dir".to_string(),
            "det-kitti-gt".to_string(),
            "--eval-detection-kitti-pred-dir".to_string(),
            "det-kitti-pred".to_string(),
            "--eval-detection-widerface-gt".to_string(),
            "det-widerface-gt.txt".to_string(),
            "--eval-detection-widerface-pred".to_string(),
            "det-widerface-pred.txt".to_string(),
            "--eval-tracking-jsonl".to_string(),
            "trk.jsonl".to_string(),
            "--eval-tracking-mot-gt".to_string(),
            "trk-gt.txt".to_string(),
            "--eval-tracking-mot-pred".to_string(),
            "trk-pred.txt".to_string(),
            "--eval-iou".to_string(),
            "0.55".to_string(),
            "--eval-score".to_string(),
            "0.25".to_string(),
        ])
        .unwrap();
        assert_eq!(
            config.eval_detection_dataset_path,
            Some(std::path::PathBuf::from("det.jsonl"))
        );
        assert_eq!(
            config.eval_detection_coco_gt_path,
            Some(std::path::PathBuf::from("det-gt.json"))
        );
        assert_eq!(
            config.eval_detection_coco_pred_path,
            Some(std::path::PathBuf::from("det-pred.json"))
        );
        assert_eq!(
            config.eval_detection_openimages_gt_path,
            Some(std::path::PathBuf::from("det-openimages-gt.csv"))
        );
        assert_eq!(
            config.eval_detection_openimages_pred_path,
            Some(std::path::PathBuf::from("det-openimages-pred.csv"))
        );
        assert_eq!(
            config.eval_detection_yolo_manifest_path,
            Some(std::path::PathBuf::from("det-yolo-manifest.txt"))
        );
        assert_eq!(
            config.eval_detection_yolo_gt_dir_path,
            Some(std::path::PathBuf::from("det-yolo-gt"))
        );
        assert_eq!(
            config.eval_detection_yolo_pred_dir_path,
            Some(std::path::PathBuf::from("det-yolo-pred"))
        );
        assert_eq!(
            config.eval_detection_voc_manifest_path,
            Some(std::path::PathBuf::from("det-voc-manifest.txt"))
        );
        assert_eq!(
            config.eval_detection_voc_gt_dir_path,
            Some(std::path::PathBuf::from("det-voc-gt"))
        );
        assert_eq!(
            config.eval_detection_voc_pred_dir_path,
            Some(std::path::PathBuf::from("det-voc-pred"))
        );
        assert_eq!(
            config.eval_detection_kitti_manifest_path,
            Some(std::path::PathBuf::from("det-kitti-manifest.txt"))
        );
        assert_eq!(
            config.eval_detection_kitti_gt_dir_path,
            Some(std::path::PathBuf::from("det-kitti-gt"))
        );
        assert_eq!(
            config.eval_detection_kitti_pred_dir_path,
            Some(std::path::PathBuf::from("det-kitti-pred"))
        );
        assert_eq!(
            config.eval_detection_widerface_gt_path,
            Some(std::path::PathBuf::from("det-widerface-gt.txt"))
        );
        assert_eq!(
            config.eval_detection_widerface_pred_path,
            Some(std::path::PathBuf::from("det-widerface-pred.txt"))
        );
        assert_eq!(
            config.eval_tracking_dataset_path,
            Some(std::path::PathBuf::from("trk.jsonl"))
        );
        assert_eq!(
            config.eval_tracking_mot_gt_path,
            Some(std::path::PathBuf::from("trk-gt.txt"))
        );
        assert_eq!(
            config.eval_tracking_mot_pred_path,
            Some(std::path::PathBuf::from("trk-pred.txt"))
        );
        assert_eq!(config.eval_iou_threshold, 0.55);
        assert_eq!(config.eval_score_threshold, 0.25);
    }

    #[test]
    fn parse_rejects_partial_coco_eval_flags() {
        let err = CliConfig::parse_from(vec![
            "--eval-detection-coco-gt".to_string(),
            "det-gt.json".to_string(),
        ])
        .unwrap_err();
        assert_eq!(
            err,
            CliError::Message(
                "`--eval-detection-coco-gt` and `--eval-detection-coco-pred` must be provided together".to_string()
            )
        );
    }

    #[test]
    fn parse_rejects_partial_openimages_eval_flags() {
        let err = CliConfig::parse_from(vec![
            "--eval-detection-openimages-gt".to_string(),
            "det-openimages-gt.csv".to_string(),
        ])
        .unwrap_err();
        assert_eq!(
            err,
            CliError::Message(
                "`--eval-detection-openimages-gt` and `--eval-detection-openimages-pred` must be provided together".to_string()
            )
        );
    }

    #[test]
    fn parse_rejects_partial_yolo_eval_flags() {
        let err = CliConfig::parse_from(vec![
            "--eval-detection-yolo-manifest".to_string(),
            "det-yolo-manifest.txt".to_string(),
        ])
        .unwrap_err();
        assert_eq!(
            err,
            CliError::Message(
                "`--eval-detection-yolo-manifest`, `--eval-detection-yolo-gt-dir`, and `--eval-detection-yolo-pred-dir` must be provided together".to_string()
            )
        );
    }

    #[test]
    fn parse_rejects_partial_voc_eval_flags() {
        let err = CliConfig::parse_from(vec![
            "--eval-detection-voc-manifest".to_string(),
            "det-voc-manifest.txt".to_string(),
        ])
        .unwrap_err();
        assert_eq!(
            err,
            CliError::Message(
                "`--eval-detection-voc-manifest`, `--eval-detection-voc-gt-dir`, and `--eval-detection-voc-pred-dir` must be provided together".to_string()
            )
        );
    }

    #[test]
    fn parse_rejects_partial_kitti_eval_flags() {
        let err = CliConfig::parse_from(vec![
            "--eval-detection-kitti-manifest".to_string(),
            "det-kitti-manifest.txt".to_string(),
        ])
        .unwrap_err();
        assert_eq!(
            err,
            CliError::Message(
                "`--eval-detection-kitti-manifest`, `--eval-detection-kitti-gt-dir`, and `--eval-detection-kitti-pred-dir` must be provided together".to_string()
            )
        );
    }

    #[test]
    fn parse_rejects_partial_widerface_eval_flags() {
        let err = CliConfig::parse_from(vec![
            "--eval-detection-widerface-gt".to_string(),
            "det-widerface-gt.txt".to_string(),
        ])
        .unwrap_err();
        assert_eq!(
            err,
            CliError::Message(
                "`--eval-detection-widerface-gt` and `--eval-detection-widerface-pred` must be provided together".to_string()
            )
        );
    }

    #[test]
    fn parse_rejects_partial_tracking_mot_eval_flags() {
        let err = CliConfig::parse_from(vec![
            "--eval-tracking-mot-gt".to_string(),
            "trk-gt.txt".to_string(),
        ])
        .unwrap_err();
        assert_eq!(
            err,
            CliError::Message(
                "`--eval-tracking-mot-gt` and `--eval-tracking-mot-pred` must be provided together"
                    .to_string()
            )
        );
    }

    #[test]
    fn parse_validate_diagnostics_flags() {
        let config = CliConfig::parse_from(vec![
            "--validate-diagnostics-report".to_string(),
            "diag.json".to_string(),
            "--validate-diagnostics-min-frames".to_string(),
            "10".to_string(),
            "--validate-diagnostics-max-drift-pct".to_string(),
            "12.5".to_string(),
            "--validate-diagnostics-max-dropped".to_string(),
            "2".to_string(),
        ])
        .unwrap();
        assert_eq!(
            config.validate_diagnostics_report_path,
            Some(std::path::PathBuf::from("diag.json"))
        );
        assert_eq!(config.validate_diagnostics_min_frames, 10);
        assert_eq!(config.validate_diagnostics_max_drift_pct, 12.5);
        assert_eq!(config.validate_diagnostics_max_dropped_frames, 2);
    }

    #[test]
    fn parse_rejects_eval_mode_with_camera() {
        let err = CliConfig::parse_from(vec![
            "--camera".to_string(),
            "--eval-detection-jsonl".to_string(),
            "det.jsonl".to_string(),
        ])
        .unwrap_err();
        assert_eq!(
            err,
            CliError::Message(
                "evaluation dataset mode cannot be combined with camera/listing mode".to_string()
            )
        );
    }

    #[test]
    fn parse_rejects_diagnostics_validation_with_camera() {
        let err = CliConfig::parse_from(vec![
            "--validate-diagnostics-report".to_string(),
            "diag.json".to_string(),
            "--camera".to_string(),
        ])
        .unwrap_err();
        assert_eq!(
            err,
            CliError::Message(
                "diagnostics report validation mode cannot be combined with camera/diagnostics/eval/benchmark/event-log modes".to_string()
            )
        );
    }

    #[test]
    fn parse_rejects_eval_mode_with_diagnostics() {
        let err = CliConfig::parse_from(vec![
            "--diagnose-camera".to_string(),
            "--eval-detection-jsonl".to_string(),
            "det.jsonl".to_string(),
        ])
        .unwrap_err();
        assert_eq!(
            err,
            CliError::Message(
                "evaluation dataset mode cannot be combined with `--diagnose-camera`".to_string()
            )
        );
    }

    #[test]
    fn parse_rejects_eval_mode_with_event_log() {
        let err = CliConfig::parse_from(vec![
            "--eval-tracking-jsonl".to_string(),
            "trk.jsonl".to_string(),
            "--event-log".to_string(),
            "events.jsonl".to_string(),
        ])
        .unwrap_err();
        assert_eq!(
            err,
            CliError::Message(
                "evaluation dataset mode cannot be combined with `--event-log`".to_string()
            )
        );
    }

    #[test]
    fn parse_rejects_eval_mode_with_benchmark() {
        let err = CliConfig::parse_from(vec![
            "--eval-detection-jsonl".to_string(),
            "det.jsonl".to_string(),
            "--benchmark".to_string(),
        ])
        .unwrap_err();
        assert_eq!(
            err,
            CliError::Message(
                "evaluation dataset mode cannot be combined with benchmark mode".to_string()
            )
        );
    }

    #[test]
    fn parse_rejects_diagnostics_with_list_cameras() {
        let err = CliConfig::parse_from(vec![
            "--diagnose-camera".to_string(),
            "--list-cameras".to_string(),
        ])
        .unwrap_err();
        assert_eq!(
            err,
            CliError::Message(
                "`--diagnose-camera` cannot be used together with `--list-cameras`".to_string()
            )
        );
    }

    #[test]
    fn parse_rejects_diagnostics_with_benchmark() {
        let err = CliConfig::parse_from(vec![
            "--diagnose-camera".to_string(),
            "--benchmark".to_string(),
        ])
        .unwrap_err();
        assert_eq!(
            err,
            CliError::Message(
                "`--diagnose-camera` cannot be used together with `--benchmark`".to_string()
            )
        );
    }

    #[test]
    fn parse_rejects_diagnostics_validation_thresholds_without_report() {
        let err = CliConfig::parse_from(vec![
            "--validate-diagnostics-max-drift-pct".to_string(),
            "10".to_string(),
        ])
        .unwrap_err();
        assert_eq!(
            err,
            CliError::Message(
                "diagnostics validation thresholds require `--validate-diagnostics-report`"
                    .to_string()
            )
        );
    }

    #[test]
    fn parse_rejects_diagnose_report_without_diagnostics_mode() {
        let err = CliConfig::parse_from(vec![
            "--diagnose-report".to_string(),
            "diag.json".to_string(),
        ])
        .unwrap_err();
        assert_eq!(
            err,
            CliError::Message("`--diagnose-report` requires `--diagnose-camera`".to_string())
        );
    }

    #[test]
    fn parse_rejects_invalid_eval_iou() {
        let err =
            CliConfig::parse_from(vec!["--eval-iou".to_string(), "2.0".to_string()]).unwrap_err();
        assert_eq!(
            err,
            CliError::Message("--eval-iou must be finite and in [0, 1]".to_string())
        );
    }

    #[test]
    fn parse_camera_mode_and_options() {
        let args = vec![
            "--camera".to_string(),
            "--device".to_string(),
            "2".to_string(),
            "--width".to_string(),
            "1280".to_string(),
            "--height".to_string(),
            "720".to_string(),
            "--fps".to_string(),
            "60".to_string(),
            "--track-iou".to_string(),
            "0.35".to_string(),
            "--track-max-missed".to_string(),
            "4".to_string(),
            "--track-max".to_string(),
            "128".to_string(),
            "--recognition-threshold".to_string(),
            "0.88".to_string(),
            "--max-frames".to_string(),
            "120".to_string(),
            "--identities".to_string(),
            "ids.json".to_string(),
        ];
        let config = CliConfig::parse_from(args).unwrap();
        assert!(config.camera);
        assert_eq!(config.device_index, 2);
        assert_eq!(config.width, 1280);
        assert_eq!(config.height, 720);
        assert_eq!(config.fps, 60);
        assert_eq!(config.track_iou_threshold, 0.35);
        assert_eq!(config.track_max_missed_frames, 4);
        assert_eq!(config.track_max_tracks, 128);
        assert_eq!(config.recognition_threshold, 0.88);
        assert_eq!(config.max_frames, Some(120));
        assert_eq!(
            config.identities_path,
            Some(std::path::PathBuf::from("ids.json"))
        );
    }

    #[test]
    fn parse_rejects_unknown_argument() {
        let err = CliConfig::parse_from(vec!["--what".to_string()]).unwrap_err();
        assert_eq!(
            err,
            CliError::Message("unknown argument: --what; run with --help for usage".to_string())
        );
    }

    #[test]
    fn parse_rejects_non_positive_resolution() {
        let err = CliConfig::parse_from(vec!["--width".to_string(), "0".to_string()]).unwrap_err();
        assert_eq!(
            err,
            CliError::Message("--width must be greater than 0".to_string())
        );
    }

    #[test]
    fn parse_benchmark_flags() {
        let args = vec![
            "--benchmark".to_string(),
            "--detect-target".to_string(),
            "face".to_string(),
            "--benchmark-report".to_string(),
            "bench.txt".to_string(),
            "--benchmark-baseline".to_string(),
            "baseline.txt".to_string(),
        ];
        let config = CliConfig::parse_from(args).unwrap();
        assert!(config.benchmark);
        assert_eq!(config.detect_target, DetectTarget::Faces);
        assert_eq!(
            config.benchmark_report_path,
            Some(std::path::PathBuf::from("bench.txt"))
        );
        assert_eq!(
            config.benchmark_baseline_path,
            Some(std::path::PathBuf::from("baseline.txt"))
        );
    }

    #[test]
    fn parse_baseline_enables_benchmark_mode() {
        let config = CliConfig::parse_from(vec![
            "--benchmark-baseline".to_string(),
            "baseline.txt".to_string(),
        ])
        .unwrap();
        assert!(config.benchmark);
        assert_eq!(
            config.benchmark_baseline_path,
            Some(std::path::PathBuf::from("baseline.txt"))
        );
    }

    #[test]
    fn parse_rejects_invalid_track_iou() {
        let err =
            CliConfig::parse_from(vec!["--track-iou".to_string(), "1.5".to_string()]).unwrap_err();
        assert_eq!(
            err,
            CliError::Message("--track-iou must be finite and in [0, 1]".to_string())
        );
    }

    #[test]
    fn parse_rejects_invalid_recognition_threshold() {
        let err = CliConfig::parse_from(vec![
            "--recognition-threshold".to_string(),
            "2.0".to_string(),
        ])
        .unwrap_err();
        assert_eq!(
            err,
            CliError::Message("--recognition-threshold must be finite and in [-1, 1]".to_string())
        );
    }

    #[test]
    fn parse_rejects_invalid_detect_target() {
        let err = CliConfig::parse_from(vec!["--detect-target".to_string(), "animals".to_string()])
            .unwrap_err();
        assert_eq!(
            err,
            CliError::Message(
                "invalid --detect-target `animals`; expected one of: people, face".to_string()
            )
        );
    }

    #[test]
    fn parse_detect_overrides() {
        let config = CliConfig::parse_from(vec![
            "--detect-score".to_string(),
            "0.42".to_string(),
            "--detect-min-area".to_string(),
            "12".to_string(),
            "--detect-iou".to_string(),
            "0.33".to_string(),
            "--detect-max".to_string(),
            "20".to_string(),
        ])
        .unwrap();
        assert_eq!(config.detect_score_threshold, Some(0.42));
        assert_eq!(config.detect_min_area, Some(12));
        assert_eq!(config.detect_iou_threshold, Some(0.33));
        assert_eq!(config.detect_max_detections, Some(20));
    }

    #[test]
    fn parse_rejects_invalid_detect_score() {
        let err = CliConfig::parse_from(vec!["--detect-score".to_string(), "1.2".to_string()])
            .unwrap_err();
        assert_eq!(
            err,
            CliError::Message("--detect-score must be finite and in [0, 1]".to_string())
        );
    }

    #[test]
    fn parse_rejects_zero_detect_limits() {
        let err = CliConfig::parse_from(vec!["--detect-min-area".to_string(), "0".to_string()])
            .unwrap_err();
        assert_eq!(
            err,
            CliError::Message("--detect-min-area must be greater than 0".to_string())
        );

        let err =
            CliConfig::parse_from(vec!["--detect-max".to_string(), "0".to_string()]).unwrap_err();
        assert_eq!(
            err,
            CliError::Message("--detect-max must be greater than 0".to_string())
        );
    }
}
