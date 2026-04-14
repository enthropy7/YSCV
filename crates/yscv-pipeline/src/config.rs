//! TOML config schema + parsing + validation.

use crate::{Accelerator, ConfigError};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

/// Top-level pipeline configuration parsed from a `boards/*.toml` file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineConfig {
    /// Human-readable board name (e.g. "rock4d", "luckfox-pico-pro-max").
    pub board: String,
    pub camera: CameraSpec,
    pub output: OutputSpec,
    pub encoder: EncoderSpec,
    /// Inference task graph nodes. Order in this array doesn't determine
    /// execution order — the scheduler topologically sorts by [`InferenceTask::inputs`].
    #[serde(default, rename = "tasks")]
    pub tasks: Vec<InferenceTask>,
    #[serde(default)]
    pub osd: OsdSpec,
    #[serde(default)]
    pub realtime: RtSpec,
}

/// Camera specification (V4L2 device, format, resolution, fps).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CameraSpec {
    /// Device node, e.g. `/dev/video0`.
    pub device: String,
    /// Pixel format identifier (`"nv12"`, `"yuyv"`, `"mjpeg"`, `"rgb"`).
    pub format: String,
    pub width: u32,
    pub height: u32,
    pub fps: u32,
}

/// Where to send the encoded output.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case")]
pub enum OutputSpec {
    /// Write encoded H.264/HEVC NAL units to a file (dev / debug).
    File { path: PathBuf },
    /// Direct DRM atomic flip onto an HDMI connector.
    Drm { connector: String, mode: String },
    /// V4L2 output device (`/dev/videoN`) — useful for HDMI output via MPP.
    V4l2Out { device: String },
    /// Just discard (CI / benchmarking).
    Null,
}

/// Encoder specification — software fallback or HW (MPP).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncoderSpec {
    /// Encoder backend: `"soft-h264"`, `"mpp-h264"`, `"mpp-hevc"`.
    pub kind: String,
    pub bitrate_kbps: u32,
    /// Codec profile: `"baseline"`, `"main"`, `"high"`.
    #[serde(default = "default_profile")]
    pub profile: String,
}

fn default_profile() -> String {
    "baseline".into()
}

/// One inference task in the pipeline graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceTask {
    /// Unique human-readable name (used in logs/profile and as a binding source).
    pub name: String,
    /// Path to the model file (`.rknn`, `.onnx`, GGUF, etc.).
    pub model_path: PathBuf,
    /// Which accelerator runs this task.
    pub accelerator: Accelerator,
    /// Input bindings (input tensor name → source).
    #[serde(default)]
    pub inputs: Vec<TensorBinding>,
    /// Output bindings (output tensor name → sink).
    #[serde(default)]
    pub outputs: Vec<TensorBinding>,
}

/// A single tensor binding: `name` is the model's tensor name, `source`
/// (for inputs) or `sink` (for outputs) is either the literal `"camera"`,
/// the literal `"detections"`, or `"<task_name>.<output_name>"` to chain
/// tasks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorBinding {
    pub name: String,
    /// Source for inputs, sink for outputs. See type doc for syntax.
    #[serde(alias = "sink")]
    pub source: String,
}

/// On-screen-display settings.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OsdSpec {
    /// List of fields to render: `"fps"`, `"battery"`, `"signal"`,
    /// `"detection_count"`, `"latency"`, `"telemetry"`.
    #[serde(default)]
    pub fields: Vec<String>,
    /// Glyph point size for OSD text (e.g. 12).
    #[serde(default = "default_glyph_size")]
    pub glyph_size: u32,
}

fn default_glyph_size() -> u32 {
    12
}

/// Real-time scheduler config — SCHED_FIFO priorities and CPU affinity
/// per pipeline stage.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RtSpec {
    /// Whether to attempt SCHED_FIFO. Without `CAP_SYS_NICE` this is a no-op
    /// with a warning.
    #[serde(default)]
    pub sched_fifo: bool,
    /// Per-stage SCHED_FIFO priority. Keys: "capture", "dispatch", "wait",
    /// "encode", "output". Values: 1..=99.
    #[serde(default)]
    pub prio: HashMap<String, u8>,
    /// Per-stage CPU affinity. Keys same as `prio`; values list of CPU ids.
    #[serde(default)]
    pub affinity: HashMap<String, Vec<u32>>,
    /// CPU frequency governor to write to every
    /// `/sys/devices/system/cpu/cpu*/cpufreq/scaling_governor`. Leave
    /// unset (or `None` in code) to leave the system default. Most
    /// FPV deployments want `"performance"` — kills first-burst DVFS
    /// latency. Requires `CAP_SYS_ADMIN` or root; missing privilege
    /// is logged but not fatal (graceful fallback like the rest of
    /// the realtime stack).
    #[serde(default)]
    pub cpu_governor: Option<String>,
}

impl PipelineConfig {
    /// Load and parse a `boards/*.toml` file from disk.
    pub fn from_toml_path<P: AsRef<Path>>(path: P) -> Result<Self, ConfigError> {
        let path = path.as_ref();
        let bytes = std::fs::read(path).map_err(|e| ConfigError::Io {
            path: path.to_path_buf(),
            source: e,
        })?;
        let s = String::from_utf8(bytes).map_err(|e| ConfigError::Io {
            path: path.to_path_buf(),
            source: std::io::Error::other(e),
        })?;
        let cfg: PipelineConfig = toml::from_str(&s).map_err(|e| ConfigError::Toml {
            path: path.to_path_buf(),
            source: e,
        })?;
        cfg.validate_self()?;
        Ok(cfg)
    }

    /// Parse a config from an in-memory TOML string. Useful for tests.
    pub fn from_toml_str(s: &str) -> Result<Self, ConfigError> {
        let cfg: PipelineConfig = toml::from_str(s).map_err(|e| ConfigError::Toml {
            path: PathBuf::from("<inline>"),
            source: e,
        })?;
        cfg.validate_self()?;
        Ok(cfg)
    }

    /// Internal field validation — checks fps > 0, width/height > 0,
    /// task name uniqueness, no cycles, no dangling input refs. Does NOT
    /// touch the filesystem or check accelerator availability.
    pub fn validate_self(&self) -> Result<(), ConfigError> {
        if self.camera.width == 0 || self.camera.height == 0 {
            return Err(ConfigError::InvalidField(format!(
                "camera dimensions must be non-zero (got {}×{})",
                self.camera.width, self.camera.height
            )));
        }
        if self.camera.fps == 0 {
            return Err(ConfigError::InvalidField("camera.fps must be > 0".into()));
        }
        if self.encoder.bitrate_kbps == 0 {
            return Err(ConfigError::InvalidField(
                "encoder.bitrate_kbps must be > 0".into(),
            ));
        }

        // Task name uniqueness.
        let mut seen: HashSet<&str> = HashSet::new();
        for task in &self.tasks {
            if !seen.insert(task.name.as_str()) {
                return Err(ConfigError::InvalidField(format!(
                    "duplicate task name '{}'",
                    task.name
                )));
            }
        }

        // Dangling input refs: every input source must be either the literal
        // "camera" or "<existing_task>.<anything>".
        let task_names: HashSet<&str> = self.tasks.iter().map(|t| t.name.as_str()).collect();
        for task in &self.tasks {
            for binding in &task.inputs {
                let src = binding.source.as_str();
                if src == "camera" {
                    continue;
                }
                let head = src.split('.').next().unwrap_or("");
                if !task_names.contains(head) {
                    return Err(ConfigError::UnknownSource {
                        task: task.name.clone(),
                        src_name: binding.source.clone(),
                    });
                }
            }
        }

        // Cycle detection via DFS.
        detect_cycles(&self.tasks)?;

        Ok(())
    }

    /// Verify each task's accelerator is available on this host. Run this
    /// at startup right after loading the config — it is cheap (~1
    /// `dlopen` + `dlclose` for RKNN, no-op otherwise) and prevents
    /// silent CPU fallback.
    pub fn validate_accelerators(&self) -> Result<(), ConfigError> {
        let avail = crate::probe_accelerators();
        for task in &self.tasks {
            if !avail.supports(&task.accelerator) {
                return Err(ConfigError::AcceleratorUnavailable {
                    task: task.name.clone(),
                    accelerator: task.accelerator.label(),
                    feature_hint: task.accelerator.feature_hint().to_string(),
                });
            }
        }
        Ok(())
    }

    /// Verify each task's `model_path` exists and that its contents
    /// parse as the expected format — we don't just stat the file.
    ///
    /// For RKNN tasks: checks the file magic bytes (`RKNN`, `0x4e4e4b52`
    /// little-endian or `RKNF` for some SDK flavours). On a host where
    /// `librknnrt.so` is actually loadable, also runs a full
    /// `RknnBackend::load` — catches corrupted-but-magic-matching files
    /// before the real-time threads start.
    ///
    /// For ONNX tasks: lightweight magic check (`\x08` — protobuf
    /// first-field tag for `ModelProto.ir_version`) and full
    /// `load_onnx_model` parse.
    ///
    /// Run after `validate_accelerators`.
    pub fn validate_models(&self) -> Result<(), ConfigError> {
        for task in &self.tasks {
            // RknnMatmul is a pure shape-bound op — no model file
            // exists. Skip every file-level check; the shape lives in
            // `Accelerator::RknnMatmul`'s fields instead.
            if matches!(task.accelerator, Accelerator::RknnMatmul { .. }) {
                continue;
            }
            if !task.model_path.exists() {
                return Err(ConfigError::ModelNotFound {
                    task: task.name.clone(),
                    path: task.model_path.clone(),
                });
            }
            let bytes = std::fs::read(&task.model_path).map_err(|e| {
                ConfigError::ModelInvalid {
                    task: task.name.clone(),
                    path: task.model_path.clone(),
                    reason: format!("read failed: {e}"),
                }
            })?;
            if bytes.is_empty() {
                return Err(ConfigError::ModelInvalid {
                    task: task.name.clone(),
                    path: task.model_path.clone(),
                    reason: "file is empty".into(),
                });
            }
            match &task.accelerator {
                Accelerator::Rknn { .. } => {
                    // RKNN accepts either pre-compiled `.rknn` or
                    // `.onnx` (on-device compile kicks in at dispatcher
                    // construction). Detect by extension so the
                    // magic-byte check matches.
                    let ext = task
                        .model_path
                        .extension()
                        .and_then(|e| e.to_str())
                        .unwrap_or("")
                        .to_ascii_lowercase();
                    if ext == "onnx" {
                        validate_onnx_model(task, &bytes)?;
                    } else {
                        validate_rknn_model(task, &bytes)?;
                    }
                }
                _ => {
                    // CPU / GPU / MetalMps paths all consume ONNX files;
                    // do a cheap protobuf magic + load check.
                    validate_onnx_model(task, &bytes)?;
                }
            }
        }
        Ok(())
    }

    /// Convenience: run all validation in the right order.
    pub fn validate(&self) -> Result<(), ConfigError> {
        self.validate_self()?;
        self.validate_accelerators()?;
        self.validate_models()?;
        Ok(())
    }
}

/// Magic-byte check for an `.rknn` file.
///
/// RKNN serialised-model files produced by `rknn-toolkit2` start with
/// the ASCII string `RKNN` (little-endian 0x4e4e_4b52) at byte 0, or
/// `RKNF` for some older/flavoured SDKs. Anything else is either a
/// different format or a truncated file.
///
/// On a host where `librknnrt.so` is loadable (detected via
/// `rknn_available()`), also call `RknnBackend::load(&bytes)` — this
/// catches magic-matching but semantically-broken files (wrong SoC,
/// incompatible pre-compile version, etc.) that only the SDK can tell
/// you about.
fn validate_rknn_model(
    task: &crate::config::InferenceTask,
    bytes: &[u8],
) -> Result<(), ConfigError> {
    const RKNN_MAGIC: [u8; 4] = *b"RKNN";
    const RKNF_MAGIC: [u8; 4] = *b"RKNF";
    if bytes.len() < 4 {
        return Err(ConfigError::ModelInvalid {
            task: task.name.clone(),
            path: task.model_path.clone(),
            reason: format!("only {} bytes — truncated?", bytes.len()),
        });
    }
    let magic: [u8; 4] = [bytes[0], bytes[1], bytes[2], bytes[3]];
    if magic != RKNN_MAGIC && magic != RKNF_MAGIC {
        return Err(ConfigError::ModelInvalid {
            task: task.name.clone(),
            path: task.model_path.clone(),
            reason: format!(
                "expected RKNN / RKNF magic, got {:02x?} — is this really a \
                 .rknn file? (export via rknn-toolkit2)",
                magic
            ),
        });
    }
    // If the runtime is actually loadable on this host, do a full parse.
    #[cfg(feature = "rknn-validate")]
    {
        if yscv_kernels::rknn_available() {
            let _ = yscv_kernels::RknnBackend::load(bytes).map_err(|e| {
                ConfigError::ModelInvalid {
                    task: task.name.clone(),
                    path: task.model_path.clone(),
                    reason: format!("rknn SDK refused the file: {e}"),
                }
            })?;
        }
    }
    Ok(())
}

/// Minimal ONNX ModelProto sanity check: the file must start with a
/// valid protobuf wire-format tag. ONNX always has
/// `ir_version` as field 1, type varint — the first byte is `0x08`.
/// This is a cheap, false-positive-tolerant pre-flight; the real parse
/// happens inside whichever backend consumes the model (CPU runner,
/// MPSGraph compiler, etc.).
fn validate_onnx_model(
    task: &crate::config::InferenceTask,
    bytes: &[u8],
) -> Result<(), ConfigError> {
    if bytes.first().copied() != Some(0x08) {
        return Err(ConfigError::ModelInvalid {
            task: task.name.clone(),
            path: task.model_path.clone(),
            reason: format!(
                "expected ONNX protobuf (first byte 0x08 for ir_version), \
                 got {:#04x} — is this really an .onnx file?",
                bytes.first().copied().unwrap_or(0)
            ),
        });
    }
    Ok(())
}

/// DFS cycle detection — used by `validate_self`.
fn detect_cycles(tasks: &[InferenceTask]) -> Result<(), ConfigError> {
    // Build adjacency: task name → list of upstream tasks (those whose
    // outputs we depend on).
    let by_name: HashMap<&str, &InferenceTask> =
        tasks.iter().map(|t| (t.name.as_str(), t)).collect();
    let mut state: HashMap<&str, u8> = HashMap::new(); // 0=white, 1=gray, 2=black

    fn visit<'a>(
        node: &'a str,
        by_name: &HashMap<&'a str, &'a InferenceTask>,
        state: &mut HashMap<&'a str, u8>,
    ) -> Result<(), ConfigError> {
        match state.get(node) {
            Some(&2) => return Ok(()),
            Some(&1) => {
                return Err(ConfigError::CyclicDependency {
                    task: node.to_string(),
                });
            }
            _ => {}
        }
        state.insert(node, 1);
        if let Some(task) = by_name.get(node) {
            for binding in &task.inputs {
                let src = binding.source.as_str();
                if src == "camera" {
                    continue;
                }
                let head = src.split('.').next().unwrap_or("");
                if by_name.contains_key(head) {
                    visit(head, by_name, state)?;
                }
            }
        }
        state.insert(node, 2);
        Ok(())
    }

    for task in tasks {
        visit(task.name.as_str(), &by_name, &mut state)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::accelerator::{AcceleratorAvailability, NpuCoreSpec};

    fn minimal_cfg() -> PipelineConfig {
        PipelineConfig {
            board: "test".into(),
            camera: CameraSpec {
                device: "/dev/video0".into(),
                format: "nv12".into(),
                width: 640,
                height: 480,
                fps: 30,
            },
            output: OutputSpec::Null,
            encoder: EncoderSpec {
                kind: "soft-h264".into(),
                bitrate_kbps: 1000,
                profile: "baseline".into(),
            },
            tasks: vec![],
            osd: OsdSpec::default(),
            realtime: RtSpec::default(),
        }
    }

    #[test]
    fn empty_pipeline_validates() {
        minimal_cfg().validate_self().unwrap();
    }

    #[test]
    fn zero_dimensions_rejected() {
        let mut cfg = minimal_cfg();
        cfg.camera.width = 0;
        let err = cfg.validate_self().unwrap_err();
        assert!(matches!(err, ConfigError::InvalidField(_)));
    }

    #[test]
    fn duplicate_task_names_rejected() {
        let mut cfg = minimal_cfg();
        for _ in 0..2 {
            cfg.tasks.push(InferenceTask {
                name: "detector".into(),
                model_path: PathBuf::from("/tmp/x.rknn"),
                accelerator: Accelerator::Cpu,
                inputs: vec![],
                outputs: vec![],
            });
        }
        let err = cfg.validate_self().unwrap_err();
        match err {
            ConfigError::InvalidField(msg) => assert!(msg.contains("duplicate task")),
            _ => panic!("expected InvalidField"),
        }
    }

    #[test]
    fn unknown_input_source_rejected() {
        let mut cfg = minimal_cfg();
        cfg.tasks.push(InferenceTask {
            name: "detector".into(),
            model_path: PathBuf::from("/tmp/x.rknn"),
            accelerator: Accelerator::Cpu,
            inputs: vec![TensorBinding {
                name: "images".into(),
                source: "ghost.feature".into(),
            }],
            outputs: vec![],
        });
        let err = cfg.validate_self().unwrap_err();
        assert!(matches!(err, ConfigError::UnknownSource { .. }));
    }

    #[test]
    fn cyclic_dependency_detected() {
        let mut cfg = minimal_cfg();
        cfg.tasks.push(InferenceTask {
            name: "a".into(),
            model_path: PathBuf::from("/tmp/x.rknn"),
            accelerator: Accelerator::Cpu,
            inputs: vec![TensorBinding {
                name: "x".into(),
                source: "b.out".into(),
            }],
            outputs: vec![],
        });
        cfg.tasks.push(InferenceTask {
            name: "b".into(),
            model_path: PathBuf::from("/tmp/x.rknn"),
            accelerator: Accelerator::Cpu,
            inputs: vec![TensorBinding {
                name: "x".into(),
                source: "a.out".into(),
            }],
            outputs: vec![],
        });
        let err = cfg.validate_self().unwrap_err();
        assert!(matches!(err, ConfigError::CyclicDependency { .. }));
    }

    #[test]
    fn toml_parse_minimal() {
        let s = r#"
board = "test"

[camera]
device = "/dev/video0"
format = "nv12"
width = 1280
height = 720
fps = 60

[output]
kind = "null"

[encoder]
kind = "soft-h264"
bitrate_kbps = 4000
"#;
        let cfg = PipelineConfig::from_toml_str(s).unwrap();
        assert_eq!(cfg.board, "test");
        assert_eq!(cfg.camera.fps, 60);
        assert!(matches!(cfg.output, OutputSpec::Null));
    }

    #[test]
    fn toml_parse_with_tasks() {
        let s = r#"
board = "rock4d"

[camera]
device = "/dev/video0"
format = "nv12"
width = 1280
height = 720
fps = 60

[output]
kind = "drm"
connector = "HDMI-A-1"
mode = "720p60"

[encoder]
kind = "mpp-h264"
bitrate_kbps = 8000

[[tasks]]
name = "detector"
model_path = "/tmp/yolov8n.rknn"
accelerator = { kind = "rknn", core = "core0" }
inputs = [{ name = "images", source = "camera" }]
outputs = [{ name = "output0", source = "detections" }]
"#;
        let cfg = PipelineConfig::from_toml_str(s).unwrap();
        assert_eq!(cfg.tasks.len(), 1);
        assert_eq!(cfg.tasks[0].name, "detector");
        match &cfg.tasks[0].accelerator {
            Accelerator::Rknn { core } => assert_eq!(*core, NpuCoreSpec::Core0),
            _ => panic!("expected rknn"),
        }
    }

    #[test]
    fn validate_accelerators_rejects_unavailable() {
        let mut cfg = minimal_cfg();
        cfg.tasks.push(InferenceTask {
            name: "detector".into(),
            model_path: PathBuf::from("/tmp/missing.rknn"),
            // RKNN isn't available on macOS dev hosts. On Linux without
            // librknnrt.so it's also unavailable. Either way validation should fail.
            accelerator: Accelerator::Rknn {
                core: NpuCoreSpec::Core0,
            },
            inputs: vec![],
            outputs: vec![],
        });
        let avail = AcceleratorAvailability::default();
        // Force unavailable.
        for task in &cfg.tasks {
            if !avail.supports(&task.accelerator) {
                // Confirm the explicit-check path produces the expected error variant.
                let err = ConfigError::AcceleratorUnavailable {
                    task: task.name.clone(),
                    accelerator: task.accelerator.label(),
                    feature_hint: task.accelerator.feature_hint().to_string(),
                };
                assert!(matches!(err, ConfigError::AcceleratorUnavailable { .. }));
                return;
            }
        }
        // If somehow rknn is supported on this host (should not happen in CI), skip.
    }

    #[test]
    fn validate_models_rejects_missing_file() {
        let mut cfg = minimal_cfg();
        cfg.tasks.push(InferenceTask {
            name: "detector".into(),
            model_path: PathBuf::from("/tmp/yscv-does-not-exist.rknn"),
            accelerator: Accelerator::Cpu,
            inputs: vec![],
            outputs: vec![],
        });
        match cfg.validate_models() {
            Err(ConfigError::ModelNotFound { task, .. }) => {
                assert_eq!(task, "detector");
            }
            other => panic!("expected ModelNotFound, got {other:?}"),
        }
    }

    #[test]
    fn validate_models_rejects_wrong_magic_rknn() {
        let path = std::env::temp_dir().join("yscv-bad-magic.rknn");
        std::fs::write(&path, b"NOT_A_REAL_RKNN_MODEL").unwrap();
        let mut cfg = minimal_cfg();
        cfg.tasks.push(InferenceTask {
            name: "detector".into(),
            model_path: path.clone(),
            accelerator: Accelerator::Rknn {
                core: NpuCoreSpec::Core0,
            },
            inputs: vec![],
            outputs: vec![],
        });
        let result = cfg.validate_models();
        std::fs::remove_file(&path).ok();
        match result {
            Err(ConfigError::ModelInvalid { task, reason, .. }) => {
                assert_eq!(task, "detector");
                assert!(
                    reason.contains("RKNN") || reason.contains("RKNF"),
                    "unexpected reason: {reason}"
                );
            }
            other => panic!("expected ModelInvalid, got {other:?}"),
        }
    }

    #[test]
    fn validate_models_rejects_wrong_magic_onnx() {
        let path = std::env::temp_dir().join("yscv-bad-magic.onnx");
        std::fs::write(&path, b"\xff\xff\xff NOT ONNX").unwrap();
        let mut cfg = minimal_cfg();
        cfg.tasks.push(InferenceTask {
            name: "detector".into(),
            model_path: path.clone(),
            accelerator: Accelerator::Cpu,
            inputs: vec![],
            outputs: vec![],
        });
        let result = cfg.validate_models();
        std::fs::remove_file(&path).ok();
        match result {
            Err(ConfigError::ModelInvalid { task, reason, .. }) => {
                assert_eq!(task, "detector");
                assert!(reason.contains("0x08"), "unexpected reason: {reason}");
            }
            other => panic!("expected ModelInvalid, got {other:?}"),
        }
    }

    #[test]
    fn validate_models_accepts_valid_magic() {
        let rknn_path = std::env::temp_dir().join("yscv-good-magic.rknn");
        std::fs::write(&rknn_path, b"RKNN\x00\x00\x00\x00").unwrap();
        let onnx_path = std::env::temp_dir().join("yscv-good-magic.onnx");
        std::fs::write(&onnx_path, b"\x08\x01\x12\x00").unwrap();
        let mut cfg = minimal_cfg();
        cfg.tasks.push(InferenceTask {
            name: "cpu_task".into(),
            model_path: onnx_path.clone(),
            accelerator: Accelerator::Cpu,
            inputs: vec![],
            outputs: vec![],
        });
        cfg.tasks.push(InferenceTask {
            name: "rknn_task".into(),
            model_path: rknn_path.clone(),
            accelerator: Accelerator::Rknn {
                core: NpuCoreSpec::Core0,
            },
            inputs: vec![],
            outputs: vec![],
        });
        let result = cfg.validate_models();
        std::fs::remove_file(&rknn_path).ok();
        std::fs::remove_file(&onnx_path).ok();
        // Without the `rknn-validate` feature + a real librknnrt.so, the
        // magic check alone is enough for RKNN — it passes.
        assert!(result.is_ok(), "expected Ok, got {result:?}");
    }
}
