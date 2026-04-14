//! Topological scheduler + dispatcher trait + skeleton runtime.
//!
//! This module lays the API surface (`AcceleratorDispatcher`,
//! `TaskScheduler`, `run_pipeline`). The actual hot-path implementation
//! (capture → infer → encode → output) lives in `yscv-video::frame_pipeline`
//! and is wired up in the `examples/edge_pipeline.rs` example.

use crate::config::{InferenceTask, PipelineConfig};
use crate::dispatch::{dispatcher_for, AcceleratorDispatcher};
use crate::error::{ConfigError, Error};
use std::collections::{HashMap, HashSet};

/// Topologically-sorted task list: produces an execution order such that
/// every task appears after all its dependencies.
pub struct TaskScheduler {
    /// Tasks in an order safe to execute sequentially. (Parallel
    /// execution is the responsibility of the runtime — this only
    /// guarantees a valid sequential order.)
    pub order: Vec<String>,
    /// Per-task list of upstream task names (inputs whose source is
    /// `<task>.<output>`).
    pub deps: HashMap<String, Vec<String>>,
}

impl TaskScheduler {
    /// Build a scheduler from a validated pipeline config. Returns an
    /// error if cycles are detected (should be impossible if
    /// `PipelineConfig::validate_self` was called first, but cheap to
    /// double-check).
    pub fn from_config(cfg: &PipelineConfig) -> Result<Self, ConfigError> {
        let mut deps: HashMap<String, Vec<String>> = HashMap::new();
        for task in &cfg.tasks {
            let mut up: Vec<String> = Vec::new();
            for binding in &task.inputs {
                let head = binding
                    .source
                    .as_str()
                    .split('.')
                    .next()
                    .unwrap_or("")
                    .to_string();
                if head == "camera" || head.is_empty() {
                    continue;
                }
                up.push(head);
            }
            deps.insert(task.name.clone(), up);
        }
        let order = topo_sort(&cfg.tasks, &deps)?;
        Ok(Self { order, deps })
    }

    /// Identifies tasks safe to run in parallel — those with all
    /// dependencies already satisfied. Used by a future parallel runtime.
    pub fn ready_tasks(&self, completed: &HashSet<String>) -> Vec<String> {
        self.order
            .iter()
            .filter(|name| {
                !completed.contains(name.as_str())
                    && self
                        .deps
                        .get(name.as_str())
                        .map(|d| d.iter().all(|u| completed.contains(u.as_str())))
                        .unwrap_or(true)
            })
            .cloned()
            .collect()
    }
}

/// DFS-based topological sort. Returns task names in an order where
/// every task follows all its dependencies.
fn topo_sort(
    tasks: &[InferenceTask],
    deps: &HashMap<String, Vec<String>>,
) -> Result<Vec<String>, ConfigError> {
    let mut state: HashMap<String, u8> = HashMap::new();
    let mut order = Vec::with_capacity(tasks.len());

    fn visit(
        node: &str,
        deps: &HashMap<String, Vec<String>>,
        state: &mut HashMap<String, u8>,
        order: &mut Vec<String>,
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
        state.insert(node.to_string(), 1);
        if let Some(ups) = deps.get(node) {
            for up in ups {
                visit(up, deps, state, order)?;
            }
        }
        state.insert(node.to_string(), 2);
        order.push(node.to_string());
        Ok(())
    }

    for task in tasks {
        visit(task.name.as_str(), deps, &mut state, &mut order)?;
    }
    Ok(order)
}

/// Opaque handle returned from [`run_pipeline`]. Holds the long-lived
/// dispatchers + scheduler. Drop to stop the pipeline.
pub struct PipelineHandle {
    /// Computed execution order at construction.
    pub order: Vec<String>,
    /// One dispatcher per task, keyed by task name.
    dispatchers: HashMap<String, Box<dyn AcceleratorDispatcher>>,
    /// Canonical task metadata (inputs + outputs bindings) so
    /// `dispatch_frame` can route tensor names from upstream task
    /// outputs into downstream task inputs.
    tasks: HashMap<String, InferenceTask>,
}

impl PipelineHandle {
    /// Run one full pass through the task graph.
    ///
    /// `camera_inputs` is a slice of `(name, bytes)` pairs representing
    /// the ingress — tensor names here match any task whose
    /// `inputs[i].source == "camera"`. Internal task-to-task wiring
    /// (`source = "<task>.<output>"`) is resolved automatically.
    ///
    /// Returns the final output map: every `source = "<task>.<name>"`
    /// reference plus every task's named outputs, so callers can pick
    /// up whatever the pipeline's "sink" task produced.
    pub fn dispatch_frame(
        &self,
        camera_inputs: &[(&str, &[u8])],
    ) -> Result<HashMap<String, Vec<u8>>, Error> {
        // Accumulator: scratch buffer of every tensor produced so far.
        // Key format:
        //   - "camera.<name>" for camera ingress
        //   - "<task>.<output_name>" for task outputs
        let mut world: HashMap<String, Vec<u8>> = HashMap::with_capacity(camera_inputs.len() * 2);
        for (name, bytes) in camera_inputs {
            world.insert(format!("camera.{name}"), bytes.to_vec());
            // Also expose under bare "camera" so single-input tasks
            // using `source = "camera"` find their bytes without
            // knowing a name. First camera input wins.
            world.entry("camera".to_string()).or_insert_with(|| bytes.to_vec());
        }

        for task_name in &self.order {
            let task = self.tasks.get(task_name).ok_or_else(|| {
                Error::Other(format!("unknown task '{task_name}' in scheduler order"))
            })?;
            let dispatcher = self.dispatchers.get(task_name).ok_or_else(|| {
                Error::Other(format!("no dispatcher for task '{task_name}'"))
            })?;
            // Resolve each input binding to bytes in `world`.
            let mut gathered: Vec<(&str, Vec<u8>)> = Vec::with_capacity(task.inputs.len());
            for binding in &task.inputs {
                let bytes = world.get(&binding.source).cloned().ok_or_else(|| {
                    Error::Other(format!(
                        "task '{}': input '{}' source '{}' has no producer",
                        task_name, binding.name, binding.source
                    ))
                })?;
                gathered.push((binding.name.as_str(), bytes));
            }
            // Slice-of-refs view for the dispatcher.
            let view: Vec<(&str, &[u8])> = gathered
                .iter()
                .map(|(n, b)| (*n, b.as_slice()))
                .collect();
            let outputs = dispatcher.dispatch(&view)?;
            for (out_name, out_bytes) in outputs {
                world.insert(format!("{task_name}.{out_name}"), out_bytes);
            }
        }
        Ok(world)
    }

    /// Best-effort recovery: walk every dispatcher and invoke its
    /// `recover`. Dispatchers that don't support recovery return Ok.
    /// On first error, stops and returns that error.
    pub fn recover_all(&self) -> Result<(), Error> {
        for (name, d) in &self.dispatchers {
            d.recover().map_err(|e| {
                Error::Other(format!("recover dispatcher '{name}' failed: {e}"))
            })?;
        }
        Ok(())
    }

    /// Label of the dispatcher assigned to a task, for logs / OSD.
    pub fn dispatcher_label(&self, task_name: &str) -> Option<&str> {
        self.dispatchers.get(task_name).map(|d| d.label())
    }

    /// Spawn a background watchdog thread that polls a
    /// `PipelineStats5` for alarm conditions and invokes
    /// [`PipelineHandle::recover_all`] when any stage raises one.
    ///
    /// This glues together two pieces that existed independently: the
    /// `StageWatchdog` in `yscv-video` tracks per-stage budget
    /// overruns; the pipeline handle owns recovery. The watchdog is
    /// the supervisor that couples them automatically.
    ///
    /// The returned [`Watchdog`] can be dropped to stop the thread.
    /// Gated on the `realtime` feature because the stats type lives in
    /// `yscv-video`, which this crate only depends on through that
    /// same feature.
    #[cfg(feature = "realtime")]
    pub fn spawn_watchdog(
        self: std::sync::Arc<Self>,
        stats: std::sync::Arc<yscv_video::frame_pipeline_5stage::PipelineStats5>,
        poll_interval: std::time::Duration,
    ) -> Watchdog {
        use std::sync::atomic::{AtomicBool, Ordering};
        let stop = std::sync::Arc::new(AtomicBool::new(false));
        let stop_thread = stop.clone();
        let join = std::thread::spawn(move || {
            while !stop_thread.load(Ordering::Relaxed) {
                std::thread::sleep(poll_interval);
                if stats.watchdog_alarm.load(Ordering::Acquire) {
                    eprintln!(
                        "[yscv-pipeline] watchdog: stage alarm observed, calling recover_all"
                    );
                    if let Err(e) = self.recover_all() {
                        eprintln!(
                            "[yscv-pipeline] watchdog: recover_all failed: {e} — \
                             clearing alarm anyway so we can retry"
                        );
                    }
                    // Clear the top-level + per-stage alarms so the
                    // next overrun is visible.
                    stats.watchdog_capture.clear_alarm();
                    stats.watchdog_infer.clear_alarm();
                    stats.watchdog_encode.clear_alarm();
                    stats.watchdog_alarm.store(false, Ordering::Release);
                }
            }
        });
        Watchdog {
            stop,
            join: Some(join),
        }
    }
}

/// Handle to a running background watchdog spawned by
/// [`PipelineHandle::spawn_watchdog`]. Dropping this flips the stop
/// flag and joins the thread — real-time shutdown ~poll_interval
/// after drop.
#[cfg(feature = "realtime")]
pub struct Watchdog {
    stop: std::sync::Arc<std::sync::atomic::AtomicBool>,
    join: Option<std::thread::JoinHandle<()>>,
}

#[cfg(feature = "realtime")]
impl Watchdog {
    /// Explicitly stop + join. Equivalent to dropping, but returns
    /// the thread's join result so the caller can propagate panics.
    pub fn stop(mut self) -> std::thread::Result<()> {
        self.stop.store(true, std::sync::atomic::Ordering::Release);
        if let Some(j) = self.join.take() {
            j.join()
        } else {
            Ok(())
        }
    }
}

#[cfg(feature = "realtime")]
impl Drop for Watchdog {
    fn drop(&mut self) {
        self.stop.store(true, std::sync::atomic::Ordering::Release);
        if let Some(j) = self.join.take() {
            let _ = j.join();
        }
    }
}

/// Build a runnable pipeline from a config. Validates everything,
/// constructs one dispatcher per task, optionally applies SCHED_FIFO /
/// CPU affinity / mlockall when built with `--features realtime`, and
/// returns a handle the caller drives with `dispatch_frame` from their
/// real-time loop.
///
/// Fail-loud contract: any validation failure, missing feature flag,
/// missing runtime library, or unreadable model file aborts here — the
/// pipeline never "degrades silently" to a slower backend.
///
/// Real-time wiring (feature `realtime`): if
/// `cfg.realtime.sched_fifo == true`, the current thread is elevated
/// to SCHED_FIFO with the priority from `cfg.realtime.prio["dispatch"]`
/// (fallback: no elevation), pinned to the cores in
/// `cfg.realtime.affinity["dispatch"]`, and `mlockall` is attempted.
/// Failures (missing CAP_SYS_NICE, RLIMIT_MEMLOCK too low) are logged
/// to stderr and do NOT abort startup — matches the `realtime.rs`
/// graceful-fallback contract so development-host runs still work.
pub fn run_pipeline(cfg: PipelineConfig) -> Result<PipelineHandle, Error> {
    cfg.validate()?;
    let scheduler = TaskScheduler::from_config(&cfg)?;

    // Real-time setup for the thread that calls `dispatch_frame`.
    // Individual capture / encode / output threads typically call
    // `apply_rt_config` themselves with their own stage-specific
    // priority + affinity; this call elevates just the driver thread.
    #[cfg(feature = "realtime")]
    if cfg.realtime.sched_fifo {
        let prio = cfg.realtime.prio.get("dispatch").copied().unwrap_or(0);
        let affinity: Vec<u32> = cfg
            .realtime
            .affinity
            .get("dispatch")
            .cloned()
            .unwrap_or_default();
        let governor = cfg.realtime.cpu_governor.as_deref();
        let applied = yscv_video::realtime::apply_rt_config_with_governor(
            prio,
            &affinity,
            true,
            governor,
        );
        if !applied.sched_fifo || !applied.mlockall {
            eprintln!(
                "[yscv-pipeline] realtime partial: sched_fifo={} affinity={} mlockall={} \
                 governor_cores={} (missing CAP_SYS_NICE / RLIMIT_MEMLOCK / \
                 CAP_SYS_ADMIN? dev-host runs are fine)",
                applied.sched_fifo,
                applied.affinity,
                applied.mlockall,
                applied.cpu_governor_cores,
            );
        }
    }
    #[cfg(not(feature = "realtime"))]
    if cfg.realtime.sched_fifo {
        eprintln!(
            "[yscv-pipeline] config requests `sched_fifo = true` but this build lacks \
             `--features realtime`; SCHED_FIFO / affinity / mlockall NOT applied"
        );
    }

    let mut dispatchers: HashMap<String, Box<dyn AcceleratorDispatcher>> =
        HashMap::with_capacity(cfg.tasks.len());
    let mut tasks: HashMap<String, InferenceTask> = HashMap::with_capacity(cfg.tasks.len());
    for task in &cfg.tasks {
        let d = dispatcher_for(task)?;
        dispatchers.insert(task.name.clone(), d);
        tasks.insert(task.name.clone(), task.clone());
    }

    Ok(PipelineHandle {
        order: scheduler.order,
        dispatchers,
        tasks,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::accelerator::{Accelerator, NpuCoreSpec};
    use crate::config::{
        CameraSpec, EncoderSpec, InferenceTask, OsdSpec, OutputSpec, RtSpec, TensorBinding,
    };
    use std::path::PathBuf;

    fn cfg_with(tasks: Vec<InferenceTask>) -> PipelineConfig {
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
            tasks,
            osd: OsdSpec::default(),
            realtime: RtSpec::default(),
        }
    }

    #[test]
    fn topo_sort_linear() {
        let tasks = vec![
            InferenceTask {
                name: "detector".into(),
                model_path: PathBuf::from("/tmp/x"),
                accelerator: Accelerator::Cpu,
                inputs: vec![TensorBinding {
                    name: "images".into(),
                    source: "camera".into(),
                }],
                outputs: vec![],
            },
            InferenceTask {
                name: "tracker".into(),
                model_path: PathBuf::from("/tmp/x"),
                accelerator: Accelerator::Cpu,
                inputs: vec![TensorBinding {
                    name: "dets".into(),
                    source: "detector.output0".into(),
                }],
                outputs: vec![],
            },
        ];
        let cfg = cfg_with(tasks);
        let sched = TaskScheduler::from_config(&cfg).unwrap();
        // detector must come before tracker.
        let det_pos = sched.order.iter().position(|n| n == "detector").unwrap();
        let trk_pos = sched.order.iter().position(|n| n == "tracker").unwrap();
        assert!(det_pos < trk_pos);
    }

    #[test]
    fn topo_sort_diamond() {
        let tasks = vec![
            InferenceTask {
                name: "src".into(),
                model_path: PathBuf::from("/tmp/x"),
                accelerator: Accelerator::Cpu,
                inputs: vec![TensorBinding {
                    name: "in".into(),
                    source: "camera".into(),
                }],
                outputs: vec![],
            },
            InferenceTask {
                name: "left".into(),
                model_path: PathBuf::from("/tmp/x"),
                accelerator: Accelerator::Cpu,
                inputs: vec![TensorBinding {
                    name: "x".into(),
                    source: "src.out".into(),
                }],
                outputs: vec![],
            },
            InferenceTask {
                name: "right".into(),
                model_path: PathBuf::from("/tmp/x"),
                accelerator: Accelerator::Rknn {
                    core: NpuCoreSpec::Core0,
                },
                inputs: vec![TensorBinding {
                    name: "x".into(),
                    source: "src.out".into(),
                }],
                outputs: vec![],
            },
            InferenceTask {
                name: "merge".into(),
                model_path: PathBuf::from("/tmp/x"),
                accelerator: Accelerator::Cpu,
                inputs: vec![
                    TensorBinding {
                        name: "l".into(),
                        source: "left.out".into(),
                    },
                    TensorBinding {
                        name: "r".into(),
                        source: "right.out".into(),
                    },
                ],
                outputs: vec![],
            },
        ];
        let cfg = cfg_with(tasks);
        let sched = TaskScheduler::from_config(&cfg).unwrap();
        let pos = |n: &str| sched.order.iter().position(|x| x == n).unwrap();
        assert!(pos("src") < pos("left"));
        assert!(pos("src") < pos("right"));
        assert!(pos("left") < pos("merge"));
        assert!(pos("right") < pos("merge"));
    }

    #[test]
    fn ready_tasks_respects_completion() {
        let tasks = vec![
            InferenceTask {
                name: "a".into(),
                model_path: PathBuf::from("/tmp/x"),
                accelerator: Accelerator::Cpu,
                inputs: vec![TensorBinding {
                    name: "in".into(),
                    source: "camera".into(),
                }],
                outputs: vec![],
            },
            InferenceTask {
                name: "b".into(),
                model_path: PathBuf::from("/tmp/x"),
                accelerator: Accelerator::Cpu,
                inputs: vec![TensorBinding {
                    name: "x".into(),
                    source: "a.out".into(),
                }],
                outputs: vec![],
            },
        ];
        let sched = TaskScheduler::from_config(&cfg_with(tasks)).unwrap();
        let mut done: HashSet<String> = HashSet::new();
        let ready_initial = sched.ready_tasks(&done);
        assert!(ready_initial.contains(&"a".to_string()));
        assert!(!ready_initial.contains(&"b".to_string()));
        done.insert("a".into());
        let ready_after = sched.ready_tasks(&done);
        assert!(ready_after.contains(&"b".to_string()));
    }
}
