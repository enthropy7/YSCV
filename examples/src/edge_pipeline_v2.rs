//! End-to-end FPV-style pipeline driven by a TOML config.
//!
//! What this example demonstrates (all wired through the public API,
//! no hand-rolled glue):
//!
//! 1. `PipelineConfig::from_toml_path` parses + validates the config,
//!    including dry-run loading every `.rknn` / `.onnx` referenced by
//!    a task.
//! 2. `run_pipeline(cfg)` builds per-task `AcceleratorDispatcher`s —
//!    CPU / RKNN automatically match the `accelerator` enum in the
//!    TOML. Real-time (SCHED_FIFO + affinity + mlockall) is applied
//!    here when the example is built with `--features realtime`.
//! 3. A synthetic gradient-box frame generator replaces the real
//!    camera — lets the example run on any host (macOS, Linux CI, or
//!    a real Rockchip board). Real deployments swap this for
//!    `yscv_video::V4l2Camera::export_dmabuf` + `FramePayload::dma_buf`.
//! 4. Inside the hot loop: `handle.dispatch_frame(&[("camera", bytes)])`
//!    walks the task graph; output bytes flow through
//!    `world["<task>.<output>"]` so downstream tasks compose.
//! 5. Latency distribution (`PipelineStats5` + `LatencyHistogram`) is
//!    sampled each frame and dumped every second to stderr as
//!    p50/p95/p99.
//!
//! Run with:
//!     cargo run --release --example edge_pipeline_v2 -- path/to/pipeline.toml
//!
//! Add `--features "rknn realtime"` on a Rockchip board with
//! `librknnrt.so` + `CAP_SYS_NICE` for the production-like run.

use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use yscv_pipeline::{PipelineConfig, run_pipeline};
use yscv_video::frame_common::FramePayload;
use yscv_video::frame_pipeline_5stage::PipelineStats5;
use yscv_video::latency_histogram::LatencyHistogram;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    let Some(config_path) = args.get(1).map(PathBuf::from) else {
        eprintln!("usage: edge_pipeline_v2 <config.toml> [max_seconds]");
        return ExitCode::FAILURE;
    };
    let max_seconds: u64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(5);

    if let Err(e) = run(&config_path, max_seconds) {
        eprintln!("error: {e}");
        return ExitCode::FAILURE;
    }
    ExitCode::SUCCESS
}

fn run(config_path: &PathBuf, max_seconds: u64) -> Result<(), Box<dyn std::error::Error>> {
    let cfg = PipelineConfig::from_toml_path(config_path)?;
    eprintln!("=== Loaded config: {} ===", cfg.board);
    eprintln!(
        "Camera: {} @ {}×{} {}fps ({})",
        cfg.camera.device, cfg.camera.width, cfg.camera.height, cfg.camera.fps, cfg.camera.format
    );
    eprintln!("Tasks: {}", cfg.tasks.len());
    for task in &cfg.tasks {
        eprintln!(
            "  - {:<16} {}  → model: {}",
            task.name,
            task.accelerator.label(),
            task.model_path.display()
        );
    }

    // Build the runtime handle — validates every model, constructs
    // dispatchers, applies real-time config if the feature is on.
    let handle = run_pipeline(cfg.clone())?;
    eprintln!();
    eprintln!("=== Pipeline wired ===");
    for name in &handle.order {
        if let Some(label) = handle.dispatcher_label(name) {
            eprintln!("  {name} → {label}");
        }
    }

    // Latency tracking. We synthesise a minimal `PipelineStats5` even
    // though we're not running its 5-stage ring here — it gives us a
    // battle-tested container for the histograms + watchdogs.
    let stats = Arc::new(PipelineStats5::new(
        16_000, // capture budget — 16 ms for 60 FPS
        10_000, // inference budget — 10 ms
        16_000, // encode budget — 16 ms
    ));
    // Extra per-call dispatch histogram (sums input assembly + graph
    // traversal + serialise). `PipelineStats5` only has per-stage
    // watchdogs; this is the "end-to-end per dispatch_frame" metric.
    let dispatch_latency = Arc::new(LatencyHistogram::new());

    // Example auto-stops after `max_seconds`; real FPV apps hook
    // their own SIGINT via the `ctrlc` crate to flip this flag.
    let stop = Arc::new(AtomicBool::new(false));

    // Synthetic frame generator sized for the TOML's camera.width/height
    // in whatever format is declared (nv12/rgb/yuyv). For simplicity,
    // we always produce RGB8 and let the config's task inputs dictate
    // byte-count matching.
    let frame_bytes = (cfg.camera.width * cfg.camera.height * 3) as usize;
    let mut frame_buf = vec![0u8; frame_bytes];

    eprintln!();
    eprintln!("=== Hot loop (Ctrl+C to stop, or auto-stop at {max_seconds}s) ===");
    let t_start = Instant::now();
    let mut frame_idx: u64 = 0;
    let mut last_report = Instant::now();

    let mut consecutive_failures: u32 = 0;
    const MAX_CONSECUTIVE_FAILURES: u32 = 5;

    while !stop.load(Ordering::Acquire) {
        if t_start.elapsed().as_secs() >= max_seconds {
            break;
        }

        generate_synthetic_frame(
            &mut frame_buf,
            cfg.camera.width,
            cfg.camera.height,
            frame_idx,
        );
        let payload = FramePayload::owned(frame_buf.clone()); // copy-based ingress
        // `payload` demonstrates the FramePayload API; real FPV swaps to
        // `FramePayload::dma_buf(fd, len)` against V4L2 export_dmabuf.
        let frame_bytes_ref = payload.bytes().expect("owned payload has bytes");

        let t0 = Instant::now();
        let _world = match handle.dispatch_frame(&[("images", frame_bytes_ref)]) {
            Ok(w) => {
                consecutive_failures = 0;
                w
            }
            Err(e) => {
                consecutive_failures += 1;
                eprintln!(
                    "[edge_pipeline_v2] dispatch_frame failed (#{consecutive_failures}): {e}"
                );
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES {
                    eprintln!(
                        "[edge_pipeline_v2] {MAX_CONSECUTIVE_FAILURES} consecutive \
                         failures — giving up (check input shape / dtype against \
                         model's declared inputs)"
                    );
                    break;
                }
                if let Err(re) = handle.recover_all() {
                    eprintln!("[edge_pipeline_v2] recover_all failed: {re}");
                    break;
                }
                continue;
            }
        };
        let elapsed_us = t0.elapsed().as_micros() as u64;
        dispatch_latency.record(elapsed_us);
        stats.infer_latency.record(elapsed_us);
        stats.processed.fetch_add(1, Ordering::Relaxed);

        // Report every second.
        if last_report.elapsed() >= Duration::from_secs(1) {
            let snap = stats.snapshot();
            let d = dispatch_latency.snapshot();
            let fps = d.fps_p50();
            eprintln!(
                "[{:>3}s] frames={:>5}  p50={:>5} µs  p95={:>5} µs  p99={:>5} µs  max={:>5} µs  FPS≈{:>5.1}  alarm={}",
                t_start.elapsed().as_secs(),
                snap.processed,
                d.p50_us,
                d.p95_us,
                d.p99_us,
                d.max_us,
                fps,
                snap.watchdog_alarm,
            );
            last_report = Instant::now();
        }
        frame_idx += 1;
    }

    // Final stats.
    let snap = stats.snapshot();
    let d = dispatch_latency.snapshot();
    eprintln!();
    eprintln!(
        "=== Final ({} frames, {:.1}s wall) ===",
        snap.processed,
        t_start.elapsed().as_secs_f64()
    );
    eprintln!(
        "  dispatch_frame  p50={} µs  p95={} µs  p99={} µs  max={} µs  FPS(p50)≈{:.1}",
        d.p50_us,
        d.p95_us,
        d.p99_us,
        d.max_us,
        d.fps_p50()
    );
    Ok(())
}

/// Fill `buf` (assumed RGB8 width×height) with a diagonal gradient
/// plus a white box whose position rotates with `frame_idx`.
fn generate_synthetic_frame(buf: &mut [u8], width: u32, height: u32, frame_idx: u64) {
    let w = width as usize;
    let h = height as usize;
    let box_size = w.min(h) / 6;
    let box_x = (frame_idx as usize * 4) % w.saturating_sub(box_size).max(1);
    let box_y = (frame_idx as usize * 2) % h.saturating_sub(box_size).max(1);
    for y in 0..h {
        for x in 0..w {
            let idx = (y * w + x) * 3;
            if idx + 2 >= buf.len() {
                continue;
            }
            if x >= box_x && x < box_x + box_size && y >= box_y && y < box_y + box_size {
                buf[idx] = 255;
                buf[idx + 1] = 255;
                buf[idx + 2] = 255;
            } else {
                buf[idx] = (x * 255 / w.max(1)) as u8;
                buf[idx + 1] = (y * 255 / h.max(1)) as u8;
                buf[idx + 2] = (frame_idx & 0xff) as u8;
            }
        }
    }
}
