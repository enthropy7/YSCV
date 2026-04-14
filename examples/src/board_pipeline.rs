//! Config-driven pipeline demo.
//!
//! Loads a pipeline config TOML, validates accelerators + models against
//! the current host, and prints the topological task execution order.
//!
//! See `docs/pipeline-config.md` for the TOML schema.
//!
//! Run with:
//!     cargo run --example board_pipeline -- path/to/pipeline.toml

use std::path::PathBuf;
use std::process::ExitCode;
use yscv_pipeline::{PipelineConfig, TaskScheduler, probe_accelerators};

fn main() -> ExitCode {
    let argv: Vec<String> = std::env::args().collect();
    if argv.len() < 2 {
        eprintln!("usage: board_pipeline <config.toml>");
        return ExitCode::FAILURE;
    }
    let path = PathBuf::from(&argv[1]);

    let cfg = match PipelineConfig::from_toml_path(&path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("config parse failed: {e}");
            return ExitCode::FAILURE;
        }
    };

    eprintln!("=== Loaded config: {} ===", cfg.board);
    eprintln!(
        "Camera:   {} @ {}×{} {}fps ({})",
        cfg.camera.device, cfg.camera.width, cfg.camera.height, cfg.camera.fps, cfg.camera.format
    );
    eprintln!(
        "Encoder:  {} @ {} kbps",
        cfg.encoder.kind, cfg.encoder.bitrate_kbps
    );
    eprintln!("Tasks:    {}", cfg.tasks.len());
    for task in &cfg.tasks {
        eprintln!(
            "  - {:<12} → {} (model: {})",
            task.name,
            task.accelerator.label(),
            task.model_path.display()
        );
    }

    // Availability probe — explicit, user-visible.
    let avail = probe_accelerators();
    eprintln!();
    eprintln!("=== Host accelerator availability ===");
    eprintln!("CPU:       {}", avail.cpu);
    eprintln!("GPU:       {}", avail.gpu);
    eprintln!("RKNN NPU:  {}", avail.rknn);
    eprintln!("Metal MPS: {}", avail.metal_mps);

    if let Err(e) = cfg.validate_accelerators() {
        eprintln!();
        eprintln!("!!! accelerator validation failed !!!");
        eprintln!("{e}");
        eprintln!();
        eprintln!("This host cannot run the requested pipeline. Rebuild with the");
        eprintln!("matching feature flags or deploy to a host where the runtime is present.");
        return ExitCode::FAILURE;
    }

    match TaskScheduler::from_config(&cfg) {
        Ok(sched) => {
            eprintln!();
            eprintln!("=== Execution order ===");
            for (i, name) in sched.order.iter().enumerate() {
                eprintln!("  {}. {name}", i + 1);
            }
        }
        Err(e) => {
            eprintln!("scheduler failed: {e}");
            return ExitCode::FAILURE;
        }
    }

    eprintln!();
    eprintln!("Config OK. The runtime would start the pipeline here.");
    ExitCode::SUCCESS
}
