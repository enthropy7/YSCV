//! FPV pipeline latency benchmark.
//!
//! Runs a N-frame synthetic pipeline through the 5-stage lock-free
//! executor and reports p50/p95/p99 per-stage + end-to-end latency as
//! JSON on stdout. Used for regression tracking in CI and for validating
//! that the HW acceleration stack meets the target e2e budget (10 ms
//! Rock 4D, 25 ms LuckFox).
//!
//! Usage:
//!     fpv-latency [--frames N] [--capture-us X] [--infer-us Y] [--encode-us Z]
//!
//! The `--*-us` knobs inject synthetic per-stage delays so the harness
//! can model a real board's timing distribution without requiring that
//! HW to be physically attached.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};
use yscv_video::frame_pipeline_5stage::{Pipeline5Builder, PipelinedFrame};

#[derive(Debug, Clone, Copy)]
struct Args {
    frames: usize,
    capture_us: u64,
    infer_us: u64,
    encode_us: u64,
    json: bool,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            frames: 500,
            capture_us: 1_000,
            infer_us: 6_000,
            encode_us: 2_500,
            json: true,
        }
    }
}

fn parse_args() -> Args {
    let mut out = Args::default();
    let argv: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--frames" => {
                i += 1;
                out.frames = argv[i].parse().unwrap_or(500);
            }
            "--capture-us" => {
                i += 1;
                out.capture_us = argv[i].parse().unwrap_or(1_000);
            }
            "--infer-us" => {
                i += 1;
                out.infer_us = argv[i].parse().unwrap_or(6_000);
            }
            "--encode-us" => {
                i += 1;
                out.encode_us = argv[i].parse().unwrap_or(2_500);
            }
            "--text" => out.json = false,
            "--help" | "-h" => {
                eprintln!(
                    "usage: fpv-latency [--frames N] [--capture-us X] [--infer-us Y] [--encode-us Z] [--text]"
                );
                std::process::exit(0);
            }
            other => {
                eprintln!("ignoring unknown arg: {other}");
            }
        }
        i += 1;
    }
    out
}

/// Busy-wait for approximately `us` microseconds. Uses spin so that
/// SCHED_FIFO threads don't get deprioritised via sleep → more accurate
/// emulation of a compute-bound stage.
fn spin_us(us: u64) {
    let t = Instant::now();
    while t.elapsed() < Duration::from_micros(us) {
        std::hint::spin_loop();
    }
}

fn percentile(sorted: &[u64], p: f64) -> u64 {
    if sorted.is_empty() {
        return 0;
    }
    let idx = ((sorted.len() as f64 - 1.0) * p).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn main() {
    let args = parse_args();

    let builder = Pipeline5Builder::new();
    let stats = builder.stats.clone();
    let stop = builder.stop.clone();
    let captured = Arc::new(AtomicUsize::new(0));

    // Per-frame timing samples, collected during the output stage.
    let e2e_samples = Arc::new(std::sync::Mutex::new(Vec::<u64>::with_capacity(
        args.frames,
    )));

    let captured_c = captured.clone();
    let stop_c = stop.clone();
    let frames = args.frames;
    let capture_us = args.capture_us;
    let infer_us = args.infer_us;
    let encode_us = args.encode_us;
    let e2e_ref = e2e_samples.clone();

    let t_start = Instant::now();
    builder.run(
        // capture
        move || {
            spin_us(capture_us);
            let i = captured_c.fetch_add(1, Ordering::Relaxed);
            if i >= frames {
                stop_c.store(true, Ordering::Release);
                return None;
            }
            Some(PipelinedFrame::new(i as u64, -1, 0))
        },
        // infer
        move |_f| {
            spin_us(infer_us);
        },
        // encode
        move |_f| {
            spin_us(encode_us);
        },
        // output — record e2e latency
        move |f: &PipelinedFrame| {
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_micros() as u64)
                .unwrap_or(0);
            let e2e = now.saturating_sub(f.captured_at_us);
            let mut g = e2e_ref.lock().unwrap_or_else(|e| e.into_inner());
            g.push(e2e);
        },
    );
    let wall_elapsed = t_start.elapsed();

    let snap = stats.snapshot();
    let mut samples = {
        let g = e2e_samples.lock().unwrap_or_else(|e| e.into_inner());
        g.clone()
    };
    samples.sort_unstable();

    let p50 = percentile(&samples, 0.50);
    let p95 = percentile(&samples, 0.95);
    let p99 = percentile(&samples, 0.99);
    let min = *samples.first().unwrap_or(&0);
    let max = *samples.last().unwrap_or(&0);
    let avg = if samples.is_empty() {
        0
    } else {
        samples.iter().sum::<u64>() / samples.len() as u64
    };
    let wall_ms = wall_elapsed.as_secs_f64() * 1000.0;
    let fps = (snap.displayed as f64) / wall_elapsed.as_secs_f64();

    if args.json {
        println!(
            r#"{{
  "frames": {},
  "wall_ms": {:.1},
  "fps": {:.1},
  "captured": {},
  "processed": {},
  "encoded": {},
  "displayed": {},
  "dropped": {},
  "watchdog_alarm": {},
  "e2e_latency_us": {{
    "p50": {},
    "p95": {},
    "p99": {},
    "min": {},
    "max": {},
    "avg": {}
  }},
  "budgets_us": {{
    "capture": {},
    "infer": {},
    "encode": {}
  }}
}}"#,
            args.frames,
            wall_ms,
            fps,
            snap.captured,
            snap.processed,
            snap.encoded,
            snap.displayed,
            snap.dropped,
            snap.watchdog_alarm,
            p50,
            p95,
            p99,
            min,
            max,
            avg,
            args.capture_us,
            args.infer_us,
            args.encode_us,
        );
    } else {
        println!("=== FPV Pipeline Latency Benchmark ===");
        println!("Frames:            {}", args.frames);
        println!("Wall time:         {:.1} ms", wall_ms);
        println!("Throughput:        {:.1} FPS", fps);
        println!("Displayed / dropped: {} / {}", snap.displayed, snap.dropped);
        println!("Watchdog alarm:    {}", snap.watchdog_alarm);
        println!();
        println!("End-to-end latency (capture → output):");
        println!("  p50:   {:>6} µs ({:.2} ms)", p50, p50 as f64 / 1000.0);
        println!("  p95:   {:>6} µs ({:.2} ms)", p95, p95 as f64 / 1000.0);
        println!("  p99:   {:>6} µs ({:.2} ms)", p99, p99 as f64 / 1000.0);
        println!("  min:   {:>6} µs", min);
        println!("  max:   {:>6} µs", max);
        println!("  avg:   {:>6} µs", avg);
        println!();
        println!(
            "Synthetic budgets: capture={}µs, infer={}µs, encode={}µs",
            args.capture_us, args.infer_us, args.encode_us
        );
    }
}
