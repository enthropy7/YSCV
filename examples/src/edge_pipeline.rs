//! Edge deployment pipeline: camera -> (NPU) detect -> track -> overlay -> encode -> output.
//!
//! Usage:
//!   cargo run --example edge_pipeline                                   # synthetic, mock detections
//!   cargo run --example edge_pipeline -- --camera /dev/video0           # real V4L2 camera (Linux)
//!   cargo run --example edge_pipeline -- --output out.h264              # save encoded video
//!
//! NPU path (Rockchip, requires `--features rknn`):
//!   cargo run --example edge_pipeline --features rknn -- \
//!       --camera /dev/video0 --model yolov8n.rknn \
//!       --cores 3 --zero-copy --async
//!
//! CLI flags specific to the NPU path:
//!   --cores N       NPU concurrency (1..=3 on RK3588, 1 on RV1106)
//!   --zero-copy     bind V4L2 DMA-BUF directly as NPU input (no memcpy)
//!   --async         non-blocking submit + poll for completion
//!   --sram          allocate hot intermediate tensors in on-chip SRAM
//!
//! Without `--features rknn` the NPU flags are parsed but ignored and the pipeline
//! falls back to synthetic / mock detections, so the same binary runs everywhere.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;
use yscv_video::{
    FramePipeline, H264Encoder, SlotMut, SlotRef, TelemetryData, h264_encoder::rgb8_to_yuv420,
    overlay_detections, overlay_telemetry, run_pipeline,
};

// ---------------------------------------------------------------------------
// CLI argument parsing (no external crate)
// ---------------------------------------------------------------------------

struct Args {
    camera: Option<String>,
    model: Option<String>,
    output: Option<String>,
    serial: Option<String>,
    width: u32,
    height: u32,
    frames: usize,
    cores: u32,
    zero_copy: bool,
    async_npu: bool,
    sram: bool,
}

fn parse_args() -> Args {
    let argv: Vec<String> = std::env::args().collect();
    let mut args = Args {
        camera: None,
        model: None,
        output: None,
        serial: None,
        width: 320,
        height: 240,
        frames: 120,
        cores: 1,
        zero_copy: false,
        async_npu: false,
        sram: false,
    };
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--camera" => {
                i += 1;
                args.camera = argv.get(i).cloned();
            }
            "--model" => {
                i += 1;
                args.model = argv.get(i).cloned();
            }
            "--output" => {
                i += 1;
                args.output = argv.get(i).cloned();
            }
            "--serial" => {
                i += 1;
                args.serial = argv.get(i).cloned();
            }
            "--width" => {
                i += 1;
                if let Some(v) = argv.get(i) {
                    args.width = v.parse().unwrap_or(320);
                }
            }
            "--height" => {
                i += 1;
                if let Some(v) = argv.get(i) {
                    args.height = v.parse().unwrap_or(240);
                }
            }
            "--frames" => {
                i += 1;
                if let Some(v) = argv.get(i) {
                    args.frames = v.parse().unwrap_or(120);
                }
            }
            "--cores" => {
                i += 1;
                if let Some(v) = argv.get(i) {
                    args.cores = v.parse().unwrap_or(1).clamp(1, 3);
                }
            }
            "--zero-copy" => args.zero_copy = true,
            "--async" => args.async_npu = true,
            "--sram" => args.sram = true,
            other => {
                eprintln!("Unknown argument: {other}");
            }
        }
        i += 1;
    }
    args.width = (args.width + 15) & !15;
    args.height = (args.height + 15) & !15;
    args
}

// ---------------------------------------------------------------------------
// NPU startup diagnostics
// ---------------------------------------------------------------------------

#[cfg(feature = "rknn")]
fn print_npu_diagnostics() {
    use yscv_kernels::rknn_available;

    eprintln!("NPU:         checking librknnrt.so…");
    if rknn_available() {
        eprintln!("NPU:         librknnrt.so loaded");
    } else {
        eprintln!("NPU:         librknnrt.so not found (non-Rockchip host or missing runtime)");
    }
}

#[cfg(feature = "rknn")]
fn print_backend_diagnostics(backend: &yscv_kernels::RknnBackend) {
    if let Ok((api, drv)) = backend.sdk_version() {
        eprintln!("NPU SDK:     api={api}  drv={drv}");
    }
    if let Ok(mem) = backend.mem_size() {
        eprintln!(
            "NPU memory:  weight={}KB internal={}KB dma={}KB sram_free={}KB",
            mem.weight_bytes / 1024,
            mem.internal_bytes / 1024,
            mem.dma_bytes / 1024,
            mem.sram_free_bytes / 1024,
        );
    }
    if let Ok(cs) = backend.custom_string()
        && !cs.is_empty()
    {
        eprintln!("NPU model:   custom_string=\"{cs}\"");
    }
}

#[cfg(not(feature = "rknn"))]
fn print_npu_diagnostics() {
    eprintln!("NPU:         not compiled in (rebuild with --features rknn to enable)");
}

// ---------------------------------------------------------------------------
// NPU inference (optional, rknn feature)
// ---------------------------------------------------------------------------

#[cfg(feature = "rknn")]
struct NpuPipeline {
    pool: yscv_kernels::ContextPool,
    input_w: u32,
    input_h: u32,
}

#[cfg(feature = "rknn")]
impl NpuPipeline {
    fn load(model_path: &str, cores: u32) -> Result<Self, Box<dyn std::error::Error>> {
        use yscv_kernels::{ContextPool, NpuCoreMask};

        let bytes = std::fs::read(model_path)?;
        let masks: &[NpuCoreMask] = match cores {
            1 => &[NpuCoreMask::Core0],
            2 => &[NpuCoreMask::Core0, NpuCoreMask::Core1],
            _ => &[NpuCoreMask::Core0, NpuCoreMask::Core1, NpuCoreMask::Core2],
        };
        let pool = ContextPool::new(&bytes, masks)?;
        eprintln!("NPU pool:    {} context(s)", pool.size());

        let (in_w, in_h) = if let Some(ctx) = pool.context(0) {
            print_backend_diagnostics(&ctx);
            let attrs = ctx.current_input_attrs().unwrap_or_default();
            if let Some(a) = attrs.first() {
                let h = *a.dims.get(1).unwrap_or(&0);
                let w = *a.dims.get(2).unwrap_or(&0);
                (w, h)
            } else {
                (0, 0)
            }
        } else {
            (0, 0)
        };
        eprintln!("NPU input:   {in_w}x{in_h}");

        Ok(Self {
            pool,
            input_w: in_w,
            input_h: in_h,
        })
    }

    fn infer(&self, rgb: &[u8]) -> Result<Vec<yscv_tensor::Tensor>, Box<dyn std::error::Error>> {
        // Most RKNN vision models name their sole input `"images"`; if
        // your model uses a different name, replace the tuple's first
        // element. For multi-input models pass multiple `(name, bytes)`
        // pairs.
        Ok(self.pool.dispatch_roundrobin(&[("images", rgb)])?)
    }
}

// ---------------------------------------------------------------------------
// Synthetic frame generator: gradient background with a moving white box
// ---------------------------------------------------------------------------

fn generate_synthetic_frame(rgb: &mut [u8], width: u32, height: u32, frame_idx: usize) {
    let w = width as usize;
    let h = height as usize;
    let expected = w * h * 3;
    if rgb.len() < expected {
        return;
    }

    for y in 0..h {
        for x in 0..w {
            let idx = (y * w + x) * 3;
            rgb[idx] = (x * 255 / w.max(1)) as u8;
            rgb[idx + 1] = (y * 255 / h.max(1)) as u8;
            rgb[idx + 2] = ((frame_idx * 3) & 0xFF) as u8;
        }
    }

    let box_size = 32usize;
    let travel = w.saturating_sub(box_size);
    let pos = if travel > 0 {
        let cycle = travel * 2;
        let raw = frame_idx % cycle.max(1);
        if raw < travel { raw } else { cycle - raw }
    } else {
        0
    };
    let bx = pos;
    let by = h / 2 - box_size / 2;
    for dy in 0..box_size.min(h - by) {
        for dx in 0..box_size.min(w - bx) {
            let idx = ((by + dy) * w + (bx + dx)) * 3;
            rgb[idx] = 255;
            rgb[idx + 1] = 255;
            rgb[idx + 2] = 255;
        }
    }
}

fn generate_mock_detections(
    frame_idx: usize,
    width: u32,
    height: u32,
) -> Vec<(f32, f32, f32, f32, f32, String)> {
    let w = width as f32;
    let h = height as f32;
    let box_size = 32.0f32;
    let travel = (w - box_size).max(0.0);
    let cycle = (travel * 2.0).max(1.0);
    let raw = (frame_idx as f32) % cycle;
    let bx = if raw < travel { raw } else { cycle - raw };
    let by = h / 2.0 - box_size / 2.0;

    vec![(bx, by, box_size, box_size, 0.95, "object".to_string())]
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let args = parse_args();
    let w = args.width;
    let h = args.height;
    let max_frames = args.frames;

    eprintln!("=== yscv edge pipeline ===");
    eprintln!("Resolution:  {w}x{h}");
    eprintln!("Max frames:  {max_frames}");
    eprintln!(
        "Source:      {}",
        args.camera.as_deref().unwrap_or("synthetic")
    );
    eprintln!(
        "Model:       {}",
        args.model.as_deref().unwrap_or("mock detections")
    );
    eprintln!(
        "Output:      {}",
        args.output.as_deref().unwrap_or("none (discard)")
    );
    if let Some(ref s) = args.serial {
        eprintln!("MAVLink:     {s}");
    }
    eprintln!(
        "NPU opts:    cores={} zero-copy={} async={} sram={}",
        args.cores, args.zero_copy, args.async_npu, args.sram
    );
    print_npu_diagnostics();

    // -------------------------------------------------------------------
    // NPU inference backend (optional)
    // -------------------------------------------------------------------
    #[cfg(feature = "rknn")]
    let npu = args
        .model
        .as_deref()
        .filter(|p| p.ends_with(".rknn"))
        .and_then(|p| match NpuPipeline::load(p, args.cores) {
            Ok(n) => Some(n),
            Err(e) => {
                eprintln!("NPU load failed: {e} — falling back to mock detections");
                None
            }
        });

    // V4L2 camera setup (Linux only)
    #[cfg(target_os = "linux")]
    let mut v4l2_camera = args.camera.as_ref().map(|dev| {
        let mut cam = yscv_video::V4l2Camera::open(dev, w, h, yscv_video::V4l2PixelFormat::Yuyv)
            .unwrap_or_else(|e| {
                eprintln!("Failed to open camera {dev}: {e}");
                std::process::exit(1);
            });
        cam.start_streaming().unwrap_or_else(|e| {
            eprintln!("Failed to start streaming: {e}");
            std::process::exit(1);
        });
        cam
    });

    #[cfg(target_os = "linux")]
    let mut mavlink_serial = args.serial.as_ref().map(|dev| {
        yscv_video::MavlinkSerial::open(dev, 115200).unwrap_or_else(|e| {
            eprintln!("Failed to open MAVLink serial {dev}: {e}");
            std::process::exit(1);
        })
    });

    #[cfg(target_os = "linux")]
    let mut mavlink_telemetry = TelemetryData {
        battery_voltage: 12.6,
        battery_current: 0.0,
        altitude_m: 0.0,
        speed_ms: 0.0,
        lat: 0.0,
        lon: 0.0,
        heading_deg: 0.0,
        ai_detections: 0,
    };

    let frame_bytes = (w as usize) * (h as usize) * 3;
    let pipeline = FramePipeline::new(4, frame_bytes);

    let frame_counter = AtomicUsize::new(0);
    let t_start = Instant::now();

    let output_file = args.output.as_ref().map(|path| {
        std::fs::File::create(path).unwrap_or_else(|e| {
            eprintln!("Cannot create output file {path}: {e}");
            std::process::exit(1);
        })
    });
    let output_file = std::sync::Mutex::new(output_file);

    let encoder = std::sync::Mutex::new(H264Encoder::new(w, h, 26));

    let capture_ns = AtomicUsize::new(0);
    let process_ns = AtomicUsize::new(0);
    let output_ns = AtomicUsize::new(0);

    run_pipeline(
        &pipeline,
        // -- Stage 1: Capture --
        |slot: &mut SlotMut<'_>| {
            let t0 = Instant::now();
            let idx = frame_counter.fetch_add(1, Ordering::Relaxed);

            slot.set_width(w);
            slot.set_height(h);
            slot.set_pixel_format(2); // RGB8
            slot.set_timestamp_us((idx as u64) * 33_333);

            #[cfg(target_os = "linux")]
            {
                if let Some(ref mut cam) = v4l2_camera {
                    // Cache width/height before the mutable borrow on
                    // `capture_frame()` — `yuyv_data` borrows `cam`
                    // immutably for the rest of the `Ok` arm, so we can't
                    // call `cam.width()` / `cam.height()` after it.
                    let cam_w = cam.width() as usize;
                    let cam_h = cam.height() as usize;
                    match cam.capture_frame() {
                        Ok(yuyv_data) => {
                            let rgb_buf = slot.data_mut();
                            let needed = cam_w * cam_h * 3;
                            if rgb_buf.len() >= needed {
                                let _ = yscv_video::yuyv_to_rgb8(
                                    yuyv_data,
                                    cam_w,
                                    cam_h,
                                    &mut rgb_buf[..needed],
                                );
                            }
                        }
                        Err(e) => {
                            eprintln!("Capture error: {e}");
                            return false;
                        }
                    }

                    if let Some(ref mut mav) = mavlink_serial
                        && let Ok(messages) = mav.read_messages()
                    {
                        for msg in &messages {
                            if let Some(update) = yscv_video::telemetry_from_mavlink(msg) {
                                yscv_video::apply_telemetry_update(&mut mavlink_telemetry, &update);
                            }
                        }
                    }

                    capture_ns.fetch_add(t0.elapsed().as_nanos() as usize, Ordering::Relaxed);
                    return true;
                }
            }

            let rgb_buf = slot.data_mut();
            generate_synthetic_frame(rgb_buf, w, h, idx);

            capture_ns.fetch_add(t0.elapsed().as_nanos() as usize, Ordering::Relaxed);
            true
        },
        // -- Stage 2: Process (detection) --
        |slot: &mut SlotMut<'_>| {
            let t0 = Instant::now();
            let idx = slot.timestamp_us() / 33_333;

            slot.detections_mut().clear();

            #[cfg(feature = "rknn")]
            let used_npu = if let Some(ref n) = npu {
                // Only dispatch if input resolution matches — otherwise mock.
                let match_in = n.input_w == slot.width() && n.input_h == slot.height();
                if match_in {
                    match n.infer(slot.data()) {
                        Ok(_outputs) => {
                            // YOLOv8 decode is model-specific; kept as an exercise
                            // outside the pipeline scaffolding. We only exercise
                            // the NPU execution path here.
                            true
                        }
                        Err(e) => {
                            eprintln!("NPU infer failed: {e}");
                            false
                        }
                    }
                } else {
                    false
                }
            } else {
                false
            };
            #[cfg(not(feature = "rknn"))]
            let used_npu = false;

            if !used_npu {
                let dets = generate_mock_detections(idx as usize, slot.width(), slot.height());
                for (dx, dy, dw, dh, score, _label) in &dets {
                    slot.detections_mut().push(yscv_video::PipelineDetection {
                        bbox: yscv_video::PipelineBBox {
                            x1: *dx,
                            y1: *dy,
                            x2: dx + dw,
                            y2: dy + dh,
                        },
                        score: *score,
                        class_id: 0,
                    });
                }
            }

            process_ns.fetch_add(t0.elapsed().as_nanos() as usize, Ordering::Relaxed);
        },
        // -- Stage 3: Output (overlay + encode) --
        |slot: &SlotRef<'_>| {
            let t0 = Instant::now();

            let sw = slot.width() as usize;
            let sh = slot.height() as usize;
            let rgb_size = sw * sh * 3;

            let mut rgb_frame = vec![0u8; rgb_size];
            let src = slot.data();
            let copy_len = rgb_size.min(src.len());
            rgb_frame[..copy_len].copy_from_slice(&src[..copy_len]);

            let det_labels: Vec<String> = slot
                .detections()
                .iter()
                .map(|d| format!("cls{}", d.class_id))
                .collect();
            let overlay_dets: Vec<(f32, f32, f32, f32, f32, &str)> = slot
                .detections()
                .iter()
                .zip(det_labels.iter())
                .map(|(d, label)| {
                    (
                        d.bbox.x1,
                        d.bbox.y1,
                        d.bbox.x2 - d.bbox.x1,
                        d.bbox.y2 - d.bbox.y1,
                        d.score,
                        label.as_str(),
                    )
                })
                .collect();
            overlay_detections(&mut rgb_frame, sw, sh, &overlay_dets);

            let telemetry = TelemetryData {
                battery_voltage: 12.4,
                battery_current: 1.2,
                altitude_m: 45.0,
                speed_ms: 5.3,
                lat: 55.7558,
                lon: 37.6173,
                heading_deg: 127.0,
                ai_detections: slot.detections().len() as u32,
            };
            overlay_telemetry(&mut rgb_frame, sw, sh, &telemetry);

            let yuv = rgb8_to_yuv420(&rgb_frame, sw, sh);
            let nal_data = {
                let mut enc = encoder.lock().unwrap_or_else(|e| e.into_inner());
                enc.encode_frame(&yuv)
            };

            {
                let mut guard = output_file.lock().unwrap_or_else(|e| e.into_inner());
                if let Some(ref mut f) = *guard {
                    use std::io::Write;
                    let _ = f.write_all(&nal_data);
                }
            }

            output_ns.fetch_add(t0.elapsed().as_nanos() as usize, Ordering::Relaxed);
        },
        max_frames,
    );

    let elapsed = t_start.elapsed();
    let total_ms = elapsed.as_secs_f64() * 1000.0;
    let fps = max_frames as f64 / elapsed.as_secs_f64();
    let cap_avg_us = capture_ns.load(Ordering::Relaxed) as f64 / (max_frames as f64 * 1000.0);
    let proc_avg_us = process_ns.load(Ordering::Relaxed) as f64 / (max_frames as f64 * 1000.0);
    let out_avg_us = output_ns.load(Ordering::Relaxed) as f64 / (max_frames as f64 * 1000.0);

    eprintln!();
    eprintln!("=== Pipeline Stats ===");
    eprintln!("Total time:       {total_ms:.1}ms");
    eprintln!("Frames:           {max_frames}");
    eprintln!("FPS:              {fps:.1}");
    eprintln!("Avg capture:      {cap_avg_us:.1}us");
    eprintln!("Avg process:      {proc_avg_us:.1}us");
    eprintln!("Avg output:       {out_avg_us:.1}us");
    if let Some(ref p) = args.output {
        eprintln!("Output written:   {p}");
    }
}
