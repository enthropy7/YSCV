//! Benchmark: decode H.264/HEVC MP4 video with yscv Mp4VideoReader.
//!
//! Usage: cargo run --release --example bench_video_decode -- <video.mp4> [--luma-only]
//!
//! The `--luma-only` flag skips YUV-to-RGB conversion for fair comparison with
//! ffmpeg's `-f null -` which also skips color conversion.

use std::path::Path;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path_str = args
        .get(1)
        .map(|s| s.as_str())
        .unwrap_or("examples/src/CENSUSWITHOUTLOGO.mp4");
    let luma_only = args.iter().any(|a| a == "--luma-only");
    let path = Path::new(path_str);

    if !path.exists() {
        eprintln!("File not found: {}", path.display());
        std::process::exit(1);
    }

    println!("=== yscv Mp4VideoReader benchmark ===");
    println!("File: {}", path.display());
    if luma_only {
        println!("Mode: LUMA-ONLY (skip YUV→RGB, fair vs ffmpeg -f null)");
    }

    let t0 = Instant::now();
    let mut reader = match yscv_video::Mp4VideoReader::open(path) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Failed to open: {e}");
            std::process::exit(1);
        }
    };
    let open_time = t0.elapsed();
    println!("Open + parse: {:.0}ms", open_time.as_secs_f64() * 1000.0);
    println!("NAL count: {}", reader.nal_count());
    println!("Codec: {:?}", reader.codec());

    let t1 = Instant::now();
    let mut decoded = 0u32;
    let mut errors = 0u32;
    let mut first_frame_time = None;

    loop {
        let result = if luma_only {
            reader.next_frame_luma_only()
        } else {
            reader.next_frame()
        };
        match result {
            Ok(Some(frame)) => {
                if decoded == 0 {
                    first_frame_time = Some(t1.elapsed());
                    let min = frame.rgb8_data.iter().copied().min().unwrap_or(0);
                    let max = frame.rgb8_data.iter().copied().max().unwrap_or(0);
                    println!(
                        "Frame 0: {}x{} keyframe={} pixel_range=[{}, {}]",
                        frame.width, frame.height, frame.keyframe, min, max
                    );
                    if min == max {
                        println!("WARNING: frame is uniform color — possible decode issue");
                    }
                }
                decoded += 1;
            }
            Ok(None) => break,
            Err(e) => {
                if errors == 0 {
                    eprintln!("First error at frame {decoded}: {e}");
                }
                errors += 1;
                if errors > 200 {
                    break;
                }
            }
        }
    }

    let total_time = t1.elapsed();
    println!("\n--- Results ---");
    println!("Decoded: {decoded} frames");
    println!("Errors:  {errors}");
    if let Some(ft) = first_frame_time {
        println!("First frame: {:.1}ms", ft.as_secs_f64() * 1000.0);
    }
    println!("Total decode: {:.0}ms", total_time.as_secs_f64() * 1000.0);
    if decoded > 0 {
        println!(
            "Per frame: {:.2}ms ({:.1} FPS)",
            total_time.as_secs_f64() * 1000.0 / decoded as f64,
            decoded as f64 / total_time.as_secs_f64()
        );
    }
}
