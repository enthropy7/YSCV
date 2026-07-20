//! Decode-only throughput benchmark: parses a raw Annex-B H.264 file once, then
//! decodes the whole stream repeatedly (luma-only, no YUV→RGB — the fair
//! comparison against ffmpeg `-f null`). Reports the best frames-per-second.
//! Usage: `ybench <file.264> [passes]`.
use std::time::Instant;
use yscv_video::VideoDecoder;

fn main() {
    let path = std::env::args().nth(1).expect("usage: ybench <file.264> [passes]");
    let passes: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(7);
    let data = std::fs::read(&path).expect("read input");
    let nals = yscv_video::parse_annex_b(&data);

    // Warm-up + frame count.
    let mut frames = 0usize;
    {
        let mut dec = yscv_video::H264Decoder::new();
        dec.skip_rgb = true;
        for nal in &nals {
            if matches!(dec.process_nal(nal), Ok(Some(_))) {
                frames += 1;
            }
        }
        frames += dec.flush().map(|v| v.len()).unwrap_or(0);
    }

    let mut best_fps = 0.0f64;
    for _ in 0..passes {
        let mut dec = yscv_video::H264Decoder::new();
        dec.skip_rgb = true;
        let t = Instant::now();
        for nal in &nals {
            let _ = dec.process_nal(nal);
        }
        let _ = dec.flush();
        let secs = t.elapsed().as_secs_f64();
        let fps = frames as f64 / secs;
        if fps > best_fps {
            best_fps = fps;
        }
    }
    println!("{frames} frames, best {best_fps:.1} fps decode-only");
}
