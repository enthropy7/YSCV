//! Decodes a raw Annex-B H.264 file to planar YUV420p on stdout, in display
//! (POC) order, for bit-exact conformance comparison against a reference
//! decoder. Usage: `ydump <file.264> > out.yuv`.
//!
//! In `dump_yuv` mode the decoder packs the cropped Y/U/V planes into the
//! frame's `rgb8_data`, so reordered frames carry their own pixels.
use std::io::Write;
use yscv_video::VideoDecoder;

fn main() {
    let path = std::env::args().nth(1).expect("usage: ydump <file.264>");
    let data = std::fs::read(&path).expect("read input");
    let nals = yscv_video::parse_annex_b(&data);

    let mut dec = yscv_video::H264Decoder::new();
    dec.dump_yuv = true;
    let stdout = std::io::stdout();
    let mut out = stdout.lock();

    for nal in &nals {
        if let Ok(Some(frame)) = dec.process_nal(nal) {
            out.write_all(&frame.rgb8_data).unwrap();
        }
    }
    for frame in dec.flush().unwrap_or_default() {
        out.write_all(&frame.rgb8_data).unwrap();
    }
}
