# yscv-video

H.264 and HEVC video decoders, MP4/MKV container parsers, hardware decode, and camera input. Pure Rust, no FFmpeg.

```rust
use yscv_video::Mp4VideoReader;

let mut reader = Mp4VideoReader::open("video.mp4")?;
while let Some(frame) = reader.next_frame()? {
    // frame.rgb8_data: Vec<u8>, frame.width, frame.height
}
```

## Capabilities

| Feature | Details |
|---------|---------|
| **H.264 decode** | CAVLC, CABAC, I/P/B slices, 4x4/8x8 DCT, weighted prediction |
| **HEVC decode** | CTU quad-tree, branchless CABAC, 8-tap MC, SAO, deblock |
| **MP4 parse** | Streaming moov-only reader, O(1) memory per frame |
| **MKV parse** | EBML demuxer with frame index |
| **HW decode** | VideoToolbox (macOS), VA-API (Linux), NVDEC, MediaFoundation |
| **Camera** | Live capture via Nokhwa (optional) |
| **Audio** | AAC/ALAC/Opus/Vorbis/MP3/FLAC codec detection |
| **Color** | YUV420→RGB8 with NEON/SSE2 SIMD |

## Performance

- H.264 decode: **4.5x faster** than ffmpeg (1080p real camera, Apple M-series)
- HEVC decode: **1.4x faster** than ffmpeg (P/B frames, full color)
- 21 named SIMD functions (8 NEON + 11 SSE2 + 2 AVX2) with `#[target_feature]` runtime dispatch on x86 and compile-time gating on aarch64

## Features

```toml
[features]
videotoolbox = []      # macOS HW decode
vaapi = []             # Linux HW decode
nvdec = []             # NVIDIA HW decode
media-foundation = []  # Windows HW decode
native-camera = []     # Live camera input
```

## Tests

220 tests covering decode correctness, container parsing, edge cases.
