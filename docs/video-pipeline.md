# Video Pipeline

Pure Rust video decode pipeline — H.264 and HEVC codecs, MP4/MKV containers, hardware acceleration, streaming I/O.

## Architecture

```
MP4/MKV file
    │
    ├── Mp4VideoReader::open()     ← streaming reader, O(1) memory
    │   ├── moov box parse (1-5MB) ← sample table, SPS/PPS, audio info
    │   └── per-frame: seek + read one sample from disk
    │
    ├── Software Decode (default)
    │   ├── H.264: Baseline/Main/High, I/P/B, CABAC/CAVLC
    │   └── HEVC: Main/Main10, I/P/B, branchless CABAC
    │
    ├── Hardware Decode (opt-in features)
    │   ├── VideoToolbox (macOS)   ← feature = "videotoolbox"
    │   ├── VA-API (Linux)         ← feature = "vaapi"
    │   ├── NVDEC (NVIDIA)         ← feature = "nvdec"
    │   └── MediaFoundation (Win)  ← feature = "media-foundation"
    │
    └── Output: DecodedFrame { width, height, rgb8_data, timestamp, keyframe }
```

## Software Decode

### H.264/AVC

| Feature | Status |
|---------|--------|
| Baseline (CAVLC) | Full |
| Main (CABAC) | Full |
| High (8x8 DCT) | Full |
| I/P/B slices | All three |
| Weighted prediction | Explicit mode, P-slice luma |
| Sub-MB partitions | 16x16, 16x8, 8x16, 8x8 |
| Deblocking filter | Rayon parallel, skip-aware |
| Scaling lists | Parsed and stored |
| Interlaced (MBAFF/PAFF) | Parsed |
| 4:2:0 / 4:2:2 / 4:4:4 | All formats |

### HEVC/H.265

| Feature | Status |
|---------|--------|
| Main profile (8-bit) | Full |
| Main10 (10-bit, u16 DPB) | Full |
| I/P/B slices | All three |
| CTU quad-tree (16/32/64) | Full |
| 35 intra modes | Planar + DC + angular 2-34 |
| Chroma MC (4-tap filter) | Full color output |
| Branchless CABAC | Packed transitions, CLZ renorm, 32-bit buffered reader |
| Deblocking | BS=0 skip (~85% edges skipped on inter), luma-only mode |
| SAO (band + edge offset) | CTU-only 4KB buffer |
| Tiles/WPP | Parsed (PPS boundaries stored) |

### Performance vs ffmpeg

| Video | yscv | ffmpeg | Ratio |
|-------|------|--------|-------|
| H.264 Baseline 1080p (300fr) | 324ms | 519ms | **1.60x** |
| H.264 High 1080p (300fr) | 332ms | 760ms | **2.28x** |
| H.264 Real Camera 1080p60 (1100fr) | 1187ms | 5372ms | **4.52x** |
| HEVC Main 1080p P/B (300fr) | 575ms | 806ms | **1.40x** |
| HEVC Main 1080p P/B (600fr) | 1288ms | 1808ms | **1.40x** |
| HEVC Main 1080p I-only (180fr) | 1538ms | 1483ms | 0.97x |

All numbers: Apple M-series, `--release`, LTO=thin, single-threaded, best of 5. April 2026.

### SIMD Coverage

`yscv-video` ships **21 named SIMD functions** (8 NEON + 11 SSE2 + 2 AVX2) plus their scalar fallbacks. Each is suffixed with `_neon` / `_sse2` / `_avx2` for grep visibility:

| Domain | NEON | SSE2 | AVX2 |
|---|---|---|---|
| HEVC inverse DCT 4×4 / 16×16 / 32×32 | ✓ ✓ ✓ | ✓ ✓ | — |
| HEVC inverse-DCT 16×16 (separable) | — | ✓ | — |
| HEVC dequant | ✓ | ✓ | — |
| HEVC inter 8-tap horizontal/vertical filters | ✓ | ✓ ✓ | — |
| HEVC inter helpers (`abs_epi32`, `mullo_epi32`, `filter_4rows_v`) | — | ✓ ✓ ✓ | — |
| YUV420 → RGB8 row converter | ✓ | ✓ | ✓ |
| NV12 BT.601 → RGB | ✓ | ✓ | — |
| u8 → f32 normalize for ML preprocessing | ✓ | ✓ | ✓ |

In addition to these named functions, the H.264 path uses scalar code that the Rust compiler auto-vectorises with `-C target-cpu=native`. The pure-Rust SIMD implementations are gated by `#[cfg(target_arch)]` (NEON is compile-time, SSE2/AVX2 use `is_x86_feature_detected!` runtime dispatch).

### Memory

- **Streaming MP4 reader**: reads only moov box at open (~1-5MB), samples lazily via seek
- **27MB RSS** for 41MB H.264 file (< file size)
- **129MB RSS** for HEVC 1080p (DPB + recon buffers)
- No unbounded growth — DPB bounded by SPS, all buffers reused across frames
- MKV: 512MB file size limit (in-memory EBML traversal)

## Containers

| Format | Read | Write |
|--------|------|-------|
| MP4 (avcC/hvcC) | Streaming | No |
| MKV/WebM (EBML) | In-memory (<512MB) | No |
| Annex B raw stream | H.264 + HEVC | No |

## Audio

Audio tracks are detected and metadata extracted, but not decoded:

```rust
if let Some(audio) = reader.audio_info() {
    println!("{:?} {}Hz {}ch", audio.codec, audio.sample_rate, audio.channels);
    // → Aac 44100Hz 2ch
}
```

Supported codecs: AAC, ALAC, Opus, Vorbis, MP3, FLAC (metadata only).

## Hardware Decode

Optional feature-gated backends. Auto-dispatch tries HW, falls back to SW:

```rust
use yscv_video::hw_decode::HwVideoDecoder;
let decoder = HwVideoDecoder::new(VideoCodec::H264)?;
println!("Backend: {}", decoder.backend()); // VideoToolbox / Software
```

| Backend | Platform | Status |
|---------|----------|--------|
| VideoToolbox | macOS | Full (BT.601 NV12→RGB, tested) |
| NVDEC | NVIDIA (Linux/Win) | Parser pipeline written, not tested |
| VA-API | Linux (Intel/AMD) | Init + SW fallback |
| MediaFoundation | Windows | Init + MFT enum + SW fallback |

Note: SW decode is faster than HW for CPU→RGB pipeline (no GPU→CPU copy overhead). HW is useful for 4K+, GPU→display, or Metal inference pipeline.

## Fuzz Testing

3 targets with seed corpus in `fuzz/`:

```bash
cargo install cargo-fuzz
cd fuzz
cargo fuzz run fuzz_h264_nal -- -max_total_time=60
cargo fuzz run fuzz_hevc_nal -- -max_total_time=60
cargo fuzz run fuzz_mkv -- -max_total_time=60
```

## Usage

```rust
use yscv_video::Mp4VideoReader;

let mut reader = Mp4VideoReader::open("input.mp4")?;
println!("Codec: {:?}, Samples: {}", reader.codec(), reader.nal_count());

if let Some(audio) = reader.audio_info() {
    println!("Audio: {:?} {}Hz", audio.codec, audio.sample_rate);
}

while let Ok(Some(frame)) = reader.next_frame() {
    // frame.rgb8_data: Vec<u8>, frame.width, frame.height
    process_frame(&frame);
}
```
