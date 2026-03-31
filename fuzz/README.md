# Fuzz Testing for yscv-video

## Targets

| Target | What it fuzzes |
|--------|---------------|
| `fuzz_h264_nal` | H.264 Annex B NAL unit parsing + decode |
| `fuzz_hevc_nal` | HEVC NAL unit parsing + decode |
| `fuzz_mkv` | MKV/WebM EBML container parsing |

## Setup

```bash
cargo install cargo-fuzz
```

## Running

```bash
cd fuzz

# H.264 decoder (10 seconds)
cargo fuzz run fuzz_h264_nal -- -max_total_time=10

# HEVC decoder (10 seconds)
cargo fuzz run fuzz_hevc_nal -- -max_total_time=10

# MKV parser (10 seconds)
cargo fuzz run fuzz_mkv -- -max_total_time=10

# Long run (1 hour)
cargo fuzz run fuzz_h264_nal -- -max_total_time=3600
```

## Seed corpus

Pre-generated seed files in `corpus/` directories provide minimal valid
bitstreams as starting points. The fuzzer mutates these to find crashes.
