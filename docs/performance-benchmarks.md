# yscv Performance Benchmarks

Comprehensive benchmark results comparing yscv against NumPy, PyTorch, OpenCV, ffmpeg, onnxruntime, and CoreML.

**Last updated**: 2026-04-25 | **Tests**: 1,861 default / 1,897 all-features across 18 crates | **CI**: macOS + Linux + Windows + ARM64

## Hardware & Methodology

- **CPU**: Apple M-series (unified memory architecture)
- **SIMD**: 315 `#[target_feature]`-gated SIMD functions across the workspace (NEON / AVX / AVX2 / SSE / SSE2 / SSSE3), runtime-dispatched on x86 and compile-time-gated on aarch64
- **Threading**: GCD dispatch_apply (macOS), std::thread::scope (Linux/Windows), rayon work-stealing
- **Allocator**: mimalloc (global)
- **Rust**: stable 1.94+, `--release` with `lto = "thin"`, `codegen-units = 1`
- **Measurement**: Best-of-100 for microbenchmarks (minimum of 100 runs after 1 warmup); best-of-5 for video decode
- **Memory**: Streaming I/O — O(1) relative to file size for MP4; bounded DPB for video decoders
- **Correctness**: All results verified against reference implementations (pixel-exact for video, tolerance-based for float ops)

```bash
cargo run -p yscv-bench --release
python benchmarks/python/bench_tensor.py    # NumPy
python benchmarks/python/bench_kernels.py   # PyTorch
python benchmarks/python/bench_opencv.py    # OpenCV
```

## Scorecard Summary

| Category | Wins | Parity | Close | Loss |
|----------|------|--------|-------|------|
| Tensor ops (vs NumPy) | 15 | 2 | 0 | 0 |
| Unary ops (vs NumPy) | 9 | 1 | 0 | 0 |
| Activations (vs PyTorch) | 4 | 1 | 0 | 0 |
| Normalization (vs PyTorch) | 4 | 0 | 0 | 0 |
| MatMul/Conv (vs PyTorch) | 2 | 0 | 0 | 0 |
| u8 imgproc (vs OpenCV) | 10 | 0 | 0 | 0 |
| f32 imgproc (vs OpenCV) | 6 | 0 | 0 | 0 |
| H.264 decode (vs ffmpeg) | 3 | 0 | 0 | 0 |
| HEVC decode (vs ffmpeg) | 2 | 0 | 1 (I-only 0.97×) | 0 |
| Video pixel ops (vs OpenCV) | 1 | 0 | 0 | 0 |
| ONNX inference (vs onnxruntime/tract) | 8 | 0 | 0 | 0 |
| **Total** | **85** | **~4** | **1** | **0** |

## Tensor Elementwise Ops (1M f32, vs NumPy)

| Operation | yscv | NumPy | Ratio | Status |
|-----------|------|-------|-------|--------|
| add | 0.128ms | 0.142ms | 1.11× | WIN |
| sub | 0.154ms | 0.142ms | 0.92× | PARITY |
| mul | 0.134ms | 0.142ms | 1.06× | PARITY |
| sum | **0.020ms** | 0.172ms | **8.6×** | WIN |
| max | **0.020ms** | 0.053ms | **2.7×** | WIN |
| min | **0.020ms** | 0.053ms | **2.7×** | WIN |
| exp | **0.389ms** | 1.704ms | **4.4×** | WIN |
| relu | **0.082ms** | 0.402ms | **4.9×** | WIN |
| argmax | **<0.001ms** | 0.429ms | **>400×** | WIN |
| gt/eq/lt _into | **0.116-0.130ms** | 0.314ms | **2.5×** | WIN |
| transpose 512² | **0.112ms** | 0.184ms | **1.6×** | WIN |

## Tensor Unary Ops (1M f32, vs NumPy)

| Operation | yscv | NumPy | Ratio | Status |
|-----------|------|-------|-------|--------|
| abs | **0.080ms** | 0.088ms | **1.1×** | WIN |
| neg | **0.080ms** | ~0.126ms | **1.6×** | WIN |
| floor | **0.077ms** | 0.088ms | **1.1×** | WIN |
| ceil | **0.077ms** | ~0.350ms | **4.5×** | WIN |
| round | **0.077ms** | ~0.350ms | **4.5×** | WIN |
| sign | **0.099ms** | ~0.350ms | **3.5×** | WIN |
| reciprocal | **0.083ms** | ~0.200ms | **2.4×** | WIN |
| clamp | **0.090ms** | ~0.350ms | **3.9×** | WIN |
| sqrt | **0.156ms** | 0.163ms | 1.04× | PARITY |
| ln | **0.370ms** | ~1.200ms | **3.2×** | WIN |

## Activations (vs PyTorch 2.8, CPU 1-thread)

| Operation | yscv | PyTorch | Ratio | Status |
|-----------|------|---------|-------|--------|
| sigmoid 921K f32 | **0.217ms** | 1.296ms | **6.0×** | WIN |
| softmax 512×256 | **0.098ms** | 0.216ms | **2.2×** | WIN |
| relu 921K f32 | **0.069ms** | 0.105ms | **1.5×** | WIN |
| layer_norm 512×256 | **0.065ms** | 0.117ms | **1.8×** | WIN |
| gelu | — | 2.522ms | — | WIN (old: 0.333ms vs ~0.400ms) |

## MatMul & Conv2d (vs PyTorch 2.8, CPU 1-thread)

| Operation | yscv | PyTorch | Ratio | Status |
|-----------|------|---------|-------|--------|
| matmul 128² | **0.0055ms** | 0.0062ms | **1.13×** | WIN |
| conv2d 32² 3×3 | **0.074ms** | 0.080ms | **1.08×** | WIN |

## Normalization (vs PyTorch 2.8, CPU 1-thread)

| Operation | yscv | PyTorch | Ratio | Status |
|-----------|------|---------|-------|--------|
| layer_norm 512×256 | **0.065ms** | 0.117ms | **1.80×** | WIN |
| batch_norm 64²×16 | **0.028ms** | 0.045ms | **1.61×** | WIN |

## u8 Image Processing (640×480, vs OpenCV 4.13)

| Operation | yscv | OpenCV | Ratio | Status |
|-----------|------|--------|-------|--------|
| resize nearest 320→640 | **0.048ms** | 0.157ms | **3.27×** | WIN |
| resize bilinear 320→640 | **0.068ms** | 0.201ms | **2.96×** | WIN |
| sobel 3×3 | **0.074ms** | 0.169ms | **2.28×** | WIN |
| dilate 3×3 | **0.031ms** | 0.047ms | **1.52×** | WIN |
| erode 3×3 | **0.030ms** | 0.051ms | **1.70×** | WIN |
| box blur 3×3 | **0.049ms** | 0.071ms | **1.45×** | WIN |
| grayscale | **0.025ms** | 0.030ms | **1.20×** | WIN |
| gaussian 3×3 | **0.049ms** | 0.063ms | **1.29×** | WIN |
| median 3×3 | 0.029ms | 0.072ms | 2.48× | WIN |

## f32 Image Processing (ImageF32, 480×640, vs OpenCV)

| Operation | yscv | OpenCV | Ratio | Status |
|-----------|------|--------|-------|--------|
| grayscale | **0.022ms** | 0.027ms | **1.23×** | WIN |
| gaussian 3×3 | **0.051ms** | 0.113ms | **2.22×** | WIN |
| box blur 3×3 | **0.049ms** | 0.131ms | **2.67×** | WIN |
| dilate 3×3 | **0.047ms** | 0.104ms | **2.21×** | WIN |
| sobel 3×3 | **0.055ms** | 0.297ms | **5.40×** | WIN |
| threshold | **0.015ms** | 0.017ms | **1.13×** | WIN |

## Video Decode (vs ffmpeg, single-threaded)

H.264 and HEVC MP4 decode. Pure Rust decoder vs ffmpeg libavcodec (C, `ffmpeg -threads 1`).

**Test methodology:**
- Hardware: Apple M-series (unified memory, NEON SIMD)
- Build: `--release`, LTO=thin, codegen-units=1
- Both decoders single-threaded for fair comparison
- Best of 5 runs, cold CPU between runs
- ffmpeg command: `ffmpeg -threads 1 -benchmark -i <file> -f null -`
- yscv command: `cargo run --release --example bench_video_decode -- <file>`
- Correctness: all frames decoded, pixel_range [0,255], frame count matches ffprobe
- Memory: streaming reader (27MB RSS for 41MB file, O(1) relative to file size)
- Date: April 2026

### H.264

| Video | Frames | yscv | ffmpeg | Ratio | Pixels |
|-------|--------|------|--------|-------|--------|
| H.264 Baseline 1080p | 300 | **324ms** | 519ms | **1.60×** | [0, 255] ✓ |
| H.264 High 1080p | 300 | **332ms** | 760ms | **2.28×** | [0, 255] ✓ |
| **Real Camera H.264 1080p60** | **1100** | **1187ms** | **5372ms** | **4.52×** | **[0, 255] ✓** |

### HEVC (full color — chroma MC enabled)

| Video | Frames | yscv | ffmpeg | Ratio | Pixels |
|-------|--------|------|--------|-------|--------|
| **HEVC Main 1080p P/B 5s** | **300** | **575ms** | **806ms** | **1.40×** | **[0, 255] ✓** |
| **HEVC Main 1080p P/B 10s** | **600** | **1288ms** | **1808ms** | **1.40×** | **[0, 255] ✓** |
| HEVC Main 1080p I-only | 180 | 1538ms | 1483ms | **0.97×** | [0, 255] ✓ |

### Key observations

**H.264:**
- **1.6–4.5× faster than ffmpeg across all profiles** — pure Rust with SIMD IDCT/dequant (NEON + SSE2), rayon parallel deblocking, skip-aware edge filtering, zero-copy reference frames
- **Real camera 1080p60: 4.5× faster** — 1100 frames decoded in 1.2 seconds
- Full pixel range [0, 255] on all supported profiles
- Weighted prediction, 8x8 DCT (High profile), sub-MB partitions (16x8, 8x16, 8x8)
- **Streaming reader**: O(1) memory — 27MB RSS for 41MB file (no full-file loading)

**HEVC:**
- **1.4× faster than ffmpeg on P/B frames** (full color decode with chroma MC + deblock + SAO + YUV→RGB)
- **I-frame near-parity** (0.97×) — intra-only content is CABAC-bound
- **Full color output**: chroma motion compensation with 4-tap filter, real YUV420→RGB
- All profiles decode correctly ([0, 255]) including 10-bit Main10 (u16 DPB)
- **BS=0 edge skip** eliminates ~85% of deblock work on inter-coded frames

**Memory:**
- **Streaming MP4 reader**: reads only moov box at open (1-5MB), samples lazily via seek
- 41MB H.264 file: **27MB RSS** (< file size)
- 3.2MB HEVC file: **129MB RSS** (DPB + recon buffers for 1080p)
- No unbounded growth — DPB bounded by SPS, all buffers reused across frames

**Optimizations applied:**
- Branchless CABAC: mask-based MPS/LPS selection, packed transition tables (128-entry lookup), CLZ batch renormalize, 32-bit buffered bit reader
- Unsafe hot paths: `get_unchecked` for all CABAC table lookups, `ptr::add` for deblock filter, pre-computed scan/context tables, branchless sign `(val ^ -sign) + sign`
- Zero-copy frame management: reusable mv_field, CU list, Y-plane, recon buffers across frames
- NEON (29 blocks) + SSE2 (31 blocks): MC 8-tap horizontal/vertical filter, bipred/unipred clip, DC intra prediction, dequant, DCT 16x16/32x32, i16→u8 saturation, Y→grayscale RGB interleave
- Deblock: BS=0 skip (pred_mode grid), pre-computed tc/beta thresholds, early whole-edge skip, luma-only mode (skip chroma deblock)
- SAO: CTU-only 4KB stack buffer (not full-frame copy)

**Supported formats:**
- H.264: Baseline (CAVLC), Main (CABAC), High (CABAC + 8x8 transform), I/P/B slices, weighted prediction, sub-MB partitions, scaling lists, parallel deblocking
- HEVC: Main, Main10 (10-bit u16 DPB), I/P/B slices, CABAC, deblocking + SAO, CTU quad-tree, tiles (parsed), chroma residual parsing
- MP4 container: avcC/hvcC parameter extraction, stbl/stco/stsz sample table navigation
- MKV/WebM container: EBML demuxer with track/cluster parsing
- Annex B raw stream parser (H.264 + HEVC)
- SIMD: NEON (aarch64) 29 blocks, SSE2 (x86_64) 31 blocks — full cross-architecture coverage
- Parallelism: rayon parallel deblocking, skip-aware edge filtering, zero-copy reference frames

## Video (vs OpenCV)

| Operation | yscv | OpenCV | Ratio | Status |
|-----------|------|--------|-------|--------|
| YUV420→RGB 1080p | **0.166ms** | 0.178ms | **1.07×** | WIN |

## Additional Operations (Apple Silicon, March 2026)

| Operation | Time |
|-----------|------|
| Tensor add 100K | 0.0143ms |
| Tensor mul 100K | 0.0118ms |
| Broadcast add | 0.226ms |
| Broadcast mul | 0.211ms |
| matmul 128² | 0.0055ms |
| matmul rect 96×192×64 | 0.0036ms |
| ReLU 921K f32 | 0.069ms (threaded: 0.062ms) |
| Sigmoid 921K f32 | 0.217ms |
| Add 921K same-shape | 0.126ms |
| BatchNorm 64²×16 | 0.028ms (threaded: 0.023ms) |
| Softmax 512×256 | 0.098ms (threaded: 0.063ms) |
| LayerNorm 512×256 | 0.065ms (threaded: 0.044ms) |
| Conv2d 32² 3×3 | 0.074ms |
| MaxPool 120×160 | 0.159ms (threaded: 0.096ms) |
| Grayscale u8 | 0.025ms |
| Resize nearest u8 | 0.048ms |
| Resize bilinear u8 | 0.068ms |
| Dilate u8 | 0.031ms |
| Erode u8 | 0.030ms |
| Box blur u8 | 0.049ms |
| Sobel u8 | 0.074ms |
| Autograd backward 32² | 0.0041ms |
| Autograd broadcast | 0.0067ms |
| Model linear batch32 | 0.000905ms |
| Model linear+relu+linear | 0.0024ms |
| SGD step batch16 | 0.0096ms |
| SGD step batch64 | 0.0147ms |
| Detect people | 0.060ms |
| Detect faces | 0.165ms |
| Detect heatmap | 0.046ms |
| Track | 0.487ms |
| Recognize query | 0.000448ms |
| CLI people pipeline | 0.075ms |
| CLI face pipeline | 0.162ms |

## ONNX Inference (YOLOv8n / YOLO11n, 640×640 input)

End-to-end model inference benchmarks against onnxruntime, Apple CoreML, and tract.
Methodology: 50 timed runs after warmup, min reported. Apple M1 MacBook Air.

### CPU Inference

| Runtime | YOLOv8n | YOLO11n | Notes |
|---------|---------|---------|-------|
| **yscv** | **30.4ms** | **33.7ms** | Pure Rust, NHWC layout, BLAS matmul |
| onnxruntime 1.19 CPU | 37.4ms | 35.2ms* | *Requires opset 21 conversion; native opset 22 fails |
| onnxruntime 1.19 CoreML | 15.5ms | 47.6ms* | CoreML accelerator; YOLO11n perf degrades with partial coverage |
| tract 0.21 | 217.2ms | FAILED | TDim parse error |

**yscv CPU is 1.2× faster than onnxruntime on YOLOv8n** (30.4ms vs 37.4ms) and comparable on YOLO11n (33.7ms vs 35.2ms). onnxruntime requires manual opset downgrade (22→21) for YOLO11n; yscv handles opset 22 natively.

yscv MPSGraph is **4× faster than ORT CoreML** on YOLOv8n (3.5ms vs 15.5ms). ORT CoreML on YOLO11n degrades to 47.6ms due to partial operator coverage.

### GPU Inference (Metal, Apple M1)

| Runtime | YOLOv8n | YOLO11n | Notes |
|---------|:---:|:---:|-------|
| **yscv MPSGraph** | **3.5ms** | **5.0ms** | Whole-model graph compilation, single GPU dispatch |
| yscv Metal per-op | 12.1ms | 12.6ms | Per-op command buffer, Winograd + MPS GEMM |
| onnxruntime CoreML | 14.2ms | FAILED | Apple Neural Engine delegation |

**MPSGraph** compiles the entire ONNX model into an `MPSGraphExecutable` and runs it as a single GPU dispatch — eliminating per-op encoder transitions. **4× faster than CoreML** on YOLOv8n.

yscv is the only runtime that runs both YOLOv8n and YOLO11n on GPU. CoreML fails on YOLO11n (opset 22).

### Pipelined Throughput (MPSGraph submit/wait, Apple M1)

The pipelined API (`submit_mpsgraph_plan` + `wait_mpsgraph_plan`) triple-buffers input/output buffers and overlaps CPU marshaling with GPU compute. Sustained per-frame wall-time (1000 iter, Siamese tracker, 2 inputs @ 1×3×128×128 + 1×3×256×256):

| Mode | p50 | p99 | Sustained FPS |
|---|---:|---:|---:|
| yscv sync (`--pipeline 1`) | 1.65 ms | 3.15 ms | 605 |
| yscv `--pipeline 2` | **0.37 ms** | **0.62 ms** | **2688** |
| yscv `--pipeline 3` | 0.46 ms | 1.01 ms | 2155 |
| yscv `--pipeline 4` | 0.55 ms | 0.64 ms | 1818 |
| onnxruntime CoreML MLProgram | 1.58 ms | 2.18 ms | 631 |

Depth 2 is the throughput sweet spot (4.3× vs ORT CoreML); depth 4 trades raw p50 for the tightest tail (p99=0.64 ms, max=0.78 ms). Pipeline depth is chosen via `YSCV_MPS_PIPELINE` env var (default 3, clamped 1..=8). The API itself is safe regardless: `submit_mpsgraph_plan` back-pressures if the caller has more outstanding handles than the pipeline depth.

### Siamese Tracker — Full Backend Comparison (Apple Silicon, Apr 2026)

Same two-tower Siamese tracker (219 ONNX nodes, inputs 1×3×128×128 +
1×3×256×256, fp32 zero-fill). Compares every backend yscv ships
against the corresponding ORT provider on the identical host (M-series
macOS). 200 iterations after 40-iter warmup; p50 is the reported number.

| Backend | yscv 0.1.9 | ORT 1.19.2 | yscv vs ORT |
|---|---:|---:|---:|
| **CPU** | **43.1 FPS** (23.2 ms) | 18.9 FPS (53.0 ms) | **2.3× faster** |
| **GPU sync** | **728 FPS** (1.37 ms) | 532 FPS (1.88 ms, CoreML) | **1.4× faster** |
| **GPU pipelined×3** | **1510 FPS** (0.66 ms) | — | **2.9×** over ORT CoreML |
| **GPU peak burst** | 2801 FPS (0.36 ms min) | — | — |

**Why yscv CPU beats ORT CPU on Apple Silicon**: `Accelerate.framework`
dispatches BLAS through Apple's AMX (Advanced Matrix eXtensions) block
— a dedicated matrix accelerator inside the CPU complex, separate from
the Neural Engine. ORT's `CPUExecutionProvider` uses its own
general-purpose SIMD kernels that don't hit AMX, so it leaves ~2.3×
throughput on the table. On Intel the opposite holds: ORT's oneDNN is
highly-tuned for AVX-512 where yscv (which dispatches through
OpenBLAS) is typically 2-3× slower.

**Why yscv Metal beats ORT CoreML on Apple Silicon**: ORT's
`CoreMLExecutionProvider` compiles the graph to CoreML and routes
compatible ops to the **Apple Neural Engine (ANE)** + Metal hybrid. On
the Siamese tracker 216/219 ops run on CoreML; the remaining 3 fall
back to CPU. Every fallback crosses a CPU↔accelerator boundary,
costing synchronization + marshalling. yscv's MPSGraph path compiles
100% of the graph to pure Metal and avoids the hybrid overhead
entirely. Pipelining (3-buffered submit/wait) then overlaps CPU
marshal with GPU compute for another 2× on sustained throughput.

### Pipelined Throughput (RKNN submit/wait, RK3588)

`RknnPipelinedPool` applies the same pattern to Rockchip NPU cores: one slot per `NpuCoreMask`, pre-allocated + pre-bound `RknnMem` per input and output, back-pressured `submit`/`wait`. On RK3588 the pool can drive all 3 NPU cores concurrently; on RV1106 pass `&[Core0]` for a cleanly-typed single-slot async path.

On-device numbers (YOLO / Siamese tracker, int8-quantized `.rknn`) will be added once captured against a physical Rock 4D. Relative gains are expected to mirror the MPSGraph path — pipeline depth equal to NPU-core count ≈ 3× sync throughput, with tail latency tightening as CPU and NPU stop serialising their handshake.

### Metal Pipeline Architecture

The Metal backend compiles an ONNX graph into a sequence of `MetalOp`s executed in a single
fused command buffer. Key optimizations (in order of impact):

| Optimization | Impact | Description |
|-------------|--------|-------------|
| **Winograd F(4×4, 3×3)** | ~40% of GPU time | 2.25× FLOP reduction for stride-1 3×3 convs; SIMD group matrix multiply with f32 accumulation |
| **F16 inter-op pipeline** | Halves bandwidth | All intermediate buffers use f16; weights pre-packed as f16 at compile time |
| **NEON input upload** | Eliminates GPU cast | CPU-side `fcvtn`+`st3` converts f32 NCHW → f16 NHWC faster than GPU kernel |
| **Conv+SiLU+Add fusion** | Fewer ops | Residual addition and activation fused into conv write-back epilogue |
| **Vectorized f16 kernels** | Better throughput | All utility ops (concat, split, permute, resize) use `half4` vectorized I/O |
| **Concat fusion** | Eliminates copies | Conv outputs write directly into concat buffer via `out_stride`/`out_offset` |
| **Detection head fusion** | Fused permute+concat | NHWC→NCHW permutations + spatial concat fused into single `NhwcToFlatConcat` kernel |
| **Zero-cost buffer aliasing** | No-op reshapes | Reshape/Flatten/Squeeze/Unsqueeze alias existing buffers |
| **Parallel softmax** | Threadgroup reduction | Adaptive threadgroup size (32/128/256) with shared-memory reduction |
| **Widened SiLU look-ahead** | Fewer Metal ops | Detects SiLU patterns up to 5 nodes ahead (detection head interleaving) |
| **In-place SiLU/Binary** | Fewer buffers | Dead input buffers reused as output for elementwise ops |

### Metal Per-Op Distribution (YOLOv8n: 110 ops, YOLO11n: 204 ops)

| Op Type | YOLOv8n | YOLO11n |
|---------|:---:|:---:|
| ConvWinograd (3×3 stride=1) | 32 | 28 |
| MpsConv (MPS GEMM for 1×1+) | 30 | 51 |
| Concat | 13 | 34 |
| SplitFused | 8 | 9 |
| CpuReshape (GPU permute) | 4 | 13 |
| Binary/BroadcastBinary | 6 | 28 |
| DepthwiseConv | — | 7 |
| Other (MaxPool, Resize, etc.) | 17 | 34 |

### Competitor Scorecard

| Metric | yscv | onnxruntime | tract |
|--------|------|-------------|-------|
| YOLOv8n CPU | **WIN** (31.7ms vs 100.8ms, 3.2×) | baseline | 6.7× slower |
| YOLO11n CPU | **WIN** (34.3ms) | FAIL | FAIL |
| VballNet CPU | **WIN** (124.1ms vs 196.7ms, 1.6×) | baseline | N/A |
| YOLOv8n GPU (MPSGraph) | **WIN** (3.5ms vs CoreML 14.2ms, 4.1×) | CoreML (ANE hw) | N/A |
| YOLO11n GPU (MPSGraph) | **WIN** (5.0ms) | FAIL | N/A |
| VballNet GPU (MPSGraph) | **WIN** (7.8ms vs CoreML 8.6ms, 1.1×) | CoreML CPU_ONLY (BNNS/AMX) | N/A |
| Opset 22 support | Yes | No | No |

## VballNetGrid Inference (DSConv model, 16.3 GFLOP)

Model: VballNetGridV1b — 13 DSConvBlocks (depthwise 3×3 + pointwise 1×1), 4 MaxPool, head Conv+Sigmoid.
Input `[1, 9, 432, 768]`, output `[1, 27, 27, 48]`, 42 ONNX nodes.

### Optimization Progression (Apple Silicon)

| Stage | Time | FPS | Speedup | What changed |
|-------|------|-----|---------|--------------|
| yscv BEFORE | 558 ms | 1.7 | — | Single-threaded, scalar depthwise |
| + Multi-threading | 257 ms | 3.9 | 2.1× | `ParallelElementwiseConfig::default()` in public API |
| + SIMD depthwise | **124.1 ms** | **8.1** | **4.5×** | NEON/AVX/SSE vectorized depthwise conv |
| onnxruntime CPU | 196.7 ms | 5.1 | — | CPUExecutionProvider baseline |
| onnxruntime CoreML CPU_ONLY | 8.6 ms | 116 | — | BNNS/AMX via CoreML delegate |
| yscv Metal per-op | 47.3 ms | 21.1 | 11.8× | Metal-native fused pipeline, MPS GEMM |
| **yscv MPSGraph** | **7.8 ms** | **128** | **71.5×** | Whole-model GPU graph compilation |

### Key Takeaway

yscv CPU (124.1ms) is **1.6× faster than onnxruntime CPU** (196.7ms) on depthwise-separable models — no special flags needed. MPSGraph (7.8ms) **beats CoreML CPU_ONLY** (8.6ms) which uses Apple's dedicated AMX coprocessor via BNNS — a **1.1× speedup**. MPSGraph provides **16× over CPU**, reaching 128 FPS on Apple Silicon.

## ONNX Siamese Tracker (Zen 4 historical arc, AMD Ryzen 5 7500F, 6C/12T, fp32 CPU)

Model: Siamese tracker, 156 ops after graph optimization, two input branches
(`input.1` 1×3×128×128 template, `input.249` 1×3×256×256 search)
joined in `connect_model`. Primary fp32 CPU benchmark target of the
S.*/A.*/R.* perf arc (Apr 2026, 19 sessions).

Methodology: `RAYON_NUM_THREADS=N ./onnx-fps --iters 500`, median of
3 runs per thread-count, bitwise-identical outputs across all Ns.
ORT 1.24.4 CPUExecutionProvider as the reference.

| Threads | yscv p50 | ORT p50 | gap | yscv scaling | ORT scaling |
|---:|---:|---:|---:|---:|---:|
|  1 | 11.43 ms | 8.05 ms | 1.42× | 1.00× | 1.00× |
|  2 |  6.55 ms | 4.42 ms | 1.48× | 1.74× | 1.82× |
|  4 |  4.15 ms | 2.36 ms | 1.76× | 2.75× | 3.41× |
|  **6** |  **3.66 ms** | **1.74 ms** | **2.10×** | 3.12× | 4.62× |
|  8 |  3.87 ms | 2.28 ms | 1.70× | 2.95× | 3.53× |
| 12 |  4.02 ms | 1.93 ms | 2.08× | 2.84× | 4.16× |

6T is the sweet spot (physical-core count). Beyond 6T, SMT contention
hurts both engines; 12T is strictly worse than 6T.

### Where the remaining gap lives (6T profile, sequential sums)

| op | yscv | ORT | gap | ratio |
|---|---:|---:|---:|---:|
| Conv | 6.31 ms (78 ops) | 2.22 ms (114 ops) | +4.09 ms | **2.84×** |
| MatMul | 0.19 ms (2) | 0.04 ms (2) | +0.14 ms | 4.17× |
| Reshape | 0.11 ms (5) | 0.01 ms (5) | +0.10 ms | 8.01× |
| Reorder (NCHWc) | — | 0.05 ms (7) | — | ORT-only |

**Conv dominates 94% of the gap** — the bulk live in mid-sized pointwise
and inverted-bottleneck layers; ORT uses NCHWc layout throughout while
yscv runs NHWC.

### Landed perf arc (cumulative ~−953 µs @ 6T p50, default-ON)

| step | kernel / change | win @ 6T |
|---|---|---:|
| S.3 | AVX 8×8 NCHW↔NHWC block transpose | −99 µs |
| A2  | AVX-512 depthwise row kernel | −117 µs |
| A3  | First-layer AVX-512 default-on | −64 µs |
| R1  | KHWC-weight fast-path fix for Conv+Add fusion | −91 µs |
| R4  | First-layer 3×3 row-level parallelism | −255 µs |
| R7  | Streaming FusedPwDw AVX-512 register-blocked | −146 µs |
| R9  | FusedTransposeMatMul (mirrors ORT `MatmulTransposeFusion`) | −291 µs |

All landings bitwise-identical or 1-ULP-close to reference. aarch64
cross-compile clean, no-default-features (non-BLAS) build OK, 610+
tests green.

Detailed per-op gap report:
[gap-report-2026-04-20.md](gap-report-2026-04-20.md).

### GPU rerun on the same host (2026-04-25)

Same model and inputs, RTX 4060 added through `--features gpu` (wgpu
backend over Vulkan) and `onnxruntime-gpu` 1.25 with CUDAExecutionProvider:

| backend | p50 | min | output `882` max | drift vs CPU ref |
|---|---:|---:|---:|---:|
| **ORT CUDA EP fp32** (cuDNN, Tensor Cores) | **1.42 ms** | 1.40 ms | 48.9193 | −0.17 (TF32) |
| yscv `gpu` fp16 (wgpu Vulkan) | 5.25 ms | 5.18 ms | 48.1491 | −0.94 |
| yscv `gpu` fp32 (`YSCV_GPU_FP32=1`) | 5.82 ms | 5.72 ms | **49.0915** | **−1 ULP** |
| yscv CPU 6T no-BLAS | 3.05 ms | 2.84 ms | 49.0920 | +1 ULP |
| ORT CPU 1T | 8.07 ms | 8.03 ms | 49.0916 | reference |

Two takeaways:

- **yscv fp32 GPU output is 1-ULP from the CPU reference**, while
  ORT CUDA EP drifts −0.17 because cuDNN auto-uses Tensor Cores in
  TF32 / mixed precision on Ampere+ and downcasts implicitly. That
  is, our fp32 GPU path is the *more numerically faithful* one
  against the CPU reference.
- ORT CUDA EP is **4× faster** than yscv wgpu — structural (cuDNN
  ships shape-specific kernels and uses Tensor Cores via
  `cooperative_matrix`, wgpu compute shaders are vendor-portable
  WGSL with no MMA path). Closing this requires either a vendor
  Vulkan extension wgpu doesn't expose or a separate `cuda-backend`.

For Conv-heavy small-batch graphs like this tracker, the CPU runner
(3.05 ms) beats wgpu (5.25 ms) on the same host — GPU launch latency
dominates the actual compute. wgpu starts to win above batch ≥ 4 or
inference ≥ 30 ms on CPU. See
[`gpu-backend-guide.md`](gpu-backend-guide.md#performance-vs-ort-cuda-ep)
for the full positioning matrix and `YSCV_GPU_FP32` env-flag docs.

### Latest rerun (2026-04-25, post-R10 correctness fix)

R10 fixed a silent-drop `residual_tile` in `microkernel_4x8_dispatch`
on the x86 1×NR-tile path — SIMD/ASM 4×8 variants never received the
residual pointer and dropped the add for Conv+Add shapes whose `n`
left an 8-wide scalar tail. Output `882` max went 84.78 → **49.0920**
(ORT: 49.0916, 1-ULP FP-ordering drift). Perf also improved
because non-BLAS path is now fully correct on tracker shapes.

| build | 1T p50 | 6T p50 | 12T p50 | output 882 max |
|---|---:|---:|---:|---:|
| yscv **no-BLAS** (default for onnx-fps) | **11.22 ms** | **3.17 ms** | 3.34 ms | 49.0920 ✓ |
| yscv **with BLAS** (OpenBLAS 0.3.31)    | 13.00 ms | 8.75 ms | 9.01 ms | 49.0922 ✓ |
| ORT 1.24.4 CPU                          | 8.07 ms  | 1.74 ms | 1.91 ms | 49.0916 ✓ |

yscv no-BLAS vs ORT: 1T **1.39×**, 6T **1.82×**, 12T 1.75× behind.

On this graph BLAS is a **net regression** — 2.76× slower at 6T than
the non-BLAS path. Root causes: `matmul_2d_slices_fused` with BLAS
splits into `blas_sgemm` + `apply_epilogue_fallback` (two passes over
out, ~5.5 ms of extra L2/L3 traffic at 6T), and the whole arc
(R4/R7/R9/A2) only fires on the non-BLAS branch. `OPENBLAS_NUM_THREADS=1`
does not close the gap, so it's not pure thread oversubscription —
rayon workers block serially on sgemm instead of running their own
A/B tiles in parallel.

See [`feature-flags.md`](feature-flags.md#blas--blas-accelerated-matmul)
for the full when-to-enable / when-to-disable BLAS checklist.

## ONNX Siamese Tracker (Orange Pi Zero 3, Cortex-A55, fp32 CPU)

Latest public rerun on Orange Pi Zero 3 (2026-04-21), same model and inputs
for both engines:

- `model sha256`: `6336fbde82e3996128cd18e2141682c7a6b9a7575018ca9ffee974df546f22ab`
- Command (yscv): `./target/release/onnx-fps --model ../model.onnx --input input.1:1x3x128x128 --input input.249:1x3x256x256 --iters 200 --threads N --text`
- Command (ORT): `python3 ./bench_ort_onnx_fps.py --model ../model.onnx --input input.1:1x3x128x128 --input input.249:1x3x256x256 --iters 200 --threads N --text`

| Threads | yscv p50 | ORT p50 | yscv vs ORT |
|---:|---:|---:|---:|
| 1 | **461.63 ms** | 499.25 ms | **1.08× faster** |
| 2 | **252.08 ms** | 273.18 ms | **1.08× faster** |
| 3 | **192.91 ms** | 199.41 ms | **1.03× faster** |
| 4 | **150.17 ms** | 164.56 ms | **1.10× faster** |

Takeaway: on this ARM target, current default `yscv-onnx` is now ahead of
ORT CPU across all tested thread counts (1..4), with the largest advantage
at 4 threads.

Kernel-path notes for this run (streaming fused Conv paths, asm vs
intrinsics, and runtime A/B toggles):
[`onnx-cpu-kernels.md`](onnx-cpu-kernels.md).
Latest private Zen4 rerun snapshot (fp32 / QDQ-fast / QLinear, 1T/6T):
[`onnx-cpu-kernels.md`](onnx-cpu-kernels.md) and
[`perf-arc-2026-04.md`](perf-arc-2026-04.md).

## Cross-Platform SIMD Coverage

| Operation | NEON | SSE | AVX |
|-----------|:---:|:---:|:---:|
| Tensor binary/unary (1M f32) | ✅ 4× unroll | ✅ 4-wide | ✅ 4× unroll (32 elem) |
| Activations (sigmoid/tanh/silu) | ✅ 3-term poly | ✅ poly | ✅ poly |
| Softmax/LogSoftmax | ✅ fused | ✅ fused | ✅ fused |
| MatMul | ✅ BLAS | ✅ BLAS | ✅ BLAS + FMA |
| Conv2d 3×3 | ✅ direct NEON | ✅ direct SSE | ✅ im2col + BLAS |
| Depthwise Conv2d | ✅ 4-wide FMA | ✅ 4-wide | ✅ 8-wide |
| u8 morphology/filter/sobel | ✅ 16B/iter | ✅ 16B/iter | ✅ 32B/iter (AVX2) |
| f32 filter/morphology/geometry | ✅ 4-wide | ✅ 4-wide | ✅ 8-wide |
| Median u8 | ✅ sort network | ✅ sort network | — |
| YUV→RGB | ✅ NEON + GCD | ✅ SSE + threads | ✅ AVX2 + threads |

## Optimization Techniques

- **315 `#[target_feature]`-gated SIMD functions** with runtime CPU detection
- **All dispatch functions `#[inline]`** for cross-crate inlining
- **AlignedVec::uninitialized** — skip output zeroing in hot paths
- **ImageU8/ImageF32** — zero-overhead wrappers bypass Tensor allocation
- **GCD dispatch_apply** — macOS near-zero threading (~0.3µs)
- **mimalloc** — thread-local arena pools
- **Fused kernels** — single-pass softmax, sigmoid, attention
- **im2col + BLAS** — Accelerate/OpenBLAS for matmul/conv2d/conv3d
- **Flash Attention** — tiled O(Br×Bc) memory, online softmax
- **Integer GEMM** — quantized matmul with i32 accumulation (no dequant overhead)

## Test Infrastructure

### Test Suite Summary (April 2026)

| Crate | Tests | Coverage |
|-------|-------|----------|
| yscv-model | 365 | Serialization, training loops, data loading, distributed |
| yscv-imgproc | 225 | All u8/f32 ops, SIMD paths, color conversion |
| yscv-video | 230 | H.264/HEVC decode (Main + Main10 + Rext + weighted prediction + tiles + WPP + chroma deblock/SAO), MP4/MKV parsing, HW detect |
| yscv-tensor | 207 | Elementwise, matmul, broadcast, BLAS dispatch |
| yscv-kernels | 120 | CPU ops, GPU backend, SIMD activation, GEMM |
| yscv-autograd | 106 | Forward/backward graph, all op gradients |
| yscv-eval | 95 | COCO/YOLO/VOC/KITTI/WiderFace/MOT metrics |
| yscv-onnx | 166 | Per-operator coverage for all 122 CPU dispatch arms, fusion regressions, quantization, vision ops |
| yscv-optim | 76 | SGD, Adam, LR schedulers, weight decay |
| yscv-detect | 60 | YOLOv8/v11 decode, NMS, letterbox |
| yscv-track | 57 | Hungarian, Kalman, IoU matching |
| yscv-cli | 42 | Config parsing, diagnostics |
| yscv-recognize | 16 | Embedding extraction, cosine similarity |
| **Total** | **1808** | |

### CI Matrix

| Platform | Runner | Features | What's Tested |
|----------|--------|----------|---------------|
| macOS (ARM) | macos-latest | default + videotoolbox | Full workspace + HW decode |
| Linux (x86) | ubuntu-latest | default + gpu | Full workspace + WGPU |
| Linux (ARM) | ubuntu-24.04-arm | default | Full workspace + NEON |
| Windows | windows-latest | default | Full workspace |

### CI Jobs
- **workspace-compat**: Multi-platform build + test
- **quality**: fmt + clippy + CLI integration + benchmark gates + eval format verification
- **miri**: Unsafe code soundness (yscv-tensor, yscv-kernels)
- **hw-decode**: VideoToolbox (macOS), SW fallback (Linux/Windows)
- **benchmark gates**: Criterion microbenchmarks for 12 crates with trend tracking

### Test Data
- **Video**: `examples/src/CENSUSWITHOUTLOGO.mp4` (41MB, H.264 1080p60, 1100 frames)
- **Images**: `examples/src/testtraffic.png`, `testtraffic2.png` (3.4-3.5MB)
- **Models**: `examples/src/slowwork/yolo{v8n,11n}.onnx` (10-12MB, gitignored)
- **Eval samples**: `benchmarks/eval-*` (all formats, <10KB each)
- **Fuzz corpus**: `fuzz/corpus/` (H.264, HEVC, MKV seed files)
- **Baselines**: `benchmarks/ci-baseline-*.txt`, `trend-baseline-*.tsv`

### How to Run

```bash
# Full workspace test (1693 tests)
cargo test

# Single crate
cargo test -p yscv-video

# Video decode benchmark
cargo run --release --example bench_video_decode -- examples/src/CENSUSWITHOUTLOGO.mp4

# Compare with ffmpeg
ffmpeg -threads 1 -benchmark -i examples/src/CENSUSWITHOUTLOGO.mp4 -f null -

# Criterion microbenchmarks
cargo bench -p yscv-kernels
cargo bench -p yscv-imgproc

# Miri soundness check
cargo +nightly miri test -p yscv-tensor --lib

# Fuzz testing
cd fuzz && cargo fuzz run fuzz_h264_nal -- -max_total_time=60
```
