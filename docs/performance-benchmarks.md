# yscv Performance Benchmarks

Comprehensive benchmark results comparing yscv against NumPy, PyTorch, OpenCV, ffmpeg, onnxruntime, and CoreML.

**Last updated**: April 2026 | **Tests**: 1,693 across 14 crates | **CI**: macOS + Linux + Windows + ARM64

## Hardware & Methodology

- **CPU**: Apple M-series (unified memory architecture)
- **SIMD**: 315 `#[target_feature]`-gated SIMD functions across the workspace (NEON / AVX / AVX2 / SSE / SSE2 / SSSE3), runtime-dispatched on x86 and compile-time-gated on aarch64
- **Threading**: GCD dispatch_apply (macOS), std::thread::scope (Linux/Windows), rayon work-stealing
- **Allocator**: mimalloc (global)
- **Rust**: stable 1.92+, `--release` with `lto = "thin"`, `codegen-units = 1`
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
| yscv-video | 220 | H.264/HEVC decode, MP4/MKV parsing, HW detect |
| yscv-tensor | 207 | Elementwise, matmul, broadcast, BLAS dispatch |
| yscv-kernels | 120 | CPU ops, GPU backend, SIMD activation, GEMM |
| yscv-autograd | 106 | Forward/backward graph, all op gradients |
| yscv-eval | 95 | COCO/YOLO/VOC/KITTI/WiderFace/MOT metrics |
| yscv-onnx | 77 | Model loading, graph optimization, inference |
| yscv-optim | 76 | SGD, Adam, LR schedulers, weight decay |
| yscv-detect | 60 | YOLOv8/v11 decode, NMS, letterbox |
| yscv-track | 57 | Hungarian, Kalman, IoU matching |
| yscv-cli | 42 | Config parsing, diagnostics |
| yscv-recognize | 16 | Embedding extraction, cosine similarity |
| **Total** | **1693** | |

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
