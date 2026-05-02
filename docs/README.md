# yscv documentation

Pick your starting point.

## Just landed here?

| You want to... | Read first |
|---|---|
| Run something **right now** (5 min) | [QUICKSTART.md](../QUICKSTART.md) |
| Walk through the framework end-to-end | [getting-started.md](getting-started.md) |
| Find recipes for a specific task | [cookbook.md](cookbook.md) |
| Try the worked examples | [../examples/README.md](../examples/README.md) |
| Something is broken | [troubleshooting.md](troubleshooting.md) |

---

## By topic

### Getting started

- **[QUICKSTART.md](../QUICKSTART.md)** — three 5-minute paths: image
  processing, training, edge deployment. Pick one, copy code, you're
  running.
- **[getting-started.md](getting-started.md)** — full progressive
  tutorial through every layer of the framework. Start here for a
  proper walkthrough.
- **[cookbook.md](cookbook.md)** — 1400-line recipe collection.
  Self-contained code for every common task with copy-paste commands.

### Pipeline framework (TOML-driven runtime)

- **[pipeline-config.md](pipeline-config.md)** — full TOML schema
  reference: `[camera]`, `[output]`, `[encoder]`, `[[tasks]]`,
  `[realtime]`. Validation order, accelerator-by-feature table.
- **[edge-deployment.md](edge-deployment.md)** — Rockchip / NPU deep
  dive: DMA-BUF zero-copy V4L2→NPU, on-chip SRAM, MPP zero-copy from
  hardware decoder, dynamic-shape matmul for LLMs, custom OpenCL ops
  with Rust callbacks.

### Inference

- **[onnx-inference.md](onnx-inference.md)** — ONNX runtime: CPU
  (122 ops), Apple MPSGraph, wgpu (cross-platform GPU). Triple-
  buffered submit/wait API for sustained throughput.
- **[onnx-cpu-kernels.md](onnx-cpu-kernels.md)** — CPU hot-path map
  for ONNX Conv/MatMul: fused kernels, asm vs intrinsics coverage,
  Orange Pi tracker numbers, and A/B env toggles.
- **[mpsgraph-guide.md](mpsgraph-guide.md)** — standalone guide for
  the Apple Silicon MPSGraph path: when to use, full API reference,
  sync vs pipelined, multi-input models, fallback, troubleshooting.
  One-stop shop if you want the fastest inference on macOS.
- **[gpu-backend-guide.md](gpu-backend-guide.md)** — standalone guide
  for the wgpu cross-platform GPU path (Vulkan / Metal / DX12 / GL):
  platform selection, compiled f16 plans, multi-GPU,
  WGPU_BACKEND env var, troubleshooting. The one to read on
  Linux/Windows.
- **[feature-flags.md](feature-flags.md)** — canonical reference for
  every Cargo feature flag across the workspace: what it does, setup
  per platform, combination recipes.
- **[../crates/yscv-pipeline/README.md](../crates/yscv-pipeline/README.md)** —
  multi-accelerator dispatcher, recovery, hot-reload, watchdog.
- **[../crates/yscv-onnx/README.md](../crates/yscv-onnx/README.md)** —
  ONNX-specific layer documentation.
- **[../crates/yscv-kernels/README.md](../crates/yscv-kernels/README.md)** —
  CPU SIMD + GPU kernels, RKNN backend with full SDK 2.4.3a0
  coverage.

### Video

- **[video-pipeline.md](video-pipeline.md)** — H.264 / HEVC / AV1
  software decode (faster than ffmpeg), hardware decode
  (VideoToolbox / VAAPI / NVDEC / MediaFoundation), MP4 / MKV
  container parsing, audio metadata.
### Training

- **[training-optimizers.md](training-optimizers.md)** — 8
  optimizers (SGD → LARS), Lookahead meta-optimizer, 11 LR
  schedulers, 17 loss functions, gradient clipping, the high-level
  Trainer API.
- **[training-augmentation.md](training-augmentation.md)** — data
  augmentation pipeline: 12+ transforms, MixUp/CutMix, sampling,
  reproducibility (deterministic via seed).
- **[dataset-adapters.md](dataset-adapters.md)** — supported dataset
  formats: training (JSONL, CSV, ImageManifest, ImageFolder),
  evaluation (COCO, OpenImages, YOLO, VOC, KITTI, WIDER FACE, MOT
  Challenge).

### Architecture & references

- **[architecture.md](architecture.md)** — crate dependency layers,
  SIMD dispatch model (AVX/SSE/NEON + scalar fallback), threading
  strategy, memory patterns, source-file map.
- **[ecosystem-capability-matrix.md](ecosystem-capability-matrix.md)** —
  canonical map of every capability area, status, and gap relative
  to a Python CV/DL stack.
- **[performance-benchmarks.md](performance-benchmarks.md)** — full
  methodology + scorecard vs OpenCV / ffmpeg / NumPy / PyTorch /
  onnxruntime / CoreML. Reproduction commands.
- **[api-stability.md](api-stability.md)** — versioning policy,
  per-crate stability tiers, release checklist, publish order.

### Operations

- **[troubleshooting.md](troubleshooting.md)** — common build /
  runtime / performance errors with concrete fixes per platform.

### Per-crate READMEs

| Crate | What it does |
|---|---|
| [yscv](../crates/yscv/README.md) | Umbrella — re-exports prelude + per-crate APIs. |
| [yscv-tensor](../crates/yscv-tensor/README.md) | N-dim tensor with 115 ops, f32/f16/bf16, SIMD-aligned storage. |
| [yscv-kernels](../crates/yscv-kernels/README.md) | CPU + GPU compute backends, 315 SIMD functions, RKNN bindings. |
| [yscv-autograd](../crates/yscv-autograd/README.md) | Reverse-mode autodiff, 61 backward op variants. |
| [yscv-optim](../crates/yscv-optim/README.md) | Optimizers + LR schedulers + Lookahead. |
| [yscv-model](../crates/yscv-model/README.md) | 39 layer types, Trainer, model zoo (17 architectures), LoRA, EMA. |
| [yscv-imgproc](../crates/yscv-imgproc/README.md) | 160 image-processing ops (blur, edges, morphology, features, color). |
| [yscv-video](../crates/yscv-video/README.md) | H.264/HEVC/AV1 codecs, hardware decode, MP4/MKV, V4L2 camera, audio. |
| [yscv-detect](../crates/yscv-detect/README.md) | YOLOv8/v11 pipeline, NMS, heatmap decoding. |
| [yscv-track](../crates/yscv-track/README.md) | DeepSORT, ByteTrack, Kalman filter, Hungarian assignment. |
| [yscv-recognize](../crates/yscv-recognize/README.md) | Cosine matching, VP-Tree ANN indexing, enroll/match. |
| [yscv-eval](../crates/yscv-eval/README.md) | Classification / detection / tracking / regression / image-quality metrics. |
| [yscv-onnx](../crates/yscv-onnx/README.md) | 122-op ONNX CPU runtime, INT4/INT8 quantization, LLM generation, MPSGraph GPU. |
| [yscv-pipeline](../crates/yscv-pipeline/README.md) | TOML-driven multi-accelerator dispatch, RT wiring, recovery. |
| [yscv-cli](../crates/yscv-cli/README.md) | Inference + evaluation CLI: camera diagnostics, dataset eval, pipeline runner. |

---

## By role

### "I'm building a CV product, want a single binary"

1. [QUICKSTART §1](../QUICKSTART.md#1-image-processing--detection-cv-user) — get running in 3 minutes.
2. [cookbook.md §image processing](cookbook.md) — find the ops you need.
3. [yscv-imgproc README](../crates/yscv-imgproc/README.md) — the full op catalogue.

### "I'm training a model"

1. [QUICKSTART §2](../QUICKSTART.md#2-train-a-neural-network-ml-user) — first training loop.
2. [getting-started.md §step 4](getting-started.md#step-4--train-a-model) — full Trainer walkthrough.
3. [training-optimizers.md](training-optimizers.md) + [training-augmentation.md](training-augmentation.md) + [dataset-adapters.md](dataset-adapters.md) — when you need control.

### "I'm deploying on a Rockchip / drone / FPV board"

1. [QUICKSTART §3](../QUICKSTART.md#3-edge-deployment-rockchip-npu) — minimal TOML + 5-line `main.rs`.
2. [edge-deployment.md](edge-deployment.md) — every NPU feature (DMA-BUF, SRAM, MPP, custom ops).
3. [pipeline-config.md](pipeline-config.md) — TOML schema reference.
4. [yscv-pipeline README](../crates/yscv-pipeline/README.md) — runtime entry point.
5. [troubleshooting.md](troubleshooting.md) — when `librknnrt.so` isn't found, when SCHED_FIFO needs CAP_SYS_NICE, etc.

### "I'm evaluating yscv for production"

1. [README.md (top-level)](../README.md) — what is yscv, what's the pitch.
2. [performance-benchmarks.md](performance-benchmarks.md) — how fast on what hardware vs whom.
3. [ecosystem-capability-matrix.md](ecosystem-capability-matrix.md) — capability gaps vs Python stack.
4. [architecture.md](architecture.md) — design philosophy + crate layering.
5. [api-stability.md](api-stability.md) — versioning + breaking-change policy.

### "I want to contribute"

1. [architecture.md](architecture.md) — understand the layers first.
2. [ecosystem-capability-matrix.md](ecosystem-capability-matrix.md) — find what's missing.
3. [api-stability.md](api-stability.md) — what's safe to change vs needs RFC.
4. Open an issue or PR. We respond fast.
