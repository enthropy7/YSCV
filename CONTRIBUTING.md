# yscv — Agent and Contributor Guide

This document describes what the yscv project is, how it works, and the rules for making changes. It is written for both human contributors and AI coding agents.

## What is yscv?

yscv is a Rust-native computer vision and deep learning framework. It replaces the typical Python CV/DL stack — OpenCV, NumPy, PyTorch, ONNX Runtime — with a single pure-Rust workspace. There are no Python bindings or runtime dependencies.

The framework covers the full pipeline: tensors and autograd, neural network layers and training, image processing with SIMD optimization, video decoding (H.264 + HEVC software + hardware backends), object detection (YOLOv8 / YOLOv11), multi-object tracking (DeepSORT / ByteTrack / Kalman), face recognition (VP-Tree ANN), and ONNX model loading.

## Project shape

The workspace has 16 library crates, 2 application binaries (`apps/bench`, `apps/camera-face-tool`), and an examples crate (21 examples in `examples/src/`). There are 1,861 default tests (1,897 with `--features "rknn metal-backend gpu realtime rknn-validate"`) across the 16 crates, criterion microbenchmarks, and CI with regression gates on GitHub Actions (macOS + Linux + Windows + ARM64). All crates share workspace version `0.1.7`.

Key crates and what they do:

- **yscv-tensor** — the foundation. 115 tensor ops in `ops.rs`, f32/f16/bf16 dtype, SIMD (AVX/SSE/NEON).
- **yscv-kernels** — CPU and GPU compute backends. SIMD dispatch (AVX + SSE + NEON with scalar fallback), 61 WGSL + 4 Metal compute shaders, rayon threading.
- **yscv-autograd** — dynamic computation graph with 61 `Op` variants and gradient checkpointing.
- **yscv-optim** — 8 optimizers (SGD/Adam/AdamW/RAdam/RmsProp/Adagrad/Lamb/Lars) all with NEON+AVX+SSE SIMD, Lookahead meta-optimizer, 11 LR schedulers.
- **yscv-model** — 39 `ModelLayer` variants, Trainer API, model zoo (13 architectures: ResNet18/34/50/101, VGG16/19, MobileNetV2, EfficientNetB0, AlexNet, ViTTiny/Base/Large, DeiTTiny), 17 loss functions, LoRA, EMA, mixed precision, TensorBoard logging, StreamingDataLoader, distributed training.
- **yscv-imgproc** — 159 free public image processing functions in `ops/`. The u8 operations (grayscale, blur, morphology, edge detection, resize) have hand-written NEON, AVX2 and SSE/SSSE3 SIMD and beat OpenCV 4.13 on all benchmarked operations.
- **yscv-video** — H.264/HEVC software decode (4.5×/1.4× faster than ffmpeg), MP4/MKV demux, HW decode (VideoToolbox/VAAPI/NVDEC/MediaFoundation), audio metadata extraction (AAC/ALAC/Opus/Vorbis/MP3/FLAC), camera I/O (V4L2/AVFoundation/MediaFoundation via `nokhwa`). 220 tests, 21 named SIMD functions (8 NEON + 11 SSE2 + 2 AVX2).
- **yscv-detect** — YOLOv8 + YOLOv11 ONNX pipelines, NMS (hard/soft/batched), heatmap decoding, anchor generation, RoI pool/align.
- **yscv-track** — DeepSORT, ByteTrack, 8-state Kalman filter, Hungarian assignment, ReId (color histogram + gallery).
- **yscv-recognize** — cosine similarity matching, VP-Tree ANN indexing, `Recognizer` with enroll/match and JSON snapshot persistence.
- **yscv-eval** — 37 public metric/eval functions (mAP, MOTA, HOTA, IDF1, PSNR, SSIM, classification, regression, counting, camera diagnostics), 8 dataset adapters (COCO, JSONL, KITTI, MOT, OpenImages, VOC, WIDERFACE, YOLO).
- **yscv-onnx** — 128 op ONNX CPU runtime with INT8 quantization, graph optimization (Conv+BN folding, constant folding, Conv+ReLU/BN+ReLU fusion, dead code elimination), fp16/bf16 cast-compute-cast, dynamic shapes; Metal/MPSGraph plan compiler with ~17 op kinds.
- **yscv-cli** — command-line tool for inference, benchmarking, and evaluation.

## Performance philosophy

Hot paths use hand-written SIMD intrinsics with runtime feature detection. The dispatch pattern is consistent across the codebase:

1. On macOS, GCD `dispatch_apply` for near-zero-overhead parallelism.
2. On all platforms, rayon work-stealing for multi-threaded execution.
3. aarch64 NEON and x86_64 SSE/SSSE3 SIMD paths for all u8 image ops.
4. Scalar fallback for everything else (other architectures, miri).

The `[profile.release]` uses `lto = "thin"` and `codegen-units = 1`. Target-specific CPU flags are set in `.cargo/config.toml` (apple-m1, neoverse-n1, x86-64-v3).

## Project priorities (read this first)

The project's centre of gravity is **CPU inference on edge devices** —
Cortex-A SBCs, Rockchip / Allwinner boards, low-power x86, drone
flight controllers. Every change is judged against that target first.
Other backends (wgpu GPU, Apple MPSGraph, Rockchip RKNN, x86 BLAS)
are valuable opt-in extensions but never trump CPU edge perf in a
trade-off. If you're not sure which side a change falls on, run
`onnx-fps` (or the closest representative microbench) on a Cortex-A
class CPU before and after — that's the canonical comparison.

Within that target, three things matter, in order:

1. **Blazing fast.** Anything that lands on a hot path needs SIMD
   coverage (NEON + AVX/SSE + scalar fallback), runtime feature
   detection, and a measured win over the previous code on the
   shape range it targets. "It compiles and the test passes" is
   not enough for an inference loop.
2. **Minimal code.** Smallest correct change. Iterators over hand
   loops where the compiler vectorises them; no unwrap, no dead
   code, no `#[allow(dead_code)]` shortcuts. New abstractions only
   if they earn their keep — three similar lines beat a premature
   trait family.
3. **New functionality is welcome** — but only with the full kit:
   correctness tests (unit + integration where shapes vary),
   criterion microbench against a known baseline, and at least one
   end-to-end use case (a public model, a CLI flag, an `examples/`
   demo, or a `private/onnx-fps` recipe). Fancy code without a
   workload that exercises it accumulates dead weight.

PRs are welcome. The bar is high but the path is documented; if
you're improving an SBC inference number or fixing an issue, that
goes in fast.

## Rules for making changes

Every change should follow this workflow:

1. **Confirm scope** — understand what you are changing and why.
2. **Implement minimally** — make the smallest correct change. Do not over-engineer.
3. **Test** — add or update tests. New logic needs tests. Performance code needs benchmarks.
4. **Document** — if behavior or public APIs changed, update docs in the same commit.
5. **Update capability matrix** — if a new capability was added, update `docs/ecosystem-capability-matrix.md`.
6. **Check freshness** — verify that `agents.md` and `context.md` still reflect reality.

### Technical constraints

- No Python bindings or Python runtime in the main path.
- `unsafe` is allowed only on measured hot paths, must include `SAFETY` comments, and must pass `cargo +nightly miri test`.
- Use `thiserror` for error enums. Do not hand-write `Display`/`Error` impls.
- Public APIs must be documented.
- Prefer Rust-native implementations over C/C++ wrappers.

### Quality gate

- New logic must have tests (unit, integration, or golden-file where relevant).
- Performance-sensitive code should have benchmarks.
- Core numerical operators need reference-parity tests against documented formulas.
- Public API changes require documentation updates.

## What "done" looks like

The framework is done when:

- It provides end-to-end training and inference for CV models without Python.
- The core operator surface reaches parity targets for tensor, autograd, and image ops.
- The camera pipeline can detect, track, count, and recognize people in real time.

## Capability tracking

The canonical map of what is implemented vs what remains is `docs/ecosystem-capability-matrix.md`. It tracks every capability area (tensor, autograd, optimizers, model layers, image processing, video, detection, tracking, ONNX, GPU, evaluation, distributed training) with status markers (Implemented / Partial / Planned) and gap descriptions.

## Roadmap

- HEVC CTU decoding (H.265 video) — infrastructure complete, CTU decode in progress.
- GPU backward kernel expansion — forward + basic backward done, full training kernel coverage on roadmap.
- Async GPU command buffer queueing — for overlapped compute/transfer.
