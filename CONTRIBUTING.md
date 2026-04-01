# yscv — Agent and Contributor Guide

This document describes what the yscv project is, how it works, and the rules for making changes. It is written for both human contributors and AI coding agents.

## What is yscv?

yscv is a Rust-native computer vision and deep learning framework. It replaces the typical Python CV/DL stack — OpenCV, NumPy, PyTorch, ONNX Runtime — with a single pure-Rust workspace. There are no Python bindings or runtime dependencies.

The framework covers the full pipeline: tensors and autograd, neural network layers and training, image processing with SIMD optimization, video decoding, object detection (YOLOv8), multi-object tracking (DeepSORT/ByteTrack), face recognition, and ONNX model loading.

## Project shape

The workspace has 14 library crates, 2 application binaries, and an examples crate. There are 1,693 tests across 15 crates, 12 criterion microbenchmarks, and CI with regression gates on GitHub Actions (macOS + Linux + Windows + ARM64).

Key crates and what they do:

- **yscv-tensor** — the foundation. 115+ tensor ops, f32/f16/bf16 dtype, SIMD (AVX/SSE/NEON).
- **yscv-kernels** — CPU and GPU compute backends. SIMD dispatch (AVX + SSE + NEON with scalar fallback), wgpu GPU shaders, rayon threading.
- **yscv-autograd** — dynamic computation graph with 40+ backward ops and gradient checkpointing.
- **yscv-optim** — 8 optimizers (SGD/Adam/AdamW/RAdam/RmsProp/Adagrad/Lamb/Lars) all with NEON+AVX+SSE SIMD, Lookahead meta-optimizer, 11 LR schedulers.
- **yscv-model** — 39 layer types (25 trainable), Trainer API, model zoo (ResNet/VGG/MobileNet/EfficientNet/AlexNet/ViT/DeiT), LoRA, EMA, mixed precision, TensorBoard logging, StreamingDataLoader, distributed training.
- **yscv-imgproc** — 178 image processing ops. The u8 operations (grayscale, blur, morphology, edge detection, resize) have hand-written NEON, AVX2 and SSE/SSSE3 SIMD and beat OpenCV 4.13 on all benchmarked operations.
- **yscv-video** — H.264/HEVC software decode (4.5×/1.4× faster than ffmpeg), MP4/MKV demux, HW decode (VideoToolbox/VAAPI/NVDEC/MediaFoundation), audio metadata extraction, camera I/O. 220 tests, 29 NEON + 31 SSE2 SIMD blocks.
- **yscv-detect** — YOLOv8 ONNX pipeline, NMS, heatmap decoding, anchor generation.
- **yscv-track** — DeepSORT, ByteTrack, Kalman filter, Hungarian assignment, re-identification.
- **yscv-recognize** — cosine similarity matching, VP-Tree ANN indexing.
- **yscv-eval** — 20+ metrics (mAP, MOTA, HOTA, IDF1, PSNR, SSIM), 11 dataset format adapters.
- **yscv-onnx** — 128+ op ONNX runtime with INT8 quantization, graph optimization (Conv+BN folding, constant folding, dead code elimination), fp16/bf16 cast-compute-cast, dynamic shapes.
- **yscv-cli** — command-line tool for inference, benchmarking, and evaluation.

## Performance philosophy

Hot paths use hand-written SIMD intrinsics with runtime feature detection. The dispatch pattern is consistent across the codebase:

1. On macOS, GCD `dispatch_apply` for near-zero-overhead parallelism.
2. On all platforms, rayon work-stealing for multi-threaded execution.
3. aarch64 NEON and x86_64 SSE/SSSE3 SIMD paths for all u8 image ops.
4. Scalar fallback for everything else (other architectures, miri).

The `[profile.release]` uses `lto = "thin"` and `codegen-units = 1`. Target-specific CPU flags are set in `.cargo/config.toml` (apple-m1, neoverse-n1, x86-64-v3).

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
