# Changelog

All notable changes to the yscv workspace will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Documentation
- **docs**: Synced root `README.md`, `docs/*`, per-crate READMEs, and `CONTRIBUTING.md` with workspace version `0.1.7` and the actual code state. Corrected stale numerical claims to match the current source: 14 crates (was 15 in several files), 1,693 cargo-test count (was 1,678 in one place), 128 ONNX CPU operators (was 126 in `crates/yscv-onnx/README.md`), 115 `Tensor` ops in `ops.rs` (was 80+), 159 `pub fn` items in `crates/yscv-imgproc/src/ops/` (was 100+ / 178), 17 loss functions (was 14+), 61 autograd `Op` variants (was 40+), 50 WGSL + 4 Metal compute shaders (was 20), 21 named SIMD functions in `yscv-video` (was 29 NEON + 31 SSE2), 13 model-zoo architectures, and HEVC software-decode speedup `1.4×` end-to-end (per-crate `yscv-video/README.md` previously said `1.3×`). Added missing `yscv-cli` and `yscv-autograd` rows to the root README crate table. Updated `docs/api-stability.md` to reflect that all crates share workspace version `0.1.7` and that `apps/` binaries are not part of the 14-crate publish set.

### Added
- **yscv-video**: HEVC chroma motion compensation (4-tap filter) — full color YUV420→RGB output instead of grayscale
- **yscv-video**: Streaming MP4 reader — O(1) memory (27MB RSS for 41MB file), lazy seek-based sample reading
- **yscv-video**: MP4 audio track detection — extracts codec, sample_rate, channels from mp4a box
- **yscv-video**: MKV/WebM EBML demuxer with frame index (no per-frame data copy)
- **yscv-video**: Hardware video decode backends — VideoToolbox (macOS, working), NVDEC (parser pipeline), VA-API (init), MediaFoundation (init), all with auto SW fallback
- **yscv-video**: Branchless CABAC engine — packed transition tables, CLZ batch renormalize, 32-bit buffered reader, unsafe get_unchecked on hot paths
- **yscv-video**: BS=0 deblock skip — pred_mode grid eliminates ~85% of deblock work on inter-coded HEVC frames
- **yscv-video**: SSE2 parity with NEON — 31 SSE2 blocks (MC filter, bipred, unipred, dequant, i16→u8, DC prediction)
- **yscv-video**: HEVC weighted prediction table parser (ITU-T H.265 §7.3.6.3)
- **yscv-video**: H.264 sub-MB partitions (P_8x8: 4 sub-blocks with per-block MVD)
- **yscv-video**: H.264 scaling lists parsed and stored in SPS
- **yscv-video**: 10-bit Main10 support (u16 DPB, NEON u16 MC filter)
- **yscv-video**: `--luma-only` and `--hw` flags in bench_video_decode example
- **yscv-video**: Fuzz testing — 3 targets (H.264 NAL, HEVC NAL, MKV) with seed corpus
- **yscv-video**: Audio module — AudioCodec enum, AudioTrackInfo, MP4/MKV codec detection
- **yscv-detect**: Bounds checks in YOLOv8/v11 decoder (guard against malformed tensor output)
- **docs**: `video-pipeline.md` — comprehensive video decode documentation
- **.github/workflows/hw-decode.yml** — CI matrix for macOS+VT, Linux, Windows

### Fixed
- **yscv-video**: OOM on large MP4 files — streaming reader replaces `std::fs::read()` whole-file load
- **yscv-video**: MKV OOM — 512MB file size limit + frame index instead of per-frame data copy
- **yscv-onnx**: CPU depthwise and grouped Conv paths now correctly apply fused SiLU activation
- **yscv-onnx**: `panic!()` in Metal/GPU dispatch replaced with `unreachable!()` (internal invariant)
- **yscv-imgproc**: Mutex poisoning — `.expect("mutex poisoned")` replaced with `.unwrap_or_else(|e| e.into_inner())`
- **yscv-video**: Integer overflow in raw video frame size calculation — uses `checked_mul()`
- **yscv-model**: Removed artificial 8GB file size limits on weight/safetensors loading
- **yscv-onnx**: Removed artificial 4GB limit on ONNX model loading
- **yscv-detect**: False `#[allow(dead_code)]` on `hwc_to_nchw` (function IS used behind cfg(feature))

### Added (earlier)
- **examples**: `bench_yolo` now supports `BENCH_COOLDOWN` env var (default 20s) to insert thermal cooldown pauses between benchmarks, preventing CPU frequency throttling on sustained runs.

## [0.2.0] — 2026-03-18

### Added
- **yscv-imgproc**: Hand-written NEON and SSE/SSSE3 SIMD for all 12 u8 image operations (grayscale, dilate, erode, gaussian, box blur, sobel, median, canny sobel, canny NMS, resize 1ch, resize RGB H-pass, resize RGB V-pass).
- **yscv-imgproc**: GCD `dispatch_apply` threading on macOS with rayon fallback on all platforms.
- **yscv-imgproc**: Direct 3x3 gaussian blur (vextq/alignr, zero intermediate buffers).
- **yscv-imgproc**: Stride-2 fast path for ~2x downscale resize.
- **yscv-track**: 27 new tests for DeepSORT and ByteTrack (57 total).
- **CI**: ARM64 Linux runner (`ubuntu-24.04-arm`).
- **CI**: GPU feature compilation check (`cargo check -p yscv-kernels --features gpu`).
- **build**: Release profile with `lto = "thin"`, `codegen-units = 1`.
- **build**: Target-specific CPU flags in `.cargo/config.toml` (apple-m1, neoverse-n1, x86-64-v3).
- **bench**: OpenCV comparison benchmarks for u8 and f32 operations.
- **bench**: CPU frequency warm-up for Apple Silicon benchmarks.
- **docs**: Architecture guide (`docs/architecture.md`).
- **docs**: OpenCV vs yscv comparison with full methodology in `docs/performance-benchmarks.md`.

### Changed
- **yscv-imgproc**: Grayscale u8 processes entire image as flat array (removed per-row GCD overhead).
- **yscv-imgproc**: Gaussian blur uses direct 3x3 approach instead of separable tiles.
- **yscv-imgproc**: Morphology uses branchless vextq/alignr inner loop.

### Fixed
- **yscv-imgproc**: Canny hysteresis buffer overflow on negative offset underflow.
- **yscv-imgproc**: `to_tensor()` uses `expect()` instead of `unwrap()` with diagnostic message.
- **docs**: All rustdoc unresolved link warnings fixed (29 warnings eliminated).
- **workspace**: All clippy warnings fixed (`cargo clippy -- -D warnings` clean).

### Removed
- `goals.md` — replaced by `docs/ecosystem-capability-matrix.md` as canonical progress tracker.

### Added
- **yscv-optim**: LAMB optimizer with trust ratio scaling for large-batch training.
- **yscv-optim**: LARS optimizer with layer-wise adaptive rate scaling.
- **yscv-optim**: Lookahead meta-optimizer wrapping any `StepOptimizer` with slow-weight interpolation.
- **yscv-tensor**: `scatter_add` operation for index-based additive scatter.
- **yscv-autograd**: Differentiable `gather` and `scatter_add` ops with full backward support.
- **yscv-recognize**: VP-Tree (vantage-point tree) for approximate nearest-neighbor search (`build_index()`, `search_indexed()`).
- **yscv-video**: H.264 P-slice motion compensation (`MotionVector`, `motion_compensate_16x16`, `ReferenceFrameBuffer`).
- **yscv-video**: H.264 B-slice bidirectional prediction (`BiMotionVector`, `BPredMode`, `motion_compensate_bipred`).
- **yscv-video**: H.264 deblocking filter (`boundary_strength`, `deblock_edge_luma`, `deblock_frame`).
- **yscv-video**: HEVC/H.265 decoder infrastructure (VPS/SPS/PPS parsing, `CodingTreeUnit`, `HevcSliceType`).
- **yscv-kernels**: Deformable Conv2d kernel (`deformable_conv2d_nhwc`) with bilinear sampling.
- **yscv-model**: `DeformableConv2dLayer` with `ModelLayer::DeformableConv2d` variant.
- **yscv-track**: Re-identification module (`ReIdExtractor` trait, `ColorHistogramReId`, `ReIdGallery`).
- **yscv-kernels**: GPU compute shaders for batch_norm, layer_norm, and transpose via wgpu.
- **yscv-imgproc**: SURF keypoint detection and descriptor matching (`detect_surf_keypoints`, `compute_surf_descriptors`, `match_surf_descriptors`).
- **yscv-onnx**: `OnnxDtype` enum (Float32/Float16/Int8/UInt8/Int32/Int64/Bool) with `OnnxTensorData` quantize/dequantize support.
- **yscv-model**: TCP transport for distributed training (`TcpTransport` with coordinator/worker roles, `send`/`recv`, `allreduce_sum`).
- **scripts**: `publish.sh` for dependency-ordered crate publishing.
- **scripts**: `bump-version.sh` for workspace-wide version bumps.
- **examples**: `train_cnn` — CNN training recipe with Conv2d + BatchNorm + pooling.
- **examples**: `image_pipeline` — composable image preprocessing pipeline.
- **yscv-model**: Pretrained model zoo with architecture builders (ResNet, VGG, MobileNetV2, EfficientNet, AlexNet) and `ModelHub` remote weight download with caching.
- **yscv-model**: Distributed training primitives — `GradientAggregator` trait, `AllReduceAggregator`, `ParameterServer`, `InProcessTransport`, gradient compression (`TopKCompressor`).
- **yscv-model**: High-level `Trainer` API with `TrainerConfig`, validation split, `EarlyStopping`, `BestModelCheckpoint` callbacks.
- **yscv-model**: Eval/train mode toggle for layers (dropout, batch norm behavior).
- **yscv-model**: Compose-based `Transform` pipeline (Resize, CenterCrop, Normalize, GaussianBlur, RandomHorizontalFlip, ScaleValues, PermuteDims).
- **yscv-kernels**: GPU multi-device scheduling — `MultiGpuBackend`, device enumeration, round-robin/data-parallel/manual scheduling strategies.
- **yscv-video**: H.264 baseline decoder infrastructure — SPS/PPS parsing, bitstream reader, Exp-Golomb decoding, YUV420-to-RGB8 conversion, H.265 NAL type classification.
- **yscv-tensor**: Native FP16/BF16 dtype support with `DType` enum, typed constructors, and `to_dtype()` conversion.
- **yscv-model**: Mixed-precision training (`MixedPrecisionConfig`, `DynamicLossScaler`, `mixed_precision_train_step`).
- **yscv-model**: Embedding, LayerNorm, GroupNorm, InstanceNorm layers with checkpoint roundtrip.
- **yscv-model**: LoRA fine-tuning, EMA, LR finder.
- **yscv-model**: SafeTensors format support.
- **yscv-onnx**: Quantized ONNX runtime ops (QLinearConv, QLinearMatMul, MatMulInteger, ConvInteger, DynamicQuantizeLinear).
- **yscv-onnx**: Expanded opset from 90 to 123 operations.
- **yscv-video**: H.264/H.265 codec infrastructure (NAL parser, MP4 box parser, VideoDecoder/VideoEncoder traits, CAVLC).
- **docs**: API stability policy and release governance (`docs/api-stability.md`).
- **docs**: Full documentation suite (ecosystem capability matrix, performance benchmarks, dataset adapters, training augmentation, training optimizers).

### Changed
- **yscv-tensor**: `DType` enum now supports F32, F16, and BF16 storage variants.
- **yscv-imgproc**: SURF descriptor matching accepts exact matches (dist < 1e-9) unconditionally, bypassing ratio test.
