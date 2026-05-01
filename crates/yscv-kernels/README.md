# yscv-kernels

CPU and GPU compute backends with SIMD dispatch and BLAS integration. Powers all neural network operations in yscv.

## Backends

| Backend | Platform | How |
|---------|----------|-----|
| CPU (SIMD) | All | NEON, SSE2, AVX2 runtime dispatch |
| CPU (BLAS) | All | MKL, Arm PL, or fallback |
| GPU (wgpu) | All | Vulkan, Metal, DX12 via wgpu |
| GPU (Metal) | macOS | Native MPSGraph for Apple Silicon |
| NPU (RKNN) | Rockchip | Native librknnrt.so via dlopen, full SDK 2.4.3a0 |

## Key Operations

- **Elementwise**: add, mul, relu, sigmoid, silu, gelu, mish, tanh, exp
- **MatMul**: blocked GEMM + row GEMM, BLAS dispatch, f16 support (see below)
- **Conv2d**: im2col + GEMM, depthwise SIMD, separable, transpose (see below)
- **Pooling**: max, average, global average
- **Normalization**: batch norm, layer norm, group norm, RMS norm
- **Attention**: multi-head scaled dot-product
- **Softmax**: fused max+exp+sum+div in one pass

### Blocked GEMM (custom, no-BLAS path)

Two-level tiling with MC=128, NC=256, KC=256. Falls back to row GEMM when
any dimension < 32 (`BLOCKED_THRESHOLD`).

**Microkernels** (MR×NR):
- 4×8 — AVX+FMA, AVX, SSE, NEON, scalar
- 4×16 paired — AVX+FMA only (two 4×8 panels fused), k-loop unrolled ×2

**k-loop unrolling (4×16 microkernel)**: inner loop processes two
k-iterations per step with interleaved accumulator writes. On Zen 4
(FMA latency=4 cycles, 2 FMA ports) this spaces each accumulator's
reuse to ≥4 cycles, eliminating pipeline stalls from read-after-write
hazards.

**Set/accumulate mode**: the first k-block (pc=0) stores results directly
into C; subsequent blocks load-accumulate-store. Eliminates the O(M×N)
`fill(0.0)` that traditional GEBP implementations require before the
k-loop.

**Pack panels**: B and A are packed into contiguous NR-wide / MR-wide
strips using `copy_nonoverlapping` and `write_bytes` (unsafe pointer ops)
to avoid bounds-checked indexing that prevents auto-vectorization of the
packing loop.

### Row GEMM (small matrices)

Cascade dispatch: 48→32→16→8→4→scalar columns. k-loop unrolled by 2
with doubled accumulator sets to break FMA latency dependency chains
(FMA latency = 4 cycles, 2 FMA ports on Zen 4 — spacing accumulator
reuse to 4+ cycles eliminates pipeline stalls).

### Depthwise Conv SIMD

Multi-accumulator cascades to saturate both FMA ports:
- AVX+FMA: 32→16→8→scalar (4 independent accumulators)
- AVX: 32→16→8→scalar
- SSE: 4→scalar
- NEON: 16→4→scalar

**Fused activation**: ReLU is applied inside SIMD registers before store
(`_mm256_max_ps` / `vmaxq_f32`), eliminating a separate full-tensor pass.
SiLU falls back to a post-pass (exp is expensive in SIMD without a
library).

### GEMM Store Fusion (bias+activation in microkernel)

`GemmEpilogue` struct carries bias pointer + `Activation` enum through
the entire GEMM call chain. On the last k-block, the microkernel applies
bias and activation directly on accumulator registers before the single
store — eliminating a separate read+write pass over the output tensor.

All 7 microkernel variants support the epilogue:
- 4×16 AVX+FMA, 4×8 AVX+FMA, 4×8 AVX, 4×8 SSE, 4×8 NEON, scalar, scalar_partial

SiLU uses `fast_exp_bittrick` (Schraudolph bit-trick) per-arch.
Identity epilogue (no bias, no activation) compiles to zero overhead.

Entry points: `matmul_2d_slices_fused()`, `blas_sgemm_fused()`.

## Recent Performance Arc (April 2026)

Closed against ONNX Runtime 1.24.4 on a Siamese tracker, Zen 4 6C/12T.
Cumulative default-ON win **~−953 µs @ 6T p50** (4619 → 3665 µs), gap
**2.38× → 2.10×**. See [docs/performance-benchmarks.md](../../docs/performance-benchmarks.md)
for the full thread sweep.

Latest public ARM rerun (Orange Pi Zero 3, 2026-04-21) is tracked in
the same benchmarks doc and currently shows yscv ahead of ORT CPU on
that tracker at 1..4 threads.

### Graph-level fusions (landed at loader, dispatched via `NodeAction`)

- `FusedDwPw` — depthwise 3×3 → pointwise 1×1 single-dispatch pair.
- `FusedPwDw` — pointwise expand → depthwise 3×3, MobileNetV2 opening.
- `FusedTransposeMatMul` — `Transpose(perm=[0,2,1])` absorbed into MatMul
  via `matmul_2d_slices_trans_a` (BLAS `CblasTrans` when available, scratch
  transpose + blocked GEMM otherwise). Mirrors ORT's `MatmulTransposeFusion`.
- `ConvAdd` — residual `Add` fused into blocked-GEMM epilogue.

### Streaming kernels (row-buffered, L1-hot intermediates)

- `fused_pw_dw_3x3` — PW-expand + DW 3×3 with 3-row ZMM register-blocked
  accumulators (AVX-512) / AVX2 / NEON / scalar. The PW intermediate
  (~6 MB on inverted-bottleneck blocks) never touches DRAM — accumulators
  flow straight into the DW window.

### New microkernels

- `yscv_sgemm_12x32_avx512` — MR=12×NR=32 AVX-512F hand-tuned `.S`
  (opt-in `YSCV_AVX512_SGEMM=1`; reaches ~80% theoretical AVX-512 peak
  single-thread on Zen 4).
- `depthwise_conv2d_nhwc_row_avx512` — 128/64/32/16-wide ZMM tiles with
  YMM and scalar tail handling.
- `depthwise_i8_i32_nhwc_dispatch` — symmetric INT8 NHWC depthwise
  3×3/5×5 accumulator for quantized tracker chains; scalar, AVX2,
  AVX-512BW and NEON paths share bitwise parity tests.
- `quantize_linear_f32_to_f32_i8_dispatch` — per-tensor `QuantizeLinear`
  hot path for ONNX QLinear/QDQ boundaries; scalar, AVX2, AVX-512F and NEON
  preserve the runner's rounded-f32 signed-int8 representation.
- `quantize_linear_f32_to_i8_dispatch` — direct activation-entry quantizer for
  internal QLinear side-table tensors; same scalar/AVX2/AVX-512F/NEON dispatch, with packed AVX2/AVX-512 i8 stores,
  but writes real `i8` storage so the ONNX runner does not allocate f32-coded
  integer tensors on the quant-domain path.
- `spec16_tile8_interior` — 8 adjacent output columns with 8 independent
  ZMM accumulators, breaks the 27-tap FMA latency chain in first-layer
  3×3 RGB.
- `matmul_2d_slices_trans_a` — `transA=1` GEMM entry point, BLAS-first
  with blocked-GEMM fallback.

### Assembly coverage (documented)

Hand-written assembly sources live in `src/asm/`:

- `x86_64_sysv.S` / `x86_64_win64.S`
- `aarch64.S`

Hot SGEMM families covered there include:

- x86_64: 4x8, 4x24 fused, 4x32 AVX-512, 12x32 AVX-512
- aarch64: 4x24 NEON, 8x12 NEON

Not every hot path is `.S`: for example, the fused PW->DW tracker path
(`fused_pw_dw_3x3`) on aarch64 is currently NEON intrinsics (including
the PW2X variant), not inline asm.

### Runtime A/B knobs (kernel path selection)

Most important ONNX CPU kernel toggles:

- `YSCV_FUSED_PW_DW_STREAM_OFF=1`
- `YSCV_FUSED_PW_DW_PW2X_OFF=1`
- `YSCV_FUSED_PW_DW_W_TILE=<N>`
- `YSCV_FUSED_DW_PW_STREAM_OFF=1`
- `YSCV_FUSED_DW_PW_STREAM_PADDED=1`
- `YSCV_DIRECT_CONV_WORK_MAX=<N>`

For full semantics/defaults and tracker reproduction commands, see
[`docs/onnx-cpu-kernels.md`](../../docs/onnx-cpu-kernels.md).

### Multi-architecture coverage

Every hot-path kernel ships AVX2 + AVX-512 (x86_64) + NEON (aarch64) +
scalar fallback, selected via `is_x86_feature_detected!` /
`std::arch::is_aarch64_feature_detected!` at runtime. Cross-compile for
`aarch64-unknown-linux-gnu` is in CI; real aarch64 hardware validation
is pending (AVX-512 DW, fused PW+DW streaming, and 8×8 NCHW↔NHWC
permute do not yet have NEON-perf-tuned counterparts — they fall back
to scalar-LLVM-autovec or slower NEON paths). See
[docs/gap-report-2026-04-20.md](../../docs/gap-report-2026-04-20.md)
for the full multi-arch status matrix.

### Bias+Activation Dispatch (NHWC post-conv fallback)

`bias_relu_nhwc_dispatch` / `bias_add_nhwc_dispatch` / `bias_silu_nhwc_dispatch`
fuse bias addition and activation into a single SIMD pass over the GEMM
output tensor. Used as fallback for row-GEMM and BLAS paths where the
microkernel epilogue isn't available. For common channel counts (N=16, N=24),
bias vectors are preloaded into registers before the row loop.
Multiplatform: AVX (8-wide), NEON (4-wide), scalar.

## Features

```toml
[features]
blas = []            # BLAS matmul (default)
mkl = []             # Intel MKL
armpl = []           # Arm Performance Libraries
gpu = []             # wgpu GPU acceleration
metal-backend = []   # macOS Metal (MPSGraph)
rknn = []            # Rockchip NPU (RK3588 / RK3576 / RV1106 / etc.)
```

## RKNN backend (Rockchip NPU)

Enable with `--features rknn`. Loads `librknnrt.so` at runtime via `dlopen`
— the binary compiles on any platform; the NPU path activates only on
Rockchip devices where the runtime library is present.

### Module layout (`src/rknn/`)

| File | Contents |
|------|----------|
| `mod.rs` | Public re-exports + safety contract |
| `consts.rs` | All SDK constants (init flags, query commands, error codes, tensor types/formats/quant types) |
| `ffi.rs` | `#[repr(C)]` structs, function-pointer types, `dlopen` loader, `RknnFunctions` table, `rknn_error_name` helper |
| `backend.rs` | Safe wrappers: `RknnBackend`, `RknnMem`, `ContextPool`, `RknnMatmul`, `CustomOp`, `AsyncFrame`, runtime detection |
| `pipeline.rs` | `RknnPipelinedPool` — triple-buffered per-core `submit`/`wait` API with `#[must_use]` `RknnInferenceHandle`, back-pressure, pre-bound input/output `RknnMem` per slot |
| `custom_op.rs` | `CustomOpHandler` trait + 16-slot trampoline dispatcher for Rust callbacks |
| `compile.rs` | On-device ONNX → RKNN compiler (toolkit2-lite) |

### SDK coverage

**Functions: 34/35 (97%)** of `librknnrt.so` exports are wrapped:

- Core runtime: `init`, `destroy`, `query`, `inputs_set`, `run`, `wait`,
  `outputs_get`, `outputs_release`, `dup_context`
- Multi-core: `set_core_mask`, `set_batch_core_num`
- Dynamic shapes: `set_input_shape`, `set_input_shapes`
- Zero-copy memory: `create_mem`, `create_mem2`, `create_mem_from_fd`,
  `create_mem_from_phys`, `create_mem_from_mb_blk`, `destroy_mem`,
  `mem_sync`, `set_weight_mem`, `set_internal_mem`, `set_io_mem`
- Matmul accelerator: `matmul_create`, `matmul_create_dynamic_shape`,
  `matmul_set_io_mem`, `matmul_set_core_mask`, `matmul_set_quant_params`,
  `matmul_get_quant_params`, `matmul_set_dynamic_shape`, `matmul_run`,
  `matmul_destroy`, `B_normal_layout_to_native_layout`
- Custom ops: `register_custom_ops`, `custom_op_get_op_attr`

**Constants: 100%** — 18 init flags, 17 query commands, 12 tensor types,
4 tensor formats, 3 quant types, 8 NPU core masks, 4 mem-alloc flags,
3 sync modes, 13 matmul types, 6 matmul quant types, 14 error codes
(named: `RKNN_ERR_FAIL` … `RKNN_ERR_TARGET_PLATFORM_UNMATCH`).

**Structs: 24/24** with `#[repr(C)]`. All sizes asserted at **compile
time** against the C ABI — drift in SDK headers fails `cargo build`:

```rust,ignore
const _: () = {
    assert!(std::mem::size_of::<RknnTensorAttr>() == 376);
    assert!(std::mem::size_of::<RknnMatmulInfo>() == 64);
    assert!(std::mem::size_of::<RknnInitExtend>() == 136);
    // … 16 total assertions
};
```

### Quick-start

```rust,ignore
use yscv_kernels::{rknn_available, RknnBackend, ContextPool, NpuCoreMask};

if rknn_available() {
    let model = std::fs::read("yolov8n.rknn")?;
    // Single-core
    let rknn = RknnBackend::load(&model)?;
    let outputs = rknn.run(&[&rgb_frame])?;
    // Or multi-core round-robin (RK3588: scales near-linearly to 3 cores)
    let pool = ContextPool::new(
        &model,
        &[NpuCoreMask::Core0, NpuCoreMask::Core1, NpuCoreMask::Core2],
    )?;
    // Multi-input `&[(name, bytes)]` contract — the old single-slice
    // signature was removed. For a typical single-input vision model:
    let outputs = pool.dispatch_roundrobin(&[("images", &rgb_frame)])?;
}
```

### Zero-copy paths

Three ways to feed data without CPU memcpy:

| Source | API | Use case |
|--------|-----|----------|
| V4L2 DMA-BUF | `RknnBackend::wrap_fd(fd, virt, offset)` | Camera → NPU |
| Rockchip MPP block | `unsafe { rknn.wrap_mb_blk(blk, offset) }` | H.264/HEVC decode → NPU |
| Physical address | `rknn.wrap_phys(phys, virt)` | Custom DMA pools |

Bind the result via `bind_input_by_name(&mem, "images")` and use
`run_bound()` for inference with no data movement.

### Async + multi-core orchestration

Single-context async (overlap one NPU core with CPU work):

```rust,ignore
let frame_id = 42;
let handle = rknn.run_async_bound(frame_id, /*deadline_ms=*/ 5)?;
// … CPU work overlapping with NPU execution …
let outputs = rknn.wait(handle)?;
```

`AsyncFrame` is `#[must_use]`; dropping without `wait()` debug-warns
and silently discards the outputs.

Multi-core pipelined (overlap all NPU cores — on RK3588 this is a ~3×
throughput win over sync `dispatch_roundrobin`):

```rust,ignore
use std::collections::VecDeque;
use yscv_kernels::{NpuCoreMask, RknnInferenceHandle, RknnPipelinedPool};

let pool = RknnPipelinedPool::new(
    &model,
    &[NpuCoreMask::Core0, NpuCoreMask::Core1, NpuCoreMask::Core2],
)?;

let mut in_flight: VecDeque<RknnInferenceHandle> = VecDeque::new();
for _ in 0..pool.slot_count() {
    in_flight.push_back(pool.submit(&[("images", &frame_bytes)])?);
}
for next_frame in stream {
    let oldest = in_flight.pop_front().unwrap();
    consume(pool.wait(oldest)?);
    in_flight.push_back(pool.submit(&[("images", &next_frame)])?);
}
```

Each pipeline slot pre-allocates + pre-binds one `RknnMem` per graph
input and output, so the hot path is `memcpy` + `rknn_run_async` —
no allocations, no binding setup per frame.

### Custom ops with Rust callbacks

```rust,ignore
use yscv_kernels::{CustomOp, CustomOpHandler, CustomOpContext, CustomOpTensor};
use std::sync::Arc;

struct Identity;
impl CustomOpHandler for Identity {
    fn compute(
        &self,
        _ctx: &mut CustomOpContext<'_>,
        ins: &[CustomOpTensor<'_>],
        outs: &[CustomOpTensor<'_>],
    ) -> Result<(), yscv_kernels::KernelError> {
        // SAFETY: SDK guarantees buffers CPU-mapped for callback duration.
        unsafe { outs[0].as_bytes_mut().copy_from_slice(ins[0].as_bytes()); }
        Ok(())
    }
}

let op = CustomOp::cpu("Identity").with_handler(Arc::new(Identity));
let _registration = rknn.register_custom_ops(vec![op])?;
// Hold _registration alive for the rest of the RknnBackend lifetime.
```

Up to **16** Rust handlers per process (one trampoline slot each). Pure
OpenCL kernels (no Rust callback) have no slot limit — drop the
`with_handler(...)` call and use `with_kernel_source(...)` instead.

### On-device compile vs offline toolkit2

The on-device API supports only **fp16** (no calibration) and **int8**
(with a calibration `.txt`):

```rust,ignore
use yscv_kernels::{compile_onnx_to_rknn, RknnCompileConfig};

// fp16 (no quantization)
let bytes = compile_onnx_to_rknn(&onnx, &RknnCompileConfig::default())?;

// int8 with calibration dataset
let cfg = RknnCompileConfig {
    dataset_path: Some("./calibration.txt".into()),
};
let bytes = compile_onnx_to_rknn(&onnx, &cfg)?;
```

For full configuration (`target_platform`, `mean_values`, `std_values`,
asymmetric quant, per-op precision overrides) — use the **offline Python
rknn-toolkit2** on the host and ship the resulting `.rknn` file. See
[`docs/edge-deployment.md`](../../docs/edge-deployment.md) for details.

### Safety contract

Every `unsafe` block carries a `// SAFETY:` comment falling into one of
five categories documented at the top of [`src/rknn/mod.rs`](src/rknn/mod.rs):
FFI calls, dlopen/dlsym, raw-pointer slice construction, RKNN-managed
memory access, COM-style vtable dispatch.

## Tests

170+ tests under `--features rknn`, 139 under `--features gpu`, 139 under
`--features metal-backend`, 165+ default. Criterion benchmarks for matmul,
conv, relu, sigmoid, pool.
