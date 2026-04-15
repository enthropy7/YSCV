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
- **MatMul**: tiled parallel, BLAS dispatch, f16 support
- **Conv2d**: im2col + GEMM, depthwise, separable, transpose
- **Pooling**: max, average, global average
- **Normalization**: batch norm, layer norm, group norm, RMS norm
- **Attention**: multi-head scaled dot-product
- **Softmax**: fused max+exp+sum+div in one pass

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
