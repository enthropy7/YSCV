# Edge Deployment: Memory Budget

Memory budget for the `FramePipeline` ring buffer at common edge
resolutions. All numbers assume **4 ring slots** (the recommended default for
3-stage capture/process/output with 1 spare).

## YUV420 capture (1.5 bytes/pixel)

| Resolution  | Pixels     | Bytes/frame | 4 slots  | 8 slots  |
|-------------|------------|-------------|----------|----------|
| 640x480     |    307 200 |      460 KB |   1.8 MB |   3.6 MB |
| 1280x720    |    921 600 |     1.35 MB |   5.4 MB |  10.8 MB |
| 1920x1080   |  2 073 600 |     3.04 MB |  12.2 MB |  24.3 MB |
| 3840x2160   |  8 294 400 |    12.15 MB |  48.6 MB |  97.2 MB |

## NV12 capture (1.5 bytes/pixel)

Same as YUV420 -- identical byte count, different plane layout.

## RGB8 (3 bytes/pixel, post-conversion or direct capture)

| Resolution  | Pixels     | Bytes/frame | 4 slots  | 8 slots  |
|-------------|------------|-------------|----------|----------|
| 640x480     |    307 200 |      900 KB |   3.6 MB |   7.2 MB |
| 1280x720    |    921 600 |     2.70 MB |  10.8 MB |  21.6 MB |
| 1920x1080   |  2 073 600 |     6.08 MB |  24.3 MB |  48.6 MB |
| 3840x2160   |  8 294 400 |    24.30 MB |  97.2 MB | 194.4 MB |

## Sizing the `max_frame_bytes` parameter

Pass the **largest** frame size your pipeline will encounter:

```
max_frame_bytes = width * height * bytes_per_pixel
```

For a YUV420 pipeline at 720p:

```rust
let pipeline = FramePipeline::new(4, 1280 * 720 * 3 / 2);
```

For an RGB8 pipeline at 1080p:

```rust
let pipeline = FramePipeline::new(4, 1920 * 1080 * 3);
```

## Detection storage

Each `PipelineDetection` is 24 bytes (bbox: 4xf32 + score: f32 + class_id:
usize). The per-slot `Vec<PipelineDetection>` starts empty and grows on first
use; after warmup it stays at peak capacity with no further allocations.

For YOLO workloads the typical peak is 20-100 detections per frame (< 2.4 KB),
negligible relative to frame data.

## Recommended configurations

| Class             | Resolution | Slots | Total RAM  |
|-------------------|------------|-------|------------|
| Low-power (e.g. Raspberry Pi 4, small Rockchip) | 640x480    | 4     |     1.8 MB |
| Mid-range edge AI | 1280x720   | 4     |     5.4 MB |
| High-end edge AI  | 1920x1080  | 4     |    12.2 MB |
| Desktop GPU       | 3840x2160  | 8     |    97.2 MB |
| 256 MB RAM class  | 640x480    | 3     |     1.4 MB |
| Multi-core NPU SoC (RK3588 etc.) | 1920x1080 | 4 | 12.2 MB |

## Rockchip NPU acceleration (`rknn` feature)

Enable with `--features rknn` on `yscv-kernels`. Binary runs everywhere: on
non-Rockchip hosts the NPU path is inert (no link-time dependency on
`librknnrt.so`, it is resolved via `dlopen` at startup).

### Runtime detection

```rust
use yscv_kernels::{rknn_available, RknnBackend, ContextPool, NpuCoreMask};

if rknn_available() {
    let model = std::fs::read("yolov8n.rknn")?;
    let pool = ContextPool::new(
        &model,
        &[NpuCoreMask::Core0, NpuCoreMask::Core1, NpuCoreMask::Core2],
    )?;
    let outputs = pool.dispatch_roundrobin(&[("images", &rgb_frame)])?;
}
```

`ContextPool` creates one `RknnBackend` per core mask and cheaply dispatches
frames round-robin across them. On RK3588 this gives near-linear scaling
across 3 NPU cores; on RV1106 the single-core pool still exposes the same
API.

### Pipelined submit/wait (overlap CPU marshaling with NPU compute)

`dispatch_roundrobin` picks a free core but then *blocks* until that core
completes. For maximum throughput, use `RknnPipelinedPool` — it
pre-allocates an `RknnMem` per input / output per slot, pins each slot to
a physical core, and exposes non-blocking `submit` + `wait`:

```rust
use std::collections::VecDeque;
use yscv_kernels::{NpuCoreMask, RknnInferenceHandle, RknnPipelinedPool};

let pool = RknnPipelinedPool::new(
    &model,
    &[NpuCoreMask::Core0, NpuCoreMask::Core1, NpuCoreMask::Core2],
)?;

// Prime the pipeline, then steady-state submit new frame + collect oldest.
let mut in_flight: VecDeque<RknnInferenceHandle> = VecDeque::new();
for _ in 0..pool.slot_count() {
    in_flight.push_back(pool.submit(&[("images", &frame_bytes)])?);
}
loop {
    let oldest = in_flight.pop_front().unwrap();
    let outputs = pool.wait(oldest)?;  // another core is running meanwhile
    let new = next_frame();
    in_flight.push_back(pool.submit(&[("images", &new)])?);
    consume(outputs);
}
```

`RknnInferenceHandle` is `#[must_use]`; over-submitting (more outstanding
handles than the pool's slot count) is safe — `submit` transparently
waits on the oldest slot's previous `AsyncFrame` before reusing its
memory. The pool is `Sync` and fine to share across capture / post-process
threads.

### Zero-copy V4L2 → NPU

DMA-BUF is the critical latency/memory win on 256 MB boards. The kernel
driver writes a frame directly into a physical page, and we hand that page
to the NPU without any CPU copy:

```rust
use yscv_kernels::RknnBackend;
use yscv_video::V4l2Camera;

let (idx, _) = camera.capture_frame_indexed()?;
let fd = camera.export_dmabuf(idx)?;          // VIDIOC_EXPBUF
let virt = camera.buffer_mut(idx)?;
let mem = rknn.wrap_fd(fd, virt, 0)?;         // rknn_create_mem_from_fd
rknn.bind_input_by_name(&mem, "images")?;
rknn.run_bound()?;                             // no memcpy, NPU reads camera page
```

Pair this with preallocated output `RknnMem` buffers (`rknn.alloc_mem(sz)` +
`bind_output_by_name`) so **no** heap traffic happens per frame.

### SRAM for hot tensors

RK3588 and RK3576 have on-chip SRAM the NPU can address directly. Allocate
hot intermediate tensors there to skip DDR bandwidth:

```rust
let scratch = rknn.alloc_sram(1 << 20)?;   // 1 MB
rknn.bind_internal_mem(&scratch)?;
```

Check availability with `RknnBackend::mem_size()` before requesting:

```rust
let mem = rknn.mem_size()?;
if mem.sram_free_bytes > needed {
    // safe to allocate
}
```

### Async pipeline (overlapped execution)

Submit frame N while frame N−1 is still running on another core:

```rust
let frame_id = 42;
let handle = rknn.run_async_bound(frame_id, /*deadline_ms=*/ 5)?;
// … do CPU work while NPU runs …
let outputs = rknn.wait(handle)?;
```

`AsyncFrame` implements `Drop`; forgetting to `wait()` is still safe (the
handle blocks on drop) but yields a debug warning.

### Performance profiling

Load the model with `RknnBackend::load_with_perf()` to arm the SDK's
per-op timer, then read back structured results:

```rust
let rknn = RknnBackend::load_with_perf(&bytes)?;
rknn.run(&[&rgb])?;
let perf = rknn.perf_detail()?;
for op in &perf.per_op {
    println!("{:>4}us  {}  ({})", op.duration_us, op.name, op.op_type);
}
println!("total: {}us", perf.total_us);
```

### Core allocation strategies

| Workload                                | Mask suggestion            |
|-----------------------------------------|----------------------------|
| Single stream, lowest latency           | `Core0`                    |
| Two independent streams                 | `Core0` + `Core1`          |
| Detect + ReID + classify pipeline       | one core per stage         |
| Batched throughput                      | `All` + `set_batch_core_num(3)` |

### MPP zero-copy (hardware-decoded video → NPU)

When you decode H.264/HEVC through Rockchip MPP, the resulting frame already
lives in NPU-addressable memory. Skip the V4L2 path entirely and wrap the
MPP block directly:

```rust
let mb_blk: *mut c_void = mpp_buffer_get_mpp_buffer(buf); // from MPP SDK
// SAFETY: mb_blk is a valid MB_BLK; the MPP buffer outlives `mem`.
let mem = unsafe { rknn.wrap_mb_blk(mb_blk, /*offset=*/ 0)? };
rknn.bind_input_by_name(&mem, "images")?;
rknn.run_bound()?;
```

Same zero-copy guarantees as DMA-BUF, but for the H.264 → NPU pipeline
instead of camera → NPU.

### Dynamic-shape matmul (LLM batching, attention)

`RknnMatmul` runs as an independent NPU unit (parallel to conv graph).
For LLMs where M (batch tokens) varies per call, use the dynamic variant:

```rust
use yscv_kernels::{RknnMatmul, RknnMatmulShape, RknnMatmulType};

let shapes = [
    RknnMatmulShape { m: 1,   k: 4096, n: 4096 },  // single-token decode
    RknnMatmulShape { m: 16,  k: 4096, n: 4096 },  // small batch
    RknnMatmulShape { m: 256, k: 4096, n: 4096 },  // prefill
];
let mm = RknnMatmul::new_dynamic(&shapes, RknnMatmulType::Float16MmInt4ToFloat16)?;

mm.set_shape(shapes[0])?;          // switch to M=1 for next decode step
mm.bind_a(&q_mem)?;                // bind inputs (zero-copy)
mm.bind_b(&w_mem)?;
mm.bind_c(&out_mem)?;
mm.run()?;
```

Inspect quant params at runtime (useful to verify calibration applied):

```rust
let q = mm.quant_params()?;
println!("scale[0..4] = {:?}, zp[0..4] = {:?}", &q.scale[..4.min(q.scale.len())], &q.zero_point[..4.min(q.zero_point.len())]);
```

### Custom operators with Rust callbacks

Operators not covered by the RKNN op set can be implemented as a custom op
with either an embedded OpenCL kernel, a Rust callback, or both. Up to
`MAX_CUSTOM_OP_SLOTS` (= 16) Rust handlers may be active per process.

```rust
use yscv_kernels::{CustomOp, CustomOpHandler, CustomOpContext, CustomOpTensor, CustomOpTarget};
use std::sync::Arc;

struct MyDecoder;
impl CustomOpHandler for MyDecoder {
    fn compute(
        &self,
        _ctx: &mut CustomOpContext<'_>,
        ins: &[CustomOpTensor<'_>],
        outs: &[CustomOpTensor<'_>],
    ) -> Result<(), yscv_kernels::KernelError> {
        // SAFETY: SDK guarantees buffers are CPU-mapped and live for this call.
        let src = unsafe { ins[0].as_bytes() };
        let dst = unsafe { outs[0].as_bytes_mut() };
        // … your op logic …
        dst[..src.len()].copy_from_slice(src);
        Ok(())
    }
}

let op = CustomOp::cpu("MyDecoder")
    .with_handler(Arc::new(MyDecoder));
let _registration = rknn.register_custom_ops(vec![op])?;
// Hold _registration alive for the lifetime of the RknnBackend.
```

For pure OpenCL ops (no Rust callback), drop `with_handler(...)` and pass
`with_kernel_source(/* OpenCL C string */)` instead.

### On-device ONNX → RKNN compilation (limitations)

The on-device `librknn_api.so` (rknn-toolkit2-lite) exposes only
`rknn_init / rknn_build / rknn_export_rknn`. There is **no SDK function**
to set `target_platform`, `quantized_dtype`, channel `mean_values` /
`std_values`, output node list, or per-op precision overrides — those live
exclusively in the **offline host-side Python toolkit2**:

```python
from rknn.api import RKNN
rknn = RKNN()
rknn.config(target_platform='rk3588', mean_values=[[0,0,0]],
            std_values=[[255,255,255]], quantized_dtype='w8a8')
rknn.load_onnx(model='yolov8n.onnx')
rknn.build(do_quantization=True, dataset='./calibration.txt')
rknn.export_rknn('./yolov8n.rknn')  # ship this to device
```

The on-device path supports two modes only:

- **fp16 conversion** — `RknnCompileConfig::default()`; no calibration:
  ```rust
  let bytes = compile_onnx_to_rknn(&onnx, &RknnCompileConfig::default())?;
  ```
- **int8 post-training quantization** — supply a calibration `.txt` file
  listing one preprocessed image path per line:
  ```rust
  let cfg = RknnCompileConfig {
      dataset_path: Some("./calibration.txt".into()),
  };
  let bytes = compile_onnx_to_rknn(&onnx, &cfg)?;
  ```

For every other configuration knob (target SoC, mean/std, asymmetric
quant, per-channel groups, op skips), compile on the host with toolkit2
and deploy the resulting `.rknn` file. Use the `cache_path` argument of
[`load_onnx_as_rknn`](../crates/yscv-kernels/src/rknn/compile.rs) to keep
a fresh `.rknn` cached locally across runs.

### End-to-end example

The [`edge_pipeline`](../examples/src/edge_pipeline.rs) example wires up
camera capture → NPU inference → telemetry overlay → H.264 encode with all
of the above. Run with:

```bash
cargo run --release --example edge_pipeline --features rknn -- \
    --camera /dev/video0 --model yolov8n.rknn \
    --cores 3 --zero-copy --async --sram
```

On a non-Rockchip host the same binary still runs, reporting
`librknnrt.so not found` and falling back to mock detections — same
executable on the dev box and on the embedded target.
