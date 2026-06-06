# Single-Compute CPU Snapshot

Date: 2026-06-05

Git: `68b549c75bd7` (dirty)

Host: nixos

Uname: `Linux nixos 6.18.22 #1-NixOS SMP PREEMPT_DYNAMIC Sat Apr 11 12:26:52 UTC 2026 x86_64 GNU/Linux`

Threads: 12

Iterations: 2000, warmup: 250

YSCV env: `RAYON_NUM_THREADS=12 YSCV_POOL=yscv YSCV_POOL_SPIN_US=200`

Raw logs: `artifacts/single-compute-zen4-2026-06-05-autogen`

## Toolchain

- Rust: `rustc 1.95.0 (59807616e 2026-04-14);binary: rustc;commit-hash: 59807616e1fa2540724bfbac14d7976d7e4a3860;commit-date: 2026-04-14;host: x86_64-unknown-linux-gnu;release: 1.95.0;LLVM version: 22.1.2;`
- Cargo: `cargo 1.95.0 (f2d3ce0bd 2026-03-21)`
- Python: `Python 3.12.13`
- PyTorch: `2.12.0+cpu`
- ONNX Runtime: `1.26.0`
- NumPy: `2.4.6`

## Commands

```bash
RAYON_NUM_THREADS=12 YSCV_POOL_SPIN_US=200 ITERS=2000 WARMUP=250 \
  OUT=benchmarks/single-compute-zen4-2026-06-05.md bash benchmarks/run-single-compute.sh
```

## Methodology

- Each row is measured as an isolated per-op process.
- Status is based on p50. `parity` means the p50 delta is at most 1 us.
- GELU is the sigmoid approximation formula/graph: `x * sigmoid(1.702 * x)`.
- YSCV uses NHWC for batch norm; PyTorch and ONNX Runtime use NCHW with the same data volume.

## Results

Times are microseconds. Ratios are competitor p50 divided by YSCV p50.

| Operation | Shape | YSCV p50 | PyTorch p50 | PyTorch/YSCV | ORT p50 | ORT/YSCV | Status |
|---|---:|---:|---:|---:|---:|---:|---|
| add | 1024x1024 | 21 | 22 | 1.05x | 29 | 1.38x | parity vs PyTorch, YSCV win vs ORT |
| mul | 1024x1024 | 21 | 21 | 1.00x | 30 | 1.43x | parity vs PyTorch, YSCV win vs ORT |
| exp | 1024x1024 | 26 | 130 | 5.00x | 35 | 1.35x | YSCV win |
| relu | 921600 | 13 | 13 | 1.00x | 19 | 1.46x | parity vs PyTorch, YSCV win vs ORT |
| sigmoid | 921600 | 15 | 50 | 3.33x | 51 | 3.40x | YSCV win |
| tanh | 1024x1024 | 34 | 372 | 10.94x | 45 | 1.32x | YSCV win |
| gelu sigmoid approximation | 1024x1024 | 31 | 1011 | 32.61x | 71 | 2.29x | YSCV win |
| silu | 1024x1024 | 30 | 62 | 2.07x | 45 | 1.50x | YSCV win |
| softmax | 32x1000 | 3 | 5 | 1.67x | 8 | 2.67x | YSCV win |
| log_softmax | 32x1000 | 3 | 5 | 1.67x | 7 | 2.33x | YSCV win |
| softmax | 512x256 | 7 | 11 | 1.57x | 11 | 1.57x | YSCV win |
| layer_norm | 512x256 | 11 | 12 | 1.09x | 17 | 1.55x | parity vs PyTorch, YSCV win vs ORT |
| batch_norm | 1x64x64x3 / 1x3x64x64 | 2 | 11 | 5.50x | 5 | 2.50x | YSCV win |

## Raw Rows

| Runtime | Operation | Shape | Min us | P50 us | Avg us | Status vs YSCV |
|---|---|---:|---:|---:|---:|---|
| yscv | add_1M | 1024x1024 | 17 | 21 | 31 | self |
| pytorch | add_1M | 1024x1024 | 21 | 22 | 28 | parity |
| onnxruntime | add_1M | 1024x1024 | 27 | 29 | 33 | YSCV win |
| yscv | mul_1M | 1024x1024 | 17 | 21 | 23 | self |
| pytorch | mul_1M | 1024x1024 | 20 | 21 | 25 | parity |
| onnxruntime | mul_1M | 1024x1024 | 28 | 30 | 34 | YSCV win |
| yscv | exp_1M | 1024x1024 | 20 | 26 | 34 | self |
| pytorch | exp_1M | 1024x1024 | 104 | 130 | 215 | YSCV win |
| onnxruntime | exp_1M | 1024x1024 | 29 | 35 | 42 | YSCV win |
| yscv | relu_921K | 921600 | 11 | 13 | 13 | self |
| pytorch | relu_921K | 921600 | 13 | 13 | 18 | parity |
| onnxruntime | relu_921K | 921600 | 18 | 19 | 27 | YSCV win |
| yscv | sigmoid_921K | 921600 | 14 | 15 | 16 | self |
| pytorch | sigmoid_921K | 921600 | 48 | 50 | 57 | YSCV win |
| onnxruntime | sigmoid_921K | 921600 | 50 | 51 | 58 | YSCV win |
| yscv | tanh_1M | 1024x1024 | 32 | 34 | 53 | self |
| pytorch | tanh_1M | 1024x1024 | 311 | 372 | 448 | YSCV win |
| onnxruntime | tanh_1M | 1024x1024 | 45 | 45 | 52 | YSCV win |
| yscv | gelu_1M | 1024x1024 | 29 | 31 | 36 | self |
| pytorch | gelu_1M_sigmoid_formula | 1024x1024 | 648 | 1011 | 1440 | YSCV win |
| onnxruntime | gelu_1M_sigmoid_graph | 1024x1024 | 65 | 71 | 96 | YSCV win |
| yscv | silu_1M | 1024x1024 | 28 | 30 | 51 | self |
| pytorch | silu_1M | 1024x1024 | 58 | 62 | 81 | YSCV win |
| onnxruntime | silu_1M_sigmoid_mul_graph | 1024x1024 | 45 | 45 | 54 | YSCV win |
| yscv | softmax_32x1000 | 32x1000 | 3 | 3 | 3 | self |
| pytorch | softmax_32x1000 | 32x1000 | 4 | 5 | 9 | YSCV win |
| onnxruntime | softmax_32x1000 | 32x1000 | 8 | 8 | 8 | YSCV win |
| yscv | log_softmax_32x1000 | 32x1000 | 3 | 3 | 3 | self |
| pytorch | log_softmax_32x1000 | 32x1000 | 4 | 5 | 5 | YSCV win |
| onnxruntime | log_softmax_32x1000 | 32x1000 | 6 | 7 | 12 | YSCV win |
| yscv | softmax_512x256 | 512x256 | 6 | 7 | 7 | self |
| pytorch | softmax_512x256 | 512x256 | 11 | 11 | 16 | YSCV win |
| onnxruntime | softmax_512x256 | 512x256 | 10 | 11 | 12 | YSCV win |
| yscv | layer_norm_512x256 | 512x256 | 10 | 11 | 11 | self |
| pytorch | layer_norm_512x256 | 512x256 | 11 | 12 | 15 | parity |
| onnxruntime | layer_norm_512x256 | 512x256 | 16 | 17 | 19 | YSCV win |
| yscv | batch_norm_1x64x64x3 | 1x64x64x3 / 1x3x64x64 | 2 | 2 | 2 | self |
| pytorch | batch_norm_1x3x64x64 | 1x64x64x3 / 1x3x64x64 | 10 | 11 | 15 | YSCV win |
| onnxruntime | batch_norm_1x3x64x64 | 1x64x64x3 / 1x3x64x64 | 4 | 5 | 5 | YSCV win |
