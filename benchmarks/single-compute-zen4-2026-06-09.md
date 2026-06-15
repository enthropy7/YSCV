# Single-Compute CPU Snapshot

Date: 2026-06-10

Git: `241f36cb9e90` (clean)

Host: nixos

Uname: `Linux nixos 6.18.22 #1-NixOS SMP PREEMPT_DYNAMIC Sat Apr 11 12:26:52 UTC 2026 x86_64 GNU/Linux`

Threads: 1

Iterations: 1000, warmup: 200

YSCV env: `RAYON_NUM_THREADS=1 YSCV_POOL=yscv YSCV_POOL_SPIN_US=200`

Raw logs: `artifacts/single-compute-2026-06-10-191626`

## Toolchain

- Rust: `rustc 1.95.0 (59807616e 2026-04-14);binary: rustc;commit-hash: 59807616e1fa2540724bfbac14d7976d7e4a3860;commit-date: 2026-04-14;host: x86_64-unknown-linux-gnu;release: 1.95.0;LLVM version: 22.1.2;`
- Cargo: `cargo 1.95.0 (f2d3ce0bd 2026-03-21)`
- Python: `Python 3.12.13`
- PyTorch: `2.12.0+cpu`
- ONNX Runtime: `1.26.0`
- NumPy: `2.4.6`

## Commands

```bash
RAYON_NUM_THREADS=1 YSCV_POOL_SPIN_US=200 ITERS=1000 WARMUP=200 \
  OUT=benchmarks/single-compute-zen4-2026-06-09.md bash benchmarks/run-single-compute.sh
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
| add | 1024x1024 | 100 | 98 | 0.98x | 111 | 1.11x | YSCV slower vs PyTorch, YSCV win vs ORT |
| mul | 1024x1024 | 97 | 97 | 1.00x | 116 | 1.20x | parity vs PyTorch, YSCV win vs ORT |
| exp | 1024x1024 | 110 | 634 | 5.76x | 138 | 1.25x | YSCV win |
| sum | 1024x1024 | 29 | 32 | 1.10x | 83 | 2.86x | YSCV win |
| max | 1024x1024 | 30 | 79 | 2.63x | 56 | 1.87x | YSCV win |
| add broadcast last dim | 1024x1024 + 1024 | 72 | 67 | 0.93x | 91 | 1.26x | YSCV slower vs PyTorch, YSCV win vs ORT |
| sub broadcast row minus matrix | 1024 - 1024x1024 | 69 | 68 | 0.99x | 93 | 1.35x | parity vs PyTorch, YSCV win vs ORT |
| relu | 921600 | 55 | 57 | 1.04x | 64 | 1.16x | YSCV win |
| sigmoid | 921600 | 74 | 364 | 4.92x | 202 | 2.73x | YSCV win |
| tanh | 1024x1024 | 188 | 2004 | 10.66x | 208 | 1.11x | YSCV win |
| gelu sigmoid approximation | 1024x1024 | 176 | 3384 | 19.23x | 419 | 2.38x | YSCV win |
| silu | 1024x1024 | 162 | 350 | 2.16x | 227 | 1.40x | YSCV win |
| softmax | 32x1000 | 7 | 15 | 2.14x | 9 | 1.29x | YSCV win |
| log_softmax | 32x1000 | 7 | 14 | 2.00x | 9 | 1.29x | YSCV win |
| softmax | 512x256 | 26 | 63 | 2.42x | 36 | 1.38x | YSCV win |
| layer_norm | 512x256 | 11 | 39 | 3.55x | 111 | 10.09x | YSCV win |
| batch_norm | 1x64x64x3 / 1x3x64x64 | 2 | 8 | 4.00x | 3 | 1.50x | YSCV win vs PyTorch, parity vs ORT |

## Raw Rows

| Runtime | Operation | Shape | Min us | P50 us | Avg us | Status vs YSCV |
|---|---|---:|---:|---:|---:|---|
| yscv | add_1M | 1024x1024 | 96 | 100 | 100 | self |
| pytorch | add_1M | 1024x1024 | 97 | 98 | 100 | YSCV slower |
| onnxruntime | add_1M | 1024x1024 | 110 | 111 | 113 | YSCV win |
| yscv | mul_1M | 1024x1024 | 96 | 97 | 98 | self |
| pytorch | mul_1M | 1024x1024 | 96 | 97 | 98 | parity |
| onnxruntime | mul_1M | 1024x1024 | 109 | 116 | 116 | YSCV win |
| yscv | exp_1M | 1024x1024 | 108 | 110 | 111 | self |
| pytorch | exp_1M | 1024x1024 | 623 | 634 | 635 | YSCV win |
| onnxruntime | exp_1M | 1024x1024 | 136 | 138 | 138 | YSCV win |
| yscv | sum_1M_raw_slice | 1024x1024 | 29 | 29 | 29 | self |
| pytorch | sum_1M | 1024x1024 | 32 | 32 | 32 | YSCV win |
| onnxruntime | sum_1M | 1024x1024 | 82 | 83 | 83 | YSCV win |
| yscv | max_1M_raw_slice | 1024x1024 | 30 | 30 | 30 | self |
| pytorch | max_1M | 1024x1024 | 79 | 79 | 79 | YSCV win |
| onnxruntime | max_1M | 1024x1024 | 56 | 56 | 56 | YSCV win |
| yscv | add_broadcast_1024x1024_by_1024 | 1024x1024 + 1024 | 69 | 72 | 72 | self |
| pytorch | add_broadcast_1024x1024_by_1024 | 1024x1024 + 1024 | 67 | 67 | 68 | YSCV slower |
| onnxruntime | add_broadcast_1024x1024_by_1024 | 1024x1024 + 1024 | 89 | 91 | 91 | YSCV win |
| yscv | sub_broadcast_1024_by_1024x1024 | 1024 - 1024x1024 | 68 | 69 | 69 | self |
| pytorch | sub_broadcast_1024_by_1024x1024 | 1024 - 1024x1024 | 67 | 68 | 69 | parity |
| onnxruntime | sub_broadcast_1024_by_1024x1024 | 1024 - 1024x1024 | 85 | 93 | 92 | YSCV win |
| yscv | relu_921K | 921600 | 55 | 55 | 55 | self |
| pytorch | relu_921K | 921600 | 57 | 57 | 58 | YSCV win |
| onnxruntime | relu_921K | 921600 | 60 | 64 | 65 | YSCV win |
| yscv | sigmoid_921K | 921600 | 73 | 74 | 75 | self |
| pytorch | sigmoid_921K | 921600 | 287 | 364 | 338 | YSCV win |
| onnxruntime | sigmoid_921K | 921600 | 197 | 202 | 202 | YSCV win |
| yscv | tanh_1M | 1024x1024 | 187 | 188 | 190 | self |
| pytorch | tanh_1M | 1024x1024 | 1929 | 2004 | 2031 | YSCV win |
| onnxruntime | tanh_1M | 1024x1024 | 204 | 208 | 210 | YSCV win |
| yscv | gelu_1M | 1024x1024 | 176 | 176 | 177 | self |
| pytorch | gelu_1M_sigmoid_formula | 1024x1024 | 1042 | 3384 | 2325 | YSCV win |
| onnxruntime | gelu_1M_sigmoid_graph | 1024x1024 | 400 | 419 | 419 | YSCV win |
| yscv | silu_1M | 1024x1024 | 162 | 162 | 165 | self |
| pytorch | silu_1M | 1024x1024 | 328 | 350 | 345 | YSCV win |
| onnxruntime | silu_1M_sigmoid_mul_graph | 1024x1024 | 226 | 227 | 228 | YSCV win |
| yscv | softmax_32x1000 | 32x1000 | 6 | 7 | 6 | self |
| pytorch | softmax_32x1000 | 32x1000 | 14 | 15 | 15 | YSCV win |
| onnxruntime | softmax_32x1000 | 32x1000 | 9 | 9 | 9 | YSCV win |
| yscv | log_softmax_32x1000 | 32x1000 | 6 | 7 | 7 | self |
| pytorch | log_softmax_32x1000 | 32x1000 | 13 | 14 | 13 | YSCV win |
| onnxruntime | log_softmax_32x1000 | 32x1000 | 9 | 9 | 9 | YSCV win |
| yscv | softmax_512x256 | 512x256 | 26 | 26 | 26 | self |
| pytorch | softmax_512x256 | 512x256 | 58 | 63 | 61 | YSCV win |
| onnxruntime | softmax_512x256 | 512x256 | 36 | 36 | 37 | YSCV win |
| yscv | layer_norm_512x256 | 512x256 | 11 | 11 | 11 | self |
| pytorch | layer_norm_512x256 | 512x256 | 38 | 39 | 39 | YSCV win |
| onnxruntime | layer_norm_512x256 | 512x256 | 106 | 111 | 112 | YSCV win |
| yscv | batch_norm_1x64x64x3 | 1x64x64x3 / 1x3x64x64 | 2 | 2 | 2 | self |
| pytorch | batch_norm_1x3x64x64 | 1x64x64x3 / 1x3x64x64 | 8 | 8 | 8 | YSCV win |
| onnxruntime | batch_norm_1x3x64x64 | 1x64x64x3 / 1x3x64x64 | 3 | 3 | 3 | parity |
