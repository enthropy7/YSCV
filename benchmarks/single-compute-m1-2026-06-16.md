# Single-Compute CPU Snapshot

Date: 2026-06-16

Git: `728a550b1b2b` (clean)

Host: MacBookAir

Uname: `Darwin MacBookAir 25.0.0 Darwin Kernel Version 25.0.0: Mon Aug 25 21:17:45 PDT 2025; root:xnu-12377.1.9~3/RELEASE_ARM64_T8103 arm64`

Threads: 1

Iterations: 1000, warmup: 200

YSCV env: `RAYON_NUM_THREADS=1 YSCV_POOL=yscv YSCV_POOL_SPIN_US=200`

Raw logs: `artifacts/single-compute-2026-06-16-022941`

## Toolchain

- Rust: `rustc 1.95.0 (59807616e 2026-04-14);binary: rustc;commit-hash: 59807616e1fa2540724bfbac14d7976d7e4a3860;commit-date: 2026-04-14;host: aarch64-apple-darwin;release: 1.95.0;LLVM version: 22.1.2;`
- Cargo: `cargo 1.95.0 (f2d3ce0bd 2026-03-21)`
- Python: `Python 3.9.6`
- PyTorch: `2.8.0`
- ONNX Runtime: `1.19.2`
- NumPy: `2.0.2`

## Commands

```bash
RAYON_NUM_THREADS=1 YSCV_POOL_SPIN_US=200 ITERS=1000 WARMUP=200 \
  OUT=benchmarks/single-compute-m1-2026-06-16.md bash benchmarks/run-single-compute.sh
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
| add | 1024x1024 | 95 | 143 | 1.51x | 141 | 1.48x | YSCV win |
| mul | 1024x1024 | 92 | 144 | 1.57x | 146 | 1.59x | YSCV win |
| exp | 1024x1024 | 446 | 979 | 2.20x | 739 | 1.66x | YSCV win |
| sum | 1024x1024 | 67 | 51 | 0.76x | 127 | 1.90x | YSCV slower vs PyTorch, YSCV win vs ORT |
| max | 1024x1024 | 57 | 167 | 2.93x | 87 | 1.53x | YSCV win |
| add broadcast last dim | 1024x1024 + 1024 | 128 | 101 | 0.79x | 143 | 1.12x | YSCV slower vs PyTorch, YSCV win vs ORT |
| sub broadcast row minus matrix | 1024 - 1024x1024 | 131 | 101 | 0.77x | 145 | 1.11x | YSCV slower vs PyTorch, YSCV win vs ORT |
| relu | 921600 | 81 | 79 | 0.98x | 81 | 1.00x | YSCV slower vs PyTorch, parity vs ORT |
| sigmoid | 921600 | 184 | 981 | 5.33x | 520 | 2.83x | YSCV win |
| tanh | 1024x1024 | 436 | 3727 | 8.55x | 573 | 1.31x | YSCV win |
| gelu sigmoid approximation | 1024x1024 | 442 | 1366 | 3.09x | 756 | 1.71x | YSCV win |
| silu | 1024x1024 | 436 | 1149 | 2.64x | 752 | 1.72x | YSCV win |
| softmax | 32x1000 | 23 | 47 | 2.04x | 28 | 1.22x | YSCV win |
| log_softmax | 32x1000 | 23 | 40 | 1.74x | 28 | 1.22x | YSCV win |
| softmax | 512x256 | 76 | 193 | 2.54x | 110 | 1.45x | YSCV win |
| layer_norm | 512x256 | 55 | 95 | 1.73x | 263 | 4.78x | YSCV win |
| batch_norm | 1x64x64x3 / 1x3x64x64 | 2 | 10 | 5.00x | 4 | 2.00x | YSCV win |

## Raw Rows

| Runtime | Operation | Shape | Min us | P50 us | Avg us | Status vs YSCV |
|---|---|---:|---:|---:|---:|---|
| yscv | add_1M | 1024x1024 | 89 | 95 | 101 | self |
| pytorch | add_1M | 1024x1024 | 139 | 143 | 144 | YSCV win |
| onnxruntime | add_1M | 1024x1024 | 138 | 141 | 143 | YSCV win |
| yscv | mul_1M | 1024x1024 | 84 | 92 | 95 | self |
| pytorch | mul_1M | 1024x1024 | 139 | 144 | 147 | YSCV win |
| onnxruntime | mul_1M | 1024x1024 | 142 | 146 | 147 | YSCV win |
| yscv | exp_1M | 1024x1024 | 436 | 446 | 447 | self |
| pytorch | exp_1M | 1024x1024 | 958 | 979 | 985 | YSCV win |
| onnxruntime | exp_1M | 1024x1024 | 724 | 739 | 751 | YSCV win |
| yscv | sum_1M_raw_slice | 1024x1024 | 59 | 67 | 69 | self |
| pytorch | sum_1M | 1024x1024 | 51 | 51 | 51 | YSCV slower |
| onnxruntime | sum_1M | 1024x1024 | 126 | 127 | 128 | YSCV win |
| yscv | max_1M_raw_slice | 1024x1024 | 48 | 57 | 58 | self |
| pytorch | max_1M | 1024x1024 | 164 | 167 | 166 | YSCV win |
| onnxruntime | max_1M | 1024x1024 | 85 | 87 | 88 | YSCV win |
| yscv | add_broadcast_1024x1024_by_1024 | 1024x1024 + 1024 | 125 | 128 | 130 | self |
| pytorch | add_broadcast_1024x1024_by_1024 | 1024x1024 + 1024 | 101 | 101 | 101 | YSCV slower |
| onnxruntime | add_broadcast_1024x1024_by_1024 | 1024x1024 + 1024 | 139 | 143 | 142 | YSCV win |
| yscv | sub_broadcast_1024_by_1024x1024 | 1024 - 1024x1024 | 122 | 131 | 135 | self |
| pytorch | sub_broadcast_1024_by_1024x1024 | 1024 - 1024x1024 | 101 | 101 | 102 | YSCV slower |
| onnxruntime | sub_broadcast_1024_by_1024x1024 | 1024 - 1024x1024 | 143 | 145 | 146 | YSCV win |
| yscv | relu_921K | 921600 | 76 | 81 | 85 | self |
| pytorch | relu_921K | 921600 | 79 | 79 | 79 | YSCV slower |
| onnxruntime | relu_921K | 921600 | 80 | 81 | 82 | parity |
| yscv | sigmoid_921K | 921600 | 180 | 184 | 185 | self |
| pytorch | sigmoid_921K | 921600 | 978 | 981 | 1046 | YSCV win |
| onnxruntime | sigmoid_921K | 921600 | 509 | 520 | 523 | YSCV win |
| yscv | tanh_1M | 1024x1024 | 428 | 436 | 438 | self |
| pytorch | tanh_1M | 1024x1024 | 3650 | 3727 | 3728 | YSCV win |
| onnxruntime | tanh_1M | 1024x1024 | 557 | 573 | 576 | YSCV win |
| yscv | gelu_1M | 1024x1024 | 431 | 442 | 444 | self |
| pytorch | gelu_1M_sigmoid_formula | 1024x1024 | 1353 | 1366 | 1382 | YSCV win |
| onnxruntime | gelu_1M_sigmoid_graph | 1024x1024 | 749 | 756 | 760 | YSCV win |
| yscv | silu_1M | 1024x1024 | 428 | 436 | 440 | self |
| pytorch | silu_1M | 1024x1024 | 1132 | 1149 | 1156 | YSCV win |
| onnxruntime | silu_1M_sigmoid_mul_graph | 1024x1024 | 746 | 752 | 756 | YSCV win |
| yscv | softmax_32x1000 | 32x1000 | 21 | 23 | 23 | self |
| pytorch | softmax_32x1000 | 32x1000 | 47 | 47 | 47 | YSCV win |
| onnxruntime | softmax_32x1000 | 32x1000 | 28 | 28 | 28 | YSCV win |
| yscv | log_softmax_32x1000 | 32x1000 | 21 | 23 | 23 | self |
| pytorch | log_softmax_32x1000 | 32x1000 | 40 | 40 | 40 | YSCV win |
| onnxruntime | log_softmax_32x1000 | 32x1000 | 27 | 28 | 28 | YSCV win |
| yscv | softmax_512x256 | 512x256 | 72 | 76 | 78 | self |
| pytorch | softmax_512x256 | 512x256 | 190 | 193 | 244 | YSCV win |
| onnxruntime | softmax_512x256 | 512x256 | 110 | 110 | 111 | YSCV win |
| yscv | layer_norm_512x256 | 512x256 | 54 | 55 | 56 | self |
| pytorch | layer_norm_512x256 | 512x256 | 95 | 95 | 96 | YSCV win |
| onnxruntime | layer_norm_512x256 | 512x256 | 259 | 263 | 267 | YSCV win |
| yscv | batch_norm_1x64x64x3 | 1x64x64x3 / 1x3x64x64 | 2 | 2 | 2 | self |
| pytorch | batch_norm_1x3x64x64 | 1x64x64x3 / 1x3x64x64 | 9 | 10 | 9 | YSCV win |
| onnxruntime | batch_norm_1x3x64x64 | 1x64x64x3 / 1x3x64x64 | 4 | 4 | 4 | YSCV win |
