# yscv-kernels

CPU and GPU compute backends with SIMD dispatch and BLAS integration. Powers all neural network operations in yscv.

## Backends

| Backend | Platform | How |
|---------|----------|-----|
| CPU (SIMD) | All | NEON, SSE2, AVX2 runtime dispatch |
| CPU (BLAS) | All | MKL, Arm PL, or fallback |
| GPU (wgpu) | All | Vulkan, Metal, DX12 via wgpu |
| GPU (Metal) | macOS | Native MPSGraph for Apple Silicon |

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
```

## Tests

120 tests. Criterion benchmarks for matmul, conv, relu, sigmoid, pool.
