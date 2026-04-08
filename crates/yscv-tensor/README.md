# yscv-tensor

SIMD-accelerated tensor library. 115 `Tensor` operations in `ops.rs`, f32/f16/bf16 support, NumPy-style broadcasting, 32-byte aligned memory.

```rust
use yscv_tensor::Tensor;

let a = Tensor::from_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])?;
let b = Tensor::ones(vec![2, 3]);
let c = (&a + &b)?;
```

## Features

- **Data types**: f32, f16 (IEEE 754), bf16 (Brain Float)
- **Broadcasting**: automatic shape expansion following NumPy rules
- **SIMD**: runtime dispatch for NEON (aarch64) and SSE2/AVX (x86_64)
- **Aligned memory**: 32-byte aligned allocations for AVX
- **Operations**: arithmetic, matmul, transpose, reshape, slice, gather, scatter, reduce, clamp, pad, concat, split, topk, sort

## Optional Features

```toml
[features]
mkl = []      # Intel MKL BLAS backend
armpl = []    # Arm Performance Libraries backend
```

## Tests

207 tests covering shapes, broadcasting, dtypes, edge cases.
