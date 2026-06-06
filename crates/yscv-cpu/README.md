# yscv-cpu

Cached host CPU identity and runtime feature detection for yscv.

This crate is the single source of truth for CPU dispatch across the
workspace. It detects the host once, caches the result, and exposes:

- `Microarch` - coarse CPU family such as `Zen4`, `CortexA53`, or
  `GenericX86`.
- `CpuFeatures` - runtime ISA features such as `avx2`, `avx512f`,
  `neon`, `dotprod`, and `i8mm`.
- `host_cpu()` - returns the cached `Cpu { uarch, features }`.

Raw `is_*_feature_detected!` calls belong only in `src/detect_*`.
Other crates use `yscv_cpu::host_cpu().features` or a crate-local
re-export, and `scripts/check-runtime-dispatch.sh` enforces that rule.

```rust
let cpu = yscv_cpu::host_cpu();
println!("{:?} {:?}", cpu.uarch, cpu.features);
```
