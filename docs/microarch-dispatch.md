# Vision & Microarchitecture Dispatch

This document is the north star for how yscv's CPU compute evolves. It states
what the framework *is* and is *becoming*, then specifies the architecture that
gets us there: a per-microarchitecture kernel dispatch layer, built in
zero-regression phases. It is written for contributors and for AI agents that
need to know where new kernels go and why.

For how the framework is wired *today* (crate layers, the current SIMD dispatch,
threading), read [architecture.md](architecture.md) first — this document
extends its "SIMD dispatch model" section, it does not replace it.

## What yscv is — and aims to be

yscv is a **pure-Rust** ML and inference framework: one workspace, no external
compute backends. It does not link MKL, oneDNN, XNNPACK, OpenBLAS-as-a-hard-dep,
cuDNN, or any vendor kernel library to do its work — the kernels live *in the
crate* and are selected at runtime by what hardware is actually present.

The north star for the inference path:

> **A drop-in replacement for ONNX Runtime's CPU execution provider: load an
> ONNX model, call run, and get the best CPU path for the host automatically —
> no execution-provider wiring, no backend selection, no build-time target
> pinning. One crate, runtime auto-detect, maximum performance on every
> platform.**

| yscv **is** | yscv is **not** |
|---|---|
| Self-contained Rust kernels (intrinsics + selective hand asm) | A thin wrapper over a vendor kernel library |
| Runtime auto-detect of the optimal path (ISA **and** microarch) | A build that must be re-targeted per CPU |
| Edge-first: tuned hardest where in-order cores live | A single one-size path that ignores the core it runs on |
| Correct-and-fast *everywhere* via graceful fallback | Fast only on the one box it was measured on |
| Chasing **broad, validated** hardware coverage | Chasing a single benchmark number |

This is the same architecture MLAS (ORT-CPU's own backend), oneDNN, and XNNPACK
all converge on. Adopting it is not reinventing a wheel — it is building the
wheel the incumbents already proved is the right shape, inside one Rust crate.

## Where we are: ISA-level dispatch

Today the f32 hot path uses a **three-tier ISA dispatch** with runtime feature
detection (see [architecture.md](architecture.md)):

```
x86_64 → AVX-512 → AVX2/FMA → SSE → scalar
aarch64 → NEON → scalar
```

The fn-pointer table pattern already exists in the codebase for one kernel
family (`select_dw3_row_fn` / `Dw3RowFn` in `fused_pw_dw_3x3`) — the new layer
generalises that pattern to every hot kernel.

The kernels themselves are already, in places, **microarch-tuned** — the
Cortex-A53 work (the pipelined `yscv_sgemm_8x8_neon` at silicon peak, the
`dw5_creuse` column-reuse asm, `ld64` split loads, fmla-by-lane) is A53-specific
scheduling. It just lives under a blanket `cfg(target_arch = "aarch64")`, not
under a `cortex_a53` selector.

## The gap: no microarchitecture layer

Within an ISA there is exactly one path, tuned for whatever core it was last
measured on. That leaves performance on the table the moment the host is a
*different* core of the same ISA:

| Same code, different core | Why the optimum differs |
|---|---|
| Cortex-A53 vs **A55** | A55 has a 128-bit NEON datapath; the A53 `ld64` split-load trick is wasted, a different schedule wins. |
| Cortex-A53 vs **A72/A73/A76** | Out-of-order — the hardware reorders, so baked-in in-order scheduling is moot; tile sizes + ISA features (dotprod, i8mm) dominate. |
| Zen4 vs **Sapphire Rapids** | Zen4 double-pumps 512-bit FMA; SPR has native 512-bit units. Optimal unroll / tile / accumulator count differs. |
| Zen4 vs **Zen5** | Different FMA/load throughput → different MR×NR sweet spot. |

The size of the prize is **not uniform** — and knowing this is what keeps the
effort honest:

| Core class | Gain from per-microarch tuning | Reason |
|---|---|---|
| **In-order ARM** (A53/A55/A35/A7) | **30–50%+** | No scheduler — the instruction schedule must be hand-baked; the compiler can't recover it. This is where it pays most. |
| **OoO ARM** (A72/A73/A76/Neoverse) | 10–20% | Hardware reorders; wins come from tile sizes + feature selection, not schedule. |
| **x86** (Zen / Intel) | 5–15% | Deep OoO hides most schedule imperfection. The big lever — ISA selection — already exists. |

**Edge is where in-order cores live, so this layer is best aligned with an
edge-first framework.** On mainstream OoO x86 the marginal gain is real but
small, because ISA selection already captures most of it.

## Target architecture — four layers

```
┌─────────────────────────────────────────────────────────────┐
│ Layer 4 · Per-microarch kernels   (separate files per core)  │
│   ops/gemm/cortex_a53.rs  generic_neon.rs  avx512.rs  …       │
├─────────────────────────────────────────────────────────────┤
│ Layer 3 · Dispatch table          select_gemm(cpu) -> fn-ptr │
│   capability-first, microarch-second; resolved ONCE at start │
├─────────────────────────────────────────────────────────────┤
│ Layer 2 · Detector                arch/detect_{aarch64,x86}  │
│   MIDR / CPUID → Microarch + features, with a fallback chain  │
├─────────────────────────────────────────────────────────────┤
│ Layer 1 · Hardware identity       Cpu { uarch, features }    │
│   host_cpu() -> &'static Cpu   (OnceLock, single source)     │
└─────────────────────────────────────────────────────────────┘
```

**Layer 1 — Hardware identity (`arch/`).** The single source of truth, computed
once at startup and cached.

```rust
pub enum Microarch {
    // aarch64
    CortexA53, CortexA55, CortexA72, CortexA73, CortexA76, NeoverseN1, AppleM,
    // x86_64
    Zen2, Zen3, Zen4, Zen5, Skylake, IceLake, SapphireRapids,
    // fallbacks — never an error, always a correct path
    GenericAarch64, GenericX86, Scalar,
}

bitflags! { pub struct CpuFeatures {
    // aarch64
    NEON; DOTPROD; I8MM; FP16; SVE;
    // x86_64
    AVX2; FMA; AVX512F; AVX512VNNI; AMX;
} }

pub struct Cpu { pub uarch: Microarch, pub features: CpuFeatures }
pub fn host_cpu() -> &'static Cpu;   // OnceLock
```

**Layer 2 — Detector.** Identifies the exact core, never panics, always lands on
a usable value.

- **aarch64:** read MIDR (`CPU part 0xd03` = A53, `0xd05` = A55, `0xd08` = A72,
  `0xd0b` = A76, …) via a **fallback chain** — `mrs` (where the kernel emulates
  the ID register) → `/proc/cpuinfo` `CPU part` → macOS `sysctl` (Apple) →
  `GenericAarch64`. Features via `getauxval(AT_HWCAP)` /
  `is_aarch64_feature_detected!`.
- **x86_64:** `CPUID` — vendor + family/model → Zen/Intel microarch; features via
  `is_x86_feature_detected!`.

**Layer 3 — Dispatch table.** Per hot-kernel family, a selector returning a
fn-pointer. Two rules make it safe:

- **Capability-first, microarch-second.** Select by feature for *correctness*
  (never emit AVX-512 without `AVX512F`), then refine by microarch for
  *performance*. An unknown core with known features still gets a correct,
  feature-appropriate kernel.
- **Resolve once.** The selector runs at session / model-load time; the chosen
  fn-pointers are stored on the runner. The hot path never re-dispatches.

```rust
type GemmKernel = fn(&[f32], usize, usize, &[f32], usize, &mut [f32]);

pub fn select_gemm(cpu: &Cpu) -> GemmKernel {
    match cpu.uarch {
        Microarch::CortexA53                               => a53::gemm_8x8,
        _ if cpu.features.contains(CpuFeatures::AVX512F)   => avx512::gemm,
        _ if cpu.features.contains(CpuFeatures::NEON)      => generic_neon::gemm,
        _                                                  => scalar::gemm,
    }
}
```

**Layer 4 — Per-microarch kernels (separate files).** One file per (op,
microarch) so schedules don't collide in a single mega-file:

```
ops/gemm/
  mod.rs          # select_gemm + the table
  cortex_a53.rs   # the pipelined 8×8 + ld64 we already have
  generic_neon.rs # clean NEON fallback (any aarch64)
  avx512.rs  avx2.rs  scalar.rs
  cortex_a55.rs   # added later, when A55 silicon is in hand
```

## Design principles

1. **Never fail to run.** Unknown microarch → `Generic*`; unknown features →
   scalar. Correctness is independent of whether we recognise the core.
2. **Capability-first selection.** Features gate correctness; microarch only
   refines the performance pick on top of an already-correct choice.
3. **Resolve once, call many.** Dispatch cost is paid at startup, never per
   call — a fn-pointer indirect branch is free against a GEMM/conv tile.
4. **Separate files per microarch.** Schedules are easier to read, tune, and
   diff when they are not interleaved by `cfg`/`match` in one file.
5. **Bit-identical regression gate.** Every refactor step keeps the existing
   `*_matches_scalar` / `*_matches_nhwc` parity tests green; the tracker output
   (882/894) stays bitwise stable.
6. **Validation-gated specialisation.** A microarch kernel is only added once it
   is *measured* on that silicon. No blind tuning — a wrong schedule can be
   slower than the generic path.
7. **Share where the schedule is identical (BLIS-style).** Where two cores only
   differ in parameters (tile size, prefetch distance, unroll), share one
   parameterised implementation rather than copy a whole kernel. Separate files
   are for genuinely different schedules, not for duplicated code.

## Roadmap — zero-regression phases

| Phase | Work | Status |
|---|---|---|
| **0 ✓** | `arch/` module: `Microarch`, `CpuFeatures`, `Cpu`, `host_cpu()` + detector (aarch64 MIDR chain, x86 CPUID). No kernel changes. | **DONE.** Detects `Zen4` (dev) and `CortexA53` (Orange Pi, MIDR `0x…d034`, with correct `neon`-only feature set); 0 perf change. |
| **1 ✓** | Route one op family's dispatch through `host_cpu()`. (Did the fused PW/DW selectors — `select_variant` / `select_dw5_variant` / `select_tile_variant` — reading `cpu.features` instead of ad-hoc `is_*_feature_detected!`, with an `is_in_order()` hook comment for the future asm split.) | **DONE.** Inert (`cpu.features.x` == the cached macro result); 358 tests, `CortexA53` confirmed in the live aarch64 binary. |
| **2 ✓** | Extend `host_cpu()` routing across the rest of the hot path (GEMM/matmul, conv, first-layer, single-op SIMD, int8 selectors). | **DONE.** Hot-path feature gates now read the cached `host_cpu().features` source instead of ad-hoc raw feature probes. CI guards raw `is_*_feature_detected!` calls so new probes stay inside `arch/detect_*`, and benchmark logs print the selected dispatch paths for reproducibility. |
| **3+** | Add per-microarch kernel **variants** — the actual performance work. | **Needs the board.** Per-board before/after, net-positive defaults-on. See the backlog below. |

**Where we are.** Phases 0–2 are done and proven: the framework now *knows* its
core (`host_cpu()`), hot-path feature gates route through that cached identity,
and the benchmark harnesses report the resolved single-op paths. That is the
hard, valuable part — the foundation. Phase 3 is the real win and is **gated on
hardware**, because a microarch kernel cannot be tuned without that silicon to
measure on.

### Phase 3 backlog (per board)

Concrete first targets, each gated on having the core. These are where the single
A53-tuned NEON path is most likely leaving performance on the table:

| Core (board) | Kernels | Technique vs the current A53 path | Why it should differ |
|---|---|---|---|
| **Cortex-A55** (Allwinner H616/H618 A55 SKUs, RK3566/3568) | GEMM, DW | `ld128` wide loads + a wider schedule; drop the A53 `ld64` split | A55 has a **128-bit** NEON datapath (A53 is 64-bit) — the `ld64` integer-pipe trick is wasted and the schedule under-feeds the FMA units. |
| **Cortex-A72 / A73** (RPi4 = A72) | DW, GEMM | Select the plain NEON intrinsic (drop the column-reuse `dw5_creuse`/`dw3s2_creuse` asm) for `!is_in_order()` | Out-of-order — the asm's hand-baked in-order interleave is moot, and its wasted-FMA edge cases (KY-zero taps) may cost more than the hardware-reordered intrinsic. |
| **Cortex-A76** (RPi5) | GEMM, int8 | Bigger OoO tiles; `dotprod`/`i8mm` GEMM for int8 | Wider machine + ARMv8.2 features (`dotprod`/`i8mm`) the A53 lacks; the feature flags already gate this correctly. |
| **Zen4 vs Intel** (have Zen4) | AVX-512 GEMM/PW | Re-validate the existing opt-in `MR16` / `KCBLOCK` AVX-512 variants as a `Zen4`-default | Zen4 double-pumps 512-bit FMA; fewer-accumulator tiles may suit it. (Were tracker-flat before — re-measure under the new dispatch.) |

Each entry: add the variant file, wire it as a `match cpu.uarch` arm in the
selector (the hook is already commented in `select_variant`), A/B on the board,
keep only net-positive, document the rest.

## Scope & non-goals

- **Not a 1000-kernel port of XNNPACK.** We build the *dispatch infrastructure*
  plus a *handful* of validated kernels, not the entire microkernel zoo. Most of
  the per-microarch benefit comes from the table + selective tuning.
- **Not the lever for beating XNNPACK on the A53.** On A53 our GEMM is already at
  the 64-bit-NEON silicon peak (≈6.3 GF/s, matched to XNNPACK and OpenBLAS); the
  residual gap there is memory bandwidth, not the kernel. This layer is a
  **coverage** play (run well on *all* edge cores), not a win on the one box.
- **Needs the silicon.** Per-microarch kernels cannot be tuned blind. Each new
  core variant requires that board (A53 in hand; A72≈RPi4, A76≈RPi5,
  Neoverse≈Graviton) or a cycle-accurate model.

## Where this lives in the code

```
crates/yscv-kernels/src/
  arch/
    mod.rs            # Microarch, CpuFeatures, Cpu, host_cpu()
    detect_aarch64.rs # MIDR / cpuinfo / sysctl fallback chain
    detect_x86.rs     # CPUID vendor + family/model
  ops/
    gemm/             # Phase 1 first mover
      mod.rs          # select_gemm + table
      cortex_a53.rs  generic_neon.rs  avx512.rs  avx2.rs  scalar.rs
    depthwise/ pointwise/ …   # follow the same shape in Phase 3+
```

The existing `select_dw3_row_fn` is the precedent; the `arch/` layer makes that
selection *hardware-aware* and generalises it across kernels.
