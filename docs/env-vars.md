# Environment variables — complete reference

Canonical list of every environment variable yscv reads, with defaults and
scope. Cargo feature flags (build-time) are a separate axis — see
[`feature-flags.md`](feature-flags.md).

Most entries here are **A/B and rollback knobs**, not production tuning. The
defaults are what the tracker and bench suites are tuned against; changing them
is for measurement, bisecting a regression, or working around a bad dispatch on
an untested microarch. A handful (marked **user-facing**) are ordinary
configuration.

---

## How values are parsed

This bites people, so read it first. Three different conventions are in use:

| Convention | Meaning | Example |
|---|---|---|
| **presence** | *Any* value enables — including `=0` and the empty string | `YSCV_REORDER_FUSION_OFF=0` still **disables** the pass |
| **exact value** | Only the listed value has an effect | `YSCV_POOL=yscv`, `YSCV_MR6=1` |
| **numeric** | Parsed as an integer; unparseable or out-of-range falls back to the default | `YSCV_KCBLOCK_KC=256` |

Unless a row says otherwise, the variable is **presence**-checked. The
value-checked ones are called out explicitly in the tables below.

Almost every knob is read **once per process** and cached in a `OnceLock` — the
dispatch sites are hot enough that a per-call `env::var` would show up in
profiles. Setting a variable from inside a running process (`std::env::set_var`)
after the first inference has no effect.

## Seeing what is active

`yscv_kernels::runtime_config_report()` dumps every `YSCV_*` variable currently
set, in stable name order, and is included in
`runtime_dispatch_report()`. Use it to confirm a knob actually reached the
process:

```rust
println!("{}", yscv_kernels::runtime_config_report());
// YSCV_POOL=yscv, YSCV_POOL_SPIN_US=200
```

Note this only covers the `YSCV_` prefix. The Metal, wgpu and trace variables in
[Apple Metal](#apple-metal--mpsgraph-yscv-onnx-metal-backend) and
[Debugging and tracing](#debugging-and-tracing) use unprefixed names and will
not appear.

---

## General (user-facing)

| Variable | Default | Effect |
|---|---|---|
| `YSCV_CACHE_DIR=<path>` | `$HOME/.yscv/models` | Cache directory for model weights downloaded by `yscv-model`'s hub. Falls back to `./.yscv/models` when `$HOME` is unset. |
| `RAYON_NUM_THREADS=<N>` | rayon default (logical cores) | Standard rayon control. Also sizes the yscv pool when `YSCV_POOL=yscv`. |

## Thread pool and parallelism (`yscv-threadpool`, `yscv-onnx`)

| Variable | Default | Effect |
|---|---|---|
| `YSCV_POOL=rayon\|yscv` | `rayon` | **Exact value.** `yscv` routes `par_chunks_mut_dispatch` kernels through the pinned yscv pool instead of rayon, and lowers the softmax parallel threshold. Useful for single-op benches. |
| `YSCV_POOL_AFFINITY=none\|big\|physical` | `physical` | **Exact value.** Worker pinning policy for the yscv pool. `physical` = one worker per physical core, SMT sibling left free for the submitter. `big` = P-cores / big cores only, falling back to `physical` on symmetric CPUs. Any unrecognised value means `physical`. |
| `YSCV_POOL_SPIN_US=<N>` | `0` | Idle spin (µs) before a yscv-pool worker parks. `~200` can remove park/unpark overhead in tight single-op microbench loops; higher values regress normal inference graphs. |
| `YSCV_POOL_SLEEPY_ROUNDS=<N>` | `32` | Steal rounds a yscv-pool worker makes in the "sleepy" state before sleeping. Raise for coarse workloads whose dispatches land further apart than the current window. |
| `YSCV_SESSION_POOL=1` | unset | **Exact value `1`.** Route inference through `PersistentSection::parallel_for` instead of rayon fork-join. Default off: the section's `dispatch_busy` spin-lock costs more than rayon work-stealing on this workload. Kept for A/B on other microarchs. |
| `YSCV_ALLOW_SMT=1` | unset | Let `OnnxRunner::with_threads(N)` exceed the physical core count. Default caps at physical cores. |
| `YSCV_ST_INLINE=1` | unset | In single-thread mode, run inference inline on the caller thread instead of a 1-thread rayon pool. Only helps when combined with `RAYON_NUM_THREADS=1` **and** caller pinning (`taskset`); without pinning the lone thread still migrates, which is why it is opt-in. |
| `YSCV_NO_SCOPE_SEQ_FASTPATH=1` | unset (fast path **on**) | Disable the sequential fast path in `install_scope` dispatch. |

### Tower-parallel execution (`yscv-onnx`)

Independent graph branches ("towers") run in parallel above a thread-count gate.

| Variable | Default | Effect |
|---|---|---|
| `YSCV_NO_TOWER_PARALLEL=1` | unset | Force tower-parallel execution **off**. Takes precedence over the force-on knob. |
| `YSCV_FORCE_TOWER_PARALLEL=1` | unset | **Exact value `1`.** Force tower-parallel execution **on**, bypassing the thread gate. |
| `YSCV_TOWER_MIN_THREADS=<N>` | `2` | Minimum thread count at which tower-parallel turns on by default. |

## ONNX graph optimizer and loader (`yscv-onnx`)

| Variable | Default | Effect |
|---|---|---|
| `YSCV_REORDER_FUSION_OFF=1` | unset (reorder **on**) | Skip the node-reorder pass that restores producer/consumer adjacency before the fusion passes. The reorder only permutes independent nodes, so outputs are unchanged either way — disabling it just costs fusions on models whose export order interleaves branches. |
| `YSCV_NCHWC=on` | unset | **Exact value `on`.** Print NCHWc capability stats (`capable`/`chains`/`max_chain`/`mean_chain`) to stderr after optimization. Diagnostic only; does not change execution. |
| `YSCV_NO_PREPACKED_BY_ID=1` | unset | Disable the prepacked-weight-by-id lookup for Conv nodes, forcing a repack per call. |
| `YSCV_RESHAPE_NHWC_PASSTHROUGH_OFF=1` | unset (passthrough **on**) | Disable the fast path that skips the `ensure_nchw` permute before a rank-4 NHWC `Reshape`. |
| `YSCV_FUSED_PW_DW_PW_REDUCE_OFF=1` | unset (fusion **on**) | Keep PW-expand → DW → PW-reduce as separate `FusedPwDw` + Conv actions instead of the full-block fusion that holds the expanded intermediate in cache. |
| `YSCV_FUSED_PW_DW_PW_REDUCE_DEBUG=1` | unset | Print per-node decisions from the PW→DW→PW-reduce fusion walk at load time. |

## Conv routing (`yscv-onnx` runner)

| Variable | Default | Effect |
|---|---|---|
| `YSCV_INDIRECT_MAX_COUT=<N>` | `0` (route off) | Output-channel ceiling below which Conv uses the indirect-buffer path instead of blocked-GEMM im2col. The indirect path re-gathers per (tap, in-channel), so it loses at non-trivial channel counts — hence off by default. |
| `YSCV_NCHWC_DW=1` | unset | Opt into the NCHWc depthwise route for 3×3 stride-1 SAME-pad DW convs with `C % 8 == 0`. |
| `YSCV_BNNS=1` | unset | **macOS only.** Use Apple Accelerate BNNS directly for NCHW-input Conv, skipping layout conversion. |
| `YSCV_BACKBONE_NCHWC=1` | unset | Keep backbone chains in NCHWc between eligible ops (`c_out ≥ 16` and a multiple of 16, chain-eligible), avoiding per-op conversions. |
| `YSCV_NCHWC_STREAM=1` | unset | **x86_64 only.** Hold the streaming ring in NCHWc-blocked layout `[3, c_blocks, w, 16]`. Microbench-faster but tracker-neutral, so opt-in. |

## Fused conv streaming paths

The DW→PW and PW→DW streaming fusions are **enabled by default on x86_64 and
disabled by default on aarch64** — they regress the small-L2 Cortex-A53. That is
why each has both an `_ON` and an `_OFF` knob.

| Variable | Default | Effect |
|---|---|---|
| `YSCV_FUSED_DW_PW_STREAM_OFF=1` | unset | Disable the `FusedDwPw` streaming path everywhere. |
| `YSCV_FUSED_DW_PW_STREAM_ON=1` | unset | **aarch64 only.** Opt the `FusedDwPw` streaming path back in (no effect elsewhere — it is already on). |
| `YSCV_FUSED_DW_PW_STREAM_PADDED=1` | unset (**off on all targets**) | Enable the padded `FusedDwPw` streaming variant. Currently slower than the non-stream fused route pending a tuned padded schedule. |
| `YSCV_FUSED_PW_DW_STREAM_OFF=1` | unset | Disable the `FusedPwDw` streaming path; forces the fallback compute chain. |
| `YSCV_FUSED_PW_DW_STREAM_ON=1` | unset | **aarch64 only.** Opt the `FusedPwDw` streaming path back in. |
| `YSCV_FUSED_PW_DW_5X5_OFF=1` | unset | Disable the 5×5 variant of the `FusedPwDw` streaming path. |
| `YSCV_FUSED_DW_PW_NCHWC=1` | unset | Enable the NCHWc DW+PW pair route. |
| `YSCV_FUSED_DW_PW_NCHWC_AVX512=1` | unset | Enable the AVX-512 NCHWc pair route. Decoupled from DW streamability: the rewritten `conv2d_nchwc_dw3x3_s1_same_pad` handles SAME-pad natively. |
| `YSCV_FUSED_DW_PW_TRUE_FUSED_ON=1` | unset | Force the true-fused depth-multiplier-1 path on. |
| `YSCV_FUSED_DW_PW_TRUE_FUSED_OFF=1` | unset | Force it off. Wins over `_ON` and `_AUTO`. |
| `YSCV_FUSED_DW_PW_TRUE_FUSED_AUTO=1` | unset | Let the heuristic decide per shape instead of the fixed default. |
| `YSCV_FUSED_DW_PW_ROW_BATCH=<N>` | shape-derived | Override the `FusedDwPw` streaming y-band size. Must be `> 0`. |

## Fused PW→DW microkernels (`yscv-kernels`)

| Variable | Default | Effect |
|---|---|---|
| `YSCV_FUSED_PW_DW_W_TILE=<N>` | auto | Strip-mining tile width in `fused_pw_dw_3x3`. Values `< 4` are ignored. |
| `YSCV_FUSED_PW_DW_5X5_W_TILE=<N>` | auto | Same, for the 5×5 path. Values `< 4` ignored. |
| `YSCV_FUSED_PW_DW_5X5_OC_TILE=<N>` | auto (from `c_exp`) | Output-channel tile for the 5×5 path. Values `< 16` ignored. |
| `YSCV_FUSED_PW_DW_PW2X_OFF=1` | unset (PW2X **on**) | **aarch64.** Disable the NEON two-column PW inner-loop variant. |
| `YSCV_FUSED_PW_DW_4X6_OFF=1` | unset (4×6 **on**) | **AVX-512.** Disable the 4×6 OC×OW PW tile, forcing the older 6-ZMM register-blocked path. |
| `YSCV_FUSED_PW_DW_TILED4X6_OFF=1` | unset | **AVX-512.** Disable the tiled 4×6 variant. |
| `YSCV_FUSED_PW_DW_FULLROW_OFF=1` | unset | **AVX-512.** Disable the full-row PW variant. |
| `YSCV_FUSED_PW_DW_DW_INTERIOR=1` | unset | **AVX-512, experimental.** Opt into the stride-1/stride-2 depthwise interior fast path inside PW→DW fusion. |
| `YSCV_DW5_REUSE_OFF=1` | unset (reuse **on**) | **AVX-512.** Disable DW 5×5 filter-tap reuse (the MLAS strategy: ~1.2 loads/FMA instead of 2), reverting to the 1-pixel tiled body. |
| `YSCV_DW5_ASM_OFF=1` | unset | **aarch64.** Disable the DW 5×5 assembly kernel. |
| `YSCV_PW_GEMM_OFF=1` | unset | **aarch64.** Disable the pipelined 8×8 GEMM PW path (cached B-pack), reverting to the broadcast path. Bit-identical results either way. |
| `YSCV_PW_GEMM_MAX_THREADS=<N>` | `4` | **aarch64.** Thread ceiling for the PW GEMM path. |
| `YSCV_PW_PREFETCH_OFF=1` | unset (prefetch **on**) | Disable the software weight-cacheline prefetch in the AVX-512 and NEON pointwise kernels. The weight stride (`n*4` B) is beyond the HW prefetcher, so this normally costs performance. |
| `YSCV_DW_PROFILE=1` | unset | Accumulate and report internal PW / DW / PW-reduce nanosecond counters. |

## Conv kernels (`yscv-kernels`)

| Variable | Default | Effect |
|---|---|---|
| `YSCV_DIRECT_CONV_WORK_MAX=<N>` | arch/thread auto | Work threshold for the direct 3×3 conv path. Must be `> 0`. |
| `YSCV_AVX512_DW_OFF=1` | unset | Disable the AVX-512 depthwise row kernel. |
| `YSCV_DW3X3_C16_OFF=1` | unset | **x86_64.** Disable the `c=16` depthwise 3×3 fast path. |
| `YSCV_NO_FIRST_LAYER_KERNEL=1` | unset | Disable the register-blocked 3-channel 3×3 stride-2 first-layer kernel; falls back to generic GEMM conv. |
| `YSCV_FIRST_LAYER_PAR_OFF=1` | unset | Force the first-layer kernel sequential, for A/B and worst-case profiling. |
| `YSCV_FIRST_LAYER_AVX512_OFF=1` | unset | Disable the AVX-512 first-layer variant. |
| `YSCV_FIRST_BYLANE_OFF=1` | unset (by-lane **on**) | **aarch64.** Revert the first-layer kernel to `vdupq_n` broadcast instead of `fmla`-by-lane. By-lane keeps the NEON pipe free for FMAs. |
| `YSCV_WINO_MIN_CH=<N>` | `1` on x86; `usize::MAX` (off) on non-macOS aarch64 | Minimum channel count for the Winograd conv path. Off by default on aarch64 — measured no end-to-end gain over blocked-GEMM im2col. Not read on macOS (Accelerate AMX sgemm). |
| `YSCV_CONV_MIN_PARALLEL=<N>` | `4096` | Minimum element count before conv goes parallel. |
| `YSCV_DEPTHWISE_MIN_PARALLEL=<N>` | `4096` | Same, for depthwise conv. |
| `YSCV_SEPARABLE_MIN_PARALLEL=<N>` | `4096` | Same, for separable conv. |
| `YSCV_MIN_PAR_POINTWISE_CONV_ELEMS=<N>` | `16384`; `8192` on aarch64 | Element threshold for parallel pointwise conv dispatch. Must be `> 0`. A53/A55-class cores benefit from entering parallel earlier, hence the halved aarch64 default. |
| `YSCV_MIN_PAR_POINTWISE_CONV_FLOPS=<N>` | `1500000`; `750000` on aarch64 | FLOP threshold (`work = 2*m*k*n`) for the same dispatch. Must be `> 0`. |
| `YSCV_LAYOUT_PAR_OFF=1` | unset (parallel **on**) | Force the NHWC→NCHW transpose serial per-batch instead of parallel over output-channel blocks. |

### Pointwise (1×1) conv

| Variable | Default | Effect |
|---|---|---|
| `YSCV_NO_POINTWISE_16X16_DIRECT=1` | unset | Disable the direct NHWC 1×1 `K=16,N=16` kernel with fused bias/residual/activation epilogue. |
| `YSCV_NO_POINTWISE_NX16_DIRECT=1` | unset | Disable the single-thread direct NHWC residual pointwise kernel for small-`M`, `N % 16 == 0` ConvAdd blocks. |
| `YSCV_POINTWISE_NX16_DIRECT_ON=1` | unset | **aarch64 only.** Opt the nx16 direct kernel back in — it is off by default there (measured ~+190 ms/inf vs the packed blocked-GEMM path on a Cortex-A53 tracker). |
| `YSCV_NX16_MT_OFF=1` | unset | Disable multi-threaded dispatch for the nx16 direct kernel. |
| `YSCV_KCBLOCK=1` | unset | **x86.** Opt into the K-cache-blocked AVX-512 nx16 variant. Tracker-flat; useful on K-heavy `Conv_Add` shapes. |
| `YSCV_KCBLOCK_KC=<N>` | `128` | K chunk size for that variant; clamped to `8..=1024`. KC=128 keeps 8 KB of weight per oc_block L1-resident. |
| `YSCV_MR16=1` | unset | **x86.** Opt into the 16-row MR AVX-512 nx16 variant. Tracker-flat. |
| `YSCV_REDUCE_GEMM_ON=1` | unset | Route reduce / standalone-PW shapes through the blocked GEMM. Off by default: at `c_out` 24–112 it loses to the weight-stationary nx16 direct kernel (−2 ms/1T, −4.5 ms/2T). |
| `YSCV_REDUCE_BYLANE_OFF=1` | unset (by-lane **on**) | **aarch64.** Revert the NEON reduce kernel to per-activation broadcast (`ld1r`) instead of `fmla`-by-lane. |

### NCHWc kernels

| Variable | Default | Effect |
|---|---|---|
| `YSCV_NCHWC_PW_DIRECT_OFF=1` | unset | Disable the direct NCHWc PW AVX-512 4-pixel × 16-channel kernel; falls back to blocked-GEMM NCHWc. |
| `YSCV_NCHWC_PW_LEGACY=1` | unset (native **on**) | Revert to the legacy NCHWc pointwise path. |
| `YSCV_NCHWC_DW3X3_NOPAD_OFF=1` | unset | **x86.** Disable the no-pad NCHWc DW 3×3 kernel, reverting to the legacy path. |

## MatMul / GEMM (`yscv-kernels`)

| Variable | Default | Effect |
|---|---|---|
| `YSCV_NO_AVX512=1` | unset | **x86_64 Linux/macOS.** Force the AVX-512F dispatch off regardless of CPUID. |
| `YSCV_AVX512_SGEMM=1` | unset | **Exact value `1`.** Enable the AVX-512 MR=12×NR=32 GEMM kernel. Off by default because Zen 4 double-pumps ZMM (1 µop/clk/unit) while AVX2 4×24 runs 2 YMM FMAs/clk; expected to win on Intel and Zen 5 true-512 silicon. |
| `YSCV_AVX512_RELU=1` | unset | Enable the AVX-512 `Conv_Relu` path. Same Zen 4 reasoning as above. |
| `YSCV_ASM_GEMM=1` | unset | **Exact value `1`.** Enable the 4×24 AVX2 pure-`.S` kernel (requires AVX+FMA). |
| `YSCV_MR6=1` | unset | **Exact value `1`.** Enable the x86 AVX2 MR=6×16 tile (requires AVX+FMA). Kept for a future asm 6×16 rewrite and for microarchs with different register pressure. |
| `YSCV_NO_MR8=1` | unset (MR8 **on**) | **aarch64.** Disable the NEON MR=8 / 8×12 fast path. |
| `YSCV_MR8_RESIDUAL=1` | unset | **Exact value `1`, aarch64.** Enable the compact residual post-pass on full MR8 tiles at `is_last_k`. |
| `YSCV_NO_MR8_TAIL8_ASM=1` | unset (tail-8 asm **on**) | **aarch64.** Disable the MR8 tail-8 asm path (`nr == 8` in the NR=12 kernel), reverting to the scalar tail. |
| `YSCV_GEMM_8X8_OFF=1` | unset | **aarch64.** Disable the blocked 8×8 GEMM. |
| `YSCV_NEON_4X24_ON=1` | unset | **aarch64.** Opt into the 4×24 tile. Off by default: it needs 34 vector registers and aarch64 has 32, so it spills every k-iteration. |
| `YSCV_AVX512_ROWGEMM_OFF=1` | unset | Disable the AVX-512 row-set GEMM kernel. |
| `YSCV_AVX512_ROWGEMM_MIN_DIM=<N>` | `4096` | Minimum `k` or `n` before the AVX-512 row-set kernel is preferred over the AVX FMA 6×8 tile. Below this, 6×8 wins — more accumulator chains hide latency at small K, and it has SW prefetch the row-set kernel lacks. |
| `YSCV_LOW_K_TILE=1` | unset | **x86.** Opt into the specialized low-k pointwise tile. |
| `YSCV_NO_X86_LOW_K_BLOCKED=1` | unset (route **on**) | Disable the x86 low-k pointwise route through the blocked 4x24/4x16 AVX+FMA kernels. |
| `YSCV_NO_AARCH64_LOW_K_BLOCKED=1` | unset (route **on**) | Disable the aarch64 low-k blocked matmul route. |
| `YSCV_AARCH64_LOW_K_BLOCKED_MIN_WORK_FMAS=<N>` | `1048576` | `m*k*n` work threshold for that route. Must be `> 0`. |
| `YSCV_NO_AARCH64_RESIDUAL_BLOCKED=1` | unset (**on**) | Disable the blocked residual epilogue on aarch64 NEON matmul kernels. |
| `YSCV_TRANS_A_DIRECT_OFF=1` | unset | Disable the direct transposed-A `FusedTransposeMatMul` path. |
| `YSCV_NON_TRANS_4ROW_OFF=1` | unset | Disable the non-transposed 4-row × NR=16 tile. |
| `YSCV_FTMM_4ROW_OFF=1` | unset | Disable the `FusedTransposeMatMul` 4-row tile. |
| `YSCV_X86_MEMORY_SIMD=avx2` | unset | **Exact value.** On x86 with AVX-512 available, route memory-bound standalone elementwise/ReLU dispatch through the 256-bit AVX path for A/B. |

### BLAS routing

These override the `blas` Cargo feature **at runtime** — worth knowing when a
build appears to ignore its feature flags.

| Variable | Default | Effect |
|---|---|---|
| `YSCV_FORCE_NO_BLAS=1` | unset | Never call BLAS; always use the custom GEMM. Wins over `YSCV_FORCE_BLAS`. |
| `YSCV_FORCE_BLAS=1` | unset | Always prefer BLAS over the custom GEMM, bypassing the size heuristic. |

## Quantization

| Variable | Default | Effect |
|---|---|---|
| `YSCV_QUANT_INT8_FAST=0` | unset (**on**) | **Value-checked (`!= "0"`).** Setting `0` disables internal quant-domain boundary folding while keeping standard QLinear kernels — a true bitwise reference for the fused chain. |
| `YSCV_QUANT_FAST=0` | unset (**on**) | **Value-checked (`!= "0"`), `quantize_tracker` tool.** Setting `0` skips constant-QDQ weight folding, fusion-chain QDQ stripping, and initializer pruning when emitting QDQ format. |

## Profiling

Requires the `profile` Cargo feature where noted. See
[`feature-flags.md`](feature-flags.md#profile--per-op-onnx-profiling).

| Variable | Default | Effect |
|---|---|---|
| `YSCV_PROFILE=1` | unset | Accumulate per-op Conv / non-Conv timings and counts during execution. |
| `YSCV_PROFILE_FILTER=<spec>` | unset | Narrow the detail table and JSON to a node subset. Comma-separated; a bare token matches an op type (`Conv`, `Conv,MatMul`), `name:<substr>` matches a node-name substring. |
| `YSCV_PROFILE_JSON=<path>` | unset | Also write machine-readable per-node JSON (name/op/ms/shapes; Conv nodes carry `kernel_shape`, `strides`, dispatched `kernel`). Consumed by `scripts/gap_diff.py`. |
| `YSCV_RUNNER_PROFILE=<path>` | unset | Aggregate per-node timings over the *fused* runner path across a bench loop; dump with `dump_runner_profile`. Fused streaming kernels bypass instrumented dispatch and carry no kernel label. |

## Debugging and tracing

| Variable | Default | Effect |
|---|---|---|
| `YSCV_TRACE_SHAPES=1` | unset | **Value-checked** (set and `!= "0"`). Trace per-node layout kind and shapes during execution. |
| `CPU_TRACE=1` | unset | Dump every node's output tensor after execution, for bisecting inference divergence. **Unprefixed name.** |

## GPU — wgpu backend (`yscv-kernels`, `gpu` feature)

| Variable | Default | Effect |
|---|---|---|
| `YSCV_GPU_FP32=1` | unset (f16 when supported) | Force f32 compute instead of `SHADER_F16`. fp16 carries 1–2 % drift on Conv-heavy graphs — invisible after detection NMS, but it matters for bit-level checks against CPU/ORT. |
| `FORCE_F32_CONV=1` | unset | Force the f32 `conv_gemm` pipeline for conv dispatch. **Unprefixed name.** |
| `USE_V2_CONV=1` | unset | Select the v2 conv pipeline when `oc > 48`. **Unprefixed name.** |

## Apple Metal / MPSGraph (`yscv-onnx`, metal backend)

All of these use **unprefixed `METAL_*` names** and most require the `profile`
Cargo feature. They will not show up in `runtime_config_report()`.

| Variable | Default | Effect |
|---|---|---|
| `YSCV_MPS_PIPELINE=1` | unset | Enable the MPSGraph pipeline. See [`mpsgraph-guide.md`](mpsgraph-guide.md). |
| `METAL_DEBUG=1` | unset | General Metal compile/record tracing, including softmax shape decisions and in-place aliasing counts. |
| `MPSGRAPH_DEBUG=1` | unset | Trace MPSGraph op construction during graph build. |
| `METAL_COMPARE=1` | unset | Keep the CPU reference pre-pass and compare every Metal buffer against it. Also forces the CPU shape-discovery walk that is otherwise skipped for fully-known graphs. |
| `METAL_NAN=1` | unset | Scan each op's output buffer to find the first producing NaN. |
| `METAL_NO_CONV_CONCAT=1` | unset | Disable writing conv output interleaved directly into a pre-allocated concat buffer (which otherwise eliminates the concat copy). |
| `METAL_NO_FUSION=1` | unset | Disable the detection-head `CpuReshape` + `FlatConcat` → `NhwcToFlatConcat` fusion. |
| `METAL_NO_INPLACE=1` | unset | Disable aliasing an elementwise op's output onto a dead input buffer. |
| `METAL_NO_WINO=1` | unset | Disable the Winograd path for 3×3 stride-1 convs (4× FLOP reduction, stays in unified memory). |
| `METAL_MPS=1` | unset | Use MPS GEMM for non-depthwise, non-Winograd conv instead of the `ConvGemm` compute shaders. Known precision issues at non-aligned column counts (e.g. `K=27` for a first 3×3 stride-2 conv). |
| `METAL_LARGE_1X1=1` | unset | Select the large-tile variant for f16 1×1 convs. |
| `METAL_TIME=1` | unset | Print upload / encode / GPU milliseconds per run. |
| `METAL_TIMING=1` | unset | Print upload / encode / GPU / total milliseconds (separate call site from `METAL_TIME`). |
| `METAL_TIMING2=1` | unset | Print total wall time including output readback. |
| `METAL_PROFILE=1` | unset | Encode each op type in isolation to measure per-op-type GPU time. |
| `METAL_DEBUG_DL=1` | unset | Trace output download from the f32 readback buffers. |
| `METAL_CONV_SIZES=1` | unset | Print `M`/`K`/`N` and MFLOPs per conv (1×1, 3×3, Winograd) during planning. |
| `METAL_SHAPE_DBG=1` | unset | Print output shapes for DFL-path nodes whose name contains `model.22`. |
| `METAL_PERM_DBG=1` | unset | Print NHWC permutation name and shape. |

## Benchmarks and examples

Not part of the library surface; listed so a grep for a name lands somewhere.

| Variable | Default | Effect |
|---|---|---|
| `YSCV_ONNX_MODEL_BENCH_ASSET_DIR=<path>` | `<manifest>/target/assets` | Where the ONNX model bench suite looks for downloaded assets. |
| `BENCH_COOLDOWN=<secs>` | `0` | Cooldown pause between benchmarks in `bench_yolo`, for thermal control. **Unprefixed name.** |
| `YSCV_SGEMM_AB_ITERS=<N>` | example default | Iteration count for the `sgemm_ab` example. |
| `YSCV_SGEMM_AB_WARMUP=<N>` | example default | Warmup count for the same. |
| `YSCV_AVX512_BENCH=1` | unset | Run the `#[ignore]`d MR=12×NR=32 vs MR=4×NR=24 single-tile bench. |
| `YSCV_DW_NCHWC_BENCH=1` | unset | Run the `#[ignore]`d NCHWc DW 3×3 micro-bench. |
| `DUMP_RAW=1` | unset | Write the raw output tensor to `/tmp/yolo_vis/yscv_raw.bin` in the `yolo_detect` example, for ORT comparison. **Unprefixed name.** |

## CI and benchmark scripts

Read by `scripts/` and `benchmarks/`, not by any crate.

| Variable | Default | Effect |
|---|---|---|
| `YSCV_ROOT=<path>` | repo root from `$BASH_SOURCE` | Repo root override for `bench_tracker_quant_matrix.sh`. |
| `YSCV_PR_BENCH_SUITE=<path>` | `<runner_dir>/suite.json` | Suite definition for the ONNX model bench runner and asset download. |
| `YSCV_PR_BENCH_RUNS=<N>` | suite `defaults.runs`, else `3` | Runs per model. |
| `YSCV_PR_BENCH_THREADS=<N>` | suite `defaults.threads`, else `0` | Thread count (`0` = auto). |
| `YSCV_CRITERION_GATE_ATTEMPTS=<N>` | `2` | Retry attempts in `run-criterion-gate.sh`. Must be an integer `>= 1`. |
| `YSCV_TREND_BASELINE_ENFORCE=1` | `0` | Fail rather than warn when the trend baseline is stale. |
| `YSCV_TREND_BASELINE_MAX_AGE_DAYS=<N>` | `14` | Age at which the trend baseline counts as stale. |
| `YSCV_TREND_MAX_REGRESSION_PCT_MICRO=<pct>` | unset (no gate) | Max tolerated micro-benchmark regression. Non-negative number. |
| `YSCV_TREND_MAX_REGRESSION_PCT_RUNTIME=<pct>` | unset (no gate) | Max tolerated runtime-benchmark regression. Non-negative number. |

## Build-time (`build.rs`)

Standard Cargo-provided variables (`CARGO_CFG_TARGET_OS`, `CARGO_CFG_TARGET_ARCH`,
`CARGO_CFG_TARGET_ENV`, `HOST`, `TARGET`, `OUT_DIR`, `PROFILE`, `PATH`) are read
by `crates/yscv-kernels/build.rs` and behave as Cargo documents them. Three
non-Cargo variables affect the BLAS link search:

| Variable | Platform | Effect |
|---|---|---|
| `OPENBLAS_PATH=<dir>` | Windows | OpenBLAS install prefix to link against. |
| `VCPKG_ROOT=<dir>` | Windows | vcpkg root; the triplet is derived from `CARGO_CFG_TARGET_ARCH`. |
| `CONDA_PREFIX=<dir>` | Windows | Conda environment prefix searched for BLAS. |

---

## Related

- [`feature-flags.md`](feature-flags.md) — build-time Cargo features, plus the short list of most-used runtime knobs.
- [`onnx-cpu-kernels.md`](onnx-cpu-kernels.md) — kernel routing map and tracker reproduction commands.
- [`mpsgraph-guide.md`](mpsgraph-guide.md), [`gpu-backend-guide.md`](gpu-backend-guide.md) — backend-specific guides.
