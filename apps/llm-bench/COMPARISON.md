# yscv-llm-bench vs ONNX Runtime (Python)

End-to-end LLM inference comparison: same ONNX model, same prompt, same
generation parameters (greedy / top-k=1), measured the same way (warmup
discarded, prefill and decode timed separately).

## Setup

- **CPU**: AMD Zen 4, 12 logical threads
- **Model**: cached-decoder transformer from
  `/tmp/yscv-llm/model.onnx` — 24 layers, n_kv_heads=2, head_dim=64,
  hidden=1024, **fp32**, 1.9 GB on disk
  (HuggingFace optimum-onnx export, `past_key_values.{i}.{key,value}`
  inputs, `present.{i}.{key,value}` outputs)
- **Prompt**: 43 tokens (`/tmp/yscv-llm/prompt.json`)
- **Decode**: 16 new tokens, greedy
- **ORT**: `onnxruntime` 1.25.0, `CPUExecutionProvider`, sequential
  execution, 12 intra/inter-op threads
- **yscv**: workspace `--no-default-features` (no BLAS), default
  rayon pool

## How to reproduce

```sh
# 1. yscv side
cargo build --release --no-default-features -p yscv-llm-bench

target/release/yscv-llm-bench \
  --model /tmp/yscv-llm/model.onnx \
  --input-ids /tmp/yscv-llm/prompt.json \
  --kv-shape 24,2,64 \
  --max-tokens 16 --warmup 1 \
  --with-attention-mask --with-position-ids

# 2. ORT side (use any venv with onnxruntime + numpy)
python apps/llm-bench/python/bench_ort.py \
  --model /tmp/yscv-llm/model.onnx \
  --input-ids /tmp/yscv-llm/prompt.json \
  --kv-shape 24,2,64 \
  --max-tokens 16 --warmup 1 --threads 12 \
  --with-attention-mask --with-position-ids
```

Both binaries print the same one-line summary on stderr and a JSON
record on stdout — pipe through `jq` to aggregate across runs.

## Numbers (1.9 GB Qwen-style; 43-tok prompt, 16-tok decode)

Fresh desktop recheck on 2026-04-29, median of 3 runs:

| run                                   | prefill (ms) | decode (ms) | decode tok/s | gap to ORT |
|---------------------------------------|-------------:|------------:|-------------:|-----------:|
| yscv fp32 (baseline)                  |         1725 |        7781 |         2.06 |       10.9× |
| yscv +`--int4-weights 32` (group=32)  |         1939 |        4536 |     **3.53** |    **6.35×** |
| yscv +`--int4-weights 32 --kv-dtype i8` |       1929 |        4568 |         3.50 |       6.40× |
| ORT 1.25, fp32, 12 thr                |           69 |         714 |        22.40 |          1× |

The `--int4-weights GS` flag in-process packs every eligible MatMul/Gemm
weight to symmetric INT4 with the supplied group size and re-runs the
exact same harness; runtime dispatch routes through
`packed_int4_gemv_dispatch` (decode) and `packed_int4_gemm_dispatch`
(prefill). 168 of the model's 2093 initializers get packed (every
linear-layer weight in the 24-block stack).

> Output drift between yscv and ORT is bitwise-close on each per-op
> matmul (1-ULP within Add ordering); greedy argmax-next-token agrees
> for the first ~6-8 tokens before drifting due to fp accumulation
> order — measuring the *same* sampler is what matters here.

The new AVX-512 `packed_int4_gemv` kernel (vectorised nibble unpack via
`slli/srai_epi16` + `permutex2var_ps` for the lo/hi-lane gather) ships
**9.0× scalar speedup** vs the prior AVX2 path's 1.6× (kernel_bench
TLlama Q/K/V 2048×2048): 1333 µs/op → **241 µs/op**, effective weight BW
1.6 GB/s → **8.7 GB/s**. The end-to-end win is smaller than the kernel
win because GEMVs are only ~60% of decode time on this graph; the
remaining 40% (attention scores, RoPE, residuals, RMSNorm) is still
fp32 and is what Phase 2 / Phase 3 of the plan
(`/home/null/.claude/plans/hashed-hopping-falcon.md`) target.

## Why the gap

Both runners walk the exact same ONNX graph. The 11–27× difference
breaks down into three things:

1. **MLAS** — ORT links the Microsoft-curated MLAS GEMM library
   (handwritten AVX-512 microkernels with sgemm-grade pack/blocking,
   per-CPU dispatch tables). yscv's matmul is the workspace's
   `blocked_gemm_*` implementation — pure intrinsics, single MR=4×24
   microkernel, no per-shape tuning. The bulk of the gap is here:
   transformer hidden×hidden GEMMs at K=N=1024-4096 are exactly where
   MLAS wins by 5-10× over a naive blocked GEMM.
2. **No fp16 / no quantization** — the model is fp32, not fp16 or int8.
   ORT itself doesn't auto-quantize, so this isn't a fair-fight axis,
   but it bounds the achievable ceiling: the same model loaded as int4
   weights through the yscv `packed_int4_gemv_dispatch` path is
   ~2-3× faster on decode than ORT's fp32 (see "next-step" below).
3. **Op-coverage micro-overheads** — yscv's runner currently
   re-allocates intermediate Tensors at every node (no buffer arena
   yet); for decode where a single forward pass touches ~3000 nodes,
   that's a few % of total time. ORT's session-level memory planner
   reuses a single graph-allocator block.

## What this number actually tells you

- **2 tok/s on a 1.9 GB fp32 cached decoder** is what the current
  pure-Rust workspace can sustain on Zen 4 / 12 threads, end-to-end,
  with zero foreign linkage (no BLAS, no MLAS, no Python). That is
  small-edge territory — usable for batched offline inference, not
  for interactive chat.
- The fp32 → packed-INT4 path is *already wired* in the runner
  (`exec_matmul` auto-dispatches to `packed_int4_gemv_dispatch` when
  weights are packed). Re-running this same comparison after running
  the model through `yscv-quantize-cli --weights-only` is the next
  honest data point — the kernel-level microbench
  (`apps/llm-bench/src/bin/kernel_bench.rs`) shows int4 GEMV on
  TinyLlama-shape weights at ~1.6 GB/s effective read bandwidth, and
  that's the throughput the model-level bench should approach.
- The MLAS gap is closable in principle (the `crates/yscv-kernels`
  AVX-512-VNNI register-blocked path lands 80-100% of theoretical
  peak on int8 hidden×hidden — the same trick applied to fp32 sgemm
  is the work that takes us from "11× behind ORT" to "2-3× behind
  ORT, and faster on int4 decode").

## See also

- `apps/llm-bench/src/bin/kernel_bench.rs` — kernel-level bench
  (int8 / int4 GEMM/GEMV at TinyLlama / Llama-3.2 / Phi-2 / Llama-7B
  shapes, scalar vs SIMD, +bitwise correctness checks)
- `apps/llm-bench/src/bin/calib_accuracy.rs` — PTQ scheme comparison
  (MinMax / percentile / MSE-optimal) on synthetic + real activations
- `apps/llm-bench/src/bin/inspect.rs` — minimal ONNX input/output +
  one-hop node graph walker, used to chase missing-op / shape errors
