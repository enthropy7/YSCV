# yscv-llm-bench

Decode-throughput bench harness for Llama / Qwen / Phi-style ONNX LLMs
running through `yscv-onnx`. Times **prefill** (one full forward pass
on the prompt) and **decode** (N autoregressive forward passes, one
token each) separately; emits a one-line summary plus a JSON record on
stdout for ingestion by external comparison scripts.

The crate also ships companion bins:

- **`kernel_bench`** — INT8 / INT4 GEMM / GEMV plus INT8 depthwise
  microbench across tracker and
  TinyLlama-1.1B / Llama-3.2-1B / Phi-2 / Llama-7B linear-layer shape
  set, with bitwise / numerical correctness checks against the scalar
  reference. See [`src/bin/kernel_bench.rs`](src/bin/kernel_bench.rs).
- **`calib_accuracy`** — PTQ calibration accuracy harness comparing
  MinMax vs percentile vs MSE-optimal scale derivation on both
  synthetic distributions and real ONNX-runner activations.
  See [`src/bin/calib_accuracy.rs`](src/bin/calib_accuracy.rs).
- **`inspect`** — minimal ONNX input/output + one-hop graph walker.
  Useful when chasing missing-op / shape-mismatch errors backwards
  through a graph.
- **`quantize_tracker`** — end-to-end multi-input tracker PTQ demo,
  QDQ/QLinear exporter smoke test, and fp32-vs-quantized accuracy gate.
- **`bench_tracker`** — workspace-native yscv tracker benchmark for the
  private two-input model. Prints min/p50/avg/p95 plus quant-runtime counters
  so hot-path claims measure the current `crates/yscv-onnx`, not a copied
  private harness.

For an end-to-end comparison vs `onnxruntime` on the same model and
prompt, see [`COMPARISON.md`](COMPARISON.md).

## Build

```sh
cargo build --release --no-default-features -p yscv-llm-bench
# (default features pull in BLAS — only needed for the BLAS sgemm A/B
# in the workspace, not for this bench.)
```

## Run (stateless decoder — distilgpt2 style)

```sh
yscv-llm-bench \
  --model path/to/model.onnx \
  --input-ids prompt-tokens.json \
  --max-tokens 64 \
  --warmup 2 \
  --with-attention-mask
```

## Run (cached decoder — Llama / Qwen / Phi style)

```sh
yscv-llm-bench \
  --model path/to/model.onnx \
  --input-ids prompt-tokens.json \
  --max-tokens 64 \
  --warmup 1 \
  --with-attention-mask \
  --with-position-ids \
  --kv-shape 24,2,64
#                ^  ^ ^
#                |  | head_dim
#                |  n_kv_heads (use n_q_heads if model is MHA, not GQA)
#                n_layers
```

When `--kv-shape` is supplied, the harness:

- auto-detects `past_key_values.{i}.{key,value}` inputs by name,
- allocates empty past tensors `[1, n_heads, 0, head_dim]` before
  prefill,
- feeds them every forward pass and cycles `present.{i}.*` outputs
  back into past for the next call (zero-copy via `HashMap::remove`),
- decode loop feeds **only the last token**, tracking
  `position_offset` for the position_ids / attention_mask extension.

Without `--kv-shape` past inputs (if any) are not fed and the model
runs in stateless mode (re-encodes the whole sequence per step) via
the existing `yscv_onnx::generate` helper.

## Flags

- `--model PATH` — decoder-only ONNX (TinyLlama, Phi-2, Qwen, Llama-style).
- `--input-ids PATH` — JSON array of `u32` token IDs (the prompt).
  Pre-tokenize externally (yscv keeps the no-Python promise at
  *runtime*; tokenization is a one-time prep step).
- `--max-tokens N` — number of new tokens to generate (default 64).
- `--warmup K` — discarded warm-up forward passes before the timed
  run (default 2). Catches first-touch allocs / page faults.
- `--input-name NAME` / `--output-name NAME` — override the input
  tensor name (default `input_ids`) / logits output name (default
  `logits`) for non-standard exports.
- `--with-attention-mask` — synthesise an all-ones `attention_mask`
  per forward pass. Many HF exports require it.
- `--with-position-ids` — synthesise `position_ids = [pos_off,
  pos_off+1, …]`. Llama / Qwen / Phi exports require it; GPT-2-style
  exports usually don't.
- `--kv-shape n_layers,n_heads,head_dim` — see "cached decoder"
  section above.

`prompt-tokens.json` is a flat JSON array of `u32` token IDs. Pre-tokenize
externally (the harness deliberately doesn't bundle a tokenizer — yscv
keeps the no-Python promise at runtime; tokenization is a one-time prep
step you run with whatever stack you prefer).

## Reproducing the TinyLlama numbers

### 1. Download the ONNX export

The simplest source is the `optimum-onnx` mirror on HuggingFace:

```sh
mkdir -p ~/models/tinyllama
cd ~/models/tinyllama
huggingface-cli download \
  TinyLlama/TinyLlama-1.1B-Chat-v1.0-ONNX \
  --local-dir .
```

(or any equivalent decoder-only export — Phi-2, Qwen-0.5B, Llama-3.2-1B
all work as long as the graph exposes `input_ids` → `logits`.)

### 2. Pre-tokenize a prompt

Once, with the matching tokenizer:

```python
from transformers import AutoTokenizer
import json

tok = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
ids = tok.encode("Explain quantization in two sentences.", add_special_tokens=True)
json.dump(ids, open("prompt.json", "w"))
```

### 3. Quantize (optional)

```sh
cargo run --release -p yscv-quantize-cli -- \
  ~/models/tinyllama/model.onnx \
  --output ~/models/tinyllama/model-int4.onnx \
  --weights-only
```

`--weights-only` produces a packed-INT4-weights model that uses the
`packed_int4_gemv_dispatch` / `packed_int4_gemm_dispatch` runtime path
on rank-2 MatMul/Gemm nodes.

### 4. Bench

```sh
# fp32 baseline:
yscv-llm-bench --model ~/models/tinyllama/model.onnx \
  --input-ids prompt.json --max-tokens 64 --warmup 2

# packed-INT4:
yscv-llm-bench --model ~/models/tinyllama/model-int4.onnx \
  --input-ids prompt.json --max-tokens 64 --warmup 2
```

Compare the `decode tokens/sec` line; INT4 should win on memory-bound
edge hardware (Pi 5 / RK3588 / Apple M-series) by ~2-3×.

### 5. Cross-check vs llama.cpp

```sh
# in llama.cpp checkout, after `make`:
./main -m ~/models/tinyllama/q4_0.gguf -p "Explain quantization …" \
  -n 64 --n-predict 64 --temp 1.0 --no-display-prompt
# look for "eval time" / "ms per token"
```

## Output

Single human-readable line on stderr plus a JSON line on stdout:

```
prompt=32t  prefill=215.4ms  decode=64t / 4823.2ms = 13.27 tok/s  total=5039.2ms
{"model":"…/model.onnx","prompt_tokens":32,"decode_tokens":64,"prefill_ms":215.4,"decode_total_ms":4823.2,"decode_tokens_per_sec":13.27,"total_wall_ms":5039.2}
```

Pipe stdout through `jq` to aggregate across runs.

## Tracker Multi-Input Quantization

The quantization tooling supports models with multiple simultaneous inputs,
including the private Siamese tracker shape:

```sh
cargo run --release --no-default-features -p yscv-llm-bench --bin quantize_tracker -- \
  --model /home/null/YSCV/private/private/model.onnx \
  --shape input.1:1x3x128x128,input.249:1x3x256x256 \
  --output /tmp/tracker_int8.onnx \
  --calib-samples 32 --eval-samples 4 \
  --format qdq
```

For shipping accuracy, feed representative paired crops instead of synthetic
random tensors:

```sh
cargo run --release --no-default-features -p yscv-llm-bench --bin quantize_tracker -- \
  --model /home/null/YSCV/private/private/model.onnx \
  --shape input.1:1x3x128x128,input.249:1x3x256x256 \
  --calibration-jsonl input.1=tmpl_calib.jsonl,input.249=search_calib.jsonl \
  --eval-jsonl input.1=tmpl_eval.jsonl,input.249=search_eval.jsonl \
  --format qdq
```

With `--eval-jsonl`, `quantize_tracker` enforces the tracker ship gate on both
outputs: finite tensors, rel-RMSE <= 2%, and rel-L∞ <= 10%. Without
`--eval-jsonl`, synthetic random eval remains smoke-only and prints warnings
instead of failing.

For CLI quantization, either provide one JSONL row containing both inputs, or
zip paired streams row-by-row:

```sh
cargo run --release -p yscv-quantize-cli -- \
  /home/null/YSCV/private/private/model.onnx \
  --output /tmp/tracker_int8.onnx \
  --calibration input.1=tmpl.jsonl,input.249=search.jsonl
```

The exported QDQ ONNX now round-trips through yscv, passes `onnx.checker`, and
loads in ONNX Runtime. On random synthetic calibration data the accuracy warning
is expected; use representative template/search image crops for shipping
thresholds.

`quantize_tracker --format qdq` enables the yscv-fast cleanup path by default:
constant weight DequantizeLinear nodes are folded back into initializers and
inner QDQ pairs between Conv-like nodes are stripped so loader-time Conv layout
normalisation and prepacking still fire. QDQ export quantizes regular,
grouped, and depthwise Conv weights after normalising loader-internal weight
layouts back to standard OIHW, so the model stays checker/ORT-friendly while
yscv can still recover its NHWC depthwise/pointwise layout on reload. The
cleanup pass also prunes quantized tensors/scales made dead by fold/strip so
ORT does not warn about hundreds of unused initializers. Set `YSCV_QUANT_FAST=0`
to keep the fully explicit QDQ graph for debugging. `--format qlinear` emits
the standard QLinear operator form for checker/runtime interoperability
experiments. Both formats rewrite the same optimized graph used during
calibration, so fused Conv output names match collected activation stats; this
is especially important for QLinear coverage because it requires both input and
output activation scales. During calibration the runner also records local
intermediates inside fused DW/PW Conv chains; regular inference still uses the
zero-env streaming path.

yscv's explicit QLinear runtime has a targeted depthwise 3×3/5×5 fast path:
stride-1 nodes and measured-win stride-2 tracker shapes are packed once at
load, then inference reuses runner scratch for the NCHW/NHWC int8 staging and
int32 accumulator buffers. Matching `DequantizeLinear -> [Relu] ->
QuantizeLinear` boundaries are folded in the quant domain by default; set
`YSCV_QUANT_INT8_FAST=0` to force explicit boundaries for A/B. This keeps the
standard QLinear model useful for ORT
interoperability while moving the tracker depthwise bottleneck toward the same
multi-arch INT8 kernels used by the internal quant path.
QLinearConv and QuantizeLinear outputs are stored internally as real i8 tensors;
`bench_tracker` reports `quant_i8_stores` and `quant_i8_materializations` so a
run can prove whether it stayed quantized between Conv-like nodes. Entry
`QuantizeLinear` nodes that feed QLinear activations quantize directly into i8
storage through the scalar/AVX2/AVX-512F/NEON dispatch in `yscv-kernels` with packed x86 i8 stores; the
old scalar iterator path is gone from that hot route. For explicit QLinear
pointwise/MatMul shapes, yscv also builds the AVX-512 VNNI 4x16 RHS layout at
model load so repeated tracker runs do not repack the same INT8 weights in the
hot path. The benchmark JSON includes `quant_chain_candidates`,
`quant_chain_executed`, and `quant_chain_fallback`; until fused INT8 chain
actions land, the private tracker QLinear export should show candidates but
zero executed chains.

To reproduce the tracker comparison matrix while tuning quant kernels:

```sh
ITERS=200 RUNS=3 apps/llm-bench/scripts/bench_tracker_quant_matrix.sh
```

The script exports QDQ-fast and QLinear models, then reports yscv fp32,
yscv QDQ-fast, yscv QLinear, ORT fp32, ORT QDQ-fast, and ORT QLinear at 1T
and 6T. yscv rows are produced by `bench_tracker` and include executed
quant-action counters. Use this before claiming any tracker hot-path speedup.
