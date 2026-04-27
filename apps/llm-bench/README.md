# yscv-llm-bench

Decode-throughput bench harness for Llama-style ONNX LLMs running through
`yscv-onnx`. Times **prefill** (one full forward pass on the prompt) and
**decode** (N autoregressive forward passes, one token each) separately;
emits a one-line summary plus a JSON record on stdout for ingestion by
external comparison scripts.

## Build

```sh
cargo build --release -p yscv-llm-bench
```

## Run

```sh
yscv-llm-bench \
  --model path/to/model.onnx \
  --input-ids prompt-tokens.json \
  --max-tokens 64 \
  --warmup 2
```

Optional:
- `--input-name <name>` — override the model's input tensor name (default `input_ids`).
- `--output-name <name>` — override the logits output name (default `logits`).

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
