# yscv-quantize

Post-training quantization CLI for `yscv-onnx` models.

## Single-File Calibration

Each JSONL row is one full inference sample. Top-level keys are graph input
names; values carry `shape` and flat row-major `values`.

```json
{"input.1":{"shape":[1,3,128,128],"values":[0.0]},"input.249":{"shape":[1,3,256,256],"values":[0.0]}}
```

```sh
cargo run --release -p yscv-quantize-cli -- \
  private/private/model.onnx \
  --output /tmp/tracker_int8.onnx \
  --calibration tracker_calib.jsonl
```

## Paired Multi-Input Streams

For Siamese trackers and other dual-input models, calibration can also be
provided as one stream per input. Streams are zipped row-by-row, so row `N` in
`tmpl.jsonl` is paired with row `N` in `search.jsonl`.

Each stream row can be either a bare tensor:

```json
{"shape":[1,3,128,128],"values":[0.0]}
```

or the existing wrapped form:

```json
{"input.1":{"shape":[1,3,128,128],"values":[0.0]}}
```

```sh
cargo run --release -p yscv-quantize-cli -- \
  private/private/model.onnx \
  --output /tmp/tracker_int8.onnx \
  --calibration input.1=tmpl.jsonl,input.249=search.jsonl \
  --format qdq
```

Use `apps/llm-bench/python/prepare_paired_calibration.py` to turn paired
`.npy` arrays into the single-file JSONL format when that is more convenient.

## Notes

`--format qdq` is the default and keeps a standard QuantizeLinear /
DequantizeLinear graph. `--format qlinear` emits QLinearConv /
QLinearMatMul shells for interoperability experiments with runtimes that prefer
the operator form.

Conv weights are quantized in standard OIHW form even when yscv loaded them
into an internal KHWC, grouped-KHWC, or depthwise-KHWC layout for fast NHWC
execution. This keeps exported QDQ/QLinear graphs compatible with
`onnx.checker` and ONNX Runtime, while yscv can restore its internal Conv
layouts after save/reload. The CLI optimizes the graph before calibration and
rewrite so activation-stat tensor names line up with fused Conv/Relu and
QLinear export can cover the same graph the runner actually executes. After
rewrite, unused original fp32 weights and stripped QDQ metadata are pruned from
both QDQ and QLinear exports.

`--weights-only` skips activation calibration. `--strip-inner-qdq` removes QDQ
pairs between Conv-like ops to restore graph fusions for speed-oriented A/Bs.
For tracker speed work, use `apps/llm-bench/scripts/bench_tracker_quant_matrix.sh`;
its yscv rows are produced by the workspace-native `bench_tracker` binary and
include quant-runtime counters so QDQ cleanup is not mistaken for actual INT8
Conv execution.
