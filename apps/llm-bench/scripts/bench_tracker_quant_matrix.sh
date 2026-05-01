#!/usr/bin/env bash
set -euo pipefail

ROOT=${YSCV_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}
MODEL=${MODEL:-$ROOT/private/private/model.onnx}
ONNX_FPS_DIR=${ONNX_FPS_DIR:-$ROOT/private/private/onnx-fps}
BENCH_TRACKER=${BENCH_TRACKER:-$ROOT/target/release/bench_tracker}
ORT_PY=${ORT_PY:-$ONNX_FPS_DIR/bench_ort_onnx_fps.py}
PYTHON=${PYTHON:-$ONNX_FPS_DIR/.venv/bin/python}
QDQ_MODEL=${QDQ_MODEL:-/tmp/tracker_fast_qdq.onnx}
QLINEAR_MODEL=${QLINEAR_MODEL:-/tmp/tracker_qlinear.onnx}
ITERS=${ITERS:-200}
RUNS=${RUNS:-3}
CALIB_SAMPLES=${CALIB_SAMPLES:-16}
EVAL_SAMPLES=${EVAL_SAMPLES:-2}
SHAPES=${SHAPES:-input.1:1x3x128x128,input.249:1x3x256x256}
INPUT_ARGS=(--input input.1:1x3x128x128 --input input.249:1x3x256x256)

cd "$ROOT"

cargo build --release -p yscv-llm-bench --bin quantize_tracker --no-default-features >/dev/null
cargo build --release -p yscv-llm-bench --bin bench_tracker --no-default-features >/dev/null

printf '\n== export qdq-fast ==\n'
"$ROOT/target/release/quantize_tracker" \
  --model "$MODEL" \
  --shape "$SHAPES" \
  --format qdq \
  --calib-samples "$CALIB_SAMPLES" \
  --eval-samples "$EVAL_SAMPLES" \
  --output "$QDQ_MODEL"

printf '\n== export qlinear ==\n'
"$ROOT/target/release/quantize_tracker" \
  --model "$MODEL" \
  --shape "$SHAPES" \
  --format qlinear \
  --calib-samples "$CALIB_SAMPLES" \
  --eval-samples 1 \
  --output "$QLINEAR_MODEL"

run_yscv() {
  local label=$1 model=$2 threads=$3
  printf '\n== yscv %s %sT ==\n' "$label" "$threads"
  for run in $(seq 1 "$RUNS"); do
    printf -- '-- run %s\n' "$run"
    RAYON_NUM_THREADS=$threads "$BENCH_TRACKER" \
      --model "$model" "${INPUT_ARGS[@]}" \
      --iters "$ITERS" --threads "$threads" --json
  done
}

run_ort() {
  local label=$1 model=$2 threads=$3
  printf '\n== ort %s %sT ==\n' "$label" "$threads"
  for run in $(seq 1 "$RUNS"); do
    printf -- '-- run %s\n' "$run"
    "$PYTHON" "$ORT_PY" \
      --model "$model" "${INPUT_ARGS[@]}" \
      --iters "$ITERS" --threads "$threads" --json
  done
}

run_yscv fp32 "$MODEL" 1
run_yscv fp32 "$MODEL" 6
run_yscv qdq-fast "$QDQ_MODEL" 1
run_yscv qdq-fast "$QDQ_MODEL" 6
run_yscv qlinear "$QLINEAR_MODEL" 1
run_yscv qlinear "$QLINEAR_MODEL" 6
run_ort fp32 "$MODEL" 1
run_ort fp32 "$MODEL" 6
run_ort qdq-fast "$QDQ_MODEL" 1
run_ort qdq-fast "$QDQ_MODEL" 6
run_ort qlinear "$QLINEAR_MODEL" 1
run_ort qlinear "$QLINEAR_MODEL" 6
