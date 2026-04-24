#!/usr/bin/env bash
# Thread-sweep yscv + ORT on the Siamese tracker.
#
# Hardware: AMD Ryzen 5 7500F (Zen 4, 6C/12T).
# 12T = SMT-oversubscribed; captured for scaling analysis, not a
# recommended deployment target.
#
# Usage: ./scripts/bench_sweep.sh > /tmp/sweep.txt
# Requires: onnx-fps release binary at
#   <repo>/onnx-fps/target/release/onnx-fps
# and OpenBLAS on LD path (via RUSTFLAGS at build time).

set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
ONNX_FPS_DIR="${ONNX_FPS_DIR:-$REPO/onnx-fps}"
FPS="${FPS:-$ONNX_FPS_DIR/target/release/onnx-fps}"
MODEL="${MODEL:-$ONNX_FPS_DIR/../model.onnx}"
BENCH_ORT="${BENCH_ORT:-$ONNX_FPS_DIR/bench_ort_onnx_fps.py}"
if [[ ! -f "$BENCH_ORT" ]]; then
  BENCH_ORT="$ONNX_FPS_DIR/bench_ort.py"
fi
VENV_PY="${VENV_PY:-$ONNX_FPS_DIR/.venv/bin/python}"

THREADS="${THREADS:-1 2 4 6 8 12}"
ITERS="${ITERS:-500}"
RUNS="${RUNS:-3}"

if [[ ! -x "$FPS" ]]; then
  echo "error: $FPS not built. Run cargo build --release in $ONNX_FPS_DIR first." >&2
  exit 1
fi
if [[ ! -f "$MODEL" ]]; then
  echo "error: $MODEL missing." >&2
  exit 1
fi

echo "### yscv thread-sweep"
echo "# iters=$ITERS runs=$RUNS per thread-count"
for t in $THREADS; do
  echo "=== yscv T=$t ==="
  for r in $(seq 1 "$RUNS"); do
    RAYON_NUM_THREADS="$t" "$FPS" \
      --model "$MODEL" \
      --input input.1:1x3x128x128 --input input.249:1x3x256x256 \
      --iters "$ITERS" --cpu 2>&1 | grep -E '^min:|^p50:|^  882:|^  894:'
  done
done

echo
echo "### ORT thread-sweep"
echo "# iters=$ITERS runs=$RUNS per thread-count"
for t in $THREADS; do
  echo "=== ORT T=$t ==="
  for r in $(seq 1 "$RUNS"); do
    "$VENV_PY" "$BENCH_ORT" \
      --model "$MODEL" --iters "$ITERS" --threads "$t" 2>&1 \
      | grep -E '^min|^p50'
  done
done
