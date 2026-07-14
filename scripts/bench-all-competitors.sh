#!/usr/bin/env bash
# Benchmark yscv against all competitors: onnxruntime (CPU + CoreML), tract, candle.
# Outputs JSON results to artifacts/competitor-benchmarks/
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT_DIR="$REPO_ROOT/artifacts/competitor-benchmarks"
mkdir -p "$OUT_DIR"

echo "=== yscv competitor benchmark suite ==="
echo "Output: $OUT_DIR"
echo ""

# ── 1. yscv CPU ──
echo "▸ yscv CPU..."
(cd "$REPO_ROOT" && BENCH_COOLDOWN=20 cargo run --release --manifest-path benchmarks/onnx-models/Cargo.toml --example bench_yolo 2>/dev/null) \
  | tee "$OUT_DIR/yscv-cpu.txt"
echo ""

# ── 2. yscv Metal (if available) ──
if cargo metadata --manifest-path "$REPO_ROOT/Cargo.toml" --format-version 1 2>/dev/null \
   | grep -q 'metal-backend'; then
  echo "▸ yscv Metal..."
  (cd "$REPO_ROOT" && cargo run --release --example bench_metal_yolo --features metal-backend 2>/dev/null) \
    | tee "$OUT_DIR/yscv-metal.txt"
  echo ""
fi

# ── 3. onnxruntime (Python) ──
if python3 -c "import onnxruntime" 2>/dev/null; then
  echo "▸ onnxruntime (CPU + CoreML)..."
  python3 "$REPO_ROOT/benchmarks/python/bench_yolo_onnxruntime.py" 2>/dev/null \
    | tee "$OUT_DIR/onnxruntime.txt"
  echo ""
else
  echo "▸ onnxruntime: not installed (pip install onnxruntime)"
fi

# ── 4. tract (Rust) ──
TRACT_BIN="$REPO_ROOT/benchmarks/rust-competitors/target/release/bench-competitors"
if [ -x "$TRACT_BIN" ]; then
  echo "▸ tract..."
  "$TRACT_BIN" | tee "$OUT_DIR/tract.txt"
  echo ""
else
  echo "▸ tract: building..."
  (cd "$REPO_ROOT/benchmarks/rust-competitors" && cargo build --release 2>/dev/null)
  if [ -x "$TRACT_BIN" ]; then
    "$TRACT_BIN" | tee "$OUT_DIR/tract.txt"
  else
    echo "  FAILED to build tract benchmark"
  fi
  echo ""
fi

echo ""
echo "=== Done. Results in $OUT_DIR ==="
