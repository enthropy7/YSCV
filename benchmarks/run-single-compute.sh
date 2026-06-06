#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Generate a reproducible single-compute markdown snapshot.
#
# The benchmark intentionally runs each operation in its own process. PyTorch
# full-suite single-process runs can contaminate later memory-bound ops through
# allocator/cache state; isolated per-op p50 is the comparison target here.

threads="${RAYON_NUM_THREADS:-12}"
iters="${ITERS:-2000}"
warmup="${WARMUP:-250}"
spin_us="${YSCV_POOL_SPIN_US:-200}"
python="${PYTHON:-.venv/bin/python}"
date_utc="$(date -u +%F)"
out="${OUT:-benchmarks/single-compute-${date_utc}.md}"
raw_dir="${RAW_DIR:-artifacts/single-compute-${date_utc}-$(date -u +%H%M%S)}"

if [ ! -x "$python" ]; then
  echo "error: python runner '$python' is not executable" >&2
  echo "hint: set PYTHON=/path/to/python or create .venv with torch/onnxruntime" >&2
  exit 1
fi

mkdir -p "$(dirname "$out")" "$raw_dir"

git_commit="$(git rev-parse --short=12 HEAD 2>/dev/null || echo unknown)"
git_dirty="clean"
if ! git diff --quiet --ignore-submodules -- 2>/dev/null || ! git diff --cached --quiet --ignore-submodules -- 2>/dev/null; then
  git_dirty="dirty"
fi
host="$(hostname 2>/dev/null || echo unknown)"
uname_s="$(uname -a 2>/dev/null || echo unknown)"
rustc_v="$(rustc -Vv 2>/dev/null | tr '\n' '; ' || echo unknown)"
cargo_v="$(cargo -V 2>/dev/null || echo unknown)"
python_v="$("$python" --version 2>&1 || echo unknown)"
torch_v="$("$python" - <<'PY' 2>/dev/null || echo unavailable
try:
    import torch
    print(torch.__version__)
except Exception as exc:
    print(f"unavailable: {exc}")
PY
)"
ort_v="$("$python" - <<'PY' 2>/dev/null || echo unavailable
try:
    import onnxruntime as ort
    print(ort.__version__)
except Exception as exc:
    print(f"unavailable: {exc}")
PY
)"
numpy_v="$("$python" - <<'PY' 2>/dev/null || echo unavailable
try:
    import numpy
    print(numpy.__version__)
except Exception as exc:
    print(f"unavailable: {exc}")
PY
)"

rows=(
  "add|1024x1024|add_1M|add_1M|add_1M|add_1M|add_1M"
  "mul|1024x1024|mul_1M|mul_1M|mul_1M|mul_1M|mul_1M"
  "exp|1024x1024|exp_1M|exp_1M|exp_1M|exp_1M|exp_1M"
  "sum|1024x1024|sum_1M_raw_slice|sum_1M|sum_1M|sum_1M|sum_1M"
  "max|1024x1024|max_1M_raw_slice|max_1M|max_1M|max_1M|max_1M"
  "add broadcast last dim|1024x1024 + 1024|add_broadcast_1024x1024_by_1024|add_broadcast_1024x1024_by_1024|add_broadcast_1024x1024_by_1024|add_broadcast_1024x1024_by_1024|add_broadcast_1024x1024_by_1024"
  "sub broadcast row minus matrix|1024 - 1024x1024|sub_broadcast_1024_by_1024x1024|sub_broadcast_1024_by_1024x1024|sub_broadcast_1024_by_1024x1024|sub_broadcast_1024_by_1024x1024|sub_broadcast_1024_by_1024x1024"
  "relu|921600|relu_921K|relu_921K|relu_921K|relu_921K|relu_921K"
  "sigmoid|921600|sigmoid_921K|sigmoid_921K|sigmoid_921K|sigmoid_921K|sigmoid_921K"
  "tanh|1024x1024|tanh_1M|tanh_1M|tanh_1M|tanh_1M|tanh_1M"
  "gelu sigmoid approximation|1024x1024|gelu_1M|gelu_1M_sigmoid_formula|gelu_1M_sigmoid_graph|gelu_1M|gelu_1M"
  "silu|1024x1024|silu_1M|silu_1M|silu_1M_sigmoid_mul_graph|silu_1M|silu_1M"
  "softmax|32x1000|softmax_32x1000|softmax_32x1000|softmax_32x1000|softmax_32x1000|softmax_32x1000"
  "log_softmax|32x1000|log_softmax_32x1000|log_softmax_32x1000|log_softmax_32x1000|log_softmax_32x1000|log_softmax_32x1000"
  "softmax|512x256|softmax_512x256|softmax_512x256|softmax_512x256|softmax_512x256|softmax_512x256"
  "layer_norm|512x256|layer_norm_512x256|layer_norm_512x256|layer_norm_512x256|layer_norm_512x256|layer_norm_512x256"
  "batch_norm|1x64x64x3 / 1x3x64x64|batch_norm_1x64x64x3|batch_norm_1x3x64x64|batch_norm_1x3x64x64|batch_norm_1x64x64x3|batch_norm_1x3x64x64"
)

extract_stats() {
  local file="$1"
  local label="$2"
  awk -v label="$label" '
    $1 == label {
      min = p50 = avg = ""
      for (i = 1; i <= NF; i++) {
        if ($i == "min=") { min = $(i + 1) }
        else if ($i ~ /^min=/) { min = $i }
        if ($i == "p50=") { p50 = $(i + 1) }
        else if ($i ~ /^p50=/) { p50 = $i }
        if ($i == "avg=") { avg = $(i + 1) }
        else if ($i ~ /^avg=/) { avg = $i }
      }
      gsub(/min=/, "", min); gsub(/p50=/, "", p50); gsub(/avg=/, "", avg)
      gsub(/us/, "", min); gsub(/us/, "", p50); gsub(/us/, "", avg)
      if (min != "" && p50 != "" && avg != "") {
        print min, p50, avg
        exit
      }
    }
  ' "$file"
}

status_vs_yscv() {
  local y="$1"
  local other="$2"
  awk -v y="$y" -v o="$other" 'BEGIN {
    d = y - o
    if (d < 0) d = -d
    if (d <= 1) print "parity"
    else if (y < o) print "YSCV win"
    else print "YSCV slower"
  }'
}

ratio_vs_yscv() {
  local y="$1"
  local other="$2"
  awk -v y="$y" -v o="$other" 'BEGIN {
    if (y <= 0) print "n/a"
    else printf "%.2fx", o / y
  }'
}

run_yscv() {
  local filter="$1"
  local file="$2"
  RAYON_NUM_THREADS="$threads" \
    YSCV_POOL=yscv \
    YSCV_POOL_SPIN_US="$spin_us" \
    cargo run -q -p yscv-llm-bench --release --bin compute_gap -- \
      --iters "$iters" --warmup "$warmup" --filter "$filter" | tee "$file"
}

run_torch() {
  local filter="$1"
  local file="$2"
  "$python" benchmarks/python/bench_torch_single_ops.py \
    --threads "$threads" --iters "$iters" --warmup "$warmup" --filter "$filter" | tee "$file"
}

run_ort() {
  local filter="$1"
  local file="$2"
  "$python" benchmarks/python/bench_ort_single_ops.py \
    --threads "$threads" --iters "$iters" --warmup "$warmup" --filter "$filter" | tee "$file"
}

summary_tmp="$raw_dir/summary.rows"
: > "$summary_tmp"

for row in "${rows[@]}"; do
  IFS='|' read -r display shape y_label t_label o_label y_filter py_filter <<< "$row"
  safe="${y_label//[^A-Za-z0-9_]/_}"
  y_file="$raw_dir/yscv_${safe}.txt"
  t_file="$raw_dir/pytorch_${safe}.txt"
  o_file="$raw_dir/onnxruntime_${safe}.txt"

  echo
  echo "=== yscv ${y_label}"
  run_yscv "$y_filter" "$y_file"
  echo "=== pytorch ${t_label}"
  run_torch "$py_filter" "$t_file"
  echo "=== onnxruntime ${o_label}"
  run_ort "$py_filter" "$o_file"

  y_stats="$(extract_stats "$y_file" "$y_label")"
  t_stats="$(extract_stats "$t_file" "$t_label")"
  o_stats="$(extract_stats "$o_file" "$o_label")"
  if [ -z "$y_stats" ] || [ -z "$t_stats" ] || [ -z "$o_stats" ]; then
    echo "error: failed to parse stats for ${display}" >&2
    echo "  yscv='$y_stats' pytorch='$t_stats' ort='$o_stats'" >&2
    exit 1
  fi

  read -r y_min y_p50 y_avg <<< "$y_stats"
  read -r t_min t_p50 t_avg <<< "$t_stats"
  read -r o_min o_p50 o_avg <<< "$o_stats"
  t_status="$(status_vs_yscv "$y_p50" "$t_p50")"
  o_status="$(status_vs_yscv "$y_p50" "$o_p50")"
  status="$t_status vs PyTorch, $o_status vs ORT"
  if [ "$t_status" = "YSCV win" ] && [ "$o_status" = "YSCV win" ]; then
    status="YSCV win"
  elif [ "$t_status" = "parity" ] && [ "$o_status" = "YSCV win" ]; then
    status="parity vs PyTorch, YSCV win vs ORT"
  fi
  t_ratio="$(ratio_vs_yscv "$y_p50" "$t_p50")"
  o_ratio="$(ratio_vs_yscv "$y_p50" "$o_p50")"

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$display" "$shape" "$y_label" "$t_label" "$o_label" \
    "$y_min" "$y_p50" "$y_avg" \
    "$t_min" "$t_p50" "$t_avg" \
    "$o_min" "$o_p50" "$o_avg" \
    "$status|$t_ratio|$o_ratio" >> "$summary_tmp"
done

{
  echo "# Single-Compute CPU Snapshot"
  echo
  echo "Date: ${date_utc}"
  echo
  echo "Git: \`${git_commit}\` (${git_dirty})"
  echo
  echo "Host: ${host}"
  echo
  echo "Uname: \`${uname_s}\`"
  echo
  echo "Threads: ${threads}"
  echo
  echo "Iterations: ${iters}, warmup: ${warmup}"
  echo
  echo "YSCV env: \`RAYON_NUM_THREADS=${threads} YSCV_POOL=yscv YSCV_POOL_SPIN_US=${spin_us}\`"
  echo
  echo "Raw logs: \`${raw_dir}\`"
  echo
  echo "## Toolchain"
  echo
  echo "- Rust: \`${rustc_v}\`"
  echo "- Cargo: \`${cargo_v}\`"
  echo "- Python: \`${python_v}\`"
  echo "- PyTorch: \`${torch_v}\`"
  echo "- ONNX Runtime: \`${ort_v}\`"
  echo "- NumPy: \`${numpy_v}\`"
  echo
  echo "## Commands"
  echo
  echo '```bash'
  echo "RAYON_NUM_THREADS=${threads} YSCV_POOL_SPIN_US=${spin_us} ITERS=${iters} WARMUP=${warmup} \\"
  echo "  OUT=${out} bash benchmarks/run-single-compute.sh"
  echo '```'
  echo
  echo "## Methodology"
  echo
  echo "- Each row is measured as an isolated per-op process."
  echo "- Status is based on p50. \`parity\` means the p50 delta is at most 1 us."
  echo "- GELU is the sigmoid approximation formula/graph: \`x * sigmoid(1.702 * x)\`."
  echo "- YSCV uses NHWC for batch norm; PyTorch and ONNX Runtime use NCHW with the same data volume."
  echo
  echo "## Results"
  echo
  echo "Times are microseconds. Ratios are competitor p50 divided by YSCV p50."
  echo
  echo "| Operation | Shape | YSCV p50 | PyTorch p50 | PyTorch/YSCV | ORT p50 | ORT/YSCV | Status |"
  echo "|---|---:|---:|---:|---:|---:|---:|---|"
  while IFS=$'\t' read -r display shape _y_label _t_label _o_label _y_min y_p50 _y_avg _t_min t_p50 _t_avg _o_min o_p50 _o_avg status_blob; do
    IFS='|' read -r status t_ratio o_ratio <<< "$status_blob"
    echo "| ${display} | ${shape} | ${y_p50} | ${t_p50} | ${t_ratio} | ${o_p50} | ${o_ratio} | ${status} |"
  done < "$summary_tmp"
  echo
  echo "## Raw Rows"
  echo
  echo "| Runtime | Operation | Shape | Min us | P50 us | Avg us | Status vs YSCV |"
  echo "|---|---|---:|---:|---:|---:|---|"
  while IFS=$'\t' read -r _display shape y_label t_label o_label y_min y_p50 y_avg t_min t_p50 t_avg o_min o_p50 o_avg _status_blob; do
    echo "| yscv | ${y_label} | ${shape} | ${y_min} | ${y_p50} | ${y_avg} | self |"
    echo "| pytorch | ${t_label} | ${shape} | ${t_min} | ${t_p50} | ${t_avg} | $(status_vs_yscv "$y_p50" "$t_p50") |"
    echo "| onnxruntime | ${o_label} | ${shape} | ${o_min} | ${o_p50} | ${o_avg} | $(status_vs_yscv "$y_p50" "$o_p50") |"
  done < "$summary_tmp"
} > "$out"

echo
echo "wrote ${out}"
echo "raw logs in ${raw_dir}"
