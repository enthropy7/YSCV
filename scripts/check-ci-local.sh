#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

full=0
cross=0
all_features=0
trend_baseline_max_age_days="${YSCV_TREND_BASELINE_MAX_AGE_DAYS:-14}"

usage() {
  cat <<'EOF'
Usage: scripts/check-ci-local.sh [--full] [--cross] [--all-features]

Runs the Linux-local subset of .github/workflows/ci.yml before pushing.

Default:
  - fmt, doc/safety gates
  - workspace check/test
  - clippy default + CI feature combinations
  - Linux GPU/RKNN/native-camera feature checks/tests

Options:
  --full
      Also run release workspace tests, extended proptests, CLI UX smoke,
      deterministic benchmark gates, and criterion trend gates.
  --cross
      Also run local cross-checks for aarch64 Linux and kernel-only
      Windows/macOS targets when the required toolchains are available.
  --all-features
      Also run workspace all-features clippy. Requires LIBCLANG_PATH for
      bindgen-backed native-camera/V4L2 dependencies.
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --full) full=1 ;;
    --cross) cross=1 ;;
    --all-features) all_features=1 ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

run() {
  echo
  echo "==> $*"
  "$@"
}

run_sh() {
  echo
  echo "==> $*"
  bash -lc "$*"
}

ensure_libclang() {
  if [ -n "${LIBCLANG_PATH:-}" ]; then
    return
  fi
  if ! command -v nix >/dev/null 2>&1; then
    echo "error: native-camera/all-features checks need LIBCLANG_PATH for bindgen" >&2
    echo "hint: install libclang or run with LIBCLANG_PATH=/path/to/lib" >&2
    exit 1
  fi

  echo
  echo "==> provisioning libclang via nix"
  mapfile -t nix_paths < <(
    nix --extra-experimental-features 'nix-command flakes' build \
      --no-link --print-out-paths \
      nixpkgs#llvmPackages.libclang.lib \
      nixpkgs#llvmPackages.llvm.lib
  )
  if [ "${#nix_paths[@]}" -lt 2 ]; then
    echo "error: nix did not return libclang + llvm paths" >&2
    exit 1
  fi
  export LIBCLANG_PATH="${nix_paths[0]}/lib"
  export LD_LIBRARY_PATH="${nix_paths[0]}/lib:${nix_paths[1]}/lib:${LD_LIBRARY_PATH:-}"
  echo "LIBCLANG_PATH=${LIBCLANG_PATH}"
}

ensure_aarch64_linux_cross() {
  if [ -n "${CC_aarch64_unknown_linux_gnu:-}" ] || command -v aarch64-linux-gnu-gcc >/dev/null 2>&1; then
    return 0
  fi
  if ! command -v nix >/dev/null 2>&1; then
    return 1
  fi

  echo
  echo "==> provisioning aarch64 Linux cross compiler via nix"
  mapfile -t nix_paths < <(
    nix --extra-experimental-features 'nix-command flakes' build \
      --no-link --print-out-paths \
      nixpkgs#pkgsCross.aarch64-multiplatform.stdenv.cc
  )
  local cc_root=""
  for path in "${nix_paths[@]}"; do
    if [ -x "${path}/bin/aarch64-unknown-linux-gnu-gcc" ]; then
      cc_root="$path"
      break
    fi
  done
  if [ -z "$cc_root" ]; then
    return 1
  fi
  export CC_aarch64_unknown_linux_gnu="${cc_root}/bin/aarch64-unknown-linux-gnu-gcc"
  export CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER="${cc_root}/bin/aarch64-unknown-linux-gnu-gcc"
  echo "CC_aarch64_unknown_linux_gnu=${CC_aarch64_unknown_linux_gnu}"
  return 0
}

need_linux_note() {
  case "$(uname -s)" in
    Linux) ;;
    *)
      echo "warning: this script mirrors the Linux CI path; current host is $(uname -s)" >&2
      ;;
  esac
}

need_linux_note

run cargo fmt --check
run cargo check --workspace --all-targets
run cargo clippy --workspace --all-targets -- -D warnings
run cargo clippy --workspace --all-targets --features gpu -- -D warnings
ensure_libclang
run cargo clippy --workspace --all-targets --features "gpu rknn native-camera" -- -D warnings
run cargo clippy -p yscv-onnx --all-targets --features "gpu profile" -- -D warnings
run cargo test --workspace
run bash scripts/check-doc-counts.sh
run bash scripts/check-safety-comments.sh
run bash scripts/check-runtime-dispatch.sh

run cargo check -p yscv-kernels --features gpu
run cargo check --workspace --features gpu
run cargo clippy -p yscv-kernels --features gpu --all-targets -- -D warnings
run cargo clippy -p yscv-onnx --features gpu --all-targets -- -D warnings
run cargo test -p yscv-kernels --features gpu --lib
run cargo test -p yscv-onnx --features gpu --lib

run cargo check -p yscv-kernels --features rknn
run cargo clippy -p yscv-kernels --features rknn --all-targets -- -D warnings
run cargo test -p yscv-kernels --features rknn --lib

run cargo check -p yscv-video --features native-camera
run cargo clippy -p yscv-video --features native-camera --all-targets -- -D warnings
run cargo test -p yscv-video --features native-camera --lib

if [ "$all_features" -eq 1 ]; then
  ensure_libclang
  run cargo clippy --workspace --all-targets --all-features -- -D warnings
fi

if [ "$cross" -eq 1 ]; then
  if command -v rustup >/dev/null 2>&1; then
    run rustup target add aarch64-unknown-linux-gnu x86_64-pc-windows-msvc aarch64-apple-darwin x86_64-apple-darwin
  fi

  if ensure_aarch64_linux_cross; then
    run cargo check --workspace --target aarch64-unknown-linux-gnu --no-default-features
    run cargo clippy -p yscv-kernels --features rknn --target aarch64-unknown-linux-gnu -- -D warnings
  else
    echo "skip: aarch64 workspace cross-check needs CC_aarch64_unknown_linux_gnu or aarch64-linux-gnu-gcc" >&2
  fi

  run cargo check -p yscv-kernels --target x86_64-pc-windows-msvc --no-default-features
  run cargo check -p yscv-kernels --target aarch64-apple-darwin --no-default-features
  run cargo clippy -p yscv-kernels --target aarch64-apple-darwin --no-default-features --lib -- -D warnings
  run cargo check -p yscv-kernels --target x86_64-apple-darwin --no-default-features
fi

if [ "$full" -eq 1 ]; then
  run cargo test --workspace --release
  run_sh "PROPTEST_CASES=4096 cargo test --workspace --lib"

  mkdir -p artifacts
  run cargo run -p yscv-cli --bin yscv-cli -- --list-cameras
  run cargo run -p yscv-cli --bin yscv-cli -- --list-cameras --device-name cam
  run cargo run -p camera-face-tool -- --list-cameras
  run cargo run -p camera-face-tool -- --list-cameras --device-name cam
  run cargo run -p yscv-cli --bin yscv-cli -- --diagnose-camera --diagnose-frames 5 --diagnose-report artifacts/yscv-cli-diagnostics.json
  run cargo run -p camera-face-tool -- --diagnose-camera --diagnose-frames 5 --diagnose-report artifacts/camera-face-tool-diagnostics.json
  run test -s artifacts/yscv-cli-diagnostics.json
  run test -s artifacts/camera-face-tool-diagnostics.json
  run cargo run -p yscv-cli --bin yscv-cli -- --max-frames 3 --event-log artifacts/events.jsonl
  run test -s artifacts/events.jsonl

  run_sh 'set -o pipefail
    cargo run -p yscv-cli --bin yscv-cli -- \
      --eval-detection-jsonl benchmarks/eval-detection-sample.jsonl \
      --eval-tracking-jsonl benchmarks/eval-tracking-sample.jsonl \
      --eval-iou 0.5 --eval-score 0.0 \
      2>&1 | tee artifacts/eval-dataset-output.txt
    cargo run -p yscv-cli --bin yscv-cli -- \
      --eval-detection-coco-gt benchmarks/eval-detection-coco-gt-sample.json \
      --eval-detection-coco-pred benchmarks/eval-detection-coco-pred-sample.json \
      --eval-iou 0.5 --eval-score 0.0 \
      2>&1 | tee -a artifacts/eval-dataset-output.txt
    cargo run -p yscv-cli --bin yscv-cli -- \
      --eval-detection-openimages-gt benchmarks/eval-detection-openimages-gt-sample.csv \
      --eval-detection-openimages-pred benchmarks/eval-detection-openimages-pred-sample.csv \
      --eval-iou 0.5 --eval-score 0.0 \
      2>&1 | tee -a artifacts/eval-dataset-output.txt
    cargo run -p yscv-cli --bin yscv-cli -- \
      --eval-detection-yolo-manifest benchmarks/eval-detection-yolo-manifest-sample.txt \
      --eval-detection-yolo-gt-dir benchmarks/eval-detection-yolo-gt \
      --eval-detection-yolo-pred-dir benchmarks/eval-detection-yolo-pred \
      --eval-iou 0.5 --eval-score 0.0 \
      2>&1 | tee -a artifacts/eval-dataset-output.txt
    cargo run -p yscv-cli --bin yscv-cli -- \
      --eval-detection-voc-manifest benchmarks/eval-detection-voc-manifest-sample.txt \
      --eval-detection-voc-gt-dir benchmarks/eval-detection-voc-gt \
      --eval-detection-voc-pred-dir benchmarks/eval-detection-voc-pred \
      --eval-iou 0.5 --eval-score 0.0 \
      2>&1 | tee -a artifacts/eval-dataset-output.txt
    cargo run -p yscv-cli --bin yscv-cli -- \
      --eval-detection-kitti-manifest benchmarks/eval-detection-kitti-manifest-sample.txt \
      --eval-detection-kitti-gt-dir benchmarks/eval-detection-kitti-gt \
      --eval-detection-kitti-pred-dir benchmarks/eval-detection-kitti-pred \
      --eval-iou 0.5 --eval-score 0.0 \
      2>&1 | tee -a artifacts/eval-dataset-output.txt
    cargo run -p yscv-cli --bin yscv-cli -- \
      --eval-detection-widerface-gt benchmarks/eval-detection-widerface-gt-sample.txt \
      --eval-detection-widerface-pred benchmarks/eval-detection-widerface-pred-sample.txt \
      --eval-iou 0.5 --eval-score 0.0 \
      2>&1 | tee -a artifacts/eval-dataset-output.txt
    cargo run -p yscv-cli --bin yscv-cli -- \
      --eval-tracking-mot-gt benchmarks/eval-tracking-mot-gt-sample.txt \
      --eval-tracking-mot-pred benchmarks/eval-tracking-mot-pred-sample.txt \
      --eval-iou 0.5 --eval-score 0.0 \
      2>&1 | tee -a artifacts/eval-dataset-output.txt
    test -s artifacts/eval-dataset-output.txt'

  run cargo run -p yscv-cli --bin yscv-cli -- --validate-diagnostics-report benchmarks/diagnostics-sample-ok.json --validate-diagnostics-min-frames 2 --validate-diagnostics-max-drift-pct 25 --validate-diagnostics-max-dropped 0
  run cargo run -p yscv-cli --bin yscv-cli -- --benchmark --max-frames 3 --benchmark-report artifacts/benchmark.txt --benchmark-baseline benchmarks/ci-baseline-people.txt
  run cargo run -p yscv-cli --bin yscv-cli -- --detect-target face --benchmark --max-frames 3 --benchmark-report artifacts/benchmark-face.txt --benchmark-baseline benchmarks/ci-baseline-face.txt
  run bash benchmarks/export-runtime-trend.sh artifacts/benchmark-runtime-trend.tsv artifacts/benchmark.txt artifacts/benchmark-face.txt artifacts/eval-dataset-output.txt
  run bash benchmarks/compare-trend-snapshots.sh runtime benchmarks/trend-baseline-runtime.tsv artifacts/benchmark-runtime-trend.tsv artifacts/benchmark-runtime-trend-diff.tsv

  gates=(
    "yscv-detect detect_heatmap_scratch benchmark-detect-micro ci-baseline-detect-micro"
    "yscv-track tracker_update benchmark-track-micro ci-baseline-track-micro"
    "yscv-recognize recognize_slice_vs_tensor benchmark-recognize-micro ci-baseline-recognize-micro"
    "yscv-video normalize_rgb8 benchmark-video-micro ci-baseline-video-micro"
    "yscv-tensor tensor_elementwise_modes benchmark-tensor-micro ci-baseline-tensor-micro"
    "yscv-kernels kernels_cpu_ops benchmark-kernels-micro ci-baseline-kernels-micro"
    "yscv-model model_runtime_ops benchmark-model-micro ci-baseline-model-micro"
    "yscv-imgproc imgproc_ops benchmark-imgproc-micro ci-baseline-imgproc-micro"
    "yscv-autograd autograd_graph_ops benchmark-autograd-micro ci-baseline-autograd-micro"
    "yscv-eval eval_metrics_ops benchmark-eval-micro ci-baseline-eval-micro"
    "yscv-cli cli_runtime_ops benchmark-cli-micro ci-baseline-cli-micro"
    "camera-face-tool camera_face_runtime_ops benchmark-camera-face-tool-micro ci-baseline-camera-face-tool-micro"
  )
  for gate in "${gates[@]}"; do
    set -- $gate
    run bash benchmarks/run-criterion-gate.sh "$1" "$2" "artifacts/$3.txt" "benchmarks/$4.txt"
  done

  run bash benchmarks/export-criterion-trend.sh \
    artifacts/benchmark-micro-trend.tsv \
    artifacts/benchmark-detect-micro.txt \
    artifacts/benchmark-track-micro.txt \
    artifacts/benchmark-recognize-micro.txt \
    artifacts/benchmark-video-micro.txt \
    artifacts/benchmark-tensor-micro.txt \
    artifacts/benchmark-kernels-micro.txt \
    artifacts/benchmark-model-micro.txt \
    artifacts/benchmark-imgproc-micro.txt \
    artifacts/benchmark-autograd-micro.txt \
    artifacts/benchmark-eval-micro.txt \
    artifacts/benchmark-cli-micro.txt \
    artifacts/benchmark-camera-face-tool-micro.txt
  run bash benchmarks/compare-trend-snapshots.sh micro benchmarks/trend-baseline-micro.tsv artifacts/benchmark-micro-trend.tsv artifacts/benchmark-micro-trend-diff.tsv
  run bash benchmarks/check-trend-baseline-freshness.sh benchmarks/trend-baseline-micro.tsv "$trend_baseline_max_age_days"
  run bash benchmarks/check-trend-baseline-freshness.sh benchmarks/trend-baseline-runtime.tsv "$trend_baseline_max_age_days"
fi

echo
echo "local CI subset passed"
