#!/usr/bin/env bash
set -euo pipefail

# Publish all yscv crates to crates.io in dependency order.
# Usage: ./scripts/publish.sh [--dry-run]

DRY_RUN=""
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN="--dry-run"
    echo "Dry run mode — no actual publishing."
fi

# Publish order (leaves first, dependents last):
#   yscv-video-mpp is standalone (no yscv deps) — can sit anywhere.
#   yscv-pipeline depends on kernels + tensor + onnx + video so it
#   goes after onnx.
CRATES=(
    yscv-tensor
    yscv-autograd
    yscv-optim
    yscv-kernels
    yscv-imgproc
    yscv-eval
    yscv-video
    yscv-video-mpp
    yscv-onnx
    yscv-pipeline
    yscv-model
    yscv-detect
    yscv-recognize
    yscv-track
    yscv-cli
    yscv
)

for crate in "${CRATES[@]}"; do
    echo "Publishing $crate..."
    cargo publish -p "$crate" $DRY_RUN --allow-dirty
    if [[ -z "$DRY_RUN" ]]; then
        echo "Waiting for crates.io to index $crate..."
        sleep 30
    fi
done

echo "All crates published."
