#!/usr/bin/env bash
set -euo pipefail

# Publish all yscv crates to crates.io in dependency order.
# Usage: ./scripts/publish.sh [--dry-run]

DRY_RUN=""
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN="--dry-run"
    echo "Dry run mode — no actual publishing."
fi

# Topological publish order (leaves first, dependents last).
# Verified via `grep ^yscv- crates/*/Cargo.toml` — every crate appears
# after all of its yscv-* path deps.
CRATES=(
    yscv-tensor       # no yscv deps
    yscv-video-mpp    # no yscv deps, standalone
    yscv-threadpool   # no yscv deps (kernels + onnx depend on it)
    yscv-kernels      # <- tensor, threadpool
    yscv-imgproc      # <- tensor
    yscv-video        # <- tensor
    yscv-recognize    # <- tensor
    yscv-autograd     # <- kernels, tensor
    yscv-onnx         # <- kernels, tensor, threadpool
    yscv-quantize-cli # <- onnx, tensor (binary crate)
    yscv-optim        # <- autograd, tensor
    yscv-detect       # <- onnx, tensor, video
    yscv-pipeline     # <- kernels, tensor, onnx, (video optional)
    yscv-track        # <- detect, tensor
    yscv-eval         # <- detect, track
    yscv-model        # <- autograd, imgproc, kernels, onnx, optim, tensor
    yscv-cli          # <- detect, eval, recognize, track, tensor, video
    yscv              # <- umbrella, everything
)

for crate in "${CRATES[@]}"; do
    echo ""
    echo "=== Publishing $crate ==="
    # Capture stderr so we can detect "already published" and continue
    # idempotently — useful when resuming after a partial run.
    if output=$(cargo publish -p "$crate" $DRY_RUN --allow-dirty 2>&1); then
        echo "$output" | tail -5
        if [[ -z "$DRY_RUN" ]]; then
            echo "Waiting 30s for crates.io to index $crate..."
            sleep 30
        fi
    else
        # `cargo publish` exits non-zero. Check whether the reason is
        # "version already on crates.io" — if so, skip and continue.
        if echo "$output" | grep -qE "already (uploaded|exists)|is already uploaded"; then
            echo "↷ $crate @ this version already on crates.io — skipping."
            continue
        fi
        echo "$output" | tail -20
        echo "✗ Failed to publish $crate — aborting. Fix the error and re-run."
        exit 1
    fi
done

echo ""
echo "All crates published."
