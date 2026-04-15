#!/usr/bin/env bash
set -euo pipefail

# scripts/check-doc-counts.sh
#
# Documentation drift gate. Asserts that hand-pinned counts in this script
# still match what `grep` says about the current source. When the source
# legitimately changes, bump the constant in the same commit so the README,
# context.md, docs/, and per-crate READMEs can be updated atomically.
#
# This script is the source-of-truth for the numbers we publish. Any number
# that appears in user-facing prose should also have a row here.

# Always run from the repo root regardless of caller's CWD.
cd "$(dirname "$0")/.."

# ── Expected counts ──────────────────────────────────────────────────
EXPECTED_CRATES=16
EXPECTED_VERSION="0.1.8"
EXPECTED_ONNX_OPS=130
EXPECTED_TENSOR_METHODS=156          # ops.rs (115) + tensor.rs (32) + linalg.rs (9)
EXPECTED_IMGPROC_FNS=160             # standalone pub fn in crates/yscv-imgproc/src/ops/
EXPECTED_AUTOGRAD_OPS=61             # variants of pub(crate) enum Op
EXPECTED_MODEL_LAYERS=39             # variants of pub enum ModelLayer
EXPECTED_LOSS_FNS=17                 # pub fn items in crates/yscv-model/src/loss.rs
EXPECTED_OPTIMIZERS=8                # one file per optimizer
EXPECTED_LR_SCHEDULERS=11
EXPECTED_MODEL_ZOO=17                # variants of pub enum ModelArchitecture
EXPECTED_WGSL_SHADERS=61
EXPECTED_METAL_SHADERS=4
EXPECTED_BACKEND_METHODS=27          # required methods on `pub trait Backend`

# ── Actual counts (greps) ────────────────────────────────────────────
actual_crates=$(find crates -mindepth 1 -maxdepth 1 -type d | wc -l | tr -d ' ')
actual_version=$(awk -F\" '/^version = "/{print $2; exit}' Cargo.toml)
actual_onnx_ops=$(awk '/match node.op_type/,/^}$/' crates/yscv-onnx/src/runner/mod.rs \
    | grep -cE '^[[:space:]]+"[A-Z]')
tensor_ops=$(grep -c '^    pub fn' crates/yscv-tensor/src/ops.rs)
tensor_main=$(grep -c '^    pub fn' crates/yscv-tensor/src/tensor.rs)
tensor_linalg=$(grep -c '^    pub fn' crates/yscv-tensor/src/linalg.rs)
actual_tensor_methods=$((tensor_ops + tensor_main + tensor_linalg))
actual_imgproc_fns=$(grep -h '^pub fn' crates/yscv-imgproc/src/ops/*.rs | wc -l | tr -d ' ')
actual_autograd_ops=$(awk '/enum Op \{/,/^}/' crates/yscv-autograd/src/node.rs \
    | grep -cE '^    [A-Z]')
actual_model_layers=$(awk '/enum ModelLayer/,/^}/' crates/yscv-model/src/layers/mod.rs \
    | grep -cE '^    [A-Z]')
actual_loss_fns=$(grep -c '^pub fn ' crates/yscv-model/src/loss.rs)
actual_optimizers=$(ls crates/yscv-optim/src/{sgd,adam,adamw,adagrad,radam,rmsprop,lamb,lars}.rs 2>/dev/null \
    | wc -l | tr -d ' ')
actual_lr_schedulers=$(grep -c '^pub struct' crates/yscv-optim/src/scheduler.rs)
actual_model_zoo=$(awk '/enum ModelArchitecture/,/^}/' crates/yscv-model/src/zoo.rs \
    | grep -cE '^    [A-Z]')
actual_wgsl=$(find crates/yscv-kernels/src/shaders -name '*.wgsl' | wc -l | tr -d ' ')
actual_metal=$(find crates/yscv-kernels/src/shaders -name '*.metal' | wc -l | tr -d ' ')
actual_backend_methods=$(awk '/pub trait Backend/,/^}/' crates/yscv-kernels/src/backend.rs \
    | grep -cE '^    fn ')

# ── Comparison harness ───────────────────────────────────────────────
fail=0
check() {
    local label="$1"
    local expected="$2"
    local actual="$3"
    if [[ "$expected" != "$actual" ]]; then
        printf 'FAIL  %-22s expected %s, got %s\n' "$label" "$expected" "$actual"
        fail=1
    else
        printf 'OK    %-22s = %s\n' "$label" "$actual"
    fi
}

check "crates"            "$EXPECTED_CRATES"          "$actual_crates"
check "workspace-version" "$EXPECTED_VERSION"         "$actual_version"
check "onnx-cpu-ops"      "$EXPECTED_ONNX_OPS"        "$actual_onnx_ops"
check "tensor-methods"    "$EXPECTED_TENSOR_METHODS"  "$actual_tensor_methods"
check "imgproc-fns"       "$EXPECTED_IMGPROC_FNS"     "$actual_imgproc_fns"
check "autograd-ops"      "$EXPECTED_AUTOGRAD_OPS"    "$actual_autograd_ops"
check "model-layers"      "$EXPECTED_MODEL_LAYERS"    "$actual_model_layers"
check "loss-functions"    "$EXPECTED_LOSS_FNS"        "$actual_loss_fns"
check "optimizers"        "$EXPECTED_OPTIMIZERS"      "$actual_optimizers"
check "lr-schedulers"     "$EXPECTED_LR_SCHEDULERS"   "$actual_lr_schedulers"
check "model-zoo"         "$EXPECTED_MODEL_ZOO"       "$actual_model_zoo"
check "wgsl-shaders"      "$EXPECTED_WGSL_SHADERS"    "$actual_wgsl"
check "metal-shaders"     "$EXPECTED_METAL_SHADERS"   "$actual_metal"
check "backend-methods"   "$EXPECTED_BACKEND_METHODS" "$actual_backend_methods"

if [[ "$fail" -ne 0 ]]; then
    echo
    echo "Documentation counts have drifted from source. Update both this"
    echo "script's expected values and the user-facing docs (root README,"
    echo "docs/, per-crate READMEs, context.md) in the same commit."
    exit 1
fi

echo
echo "All documentation counts match source."
