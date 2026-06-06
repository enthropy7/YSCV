#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

pattern='std::is_x86_feature_detected|std::arch::is_aarch64_feature_detected|is_x86_feature_detected!|is_aarch64_feature_detected!'
violations="$(
  rg -n --glob '*.rs' "$pattern" crates/yscv-kernels/src \
    | rg -v '^crates/yscv-kernels/src/arch/detect_(x86|aarch64)\.rs:' || true
)"

if [ -n "$violations" ]; then
  cat >&2 <<'EOF'
error: raw runtime feature detection must stay inside yscv-kernels/src/arch/detect_*.

Use crate::host_cpu().features or a named CpuFeatures helper from ops/tests.
Violations:
EOF
  printf '%s\n' "$violations" >&2
  exit 1
fi

echo "runtime dispatch guard: OK"
