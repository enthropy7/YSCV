#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

pattern='std::is_x86_feature_detected|std::arch::is_aarch64_feature_detected|is_x86_feature_detected!|is_aarch64_feature_detected!'
violations="$(
  rg -n --glob '*.rs' "$pattern" crates \
    | rg -v '^crates/yscv-cpu/src/detect_(x86|aarch64)\.rs:' || true
)"

if [ -n "$violations" ]; then
  cat >&2 <<'EOF'
error: raw runtime feature detection must stay inside yscv-cpu/src/detect_*.

Use yscv_cpu::host_cpu().features, a crate-local re-export, or a named CpuFeatures helper.
Violations:
EOF
  printf '%s\n' "$violations" >&2
  exit 1
fi

echo "runtime dispatch guard: OK"
