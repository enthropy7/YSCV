#!/usr/bin/env bash
set -euo pipefail

# scripts/check-safety-comments.sh
#
# Phase 3 of the 1.0 roadmap: BLOCKER and SHOULD-FIX files must have a
# `// SAFETY:` comment immediately above every `unsafe { ... }` block.
# This script enforces that contract on the files listed below. New
# unsafe blocks added without a SAFETY comment will fail the gate.
#
# A SAFETY comment may live in the lines directly above the `unsafe`
# block, optionally separated by:
#   - blank lines
#   - other comment lines belonging to the same SAFETY block
#   - the `let xxx =` head of a `let xxx = unsafe { ... };` statement
#
# This matches what `clippy::undocumented_unsafe_blocks` accepts but is
# stricter about coverage (clippy only fires on top-level statements).

cd "$(dirname "$0")/.."

# Files under SAFETY enforcement.
FILES=(
    "crates/yscv-video/src/hw_decode.rs"
    "crates/yscv-kernels/src/metal_backend.rs"
    "crates/yscv-onnx/src/runner/metal/run.rs"
    "crates/yscv-imgproc/src/ops/u8_features.rs"
    "crates/yscv-imgproc/src/ops/color.rs"
    "crates/yscv-imgproc/src/ops/u8_filters.rs"
    "crates/yscv-imgproc/src/ops/f32_ops.rs"
    "crates/yscv-imgproc/src/ops/fast.rs"
    "crates/yscv-imgproc/src/ops/features.rs"
)
if [ ${#FILES[@]} -eq 0 ]; then
    echo "OK: no files under SAFETY enforcement (list empty)."
    exit 0
fi

python3 - "${FILES[@]}" <<'PY'
import sys

def is_real_unsafe_block(line):
    """True if this source line contains an `unsafe { ... }` opener that
    isn't a declaration (`unsafe fn` etc.) and isn't text inside a
    line comment."""
    s = line.strip()
    if not s:
        return False
    # Strip line comments first — anything after `//` is text, not code.
    code = s.split('//', 1)[0]
    if 'unsafe {' not in code:
        return False
    # Skip declarations.
    code_stripped = code.strip()
    for prefix in ('unsafe extern', 'unsafe fn', 'unsafe impl', 'unsafe trait'):
        if code_stripped.startswith(prefix):
            return False
    return True

failures = []
for path in sys.argv[1:]:
    with open(path) as f:
        lines = f.read().splitlines()
    for i, line in enumerate(lines):
        if not is_real_unsafe_block(line):
            continue
        # Walk back over (a) blank lines, (b) `let xxx =` heads,
        # (c) consecutive comment lines, looking for `// SAFETY:`.
        j = i - 1
        had_safety = False
        in_comment_block = False
        steps = 0
        while j >= 0 and steps < 30:
            ls = lines[j].strip()
            if ls == '':
                j -= 1; steps += 1; continue
            if ls.startswith('//'):
                in_comment_block = True
                if 'SAFETY' in ls:
                    had_safety = True
                    break
                j -= 1; steps += 1; continue
            if in_comment_block:
                break  # comment block ended without SAFETY
            if ls.startswith('let ') or ls.endswith('='):
                j -= 1; steps += 1; continue
            break
        if not had_safety:
            failures.append((path, i + 1, line.strip()))

if failures:
    print(f"FAIL: {len(failures)} unsafe block(s) missing // SAFETY: comment")
    print()
    for path, ln, code in failures:
        print(f"  {path}:{ln}: {code}")
    print()
    print("Add a `// SAFETY: …` comment immediately above each block")
    print("explaining the invariants the unsafe code relies on.")
    sys.exit(1)

total = 0
for path in sys.argv[1:]:
    with open(path) as f:
        for line in f:
            if is_real_unsafe_block(line):
                total += 1
print(f"OK: every one of {total} unsafe block(s) across {len(sys.argv) - 1} file(s) has a // SAFETY: comment.")
PY
