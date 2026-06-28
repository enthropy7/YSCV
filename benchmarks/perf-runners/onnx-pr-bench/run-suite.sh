#!/usr/bin/env bash
set -euo pipefail

asset_dir="${1:?usage: run-suite.sh ASSET_DIR OUT_DIR}"
out_dir="${2:?usage: run-suite.sh ASSET_DIR OUT_DIR}"

runner_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
suite="${YSCV_PR_BENCH_SUITE:-$runner_dir/suite.json}"
mkdir -p "$asset_dir" "$out_dir"

python3 - "$runner_dir" "$suite" "$asset_dir" "$out_dir" <<'PY'
import json
import os
import subprocess
import sys
from pathlib import Path

runner_dir = Path(sys.argv[1])
suite_path = Path(sys.argv[2])
asset_dir = Path(sys.argv[3])
out_dir = Path(sys.argv[4])
suite = json.loads(suite_path.read_text())

manifest = runner_dir / "Cargo.toml"
base_cmd = ["cargo", "run", "--release", "--manifest-path", str(manifest), "--"]

subprocess.run(
    base_cmd + ["prepare-small", "--asset-dir", str(asset_dir)],
    check=True,
)

defaults = suite.get("defaults", {})
runs = int(os.environ.get("YSCV_PR_BENCH_RUNS", defaults.get("runs", 3)))
threads = int(os.environ.get("YSCV_PR_BENCH_THREADS", defaults.get("threads", 0)))

for case in suite.get("cases", []):
    iters = int(os.environ.get(case.get("iters_env", ""), case.get("iters", 100)))
    args = [
        "run-case",
        "--case",
        case["name"],
        "--model",
        str(asset_dir / case["model"]),
        "--iters",
        str(iters),
        "--runs",
        str(runs),
        "--threads",
        str(threads),
        "--output",
        str(out_dir / f"{case['name']}.json"),
    ]

    fill = case.get("fill")
    if fill is None and any(i.get("source") == "random" for i in case.get("inputs", [])):
        fill = "random"
    if fill:
        args += ["--fill", fill]

    for inp in case.get("inputs", []):
        shape = "x".join(str(d) for d in inp["shape"])
        args += ["--input", f"{inp['name']}:{shape}"]

    if "image" in case:
        args += ["--image", str(asset_dir / case["image"])]
        if "image_input" in case:
            args += ["--image-input", case["image_input"]]
        if "image_size" in case:
            args += ["--image-size", str(case["image_size"])]

    subprocess.run(base_cmd + args, check=True)
PY
