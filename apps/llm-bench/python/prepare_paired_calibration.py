#!/usr/bin/env python3
"""Build yscv-quantize multi-input calibration JSONL from paired .npy arrays.

Example:
  python apps/llm-bench/python/prepare_paired_calibration.py \
    --inputs input.1:tmpl.npy,input.249:search.npy \
    --output tracker_calib.jsonl

Each .npy must have a leading sample dimension, e.g. [N,1,3,128,128]
and [N,1,3,256,256]. The script writes one JSON object per sample:
{"input.1": {"shape": [...], "values": [...]}, "input.249": ...}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_inputs(spec: str) -> list[tuple[str, Path]]:
    out: list[tuple[str, Path]] = []
    for item in spec.split(","):
        if ":" not in item:
            raise SystemExit(f"bad --inputs entry {item!r}: expected NAME:path.npy")
        name, path = item.split(":", 1)
        if not name or not path:
            raise SystemExit(f"bad --inputs entry {item!r}: expected NAME:path.npy")
        out.append((name, Path(path)))
    if not out:
        raise SystemExit("--inputs must contain at least one NAME:path.npy entry")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True, help="NAME:file.npy[,NAME:file.npy]*")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    arrays: list[tuple[str, np.ndarray]] = []
    for name, path in parse_inputs(args.inputs):
        arr = np.load(path).astype(np.float32, copy=False)
        if arr.ndim < 2:
            raise SystemExit(f"{path}: expected leading sample dimension plus tensor dims")
        arrays.append((name, arr))

    n = arrays[0][1].shape[0]
    for name, arr in arrays:
        if arr.shape[0] != n:
            raise SystemExit(f"{name}: {arr.shape[0]} samples, expected {n}")

    with open(args.output, "w", encoding="utf-8") as f:
        for i in range(n):
            sample = {}
            for name, arr in arrays:
                tensor = np.ascontiguousarray(arr[i], dtype=np.float32)
                sample[name] = {
                    "shape": list(tensor.shape),
                    "values": tensor.reshape(-1).tolist(),
                }
            f.write(json.dumps(sample, separators=(",", ":")) + "\n")

    print(f"wrote {n} sample(s) to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
