#!/usr/bin/env python3
"""ONNX Runtime tracker bench with per-node Chrome-trace profile.

Same shape/CLI as private/private/onnx-fps/bench_ort_onnx_fps.py with
two additions:
    --emit-profile PATH    write ORT's per-node profile JSON here
    --strip-warmup N       drop the first N iterations from the trace

Used by D1 of the gap-diag plan to produce a per-op profile that can
be aligned against YSCV's `YSCV_RUNNER_PROFILE` JSON.

Run via:
    nix-shell -p 'python3.withPackages(ps: with ps; [ onnxruntime numpy onnx ])' \
        --run "python3 scripts/gap_diag/bench_ort_profile.py \
            --model private/private/model.onnx \
            --input input.1:1x3x128x128 --input input.249:1x3x256x256 \
            --iters 500 --warmup 50 --threads 1 \
            --emit-profile /tmp/gap_diag/ort_profile.json"
"""
from __future__ import annotations
import argparse
import json
import shutil
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import onnxruntime as ort


def parse_input_spec(spec: str) -> Tuple[str, List[int]]:
    if ":" not in spec:
        raise ValueError(f"bad --input '{spec}': expected NAME:DxDxD")
    name, shape_s = spec.split(":", 1)
    dims = [int(d) for d in shape_s.split("x") if d]
    if not name or not dims:
        raise ValueError(f"bad --input '{spec}'")
    return name, dims


def percentile_us(sorted_samples_us, p):
    if not sorted_samples_us:
        return 0
    idx = max(0, min(int(round((len(sorted_samples_us) - 1) * p)), len(sorted_samples_us) - 1))
    return sorted_samples_us[idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--input", action="append", default=[])
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--emit-profile", default=None,
                    help="write per-node Chrome-trace profile JSON here")
    args = ap.parse_args()

    parsed_inputs = [parse_input_spec(s) for s in args.input]

    so = ort.SessionOptions()
    so.intra_op_num_threads = args.threads
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if args.emit_profile:
        so.enable_profiling = True
        Path(args.emit_profile).parent.mkdir(parents=True, exist_ok=True)
        so.profile_file_prefix = str(Path(args.emit_profile).with_suffix(""))

    sess = ort.InferenceSession(
        args.model,
        sess_options=so,
        providers=["CPUExecutionProvider"],
    )

    feed: Dict[str, np.ndarray] = {}
    for i, (name, shape) in enumerate(parsed_inputs):
        rng = np.random.default_rng(args.seed + i)
        feed[name] = rng.random(shape, dtype=np.float32)

    print(f"Model:    {args.model}", file=sys.stderr)
    print(f"Threads:  {args.threads}  iters={args.iters} warmup={args.warmup}",
          file=sys.stderr)

    for _ in range(args.warmup):
        sess.run(None, feed)

    samples_us: List[int] = []
    for _ in range(args.iters):
        t0 = time.perf_counter_ns()
        sess.run(None, feed)
        samples_us.append((time.perf_counter_ns() - t0) // 1000)
    samples_us.sort()

    mn = samples_us[0]
    p50 = percentile_us(samples_us, 0.50)
    avg = float(statistics.mean(samples_us))
    p95 = percentile_us(samples_us, 0.95)

    print(f"min={mn/1000:.3f}ms p50={p50/1000:.3f}ms avg={avg/1000:.3f}ms p95={p95/1000:.3f}ms",
          file=sys.stderr)

    if args.emit_profile:
        # `end_profiling` returns the actual path ORT wrote.
        actual_path = sess.end_profiling()
        target = Path(args.emit_profile)
        if Path(actual_path).resolve() != target.resolve():
            shutil.move(actual_path, target)
        print(f"profile: {target}", file=sys.stderr)

    payload = {
        "model": args.model,
        "backend": "ort-cpu",
        "threads": args.threads,
        "iters": args.iters,
        "warmup": args.warmup,
        "min_us": mn,
        "p50_us": p50,
        "avg_us": avg,
        "p95_us": p95,
    }
    print(json.dumps(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
