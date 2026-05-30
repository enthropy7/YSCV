#!/usr/bin/env python3
"""Block-level alignment: groups both engines' events by ONNX layer block.

A "block" is the top-level segment of a node path, e.g.:
    /xif4_5/...         -> block "xif4_5"
    /connect_model/cls_dw/enc/enc.0/... -> block "connect_model/cls_dw"
    /xif0_0/...         -> block "xif0_0"

Within a block, sum the time across all our fused nodes vs all ORT's
NCHWc-fused nodes. This gives a clean apples-to-apples bucket regardless
of how each engine internally fuses ops.
"""
from __future__ import annotations
import json
import re
import sys
from collections import defaultdict
from pathlib import Path


def block_key(name: str) -> str:
    """Map a node path to its primary block label.

    /xif4_5/pw/conv_1 → xif4_5
    /connect_model/cls_dw/enc/enc.0/depthwise → connect_model/cls_dw
    /connect_model/Constant → connect_model/Constant (terminal name)
    """
    if not name.startswith("/"):
        return name
    parts = [p for p in name.split("/") if p]
    if not parts:
        return "?"
    # connect_model has nested heads (cls_dw, reg_dw, bbox_pred…).
    if parts[0] == "connect_model" and len(parts) >= 2:
        return f"connect_model/{parts[1]}"
    return parts[0]


def main():
    yscv = json.loads(Path(sys.argv[1]).read_text())
    ort_events = json.loads(Path(sys.argv[2]).read_text())

    # YSCV: aggregate by block
    yscv_block = defaultdict(lambda: {"us": 0.0, "ops": defaultdict(int), "nodes": 0})
    for n in yscv["nodes"]:
        # For fused nodes (name contains "+"), use the first sub-name's block —
        # both halves are always in the same block in tracker, so this works.
        first_sub = n["name"].split("+")[0]
        b = block_key(first_sub)
        yscv_block[b]["us"] += n["ms"] * 1000
        yscv_block[b]["ops"][n["op"]] += 1
        yscv_block[b]["nodes"] += 1

    # ORT: aggregate per-event by block
    ort_block = defaultdict(lambda: {"us": 0.0, "ops": defaultdict(int), "nodes": set()})
    iters = 0
    session_total = 0
    for e in ort_events:
        cat = e.get("cat")
        if cat == "Session" and e.get("name") == "SequentialExecutor::Execute":
            iters += 1
            session_total += e.get("dur", 0)
            continue
        if cat != "Node":
            continue
        name = e.get("name", "")
        if not name.endswith("_kernel_time"):
            continue
        b = block_key(name)
        op = e.get("args", {}).get("op_name", "?")
        rec = ort_block[b]
        rec["us"] += e.get("dur", 0)
        rec["ops"][op] += 1
        rec["nodes"].add(name)
    if iters == 0:
        iters = 1
    # Normalize ORT to per-inference
    for rec in ort_block.values():
        rec["us"] /= iters
        rec["nodes"] = len(rec["nodes"])

    yscv_total = yscv["total_ms"] * 1000
    ort_total = session_total / iters if iters else 0

    print("# Block-level alignment\n")
    print(f"**Totals:** YSCV {yscv_total:.0f} µs/inf  ORT {ort_total:.0f} µs/inf  "
          f"gap = +{yscv_total - ort_total:.0f} µs ({yscv_total / ort_total:.2f}×)\n")
    print(f"_ORT iters detected: {iters}_\n")

    print("## Per-block time (sorted by YSCV-ORT delta, descending)\n")
    print("| block | yscv µs | yscv nodes | ort µs | ort nodes | delta µs |")
    print("|---|---:|---:|---:|---:|---:|")

    all_blocks = sorted(set(yscv_block) | set(ort_block),
                        key=lambda b: -(yscv_block.get(b, {"us": 0})["us"]
                                        - ort_block.get(b, {"us": 0})["us"]))
    sum_delta = 0.0
    for b in all_blocks:
        y = yscv_block.get(b, {"us": 0.0, "nodes": 0})
        o = ort_block.get(b, {"us": 0.0, "nodes": 0})
        delta = y["us"] - o["us"]
        sum_delta += delta
        print(f"| `{b}` | {y['us']:.0f} | {y['nodes']} | {o['us']:.0f} | {o['nodes']} | "
              f"{delta:+.0f} |")
    print(f"\n**Sum of per-block deltas: {sum_delta:+.0f} µs** "
          f"(should ≈ YSCV total − ORT kernel-sum total)")

    # Op-count diff per block (where YSCV ≠ ORT op count)
    print("\n## Notable op-count differences\n")
    print("| block | yscv ops | ort ops |")
    print("|---|---|---|")
    for b in sorted(set(yscv_block) | set(ort_block)):
        y_ops = dict(yscv_block.get(b, {}).get("ops", {}))
        o_ops = dict(ort_block.get(b, {}).get("ops", {}))
        if y_ops == o_ops:
            continue
        y_str = ", ".join(f"{k}×{v}" for k, v in sorted(y_ops.items()))
        o_str = ", ".join(f"{k}×{v}" for k, v in sorted(o_ops.items()))
        print(f"| `{b}` | {y_str} | {o_str} |")


if __name__ == "__main__":
    main()
