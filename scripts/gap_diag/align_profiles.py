#!/usr/bin/env python3
"""Align YSCV's per-node profile against ORT's Chrome-trace profile.

Inputs:
    /tmp/gap_diag/yscv_profile.json   produced by `YSCV_RUNNER_PROFILE=...`
    /tmp/gap_diag/ort_profile.json    produced by `bench_ort_profile.py --emit-profile`

Output (stdout, markdown):
    - bucket summary (total µs per op family in each engine)
    - per-node aligned table (top 30 by yscv_us)
    - unmatched-on-each-side lists

A YSCV "fused" node like `A+B` matches ORT events whose node-name (stripped of
suffix `_kernel_time` and any layout decoration like `_nchwc`) equals A or B.
The fused bucket gets compared to (sum of ORT events matching its sub-names).
"""
from __future__ import annotations
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

KERNEL_TIME_SUFFIX = "_kernel_time"
NCHWC_SUFFIX = "_nchwc"
FENCE_SUFFIXES = ("_fence_before", "_fence_after")


def load_yscv(path: Path):
    """Parse YSCV profile: returns (per_inference_us_total, nodes_list).

    Each node has: name, op, us_per_inf, in_shape, out_shape.
    """
    d = json.loads(path.read_text())
    iters = d.get("iterations", 1)
    nodes = []
    for n in d["nodes"]:
        nodes.append({
            "name": n["name"],
            "op": n["op"],
            "us_per_inf": n["ms"] * 1000,  # ms is per-inference avg
            "in_shape": n.get("in_shape", []),
            "out_shape": n.get("out_shape", []),
            "count": n.get("count", iters),
        })
    return d.get("total_ms", 0) * 1000, nodes


def strip_ort_suffix(name: str) -> str:
    """Strip `_kernel_time` and any layout decoration to recover the
    original ONNX node name."""
    if name.endswith(KERNEL_TIME_SUFFIX):
        name = name[: -len(KERNEL_TIME_SUFFIX)]
    # Layout-transform-rewritten nodes get a layout suffix BEFORE _kernel_time.
    # Examples: `_nchwc`, `_output_0_nchwc`. Strip common variants.
    name = re.sub(r"_output_\d+_nchwc$", "", name)
    name = re.sub(r"_nchwc$", "", name)
    name = re.sub(r"_output_\d+$", "", name)  # `..._output_0`
    return name


def load_ort(path: Path):
    """Parse ORT Chrome trace: returns (per_inference_us_total, per_node_map).

    per_node_map[original_name] = {
        "op_name": str,
        "us_total": float,  # over the iters
        "calls": int,
        "out_shape": list,
        "in_shape": list,
    }
    """
    events = json.loads(path.read_text())
    nodes_us = defaultdict(lambda: {
        "op_name": None,
        "us_total": 0.0,
        "calls": 0,
        "out_shape": [],
        "in_shape": [],
    })
    session_total_us = 0.0
    inferences = 0
    for e in events:
        cat = e.get("cat")
        if cat == "Session":
            if e.get("name") == "SequentialExecutor::Execute":
                session_total_us += e.get("dur", 0)
                inferences += 1
            continue
        if cat != "Node":
            continue
        name = e.get("name", "")
        if not name.endswith(KERNEL_TIME_SUFFIX):
            continue  # skip fence_before/after — those are graph-level overhead
        if any(s in name for s in FENCE_SUFFIXES):
            continue
        original = strip_ort_suffix(name)
        rec = nodes_us[original]
        rec["us_total"] += e.get("dur", 0)
        rec["calls"] += 1
        args = e.get("args", {})
        if rec["op_name"] is None:
            rec["op_name"] = args.get("op_name", "?")
        out_shape = args.get("output_type_shape", [])
        in_shape = args.get("input_type_shape", [])
        if not rec["out_shape"] and out_shape:
            rec["out_shape"] = out_shape
        if not rec["in_shape"] and in_shape:
            rec["in_shape"] = in_shape

    # Total iters = inferences if we found SequentialExecutor markers,
    # else fall back to per-node call counts.
    if inferences == 0 and nodes_us:
        # Median call count across nodes is a robust iter estimate.
        counts = sorted(rec["calls"] for rec in nodes_us.values() if rec["calls"] > 0)
        inferences = counts[len(counts) // 2] if counts else 1

    per_inf_total = session_total_us / inferences if inferences else 0
    per_node = {}
    for name, rec in nodes_us.items():
        per_node[name] = {
            "op_name": rec["op_name"],
            "us_per_inf": rec["us_total"] / inferences if inferences else 0,
            "calls_per_inf": rec["calls"] / inferences if inferences else 0,
            "out_shape": rec["out_shape"],
            "in_shape": rec["in_shape"],
        }
    return per_inf_total, per_node, inferences


def split_fused_name(name: str):
    """Split a YSCV fused node name (e.g. `A+B`) into component ONNX names."""
    return name.split("+")


def main():
    if len(sys.argv) < 3:
        print("usage: align_profiles.py YSCV_JSON ORT_JSON", file=sys.stderr)
        sys.exit(1)
    yscv_total, yscv_nodes = load_yscv(Path(sys.argv[1]))
    ort_total, ort_nodes, ort_iters = load_ort(Path(sys.argv[2]))

    # ---- Bucket summary ----
    yscv_by_op = defaultdict(lambda: {"us": 0.0, "count": 0})
    for n in yscv_nodes:
        b = yscv_by_op[n["op"]]
        b["us"] += n["us_per_inf"]
        b["count"] += 1

    ort_by_op = defaultdict(lambda: {"us": 0.0, "count": 0})
    for name, rec in ort_nodes.items():
        b = ort_by_op[rec["op_name"]]
        b["us"] += rec["us_per_inf"]
        b["count"] += 1

    print("# Gap diagnostic — aligned per-op profile (1T, 500 iters)\n")
    print(f"**Totals:** YSCV {yscv_total:.0f} µs/inf  vs  ORT {ort_total:.0f} µs/inf "
          f"(gap = {yscv_total - ort_total:+.0f} µs)\n")
    print(f"ORT iters detected from profile: {ort_iters}\n")

    print("## Bucket totals\n")
    print("| op family | YSCV µs/inf | YSCV nodes | ORT µs/inf | ORT nodes |")
    print("|---|---:|---:|---:|---:|")
    # Sort by YSCV's biggest buckets first
    for op, b in sorted(yscv_by_op.items(), key=lambda kv: -kv[1]["us"]):
        ort_b = ort_by_op.get(op, {"us": 0.0, "count": 0})
        # YSCV's fused ops don't exist in ORT — show YSCV value with `—` on ORT.
        ort_us = f"{ort_b['us']:.0f}" if ort_b["count"] else "—"
        ort_n = f"{ort_b['count']}" if ort_b["count"] else "—"
        print(f"| {op} | {b['us']:.0f} | {b['count']} | {ort_us} | {ort_n} |")
    # ORT-only buckets
    for op, b in sorted(ort_by_op.items(), key=lambda kv: -kv[1]["us"]):
        if op in yscv_by_op:
            continue
        print(f"| {op} (ORT only) | — | — | {b['us']:.0f} | {b['count']} |")

    # ---- Per-node alignment (top YSCV nodes by us) ----
    print("\n## Top 30 YSCV nodes vs matched ORT total\n")
    print("| YSCV node | op | yscv µs | ort sub-nodes | ort µs | ratio | out_shape |")
    print("|---|---|---:|---|---:|---:|---|")
    sorted_yscv = sorted(yscv_nodes, key=lambda n: -n["us_per_inf"])[:30]
    for n in sorted_yscv:
        subs = split_fused_name(n["name"])
        ort_us_sum = 0.0
        sub_names = []
        for s in subs:
            rec = ort_nodes.get(s)
            if rec:
                ort_us_sum += rec["us_per_inf"]
                sub_names.append(f"{rec['op_name']}({rec['us_per_inf']:.0f})")
            else:
                sub_names.append(f"?({s.split('/')[-1]})")
        ratio = (n["us_per_inf"] / ort_us_sum) if ort_us_sum else float("inf")
        out_shape = n.get("out_shape") or []
        out_str = "x".join(str(d) for d in out_shape)
        ort_us_str = f"{ort_us_sum:.0f}" if ort_us_sum else "—"
        ratio_str = f"{ratio:.2f}×" if ort_us_sum else "—"
        node_short = "/".join(n["name"].split("/")[-2:]) if "/" in n["name"] else n["name"]
        sub_str = " + ".join(sub_names)
        print(f"| {node_short} | {n['op']} | {n['us_per_inf']:.0f} | {sub_str} | {ort_us_str} | {ratio_str} | {out_str} |")

    # ---- Unmatched ----
    yscv_subs = set()
    for n in yscv_nodes:
        yscv_subs.update(split_fused_name(n["name"]))
    ort_only = [(name, rec) for name, rec in ort_nodes.items() if name not in yscv_subs]
    ort_only.sort(key=lambda kv: -kv[1]["us_per_inf"])
    if ort_only:
        print("\n## Top 15 ORT-only nodes (no YSCV match)\n")
        print("| ORT node | op | µs |")
        print("|---|---|---:|")
        for name, rec in ort_only[:15]:
            short = "/".join(name.split("/")[-2:]) if "/" in name else name
            print(f"| {short} | {rec['op_name']} | {rec['us_per_inf']:.0f} |")


if __name__ == "__main__":
    main()
