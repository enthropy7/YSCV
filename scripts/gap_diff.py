#!/usr/bin/env python3
"""Compare yscv profiling JSON against ORT profiling JSON, emit per-op gap.

Reads two JSON files in the shape emitted by:
  * `profile_onnx_model_cpu` when `YSCV_PROFILE_JSON=path` is set
  * `bench_ort.py --emit-profile path`

Both formats share `{"engine","total_ms","nodes":[{"name","op","ms",...}]}`.
The report groups by `op` (op_type), sums timings per engine, prints a table
sorted by gap_ms descending, and classifies the overall gap as
MT_OVERHEAD / KERNEL / GRAPH.

Usage:
    python scripts/gap_diff.py yscv.json ort.json
    python scripts/gap_diff.py yscv.json ort.json --out report.md
    # only Conv / a named subgraph:
    python scripts/gap_diff.py yscv.json ort.json --filter Conv,name:head
    # per-node diff of two same-engine runs (baseline vs change), showing any
    # dispatched-kernel change (old→new) the yscv JSON recorded:
    python scripts/gap_diff.py base.json change.json --per-node --filter Conv

Both JSONs must represent the same model. ORT fuses Conv+Relu and uses its
own op names (e.g. "FusedMatMul" vs "MatMul"); `--alias` flags let you
bucket them together for the summary. `--filter` takes the same spec as the
profiler's `YSCV_PROFILE_FILTER`.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


# ORT emits its own op-type names after graph optimization. Bucket them
# against our yscv op names so Conv_Relu (yscv fusion) groups with Conv
# (ORT's post-fusion Conv). Keep this conservative — false groupings
# make the gap attribution wrong.
DEFAULT_ALIASES: dict[str, str] = {
    # yscv-side fusions → base op
    "Conv_Relu": "Conv",
    "SiLU(fused)": "SiLU",
    # ORT-side artifacts → base op
    "FusedMatMul": "MatMul",
    "FusedConv": "Conv",
    # ORT layout shims — these ARE work, but they exist only because ORT
    # uses NCHWc internally. We keep them in their own "Reorder" bucket
    # rather than merging into Conv, so the report shows the NCHWc tax.
    "ReorderInput": "Reorder",
    "ReorderOutput": "Reorder",
}


def normalize_op(op: str, aliases: dict[str, str]) -> str:
    return aliases.get(op, op)


def load(path: Path) -> dict:
    with open(path) as f:
        d = json.load(f)
    required = {"engine", "total_ms", "nodes"}
    missing = required - set(d)
    if missing:
        sys.exit(f"{path}: missing keys {missing}")
    return d


def group_by_op(nodes: list[dict], aliases: dict[str, str]) -> dict[str, tuple[float, int]]:
    agg: dict[str, tuple[float, int]] = defaultdict(lambda: (0.0, 0))
    for n in nodes:
        key = normalize_op(n.get("op", "?"), aliases)
        t, c = agg[key]
        agg[key] = (t + float(n.get("ms", 0.0)), c + 1)
    return agg


def parse_filter(spec: str) -> tuple[set[str], list[str]]:
    """Parse a `YSCV_PROFILE_FILTER`-style spec: comma-separated tokens, a bare
    token matches an op type exactly, `name:<substr>` a node-name substring."""
    ops: set[str] = set()
    name_subs: list[str] = []
    for tok in (t.strip() for t in spec.split(",")):
        if not tok:
            continue
        if tok.startswith("name:"):
            name_subs.append(tok[len("name:"):])
        else:
            ops.add(tok)
    return ops, name_subs


def matches_filter(node: dict, ops: set[str], name_subs: list[str]) -> bool:
    if not ops and not name_subs:
        return True
    return node.get("op", "") in ops or any(
        s in node.get("name", "") for s in name_subs
    )


def format_per_node_report(
    a: dict,
    b: dict,
    threads_hint: str | None = None,
) -> str:
    """Per-node diff matched by node name. Best for two same-engine runs
    (e.g. baseline vs change) where names line up; then the yscv-side
    `kernel` field also surfaces dispatch changes (a→b)."""
    a_by_name = {n.get("name"): n for n in a["nodes"]}
    b_by_name = {n.get("name"): n for n in b["nodes"]}
    names = set(a_by_name) | set(b_by_name)

    rows = []
    for name in names:
        an = a_by_name.get(name)
        bn = b_by_name.get(name)
        a_ms = float(an["ms"]) if an else 0.0
        b_ms = float(bn["ms"]) if bn else 0.0
        op = (an or bn).get("op", "?")
        # Kernel annotation: show a dispatch change as `old→new` when the two
        # sides disagree, else the single kernel. Only yscv JSONs carry it.
        ak = an.get("kernel") if an else None
        bk = bn.get("kernel") if bn else None
        if ak and bk and ak != bk:
            kernel = f"{ak}→{bk}"
        else:
            kernel = ak or bk or ""
        meta = an or bn or {}
        ks = meta.get("kernel_shape")
        st = meta.get("strides")
        rows.append((name, op, a_ms, b_ms, a_ms - b_ms, kernel, ks, st))
    rows.sort(key=lambda r: -r[4])

    lines: list[str] = ["# gap_diff per-node report"]
    if threads_hint:
        lines.append(f"*Threads: {threads_hint}*")
    lines.append("")
    lines.append(f"- A total: {a['total_ms']:.2f} ms ({len(a['nodes'])} nodes)")
    lines.append(f"- B total: {b['total_ms']:.2f} ms ({len(b['nodes'])} nodes)")
    lines.append("")
    lines.append("| node | op | a_ms | b_ms | Δms | kernel | k | s |")
    lines.append("|---|---|---:|---:|---:|---|---|---|")
    for name, op, a_ms, b_ms, dms, kernel, ks, st in rows:
        ks_s = "×".join(str(x) for x in ks) if ks else ""
        st_s = "×".join(str(x) for x in st) if st else ""
        lines.append(
            f"| {name} | {op} | {a_ms:.3f} | {b_ms:.3f} | {dms:+.3f} | "
            f"{kernel} | {ks_s} | {st_s} |"
        )
    return "\n".join(lines) + "\n"


def classify(yscv_total: float, ort_total: float, threads_hint: str | None) -> str:
    """Heuristic classifier — the real decision uses BOTH 1T and 6T runs
    compared against each other. This single-run version just flags whether
    the gap exists and hands off rationale to the report reader.
    """
    if ort_total <= 0:
        return "UNKNOWN (ORT total is zero)"
    ratio = yscv_total / ort_total
    if ratio < 1.05:
        return f"CLOSE — yscv within 5% of ORT (ratio {ratio:.2f}×)"
    return f"GAP — yscv is {ratio:.2f}× slower than ORT on this run"


def format_report(
    yscv: dict,
    ort: dict,
    aliases: dict[str, str],
    threads_hint: str | None = None,
) -> str:
    yscv_agg = group_by_op(yscv["nodes"], aliases)
    ort_agg = group_by_op(ort["nodes"], aliases)
    all_ops = sorted(set(yscv_agg) | set(ort_agg))

    rows: list[tuple[str, float, int, float, int, float, float]] = []
    for op in all_ops:
        y_ms, y_n = yscv_agg.get(op, (0.0, 0))
        o_ms, o_n = ort_agg.get(op, (0.0, 0))
        gap_ms = y_ms - o_ms
        ratio = (y_ms / o_ms) if o_ms > 0 else float("inf")
        rows.append((op, y_ms, y_n, o_ms, o_n, gap_ms, ratio))
    rows.sort(key=lambda r: -r[5])

    yscv_total = yscv["total_ms"]
    ort_total = ort["total_ms"]
    classification = classify(yscv_total, ort_total, threads_hint)

    lines: list[str] = []
    lines.append("# gap_diff report")
    if threads_hint:
        lines.append(f"*Threads: {threads_hint}*")
    lines.append("")
    lines.append(f"- yscv total: {yscv_total:.2f} ms ({len(yscv['nodes'])} nodes)")
    lines.append(f"- ORT  total: {ort_total:.2f} ms ({len(ort['nodes'])} nodes)")
    lines.append(f"- classification: {classification}")
    lines.append("")
    lines.append("| op | yscv_ms | yscv_n | ort_ms | ort_n | gap_ms | ratio |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for op, y_ms, y_n, o_ms, o_n, gap_ms, ratio in rows:
        ratio_s = f"{ratio:.2f}×" if ratio != float("inf") else "∞"
        lines.append(
            f"| {op} | {y_ms:.2f} | {y_n} | {o_ms:.2f} | {o_n} | {gap_ms:+.2f} | {ratio_s} |"
        )

    # Contribution column — what fraction of the gap comes from this op.
    lines.append("")
    lines.append("## Gap contribution (top 10)")
    total_gap = sum(r[5] for r in rows if r[5] > 0)
    if total_gap > 0:
        positives = sorted([r for r in rows if r[5] > 0], key=lambda r: -r[5])[:10]
        lines.append("| op | gap_ms | % of positive gap |")
        lines.append("|---|---:|---:|")
        for op, _, _, _, _, gap_ms, _ in positives:
            pct = 100.0 * gap_ms / total_gap
            lines.append(f"| {op} | {gap_ms:+.2f} | {pct:5.1f}% |")

    return "\n".join(lines) + "\n"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("yscv_json", type=Path)
    p.add_argument("ort_json", type=Path)
    p.add_argument("--out", type=Path, default=None,
                   help="write report to PATH (otherwise stdout)")
    p.add_argument("--threads", default=None,
                   help="threads hint for report header (e.g. '1', '6')")
    p.add_argument("--alias", action="append", default=[],
                   help="extra alias pairs, e.g. --alias FusedConv=Conv")
    p.add_argument("--filter", default="",
                   help="restrict to nodes matching a YSCV_PROFILE_FILTER-style "
                        "spec (op types and/or name:<substr>), e.g. 'Conv,name:head'")
    p.add_argument("--per-node", action="store_true",
                   help="diff per node by name instead of grouping by op type; "
                        "best for two same-engine runs (baseline vs change)")
    args = p.parse_args()

    aliases = dict(DEFAULT_ALIASES)
    for spec in args.alias:
        if "=" not in spec:
            sys.exit(f"bad --alias {spec!r}; expected FROM=TO")
        src, dst = spec.split("=", 1)
        aliases[src] = dst

    yscv = load(args.yscv_json)
    ort = load(args.ort_json)

    if args.filter:
        ops, name_subs = parse_filter(args.filter)
        yscv["nodes"] = [n for n in yscv["nodes"] if matches_filter(n, ops, name_subs)]
        ort["nodes"] = [n for n in ort["nodes"] if matches_filter(n, ops, name_subs)]

    if args.per_node:
        report = format_per_node_report(yscv, ort, args.threads)
    else:
        report = format_report(yscv, ort, aliases, args.threads)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report)
        print(f"wrote {args.out}", file=sys.stderr)
    else:
        print(report, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
