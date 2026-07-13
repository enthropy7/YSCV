#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def load_results(root: Path):
    results = {}
    for path in root.rglob("*.json"):
        if path.parent.name not in {"base", "head"}:
            continue
        side = path.parent.name
        data = json.loads(path.read_text())
        if "case" not in data or "median_p50_us" not in data:
            continue
        key = (data["arch"], data["case"])
        results.setdefault(key, {})[side] = data
    return results


def pct(head, base):
    if base == 0:
        return 0.0
    return (head - base) * 100.0 / base


def fmt_us(v):
    if v >= 1000:
        return f"{v / 1000.0:.2f} ms"
    return f"{v} us"


def compact_counts(counts, limit=6):
    items = sorted(counts.items(), key=lambda item: (-int(item[1]), item[0]))
    shown = [f"`{name}` x{count}" for name, count in items[:limit]]
    extra = len(items) - len(shown)
    if extra > 0:
        shown.append(f"+{extra} more")
    return ", ".join(shown) if shown else "-"


def fmt_cost_delta(base, head):
    return f"{base:,} → {head:,} ({pct(head, base):+.1f}%)"


def graph_cost_changed(base, head):
    fields = (
        "node_count",
        "known_nodes",
        "unknown_nodes",
        "estimated_macs",
        "estimated_element_ops",
        "estimated_bytes_read",
        "estimated_bytes_written",
        "score",
    )
    return all(field in base and field in head for field in fields) and any(
        base[field] != head[field] for field in fields
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    results = load_results(args.artifact_root)
    lines = [
        "## ONNX CPU benchmark",
        "",
        "GitHub-hosted runner signal, advisory only. Base and head are compared within the same architecture job.",
    ]

    graph_cost_lines = []
    for (arch, case), pair in sorted(results.items()):
        base = pair.get("base", {}).get("graph_cost")
        head = pair.get("head", {}).get("graph_cost")
        if (
            not isinstance(base, dict)
            or not isinstance(head, dict)
            or not graph_cost_changed(base, head)
        ):
            continue
        graph_cost_lines.append(
            f"| `{arch}` | `{case}` | {fmt_cost_delta(base['node_count'], head['node_count'])} | "
            f"{fmt_cost_delta(base['estimated_macs'], head['estimated_macs'])} | "
            f"{fmt_cost_delta(base['estimated_element_ops'], head['estimated_element_ops'])} | "
            f"{fmt_cost_delta(base['estimated_bytes_read'] + base['estimated_bytes_written'], head['estimated_bytes_read'] + head['estimated_bytes_written'])} | "
            f"{fmt_cost_delta(base['score'], head['score'])} |"
        )
    if graph_cost_lines:
        lines.extend(
            [
                "",
                "### Graph-cost changes",
                "",
                "| arch | case | nodes | MACs | element ops | bytes | score |",
                "|---|---|---:|---:|---:|---:|---:|",
                *graph_cost_lines,
            ]
        )

    lines.extend(
        [
            "",
            "| arch | case | base p50 | head p50 | delta |",
            "|---|---|---:|---:|---:|",
        ]
    )
    missing = []
    for (arch, case), pair in sorted(results.items()):
        if "base" not in pair or "head" not in pair:
            missing.append(f"{arch}/{case}")
            continue
        base = int(pair["base"]["median_p50_us"])
        head = int(pair["head"]["median_p50_us"])
        delta = pct(head, base)
        marker = "faster" if delta < -1.0 else "slower" if delta > 1.0 else "flat"
        lines.append(
            f"| `{arch}` | `{case}` | {fmt_us(base)} | {fmt_us(head)} | {delta:+.1f}% {marker} |"
        )

    kernel_lines = [
        "",
        "### Head op/kernel summary",
        "",
        "| arch | case | ops | dispatched kernels |",
        "|---|---|---|---|",
    ]
    for (arch, case), pair in sorted(results.items()):
        head = pair.get("head")
        if not head:
            continue
        summary = head.get("profile_summary", {})
        ops = compact_counts(summary.get("op_counts", {}))
        kernels = compact_counts(summary.get("op_kernel_counts", {}))
        kernel_lines.append(f"| `{arch}` | `{case}` | {ops} | {kernels} |")
    lines.extend(kernel_lines)

    if missing:
        lines.extend(["", "Missing paired result(s): " + ", ".join(sorted(missing))])

    lines.extend(
        [
            "",
            "Raw JSON artifacts include per-run min/p50/avg/p95/p99, optimized node counts, graph-cost summaries, yscv CPU dispatch report, and per-node runner profile summaries.",
        ]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
