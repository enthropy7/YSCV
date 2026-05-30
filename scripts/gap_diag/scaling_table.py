#!/usr/bin/env python3
"""Per-op 1T vs 6T scaling table.

Consumes two YSCV profile JSONs (1T and 6T) and emits per-node:
    name, 1T µs, 6T µs, ratio (1T/6T), lost-speedup-µs

Sorted by `lost-speedup-µs = 1T_µs * (1 - ratio/6)` — the contribution
each op makes to the gap between actual scaling and ideal 6× scaling.
Nodes at the top of the list are the biggest contributors to poor
6T scaling.

Usage:
    python3 scripts/gap_diag/scaling_table.py \\
        /tmp/gap_diag/yscv_post_a1a3.json \\
        /tmp/gap_diag/yscv_6t_post_m1m2.json
"""

from __future__ import annotations
import json
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) < 3:
        print("usage: scaling_table.py <yscv_1t.json> <yscv_6t.json>", file=sys.stderr)
        return 2
    one_t = json.loads(Path(sys.argv[1]).read_text())
    six_t = json.loads(Path(sys.argv[2]).read_text())

    one = {n["name"]: n["ms"] * 1000 for n in one_t["nodes"]}
    six = {n["name"]: n["ms"] * 1000 for n in six_t["nodes"]}

    rows: list[tuple[str, float, float, float, float]] = []
    for name, t1 in one.items():
        t6 = six.get(name, 0.0)
        ratio = (t1 / t6) if t6 > 0.001 else float("inf")
        # Ideal 6T would be t1 / 6. Lost-speedup = how much extra 6T pays
        # relative to ideal scaling, capped at t1 (single-threaded floor).
        lost = max(0.0, t6 - t1 / 6.0) if t6 > 0 else 0.0
        rows.append((name, t1, t6, ratio, lost))

    # Sort by lost-speedup descending — biggest contributors to 6T gap first.
    rows.sort(key=lambda r: -r[4])

    print(f"{'node':70} {'1T µs':>10} {'6T µs':>10} {'ratio':>7} {'lost µs':>9}")
    print("-" * 110)
    sum_lost = 0.0
    sum_1t = 0.0
    sum_6t = 0.0
    for name, t1, t6, ratio, lost in rows[:30]:
        rr = f"{ratio:.2f}×" if ratio != float("inf") else " inf"
        short = name.replace("/connect_model", "")[-70:]
        print(f"{short:70} {t1:>10.1f} {t6:>10.1f} {rr:>7} {lost:>9.1f}")
        sum_lost += lost

    for _, t1, t6, _, lost in rows:
        sum_1t += t1
        sum_6t += t6
        sum_lost = sum_lost  # already counted; recompute below

    sum_lost_all = sum(r[4] for r in rows)
    total_ratio = sum_1t / sum_6t if sum_6t > 0 else float("inf")
    print("-" * 110)
    print(f"Total 1T sum:     {sum_1t:>10.1f} µs")
    print(f"Total 6T sum:     {sum_6t:>10.1f} µs")
    print(f"Aggregate ratio:  {total_ratio:>10.2f}× (ideal 6.00×)")
    print(f"Total lost-µs:    {sum_lost_all:>10.1f} µs (vs ideal 6T)")
    print(f"Top-30 lost-µs:   {sum_lost:>10.1f} µs ({sum_lost / sum_lost_all * 100:.0f}% of total)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
