#!/usr/bin/env python3
"""Phase 0 gap attribution: join yscv + ORT JSON profiles, output per-component table.

Usage:
    python3 scripts/gap_attribution.py \
        --yscv-1t /tmp/yscv_profiles/yscv_1t_run1.json \
        --yscv-6t /tmp/yscv_profiles/yscv_6t_run2.json \
        --ort-1t  /tmp/yscv_profiles/ort_1t.json \
        --ort-6t  /tmp/yscv_profiles/ort_6t.json

Output format: Markdown table with yscv/ORT sequential sum at 1T and 6T, gap, and % of
total gap. Also prints parallel-efficiency per op class (1T/6T ratio).

ORT name normalization: strips _nchwc, _output_0, /Relu_output_0, /Conv_output_0 and
trailing token numbers. yscv fused names (PW+DW) split on '+' to find sub-op prefixes.
"""

import argparse
import json
import re
from collections import defaultdict


def load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def ort_canonical(name: str) -> str:
    """Normalize ORT op name to a key matchable against yscv op names."""
    n = name
    n = re.sub(r'_nchwc$', '', n)
    n = re.sub(r'/Relu_output_0$', '', n)
    n = re.sub(r'/Conv_output_0$', '', n)
    n = re.sub(r'_token_\d+$', '', n)
    n = re.sub(r'_output_0$', '', n)
    n = n.rstrip('/')
    return n


def yscv_sub_keys(yscv_name: str) -> list[str]:
    """Extract sub-op keys from a yscv fused name like 'a/Conv+b/Conv'."""
    parts = yscv_name.split('+')
    keys = []
    for p in parts:
        # strip trailing /Conv, /relu, etc.
        k = re.sub(r'/(Conv|Relu|relu)[^/]*$', '', p)
        keys.append(k)
    return keys


def aggregate_by_op_class(nodes: list[dict], label: str) -> dict[str, float]:
    totals: dict[str, float] = defaultdict(float)
    for n in nodes:
        op = n['op']
        # Simplify ORT op names
        if label == 'ort':
            if op in ('ReorderInput', 'ReorderOutput'):
                op = 'Layout'
            elif op == 'FusedMatMul':
                op = 'MatMul'
        # Simplify yscv op names
        if label == 'yscv':
            if op in ('FusedPwDw', 'FusedDwPw', 'Conv', 'Conv_Relu', 'Conv_Add_fused',
                      'Conv_Add_Relu_fused', 'Conv_Relu_Add_fused'):
                op = 'Conv (all)'
            elif op == 'FusedTransposeMatMul':
                op = 'MatMul'
        totals[op] += n['ms']
    return dict(totals)


def op_class_table(
    yscv_1t_nodes, yscv_6t_nodes, ort_1t_nodes, ort_6t_nodes,
    yscv_1t_total, yscv_6t_total, ort_1t_total, ort_6t_total,
) -> str:
    y1 = aggregate_by_op_class(yscv_1t_nodes, 'yscv')
    y6 = aggregate_by_op_class(yscv_6t_nodes, 'yscv')
    o1 = aggregate_by_op_class(ort_1t_nodes, 'ort')
    o6 = aggregate_by_op_class(ort_6t_nodes, 'ort')

    all_ops = sorted(set(y1) | set(o1), key=lambda k: -(y1.get(k, 0) + o1.get(k, 0)))

    total_gap_1t = yscv_1t_total - ort_1t_total
    total_gap_6t = yscv_6t_total - ort_6t_total

    lines = []
    lines.append('### Op-class attribution (sequential sums, profiled runs)\n')
    lines.append(
        '| Class | yscv 1T | ORT 1T | gap 1T | yscv 6T | ORT 6T | gap 6T |'
        ' yscv scale | ORT scale |'
    )
    lines.append(
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|'
    )
    for op in all_ops:
        y1v = y1.get(op, 0.0)
        y6v = y6.get(op, 0.0)
        o1v = o1.get(op, 0.0)
        o6v = o6.get(op, 0.0)
        g1 = y1v - o1v
        g6 = y6v - o6v
        ys = y1v / y6v if y6v > 0 else float('nan')
        os = o1v / o6v if o6v > 0 else float('nan')
        lines.append(
            f'| {op} | {y1v:.3f} | {o1v:.3f} | {g1:+.3f} |'
            f' {y6v:.3f} | {o6v:.3f} | {g6:+.3f} | {ys:.2f}× | {os:.2f}× |'
        )
    # Totals
    g1t = yscv_1t_total - ort_1t_total
    g6t = yscv_6t_total - ort_6t_total
    yt = yscv_1t_total / yscv_6t_total if yscv_6t_total > 0 else float('nan')
    ot = ort_1t_total / ort_6t_total if ort_6t_total > 0 else float('nan')
    lines.append(
        f'| **Total** | **{yscv_1t_total:.3f}** | **{ort_1t_total:.3f}** | **{g1t:+.3f}** |'
        f' **{yscv_6t_total:.3f}** | **{ort_6t_total:.3f}** | **{g6t:+.3f}** |'
        f' **{yt:.2f}×** | **{ot:.2f}×** |'
    )
    return '\n'.join(lines)


def flop_saturation_table(yscv_1t_nodes) -> str:
    """Compute FLOP efficiency for top FusedPwDw ops given shapes in profile."""
    # Zen 4 Ryzen 5 7500F: 2 AVX-512 FMA units × 16 SP/FMA × 2 FLOPs/FMA × ~4.7 GHz
    PEAK_GFLOPS_1T = 2 * 16 * 2 * 4.7  # ≈ 300.8 GFLOPS

    lines = []
    lines.append('\n### Microkernel saturation — FusedPwDw PW component (1T)\n')
    lines.append('Shape: [N, H, W, C_in] → [N, H, W, C_out] (M=H·W, K=C_in, N=C_out)\n')
    lines.append('| Op | shape (in→out) | M | K | N | MFLOP | yscv 1T ms | GFLOPS | % peak |')
    lines.append('|---|---|---:|---:|---:|---:|---:|---:|---:|')

    nodes = [n for n in yscv_1t_nodes if n['op'] == 'FusedPwDw' and n.get('in_shape') and n.get('out_shape')]
    nodes = sorted(nodes, key=lambda n: -n['ms'])

    for n in nodes[:12]:
        ishape = n['in_shape']
        oshape = n['out_shape']
        if len(ishape) < 4 or len(oshape) < 4:
            continue
        h, w, c_in = ishape[1], ishape[2], ishape[3]
        c_out = oshape[3]
        M = h * w
        K = c_in
        N = c_out
        # PW FLOPs = 2 * M * K * N, DW FLOPs = 2 * M * C_out * 9
        pw_flops = 2 * M * K * N
        dw_flops = 2 * M * c_out * 9
        total_mflop = (pw_flops + dw_flops) / 1e6
        ms = n['ms']
        gflops = total_mflop / ms * 1000  # ms→s: gflops = mflop / (ms*1e-3) / 1e9 = mflop/(ms*1e-6)
        gflops = total_mflop / (ms * 1e-3) / 1e3  # GFLOP/s
        pct = gflops / PEAK_GFLOPS_1T * 100
        name = n['name'].split('+')[0].split('/')[-3] if '/' in n['name'] else n['name']
        lines.append(
            f'| {name[:20]} | [{h},{w},{c_in}]→[{h},{w},{c_out}] | {M} | {K} | {N} |'
            f' {total_mflop:.1f} | {ms:.3f} | {gflops:.1f} | {pct:.0f}% |'
        )

    lines.append(f'\nPeak reference: {PEAK_GFLOPS_1T:.0f} GFLOPS (Zen 4, 2× AVX-512 FMA @ 4.7 GHz, 1T)')
    return '\n'.join(lines)


def per_block_table(yscv_1t_nodes, ort_1t_nodes) -> str:
    """Per block-group gap (xif2_0, xif3_0, xif4_5, etc.)."""
    groups = ['xif0_', 'xif1_', 'xif2_', 'xif3_', 'xif4_', 'connect_model', 'neck']
    lines = []
    lines.append('\n### Per-block gap (1T, all ops in block)\n')
    lines.append('| Block | yscv ms | ORT ms | ratio |')
    lines.append('|---|---:|---:|---:|')

    for g in groups:
        yn = [n for n in yscv_1t_nodes if g in n['name']]
        on = [n for n in ort_1t_nodes if g in n['name']]
        yt = sum(n['ms'] for n in yn)
        ot = sum(n['ms'] for n in on)
        if yt < 0.01 and ot < 0.01:
            continue
        ratio = yt / ot if ot > 0 else float('nan')
        lines.append(f'| {g} | {yt:.3f} | {ot:.3f} | {ratio:.2f}× |')

    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--yscv-1t', required=True)
    parser.add_argument('--yscv-6t', required=True)
    parser.add_argument('--ort-1t', required=True)
    parser.add_argument('--ort-6t', required=True)
    parser.add_argument('--out', default=None, help='Write output to file (default: stdout)')
    args = parser.parse_args()

    y1 = load(args.yscv_1t)
    y6 = load(args.yscv_6t)
    o1 = load(args.ort_1t)
    o6 = load(args.ort_6t)

    output = []
    output.append('# Gap attribution — yscv vs ORT (Phase 0)\n')
    output.append(f'- yscv 1T: {y1["total_ms"]:.3f}ms | 6T profiled: {y6["total_ms"]:.3f}ms')
    output.append(f'- ORT  1T: {o1["total_ms"]:.3f}ms | 6T profiled: {o6["total_ms"]:.3f}ms')
    output.append(f'- Gap  1T: {(y1["total_ms"]/o1["total_ms"]):.2f}× | 6T profiled: {(y6["total_ms"]/o6["total_ms"]):.2f}×\n')

    output.append(op_class_table(
        y1['nodes'], y6['nodes'], o1['nodes'], o6['nodes'],
        y1['total_ms'], y6['total_ms'], o1['total_ms'], o6['total_ms'],
    ))
    output.append(flop_saturation_table(y1['nodes']))
    output.append(per_block_table(y1['nodes'], o1['nodes']))

    text = '\n'.join(output) + '\n'

    if args.out:
        with open(args.out, 'w') as f:
            f.write(text)
        print(f'Wrote {args.out}')
    else:
        print(text)


if __name__ == '__main__':
    main()
