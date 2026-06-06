#!/usr/bin/env python3
"""PyTorch CPU microbench for the single-op compute-gap shapes.

Run via:
    nix-shell -p 'python3.withPackages(ps: with ps; [ torch numpy ])' \
        --run "python3 benchmarks/python/bench_torch_single_ops.py --iters 500"
"""
from __future__ import annotations

import argparse
import statistics
import time
from typing import Callable, Iterable, List, Tuple

import torch


OpSpec = Tuple[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor], List[int]]


def sample_stats(samples_us: Iterable[int]) -> Tuple[int, int, int]:
    samples = sorted(samples_us)
    return samples[0], samples[len(samples) // 2], int(statistics.fmean(samples))


def make_input(shape: List[int], seed: int) -> torch.Tensor:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    return (torch.rand(shape, generator=gen, dtype=torch.float32) * 6.0) - 3.0


def bench(
    op: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    shape: List[int],
    threads: int,
    iters: int,
    warmup: int,
    seed: int,
) -> Tuple[int, int, int]:
    torch.set_num_threads(threads)
    x = make_input(shape, seed)
    y = make_input(shape, seed + 10_000)

    with torch.no_grad():
        for _ in range(warmup):
            z = op(x, y)
            if z.numel() == 0:
                raise RuntimeError("unexpected empty output")
        samples = []
        for _ in range(iters):
            t0 = time.perf_counter_ns()
            z = op(x, y)
            # Keep the result alive across the timed region.
            if z.numel() == 0:
                raise RuntimeError("unexpected empty output")
            samples.append((time.perf_counter_ns() - t0) // 1000)
    return sample_stats(samples)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--threads", type=int, action="append", default=None)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--filter", default=None)
    args = ap.parse_args()
    torch.set_num_interop_threads(1)

    ops: List[OpSpec] = [
        ("add_1M", lambda x, y: x + y, [1024, 1024]),
        ("mul_1M", lambda x, y: x * y, [1024, 1024]),
        ("exp_1M", lambda x, _y: torch.exp(x), [1024, 1024]),
        ("relu_921K", lambda x, _y: torch.relu(x), [921600]),
        ("sigmoid_921K", lambda x, _y: torch.sigmoid(x), [921600]),
        ("tanh_1M", lambda x, _y: torch.tanh(x), [1024, 1024]),
        ("gelu_1M_sigmoid_formula", lambda x, _y: x * torch.sigmoid(1.702 * x), [1024, 1024]),
        ("silu_1M", lambda x, _y: torch.nn.functional.silu(x), [1024, 1024]),
        ("softmax_32x1000", lambda x, _y: torch.softmax(x, dim=-1), [32, 1000]),
        ("log_softmax_32x1000", lambda x, _y: torch.log_softmax(x, dim=-1), [32, 1000]),
        ("softmax_512x256", lambda x, _y: torch.softmax(x, dim=-1), [512, 256]),
        (
            "layer_norm_512x256",
            lambda x, _y: torch.nn.functional.layer_norm(
                x,
                normalized_shape=(256,),
                weight=torch.ones(256, dtype=torch.float32),
                bias=torch.zeros(256, dtype=torch.float32),
                eps=1e-5,
            ),
            [512, 256],
        ),
        (
            "batch_norm_1x3x64x64",
            lambda x, _y: torch.nn.functional.batch_norm(
                x,
                running_mean=torch.zeros(3, dtype=torch.float32),
                running_var=torch.ones(3, dtype=torch.float32),
                weight=torch.ones(3, dtype=torch.float32),
                bias=torch.zeros(3, dtype=torch.float32),
                training=False,
                eps=1e-5,
            ),
            [1, 3, 64, 64],
        ),
    ]

    threads = args.threads or [1, 12]
    print(f"torch={torch.__version__} iters={args.iters} warmup={args.warmup}")
    for thread_count in threads:
        print(f"\nthreads={thread_count}")
        print(f"{'operation':34s} {'min':>9s} {'p50':>9s} {'avg':>9s}")
        print("-" * 66)
        for index, (label, op, shape) in enumerate(ops):
            if args.filter is not None and args.filter not in label:
                continue
            mn, p50, avg = bench(op, shape, thread_count, args.iters, args.warmup, args.seed + index)
            print(f"{label:34s} min={mn:6d}us p50={p50:6d}us avg={avg:6d}us")


if __name__ == "__main__":
    main()
