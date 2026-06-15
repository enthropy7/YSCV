#!/usr/bin/env python3
"""NumPy CPU microbench for the single-op compute-gap shapes.

Mirrors ``bench_ort_single_ops.py`` (same op set, shapes, and output format) so
the single-compute harness can line NumPy up against YSCV / PyTorch / ORT.

Run via:
    python3 benchmarks/python/bench_numpy_single_ops.py --iters 1000 --threads 1
"""
from __future__ import annotations

import argparse
import time
from typing import Callable, List, Tuple

import numpy as np

OpSpec = Tuple[str, str, List[int]]

OPS: List[OpSpec] = [
    ("add_1M", "add", [1024, 1024]),
    ("mul_1M", "mul", [1024, 1024]),
    ("exp_1M", "exp", [1024, 1024]),
    ("sum_1M", "sum", [1024, 1024]),
    ("max_1M", "max", [1024, 1024]),
    ("add_broadcast_1024x1024_by_1024", "add_broadcast_lastdim", [1024, 1024]),
    ("sub_broadcast_1024_by_1024x1024", "sub_broadcast_row_by_mat", [1024, 1024]),
    ("relu_921K", "relu", [921600]),
    ("sigmoid_921K", "sigmoid", [921600]),
    ("tanh_1M", "tanh", [1024, 1024]),
    ("gelu_1M_sigmoid_graph", "gelu_sigmoid", [1024, 1024]),
    ("silu_1M_sigmoid_mul_graph", "silu", [1024, 1024]),
    ("softmax_32x1000", "softmax", [32, 1000]),
    ("log_softmax_32x1000", "log_softmax", [32, 1000]),
    ("softmax_512x256", "softmax", [512, 256]),
    ("layer_norm_512x256", "layer_norm", [512, 256]),
    ("batch_norm_1x3x64x64", "batch_norm", [1, 3, 64, 64]),
]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _softmax(x: np.ndarray) -> np.ndarray:
    m = x.max(axis=-1, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=-1, keepdims=True)


def make_op(op: str, shape: List[int], rng: np.random.Generator) -> Callable[[], np.ndarray]:
    x = rng.standard_normal(shape, dtype=np.float32)
    if op == "add":
        b = rng.standard_normal(shape, dtype=np.float32)
        return lambda: np.add(x, b)
    if op == "mul":
        b = rng.standard_normal(shape, dtype=np.float32)
        return lambda: np.multiply(x, b)
    if op == "exp":
        return lambda: np.exp(x)
    if op == "sum":
        return lambda: x.sum()
    if op == "max":
        return lambda: x.max()
    if op == "add_broadcast_lastdim":
        b = rng.standard_normal((shape[-1],), dtype=np.float32)
        return lambda: x + b
    if op == "sub_broadcast_row_by_mat":
        a = rng.standard_normal((shape[-1],), dtype=np.float32)
        return lambda: a - x
    if op == "relu":
        return lambda: np.maximum(x, np.float32(0))
    if op == "sigmoid":
        return lambda: _sigmoid(x)
    if op == "tanh":
        return lambda: np.tanh(x)
    if op == "gelu_sigmoid":
        return lambda: x * _sigmoid(np.float32(1.702) * x)
    if op == "silu":
        return lambda: x * _sigmoid(x)
    if op == "softmax":
        return lambda: _softmax(x)
    if op == "log_softmax":
        return lambda: x - x.max(axis=-1, keepdims=True) - np.log(
            np.exp(x - x.max(axis=-1, keepdims=True)).sum(axis=-1, keepdims=True)
        )
    if op == "layer_norm":
        g = rng.standard_normal((shape[-1],), dtype=np.float32)
        b = rng.standard_normal((shape[-1],), dtype=np.float32)
        eps = np.float32(1e-5)
        def f():
            mu = x.mean(axis=-1, keepdims=True)
            var = x.var(axis=-1, keepdims=True)
            return (x - mu) / np.sqrt(var + eps) * g + b
        return f
    if op == "batch_norm":
        c = shape[1]
        g = rng.standard_normal((c,), dtype=np.float32)
        b = rng.standard_normal((c,), dtype=np.float32)
        mean = rng.standard_normal((c,), dtype=np.float32)
        var = np.abs(rng.standard_normal((c,), dtype=np.float32)) + np.float32(1.0)
        eps = np.float32(1e-5)
        g_ = g.reshape(1, c, 1, 1)
        b_ = b.reshape(1, c, 1, 1)
        m_ = mean.reshape(1, c, 1, 1)
        v_ = var.reshape(1, c, 1, 1)
        return lambda: (x - m_) / np.sqrt(v_ + eps) * g_ + b_
    raise ValueError(f"unknown op {op}")


def bench(op: str, shape: List[int], iters: int, warmup: int, seed: int) -> Tuple[int, int, int]:
    rng = np.random.default_rng(seed)
    run = make_op(op, shape, rng)
    for _ in range(warmup):
        run()
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter_ns()
        run()
        samples.append((time.perf_counter_ns() - t0) // 1000)
    samples.sort()
    return samples[0], samples[len(samples) // 2], sum(samples) // len(samples)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--threads", type=int, action="append", default=None)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--filter", default=None)
    args = ap.parse_args()

    threads = args.threads or [1]
    print(f"numpy={np.__version__} iters={args.iters} warmup={args.warmup}")
    for thread_count in threads:
        print(f"\nthreads={thread_count}")
        print(f"{'operation':34s} {'min':>9s} {'p50':>9s} {'avg':>9s}")
        print("-" * 66)
        for index, (label, op, shape) in enumerate(OPS):
            if args.filter is not None and args.filter not in label:
                continue
            mn, p50, avg = bench(op, shape, args.iters, args.warmup, args.seed + index)
            print(f"{label:34s} min={mn:6d}us p50={p50:6d}us avg={avg:6d}us")


if __name__ == "__main__":
    main()
