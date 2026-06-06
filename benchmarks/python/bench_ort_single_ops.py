#!/usr/bin/env python3
"""ONNX Runtime CPU-EP microbench for the single-op compute-gap shapes.

Run via:
    nix-shell -p 'python3.withPackages(ps: with ps; [ onnxruntime numpy onnx ])' \
        --run "python3 benchmarks/python/bench_ort_single_ops.py --iters 500"
"""
from __future__ import annotations

import argparse
import statistics
import time
from typing import Dict, Iterable, List, Tuple

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper


OpSpec = Tuple[str, str, List[int]]


OPS: List[OpSpec] = [
    ("add_1M", "add", [1024, 1024]),
    ("mul_1M", "mul", [1024, 1024]),
    ("exp_1M", "exp", [1024, 1024]),
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


def init(name: str, value: np.ndarray) -> onnx.TensorProto:
    return numpy_helper.from_array(value.astype(np.float32), name=name)


def make_model(op: str, shape: List[int]) -> bytes:
    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, shape)
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, shape)
    inputs = [x]
    initializers: List[onnx.TensorProto] = []
    nodes: List[onnx.NodeProto] = []

    if op == "add":
        inputs.append(helper.make_tensor_value_info("B", TensorProto.FLOAT, shape))
        nodes.append(helper.make_node("Add", ["X", "B"], ["Y"]))
    elif op == "mul":
        inputs.append(helper.make_tensor_value_info("B", TensorProto.FLOAT, shape))
        nodes.append(helper.make_node("Mul", ["X", "B"], ["Y"]))
    elif op in ("exp", "relu", "sigmoid", "tanh"):
        nodes.append(helper.make_node(op.capitalize() if op != "relu" else "Relu", ["X"], ["Y"]))
    elif op == "softmax":
        nodes.append(helper.make_node("Softmax", ["X"], ["Y"], axis=-1))
    elif op == "log_softmax":
        nodes.append(helper.make_node("LogSoftmax", ["X"], ["Y"], axis=-1))
    elif op == "silu":
        nodes.append(helper.make_node("Sigmoid", ["X"], ["S"]))
        nodes.append(helper.make_node("Mul", ["X", "S"], ["Y"]))
    elif op == "gelu_sigmoid":
        initializers.extend(
            [
                init("c1702", np.array([1.702], dtype=np.float32)),
            ]
        )
        nodes.extend(
            [
                helper.make_node("Mul", ["X", "c1702"], ["A"]),
                helper.make_node("Sigmoid", ["A"], ["S"]),
                helper.make_node("Mul", ["X", "S"], ["Y"]),
            ]
        )
    elif op == "layer_norm":
        initializers.extend(
            [
                init("scale", np.ones((shape[-1],), dtype=np.float32)),
                init("bias", np.zeros((shape[-1],), dtype=np.float32)),
            ]
        )
        nodes.append(
            helper.make_node(
                "LayerNormalization",
                ["X", "scale", "bias"],
                ["Y"],
                axis=-1,
                epsilon=1e-5,
            )
        )
    elif op == "batch_norm":
        channels = shape[1]
        initializers.extend(
            [
                init("scale", np.ones((channels,), dtype=np.float32)),
                init("bias", np.zeros((channels,), dtype=np.float32)),
                init("mean", np.zeros((channels,), dtype=np.float32)),
                init("var", np.ones((channels,), dtype=np.float32)),
            ]
        )
        nodes.append(
            helper.make_node(
                "BatchNormalization",
                ["X", "scale", "bias", "mean", "var"],
                ["Y"],
                epsilon=1e-5,
                training_mode=0,
            )
        )
    else:
        raise ValueError(f"unknown op: {op}")

    graph = helper.make_graph(nodes, f"ort_single_{op}", inputs, [y], initializer=initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    return model.SerializeToString()


def make_session(op: str, shape: List[int], threads: int, rng: np.random.Generator):
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(make_model(op, shape), sess_options=so, providers=["CPUExecutionProvider"])

    feed: Dict[str, np.ndarray] = {
        "X": rng.uniform(-3.0, 3.0, size=shape).astype(np.float32),
    }
    if op in ("add", "mul"):
        feed["B"] = rng.uniform(-1.0, 1.0, size=shape).astype(np.float32)
    return sess, feed


def sample_stats(samples_us: Iterable[int]) -> Tuple[int, int, int]:
    samples = sorted(samples_us)
    return samples[0], samples[len(samples) // 2], int(statistics.fmean(samples))


def bench(op: str, shape: List[int], threads: int, iters: int, warmup: int, seed: int) -> Tuple[int, int, int]:
    rng = np.random.default_rng(seed)
    sess, feed = make_session(op, shape, threads, rng)
    for _ in range(warmup):
        sess.run(None, feed)

    samples = []
    for _ in range(iters):
        t0 = time.perf_counter_ns()
        sess.run(None, feed)
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

    threads = args.threads or [1, 12]
    print(f"ort={ort.__version__} iters={args.iters} warmup={args.warmup}")
    for thread_count in threads:
        print(f"\nthreads={thread_count}")
        print(f"{'operation':34s} {'min':>9s} {'p50':>9s} {'avg':>9s}")
        print("-" * 66)
        for index, (label, op, shape) in enumerate(OPS):
            if args.filter is not None and args.filter not in label:
                continue
            mn, p50, avg = bench(op, shape, thread_count, args.iters, args.warmup, args.seed + index)
            print(f"{label:34s} min={mn:6d}us p50={p50:6d}us avg={avg:6d}us")


if __name__ == "__main__":
    main()
