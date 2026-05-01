#!/usr/bin/env python3
"""ONNXRuntime baseline for the cached-decoder LLM bench.

Mirrors yscv-llm-bench's prefill/decode timing so the two runs are
directly comparable. Reads the same prompt-token JSON, runs the same
forward passes, picks greedy next-token (top-k=1) the same way, and
emits the same one-line stderr summary plus a JSON record on stdout.

Usage:
    python bench_ort.py --model model.onnx --input-ids prompt.json \\
        --kv-shape 24,2,64 --max-tokens 16 --warmup 1

Pass --kv-shape NLAYERS,NHEADS,HEAD_DIM if the model has
past_key_values.* inputs (cached decoder mode); omit it for stateless
decoder graphs.
"""
import argparse
import json
import sys
import time

import numpy as np
import onnxruntime as ort


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--input-ids", required=True)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--kv-shape", default=None,
                    help="n_layers,n_heads,head_dim (cached decoder)")
    ap.add_argument("--with-attention-mask", action="store_true")
    ap.add_argument("--with-position-ids", action="store_true")
    ap.add_argument("--threads", type=int, default=0,
                    help="0 = ORT default; otherwise sets intra_op + inter_op")
    args = ap.parse_args()

    so = ort.SessionOptions()
    if args.threads > 0:
        so.intra_op_num_threads = args.threads
        so.inter_op_num_threads = args.threads
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    sess = ort.InferenceSession(
        args.model, sess_options=so, providers=["CPUExecutionProvider"]
    )

    print(f"loading model {args.model}…", file=sys.stderr)
    inputs = {i.name: i for i in sess.get_inputs()}
    print(f"inputs: {list(inputs)}", file=sys.stderr)

    with open(args.input_ids) as f:
        prompt_ids = json.load(f)
    if not prompt_ids:
        print("empty token list", file=sys.stderr)
        sys.exit(1)
    print(f"prompt: {len(prompt_ids)} tokens", file=sys.stderr)

    use_kv = args.kv_shape is not None and any(
        n.startswith("past_key_values.") for n in inputs
    )
    if use_kv:
        n_layers, n_heads, head_dim = (int(x) for x in args.kv_shape.split(","))
        past = [
            (
                np.zeros((1, n_heads, 0, head_dim), dtype=np.float32),
                np.zeros((1, n_heads, 0, head_dim), dtype=np.float32),
            )
            for _ in range(n_layers)
        ]
        print(
            f"kv-cache: shape per K/V = [1, {n_heads}, past_len, {head_dim}], "
            f"n_layers={n_layers}",
            file=sys.stderr,
        )
    else:
        past = []

    def feed_dict(tokens, position_offset):
        cur_len = len(tokens)
        total_len = position_offset + cur_len
        feed = {"input_ids": np.array([tokens], dtype=np.int64)}
        if args.with_attention_mask and "attention_mask" in inputs:
            feed["attention_mask"] = np.ones((1, total_len), dtype=np.int64)
        if args.with_position_ids and "position_ids" in inputs:
            feed["position_ids"] = np.arange(
                position_offset, position_offset + cur_len, dtype=np.int64
            ).reshape(1, cur_len)
        for layer, (k, v) in enumerate(past):
            kn = f"past_key_values.{layer}.key"
            vn = f"past_key_values.{layer}.value"
            if kn in inputs:
                feed[kn] = k
            if vn in inputs:
                feed[vn] = v
        return feed

    def run_once(tokens, position_offset):
        feed = feed_dict(tokens, position_offset)
        outs = sess.run(None, feed)
        names = [o.name for o in sess.get_outputs()]
        out_map = dict(zip(names, outs))
        # Cycle present → past in-place; ORT returns numpy arrays we own.
        for layer in range(len(past)):
            kn = f"present.{layer}.key"
            vn = f"present.{layer}.value"
            if kn in out_map and vn in out_map:
                past[layer] = (out_map[kn], out_map[vn])
        return out_map["logits"]

    print(f"warm-up: {args.warmup} run(s)…", file=sys.stderr)
    saved_past = [(k.copy(), v.copy()) for k, v in past]
    for _ in range(args.warmup):
        run_once(prompt_ids, 0)
        # Restore so timed runs see a fresh empty cache.
        for layer in range(len(past)):
            past[layer] = (saved_past[layer][0].copy(), saved_past[layer][1].copy())

    total_t0 = time.perf_counter()
    pre_t0 = time.perf_counter()
    logits = run_once(prompt_ids, 0)
    pre_ms = (time.perf_counter() - pre_t0) * 1000
    position_offset = len(prompt_ids) if use_kv else 0

    # Decode: greedy argmax, feed only the new token if cached.
    decode_t0 = time.perf_counter()
    last_token = int(prompt_ids[-1])
    generated = []
    for _ in range(args.max_tokens):
        if use_kv:
            logits = run_once([last_token], position_offset)
            position_offset += 1
        else:
            # Stateless: re-feed the whole sequence.
            seq = list(prompt_ids) + generated
            logits = run_once(seq, 0)
        # logits is shape [1, seq, vocab] or [1, 1, vocab]; argmax over last
        # token's vocab axis.
        flat = np.array(logits).reshape(-1)
        # logits[..., -1, :] equivalent: take the last vocab block.
        vocab = flat.shape[0] // (
            1 if use_kv else (len(prompt_ids) + len(generated))
        )
        last_block = flat[-vocab:]
        nxt = int(np.argmax(last_block))
        generated.append(nxt)
        last_token = nxt
    decode_ms = (time.perf_counter() - decode_t0) * 1000
    total_ms = (time.perf_counter() - total_t0) * 1000

    decode_tps = len(generated) * 1000.0 / decode_ms if decode_ms > 0 else 0.0
    summary = {
        "model": args.model,
        "prompt_tokens": len(prompt_ids),
        "decode_tokens": len(generated),
        "prefill_ms": pre_ms,
        "decode_total_ms": decode_ms,
        "decode_tokens_per_sec": decode_tps,
        "total_wall_ms": total_ms,
    }
    print(
        f"prompt={len(prompt_ids)}t  prefill={pre_ms:.1f}ms  "
        f"decode={len(generated)}t / {decode_ms:.1f}ms = {decode_tps:.2f} tok/s  "
        f"total={total_ms:.1f}ms",
        file=sys.stderr,
    )
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
