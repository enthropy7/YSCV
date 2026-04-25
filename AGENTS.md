# AGENTS.md — rules for AI coding agents working on yscv

This file is the contract between this repository and any AI coding
agent (Claude Code, Copilot Workspace, Cursor, Aider, etc.) that
proposes changes here. Concrete, enforceable, short.

The same rules apply to human contributors.

---

## Responsibility

Every PR has exactly one responsible party: the human who opens it.
This holds whether the diff was hand-typed, drafted by an agent, or
copy-pasted from a chat log. By submitting, you assert four things
about the change:

1. You have the right to submit it under this repository's licence.
2. You have read the code you are submitting and understand what it
   does. If asked "why this line?", you can answer without
   re-prompting an agent.
3. You have run the local gates listed in [section 2](#2-the-workflow)
   and the result is what you say it is. "The agent said tests pass"
   is not a substitute for running them.
4. For any performance claim, you measured it yourself on your own
   hardware and the numbers in the PR description are yours.

There is no "the agent did it" defence. A regression introduced by
an agent-drafted patch is the submitter's regression, and reverting
or fixing it is the submitter's job.

Disclosure of method: if a change is non-trivial and was drafted
with an agent, mention that in the PR description in one line —
what the agent did, what you verified. This is for context, not
credit; do not add `Co-Authored-By:` lines for the agent (see
[section 3](#3-the-do-not-list)).

Anonymous or pseudonymous PRs that exist only to relay agent output
without a human standing behind them will be closed unread.

---

## 0. Read these first, in order

1. [`CONTRIBUTING.md`](CONTRIBUTING.md) — project priorities and the
   change workflow.
2. [`docs/feature-flags.md`](docs/feature-flags.md) — what each
   Cargo feature does, when to enable it, runtime env knobs.
3. [`docs/architecture.md`](docs/architecture.md) — the high-level
   shape of the workspace, dispatch boundaries, dataflow.
4. The crate-local `README.md` of the crate you are touching.
5. The latest `docs/perf-arc-*.md` if you're touching a hot path —
   it lists what landed, what was tried and reverted, and why.

That's about thirty minutes of reading and prevents the most common
class of "agent re-proposes a previously-tried dead-end" PRs.

---

## 1. The five commandments

Non-negotiable. Violating any one is grounds for the PR to be
reverted on sight, regardless of benchmark numbers.

1. Blazing fast or out. Every change on a hot path must show a
   measured win on the shape range it targets. "It compiles" and
   "tests pass" are not enough for an inference loop. Use criterion
   benches or a representative end-to-end harness; never push a
   perf-claimed change without numbers in the commit message.

2. Multi-arch SIMD or scalar. No SIMD function lands as x86-only
   or aarch64-only. Either ship NEON + AVX/SSE + scalar fallback
   together, or stay scalar. Runtime feature detection via
   `std::arch::is_*_feature_detected!`. The dispatch pattern is
   consistent across the codebase — copy from
   `crates/yscv-kernels/src/ops/matmul.rs` if in doubt.

3. Minimal code. Smallest correct change. Iterators where the
   compiler vectorises them, no `unwrap`, no `#[allow(dead_code)]`,
   no half-finished implementations. New abstractions only when
   they earn their keep — three similar lines beat a premature
   trait family. If you find yourself adding a helper to make the
   diff "look cleaner", delete the helper.

4. Numerical correctness is non-negotiable. Default paths produce
   bitwise-identical or 1-ULP-close outputs against the scalar /
   reference path on supported shapes. Approximations (fp16
   storage, mixed-precision compute, quantization, fast-math
   reductions) ship behind explicit env knobs or Cargo features,
   never as default. A commit that "speeds things up" by silently
   relaxing precision is a regression — measure the output drift
   in the same suite that measures latency.

5. Document the change in the same commit. If you change a public
   API or runtime behaviour, the docs change with the code: the
   crate `README.md`, `docs/feature-flags.md` for new flags,
   `docs/ecosystem-capability-matrix.md` for new capabilities,
   `context.md` for surface counts. The
   `bash scripts/check-doc-counts.sh` gate enforces this for the
   counted surfaces (crate count, ONNX op count, tensor methods,
   imgproc fns, etc.).

---

## 2. The workflow

1. Confirm scope. Restate to yourself in one sentence what is
   changing and why. If you can't, you don't have scope yet — go
   back to the issue or prompt and clarify.

2. Read the prior art. `git log --oneline -- <path>` for the files
   you're about to touch, and `git log --grep="<topic>"` for any
   prior arc on the same problem. Reverted commits and their
   commit messages are the canonical record of what didn't work.

3. Implement minimally. See commandment #3.

4. Test. Unit + integration where shape varies. Numerical ops
   need reference-parity tests against documented formulas.
   Hot-path code needs a criterion bench.

5. Bench. For perf changes, measure on the actual workload,
   three runs minimum, report median + min. The repo's bench
   harness is the source of truth, not one-shot timing.

6. Verify. Locally, run the exact gates CI runs in the same
   order:
   ```sh
   cargo fmt --check
   cargo clippy --workspace --all-targets -- -D warnings
   cargo test --workspace
   bash scripts/check-doc-counts.sh
   ```
   For accelerator features:
   ```sh
   cargo clippy --workspace --features gpu -- -D warnings
   cargo clippy --workspace --features "gpu rknn native-camera" -- -D warnings
   cargo check --target x86_64-pc-windows-msvc
   ```

7. Commit. See [section 4](#4-commit-message-style).

---

## 3. The "do not" list

Things that look reasonable from outside but are project-poison
inside.

- Do not add `#[allow(dead_code)]` or `#[allow(clippy::*)]` to
  silence warnings. Either wire the code into the dispatch
  immediately, delete it, or fix the lint properly.
- Do not suppress Miri findings. Stacked Borrows / leak reports
  are real bugs; fix the root cause. The only allowed pattern is
  `#[cfg_attr(miri, ignore)]` on a test that uses AVX/FMA
  intrinsics, because Miri can't emulate them.
- Do not revert a previously-landed change without reading the
  commit message that introduced it. The code may look "wrong"
  but be the third correct attempt that beat two earlier obvious
  approaches.
- Do not add Co-Authored-By tags for an AI agent on the commit.
  Attribution belongs to the human author of the PR.
- Do not reimplement a mature ecosystem dependency (rayon,
  crossbeam, tokio, ndarray, image) inline for a marginal
  expected win. Battle-tested concurrency / numeric crates have
  absorbed edge cases yours hasn't yet. Measure first; if the
  win is real, the right move is usually a contribution upstream
  rather than a parallel implementation here.
- Do not introduce backwards-compat shims, feature flags, or
  `// removed` markers for deleted code. If something is gone,
  it's gone.
- Do not add a Python binding to the main path or a C++ wrapper
  around a Rust-native option. The whole pitch is "no Python or
  C++ runtime"; that promise is load-bearing.

---

## 4. Commit message style

Two registers, depending on the change size.

Trivial changes (clippy lint fix, fmt, rename, dead-code removal,
doc typo) — one-liner: `clippy`, `fmt`, `nice`, `tests`, `opti`.
Match the existing repo style.

Substantive changes (race fix, SIGSEGV, correctness regression,
silent-drop bug, new kernel / fusion pass, multi-step arc
landing) — expanded form:

```
short-punchy-subject (one line, verb-first, under ~60 chars)

Root cause paragraph: what was happening, which invariant was
broken, which code path triggered it. Name the function / struct
/ test that demonstrates it so `git blame` lands on the right
context six months from now.

If there's a second independent fix bundled in, give it its own
paragraph with its own root cause + mechanism. Don't squash them
into bullets.

Fix paragraph: the one-sentence mechanism (counter, gate, cfg,
etc.) plus any cost note.

Validation: list the suites you ran, ideally with output counts
(e.g. "500-iter stress: 0 crashes, was 1 % before").
```

When unsure, lean toward expanded.

---

## 5. When to ask, when to ship

Ask before committing if any of these apply:

- The change is hard to reverse (force-push to a shared branch,
  schema changes, deletions of public API).
- The change touches an external system (CI config, GitHub
  Actions, deployment infra).
- You're about to revert a decision an upstream maintainer or
  the user explicitly approved earlier in the same session.

Ship without asking when:

- It's a local, reversible change (`cargo fmt`, doc fix, test
  addition).
- The broader plan is already approved and this is one of its
  concrete steps.
- It's a CI-fix following an exact failure log — fix the root
  cause, push, watch CI.

When unsure, lean ask. The cost of one extra round-trip is small;
the cost of an unwanted force-push is high. Approval given for
one specific action stands for the scope it was given, not in
perpetuity.

---

## 6. Performance claims need numbers in the PR

If a change touches a hot path or claims a perf win, the PR
description has to include before / after numbers measured by the
PR author. Don't trust someone else's machine to disagree later;
take responsibility for the comparison.

The expected shape:

- One Rust harness using `yscv-onnx` directly (or a representative
  yscv API), iterating the workload and printing min / p50 / avg
  in microseconds.
- One Python harness using `onnxruntime` against the same model
  and inputs, printing the same statistics.
- Three runs of each, median reported, on the host where the
  change is supposed to win.

A minimal Rust harness looks like this:

```rust
// bench.rs — `cargo run --release --bin bench`
use std::time::Instant;
use yscv_onnx::{load_onnx_model_from_file, OnnxRunner};
use yscv_tensor::Tensor;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = load_onnx_model_from_file("model.onnx")?;
    let runner = OnnxRunner::new(&model)?;
    let input = Tensor::zeros(vec![1, 3, 224, 224])?;
    let feed: Vec<(&str, &Tensor)> = vec![("input", &input)];

    runner.run(&feed)?; // warm-up

    let mut samples_us = Vec::with_capacity(500);
    for _ in 0..500 {
        let t = Instant::now();
        let _ = runner.run(&feed)?;
        samples_us.push(t.elapsed().as_micros() as u64);
    }
    samples_us.sort_unstable();
    let p50 = samples_us[samples_us.len() / 2];
    let avg = samples_us.iter().sum::<u64>() / samples_us.len() as u64;
    println!("min={} p50={} avg={}", samples_us[0], p50, avg);
    Ok(())
}
```

The matching Python harness:

```python
# bench.py — onnxruntime reference
import time, statistics, numpy as np, onnxruntime as ort

so = ort.SessionOptions()
so.intra_op_num_threads = 6
sess = ort.InferenceSession("model.onnx", sess_options=so,
                            providers=["CPUExecutionProvider"])
feed = {"input": np.zeros((1, 3, 224, 224), dtype=np.float32)}

sess.run(None, feed)  # warm-up

samples_us = []
for _ in range(500):
    t = time.perf_counter()
    sess.run(None, feed)
    samples_us.append(int((time.perf_counter() - t) * 1e6))

samples_us.sort()
print(f"min={samples_us[0]} p50={samples_us[len(samples_us)//2]} "
      f"avg={int(statistics.fmean(samples_us))}")
```

Adapt thread counts, providers, and the input shape to the
workload you're optimising. Paste the output of both harnesses
into the PR description, plus a one-line summary in the form
"before X µs → after Y µs (Z % win), N runs median, hardware
description". If no perf delta is claimed, none of this is
required.
