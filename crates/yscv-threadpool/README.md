# yscv-threadpool

Custom work-stealing thread pool with CPU affinity pinning, priority
backoff, and a `ParallelScope` abstraction shared with `rayon::ThreadPool`.
Used by `yscv-kernels` and `yscv-onnx` for the inference dispatch path.

## What's in here

- `YscvPool` — Chase-Lev work-stealing pool built on `crossbeam-deque`,
  with cold-idle backoff and optional pinning to physical or
  big-cluster cores.
- `ParallelScope` trait — a backend-agnostic version of rayon's
  `scope` / `par_chunks_mut` so callers can switch between
  `rayon::ThreadPool` and `YscvPool` without recompiling. Pick the
  backend at runtime via `YSCV_POOL=rayon|yscv`.
- `PersistentSection` — opt-in session-scoped parallel region
  (`YSCV_SESSION_POOL=1`). Workers spin-poll a single
  `current_loop` atomic for the duration of a session so each
  `parallel_for` becomes a pointer store + chunk-counter CAS,
  no fork/join setup. Aimed at very-fine-grained dispatch
  (~3 µs/op vs ~12 µs/op for rayon scope) and currently
  experimental — kept off by default, see kill-switch and the
  on-the-record limits in `lib.rs`.

## Why a custom pool exists

Rayon is the default in this workspace and is fast. The custom pool
exists to let us experiment with three things rayon doesn't expose:

1. CPU affinity pinning (Linux `sched_setaffinity`, macOS
   `thread_policy_set`) — keeps workers off SMT siblings of the
   submitter on Zen / Apple-M-style asymmetric topologies.
2. Per-worker `current_section` TLS for `PersistentSection` so a
   downstream `par_chunks_mut_dispatch` can stay inside the section
   without round-tripping through rayon scope.
3. A controlled spin-and-park epoch so worker wake-up latency is
   visible and tunable rather than hidden behind rayon's internal
   sleepy state.

In production rayon stays the default; the custom pool ships as
opt-in (`YSCV_POOL=yscv`) so its perf can be measured against
rayon on the actual workload, not in isolation.

## Configuration

```
YSCV_POOL={rayon|yscv}              backend selection (default: rayon)
YSCV_POOL_AFFINITY={none|big|physical}  worker pinning policy (yscv only)
YSCV_SESSION_POOL=1                 enable PersistentSection (experimental)
YSCV_ALLOW_SMT=1                    let `OnnxRunner::with_threads(N)` go above physical cores
```

## Status

Library; no `cargo run` entry point. Used internally by `yscv-onnx::OnnxRunner`
when `YSCV_POOL=yscv` is set; otherwise inactive. See `crates/yscv-onnx/README.md`
for the consumer-facing knob.
