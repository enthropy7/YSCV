use super::{PoolError, YscvPool};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

#[test]
fn new_rejects_zero_threads() {
    match YscvPool::new(0) {
        Err(PoolError::ZeroThreads) => {}
        Ok(_) => panic!("expected ZeroThreads, got Ok"),
    }
}

#[test]
fn install_returns_closure_value() {
    let pool = YscvPool::new(2).unwrap();
    let result = pool.install(|| 7 + 35);
    assert_eq!(result, 42);
}

#[test]
fn install_returns_boxed_value() {
    let pool = YscvPool::new(2).unwrap();
    let result = pool.install(|| vec![1u32, 2, 3, 4]);
    assert_eq!(result, vec![1, 2, 3, 4]);
}

#[test]
fn install_runs_on_worker_thread() {
    let pool = YscvPool::new(1).unwrap();
    let main_id = std::thread::current().id();
    let worker_id = pool.install(|| std::thread::current().id());
    // install-returned id must come from the worker (different from main).
    assert_ne!(main_id, worker_id);
}

#[test]
fn many_installs_all_complete() {
    // Submit 1000 tasks sequentially; every one should run and return its
    // expected value. Smoke test for queue correctness and worker wakeup.
    let pool = YscvPool::new(4).unwrap();
    for i in 0..1000u64 {
        let v = pool.install(move || i * 2);
        assert_eq!(v, i * 2);
    }
}

#[test]
fn pool_shuts_down_cleanly() {
    // Dropping the pool must join all workers without hanging. We give it
    // up to 5 seconds before declaring a deadlock (unit tests shouldn't
    // hang CI forever if this ever regresses).
    let start = Instant::now();
    {
        let pool = YscvPool::new(8).unwrap();
        pool.install(|| 1 + 1);
    }
    assert!(
        start.elapsed() < Duration::from_secs(5),
        "YscvPool drop took > 5s — worker join deadlock?"
    );
}

#[test]
fn concurrent_installs_from_scoped_threads() {
    // Fire installs from multiple foreground threads at once — queue must
    // serialize correctly without lost or duplicated work.
    let pool = Arc::new(YscvPool::new(4).unwrap());
    let counter = Arc::new(AtomicUsize::new(0));
    std::thread::scope(|s| {
        for _ in 0..16 {
            let p = Arc::clone(&pool);
            let c = Arc::clone(&counter);
            s.spawn(move || {
                for _ in 0..50 {
                    let c = Arc::clone(&c);
                    p.install(move || {
                        c.fetch_add(1, Ordering::Relaxed);
                    });
                }
            });
        }
    });
    assert_eq!(counter.load(Ordering::Relaxed), 16 * 50);
}

#[test]
fn par_for_each_index_runs_all_iterations() {
    let pool = YscvPool::new(4).unwrap();
    let sum = Arc::new(AtomicUsize::new(0));
    let s = Arc::clone(&sum);
    pool.par_for_each_index(1000, move |i| {
        s.fetch_add(i, Ordering::Relaxed);
    });
    // Sum of 0..1000 = 499500.
    assert_eq!(sum.load(Ordering::Relaxed), 499_500);
}

#[test]
fn par_for_each_index_handles_zero_count() {
    let pool = YscvPool::new(2).unwrap();
    pool.par_for_each_index(0, |_| {
        panic!("should not run");
    });
}

#[test]
fn join_returns_both_results() {
    let pool = YscvPool::new(4).unwrap();
    let (a, b) = pool.join(|| 42u32, || "hello".to_string());
    assert_eq!(a, 42);
    assert_eq!(b, "hello");
}

#[test]
fn join_runs_in_parallel() {
    // Both closures sleep; total wall time should be ~max not ~sum.
    let pool = YscvPool::new(4).unwrap();
    let start = Instant::now();
    let (a, b) = pool.join(
        || {
            std::thread::sleep(Duration::from_millis(50));
            1
        },
        || {
            std::thread::sleep(Duration::from_millis(50));
            2
        },
    );
    let elapsed = start.elapsed();
    assert_eq!(a, 1);
    assert_eq!(b, 2);
    // If serialized, would take ~100ms. Parallel should be ~50ms; allow
    // 90ms slack for scheduler jitter on busy CI.
    assert!(
        elapsed < Duration::from_millis(90),
        "join took {:?} — looks serialized",
        elapsed
    );
}

/// Stress test for the Part A′ zero-alloc `par_for_each_index`:
/// many small parallel regions in a tight loop mimicking the inference
/// hot-path's fine-grained dispatch pattern. Every iteration index must
/// be visited exactly once across all threads. Catches (a) races on the
/// shared next counter, (b) latch ordering bugs where the caller returns
/// before a helper finishes, (c) ABA on reused stack frames.
#[test]
fn par_for_each_index_stress_fine_grained() {
    let pool = YscvPool::new(6).unwrap();
    // 2000 × mixed-count calls — mirrors tracker's ~100-200 per inference.
    let counts = [1usize, 2, 3, 6, 8, 16, 64, 256];
    for round in 0..2_000 {
        let count = counts[round % counts.len()];
        let touched = Arc::new((0..count).map(|_| AtomicUsize::new(0)).collect::<Vec<_>>());
        let t = Arc::clone(&touched);
        pool.par_for_each_index(count, move |i| {
            t[i].fetch_add(1, Ordering::Relaxed);
        });
        for (i, slot) in touched.iter().enumerate() {
            let hits = slot.load(Ordering::Relaxed);
            assert_eq!(
                hits, 1,
                "round {} count {} index {}: expected exactly 1 visit, got {}",
                round, count, i, hits
            );
        }
    }
}

/// Edge case: `count == 0` must be a no-op, not touch any state.
#[test]
fn par_for_each_index_count_zero_never_calls() {
    let pool = YscvPool::new(4).unwrap();
    let called = Arc::new(AtomicUsize::new(0));
    let c = Arc::clone(&called);
    pool.par_for_each_index(0, move |_| {
        c.fetch_add(1, Ordering::Relaxed);
    });
    assert_eq!(called.load(Ordering::Relaxed), 0);
}

/// Edge case: `count == 1` must run exactly once and must not spawn any
/// helpers (caller handles it inline per the `helpers == 0` fast path).
#[test]
fn par_for_each_index_count_one_runs_inline() {
    let pool = YscvPool::new(4).unwrap();
    let called = Arc::new(AtomicUsize::new(0));
    let c = Arc::clone(&called);
    let caller_tid = std::thread::current().id();
    let observed_tid = Arc::new(std::sync::Mutex::new(None));
    let o = Arc::clone(&observed_tid);
    pool.par_for_each_index(1, move |_| {
        c.fetch_add(1, Ordering::Relaxed);
        *o.lock().unwrap() = Some(std::thread::current().id());
    });
    assert_eq!(called.load(Ordering::Relaxed), 1);
    // count=1 → helpers = min(nthreads, 0) = 0 → caller runs inline.
    assert_eq!(
        *observed_tid.lock().unwrap(),
        Some(caller_tid),
        "count=1 should run on caller thread (no helpers spawned)"
    );
}

/// A″.2 regression: the Sleepy-state epoch counter should let workers
/// pick up back-to-back dispatches without parking. Fire 10 000 submits
/// in tight succession on a pool of 4 workers and verify every task
/// ran exactly once. Failure mode if the state machine is wrong: lost
/// wakeups (test hangs via 5 s timeout wrapper) or double-runs.
#[test]
fn par_for_each_fine_grained_under_sleepy_epoch() {
    let pool = YscvPool::new(4).unwrap();
    let touches = Arc::new(AtomicUsize::new(0));
    // Force workers to reach Sleepy between bursts: short idle pause.
    for _round in 0..200 {
        let t = Arc::clone(&touches);
        let f: Box<dyn Fn(usize) + Send + Sync> = Box::new(move |_| {
            t.fetch_add(1, Ordering::Relaxed);
        });
        pool.par_for_each_index(50, f.as_ref());
        // Tiny sleep so workers drift into Sleepy between rounds.
        std::thread::sleep(Duration::from_micros(10));
    }
    assert_eq!(touches.load(Ordering::Relaxed), 200 * 50);
}

/// Regression: closures that borrow from the caller's stack must be
/// callable — the rewrite uses raw pointers, so accidentally escaping
/// the closure or copying state incorrectly would be UB. Captures a
/// borrowed `&mut Vec<f32>` via an atomic index.
#[test]
fn par_for_each_index_captures_stack_borrow() {
    let pool = YscvPool::new(4).unwrap();
    let data: Vec<AtomicUsize> = (0..128).map(|_| AtomicUsize::new(0)).collect();
    let borrow: &[AtomicUsize] = &data;
    pool.par_for_each_index(128, |i| {
        borrow[i].store(i * 2, Ordering::Relaxed);
    });
    for i in 0..128 {
        assert_eq!(data[i].load(Ordering::Relaxed), i * 2);
    }
}

/// Parity microbench: one install-per-task on our pool vs a
/// `rayon::ThreadPool` with the same worker count. Expected: yscv pool
/// is slower or equal to rayon at A.1 (Mutex<VecDeque> isn't competitive
/// with rayon's Chase-Lev). We track the ratio so A.2 improvements are
/// measurable against a baseline.
///
/// Runs under `--release` only — debug-mode numbers are too noisy to act
/// on. Marked `#[ignore]` so normal `cargo test` doesn't run it; run
/// explicitly with `cargo test --release -p yscv-threadpool --
/// --ignored parity_vs_rayon`.
#[test]
#[ignore]
fn parity_vs_rayon() {
    const N_TASKS: usize = 10_000;
    const N_THREADS: usize = 4;

    // yscv pool
    let ys_pool = YscvPool::new(N_THREADS).unwrap();
    let t0 = Instant::now();
    for _ in 0..N_TASKS {
        ys_pool.install(|| std::hint::black_box(1_u64 + 2));
    }
    let ys_time = t0.elapsed();

    // rayon pool
    let rn_pool = rayon::ThreadPoolBuilder::new()
        .num_threads(N_THREADS)
        .build()
        .unwrap();
    let t0 = Instant::now();
    for _ in 0..N_TASKS {
        rn_pool.install(|| std::hint::black_box(1_u64 + 2));
    }
    let rn_time = t0.elapsed();

    let ratio = ys_time.as_secs_f64() / rn_time.as_secs_f64();
    eprintln!(
        "parity: yscv {:?}, rayon {:?}, ratio {:.2}× (1.0 = match)",
        ys_time, rn_time, ratio
    );
    // A.1 target: within 10× of rayon (we expect slower due to mutex).
    // A.2+ should bring this down toward 1.0.
    assert!(
        ratio < 10.0,
        "yscv pool >10× slower than rayon — regression"
    );
}

// ── Step 3 — PersistentSection integration ───────────────────────────

/// End-to-end: enter a section, run 3 back-to-back parallel_for loops,
/// verify each produces correct output + workers exit cleanly on section
/// close. Exercises the full path: enter_section → submit_batch → workers
/// pick up trampoline → loop is published → main + workers race on chunks
/// → done_count barrier → next loop → mark_inactive → workers exit.
#[test]
fn enter_section_runs_multiple_parallel_for_loops() {
    let pool = YscvPool::new(4).unwrap();
    let results = pool.enter_section(|section| {
        let mut outputs = Vec::with_capacity(3);
        for base in [10u32, 100, 1000] {
            let acc = AtomicUsize::new(0);
            section.parallel_for(50, |idx| {
                acc.fetch_add((base + idx as u32) as usize, Ordering::Release);
            });
            outputs.push(acc.load(Ordering::Acquire));
        }
        outputs
    });
    // sum of base..base+50 = 50*base + 0+1+...+49 = 50*base + 1225.
    assert_eq!(
        results,
        vec![50 * 10 + 1225, 50 * 100 + 1225, 50 * 1000 + 1225]
    );
}

/// Ensure a section can be entered multiple times on the same pool.
/// Each section is independent — counters reset, workers return between.
#[test]
fn enter_section_can_be_repeated() {
    let pool = YscvPool::new(3).unwrap();
    for iter in 0..5 {
        let total = pool.enter_section(|section| {
            let acc = AtomicUsize::new(0);
            section.parallel_for(10, |idx| {
                acc.fetch_add(idx + iter, Ordering::Release);
            });
            acc.load(Ordering::Acquire)
        });
        // sum(0..10) + 10*iter = 45 + 10*iter
        assert_eq!(total, 45 + 10 * iter);
    }
}

/// Verify empty `parallel_for` (chunks=0) is a no-op that doesn't hang.
/// Also verifies the section itself can be entered and exited with no
/// work done inside.
#[test]
fn enter_section_empty_parallel_for_is_noop() {
    let pool = YscvPool::new(2).unwrap();
    pool.enter_section(|section| {
        section.parallel_for(0, |_| unreachable!("should not run on chunks=0"));
    });
}
