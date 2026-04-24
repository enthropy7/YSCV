//! Custom work-stealing thread pool with priority-aware cold-idle spin.
//!
//! Designed to replace `rayon::ThreadPool` in yscv inference hot paths.
//! On sub-4ms inferences, rayon's park-on-idle policy dominates MT overhead —
//! the park/unpark round-trip between back-to-back submissions costs hundreds
//! of nanoseconds per subtask. ORT's Eigen ThreadPool sidesteps this by
//! holding workers in a cold-idle spin (~300µs) before parking.
//!
//! This crate lands the replacement in phases:
//!   A.1 — skeleton pool with shared `Mutex<VecDeque<Task>>`, `install` API.
//!   A.2 — per-worker Chase-Lev deques (via `crossbeam-deque`) with
//!     random-victim work-stealing, still parks on empty.
//!   **A.3 (current)** — cold-idle spin before park. Workers spin for
//!     `YSCV_POOL_SPIN_US` microseconds (default 300) watching for new
//!     work before blocking on the condvar. Back-to-back inferences
//!     arriving within the spin window skip the park/unpark round-trip.
//!   A.4 — priority-aware scheduling + big/physical-core affinity.
//!   A.5 — `ParallelScope` trait + drop-in into `OnnxRunner`.
//!   A.6 — empirical tuning and default switch decision.
//!
//! ## API
//!
//! ```no_run
//! use yscv_threadpool::YscvPool;
//! let pool = YscvPool::new(6).expect("build pool");
//! let result = pool.install(|| {
//!     // Runs on one of the pool's worker threads.
//!     42
//! });
//! assert_eq!(result, 42);
//! ```
//!
//! ## Unsafe policy
//!
//! This crate opts **into** `unsafe` (overriding the workspace-wide
//! `unsafe_code = deny` lint) for the narrow case of extending closure
//! lifetimes in `install`, matching `rayon::ThreadPool::install`'s
//! design. Every `unsafe` block carries a `SAFETY:` comment documenting
//! the invariants that make it sound.

use std::cell::Cell;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
use std::thread;
use std::time::{Duration, Instant};

// Thread-local set on every pool worker at spawn; read by
// `YscvPool::current_worker_index()`. None for non-pool threads (main,
// tests, rayon workers). Used by yscv-kernels to detect re-entrant
// dispatch and avoid nested parallelism that over-subscribes the pool.
thread_local! {
    static CURRENT_WORKER: Cell<Option<usize>> = const { Cell::new(None) };
}

// Step 3 Session C: per-worker dispatch context installed by
// `worker_loop` before entering the steal/sleep cycle. Used by
// `section::try_run_regular_task` when a section is active but
// `current_loop` is empty — instead of spinning idle, the worker
// attempts to pick up a regular task (e.g. a `join_dyn` submit from
// tower-parallel) so tower-parallel graphs don't deadlock.
//
// The pointers are valid for the duration of `worker_loop` — cleared
// before the function returns.
thread_local! {
    pub(crate) static WORKER_DISPATCH_CTX: Cell<Option<WorkerDispatchCtx>> =
        const { Cell::new(None) };
}

#[derive(Clone, Copy)]
pub(crate) struct WorkerDispatchCtx {
    pub(crate) local: *const DequeWorker<Task>,
    pub(crate) shared: *const Shared,
    pub(crate) worker_id: usize,
    pub(crate) nworkers: usize,
}

pub mod affinity;
mod scope;
mod section;
pub use scope::ParallelScope;
pub use section::{PersistentSection, SectionGuard, current_section};

/// `YSCV_POOL_SPIN_US` — microseconds workers spin on empty before parking.
/// Default 0 — **measured optimum for ONNX inference workloads** on
/// tracker-style graphs (fine-grained, 100–200 dispatches/inf, 10–50µs
/// apart). Spin burns CPU cycles that steal from the main thread's
/// useful work; zero-spin + park defers to the OS scheduler which wakes
/// fast enough via `thread::unpark` (futex). Raise via env for coarse
/// install-per-task patterns where spin pays off.
///
/// Measured on Zen 4 Siamese tracker (2026-04-19): spin=0 gave 4.29ms/6T,
/// spin=50 gave 6.0ms, spin=300 gave 9.3ms, spin=1000 gave 19ms.
fn spin_us() -> u64 {
    static CACHED: OnceLock<u64> = OnceLock::new();
    *CACHED.get_or_init(|| {
        std::env::var("YSCV_POOL_SPIN_US")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(0)
    })
}

use crossbeam_deque::{Injector, Steal, Stealer, Worker as DequeWorker};

mod job;
pub(crate) use job::JobRef;

/// Worker task transport. `JobRef` is a 16-byte Copy type: `(fn_ptr,
/// data_ptr)`. The caller stores the closure/state on its own stack
/// frame and blocks on a latch/channel until every worker finishes —
/// zero heap alloc per task. Replaces the pre-A″.4 `Box<dyn FnOnce>`
/// which cost ~1200 `Box::new` per 6T inference on the tracker.
type Task = JobRef;

/// Shared state: one global injector queue for incoming submissions, one
/// stealer per worker so other workers can steal its leftovers, a
/// shutdown flag, and the lock-free sleep-state machinery.
///
/// ## Sleep state machine (A″.2)
///
/// Workers transition through THREE states to avoid futex syscalls in the
/// hot inference path (where dispatches are 10-50 µs apart):
///
/// - `Active` (0): running a task, actively stealing, or in the optional
///   spin window. `submit` never needs to signal an Active worker.
/// - `Sleepy` (1): "ready to park" — the worker has exhausted one steal
///   loop and is yield-watching `jobs_event`. If the counter changes
///   while we're Sleepy, we observe it on the next yield_now iteration
///   and drop back to Active, fetching the new work. **Critical: no
///   park/unpark syscall fires in this state.**
/// - `Sleeping` (2): past the Sleepy watch window; blocked on
///   `thread::park_timeout`. Only reached when the worker is truly idle
///   (no new dispatches in ~32 yield iterations ≈ 100 µs+).
///
/// ## JobsEventCounter
///
/// Packed `AtomicU64` — the rayon-equivalent of their `JobsEventCounter`:
/// - bits 0..32   — `counter`: incremented by `submit` when `num_sleepy
///   > 0 && was_empty`. Sleepy workers captured the old value; counter
///   > inequality tells them new work exists.
/// - bits 32..48  — `num_sleepy`: count of workers currently in Sleepy.
/// - bits 48..64  — `num_sleeping`: count of workers actually parked.
///
/// The three fields fit in one atomic word so `submit` reads all of them
/// with a single `load(Acquire)`. Increments to any field use
/// `fetch_add`/`fetch_sub` with the right stride (`1`, `1 << 32`,
/// `1 << 48`). Overflow of `counter` is fine — workers compare equality,
/// and wrap-around only misses a wake when `num_sleepy` workers captured
/// the exact same pre-wrap value (vanishingly improbable across 2^32
/// submits).
const WORKER_ACTIVE: u8 = 0;
const WORKER_SLEEPY: u8 = 1;
const WORKER_SLEEPING: u8 = 2;

/// Number of yield-watch iterations a Sleepy worker performs before
/// transitioning to Sleeping. Chosen so that under tracker-style
/// fine-grained dispatch (10-50 µs apart), sequential dispatches land
/// inside the window and the worker never reaches Sleeping. Tunable via
/// `YSCV_POOL_SLEEPY_ROUNDS` for coarse workloads.
fn sleepy_rounds() -> u32 {
    static CACHED: OnceLock<u32> = OnceLock::new();
    *CACHED.get_or_init(|| {
        std::env::var("YSCV_POOL_SLEEPY_ROUNDS")
            .ok()
            .and_then(|s| s.parse::<u32>().ok())
            .unwrap_or(32)
    })
}

const JOBS_EVENT_COUNTER_MASK: u64 = 0x0000_0000_FFFF_FFFF;
const JOBS_EVENT_SLEEPY_SHIFT: u32 = 32;
const JOBS_EVENT_SLEEPY_INC: u64 = 1u64 << JOBS_EVENT_SLEEPY_SHIFT;
const JOBS_EVENT_SLEEPY_MASK: u64 = 0x0000_FFFF_0000_0000;
const JOBS_EVENT_SLEEPING_SHIFT: u32 = 48;
const JOBS_EVENT_SLEEPING_INC: u64 = 1u64 << JOBS_EVENT_SLEEPING_SHIFT;
const JOBS_EVENT_SLEEPING_MASK: u64 = 0xFFFF_0000_0000_0000;

#[inline]
fn unpack_counter(event: u64) -> u32 {
    (event & JOBS_EVENT_COUNTER_MASK) as u32
}
#[inline]
fn unpack_sleepy(event: u64) -> u32 {
    ((event & JOBS_EVENT_SLEEPY_MASK) >> JOBS_EVENT_SLEEPY_SHIFT) as u32
}
#[inline]
fn unpack_sleeping(event: u64) -> u32 {
    ((event & JOBS_EVENT_SLEEPING_MASK) >> JOBS_EVENT_SLEEPING_SHIFT) as u32
}

struct WorkerSleepState {
    state: AtomicU8,
    /// Counter value captured when the worker entered Sleepy. When
    /// `jobs_event.counter != captured_counter`, new work has been posted
    /// and the worker drops back to Active — no park needed.
    captured_counter: AtomicU32,
    /// Set once the worker thread starts; `submit()` uses this to
    /// `unpark()` a sleeping worker without holding any lock.
    thread: OnceLock<thread::Thread>,
}

impl WorkerSleepState {
    fn new() -> Self {
        Self {
            state: AtomicU8::new(WORKER_ACTIVE),
            captured_counter: AtomicU32::new(0),
            thread: OnceLock::new(),
        }
    }
}

struct Shared {
    injector: Injector<Task>,
    stealers: Vec<Stealer<Task>>,
    shutdown: AtomicBool,
    /// One entry per worker. Lock-free three-state machine.
    sleep_states: Vec<WorkerSleepState>,
    /// Packed (counter, num_sleepy, num_sleeping). See module-level
    /// comment for bit layout. Drives the epoch-based wake that keeps
    /// fine-grained dispatches syscall-free.
    jobs_event: AtomicU64,
}

impl std::fmt::Debug for Shared {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Shared")
            .field("workers", &self.stealers.len())
            .field("shutdown", &self.shutdown.load(Ordering::Relaxed))
            .finish_non_exhaustive()
    }
}

/// Thread pool with a fixed number of worker threads. Uses per-worker
/// Chase-Lev deques (via `crossbeam-deque`) + random-victim work-stealing.
/// Use [`YscvPool::new`] to construct, [`YscvPool::install`] to run work.
#[derive(Debug)]
pub struct YscvPool {
    shared: Arc<Shared>,
    workers: Vec<thread::JoinHandle<()>>,
    nthreads: usize,
}

impl YscvPool {
    /// Builds a pool with `nthreads` worker threads. Returns `Err` on
    /// `nthreads == 0` (rayon also rejects that).
    pub fn new(nthreads: usize) -> Result<Self, PoolError> {
        if nthreads == 0 {
            return Err(PoolError::ZeroThreads);
        }

        // Create one deque per worker; collect the stealer halves into
        // `shared.stealers` so everyone can steal from everyone.
        let mut worker_deques: Vec<DequeWorker<Task>> = Vec::with_capacity(nthreads);
        let mut stealers: Vec<Stealer<Task>> = Vec::with_capacity(nthreads);
        for _ in 0..nthreads {
            let w = DequeWorker::new_fifo();
            stealers.push(w.stealer());
            worker_deques.push(w);
        }

        let sleep_states: Vec<WorkerSleepState> =
            (0..nthreads).map(|_| WorkerSleepState::new()).collect();
        let shared = Arc::new(Shared {
            injector: Injector::new(),
            stealers,
            shutdown: AtomicBool::new(false),
            sleep_states,
            jobs_event: AtomicU64::new(0),
        });

        // Resolve CPU affinity once at pool construction. Policy from env,
        // defaulting to `physical` (avoid SMT siblings on symmetric SMT).
        let pinning = affinity::resolve_pinning(affinity::policy_from_env(), nthreads);

        // Spawn workers. Each owns a `DequeWorker<Task>` (not Send on its
        // own, but we move it into the spawned thread immediately).
        let mut workers = Vec::with_capacity(nthreads);
        for (worker_id, local) in worker_deques.into_iter().enumerate() {
            let s = Arc::clone(&shared);
            let cpu = pinning.get(worker_id).copied();
            workers.push(thread::spawn(move || {
                if let Some(cpu_id) = cpu {
                    affinity::pin_current_thread(cpu_id);
                }
                // Publish our Thread handle so the waker path can
                // `unpark()` us without holding any lock.
                let _ = s.sleep_states[worker_id].thread.set(thread::current());
                // Record our index in TLS so `current_worker_index()`
                // can report it to nested-dispatch check sites.
                CURRENT_WORKER.with(|c| c.set(Some(worker_id)));
                worker_loop(worker_id, local, s)
            }));
        }

        Ok(Self {
            shared,
            workers,
            nthreads,
        })
    }

    /// Number of worker threads in this pool.
    pub fn num_threads(&self) -> usize {
        self.nthreads
    }

    /// The caller's worker index within this pool (0..num_threads), or
    /// `None` if called from a thread that isn't one of our workers
    /// (e.g. the main thread before `install`, a test thread, a rayon
    /// worker). Mirrors `rayon::current_thread_index()`.
    ///
    /// Note: TLS is process-global, so this returns `Some(i)` on a
    /// yscv-pool worker even when the function is invoked via a
    /// different `YscvPool` instance — acceptable because kernels use
    /// this purely to avoid re-entrant dispatch, and any yscv worker
    /// is by definition busy.
    pub fn current_worker_index(&self) -> Option<usize> {
        CURRENT_WORKER.with(|c| c.get())
    }

    /// Runs `f` on one of the pool's worker threads and blocks the caller
    /// until it returns. Matches `rayon::ThreadPool::install` semantics —
    /// the closure may borrow from the caller's stack.
    ///
    /// # Safety
    ///
    /// The call blocks until the task completes, so any references captured
    /// in `f` remain valid for the full task lifetime. To fit the closure
    /// in the shared `Task` queue (`FnOnce + 'static`), we widen its
    /// lifetime via `transmute`. Same pattern as `rayon::ThreadPool::install`.
    pub fn install<F, R>(&self, f: F) -> R
    where
        F: FnOnce() -> R + Send,
        R: Send,
    {
        // A″.4: zero heap alloc path. The closure + result channel live
        // on this stack frame via `InstallCell`; we submit a `JobRef`
        // pointing at it. Caller blocks on `rx.recv()` until the worker
        // signals, keeping the stack frame alive for the worker's
        // dereference.
        let (tx, rx) = std::sync::mpsc::sync_channel::<R>(1);
        // Put the closure + sender together so the trampoline can
        // reconstitute them. Wrapped in `ManuallyDrop` because the
        // trampoline consumes them via `ptr::read` (matches FnOnce
        // semantics — the closure is moved out before execution).
        use std::mem::ManuallyDrop;
        struct InstallCell<F, R: Send>
        where
            F: FnOnce() -> R,
        {
            f: ManuallyDrop<F>,
            tx: ManuallyDrop<std::sync::mpsc::SyncSender<R>>,
        }
        #[allow(unsafe_code)]
        unsafe fn install_trampoline<F, R>(data: *const ())
        where
            F: FnOnce() -> R + Send,
            R: Send,
        {
            // SAFETY: `data` was `&cell as *const InstallCell<F, R> as
            // *const ()`. The caller (`install`) is blocked on `rx`
            // until we finish, so `cell` is alive. We `ptr::read` F
            // and the sender — one-shot consume, matching FnOnce.
            #[allow(unsafe_code)]
            let cell = unsafe { &*(data as *const InstallCell<F, R>) };
            #[allow(unsafe_code)]
            let f = unsafe { std::ptr::read(&*cell.f) };
            #[allow(unsafe_code)]
            let tx = unsafe { std::ptr::read(&*cell.tx) };
            let value = f();
            let _ = tx.send(value);
        }
        let cell = InstallCell {
            f: ManuallyDrop::new(f),
            tx: ManuallyDrop::new(tx),
        };
        // SAFETY: `install_trampoline::<F, R>` is a `fn` item, `'static`
        // by definition. `&cell` outlives the blocking `rx.recv()` below,
        // so the worker's dereference is valid. Same lifetime invariant
        // as the previous Box-based path.
        #[allow(unsafe_code)]
        let job = unsafe {
            JobRef::new(
                &cell as *const InstallCell<F, R> as *const (),
                install_trampoline::<F, R>,
            )
        };
        self.submit(job);
        rx.recv().expect("worker dropped the result channel")
    }

    /// Internal: push a task onto the global injector and wake workers.
    ///
    /// Three-path fast route:
    /// 1. Load packed `jobs_event` once. If both `num_sleepy == 0` and
    ///    `num_sleeping == 0` → push and return (everyone is Active,
    ///    they'll find the task via steal/injector on their next loop).
    /// 2. If `num_sleepy > 0` → bump the counter. Sleepy workers are
    ///    yield-watching it and observe the change without any syscall.
    /// 3. If `num_sleeping > 0` → scan `sleep_states` for a Sleeping
    ///    worker, CAS it to Active, and `unpark()`.
    fn submit(&self, task: Task) {
        self.shared.injector.push(task);
        self.notify_workers(1);
    }

    /// Batch notification path used by `par_for_each_index` after
    /// pushing N helper tasks. A single `fetch_add(N)` on the counter +
    /// optionally up to `n` unparks wakes at most `n` Sleeping workers
    /// instead of doing one `submit` per helper with N cascaded atomic
    /// RMWs. Measured to save ~50-100 µs per 6T inference on the
    /// tracker (200 parallel regions × ~6 helpers each = 1200 fewer
    /// atomic RMWs).
    fn notify_workers(&self, job_count: usize) {
        let event = self.shared.jobs_event.load(Ordering::Acquire);
        let num_sleepy = unpack_sleepy(event);
        let num_sleeping = unpack_sleeping(event);
        if num_sleepy == 0 && num_sleeping == 0 {
            return;
        }
        if num_sleepy > 0 {
            // Single counter increment per notify — Sleepy workers compare
            // equality, not delta, so one bump is sufficient even for
            // multi-task batches. Release pairs with workers' Acquire.
            self.shared.jobs_event.fetch_add(1, Ordering::Release);
        }
        if num_sleeping > 0 {
            // Wake up to `job_count` Sleeping workers via unpark. Without
            // this bound, a single submit would wake all sleepers when we
            // only have one task for them. With it, each futex_wake
            // corresponds to exactly one available task.
            let wakes_wanted = job_count.min(num_sleeping as usize);
            let mut woken = 0usize;
            for st in &self.shared.sleep_states {
                if woken >= wakes_wanted {
                    break;
                }
                if st.state.load(Ordering::Relaxed) != WORKER_SLEEPING {
                    continue;
                }
                if st
                    .state
                    .compare_exchange(
                        WORKER_SLEEPING,
                        WORKER_ACTIVE,
                        Ordering::AcqRel,
                        Ordering::Relaxed,
                    )
                    .is_ok()
                {
                    self.shared
                        .jobs_event
                        .fetch_sub(JOBS_EVENT_SLEEPING_INC, Ordering::Release);
                    if let Some(t) = st.thread.get() {
                        t.unpark();
                    }
                    woken += 1;
                }
            }
        }
    }

    /// Push a batch of tasks onto the injector atomically from the
    /// caller's perspective (still one-by-one into the lock-free queue,
    /// but we defer notify to a single atomic update + bounded wake).
    /// Used by `par_for_each_index` where all helpers share the same
    /// closure and can be enqueued in one burst.
    fn submit_batch<I: IntoIterator<Item = Task>>(&self, tasks: I) {
        let mut count = 0usize;
        for t in tasks {
            self.shared.injector.push(t);
            count += 1;
        }
        if count > 0 {
            self.notify_workers(count);
        }
    }

    /// Parallel for-each over `0..count`. Runs `f(index)` on pool workers
    /// in parallel; blocks until all iterations complete. Equivalent to
    /// rayon's `(0..count).into_par_iter().for_each(f)`.
    ///
    /// Dispatch strategy: **zero `Arc`, no mutex, no condvar**. All shared
    /// state (the next-index counter and the completion latch) lives on
    /// this function's stack frame. Helpers receive raw pointers into that
    /// frame; the function blocks on the latch with a spin→yield loop
    /// until every helper has finished, guaranteeing the frame outlives
    /// all accesses. The closure is passed via a `&dyn Fn` trait-object
    /// pointer — no allocation, no `Arc<ClosureHolder>` layer.
    ///
    /// This is the hot path of the pool on inference workloads (100-200
    /// fine-grained parallel regions per inference). The previous design
    /// allocated 3 `Arc`s + 1 `Arc<ClosureHolder>` + 4 `Arc::clone`s per
    /// helper + waited on a `Condvar`. Each of those disappears here; the
    /// only remaining heap alloc per call is `helpers` `Box`es (required
    /// by the `Injector<Box<dyn FnOnce>>` task type — addressing that
    /// would need a custom lock-free queue).
    ///
    /// # Safety
    ///
    /// This function contains `unsafe` blocks — see the `SAFETY:` comments
    /// at each site for the invariants. In summary: the function blocks
    /// on `latch` until every helper decrements it, guaranteeing that
    /// `next`, `latch`, and the closure `f` all outlive any helper
    /// access through the raw pointers we hand them.
    pub fn par_for_each_index<F>(&self, count: usize, f: F)
    where
        F: Fn(usize) + Send + Sync,
    {
        if count == 0 {
            return;
        }

        use std::sync::atomic::AtomicUsize;

        // Stack-allocated shared state. Lives for the duration of this
        // function's frame. Helpers access via raw pointers.
        let next = AtomicUsize::new(0);

        // Caller participates — it counts as one puller. Cap helpers at
        // `count - 1` so we don't oversubscribe workers on tiny counts.
        let helpers = self.nthreads.min(count.saturating_sub(1));

        // `latch` counts down as each helper exits its loop. Caller waits
        // until it hits 0. When `helpers == 0` we never touch it.
        let latch = AtomicUsize::new(helpers);

        // Monomorphized trampoline: `fn(*const (), usize)` — a thin fn
        // pointer, Send/Sync by value and free of lifetime gymnastics.
        // The context pointer is `&f as *const F as *const ()`. Each
        // helper calls `(trampoline)(ctx, idx)` which casts back to
        // `&F` and invokes. This avoids the `&dyn Fn + '_` fat pointer
        // and its `F: 'static` bound noise.
        fn trampoline<F: Fn(usize) + Send + Sync>(ctx: *const (), idx: usize) {
            // SAFETY: `ctx` was constructed as `&f as *const F as *const ()`
            // in the parent frame. That frame outlives every call via
            // the latch/spin invariant documented on the outer function.
            #[allow(unsafe_code)]
            let f = unsafe { &*(ctx as *const F) };
            f(idx);
        }
        let f_ctx: *const () = (&f as *const F) as *const ();
        let f_trampoline: fn(*const (), usize) = trampoline::<F>;

        // All helpers need to see: &AtomicUsize next, &AtomicUsize latch,
        // and the closure fat pointer. We bundle them in a `Copy`
        // struct that we mark `Send + Sync` manually — the individual
        // raw pointers are Send/Sync-safe because of the function-scope
        // lifetime invariant documented above.
        #[derive(Clone, Copy)]
        struct Ptrs {
            next: *const AtomicUsize,
            latch: *const AtomicUsize,
            f_ctx: *const (),
            f_trampoline: fn(*const (), usize),
            /// Total index count — embedded here rather than captured by
            /// the helper trampoline, because trampolines are `fn` items
            /// (no closure environment).
            count: usize,
        }
        // SAFETY: The pointer fields target data that outlives every
        // helper task — `next`, `latch`, and the closure `f` all live on
        // this function's stack frame, which we refuse to exit until
        // every helper has bumped `latch` down to 0 at the spin-wait
        // loop at the end of this function. The `AtomicUsize` targets
        // are race-safe; the closure is `Fn + Sync` so concurrent
        // invocation is sound. `f_trampoline` is a `fn` item, already
        // Send+Sync.
        #[allow(unsafe_code)]
        unsafe impl Send for Ptrs {}
        #[allow(unsafe_code)]
        unsafe impl Sync for Ptrs {}

        // Accessor methods so the spawned closure captures `Ptrs` as a
        // whole (getting the `unsafe impl Send for Ptrs` stamp) rather
        // than letting Rust-2021 disjoint field capture strip to the
        // bare raw pointers, which would fail Send.
        impl Ptrs {
            #[inline]
            fn next(&self) -> *const AtomicUsize {
                self.next
            }
            #[inline]
            fn latch(&self) -> *const AtomicUsize {
                self.latch
            }
            #[inline]
            fn call(&self, idx: usize) {
                (self.f_trampoline)(self.f_ctx, idx);
            }
        }

        let ptrs = Ptrs {
            next: &next as *const AtomicUsize,
            latch: &latch as *const AtomicUsize,
            f_ctx,
            f_trampoline,
            count,
        };

        // A″.4: all helpers share the same `Ptrs` on the caller's stack
        // frame; each worker's `JobRef` is just a pair of pointers
        // (trampoline fn + &ptrs). Zero heap allocations per helper vs
        // the prior `Box<dyn FnOnce>` path which cost one `Box::new`
        // each. The trampoline runs the pull-from-counter / call / latch
        // dec loop inline.
        #[allow(unsafe_code)]
        unsafe fn helper_trampoline(data: *const ()) {
            // SAFETY: `data` was `&ptrs as *const Ptrs as *const ()`.
            // The caller (`par_for_each_index`) blocks on `latch` until
            // every helper decrements it, so `ptrs` and everything it
            // points to remain valid.
            #[allow(unsafe_code)]
            let p = unsafe { *(data as *const Ptrs) };
            // count is encoded in the caller's stack frame via the
            // trampoline's closure? — no, trampolines can't capture.
            // We read count via Ptrs (need to add it).
            loop {
                // SAFETY: p.next outlives helper via blocking latch.
                #[allow(unsafe_code)]
                let idx = unsafe { (*p.next()).fetch_add(1, Ordering::Relaxed) };
                if idx >= p.count {
                    break;
                }
                p.call(idx);
            }
            // SAFETY: p.latch outlives helper via blocking latch.
            #[allow(unsafe_code)]
            unsafe {
                (*p.latch()).fetch_sub(1, Ordering::Release);
            }
        }

        // Build `helpers` copies of the same `JobRef` pointing to the
        // shared `ptrs` cell. Using a Vec<Task> (= Vec<JobRef>, 16 B
        // each) here — allocation is once per call, amortized across
        // all helpers; far cheaper than N Box<dyn FnOnce>.
        // SAFETY: `helper_trampoline` is a `fn` item (`'static`), and
        // `&ptrs` outlives the blocking spin-wait below. Same invariant
        // as the previous Box-based path.
        #[allow(unsafe_code)]
        let job = unsafe { JobRef::new(&ptrs as *const Ptrs as *const (), helper_trampoline) };
        let tasks: Vec<Task> = (0..helpers).map(|_| job).collect();
        self.submit_batch(tasks);

        // Caller participates — drains indices inline until the counter
        // is exhausted. Handles `helpers == 0` (count == 1) entirely
        // here: no helpers were spawned, the caller is the only puller.
        loop {
            let idx = next.fetch_add(1, Ordering::Relaxed);
            if idx >= count {
                break;
            }
            f(idx);
        }

        if helpers == 0 {
            return;
        }

        // Spin-yield wait for all helpers to exit their loops. They'll
        // hit the drained counter almost immediately since the caller
        // also pulled — this is pure join overhead. No condvar: the
        // window here is microseconds, parking on a cvar would cost
        // more than it saves.
        //
        // Acquire: pair with the helpers' Release on `latch.fetch_sub`
        // so we observe their memory writes before returning.
        let mut spin = 0u32;
        while latch.load(Ordering::Acquire) != 0 {
            if spin < 64 {
                std::hint::spin_loop();
                spin += 1;
            } else {
                std::thread::yield_now();
            }
        }
    }

    /// Fork-join: run `a` and `b` in parallel, return both results.
    /// Equivalent to `rayon::join(a, b)`. Implemented by submitting `a`
    /// to the pool and running `b` inline on the caller's thread.
    ///
    /// # Safety
    ///
    /// Same lifetime-extension rule as `install` — we block on the result
    /// channel for `a`, and `b` runs synchronously, so both closures'
    /// captures remain live.
    pub fn join<A, B, RA, RB>(&self, a: A, b: B) -> (RA, RB)
    where
        A: FnOnce() -> RA + Send,
        B: FnOnce() -> RB + Send,
        RA: Send,
        RB: Send,
    {
        // A″.4: stack-allocate the `a` closure + its result sender into
        // a `JoinCell` and submit a zero-alloc `JobRef`.
        let (tx_a, rx_a) = std::sync::mpsc::sync_channel::<RA>(1);
        use std::mem::ManuallyDrop;
        struct JoinCell<A, RA: Send>
        where
            A: FnOnce() -> RA,
        {
            a: ManuallyDrop<A>,
            tx: ManuallyDrop<std::sync::mpsc::SyncSender<RA>>,
        }
        #[allow(unsafe_code)]
        unsafe fn join_trampoline<A, RA>(data: *const ())
        where
            A: FnOnce() -> RA + Send,
            RA: Send,
        {
            // SAFETY: `data` points to a stack-allocated `JoinCell<A, RA>`
            // held alive by the caller blocking on `rx_a.recv()` below.
            #[allow(unsafe_code)]
            let cell = unsafe { &*(data as *const JoinCell<A, RA>) };
            #[allow(unsafe_code)]
            let a = unsafe { std::ptr::read(&*cell.a) };
            #[allow(unsafe_code)]
            let tx = unsafe { std::ptr::read(&*cell.tx) };
            let _ = tx.send(a());
        }
        let cell = JoinCell {
            a: ManuallyDrop::new(a),
            tx: ManuallyDrop::new(tx_a),
        };
        // SAFETY: `join_trampoline::<A, RA>` is a `fn` item (`'static`).
        // `&cell` outlives the `rx_a.recv()` below. See install for the
        // same pattern.
        #[allow(unsafe_code)]
        let job = unsafe {
            JobRef::new(
                &cell as *const JoinCell<A, RA> as *const (),
                join_trampoline::<A, RA>,
            )
        };
        self.submit(job);
        // Run b synchronously on this thread. Uses one less worker =
        // better CPU utilization during small fork-join patterns.
        let rb = b();
        let ra = rx_a.recv().expect("worker dropped join channel");
        (ra, rb)
    }

    /// Step 3: enter a session-scoped [`PersistentSection`]. All pool
    /// workers are held in [`section_worker_loop`] for the duration of
    /// `f`; regular `submit` / `install` paths are effectively blocked
    /// during the section (all workers busy). Each
    /// [`PersistentSection::parallel_for`] call inside the closure
    /// dispatches work via a single atomic pointer store + chunk-counter
    /// CAS — no rayon scope creation, no `Box` alloc, no condvar/futex.
    ///
    /// Target: the tracker's ~1.6 ms MT-overhead budget. See
    /// `reflective-fluttering-backus.md` Step 3 for projection.
    ///
    /// # Safety
    ///
    /// Section and trampoline context are stack-allocated on this
    /// function's frame. Main thread spins on `workers_remaining` before
    /// returning, guaranteeing every worker's `section_worker_loop` has
    /// exited and released the pointer before the stack frame is freed.
    pub fn enter_section<R, F>(&self, f: F) -> R
    where
        F: FnOnce(&PersistentSection) -> R,
    {
        let section = PersistentSection::new();

        // Stack-allocated trampoline context — just the section pointer.
        // Workers read `section_ptr`, cast back to `&PersistentSection`,
        // run [`section_worker_loop`].
        #[repr(C)]
        struct SectionCtx {
            section_ptr: *const PersistentSection,
        }
        // SAFETY: `SectionCtx` holds a pointer to a stack-rooted
        // `PersistentSection` which outlives every worker dereference
        // (main thread blocks on `workers_remaining == 0` before
        // returning). No shared mutable state beyond the section's own
        // atomics, which are Send/Sync by construction.
        #[allow(unsafe_code)]
        unsafe impl Send for SectionCtx {}
        #[allow(unsafe_code)]
        unsafe impl Sync for SectionCtx {}

        #[allow(unsafe_code)]
        unsafe fn section_worker_trampoline(data: *const ()) {
            // SAFETY: `data` was constructed as `&ctx as *const _` in
            // `enter_section`. `ctx` lives on the `enter_section` stack
            // frame, kept alive by the `workers_remaining` spin barrier.
            #[allow(unsafe_code)]
            let ctx = unsafe { &*(data as *const SectionCtx) };
            #[allow(unsafe_code)]
            let section = unsafe { &*ctx.section_ptr };
            // SAFETY: see `section` reference above.
            #[allow(unsafe_code)]
            unsafe {
                section::section_worker_loop(section);
            }
        }

        let ctx = SectionCtx {
            section_ptr: &section as *const PersistentSection,
        };

        // SAFETY: `section_worker_trampoline` is a `fn` item (`'static`).
        // `&ctx` outlives every worker dereference by the barrier below.
        #[allow(unsafe_code)]
        let job = unsafe {
            JobRef::new(
                &ctx as *const SectionCtx as *const (),
                section_worker_trampoline,
            )
        };

        // Submit one copy per worker. JobRef is Copy (16 B), so the
        // Vec is a cheap dense buffer of identical handles.
        let tasks: Vec<Task> = (0..self.nthreads).map(|_| job).collect();
        self.submit_batch(tasks);

        // Entry barrier: wait for every worker to pick up its job and
        // increment `workers_remaining`. Typical time: <10 µs on Zen 4
        // (workers already Active or in Sleepy). If any worker is
        // Sleeping, submit_batch's bounded unpark already kicked it.
        let expected = self.nthreads;
        let mut spin = 0u32;
        while section.workers_remaining_count() < expected {
            if spin < 256 {
                std::hint::spin_loop();
                spin = spin.saturating_add(1);
            } else {
                std::thread::yield_now();
            }
        }

        let result = f(&section);

        // Signal exit. Workers observe `active == false` on their next
        // loop iteration and `fetch_sub` workers_remaining.
        section.mark_inactive();

        // Exit barrier: wait for every worker to exit the section
        // loop. Workers currently in their back-off yield may take a
        // few µs to observe the flag.
        let mut spin = 0u32;
        while section.workers_remaining_count() > 0 {
            if spin < 256 {
                std::hint::spin_loop();
                spin = spin.saturating_add(1);
            } else {
                std::thread::yield_now();
            }
        }

        result
    }
}

impl Drop for YscvPool {
    fn drop(&mut self) {
        // Signal shutdown and wake every worker so they exit cleanly.
        // Each worker has its own Thread handle; unpark them all to
        // kick any Sleeping ones out of their futex wait.
        self.shared.shutdown.store(true, Ordering::Release);
        for st in &self.shared.sleep_states {
            if let Some(t) = st.thread.get() {
                t.unpark();
            }
        }
        for handle in self.workers.drain(..) {
            let _ = handle.join();
        }
    }
}

/// Worker main loop: run own deque tasks first (LIFO for good cache
/// locality), fall back to injector global queue, then random-victim
/// steal. Cold-idle spin on empty for `YSCV_POOL_SPIN_US` µs before
/// falling through to a condvar park.
fn worker_loop(worker_id: usize, local: DequeWorker<Task>, shared: Arc<Shared>) {
    let mut steal_rng_state: u32 = 0x9e37_79b1u32.wrapping_mul(worker_id as u32 + 1);
    let nworkers = shared.stealers.len();
    let spin_duration = Duration::from_micros(spin_us());
    let spin_enabled = spin_duration > Duration::ZERO;

    // Step 3 Session C: stash dispatch context in TLS so
    // `section::section_worker_loop` can call `find_task` to pick up
    // regular tasks (e.g. `join_dyn` submits from tower-parallel)
    // while the worker is ALSO processing section chunks. Without this,
    // tower-parallel graphs deadlock when all workers are spinning on
    // `section.current_loop` and `join_dyn` submits a task to the
    // injector with no available worker to pick it up.
    WORKER_DISPATCH_CTX.with(|cell| {
        cell.set(Some(WorkerDispatchCtx {
            local: &local as *const _,
            shared: Arc::as_ptr(&shared),
            worker_id,
            nworkers,
        }));
    });
    // Guard that clears the TLS on worker exit — RAII so it fires on
    // both normal return and unwinding paths (worker panic would leave
    // a dangling pointer otherwise).
    struct ClearDispatchCtxGuard;
    impl Drop for ClearDispatchCtxGuard {
        fn drop(&mut self) {
            WORKER_DISPATCH_CTX.with(|cell| cell.set(None));
        }
    }
    let _dispatch_ctx_guard = ClearDispatchCtxGuard;

    'outer: loop {
        // Drain local deque first.
        while let Some(task) = local.pop() {
            #[allow(unsafe_code)]
            unsafe {
                task.execute();
            }
        }

        // Try injector + work-stealing.
        if let Some(task) = find_task(&local, &shared, &mut steal_rng_state, worker_id, nworkers) {
            #[allow(unsafe_code)]
            unsafe {
                task.execute();
            }
            continue;
        }

        if shared.shutdown.load(Ordering::Acquire) {
            return;
        }

        // Cold-idle spin (deprecated default: spin_us=0). Still honoured
        // via env override for callers that want busy-spin instead of
        // the Sleepy yield-watch below.
        if spin_enabled {
            let deadline = Instant::now() + spin_duration;
            while Instant::now() < deadline {
                if shared.shutdown.load(Ordering::Acquire) {
                    return;
                }
                if let Some(task) =
                    find_task(&local, &shared, &mut steal_rng_state, worker_id, nworkers)
                {
                    #[allow(unsafe_code)]
                    unsafe {
                        task.execute();
                    }
                    break;
                }
                std::hint::spin_loop();
            }
            if let Some(task) =
                find_task(&local, &shared, &mut steal_rng_state, worker_id, nworkers)
            {
                #[allow(unsafe_code)]
                unsafe {
                    task.execute();
                }
                continue;
            }
        }

        // ── Sleepy state (A″.2) ──────────────────────────────────
        //
        // Instead of jumping straight to `park`, we enter Sleepy: capture
        // the current `jobs_event` counter, announce ourselves via the
        // packed `num_sleepy` field, and yield-watch the counter for
        // `sleepy_rounds()` iterations. On each iteration we also retry
        // `find_task` — a submit pushed the job before its counter bump
        // may be visible already.
        //
        // Submit's fast path increments the counter ONLY IF `num_sleepy >
        // 0`, so our Sleepy announcement is load-bearing: it tells submit
        // "I'm watching, no syscall needed, just bump the counter". The
        // counter change propagates via cache coherence in ~50-100 ns —
        // far cheaper than park/unpark (~1 µs each).
        let st = &shared.sleep_states[worker_id];
        let initial_event = shared.jobs_event.load(Ordering::Acquire);
        let captured = unpack_counter(initial_event);
        st.captured_counter.store(captured, Ordering::Release);
        st.state.store(WORKER_SLEEPY, Ordering::Release);
        shared
            .jobs_event
            .fetch_add(JOBS_EVENT_SLEEPY_INC, Ordering::Release);

        let rounds = sleepy_rounds();
        let mut woken_by_counter = false;
        // `yield_now` (sched_yield on Linux, ~300 ns) spreads the load
        // across other runnable threads and — measured — outperforms
        // tight `spin_loop` here. The bottleneck on tracker workloads
        // isn't the yield cost, it's the cache-coherence ping-pong on
        // `jobs_event` if 6 workers hammer the atomic in a tight loop.
        // yield_now gives the cache line a chance to quiesce between
        // reads.
        for _ in 0..rounds {
            thread::yield_now();

            let cur = unpack_counter(shared.jobs_event.load(Ordering::Acquire));
            if cur != captured {
                woken_by_counter = true;
                break;
            }
            if shared.shutdown.load(Ordering::Acquire) {
                shared
                    .jobs_event
                    .fetch_sub(JOBS_EVENT_SLEEPY_INC, Ordering::Release);
                st.state.store(WORKER_ACTIVE, Ordering::Release);
                return;
            }
            // Opportunistic find_task — some submits may race the
            // counter bump (pushed but counter not yet incremented).
            if let Some(task) =
                find_task(&local, &shared, &mut steal_rng_state, worker_id, nworkers)
            {
                shared
                    .jobs_event
                    .fetch_sub(JOBS_EVENT_SLEEPY_INC, Ordering::Release);
                st.state.store(WORKER_ACTIVE, Ordering::Release);
                #[allow(unsafe_code)]
                unsafe {
                    task.execute();
                }
                continue 'outer;
            }
        }

        // Leave Sleepy. Either the counter moved (woken_by_counter) or we
        // exhausted the window (truly idle → go Sleeping).
        shared
            .jobs_event
            .fetch_sub(JOBS_EVENT_SLEEPY_INC, Ordering::Release);
        if woken_by_counter {
            st.state.store(WORKER_ACTIVE, Ordering::Release);
            continue;
        }

        // ── Sleeping state ───────────────────────────────────────
        //
        // Truly idle past the Sleepy window. CAS Sleepy → Sleeping; if
        // submit raced in and already flipped us to Active, skip the park.
        if st
            .state
            .compare_exchange(
                WORKER_SLEEPY,
                WORKER_SLEEPING,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_err()
        {
            // submit flipped us to Active — re-loop.
            continue;
        }
        shared
            .jobs_event
            .fetch_add(JOBS_EVENT_SLEEPING_INC, Ordering::Release);

        // Recheck for work + shutdown before parking (closes the race
        // where submit pushed work between our Sleepy exit and Sleeping
        // entry).
        let quick_work = find_task(&local, &shared, &mut steal_rng_state, worker_id, nworkers);
        if quick_work.is_some() || shared.shutdown.load(Ordering::Acquire) {
            if st
                .state
                .compare_exchange(
                    WORKER_SLEEPING,
                    WORKER_ACTIVE,
                    Ordering::AcqRel,
                    Ordering::Relaxed,
                )
                .is_ok()
            {
                shared
                    .jobs_event
                    .fetch_sub(JOBS_EVENT_SLEEPING_INC, Ordering::Release);
            }
            if let Some(task) = quick_work {
                #[allow(unsafe_code)]
                unsafe {
                    task.execute();
                }
            }
            if shared.shutdown.load(Ordering::Acquire) {
                return;
            }
            continue;
        }

        // Blocking park. submit wakes via unpark; 1 ms timeout is a
        // safety net (rarely fires in practice once Sleepy catches the
        // fast path).
        thread::park_timeout(Duration::from_millis(1));
        if st
            .state
            .compare_exchange(
                WORKER_SLEEPING,
                WORKER_ACTIVE,
                Ordering::AcqRel,
                Ordering::Relaxed,
            )
            .is_ok()
        {
            shared
                .jobs_event
                .fetch_sub(JOBS_EVENT_SLEEPING_INC, Ordering::Release);
        }
        if shared.shutdown.load(Ordering::Acquire) {
            return;
        }
    }
}

/// Find a task to run: injector first, then steal from a random peer.
/// Xorshift32 RNG for the victim pick — avoids a hot shared counter
/// while still giving reasonable spread.
pub(crate) fn find_task(
    local: &DequeWorker<Task>,
    shared: &Shared,
    rng: &mut u32,
    worker_id: usize,
    nworkers: usize,
) -> Option<Task> {
    // Try the injector — pumps a batch into our local deque if full.
    loop {
        match shared.injector.steal_batch_and_pop(local) {
            Steal::Success(t) => return Some(t),
            Steal::Empty => break,
            Steal::Retry => continue,
        }
    }

    // Random-victim steal.
    for _ in 0..(nworkers * 2) {
        // xorshift32.
        *rng ^= *rng << 13;
        *rng ^= *rng >> 17;
        *rng ^= *rng << 5;
        let victim = (*rng as usize) % nworkers;
        if victim == worker_id {
            continue;
        }
        match shared.stealers[victim].steal_batch_and_pop(local) {
            Steal::Success(t) => return Some(t),
            Steal::Empty => {}
            Steal::Retry => {}
        }
    }
    None
}

/// Errors returned by [`YscvPool::new`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PoolError {
    /// `nthreads == 0`.
    ZeroThreads,
}

impl std::fmt::Display for PoolError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PoolError::ZeroThreads => write!(f, "thread count must be > 0"),
        }
    }
}

impl std::error::Error for PoolError {}

#[cfg(test)]
mod tests;
