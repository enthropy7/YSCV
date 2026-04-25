//! `PersistentSection` — session-scoped parallel region.
//!
//! Step 3 of the fp32 ORT-gap arc. ORT/Eigen's `ParallelSection` sits all
//! pool workers in a tight spin-loop polling a single `current_loop`
//! atomic pointer for the duration of one `Session::run()`. Each op's
//! `parallel_for` becomes a single pointer store + atomic chunk-counter
//! CAS — no rayon scope creation, no `Box<dyn FnOnce>` allocation, no
//! condvar wake, no park/unpark round-trip. On Zen 4 this is measured
//! at ~3 µs/op dispatch vs ~12 µs for rayon fork-join per op.
//!
//! ## Design (Session A — 3.1 + 3.2)
//!
//! Rather than modify the existing `worker_loop` to poll a section
//! pointer on every cycle (which would slow down non-section workloads),
//! the section is expressed as **N special jobs** — one per worker —
//! each of which runs [`section_worker_loop`] until `active` flips to
//! false. Main thread submits all N via the pool's existing injector +
//! wake path, then dispatches work via [`PersistentSection::parallel_for`]
//! which toggles `current_loop` to point at a caller-stack
//! [`LoopWorkItem`]. Workers claim chunks via `fetch_add` on the shared
//! counter, run the trampoline, increment `done_count`. Main thread
//! participates (runs chunk 0 inline) and blocks until `done_count`
//! equals `chunks`.
//!
//! ## Lifetime & aliasing discipline
//!
//! - Section struct lives for the scope of `YscvPool::enter_section`'s
//!   closure; workers hold no `Arc` on it — the backing storage is on
//!   the enter_section stack frame, kept alive by a join-barrier spin
//!   loop on `workers_remaining`.
//! - `LoopWorkItem` lives on the caller's stack during a single
//!   `parallel_for` call; main thread blocks on `done_count` before
//!   clearing `current_loop`, guaranteeing all workers have finished
//!   their chunk before the `LoopWorkItem` is reclaimed.
//! - `active` uses `Release`/`Acquire` so workers observe the `false`
//!   transition without further synchronisation.

use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicU64, AtomicUsize, Ordering};

/// Descriptor for one `parallel_for` loop posted inside a
/// [`PersistentSection`]. Lives on the main-thread stack during
/// [`PersistentSection::parallel_for`]; workers observe via
/// `current_loop` pointer.
///
/// `exec(ctx, idx)` runs chunk `idx` of the loop. The trampoline
/// pattern (monomorphised `fn` item + opaque ctx pointer) mirrors
/// [`JobRef`](crate::JobRef) so the section stays Send+Sync without
/// requiring a generic parameter on [`PersistentSection`] itself.
pub(crate) struct LoopWorkItem {
    /// Trampoline: decodes `ctx` into the loop's closure and invokes
    /// it with `idx`.
    exec: unsafe fn(ctx: *const (), idx: usize),
    /// Opaque pointer to a caller-stack cell holding the closure.
    ctx: *const (),
    /// Total chunks in this loop; workers stop claiming when
    /// `chunk_counter >= total_chunks`.
    total_chunks: usize,
}

// SAFETY: `LoopWorkItem` is only accessed by the section's main thread
// (store into `current_loop`, clear after join) and by workers (read
// via pointer, call `exec` on a chunk). The main thread blocks on
// `done_count` before clearing, so the pointee is alive for every
// worker dereference. `exec` is a `fn` item (statically `'static`).
// `ctx` points to a caller-stack closure whose lifetime is guarded by
// the same join barrier.
#[allow(unsafe_code)]
unsafe impl Send for LoopWorkItem {}
#[allow(unsafe_code)]
unsafe impl Sync for LoopWorkItem {}

/// Session-scoped parallel region. Construct via
/// [`YscvPool::enter_section`](crate::YscvPool::enter_section). Provides
/// [`parallel_for`](Self::parallel_for) for the duration of the section;
/// the section holds all pool workers in the section worker loop.
pub struct PersistentSection {
    /// Current loop pointer, or null when idle. Workers read this on
    /// every iteration; main thread stores it before dispatch and
    /// clears it after join.
    current_loop: AtomicPtr<LoopWorkItem>,
    /// Chunk index claim counter. Workers and main thread
    /// `fetch_add(1)` to claim the next chunk; once the return
    /// value ≥ `total_chunks`, nothing left to do.
    chunk_counter: AtomicUsize,
    /// Number of chunks that have **finished**. Main thread blocks on
    /// `done_count >= chunks` before returning from `parallel_for`.
    done_count: AtomicUsize,
    /// Number of workers currently in [`section_worker_loop`]. Used as
    /// entry/exit barrier: main thread waits for this to reach
    /// `expected_workers` on section enter, and back to 0 on exit.
    workers_remaining: AtomicUsize,
    /// Workers observe `false` and exit their loop. Main thread stores
    /// `false` on section drop.
    active: AtomicBool,
    /// Step 3 Session C: spin-lock to serialise concurrent
    /// `parallel_for` calls from different threads — critical for
    /// tower-parallel graphs where both branches post loops to the
    /// same section. Without this, branch A's `current_loop` store
    /// races with branch B's store, corrupting the work pointer.
    /// Contention window is microseconds (the duration of one loop),
    /// so spin-lock is cheaper than Mutex's futex-based park.
    dispatch_busy: AtomicBool,
    /// Number of workers currently inside the `loop_ptr` critical
    /// section (between the increment here and the matching decrement
    /// on exit). Main thread spin-waits for this to reach 0 after
    /// storing `current_loop = null` — otherwise a worker that read
    /// `loop_ptr` right before the clear could dereference a
    /// destroyed stack-allocated `Ctx<F>` when `parallel_for` returns.
    inflight_derefs: AtomicUsize,
    /// Monotonic loop generation. Main thread increments twice per
    /// `parallel_for`: once at publish (`loop_gen % 2 == 1` while a
    /// loop is live), once at clear (`loop_gen % 2 == 0` while idle).
    /// Workers snapshot this on entry and re-check under the
    /// `inflight_derefs` guard; a mismatch means stack-reuse ABA
    /// happened (main returned + next loop at the same stack slot)
    /// and the pointer comparison alone would silently read a
    /// reinitialised `LoopWorkItem`. With the counter matching,
    /// `current_loop` still pointing at the same address, and
    /// `inflight_derefs > 0` blocking main's return, the frame is
    /// guaranteed alive. Combined invariant retires the 1 % SIGSEGV
    /// stress flake without falling back to a per-call heap alloc.
    loop_gen: AtomicU64,
}

impl PersistentSection {
    pub(crate) fn new() -> Self {
        Self {
            current_loop: AtomicPtr::new(std::ptr::null_mut()),
            chunk_counter: AtomicUsize::new(0),
            done_count: AtomicUsize::new(0),
            workers_remaining: AtomicUsize::new(0),
            active: AtomicBool::new(true),
            dispatch_busy: AtomicBool::new(false),
            inflight_derefs: AtomicUsize::new(0),
            loop_gen: AtomicU64::new(0),
        }
    }

    /// Number of workers currently inside [`section_worker_loop`].
    /// Main thread uses this for the enter/exit barriers in
    /// [`YscvPool::enter_section`](crate::YscvPool::enter_section).
    #[inline]
    pub(crate) fn workers_remaining_count(&self) -> usize {
        self.workers_remaining.load(Ordering::Acquire)
    }

    /// Signal workers to exit the section loop on their next iteration.
    /// Called from [`YscvPool::enter_section`](crate::YscvPool::enter_section)
    /// when the user closure returns.
    #[inline]
    pub(crate) fn mark_inactive(&self) {
        self.active.store(false, Ordering::Release);
    }

    /// Install this section as the thread-local "current section" for
    /// the returned [`SectionGuard`]'s lifetime. Consumers call
    /// [`current_section`] and route `par_chunks_mut_dispatch` through
    /// [`Self::parallel_for`] when the guard is alive.
    ///
    /// Returns a `!Send` RAII guard; Drop restores the previous TLS
    /// value (nested `install_tls` stacks correctly).
    #[inline]
    pub fn install_tls(&self) -> SectionGuard<'_> {
        let prev = CURRENT_SECTION_TLS.with(|cell| {
            let prev = cell.get();
            cell.set(self as *const PersistentSection);
            prev
        });
        SectionGuard {
            prev,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Post a parallel_for loop. Runs `f(0..chunks)` concurrently across
    /// the section's workers plus the main thread (which runs chunks
    /// inline between checks). Blocks until every chunk has completed.
    ///
    /// Design notes:
    /// - Zero heap allocation — `Ctx<F>` and `LoopWorkItem` both live
    ///   on the caller's stack frame. Workers observe them via raw
    ///   pointers, guarded by two independent invariants:
    ///   1. `inflight_derefs`: workers increment before any deref of
    ///      `loop_ptr`; main spin-waits it back to zero after clearing
    ///      `current_loop`, so main cannot return (and the stack frame
    ///      cannot drop) while any worker is mid-deref.
    ///   2. `loop_gen`: monotonic counter, bumped by main at publish
    ///      and again at clear. Workers snapshot it before the
    ///      `inflight_derefs` bump and re-check under the guard.
    ///      Covers the stack-reuse ABA window where main returns,
    ///      re-enters `parallel_for`, and allocates a new `LoopWorkItem`
    ///      at the same stack address: the pointer match is
    ///      not enough to distinguish "same loop" from "recycled frame
    ///      with different `exec`/`ctx`/`total_chunks` values", but
    ///      `loop_gen` changes across every publish, so the generation
    ///      mismatch makes the worker skip the deref.
    /// - Condvar/futex-free — workers spin-poll `current_loop`; main
    ///   signals via a pointer store.
    /// - Main thread participates: claims chunks via `fetch_add` on
    ///   the same counter, amortising the launch latency.
    pub fn parallel_for<F>(&self, chunks: usize, f: F)
    where
        F: Fn(usize) + Send + Sync,
    {
        if chunks == 0 {
            return;
        }

        // Session C: acquire dispatch-busy spin-lock. Concurrent
        // `parallel_for` calls from different threads (e.g. two
        // tower-parallel branches) would otherwise race on
        // `current_loop`. Spin-lock because the critical section is
        // short (microseconds); futex-park would cost more than it saves.
        while self
            .dispatch_busy
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            // While waiting for the in-flight loop, this thread (often
            // a worker via `try_run_regular_task`) is doing nothing
            // useful. Could opportunistically run section chunks here
            // — but the section-worker-loop path already does that on
            // the worker side, and the main thread is typically the
            // contended one, so plain spin is fine.
            std::hint::spin_loop();
        }
        // Guard to release the spin-lock on panic as well as normal
        // exit — `parallel_for` runs arbitrary closures that may
        // panic.
        struct BusyGuard<'a>(&'a AtomicBool);
        impl Drop for BusyGuard<'_> {
            fn drop(&mut self) {
                self.0.store(false, Ordering::Release);
            }
        }
        let _busy_guard = BusyGuard(&self.dispatch_busy);

        struct Ctx<F: Fn(usize)> {
            f: F,
        }
        #[allow(unsafe_code)]
        unsafe fn trampoline<F: Fn(usize) + Send + Sync>(ctx: *const (), idx: usize) {
            // SAFETY: `ctx` was constructed as `&Ctx<F> as *const _` in
            // `parallel_for`. The caller blocks on `done_count` and
            // `inflight_derefs` before returning, and bumps `loop_gen`
            // to invalidate any stale observer; together these keep
            // `Ctx<F>` alive for every dereference. `F: Fn + Sync`
            // permits concurrent invocation.
            #[allow(unsafe_code)]
            let c = unsafe { &*(ctx as *const Ctx<F>) };
            (c.f)(idx);
        }

        let ctx = Ctx { f };
        let work = LoopWorkItem {
            exec: trampoline::<F>,
            ctx: &ctx as *const Ctx<F> as *const (),
            total_chunks: chunks,
        };

        // Reset per-loop counters BEFORE publishing the loop pointer
        // so workers never see a stale done_count.
        self.chunk_counter.store(0, Ordering::Release);
        self.done_count.store(0, Ordering::Release);
        // Bump loop_gen BEFORE publishing the pointer. Workers snapshot
        // this as part of their entry sequence; any worker that observed
        // a previous generation's pointer will see the new gen on
        // re-check and skip the deref. Release-ordered so subsequent
        // `current_loop.store(work, Release)` happens-after this in
        // other threads' view.
        self.loop_gen.fetch_add(1, Ordering::Release);
        // Publish the loop. Workers spin-polling `current_loop` will
        // observe it on their next Acquire-load iteration.
        self.current_loop.store(
            &work as *const LoopWorkItem as *mut LoopWorkItem,
            Ordering::Release,
        );

        // Main thread participates — drain chunks via the shared counter.
        loop {
            let idx = self.chunk_counter.fetch_add(1, Ordering::AcqRel);
            if idx >= chunks {
                break;
            }
            // SAFETY: trampoline invariant — `ctx` and `work` are alive
            // (stack frame held by this function), `exec` is a `fn` item.
            #[allow(unsafe_code)]
            unsafe {
                (work.exec)(work.ctx, idx);
            }
            self.done_count.fetch_add(1, Ordering::Release);
        }

        // Wait for every claimed chunk to finish. Spin — typical wait
        // is microseconds; condvar would cost more than it saves.
        let mut spin = 0u32;
        while self.done_count.load(Ordering::Acquire) < chunks {
            if spin < 128 {
                std::hint::spin_loop();
                spin = spin.saturating_add(1);
            } else {
                std::thread::yield_now();
            }
        }

        // Clear the loop pointer, then bump loop_gen again so any
        // worker that still holds the old pointer and is about to
        // fetch_add `inflight_derefs` will observe a mismatched
        // generation on its re-check and skip the deref.
        self.current_loop
            .store(std::ptr::null_mut(), Ordering::Release);
        self.loop_gen.fetch_add(1, Ordering::Release);

        // Drain any worker that already passed the gen+pointer
        // re-check and is inside the `inflight_derefs` guard. Without
        // this spin the stack-local `work` / `Ctx<F>` would drop under
        // them when `parallel_for` returns.
        let mut spin = 0u32;
        while self.inflight_derefs.load(Ordering::Acquire) > 0 {
            if spin < 128 {
                std::hint::spin_loop();
                spin = spin.saturating_add(1);
            } else {
                std::thread::yield_now();
            }
        }
    }
}

/// Worker-side loop. Submitted to each pool worker when a section
/// is entered; returns when `section.active` flips to `false`.
///
/// Step 3 Session C: when no section chunk is available, the worker
/// also attempts to pick up a regular task (e.g. a `join_dyn` submit
/// from tower-parallel). Without this, tower-parallel graphs would
/// deadlock — all workers in section mode, `join_dyn` submit has no
/// picker. The `WorkerDispatchCtx` stashed in TLS by `worker_loop`
/// gives us access to the worker's local deque + shared pool state.
///
/// # Safety
/// - `section` must outlive the loop; enforced by the main thread
///   spinning on `workers_remaining` before dropping the section.
/// - `WORKER_DISPATCH_CTX` TLS is set by `worker_loop` for the lifetime
///   of the worker thread; pointers inside outlive this call.
#[allow(unsafe_code)]
pub(crate) unsafe fn section_worker_loop(section: &PersistentSection) {
    // Signal entry.
    section.workers_remaining.fetch_add(1, Ordering::Release);

    let mut steal_rng: u32 = 0x9e37_79b1;
    let mut idle_spins: u32 = 0;
    while section.active.load(Ordering::Acquire) {
        let loop_ptr = section.current_loop.load(Ordering::Acquire);
        if !loop_ptr.is_null() {
            // Snapshot `loop_gen` BEFORE the `inflight_derefs` bump
            // so we lock in "the generation that was live when we
            // observed `loop_ptr`". Any subsequent mismatch on
            // re-check means main has moved on (cleared, re-published,
            // possibly at the same stack address) and the pointer we
            // hold is stale.
            let observed_gen = section.loop_gen.load(Ordering::Acquire);

            // Enter critical section: bump `inflight_derefs` BEFORE
            // dereferencing `loop_ptr`. Main thread's `parallel_for`
            // spins on this counter after clearing `current_loop`, so
            // the ctx frame is kept alive until we decrement on exit.
            section.inflight_derefs.fetch_add(1, Ordering::AcqRel);

            // Re-check both the pointer and the generation. Pointer
            // alone is insufficient because of stack reuse (main
            // returns + next parallel_for at the same frame address);
            // the gen counter catches that case. Gen alone is also
            // insufficient — a null loop_ptr with a matching old gen
            // is possible if main cleared without re-publishing yet.
            let loop_ptr2 = section.current_loop.load(Ordering::Acquire);
            let current_gen = section.loop_gen.load(Ordering::Acquire);
            if !loop_ptr2.is_null() && loop_ptr2 == loop_ptr && current_gen == observed_gen {
                // SAFETY: main thread holds the `LoopWorkItem` and
                // associated `Ctx<F>` alive until `done_count >= chunks`
                // AND `inflight_derefs == 0`. We incremented above and
                // re-checked both pointer + generation — main cannot
                // clear + return while our guard is live AND the gen
                // is stable.
                let work = unsafe { &*loop_ptr };
                let idx = section.chunk_counter.fetch_add(1, Ordering::AcqRel);
                if idx < work.total_chunks {
                    // SAFETY: trampoline invariant (see LoopWorkItem docs).
                    unsafe {
                        (work.exec)(work.ctx, idx);
                    }
                    section.done_count.fetch_add(1, Ordering::Release);
                    section.inflight_derefs.fetch_sub(1, Ordering::AcqRel);
                    idle_spins = 0;
                    continue;
                }
                // All chunks claimed — exit guard and fall through.
            }
            section.inflight_derefs.fetch_sub(1, Ordering::AcqRel);
        }

        // Step 3 Session C: try regular tasks (join_dyn, submit).
        // When called on a pool worker, `WORKER_DISPATCH_CTX` gives us
        // access to `find_task` to pick up the submitted task. Outside
        // a pool worker (e.g. unit tests driving section_worker_loop
        // directly) the ctx is None — fall through to back-off.
        // SAFETY: called from section_worker_loop which runs inside a worker_loop frame.
        if unsafe { try_run_regular_task(&mut steal_rng) } {
            idle_spins = 0;
            continue;
        }

        // No loop + no regular task. Adaptive back-off.
        if idle_spins < 1024 {
            std::hint::spin_loop();
            idle_spins = idle_spins.saturating_add(1);
        } else {
            std::thread::yield_now();
        }
    }

    // Signal exit. Main thread spins on this reaching 0.
    section.workers_remaining.fetch_sub(1, Ordering::Release);
}

/// Step 3 Session C: pick up a regular task from the worker's local
/// deque / injector / steal-from-peer. Returns `true` if a task was
/// executed. Only fires when `WORKER_DISPATCH_CTX` is set (i.e. we're
/// on a pool worker inside `worker_loop`). Caller keeps its own rng
/// state for the random-victim steal.
///
/// # Safety
/// - `WORKER_DISPATCH_CTX.local` / `.shared` pointers are valid for
///   the lifetime of the worker's `worker_loop` frame. This function
///   is only called from `section_worker_loop`, which runs inside
///   that frame via the section_worker_trampoline dispatch.
#[allow(unsafe_code)]
#[inline]
unsafe fn try_run_regular_task(rng: &mut u32) -> bool {
    use crate::WORKER_DISPATCH_CTX;
    let ctx = WORKER_DISPATCH_CTX.with(|c| c.get());
    let Some(ctx) = ctx else { return false };
    // SAFETY: pointers are valid for the worker's `worker_loop` frame;
    // section_worker_loop is only invoked from within that frame.
    let local = unsafe { &*ctx.local };
    let shared = unsafe { &*ctx.shared };

    // Drain local deque first.
    if let Some(task) = local.pop() {
        unsafe {
            task.execute();
        }
        return true;
    }
    // Then injector + peer steal.
    if let Some(task) = crate::find_task(local, shared, rng, ctx.worker_id, ctx.nworkers) {
        unsafe {
            task.execute();
        }
        return true;
    }
    false
}

thread_local! {
    /// Per-thread current section pointer. Null when no section installed.
    /// Set via [`PersistentSection::install_tls`] (returns a [`SectionGuard`]
    /// whose `Drop` restores the previous value). Read via
    /// [`current_section`] from downstream crates (e.g. `yscv-kernels`'s
    /// `par_chunks_mut_dispatch`) to decide whether to route parallel work
    /// through the section or fall back to rayon.
    static CURRENT_SECTION_TLS: std::cell::Cell<*const PersistentSection> =
        const { std::cell::Cell::new(std::ptr::null()) };
}

/// RAII guard returned by [`PersistentSection::install_tls`]. Drops the
/// thread-local pointer back to its previous value (null or an outer
/// section) on scope exit. Not `Send`: TLS is per-thread.
pub struct SectionGuard<'a> {
    prev: *const PersistentSection,
    _phantom: std::marker::PhantomData<&'a PersistentSection>,
}

impl Drop for SectionGuard<'_> {
    fn drop(&mut self) {
        CURRENT_SECTION_TLS.with(|cell| cell.set(self.prev));
    }
}

/// Returns the current thread-local section, if one is installed via
/// [`PersistentSection::install_tls`]. Used by `yscv-kernels` to route
/// `par_chunks_mut_dispatch` through [`PersistentSection::parallel_for`]
/// instead of rayon's fork-join when a session is active.
///
/// # Safety (caller discipline)
/// The `'static` lifetime is a white lie — the returned reference is
/// only valid while the corresponding [`SectionGuard`] is alive on this
/// thread. Callers must not stash the reference across points where the
/// guard could drop. The standard usage pattern (look up, call
/// `parallel_for`, discard) is safe because the guard is rooted in
/// `OnnxRunner::run` and outlives every synchronous dispatch.
#[inline]
pub fn current_section() -> Option<&'static PersistentSection> {
    CURRENT_SECTION_TLS.with(|cell| {
        let p = cell.get();
        if p.is_null() {
            None
        } else {
            // SAFETY: pointer non-null only while a SectionGuard is
            // alive; see the caller-discipline note above.
            #[allow(unsafe_code)]
            Some(unsafe { &*p })
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicU32;

    /// Smoke test: a freshly-constructed section has the expected
    /// initial state (no active loop, zero counters, active=true).
    #[test]
    fn section_initial_state() {
        let s = PersistentSection::new();
        assert!(s.current_loop.load(Ordering::Acquire).is_null());
        assert_eq!(s.chunk_counter.load(Ordering::Acquire), 0);
        assert_eq!(s.done_count.load(Ordering::Acquire), 0);
        assert_eq!(s.workers_remaining.load(Ordering::Acquire), 0);
        assert!(s.active.load(Ordering::Acquire));
    }

    /// Drive the section from a single main thread — no worker
    /// threads. `parallel_for` should still complete because the main
    /// thread participates. Verifies the trampoline dispatch and
    /// counter bookkeeping on the happy path.
    #[test]
    fn parallel_for_single_thread_runs_all_chunks() {
        let s = PersistentSection::new();
        let seen = AtomicU32::new(0);
        s.parallel_for(10, |idx| {
            let bit = 1u32 << idx;
            seen.fetch_or(bit, Ordering::Release);
        });
        assert_eq!(seen.load(Ordering::Acquire), 0x3FF); // bits 0..9 set
        // After parallel_for: loop cleared, counters at 10.
        assert!(s.current_loop.load(Ordering::Acquire).is_null());
        assert_eq!(s.chunk_counter.load(Ordering::Acquire), 11); // 10 claimed + 1 overflow
        assert_eq!(s.done_count.load(Ordering::Acquire), 10);
    }

    /// Verify `section_worker_loop` exits cleanly when `active` flips
    /// to false. Uses a separate thread as the "worker" and the main
    /// thread as "driver".
    #[test]
    fn worker_loop_exits_on_active_false() {
        let s = Box::new(PersistentSection::new());
        let s_ptr: *const PersistentSection = &*s;

        struct SendPtr(*const PersistentSection);
        impl SendPtr {
            #[inline]
            fn ptr(&self) -> *const PersistentSection {
                self.0
            }
        }
        // SAFETY: test-only wrapper so we can pass a stack pointer into a
        // spawned thread. Main keeps the Box alive until handle.join().
        #[allow(unsafe_code)]
        unsafe impl Send for SendPtr {}
        let send_ptr = SendPtr(s_ptr);

        let handle = std::thread::spawn(move || {
            // SAFETY: `s` is Box'd on the main thread's stack and
            // kept alive until after join() below. `send_ptr.ptr()`
            // is a method call so Rust 2021 disjoint-capture doesn't
            // strip the SendPtr wrapper to the raw pointer.
            #[allow(unsafe_code)]
            let section = unsafe { &*send_ptr.ptr() };
            #[allow(unsafe_code)]
            unsafe {
                section_worker_loop(section);
            }
        });

        // Let the worker enter the loop.
        while s.workers_remaining.load(Ordering::Acquire) == 0 {
            std::thread::yield_now();
        }
        assert_eq!(s.workers_remaining.load(Ordering::Acquire), 1);

        // Signal exit; worker should observe on next iteration.
        s.active.store(false, Ordering::Release);
        handle.join().expect("worker panicked");
        assert_eq!(s.workers_remaining.load(Ordering::Acquire), 0);
    }

    /// Full end-to-end: 2 worker threads + main driving a parallel_for
    /// over 100 chunks. Verifies chunk dispatch works with multiple
    /// concurrent workers and that done_count always reaches the
    /// expected value.
    #[test]
    fn parallel_for_two_workers_full_dispatch() {
        let s = Box::new(PersistentSection::new());
        let s_ptr: *const PersistentSection = &*s;

        struct SendPtr(*const PersistentSection);
        impl SendPtr {
            #[inline]
            fn ptr(&self) -> *const PersistentSection {
                self.0
            }
        }
        // SAFETY: test-only wrapper — main keeps Box alive until
        // handle.join() below, so dereferencing from workers is sound.
        #[allow(unsafe_code)]
        unsafe impl Send for SendPtr {}
        #[allow(unsafe_code)]
        unsafe impl Sync for SendPtr {}
        let send_ptr = std::sync::Arc::new(SendPtr(s_ptr));

        let handles: Vec<_> = (0..2)
            .map(|_| {
                let p = std::sync::Arc::clone(&send_ptr);
                std::thread::spawn(move || {
                    // SAFETY: see `send_ptr` safety comment above.
                    #[allow(unsafe_code)]
                    let section = unsafe { &*p.ptr() };
                    #[allow(unsafe_code)]
                    unsafe {
                        section_worker_loop(section);
                    }
                })
            })
            .collect();

        while s.workers_remaining.load(Ordering::Acquire) < 2 {
            std::thread::yield_now();
        }

        // Drive several loops back-to-back to exercise the
        // publish/clear cycle multiple times.
        for _ in 0..5 {
            let seen = AtomicU32::new(0);
            s.parallel_for(20, |idx| {
                seen.fetch_add(idx as u32, Ordering::Release);
            });
            // Sum 0..=19 = 190.
            assert_eq!(seen.load(Ordering::Acquire), 190);
        }

        s.active.store(false, Ordering::Release);
        for h in handles {
            h.join().expect("worker panicked");
        }
        assert_eq!(s.workers_remaining.load(Ordering::Acquire), 0);
    }
}
