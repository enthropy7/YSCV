//! Thread-local `ParallelScope` context for migrated kernels.
//!
//! The Part 2A refactor converts 33 parallel call sites from direct rayon
//! dispatch (`.par_chunks_mut`, `rayon::join`, ...) to going through the
//! pool-agnostic `ParallelScope` trait. Rather than threading a
//! `scope: &dyn ParallelScope` parameter through every kernel function
//! signature (10+ top-level functions, 30+ internal helpers), we install
//! the active scope into a thread-local for the duration of one inference
//! and let the kernels pick it up via [`with_scope`].
//!
//! This matches the pattern ONNX Runtime uses internally and lets yscv
//! keep the `&rayon::ThreadPool` signatures stable during the 2A.3
//! migration. The trade-off is that sites that spawn onto worker threads
//! (e.g. nested parallelism inside matmul) must explicitly propagate the
//! scope — done by capturing an `Arc<dyn ParallelScope>` into the task
//! closure, not by reading TLS on the worker.
//!
//! ## API
//!
//! * [`install_scope`] — runner wraps one inference in this, RAII-clears
//!   on drop so the TLS never leaks across runners.
//! * [`with_scope`] — kernels call this to read the currently-installed
//!   scope, if any. Returns `None` when called outside `install_scope`
//!   (e.g. in unit tests of a kernel in isolation) — callers fall back
//!   to rayon's global pool in that case.
//!
//! ## Why not pass the scope explicitly?
//!
//! Three reasons:
//! 1. We'd need to modify ~10 public function signatures and all their
//!    call sites — enormous churn for a bet that may or may not pay off.
//! 2. The current plan stage is "land infrastructure and measure"; if
//!    Part 2A turns into a dead-end like Part A did, rolling back a
//!    TLS-based integration is a 5-line delete. Rolling back a signature
//!    refactor is a much bigger job.
//! 3. TLS read overhead is negligible (~2ns) compared to the work each
//!    par_chunks_mut site does (µs-range).

use std::cell::RefCell;
use std::sync::Arc;

use yscv_threadpool::ParallelScope;

thread_local! {
    /// The scope installed by the current inference, if any. `None` means
    /// no one has wrapped us in `install_scope` — kernels fall back to
    /// their rayon path.
    static CURRENT_SCOPE: RefCell<Option<Arc<dyn ParallelScope>>> =
        const { RefCell::new(None) };
}

/// RAII guard that installs `scope` as the thread-local `CURRENT_SCOPE`
/// for this thread and clears it on drop. Use via [`install_scope`];
/// never construct directly.
pub struct ScopeGuard {
    /// Previous value to restore — makes `install_scope` nestable.
    prev: Option<Arc<dyn ParallelScope>>,
}

impl Drop for ScopeGuard {
    fn drop(&mut self) {
        let prev = self.prev.take();
        CURRENT_SCOPE.with(|c| {
            *c.borrow_mut() = prev;
        });
    }
}

/// Install `scope` as the active `ParallelScope` on the current thread
/// for as long as the returned guard is alive. Nests correctly — pre-
/// existing value is restored on drop.
///
/// Typical usage in the ONNX runner:
/// ```ignore
/// let _guard = yscv_kernels::install_scope(runner.scope_arc());
/// run_onnx_model_inner(&model, env)?;
/// // _guard drops here, TLS clears
/// ```
pub fn install_scope(scope: Arc<dyn ParallelScope>) -> ScopeGuard {
    let prev = CURRENT_SCOPE.with(|c| c.borrow_mut().replace(scope));
    ScopeGuard { prev }
}

/// Runs `f` with access to the current scope, if any. Returns whatever
/// `f` returns. When no scope is installed, `f` gets `None`.
///
/// Kernel sites use this to branch between `scope.par_chunks_mut_dyn(...)`
/// (fast path) and the legacy `rayon::par_chunks_mut(...)` fallback. The
/// fallback path is preserved so the crate still works standalone (e.g.
/// during criterion benches and unit tests where no runner is active).
pub fn with_scope<F, R>(f: F) -> R
where
    F: FnOnce(Option<&dyn ParallelScope>) -> R,
{
    CURRENT_SCOPE.with(|c| {
        let borrow = c.borrow();
        f(borrow.as_deref())
    })
}

/// Step 3 convenience: run `f` inside a session-scoped parallel region.
/// Bridges the dyn-compatible `ParallelScope::install_session` (which
/// uses `&mut dyn FnMut`) back to the ergonomic `FnOnce` shape callers
/// actually need. On `YscvPool`, enters a [`PersistentSection`] +
/// installs TLS before invoking `f`. On `rayon::ThreadPool`, invokes
/// `f` directly (no session support on that backend).
pub fn with_installed_session<R>(scope: &dyn ParallelScope, f: impl FnOnce() -> R) -> R {
    let mut f_opt = Some(f);
    let mut result: Option<R> = None;
    // Closure captures `result` and `f_opt` by &mut. `install_session`
    // calls the closure once; we panic if it either didn't call it or
    // called it twice.
    {
        let result_ref = &mut result;
        let f_opt_ref = &mut f_opt;
        scope.install_session(&mut || {
            let f = f_opt_ref
                .take()
                .expect("install_session must call closure at most once");
            *result_ref = Some(f());
        });
    }
    result.expect("install_session did not invoke the closure")
}

/// Dispatch a `par_chunks_mut().enumerate().for_each(...)` pattern through
/// the active scope when one is installed, otherwise fall back to rayon
/// directly. Hot-path call site for 14+ kernel sites migrated in 2A.3.
///
/// The closure's `F: Fn + Send + Sync` bound matches what both the trait
/// (`&dyn Fn`) and `rayon::par_chunks_mut` require, so the same closure
/// can feed either backend without adapter layers.
pub(crate) fn par_chunks_mut_dispatch<F>(data: &mut [f32], chunk_size: usize, f: F)
where
    F: Fn(usize, &mut [f32]) + Send + Sync,
{
    if chunk_size == 0 || data.is_empty() {
        return;
    }
    // Step 3: when a `PersistentSection` is installed in TLS (done by
    // `OnnxRunner::run` via `install_session`), route the dispatch through
    // the section's `parallel_for` instead of rayon. This eliminates the
    // per-op rayon fork/join epoch — workers are already spin-polling
    // `current_loop`, so dispatch is a single pointer store + chunk
    // counter CAS.
    if let Some(section) = yscv_threadpool::current_section() {
        let len = data.len();
        let n_chunks = len.div_ceil(chunk_size);
        // Share the mutable slice across workers via raw pointer +
        // disjoint chunk ranges — same discipline as
        // `ParallelScope::par_chunks_mut_dyn` for YscvPool.
        struct PtrWrap(*mut f32);
        impl PtrWrap {
            #[inline]
            fn as_ptr(&self) -> *mut f32 {
                self.0
            }
        }
        // SAFETY: `section.parallel_for` blocks until every chunk
        // completes, so `data` outlives all worker accesses. Chunks are
        // pairwise disjoint and in-bounds. `f: Fn + Sync` allows
        // concurrent invocation. PtrWrap's Send/Sync assertions are
        // scoped to the disjoint-chunk derivation only.
        #[allow(unsafe_code)]
        unsafe impl Send for PtrWrap {}
        #[allow(unsafe_code)]
        unsafe impl Sync for PtrWrap {}
        let wrap = PtrWrap(data.as_mut_ptr());
        section.parallel_for(n_chunks, move |i| {
            let start = i * chunk_size;
            let end = (start + chunk_size).min(len);
            // SAFETY: see the SAFETY block above.
            #[allow(unsafe_code)]
            let chunk =
                unsafe { std::slice::from_raw_parts_mut(wrap.as_ptr().add(start), end - start) };
            f(i, chunk);
        });
        return;
    }
    with_scope(|scope| {
        let run_seq = |data: &mut [f32]| {
            for (i, chunk) in data.chunks_mut(chunk_size).enumerate() {
                f(i, chunk);
            }
        };
        if let Some(s) = scope {
            if scope_seq_fastpath_enabled() && s.num_threads() <= 1 {
                run_seq(data);
                return;
            }
            s.par_chunks_mut_dyn(data, chunk_size, &f);
            return;
        }
        // Fallback: rayon's global pool. Preserved for standalone uses
        // (benches, kernel unit tests) where no runner has installed
        // a scope.
        use rayon::prelude::*;
        if scope_seq_fastpath_enabled() && rayon::current_num_threads() <= 1 {
            run_seq(data);
            return;
        }
        data.par_chunks_mut(chunk_size)
            .enumerate()
            .for_each(|(i, c)| f(i, c));
    });
}

#[inline]
fn scope_seq_fastpath_enabled() -> bool {
    use std::sync::OnceLock;
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| std::env::var_os("YSCV_NO_SCOPE_SEQ_FASTPATH").is_none())
}

#[cfg(test)]
mod tests {
    use super::*;
    use yscv_threadpool::YscvPool;

    #[test]
    fn with_scope_is_none_when_uninstalled() {
        let saw_scope = with_scope(|s| s.is_some());
        assert!(!saw_scope, "expected no scope installed on a fresh thread");
    }

    #[test]
    fn install_scope_sets_and_clears() {
        let pool: Arc<dyn ParallelScope> = Arc::new(YscvPool::new(2).unwrap());
        {
            let _g = install_scope(pool.clone());
            let saw = with_scope(|s| s.is_some());
            assert!(saw, "scope should be visible inside guard lifetime");
        }
        let after = with_scope(|s| s.is_some());
        assert!(!after, "scope should be cleared after guard drop");
    }

    #[test]
    fn install_scope_nests() {
        let inner: Arc<dyn ParallelScope> = Arc::new(YscvPool::new(1).unwrap());
        let outer: Arc<dyn ParallelScope> = Arc::new(YscvPool::new(2).unwrap());
        let _g1 = install_scope(outer.clone());
        let outer_nt = with_scope(|s| s.unwrap().num_threads());
        assert_eq!(outer_nt, 2);
        {
            let _g2 = install_scope(inner.clone());
            let inner_nt = with_scope(|s| s.unwrap().num_threads());
            assert_eq!(inner_nt, 1);
        }
        let restored_nt = with_scope(|s| s.unwrap().num_threads());
        assert_eq!(restored_nt, 2, "outer scope should be restored");
    }
}
