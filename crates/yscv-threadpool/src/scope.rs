//! `ParallelScope` — uniform parallelism trait over `YscvPool` and
//! `rayon::ThreadPool`. Lets the rest of the yscv stack be pool-agnostic
//! and switch backends at runtime via `YSCV_POOL=rayon|yscv`.
//!
//! Surface grew during the Part 2A full-rayon-replacement refactor to cover
//! every parallel call site in yscv-kernels / yscv-onnx:
//!   - `par_for_each_index(n, f)` — mirrors `(0..n).into_par_iter().for_each(f)`.
//!   - `par_chunks_mut_dyn(data, chunk, f)` — mirrors `data.par_chunks_mut(chunk).enumerate().for_each(f)`.
//!     The enumeration form is chosen because most kernel sites need the
//!     chunk index to compute row/tile offsets into co-owned inputs.
//!   - `join_dyn(a, b)` — mirrors `rayon::join(a, b)` over `FnMut`.
//!   - `current_worker_index()` — mirrors `rayon::current_thread_index()`,
//!     used by matmul.rs to avoid re-entrant pool dispatch.
//!   - `num_threads()` — same as both pools' existing method.
//!
//! More complex rayon patterns (`scope` with arbitrary spawn trees, `par_sort`,
//! data-parallel reductions over borrowed structures) aren't used by the
//! inference hot path, so they're not in the trait.

use rayon::prelude::*;

use crate::YscvPool;

/// Trait abstracting over thread-pool backends for the inference hot path.
pub trait ParallelScope: Send + Sync {
    /// Step 3: enter a session-scoped parallel region. On backends that
    /// support persistent sections (currently [`YscvPool`]), all pool
    /// workers enter a tight-poll loop, and parallel dispatch inside
    /// `f` routes through the section's single-atomic-pointer
    /// mechanism instead of rayon's fork-join. On rayon, this is a
    /// pass-through that just invokes `f`.
    ///
    /// The downstream dispatch hook (`yscv-kernels`'s
    /// `par_chunks_mut_dispatch`) reads the thread-local section
    /// pointer via [`crate::current_section`] and picks the session
    /// path when present.
    ///
    /// Uses a `&mut dyn FnMut()` rather than generic `FnOnce` so the
    /// trait stays dyn-compatible (the runner holds a
    /// `Arc<dyn ParallelScope>`).
    fn install_session(&self, f: &mut dyn FnMut());

    /// Runs `f(i)` for each `i` in `0..count`, in parallel. Blocks until
    /// every iteration completes. Equivalent to rayon's
    /// `(0..count).into_par_iter().for_each(f)`.
    fn par_for_each_index(&self, count: usize, f: &(dyn Fn(usize) + Send + Sync));

    /// Runs `f(chunk_idx, chunk)` for each `chunk_size`-wide slice of `data`,
    /// in parallel. The last chunk may be short when `data.len()` is not a
    /// multiple of `chunk_size`. Blocks until every chunk completes.
    /// Equivalent to `data.par_chunks_mut(chunk_size).enumerate().for_each(…)`.
    fn par_chunks_mut_dyn(
        &self,
        data: &mut [f32],
        chunk_size: usize,
        f: &(dyn Fn(usize, &mut [f32]) + Send + Sync),
    );

    /// Runs `a` and `b` in parallel, returns both results. Equivalent
    /// to `rayon::join(a, b)`.
    fn join_dyn(&self, a: &mut (dyn FnMut() + Send), b: &mut (dyn FnMut() + Send));

    /// The caller's worker index, 0..num_threads, if running on one of this
    /// pool's workers. `None` when called from outside the pool (e.g. from
    /// the main thread before `install`). Used by matmul.rs at 538/684 to
    /// decide whether to further parallelise — we don't want nested pool
    /// dispatch on an already-parallel worker.
    fn current_worker_index(&self) -> Option<usize>;

    /// Number of worker threads.
    fn num_threads(&self) -> usize;
}

impl ParallelScope for YscvPool {
    fn install_session(&self, f: &mut dyn FnMut()) {
        // Create section + hold workers in `section_worker_loop` for
        // the closure's lifetime. Install TLS so downstream dispatch
        // routes through the section's parallel_for.
        self.enter_section(|section| {
            let _guard = section.install_tls();
            f();
        });
    }

    fn par_for_each_index(&self, count: usize, f: &(dyn Fn(usize) + Send + Sync)) {
        // Bridge the dyn reference into the generic API. Closure captures
        // `f` by reference — `par_for_each_index` blocks so lifetime is OK.
        self.par_for_each_index(count, f);
    }

    fn par_chunks_mut_dyn(
        &self,
        data: &mut [f32],
        chunk_size: usize,
        f: &(dyn Fn(usize, &mut [f32]) + Send + Sync),
    ) {
        if chunk_size == 0 || data.is_empty() {
            return;
        }
        let len = data.len();
        let n_chunks = len.div_ceil(chunk_size);
        // Share the slice across workers via its raw pointer + length.
        // Disjoint chunk ranges give non-overlapping `&mut [f32]` slices,
        // which is exactly the rayon `par_chunks_mut` safety invariant.
        //
        // SAFETY: (1) `par_for_each_index` blocks until every chunk
        // finishes, so `data` outlives all task accesses. (2) Chunk ranges
        // `[i*chunk_size .. min((i+1)*chunk_size, len)]` are pairwise
        // disjoint and in-bounds, so no two tasks alias the same byte.
        // (3) `f: Fn` is `Send + Sync`, required by the trait bound.
        // (4) PtrWrap's Send/Sync are safe to assert because the pointer
        //     is only used to derive the disjoint chunk slices described
        //     in (1)-(3); we never share or read raw memory beyond them.
        // The `as_ptr()` accessor routes through `&self`, which forces
        // the closure to capture the whole `PtrWrap` struct rather than
        // the bare `*mut f32` field — Rust 2021 disjoint captures would
        // otherwise strip the Send/Sync impls off and leave us with a
        // raw pointer capture the compiler rejects.
        struct PtrWrap(*mut f32);
        impl PtrWrap {
            #[inline]
            fn as_ptr(&self) -> *mut f32 {
                self.0
            }
        }
        #[allow(unsafe_code)]
        unsafe impl Send for PtrWrap {}
        #[allow(unsafe_code)]
        unsafe impl Sync for PtrWrap {}
        let wrap = PtrWrap(data.as_mut_ptr());
        self.par_for_each_index(n_chunks, move |i| {
            let start = i * chunk_size;
            let end = (start + chunk_size).min(len);
            #[allow(unsafe_code)]
            let chunk =
                unsafe { std::slice::from_raw_parts_mut(wrap.as_ptr().add(start), end - start) };
            f(i, chunk);
        });
    }

    fn join_dyn(&self, a: &mut (dyn FnMut() + Send), b: &mut (dyn FnMut() + Send)) {
        // Wrap the &mut dyn FnMut in closures that take ownership of the
        // reference. Uses a single call each — FnMut → FnOnce semantically.
        self.join(a, b);
    }

    fn current_worker_index(&self) -> Option<usize> {
        self.current_worker_index()
    }

    fn num_threads(&self) -> usize {
        YscvPool::num_threads(self)
    }
}

impl ParallelScope for rayon::ThreadPool {
    fn install_session(&self, f: &mut dyn FnMut()) {
        // rayon has no persistent section — just invoke the closure.
        // CURRENT_SECTION_TLS stays null; downstream dispatch falls
        // through to rayon's normal fork-join path.
        f();
    }

    fn par_for_each_index(&self, count: usize, f: &(dyn Fn(usize) + Send + Sync)) {
        // Run inside `install` so par_iter picks up the correct pool.
        self.install(|| (0..count).into_par_iter().for_each(f));
    }

    fn par_chunks_mut_dyn(
        &self,
        data: &mut [f32],
        chunk_size: usize,
        f: &(dyn Fn(usize, &mut [f32]) + Send + Sync),
    ) {
        if chunk_size == 0 || data.is_empty() {
            return;
        }
        // Skip `self.install(...)` — rayon's par_iter uses the ambient
        // pool via TLS, which the ONNX runner's outer `pool.install(...)`
        // already installed. A nested `install` call per site adds real
        // cost (measured ~4% regression at 6T on tracker) with no
        // correctness benefit.
        data.par_chunks_mut(chunk_size)
            .enumerate()
            .for_each(|(i, c)| f(i, c));
    }

    fn join_dyn(&self, a: &mut (dyn FnMut() + Send), b: &mut (dyn FnMut() + Send)) {
        // Rely on ambient install (same reasoning as above).
        rayon::join(a, b);
    }

    fn current_worker_index(&self) -> Option<usize> {
        rayon::current_thread_index()
    }

    fn num_threads(&self) -> usize {
        self.current_num_threads()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn sum_via_scope(scope: &dyn ParallelScope, count: usize) -> usize {
        let total = Arc::new(AtomicUsize::new(0));
        let t = Arc::clone(&total);
        let f: Box<dyn Fn(usize) + Send + Sync> = Box::new(move |i| {
            t.fetch_add(i, Ordering::Relaxed);
        });
        scope.par_for_each_index(count, f.as_ref());
        total.load(Ordering::Relaxed)
    }

    #[test]
    fn yscv_pool_scope_sums_correctly() {
        let pool = YscvPool::new(4).unwrap();
        assert_eq!(sum_via_scope(&pool, 1000), 499_500);
    }

    #[test]
    fn rayon_pool_scope_sums_correctly() {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        assert_eq!(sum_via_scope(&pool, 1000), 499_500);
    }

    fn doubles_chunks_via_scope(scope: &dyn ParallelScope) -> Vec<f32> {
        let mut data: Vec<f32> = (0..32).map(|i| i as f32).collect();
        let f: Box<dyn Fn(usize, &mut [f32]) + Send + Sync> = Box::new(|_idx, chunk| {
            for v in chunk {
                *v *= 2.0;
            }
        });
        scope.par_chunks_mut_dyn(&mut data, 4, f.as_ref());
        data
    }

    #[test]
    fn yscv_pool_par_chunks_mut_dyn_matches_rayon() {
        let pool = YscvPool::new(4).unwrap();
        let out = doubles_chunks_via_scope(&pool);
        let expected: Vec<f32> = (0..32).map(|i| (i as f32) * 2.0).collect();
        assert_eq!(out, expected);
    }

    #[test]
    fn rayon_pool_par_chunks_mut_dyn_matches_rayon() {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        let out = doubles_chunks_via_scope(&pool);
        let expected: Vec<f32> = (0..32).map(|i| (i as f32) * 2.0).collect();
        assert_eq!(out, expected);
    }

    #[test]
    fn par_chunks_mut_dyn_handles_uneven_last_chunk() {
        let pool = YscvPool::new(2).unwrap();
        let mut data: Vec<f32> = vec![1.0; 17];
        let f: Box<dyn Fn(usize, &mut [f32]) + Send + Sync> = Box::new(|idx, chunk| {
            for v in chunk {
                *v += idx as f32;
            }
        });
        pool.par_chunks_mut_dyn(&mut data, 5, f.as_ref());
        // chunks: [0..5]=idx 0, [5..10]=idx 1, [10..15]=idx 2, [15..17]=idx 3
        let expected = vec![
            1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0, 3.0, 4.0, 4.0,
        ];
        assert_eq!(data, expected);
    }

    #[test]
    fn yscv_pool_current_worker_index_is_some_inside_install() {
        let pool = YscvPool::new(2).unwrap();
        let idx = pool.install(|| pool.current_worker_index());
        assert!(idx.is_some(), "expected Some(worker_idx) inside install");
    }

    #[test]
    fn yscv_pool_current_worker_index_is_none_on_main() {
        let pool = YscvPool::new(2).unwrap();
        assert_eq!(pool.current_worker_index(), None);
    }
}
