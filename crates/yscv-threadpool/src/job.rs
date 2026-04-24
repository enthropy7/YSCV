//! `JobRef` — inline 16-byte task representation that replaces
//! `Box<dyn FnOnce>` as the pool's transport type.
//!
//! Rayon's `rayon-core::job::JobRef` is the pattern: a pair of
//! `(fn_ptr, data_ptr)` carrying a monomorphized trampoline plus an
//! opaque pointer to a caller-owned heap or stack cell. Copy-safe,
//! Send by assertion, 16 bytes on 64-bit targets. The old `Box<dyn
//! FnOnce>` costs one `Box::new` allocation per task; at ~1200 submits
//! per Siamese-tracker inference on 6T that's a non-trivial 50-100 µs
//! on the hot path plus the cache footprint of every Box allocation.
//!
//! Used by `YscvPool::install`, `YscvPool::join`, and
//! `YscvPool::par_for_each_index`. Each caller holds the backing
//! storage (a `StackJob<F>` or a `MultiJob<F>` in the par_for_each case)
//! on its own stack frame and passes `&raw` pointers through; the pool
//! type-erases them into `JobRef`.
//!
//! # Safety invariant
//!
//! `JobRef::execute` dereferences `data_ptr` and invokes `fn_ptr`. The
//! caller must guarantee the backing storage outlives every worker's
//! dereference. The public `YscvPool` APIs enforce this by blocking
//! the caller until every job has executed:
//!
//! - `install` blocks on the result channel recv.
//! - `join` blocks on the second-job channel recv.
//! - `par_for_each_index` spin-waits on the stack latch.
//!
//! All three paths hold the relevant stack cells alive for the full
//! lifetime of any worker's dereference, matching rayon's own
//! discipline.

/// Type-erased job descriptor used by the pool's injector and deques.
///
/// The `fn_ptr` is a static function item (no heap, no closure env),
/// and `data` points to caller-owned storage that the `fn_ptr`
/// re-interprets. See module docs for the lifetime contract.
#[derive(Clone, Copy)]
pub(crate) struct JobRef {
    data: *const (),
    fn_ptr: unsafe fn(*const ()),
}

// SAFETY: `JobRef` carries raw pointers but the contract (caller-enforced
// via blocking) says the pointee outlives every dereference. The
// `fn_ptr` is a `fn` item, statically `'static`, always Send+Sync.
#[allow(unsafe_code)]
unsafe impl Send for JobRef {}
#[allow(unsafe_code)]
unsafe impl Sync for JobRef {}

impl JobRef {
    /// Construct a `JobRef` from a static function pointer and an
    /// opaque data pointer. Intended for use by the pool internals and
    /// matching helper types like `StackJob`; callers don't build these
    /// directly.
    ///
    /// # Safety
    /// - `fn_ptr` must remain callable for the lifetime of this JobRef.
    /// - `data` must point to a value whose representation matches what
    ///   `fn_ptr` expects when it casts the `*const ()` back.
    /// - `data` must outlive every call to `execute()` (caller must
    ///   block on a latch/channel until the worker finishes).
    #[allow(unsafe_code)]
    pub(crate) unsafe fn new(data: *const (), fn_ptr: unsafe fn(*const ())) -> Self {
        Self { data, fn_ptr }
    }

    /// Run this job. Exactly one call per `JobRef` — the trampoline may
    /// move out of `*data` (e.g. via `ptr::read` for `FnOnce` closures).
    ///
    /// # Safety
    /// - The caller promises `data` still points to a live value.
    /// - The trampoline may consume the pointee; calling `execute` twice
    ///   on the same `JobRef` is UB.
    #[inline]
    #[allow(unsafe_code)]
    pub(crate) unsafe fn execute(self) {
        unsafe {
            (self.fn_ptr)(self.data);
        }
    }
}
