//! A Vec-like container guaranteeing 32-byte aligned allocation for SIMD operations.

use std::fmt;
use std::iter::FromIterator;
use std::ops::{Deref, DerefMut};

/// Alignment in bytes required for AVX operations.
// WHY 32: AVX/AVX2 aligned load/store (_mm256_load_ps) requires 32-byte alignment; also satisfies NEON (16B).
const ALIGN: usize = 32;

/// A Vec-like container that guarantees 32-byte alignment for the data pointer.
///
/// This is required for AVX/AVX2 SIMD instructions which expect 32-byte aligned
/// memory. Standard `Vec<f32>` only guarantees 4-byte alignment.
pub struct AlignedVec<T> {
    ptr: *mut T,
    len: usize,
    cap: usize,
}

// SAFETY: AlignedVec owns its data exclusively, so it is safe to Send/Sync
// when T is Send/Sync (same guarantees as Vec<T>).
#[allow(unsafe_code)]
unsafe impl<T: Send> Send for AlignedVec<T> {}
#[allow(unsafe_code)]
unsafe impl<T: Sync> Sync for AlignedVec<T> {}

impl<T> Default for AlignedVec<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> AlignedVec<T> {
    /// Creates a new empty `AlignedVec` with no allocation.
    #[inline]
    pub fn new() -> Self {
        Self {
            ptr: std::ptr::NonNull::dangling().as_ptr(),
            len: 0,
            cap: 0,
        }
    }

    /// Creates an `AlignedVec` with the given capacity (aligned to 32 bytes).
    #[allow(unsafe_code)]
    pub fn with_capacity(cap: usize) -> Self {
        if cap == 0 {
            return Self::new();
        }
        let ptr = alloc_aligned::<T>(cap);
        Self { ptr, len: 0, cap }
    }

    /// Creates an `AlignedVec` from an existing `Vec<T>`, copying data into
    /// aligned storage. Requires `T: Copy` to ensure bitwise copy is sound.
    #[allow(unsafe_code)]
    pub fn from_vec(v: Vec<T>) -> Self
    where
        T: Copy,
    {
        let len = v.len();
        if len == 0 {
            return Self::new();
        }
        let ptr = alloc_aligned::<T>(len);
        // SAFETY: ptr is valid for len elements, v.as_ptr() is valid for len elements,
        // they do not overlap (fresh allocation).
        unsafe {
            std::ptr::copy_nonoverlapping(v.as_ptr(), ptr, len);
        }
        // v is dropped here normally — T: Copy so no element destructors,
        // and Vec's drop frees the backing allocation.
        drop(v);
        Self { ptr, len, cap: len }
    }

    /// Returns the number of elements.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the vec is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns a raw pointer to the aligned data.
    #[inline]
    pub fn as_ptr(&self) -> *const T {
        self.ptr
    }

    /// Returns a mutable raw pointer to the aligned data.
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr
    }

    /// Returns an immutable slice over the contained elements.
    #[inline]
    #[allow(unsafe_code)]
    pub fn as_slice(&self) -> &[T] {
        if self.len == 0 {
            return &[];
        }
        // SAFETY: ptr is valid for self.len elements, properly aligned.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    /// Returns a mutable slice over the contained elements.
    #[inline]
    #[allow(unsafe_code)]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        if self.len == 0 {
            return &mut [];
        }
        // SAFETY: ptr is valid for self.len elements, properly aligned, uniquely borrowed.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }

    /// Appends a value to the end, growing if necessary.
    #[allow(unsafe_code)]
    pub fn push(&mut self, val: T) {
        if self.len == self.cap {
            self.grow();
        }
        // SAFETY: after grow, self.cap > self.len, so ptr.add(self.len) is valid.
        unsafe {
            self.ptr.add(self.len).write(val);
        }
        self.len += 1;
    }

    /// Grows the backing allocation (doubles capacity, minimum 4).
    #[allow(unsafe_code)]
    fn grow(&mut self) {
        let new_cap = if self.cap == 0 {
            4
        } else {
            self.cap.checked_mul(2).expect("capacity overflow")
        };
        let new_ptr = alloc_aligned::<T>(new_cap);
        if self.cap > 0 {
            if self.len > 0 {
                // SAFETY: old ptr valid for self.len elements, new ptr valid for new_cap >= self.len.
                unsafe {
                    std::ptr::copy_nonoverlapping(self.ptr, new_ptr, self.len);
                }
            }
            dealloc_aligned::<T>(self.ptr, self.cap);
        }
        self.ptr = new_ptr;
        self.cap = new_cap;
    }
}

impl<T: Copy> AlignedVec<T> {
    /// Creates an `AlignedVec` with `len` elements of **uninitialized** memory.
    ///
    /// The caller **must** write every element before reading any of them.
    /// This avoids the cost of zeroing the buffer when a subsequent SIMD pass
    /// will overwrite every element anyway.
    ///
    /// # Safety
    /// The allocation is valid and aligned, but the contents are indeterminate.
    /// Reading before writing is undefined behaviour.
    #[allow(unsafe_code)]
    #[inline]
    pub fn uninitialized(len: usize) -> Self {
        if len == 0 {
            return Self::new();
        }
        let ptr = alloc_aligned::<T>(len);
        Self { ptr, len, cap: len }
    }
}

impl<T: Default + Copy> AlignedVec<T> {
    /// Creates an `AlignedVec` of `len` elements, each set to `val`.
    #[allow(unsafe_code)]
    pub fn filled(len: usize, val: T) -> Self {
        if len == 0 {
            return Self::new();
        }
        let ptr = alloc_aligned::<T>(len);
        // SAFETY: ptr is valid for len elements.
        unsafe {
            for i in 0..len {
                ptr.add(i).write(val);
            }
        }
        Self { ptr, len, cap: len }
    }

    /// Creates an `AlignedVec` of `len` zero/default elements.
    pub fn zeros(len: usize) -> Self {
        Self::filled(len, T::default())
    }

    /// Creates an `AlignedVec` of `len` zero-initialized elements using `alloc_zeroed`.
    ///
    /// This is faster than `filled(len, T::default())` for large allocations because
    /// the OS can provide pre-zeroed pages without writing every byte.
    ///
    /// # Safety requirement
    /// Only correct when all-zero bytes is a valid representation for `T` (true for
    /// all primitive numeric types like f32, u8, i32, etc.).
    #[allow(unsafe_code)]
    pub fn calloc(len: usize) -> Self {
        if len == 0 {
            return Self::new();
        }
        let size = len
            .checked_mul(std::mem::size_of::<T>())
            .expect("allocation size overflow");
        let size = size.max(1);
        let layout =
            std::alloc::Layout::from_size_align(size, ALIGN).expect("invalid allocation layout");
        // SAFETY: layout has non-zero size. alloc_zeroed returns zeroed memory.
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
        if ptr.is_null() {
            std::alloc::handle_alloc_error(layout);
        }
        Self {
            ptr: ptr as *mut T,
            len,
            cap: len,
        }
    }
}

impl<T> Drop for AlignedVec<T> {
    #[allow(unsafe_code)]
    fn drop(&mut self) {
        if self.cap == 0 {
            return;
        }
        // Drop elements in place.
        unsafe {
            std::ptr::drop_in_place(std::ptr::slice_from_raw_parts_mut(self.ptr, self.len));
        }
        dealloc_aligned::<T>(self.ptr, self.cap);
    }
}

impl<T: Clone> Clone for AlignedVec<T> {
    #[allow(unsafe_code)]
    fn clone(&self) -> Self {
        if self.len == 0 {
            return Self::new();
        }
        let ptr = alloc_aligned::<T>(self.len);
        // SAFETY: ptr valid for self.len elements, we write each one via clone.
        unsafe {
            for i in 0..self.len {
                ptr.add(i).write((*self.ptr.add(i)).clone());
            }
        }
        Self {
            ptr,
            len: self.len,
            cap: self.len,
        }
    }
}

impl<T: fmt::Debug> fmt::Debug for AlignedVec<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.as_slice().iter()).finish()
    }
}

impl<T: PartialEq> PartialEq for AlignedVec<T> {
    fn eq(&self, other: &Self) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl<T> Deref for AlignedVec<T> {
    type Target = [T];

    #[inline]
    fn deref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T> DerefMut for AlignedVec<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

impl<T> FromIterator<T> for AlignedVec<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let iter = iter.into_iter();
        let (lower, _) = iter.size_hint();
        let mut v = AlignedVec::with_capacity(lower);
        for item in iter {
            v.push(item);
        }
        v
    }
}

// ── Allocation helpers ─────────────────────────────────────────────

/// Allocates aligned memory for `count` elements of type `T`.
/// Thread-local cache of freed aligned allocations.
/// Avoids repeated mmap/munmap for same-size buffers (like PyTorch's CachingAllocator).
/// Each entry: (pointer, byte_size).
#[allow(unsafe_code)]
mod alloc_cache {

    use std::cell::RefCell;

    // WHY 8: 8 cached allocations per thread balances memory reuse with bounded memory growth.
    const MAX_CACHED: usize = 8;
    const ALIGN: usize = super::ALIGN;

    /// RAII wrapper that properly frees all cached aligned allocations when dropped.
    /// This prevents SIGSEGV/SIGABRT during process exit from leaked cached pointers.
    struct AllocCache {
        entries: Vec<(*mut u8, usize)>,
    }

    impl AllocCache {
        const fn new() -> Self {
            Self {
                entries: Vec::new(),
            }
        }
    }

    impl Drop for AllocCache {
        fn drop(&mut self) {
            for &(ptr, size) in &self.entries {
                if !ptr.is_null()
                    && let Ok(layout) = std::alloc::Layout::from_size_align(size, ALIGN)
                {
                    unsafe {
                        std::alloc::dealloc(ptr, layout);
                    }
                }
            }
        }
    }

    thread_local! {
        static CACHE: RefCell<AllocCache> = const { RefCell::new(AllocCache::new()) };
    }

    pub(super) fn try_alloc(size: usize) -> Option<*mut u8> {
        if cfg!(miri) {
            return None;
        } // Disable cache under Miri to avoid false leak reports
        // Use try_with to gracefully handle TLS already destroyed during thread/process exit
        CACHE
            .try_with(|c| {
                let mut cache = c.borrow_mut();
                if let Some(pos) = cache.entries.iter().position(|&(_, s)| s == size) {
                    let (ptr, _) = cache.entries.swap_remove(pos);
                    Some(ptr)
                } else {
                    None
                }
            })
            .ok()
            .flatten()
    }

    pub(super) fn try_dealloc(ptr: *mut u8, size: usize) -> bool {
        if cfg!(miri) {
            return false;
        } // Always free under Miri to avoid false leak reports
        // Use try_with: if TLS is destroyed (thread exiting), fall through to real dealloc
        CACHE
            .try_with(|c| {
                let mut cache = c.borrow_mut();
                if cache.entries.len() < MAX_CACHED {
                    cache.entries.push((ptr, size));
                    true
                } else {
                    false
                }
            })
            .unwrap_or(false)
    }
}

#[allow(unsafe_code)]
fn alloc_aligned<T>(count: usize) -> *mut T {
    assert!(count > 0, "cannot allocate zero-sized aligned buffer");
    let size = count
        .checked_mul(std::mem::size_of::<T>())
        .expect("allocation size overflow");
    let size = size.max(1);

    // Try thread-local cache first
    if let Some(ptr) = alloc_cache::try_alloc(size) {
        return ptr as *mut T;
    }

    let layout =
        std::alloc::Layout::from_size_align(size, ALIGN).expect("invalid allocation layout");
    // SAFETY: layout has non-zero size.
    let ptr = unsafe { std::alloc::alloc(layout) };
    if ptr.is_null() {
        std::alloc::handle_alloc_error(layout);
    }
    ptr as *mut T
}

/// Deallocates aligned memory previously allocated with `alloc_aligned`.
#[allow(unsafe_code)]
fn dealloc_aligned<T>(ptr: *mut T, cap: usize) {
    let size = match cap.checked_mul(std::mem::size_of::<T>()) {
        Some(s) => s.max(1),
        None => return, // overflow — cannot reconstruct layout, leak rather than panic in Drop
    };

    // Try to cache instead of freeing
    if alloc_cache::try_dealloc(ptr as *mut u8, size) {
        return;
    }

    if let Ok(layout) = std::alloc::Layout::from_size_align(size, ALIGN) {
        // SAFETY: ptr was allocated with this layout via alloc_aligned.
        unsafe {
            std::alloc::dealloc(ptr as *mut u8, layout);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alignment_is_32_bytes() {
        let v = AlignedVec::<f32>::with_capacity(64);
        assert_eq!(v.as_ptr() as usize % 32, 0);
    }

    #[test]
    fn filled_and_len() {
        let v = AlignedVec::filled(100, 1.0f32);
        assert_eq!(v.len(), 100);
        assert!(v.iter().all(|&x| x == 1.0));
        assert_eq!(v.as_ptr() as usize % 32, 0);
    }

    #[test]
    fn zeros_default() {
        let v = AlignedVec::<f32>::zeros(16);
        assert_eq!(v.len(), 16);
        assert!(v.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn from_vec_copies_and_aligns() {
        let orig = vec![1.0f32, 2.0, 3.0, 4.0];
        let aligned = AlignedVec::from_vec(orig);
        assert_eq!(aligned.as_slice(), &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(aligned.as_ptr() as usize % 32, 0);
    }

    #[test]
    fn push_and_grow() {
        let mut v = AlignedVec::<f32>::new();
        for i in 0..100 {
            v.push(i as f32);
        }
        assert_eq!(v.len(), 100);
        assert_eq!(v[50], 50.0);
        assert_eq!(v.as_ptr() as usize % 32, 0);
    }

    #[test]
    fn clone_preserves_alignment() {
        let v = AlignedVec::filled(10, 42.0f32);
        let v2 = v.clone();
        assert_eq!(v.as_slice(), v2.as_slice());
        assert_eq!(v2.as_ptr() as usize % 32, 0);
    }

    #[test]
    fn from_iterator() {
        let v: AlignedVec<f32> = (0..10).map(|i| i as f32).collect();
        assert_eq!(v.len(), 10);
        assert_eq!(v[5], 5.0);
        assert_eq!(v.as_ptr() as usize % 32, 0);
    }

    #[test]
    fn empty_vec_operations() {
        let v = AlignedVec::<f32>::new();
        assert!(v.is_empty());
        assert_eq!(v.len(), 0);
        assert_eq!(v.as_slice(), &[] as &[f32]);
    }

    #[test]
    fn deref_slice_access() {
        let v = AlignedVec::filled(5, 3.0f32);
        // Test that Deref to [T] works
        let sum: f32 = v.iter().sum();
        assert_eq!(sum, 15.0);
    }
}
