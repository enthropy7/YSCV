/// A tensor that lives on GPU memory. No host copy until explicitly requested.
#[non_exhaustive]
pub struct GpuBuffer {
    pub(crate) buffer: wgpu::Buffer,
    /// Number of f32 elements.
    pub(crate) size: usize,
    pub(crate) shape: Vec<usize>,
}

impl GpuBuffer {
    /// Returns the shape of this GPU-resident tensor.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Returns the number of f32 elements.
    pub fn len(&self) -> usize {
        self.size
    }

    /// Returns true if this buffer contains no elements.
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    /// Consume this GpuBuffer and return the inner wgpu::Buffer + element count
    /// so the caller can return it to a buffer pool.
    pub fn into_raw(self) -> (wgpu::Buffer, usize) {
        (self.buffer, self.size)
    }

    /// Get a reference to the underlying wgpu::Buffer.
    pub fn raw_buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }

    /// Construct a GpuBuffer from raw parts (for compiled plan replay).
    /// The caller is responsible for ensuring the buffer matches the declared size/shape.
    pub fn from_raw_parts(buffer: wgpu::Buffer, size: usize, shape: Vec<usize>) -> Self {
        Self {
            buffer,
            size,
            shape,
        }
    }
}

// ── GpuBackend ─────────────────────────────────────────────────────

/// Simple size-bucketed buffer pool for GPU buffer reuse across dispatches.
/// Uses `RefCell` for interior mutability so `&self` methods can pool/reclaim.
pub(crate) struct BufferPool {
    /// Available output buffers keyed by capacity in bytes.
    pub(crate) output: std::cell::RefCell<Vec<(u64, wgpu::Buffer)>>,
    /// Available storage buffers keyed by capacity in bytes.
    pub(crate) storage: std::cell::RefCell<Vec<(u64, wgpu::Buffer)>>,
    /// Maximum pool depth per category.
    pub(crate) max_depth: usize,
    /// Total allocations saved (diagnostic counter).
    pub(crate) hits: std::cell::Cell<u64>,
}

impl BufferPool {
    pub(crate) fn new(max_depth: usize) -> Self {
        Self {
            output: std::cell::RefCell::new(Vec::with_capacity(max_depth)),
            storage: std::cell::RefCell::new(Vec::with_capacity(max_depth)),
            max_depth,
            hits: std::cell::Cell::new(0),
        }
    }

    /// Try to reclaim an output buffer with at least `size_bytes` capacity.
    /// Uses best-fit: picks the smallest buffer that's >= size_bytes and <= 4x.
    pub(crate) fn take_output(&self, size_bytes: u64) -> Option<wgpu::Buffer> {
        let mut pool = self.output.borrow_mut();
        let max_cap = size_bytes.saturating_mul(4);
        let mut best: Option<(usize, u64)> = None;
        for (i, &(cap, _)) in pool.iter().enumerate() {
            if cap >= size_bytes && cap <= max_cap && best.is_none_or(|(_, bc)| cap < bc) {
                best = Some((i, cap));
            }
        }
        if let Some((pos, _)) = best {
            self.hits.set(self.hits.get() + 1);
            Some(pool.swap_remove(pos).1)
        } else {
            None
        }
    }

    /// Return an output buffer to the pool for future reuse.
    pub(crate) fn return_output(&self, size_bytes: u64, buf: wgpu::Buffer) {
        let mut pool = self.output.borrow_mut();
        if pool.len() < self.max_depth {
            pool.push((size_bytes, buf));
        }
        // else: drop the buffer (exceeds pool capacity)
    }

    /// Try to reclaim a storage buffer with at least `size_bytes` capacity.
    pub(crate) fn take_storage(&self, size_bytes: u64) -> Option<wgpu::Buffer> {
        let mut pool = self.storage.borrow_mut();
        let max_cap = size_bytes.saturating_mul(4);
        let mut best: Option<(usize, u64)> = None;
        for (i, &(cap, _)) in pool.iter().enumerate() {
            if cap >= size_bytes && cap <= max_cap && best.is_none_or(|(_, bc)| cap < bc) {
                best = Some((i, cap));
            }
        }
        if let Some((pos, _)) = best {
            self.hits.set(self.hits.get() + 1);
            Some(pool.swap_remove(pos).1)
        } else {
            None
        }
    }

    /// Return a storage buffer to the pool.
    pub(crate) fn return_storage(&self, size_bytes: u64, buf: wgpu::Buffer) {
        let mut pool = self.storage.borrow_mut();
        if pool.len() < self.max_depth {
            pool.push((size_bytes, buf));
        }
    }

    /// Total cache hits (diagnostic).
    pub(crate) fn cache_hits(&self) -> u64 {
        self.hits.get()
    }
}
