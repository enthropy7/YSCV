//! Rust callback dispatcher for RKNN custom ops.
//!
//! When a user registers a `CustomOp` with an attached `CustomOpHandler`,
//! we reserve a slot in the static [`SLOTS`] table and hand the matching
//! per-slot trampoline function pointers to the SDK. The SDK then calls
//! those C ABI trampolines from inside `rknn_run`; each trampoline knows
//! its own slot index at compile time and dispatches to the registered
//! Rust closure.
//!
//! Up to [`MAX_CUSTOM_OP_SLOTS`] (= 16) handlers can coexist per process.
//! That covers any realistic NPU model — typical YOLO-class graphs use
//! 0–2 custom ops.

use super::ffi::{
    FnRknnCustomOpCompute, FnRknnCustomOpDestroy, FnRknnCustomOpInit, FnRknnCustomOpPrepare,
    RknnCustomOpAttrRaw, RknnCustomOpContextRaw, RknnCustomOpTensorRaw, RknnFunctions,
    RknnTensorAttr, rknn_error_name,
};
use crate::KernelError;
use std::ffi::{CString, c_void};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};

const RKNN_SUCC: i32 = 0;
const RKNN_ERR_FAIL: i32 = -1;
const RKNN_ERR_CTX_INVALID: i32 = -7;

/// Maximum simultaneously registered custom-op handlers per process.
pub const MAX_CUSTOM_OP_SLOTS: usize = 16;

// ── Public surface ──────────────────────────────────────────────────

/// Trait implemented by user-supplied callbacks for a custom RKNN op.
///
/// `compute` is required; `init` / `prepare` / `destroy` are optional
/// lifecycle hooks. All four run on whatever thread the RKNN runtime
/// chooses — implementations must be `Send + Sync`.
pub trait CustomOpHandler: Send + Sync + 'static {
    /// Called once per op instance after construction. Default: no-op.
    fn init(
        &self,
        _ctx: &mut CustomOpContext<'_>,
        _inputs: &[CustomOpTensor<'_>],
        _outputs: &[CustomOpTensor<'_>],
    ) -> Result<(), KernelError> {
        Ok(())
    }

    /// Called before each `compute` to allow shape-dependent setup.
    /// Default: no-op.
    fn prepare(
        &self,
        _ctx: &mut CustomOpContext<'_>,
        _inputs: &[CustomOpTensor<'_>],
        _outputs: &[CustomOpTensor<'_>],
    ) -> Result<(), KernelError> {
        Ok(())
    }

    /// Required: perform the actual op computation.
    fn compute(
        &self,
        ctx: &mut CustomOpContext<'_>,
        inputs: &[CustomOpTensor<'_>],
        outputs: &[CustomOpTensor<'_>],
    ) -> Result<(), KernelError>;

    /// Called once when the op is being destroyed. Default: no-op.
    fn destroy(&self, _ctx: &mut CustomOpContext<'_>) -> Result<(), KernelError> {
        Ok(())
    }
}

/// Mutable view onto the SDK-owned `rknn_custom_op_context` for one call.
pub struct CustomOpContext<'a> {
    raw: &'a mut RknnCustomOpContextRaw,
    funcs: Arc<RknnFunctions>,
}

impl<'a> CustomOpContext<'a> {
    /// Backend target the SDK selected for this op (CPU or GPU).
    pub fn target(&self) -> u32 {
        self.raw.target
    }

    /// OpenCL context handle (only valid when `target() == GPU == 2`).
    pub fn cl_context(&self) -> *mut c_void {
        self.raw.gpu_ctx.cl_context
    }

    /// OpenCL command-queue handle (only valid when target is GPU).
    pub fn cl_command_queue(&self) -> *mut c_void {
        self.raw.gpu_ctx.cl_command_queue
    }

    /// OpenCL kernel handle (only valid when target is GPU).
    pub fn cl_kernel(&self) -> *mut c_void {
        self.raw.gpu_ctx.cl_kernel
    }

    /// User-managed scratch pointer. Set in `init`, read in
    /// `prepare`/`compute`, freed in `destroy`. SDK does not interpret it.
    pub fn priv_data(&self) -> *mut c_void {
        self.raw.priv_data
    }

    /// Replace the user-managed scratch pointer.
    ///
    /// # Safety
    /// Caller is responsible for ownership: typically a `Box::into_raw` in
    /// `init` paired with `Box::from_raw` in `destroy`.
    pub unsafe fn set_priv_data(&mut self, ptr: *mut c_void) {
        self.raw.priv_data = ptr;
    }

    /// Look up a named ONNX attribute embedded in the op (e.g. `"epsilon"`,
    /// `"axis"`). Returns `None` when the runtime symbol is unavailable.
    pub fn get_attr(&mut self, name: &str) -> Option<CustomOpAttr> {
        let getter = self.funcs.custom_op_get_op_attr?;
        let cname = CString::new(name).ok()?;
        let mut raw = RknnCustomOpAttrRaw {
            name: [0u8; 256],
            dtype: 0,
            n_elems: 0,
            data: std::ptr::null_mut(),
        };
        // SAFETY: getter is a valid SDK function pointer; raw is a valid
        // out-parameter; cname is null-terminated.
        unsafe {
            getter(self.raw, cname.as_ptr().cast(), &mut raw);
        }
        if raw.data.is_null() || raw.n_elems == 0 {
            return None;
        }
        let elem_bytes = match raw.dtype {
            0 => 4, // FLOAT32
            1 => 2, // FLOAT16
            2 | 3 => 1,
            4 | 5 => 2,
            6 | 7 => 4,
            8 => 8,
            9 => 1,  // BOOL
            10 => 1, // INT4 (packed; rounded up)
            11 => 2, // BFLOAT16
            _ => 4,
        };
        let len = (raw.n_elems as usize) * elem_bytes;
        // SAFETY: SDK promises raw.data points to n_elems × elem_bytes
        // valid bytes for the lifetime of this call.
        let data: Vec<u8> =
            unsafe { std::slice::from_raw_parts(raw.data.cast::<u8>(), len).to_vec() };
        Some(CustomOpAttr {
            dtype: raw.dtype,
            data,
        })
    }
}

/// Read-only view of a tensor passed to a custom op callback.
pub struct CustomOpTensor<'a> {
    raw: &'a RknnCustomOpTensorRaw,
}

impl<'a> CustomOpTensor<'a> {
    /// Tensor descriptor (shape, dtype, strides — see `RknnTensorAttr`).
    pub fn attr(&self) -> &RknnTensorAttr {
        &self.raw.attr
    }

    /// Pointer to the tensor buffer. Layout determined by `attr().fmt`.
    pub fn virt_addr(&self) -> *mut c_void {
        self.raw.mem.virt_addr
    }

    /// DMA-BUF file descriptor backing the buffer (`-1` if not exported).
    pub fn fd(&self) -> i32 {
        self.raw.mem.fd
    }

    /// Buffer size in bytes.
    pub fn size(&self) -> u32 {
        self.raw.mem.size
    }

    /// Read the buffer as a raw byte slice. Length = `size()`.
    ///
    /// # Safety
    /// Caller must ensure buffer is CPU-mapped (default for `alloc_mem`,
    /// not guaranteed for some `wrap_phys` configurations).
    pub unsafe fn as_bytes(&self) -> &[u8] {
        // SAFETY: caller guarantees the buffer is CPU-mapped and the
        // backing memory remains valid for the slice lifetime (bounded by
        // the SDK callback duration).
        unsafe {
            std::slice::from_raw_parts(
                self.raw.mem.virt_addr.cast::<u8>(),
                self.raw.mem.size as usize,
            )
        }
    }

    /// Mutable byte view. Same safety contract as `as_bytes`.
    ///
    /// # Safety
    /// Caller must guarantee no concurrent reader exists on this tensor.
    pub unsafe fn as_bytes_mut(&mut self) -> &mut [u8] {
        // SAFETY: same as as_bytes; caller guarantees exclusive access.
        unsafe {
            std::slice::from_raw_parts_mut(
                self.raw.mem.virt_addr.cast::<u8>(),
                self.raw.mem.size as usize,
            )
        }
    }
}

/// One ONNX attribute value retrieved via `CustomOpContext::get_attr`.
pub struct CustomOpAttr {
    /// `RknnTensorType` discriminant (e.g. 0 = FLOAT32, 6 = INT32).
    pub dtype: u32,
    /// Raw little-endian bytes; interpret based on `dtype`.
    pub data: Vec<u8>,
}

// ── Trampoline dispatch table ───────────────────────────────────────

struct SlotState {
    handler: Option<Arc<dyn CustomOpHandler>>,
    funcs: Option<Arc<RknnFunctions>>,
}

// Each slot is a RwLock so concurrent reads (compute callbacks on different
// ops) don't block each other; allocation/release take the write lock.
// `clippy::declare_interior_mutable_const` flags this pattern, but the
// `[EMPTY; N]` repeat-expr is the only way to initialise a static array of
// non-Copy types in current stable Rust — we accept the lint here.
#[allow(clippy::declare_interior_mutable_const)]
const EMPTY: RwLock<SlotState> = RwLock::new(SlotState {
    handler: None,
    funcs: None,
});
static SLOTS: [RwLock<SlotState>; MAX_CUSTOM_OP_SLOTS] = [EMPTY; MAX_CUSTOM_OP_SLOTS];
static NEXT_FREE_HINT: AtomicUsize = AtomicUsize::new(0);

/// Reserve a slot for a new handler. Linear scan from `NEXT_FREE_HINT` so
/// fresh allocations fast-path past freed slots without retrying earlier
/// indices on every call.
pub(crate) fn allocate_slot(
    handler: Arc<dyn CustomOpHandler>,
    funcs: Arc<RknnFunctions>,
) -> Result<usize, KernelError> {
    let start = NEXT_FREE_HINT.load(Ordering::Relaxed);
    for offset in 0..MAX_CUSTOM_OP_SLOTS {
        let idx = (start + offset) % MAX_CUSTOM_OP_SLOTS;
        let mut guard = SLOTS[idx].write().unwrap();
        if guard.handler.is_none() {
            guard.handler = Some(handler);
            guard.funcs = Some(funcs);
            NEXT_FREE_HINT.store((idx + 1) % MAX_CUSTOM_OP_SLOTS, Ordering::Relaxed);
            return Ok(idx);
        }
    }
    Err(KernelError::Rknn {
        message: format!("too many custom-op handlers registered (>{MAX_CUSTOM_OP_SLOTS})"),
    })
}

/// Free a slot reserved by `allocate_slot`. Called from
/// `CustomOpRegistration::Drop`.
pub(crate) fn release_slot(idx: usize) {
    if idx >= MAX_CUSTOM_OP_SLOTS {
        return;
    }
    let mut guard = SLOTS[idx].write().unwrap();
    guard.handler = None;
    guard.funcs = None;
}

#[derive(Clone, Copy)]
pub(crate) struct TrampolineSet {
    pub init: FnRknnCustomOpInit,
    pub prepare: FnRknnCustomOpPrepare,
    pub compute: FnRknnCustomOpCompute,
    pub destroy: FnRknnCustomOpDestroy,
}

#[derive(Clone, Copy)]
enum CbKind {
    Init,
    Prepare,
    Compute,
}

unsafe fn dispatch(
    slot: usize,
    kind: CbKind,
    op_ctx: *mut RknnCustomOpContextRaw,
    ins: *mut RknnCustomOpTensorRaw,
    n_in: u32,
    outs: *mut RknnCustomOpTensorRaw,
    n_out: u32,
) -> i32 {
    if op_ctx.is_null() {
        return RKNN_ERR_CTX_INVALID;
    }
    let guard = SLOTS[slot].read().unwrap();
    let handler = match guard.handler.as_ref() {
        Some(h) => h.clone(),
        None => return RKNN_ERR_CTX_INVALID,
    };
    let funcs = match guard.funcs.as_ref() {
        Some(f) => f.clone(),
        None => return RKNN_ERR_CTX_INVALID,
    };
    drop(guard);

    // SAFETY: SDK guarantees op_ctx, ins, outs valid for the call duration.
    let mut ctx = CustomOpContext {
        raw: unsafe { &mut *op_ctx },
        funcs,
    };
    let ins_slice: &[RknnCustomOpTensorRaw] = if ins.is_null() || n_in == 0 {
        &[]
    } else {
        unsafe { std::slice::from_raw_parts(ins, n_in as usize) }
    };
    let outs_slice: &[RknnCustomOpTensorRaw] = if outs.is_null() || n_out == 0 {
        &[]
    } else {
        unsafe { std::slice::from_raw_parts(outs, n_out as usize) }
    };
    let ins_wrapped: Vec<CustomOpTensor<'_>> = ins_slice
        .iter()
        .map(|r| CustomOpTensor { raw: r })
        .collect();
    let outs_wrapped: Vec<CustomOpTensor<'_>> = outs_slice
        .iter()
        .map(|r| CustomOpTensor { raw: r })
        .collect();

    // A panic crossing an `extern "C"` boundary is UB in Rust 2021 and an
    // abort in 2024. Catch here, convert to an error code, keep the NPU
    // runtime alive. `AssertUnwindSafe` is justified: `handler` is `Send +
    // Sync + 'static`, and `ctx`/wrapped tensors are stack-local — no
    // observable interior state leaks if the handler panics mid-call.
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| match kind {
        CbKind::Init => handler.init(&mut ctx, &ins_wrapped, &outs_wrapped),
        CbKind::Prepare => handler.prepare(&mut ctx, &ins_wrapped, &outs_wrapped),
        CbKind::Compute => handler.compute(&mut ctx, &ins_wrapped, &outs_wrapped),
    }));

    match result {
        Ok(Ok(())) => RKNN_SUCC,
        Ok(Err(e)) => {
            eprintln!(
                "[yscv-rknn] custom op slot {slot} {kind:?} failed: {e} ({})",
                rknn_error_name(RKNN_ERR_FAIL)
            );
            RKNN_ERR_FAIL
        }
        Err(payload) => {
            let msg = panic_message(&payload);
            eprintln!("[yscv-rknn] custom op slot {slot} {kind:?} PANICKED: {msg}");
            RKNN_ERR_FAIL
        }
    }
}

/// Extract a human-readable message from a panic payload, if possible.
fn panic_message(payload: &Box<dyn std::any::Any + Send>) -> &str {
    if let Some(s) = payload.downcast_ref::<&'static str>() {
        s
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.as_str()
    } else {
        "<non-string panic payload>"
    }
}

unsafe fn dispatch_destroy(slot: usize, op_ctx: *mut RknnCustomOpContextRaw) -> i32 {
    if op_ctx.is_null() {
        return RKNN_ERR_CTX_INVALID;
    }
    let guard = SLOTS[slot].read().unwrap();
    let handler = match guard.handler.as_ref() {
        Some(h) => h.clone(),
        None => return RKNN_SUCC,
    };
    let funcs = match guard.funcs.as_ref() {
        Some(f) => f.clone(),
        None => return RKNN_SUCC,
    };
    drop(guard);

    // SAFETY: same contract as dispatch().
    let mut ctx = CustomOpContext {
        raw: unsafe { &mut *op_ctx },
        funcs,
    };
    // Catch panics — destroy() often does cleanup work, but we still can't
    // let a panic cross the extern "C" boundary.
    let result =
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| handler.destroy(&mut ctx)));
    match result {
        Ok(Ok(())) => RKNN_SUCC,
        Ok(Err(_)) => RKNN_ERR_FAIL,
        Err(payload) => {
            let msg = panic_message(&payload);
            eprintln!("[yscv-rknn] custom op slot {slot} destroy PANICKED: {msg}");
            RKNN_ERR_FAIL
        }
    }
}

// CbKind needs Debug for the eprintln above.
impl std::fmt::Debug for CbKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            CbKind::Init => "init",
            CbKind::Prepare => "prepare",
            CbKind::Compute => "compute",
        })
    }
}

// ── 16 × 4 trampolines, generated by macro_rules ───────────────────

macro_rules! gen_slot {
    ($slot:expr, $init:ident, $prep:ident, $comp:ident, $dest:ident) => {
        unsafe extern "C" fn $init(
            op_ctx: *mut RknnCustomOpContextRaw,
            ins: *mut RknnCustomOpTensorRaw,
            n_in: u32,
            outs: *mut RknnCustomOpTensorRaw,
            n_out: u32,
        ) -> i32 {
            unsafe { dispatch($slot, CbKind::Init, op_ctx, ins, n_in, outs, n_out) }
        }
        unsafe extern "C" fn $prep(
            op_ctx: *mut RknnCustomOpContextRaw,
            ins: *mut RknnCustomOpTensorRaw,
            n_in: u32,
            outs: *mut RknnCustomOpTensorRaw,
            n_out: u32,
        ) -> i32 {
            unsafe { dispatch($slot, CbKind::Prepare, op_ctx, ins, n_in, outs, n_out) }
        }
        unsafe extern "C" fn $comp(
            op_ctx: *mut RknnCustomOpContextRaw,
            ins: *mut RknnCustomOpTensorRaw,
            n_in: u32,
            outs: *mut RknnCustomOpTensorRaw,
            n_out: u32,
        ) -> i32 {
            unsafe { dispatch($slot, CbKind::Compute, op_ctx, ins, n_in, outs, n_out) }
        }
        unsafe extern "C" fn $dest(op_ctx: *mut RknnCustomOpContextRaw) -> i32 {
            unsafe { dispatch_destroy($slot, op_ctx) }
        }
    };
}

gen_slot!(0, init_0, prepare_0, compute_0, destroy_0);
gen_slot!(1, init_1, prepare_1, compute_1, destroy_1);
gen_slot!(2, init_2, prepare_2, compute_2, destroy_2);
gen_slot!(3, init_3, prepare_3, compute_3, destroy_3);
gen_slot!(4, init_4, prepare_4, compute_4, destroy_4);
gen_slot!(5, init_5, prepare_5, compute_5, destroy_5);
gen_slot!(6, init_6, prepare_6, compute_6, destroy_6);
gen_slot!(7, init_7, prepare_7, compute_7, destroy_7);
gen_slot!(8, init_8, prepare_8, compute_8, destroy_8);
gen_slot!(9, init_9, prepare_9, compute_9, destroy_9);
gen_slot!(10, init_10, prepare_10, compute_10, destroy_10);
gen_slot!(11, init_11, prepare_11, compute_11, destroy_11);
gen_slot!(12, init_12, prepare_12, compute_12, destroy_12);
gen_slot!(13, init_13, prepare_13, compute_13, destroy_13);
gen_slot!(14, init_14, prepare_14, compute_14, destroy_14);
gen_slot!(15, init_15, prepare_15, compute_15, destroy_15);

pub(crate) fn trampolines_for_slot(slot: usize) -> TrampolineSet {
    match slot {
        0 => TrampolineSet {
            init: init_0,
            prepare: prepare_0,
            compute: compute_0,
            destroy: destroy_0,
        },
        1 => TrampolineSet {
            init: init_1,
            prepare: prepare_1,
            compute: compute_1,
            destroy: destroy_1,
        },
        2 => TrampolineSet {
            init: init_2,
            prepare: prepare_2,
            compute: compute_2,
            destroy: destroy_2,
        },
        3 => TrampolineSet {
            init: init_3,
            prepare: prepare_3,
            compute: compute_3,
            destroy: destroy_3,
        },
        4 => TrampolineSet {
            init: init_4,
            prepare: prepare_4,
            compute: compute_4,
            destroy: destroy_4,
        },
        5 => TrampolineSet {
            init: init_5,
            prepare: prepare_5,
            compute: compute_5,
            destroy: destroy_5,
        },
        6 => TrampolineSet {
            init: init_6,
            prepare: prepare_6,
            compute: compute_6,
            destroy: destroy_6,
        },
        7 => TrampolineSet {
            init: init_7,
            prepare: prepare_7,
            compute: compute_7,
            destroy: destroy_7,
        },
        8 => TrampolineSet {
            init: init_8,
            prepare: prepare_8,
            compute: compute_8,
            destroy: destroy_8,
        },
        9 => TrampolineSet {
            init: init_9,
            prepare: prepare_9,
            compute: compute_9,
            destroy: destroy_9,
        },
        10 => TrampolineSet {
            init: init_10,
            prepare: prepare_10,
            compute: compute_10,
            destroy: destroy_10,
        },
        11 => TrampolineSet {
            init: init_11,
            prepare: prepare_11,
            compute: compute_11,
            destroy: destroy_11,
        },
        12 => TrampolineSet {
            init: init_12,
            prepare: prepare_12,
            compute: compute_12,
            destroy: destroy_12,
        },
        13 => TrampolineSet {
            init: init_13,
            prepare: prepare_13,
            compute: compute_13,
            destroy: destroy_13,
        },
        14 => TrampolineSet {
            init: init_14,
            prepare: prepare_14,
            compute: compute_14,
            destroy: destroy_14,
        },
        15 => TrampolineSet {
            init: init_15,
            prepare: prepare_15,
            compute: compute_15,
            destroy: destroy_15,
        },
        _ => unreachable!("slot out of range — allocate_slot guards this"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;
    use std::sync::atomic::AtomicBool;

    /// Serialise tests that mutate the global SLOTS table. `cargo test`
    /// runs tests in parallel by default, and our pool is shared process-wide.
    static TEST_LOCK: Mutex<()> = Mutex::new(());

    struct CountingHandler {
        compute_called: AtomicBool,
    }
    impl CustomOpHandler for CountingHandler {
        fn compute(
            &self,
            _ctx: &mut CustomOpContext<'_>,
            _ins: &[CustomOpTensor<'_>],
            _outs: &[CustomOpTensor<'_>],
        ) -> Result<(), KernelError> {
            self.compute_called.store(true, Ordering::SeqCst);
            Ok(())
        }
    }

    /// Allocate 16 slots, verify the 17th fails, free one, verify reuse.
    /// Build an `Arc<RknnFunctions>` for tests that exercise slot allocation
    /// without actually calling any FFI. We allocate enough zeroed bytes to
    /// satisfy the layout, then transmute — the resulting Arc must never be
    /// dereferenced via the function-pointer fields (those are null/UB).
    fn stub_funcs() -> Arc<RknnFunctions> {
        let layout = std::alloc::Layout::new::<RknnFunctions>();
        // SAFETY: writing zeros into a freshly allocated buffer of the right
        // layout, then producing an Arc via Box. The required-fn-ptr fields
        // are now null pointers — calling them is UB, so the test must not
        // do that. We only use the Arc as opaque storage in the slot table.
        unsafe {
            let raw = std::alloc::alloc_zeroed(layout) as *mut RknnFunctions;
            assert!(!raw.is_null(), "alloc failed");
            Arc::from(Box::from_raw(raw))
        }
    }

    #[test]
    fn slot_pool_capacity_and_reuse() {
        let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let stub = stub_funcs();
        let h: Arc<dyn CustomOpHandler> = Arc::new(CountingHandler {
            compute_called: AtomicBool::new(false),
        });

        let mut acquired = Vec::new();
        for _ in 0..MAX_CUSTOM_OP_SLOTS {
            acquired.push(allocate_slot(h.clone(), stub.clone()).expect("slot"));
        }
        // Pool full now.
        assert!(allocate_slot(h.clone(), stub.clone()).is_err());

        // Free middle slot, verify reuse picks a free index.
        release_slot(acquired[5]);
        let reused = allocate_slot(h.clone(), stub.clone()).expect("post-release");
        assert!(reused < MAX_CUSTOM_OP_SLOTS);

        for s in acquired {
            release_slot(s);
        }
        release_slot(reused);
    }

    #[test]
    fn trampolines_for_slot_returns_distinct_pointers() {
        let t0 = trampolines_for_slot(0);
        let t1 = trampolines_for_slot(1);
        // Function pointers must differ — each slot has its own compiled trampoline.
        assert_ne!(t0.compute as usize, t1.compute as usize);
        assert_ne!(t0.init as usize, t1.init as usize);
    }

    #[test]
    fn cb_kind_debug() {
        assert_eq!(format!("{:?}", CbKind::Init), "init");
        assert_eq!(format!("{:?}", CbKind::Compute), "compute");
    }

    /// A handler that panics on compute — trampoline must catch it and
    /// return RKNN_ERR_FAIL rather than abort the process.
    struct PanickingHandler;
    impl CustomOpHandler for PanickingHandler {
        fn compute(
            &self,
            _ctx: &mut CustomOpContext<'_>,
            _ins: &[CustomOpTensor<'_>],
            _outs: &[CustomOpTensor<'_>],
        ) -> Result<(), KernelError> {
            panic!("handler intentionally panicked");
        }
    }

    #[test]
    fn dispatch_catches_handler_panic() {
        let _guard = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let stub = stub_funcs();
        let h: Arc<dyn CustomOpHandler> = Arc::new(PanickingHandler);
        let slot = allocate_slot(h, stub).expect("slot");

        // Build a minimal RknnCustomOpContextRaw on the stack.
        let mut op_ctx = RknnCustomOpContextRaw {
            target: 1,
            internal_ctx: 0,
            gpu_ctx: super::super::ffi::RknnGpuOpContext {
                cl_context: std::ptr::null_mut(),
                cl_command_queue: std::ptr::null_mut(),
                cl_kernel: std::ptr::null_mut(),
            },
            priv_data: std::ptr::null_mut(),
        };

        // SAFETY: op_ctx lives on this stack frame for the call duration;
        // ins/outs null is handled by dispatch(); no SDK callback is
        // dereferenced because we go through the trampoline directly.
        let ret = unsafe {
            dispatch(
                slot,
                CbKind::Compute,
                &mut op_ctx as *mut _,
                std::ptr::null_mut(),
                0,
                std::ptr::null_mut(),
                0,
            )
        };
        assert_eq!(ret, RKNN_ERR_FAIL, "panic must be caught, not unwound");

        release_slot(slot);
    }
}
