use super::consts::{
    RKNN_MAX_DIMS, RKNN_MAX_DYNAMIC_SHAPE_NUM, RKNN_MAX_NAME_LEN, RKNN_SUCC, RknnQuantType,
    RknnTensorFormat, RknnTensorType,
};
use crate::KernelError;
use std::ffi::c_void;

// ══════════════════════════════════════════════════════════════════════
// FFI struct definitions — match C ABI exactly
// ══════════════════════════════════════════════════════════════════════

/// Opaque context handle for both inference (`rknn_context`) and matmul
/// (`rknn_matmul_ctx`) APIs. Both are `uint64_t` in the SDK.
pub(crate) type RknnContext = u64;
pub(crate) type RknnMatmulCtx = u64;

/// `rknn_input` — per-input tensor descriptor for `rknn_inputs_set`.
#[repr(C)]
pub(crate) struct RknnInput {
    pub(crate) index: u32,
    pub(crate) buf: *const u8,
    pub(crate) size: u32,
    pub(crate) pass_through: u8,
    pub(crate) typ: u32,
    pub(crate) fmt: u32,
}

/// `rknn_output` — per-output tensor descriptor for `rknn_outputs_get`.
#[repr(C)]
pub(crate) struct RknnOutput {
    pub(crate) want_float: u8,
    pub(crate) is_prealloc: u8,
    pub(crate) index: u32,
    pub(crate) buf: *mut u8,
    pub(crate) size: u32,
}

/// `rknn_input_output_num` — input/output count returned by `RKNN_QUERY_IN_OUT_NUM`.
#[repr(C)]
#[derive(Default)]
pub(crate) struct RknnInOutNum {
    pub(crate) n_input: u32,
    pub(crate) n_output: u32,
}

/// `rknn_tensor_attr` — full tensor descriptor returned by attribute queries.
///
/// Fields from SDK 2.4.3a0 (`rknn_api.h` lines 280-310). This struct is
/// the widest FFI surface and must match the C ABI exactly.
#[repr(C)]
#[derive(Clone)]
pub struct RknnTensorAttr {
    /// Tensor index.
    pub index: u32,
    /// Number of dimensions (≤ `RKNN_MAX_DIMS = 16`).
    pub n_dims: u32,
    /// Dimension sizes.
    pub dims: [u32; RKNN_MAX_DIMS],
    /// Tensor name (null-terminated C string).
    pub name: [u8; RKNN_MAX_NAME_LEN],
    /// Total number of elements.
    pub n_elems: u32,
    /// Total byte size of the tensor.
    pub size: u32,
    /// Tensor format: see `RknnTensorFormat`.
    pub fmt: u32,
    /// Data type: see `RknnTensorType`.
    pub typ: u32,
    /// Quantization type: see `RknnQuantType`.
    pub qnt_type: u32,
    /// DFP fractional length.
    pub fl: i8,
    /// AFFINE zero point.
    pub zp: i32,
    /// AFFINE scale factor.
    pub scale: f32,
    /// Per-row stride in bytes (width dimension).
    pub w_stride: u32,
    /// Total size including stride padding.
    pub size_with_stride: u32,
    /// Pass-through flag: input data copied verbatim to NPU buffer.
    pub pass_through: u8,
    /// Per-column stride in bytes (height dimension).
    pub h_stride: u32,
}

impl Default for RknnTensorAttr {
    fn default() -> Self {
        Self {
            index: 0,
            n_dims: 0,
            dims: [0; RKNN_MAX_DIMS],
            name: [0; RKNN_MAX_NAME_LEN],
            n_elems: 0,
            size: 0,
            fmt: 0,
            typ: 0,
            qnt_type: 0,
            fl: 0,
            zp: 0,
            scale: 0.0,
            w_stride: 0,
            size_with_stride: 0,
            pass_through: 0,
            h_stride: 0,
        }
    }
}

impl RknnTensorAttr {
    /// Tensor name as a UTF-8 string (stops at first null byte).
    pub fn name_str(&self) -> &str {
        let end = self
            .name
            .iter()
            .position(|&b| b == 0)
            .unwrap_or(self.name.len());
        std::str::from_utf8(&self.name[..end]).unwrap_or("<invalid-utf8>")
    }

    /// Shape as a `Vec<usize>` using only the first `n_dims` dimensions.
    pub fn shape(&self) -> Vec<usize> {
        self.dims[..self.n_dims as usize]
            .iter()
            .map(|&d| d as usize)
            .collect()
    }

    /// Decoded data type.
    pub fn data_type(&self) -> RknnTensorType {
        RknnTensorType::from_u32(self.typ)
    }

    /// Decoded tensor format.
    pub fn format(&self) -> RknnTensorFormat {
        RknnTensorFormat::from_u32(self.fmt)
    }

    /// Decoded quantization type.
    pub fn quant_type(&self) -> RknnQuantType {
        RknnQuantType::from_u32(self.qnt_type)
    }
}

/// `rknn_input_range` — dynamic input shape descriptor (query 13).
#[repr(C)]
pub struct RknnInputRange {
    pub index: u32,
    pub shape_number: u32,
    pub fmt: u32,
    pub name: [u8; RKNN_MAX_NAME_LEN],
    pub dyn_range: [[u32; RKNN_MAX_DIMS]; RKNN_MAX_DYNAMIC_SHAPE_NUM],
    pub n_dims: u32,
}

/// `rknn_perf_detail` — per-op performance output (query 3).
#[repr(C)]
pub(crate) struct RknnPerfDetailRaw {
    pub(crate) perf_data: *mut u8,
    pub(crate) data_len: u64,
}

/// `rknn_perf_run` — total inference duration (query 4).
#[repr(C)]
#[derive(Default, Clone, Copy)]
pub(crate) struct RknnPerfRun {
    pub(crate) run_duration: i64,
}

/// `rknn_sdk_version` — driver/API version strings (query 5).
#[repr(C)]
pub(crate) struct RknnSdkVersionRaw {
    pub(crate) api_version: [u8; 256],
    pub(crate) drv_version: [u8; 256],
}

impl Default for RknnSdkVersionRaw {
    fn default() -> Self {
        Self {
            api_version: [0; 256],
            drv_version: [0; 256],
        }
    }
}

/// `rknn_mem_size` — memory footprint (query 6).
#[repr(C)]
#[derive(Default, Clone, Copy)]
pub(crate) struct RknnMemSizeRaw {
    pub(crate) total_weight_size: u32,
    pub(crate) total_internal_size: u32,
    pub(crate) total_dma_allocated_size: u64,
    pub(crate) total_sram_size: u32,
    pub(crate) free_sram_size: u32,
    pub(crate) reserved: [u32; 10],
}

/// `rknn_custom_string` — model-embedded metadata (query 7).
#[repr(C)]
pub(crate) struct RknnCustomStringRaw {
    pub(crate) string: [u8; 1024],
}

impl Default for RknnCustomStringRaw {
    fn default() -> Self {
        Self { string: [0; 1024] }
    }
}

/// `rknn_init_extend` — optional extended init parameters.
#[repr(C)]
pub(crate) struct RknnInitExtend {
    pub(crate) ctx: RknnContext,
    pub(crate) real_model_offset: i32,
    pub(crate) real_model_size: u32,
    pub(crate) model_buffer_fd: i32,
    pub(crate) model_buffer_flags: u32,
    pub(crate) reserved: [u8; 112],
}

impl Default for RknnInitExtend {
    fn default() -> Self {
        Self {
            ctx: 0,
            real_model_offset: 0,
            real_model_size: 0,
            model_buffer_fd: -1,
            model_buffer_flags: 0,
            reserved: [0; 112],
        }
    }
}

/// `rknn_run_extend` — async/fence parameters for `rknn_run`.
#[repr(C)]
#[derive(Default, Clone, Copy)]
pub(crate) struct RknnRunExtend {
    pub(crate) frame_id: u64,
    pub(crate) non_block: i32,
    pub(crate) timeout_ms: i32,
    pub(crate) fence_fd: i32,
}

/// `rknn_output_extend` — frame ID matching for async outputs.
#[repr(C)]
#[derive(Default, Clone, Copy)]
pub(crate) struct RknnOutputExtend {
    pub(crate) frame_id: u64,
}

/// `rknn_tensor_mem` — zero-copy memory handle.
#[repr(C)]
pub(crate) struct RknnTensorMemRaw {
    pub(crate) virt_addr: *mut c_void,
    pub(crate) phys_addr: u64,
    pub(crate) fd: i32,
    pub(crate) offset: i32,
    pub(crate) size: u32,
    pub(crate) flags: u32,
    pub(crate) priv_data: *mut c_void,
}

// Matmul structs ────────────────────────────────────────────────────────

/// `rknn_quant_params` for matmul — per-channel/per-group quantization.
#[repr(C)]
pub(crate) struct RknnQuantParamsRaw {
    pub(crate) name: [u8; RKNN_MAX_NAME_LEN],
    pub(crate) scale: *const f32,
    pub(crate) scale_len: i32,
    pub(crate) zp: *const i32,
    pub(crate) zp_len: i32,
}

/// `rknn_matmul_tensor_attr` — A/B/C tensor descriptor.
#[repr(C)]
#[derive(Clone)]
pub struct RknnMatmulTensorAttr {
    pub name: [u8; RKNN_MAX_NAME_LEN],
    pub n_dims: u32,
    pub dims: [u32; RKNN_MAX_DIMS],
    pub size: u32,
    pub typ: u32,
}

impl Default for RknnMatmulTensorAttr {
    fn default() -> Self {
        Self {
            name: [0; RKNN_MAX_NAME_LEN],
            n_dims: 0,
            dims: [0; RKNN_MAX_DIMS],
            size: 0,
            typ: 0,
        }
    }
}

/// `rknn_matmul_io_attr` — triple of A/B/C descriptors.
#[repr(C)]
#[derive(Default, Clone)]
pub struct RknnMatmulIoAttr {
    pub a: RknnMatmulTensorAttr,
    pub b: RknnMatmulTensorAttr,
    pub c: RknnMatmulTensorAttr,
}

/// `rknn_matmul_shape` — M, K, N dimensions.
#[repr(C)]
#[derive(Default, Clone, Copy)]
pub struct RknnMatmulShape {
    pub m: i32,
    pub k: i32,
    pub n: i32,
}

/// `rknn_matmul_info` — full matmul configuration.
#[repr(C)]
pub struct RknnMatmulInfo {
    pub m: i32,
    pub k: i32,
    pub n: i32,
    pub typ: i32, // RknnMatmulType as i32
    pub b_layout: i16,
    pub b_quant_type: i16,
    pub ac_layout: i16,
    pub ac_quant_type: i16,
    pub iommu_domain_id: i32,
    pub group_size: i16,
    pub(crate) reserved: [i8; 34],
}

impl Default for RknnMatmulInfo {
    fn default() -> Self {
        Self {
            m: 0,
            k: 0,
            n: 0,
            typ: 0,
            b_layout: 0,
            b_quant_type: 0,
            ac_layout: 0,
            ac_quant_type: 0,
            iommu_domain_id: 0,
            group_size: 0,
            reserved: [0; 34],
        }
    }
}

// Custom-op FFI structs ─────────────────────────────────────────────────

/// `rknn_gpu_op_context` — OpenCL context handles passed to GPU custom ops.
#[repr(C)]
pub(crate) struct RknnGpuOpContext {
    pub(crate) cl_context: *mut c_void,
    pub(crate) cl_command_queue: *mut c_void,
    pub(crate) cl_kernel: *mut c_void,
}

/// `rknn_custom_op_context` — runtime state handed to custom-op callbacks.
#[repr(C)]
pub(crate) struct RknnCustomOpContextRaw {
    pub(crate) target: u32,
    pub(crate) internal_ctx: u64,
    pub(crate) gpu_ctx: RknnGpuOpContext,
    pub(crate) priv_data: *mut c_void,
}

/// `rknn_custom_op_tensor` — tensor view passed to custom-op callbacks.
#[repr(C)]
pub(crate) struct RknnCustomOpTensorRaw {
    pub(crate) attr: RknnTensorAttr,
    pub(crate) mem: RknnTensorMemRaw,
}

/// `rknn_custom_op_attr` — single ONNX-attribute query result.
#[repr(C)]
pub(crate) struct RknnCustomOpAttrRaw {
    pub(crate) name: [u8; RKNN_MAX_NAME_LEN],
    pub(crate) dtype: u32,
    pub(crate) n_elems: u32,
    pub(crate) data: *mut c_void,
}

/// `rknn_custom_op` — per-op descriptor passed to `rknn_register_custom_ops`.
#[repr(C)]
pub(crate) struct RknnCustomOpRaw {
    pub(crate) version: u32,
    pub(crate) target: u32,
    pub(crate) op_type: [u8; RKNN_MAX_NAME_LEN],
    pub(crate) cl_kernel_name: [u8; RKNN_MAX_NAME_LEN],
    pub(crate) cl_kernel_source: *mut u8,
    pub(crate) cl_source_size: u64,
    pub(crate) cl_build_options: [u8; RKNN_MAX_NAME_LEN],
    /// `init` callback (optional, may be null).
    pub(crate) init: *const c_void,
    /// `prepare` callback (optional).
    pub(crate) prepare: *const c_void,
    /// `compute` callback (required — null only if pure OpenCL kernel).
    pub(crate) compute: *const c_void,
    /// `compute_native` callback (optional, currently unused by SDK).
    pub(crate) compute_native: *const c_void,
    /// `destroy` callback (optional).
    pub(crate) destroy: *const c_void,
}

// ══════════════════════════════════════════════════════════════════════
// FFI function pointer types
// ══════════════════════════════════════════════════════════════════════

// Core runtime
pub(crate) type FnRknnInit =
    unsafe extern "C" fn(*mut RknnContext, *const u8, u32, u32, *mut RknnInitExtend) -> i32;
pub(crate) type FnRknnDupContext = unsafe extern "C" fn(*mut RknnContext, *mut RknnContext) -> i32;
pub(crate) type FnRknnDestroy = unsafe extern "C" fn(RknnContext) -> i32;
pub(crate) type FnRknnQuery = unsafe extern "C" fn(RknnContext, u32, *mut u8, u32) -> i32;
pub(crate) type FnRknnInputsSet = unsafe extern "C" fn(RknnContext, u32, *const RknnInput) -> i32;
pub(crate) type FnRknnRun = unsafe extern "C" fn(RknnContext, *mut RknnRunExtend) -> i32;
pub(crate) type FnRknnWait = unsafe extern "C" fn(RknnContext, *mut RknnRunExtend) -> i32;
pub(crate) type FnRknnOutputsGet =
    unsafe extern "C" fn(RknnContext, u32, *mut RknnOutput, *mut RknnOutputExtend) -> i32;
pub(crate) type FnRknnOutputsRelease =
    unsafe extern "C" fn(RknnContext, u32, *mut RknnOutput) -> i32;
pub(crate) type FnRknnSetCoreMask = unsafe extern "C" fn(RknnContext, u32) -> i32;
pub(crate) type FnRknnSetBatchCoreNum = unsafe extern "C" fn(RknnContext, i32) -> i32;
pub(crate) type FnRknnSetInputShape = unsafe extern "C" fn(RknnContext, *mut RknnTensorAttr) -> i32;
pub(crate) type FnRknnSetInputShapes =
    unsafe extern "C" fn(RknnContext, u32, *const RknnTensorAttr) -> i32;

// Memory management
pub(crate) type FnRknnCreateMem = unsafe extern "C" fn(RknnContext, u32) -> *mut RknnTensorMemRaw;
pub(crate) type FnRknnCreateMem2 =
    unsafe extern "C" fn(RknnContext, u64, u64) -> *mut RknnTensorMemRaw;
pub(crate) type FnRknnCreateMemFromFd =
    unsafe extern "C" fn(RknnContext, i32, *mut c_void, u32, i32) -> *mut RknnTensorMemRaw;
pub(crate) type FnRknnCreateMemFromPhys =
    unsafe extern "C" fn(RknnContext, u64, *mut c_void, u32) -> *mut RknnTensorMemRaw;
pub(crate) type FnRknnDestroyMem = unsafe extern "C" fn(RknnContext, *mut RknnTensorMemRaw) -> i32;
pub(crate) type FnRknnSetWeightMem =
    unsafe extern "C" fn(RknnContext, *mut RknnTensorMemRaw) -> i32;
pub(crate) type FnRknnSetInternalMem =
    unsafe extern "C" fn(RknnContext, *mut RknnTensorMemRaw) -> i32;
pub(crate) type FnRknnSetIoMem =
    unsafe extern "C" fn(RknnContext, *mut RknnTensorMemRaw, *mut RknnTensorAttr) -> i32;
pub(crate) type FnRknnMemSync =
    unsafe extern "C" fn(RknnContext, *mut RknnTensorMemRaw, u32) -> i32;

// Matmul API
pub(crate) type FnRknnMatmulCreate =
    unsafe extern "C" fn(*mut RknnMatmulCtx, *mut RknnMatmulInfo, *mut RknnMatmulIoAttr) -> i32;
pub(crate) type FnRknnMatmulSetIoMem =
    unsafe extern "C" fn(RknnMatmulCtx, *mut RknnTensorMemRaw, *mut RknnMatmulTensorAttr) -> i32;
pub(crate) type FnRknnMatmulSetCoreMask = unsafe extern "C" fn(RknnMatmulCtx, u32) -> i32;
pub(crate) type FnRknnMatmulSetQuantParams =
    unsafe extern "C" fn(RknnMatmulCtx, *mut RknnQuantParamsRaw) -> i32;
pub(crate) type FnRknnMatmulSetDynamicShape =
    unsafe extern "C" fn(RknnMatmulCtx, *mut RknnMatmulShape) -> i32;
pub(crate) type FnRknnMatmulRun = unsafe extern "C" fn(RknnMatmulCtx) -> i32;
pub(crate) type FnRknnMatmulDestroy = unsafe extern "C" fn(RknnMatmulCtx) -> i32;
pub(crate) type FnRknnBLayoutTransform =
    unsafe extern "C" fn(*mut c_void, *mut c_void, i32, i32, *mut RknnMatmulInfo) -> i32;

// MPP zero-copy
pub(crate) type FnRknnCreateMemFromMbBlk =
    unsafe extern "C" fn(RknnContext, *mut c_void, i32) -> *mut RknnTensorMemRaw;

// Dynamic-shape matmul
pub(crate) type FnRknnMatmulCreateDynamicShape = unsafe extern "C" fn(
    *mut RknnMatmulCtx,
    *mut RknnMatmulInfo,
    i32,                  // num_shapes
    *mut RknnMatmulShape, // allowed shapes
    *mut RknnMatmulIoAttr,
) -> i32;
pub(crate) type FnRknnMatmulGetQuantParams =
    unsafe extern "C" fn(RknnMatmulCtx, *mut RknnQuantParamsRaw, *mut f32) -> i32;

// Custom-op registration + attribute query
pub(crate) type FnRknnRegisterCustomOps =
    unsafe extern "C" fn(RknnContext, *mut RknnCustomOpRaw, u32) -> i32;
pub(crate) type FnRknnCustomOpGetOpAttr = unsafe extern "C" fn(
    *mut RknnCustomOpContextRaw,
    *const u8, // attr_name (null-terminated)
    *mut RknnCustomOpAttrRaw,
);

// Custom-op callback signatures (used as trampoline targets, not resolved via dlsym).
pub(crate) type FnRknnCustomOpInit = unsafe extern "C" fn(
    *mut RknnCustomOpContextRaw,
    *mut RknnCustomOpTensorRaw,
    u32,
    *mut RknnCustomOpTensorRaw,
    u32,
) -> i32;
pub(crate) type FnRknnCustomOpPrepare = FnRknnCustomOpInit;
pub(crate) type FnRknnCustomOpCompute = FnRknnCustomOpInit;
pub(crate) type FnRknnCustomOpDestroy = unsafe extern "C" fn(*mut RknnCustomOpContextRaw) -> i32;

// ══════════════════════════════════════════════════════════════════════
// Dynamic library loading
// ══════════════════════════════════════════════════════════════════════

/// Resolved function pointers from `librknnrt.so`.
///
/// Optional fields are symbols that may not exist on older runtimes —
/// absence does not fail the library load; features that depend on
/// missing symbols return `Err` at call time.
pub(crate) struct RknnFunctions {
    // Always present (core API)
    pub(crate) init: FnRknnInit,
    pub(crate) destroy: FnRknnDestroy,
    pub(crate) query: FnRknnQuery,
    pub(crate) inputs_set: FnRknnInputsSet,
    pub(crate) run: FnRknnRun,
    pub(crate) outputs_get: FnRknnOutputsGet,
    pub(crate) outputs_release: FnRknnOutputsRelease,

    // Multi-core / async (SDK ≥ 1.6)
    pub(crate) dup_context: Option<FnRknnDupContext>,
    pub(crate) wait: Option<FnRknnWait>,
    pub(crate) set_core_mask: Option<FnRknnSetCoreMask>,
    pub(crate) set_batch_core_num: Option<FnRknnSetBatchCoreNum>,
    pub(crate) set_input_shape: Option<FnRknnSetInputShape>,
    pub(crate) set_input_shapes: Option<FnRknnSetInputShapes>,

    // Zero-copy memory (SDK ≥ 1.4)
    pub(crate) create_mem: Option<FnRknnCreateMem>,
    pub(crate) create_mem2: Option<FnRknnCreateMem2>,
    pub(crate) create_mem_from_fd: Option<FnRknnCreateMemFromFd>,
    pub(crate) create_mem_from_phys: Option<FnRknnCreateMemFromPhys>,
    pub(crate) create_mem_from_mb_blk: Option<FnRknnCreateMemFromMbBlk>,
    pub(crate) destroy_mem: Option<FnRknnDestroyMem>,
    pub(crate) set_weight_mem: Option<FnRknnSetWeightMem>,
    pub(crate) set_internal_mem: Option<FnRknnSetInternalMem>,
    pub(crate) set_io_mem: Option<FnRknnSetIoMem>,
    pub(crate) mem_sync: Option<FnRknnMemSync>,

    // Matmul (SDK ≥ 1.5)
    pub(crate) matmul_create: Option<FnRknnMatmulCreate>,
    pub(crate) matmul_create_dynamic_shape: Option<FnRknnMatmulCreateDynamicShape>,
    pub(crate) matmul_set_io_mem: Option<FnRknnMatmulSetIoMem>,
    pub(crate) matmul_set_core_mask: Option<FnRknnMatmulSetCoreMask>,
    pub(crate) matmul_set_quant_params: Option<FnRknnMatmulSetQuantParams>,
    pub(crate) matmul_get_quant_params: Option<FnRknnMatmulGetQuantParams>,
    pub(crate) matmul_set_dynamic_shape: Option<FnRknnMatmulSetDynamicShape>,
    pub(crate) matmul_run: Option<FnRknnMatmulRun>,
    pub(crate) matmul_destroy: Option<FnRknnMatmulDestroy>,
    pub(crate) matmul_b_layout_transform: Option<FnRknnBLayoutTransform>,

    // Custom ops (SDK ≥ 2.0)
    pub(crate) register_custom_ops: Option<FnRknnRegisterCustomOps>,
    pub(crate) custom_op_get_op_attr: Option<FnRknnCustomOpGetOpAttr>,
}

/// Raw handle to a dynamically loaded library.
pub(crate) struct DlHandle {
    pub(crate) handle: *mut c_void,
}

// SAFETY: The library handle is only used for dlsym/dlclose which are
// thread-safe for read-only symbol resolution after dlopen completes.
unsafe impl Send for DlHandle {}
// SAFETY: All function pointers obtained from the handle are pure FFI
// calls into the RKNN runtime, which is documented as thread-safe for
// independent contexts.
unsafe impl Sync for DlHandle {}

impl Drop for DlHandle {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            // SAFETY: handle was obtained from a successful dlopen call.
            unsafe {
                libc::dlclose(self.handle);
            }
        }
    }
}

/// Resolve a required symbol. Returns `Err` if the symbol is missing.
///
/// # Safety
/// Caller must ensure `handle` is a valid dl handle and the type `T`
/// matches the symbol's C function signature.
unsafe fn resolve_required<T: Copy>(handle: *mut c_void, name: &[u8]) -> Result<T, KernelError> {
    // SAFETY: handle is valid; name is a null-terminated byte string.
    let sym = unsafe { libc::dlsym(handle, name.as_ptr().cast()) };
    if sym.is_null() {
        let sym_name =
            std::str::from_utf8(&name[..name.len().saturating_sub(1)]).unwrap_or("<invalid>");
        return Err(KernelError::Rknn {
            message: format!("required symbol `{sym_name}` not found in librknnrt.so"),
        });
    }
    // SAFETY: sym is non-null; caller guarantees `T` matches.
    Ok(unsafe { std::mem::transmute_copy::<*mut c_void, T>(&sym) })
}

/// Resolve an optional symbol. Returns `None` if missing (soft feature gate).
///
/// # Safety
/// Same as `resolve_required`.
pub(crate) unsafe fn resolve_optional<T: Copy>(handle: *mut c_void, name: &[u8]) -> Option<T> {
    // SAFETY: handle is valid; name is a null-terminated byte string.
    let sym = unsafe { libc::dlsym(handle, name.as_ptr().cast()) };
    if sym.is_null() {
        None
    } else {
        // SAFETY: sym is non-null; caller guarantees `T` matches.
        Some(unsafe { std::mem::transmute_copy::<*mut c_void, T>(&sym) })
    }
}

pub(crate) fn load_rknn_library() -> Result<(DlHandle, RknnFunctions), KernelError> {
    // SAFETY: dlopen with RTLD_LAZY; handle is checked non-null.
    let handle = unsafe { libc::dlopen(c"librknnrt.so".as_ptr(), libc::RTLD_LAZY) };
    if handle.is_null() {
        return Err(KernelError::Rknn {
            message: "failed to load librknnrt.so — RKNN runtime not available".into(),
        });
    }

    // SAFETY: handle is valid; each symbol name matches its declared function type.
    let funcs = unsafe {
        RknnFunctions {
            init: resolve_required(handle, b"rknn_init\0")?,
            destroy: resolve_required(handle, b"rknn_destroy\0")?,
            query: resolve_required(handle, b"rknn_query\0")?,
            inputs_set: resolve_required(handle, b"rknn_inputs_set\0")?,
            run: resolve_required(handle, b"rknn_run\0")?,
            outputs_get: resolve_required(handle, b"rknn_outputs_get\0")?,
            outputs_release: resolve_required(handle, b"rknn_outputs_release\0")?,

            dup_context: resolve_optional(handle, b"rknn_dup_context\0"),
            wait: resolve_optional(handle, b"rknn_wait\0"),
            set_core_mask: resolve_optional(handle, b"rknn_set_core_mask\0"),
            set_batch_core_num: resolve_optional(handle, b"rknn_set_batch_core_num\0"),
            set_input_shape: resolve_optional(handle, b"rknn_set_input_shape\0"),
            set_input_shapes: resolve_optional(handle, b"rknn_set_input_shapes\0"),

            create_mem: resolve_optional(handle, b"rknn_create_mem\0"),
            create_mem2: resolve_optional(handle, b"rknn_create_mem2\0"),
            create_mem_from_fd: resolve_optional(handle, b"rknn_create_mem_from_fd\0"),
            create_mem_from_phys: resolve_optional(handle, b"rknn_create_mem_from_phys\0"),
            create_mem_from_mb_blk: resolve_optional(handle, b"rknn_create_mem_from_mb_blk\0"),
            destroy_mem: resolve_optional(handle, b"rknn_destroy_mem\0"),
            set_weight_mem: resolve_optional(handle, b"rknn_set_weight_mem\0"),
            set_internal_mem: resolve_optional(handle, b"rknn_set_internal_mem\0"),
            set_io_mem: resolve_optional(handle, b"rknn_set_io_mem\0"),
            mem_sync: resolve_optional(handle, b"rknn_mem_sync\0"),

            matmul_create: resolve_optional(handle, b"rknn_matmul_create\0"),
            matmul_create_dynamic_shape: resolve_optional(
                handle,
                b"rknn_matmul_create_dynamic_shape\0",
            ),
            matmul_set_io_mem: resolve_optional(handle, b"rknn_matmul_set_io_mem\0"),
            matmul_set_core_mask: resolve_optional(handle, b"rknn_matmul_set_core_mask\0"),
            matmul_set_quant_params: resolve_optional(handle, b"rknn_matmul_set_quant_params\0"),
            matmul_get_quant_params: resolve_optional(handle, b"rknn_matmul_get_quant_params\0"),
            matmul_set_dynamic_shape: resolve_optional(handle, b"rknn_matmul_set_dynamic_shape\0"),
            matmul_run: resolve_optional(handle, b"rknn_matmul_run\0"),
            matmul_destroy: resolve_optional(handle, b"rknn_matmul_destroy\0"),
            matmul_b_layout_transform: resolve_optional(
                handle,
                b"rknn_B_normal_layout_to_native_layout\0",
            ),

            register_custom_ops: resolve_optional(handle, b"rknn_register_custom_ops\0"),
            custom_op_get_op_attr: resolve_optional(handle, b"rknn_custom_op_get_op_attr\0"),
        }
    };

    Ok((DlHandle { handle }, funcs))
}

/// Helper to call `rknn_query` and parse the error code.
#[allow(clippy::too_many_arguments)]
pub(crate) fn query_raw<T>(
    funcs: &RknnFunctions,
    ctx: RknnContext,
    cmd: u32,
    cmd_name: &str,
    info: *mut T,
) -> Result<(), KernelError> {
    // SAFETY: info points to a valid struct of type T; caller provides
    // matching size via `size_of::<T>()`. cmd is a valid query command.
    let ret = unsafe { (funcs.query)(ctx, cmd, info.cast(), std::mem::size_of::<T>() as u32) };
    if ret != RKNN_SUCC {
        return Err(KernelError::Rknn {
            message: format!(
                "rknn_query({cmd_name}) failed: {} ({ret})",
                rknn_error_name(ret)
            ),
        });
    }
    Ok(())
}

/// Convert an RKNN return code into a stable short label.
///
/// Used in error messages to make `KernelError::Rknn` self-explanatory.
/// Unknown codes (e.g. negative values from custom-op handlers) return
/// `"UNKNOWN"`.
pub(crate) fn rknn_error_name(code: i32) -> &'static str {
    use super::consts::*;
    match code {
        RKNN_SUCC => "SUCC",
        RKNN_ERR_FAIL => "FAIL",
        RKNN_ERR_TIMEOUT => "TIMEOUT",
        RKNN_ERR_DEVICE_UNAVAILABLE => "DEVICE_UNAVAILABLE",
        RKNN_ERR_MALLOC_FAIL => "MALLOC_FAIL",
        RKNN_ERR_PARAM_INVALID => "PARAM_INVALID",
        RKNN_ERR_MODEL_INVALID => "MODEL_INVALID",
        RKNN_ERR_CTX_INVALID => "CTX_INVALID",
        RKNN_ERR_INPUT_INVALID => "INPUT_INVALID",
        RKNN_ERR_OUTPUT_INVALID => "OUTPUT_INVALID",
        RKNN_ERR_DEVICE_UNMATCH => "DEVICE_UNMATCH",
        RKNN_ERR_INCOMPATIBLE_PRE_COMPILE_MODEL => "INCOMPATIBLE_PRE_COMPILE_MODEL",
        RKNN_ERR_INCOMPATIBLE_OPTIMIZATION_LEVEL_VERSION => {
            "INCOMPATIBLE_OPTIMIZATION_LEVEL_VERSION"
        }
        RKNN_ERR_TARGET_PLATFORM_UNMATCH => "TARGET_PLATFORM_UNMATCH",
        _ => "UNKNOWN",
    }
}

/// Returns `true` when a `KernelError::Rknn` error message indicates a
/// *recoverable* runtime fault — a stuck NPU, a dropped device handle,
/// or a transient timeout. Caller-bug errors (`PARAM_INVALID`,
/// `INPUT_INVALID`, `OUTPUT_INVALID`, `MODEL_INVALID`, etc.) return
/// `false` — re-initialising the context won't help if the input
/// contract is wrong.
///
/// Used by `ContextPool::dispatch_on` and
/// `RknnPipelinedPool::submit`/`wait` to decide whether to auto-reset
/// a slot after a dispatch failure.
pub(crate) fn is_recoverable_rknn_error(err_message: &str) -> bool {
    // Match on the symbolic name that `rknn_error_name` embeds into
    // every `KernelError::Rknn::message`. Tests pin this contract.
    err_message.contains("TIMEOUT")
        || err_message.contains("CTX_INVALID")
        || err_message.contains("DEVICE_UNAVAILABLE")
        || err_message.contains("DEVICE_UNMATCH")
}
