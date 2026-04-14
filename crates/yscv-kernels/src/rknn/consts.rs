// ═══════════════════════════════════════════════════════════════════════
// Phase 1 — Complete SDK constant coverage
// ═══════════════════════════════════════════════════════════════════════

// ── Error codes ────────────────────────────────────────────────────────
//
// All values from `rknn_api.h` SDK 2.4.3a0. `rknn_*` functions return one of
// these as `i32`. Use `crate::rknn::ffi::rknn_error_name(code)` for a
// human-readable label in error messages.

pub(crate) const RKNN_SUCC: i32 = 0;
/// Generic failure (catch-all when no other category fits).
pub const RKNN_ERR_FAIL: i32 = -1;
/// Inference timed out (exceeded `RknnRunExtend::timeout_ms` or driver default).
pub const RKNN_ERR_TIMEOUT: i32 = -2;
/// NPU device offline / no driver / busy beyond retry.
pub const RKNN_ERR_DEVICE_UNAVAILABLE: i32 = -3;
/// `malloc` / DMA-buffer allocation failed (often OOM on RV1106 256MB).
pub const RKNN_ERR_MALLOC_FAIL: i32 = -4;
/// Caller passed null, bad enum value, or out-of-range argument.
pub const RKNN_ERR_PARAM_INVALID: i32 = -5;
/// `.rknn` model bytes are corrupted or wrong magic.
pub const RKNN_ERR_MODEL_INVALID: i32 = -6;
/// `RknnContext` handle is null, destroyed, or from a different process.
pub const RKNN_ERR_CTX_INVALID: i32 = -7;
/// Input tensor mismatch (count, dtype, shape, or buffer size).
pub const RKNN_ERR_INPUT_INVALID: i32 = -8;
/// Output tensor mismatch (same kinds as INPUT_INVALID).
pub const RKNN_ERR_OUTPUT_INVALID: i32 = -9;
/// Model compiled for a different SoC than the current device.
pub const RKNN_ERR_DEVICE_UNMATCH: i32 = -10;
/// Pre-compiled model from older SDK incompatible with current runtime.
pub const RKNN_ERR_INCOMPATIBLE_PRE_COMPILE_MODEL: i32 = -11;
/// Optimization level in model file too new for current runtime.
pub const RKNN_ERR_INCOMPATIBLE_OPTIMIZATION_LEVEL_VERSION: i32 = -12;
/// `target_platform` in model doesn't match current SoC.
pub const RKNN_ERR_TARGET_PLATFORM_UNMATCH: i32 = -13;

/// Sizes of scalar RKNN fields used by fixed-size arrays throughout the API.
pub(crate) const RKNN_MAX_DIMS: usize = 16;
pub(crate) const RKNN_MAX_NAME_LEN: usize = 256;
pub(crate) const RKNN_MAX_DYNAMIC_SHAPE_NUM: usize = 512;

// ── Init flags ─────────────────────────────────────────────────────────

/// Priority: highest (default). Model execution preempts lower-priority contexts.
pub const RKNN_FLAG_PRIOR_HIGH: u32 = 0x00000000;
/// Priority: medium.
pub const RKNN_FLAG_PRIOR_MEDIUM: u32 = 0x00000001;
/// Priority: lowest.
pub const RKNN_FLAG_PRIOR_LOW: u32 = 0x00000002;
/// Enable asynchronous inference.
pub const RKNN_FLAG_ASYNC_MASK: u32 = 0x00000004;
/// Enable per-op performance collection (required for `perf_detail()`).
pub const RKNN_FLAG_COLLECT_PERF_MASK: u32 = 0x00000008;
/// I/O memory allocated outside the runtime — we manage buffers ourselves.
pub const RKNN_FLAG_MEM_ALLOC_OUTSIDE: u32 = 0x00000010;
/// Share weight memory across multiple contexts for the same model.
pub const RKNN_FLAG_SHARE_WEIGHT_MEM: u32 = 0x00000020;
/// Input fence passed externally (explicit inter-process sync).
pub const RKNN_FLAG_FENCE_IN_OUTSIDE: u32 = 0x00000040;
/// Output fence passed externally.
pub const RKNN_FLAG_FENCE_OUT_OUTSIDE: u32 = 0x00000080;
/// Init collects model info only; does not allocate inference resources.
pub const RKNN_FLAG_COLLECT_MODEL_INFO_ONLY: u32 = 0x00000100;
/// Internal scratch memory allocated outside.
pub const RKNN_FLAG_INTERNAL_ALLOC_OUTSIDE: u32 = 0x00000200;
/// Fall back to GPU if NPU cannot execute an op.
pub const RKNN_FLAG_EXECUTE_FALLBACK_PRIOR_DEVICE_GPU: u32 = 0x00000400;
/// Enable on-chip SRAM usage.
pub const RKNN_FLAG_ENABLE_SRAM: u32 = 0x00000800;
/// Share SRAM across contexts.
pub const RKNN_FLAG_SHARE_SRAM: u32 = 0x00001000;
/// Disable the runtime's high-priority worker process (power save).
pub const RKNN_FLAG_DISABLE_PROC_HIGH_PRIORITY: u32 = 0x00002000;
/// Skip input memory cache flush (caller handles cache management).
pub const RKNN_FLAG_DISABLE_FLUSH_INPUT_MEM_CACHE: u32 = 0x00004000;
/// Skip output memory cache flush.
pub const RKNN_FLAG_DISABLE_FLUSH_OUTPUT_MEM_CACHE: u32 = 0x00008000;
/// Model buffer is already in NPU-accessible zero-copy memory.
pub const RKNN_FLAG_MODEL_BUFFER_ZERO_COPY: u32 = 0x00010000;
/// Memory allocation does not require an active context.
pub const RKNN_MEM_FLAG_ALLOC_NO_CONTEXT: u32 = 0x00020000;

// ── Query commands ─────────────────────────────────────────────────────

// NOTE: SDK exposes `RKNN_QUERY_NATIVE_NC1HWC2_INPUT_ATTR` (= 8) and
// `RKNN_QUERY_NATIVE_NC1HWC2_OUTPUT_ATTR` (= 9) as aliases for
// `RKNN_QUERY_NATIVE_INPUT_ATTR` / `RKNN_QUERY_NATIVE_OUTPUT_ATTR`. They
// share the same numeric values; we only declare the canonical names below.
pub(crate) const RKNN_QUERY_IN_OUT_NUM: u32 = 0;
pub(crate) const RKNN_QUERY_INPUT_ATTR: u32 = 1;
pub(crate) const RKNN_QUERY_OUTPUT_ATTR: u32 = 2;
pub(crate) const RKNN_QUERY_PERF_DETAIL: u32 = 3;
pub(crate) const RKNN_QUERY_PERF_RUN: u32 = 4;
pub(crate) const RKNN_QUERY_SDK_VERSION: u32 = 5;
pub(crate) const RKNN_QUERY_MEM_SIZE: u32 = 6;
pub(crate) const RKNN_QUERY_CUSTOM_STRING: u32 = 7;
pub(crate) const RKNN_QUERY_NATIVE_INPUT_ATTR: u32 = 8;
pub(crate) const RKNN_QUERY_NATIVE_OUTPUT_ATTR: u32 = 9;
pub(crate) const RKNN_QUERY_NATIVE_NHWC_INPUT_ATTR: u32 = 10;
pub(crate) const RKNN_QUERY_NATIVE_NHWC_OUTPUT_ATTR: u32 = 11;
pub(crate) const RKNN_QUERY_DEVICE_MEM_INFO: u32 = 12;
pub(crate) const RKNN_QUERY_INPUT_DYNAMIC_RANGE: u32 = 13;
pub(crate) const RKNN_QUERY_CURRENT_INPUT_ATTR: u32 = 14;
pub(crate) const RKNN_QUERY_CURRENT_OUTPUT_ATTR: u32 = 15;
pub(crate) const RKNN_QUERY_CURRENT_NATIVE_INPUT_ATTR: u32 = 16;
pub(crate) const RKNN_QUERY_CURRENT_NATIVE_OUTPUT_ATTR: u32 = 17;

// ── Tensor type codes ──────────────────────────────────────────────────

/// Tensor data type (matches `rknn_tensor_type` enum).
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RknnTensorType {
    Float32 = 0,
    Float16 = 1,
    Int8 = 2,
    Uint8 = 3,
    Int16 = 4,
    Uint16 = 5,
    Int32 = 6,
    Uint32 = 7,
    Int64 = 8,
    Bool = 9,
    Int4 = 10,
    BFloat16 = 11,
}

impl RknnTensorType {
    /// Create from a raw `u32`. Unknown values map to `Uint8` for safety.
    pub fn from_u32(v: u32) -> Self {
        match v {
            0 => Self::Float32,
            1 => Self::Float16,
            2 => Self::Int8,
            3 => Self::Uint8,
            4 => Self::Int16,
            5 => Self::Uint16,
            6 => Self::Int32,
            7 => Self::Uint32,
            8 => Self::Int64,
            9 => Self::Bool,
            10 => Self::Int4,
            11 => Self::BFloat16,
            _ => Self::Uint8,
        }
    }
}

// ── Tensor format codes ────────────────────────────────────────────────

/// Tensor data layout (matches `rknn_tensor_format` enum).
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RknnTensorFormat {
    Nchw = 0,
    Nhwc = 1,
    /// Native NPU tile layout, two-channel-dim variant used by RK3588 etc.
    Nc1hwc2 = 2,
    Undefined = 3,
}

impl RknnTensorFormat {
    pub fn from_u32(v: u32) -> Self {
        match v {
            0 => Self::Nchw,
            1 => Self::Nhwc,
            2 => Self::Nc1hwc2,
            _ => Self::Undefined,
        }
    }
}

// ── Quantization type codes ────────────────────────────────────────────

/// Tensor quantization scheme (matches `rknn_tensor_qnt_type` enum).
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RknnQuantType {
    None = 0,
    /// Dynamic fixed-point with `fl` (fractional length) parameter.
    Dfp = 1,
    /// Asymmetric affine quantization with scale and zero-point.
    AffineAsymmetric = 2,
}

impl RknnQuantType {
    pub fn from_u32(v: u32) -> Self {
        match v {
            0 => Self::None,
            1 => Self::Dfp,
            2 => Self::AffineAsymmetric,
            _ => Self::None,
        }
    }
}

// ── NPU core mask ──────────────────────────────────────────────────────

/// NPU core selection for multi-core SoCs (RK3588 has 3 cores).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NpuCoreMask {
    /// Runtime chooses core automatically.
    Auto,
    /// Pin to NPU core 0.
    Core0,
    /// Pin to NPU core 1.
    Core1,
    /// Pin to NPU core 2 (RK3588 only).
    Core2,
    /// Use cores 0 and 1 in batch / parallel mode.
    Cores01,
    /// Use cores 0 and 2.
    Cores02,
    /// Use cores 1 and 2.
    Cores12,
    /// Use all 3 cores (RK3588) for maximum throughput.
    Cores012,
    /// All available cores (driver-decided mask).
    All,
}

impl NpuCoreMask {
    /// Raw bitmask value passed to `rknn_set_core_mask`.
    pub fn as_raw(self) -> u32 {
        match self {
            Self::Auto => 0,
            Self::Core0 => 1,
            Self::Core1 => 2,
            Self::Core2 => 4,
            Self::Cores01 => 3,
            Self::Cores02 => 5,
            Self::Cores12 => 6,
            Self::Cores012 => 7,
            Self::All => 0xffff,
        }
    }
}

// ── Memory flags ───────────────────────────────────────────────────────

/// Allocation flags for `RknnMem` (matches `rknn_mem_alloc_flags`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemAllocFlags {
    /// Driver default (typically cacheable on RK3588, non-cacheable on smaller SoCs).
    Default,
    /// CPU-cacheable allocation; call `sync_to_device()` before NPU read.
    Cacheable,
    /// Non-cacheable allocation; no manual sync required but slower CPU access.
    NonCacheable,
    /// Try to allocate in on-chip SRAM; falls back to DMA memory on failure.
    TryAllocSram,
}

impl MemAllocFlags {
    pub fn as_raw(self) -> u64 {
        match self {
            Self::Default => 0,
            Self::Cacheable => 1,
            Self::NonCacheable => 2,
            Self::TryAllocSram => 4,
        }
    }
}

/// Cache synchronisation direction (matches `rknn_mem_sync_mode`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemSyncMode {
    /// CPU wrote — flush cache so NPU sees latest data.
    ToDevice,
    /// NPU wrote — invalidate cache so CPU reads latest data.
    FromDevice,
    /// Both directions.
    Bidirectional,
}

impl MemSyncMode {
    pub fn as_raw(self) -> u32 {
        match self {
            Self::ToDevice => 0x1,
            Self::FromDevice => 0x2,
            Self::Bidirectional => 0x3,
        }
    }
}

// ── Matmul type codes ──────────────────────────────────────────────────

/// Matmul input/output type combination (matches `rknn_matmul_type`).
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RknnMatmulType {
    Float16MmFloat16ToFloat32 = 1,
    Int8MmInt8ToInt32 = 2,
    Int8MmInt8ToInt8 = 3,
    Float16MmFloat16ToFloat16 = 4,
    Float16MmInt8ToFloat32 = 5,
    Float16MmInt8ToFloat16 = 6,
    Float16MmInt4ToFloat32 = 7,
    Float16MmInt4ToFloat16 = 8,
    Int8MmInt8ToFloat32 = 9,
    Int4MmInt4ToInt16 = 10,
    Int8MmInt4ToInt32 = 11,
    Float16MmInt4ToBFloat16 = 12,
    Int8MmInt4ToFloat16 = 15,
}

/// Matmul quantization scheme (matches `rknn_matmul_quant_type`).
#[repr(i16)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RknnMatmulQuantType {
    PerLayerSym = 0,
    PerLayerAsym = 1,
    PerChannelSym = 2,
    PerChannelAsym = 3,
    PerGroupSym = 4,
    PerGroupAsym = 5,
}

/// B-tensor layout for matmul (matches `rknn_matmul_layout`).
#[repr(i16)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RknnMatmulLayout {
    /// Normal row-major layout.
    Norm = 0,
    /// NPU-native tiled layout (use `transform_b_layout()` to convert).
    Native = 1,
    /// Transposed normal layout.
    TpNorm = 2,
}
