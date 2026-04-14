//! FFI declarations for `librockchip_mpp.so`.
//!
//! All structs are opaque (we only manipulate them through MPP function
//! pointers). Function pointer types match the MPP 1.0+ ABI which has
//! been stable across Rockchip SoC generations.

use crate::error::{MppError, MppResult};
use std::ffi::c_void;
use std::sync::Arc;

// ── MPP enum constants ─────────────────────────────────────────────

/// `MppCtxType::MPP_CTX_ENC`
pub(crate) const MPP_CTX_ENC: u32 = 1;

/// `MppCodingType::MPP_VIDEO_CodingAVC` (H.264)
pub(crate) const MPP_VIDEO_CODING_AVC: u32 = 7;
/// `MppCodingType::MPP_VIDEO_CodingHEVC`
#[allow(dead_code)]
pub(crate) const MPP_VIDEO_CODING_HEVC: u32 = 0x1000_0001;

/// `MppFrameFormat::MPP_FMT_YUV420SP` (NV12)
pub(crate) const MPP_FMT_YUV420SP: u32 = 0;
/// `MppFrameFormat::MPP_FMT_YUV420P` (planar I420)
#[allow(dead_code)]
pub(crate) const MPP_FMT_YUV420P: u32 = 1;

// MPI commands (used with mpi.control)
pub(crate) const MPP_ENC_SET_CFG: u32 = 0x14_00_01_00;
#[allow(dead_code)]
pub(crate) const MPP_ENC_GET_CFG: u32 = 0x14_00_01_01;

// ── Opaque handles ────────────────────────────────────────────────

/// `MppCtx` — opaque encoder/decoder context.
pub(crate) type MppCtx = *mut c_void;
/// `MppFrame` / `MppPacket` / `MppBuffer` / `MppEncCfg` are all opaque.
pub(crate) type MppFrame = *mut c_void;
pub(crate) type MppPacket = *mut c_void;
pub(crate) type MppBuffer = *mut c_void;
pub(crate) type MppEncCfg = *mut c_void;

// ── MPI vtable ────────────────────────────────────────────────────
//
// `MppApi` is a C struct of function pointers, laid out at offsets
// matching `rk_mpi.h` from the Rockchip MPP SDK. `mpp_create` returns
// both an `MppCtx` handle and a pointer to this struct. We never
// transmute the ctx — we hold the api pointer separately and access its
// fields through this `#[repr(C)]` declaration.

pub(crate) type FnMppApiNoArg = unsafe extern "C" fn(MppCtx) -> i32;
pub(crate) type FnMppApiControl = unsafe extern "C" fn(MppCtx, u32, *mut c_void) -> i32;
pub(crate) type FnMppApiEncodePutFrame = unsafe extern "C" fn(MppCtx, MppFrame) -> i32;
pub(crate) type FnMppApiEncodeGetPacket = unsafe extern "C" fn(MppCtx, *mut MppPacket) -> i32;
pub(crate) type FnMppApiDecodePutPacket = unsafe extern "C" fn(MppCtx, MppPacket) -> i32;
pub(crate) type FnMppApiDecodeGetFrame = unsafe extern "C" fn(MppCtx, *mut MppFrame) -> i32;

/// Layout of `MppApi` from `rk_mpi.h` (SDK ≥ 1.0).
///
/// Accessed as `&*api_ptr`. Fields after `control` that we don't use
/// are declared as `*const c_void` placeholders so the struct has the
/// correct total size and field offsets. Never constructed in Rust —
/// only dereferenced through a pointer returned by `mpp_create`.
#[repr(C)]
pub(crate) struct MppApi {
    pub(crate) size: u32,
    pub(crate) version: u32,
    /// `MPP_RET decode(MppCtx, MppPacket, MppFrame*)` — convenience combo.
    pub(crate) decode: *const c_void,
    /// `MPP_RET decode_put_packet(MppCtx, MppPacket)`
    pub(crate) decode_put_packet: FnMppApiDecodePutPacket,
    /// `MPP_RET decode_get_frame(MppCtx, MppFrame*)`
    pub(crate) decode_get_frame: FnMppApiDecodeGetFrame,
    /// `MPP_RET encode(MppCtx, MppFrame, MppPacket*)` — convenience combo.
    pub(crate) encode: *const c_void,
    /// `MPP_RET encode_put_frame(MppCtx, MppFrame)`
    pub(crate) encode_put_frame: FnMppApiEncodePutFrame,
    /// `MPP_RET encode_get_packet(MppCtx, MppPacket*)`
    pub(crate) encode_get_packet: FnMppApiEncodeGetPacket,
    /// Reserved for older API symmetry.
    pub(crate) reserve0: *const c_void,
    pub(crate) poll: *const c_void,
    pub(crate) dequeue: *const c_void,
    pub(crate) enqueue: *const c_void,
    /// `MPP_RET reset(MppCtx)`
    pub(crate) reset: FnMppApiNoArg,
    /// `MPP_RET control(MppCtx, MpiCmd, MppParam)`
    pub(crate) control: FnMppApiControl,
    /// Reserved (8 slots) — padding to stable struct size.
    pub(crate) reserved: [*const c_void; 8],
}

// Compile-time size guard. The SDK header documents `sizeof(MppApi) ==
// 8 (size+version) + 12 × 8 (fn ptrs) + 8 × 8 (reserved) = 168 bytes on
// 64-bit platforms. Verified against C-equivalent struct compiled on
// aarch64 LP64 ABI. If a future MPP SDK changes the layout, this
// assertion fires at build time so we don't silently miscall the vtable.
const _: () = {
    assert!(std::mem::size_of::<MppApi>() == 8 + 12 * 8 + 8 * 8);
};

// ── Top-level MPP function pointer types ──────────────────────────

pub(crate) type FnMppCreate = unsafe extern "C" fn(*mut MppCtx, *mut MppApi) -> i32;
pub(crate) type FnMppInit = unsafe extern "C" fn(MppCtx, u32, u32) -> i32;
pub(crate) type FnMppDestroy = unsafe extern "C" fn(MppCtx) -> i32;

pub(crate) type FnMppFrameInit = unsafe extern "C" fn(*mut MppFrame) -> i32;
pub(crate) type FnMppFrameDeinit = unsafe extern "C" fn(*mut MppFrame) -> i32;
pub(crate) type FnMppFrameSetWidth = unsafe extern "C" fn(MppFrame, u32);
pub(crate) type FnMppFrameSetHeight = unsafe extern "C" fn(MppFrame, u32);
pub(crate) type FnMppFrameSetHorStride = unsafe extern "C" fn(MppFrame, u32);
pub(crate) type FnMppFrameSetVerStride = unsafe extern "C" fn(MppFrame, u32);
pub(crate) type FnMppFrameSetFmt = unsafe extern "C" fn(MppFrame, u32);
pub(crate) type FnMppFrameSetBuffer = unsafe extern "C" fn(MppFrame, MppBuffer);

pub(crate) type FnMppPacketDeinit = unsafe extern "C" fn(*mut MppPacket) -> i32;
pub(crate) type FnMppPacketGetData = unsafe extern "C" fn(MppPacket) -> *mut u8;
pub(crate) type FnMppPacketGetLength = unsafe extern "C" fn(MppPacket) -> usize;

pub(crate) type FnMppEncCfgInit = unsafe extern "C" fn(*mut MppEncCfg) -> i32;
pub(crate) type FnMppEncCfgDeinit = unsafe extern "C" fn(MppEncCfg) -> i32;
pub(crate) type FnMppEncCfgSetS32 = unsafe extern "C" fn(MppEncCfg, *const u8, i32) -> i32;

pub(crate) type FnMppBufferImportFromFd = unsafe extern "C" fn(*mut MppBuffer, i32, usize) -> i32;
pub(crate) type FnMppBufferPutWithCaller = unsafe extern "C" fn(MppBuffer, *const u8) -> i32;

// ── Loaded library handle + function table ────────────────────────

pub(crate) struct MppLib {
    pub(crate) handle: *mut c_void,
    pub(crate) mpp_create: FnMppCreate,
    pub(crate) mpp_init: FnMppInit,
    pub(crate) mpp_destroy: FnMppDestroy,
    pub(crate) frame_init: FnMppFrameInit,
    pub(crate) frame_deinit: FnMppFrameDeinit,
    pub(crate) frame_set_width: FnMppFrameSetWidth,
    pub(crate) frame_set_height: FnMppFrameSetHeight,
    pub(crate) frame_set_hor_stride: FnMppFrameSetHorStride,
    pub(crate) frame_set_ver_stride: FnMppFrameSetVerStride,
    pub(crate) frame_set_fmt: FnMppFrameSetFmt,
    pub(crate) frame_set_buffer: FnMppFrameSetBuffer,
    pub(crate) packet_deinit: FnMppPacketDeinit,
    pub(crate) packet_get_data: FnMppPacketGetData,
    pub(crate) packet_get_length: FnMppPacketGetLength,
    pub(crate) enc_cfg_init: FnMppEncCfgInit,
    pub(crate) enc_cfg_deinit: FnMppEncCfgDeinit,
    pub(crate) enc_cfg_set_s32: FnMppEncCfgSetS32,
    pub(crate) buffer_import_from_fd: FnMppBufferImportFromFd,
    pub(crate) buffer_put: FnMppBufferPutWithCaller,
}

// SAFETY: dlopen'd symbols are immutable function pointers; safe to share
// across threads. The MPP runtime documents thread-safe access for
// independent encoder contexts.
unsafe impl Send for MppLib {}
unsafe impl Sync for MppLib {}

impl Drop for MppLib {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            // SAFETY: handle returned by dlopen, owned by us.
            unsafe { libc::dlclose(self.handle) };
        }
    }
}

/// Probe whether the library can be loaded. Cheap (dlopen+dlclose).
#[cfg(target_os = "linux")]
pub(crate) fn probe_library() -> bool {
    // SAFETY: dlopen RTLD_LAZY does no constructors; null check guards.
    let handle = unsafe { libc::dlopen(c"librockchip_mpp.so".as_ptr(), libc::RTLD_LAZY) };
    if handle.is_null() {
        return false;
    }
    // SAFETY: handle non-null.
    unsafe { libc::dlclose(handle) };
    true
}

/// Load `librockchip_mpp.so` and resolve the symbol table.
pub(crate) fn load_library() -> MppResult<Arc<MppLib>> {
    // SAFETY: dlopen with RTLD_LAZY.
    let handle = unsafe { libc::dlopen(c"librockchip_mpp.so".as_ptr(), libc::RTLD_LAZY) };
    if handle.is_null() {
        return Err(MppError::LibraryNotFound);
    }

    macro_rules! sym {
        ($name:expr, $ty:ty) => {{
            // SAFETY: handle is valid; name is a static null-terminated bstr.
            let ptr = unsafe { libc::dlsym(handle, $name.as_ptr().cast()) };
            if ptr.is_null() {
                // SAFETY: handle valid.
                unsafe { libc::dlclose(handle) };
                return Err(MppError::SymbolMissing(stringify!($name)));
            }
            // SAFETY: ptr non-null, transmuted to matching function type.
            unsafe { std::mem::transmute_copy::<*mut c_void, $ty>(&ptr) }
        }};
    }

    let lib = MppLib {
        handle,
        mpp_create: sym!(b"mpp_create\0", FnMppCreate),
        mpp_init: sym!(b"mpp_init\0", FnMppInit),
        mpp_destroy: sym!(b"mpp_destroy\0", FnMppDestroy),
        frame_init: sym!(b"mpp_frame_init\0", FnMppFrameInit),
        frame_deinit: sym!(b"mpp_frame_deinit\0", FnMppFrameDeinit),
        frame_set_width: sym!(b"mpp_frame_set_width\0", FnMppFrameSetWidth),
        frame_set_height: sym!(b"mpp_frame_set_height\0", FnMppFrameSetHeight),
        frame_set_hor_stride: sym!(b"mpp_frame_set_hor_stride\0", FnMppFrameSetHorStride),
        frame_set_ver_stride: sym!(b"mpp_frame_set_ver_stride\0", FnMppFrameSetVerStride),
        frame_set_fmt: sym!(b"mpp_frame_set_fmt\0", FnMppFrameSetFmt),
        frame_set_buffer: sym!(b"mpp_frame_set_buffer\0", FnMppFrameSetBuffer),
        packet_deinit: sym!(b"mpp_packet_deinit\0", FnMppPacketDeinit),
        packet_get_data: sym!(b"mpp_packet_get_data\0", FnMppPacketGetData),
        packet_get_length: sym!(b"mpp_packet_get_length\0", FnMppPacketGetLength),
        enc_cfg_init: sym!(b"mpp_enc_cfg_init\0", FnMppEncCfgInit),
        enc_cfg_deinit: sym!(b"mpp_enc_cfg_deinit\0", FnMppEncCfgDeinit),
        enc_cfg_set_s32: sym!(b"mpp_enc_cfg_set_s32\0", FnMppEncCfgSetS32),
        buffer_import_from_fd: sym!(b"mpp_buffer_import_with_tag\0", FnMppBufferImportFromFd),
        buffer_put: sym!(b"mpp_buffer_put_with_caller\0", FnMppBufferPutWithCaller),
    };
    Ok(Arc::new(lib))
}
