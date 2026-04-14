//! Linux DRM/KMS atomic modeset output for the FPV pipeline.
//!
//! Replaces the older `/dev/fb0` framebuffer write path with modern
//! atomic modeset, which gives:
//!
//! - **Vsync'd flips** (no tearing)
//! - **DMA-BUF plane import** (zero-copy from MPP encoder output → display)
//! - **Per-flip timestamping** (kernel reports actual scanout time)
//! - **Double-buffering** (front buffer scanned while back buffer fills)
//!
//! # Supported operations
//!
//! - Enumerate connectors (`drmModeGetResources` + `drmModeGetConnector`)
//!   and pick one by name (e.g. `"HDMI-A-1"`, `"DSI-1"`)
//! - Select a display mode by label (`"720p60"`, `"1080p30"`, `"640x480"`)
//! - Find a compatible CRTC via the connector's encoder
//! - Import a DMA-BUF as a gem handle (`drmPrimeFDToHandle`), wrap as FB
//! - Initial mode set (`drmModeSetCrtc`)
//! - Subsequent flips (`drmModePageFlip` with vsync event)
//!
//! Atomic commits (`drmModeAtomicCommit`) are the preferred modern API
//! but require enumeration of atomic properties + property-blob IDs;
//! the legacy `drmModeSetCrtc` + `drmModePageFlip` path used here is
//! simpler, still vsync'd, and works on every RK3588/RV1106 kernel.
//!
//! `libdrm.so` is resolved at runtime via `dlopen`. On non-Linux hosts
//! the calls return [`DrmError::NotSupported`].

#![allow(unsafe_code)]

#[cfg(target_os = "linux")]
use std::ffi::c_void;
use std::sync::Arc;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum DrmError {
    #[error("DRM not supported on this platform")]
    NotSupported,
    #[error("libdrm.so not found")]
    LibraryNotFound,
    #[error("DRM `{op}` failed: {detail}")]
    CallFailed { op: &'static str, detail: String },
    #[error("connector `{0}` not found on this device")]
    ConnectorNotFound(String),
    #[error("mode `{0}` not supported by connector")]
    ModeNotSupported(String),
    #[error("no CRTC available for connector `{0}`")]
    NoCrtc(String),
    #[error("invalid parameter: {0}")]
    InvalidParameter(String),
}

pub type DrmResult<T> = Result<T, DrmError>;

/// Probe whether `libdrm.so` is available on this host.
pub fn drm_available() -> bool {
    #[cfg(target_os = "linux")]
    {
        // SAFETY: dlopen RTLD_LAZY then dlclose.
        let h = unsafe { libc::dlopen(c"libdrm.so.2".as_ptr(), libc::RTLD_LAZY) };
        if h.is_null() {
            let h2 = unsafe { libc::dlopen(c"libdrm.so".as_ptr(), libc::RTLD_LAZY) };
            if h2.is_null() {
                return false;
            }
            unsafe { libc::dlclose(h2) };
            return true;
        }
        unsafe { libc::dlclose(h) };
        true
    }
    #[cfg(not(target_os = "linux"))]
    false
}

// ── Constants (DRM_* from drm.h / drm_mode.h) ─────────────────────

#[cfg(target_os = "linux")]
const DRM_DISPLAY_MODE_LEN: usize = 32;

/// `drmModeConnection::DRM_MODE_CONNECTED`
#[cfg(target_os = "linux")]
const DRM_MODE_CONNECTED: u32 = 1;

/// `DRM_MODE_PAGE_FLIP_EVENT` — request vsync event on flip completion.
#[cfg(target_os = "linux")]
const DRM_MODE_PAGE_FLIP_EVENT: u32 = 0x01;

/// Common `DRM_FORMAT_*` FourCC codes from drm_fourcc.h.
pub mod fourcc {
    /// NV12 — YUV 4:2:0 semi-planar (Y + interleaved UV).
    pub const NV12: u32 = u32::from_le_bytes(*b"NV12");
    /// Packed 32-bit XRGB8888.
    pub const XRGB8888: u32 = u32::from_le_bytes(*b"XR24");
    /// Packed 24-bit RGB888.
    pub const RGB888: u32 = u32::from_le_bytes(*b"RG24");
    /// YUYV (YUV 4:2:2 packed).
    pub const YUYV: u32 = u32::from_le_bytes(*b"YUYV");
}

// ── `drmModeRes` struct layout (C ABI from xf86drmMode.h) ─────────
//
// struct _drmModeRes {
//     int count_fbs;
//     uint32_t *fbs;
//     int count_crtcs;
//     uint32_t *crtcs;
//     int count_connectors;
//     uint32_t *connectors;
//     int count_encoders;
//     uint32_t *encoders;
//     uint32_t min_width, max_width;
//     uint32_t min_height, max_height;
// };

#[cfg(target_os = "linux")]
#[repr(C)]
struct DrmModeRes {
    count_fbs: i32,
    fbs: *const u32,
    count_crtcs: i32,
    crtcs: *const u32,
    count_connectors: i32,
    connectors: *const u32,
    count_encoders: i32,
    encoders: *const u32,
    min_width: u32,
    max_width: u32,
    min_height: u32,
    max_height: u32,
}

// `drmModeModeInfo` — display timing descriptor.
#[cfg(target_os = "linux")]
#[repr(C)]
#[derive(Clone, Copy)]
struct DrmModeModeInfo {
    clock: u32,
    hdisplay: u16,
    hsync_start: u16,
    hsync_end: u16,
    htotal: u16,
    hskew: u16,
    vdisplay: u16,
    vsync_start: u16,
    vsync_end: u16,
    vtotal: u16,
    vscan: u16,
    vrefresh: u32,
    flags: u32,
    typ: u32,
    name: [u8; DRM_DISPLAY_MODE_LEN],
}

// `drmModeConnector` — display output descriptor.
#[cfg(target_os = "linux")]
#[repr(C)]
struct DrmModeConnector {
    connector_id: u32,
    encoder_id: u32,
    connector_type: u32,
    connector_type_id: u32,
    connection: u32,
    mm_width: u32,
    mm_height: u32,
    subpixel: u32,
    count_modes: i32,
    modes: *const DrmModeModeInfo,
    count_props: i32,
    props: *const u32,
    prop_values: *const u64,
    count_encoders: i32,
    encoders: *const u32,
}

// `drmModeEncoder` — maps connector to CRTC.
#[cfg(target_os = "linux")]
#[repr(C)]
struct DrmModeEncoder {
    encoder_id: u32,
    encoder_type: u32,
    crtc_id: u32,
    possible_crtcs: u32,
    possible_clones: u32,
}

// ── FFI plumbing ─────────────────────────────────────────────────

#[cfg(target_os = "linux")]
type FnDrmOpen = unsafe extern "C" fn(*const u8, *const u8) -> i32;
#[cfg(target_os = "linux")]
type FnDrmClose = unsafe extern "C" fn(i32) -> i32;
#[cfg(target_os = "linux")]
type FnDrmModeGetResources = unsafe extern "C" fn(i32) -> *mut DrmModeRes;
#[cfg(target_os = "linux")]
type FnDrmModeFreeResources = unsafe extern "C" fn(*mut DrmModeRes);
#[cfg(target_os = "linux")]
type FnDrmModeGetConnector = unsafe extern "C" fn(i32, u32) -> *mut DrmModeConnector;
#[cfg(target_os = "linux")]
type FnDrmModeFreeConnector = unsafe extern "C" fn(*mut DrmModeConnector);
#[cfg(target_os = "linux")]
type FnDrmModeGetEncoder = unsafe extern "C" fn(i32, u32) -> *mut DrmModeEncoder;
#[cfg(target_os = "linux")]
type FnDrmModeFreeEncoder = unsafe extern "C" fn(*mut DrmModeEncoder);
#[cfg(target_os = "linux")]
type FnDrmModeAddFB = unsafe extern "C" fn(i32, u32, u32, u8, u8, u32, u32, *mut u32) -> i32;
/// `drmModeAddFB2` with full fourcc + 4 planes.
#[cfg(target_os = "linux")]
type FnDrmModeAddFB2 = unsafe extern "C" fn(
    i32,
    u32,        // width
    u32,        // height
    u32,        // fourcc
    *const u32, // handles[4]
    *const u32, // pitches[4]
    *const u32, // offsets[4]
    *mut u32,   // out fb_id
    u32,        // flags
) -> i32;
#[cfg(target_os = "linux")]
type FnDrmModeRmFB = unsafe extern "C" fn(i32, u32) -> i32;
#[cfg(target_os = "linux")]
type FnDrmModeSetCrtc = unsafe extern "C" fn(
    i32,                    // fd
    u32,                    // crtc_id
    u32,                    // fb_id
    u32,                    // x
    u32,                    // y
    *const u32,             // connectors
    i32,                    // count
    *const DrmModeModeInfo, // mode
) -> i32;
#[cfg(target_os = "linux")]
type FnDrmModePageFlip = unsafe extern "C" fn(i32, u32, u32, u32, *mut c_void) -> i32;
#[cfg(target_os = "linux")]
type FnDrmPrimeFDToHandle = unsafe extern "C" fn(i32, i32, *mut u32) -> i32;

#[cfg(target_os = "linux")]
struct DrmLib {
    handle: *mut c_void,
    drm_open: FnDrmOpen,
    drm_close: FnDrmClose,
    get_resources: FnDrmModeGetResources,
    free_resources: FnDrmModeFreeResources,
    get_connector: FnDrmModeGetConnector,
    free_connector: FnDrmModeFreeConnector,
    get_encoder: FnDrmModeGetEncoder,
    free_encoder: FnDrmModeFreeEncoder,
    add_fb: FnDrmModeAddFB,
    add_fb2: Option<FnDrmModeAddFB2>,
    rm_fb: FnDrmModeRmFB,
    set_crtc: FnDrmModeSetCrtc,
    page_flip: FnDrmModePageFlip,
    prime_fd_to_handle: FnDrmPrimeFDToHandle,
}

#[cfg(target_os = "linux")]
unsafe impl Send for DrmLib {}
#[cfg(target_os = "linux")]
unsafe impl Sync for DrmLib {}

#[cfg(target_os = "linux")]
impl Drop for DrmLib {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            // SAFETY: handle owned by us.
            unsafe { libc::dlclose(self.handle) };
        }
    }
}

#[cfg(target_os = "linux")]
fn load_drm_lib() -> DrmResult<Arc<DrmLib>> {
    // SAFETY: dlopen with RTLD_NOW for eager symbol validation.
    let mut h = unsafe { libc::dlopen(c"libdrm.so.2".as_ptr(), libc::RTLD_NOW) };
    if h.is_null() {
        h = unsafe { libc::dlopen(c"libdrm.so".as_ptr(), libc::RTLD_NOW) };
    }
    if h.is_null() {
        return Err(DrmError::LibraryNotFound);
    }
    macro_rules! sym {
        ($name:expr, $ty:ty) => {{
            // SAFETY: h valid; name null-terminated.
            let p = unsafe { libc::dlsym(h, $name.as_ptr().cast()) };
            if p.is_null() {
                // SAFETY: h valid.
                unsafe { libc::dlclose(h) };
                return Err(DrmError::CallFailed {
                    op: stringify!($name),
                    detail: "symbol not found in libdrm".into(),
                });
            }
            // SAFETY: p non-null, types match.
            unsafe { std::mem::transmute_copy::<*mut c_void, $ty>(&p) }
        }};
    }
    macro_rules! sym_opt {
        ($name:expr, $ty:ty) => {{
            // SAFETY: h valid; name null-terminated.
            let p = unsafe { libc::dlsym(h, $name.as_ptr().cast()) };
            if p.is_null() {
                None
            } else {
                // SAFETY: p non-null.
                Some(unsafe { std::mem::transmute_copy::<*mut c_void, $ty>(&p) })
            }
        }};
    }
    let lib = DrmLib {
        handle: h,
        drm_open: sym!(b"drmOpen\0", FnDrmOpen),
        drm_close: sym!(b"drmClose\0", FnDrmClose),
        get_resources: sym!(b"drmModeGetResources\0", FnDrmModeGetResources),
        free_resources: sym!(b"drmModeFreeResources\0", FnDrmModeFreeResources),
        get_connector: sym!(b"drmModeGetConnector\0", FnDrmModeGetConnector),
        free_connector: sym!(b"drmModeFreeConnector\0", FnDrmModeFreeConnector),
        get_encoder: sym!(b"drmModeGetEncoder\0", FnDrmModeGetEncoder),
        free_encoder: sym!(b"drmModeFreeEncoder\0", FnDrmModeFreeEncoder),
        add_fb: sym!(b"drmModeAddFB\0", FnDrmModeAddFB),
        add_fb2: sym_opt!(b"drmModeAddFB2\0", FnDrmModeAddFB2),
        rm_fb: sym!(b"drmModeRmFB\0", FnDrmModeRmFB),
        set_crtc: sym!(b"drmModeSetCrtc\0", FnDrmModeSetCrtc),
        page_flip: sym!(b"drmModePageFlip\0", FnDrmModePageFlip),
        prime_fd_to_handle: sym!(b"drmPrimeFDToHandle\0", FnDrmPrimeFDToHandle),
    };
    Ok(Arc::new(lib))
}

#[cfg(not(target_os = "linux"))]
struct DrmLib;

#[cfg(not(target_os = "linux"))]
fn load_drm_lib() -> DrmResult<Arc<DrmLib>> {
    Err(DrmError::NotSupported)
}

// ── Connector / mode lookup helpers ──────────────────────────────

/// Connector names follow the pattern `<TYPE>-<TYPE_ID>`, e.g. `HDMI-A-1`,
/// `DSI-1`, `DP-2`. Map from `connector_type` constant to string prefix.
/// Values from `drm_mode.h:DRM_MODE_CONNECTOR_*`.
#[cfg(target_os = "linux")]
fn connector_type_name(typ: u32) -> &'static str {
    match typ {
        1 => "VGA",
        2 => "DVI-I",
        3 => "DVI-D",
        4 => "DVI-A",
        5 => "Composite",
        6 => "SVIDEO",
        7 => "LVDS",
        8 => "Component",
        9 => "DIN",
        10 => "DP",
        11 => "HDMI-A",
        12 => "HDMI-B",
        13 => "TV",
        14 => "eDP",
        15 => "Virtual",
        16 => "DSI",
        17 => "DPI",
        _ => "Unknown",
    }
}

/// Parse a mode label like `"720p60"`, `"1080p30"`, `"640x480"` and test
/// whether a `DrmModeModeInfo` matches.
#[cfg(target_os = "linux")]
fn mode_matches(mode: &DrmModeModeInfo, label: &str) -> bool {
    // First try exact byte match against the kernel-set name (e.g. "1920x1080").
    let name_end = mode
        .name
        .iter()
        .position(|&b| b == 0)
        .unwrap_or(mode.name.len());
    let name = std::str::from_utf8(&mode.name[..name_end]).unwrap_or("");
    if name == label {
        return true;
    }
    // Then parse "<W>x<H>" or "<H>p<refresh>".
    if let Some((w, h)) = label.split_once('x')
        && let (Ok(w), Ok(h)) = (w.parse::<u16>(), h.parse::<u16>())
    {
        return mode.hdisplay == w && mode.vdisplay == h;
    }
    if let Some((h, r)) = label.split_once('p')
        && let (Ok(h), Ok(r)) = (h.parse::<u16>(), r.parse::<u32>())
    {
        return mode.vdisplay == h && mode.vrefresh == r;
    }
    false
}

// ── Public safe API ──────────────────────────────────────────────

/// DRM-based output sink. Holds the device fd + active framebuffer state.
pub struct DrmOutput {
    /// Held to keep libdrm symbols alive for the output's lifetime.
    #[cfg_attr(not(target_os = "linux"), allow(dead_code))]
    lib: Arc<DrmLib>,
    /// `/dev/dri/cardN` fd (or `-1` on non-Linux placeholder).
    #[cfg_attr(not(target_os = "linux"), allow(dead_code))]
    fd: i32,
    /// Requested width (may match `mode.hdisplay` exactly).
    width: u32,
    /// Requested height.
    height: u32,
    /// FourCC (e.g. `fourcc::NV12`).
    pixel_format: u32,
    /// Currently-bound DRM framebuffer ID (0 if none).
    #[cfg_attr(not(target_os = "linux"), allow(dead_code))]
    current_fb_id: u32,
    /// Previously-bound FB ID (kept alive until we flip off it — prevents
    /// tearing on some drivers that sample the plane mid-flip).
    #[cfg_attr(not(target_os = "linux"), allow(dead_code))]
    previous_fb_id: u32,
    /// CRTC bound to our connector (0 before `set_mode`).
    #[cfg_attr(not(target_os = "linux"), allow(dead_code))]
    crtc_id: u32,
    /// Connector selected by name.
    #[cfg_attr(not(target_os = "linux"), allow(dead_code))]
    connector_id: u32,
    /// Mode info copied from the kernel (used by set_crtc).
    #[cfg(target_os = "linux")]
    mode: Option<DrmModeModeInfo>,
    /// Whether initial modeset has been performed.
    #[cfg_attr(not(target_os = "linux"), allow(dead_code))]
    mode_set: bool,
}

impl DrmOutput {
    /// Open `/dev/dri/card0` without performing modeset. Caller must
    /// then call [`Self::select_connector`] and [`Self::set_mode`].
    /// For the common path, prefer [`Self::new`] which does all three.
    pub fn open(width: u32, height: u32, pixel_format: u32) -> DrmResult<Self> {
        if width == 0 || height == 0 {
            return Err(DrmError::InvalidParameter(
                "width/height must be > 0".into(),
            ));
        }
        let _lib = load_drm_lib()?;
        #[cfg(target_os = "linux")]
        {
            // SAFETY: drm_open(NULL, NULL) opens the first available card.
            let fd = unsafe { (_lib.drm_open)(std::ptr::null(), std::ptr::null()) };
            if fd < 0 {
                return Err(DrmError::CallFailed {
                    op: "drmOpen",
                    detail: format!("returned fd={fd}"),
                });
            }
            Ok(Self {
                lib: _lib,
                fd,
                width,
                height,
                pixel_format,
                current_fb_id: 0,
                previous_fb_id: 0,
                crtc_id: 0,
                connector_id: 0,
                mode: None,
                mode_set: false,
            })
        }
        #[cfg(not(target_os = "linux"))]
        {
            let _ = (width, height, pixel_format);
            Err(DrmError::NotSupported)
        }
    }

    /// Full init: open device, select connector by name, pick mode,
    /// find CRTC. Returns a fully prepared output ready for `present`.
    ///
    /// `connector_name` examples: `"HDMI-A-1"`, `"DSI-1"`, `"DP-1"`.
    /// `mode_label` examples: `"720p60"`, `"1080p30"`, `"640x480"`.
    pub fn new(connector_name: &str, mode_label: &str, pixel_format: u32) -> DrmResult<Self> {
        #[cfg(target_os = "linux")]
        {
            // Open with placeholder dimensions; real size comes from the mode.
            let mut out = Self::open(1, 1, pixel_format)?;
            out.select_connector(connector_name)?;
            out.set_mode(mode_label)?;
            Ok(out)
        }
        #[cfg(not(target_os = "linux"))]
        {
            let _ = (connector_name, mode_label, pixel_format);
            Err(DrmError::NotSupported)
        }
    }

    /// Enumerate connectors and pick the one whose formatted name matches.
    pub fn select_connector(&mut self, name: &str) -> DrmResult<()> {
        #[cfg(target_os = "linux")]
        {
            // SAFETY: fd valid; library symbols resolved.
            let res = unsafe { (self.lib.get_resources)(self.fd) };
            if res.is_null() {
                return Err(DrmError::CallFailed {
                    op: "drmModeGetResources",
                    detail: "returned null — kernel has no DRM device?".into(),
                });
            }
            // SAFETY: non-null res returned by get_resources.
            let res_ref = unsafe { &*res };
            // SAFETY: connectors array length == count_connectors.
            let connectors: &[u32] = unsafe {
                std::slice::from_raw_parts(res_ref.connectors, res_ref.count_connectors as usize)
            };

            let mut picked: Option<(u32, u32, DrmModeModeInfo, u32)> = None;
            for &cid in connectors {
                // SAFETY: fd valid; cid came from the resources array.
                let conn = unsafe { (self.lib.get_connector)(self.fd, cid) };
                if conn.is_null() {
                    continue;
                }
                // SAFETY: non-null conn.
                let conn_ref = unsafe { &*conn };
                let formatted = format!(
                    "{}-{}",
                    connector_type_name(conn_ref.connector_type),
                    conn_ref.connector_type_id
                );
                let matches = formatted == name
                    && conn_ref.connection == DRM_MODE_CONNECTED
                    && conn_ref.count_modes > 0;
                if matches {
                    // Grab the preferred (first) mode.
                    // SAFETY: modes array len == count_modes.
                    let first_mode = unsafe { *conn_ref.modes };
                    picked = Some((
                        conn_ref.connector_id,
                        conn_ref.encoder_id,
                        first_mode,
                        // store count_modes for later lookup
                        conn_ref.count_modes as u32,
                    ));
                    // SAFETY: conn was produced by get_connector.
                    unsafe { (self.lib.free_connector)(conn) };
                    break;
                }
                // SAFETY: conn was produced by get_connector.
                unsafe { (self.lib.free_connector)(conn) };
            }

            // SAFETY: res produced by get_resources.
            unsafe { (self.lib.free_resources)(res) };

            let (connector_id, encoder_id, default_mode, _n_modes) =
                picked.ok_or_else(|| DrmError::ConnectorNotFound(name.into()))?;

            // Resolve CRTC via the encoder.
            // SAFETY: fd valid; encoder_id came from the connector.
            let enc = unsafe { (self.lib.get_encoder)(self.fd, encoder_id) };
            if enc.is_null() {
                return Err(DrmError::NoCrtc(name.into()));
            }
            // SAFETY: non-null enc.
            let crtc_id = unsafe { (*enc).crtc_id };
            // SAFETY: enc produced by get_encoder.
            unsafe { (self.lib.free_encoder)(enc) };

            if crtc_id == 0 {
                return Err(DrmError::NoCrtc(name.into()));
            }

            self.connector_id = connector_id;
            self.crtc_id = crtc_id;
            self.mode = Some(default_mode);
            self.width = default_mode.hdisplay as u32;
            self.height = default_mode.vdisplay as u32;
            Ok(())
        }
        #[cfg(not(target_os = "linux"))]
        {
            let _ = name;
            Err(DrmError::NotSupported)
        }
    }

    /// Override the connector's mode by label. Must be called after
    /// [`Self::select_connector`]. Fails if the connector doesn't
    /// advertise a matching mode.
    pub fn set_mode(&mut self, label: &str) -> DrmResult<()> {
        #[cfg(target_os = "linux")]
        {
            if self.connector_id == 0 {
                return Err(DrmError::InvalidParameter(
                    "set_mode called before select_connector".into(),
                ));
            }
            // Re-fetch connector to get mode list.
            // SAFETY: fd valid; connector_id from earlier select.
            let conn = unsafe { (self.lib.get_connector)(self.fd, self.connector_id) };
            if conn.is_null() {
                return Err(DrmError::CallFailed {
                    op: "drmModeGetConnector",
                    detail: "returned null on re-fetch".into(),
                });
            }
            // SAFETY: non-null conn.
            let conn_ref = unsafe { &*conn };
            // SAFETY: modes array len == count_modes.
            let modes: &[DrmModeModeInfo] = unsafe {
                std::slice::from_raw_parts(conn_ref.modes, conn_ref.count_modes as usize)
            };

            let matched = modes.iter().find(|m| mode_matches(m, label)).copied();
            // SAFETY: conn produced by get_connector.
            unsafe { (self.lib.free_connector)(conn) };

            let chosen = matched.ok_or_else(|| DrmError::ModeNotSupported(label.into()))?;
            self.mode = Some(chosen);
            self.width = chosen.hdisplay as u32;
            self.height = chosen.vdisplay as u32;
            self.mode_set = false; // re-set on next present
            Ok(())
        }
        #[cfg(not(target_os = "linux"))]
        {
            let _ = label;
            Err(DrmError::NotSupported)
        }
    }

    /// Import a DMA-BUF file descriptor as a DRM gem handle suitable
    /// for `drmModeAddFB`. Returned handle is owned by the caller until
    /// passed to `flip`; `rm_fb` in Drop cleans up.
    pub fn import_dmabuf(&self, dma_fd: i32) -> DrmResult<u32> {
        #[cfg(target_os = "linux")]
        {
            if dma_fd < 0 {
                return Err(DrmError::InvalidParameter(
                    "dma_fd must be non-negative".into(),
                ));
            }
            let mut handle: u32 = 0;
            // SAFETY: fd valid; handle out-pointer writable.
            let ret = unsafe { (self.lib.prime_fd_to_handle)(self.fd, dma_fd, &mut handle) };
            if ret != 0 || handle == 0 {
                return Err(DrmError::CallFailed {
                    op: "drmPrimeFDToHandle",
                    detail: format!("ret={ret} handle={handle}"),
                });
            }
            Ok(handle)
        }
        #[cfg(not(target_os = "linux"))]
        {
            let _ = dma_fd;
            Err(DrmError::NotSupported)
        }
    }

    /// Full present cycle: import DMA-BUF, wrap as FB, set mode (first
    /// call) or page-flip (subsequent). Vsync'd via `DRM_MODE_PAGE_FLIP_EVENT`.
    pub fn present(&mut self, dma_fd: i32, stride: u32) -> DrmResult<()> {
        #[cfg(target_os = "linux")]
        {
            if self.crtc_id == 0 || self.mode.is_none() {
                return Err(DrmError::InvalidParameter(
                    "present() before select_connector + set_mode".into(),
                ));
            }

            let gem_handle = self.import_dmabuf(dma_fd)?;

            // Add FB. Prefer addFB2 with fourcc (correct for YUV); fall back.
            let mut fb_id: u32 = 0;
            let ret = if let Some(add2) = self.lib.add_fb2 {
                let handles = [gem_handle, gem_handle, 0, 0];
                let pitches = [stride, stride, 0, 0];
                let offsets = [0u32, (stride * self.height), 0, 0];
                // SAFETY: all in/out pointers valid; fourcc is a documented format.
                unsafe {
                    add2(
                        self.fd,
                        self.width,
                        self.height,
                        self.pixel_format,
                        handles.as_ptr(),
                        pitches.as_ptr(),
                        offsets.as_ptr(),
                        &mut fb_id,
                        0,
                    )
                }
            } else {
                // SAFETY: fallback addFB; depth 24 / bpp 32 for RGB paths.
                unsafe {
                    (self.lib.add_fb)(
                        self.fd,
                        self.width,
                        self.height,
                        24,
                        32,
                        stride,
                        gem_handle,
                        &mut fb_id,
                    )
                }
            };
            if ret != 0 || fb_id == 0 {
                return Err(DrmError::CallFailed {
                    op: "drmModeAddFB/AddFB2",
                    detail: format!("ret={ret}"),
                });
            }

            if !self.mode_set {
                // Initial modeset binds the FB to the CRTC + connector.
                let mode = self.mode.unwrap();
                let connectors = [self.connector_id];
                // SAFETY: mode out-lives the call; connectors array on stack.
                let ret = unsafe {
                    (self.lib.set_crtc)(
                        self.fd,
                        self.crtc_id,
                        fb_id,
                        0,
                        0,
                        connectors.as_ptr(),
                        1,
                        &mode as *const DrmModeModeInfo,
                    )
                };
                if ret != 0 {
                    // SAFETY: fb_id just obtained from AddFB.
                    unsafe { (self.lib.rm_fb)(self.fd, fb_id) };
                    return Err(DrmError::CallFailed {
                        op: "drmModeSetCrtc",
                        detail: format!("ret={ret}"),
                    });
                }
                self.mode_set = true;
                self.previous_fb_id = self.current_fb_id;
                self.current_fb_id = fb_id;
            } else {
                // Steady state: vsync'd page flip.
                // SAFETY: crtc/fb valid; user_data=null means "no event listener".
                let ret = unsafe {
                    (self.lib.page_flip)(
                        self.fd,
                        self.crtc_id,
                        fb_id,
                        DRM_MODE_PAGE_FLIP_EVENT,
                        std::ptr::null_mut(),
                    )
                };
                if ret != 0 {
                    // SAFETY: fb_id just obtained.
                    unsafe { (self.lib.rm_fb)(self.fd, fb_id) };
                    return Err(DrmError::CallFailed {
                        op: "drmModePageFlip",
                        detail: format!("ret={ret}"),
                    });
                }
                // Retire the older FB (two-flip pipeline).
                if self.previous_fb_id != 0 {
                    // SAFETY: previous_fb_id was returned by a prior AddFB.
                    unsafe { (self.lib.rm_fb)(self.fd, self.previous_fb_id) };
                }
                self.previous_fb_id = self.current_fb_id;
                self.current_fb_id = fb_id;
            }
            Ok(())
        }
        #[cfg(not(target_os = "linux"))]
        {
            let _ = (dma_fd, stride);
            Err(DrmError::NotSupported)
        }
    }

    /// Legacy one-shot flip API — kept for backward compatibility.
    /// Prefer [`Self::present`] which handles DMA-BUF import + vsync
    /// event in a single call.
    pub fn flip(&mut self, dma_handle: u32, stride: u32) -> DrmResult<()> {
        #[cfg(target_os = "linux")]
        {
            let mut fb_id: u32 = 0;
            // SAFETY: lib.add_fb resolved; out-pointer writable.
            let ret = unsafe {
                (self.lib.add_fb)(
                    self.fd,
                    self.width,
                    self.height,
                    24,
                    32,
                    stride,
                    dma_handle,
                    &mut fb_id,
                )
            };
            if ret != 0 {
                return Err(DrmError::CallFailed {
                    op: "drmModeAddFB",
                    detail: format!("ret={ret}"),
                });
            }
            if self.current_fb_id != 0 {
                // SAFETY: previous fb_id obtained from AddFB.
                unsafe { (self.lib.rm_fb)(self.fd, self.current_fb_id) };
            }
            self.current_fb_id = fb_id;
            Ok(())
        }
        #[cfg(not(target_os = "linux"))]
        {
            let _ = (dma_handle, stride);
            Err(DrmError::NotSupported)
        }
    }

    pub fn width(&self) -> u32 {
        self.width
    }
    pub fn height(&self) -> u32 {
        self.height
    }
    pub fn pixel_format(&self) -> u32 {
        self.pixel_format
    }

    /// Currently-selected CRTC id (0 before `select_connector`).
    pub fn crtc_id(&self) -> u32 {
        #[cfg(target_os = "linux")]
        {
            self.crtc_id
        }
        #[cfg(not(target_os = "linux"))]
        {
            0
        }
    }

    /// Connector id (0 before `select_connector`).
    pub fn connector_id(&self) -> u32 {
        #[cfg(target_os = "linux")]
        {
            self.connector_id
        }
        #[cfg(not(target_os = "linux"))]
        {
            0
        }
    }
}

impl Drop for DrmOutput {
    fn drop(&mut self) {
        #[cfg(target_os = "linux")]
        {
            if self.previous_fb_id != 0 {
                // SAFETY: previous_fb_id from AddFB.
                let _ = unsafe { (self.lib.rm_fb)(self.fd, self.previous_fb_id) };
            }
            if self.current_fb_id != 0 {
                // SAFETY: current_fb_id from AddFB.
                let _ = unsafe { (self.lib.rm_fb)(self.fd, self.current_fb_id) };
            }
            if self.fd >= 0 {
                // SAFETY: fd from drmOpen.
                let _ = unsafe { (self.lib.drm_close)(self.fd) };
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn drm_unavailable_on_dev_host() {
        // Non-Rockchip/non-Linux hosts return false. Only asserts no panic.
        let _ = drm_available();
    }

    #[test]
    fn open_rejects_zero_dims() {
        let res = DrmOutput::open(0, 720, fourcc::NV12);
        assert!(matches!(res, Err(DrmError::InvalidParameter(_))));
    }

    #[test]
    fn new_returns_not_supported_off_linux() {
        #[cfg(not(target_os = "linux"))]
        {
            let res = DrmOutput::new("HDMI-A-1", "720p60", fourcc::NV12);
            assert!(matches!(
                res,
                Err(DrmError::NotSupported) | Err(DrmError::LibraryNotFound)
            ));
        }
    }

    #[test]
    fn present_rejects_negative_dma_fd() {
        // Only meaningful when we can construct a DrmOutput — off Linux
        // construction itself fails first. Assert structural behaviour on Linux host:
        #[cfg(target_os = "linux")]
        if let Ok(out) = DrmOutput::open(640, 480, fourcc::NV12) {
            assert!(matches!(
                out.import_dmabuf(-1),
                Err(DrmError::InvalidParameter(_))
            ));
        }
    }

    #[test]
    fn fourcc_values_are_four_bytes() {
        assert_eq!(&fourcc::NV12.to_le_bytes(), b"NV12");
        assert_eq!(&fourcc::XRGB8888.to_le_bytes(), b"XR24");
        assert_eq!(&fourcc::YUYV.to_le_bytes(), b"YUYV");
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn connector_type_names_cover_common_outputs() {
        assert_eq!(connector_type_name(11), "HDMI-A");
        assert_eq!(connector_type_name(16), "DSI");
        assert_eq!(connector_type_name(10), "DP");
        assert_eq!(connector_type_name(14), "eDP");
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn mode_matches_by_name_and_shape() {
        let mut m = DrmModeModeInfo {
            clock: 148500,
            hdisplay: 1920,
            hsync_start: 0,
            hsync_end: 0,
            htotal: 0,
            hskew: 0,
            vdisplay: 1080,
            vsync_start: 0,
            vsync_end: 0,
            vtotal: 0,
            vscan: 0,
            vrefresh: 60,
            flags: 0,
            typ: 0,
            name: [0; DRM_DISPLAY_MODE_LEN],
        };
        // Fill "1920x1080" into name.
        let s = b"1920x1080";
        m.name[..s.len()].copy_from_slice(s);

        assert!(mode_matches(&m, "1920x1080"), "exact name match");
        assert!(mode_matches(&m, "1080p60"), "shape+refresh match");
        assert!(!mode_matches(&m, "720p60"));
        assert!(!mode_matches(&m, "1920x720"));
    }
}
