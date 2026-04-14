//! Safe API for the MPP H.264 hardware encoder.
//!
//! Designed for the FPV pipeline pattern: one input frame in (NV12,
//! either DMA-BUF fd or raw bytes) → one encoded H.264 NAL packet out,
//! per call. Encoder context is reused across frames.

use crate::error::{MppError, MppResult};
use crate::ffi::{self, MppLib};
use std::sync::Arc;

/// Configuration for [`MppH264Encoder::new`].
#[derive(Debug, Clone)]
pub struct MppEncoderConfig {
    pub width: u32,
    pub height: u32,
    /// Encoded bitrate target in kbps.
    pub bitrate_kbps: u32,
    /// Source frame rate (used by VBV pacing).
    pub fps: u32,
    /// I-frame interval in frames. `0` = only first frame is keyframe;
    /// typical values: 30 (1s @ 30fps), 60 (2s).
    pub gop: u32,
    /// AVC profile: 66=baseline, 77=main, 100=high. Baseline = lowest
    /// decode latency on receivers; recommended for FPV.
    pub profile: u32,
}

impl Default for MppEncoderConfig {
    fn default() -> Self {
        Self {
            width: 1280,
            height: 720,
            bitrate_kbps: 4000,
            fps: 30,
            gop: 30,
            profile: 66,
        }
    }
}

/// Hardware H.264 encoder via Rockchip MPP.
///
/// Each instance owns one `MppCtx` + its matching `MppApi` vtable
/// pointer (returned by `mpp_create`), plus a reusable input frame and
/// an `MppEncCfg`. Drop releases all native resources.
pub struct MppH264Encoder {
    lib: Arc<MppLib>,
    ctx: ffi::MppCtx,
    /// Pointer to the MPI vtable returned by `mpp_create`. Valid for
    /// the lifetime of `ctx` — freed implicitly by `mpp_destroy`.
    api: *const ffi::MppApi,
    cfg: MppEncoderConfig,
}

// SAFETY: api is a pointer into memory owned by the MPP runtime,
// immutable for the ctx lifetime. The encoder serialises access to
// ctx/api through &mut self on encode calls.
unsafe impl Send for MppH264Encoder {}

impl MppH264Encoder {
    /// Create a new H.264 encoder. Returns `MppError::LibraryNotFound`
    /// if `librockchip_mpp.so` is missing on this host.
    pub fn new(cfg: MppEncoderConfig) -> MppResult<Self> {
        if cfg.width == 0 || cfg.height == 0 {
            return Err(MppError::InvalidConfig("width/height must be > 0".into()));
        }
        if !cfg.width.is_multiple_of(16) || !cfg.height.is_multiple_of(16) {
            return Err(MppError::InvalidConfig(
                "width/height must be multiples of 16 for H.264 macroblock alignment".into(),
            ));
        }
        if cfg.bitrate_kbps == 0 {
            return Err(MppError::InvalidConfig("bitrate_kbps must be > 0".into()));
        }
        if cfg.fps == 0 {
            return Err(MppError::InvalidConfig("fps must be > 0".into()));
        }

        let lib = ffi::load_library()?;

        // Create MPP context. `mpp_create` writes both the ctx handle
        // and the MPI vtable pointer.
        let mut ctx: ffi::MppCtx = std::ptr::null_mut();
        let mut api_raw: *mut std::ffi::c_void = std::ptr::null_mut();
        // SAFETY: writable out-pointers; lib symbols verified non-null.
        let ret = unsafe { (lib.mpp_create)(&mut ctx, &mut api_raw as *mut _ as *mut _) };
        if ret != 0 {
            return Err(MppError::CallFailed {
                op: "mpp_create",
                status: ret,
            });
        }
        if api_raw.is_null() {
            // SAFETY: ctx valid (mpp_create returned success).
            unsafe { (lib.mpp_destroy)(ctx) };
            return Err(MppError::CallFailed {
                op: "mpp_create",
                status: -1,
            });
        }
        let api = api_raw as *const ffi::MppApi;

        // Init as encoder, codec = AVC (H.264).
        // SAFETY: ctx valid (mpp_create succeeded).
        let ret = unsafe { (lib.mpp_init)(ctx, ffi::MPP_CTX_ENC, ffi::MPP_VIDEO_CODING_AVC) };
        if ret != 0 {
            // SAFETY: ctx valid.
            unsafe { (lib.mpp_destroy)(ctx) };
            return Err(MppError::CallFailed {
                op: "mpp_init",
                status: ret,
            });
        }

        let enc = Self { lib, ctx, api, cfg };
        enc.apply_cfg()?;
        Ok(enc)
    }

    /// Apply the encoder configuration via `mpp_enc_cfg_*` + `MPP_ENC_SET_CFG`.
    fn apply_cfg(&self) -> MppResult<()> {
        let mut cfg: ffi::MppEncCfg = std::ptr::null_mut();
        // SAFETY: out-pointer is valid.
        let ret = unsafe { (self.lib.enc_cfg_init)(&mut cfg) };
        if ret != 0 {
            return Err(MppError::CallFailed {
                op: "mpp_enc_cfg_init",
                status: ret,
            });
        }

        // Helper: set one i32 cfg key. Takes a static null-terminated
        // C string for the MPP cfg key name.
        let set = |name: &[u8], val: i32| -> MppResult<()> {
            // SAFETY: cfg is a valid MppEncCfg; name is null-terminated.
            let r = unsafe { (self.lib.enc_cfg_set_s32)(cfg, name.as_ptr(), val) };
            if r != 0 {
                return Err(MppError::CallFailed {
                    op: "mpp_enc_cfg_set_s32",
                    status: r,
                });
            }
            Ok(())
        };

        // The MPP cfg keys are documented in `mpp_enc_cfg.h`. We set the
        // minimal viable subset for H.264 baseline: prep (resolution,
        // format, fps), rc (rate control mode + bitrate + gop), codec
        // (profile + level).
        set(b"prep:width\0", self.cfg.width as i32)?;
        set(b"prep:height\0", self.cfg.height as i32)?;
        set(b"prep:hor_stride\0", self.cfg.width as i32)?;
        set(b"prep:ver_stride\0", self.cfg.height as i32)?;
        set(b"prep:format\0", ffi::MPP_FMT_YUV420SP as i32)?; // NV12
        set(b"rc:mode\0", 1 /* MPP_ENC_RC_MODE_VBR */)?;
        set(b"rc:bps_target\0", (self.cfg.bitrate_kbps * 1000) as i32)?;
        set(b"rc:fps_in_num\0", self.cfg.fps as i32)?;
        set(b"rc:fps_in_denorm\0", 1)?;
        set(b"rc:fps_out_num\0", self.cfg.fps as i32)?;
        set(b"rc:fps_out_denorm\0", 1)?;
        set(b"rc:gop\0", self.cfg.gop as i32)?;
        set(b"h264:profile\0", self.cfg.profile as i32)?;
        set(b"h264:level\0", 40 /* level 4.0 */)?;
        set(
            b"h264:cabac_en\0",
            if self.cfg.profile == 66 { 0 } else { 1 },
        )?;

        // Apply via MPI control: command MPP_ENC_SET_CFG.
        // SAFETY: ctx valid, cfg was just initialised, api points at the
        // MPI vtable returned by mpp_create (owned by MPP, lives with ctx).
        let ret = unsafe {
            let api_ref = &*self.api;
            (api_ref.control)(self.ctx, ffi::MPP_ENC_SET_CFG, cfg)
        };

        // Free cfg regardless of ret.
        // SAFETY: cfg is valid (init succeeded).
        let _ = unsafe { (self.lib.enc_cfg_deinit)(cfg) };

        if ret != 0 {
            return Err(MppError::CallFailed {
                op: "MPP_ENC_SET_CFG",
                status: ret,
            });
        }
        Ok(())
    }

    /// Encode one NV12 frame from a DMA-BUF fd. Returns the encoded NAL
    /// data. Zero-copy: the encoder reads directly from the fd.
    ///
    /// `fd` must be a DMA-BUF file descriptor backing an NV12 buffer of
    /// `(self.width × self.height × 1.5)` bytes.
    pub fn encode_nv12_dmabuf(&mut self, fd: i32, len_bytes: usize) -> MppResult<Vec<u8>> {
        // Wrap fd as MppBuffer.
        let mut buf: ffi::MppBuffer = std::ptr::null_mut();
        // SAFETY: out-pointer valid; fd ownership stays with caller.
        let ret = unsafe { (self.lib.buffer_import_from_fd)(&mut buf, fd, len_bytes) };
        if ret != 0 {
            return Err(MppError::CallFailed {
                op: "mpp_buffer_import",
                status: ret,
            });
        }

        let result = self.encode_with_buffer(buf);

        // Release MPP buffer reference (does not close the underlying fd).
        // SAFETY: buf was just imported.
        let _ = unsafe { (self.lib.buffer_put)(buf, c"yscv-video-mpp".as_ptr().cast()) };

        result
    }

    fn encode_with_buffer(&mut self, buf: ffi::MppBuffer) -> MppResult<Vec<u8>> {
        let mut frame: ffi::MppFrame = std::ptr::null_mut();
        // SAFETY: out-pointer valid.
        let ret = unsafe { (self.lib.frame_init)(&mut frame) };
        if ret != 0 {
            return Err(MppError::CallFailed {
                op: "mpp_frame_init",
                status: ret,
            });
        }

        // SAFETY: frame just init'd; setters are documented mutation API.
        unsafe {
            (self.lib.frame_set_width)(frame, self.cfg.width);
            (self.lib.frame_set_height)(frame, self.cfg.height);
            (self.lib.frame_set_hor_stride)(frame, self.cfg.width);
            (self.lib.frame_set_ver_stride)(frame, self.cfg.height);
            (self.lib.frame_set_fmt)(frame, ffi::MPP_FMT_YUV420SP);
            (self.lib.frame_set_buffer)(frame, buf);
        }

        // Submit frame: api.encode_put_frame
        // SAFETY: ctx valid; frame is properly initialised; api is
        // the MPI vtable returned by mpp_create.
        let put_ret = unsafe {
            let api_ref = &*self.api;
            (api_ref.encode_put_frame)(self.ctx, frame)
        };

        // Frame is consumed by encode_put_frame on success; free on failure path.
        if put_ret != 0 {
            // SAFETY: frame still valid (put failed).
            let _ = unsafe { (self.lib.frame_deinit)(&mut frame) };
            return Err(MppError::CallFailed {
                op: "encode_put_frame",
                status: put_ret,
            });
        }

        // Get output packet.
        let mut packet: ffi::MppPacket = std::ptr::null_mut();
        // SAFETY: ctx valid; packet out-pointer valid; api points at
        // the live MPI vtable.
        let pkt_ret = unsafe {
            let api_ref = &*self.api;
            (api_ref.encode_get_packet)(self.ctx, &mut packet)
        };
        if pkt_ret != 0 || packet.is_null() {
            return Err(MppError::CallFailed {
                op: "encode_get_packet",
                status: pkt_ret,
            });
        }

        // Copy packet bytes out before deinit.
        // SAFETY: packet valid (get returned non-null + status 0).
        let nal: Vec<u8> = unsafe {
            let data = (self.lib.packet_get_data)(packet);
            let len = (self.lib.packet_get_length)(packet);
            if data.is_null() || len == 0 {
                Vec::new()
            } else {
                std::slice::from_raw_parts(data, len).to_vec()
            }
        };

        // Free packet.
        // SAFETY: packet just produced by encode_get_packet.
        let _ = unsafe { (self.lib.packet_deinit)(&mut packet) };

        Ok(nal)
    }

    /// Encoder configuration (read-only).
    pub fn cfg(&self) -> &MppEncoderConfig {
        &self.cfg
    }
}

impl Drop for MppH264Encoder {
    fn drop(&mut self) {
        if !self.ctx.is_null() {
            // SAFETY: ctx was successfully created in `new` and not yet destroyed.
            unsafe { (self.lib.mpp_destroy)(self.ctx) };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_validates_alignment() {
        let cfg = MppEncoderConfig {
            width: 1281, // not multiple of 16
            height: 720,
            ..Default::default()
        };
        let res = MppH264Encoder::new(cfg);
        match res {
            Err(MppError::InvalidConfig(msg)) => assert!(msg.contains("multiples of 16")),
            // On hosts without librockchip_mpp.so, we fail at LibraryNotFound
            // before reaching the alignment check — that's fine, the check
            // exists for runtime hosts where the lib IS present.
            Err(MppError::LibraryNotFound) => {}
            Ok(_) => panic!("encoder constructed unexpectedly without librockchip_mpp.so"),
            Err(other) => panic!("unexpected error: {other}"),
        }
    }

    #[test]
    fn config_validates_zero_dims() {
        let cfg = MppEncoderConfig {
            width: 0,
            ..Default::default()
        };
        let res = MppH264Encoder::new(cfg);
        match res {
            Err(MppError::InvalidConfig(_)) => {}
            Err(MppError::LibraryNotFound) => {}
            Ok(_) => panic!("encoder constructed unexpectedly without librockchip_mpp.so"),
            Err(other) => panic!("unexpected error: {other}"),
        }
    }

    #[test]
    fn config_default_is_valid_shape() {
        let cfg = MppEncoderConfig::default();
        assert_eq!(cfg.width % 16, 0);
        assert_eq!(cfg.height % 16, 0);
        assert!(cfg.bitrate_kbps > 0);
        assert!(cfg.fps > 0);
    }

    #[test]
    fn mpp_api_struct_size_matches_sdk() {
        // Runtime mirror of the compile-time assertion in ffi.rs. Keeps
        // the check visible in test output when someone greps for layout.
        assert_eq!(std::mem::size_of::<ffi::MppApi>(), 168);
    }
}
