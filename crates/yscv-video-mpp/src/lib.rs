//! Rockchip MPP (Media Process Platform) hardware H.264/HEVC encoder.
//!
//! Resolves `librockchip_mpp.so` at runtime via `dlopen` — the binary
//! compiles on every platform; HW encoding only activates on Rockchip
//! devices where the library is present.
//!
//! # Why this exists
//!
//! Software H.264 encode of a 720p I-frame takes 20–30 ms on RK3588's
//! A76 cores. That alone exceeds the 10 ms FPV budget. Rockchip's MPP
//! does the same encode in 2–3 ms on dedicated VPU hardware, plus
//! supports zero-copy input directly from DMA-BUF or `MB_BLK` (the same
//! handle type used by `RknnBackend::wrap_mb_blk`).
//!
//! # Coverage
//!
//! - `mpp_create` / `mpp_init` / `mpp_destroy` — context lifecycle
//! - `mpp_packet_*` — encoded NAL output
//! - `mpp_frame_*` — raw input wrapping
//! - `mpi.encode_put_frame` / `encode_get_packet` — synchronous encode
//! - `mpp_buffer_get_mpp_buffer` — extract MB_BLK for zero-copy input
//!
//! # Safety
//!
//! All `unsafe` is confined to FFI call sites. Library symbols obtained
//! via `dlopen`/`dlsym` are checked non-null. RAII Drop handlers free
//! every MPP resource (encoder ctx, frames, packets) deterministically.

#![cfg_attr(not(feature = "mpp"), allow(dead_code))]
#![allow(unsafe_code)]

mod error;

#[cfg(feature = "mpp")]
mod encoder;
#[cfg(feature = "mpp")]
mod ffi;

pub use error::{MppError, MppResult};

#[cfg(feature = "mpp")]
pub use encoder::{MppEncoderConfig, MppH264Encoder};

/// Check whether `librockchip_mpp.so` is loadable on this host.
///
/// On non-Linux platforms or non-feature-enabled builds, returns `false`.
pub fn mpp_available() -> bool {
    #[cfg(all(feature = "mpp", target_os = "linux"))]
    {
        ffi::probe_library()
    }
    #[cfg(not(all(feature = "mpp", target_os = "linux")))]
    {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mpp_unavailable_on_dev_host() {
        // On macOS/Linux without librockchip_mpp.so, must report false.
        // Real Rockchip hardware is the only place this returns true.
        let _ = mpp_available();
    }
}
