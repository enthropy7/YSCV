//! Linux framebuffer output for direct rendering to `/dev/fb0`.
//!
//! Opens the framebuffer device, queries geometry via `ioctl`, maps the
//! framebuffer memory, and provides `write_rgb8` for blitting RGB8 frames.
//! Supports RGB565, RGB888, and ARGB8888 pixel formats.

#[cfg(target_os = "linux")]
mod linux_impl {
    use crate::VideoError;

    // -----------------------------------------------------------------------
    // C FFI — minimal declarations, zero deps
    // -----------------------------------------------------------------------

    unsafe extern "C" {
        fn open(path: *const u8, flags: i32) -> i32;
        fn close(fd: i32) -> i32;
        fn ioctl(fd: i32, request: u64, ...) -> i32;
        fn mmap(addr: *mut u8, len: usize, prot: i32, flags: i32, fd: i32, offset: i64) -> *mut u8;
        fn munmap(addr: *mut u8, len: usize) -> i32;
    }

    const O_RDWR: i32 = 2;
    const PROT_READ: i32 = 0x1;
    const PROT_WRITE: i32 = 0x2;
    const MAP_SHARED: i32 = 0x01;
    const MAP_FAILED: *mut u8 = !0usize as *mut u8;

    // ioctl numbers for fbdev (from <linux/fb.h>)
    const FBIOGET_VSCREENINFO: u64 = 0x4600;
    const FBIOGET_FSCREENINFO: u64 = 0x4602;

    // -----------------------------------------------------------------------
    // fbdev kernel ABI structs
    // -----------------------------------------------------------------------

    /// `struct fb_var_screeninfo` — only the fields we need.
    #[repr(C)]
    #[derive(Default)]
    struct FbVarScreenInfo {
        xres: u32,
        yres: u32,
        xres_virtual: u32,
        yres_virtual: u32,
        xoffset: u32,
        yoffset: u32,
        bits_per_pixel: u32,
        grayscale: u32,
        red: FbBitfield,
        green: FbBitfield,
        blue: FbBitfield,
        transp: FbBitfield,
        nonstd: u32,
        activate: u32,
        height: u32,
        width: u32,
        accel_flags: u32,
        pixclock: u32,
        left_margin: u32,
        right_margin: u32,
        upper_margin: u32,
        lower_margin: u32,
        hsync_len: u32,
        vsync_len: u32,
        sync: u32,
        vmode: u32,
        rotate: u32,
        colorspace: u32,
        reserved: [u32; 4],
    }

    #[repr(C)]
    #[derive(Default)]
    struct FbBitfield {
        offset: u32,
        length: u32,
        msb_right: u32,
    }

    /// `struct fb_fix_screeninfo` — only the fields we need.
    #[repr(C)]
    #[derive(Default)]
    struct FbFixScreenInfo {
        id: [u8; 16],
        smem_start: u64,
        smem_len: u32,
        fb_type: u32,
        type_aux: u32,
        visual: u32,
        xpanstep: u16,
        ypanstep: u16,
        ywrapstep: u16,
        _pad: u16,
        line_length: u32,
        mmio_start: u64,
        mmio_len: u32,
        accel: u32,
        capabilities: u16,
        reserved: [u16; 2],
    }

    // -----------------------------------------------------------------------
    // LinuxFramebuffer
    // -----------------------------------------------------------------------

    /// Direct access to a Linux framebuffer device (`/dev/fbN`).
    ///
    /// Opened via `ioctl` for geometry, `mmap` for pixel data. Supports
    /// RGB565 (16bpp), RGB888 (24bpp), and ARGB8888 (32bpp) modes.
    pub struct LinuxFramebuffer {
        fd: i32,
        mmap_ptr: *mut u8,
        mmap_len: usize,
        width: u32,
        height: u32,
        stride: u32,
        bpp: u8,
    }

    // Safety: the mmap pointer is only accessed through &self/&mut self,
    // and the fd is not shared across threads.
    unsafe impl Send for LinuxFramebuffer {}

    impl LinuxFramebuffer {
        /// Open a framebuffer device (e.g., `/dev/fb0`).
        pub fn open(device: &str) -> Result<Self, VideoError> {
            let mut path_buf = Vec::with_capacity(device.len() + 1);
            path_buf.extend_from_slice(device.as_bytes());
            path_buf.push(0); // null terminator

            let fd = unsafe { open(path_buf.as_ptr(), O_RDWR) };
            if fd < 0 {
                return Err(VideoError::Codec(format!(
                    "failed to open framebuffer device: {device}"
                )));
            }

            // Query variable screen info
            let mut vinfo = FbVarScreenInfo::default();
            let ret = unsafe { ioctl(fd, FBIOGET_VSCREENINFO, &mut vinfo as *mut FbVarScreenInfo) };
            if ret < 0 {
                unsafe {
                    close(fd);
                }
                return Err(VideoError::Codec("FBIOGET_VSCREENINFO ioctl failed".into()));
            }

            // Query fixed screen info
            let mut finfo = FbFixScreenInfo::default();
            let ret = unsafe { ioctl(fd, FBIOGET_FSCREENINFO, &mut finfo as *mut FbFixScreenInfo) };
            if ret < 0 {
                unsafe {
                    close(fd);
                }
                return Err(VideoError::Codec("FBIOGET_FSCREENINFO ioctl failed".into()));
            }

            let bpp = vinfo.bits_per_pixel as u8;
            if !matches!(bpp, 16 | 24 | 32) {
                unsafe {
                    close(fd);
                }
                return Err(VideoError::Codec(format!(
                    "unsupported framebuffer bpp: {bpp} (expected 16, 24, or 32)"
                )));
            }

            let mmap_len = finfo.smem_len as usize;
            let ptr = unsafe {
                mmap(
                    std::ptr::null_mut(),
                    mmap_len,
                    PROT_READ | PROT_WRITE,
                    MAP_SHARED,
                    fd,
                    0,
                )
            };
            if ptr == MAP_FAILED {
                unsafe {
                    close(fd);
                }
                return Err(VideoError::Codec("mmap of framebuffer failed".into()));
            }

            Ok(Self {
                fd,
                mmap_ptr: ptr,
                mmap_len,
                width: vinfo.xres,
                height: vinfo.yres,
                stride: finfo.line_length,
                bpp,
            })
        }

        /// Framebuffer display width in pixels.
        pub fn width(&self) -> u32 {
            self.width
        }

        /// Framebuffer display height in pixels.
        pub fn height(&self) -> u32 {
            self.height
        }

        /// Bits per pixel (16, 24, or 32).
        pub fn bpp(&self) -> u8 {
            self.bpp
        }

        /// Write an RGB8 frame to the framebuffer, converting pixel format as needed.
        ///
        /// The source frame is clipped to the framebuffer dimensions. If the frame
        /// is smaller than the display, only the top-left portion is written.
        pub fn write_rgb8(&self, frame: &[u8], frame_width: usize, frame_height: usize) {
            let copy_w = (frame_width as u32).min(self.width) as usize;
            let copy_h = (frame_height as u32).min(self.height) as usize;

            match self.bpp {
                16 => self.write_rgb565(frame, frame_width, copy_w, copy_h),
                24 => self.write_rgb888(frame, frame_width, copy_w, copy_h),
                32 => self.write_argb8888(frame, frame_width, copy_w, copy_h),
                _ => {} // already validated in open()
            }
        }

        fn write_rgb565(&self, frame: &[u8], src_stride: usize, copy_w: usize, copy_h: usize) {
            let fb = unsafe { std::slice::from_raw_parts_mut(self.mmap_ptr, self.mmap_len) };
            let dst_stride = self.stride as usize;

            for y in 0..copy_h {
                let src_row = y * src_stride * 3;
                let dst_row = y * dst_stride;

                for x in 0..copy_w {
                    let si = src_row + x * 3;
                    let r = frame[si] as u16;
                    let g = frame[si + 1] as u16;
                    let b = frame[si + 2] as u16;
                    // RGB565: RRRRRGGG GGGBBBBB
                    let pixel = ((r >> 3) << 11) | ((g >> 2) << 5) | (b >> 3);
                    let di = dst_row + x * 2;
                    if di + 1 < fb.len() {
                        fb[di] = pixel as u8;
                        fb[di + 1] = (pixel >> 8) as u8;
                    }
                }
            }
        }

        fn write_rgb888(&self, frame: &[u8], src_stride: usize, copy_w: usize, copy_h: usize) {
            let fb = unsafe { std::slice::from_raw_parts_mut(self.mmap_ptr, self.mmap_len) };
            let dst_stride = self.stride as usize;

            for y in 0..copy_h {
                let src_start = y * src_stride * 3;
                let dst_start = y * dst_stride;
                let count = copy_w * 3;
                let src_end = src_start + count;
                let dst_end = dst_start + count;
                if src_end <= frame.len() && dst_end <= fb.len() {
                    fb[dst_start..dst_end].copy_from_slice(&frame[src_start..src_end]);
                }
            }
        }

        fn write_argb8888(&self, frame: &[u8], src_stride: usize, copy_w: usize, copy_h: usize) {
            let fb = unsafe { std::slice::from_raw_parts_mut(self.mmap_ptr, self.mmap_len) };
            let dst_stride = self.stride as usize;

            for y in 0..copy_h {
                let src_row = y * src_stride * 3;
                let dst_row = y * dst_stride;

                for x in 0..copy_w {
                    let si = src_row + x * 3;
                    let di = dst_row + x * 4;
                    if si + 2 < frame.len() && di + 3 < fb.len() {
                        fb[di] = frame[si + 2]; // B
                        fb[di + 1] = frame[si + 1]; // G
                        fb[di + 2] = frame[si]; // R
                        fb[di + 3] = 0xFF; // A (opaque)
                    }
                }
            }
        }
    }

    impl Drop for LinuxFramebuffer {
        fn drop(&mut self) {
            unsafe {
                if !self.mmap_ptr.is_null() && self.mmap_ptr != MAP_FAILED {
                    munmap(self.mmap_ptr, self.mmap_len);
                }
                if self.fd >= 0 {
                    close(self.fd);
                }
            }
        }
    }
}

#[cfg(target_os = "linux")]
pub use linux_impl::LinuxFramebuffer;

// ---------------------------------------------------------------------------
// Tests (compile-time structure validation only — actual fbdev tests need hw)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    #[test]
    fn framebuffer_module_compiles() {
        // Structural test: ensure the module compiles on all platforms.
        // Actual LinuxFramebuffer functionality requires a real /dev/fbN device.
        #[cfg(target_os = "linux")]
        {
            // Verify struct layout constraints
            assert!(std::mem::size_of::<super::linux_impl::LinuxFramebuffer>() > 0);
        }
    }

    #[test]
    #[allow(clippy::identity_op)]
    // The `0u16 >> N` / `<< N` shifts look like identity ops but are the whole
    // point of the test: they document the RGB565 bit layout for each
    // channel. Collapsing them would defeat the didactic value.
    fn rgb565_conversion_correctness() {
        // Verify RGB565 packing logic: R=255,G=255,B=255 -> 0xFFFF
        let r: u16 = 255;
        let g: u16 = 255;
        let b: u16 = 255;
        let pixel = ((r >> 3) << 11) | ((g >> 2) << 5) | (b >> 3);
        assert_eq!(pixel, 0xFFFF, "white should produce 0xFFFF in RGB565");

        // R=255,G=0,B=0 -> 0xF800
        let pixel_red = ((255u16 >> 3) << 11) | ((0u16 >> 2) << 5) | (0u16 >> 3);
        assert_eq!(
            pixel_red, 0xF800,
            "pure red should produce 0xF800 in RGB565"
        );

        // R=0,G=255,B=0 -> 0x07E0
        let pixel_green = ((0u16 >> 3) << 11) | ((255u16 >> 2) << 5) | (0u16 >> 3);
        assert_eq!(
            pixel_green, 0x07E0,
            "pure green should produce 0x07E0 in RGB565"
        );

        // R=0,G=0,B=255 -> 0x001F
        let pixel_blue = ((0u16 >> 3) << 11) | ((0u16 >> 2) << 5) | (255u16 >> 3);
        assert_eq!(
            pixel_blue, 0x001F,
            "pure blue should produce 0x001F in RGB565"
        );
    }
}
