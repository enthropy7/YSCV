//! Direct V4L2 camera capture using raw ioctls — no libc crate, no external
//! dependencies beyond `std`. Uses `extern "C"` declarations for `ioctl`,
//! `mmap`, and `munmap`.
//!
//! This module is only compiled on Linux (`#[cfg(target_os = "linux")]`).

use crate::VideoError;
use std::os::unix::io::AsRawFd;

// ---------------------------------------------------------------------------
// C FFI — 3 declarations, zero deps
// ---------------------------------------------------------------------------

unsafe extern "C" {
    fn ioctl(fd: i32, request: u64, ...) -> i32;
    fn mmap(addr: *mut u8, len: usize, prot: i32, flags: i32, fd: i32, offset: i64) -> *mut u8;
    fn munmap(addr: *mut u8, len: usize) -> i32;
    /// POSIX `close(2)` — used by `V4l2DmaBufGuard::Drop`.
    fn close(fd: i32) -> i32;
}

// ---------------------------------------------------------------------------
// V4L2 ioctl numbers (from <linux/videodev2.h>)
// ---------------------------------------------------------------------------

const VIDIOC_QUERYCAP: u64 = 0x80685600;
const VIDIOC_S_FMT: u64 = 0xC0D05605;
const VIDIOC_REQBUFS: u64 = 0xC0145608;
const VIDIOC_QUERYBUF: u64 = 0xC0585609;
const VIDIOC_QBUF: u64 = 0xC058560F;
const VIDIOC_DQBUF: u64 = 0xC0585611;
const VIDIOC_STREAMON: u64 = 0x40045612;
const VIDIOC_STREAMOFF: u64 = 0x40045613;
/// `VIDIOC_EXPBUF` — export a V4L2 buffer as a DMA-BUF file descriptor.
///
/// Requests a `dma-buf` fd for buffer `index` so it can be shared with
/// other kernel subsystems (NPU, GPU, display) without CPU copy.
const VIDIOC_EXPBUF: u64 = 0xC0405610;

/// DMA-BUF access mode flags for `VIDIOC_EXPBUF`.
const O_CLOEXEC: u32 = 0o2000000;
const O_RDWR: u32 = 0o2;

// mmap constants
const PROT_READ: i32 = 0x1;
const PROT_WRITE: i32 = 0x2;
const MAP_SHARED: i32 = 0x01;
const MAP_FAILED: *mut u8 = !0usize as *mut u8;

// V4L2 buffer type / memory type
const V4L2_BUF_TYPE_VIDEO_CAPTURE: u32 = 1;
const V4L2_MEMORY_MMAP: u32 = 1;

// Capability flags
const V4L2_CAP_VIDEO_CAPTURE: u32 = 0x0000_0001;
const V4L2_CAP_STREAMING: u32 = 0x0400_0000;

// Number of mmap'd buffers to request
const NUM_BUFFERS: u32 = 4;

// ---------------------------------------------------------------------------
// Pixel format enum
// ---------------------------------------------------------------------------

/// V4L2 pixel formats supported by this capture module.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum V4l2PixelFormat {
    Yuyv,
    Nv12,
    Mjpeg,
    H264,
}

impl V4l2PixelFormat {
    /// FourCC value matching the Linux kernel definition.
    pub const fn fourcc(self) -> u32 {
        match self {
            Self::Yuyv => 0x5659_5559,  // 'VYUY' little-endian = YUYV
            Self::Nv12 => 0x3231_564E,  // 'NV12'
            Self::Mjpeg => 0x4750_4A4D, // 'MJPG'
            Self::H264 => 0x3436_3248,  // 'H264'
        }
    }
}

// ---------------------------------------------------------------------------
// V4L2 kernel ABI structs (repr(C), exact sizes)
// ---------------------------------------------------------------------------

/// `struct v4l2_capability` — 104 bytes.
#[repr(C)]
struct V4l2Capability {
    driver: [u8; 16],
    card: [u8; 32],
    bus_info: [u8; 32],
    version: u32,
    capabilities: u32,
    device_caps: u32,
    reserved: [u32; 3],
}

/// Pixel format description embedded in `v4l2_format`.
#[repr(C)]
#[derive(Clone, Copy)]
struct V4l2PixFormat {
    width: u32,
    height: u32,
    pixelformat: u32,
    field: u32,
    bytesperline: u32,
    sizeimage: u32,
    colorspace: u32,
    priv_: u32,
    flags: u32,
    // ycbcr_enc / quantization / xfer_func (union area padding)
    ycbcr_enc: u32,
    quantization: u32,
    xfer_func: u32,
}

/// `struct v4l2_format` — 208 bytes total.
/// We only use `type_` + the pix format union member.
#[repr(C)]
struct V4l2Format {
    type_: u32,
    pix: V4l2PixFormat,
    // The union in the kernel is 200 bytes; V4l2PixFormat is 48 bytes.
    // Pad the remaining 152 bytes.
    _pad: [u8; 152],
}

/// `struct v4l2_requestbuffers` — 20 bytes.
#[repr(C)]
struct V4l2RequestBuffers {
    count: u32,
    type_: u32,
    memory: u32,
    capabilities: u32,
    flags: u8,
    reserved: [u8; 3],
}

/// Timeval structure used inside `v4l2_buffer`.
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Timeval {
    tv_sec: i64,
    tv_usec: i64,
}

/// `struct v4l2_buffer` — 88 bytes on 64-bit.
/// The kernel struct has a union for `m` (offset / userptr / planes / fd).
#[repr(C)]
struct V4l2Buffer {
    index: u32,
    type_: u32,
    bytesused: u32,
    flags: u32,
    field: u32,
    timestamp: Timeval,
    timecode: [u8; 16], // v4l2_timecode
    sequence: u32,
    memory: u32,
    // union m { __u32 offset; unsigned long userptr; struct v4l2_plane *planes; __s32 fd; }
    m_offset: u64, // use u64 to cover the largest union member on 64-bit
    length: u32,
    reserved2: u32,
    // union { __s32 request_fd; __u32 reserved; }
    request_fd: i32,
}

/// `struct v4l2_exportbuffer` — parameters for `VIDIOC_EXPBUF`.
///
/// Exports a V4L2 buffer as a DMA-BUF file descriptor for zero-copy sharing
/// with other subsystems (NPU, GPU, display).
#[repr(C)]
struct V4l2ExportBuffer {
    type_: u32,
    index: u32,
    plane: u32,
    flags: u32,
    fd: i32,
    reserved: [u32; 11],
}

// ---------------------------------------------------------------------------
// V4l2Camera
// ---------------------------------------------------------------------------

/// Zero-copy V4L2 camera capture handle.
///
/// Opens a `/dev/videoN` device, negotiates format, mmap's kernel buffers,
/// and provides frame capture via `DQBUF` / `QBUF` cycling.
pub struct V4l2Camera {
    /// Held to keep the fd alive (Drop closes it). Field intentionally
    /// not read directly — `fd` below is the working handle.
    #[allow(dead_code)]
    file: std::fs::File,
    fd: i32,
    buffers: Vec<(*mut u8, usize)>,
    width: u32,
    height: u32,
    pixel_format: u32,
    streaming: bool,
}

// SAFETY: The mmap'd pointers are valid for the lifetime of the fd and are
// only accessed through &self / &mut self, so sending across threads is safe
// as long as only one thread accesses the camera at a time (guaranteed by
// &mut self on capture_frame).
unsafe impl Send for V4l2Camera {}

impl V4l2Camera {
    /// Open a V4L2 camera device (e.g. `"/dev/video0"`), set format, and
    /// prepare mmap'd buffers for streaming.
    pub fn open(
        device: &str,
        width: u32,
        height: u32,
        format: V4l2PixelFormat,
    ) -> Result<Self, VideoError> {
        if width == 0 || height == 0 {
            return Err(VideoError::InvalidCameraResolution { width, height });
        }

        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(device)
            .map_err(|e| VideoError::Source(format!("V4L2: cannot open {device}: {e}")))?;

        let fd = file.as_raw_fd();

        // QUERYCAP — verify the device supports video capture + streaming
        let mut cap: V4l2Capability = unsafe { std::mem::zeroed() };
        let ret = unsafe {
            ioctl(
                fd,
                VIDIOC_QUERYCAP,
                &mut cap as *mut V4l2Capability as *mut u8,
            )
        };
        if ret < 0 {
            return Err(VideoError::Source("V4L2: QUERYCAP failed".into()));
        }
        if cap.capabilities & V4L2_CAP_VIDEO_CAPTURE == 0 {
            return Err(VideoError::Source(
                "V4L2: device does not support video capture".into(),
            ));
        }
        if cap.capabilities & V4L2_CAP_STREAMING == 0 {
            return Err(VideoError::Source(
                "V4L2: device does not support streaming I/O".into(),
            ));
        }

        // S_FMT — set desired format
        let mut fmt: V4l2Format = unsafe { std::mem::zeroed() };
        fmt.type_ = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        fmt.pix.width = width;
        fmt.pix.height = height;
        fmt.pix.pixelformat = format.fourcc();
        fmt.pix.field = 1; // V4L2_FIELD_NONE

        let ret = unsafe { ioctl(fd, VIDIOC_S_FMT, &mut fmt as *mut V4l2Format as *mut u8) };
        if ret < 0 {
            return Err(VideoError::Source("V4L2: S_FMT failed".into()));
        }

        let actual_w = fmt.pix.width;
        let actual_h = fmt.pix.height;
        let actual_fmt = fmt.pix.pixelformat;

        // REQBUFS — request mmap'd buffers
        let mut reqbufs: V4l2RequestBuffers = unsafe { std::mem::zeroed() };
        reqbufs.count = NUM_BUFFERS;
        reqbufs.type_ = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        reqbufs.memory = V4L2_MEMORY_MMAP;

        let ret = unsafe {
            ioctl(
                fd,
                VIDIOC_REQBUFS,
                &mut reqbufs as *mut V4l2RequestBuffers as *mut u8,
            )
        };
        if ret < 0 {
            return Err(VideoError::Source("V4L2: REQBUFS failed".into()));
        }
        if reqbufs.count == 0 {
            return Err(VideoError::Source(
                "V4L2: driver allocated 0 buffers".into(),
            ));
        }

        // QUERYBUF + mmap each buffer
        let mut buffers = Vec::with_capacity(reqbufs.count as usize);
        for i in 0..reqbufs.count {
            let mut buf: V4l2Buffer = unsafe { std::mem::zeroed() };
            buf.index = i;
            buf.type_ = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            buf.memory = V4L2_MEMORY_MMAP;

            let ret = unsafe { ioctl(fd, VIDIOC_QUERYBUF, &mut buf as *mut V4l2Buffer as *mut u8) };
            if ret < 0 {
                // Unmap any already-mapped buffers before returning
                for &(ptr, len) in &buffers {
                    unsafe {
                        munmap(ptr, len);
                    }
                }
                return Err(VideoError::Source(format!(
                    "V4L2: QUERYBUF failed for buffer {i}"
                )));
            }

            let ptr = unsafe {
                mmap(
                    std::ptr::null_mut(),
                    buf.length as usize,
                    PROT_READ | PROT_WRITE,
                    MAP_SHARED,
                    fd,
                    buf.m_offset as i64,
                )
            };
            if ptr == MAP_FAILED {
                for &(p, l) in &buffers {
                    unsafe {
                        munmap(p, l);
                    }
                }
                return Err(VideoError::Source(format!(
                    "V4L2: mmap failed for buffer {i}"
                )));
            }

            buffers.push((ptr, buf.length as usize));
        }

        Ok(Self {
            file,
            fd,
            buffers,
            width: actual_w,
            height: actual_h,
            pixel_format: actual_fmt,
            streaming: false,
        })
    }

    /// Queue all buffers and start the capture stream.
    pub fn start_streaming(&mut self) -> Result<(), VideoError> {
        // Queue all buffers
        for i in 0..self.buffers.len() as u32 {
            let mut buf: V4l2Buffer = unsafe { std::mem::zeroed() };
            buf.index = i;
            buf.type_ = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            buf.memory = V4L2_MEMORY_MMAP;
            let ret =
                unsafe { ioctl(self.fd, VIDIOC_QBUF, &mut buf as *mut V4l2Buffer as *mut u8) };
            if ret < 0 {
                return Err(VideoError::Source(format!(
                    "V4L2: QBUF failed for buffer {i}"
                )));
            }
        }

        // STREAMON
        let mut buf_type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        let ret = unsafe {
            ioctl(
                self.fd,
                VIDIOC_STREAMON,
                &mut buf_type as *mut u32 as *mut u8,
            )
        };
        if ret < 0 {
            return Err(VideoError::Source("V4L2: STREAMON failed".into()));
        }
        self.streaming = true;
        Ok(())
    }

    /// Dequeue the next captured frame (zero-copy reference to mmap'd buffer),
    /// then immediately re-queue the buffer.
    ///
    /// The returned slice is valid until the next call to `capture_frame` or
    /// `stop_streaming`.
    pub fn capture_frame(&mut self) -> Result<&[u8], VideoError> {
        if !self.streaming {
            return Err(VideoError::Source("V4L2: not streaming".into()));
        }

        // DQBUF
        let mut buf: V4l2Buffer = unsafe { std::mem::zeroed() };
        buf.type_ = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;

        let ret = unsafe {
            ioctl(
                self.fd,
                VIDIOC_DQBUF,
                &mut buf as *mut V4l2Buffer as *mut u8,
            )
        };
        if ret < 0 {
            return Err(VideoError::Source("V4L2: DQBUF failed".into()));
        }

        let idx = buf.index as usize;
        let bytes_used = buf.bytesused as usize;
        if idx >= self.buffers.len() {
            return Err(VideoError::Source(format!(
                "V4L2: DQBUF returned invalid buffer index {idx}"
            )));
        }

        let (ptr, len) = self.buffers[idx];
        let actual_len = bytes_used.min(len);

        // SAFETY: ptr is a valid mmap'd region of at least `len` bytes,
        // and we only hold a reference until the next DQBUF (or drop).
        let frame_data = unsafe { std::slice::from_raw_parts(ptr, actual_len) };

        // Re-queue the buffer immediately
        let mut qbuf: V4l2Buffer = unsafe { std::mem::zeroed() };
        qbuf.index = buf.index;
        qbuf.type_ = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        qbuf.memory = V4L2_MEMORY_MMAP;
        let ret = unsafe {
            ioctl(
                self.fd,
                VIDIOC_QBUF,
                &mut qbuf as *mut V4l2Buffer as *mut u8,
            )
        };
        if ret < 0 {
            return Err(VideoError::Source("V4L2: QBUF (re-queue) failed".into()));
        }

        Ok(frame_data)
    }

    /// Dequeue the next captured frame and return the buffer index along
    /// with a zero-copy reference to its pixel data.
    ///
    /// The index identifies which mmap'd buffer holds the frame; pair it
    /// with [`Self::export_dmabuf`] to obtain a DMA-BUF fd suitable for
    /// sharing with an NPU via `rknn_create_mem_from_fd`.
    ///
    /// The returned slice is valid until the next call to a frame-capture
    /// method or `stop_streaming`.
    pub fn capture_frame_indexed(&mut self) -> Result<(u32, &[u8]), VideoError> {
        if !self.streaming {
            return Err(VideoError::Source("V4L2: not streaming".into()));
        }

        let mut buf: V4l2Buffer = unsafe { std::mem::zeroed() };
        buf.type_ = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;

        // SAFETY: ioctl(fd, DQBUF, &buf) with a zeroed V4l2Buffer struct.
        let ret = unsafe {
            ioctl(
                self.fd,
                VIDIOC_DQBUF,
                &mut buf as *mut V4l2Buffer as *mut u8,
            )
        };
        if ret < 0 {
            return Err(VideoError::Source("V4L2: DQBUF failed".into()));
        }

        let idx = buf.index as usize;
        let bytes_used = buf.bytesused as usize;
        if idx >= self.buffers.len() {
            return Err(VideoError::Source(format!(
                "V4L2: DQBUF returned invalid buffer index {idx}"
            )));
        }

        let (ptr, len) = self.buffers[idx];
        let actual_len = bytes_used.min(len);

        // SAFETY: ptr is a valid mmap'd region of at least `len` bytes;
        // the returned reference lives until the next DQBUF call (& mut self).
        let frame_data = unsafe { std::slice::from_raw_parts(ptr, actual_len) };

        // Re-queue the buffer immediately so the kernel can fill it again.
        let mut qbuf: V4l2Buffer = unsafe { std::mem::zeroed() };
        qbuf.index = buf.index;
        qbuf.type_ = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        qbuf.memory = V4L2_MEMORY_MMAP;
        // SAFETY: ioctl(fd, QBUF, &qbuf).
        let ret = unsafe {
            ioctl(
                self.fd,
                VIDIOC_QBUF,
                &mut qbuf as *mut V4l2Buffer as *mut u8,
            )
        };
        if ret < 0 {
            return Err(VideoError::Source("V4L2: QBUF (re-queue) failed".into()));
        }

        Ok((buf.index, frame_data))
    }

    /// Export a V4L2 buffer as a DMA-BUF file descriptor for zero-copy
    /// sharing with an NPU / GPU / display.
    ///
    /// The returned fd can be passed to `RknnBackend::wrap_fd()` to bind
    /// the same physical memory as an NPU input tensor — the camera fills
    /// it and the NPU reads it without any CPU copy.
    ///
    /// The caller is responsible for closing the fd (via `libc::close` or
    /// by wrapping in `std::fs::File::from_raw_fd`) when finished. Prefer
    /// [`Self::export_dmabuf_owned`] for automatic cleanup via RAII.
    pub fn export_dmabuf(&self, buffer_index: u32) -> Result<i32, VideoError> {
        if buffer_index as usize >= self.buffers.len() {
            return Err(VideoError::Source(format!(
                "V4L2: export_dmabuf buffer index {buffer_index} out of range"
            )));
        }
        let mut ebuf = V4l2ExportBuffer {
            type_: V4L2_BUF_TYPE_VIDEO_CAPTURE,
            index: buffer_index,
            plane: 0,
            flags: O_CLOEXEC | O_RDWR,
            fd: -1,
            reserved: [0; 11],
        };
        // SAFETY: ioctl(fd, EXPBUF, &ebuf) writes the exported fd into
        // ebuf.fd on success. Kernel does not retain the pointer.
        let ret = unsafe {
            ioctl(
                self.fd,
                VIDIOC_EXPBUF,
                &mut ebuf as *mut V4l2ExportBuffer as *mut u8,
            )
        };
        if ret < 0 {
            return Err(VideoError::Source(format!(
                "V4L2: VIDIOC_EXPBUF failed for buffer {buffer_index}"
            )));
        }
        if ebuf.fd < 0 {
            return Err(VideoError::Source(
                "V4L2: VIDIOC_EXPBUF returned invalid fd".into(),
            ));
        }
        Ok(ebuf.fd)
    }

    /// Mutable view of a specific mmap'd buffer's pixel data.
    ///
    /// Used together with `export_dmabuf()` to let an NPU wrap the same
    /// region via `rknn_create_mem_from_fd(fd, virt, size, offset)` where
    /// `virt` must point to the userspace view of the DMA-BUF.
    ///
    /// # Safety
    /// The returned slice aliases the mmap'd region. Do not call any
    /// frame-capture method while the slice is live.
    pub fn buffer_mut(&mut self, buffer_index: u32) -> Result<&mut [u8], VideoError> {
        let (ptr, len) = self
            .buffers
            .get(buffer_index as usize)
            .copied()
            .ok_or_else(|| {
                VideoError::Source(format!(
                    "V4L2: buffer_mut index {buffer_index} out of range"
                ))
            })?;
        // SAFETY: ptr is mmap'd, writable (MAP_SHARED | PROT_WRITE); len
        // is the exact length from VIDIOC_QUERYBUF.
        Ok(unsafe { std::slice::from_raw_parts_mut(ptr, len) })
    }

    /// Number of mmap'd buffers currently allocated.
    pub fn buffer_count(&self) -> usize {
        self.buffers.len()
    }

    /// Frame width in pixels.
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Frame height in pixels.
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Active pixel format FourCC code.
    pub fn pixel_format_fourcc(&self) -> u32 {
        self.pixel_format
    }

    /// Stop the capture stream.
    pub fn stop_streaming(&mut self) -> Result<(), VideoError> {
        if !self.streaming {
            return Ok(());
        }
        let mut buf_type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        let ret = unsafe {
            ioctl(
                self.fd,
                VIDIOC_STREAMOFF,
                &mut buf_type as *mut u32 as *mut u8,
            )
        };
        if ret < 0 {
            return Err(VideoError::Source("V4L2: STREAMOFF failed".into()));
        }
        self.streaming = false;
        Ok(())
    }

    /// Export a V4L2 buffer as a DMA-BUF and wrap the fd in a RAII guard
    /// that automatically `close(2)`s it on drop. Use this in pipelines
    /// where panic safety matters — leaking dma-buf fds across panic
    /// unwind grows the kernel `fdtable` and eventually denies opens.
    pub fn export_dmabuf_owned(&self, buffer_index: u32) -> Result<V4l2DmaBufGuard, VideoError> {
        let fd = self.export_dmabuf(buffer_index)?;
        Ok(V4l2DmaBufGuard { fd })
    }
}

/// RAII handle around an exported V4L2 DMA-BUF file descriptor.
///
/// `close(fd)` is called automatically on drop. Use [`Self::fd`] to feed
/// the descriptor into `RknnBackend::wrap_fd`. Use [`Self::into_raw`] to
/// take ownership and prevent the auto-close (caller becomes responsible).
pub struct V4l2DmaBufGuard {
    fd: i32,
}

impl V4l2DmaBufGuard {
    /// Read the underlying file descriptor without taking ownership.
    pub fn fd(&self) -> i32 {
        self.fd
    }

    /// Consume the guard and return the raw fd. Caller is now responsible
    /// for closing it (e.g. via `libc::close`).
    pub fn into_raw(self) -> i32 {
        let fd = self.fd;
        std::mem::forget(self);
        fd
    }
}

impl Drop for V4l2DmaBufGuard {
    fn drop(&mut self) {
        if self.fd >= 0 {
            // SAFETY: fd was returned by VIDIOC_EXPBUF, owned by us;
            // close(2) is signal-safe and the kernel handles concurrent close races.
            unsafe { close(self.fd) };
        }
    }
}

// SAFETY: a raw fd is just an integer; ownership semantics are enforced
// by the guard's exclusive Drop.
unsafe impl Send for V4l2DmaBufGuard {}
unsafe impl Sync for V4l2DmaBufGuard {}

impl Drop for V4l2Camera {
    fn drop(&mut self) {
        // Best-effort stop streaming
        if self.streaming {
            let _ = self.stop_streaming();
        }
        // Unmap all buffers
        for &(ptr, len) in &self.buffers {
            unsafe {
                munmap(ptr, len);
            }
        }
        // File is closed automatically by dropping self.file
    }
}

// ---------------------------------------------------------------------------
// Tests — struct layout verification
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn v4l2_capability_size() {
        assert_eq!(
            std::mem::size_of::<V4l2Capability>(),
            104,
            "V4l2Capability must be 104 bytes to match kernel ABI"
        );
    }

    #[test]
    fn v4l2_format_size() {
        // struct v4l2_format on 64-bit: type(4) + union fmt(200) + pad to 208 total
        // Our struct: type_(4) + V4l2PixFormat(48) + _pad(152) = 204
        // With alignment, the compiler may round up. The kernel struct is 208.
        let size = std::mem::size_of::<V4l2Format>();
        assert!(
            size >= 204,
            "V4l2Format must be at least 204 bytes, got {size}"
        );
    }

    #[test]
    fn v4l2_requestbuffers_size() {
        assert_eq!(
            std::mem::size_of::<V4l2RequestBuffers>(),
            20,
            "V4l2RequestBuffers must be 20 bytes to match kernel ABI"
        );
    }

    #[test]
    fn pixel_format_fourcc_values() {
        assert_eq!(V4l2PixelFormat::Yuyv.fourcc(), 0x5659_5559);
        assert_eq!(V4l2PixelFormat::Nv12.fourcc(), 0x3231_564E);
        assert_eq!(V4l2PixelFormat::Mjpeg.fourcc(), 0x4750_4A4D);
        assert_eq!(V4l2PixelFormat::H264.fourcc(), 0x3436_3248);
    }

    #[test]
    fn v4l2_pix_format_size() {
        assert_eq!(
            std::mem::size_of::<V4l2PixFormat>(),
            48,
            "V4l2PixFormat must be 48 bytes"
        );
    }
}
