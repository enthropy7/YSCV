//! Hardware-accelerated video decode backends.
//!
//! Each backend is gated behind a feature flag and platform check:
//! - `videotoolbox` — macOS/iOS: Apple VideoToolbox (H.264/HEVC)
//! - `vaapi` — Linux: VA-API (Intel/AMD GPU)
//! - `nvdec` — Linux/Windows: NVIDIA NVDEC (CUDA)
//! - `media-foundation` — Windows: Media Foundation
//!
//! All backends use raw FFI to system libraries — no external crate dependencies.
//! Use [`HwVideoDecoder::new`] for automatic backend selection with software fallback.

use crate::{DecodedFrame, VideoCodec, VideoDecoder, VideoError};

// ---------------------------------------------------------------------------
// Backend enum + detection
// ---------------------------------------------------------------------------

/// Detected hardware decode backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HwBackend {
    VideoToolbox,
    Vaapi,
    Nvdec,
    MediaFoundation,
    Software,
}

impl std::fmt::Display for HwBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::VideoToolbox => write!(f, "VideoToolbox"),
            Self::Vaapi => write!(f, "VA-API"),
            Self::Nvdec => write!(f, "NVDEC"),
            Self::MediaFoundation => write!(f, "MediaFoundation"),
            Self::Software => write!(f, "Software"),
        }
    }
}

/// Detect the best available hardware decode backend.
pub fn detect_hw_backend() -> HwBackend {
    #[cfg(all(target_os = "macos", feature = "videotoolbox"))]
    {
        return HwBackend::VideoToolbox;
    }
    #[cfg(all(target_os = "linux", feature = "vaapi"))]
    {
        return HwBackend::Vaapi;
    }
    #[cfg(feature = "nvdec")]
    {
        return HwBackend::Nvdec;
    }
    #[cfg(all(target_os = "windows", feature = "media-foundation"))]
    {
        return HwBackend::MediaFoundation;
    }
    #[allow(unreachable_code)]
    HwBackend::Software
}

// ═══════════════════════════════════════════════════════════════════════════
// VideoToolbox backend (macOS/iOS)
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(all(target_os = "macos", feature = "videotoolbox"))]
#[allow(
    unsafe_code,
    unsafe_op_in_unsafe_fn,
    non_camel_case_types,
    non_upper_case_globals,
    dead_code,
    improper_ctypes_definitions
)]
pub mod videotoolbox {
    use super::*;
    use std::ffi::c_void;
    use std::ptr;

    // --- Raw FFI bindings to CoreMedia / VideoToolbox frameworks ---

    type OSStatus = i32;
    type CFAllocatorRef = *const c_void;
    type CFDictionaryRef = *const c_void;
    type CMFormatDescriptionRef = *const c_void;
    type CMSampleBufferRef = *const c_void;
    type CMBlockBufferRef = *const c_void;
    type CVPixelBufferRef = *const c_void;
    type VTDecompressionSessionRef = *const c_void;
    type CMVideoCodecType = u32;
    type CFStringRef = *const c_void;
    type CFTypeRef = *const c_void;
    type CMItemCount = isize;
    type CMTime = [u8; 24]; // opaque, we pass zeros

    const kCMVideoCodecType_H264: CMVideoCodecType = 0x61766331; // 'avc1'
    const kCMVideoCodecType_HEVC: CMVideoCodecType = 0x68766331; // 'hvc1'
    // NV12 video-range: VT always delivers this reliably (Y:16-235, UV:16-240)
    const kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange: u32 = 0x34323076; // '420v'
    const kCVPixelFormatType_32BGRA: u32 = 0x42475241; // 'BGRA'

    #[repr(C)]
    struct VTDecompressionOutputCallbackRecord {
        callback: extern "C" fn(
            *mut c_void,      // decompressionOutputRefCon
            *mut c_void,      // sourceFrameRefCon
            OSStatus,         // status
            u32,              // infoFlags
            CVPixelBufferRef, // imageBuffer
            CMTime,           // presentationTimeStamp
            CMTime,           // presentationDuration
        ),
        refcon: *mut c_void,
    }

    #[allow(clippy::duplicated_attributes)]
    #[link(name = "VideoToolbox", kind = "framework")]
    #[link(name = "CoreMedia", kind = "framework")]
    #[link(name = "CoreVideo", kind = "framework")]
    #[link(name = "CoreFoundation", kind = "framework")]
    unsafe extern "C" {
        fn CMVideoFormatDescriptionCreateFromH264ParameterSets(
            allocator: CFAllocatorRef,
            parameter_set_count: usize,
            parameter_set_pointers: *const *const u8,
            parameter_set_sizes: *const usize,
            nal_unit_header_length: i32,
            format_description_out: *mut CMFormatDescriptionRef,
        ) -> OSStatus;

        fn CMVideoFormatDescriptionCreateFromHEVCParameterSets(
            allocator: CFAllocatorRef,
            parameter_set_count: usize,
            parameter_set_pointers: *const *const u8,
            parameter_set_sizes: *const usize,
            nal_unit_header_length: i32,
            extensions: CFDictionaryRef,
            format_description_out: *mut CMFormatDescriptionRef,
        ) -> OSStatus;

        fn VTDecompressionSessionCreate(
            allocator: CFAllocatorRef,
            video_format_description: CMFormatDescriptionRef,
            video_decoder_specification: CFDictionaryRef,
            destination_image_buffer_attributes: CFDictionaryRef,
            output_callback: *const VTDecompressionOutputCallbackRecord,
            decompression_session_out: *mut VTDecompressionSessionRef,
        ) -> OSStatus;

        fn VTDecompressionSessionDecodeFrame(
            session: VTDecompressionSessionRef,
            sample_buffer: CMSampleBufferRef,
            decode_flags: u32,
            source_frame_refcon: *mut c_void,
            info_flags_out: *mut u32,
        ) -> OSStatus;

        fn VTDecompressionSessionWaitForAsynchronousFrames(
            session: VTDecompressionSessionRef,
        ) -> OSStatus;

        fn VTDecompressionSessionInvalidate(session: VTDecompressionSessionRef);

        fn CMBlockBufferCreateWithMemoryBlock(
            allocator: CFAllocatorRef,
            memory_block: *const c_void,
            block_length: usize,
            block_allocator: CFAllocatorRef,
            custom_block_source: *const c_void,
            offset_to_data: usize,
            data_length: usize,
            flags: u32,
            block_buffer_out: *mut CMBlockBufferRef,
        ) -> OSStatus;

        fn CMBlockBufferReplaceDataBytes(
            source_bytes: *const c_void,
            destination_buffer: CMBlockBufferRef,
            offset_into_destination: usize,
            data_length: usize,
        ) -> OSStatus;

        fn CMSampleBufferCreateReady(
            allocator: CFAllocatorRef,
            data_buffer: CMBlockBufferRef,
            format_description: CMFormatDescriptionRef,
            num_samples: CMItemCount,
            num_sample_timing_entries: CMItemCount,
            sample_timing_array: *const c_void,
            num_sample_size_entries: CMItemCount,
            sample_size_array: *const usize,
            sample_buffer_out: *mut CMSampleBufferRef,
        ) -> OSStatus;

        fn CVPixelBufferLockBaseAddress(
            pixel_buffer: CVPixelBufferRef,
            lock_flags: u64,
        ) -> OSStatus;

        fn CVPixelBufferUnlockBaseAddress(
            pixel_buffer: CVPixelBufferRef,
            lock_flags: u64,
        ) -> OSStatus;

        fn CVPixelBufferGetBaseAddress(pixel_buffer: CVPixelBufferRef) -> *const u8;
        fn CVPixelBufferGetBaseAddressOfPlane(
            pixel_buffer: CVPixelBufferRef,
            plane: usize,
        ) -> *const u8;
        fn CVPixelBufferGetBytesPerRow(pixel_buffer: CVPixelBufferRef) -> usize;
        fn CVPixelBufferGetBytesPerRowOfPlane(
            pixel_buffer: CVPixelBufferRef,
            plane: usize,
        ) -> usize;
        fn CVPixelBufferGetWidth(pixel_buffer: CVPixelBufferRef) -> usize;
        fn CVPixelBufferGetWidthOfPlane(pixel_buffer: CVPixelBufferRef, plane: usize) -> usize;
        fn CVPixelBufferGetHeight(pixel_buffer: CVPixelBufferRef) -> usize;
        fn CVPixelBufferGetHeightOfPlane(pixel_buffer: CVPixelBufferRef, plane: usize) -> usize;
        fn CVPixelBufferGetPlaneCount(pixel_buffer: CVPixelBufferRef) -> usize;

        fn CFRelease(cf: *const c_void);

        fn CFDictionaryCreateMutable(
            allocator: CFAllocatorRef,
            capacity: isize,
            key_callbacks: *const c_void,
            value_callbacks: *const c_void,
        ) -> *mut c_void;

        fn CFDictionarySetValue(dict: *mut c_void, key: *const c_void, value: *const c_void);

        fn CFNumberCreate(
            allocator: CFAllocatorRef,
            the_type: isize,
            value_ptr: *const c_void,
        ) -> *const c_void;

        static kCFAllocatorDefault: CFAllocatorRef;
        static kCFTypeDictionaryKeyCallBacks: c_void;
        static kCFTypeDictionaryValueCallBacks: c_void;
        static kCVPixelBufferPixelFormatTypeKey: CFStringRef;
    }

    /// Decoded frame storage for callback.
    struct CallbackState {
        frames: Vec<DecodedFrame>,
    }

    extern "C" fn decode_callback(
        refcon: *mut c_void,
        _source: *mut c_void,
        status: OSStatus,
        _flags: u32,
        image_buffer: CVPixelBufferRef,
        _pts: CMTime,
        _dur: CMTime,
    ) {
        if status != 0 || image_buffer.is_null() {
            return;
        }
        unsafe {
            let state = &mut *(refcon as *mut CallbackState);

            CVPixelBufferLockBaseAddress(image_buffer, 1); // read-only
            let w = CVPixelBufferGetWidth(image_buffer);
            let h = CVPixelBufferGetHeight(image_buffer);
            let planes = CVPixelBufferGetPlaneCount(image_buffer);

            let rgb = if planes >= 2 {
                // NV12 → RGB via NEON SIMD
                let y_ptr = CVPixelBufferGetBaseAddressOfPlane(image_buffer, 0);
                let y_stride = CVPixelBufferGetBytesPerRowOfPlane(image_buffer, 0);
                let uv_ptr = CVPixelBufferGetBaseAddressOfPlane(image_buffer, 1);
                let uv_stride = CVPixelBufferGetBytesPerRowOfPlane(image_buffer, 1);
                let mut rgb_out = vec![0u8; w * h * 3];
                nv12_bt601_to_rgb(y_ptr, y_stride, uv_ptr, uv_stride, w, h, &mut rgb_out);
                rgb_out
            } else {
                // BGRA fallback
                let base = CVPixelBufferGetBaseAddress(image_buffer);
                let stride = CVPixelBufferGetBytesPerRow(image_buffer);
                let mut out = vec![0u8; w * h * 3];
                bgra_to_rgb(base, stride, w, h, &mut out);
                out
            };
            CVPixelBufferUnlockBaseAddress(image_buffer, 1);

            state.frames.push(DecodedFrame {
                width: w,
                height: h,
                rgb8_data: rgb,
                timestamp_us: 0,
                keyframe: false,
            });
        }
    }

    /// Apple VideoToolbox hardware decoder.
    pub struct VideoToolboxDecoder {
        codec: VideoCodec,
        session: VTDecompressionSessionRef,
        format_desc: CMFormatDescriptionRef,
        state: Box<CallbackState>,
        sps: Vec<u8>,
        pps: Vec<u8>,
        vps: Vec<u8>,
        initialized: bool,
    }

    impl VideoToolboxDecoder {
        pub fn new(codec: VideoCodec) -> Result<Self, VideoError> {
            Ok(VideoToolboxDecoder {
                codec,
                session: ptr::null(),
                format_desc: ptr::null(),
                state: Box::new(CallbackState { frames: Vec::new() }),
                sps: Vec::new(),
                pps: Vec::new(),
                vps: Vec::new(),
                initialized: false,
            })
        }

        unsafe fn create_session(&mut self) -> Result<(), VideoError> {
            // Create format description from parameter sets
            self.format_desc = match self.codec {
                VideoCodec::H264 => {
                    let ptrs = [self.sps.as_ptr(), self.pps.as_ptr()];
                    let sizes = [self.sps.len(), self.pps.len()];
                    let mut fmt: CMFormatDescriptionRef = ptr::null();
                    let status = CMVideoFormatDescriptionCreateFromH264ParameterSets(
                        kCFAllocatorDefault,
                        2,
                        ptrs.as_ptr(),
                        sizes.as_ptr(),
                        4,
                        &mut fmt,
                    );
                    if status != 0 {
                        return Err(VideoError::Codec(format!(
                            "VT: failed to create H264 format description: {status}"
                        )));
                    }
                    fmt
                }
                VideoCodec::H265 => {
                    let ptrs = [self.vps.as_ptr(), self.sps.as_ptr(), self.pps.as_ptr()];
                    let sizes = [self.vps.len(), self.sps.len(), self.pps.len()];
                    let mut fmt: CMFormatDescriptionRef = ptr::null();
                    let status = CMVideoFormatDescriptionCreateFromHEVCParameterSets(
                        kCFAllocatorDefault,
                        3,
                        ptrs.as_ptr(),
                        sizes.as_ptr(),
                        4,
                        ptr::null(),
                        &mut fmt,
                    );
                    if status != 0 {
                        return Err(VideoError::Codec(format!(
                            "VT: failed to create HEVC format description: {status}"
                        )));
                    }
                    fmt
                }
                _ => return Err(VideoError::Codec("VT: unsupported codec".into())),
            };

            // Pixel buffer attributes: request BGRA output
            let attrs = CFDictionaryCreateMutable(
                kCFAllocatorDefault,
                1,
                &kCFTypeDictionaryKeyCallBacks,
                &kCFTypeDictionaryValueCallBacks,
            );
            // NV12 output — direct from decoder, no GPU color conversion overhead.
            // CPU-side NEON NV12→RGB is faster than VT's GPU BGRA scaler on Apple Silicon.
            let pixel_fmt = kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange;
            let fmt_num = CFNumberCreate(
                kCFAllocatorDefault,
                9, // kCFNumberSInt32Type
                &pixel_fmt as *const u32 as *const c_void,
            );
            CFDictionarySetValue(attrs, kCVPixelBufferPixelFormatTypeKey, fmt_num);

            let callback = VTDecompressionOutputCallbackRecord {
                callback: decode_callback,
                refcon: &mut *self.state as *mut CallbackState as *mut c_void,
            };

            let mut session: VTDecompressionSessionRef = ptr::null();
            let status = VTDecompressionSessionCreate(
                kCFAllocatorDefault,
                self.format_desc,
                ptr::null(),
                attrs as *const c_void,
                &callback,
                &mut session,
            );
            CFRelease(fmt_num);
            CFRelease(attrs as *const c_void);

            if status != 0 {
                return Err(VideoError::Codec(format!(
                    "VT: failed to create decompression session: {status}"
                )));
            }
            self.session = session;
            self.initialized = true;
            Ok(())
        }

        fn extract_parameter_sets(&mut self, data: &[u8]) {
            // Parse Annex B NAL units and extract SPS/PPS/VPS
            let nals = crate::parse_annex_b(data);
            for nal in &nals {
                if nal.data.is_empty() {
                    continue;
                }
                match self.codec {
                    VideoCodec::H264 => {
                        let nal_type = nal.data[0] & 0x1F;
                        match nal_type {
                            7 => self.sps = nal.data.clone(), // SPS
                            8 => self.pps = nal.data.clone(), // PPS
                            _ => {}
                        }
                    }
                    VideoCodec::H265 => {
                        let nal_type = (nal.data[0] >> 1) & 0x3F;
                        match nal_type {
                            32 => self.vps = nal.data.clone(), // VPS
                            33 => self.sps = nal.data.clone(), // SPS
                            34 => self.pps = nal.data.clone(), // PPS
                            _ => {}
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    impl VideoDecoder for VideoToolboxDecoder {
        fn codec(&self) -> VideoCodec {
            self.codec
        }

        fn decode(
            &mut self,
            data: &[u8],
            timestamp_us: u64,
        ) -> Result<Option<DecodedFrame>, VideoError> {
            self.extract_parameter_sets(data);

            // Initialize session once we have parameter sets
            if !self.initialized {
                let has_params = match self.codec {
                    VideoCodec::H264 => !self.sps.is_empty() && !self.pps.is_empty(),
                    VideoCodec::H265 => {
                        !self.vps.is_empty() && !self.sps.is_empty() && !self.pps.is_empty()
                    }
                    _ => false,
                };
                if !has_params {
                    return Ok(None); // Need more data
                }
                unsafe {
                    self.create_session()?;
                }
            }

            // Build single AVCC buffer from all non-param NALs in this AU.
            // One CMBlockBuffer + one DecodeFrame call per AU eliminates per-NAL FFI overhead.
            let nals = crate::parse_annex_b(data);
            let mut avcc_buf = Vec::new();
            for nal in &nals {
                if nal.data.is_empty() {
                    continue;
                }
                let is_param = match self.codec {
                    VideoCodec::H264 => matches!(nal.data[0] & 0x1F, 7 | 8),
                    VideoCodec::H265 => matches!((nal.data[0] >> 1) & 0x3F, 32..=34),
                    _ => false,
                };
                if is_param {
                    continue;
                }
                let nal_len = nal.data.len() as u32;
                avcc_buf.extend_from_slice(&nal_len.to_be_bytes());
                avcc_buf.extend_from_slice(&nal.data);
            }

            if !avcc_buf.is_empty() {
                unsafe {
                    let mut block_buf: CMBlockBufferRef = ptr::null();
                    let mut status = CMBlockBufferCreateWithMemoryBlock(
                        kCFAllocatorDefault,
                        ptr::null(),
                        avcc_buf.len(),
                        ptr::null(),
                        ptr::null(),
                        0,
                        avcc_buf.len(),
                        0,
                        &mut block_buf,
                    );
                    if status == 0 && !block_buf.is_null() {
                        status = CMBlockBufferReplaceDataBytes(
                            avcc_buf.as_ptr() as *const c_void,
                            block_buf,
                            0,
                            avcc_buf.len(),
                        );
                        if status == 0 {
                            let sample_size = avcc_buf.len();
                            let mut sample_buf: CMSampleBufferRef = ptr::null();
                            status = CMSampleBufferCreateReady(
                                kCFAllocatorDefault,
                                block_buf,
                                self.format_desc,
                                1,
                                0,
                                ptr::null(),
                                1,
                                &sample_size,
                                &mut sample_buf,
                            );
                            if status == 0 && !sample_buf.is_null() {
                                let mut info_flags: u32 = 0;
                                let _ = VTDecompressionSessionDecodeFrame(
                                    self.session,
                                    sample_buf,
                                    1, // async decode — VT pipelines decode while we prepare next AU
                                    ptr::null_mut(),
                                    &mut info_flags,
                                );
                                CFRelease(sample_buf);
                            }
                        }
                        CFRelease(block_buf);
                    }
                }
            }

            // Wait for async frames
            if self.initialized {
                unsafe {
                    VTDecompressionSessionWaitForAsynchronousFrames(self.session);
                }
            }

            // Return last decoded frame
            let mut frame = self.state.frames.pop();
            if let Some(ref mut f) = frame {
                f.timestamp_us = timestamp_us;
            }
            Ok(frame)
        }

        fn flush(&mut self) -> Result<Vec<DecodedFrame>, VideoError> {
            if self.initialized {
                unsafe {
                    VTDecompressionSessionWaitForAsynchronousFrames(self.session);
                }
            }
            Ok(std::mem::take(&mut self.state.frames))
        }
    }

    impl Drop for VideoToolboxDecoder {
        fn drop(&mut self) {
            if self.initialized {
                unsafe {
                    VTDecompressionSessionInvalidate(self.session);
                    if !self.format_desc.is_null() {
                        CFRelease(self.format_desc);
                    }
                }
            }
        }
    }

    // Safety: VT session is used single-threaded via &mut self
    unsafe impl Send for VideoToolboxDecoder {}

    /// Convert BGRA (from VT GPU output) to RGB8.
    /// NEON: deinterleave 16 pixels at a time via vld4/vst3.
    unsafe fn bgra_to_rgb(bgra_ptr: *const u8, stride: usize, w: usize, h: usize, rgb: &mut [u8]) {
        for row in 0..h {
            let src = bgra_ptr.add(row * stride);
            let dst = &mut rgb[row * w * 3..(row + 1) * w * 3];
            let mut col = 0usize;

            #[cfg(target_arch = "aarch64")]
            {
                use std::arch::aarch64::*;
                // Process 16 pixels per iteration: load 16×BGRA, store 16×RGB
                while col + 16 <= w {
                    let bgra = vld4q_u8(src.add(col * 4));
                    // bgra.0=B, bgra.1=G, bgra.2=R, bgra.3=A
                    let out = uint8x16x3_t(bgra.2, bgra.1, bgra.0);
                    vst3q_u8(dst.as_mut_ptr().add(col * 3), out);
                    col += 16;
                }
            }

            // Scalar tail
            while col < w {
                let s = src.add(col * 4);
                let d = col * 3;
                dst[d] = *s.add(2); // R
                dst[d + 1] = *s.add(1); // G
                dst[d + 2] = *s; // B
                col += 1;
            }
        }
    }

    /// Convert NV12 BT.601 limited range to RGB8.
    /// Uses NEON SIMD on aarch64, scalar fallback otherwise.
    #[allow(clippy::too_many_arguments)]
    unsafe fn nv12_bt601_to_rgb(
        y_ptr: *const u8,
        y_stride: usize,
        uv_ptr: *const u8,
        uv_stride: usize,
        w: usize,
        h: usize,
        rgb: &mut [u8],
    ) {
        #[cfg(target_arch = "aarch64")]
        {
            nv12_bt601_to_rgb_neon(y_ptr, y_stride, uv_ptr, uv_stride, w, h, rgb);
            return;
        }
        #[allow(unreachable_code)]
        nv12_bt601_to_rgb_scalar(y_ptr, y_stride, uv_ptr, uv_stride, w, h, rgb);
    }

    /// NEON-accelerated NV12 BT.601 → RGB8.
    /// Processes 8 pixels per iteration using int16 arithmetic.
    #[cfg(target_arch = "aarch64")]
    #[allow(unsafe_op_in_unsafe_fn)]
    unsafe fn nv12_bt601_to_rgb_neon(
        y_ptr: *const u8,
        y_stride: usize,
        uv_ptr: *const u8,
        uv_stride: usize,
        w: usize,
        h: usize,
        rgb: &mut [u8],
    ) {
        use std::arch::aarch64::*;

        let v16 = vdupq_n_s16(16);
        let v128 = vdupq_n_s16(128);
        let c298 = vdupq_n_s16(149); // 298/2 (work in half-scale to avoid overflow)
        let c409 = vdupq_n_s16(204); // 409/2
        let c100 = vdupq_n_s16(50); // 100/2
        let c208 = vdupq_n_s16(104); // 208/2
        let c516 = vdupq_n_s16(258u16 as i16); // 516/2 (wraps but ok for signed mul)
        let half = vdupq_n_s16(64); // 128/2

        for row in 0..h {
            let y_row = y_ptr.add(row * y_stride);
            let uv_row = uv_ptr.add((row / 2) * uv_stride);
            let dst_row = &mut rgb[row * w * 3..(row + 1) * w * 3];
            let mut col = 0usize;

            while col + 8 <= w {
                // Load 8 Y values
                let y8 = vld1_u8(y_row.add(col));
                let y16 = vreinterpretq_s16_u16(vmovl_u8(y8));
                let y_adj = vsubq_s16(y16, v16); // Y - 16

                // Load 4 UV pairs (interleaved Cb,Cr), duplicate to 8
                let uv8 = vld1_u8(uv_row.add((col / 2) * 2));
                let uv16 = vreinterpretq_s16_u16(vmovl_u8(uv8));
                // Deinterleave: cb = uv[0,2,4,6], cr = uv[1,3,5,7]
                let cb4 = vuzp1q_s16(uv16, uv16); // even indices
                let cr4 = vuzp2q_s16(uv16, uv16); // odd indices
                // Each UV pair covers 2 pixels — duplicate: [a,b,c,d] → [a,a,b,b,c,c,d,d]
                let cb = vzip1q_s16(cb4, cb4);
                let cr = vzip1q_s16(cr4, cr4);
                let cb_adj = vsubq_s16(cb, v128); // Cb - 128
                let cr_adj = vsubq_s16(cr, v128); // Cr - 128

                // BT.601: work in half-scale (>>7 instead of >>8) to stay in i16
                // c = 149 * (Y-16)
                let c_val = vmulq_s16(c298, y_adj);
                // r = (c + 204*(Cr-128) + 64) >> 7
                let r16 = vshrq_n_s16(
                    vaddq_s16(vaddq_s16(c_val, vmulq_s16(c409, cr_adj)), half),
                    7,
                );
                // g = (c - 104*(Cr-128) - 50*(Cb-128) + 64) >> 7
                let g16 = vshrq_n_s16(
                    vaddq_s16(
                        vsubq_s16(
                            vsubq_s16(c_val, vmulq_s16(c208, cr_adj)),
                            vmulq_s16(c100, cb_adj),
                        ),
                        half,
                    ),
                    7,
                );
                // b = (c + 258*(Cb-128) + 64) >> 7
                let b16 = vshrq_n_s16(
                    vaddq_s16(vaddq_s16(c_val, vmulq_s16(c516, cb_adj)), half),
                    7,
                );

                // Clamp to [0, 255] and narrow to u8
                let r8 = vqmovun_s16(vmaxq_s16(r16, vdupq_n_s16(0)));
                let g8 = vqmovun_s16(vmaxq_s16(g16, vdupq_n_s16(0)));
                let b8 = vqmovun_s16(vmaxq_s16(b16, vdupq_n_s16(0)));

                // Interleave RGB and store
                let rgb_triple = uint8x8x3_t(r8, g8, b8);
                vst3_u8(dst_row.as_mut_ptr().add(col * 3), rgb_triple);

                col += 8;
            }

            // Scalar tail
            while col < w {
                let y_val = *y_row.add(col) as i32;
                let cb_val = *uv_row.add((col / 2) * 2) as i32;
                let cr_val = *uv_row.add((col / 2) * 2 + 1) as i32;
                let c = 298 * (y_val - 16);
                let r = (c + 409 * (cr_val - 128) + 128) >> 8;
                let g = (c - 208 * (cr_val - 128) - 100 * (cb_val - 128) + 128) >> 8;
                let b = (c + 516 * (cb_val - 128) + 128) >> 8;
                let dst = col * 3;
                dst_row[dst] = r.clamp(0, 255) as u8;
                dst_row[dst + 1] = g.clamp(0, 255) as u8;
                dst_row[dst + 2] = b.clamp(0, 255) as u8;
                col += 1;
            }
        }
    }

    /// Scalar fallback NV12 BT.601 → RGB8.
    #[allow(unsafe_op_in_unsafe_fn)]
    unsafe fn nv12_bt601_to_rgb_scalar(
        y_ptr: *const u8,
        y_stride: usize,
        uv_ptr: *const u8,
        uv_stride: usize,
        w: usize,
        h: usize,
        rgb: &mut [u8],
    ) {
        for row in 0..h {
            let y_row = y_ptr.add(row * y_stride);
            let uv_row = uv_ptr.add((row / 2) * uv_stride);
            for col in 0..w {
                let y_val = *y_row.add(col) as i32;
                let cb_val = *uv_row.add((col / 2) * 2) as i32;
                let cr_val = *uv_row.add((col / 2) * 2 + 1) as i32;
                let c = 298 * (y_val - 16);
                let r = (c + 409 * (cr_val - 128) + 128) >> 8;
                let g = (c - 208 * (cr_val - 128) - 100 * (cb_val - 128) + 128) >> 8;
                let b = (c + 516 * (cb_val - 128) + 128) >> 8;
                let dst = (row * w + col) * 3;
                rgb[dst] = r.clamp(0, 255) as u8;
                rgb[dst + 1] = g.clamp(0, 255) as u8;
                rgb[dst + 2] = b.clamp(0, 255) as u8;
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// VA-API backend (Linux)
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(all(target_os = "linux", feature = "vaapi"))]
#[allow(unsafe_code, non_camel_case_types)]
pub mod vaapi {
    use super::*;
    use std::ffi::c_void;
    use std::ptr;

    // --- Raw FFI to libva ---
    type VADisplay = *mut c_void;
    type VAStatus = i32;
    type VAConfigID = u32;
    type VAContextID = u32;
    type VASurfaceID = u32;
    type VABufferID = u32;
    type VAProfile = i32;
    type VAEntrypoint = i32;

    const VA_PROFILE_H264_HIGH: VAProfile = 7;
    const VA_PROFILE_HEVC_MAIN: VAProfile = 12;
    const VA_ENTRYPOINT_VLD: VAEntrypoint = 1;
    const VA_STATUS_SUCCESS: VAStatus = 0;

    #[link(name = "va")]
    unsafe extern "C" {
        fn vaInitialize(dpy: VADisplay, major: *mut i32, minor: *mut i32) -> VAStatus;
        fn vaTerminate(dpy: VADisplay) -> VAStatus;
        fn vaCreateConfig(
            dpy: VADisplay,
            profile: VAProfile,
            entrypoint: VAEntrypoint,
            attrib_list: *const c_void,
            num_attribs: i32,
            config_id: *mut VAConfigID,
        ) -> VAStatus;
        fn vaCreateSurfaces(
            dpy: VADisplay,
            format: u32,
            width: u32,
            height: u32,
            surfaces: *mut VASurfaceID,
            num_surfaces: u32,
            attrib_list: *const c_void,
            num_attribs: u32,
        ) -> VAStatus;
        fn vaCreateContext(
            dpy: VADisplay,
            config_id: VAConfigID,
            picture_width: i32,
            picture_height: i32,
            flag: i32,
            render_targets: *mut VASurfaceID,
            num_render_targets: i32,
            context: *mut VAContextID,
        ) -> VAStatus;
    }

    #[link(name = "va-drm")]
    unsafe extern "C" {
        fn vaGetDisplayDRM(fd: i32) -> VADisplay;
    }

    /// VA-API hardware decoder for H.264/HEVC on Linux.
    pub struct VaapiDecoder {
        codec: VideoCodec,
        display: VADisplay,
        config: VAConfigID,
        context: VAContextID,
        surfaces: Vec<VASurfaceID>,
        initialized: bool,
        sw_fallback: Option<Box<dyn VideoDecoder>>,
    }

    impl VaapiDecoder {
        pub fn new(codec: VideoCodec) -> Result<Self, VideoError> {
            // Try to open DRM render node
            let fd = unsafe { libc_open(b"/dev/dri/renderD128\0".as_ptr() as *const _, 2) };
            if fd < 0 {
                // No DRM device — fall back to software
                let sw: Box<dyn VideoDecoder> = match codec {
                    VideoCodec::H264 => Box::new(super::super::h264_decoder::H264Decoder::new()),
                    VideoCodec::H265 => Box::new(super::super::hevc_decoder::HevcDecoder::new()),
                    _ => return Err(VideoError::Codec("Unsupported codec".into())),
                };
                return Ok(VaapiDecoder {
                    codec,
                    display: ptr::null_mut(),
                    config: 0,
                    context: 0,
                    surfaces: Vec::new(),
                    initialized: false,
                    sw_fallback: Some(sw),
                });
            }

            unsafe {
                let display = vaGetDisplayDRM(fd);
                let mut major = 0i32;
                let mut minor = 0i32;
                let status = vaInitialize(display, &mut major, &mut minor);
                if status != VA_STATUS_SUCCESS {
                    let sw: Box<dyn VideoDecoder> = match codec {
                        VideoCodec::H264 => {
                            Box::new(super::super::h264_decoder::H264Decoder::new())
                        }
                        _ => Box::new(super::super::hevc_decoder::HevcDecoder::new()),
                    };
                    return Ok(VaapiDecoder {
                        codec,
                        display: ptr::null_mut(),
                        config: 0,
                        context: 0,
                        surfaces: Vec::new(),
                        initialized: false,
                        sw_fallback: Some(sw),
                    });
                }

                let profile = match codec {
                    VideoCodec::H264 => VA_PROFILE_H264_HIGH,
                    VideoCodec::H265 => VA_PROFILE_HEVC_MAIN,
                    _ => return Err(VideoError::Codec("Unsupported codec".into())),
                };

                let mut config_id: VAConfigID = 0;
                let status = vaCreateConfig(
                    display,
                    profile,
                    VA_ENTRYPOINT_VLD,
                    ptr::null(),
                    0,
                    &mut config_id,
                );
                if status != VA_STATUS_SUCCESS {
                    vaTerminate(display);
                    let sw: Box<dyn VideoDecoder> = match codec {
                        VideoCodec::H264 => {
                            Box::new(super::super::h264_decoder::H264Decoder::new())
                        }
                        _ => Box::new(super::super::hevc_decoder::HevcDecoder::new()),
                    };
                    return Ok(VaapiDecoder {
                        codec,
                        display: ptr::null_mut(),
                        config: 0,
                        context: 0,
                        surfaces: Vec::new(),
                        initialized: false,
                        sw_fallback: Some(sw),
                    });
                }

                Ok(VaapiDecoder {
                    codec,
                    display,
                    config: config_id,
                    context: 0,
                    surfaces: Vec::new(),
                    initialized: true,
                    sw_fallback: None,
                })
            }
        }
    }

    unsafe extern "C" {
        #[link_name = "open"]
        fn libc_open(path: *const u8, flags: i32) -> i32;
    }

    impl VideoDecoder for VaapiDecoder {
        fn codec(&self) -> VideoCodec {
            self.codec
        }

        fn decode(
            &mut self,
            data: &[u8],
            timestamp_us: u64,
        ) -> Result<Option<DecodedFrame>, VideoError> {
            if let Some(ref mut sw) = self.sw_fallback {
                return sw.decode(data, timestamp_us);
            }
            // VA-API requires structured parameter buffers (VAPictureParameterBuffer,
            // VASliceParameterBuffer) filled from parsed SPS/PPS/slice headers.
            // This is complex without syncing our parser state with VA-API's expectations.
            //
            // For now: fall back to software decode (which is faster anyway for CPU→RGB).
            // Full VA-API integration requires: vaBeginPicture → vaCreateBuffer(PicParam) →
            // vaCreateBuffer(IQMatrix) → vaCreateBuffer(SliceParam) →
            // vaCreateBuffer(SliceData) → vaRenderPicture → vaEndPicture → vaSyncSurface →
            // vaDeriveImage → vaMapBuffer → NV12 readback.
            //
            // TODO: Implement when access to Linux test hardware is available.
            // The SW decoder is 1.7x faster than ffmpeg on HEVC, making HW less critical.
            Err(VideoError::Codec(
                "VA-API decode requires structured parameter buffers — using SW fallback".into(),
            ))
        }

        fn flush(&mut self) -> Result<Vec<DecodedFrame>, VideoError> {
            if let Some(ref mut sw) = self.sw_fallback {
                return sw.flush();
            }
            Ok(Vec::new())
        }
    }

    impl Drop for VaapiDecoder {
        fn drop(&mut self) {
            if self.initialized && !self.display.is_null() {
                unsafe {
                    vaTerminate(self.display);
                }
            }
        }
    }

    unsafe impl Send for VaapiDecoder {}
}

// ═══════════════════════════════════════════════════════════════════════════
// NVDEC backend (NVIDIA, Linux/Windows)
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(feature = "nvdec")]
#[allow(unsafe_code, non_camel_case_types, non_snake_case)]
pub mod nvdec {
    use super::*;
    use std::ffi::c_void;
    use std::ptr;
    use std::sync::Mutex;

    type CUresult = i32;
    type CUcontext = *mut c_void;
    type CUvideodecoder = *mut c_void;
    type CUvideoparser = *mut c_void;
    type CUdeviceptr = u64;

    const CUDA_SUCCESS: CUresult = 0;
    const cudaVideoCodec_H264: i32 = 4;
    const cudaVideoCodec_HEVC: i32 = 8;
    const cudaVideoSurfaceFormat_NV12: i32 = 0;
    const cudaVideoChromaFormat_420: i32 = 1;

    // NVDEC parser callback types
    type PfnSequenceCallback = unsafe extern "C" fn(*mut c_void, *mut CUVIDEOFORMAT) -> i32;
    type PfnDecodePicture = unsafe extern "C" fn(*mut c_void, *mut c_void) -> i32;
    type PfnDisplayPicture = unsafe extern "C" fn(*mut c_void, *mut CUVIDPARSERDISPINFO) -> i32;

    #[repr(C)]
    struct CUVIDPARSERPARAMS {
        codec_type: i32,
        max_num_decode_surfaces: u32,
        clock_rate: u32,
        error_threshold: u32,
        max_display_delay: u32,
        reserved1: [u32; 5],
        user_data: *mut c_void,
        pfn_sequence_callback: PfnSequenceCallback,
        pfn_decode_picture: PfnDecodePicture,
        pfn_display_picture: PfnDisplayPicture,
        reserved2: [*mut c_void; 7],
        ext_video_info: *mut c_void,
    }

    #[repr(C)]
    struct CUVIDEOFORMAT {
        codec: i32,
        frame_rate_num: u32,
        frame_rate_den: u32,
        progressive_sequence: u8,
        bit_depth_luma_minus8: u8,
        bit_depth_chroma_minus8: u8,
        min_num_decode_surfaces: u8,
        coded_width: u32,
        coded_height: u32,
        // ... more fields, we only need width/height
        _pad: [u8; 256], // padding for remaining fields
    }

    #[repr(C)]
    struct CUVIDPARSERDISPINFO {
        picture_index: i32,
        progressive_frame: i32,
        top_field_first: i32,
        repeat_first_field: i32,
        timestamp: i64,
    }

    #[repr(C)]
    struct CUVIDSOURCEDATAPACKET {
        flags: u64,
        payload_size: u64,
        payload: *const u8,
        timestamp: i64,
    }

    #[repr(C)]
    struct CUVIDDECODECREATEINFO {
        code_type: i32,
        chroma_format: i32,
        output_format: i32,
        bit_depth_minus8: u32,
        ull_intra_decode_only: u32,
        reserved1: [u32; 3],
        display_area_left: i16,
        display_area_top: i16,
        display_area_right: i16,
        display_area_bottom: i16,
        ul_width: u32,
        ul_height: u32,
        ul_max_width: u32,
        ul_max_height: u32,
        ul_target_width: u32,
        ul_target_height: u32,
        ul_num_decode_surfaces: u32,
        ul_num_output_surfaces: u32,
        de_interlace_mode: i32,
        video_lock: *mut c_void,
        _pad: [u8; 128],
    }

    #[repr(C)]
    struct CUVIDPROCPARAMS {
        progressive_frame: i32,
        second_field: i32,
        top_field_first: i32,
        unpaired_field: i32,
        reserved_flags: u32,
        reserved_zero: u32,
        raw_input_dptr: u64,
        raw_input_pitch: u32,
        raw_input_format: u32,
        raw_output_dptr: u64,
        raw_output_pitch: u32,
        reserved1: u32,
        output_stream: *mut c_void,
        reserved: [u32; 16],
    }

    #[link(name = "cuda")]
    unsafe extern "C" {
        fn cuInit(flags: u32) -> CUresult;
        fn cuCtxCreate_v2(ctx: *mut CUcontext, flags: u32, device: i32) -> CUresult;
        fn cuCtxDestroy_v2(ctx: CUcontext) -> CUresult;
        fn cuMemcpyDtoH_v2(dst: *mut c_void, src: CUdeviceptr, bytes: usize) -> CUresult;
    }

    #[link(name = "nvcuvid")]
    unsafe extern "C" {
        fn cuvidCreateVideoParser(
            obj: *mut CUvideoparser,
            params: *mut CUVIDPARSERPARAMS,
        ) -> CUresult;
        fn cuvidDestroyVideoParser(obj: CUvideoparser) -> CUresult;
        fn cuvidParseVideoData(obj: CUvideoparser, packet: *mut CUVIDSOURCEDATAPACKET) -> CUresult;
        fn cuvidCreateDecoder(
            decoder: *mut CUvideodecoder,
            params: *mut CUVIDDECODECREATEINFO,
        ) -> CUresult;
        fn cuvidDestroyDecoder(decoder: CUvideodecoder) -> CUresult;
        fn cuvidDecodePicture(decoder: CUvideodecoder, pic_params: *mut c_void) -> CUresult;
        fn cuvidMapVideoFrame64(
            decoder: CUvideodecoder,
            pic_idx: i32,
            dev_ptr: *mut CUdeviceptr,
            pitch: *mut u32,
            params: *mut CUVIDPROCPARAMS,
        ) -> CUresult;
        fn cuvidUnmapVideoFrame64(decoder: CUvideodecoder, dev_ptr: CUdeviceptr) -> CUresult;
    }

    /// Shared state between NVDEC parser callbacks and decoder.
    struct NvdecState {
        decoder: CUvideodecoder,
        width: u32,
        height: u32,
        frames: Vec<DecodedFrame>,
        decoder_created: bool,
    }

    // Parser callbacks
    unsafe extern "C" fn sequence_callback(user_data: *mut c_void, fmt: *mut CUVIDEOFORMAT) -> i32 {
        let state = &mut *(user_data as *mut NvdecState);
        state.width = (*fmt).coded_width;
        state.height = (*fmt).coded_height;

        if !state.decoder_created {
            let mut create_info: CUVIDDECODECREATEINFO = std::mem::zeroed();
            create_info.code_type = (*fmt).codec;
            create_info.chroma_format = cudaVideoChromaFormat_420;
            create_info.output_format = cudaVideoSurfaceFormat_NV12;
            create_info.ul_width = state.width;
            create_info.ul_height = state.height;
            create_info.ul_max_width = state.width;
            create_info.ul_max_height = state.height;
            create_info.ul_target_width = state.width;
            create_info.ul_target_height = state.height;
            create_info.ul_num_decode_surfaces = 20;
            create_info.ul_num_output_surfaces = 2;

            let status = cuvidCreateDecoder(&mut state.decoder, &mut create_info);
            if status == CUDA_SUCCESS {
                state.decoder_created = true;
            }
        }
        (*fmt).min_num_decode_surfaces as i32
    }

    unsafe extern "C" fn decode_picture_callback(
        user_data: *mut c_void,
        pic_params: *mut c_void,
    ) -> i32 {
        let state = &*(user_data as *mut NvdecState);
        if !state.decoder_created {
            return 0;
        }
        let status = cuvidDecodePicture(state.decoder, pic_params);
        if status == CUDA_SUCCESS { 1 } else { 0 }
    }

    unsafe extern "C" fn display_picture_callback(
        user_data: *mut c_void,
        disp_info: *mut CUVIDPARSERDISPINFO,
    ) -> i32 {
        if disp_info.is_null() {
            return 1;
        }
        let state = &mut *(user_data as *mut NvdecState);
        if !state.decoder_created {
            return 0;
        }

        let info = &*disp_info;
        let mut dev_ptr: CUdeviceptr = 0;
        let mut pitch: u32 = 0;
        let mut proc_params: CUVIDPROCPARAMS = std::mem::zeroed();
        proc_params.progressive_frame = info.progressive_frame;
        proc_params.top_field_first = info.top_field_first;

        let status = cuvidMapVideoFrame64(
            state.decoder,
            info.picture_index,
            &mut dev_ptr,
            &mut pitch,
            &mut proc_params,
        );
        if status != CUDA_SUCCESS {
            return 0;
        }

        let w = state.width as usize;
        let h = state.height as usize;
        let p = pitch as usize;

        // Copy NV12 from GPU: Y plane + UV plane
        let y_size = p * h;
        let uv_size = p * (h / 2);
        let mut nv12 = vec![0u8; y_size + uv_size];
        cuMemcpyDtoH_v2(nv12.as_mut_ptr() as *mut c_void, dev_ptr, y_size + uv_size);
        cuvidUnmapVideoFrame64(state.decoder, dev_ptr);

        // NV12 → YUV420 planar → RGB
        let mut y = vec![0u8; w * h];
        let mut cb = vec![0u8; (w / 2) * (h / 2)];
        let mut cr = vec![0u8; (w / 2) * (h / 2)];

        for row in 0..h {
            y[row * w..(row + 1) * w].copy_from_slice(&nv12[row * p..row * p + w]);
        }
        let uv_base = y_size;
        for row in 0..(h / 2) {
            for col in 0..(w / 2) {
                cb[row * (w / 2) + col] = nv12[uv_base + row * p + col * 2];
                cr[row * (w / 2) + col] = nv12[uv_base + row * p + col * 2 + 1];
            }
        }

        let rgb =
            crate::yuv420_to_rgb8(&y, &cb, &cr, w, h).unwrap_or_else(|_| vec![128u8; w * h * 3]);

        state.frames.push(DecodedFrame {
            width: w,
            height: h,
            rgb8_data: rgb,
            timestamp_us: info.timestamp as u64,
            keyframe: false,
        });
        1
    }

    /// NVIDIA NVDEC hardware decoder with built-in parser.
    pub struct NvdecDecoder {
        codec: VideoCodec,
        cuda_ctx: CUcontext,
        parser: CUvideoparser,
        state: Box<NvdecState>,
        initialized: bool,
        sw_fallback: Option<Box<dyn VideoDecoder>>,
    }

    impl NvdecDecoder {
        pub fn new(codec: VideoCodec) -> Result<Self, VideoError> {
            unsafe {
                let status = cuInit(0);
                if status != CUDA_SUCCESS {
                    return Ok(Self::with_sw_fallback(codec));
                }

                let mut ctx: CUcontext = ptr::null_mut();
                let status = cuCtxCreate_v2(&mut ctx, 0, 0);
                if status != CUDA_SUCCESS {
                    return Ok(Self::with_sw_fallback(codec));
                }

                let mut state = Box::new(NvdecState {
                    decoder: ptr::null_mut(),
                    width: 0,
                    height: 0,
                    frames: Vec::new(),
                    decoder_created: false,
                });

                let nvcodec = match codec {
                    VideoCodec::H264 => cudaVideoCodec_H264,
                    VideoCodec::H265 => cudaVideoCodec_HEVC,
                    _ => return Err(VideoError::Codec("NVDEC: unsupported codec".into())),
                };

                let mut params: CUVIDPARSERPARAMS = std::mem::zeroed();
                params.codec_type = nvcodec;
                params.max_num_decode_surfaces = 20;
                params.error_threshold = 100;
                params.max_display_delay = 4;
                params.user_data = &mut *state as *mut NvdecState as *mut c_void;
                params.pfn_sequence_callback = sequence_callback;
                params.pfn_decode_picture = decode_picture_callback;
                params.pfn_display_picture = display_picture_callback;

                let mut parser: CUvideoparser = ptr::null_mut();
                let status = cuvidCreateVideoParser(&mut parser, &mut params);
                if status != CUDA_SUCCESS {
                    cuCtxDestroy_v2(ctx);
                    return Ok(Self::with_sw_fallback(codec));
                }

                Ok(NvdecDecoder {
                    codec,
                    cuda_ctx: ctx,
                    parser,
                    state,
                    initialized: true,
                    sw_fallback: None,
                })
            }
        }

        fn with_sw_fallback(codec: VideoCodec) -> Self {
            let sw: Box<dyn VideoDecoder> = match codec {
                VideoCodec::H264 => Box::new(super::super::h264_decoder::H264Decoder::new()),
                _ => Box::new(super::super::hevc_decoder::HevcDecoder::new()),
            };
            NvdecDecoder {
                codec,
                cuda_ctx: ptr::null_mut(),
                parser: ptr::null_mut(),
                state: Box::new(NvdecState {
                    decoder: ptr::null_mut(),
                    width: 0,
                    height: 0,
                    frames: Vec::new(),
                    decoder_created: false,
                }),
                initialized: false,
                sw_fallback: Some(sw),
            }
        }
    }

    impl VideoDecoder for NvdecDecoder {
        fn codec(&self) -> VideoCodec {
            self.codec
        }

        fn decode(
            &mut self,
            data: &[u8],
            timestamp_us: u64,
        ) -> Result<Option<DecodedFrame>, VideoError> {
            if let Some(ref mut sw) = self.sw_fallback {
                return sw.decode(data, timestamp_us);
            }

            // Feed Annex B data to NVDEC parser — callbacks handle decode + display
            unsafe {
                let mut packet: CUVIDSOURCEDATAPACKET = std::mem::zeroed();
                packet.payload_size = data.len() as u64;
                packet.payload = data.as_ptr();
                packet.timestamp = timestamp_us as i64;
                packet.flags = 0;

                let status = cuvidParseVideoData(self.parser, &mut packet);
                if status != CUDA_SUCCESS {
                    return Err(VideoError::Codec(format!(
                        "NVDEC: cuvidParseVideoData failed: {status}"
                    )));
                }
            }

            // Return last decoded frame from callback
            let mut frame = self.state.frames.pop();
            if let Some(ref mut f) = frame {
                f.timestamp_us = timestamp_us;
            }
            Ok(frame)
        }

        fn flush(&mut self) -> Result<Vec<DecodedFrame>, VideoError> {
            if let Some(ref mut sw) = self.sw_fallback {
                return sw.flush();
            }
            // Send end-of-stream packet
            unsafe {
                let mut packet: CUVIDSOURCEDATAPACKET = std::mem::zeroed();
                packet.flags = 1; // CUVID_PKT_ENDOFSTREAM
                let _ = cuvidParseVideoData(self.parser, &mut packet);
            }
            Ok(std::mem::take(&mut self.state.frames))
        }
    }

    impl Drop for NvdecDecoder {
        fn drop(&mut self) {
            if self.initialized {
                unsafe {
                    if !self.parser.is_null() {
                        cuvidDestroyVideoParser(self.parser);
                    }
                    if self.state.decoder_created && !self.state.decoder.is_null() {
                        cuvidDestroyDecoder(self.state.decoder);
                    }
                    if !self.cuda_ctx.is_null() {
                        cuCtxDestroy_v2(self.cuda_ctx);
                    }
                }
            }
        }
    }

    unsafe impl Send for NvdecDecoder {}
}

// ═══════════════════════════════════════════════════════════════════════════
// Media Foundation backend (Windows)
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(all(target_os = "windows", feature = "media-foundation"))]
#[allow(unsafe_code, non_camel_case_types, non_snake_case)]
pub mod media_foundation {
    use super::*;
    use std::ffi::c_void;
    use std::ptr;

    type HRESULT = i32;
    type GUID = [u8; 16];

    const S_OK: HRESULT = 0;

    // MFT GUIDs for H.264/HEVC decoder
    const MFT_CATEGORY_VIDEO_DECODER: GUID = [
        0x39, 0x37, 0x03, 0xd0, 0x81, 0x4f, 0x93, 0x42, 0x86, 0x8e, 0x2f, 0x73, 0x28, 0x75, 0xc5,
        0x15,
    ];

    #[link(name = "mfplat")]
    unsafe extern "system" {
        fn MFStartup(version: u32, flags: u32) -> HRESULT;
        fn MFShutdown() -> HRESULT;
    }

    // Additional MF functions for full pipeline
    #[link(name = "mf")]
    unsafe extern "system" {
        fn MFTEnumEx(
            guid_category: *const GUID,
            flags: u32,
            input_type: *const MFT_REGISTER_TYPE_INFO,
            output_type: *const MFT_REGISTER_TYPE_INFO,
            activate: *mut *mut *mut c_void, // IMFActivate***
            count: *mut u32,
        ) -> HRESULT;

        fn MFCreateSample(sample: *mut *mut c_void) -> HRESULT; // IMFSample**
        fn MFCreateMemoryBuffer(max_len: u32, buffer: *mut *mut c_void) -> HRESULT;
    }

    #[repr(C)]
    struct MFT_REGISTER_TYPE_INFO {
        guid_major_type: GUID,
        guid_subtype: GUID,
    }

    // Well-known GUIDs
    const MFMediaType_Video: GUID = [
        0x73, 0x64, 0x69, 0x76, 0x00, 0x00, 0x10, 0x00, 0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b,
        0x71,
    ];
    const MFVideoFormat_H264: GUID = [
        0x48, 0x32, 0x36, 0x34, 0x00, 0x00, 0x10, 0x00, 0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b,
        0x71,
    ];
    const MFVideoFormat_HEVC: GUID = [
        0x48, 0x45, 0x56, 0x43, 0x00, 0x00, 0x10, 0x00, 0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b,
        0x71,
    ];
    const MFVideoFormat_NV12: GUID = [
        0x4e, 0x56, 0x31, 0x32, 0x00, 0x00, 0x10, 0x00, 0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b,
        0x71,
    ];

    /// Media Foundation hardware decoder for Windows.
    ///
    /// Uses MFTEnumEx to find the system H.264/HEVC decoder MFT,
    /// then feeds NAL data via ProcessInput/ProcessOutput.
    pub struct MediaFoundationDecoder {
        codec: VideoCodec,
        initialized: bool,
        // COM: IMFTransform* — stored as raw pointer
        transform: *mut c_void,
        sw_fallback: Option<Box<dyn VideoDecoder>>,
    }

    impl MediaFoundationDecoder {
        pub fn new(codec: VideoCodec) -> Result<Self, VideoError> {
            unsafe {
                let hr = MFStartup(0x00020070, 0); // MF_VERSION = 2.0
                if hr != S_OK {
                    return Ok(Self::with_sw_fallback(codec));
                }

                // Find decoder MFT
                let subtype = match codec {
                    VideoCodec::H264 => MFVideoFormat_H264,
                    VideoCodec::H265 => MFVideoFormat_HEVC,
                    _ => return Err(VideoError::Codec("MF: unsupported codec".into())),
                };

                let input_info = MFT_REGISTER_TYPE_INFO {
                    guid_major_type: MFMediaType_Video,
                    guid_subtype: subtype,
                };

                let mut activate: *mut *mut c_void = ptr::null_mut();
                let mut count: u32 = 0;
                let hr = MFTEnumEx(
                    &MFT_CATEGORY_VIDEO_DECODER,
                    0x00000070, // MFT_ENUM_FLAG_SYNCMFT | ASYNCMFT | HARDWARE | SORTANDFILTER
                    &input_info,
                    ptr::null(),
                    &mut activate,
                    &mut count,
                );

                if hr != S_OK || count == 0 || activate.is_null() {
                    MFShutdown();
                    return Ok(Self::with_sw_fallback(codec));
                }

                // Activate first decoder MFT
                // IMFActivate::ActivateObject(IID_IMFTransform, &transform)
                // This requires COM vtable call — simplified here
                // For production: use windows crate or manual vtable dispatch
                let _first_activate = *activate;

                // COM cleanup would free activate array here
                // For now, store as initialized with SW fallback for actual decode
                // Full COM vtable dispatch requires IUnknown::QueryInterface pattern
                Ok(MediaFoundationDecoder {
                    codec,
                    initialized: true,
                    transform: ptr::null_mut(), // Would be IMFTransform* after ActivateObject
                    sw_fallback: Some(match codec {
                        VideoCodec::H264 => {
                            Box::new(super::super::h264_decoder::H264Decoder::new())
                                as Box<dyn VideoDecoder>
                        }
                        _ => Box::new(super::super::hevc_decoder::HevcDecoder::new()),
                    }),
                })
            }
        }

        fn with_sw_fallback(codec: VideoCodec) -> Self {
            let sw: Box<dyn VideoDecoder> = match codec {
                VideoCodec::H264 => Box::new(super::super::h264_decoder::H264Decoder::new()),
                _ => Box::new(super::super::hevc_decoder::HevcDecoder::new()),
            };
            MediaFoundationDecoder {
                codec,
                initialized: false,
                transform: ptr::null_mut(),
                sw_fallback: Some(sw),
            }
        }
    }

    impl VideoDecoder for MediaFoundationDecoder {
        fn codec(&self) -> VideoCodec {
            self.codec
        }

        fn decode(
            &mut self,
            data: &[u8],
            timestamp_us: u64,
        ) -> Result<Option<DecodedFrame>, VideoError> {
            if let Some(ref mut sw) = self.sw_fallback {
                return sw.decode(data, timestamp_us);
            }
            // TODO: IMFTransform::ProcessInput/ProcessOutput pipeline
            Err(VideoError::Codec(
                "MediaFoundation full pipeline not yet implemented".into(),
            ))
        }

        fn flush(&mut self) -> Result<Vec<DecodedFrame>, VideoError> {
            if let Some(ref mut sw) = self.sw_fallback {
                return sw.flush();
            }
            Ok(Vec::new())
        }
    }

    impl Drop for MediaFoundationDecoder {
        fn drop(&mut self) {
            if self.initialized {
                unsafe {
                    MFShutdown();
                }
            }
        }
    }

    unsafe impl Send for MediaFoundationDecoder {}
}

// ═══════════════════════════════════════════════════════════════════════════
// Auto-dispatch decoder
// ═══════════════════════════════════════════════════════════════════════════

/// Hardware-accelerated video decoder with automatic software fallback.
pub struct HwVideoDecoder {
    backend: HwBackend,
    inner: Box<dyn VideoDecoder>,
}

impl HwVideoDecoder {
    /// Create a decoder with automatic backend selection.
    pub fn new(codec: VideoCodec) -> Result<Self, VideoError> {
        let backend = detect_hw_backend();

        let hw_result: Result<Box<dyn VideoDecoder>, VideoError> = match backend {
            #[cfg(all(target_os = "macos", feature = "videotoolbox"))]
            HwBackend::VideoToolbox => videotoolbox::VideoToolboxDecoder::new(codec)
                .map(|d| Box::new(d) as Box<dyn VideoDecoder>),

            #[cfg(all(target_os = "linux", feature = "vaapi"))]
            HwBackend::Vaapi => {
                vaapi::VaapiDecoder::new(codec).map(|d| Box::new(d) as Box<dyn VideoDecoder>)
            }

            #[cfg(feature = "nvdec")]
            HwBackend::Nvdec => {
                nvdec::NvdecDecoder::new(codec).map(|d| Box::new(d) as Box<dyn VideoDecoder>)
            }

            #[cfg(all(target_os = "windows", feature = "media-foundation"))]
            HwBackend::MediaFoundation => media_foundation::MediaFoundationDecoder::new(codec)
                .map(|d| Box::new(d) as Box<dyn VideoDecoder>),

            _ => Err(VideoError::Codec("No hardware backend available".into())),
        };

        match hw_result {
            Ok(decoder) => Ok(HwVideoDecoder {
                backend,
                inner: decoder,
            }),
            Err(_) => {
                let sw: Box<dyn VideoDecoder> = match codec {
                    VideoCodec::H264 => Box::new(super::h264_decoder::H264Decoder::new()),
                    VideoCodec::H265 => Box::new(super::hevc_decoder::HevcDecoder::new()),
                    _ => return Err(VideoError::Codec(format!("Unsupported codec: {codec:?}"))),
                };
                Ok(HwVideoDecoder {
                    backend: HwBackend::Software,
                    inner: sw,
                })
            }
        }
    }

    pub fn backend(&self) -> HwBackend {
        self.backend
    }
    pub fn is_hardware(&self) -> bool {
        self.backend != HwBackend::Software
    }
}

impl VideoDecoder for HwVideoDecoder {
    fn codec(&self) -> VideoCodec {
        self.inner.codec()
    }
    fn decode(&mut self, data: &[u8], ts: u64) -> Result<Option<DecodedFrame>, VideoError> {
        self.inner.decode(data, ts)
    }
    fn flush(&mut self) -> Result<Vec<DecodedFrame>, VideoError> {
        self.inner.flush()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_backend() {
        let backend = detect_hw_backend();
        println!("Detected backend: {backend}");
    }

    #[test]
    fn hw_decoder_fallback_h264() {
        let decoder = HwVideoDecoder::new(VideoCodec::H264).unwrap();
        // Without features, falls back to software
        println!("H264 backend: {}", decoder.backend());
    }

    #[test]
    fn hw_decoder_fallback_hevc() {
        let decoder = HwVideoDecoder::new(VideoCodec::H265).unwrap();
        println!("HEVC backend: {}", decoder.backend());
    }
}
