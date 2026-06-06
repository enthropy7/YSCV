use std::mem::MaybeUninit;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::{
    vcvtq_f32_u32, vdupq_n_f32, vget_high_u16, vget_low_u16, vld1_u8, vmovl_u8, vmovl_u16,
    vmulq_f32, vst1q_f32,
};
#[cfg(target_arch = "x86")]
use std::arch::x86::{
    _mm_cvtepi32_ps, _mm_loadl_epi64, _mm_loadu_si128, _mm_mul_ps, _mm_set1_ps, _mm_setzero_si128,
    _mm_storeu_ps, _mm_unpackhi_epi8, _mm_unpackhi_epi16, _mm_unpacklo_epi8, _mm_unpacklo_epi16,
    _mm256_cvtepi32_ps, _mm256_cvtepu8_epi32, _mm256_mul_ps, _mm256_set1_ps, _mm256_storeu_ps,
};
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{
    _mm_cvtepi32_ps, _mm_loadl_epi64, _mm_loadu_si128, _mm_mul_ps, _mm_set1_ps, _mm_setzero_si128,
    _mm_storeu_ps, _mm_unpackhi_epi8, _mm_unpackhi_epi16, _mm_unpacklo_epi8, _mm_unpacklo_epi16,
    _mm256_cvtepi32_ps, _mm256_cvtepu8_epi32, _mm256_mul_ps, _mm256_set1_ps, _mm256_storeu_ps,
};

#[cfg(any(test, feature = "native-camera"))]
use yscv_tensor::Tensor;

#[cfg(any(test, feature = "native-camera"))]
use crate::Frame;
use crate::VideoError;

const U8_TO_F32_SCALE: f32 = 1.0 / 255.0;

/// Converts raw RGB8 bytes to normalized f32 values (`0.0..=1.0`) in-place.
///
/// This API is intended for reusable decode buffers in hot frame loops.
#[allow(unsafe_code)]
pub fn normalize_rgb8_to_f32_inplace(rgb8: &[u8], out: &mut [f32]) -> Result<(), VideoError> {
    if out.len() != rgb8.len() {
        return Err(VideoError::NormalizedBufferSizeMismatch {
            expected: rgb8.len(),
            got: out.len(),
        });
    }
    // SAFETY:
    // - `out.as_mut_ptr()` points to `out.len()` valid writable `f32` slots.
    // - casting to `MaybeUninit<f32>` is sound because each slot is fully overwritten.
    // - `rgb8.as_ptr()` is valid for `rgb8.len()` reads.
    unsafe {
        normalize_u8_to_f32(
            out.as_mut_ptr().cast::<MaybeUninit<f32>>(),
            rgb8.as_ptr(),
            rgb8.len(),
        );
    }
    Ok(())
}

#[allow(unsafe_code)]
#[cfg(any(test, feature = "native-camera"))]
pub(crate) fn rgb8_bytes_to_frame(
    index: u64,
    timestamp_us: u64,
    width: usize,
    height: usize,
    bytes: &[u8],
) -> Result<Frame, VideoError> {
    let expected = width
        .checked_mul(height)
        .and_then(|pixels| pixels.checked_mul(3))
        .ok_or_else(|| {
            VideoError::Source(format!(
                "raw frame dimensions overflow for width={width}, height={height}"
            ))
        })?;
    if bytes.len() != expected {
        return Err(VideoError::RawFrameSizeMismatch {
            expected,
            got: bytes.len(),
        });
    }

    let mut data_uninit = Box::<[f32]>::new_uninit_slice(expected);
    normalize_rgb8_to_f32_uninit_inplace(bytes, &mut data_uninit)?;

    // SAFETY: `normalize_rgb8_to_f32_uninit_inplace` writes all slots in `data_uninit`.
    let data = unsafe { data_uninit.assume_init().into_vec() };

    let image = Tensor::from_vec(vec![height, width, 3], data)?;
    Frame::new(index, timestamp_us, image)
}

#[allow(unsafe_code)]
#[cfg(any(test, feature = "native-camera"))]
fn normalize_rgb8_to_f32_uninit_inplace(
    rgb8: &[u8],
    out: &mut [MaybeUninit<f32>],
) -> Result<(), VideoError> {
    if out.len() != rgb8.len() {
        return Err(VideoError::NormalizedBufferSizeMismatch {
            expected: rgb8.len(),
            got: out.len(),
        });
    }
    // SAFETY:
    // - `out.as_mut_ptr()` points to `out.len()` writable slots.
    // - `rgb8.as_ptr()` is valid for `rgb8.len()` reads.
    // - pointers are non-overlapping because element types differ and callers provide distinct buffers.
    unsafe {
        normalize_u8_to_f32(out.as_mut_ptr(), rgb8.as_ptr(), rgb8.len());
    }
    Ok(())
}

#[allow(unsafe_code)]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn normalize_u8_to_f32(dst: *mut MaybeUninit<f32>, src: *const u8, len: usize) {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if yscv_cpu::host_cpu().features.avx2 {
            // SAFETY: caller guarantees valid, non-overlapping in-bounds pointers for `len`.
            unsafe {
                normalize_u8_to_f32_avx2(dst, src, len);
            }
            return;
        }
        if yscv_cpu::host_cpu().features.sse2 {
            // SAFETY: caller guarantees valid, non-overlapping in-bounds pointers for `len`.
            unsafe {
                normalize_u8_to_f32_sse2(dst, src, len);
            }
            return;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if yscv_cpu::host_cpu().features.neon {
            // SAFETY: caller guarantees valid, non-overlapping in-bounds pointers for `len`.
            unsafe {
                normalize_u8_to_f32_neon(dst, src, len);
            }
            return;
        }
    }

    // SAFETY: caller guarantees valid, non-overlapping in-bounds pointers for `len`.
    unsafe {
        normalize_u8_to_f32_scalar(dst, src, len);
    }
}

#[allow(unsafe_code)]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn normalize_u8_to_f32_scalar(dst: *mut MaybeUninit<f32>, src: *const u8, len: usize) {
    let mut index = 0usize;
    while index + 8 <= len {
        dst.add(index)
            .write(MaybeUninit::new(*src.add(index) as f32 * U8_TO_F32_SCALE));
        dst.add(index + 1).write(MaybeUninit::new(
            *src.add(index + 1) as f32 * U8_TO_F32_SCALE,
        ));
        dst.add(index + 2).write(MaybeUninit::new(
            *src.add(index + 2) as f32 * U8_TO_F32_SCALE,
        ));
        dst.add(index + 3).write(MaybeUninit::new(
            *src.add(index + 3) as f32 * U8_TO_F32_SCALE,
        ));
        dst.add(index + 4).write(MaybeUninit::new(
            *src.add(index + 4) as f32 * U8_TO_F32_SCALE,
        ));
        dst.add(index + 5).write(MaybeUninit::new(
            *src.add(index + 5) as f32 * U8_TO_F32_SCALE,
        ));
        dst.add(index + 6).write(MaybeUninit::new(
            *src.add(index + 6) as f32 * U8_TO_F32_SCALE,
        ));
        dst.add(index + 7).write(MaybeUninit::new(
            *src.add(index + 7) as f32 * U8_TO_F32_SCALE,
        ));
        index += 8;
    }
    while index < len {
        dst.add(index)
            .write(MaybeUninit::new(*src.add(index) as f32 * U8_TO_F32_SCALE));
        index += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code)]
#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "sse2")]
unsafe fn normalize_u8_to_f32_sse2(dst: *mut MaybeUninit<f32>, src: *const u8, len: usize) {
    let zero = _mm_setzero_si128();
    let scale = _mm_set1_ps(U8_TO_F32_SCALE);

    let mut index = 0usize;
    while index + 16 <= len {
        let bytes = _mm_loadu_si128(src.add(index).cast());
        let low16 = _mm_unpacklo_epi8(bytes, zero);
        let high16 = _mm_unpackhi_epi8(bytes, zero);

        let low32_a = _mm_unpacklo_epi16(low16, zero);
        let low32_b = _mm_unpackhi_epi16(low16, zero);
        let high32_a = _mm_unpacklo_epi16(high16, zero);
        let high32_b = _mm_unpackhi_epi16(high16, zero);

        let f0 = _mm_mul_ps(_mm_cvtepi32_ps(low32_a), scale);
        let f1 = _mm_mul_ps(_mm_cvtepi32_ps(low32_b), scale);
        let f2 = _mm_mul_ps(_mm_cvtepi32_ps(high32_a), scale);
        let f3 = _mm_mul_ps(_mm_cvtepi32_ps(high32_b), scale);

        _mm_storeu_ps(dst.add(index).cast(), f0);
        _mm_storeu_ps(dst.add(index + 4).cast(), f1);
        _mm_storeu_ps(dst.add(index + 8).cast(), f2);
        _mm_storeu_ps(dst.add(index + 12).cast(), f3);

        index += 16;
    }

    if index < len {
        normalize_u8_to_f32_scalar(dst.add(index), src.add(index), len - index);
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code)]
#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx2")]
unsafe fn normalize_u8_to_f32_avx2(dst: *mut MaybeUninit<f32>, src: *const u8, len: usize) {
    let scale = _mm256_set1_ps(U8_TO_F32_SCALE);
    let mut index = 0usize;

    while index + 32 <= len {
        let b0 = _mm_loadl_epi64(src.add(index).cast());
        let b1 = _mm_loadl_epi64(src.add(index + 8).cast());
        let b2 = _mm_loadl_epi64(src.add(index + 16).cast());
        let b3 = _mm_loadl_epi64(src.add(index + 24).cast());

        let f0 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(b0)), scale);
        let f1 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(b1)), scale);
        let f2 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(b2)), scale);
        let f3 = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(b3)), scale);

        _mm256_storeu_ps(dst.add(index).cast(), f0);
        _mm256_storeu_ps(dst.add(index + 8).cast(), f1);
        _mm256_storeu_ps(dst.add(index + 16).cast(), f2);
        _mm256_storeu_ps(dst.add(index + 24).cast(), f3);

        index += 32;
    }

    if index < len {
        normalize_u8_to_f32_sse2(dst.add(index), src.add(index), len - index);
    }
}

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code)]
#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "neon")]
unsafe fn normalize_u8_to_f32_neon(dst: *mut MaybeUninit<f32>, src: *const u8, len: usize) {
    let scale = vdupq_n_f32(U8_TO_F32_SCALE);
    let mut index = 0usize;

    while index + 8 <= len {
        let bytes = vld1_u8(src.add(index));
        let widened_u16 = vmovl_u8(bytes);
        let lo_u32 = vmovl_u16(vget_low_u16(widened_u16));
        let hi_u32 = vmovl_u16(vget_high_u16(widened_u16));

        let lo_f32 = vmulq_f32(vcvtq_f32_u32(lo_u32), scale);
        let hi_f32 = vmulq_f32(vcvtq_f32_u32(hi_u32), scale);

        vst1q_f32(dst.add(index).cast(), lo_f32);
        vst1q_f32(dst.add(index + 4).cast(), hi_f32);
        index += 8;
    }

    if index < len {
        normalize_u8_to_f32_scalar(dst.add(index), src.add(index), len - index);
    }
}

#[cfg(feature = "native-camera")]
pub(crate) fn micros_to_u64(value: u128) -> u64 {
    value.min(u64::MAX as u128) as u64
}
