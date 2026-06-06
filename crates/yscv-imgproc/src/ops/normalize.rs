#![allow(unsafe_op_in_unsafe_fn)]

use yscv_tensor::Tensor;

use super::super::ImgProcError;
use super::super::shape::hwc_shape;

/// Per-channel normalize in HWC layout: `(x - mean[c]) / std[c]`.
///
/// Optimized: precomputes `inv_std = 1/std` to replace division with multiply,
/// and iterates by pixel (row-major) to avoid per-element modulo.
/// Uses SIMD (NEON/AVX/SSE) where available for all channel counts.
#[allow(unsafe_code)]
pub fn normalize(input: &Tensor, mean: &[f32], std: &[f32]) -> Result<Tensor, ImgProcError> {
    let (h, w, channels) = hwc_shape(input)?;
    if mean.len() != channels || std.len() != channels {
        return Err(ImgProcError::InvalidNormalizationParams {
            expected_channels: channels,
            mean_len: mean.len(),
            std_len: std.len(),
        });
    }
    for (channel, value) in std.iter().enumerate() {
        if *value == 0.0 {
            return Err(ImgProcError::ZeroStdAtChannel { channel });
        }
    }

    // Precompute reciprocal of std to replace division with multiplication
    let inv_std: Vec<f32> = std.iter().map(|&s| 1.0 / s).collect();

    let len = h * w * channels;
    let mut out = vec![0.0f32; len];

    let src = input.data();
    let num_pixels = h * w;

    // SAFETY: all pointer arithmetic stays in bounds (validated by shape).
    unsafe {
        let src_ptr = src.as_ptr();
        let dst_ptr = out.as_mut_ptr();

        // Fast path for common channel counts: avoid inner loop overhead
        match channels {
            3 => {
                normalize_3ch(src_ptr, dst_ptr, mean, &inv_std, num_pixels);
            }
            1 => {
                normalize_1ch(src_ptr, dst_ptr, mean[0], inv_std[0], len);
            }
            _ => {
                normalize_generic(src_ptr, dst_ptr, mean, &inv_std, channels, num_pixels);
            }
        }
    }

    Tensor::from_vec(vec![h, w, channels], out).map_err(Into::into)
}

#[allow(unsafe_code)]
unsafe fn normalize_3ch(
    src_ptr: *const f32,
    dst_ptr: *mut f32,
    mean: &[f32],
    inv_std: &[f32],
    num_pixels: usize,
) {
    let (m0, m1, m2) = (mean[0], mean[1], mean[2]);
    let (s0, s1, s2) = (inv_std[0], inv_std[1], inv_std[2]);

    #[cfg(target_arch = "aarch64")]
    if !cfg!(miri) && yscv_cpu::host_cpu().features.neon {
        use std::arch::aarch64::*;
        let vm = vld1q_f32([m0, m1, m2, 0.0].as_ptr());
        let vs = vld1q_f32([s0, s1, s2, 0.0].as_ptr());
        let full_quads = num_pixels / 4;
        for q in 0..full_quads {
            let base = q * 12;
            for p in 0..4 {
                let off = base + p * 3;
                let v = vld1q_f32(
                    [
                        *src_ptr.add(off),
                        *src_ptr.add(off + 1),
                        *src_ptr.add(off + 2),
                        0.0,
                    ]
                    .as_ptr(),
                );
                let r = vmulq_f32(vsubq_f32(v, vm), vs);
                *dst_ptr.add(off) = vgetq_lane_f32::<0>(r);
                *dst_ptr.add(off + 1) = vgetq_lane_f32::<1>(r);
                *dst_ptr.add(off + 2) = vgetq_lane_f32::<2>(r);
            }
        }
        for i in (full_quads * 4)..num_pixels {
            let off = i * 3;
            *dst_ptr.add(off) = (*src_ptr.add(off) - m0) * s0;
            *dst_ptr.add(off + 1) = (*src_ptr.add(off + 1) - m1) * s1;
            *dst_ptr.add(off + 2) = (*src_ptr.add(off + 2) - m2) * s2;
        }
        return;
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if !cfg!(miri) && yscv_cpu::host_cpu().features.avx {
        normalize_3ch_avx(src_ptr, dst_ptr, m0, m1, m2, s0, s1, s2, num_pixels);
        return;
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if !cfg!(miri) && yscv_cpu::host_cpu().features.sse2 {
        normalize_3ch_sse(src_ptr, dst_ptr, m0, m1, m2, s0, s1, s2, num_pixels);
        return;
    }

    // Scalar fallback
    for i in 0..num_pixels {
        let off = i * 3;
        *dst_ptr.add(off) = (*src_ptr.add(off) - m0) * s0;
        *dst_ptr.add(off + 1) = (*src_ptr.add(off + 1) - m1) * s1;
        *dst_ptr.add(off + 2) = (*src_ptr.add(off + 2) - m2) * s2;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx")]
#[allow(unsafe_code)]
unsafe fn normalize_3ch_avx(
    src_ptr: *const f32,
    dst_ptr: *mut f32,
    m0: f32,
    m1: f32,
    m2: f32,
    s0: f32,
    s1: f32,
    s2: f32,
    num_pixels: usize,
) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    // Process 8 pixels (24 floats) at a time.
    // Load 3x8 floats, apply (x - mean) * inv_std per channel.
    // Pack mean/inv_std as repeating pattern: [m0,m1,m2,m0,m1,m2,m0,m1]
    let vm_a = _mm256_set_ps(m1, m0, m2, m1, m0, m2, m1, m0);
    let vm_b = _mm256_set_ps(m2, m1, m0, m2, m1, m0, m2, m1);
    let vm_c = _mm256_set_ps(m0, m2, m1, m0, m2, m1, m0, m2);
    let vs_a = _mm256_set_ps(s1, s0, s2, s1, s0, s2, s1, s0);
    let vs_b = _mm256_set_ps(s2, s1, s0, s2, s1, s0, s2, s1);
    let vs_c = _mm256_set_ps(s0, s2, s1, s0, s2, s1, s0, s2);

    let full_groups = num_pixels / 8;
    for g in 0..full_groups {
        let base = g * 24;
        // Load 24 floats as 3 x __m256
        let a = _mm256_loadu_ps(src_ptr.add(base));
        let b = _mm256_loadu_ps(src_ptr.add(base + 8));
        let c = _mm256_loadu_ps(src_ptr.add(base + 16));

        let ra = _mm256_mul_ps(_mm256_sub_ps(a, vm_a), vs_a);
        let rb = _mm256_mul_ps(_mm256_sub_ps(b, vm_b), vs_b);
        let rc = _mm256_mul_ps(_mm256_sub_ps(c, vm_c), vs_c);

        _mm256_storeu_ps(dst_ptr.add(base), ra);
        _mm256_storeu_ps(dst_ptr.add(base + 8), rb);
        _mm256_storeu_ps(dst_ptr.add(base + 16), rc);
    }
    // Remainder pixels
    for i in (full_groups * 8)..num_pixels {
        let off = i * 3;
        *dst_ptr.add(off) = (*src_ptr.add(off) - m0) * s0;
        *dst_ptr.add(off + 1) = (*src_ptr.add(off + 1) - m1) * s1;
        *dst_ptr.add(off + 2) = (*src_ptr.add(off + 2) - m2) * s2;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
#[allow(unsafe_code)]
unsafe fn normalize_3ch_sse(
    src_ptr: *const f32,
    dst_ptr: *mut f32,
    m0: f32,
    m1: f32,
    m2: f32,
    s0: f32,
    s1: f32,
    s2: f32,
    num_pixels: usize,
) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    // Process 4 pixels (12 floats) at a time as 3 x __m128
    let vm_a = _mm_set_ps(m0, m2, m1, m0);
    let vm_b = _mm_set_ps(m1, m0, m2, m1);
    let vm_c = _mm_set_ps(m2, m1, m0, m2);
    let vs_a = _mm_set_ps(s0, s2, s1, s0);
    let vs_b = _mm_set_ps(s1, s0, s2, s1);
    let vs_c = _mm_set_ps(s2, s1, s0, s2);

    let full_groups = num_pixels / 4;
    for g in 0..full_groups {
        let base = g * 12;
        let a = _mm_loadu_ps(src_ptr.add(base));
        let b = _mm_loadu_ps(src_ptr.add(base + 4));
        let c = _mm_loadu_ps(src_ptr.add(base + 8));

        let ra = _mm_mul_ps(_mm_sub_ps(a, vm_a), vs_a);
        let rb = _mm_mul_ps(_mm_sub_ps(b, vm_b), vs_b);
        let rc = _mm_mul_ps(_mm_sub_ps(c, vm_c), vs_c);

        _mm_storeu_ps(dst_ptr.add(base), ra);
        _mm_storeu_ps(dst_ptr.add(base + 4), rb);
        _mm_storeu_ps(dst_ptr.add(base + 8), rc);
    }
    for i in (full_groups * 4)..num_pixels {
        let off = i * 3;
        *dst_ptr.add(off) = (*src_ptr.add(off) - m0) * s0;
        *dst_ptr.add(off + 1) = (*src_ptr.add(off + 1) - m1) * s1;
        *dst_ptr.add(off + 2) = (*src_ptr.add(off + 2) - m2) * s2;
    }
}

#[allow(unsafe_code)]
unsafe fn normalize_1ch(
    src_ptr: *const f32,
    dst_ptr: *mut f32,
    mean: f32,
    inv_std: f32,
    len: usize,
) {
    #[cfg(target_arch = "aarch64")]
    if !cfg!(miri) && yscv_cpu::host_cpu().features.neon {
        use std::arch::aarch64::*;
        let vm = vdupq_n_f32(mean);
        let vs = vdupq_n_f32(inv_std);
        let mut i = 0usize;
        while i + 4 <= len {
            let v = vld1q_f32(src_ptr.add(i));
            let r = vmulq_f32(vsubq_f32(v, vm), vs);
            vst1q_f32(dst_ptr.add(i), r);
            i += 4;
        }
        while i < len {
            *dst_ptr.add(i) = (*src_ptr.add(i) - mean) * inv_std;
            i += 1;
        }
        return;
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if !cfg!(miri) && yscv_cpu::host_cpu().features.avx {
        normalize_1ch_avx(src_ptr, dst_ptr, mean, inv_std, len);
        return;
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if !cfg!(miri) && yscv_cpu::host_cpu().features.sse2 {
        normalize_1ch_sse(src_ptr, dst_ptr, mean, inv_std, len);
        return;
    }

    // Scalar fallback
    for i in 0..len {
        *dst_ptr.add(i) = (*src_ptr.add(i) - mean) * inv_std;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx")]
#[allow(unsafe_code)]
unsafe fn normalize_1ch_avx(
    src_ptr: *const f32,
    dst_ptr: *mut f32,
    mean: f32,
    inv_std: f32,
    len: usize,
) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let vm = _mm256_set1_ps(mean);
    let vs = _mm256_set1_ps(inv_std);
    let mut i = 0usize;
    while i + 8 <= len {
        let v = _mm256_loadu_ps(src_ptr.add(i));
        let r = _mm256_mul_ps(_mm256_sub_ps(v, vm), vs);
        _mm256_storeu_ps(dst_ptr.add(i), r);
        i += 8;
    }
    while i < len {
        *dst_ptr.add(i) = (*src_ptr.add(i) - mean) * inv_std;
        i += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
#[allow(unsafe_code)]
unsafe fn normalize_1ch_sse(
    src_ptr: *const f32,
    dst_ptr: *mut f32,
    mean: f32,
    inv_std: f32,
    len: usize,
) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let vm = _mm_set1_ps(mean);
    let vs = _mm_set1_ps(inv_std);
    let mut i = 0usize;
    while i + 4 <= len {
        let v = _mm_loadu_ps(src_ptr.add(i));
        let r = _mm_mul_ps(_mm_sub_ps(v, vm), vs);
        _mm_storeu_ps(dst_ptr.add(i), r);
        i += 4;
    }
    while i < len {
        *dst_ptr.add(i) = (*src_ptr.add(i) - mean) * inv_std;
        i += 1;
    }
}

#[allow(unsafe_code)]
unsafe fn normalize_generic(
    src_ptr: *const f32,
    dst_ptr: *mut f32,
    mean: &[f32],
    inv_std: &[f32],
    channels: usize,
    num_pixels: usize,
) {
    #[cfg(target_arch = "aarch64")]
    if !cfg!(miri) && yscv_cpu::host_cpu().features.neon {
        use std::arch::aarch64::*;
        let simd_end = channels & !3;
        for px in 0..num_pixels {
            let base = px * channels;
            let mut c = 0usize;
            while c < simd_end {
                let off = base + c;
                let v = vld1q_f32(src_ptr.add(off));
                let vm = vld1q_f32(mean.as_ptr().add(c));
                let vs = vld1q_f32(inv_std.as_ptr().add(c));
                let r = vmulq_f32(vsubq_f32(v, vm), vs);
                vst1q_f32(dst_ptr.add(off), r);
                c += 4;
            }
            while c < channels {
                let off = base + c;
                *dst_ptr.add(off) = (*src_ptr.add(off) - mean[c]) * inv_std[c];
                c += 1;
            }
        }
        return;
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if !cfg!(miri) && yscv_cpu::host_cpu().features.avx {
        normalize_generic_avx(src_ptr, dst_ptr, mean, inv_std, channels, num_pixels);
        return;
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if !cfg!(miri) && yscv_cpu::host_cpu().features.sse2 {
        normalize_generic_sse(src_ptr, dst_ptr, mean, inv_std, channels, num_pixels);
        return;
    }

    // Scalar fallback
    for px in 0..num_pixels {
        let base = px * channels;
        for c in 0..channels {
            let off = base + c;
            *dst_ptr.add(off) = (*src_ptr.add(off) - mean[c]) * inv_std[c];
        }
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx")]
#[allow(unsafe_code)]
unsafe fn normalize_generic_avx(
    src_ptr: *const f32,
    dst_ptr: *mut f32,
    mean: &[f32],
    inv_std: &[f32],
    channels: usize,
    num_pixels: usize,
) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let simd_end = channels & !7;
    for px in 0..num_pixels {
        let base = px * channels;
        let mut c = 0usize;
        while c < simd_end {
            let off = base + c;
            let v = _mm256_loadu_ps(src_ptr.add(off));
            let vm = _mm256_loadu_ps(mean.as_ptr().add(c));
            let vs = _mm256_loadu_ps(inv_std.as_ptr().add(c));
            let r = _mm256_mul_ps(_mm256_sub_ps(v, vm), vs);
            _mm256_storeu_ps(dst_ptr.add(off), r);
            c += 8;
        }
        while c < channels {
            let off = base + c;
            *dst_ptr.add(off) = (*src_ptr.add(off) - mean[c]) * inv_std[c];
            c += 1;
        }
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
#[allow(unsafe_code)]
unsafe fn normalize_generic_sse(
    src_ptr: *const f32,
    dst_ptr: *mut f32,
    mean: &[f32],
    inv_std: &[f32],
    channels: usize,
    num_pixels: usize,
) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let simd_end = channels & !3;
    for px in 0..num_pixels {
        let base = px * channels;
        let mut c = 0usize;
        while c < simd_end {
            let off = base + c;
            let v = _mm_loadu_ps(src_ptr.add(off));
            let vm = _mm_loadu_ps(mean.as_ptr().add(c));
            let vs = _mm_loadu_ps(inv_std.as_ptr().add(c));
            let r = _mm_mul_ps(_mm_sub_ps(v, vm), vs);
            _mm_storeu_ps(dst_ptr.add(off), r);
            c += 4;
        }
        while c < channels {
            let off = base + c;
            *dst_ptr.add(off) = (*src_ptr.add(off) - mean[c]) * inv_std[c];
            c += 1;
        }
    }
}
