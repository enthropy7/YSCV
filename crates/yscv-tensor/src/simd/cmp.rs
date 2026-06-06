//! Clamp and comparison operations with SIMD dispatch.

use super::*;

// ===========================================================================
// clamp: out[i] = data[i].clamp(min_val, max_val)
// ===========================================================================

/// SIMD-accelerated clamp using min/max intrinsics.
#[allow(unsafe_code)]
#[inline]
pub(crate) fn clamp_dispatch(data: &[f32], out: &mut [f32], min_val: f32, max_val: f32) {
    debug_assert_eq!(data.len(), out.len());

    if cfg!(miri) {
        for i in 0..data.len() {
            out[i] = data[i].clamp(min_val, max_val);
        }
        return;
    }

    #[cfg(target_arch = "aarch64")]
    {
        if yscv_cpu::host_cpu().features.neon {
            unsafe { clamp_neon(data, out, min_val, max_val) };
            return;
        }
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if yscv_cpu::host_cpu().features.avx {
            unsafe { clamp_avx(data, out, min_val, max_val) };
            return;
        }
        if yscv_cpu::host_cpu().features.sse {
            unsafe { clamp_sse(data, out, min_val, max_val) };
            return;
        }
    }

    for i in 0..data.len() {
        out[i] = data[i].clamp(min_val, max_val);
    }
}

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "neon")]
unsafe fn clamp_neon(data: &[f32], out: &mut [f32], min_val: f32, max_val: f32) {
    let len = data.len();
    let inp = data.as_ptr();
    let op = out.as_mut_ptr();
    let vmin = vdupq_n_f32(min_val);
    let vmax = vdupq_n_f32(max_val);
    let mut i = 0usize;

    while i + 4 <= len {
        let v = vld1q_f32(inp.add(i));
        vst1q_f32(op.add(i), vminq_f32(vmaxq_f32(v, vmin), vmax));
        i += 4;
    }

    while i < len {
        *op.add(i) = (*inp.add(i)).clamp(min_val, max_val);
        i += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "sse")]
unsafe fn clamp_sse(data: &[f32], out: &mut [f32], min_val: f32, max_val: f32) {
    let len = data.len();
    let inp = data.as_ptr();
    let op = out.as_mut_ptr();
    let vmin = _mm_set1_ps(min_val);
    let vmax = _mm_set1_ps(max_val);
    let mut i = 0usize;

    while i + 4 <= len {
        let v = _mm_loadu_ps(inp.add(i));
        _mm_storeu_ps(op.add(i), _mm_min_ps(_mm_max_ps(v, vmin), vmax));
        i += 4;
    }

    while i < len {
        *op.add(i) = (*inp.add(i)).clamp(min_val, max_val);
        i += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx")]
unsafe fn clamp_avx(data: &[f32], out: &mut [f32], min_val: f32, max_val: f32) {
    let len = data.len();
    let inp = data.as_ptr();
    let op = out.as_mut_ptr();
    let vmin = _mm256_set1_ps(min_val);
    let vmax = _mm256_set1_ps(max_val);
    let mut i = 0usize;

    while i + 8 <= len {
        let v = _mm256_loadu_ps(inp.add(i));
        _mm256_storeu_ps(op.add(i), _mm256_min_ps(_mm256_max_ps(v, vmin), vmax));
        i += 8;
    }

    if i < len {
        clamp_sse(&data[i..], &mut out[i..], min_val, max_val);
    }
}

// ===========================================================================
// Comparison dispatch: gt, lt, eq -> 1.0 / 0.0
// ===========================================================================

#[derive(Clone, Copy)]
pub(crate) enum CmpKind {
    Gt,
    Lt,
    Eq,
}

#[allow(unsafe_code)]
#[inline]
pub(crate) fn cmp_dispatch(lhs: &[f32], rhs: &[f32], out: &mut [f32], kind: CmpKind) {
    debug_assert_eq!(lhs.len(), rhs.len());
    debug_assert_eq!(lhs.len(), out.len());
    if cfg!(miri) {
        cmp_scalar(lhs, rhs, out, kind);
        return;
    }

    #[cfg(target_arch = "aarch64")]
    {
        if yscv_cpu::host_cpu().features.neon {
            unsafe { cmp_neon(lhs, rhs, out, lhs.len(), kind) };
            return;
        }
    }

    cmp_scalar(lhs, rhs, out, kind);
}

fn cmp_scalar(lhs: &[f32], rhs: &[f32], out: &mut [f32], kind: CmpKind) {
    for i in 0..lhs.len() {
        out[i] = match kind {
            CmpKind::Gt => {
                if lhs[i] > rhs[i] {
                    1.0
                } else {
                    0.0
                }
            }
            CmpKind::Lt => {
                if lhs[i] < rhs[i] {
                    1.0
                } else {
                    0.0
                }
            }
            CmpKind::Eq => {
                if (lhs[i] - rhs[i]).abs() < f32::EPSILON {
                    1.0
                } else {
                    0.0
                }
            }
        };
    }
}

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "neon")]
unsafe fn cmp_neon(lhs: &[f32], rhs: &[f32], out: &mut [f32], len: usize, kind: CmpKind) {
    use std::arch::aarch64::*;
    let lp = lhs.as_ptr();
    let rp = rhs.as_ptr();
    let op = out.as_mut_ptr();
    let one = vdupq_n_f32(1.0);
    let zero = vdupq_n_f32(0.0);
    let mut i = 0usize;

    while i + 16 <= len {
        for off in [0, 4, 8, 12] {
            let l = vld1q_f32(lp.add(i + off));
            let r = vld1q_f32(rp.add(i + off));
            let mask = match kind {
                CmpKind::Gt => vcgtq_f32(l, r),
                CmpKind::Lt => vcltq_f32(l, r),
                CmpKind::Eq => vceqq_f32(l, r),
            };
            vst1q_f32(op.add(i + off), vbslq_f32(mask, one, zero));
        }
        i += 16;
    }

    while i + 4 <= len {
        let l = vld1q_f32(lp.add(i));
        let r = vld1q_f32(rp.add(i));
        let mask = match kind {
            CmpKind::Gt => vcgtq_f32(l, r),
            CmpKind::Lt => vcltq_f32(l, r),
            CmpKind::Eq => vceqq_f32(l, r),
        };
        vst1q_f32(op.add(i), vbslq_f32(mask, one, zero));
        i += 4;
    }

    while i < len {
        out[i] = match kind {
            CmpKind::Gt => {
                if lhs[i] > rhs[i] {
                    1.0
                } else {
                    0.0
                }
            }
            CmpKind::Lt => {
                if lhs[i] < rhs[i] {
                    1.0
                } else {
                    0.0
                }
            }
            CmpKind::Eq => {
                if (lhs[i] - rhs[i]).abs() < f32::EPSILON {
                    1.0
                } else {
                    0.0
                }
            }
        };
        i += 1;
    }
}
