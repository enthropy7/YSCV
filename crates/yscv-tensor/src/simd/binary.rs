//! Binary same-shape operations with SIMD dispatch.

use super::*;

#[derive(Clone, Copy)]
pub(crate) enum BinaryKind {
    Add,
    Sub,
    Mul,
    Div,
}

#[allow(unsafe_code, unreachable_code)]
#[inline]
pub(crate) fn binary_dispatch(lhs: &[f32], rhs: &[f32], out: &mut [f32], kind: BinaryKind) {
    debug_assert_eq!(lhs.len(), rhs.len());
    debug_assert_eq!(lhs.len(), out.len());

    if cfg!(miri) {
        binary_scalar(lhs, rhs, out, kind);
        return;
    }

    // macOS aarch64: use Apple vDSP for add/sub/mul/div.
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        let n = lhs.len() as u32;
        // SAFETY: vDSP functions operate on contiguous slices of equal length.
        unsafe {
            match kind {
                BinaryKind::Add => {
                    vDSP_vadd(lhs.as_ptr(), 1, rhs.as_ptr(), 1, out.as_mut_ptr(), 1, n)
                }
                BinaryKind::Sub => {
                    vDSP_vsub(rhs.as_ptr(), 1, lhs.as_ptr(), 1, out.as_mut_ptr(), 1, n)
                }
                BinaryKind::Mul => {
                    vDSP_vmul(lhs.as_ptr(), 1, rhs.as_ptr(), 1, out.as_mut_ptr(), 1, n)
                }
                BinaryKind::Div => {
                    vDSP_vdiv(rhs.as_ptr(), 1, lhs.as_ptr(), 1, out.as_mut_ptr(), 1, n)
                }
            }
        }
        return;
    }

    // x86/x86_64 with MKL: use Intel VML for add/sub/mul/div (heavily optimized).
    #[cfg(all(feature = "mkl", any(target_arch = "x86", target_arch = "x86_64")))]
    {
        let n = lhs.len() as i32;
        // SAFETY: VML functions read `n` floats from contiguous slices and write to `out`.
        unsafe {
            match kind {
                BinaryKind::Add => vsAdd(n, lhs.as_ptr(), rhs.as_ptr(), out.as_mut_ptr()),
                BinaryKind::Sub => vsSub(n, lhs.as_ptr(), rhs.as_ptr(), out.as_mut_ptr()),
                BinaryKind::Mul => vsMul(n, lhs.as_ptr(), rhs.as_ptr(), out.as_mut_ptr()),
                BinaryKind::Div => vsDiv(n, lhs.as_ptr(), rhs.as_ptr(), out.as_mut_ptr()),
            }
        }
        return;
    }

    // aarch64 Linux with ARMPL: use ARM Performance Libraries for add/sub/mul/div.
    #[cfg(all(feature = "armpl", target_arch = "aarch64", not(target_os = "macos")))]
    {
        let n = lhs.len() as i32;
        // SAFETY: ARMPL functions read `n` floats from contiguous slices and write to `out`.
        unsafe {
            match kind {
                BinaryKind::Add => armpl_svadd_f32(n, lhs.as_ptr(), rhs.as_ptr(), out.as_mut_ptr()),
                BinaryKind::Sub => armpl_svsub_f32(n, lhs.as_ptr(), rhs.as_ptr(), out.as_mut_ptr()),
                BinaryKind::Mul => armpl_svmul_f32(n, lhs.as_ptr(), rhs.as_ptr(), out.as_mut_ptr()),
                BinaryKind::Div => armpl_svdiv_f32(n, lhs.as_ptr(), rhs.as_ptr(), out.as_mut_ptr()),
            }
        }
        return;
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if yscv_cpu::host_cpu().features.avx {
            unsafe { binary_avx(lhs, rhs, out, kind) };
            return;
        }
        if yscv_cpu::host_cpu().features.sse {
            unsafe { binary_sse(lhs, rhs, out, kind) };
            return;
        }
    }

    #[cfg(all(target_arch = "aarch64", not(target_os = "macos")))]
    {
        if yscv_cpu::host_cpu().features.neon {
            unsafe { binary_neon(lhs, rhs, out, kind) };
            return;
        }
    }

    binary_scalar(lhs, rhs, out, kind);
}

fn binary_scalar(lhs: &[f32], rhs: &[f32], out: &mut [f32], kind: BinaryKind) {
    match kind {
        BinaryKind::Add => {
            for i in 0..lhs.len() {
                out[i] = lhs[i] + rhs[i];
            }
        }
        BinaryKind::Sub => {
            for i in 0..lhs.len() {
                out[i] = lhs[i] - rhs[i];
            }
        }
        BinaryKind::Mul => {
            for i in 0..lhs.len() {
                out[i] = lhs[i] * rhs[i];
            }
        }
        BinaryKind::Div => {
            for i in 0..lhs.len() {
                out[i] = lhs[i] / rhs[i];
            }
        }
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "sse")]
unsafe fn binary_sse(lhs: &[f32], rhs: &[f32], out: &mut [f32], kind: BinaryKind) {
    let len = lhs.len();
    let lp = lhs.as_ptr();
    let rp = rhs.as_ptr();
    let op = out.as_mut_ptr();
    let mut i = 0usize;

    while i + 4 <= len {
        let l = _mm_loadu_ps(lp.add(i));
        let r = _mm_loadu_ps(rp.add(i));
        let result = match kind {
            BinaryKind::Add => _mm_add_ps(l, r),
            BinaryKind::Sub => _mm_sub_ps(l, r),
            BinaryKind::Mul => _mm_mul_ps(l, r),
            BinaryKind::Div => {
                #[cfg(target_arch = "x86")]
                use std::arch::x86::_mm_div_ps;
                #[cfg(target_arch = "x86_64")]
                use std::arch::x86_64::_mm_div_ps;
                _mm_div_ps(l, r)
            }
        };
        _mm_storeu_ps(op.add(i), result);
        i += 4;
    }

    while i < len {
        *op.add(i) = match kind {
            BinaryKind::Add => *lp.add(i) + *rp.add(i),
            BinaryKind::Sub => *lp.add(i) - *rp.add(i),
            BinaryKind::Mul => *lp.add(i) * *rp.add(i),
            BinaryKind::Div => *lp.add(i) / *rp.add(i),
        };
        i += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx")]
unsafe fn binary_avx(lhs: &[f32], rhs: &[f32], out: &mut [f32], kind: BinaryKind) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::_mm256_div_ps;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::_mm256_div_ps;

    let len = lhs.len();
    let lp = lhs.as_ptr();
    let rp = rhs.as_ptr();
    let op = out.as_mut_ptr();
    let mut i = 0usize;

    // 8x unrolled: process 64 floats per iteration for better ILP.
    // Loads are interleaved with compute/stores to keep the OoO pipeline busy.
    match kind {
        BinaryKind::Add => {
            while i + 64 <= len {
                let a0 = _mm256_loadu_ps(lp.add(i));
                let a1 = _mm256_loadu_ps(lp.add(i + 8));
                let b0 = _mm256_loadu_ps(rp.add(i));
                let b1 = _mm256_loadu_ps(rp.add(i + 8));
                let a2 = _mm256_loadu_ps(lp.add(i + 16));
                let a3 = _mm256_loadu_ps(lp.add(i + 24));
                _mm256_storeu_ps(op.add(i), _mm256_add_ps(a0, b0));
                _mm256_storeu_ps(op.add(i + 8), _mm256_add_ps(a1, b1));
                let b2 = _mm256_loadu_ps(rp.add(i + 16));
                let b3 = _mm256_loadu_ps(rp.add(i + 24));
                let a4 = _mm256_loadu_ps(lp.add(i + 32));
                let a5 = _mm256_loadu_ps(lp.add(i + 40));
                _mm256_storeu_ps(op.add(i + 16), _mm256_add_ps(a2, b2));
                _mm256_storeu_ps(op.add(i + 24), _mm256_add_ps(a3, b3));
                let b4 = _mm256_loadu_ps(rp.add(i + 32));
                let b5 = _mm256_loadu_ps(rp.add(i + 40));
                let a6 = _mm256_loadu_ps(lp.add(i + 48));
                let a7 = _mm256_loadu_ps(lp.add(i + 56));
                _mm256_storeu_ps(op.add(i + 32), _mm256_add_ps(a4, b4));
                _mm256_storeu_ps(op.add(i + 40), _mm256_add_ps(a5, b5));
                let b6 = _mm256_loadu_ps(rp.add(i + 48));
                let b7 = _mm256_loadu_ps(rp.add(i + 56));
                _mm256_storeu_ps(op.add(i + 48), _mm256_add_ps(a6, b6));
                _mm256_storeu_ps(op.add(i + 56), _mm256_add_ps(a7, b7));
                i += 64;
            }
        }
        BinaryKind::Sub => {
            while i + 64 <= len {
                let a0 = _mm256_loadu_ps(lp.add(i));
                let a1 = _mm256_loadu_ps(lp.add(i + 8));
                let b0 = _mm256_loadu_ps(rp.add(i));
                let b1 = _mm256_loadu_ps(rp.add(i + 8));
                let a2 = _mm256_loadu_ps(lp.add(i + 16));
                let a3 = _mm256_loadu_ps(lp.add(i + 24));
                _mm256_storeu_ps(op.add(i), _mm256_sub_ps(a0, b0));
                _mm256_storeu_ps(op.add(i + 8), _mm256_sub_ps(a1, b1));
                let b2 = _mm256_loadu_ps(rp.add(i + 16));
                let b3 = _mm256_loadu_ps(rp.add(i + 24));
                let a4 = _mm256_loadu_ps(lp.add(i + 32));
                let a5 = _mm256_loadu_ps(lp.add(i + 40));
                _mm256_storeu_ps(op.add(i + 16), _mm256_sub_ps(a2, b2));
                _mm256_storeu_ps(op.add(i + 24), _mm256_sub_ps(a3, b3));
                let b4 = _mm256_loadu_ps(rp.add(i + 32));
                let b5 = _mm256_loadu_ps(rp.add(i + 40));
                let a6 = _mm256_loadu_ps(lp.add(i + 48));
                let a7 = _mm256_loadu_ps(lp.add(i + 56));
                _mm256_storeu_ps(op.add(i + 32), _mm256_sub_ps(a4, b4));
                _mm256_storeu_ps(op.add(i + 40), _mm256_sub_ps(a5, b5));
                let b6 = _mm256_loadu_ps(rp.add(i + 48));
                let b7 = _mm256_loadu_ps(rp.add(i + 56));
                _mm256_storeu_ps(op.add(i + 48), _mm256_sub_ps(a6, b6));
                _mm256_storeu_ps(op.add(i + 56), _mm256_sub_ps(a7, b7));
                i += 64;
            }
        }
        BinaryKind::Mul => {
            while i + 64 <= len {
                let a0 = _mm256_loadu_ps(lp.add(i));
                let a1 = _mm256_loadu_ps(lp.add(i + 8));
                let b0 = _mm256_loadu_ps(rp.add(i));
                let b1 = _mm256_loadu_ps(rp.add(i + 8));
                let a2 = _mm256_loadu_ps(lp.add(i + 16));
                let a3 = _mm256_loadu_ps(lp.add(i + 24));
                _mm256_storeu_ps(op.add(i), _mm256_mul_ps(a0, b0));
                _mm256_storeu_ps(op.add(i + 8), _mm256_mul_ps(a1, b1));
                let b2 = _mm256_loadu_ps(rp.add(i + 16));
                let b3 = _mm256_loadu_ps(rp.add(i + 24));
                let a4 = _mm256_loadu_ps(lp.add(i + 32));
                let a5 = _mm256_loadu_ps(lp.add(i + 40));
                _mm256_storeu_ps(op.add(i + 16), _mm256_mul_ps(a2, b2));
                _mm256_storeu_ps(op.add(i + 24), _mm256_mul_ps(a3, b3));
                let b4 = _mm256_loadu_ps(rp.add(i + 32));
                let b5 = _mm256_loadu_ps(rp.add(i + 40));
                let a6 = _mm256_loadu_ps(lp.add(i + 48));
                let a7 = _mm256_loadu_ps(lp.add(i + 56));
                _mm256_storeu_ps(op.add(i + 32), _mm256_mul_ps(a4, b4));
                _mm256_storeu_ps(op.add(i + 40), _mm256_mul_ps(a5, b5));
                let b6 = _mm256_loadu_ps(rp.add(i + 48));
                let b7 = _mm256_loadu_ps(rp.add(i + 56));
                _mm256_storeu_ps(op.add(i + 48), _mm256_mul_ps(a6, b6));
                _mm256_storeu_ps(op.add(i + 56), _mm256_mul_ps(a7, b7));
                i += 64;
            }
        }
        BinaryKind::Div => {
            while i + 64 <= len {
                let a0 = _mm256_loadu_ps(lp.add(i));
                let a1 = _mm256_loadu_ps(lp.add(i + 8));
                let b0 = _mm256_loadu_ps(rp.add(i));
                let b1 = _mm256_loadu_ps(rp.add(i + 8));
                let a2 = _mm256_loadu_ps(lp.add(i + 16));
                let a3 = _mm256_loadu_ps(lp.add(i + 24));
                _mm256_storeu_ps(op.add(i), _mm256_div_ps(a0, b0));
                _mm256_storeu_ps(op.add(i + 8), _mm256_div_ps(a1, b1));
                let b2 = _mm256_loadu_ps(rp.add(i + 16));
                let b3 = _mm256_loadu_ps(rp.add(i + 24));
                let a4 = _mm256_loadu_ps(lp.add(i + 32));
                let a5 = _mm256_loadu_ps(lp.add(i + 40));
                _mm256_storeu_ps(op.add(i + 16), _mm256_div_ps(a2, b2));
                _mm256_storeu_ps(op.add(i + 24), _mm256_div_ps(a3, b3));
                let b4 = _mm256_loadu_ps(rp.add(i + 32));
                let b5 = _mm256_loadu_ps(rp.add(i + 40));
                let a6 = _mm256_loadu_ps(lp.add(i + 48));
                let a7 = _mm256_loadu_ps(lp.add(i + 56));
                _mm256_storeu_ps(op.add(i + 32), _mm256_div_ps(a4, b4));
                _mm256_storeu_ps(op.add(i + 40), _mm256_div_ps(a5, b5));
                let b6 = _mm256_loadu_ps(rp.add(i + 48));
                let b7 = _mm256_loadu_ps(rp.add(i + 56));
                _mm256_storeu_ps(op.add(i + 48), _mm256_div_ps(a6, b6));
                _mm256_storeu_ps(op.add(i + 56), _mm256_div_ps(a7, b7));
                i += 64;
            }
        }
    }

    // Handle remaining 32-element chunks (4x unrolled fallback)
    match kind {
        BinaryKind::Add => {
            while i + 32 <= len {
                let a0 = _mm256_loadu_ps(lp.add(i));
                let b0 = _mm256_loadu_ps(rp.add(i));
                _mm256_storeu_ps(op.add(i), _mm256_add_ps(a0, b0));
                let a1 = _mm256_loadu_ps(lp.add(i + 8));
                let b1 = _mm256_loadu_ps(rp.add(i + 8));
                _mm256_storeu_ps(op.add(i + 8), _mm256_add_ps(a1, b1));
                let a2 = _mm256_loadu_ps(lp.add(i + 16));
                let b2 = _mm256_loadu_ps(rp.add(i + 16));
                _mm256_storeu_ps(op.add(i + 16), _mm256_add_ps(a2, b2));
                let a3 = _mm256_loadu_ps(lp.add(i + 24));
                let b3 = _mm256_loadu_ps(rp.add(i + 24));
                _mm256_storeu_ps(op.add(i + 24), _mm256_add_ps(a3, b3));
                i += 32;
            }
        }
        BinaryKind::Sub => {
            while i + 32 <= len {
                let a0 = _mm256_loadu_ps(lp.add(i));
                let b0 = _mm256_loadu_ps(rp.add(i));
                _mm256_storeu_ps(op.add(i), _mm256_sub_ps(a0, b0));
                let a1 = _mm256_loadu_ps(lp.add(i + 8));
                let b1 = _mm256_loadu_ps(rp.add(i + 8));
                _mm256_storeu_ps(op.add(i + 8), _mm256_sub_ps(a1, b1));
                let a2 = _mm256_loadu_ps(lp.add(i + 16));
                let b2 = _mm256_loadu_ps(rp.add(i + 16));
                _mm256_storeu_ps(op.add(i + 16), _mm256_sub_ps(a2, b2));
                let a3 = _mm256_loadu_ps(lp.add(i + 24));
                let b3 = _mm256_loadu_ps(rp.add(i + 24));
                _mm256_storeu_ps(op.add(i + 24), _mm256_sub_ps(a3, b3));
                i += 32;
            }
        }
        BinaryKind::Mul => {
            while i + 32 <= len {
                let a0 = _mm256_loadu_ps(lp.add(i));
                let b0 = _mm256_loadu_ps(rp.add(i));
                _mm256_storeu_ps(op.add(i), _mm256_mul_ps(a0, b0));
                let a1 = _mm256_loadu_ps(lp.add(i + 8));
                let b1 = _mm256_loadu_ps(rp.add(i + 8));
                _mm256_storeu_ps(op.add(i + 8), _mm256_mul_ps(a1, b1));
                let a2 = _mm256_loadu_ps(lp.add(i + 16));
                let b2 = _mm256_loadu_ps(rp.add(i + 16));
                _mm256_storeu_ps(op.add(i + 16), _mm256_mul_ps(a2, b2));
                let a3 = _mm256_loadu_ps(lp.add(i + 24));
                let b3 = _mm256_loadu_ps(rp.add(i + 24));
                _mm256_storeu_ps(op.add(i + 24), _mm256_mul_ps(a3, b3));
                i += 32;
            }
        }
        BinaryKind::Div => {
            while i + 32 <= len {
                let a0 = _mm256_loadu_ps(lp.add(i));
                let b0 = _mm256_loadu_ps(rp.add(i));
                _mm256_storeu_ps(op.add(i), _mm256_div_ps(a0, b0));
                let a1 = _mm256_loadu_ps(lp.add(i + 8));
                let b1 = _mm256_loadu_ps(rp.add(i + 8));
                _mm256_storeu_ps(op.add(i + 8), _mm256_div_ps(a1, b1));
                let a2 = _mm256_loadu_ps(lp.add(i + 16));
                let b2 = _mm256_loadu_ps(rp.add(i + 16));
                _mm256_storeu_ps(op.add(i + 16), _mm256_div_ps(a2, b2));
                let a3 = _mm256_loadu_ps(lp.add(i + 24));
                let b3 = _mm256_loadu_ps(rp.add(i + 24));
                _mm256_storeu_ps(op.add(i + 24), _mm256_div_ps(a3, b3));
                i += 32;
            }
        }
    }

    // Handle remaining 8-element chunks
    while i + 8 <= len {
        let l = _mm256_loadu_ps(lp.add(i));
        let r = _mm256_loadu_ps(rp.add(i));
        let result = match kind {
            BinaryKind::Add => _mm256_add_ps(l, r),
            BinaryKind::Sub => _mm256_sub_ps(l, r),
            BinaryKind::Mul => _mm256_mul_ps(l, r),
            BinaryKind::Div => _mm256_div_ps(l, r),
        };
        _mm256_storeu_ps(op.add(i), result);
        i += 8;
    }

    if i < len {
        binary_sse(&lhs[i..], &rhs[i..], &mut out[i..], kind);
    }
}

#[cfg(all(target_arch = "aarch64", not(target_os = "macos")))]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "neon")]
unsafe fn binary_neon(lhs: &[f32], rhs: &[f32], out: &mut [f32], kind: BinaryKind) {
    use std::arch::aarch64::vdivq_f32;

    let len = lhs.len();
    let lp = lhs.as_ptr();
    let rp = rhs.as_ptr();
    let op = out.as_mut_ptr();
    let mut i = 0usize;

    // 8x unrolled: process 32 floats per iteration for better ILP.
    // Loads are interleaved with compute/stores to keep the OoO pipeline busy.
    match kind {
        BinaryKind::Add => {
            while i + 32 <= len {
                let a0 = vld1q_f32(lp.add(i));
                let a1 = vld1q_f32(lp.add(i + 4));
                let b0 = vld1q_f32(rp.add(i));
                let b1 = vld1q_f32(rp.add(i + 4));
                let a2 = vld1q_f32(lp.add(i + 8));
                let a3 = vld1q_f32(lp.add(i + 12));
                vst1q_f32(op.add(i), vaddq_f32(a0, b0));
                vst1q_f32(op.add(i + 4), vaddq_f32(a1, b1));
                let b2 = vld1q_f32(rp.add(i + 8));
                let b3 = vld1q_f32(rp.add(i + 12));
                let a4 = vld1q_f32(lp.add(i + 16));
                let a5 = vld1q_f32(lp.add(i + 20));
                vst1q_f32(op.add(i + 8), vaddq_f32(a2, b2));
                vst1q_f32(op.add(i + 12), vaddq_f32(a3, b3));
                let b4 = vld1q_f32(rp.add(i + 16));
                let b5 = vld1q_f32(rp.add(i + 20));
                let a6 = vld1q_f32(lp.add(i + 24));
                let a7 = vld1q_f32(lp.add(i + 28));
                vst1q_f32(op.add(i + 16), vaddq_f32(a4, b4));
                vst1q_f32(op.add(i + 20), vaddq_f32(a5, b5));
                let b6 = vld1q_f32(rp.add(i + 24));
                let b7 = vld1q_f32(rp.add(i + 28));
                vst1q_f32(op.add(i + 24), vaddq_f32(a6, b6));
                vst1q_f32(op.add(i + 28), vaddq_f32(a7, b7));
                i += 32;
            }
        }
        BinaryKind::Sub => {
            while i + 32 <= len {
                let a0 = vld1q_f32(lp.add(i));
                let a1 = vld1q_f32(lp.add(i + 4));
                let b0 = vld1q_f32(rp.add(i));
                let b1 = vld1q_f32(rp.add(i + 4));
                let a2 = vld1q_f32(lp.add(i + 8));
                let a3 = vld1q_f32(lp.add(i + 12));
                vst1q_f32(op.add(i), vsubq_f32(a0, b0));
                vst1q_f32(op.add(i + 4), vsubq_f32(a1, b1));
                let b2 = vld1q_f32(rp.add(i + 8));
                let b3 = vld1q_f32(rp.add(i + 12));
                let a4 = vld1q_f32(lp.add(i + 16));
                let a5 = vld1q_f32(lp.add(i + 20));
                vst1q_f32(op.add(i + 8), vsubq_f32(a2, b2));
                vst1q_f32(op.add(i + 12), vsubq_f32(a3, b3));
                let b4 = vld1q_f32(rp.add(i + 16));
                let b5 = vld1q_f32(rp.add(i + 20));
                let a6 = vld1q_f32(lp.add(i + 24));
                let a7 = vld1q_f32(lp.add(i + 28));
                vst1q_f32(op.add(i + 16), vsubq_f32(a4, b4));
                vst1q_f32(op.add(i + 20), vsubq_f32(a5, b5));
                let b6 = vld1q_f32(rp.add(i + 24));
                let b7 = vld1q_f32(rp.add(i + 28));
                vst1q_f32(op.add(i + 24), vsubq_f32(a6, b6));
                vst1q_f32(op.add(i + 28), vsubq_f32(a7, b7));
                i += 32;
            }
        }
        BinaryKind::Mul => {
            while i + 32 <= len {
                let a0 = vld1q_f32(lp.add(i));
                let a1 = vld1q_f32(lp.add(i + 4));
                let b0 = vld1q_f32(rp.add(i));
                let b1 = vld1q_f32(rp.add(i + 4));
                let a2 = vld1q_f32(lp.add(i + 8));
                let a3 = vld1q_f32(lp.add(i + 12));
                vst1q_f32(op.add(i), vmulq_f32(a0, b0));
                vst1q_f32(op.add(i + 4), vmulq_f32(a1, b1));
                let b2 = vld1q_f32(rp.add(i + 8));
                let b3 = vld1q_f32(rp.add(i + 12));
                let a4 = vld1q_f32(lp.add(i + 16));
                let a5 = vld1q_f32(lp.add(i + 20));
                vst1q_f32(op.add(i + 8), vmulq_f32(a2, b2));
                vst1q_f32(op.add(i + 12), vmulq_f32(a3, b3));
                let b4 = vld1q_f32(rp.add(i + 16));
                let b5 = vld1q_f32(rp.add(i + 20));
                let a6 = vld1q_f32(lp.add(i + 24));
                let a7 = vld1q_f32(lp.add(i + 28));
                vst1q_f32(op.add(i + 16), vmulq_f32(a4, b4));
                vst1q_f32(op.add(i + 20), vmulq_f32(a5, b5));
                let b6 = vld1q_f32(rp.add(i + 24));
                let b7 = vld1q_f32(rp.add(i + 28));
                vst1q_f32(op.add(i + 24), vmulq_f32(a6, b6));
                vst1q_f32(op.add(i + 28), vmulq_f32(a7, b7));
                i += 32;
            }
        }
        BinaryKind::Div => {
            while i + 32 <= len {
                let a0 = vld1q_f32(lp.add(i));
                let a1 = vld1q_f32(lp.add(i + 4));
                let b0 = vld1q_f32(rp.add(i));
                let b1 = vld1q_f32(rp.add(i + 4));
                let a2 = vld1q_f32(lp.add(i + 8));
                let a3 = vld1q_f32(lp.add(i + 12));
                vst1q_f32(op.add(i), vdivq_f32(a0, b0));
                vst1q_f32(op.add(i + 4), vdivq_f32(a1, b1));
                let b2 = vld1q_f32(rp.add(i + 8));
                let b3 = vld1q_f32(rp.add(i + 12));
                let a4 = vld1q_f32(lp.add(i + 16));
                let a5 = vld1q_f32(lp.add(i + 20));
                vst1q_f32(op.add(i + 8), vdivq_f32(a2, b2));
                vst1q_f32(op.add(i + 12), vdivq_f32(a3, b3));
                let b4 = vld1q_f32(rp.add(i + 16));
                let b5 = vld1q_f32(rp.add(i + 20));
                let a6 = vld1q_f32(lp.add(i + 24));
                let a7 = vld1q_f32(lp.add(i + 28));
                vst1q_f32(op.add(i + 16), vdivq_f32(a4, b4));
                vst1q_f32(op.add(i + 20), vdivq_f32(a5, b5));
                let b6 = vld1q_f32(rp.add(i + 24));
                let b7 = vld1q_f32(rp.add(i + 28));
                vst1q_f32(op.add(i + 24), vdivq_f32(a6, b6));
                vst1q_f32(op.add(i + 28), vdivq_f32(a7, b7));
                i += 32;
            }
        }
    }

    // Handle remaining 16-element chunks (4x unrolled fallback)
    match kind {
        BinaryKind::Add => {
            while i + 16 <= len {
                let a0 = vld1q_f32(lp.add(i));
                let b0 = vld1q_f32(rp.add(i));
                vst1q_f32(op.add(i), vaddq_f32(a0, b0));
                let a1 = vld1q_f32(lp.add(i + 4));
                let b1 = vld1q_f32(rp.add(i + 4));
                vst1q_f32(op.add(i + 4), vaddq_f32(a1, b1));
                let a2 = vld1q_f32(lp.add(i + 8));
                let b2 = vld1q_f32(rp.add(i + 8));
                vst1q_f32(op.add(i + 8), vaddq_f32(a2, b2));
                let a3 = vld1q_f32(lp.add(i + 12));
                let b3 = vld1q_f32(rp.add(i + 12));
                vst1q_f32(op.add(i + 12), vaddq_f32(a3, b3));
                i += 16;
            }
        }
        BinaryKind::Sub => {
            while i + 16 <= len {
                let a0 = vld1q_f32(lp.add(i));
                let b0 = vld1q_f32(rp.add(i));
                vst1q_f32(op.add(i), vsubq_f32(a0, b0));
                let a1 = vld1q_f32(lp.add(i + 4));
                let b1 = vld1q_f32(rp.add(i + 4));
                vst1q_f32(op.add(i + 4), vsubq_f32(a1, b1));
                let a2 = vld1q_f32(lp.add(i + 8));
                let b2 = vld1q_f32(rp.add(i + 8));
                vst1q_f32(op.add(i + 8), vsubq_f32(a2, b2));
                let a3 = vld1q_f32(lp.add(i + 12));
                let b3 = vld1q_f32(rp.add(i + 12));
                vst1q_f32(op.add(i + 12), vsubq_f32(a3, b3));
                i += 16;
            }
        }
        BinaryKind::Mul => {
            while i + 16 <= len {
                let a0 = vld1q_f32(lp.add(i));
                let b0 = vld1q_f32(rp.add(i));
                vst1q_f32(op.add(i), vmulq_f32(a0, b0));
                let a1 = vld1q_f32(lp.add(i + 4));
                let b1 = vld1q_f32(rp.add(i + 4));
                vst1q_f32(op.add(i + 4), vmulq_f32(a1, b1));
                let a2 = vld1q_f32(lp.add(i + 8));
                let b2 = vld1q_f32(rp.add(i + 8));
                vst1q_f32(op.add(i + 8), vmulq_f32(a2, b2));
                let a3 = vld1q_f32(lp.add(i + 12));
                let b3 = vld1q_f32(rp.add(i + 12));
                vst1q_f32(op.add(i + 12), vmulq_f32(a3, b3));
                i += 16;
            }
        }
        BinaryKind::Div => {
            while i + 16 <= len {
                let a0 = vld1q_f32(lp.add(i));
                let b0 = vld1q_f32(rp.add(i));
                vst1q_f32(op.add(i), vdivq_f32(a0, b0));
                let a1 = vld1q_f32(lp.add(i + 4));
                let b1 = vld1q_f32(rp.add(i + 4));
                vst1q_f32(op.add(i + 4), vdivq_f32(a1, b1));
                let a2 = vld1q_f32(lp.add(i + 8));
                let b2 = vld1q_f32(rp.add(i + 8));
                vst1q_f32(op.add(i + 8), vdivq_f32(a2, b2));
                let a3 = vld1q_f32(lp.add(i + 12));
                let b3 = vld1q_f32(rp.add(i + 12));
                vst1q_f32(op.add(i + 12), vdivq_f32(a3, b3));
                i += 16;
            }
        }
    }

    // Handle remaining 4-element chunks
    while i + 4 <= len {
        let l = vld1q_f32(lp.add(i));
        let r = vld1q_f32(rp.add(i));
        let result = match kind {
            BinaryKind::Add => vaddq_f32(l, r),
            BinaryKind::Sub => vsubq_f32(l, r),
            BinaryKind::Mul => vmulq_f32(l, r),
            BinaryKind::Div => vdivq_f32(l, r),
        };
        vst1q_f32(op.add(i), result);
        i += 4;
    }

    // Scalar tail
    while i < len {
        *op.add(i) = match kind {
            BinaryKind::Add => *lp.add(i) + *rp.add(i),
            BinaryKind::Sub => *lp.add(i) - *rp.add(i),
            BinaryKind::Mul => *lp.add(i) * *rp.add(i),
            BinaryKind::Div => *lp.add(i) / *rp.add(i),
        };
        i += 1;
    }
}
