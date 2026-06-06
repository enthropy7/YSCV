//! Unary operations (neg, abs, sqrt, recip, floor, ceil, round, sign) with SIMD dispatch.

use super::*;

#[derive(Clone, Copy, Debug)]
pub(crate) enum UnaryKind {
    Neg,
    Abs,
    Sqrt,
    Recip,
    Floor,
    Ceil,
    Round,
    Sign,
}

#[allow(unsafe_code)]
#[inline]
pub(crate) fn unary_dispatch(data: &[f32], out: &mut [f32], kind: UnaryKind) {
    debug_assert_eq!(data.len(), out.len());

    if cfg!(miri) {
        unary_scalar(data, out, kind);
        return;
    }

    // macOS aarch64: use vDSP_vneg (faster than NEON for negation).
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        if matches!(kind, UnaryKind::Neg) {
            let n = data.len() as u32;
            unsafe {
                vDSP_vneg(data.as_ptr(), 1, out.as_mut_ptr(), 1, n);
            }
            return;
        }
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if yscv_cpu::host_cpu().features.avx {
            unsafe { unary_avx(data, out, kind) };
            return;
        }
        if yscv_cpu::host_cpu().features.sse {
            unsafe { unary_sse(data, out, kind) };
            return;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if yscv_cpu::host_cpu().features.neon {
            unsafe { unary_neon(data, out, kind) };
            return;
        }
    }

    unary_scalar(data, out, kind);
}

fn unary_scalar(data: &[f32], out: &mut [f32], kind: UnaryKind) {
    match kind {
        UnaryKind::Neg => {
            for i in 0..data.len() {
                out[i] = -data[i];
            }
        }
        UnaryKind::Abs => {
            for i in 0..data.len() {
                out[i] = data[i].abs();
            }
        }
        UnaryKind::Sqrt => {
            for i in 0..data.len() {
                out[i] = data[i].sqrt();
            }
        }
        UnaryKind::Recip => {
            for i in 0..data.len() {
                out[i] = 1.0 / data[i];
            }
        }
        UnaryKind::Floor => {
            for i in 0..data.len() {
                out[i] = data[i].floor();
            }
        }
        UnaryKind::Ceil => {
            for i in 0..data.len() {
                out[i] = data[i].ceil();
            }
        }
        UnaryKind::Round => {
            for i in 0..data.len() {
                out[i] = data[i].round();
            }
        }
        UnaryKind::Sign => {
            for i in 0..data.len() {
                out[i] = if data[i] > 0.0 {
                    1.0
                } else if data[i] < 0.0 {
                    -1.0
                } else {
                    0.0
                };
            }
        }
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "sse")]
unsafe fn unary_sse(data: &[f32], out: &mut [f32], kind: UnaryKind) {
    // Floor/Ceil/Round need SSE4.1; fall back to scalar on SSE-only CPUs.
    if matches!(kind, UnaryKind::Floor | UnaryKind::Ceil | UnaryKind::Round) {
        unary_scalar(data, out, kind);
        return;
    }

    let len = data.len();
    let inp = data.as_ptr();
    let op = out.as_mut_ptr();
    let mut i = 0usize;

    // Match OUTSIDE loop for better branch prediction + constant hoisting.
    match kind {
        UnaryKind::Neg => {
            let zero = _mm_setzero_ps();
            while i + 4 <= len {
                _mm_storeu_ps(op.add(i), _mm_sub_ps(zero, _mm_loadu_ps(inp.add(i))));
                i += 4;
            }
        }
        UnaryKind::Abs => {
            let sign_mask = _mm_set1_ps(-0.0);
            while i + 4 <= len {
                _mm_storeu_ps(
                    op.add(i),
                    _mm_andnot_ps(sign_mask, _mm_loadu_ps(inp.add(i))),
                );
                i += 4;
            }
        }
        UnaryKind::Sqrt => {
            while i + 4 <= len {
                _mm_storeu_ps(op.add(i), _mm_sqrt_ps(_mm_loadu_ps(inp.add(i))));
                i += 4;
            }
        }
        UnaryKind::Recip => {
            let two = _mm_set1_ps(2.0);
            while i + 4 <= len {
                let v = _mm_loadu_ps(inp.add(i));
                let r = _mm_rcp_ps(v);
                _mm_storeu_ps(op.add(i), _mm_mul_ps(r, _mm_sub_ps(two, _mm_mul_ps(v, r))));
                i += 4;
            }
        }
        UnaryKind::Sign => {
            let zero = _mm_setzero_ps();
            let one = _mm_set1_ps(1.0);
            let neg_one = _mm_set1_ps(-1.0);
            while i + 4 <= len {
                let v = _mm_loadu_ps(inp.add(i));
                let pos_mask = _mm_cmpgt_ps(v, zero);
                let neg_mask = _mm_cmplt_ps(v, zero);
                _mm_storeu_ps(
                    op.add(i),
                    _mm_or_ps(_mm_and_ps(pos_mask, one), _mm_and_ps(neg_mask, neg_one)),
                );
                i += 4;
            }
        }
        UnaryKind::Floor | UnaryKind::Ceil | UnaryKind::Round => unreachable!(),
    }

    // Scalar tail
    while i < len {
        *op.add(i) = match kind {
            UnaryKind::Neg => -*inp.add(i),
            UnaryKind::Abs => (*inp.add(i)).abs(),
            UnaryKind::Sqrt => (*inp.add(i)).sqrt(),
            UnaryKind::Recip => 1.0 / *inp.add(i),
            UnaryKind::Sign => {
                let v = *inp.add(i);
                if v > 0.0 {
                    1.0
                } else if v < 0.0 {
                    -1.0
                } else {
                    0.0
                }
            }
            UnaryKind::Floor | UnaryKind::Ceil | UnaryKind::Round => unreachable!(),
        };
        i += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx")]
unsafe fn unary_avx(data: &[f32], out: &mut [f32], kind: UnaryKind) {
    let len = data.len();
    let inp = data.as_ptr();
    let op = out.as_mut_ptr();
    let mut i = 0usize;

    // Match OUTSIDE loop: eliminates branch per iteration, hoists constants.
    match kind {
        UnaryKind::Neg => {
            let zero = _mm256_setzero_ps();
            while i + 32 <= len {
                _mm256_storeu_ps(op.add(i), _mm256_sub_ps(zero, _mm256_loadu_ps(inp.add(i))));
                _mm256_storeu_ps(
                    op.add(i + 8),
                    _mm256_sub_ps(zero, _mm256_loadu_ps(inp.add(i + 8))),
                );
                _mm256_storeu_ps(
                    op.add(i + 16),
                    _mm256_sub_ps(zero, _mm256_loadu_ps(inp.add(i + 16))),
                );
                _mm256_storeu_ps(
                    op.add(i + 24),
                    _mm256_sub_ps(zero, _mm256_loadu_ps(inp.add(i + 24))),
                );
                i += 32;
            }
        }
        UnaryKind::Abs => {
            let sign_mask = _mm256_set1_ps(-0.0);
            while i + 32 <= len {
                _mm256_storeu_ps(
                    op.add(i),
                    _mm256_andnot_ps(sign_mask, _mm256_loadu_ps(inp.add(i))),
                );
                _mm256_storeu_ps(
                    op.add(i + 8),
                    _mm256_andnot_ps(sign_mask, _mm256_loadu_ps(inp.add(i + 8))),
                );
                _mm256_storeu_ps(
                    op.add(i + 16),
                    _mm256_andnot_ps(sign_mask, _mm256_loadu_ps(inp.add(i + 16))),
                );
                _mm256_storeu_ps(
                    op.add(i + 24),
                    _mm256_andnot_ps(sign_mask, _mm256_loadu_ps(inp.add(i + 24))),
                );
                i += 32;
            }
        }
        UnaryKind::Sqrt => {
            while i + 32 <= len {
                _mm256_storeu_ps(op.add(i), _mm256_sqrt_ps(_mm256_loadu_ps(inp.add(i))));
                _mm256_storeu_ps(
                    op.add(i + 8),
                    _mm256_sqrt_ps(_mm256_loadu_ps(inp.add(i + 8))),
                );
                _mm256_storeu_ps(
                    op.add(i + 16),
                    _mm256_sqrt_ps(_mm256_loadu_ps(inp.add(i + 16))),
                );
                _mm256_storeu_ps(
                    op.add(i + 24),
                    _mm256_sqrt_ps(_mm256_loadu_ps(inp.add(i + 24))),
                );
                i += 32;
            }
        }
        UnaryKind::Recip => {
            let two = _mm256_set1_ps(2.0);
            while i + 32 <= len {
                for off in [0, 8, 16, 24] {
                    let v = _mm256_loadu_ps(inp.add(i + off));
                    let r = _mm256_rcp_ps(v);
                    _mm256_storeu_ps(
                        op.add(i + off),
                        _mm256_mul_ps(r, _mm256_sub_ps(two, _mm256_mul_ps(v, r))),
                    );
                }
                i += 32;
            }
        }
        UnaryKind::Floor => {
            while i + 32 <= len {
                _mm256_storeu_ps(op.add(i), _mm256_floor_ps(_mm256_loadu_ps(inp.add(i))));
                _mm256_storeu_ps(
                    op.add(i + 8),
                    _mm256_floor_ps(_mm256_loadu_ps(inp.add(i + 8))),
                );
                _mm256_storeu_ps(
                    op.add(i + 16),
                    _mm256_floor_ps(_mm256_loadu_ps(inp.add(i + 16))),
                );
                _mm256_storeu_ps(
                    op.add(i + 24),
                    _mm256_floor_ps(_mm256_loadu_ps(inp.add(i + 24))),
                );
                i += 32;
            }
        }
        UnaryKind::Ceil => {
            while i + 32 <= len {
                _mm256_storeu_ps(op.add(i), _mm256_ceil_ps(_mm256_loadu_ps(inp.add(i))));
                _mm256_storeu_ps(
                    op.add(i + 8),
                    _mm256_ceil_ps(_mm256_loadu_ps(inp.add(i + 8))),
                );
                _mm256_storeu_ps(
                    op.add(i + 16),
                    _mm256_ceil_ps(_mm256_loadu_ps(inp.add(i + 16))),
                );
                _mm256_storeu_ps(
                    op.add(i + 24),
                    _mm256_ceil_ps(_mm256_loadu_ps(inp.add(i + 24))),
                );
                i += 32;
            }
        }
        UnaryKind::Round => {
            let neg_zero = _mm256_set1_ps(-0.0);
            let half = _mm256_set1_ps(0.5);
            while i + 32 <= len {
                for off in [0, 8, 16, 24] {
                    let v = _mm256_loadu_ps(inp.add(i + off));
                    let sign = _mm256_and_ps(v, neg_zero);
                    let abs_v = _mm256_andnot_ps(neg_zero, v);
                    _mm256_storeu_ps(
                        op.add(i + off),
                        _mm256_or_ps(_mm256_floor_ps(_mm256_add_ps(abs_v, half)), sign),
                    );
                }
                i += 32;
            }
        }
        UnaryKind::Sign => {
            let zero = _mm256_setzero_ps();
            let one = _mm256_set1_ps(1.0);
            let neg_one = _mm256_set1_ps(-1.0);
            while i + 32 <= len {
                for off in [0, 8, 16, 24] {
                    let v = _mm256_loadu_ps(inp.add(i + off));
                    let pos_mask = _mm256_cmp_ps::<14>(v, zero);
                    let neg_mask = _mm256_cmp_ps::<1>(v, zero);
                    _mm256_storeu_ps(
                        op.add(i + off),
                        _mm256_or_ps(
                            _mm256_and_ps(pos_mask, one),
                            _mm256_and_ps(neg_mask, neg_one),
                        ),
                    );
                }
                i += 32;
            }
        }
    }

    // Tail: process remaining elements via SSE
    if i < len {
        unary_sse(&data[i..], &mut out[i..], kind);
    }
}

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "neon")]
unsafe fn unary_neon(data: &[f32], out: &mut [f32], kind: UnaryKind) {
    let len = data.len();
    let inp = data.as_ptr();
    let op = out.as_mut_ptr();
    let mut i = 0usize;

    // Match OUTSIDE loop: eliminates branch per iteration, enables unrolling,
    // and hoists constants. Same pattern as binary_neon.
    match kind {
        UnaryKind::Neg => {
            while i + 16 <= len {
                vst1q_f32(op.add(i), vnegq_f32(vld1q_f32(inp.add(i))));
                vst1q_f32(op.add(i + 4), vnegq_f32(vld1q_f32(inp.add(i + 4))));
                vst1q_f32(op.add(i + 8), vnegq_f32(vld1q_f32(inp.add(i + 8))));
                vst1q_f32(op.add(i + 12), vnegq_f32(vld1q_f32(inp.add(i + 12))));
                i += 16;
            }
        }
        UnaryKind::Abs => {
            while i + 16 <= len {
                vst1q_f32(op.add(i), vabsq_f32(vld1q_f32(inp.add(i))));
                vst1q_f32(op.add(i + 4), vabsq_f32(vld1q_f32(inp.add(i + 4))));
                vst1q_f32(op.add(i + 8), vabsq_f32(vld1q_f32(inp.add(i + 8))));
                vst1q_f32(op.add(i + 12), vabsq_f32(vld1q_f32(inp.add(i + 12))));
                i += 16;
            }
        }
        UnaryKind::Sqrt => {
            while i + 16 <= len {
                vst1q_f32(op.add(i), vsqrtq_f32(vld1q_f32(inp.add(i))));
                vst1q_f32(op.add(i + 4), vsqrtq_f32(vld1q_f32(inp.add(i + 4))));
                vst1q_f32(op.add(i + 8), vsqrtq_f32(vld1q_f32(inp.add(i + 8))));
                vst1q_f32(op.add(i + 12), vsqrtq_f32(vld1q_f32(inp.add(i + 12))));
                i += 16;
            }
        }
        UnaryKind::Recip => {
            while i + 16 <= len {
                let v0 = vld1q_f32(inp.add(i));
                let v1 = vld1q_f32(inp.add(i + 4));
                let v2 = vld1q_f32(inp.add(i + 8));
                let v3 = vld1q_f32(inp.add(i + 12));
                let r0 = vrecpeq_f32(v0);
                let s0 = vrecpsq_f32(v0, r0);
                let r1 = vrecpeq_f32(v1);
                let s1 = vrecpsq_f32(v1, r1);
                let r2 = vrecpeq_f32(v2);
                let s2 = vrecpsq_f32(v2, r2);
                let r3 = vrecpeq_f32(v3);
                let s3 = vrecpsq_f32(v3, r3);
                vst1q_f32(op.add(i), vmulq_f32(r0, s0));
                vst1q_f32(op.add(i + 4), vmulq_f32(r1, s1));
                vst1q_f32(op.add(i + 8), vmulq_f32(r2, s2));
                vst1q_f32(op.add(i + 12), vmulq_f32(r3, s3));
                i += 16;
            }
        }
        UnaryKind::Floor => {
            while i + 16 <= len {
                vst1q_f32(op.add(i), vrndmq_f32(vld1q_f32(inp.add(i))));
                vst1q_f32(op.add(i + 4), vrndmq_f32(vld1q_f32(inp.add(i + 4))));
                vst1q_f32(op.add(i + 8), vrndmq_f32(vld1q_f32(inp.add(i + 8))));
                vst1q_f32(op.add(i + 12), vrndmq_f32(vld1q_f32(inp.add(i + 12))));
                i += 16;
            }
        }
        UnaryKind::Ceil => {
            while i + 16 <= len {
                vst1q_f32(op.add(i), vrndpq_f32(vld1q_f32(inp.add(i))));
                vst1q_f32(op.add(i + 4), vrndpq_f32(vld1q_f32(inp.add(i + 4))));
                vst1q_f32(op.add(i + 8), vrndpq_f32(vld1q_f32(inp.add(i + 8))));
                vst1q_f32(op.add(i + 12), vrndpq_f32(vld1q_f32(inp.add(i + 12))));
                i += 16;
            }
        }
        UnaryKind::Round => {
            while i + 16 <= len {
                vst1q_f32(op.add(i), vrndaq_f32(vld1q_f32(inp.add(i))));
                vst1q_f32(op.add(i + 4), vrndaq_f32(vld1q_f32(inp.add(i + 4))));
                vst1q_f32(op.add(i + 8), vrndaq_f32(vld1q_f32(inp.add(i + 8))));
                vst1q_f32(op.add(i + 12), vrndaq_f32(vld1q_f32(inp.add(i + 12))));
                i += 16;
            }
        }
        UnaryKind::Sign => {
            let zero = vdupq_n_f32(0.0);
            let one = vdupq_n_f32(1.0);
            let neg_one = vdupq_n_f32(-1.0);
            while i + 16 <= len {
                let v0 = vld1q_f32(inp.add(i));
                let v1 = vld1q_f32(inp.add(i + 4));
                let v2 = vld1q_f32(inp.add(i + 8));
                let v3 = vld1q_f32(inp.add(i + 12));
                vst1q_f32(
                    op.add(i),
                    vaddq_f32(
                        vbslq_f32(vcgtq_f32(v0, zero), one, zero),
                        vbslq_f32(vcltq_f32(v0, zero), neg_one, zero),
                    ),
                );
                vst1q_f32(
                    op.add(i + 4),
                    vaddq_f32(
                        vbslq_f32(vcgtq_f32(v1, zero), one, zero),
                        vbslq_f32(vcltq_f32(v1, zero), neg_one, zero),
                    ),
                );
                vst1q_f32(
                    op.add(i + 8),
                    vaddq_f32(
                        vbslq_f32(vcgtq_f32(v2, zero), one, zero),
                        vbslq_f32(vcltq_f32(v2, zero), neg_one, zero),
                    ),
                );
                vst1q_f32(
                    op.add(i + 12),
                    vaddq_f32(
                        vbslq_f32(vcgtq_f32(v3, zero), one, zero),
                        vbslq_f32(vcltq_f32(v3, zero), neg_one, zero),
                    ),
                );
                i += 16;
            }
        }
    }

    // Scalar tail for remaining < 16 elements
    while i < len {
        *op.add(i) = match kind {
            UnaryKind::Neg => -*inp.add(i),
            UnaryKind::Abs => (*inp.add(i)).abs(),
            UnaryKind::Sqrt => (*inp.add(i)).sqrt(),
            UnaryKind::Recip => 1.0 / *inp.add(i),
            UnaryKind::Floor => (*inp.add(i)).floor(),
            UnaryKind::Ceil => (*inp.add(i)).ceil(),
            UnaryKind::Round => (*inp.add(i)).round(),
            UnaryKind::Sign => {
                let v = *inp.add(i);
                if v > 0.0 {
                    1.0
                } else if v < 0.0 {
                    -1.0
                } else {
                    0.0
                }
            }
        };
        i += 1;
    }
}
