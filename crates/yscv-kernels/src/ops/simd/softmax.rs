// ===========================================================================
// Fused softmax + log-softmax: dispatch + implementations
// ===========================================================================

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::{
    vaddq_f32, vdupq_n_f32, vld1q_f32, vmaxq_f32, vmulq_f32, vst1q_f32, vsubq_f32,
};
#[cfg(target_arch = "x86")]
use std::arch::x86::{
    _mm_add_ps, _mm_loadu_ps, _mm_max_ps, _mm_mul_ps, _mm_set1_ps, _mm_setzero_ps, _mm_storeu_ps,
    _mm_sub_ps, _mm256_add_ps, _mm256_loadu_ps, _mm256_max_ps, _mm256_mul_ps, _mm256_set1_ps,
    _mm256_setzero_ps, _mm256_storeu_ps, _mm256_sub_ps,
};
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{
    _mm_add_ps, _mm_loadu_ps, _mm_max_ps, _mm_mul_ps, _mm_set1_ps, _mm_setzero_ps, _mm_storeu_ps,
    _mm_sub_ps, _mm256_add_ps, _mm256_loadu_ps, _mm256_max_ps, _mm256_mul_ps, _mm256_set1_ps,
    _mm256_setzero_ps, _mm256_storeu_ps, _mm256_sub_ps,
};
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{
    _mm512_add_ps, _mm512_loadu_ps, _mm512_max_ps, _mm512_mul_ps, _mm512_reduce_add_ps,
    _mm512_reduce_max_ps, _mm512_set1_ps, _mm512_setzero_ps, _mm512_storeu_ps, _mm512_sub_ps,
};

#[cfg(target_arch = "x86_64")]
use super::exp::fast_exp_avx512;
#[cfg(target_arch = "aarch64")]
use super::exp::fast_exp_neon;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use super::exp::{fast_exp_avx, fast_exp_sse};
use super::{SimdDispatchPath, dispatch_path};

// ===========================================================================
// Softmax dispatch
// ===========================================================================

#[allow(unsafe_code)]
#[inline]
pub fn softmax_rows_fused_dispatch(input: &[f32], output: &mut [f32], row_len: usize) {
    debug_assert_eq!(input.len(), output.len());
    debug_assert!(row_len > 0);
    debug_assert_eq!(input.len() % row_len, 0);

    if cfg!(miri) || input.is_empty() {
        for (in_row, out_row) in input.chunks(row_len).zip(output.chunks_mut(row_len)) {
            softmax_row_fused_scalar(in_row, out_row);
        }
        return;
    }

    let path = dispatch_path(true, false);

    #[cfg(target_arch = "aarch64")]
    {
        if path == SimdDispatchPath::Neon {
            // SAFETY: guarded by runtime feature detection in `dispatch_path`.
            unsafe {
                for (in_row, out_row) in input.chunks(row_len).zip(output.chunks_mut(row_len)) {
                    softmax_row_fused_neon(in_row, out_row);
                }
            }
            return;
        }
    }

    #[cfg(target_arch = "x86_64")]
    if path == SimdDispatchPath::Avx512 {
        // SAFETY: guarded by runtime feature detection in `dispatch_path`.
        unsafe {
            if row_len == 256 {
                for (in_row, out_row) in input.chunks(row_len).zip(output.chunks_mut(row_len)) {
                    softmax_row_256_avx512(in_row, out_row);
                }
            } else {
                for (in_row, out_row) in input.chunks(row_len).zip(output.chunks_mut(row_len)) {
                    softmax_row_fused_avx512(in_row, out_row);
                }
            }
        }
        return;
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if path == SimdDispatchPath::Avx {
            // SAFETY: guarded by runtime feature detection in `dispatch_path`.
            unsafe {
                for (in_row, out_row) in input.chunks(row_len).zip(output.chunks_mut(row_len)) {
                    softmax_row_fused_avx(in_row, out_row);
                }
            }
            return;
        }
        if path == SimdDispatchPath::Sse {
            // SAFETY: guarded by runtime feature detection in `dispatch_path`.
            unsafe {
                for (in_row, out_row) in input.chunks(row_len).zip(output.chunks_mut(row_len)) {
                    softmax_row_fused_sse(in_row, out_row);
                }
            }
            return;
        }
    }

    for (in_row, out_row) in input.chunks(row_len).zip(output.chunks_mut(row_len)) {
        softmax_row_fused_scalar(in_row, out_row);
    }
}

// ===========================================================================
// Log-softmax dispatch
// ===========================================================================

#[allow(unsafe_code)]
#[inline]
pub fn log_softmax_rows_fused_dispatch(input: &[f32], output: &mut [f32], row_len: usize) {
    debug_assert_eq!(input.len(), output.len());
    debug_assert!(row_len > 0);
    debug_assert_eq!(input.len() % row_len, 0);

    if cfg!(miri) || input.is_empty() {
        for (in_row, out_row) in input.chunks(row_len).zip(output.chunks_mut(row_len)) {
            log_softmax_row_fused_scalar(in_row, out_row);
        }
        return;
    }

    let path = dispatch_path(true, false);

    #[cfg(target_arch = "aarch64")]
    {
        if path == SimdDispatchPath::Neon {
            // SAFETY: guarded by runtime feature detection in `dispatch_path`.
            unsafe {
                for (in_row, out_row) in input.chunks(row_len).zip(output.chunks_mut(row_len)) {
                    log_softmax_row_fused_neon(in_row, out_row);
                }
            }
            return;
        }
    }

    #[cfg(target_arch = "x86_64")]
    if path == SimdDispatchPath::Avx512 {
        // SAFETY: guarded by runtime feature detection in `dispatch_path`.
        unsafe {
            for (in_row, out_row) in input.chunks(row_len).zip(output.chunks_mut(row_len)) {
                log_softmax_row_fused_avx512(in_row, out_row);
            }
        }
        return;
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if path == SimdDispatchPath::Avx {
            // SAFETY: guarded by runtime feature detection in `dispatch_path`.
            unsafe {
                for (in_row, out_row) in input.chunks(row_len).zip(output.chunks_mut(row_len)) {
                    log_softmax_row_fused_avx(in_row, out_row);
                }
            }
            return;
        }
        if path == SimdDispatchPath::Sse {
            // SAFETY: guarded by runtime feature detection in `dispatch_path`.
            unsafe {
                for (in_row, out_row) in input.chunks(row_len).zip(output.chunks_mut(row_len)) {
                    log_softmax_row_fused_sse(in_row, out_row);
                }
            }
            return;
        }
    }

    for (in_row, out_row) in input.chunks(row_len).zip(output.chunks_mut(row_len)) {
        log_softmax_row_fused_scalar(in_row, out_row);
    }
}

// ===========================================================================
// Scalar fallbacks
// ===========================================================================

fn softmax_row_fused_scalar(input: &[f32], output: &mut [f32]) {
    if input.is_empty() {
        return;
    }

    // 1. max
    let mut max_val = f32::NEG_INFINITY;
    for &v in input {
        max_val = max_val.max(v);
    }

    // 2. sub+exp + 3. accumulate sum
    let mut sum_exp = 0.0f32;
    for (o, &v) in output.iter_mut().zip(input.iter()) {
        let e = (v - max_val).exp();
        *o = e;
        sum_exp += e;
    }

    // 4. divide
    let inv = 1.0 / sum_exp;
    for o in output.iter_mut() {
        *o *= inv;
    }
}

fn log_softmax_row_fused_scalar(input: &[f32], output: &mut [f32]) {
    if input.is_empty() {
        return;
    }

    // 1. max
    let mut max_val = f32::NEG_INFINITY;
    for &v in input {
        max_val = max_val.max(v);
    }

    // 2. sum(exp(x - max))
    let mut sum_exp = 0.0f32;
    for &v in input {
        sum_exp += (v - max_val).exp();
    }

    // 3. output[i] = x[i] - max - log(sum_exp)
    let log_denom = max_val + sum_exp.ln();
    for (o, &v) in output.iter_mut().zip(input.iter()) {
        *o = v - log_denom;
    }
}

// ===========================================================================
// Softmax SIMD implementations
// ===========================================================================

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "neon")]
unsafe fn softmax_row_fused_neon(input: &[f32], output: &mut [f32]) {
    use std::arch::aarch64::{vaddvq_f32, vmaxvq_f32};

    let len = input.len();
    let in_ptr = input.as_ptr();
    let out_ptr = output.as_mut_ptr();

    // 1. Find max (NEON reduce)
    let mut acc_max = vdupq_n_f32(f32::NEG_INFINITY);
    let mut i = 0usize;
    while i + 16 <= len {
        let v0 = vld1q_f32(in_ptr.add(i));
        let v1 = vld1q_f32(in_ptr.add(i + 4));
        let v2 = vld1q_f32(in_ptr.add(i + 8));
        let v3 = vld1q_f32(in_ptr.add(i + 12));
        acc_max = vmaxq_f32(acc_max, vmaxq_f32(vmaxq_f32(v0, v1), vmaxq_f32(v2, v3)));
        i += 16;
    }
    while i + 4 <= len {
        let v = vld1q_f32(in_ptr.add(i));
        acc_max = vmaxq_f32(acc_max, v);
        i += 4;
    }
    let mut max_val = vmaxvq_f32(acc_max);
    while i < len {
        max_val = max_val.max(*in_ptr.add(i));
        i += 1;
    }

    // 2. sub+exp (NEON fast_exp, writes output) + 3. accumulate sum
    let off = vdupq_n_f32(max_val);
    let mut acc_sum = vdupq_n_f32(0.0);
    i = 0;
    while i + 16 <= len {
        let v0 = fast_exp_neon(vsubq_f32(vld1q_f32(in_ptr.add(i)), off));
        let v1 = fast_exp_neon(vsubq_f32(vld1q_f32(in_ptr.add(i + 4)), off));
        let v2 = fast_exp_neon(vsubq_f32(vld1q_f32(in_ptr.add(i + 8)), off));
        let v3 = fast_exp_neon(vsubq_f32(vld1q_f32(in_ptr.add(i + 12)), off));
        vst1q_f32(out_ptr.add(i), v0);
        vst1q_f32(out_ptr.add(i + 4), v1);
        vst1q_f32(out_ptr.add(i + 8), v2);
        vst1q_f32(out_ptr.add(i + 12), v3);
        acc_sum = vaddq_f32(acc_sum, vaddq_f32(vaddq_f32(v0, v1), vaddq_f32(v2, v3)));
        i += 16;
    }
    while i + 4 <= len {
        let v = fast_exp_neon(vsubq_f32(vld1q_f32(in_ptr.add(i)), off));
        vst1q_f32(out_ptr.add(i), v);
        acc_sum = vaddq_f32(acc_sum, v);
        i += 4;
    }
    let mut sum_exp = vaddvq_f32(acc_sum);
    while i < len {
        let e = (*in_ptr.add(i) - max_val).exp();
        *out_ptr.add(i) = e;
        sum_exp += e;
        i += 1;
    }

    // 4. divide (NEON multiply by 1/sum)
    let inv = vdupq_n_f32(1.0 / sum_exp);
    i = 0;
    while i + 16 <= len {
        vst1q_f32(out_ptr.add(i), vmulq_f32(vld1q_f32(out_ptr.add(i)), inv));
        vst1q_f32(
            out_ptr.add(i + 4),
            vmulq_f32(vld1q_f32(out_ptr.add(i + 4)), inv),
        );
        vst1q_f32(
            out_ptr.add(i + 8),
            vmulq_f32(vld1q_f32(out_ptr.add(i + 8)), inv),
        );
        vst1q_f32(
            out_ptr.add(i + 12),
            vmulq_f32(vld1q_f32(out_ptr.add(i + 12)), inv),
        );
        i += 16;
    }
    while i + 4 <= len {
        vst1q_f32(out_ptr.add(i), vmulq_f32(vld1q_f32(out_ptr.add(i)), inv));
        i += 4;
    }
    let inv_s = 1.0 / sum_exp;
    while i < len {
        *out_ptr.add(i) *= inv_s;
        i += 1;
    }
}

/// SSE fused softmax fallback.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "sse")]
unsafe fn softmax_row_fused_sse(input: &[f32], output: &mut [f32]) {
    let len = input.len();
    let in_ptr = input.as_ptr();
    let out_ptr = output.as_mut_ptr();

    // 1. max
    let mut acc_max = _mm_set1_ps(f32::NEG_INFINITY);
    let mut i = 0usize;
    while i + 4 <= len {
        acc_max = _mm_max_ps(acc_max, _mm_loadu_ps(in_ptr.add(i)));
        i += 4;
    }
    let mut buf = [0.0f32; 4];
    _mm_storeu_ps(buf.as_mut_ptr(), acc_max);
    let mut max_val = buf[0].max(buf[1]).max(buf[2].max(buf[3]));
    while i < len {
        max_val = max_val.max(*in_ptr.add(i));
        i += 1;
    }

    // 2. sub+exp + 3. sum
    let off = _mm_set1_ps(max_val);
    let mut acc_sum = _mm_setzero_ps();
    i = 0;
    while i + 4 <= len {
        let v = fast_exp_sse(_mm_sub_ps(_mm_loadu_ps(in_ptr.add(i)), off));
        _mm_storeu_ps(out_ptr.add(i), v);
        acc_sum = _mm_add_ps(acc_sum, v);
        i += 4;
    }
    _mm_storeu_ps(buf.as_mut_ptr(), acc_sum);
    let mut sum_exp = buf[0] + buf[1] + buf[2] + buf[3];
    while i < len {
        let e = (*in_ptr.add(i) - max_val).exp();
        *out_ptr.add(i) = e;
        sum_exp += e;
        i += 1;
    }

    // 4. divide
    let inv = _mm_set1_ps(1.0 / sum_exp);
    i = 0;
    while i + 4 <= len {
        _mm_storeu_ps(
            out_ptr.add(i),
            _mm_mul_ps(_mm_loadu_ps(out_ptr.add(i)), inv),
        );
        i += 4;
    }
    let inv_s = 1.0 / sum_exp;
    while i < len {
        *out_ptr.add(i) *= inv_s;
        i += 1;
    }
}

/// AVX fused softmax fallback -- delegates tail to SSE.
#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx512f")]
unsafe fn softmax_row_fused_avx512(input: &[f32], output: &mut [f32]) {
    let len = input.len();
    let in_ptr = input.as_ptr();
    let out_ptr = output.as_mut_ptr();

    let mut acc_max = _mm512_set1_ps(f32::NEG_INFINITY);
    let mut i = 0usize;
    while i + 16 <= len {
        acc_max = _mm512_max_ps(acc_max, _mm512_loadu_ps(in_ptr.add(i)));
        i += 16;
    }
    let mut max_val = _mm512_reduce_max_ps(acc_max);
    if i + 8 <= len {
        let v = _mm256_loadu_ps(in_ptr.add(i));
        let mut buf8 = [0.0f32; 8];
        _mm256_storeu_ps(buf8.as_mut_ptr(), v);
        for &x in &buf8 {
            max_val = max_val.max(x);
        }
        i += 8;
    }
    while i < len {
        max_val = max_val.max(*in_ptr.add(i));
        i += 1;
    }

    let off = _mm512_set1_ps(max_val);
    let mut acc_sum = _mm512_setzero_ps();
    i = 0;
    while i + 16 <= len {
        let v = fast_exp_avx512(_mm512_sub_ps(_mm512_loadu_ps(in_ptr.add(i)), off));
        _mm512_storeu_ps(out_ptr.add(i), v);
        acc_sum = _mm512_add_ps(acc_sum, v);
        i += 16;
    }
    let mut sum_exp = _mm512_reduce_add_ps(acc_sum);
    if i + 8 <= len {
        let off8 = _mm256_set1_ps(max_val);
        let v = fast_exp_avx(_mm256_sub_ps(_mm256_loadu_ps(in_ptr.add(i)), off8));
        _mm256_storeu_ps(out_ptr.add(i), v);
        let mut buf8 = [0.0f32; 8];
        _mm256_storeu_ps(buf8.as_mut_ptr(), v);
        sum_exp += buf8.iter().sum::<f32>();
        i += 8;
    }
    while i < len {
        let e = (*in_ptr.add(i) - max_val).exp();
        *out_ptr.add(i) = e;
        sum_exp += e;
        i += 1;
    }

    let inv = _mm512_set1_ps(1.0 / sum_exp);
    i = 0;
    while i + 16 <= len {
        _mm512_storeu_ps(
            out_ptr.add(i),
            _mm512_mul_ps(_mm512_loadu_ps(out_ptr.add(i)), inv),
        );
        i += 16;
    }
    if i + 8 <= len {
        let inv8 = _mm256_set1_ps(1.0 / sum_exp);
        _mm256_storeu_ps(
            out_ptr.add(i),
            _mm256_mul_ps(_mm256_loadu_ps(out_ptr.add(i)), inv8),
        );
        i += 8;
    }
    let inv_s = 1.0 / sum_exp;
    while i < len {
        *out_ptr.add(i) *= inv_s;
        i += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx512f")]
unsafe fn softmax_row_256_avx512(input: &[f32], output: &mut [f32]) {
    debug_assert_eq!(input.len(), 256);
    debug_assert_eq!(output.len(), 256);

    let in_ptr = input.as_ptr();
    let out_ptr = output.as_mut_ptr();
    let mut acc_max = _mm512_set1_ps(f32::NEG_INFINITY);
    let mut i = 0usize;
    while i < 256 {
        acc_max = _mm512_max_ps(acc_max, _mm512_loadu_ps(in_ptr.add(i)));
        i += 16;
    }
    let max_val = _mm512_reduce_max_ps(acc_max);

    let off = _mm512_set1_ps(max_val);
    let e0 = fast_exp_avx512(_mm512_sub_ps(_mm512_loadu_ps(in_ptr), off));
    let e1 = fast_exp_avx512(_mm512_sub_ps(_mm512_loadu_ps(in_ptr.add(16)), off));
    let e2 = fast_exp_avx512(_mm512_sub_ps(_mm512_loadu_ps(in_ptr.add(32)), off));
    let e3 = fast_exp_avx512(_mm512_sub_ps(_mm512_loadu_ps(in_ptr.add(48)), off));
    let e4 = fast_exp_avx512(_mm512_sub_ps(_mm512_loadu_ps(in_ptr.add(64)), off));
    let e5 = fast_exp_avx512(_mm512_sub_ps(_mm512_loadu_ps(in_ptr.add(80)), off));
    let e6 = fast_exp_avx512(_mm512_sub_ps(_mm512_loadu_ps(in_ptr.add(96)), off));
    let e7 = fast_exp_avx512(_mm512_sub_ps(_mm512_loadu_ps(in_ptr.add(112)), off));
    let e8 = fast_exp_avx512(_mm512_sub_ps(_mm512_loadu_ps(in_ptr.add(128)), off));
    let e9 = fast_exp_avx512(_mm512_sub_ps(_mm512_loadu_ps(in_ptr.add(144)), off));
    let e10 = fast_exp_avx512(_mm512_sub_ps(_mm512_loadu_ps(in_ptr.add(160)), off));
    let e11 = fast_exp_avx512(_mm512_sub_ps(_mm512_loadu_ps(in_ptr.add(176)), off));
    let e12 = fast_exp_avx512(_mm512_sub_ps(_mm512_loadu_ps(in_ptr.add(192)), off));
    let e13 = fast_exp_avx512(_mm512_sub_ps(_mm512_loadu_ps(in_ptr.add(208)), off));
    let e14 = fast_exp_avx512(_mm512_sub_ps(_mm512_loadu_ps(in_ptr.add(224)), off));
    let e15 = fast_exp_avx512(_mm512_sub_ps(_mm512_loadu_ps(in_ptr.add(240)), off));

    let s0 = _mm512_add_ps(e0, e1);
    let s1 = _mm512_add_ps(e2, e3);
    let s2 = _mm512_add_ps(e4, e5);
    let s3 = _mm512_add_ps(e6, e7);
    let s4 = _mm512_add_ps(e8, e9);
    let s5 = _mm512_add_ps(e10, e11);
    let s6 = _mm512_add_ps(e12, e13);
    let s7 = _mm512_add_ps(e14, e15);
    let sum_vec = _mm512_add_ps(
        _mm512_add_ps(_mm512_add_ps(s0, s1), _mm512_add_ps(s2, s3)),
        _mm512_add_ps(_mm512_add_ps(s4, s5), _mm512_add_ps(s6, s7)),
    );
    let sum_exp = _mm512_reduce_add_ps(sum_vec);
    let inv = _mm512_set1_ps(1.0 / sum_exp);

    _mm512_storeu_ps(out_ptr, _mm512_mul_ps(e0, inv));
    _mm512_storeu_ps(out_ptr.add(16), _mm512_mul_ps(e1, inv));
    _mm512_storeu_ps(out_ptr.add(32), _mm512_mul_ps(e2, inv));
    _mm512_storeu_ps(out_ptr.add(48), _mm512_mul_ps(e3, inv));
    _mm512_storeu_ps(out_ptr.add(64), _mm512_mul_ps(e4, inv));
    _mm512_storeu_ps(out_ptr.add(80), _mm512_mul_ps(e5, inv));
    _mm512_storeu_ps(out_ptr.add(96), _mm512_mul_ps(e6, inv));
    _mm512_storeu_ps(out_ptr.add(112), _mm512_mul_ps(e7, inv));
    _mm512_storeu_ps(out_ptr.add(128), _mm512_mul_ps(e8, inv));
    _mm512_storeu_ps(out_ptr.add(144), _mm512_mul_ps(e9, inv));
    _mm512_storeu_ps(out_ptr.add(160), _mm512_mul_ps(e10, inv));
    _mm512_storeu_ps(out_ptr.add(176), _mm512_mul_ps(e11, inv));
    _mm512_storeu_ps(out_ptr.add(192), _mm512_mul_ps(e12, inv));
    _mm512_storeu_ps(out_ptr.add(208), _mm512_mul_ps(e13, inv));
    _mm512_storeu_ps(out_ptr.add(224), _mm512_mul_ps(e14, inv));
    _mm512_storeu_ps(out_ptr.add(240), _mm512_mul_ps(e15, inv));
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx")]
unsafe fn softmax_row_fused_avx(input: &[f32], output: &mut [f32]) {
    let len = input.len();
    let in_ptr = input.as_ptr();
    let out_ptr = output.as_mut_ptr();

    // 1. max
    let mut acc_max = _mm256_set1_ps(f32::NEG_INFINITY);
    let mut i = 0usize;
    while i + 8 <= len {
        acc_max = _mm256_max_ps(acc_max, _mm256_loadu_ps(in_ptr.add(i)));
        i += 8;
    }
    let mut buf8 = [0.0f32; 8];
    _mm256_storeu_ps(buf8.as_mut_ptr(), acc_max);
    let mut max_val = buf8[0];
    for &v in &buf8[1..] {
        max_val = max_val.max(v);
    }
    while i < len {
        max_val = max_val.max(*in_ptr.add(i));
        i += 1;
    }

    // 2. sub+exp + 3. sum
    let off = _mm256_set1_ps(max_val);
    let mut acc_sum = _mm256_setzero_ps();
    i = 0;
    while i + 8 <= len {
        let v = fast_exp_avx(_mm256_sub_ps(_mm256_loadu_ps(in_ptr.add(i)), off));
        _mm256_storeu_ps(out_ptr.add(i), v);
        acc_sum = _mm256_add_ps(acc_sum, v);
        i += 8;
    }
    _mm256_storeu_ps(buf8.as_mut_ptr(), acc_sum);
    let mut sum_exp: f32 = buf8.iter().sum();
    // SSE tail for remaining < 8 elements
    let off4 = _mm_set1_ps(max_val);
    while i + 4 <= len {
        let v = fast_exp_sse(_mm_sub_ps(_mm_loadu_ps(in_ptr.add(i)), off4));
        _mm_storeu_ps(out_ptr.add(i), v);
        let mut b4 = [0.0f32; 4];
        _mm_storeu_ps(b4.as_mut_ptr(), v);
        sum_exp += b4[0] + b4[1] + b4[2] + b4[3];
        i += 4;
    }
    while i < len {
        let e = (*in_ptr.add(i) - max_val).exp();
        *out_ptr.add(i) = e;
        sum_exp += e;
        i += 1;
    }

    // 4. divide
    let inv8 = _mm256_set1_ps(1.0 / sum_exp);
    i = 0;
    while i + 8 <= len {
        _mm256_storeu_ps(
            out_ptr.add(i),
            _mm256_mul_ps(_mm256_loadu_ps(out_ptr.add(i)), inv8),
        );
        i += 8;
    }
    let inv4 = _mm_set1_ps(1.0 / sum_exp);
    while i + 4 <= len {
        _mm_storeu_ps(
            out_ptr.add(i),
            _mm_mul_ps(_mm_loadu_ps(out_ptr.add(i)), inv4),
        );
        i += 4;
    }
    let inv_s = 1.0 / sum_exp;
    while i < len {
        *out_ptr.add(i) *= inv_s;
        i += 1;
    }
}

// ===========================================================================
// Log-softmax SIMD implementations
// ===========================================================================

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "neon")]
unsafe fn log_softmax_row_fused_neon(input: &[f32], output: &mut [f32]) {
    use std::arch::aarch64::{vaddvq_f32, vmaxvq_f32};

    let len = input.len();
    let in_ptr = input.as_ptr();
    let out_ptr = output.as_mut_ptr();

    // 1. Find max (NEON reduce)
    let mut acc_max = vdupq_n_f32(f32::NEG_INFINITY);
    let mut i = 0usize;
    while i + 16 <= len {
        let v0 = vld1q_f32(in_ptr.add(i));
        let v1 = vld1q_f32(in_ptr.add(i + 4));
        let v2 = vld1q_f32(in_ptr.add(i + 8));
        let v3 = vld1q_f32(in_ptr.add(i + 12));
        acc_max = vmaxq_f32(acc_max, vmaxq_f32(vmaxq_f32(v0, v1), vmaxq_f32(v2, v3)));
        i += 16;
    }
    while i + 4 <= len {
        acc_max = vmaxq_f32(acc_max, vld1q_f32(in_ptr.add(i)));
        i += 4;
    }
    let mut max_val = vmaxvq_f32(acc_max);
    while i < len {
        max_val = max_val.max(*in_ptr.add(i));
        i += 1;
    }

    // 2. sum(exp(x - max))
    let off = vdupq_n_f32(max_val);
    let mut acc_sum = vdupq_n_f32(0.0);
    i = 0;
    while i + 16 <= len {
        let e0 = fast_exp_neon(vsubq_f32(vld1q_f32(in_ptr.add(i)), off));
        let e1 = fast_exp_neon(vsubq_f32(vld1q_f32(in_ptr.add(i + 4)), off));
        let e2 = fast_exp_neon(vsubq_f32(vld1q_f32(in_ptr.add(i + 8)), off));
        let e3 = fast_exp_neon(vsubq_f32(vld1q_f32(in_ptr.add(i + 12)), off));
        acc_sum = vaddq_f32(acc_sum, vaddq_f32(vaddq_f32(e0, e1), vaddq_f32(e2, e3)));
        i += 16;
    }
    while i + 4 <= len {
        let e = fast_exp_neon(vsubq_f32(vld1q_f32(in_ptr.add(i)), off));
        acc_sum = vaddq_f32(acc_sum, e);
        i += 4;
    }
    let mut sum_exp = vaddvq_f32(acc_sum);
    while i < len {
        sum_exp += (*in_ptr.add(i) - max_val).exp();
        i += 1;
    }

    // 3. output[i] = x[i] - (max + log(sum_exp))
    let log_denom = vdupq_n_f32(max_val + sum_exp.ln());
    i = 0;
    while i + 16 <= len {
        vst1q_f32(
            out_ptr.add(i),
            vsubq_f32(vld1q_f32(in_ptr.add(i)), log_denom),
        );
        vst1q_f32(
            out_ptr.add(i + 4),
            vsubq_f32(vld1q_f32(in_ptr.add(i + 4)), log_denom),
        );
        vst1q_f32(
            out_ptr.add(i + 8),
            vsubq_f32(vld1q_f32(in_ptr.add(i + 8)), log_denom),
        );
        vst1q_f32(
            out_ptr.add(i + 12),
            vsubq_f32(vld1q_f32(in_ptr.add(i + 12)), log_denom),
        );
        i += 16;
    }
    while i + 4 <= len {
        vst1q_f32(
            out_ptr.add(i),
            vsubq_f32(vld1q_f32(in_ptr.add(i)), log_denom),
        );
        i += 4;
    }
    let log_denom_s = max_val + sum_exp.ln();
    while i < len {
        *out_ptr.add(i) = *in_ptr.add(i) - log_denom_s;
        i += 1;
    }
}

/// SSE fused log-softmax.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "sse")]
unsafe fn log_softmax_row_fused_sse(input: &[f32], output: &mut [f32]) {
    let len = input.len();
    let in_ptr = input.as_ptr();
    let out_ptr = output.as_mut_ptr();

    // 1. max
    let mut acc_max = _mm_set1_ps(f32::NEG_INFINITY);
    let mut i = 0usize;
    while i + 4 <= len {
        acc_max = _mm_max_ps(acc_max, _mm_loadu_ps(in_ptr.add(i)));
        i += 4;
    }
    let mut buf = [0.0f32; 4];
    _mm_storeu_ps(buf.as_mut_ptr(), acc_max);
    let mut max_val = buf[0].max(buf[1]).max(buf[2].max(buf[3]));
    while i < len {
        max_val = max_val.max(*in_ptr.add(i));
        i += 1;
    }

    // 2. sum(exp(x - max))
    let off = _mm_set1_ps(max_val);
    let mut acc_sum = _mm_setzero_ps();
    i = 0;
    while i + 4 <= len {
        let e = fast_exp_sse(_mm_sub_ps(_mm_loadu_ps(in_ptr.add(i)), off));
        acc_sum = _mm_add_ps(acc_sum, e);
        i += 4;
    }
    _mm_storeu_ps(buf.as_mut_ptr(), acc_sum);
    let mut sum_exp = buf[0] + buf[1] + buf[2] + buf[3];
    while i < len {
        sum_exp += (*in_ptr.add(i) - max_val).exp();
        i += 1;
    }

    // 3. output[i] = x[i] - (max + log(sum_exp))
    let log_denom = _mm_set1_ps(max_val + sum_exp.ln());
    i = 0;
    while i + 4 <= len {
        _mm_storeu_ps(
            out_ptr.add(i),
            _mm_sub_ps(_mm_loadu_ps(in_ptr.add(i)), log_denom),
        );
        i += 4;
    }
    let log_denom_s = max_val + sum_exp.ln();
    while i < len {
        *out_ptr.add(i) = *in_ptr.add(i) - log_denom_s;
        i += 1;
    }
}

/// AVX fused log-softmax.
#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx512f")]
unsafe fn log_softmax_row_fused_avx512(input: &[f32], output: &mut [f32]) {
    let len = input.len();
    let in_ptr = input.as_ptr();
    let out_ptr = output.as_mut_ptr();

    let mut acc_max = _mm512_set1_ps(f32::NEG_INFINITY);
    let mut i = 0usize;
    while i + 16 <= len {
        acc_max = _mm512_max_ps(acc_max, _mm512_loadu_ps(in_ptr.add(i)));
        i += 16;
    }
    let mut max_val = _mm512_reduce_max_ps(acc_max);
    if i + 8 <= len {
        let v = _mm256_loadu_ps(in_ptr.add(i));
        let mut buf8 = [0.0f32; 8];
        _mm256_storeu_ps(buf8.as_mut_ptr(), v);
        for &x in &buf8 {
            max_val = max_val.max(x);
        }
        i += 8;
    }
    while i < len {
        max_val = max_val.max(*in_ptr.add(i));
        i += 1;
    }

    let off = _mm512_set1_ps(max_val);
    let mut acc_sum = _mm512_setzero_ps();
    i = 0;
    while i + 16 <= len {
        let e = fast_exp_avx512(_mm512_sub_ps(_mm512_loadu_ps(in_ptr.add(i)), off));
        acc_sum = _mm512_add_ps(acc_sum, e);
        i += 16;
    }
    let mut sum_exp = _mm512_reduce_add_ps(acc_sum);
    if i + 8 <= len {
        let off8 = _mm256_set1_ps(max_val);
        let e = fast_exp_avx(_mm256_sub_ps(_mm256_loadu_ps(in_ptr.add(i)), off8));
        let mut buf8 = [0.0f32; 8];
        _mm256_storeu_ps(buf8.as_mut_ptr(), e);
        sum_exp += buf8.iter().sum::<f32>();
        i += 8;
    }
    while i < len {
        sum_exp += (*in_ptr.add(i) - max_val).exp();
        i += 1;
    }

    let log_denom_val = max_val + sum_exp.ln();
    let log_denom = _mm512_set1_ps(log_denom_val);
    i = 0;
    while i + 16 <= len {
        _mm512_storeu_ps(
            out_ptr.add(i),
            _mm512_sub_ps(_mm512_loadu_ps(in_ptr.add(i)), log_denom),
        );
        i += 16;
    }
    if i + 8 <= len {
        let log_denom8 = _mm256_set1_ps(log_denom_val);
        _mm256_storeu_ps(
            out_ptr.add(i),
            _mm256_sub_ps(_mm256_loadu_ps(in_ptr.add(i)), log_denom8),
        );
        i += 8;
    }
    while i < len {
        *out_ptr.add(i) = *in_ptr.add(i) - log_denom_val;
        i += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx")]
unsafe fn log_softmax_row_fused_avx(input: &[f32], output: &mut [f32]) {
    let len = input.len();
    let in_ptr = input.as_ptr();
    let out_ptr = output.as_mut_ptr();

    // 1. max
    let mut acc_max = _mm256_set1_ps(f32::NEG_INFINITY);
    let mut i = 0usize;
    while i + 8 <= len {
        acc_max = _mm256_max_ps(acc_max, _mm256_loadu_ps(in_ptr.add(i)));
        i += 8;
    }
    let mut buf8 = [0.0f32; 8];
    _mm256_storeu_ps(buf8.as_mut_ptr(), acc_max);
    let mut max_val = buf8[0];
    for &v in &buf8[1..] {
        max_val = max_val.max(v);
    }
    while i < len {
        max_val = max_val.max(*in_ptr.add(i));
        i += 1;
    }

    // 2. sum(exp(x - max))
    let off = _mm256_set1_ps(max_val);
    let mut acc_sum = _mm256_setzero_ps();
    i = 0;
    while i + 8 <= len {
        let e = fast_exp_avx(_mm256_sub_ps(_mm256_loadu_ps(in_ptr.add(i)), off));
        acc_sum = _mm256_add_ps(acc_sum, e);
        i += 8;
    }
    _mm256_storeu_ps(buf8.as_mut_ptr(), acc_sum);
    let mut sum_exp: f32 = buf8.iter().sum();
    // SSE tail for remaining < 8 elements
    let off4 = _mm_set1_ps(max_val);
    while i + 4 <= len {
        let e = fast_exp_sse(_mm_sub_ps(_mm_loadu_ps(in_ptr.add(i)), off4));
        let mut b4 = [0.0f32; 4];
        _mm_storeu_ps(b4.as_mut_ptr(), e);
        sum_exp += b4[0] + b4[1] + b4[2] + b4[3];
        i += 4;
    }
    while i < len {
        sum_exp += (*in_ptr.add(i) - max_val).exp();
        i += 1;
    }

    // 3. output[i] = x[i] - (max + log(sum_exp))
    let log_denom_val = max_val + sum_exp.ln();
    let log_denom8 = _mm256_set1_ps(log_denom_val);
    i = 0;
    while i + 8 <= len {
        _mm256_storeu_ps(
            out_ptr.add(i),
            _mm256_sub_ps(_mm256_loadu_ps(in_ptr.add(i)), log_denom8),
        );
        i += 8;
    }
    let log_denom4 = _mm_set1_ps(log_denom_val);
    while i + 4 <= len {
        _mm_storeu_ps(
            out_ptr.add(i),
            _mm_sub_ps(_mm_loadu_ps(in_ptr.add(i)), log_denom4),
        );
        i += 4;
    }
    while i < len {
        *out_ptr.add(i) = *in_ptr.add(i) - log_denom_val;
        i += 1;
    }
}
