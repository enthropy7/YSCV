// ===========================================================================
// ReLU, Sigmoid, SiLU dispatch + implementations
// ===========================================================================

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::{
    vaddq_f32, vdivq_f32, vdupq_n_f32, vld1q_f32, vmaxq_f32, vmulq_f32, vnegq_f32, vst1q_f32,
};
#[cfg(target_arch = "x86")]
use std::arch::x86::{
    _mm_add_ps, _mm_loadu_ps, _mm_max_ps, _mm_mul_ps, _mm_set1_ps, _mm_setzero_ps, _mm_storeu_ps,
    _mm_sub_ps, _mm256_add_ps, _mm256_div_ps, _mm256_loadu_ps, _mm256_max_ps, _mm256_mul_ps,
    _mm256_set1_ps, _mm256_setzero_ps, _mm256_storeu_ps, _mm256_sub_ps,
};
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{
    _mm_add_ps, _mm_loadu_ps, _mm_max_ps, _mm_mul_ps, _mm_set1_ps, _mm_setzero_ps, _mm_storeu_ps,
    _mm_sub_ps, _mm256_add_ps, _mm256_div_ps, _mm256_loadu_ps, _mm256_max_ps, _mm256_mul_ps,
    _mm256_set1_ps, _mm256_setzero_ps, _mm256_storeu_ps, _mm256_sub_ps,
};

#[cfg(target_arch = "aarch64")]
use super::exp::fast_exp_sigmoid_neon;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use super::exp::{fast_exp_avx, fast_exp_bittrick_avx, fast_exp_bittrick_sse, fast_exp_sse};

// ===========================================================================
// ReLU dispatch
// ===========================================================================

#[allow(unsafe_code)]
#[inline]
pub fn relu_slice_dispatch(values: &mut [f32]) {
    if cfg!(miri) {
        // SAFETY: scalar path only reads/writes within `values` bounds.
        unsafe {
            relu_slice_scalar(values);
        }
        return;
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("avx") {
            // SAFETY: guarded by runtime feature detection.
            unsafe {
                relu_slice_avx(values);
            }
            return;
        }
        if std::is_x86_feature_detected!("sse") {
            // SAFETY: guarded by runtime feature detection.
            unsafe {
                relu_slice_sse(values);
            }
            return;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            // SAFETY: guarded by runtime feature detection.
            unsafe {
                relu_slice_neon(values);
            }
            return;
        }
    }

    // SAFETY: scalar path only reads/writes within `values` bounds.
    unsafe {
        relu_slice_scalar(values);
    }
}

/// Two-argument ReLU: `output[i] = max(0, input[i])`.
///
/// Avoids the clone+in-place pattern by reading from `input` and writing to
/// `output` in a single pass, halving memory traffic.
#[allow(unsafe_code)]
#[inline]
pub fn relu_to_slice_dispatch(input: &[f32], output: &mut [f32]) {
    debug_assert_eq!(input.len(), output.len());

    if cfg!(miri) {
        // SAFETY: scalar path only reads/writes within bounds.
        unsafe {
            relu_to_slice_scalar(input, output);
        }
        return;
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("avx") {
            // SAFETY: guarded by runtime feature detection.
            unsafe {
                relu_to_slice_avx(input, output);
            }
            return;
        }
        if std::is_x86_feature_detected!("sse") {
            // SAFETY: guarded by runtime feature detection.
            unsafe {
                relu_to_slice_sse(input, output);
            }
            return;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            // SAFETY: guarded by runtime feature detection.
            unsafe {
                relu_to_slice_neon(input, output);
            }
            return;
        }
    }

    // SAFETY: scalar path only reads/writes within bounds.
    unsafe {
        relu_to_slice_scalar(input, output);
    }
}

// ===========================================================================
// Sigmoid dispatch
// ===========================================================================

#[inline]
#[allow(dead_code)]
pub(crate) fn sigmoid_slice(values: &mut [f32]) {
    for value in values {
        *value = sigmoid_scalar(*value);
    }
}

#[inline]
pub(crate) fn sigmoid_scalar(value: f32) -> f32 {
    if value >= 0.0 {
        let z = (-value).exp();
        1.0 / (1.0 + z)
    } else {
        let z = value.exp();
        z / (1.0 + z)
    }
}

/// Fast sigmoid applied element-wise: `output[i] = 1 / (1 + exp(-input[i]))`.
#[allow(unsafe_code, clippy::needless_return)]
#[inline]
pub fn sigmoid_slice_dispatch(input: &[f32], output: &mut [f32]) {
    debug_assert_eq!(input.len(), output.len());

    if cfg!(miri) {
        sigmoid_slice_dispatch_scalar(input, output);
        return;
    }

    // NEON / AVX / SSE dispatch for sigmoid.
    {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("avx") {
                // SAFETY: guarded by runtime feature detection.
                unsafe {
                    sigmoid_slice_avx(input, output);
                }
                return;
            }
            if std::is_x86_feature_detected!("sse") {
                // SAFETY: guarded by runtime feature detection.
                unsafe {
                    sigmoid_slice_sse(input, output);
                }
                return;
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                unsafe {
                    sigmoid_slice_neon(input, output);
                }
                return;
            }
        }

        sigmoid_slice_dispatch_scalar(input, output);
    }
}

// ===========================================================================
// SiLU dispatch
// ===========================================================================

/// Fused SiLU (Swish) applied element-wise: `output[i] = input[i] * sigmoid(input[i])`.
///
/// Single-pass over the data avoids the 2x bandwidth penalty of separate sigmoid + multiply.
#[allow(unsafe_code)]
#[inline]
pub fn silu_slice_dispatch(input: &[f32], output: &mut [f32]) {
    debug_assert_eq!(input.len(), output.len());

    if cfg!(miri) {
        silu_slice_dispatch_scalar(input, output);
        return;
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe {
                silu_slice_neon(input, output);
            }
            return;
        }
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("avx") {
            unsafe { silu_slice_avx(input, output) };
            return;
        }
        if std::is_x86_feature_detected!("sse") {
            unsafe { silu_slice_sse(input, output) };
            return;
        }
    }

    silu_slice_dispatch_scalar(input, output);
}

/// In-place SiLU on a mutable slice. Avoids creating aliasing `&[f32]` +
/// `&mut [f32]` references (which would be UB) by working through a single
/// `&mut [f32]` and deriving only raw pointers for SIMD.
#[allow(unsafe_code)]
#[inline]
pub fn silu_inplace(data: &mut [f32]) {
    #[cfg(target_arch = "aarch64")]
    {
        if !cfg!(miri) && std::arch::is_aarch64_feature_detected!("neon") {
            // SAFETY: NEON loads full 4-element vectors before storing back,
            // so src==dst is safe. Only raw pointers used -- no aliasing refs.
            unsafe {
                silu_inplace_neon(data.as_mut_ptr(), data.len());
            }
            return;
        }
    }

    // Scalar fallback (auto-vectorizable, no aliasing issues)
    for v in data.iter_mut() {
        let x = *v;
        *v = x / (1.0 + (-x).exp());
    }
}

/// Fused bias + ReLU for NHWC output: `output[row*n + c] = max(0, output[row*n + c] + bias[c])`.
#[allow(unsafe_code)]
pub fn bias_relu_nhwc_dispatch(output: &mut [f32], bias: &[f32], m: usize, n: usize) {
    debug_assert!(output.len() >= m * n);
    debug_assert!(bias.len() >= n);

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("avx") {
            unsafe { bias_relu_nhwc_avx(output, bias, m, n) };
            return;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe { bias_relu_nhwc_neon(output, bias, m, n) };
            return;
        }
    }

    for row in output.chunks_exact_mut(n).take(m) {
        for (dst, b) in row.iter_mut().zip(bias.iter()) {
            *dst = (*dst + *b).max(0.0);
        }
    }
}

/// Fused bias add for NHWC output: `output[row*n + c] += bias[c]`.
#[allow(unsafe_code)]
pub fn bias_add_nhwc_dispatch(output: &mut [f32], bias: &[f32], m: usize, n: usize) {
    debug_assert!(output.len() >= m * n);
    debug_assert!(bias.len() >= n);

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("avx") {
            unsafe { bias_add_nhwc_avx(output, bias, m, n) };
            return;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe { bias_add_nhwc_neon(output, bias, m, n) };
            return;
        }
    }

    for row in output.chunks_exact_mut(n).take(m) {
        for (dst, b) in row.iter_mut().zip(bias.iter()) {
            *dst += *b;
        }
    }
}

/// Step S.4: fused per-row epilogue — residual + bias + activation in
/// a single SIMD pass over `out_row`. Replaces the 2-3 separate passes
/// that `row_gemm_set_parallel_fused` used to do (matmul store →
/// scalar residual add → `bias_relu_nhwc_dispatch`). Cuts post-matmul
/// memory traffic from 3× to 1× over the row.
///
/// `activation_id`: 0 = None, 1 = Relu, 2 = SiLU (matches conv::Activation
/// numeric ordering — caller converts once).
///
/// Runtime feature dispatch: AVX+FMA → fast SIMD path; otherwise scalar.
#[allow(unsafe_code)]
pub fn fused_row_epilogue_dispatch(
    out_row: &mut [f32],
    residual: Option<&[f32]>,
    bias: Option<&[f32]>,
    activation_id: u8,
    n: usize,
) {
    debug_assert!(out_row.len() >= n);
    if let Some(r) = residual {
        debug_assert!(r.len() >= n);
    }
    if let Some(b) = bias {
        debug_assert!(b.len() >= n);
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("avx") && std::is_x86_feature_detected!("fma") {
            // SAFETY: AVX+FMA feature-gated above. Slice lengths
            // checked against n by debug_asserts.
            unsafe {
                fused_row_epilogue_avx_fma(out_row, residual, bias, activation_id, n);
            }
            return;
        }
    }
    fused_row_epilogue_scalar(out_row, residual, bias, activation_id, n);
}

fn fused_row_epilogue_scalar(
    out_row: &mut [f32],
    residual: Option<&[f32]>,
    bias: Option<&[f32]>,
    activation_id: u8,
    n: usize,
) {
    for j in 0..n {
        let mut x = out_row[j];
        if let Some(r) = residual {
            x += r[j];
        }
        if let Some(b) = bias {
            x += b[j];
        }
        x = match activation_id {
            1 => x.max(0.0),
            2 => x / (1.0 + (-x).exp()),
            _ => x,
        };
        out_row[j] = x;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx,fma")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
unsafe fn fused_row_epilogue_avx_fma(
    out_row: &mut [f32],
    residual: Option<&[f32]>,
    bias: Option<&[f32]>,
    activation_id: u8,
    n: usize,
) {
    let out_ptr = out_row.as_mut_ptr();
    let res_ptr = residual.map(|r| r.as_ptr());
    let bias_ptr = bias.map(|b| b.as_ptr());
    let zero = _mm256_setzero_ps();
    let mut j = 0usize;

    // SIMD body: process 8 lanes per iteration. Has-residual / has-bias
    // conditions evaluate to the same branch for every iteration in this
    // call, so the branch predictor trivially handles them.
    while j + 8 <= n {
        let mut v = _mm256_loadu_ps(out_ptr.add(j));
        if let Some(rp) = res_ptr {
            v = _mm256_add_ps(v, _mm256_loadu_ps(rp.add(j)));
        }
        if let Some(bp) = bias_ptr {
            v = _mm256_add_ps(v, _mm256_loadu_ps(bp.add(j)));
        }
        match activation_id {
            1 => v = _mm256_max_ps(v, zero),
            2 => {
                // SiLU via fast bit-trick exp.
                let one = _mm256_set1_ps(1.0);
                let neg_x = _mm256_sub_ps(zero, v);
                let exp_neg = fast_exp_bittrick_avx(neg_x);
                let denom = _mm256_add_ps(one, exp_neg);
                v = _mm256_div_ps(v, denom);
            }
            _ => {}
        }
        _mm256_storeu_ps(out_ptr.add(j), v);
        j += 8;
    }
    // Scalar tail.
    while j < n {
        let mut x = *out_ptr.add(j);
        if let Some(rp) = res_ptr {
            x += *rp.add(j);
        }
        if let Some(bp) = bias_ptr {
            x += *bp.add(j);
        }
        x = match activation_id {
            1 => x.max(0.0),
            2 => x / (1.0 + (-x).exp()),
            _ => x,
        };
        *out_ptr.add(j) = x;
        j += 1;
    }
}

/// Fused bias + SiLU for NHWC output: `output[row*n + c] = silu(output[row*n + c] + bias[c])`.
#[allow(unsafe_code)]
pub fn bias_silu_nhwc_dispatch(output: &mut [f32], bias: &[f32], m: usize, n: usize) {
    debug_assert!(output.len() >= m * n);
    debug_assert!(bias.len() >= n);

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("avx") {
            unsafe { bias_silu_nhwc_avx(output, bias, m, n) };
            return;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe { bias_silu_nhwc_neon(output, bias, m, n) };
            return;
        }
    }

    unsafe {
        let out_ptr = output.as_mut_ptr();
        let bias_ptr = bias.as_ptr();
        for row in 0..m {
            let base = out_ptr.add(row * n);
            for c in 0..n {
                let v = *base.add(c) + *bias_ptr.add(c);
                let s = 1.0 / (1.0 + (-v).exp());
                *base.add(c) = v * s;
            }
        }
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
unsafe fn bias_relu_nhwc_avx(output: &mut [f32], bias: &[f32], m: usize, n: usize) {
    let zero = _mm256_setzero_ps();
    let out_ptr = output.as_mut_ptr();
    let bias_ptr = bias.as_ptr();
    // Fast path: preload bias for small N to eliminate per-row bias loads.
    if n == 16 {
        let b0 = _mm256_loadu_ps(bias_ptr);
        let b1 = _mm256_loadu_ps(bias_ptr.add(8));
        for row in 0..m {
            let base = out_ptr.add(row * 16);
            _mm256_storeu_ps(
                base,
                _mm256_max_ps(_mm256_add_ps(_mm256_loadu_ps(base), b0), zero),
            );
            _mm256_storeu_ps(
                base.add(8),
                _mm256_max_ps(_mm256_add_ps(_mm256_loadu_ps(base.add(8)), b1), zero),
            );
        }
        return;
    }
    if n == 24 {
        let b0 = _mm256_loadu_ps(bias_ptr);
        let b1 = _mm256_loadu_ps(bias_ptr.add(8));
        let b2 = _mm256_loadu_ps(bias_ptr.add(16));
        for row in 0..m {
            let base = out_ptr.add(row * 24);
            _mm256_storeu_ps(
                base,
                _mm256_max_ps(_mm256_add_ps(_mm256_loadu_ps(base), b0), zero),
            );
            _mm256_storeu_ps(
                base.add(8),
                _mm256_max_ps(_mm256_add_ps(_mm256_loadu_ps(base.add(8)), b1), zero),
            );
            _mm256_storeu_ps(
                base.add(16),
                _mm256_max_ps(_mm256_add_ps(_mm256_loadu_ps(base.add(16)), b2), zero),
            );
        }
        return;
    }
    for row in 0..m {
        let base = out_ptr.add(row * n);
        let mut c = 0usize;
        while c + 8 <= n {
            let v = _mm256_add_ps(
                _mm256_loadu_ps(base.add(c)),
                _mm256_loadu_ps(bias_ptr.add(c)),
            );
            _mm256_storeu_ps(base.add(c), _mm256_max_ps(v, zero));
            c += 8;
        }
        while c < n {
            *base.add(c) = (*base.add(c) + *bias_ptr.add(c)).max(0.0);
            c += 1;
        }
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
unsafe fn bias_add_nhwc_avx(output: &mut [f32], bias: &[f32], m: usize, n: usize) {
    let out_ptr = output.as_mut_ptr();
    let bias_ptr = bias.as_ptr();
    if n == 16 {
        let b0 = _mm256_loadu_ps(bias_ptr);
        let b1 = _mm256_loadu_ps(bias_ptr.add(8));
        for row in 0..m {
            let base = out_ptr.add(row * 16);
            _mm256_storeu_ps(base, _mm256_add_ps(_mm256_loadu_ps(base), b0));
            _mm256_storeu_ps(base.add(8), _mm256_add_ps(_mm256_loadu_ps(base.add(8)), b1));
        }
        return;
    }
    if n == 24 {
        let b0 = _mm256_loadu_ps(bias_ptr);
        let b1 = _mm256_loadu_ps(bias_ptr.add(8));
        let b2 = _mm256_loadu_ps(bias_ptr.add(16));
        for row in 0..m {
            let base = out_ptr.add(row * 24);
            _mm256_storeu_ps(base, _mm256_add_ps(_mm256_loadu_ps(base), b0));
            _mm256_storeu_ps(base.add(8), _mm256_add_ps(_mm256_loadu_ps(base.add(8)), b1));
            _mm256_storeu_ps(
                base.add(16),
                _mm256_add_ps(_mm256_loadu_ps(base.add(16)), b2),
            );
        }
        return;
    }
    for row in 0..m {
        let base = out_ptr.add(row * n);
        let mut c = 0usize;
        while c + 8 <= n {
            let v = _mm256_add_ps(
                _mm256_loadu_ps(base.add(c)),
                _mm256_loadu_ps(bias_ptr.add(c)),
            );
            _mm256_storeu_ps(base.add(c), v);
            c += 8;
        }
        while c < n {
            *base.add(c) += *bias_ptr.add(c);
            c += 1;
        }
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
unsafe fn bias_silu_nhwc_avx(output: &mut [f32], bias: &[f32], m: usize, n: usize) {
    let one = _mm256_set1_ps(1.0);
    let out_ptr = output.as_mut_ptr();
    let bias_ptr = bias.as_ptr();
    for row in 0..m {
        let base = out_ptr.add(row * n);
        let mut c = 0usize;
        while c + 8 <= n {
            let v = _mm256_add_ps(
                _mm256_loadu_ps(base.add(c)),
                _mm256_loadu_ps(bias_ptr.add(c)),
            );
            let neg_v = _mm256_sub_ps(_mm256_setzero_ps(), v);
            let exp_neg = fast_exp_avx(neg_v);
            let sig = _mm256_div_ps(one, _mm256_add_ps(one, exp_neg));
            _mm256_storeu_ps(base.add(c), _mm256_mul_ps(v, sig));
            c += 8;
        }
        while c < n {
            let v = *base.add(c) + *bias_ptr.add(c);
            let s = 1.0 / (1.0 + (-v).exp());
            *base.add(c) = v * s;
            c += 1;
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
unsafe fn bias_relu_nhwc_neon(output: &mut [f32], bias: &[f32], m: usize, n: usize) {
    let zero = vdupq_n_f32(0.0);
    let out_ptr = output.as_mut_ptr();
    let bias_ptr = bias.as_ptr();
    // Preloaded fast paths for common channel counts.
    if n == 16 {
        let b0 = vld1q_f32(bias_ptr);
        let b1 = vld1q_f32(bias_ptr.add(4));
        let b2 = vld1q_f32(bias_ptr.add(8));
        let b3 = vld1q_f32(bias_ptr.add(12));
        for row in 0..m {
            let base = out_ptr.add(row * 16);
            vst1q_f32(base, vmaxq_f32(vaddq_f32(vld1q_f32(base), b0), zero));
            vst1q_f32(
                base.add(4),
                vmaxq_f32(vaddq_f32(vld1q_f32(base.add(4)), b1), zero),
            );
            vst1q_f32(
                base.add(8),
                vmaxq_f32(vaddq_f32(vld1q_f32(base.add(8)), b2), zero),
            );
            vst1q_f32(
                base.add(12),
                vmaxq_f32(vaddq_f32(vld1q_f32(base.add(12)), b3), zero),
            );
        }
        return;
    }
    if n == 24 {
        let b0 = vld1q_f32(bias_ptr);
        let b1 = vld1q_f32(bias_ptr.add(4));
        let b2 = vld1q_f32(bias_ptr.add(8));
        let b3 = vld1q_f32(bias_ptr.add(12));
        let b4 = vld1q_f32(bias_ptr.add(16));
        let b5 = vld1q_f32(bias_ptr.add(20));
        for row in 0..m {
            let base = out_ptr.add(row * 24);
            vst1q_f32(base, vmaxq_f32(vaddq_f32(vld1q_f32(base), b0), zero));
            vst1q_f32(
                base.add(4),
                vmaxq_f32(vaddq_f32(vld1q_f32(base.add(4)), b1), zero),
            );
            vst1q_f32(
                base.add(8),
                vmaxq_f32(vaddq_f32(vld1q_f32(base.add(8)), b2), zero),
            );
            vst1q_f32(
                base.add(12),
                vmaxq_f32(vaddq_f32(vld1q_f32(base.add(12)), b3), zero),
            );
            vst1q_f32(
                base.add(16),
                vmaxq_f32(vaddq_f32(vld1q_f32(base.add(16)), b4), zero),
            );
            vst1q_f32(
                base.add(20),
                vmaxq_f32(vaddq_f32(vld1q_f32(base.add(20)), b5), zero),
            );
        }
        return;
    }
    for row in 0..m {
        let base = out_ptr.add(row * n);
        let mut c = 0usize;
        while c + 4 <= n {
            let v = vaddq_f32(vld1q_f32(base.add(c)), vld1q_f32(bias_ptr.add(c)));
            vst1q_f32(base.add(c), vmaxq_f32(v, zero));
            c += 4;
        }
        while c < n {
            *base.add(c) = (*base.add(c) + *bias_ptr.add(c)).max(0.0);
            c += 1;
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
unsafe fn bias_add_nhwc_neon(output: &mut [f32], bias: &[f32], m: usize, n: usize) {
    let out_ptr = output.as_mut_ptr();
    let bias_ptr = bias.as_ptr();
    for row in 0..m {
        let base = out_ptr.add(row * n);
        let mut c = 0usize;
        while c + 4 <= n {
            let v = vaddq_f32(vld1q_f32(base.add(c)), vld1q_f32(bias_ptr.add(c)));
            vst1q_f32(base.add(c), v);
            c += 4;
        }
        while c < n {
            *base.add(c) += *bias_ptr.add(c);
            c += 1;
        }
    }
}

// ===========================================================================
// Scalar fallbacks
// ===========================================================================

#[allow(unsafe_code)]
#[allow(unsafe_op_in_unsafe_fn)]
pub(super) unsafe fn relu_slice_scalar(values: &mut [f32]) {
    let len = values.len();
    let ptr = values.as_mut_ptr();
    let mut index = 0usize;

    while index + 8 <= len {
        let v0 = *ptr.add(index);
        let v1 = *ptr.add(index + 1);
        let v2 = *ptr.add(index + 2);
        let v3 = *ptr.add(index + 3);
        let v4 = *ptr.add(index + 4);
        let v5 = *ptr.add(index + 5);
        let v6 = *ptr.add(index + 6);
        let v7 = *ptr.add(index + 7);
        *ptr.add(index) = v0.max(0.0);
        *ptr.add(index + 1) = v1.max(0.0);
        *ptr.add(index + 2) = v2.max(0.0);
        *ptr.add(index + 3) = v3.max(0.0);
        *ptr.add(index + 4) = v4.max(0.0);
        *ptr.add(index + 5) = v5.max(0.0);
        *ptr.add(index + 6) = v6.max(0.0);
        *ptr.add(index + 7) = v7.max(0.0);
        index += 8;
    }

    while index < len {
        *ptr.add(index) = (*ptr.add(index)).max(0.0);
        index += 1;
    }
}

#[allow(unsafe_code)]
#[allow(unsafe_op_in_unsafe_fn)]
pub(super) unsafe fn relu_to_slice_scalar(input: &[f32], output: &mut [f32]) {
    let len = input.len();
    let in_ptr = input.as_ptr();
    let out_ptr = output.as_mut_ptr();
    let mut index = 0usize;

    while index + 8 <= len {
        *out_ptr.add(index) = (*in_ptr.add(index)).max(0.0);
        *out_ptr.add(index + 1) = (*in_ptr.add(index + 1)).max(0.0);
        *out_ptr.add(index + 2) = (*in_ptr.add(index + 2)).max(0.0);
        *out_ptr.add(index + 3) = (*in_ptr.add(index + 3)).max(0.0);
        *out_ptr.add(index + 4) = (*in_ptr.add(index + 4)).max(0.0);
        *out_ptr.add(index + 5) = (*in_ptr.add(index + 5)).max(0.0);
        *out_ptr.add(index + 6) = (*in_ptr.add(index + 6)).max(0.0);
        *out_ptr.add(index + 7) = (*in_ptr.add(index + 7)).max(0.0);
        index += 8;
    }

    while index < len {
        *out_ptr.add(index) = (*in_ptr.add(index)).max(0.0);
        index += 1;
    }
}

fn sigmoid_slice_dispatch_scalar(input: &[f32], output: &mut [f32]) {
    for (o, &v) in output.iter_mut().zip(input.iter()) {
        *o = sigmoid_scalar(v);
    }
}

fn silu_slice_dispatch_scalar(input: &[f32], output: &mut [f32]) {
    for (o, &v) in output.iter_mut().zip(input.iter()) {
        let s = 1.0 / (1.0 + (-v).exp());
        *o = v * s;
    }
}

// ===========================================================================
// ReLU SIMD implementations
// ===========================================================================

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code)]
#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "sse")]
unsafe fn relu_slice_sse(values: &mut [f32]) {
    let len = values.len();
    let ptr = values.as_mut_ptr();
    let zero = _mm_setzero_ps();
    let mut index = 0usize;

    while index + 4 <= len {
        let input = _mm_loadu_ps(ptr.add(index));
        let out = _mm_max_ps(input, zero);
        _mm_storeu_ps(ptr.add(index), out);
        index += 4;
    }

    if index < len {
        relu_slice_scalar(&mut values[index..]);
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code)]
#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx")]
unsafe fn relu_slice_avx(values: &mut [f32]) {
    let len = values.len();
    let ptr = values.as_mut_ptr();
    let zero = _mm256_setzero_ps();
    let mut index = 0usize;

    // 4x unrolled: 32 elements per iteration
    while index + 32 <= len {
        let v0 = _mm256_max_ps(_mm256_loadu_ps(ptr.add(index)), zero);
        let v1 = _mm256_max_ps(_mm256_loadu_ps(ptr.add(index + 8)), zero);
        let v2 = _mm256_max_ps(_mm256_loadu_ps(ptr.add(index + 16)), zero);
        let v3 = _mm256_max_ps(_mm256_loadu_ps(ptr.add(index + 24)), zero);
        _mm256_storeu_ps(ptr.add(index), v0);
        _mm256_storeu_ps(ptr.add(index + 8), v1);
        _mm256_storeu_ps(ptr.add(index + 16), v2);
        _mm256_storeu_ps(ptr.add(index + 24), v3);
        index += 32;
    }

    while index + 8 <= len {
        _mm256_storeu_ps(
            ptr.add(index),
            _mm256_max_ps(_mm256_loadu_ps(ptr.add(index)), zero),
        );
        index += 8;
    }

    if index < len {
        relu_slice_sse(&mut values[index..]);
    }
}

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code)]
#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "neon")]
unsafe fn relu_slice_neon(values: &mut [f32]) {
    let len = values.len();
    let ptr = values.as_mut_ptr();
    let zero = vdupq_n_f32(0.0);
    let mut index = 0usize;

    // 8x unrolled: 32 elements per iteration
    while index + 32 <= len {
        let v0 = vmaxq_f32(vld1q_f32(ptr.add(index)), zero);
        let v1 = vmaxq_f32(vld1q_f32(ptr.add(index + 4)), zero);
        let v2 = vmaxq_f32(vld1q_f32(ptr.add(index + 8)), zero);
        let v3 = vmaxq_f32(vld1q_f32(ptr.add(index + 12)), zero);
        let v4 = vmaxq_f32(vld1q_f32(ptr.add(index + 16)), zero);
        let v5 = vmaxq_f32(vld1q_f32(ptr.add(index + 20)), zero);
        let v6 = vmaxq_f32(vld1q_f32(ptr.add(index + 24)), zero);
        let v7 = vmaxq_f32(vld1q_f32(ptr.add(index + 28)), zero);
        vst1q_f32(ptr.add(index), v0);
        vst1q_f32(ptr.add(index + 4), v1);
        vst1q_f32(ptr.add(index + 8), v2);
        vst1q_f32(ptr.add(index + 12), v3);
        vst1q_f32(ptr.add(index + 16), v4);
        vst1q_f32(ptr.add(index + 20), v5);
        vst1q_f32(ptr.add(index + 24), v6);
        vst1q_f32(ptr.add(index + 28), v7);
        index += 32;
    }

    while index + 4 <= len {
        vst1q_f32(ptr.add(index), vmaxq_f32(vld1q_f32(ptr.add(index)), zero));
        index += 4;
    }

    if index < len {
        relu_slice_scalar(&mut values[index..]);
    }
}

// ===========================================================================
// Two-argument ReLU SIMD implementations (input -> output)
// ===========================================================================

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code)]
#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "sse")]
unsafe fn relu_to_slice_sse(input: &[f32], output: &mut [f32]) {
    let len = input.len();
    let in_ptr = input.as_ptr();
    let out_ptr = output.as_mut_ptr();
    let zero = _mm_setzero_ps();
    let mut index = 0usize;

    while index + 4 <= len {
        let v = _mm_loadu_ps(in_ptr.add(index));
        let r = _mm_max_ps(v, zero);
        _mm_storeu_ps(out_ptr.add(index), r);
        index += 4;
    }

    if index < len {
        relu_to_slice_scalar(&input[index..], &mut output[index..]);
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code)]
#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx")]
unsafe fn relu_to_slice_avx(input: &[f32], output: &mut [f32]) {
    let len = input.len();
    let in_ptr = input.as_ptr();
    let out_ptr = output.as_mut_ptr();
    let zero = _mm256_setzero_ps();
    let mut index = 0usize;

    // 4x unrolled: 32 elements per iteration (matches NEON unrolling)
    while index + 32 <= len {
        let a0 = _mm256_loadu_ps(in_ptr.add(index));
        let a1 = _mm256_loadu_ps(in_ptr.add(index + 8));
        let a2 = _mm256_loadu_ps(in_ptr.add(index + 16));
        let a3 = _mm256_loadu_ps(in_ptr.add(index + 24));
        _mm256_storeu_ps(out_ptr.add(index), _mm256_max_ps(a0, zero));
        _mm256_storeu_ps(out_ptr.add(index + 8), _mm256_max_ps(a1, zero));
        _mm256_storeu_ps(out_ptr.add(index + 16), _mm256_max_ps(a2, zero));
        _mm256_storeu_ps(out_ptr.add(index + 24), _mm256_max_ps(a3, zero));
        index += 32;
    }

    while index + 8 <= len {
        _mm256_storeu_ps(
            out_ptr.add(index),
            _mm256_max_ps(_mm256_loadu_ps(in_ptr.add(index)), zero),
        );
        index += 8;
    }

    if index < len {
        relu_to_slice_sse(&input[index..], &mut output[index..]);
    }
}

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code)]
#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "neon")]
unsafe fn relu_to_slice_neon(input: &[f32], output: &mut [f32]) {
    let len = input.len();
    let in_ptr = input.as_ptr();
    let out_ptr = output.as_mut_ptr();
    let zero = vdupq_n_f32(0.0);
    let mut index = 0usize;

    // 8x unrolled with interleaved load/compute/store for better OoO pipelining
    while index + 32 <= len {
        let a0 = vld1q_f32(in_ptr.add(index));
        let a1 = vld1q_f32(in_ptr.add(index + 4));
        let a2 = vld1q_f32(in_ptr.add(index + 8));
        let a3 = vld1q_f32(in_ptr.add(index + 12));
        vst1q_f32(out_ptr.add(index), vmaxq_f32(a0, zero));
        vst1q_f32(out_ptr.add(index + 4), vmaxq_f32(a1, zero));
        let a4 = vld1q_f32(in_ptr.add(index + 16));
        let a5 = vld1q_f32(in_ptr.add(index + 20));
        vst1q_f32(out_ptr.add(index + 8), vmaxq_f32(a2, zero));
        vst1q_f32(out_ptr.add(index + 12), vmaxq_f32(a3, zero));
        let a6 = vld1q_f32(in_ptr.add(index + 24));
        let a7 = vld1q_f32(in_ptr.add(index + 28));
        vst1q_f32(out_ptr.add(index + 16), vmaxq_f32(a4, zero));
        vst1q_f32(out_ptr.add(index + 20), vmaxq_f32(a5, zero));
        vst1q_f32(out_ptr.add(index + 24), vmaxq_f32(a6, zero));
        vst1q_f32(out_ptr.add(index + 28), vmaxq_f32(a7, zero));
        index += 32;
    }

    while index + 4 <= len {
        vst1q_f32(
            out_ptr.add(index),
            vmaxq_f32(vld1q_f32(in_ptr.add(index)), zero),
        );
        index += 4;
    }

    if index < len {
        relu_to_slice_scalar(&input[index..], &mut output[index..]);
    }
}

// ===========================================================================
// Sigmoid slice implementations: sigmoid(x) = 1 / (1 + exp(-x))
// ===========================================================================

/// Sigmoid via hand-scheduled NEON assembly.
///
/// Processes 4 elements per iteration with interleaved load/compute/store.
/// The FMA pipeline is kept fully saturated by overlapping independent operations.
#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code)]
unsafe fn sigmoid_slice_neon(input: &[f32], output: &mut [f32]) {
    let len = input.len();
    let mut inp = input.as_ptr();
    let mut out = output.as_mut_ptr();
    let mut remaining = len;

    // Load all constants ONCE before the loop, keep in NEON registers
    if remaining >= 4 {
        unsafe {
            // Constants on stack for ld1r broadcast
            let c_neg88: f32 = -88.0;
            let c_pos88: f32 = 88.0;
            // Schraudolph 1999 constants: exp(x) ~ reinterpret(int(x * C + B))
            // C = 2^23 / ln(2) = 12102203.16, B = 127 * 2^23 = 1065353216
            // WHY: 2^23/ln(2) maps float mantissa bits to IEEE 754 exponent field; 127*2^23 adds the exponent bias.
            let c_schr_c: f32 = 12102203.0; // 2^23 / ln(2)
            let c_schr_b: i32 = 127 << 23; // 1065353216 as integer
            let c_sixth: f32 = 1.0 / 6.0;
            let c_half: f32 = 0.5;
            let c_one: f32 = 1.0;
            let c_127: i32 = 127;

            // Load constants into NEON registers (stays there for entire loop)
            std::arch::asm!(
                "ld1r {{v16.4s}}, [{p_neg88}]",
                "ld1r {{v17.4s}}, [{p_pos88}]",
                "ld1r {{v18.4s}}, [{p_schr_c}]",   // Schraudolph C (float)
                "dup  v19.4s, {p_schr_b:w}",        // Schraudolph B (integer 127<<23)
                "ld1r {{v20.4s}}, [{p_sixth}]",
                "ld1r {{v21.4s}}, [{p_half}]",
                "ld1r {{v22.4s}}, [{p_one}]",
                "dup  v23.4s, {p_127:w}",
                p_neg88 = in(reg) &c_neg88,
                p_pos88 = in(reg) &c_pos88,
                p_schr_c = in(reg) &c_schr_c,
                p_schr_b = in(reg) c_schr_b,
                p_sixth = in(reg) &c_sixth,
                p_half = in(reg) &c_half,
                p_one = in(reg) &c_one,
                p_127 = in(reg) c_127,
                out("v16") _, out("v17") _, out("v18") _, out("v19") _,
                out("v20") _, out("v21") _, out("v22") _, out("v23") _,
            );

            // Schraudolph bit-trick: exp(x) ~ reinterpret_f32(int(x * 2^23/ln2) + 127<<23)
            // Proper integer arithmetic: fcvtzs to get int, then add bias as int, then reinterpret
            // 4x unrolled, 16 elements per iteration
            while remaining >= 16 {
                std::arch::asm!(
                    "ldp q0, q1, [{inp}]",
                    "ldp q2, q3, [{inp}, #32]",
                    "add {inp}, {inp}, #64",
                    "fneg v0.4s, v0.4s",
                    "fneg v1.4s, v1.4s",
                    "fneg v2.4s, v2.4s",
                    "fneg v3.4s, v3.4s",
                    "fmax v0.4s, v0.4s, v16.4s",
                    "fmax v1.4s, v1.4s, v16.4s",
                    "fmax v2.4s, v2.4s, v16.4s",
                    "fmax v3.4s, v3.4s, v16.4s",
                    "fmin v0.4s, v0.4s, v17.4s",
                    "fmin v1.4s, v1.4s, v17.4s",
                    "fmin v2.4s, v2.4s, v17.4s",
                    "fmin v3.4s, v3.4s, v17.4s",
                    // x * (2^23/ln2) -> convert to int
                    "fmul v0.4s, v0.4s, v18.4s",
                    "fmul v1.4s, v1.4s, v18.4s",
                    "fmul v2.4s, v2.4s, v18.4s",
                    "fmul v3.4s, v3.4s, v18.4s",
                    "fcvtzs v0.4s, v0.4s",
                    "fcvtzs v1.4s, v1.4s",
                    "fcvtzs v2.4s, v2.4s",
                    "fcvtzs v3.4s, v3.4s",
                    // + 127*2^23 (integer add)
                    "add v0.4s, v0.4s, v19.4s",
                    "add v1.4s, v1.4s, v19.4s",
                    "add v2.4s, v2.4s, v19.4s",
                    "add v3.4s, v3.4s, v19.4s",
                    // v0-v3 bits ARE exp(-x) when reinterpreted as float
                    // sigmoid = 1 / (1 + exp)
                    "fadd v0.4s, v22.4s, v0.4s",
                    "fadd v1.4s, v22.4s, v1.4s",
                    "fadd v2.4s, v22.4s, v2.4s",
                    "fadd v3.4s, v22.4s, v3.4s",
                    "fdiv v0.4s, v22.4s, v0.4s",
                    "fdiv v1.4s, v22.4s, v1.4s",
                    "fdiv v2.4s, v22.4s, v2.4s",
                    "fdiv v3.4s, v22.4s, v3.4s",
                    "stp q0, q1, [{out}]",
                    "stp q2, q3, [{out}, #32]",
                    "add {out}, {out}, #64",
                    inp = inout(reg) inp,
                    out = inout(reg) out,
                    out("v0") _, out("v1") _, out("v2") _, out("v3") _,
                );
                remaining -= 16;
            }
            // 4-element tail -- Schraudolph
            while remaining >= 4 {
                std::arch::asm!(
                    "ld1 {{v0.4s}}, [{inp}], #16",
                    "fneg v0.4s, v0.4s",
                    "fmax v0.4s, v0.4s, v16.4s",
                    "fmin v0.4s, v0.4s, v17.4s",
                    "fmul v0.4s, v0.4s, v18.4s",
                    "fcvtzs v0.4s, v0.4s",
                    "add v0.4s, v0.4s, v19.4s",
                    "fadd v0.4s, v22.4s, v0.4s",
                    "fdiv v0.4s, v22.4s, v0.4s",
                    "st1 {{v0.4s}}, [{out}], #16",
                    inp = inout(reg) inp,
                    out = inout(reg) out,
                    out("v0") _,
                );
                remaining -= 4;
            }
            // 4-element tail -- Schraudolph
            while remaining >= 4 {
                std::arch::asm!(
                    "ld1 {{v0.4s}}, [{inp}], #16",
                    "fneg v0.4s, v0.4s",
                    "fmax v0.4s, v0.4s, v16.4s",
                    "fmin v0.4s, v0.4s, v17.4s",
                    "fmul v0.4s, v0.4s, v18.4s",
                    "fcvtzs v0.4s, v0.4s",
                    "add v0.4s, v0.4s, v19.4s",
                    "fadd v0.4s, v22.4s, v0.4s",
                    "fdiv v0.4s, v22.4s, v0.4s",
                    "st1 {{v0.4s}}, [{out}], #16",
                    inp = inout(reg) inp,
                    out = inout(reg) out,
                    out("v0") _,
                );
                remaining -= 4;
            }
        }
    }

    // Scalar tail
    for i in 0..remaining {
        unsafe {
            let x = *inp.add(i);
            *out.add(i) = 1.0 / (1.0 + (-x).exp());
        }
    }
}

// (sigmoid_vdsp and silu_vdsp removed -- benchmarked slower than NEON polynomial)

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code)]
#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "sse")]
unsafe fn sigmoid_slice_sse(input: &[f32], output: &mut [f32]) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::_mm_div_ps;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::_mm_div_ps;

    let len = input.len();
    let in_ptr = input.as_ptr();
    let out_ptr = output.as_mut_ptr();
    let one = _mm_set1_ps(1.0);
    let zero = _mm_setzero_ps();
    let mut index = 0usize;

    // Process 16 elements per iteration (4 SSE registers)
    while index + 16 <= len {
        let x0 = _mm_loadu_ps(in_ptr.add(index));
        let x1 = _mm_loadu_ps(in_ptr.add(index + 4));
        let x2 = _mm_loadu_ps(in_ptr.add(index + 8));
        let x3 = _mm_loadu_ps(in_ptr.add(index + 12));

        // Bit-trick exp is sufficient for sigmoid (output clamped 0-1, errors wash out)
        let e0 = fast_exp_bittrick_sse(_mm_sub_ps(zero, x0));
        let e1 = fast_exp_bittrick_sse(_mm_sub_ps(zero, x1));
        let e2 = fast_exp_bittrick_sse(_mm_sub_ps(zero, x2));
        let e3 = fast_exp_bittrick_sse(_mm_sub_ps(zero, x3));

        let r0 = _mm_div_ps(one, _mm_add_ps(one, e0));
        let r1 = _mm_div_ps(one, _mm_add_ps(one, e1));
        let r2 = _mm_div_ps(one, _mm_add_ps(one, e2));
        let r3 = _mm_div_ps(one, _mm_add_ps(one, e3));

        _mm_storeu_ps(out_ptr.add(index), r0);
        _mm_storeu_ps(out_ptr.add(index + 4), r1);
        _mm_storeu_ps(out_ptr.add(index + 8), r2);
        _mm_storeu_ps(out_ptr.add(index + 12), r3);

        index += 16;
    }

    // Remaining 4 at a time
    while index + 4 <= len {
        let x = _mm_loadu_ps(in_ptr.add(index));
        let neg_x = _mm_sub_ps(zero, x);
        let exp_neg_x = fast_exp_bittrick_sse(neg_x);
        let denom = _mm_add_ps(one, exp_neg_x);
        let result = _mm_div_ps(one, denom);
        _mm_storeu_ps(out_ptr.add(index), result);
        index += 4;
    }

    while index < len {
        *out_ptr.add(index) = sigmoid_scalar(*in_ptr.add(index));
        index += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code)]
#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx")]
unsafe fn sigmoid_slice_avx(input: &[f32], output: &mut [f32]) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::_mm256_div_ps;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::_mm256_div_ps;

    let len = input.len();
    let in_ptr = input.as_ptr();
    let out_ptr = output.as_mut_ptr();
    let one = _mm256_set1_ps(1.0);
    let zero = _mm256_setzero_ps();
    let mut index = 0usize;

    // Process 32 elements per iteration (4 AVX registers)
    while index + 32 <= len {
        let x0 = _mm256_loadu_ps(in_ptr.add(index));
        let x1 = _mm256_loadu_ps(in_ptr.add(index + 8));
        let x2 = _mm256_loadu_ps(in_ptr.add(index + 16));
        let x3 = _mm256_loadu_ps(in_ptr.add(index + 24));

        // Use Schraudolph bit-trick exp for ~3x speedup over polynomial
        let e0 = fast_exp_bittrick_avx(_mm256_sub_ps(zero, x0));
        let e1 = fast_exp_bittrick_avx(_mm256_sub_ps(zero, x1));
        let e2 = fast_exp_bittrick_avx(_mm256_sub_ps(zero, x2));
        let e3 = fast_exp_bittrick_avx(_mm256_sub_ps(zero, x3));

        let r0 = _mm256_div_ps(one, _mm256_add_ps(one, e0));
        let r1 = _mm256_div_ps(one, _mm256_add_ps(one, e1));
        let r2 = _mm256_div_ps(one, _mm256_add_ps(one, e2));
        let r3 = _mm256_div_ps(one, _mm256_add_ps(one, e3));

        _mm256_storeu_ps(out_ptr.add(index), r0);
        _mm256_storeu_ps(out_ptr.add(index + 8), r1);
        _mm256_storeu_ps(out_ptr.add(index + 16), r2);
        _mm256_storeu_ps(out_ptr.add(index + 24), r3);

        index += 32;
    }

    // Remaining 8 at a time
    while index + 8 <= len {
        let x = _mm256_loadu_ps(in_ptr.add(index));
        let neg_x = _mm256_sub_ps(zero, x);
        let exp_neg_x = fast_exp_bittrick_avx(neg_x);
        let denom = _mm256_add_ps(one, exp_neg_x);
        let result = _mm256_div_ps(one, denom);
        _mm256_storeu_ps(out_ptr.add(index), result);
        index += 8;
    }

    if index < len {
        sigmoid_slice_sse(&input[index..], &mut output[index..]);
    }
}

// (sigmoid_slice_neon defined above at line ~291)

// ===========================================================================
// SiLU SIMD implementations
// ===========================================================================

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code, dead_code)]
#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "neon")]
/// Fused SiLU: output[i] = x * sigmoid(x) in a single pass.
/// 8x unrolled with fast 3-term exp polynomial.
unsafe fn silu_slice_neon(input: &[f32], output: &mut [f32]) {
    let len = input.len();
    let in_ptr = input.as_ptr();
    let out_ptr = output.as_mut_ptr();
    let one = vdupq_n_f32(1.0);
    let mut index = 0usize;

    // 8x unrolled: 32 elements per iteration
    while index + 32 <= len {
        let x0 = vld1q_f32(in_ptr.add(index));
        let x1 = vld1q_f32(in_ptr.add(index + 4));
        let x2 = vld1q_f32(in_ptr.add(index + 8));
        let x3 = vld1q_f32(in_ptr.add(index + 12));
        let x4 = vld1q_f32(in_ptr.add(index + 16));
        let x5 = vld1q_f32(in_ptr.add(index + 20));
        let x6 = vld1q_f32(in_ptr.add(index + 24));
        let x7 = vld1q_f32(in_ptr.add(index + 28));

        // sigmoid(x) = 1 / (1 + exp(-x))
        let e0 = fast_exp_sigmoid_neon(vnegq_f32(x0));
        let e1 = fast_exp_sigmoid_neon(vnegq_f32(x1));
        let e2 = fast_exp_sigmoid_neon(vnegq_f32(x2));
        let e3 = fast_exp_sigmoid_neon(vnegq_f32(x3));
        let e4 = fast_exp_sigmoid_neon(vnegq_f32(x4));
        let e5 = fast_exp_sigmoid_neon(vnegq_f32(x5));
        let e6 = fast_exp_sigmoid_neon(vnegq_f32(x6));
        let e7 = fast_exp_sigmoid_neon(vnegq_f32(x7));

        // silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
        vst1q_f32(
            out_ptr.add(index),
            vmulq_f32(x0, vdivq_f32(one, vaddq_f32(one, e0))),
        );
        vst1q_f32(
            out_ptr.add(index + 4),
            vmulq_f32(x1, vdivq_f32(one, vaddq_f32(one, e1))),
        );
        vst1q_f32(
            out_ptr.add(index + 8),
            vmulq_f32(x2, vdivq_f32(one, vaddq_f32(one, e2))),
        );
        vst1q_f32(
            out_ptr.add(index + 12),
            vmulq_f32(x3, vdivq_f32(one, vaddq_f32(one, e3))),
        );
        vst1q_f32(
            out_ptr.add(index + 16),
            vmulq_f32(x4, vdivq_f32(one, vaddq_f32(one, e4))),
        );
        vst1q_f32(
            out_ptr.add(index + 20),
            vmulq_f32(x5, vdivq_f32(one, vaddq_f32(one, e5))),
        );
        vst1q_f32(
            out_ptr.add(index + 24),
            vmulq_f32(x6, vdivq_f32(one, vaddq_f32(one, e6))),
        );
        vst1q_f32(
            out_ptr.add(index + 28),
            vmulq_f32(x7, vdivq_f32(one, vaddq_f32(one, e7))),
        );
        index += 32;
    }

    while index + 4 <= len {
        let x = vld1q_f32(in_ptr.add(index));
        let e = fast_exp_sigmoid_neon(vnegq_f32(x));
        let sig = vdivq_f32(one, vaddq_f32(one, e));
        vst1q_f32(out_ptr.add(index), vmulq_f32(x, sig));
        index += 4;
    }

    while index < len {
        let x = *in_ptr.add(index);
        let s = 1.0 / (1.0 + (-x).exp());
        *out_ptr.add(index) = x * s;
        index += 1;
    }
}

/// In-place SiLU using NEON. Takes raw pointer to avoid aliasing references.
#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "neon")]
unsafe fn silu_inplace_neon(ptr: *mut f32, len: usize) {
    use std::arch::aarch64::*;
    let one = vdupq_n_f32(1.0);
    let mut index = 0usize;

    while index + 32 <= len {
        let x0 = vld1q_f32(ptr.add(index));
        let x1 = vld1q_f32(ptr.add(index + 4));
        let x2 = vld1q_f32(ptr.add(index + 8));
        let x3 = vld1q_f32(ptr.add(index + 12));
        let x4 = vld1q_f32(ptr.add(index + 16));
        let x5 = vld1q_f32(ptr.add(index + 20));
        let x6 = vld1q_f32(ptr.add(index + 24));
        let x7 = vld1q_f32(ptr.add(index + 28));

        let e0 = fast_exp_sigmoid_neon(vnegq_f32(x0));
        let e1 = fast_exp_sigmoid_neon(vnegq_f32(x1));
        let e2 = fast_exp_sigmoid_neon(vnegq_f32(x2));
        let e3 = fast_exp_sigmoid_neon(vnegq_f32(x3));
        let e4 = fast_exp_sigmoid_neon(vnegq_f32(x4));
        let e5 = fast_exp_sigmoid_neon(vnegq_f32(x5));
        let e6 = fast_exp_sigmoid_neon(vnegq_f32(x6));
        let e7 = fast_exp_sigmoid_neon(vnegq_f32(x7));

        vst1q_f32(
            ptr.add(index),
            vmulq_f32(x0, vdivq_f32(one, vaddq_f32(one, e0))),
        );
        vst1q_f32(
            ptr.add(index + 4),
            vmulq_f32(x1, vdivq_f32(one, vaddq_f32(one, e1))),
        );
        vst1q_f32(
            ptr.add(index + 8),
            vmulq_f32(x2, vdivq_f32(one, vaddq_f32(one, e2))),
        );
        vst1q_f32(
            ptr.add(index + 12),
            vmulq_f32(x3, vdivq_f32(one, vaddq_f32(one, e3))),
        );
        vst1q_f32(
            ptr.add(index + 16),
            vmulq_f32(x4, vdivq_f32(one, vaddq_f32(one, e4))),
        );
        vst1q_f32(
            ptr.add(index + 20),
            vmulq_f32(x5, vdivq_f32(one, vaddq_f32(one, e5))),
        );
        vst1q_f32(
            ptr.add(index + 24),
            vmulq_f32(x6, vdivq_f32(one, vaddq_f32(one, e6))),
        );
        vst1q_f32(
            ptr.add(index + 28),
            vmulq_f32(x7, vdivq_f32(one, vaddq_f32(one, e7))),
        );
        index += 32;
    }
    while index + 4 <= len {
        let x = vld1q_f32(ptr.add(index));
        let e = fast_exp_sigmoid_neon(vnegq_f32(x));
        let sig = vdivq_f32(one, vaddq_f32(one, e));
        vst1q_f32(ptr.add(index), vmulq_f32(x, sig));
        index += 4;
    }
    while index < len {
        let x = *ptr.add(index);
        let s = 1.0 / (1.0 + (-x).exp());
        *ptr.add(index) = x * s;
        index += 1;
    }
}

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
unsafe fn bias_silu_nhwc_neon(output: &mut [f32], bias: &[f32], m: usize, n: usize) {
    let one = vdupq_n_f32(1.0);
    let out_ptr = output.as_mut_ptr();
    let bias_ptr = bias.as_ptr();

    for row in 0..m {
        let base = out_ptr.add(row * n);
        let mut c = 0usize;

        // SIMD: process 4 channels at a time
        while c + 4 <= n {
            let x = vld1q_f32(base.add(c));
            let b = vld1q_f32(bias_ptr.add(c));
            let v = vaddq_f32(x, b);
            let e = fast_exp_sigmoid_neon(vnegq_f32(v));
            let sig = vdivq_f32(one, vaddq_f32(one, e));
            vst1q_f32(base.add(c), vmulq_f32(v, sig));
            c += 4;
        }

        // Scalar tail
        while c < n {
            let v = *base.add(c) + *bias_ptr.add(c);
            let s = 1.0 / (1.0 + (-v).exp());
            *base.add(c) = v * s;
            c += 1;
        }
    }
}

/// Fused SiLU (x * sigmoid(x)) using SSE.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "sse")]
unsafe fn silu_slice_sse(input: &[f32], output: &mut [f32]) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::_mm_div_ps;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::_mm_div_ps;

    let len = input.len();
    let in_ptr = input.as_ptr();
    let out_ptr = output.as_mut_ptr();
    let one = _mm_set1_ps(1.0);
    let zero = _mm_setzero_ps();
    let mut index = 0usize;

    while index + 16 <= len {
        let x0 = _mm_loadu_ps(in_ptr.add(index));
        let x1 = _mm_loadu_ps(in_ptr.add(index + 4));
        let x2 = _mm_loadu_ps(in_ptr.add(index + 8));
        let x3 = _mm_loadu_ps(in_ptr.add(index + 12));

        // Use Schraudolph bit-trick exp for ~3x speedup
        let e0 = fast_exp_sse(_mm_sub_ps(zero, x0));
        let e1 = fast_exp_sse(_mm_sub_ps(zero, x1));
        let e2 = fast_exp_sse(_mm_sub_ps(zero, x2));
        let e3 = fast_exp_sse(_mm_sub_ps(zero, x3));

        // silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
        _mm_storeu_ps(
            out_ptr.add(index),
            _mm_mul_ps(x0, _mm_div_ps(one, _mm_add_ps(one, e0))),
        );
        _mm_storeu_ps(
            out_ptr.add(index + 4),
            _mm_mul_ps(x1, _mm_div_ps(one, _mm_add_ps(one, e1))),
        );
        _mm_storeu_ps(
            out_ptr.add(index + 8),
            _mm_mul_ps(x2, _mm_div_ps(one, _mm_add_ps(one, e2))),
        );
        _mm_storeu_ps(
            out_ptr.add(index + 12),
            _mm_mul_ps(x3, _mm_div_ps(one, _mm_add_ps(one, e3))),
        );

        index += 16;
    }

    while index + 4 <= len {
        let x = _mm_loadu_ps(in_ptr.add(index));
        let e = fast_exp_sse(_mm_sub_ps(zero, x));
        let sig = _mm_div_ps(one, _mm_add_ps(one, e));
        _mm_storeu_ps(out_ptr.add(index), _mm_mul_ps(x, sig));
        index += 4;
    }

    while index < len {
        let v = *in_ptr.add(index);
        let s = 1.0 / (1.0 + (-v).exp());
        *out_ptr.add(index) = v * s;
        index += 1;
    }
}

/// Fused SiLU (x * sigmoid(x)) using AVX.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx")]
unsafe fn silu_slice_avx(input: &[f32], output: &mut [f32]) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::_mm256_div_ps;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::_mm256_div_ps;

    let len = input.len();
    let in_ptr = input.as_ptr();
    let out_ptr = output.as_mut_ptr();
    let one = _mm256_set1_ps(1.0);
    let zero = _mm256_setzero_ps();
    let mut index = 0usize;

    while index + 32 <= len {
        let x0 = _mm256_loadu_ps(in_ptr.add(index));
        let x1 = _mm256_loadu_ps(in_ptr.add(index + 8));
        let x2 = _mm256_loadu_ps(in_ptr.add(index + 16));
        let x3 = _mm256_loadu_ps(in_ptr.add(index + 24));

        // Use Schraudolph bit-trick exp for ~3x speedup
        let e0 = fast_exp_avx(_mm256_sub_ps(zero, x0));
        let e1 = fast_exp_avx(_mm256_sub_ps(zero, x1));
        let e2 = fast_exp_avx(_mm256_sub_ps(zero, x2));
        let e3 = fast_exp_avx(_mm256_sub_ps(zero, x3));

        // silu(x) = x / (1 + exp(-x))
        _mm256_storeu_ps(
            out_ptr.add(index),
            _mm256_mul_ps(x0, _mm256_div_ps(one, _mm256_add_ps(one, e0))),
        );
        _mm256_storeu_ps(
            out_ptr.add(index + 8),
            _mm256_mul_ps(x1, _mm256_div_ps(one, _mm256_add_ps(one, e1))),
        );
        _mm256_storeu_ps(
            out_ptr.add(index + 16),
            _mm256_mul_ps(x2, _mm256_div_ps(one, _mm256_add_ps(one, e2))),
        );
        _mm256_storeu_ps(
            out_ptr.add(index + 24),
            _mm256_mul_ps(x3, _mm256_div_ps(one, _mm256_add_ps(one, e3))),
        );

        index += 32;
    }

    while index + 8 <= len {
        let x = _mm256_loadu_ps(in_ptr.add(index));
        let e = fast_exp_avx(_mm256_sub_ps(zero, x));
        let sig = _mm256_div_ps(one, _mm256_add_ps(one, e));
        _mm256_storeu_ps(out_ptr.add(index), _mm256_mul_ps(x, sig));
        index += 8;
    }

    if index < len {
        silu_slice_sse(&input[index..], &mut output[index..]);
    }
}
