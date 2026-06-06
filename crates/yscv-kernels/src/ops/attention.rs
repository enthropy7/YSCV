use yscv_tensor::{AlignedVec, Tensor};

use super::super::error::KernelError;
use super::config::ParallelElementwiseConfig;
use super::matmul::matmul_2d;
use super::norm::softmax_last_dim_with_config_and_pool;

// ---------------------------------------------------------------------------
// SIMD-accelerated dot product (multiply-accumulate)
// ---------------------------------------------------------------------------

/// Compute the dot product of two equal-length f32 slices, using SIMD when
/// available (NEON on aarch64, AVX/SSE on x86).
#[allow(unused_variables, unsafe_code)]
fn dot_product_simd(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let features = crate::host_cpu().features;

    #[cfg(target_arch = "aarch64")]
    if features.neon {
        return unsafe { dot_product_neon(a, b) };
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if features.x86_avx_fma() {
            return unsafe { dot_product_avx(a, b) };
        }
        if features.sse {
            return unsafe { dot_product_sse(a, b) };
        }
    }

    dot_product_scalar(a, b)
}

#[inline]
fn dot_product_scalar(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..a.len() {
        sum += a[i] * b[i];
    }
    sum
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
unsafe fn dot_product_neon(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64::*;
    let n = a.len();
    let ap = a.as_ptr();
    let bp = b.as_ptr();
    let mut acc0 = vdupq_n_f32(0.0);
    let mut acc1 = vdupq_n_f32(0.0);
    let mut i = 0usize;
    while i + 8 <= n {
        let va0 = vld1q_f32(ap.add(i));
        let vb0 = vld1q_f32(bp.add(i));
        acc0 = vfmaq_f32(acc0, va0, vb0);
        let va1 = vld1q_f32(ap.add(i + 4));
        let vb1 = vld1q_f32(bp.add(i + 4));
        acc1 = vfmaq_f32(acc1, va1, vb1);
        i += 8;
    }
    while i + 4 <= n {
        let va = vld1q_f32(ap.add(i));
        let vb = vld1q_f32(bp.add(i));
        acc0 = vfmaq_f32(acc0, va, vb);
        i += 4;
    }
    acc0 = vaddq_f32(acc0, acc1);
    let mut sum = vaddvq_f32(acc0);
    while i < n {
        sum += *ap.add(i) * *bp.add(i);
        i += 1;
    }
    sum
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx", enable = "fma")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
unsafe fn dot_product_avx(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let n = a.len();
    let ap = a.as_ptr();
    let bp = b.as_ptr();
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut i = 0usize;
    while i + 16 <= n {
        let va0 = _mm256_loadu_ps(ap.add(i));
        let vb0 = _mm256_loadu_ps(bp.add(i));
        acc0 = _mm256_fmadd_ps(va0, vb0, acc0);
        let va1 = _mm256_loadu_ps(ap.add(i + 8));
        let vb1 = _mm256_loadu_ps(bp.add(i + 8));
        acc1 = _mm256_fmadd_ps(va1, vb1, acc1);
        i += 16;
    }
    while i + 8 <= n {
        let va = _mm256_loadu_ps(ap.add(i));
        let vb = _mm256_loadu_ps(bp.add(i));
        acc0 = _mm256_fmadd_ps(va, vb, acc0);
        i += 8;
    }
    acc0 = _mm256_add_ps(acc0, acc1);
    // Horizontal sum of 8 floats
    let hi = _mm256_extractf128_ps(acc0, 1);
    let lo = _mm256_castps256_ps128(acc0);
    let sum128 = _mm_add_ps(lo, hi);
    let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
    let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 1));
    let mut sum = _mm_cvtss_f32(sum32);
    while i < n {
        sum += *ap.add(i) * *bp.add(i);
        i += 1;
    }
    sum
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
unsafe fn dot_product_sse(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let n = a.len();
    let ap = a.as_ptr();
    let bp = b.as_ptr();
    let mut acc0 = _mm_setzero_ps();
    let mut acc1 = _mm_setzero_ps();
    let mut i = 0usize;
    while i + 8 <= n {
        let va0 = _mm_loadu_ps(ap.add(i));
        let vb0 = _mm_loadu_ps(bp.add(i));
        acc0 = _mm_add_ps(acc0, _mm_mul_ps(va0, vb0));
        let va1 = _mm_loadu_ps(ap.add(i + 4));
        let vb1 = _mm_loadu_ps(bp.add(i + 4));
        acc1 = _mm_add_ps(acc1, _mm_mul_ps(va1, vb1));
        i += 8;
    }
    while i + 4 <= n {
        let va = _mm_loadu_ps(ap.add(i));
        let vb = _mm_loadu_ps(bp.add(i));
        acc0 = _mm_add_ps(acc0, _mm_mul_ps(va, vb));
        i += 4;
    }
    acc0 = _mm_add_ps(acc0, acc1);
    let shuf = _mm_movehl_ps(acc0, acc0);
    let sum64 = _mm_add_ps(acc0, shuf);
    let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 1));
    let mut sum = _mm_cvtss_f32(sum32);
    while i < n {
        sum += *ap.add(i) * *bp.add(i);
        i += 1;
    }
    sum
}

// ---------------------------------------------------------------------------
// SIMD-accelerated vector scale: out[i] *= scalar
// ---------------------------------------------------------------------------

#[allow(unused_variables, unsafe_code)]
fn scale_slice_simd(data: &mut [f32], scalar: f32) {
    let features = crate::host_cpu().features;

    #[cfg(target_arch = "aarch64")]
    if features.neon {
        unsafe { scale_slice_neon(data, scalar) };
        return;
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if features.avx {
            unsafe { scale_slice_avx(data, scalar) };
            return;
        }
        if features.sse {
            unsafe { scale_slice_sse(data, scalar) };
            return;
        }
    }

    for v in data.iter_mut() {
        *v *= scalar;
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
unsafe fn scale_slice_neon(data: &mut [f32], scalar: f32) {
    use std::arch::aarch64::*;
    let n = data.len();
    let p = data.as_mut_ptr();
    let vs = vdupq_n_f32(scalar);
    let mut i = 0usize;
    while i + 4 <= n {
        let v = vld1q_f32(p.add(i));
        vst1q_f32(p.add(i), vmulq_f32(v, vs));
        i += 4;
    }
    while i < n {
        *p.add(i) *= scalar;
        i += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
unsafe fn scale_slice_avx(data: &mut [f32], scalar: f32) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;
    let n = data.len();
    let p = data.as_mut_ptr();
    let vs = _mm256_set1_ps(scalar);
    let mut i = 0usize;
    while i + 8 <= n {
        let v = _mm256_loadu_ps(p.add(i));
        _mm256_storeu_ps(p.add(i), _mm256_mul_ps(v, vs));
        i += 8;
    }
    while i < n {
        *p.add(i) *= scalar;
        i += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
unsafe fn scale_slice_sse(data: &mut [f32], scalar: f32) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;
    let n = data.len();
    let p = data.as_mut_ptr();
    let vs = _mm_set1_ps(scalar);
    let mut i = 0usize;
    while i + 4 <= n {
        let v = _mm_loadu_ps(p.add(i));
        _mm_storeu_ps(p.add(i), _mm_mul_ps(v, vs));
        i += 4;
    }
    while i < n {
        *p.add(i) *= scalar;
        i += 1;
    }
}

// ---------------------------------------------------------------------------
// SIMD-accelerated fused multiply-add: out[i] += w * v[i]
// ---------------------------------------------------------------------------

#[allow(unused_variables, unsafe_code)]
fn fma_slice_simd(out: &mut [f32], v: &[f32], w: f32) {
    debug_assert_eq!(out.len(), v.len());
    let features = crate::host_cpu().features;

    #[cfg(target_arch = "aarch64")]
    if features.neon {
        unsafe { fma_slice_neon(out, v, w) };
        return;
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if features.x86_avx_fma() {
            unsafe { fma_slice_avx(out, v, w) };
            return;
        }
        if features.sse {
            unsafe { fma_slice_sse(out, v, w) };
            return;
        }
    }

    for i in 0..out.len() {
        out[i] += w * v[i];
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
unsafe fn fma_slice_neon(out: &mut [f32], v: &[f32], w: f32) {
    use std::arch::aarch64::*;
    let n = out.len();
    let op = out.as_mut_ptr();
    let vp = v.as_ptr();
    let vw = vdupq_n_f32(w);
    let mut i = 0usize;
    while i + 4 <= n {
        let o = vld1q_f32(op.add(i));
        let x = vld1q_f32(vp.add(i));
        vst1q_f32(op.add(i), vfmaq_f32(o, vw, x));
        i += 4;
    }
    while i < n {
        *op.add(i) += w * *vp.add(i);
        i += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx", enable = "fma")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
unsafe fn fma_slice_avx(out: &mut [f32], v: &[f32], w: f32) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;
    let n = out.len();
    let op = out.as_mut_ptr();
    let vp = v.as_ptr();
    let vw = _mm256_set1_ps(w);
    let mut i = 0usize;
    while i + 8 <= n {
        let o = _mm256_loadu_ps(op.add(i));
        let x = _mm256_loadu_ps(vp.add(i));
        _mm256_storeu_ps(op.add(i), _mm256_fmadd_ps(vw, x, o));
        i += 8;
    }
    while i < n {
        *op.add(i) += w * *vp.add(i);
        i += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
unsafe fn fma_slice_sse(out: &mut [f32], v: &[f32], w: f32) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;
    let n = out.len();
    let op = out.as_mut_ptr();
    let vp = v.as_ptr();
    let vw = _mm_set1_ps(w);
    let mut i = 0usize;
    while i + 4 <= n {
        let o = _mm_loadu_ps(op.add(i));
        let x = _mm_loadu_ps(vp.add(i));
        _mm_storeu_ps(op.add(i), _mm_add_ps(o, _mm_mul_ps(vw, x)));
        i += 4;
    }
    while i < n {
        *op.add(i) += w * *vp.add(i);
        i += 1;
    }
}

/// Scaled dot-product attention for 2-D (unbatched) inputs.
///
/// ```text
/// Attention(Q, K, V, mask?) = softmax(Q @ K^T / sqrt(d_k) + mask) @ V
/// ```
///
/// * `query`:  `[seq_q, d_k]`
/// * `key`:    `[seq_k, d_k]`
/// * `value`:  `[seq_k, d_v]`
/// * `mask`:   optional `[seq_q, seq_k]` additive mask (e.g. `-inf` for positions to ignore)
///
/// Returns `[seq_q, d_v]`.
pub fn scaled_dot_product_attention(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    mask: Option<&Tensor>,
) -> Result<Tensor, KernelError> {
    // --- Validate ranks -----------------------------------------------------------
    if query.rank() != 2 || key.rank() != 2 || value.rank() != 2 {
        return Err(KernelError::InvalidAttentionRank {
            query_rank: query.rank(),
            key_rank: key.rank(),
            value_rank: value.rank(),
        });
    }

    let seq_q = query.shape()[0];
    let d_k = query.shape()[1];
    let seq_k = key.shape()[0];
    let d_v = value.shape()[1];

    // Q and K must share d_k
    if key.shape()[1] != d_k {
        return Err(KernelError::AttentionDkMismatch {
            query_dk: d_k,
            key_dk: key.shape()[1],
        });
    }

    // K and V must share seq_k
    if value.shape()[0] != seq_k {
        return Err(KernelError::AttentionSeqKMismatch {
            key_seq: seq_k,
            value_seq: value.shape()[0],
        });
    }

    // Validate mask shape if provided
    if let Some(m) = mask
        && (m.rank() != 2 || m.shape()[0] != seq_q || m.shape()[1] != seq_k)
    {
        return Err(KernelError::AttentionMaskShapeMismatch {
            expected: vec![seq_q, seq_k],
            got: m.shape().to_vec(),
        });
    }

    if d_k == 0 {
        // Degenerate case: zero-width key dimension → output is all zeros.
        let out = AlignedVec::<f32>::calloc(seq_q * d_v);
        return Tensor::from_aligned(vec![seq_q, d_v], out).map_err(Into::into);
    }

    // --- 1. Transpose K → K^T [d_k, seq_k] --------------------------------------
    let key_t = transpose_2d(key);

    // --- 2. scores = Q @ K^T  →  [seq_q, seq_k] ---------------------------------
    let scores = matmul_2d(query, &key_t)?;

    // --- 3. Scale by 1/sqrt(d_k) and optionally add mask -------------------------
    let scale = (d_k as f32).sqrt().recip();
    let scores_data = scores.data();
    let total = seq_q * seq_k;
    let mut scaled = AlignedVec::<f32>::uninitialized(total);

    match mask {
        Some(m) => {
            let mask_data = m.data();
            for i in 0..total {
                scaled[i] = scores_data[i] * scale + mask_data[i];
            }
        }
        None => {
            for i in 0..total {
                scaled[i] = scores_data[i] * scale;
            }
        }
    }

    let scaled_tensor = Tensor::from_aligned(vec![seq_q, seq_k], scaled)?;

    // --- 4. Softmax along last dim (seq_k) ----------------------------------------
    let weights = softmax_last_dim_with_config_and_pool(
        &scaled_tensor,
        ParallelElementwiseConfig::disabled(),
        None,
    )?;

    // --- 5. output = weights @ V  →  [seq_q, d_v] --------------------------------
    matmul_2d(&weights, value)
}

/// Memory-efficient (flash) attention for 2-D unbatched inputs.
///
/// Instead of materializing the full `[seq_q, seq_k]` attention matrix,
/// processes in tiles of `Br × Bc` using the online softmax trick.
/// Peak memory: `O(seq_q × d_v + Br × Bc)` instead of `O(seq_q × seq_k)`.
///
/// Same API and result as `scaled_dot_product_attention`, but much lower
/// memory usage for long sequences.
pub fn flash_attention(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    mask: Option<&Tensor>,
) -> Result<Tensor, KernelError> {
    if query.rank() != 2 || key.rank() != 2 || value.rank() != 2 {
        return Err(KernelError::InvalidAttentionRank {
            query_rank: query.rank(),
            key_rank: key.rank(),
            value_rank: value.rank(),
        });
    }

    let seq_q = query.shape()[0];
    let d_k = query.shape()[1];
    let seq_k = key.shape()[0];
    let d_v = value.shape()[1];

    if key.shape()[1] != d_k {
        return Err(KernelError::AttentionDkMismatch {
            query_dk: d_k,
            key_dk: key.shape()[1],
        });
    }
    if value.shape()[0] != seq_k {
        return Err(KernelError::AttentionSeqKMismatch {
            key_seq: seq_k,
            value_seq: value.shape()[0],
        });
    }
    if let Some(m) = mask
        && (m.rank() != 2 || m.shape()[0] != seq_q || m.shape()[1] != seq_k)
    {
        return Err(KernelError::AttentionMaskShapeMismatch {
            expected: vec![seq_q, seq_k],
            got: m.shape().to_vec(),
        });
    }

    if d_k == 0 {
        let out = AlignedVec::<f32>::calloc(seq_q * d_v);
        return Tensor::from_aligned(vec![seq_q, d_v], out).map_err(Into::into);
    }

    let q = query.data();
    let k = key.data();
    let v = value.data();
    let mask_data = mask.map(Tensor::data);
    let scale = (d_k as f32).sqrt().recip();

    // Tile sizes — tuned for L1 cache (~32KB).
    // Each tile needs Br×Bc floats for scores + Br×d_v for output accumulator.
    let br = 32.min(seq_q); // query block size
    let bc = 32.min(seq_k); // key/value block size

    // Output accumulator: O[seq_q, d_v] and running log-sum-exp per query row.
    let mut out = AlignedVec::<f32>::calloc(seq_q * d_v);
    let mut row_max = vec![f32::NEG_INFINITY; seq_q]; // running max per row
    let mut row_sum = AlignedVec::<f32>::calloc(seq_q); // running exp-sum per row

    // Tile buffer for scores: Br × Bc
    let mut scores_buf = AlignedVec::<f32>::calloc(br * bc);

    // Outer loop: iterate over key/value blocks
    for j_start in (0..seq_k).step_by(bc) {
        let j_end = (j_start + bc).min(seq_k);
        let bc_actual = j_end - j_start;

        // Inner loop: iterate over query blocks
        for i_start in (0..seq_q).step_by(br) {
            let i_end = (i_start + br).min(seq_q);
            let br_actual = i_end - i_start;

            // 1. Compute scores[br_actual, bc_actual] = Q_block @ K_block^T * scale
            for qi in 0..br_actual {
                let q_row = i_start + qi;
                let q_slice = &q[q_row * d_k..(q_row + 1) * d_k];
                for ki in 0..bc_actual {
                    let k_row = j_start + ki;
                    let k_slice = &k[k_row * d_k..(k_row + 1) * d_k];
                    let dot = dot_product_simd(q_slice, k_slice);
                    let mut s = dot * scale;
                    if let Some(md) = mask_data {
                        s += md[q_row * seq_k + k_row];
                    }
                    scores_buf[qi * bc_actual + ki] = s;
                }
            }

            // 2. Online softmax update per query row
            for qi in 0..br_actual {
                let q_row = i_start + qi;
                let scores_row = &scores_buf[qi * bc_actual..(qi + 1) * bc_actual];

                // Find max in this tile's row
                let tile_max = scores_row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let prev_max = row_max[q_row];
                let new_max = prev_max.max(tile_max);

                // Rescale previous accumulator
                let rescale = (prev_max - new_max).exp();
                let prev_sum_rescaled = row_sum[q_row] * rescale;

                // Rescale previous output (SIMD)
                let out_row = &mut out[q_row * d_v..(q_row + 1) * d_v];
                scale_slice_simd(out_row, rescale);

                // Accumulate this tile: exp(score - new_max) * V (SIMD)
                let mut tile_sum = 0.0f32;
                for ki in 0..bc_actual {
                    let w = (scores_row[ki] - new_max).exp();
                    tile_sum += w;
                    let v_row = j_start + ki;
                    let v_slice = &v[v_row * d_v..(v_row + 1) * d_v];
                    fma_slice_simd(out_row, v_slice, w);
                }

                row_max[q_row] = new_max;
                row_sum[q_row] = prev_sum_rescaled + tile_sum;
            }
        }
    }

    // 3. Final normalization: divide by row_sum (SIMD)
    for qi in 0..seq_q {
        let inv_sum = if row_sum[qi] > 0.0 {
            1.0 / row_sum[qi]
        } else {
            0.0
        };
        let out_row = &mut out[qi * d_v..(qi + 1) * d_v];
        scale_slice_simd(out_row, inv_sum);
    }

    Tensor::from_aligned(vec![seq_q, d_v], out).map_err(Into::into)
}

/// Simple 2-D transpose: `[rows, cols]` → `[cols, rows]`.
fn transpose_2d(tensor: &Tensor) -> Tensor {
    let rows = tensor.shape()[0];
    let cols = tensor.shape()[1];
    let src = tensor.data();
    let mut dst = AlignedVec::<f32>::uninitialized(rows * cols);
    for r in 0..rows {
        for c in 0..cols {
            dst[c * rows + r] = src[r * cols + c];
        }
    }
    Tensor::from_aligned(vec![cols, rows], dst).expect("shape matches data")
}
