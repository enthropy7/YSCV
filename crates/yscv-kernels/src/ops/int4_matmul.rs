//! Packed-INT4 weight × fp32 activation GEMV — the LLM decode hot path.
//!
//! For `M=1` matmul (one token at a time), inference is memory-bound on
//! the weight tensor. Packing 4-bit weights doubles the bytes-per-FMA
//! efficiency over fp16 storage and quadruples it over fp32. The kernel
//! dequantises inline: load packed bytes, unpack to two `i8` nibbles,
//! sign-extend, scale by the per-group fp32 factor, multiply with the
//! corresponding activation chunk, accumulate.
//!
//! ## Layout
//!
//! - `weight_packed`: `m_w * k / 2` bytes. Each byte holds two 4-bit
//!   signed nibbles, low nibble first. Row-major: row `m` starts at
//!   byte index `m * k / 2`.
//! - `scales`: `m_w * (k / group_size)` fp32 scales. One scale per
//!   `(row, group)` pair; row-major in the same order.
//! - `activation`: `k` fp32 values, one per input channel.
//! - `output`: `m_w` fp32 accumulator targets.
//!
//! Symmetric quantization only (zero-point = 0): standard for LLM
//! weight quantization (AWQ, GPTQ, GGUF Q4_0) where the weight
//! distribution is approximately zero-mean.
//!
//! `group_size` must be even (so packed bytes don't straddle two
//! groups) and divide `k`. 32 is the industry default.

#![allow(unsafe_code, unsafe_op_in_unsafe_fn)]

#[inline]
fn unpack_low(byte: u8) -> i8 {
    // sign-extend low 4 bits: cast to i8 then arithmetic shift left 4
    // and right 4 to fill the upper bits with the sign.
    let v = (byte as i8) << 4;
    v >> 4
}

#[inline]
fn unpack_high(byte: u8) -> i8 {
    (byte as i8) >> 4
}

/// Scalar reference. Always correct; used as the test oracle and as
/// the fallback when no SIMD path matches at runtime.
pub fn packed_int4_gemv_scalar(
    weight_packed: &[u8],
    scales: &[f32],
    activation: &[f32],
    output: &mut [f32],
    m_w: usize,
    k: usize,
    group_size: usize,
) {
    debug_assert!(group_size > 0 && group_size.is_multiple_of(2));
    debug_assert!(k.is_multiple_of(group_size));
    let groups = k / group_size;
    debug_assert_eq!(weight_packed.len(), m_w * k / 2);
    debug_assert_eq!(scales.len(), m_w * groups);
    debug_assert_eq!(activation.len(), k);
    debug_assert_eq!(output.len(), m_w);

    for row in 0..m_w {
        let row_bytes = &weight_packed[row * k / 2..(row + 1) * k / 2];
        let row_scales = &scales[row * groups..(row + 1) * groups];
        let mut acc: f32 = 0.0;
        for g in 0..groups {
            let scale = row_scales[g];
            let mut group_acc: f32 = 0.0;
            let byte_base = g * group_size / 2;
            let act_base = g * group_size;
            for i in 0..group_size / 2 {
                let byte = row_bytes[byte_base + i];
                let q_lo = unpack_low(byte) as f32;
                let q_hi = unpack_high(byte) as f32;
                group_acc += q_lo * activation[act_base + 2 * i];
                group_acc += q_hi * activation[act_base + 2 * i + 1];
            }
            acc += scale * group_acc;
        }
        output[row] = acc;
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn packed_int4_gemv_neon(
    weight_packed: &[u8],
    scales: &[f32],
    activation: &[f32],
    output: &mut [f32],
    m_w: usize,
    k: usize,
    group_size: usize,
) {
    use std::arch::aarch64::*;
    debug_assert!(group_size.is_multiple_of(16));
    let groups = k / group_size;
    let mask_lo = vdupq_n_s8(0x0F);

    for row in 0..m_w {
        let row_bytes = &weight_packed[row * k / 2..(row + 1) * k / 2];
        let row_scales = &scales[row * groups..(row + 1) * groups];
        let mut acc: f32 = 0.0;
        for g in 0..groups {
            let scale = row_scales[g];
            let byte_base = g * group_size / 2;
            let act_base = g * group_size;
            let mut group_acc = vdupq_n_f32(0.0);
            // Process 16 packed bytes = 32 nibbles per inner iter.
            let mut i = 0;
            while i + 16 <= group_size / 2 {
                let raw = vld1q_u8(row_bytes.as_ptr().add(byte_base + i));
                // Low nibble: sign-extend by shifting left 4 then arithmetic
                // right 4 (within signed lanes).
                let lo_unsigned = vandq_s8(vreinterpretq_s8_u8(raw), mask_lo);
                let lo_shifted = vshlq_n_s8(lo_unsigned, 4);
                let lo = vshrq_n_s8::<4>(lo_shifted);
                let hi = vshrq_n_s8::<4>(vreinterpretq_s8_u8(raw));

                // Widen to i16, then to f32 in 4-lane chunks.
                let lo_lo16 = vmovl_s8(vget_low_s8(lo));
                let lo_hi16 = vmovl_s8(vget_high_s8(lo));
                let hi_lo16 = vmovl_s8(vget_low_s8(hi));
                let hi_hi16 = vmovl_s8(vget_high_s8(hi));

                // Activations come in pairs (lo, hi) per byte. Layout in
                // memory: a[2i], a[2i+1], a[2i+2], a[2i+3], ...
                // We need to multiply lo[j] by a[2j], hi[j] by a[2j+1].
                // Easier: scalarise this part — the compiler vectorises
                // f32 loads anyway.
                let mut tmp = [0_i8; 32];
                vst1q_s8(tmp.as_mut_ptr(), lo);
                vst1q_s8(tmp.as_mut_ptr().add(16), hi);
                let _ = (lo_lo16, lo_hi16, hi_lo16, hi_hi16);

                // Interleave: produce f32 array of length 32 with
                // [lo[0], hi[0], lo[1], hi[1], ...].
                let mut interleaved = [0.0_f32; 32];
                for j in 0..16 {
                    interleaved[2 * j] = tmp[j] as f32;
                    interleaved[2 * j + 1] = tmp[16 + j] as f32;
                }
                let act_slice = &activation[act_base + 2 * i..act_base + 2 * i + 32];
                let mut chunk_sum = vdupq_n_f32(0.0);
                for c in 0..8 {
                    let q = vld1q_f32(interleaved.as_ptr().add(c * 4));
                    let a = vld1q_f32(act_slice.as_ptr().add(c * 4));
                    chunk_sum = vfmaq_f32(chunk_sum, q, a);
                }
                group_acc = vaddq_f32(group_acc, chunk_sum);
                i += 16;
            }
            let mut group_total = vaddvq_f32(group_acc);
            // Scalar tail.
            while i < group_size / 2 {
                let byte = row_bytes[byte_base + i];
                let q_lo = unpack_low(byte) as f32;
                let q_hi = unpack_high(byte) as f32;
                group_total += q_lo * activation[act_base + 2 * i];
                group_total += q_hi * activation[act_base + 2 * i + 1];
                i += 1;
            }
            acc += scale * group_total;
        }
        output[row] = acc;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn packed_int4_gemv_avx2(
    weight_packed: &[u8],
    scales: &[f32],
    activation: &[f32],
    output: &mut [f32],
    m_w: usize,
    k: usize,
    group_size: usize,
) {
    use std::arch::x86_64::*;
    debug_assert!(group_size.is_multiple_of(16));
    let groups = k / group_size;

    for row in 0..m_w {
        let row_bytes = &weight_packed[row * k / 2..(row + 1) * k / 2];
        let row_scales = &scales[row * groups..(row + 1) * groups];
        let mut acc: f32 = 0.0;
        for g in 0..groups {
            let scale = row_scales[g];
            let byte_base = g * group_size / 2;
            let act_base = g * group_size;
            let mut group_acc = _mm256_setzero_ps();
            let mut i = 0;
            // 8 packed bytes = 16 nibbles per iter; 16 fp32 activations.
            while i + 8 <= group_size / 2 {
                let raw = _mm_loadl_epi64(row_bytes.as_ptr().add(byte_base + i) as *const __m128i);
                // Unpack 8 bytes into low/high nibbles.
                let mut tmp = [0_i8; 16];
                _mm_storel_epi64(tmp.as_mut_ptr() as *mut __m128i, raw);
                let mut interleaved = [0.0_f32; 16];
                for j in 0..8 {
                    let byte = tmp[j] as u8;
                    interleaved[2 * j] = unpack_low(byte) as f32;
                    interleaved[2 * j + 1] = unpack_high(byte) as f32;
                }
                let act_slice = &activation[act_base + 2 * i..act_base + 2 * i + 16];
                let q0 = _mm256_loadu_ps(interleaved.as_ptr());
                let q1 = _mm256_loadu_ps(interleaved.as_ptr().add(8));
                let a0 = _mm256_loadu_ps(act_slice.as_ptr());
                let a1 = _mm256_loadu_ps(act_slice.as_ptr().add(8));
                group_acc = _mm256_fmadd_ps(q0, a0, group_acc);
                group_acc = _mm256_fmadd_ps(q1, a1, group_acc);
                i += 8;
            }
            let mut buf = [0.0_f32; 8];
            _mm256_storeu_ps(buf.as_mut_ptr(), group_acc);
            let mut group_total = buf.iter().sum::<f32>();
            while i < group_size / 2 {
                let byte = row_bytes[byte_base + i];
                let q_lo = unpack_low(byte) as f32;
                let q_hi = unpack_high(byte) as f32;
                group_total += q_lo * activation[act_base + 2 * i];
                group_total += q_hi * activation[act_base + 2 * i + 1];
                i += 1;
            }
            acc += scale * group_total;
        }
        output[row] = acc;
    }
}

/// AVX-512 + FMA GEMV with vectorised nibble unpack.
///
/// Inner loop processes one 32-nibble block per FMA pair: loads 16
/// packed bytes into XMM, sign-extends each pair of nibbles into two
/// 16-element i32 lanes via `slli/srai_epi16`, converts to f32, FMAs
/// against 32 contiguous activation lanes (two ZMM). The bottleneck of
/// the AVX2 path was the per-byte scalar nibble decode into a temp
/// buffer; this kernel does the decode in 4 SIMD ops per 16 bytes.
///
/// Requires `group_size` ≥ 32 and a multiple of 32.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn packed_int4_gemv_avx512(
    weight_packed: &[u8],
    scales: &[f32],
    activation: &[f32],
    output: &mut [f32],
    m_w: usize,
    k: usize,
    group_size: usize,
) {
    use std::arch::x86_64::*;
    debug_assert!(group_size.is_multiple_of(32));
    let groups = k / group_size;
    let blocks_per_group = group_size / 32; // each block = 16 bytes = 32 nibbles

    // Constant index vectors for gathering even/odd activation lanes.
    // _mm512_permutex2var_ps(a0, idx, a1) treats `a0` as lanes 0-15 and
    // `a1` as lanes 16-31; an index `i` selects lane i across the
    // concatenation. For 32 contiguous activations, even-indexed = 2*i.
    let idx_even = _mm512_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30);
    let idx_odd = _mm512_setr_epi32(1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31);

    for row in 0..m_w {
        let row_bytes = &weight_packed[row * k / 2..(row + 1) * k / 2];
        let row_scales = &scales[row * groups..(row + 1) * groups];
        let mut acc_total = _mm512_setzero_ps();
        // Defer scaling until each group's accumulator is collapsed; we
        // multiply by the per-group fp32 scale and add into a row-wide
        // ZMM accumulator. Cheaper than scaling per-block.
        for g in 0..groups {
            let scale = row_scales[g];
            let scale_v = _mm512_set1_ps(scale);
            let byte_base = g * group_size / 2;
            let act_base = g * group_size;
            let mut group_acc = _mm512_setzero_ps();
            for blk in 0..blocks_per_group {
                let byte_off = byte_base + blk * 16;
                let act_off = act_base + blk * 32;
                // Load 16 packed bytes (32 nibbles).
                let packed = _mm_loadu_si128(row_bytes.as_ptr().add(byte_off) as *const __m128i);
                // Zero-extend bytes → 16 i16 lanes.
                let p16 = _mm256_cvtepu8_epi16(packed);
                // Low nibble: place in upper 4 bits of each 16-bit lane,
                // arithmetic-shift right 12 → signed [-8, 7].
                let lo = _mm256_srai_epi16::<12>(_mm256_slli_epi16::<12>(p16));
                // High nibble: shift left 8, arithmetic-shift right 12.
                let hi = _mm256_srai_epi16::<12>(_mm256_slli_epi16::<8>(p16));
                // i16 → i32 → f32 (16 lanes each, no cross-lane shuffle).
                let lo_f = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(lo));
                let hi_f = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(hi));
                // Gather even/odd activation lanes so the FMA pairs
                // weight[2*i] = lo[i] with activation[2*i], and
                // weight[2*i+1] = hi[i] with activation[2*i+1] — same
                // ordering the scalar reference produces.
                let a0 = _mm512_loadu_ps(activation.as_ptr().add(act_off));
                let a1 = _mm512_loadu_ps(activation.as_ptr().add(act_off + 16));
                let a_even = _mm512_permutex2var_ps(a0, idx_even, a1);
                let a_odd = _mm512_permutex2var_ps(a0, idx_odd, a1);
                group_acc = _mm512_fmadd_ps(lo_f, a_even, group_acc);
                group_acc = _mm512_fmadd_ps(hi_f, a_odd, group_acc);
            }
            acc_total = _mm512_fmadd_ps(group_acc, scale_v, acc_total);
        }
        output[row] = _mm512_reduce_add_ps(acc_total);
    }
}

/// Runtime-dispatched packed-INT4 GEMV. NEON / AVX-512 / AVX2+FMA / scalar.
pub fn packed_int4_gemv_dispatch(
    weight_packed: &[u8],
    scales: &[f32],
    activation: &[f32],
    output: &mut [f32],
    m_w: usize,
    k: usize,
    group_size: usize,
) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx512f")
            && std::is_x86_feature_detected!("avx512bw")
            && group_size.is_multiple_of(32)
        {
            unsafe {
                packed_int4_gemv_avx512(
                    weight_packed,
                    scales,
                    activation,
                    output,
                    m_w,
                    k,
                    group_size,
                )
            };
            return;
        }
        if std::is_x86_feature_detected!("avx2")
            && std::is_x86_feature_detected!("fma")
            && group_size.is_multiple_of(16)
        {
            unsafe {
                packed_int4_gemv_avx2(
                    weight_packed,
                    scales,
                    activation,
                    output,
                    m_w,
                    k,
                    group_size,
                )
            };
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") && group_size.is_multiple_of(16) {
            unsafe {
                packed_int4_gemv_neon(
                    weight_packed,
                    scales,
                    activation,
                    output,
                    m_w,
                    k,
                    group_size,
                )
            };
            return;
        }
    }
    packed_int4_gemv_scalar(
        weight_packed,
        scales,
        activation,
        output,
        m_w,
        k,
        group_size,
    );
}

/// Scalar GEMM reference. `output[m, n] = Σ_k weight[n, k] * activation[m, k]`
/// with weight packed-INT4 + per-group fp32 scales. Layout matches the
/// GEMV path (n indexes the packed M_w rows). Scaling: weights for each
/// (n, group) share one fp32 scale; output rows accumulate over all groups.
///
/// Cache reuse: each weight nibble is unpacked once per (n, group) and
/// reused across all M activation rows — the win over M parallel GEMV
/// calls is the absence of M× redundant nibble unpacks and weight reads.
pub fn packed_int4_gemm_scalar(
    weight_packed: &[u8],
    scales: &[f32],
    activation: &[f32],
    output: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    group_size: usize,
) {
    debug_assert!(group_size > 0 && group_size.is_multiple_of(2));
    debug_assert!(k.is_multiple_of(group_size));
    let groups = k / group_size;
    debug_assert_eq!(weight_packed.len(), n * k / 2);
    debug_assert_eq!(scales.len(), n * groups);
    debug_assert_eq!(activation.len(), m * k);
    debug_assert_eq!(output.len(), m * n);

    output.fill(0.0);

    // Local unpack buffer; sized for the largest expected group (256 covers
    // GGUF Q4_K up through Llama-style 256-group). Larger groups fall back
    // to scalar in the dispatch.
    let mut unpacked = [0_i8; 256];
    debug_assert!(group_size <= unpacked.len());

    for nn in 0..n {
        let row_bytes = &weight_packed[nn * k / 2..(nn + 1) * k / 2];
        let row_scales = &scales[nn * groups..(nn + 1) * groups];
        for g in 0..groups {
            let scale = row_scales[g];
            let byte_base = g * group_size / 2;
            let act_base = g * group_size;
            for i in 0..group_size / 2 {
                let byte = row_bytes[byte_base + i];
                unpacked[2 * i] = unpack_low(byte);
                unpacked[2 * i + 1] = unpack_high(byte);
            }
            for mi in 0..m {
                let mut sum: f32 = 0.0;
                let row_act = &activation[mi * k + act_base..mi * k + act_base + group_size];
                for j in 0..group_size {
                    sum += (unpacked[j] as f32) * row_act[j];
                }
                output[mi * n + nn] += scale * sum;
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn packed_int4_gemm_avx2(
    weight_packed: &[u8],
    scales: &[f32],
    activation: &[f32],
    output: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    group_size: usize,
) {
    use std::arch::x86_64::*;
    let groups = k / group_size;
    output.fill(0.0);
    let mut unpacked = [0.0_f32; 256];
    debug_assert!(group_size <= unpacked.len());
    debug_assert!(group_size.is_multiple_of(8));

    for nn in 0..n {
        let row_bytes = &weight_packed[nn * k / 2..(nn + 1) * k / 2];
        let row_scales = &scales[nn * groups..(nn + 1) * groups];
        for g in 0..groups {
            let scale = row_scales[g];
            let byte_base = g * group_size / 2;
            let act_base = g * group_size;
            for i in 0..group_size / 2 {
                let byte = row_bytes[byte_base + i];
                unpacked[2 * i] = unpack_low(byte) as f32;
                unpacked[2 * i + 1] = unpack_high(byte) as f32;
            }
            for mi in 0..m {
                let act_row = &activation[mi * k + act_base..mi * k + act_base + group_size];
                let mut acc = _mm256_setzero_ps();
                let mut j = 0;
                while j + 8 <= group_size {
                    let q = _mm256_loadu_ps(unpacked.as_ptr().add(j));
                    let a = _mm256_loadu_ps(act_row.as_ptr().add(j));
                    acc = _mm256_fmadd_ps(q, a, acc);
                    j += 8;
                }
                let mut buf = [0.0_f32; 8];
                _mm256_storeu_ps(buf.as_mut_ptr(), acc);
                let mut sum = buf.iter().sum::<f32>();
                while j < group_size {
                    sum += unpacked[j] * act_row[j];
                    j += 1;
                }
                output[mi * n + nn] += scale * sum;
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn packed_int4_gemm_neon(
    weight_packed: &[u8],
    scales: &[f32],
    activation: &[f32],
    output: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    group_size: usize,
) {
    use std::arch::aarch64::*;
    let groups = k / group_size;
    output.fill(0.0);
    let mut unpacked = [0.0_f32; 256];
    debug_assert!(group_size <= unpacked.len());
    debug_assert!(group_size.is_multiple_of(4));

    for nn in 0..n {
        let row_bytes = &weight_packed[nn * k / 2..(nn + 1) * k / 2];
        let row_scales = &scales[nn * groups..(nn + 1) * groups];
        for g in 0..groups {
            let scale = row_scales[g];
            let byte_base = g * group_size / 2;
            let act_base = g * group_size;
            for i in 0..group_size / 2 {
                let byte = row_bytes[byte_base + i];
                unpacked[2 * i] = unpack_low(byte) as f32;
                unpacked[2 * i + 1] = unpack_high(byte) as f32;
            }
            for mi in 0..m {
                let act_row = &activation[mi * k + act_base..mi * k + act_base + group_size];
                let mut acc = vdupq_n_f32(0.0);
                let mut j = 0;
                while j + 4 <= group_size {
                    let q = vld1q_f32(unpacked.as_ptr().add(j));
                    let a = vld1q_f32(act_row.as_ptr().add(j));
                    acc = vfmaq_f32(acc, q, a);
                    j += 4;
                }
                let mut sum = vaddvq_f32(acc);
                while j < group_size {
                    sum += unpacked[j] * act_row[j];
                    j += 1;
                }
                output[mi * n + nn] += scale * sum;
            }
        }
    }
}

/// Runtime-dispatched packed-INT4 GEMM. Handles `M >= 1` activation rows
/// efficiently by unpacking each weight group once and reusing across
/// all rows. For `M == 1` the single-row dispatch via
/// `packed_int4_gemv_dispatch` still has a slight edge (less per-iter
/// overhead); the runner's `exec_matmul` thresholds between the two.
pub fn packed_int4_gemm_dispatch(
    weight_packed: &[u8],
    scales: &[f32],
    activation: &[f32],
    output: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    group_size: usize,
) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2")
            && std::is_x86_feature_detected!("fma")
            && group_size.is_multiple_of(8)
            && group_size <= 256
        {
            unsafe {
                packed_int4_gemm_avx2(
                    weight_packed,
                    scales,
                    activation,
                    output,
                    m,
                    n,
                    k,
                    group_size,
                )
            };
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon")
            && group_size.is_multiple_of(4)
            && group_size <= 256
        {
            unsafe {
                packed_int4_gemm_neon(
                    weight_packed,
                    scales,
                    activation,
                    output,
                    m,
                    n,
                    k,
                    group_size,
                )
            };
            return;
        }
    }
    packed_int4_gemm_scalar(
        weight_packed,
        scales,
        activation,
        output,
        m,
        n,
        k,
        group_size,
    );
}

/// Helper: pack a row-major fp32 weight matrix into the symmetric INT4
/// format the GEMV kernel consumes. Computes per-group abs-max scales
/// (`scale = abs_max / 7`), quantises each element to `[-8, 7]`, packs
/// two nibbles per byte. Used by tests and the LLM weight pipeline.
pub fn pack_int4_symmetric_per_group(
    weights: &[f32],
    m_w: usize,
    k: usize,
    group_size: usize,
) -> (Vec<u8>, Vec<f32>) {
    debug_assert_eq!(weights.len(), m_w * k);
    debug_assert!(group_size > 0 && group_size.is_multiple_of(2));
    debug_assert!(k.is_multiple_of(group_size));
    let groups = k / group_size;

    let mut packed = vec![0_u8; m_w * k / 2];
    let mut scales = vec![0.0_f32; m_w * groups];

    for row in 0..m_w {
        for g in 0..groups {
            let slice = &weights[row * k + g * group_size..row * k + (g + 1) * group_size];
            let abs_max = slice.iter().fold(0.0_f32, |m, &v| m.max(v.abs()));
            let scale = if abs_max <= f32::EPSILON {
                1.0
            } else {
                abs_max / 7.0
            };
            scales[row * groups + g] = scale;
            let inv = 1.0 / scale;
            for i in 0..group_size / 2 {
                let v0 = (slice[2 * i] * inv).round().clamp(-8.0, 7.0) as i8;
                let v1 = (slice[2 * i + 1] * inv).round().clamp(-8.0, 7.0) as i8;
                let byte = ((v0 as u8) & 0x0F) | (((v1 as u8) & 0x0F) << 4);
                packed[row * k / 2 + g * group_size / 2 + i] = byte;
            }
        }
    }

    (packed, scales)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pseudo_random(seed: u64, n: usize) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (((s >> 33) as i64 % 2001) - 1000) as f32 * 0.001
            })
            .collect()
    }

    #[test]
    fn unpack_round_trips_extreme_nibbles() {
        // -8 (0x8) low + +7 (0x7) high
        let byte: u8 = 0x78;
        assert_eq!(unpack_low(byte), -8);
        assert_eq!(unpack_high(byte), 7);
        // 0 + 0
        assert_eq!(unpack_low(0x00), 0);
        assert_eq!(unpack_high(0x00), 0);
        // -1 (0xF) + -1 (0xF)
        assert_eq!(unpack_low(0xFF), -1);
        assert_eq!(unpack_high(0xFF), -1);
    }

    #[test]
    fn pack_then_unpack_round_trip_within_quantisation_step() {
        let m_w = 2;
        let k = 32;
        let group_size = 16;
        let weights = pseudo_random(0xDEADBEEF, m_w * k);
        let (packed, scales) = pack_int4_symmetric_per_group(&weights, m_w, k, group_size);
        // Compute reference GEMV against an all-ones activation, compare
        // against direct fp32 dot.
        let activation = vec![1.0_f32; k];
        let mut got_quant = vec![0.0_f32; m_w];
        packed_int4_gemv_scalar(
            &packed,
            &scales,
            &activation,
            &mut got_quant,
            m_w,
            k,
            group_size,
        );
        let mut expected_fp32 = vec![0.0_f32; m_w];
        for row in 0..m_w {
            for kk in 0..k {
                expected_fp32[row] += weights[row * k + kk] * activation[kk];
            }
        }
        // Symmetric int4 has per-element max error ≈ 1/(2·7) ≈ 7 %; on
        // a 16-element group the dot can drift up to ~10 % relative
        // before the law of averages kicks in. Loose check.
        for row in 0..m_w {
            let rel =
                (got_quant[row] - expected_fp32[row]).abs() / expected_fp32[row].abs().max(1e-6);
            assert!(
                rel < 0.12,
                "row {row}: got={} expected={} rel={rel}",
                got_quant[row],
                expected_fp32[row]
            );
        }
    }

    #[test]
    fn dispatch_matches_scalar_on_random_data() {
        let m_w = 4;
        let k = 128;
        let group_size = 32;
        let weights = pseudo_random(0xCAFEBABE, m_w * k);
        let (packed, scales) = pack_int4_symmetric_per_group(&weights, m_w, k, group_size);
        let activation = pseudo_random(0xF00DBA12, k);

        let mut scalar_out = vec![0.0_f32; m_w];
        packed_int4_gemv_scalar(
            &packed,
            &scales,
            &activation,
            &mut scalar_out,
            m_w,
            k,
            group_size,
        );

        let mut dispatch_out = vec![0.0_f32; m_w];
        packed_int4_gemv_dispatch(
            &packed,
            &scales,
            &activation,
            &mut dispatch_out,
            m_w,
            k,
            group_size,
        );

        for row in 0..m_w {
            assert!(
                (scalar_out[row] - dispatch_out[row]).abs() <= 1e-4,
                "row {row}: scalar={} dispatch={}",
                scalar_out[row],
                dispatch_out[row]
            );
        }
    }

    #[test]
    fn gemv_with_zero_weights_emits_zeros() {
        let m_w = 3;
        let k = 32;
        let group_size = 16;
        let weights = vec![0.0_f32; m_w * k];
        let (packed, scales) = pack_int4_symmetric_per_group(&weights, m_w, k, group_size);
        let activation = vec![1.5_f32; k];
        let mut out = vec![999.0_f32; m_w];
        packed_int4_gemv_dispatch(&packed, &scales, &activation, &mut out, m_w, k, group_size);
        for &v in &out {
            assert!(v.abs() <= 1e-5, "expected ≈0, got {v}");
        }
    }

    #[test]
    fn gemm_with_m1_matches_gemv() {
        // M=1 GEMM should produce same result as GEMV for the same data.
        let n = 4;
        let k = 64;
        let group_size = 32;
        let weights = pseudo_random(0xAA, n * k);
        let (packed, scales) = pack_int4_symmetric_per_group(&weights, n, k, group_size);
        let activation = pseudo_random(0xBB, k);
        let mut gemv_out = vec![0.0_f32; n];
        packed_int4_gemv_scalar(
            &packed,
            &scales,
            &activation,
            &mut gemv_out,
            n,
            k,
            group_size,
        );
        let mut gemm_out = vec![0.0_f32; n];
        packed_int4_gemm_scalar(
            &packed,
            &scales,
            &activation,
            &mut gemm_out,
            1,
            n,
            k,
            group_size,
        );
        for i in 0..n {
            assert!(
                (gemv_out[i] - gemm_out[i]).abs() <= 1e-5,
                "row {i}: gemv={} gemm={}",
                gemv_out[i],
                gemm_out[i]
            );
        }
    }

    #[test]
    fn gemm_dispatch_matches_scalar_on_random_shapes() {
        let n = 8;
        let k = 64;
        let group_size = 32;
        let weights = pseudo_random(0x100, n * k);
        let (packed, scales) = pack_int4_symmetric_per_group(&weights, n, k, group_size);
        for &m in &[1, 4, 16, 32] {
            let activation = pseudo_random(0x200 + m as u64, m * k);
            let mut scalar_out = vec![0.0_f32; m * n];
            packed_int4_gemm_scalar(
                &packed,
                &scales,
                &activation,
                &mut scalar_out,
                m,
                n,
                k,
                group_size,
            );
            let mut dispatch_out = vec![0.0_f32; m * n];
            packed_int4_gemm_dispatch(
                &packed,
                &scales,
                &activation,
                &mut dispatch_out,
                m,
                n,
                k,
                group_size,
            );
            for i in 0..(m * n) {
                assert!(
                    (scalar_out[i] - dispatch_out[i]).abs() <= 1e-3,
                    "M={m} idx={i}: scalar={} dispatch={}",
                    scalar_out[i],
                    dispatch_out[i]
                );
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_matches_scalar_when_available() {
        if !(std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")) {
            return;
        }
        let m_w = 8;
        let k = 256;
        let group_size = 32;
        let weights = pseudo_random(0x1234_5678, m_w * k);
        let (packed, scales) = pack_int4_symmetric_per_group(&weights, m_w, k, group_size);
        let activation = pseudo_random(0x9ABC_DEF0, k);
        let mut scalar_out = vec![0.0_f32; m_w];
        packed_int4_gemv_scalar(
            &packed,
            &scales,
            &activation,
            &mut scalar_out,
            m_w,
            k,
            group_size,
        );
        let mut avx_out = vec![0.0_f32; m_w];
        unsafe {
            packed_int4_gemv_avx2(
                &packed,
                &scales,
                &activation,
                &mut avx_out,
                m_w,
                k,
                group_size,
            )
        };
        for row in 0..m_w {
            assert!(
                (scalar_out[row] - avx_out[row]).abs() <= 1e-4,
                "row {row}: scalar={} avx2={}",
                scalar_out[row],
                avx_out[row]
            );
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_matches_scalar_when_available() {
        if !std::arch::is_aarch64_feature_detected!("neon") {
            return;
        }
        let m_w = 8;
        let k = 256;
        let group_size = 32;
        let weights = pseudo_random(0x1234_5678, m_w * k);
        let (packed, scales) = pack_int4_symmetric_per_group(&weights, m_w, k, group_size);
        let activation = pseudo_random(0x9ABC_DEF0, k);
        let mut scalar_out = vec![0.0_f32; m_w];
        packed_int4_gemv_scalar(
            &packed,
            &scales,
            &activation,
            &mut scalar_out,
            m_w,
            k,
            group_size,
        );
        let mut neon_out = vec![0.0_f32; m_w];
        unsafe {
            packed_int4_gemv_neon(
                &packed,
                &scales,
                &activation,
                &mut neon_out,
                m_w,
                k,
                group_size,
            )
        };
        for row in 0..m_w {
            assert!(
                (scalar_out[row] - neon_out[row]).abs() <= 1e-4,
                "row {row}: scalar={} neon={}",
                scalar_out[row],
                neon_out[row]
            );
        }
    }
}
