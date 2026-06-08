//! NHWC pointwise (1×1) conv kernels: the nx16 and 16x16 row tiles with
//! per-arch implementations (AVX-512/AVX2/NEON/scalar) + the MR16/kcblock
//! variants and their dispatchers.

use super::*;

pub(super) fn pointwise_16x16_direct_disabled() -> bool {
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| std::env::var_os("YSCV_NO_POINTWISE_16X16_DIRECT").is_some())
}

pub(super) fn pointwise_nx16_direct_disabled() -> bool {
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| {
        if std::env::var_os("YSCV_NO_POINTWISE_NX16_DIRECT").is_some() {
            return true;
        }
        // The direct (unpacked) pointwise kernel trades weight-pack cost for a
        // streaming dot product — a win on out-of-order x86 with large caches,
        // a loss on in-order aarch64 cores with small caches (Cortex-A53 tracker
        // measured ~+190 ms/inf vs the packed blocked-GEMM path). Default it off
        // on aarch64; opt back in with YSCV_POINTWISE_NX16_DIRECT_ON for A/B.
        #[cfg(target_arch = "aarch64")]
        {
            std::env::var_os("YSCV_POINTWISE_NX16_DIRECT_ON").is_none()
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            false
        }
    })
}

/// Whether the direct nx16 pointwise kernel should run at the current thread
/// count for an `m`-row GEMM. Single-thread always uses it (the streaming dot
/// beats packed blocked GEMM on x86). Multi-thread only when each thread gets
/// ≥64 output rows (`m / threads ≥ 64`) and `m ≥ 256` so the kernel's row-chunk
/// parallelism balances; below that the packed blocked GEMM scales better
/// (measured: the tower 256×256×256 PW steps win with parallel nx16 at 4T but
/// lose at 6T where 256/6 ≈ 43 rows/thread is too few). Kill-switch
/// `YSCV_NX16_MT_OFF`.
pub(super) fn nx16_threads_ok(m: usize) -> bool {
    let threads = rayon::current_num_threads();
    if threads <= 1 {
        return true;
    }
    static OFF: OnceLock<bool> = OnceLock::new();
    let off = *OFF.get_or_init(|| std::env::var_os("YSCV_NX16_MT_OFF").is_some());
    !off && m >= 256 && m / threads >= 64
}

#[allow(unsafe_code)]
pub(super) fn pointwise_16x16_direct(
    input: &[f32],
    kernel: &[f32],
    bias: Option<&[f32]>,
    residual: Option<&[f32]>,
    output: &mut [f32],
    m: usize,
    activation: Activation,
) {
    debug_assert!(input.len() >= m * 16);
    debug_assert!(kernel.len() >= 16 * 16);
    debug_assert!(output.len() >= m * 16);
    debug_assert!(residual.is_none_or(|r| r.len() >= m * 16));

    let rows_per_chunk = if rayon::current_num_threads() > 1 && m >= 4096 {
        1024
    } else {
        m.max(1)
    };
    let chunk_len = rows_per_chunk * 16;

    let compute_chunk = |chunk_idx: usize, out_chunk: &mut [f32]| {
        let row_start = chunk_idx * rows_per_chunk;
        let rows = out_chunk.len() / 16;
        let input_chunk = &input[row_start * 16..(row_start + rows) * 16];
        let residual_chunk = residual.map(|r| &r[row_start * 16..(row_start + rows) * 16]);
        pointwise_16x16_direct_rows(
            input_chunk,
            kernel,
            bias,
            residual_chunk,
            out_chunk,
            rows,
            activation,
        );
    };

    if rows_per_chunk < m {
        super::super::super::scope_ctx::par_chunks_mut_dispatch(output, chunk_len, compute_chunk);
    } else {
        compute_chunk(0, output);
    }
}

#[inline]
fn pointwise_16x16_direct_rows(
    input: &[f32],
    kernel: &[f32],
    bias: Option<&[f32]>,
    residual: Option<&[f32]>,
    output: &mut [f32],
    rows: usize,
    activation: Activation,
) {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if crate::host_cpu().features.avx512f {
            #[allow(unsafe_code)]
            unsafe {
                pointwise_16x16_direct_rows_avx512(
                    input, kernel, bias, residual, output, rows, activation,
                );
            }
            return;
        }
        if crate::host_cpu().features.avx && crate::host_cpu().features.fma {
            #[allow(unsafe_code)]
            unsafe {
                pointwise_16x16_direct_rows_avx2(
                    input, kernel, bias, residual, output, rows, activation,
                );
            }
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    if crate::host_cpu().features.neon {
        #[allow(unsafe_code)]
        unsafe {
            pointwise_16x16_direct_rows_neon(
                input, kernel, bias, residual, output, rows, activation,
            );
        }
        return;
    }
    pointwise_16x16_direct_rows_scalar(input, kernel, bias, residual, output, rows, activation);
}

#[allow(clippy::too_many_arguments)]
fn pointwise_16x16_direct_rows_scalar(
    input: &[f32],
    kernel: &[f32],
    bias: Option<&[f32]>,
    residual: Option<&[f32]>,
    output: &mut [f32],
    rows: usize,
    activation: Activation,
) {
    for row in 0..rows {
        for oc in 0..16 {
            let mut acc = bias.map_or(0.0, |b| b[oc]);
            if let Some(skip) = residual {
                acc += skip[row * 16 + oc];
            }
            for ic in 0..16 {
                acc += input[row * 16 + ic] * kernel[ic * 16 + oc];
            }
            output[row * 16 + oc] = apply_conv_activation_scalar(acc, activation);
        }
    }
}

#[allow(unsafe_code, clippy::too_many_arguments)]
pub(super) fn pointwise_nx16_direct(
    input: &[f32],
    kernel: &[f32],
    bias: Option<&[f32]>,
    residual: Option<&[f32]>,
    output: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    activation: Activation,
) {
    debug_assert!(n.is_multiple_of(16));
    debug_assert!(input.len() >= m * k);
    debug_assert!(kernel.len() >= k * n);
    debug_assert!(residual.is_none_or(|r| r.len() >= m * n));
    debug_assert!(output.len() >= m * n);

    let rows_per_chunk = if rayon::current_num_threads() > 1 && m >= 256 {
        64
    } else {
        m.max(1)
    };
    let chunk_len = rows_per_chunk * n;

    let compute_chunk = |chunk_idx: usize, out_chunk: &mut [f32]| {
        let row_start = chunk_idx * rows_per_chunk;
        let rows = out_chunk.len() / n;
        let input_chunk = &input[row_start * k..(row_start + rows) * k];
        let residual_chunk = residual.map(|r| &r[row_start * n..(row_start + rows) * n]);
        pointwise_nx16_direct_rows(
            input_chunk,
            kernel,
            bias,
            residual_chunk,
            out_chunk,
            rows,
            k,
            n,
            activation,
        );
    };

    if rows_per_chunk < m {
        super::super::super::scope_ctx::par_chunks_mut_dispatch(output, chunk_len, compute_chunk);
    } else {
        compute_chunk(0, output);
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
fn reduce_gemm_enabled() -> bool {
    // Default OFF: the reduce/standalone-PW shapes are low-N (c_out 24-112), and
    // routing them through the blocked GEMM (now the 8×8) loses to the
    // weight-stationary nx16 direct kernel — disabling it is −2 ms/1T, −4.5 ms/2T.
    // `YSCV_REDUCE_GEMM_ON` re-enables for A/B.
    static C: OnceLock<bool> = OnceLock::new();
    *C.get_or_init(|| std::env::var_os("YSCV_REDUCE_GEMM_ON").is_some())
}

/// Default ON: the NEON reduce broadcasts each activation (`vdupq_n` = `ld1r`,
/// a NEON-pipe op that competes with the FMAs); the `fmla`-by-lane path loads
/// 4 activations per `vld1q` and reads lanes directly, freeing the NEON pipe.
/// `YSCV_REDUCE_BYLANE_OFF` reverts to the broadcast path for A/B.
#[cfg(target_arch = "aarch64")]
#[inline]
fn reduce_bylane() -> bool {
    static C: OnceLock<bool> = OnceLock::new();
    *C.get_or_init(|| std::env::var_os("YSCV_REDUCE_BYLANE_OFF").is_none())
}

/// Per-row 1×1 pointwise dispatcher (AVX-512 → AVX2 → NEON → scalar). Used
/// internally by Conv 1×1 and by the full-block streaming kernel
/// [`super::fused_pw_dw_3x3::fused_pw_expand_dw_pw_reduce_3x3`] which calls
/// it once per DW output row with an L1-hot input scratch.
///
/// Contracts: `input` is `[rows, k]` row-major, `kernel` is `[k, n]` with `n`
/// a multiple of 16 (else dispatches to scalar), `output` is `[rows, n]`.
#[allow(clippy::too_many_arguments)]
#[inline]
pub(crate) fn pointwise_nx16_direct_rows(
    input: &[f32],
    kernel: &[f32],
    bias: Option<&[f32]>,
    residual: Option<&[f32]>,
    output: &mut [f32],
    rows: usize,
    k: usize,
    n: usize,
    activation: Activation,
) {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if crate::host_cpu().features.avx512f {
            // K-cache-blocked variant for K-heavy shapes. Microbench-faster
            // but tracker-flat, so opt-in via `YSCV_KCBLOCK=1`, default OFF.
            if k > 512 && kcblock_enabled() {
                #[allow(unsafe_code)]
                unsafe {
                    pointwise_nx16_direct_rows_avx512_kcblock(
                        input, kernel, bias, residual, output, rows, k, n, activation,
                    );
                }
                return;
            }
            // 16-row MR tile to hide Zen 4's double-pumped FMA latency.
            // Tracker-flat, so opt-in via `YSCV_MR16=1`, default OFF.
            if k >= 256 && rows >= 16 && mr16_enabled() {
                #[allow(unsafe_code)]
                unsafe {
                    pointwise_nx16_direct_rows_avx512_mr16(
                        input, kernel, bias, residual, output, rows, k, n, activation,
                    );
                }
                return;
            }
            #[allow(unsafe_code)]
            unsafe {
                pointwise_nx16_direct_rows_avx512(
                    input, kernel, bias, residual, output, rows, k, n, activation,
                );
            }
            return;
        }
        if crate::host_cpu().features.avx && crate::host_cpu().features.fma {
            #[allow(unsafe_code)]
            unsafe {
                pointwise_nx16_direct_rows_avx2(
                    input, kernel, bias, residual, output, rows, k, n, activation,
                );
            }
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    if crate::host_cpu().features.neon {
        // The PW-reduce is a [rows, k]×[k, n] GEMM (weight already [c_exp, c_out]
        // = B); the blocked 8×12 asm beats the broadcast kernel. bias/residual/
        // activation fold into the GEMM epilogue (residual is n-strided here, so
        // it matches the output). ≤2-thread gate mirrors the PW-expand path.
        if reduce_gemm_enabled() && rows > 0 && rayon::current_num_threads() <= 2 {
            let ep = super::super::matmul::GemmEpilogue {
                bias: bias.map(|b| b.as_ptr()),
                activation,
                residual: residual.map(|r| r.as_ptr()),
            };
            super::super::matmul::matmul_2d_slices_blocked_fused(
                input, rows, k, kernel, n, output, ep,
            );
            return;
        }
        #[allow(unsafe_code)]
        unsafe {
            pointwise_nx16_direct_rows_neon(
                input, kernel, bias, residual, output, rows, k, n, activation,
            );
        }
        return;
    }
    pointwise_nx16_direct_rows_scalar(
        input, kernel, bias, residual, output, rows, k, n, activation,
    );
}

#[allow(clippy::too_many_arguments)]
fn pointwise_nx16_direct_rows_scalar(
    input: &[f32],
    kernel: &[f32],
    bias: Option<&[f32]>,
    residual: Option<&[f32]>,
    output: &mut [f32],
    rows: usize,
    k: usize,
    n: usize,
    activation: Activation,
) {
    for row in 0..rows {
        for oc_base in (0..n).step_by(16) {
            for lane in 0..16 {
                let oc = oc_base + lane;
                let mut acc =
                    bias.map_or(0.0, |b| b[oc]) + residual.map_or(0.0, |r| r[row * n + oc]);
                for ic in 0..k {
                    acc += input[row * k + ic] * kernel[ic * n + oc];
                }
                output[row * n + oc] = apply_conv_activation_scalar(acc, activation);
            }
        }
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn, clippy::too_many_arguments)]
unsafe fn pointwise_16x16_direct_rows_avx512(
    input: &[f32],
    kernel: &[f32],
    bias: Option<&[f32]>,
    residual: Option<&[f32]>,
    output: &mut [f32],
    rows: usize,
    activation: Activation,
) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let zero = _mm512_setzero_ps();
    let bias_v = bias.map_or(zero, |b| _mm512_loadu_ps(b.as_ptr()));
    let do_relu = matches!(activation, Activation::Relu);
    let do_silu = matches!(activation, Activation::Silu);
    let mut row = 0usize;

    while row + 4 <= rows {
        let mut a0 = bias_v;
        let mut a1 = bias_v;
        let mut a2 = bias_v;
        let mut a3 = bias_v;
        if let Some(skip) = residual {
            let rp = skip.as_ptr().add(row * 16);
            a0 = _mm512_add_ps(a0, _mm512_loadu_ps(rp));
            a1 = _mm512_add_ps(a1, _mm512_loadu_ps(rp.add(16)));
            a2 = _mm512_add_ps(a2, _mm512_loadu_ps(rp.add(32)));
            a3 = _mm512_add_ps(a3, _mm512_loadu_ps(rp.add(48)));
        }
        let ip = input.as_ptr().add(row * 16);
        for ic in 0..16 {
            let w = _mm512_loadu_ps(kernel.as_ptr().add(ic * 16));
            a0 = _mm512_fmadd_ps(_mm512_set1_ps(*ip.add(ic)), w, a0);
            a1 = _mm512_fmadd_ps(_mm512_set1_ps(*ip.add(16 + ic)), w, a1);
            a2 = _mm512_fmadd_ps(_mm512_set1_ps(*ip.add(32 + ic)), w, a2);
            a3 = _mm512_fmadd_ps(_mm512_set1_ps(*ip.add(48 + ic)), w, a3);
        }
        if do_relu {
            a0 = _mm512_max_ps(a0, zero);
            a1 = _mm512_max_ps(a1, zero);
            a2 = _mm512_max_ps(a2, zero);
            a3 = _mm512_max_ps(a3, zero);
        }
        let op = output.as_mut_ptr().add(row * 16);
        _mm512_storeu_ps(op, a0);
        _mm512_storeu_ps(op.add(16), a1);
        _mm512_storeu_ps(op.add(32), a2);
        _mm512_storeu_ps(op.add(48), a3);
        if do_silu {
            for v in &mut output[row * 16..(row + 4) * 16] {
                *v = apply_conv_activation_scalar(*v, Activation::Silu);
            }
        }
        row += 4;
    }

    while row < rows {
        let mut acc = bias_v;
        if let Some(skip) = residual {
            acc = _mm512_add_ps(acc, _mm512_loadu_ps(skip.as_ptr().add(row * 16)));
        }
        let ip = input.as_ptr().add(row * 16);
        for ic in 0..16 {
            let w = _mm512_loadu_ps(kernel.as_ptr().add(ic * 16));
            acc = _mm512_fmadd_ps(_mm512_set1_ps(*ip.add(ic)), w, acc);
        }
        if do_relu {
            acc = _mm512_max_ps(acc, zero);
        }
        _mm512_storeu_ps(output.as_mut_ptr().add(row * 16), acc);
        if do_silu {
            for v in &mut output[row * 16..(row + 1) * 16] {
                *v = apply_conv_activation_scalar(*v, Activation::Silu);
            }
        }
        row += 1;
    }
}

/// 16-row tile variant of [`pointwise_nx16_direct_rows_avx512`]. The
/// 8-row kernel's 8 accumulator chains aren't enough to hide Zen 4's
/// effective ZMM-FMA latency (~8 cycles, double-pumped from 256-bit
/// internal pipes), leaving the FMA pipe at half-throughput. With 16
/// chains the pipe stays saturated: 16 FMAs per ic / 2-per-cyc-peak = 8
/// cycles, matching the FMA-latency-limited throughput of 16/8 = 2 per
/// cycle.
///
/// Register budget: 16 acc + 1 weight + 1 bias + 1 zero = 19 ZMMs,
/// well within the 32-register AVX-512 file. The 16 broadcasts per ic
/// fold into `vfmadd231ps mem{1to16}`, so no broadcast registers needed.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn, clippy::too_many_arguments)]
unsafe fn pointwise_nx16_direct_rows_avx512_mr16(
    input: &[f32],
    kernel: &[f32],
    bias: Option<&[f32]>,
    residual: Option<&[f32]>,
    output: &mut [f32],
    rows: usize,
    k: usize,
    n: usize,
    activation: Activation,
) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let zero = _mm512_setzero_ps();
    let do_relu = matches!(activation, Activation::Relu);
    let do_silu = matches!(activation, Activation::Silu);

    for oc_base in (0..n).step_by(16) {
        let bias_v = bias.map_or(zero, |b| _mm512_loadu_ps(b.as_ptr().add(oc_base)));
        let mut row = 0usize;
        // 16-row tile: 16 independent FMA chains saturate the FMA pipe at
        // Zen 4's ~8-cycle double-pumped ZMM-FMA latency.
        while row + 16 <= rows {
            let mut a0 = bias_v;
            let mut a1 = bias_v;
            let mut a2 = bias_v;
            let mut a3 = bias_v;
            let mut a4 = bias_v;
            let mut a5 = bias_v;
            let mut a6 = bias_v;
            let mut a7 = bias_v;
            let mut a8 = bias_v;
            let mut a9 = bias_v;
            let mut a10 = bias_v;
            let mut a11 = bias_v;
            let mut a12 = bias_v;
            let mut a13 = bias_v;
            let mut a14 = bias_v;
            let mut a15 = bias_v;
            if let Some(res) = residual {
                let rp = res.as_ptr().add(row * n + oc_base);
                a0 = _mm512_add_ps(a0, _mm512_loadu_ps(rp));
                a1 = _mm512_add_ps(a1, _mm512_loadu_ps(rp.add(n)));
                a2 = _mm512_add_ps(a2, _mm512_loadu_ps(rp.add(2 * n)));
                a3 = _mm512_add_ps(a3, _mm512_loadu_ps(rp.add(3 * n)));
                a4 = _mm512_add_ps(a4, _mm512_loadu_ps(rp.add(4 * n)));
                a5 = _mm512_add_ps(a5, _mm512_loadu_ps(rp.add(5 * n)));
                a6 = _mm512_add_ps(a6, _mm512_loadu_ps(rp.add(6 * n)));
                a7 = _mm512_add_ps(a7, _mm512_loadu_ps(rp.add(7 * n)));
                a8 = _mm512_add_ps(a8, _mm512_loadu_ps(rp.add(8 * n)));
                a9 = _mm512_add_ps(a9, _mm512_loadu_ps(rp.add(9 * n)));
                a10 = _mm512_add_ps(a10, _mm512_loadu_ps(rp.add(10 * n)));
                a11 = _mm512_add_ps(a11, _mm512_loadu_ps(rp.add(11 * n)));
                a12 = _mm512_add_ps(a12, _mm512_loadu_ps(rp.add(12 * n)));
                a13 = _mm512_add_ps(a13, _mm512_loadu_ps(rp.add(13 * n)));
                a14 = _mm512_add_ps(a14, _mm512_loadu_ps(rp.add(14 * n)));
                a15 = _mm512_add_ps(a15, _mm512_loadu_ps(rp.add(15 * n)));
            }
            let ip = input.as_ptr().add(row * k);
            for ic in 0..k {
                let w = _mm512_loadu_ps(kernel.as_ptr().add(ic * n + oc_base));
                a0 = _mm512_fmadd_ps(_mm512_set1_ps(*ip.add(ic)), w, a0);
                a1 = _mm512_fmadd_ps(_mm512_set1_ps(*ip.add(k + ic)), w, a1);
                a2 = _mm512_fmadd_ps(_mm512_set1_ps(*ip.add(2 * k + ic)), w, a2);
                a3 = _mm512_fmadd_ps(_mm512_set1_ps(*ip.add(3 * k + ic)), w, a3);
                a4 = _mm512_fmadd_ps(_mm512_set1_ps(*ip.add(4 * k + ic)), w, a4);
                a5 = _mm512_fmadd_ps(_mm512_set1_ps(*ip.add(5 * k + ic)), w, a5);
                a6 = _mm512_fmadd_ps(_mm512_set1_ps(*ip.add(6 * k + ic)), w, a6);
                a7 = _mm512_fmadd_ps(_mm512_set1_ps(*ip.add(7 * k + ic)), w, a7);
                a8 = _mm512_fmadd_ps(_mm512_set1_ps(*ip.add(8 * k + ic)), w, a8);
                a9 = _mm512_fmadd_ps(_mm512_set1_ps(*ip.add(9 * k + ic)), w, a9);
                a10 = _mm512_fmadd_ps(_mm512_set1_ps(*ip.add(10 * k + ic)), w, a10);
                a11 = _mm512_fmadd_ps(_mm512_set1_ps(*ip.add(11 * k + ic)), w, a11);
                a12 = _mm512_fmadd_ps(_mm512_set1_ps(*ip.add(12 * k + ic)), w, a12);
                a13 = _mm512_fmadd_ps(_mm512_set1_ps(*ip.add(13 * k + ic)), w, a13);
                a14 = _mm512_fmadd_ps(_mm512_set1_ps(*ip.add(14 * k + ic)), w, a14);
                a15 = _mm512_fmadd_ps(_mm512_set1_ps(*ip.add(15 * k + ic)), w, a15);
            }
            if do_relu {
                a0 = _mm512_max_ps(a0, zero);
                a1 = _mm512_max_ps(a1, zero);
                a2 = _mm512_max_ps(a2, zero);
                a3 = _mm512_max_ps(a3, zero);
                a4 = _mm512_max_ps(a4, zero);
                a5 = _mm512_max_ps(a5, zero);
                a6 = _mm512_max_ps(a6, zero);
                a7 = _mm512_max_ps(a7, zero);
                a8 = _mm512_max_ps(a8, zero);
                a9 = _mm512_max_ps(a9, zero);
                a10 = _mm512_max_ps(a10, zero);
                a11 = _mm512_max_ps(a11, zero);
                a12 = _mm512_max_ps(a12, zero);
                a13 = _mm512_max_ps(a13, zero);
                a14 = _mm512_max_ps(a14, zero);
                a15 = _mm512_max_ps(a15, zero);
            }
            let op = output.as_mut_ptr().add(row * n + oc_base);
            _mm512_storeu_ps(op, a0);
            _mm512_storeu_ps(op.add(n), a1);
            _mm512_storeu_ps(op.add(2 * n), a2);
            _mm512_storeu_ps(op.add(3 * n), a3);
            _mm512_storeu_ps(op.add(4 * n), a4);
            _mm512_storeu_ps(op.add(5 * n), a5);
            _mm512_storeu_ps(op.add(6 * n), a6);
            _mm512_storeu_ps(op.add(7 * n), a7);
            _mm512_storeu_ps(op.add(8 * n), a8);
            _mm512_storeu_ps(op.add(9 * n), a9);
            _mm512_storeu_ps(op.add(10 * n), a10);
            _mm512_storeu_ps(op.add(11 * n), a11);
            _mm512_storeu_ps(op.add(12 * n), a12);
            _mm512_storeu_ps(op.add(13 * n), a13);
            _mm512_storeu_ps(op.add(14 * n), a14);
            _mm512_storeu_ps(op.add(15 * n), a15);
            if do_silu {
                for r in 0..16 {
                    for v in &mut output[(row + r) * n + oc_base..(row + r) * n + oc_base + 16] {
                        *v = apply_conv_activation_scalar(*v, Activation::Silu);
                    }
                }
            }
            row += 16;
        }
        // Fall through to the 8-row tile for the row tail.
        while row + 8 <= rows {
            let mut a0 = bias_v;
            let mut a1 = bias_v;
            let mut a2 = bias_v;
            let mut a3 = bias_v;
            let mut a4 = bias_v;
            let mut a5 = bias_v;
            let mut a6 = bias_v;
            let mut a7 = bias_v;
            if let Some(res) = residual {
                let rp = res.as_ptr().add(row * n + oc_base);
                a0 = _mm512_add_ps(a0, _mm512_loadu_ps(rp));
                a1 = _mm512_add_ps(a1, _mm512_loadu_ps(rp.add(n)));
                a2 = _mm512_add_ps(a2, _mm512_loadu_ps(rp.add(2 * n)));
                a3 = _mm512_add_ps(a3, _mm512_loadu_ps(rp.add(3 * n)));
                a4 = _mm512_add_ps(a4, _mm512_loadu_ps(rp.add(4 * n)));
                a5 = _mm512_add_ps(a5, _mm512_loadu_ps(rp.add(5 * n)));
                a6 = _mm512_add_ps(a6, _mm512_loadu_ps(rp.add(6 * n)));
                a7 = _mm512_add_ps(a7, _mm512_loadu_ps(rp.add(7 * n)));
            }
            let ip = input.as_ptr().add(row * k);
            // Software-prefetch the weight cacheline 8 K-iters ahead: the
            // weight stride (`n*4` bytes, 448-1280 B for tracker shapes) is too
            // large for the HW prefetcher's stride detector, so a manual
            // `_mm_prefetch(_MM_HINT_T0)` hides the load latency behind the FMA
            // pipe. Kill switch `YSCV_PW_PREFETCH_OFF=1`. (Lookahead distance
            // is flat past 8 — the kernel is FMA-bound, not weight-load-bound.)
            const PREFETCH_AHEAD: usize = 8;
            let pf_enabled = pw_prefetch_enabled();
            for ic in 0..k {
                if pf_enabled && ic + PREFETCH_AHEAD < k {
                    let pf_ptr = kernel.as_ptr().add((ic + PREFETCH_AHEAD) * n + oc_base);
                    _mm_prefetch(pf_ptr as *const i8, _MM_HINT_T0);
                }
                let w = _mm512_loadu_ps(kernel.as_ptr().add(ic * n + oc_base));
                a0 = _mm512_fmadd_ps(_mm512_set1_ps(*ip.add(ic)), w, a0);
                a1 = _mm512_fmadd_ps(_mm512_set1_ps(*ip.add(k + ic)), w, a1);
                a2 = _mm512_fmadd_ps(_mm512_set1_ps(*ip.add(2 * k + ic)), w, a2);
                a3 = _mm512_fmadd_ps(_mm512_set1_ps(*ip.add(3 * k + ic)), w, a3);
                a4 = _mm512_fmadd_ps(_mm512_set1_ps(*ip.add(4 * k + ic)), w, a4);
                a5 = _mm512_fmadd_ps(_mm512_set1_ps(*ip.add(5 * k + ic)), w, a5);
                a6 = _mm512_fmadd_ps(_mm512_set1_ps(*ip.add(6 * k + ic)), w, a6);
                a7 = _mm512_fmadd_ps(_mm512_set1_ps(*ip.add(7 * k + ic)), w, a7);
            }
            if do_relu {
                a0 = _mm512_max_ps(a0, zero);
                a1 = _mm512_max_ps(a1, zero);
                a2 = _mm512_max_ps(a2, zero);
                a3 = _mm512_max_ps(a3, zero);
                a4 = _mm512_max_ps(a4, zero);
                a5 = _mm512_max_ps(a5, zero);
                a6 = _mm512_max_ps(a6, zero);
                a7 = _mm512_max_ps(a7, zero);
            }
            let op = output.as_mut_ptr().add(row * n + oc_base);
            _mm512_storeu_ps(op, a0);
            _mm512_storeu_ps(op.add(n), a1);
            _mm512_storeu_ps(op.add(2 * n), a2);
            _mm512_storeu_ps(op.add(3 * n), a3);
            _mm512_storeu_ps(op.add(4 * n), a4);
            _mm512_storeu_ps(op.add(5 * n), a5);
            _mm512_storeu_ps(op.add(6 * n), a6);
            _mm512_storeu_ps(op.add(7 * n), a7);
            if do_silu {
                for r in 0..8 {
                    for v in &mut output[(row + r) * n + oc_base..(row + r) * n + oc_base + 16] {
                        *v = apply_conv_activation_scalar(*v, Activation::Silu);
                    }
                }
            }
            row += 8;
        }
        while row + 4 <= rows {
            let mut a0 = bias_v;
            let mut a1 = bias_v;
            let mut a2 = bias_v;
            let mut a3 = bias_v;
            if let Some(res) = residual {
                let rp = res.as_ptr().add(row * n + oc_base);
                a0 = _mm512_add_ps(a0, _mm512_loadu_ps(rp));
                a1 = _mm512_add_ps(a1, _mm512_loadu_ps(rp.add(n)));
                a2 = _mm512_add_ps(a2, _mm512_loadu_ps(rp.add(2 * n)));
                a3 = _mm512_add_ps(a3, _mm512_loadu_ps(rp.add(3 * n)));
            }
            let ip = input.as_ptr().add(row * k);
            for ic in 0..k {
                let w = _mm512_loadu_ps(kernel.as_ptr().add(ic * n + oc_base));
                a0 = _mm512_fmadd_ps(_mm512_set1_ps(*ip.add(ic)), w, a0);
                a1 = _mm512_fmadd_ps(_mm512_set1_ps(*ip.add(k + ic)), w, a1);
                a2 = _mm512_fmadd_ps(_mm512_set1_ps(*ip.add(2 * k + ic)), w, a2);
                a3 = _mm512_fmadd_ps(_mm512_set1_ps(*ip.add(3 * k + ic)), w, a3);
            }
            if do_relu {
                a0 = _mm512_max_ps(a0, zero);
                a1 = _mm512_max_ps(a1, zero);
                a2 = _mm512_max_ps(a2, zero);
                a3 = _mm512_max_ps(a3, zero);
            }
            let op = output.as_mut_ptr().add(row * n + oc_base);
            _mm512_storeu_ps(op, a0);
            _mm512_storeu_ps(op.add(n), a1);
            _mm512_storeu_ps(op.add(2 * n), a2);
            _mm512_storeu_ps(op.add(3 * n), a3);
            if do_silu {
                for r in 0..4 {
                    for v in &mut output[(row + r) * n + oc_base..(row + r) * n + oc_base + 16] {
                        *v = apply_conv_activation_scalar(*v, Activation::Silu);
                    }
                }
            }
            row += 4;
        }
        while row < rows {
            let mut acc = bias_v;
            if let Some(res) = residual {
                acc = _mm512_add_ps(acc, _mm512_loadu_ps(res.as_ptr().add(row * n + oc_base)));
            }
            let ip = input.as_ptr().add(row * k);
            for ic in 0..k {
                let w = _mm512_loadu_ps(kernel.as_ptr().add(ic * n + oc_base));
                acc = _mm512_fmadd_ps(_mm512_set1_ps(*ip.add(ic)), w, acc);
            }
            if do_relu {
                acc = _mm512_max_ps(acc, zero);
            }
            _mm512_storeu_ps(output.as_mut_ptr().add(row * n + oc_base), acc);
            if do_silu {
                for v in &mut output[row * n + oc_base..row * n + oc_base + 16] {
                    *v = apply_conv_activation_scalar(*v, Activation::Silu);
                }
            }
            row += 1;
        }
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn, clippy::too_many_arguments)]
unsafe fn pointwise_nx16_direct_rows_avx512(
    input: &[f32],
    kernel: &[f32],
    bias: Option<&[f32]>,
    residual: Option<&[f32]>,
    output: &mut [f32],
    rows: usize,
    k: usize,
    n: usize,
    activation: Activation,
) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let zero = _mm512_setzero_ps();
    let do_relu = matches!(activation, Activation::Relu);
    let do_silu = matches!(activation, Activation::Silu);

    for oc_base in (0..n).step_by(16) {
        let bias_v = bias.map_or(zero, |b| _mm512_loadu_ps(b.as_ptr().add(oc_base)));
        let mut row = 0usize;
        // 8-row unrolling: 8 independent accumulators saturate both FMA ports
        // (4-cycle FMA latency × 2 ports = 8 in-flight chains needed for peak throughput).
        while row + 8 <= rows {
            let mut a0 = bias_v;
            let mut a1 = bias_v;
            let mut a2 = bias_v;
            let mut a3 = bias_v;
            let mut a4 = bias_v;
            let mut a5 = bias_v;
            let mut a6 = bias_v;
            let mut a7 = bias_v;
            if let Some(res) = residual {
                let rp = res.as_ptr().add(row * n + oc_base);
                a0 = _mm512_add_ps(a0, _mm512_loadu_ps(rp));
                a1 = _mm512_add_ps(a1, _mm512_loadu_ps(rp.add(n)));
                a2 = _mm512_add_ps(a2, _mm512_loadu_ps(rp.add(2 * n)));
                a3 = _mm512_add_ps(a3, _mm512_loadu_ps(rp.add(3 * n)));
                a4 = _mm512_add_ps(a4, _mm512_loadu_ps(rp.add(4 * n)));
                a5 = _mm512_add_ps(a5, _mm512_loadu_ps(rp.add(5 * n)));
                a6 = _mm512_add_ps(a6, _mm512_loadu_ps(rp.add(6 * n)));
                a7 = _mm512_add_ps(a7, _mm512_loadu_ps(rp.add(7 * n)));
            }
            let ip = input.as_ptr().add(row * k);
            // Software-prefetch the weight cacheline 8 K-iters ahead: the
            // weight stride (`n*4` bytes, 448-1280 B for tracker shapes) is too
            // large for the HW prefetcher's stride detector, so a manual
            // `_mm_prefetch(_MM_HINT_T0)` hides the load latency behind the FMA
            // pipe. Kill switch `YSCV_PW_PREFETCH_OFF=1`. (Lookahead distance
            // is flat past 8 — the kernel is FMA-bound, not weight-load-bound.)
            const PREFETCH_AHEAD: usize = 8;
            let pf_enabled = pw_prefetch_enabled();
            for ic in 0..k {
                if pf_enabled && ic + PREFETCH_AHEAD < k {
                    let pf_ptr = kernel.as_ptr().add((ic + PREFETCH_AHEAD) * n + oc_base);
                    _mm_prefetch(pf_ptr as *const i8, _MM_HINT_T0);
                }
                let w = _mm512_loadu_ps(kernel.as_ptr().add(ic * n + oc_base));
                a0 = _mm512_fmadd_ps(_mm512_set1_ps(*ip.add(ic)), w, a0);
                a1 = _mm512_fmadd_ps(_mm512_set1_ps(*ip.add(k + ic)), w, a1);
                a2 = _mm512_fmadd_ps(_mm512_set1_ps(*ip.add(2 * k + ic)), w, a2);
                a3 = _mm512_fmadd_ps(_mm512_set1_ps(*ip.add(3 * k + ic)), w, a3);
                a4 = _mm512_fmadd_ps(_mm512_set1_ps(*ip.add(4 * k + ic)), w, a4);
                a5 = _mm512_fmadd_ps(_mm512_set1_ps(*ip.add(5 * k + ic)), w, a5);
                a6 = _mm512_fmadd_ps(_mm512_set1_ps(*ip.add(6 * k + ic)), w, a6);
                a7 = _mm512_fmadd_ps(_mm512_set1_ps(*ip.add(7 * k + ic)), w, a7);
            }
            if do_relu {
                a0 = _mm512_max_ps(a0, zero);
                a1 = _mm512_max_ps(a1, zero);
                a2 = _mm512_max_ps(a2, zero);
                a3 = _mm512_max_ps(a3, zero);
                a4 = _mm512_max_ps(a4, zero);
                a5 = _mm512_max_ps(a5, zero);
                a6 = _mm512_max_ps(a6, zero);
                a7 = _mm512_max_ps(a7, zero);
            }
            let op = output.as_mut_ptr().add(row * n + oc_base);
            _mm512_storeu_ps(op, a0);
            _mm512_storeu_ps(op.add(n), a1);
            _mm512_storeu_ps(op.add(2 * n), a2);
            _mm512_storeu_ps(op.add(3 * n), a3);
            _mm512_storeu_ps(op.add(4 * n), a4);
            _mm512_storeu_ps(op.add(5 * n), a5);
            _mm512_storeu_ps(op.add(6 * n), a6);
            _mm512_storeu_ps(op.add(7 * n), a7);
            if do_silu {
                for r in 0..8 {
                    for v in &mut output[(row + r) * n + oc_base..(row + r) * n + oc_base + 16] {
                        *v = apply_conv_activation_scalar(*v, Activation::Silu);
                    }
                }
            }
            row += 8;
        }
        while row + 4 <= rows {
            let mut a0 = bias_v;
            let mut a1 = bias_v;
            let mut a2 = bias_v;
            let mut a3 = bias_v;
            if let Some(res) = residual {
                let rp = res.as_ptr().add(row * n + oc_base);
                a0 = _mm512_add_ps(a0, _mm512_loadu_ps(rp));
                a1 = _mm512_add_ps(a1, _mm512_loadu_ps(rp.add(n)));
                a2 = _mm512_add_ps(a2, _mm512_loadu_ps(rp.add(2 * n)));
                a3 = _mm512_add_ps(a3, _mm512_loadu_ps(rp.add(3 * n)));
            }
            let ip = input.as_ptr().add(row * k);
            for ic in 0..k {
                let w = _mm512_loadu_ps(kernel.as_ptr().add(ic * n + oc_base));
                a0 = _mm512_fmadd_ps(_mm512_set1_ps(*ip.add(ic)), w, a0);
                a1 = _mm512_fmadd_ps(_mm512_set1_ps(*ip.add(k + ic)), w, a1);
                a2 = _mm512_fmadd_ps(_mm512_set1_ps(*ip.add(2 * k + ic)), w, a2);
                a3 = _mm512_fmadd_ps(_mm512_set1_ps(*ip.add(3 * k + ic)), w, a3);
            }
            if do_relu {
                a0 = _mm512_max_ps(a0, zero);
                a1 = _mm512_max_ps(a1, zero);
                a2 = _mm512_max_ps(a2, zero);
                a3 = _mm512_max_ps(a3, zero);
            }
            let op = output.as_mut_ptr().add(row * n + oc_base);
            _mm512_storeu_ps(op, a0);
            _mm512_storeu_ps(op.add(n), a1);
            _mm512_storeu_ps(op.add(2 * n), a2);
            _mm512_storeu_ps(op.add(3 * n), a3);
            if do_silu {
                for r in 0..4 {
                    for v in &mut output[(row + r) * n + oc_base..(row + r) * n + oc_base + 16] {
                        *v = apply_conv_activation_scalar(*v, Activation::Silu);
                    }
                }
            }
            row += 4;
        }

        while row < rows {
            let mut acc = bias_v;
            if let Some(res) = residual {
                acc = _mm512_add_ps(acc, _mm512_loadu_ps(res.as_ptr().add(row * n + oc_base)));
            }
            let ip = input.as_ptr().add(row * k);
            for ic in 0..k {
                let w = _mm512_loadu_ps(kernel.as_ptr().add(ic * n + oc_base));
                acc = _mm512_fmadd_ps(_mm512_set1_ps(*ip.add(ic)), w, acc);
            }
            if do_relu {
                acc = _mm512_max_ps(acc, zero);
            }
            _mm512_storeu_ps(output.as_mut_ptr().add(row * n + oc_base), acc);
            if do_silu {
                for v in &mut output[row * n + oc_base..row * n + oc_base + 16] {
                    *v = apply_conv_activation_scalar(*v, Activation::Silu);
                }
            }
            row += 1;
        }
    }
}

/// K-cache-blocked variant of [`pointwise_nx16_direct_rows_avx512`] for
/// shapes where the per-oc_block weight footprint (K × 16 × 4 bytes)
/// exceeds L1 (32 KB on Zen 4 = K > 512).
///
/// The original kernel loops `oc_base outer → row_tile middle → ic
/// inner`. For each (oc_base, row_tile), it loads K weight ZMMs. With
/// K=672 and 16-lane oc-blocks, that's 43 KB per oc_block which
/// overflows L1 — every row_tile re-fetches weights from L2.
///
/// This variant adds an outer K-chunk loop. With `KC = 128`, weight per
/// oc_block per chunk is `128 × 16 × 4 = 8 KB` → L1-resident. Across the
/// 32 row_tiles per oc_block (M=256 / 8-row tile), the same 8 KB are
/// reused → 32× hit rate. Partial sums accumulate in the output buffer
/// across K-chunks; bias + residual are applied on the first chunk only,
/// activation on the last. Trades a small amount of partial-sum write traffic
/// for eliminating the redundant per-row-tile weight reloads from L2.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn, clippy::too_many_arguments)]
unsafe fn pointwise_nx16_direct_rows_avx512_kcblock(
    input: &[f32],
    kernel: &[f32],
    bias: Option<&[f32]>,
    residual: Option<&[f32]>,
    output: &mut [f32],
    rows: usize,
    k: usize,
    n: usize,
    activation: Activation,
) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    /// K chunk size. KC=128 → 8 KB of weight per oc_block per chunk
    /// (L1-resident). Override via `YSCV_KCBLOCK_KC` for tuning.
    const KC_DEFAULT: usize = 128;
    let kc_size: usize = {
        use std::sync::OnceLock;
        static CACHED: OnceLock<usize> = OnceLock::new();
        *CACHED.get_or_init(|| {
            std::env::var("YSCV_KCBLOCK_KC")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .filter(|kc| (8..=1024).contains(kc))
                .unwrap_or(KC_DEFAULT)
        })
    };

    let zero = _mm512_setzero_ps();
    let do_relu = matches!(activation, Activation::Relu);
    let do_silu = matches!(activation, Activation::Silu);

    let mut kc_start = 0usize;
    while kc_start < k {
        let kc_end = (kc_start + kc_size).min(k);
        let is_first = kc_start == 0;
        let is_last = kc_end == k;

        for oc_base in (0..n).step_by(16) {
            let bias_v = if is_first {
                bias.map_or(zero, |b| _mm512_loadu_ps(b.as_ptr().add(oc_base)))
            } else {
                // Mid-K-chunk: bias was already folded into output on the
                // first chunk; don't add it again here.
                zero
            };

            let mut row = 0usize;
            // 8-row tile.
            while row + 8 <= rows {
                let op = output.as_mut_ptr().add(row * n + oc_base);
                // Seed accumulators: bias+residual on first chunk, else
                // load the partial sums written by the previous chunk.
                let (mut a0, mut a1, mut a2, mut a3, mut a4, mut a5, mut a6, mut a7) = if is_first {
                    let mut a0 = bias_v;
                    let mut a1 = bias_v;
                    let mut a2 = bias_v;
                    let mut a3 = bias_v;
                    let mut a4 = bias_v;
                    let mut a5 = bias_v;
                    let mut a6 = bias_v;
                    let mut a7 = bias_v;
                    if let Some(res) = residual {
                        let rp = res.as_ptr().add(row * n + oc_base);
                        a0 = _mm512_add_ps(a0, _mm512_loadu_ps(rp));
                        a1 = _mm512_add_ps(a1, _mm512_loadu_ps(rp.add(n)));
                        a2 = _mm512_add_ps(a2, _mm512_loadu_ps(rp.add(2 * n)));
                        a3 = _mm512_add_ps(a3, _mm512_loadu_ps(rp.add(3 * n)));
                        a4 = _mm512_add_ps(a4, _mm512_loadu_ps(rp.add(4 * n)));
                        a5 = _mm512_add_ps(a5, _mm512_loadu_ps(rp.add(5 * n)));
                        a6 = _mm512_add_ps(a6, _mm512_loadu_ps(rp.add(6 * n)));
                        a7 = _mm512_add_ps(a7, _mm512_loadu_ps(rp.add(7 * n)));
                    }
                    (a0, a1, a2, a3, a4, a5, a6, a7)
                } else {
                    (
                        _mm512_loadu_ps(op),
                        _mm512_loadu_ps(op.add(n)),
                        _mm512_loadu_ps(op.add(2 * n)),
                        _mm512_loadu_ps(op.add(3 * n)),
                        _mm512_loadu_ps(op.add(4 * n)),
                        _mm512_loadu_ps(op.add(5 * n)),
                        _mm512_loadu_ps(op.add(6 * n)),
                        _mm512_loadu_ps(op.add(7 * n)),
                    )
                };
                let ip = input.as_ptr().add(row * k);
                for ic in kc_start..kc_end {
                    let w = _mm512_loadu_ps(kernel.as_ptr().add(ic * n + oc_base));
                    a0 = _mm512_fmadd_ps(_mm512_set1_ps(*ip.add(ic)), w, a0);
                    a1 = _mm512_fmadd_ps(_mm512_set1_ps(*ip.add(k + ic)), w, a1);
                    a2 = _mm512_fmadd_ps(_mm512_set1_ps(*ip.add(2 * k + ic)), w, a2);
                    a3 = _mm512_fmadd_ps(_mm512_set1_ps(*ip.add(3 * k + ic)), w, a3);
                    a4 = _mm512_fmadd_ps(_mm512_set1_ps(*ip.add(4 * k + ic)), w, a4);
                    a5 = _mm512_fmadd_ps(_mm512_set1_ps(*ip.add(5 * k + ic)), w, a5);
                    a6 = _mm512_fmadd_ps(_mm512_set1_ps(*ip.add(6 * k + ic)), w, a6);
                    a7 = _mm512_fmadd_ps(_mm512_set1_ps(*ip.add(7 * k + ic)), w, a7);
                }
                if is_last && do_relu {
                    a0 = _mm512_max_ps(a0, zero);
                    a1 = _mm512_max_ps(a1, zero);
                    a2 = _mm512_max_ps(a2, zero);
                    a3 = _mm512_max_ps(a3, zero);
                    a4 = _mm512_max_ps(a4, zero);
                    a5 = _mm512_max_ps(a5, zero);
                    a6 = _mm512_max_ps(a6, zero);
                    a7 = _mm512_max_ps(a7, zero);
                }
                _mm512_storeu_ps(op, a0);
                _mm512_storeu_ps(op.add(n), a1);
                _mm512_storeu_ps(op.add(2 * n), a2);
                _mm512_storeu_ps(op.add(3 * n), a3);
                _mm512_storeu_ps(op.add(4 * n), a4);
                _mm512_storeu_ps(op.add(5 * n), a5);
                _mm512_storeu_ps(op.add(6 * n), a6);
                _mm512_storeu_ps(op.add(7 * n), a7);
                if is_last && do_silu {
                    for r in 0..8 {
                        for v in &mut output[(row + r) * n + oc_base..(row + r) * n + oc_base + 16]
                        {
                            *v = apply_conv_activation_scalar(*v, Activation::Silu);
                        }
                    }
                }
                row += 8;
            }
            // 4-row tile.
            while row + 4 <= rows {
                let op = output.as_mut_ptr().add(row * n + oc_base);
                let (mut a0, mut a1, mut a2, mut a3) = if is_first {
                    let mut a0 = bias_v;
                    let mut a1 = bias_v;
                    let mut a2 = bias_v;
                    let mut a3 = bias_v;
                    if let Some(res) = residual {
                        let rp = res.as_ptr().add(row * n + oc_base);
                        a0 = _mm512_add_ps(a0, _mm512_loadu_ps(rp));
                        a1 = _mm512_add_ps(a1, _mm512_loadu_ps(rp.add(n)));
                        a2 = _mm512_add_ps(a2, _mm512_loadu_ps(rp.add(2 * n)));
                        a3 = _mm512_add_ps(a3, _mm512_loadu_ps(rp.add(3 * n)));
                    }
                    (a0, a1, a2, a3)
                } else {
                    (
                        _mm512_loadu_ps(op),
                        _mm512_loadu_ps(op.add(n)),
                        _mm512_loadu_ps(op.add(2 * n)),
                        _mm512_loadu_ps(op.add(3 * n)),
                    )
                };
                let ip = input.as_ptr().add(row * k);
                for ic in kc_start..kc_end {
                    let w = _mm512_loadu_ps(kernel.as_ptr().add(ic * n + oc_base));
                    a0 = _mm512_fmadd_ps(_mm512_set1_ps(*ip.add(ic)), w, a0);
                    a1 = _mm512_fmadd_ps(_mm512_set1_ps(*ip.add(k + ic)), w, a1);
                    a2 = _mm512_fmadd_ps(_mm512_set1_ps(*ip.add(2 * k + ic)), w, a2);
                    a3 = _mm512_fmadd_ps(_mm512_set1_ps(*ip.add(3 * k + ic)), w, a3);
                }
                if is_last && do_relu {
                    a0 = _mm512_max_ps(a0, zero);
                    a1 = _mm512_max_ps(a1, zero);
                    a2 = _mm512_max_ps(a2, zero);
                    a3 = _mm512_max_ps(a3, zero);
                }
                _mm512_storeu_ps(op, a0);
                _mm512_storeu_ps(op.add(n), a1);
                _mm512_storeu_ps(op.add(2 * n), a2);
                _mm512_storeu_ps(op.add(3 * n), a3);
                if is_last && do_silu {
                    for r in 0..4 {
                        for v in &mut output[(row + r) * n + oc_base..(row + r) * n + oc_base + 16]
                        {
                            *v = apply_conv_activation_scalar(*v, Activation::Silu);
                        }
                    }
                }
                row += 4;
            }
            // 1-row tail.
            while row < rows {
                let op = output.as_mut_ptr().add(row * n + oc_base);
                let mut acc = if is_first {
                    let mut acc = bias_v;
                    if let Some(res) = residual {
                        acc = _mm512_add_ps(
                            acc,
                            _mm512_loadu_ps(res.as_ptr().add(row * n + oc_base)),
                        );
                    }
                    acc
                } else {
                    _mm512_loadu_ps(op)
                };
                let ip = input.as_ptr().add(row * k);
                for ic in kc_start..kc_end {
                    let w = _mm512_loadu_ps(kernel.as_ptr().add(ic * n + oc_base));
                    acc = _mm512_fmadd_ps(_mm512_set1_ps(*ip.add(ic)), w, acc);
                }
                if is_last && do_relu {
                    acc = _mm512_max_ps(acc, zero);
                }
                _mm512_storeu_ps(op, acc);
                if is_last && do_silu {
                    for v in &mut output[row * n + oc_base..row * n + oc_base + 16] {
                        *v = apply_conv_activation_scalar(*v, Activation::Silu);
                    }
                }
                row += 1;
            }
        }

        kc_start = kc_end;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx,fma")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn, clippy::too_many_arguments)]
unsafe fn pointwise_16x16_direct_rows_avx2(
    input: &[f32],
    kernel: &[f32],
    bias: Option<&[f32]>,
    residual: Option<&[f32]>,
    output: &mut [f32],
    rows: usize,
    activation: Activation,
) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let zero = _mm256_setzero_ps();
    let (bias0, bias1) = if let Some(b) = bias {
        (
            _mm256_loadu_ps(b.as_ptr()),
            _mm256_loadu_ps(b.as_ptr().add(8)),
        )
    } else {
        (zero, zero)
    };
    let do_relu = matches!(activation, Activation::Relu);
    let do_silu = matches!(activation, Activation::Silu);

    for row in 0..rows {
        let mut a0 = bias0;
        let mut a1 = bias1;
        if let Some(skip) = residual {
            let rp = skip.as_ptr().add(row * 16);
            a0 = _mm256_add_ps(a0, _mm256_loadu_ps(rp));
            a1 = _mm256_add_ps(a1, _mm256_loadu_ps(rp.add(8)));
        }
        let ip = input.as_ptr().add(row * 16);
        for ic in 0..16 {
            let x = _mm256_set1_ps(*ip.add(ic));
            let kp = kernel.as_ptr().add(ic * 16);
            a0 = _mm256_fmadd_ps(x, _mm256_loadu_ps(kp), a0);
            a1 = _mm256_fmadd_ps(x, _mm256_loadu_ps(kp.add(8)), a1);
        }
        if do_relu {
            a0 = _mm256_max_ps(a0, zero);
            a1 = _mm256_max_ps(a1, zero);
        }
        let op = output.as_mut_ptr().add(row * 16);
        _mm256_storeu_ps(op, a0);
        _mm256_storeu_ps(op.add(8), a1);
        if do_silu {
            for v in &mut output[row * 16..(row + 1) * 16] {
                *v = apply_conv_activation_scalar(*v, Activation::Silu);
            }
        }
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx,fma")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn, clippy::too_many_arguments)]
unsafe fn pointwise_nx16_direct_rows_avx2(
    input: &[f32],
    kernel: &[f32],
    bias: Option<&[f32]>,
    residual: Option<&[f32]>,
    output: &mut [f32],
    rows: usize,
    k: usize,
    n: usize,
    activation: Activation,
) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let zero = _mm256_setzero_ps();
    let do_relu = matches!(activation, Activation::Relu);
    let do_silu = matches!(activation, Activation::Silu);

    for oc_base in (0..n).step_by(16) {
        let (bias0, bias1) = if let Some(b) = bias {
            (
                _mm256_loadu_ps(b.as_ptr().add(oc_base)),
                _mm256_loadu_ps(b.as_ptr().add(oc_base + 8)),
            )
        } else {
            (zero, zero)
        };
        for row in 0..rows {
            let mut a0 = bias0;
            let mut a1 = bias1;
            if let Some(res) = residual {
                let rp = res.as_ptr().add(row * n + oc_base);
                a0 = _mm256_add_ps(a0, _mm256_loadu_ps(rp));
                a1 = _mm256_add_ps(a1, _mm256_loadu_ps(rp.add(8)));
            }
            let ip = input.as_ptr().add(row * k);
            for ic in 0..k {
                let x = _mm256_set1_ps(*ip.add(ic));
                let kp = kernel.as_ptr().add(ic * n + oc_base);
                a0 = _mm256_fmadd_ps(x, _mm256_loadu_ps(kp), a0);
                a1 = _mm256_fmadd_ps(x, _mm256_loadu_ps(kp.add(8)), a1);
            }
            if do_relu {
                a0 = _mm256_max_ps(a0, zero);
                a1 = _mm256_max_ps(a1, zero);
            }
            let op = output.as_mut_ptr().add(row * n + oc_base);
            _mm256_storeu_ps(op, a0);
            _mm256_storeu_ps(op.add(8), a1);
            if do_silu {
                for v in &mut output[row * n + oc_base..row * n + oc_base + 16] {
                    *v = apply_conv_activation_scalar(*v, Activation::Silu);
                }
            }
        }
    }
}

/// L1 prefetch hint (`prfm pldl1keep`) for the strided PW-reduce weight stream.
/// The weight stride is `n*4` bytes (96-448 B for tracker reduce shapes) — too
/// large for the in-order A53's HW prefetcher stride detector, so a manual hint
/// several K-iters ahead hides the load latency behind the FMA pipe.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
unsafe fn prefetch_l1_keep(p: *const f32) {
    core::arch::asm!("prfm pldl1keep, [{p}]", p = in(reg) p, options(nostack, preserves_flags, readonly));
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn, clippy::too_many_arguments)]
unsafe fn pointwise_16x16_direct_rows_neon(
    input: &[f32],
    kernel: &[f32],
    bias: Option<&[f32]>,
    residual: Option<&[f32]>,
    output: &mut [f32],
    rows: usize,
    activation: Activation,
) {
    use std::arch::aarch64::*;

    let (bias0, bias1, bias2, bias3) = if let Some(b) = bias {
        (
            vld1q_f32(b.as_ptr()),
            vld1q_f32(b.as_ptr().add(4)),
            vld1q_f32(b.as_ptr().add(8)),
            vld1q_f32(b.as_ptr().add(12)),
        )
    } else {
        let z = vdupq_n_f32(0.0);
        (z, z, z, z)
    };
    let do_relu = matches!(activation, Activation::Relu);
    let do_silu = matches!(activation, Activation::Silu);
    let zero = vdupq_n_f32(0.0);

    for row in 0..rows {
        let mut a0 = bias0;
        let mut a1 = bias1;
        let mut a2 = bias2;
        let mut a3 = bias3;
        if let Some(skip) = residual {
            let rp = skip.as_ptr().add(row * 16);
            a0 = vaddq_f32(a0, vld1q_f32(rp));
            a1 = vaddq_f32(a1, vld1q_f32(rp.add(4)));
            a2 = vaddq_f32(a2, vld1q_f32(rp.add(8)));
            a3 = vaddq_f32(a3, vld1q_f32(rp.add(12)));
        }
        let ip = input.as_ptr().add(row * 16);
        for ic in 0..16 {
            let x = vdupq_n_f32(*ip.add(ic));
            let kp = kernel.as_ptr().add(ic * 16);
            a0 = vfmaq_f32(a0, x, vld1q_f32(kp));
            a1 = vfmaq_f32(a1, x, vld1q_f32(kp.add(4)));
            a2 = vfmaq_f32(a2, x, vld1q_f32(kp.add(8)));
            a3 = vfmaq_f32(a3, x, vld1q_f32(kp.add(12)));
        }
        if do_relu {
            a0 = vmaxq_f32(a0, zero);
            a1 = vmaxq_f32(a1, zero);
            a2 = vmaxq_f32(a2, zero);
            a3 = vmaxq_f32(a3, zero);
        }
        let op = output.as_mut_ptr().add(row * 16);
        vst1q_f32(op, a0);
        vst1q_f32(op.add(4), a1);
        vst1q_f32(op.add(8), a2);
        vst1q_f32(op.add(12), a3);
        if do_silu {
            for v in &mut output[row * 16..(row + 1) * 16] {
                *v = apply_conv_activation_scalar(*v, Activation::Silu);
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn, clippy::too_many_arguments)]
unsafe fn pointwise_nx16_direct_rows_neon(
    input: &[f32],
    kernel: &[f32],
    bias: Option<&[f32]>,
    residual: Option<&[f32]>,
    output: &mut [f32],
    rows: usize,
    k: usize,
    n: usize,
    activation: Activation,
) {
    use std::arch::aarch64::*;

    let zero = vdupq_n_f32(0.0);
    let do_relu = matches!(activation, Activation::Relu);
    let do_silu = matches!(activation, Activation::Silu);

    for oc_base in (0..n).step_by(16) {
        let (bias0, bias1, bias2, bias3) = if let Some(b) = bias {
            (
                vld1q_f32(b.as_ptr().add(oc_base)),
                vld1q_f32(b.as_ptr().add(oc_base + 4)),
                vld1q_f32(b.as_ptr().add(oc_base + 8)),
                vld1q_f32(b.as_ptr().add(oc_base + 12)),
            )
        } else {
            (zero, zero, zero, zero)
        };
        // 4-row weight-stationary tile: the PW-reduce weight is independent of
        // the output row, so loading each weight quad once and feeding 4 rows
        // (16 accumulators in flight) cuts the weight stream 4× and hides the
        // in-order load latency — the same register-blocking the DW kernel uses.
        const PREFETCH_AHEAD: usize = 8;
        let pf = pw_prefetch_enabled();
        let mut row = 0;
        while row + 4 <= rows {
            let mut a00 = bias0;
            let mut a01 = bias1;
            let mut a02 = bias2;
            let mut a03 = bias3;
            let mut a10 = bias0;
            let mut a11 = bias1;
            let mut a12 = bias2;
            let mut a13 = bias3;
            let mut a20 = bias0;
            let mut a21 = bias1;
            let mut a22 = bias2;
            let mut a23 = bias3;
            let mut a30 = bias0;
            let mut a31 = bias1;
            let mut a32 = bias2;
            let mut a33 = bias3;
            if let Some(res) = residual {
                let rp = res.as_ptr().add(row * n + oc_base);
                a00 = vaddq_f32(a00, vld1q_f32(rp));
                a01 = vaddq_f32(a01, vld1q_f32(rp.add(4)));
                a02 = vaddq_f32(a02, vld1q_f32(rp.add(8)));
                a03 = vaddq_f32(a03, vld1q_f32(rp.add(12)));
                let rp = rp.add(n);
                a10 = vaddq_f32(a10, vld1q_f32(rp));
                a11 = vaddq_f32(a11, vld1q_f32(rp.add(4)));
                a12 = vaddq_f32(a12, vld1q_f32(rp.add(8)));
                a13 = vaddq_f32(a13, vld1q_f32(rp.add(12)));
                let rp = rp.add(n);
                a20 = vaddq_f32(a20, vld1q_f32(rp));
                a21 = vaddq_f32(a21, vld1q_f32(rp.add(4)));
                a22 = vaddq_f32(a22, vld1q_f32(rp.add(8)));
                a23 = vaddq_f32(a23, vld1q_f32(rp.add(12)));
                let rp = rp.add(n);
                a30 = vaddq_f32(a30, vld1q_f32(rp));
                a31 = vaddq_f32(a31, vld1q_f32(rp.add(4)));
                a32 = vaddq_f32(a32, vld1q_f32(rp.add(8)));
                a33 = vaddq_f32(a33, vld1q_f32(rp.add(12)));
            }
            let ip0 = input.as_ptr().add(row * k);
            let ip1 = ip0.add(k);
            let ip2 = ip1.add(k);
            let ip3 = ip2.add(k);
            macro_rules! fma16 {
                ($ic:expr) => {{
                    let kp = kernel.as_ptr().add($ic * n + oc_base);
                    let w0 = vld1q_f32(kp);
                    let w1 = vld1q_f32(kp.add(4));
                    let w2 = vld1q_f32(kp.add(8));
                    let w3 = vld1q_f32(kp.add(12));
                    let x0 = vdupq_n_f32(*ip0.add($ic));
                    a00 = vfmaq_f32(a00, x0, w0);
                    a01 = vfmaq_f32(a01, x0, w1);
                    a02 = vfmaq_f32(a02, x0, w2);
                    a03 = vfmaq_f32(a03, x0, w3);
                    let x1 = vdupq_n_f32(*ip1.add($ic));
                    a10 = vfmaq_f32(a10, x1, w0);
                    a11 = vfmaq_f32(a11, x1, w1);
                    a12 = vfmaq_f32(a12, x1, w2);
                    a13 = vfmaq_f32(a13, x1, w3);
                    let x2 = vdupq_n_f32(*ip2.add($ic));
                    a20 = vfmaq_f32(a20, x2, w0);
                    a21 = vfmaq_f32(a21, x2, w1);
                    a22 = vfmaq_f32(a22, x2, w2);
                    a23 = vfmaq_f32(a23, x2, w3);
                    let x3 = vdupq_n_f32(*ip3.add($ic));
                    a30 = vfmaq_f32(a30, x3, w0);
                    a31 = vfmaq_f32(a31, x3, w1);
                    a32 = vfmaq_f32(a32, x3, w2);
                    a33 = vfmaq_f32(a33, x3, w3);
                }};
            }
            // fmla-by-lane: one weight-quad set, 16 FMAs reading activation
            // lanes directly (no per-k `vdupq_n` broadcast on the NEON pipe).
            macro_rules! ko_block {
                ($kk:expr, $av0:expr, $av1:expr, $av2:expr, $av3:expr, $lane:literal) => {{
                    let kp = kernel.as_ptr().add($kk * n + oc_base);
                    let w0 = vld1q_f32(kp);
                    let w1 = vld1q_f32(kp.add(4));
                    let w2 = vld1q_f32(kp.add(8));
                    let w3 = vld1q_f32(kp.add(12));
                    a00 = vfmaq_laneq_f32::<$lane>(a00, w0, $av0);
                    a01 = vfmaq_laneq_f32::<$lane>(a01, w1, $av0);
                    a02 = vfmaq_laneq_f32::<$lane>(a02, w2, $av0);
                    a03 = vfmaq_laneq_f32::<$lane>(a03, w3, $av0);
                    a10 = vfmaq_laneq_f32::<$lane>(a10, w0, $av1);
                    a11 = vfmaq_laneq_f32::<$lane>(a11, w1, $av1);
                    a12 = vfmaq_laneq_f32::<$lane>(a12, w2, $av1);
                    a13 = vfmaq_laneq_f32::<$lane>(a13, w3, $av1);
                    a20 = vfmaq_laneq_f32::<$lane>(a20, w0, $av2);
                    a21 = vfmaq_laneq_f32::<$lane>(a21, w1, $av2);
                    a22 = vfmaq_laneq_f32::<$lane>(a22, w2, $av2);
                    a23 = vfmaq_laneq_f32::<$lane>(a23, w3, $av2);
                    a30 = vfmaq_laneq_f32::<$lane>(a30, w0, $av3);
                    a31 = vfmaq_laneq_f32::<$lane>(a31, w1, $av3);
                    a32 = vfmaq_laneq_f32::<$lane>(a32, w2, $av3);
                    a33 = vfmaq_laneq_f32::<$lane>(a33, w3, $av3);
                }};
            }
            macro_rules! fma16_lane {
                ($ic:expr) => {{
                    let av0 = vld1q_f32(ip0.add($ic));
                    let av1 = vld1q_f32(ip1.add($ic));
                    let av2 = vld1q_f32(ip2.add($ic));
                    let av3 = vld1q_f32(ip3.add($ic));
                    ko_block!($ic, av0, av1, av2, av3, 0);
                    ko_block!($ic + 1, av0, av1, av2, av3, 1);
                    ko_block!($ic + 2, av0, av1, av2, av3, 2);
                    ko_block!($ic + 3, av0, av1, av2, av3, 3);
                }};
            }
            if reduce_bylane() {
                let kb = (k / 4) * 4;
                let mut ic = 0;
                while ic < kb {
                    if pf && ic + PREFETCH_AHEAD + 4 <= k {
                        prefetch_l1_keep(kernel.as_ptr().add((ic + PREFETCH_AHEAD) * n + oc_base));
                        prefetch_l1_keep(
                            kernel.as_ptr().add((ic + PREFETCH_AHEAD + 2) * n + oc_base),
                        );
                    }
                    fma16_lane!(ic);
                    ic += 4;
                }
                for ic in kb..k {
                    fma16!(ic);
                }
            } else {
                let sp = if pf {
                    k.saturating_sub(PREFETCH_AHEAD)
                } else {
                    0
                };
                for ic in 0..sp {
                    prefetch_l1_keep(kernel.as_ptr().add((ic + PREFETCH_AHEAD) * n + oc_base));
                    fma16!(ic);
                }
                for ic in sp..k {
                    fma16!(ic);
                }
            }
            if do_relu {
                a00 = vmaxq_f32(a00, zero);
                a01 = vmaxq_f32(a01, zero);
                a02 = vmaxq_f32(a02, zero);
                a03 = vmaxq_f32(a03, zero);
                a10 = vmaxq_f32(a10, zero);
                a11 = vmaxq_f32(a11, zero);
                a12 = vmaxq_f32(a12, zero);
                a13 = vmaxq_f32(a13, zero);
                a20 = vmaxq_f32(a20, zero);
                a21 = vmaxq_f32(a21, zero);
                a22 = vmaxq_f32(a22, zero);
                a23 = vmaxq_f32(a23, zero);
                a30 = vmaxq_f32(a30, zero);
                a31 = vmaxq_f32(a31, zero);
                a32 = vmaxq_f32(a32, zero);
                a33 = vmaxq_f32(a33, zero);
            }
            let op = output.as_mut_ptr().add(row * n + oc_base);
            vst1q_f32(op, a00);
            vst1q_f32(op.add(4), a01);
            vst1q_f32(op.add(8), a02);
            vst1q_f32(op.add(12), a03);
            let op = op.add(n);
            vst1q_f32(op, a10);
            vst1q_f32(op.add(4), a11);
            vst1q_f32(op.add(8), a12);
            vst1q_f32(op.add(12), a13);
            let op = op.add(n);
            vst1q_f32(op, a20);
            vst1q_f32(op.add(4), a21);
            vst1q_f32(op.add(8), a22);
            vst1q_f32(op.add(12), a23);
            let op = op.add(n);
            vst1q_f32(op, a30);
            vst1q_f32(op.add(4), a31);
            vst1q_f32(op.add(8), a32);
            vst1q_f32(op.add(12), a33);
            if do_silu {
                for r in 0..4 {
                    let base = (row + r) * n + oc_base;
                    for v in &mut output[base..base + 16] {
                        *v = apply_conv_activation_scalar(*v, Activation::Silu);
                    }
                }
            }
            row += 4;
        }
        for row in row..rows {
            let mut a0 = bias0;
            let mut a1 = bias1;
            let mut a2 = bias2;
            let mut a3 = bias3;
            if let Some(res) = residual {
                let rp = res.as_ptr().add(row * n + oc_base);
                a0 = vaddq_f32(a0, vld1q_f32(rp));
                a1 = vaddq_f32(a1, vld1q_f32(rp.add(4)));
                a2 = vaddq_f32(a2, vld1q_f32(rp.add(8)));
                a3 = vaddq_f32(a3, vld1q_f32(rp.add(12)));
            }
            let ip = input.as_ptr().add(row * k);
            macro_rules! fma4 {
                ($ic:expr) => {{
                    let x = vdupq_n_f32(*ip.add($ic));
                    let kp = kernel.as_ptr().add($ic * n + oc_base);
                    a0 = vfmaq_f32(a0, x, vld1q_f32(kp));
                    a1 = vfmaq_f32(a1, x, vld1q_f32(kp.add(4)));
                    a2 = vfmaq_f32(a2, x, vld1q_f32(kp.add(8)));
                    a3 = vfmaq_f32(a3, x, vld1q_f32(kp.add(12)));
                }};
            }
            let sp = if pf {
                k.saturating_sub(PREFETCH_AHEAD)
            } else {
                0
            };
            for ic in 0..sp {
                prefetch_l1_keep(kernel.as_ptr().add((ic + PREFETCH_AHEAD) * n + oc_base));
                fma4!(ic);
            }
            for ic in sp..k {
                fma4!(ic);
            }
            if do_relu {
                a0 = vmaxq_f32(a0, zero);
                a1 = vmaxq_f32(a1, zero);
                a2 = vmaxq_f32(a2, zero);
                a3 = vmaxq_f32(a3, zero);
            }
            let op = output.as_mut_ptr().add(row * n + oc_base);
            vst1q_f32(op, a0);
            vst1q_f32(op.add(4), a1);
            vst1q_f32(op.add(8), a2);
            vst1q_f32(op.add(12), a3);
            if do_silu {
                for v in &mut output[row * n + oc_base..row * n + oc_base + 16] {
                    *v = apply_conv_activation_scalar(*v, Activation::Silu);
                }
            }
        }
    }
}
