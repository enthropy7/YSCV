use rayon::ThreadPool;
use yscv_tensor::{AlignedVec, Tensor, TensorError};

use super::super::error::KernelError;
use super::config::{
    ParallelElementwiseConfig, Pool2dKind, Pool2dPlan, Pool2dSpec, should_parallelize_len,
};

pub fn max_pool2d_nhwc_with_config_and_pool(
    input: &Tensor,
    spec: Pool2dSpec,
    config: ParallelElementwiseConfig,
    thread_pool: Option<&ThreadPool>,
) -> Result<Tensor, KernelError> {
    pool2d_nhwc_with_config_and_pool(input, spec, config, thread_pool, Pool2dKind::Max)
}

pub fn avg_pool2d_nhwc_with_config_and_pool(
    input: &Tensor,
    spec: Pool2dSpec,
    config: ParallelElementwiseConfig,
    thread_pool: Option<&ThreadPool>,
) -> Result<Tensor, KernelError> {
    pool2d_nhwc_with_config_and_pool(input, spec, config, thread_pool, Pool2dKind::Avg)
}

fn pool2d_nhwc_with_config_and_pool(
    input: &Tensor,
    spec: Pool2dSpec,
    config: ParallelElementwiseConfig,
    thread_pool: Option<&ThreadPool>,
    kind: Pool2dKind,
) -> Result<Tensor, KernelError> {
    let plan = build_pool2d_plan(input, spec)?;
    let data = input.data();
    let out_row_len = plan.out_w * plan.channels;
    if plan.output_len == 0 || out_row_len == 0 {
        return Tensor::from_aligned(
            vec![plan.batch, plan.out_h, plan.out_w, plan.channels],
            AlignedVec::<f32>::calloc(plan.output_len),
        )
        .map_err(Into::into);
    }

    let mut output = AlignedVec::<f32>::uninitialized(plan.output_len);

    if should_parallelize_len(plan.output_len, config.min_parallel_elements, thread_pool) {
        // Route through `scope_ctx::par_chunks_mut_dispatch` — if the
        // ONNX runner installed a ParallelScope via `install_scope`,
        // all chunks go through it; otherwise falls back to rayon.
        super::super::scope_ctx::par_chunks_mut_dispatch(
            output.as_mut_slice(),
            out_row_len,
            |row_idx, out_row| {
                pool2d_nhwc_row(data, plan, row_idx, out_row, kind);
            },
        );
    } else {
        for (row_idx, out_row) in output.chunks_mut(out_row_len).enumerate() {
            pool2d_nhwc_row(data, plan, row_idx, out_row, kind);
        }
    }

    Tensor::from_aligned(
        vec![plan.batch, plan.out_h, plan.out_w, plan.channels],
        output,
    )
    .map_err(Into::into)
}

fn build_pool2d_plan(input: &Tensor, spec: Pool2dSpec) -> Result<Pool2dPlan, KernelError> {
    let kernel_h = spec.kernel_h;
    let kernel_w = spec.kernel_w;
    let stride_h = spec.stride_h;
    let stride_w = spec.stride_w;
    if input.rank() != 4 {
        return Err(KernelError::InvalidPoolRank {
            got_rank: input.rank(),
        });
    }
    if kernel_h == 0 || kernel_w == 0 || stride_h == 0 || stride_w == 0 {
        return Err(KernelError::InvalidPoolParameters {
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
        });
    }

    let batch = input.shape()[0];
    let in_h = input.shape()[1];
    let in_w = input.shape()[2];
    let channels = input.shape()[3];
    if kernel_h > in_h || kernel_w > in_w {
        return Err(KernelError::PoolKernelLargerThanInput {
            input_h: in_h,
            input_w: in_w,
            kernel_h,
            kernel_w,
        });
    }

    let out_h = (in_h - kernel_h) / stride_h + 1;
    let out_w = (in_w - kernel_w) / stride_w + 1;

    let output_len = batch
        .checked_mul(out_h)
        .and_then(|v| v.checked_mul(out_w))
        .and_then(|v| v.checked_mul(channels))
        .ok_or_else(|| {
            KernelError::Tensor(TensorError::SizeOverflow {
                shape: vec![batch, out_h, out_w, channels],
            })
        })?;

    Ok(Pool2dPlan {
        batch,
        in_h,
        in_w,
        channels,
        out_h,
        out_w,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        output_len,
    })
}

fn pool2d_nhwc_row(
    input: &[f32],
    plan: Pool2dPlan,
    row_idx: usize,
    out_row: &mut [f32],
    kind: Pool2dKind,
) {
    let batch_idx = row_idx / plan.out_h;
    let out_y = row_idx % plan.out_h;
    let in_y0 = out_y * plan.stride_h;
    let batch_input_base = batch_idx * plan.in_h * plan.in_w * plan.channels;
    let window_area = (plan.kernel_h * plan.kernel_w) as f32;
    let inv_area = 1.0 / window_area;

    // Fast path: 2×2 max pool with stride 2.
    if plan.kernel_h == 2
        && plan.kernel_w == 2
        && plan.stride_h == 2
        && plan.stride_w == 2
        && matches!(kind, Pool2dKind::Max)
    {
        if plan.channels == 1 {
            // Single-channel: pairwise SIMD across output pixels.
            pool2d_2x2s2_max_row(input, plan, batch_input_base, in_y0, out_row);
        } else {
            // Multi-channel: SIMD across channel dimension, 4 loads → 1 store.
            pool2d_2x2s2_max_row_mc(input, plan, batch_input_base, in_y0, out_row);
        }
        return;
    }

    for out_x in 0..plan.out_w {
        let in_x0 = out_x * plan.stride_w;
        let out_cell_base = out_x * plan.channels;
        let out_slice = &mut out_row[out_cell_base..out_cell_base + plan.channels];

        // Initialize
        match kind {
            Pool2dKind::Max => out_slice.fill(f32::NEG_INFINITY),
            Pool2dKind::Avg => out_slice.fill(0.0),
        }

        // Accumulate over kernel window
        for ky in 0..plan.kernel_h {
            let in_y = in_y0 + ky;
            let row_base = batch_input_base + (in_y * plan.in_w + in_x0) * plan.channels;
            for kx in 0..plan.kernel_w {
                let pixel_base = row_base + kx * plan.channels;
                let in_slice = &input[pixel_base..pixel_base + plan.channels];
                pool_accumulate(out_slice, in_slice, kind);
            }
        }

        // Finalize avg
        if matches!(kind, Pool2dKind::Avg) {
            for v in out_slice.iter_mut() {
                *v *= inv_area;
            }
        }
    }
}

/// Optimized 2×2 max-pool with stride 2, multi-channel (NHWC).
/// Loads all 4 kernel positions directly and computes max in registers,
/// avoiding the init + 4× accumulate pattern of the generic path.
#[allow(unsafe_code)]
fn pool2d_2x2s2_max_row_mc(
    input: &[f32],
    plan: Pool2dPlan,
    batch_input_base: usize,
    in_y0: usize,
    out_row: &mut [f32],
) {
    let c = plan.channels;
    let in_w_c = plan.in_w * c;
    let row0_base = batch_input_base + in_y0 * in_w_c;
    let row1_base = row0_base + in_w_c;

    for out_x in 0..plan.out_w {
        let in_x0 = out_x * 2;
        let out_off = out_x * c;
        // 4 pixel offsets in the 2×2 window
        let p00 = row0_base + in_x0 * c;
        let p01 = p00 + c;
        let p10 = row1_base + in_x0 * c;
        let p11 = p10 + c;
        let out_slice = &mut out_row[out_off..out_off + c];

        let mut i = 0usize;

        #[cfg(target_arch = "aarch64")]
        if !cfg!(miri) && c >= 4 && std::arch::is_aarch64_feature_detected!("neon") {
            unsafe {
                pool2d_2x2s2_max_mc_neon(input, p00, p01, p10, p11, out_slice, &mut i);
            }
        }

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if !cfg!(miri) && c >= 4 && std::is_x86_feature_detected!("sse") {
            unsafe {
                pool2d_2x2s2_max_mc_sse(input, p00, p01, p10, p11, out_slice, &mut i);
            }
        }

        // Scalar tail
        while i < c {
            let a = input[p00 + i];
            let b = input[p01 + i];
            let c_val = input[p10 + i];
            let d = input[p11 + i];
            out_slice[i] = a.max(b).max(c_val.max(d));
            i += 1;
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
unsafe fn pool2d_2x2s2_max_mc_neon(
    input: &[f32],
    p00: usize,
    p01: usize,
    p10: usize,
    p11: usize,
    out: &mut [f32],
    i: &mut usize,
) {
    use std::arch::aarch64::*;
    let ip = input.as_ptr();
    let op = out.as_mut_ptr();
    let len = out.len();
    while *i + 4 <= len {
        let a = vld1q_f32(ip.add(p00 + *i));
        let b = vld1q_f32(ip.add(p01 + *i));
        let c = vld1q_f32(ip.add(p10 + *i));
        let d = vld1q_f32(ip.add(p11 + *i));
        let m = vmaxq_f32(vmaxq_f32(a, b), vmaxq_f32(c, d));
        vst1q_f32(op.add(*i), m);
        *i += 4;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
unsafe fn pool2d_2x2s2_max_mc_sse(
    input: &[f32],
    p00: usize,
    p01: usize,
    p10: usize,
    p11: usize,
    out: &mut [f32],
    i: &mut usize,
) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;
    let ip = input.as_ptr();
    let op = out.as_mut_ptr();
    let len = out.len();
    while *i + 4 <= len {
        let a = _mm_loadu_ps(ip.add(p00 + *i));
        let b = _mm_loadu_ps(ip.add(p01 + *i));
        let c = _mm_loadu_ps(ip.add(p10 + *i));
        let d = _mm_loadu_ps(ip.add(p11 + *i));
        let m = _mm_max_ps(_mm_max_ps(a, b), _mm_max_ps(c, d));
        _mm_storeu_ps(op.add(*i), m);
        *i += 4;
    }
}

/// Optimized 2x2 max-pool with stride 2, channels==1.
/// Processes 4 output pixels at a time using SIMD pairwise max.
#[allow(unsafe_code)]
fn pool2d_2x2s2_max_row(
    input: &[f32],
    plan: Pool2dPlan,
    batch_input_base: usize,
    in_y0: usize,
    out_row: &mut [f32],
) {
    let in_w = plan.in_w;
    let row0_base = batch_input_base + in_y0 * in_w;
    let row1_base = batch_input_base + (in_y0 + 1) * in_w;

    let mut out_x = 0usize;

    // SIMD batch: process 4 output pixels (= 8 input pixels per row) at a time.
    #[cfg(target_arch = "aarch64")]
    if !cfg!(miri) && std::arch::is_aarch64_feature_detected!("neon") {
        unsafe {
            pool2d_2x2s2_max_row_neon(input, row0_base, row1_base, out_row, plan.out_w, &mut out_x);
        }
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if !cfg!(miri) && std::is_x86_feature_detected!("sse") {
        unsafe {
            pool2d_2x2s2_max_row_sse(input, row0_base, row1_base, out_row, plan.out_w, &mut out_x);
        }
    }

    // Scalar tail
    while out_x < plan.out_w {
        let in_x0 = out_x * 2;
        let a = input[row0_base + in_x0];
        let b = input[row0_base + in_x0 + 1];
        let c = input[row1_base + in_x0];
        let d = input[row1_base + in_x0 + 1];
        out_row[out_x] = a.max(b).max(c.max(d));
        out_x += 1;
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
unsafe fn pool2d_2x2s2_max_row_neon(
    input: &[f32],
    row0_base: usize,
    row1_base: usize,
    out_row: &mut [f32],
    out_w: usize,
    out_x: &mut usize,
) {
    use std::arch::aarch64::*;
    let ip = input.as_ptr();
    let op = out_row.as_mut_ptr();
    while *out_x + 4 <= out_w {
        let in_x0 = *out_x * 2;
        // Load 8 consecutive floats from each row (covers 4 output pixels × stride 2)
        let r0a = vld1q_f32(ip.add(row0_base + in_x0));
        let r0b = vld1q_f32(ip.add(row0_base + in_x0 + 4));
        let r1a = vld1q_f32(ip.add(row1_base + in_x0));
        let r1b = vld1q_f32(ip.add(row1_base + in_x0 + 4));
        // Max across rows
        let max0 = vmaxq_f32(r0a, r1a);
        let max1 = vmaxq_f32(r0b, r1b);
        // Pairwise max: take max of even/odd elements → 4 results
        let result = vpmaxq_f32(max0, max1);
        vst1q_f32(op.add(*out_x), result);
        *out_x += 4;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse3")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
unsafe fn pool2d_2x2s2_max_row_sse(
    input: &[f32],
    row0_base: usize,
    row1_base: usize,
    out_row: &mut [f32],
    out_w: usize,
    out_x: &mut usize,
) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;
    let ip = input.as_ptr();
    let op = out_row.as_mut_ptr();
    while *out_x + 4 <= out_w {
        let in_x0 = *out_x * 2;
        // Load 8 consecutive floats from each row
        let r0a = _mm_loadu_ps(ip.add(row0_base + in_x0));
        let r0b = _mm_loadu_ps(ip.add(row0_base + in_x0 + 4));
        let r1a = _mm_loadu_ps(ip.add(row1_base + in_x0));
        let r1b = _mm_loadu_ps(ip.add(row1_base + in_x0 + 4));
        // Max across rows
        let max0 = _mm_max_ps(r0a, r1a); // [m0, m1, m2, m3]
        let max1 = _mm_max_ps(r0b, r1b); // [m4, m5, m6, m7]
        // Pairwise max: shuffle to get even/odd pairs then max
        // max0 = [m0, m1, m2, m3], max1 = [m4, m5, m6, m7]
        // We want: max(m0,m1), max(m2,m3), max(m4,m5), max(m6,m7)
        let evens = _mm_shuffle_ps(max0, max1, 0b10_00_10_00); // [m0, m2, m4, m6]
        let odds = _mm_shuffle_ps(max0, max1, 0b11_01_11_01); // [m1, m3, m5, m7]
        let result = _mm_max_ps(evens, odds);
        _mm_storeu_ps(op.add(*out_x), result);
        *out_x += 4;
    }
}

/// SIMD-accelerated pool accumulation across channels
#[allow(unsafe_code)]
fn pool_accumulate(out: &mut [f32], input: &[f32], kind: Pool2dKind) {
    let len = out.len();
    debug_assert_eq!(len, input.len());

    if cfg!(miri) || len < 4 {
        match kind {
            Pool2dKind::Max => {
                for i in 0..len {
                    out[i] = out[i].max(input[i]);
                }
            }
            Pool2dKind::Avg => {
                for i in 0..len {
                    out[i] += input[i];
                }
            }
        }
        return;
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe { pool_accumulate_neon(out, input, kind) };
            return;
        }
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("sse") {
            unsafe { pool_accumulate_sse(out, input, kind) };
            return;
        }
    }

    match kind {
        Pool2dKind::Max => {
            for i in 0..len {
                out[i] = out[i].max(input[i]);
            }
        }
        Pool2dKind::Avg => {
            for i in 0..len {
                out[i] += input[i];
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "neon")]
unsafe fn pool_accumulate_neon(out: &mut [f32], input: &[f32], kind: Pool2dKind) {
    use std::arch::aarch64::*;
    let len = out.len();
    let op = out.as_mut_ptr();
    let ip = input.as_ptr();
    let mut i = 0usize;
    match kind {
        Pool2dKind::Max => {
            while i + 4 <= len {
                let o = vld1q_f32(op.add(i));
                let v = vld1q_f32(ip.add(i));
                vst1q_f32(op.add(i), vmaxq_f32(o, v));
                i += 4;
            }
            while i < len {
                let o = *op.add(i);
                let v = *ip.add(i);
                *op.add(i) = if o > v { o } else { v };
                i += 1;
            }
        }
        Pool2dKind::Avg => {
            while i + 4 <= len {
                let o = vld1q_f32(op.add(i));
                let v = vld1q_f32(ip.add(i));
                vst1q_f32(op.add(i), vaddq_f32(o, v));
                i += 4;
            }
            while i < len {
                *op.add(i) += *ip.add(i);
                i += 1;
            }
        }
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "sse")]
unsafe fn pool_accumulate_sse(out: &mut [f32], input: &[f32], kind: Pool2dKind) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;
    let len = out.len();
    let op = out.as_mut_ptr();
    let ip = input.as_ptr();
    let mut i = 0usize;
    match kind {
        Pool2dKind::Max => {
            while i + 4 <= len {
                let o = _mm_loadu_ps(op.add(i));
                let v = _mm_loadu_ps(ip.add(i));
                _mm_storeu_ps(op.add(i), _mm_max_ps(o, v));
                i += 4;
            }
            while i < len {
                let o = *op.add(i);
                let v = *ip.add(i);
                *op.add(i) = if o > v { o } else { v };
                i += 1;
            }
        }
        Pool2dKind::Avg => {
            while i + 4 <= len {
                let o = _mm_loadu_ps(op.add(i));
                let v = _mm_loadu_ps(ip.add(i));
                _mm_storeu_ps(op.add(i), _mm_add_ps(o, v));
                i += 4;
            }
            while i < len {
                *op.add(i) += *ip.add(i);
                i += 1;
            }
        }
    }
}

// ── NCHW pool functions ────────────────────────────────────────────

/// NCHW max pool: parallelized across channel planes, SIMD within each plane.
pub fn max_pool2d_nchw(
    input: &Tensor,
    kernel_h: usize,
    kernel_w: usize,
    stride_h: usize,
    stride_w: usize,
    pad_top: usize,
    pad_left: usize,
    pad_bottom: usize,
    pad_right: usize,
) -> Result<Tensor, KernelError> {
    pool2d_nchw(
        input,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        pad_top,
        pad_left,
        pad_bottom,
        pad_right,
        Pool2dKind::Max,
    )
}

/// NCHW avg pool: parallelized across channel planes, SIMD within each plane.
pub fn avg_pool2d_nchw(
    input: &Tensor,
    kernel_h: usize,
    kernel_w: usize,
    stride_h: usize,
    stride_w: usize,
    pad_top: usize,
    pad_left: usize,
    pad_bottom: usize,
    pad_right: usize,
) -> Result<Tensor, KernelError> {
    pool2d_nchw(
        input,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        pad_top,
        pad_left,
        pad_bottom,
        pad_right,
        Pool2dKind::Avg,
    )
}

fn pool2d_nchw(
    input: &Tensor,
    kernel_h: usize,
    kernel_w: usize,
    stride_h: usize,
    stride_w: usize,
    pad_top: usize,
    pad_left: usize,
    pad_bottom: usize,
    pad_right: usize,
    kind: Pool2dKind,
) -> Result<Tensor, KernelError> {
    if input.rank() != 4 {
        return Err(KernelError::InvalidPoolRank {
            got_rank: input.rank(),
        });
    }
    let s = input.shape();
    let (n, c, ih, iw) = (s[0], s[1], s[2], s[3]);
    let oh = (ih + pad_top + pad_bottom - kernel_h) / stride_h + 1;
    let ow = (iw + pad_left + pad_right - kernel_w) / stride_w + 1;
    let plane_out = oh * ow;
    let total_planes = n * c;
    let total_out = total_planes * plane_out;

    let data = input.data();
    let no_pad = pad_top == 0 && pad_left == 0 && pad_bottom == 0 && pad_right == 0;
    let inv_area = 1.0 / (kernel_h * kernel_w) as f32;

    let mut output = AlignedVec::<f32>::uninitialized(total_out);

    let work = |out_chunk: &mut [f32], plane_idx: usize| {
        let in_base = plane_idx * ih * iw;
        let plane = &data[in_base..in_base + ih * iw];

        // Fast path: 2×2 stride 2 max pool, no padding — reuse SIMD row helper.
        if no_pad
            && kernel_h == 2
            && kernel_w == 2
            && stride_h == 2
            && stride_w == 2
            && matches!(kind, Pool2dKind::Max)
        {
            let tmp_plan = Pool2dPlan {
                batch: 1,
                in_h: ih,
                in_w: iw,
                channels: 1,
                out_h: oh,
                out_w: ow,
                kernel_h: 2,
                kernel_w: 2,
                stride_h: 2,
                stride_w: 2,
                output_len: plane_out,
            };
            for oy in 0..oh {
                let out_row = &mut out_chunk[oy * ow..(oy + 1) * ow];
                pool2d_2x2s2_max_row(plane, tmp_plan, 0, oy * 2, out_row);
            }
            return;
        }

        // General path
        for oy in 0..oh {
            for ox in 0..ow {
                let mut val = match kind {
                    Pool2dKind::Max => f32::NEG_INFINITY,
                    Pool2dKind::Avg => 0.0,
                };
                if no_pad {
                    let iy0 = oy * stride_h;
                    let ix0 = ox * stride_w;
                    for ky in 0..kernel_h {
                        let row_off = (iy0 + ky) * iw + ix0;
                        for kx in 0..kernel_w {
                            let v = plane[row_off + kx];
                            match kind {
                                Pool2dKind::Max => {
                                    if v > val {
                                        val = v;
                                    }
                                }
                                Pool2dKind::Avg => val += v,
                            }
                        }
                    }
                } else {
                    for ky in 0..kernel_h {
                        let iy = oy * stride_h + ky;
                        if iy < pad_top || iy >= ih + pad_top {
                            continue;
                        }
                        let row_off = (iy - pad_top) * iw;
                        for kx in 0..kernel_w {
                            let ix = ox * stride_w + kx;
                            if ix < pad_left || ix >= iw + pad_left {
                                continue;
                            }
                            let v = plane[row_off + ix - pad_left];
                            match kind {
                                Pool2dKind::Max => {
                                    if v > val {
                                        val = v;
                                    }
                                }
                                Pool2dKind::Avg => val += v,
                            }
                        }
                    }
                }
                if matches!(kind, Pool2dKind::Avg) {
                    val *= inv_area;
                }
                out_chunk[oy * ow + ox] = val;
            }
        }
    };

    if total_out >= 262_144 {
        super::super::scope_ctx::par_chunks_mut_dispatch(
            output.as_mut_slice(),
            plane_out,
            |idx, chunk| work(chunk, idx),
        );
    } else {
        for (idx, chunk) in output.chunks_mut(plane_out).enumerate() {
            work(chunk, idx);
        }
    }

    Tensor::from_aligned(vec![n, c, oh, ow], output).map_err(Into::into)
}
