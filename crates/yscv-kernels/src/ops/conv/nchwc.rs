//! NCHWc-layout pointwise conv kernels: the per-arch block tiles
//! (AVX-512/AVX2/NEON) and the public NCHWc conv entry points.

use super::*;

/// Generic-stride Conv2d on an NCHWc input, producing NCHWc output.
///
/// Boundary-converting wrapper around the NHWC conv fast paths. The plan
/// (B.4) projects the largest win for 3×3 non-depthwise Conv when the
/// kernel can use direct SIMD loads along the C-block stride; a native
/// NCHWc kernel would emit `_mm256_loadu_ps` / `ld1` loops walking the
/// block dim directly. That native kernel is a separate landing —
/// here we provide the composable entry point so that graph passes
/// (B.6) can keep subgraphs in NCHWc and amortize the conversion cost
/// across multiple ops (3×3 Conv + Pool + Add + BN in a chain only pays
/// one round-trip conversion at the subgraph boundary, not per op).
///
/// Correctness: output is bitwise-identical to running the NHWC path on
/// the equivalent NHWC input, up to fp32 ULP. Validated in the
/// `nchwc` integration tests.
///
/// - `input` must be `Layout::NCHWc { block }` with shape
///   `[N, C_in/block, H, W, block]`.
/// - `kernel` is the standard NHWC kernel `[KH, KW, C_in, C_out]`.
/// - `actual_in_channels` is the real (non-padded) C_in, used to strip
///   zero-padded lanes on conversion.
/// - `spec.stride` / `bias` / `activation` match the NHWC path exactly.
///
/// Output is tagged `Layout::NCHWc { block }` with C_out padded up to a
/// block boundary if needed.
#[allow(clippy::too_many_arguments)]
pub fn conv2d_nchwc_with_activation_prepacked(
    input: &Tensor,
    kernel: &Tensor,
    bias: Option<&Tensor>,
    actual_in_channels: usize,
    spec: Conv2dSpec,
    activation: Activation,
    config: ParallelElementwiseConfig,
    thread_pool: Option<&ThreadPool>,
    prepacked_b: Option<&super::super::matmul::PackedB>,
) -> Result<Tensor, KernelError> {
    let in_shape = input.shape();
    if in_shape.len() != 5 {
        return Err(KernelError::LayoutConversion(format!(
            "conv2d_nchwc: expected 5-D NCHWc input, got {in_shape:?}"
        )));
    }
    let block = in_shape[4];
    if block == 0 || block > u8::MAX as usize {
        return Err(KernelError::LayoutConversion(format!(
            "conv2d_nchwc: invalid block size {block}"
        )));
    }

    // NCHWc -> NHWC, stripping padded channels.
    let input_nhwc = super::super::layout::nchwc_to_nhwc(input, actual_in_channels)?;

    let output_nhwc = conv2d_nhwc_with_activation_prepacked(
        &input_nhwc,
        kernel,
        bias,
        spec,
        activation,
        config,
        thread_pool,
        prepacked_b,
    )?;

    // NHWC -> NCHWc, padding C_out tail to block boundary.
    super::super::layout::nhwc_to_nchwc(&output_nhwc, block)
}

#[allow(clippy::too_many_arguments)]
#[inline]
fn nchwc_pointwise_full_block_dispatch(
    input: &[f32],
    kernel: &[f32],
    bias: Option<&[f32]>,
    residual: Option<&[f32]>,
    out_cell: &mut [f32],
    n: usize,
    pos: usize,
    oc_base: usize,
    actual_in_channels: usize,
    out_channels: usize,
    ci_blocks: usize,
    in_n_stride: usize,
    in_cb_stride: usize,
    block: usize,
    activation: Activation,
) {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if block == 8 && std::is_x86_feature_detected!("avx") && std::is_x86_feature_detected!("fma") {
        #[allow(unsafe_code)]
        unsafe {
            nchwc_pointwise_block8_avx_fma(
                input,
                kernel,
                bias,
                residual,
                out_cell,
                n,
                pos,
                oc_base,
                actual_in_channels,
                out_channels,
                ci_blocks,
                in_n_stride,
                in_cb_stride,
                activation,
            );
        }
        return;
    }
    #[cfg(target_arch = "aarch64")]
    if block == 8 && std::arch::is_aarch64_feature_detected!("neon") {
        #[allow(unsafe_code)]
        unsafe {
            nchwc_pointwise_block8_neon(
                input,
                kernel,
                bias,
                residual,
                out_cell,
                n,
                pos,
                oc_base,
                actual_in_channels,
                out_channels,
                ci_blocks,
                in_n_stride,
                in_cb_stride,
                activation,
            );
        }
        return;
    }
    nchwc_pointwise_tail_block_scalar(
        input,
        kernel,
        bias,
        residual,
        out_cell,
        n,
        pos,
        oc_base,
        actual_in_channels,
        out_channels,
        ci_blocks,
        in_n_stride,
        in_cb_stride,
        block,
        activation,
    );
}

#[allow(clippy::too_many_arguments)]
#[inline]
fn nchwc_pointwise_full_block4_dispatch(
    input: &[f32],
    kernel: &[f32],
    bias: Option<&[f32]>,
    residual: Option<&[f32]>,
    out_cells: &mut [f32],
    n: usize,
    pos: usize,
    oc_base: usize,
    actual_in_channels: usize,
    out_channels: usize,
    ci_blocks: usize,
    in_n_stride: usize,
    in_cb_stride: usize,
    activation: Activation,
) {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if std::is_x86_feature_detected!("avx") && std::is_x86_feature_detected!("fma") {
        #[allow(unsafe_code)]
        unsafe {
            nchwc_pointwise_block8x4_avx_fma(
                input,
                kernel,
                bias,
                residual,
                out_cells,
                n,
                pos,
                oc_base,
                actual_in_channels,
                out_channels,
                ci_blocks,
                in_n_stride,
                in_cb_stride,
                activation,
            );
        }
        return;
    }
    #[cfg(target_arch = "aarch64")]
    if std::arch::is_aarch64_feature_detected!("neon") {
        #[allow(unsafe_code)]
        unsafe {
            nchwc_pointwise_block8x4_neon(
                input,
                kernel,
                bias,
                residual,
                out_cells,
                n,
                pos,
                oc_base,
                actual_in_channels,
                out_channels,
                ci_blocks,
                in_n_stride,
                in_cb_stride,
                activation,
            );
        }
        return;
    }
    for local in 0..4 {
        nchwc_pointwise_tail_block_scalar(
            input,
            kernel,
            bias,
            residual.map(|r| &r[local * 8..(local + 1) * 8]),
            &mut out_cells[local * 8..(local + 1) * 8],
            n,
            pos + local,
            oc_base,
            actual_in_channels,
            out_channels,
            ci_blocks,
            in_n_stride,
            in_cb_stride,
            8,
            activation,
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn nchwc_pointwise_tail_block_scalar(
    input: &[f32],
    kernel: &[f32],
    bias: Option<&[f32]>,
    residual: Option<&[f32]>,
    out_cell: &mut [f32],
    n: usize,
    pos: usize,
    oc_base: usize,
    actual_in_channels: usize,
    out_channels: usize,
    _ci_blocks: usize,
    in_n_stride: usize,
    in_cb_stride: usize,
    block: usize,
    activation: Activation,
) {
    for lane in 0..block {
        let oc = oc_base + lane;
        let mut acc = if oc < out_channels {
            bias.map_or(0.0, |b| b[oc])
        } else {
            0.0
        };
        if oc < out_channels {
            for ic in 0..actual_in_channels {
                let icb = ic / block;
                let ilane = ic % block;
                let input_idx = n * in_n_stride + icb * in_cb_stride + pos * block + ilane;
                acc += input[input_idx] * kernel[ic * out_channels + oc];
            }
            if let Some(skip) = residual {
                acc += skip[lane];
            }
            out_cell[lane] = apply_conv_activation_scalar(acc, activation);
        } else {
            out_cell[lane] = 0.0;
        }
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx,fma")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn, clippy::too_many_arguments)]
unsafe fn nchwc_pointwise_block8x4_avx_fma(
    input: &[f32],
    kernel: &[f32],
    bias: Option<&[f32]>,
    residual: Option<&[f32]>,
    out_cells: &mut [f32],
    n: usize,
    pos: usize,
    oc_base: usize,
    actual_in_channels: usize,
    out_channels: usize,
    _ci_blocks: usize,
    in_n_stride: usize,
    in_cb_stride: usize,
    activation: Activation,
) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let bias_v = if let Some(b) = bias {
        _mm256_loadu_ps(b.as_ptr().add(oc_base))
    } else {
        _mm256_setzero_ps()
    };
    let mut acc0 = bias_v;
    let mut acc1 = bias_v;
    let mut acc2 = bias_v;
    let mut acc3 = bias_v;
    for ic in 0..actual_in_channels {
        let icb = ic / 8;
        let ilane = ic & 7;
        let base = n * in_n_stride + icb * in_cb_stride + pos * 8 + ilane;
        let w = _mm256_loadu_ps(kernel.as_ptr().add(ic * out_channels + oc_base));
        acc0 = _mm256_fmadd_ps(_mm256_set1_ps(*input.get_unchecked(base)), w, acc0);
        acc1 = _mm256_fmadd_ps(_mm256_set1_ps(*input.get_unchecked(base + 8)), w, acc1);
        acc2 = _mm256_fmadd_ps(_mm256_set1_ps(*input.get_unchecked(base + 16)), w, acc2);
        acc3 = _mm256_fmadd_ps(_mm256_set1_ps(*input.get_unchecked(base + 24)), w, acc3);
    }
    if let Some(skip) = residual {
        acc0 = _mm256_add_ps(acc0, _mm256_loadu_ps(skip.as_ptr()));
        acc1 = _mm256_add_ps(acc1, _mm256_loadu_ps(skip.as_ptr().add(8)));
        acc2 = _mm256_add_ps(acc2, _mm256_loadu_ps(skip.as_ptr().add(16)));
        acc3 = _mm256_add_ps(acc3, _mm256_loadu_ps(skip.as_ptr().add(24)));
    }
    if matches!(activation, Activation::Relu) {
        let z = _mm256_setzero_ps();
        acc0 = _mm256_max_ps(acc0, z);
        acc1 = _mm256_max_ps(acc1, z);
        acc2 = _mm256_max_ps(acc2, z);
        acc3 = _mm256_max_ps(acc3, z);
    }
    _mm256_storeu_ps(out_cells.as_mut_ptr(), acc0);
    _mm256_storeu_ps(out_cells.as_mut_ptr().add(8), acc1);
    _mm256_storeu_ps(out_cells.as_mut_ptr().add(16), acc2);
    _mm256_storeu_ps(out_cells.as_mut_ptr().add(24), acc3);
    if matches!(activation, Activation::Silu) {
        for v in out_cells.iter_mut().take(32) {
            *v = apply_conv_activation_scalar(*v, Activation::Silu);
        }
    }
}

/// Strip kernel: processes a contiguous spatial range `[pos_start, pos_end)`
/// for one (n, ocb) chunk. Internal 8-pixel + 4-pixel + 1-pixel tiling
/// amortises the `#[target_feature]` no-inline cost across all tiles for
/// the same ocb. Replaces the per-tile dispatch in the M2 wiring.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn, clippy::too_many_arguments)]
unsafe fn nchwc_pointwise_block16_strip_avx512(
    input: &[f32],
    kernel: &[f32],
    bias: Option<&[f32]>,
    residual: Option<&[f32]>,
    out_chunk: &mut [f32],
    n: usize,
    pos_end: usize,
    oc_base: usize,
    actual_in_channels: usize,
    out_channels: usize,
    ci_blocks: usize,
    in_n_stride: usize,
    in_cb_stride: usize,
    activation: Activation,
) {
    let block = 16usize;
    let mut pos = 0usize;
    while pos + 8 <= pos_end {
        nchwc_pointwise_block16x8_avx512(
            input,
            kernel,
            bias,
            residual.map(|r| &r[pos * block..(pos + 8) * block]),
            &mut out_chunk[pos * block..(pos + 8) * block],
            n,
            pos,
            oc_base,
            actual_in_channels,
            out_channels,
            ci_blocks,
            in_n_stride,
            in_cb_stride,
            activation,
        );
        pos += 8;
    }
    while pos + 4 <= pos_end {
        nchwc_pointwise_block16x4_avx512(
            input,
            kernel,
            bias,
            residual.map(|r| &r[pos * block..(pos + 4) * block]),
            &mut out_chunk[pos * block..(pos + 4) * block],
            n,
            pos,
            oc_base,
            actual_in_channels,
            out_channels,
            ci_blocks,
            in_n_stride,
            in_cb_stride,
            activation,
        );
        pos += 4;
    }
    while pos < pos_end {
        let out_cell = &mut out_chunk[pos * block..(pos + 1) * block];
        let residual_cell = residual.map(|r| &r[pos * block..(pos + 1) * block]);
        nchwc_pointwise_full_block_dispatch(
            input,
            kernel,
            bias,
            residual_cell,
            out_cell,
            n,
            pos,
            oc_base,
            actual_in_channels,
            out_channels,
            ci_blocks,
            in_n_stride,
            in_cb_stride,
            block,
            activation,
        );
        pos += 1;
    }
}

/// AVX-512 8-pixel × 16-channel NCHWc pointwise kernel.
///
/// Pattern: 8 acc ZMMs (1 per pixel, 16 lanes = full oc-block). Per
/// `ic`: 1 weight ZMM load + 8 input-scalar broadcasts + 8 FMAs. The
/// 8-wide pixel tile amortises weight loads 8× and saturates the 2-FMA/
/// cycle Zen 4 pipe (8 FMAs / 2 = 4 cyc per `ic`). For K=320:
/// 320 × 4 = 1280 cycles per 8-pixel tile ≈ 285 ns at 4.5 GHz.
///
/// Register budget: 8 acc + 8 input bcast + 1 weight + 1 bias = 18 ZMMs.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn, clippy::too_many_arguments)]
unsafe fn nchwc_pointwise_block16x8_avx512(
    input: &[f32],
    kernel: &[f32],
    bias: Option<&[f32]>,
    residual: Option<&[f32]>,
    out_cells: &mut [f32],
    n: usize,
    pos: usize,
    oc_base: usize,
    actual_in_channels: usize,
    out_channels: usize,
    _ci_blocks: usize,
    in_n_stride: usize,
    in_cb_stride: usize,
    activation: Activation,
) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    const B: usize = 16;
    let bias_v = if let Some(b) = bias {
        _mm512_loadu_ps(b.as_ptr().add(oc_base))
    } else {
        _mm512_setzero_ps()
    };
    let mut acc0 = bias_v;
    let mut acc1 = bias_v;
    let mut acc2 = bias_v;
    let mut acc3 = bias_v;
    let mut acc4 = bias_v;
    let mut acc5 = bias_v;
    let mut acc6 = bias_v;
    let mut acc7 = bias_v;
    let in_ptr = input.as_ptr();
    let k_ptr = kernel.as_ptr();
    for ic in 0..actual_in_channels {
        let icb = ic / B;
        let ilane = ic & (B - 1);
        let base = n * in_n_stride + icb * in_cb_stride + pos * B + ilane;
        let w = _mm512_loadu_ps(k_ptr.add(ic * out_channels + oc_base));
        acc0 = _mm512_fmadd_ps(_mm512_set1_ps(*in_ptr.add(base)), w, acc0);
        acc1 = _mm512_fmadd_ps(_mm512_set1_ps(*in_ptr.add(base + B)), w, acc1);
        acc2 = _mm512_fmadd_ps(_mm512_set1_ps(*in_ptr.add(base + 2 * B)), w, acc2);
        acc3 = _mm512_fmadd_ps(_mm512_set1_ps(*in_ptr.add(base + 3 * B)), w, acc3);
        acc4 = _mm512_fmadd_ps(_mm512_set1_ps(*in_ptr.add(base + 4 * B)), w, acc4);
        acc5 = _mm512_fmadd_ps(_mm512_set1_ps(*in_ptr.add(base + 5 * B)), w, acc5);
        acc6 = _mm512_fmadd_ps(_mm512_set1_ps(*in_ptr.add(base + 6 * B)), w, acc6);
        acc7 = _mm512_fmadd_ps(_mm512_set1_ps(*in_ptr.add(base + 7 * B)), w, acc7);
    }
    if let Some(skip) = residual {
        let sp = skip.as_ptr();
        acc0 = _mm512_add_ps(acc0, _mm512_loadu_ps(sp));
        acc1 = _mm512_add_ps(acc1, _mm512_loadu_ps(sp.add(B)));
        acc2 = _mm512_add_ps(acc2, _mm512_loadu_ps(sp.add(2 * B)));
        acc3 = _mm512_add_ps(acc3, _mm512_loadu_ps(sp.add(3 * B)));
        acc4 = _mm512_add_ps(acc4, _mm512_loadu_ps(sp.add(4 * B)));
        acc5 = _mm512_add_ps(acc5, _mm512_loadu_ps(sp.add(5 * B)));
        acc6 = _mm512_add_ps(acc6, _mm512_loadu_ps(sp.add(6 * B)));
        acc7 = _mm512_add_ps(acc7, _mm512_loadu_ps(sp.add(7 * B)));
    }
    if matches!(activation, Activation::Relu) {
        let z = _mm512_setzero_ps();
        acc0 = _mm512_max_ps(acc0, z);
        acc1 = _mm512_max_ps(acc1, z);
        acc2 = _mm512_max_ps(acc2, z);
        acc3 = _mm512_max_ps(acc3, z);
        acc4 = _mm512_max_ps(acc4, z);
        acc5 = _mm512_max_ps(acc5, z);
        acc6 = _mm512_max_ps(acc6, z);
        acc7 = _mm512_max_ps(acc7, z);
    }
    let op = out_cells.as_mut_ptr();
    _mm512_storeu_ps(op, acc0);
    _mm512_storeu_ps(op.add(B), acc1);
    _mm512_storeu_ps(op.add(2 * B), acc2);
    _mm512_storeu_ps(op.add(3 * B), acc3);
    _mm512_storeu_ps(op.add(4 * B), acc4);
    _mm512_storeu_ps(op.add(5 * B), acc5);
    _mm512_storeu_ps(op.add(6 * B), acc6);
    _mm512_storeu_ps(op.add(7 * B), acc7);
    if matches!(activation, Activation::Silu) {
        for v in out_cells.iter_mut().take(8 * B) {
            *v = apply_conv_activation_scalar(*v, Activation::Silu);
        }
    }
}

/// AVX-512 4-pixel × 16-channel NCHWc pointwise kernel. Tail kernel for
/// when 8-pixel tiles don't divide the remaining spatial range.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn, clippy::too_many_arguments)]
unsafe fn nchwc_pointwise_block16x4_avx512(
    input: &[f32],
    kernel: &[f32],
    bias: Option<&[f32]>,
    residual: Option<&[f32]>,
    out_cells: &mut [f32],
    n: usize,
    pos: usize,
    oc_base: usize,
    actual_in_channels: usize,
    out_channels: usize,
    _ci_blocks: usize,
    in_n_stride: usize,
    in_cb_stride: usize,
    activation: Activation,
) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    const B: usize = 16;
    let bias_v = if let Some(b) = bias {
        _mm512_loadu_ps(b.as_ptr().add(oc_base))
    } else {
        _mm512_setzero_ps()
    };
    let mut acc0 = bias_v;
    let mut acc1 = bias_v;
    let mut acc2 = bias_v;
    let mut acc3 = bias_v;
    let in_ptr = input.as_ptr();
    let k_ptr = kernel.as_ptr();
    for ic in 0..actual_in_channels {
        let icb = ic / B;
        let ilane = ic & (B - 1);
        let base = n * in_n_stride + icb * in_cb_stride + pos * B + ilane;
        let w = _mm512_loadu_ps(k_ptr.add(ic * out_channels + oc_base));
        let a0 = _mm512_set1_ps(*in_ptr.add(base));
        let a1 = _mm512_set1_ps(*in_ptr.add(base + B));
        let a2 = _mm512_set1_ps(*in_ptr.add(base + 2 * B));
        let a3 = _mm512_set1_ps(*in_ptr.add(base + 3 * B));
        acc0 = _mm512_fmadd_ps(a0, w, acc0);
        acc1 = _mm512_fmadd_ps(a1, w, acc1);
        acc2 = _mm512_fmadd_ps(a2, w, acc2);
        acc3 = _mm512_fmadd_ps(a3, w, acc3);
    }
    if let Some(skip) = residual {
        let sp = skip.as_ptr();
        acc0 = _mm512_add_ps(acc0, _mm512_loadu_ps(sp));
        acc1 = _mm512_add_ps(acc1, _mm512_loadu_ps(sp.add(B)));
        acc2 = _mm512_add_ps(acc2, _mm512_loadu_ps(sp.add(2 * B)));
        acc3 = _mm512_add_ps(acc3, _mm512_loadu_ps(sp.add(3 * B)));
    }
    if matches!(activation, Activation::Relu) {
        let z = _mm512_setzero_ps();
        acc0 = _mm512_max_ps(acc0, z);
        acc1 = _mm512_max_ps(acc1, z);
        acc2 = _mm512_max_ps(acc2, z);
        acc3 = _mm512_max_ps(acc3, z);
    }
    let op = out_cells.as_mut_ptr();
    _mm512_storeu_ps(op, acc0);
    _mm512_storeu_ps(op.add(B), acc1);
    _mm512_storeu_ps(op.add(2 * B), acc2);
    _mm512_storeu_ps(op.add(3 * B), acc3);
    if matches!(activation, Activation::Silu) {
        for v in out_cells.iter_mut().take(4 * B) {
            *v = apply_conv_activation_scalar(*v, Activation::Silu);
        }
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx,fma")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn, clippy::too_many_arguments)]
unsafe fn nchwc_pointwise_block8_avx_fma(
    input: &[f32],
    kernel: &[f32],
    bias: Option<&[f32]>,
    residual: Option<&[f32]>,
    out_cell: &mut [f32],
    n: usize,
    pos: usize,
    oc_base: usize,
    actual_in_channels: usize,
    out_channels: usize,
    _ci_blocks: usize,
    in_n_stride: usize,
    in_cb_stride: usize,
    activation: Activation,
) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let mut acc = if let Some(b) = bias {
        _mm256_loadu_ps(b.as_ptr().add(oc_base))
    } else {
        _mm256_setzero_ps()
    };
    for ic in 0..actual_in_channels {
        let icb = ic / 8;
        let ilane = ic & 7;
        let input_idx = n * in_n_stride + icb * in_cb_stride + pos * 8 + ilane;
        let a = _mm256_set1_ps(*input.get_unchecked(input_idx));
        let w = _mm256_loadu_ps(kernel.as_ptr().add(ic * out_channels + oc_base));
        acc = _mm256_fmadd_ps(a, w, acc);
    }
    if let Some(skip) = residual {
        acc = _mm256_add_ps(acc, _mm256_loadu_ps(skip.as_ptr()));
    }
    if matches!(activation, Activation::Relu) {
        acc = _mm256_max_ps(acc, _mm256_setzero_ps());
    }
    _mm256_storeu_ps(out_cell.as_mut_ptr(), acc);
    if matches!(activation, Activation::Silu) {
        for v in out_cell.iter_mut().take(8) {
            *v = apply_conv_activation_scalar(*v, Activation::Silu);
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn, clippy::too_many_arguments)]
unsafe fn nchwc_pointwise_block8x4_neon(
    input: &[f32],
    kernel: &[f32],
    bias: Option<&[f32]>,
    residual: Option<&[f32]>,
    out_cells: &mut [f32],
    n: usize,
    pos: usize,
    oc_base: usize,
    actual_in_channels: usize,
    out_channels: usize,
    _ci_blocks: usize,
    in_n_stride: usize,
    in_cb_stride: usize,
    activation: Activation,
) {
    use std::arch::aarch64::*;
    let (bias0, bias1) = if let Some(b) = bias {
        (
            vld1q_f32(b.as_ptr().add(oc_base)),
            vld1q_f32(b.as_ptr().add(oc_base + 4)),
        )
    } else {
        (vdupq_n_f32(0.0), vdupq_n_f32(0.0))
    };
    let mut a00 = bias0;
    let mut a01 = bias1;
    let mut a10 = bias0;
    let mut a11 = bias1;
    let mut a20 = bias0;
    let mut a21 = bias1;
    let mut a30 = bias0;
    let mut a31 = bias1;
    for ic in 0..actual_in_channels {
        let icb = ic / 8;
        let ilane = ic & 7;
        let base = n * in_n_stride + icb * in_cb_stride + pos * 8 + ilane;
        let w0 = vld1q_f32(kernel.as_ptr().add(ic * out_channels + oc_base));
        let w1 = vld1q_f32(kernel.as_ptr().add(ic * out_channels + oc_base + 4));
        let x0 = vdupq_n_f32(*input.get_unchecked(base));
        let x1 = vdupq_n_f32(*input.get_unchecked(base + 8));
        let x2 = vdupq_n_f32(*input.get_unchecked(base + 16));
        let x3 = vdupq_n_f32(*input.get_unchecked(base + 24));
        a00 = vfmaq_f32(a00, x0, w0);
        a01 = vfmaq_f32(a01, x0, w1);
        a10 = vfmaq_f32(a10, x1, w0);
        a11 = vfmaq_f32(a11, x1, w1);
        a20 = vfmaq_f32(a20, x2, w0);
        a21 = vfmaq_f32(a21, x2, w1);
        a30 = vfmaq_f32(a30, x3, w0);
        a31 = vfmaq_f32(a31, x3, w1);
    }
    if let Some(skip) = residual {
        a00 = vaddq_f32(a00, vld1q_f32(skip.as_ptr()));
        a01 = vaddq_f32(a01, vld1q_f32(skip.as_ptr().add(4)));
        a10 = vaddq_f32(a10, vld1q_f32(skip.as_ptr().add(8)));
        a11 = vaddq_f32(a11, vld1q_f32(skip.as_ptr().add(12)));
        a20 = vaddq_f32(a20, vld1q_f32(skip.as_ptr().add(16)));
        a21 = vaddq_f32(a21, vld1q_f32(skip.as_ptr().add(20)));
        a30 = vaddq_f32(a30, vld1q_f32(skip.as_ptr().add(24)));
        a31 = vaddq_f32(a31, vld1q_f32(skip.as_ptr().add(28)));
    }
    if matches!(activation, Activation::Relu) {
        let z = vdupq_n_f32(0.0);
        a00 = vmaxq_f32(a00, z);
        a01 = vmaxq_f32(a01, z);
        a10 = vmaxq_f32(a10, z);
        a11 = vmaxq_f32(a11, z);
        a20 = vmaxq_f32(a20, z);
        a21 = vmaxq_f32(a21, z);
        a30 = vmaxq_f32(a30, z);
        a31 = vmaxq_f32(a31, z);
    }
    vst1q_f32(out_cells.as_mut_ptr(), a00);
    vst1q_f32(out_cells.as_mut_ptr().add(4), a01);
    vst1q_f32(out_cells.as_mut_ptr().add(8), a10);
    vst1q_f32(out_cells.as_mut_ptr().add(12), a11);
    vst1q_f32(out_cells.as_mut_ptr().add(16), a20);
    vst1q_f32(out_cells.as_mut_ptr().add(20), a21);
    vst1q_f32(out_cells.as_mut_ptr().add(24), a30);
    vst1q_f32(out_cells.as_mut_ptr().add(28), a31);
    if matches!(activation, Activation::Silu) {
        for v in out_cells.iter_mut().take(32) {
            *v = apply_conv_activation_scalar(*v, Activation::Silu);
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn, clippy::too_many_arguments)]
unsafe fn nchwc_pointwise_block8_neon(
    input: &[f32],
    kernel: &[f32],
    bias: Option<&[f32]>,
    residual: Option<&[f32]>,
    out_cell: &mut [f32],
    n: usize,
    pos: usize,
    oc_base: usize,
    actual_in_channels: usize,
    out_channels: usize,
    _ci_blocks: usize,
    in_n_stride: usize,
    in_cb_stride: usize,
    activation: Activation,
) {
    use std::arch::aarch64::*;
    let mut acc0;
    let mut acc1;
    if let Some(b) = bias {
        acc0 = vld1q_f32(b.as_ptr().add(oc_base));
        acc1 = vld1q_f32(b.as_ptr().add(oc_base + 4));
    } else {
        acc0 = vdupq_n_f32(0.0);
        acc1 = vdupq_n_f32(0.0);
    }
    for ic in 0..actual_in_channels {
        let icb = ic / 8;
        let ilane = ic & 7;
        let input_idx = n * in_n_stride + icb * in_cb_stride + pos * 8 + ilane;
        let a = vdupq_n_f32(*input.get_unchecked(input_idx));
        acc0 = vfmaq_f32(
            acc0,
            a,
            vld1q_f32(kernel.as_ptr().add(ic * out_channels + oc_base)),
        );
        acc1 = vfmaq_f32(
            acc1,
            a,
            vld1q_f32(kernel.as_ptr().add(ic * out_channels + oc_base + 4)),
        );
    }
    if let Some(skip) = residual {
        acc0 = vaddq_f32(acc0, vld1q_f32(skip.as_ptr()));
        acc1 = vaddq_f32(acc1, vld1q_f32(skip.as_ptr().add(4)));
    }
    if matches!(activation, Activation::Relu) {
        let zero = vdupq_n_f32(0.0);
        acc0 = vmaxq_f32(acc0, zero);
        acc1 = vmaxq_f32(acc1, zero);
    }
    vst1q_f32(out_cell.as_mut_ptr(), acc0);
    vst1q_f32(out_cell.as_mut_ptr().add(4), acc1);
    if matches!(activation, Activation::Silu) {
        for v in out_cell.iter_mut().take(8) {
            *v = apply_conv_activation_scalar(*v, Activation::Silu);
        }
    }
}

/// 1×1 Conv over an NCHWc-laid-out input, producing an NCHWc output.
///
/// This is the native blocked 1×1 path: it reads NCHWc directly,
/// broadcasts each input lane, accumulates a full output-channel block in
/// registers, optionally adds an NCHWc residual, then applies activation.
///
/// - `input` must carry `Layout::NCHWc { block }` and shape
///   `[N, C_in/block, H, W, block]`.
/// - `kernel` is the ordinary `[1, 1, C_in, C_out]` NHWC 1×1 kernel as
///   used by the NHWC path — no reblocking.
/// - `actual_in_channels` is the real (non-padded) in-channel count. NCHWc
///   storage rounds C up to `(CI/block)*block`; we need the logical
///   channel count to strip padding on conversion.
/// - `block` is recovered from `input.shape()[4]`.
///
/// Output is tagged `Layout::NCHWc { block }` with shape
/// `[N, C_out/block, H, W, block]`, padded C_out-side where needed.
#[allow(clippy::too_many_arguments)]
fn conv2d_nchwc_pointwise_with_activation_impl(
    input: &Tensor,
    kernel: &Tensor,
    bias: Option<&Tensor>,
    residual: Option<&Tensor>,
    actual_in_channels: usize,
    activation: Activation,
    config: ParallelElementwiseConfig,
    _thread_pool: Option<&ThreadPool>,
    prepacked_b: Option<&super::super::matmul::PackedB>,
) -> Result<Tensor, KernelError> {
    let _ = config;
    let _ = prepacked_b;
    let in_shape = input.shape();
    if in_shape.len() != 5 {
        return Err(KernelError::LayoutConversion(format!(
            "conv2d_nchwc_pointwise: expected 5-D NCHWc input, got {in_shape:?}"
        )));
    }
    let block = in_shape[4];
    if block == 0 || block > u8::MAX as usize {
        return Err(KernelError::LayoutConversion(format!(
            "conv2d_nchwc_pointwise: invalid block size {block}"
        )));
    }
    let out_channels = match kernel.shape() {
        [1, 1, _, oc] => *oc,
        s => {
            return Err(KernelError::LayoutConversion(format!(
                "conv2d_nchwc_pointwise: expected 1×1 NHWC kernel [1,1,Ci,Co], got {s:?}"
            )));
        }
    };
    let [n_batch, ci_blocks, h, w, _] = *<&[usize; 5]>::try_from(in_shape)
        .map_err(|_| KernelError::LayoutConversion("pointwise nchwc: rank mismatch".into()))?;
    if actual_in_channels > ci_blocks * block {
        return Err(KernelError::LayoutConversion(format!(
            "pointwise nchwc: actual_in_channels {actual_in_channels} > capacity {}",
            ci_blocks * block
        )));
    }
    let kernel_in_channels = kernel.shape()[2];
    if kernel_in_channels != actual_in_channels {
        return Err(KernelError::LayoutConversion(format!(
            "pointwise nchwc: kernel Cin {kernel_in_channels} != actual_in_channels {actual_in_channels}"
        )));
    }
    if let Some(b) = bias {
        let blen: usize = b.shape().iter().product();
        if blen != out_channels {
            return Err(KernelError::ConvBiasShapeMismatch {
                bias_shape: b.shape().to_vec(),
                out_channels,
            });
        }
    }

    let co_blocks = out_channels.div_ceil(block);
    if let Some(skip) = residual {
        let expected = [n_batch, co_blocks, h, w, block];
        if skip.shape() != expected {
            return Err(KernelError::LayoutConversion(format!(
                "pointwise nchwc residual: expected {:?}, got {:?}",
                expected,
                skip.shape()
            )));
        }
    }
    let out_len = n_batch * co_blocks * h * w * block;
    // NCHWc PW kernel writes every output cell (GEMM + scatter), so the
    // zero-init in calloc is wasted work and bypasses the thread-local
    // alloc_cache. Switch to uninitialized for hot-path reuse. The tail
    // case (co_blocks * block > out_channels) is handled inside the
    // scatter: `flat_to_nchwc_fused` always writes c_take = min(block,
    // remaining) lanes per (cb, pos), and tail lanes (cb_used < cb) only
    // exist when out_channels is not a multiple of block. For the
    // tracker, all PW out_channels are multiples of 16; the assert
    // below guards against accidental tail breakage on other models.
    let mut output = if out_channels.is_multiple_of(block) {
        AlignedVec::<f32>::uninitialized(out_len)
    } else {
        AlignedVec::<f32>::calloc(out_len)
    };
    let input_data = input.data();
    let kernel_data = kernel.data();
    let bias_data = bias.map(Tensor::data);
    let residual_data = residual.map(Tensor::data);
    let out_cb_stride = h * w * block;
    let in_n_stride = ci_blocks * h * w * block;
    let in_cb_stride = h * w * block;
    let spatial = h * w;

    // Direct AVX-512 4-pixel × 16-channel NCHWc PW path. Bypasses the blocked
    // GEMM machinery (pack-A + GEBP + flat scratch + scatter) for block=16
    // shapes with full oc-block coverage: one 4-pixel tile per inner call with
    // 4 acc ZMMs + per-`ic` weight broadcast — the MR=4 × NR=block pattern of
    // MLAS's `MlasConvPointwiseKernel`. Gated by `YSCV_NCHWC_PW_DIRECT_OFF=1`.
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if !cfg!(miri)
        && block == 16
        && out_channels.is_multiple_of(block)
        && std::is_x86_feature_detected!("avx512f")
        && !nchwc_pw_direct_disabled()
    {
        let out_slice = output.as_mut_slice();
        super::super::super::scope_ctx::par_chunks_mut_dispatch(
            out_slice,
            out_cb_stride,
            |unit_idx, out_chunk| {
                let n = unit_idx / co_blocks;
                let ocb = unit_idx % co_blocks;
                let oc_base = ocb * block;
                let residual_chunk = residual_data
                    .map(|r| &r[unit_idx * out_cb_stride..(unit_idx + 1) * out_cb_stride]);
                // SAFETY: avx512f detected at runtime above; slice bounds
                // verified by NCHWc layout (caller-validated shapes).
                // Single function-call boundary across all spatial tiles for
                // this (n, ocb) chunk amortises the `#[target_feature]`
                // no-inline overhead.
                #[allow(unsafe_code)]
                unsafe {
                    nchwc_pointwise_block16_strip_avx512(
                        input_data,
                        kernel_data,
                        bias_data,
                        residual_chunk,
                        out_chunk,
                        n,
                        spatial,
                        oc_base,
                        actual_in_channels,
                        out_channels,
                        ci_blocks,
                        in_n_stride,
                        in_cb_stride,
                        activation,
                    );
                }
            },
        );
        let out_tensor = Tensor::from_aligned(vec![n_batch, co_blocks, h, w, block], output)
            .map_err(KernelError::from)?;
        return Ok(out_tensor.with_layout(yscv_tensor::Layout::NCHWc { block: block as u8 }));
    }

    // K.2: native NCHWc PW path — all shapes, vectorized output conversion.
    if !cfg!(miri) && super::super::nchwc_pointwise::nchwc_pw_native_enabled() {
        super::super::nchwc_pointwise::nchwc_pw_compute(
            input_data,
            kernel_data,
            bias_data,
            residual_data,
            output.as_mut_slice(),
            n_batch,
            ci_blocks,
            h,
            w,
            actual_in_channels,
            out_channels,
            co_blocks,
            block,
            activation,
            prepacked_b,
        );
        let out_tensor = Tensor::from_aligned(vec![n_batch, co_blocks, h, w, block], output)
            .map_err(KernelError::from)?;
        return Ok(out_tensor.with_layout(yscv_tensor::Layout::NCHWc { block: block as u8 }));
    }

    // Legacy path (YSCV_NCHWC_PW_LEGACY=1): blocked GEMM with scalar flat→NCHWc
    // conversion + per-pixel fallback for small shapes.
    let use_gemm = !cfg!(miri) && spatial >= 32 && actual_in_channels >= 32 && out_channels >= 16;

    if use_gemm {
        // kernel is [1, 1, Ci, Co] flattened = [Ci * Co]. Treat as [K=Ci, N=Co].
        let out_slice = output.as_mut_slice();
        // flat_buf: [n_batch, spatial, out_channels] scratch for GEMM output.
        let mut flat_buf = AlignedVec::<f32>::uninitialized(n_batch * spatial * out_channels);
        for n_idx in 0..n_batch {
            let in_slice = &input_data[n_idx * in_n_stride..(n_idx + 1) * in_n_stride];
            let flat_out = &mut flat_buf.as_mut_slice()
                [n_idx * spatial * out_channels..(n_idx + 1) * spatial * out_channels];
            super::super::matmul::blocked_gemm_nchwc_a_parallel(
                in_slice,
                spatial,
                block,
                actual_in_channels,
                kernel_data,
                flat_out,
                out_channels,
                super::super::matmul::GemmEpilogue::IDENTITY,
                _thread_pool,
                prepacked_b,
            );
        }
        // Convert flat [n, spatial, out_channels] → NCHWc [n, co_blocks, h*w, block]
        // while fusing bias + residual + activation in one pass.
        let flat_slice = flat_buf.as_slice();
        for n_idx in 0..n_batch {
            let flat_n =
                &flat_slice[n_idx * spatial * out_channels..(n_idx + 1) * spatial * out_channels];
            let out_n = &mut out_slice
                [n_idx * co_blocks * out_cb_stride..(n_idx + 1) * co_blocks * out_cb_stride];
            let res_n = residual_data.map(|r| {
                &r[n_idx * co_blocks * out_cb_stride..(n_idx + 1) * co_blocks * out_cb_stride]
            });
            for pos in 0..spatial {
                let flat_row = pos * out_channels;
                for ocb in 0..co_blocks {
                    let c_start = ocb * block;
                    let c_take = block.min(out_channels - c_start);
                    let nchwc_off = ocb * out_cb_stride + pos * block;
                    for lane in 0..c_take {
                        let mut val = flat_n[flat_row + c_start + lane];
                        if let Some(b) = bias_data {
                            val += b[c_start + lane];
                        }
                        if let Some(r) = res_n {
                            val += r[nchwc_off + lane];
                        }
                        out_n[nchwc_off + lane] = apply_conv_activation_scalar(val, activation);
                    }
                }
            }
        }
    } else {
        super::super::super::scope_ctx::par_chunks_mut_dispatch(
            output.as_mut_slice(),
            out_cb_stride,
            |unit_idx, out_chunk| {
                let n = unit_idx / co_blocks;
                let ocb = unit_idx % co_blocks;
                let oc_base = ocb * block;
                let residual_chunk = residual_data
                    .map(|r| &r[unit_idx * out_cb_stride..(unit_idx + 1) * out_cb_stride]);
                let full_oc_block = oc_base + block <= out_channels;
                let mut pos = 0usize;
                if full_oc_block && block == 8 {
                    while pos + 4 <= spatial {
                        nchwc_pointwise_full_block4_dispatch(
                            input_data,
                            kernel_data,
                            bias_data,
                            residual_chunk.map(|r| &r[pos * block..(pos + 4) * block]),
                            &mut out_chunk[pos * block..(pos + 4) * block],
                            n,
                            pos,
                            oc_base,
                            actual_in_channels,
                            out_channels,
                            ci_blocks,
                            in_n_stride,
                            in_cb_stride,
                            activation,
                        );
                        pos += 4;
                    }
                }
                while pos < spatial {
                    let out_cell = &mut out_chunk[pos * block..(pos + 1) * block];
                    let residual_cell = residual_chunk.map(|r| &r[pos * block..(pos + 1) * block]);
                    if full_oc_block {
                        nchwc_pointwise_full_block_dispatch(
                            input_data,
                            kernel_data,
                            bias_data,
                            residual_cell,
                            out_cell,
                            n,
                            pos,
                            oc_base,
                            actual_in_channels,
                            out_channels,
                            ci_blocks,
                            in_n_stride,
                            in_cb_stride,
                            block,
                            activation,
                        );
                    } else {
                        nchwc_pointwise_tail_block_scalar(
                            input_data,
                            kernel_data,
                            bias_data,
                            residual_cell,
                            out_cell,
                            n,
                            pos,
                            oc_base,
                            actual_in_channels,
                            out_channels,
                            ci_blocks,
                            in_n_stride,
                            in_cb_stride,
                            block,
                            activation,
                        );
                    }
                    pos += 1;
                }
            },
        );
    }

    let out_tensor = Tensor::from_aligned(vec![n_batch, co_blocks, h, w, block], output)
        .map_err(KernelError::from)?;
    Ok(out_tensor.with_layout(yscv_tensor::Layout::NCHWc { block: block as u8 }))
}

/// NCHWc pointwise (1x1) convolution with fused activation using pre-packed weights.
#[allow(clippy::too_many_arguments)]
pub fn conv2d_nchwc_pointwise_with_activation_prepacked(
    input: &Tensor,
    kernel: &Tensor,
    bias: Option<&Tensor>,
    actual_in_channels: usize,
    activation: Activation,
    config: ParallelElementwiseConfig,
    thread_pool: Option<&ThreadPool>,
    prepacked_b: Option<&super::super::matmul::PackedB>,
) -> Result<Tensor, KernelError> {
    conv2d_nchwc_pointwise_with_activation_impl(
        input,
        kernel,
        bias,
        None,
        actual_in_channels,
        activation,
        config,
        thread_pool,
        prepacked_b,
    )
}

/// NCHWc pointwise convolution with a fused residual add and activation, pre-packed weights.
#[allow(clippy::too_many_arguments)]
pub fn conv2d_nchwc_pointwise_with_residual_activation_prepacked(
    input: &Tensor,
    kernel: &Tensor,
    bias: Option<&Tensor>,
    residual: &Tensor,
    actual_in_channels: usize,
    activation: Activation,
    config: ParallelElementwiseConfig,
    thread_pool: Option<&ThreadPool>,
    prepacked_b: Option<&super::super::matmul::PackedB>,
) -> Result<Tensor, KernelError> {
    conv2d_nchwc_pointwise_with_activation_impl(
        input,
        kernel,
        bias,
        Some(residual),
        actual_in_channels,
        activation,
        config,
        thread_pool,
        prepacked_b,
    )
}
