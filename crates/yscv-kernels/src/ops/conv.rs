use rayon::ThreadPool;
use std::sync::OnceLock;
use yscv_tensor::{AlignedVec, Tensor, TensorError};

use super::super::error::KernelError;
use super::config::{
    Conv2dPlan, Conv2dSpec, DEFAULT_POINTWISE_CONV_MIN_PARALLEL_ELEMENTS,
    DEFAULT_POINTWISE_CONV_MIN_PARALLEL_FLOPS, DepthwiseConv2dPlan, DepthwiseConv2dSpec,
    ParallelElementwiseConfig, ParallelMatmulConfig, SeparableConv2dKernels, SeparableConv2dSpec,
    should_parallelize_len,
};

/// Step 1: cached pointwise Conv parallel-dispatch threshold (elems + flops).
/// Env overrides `YSCV_MIN_PAR_POINTWISE_CONV_ELEMS` / `_FLOPS` consulted
/// once per process. Tuned against Siamese tracker profile — raised from
/// hardcoded 4 096 to 16 384 elems / 1.5 MFlops to skip the rayon fork-join
/// path for tiny shapes where 5-8 µs dispatch overhead dwarfed compute.
/// Cached `YSCV_AVX512_DW_OFF` kill-switch (Session 10 polish). The DW
/// row dispatch fires once per DW output row — with tracker's 18 DW ops
/// and up to 128 spatial rows each, raw `std::env::var_os` reads were
/// showing up at 0.15% of cycles. Read once per process via OnceLock.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn avx512_dw_disabled() -> bool {
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| std::env::var_os("YSCV_AVX512_DW_OFF").is_some())
}

fn pointwise_conv_parallel_threshold() -> (usize, usize) {
    static CACHED: OnceLock<(usize, usize)> = OnceLock::new();
    *CACHED.get_or_init(|| {
        // ARM A53/A55 class cores benefit from entering parallel pointwise
        // slightly earlier than x86-tuned defaults. Keep host overrides
        // via env as top priority.
        #[cfg(target_arch = "aarch64")]
        let default_elems = DEFAULT_POINTWISE_CONV_MIN_PARALLEL_ELEMENTS / 2;
        #[cfg(not(target_arch = "aarch64"))]
        let default_elems = DEFAULT_POINTWISE_CONV_MIN_PARALLEL_ELEMENTS;
        #[cfg(target_arch = "aarch64")]
        let default_flops = DEFAULT_POINTWISE_CONV_MIN_PARALLEL_FLOPS / 2;
        #[cfg(not(target_arch = "aarch64"))]
        let default_flops = DEFAULT_POINTWISE_CONV_MIN_PARALLEL_FLOPS;

        let elems = std::env::var("YSCV_MIN_PAR_POINTWISE_CONV_ELEMS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&v| v > 0)
            .unwrap_or(default_elems);
        let flops = std::env::var("YSCV_MIN_PAR_POINTWISE_CONV_FLOPS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&v| v > 0)
            .unwrap_or(default_flops);
        (elems, flops)
    })
}

/// Build a `ParallelMatmulConfig` for pointwise Conv that respects the
/// Step 1 threshold. Gates on `work = 2*m*k*n` (FLOPs). When work is
/// below `min_flops`, the config forces sequential execution by setting
/// `min_parallel_output_elements = usize::MAX`. The `min_elems` env is
/// an optional secondary gate — only applied when compute-per-element
/// ratio is low (`2*k < 32`) i.e. the op is memory-bound rather than
/// compute-bound. Compute-bound small-output shapes (high-k, small m,
/// small n) still benefit from parallelism since per-thread work scales
/// with k — even at m*n=4 096 elems, 2*m*n*k=2 MFlops @ 6 threads means
/// 333 µs sequential → 55 µs parallel + 6 µs dispatch net −272 µs win.
fn pointwise_conv_matmul_config(m: usize, k: usize, n: usize) -> ParallelMatmulConfig {
    let (min_elems, min_flops) = pointwise_conv_parallel_threshold();
    let out_elems = m * n;
    // Use u64 to dodge 32-bit overflow on large shapes.
    let flops = 2u64
        .saturating_mul(m as u64)
        .saturating_mul(k as u64)
        .saturating_mul(n as u64);
    // Memory-bound regime: low-k (2k<32 ⇒ <4 FMAs per output lane), tiny
    // compute per byte — here output elems is the driver. Under `min_elems`
    // AND low-k → sequential. Otherwise flops gate alone decides.
    let memory_bound = 2 * k < 32;
    let force_sequential = flops < min_flops as u64 || (memory_bound && out_elems < min_elems);
    if force_sequential {
        ParallelMatmulConfig {
            min_parallel_shared_dim: usize::MAX,
            min_parallel_output_elements: usize::MAX,
        }
    } else {
        ParallelMatmulConfig {
            min_parallel_shared_dim: 1,
            min_parallel_output_elements: 4096,
        }
    }
}

/// Post-convolution fused activation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Activation {
    None,
    Relu,
    Silu,
}

/// Apply SiLU in-place on a mutable f32 slice.
#[inline]
pub(super) fn silu_slice_inplace(data: &mut [f32]) {
    super::simd::silu_inplace(data);
}

#[inline]
fn relu_slice_inplace(data: &mut [f32]) {
    super::simd::relu_slice_dispatch(data);
}

#[inline]
fn apply_conv_activation_inplace(data: &mut [f32], activation: Activation) {
    match activation {
        Activation::None => {}
        Activation::Relu => relu_slice_inplace(data),
        Activation::Silu => silu_slice_inplace(data),
    }
}

#[inline]
fn apply_conv_activation_scalar(x: f32, activation: Activation) -> f32 {
    match activation {
        Activation::None => x,
        Activation::Relu => x.max(0.0),
        Activation::Silu => x / (1.0 + (-x).exp()),
    }
}

#[inline]
fn direct_conv_work_threshold() -> usize {
    static THRESHOLD: OnceLock<usize> = OnceLock::new();
    *THRESHOLD.get_or_init(|| {
        if let Ok(v) = std::env::var("YSCV_DIRECT_CONV_WORK_MAX")
            && let Ok(parsed) = v.parse::<usize>()
            && parsed > 0
        {
            return parsed;
        }
        // Empirically on x86 Linux tracker workloads, direct 3x3 kernels start
        // losing to the generic path once total work grows past low hundreds
        // of thousands of MACs. Keep direct path for very small 3x3 only.
        //
        // On aarch64, the direct 3x3 kernel is single-threaded; under MT runs
        // we should hand off to GEMM/im2col earlier to leverage blocked/parallel
        // matmul wins landed in this arc.
        #[cfg(target_arch = "aarch64")]
        {
            if rayon::current_num_threads() <= 1 {
                300_000
            } else {
                120_000
            }
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            if rayon::current_num_threads() <= 1 {
                500_000
            } else {
                250_000
            }
        }
    })
}

#[inline]
fn use_true_fused_dw_pw_dm1(_plan: DepthwiseConv2dPlan, dw_activation: Activation) -> bool {
    static FORCE_ON: OnceLock<bool> = OnceLock::new();
    static AUTO_ON: OnceLock<bool> = OnceLock::new();
    static FORCE_OFF: OnceLock<bool> = OnceLock::new();
    let force_on =
        *FORCE_ON.get_or_init(|| std::env::var_os("YSCV_FUSED_DW_PW_TRUE_FUSED_ON").is_some());
    let auto_on =
        *AUTO_ON.get_or_init(|| std::env::var_os("YSCV_FUSED_DW_PW_TRUE_FUSED_AUTO").is_some());
    let force_off =
        *FORCE_OFF.get_or_init(|| std::env::var_os("YSCV_FUSED_DW_PW_TRUE_FUSED_OFF").is_some());
    if force_off {
        return false;
    }
    if force_on {
        return true;
    }
    if !auto_on {
        return false;
    }
    // SiLU on DW output is expensive in scalar form; keep the row-GEMM path
    // there until we land a vectorized direct epilogue.
    if matches!(dw_activation, Activation::Silu) {
        return false;
    }
    #[cfg(target_arch = "aarch64")]
    {
        // Prefer direct path when the per-row DW staging buffer is large
        // enough to pressure L1D on A53/A55-class cores.
        let row_bytes = _plan
            .out_w
            .saturating_mul(_plan.out_channels)
            .saturating_mul(std::mem::size_of::<f32>());
        row_bytes >= 16 * 1024 || _plan.out_channels >= 192
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        false
    }
}

pub fn conv2d_nhwc_with_config_and_pool(
    input: &Tensor,
    kernel: &Tensor,
    bias: Option<&Tensor>,
    spec: Conv2dSpec,
    config: ParallelElementwiseConfig,
    thread_pool: Option<&ThreadPool>,
) -> Result<Tensor, KernelError> {
    conv2d_nhwc_with_activation_with_config_and_pool(
        input,
        kernel,
        bias,
        spec,
        Activation::None,
        config,
        thread_pool,
    )
}

pub fn conv2d_nhwc_with_activation_with_config_and_pool(
    input: &Tensor,
    kernel: &Tensor,
    bias: Option<&Tensor>,
    spec: Conv2dSpec,
    activation: Activation,
    config: ParallelElementwiseConfig,
    thread_pool: Option<&ThreadPool>,
) -> Result<Tensor, KernelError> {
    conv2d_nhwc_with_activation_prepacked(
        input,
        kernel,
        bias,
        spec,
        activation,
        config,
        thread_pool,
        None,
    )
}

/// Same as `conv2d_nhwc_with_activation_with_config_and_pool` but accepts an
/// optional pre-packed kernel B (built at model load via
/// `pack_b_for_session`). When provided, the 1×1 pointwise GEMM path skips
/// both the fingerprint-cache lookup and the `pack_b_panel` work — this is
/// the fast path for static Conv weights.
///
/// For non-pointwise convs (3×3 direct, grouped, etc.) the prepacked handle
/// is unused today because those paths don't go through the blocked GEMM
/// microkernel. Passing `Some` is safe — it will be ignored silently for
/// those shape classes.
/// Pointwise 1×1 Conv with an optional residual tensor fused into the
/// GEMM epilogue. Produces `out = activation(conv(input, weight) + bias + residual)`
/// in a single memory pass over the output — the runner's `Conv+Add+Relu`
/// fusion used to do a separate `add_relu_inplace` pass, doubling the
/// output-side memory traffic. `residual` must match `output` shape.
///
/// Falls through to the non-residual pointwise path when `residual` is
/// `None`. Only valid for 1×1 stride-1 Conv; callers must pre-check.
#[allow(clippy::too_many_arguments)]
pub fn conv2d_nhwc_pointwise_with_residual_relu(
    input: &Tensor,
    kernel: &Tensor,
    bias: Option<&Tensor>,
    residual: &Tensor,
    activation: Activation,
    thread_pool: Option<&ThreadPool>,
    prepacked_b: Option<&super::matmul::PackedB>,
) -> Result<Tensor, KernelError> {
    let spec = Conv2dSpec {
        stride_h: 1,
        stride_w: 1,
    };
    let plan = build_conv2d_plan(input, kernel, bias, spec)?;
    if plan.kernel_h != 1 || plan.kernel_w != 1 {
        return Err(KernelError::InvalidConvRank {
            input_rank: input.rank(),
            kernel_rank: kernel.rank(),
        });
    }
    if residual.data().len() != plan.output_len {
        // Silently fall through — caller handles residual itself.
        return Err(KernelError::ConvBiasShapeMismatch {
            bias_shape: residual.shape().to_vec(),
            out_channels: plan.out_channels,
        });
    }
    #[allow(unsafe_code)]
    let mut output = AlignedVec::<f32>::uninitialized(plan.output_len);
    let m = plan.batch * plan.out_h * plan.out_w;
    let k = plan.in_channels;
    let n = plan.out_channels;
    let pointwise_matmul_cfg = pointwise_conv_matmul_config(m, k, n);
    let epilogue = super::matmul::GemmEpilogue {
        bias: bias.map(|b| b.data().as_ptr()),
        activation,
        residual: Some(residual.data().as_ptr()),
    };
    let pp_matched = prepacked_b.filter(|p| p.dims() == (k, n));
    super::matmul::matmul_2d_slices_fused_maybe_packed(
        input.data(),
        m,
        k,
        kernel.data(),
        n,
        &mut output,
        pp_matched,
        epilogue,
        pointwise_matmul_cfg,
        thread_pool,
    );
    Tensor::from_aligned(
        vec![plan.batch, plan.out_h, plan.out_w, plan.out_channels],
        output,
    )
    .map_err(Into::into)
}

pub fn conv2d_nhwc_with_activation_prepacked(
    input: &Tensor,
    kernel: &Tensor,
    bias: Option<&Tensor>,
    spec: Conv2dSpec,
    activation: Activation,
    config: ParallelElementwiseConfig,
    thread_pool: Option<&ThreadPool>,
    prepacked_b: Option<&super::matmul::PackedB>,
) -> Result<Tensor, KernelError> {
    let _ = config;
    let plan = build_conv2d_plan(input, kernel, bias, spec)?;

    // Dedicated first-layer 3×3 stride-2 RGB microkernel. In the no-BLAS
    // runner path this is hit with the input already zero-padded (so we
    // pass pad=0 to the kernel). Saves im2col + blocked-GEMM overhead
    // that's crippling on k = 3·3·3 = 27.
    //
    // `YSCV_NO_FIRST_LAYER_KERNEL=1` env disables the fast path for A/B
    // measurements against the generic GEMM route.
    if plan.kernel_h == 3
        && plan.kernel_w == 3
        && plan.stride_h == 2
        && plan.stride_w == 2
        && plan.in_channels == 3
        && (matches!(activation, Activation::None) || matches!(activation, Activation::Relu))
        && std::env::var("YSCV_NO_FIRST_LAYER_KERNEL").is_err()
    {
        #[allow(unsafe_code)]
        let mut output = AlignedVec::<f32>::uninitialized(plan.output_len);
        super::first_layer_3x3::conv2d_nhwc_3ch_3x3_s2_padded(
            input.data(),
            kernel.data(),
            bias.map(|b| b.data()),
            &mut output,
            plan.batch,
            plan.in_h,
            plan.in_w,
            plan.out_channels,
            0,
            0,
            0,
            0,
            activation,
            thread_pool,
        );
        return Tensor::from_aligned(
            vec![plan.batch, plan.out_h, plan.out_w, plan.out_channels],
            output,
        )
        .map_err(Into::into);
    }

    // 1x1 pointwise conv is exactly GEMM over flattened NHWC rows:
    // [N*H*W, C_in] × [C_in, C_out] -> [N*H*W, C_out].
    // This path is zero-copy on inputs and avoids per-pixel kernel loops.
    if plan.kernel_h == 1 && plan.kernel_w == 1 && plan.stride_h == 1 && plan.stride_w == 1 {
        #[allow(unsafe_code)]
        let mut output = AlignedVec::<f32>::uninitialized(plan.output_len);
        let m = plan.batch * plan.out_h * plan.out_w;
        let k = plan.in_channels;
        let n = plan.out_channels;
        let epilogue = super::matmul::GemmEpilogue {
            bias: bias.map(|b| b.data().as_ptr()),
            activation,
            residual: None,
        };

        // Step S.1′: low-k tile path now lives inside
        // `matmul_2d_slices_fused_maybe_packed` (see Step S.1′ block in
        // matmul.rs). The dispatch here just feeds into matmul with the
        // GemmEpilogue and the matmul function picks the right kernel
        // based on shape + CPU features.

        let pointwise_matmul_cfg = pointwise_conv_matmul_config(m, k, n);
        let pp_matched = prepacked_b.filter(|p| p.dims() == (k, n));
        super::matmul::matmul_2d_slices_fused_maybe_packed(
            input.data(),
            m,
            k,
            kernel.data(),
            n,
            &mut output,
            pp_matched,
            epilogue,
            pointwise_matmul_cfg,
            thread_pool,
        );
        return Tensor::from_aligned(
            vec![plan.batch, plan.out_h, plan.out_w, plan.out_channels],
            output,
        )
        .map_err(Into::into);
    }
    // Non-pointwise: delegate to the original non-prepacked path by calling
    // the inner body (copy-paste of the rest of the function). We keep the
    // two-entry-point design to avoid threading `Option<&PackedB>` through
    // code paths that ignore it. See fallthrough below.
    let _ = prepacked_b;

    // Direct 3×3 microkernel for small inputs (no im2col overhead).
    // For small spatial sizes the im2col copy + BLAS call setup dominates; a
    // direct SIMD kernel that walks the input in-place is significantly faster.
    // However, for deep channels (large C_in * C_out) the BLAS GEMM has better
    // cache tiling, so we use a total-work threshold instead of spatial-only.
    // 2M FLOPs ≈ 0.08ms at 25 GFLOPS — below this the direct kernel wins.
    // Note: NEON kernel is single-threaded, so BLAS GEMM wins for large spatial
    // even with few channels due to multi-core parallelism.
    #[cfg(target_arch = "aarch64")]
    if plan.kernel_h == 3
        && plan.kernel_w == 3
        && plan.batch == 1
        && !cfg!(miri)
        && std::arch::is_aarch64_feature_detected!("neon")
        && (plan.out_h * plan.out_w)
            .saturating_mul(plan.in_channels)
            .saturating_mul(plan.out_channels)
            < direct_conv_work_threshold()
    {
        #[allow(unsafe_code)]
        let mut output = AlignedVec::<f32>::uninitialized(plan.output_len);
        #[allow(unsafe_code)]
        unsafe {
            conv2d_3x3_direct_neon(
                input.data(),
                kernel.data(),
                &mut output,
                plan.in_w,
                plan.in_channels,
                plan.out_channels,
                plan.out_h,
                plan.out_w,
                plan.stride_h,
                plan.stride_w,
            );
        }
        let m = plan.out_h * plan.out_w;
        let n = plan.out_channels;
        match (bias.map(Tensor::data), activation) {
            (Some(bv), Activation::Relu) => {
                super::simd::bias_relu_nhwc_dispatch(&mut output, bv, m, n)
            }
            (Some(bv), Activation::Silu) => {
                super::simd::bias_silu_nhwc_dispatch(&mut output, bv, m, n)
            }
            (Some(bv), Activation::None) => {
                super::simd::bias_add_nhwc_dispatch(&mut output, bv, m, n)
            }
            (None, Activation::Relu) => relu_slice_inplace(&mut output),
            (None, Activation::Silu) => silu_slice_inplace(&mut output),
            (None, Activation::None) => {}
        }
        return Tensor::from_aligned(vec![1, plan.out_h, plan.out_w, plan.out_channels], output)
            .map_err(Into::into);
    }

    #[cfg(target_arch = "x86_64")]
    if plan.kernel_h == 3
        && plan.kernel_w == 3
        && plan.batch == 1
        && !cfg!(miri)
        && is_x86_feature_detected!("avx")
        && is_x86_feature_detected!("fma")
        && (plan.out_h * plan.out_w)
            .saturating_mul(plan.in_channels)
            .saturating_mul(plan.out_channels)
            < direct_conv_work_threshold()
    {
        #[allow(unsafe_code)]
        let mut output = AlignedVec::<f32>::uninitialized(plan.output_len);
        #[allow(unsafe_code)]
        unsafe {
            conv2d_3x3_direct_avx(
                input.data(),
                kernel.data(),
                &mut output,
                plan.in_w,
                plan.in_channels,
                plan.out_channels,
                plan.out_h,
                plan.out_w,
                plan.stride_h,
                plan.stride_w,
            );
        }
        let m = plan.out_h * plan.out_w;
        let n = plan.out_channels;
        match (bias.map(Tensor::data), activation) {
            (Some(bv), Activation::Relu) => {
                super::simd::bias_relu_nhwc_dispatch(&mut output, bv, m, n)
            }
            (Some(bv), Activation::Silu) => {
                super::simd::bias_silu_nhwc_dispatch(&mut output, bv, m, n)
            }
            (Some(bv), Activation::None) => {
                super::simd::bias_add_nhwc_dispatch(&mut output, bv, m, n)
            }
            (None, Activation::Relu) => relu_slice_inplace(&mut output),
            (None, Activation::Silu) => silu_slice_inplace(&mut output),
            (None, Activation::None) => {}
        }
        return Tensor::from_aligned(vec![1, plan.out_h, plan.out_w, plan.out_channels], output)
            .map_err(Into::into);
    }

    #[cfg(target_arch = "x86_64")]
    if plan.kernel_h == 3
        && plan.kernel_w == 3
        && plan.batch == 1
        && !cfg!(miri)
        && is_x86_feature_detected!("fma")
        && (plan.out_h * plan.out_w)
            .saturating_mul(plan.in_channels)
            .saturating_mul(plan.out_channels)
            < direct_conv_work_threshold()
    {
        #[allow(unsafe_code)]
        let mut output = AlignedVec::<f32>::uninitialized(plan.output_len);
        #[allow(unsafe_code)]
        unsafe {
            conv2d_3x3_direct_sse(
                input.data(),
                kernel.data(),
                &mut output,
                plan.in_w,
                plan.in_channels,
                plan.out_channels,
                plan.out_h,
                plan.out_w,
                plan.stride_h,
                plan.stride_w,
            );
        }
        let m = plan.out_h * plan.out_w;
        let n = plan.out_channels;
        match (bias.map(Tensor::data), activation) {
            (Some(bv), Activation::Relu) => {
                super::simd::bias_relu_nhwc_dispatch(&mut output, bv, m, n)
            }
            (Some(bv), Activation::Silu) => {
                super::simd::bias_silu_nhwc_dispatch(&mut output, bv, m, n)
            }
            (Some(bv), Activation::None) => {
                super::simd::bias_add_nhwc_dispatch(&mut output, bv, m, n)
            }
            (None, Activation::Relu) => relu_slice_inplace(&mut output),
            (None, Activation::Silu) => silu_slice_inplace(&mut output),
            (None, Activation::None) => {}
        }
        return Tensor::from_aligned(vec![1, plan.out_h, plan.out_w, plan.out_channels], output)
            .map_err(Into::into);
    }

    // im2col + BLAS sgemm path — only beneficial with external BLAS whose highly
    // optimized GEMM offsets the im2col copy overhead.  Without BLAS, the row-FMA
    // path below is typically faster for the channel counts seen in mobile nets.
    #[cfg(feature = "blas")]
    if !cfg!(miri) && plan.batch == 1 {
        let out = conv2d_im2col_gemm_fused(
            &plan,
            input.data(),
            kernel.data(),
            bias.map(Tensor::data),
            activation,
        )?;
        return Ok(out);
    }

    let input_data = input.data();
    let kernel_data = kernel.data();
    let bias_data = bias.map(Tensor::data);
    let out_row_len = plan.out_w * plan.out_channels;
    if plan.output_len == 0 || out_row_len == 0 {
        return Tensor::from_vec(
            vec![plan.batch, plan.out_h, plan.out_w, plan.out_channels],
            vec![],
        )
        .map_err(Into::into);
    }

    // SAFETY: `conv2d_nhwc_row` writes every element in each output row.
    #[allow(unsafe_code)]
    let mut output = AlignedVec::<f32>::uninitialized(plan.output_len);

    let total_rows = plan.batch * plan.out_h;
    let nthreads = super::config::available_threads(thread_pool);
    // Chunk rows so each thread gets enough work to amortize dispatch overhead.
    // Minimum ~16 rows per thread; fewer threads means more rows per chunk.
    let rows_per_chunk = if nthreads > 1 && total_rows >= nthreads * 2 {
        total_rows.div_ceil(nthreads)
    } else {
        total_rows // sequential
    };
    let chunk_elements = rows_per_chunk * out_row_len;

    if rows_per_chunk < total_rows
        && should_parallelize_len(plan.output_len, config.min_parallel_elements, thread_pool)
    {
        super::super::scope_ctx::par_chunks_mut_dispatch(
            output.as_mut_slice(),
            chunk_elements,
            |chunk_idx, chunk| {
                let start_row = chunk_idx * rows_per_chunk;
                let n_rows = chunk.len() / out_row_len;
                for local_row in 0..n_rows {
                    let row_idx = start_row + local_row;
                    let row_start = local_row * out_row_len;
                    conv2d_nhwc_row(
                        input_data,
                        kernel_data,
                        bias_data,
                        plan,
                        row_idx,
                        &mut chunk[row_start..row_start + out_row_len],
                    );
                }
                apply_conv_activation_inplace(chunk, activation);
            },
        );
    } else {
        for (row_idx, out_row) in output.chunks_mut(out_row_len).enumerate() {
            conv2d_nhwc_row(input_data, kernel_data, bias_data, plan, row_idx, out_row);
        }
        apply_conv_activation_inplace(&mut output, activation);
    }

    Tensor::from_aligned(
        vec![plan.batch, plan.out_h, plan.out_w, plan.out_channels],
        output,
    )
    .map_err(Into::into)
}

pub fn depthwise_conv2d_nhwc_with_config_and_pool(
    input: &Tensor,
    kernel: &Tensor,
    bias: Option<&Tensor>,
    spec: DepthwiseConv2dSpec,
    config: ParallelElementwiseConfig,
    thread_pool: Option<&ThreadPool>,
) -> Result<Tensor, KernelError> {
    depthwise_conv2d_nhwc_with_activation_with_config_and_pool(
        input,
        kernel,
        bias,
        spec,
        Activation::None,
        config,
        thread_pool,
    )
}

pub fn depthwise_conv2d_nhwc_with_activation_with_config_and_pool(
    input: &Tensor,
    kernel: &Tensor,
    bias: Option<&Tensor>,
    spec: DepthwiseConv2dSpec,
    activation: Activation,
    config: ParallelElementwiseConfig,
    thread_pool: Option<&ThreadPool>,
) -> Result<Tensor, KernelError> {
    let plan = build_depthwise_conv2d_plan(input, kernel, bias, spec)?;
    let input_data = input.data();
    let kernel_data = kernel.data();
    let bias_data = bias.map(Tensor::data);
    let out_row_len = plan.out_w * plan.out_channels;
    if plan.output_len == 0 || out_row_len == 0 {
        return Tensor::from_aligned(
            vec![plan.batch, plan.out_h, plan.out_w, plan.out_channels],
            AlignedVec::<f32>::calloc(plan.output_len),
        )
        .map_err(Into::into);
    }

    let mut output = AlignedVec::<f32>::uninitialized(plan.output_len);

    if should_parallelize_len(plan.output_len, config.min_parallel_elements, thread_pool) {
        super::super::scope_ctx::par_chunks_mut_dispatch(
            output.as_mut_slice(),
            out_row_len,
            |row_idx, out_row| {
                depthwise_conv2d_nhwc_row(
                    input_data,
                    kernel_data,
                    bias_data,
                    plan,
                    row_idx,
                    out_row,
                    activation,
                );
            },
        );
    } else {
        for (row_idx, out_row) in output.chunks_mut(out_row_len).enumerate() {
            depthwise_conv2d_nhwc_row(
                input_data,
                kernel_data,
                bias_data,
                plan,
                row_idx,
                out_row,
                activation,
            );
        }
    }

    Tensor::from_aligned(
        vec![plan.batch, plan.out_h, plan.out_w, plan.out_channels],
        output,
    )
    .map_err(Into::into)
}

/// Depthwise Conv2D with implicit zero-padding — avoids separate padded input allocation.
///
/// `input` is NHWC `[N, H, W, C]`, `kernel` is `[KH, KW, C, depth_multiplier]`.
pub fn depthwise_conv2d_nhwc_padded_with_config_and_pool(
    input: &Tensor,
    kernel: &Tensor,
    bias: Option<&Tensor>,
    spec: DepthwiseConv2dSpec,
    pad_top: usize,
    pad_left: usize,
    pad_bottom: usize,
    pad_right: usize,
    config: ParallelElementwiseConfig,
    thread_pool: Option<&ThreadPool>,
) -> Result<Tensor, KernelError> {
    depthwise_conv2d_nhwc_padded_with_activation_with_config_and_pool(
        input,
        kernel,
        bias,
        spec,
        pad_top,
        pad_left,
        pad_bottom,
        pad_right,
        Activation::None,
        config,
        thread_pool,
    )
}

pub fn depthwise_conv2d_nhwc_padded_with_activation_with_config_and_pool(
    input: &Tensor,
    kernel: &Tensor,
    bias: Option<&Tensor>,
    spec: DepthwiseConv2dSpec,
    pad_top: usize,
    pad_left: usize,
    pad_bottom: usize,
    pad_right: usize,
    activation: Activation,
    config: ParallelElementwiseConfig,
    thread_pool: Option<&ThreadPool>,
) -> Result<Tensor, KernelError> {
    let stride_h = spec.stride_h;
    let stride_w = spec.stride_w;
    if input.rank() != 4 || kernel.rank() != 4 {
        return Err(KernelError::InvalidDepthwiseConvRank {
            input_rank: input.rank(),
            kernel_rank: kernel.rank(),
        });
    }
    if stride_h == 0 || stride_w == 0 {
        return Err(KernelError::InvalidDepthwiseConvParameters {
            kernel_h: kernel.shape()[0],
            kernel_w: kernel.shape()[1],
            stride_h,
            stride_w,
        });
    }

    let (batch, in_h, in_w, channels) = (
        input.shape()[0],
        input.shape()[1],
        input.shape()[2],
        input.shape()[3],
    );
    let (kernel_h, kernel_w, kernel_channels, depth_multiplier) = (
        kernel.shape()[0],
        kernel.shape()[1],
        kernel.shape()[2],
        kernel.shape()[3],
    );
    if kernel_h == 0 || kernel_w == 0 || depth_multiplier == 0 {
        return Err(KernelError::InvalidDepthwiseConvParameters {
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
        });
    }
    if kernel_channels != channels {
        return Err(KernelError::DepthwiseConvChannelMismatch {
            input_channels: channels,
            kernel_channels,
        });
    }

    let padded_h = in_h + pad_top + pad_bottom;
    let padded_w = in_w + pad_left + pad_right;
    if kernel_h > padded_h || kernel_w > padded_w {
        return Err(KernelError::DepthwiseConvKernelLargerThanInput {
            input_h: padded_h,
            input_w: padded_w,
            kernel_h,
            kernel_w,
        });
    }

    let out_channels = channels.checked_mul(depth_multiplier).ok_or_else(|| {
        KernelError::Tensor(TensorError::SizeOverflow {
            shape: vec![channels, depth_multiplier],
        })
    })?;
    if let Some(bias_tensor) = bias
        && (bias_tensor.rank() != 1 || bias_tensor.shape()[0] != out_channels)
    {
        return Err(KernelError::DepthwiseConvBiasShapeMismatch {
            bias_shape: bias_tensor.shape().to_vec(),
            out_channels,
        });
    }

    let out_h = (padded_h - kernel_h) / stride_h + 1;
    let out_w = (padded_w - kernel_w) / stride_w + 1;
    let output_len = batch
        .checked_mul(out_h)
        .and_then(|v| v.checked_mul(out_w))
        .and_then(|v| v.checked_mul(out_channels))
        .ok_or_else(|| {
            KernelError::Tensor(TensorError::SizeOverflow {
                shape: vec![batch, out_h, out_w, out_channels],
            })
        })?;
    let out_row_len = out_w * out_channels;
    if output_len == 0 || out_row_len == 0 {
        return Tensor::from_aligned(
            vec![batch, out_h, out_w, out_channels],
            AlignedVec::<f32>::calloc(output_len),
        )
        .map_err(Into::into);
    }

    let in_data = input.data();
    let kernel_data = kernel.data();
    let bias_data = bias.map(Tensor::data);
    let mut output = AlignedVec::<f32>::uninitialized(output_len);

    // Interior region where the full kernel footprint is guaranteed to be in
    // bounds; this lets us skip per-tap boundary checks on most pixels.
    let oy_start = if pad_top > 0 {
        pad_top.div_ceil(stride_h)
    } else {
        0
    };
    let oy_end = if in_h + pad_top >= kernel_h {
        (in_h + pad_top - kernel_h) / stride_h + 1
    } else {
        0
    }
    .min(out_h);
    let ox_start = if pad_left > 0 {
        pad_left.div_ceil(stride_w)
    } else {
        0
    };
    let ox_end = if in_w + pad_left >= kernel_w {
        (in_w + pad_left - kernel_w) / stride_w + 1
    } else {
        0
    }
    .min(out_w);
    let has_interior = oy_start < oy_end && ox_start < ox_end;
    let interior_out_w = ox_end.saturating_sub(ox_start);
    let interior_dm1_plan = if depth_multiplier == 1 && has_interior && interior_out_w > 0 {
        Some(DepthwiseConv2dPlan {
            batch: 1,
            in_h,
            in_w,
            channels,
            depth_multiplier: 1,
            out_h: 1,
            out_w: interior_out_w,
            out_channels: channels,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            output_len: interior_out_w * channels,
        })
    } else {
        None
    };

    let process_row = |row_idx: usize, out_row: &mut [f32]| {
        let batch_idx = row_idx / out_h;
        let out_y = row_idx % out_h;
        let in_y0 = out_y * stride_h;
        let row_interior = has_interior && out_y >= oy_start && out_y < oy_end;

        if let Some(dm1_plan) = interior_dm1_plan
            && row_interior
        {
            for out_x in 0..ox_start {
                let in_x0 = out_x * stride_w;
                let out_base = out_x * out_channels;
                let out_cell = &mut out_row[out_base..out_base + out_channels];
                if let Some(bd) = bias_data {
                    out_cell.copy_from_slice(&bd[..out_channels]);
                } else {
                    out_cell.fill(0.0);
                }
                for ky in 0..kernel_h {
                    let in_y_raw = in_y0 + ky;
                    if in_y_raw < pad_top || in_y_raw >= pad_top + in_h {
                        continue;
                    }
                    let in_y = in_y_raw - pad_top;
                    for kx in 0..kernel_w {
                        let in_x_raw = in_x0 + kx;
                        if in_x_raw < pad_left || in_x_raw >= pad_left + in_w {
                            continue;
                        }
                        let in_x = in_x_raw - pad_left;
                        let input_base = ((batch_idx * in_h + in_y) * in_w + in_x) * channels;
                        let kernel_base = (ky * kernel_w + kx) * channels;
                        for ch in 0..channels {
                            out_cell[ch] +=
                                in_data[input_base + ch] * kernel_data[kernel_base + ch];
                        }
                    }
                }
            }

            let interior_y = in_y0 - pad_top;
            let interior_x = ox_start * stride_w - pad_left;
            let batch_base = batch_idx * in_h * in_w * channels;
            let interior_base = (interior_y * in_w + interior_x) * channels;
            let batch_end = batch_base + in_h * in_w * channels;
            let interior_in = &in_data[batch_base + interior_base..batch_end];
            let interior_out =
                &mut out_row[ox_start * out_channels..(ox_start + dm1_plan.out_w) * out_channels];
            depthwise_conv2d_nhwc_row(
                interior_in,
                kernel_data,
                bias_data,
                dm1_plan,
                0,
                interior_out,
                activation,
            );

            for out_x in ox_end..out_w {
                let in_x0 = out_x * stride_w;
                let out_base = out_x * out_channels;
                let out_cell = &mut out_row[out_base..out_base + out_channels];
                if let Some(bd) = bias_data {
                    out_cell.copy_from_slice(&bd[..out_channels]);
                } else {
                    out_cell.fill(0.0);
                }
                for ky in 0..kernel_h {
                    let in_y_raw = in_y0 + ky;
                    if in_y_raw < pad_top || in_y_raw >= pad_top + in_h {
                        continue;
                    }
                    let in_y = in_y_raw - pad_top;
                    for kx in 0..kernel_w {
                        let in_x_raw = in_x0 + kx;
                        if in_x_raw < pad_left || in_x_raw >= pad_left + in_w {
                            continue;
                        }
                        let in_x = in_x_raw - pad_left;
                        let input_base = ((batch_idx * in_h + in_y) * in_w + in_x) * channels;
                        let kernel_base = (ky * kernel_w + kx) * channels;
                        for ch in 0..channels {
                            out_cell[ch] +=
                                in_data[input_base + ch] * kernel_data[kernel_base + ch];
                        }
                    }
                }
            }
            // Interior already has activation fused; apply to border pixels only.
            let left_border = &mut out_row[..ox_start * out_channels];
            apply_conv_activation_inplace(left_border, activation);
            let right_border = &mut out_row[ox_end * out_channels..];
            apply_conv_activation_inplace(right_border, activation);
            return;
        }

        if depth_multiplier > 1 && row_interior {
            // Left border (needs bounds checks on X).
            for out_x in 0..ox_start {
                let in_x0 = out_x * stride_w;
                let out_base = out_x * out_channels;
                let out_cell = &mut out_row[out_base..out_base + out_channels];
                if let Some(bd) = bias_data {
                    out_cell.copy_from_slice(&bd[..out_channels]);
                } else {
                    out_cell.fill(0.0);
                }
                for ky in 0..kernel_h {
                    let in_y_raw = in_y0 + ky;
                    if in_y_raw < pad_top || in_y_raw >= pad_top + in_h {
                        continue;
                    }
                    let in_y = in_y_raw - pad_top;
                    for kx in 0..kernel_w {
                        let in_x_raw = in_x0 + kx;
                        if in_x_raw < pad_left || in_x_raw >= pad_left + in_w {
                            continue;
                        }
                        let in_x = in_x_raw - pad_left;
                        let input_base = ((batch_idx * in_h + in_y) * in_w + in_x) * channels;
                        let kernel_base = (ky * kernel_w + kx) * channels * depth_multiplier;
                        for ch in 0..channels {
                            let input_val = in_data[input_base + ch];
                            let out_ch_base = ch * depth_multiplier;
                            let kernel_ch_base = kernel_base + ch * depth_multiplier;
                            depthwise_accumulate_dm(
                                out_cell,
                                out_ch_base,
                                kernel_data,
                                kernel_ch_base,
                                input_val,
                                depth_multiplier,
                            );
                        }
                    }
                }
            }

            // Interior span: no bounds checks on either X or Y.
            let in_y_base0 = in_y0 - pad_top;
            for out_x in ox_start..ox_end {
                let in_x_base0 = out_x * stride_w - pad_left;
                let out_base = out_x * out_channels;
                let out_cell = &mut out_row[out_base..out_base + out_channels];
                if let Some(bd) = bias_data {
                    out_cell.copy_from_slice(&bd[..out_channels]);
                } else {
                    out_cell.fill(0.0);
                }
                for ky in 0..kernel_h {
                    let in_y = in_y_base0 + ky;
                    let input_row_base = ((batch_idx * in_h + in_y) * in_w + in_x_base0) * channels;
                    let kernel_row_base = (ky * kernel_w) * channels * depth_multiplier;
                    for kx in 0..kernel_w {
                        let input_base = input_row_base + kx * channels;
                        let kernel_base = kernel_row_base + kx * channels * depth_multiplier;
                        for ch in 0..channels {
                            let input_val = in_data[input_base + ch];
                            let out_ch_base = ch * depth_multiplier;
                            let kernel_ch_base = kernel_base + ch * depth_multiplier;
                            depthwise_accumulate_dm(
                                out_cell,
                                out_ch_base,
                                kernel_data,
                                kernel_ch_base,
                                input_val,
                                depth_multiplier,
                            );
                        }
                    }
                }
            }

            // Right border (needs bounds checks on X).
            for out_x in ox_end..out_w {
                let in_x0 = out_x * stride_w;
                let out_base = out_x * out_channels;
                let out_cell = &mut out_row[out_base..out_base + out_channels];
                if let Some(bd) = bias_data {
                    out_cell.copy_from_slice(&bd[..out_channels]);
                } else {
                    out_cell.fill(0.0);
                }
                for ky in 0..kernel_h {
                    let in_y_raw = in_y0 + ky;
                    if in_y_raw < pad_top || in_y_raw >= pad_top + in_h {
                        continue;
                    }
                    let in_y = in_y_raw - pad_top;
                    for kx in 0..kernel_w {
                        let in_x_raw = in_x0 + kx;
                        if in_x_raw < pad_left || in_x_raw >= pad_left + in_w {
                            continue;
                        }
                        let in_x = in_x_raw - pad_left;
                        let input_base = ((batch_idx * in_h + in_y) * in_w + in_x) * channels;
                        let kernel_base = (ky * kernel_w + kx) * channels * depth_multiplier;
                        for ch in 0..channels {
                            let input_val = in_data[input_base + ch];
                            let out_ch_base = ch * depth_multiplier;
                            let kernel_ch_base = kernel_base + ch * depth_multiplier;
                            depthwise_accumulate_dm(
                                out_cell,
                                out_ch_base,
                                kernel_data,
                                kernel_ch_base,
                                input_val,
                                depth_multiplier,
                            );
                        }
                    }
                }
            }
            apply_conv_activation_inplace(out_row, activation);
            return;
        }

        for out_x in 0..out_w {
            let in_x0 = out_x * stride_w;
            let out_base = out_x * out_channels;
            let out_cell = &mut out_row[out_base..out_base + out_channels];
            if let Some(bd) = bias_data {
                out_cell.copy_from_slice(&bd[..out_channels]);
            } else {
                out_cell.fill(0.0);
            }

            let interior = row_interior && out_x >= ox_start && out_x < ox_end;
            if depth_multiplier == 1 {
                if interior {
                    let in_y_base0 = in_y0 - pad_top;
                    let in_x_base0 = in_x0 - pad_left;
                    for ky in 0..kernel_h {
                        let in_y = in_y_base0 + ky;
                        let input_row_base =
                            ((batch_idx * in_h + in_y) * in_w + in_x_base0) * channels;
                        let kernel_row_base = (ky * kernel_w) * channels;
                        for kx in 0..kernel_w {
                            let input_base = input_row_base + kx * channels;
                            let kernel_base = kernel_row_base + kx * channels;
                            for ch in 0..channels {
                                out_cell[ch] +=
                                    in_data[input_base + ch] * kernel_data[kernel_base + ch];
                            }
                        }
                    }
                } else {
                    for ky in 0..kernel_h {
                        let in_y_raw = in_y0 + ky;
                        if in_y_raw < pad_top || in_y_raw >= pad_top + in_h {
                            continue;
                        }
                        let in_y = in_y_raw - pad_top;
                        for kx in 0..kernel_w {
                            let in_x_raw = in_x0 + kx;
                            if in_x_raw < pad_left || in_x_raw >= pad_left + in_w {
                                continue;
                            }
                            let in_x = in_x_raw - pad_left;
                            let input_base = ((batch_idx * in_h + in_y) * in_w + in_x) * channels;
                            let kernel_base = (ky * kernel_w + kx) * channels;
                            for ch in 0..channels {
                                out_cell[ch] +=
                                    in_data[input_base + ch] * kernel_data[kernel_base + ch];
                            }
                        }
                    }
                }
            } else if interior {
                let in_y_base0 = in_y0 - pad_top;
                let in_x_base0 = in_x0 - pad_left;
                for ky in 0..kernel_h {
                    let in_y = in_y_base0 + ky;
                    let input_row_base = ((batch_idx * in_h + in_y) * in_w + in_x_base0) * channels;
                    let kernel_row_base = (ky * kernel_w) * channels * depth_multiplier;
                    for kx in 0..kernel_w {
                        let input_base = input_row_base + kx * channels;
                        let kernel_base = kernel_row_base + kx * channels * depth_multiplier;
                        for ch in 0..channels {
                            let input_val = in_data[input_base + ch];
                            let out_ch_base = ch * depth_multiplier;
                            let kernel_ch_base = kernel_base + ch * depth_multiplier;
                            depthwise_accumulate_dm(
                                out_cell,
                                out_ch_base,
                                kernel_data,
                                kernel_ch_base,
                                input_val,
                                depth_multiplier,
                            );
                        }
                    }
                }
            } else {
                for ky in 0..kernel_h {
                    let in_y_raw = in_y0 + ky;
                    if in_y_raw < pad_top || in_y_raw >= pad_top + in_h {
                        continue;
                    }
                    let in_y = in_y_raw - pad_top;
                    for kx in 0..kernel_w {
                        let in_x_raw = in_x0 + kx;
                        if in_x_raw < pad_left || in_x_raw >= pad_left + in_w {
                            continue;
                        }
                        let in_x = in_x_raw - pad_left;
                        let input_base = ((batch_idx * in_h + in_y) * in_w + in_x) * channels;
                        let kernel_base = (ky * kernel_w + kx) * channels * depth_multiplier;
                        for ch in 0..channels {
                            let input_val = in_data[input_base + ch];
                            let out_ch_base = ch * depth_multiplier;
                            let kernel_ch_base = kernel_base + ch * depth_multiplier;
                            depthwise_accumulate_dm(
                                out_cell,
                                out_ch_base,
                                kernel_data,
                                kernel_ch_base,
                                input_val,
                                depth_multiplier,
                            );
                        }
                    }
                }
            }
        }
        apply_conv_activation_inplace(out_row, activation);
    };

    if should_parallelize_len(output_len, config.min_parallel_elements, thread_pool) {
        super::super::scope_ctx::par_chunks_mut_dispatch(
            output.as_mut_slice(),
            out_row_len,
            |row_idx, out_row| process_row(row_idx, out_row),
        );
    } else {
        for (row_idx, out_row) in output.chunks_mut(out_row_len).enumerate() {
            process_row(row_idx, out_row);
        }
    }

    Tensor::from_aligned(vec![batch, out_h, out_w, out_channels], output).map_err(Into::into)
}

pub fn separable_conv2d_nhwc_with_config_and_pool(
    input: &Tensor,
    kernels: SeparableConv2dKernels<'_>,
    spec: SeparableConv2dSpec,
    config: ParallelElementwiseConfig,
    thread_pool: Option<&ThreadPool>,
) -> Result<Tensor, KernelError> {
    if kernels.pointwise_kernel.rank() != 4
        || kernels.pointwise_kernel.shape()[0] != 1
        || kernels.pointwise_kernel.shape()[1] != 1
    {
        return Err(KernelError::InvalidSeparablePointwiseKernelShape {
            pointwise_shape: kernels.pointwise_kernel.shape().to_vec(),
        });
    }

    // Try fused per-row DW+PW path — keeps DW output in L1 cache
    let dw_plan = build_depthwise_conv2d_plan(
        input,
        kernels.depthwise_kernel,
        kernels.depthwise_bias,
        DepthwiseConv2dSpec {
            stride_h: spec.stride_h,
            stride_w: spec.stride_w,
        },
    );
    if let Ok(dw_plan) = dw_plan
        && dw_plan.depth_multiplier == 1
    {
        let pw_shape = kernels.pointwise_kernel.shape();
        let pw_out_ch = pw_shape[3]; // [1, 1, C_in, C_out] in KHWC
        let pw_in_ch = pw_shape[2];

        if pw_in_ch == dw_plan.out_channels {
            return fused_dw_pw_nhwc(
                input.data(),
                kernels.depthwise_kernel.data(),
                kernels.depthwise_bias.map(|b| b.data()),
                kernels.pointwise_kernel.data(),
                kernels.pointwise_bias.map(|b| b.data()),
                dw_plan,
                pw_out_ch,
                Activation::None, // DW activation applied inline
                Activation::None, // PW activation
            );
        }
    }

    // Fallback: separate DW then PW
    let depthwise_out = depthwise_conv2d_nhwc_with_config_and_pool(
        input,
        kernels.depthwise_kernel,
        kernels.depthwise_bias,
        DepthwiseConv2dSpec {
            stride_h: spec.stride_h,
            stride_w: spec.stride_w,
        },
        config,
        thread_pool,
    )?;

    conv2d_nhwc_with_config_and_pool(
        &depthwise_out,
        kernels.pointwise_kernel,
        kernels.pointwise_bias,
        Conv2dSpec {
            stride_h: 1,
            stride_w: 1,
        },
        config,
        thread_pool,
    )
}

/// Streaming fused depthwise+pointwise path for NHWC separable blocks.
///
/// Runs depthwise per output row into a small temporary row buffer and feeds
/// it directly into pointwise GEMM, avoiding full DW intermediate materialization.
/// Keeps activations equivalent to the unfused chain:
/// `DW(spec.stride, dw_activation) -> PW(1x1, pw_activation)`.
#[allow(clippy::too_many_arguments)]
pub fn fused_dw_pw_nhwc_streaming(
    input: &Tensor,
    depthwise_kernel: &Tensor,
    depthwise_bias: Option<&Tensor>,
    pointwise_kernel: &Tensor,
    pointwise_bias: Option<&Tensor>,
    stride_h: usize,
    stride_w: usize,
    pad_top: usize,
    pad_left: usize,
    pad_bottom: usize,
    pad_right: usize,
    dw_activation: Activation,
    pw_activation: Activation,
) -> Result<Tensor, KernelError> {
    if pointwise_kernel.rank() != 4
        || pointwise_kernel.shape()[0] != 1
        || pointwise_kernel.shape()[1] != 1
    {
        return Err(KernelError::InvalidSeparablePointwiseKernelShape {
            pointwise_shape: pointwise_kernel.shape().to_vec(),
        });
    }

    // Padded depthwise validation mirrors `depthwise_conv2d_nhwc_padded_*`.
    if input.rank() != 4 || depthwise_kernel.rank() != 4 {
        return Err(KernelError::InvalidDepthwiseConvRank {
            input_rank: input.rank(),
            kernel_rank: depthwise_kernel.rank(),
        });
    }
    if stride_h == 0 || stride_w == 0 {
        return Err(KernelError::InvalidDepthwiseConvParameters {
            kernel_h: depthwise_kernel.shape()[0],
            kernel_w: depthwise_kernel.shape()[1],
            stride_h,
            stride_w,
        });
    }

    let (batch, in_h, in_w, channels) = (
        input.shape()[0],
        input.shape()[1],
        input.shape()[2],
        input.shape()[3],
    );
    let (kernel_h, kernel_w, kernel_channels, depth_multiplier) = (
        depthwise_kernel.shape()[0],
        depthwise_kernel.shape()[1],
        depthwise_kernel.shape()[2],
        depthwise_kernel.shape()[3],
    );
    if kernel_h == 0 || kernel_w == 0 || depth_multiplier == 0 {
        return Err(KernelError::InvalidDepthwiseConvParameters {
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
        });
    }
    if kernel_channels != channels {
        return Err(KernelError::DepthwiseConvChannelMismatch {
            input_channels: channels,
            kernel_channels,
        });
    }
    let padded_h = in_h + pad_top + pad_bottom;
    let padded_w = in_w + pad_left + pad_right;
    if kernel_h > padded_h || kernel_w > padded_w {
        return Err(KernelError::DepthwiseConvKernelLargerThanInput {
            input_h: padded_h,
            input_w: padded_w,
            kernel_h,
            kernel_w,
        });
    }
    let out_channels = channels.checked_mul(depth_multiplier).ok_or_else(|| {
        KernelError::Tensor(TensorError::SizeOverflow {
            shape: vec![channels, depth_multiplier],
        })
    })?;
    if let Some(bias_tensor) = depthwise_bias
        && (bias_tensor.rank() != 1 || bias_tensor.shape()[0] != out_channels)
    {
        return Err(KernelError::DepthwiseConvBiasShapeMismatch {
            bias_shape: bias_tensor.shape().to_vec(),
            out_channels,
        });
    }

    let pw_shape = pointwise_kernel.shape();
    let pw_out_ch = pw_shape[3]; // [1, 1, C_in, C_out] in KHWC
    let pw_in_ch = pw_shape[2];
    if pw_in_ch != out_channels {
        return Err(KernelError::InvalidSeparablePointwiseKernelShape {
            pointwise_shape: pw_shape.to_vec(),
        });
    }

    let has_pad = pad_top != 0 || pad_left != 0 || pad_bottom != 0 || pad_right != 0;
    if depth_multiplier != 1 {
        // Streaming kernel currently supports only depth_multiplier=1.
        // Fall back to the exact unfused chain.
        let depthwise_out = if has_pad {
            depthwise_conv2d_nhwc_padded_with_activation_with_config_and_pool(
                input,
                depthwise_kernel,
                depthwise_bias,
                DepthwiseConv2dSpec { stride_h, stride_w },
                pad_top,
                pad_left,
                pad_bottom,
                pad_right,
                dw_activation,
                ParallelElementwiseConfig::default(),
                None,
            )?
        } else {
            depthwise_conv2d_nhwc_with_activation_with_config_and_pool(
                input,
                depthwise_kernel,
                depthwise_bias,
                DepthwiseConv2dSpec { stride_h, stride_w },
                dw_activation,
                ParallelElementwiseConfig::default(),
                None,
            )?
        };
        return conv2d_nhwc_with_activation_with_config_and_pool(
            &depthwise_out,
            pointwise_kernel,
            pointwise_bias,
            Conv2dSpec {
                stride_h: 1,
                stride_w: 1,
            },
            pw_activation,
            ParallelElementwiseConfig::default(),
            None,
        );
    }

    if !has_pad {
        let dw_plan = build_depthwise_conv2d_plan(
            input,
            depthwise_kernel,
            depthwise_bias,
            DepthwiseConv2dSpec { stride_h, stride_w },
        )?;
        if use_true_fused_dw_pw_dm1(dw_plan, dw_activation) {
            return fused_dw_pw_nhwc_true_fused_dm1(
                input.data(),
                depthwise_kernel.data(),
                depthwise_bias.map(|b| b.data()),
                pointwise_kernel.data(),
                pointwise_bias.map(|b| b.data()),
                dw_plan,
                pw_out_ch,
                dw_activation,
                pw_activation,
            );
        }
        return fused_dw_pw_nhwc(
            input.data(),
            depthwise_kernel.data(),
            depthwise_bias.map(|b| b.data()),
            pointwise_kernel.data(),
            pointwise_bias.map(|b| b.data()),
            dw_plan,
            pw_out_ch,
            dw_activation,
            pw_activation,
        );
    }

    let out_h = (padded_h - kernel_h) / stride_h + 1;
    let out_w = (padded_w - kernel_w) / stride_w + 1;
    let dw_plan = DepthwiseConv2dPlan {
        batch,
        in_h,
        in_w,
        channels,
        depth_multiplier: 1,
        out_h,
        out_w,
        out_channels,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        output_len: batch
            .checked_mul(out_h)
            .and_then(|v| v.checked_mul(out_w))
            .and_then(|v| v.checked_mul(out_channels))
            .ok_or_else(|| {
                KernelError::Tensor(TensorError::SizeOverflow {
                    shape: vec![batch, out_h, out_w, out_channels],
                })
            })?,
    };

    fused_dw_pw_nhwc_padded_dm1(
        input.data(),
        depthwise_kernel.data(),
        depthwise_bias.map(|b| b.data()),
        pointwise_kernel.data(),
        pointwise_bias.map(|b| b.data()),
        dw_plan,
        pad_top,
        pad_left,
        pad_bottom,
        pad_right,
        pw_out_ch,
        dw_activation,
        pw_activation,
    )
}

#[allow(clippy::too_many_arguments)]
fn fused_dw_pw_nhwc_true_fused_dm1(
    input: &[f32],
    dw_kernel: &[f32],
    dw_bias: Option<&[f32]>,
    pw_kernel: &[f32], // [dw_channels, pw_out_channels]
    pw_bias: Option<&[f32]>,
    dw_plan: DepthwiseConv2dPlan,
    pw_out_ch: usize,
    dw_activation: Activation,
    pw_activation: Activation,
) -> Result<Tensor, KernelError> {
    let out_h = dw_plan.out_h;
    let out_w = dw_plan.out_w;
    let channels = dw_plan.channels;
    let batch = dw_plan.batch;
    let total_rows = batch * out_h;
    let pw_row_len = out_w * pw_out_ch;
    let output_len = batch * out_h * pw_row_len;
    let mut output = AlignedVec::<f32>::uninitialized(output_len);

    for row_idx in 0..total_rows {
        let batch_idx = row_idx / out_h;
        let out_y = row_idx % out_h;
        let in_y0 = out_y * dw_plan.stride_h;
        let out_start = row_idx * pw_row_len;
        let out_row = &mut output[out_start..out_start + pw_row_len];

        if let Some(pb) = pw_bias {
            for out_x in 0..out_w {
                let out_cell = &mut out_row[out_x * pw_out_ch..(out_x + 1) * pw_out_ch];
                out_cell.copy_from_slice(&pb[..pw_out_ch]);
            }
        } else {
            out_row.fill(0.0);
        }

        for out_x in 0..out_w {
            let in_x0 = out_x * dw_plan.stride_w;
            let out_cell = &mut out_row[out_x * pw_out_ch..(out_x + 1) * pw_out_ch];
            for ch in 0..channels {
                let mut dw_acc = if let Some(bd) = dw_bias { bd[ch] } else { 0.0 };
                for ky in 0..dw_plan.kernel_h {
                    let in_y = in_y0 + ky;
                    let in_row_base =
                        ((batch_idx * dw_plan.in_h + in_y) * dw_plan.in_w + in_x0) * channels;
                    let ker_row_base = (ky * dw_plan.kernel_w) * channels;
                    for kx in 0..dw_plan.kernel_w {
                        let input_idx = in_row_base + kx * channels + ch;
                        let ker_idx = ker_row_base + kx * channels + ch;
                        dw_acc += input[input_idx] * dw_kernel[ker_idx];
                    }
                }
                let dw_val = apply_conv_activation_scalar(dw_acc, dw_activation);
                let pw_row = &pw_kernel[ch * pw_out_ch..(ch + 1) * pw_out_ch];
                let mut o = 0usize;
                while o + 4 <= pw_out_ch {
                    out_cell[o] += dw_val * pw_row[o];
                    out_cell[o + 1] += dw_val * pw_row[o + 1];
                    out_cell[o + 2] += dw_val * pw_row[o + 2];
                    out_cell[o + 3] += dw_val * pw_row[o + 3];
                    o += 4;
                }
                while o < pw_out_ch {
                    out_cell[o] += dw_val * pw_row[o];
                    o += 1;
                }
            }
        }

        apply_conv_activation_inplace(out_row, pw_activation);
    }

    Tensor::from_aligned(vec![batch, out_h, out_w, pw_out_ch], output).map_err(Into::into)
}

/// Fused depthwise + pointwise convolution.
/// Per-row tiling: DW output per row stays in L1 cache, feeds directly to PW GEMM.
/// Eliminates materializing the full DW intermediate tensor.
#[allow(unsafe_code)]
fn fused_dw_pw_nhwc(
    input: &[f32],
    dw_kernel: &[f32],
    dw_bias: Option<&[f32]>,
    pw_kernel: &[f32], // [1, 1, dw_channels, pw_out_channels] = [dw_ch, pw_out_ch]
    pw_bias: Option<&[f32]>,
    dw_plan: DepthwiseConv2dPlan,
    pw_out_ch: usize,
    dw_activation: Activation,
    pw_activation: Activation,
) -> Result<Tensor, KernelError> {
    let out_h = dw_plan.out_h;
    let out_w = dw_plan.out_w;
    let dw_channels = dw_plan.out_channels;
    let batch = dw_plan.batch;
    let total_rows = batch * out_h;

    // Temp buffer for one DW output row: out_w × dw_channels
    // Reused per row — fits in L1 cache (e.g., 16 × 672 = 10K floats = 40KB)
    let dw_row_len = out_w * dw_channels;
    let pw_row_len = out_w * pw_out_ch;
    let output_len = batch * out_h * pw_row_len;

    let mut output = AlignedVec::<f32>::uninitialized(output_len);

    // PW GEMM epilogue
    let pw_epilogue = super::matmul::GemmEpilogue {
        bias: pw_bias.map(|b| b.as_ptr()),
        activation: pw_activation,
        residual: None,
    };
    let pw_config = ParallelMatmulConfig {
        min_parallel_shared_dim: 1,
        min_parallel_output_elements: usize::MAX, // no parallel within fused
    };

    // Per-row: DW → temp → PW GEMM
    thread_local! {
        static DW_BUF: std::cell::RefCell<Vec<f32>> = const { std::cell::RefCell::new(Vec::new()) };
    }

    DW_BUF.with(|cell| {
        let mut dw_buf = cell.borrow_mut();
        if dw_buf.len() < dw_row_len {
            dw_buf.resize(dw_row_len, 0.0);
        }

        for row_idx in 0..total_rows {
            // Step 1: compute DW for this row → dw_buf
            depthwise_conv2d_nhwc_row(
                input,
                dw_kernel,
                dw_bias,
                dw_plan,
                row_idx,
                &mut dw_buf[..dw_row_len],
                dw_activation,
            );

            // Step 2: PW GEMM — [out_w, dw_channels] × [dw_channels, pw_out_ch]
            let out_start = row_idx * pw_row_len;
            super::matmul::matmul_2d_slices_fused(
                &dw_buf[..dw_row_len],
                out_w,
                dw_channels,
                pw_kernel,
                pw_out_ch,
                &mut output[out_start..out_start + pw_row_len],
                pw_epilogue,
                pw_config,
                None,
            );
        }
    });

    Tensor::from_aligned(vec![batch, out_h, out_w, pw_out_ch], output).map_err(Into::into)
}

#[allow(unsafe_code, clippy::too_many_arguments)]
fn fused_dw_pw_nhwc_padded_dm1(
    input: &[f32],
    dw_kernel: &[f32],
    dw_bias: Option<&[f32]>,
    pw_kernel: &[f32],
    pw_bias: Option<&[f32]>,
    dw_plan: DepthwiseConv2dPlan,
    pad_top: usize,
    pad_left: usize,
    pad_bottom: usize,
    pad_right: usize,
    pw_out_ch: usize,
    dw_activation: Activation,
    pw_activation: Activation,
) -> Result<Tensor, KernelError> {
    let out_h = dw_plan.out_h;
    let out_w = dw_plan.out_w;
    let dw_channels = dw_plan.out_channels;
    let batch = dw_plan.batch;
    let total_rows = batch * out_h;

    let dw_row_len = out_w * dw_channels;
    let pw_row_len = out_w * pw_out_ch;
    let output_len = batch * out_h * pw_row_len;
    let mut output = AlignedVec::<f32>::uninitialized(output_len);

    let pw_epilogue = super::matmul::GemmEpilogue {
        bias: pw_bias.map(|b| b.as_ptr()),
        activation: pw_activation,
        residual: None,
    };
    let pw_config = ParallelMatmulConfig {
        min_parallel_shared_dim: 1,
        min_parallel_output_elements: usize::MAX,
    };

    thread_local! {
        static DW_BUF: std::cell::RefCell<Vec<f32>> = const { std::cell::RefCell::new(Vec::new()) };
    }

    DW_BUF.with(|cell| {
        let mut dw_buf = cell.borrow_mut();
        if dw_buf.len() < dw_row_len {
            dw_buf.resize(dw_row_len, 0.0);
        }

        for row_idx in 0..total_rows {
            depthwise_conv2d_nhwc_padded_row_dm1_stream(
                input,
                dw_kernel,
                dw_bias,
                dw_plan,
                pad_top,
                pad_left,
                pad_bottom,
                pad_right,
                row_idx,
                &mut dw_buf[..dw_row_len],
                dw_activation,
            );

            let out_start = row_idx * pw_row_len;
            super::matmul::matmul_2d_slices_fused(
                &dw_buf[..dw_row_len],
                out_w,
                dw_channels,
                pw_kernel,
                pw_out_ch,
                &mut output[out_start..out_start + pw_row_len],
                pw_epilogue,
                pw_config,
                None,
            );
        }
    });

    Tensor::from_aligned(vec![batch, out_h, out_w, pw_out_ch], output).map_err(Into::into)
}

#[allow(clippy::too_many_arguments)]
fn depthwise_conv2d_nhwc_padded_row_dm1_stream(
    input: &[f32],
    kernel: &[f32],
    bias: Option<&[f32]>,
    plan: DepthwiseConv2dPlan,
    pad_top: usize,
    pad_left: usize,
    _pad_bottom: usize,
    _pad_right: usize,
    row_idx: usize,
    out_row: &mut [f32],
    activation: Activation,
) {
    let batch_idx = row_idx / plan.out_h;
    let out_y = row_idx % plan.out_h;
    let in_y0 = out_y * plan.stride_h;
    let in_h = plan.in_h;
    let in_w = plan.in_w;
    let channels = plan.channels;
    let kernel_h = plan.kernel_h;
    let kernel_w = plan.kernel_w;
    let out_w = plan.out_w;
    let out_channels = plan.out_channels;
    debug_assert_eq!(out_channels, channels);
    debug_assert_eq!(plan.depth_multiplier, 1);

    let oy_start = if pad_top > 0 {
        pad_top.div_ceil(plan.stride_h)
    } else {
        0
    };
    let oy_end = if in_h + pad_top >= kernel_h {
        (in_h + pad_top - kernel_h) / plan.stride_h + 1
    } else {
        0
    }
    .min(plan.out_h);
    let ox_start = if pad_left > 0 {
        pad_left.div_ceil(plan.stride_w)
    } else {
        0
    };
    let ox_end = if in_w + pad_left >= kernel_w {
        (in_w + pad_left - kernel_w) / plan.stride_w + 1
    } else {
        0
    }
    .min(out_w);
    let row_interior = out_y >= oy_start && out_y < oy_end && ox_start < ox_end;

    if row_interior {
        for out_x in 0..ox_start {
            let in_x0 = out_x * plan.stride_w;
            let out_base = out_x * out_channels;
            let out_cell = &mut out_row[out_base..out_base + out_channels];
            if let Some(bd) = bias {
                out_cell.copy_from_slice(&bd[..out_channels]);
            } else {
                out_cell.fill(0.0);
            }
            for ky in 0..kernel_h {
                let in_y_raw = in_y0 + ky;
                if in_y_raw < pad_top || in_y_raw >= pad_top + in_h {
                    continue;
                }
                let in_y = in_y_raw - pad_top;
                for kx in 0..kernel_w {
                    let in_x_raw = in_x0 + kx;
                    if in_x_raw < pad_left || in_x_raw >= pad_left + in_w {
                        continue;
                    }
                    let in_x = in_x_raw - pad_left;
                    let input_base = ((batch_idx * in_h + in_y) * in_w + in_x) * channels;
                    let kernel_base = (ky * kernel_w + kx) * channels;
                    for ch in 0..channels {
                        out_cell[ch] += input[input_base + ch] * kernel[kernel_base + ch];
                    }
                }
            }
        }

        let interior_out_w = ox_end - ox_start;
        let interior_y = in_y0 - pad_top;
        let interior_x = ox_start * plan.stride_w - pad_left;
        let batch_base = batch_idx * in_h * in_w * channels;
        let interior_base = (interior_y * in_w + interior_x) * channels;
        let batch_end = batch_base + in_h * in_w * channels;
        let interior_in = &input[batch_base + interior_base..batch_end];
        let interior_plan = DepthwiseConv2dPlan {
            batch: 1,
            in_h,
            in_w,
            channels,
            depth_multiplier: 1,
            out_h: 1,
            out_w: interior_out_w,
            out_channels: channels,
            kernel_h,
            kernel_w,
            stride_h: plan.stride_h,
            stride_w: plan.stride_w,
            output_len: interior_out_w * channels,
        };
        let interior_out =
            &mut out_row[ox_start * out_channels..(ox_start + interior_out_w) * out_channels];
        depthwise_conv2d_nhwc_row(
            interior_in,
            kernel,
            bias,
            interior_plan,
            0,
            interior_out,
            activation,
        );

        for out_x in ox_end..out_w {
            let in_x0 = out_x * plan.stride_w;
            let out_base = out_x * out_channels;
            let out_cell = &mut out_row[out_base..out_base + out_channels];
            if let Some(bd) = bias {
                out_cell.copy_from_slice(&bd[..out_channels]);
            } else {
                out_cell.fill(0.0);
            }
            for ky in 0..kernel_h {
                let in_y_raw = in_y0 + ky;
                if in_y_raw < pad_top || in_y_raw >= pad_top + in_h {
                    continue;
                }
                let in_y = in_y_raw - pad_top;
                for kx in 0..kernel_w {
                    let in_x_raw = in_x0 + kx;
                    if in_x_raw < pad_left || in_x_raw >= pad_left + in_w {
                        continue;
                    }
                    let in_x = in_x_raw - pad_left;
                    let input_base = ((batch_idx * in_h + in_y) * in_w + in_x) * channels;
                    let kernel_base = (ky * kernel_w + kx) * channels;
                    for ch in 0..channels {
                        out_cell[ch] += input[input_base + ch] * kernel[kernel_base + ch];
                    }
                }
            }
        }

        let left_border = &mut out_row[..ox_start * out_channels];
        apply_conv_activation_inplace(left_border, activation);
        let right_border = &mut out_row[ox_end * out_channels..];
        apply_conv_activation_inplace(right_border, activation);
        return;
    }

    for out_x in 0..out_w {
        let in_x0 = out_x * plan.stride_w;
        let out_base = out_x * out_channels;
        let out_cell = &mut out_row[out_base..out_base + out_channels];
        if let Some(bd) = bias {
            out_cell.copy_from_slice(&bd[..out_channels]);
        } else {
            out_cell.fill(0.0);
        }
        for ky in 0..kernel_h {
            let in_y_raw = in_y0 + ky;
            if in_y_raw < pad_top || in_y_raw >= pad_top + in_h {
                continue;
            }
            let in_y = in_y_raw - pad_top;
            for kx in 0..kernel_w {
                let in_x_raw = in_x0 + kx;
                if in_x_raw < pad_left || in_x_raw >= pad_left + in_w {
                    continue;
                }
                let in_x = in_x_raw - pad_left;
                let input_base = ((batch_idx * in_h + in_y) * in_w + in_x) * channels;
                let kernel_base = (ky * kernel_w + kx) * channels;
                for ch in 0..channels {
                    out_cell[ch] += input[input_base + ch] * kernel[kernel_base + ch];
                }
            }
        }
    }
    apply_conv_activation_inplace(out_row, activation);
}

fn build_conv2d_plan(
    input: &Tensor,
    kernel: &Tensor,
    bias: Option<&Tensor>,
    spec: Conv2dSpec,
) -> Result<Conv2dPlan, KernelError> {
    let stride_h = spec.stride_h;
    let stride_w = spec.stride_w;
    if input.rank() != 4 || kernel.rank() != 4 {
        return Err(KernelError::InvalidConvRank {
            input_rank: input.rank(),
            kernel_rank: kernel.rank(),
        });
    }
    if stride_h == 0 || stride_w == 0 {
        return Err(KernelError::InvalidConvParameters {
            kernel_h: kernel.shape()[0],
            kernel_w: kernel.shape()[1],
            stride_h,
            stride_w,
        });
    }

    let batch = input.shape()[0];
    let in_h = input.shape()[1];
    let in_w = input.shape()[2];
    let in_channels = input.shape()[3];
    let kernel_h = kernel.shape()[0];
    let kernel_w = kernel.shape()[1];
    let kernel_in_channels = kernel.shape()[2];
    let out_channels = kernel.shape()[3];

    if kernel_h == 0 || kernel_w == 0 {
        return Err(KernelError::InvalidConvParameters {
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
        });
    }
    if kernel_in_channels != in_channels {
        return Err(KernelError::ConvChannelMismatch {
            input_channels: in_channels,
            kernel_in_channels,
        });
    }
    if kernel_h > in_h || kernel_w > in_w {
        return Err(KernelError::ConvKernelLargerThanInput {
            input_h: in_h,
            input_w: in_w,
            kernel_h,
            kernel_w,
        });
    }
    if let Some(bias_tensor) = bias
        && (bias_tensor.rank() != 1 || bias_tensor.shape()[0] != out_channels)
    {
        return Err(KernelError::ConvBiasShapeMismatch {
            bias_shape: bias_tensor.shape().to_vec(),
            out_channels,
        });
    }

    let out_h = (in_h - kernel_h) / stride_h + 1;
    let out_w = (in_w - kernel_w) / stride_w + 1;
    let output_len = batch
        .checked_mul(out_h)
        .and_then(|v| v.checked_mul(out_w))
        .and_then(|v| v.checked_mul(out_channels))
        .ok_or_else(|| {
            KernelError::Tensor(TensorError::SizeOverflow {
                shape: vec![batch, out_h, out_w, out_channels],
            })
        })?;

    Ok(Conv2dPlan {
        batch,
        in_h,
        in_w,
        in_channels,
        out_h,
        out_w,
        out_channels,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        output_len,
    })
}

fn build_depthwise_conv2d_plan(
    input: &Tensor,
    kernel: &Tensor,
    bias: Option<&Tensor>,
    spec: DepthwiseConv2dSpec,
) -> Result<DepthwiseConv2dPlan, KernelError> {
    let stride_h = spec.stride_h;
    let stride_w = spec.stride_w;
    if input.rank() != 4 || kernel.rank() != 4 {
        return Err(KernelError::InvalidDepthwiseConvRank {
            input_rank: input.rank(),
            kernel_rank: kernel.rank(),
        });
    }
    if stride_h == 0 || stride_w == 0 {
        return Err(KernelError::InvalidDepthwiseConvParameters {
            kernel_h: kernel.shape()[0],
            kernel_w: kernel.shape()[1],
            stride_h,
            stride_w,
        });
    }

    let batch = input.shape()[0];
    let in_h = input.shape()[1];
    let in_w = input.shape()[2];
    let channels = input.shape()[3];
    let kernel_h = kernel.shape()[0];
    let kernel_w = kernel.shape()[1];
    let kernel_channels = kernel.shape()[2];
    let depth_multiplier = kernel.shape()[3];

    if kernel_h == 0 || kernel_w == 0 || depth_multiplier == 0 {
        return Err(KernelError::InvalidDepthwiseConvParameters {
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
        });
    }
    if kernel_channels != channels {
        return Err(KernelError::DepthwiseConvChannelMismatch {
            input_channels: channels,
            kernel_channels,
        });
    }
    if kernel_h > in_h || kernel_w > in_w {
        return Err(KernelError::DepthwiseConvKernelLargerThanInput {
            input_h: in_h,
            input_w: in_w,
            kernel_h,
            kernel_w,
        });
    }

    let out_channels = channels.checked_mul(depth_multiplier).ok_or_else(|| {
        KernelError::Tensor(TensorError::SizeOverflow {
            shape: vec![channels, depth_multiplier],
        })
    })?;
    if let Some(bias_tensor) = bias
        && (bias_tensor.rank() != 1 || bias_tensor.shape()[0] != out_channels)
    {
        return Err(KernelError::DepthwiseConvBiasShapeMismatch {
            bias_shape: bias_tensor.shape().to_vec(),
            out_channels,
        });
    }

    let out_h = (in_h - kernel_h) / stride_h + 1;
    let out_w = (in_w - kernel_w) / stride_w + 1;
    let output_len = batch
        .checked_mul(out_h)
        .and_then(|v| v.checked_mul(out_w))
        .and_then(|v| v.checked_mul(out_channels))
        .ok_or_else(|| {
            KernelError::Tensor(TensorError::SizeOverflow {
                shape: vec![batch, out_h, out_w, out_channels],
            })
        })?;

    Ok(DepthwiseConv2dPlan {
        batch,
        in_h,
        in_w,
        channels,
        depth_multiplier,
        out_h,
        out_w,
        out_channels,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        output_len,
    })
}

fn conv2d_nhwc_row(
    input: &[f32],
    kernel: &[f32],
    bias: Option<&[f32]>,
    plan: Conv2dPlan,
    row_idx: usize,
    out_row: &mut [f32],
) {
    // For small C_out, use inlined SIMD to avoid 27+ function-pointer dispatches
    // per output pixel (3ns each × 110K calls = 330µs overhead for stem conv).
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if plan.out_channels >= 8
        && plan.out_channels <= 32
        && plan.out_channels.is_multiple_of(8)
        && !cfg!(miri)
        && std::is_x86_feature_detected!("avx")
        && std::is_x86_feature_detected!("fma")
    {
        #[allow(unsafe_code)]
        unsafe {
            conv2d_nhwc_row_inline_avx_fma(input, kernel, bias, plan, row_idx, out_row);
        }
        return;
    }

    conv2d_nhwc_row_generic(input, kernel, bias, plan, row_idx, out_row);
}

fn conv2d_nhwc_row_generic(
    input: &[f32],
    kernel: &[f32],
    bias: Option<&[f32]>,
    plan: Conv2dPlan,
    row_idx: usize,
    out_row: &mut [f32],
) {
    let batch_idx = row_idx / plan.out_h;
    let out_y = row_idx % plan.out_h;
    let in_y0 = out_y * plan.stride_h;
    let batch_input_base = batch_idx * plan.in_h * plan.in_w * plan.in_channels;

    for out_x in 0..plan.out_w {
        let in_x0 = out_x * plan.stride_w;
        let out_cell_base = out_x * plan.out_channels;
        let out_slice = &mut out_row[out_cell_base..out_cell_base + plan.out_channels];

        if let Some(bias_values) = bias {
            out_slice.copy_from_slice(&bias_values[..plan.out_channels]);
        } else {
            out_slice.fill(0.0);
        }

        for ky in 0..plan.kernel_h {
            let in_y = in_y0 + ky;
            let input_row_base = batch_input_base + (in_y * plan.in_w + in_x0) * plan.in_channels;
            let kernel_row_base = ky * plan.kernel_w * plan.in_channels * plan.out_channels;

            for kx in 0..plan.kernel_w {
                let input_pixel_base = input_row_base + kx * plan.in_channels;
                let kernel_pixel_base = kernel_row_base + kx * plan.in_channels * plan.out_channels;

                for in_channel in 0..plan.in_channels {
                    let input_val = input[input_pixel_base + in_channel];
                    let k_base = kernel_pixel_base + in_channel * plan.out_channels;
                    conv_fma_row(
                        out_slice,
                        &kernel[k_base..k_base + plan.out_channels],
                        input_val,
                    );
                }
            }
        }
    }
}

/// Inlined AVX+FMA conv row for small C_out (≤32).
/// Eliminates function-pointer dispatch overhead by keeping accumulators in
/// AVX registers across all kernel positions.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx,fma")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
unsafe fn conv2d_nhwc_row_inline_avx_fma(
    input: &[f32],
    kernel: &[f32],
    bias: Option<&[f32]>,
    plan: Conv2dPlan,
    row_idx: usize,
    out_row: &mut [f32],
) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let n = plan.out_channels;
    let batch_idx = row_idx / plan.out_h;
    let out_y = row_idx % plan.out_h;
    let in_y0 = out_y * plan.stride_h;
    let batch_input_base = batch_idx * plan.in_h * plan.in_w * plan.in_channels;
    let inp = input.as_ptr();
    let ker = kernel.as_ptr();
    let bias_ptr = bias.map(|b| b.as_ptr());
    let out_ptr = out_row.as_mut_ptr();

    for out_x in 0..plan.out_w {
        let in_x0 = out_x * plan.stride_w;
        let ob = out_x * n;

        // Initialize accumulators from bias (or zero).
        // Use exact multiples of 8 to avoid OOB reads/writes.
        let (mut a0, mut a1, mut a2, mut a3);
        if let Some(bp) = bias_ptr {
            a0 = _mm256_loadu_ps(bp);
            a1 = if n >= 16 {
                _mm256_loadu_ps(bp.add(8))
            } else {
                _mm256_setzero_ps()
            };
            a2 = if n >= 24 {
                _mm256_loadu_ps(bp.add(16))
            } else {
                _mm256_setzero_ps()
            };
            a3 = if n >= 32 {
                _mm256_loadu_ps(bp.add(24))
            } else {
                _mm256_setzero_ps()
            };
        } else {
            a0 = _mm256_setzero_ps();
            a1 = _mm256_setzero_ps();
            a2 = _mm256_setzero_ps();
            a3 = _mm256_setzero_ps();
        }

        // Accumulate all kernel positions inline — no function pointer dispatch
        for ky in 0..plan.kernel_h {
            let in_y = in_y0 + ky;
            let input_row_base = batch_input_base + (in_y * plan.in_w + in_x0) * plan.in_channels;
            let kernel_row_base = ky * plan.kernel_w * plan.in_channels * n;

            for kx in 0..plan.kernel_w {
                let input_pixel_base = input_row_base + kx * plan.in_channels;
                let kernel_pixel_base = kernel_row_base + kx * plan.in_channels * n;

                for ic in 0..plan.in_channels {
                    let iv = _mm256_set1_ps(*inp.add(input_pixel_base + ic));
                    let kb = kernel_pixel_base + ic * n;
                    a0 = _mm256_fmadd_ps(iv, _mm256_loadu_ps(ker.add(kb)), a0);
                    if n >= 16 {
                        a1 = _mm256_fmadd_ps(iv, _mm256_loadu_ps(ker.add(kb + 8)), a1);
                    }
                    if n >= 24 {
                        a2 = _mm256_fmadd_ps(iv, _mm256_loadu_ps(ker.add(kb + 16)), a2);
                    }
                    if n >= 32 {
                        a3 = _mm256_fmadd_ps(iv, _mm256_loadu_ps(ker.add(kb + 24)), a3);
                    }
                }
            }
        }

        // Store
        _mm256_storeu_ps(out_ptr.add(ob), a0);
        if n >= 16 {
            _mm256_storeu_ps(out_ptr.add(ob + 8), a1);
        }
        if n >= 24 {
            _mm256_storeu_ps(out_ptr.add(ob + 16), a2);
        }
        if n >= 32 {
            _mm256_storeu_ps(out_ptr.add(ob + 24), a3);
        }
    }
}

/// Direct 3×3 convolution microkernel — no im2col overhead.
/// For each output pixel, load 3×3×C_in input values and multiply with kernel.
/// Accumulate C_out output channels using SIMD FMA.
///
/// When stride_w == 1, processes two adjacent output pixels at a time.
/// Adjacent pixels at (ox, ox+1) share input columns: pixel ox uses columns
/// [ix, ix+1, ix+2] and pixel ox+1 uses [ix+1, ix+2, ix+3]. The middle two
/// columns are shared, saving ~33% of input loads.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
unsafe fn conv2d_3x3_direct_neon(
    input: &[f32],      // [H, W, C_in] NHWC (batch-dim already stripped)
    kernel: &[f32],     // [3, 3, C_in, C_out]
    output: &mut [f32], // [out_H, out_W, C_out]
    w: usize,
    c_in: usize,
    c_out: usize,
    out_h: usize,
    out_w: usize,
    stride_h: usize,
    stride_w: usize,
) {
    use std::arch::aarch64::*;

    // Bounds proof: max input index = ((out_h-1)*stride_h + 2) * w * c_in + (out_w-1)*stride_w + 2) * c_in + c_in - 1
    debug_assert!(
        input.len() >= ((out_h.saturating_sub(1)) * stride_h + 3) * w * c_in,
        "conv2d_3x3_direct_neon: input too small"
    );
    debug_assert!(
        output.len() >= out_h * out_w * c_out,
        "conv2d_3x3_direct_neon: output too small"
    );
    debug_assert!(
        kernel.len() >= 3 * 3 * c_in * c_out,
        "conv2d_3x3_direct_neon: kernel too small"
    );

    for oy in 0..out_h {
        let iy_base = oy * stride_h;

        // When stride_w == 1 and we have at least 2 output pixels remaining,
        // process pairs of adjacent ox positions. For kernel column kx:
        //   pixel at ox   reads input column (ix_base + kx)
        //   pixel at ox+1 reads input column (ix_base + 1 + kx) = (ix_base + kx + 1)
        // So across kx=0,1,2 the pair reads columns ix_base..ix_base+3,
        // and columns ix_base+1 and ix_base+2 are shared.
        let mut ox = 0usize;
        if stride_w == 1 {
            while ox + 2 <= out_w {
                let ix_base = ox; // stride_w == 1
                let out_off_a = (oy * out_w + ox) * c_out;
                let out_off_b = out_off_a + c_out;

                let mut co = 0;
                while co + 8 <= c_out {
                    let mut acc_a0 = vdupq_n_f32(0.0);
                    let mut acc_a1 = vdupq_n_f32(0.0);
                    let mut acc_b0 = vdupq_n_f32(0.0);
                    let mut acc_b1 = vdupq_n_f32(0.0);

                    for ky in 0..3 {
                        let iy = iy_base + ky;
                        let row_base = iy * w;
                        // Load input values for 4 adjacent columns: ix_base..ix_base+3
                        // pixel A uses cols 0,1,2; pixel B uses cols 1,2,3
                        for ci in 0..c_in {
                            let in0 = *input.get_unchecked((row_base + ix_base) * c_in + ci);
                            let in1 = *input.get_unchecked((row_base + ix_base + 1) * c_in + ci);
                            let in2 = *input.get_unchecked((row_base + ix_base + 2) * c_in + ci);
                            let in3 = *input.get_unchecked((row_base + ix_base + 3) * c_in + ci);

                            // kernel weights for kx=0,1,2
                            let k0_off = ky * 3 * c_in * c_out + ci * c_out + co;
                            let k1_off = (ky * 3 + 1) * c_in * c_out + ci * c_out + co;
                            let k2_off = (ky * 3 + 2) * c_in * c_out + ci * c_out + co;
                            let kw0_lo = vld1q_f32(kernel.as_ptr().add(k0_off));
                            let kw0_hi = vld1q_f32(kernel.as_ptr().add(k0_off + 4));
                            let kw1_lo = vld1q_f32(kernel.as_ptr().add(k1_off));
                            let kw1_hi = vld1q_f32(kernel.as_ptr().add(k1_off + 4));
                            let kw2_lo = vld1q_f32(kernel.as_ptr().add(k2_off));
                            let kw2_hi = vld1q_f32(kernel.as_ptr().add(k2_off + 4));

                            // Pixel A: in0*k0 + in1*k1 + in2*k2
                            let va0 = vdupq_n_f32(in0);
                            let va1 = vdupq_n_f32(in1);
                            let va2 = vdupq_n_f32(in2);
                            acc_a0 = vfmaq_f32(acc_a0, va0, kw0_lo);
                            acc_a1 = vfmaq_f32(acc_a1, va0, kw0_hi);
                            acc_a0 = vfmaq_f32(acc_a0, va1, kw1_lo);
                            acc_a1 = vfmaq_f32(acc_a1, va1, kw1_hi);
                            acc_a0 = vfmaq_f32(acc_a0, va2, kw2_lo);
                            acc_a1 = vfmaq_f32(acc_a1, va2, kw2_hi);

                            // Pixel B: in1*k0 + in2*k1 + in3*k2
                            let vb3 = vdupq_n_f32(in3);
                            acc_b0 = vfmaq_f32(acc_b0, va1, kw0_lo);
                            acc_b1 = vfmaq_f32(acc_b1, va1, kw0_hi);
                            acc_b0 = vfmaq_f32(acc_b0, va2, kw1_lo);
                            acc_b1 = vfmaq_f32(acc_b1, va2, kw1_hi);
                            acc_b0 = vfmaq_f32(acc_b0, vb3, kw2_lo);
                            acc_b1 = vfmaq_f32(acc_b1, vb3, kw2_hi);
                        }
                    }

                    vst1q_f32(output.as_mut_ptr().add(out_off_a + co), acc_a0);
                    vst1q_f32(output.as_mut_ptr().add(out_off_a + co + 4), acc_a1);
                    vst1q_f32(output.as_mut_ptr().add(out_off_b + co), acc_b0);
                    vst1q_f32(output.as_mut_ptr().add(out_off_b + co + 4), acc_b1);
                    co += 8;
                }

                while co + 4 <= c_out {
                    let mut acc_a = vdupq_n_f32(0.0);
                    let mut acc_b = vdupq_n_f32(0.0);

                    for ky in 0..3 {
                        let iy = iy_base + ky;
                        let row_base = iy * w;
                        for ci in 0..c_in {
                            let in0 = *input.get_unchecked((row_base + ix_base) * c_in + ci);
                            let in1 = *input.get_unchecked((row_base + ix_base + 1) * c_in + ci);
                            let in2 = *input.get_unchecked((row_base + ix_base + 2) * c_in + ci);
                            let in3 = *input.get_unchecked((row_base + ix_base + 3) * c_in + ci);

                            let k0_off = ky * 3 * c_in * c_out + ci * c_out + co;
                            let k1_off = (ky * 3 + 1) * c_in * c_out + ci * c_out + co;
                            let k2_off = (ky * 3 + 2) * c_in * c_out + ci * c_out + co;
                            let kw0 = vld1q_f32(kernel.as_ptr().add(k0_off));
                            let kw1 = vld1q_f32(kernel.as_ptr().add(k1_off));
                            let kw2 = vld1q_f32(kernel.as_ptr().add(k2_off));

                            let va0 = vdupq_n_f32(in0);
                            let va1 = vdupq_n_f32(in1);
                            let va2 = vdupq_n_f32(in2);
                            acc_a = vfmaq_f32(acc_a, va0, kw0);
                            acc_a = vfmaq_f32(acc_a, va1, kw1);
                            acc_a = vfmaq_f32(acc_a, va2, kw2);

                            let vb3 = vdupq_n_f32(in3);
                            acc_b = vfmaq_f32(acc_b, va1, kw0);
                            acc_b = vfmaq_f32(acc_b, va2, kw1);
                            acc_b = vfmaq_f32(acc_b, vb3, kw2);
                        }
                    }

                    vst1q_f32(output.as_mut_ptr().add(out_off_a + co), acc_a);
                    vst1q_f32(output.as_mut_ptr().add(out_off_b + co), acc_b);
                    co += 4;
                }

                while co < c_out {
                    let mut acc_a = 0.0f32;
                    let mut acc_b = 0.0f32;
                    for ky in 0..3 {
                        let iy = iy_base + ky;
                        let row_base = iy * w;
                        for ci in 0..c_in {
                            let in0 = input[(row_base + ix_base) * c_in + ci];
                            let in1 = input[(row_base + ix_base + 1) * c_in + ci];
                            let in2 = input[(row_base + ix_base + 2) * c_in + ci];
                            let in3 = input[(row_base + ix_base + 3) * c_in + ci];
                            let k0 = kernel[ky * 3 * c_in * c_out + ci * c_out + co];
                            let k1 = kernel[(ky * 3 + 1) * c_in * c_out + ci * c_out + co];
                            let k2 = kernel[(ky * 3 + 2) * c_in * c_out + ci * c_out + co];
                            acc_a += in0 * k0 + in1 * k1 + in2 * k2;
                            acc_b += in1 * k0 + in2 * k1 + in3 * k2;
                        }
                    }
                    *output.get_unchecked_mut(out_off_a + co) = acc_a;
                    *output.get_unchecked_mut(out_off_b + co) = acc_b;
                    co += 1;
                }

                ox += 2;
            }
        }

        // Handle remaining single pixels (odd out_w or stride_w != 1).
        while ox < out_w {
            let ix_base = ox * stride_w;
            let out_off = (oy * out_w + ox) * c_out;

            let mut co = 0;
            while co + 8 <= c_out {
                let mut acc0 = vdupq_n_f32(0.0);
                let mut acc1 = vdupq_n_f32(0.0);

                for ky in 0..3 {
                    for kx in 0..3 {
                        let iy = iy_base + ky;
                        let ix = ix_base + kx;
                        let in_off = (iy * w + ix) * c_in;
                        let k_base = (ky * 3 + kx) * c_in * c_out;

                        for ci in 0..c_in {
                            let iv = vdupq_n_f32(*input.get_unchecked(in_off + ci));
                            let koff = k_base + ci * c_out + co;
                            acc0 = vfmaq_f32(acc0, iv, vld1q_f32(kernel.as_ptr().add(koff)));
                            acc1 = vfmaq_f32(acc1, iv, vld1q_f32(kernel.as_ptr().add(koff + 4)));
                        }
                    }
                }

                vst1q_f32(output.as_mut_ptr().add(out_off + co), acc0);
                vst1q_f32(output.as_mut_ptr().add(out_off + co + 4), acc1);
                co += 8;
            }

            while co + 4 <= c_out {
                let mut acc = vdupq_n_f32(0.0);
                for ky in 0..3 {
                    for kx in 0..3 {
                        let iy = iy_base + ky;
                        let ix = ix_base + kx;
                        let in_off = (iy * w + ix) * c_in;
                        for ci in 0..c_in {
                            let iv = vdupq_n_f32(*input.get_unchecked(in_off + ci));
                            acc = vfmaq_f32(
                                acc,
                                iv,
                                vld1q_f32(
                                    kernel
                                        .as_ptr()
                                        .add((ky * 3 + kx) * c_in * c_out + ci * c_out + co),
                                ),
                            );
                        }
                    }
                }
                vst1q_f32(output.as_mut_ptr().add(out_off + co), acc);
                co += 4;
            }

            // Handle remaining channels scalar
            while co < c_out {
                let mut acc = 0.0f32;
                for ky in 0..3 {
                    for kx in 0..3 {
                        let iy = iy_base + ky;
                        let ix = ix_base + kx;
                        for ci in 0..c_in {
                            acc += input[(iy * w + ix) * c_in + ci]
                                * kernel[(ky * 3 + kx) * c_in * c_out + ci * c_out + co];
                        }
                    }
                }
                *output.get_unchecked_mut(out_off + co) = acc;
                co += 1;
            }

            ox += 1;
        }
    }
}

/// Direct 3×3 convolution microkernel for x86_64 with AVX-256 + FMA.
/// Processes 8 output channels per iteration (vs 4 for SSE), doubling
/// throughput on the inner c_out loop. Falls back to scalar for tail channels.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx", enable = "fma")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
unsafe fn conv2d_3x3_direct_avx(
    input: &[f32],      // [H, W, C_in] NHWC (batch-dim already stripped)
    kernel: &[f32],     // [3, 3, C_in, C_out]
    output: &mut [f32], // [out_H, out_W, C_out]
    w: usize,
    c_in: usize,
    c_out: usize,
    out_h: usize,
    out_w: usize,
    stride_h: usize,
    stride_w: usize,
) {
    use std::arch::x86_64::*;

    for oy in 0..out_h {
        let iy_base = oy * stride_h;

        let mut ox = 0usize;
        if stride_w == 1 {
            while ox + 2 <= out_w {
                let ix_base = ox; // stride_w == 1
                let out_off_a = (oy * out_w + ox) * c_out;
                let out_off_b = out_off_a + c_out;

                // Process 16 output channels per iteration (2x AVX-256 registers)
                let mut co = 0;
                while co + 16 <= c_out {
                    let mut acc_a0 = _mm256_setzero_ps();
                    let mut acc_a1 = _mm256_setzero_ps();
                    let mut acc_b0 = _mm256_setzero_ps();
                    let mut acc_b1 = _mm256_setzero_ps();

                    for ky in 0..3 {
                        let iy = iy_base + ky;
                        let row_base = iy * w;
                        for ci in 0..c_in {
                            let in0 = *input.get_unchecked((row_base + ix_base) * c_in + ci);
                            let in1 = *input.get_unchecked((row_base + ix_base + 1) * c_in + ci);
                            let in2 = *input.get_unchecked((row_base + ix_base + 2) * c_in + ci);
                            let in3 = *input.get_unchecked((row_base + ix_base + 3) * c_in + ci);

                            let k0_off = ky * 3 * c_in * c_out + ci * c_out + co;
                            let k1_off = (ky * 3 + 1) * c_in * c_out + ci * c_out + co;
                            let k2_off = (ky * 3 + 2) * c_in * c_out + ci * c_out + co;
                            let kw0_lo = _mm256_loadu_ps(kernel.as_ptr().add(k0_off));
                            let kw0_hi = _mm256_loadu_ps(kernel.as_ptr().add(k0_off + 8));
                            let kw1_lo = _mm256_loadu_ps(kernel.as_ptr().add(k1_off));
                            let kw1_hi = _mm256_loadu_ps(kernel.as_ptr().add(k1_off + 8));
                            let kw2_lo = _mm256_loadu_ps(kernel.as_ptr().add(k2_off));
                            let kw2_hi = _mm256_loadu_ps(kernel.as_ptr().add(k2_off + 8));

                            // Pixel A: in0*k0 + in1*k1 + in2*k2
                            let va0 = _mm256_set1_ps(in0);
                            let va1 = _mm256_set1_ps(in1);
                            let va2 = _mm256_set1_ps(in2);
                            acc_a0 = _mm256_fmadd_ps(va0, kw0_lo, acc_a0);
                            acc_a1 = _mm256_fmadd_ps(va0, kw0_hi, acc_a1);
                            acc_a0 = _mm256_fmadd_ps(va1, kw1_lo, acc_a0);
                            acc_a1 = _mm256_fmadd_ps(va1, kw1_hi, acc_a1);
                            acc_a0 = _mm256_fmadd_ps(va2, kw2_lo, acc_a0);
                            acc_a1 = _mm256_fmadd_ps(va2, kw2_hi, acc_a1);

                            // Pixel B: in1*k0 + in2*k1 + in3*k2
                            let vb3 = _mm256_set1_ps(in3);
                            acc_b0 = _mm256_fmadd_ps(va1, kw0_lo, acc_b0);
                            acc_b1 = _mm256_fmadd_ps(va1, kw0_hi, acc_b1);
                            acc_b0 = _mm256_fmadd_ps(va2, kw1_lo, acc_b0);
                            acc_b1 = _mm256_fmadd_ps(va2, kw1_hi, acc_b1);
                            acc_b0 = _mm256_fmadd_ps(vb3, kw2_lo, acc_b0);
                            acc_b1 = _mm256_fmadd_ps(vb3, kw2_hi, acc_b1);
                        }
                    }

                    _mm256_storeu_ps(output.as_mut_ptr().add(out_off_a + co), acc_a0);
                    _mm256_storeu_ps(output.as_mut_ptr().add(out_off_a + co + 8), acc_a1);
                    _mm256_storeu_ps(output.as_mut_ptr().add(out_off_b + co), acc_b0);
                    _mm256_storeu_ps(output.as_mut_ptr().add(out_off_b + co + 8), acc_b1);
                    co += 16;
                }

                // Process 8 output channels with single AVX register pair
                while co + 8 <= c_out {
                    let mut acc_a = _mm256_setzero_ps();
                    let mut acc_b = _mm256_setzero_ps();

                    for ky in 0..3 {
                        let iy = iy_base + ky;
                        let row_base = iy * w;
                        for ci in 0..c_in {
                            let in0 = *input.get_unchecked((row_base + ix_base) * c_in + ci);
                            let in1 = *input.get_unchecked((row_base + ix_base + 1) * c_in + ci);
                            let in2 = *input.get_unchecked((row_base + ix_base + 2) * c_in + ci);
                            let in3 = *input.get_unchecked((row_base + ix_base + 3) * c_in + ci);

                            let k0_off = ky * 3 * c_in * c_out + ci * c_out + co;
                            let k1_off = (ky * 3 + 1) * c_in * c_out + ci * c_out + co;
                            let k2_off = (ky * 3 + 2) * c_in * c_out + ci * c_out + co;
                            let kw0 = _mm256_loadu_ps(kernel.as_ptr().add(k0_off));
                            let kw1 = _mm256_loadu_ps(kernel.as_ptr().add(k1_off));
                            let kw2 = _mm256_loadu_ps(kernel.as_ptr().add(k2_off));

                            let va0 = _mm256_set1_ps(in0);
                            let va1 = _mm256_set1_ps(in1);
                            let va2 = _mm256_set1_ps(in2);
                            acc_a = _mm256_fmadd_ps(va0, kw0, acc_a);
                            acc_a = _mm256_fmadd_ps(va1, kw1, acc_a);
                            acc_a = _mm256_fmadd_ps(va2, kw2, acc_a);

                            let vb3 = _mm256_set1_ps(in3);
                            acc_b = _mm256_fmadd_ps(va1, kw0, acc_b);
                            acc_b = _mm256_fmadd_ps(va2, kw1, acc_b);
                            acc_b = _mm256_fmadd_ps(vb3, kw2, acc_b);
                        }
                    }

                    _mm256_storeu_ps(output.as_mut_ptr().add(out_off_a + co), acc_a);
                    _mm256_storeu_ps(output.as_mut_ptr().add(out_off_b + co), acc_b);
                    co += 8;
                }

                // Scalar tail for remaining channels
                while co < c_out {
                    let mut acc_a = 0.0f32;
                    let mut acc_b = 0.0f32;
                    for ky in 0..3 {
                        let iy = iy_base + ky;
                        let row_base = iy * w;
                        for ci in 0..c_in {
                            let in0 = input[(row_base + ix_base) * c_in + ci];
                            let in1 = input[(row_base + ix_base + 1) * c_in + ci];
                            let in2 = input[(row_base + ix_base + 2) * c_in + ci];
                            let in3 = input[(row_base + ix_base + 3) * c_in + ci];
                            let k0 = kernel[ky * 3 * c_in * c_out + ci * c_out + co];
                            let k1 = kernel[(ky * 3 + 1) * c_in * c_out + ci * c_out + co];
                            let k2 = kernel[(ky * 3 + 2) * c_in * c_out + ci * c_out + co];
                            acc_a += in0 * k0 + in1 * k1 + in2 * k2;
                            acc_b += in1 * k0 + in2 * k1 + in3 * k2;
                        }
                    }
                    *output.get_unchecked_mut(out_off_a + co) = acc_a;
                    *output.get_unchecked_mut(out_off_b + co) = acc_b;
                    co += 1;
                }

                ox += 2;
            }
        }

        // Handle remaining single pixels (odd out_w or stride_w != 1).
        while ox < out_w {
            let ix_base = ox * stride_w;
            let out_off = (oy * out_w + ox) * c_out;

            let mut co = 0;
            while co + 8 <= c_out {
                let mut acc = _mm256_setzero_ps();

                for ky in 0..3 {
                    for kx in 0..3 {
                        let iy = iy_base + ky;
                        let ix = ix_base + kx;
                        let in_off = (iy * w + ix) * c_in;
                        let k_base = (ky * 3 + kx) * c_in * c_out;

                        for ci in 0..c_in {
                            let iv = _mm256_set1_ps(*input.get_unchecked(in_off + ci));
                            let koff = k_base + ci * c_out + co;
                            acc = _mm256_fmadd_ps(
                                iv,
                                _mm256_loadu_ps(kernel.as_ptr().add(koff)),
                                acc,
                            );
                        }
                    }
                }

                _mm256_storeu_ps(output.as_mut_ptr().add(out_off + co), acc);
                co += 8;
            }

            // Handle remaining channels scalar
            while co < c_out {
                let mut acc = 0.0f32;
                for ky in 0..3 {
                    for kx in 0..3 {
                        let iy = iy_base + ky;
                        let ix = ix_base + kx;
                        for ci in 0..c_in {
                            acc += input[(iy * w + ix) * c_in + ci]
                                * kernel[(ky * 3 + kx) * c_in * c_out + ci * c_out + co];
                        }
                    }
                }
                *output.get_unchecked_mut(out_off + co) = acc;
                co += 1;
            }

            ox += 1;
        }
    }
}

/// Direct 3×3 convolution microkernel for x86_64 with SSE + FMA.
/// Mirrors the NEON implementation: processes two adjacent output pixels at a
/// time when stride_w == 1, sharing overlapping input columns to save loads.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse", enable = "fma")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
unsafe fn conv2d_3x3_direct_sse(
    input: &[f32],      // [H, W, C_in] NHWC (batch-dim already stripped)
    kernel: &[f32],     // [3, 3, C_in, C_out]
    output: &mut [f32], // [out_H, out_W, C_out]
    w: usize,
    c_in: usize,
    c_out: usize,
    out_h: usize,
    out_w: usize,
    stride_h: usize,
    stride_w: usize,
) {
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    for oy in 0..out_h {
        let iy_base = oy * stride_h;

        let mut ox = 0usize;
        if stride_w == 1 {
            while ox + 2 <= out_w {
                let ix_base = ox; // stride_w == 1
                let out_off_a = (oy * out_w + ox) * c_out;
                let out_off_b = out_off_a + c_out;

                let mut co = 0;
                while co + 8 <= c_out {
                    let mut acc_a0 = _mm_setzero_ps();
                    let mut acc_a1 = _mm_setzero_ps();
                    let mut acc_b0 = _mm_setzero_ps();
                    let mut acc_b1 = _mm_setzero_ps();

                    for ky in 0..3 {
                        let iy = iy_base + ky;
                        let row_base = iy * w;
                        for ci in 0..c_in {
                            let in0 = *input.get_unchecked((row_base + ix_base) * c_in + ci);
                            let in1 = *input.get_unchecked((row_base + ix_base + 1) * c_in + ci);
                            let in2 = *input.get_unchecked((row_base + ix_base + 2) * c_in + ci);
                            let in3 = *input.get_unchecked((row_base + ix_base + 3) * c_in + ci);

                            let k0_off = ky * 3 * c_in * c_out + ci * c_out + co;
                            let k1_off = (ky * 3 + 1) * c_in * c_out + ci * c_out + co;
                            let k2_off = (ky * 3 + 2) * c_in * c_out + ci * c_out + co;
                            let kw0_lo = _mm_loadu_ps(kernel.as_ptr().add(k0_off));
                            let kw0_hi = _mm_loadu_ps(kernel.as_ptr().add(k0_off + 4));
                            let kw1_lo = _mm_loadu_ps(kernel.as_ptr().add(k1_off));
                            let kw1_hi = _mm_loadu_ps(kernel.as_ptr().add(k1_off + 4));
                            let kw2_lo = _mm_loadu_ps(kernel.as_ptr().add(k2_off));
                            let kw2_hi = _mm_loadu_ps(kernel.as_ptr().add(k2_off + 4));

                            // Pixel A: in0*k0 + in1*k1 + in2*k2
                            let va0 = _mm_set1_ps(in0);
                            let va1 = _mm_set1_ps(in1);
                            let va2 = _mm_set1_ps(in2);
                            acc_a0 = _mm_fmadd_ps(va0, kw0_lo, acc_a0);
                            acc_a1 = _mm_fmadd_ps(va0, kw0_hi, acc_a1);
                            acc_a0 = _mm_fmadd_ps(va1, kw1_lo, acc_a0);
                            acc_a1 = _mm_fmadd_ps(va1, kw1_hi, acc_a1);
                            acc_a0 = _mm_fmadd_ps(va2, kw2_lo, acc_a0);
                            acc_a1 = _mm_fmadd_ps(va2, kw2_hi, acc_a1);

                            // Pixel B: in1*k0 + in2*k1 + in3*k2
                            let vb3 = _mm_set1_ps(in3);
                            acc_b0 = _mm_fmadd_ps(va1, kw0_lo, acc_b0);
                            acc_b1 = _mm_fmadd_ps(va1, kw0_hi, acc_b1);
                            acc_b0 = _mm_fmadd_ps(va2, kw1_lo, acc_b0);
                            acc_b1 = _mm_fmadd_ps(va2, kw1_hi, acc_b1);
                            acc_b0 = _mm_fmadd_ps(vb3, kw2_lo, acc_b0);
                            acc_b1 = _mm_fmadd_ps(vb3, kw2_hi, acc_b1);
                        }
                    }

                    _mm_storeu_ps(output.as_mut_ptr().add(out_off_a + co), acc_a0);
                    _mm_storeu_ps(output.as_mut_ptr().add(out_off_a + co + 4), acc_a1);
                    _mm_storeu_ps(output.as_mut_ptr().add(out_off_b + co), acc_b0);
                    _mm_storeu_ps(output.as_mut_ptr().add(out_off_b + co + 4), acc_b1);
                    co += 8;
                }

                while co + 4 <= c_out {
                    let mut acc_a = _mm_setzero_ps();
                    let mut acc_b = _mm_setzero_ps();

                    for ky in 0..3 {
                        let iy = iy_base + ky;
                        let row_base = iy * w;
                        for ci in 0..c_in {
                            let in0 = *input.get_unchecked((row_base + ix_base) * c_in + ci);
                            let in1 = *input.get_unchecked((row_base + ix_base + 1) * c_in + ci);
                            let in2 = *input.get_unchecked((row_base + ix_base + 2) * c_in + ci);
                            let in3 = *input.get_unchecked((row_base + ix_base + 3) * c_in + ci);

                            let k0_off = ky * 3 * c_in * c_out + ci * c_out + co;
                            let k1_off = (ky * 3 + 1) * c_in * c_out + ci * c_out + co;
                            let k2_off = (ky * 3 + 2) * c_in * c_out + ci * c_out + co;
                            let kw0 = _mm_loadu_ps(kernel.as_ptr().add(k0_off));
                            let kw1 = _mm_loadu_ps(kernel.as_ptr().add(k1_off));
                            let kw2 = _mm_loadu_ps(kernel.as_ptr().add(k2_off));

                            let va0 = _mm_set1_ps(in0);
                            let va1 = _mm_set1_ps(in1);
                            let va2 = _mm_set1_ps(in2);
                            acc_a = _mm_fmadd_ps(va0, kw0, acc_a);
                            acc_a = _mm_fmadd_ps(va1, kw1, acc_a);
                            acc_a = _mm_fmadd_ps(va2, kw2, acc_a);

                            let vb3 = _mm_set1_ps(in3);
                            acc_b = _mm_fmadd_ps(va1, kw0, acc_b);
                            acc_b = _mm_fmadd_ps(va2, kw1, acc_b);
                            acc_b = _mm_fmadd_ps(vb3, kw2, acc_b);
                        }
                    }

                    _mm_storeu_ps(output.as_mut_ptr().add(out_off_a + co), acc_a);
                    _mm_storeu_ps(output.as_mut_ptr().add(out_off_b + co), acc_b);
                    co += 4;
                }

                while co < c_out {
                    let mut acc_a = 0.0f32;
                    let mut acc_b = 0.0f32;
                    for ky in 0..3 {
                        let iy = iy_base + ky;
                        let row_base = iy * w;
                        for ci in 0..c_in {
                            let in0 = input[(row_base + ix_base) * c_in + ci];
                            let in1 = input[(row_base + ix_base + 1) * c_in + ci];
                            let in2 = input[(row_base + ix_base + 2) * c_in + ci];
                            let in3 = input[(row_base + ix_base + 3) * c_in + ci];
                            let k0 = kernel[ky * 3 * c_in * c_out + ci * c_out + co];
                            let k1 = kernel[(ky * 3 + 1) * c_in * c_out + ci * c_out + co];
                            let k2 = kernel[(ky * 3 + 2) * c_in * c_out + ci * c_out + co];
                            acc_a += in0 * k0 + in1 * k1 + in2 * k2;
                            acc_b += in1 * k0 + in2 * k1 + in3 * k2;
                        }
                    }
                    *output.get_unchecked_mut(out_off_a + co) = acc_a;
                    *output.get_unchecked_mut(out_off_b + co) = acc_b;
                    co += 1;
                }

                ox += 2;
            }
        }

        // Handle remaining single pixels (odd out_w or stride_w != 1).
        while ox < out_w {
            let ix_base = ox * stride_w;
            let out_off = (oy * out_w + ox) * c_out;

            let mut co = 0;
            while co + 8 <= c_out {
                let mut acc0 = _mm_setzero_ps();
                let mut acc1 = _mm_setzero_ps();

                for ky in 0..3 {
                    for kx in 0..3 {
                        let iy = iy_base + ky;
                        let ix = ix_base + kx;
                        let in_off = (iy * w + ix) * c_in;
                        let k_base = (ky * 3 + kx) * c_in * c_out;

                        for ci in 0..c_in {
                            let iv = _mm_set1_ps(*input.get_unchecked(in_off + ci));
                            let koff = k_base + ci * c_out + co;
                            acc0 = _mm_fmadd_ps(iv, _mm_loadu_ps(kernel.as_ptr().add(koff)), acc0);
                            acc1 =
                                _mm_fmadd_ps(iv, _mm_loadu_ps(kernel.as_ptr().add(koff + 4)), acc1);
                        }
                    }
                }

                _mm_storeu_ps(output.as_mut_ptr().add(out_off + co), acc0);
                _mm_storeu_ps(output.as_mut_ptr().add(out_off + co + 4), acc1);
                co += 8;
            }

            while co + 4 <= c_out {
                let mut acc = _mm_setzero_ps();
                for ky in 0..3 {
                    for kx in 0..3 {
                        let iy = iy_base + ky;
                        let ix = ix_base + kx;
                        let in_off = (iy * w + ix) * c_in;
                        for ci in 0..c_in {
                            let iv = _mm_set1_ps(*input.get_unchecked(in_off + ci));
                            acc = _mm_fmadd_ps(
                                iv,
                                _mm_loadu_ps(
                                    kernel
                                        .as_ptr()
                                        .add((ky * 3 + kx) * c_in * c_out + ci * c_out + co),
                                ),
                                acc,
                            );
                        }
                    }
                }
                _mm_storeu_ps(output.as_mut_ptr().add(out_off + co), acc);
                co += 4;
            }

            // Handle remaining channels scalar
            while co < c_out {
                let mut acc = 0.0f32;
                for ky in 0..3 {
                    for kx in 0..3 {
                        let iy = iy_base + ky;
                        let ix = ix_base + kx;
                        for ci in 0..c_in {
                            acc += input[(iy * w + ix) * c_in + ci]
                                * kernel[(ky * 3 + kx) * c_in * c_out + ci * c_out + co];
                        }
                    }
                }
                *output.get_unchecked_mut(out_off + co) = acc;
                co += 1;
            }

            ox += 1;
        }
    }
}

#[inline]
fn depthwise_accumulate_dm(
    out_cell: &mut [f32],
    out_ch_base: usize,
    kernel_data: &[f32],
    kernel_ch_base: usize,
    input_val: f32,
    depth_multiplier: usize,
) {
    match depth_multiplier {
        2 => {
            out_cell[out_ch_base] += input_val * kernel_data[kernel_ch_base];
            out_cell[out_ch_base + 1] += input_val * kernel_data[kernel_ch_base + 1];
        }
        4 => {
            out_cell[out_ch_base] += input_val * kernel_data[kernel_ch_base];
            out_cell[out_ch_base + 1] += input_val * kernel_data[kernel_ch_base + 1];
            out_cell[out_ch_base + 2] += input_val * kernel_data[kernel_ch_base + 2];
            out_cell[out_ch_base + 3] += input_val * kernel_data[kernel_ch_base + 3];
        }
        _ => {
            let mut dm = 0usize;
            while dm + 8 <= depth_multiplier {
                conv_fma_row(
                    &mut out_cell[out_ch_base + dm..out_ch_base + dm + 8],
                    &kernel_data[kernel_ch_base + dm..kernel_ch_base + dm + 8],
                    input_val,
                );
                dm += 8;
            }
            while dm < depth_multiplier {
                out_cell[out_ch_base + dm] += input_val * kernel_data[kernel_ch_base + dm];
                dm += 1;
            }
        }
    }
}

/// FMA: out[i] += kernel[i] * input_val, SIMD-accelerated
#[allow(unsafe_code)]
fn conv_fma_row(out: &mut [f32], kernel: &[f32], input_val: f32) {
    let len = out.len();
    debug_assert_eq!(len, kernel.len());

    if cfg!(miri) || len < 4 {
        conv_fma_scalar(out, kernel, input_val);
        return;
    }

    conv_fma_impl()(out, kernel, input_val);
}

type ConvFmaFn = fn(&mut [f32], &[f32], f32);

#[inline]
fn conv_fma_impl() -> ConvFmaFn {
    static IMPL: OnceLock<ConvFmaFn> = OnceLock::new();
    *IMPL.get_or_init(|| {
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                return conv_fma_neon_dispatch;
            }
        }

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("avx") {
                return conv_fma_avx_dispatch;
            }
            if std::is_x86_feature_detected!("sse") {
                return conv_fma_sse_dispatch;
            }
        }

        conv_fma_scalar
    })
}

#[inline]
fn conv_fma_scalar(out: &mut [f32], kernel: &[f32], input_val: f32) {
    for i in 0..out.len() {
        out[i] += kernel[i] * input_val;
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
fn conv_fma_neon_dispatch(out: &mut [f32], kernel: &[f32], input_val: f32) {
    // SAFETY: selected only after runtime NEON feature detection.
    #[allow(unsafe_code)]
    unsafe {
        conv_fma_neon(out, kernel, input_val);
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline]
fn conv_fma_sse_dispatch(out: &mut [f32], kernel: &[f32], input_val: f32) {
    // SAFETY: selected only after runtime SSE feature detection.
    #[allow(unsafe_code)]
    unsafe {
        conv_fma_sse(out, kernel, input_val);
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline]
fn conv_fma_avx_dispatch(out: &mut [f32], kernel: &[f32], input_val: f32) {
    // SAFETY: selected only after runtime AVX feature detection.
    #[allow(unsafe_code)]
    unsafe {
        conv_fma_avx(out, kernel, input_val);
    }
}

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "neon")]
unsafe fn conv_fma_neon(out: &mut [f32], kernel: &[f32], input_val: f32) {
    use std::arch::aarch64::*;
    let len = out.len();
    let op = out.as_mut_ptr();
    let kp = kernel.as_ptr();
    let v_input = vdupq_n_f32(input_val);
    let mut i = 0usize;
    while i + 4 <= len {
        let o = vld1q_f32(op.add(i));
        let k = vld1q_f32(kp.add(i));
        vst1q_f32(op.add(i), vfmaq_f32(o, k, v_input));
        i += 4;
    }
    while i < len {
        *op.add(i) += *kp.add(i) * input_val;
        i += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "sse")]
unsafe fn conv_fma_sse(out: &mut [f32], kernel: &[f32], input_val: f32) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;
    let len = out.len();
    let op = out.as_mut_ptr();
    let kp = kernel.as_ptr();
    let v_input = _mm_set1_ps(input_val);
    let mut i = 0usize;
    while i + 4 <= len {
        let o = _mm_loadu_ps(op.add(i));
        let k = _mm_loadu_ps(kp.add(i));
        _mm_storeu_ps(op.add(i), _mm_add_ps(o, _mm_mul_ps(k, v_input)));
        i += 4;
    }
    while i < len {
        *op.add(i) += *kp.add(i) * input_val;
        i += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx")]
unsafe fn conv_fma_avx(out: &mut [f32], kernel: &[f32], input_val: f32) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;
    let len = out.len();
    let op = out.as_mut_ptr();
    let kp = kernel.as_ptr();
    let v_input = _mm256_set1_ps(input_val);
    let mut i = 0usize;
    while i + 8 <= len {
        let o = _mm256_loadu_ps(op.add(i));
        let k = _mm256_loadu_ps(kp.add(i));
        _mm256_storeu_ps(op.add(i), _mm256_add_ps(o, _mm256_mul_ps(k, v_input)));
        i += 8;
    }
    if i < len {
        conv_fma_sse(&mut out[i..], &kernel[i..], input_val);
    }
}

/// 3D convolution: input [B, D, H, W, C_in], kernel [KD, KH, KW, C_in, C_out], output [B, OD, OH, OW, C_out]
/// Supports padding and stride in all 3 dimensions.
pub fn conv3d(
    input: &[f32],
    input_shape: &[usize], // [B, D, H, W, C_in]
    kernel: &[f32],
    kernel_shape: &[usize],         // [KD, KH, KW, C_in, C_out]
    stride: (usize, usize, usize),  // (d, h, w)
    padding: (usize, usize, usize), // (d, h, w)
) -> (Vec<f32>, Vec<usize>) {
    assert_eq!(
        input_shape.len(),
        5,
        "input_shape must be [B, D, H, W, C_in]"
    );
    assert_eq!(
        kernel_shape.len(),
        5,
        "kernel_shape must be [KD, KH, KW, C_in, C_out]"
    );

    let (batch, in_d, in_h, in_w, c_in) = (
        input_shape[0],
        input_shape[1],
        input_shape[2],
        input_shape[3],
        input_shape[4],
    );
    let (kd, kh, kw, k_cin, c_out) = (
        kernel_shape[0],
        kernel_shape[1],
        kernel_shape[2],
        kernel_shape[3],
        kernel_shape[4],
    );
    let (stride_d, stride_h, stride_w) = stride;
    let (pad_d, pad_h, pad_w) = padding;

    assert_eq!(c_in, k_cin, "input C_in must match kernel C_in");
    assert!(
        stride_d > 0 && stride_h > 0 && stride_w > 0,
        "strides must be positive"
    );
    assert_eq!(input.len(), batch * in_d * in_h * in_w * c_in);
    assert_eq!(kernel.len(), kd * kh * kw * c_in * c_out);

    let out_d = (in_d + 2 * pad_d - kd) / stride_d + 1;
    let out_h = (in_h + 2 * pad_h - kh) / stride_h + 1;
    let out_w = (in_w + 2 * pad_w - kw) / stride_w + 1;

    let output_shape = vec![batch, out_d, out_h, out_w, c_out];
    let out_spatial = out_d * out_h * out_w;
    let output_len = batch * out_spatial * c_out;

    // im2col + BLAS path: reshape 3D conv into matrix multiply
    // im2col: [out_spatial, kd*kh*kw*c_in]
    // kernel reshaped: [kd*kh*kw*c_in, c_out]
    // output = im2col @ kernel_2d → [out_spatial, c_out]
    #[cfg(feature = "blas")]
    if !cfg!(miri) && batch == 1 {
        let k_spatial = kd * kh * kw;
        let col_k = k_spatial * c_in; // im2col column length
        let mut output = vec![0.0f32; output_len];
        let in_hwc = in_h * in_w * c_in;
        let in_wc = in_w * c_in;

        for b in 0..batch {
            let b_in = b * in_d * in_hwc;
            // Build im2col matrix
            let mut col = vec![0.0f32; out_spatial * col_k];
            let mut row = 0;
            for od in 0..out_d {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let mut col_idx = 0;
                        for fd in 0..kd {
                            let id_raw = od * stride_d + fd;
                            for fh in 0..kh {
                                let ih_raw = oh * stride_h + fh;
                                for fw in 0..kw {
                                    let iw_raw = ow * stride_w + fw;
                                    let in_bounds = id_raw >= pad_d
                                        && id_raw - pad_d < in_d
                                        && ih_raw >= pad_h
                                        && ih_raw - pad_h < in_h
                                        && iw_raw >= pad_w
                                        && iw_raw - pad_w < in_w;
                                    if in_bounds {
                                        let id = id_raw - pad_d;
                                        let ih = ih_raw - pad_h;
                                        let iw = iw_raw - pad_w;
                                        let base = b_in + id * in_hwc + ih * in_wc + iw * c_in;
                                        col[row * col_k + col_idx..row * col_k + col_idx + c_in]
                                            .copy_from_slice(&input[base..base + c_in]);
                                    }
                                    // else: padding zeros (already zeroed)
                                    col_idx += c_in;
                                }
                            }
                        }
                        row += 1;
                    }
                }
            }

            // BLAS: output[b] = col @ kernel_2d
            let b_out = b * out_spatial * c_out;
            super::matmul::blas_sgemm(
                &col,
                kernel,
                &mut output[b_out..b_out + out_spatial * c_out],
                out_spatial,
                col_k,
                c_out,
            );
        }
        return (output, output_shape);
    }

    // Fallback: naive 7-nested-loop implementation
    let mut output = vec![0.0f32; output_len];
    let in_dhwc = in_d * in_h * in_w * c_in;
    let in_hwc = in_h * in_w * c_in;
    let in_wc = in_w * c_in;
    let k_hwcico = kh * kw * c_in * c_out;
    let k_wcico = kw * c_in * c_out;
    let k_cico = c_in * c_out;
    let out_dhwco = out_d * out_h * out_w * c_out;
    let out_hwco = out_h * out_w * c_out;
    let out_wco = out_w * c_out;

    for b in 0..batch {
        let b_in = b * in_dhwc;
        let b_out = b * out_dhwco;
        for od in 0..out_d {
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let out_base = b_out + od * out_hwco + oh * out_wco + ow * c_out;
                    for fd in 0..kd {
                        let id = od * stride_d + fd;
                        if id < pad_d || id - pad_d >= in_d {
                            continue;
                        }
                        let id = id - pad_d;
                        for fh in 0..kh {
                            let ih = oh * stride_h + fh;
                            if ih < pad_h || ih - pad_h >= in_h {
                                continue;
                            }
                            let ih = ih - pad_h;
                            for fw in 0..kw {
                                let iw = ow * stride_w + fw;
                                if iw < pad_w || iw - pad_w >= in_w {
                                    continue;
                                }
                                let iw = iw - pad_w;
                                let in_base = b_in + id * in_hwc + ih * in_wc + iw * c_in;
                                let k_base = fd * k_hwcico + fh * k_wcico + fw * k_cico;
                                for ci in 0..c_in {
                                    let input_val = input[in_base + ci];
                                    let k_offset = k_base + ci * c_out;
                                    for co in 0..c_out {
                                        output[out_base + co] += input_val * kernel[k_offset + co];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    (output, output_shape)
}

// ---------------------------------------------------------------------------
// SIMD depthwise conv2d kernels (depth_multiplier == 1 fast path)
// ---------------------------------------------------------------------------

/// NEON-accelerated depthwise conv row for `depth_multiplier == 1`.
/// Vectorizes across the channel dimension (4 channels per `float32x4_t`).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(unsafe_code)]
unsafe fn depthwise_conv2d_nhwc_row_neon(
    input: &[f32],
    kernel: &[f32],
    bias: Option<&[f32]>,
    plan: DepthwiseConv2dPlan,
    row_idx: usize,
    out_row: &mut [f32],
    activation: Activation,
) {
    use core::arch::aarch64::*;

    let batch_idx = row_idx / plan.out_h;
    let out_y = row_idx % plan.out_h;
    let in_y0 = out_y * plan.stride_h;
    let batch_input_base = batch_idx * plan.in_h * plan.in_w * plan.channels;
    let channels = plan.channels;
    let kh = plan.kernel_h;
    let kw = plan.kernel_w;

    let inp_ptr = input.as_ptr();
    let ker_ptr = kernel.as_ptr();
    let out_ptr = out_row.as_mut_ptr();
    let bias_ptr = bias.map(|b| b.as_ptr());
    let do_relu = matches!(activation, Activation::Relu);
    let zero = vdupq_n_f32(0.0);

    for out_x in 0..plan.out_w {
        let in_x0 = out_x * plan.stride_w;
        let out_base = out_x * channels;
        let mut ch = 0;

        while ch + 16 <= channels {
            unsafe {
                let (mut a0, mut a1, mut a2, mut a3) = if let Some(bp) = bias_ptr {
                    (
                        vld1q_f32(bp.add(ch)),
                        vld1q_f32(bp.add(ch + 4)),
                        vld1q_f32(bp.add(ch + 8)),
                        vld1q_f32(bp.add(ch + 12)),
                    )
                } else {
                    (zero, zero, zero, zero)
                };
                for ky in 0..kh {
                    let in_y = in_y0 + ky;
                    let input_row_base = batch_input_base + (in_y * plan.in_w + in_x0) * channels;
                    let kernel_row_base = (ky * kw) * channels;
                    for kx in 0..kw {
                        let ib = input_row_base + kx * channels + ch;
                        let kb = kernel_row_base + kx * channels + ch;
                        a0 = vfmaq_f32(a0, vld1q_f32(inp_ptr.add(ib)), vld1q_f32(ker_ptr.add(kb)));
                        a1 = vfmaq_f32(
                            a1,
                            vld1q_f32(inp_ptr.add(ib + 4)),
                            vld1q_f32(ker_ptr.add(kb + 4)),
                        );
                        a2 = vfmaq_f32(
                            a2,
                            vld1q_f32(inp_ptr.add(ib + 8)),
                            vld1q_f32(ker_ptr.add(kb + 8)),
                        );
                        a3 = vfmaq_f32(
                            a3,
                            vld1q_f32(inp_ptr.add(ib + 12)),
                            vld1q_f32(ker_ptr.add(kb + 12)),
                        );
                    }
                }
                if do_relu {
                    a0 = vmaxq_f32(a0, zero);
                    a1 = vmaxq_f32(a1, zero);
                    a2 = vmaxq_f32(a2, zero);
                    a3 = vmaxq_f32(a3, zero);
                }
                let ob = out_base + ch;
                vst1q_f32(out_ptr.add(ob), a0);
                vst1q_f32(out_ptr.add(ob + 4), a1);
                vst1q_f32(out_ptr.add(ob + 8), a2);
                vst1q_f32(out_ptr.add(ob + 12), a3);
            }
            ch += 16;
        }
        while ch + 4 <= channels {
            unsafe {
                let mut acc = if let Some(bp) = bias_ptr {
                    vld1q_f32(bp.add(ch))
                } else {
                    vdupq_n_f32(0.0)
                };
                for ky in 0..kh {
                    let in_y = in_y0 + ky;
                    let input_row_base = batch_input_base + (in_y * plan.in_w + in_x0) * channels;
                    let kernel_row_base = (ky * kw) * channels;
                    for kx in 0..kw {
                        let in_off = input_row_base + kx * channels + ch;
                        let k_off = kernel_row_base + kx * channels + ch;
                        acc = vfmaq_f32(
                            acc,
                            vld1q_f32(inp_ptr.add(in_off)),
                            vld1q_f32(ker_ptr.add(k_off)),
                        );
                    }
                }
                if do_relu {
                    acc = vmaxq_f32(acc, zero);
                }
                vst1q_f32(out_ptr.add(out_base + ch), acc);
            }
            ch += 4;
        }
        while ch < channels {
            let mut acc = bias.map_or(0.0, |b| b[ch]);
            for ky in 0..kh {
                let in_y = in_y0 + ky;
                let input_row_base = batch_input_base + (in_y * plan.in_w + in_x0) * channels;
                let kernel_row_base = (ky * kw) * channels;
                for kx in 0..kw {
                    acc += input[input_row_base + kx * channels + ch]
                        * kernel[kernel_row_base + kx * channels + ch];
                }
            }
            if do_relu && acc < 0.0 {
                acc = 0.0;
            }
            out_row[out_base + ch] = acc;
            ch += 1;
        }
    }
    // SiLU cannot be fused cheaply in SIMD (needs exp); apply as post-pass.
    if matches!(activation, Activation::Silu) {
        silu_slice_inplace(out_row);
    }
}

/// AVX+FMA depthwise conv row for `depth_multiplier == 1`.
/// Uses `_mm256_fmadd_ps` for fused multiply-add (Haswell+ / all modern x86).
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx", enable = "fma")]
#[allow(unsafe_code)]
unsafe fn depthwise_conv2d_nhwc_row_avx_fma(
    input: &[f32],
    kernel: &[f32],
    bias: Option<&[f32]>,
    plan: DepthwiseConv2dPlan,
    row_idx: usize,
    out_row: &mut [f32],
    activation: Activation,
) {
    #[cfg(target_arch = "x86")]
    use core::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::*;

    let batch_idx = row_idx / plan.out_h;
    let out_y = row_idx % plan.out_h;
    let in_y0 = out_y * plan.stride_h;
    let batch_input_base = batch_idx * plan.in_h * plan.in_w * plan.channels;
    let channels = plan.channels;
    let kh = plan.kernel_h;
    let kw = plan.kernel_w;

    let inp_ptr = input.as_ptr();
    let ker_ptr = kernel.as_ptr();
    let out_ptr = out_row.as_mut_ptr();

    let bias_ptr = bias.map(|b| b.as_ptr());
    let do_relu = matches!(activation, Activation::Relu);
    let zero = _mm256_setzero_ps();

    for out_x in 0..plan.out_w {
        let in_x0 = out_x * plan.stride_w;
        let out_base = out_x * channels;
        let mut ch = 0;

        // 32-wide: 4 independent accumulators to saturate both FMA ports.
        while ch + 32 <= channels {
            unsafe {
                let (mut a0, mut a1, mut a2, mut a3) = if let Some(bp) = bias_ptr {
                    (
                        _mm256_loadu_ps(bp.add(ch)),
                        _mm256_loadu_ps(bp.add(ch + 8)),
                        _mm256_loadu_ps(bp.add(ch + 16)),
                        _mm256_loadu_ps(bp.add(ch + 24)),
                    )
                } else {
                    (zero, zero, zero, zero)
                };
                let in_row_stride = plan.in_w * channels;
                let kw_stride = channels;
                let mut ky_inp =
                    inp_ptr.add(batch_input_base + (in_y0 * plan.in_w + in_x0) * channels + ch);
                let mut ky_ker = ker_ptr.add(ch);
                for _ky in 0..kh {
                    let mut kx_inp = ky_inp;
                    let mut kx_ker = ky_ker;
                    for _kx in 0..kw {
                        a0 = _mm256_fmadd_ps(_mm256_loadu_ps(kx_inp), _mm256_loadu_ps(kx_ker), a0);
                        a1 = _mm256_fmadd_ps(
                            _mm256_loadu_ps(kx_inp.add(8)),
                            _mm256_loadu_ps(kx_ker.add(8)),
                            a1,
                        );
                        a2 = _mm256_fmadd_ps(
                            _mm256_loadu_ps(kx_inp.add(16)),
                            _mm256_loadu_ps(kx_ker.add(16)),
                            a2,
                        );
                        a3 = _mm256_fmadd_ps(
                            _mm256_loadu_ps(kx_inp.add(24)),
                            _mm256_loadu_ps(kx_ker.add(24)),
                            a3,
                        );
                        kx_inp = kx_inp.add(kw_stride);
                        kx_ker = kx_ker.add(kw_stride);
                    }
                    ky_inp = ky_inp.add(in_row_stride);
                    ky_ker = ky_ker.add(kw * kw_stride);
                }
                if do_relu {
                    a0 = _mm256_max_ps(a0, zero);
                    a1 = _mm256_max_ps(a1, zero);
                    a2 = _mm256_max_ps(a2, zero);
                    a3 = _mm256_max_ps(a3, zero);
                }
                let ob = out_base + ch;
                _mm256_storeu_ps(out_ptr.add(ob), a0);
                _mm256_storeu_ps(out_ptr.add(ob + 8), a1);
                _mm256_storeu_ps(out_ptr.add(ob + 16), a2);
                _mm256_storeu_ps(out_ptr.add(ob + 24), a3);
            }
            ch += 32;
        }
        // 16-wide: 2 accumulators.
        while ch + 16 <= channels {
            unsafe {
                let (mut a0, mut a1) = if let Some(bp) = bias_ptr {
                    (_mm256_loadu_ps(bp.add(ch)), _mm256_loadu_ps(bp.add(ch + 8)))
                } else {
                    (zero, zero)
                };
                let in_row_stride = plan.in_w * channels;
                let kw_stride = channels;
                let mut ky_inp =
                    inp_ptr.add(batch_input_base + (in_y0 * plan.in_w + in_x0) * channels + ch);
                let mut ky_ker = ker_ptr.add(ch);
                for _ky in 0..kh {
                    let mut kx_inp = ky_inp;
                    let mut kx_ker = ky_ker;
                    for _kx in 0..kw {
                        a0 = _mm256_fmadd_ps(_mm256_loadu_ps(kx_inp), _mm256_loadu_ps(kx_ker), a0);
                        a1 = _mm256_fmadd_ps(
                            _mm256_loadu_ps(kx_inp.add(8)),
                            _mm256_loadu_ps(kx_ker.add(8)),
                            a1,
                        );
                        kx_inp = kx_inp.add(kw_stride);
                        kx_ker = kx_ker.add(kw_stride);
                    }
                    ky_inp = ky_inp.add(in_row_stride);
                    ky_ker = ky_ker.add(kw * kw_stride);
                }
                if do_relu {
                    a0 = _mm256_max_ps(a0, zero);
                    a1 = _mm256_max_ps(a1, zero);
                }
                _mm256_storeu_ps(out_ptr.add(out_base + ch), a0);
                _mm256_storeu_ps(out_ptr.add(out_base + ch + 8), a1);
            }
            ch += 16;
        }
        // 8-wide: single accumulator.
        while ch + 8 <= channels {
            unsafe {
                let mut acc = if let Some(bp) = bias_ptr {
                    _mm256_loadu_ps(bp.add(ch))
                } else {
                    _mm256_setzero_ps()
                };
                let in_row_stride = plan.in_w * channels;
                let kw_stride = channels;
                let mut ky_inp =
                    inp_ptr.add(batch_input_base + (in_y0 * plan.in_w + in_x0) * channels + ch);
                let mut ky_ker = ker_ptr.add(ch);
                for _ky in 0..kh {
                    let mut kx_inp = ky_inp;
                    let mut kx_ker = ky_ker;
                    for _kx in 0..kw {
                        acc =
                            _mm256_fmadd_ps(_mm256_loadu_ps(kx_inp), _mm256_loadu_ps(kx_ker), acc);
                        kx_inp = kx_inp.add(kw_stride);
                        kx_ker = kx_ker.add(kw_stride);
                    }
                    ky_inp = ky_inp.add(in_row_stride);
                    ky_ker = ky_ker.add(kw * kw_stride);
                }
                if do_relu {
                    acc = _mm256_max_ps(acc, zero);
                }
                _mm256_storeu_ps(out_ptr.add(out_base + ch), acc);
            }
            ch += 8;
        }
        // Scalar tail.
        while ch < channels {
            let mut acc = bias.map_or(0.0, |b| b[ch]);
            for ky in 0..kh {
                let in_y = in_y0 + ky;
                let input_row_base = batch_input_base + (in_y * plan.in_w + in_x0) * channels;
                let kernel_row_base = (ky * kw) * channels;
                for kx in 0..kw {
                    acc += input[input_row_base + kx * channels + ch]
                        * kernel[kernel_row_base + kx * channels + ch];
                }
            }
            if do_relu && acc < 0.0 {
                acc = 0.0;
            }
            out_row[out_base + ch] = acc;
            ch += 1;
        }
    }
    // SiLU cannot be fused cheaply in SIMD (needs exp); apply as post-pass.
    if matches!(activation, Activation::Silu) {
        silu_slice_inplace(out_row);
    }
}

/// Step A2: AVX-512 depthwise conv row for `depth_multiplier == 1`.
/// Vectorizes across the channel dimension (16 channels per `__m512`).
/// Wider tiles than AVX+FMA: 128-wide (8 ZMM), 64-wide (4 ZMM),
/// 32-wide (2 ZMM), 16-wide (1 ZMM), then scalar tail.
///
/// Zen 4 AVX-512 is double-pumped (2× 256-bit FMA internally) so peak
/// FLOPS equal to AVX+FMA, but uop count halves — same work in fewer
/// front-end slots. DW has no broadcast bottleneck (each lane is an
/// independent channel), so intrinsics saturate the pipeline without
/// needing inline asm.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
unsafe fn depthwise_conv2d_nhwc_row_avx512(
    input: &[f32],
    kernel: &[f32],
    bias: Option<&[f32]>,
    plan: DepthwiseConv2dPlan,
    row_idx: usize,
    out_row: &mut [f32],
    activation: Activation,
) {
    #[cfg(target_arch = "x86")]
    use core::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::*;

    let batch_idx = row_idx / plan.out_h;
    let out_y = row_idx % plan.out_h;
    let in_y0 = out_y * plan.stride_h;
    let batch_input_base = batch_idx * plan.in_h * plan.in_w * plan.channels;
    let channels = plan.channels;
    let kh = plan.kernel_h;
    let kw = plan.kernel_w;

    let inp_ptr = input.as_ptr();
    let ker_ptr = kernel.as_ptr();
    let out_ptr = out_row.as_mut_ptr();

    let bias_ptr = bias.map(|b| b.as_ptr());
    let do_relu = matches!(activation, Activation::Relu);
    let zero = _mm512_setzero_ps();

    for out_x in 0..plan.out_w {
        let in_x0 = out_x * plan.stride_w;
        let out_base = out_x * channels;
        let mut ch = 0;

        // 128-wide (8 ZMM) — saturate deep channel counts (tracker c=320,672).
        while ch + 128 <= channels {
            let (mut a0, mut a1, mut a2, mut a3, mut a4, mut a5, mut a6, mut a7) =
                if let Some(bp) = bias_ptr {
                    (
                        _mm512_loadu_ps(bp.add(ch)),
                        _mm512_loadu_ps(bp.add(ch + 16)),
                        _mm512_loadu_ps(bp.add(ch + 32)),
                        _mm512_loadu_ps(bp.add(ch + 48)),
                        _mm512_loadu_ps(bp.add(ch + 64)),
                        _mm512_loadu_ps(bp.add(ch + 80)),
                        _mm512_loadu_ps(bp.add(ch + 96)),
                        _mm512_loadu_ps(bp.add(ch + 112)),
                    )
                } else {
                    (zero, zero, zero, zero, zero, zero, zero, zero)
                };
            let in_row_stride = plan.in_w * channels;
            let kw_stride = channels;
            let mut ky_inp =
                inp_ptr.add(batch_input_base + (in_y0 * plan.in_w + in_x0) * channels + ch);
            let mut ky_ker = ker_ptr.add(ch);
            for _ky in 0..kh {
                let mut kx_inp = ky_inp;
                let mut kx_ker = ky_ker;
                for _kx in 0..kw {
                    a0 = _mm512_fmadd_ps(_mm512_loadu_ps(kx_inp), _mm512_loadu_ps(kx_ker), a0);
                    a1 = _mm512_fmadd_ps(
                        _mm512_loadu_ps(kx_inp.add(16)),
                        _mm512_loadu_ps(kx_ker.add(16)),
                        a1,
                    );
                    a2 = _mm512_fmadd_ps(
                        _mm512_loadu_ps(kx_inp.add(32)),
                        _mm512_loadu_ps(kx_ker.add(32)),
                        a2,
                    );
                    a3 = _mm512_fmadd_ps(
                        _mm512_loadu_ps(kx_inp.add(48)),
                        _mm512_loadu_ps(kx_ker.add(48)),
                        a3,
                    );
                    a4 = _mm512_fmadd_ps(
                        _mm512_loadu_ps(kx_inp.add(64)),
                        _mm512_loadu_ps(kx_ker.add(64)),
                        a4,
                    );
                    a5 = _mm512_fmadd_ps(
                        _mm512_loadu_ps(kx_inp.add(80)),
                        _mm512_loadu_ps(kx_ker.add(80)),
                        a5,
                    );
                    a6 = _mm512_fmadd_ps(
                        _mm512_loadu_ps(kx_inp.add(96)),
                        _mm512_loadu_ps(kx_ker.add(96)),
                        a6,
                    );
                    a7 = _mm512_fmadd_ps(
                        _mm512_loadu_ps(kx_inp.add(112)),
                        _mm512_loadu_ps(kx_ker.add(112)),
                        a7,
                    );
                    kx_inp = kx_inp.add(kw_stride);
                    kx_ker = kx_ker.add(kw_stride);
                }
                ky_inp = ky_inp.add(in_row_stride);
                ky_ker = ky_ker.add(kw * kw_stride);
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
            let ob = out_base + ch;
            _mm512_storeu_ps(out_ptr.add(ob), a0);
            _mm512_storeu_ps(out_ptr.add(ob + 16), a1);
            _mm512_storeu_ps(out_ptr.add(ob + 32), a2);
            _mm512_storeu_ps(out_ptr.add(ob + 48), a3);
            _mm512_storeu_ps(out_ptr.add(ob + 64), a4);
            _mm512_storeu_ps(out_ptr.add(ob + 80), a5);
            _mm512_storeu_ps(out_ptr.add(ob + 96), a6);
            _mm512_storeu_ps(out_ptr.add(ob + 112), a7);
            ch += 128;
        }

        // 64-wide (4 ZMM).
        while ch + 64 <= channels {
            let (mut a0, mut a1, mut a2, mut a3) = if let Some(bp) = bias_ptr {
                (
                    _mm512_loadu_ps(bp.add(ch)),
                    _mm512_loadu_ps(bp.add(ch + 16)),
                    _mm512_loadu_ps(bp.add(ch + 32)),
                    _mm512_loadu_ps(bp.add(ch + 48)),
                )
            } else {
                (zero, zero, zero, zero)
            };
            let in_row_stride = plan.in_w * channels;
            let kw_stride = channels;
            let mut ky_inp =
                inp_ptr.add(batch_input_base + (in_y0 * plan.in_w + in_x0) * channels + ch);
            let mut ky_ker = ker_ptr.add(ch);
            for _ky in 0..kh {
                let mut kx_inp = ky_inp;
                let mut kx_ker = ky_ker;
                for _kx in 0..kw {
                    a0 = _mm512_fmadd_ps(_mm512_loadu_ps(kx_inp), _mm512_loadu_ps(kx_ker), a0);
                    a1 = _mm512_fmadd_ps(
                        _mm512_loadu_ps(kx_inp.add(16)),
                        _mm512_loadu_ps(kx_ker.add(16)),
                        a1,
                    );
                    a2 = _mm512_fmadd_ps(
                        _mm512_loadu_ps(kx_inp.add(32)),
                        _mm512_loadu_ps(kx_ker.add(32)),
                        a2,
                    );
                    a3 = _mm512_fmadd_ps(
                        _mm512_loadu_ps(kx_inp.add(48)),
                        _mm512_loadu_ps(kx_ker.add(48)),
                        a3,
                    );
                    kx_inp = kx_inp.add(kw_stride);
                    kx_ker = kx_ker.add(kw_stride);
                }
                ky_inp = ky_inp.add(in_row_stride);
                ky_ker = ky_ker.add(kw * kw_stride);
            }
            if do_relu {
                a0 = _mm512_max_ps(a0, zero);
                a1 = _mm512_max_ps(a1, zero);
                a2 = _mm512_max_ps(a2, zero);
                a3 = _mm512_max_ps(a3, zero);
            }
            let ob = out_base + ch;
            _mm512_storeu_ps(out_ptr.add(ob), a0);
            _mm512_storeu_ps(out_ptr.add(ob + 16), a1);
            _mm512_storeu_ps(out_ptr.add(ob + 32), a2);
            _mm512_storeu_ps(out_ptr.add(ob + 48), a3);
            ch += 64;
        }

        // 32-wide (2 ZMM).
        while ch + 32 <= channels {
            let (mut a0, mut a1) = if let Some(bp) = bias_ptr {
                (
                    _mm512_loadu_ps(bp.add(ch)),
                    _mm512_loadu_ps(bp.add(ch + 16)),
                )
            } else {
                (zero, zero)
            };
            let in_row_stride = plan.in_w * channels;
            let kw_stride = channels;
            let mut ky_inp =
                inp_ptr.add(batch_input_base + (in_y0 * plan.in_w + in_x0) * channels + ch);
            let mut ky_ker = ker_ptr.add(ch);
            for _ky in 0..kh {
                let mut kx_inp = ky_inp;
                let mut kx_ker = ky_ker;
                for _kx in 0..kw {
                    a0 = _mm512_fmadd_ps(_mm512_loadu_ps(kx_inp), _mm512_loadu_ps(kx_ker), a0);
                    a1 = _mm512_fmadd_ps(
                        _mm512_loadu_ps(kx_inp.add(16)),
                        _mm512_loadu_ps(kx_ker.add(16)),
                        a1,
                    );
                    kx_inp = kx_inp.add(kw_stride);
                    kx_ker = kx_ker.add(kw_stride);
                }
                ky_inp = ky_inp.add(in_row_stride);
                ky_ker = ky_ker.add(kw * kw_stride);
            }
            if do_relu {
                a0 = _mm512_max_ps(a0, zero);
                a1 = _mm512_max_ps(a1, zero);
            }
            let ob = out_base + ch;
            _mm512_storeu_ps(out_ptr.add(ob), a0);
            _mm512_storeu_ps(out_ptr.add(ob + 16), a1);
            ch += 32;
        }

        // 16-wide (1 ZMM).
        while ch + 16 <= channels {
            let mut acc = if let Some(bp) = bias_ptr {
                _mm512_loadu_ps(bp.add(ch))
            } else {
                zero
            };
            let in_row_stride = plan.in_w * channels;
            let kw_stride = channels;
            let mut ky_inp =
                inp_ptr.add(batch_input_base + (in_y0 * plan.in_w + in_x0) * channels + ch);
            let mut ky_ker = ker_ptr.add(ch);
            for _ky in 0..kh {
                let mut kx_inp = ky_inp;
                let mut kx_ker = ky_ker;
                for _kx in 0..kw {
                    acc = _mm512_fmadd_ps(_mm512_loadu_ps(kx_inp), _mm512_loadu_ps(kx_ker), acc);
                    kx_inp = kx_inp.add(kw_stride);
                    kx_ker = kx_ker.add(kw_stride);
                }
                ky_inp = ky_inp.add(in_row_stride);
                ky_ker = ky_ker.add(kw * kw_stride);
            }
            if do_relu {
                acc = _mm512_max_ps(acc, zero);
            }
            _mm512_storeu_ps(out_ptr.add(out_base + ch), acc);
            ch += 16;
        }

        // 8-wide tail via AVX (when channels % 16 != 0 but %8==0).
        {
            use core::arch::x86_64::{
                _mm256_fmadd_ps, _mm256_loadu_ps, _mm256_max_ps, _mm256_setzero_ps,
                _mm256_storeu_ps,
            };
            let zy = _mm256_setzero_ps();
            while ch + 8 <= channels {
                let mut acc = if let Some(bp) = bias_ptr {
                    _mm256_loadu_ps(bp.add(ch))
                } else {
                    zy
                };
                let in_row_stride = plan.in_w * channels;
                let kw_stride = channels;
                let mut ky_inp =
                    inp_ptr.add(batch_input_base + (in_y0 * plan.in_w + in_x0) * channels + ch);
                let mut ky_ker = ker_ptr.add(ch);
                for _ky in 0..kh {
                    let mut kx_inp = ky_inp;
                    let mut kx_ker = ky_ker;
                    for _kx in 0..kw {
                        acc =
                            _mm256_fmadd_ps(_mm256_loadu_ps(kx_inp), _mm256_loadu_ps(kx_ker), acc);
                        kx_inp = kx_inp.add(kw_stride);
                        kx_ker = kx_ker.add(kw_stride);
                    }
                    ky_inp = ky_inp.add(in_row_stride);
                    ky_ker = ky_ker.add(kw * kw_stride);
                }
                if do_relu {
                    acc = _mm256_max_ps(acc, zy);
                }
                _mm256_storeu_ps(out_ptr.add(out_base + ch), acc);
                ch += 8;
            }
        }

        // Scalar tail.
        while ch < channels {
            let mut acc = bias.map_or(0.0, |b| b[ch]);
            for ky in 0..kh {
                let in_y = in_y0 + ky;
                let input_row_base = batch_input_base + (in_y * plan.in_w + in_x0) * channels;
                let kernel_row_base = (ky * kw) * channels;
                for kx in 0..kw {
                    acc += input[input_row_base + kx * channels + ch]
                        * kernel[kernel_row_base + kx * channels + ch];
                }
            }
            if do_relu && acc < 0.0 {
                acc = 0.0;
            }
            out_row[out_base + ch] = acc;
            ch += 1;
        }
    }
    // SiLU cannot be fused cheaply in SIMD (needs exp); apply as post-pass.
    if matches!(activation, Activation::Silu) {
        silu_slice_inplace(out_row);
    }
}

/// AVX-accelerated depthwise conv row for `depth_multiplier == 1` (no FMA fallback).
/// Vectorizes across the channel dimension (8 channels per `__m256`).
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx")]
#[allow(unsafe_code)]
unsafe fn depthwise_conv2d_nhwc_row_avx(
    input: &[f32],
    kernel: &[f32],
    bias: Option<&[f32]>,
    plan: DepthwiseConv2dPlan,
    row_idx: usize,
    out_row: &mut [f32],
    activation: Activation,
) {
    #[cfg(target_arch = "x86")]
    use core::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::*;

    let batch_idx = row_idx / plan.out_h;
    let out_y = row_idx % plan.out_h;
    let in_y0 = out_y * plan.stride_h;
    let batch_input_base = batch_idx * plan.in_h * plan.in_w * plan.channels;
    let channels = plan.channels;
    let kh = plan.kernel_h;
    let kw = plan.kernel_w;

    let inp_ptr = input.as_ptr();
    let ker_ptr = kernel.as_ptr();
    let out_ptr = out_row.as_mut_ptr();

    let bias_ptr = bias.map(|b| b.as_ptr());
    let do_relu = matches!(activation, Activation::Relu);
    let zero = _mm256_setzero_ps();

    for out_x in 0..plan.out_w {
        let in_x0 = out_x * plan.stride_w;
        let out_base = out_x * channels;
        let mut ch = 0;

        while ch + 32 <= channels {
            unsafe {
                let (mut a0, mut a1, mut a2, mut a3) = if let Some(bp) = bias_ptr {
                    (
                        _mm256_loadu_ps(bp.add(ch)),
                        _mm256_loadu_ps(bp.add(ch + 8)),
                        _mm256_loadu_ps(bp.add(ch + 16)),
                        _mm256_loadu_ps(bp.add(ch + 24)),
                    )
                } else {
                    (zero, zero, zero, zero)
                };
                for ky in 0..kh {
                    let in_y = in_y0 + ky;
                    let input_row_base = batch_input_base + (in_y * plan.in_w + in_x0) * channels;
                    let kernel_row_base = (ky * kw) * channels;
                    for kx in 0..kw {
                        let ib = input_row_base + kx * channels + ch;
                        let kb = kernel_row_base + kx * channels + ch;
                        a0 = _mm256_add_ps(
                            a0,
                            _mm256_mul_ps(
                                _mm256_loadu_ps(inp_ptr.add(ib)),
                                _mm256_loadu_ps(ker_ptr.add(kb)),
                            ),
                        );
                        a1 = _mm256_add_ps(
                            a1,
                            _mm256_mul_ps(
                                _mm256_loadu_ps(inp_ptr.add(ib + 8)),
                                _mm256_loadu_ps(ker_ptr.add(kb + 8)),
                            ),
                        );
                        a2 = _mm256_add_ps(
                            a2,
                            _mm256_mul_ps(
                                _mm256_loadu_ps(inp_ptr.add(ib + 16)),
                                _mm256_loadu_ps(ker_ptr.add(kb + 16)),
                            ),
                        );
                        a3 = _mm256_add_ps(
                            a3,
                            _mm256_mul_ps(
                                _mm256_loadu_ps(inp_ptr.add(ib + 24)),
                                _mm256_loadu_ps(ker_ptr.add(kb + 24)),
                            ),
                        );
                    }
                }
                if do_relu {
                    a0 = _mm256_max_ps(a0, zero);
                    a1 = _mm256_max_ps(a1, zero);
                    a2 = _mm256_max_ps(a2, zero);
                    a3 = _mm256_max_ps(a3, zero);
                }
                let ob = out_base + ch;
                _mm256_storeu_ps(out_ptr.add(ob), a0);
                _mm256_storeu_ps(out_ptr.add(ob + 8), a1);
                _mm256_storeu_ps(out_ptr.add(ob + 16), a2);
                _mm256_storeu_ps(out_ptr.add(ob + 24), a3);
            }
            ch += 32;
        }
        while ch + 16 <= channels {
            unsafe {
                let (mut a0, mut a1) = if let Some(bp) = bias_ptr {
                    (_mm256_loadu_ps(bp.add(ch)), _mm256_loadu_ps(bp.add(ch + 8)))
                } else {
                    (zero, zero)
                };
                for ky in 0..kh {
                    let in_y = in_y0 + ky;
                    let input_row_base = batch_input_base + (in_y * plan.in_w + in_x0) * channels;
                    let kernel_row_base = (ky * kw) * channels;
                    for kx in 0..kw {
                        let ib = input_row_base + kx * channels + ch;
                        let kb = kernel_row_base + kx * channels + ch;
                        a0 = _mm256_add_ps(
                            a0,
                            _mm256_mul_ps(
                                _mm256_loadu_ps(inp_ptr.add(ib)),
                                _mm256_loadu_ps(ker_ptr.add(kb)),
                            ),
                        );
                        a1 = _mm256_add_ps(
                            a1,
                            _mm256_mul_ps(
                                _mm256_loadu_ps(inp_ptr.add(ib + 8)),
                                _mm256_loadu_ps(ker_ptr.add(kb + 8)),
                            ),
                        );
                    }
                }
                if do_relu {
                    a0 = _mm256_max_ps(a0, zero);
                    a1 = _mm256_max_ps(a1, zero);
                }
                _mm256_storeu_ps(out_ptr.add(out_base + ch), a0);
                _mm256_storeu_ps(out_ptr.add(out_base + ch + 8), a1);
            }
            ch += 16;
        }
        while ch + 8 <= channels {
            unsafe {
                let mut acc = if let Some(bp) = bias_ptr {
                    _mm256_loadu_ps(bp.add(ch))
                } else {
                    _mm256_setzero_ps()
                };
                for ky in 0..kh {
                    let in_y = in_y0 + ky;
                    let input_row_base = batch_input_base + (in_y * plan.in_w + in_x0) * channels;
                    let kernel_row_base = (ky * kw) * channels;
                    for kx in 0..kw {
                        let in_off = input_row_base + kx * channels + ch;
                        let k_off = kernel_row_base + kx * channels + ch;
                        acc = _mm256_add_ps(
                            acc,
                            _mm256_mul_ps(
                                _mm256_loadu_ps(inp_ptr.add(in_off)),
                                _mm256_loadu_ps(ker_ptr.add(k_off)),
                            ),
                        );
                    }
                }
                if do_relu {
                    acc = _mm256_max_ps(acc, zero);
                }
                _mm256_storeu_ps(out_ptr.add(out_base + ch), acc);
            }
            ch += 8;
        }
        while ch < channels {
            let mut acc = bias.map_or(0.0, |b| b[ch]);
            for ky in 0..kh {
                let in_y = in_y0 + ky;
                let input_row_base = batch_input_base + (in_y * plan.in_w + in_x0) * channels;
                let kernel_row_base = (ky * kw) * channels;
                for kx in 0..kw {
                    acc += input[input_row_base + kx * channels + ch]
                        * kernel[kernel_row_base + kx * channels + ch];
                }
            }
            if do_relu && acc < 0.0 {
                acc = 0.0;
            }
            out_row[out_base + ch] = acc;
            ch += 1;
        }
    }
    // SiLU cannot be fused cheaply in SIMD (needs exp); apply as post-pass.
    if matches!(activation, Activation::Silu) {
        silu_slice_inplace(out_row);
    }
}

/// SSE-accelerated depthwise conv row for `depth_multiplier == 1`.
/// Vectorizes across the channel dimension (4 channels per `__m128`).
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse")]
#[allow(unsafe_code)]
unsafe fn depthwise_conv2d_nhwc_row_sse(
    input: &[f32],
    kernel: &[f32],
    bias: Option<&[f32]>,
    plan: DepthwiseConv2dPlan,
    row_idx: usize,
    out_row: &mut [f32],
    activation: Activation,
) {
    #[cfg(target_arch = "x86")]
    use core::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::*;

    let batch_idx = row_idx / plan.out_h;
    let out_y = row_idx % plan.out_h;
    let in_y0 = out_y * plan.stride_h;
    let batch_input_base = batch_idx * plan.in_h * plan.in_w * plan.channels;
    let channels = plan.channels;
    let simd_end = channels & !3; // round down to multiple of 4
    let kh = plan.kernel_h;
    let kw = plan.kernel_w;

    let inp_ptr = input.as_ptr();
    let ker_ptr = kernel.as_ptr();
    let out_ptr = out_row.as_mut_ptr();

    let do_relu = matches!(activation, Activation::Relu);
    let zero = _mm_setzero_ps();

    for out_x in 0..plan.out_w {
        let in_x0 = out_x * plan.stride_w;
        let out_base = out_x * channels;

        // Process 4 channels at a time with accumulator in register.
        let mut ch = 0;
        while ch + 4 <= simd_end {
            // SAFETY: ch + 4 <= channels, all offsets bounded by plan dims.
            unsafe {
                let mut acc = if let Some(b) = bias {
                    _mm_loadu_ps(b.as_ptr().add(ch))
                } else {
                    zero
                };

                for ky in 0..kh {
                    let in_y = in_y0 + ky;
                    let input_row_base = batch_input_base + (in_y * plan.in_w + in_x0) * channels;
                    let kernel_row_base = (ky * kw) * channels;

                    for kx in 0..kw {
                        let in_off = input_row_base + kx * channels + ch;
                        let k_off = kernel_row_base + kx * channels + ch;
                        let inp = _mm_loadu_ps(inp_ptr.add(in_off));
                        let ker = _mm_loadu_ps(ker_ptr.add(k_off));
                        acc = _mm_add_ps(acc, _mm_mul_ps(inp, ker));
                    }
                }

                if do_relu {
                    acc = _mm_max_ps(acc, zero);
                }
                _mm_storeu_ps(out_ptr.add(out_base + ch), acc);
            }
            ch += 4;
        }
        // Scalar tail.
        while ch < channels {
            let mut acc = bias.map_or(0.0, |b| b[ch]);
            for ky in 0..kh {
                let in_y = in_y0 + ky;
                let input_row_base = batch_input_base + (in_y * plan.in_w + in_x0) * channels;
                let kernel_row_base = (ky * kw) * channels;
                for kx in 0..kw {
                    acc += input[input_row_base + kx * channels + ch]
                        * kernel[kernel_row_base + kx * channels + ch];
                }
            }
            if do_relu && acc < 0.0 {
                acc = 0.0;
            }
            out_row[out_base + ch] = acc;
            ch += 1;
        }
    }
    // SiLU cannot be fused cheaply in SIMD (needs exp); apply as post-pass.
    if matches!(activation, Activation::Silu) {
        silu_slice_inplace(out_row);
    }
}

fn depthwise_conv2d_nhwc_row(
    input: &[f32],
    kernel: &[f32],
    bias: Option<&[f32]>,
    plan: DepthwiseConv2dPlan,
    row_idx: usize,
    out_row: &mut [f32],
    activation: Activation,
) {
    // SIMD fast path for depth_multiplier == 1 (standard depthwise conv).
    // When dm=1, out_channels == channels and the kernel layout simplifies to
    // [KH, KW, C] — contiguous channel data enables vectorization.
    if plan.depth_multiplier == 1 && plan.out_channels >= 4 && !cfg!(miri) {
        #[cfg(target_arch = "aarch64")]
        if std::arch::is_aarch64_feature_detected!("neon") {
            // SAFETY: NEON detected, pointers bounded by plan dimensions validated at
            // function entry. Each output element written exactly once.
            #[allow(unsafe_code)]
            unsafe {
                depthwise_conv2d_nhwc_row_neon(
                    input, kernel, bias, plan, row_idx, out_row, activation,
                );
            }
            return;
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            // Step A2: AVX-512 DW row kernel when available. Kill-switch
            // `YSCV_AVX512_DW_OFF=1` cached via OnceLock so the env read
            // doesn't repeat per DW row (Session 10 polish — was 0.15%
            // cycles / ~6 µs @ 6T).
            if is_x86_feature_detected!("avx512f") && !avx512_dw_disabled() {
                // SAFETY: AVX-512F detected, bounds guaranteed.
                #[allow(unsafe_code)]
                unsafe {
                    depthwise_conv2d_nhwc_row_avx512(
                        input, kernel, bias, plan, row_idx, out_row, activation,
                    );
                }
                return;
            }
            if is_x86_feature_detected!("avx") && is_x86_feature_detected!("fma") {
                // SAFETY: AVX+FMA detected, same bounds guarantees.
                #[allow(unsafe_code)]
                unsafe {
                    depthwise_conv2d_nhwc_row_avx_fma(
                        input, kernel, bias, plan, row_idx, out_row, activation,
                    );
                }
                return;
            }
            if is_x86_feature_detected!("avx") {
                // SAFETY: AVX detected (no FMA), same bounds guarantees.
                #[allow(unsafe_code)]
                unsafe {
                    depthwise_conv2d_nhwc_row_avx(
                        input, kernel, bias, plan, row_idx, out_row, activation,
                    );
                }
                return;
            }
            if is_x86_feature_detected!("sse") {
                // SAFETY: SSE detected, same bounds guarantees.
                #[allow(unsafe_code)]
                unsafe {
                    depthwise_conv2d_nhwc_row_sse(
                        input, kernel, bias, plan, row_idx, out_row, activation,
                    );
                }
                return;
            }
        }
    }

    // Scalar fallback (handles depth_multiplier > 1 and all other cases).
    let batch_idx = row_idx / plan.out_h;
    let out_y = row_idx % plan.out_h;
    let in_y0 = out_y * plan.stride_h;
    let batch_input_base = batch_idx * plan.in_h * plan.in_w * plan.channels;
    let do_relu = matches!(activation, Activation::Relu);

    for out_x in 0..plan.out_w {
        let in_x0 = out_x * plan.stride_w;
        let out_cell_base = out_x * plan.out_channels;

        for out_channel in 0..plan.out_channels {
            let mut acc = bias.map_or(0.0, |bias_values| bias_values[out_channel]);
            let in_channel = out_channel / plan.depth_multiplier;
            let depth_index = out_channel % plan.depth_multiplier;

            for ky in 0..plan.kernel_h {
                let in_y = in_y0 + ky;
                let input_row_base = batch_input_base + (in_y * plan.in_w + in_x0) * plan.channels;
                let kernel_row_base = ky * plan.kernel_w * plan.channels * plan.depth_multiplier;

                for kx in 0..plan.kernel_w {
                    let input_value = input[input_row_base + kx * plan.channels + in_channel];
                    let kernel_index = kernel_row_base
                        + kx * plan.channels * plan.depth_multiplier
                        + in_channel * plan.depth_multiplier
                        + depth_index;
                    acc += input_value * kernel[kernel_index];
                }
            }

            if do_relu && acc < 0.0 {
                acc = 0.0;
            }
            out_row[out_cell_base + out_channel] = acc;
        }
    }
    // SiLU cannot be fused cheaply (needs exp); apply as post-pass.
    if matches!(activation, Activation::Silu) {
        silu_slice_inplace(out_row);
    }
}

// ---------------------------------------------------------------------------
// im2col + BLAS GEMM fast path for conv2d
// ---------------------------------------------------------------------------

/// Flatten each [kH, kW, C_in] input patch into a row of the im2col matrix.
///
/// Output `col` has shape [out_h * out_w, kH * kW * C_in] (row-major).
/// The input is NHWC layout (batch dimension already stripped by caller).
/// Indirect convolution: handles padding without allocating padded tensor.
/// Uses a zero buffer for out-of-bounds input positions.
/// This replaces pad_nhwc() + conv2d_nhwc_row() for padded group=1 Conv.
#[allow(unsafe_code)]
pub fn conv2d_nhwc_indirect_padded(
    input: &Tensor,
    kernel: &Tensor,
    bias: Option<&Tensor>,
    stride_h: usize,
    stride_w: usize,
    pad_top: usize,
    pad_left: usize,
    pad_bottom: usize,
    pad_right: usize,
    activation: Activation,
) -> Result<Tensor, KernelError> {
    let in_shape = input.shape();
    if in_shape.len() != 4 || kernel.shape().len() != 4 {
        return Err(KernelError::InvalidConvRank {
            input_rank: in_shape.len(),
            kernel_rank: kernel.shape().len(),
        });
    }
    let (batch, in_h, in_w, c_in) = (in_shape[0], in_shape[1], in_shape[2], in_shape[3]);
    let k_shape = kernel.shape();
    let (kh, kw, _k_cin, c_out) = (k_shape[0], k_shape[1], k_shape[2], k_shape[3]);

    let out_h = (in_h + pad_top + pad_bottom - kh) / stride_h + 1;
    let out_w = (in_w + pad_left + pad_right - kw) / stride_w + 1;
    let output_len = batch * out_h * out_w * c_out;
    let out_row_len = out_w * c_out;

    let mut output = AlignedVec::<f32>::uninitialized(output_len);
    let in_data = input.data();
    let ker_data = kernel.data();
    let bias_data = bias.map(Tensor::data);

    // Zero buffer for padded positions
    let zero_pixel = vec![0.0f32; c_in];

    for b in 0..batch {
        let batch_in_base = b * in_h * in_w * c_in;
        for oy in 0..out_h {
            let out_row_start = (b * out_h + oy) * out_row_len;
            let out_row = &mut output[out_row_start..out_row_start + out_row_len];

            for ox in 0..out_w {
                let out_cell = &mut out_row[ox * c_out..(ox + 1) * c_out];

                // Init with bias
                if let Some(bv) = bias_data {
                    out_cell.copy_from_slice(&bv[..c_out]);
                } else {
                    out_cell.fill(0.0);
                }

                // Accumulate kernel positions with inline padding check
                for ky in 0..kh {
                    let iy = oy * stride_h + ky;
                    let in_y = iy as isize - pad_top as isize;

                    for kx in 0..kw {
                        let ix = ox * stride_w + kx;
                        let in_x = ix as isize - pad_left as isize;

                        let input_pixel = if in_y >= 0
                            && (in_y as usize) < in_h
                            && in_x >= 0
                            && (in_x as usize) < in_w
                        {
                            let offset =
                                batch_in_base + (in_y as usize * in_w + in_x as usize) * c_in;
                            &in_data[offset..offset + c_in]
                        } else {
                            &zero_pixel
                        };

                        let k_base = (ky * kw + kx) * c_in * c_out;
                        for ic in 0..c_in {
                            let iv = input_pixel[ic];
                            let kb = k_base + ic * c_out;
                            conv_fma_row(out_cell, &ker_data[kb..kb + c_out], iv);
                        }
                    }
                }
            }

            apply_conv_activation_inplace(out_row, activation);
        }
    }

    Tensor::from_aligned(vec![batch, out_h, out_w, c_out], output).map_err(Into::into)
}

/// im2col for NHWC input without padding.
/// Uses unsafe pointer arithmetic to avoid per-element bounds checks.
#[cfg(feature = "blas")]
#[allow(unsafe_code)]
fn im2col_nhwc(
    input: &[f32],
    in_w: usize,
    c: usize,
    kh: usize,
    kw: usize,
    stride_h: usize,
    stride_w: usize,
    out_h: usize,
    out_w: usize,
    col: &mut [f32],
) {
    let k = kh * kw * c;
    let in_row_stride = in_w * c;
    // SAFETY: all (oy*stride_h+ky, ox*stride_w+kx) are guaranteed in-bounds
    // because the output dimensions were computed from valid convolution params.
    unsafe {
        let inp = input.as_ptr();
        let mut dst = col.as_mut_ptr();
        for oy in 0..out_h {
            for ox in 0..out_w {
                for ky in 0..kh {
                    let src_row = inp.add((oy * stride_h + ky) * in_row_stride + ox * stride_w * c);
                    for kx in 0..kw {
                        std::ptr::copy_nonoverlapping(src_row.add(kx * c), dst, c);
                        dst = dst.add(c);
                    }
                }
            }
        }
        debug_assert_eq!(dst.offset_from(col.as_ptr()) as usize, out_h * out_w * k);
    }
}

/// im2col for a tile of output rows `[row_start .. row_start + tile_rows]`.
/// Uses unsafe pointer arithmetic for tight inner loops.
#[cfg(feature = "blas")]
#[allow(unsafe_code)]
fn im2col_nhwc_tile(
    input: &[f32],
    in_w: usize,
    c: usize,
    kh: usize,
    kw: usize,
    stride_h: usize,
    stride_w: usize,
    out_w: usize,
    row_start: usize,
    tile_rows: usize,
    col: &mut [f32],
) {
    let k = kh * kw * c;
    let in_row_stride = in_w * c;
    // SAFETY: all input indices are in-bounds (no padding case).
    unsafe {
        let inp = input.as_ptr();
        let mut dst = col.as_mut_ptr();
        for local_row in 0..tile_rows {
            let global_row = row_start + local_row;
            let oy = global_row / out_w;
            let ox = global_row % out_w;
            for ky in 0..kh {
                let src_row = inp.add((oy * stride_h + ky) * in_row_stride + ox * stride_w * c);
                for kx in 0..kw {
                    std::ptr::copy_nonoverlapping(src_row.add(kx * c), dst, c);
                    dst = dst.add(c);
                }
            }
        }
        debug_assert_eq!(dst.offset_from(col.as_ptr()) as usize, tile_rows * k);
    }
}

/// Conv2d via im2col + BLAS sgemm.
///
/// im2col matrix: [M, K] where M = out_h*out_w, K = kH*kW*C_in
/// kernel (already contiguous in NHWC): [K, N] where N = C_out
/// im2col + GEMM with fused bias+activation epilogue.
/// Works without BLAS — uses our custom blocked GEMM when BLAS is unavailable.
#[cfg(feature = "blas")]
fn conv2d_im2col_gemm_fused(
    plan: &Conv2dPlan,
    input: &[f32],
    kernel: &[f32],
    bias: Option<&[f32]>,
    activation: Activation,
) -> Result<Tensor, KernelError> {
    let out_h = plan.out_h;
    let out_w = plan.out_w;
    let k = plan.kernel_h * plan.kernel_w * plan.in_channels;
    let m = out_h * out_w;
    let n = plan.out_channels;

    let epilogue = super::matmul::GemmEpilogue {
        bias: bias.map(|b| b.as_ptr()),
        activation,
        residual: None,
    };

    // For 1×1 conv with stride 1, input IS the im2col matrix — zero-copy.
    if plan.kernel_h == 1 && plan.kernel_w == 1 && plan.stride_h == 1 && plan.stride_w == 1 {
        #[allow(unsafe_code)]
        let mut output = AlignedVec::<f32>::uninitialized(m * n);
        super::matmul::blas_sgemm_fused(&input[..m * k], kernel, &mut output, m, k, n, epilogue);
        return Tensor::from_aligned(vec![1, out_h, out_w, n], output).map_err(Into::into);
    }

    // Tile size: keep im2col_tile + output_tile in ~2 MB per thread.
    let bytes_per_row = (k + n) * std::mem::size_of::<f32>();
    let tile_m = (2usize * 1024 * 1024)
        .checked_div(bytes_per_row)
        .map(|rows| rows.max(1).min(m))
        .unwrap_or(m);

    #[allow(unsafe_code)]
    let mut output = AlignedVec::<f32>::uninitialized(m * n);

    if m > tile_m * 2 {
        super::super::scope_ctx::par_chunks_mut_dispatch(
            output.as_mut_slice(),
            tile_m * n,
            |tile_idx, out_chunk| {
                let row_start = tile_idx * tile_m;
                let actual_m = out_chunk.len() / n;
                if actual_m == 0 {
                    return;
                }
                thread_local! {
                    static COL_BUF: std::cell::RefCell<Vec<f32>> = const { std::cell::RefCell::new(Vec::new()) };
                }
                COL_BUF.with(|cell| {
                    let mut col_buf = cell.borrow_mut();
                    let needed = actual_m * k;
                    if col_buf.len() < needed {
                        col_buf.resize(needed, 0.0);
                    }
                    im2col_nhwc_tile(
                        input,
                        plan.in_w,
                        plan.in_channels,
                        plan.kernel_h,
                        plan.kernel_w,
                        plan.stride_h,
                        plan.stride_w,
                        out_w,
                        row_start,
                        actual_m,
                        &mut col_buf[..needed],
                    );
                    super::matmul::blas_sgemm_fused(
                        &col_buf[..needed],
                        kernel,
                        out_chunk,
                        actual_m,
                        k,
                        n,
                        epilogue,
                    );
                });
            },
        );
    } else {
        thread_local! {
            static MAIN_COL_BUF: std::cell::RefCell<Vec<f32>> = const { std::cell::RefCell::new(Vec::new()) };
        }
        MAIN_COL_BUF.with(|cell| {
            let mut col_buf = cell.borrow_mut();
            let needed = m * k;
            if col_buf.len() < needed {
                col_buf.resize(needed, 0.0);
            }
            im2col_nhwc(
                input,
                plan.in_w,
                plan.in_channels,
                plan.kernel_h,
                plan.kernel_w,
                plan.stride_h,
                plan.stride_w,
                out_h,
                out_w,
                &mut col_buf[..needed],
            );
            super::matmul::blas_sgemm_fused(
                &col_buf[..needed],
                kernel,
                &mut output,
                m,
                k,
                n,
                epilogue,
            );
        });
    }

    Tensor::from_aligned(vec![1, out_h, out_w, n], output).map_err(Into::into)
}

// ---------------------------------------------------------------------------
// Winograd F(2×2, 3×3) for non-Apple platforms
// ---------------------------------------------------------------------------
//
// On macOS, Apple Accelerate's AMX-backed sgemm is fast enough that Winograd's
// 16 smaller GEMMs lose more in BLAS efficiency than they gain in FLOPs.
// On other platforms (OpenBLAS, MKL, etc.) the arithmetic saving wins.

/// Transform 3×3 NHWC weights for Winograd F(2,3): G * g * G^T.
///
/// Input `kernel` is `[kH=3, kW=3, c_in, c_out]` (NHWC / HWIO).
/// Output is `[16, c_in, c_out]` (alpha-major, then c_in, then c_out).
#[cfg(all(feature = "blas", not(target_os = "macos")))]
fn winograd_transform_weights_f32(kernel: &[f32], c_in: usize, c_out: usize) -> Vec<f32> {
    // G = [[1,0,0],[0.5,0.5,0.5],[0.5,-0.5,0.5],[0,0,1]]
    let mut out = vec![0.0f32; 16 * c_in * c_out];
    for ci in 0..c_in {
        for co in 0..c_out {
            // HWIO layout: kernel[ h*kw*c_in*c_out + w*c_in*c_out + ci*c_out + co ]
            let g = |r: usize, s: usize| {
                kernel[r * 3 * c_in * c_out + s * c_in * c_out + ci * c_out + co]
            };

            // G * g → 4×3
            let mut gg = [0.0f32; 12];
            for s in 0..3 {
                gg[s] = g(0, s);
                gg[3 + s] = 0.5 * (g(0, s) + g(1, s) + g(2, s));
                gg[6 + s] = 0.5 * (g(0, s) - g(1, s) + g(2, s));
                gg[9 + s] = g(2, s);
            }

            // (G * g) * G^T → 4×4
            let mut u = [0.0f32; 16];
            for r in 0..4 {
                let row = &gg[r * 3..r * 3 + 3];
                u[r * 4] = row[0];
                u[r * 4 + 1] = 0.5 * (row[0] + row[1] + row[2]);
                u[r * 4 + 2] = 0.5 * (row[0] - row[1] + row[2]);
                u[r * 4 + 3] = row[2];
            }

            // Scatter to [alpha, c_in, c_out]
            for a in 0..16 {
                out[a * c_in * c_out + ci * c_out + co] = u[a];
            }
        }
    }
    out
}

/// Winograd input transform: B^T * d * B for one 4×4 tile.
///
/// B^T = [[1,0,-1,0],[0,1,1,0],[0,-1,1,0],[0,1,0,-1]]
#[cfg(all(feature = "blas", not(target_os = "macos")))]
#[inline]
fn winograd_input_tile(d: &[f32; 16], out: &mut [f32; 16]) {
    // B^T * d → 4×4 intermediate (rows transformed)
    let mut bd = [0.0f32; 16];
    for col in 0..4 {
        bd[col] = d[col] - d[2 * 4 + col];
        bd[4 + col] = d[4 + col] + d[2 * 4 + col];
        bd[8 + col] = -d[4 + col] + d[2 * 4 + col];
        bd[12 + col] = d[4 + col] - d[3 * 4 + col];
    }
    // (B^T * d) * B → 4×4 (columns transformed)
    for row in 0..4 {
        let r = row * 4;
        out[r] = bd[r] - bd[r + 2];
        out[r + 1] = bd[r + 1] + bd[r + 2];
        out[r + 2] = -bd[r + 1] + bd[r + 2];
        out[r + 3] = bd[r + 1] - bd[r + 3];
    }
}

/// Winograd output transform: A^T * m * A, yielding 2×2 output from 4×4 product.
///
/// A^T = [[1,1,1,0],[0,1,-1,-1]]
#[cfg(all(feature = "blas", not(target_os = "macos")))]
#[inline]
fn winograd_output_tile(m: &[f32; 16], out: &mut [f32; 4]) {
    // A^T * m → 2×4 intermediate (rows transformed)
    let mut am = [0.0f32; 8];
    for col in 0..4 {
        am[col] = m[col] + m[4 + col] + m[8 + col];
        am[4 + col] = m[4 + col] - m[8 + col] - m[12 + col];
    }
    // (A^T * m) * A → 2×2 (columns transformed)
    out[0] = am[0] + am[1] + am[2];
    out[1] = am[1] - am[2] - am[3];
    out[2] = am[4] + am[5] + am[6];
    out[3] = am[5] - am[6] - am[7];
}

/// Full Winograd F(2×2, 3×3) convolution for NHWC layout.
///
/// Only valid for 3×3 kernels with stride=1.
/// `input` NHWC `[batch, H, W, c_in]` (unpadded), `kernel` `[3, 3, c_in, c_out]`.
#[cfg(all(feature = "blas", not(target_os = "macos")))]
fn winograd_conv2d_nhwc(
    input: &[f32],
    kernel: &[f32],
    bias: Option<&[f32]>,
    batch: usize,
    in_h: usize,
    in_w: usize,
    c_in: usize,
    c_out: usize,
    pad_top: usize,
    pad_left: usize,
    pad_bottom: usize,
    pad_right: usize,
    activation: Activation,
) -> Result<Tensor, KernelError> {
    let padded_h = in_h + pad_top + pad_bottom;
    let padded_w = in_w + pad_left + pad_right;
    let out_h = padded_h - 2; // (padded_h - 3) / 1 + 1
    let out_w = padded_w - 2;

    // Number of 2×2 output tiles
    let tile_h = out_h.div_ceil(2);
    let tile_w = out_w.div_ceil(2);
    let n_tiles = tile_h * tile_w;

    // 1. Transform weights: [16, c_in, c_out]
    let u = winograd_transform_weights_f32(kernel, c_in, c_out);

    // SAFETY: every element written by the GEMM + output transform.
    #[allow(unsafe_code)]
    let mut output = AlignedVec::<f32>::uninitialized(batch * out_h * out_w * c_out);

    for b in 0..batch {
        let in_batch = &input[b * in_h * in_w * c_in..(b + 1) * in_h * in_w * c_in];

        // 2. Input transform: for each tile, for each channel, compute B^T * d * B
        //    Result layout: [16, n_tiles, c_in]
        let mut v = vec![0.0f32; 16 * n_tiles * c_in];
        for th in 0..tile_h {
            for tw in 0..tile_w {
                let tile_idx = th * tile_w + tw;
                // Top-left corner of the 4×4 input tile in padded coords
                let py0 = th * 2;
                let px0 = tw * 2;

                for ci in 0..c_in {
                    // Load 4×4 tile with implicit zero-padding
                    let mut d = [0.0f32; 16];
                    for dy in 0..4 {
                        for dx in 0..4 {
                            let iy = (py0 + dy).wrapping_sub(pad_top);
                            let ix = (px0 + dx).wrapping_sub(pad_left);
                            if iy < in_h && ix < in_w {
                                d[dy * 4 + dx] = in_batch[iy * in_w * c_in + ix * c_in + ci];
                            }
                        }
                    }
                    let mut vt = [0.0f32; 16];
                    winograd_input_tile(&d, &mut vt);
                    for a in 0..16 {
                        v[a * n_tiles * c_in + tile_idx * c_in + ci] = vt[a];
                    }
                }
            }
        }

        // 3. Batched GEMM: for each alpha ∈ 0..16,
        //    M[alpha] = V[alpha] * U[alpha]
        //    V[alpha]: [n_tiles, c_in], U[alpha]: [c_in, c_out]
        //    M[alpha]: [n_tiles, c_out]
        let mut m_buf = vec![0.0f32; 16 * n_tiles * c_out];
        for a in 0..16 {
            let v_slice = &v[a * n_tiles * c_in..(a + 1) * n_tiles * c_in];
            let u_slice = &u[a * c_in * c_out..(a + 1) * c_in * c_out];
            let m_slice = &mut m_buf[a * n_tiles * c_out..(a + 1) * n_tiles * c_out];
            super::matmul::blas_sgemm(v_slice, u_slice, m_slice, n_tiles, c_in, c_out);
        }

        // 4. Output transform: A^T * M * A → 2×2 output per tile, with bias + activation
        let out_batch = &mut output[b * out_h * out_w * c_out..(b + 1) * out_h * out_w * c_out];
        for th in 0..tile_h {
            for tw in 0..tile_w {
                let tile_idx = th * tile_w + tw;
                let oy0 = th * 2;
                let ox0 = tw * 2;
                // Clamp: last tile row/col may produce fewer than 2 valid outputs
                let valid_h = (out_h - oy0).min(2);
                let valid_w = (out_w - ox0).min(2);

                for co in 0..c_out {
                    // Gather the 4×4 product elements for this (tile, channel)
                    let mut mt = [0.0f32; 16];
                    for a in 0..16 {
                        mt[a] = m_buf[a * n_tiles * c_out + tile_idx * c_out + co];
                    }
                    let mut otile = [0.0f32; 4];
                    winograd_output_tile(&mt, &mut otile);

                    // Add bias
                    let bias_val = bias.map_or(0.0, |bd| bd[co]);
                    for dy in 0..valid_h {
                        for dx in 0..valid_w {
                            let idx = (oy0 + dy) * out_w * c_out + (ox0 + dx) * c_out + co;
                            out_batch[idx] = otile[dy * 2 + dx] + bias_val;
                        }
                    }
                }
            }
        }

        // Apply activation on the whole batch output
        match activation {
            Activation::Silu => silu_slice_inplace(out_batch),
            Activation::Relu => relu_slice_inplace(out_batch),
            Activation::None => {}
        }
    }

    Tensor::from_aligned(vec![batch, out_h, out_w, c_out], output).map_err(Into::into)
}

/// Conv2D with implicit zero-padding — avoids separate padded tensor allocation.
///
/// `input` is NHWC `[batch, H, W, C_in]` (unpadded).
/// `kernel` is `[KH, KW, C_in, C_out]`.
/// Padding is applied virtually during im2col: out-of-bounds reads yield 0.
pub fn conv2d_nhwc_padded(
    input: &Tensor,
    kernel: &Tensor,
    bias: Option<&Tensor>,
    stride_h: usize,
    stride_w: usize,
    pad_top: usize,
    pad_left: usize,
    pad_bottom: usize,
    pad_right: usize,
    activation: Activation,
) -> Result<Tensor, KernelError> {
    let in_shape = input.shape();
    if in_shape.len() != 4 || kernel.shape().len() != 4 {
        return Err(KernelError::InvalidConvRank {
            input_rank: in_shape.len(),
            kernel_rank: kernel.shape().len(),
        });
    }
    let (batch, in_h, in_w, c_in) = (in_shape[0], in_shape[1], in_shape[2], in_shape[3]);
    let k_shape = kernel.shape();
    let (kh, kw, _k_cin, c_out) = (k_shape[0], k_shape[1], k_shape[2], k_shape[3]);

    let padded_h = in_h + pad_top + pad_bottom;
    let padded_w = in_w + pad_left + pad_right;
    let out_h = (padded_h - kh) / stride_h + 1;
    let out_w = (padded_w - kw) / stride_w + 1;
    let m = out_h * out_w;
    let k = kh * kw * c_in;
    let n = c_out;

    let in_data = input.data();
    let kernel_data = kernel.data();
    let bias_data = bias.map(|b| b.data());

    // Dedicated first-layer 3×3 stride-2 RGB microkernel. Shape matches
    // the opening Conv of the Siamese tracker (and most CV models that
    // take raw RGB input). Generic im2col+BLAS here has k=27 which kills
    // blocked-GEMM efficiency; the specialised kernel skips im2col and
    // packs the unroll directly into SIMD FMAs.
    if kh == 3
        && kw == 3
        && stride_h == 2
        && stride_w == 2
        && c_in == 3
        && (matches!(activation, Activation::None) || matches!(activation, Activation::Relu))
    {
        let output_len = batch * out_h * out_w * c_out;
        #[allow(unsafe_code)]
        let mut output = AlignedVec::<f32>::uninitialized(output_len);
        super::first_layer_3x3::conv2d_nhwc_3ch_3x3_s2_padded(
            in_data,
            kernel_data,
            bias_data,
            &mut output,
            batch,
            in_h,
            in_w,
            c_out,
            pad_top,
            pad_left,
            pad_bottom,
            pad_right,
            activation,
            None,
        );
        return Tensor::from_aligned(vec![batch, out_h, out_w, c_out], output).map_err(Into::into);
    }

    // Winograd F(2×2,3×3): on non-Apple platforms, use Winograd for 3×3 stride-1
    // convolutions with enough spatial output to amortise the transform overhead.
    // On macOS, Apple Accelerate's AMX-backed sgemm makes im2col+GEMM faster.
    #[cfg(all(feature = "blas", not(target_os = "macos")))]
    if kh == 3 && kw == 3 && stride_h == 1 && stride_w == 1 && out_h * out_w >= 64 {
        return winograd_conv2d_nhwc(
            in_data,
            kernel_data,
            bias_data,
            batch,
            in_h,
            in_w,
            c_in,
            c_out,
            pad_top,
            pad_left,
            pad_bottom,
            pad_right,
            activation,
        );
    }

    // Tile size: keep im2col_tile + output_tile in ~2 MB per thread.
    let bytes_per_row = (k + n) * std::mem::size_of::<f32>();
    let tile_m = (2usize * 1024 * 1024)
        .checked_div(bytes_per_row)
        .map(|rows| rows.max(1).min(m))
        .unwrap_or(m);

    // SAFETY: every element is written by blas_sgemm (beta=0) + bias add.
    #[allow(unsafe_code)]
    let mut output = AlignedVec::<f32>::uninitialized(batch * m * n);

    for b in 0..batch {
        let in_slice = &in_data[b * in_h * in_w * c_in..(b + 1) * in_h * in_w * c_in];
        let out_batch = &mut output[b * m * n..(b + 1) * m * n];

        if m > tile_m * 2 {
            // Parallel tiled im2col + GEMM
            super::super::scope_ctx::par_chunks_mut_dispatch(
                out_batch,
                tile_m * n,
                |tile_idx, out_chunk| {
                    let row_start = tile_idx * tile_m;
                    let actual_m = out_chunk.len() / n;
                    if actual_m == 0 {
                        return;
                    }
                    thread_local! {
                        static PAD_COL_BUF: std::cell::RefCell<Vec<f32>> = const { std::cell::RefCell::new(Vec::new()) };
                    }
                    PAD_COL_BUF.with(|cell| {
                        let mut col_buf = cell.borrow_mut();
                        let needed = actual_m * k;
                        if col_buf.len() < needed {
                            col_buf.resize(needed, 0.0);
                        }
                        im2col_nhwc_padded_tile(
                            in_slice,
                            in_h,
                            in_w,
                            c_in,
                            kh,
                            kw,
                            stride_h,
                            stride_w,
                            pad_top,
                            pad_left,
                            out_w,
                            row_start,
                            actual_m,
                            &mut col_buf[..needed],
                        );
                        let epilogue = super::matmul::GemmEpilogue {
                            bias: bias_data.map(|b| b.as_ptr()),
                            activation,
                            residual: None,
                        };
                        super::matmul::blas_sgemm_fused(
                            &col_buf[..needed],
                            kernel_data,
                            out_chunk,
                            actual_m,
                            k,
                            n,
                            epilogue,
                        );
                    });
                },
            );
        } else {
            // Single tile
            thread_local! {
                static MAIN_PAD_COL_BUF: std::cell::RefCell<Vec<f32>> = const { std::cell::RefCell::new(Vec::new()) };
            }
            MAIN_PAD_COL_BUF.with(|cell| {
                let mut col_buf = cell.borrow_mut();
                let needed = m * k;
                if col_buf.len() < needed {
                    col_buf.resize(needed, 0.0);
                }
                im2col_nhwc_padded(
                    in_slice,
                    in_h,
                    in_w,
                    c_in,
                    kh,
                    kw,
                    stride_h,
                    stride_w,
                    pad_top,
                    pad_left,
                    out_h,
                    out_w,
                    &mut col_buf[..needed],
                );
                let epilogue = super::matmul::GemmEpilogue {
                    bias: bias_data.map(|b| b.as_ptr()),
                    activation,
                    residual: None,
                };
                super::matmul::blas_sgemm_fused(
                    &col_buf[..needed],
                    kernel_data,
                    out_batch,
                    m,
                    k,
                    n,
                    epilogue,
                );
            });
        }
    }

    Tensor::from_aligned(vec![batch, out_h, out_w, n], output).map_err(Into::into)
}

/// im2col with implicit zero-padding.  Out-of-bounds input reads are zero.
///
/// Interior optimization: for output rows where ALL kernel positions fall
/// within the valid input region, we skip per-element bounds checks entirely.
/// This covers ~90%+ of output positions for typical 3×3 pad=1 convolutions.
#[allow(unsafe_code)]
fn im2col_nhwc_padded(
    input: &[f32],
    in_h: usize,
    in_w: usize,
    c: usize,
    kh: usize,
    kw: usize,
    stride_h: usize,
    stride_w: usize,
    pad_top: usize,
    pad_left: usize,
    out_h: usize,
    out_w: usize,
    col: &mut [f32],
) {
    let k = kh * kw * c;
    // Compute the range of output rows/cols where ALL kernel taps are valid.
    // Interior: oy where oy*stride_h >= pad_top and oy*stride_h + kh - 1 < in_h + pad_top
    //   → oy >= ceil(pad_top / stride_h) and oy <= (in_h + pad_top - kh) / stride_h
    let oy_start = if pad_top > 0 {
        pad_top.div_ceil(stride_h)
    } else {
        0
    };
    let oy_end = if in_h + pad_top >= kh {
        (in_h + pad_top - kh) / stride_h + 1
    } else {
        0
    }
    .min(out_h);
    let ox_start = if pad_left > 0 {
        pad_left.div_ceil(stride_w)
    } else {
        0
    };
    let ox_end = if in_w + pad_left >= kw {
        (in_w + pad_left - kw) / stride_w + 1
    } else {
        0
    }
    .min(out_w);

    let in_row_stride = in_w * c;

    for oy in 0..out_h {
        let base_iy = oy * stride_h;
        let is_interior_y = oy >= oy_start && oy < oy_end;

        for ox in 0..out_w {
            let row_off = (oy * out_w + ox) * k;

            if is_interior_y && ox >= ox_start && ox < ox_end {
                // Interior: all kernel taps are valid — no bounds checks.
                let base_ix = ox * stride_w - pad_left;
                let base_iy_val = base_iy - pad_top;
                // SAFETY: we verified all (iy, ix) are in-bounds above.
                unsafe {
                    let mut dst = col.as_mut_ptr().add(row_off);
                    if stride_w == 1 {
                        // When stride_w==1, kernel taps along x are contiguous in NHWC
                        // layout, so we copy kw*c floats per kernel row instead of kw
                        // separate copies. For 3×3: 3 memcpys instead of 9.
                        let row_bytes = kw * c;
                        for ky in 0..kh {
                            let src_row = input
                                .as_ptr()
                                .add((base_iy_val + ky) * in_row_stride + base_ix * c);
                            std::ptr::copy_nonoverlapping(src_row, dst, row_bytes);
                            dst = dst.add(row_bytes);
                        }
                    } else {
                        for ky in 0..kh {
                            let src_row = input
                                .as_ptr()
                                .add((base_iy_val + ky) * in_row_stride + base_ix * c);
                            for kx in 0..kw {
                                std::ptr::copy_nonoverlapping(src_row.add(kx * c), dst, c);
                                dst = dst.add(c);
                            }
                        }
                    }
                }
            } else {
                // Border: some taps may be out of bounds.
                for ky in 0..kh {
                    let iy = (base_iy + ky) as isize - pad_top as isize;
                    for kx in 0..kw {
                        let ix = (ox * stride_w + kx) as isize - pad_left as isize;
                        let dst_off = row_off + (ky * kw + kx) * c;
                        if iy >= 0 && (iy as usize) < in_h && ix >= 0 && (ix as usize) < in_w {
                            let src_off = (iy as usize * in_w + ix as usize) * c;
                            col[dst_off..dst_off + c].copy_from_slice(&input[src_off..src_off + c]);
                        } else {
                            col[dst_off..dst_off + c].fill(0.0);
                        }
                    }
                }
            }
        }
    }
}

/// Tiled im2col with implicit padding — handles out-of-bounds as zero.
/// Same interface as `im2col_nhwc_tile` but with padding parameters.
///
/// Uses interior/border split: for output positions where all kernel taps
/// fall within valid input, skips bounds checks entirely via unsafe ptrs.
#[allow(unsafe_code)]
fn im2col_nhwc_padded_tile(
    input: &[f32],
    in_h: usize,
    in_w: usize,
    c: usize,
    kh: usize,
    kw: usize,
    stride_h: usize,
    stride_w: usize,
    pad_top: usize,
    pad_left: usize,
    out_w: usize,
    row_start: usize,
    tile_rows: usize,
    col: &mut [f32],
) {
    let k = kh * kw * c;
    let in_row_stride = in_w * c;

    // Interior bounds (same computation as non-tiled version).
    let oy_start = if pad_top > 0 {
        pad_top.div_ceil(stride_h)
    } else {
        0
    };
    let oy_end_val = if in_h + pad_top >= kh {
        (in_h + pad_top - kh) / stride_h + 1
    } else {
        0
    };
    let ox_start = if pad_left > 0 {
        pad_left.div_ceil(stride_w)
    } else {
        0
    };
    let ox_end = if in_w + pad_left >= kw {
        (in_w + pad_left - kw) / stride_w + 1
    } else {
        0
    }
    .min(out_w);

    for local_row in 0..tile_rows {
        let global_row = row_start + local_row;
        let oy = global_row / out_w;
        let ox = global_row % out_w;
        let row_off = local_row * k;

        let is_interior = oy >= oy_start && oy < oy_end_val && ox >= ox_start && ox < ox_end;

        if is_interior {
            let base_iy = oy * stride_h - pad_top;
            let base_ix = ox * stride_w - pad_left;
            // SAFETY: interior guarantees all (iy, ix) are in-bounds.
            unsafe {
                let mut dst = col.as_mut_ptr().add(row_off);
                if stride_w == 1 {
                    let row_bytes = kw * c;
                    for ky in 0..kh {
                        let src_row = input
                            .as_ptr()
                            .add((base_iy + ky) * in_row_stride + base_ix * c);
                        std::ptr::copy_nonoverlapping(src_row, dst, row_bytes);
                        dst = dst.add(row_bytes);
                    }
                } else {
                    for ky in 0..kh {
                        let src_row = input
                            .as_ptr()
                            .add((base_iy + ky) * in_row_stride + base_ix * c);
                        for kx in 0..kw {
                            std::ptr::copy_nonoverlapping(src_row.add(kx * c), dst, c);
                            dst = dst.add(c);
                        }
                    }
                }
            }
        } else {
            let base_iy = oy * stride_h;
            for ky in 0..kh {
                let iy = (base_iy + ky) as isize - pad_top as isize;
                for kx in 0..kw {
                    let ix = (ox * stride_w + kx) as isize - pad_left as isize;
                    let dst_off = row_off + (ky * kw + kx) * c;
                    if iy >= 0 && (iy as usize) < in_h && ix >= 0 && (ix as usize) < in_w {
                        let src_off = (iy as usize * in_w + ix as usize) * c;
                        col[dst_off..dst_off + c].copy_from_slice(&input[src_off..src_off + c]);
                    } else {
                        col[dst_off..dst_off + c].fill(0.0);
                    }
                }
            }
        }
    }
}

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
    prepacked_b: Option<&super::matmul::PackedB>,
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
    let input_nhwc = super::layout::nchwc_to_nhwc(input, actual_in_channels)?;

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
    super::layout::nhwc_to_nchwc(&output_nhwc, block)
}

/// 1×1 Conv over an NCHWc-laid-out input, producing an NCHWc output.
///
/// Implemented as a composed boundary wrapper: convert NCHWc → NHWC, run
/// the existing `conv2d_nhwc_with_activation_prepacked` pointwise fast
/// path (which is just a GEMM), and convert the result back NHWC →
/// NCHWc tagged with the same `block`.
///
/// Per the backus plan (B.3 — "pointwise is isomorphic to NHWC"), this
/// is infrastructure glue — the pointwise case does not benefit from
/// NCHWc layout on its own; the win lives in chaining it with the 3×3
/// direct kernels and pool/BN paths (B.4/B.5) so that conversions amortize
/// over multiple ops. We pay two linear-time layout reorderings in
/// exchange for the op's output being consumable by the rest of the
/// NCHWc subgraph without a round-trip through NHWC.
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
pub fn conv2d_nchwc_pointwise_with_activation_prepacked(
    input: &Tensor,
    kernel: &Tensor,
    bias: Option<&Tensor>,
    actual_in_channels: usize,
    activation: Activation,
    config: ParallelElementwiseConfig,
    thread_pool: Option<&ThreadPool>,
    prepacked_b: Option<&super::matmul::PackedB>,
) -> Result<Tensor, KernelError> {
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

    // Boundary in: NCHWc -> NHWC, stripping padded channels so the inner
    // GEMM works on the real `actual_in_channels` dimension.
    let input_nhwc = super::layout::nchwc_to_nhwc(input, actual_in_channels)?;

    let spec = Conv2dSpec {
        stride_h: 1,
        stride_w: 1,
    };
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

    // Boundary out: NHWC -> NCHWc with the same block size, pads C_out
    // tail with zeros so downstream NCHWc ops see a full block grid.
    super::layout::nhwc_to_nchwc(&output_nhwc, block).inspect(|t| {
        // Validate: produced output has the advertised C_out channel count.
        debug_assert_eq!(t.shape()[4], block, "nchwc output block mismatch");
        let _ = out_channels; // consumed via kernel metadata above
    })
}
