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

/// `YSCV_AVX512_DW_OFF` kill switch for the AVX-512 depthwise row kernel,
/// read once per process via OnceLock. The DW row dispatch fires once per DW
/// output row (many per inference), so a raw `std::env::var_os` per call would
/// show up in profiles. (The pointwise parallel-dispatch threshold it sits
/// near is `YSCV_MIN_PAR_POINTWISE_CONV_ELEMS` / `_FLOPS`.)
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn avx512_dw_disabled() -> bool {
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| std::env::var_os("YSCV_AVX512_DW_OFF").is_some())
}

/// `YSCV_DW3X3_C16_OFF` kill switch for the c=16 depthwise 3×3 fast path.
/// Only consulted by the AVX-512 dispatch site (x86_64); gated to keep
/// non-x86 builds warning-free.
#[cfg(target_arch = "x86_64")]
fn c16_dw_disabled() -> bool {
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| std::env::var_os("YSCV_DW3X3_C16_OFF").is_some())
}

/// `YSCV_NCHWC_PW_DIRECT_OFF` kill switch for the direct NCHWc PW AVX-512
/// 4-pixel × 16-channel kernel; falls back to the blocked-GEMM NCHWc path.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn nchwc_pw_direct_disabled() -> bool {
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| std::env::var_os("YSCV_NCHWC_PW_DIRECT_OFF").is_some())
}

/// `YSCV_KCBLOCK=1` opt-in for the K-cache-blocked AVX-512 nx16 pointwise
/// variant. Default OFF (tracker-flat); useful to A/B on K-heavy Conv_Add
/// shapes where the microbench shows a win.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn kcblock_enabled() -> bool {
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| std::env::var_os("YSCV_KCBLOCK").is_some())
}

/// `YSCV_MR16=1` opt-in for the 16-row MR AVX-512 nx16 pointwise variant.
/// Default OFF (tracker-flat).
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn mr16_enabled() -> bool {
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| std::env::var_os("YSCV_MR16").is_some())
}

/// Software-prefetch of the weight cacheline 8 K-iters ahead inside
/// `pointwise_nx16_direct_rows_avx512`. The HW prefetcher misses the weight
/// load pattern (stride = `n*4` bytes, 448-1280 B for tracker shapes), so a
/// manual prefetch hint hides the load latency behind the FMA pipe — a bigger
/// win on the in-order A53 NEON reduce kernel. Default ON; kill switch
/// `YSCV_PW_PREFETCH_OFF=1`.
#[cfg(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64"))]
fn pw_prefetch_enabled() -> bool {
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| std::env::var_os("YSCV_PW_PREFETCH_OFF").is_none())
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
/// Pointwise-Conv parallel-dispatch gate on `work = 2*m*k*n` (FLOPs). Below
/// `min_flops` the config forces sequential execution. `min_elems` is a
/// secondary gate applied only in the memory-bound regime (`2*k < 32`, i.e.
/// few FMAs per output lane); compute-bound small-output shapes (high-k) still
/// parallelize because per-thread work scales with k.
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

mod pointwise;
pub(crate) use pointwise::pointwise_nx16_direct_rows;
use pointwise::*;

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

fn fused_dw_pw_row_batch(out_h: usize, out_w: usize, channels: usize) -> usize {
    static OVERRIDE: OnceLock<Option<usize>> = OnceLock::new();
    if let Some(v) = *OVERRIDE.get_or_init(|| {
        std::env::var("YSCV_FUSED_DW_PW_ROW_BATCH")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&v| v > 0)
    }) {
        return v.min(out_h).max(1);
    }

    let row_bytes = out_w
        .saturating_mul(channels)
        .saturating_mul(std::mem::size_of::<f32>());
    let target_bytes = 256 * 1024usize;
    let rows = (target_bytes / row_bytes.max(1)).clamp(1, 8);
    rows.min(out_h).max(1)
}

/// NHWC 2-D convolution with an explicit parallel config and thread pool.
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

/// NHWC convolution with a fused activation, explicit parallel config, and thread pool.
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
    if !pointwise_16x16_direct_disabled() && k == 16 && n == 16 {
        pointwise_16x16_direct(
            input.data(),
            kernel.data(),
            bias.map(|b| b.data()),
            Some(residual.data()),
            &mut output,
            m,
            activation,
        );
        return Tensor::from_aligned(
            vec![plan.batch, plan.out_h, plan.out_w, plan.out_channels],
            output,
        )
        .map_err(Into::into);
    }
    if !pointwise_nx16_direct_disabled()
        && n.is_multiple_of(16)
        && n >= 32
        && m <= 1024
        && rayon::current_num_threads() <= 1
        && matches!(
            activation,
            Activation::None | Activation::Relu | Activation::Silu
        )
    {
        pointwise_nx16_direct(
            input.data(),
            kernel.data(),
            bias.map(|b| b.data()),
            Some(residual.data()),
            &mut output,
            m,
            k,
            n,
            activation,
        );
        return Tensor::from_aligned(
            vec![plan.batch, plan.out_h, plan.out_w, plan.out_channels],
            output,
        )
        .map_err(Into::into);
    }
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

/// NHWC convolution with fused activation using a pre-packed weight panel.
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
        if !pointwise_16x16_direct_disabled() && k == 16 && n == 16 {
            pointwise_16x16_direct(
                input.data(),
                kernel.data(),
                bias.map(|b| b.data()),
                None,
                &mut output,
                m,
                activation,
            );
            return Tensor::from_aligned(
                vec![plan.batch, plan.out_h, plan.out_w, plan.out_channels],
                output,
            )
            .map_err(Into::into);
        }
        // Mirror the same nx16 fast path used by conv2d_nhwc_pointwise_with_residual_relu.
        // Effective for non-residual 1T shapes with n%16==0, n>=32, m<=1024
        // (e.g., non-streaming FusedPwDw PW-expand for large-c_exp tracker blocks
        // where the streaming ring-buffer is less efficient than a full-M GEMM).
        if !pointwise_nx16_direct_disabled()
            && n.is_multiple_of(16)
            && n >= 32
            && m <= 1024
            && rayon::current_num_threads() <= 1
            && matches!(
                activation,
                Activation::None | Activation::Relu | Activation::Silu
            )
        {
            pointwise_nx16_direct(
                input.data(),
                kernel.data(),
                bias.map(|b| b.data()),
                None,
                &mut output,
                m,
                k,
                n,
                activation,
            );
            return Tensor::from_aligned(
                vec![plan.batch, plan.out_h, plan.out_w, plan.out_channels],
                output,
            )
            .map_err(Into::into);
        }
        let epilogue = super::matmul::GemmEpilogue {
            bias: bias.map(|b| b.data().as_ptr()),
            activation,
            residual: None,
        };

        // Pointwise Conv feeds straight into matmul with the GemmEpilogue;
        // `matmul_2d_slices_fused_maybe_packed` picks the right kernel
        // (including the low-k tile) from shape + CPU features.
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
    // The threshold is total-work (~2M FLOPs): below it the direct kernel
    // wins. The NEON kernel is single-threaded, so BLAS GEMM wins for large
    // spatial even with few channels via multi-core parallelism.
    #[cfg(target_arch = "aarch64")]
    if plan.kernel_h == 3
        && plan.kernel_w == 3
        && plan.batch == 1
        && !cfg!(miri)
        && crate::host_cpu().features.neon
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
        && crate::host_cpu().features.avx
        && crate::host_cpu().features.fma
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
        && crate::host_cpu().features.fma
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

/// NHWC depthwise convolution with an explicit parallel config and thread pool.
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

/// NHWC depthwise convolution with a fused activation, explicit config, and thread pool.
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

/// Padded NHWC depthwise convolution with fused activation, explicit config, and thread pool.
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

    // Specialised fast path: NHWC depthwise 3×3 with depth_multiplier=1 and
    // channels=16 — exactly one ZMM per pixel. The generic per-tap kernel pays
    // a load+FMA+store per (pixel, tap); pre-loading the 9 weight ZMMs once per
    // output row and keeping a single ZMM accumulator across the 9 taps is much
    // faster. Gated by `YSCV_DW3X3_C16_OFF=1`.
    #[cfg(target_arch = "x86_64")]
    if !c16_dw_disabled()
        && depth_multiplier == 1
        && channels == 16
        && kernel_h == 3
        && kernel_w == 3
        && !cfg!(miri)
        && crate::host_cpu().features.avx512f
    {
        let mut output = AlignedVec::<f32>::uninitialized(output_len);
        depthwise3x3_nhwc_c16_avx512(
            in_data,
            kernel_data,
            bias_data,
            batch,
            in_h,
            in_w,
            pad_top,
            pad_left,
            pad_bottom,
            pad_right,
            stride_h,
            stride_w,
            out_h,
            out_w,
            output.as_mut_slice(),
            activation,
            thread_pool,
        );
        return Tensor::from_aligned(vec![batch, out_h, out_w, out_channels], output)
            .map_err(Into::into);
    }

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
                        depthwise_tap_fma(
                            out_cell,
                            &in_data[input_base..input_base + channels],
                            &kernel_data[kernel_base..kernel_base + channels],
                        );
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
                        depthwise_tap_fma(
                            out_cell,
                            &in_data[input_base..input_base + channels],
                            &kernel_data[kernel_base..kernel_base + channels],
                        );
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
                            depthwise_tap_fma(
                                out_cell,
                                &in_data[input_base..input_base + channels],
                                &kernel_data[kernel_base..kernel_base + channels],
                            );
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
                            depthwise_tap_fma(
                                out_cell,
                                &in_data[input_base..input_base + channels],
                                &kernel_data[kernel_base..kernel_base + channels],
                            );
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

    let do_par = should_parallelize_len(output_len, config.min_parallel_elements, thread_pool);
    if do_par {
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

/// NHWC separable (depthwise + pointwise) convolution with explicit config and thread pool.
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
                None,
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
    pw_residual: Option<&Tensor>, // fused residual add: out = pw_activation(pw_acc + bias + residual)
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
        // Streaming kernel supports only depth_multiplier=1. Residual fusion
        // is not supported on this fallback path; callers must not pass one.
        debug_assert!(
            pw_residual.is_none(),
            "residual not supported for dm>1 fallback"
        );
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
        if pw_residual.is_none() && use_true_fused_dw_pw_dm1(dw_plan, dw_activation) {
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
            pw_residual.map(|r| r.data()),
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
        pw_residual.map(|r| r.data()),
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
    pw_residual: Option<&[f32]>, // fused residual add in PW epilogue: out = act(pw + bias + res)
    dw_plan: DepthwiseConv2dPlan,
    pw_out_ch: usize,
    dw_activation: Activation,
    pw_activation: Activation,
) -> Result<Tensor, KernelError> {
    let out_h = dw_plan.out_h;
    let out_w = dw_plan.out_w;
    let dw_channels = dw_plan.out_channels;
    let batch = dw_plan.batch;
    let dw_row_len = out_w * dw_channels;
    let pw_row_len = out_w * pw_out_ch;
    let output_len = batch * out_h * pw_row_len;
    let row_batch = fused_dw_pw_row_batch(out_h, out_w, dw_channels);
    let dw_band_len = row_batch * dw_row_len;

    let mut output = AlignedVec::<f32>::uninitialized(output_len);

    let pw_bias_ptr = pw_bias.map(|b| b.as_ptr() as usize).unwrap_or(0);
    let pw_has_bias = pw_bias.is_some();
    // Store residual base as usize so the closure is Send (raw ptr is not Send).
    let pw_residual_addr: usize = pw_residual.map_or(0, |r| r.as_ptr() as usize);
    let pw_has_residual = pw_residual.is_some();
    let pw_config = ParallelMatmulConfig {
        min_parallel_shared_dim: 1,
        min_parallel_output_elements: usize::MAX, // no parallel within fused
    };

    // Per-band DW → PW streaming: each thread processes a contiguous chunk of
    // output rows, keeping its DW scratch in L2 (never touching L3 for the
    // intermediate). Thread-local DW_BUF avoids allocation per call; rayon's
    // work-stealing splits `total_rows` into approximately one chunk per worker.
    thread_local! {
        static DW_BUF: std::cell::RefCell<Vec<f32>> = const { std::cell::RefCell::new(Vec::new()) };
    }

    let total_rows = batch * out_h;
    let nthreads = rayon::current_num_threads().max(1);
    let rows_per_chunk = if nthreads > 1 && total_rows >= nthreads * row_batch {
        total_rows.div_ceil(nthreads)
    } else {
        total_rows
    };
    let chunk_out_len = rows_per_chunk * pw_row_len;

    let run_chunk = |chunk_start: usize, out_chunk: &mut [f32]| {
        let chunk_end = (chunk_start + rows_per_chunk).min(total_rows);
        DW_BUF.with(|cell| {
            let mut dw_buf = cell.borrow_mut();
            if dw_buf.len() < dw_band_len {
                dw_buf.resize(dw_band_len, 0.0);
            }
            for band_start in (chunk_start..chunk_end).step_by(row_batch) {
                let band_rows = row_batch.min(chunk_end - band_start);
                for band_row in 0..band_rows {
                    let row_idx = band_start + band_row;
                    let scratch_start = band_row * dw_row_len;
                    depthwise_conv2d_nhwc_row(
                        input,
                        dw_kernel,
                        dw_bias,
                        dw_plan,
                        row_idx,
                        &mut dw_buf[scratch_start..scratch_start + dw_row_len],
                        dw_activation,
                    );
                }
                let out_start = (band_start - chunk_start) * pw_row_len;
                let out_len = band_rows * pw_row_len;
                // Per-band residual ptr offset so GEMM indexes residual[(ic+row)*n+col].
                // SAFETY: pw_residual_addr is the base of a live &[f32] slice with
                // total_rows × pw_out_ch elements; band_start stays within that range.
                #[allow(unsafe_code)]
                let band_epilogue = super::matmul::GemmEpilogue {
                    bias: if pw_has_bias {
                        Some(pw_bias_ptr as *const f32)
                    } else {
                        None
                    },
                    activation: pw_activation,
                    residual: if pw_has_residual {
                        Some(unsafe {
                            (pw_residual_addr as *const f32).add(band_start * pw_out_ch)
                        })
                    } else {
                        None
                    },
                };
                super::matmul::matmul_2d_slices_fused_maybe_packed(
                    &dw_buf[..band_rows * dw_row_len],
                    band_rows * out_w,
                    dw_channels,
                    pw_kernel,
                    pw_out_ch,
                    &mut out_chunk[out_start..out_start + out_len],
                    None,
                    band_epilogue,
                    pw_config,
                    None,
                );
            }
        });
    };

    super::super::scope_ctx::par_chunks_mut_dispatch(
        output.as_mut_slice(),
        chunk_out_len,
        |chunk_idx, out_chunk| {
            run_chunk(chunk_idx * rows_per_chunk, out_chunk);
        },
    );

    Tensor::from_aligned(vec![batch, out_h, out_w, pw_out_ch], output).map_err(Into::into)
}

#[allow(unsafe_code, clippy::too_many_arguments)]
fn fused_dw_pw_nhwc_padded_dm1(
    input: &[f32],
    dw_kernel: &[f32],
    dw_bias: Option<&[f32]>,
    pw_kernel: &[f32],
    pw_bias: Option<&[f32]>,
    pw_residual: Option<&[f32]>,
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
    let row_batch = fused_dw_pw_row_batch(out_h, out_w, dw_channels);
    let dw_band_len = row_batch * dw_row_len;
    let mut output = AlignedVec::<f32>::uninitialized(output_len);

    let pw_bias_ptr = pw_bias.map(|b| b.as_ptr() as usize).unwrap_or(0);
    let pw_has_bias = pw_bias.is_some();
    let pw_residual_addr: usize = pw_residual.map_or(0, |r| r.as_ptr() as usize);
    let pw_has_residual = pw_residual.is_some();
    let pw_config = ParallelMatmulConfig {
        min_parallel_shared_dim: 1,
        min_parallel_output_elements: usize::MAX,
    };

    thread_local! {
        static DW_BUF: std::cell::RefCell<Vec<f32>> = const { std::cell::RefCell::new(Vec::new()) };
    }

    let nthreads = rayon::current_num_threads().max(1);
    let rows_per_chunk = if nthreads > 1 && total_rows >= nthreads * row_batch {
        total_rows.div_ceil(nthreads)
    } else {
        total_rows
    };
    let chunk_out_len = rows_per_chunk * pw_row_len;

    let run_chunk = |chunk_start: usize, out_chunk: &mut [f32]| {
        let chunk_end = (chunk_start + rows_per_chunk).min(total_rows);
        DW_BUF.with(|cell| {
            let mut dw_buf = cell.borrow_mut();
            if dw_buf.len() < dw_band_len {
                dw_buf.resize(dw_band_len, 0.0);
            }
            for band_start in (chunk_start..chunk_end).step_by(row_batch) {
                let band_rows = row_batch.min(chunk_end - band_start);
                for band_row in 0..band_rows {
                    let row_idx = band_start + band_row;
                    let scratch_start = band_row * dw_row_len;
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
                        &mut dw_buf[scratch_start..scratch_start + dw_row_len],
                        dw_activation,
                    );
                }
                let out_start = (band_start - chunk_start) * pw_row_len;
                let out_len = band_rows * pw_row_len;
                // SAFETY: pw_residual_addr is base of live &[f32] with total_rows×pw_out_ch elements.
                #[allow(unsafe_code)]
                let band_epilogue = super::matmul::GemmEpilogue {
                    bias: if pw_has_bias {
                        Some(pw_bias_ptr as *const f32)
                    } else {
                        None
                    },
                    activation: pw_activation,
                    residual: if pw_has_residual {
                        Some(unsafe {
                            (pw_residual_addr as *const f32).add(band_start * pw_out_ch)
                        })
                    } else {
                        None
                    },
                };
                super::matmul::matmul_2d_slices_fused_maybe_packed(
                    &dw_buf[..band_rows * dw_row_len],
                    band_rows * out_w,
                    dw_channels,
                    pw_kernel,
                    pw_out_ch,
                    &mut out_chunk[out_start..out_start + out_len],
                    None,
                    band_epilogue,
                    pw_config,
                    None,
                );
            }
        });
    };

    super::super::scope_ctx::par_chunks_mut_dispatch(
        output.as_mut_slice(),
        chunk_out_len,
        |chunk_idx, out_chunk| {
            run_chunk(chunk_idx * rows_per_chunk, out_chunk);
        },
    );

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
                    depthwise_tap_fma(
                        out_cell,
                        &input[input_base..input_base + channels],
                        &kernel[kernel_base..kernel_base + channels],
                    );
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
                    depthwise_tap_fma(
                        out_cell,
                        &input[input_base..input_base + channels],
                        &kernel[kernel_base..kernel_base + channels],
                    );
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
                depthwise_tap_fma(
                    out_cell,
                    &input[input_base..input_base + channels],
                    &kernel[kernel_base..kernel_base + channels],
                );
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

mod direct;
use direct::*;

mod depthwise;
pub use depthwise::conv3d;
use depthwise::*;

mod gemm_conv;
#[cfg(feature = "blas")]
use gemm_conv::conv2d_im2col_gemm_fused;
pub use gemm_conv::{conv2d_nhwc_indirect_padded, conv2d_nhwc_padded};

mod nchwc;
pub use nchwc::{
    conv2d_nchwc_pointwise_with_activation_prepacked,
    conv2d_nchwc_pointwise_with_residual_activation_prepacked,
    conv2d_nchwc_with_activation_prepacked,
};
