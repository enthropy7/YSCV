//! NCHWc-layout entry points for ops used inside NCHWc subgraphs.
//!
//! Each function here accepts an `NCHWc`-tagged input, runs the op via
//! the existing NHWC fast path, and returns an `NCHWc`-tagged output.
//! The implementation strategy is "convert-wrap": strip the block
//! structure at function entry, run the NHWC kernel (which already has
//! all the SIMD / BLAS / fusion paths), re-block on exit.
//!
//! Conversion cost is linear in `N*H*W*C`; op compute is `O(kernel × N*H*W*C)`
//! or higher, so the ratio is 2/(kernel area × C_out factor). For 3×3 Conv
//! with C≥8 that's <3% overhead; for elementwise it's 100% overhead —
//! hence the trivial ops (Relu/Silu/Add) use a layout-preserving fast
//! path that doesn't convert.
//!
//! Where winnable: when a **chain** of NCHWc-capable ops runs back-to-back,
//! the transformer (B.6) inserts conversions only on chain boundaries,
//! amortizing the one round-trip across the whole chain. This module
//! provides the ops the transformer uses.

use rayon::ThreadPool;
use yscv_tensor::{Layout, Tensor};

use crate::core::error::KernelError;

use super::config::{BatchNorm2dTensors, ParallelElementwiseConfig, Pool2dSpec};
use super::{elementwise, layout, norm, pool};

/// Reads the block size from a tensor that carries `Layout::NCHWc { block }`.
fn nchwc_block(input: &Tensor) -> Result<usize, KernelError> {
    match input.layout() {
        Layout::NCHWc { block } => Ok(block as usize),
        other => Err(KernelError::LayoutConversion(format!(
            "expected NCHWc-tagged tensor, got {other:?}"
        ))),
    }
}

/// Max-pool 2D over NCHWc input. Pool operates only on spatial dims —
/// channel structure is preserved — so we convert to NHWC, pool, convert
/// back. Output has the same `block` as the input.
pub fn max_pool2d_nchwc(
    input: &Tensor,
    actual_channels: usize,
    spec: Pool2dSpec,
    config: ParallelElementwiseConfig,
    thread_pool: Option<&ThreadPool>,
) -> Result<Tensor, KernelError> {
    let block = nchwc_block(input)?;
    let input_nhwc = layout::nchwc_to_nhwc(input, actual_channels)?;
    let out_nhwc =
        pool::max_pool2d_nhwc_with_config_and_pool(&input_nhwc, spec, config, thread_pool)?;
    layout::nhwc_to_nchwc(&out_nhwc, block)
}

/// Average-pool 2D over NCHWc input.
pub fn avg_pool2d_nchwc(
    input: &Tensor,
    actual_channels: usize,
    spec: Pool2dSpec,
    config: ParallelElementwiseConfig,
    thread_pool: Option<&ThreadPool>,
) -> Result<Tensor, KernelError> {
    let block = nchwc_block(input)?;
    let input_nhwc = layout::nchwc_to_nhwc(input, actual_channels)?;
    let out_nhwc =
        pool::avg_pool2d_nhwc_with_config_and_pool(&input_nhwc, spec, config, thread_pool)?;
    layout::nhwc_to_nchwc(&out_nhwc, block)
}

/// BatchNorm2d over NCHWc. Per-channel scale/shift, so it's trivially
/// layout-agnostic in principle — but the NHWC kernel uses channel-packed
/// SIMD loads that don't map cleanly to NCHWc's block-strided memory.
/// Convert-wrap path keeps everything correct.
#[allow(clippy::too_many_arguments)]
pub fn batch_norm2d_nchwc<'a>(
    input: &Tensor,
    tensors: BatchNorm2dTensors<'a>,
    actual_channels: usize,
    config: ParallelElementwiseConfig,
    thread_pool: Option<&ThreadPool>,
) -> Result<Tensor, KernelError> {
    let block = nchwc_block(input)?;
    let input_nhwc = layout::nchwc_to_nhwc(input, actual_channels)?;
    let out_nhwc =
        norm::batch_norm2d_nhwc_with_config_and_pool(&input_nhwc, tensors, config, thread_pool)?;
    layout::nhwc_to_nchwc(&out_nhwc, block)
}

/// Elementwise ReLU over an NCHWc tensor. ReLU touches each element in
/// isolation, so the operation is layout-agnostic: we run it on the
/// contiguous NCHWc buffer and retag the output as `NCHWc { block }`.
/// No conversion overhead.
pub fn relu_nchwc(
    input: &Tensor,
    config: ParallelElementwiseConfig,
    thread_pool: Option<&ThreadPool>,
) -> Result<Tensor, KernelError> {
    let block = nchwc_block(input)?;
    // Run the NHWC relu on the raw buffer — it only iterates elements,
    // layout doesn't matter. Retag on exit.
    let flat = input.clone().with_layout(Layout::NHWC);
    let out = elementwise::relu_with_config_and_pool(&flat, config, thread_pool);
    Ok(out.with_layout(Layout::NCHWc { block: block as u8 }))
}

/// Elementwise add (broadcast or same-shape). Layout-agnostic same as
/// ReLU — operates on flat memory. Both inputs must carry the same
/// `NCHWc { block }` tag and have identical shape.
pub fn add_nchwc(
    a: &Tensor,
    b: &Tensor,
    config: ParallelElementwiseConfig,
    thread_pool: Option<&ThreadPool>,
) -> Result<Tensor, KernelError> {
    let block = nchwc_block(a)?;
    let block_b = nchwc_block(b)?;
    if block != block_b {
        return Err(KernelError::LayoutConversion(format!(
            "add_nchwc: block size mismatch a={block} b={block_b}"
        )));
    }
    if a.shape() != b.shape() {
        return Err(KernelError::LayoutConversion(format!(
            "add_nchwc: shape mismatch a={:?} b={:?}",
            a.shape(),
            b.shape()
        )));
    }
    let a_flat = a.clone().with_layout(Layout::NHWC);
    let b_flat = b.clone().with_layout(Layout::NHWC);
    let out = elementwise::add_with_config_and_pool(&a_flat, &b_flat, config, thread_pool)?;
    Ok(out.with_layout(Layout::NCHWc { block: block as u8 }))
}

/// Elementwise sigmoid over NCHWc. Layout-agnostic.
pub fn sigmoid_nchwc(
    input: &Tensor,
    config: ParallelElementwiseConfig,
    thread_pool: Option<&ThreadPool>,
) -> Result<Tensor, KernelError> {
    let block = nchwc_block(input)?;
    let flat = input.clone().with_layout(Layout::NHWC);
    let out = elementwise::sigmoid_with_config_and_pool(&flat, config, thread_pool);
    Ok(out.with_layout(Layout::NCHWc { block: block as u8 }))
}

/// Elementwise SiLU (swish) over NCHWc. Layout-agnostic.
pub fn silu_nchwc(input: &Tensor) -> Result<Tensor, KernelError> {
    let block = nchwc_block(input)?;
    let data = input.try_data().map_err(KernelError::from)?;
    let mut out_data = data.to_vec();
    super::simd::silu_inplace(&mut out_data);
    let out = Tensor::from_vec(input.shape().to_vec(), out_data).map_err(KernelError::from)?;
    Ok(out.with_layout(Layout::NCHWc { block: block as u8 }))
}
