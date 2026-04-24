use rayon::ThreadPool;
use yscv_tensor::{AlignedVec, Tensor};

use super::super::error::KernelError;
use super::config::{
    BinaryKind, PARALLEL_SLICE_CHUNK_ELEMENTS, ParallelElementwiseConfig, should_parallelize_len,
};
use super::simd::{
    binary_same_shape_dispatch, exp_slice_dispatch, relu_slice_dispatch, relu_to_slice_dispatch,
    sigmoid_slice_dispatch, silu_slice_dispatch, tanh_slice_dispatch,
};

/// Elementwise ReLU activation.
#[inline]
#[allow(unsafe_code)]
pub fn relu(input: &Tensor) -> Tensor {
    let input_data = input.data();
    let len = input_data.len();
    let mut output = AlignedVec::<f32>::uninitialized(len);

    const PAR_THRESH: usize = 100_000;
    if len >= PAR_THRESH {
        let n_chunks = std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(4);
        let chunk = len.div_ceil(n_chunks);
        // SAFETY: Each chunk accesses disjoint, non-overlapping index ranges.
        // input_data lives for the duration of this function. output is owned.
        let in_base = input_data.as_ptr();
        let out_base = output.as_mut_ptr();
        std::thread::scope(|s| {
            for t in 0..n_chunks {
                let start = t * chunk;
                let end = (start + chunk).min(len);
                if start >= end {
                    break;
                }
                let inp = unsafe { std::slice::from_raw_parts(in_base.add(start), end - start) };
                let out =
                    unsafe { std::slice::from_raw_parts_mut(out_base.add(start), end - start) };
                s.spawn(move || {
                    relu_to_slice_dispatch(inp, out);
                });
            }
        });
    } else {
        relu_to_slice_dispatch(input_data, &mut output);
    }

    Tensor::from_raw_parts(input.shape(), input.strides(), output)
}

/// In-place element-wise add: `lhs += rhs`. Same-shape only.
#[inline]
pub fn add_inplace(lhs: &mut Tensor, rhs: &Tensor) {
    debug_assert_eq!(lhs.shape(), rhs.shape());
    super::simd::add_inplace_dispatch(lhs.data_mut(), rhs.data());
}

/// Fused in-place add + ReLU: `lhs[i] = max(lhs[i] + rhs[i], 0)`. Single SIMD pass.
#[inline]
pub fn add_relu_inplace(lhs: &mut Tensor, rhs: &Tensor) {
    debug_assert_eq!(lhs.shape(), rhs.shape());
    super::simd::add_relu_inplace_dispatch(lhs.data_mut(), rhs.data());
}

/// In-place ReLU activation: clamps negative values to zero.
#[inline]
pub fn relu_inplace(tensor: &mut Tensor) {
    relu_slice_dispatch(tensor.data_mut());
}

/// ReLU writing into pre-allocated output tensor. Zero allocation overhead.
#[inline]
pub fn relu_out(input: &Tensor, output: &mut Tensor) {
    debug_assert_eq!(input.shape(), output.shape());
    relu_to_slice_dispatch(input.data(), output.data_mut());
}

/// Elementwise sigmoid activation.
pub fn sigmoid(input: &Tensor) -> Tensor {
    sigmoid_with_config(input, ParallelElementwiseConfig::disabled())
}

/// Elementwise ReLU activation with explicit parallelization heuristics.
pub fn relu_with_config(input: &Tensor, config: ParallelElementwiseConfig) -> Tensor {
    relu_with_config_and_pool(input, config, None)
}

/// Elementwise sigmoid activation with explicit parallelization heuristics.
pub fn sigmoid_with_config(input: &Tensor, config: ParallelElementwiseConfig) -> Tensor {
    sigmoid_with_config_and_pool(input, config, None)
}

/// # Safety
/// `AlignedVec::uninitialized` allocates without zeroing. `relu_to_slice_dispatch`
/// writes every element before anything reads from the buffer.
#[allow(unsafe_code)]
pub fn relu_with_config_and_pool(
    input: &Tensor,
    config: ParallelElementwiseConfig,
    thread_pool: Option<&ThreadPool>,
) -> Tensor {
    let input_data = input.data();
    let len = input_data.len();
    let mut output = AlignedVec::<f32>::uninitialized(len);
    if should_parallelize_len(len, config.min_parallel_elements, thread_pool) {
        super::super::scope_ctx::par_chunks_mut_dispatch(
            output.as_mut_slice(),
            PARALLEL_SLICE_CHUNK_ELEMENTS,
            |chunk_idx, out_chunk| {
                let start = chunk_idx * PARALLEL_SLICE_CHUNK_ELEMENTS;
                let end = start + out_chunk.len();
                relu_to_slice_dispatch(&input_data[start..end], out_chunk);
            },
        );
    } else {
        relu_to_slice_dispatch(input_data, &mut output);
    }
    Tensor::from_raw_parts(input.shape(), input.strides(), output)
}

/// # Safety
/// `AlignedVec::uninitialized` allocates without zeroing. `sigmoid_slice_dispatch`
/// writes every element before anything reads from the buffer.
#[allow(unsafe_code)]
pub fn sigmoid_with_config_and_pool(
    input: &Tensor,
    _config: ParallelElementwiseConfig,
    _thread_pool: Option<&ThreadPool>,
) -> Tensor {
    let input_data = input.data();
    let len = input_data.len();
    // Sigmoid uses a heavy polynomial exp + divide per element.
    // Single-threaded SIMD is faster than rayon chunking for ≤4M elements
    // due to dispatch overhead (62 tasks × 50µs > compute savings).
    let mut output = AlignedVec::<f32>::uninitialized(len);
    sigmoid_slice_dispatch(input_data, &mut output);
    Tensor::from_raw_parts(input.shape(), input.strides(), output)
}

/// Elementwise exp activation.
pub fn exp(input: &Tensor) -> Tensor {
    exp_with_config(input, ParallelElementwiseConfig::disabled())
}

/// Elementwise exp activation with explicit parallelization heuristics.
pub fn exp_with_config(input: &Tensor, config: ParallelElementwiseConfig) -> Tensor {
    exp_with_config_and_pool(input, config, None)
}

/// # Safety
/// `AlignedVec::uninitialized` allocates without zeroing. `exp_slice_dispatch`
/// writes every element before anything reads from the buffer.
#[allow(unsafe_code)]
pub fn exp_with_config_and_pool(
    input: &Tensor,
    config: ParallelElementwiseConfig,
    thread_pool: Option<&ThreadPool>,
) -> Tensor {
    let input_data = input.data();
    let len = input_data.len();
    let mut output = AlignedVec::<f32>::uninitialized(len);
    if should_parallelize_len(len, config.min_parallel_elements, thread_pool) {
        super::super::scope_ctx::par_chunks_mut_dispatch(
            output.as_mut_slice(),
            PARALLEL_SLICE_CHUNK_ELEMENTS,
            |chunk_idx, out_chunk| {
                let start = chunk_idx * PARALLEL_SLICE_CHUNK_ELEMENTS;
                let end = start + out_chunk.len();
                exp_slice_dispatch(&input_data[start..end], out_chunk);
            },
        );
    } else {
        exp_slice_dispatch(input_data, &mut output);
    }
    Tensor::from_raw_parts(input.shape(), input.strides(), output)
}

/// Elementwise tanh activation.
pub fn tanh_act(input: &Tensor) -> Tensor {
    tanh_act_with_config(input, ParallelElementwiseConfig::disabled())
}

/// Elementwise tanh activation with explicit parallelization heuristics.
pub fn tanh_act_with_config(input: &Tensor, config: ParallelElementwiseConfig) -> Tensor {
    tanh_act_with_config_and_pool(input, config, None)
}

/// # Safety
/// `AlignedVec::uninitialized` allocates without zeroing. `tanh_slice_dispatch`
/// writes every element before anything reads from the buffer.
#[allow(unsafe_code)]
pub fn tanh_act_with_config_and_pool(
    input: &Tensor,
    _config: ParallelElementwiseConfig,
    _thread_pool: Option<&ThreadPool>,
) -> Tensor {
    let input_data = input.data();
    let len = input_data.len();
    // Tanh uses a heavy polynomial exp + divide per element.
    // Single-threaded SIMD is faster than rayon chunking for typical sizes.
    let mut output = AlignedVec::<f32>::uninitialized(len);
    tanh_slice_dispatch(input_data, &mut output);
    Tensor::from_raw_parts(input.shape(), input.strides(), output)
}

const ACTIVATION_PARALLEL_THRESHOLD: usize = 65536;
const ACTIVATION_CHUNK_SIZE: usize = 8192;

/// Elementwise GELU activation (fast approximation): `x * sigmoid(1.702 * x)`.
///
/// # Safety
/// `AlignedVec::uninitialized` allocates without zeroing. `gelu_slice_out`
/// writes every element before anything reads from the buffer.
#[allow(unsafe_code)]
pub fn gelu(input: &Tensor) -> Tensor {
    let src = input.data();
    let len = src.len();
    let mut output = AlignedVec::<f32>::uninitialized(len);
    if len >= ACTIVATION_PARALLEL_THRESHOLD {
        super::super::scope_ctx::par_chunks_mut_dispatch(
            output.as_mut_slice(),
            ACTIVATION_CHUNK_SIZE,
            |ci, out_chunk| {
                let start = ci * ACTIVATION_CHUNK_SIZE;
                gelu_slice_out(&src[start..start + out_chunk.len()], out_chunk);
            },
        );
    } else {
        gelu_slice_out(src, &mut output);
    }
    Tensor::from_raw_parts(input.shape(), input.strides(), output)
}

/// Elementwise SiLU (Swish) activation: `x * sigmoid(x)`.
///
/// Uses fused SIMD kernel (sigmoid + multiply in one pass) to halve memory bandwidth.
pub fn silu(input: &Tensor) -> Tensor {
    silu_with_config(input, ParallelElementwiseConfig::disabled())
}

/// Elementwise SiLU activation with explicit parallelization heuristics.
pub fn silu_with_config(input: &Tensor, config: ParallelElementwiseConfig) -> Tensor {
    silu_with_config_and_pool(input, config, None)
}

/// # Safety
/// `AlignedVec::uninitialized` allocates without zeroing. `silu_slice_dispatch`
/// writes every element before anything reads from the buffer.
#[allow(unsafe_code)]
pub fn silu_with_config_and_pool(
    input: &Tensor,
    _config: ParallelElementwiseConfig,
    _thread_pool: Option<&ThreadPool>,
) -> Tensor {
    let input_data = input.data();
    let len = input_data.len();
    // SiLU uses a heavy polynomial exp + divide + multiply per element.
    // Single-threaded SIMD is faster than rayon chunking for typical sizes.
    let mut output = AlignedVec::<f32>::uninitialized(len);
    silu_slice_dispatch(input_data, &mut output);
    Tensor::from_raw_parts(input.shape(), input.strides(), output)
}

/// In-place SiLU: applies SiLU to a mutable tensor, avoiding allocation.
/// The SIMD kernels load all inputs before storing outputs within each block,
/// so aliasing input == output is safe.
/// Uses rayon parallelism for large tensors (>100K elements).
pub fn silu_inplace(tensor: &mut Tensor) {
    let data = tensor.data_mut();
    let len = data.len();
    const PARALLEL_THRESHOLD: usize = 100_000;
    if len >= PARALLEL_THRESHOLD {
        // Parallel: split into chunks processed by different cores.
        super::super::scope_ctx::par_chunks_mut_dispatch(data, 32_768, |_ci, chunk| {
            let ptr = chunk.as_mut_ptr();
            let clen = chunk.len();
            #[allow(unsafe_code)]
            unsafe {
                let input_slice = std::slice::from_raw_parts(ptr, clen);
                let output_slice = std::slice::from_raw_parts_mut(ptr, clen);
                silu_slice_dispatch(input_slice, output_slice);
            }
        });
    } else {
        // Small tensor: single-threaded SIMD.
        let ptr = data.as_mut_ptr();
        #[allow(unsafe_code)]
        unsafe {
            let input_slice = std::slice::from_raw_parts(ptr, len);
            let output_slice = std::slice::from_raw_parts_mut(ptr, len);
            silu_slice_dispatch(input_slice, output_slice);
        }
    }
}

/// Elementwise Mish activation: `x * tanh(softplus(x))` = `x * tanh(ln(1 + exp(x)))`.
pub fn mish(input: &Tensor) -> Tensor {
    let mut output = input.clone();
    let data = output.data_mut();
    if data.len() >= ACTIVATION_PARALLEL_THRESHOLD {
        super::super::scope_ctx::par_chunks_mut_dispatch(
            data,
            ACTIVATION_CHUNK_SIZE,
            |_ci, chunk| mish_slice(chunk),
        );
    } else {
        mish_slice(data);
    }
    output
}

fn gelu_slice_out(src: &[f32], dst: &mut [f32]) {
    for i in 0..src.len() {
        let x = src[i];
        let a = 1.702 * x;
        let ea = (-a).exp();
        let s = 1.0 / (1.0 + ea);
        dst[i] = x * s;
    }
}

fn mish_slice(data: &mut [f32]) {
    for i in 0..data.len() {
        let x = data[i];
        let sp = (1.0 + x.exp()).ln();
        data[i] = x * sp.tanh();
    }
}

/// Elementwise add with optional parallel same-shape execution.
pub fn add_with_config(
    lhs: &Tensor,
    rhs: &Tensor,
    config: ParallelElementwiseConfig,
) -> Result<Tensor, KernelError> {
    add_with_config_and_pool(lhs, rhs, config, None)
}

pub fn add_with_config_and_pool(
    lhs: &Tensor,
    rhs: &Tensor,
    config: ParallelElementwiseConfig,
    thread_pool: Option<&ThreadPool>,
) -> Result<Tensor, KernelError> {
    binary_with_config_and_pool(lhs, rhs, config, thread_pool, BinaryKind::Add)
}

/// Elementwise subtract with optional parallel same-shape execution.
pub fn sub_with_config(
    lhs: &Tensor,
    rhs: &Tensor,
    config: ParallelElementwiseConfig,
) -> Result<Tensor, KernelError> {
    sub_with_config_and_pool(lhs, rhs, config, None)
}

pub fn sub_with_config_and_pool(
    lhs: &Tensor,
    rhs: &Tensor,
    config: ParallelElementwiseConfig,
    thread_pool: Option<&ThreadPool>,
) -> Result<Tensor, KernelError> {
    binary_with_config_and_pool(lhs, rhs, config, thread_pool, BinaryKind::Sub)
}

/// Elementwise multiply with optional parallel same-shape execution.
pub fn mul_with_config(
    lhs: &Tensor,
    rhs: &Tensor,
    config: ParallelElementwiseConfig,
) -> Result<Tensor, KernelError> {
    mul_with_config_and_pool(lhs, rhs, config, None)
}

pub fn mul_with_config_and_pool(
    lhs: &Tensor,
    rhs: &Tensor,
    config: ParallelElementwiseConfig,
    thread_pool: Option<&ThreadPool>,
) -> Result<Tensor, KernelError> {
    binary_with_config_and_pool(lhs, rhs, config, thread_pool, BinaryKind::Mul)
}

fn binary_with_config_and_pool(
    lhs: &Tensor,
    rhs: &Tensor,
    config: ParallelElementwiseConfig,
    thread_pool: Option<&ThreadPool>,
    kind: BinaryKind,
) -> Result<Tensor, KernelError> {
    if lhs.shape() != rhs.shape() {
        return binary_fallback(lhs, rhs, kind);
    }

    let left = lhs.data();
    let right = rhs.data();
    let shape = lhs.shape().to_vec();
    let mut output = AlignedVec::<f32>::uninitialized(left.len());

    if should_parallelize_len(left.len(), config.min_parallel_elements, thread_pool) {
        super::super::scope_ctx::par_chunks_mut_dispatch(
            output.as_mut_slice(),
            PARALLEL_SLICE_CHUNK_ELEMENTS,
            |chunk_idx, out_chunk| {
                let start = chunk_idx * PARALLEL_SLICE_CHUNK_ELEMENTS;
                let end = start + out_chunk.len();
                binary_same_shape_dispatch(&left[start..end], &right[start..end], out_chunk, kind);
            },
        );
    } else {
        binary_same_shape_dispatch(left, right, &mut output, kind);
    }

    Tensor::from_aligned(shape, output).map_err(Into::into)
}

fn binary_fallback(lhs: &Tensor, rhs: &Tensor, kind: BinaryKind) -> Result<Tensor, KernelError> {
    match kind {
        BinaryKind::Add => lhs.add(rhs),
        BinaryKind::Sub => lhs.sub(rhs),
        BinaryKind::Mul => lhs.mul(rhs),
    }
    .map_err(Into::into)
}
