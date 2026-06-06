use rayon::ThreadPool;
use yscv_tensor::{AlignedVec, Tensor, TensorError};

use super::super::error::KernelError;
use super::config::{
    BinaryKind, PARALLEL_SLICE_CHUNK_ELEMENTS, ParallelElementwiseConfig, should_parallelize_len,
};
use super::simd::{
    binary_broadcast_lastdim_dispatch, binary_same_shape_dispatch, exp_slice_dispatch,
    gelu_sigmoid_slice_dispatch, relu_slice_dispatch, relu_to_slice_dispatch,
    sigmoid_slice_dispatch, silu_slice_dispatch, tanh_slice_dispatch,
};

// 48K floats gives exp enough rows per task to amortize rayon wake-up while
// still feeding all cores on 1M-element activation workloads.
const EXP_PARALLEL_CHUNK_ELEMENTS: usize = 49_152;

/// Elementwise ReLU activation.
#[inline]
pub fn relu(input: &Tensor) -> Tensor {
    relu_with_config(input, ParallelElementwiseConfig::default())
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

/// ReLU writing into a pre-allocated output tensor with explicit parallelization.
pub fn relu_out_with_config(
    input: &Tensor,
    output: &mut Tensor,
    config: ParallelElementwiseConfig,
) -> Result<(), KernelError> {
    relu_out_with_config_and_pool(input, output, config, None)
}

/// ReLU writing into a pre-allocated output tensor with explicit pool routing.
pub fn relu_out_with_config_and_pool(
    input: &Tensor,
    output: &mut Tensor,
    config: ParallelElementwiseConfig,
    thread_pool: Option<&ThreadPool>,
) -> Result<(), KernelError> {
    ensure_same_shape(input, output)?;
    let input_data = input.data();
    let len = input_data.len();
    let output_data = output.data_mut();
    if should_parallelize_len(len, config.min_parallel_elements, thread_pool) {
        let chunk_elements = parallel_chunk_elements_for_threads(len, thread_pool);
        super::super::scope_ctx::par_chunks_mut_dispatch(
            output_data,
            chunk_elements,
            |chunk_idx, out_chunk| {
                let start = chunk_idx * chunk_elements;
                let end = start + out_chunk.len();
                relu_to_slice_dispatch(&input_data[start..end], out_chunk);
            },
        );
    } else {
        relu_to_slice_dispatch(input_data, output_data);
    }
    Ok(())
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
        let chunk_elements = parallel_chunk_elements_for_threads(len, thread_pool);
        super::super::scope_ctx::par_chunks_mut_dispatch(
            output.as_mut_slice(),
            chunk_elements,
            |chunk_idx, out_chunk| {
                let start = chunk_idx * chunk_elements;
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
                sigmoid_slice_dispatch(&input_data[start..end], out_chunk);
            },
        );
    } else {
        sigmoid_slice_dispatch(input_data, &mut output);
    }
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
            EXP_PARALLEL_CHUNK_ELEMENTS,
            |chunk_idx, out_chunk| {
                let start = chunk_idx * EXP_PARALLEL_CHUNK_ELEMENTS;
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
                tanh_slice_dispatch(&input_data[start..end], out_chunk);
            },
        );
    } else {
        tanh_slice_dispatch(input_data, &mut output);
    }
    Tensor::from_raw_parts(input.shape(), input.strides(), output)
}

const ACTIVATION_PARALLEL_THRESHOLD: usize = 65536;
const ACTIVATION_CHUNK_SIZE: usize = 8192;

/// Elementwise GELU activation (fast approximation): `x * sigmoid(1.702 * x)`.
///
/// # Safety
/// `AlignedVec::uninitialized` allocates without zeroing. `gelu_sigmoid_slice_dispatch`
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
                gelu_sigmoid_slice_dispatch(&src[start..start + out_chunk.len()], out_chunk);
            },
        );
    } else {
        gelu_sigmoid_slice_dispatch(src, &mut output);
    }
    Tensor::from_raw_parts(input.shape(), input.strides(), output)
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
                silu_slice_dispatch(&input_data[start..end], out_chunk);
            },
        );
    } else {
        silu_slice_dispatch(input_data, &mut output);
    }
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

/// Elementwise add with an explicit parallel config and thread pool.
pub fn add_with_config_and_pool(
    lhs: &Tensor,
    rhs: &Tensor,
    config: ParallelElementwiseConfig,
    thread_pool: Option<&ThreadPool>,
) -> Result<Tensor, KernelError> {
    binary_with_config_and_pool(lhs, rhs, config, thread_pool, BinaryKind::Add)
}

/// Elementwise add writing into a pre-allocated output tensor.
pub fn add_out(lhs: &Tensor, rhs: &Tensor, output: &mut Tensor) -> Result<(), KernelError> {
    add_out_with_config(lhs, rhs, output, ParallelElementwiseConfig::default())
}

/// Elementwise add writing into a pre-allocated output tensor with explicit config.
pub fn add_out_with_config(
    lhs: &Tensor,
    rhs: &Tensor,
    output: &mut Tensor,
    config: ParallelElementwiseConfig,
) -> Result<(), KernelError> {
    binary_out_with_config_and_pool(lhs, rhs, output, config, None, BinaryKind::Add)
}

/// Elementwise subtract with optional parallel same-shape execution.
pub fn sub_with_config(
    lhs: &Tensor,
    rhs: &Tensor,
    config: ParallelElementwiseConfig,
) -> Result<Tensor, KernelError> {
    sub_with_config_and_pool(lhs, rhs, config, None)
}

/// Elementwise subtract with an explicit parallel config and thread pool.
pub fn sub_with_config_and_pool(
    lhs: &Tensor,
    rhs: &Tensor,
    config: ParallelElementwiseConfig,
    thread_pool: Option<&ThreadPool>,
) -> Result<Tensor, KernelError> {
    binary_with_config_and_pool(lhs, rhs, config, thread_pool, BinaryKind::Sub)
}

/// Elementwise subtract writing into a pre-allocated output tensor.
pub fn sub_out(lhs: &Tensor, rhs: &Tensor, output: &mut Tensor) -> Result<(), KernelError> {
    sub_out_with_config(lhs, rhs, output, ParallelElementwiseConfig::default())
}

/// Elementwise subtract writing into a pre-allocated output tensor with explicit config.
pub fn sub_out_with_config(
    lhs: &Tensor,
    rhs: &Tensor,
    output: &mut Tensor,
    config: ParallelElementwiseConfig,
) -> Result<(), KernelError> {
    binary_out_with_config_and_pool(lhs, rhs, output, config, None, BinaryKind::Sub)
}

/// Elementwise multiply with optional parallel same-shape execution.
pub fn mul_with_config(
    lhs: &Tensor,
    rhs: &Tensor,
    config: ParallelElementwiseConfig,
) -> Result<Tensor, KernelError> {
    mul_with_config_and_pool(lhs, rhs, config, None)
}

/// Elementwise multiply with an explicit parallel config and thread pool.
pub fn mul_with_config_and_pool(
    lhs: &Tensor,
    rhs: &Tensor,
    config: ParallelElementwiseConfig,
    thread_pool: Option<&ThreadPool>,
) -> Result<Tensor, KernelError> {
    binary_with_config_and_pool(lhs, rhs, config, thread_pool, BinaryKind::Mul)
}

/// Elementwise multiply writing into a pre-allocated output tensor.
pub fn mul_out(lhs: &Tensor, rhs: &Tensor, output: &mut Tensor) -> Result<(), KernelError> {
    mul_out_with_config(lhs, rhs, output, ParallelElementwiseConfig::default())
}

/// Elementwise multiply writing into a pre-allocated output tensor with explicit config.
pub fn mul_out_with_config(
    lhs: &Tensor,
    rhs: &Tensor,
    output: &mut Tensor,
    config: ParallelElementwiseConfig,
) -> Result<(), KernelError> {
    binary_out_with_config_and_pool(lhs, rhs, output, config, None, BinaryKind::Mul)
}

fn binary_with_config_and_pool(
    lhs: &Tensor,
    rhs: &Tensor,
    config: ParallelElementwiseConfig,
    thread_pool: Option<&ThreadPool>,
    kind: BinaryKind,
) -> Result<Tensor, KernelError> {
    if lhs.shape() != rhs.shape() {
        if let Some(result) =
            binary_broadcast_lastdim_with_config_and_pool(lhs, rhs, config, thread_pool, kind)
        {
            return result;
        }
        return binary_fallback(lhs, rhs, kind);
    }

    let left = lhs.data();
    let right = rhs.data();
    let mut output = AlignedVec::<f32>::uninitialized(left.len());

    if should_parallelize_len(left.len(), config.min_parallel_elements, thread_pool) {
        let chunk_elements = binary_parallel_chunk_elements(left.len(), thread_pool);
        super::super::scope_ctx::par_chunks_mut_dispatch(
            output.as_mut_slice(),
            chunk_elements,
            |chunk_idx, out_chunk| {
                let start = chunk_idx * chunk_elements;
                let end = start + out_chunk.len();
                binary_same_shape_dispatch(&left[start..end], &right[start..end], out_chunk, kind);
            },
        );
    } else {
        binary_same_shape_dispatch(left, right, &mut output, kind);
    }

    Ok(Tensor::from_raw_parts(lhs.shape(), lhs.strides(), output))
}

fn binary_out_with_config_and_pool(
    lhs: &Tensor,
    rhs: &Tensor,
    output: &mut Tensor,
    config: ParallelElementwiseConfig,
    thread_pool: Option<&ThreadPool>,
    kind: BinaryKind,
) -> Result<(), KernelError> {
    ensure_same_shape(lhs, rhs)?;
    ensure_same_shape(lhs, output)?;

    let left = lhs.data();
    let right = rhs.data();
    let len = left.len();
    let output_data = output.data_mut();

    if should_parallelize_len(len, config.min_parallel_elements, thread_pool) {
        let chunk_elements = binary_parallel_chunk_elements(len, thread_pool);
        super::super::scope_ctx::par_chunks_mut_dispatch(
            output_data,
            chunk_elements,
            |chunk_idx, out_chunk| {
                let start = chunk_idx * chunk_elements;
                let end = start + out_chunk.len();
                binary_same_shape_dispatch(&left[start..end], &right[start..end], out_chunk, kind);
            },
        );
    } else {
        binary_same_shape_dispatch(left, right, output_data, kind);
    }

    Ok(())
}

fn ensure_same_shape(lhs: &Tensor, rhs: &Tensor) -> Result<(), KernelError> {
    if lhs.shape() == rhs.shape() {
        return Ok(());
    }
    Err(TensorError::ShapeMismatch {
        left: lhs.shape().to_vec(),
        right: rhs.shape().to_vec(),
    }
    .into())
}

fn binary_parallel_chunk_elements(len: usize, thread_pool: Option<&ThreadPool>) -> usize {
    parallel_chunk_elements_for_threads(len, thread_pool)
}

fn binary_broadcast_lastdim_with_config_and_pool(
    lhs: &Tensor,
    rhs: &Tensor,
    config: ParallelElementwiseConfig,
    thread_pool: Option<&ThreadPool>,
    kind: BinaryKind,
) -> Option<Result<Tensor, KernelError>> {
    let lhs_shape = lhs.shape();
    let rhs_shape = rhs.shape();
    let lhs_last = *lhs_shape.last()?;
    let rhs_last = *rhs_shape.last()?;
    if lhs_last == 0 || lhs_last != rhs_last {
        return None;
    }

    let rhs_is_lastdim_vec = rhs_shape.iter().rev().skip(1).all(|&dim| dim == 1);
    let lhs_is_lastdim_vec = lhs_shape.iter().rev().skip(1).all(|&dim| dim == 1);

    if rhs_is_lastdim_vec && !lhs_is_lastdim_vec {
        if rhs_shape.len() > lhs_shape.len() {
            return None;
        }
        let big = lhs.data();
        let row = &rhs.data()[..lhs_last];
        let mut output = AlignedVec::<f32>::uninitialized(big.len());
        binary_broadcast_lastdim_rows(
            big,
            row,
            output.as_mut_slice(),
            lhs_last,
            true,
            config,
            thread_pool,
            kind,
        );
        return Some(Ok(Tensor::from_raw_parts(
            lhs.shape(),
            lhs.strides(),
            output,
        )));
    }

    if lhs_is_lastdim_vec && !rhs_is_lastdim_vec {
        if lhs_shape.len() > rhs_shape.len() {
            return None;
        }
        let row = &lhs.data()[..rhs_last];
        let big = rhs.data();
        let mut output = AlignedVec::<f32>::uninitialized(big.len());
        binary_broadcast_lastdim_rows(
            big,
            row,
            output.as_mut_slice(),
            rhs_last,
            false,
            config,
            thread_pool,
            kind,
        );
        return Some(Ok(Tensor::from_raw_parts(
            rhs.shape(),
            rhs.strides(),
            output,
        )));
    }

    None
}

fn binary_broadcast_lastdim_rows(
    big: &[f32],
    row: &[f32],
    output: &mut [f32],
    row_len: usize,
    big_is_lhs: bool,
    config: ParallelElementwiseConfig,
    thread_pool: Option<&ThreadPool>,
    kind: BinaryKind,
) {
    debug_assert_eq!(big.len(), output.len());
    debug_assert_eq!(row.len(), row_len);
    debug_assert_eq!(big.len() % row_len, 0);

    if should_parallelize_len(big.len(), config.min_parallel_elements, thread_pool) {
        let chunk_elements = binary_parallel_chunk_elements(big.len(), thread_pool)
            .max(row_len)
            .div_ceil(row_len)
            * row_len;
        super::super::scope_ctx::par_chunks_mut_dispatch(
            output,
            chunk_elements,
            |chunk_idx, out_chunk| {
                let start = chunk_idx * chunk_elements;
                binary_broadcast_lastdim_chunk(
                    &big[start..start + out_chunk.len()],
                    row,
                    out_chunk,
                    row_len,
                    big_is_lhs,
                    kind,
                );
            },
        );
    } else {
        binary_broadcast_lastdim_chunk(big, row, output, row_len, big_is_lhs, kind);
    }
}

fn binary_broadcast_lastdim_chunk(
    big: &[f32],
    row: &[f32],
    output: &mut [f32],
    row_len: usize,
    big_is_lhs: bool,
    kind: BinaryKind,
) {
    for (big_row, out_row) in big
        .chunks_exact(row_len)
        .zip(output.chunks_exact_mut(row_len))
    {
        binary_broadcast_lastdim_dispatch(big_row, row, out_row, row_len, big_is_lhs, kind);
    }
}

fn parallel_chunk_elements_for_threads(len: usize, thread_pool: Option<&ThreadPool>) -> usize {
    let threads = thread_pool
        .map(ThreadPool::current_num_threads)
        .unwrap_or_else(rayon::current_num_threads)
        .max(1);
    len.div_ceil(threads).max(PARALLEL_SLICE_CHUNK_ELEMENTS)
}

fn binary_fallback(lhs: &Tensor, rhs: &Tensor, kind: BinaryKind) -> Result<Tensor, KernelError> {
    match kind {
        BinaryKind::Add => lhs.add(rhs),
        BinaryKind::Sub => lhs.sub(rhs),
        BinaryKind::Mul => lhs.mul(rhs),
    }
    .map_err(Into::into)
}
