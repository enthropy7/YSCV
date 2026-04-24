use rayon::ThreadPool;
use yscv_tensor::Tensor;

pub const DEFAULT_MATMUL_MIN_PARALLEL_OUTPUT_ELEMENTS: usize = 65_536;
pub const DEFAULT_MATMUL_MIN_PARALLEL_SHARED_DIM: usize = 128;
// WHY 262144: 256K floats = 1MB; below this, rayon dispatch overhead (~3-5us) exceeds compute savings.
pub const DEFAULT_ELEMENTWISE_MIN_PARALLEL_ELEMENTS: usize = 262_144;
/// Higher threshold for transcendental ops (sigmoid, tanh, exp, etc.)
/// where per-element compute is heavy enough that threading overhead
/// is comparable to the gain at smaller sizes (~1M elements).
#[allow(dead_code)]
pub const DEFAULT_TRANSCENDENTAL_MIN_PARALLEL_ELEMENTS: usize = 1_048_576;
// WHY 16384: 64KB per chunk (16K x 4B) fits in L1 cache; enough work per thread to amortize dispatch.
pub(crate) const PARALLEL_SLICE_CHUNK_ELEMENTS: usize = 16_384;

// Step 1 (fp32 arc): per-op-family parallel-dispatch thresholds. The
// rayon fork-join path costs ~5-8 µs/call; below this threshold the
// dispatch overhead exceeds the compute savings. Thresholds are tuned
// against the Siamese tracker profile (2026-04-19 fixed thread-local
// profiler). Env overrides `YSCV_MIN_PAR_POINTWISE_CONV_ELEMS`,
// `YSCV_MIN_PAR_POINTWISE_CONV_FLOPS` etc. allow post-landing tuning.

/// Minimum output elements AND FLOPs for pointwise (1×1) Conv to take
/// the row-parallel path. 16 384 elems = 64 KB output (~2 L1 cache
/// lines per thread × 6 threads). 1.5 MFlops ≈ 200 µs compute @ 8 GFLOPS
/// sustained single-core AVX2; 6-way parallel saves ~170 µs, amortizes
/// 6 µs × 6 wakeup overhead.
pub const DEFAULT_POINTWISE_CONV_MIN_PARALLEL_ELEMENTS: usize = 16_384;
pub const DEFAULT_POINTWISE_CONV_MIN_PARALLEL_FLOPS: usize = 1_500_000;

/// Parallel heuristics for CPU elementwise operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParallelElementwiseConfig {
    /// Minimum number of tensor elements required before parallel elementwise execution.
    pub min_parallel_elements: usize,
}

impl ParallelElementwiseConfig {
    /// Disable parallel execution and force sequential elementwise execution.
    pub const fn disabled() -> Self {
        Self {
            min_parallel_elements: usize::MAX,
        }
    }
}

impl Default for ParallelElementwiseConfig {
    fn default() -> Self {
        Self {
            min_parallel_elements: DEFAULT_ELEMENTWISE_MIN_PARALLEL_ELEMENTS,
        }
    }
}

/// Parallel heuristics for CPU matmul row-splitting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParallelMatmulConfig {
    /// Minimum `m * n` output cells needed before row-parallel execution is considered.
    pub min_parallel_output_elements: usize,
    /// Minimum shared dimension (`k`) needed before row-parallel execution is considered.
    pub min_parallel_shared_dim: usize,
}

impl ParallelMatmulConfig {
    /// Disable parallel execution and force sequential matmul path.
    pub const fn disabled() -> Self {
        Self {
            min_parallel_output_elements: usize::MAX,
            min_parallel_shared_dim: usize::MAX,
        }
    }
}

impl Default for ParallelMatmulConfig {
    fn default() -> Self {
        Self {
            min_parallel_output_elements: DEFAULT_MATMUL_MIN_PARALLEL_OUTPUT_ELEMENTS,
            min_parallel_shared_dim: DEFAULT_MATMUL_MIN_PARALLEL_SHARED_DIM,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct MatMulPlan {
    pub m: usize,
    pub k: usize,
    pub n: usize,
    pub output_len: usize,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct Pool2dPlan {
    pub batch: usize,
    pub in_h: usize,
    pub in_w: usize,
    pub channels: usize,
    pub out_h: usize,
    pub out_w: usize,
    pub kernel_h: usize,
    pub kernel_w: usize,
    pub stride_h: usize,
    pub stride_w: usize,
    pub output_len: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct Pool2dSpec {
    pub kernel_h: usize,
    pub kernel_w: usize,
    pub stride_h: usize,
    pub stride_w: usize,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct Conv2dPlan {
    pub batch: usize,
    pub in_h: usize,
    pub in_w: usize,
    pub in_channels: usize,
    pub out_h: usize,
    pub out_w: usize,
    pub out_channels: usize,
    pub kernel_h: usize,
    pub kernel_w: usize,
    pub stride_h: usize,
    pub stride_w: usize,
    pub output_len: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct Conv2dSpec {
    pub stride_h: usize,
    pub stride_w: usize,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct DepthwiseConv2dPlan {
    pub batch: usize,
    pub in_h: usize,
    pub in_w: usize,
    pub channels: usize,
    pub depth_multiplier: usize,
    pub out_h: usize,
    pub out_w: usize,
    pub out_channels: usize,
    pub kernel_h: usize,
    pub kernel_w: usize,
    pub stride_h: usize,
    pub stride_w: usize,
    pub output_len: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct DepthwiseConv2dSpec {
    pub stride_h: usize,
    pub stride_w: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct SeparableConv2dSpec {
    pub stride_h: usize,
    pub stride_w: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct SeparableConv2dKernels<'a> {
    pub depthwise_kernel: &'a Tensor,
    pub depthwise_bias: Option<&'a Tensor>,
    pub pointwise_kernel: &'a Tensor,
    pub pointwise_bias: Option<&'a Tensor>,
}

#[derive(Debug, Clone, Copy)]
pub struct BatchNorm2dTensors<'a> {
    pub gamma: &'a Tensor,
    pub beta: &'a Tensor,
    pub mean: &'a Tensor,
    pub variance: &'a Tensor,
    pub epsilon: f32,
}

#[derive(Debug, Clone, Copy)]
pub struct LayerNormLastDimTensors<'a> {
    pub gamma: &'a Tensor,
    pub beta: &'a Tensor,
    pub epsilon: f32,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct BatchNorm2dPlan {
    pub batch: usize,
    pub height: usize,
    pub width: usize,
    pub channels: usize,
    pub output_len: usize,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct SoftmaxPlan {
    pub row_len: usize,
    pub output_len: usize,
}

#[derive(Debug, Clone)]
pub(crate) struct LogSumExpPlan {
    pub row_len: usize,
    pub output_shape: Vec<usize>,
    pub output_len: usize,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct LayerNormPlan {
    pub row_len: usize,
    pub output_len: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct GroupNorm2dTensors<'a> {
    pub gamma: &'a Tensor,
    pub beta: &'a Tensor,
    pub num_groups: usize,
    pub epsilon: f32,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct GroupNorm2dPlan {
    pub batch: usize,
    pub height: usize,
    pub width: usize,
    pub channels: usize,
    pub num_groups: usize,
    pub channels_per_group: usize,
    pub output_len: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct RmsNormLastDimTensors<'a> {
    pub gamma: &'a Tensor,
    pub epsilon: f32,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct RmsNormPlan {
    pub row_len: usize,
    pub output_len: usize,
}

#[derive(Debug, Clone, Copy)]
pub enum BinaryKind {
    Add,
    Sub,
    Mul,
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum Pool2dKind {
    Max,
    Avg,
}

pub(crate) fn should_parallelize_len(
    len: usize,
    min_parallel_len: usize,
    thread_pool: Option<&ThreadPool>,
) -> bool {
    if cfg!(miri) {
        return false;
    }
    if len < min_parallel_len {
        return false;
    }
    available_threads(thread_pool) > 1
}

pub(crate) fn available_threads(thread_pool: Option<&ThreadPool>) -> usize {
    thread_pool
        .map(ThreadPool::current_num_threads)
        .unwrap_or_else(rayon::current_num_threads)
}
