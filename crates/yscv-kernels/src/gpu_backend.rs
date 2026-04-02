//! wgpu-based GPU backend — dispatches to Vulkan (Linux/Win), Metal (macOS), DX12 (Win).

use wgpu::util::DeviceExt;
use yscv_tensor::Tensor;

use crate::{
    Backend, BatchNorm2dParams, GroupNormNhwcParams, KernelError, LayerNormLastDimParams,
    RmsNormLastDimParams, SeparableConv2dParams,
};

const MIN_GPU_ELEMENTS: usize = 4096;

// ── WGSL Shaders (loaded from external files) ────────────────────

const MATMUL_WGSL: &str = include_str!("shaders/matmul.wgsl");
const ELEMENTWISE_WGSL: &str = include_str!("shaders/elementwise.wgsl");
const UNARY_WGSL: &str = include_str!("shaders/unary.wgsl");
const SOFTMAX_WGSL: &str = include_str!("shaders/softmax.wgsl");
const LOG_SOFTMAX_WGSL: &str = include_str!("shaders/log_softmax.wgsl");
const LOGSUMEXP_WGSL: &str = include_str!("shaders/logsumexp.wgsl");
const CONV2D_WGSL: &str = include_str!("shaders/conv2d.wgsl");
const POOL2D_WGSL: &str = include_str!("shaders/pool2d.wgsl");
const BATCH_NORM_WGSL: &str = include_str!("shaders/batch_norm.wgsl");
const LAYER_NORM_WGSL: &str = include_str!("shaders/layer_norm.wgsl");
const DEPTHWISE_CONV2D_WGSL: &str = include_str!("shaders/depthwise_conv2d.wgsl");
const TRANSPOSE_CONV2D_WGSL: &str = include_str!("shaders/transpose_conv2d.wgsl");
const TRANSPOSE_2D_WGSL: &str = include_str!("shaders/transpose_2d.wgsl");
const GATHER_WGSL: &str = include_str!("shaders/gather.wgsl");
const ATTENTION_WGSL: &str = include_str!("shaders/attention.wgsl");
const GROUP_NORM_WGSL: &str = include_str!("shaders/group_norm.wgsl");
const RMS_NORM_WGSL: &str = include_str!("shaders/rms_norm.wgsl");
const BACKWARD_BINARY_WGSL: &str = include_str!("shaders/backward_binary.wgsl");
const CONV2D_INPUT_GRAD_WGSL: &str = include_str!("shaders/conv2d_input_grad.wgsl");
const REDUCE_SUM_BACKWARD_WGSL: &str = include_str!("shaders/reduce_sum_backward.wgsl");
const CHANNEL_SPLIT_WGSL: &str = include_str!("shaders/channel_split.wgsl");
const CHANNEL_CONCAT_WGSL: &str = include_str!("shaders/channel_concat.wgsl");
const RESIZE_NEAREST_WGSL: &str = include_str!("shaders/resize_nearest.wgsl");
const PERMUTE_NHWC_NCHW_WGSL: &str = include_str!("shaders/permute_nhwc_nchw.wgsl");
const PERMUTE_ND_WGSL: &str = include_str!("shaders/permute_nd.wgsl");
const GENERAL_CONCAT_WGSL: &str = include_str!("shaders/general_concat.wgsl");
const GENERAL_SPLIT_WGSL: &str = include_str!("shaders/general_split.wgsl");
const SLICE_ND_WGSL: &str = include_str!("shaders/slice_nd.wgsl");
const BROADCAST_BINARY_WGSL: &str = include_str!("shaders/broadcast_binary.wgsl");
const IM2COL_WGSL: &str = include_str!("shaders/im2col.wgsl");
const BIAS_ADD_WGSL: &str = include_str!("shaders/bias_add.wgsl");
const BATCHED_MATMUL_WGSL: &str = include_str!("shaders/batched_matmul.wgsl");
const CONV_GEMM_WGSL: &str = include_str!("shaders/conv_gemm.wgsl");
const CONV_GEMM_F16_WGSL: &str = include_str!("shaders/conv_gemm_f16.wgsl");
const CONV_GEMM_FAST_WGSL: &str = include_str!("shaders/conv_gemm_fast.wgsl");
const CONV_GEMM_V2_WGSL: &str = include_str!("shaders/conv_gemm_v2.wgsl");
const CONV_GEMM_FAST_N32_WGSL: &str = include_str!("shaders/conv_gemm_fast_n32.wgsl");
const CONV_GEMM_F16_N32_WGSL: &str = include_str!("shaders/conv_gemm_f16_n32.wgsl");
const CONV_GEMM_F16_IO_WGSL: &str = include_str!("shaders/conv_gemm_f16_io.wgsl");
const CONV_GEMM_F16_IO_N32_WGSL: &str = include_str!("shaders/conv_gemm_f16_io_n32.wgsl");
const ELEMENTWISE_F16_WGSL: &str = include_str!("shaders/elementwise_f16.wgsl");
const UNARY_F16_WGSL: &str = include_str!("shaders/unary_f16.wgsl");
const CHANNEL_CONCAT_F16_WGSL: &str = include_str!("shaders/channel_concat_f16.wgsl");
const CHANNEL_SPLIT_F16_WGSL: &str = include_str!("shaders/channel_split_f16.wgsl");
const RESIZE_NEAREST_F16_WGSL: &str = include_str!("shaders/resize_nearest_f16.wgsl");
const PERMUTE_NHWC_NCHW_F16_WGSL: &str = include_str!("shaders/permute_nhwc_nchw_f16.wgsl");
const CONVERT_F32_TO_F16_WGSL: &str = include_str!("shaders/convert_f32_to_f16.wgsl");
const CONVERT_F16_TO_F32_WGSL: &str = include_str!("shaders/convert_f16_to_f32.wgsl");
const WINOGRAD_INPUT_WGSL: &str = include_str!("shaders/winograd_input_transform.wgsl");
const WINOGRAD_OUTPUT_WGSL: &str = include_str!("shaders/winograd_output_transform.wgsl");

// ── Pipeline cache ─────────────────────────────────────────────────

struct Pipelines {
    matmul: wgpu::ComputePipeline,
    elementwise: wgpu::ComputePipeline,
    unary: wgpu::ComputePipeline,
    softmax: wgpu::ComputePipeline,
    log_softmax: wgpu::ComputePipeline,
    logsumexp: wgpu::ComputePipeline,
    conv2d: wgpu::ComputePipeline,
    pool2d: wgpu::ComputePipeline,
    batch_norm: wgpu::ComputePipeline,
    layer_norm: wgpu::ComputePipeline,
    depthwise_conv2d: wgpu::ComputePipeline,
    transpose_conv2d: wgpu::ComputePipeline,
    transpose_2d: wgpu::ComputePipeline,
    #[allow(dead_code)]
    gather: wgpu::ComputePipeline,
    #[allow(dead_code)]
    attention: wgpu::ComputePipeline,
    group_norm: wgpu::ComputePipeline,
    rms_norm: wgpu::ComputePipeline,
    backward_binary: wgpu::ComputePipeline,
    reduce_sum_backward: wgpu::ComputePipeline,
    conv2d_input_grad: wgpu::ComputePipeline,
    channel_split: wgpu::ComputePipeline,
    channel_concat: wgpu::ComputePipeline,
    resize_nearest: wgpu::ComputePipeline,
    permute_nhwc_nchw: wgpu::ComputePipeline,
    permute_nd: wgpu::ComputePipeline,
    general_concat: wgpu::ComputePipeline,
    general_split: wgpu::ComputePipeline,
    slice_nd: wgpu::ComputePipeline,
    broadcast_binary: wgpu::ComputePipeline,
    #[allow(dead_code)]
    im2col: wgpu::ComputePipeline,
    #[allow(dead_code)]
    bias_add: wgpu::ComputePipeline,
    batched_matmul: wgpu::ComputePipeline,
    conv_gemm: wgpu::ComputePipeline,
    conv_gemm_fast: Option<wgpu::ComputePipeline>,
    conv_gemm_v2: Option<wgpu::ComputePipeline>,
    conv_gemm_f16: Option<wgpu::ComputePipeline>,
    conv_gemm_fast_n32: Option<wgpu::ComputePipeline>,
    conv_gemm_f16_n32: Option<wgpu::ComputePipeline>,
    conv_gemm_f16_io: Option<wgpu::ComputePipeline>,
    conv_gemm_f16_io_n32: Option<wgpu::ComputePipeline>,
    elementwise_f16: Option<wgpu::ComputePipeline>,
    unary_f16: Option<wgpu::ComputePipeline>,
    channel_concat_f16: Option<wgpu::ComputePipeline>,
    channel_split_f16: Option<wgpu::ComputePipeline>,
    resize_nearest_f16: Option<wgpu::ComputePipeline>,
    permute_nhwc_nchw_f16: Option<wgpu::ComputePipeline>,
    convert_f32_to_f16: Option<wgpu::ComputePipeline>,
    convert_f16_to_f32: Option<wgpu::ComputePipeline>,
    winograd_input: wgpu::ComputePipeline,
    winograd_output: wgpu::ComputePipeline,
}

// ── GpuBuffer ─────────────────────────────────────────────────────

/// A tensor that lives on GPU memory. No host copy until explicitly requested.
pub struct GpuBuffer {
    buffer: wgpu::Buffer,
    /// Number of f32 elements.
    size: usize,
    shape: Vec<usize>,
}

impl GpuBuffer {
    /// Returns the shape of this GPU-resident tensor.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Returns the number of f32 elements.
    pub fn len(&self) -> usize {
        self.size
    }

    /// Returns true if this buffer contains no elements.
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    /// Consume this GpuBuffer and return the inner wgpu::Buffer + element count
    /// so the caller can return it to a buffer pool.
    pub fn into_raw(self) -> (wgpu::Buffer, usize) {
        (self.buffer, self.size)
    }

    /// Get a reference to the underlying wgpu::Buffer.
    pub fn raw_buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }

    /// Construct a GpuBuffer from raw parts (for compiled plan replay).
    /// The caller is responsible for ensuring the buffer matches the declared size/shape.
    pub fn from_raw_parts(buffer: wgpu::Buffer, size: usize, shape: Vec<usize>) -> Self {
        Self {
            buffer,
            size,
            shape,
        }
    }
}

// ── GpuBackend ─────────────────────────────────────────────────────

/// Simple size-bucketed buffer pool for GPU buffer reuse across dispatches.
/// Uses `RefCell` for interior mutability so `&self` methods can pool/reclaim.
struct BufferPool {
    /// Available output buffers keyed by capacity in bytes.
    output: std::cell::RefCell<Vec<(u64, wgpu::Buffer)>>,
    /// Available storage buffers keyed by capacity in bytes.
    storage: std::cell::RefCell<Vec<(u64, wgpu::Buffer)>>,
    /// Maximum pool depth per category.
    max_depth: usize,
    /// Total allocations saved (diagnostic counter).
    hits: std::cell::Cell<u64>,
}

impl BufferPool {
    fn new(max_depth: usize) -> Self {
        Self {
            output: std::cell::RefCell::new(Vec::with_capacity(max_depth)),
            storage: std::cell::RefCell::new(Vec::with_capacity(max_depth)),
            max_depth,
            hits: std::cell::Cell::new(0),
        }
    }

    /// Try to reclaim an output buffer with at least `size_bytes` capacity.
    /// Uses best-fit: picks the smallest buffer that's >= size_bytes and <= 4x.
    fn take_output(&self, size_bytes: u64) -> Option<wgpu::Buffer> {
        let mut pool = self.output.borrow_mut();
        let max_cap = size_bytes.saturating_mul(4);
        let mut best: Option<(usize, u64)> = None;
        for (i, &(cap, _)) in pool.iter().enumerate() {
            if cap >= size_bytes && cap <= max_cap && (best.is_none() || cap < best.unwrap().1) {
                best = Some((i, cap));
            }
        }
        if let Some((pos, _)) = best {
            self.hits.set(self.hits.get() + 1);
            Some(pool.swap_remove(pos).1)
        } else {
            None
        }
    }

    /// Return an output buffer to the pool for future reuse.
    fn return_output(&self, size_bytes: u64, buf: wgpu::Buffer) {
        let mut pool = self.output.borrow_mut();
        if pool.len() < self.max_depth {
            pool.push((size_bytes, buf));
        }
        // else: drop the buffer (exceeds pool capacity)
    }

    /// Try to reclaim a storage buffer with at least `size_bytes` capacity.
    fn take_storage(&self, size_bytes: u64) -> Option<wgpu::Buffer> {
        let mut pool = self.storage.borrow_mut();
        let max_cap = size_bytes.saturating_mul(4);
        let mut best: Option<(usize, u64)> = None;
        for (i, &(cap, _)) in pool.iter().enumerate() {
            if cap >= size_bytes && cap <= max_cap && (best.is_none() || cap < best.unwrap().1) {
                best = Some((i, cap));
            }
        }
        if let Some((pos, _)) = best {
            self.hits.set(self.hits.get() + 1);
            Some(pool.swap_remove(pos).1)
        } else {
            None
        }
    }

    /// Return a storage buffer to the pool.
    fn return_storage(&self, size_bytes: u64, buf: wgpu::Buffer) {
        let mut pool = self.storage.borrow_mut();
        if pool.len() < self.max_depth {
            pool.push((size_bytes, buf));
        }
    }

    /// Total cache hits (diagnostic).
    fn cache_hits(&self) -> u64 {
        self.hits.get()
    }
}

/// Cross-platform GPU compute backend via wgpu (Vulkan/Metal/DX12).
struct DeferredDispatch {
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    wg: (u32, u32, u32),
}

/// A recorded GPU operation for compiled execution replay.
pub enum RecordedOp {
    /// Single dispatch in its own compute pass.
    Single {
        pipeline: wgpu::ComputePipeline,
        bind_group: wgpu::BindGroup,
        wg: (u32, u32, u32),
    },
    /// Multiple dispatches in one compute pass (batched, no RAW hazards).
    Batch {
        pipeline: wgpu::ComputePipeline,
        bind_groups: Vec<wgpu::BindGroup>,
        wg_sizes: Vec<(u32, u32, u32)>,
    },
}

pub struct GpuBackend {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipelines: Pipelines,
    adapter_name: String,
    pool: BufferPool,
    /// Deferred command buffers — submitted in batch on flush().
    pending: std::cell::RefCell<Vec<wgpu::CommandBuffer>>,
    /// Shared command encoder — all compute passes go into one encoder.
    encoder: std::cell::RefCell<Option<wgpu::CommandEncoder>>,
    dispatch_count: std::cell::Cell<u32>,
    /// Single-pass mode: accumulate dispatches, replay in one compute pass.
    deferred: std::cell::RefCell<Vec<DeferredDispatch>>,
    single_pass: std::cell::Cell<bool>,
    /// Recording mode: when Some, dispatches are cloned into this vec.
    recording: std::cell::RefCell<Option<Vec<RecordedOp>>>,
}

impl GpuBackend {
    /// Auto-selects the best available GPU adapter.
    pub fn new() -> Result<Self, KernelError> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }))
        .map_err(|_| KernelError::Gpu {
            message: "no GPU adapter found".into(),
        })?;

        let adapter_name = adapter.get_info().name;
        let adapter_features = adapter.features();
        let use_f16 = adapter_features.contains(wgpu::Features::SHADER_F16);
        let mut required_features = wgpu::Features::empty();
        if use_f16 {
            required_features |= wgpu::Features::SHADER_F16;
            eprintln!("  f16 compute: enabled");
        } else {
            eprintln!("  f16 compute: NOT available (using f32 fallback)");
        }

        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("yscv-gpu"),
            required_features,
            required_limits: wgpu::Limits::default(),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: Default::default(),
            experimental_features: Default::default(),
        }))
        .map_err(|e| KernelError::Gpu {
            message: format!("device request failed: {e}"),
        })?;

        let mk = |src: &str| -> wgpu::ShaderModule {
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(src.into()),
            })
        };

        let pipe = |module: &wgpu::ShaderModule| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: None,
                layout: None,
                module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            })
        };

        let pipelines = Pipelines {
            matmul: pipe(&mk(MATMUL_WGSL)),
            elementwise: pipe(&mk(ELEMENTWISE_WGSL)),
            unary: pipe(&mk(UNARY_WGSL)),
            softmax: pipe(&mk(SOFTMAX_WGSL)),
            log_softmax: pipe(&mk(LOG_SOFTMAX_WGSL)),
            logsumexp: pipe(&mk(LOGSUMEXP_WGSL)),
            conv2d: pipe(&mk(CONV2D_WGSL)),
            pool2d: pipe(&mk(POOL2D_WGSL)),
            batch_norm: pipe(&mk(BATCH_NORM_WGSL)),
            layer_norm: pipe(&mk(LAYER_NORM_WGSL)),
            depthwise_conv2d: pipe(&mk(DEPTHWISE_CONV2D_WGSL)),
            transpose_conv2d: pipe(&mk(TRANSPOSE_CONV2D_WGSL)),
            transpose_2d: pipe(&mk(TRANSPOSE_2D_WGSL)),
            gather: pipe(&mk(GATHER_WGSL)),
            attention: pipe(&mk(ATTENTION_WGSL)),
            group_norm: pipe(&mk(GROUP_NORM_WGSL)),
            rms_norm: pipe(&mk(RMS_NORM_WGSL)),
            backward_binary: pipe(&mk(BACKWARD_BINARY_WGSL)),
            reduce_sum_backward: pipe(&mk(REDUCE_SUM_BACKWARD_WGSL)),
            conv2d_input_grad: pipe(&mk(CONV2D_INPUT_GRAD_WGSL)),
            channel_split: pipe(&mk(CHANNEL_SPLIT_WGSL)),
            channel_concat: pipe(&mk(CHANNEL_CONCAT_WGSL)),
            resize_nearest: pipe(&mk(RESIZE_NEAREST_WGSL)),
            permute_nhwc_nchw: pipe(&mk(PERMUTE_NHWC_NCHW_WGSL)),
            permute_nd: pipe(&mk(PERMUTE_ND_WGSL)),
            general_concat: pipe(&mk(GENERAL_CONCAT_WGSL)),
            general_split: pipe(&mk(GENERAL_SPLIT_WGSL)),
            slice_nd: pipe(&mk(SLICE_ND_WGSL)),
            broadcast_binary: pipe(&mk(BROADCAST_BINARY_WGSL)),
            im2col: pipe(&mk(IM2COL_WGSL)),
            bias_add: pipe(&mk(BIAS_ADD_WGSL)),
            batched_matmul: pipe(&mk(BATCHED_MATMUL_WGSL)),
            conv_gemm: pipe(&mk(CONV_GEMM_WGSL)),
            conv_gemm_fast: if use_f16 {
                Some(pipe(&mk(CONV_GEMM_FAST_WGSL)))
            } else {
                None
            },
            conv_gemm_v2: if use_f16 {
                Some(pipe(&mk(CONV_GEMM_V2_WGSL)))
            } else {
                None
            },
            conv_gemm_f16: if use_f16 {
                Some(pipe(&mk(CONV_GEMM_F16_WGSL)))
            } else {
                None
            },
            conv_gemm_fast_n32: if use_f16 {
                Some(pipe(&mk(CONV_GEMM_FAST_N32_WGSL)))
            } else {
                None
            },
            conv_gemm_f16_n32: if use_f16 {
                Some(pipe(&mk(CONV_GEMM_F16_N32_WGSL)))
            } else {
                None
            },
            conv_gemm_f16_io: if use_f16 {
                Some(pipe(&mk(CONV_GEMM_F16_IO_WGSL)))
            } else {
                None
            },
            conv_gemm_f16_io_n32: if use_f16 {
                Some(pipe(&mk(CONV_GEMM_F16_IO_N32_WGSL)))
            } else {
                None
            },
            elementwise_f16: if use_f16 {
                Some(pipe(&mk(ELEMENTWISE_F16_WGSL)))
            } else {
                None
            },
            unary_f16: if use_f16 {
                Some(pipe(&mk(UNARY_F16_WGSL)))
            } else {
                None
            },
            channel_concat_f16: if use_f16 {
                Some(pipe(&mk(CHANNEL_CONCAT_F16_WGSL)))
            } else {
                None
            },
            channel_split_f16: if use_f16 {
                Some(pipe(&mk(CHANNEL_SPLIT_F16_WGSL)))
            } else {
                None
            },
            resize_nearest_f16: if use_f16 {
                Some(pipe(&mk(RESIZE_NEAREST_F16_WGSL)))
            } else {
                None
            },
            permute_nhwc_nchw_f16: if use_f16 {
                Some(pipe(&mk(PERMUTE_NHWC_NCHW_F16_WGSL)))
            } else {
                None
            },
            convert_f32_to_f16: if use_f16 {
                Some(pipe(&mk(CONVERT_F32_TO_F16_WGSL)))
            } else {
                None
            },
            convert_f16_to_f32: if use_f16 {
                Some(pipe(&mk(CONVERT_F16_TO_F32_WGSL)))
            } else {
                None
            },
            winograd_input: pipe(&mk(WINOGRAD_INPUT_WGSL)),
            winograd_output: pipe(&mk(WINOGRAD_OUTPUT_WGSL)),
        };

        Ok(Self {
            device,
            queue,
            pipelines,
            adapter_name,
            pool: BufferPool::new(96),
            pending: std::cell::RefCell::new(Vec::with_capacity(256)),
            encoder: std::cell::RefCell::new(None),
            dispatch_count: std::cell::Cell::new(0),
            deferred: std::cell::RefCell::new(Vec::with_capacity(256)),
            single_pass: std::cell::Cell::new(false),
            recording: std::cell::RefCell::new(None),
        })
    }

    /// Defer a command buffer for batch submission (no immediate GPU sync).
    #[allow(dead_code)]
    fn submit_cmd(&self, cmd: wgpu::CommandBuffer) {
        self.pending.borrow_mut().push(cmd);
    }

    /// Record a compute dispatch. In single-pass mode, defers the dispatch
    /// to be replayed in one compute pass at flush time. Otherwise, creates
    /// a separate compute pass per dispatch (correct Metal barriers).
    fn record_compute(
        &self,
        pipeline: &wgpu::ComputePipeline,
        bg: &wgpu::BindGroup,
        wg: (u32, u32, u32),
    ) {
        self.dispatch_count.set(self.dispatch_count.get() + 1);
        // Capture for compiled plan if recording
        if let Some(ref mut rec) = *self.recording.borrow_mut() {
            rec.push(RecordedOp::Single {
                pipeline: pipeline.clone(),
                bind_group: bg.clone(),
                wg,
            });
        }
        if self.single_pass.get() {
            self.deferred.borrow_mut().push(DeferredDispatch {
                pipeline: pipeline.clone(),
                bind_group: bg.clone(),
                wg,
            });
        } else {
            let mut enc_opt = self.encoder.borrow_mut();
            let enc = enc_opt
                .get_or_insert_with(|| self.device.create_command_encoder(&Default::default()));
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, bg, &[]);
                pass.dispatch_workgroups(wg.0, wg.1, wg.2);
            }
        }
    }

    /// Record multiple compute dispatches in a single compute pass.
    /// SAFETY: Caller must ensure dispatches have no RAW/WAW hazards
    /// (e.g., all writes target non-overlapping buffer regions).
    fn record_compute_batch(
        &self,
        pipeline: &wgpu::ComputePipeline,
        bind_groups: &[wgpu::BindGroup],
        wg_sizes: &[(u32, u32, u32)],
    ) {
        if bind_groups.is_empty() {
            return;
        }
        self.dispatch_count
            .set(self.dispatch_count.get() + bind_groups.len() as u32);
        // Capture for compiled plan if recording
        if let Some(ref mut rec) = *self.recording.borrow_mut() {
            rec.push(RecordedOp::Batch {
                pipeline: pipeline.clone(),
                bind_groups: bind_groups.to_vec(),
                wg_sizes: wg_sizes.to_vec(),
            });
        }
        let mut enc_opt = self.encoder.borrow_mut();
        let enc =
            enc_opt.get_or_insert_with(|| self.device.create_command_encoder(&Default::default()));
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(pipeline);
            for (bg, &wg) in bind_groups.iter().zip(wg_sizes.iter()) {
                pass.set_bind_group(0, bg, &[]);
                pass.dispatch_workgroups(wg.0, wg.1, wg.2);
            }
        }
    }

    /// Flush shared encoder + deferred command buffers to the GPU.
    /// In single-pass mode, replays all deferred dispatches in ONE compute pass.
    pub fn flush(&self) {
        // Flush deferred dispatches (single-pass mode)
        let deferred: Vec<DeferredDispatch> = self.deferred.borrow_mut().drain(..).collect();
        if !deferred.is_empty() {
            let mut enc = self.device.create_command_encoder(&Default::default());
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                for d in &deferred {
                    pass.set_pipeline(&d.pipeline);
                    pass.set_bind_group(0, &d.bind_group, &[]);
                    pass.dispatch_workgroups(d.wg.0, d.wg.1, d.wg.2);
                }
            }
            self.pending.borrow_mut().push(enc.finish());
        }

        // Flush multi-pass encoder
        if let Some(enc) = self.encoder.borrow_mut().take() {
            self.pending.borrow_mut().push(enc.finish());
        }

        let cmds: Vec<_> = self.pending.borrow_mut().drain(..).collect();
        if !cmds.is_empty() {
            self.queue.submit(cmds);
        }
    }

    /// Enable or disable single-pass dispatch mode.
    /// When enabled, all dispatches are batched into one compute pass per flush.
    pub fn set_single_pass(&self, enabled: bool) {
        self.single_pass.set(enabled);
    }

    /// Wait for all submitted GPU work to complete.
    pub fn sync(&self) {
        let _ = self.device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });
    }

    /// Returns the GPU adapter name (e.g. "NVIDIA GeForce RTX 4090").
    pub fn adapter_name(&self) -> &str {
        &self.adapter_name
    }

    /// Returns and resets the total number of GPU compute dispatches since last call.
    pub fn take_dispatch_count(&self) -> u32 {
        self.dispatch_count.replace(0)
    }

    /// Start recording dispatches. Call `take_recording()` to retrieve.
    pub fn start_recording(&self) {
        *self.recording.borrow_mut() = Some(Vec::with_capacity(256));
    }

    /// Stop recording and return captured dispatch sequence.
    pub fn take_recording(&self) -> Vec<RecordedOp> {
        self.recording.borrow_mut().take().unwrap_or_default()
    }

    /// Replay a recorded dispatch sequence. Encodes all dispatches into
    /// command buffers, flushing every `flush_interval` compute passes.
    pub fn replay_recording(&self, ops: &[RecordedOp], flush_interval: usize) {
        let mut pass_count = 0usize;
        for op in ops {
            match op {
                RecordedOp::Single {
                    pipeline,
                    bind_group,
                    wg,
                } => {
                    let mut enc_opt = self.encoder.borrow_mut();
                    let enc = enc_opt.get_or_insert_with(|| {
                        self.device.create_command_encoder(&Default::default())
                    });
                    {
                        let mut pass = enc.begin_compute_pass(&Default::default());
                        pass.set_pipeline(pipeline);
                        pass.set_bind_group(0, bind_group, &[]);
                        pass.dispatch_workgroups(wg.0, wg.1, wg.2);
                    }
                    pass_count += 1;
                }
                RecordedOp::Batch {
                    pipeline,
                    bind_groups,
                    wg_sizes,
                } => {
                    let mut enc_opt = self.encoder.borrow_mut();
                    let enc = enc_opt.get_or_insert_with(|| {
                        self.device.create_command_encoder(&Default::default())
                    });
                    {
                        let mut pass = enc.begin_compute_pass(&Default::default());
                        pass.set_pipeline(pipeline);
                        for (bg, wg) in bind_groups.iter().zip(wg_sizes.iter()) {
                            pass.set_bind_group(0, bg, &[]);
                            pass.dispatch_workgroups(wg.0, wg.1, wg.2);
                        }
                    }
                    pass_count += 1;
                }
            }
            if pass_count.is_multiple_of(flush_interval) {
                self.flush();
            }
        }
    }

    /// Replay a recorded dispatch sequence in a SINGLE compute pass.
    ///
    /// On Metal, non-concurrent dispatches within one `MTLComputeCommandEncoder`
    /// have implicit memory ordering — writes from dispatch A are visible to
    /// dispatch B when A precedes B. This eliminates per-dispatch compute-pass
    /// creation overhead (~0.3-0.5ms × N passes).
    ///
    /// **Only correct on Metal / Apple Silicon.** On Vulkan/DX12, use
    /// `replay_recording` which creates separate passes with proper barriers.
    pub fn replay_recording_fused(&self, ops: &[RecordedOp]) {
        let mut enc = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            for op in ops {
                match op {
                    RecordedOp::Single {
                        pipeline,
                        bind_group,
                        wg,
                    } => {
                        pass.set_pipeline(pipeline);
                        pass.set_bind_group(0, bind_group, &[]);
                        pass.dispatch_workgroups(wg.0, wg.1, wg.2);
                    }
                    RecordedOp::Batch {
                        pipeline,
                        bind_groups,
                        wg_sizes,
                    } => {
                        pass.set_pipeline(pipeline);
                        for (bg, wg) in bind_groups.iter().zip(wg_sizes.iter()) {
                            pass.set_bind_group(0, bg, &[]);
                            pass.dispatch_workgroups(wg.0, wg.1, wg.2);
                        }
                    }
                }
            }
        }
        self.pending.borrow_mut().push(enc.finish());
        let cmds: Vec<_> = self.pending.borrow_mut().drain(..).collect();
        if !cmds.is_empty() {
            self.queue.submit(cmds);
        }
    }

    /// Benchmark per-dispatch GPU overhead by running N trivial element-wise
    /// dispatches in a single compute pass. Returns (single_pass_ms, multi_pass_ms).
    pub fn bench_dispatch_overhead(&self, n: usize) -> (f64, f64) {
        // Create a tiny buffer and a trivial dispatch (add 0 to self)
        let data = vec![1.0f32; 256];
        let buf_a = self.storage_buf(&data);
        let buf_b = self.storage_buf(&data);
        let buf_out = self.output_buf(256);
        let params: [u32; 2] = [64, 0]; // len4=64, op=0 (add)
        let buf_p = self.uniform_buf(bytemuck::cast_slice(&params));

        let bgl = self.pipelines.elementwise.get_bind_group_layout(0);
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf_a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buf_b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buf_out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: buf_p.as_entire_binding(),
                },
            ],
        });

        // Single-pass: N dispatches in ONE compute pass
        let t0 = std::time::Instant::now();
        {
            let mut enc = self.device.create_command_encoder(&Default::default());
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                for _ in 0..n {
                    pass.set_pipeline(&self.pipelines.elementwise);
                    pass.set_bind_group(0, &bg, &[]);
                    pass.dispatch_workgroups(1, 1, 1);
                }
            }
            self.queue.submit(std::iter::once(enc.finish()));
            let _ = self.device.poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            });
        }
        let single_ms = t0.elapsed().as_secs_f64() * 1000.0;

        // Multi-pass: N dispatches each in their OWN compute pass
        let t0 = std::time::Instant::now();
        {
            let mut enc = self.device.create_command_encoder(&Default::default());
            for _ in 0..n {
                let mut pass = enc.begin_compute_pass(&Default::default());
                pass.set_pipeline(&self.pipelines.elementwise);
                pass.set_bind_group(0, &bg, &[]);
                pass.dispatch_workgroups(1, 1, 1);
            }
            self.queue.submit(std::iter::once(enc.finish()));
            let _ = self.device.poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            });
        }
        let multi_ms = t0.elapsed().as_secs_f64() * 1000.0;

        (single_ms, multi_ms)
    }

    /// Write raw f32 data into an existing GPU buffer (for input tensor updates).
    pub fn write_buffer(&self, buffer: &wgpu::Buffer, data: &[f32]) {
        self.queue
            .write_buffer(buffer, 0, bytemuck::cast_slice(data));
    }

    // ── Buffer helpers ─────────────────────────────────────────────

    fn storage_buf(&self, data: &[f32]) -> wgpu::Buffer {
        let size_bytes = (data.len() * 4) as u64;
        if let Some(buf) = self.pool.take_storage(size_bytes) {
            self.queue.write_buffer(&buf, 0, bytemuck::cast_slice(data));
            return buf;
        }
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(data),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            })
    }

    fn output_buf(&self, len: usize) -> wgpu::Buffer {
        let size_bytes = (len * 4) as u64;
        if let Some(buf) = self.pool.take_output(size_bytes) {
            return buf;
        }
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: size_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    /// Allocate an f16 output buffer (2 bytes per element).
    fn output_buf_f16(&self, len: usize) -> wgpu::Buffer {
        let size_bytes = (len * 2) as u64;
        if let Some(buf) = self.pool.take_output(size_bytes) {
            return buf;
        }
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: size_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    /// Create an f16 storage buffer from f32 data (converts CPU-side).
    pub fn storage_buf_f16(&self, data: &[f32]) -> wgpu::Buffer {
        let f16_data: Vec<u16> = data.iter().map(|&v| f32_to_f16_bits(v)).collect();
        let bytes: &[u8] = bytemuck::cast_slice(&f16_data);
        let size_bytes = bytes.len() as u64;
        if let Some(buf) = self.pool.take_storage(size_bytes) {
            self.queue.write_buffer(&buf, 0, bytes);
            return buf;
        }
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            })
    }

    /// Write f32 data into an f16 GPU buffer (converts CPU-side).
    pub fn write_buffer_f16(&self, buffer: &wgpu::Buffer, data: &[f32]) {
        let f16_data: Vec<u16> = data.iter().map(|&v| f32_to_f16_bits(v)).collect();
        self.queue
            .write_buffer(buffer, 0, bytemuck::cast_slice(&f16_data));
    }

    /// Read an f16 GPU buffer back as f32 Vec.
    pub fn read_buf_f16(&self, buffer: &wgpu::Buffer, len: usize) -> Result<Vec<f32>, KernelError> {
        let size = (len * 2) as u64;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        if let Some(enc) = self.encoder.borrow_mut().take() {
            self.pending.borrow_mut().push(enc.finish());
        }

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, size);
        let mut pending = self.pending.borrow_mut();
        pending.push(encoder.finish());
        self.queue.submit(pending.drain(..));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        let _ = self.device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });
        rx.recv()
            .map_err(|_| KernelError::Gpu {
                message: "GPU channel closed during f16 read-back".into(),
            })?
            .map_err(|e| KernelError::Gpu {
                message: format!("GPU map failed during f16 read-back: {e}"),
            })?;

        let mapped = slice.get_mapped_range();
        let u16_data: &[u16] = bytemuck::cast_slice(&mapped);
        let result: Vec<f32> = u16_data.iter().map(|&bits| f16_bits_to_f32(bits)).collect();
        drop(mapped);
        staging.unmap();
        Ok(result)
    }

    /// Returns a buffer to the output pool (f16: 2 bytes per element).
    pub fn return_output_buf_f16(&self, len: usize, buf: wgpu::Buffer) {
        self.pool.return_output((len * 2) as u64, buf);
    }

    /// Returns whether f16 I/O pipelines are available.
    pub fn has_f16_io(&self) -> bool {
        self.pipelines.conv_gemm_f16_io.is_some()
    }

    /// Returns the number of GPU buffer pool cache hits (diagnostic).
    pub fn pool_cache_hits(&self) -> u64 {
        self.pool.cache_hits()
    }

    /// Returns a buffer to the output pool for future reuse.
    pub fn return_output_buf(&self, len: usize, buf: wgpu::Buffer) {
        self.pool.return_output((len * 4) as u64, buf);
    }

    /// Returns a buffer to the storage pool for future reuse.
    pub fn return_storage_buf(&self, len: usize, buf: wgpu::Buffer) {
        self.pool.return_storage((len * 4) as u64, buf);
    }

    fn uniform_buf(&self, data: &[u8]) -> wgpu::Buffer {
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: data,
                usage: wgpu::BufferUsages::UNIFORM,
            })
    }

    // ── Device-resident tensor operations ─────────────────────────

    /// Upload a CPU tensor to GPU, returning a device-resident handle.
    /// The buffer supports both compute input (STORAGE) and readback (COPY_SRC).
    pub fn upload(&self, tensor: &Tensor) -> GpuBuffer {
        let data = tensor.data();
        let buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(data),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            });
        GpuBuffer {
            buffer,
            size: data.len(),
            shape: tensor.shape().to_vec(),
        }
    }

    /// Download a GPU buffer back to a CPU tensor.
    pub fn download(&self, buf: &GpuBuffer) -> Result<Tensor, KernelError> {
        let data = self.read_buf(&buf.buffer, buf.size)?;
        Tensor::from_vec(buf.shape.clone(), data).map_err(Into::into)
    }

    /// Run a shader with GPU-resident inputs, producing a GPU-resident output.
    /// No host<->device copy occurs.
    pub fn dispatch_on_device(
        &self,
        input_bufs: &[&GpuBuffer],
        output_size: usize,
        output_shape: Vec<usize>,
        pipeline: &wgpu::ComputePipeline,
        uniform_data: &[u8],
        workgroups: (u32, u32, u32),
    ) -> GpuBuffer {
        let buf_out = self.output_buf(output_size);
        let buf_p = self.uniform_buf(uniform_data);

        let bgl = pipeline.get_bind_group_layout(0);
        let mut entries: Vec<wgpu::BindGroupEntry> = Vec::new();
        for (i, gb) in input_bufs.iter().enumerate() {
            entries.push(wgpu::BindGroupEntry {
                binding: i as u32,
                resource: gb.buffer.as_entire_binding(),
            });
        }
        entries.push(wgpu::BindGroupEntry {
            binding: input_bufs.len() as u32,
            resource: buf_out.as_entire_binding(),
        });
        entries.push(wgpu::BindGroupEntry {
            binding: (input_bufs.len() + 1) as u32,
            resource: buf_p.as_entire_binding(),
        });

        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &entries,
        });

        self.record_compute(pipeline, &bg, workgroups);

        GpuBuffer {
            buffer: buf_out,
            size: output_size,
            shape: output_shape,
        }
    }

    // ── Internal device-resident dispatch helpers ─────────────────

    /// Elementwise op entirely on device buffers, returning a device buffer.
    /// Binary elementwise op on device buffers.
    /// Uses vec4 shader: 4 elements per thread for bandwidth efficiency.
    fn gpu_elementwise_on_device(
        &self,
        a: &wgpu::Buffer,
        b: &wgpu::Buffer,
        len: usize,
        op: u32,
    ) -> wgpu::Buffer {
        let len4 = len.div_ceil(4);
        let buf_out = self.output_buf(len4 * 4);
        let params: [u32; 2] = [len4 as u32, op];
        let buf_p = self.uniform_buf(bytemuck::cast_slice(&params));

        let bgl = self.pipelines.elementwise.get_bind_group_layout(0);
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buf_out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: buf_p.as_entire_binding(),
                },
            ],
        });

        self.record_compute(
            &self.pipelines.elementwise,
            &bg,
            (div_ceil(len4 as u32, 256), 1, 1),
        );
        buf_out
    }

    /// Unary op entirely on device buffers, returning a device buffer.
    /// Uses vec4 shader: 4 elements per thread for bandwidth efficiency.
    fn gpu_unary_on_device(&self, a: &wgpu::Buffer, len: usize, op: u32) -> wgpu::Buffer {
        // Round up to vec4 boundary for the output buffer
        let len4 = len.div_ceil(4);
        let buf_out = self.output_buf(len4 * 4);
        let params: [u32; 2] = [len4 as u32, op];
        let buf_p = self.uniform_buf(bytemuck::cast_slice(&params));

        let bgl = self.pipelines.unary.get_bind_group_layout(0);
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buf_out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buf_p.as_entire_binding(),
                },
            ],
        });

        self.record_compute(
            &self.pipelines.unary,
            &bg,
            (div_ceil(len4 as u32, 256), 1, 1),
        );
        buf_out
    }

    // ── f16 I/O dispatch helpers ──────────────────────────────────────

    /// Binary elementwise op on f16 device buffers.
    fn gpu_elementwise_f16_on_device(
        &self,
        a: &wgpu::Buffer,
        b: &wgpu::Buffer,
        len: usize,
        op: u32,
    ) -> Result<wgpu::Buffer, KernelError> {
        let pipeline = self
            .pipelines
            .elementwise_f16
            .as_ref()
            .ok_or_else(|| KernelError::Gpu {
                message: "f16 elementwise pipeline not available on this device".into(),
            })?;
        let len4 = len.div_ceil(4);
        let buf_out = self.output_buf_f16(len4 * 4);
        let params: [u32; 2] = [len4 as u32, op];
        let buf_p = self.uniform_buf(bytemuck::cast_slice(&params));

        let bgl = pipeline.get_bind_group_layout(0);
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buf_out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: buf_p.as_entire_binding(),
                },
            ],
        });

        self.record_compute(pipeline, &bg, (div_ceil(len4 as u32, 256), 1, 1));
        Ok(buf_out)
    }

    /// Unary op on f16 device buffers.
    fn gpu_unary_f16_on_device(
        &self,
        a: &wgpu::Buffer,
        len: usize,
        op: u32,
    ) -> Result<wgpu::Buffer, KernelError> {
        let pipeline = self
            .pipelines
            .unary_f16
            .as_ref()
            .ok_or_else(|| KernelError::Gpu {
                message: "f16 unary pipeline not available on this device".into(),
            })?;
        let len4 = len.div_ceil(4);
        let buf_out = self.output_buf_f16(len4 * 4);
        let params: [u32; 2] = [len4 as u32, op];
        let buf_p = self.uniform_buf(bytemuck::cast_slice(&params));

        let bgl = pipeline.get_bind_group_layout(0);
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buf_out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buf_p.as_entire_binding(),
                },
            ],
        });

        self.record_compute(pipeline, &bg, (div_ceil(len4 as u32, 256), 1, 1));
        Ok(buf_out)
    }

    /// Convert an f32 buffer to f16 on device.
    pub fn convert_f32_to_f16_on_device(
        &self,
        input: &GpuBuffer,
    ) -> Result<GpuBuffer, KernelError> {
        let pipeline =
            self.pipelines
                .convert_f32_to_f16
                .as_ref()
                .ok_or_else(|| KernelError::Gpu {
                    message: "f16 conversion pipeline not available on this device".into(),
                })?;
        let len = input.size;
        let buf_out = self.output_buf_f16(len);
        let params: [u32; 2] = [len as u32, 0];
        let buf_p = self.uniform_buf(bytemuck::cast_slice(&params));

        let bgl = pipeline.get_bind_group_layout(0);
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buf_out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buf_p.as_entire_binding(),
                },
            ],
        });

        self.record_compute(pipeline, &bg, (div_ceil(len as u32, 256), 1, 1));
        Ok(GpuBuffer {
            buffer: buf_out,
            size: len,
            shape: input.shape.clone(),
        })
    }

    /// Convert an f16 storage buffer to f32 on device.
    pub fn convert_f16_to_f32_on_device(
        &self,
        input: &GpuBuffer,
    ) -> Result<GpuBuffer, KernelError> {
        let pipeline =
            self.pipelines
                .convert_f16_to_f32
                .as_ref()
                .ok_or_else(|| KernelError::Gpu {
                    message: "f16 conversion pipeline not available on this device".into(),
                })?;
        let len = input.size;
        let buf_out = self.output_buf(len); // f32 output = 4 bytes/elem
        let params: [u32; 2] = [len as u32, 0];
        let buf_p = self.uniform_buf(bytemuck::cast_slice(&params));

        let bgl = pipeline.get_bind_group_layout(0);
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buf_out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buf_p.as_entire_binding(),
                },
            ],
        });

        self.record_compute(pipeline, &bg, (div_ceil(len as u32, 256), 1, 1));
        Ok(GpuBuffer {
            buffer: buf_out,
            size: len,
            shape: input.shape.clone(),
        })
    }

    fn read_buf(&self, buffer: &wgpu::Buffer, len: usize) -> Result<Vec<f32>, KernelError> {
        let size = (len * 4) as u64;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Flush shared encoder first so all prior compute passes are submitted
        if let Some(enc) = self.encoder.borrow_mut().take() {
            self.pending.borrow_mut().push(enc.finish());
        }

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, size);
        // Flush all pending compute commands, then submit the copy
        let mut pending = self.pending.borrow_mut();
        pending.push(encoder.finish());
        self.queue.submit(pending.drain(..));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        let _ = self.device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });
        rx.recv()
            .map_err(|_| KernelError::Gpu {
                message: "GPU channel closed during read-back".into(),
            })?
            .map_err(|e| KernelError::Gpu {
                message: format!("GPU map failed during read-back: {e}"),
            })?;

        let mapped = slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&mapped).to_vec();
        drop(mapped);
        staging.unmap();
        Ok(result)
    }

    // ── Dispatch helpers ───────────────────────────────────────────

    fn dispatch_elementwise(
        &self,
        a: &[f32],
        b: &[f32],
        len: usize,
        op: u32,
    ) -> Result<Vec<f32>, KernelError> {
        let len4 = len.div_ceil(4);
        let mut pa = a.to_vec();
        pa.resize(len4 * 4, 0.0);
        let mut pb = b.to_vec();
        pb.resize(len4 * 4, 0.0);
        let buf_a = self.storage_buf(&pa);
        let buf_b = self.storage_buf(&pb);
        let buf_out = self.output_buf(len4 * 4);
        let params: [u32; 2] = [len4 as u32, op];
        let buf_p = self.uniform_buf(bytemuck::cast_slice(&params));

        let bgl = self.pipelines.elementwise.get_bind_group_layout(0);
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf_a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buf_b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buf_out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: buf_p.as_entire_binding(),
                },
            ],
        });

        self.record_compute(
            &self.pipelines.elementwise,
            &bg,
            (div_ceil(len4 as u32, 256), 1, 1),
        );
        let mut result = self.read_buf(&buf_out, len4 * 4)?;
        result.truncate(len);
        Ok(result)
    }

    fn dispatch_backward_binary(
        &self,
        upstream: &[f32],
        forward_val: &[f32],
        len: usize,
        op: u32,
    ) -> Result<Vec<f32>, KernelError> {
        let buf_up = self.storage_buf(upstream);
        let buf_fv = self.storage_buf(forward_val);
        let buf_out = self.output_buf(len);
        let params: [u32; 2] = [len as u32, op];
        let buf_p = self.uniform_buf(bytemuck::cast_slice(&params));

        let bgl = self.pipelines.backward_binary.get_bind_group_layout(0);
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf_up.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buf_fv.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buf_out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: buf_p.as_entire_binding(),
                },
            ],
        });

        self.record_compute(
            &self.pipelines.backward_binary,
            &bg,
            (div_ceil(len as u32, 256), 1, 1),
        );
        self.read_buf(&buf_out, len)
    }

    fn dispatch_reduce_sum_backward(
        &self,
        upstream: &[f32],
        output_len: usize,
    ) -> Result<Vec<f32>, KernelError> {
        let buf_up = self.storage_buf(upstream);
        let buf_out = self.output_buf(output_len);
        let params: [u32; 2] = [output_len as u32, 0];
        let buf_p = self.uniform_buf(bytemuck::cast_slice(&params));

        let bgl = self.pipelines.reduce_sum_backward.get_bind_group_layout(0);
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf_up.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buf_out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buf_p.as_entire_binding(),
                },
            ],
        });

        self.record_compute(
            &self.pipelines.reduce_sum_backward,
            &bg,
            (div_ceil(output_len as u32, 256), 1, 1),
        );
        self.read_buf(&buf_out, output_len)
    }

    fn dispatch_unary(&self, a: &[f32], len: usize, op: u32) -> Result<Vec<f32>, KernelError> {
        // Pad input to vec4 boundary for the vec4 shader
        let len4 = len.div_ceil(4);
        let mut padded = a.to_vec();
        padded.resize(len4 * 4, 0.0);
        let buf_a = self.storage_buf(&padded);
        let buf_out = self.output_buf(len4 * 4);
        let params: [u32; 2] = [len4 as u32, op];
        let buf_p = self.uniform_buf(bytemuck::cast_slice(&params));

        let bgl = self.pipelines.unary.get_bind_group_layout(0);
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf_a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buf_out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buf_p.as_entire_binding(),
                },
            ],
        });

        self.record_compute(
            &self.pipelines.unary,
            &bg,
            (div_ceil(len4 as u32, 256), 1, 1),
        );
        let mut result = self.read_buf(&buf_out, len4 * 4)?;
        result.truncate(len);
        Ok(result)
    }

    // ── Public on-device operations (no read-back) ────────────────
    //
    // These methods keep data GPU-resident between operations, avoiding
    // the per-op device.poll(Wait) sync that dominates latency in graph
    // inference.  Use `download()` only on the final output.

    /// ReLU on device (op=0).
    pub fn relu_on_device(&self, input: &GpuBuffer) -> GpuBuffer {
        let out = self.gpu_unary_on_device(&input.buffer, input.size, 0);
        GpuBuffer {
            buffer: out,
            size: input.size,
            shape: input.shape.clone(),
        }
    }

    /// Sigmoid on device (op=1).
    pub fn sigmoid_on_device(&self, input: &GpuBuffer) -> GpuBuffer {
        let out = self.gpu_unary_on_device(&input.buffer, input.size, 1);
        GpuBuffer {
            buffer: out,
            size: input.size,
            shape: input.shape.clone(),
        }
    }

    /// Exp on device (op=2).
    pub fn exp_on_device(&self, input: &GpuBuffer) -> GpuBuffer {
        let out = self.gpu_unary_on_device(&input.buffer, input.size, 2);
        GpuBuffer {
            buffer: out,
            size: input.size,
            shape: input.shape.clone(),
        }
    }

    /// Tanh on device (op=3).
    pub fn tanh_on_device(&self, input: &GpuBuffer) -> GpuBuffer {
        let out = self.gpu_unary_on_device(&input.buffer, input.size, 3);
        GpuBuffer {
            buffer: out,
            size: input.size,
            shape: input.shape.clone(),
        }
    }

    /// SiLU on device (op=4): x * sigmoid(x).
    pub fn silu_on_device(&self, input: &GpuBuffer) -> GpuBuffer {
        let out = self.gpu_unary_on_device(&input.buffer, input.size, 4);
        GpuBuffer {
            buffer: out,
            size: input.size,
            shape: input.shape.clone(),
        }
    }

    /// Add on device (op=0).
    pub fn add_on_device(&self, a: &GpuBuffer, b: &GpuBuffer) -> GpuBuffer {
        let len = a.size;
        let out = self.gpu_elementwise_on_device(&a.buffer, &b.buffer, len, 0);
        GpuBuffer {
            buffer: out,
            size: len,
            shape: a.shape.clone(),
        }
    }

    /// Sub on device (op=1).
    pub fn sub_on_device(&self, a: &GpuBuffer, b: &GpuBuffer) -> GpuBuffer {
        let len = a.size;
        let out = self.gpu_elementwise_on_device(&a.buffer, &b.buffer, len, 1);
        GpuBuffer {
            buffer: out,
            size: len,
            shape: a.shape.clone(),
        }
    }

    /// Mul on device (op=2).
    pub fn mul_on_device(&self, a: &GpuBuffer, b: &GpuBuffer) -> GpuBuffer {
        let len = a.size;
        let out = self.gpu_elementwise_on_device(&a.buffer, &b.buffer, len, 2);
        GpuBuffer {
            buffer: out,
            size: len,
            shape: a.shape.clone(),
        }
    }

    /// Div on device (op=3).
    pub fn div_on_device(&self, a: &GpuBuffer, b: &GpuBuffer) -> GpuBuffer {
        let len = a.size;
        let out = self.gpu_elementwise_on_device(&a.buffer, &b.buffer, len, 3);
        GpuBuffer {
            buffer: out,
            size: len,
            shape: a.shape.clone(),
        }
    }

    // ── f16 I/O public dispatch methods ───────────────────────────────

    /// ReLU on f16 device buffer (op=0).
    pub fn relu_f16_on_device(&self, input: &GpuBuffer) -> Result<GpuBuffer, KernelError> {
        let out = self.gpu_unary_f16_on_device(&input.buffer, input.size, 0)?;
        Ok(GpuBuffer {
            buffer: out,
            size: input.size,
            shape: input.shape.clone(),
        })
    }

    /// Sigmoid on f16 device buffer (op=1).
    pub fn sigmoid_f16_on_device(&self, input: &GpuBuffer) -> Result<GpuBuffer, KernelError> {
        let out = self.gpu_unary_f16_on_device(&input.buffer, input.size, 1)?;
        Ok(GpuBuffer {
            buffer: out,
            size: input.size,
            shape: input.shape.clone(),
        })
    }

    /// SiLU on f16 device buffer (op=4).
    pub fn silu_f16_on_device(&self, input: &GpuBuffer) -> Result<GpuBuffer, KernelError> {
        let out = self.gpu_unary_f16_on_device(&input.buffer, input.size, 4)?;
        Ok(GpuBuffer {
            buffer: out,
            size: input.size,
            shape: input.shape.clone(),
        })
    }

    /// Add on f16 device buffers (op=0).
    pub fn add_f16_on_device(
        &self,
        a: &GpuBuffer,
        b: &GpuBuffer,
    ) -> Result<GpuBuffer, KernelError> {
        let len = a.size;
        let out = self.gpu_elementwise_f16_on_device(&a.buffer, &b.buffer, len, 0)?;
        Ok(GpuBuffer {
            buffer: out,
            size: len,
            shape: a.shape.clone(),
        })
    }

    /// Mul on f16 device buffers (op=2).
    pub fn mul_f16_on_device(
        &self,
        a: &GpuBuffer,
        b: &GpuBuffer,
    ) -> Result<GpuBuffer, KernelError> {
        let len = a.size;
        let out = self.gpu_elementwise_f16_on_device(&a.buffer, &b.buffer, len, 2)?;
        Ok(GpuBuffer {
            buffer: out,
            size: len,
            shape: a.shape.clone(),
        })
    }

    /// Conv2D on f16 I/O buffers: input [NHWC] and weight are f16, bias is f32, output is f16.
    pub fn im2col_conv_f16_on_device(
        &self,
        input: &GpuBuffer,
        kernel: &GpuBuffer,
        bias: &GpuBuffer,
        stride_h: usize,
        stride_w: usize,
        pads: [usize; 4],
        act: u32,
    ) -> Result<GpuBuffer, KernelError> {
        let (n, ih, iw, ic) = (
            input.shape[0],
            input.shape[1],
            input.shape[2],
            input.shape[3],
        );
        let (kh, kw, _kc, oc) = (
            kernel.shape[0],
            kernel.shape[1],
            kernel.shape[2],
            kernel.shape[3],
        );
        let oh = (ih + pads[0] + pads[2] - kh) / stride_h + 1;
        let ow = (iw + pads[1] + pads[3] - kw) / stride_w + 1;
        let m = n * oh * ow;
        let col_k = kh * kw * ic;
        let out_total = m * oc;
        let buf_out = self.output_buf_f16(out_total);

        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct ConvGemmP {
            m: u32,
            n_out: u32,
            k: u32,
            act: u32,
            ih: u32,
            iw: u32,
            ic: u32,
            oh: u32,
            ow: u32,
            kh: u32,
            kw: u32,
            sh: u32,
            sw: u32,
            pad_h: u32,
            pad_w: u32,
            batch: u32,
        }
        let p = ConvGemmP {
            m: m as u32,
            n_out: oc as u32,
            k: col_k as u32,
            act,
            ih: ih as u32,
            iw: iw as u32,
            ic: ic as u32,
            oh: oh as u32,
            ow: ow as u32,
            kh: kh as u32,
            kw: kw as u32,
            sh: stride_h as u32,
            sw: stride_w as u32,
            pad_h: pads[0] as u32,
            pad_w: pads[1] as u32,
            batch: n as u32,
        };
        let buf_p = self.uniform_buf(bytemuck::bytes_of(&p));

        let use_n32 = oc <= 48;
        let (pipeline, bn) = if use_n32 {
            (
                self.pipelines
                    .conv_gemm_f16_io_n32
                    .as_ref()
                    .ok_or_else(|| KernelError::Gpu {
                        message: "f16 I/O conv pipeline (n32) not available on this device".into(),
                    })?,
                32u32,
            )
        } else {
            (
                self.pipelines
                    .conv_gemm_f16_io
                    .as_ref()
                    .ok_or_else(|| KernelError::Gpu {
                        message: "f16 I/O conv pipeline not available on this device".into(),
                    })?,
                64u32,
            )
        };
        let bgl = pipeline.get_bind_group_layout(0);
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: kernel.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: bias.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: buf_out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: buf_p.as_entire_binding(),
                },
            ],
        });
        let wg_x = div_ceil(oc as u32, bn);
        let wg_y = div_ceil(m as u32, 64);
        self.record_compute(pipeline, &bg, (wg_x, wg_y, 1));

        Ok(GpuBuffer {
            buffer: buf_out,
            size: out_total,
            shape: vec![n, oh, ow, oc],
        })
    }

    /// Channel concat on f16 device buffers.
    pub fn channel_concat_f16_on_device(
        &self,
        inputs: &[&GpuBuffer],
    ) -> Result<GpuBuffer, KernelError> {
        let pipeline =
            self.pipelines
                .channel_concat_f16
                .as_ref()
                .ok_or_else(|| KernelError::Gpu {
                    message: "f16 channel concat pipeline not available on this device".into(),
                })?;
        if inputs.is_empty() {
            return Err(KernelError::Gpu {
                message: "empty inputs".into(),
            });
        }
        let c_out: usize = {
            let mut sum = 0usize;
            for b in inputs {
                sum += *b.shape.last().ok_or_else(|| KernelError::Gpu {
                    message: "empty shape".into(),
                })?;
            }
            sum
        };
        let spatial: usize = inputs[0].size
            / *inputs[0].shape.last().ok_or_else(|| KernelError::Gpu {
                message: "empty shape".into(),
            })?;
        let total = spatial * c_out;
        let buf_out = self.output_buf_f16(total);

        let bgl = pipeline.get_bind_group_layout(0);
        let mut bind_groups = Vec::with_capacity(inputs.len());
        let mut wg_sizes = Vec::with_capacity(inputs.len());

        let mut ch_offset = 0usize;
        for inp in inputs {
            let c_in = *inp.shape.last().ok_or_else(|| KernelError::Gpu {
                message: "empty shape".into(),
            })?;
            #[repr(C)]
            #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
            struct P {
                spatial: u32,
                c_in: u32,
                c_out: u32,
                ch_offset: u32,
            }
            let p = P {
                spatial: spatial as u32,
                c_in: c_in as u32,
                c_out: c_out as u32,
                ch_offset: ch_offset as u32,
            };
            let buf_p = self.uniform_buf(bytemuck::bytes_of(&p));

            let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: inp.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: buf_out.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: buf_p.as_entire_binding(),
                    },
                ],
            });
            bind_groups.push(bg);
            wg_sizes.push((div_ceil((spatial * c_in) as u32, 256), 1u32, 1u32));
            ch_offset += c_in;
        }

        self.record_compute_batch(pipeline, &bind_groups, &wg_sizes);

        let mut out_shape = inputs[0].shape.clone();
        *out_shape.last_mut().ok_or_else(|| KernelError::Gpu {
            message: "empty shape".into(),
        })? = c_out;
        Ok(GpuBuffer {
            buffer: buf_out,
            size: total,
            shape: out_shape,
        })
    }

    /// Channel split on f16 device buffers.
    pub fn channel_split_f16_on_device(
        &self,
        input: &GpuBuffer,
        split_sizes: &[usize],
    ) -> Result<Vec<GpuBuffer>, KernelError> {
        let pipeline =
            self.pipelines
                .channel_split_f16
                .as_ref()
                .ok_or_else(|| KernelError::Gpu {
                    message: "f16 channel split pipeline not available on this device".into(),
                })?;
        let c_in = *input.shape.last().ok_or_else(|| KernelError::Gpu {
            message: "empty shape".into(),
        })?;
        let spatial: usize = input.size / c_in;

        let bgl = pipeline.get_bind_group_layout(0);
        let mut bind_groups = Vec::with_capacity(split_sizes.len());
        let mut wg_sizes = Vec::with_capacity(split_sizes.len());
        let mut results = Vec::with_capacity(split_sizes.len());

        let mut ch_offset = 0usize;
        for &c_out in split_sizes {
            let total = spatial * c_out;
            let buf_out = self.output_buf_f16(total);

            #[repr(C)]
            #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
            struct P {
                spatial: u32,
                c_in: u32,
                c_out: u32,
                ch_offset: u32,
            }
            let p = P {
                spatial: spatial as u32,
                c_in: c_in as u32,
                c_out: c_out as u32,
                ch_offset: ch_offset as u32,
            };
            let buf_p = self.uniform_buf(bytemuck::bytes_of(&p));

            let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: input.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: buf_out.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: buf_p.as_entire_binding(),
                    },
                ],
            });
            bind_groups.push(bg);
            wg_sizes.push((div_ceil(total as u32, 256), 1u32, 1u32));

            let mut out_shape = input.shape.clone();
            *out_shape.last_mut().ok_or_else(|| KernelError::Gpu {
                message: "empty shape".into(),
            })? = c_out;
            results.push(GpuBuffer {
                buffer: buf_out,
                size: total,
                shape: out_shape,
            });
            ch_offset += c_out;
        }

        self.record_compute_batch(pipeline, &bind_groups, &wg_sizes);
        Ok(results)
    }

    /// Nearest-neighbor resize on f16 NHWC tensor.
    pub fn resize_nearest_f16_on_device(
        &self,
        input: &GpuBuffer,
        oh: usize,
        ow: usize,
    ) -> Result<GpuBuffer, KernelError> {
        let pipeline =
            self.pipelines
                .resize_nearest_f16
                .as_ref()
                .ok_or_else(|| KernelError::Gpu {
                    message: "f16 resize nearest pipeline not available on this device".into(),
                })?;
        let (n, _ih, _iw, c) = (
            input.shape[0],
            input.shape[1],
            input.shape[2],
            input.shape[3],
        );
        let total = n * oh * ow * c;
        let buf_out = self.output_buf_f16(total);

        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct P {
            n: u32,
            ih: u32,
            iw: u32,
            c: u32,
            oh: u32,
            ow: u32,
            _p1: u32,
            _p2: u32,
        }
        let p = P {
            n: n as u32,
            ih: input.shape[1] as u32,
            iw: input.shape[2] as u32,
            c: c as u32,
            oh: oh as u32,
            ow: ow as u32,
            _p1: 0,
            _p2: 0,
        };
        let buf_p = self.uniform_buf(bytemuck::bytes_of(&p));

        let bgl = pipeline.get_bind_group_layout(0);
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buf_out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buf_p.as_entire_binding(),
                },
            ],
        });
        self.record_compute(pipeline, &bg, (div_ceil(total as u32, 256), 1, 1));
        Ok(GpuBuffer {
            buffer: buf_out,
            size: total,
            shape: vec![n, oh, ow, c],
        })
    }

    /// Permute NHWC → NCHW with f16 input, f32 output (for final readback).
    pub fn nhwc_to_nchw_f16_to_f32_on_device(
        &self,
        input: &GpuBuffer,
    ) -> Result<GpuBuffer, KernelError> {
        let pipeline = self
            .pipelines
            .permute_nhwc_nchw_f16
            .as_ref()
            .ok_or_else(|| KernelError::Gpu {
                message: "f16 NHWC-to-NCHW permute pipeline not available on this device".into(),
            })?;
        let (n, h, w, c) = (
            input.shape[0],
            input.shape[1],
            input.shape[2],
            input.shape[3],
        );
        let total = n * c * h * w;
        let buf_out = self.output_buf(total); // f32 output!

        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct P {
            n: u32,
            h: u32,
            w: u32,
            c: u32,
        }
        let p = P {
            n: n as u32,
            h: h as u32,
            w: w as u32,
            c: c as u32,
        };
        let buf_p = self.uniform_buf(bytemuck::bytes_of(&p));

        let bgl = pipeline.get_bind_group_layout(0);
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buf_out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buf_p.as_entire_binding(),
                },
            ],
        });
        self.record_compute(pipeline, &bg, (div_ceil(total as u32, 256), 1, 1));
        Ok(GpuBuffer {
            buffer: buf_out,
            size: total,
            shape: vec![n, c, h, w],
        })
    }

    /// Broadcasting binary op on device (handles ONNX broadcast rules, up to 6D).
    /// `op`: 0=add, 1=sub, 2=mul, 3=div.
    pub fn broadcast_binary_on_device(&self, a: &GpuBuffer, b: &GpuBuffer, op: u32) -> GpuBuffer {
        let ndim = a.shape.len().max(b.shape.len());
        assert!(ndim <= 6);

        // Left-pad shapes to ndim
        let pad = |s: &[usize]| -> Vec<usize> {
            let mut v = vec![1; ndim - s.len()];
            v.extend_from_slice(s);
            v
        };
        let sa = pad(&a.shape);
        let sb = pad(&b.shape);

        // Compute output shape and strides
        let mut out_shape = vec![0usize; ndim];
        for i in 0..ndim {
            out_shape[i] = sa[i].max(sb[i]);
        }
        let total: usize = out_shape.iter().product();

        // Compute input strides (0 for broadcast dims)
        let make_strides = |s: &[usize]| -> Vec<u32> {
            let mut strides = vec![1usize; ndim];
            for i in (0..ndim - 1).rev() {
                strides[i] = strides[i + 1] * s[i + 1];
            }
            // Zero out broadcast dimensions
            for i in 0..ndim {
                if s[i] == 1 && out_shape[i] > 1 {
                    strides[i] = 0;
                }
            }
            strides.iter().map(|&x| x as u32).collect()
        };
        let ast = make_strides(&sa);
        let bst = make_strides(&sb);

        // Pack params
        let mut s = [1u32; 8];
        let mut a_s = [0u32; 8];
        let mut b_s = [0u32; 8];
        for i in 0..ndim {
            s[i] = out_shape[i] as u32;
            a_s[i] = ast[i];
            b_s[i] = bst[i];
        }

        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct P {
            total: u32,
            op: u32,
            _p1: u32,
            _p2: u32,
            s: [u32; 8],
            a: [u32; 8],
            b: [u32; 8],
        }
        let p = P {
            total: total as u32,
            op,
            _p1: 0,
            _p2: 0,
            s,
            a: a_s,
            b: b_s,
        };
        let buf_p = self.uniform_buf(bytemuck::bytes_of(&p));
        let buf_out = self.output_buf(total);

        let bgl = self.pipelines.broadcast_binary.get_bind_group_layout(0);
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: a.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: b.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buf_out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: buf_p.as_entire_binding(),
                },
            ],
        });
        self.record_compute(
            &self.pipelines.broadcast_binary,
            &bg,
            (div_ceil(total as u32, 256), 1, 1),
        );
        GpuBuffer {
            buffer: buf_out,
            size: total,
            shape: out_shape,
        }
    }

    /// MatMul on device: (M×K) @ (K×N) → (M×N).
    pub fn matmul_2d_on_device(&self, a: &GpuBuffer, b: &GpuBuffer) -> GpuBuffer {
        let (m, k) = (a.shape[0], a.shape[1]);
        let n = b.shape[1];
        let out_size = m * n;
        let buf_out = self.output_buf(out_size);
        let params: [u32; 4] = [m as u32, n as u32, k as u32, 0];
        let buf_p = self.uniform_buf(bytemuck::cast_slice(&params));

        let bgl = self.pipelines.matmul.get_bind_group_layout(0);
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: a.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: b.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buf_out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: buf_p.as_entire_binding(),
                },
            ],
        });
        self.record_compute(
            &self.pipelines.matmul,
            &bg,
            (div_ceil(n as u32, 64), div_ceil(m as u32, 64), 1),
        );
        GpuBuffer {
            buffer: buf_out,
            size: out_size,
            shape: vec![m, n],
        }
    }

    /// Conv2D NHWC on device with optional padding.
    /// Input [N,H,W,Cin], kernel [KH,KW,Cin,Cout], bias [Cout].
    /// `pads`: [pad_top, pad_left, pad_bottom, pad_right]. Use [0;4] for no padding.
    pub fn conv2d_nhwc_on_device(
        &self,
        input: &GpuBuffer,
        kernel: &GpuBuffer,
        bias: &GpuBuffer,
        stride_h: usize,
        stride_w: usize,
        pads: [usize; 4],
    ) -> GpuBuffer {
        let (n, ih, iw, _ic) = (
            input.shape[0],
            input.shape[1],
            input.shape[2],
            input.shape[3],
        );
        let (kh, kw, _kc, oc) = (
            kernel.shape[0],
            kernel.shape[1],
            kernel.shape[2],
            kernel.shape[3],
        );
        let oh = (ih + pads[0] + pads[2] - kh) / stride_h + 1;
        let ow = (iw + pads[1] + pads[3] - kw) / stride_w + 1;
        let total = n * oh * ow * oc;
        let buf_out = self.output_buf(total);

        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct P {
            n: u32,
            ih: u32,
            iw: u32,
            ic: u32,
            oc: u32,
            kh: u32,
            kw: u32,
            sh: u32,
            sw: u32,
            oh: u32,
            ow: u32,
            pad_h: u32,
            pad_w: u32,
            _pad2: u32,
        }
        let p = P {
            n: n as u32,
            ih: ih as u32,
            iw: iw as u32,
            ic: input.shape[3] as u32,
            oc: oc as u32,
            kh: kh as u32,
            kw: kw as u32,
            sh: stride_h as u32,
            sw: stride_w as u32,
            oh: oh as u32,
            ow: ow as u32,
            pad_h: pads[0] as u32,
            pad_w: pads[1] as u32,
            _pad2: 0,
        };
        let buf_p = self.uniform_buf(bytemuck::bytes_of(&p));

        let bgl = self.pipelines.conv2d.get_bind_group_layout(0);
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: kernel.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: bias.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: buf_out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: buf_p.as_entire_binding(),
                },
            ],
        });
        self.record_compute(
            &self.pipelines.conv2d,
            &bg,
            (
                div_ceil(ow as u32, 8),
                div_ceil(oh as u32, 8),
                (n * oc) as u32,
            ),
        );
        GpuBuffer {
            buffer: buf_out,
            size: total,
            shape: vec![n, oh, ow, oc],
        }
    }

    /// Batched matmul: A[batch, M, K] × B[batch, K, N] = C[batch, M, N].
    /// Batch dims are flattened from the leading dims of the input shapes.
    pub fn batched_matmul_on_device(&self, a: &GpuBuffer, b: &GpuBuffer) -> GpuBuffer {
        let a_rank = a.shape.len();
        let b_rank = b.shape.len();
        let m = a.shape[a_rank - 2];
        let k = a.shape[a_rank - 1];
        let n = b.shape[b_rank - 1];
        // Batch = product of all dims except last 2
        let batch: usize = a.shape[..a_rank - 2].iter().product();
        let out_size = batch * m * n;
        let buf_out = self.output_buf(out_size);
        let params: [u32; 4] = [m as u32, n as u32, k as u32, batch as u32];
        let buf_p = self.uniform_buf(bytemuck::cast_slice(&params));

        let bgl = self.pipelines.batched_matmul.get_bind_group_layout(0);
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: a.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: b.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buf_out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: buf_p.as_entire_binding(),
                },
            ],
        });
        self.record_compute(
            &self.pipelines.batched_matmul,
            &bg,
            (div_ceil(n as u32, 64), div_ceil(m as u32, 64), batch as u32),
        );

        let mut out_shape = a.shape[..a_rank - 2].to_vec();
        out_shape.push(m);
        out_shape.push(n);
        GpuBuffer {
            buffer: buf_out,
            size: out_size,
            shape: out_shape,
        }
    }

    /// Conv2D via im2col + tiled GEMM — faster than direct conv for most sizes.
    /// Input NHWC [N,H,W,IC], weight KHWC [KH,KW,IC,OC], bias [OC].
    pub fn im2col_conv_on_device(
        &self,
        input: &GpuBuffer,
        kernel: &GpuBuffer,
        bias: &GpuBuffer,
        stride_h: usize,
        stride_w: usize,
        pads: [usize; 4],
        act: u32,
    ) -> GpuBuffer {
        let (n, ih, iw, ic) = (
            input.shape[0],
            input.shape[1],
            input.shape[2],
            input.shape[3],
        );
        let (kh, kw, _kc, oc) = (
            kernel.shape[0],
            kernel.shape[1],
            kernel.shape[2],
            kernel.shape[3],
        );
        let oh = (ih + pads[0] + pads[2] - kh) / stride_h + 1;
        let ow = (iw + pads[1] + pads[3] - kw) / stride_w + 1;
        let m = n * oh * ow;
        let col_k = kh * kw * ic;

        // All convolutions (including 1x1) go through fused conv_gemm kernel:
        // implicit im2col + tiled GEMM + bias + activation in one dispatch.
        let out_total = m * oc;
        let buf_out = self.output_buf(out_total);

        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct ConvGemmP {
            m: u32,
            n_out: u32,
            k: u32,
            act: u32,
            ih: u32,
            iw: u32,
            ic: u32,
            oh: u32,
            ow: u32,
            kh: u32,
            kw: u32,
            sh: u32,
            sw: u32,
            pad_h: u32,
            pad_w: u32,
            batch: u32,
        }
        let p = ConvGemmP {
            m: m as u32,
            n_out: oc as u32,
            k: col_k as u32,
            act,
            ih: ih as u32,
            iw: iw as u32,
            ic: ic as u32,
            oh: oh as u32,
            ow: ow as u32,
            kh: kh as u32,
            kw: kw as u32,
            sh: stride_h as u32,
            sw: stride_w as u32,
            pad_h: pads[0] as u32,
            pad_w: pads[1] as u32,
            batch: n as u32,
        };
        let buf_p = self.uniform_buf(bytemuck::bytes_of(&p));

        // Select optimal tile size: BN=32 variant for small N, BN=64 otherwise.
        let use_n32 = oc <= 48;
        let force_f32 = std::env::var("FORCE_F32_CONV").is_ok();
        let use_v2 = std::env::var("USE_V2_CONV").is_ok();
        let (pipeline, bn, bm) = if force_f32 {
            (&self.pipelines.conv_gemm, 64u32, 64u32)
        } else if use_v2 && !use_n32 {
            if let Some(ref p) = self.pipelines.conv_gemm_v2 {
                (p, 64u32, 128u32)
            } else {
                let p = self
                    .pipelines
                    .conv_gemm_f16
                    .as_ref()
                    .unwrap_or(&self.pipelines.conv_gemm);
                (p, 64u32, 64u32)
            }
        } else if use_n32 {
            // Prefer fast n32 (bitwise ops, unrolled) over original n32
            let p = self
                .pipelines
                .conv_gemm_fast_n32
                .as_ref()
                .or(self.pipelines.conv_gemm_f16_n32.as_ref());
            if let Some(p) = p {
                (p, 32u32, 64u32)
            } else {
                (&self.pipelines.conv_gemm, 64u32, 64u32)
            }
        } else {
            // Use fast shader (bitwise ops, fully unrolled inner loop)
            let p = self
                .pipelines
                .conv_gemm_fast
                .as_ref()
                .or(self.pipelines.conv_gemm_f16.as_ref())
                .unwrap_or(&self.pipelines.conv_gemm);
            (p, 64u32, 64u32)
        };
        let bgl = pipeline.get_bind_group_layout(0);
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: kernel.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: bias.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: buf_out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: buf_p.as_entire_binding(),
                },
            ],
        });
        let wg_x = div_ceil(oc as u32, bn);
        let wg_y = div_ceil(m as u32, bm);
        self.record_compute(pipeline, &bg, (wg_x, wg_y, 1));

        GpuBuffer {
            buffer: buf_out,
            size: out_total,
            shape: vec![n, oh, ow, oc],
        }
    }

    /// Pointwise (1x1) conv: reshape input + GEMM + bias + optional activation.
    #[allow(dead_code)]
    fn pointwise_conv_on_device(
        &self,
        input: &GpuBuffer,
        kernel: &GpuBuffer,
        bias: &GpuBuffer,
        oh: usize,
        ow: usize,
        act: u32,
    ) -> GpuBuffer {
        let n = input.shape[0];
        let ic = input.shape[3];
        let oc = kernel.shape[3];
        let m = n * oh * ow;
        let out_total = m * oc;

        // GEMM: input[M, IC] × kernel[IC, OC] = out[M, OC]
        let buf_out = self.output_buf(out_total);
        let gemm_p: [u32; 4] = [m as u32, oc as u32, ic as u32, 0];
        let buf_gp = self.uniform_buf(bytemuck::cast_slice(&gemm_p));

        let bgl = self.pipelines.matmul.get_bind_group_layout(0);
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: kernel.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buf_out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: buf_gp.as_entire_binding(),
                },
            ],
        });
        self.record_compute(
            &self.pipelines.matmul,
            &bg,
            (div_ceil(oc as u32, 64), div_ceil(m as u32, 64), 1),
        );

        // Bias add
        let bias_p: [u32; 2] = [out_total as u32, oc as u32];
        let buf_bp = self.uniform_buf(bytemuck::cast_slice(&bias_p));

        let bgl = self.pipelines.bias_add.get_bind_group_layout(0);
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf_out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: bias.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buf_bp.as_entire_binding(),
                },
            ],
        });
        self.record_compute(
            &self.pipelines.bias_add,
            &bg,
            (div_ceil(out_total as u32, 256), 1, 1),
        );

        // Fused activation (if any)
        if act == 1 {
            // ReLU in-place via unary
            return self.relu_on_device(&GpuBuffer {
                buffer: buf_out,
                size: out_total,
                shape: vec![n, oh, ow, oc],
            });
        } else if act == 2 {
            // SiLU in-place via unary
            return self.silu_on_device(&GpuBuffer {
                buffer: buf_out,
                size: out_total,
                shape: vec![n, oh, ow, oc],
            });
        }

        GpuBuffer {
            buffer: buf_out,
            size: out_total,
            shape: vec![n, oh, ow, oc],
        }
    }

    /// Winograd F(2,3) convolution for 3x3 stride-1 convolutions.
    /// wino_weight: pre-transformed [16, IC, OC], input: NHWC [N, IH, IW, IC].
    pub fn winograd_conv_on_device(
        &self,
        input: &GpuBuffer,
        wino_weight: &GpuBuffer,
        bias: &GpuBuffer,
        pads: [usize; 4],
        act: u32,
    ) -> GpuBuffer {
        let (n, ih, iw, ic) = (
            input.shape[0],
            input.shape[1],
            input.shape[2],
            input.shape[3],
        );
        let oc = wino_weight.shape[2];
        let oh = ih + pads[0] + pads[2] - 2; // kh=3, stride=1
        let ow = iw + pads[1] + pads[3] - 2;
        let tiles_h = oh.div_ceil(2);
        let tiles_w = ow.div_ceil(2);
        let num_tiles = n * tiles_h * tiles_w;

        // Step 1: Input transform → [16, num_tiles, IC]
        let inp_trans_size = 16 * num_tiles * ic;
        let buf_inp_trans = self.output_buf(inp_trans_size);

        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct InpP {
            num_tiles: u32,
            ic: u32,
            ih: u32,
            iw: u32,
            tiles_h: u32,
            tiles_w: u32,
            pad_h: u32,
            pad_w: u32,
            batch: u32,
            _p1: u32,
            _p2: u32,
            _p3: u32,
        }
        let inp_p = InpP {
            num_tiles: num_tiles as u32,
            ic: ic as u32,
            ih: ih as u32,
            iw: iw as u32,
            tiles_h: tiles_h as u32,
            tiles_w: tiles_w as u32,
            pad_h: pads[0] as u32,
            pad_w: pads[1] as u32,
            batch: n as u32,
            _p1: 0,
            _p2: 0,
            _p3: 0,
        };
        let buf_inp_p = self.uniform_buf(bytemuck::bytes_of(&inp_p));

        let bgl = self.pipelines.winograd_input.get_bind_group_layout(0);
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buf_inp_trans.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buf_inp_p.as_entire_binding(),
                },
            ],
        });
        let wg_count = (num_tiles * ic).div_ceil(256);
        self.record_compute(&self.pipelines.winograd_input, &bg, (wg_count as u32, 1, 1));

        // Step 2: Batched GEMM [16, num_tiles, IC] × [16, IC, OC] → [16, num_tiles, OC]
        let gemm_out_size = 16 * num_tiles * oc;
        let buf_gemm_out = self.output_buf(gemm_out_size);

        let gemm_p: [u32; 4] = [num_tiles as u32, oc as u32, ic as u32, 16];
        let buf_gp = self.uniform_buf(bytemuck::cast_slice(&gemm_p));

        let bgl = self.pipelines.batched_matmul.get_bind_group_layout(0);
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf_inp_trans.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wino_weight.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buf_gemm_out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: buf_gp.as_entire_binding(),
                },
            ],
        });
        self.record_compute(
            &self.pipelines.batched_matmul,
            &bg,
            (div_ceil(oc as u32, 64), div_ceil(num_tiles as u32, 64), 16),
        );

        // Step 3: Output transform + bias + activation → NHWC [N, OH, OW, OC]
        let out_total = n * oh * ow * oc;
        let buf_out = self.output_buf(out_total);

        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct OutP {
            num_tiles: u32,
            oc: u32,
            oh: u32,
            ow: u32,
            tiles_h: u32,
            tiles_w: u32,
            act: u32,
            batch: u32,
        }
        let out_p = OutP {
            num_tiles: num_tiles as u32,
            oc: oc as u32,
            oh: oh as u32,
            ow: ow as u32,
            tiles_h: tiles_h as u32,
            tiles_w: tiles_w as u32,
            act,
            batch: n as u32,
        };
        let buf_out_p = self.uniform_buf(bytemuck::bytes_of(&out_p));

        let bgl = self.pipelines.winograd_output.get_bind_group_layout(0);
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf_gemm_out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: bias.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buf_out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: buf_out_p.as_entire_binding(),
                },
            ],
        });
        let wg_count = (num_tiles * oc).div_ceil(256);
        self.record_compute(
            &self.pipelines.winograd_output,
            &bg,
            (wg_count as u32, 1, 1),
        );

        GpuBuffer {
            buffer: buf_out,
            size: out_total,
            shape: vec![n, oh, ow, oc],
        }
    }

    /// Depthwise conv2d NHWC on device with optional padding.
    pub fn depthwise_conv2d_nhwc_on_device(
        &self,
        input: &GpuBuffer,
        kernel: &GpuBuffer,
        bias: &GpuBuffer,
        stride_h: usize,
        stride_w: usize,
        pads: [usize; 4],
        act: u32,
    ) -> GpuBuffer {
        let (n, ih, iw, c) = (
            input.shape[0],
            input.shape[1],
            input.shape[2],
            input.shape[3],
        );
        let (kh, kw, _kc, dm) = (
            kernel.shape[0],
            kernel.shape[1],
            kernel.shape[2],
            kernel.shape[3],
        );
        let oh = (ih + pads[0] + pads[2] - kh) / stride_h + 1;
        let ow = (iw + pads[1] + pads[3] - kw) / stride_w + 1;
        let oc = c * dm;
        let total = n * oh * ow * oc;
        let buf_out = self.output_buf(total);

        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct P {
            n: u32,
            ih: u32,
            iw: u32,
            c: u32,
            dm: u32,
            kh: u32,
            kw: u32,
            sh: u32,
            sw: u32,
            oh: u32,
            ow: u32,
            pad_h: u32,
            pad_w: u32,
            act: u32,
        }
        let p = P {
            n: n as u32,
            ih: ih as u32,
            iw: iw as u32,
            c: c as u32,
            dm: dm as u32,
            kh: kh as u32,
            kw: kw as u32,
            sh: stride_h as u32,
            sw: stride_w as u32,
            oh: oh as u32,
            ow: ow as u32,
            pad_h: pads[0] as u32,
            pad_w: pads[1] as u32,
            act,
        };
        let buf_p = self.uniform_buf(bytemuck::bytes_of(&p));

        let bgl = self.pipelines.depthwise_conv2d.get_bind_group_layout(0);
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: kernel.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: bias.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: buf_out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: buf_p.as_entire_binding(),
                },
            ],
        });
        self.record_compute(
            &self.pipelines.depthwise_conv2d,
            &bg,
            (
                div_ceil(ow as u32, 16),
                div_ceil(oh as u32, 16),
                (n * oc) as u32,
            ),
        );
        GpuBuffer {
            buffer: buf_out,
            size: total,
            shape: vec![n, oh, ow, oc],
        }
    }

    /// BatchNorm NHWC on device.
    pub fn batch_norm2d_nhwc_on_device(
        &self,
        input: &GpuBuffer,
        gamma: &GpuBuffer,
        beta: &GpuBuffer,
        mean: &GpuBuffer,
        variance: &GpuBuffer,
        epsilon: f32,
    ) -> GpuBuffer {
        let c = input.shape[3];
        let total = input.size;
        let buf_out = self.output_buf(total);

        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct P {
            total: u32,
            c: u32,
            eps: f32,
            _pad: u32,
        }
        let p = P {
            total: total as u32,
            c: c as u32,
            eps: epsilon,
            _pad: 0,
        };
        let buf_p = self.uniform_buf(bytemuck::bytes_of(&p));

        let bgl = self.pipelines.batch_norm.get_bind_group_layout(0);
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: gamma.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: beta.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: mean.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: variance.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: buf_out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: buf_p.as_entire_binding(),
                },
            ],
        });
        self.record_compute(
            &self.pipelines.batch_norm,
            &bg,
            (div_ceil(total as u32, 256), 1, 1),
        );
        GpuBuffer {
            buffer: buf_out,
            size: total,
            shape: input.shape.clone(),
        }
    }

    /// Pool2D NHWC on device.  mode=0 for max, mode=1 for avg.
    /// `pads`: [pad_top, pad_left, pad_bottom, pad_right].
    pub fn pool2d_nhwc_on_device(
        &self,
        input: &GpuBuffer,
        kernel_h: usize,
        kernel_w: usize,
        stride_h: usize,
        stride_w: usize,
        mode: u32,
        pads: [usize; 4],
    ) -> GpuBuffer {
        let (n, ih, iw, c) = (
            input.shape[0],
            input.shape[1],
            input.shape[2],
            input.shape[3],
        );
        let oh = (ih + pads[0] + pads[2] - kernel_h) / stride_h + 1;
        let ow = (iw + pads[1] + pads[3] - kernel_w) / stride_w + 1;
        let total = n * oh * ow * c;
        let buf_out = self.output_buf(total);

        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct P {
            n: u32,
            ih: u32,
            iw: u32,
            c: u32,
            kh: u32,
            kw: u32,
            sh: u32,
            sw: u32,
            oh: u32,
            ow: u32,
            mode: u32,
            pad_h: u32,
            pad_w: u32,
            _pad2: u32,
        }
        let p = P {
            n: n as u32,
            ih: ih as u32,
            iw: iw as u32,
            c: c as u32,
            kh: kernel_h as u32,
            kw: kernel_w as u32,
            sh: stride_h as u32,
            sw: stride_w as u32,
            oh: oh as u32,
            ow: ow as u32,
            mode,
            pad_h: pads[0] as u32,
            pad_w: pads[1] as u32,
            _pad2: 0,
        };
        let buf_p = self.uniform_buf(bytemuck::bytes_of(&p));

        let bgl = self.pipelines.pool2d.get_bind_group_layout(0);
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buf_out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buf_p.as_entire_binding(),
                },
            ],
        });
        self.record_compute(
            &self.pipelines.pool2d,
            &bg,
            (
                div_ceil(ow as u32, 8),
                div_ceil(oh as u32, 8),
                (n * c) as u32,
            ),
        );
        GpuBuffer {
            buffer: buf_out,
            size: total,
            shape: vec![n, oh, ow, c],
        }
    }

    /// Softmax over last dimension, on device.
    pub fn softmax_on_device(&self, input: &GpuBuffer) -> GpuBuffer {
        let cols = *input.shape.last().expect("non-empty");
        let rows = input.size / cols;
        let total = rows * cols;
        let buf_out = self.output_buf(total);
        let params: [u32; 2] = [rows as u32, cols as u32];
        let buf_p = self.uniform_buf(bytemuck::cast_slice(&params));

        let bgl = self.pipelines.softmax.get_bind_group_layout(0);
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buf_out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buf_p.as_entire_binding(),
                },
            ],
        });
        self.record_compute(
            &self.pipelines.softmax,
            &bg,
            (div_ceil(rows as u32, 256), 1, 1),
        );
        GpuBuffer {
            buffer: buf_out,
            size: total,
            shape: input.shape.clone(),
        }
    }

    /// Split a tensor along the last axis on device.
    /// Returns one output for the channel slice [ch_offset .. ch_offset + c_out].
    pub fn channel_split_on_device(
        &self,
        input: &GpuBuffer,
        c_out: usize,
        ch_offset: usize,
    ) -> Result<GpuBuffer, KernelError> {
        let c_in = *input.shape.last().ok_or_else(|| KernelError::Gpu {
            message: "empty shape".into(),
        })?;
        let spatial: usize = input.size / c_in;
        let total = spatial * c_out;
        let buf_out = self.output_buf(total);

        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct P {
            spatial: u32,
            c_in: u32,
            c_out: u32,
            ch_offset: u32,
        }
        let p = P {
            spatial: spatial as u32,
            c_in: c_in as u32,
            c_out: c_out as u32,
            ch_offset: ch_offset as u32,
        };
        let buf_p = self.uniform_buf(bytemuck::bytes_of(&p));

        let bgl = self.pipelines.channel_split.get_bind_group_layout(0);
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buf_out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buf_p.as_entire_binding(),
                },
            ],
        });
        self.record_compute(
            &self.pipelines.channel_split,
            &bg,
            (div_ceil(total as u32, 256), 1, 1),
        );

        let mut out_shape = input.shape.clone();
        *out_shape.last_mut().ok_or_else(|| KernelError::Gpu {
            message: "empty shape".into(),
        })? = c_out;
        Ok(GpuBuffer {
            buffer: buf_out,
            size: total,
            shape: out_shape,
        })
    }

    /// Split a tensor along the last axis into multiple outputs, batched in one compute pass.
    /// `split_sizes` lists the channel count for each output.
    pub fn channel_split_all_on_device(
        &self,
        input: &GpuBuffer,
        split_sizes: &[usize],
    ) -> Result<Vec<GpuBuffer>, KernelError> {
        let c_in = *input.shape.last().ok_or_else(|| KernelError::Gpu {
            message: "empty shape".into(),
        })?;
        let spatial: usize = input.size / c_in;

        let bgl = self.pipelines.channel_split.get_bind_group_layout(0);
        let mut bind_groups = Vec::with_capacity(split_sizes.len());
        let mut wg_sizes = Vec::with_capacity(split_sizes.len());
        let mut results = Vec::with_capacity(split_sizes.len());

        let mut ch_offset = 0usize;
        for &c_out in split_sizes {
            let total = spatial * c_out;
            let buf_out = self.output_buf(total);

            #[repr(C)]
            #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
            struct P {
                spatial: u32,
                c_in: u32,
                c_out: u32,
                ch_offset: u32,
            }
            let p = P {
                spatial: spatial as u32,
                c_in: c_in as u32,
                c_out: c_out as u32,
                ch_offset: ch_offset as u32,
            };
            let buf_p = self.uniform_buf(bytemuck::bytes_of(&p));

            let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: input.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: buf_out.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: buf_p.as_entire_binding(),
                    },
                ],
            });
            bind_groups.push(bg);
            wg_sizes.push((div_ceil(total as u32, 256), 1u32, 1u32));

            let mut out_shape = input.shape.clone();
            *out_shape.last_mut().ok_or_else(|| KernelError::Gpu {
                message: "empty shape".into(),
            })? = c_out;
            results.push(GpuBuffer {
                buffer: buf_out,
                size: total,
                shape: out_shape,
            });
            ch_offset += c_out;
        }

        self.record_compute_batch(&self.pipelines.channel_split, &bind_groups, &wg_sizes);
        Ok(results)
    }

    /// Concatenate multiple tensors along the last axis on device.
    /// All inputs must have the same shape except for the last dimension.
    /// Batches all input copies into a single compute pass (no RAW/WAW hazards
    /// since each input writes to a non-overlapping channel range).
    pub fn channel_concat_on_device(
        &self,
        inputs: &[&GpuBuffer],
    ) -> Result<GpuBuffer, KernelError> {
        if inputs.is_empty() {
            return Err(KernelError::Gpu {
                message: "empty inputs".into(),
            });
        }
        let c_out: usize = {
            let mut sum = 0usize;
            for b in inputs {
                sum += *b.shape.last().ok_or_else(|| KernelError::Gpu {
                    message: "empty shape".into(),
                })?;
            }
            sum
        };
        let spatial: usize = inputs[0].size
            / *inputs[0].shape.last().ok_or_else(|| KernelError::Gpu {
                message: "empty shape".into(),
            })?;
        let total = spatial * c_out;
        let buf_out = self.output_buf(total);

        // Collect bind groups + workgroup sizes, then dispatch all in one pass.
        let bgl = self.pipelines.channel_concat.get_bind_group_layout(0);
        let mut bind_groups = Vec::with_capacity(inputs.len());
        let mut wg_sizes = Vec::with_capacity(inputs.len());

        let mut ch_offset = 0usize;
        for inp in inputs {
            let c_in = *inp.shape.last().ok_or_else(|| KernelError::Gpu {
                message: "empty shape".into(),
            })?;
            #[repr(C)]
            #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
            struct P {
                spatial: u32,
                c_in: u32,
                c_out: u32,
                ch_offset: u32,
            }
            let p = P {
                spatial: spatial as u32,
                c_in: c_in as u32,
                c_out: c_out as u32,
                ch_offset: ch_offset as u32,
            };
            let buf_p = self.uniform_buf(bytemuck::bytes_of(&p));

            let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: inp.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: buf_out.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: buf_p.as_entire_binding(),
                    },
                ],
            });
            bind_groups.push(bg);
            wg_sizes.push((div_ceil((spatial * c_in) as u32, 256), 1u32, 1u32));
            ch_offset += c_in;
        }

        // Single compute pass for all inputs — safe because writes are non-overlapping.
        self.record_compute_batch(&self.pipelines.channel_concat, &bind_groups, &wg_sizes);

        let mut out_shape = inputs[0].shape.clone();
        *out_shape.last_mut().ok_or_else(|| KernelError::Gpu {
            message: "empty shape".into(),
        })? = c_out;
        Ok(GpuBuffer {
            buffer: buf_out,
            size: total,
            shape: out_shape,
        })
    }

    /// Nearest-neighbor resize on NHWC tensor on device.
    pub fn resize_nearest_nhwc_on_device(
        &self,
        input: &GpuBuffer,
        oh: usize,
        ow: usize,
    ) -> GpuBuffer {
        let (n, _ih, _iw, c) = (
            input.shape[0],
            input.shape[1],
            input.shape[2],
            input.shape[3],
        );
        let total = n * oh * ow * c;
        let buf_out = self.output_buf(total);

        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct P {
            n: u32,
            ih: u32,
            iw: u32,
            c: u32,
            oh: u32,
            ow: u32,
            _p1: u32,
            _p2: u32,
        }
        let p = P {
            n: n as u32,
            ih: input.shape[1] as u32,
            iw: input.shape[2] as u32,
            c: c as u32,
            oh: oh as u32,
            ow: ow as u32,
            _p1: 0,
            _p2: 0,
        };
        let buf_p = self.uniform_buf(bytemuck::bytes_of(&p));

        let bgl = self.pipelines.resize_nearest.get_bind_group_layout(0);
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buf_out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buf_p.as_entire_binding(),
                },
            ],
        });
        self.record_compute(
            &self.pipelines.resize_nearest,
            &bg,
            (div_ceil(total as u32, 256), 1, 1),
        );
        GpuBuffer {
            buffer: buf_out,
            size: total,
            shape: vec![n, oh, ow, c],
        }
    }

    /// Permute a 4D NHWC tensor to NCHW on device.
    pub fn nhwc_to_nchw_on_device(&self, input: &GpuBuffer) -> GpuBuffer {
        let (n, h, w, c) = (
            input.shape[0],
            input.shape[1],
            input.shape[2],
            input.shape[3],
        );
        let total = n * c * h * w;
        let buf_out = self.output_buf(total);

        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct P {
            n: u32,
            h: u32,
            w: u32,
            c: u32,
        }
        let p = P {
            n: n as u32,
            h: h as u32,
            w: w as u32,
            c: c as u32,
        };
        let buf_p = self.uniform_buf(bytemuck::bytes_of(&p));

        let bgl = self.pipelines.permute_nhwc_nchw.get_bind_group_layout(0);
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buf_out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buf_p.as_entire_binding(),
                },
            ],
        });
        self.record_compute(
            &self.pipelines.permute_nhwc_nchw,
            &bg,
            (div_ceil(total as u32, 256), 1, 1),
        );
        GpuBuffer {
            buffer: buf_out,
            size: total,
            shape: vec![n, c, h, w],
        }
    }

    /// Change the shape metadata of a GpuBuffer without moving data.
    pub fn reshape_on_device(&self, input: GpuBuffer, new_shape: Vec<usize>) -> GpuBuffer {
        GpuBuffer {
            buffer: input.buffer,
            size: input.size,
            shape: new_shape,
        }
    }

    /// General N-dimensional permute on device (up to 6D).
    /// `perm` maps output dim → input dim: `out_shape[i] = in_shape[perm[i]]`.
    pub fn permute_on_device(&self, input: &GpuBuffer, perm: &[usize]) -> GpuBuffer {
        let ndim = input.shape.len();
        assert!(ndim <= 6 && perm.len() == ndim);

        // Compute input strides (row-major)
        let mut in_strides = vec![1usize; ndim];
        for i in (0..ndim - 1).rev() {
            in_strides[i] = in_strides[i + 1] * input.shape[i + 1];
        }

        // Output shape and permuted strides
        let mut out_shape = vec![0usize; ndim];
        let mut perm_strides = vec![0usize; ndim];
        for i in 0..ndim {
            out_shape[i] = input.shape[perm[i]];
            perm_strides[i] = in_strides[perm[i]];
        }
        let total: usize = input.size;

        // Pack into 6-element arrays (pad with 1 for shape, 0 for strides)
        let mut s = [1u32; 6];
        let mut t = [0u32; 6];
        for i in 0..ndim {
            s[i] = out_shape[i] as u32;
            t[i] = perm_strides[i] as u32;
        }

        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct P {
            total: u32,
            _p1: u32,
            _p2: u32,
            _p3: u32,
            s: [u32; 8], // 6 used + 2 pad
            t: [u32; 8], // 6 used + 2 pad
        }
        let p = P {
            total: total as u32,
            _p1: 0,
            _p2: 0,
            _p3: 0,
            s: [s[0], s[1], s[2], s[3], s[4], s[5], 0, 0],
            t: [t[0], t[1], t[2], t[3], t[4], t[5], 0, 0],
        };
        let buf_p = self.uniform_buf(bytemuck::bytes_of(&p));
        let buf_out = self.output_buf(total);

        let bgl = self.pipelines.permute_nd.get_bind_group_layout(0);
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buf_out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buf_p.as_entire_binding(),
                },
            ],
        });
        self.record_compute(
            &self.pipelines.permute_nd,
            &bg,
            (div_ceil(total as u32, 256), 1, 1),
        );
        GpuBuffer {
            buffer: buf_out,
            size: total,
            shape: out_shape,
        }
    }

    /// General concat along any axis on device.
    /// Batches all input copies into a single compute pass.
    pub fn general_concat_on_device(
        &self,
        inputs: &[&GpuBuffer],
        axis: usize,
    ) -> Result<GpuBuffer, KernelError> {
        if inputs.is_empty() {
            return Err(KernelError::Gpu {
                message: "empty inputs".into(),
            });
        }
        let inner: usize = inputs[0].shape[axis + 1..].iter().product();
        let outer: usize = inputs[0].shape[..axis].iter().product();
        let c_out: usize = inputs.iter().map(|b| b.shape[axis]).sum();
        let total = outer * c_out * inner;
        let buf_out = self.output_buf(total);

        let bgl = self.pipelines.general_concat.get_bind_group_layout(0);
        let mut bind_groups = Vec::with_capacity(inputs.len());
        let mut wg_sizes = Vec::with_capacity(inputs.len());

        let mut offset = 0usize;
        for inp in inputs {
            let c_in = inp.shape[axis];
            #[repr(C)]
            #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
            struct P {
                outer: u32,
                inner: u32,
                c_in: u32,
                c_out: u32,
                offset: u32,
                _p1: u32,
                _p2: u32,
                _p3: u32,
            }
            let p = P {
                outer: outer as u32,
                inner: inner as u32,
                c_in: c_in as u32,
                c_out: c_out as u32,
                offset: offset as u32,
                _p1: 0,
                _p2: 0,
                _p3: 0,
            };
            let buf_p = self.uniform_buf(bytemuck::bytes_of(&p));
            let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: inp.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: buf_out.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: buf_p.as_entire_binding(),
                    },
                ],
            });
            bind_groups.push(bg);
            wg_sizes.push((div_ceil((outer * c_in * inner) as u32, 256), 1u32, 1u32));
            offset += c_in;
        }

        self.record_compute_batch(&self.pipelines.general_concat, &bind_groups, &wg_sizes);

        let mut out_shape = inputs[0].shape.clone();
        out_shape[axis] = c_out;
        Ok(GpuBuffer {
            buffer: buf_out,
            size: total,
            shape: out_shape,
        })
    }

    /// General split along any axis on device. Returns one GpuBuffer per split.
    /// Batches all output copies into a single compute pass.
    pub fn general_split_on_device(
        &self,
        input: &GpuBuffer,
        axis: usize,
        split_sizes: &[usize],
    ) -> Vec<GpuBuffer> {
        let inner: usize = input.shape[axis + 1..].iter().product();
        let outer: usize = input.shape[..axis].iter().product();
        let c_in = input.shape[axis];

        let bgl = self.pipelines.general_split.get_bind_group_layout(0);
        let mut bind_groups = Vec::with_capacity(split_sizes.len());
        let mut wg_sizes = Vec::with_capacity(split_sizes.len());
        let mut results = Vec::with_capacity(split_sizes.len());
        let mut offset = 0usize;
        for &c_out in split_sizes {
            let total = outer * c_out * inner;
            let buf_out = self.output_buf(total);

            #[repr(C)]
            #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
            struct P {
                outer: u32,
                inner: u32,
                c_in: u32,
                c_out: u32,
                offset: u32,
                _p1: u32,
                _p2: u32,
                _p3: u32,
            }
            let p = P {
                outer: outer as u32,
                inner: inner as u32,
                c_in: c_in as u32,
                c_out: c_out as u32,
                offset: offset as u32,
                _p1: 0,
                _p2: 0,
                _p3: 0,
            };
            let buf_p = self.uniform_buf(bytemuck::bytes_of(&p));
            let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: input.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: buf_out.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: buf_p.as_entire_binding(),
                    },
                ],
            });
            bind_groups.push(bg);
            wg_sizes.push((div_ceil(total as u32, 256), 1u32, 1u32));

            let mut out_shape = input.shape.clone();
            out_shape[axis] = c_out;
            results.push(GpuBuffer {
                buffer: buf_out,
                size: total,
                shape: out_shape,
            });
            offset += c_out;
        }

        self.record_compute_batch(&self.pipelines.general_split, &bind_groups, &wg_sizes);
        results
    }

    /// General N-dimensional slice on device (step=1 only, up to 6D).
    /// `starts[d]` is the start index for dimension d. `out_shape[d]` = ends[d] - starts[d].
    pub fn slice_on_device(
        &self,
        input: &GpuBuffer,
        starts: &[usize],
        out_shape: &[usize],
    ) -> GpuBuffer {
        let ndim = input.shape.len();
        assert!(ndim <= 6 && starts.len() == ndim && out_shape.len() == ndim);

        let total: usize = out_shape.iter().product();
        let buf_out = self.output_buf(total);

        // Compute input strides
        let mut in_strides = vec![1usize; ndim];
        for i in (0..ndim - 1).rev() {
            in_strides[i] = in_strides[i + 1] * input.shape[i + 1];
        }

        // Pack into 6-element arrays
        let mut s = [1u32; 8];
        let mut t = [0u32; 8];
        let mut o = [0u32; 8];
        for i in 0..ndim {
            s[i] = out_shape[i] as u32;
            t[i] = in_strides[i] as u32;
            o[i] = starts[i] as u32;
        }

        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct P {
            total: u32,
            _p1: u32,
            _p2: u32,
            _p3: u32,
            s: [u32; 8],
            t: [u32; 8],
            o: [u32; 8],
        }
        let p = P {
            total: total as u32,
            _p1: 0,
            _p2: 0,
            _p3: 0,
            s,
            t,
            o,
        };
        let buf_p = self.uniform_buf(bytemuck::bytes_of(&p));

        let bgl = self.pipelines.slice_nd.get_bind_group_layout(0);
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buf_out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buf_p.as_entire_binding(),
                },
            ],
        });
        self.record_compute(
            &self.pipelines.slice_nd,
            &bg,
            (div_ceil(total as u32, 256), 1, 1),
        );
        GpuBuffer {
            buffer: buf_out,
            size: total,
            shape: out_shape.to_vec(),
        }
    }

    /// Copy a GPU buffer and assign a new shape. Device-side memcpy, no sync.
    pub fn copy_reshape_on_device(&self, input: &GpuBuffer, new_shape: Vec<usize>) -> GpuBuffer {
        let size_bytes = (input.size * 4) as u64;
        let buf_out = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: size_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc_opt = self.encoder.borrow_mut();
        let enc =
            enc_opt.get_or_insert_with(|| self.device.create_command_encoder(&Default::default()));
        enc.copy_buffer_to_buffer(&input.buffer, 0, &buf_out, 0, size_bytes);
        GpuBuffer {
            buffer: buf_out,
            size: input.size,
            shape: new_shape,
        }
    }
}

fn div_ceil(a: u32, b: u32) -> u32 {
    a.div_ceil(b)
}

fn same_shape_data(lhs: &Tensor, rhs: &Tensor) -> Option<usize> {
    if lhs.shape() == rhs.shape() {
        Some(lhs.data().len())
    } else {
        None
    }
}

// ── Backend trait implementation ───────────────────────────────────

impl Backend for GpuBackend {
    fn matmul_2d(&self, lhs: &Tensor, rhs: &Tensor) -> Result<Tensor, KernelError> {
        if lhs.shape().len() != 2 || rhs.shape().len() != 2 {
            return Err(KernelError::InvalidMatMulRank {
                left_rank: lhs.shape().len(),
                right_rank: rhs.shape().len(),
            });
        }
        let (m, k) = (lhs.shape()[0], lhs.shape()[1]);
        let (k2, n) = (rhs.shape()[0], rhs.shape()[1]);
        if k != k2 {
            return Err(KernelError::MatMulShapeMismatch {
                left: lhs.shape().to_vec(),
                right: rhs.shape().to_vec(),
            });
        }

        if m * n < MIN_GPU_ELEMENTS {
            return crate::matmul_2d(lhs, rhs);
        }

        let buf_a = self.storage_buf(lhs.data());
        let buf_b = self.storage_buf(rhs.data());
        let buf_out = self.output_buf(m * n);
        let params: [u32; 4] = [m as u32, n as u32, k as u32, 0];
        let buf_p = self.uniform_buf(bytemuck::cast_slice(&params));

        let bgl = self.pipelines.matmul.get_bind_group_layout(0);
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf_a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buf_b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buf_out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: buf_p.as_entire_binding(),
                },
            ],
        });

        self.record_compute(
            &self.pipelines.matmul,
            &bg,
            (div_ceil(n as u32, 64), div_ceil(m as u32, 64), 1),
        );

        let data = self.read_buf(&buf_out, m * n)?;
        Tensor::from_vec(vec![m, n], data).map_err(Into::into)
    }

    fn add(&self, lhs: &Tensor, rhs: &Tensor) -> Result<Tensor, KernelError> {
        if let Some(len) = same_shape_data(lhs, rhs)
            && len >= MIN_GPU_ELEMENTS
        {
            let out = self.dispatch_elementwise(lhs.data(), rhs.data(), len, 0)?;
            return Tensor::from_vec(lhs.shape().to_vec(), out).map_err(Into::into);
        }
        crate::add(lhs, rhs)
    }

    fn sub(&self, lhs: &Tensor, rhs: &Tensor) -> Result<Tensor, KernelError> {
        if let Some(len) = same_shape_data(lhs, rhs)
            && len >= MIN_GPU_ELEMENTS
        {
            let out = self.dispatch_elementwise(lhs.data(), rhs.data(), len, 1)?;
            return Tensor::from_vec(lhs.shape().to_vec(), out).map_err(Into::into);
        }
        crate::sub(lhs, rhs)
    }

    fn mul(&self, lhs: &Tensor, rhs: &Tensor) -> Result<Tensor, KernelError> {
        if let Some(len) = same_shape_data(lhs, rhs)
            && len >= MIN_GPU_ELEMENTS
        {
            let out = self.dispatch_elementwise(lhs.data(), rhs.data(), len, 2)?;
            return Tensor::from_vec(lhs.shape().to_vec(), out).map_err(Into::into);
        }
        crate::mul(lhs, rhs)
    }

    fn relu(&self, input: &Tensor) -> Tensor {
        let len = input.data().len();
        if len < MIN_GPU_ELEMENTS {
            return crate::relu(input);
        }
        let out = self
            .dispatch_unary(input.data(), len, 0)
            .expect("GPU read-back failed in relu (trait prevents Result propagation)");
        Tensor::from_vec(input.shape().to_vec(), out).expect("shape matches data")
    }

    fn sigmoid(&self, input: &Tensor) -> Tensor {
        let len = input.data().len();
        if len < MIN_GPU_ELEMENTS {
            return crate::sigmoid(input);
        }
        let out = self
            .dispatch_unary(input.data(), len, 1)
            .expect("GPU read-back failed in sigmoid (trait prevents Result propagation)");
        Tensor::from_vec(input.shape().to_vec(), out).expect("shape matches data")
    }

    fn exp(&self, input: &Tensor) -> Tensor {
        let len = input.data().len();
        if len < MIN_GPU_ELEMENTS {
            return crate::exp(input);
        }
        let out = self
            .dispatch_unary(input.data(), len, 2)
            .expect("GPU read-back failed in exp (trait prevents Result propagation)");
        Tensor::from_vec(input.shape().to_vec(), out).expect("shape matches data")
    }

    fn tanh_act(&self, input: &Tensor) -> Tensor {
        let len = input.data().len();
        if len < MIN_GPU_ELEMENTS {
            return crate::tanh_act(input);
        }
        let out = self
            .dispatch_unary(input.data(), len, 3)
            .expect("GPU read-back failed in tanh_act (trait prevents Result propagation)");
        Tensor::from_vec(input.shape().to_vec(), out).expect("shape matches data")
    }

    fn softmax_last_dim(&self, input: &Tensor) -> Result<Tensor, KernelError> {
        let shape = input.shape();
        if shape.is_empty() {
            return Err(KernelError::InvalidSoftmaxRank { got_rank: 0 });
        }
        let cols = *shape.last().expect("non-empty shape");
        let rows = input.data().len() / cols;
        if rows * cols < MIN_GPU_ELEMENTS {
            return crate::softmax_last_dim(input);
        }

        let buf_in = self.storage_buf(input.data());
        let buf_out = self.output_buf(rows * cols);
        let params: [u32; 2] = [rows as u32, cols as u32];
        let buf_p = self.uniform_buf(bytemuck::cast_slice(&params));

        let bgl = self.pipelines.softmax.get_bind_group_layout(0);
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf_in.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buf_out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buf_p.as_entire_binding(),
                },
            ],
        });

        self.record_compute(
            &self.pipelines.softmax,
            &bg,
            (div_ceil(rows as u32, 256), 1, 1),
        );
        let data = self.read_buf(&buf_out, rows * cols)?;
        Tensor::from_vec(shape.to_vec(), data).map_err(Into::into)
    }

    fn log_softmax_last_dim(&self, input: &Tensor) -> Result<Tensor, KernelError> {
        let shape = input.shape();
        if shape.is_empty() {
            return Err(KernelError::InvalidLogSoftmaxRank { got_rank: 0 });
        }
        let cols = *shape.last().expect("non-empty shape");
        let rows = input.data().len() / cols;
        if rows * cols < MIN_GPU_ELEMENTS {
            return crate::log_softmax_last_dim(input);
        }

        let buf_in = self.storage_buf(input.data());
        let buf_out = self.output_buf(rows * cols);
        let params: [u32; 2] = [rows as u32, cols as u32];
        let buf_p = self.uniform_buf(bytemuck::cast_slice(&params));

        let bgl = self.pipelines.log_softmax.get_bind_group_layout(0);
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf_in.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buf_out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buf_p.as_entire_binding(),
                },
            ],
        });

        self.record_compute(
            &self.pipelines.log_softmax,
            &bg,
            (div_ceil(rows as u32, 256), 1, 1),
        );
        let data = self.read_buf(&buf_out, rows * cols)?;
        Tensor::from_vec(shape.to_vec(), data).map_err(Into::into)
    }

    fn logsumexp_last_dim(&self, input: &Tensor) -> Result<Tensor, KernelError> {
        let shape = input.shape();
        if shape.is_empty() {
            return Err(KernelError::InvalidLogSumExpRank { got_rank: 0 });
        }
        let cols = *shape.last().expect("non-empty shape");
        let rows = input.data().len() / cols;
        if rows * cols < MIN_GPU_ELEMENTS {
            return crate::logsumexp_last_dim(input);
        }

        let buf_in = self.storage_buf(input.data());
        let buf_out = self.output_buf(rows);
        let params: [u32; 2] = [rows as u32, cols as u32];
        let buf_p = self.uniform_buf(bytemuck::cast_slice(&params));

        let bgl = self.pipelines.logsumexp.get_bind_group_layout(0);
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf_in.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buf_out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buf_p.as_entire_binding(),
                },
            ],
        });

        self.record_compute(
            &self.pipelines.logsumexp,
            &bg,
            (div_ceil(rows as u32, 256), 1, 1),
        );
        let data = self.read_buf(&buf_out, rows)?;
        let mut out_shape = shape[..shape.len() - 1].to_vec();
        if out_shape.is_empty() {
            out_shape.push(1);
        }
        Tensor::from_vec(out_shape, data).map_err(Into::into)
    }

    fn layer_norm_last_dim(
        &self,
        input: &Tensor,
        params: LayerNormLastDimParams<'_>,
    ) -> Result<Tensor, KernelError> {
        let shape = input.shape();
        if shape.is_empty() {
            return Err(KernelError::InvalidLayerNormRank { got_rank: 0 });
        }
        let cols = *shape.last().expect("non-empty shape");
        let rows = input.data().len() / cols;
        if rows * cols < MIN_GPU_ELEMENTS {
            return crate::layer_norm_last_dim(input, params);
        }

        let buf_in = self.storage_buf(input.data());
        let buf_g = self.storage_buf(params.gamma.data());
        let buf_b = self.storage_buf(params.beta.data());
        let buf_out = self.output_buf(rows * cols);

        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct P {
            rows: u32,
            cols: u32,
            eps: f32,
            _pad: u32,
        }
        let p = P {
            rows: rows as u32,
            cols: cols as u32,
            eps: params.epsilon,
            _pad: 0,
        };
        let buf_p = self.uniform_buf(bytemuck::bytes_of(&p));

        let bgl = self.pipelines.layer_norm.get_bind_group_layout(0);
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf_in.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buf_g.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buf_b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: buf_out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: buf_p.as_entire_binding(),
                },
            ],
        });

        self.record_compute(
            &self.pipelines.layer_norm,
            &bg,
            (div_ceil(rows as u32, 256), 1, 1),
        );
        let data = self.read_buf(&buf_out, rows * cols)?;
        Tensor::from_vec(shape.to_vec(), data).map_err(Into::into)
    }

    fn conv2d_nhwc(
        &self,
        input: &Tensor,
        kernel: &Tensor,
        bias: Option<&Tensor>,
        stride_h: usize,
        stride_w: usize,
    ) -> Result<Tensor, KernelError> {
        let is = input.shape();
        let ks = kernel.shape();
        if is.len() != 4 || ks.len() != 4 {
            return Err(KernelError::InvalidConvRank {
                input_rank: is.len(),
                kernel_rank: ks.len(),
            });
        }
        let (n, ih, iw, ic) = (is[0], is[1], is[2], is[3]);
        let (kh, kw, kc, oc) = (ks[0], ks[1], ks[2], ks[3]);
        if ic != kc {
            return Err(KernelError::ConvChannelMismatch {
                input_channels: ic,
                kernel_in_channels: kc,
            });
        }
        let oh = (ih - kh) / stride_h + 1;
        let ow = (iw - kw) / stride_w + 1;
        let total = n * oh * ow * oc;

        if total < MIN_GPU_ELEMENTS {
            return crate::conv2d_nhwc(input, kernel, bias, stride_h, stride_w);
        }

        let bias_data = if let Some(b) = bias {
            b.data().to_vec()
        } else {
            vec![0.0f32; oc]
        };

        let buf_in = self.storage_buf(input.data());
        let buf_k = self.storage_buf(kernel.data());
        let buf_bias = self.storage_buf(&bias_data);
        let buf_out = self.output_buf(total);

        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct P {
            n: u32,
            ih: u32,
            iw: u32,
            ic: u32,
            oc: u32,
            kh: u32,
            kw: u32,
            sh: u32,
            sw: u32,
            oh: u32,
            ow: u32,
            pad_h: u32,
            pad_w: u32,
            _pad2: u32,
        }
        let p = P {
            n: n as u32,
            ih: ih as u32,
            iw: iw as u32,
            ic: ic as u32,
            oc: oc as u32,
            kh: kh as u32,
            kw: kw as u32,
            sh: stride_h as u32,
            sw: stride_w as u32,
            oh: oh as u32,
            ow: ow as u32,
            pad_h: 0,
            pad_w: 0,
            _pad2: 0,
        };
        let buf_p = self.uniform_buf(bytemuck::bytes_of(&p));

        let bgl = self.pipelines.conv2d.get_bind_group_layout(0);
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf_in.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buf_k.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buf_bias.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: buf_out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: buf_p.as_entire_binding(),
                },
            ],
        });

        self.record_compute(
            &self.pipelines.conv2d,
            &bg,
            (
                div_ceil(ow as u32, 8),
                div_ceil(oh as u32, 8),
                (n * oc) as u32,
            ),
        );
        let data = self.read_buf(&buf_out, total)?;
        Tensor::from_vec(vec![n, oh, ow, oc], data).map_err(Into::into)
    }

    fn max_pool2d_nhwc(
        &self,
        input: &Tensor,
        kernel_h: usize,
        kernel_w: usize,
        stride_h: usize,
        stride_w: usize,
    ) -> Result<Tensor, KernelError> {
        self.dispatch_pool(input, kernel_h, kernel_w, stride_h, stride_w, 0)
    }

    fn avg_pool2d_nhwc(
        &self,
        input: &Tensor,
        kernel_h: usize,
        kernel_w: usize,
        stride_h: usize,
        stride_w: usize,
    ) -> Result<Tensor, KernelError> {
        self.dispatch_pool(input, kernel_h, kernel_w, stride_h, stride_w, 1)
    }

    fn batch_norm2d_nhwc(
        &self,
        input: &Tensor,
        params: BatchNorm2dParams<'_>,
    ) -> Result<Tensor, KernelError> {
        let shape = input.shape();
        if shape.len() != 4 {
            return Err(KernelError::InvalidBatchNormRank {
                got_rank: shape.len(),
            });
        }
        let c = shape[3];
        let total = input.data().len();

        if total < MIN_GPU_ELEMENTS {
            return crate::batch_norm2d_nhwc(input, params);
        }

        let buf_in = self.storage_buf(input.data());
        let buf_g = self.storage_buf(params.gamma.data());
        let buf_b = self.storage_buf(params.beta.data());
        let buf_m = self.storage_buf(params.mean.data());
        let buf_v = self.storage_buf(params.variance.data());
        let buf_out = self.output_buf(total);

        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct P {
            total: u32,
            c: u32,
            eps: f32,
            _pad: u32,
        }
        let p = P {
            total: total as u32,
            c: c as u32,
            eps: params.epsilon,
            _pad: 0,
        };
        let buf_p = self.uniform_buf(bytemuck::bytes_of(&p));

        let bgl = self.pipelines.batch_norm.get_bind_group_layout(0);
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf_in.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buf_g.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buf_b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: buf_m.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: buf_v.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: buf_out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: buf_p.as_entire_binding(),
                },
            ],
        });

        self.record_compute(
            &self.pipelines.batch_norm,
            &bg,
            (div_ceil(total as u32, 256), 1, 1),
        );
        let data = self.read_buf(&buf_out, total)?;
        Tensor::from_vec(shape.to_vec(), data).map_err(Into::into)
    }

    fn depthwise_conv2d_nhwc(
        &self,
        input: &Tensor,
        kernel: &Tensor,
        bias: Option<&Tensor>,
        stride_h: usize,
        stride_w: usize,
    ) -> Result<Tensor, KernelError> {
        let is = input.shape();
        let ks = kernel.shape();
        if is.len() != 4 || ks.len() != 4 {
            return crate::depthwise_conv2d_nhwc(input, kernel, bias, stride_h, stride_w);
        }
        let (n, ih, iw, c) = (is[0], is[1], is[2], is[3]);
        let (kh, kw, _kc, dm) = (ks[0], ks[1], ks[2], ks[3]);
        let oh = (ih - kh) / stride_h + 1;
        let ow = (iw - kw) / stride_w + 1;
        let oc = c * dm;
        let total = n * oh * ow * oc;
        if total < MIN_GPU_ELEMENTS {
            return crate::depthwise_conv2d_nhwc(input, kernel, bias, stride_h, stride_w);
        }
        let bias_data = if let Some(b) = bias {
            b.data().to_vec()
        } else {
            vec![0.0f32; oc]
        };
        let buf_in = self.storage_buf(input.data());
        let buf_k = self.storage_buf(kernel.data());
        let buf_bias = self.storage_buf(&bias_data);
        let buf_out = self.output_buf(total);

        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct P {
            n: u32,
            ih: u32,
            iw: u32,
            c: u32,
            dm: u32,
            kh: u32,
            kw: u32,
            sh: u32,
            sw: u32,
            oh: u32,
            ow: u32,
            pad_h: u32,
            pad_w: u32,
            act: u32,
        }
        let p = P {
            n: n as u32,
            ih: ih as u32,
            iw: iw as u32,
            c: c as u32,
            dm: dm as u32,
            kh: kh as u32,
            kw: kw as u32,
            sh: stride_h as u32,
            sw: stride_w as u32,
            oh: oh as u32,
            ow: ow as u32,
            pad_h: 0,
            pad_w: 0,
            act: 0,
        };
        let buf_p = self.uniform_buf(bytemuck::bytes_of(&p));
        let bgl = self.pipelines.depthwise_conv2d.get_bind_group_layout(0);
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf_in.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buf_k.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buf_bias.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: buf_out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: buf_p.as_entire_binding(),
                },
            ],
        });
        self.record_compute(
            &self.pipelines.depthwise_conv2d,
            &bg,
            (
                div_ceil(ow as u32, 16),
                div_ceil(oh as u32, 16),
                (n * oc) as u32,
            ),
        );
        let data = self.read_buf(&buf_out, total)?;
        Tensor::from_vec(vec![n, oh, ow, oc], data).map_err(Into::into)
    }

    fn separable_conv2d_nhwc(
        &self,
        input: &Tensor,
        params: SeparableConv2dParams<'_>,
        stride_h: usize,
        stride_w: usize,
    ) -> Result<Tensor, KernelError> {
        crate::separable_conv2d_nhwc(input, params, stride_h, stride_w)
    }

    fn group_norm_nhwc(
        &self,
        input: &Tensor,
        params: GroupNormNhwcParams<'_>,
    ) -> Result<Tensor, KernelError> {
        let shape = input.shape();
        if shape.len() != 4 {
            return crate::group_norm_nhwc(input, params);
        }
        let (n, h, w, c) = (shape[0], shape[1], shape[2], shape[3]);
        let spatial = h * w;
        let total = n * c * spatial;
        if total < MIN_GPU_ELEMENTS {
            return crate::group_norm_nhwc(input, params);
        }

        let buf_in = self.storage_buf(input.data());
        let buf_g = self.storage_buf(params.gamma.data());
        let buf_b = self.storage_buf(params.beta.data());
        let buf_out = self.output_buf(total);

        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct P {
            n: u32,
            c: u32,
            spatial: u32,
            groups: u32,
            eps: f32,
            _pad1: u32,
            _pad2: u32,
            _pad3: u32,
        }
        let p = P {
            n: n as u32,
            c: c as u32,
            spatial: spatial as u32,
            groups: params.num_groups as u32,
            eps: params.epsilon,
            _pad1: 0,
            _pad2: 0,
            _pad3: 0,
        };
        let buf_p = self.uniform_buf(bytemuck::bytes_of(&p));

        let bgl = self.pipelines.group_norm.get_bind_group_layout(0);
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf_in.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buf_g.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buf_b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: buf_out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: buf_p.as_entire_binding(),
                },
            ],
        });

        self.record_compute(
            &self.pipelines.group_norm,
            &bg,
            (div_ceil(total as u32, 256), 1, 1),
        );
        let data = self.read_buf(&buf_out, total)?;
        Tensor::from_vec(shape.to_vec(), data).map_err(Into::into)
    }

    fn rms_norm_last_dim(
        &self,
        input: &Tensor,
        params: RmsNormLastDimParams<'_>,
    ) -> Result<Tensor, KernelError> {
        let shape = input.shape();
        if shape.is_empty() {
            return crate::rms_norm_last_dim(input, params);
        }
        let cols = *shape.last().expect("non-empty shape");
        let rows = input.data().len() / cols;
        if rows * cols < MIN_GPU_ELEMENTS {
            return crate::rms_norm_last_dim(input, params);
        }

        let buf_in = self.storage_buf(input.data());
        let buf_g = self.storage_buf(params.gamma.data());
        let buf_out = self.output_buf(rows * cols);

        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct P {
            rows: u32,
            cols: u32,
            eps: f32,
            _pad: u32,
        }
        let p = P {
            rows: rows as u32,
            cols: cols as u32,
            eps: params.epsilon,
            _pad: 0,
        };
        let buf_p = self.uniform_buf(bytemuck::bytes_of(&p));

        let bgl = self.pipelines.rms_norm.get_bind_group_layout(0);
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf_in.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buf_g.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buf_out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: buf_p.as_entire_binding(),
                },
            ],
        });

        self.record_compute(
            &self.pipelines.rms_norm,
            &bg,
            (div_ceil(rows as u32, 256), 1, 1),
        );
        let data = self.read_buf(&buf_out, rows * cols)?;
        Tensor::from_vec(shape.to_vec(), data).map_err(Into::into)
    }
}

impl crate::BackwardOps for GpuBackend {
    fn relu_backward(
        &self,
        upstream: &Tensor,
        forward_input: &Tensor,
    ) -> Result<Tensor, KernelError> {
        let len = upstream.data().len();
        if len < MIN_GPU_ELEMENTS {
            // Fall back to default CPU implementation.
            let u = upstream.data();
            let f = forward_input.data();
            let out: Vec<f32> = u
                .iter()
                .zip(f.iter())
                .map(|(&u, &x)| if x > 0.0 { u } else { 0.0 })
                .collect();
            return Tensor::from_vec(upstream.shape().to_vec(), out).map_err(Into::into);
        }
        let out = self.dispatch_backward_binary(upstream.data(), forward_input.data(), len, 0)?;
        Tensor::from_vec(upstream.shape().to_vec(), out).map_err(Into::into)
    }

    fn sigmoid_backward(
        &self,
        upstream: &Tensor,
        forward_output: &Tensor,
    ) -> Result<Tensor, KernelError> {
        let len = upstream.data().len();
        if len < MIN_GPU_ELEMENTS {
            let u = upstream.data();
            let s = forward_output.data();
            let out: Vec<f32> = u
                .iter()
                .zip(s.iter())
                .map(|(&u, &s)| u * s * (1.0 - s))
                .collect();
            return Tensor::from_vec(upstream.shape().to_vec(), out).map_err(Into::into);
        }
        let out = self.dispatch_backward_binary(upstream.data(), forward_output.data(), len, 1)?;
        Tensor::from_vec(upstream.shape().to_vec(), out).map_err(Into::into)
    }

    fn tanh_backward(
        &self,
        upstream: &Tensor,
        forward_output: &Tensor,
    ) -> Result<Tensor, KernelError> {
        let len = upstream.data().len();
        if len < MIN_GPU_ELEMENTS {
            let u = upstream.data();
            let t = forward_output.data();
            let out: Vec<f32> = u
                .iter()
                .zip(t.iter())
                .map(|(&u, &t)| u * (1.0 - t * t))
                .collect();
            return Tensor::from_vec(upstream.shape().to_vec(), out).map_err(Into::into);
        }
        let out = self.dispatch_backward_binary(upstream.data(), forward_output.data(), len, 2)?;
        Tensor::from_vec(upstream.shape().to_vec(), out).map_err(Into::into)
    }

    fn exp_backward(
        &self,
        upstream: &Tensor,
        forward_output: &Tensor,
    ) -> Result<Tensor, KernelError> {
        let len = upstream.data().len();
        if len < MIN_GPU_ELEMENTS {
            let u = upstream.data();
            let e = forward_output.data();
            let out: Vec<f32> = u.iter().zip(e.iter()).map(|(&u, &e)| u * e).collect();
            return Tensor::from_vec(upstream.shape().to_vec(), out).map_err(Into::into);
        }
        let out = self.dispatch_backward_binary(upstream.data(), forward_output.data(), len, 3)?;
        Tensor::from_vec(upstream.shape().to_vec(), out).map_err(Into::into)
    }

    fn reduce_sum_backward(
        &self,
        upstream: &Tensor,
        original_shape: &[usize],
    ) -> Result<Tensor, KernelError> {
        let len: usize = original_shape.iter().product();
        if len < MIN_GPU_ELEMENTS {
            let grad_val = upstream.data()[0];
            let out = vec![grad_val; len];
            return Tensor::from_vec(original_shape.to_vec(), out).map_err(Into::into);
        }
        let out = self.dispatch_reduce_sum_backward(upstream.data(), len)?;
        Tensor::from_vec(original_shape.to_vec(), out).map_err(Into::into)
    }

    fn matmul_backward(
        &self,
        upstream: &Tensor,
        lhs: &Tensor,
        rhs: &Tensor,
    ) -> Result<(Tensor, Tensor), KernelError> {
        let rt = self.transpose_2d(rhs)?;
        let lt = self.transpose_2d(lhs)?;
        let grad_lhs = self.matmul_2d(upstream, &rt)?;
        let grad_rhs = self.matmul_2d(&lt, upstream)?;
        Ok((grad_lhs, grad_rhs))
    }

    fn add_backward(
        &self,
        upstream: &Tensor,
        _lhs: &Tensor,
        _rhs: &Tensor,
    ) -> Result<(Tensor, Tensor), KernelError> {
        Ok((upstream.clone(), upstream.clone()))
    }

    fn sub_backward(
        &self,
        upstream: &Tensor,
        _lhs: &Tensor,
        _rhs: &Tensor,
    ) -> Result<(Tensor, Tensor), KernelError> {
        Ok((upstream.clone(), self.neg(upstream)))
    }

    fn mul_backward(
        &self,
        upstream: &Tensor,
        lhs: &Tensor,
        rhs: &Tensor,
    ) -> Result<(Tensor, Tensor), KernelError> {
        let grad_lhs = self.mul(upstream, rhs)?;
        let grad_rhs = self.mul(upstream, lhs)?;
        Ok((grad_lhs, grad_rhs))
    }

    fn conv2d_input_backward(
        &self,
        upstream: &Tensor,
        kernel: &Tensor,
        input_shape: &[usize],
        stride_h: usize,
        stride_w: usize,
    ) -> Result<Tensor, KernelError> {
        let us = upstream.shape();
        let ks = kernel.shape();
        if us.len() != 4 || ks.len() != 4 || input_shape.len() != 4 {
            return Err(KernelError::InvalidConvRank {
                input_rank: input_shape.len(),
                kernel_rank: ks.len(),
            });
        }
        let (n, ih, iw, ic) = (
            input_shape[0],
            input_shape[1],
            input_shape[2],
            input_shape[3],
        );
        let (_n, oh, ow, oc) = (us[0], us[1], us[2], us[3]);
        let (kh, kw) = (ks[0], ks[1]);
        let total = n * ih * iw * ic;

        if total < MIN_GPU_ELEMENTS {
            // CPU fallback via default trait implementation.
            let u_data = upstream.data();
            let k_data = kernel.data();
            let mut grad_input = vec![0.0f32; total];
            for b in 0..n {
                for oy in 0..oh {
                    for ox in 0..ow {
                        for co in 0..oc {
                            let g = u_data[((b * oh + oy) * ow + ox) * oc + co];
                            if g == 0.0 {
                                continue;
                            }
                            for ky_i in 0..kh {
                                for kx_i in 0..kw {
                                    let iy = oy * stride_h + ky_i;
                                    let ix = ox * stride_w + kx_i;
                                    if iy < ih && ix < iw {
                                        for ci in 0..ic {
                                            let k_val =
                                                k_data[((ky_i * kw + kx_i) * ic + ci) * oc + co];
                                            grad_input[((b * ih + iy) * iw + ix) * ic + ci] +=
                                                g * k_val;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            return Tensor::from_vec(input_shape.to_vec(), grad_input).map_err(Into::into);
        }

        let buf_up = self.storage_buf(upstream.data());
        let buf_k = self.storage_buf(kernel.data());
        let buf_out = self.output_buf(total);

        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct P {
            n: u32,
            ih: u32,
            iw: u32,
            ic: u32,
            oc: u32,
            kh: u32,
            kw: u32,
            sh: u32,
            sw: u32,
            oh: u32,
            ow: u32,
            _pad: u32,
        }
        let p = P {
            n: n as u32,
            ih: ih as u32,
            iw: iw as u32,
            ic: ic as u32,
            oc: oc as u32,
            kh: kh as u32,
            kw: kw as u32,
            sh: stride_h as u32,
            sw: stride_w as u32,
            oh: oh as u32,
            ow: ow as u32,
            _pad: 0,
        };
        let buf_p = self.uniform_buf(bytemuck::bytes_of(&p));

        let bgl = self.pipelines.conv2d_input_grad.get_bind_group_layout(0);
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf_up.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buf_k.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buf_out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: buf_p.as_entire_binding(),
                },
            ],
        });

        self.record_compute(
            &self.pipelines.conv2d_input_grad,
            &bg,
            (
                div_ceil(iw as u32, 8),
                div_ceil(ih as u32, 8),
                (n * ic) as u32,
            ),
        );
        let data = self.read_buf(&buf_out, total)?;
        Tensor::from_vec(input_shape.to_vec(), data).map_err(Into::into)
    }
}

impl GpuBackend {
    fn dispatch_pool(
        &self,
        input: &Tensor,
        kernel_h: usize,
        kernel_w: usize,
        stride_h: usize,
        stride_w: usize,
        mode: u32,
    ) -> Result<Tensor, KernelError> {
        let shape = input.shape();
        if shape.len() != 4 {
            return Err(KernelError::InvalidPoolRank {
                got_rank: shape.len(),
            });
        }
        let (n, ih, iw, c) = (shape[0], shape[1], shape[2], shape[3]);
        let oh = (ih - kernel_h) / stride_h + 1;
        let ow = (iw - kernel_w) / stride_w + 1;
        let total = n * oh * ow * c;

        if total < MIN_GPU_ELEMENTS {
            if mode == 0 {
                return crate::max_pool2d_nhwc(input, kernel_h, kernel_w, stride_h, stride_w);
            } else {
                return crate::avg_pool2d_nhwc(input, kernel_h, kernel_w, stride_h, stride_w);
            }
        }

        let buf_in = self.storage_buf(input.data());
        let buf_out = self.output_buf(total);

        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct P {
            n: u32,
            ih: u32,
            iw: u32,
            c: u32,
            kh: u32,
            kw: u32,
            sh: u32,
            sw: u32,
            oh: u32,
            ow: u32,
            mode: u32,
            pad_h: u32,
            pad_w: u32,
            _pad2: u32,
        }
        let p = P {
            n: n as u32,
            ih: ih as u32,
            iw: iw as u32,
            c: c as u32,
            kh: kernel_h as u32,
            kw: kernel_w as u32,
            sh: stride_h as u32,
            sw: stride_w as u32,
            oh: oh as u32,
            ow: ow as u32,
            mode,
            pad_h: 0,
            pad_w: 0,
            _pad2: 0,
        };
        let buf_p = self.uniform_buf(bytemuck::bytes_of(&p));

        let bgl = self.pipelines.pool2d.get_bind_group_layout(0);
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf_in.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buf_out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buf_p.as_entire_binding(),
                },
            ],
        });

        self.record_compute(
            &self.pipelines.pool2d,
            &bg,
            (
                div_ceil(ow as u32, 8),
                div_ceil(oh as u32, 8),
                (n * c) as u32,
            ),
        );
        let data = self.read_buf(&buf_out, total)?;
        Tensor::from_vec(vec![n, oh, ow, c], data).map_err(Into::into)
    }

    /// Transposed convolution (deconvolution) on GPU.
    ///
    /// Input: `[N,H,W,C_in]`, kernel: `[KH,KW,C_in,C_out]`, bias: `[C_out]`.
    /// Output: `[N, (H-1)*stride_h + KH, (W-1)*stride_w + KW, C_out]`.
    pub fn transpose_conv2d_nhwc(
        &self,
        input: &Tensor,
        kernel: &Tensor,
        bias: Option<&Tensor>,
        stride_h: usize,
        stride_w: usize,
    ) -> Result<Tensor, KernelError> {
        let is = input.shape();
        let ks = kernel.shape();
        if is.len() != 4 || ks.len() != 4 {
            return Err(KernelError::InvalidConvRank {
                input_rank: is.len(),
                kernel_rank: ks.len(),
            });
        }
        let (n, ih, iw, ic) = (is[0], is[1], is[2], is[3]);
        let (kh, kw, _kc, oc) = (ks[0], ks[1], ks[2], ks[3]);
        let oh = (ih - 1) * stride_h + kh;
        let ow = (iw - 1) * stride_w + kw;
        let total = n * oh * ow * oc;

        let bias_data = if let Some(b) = bias {
            b.data().to_vec()
        } else {
            vec![0.0f32; oc]
        };
        let buf_in = self.storage_buf(input.data());
        let buf_k = self.storage_buf(kernel.data());
        let buf_bias = self.storage_buf(&bias_data);
        let buf_out = self.output_buf(total);

        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct P {
            n: u32,
            ih: u32,
            iw: u32,
            ic: u32,
            oc: u32,
            kh: u32,
            kw: u32,
            sh: u32,
            sw: u32,
            oh: u32,
            ow: u32,
            _pad: u32,
        }
        let p = P {
            n: n as u32,
            ih: ih as u32,
            iw: iw as u32,
            ic: ic as u32,
            oc: oc as u32,
            kh: kh as u32,
            kw: kw as u32,
            sh: stride_h as u32,
            sw: stride_w as u32,
            oh: oh as u32,
            ow: ow as u32,
            _pad: 0,
        };
        let buf_p = self.uniform_buf(bytemuck::bytes_of(&p));
        let bgl = self.pipelines.transpose_conv2d.get_bind_group_layout(0);
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf_in.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buf_k.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buf_bias.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: buf_out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: buf_p.as_entire_binding(),
                },
            ],
        });
        self.record_compute(
            &self.pipelines.transpose_conv2d,
            &bg,
            (
                div_ceil(ow as u32, 8),
                div_ceil(oh as u32, 8),
                (n * oc) as u32,
            ),
        );
        let data = self.read_buf(&buf_out, total)?;
        Tensor::from_vec(vec![n, oh, ow, oc], data).map_err(Into::into)
    }
}

// ── Standalone GPU dispatch functions ─────────────────────────────

/// GPU batch normalization: normalize across spatial dims (NHWC layout).
///
/// Applies: `output[i] = gamma[ch] * (input[i] - mean[ch]) / sqrt(var[ch] + epsilon) + beta[ch]`
/// where `ch = i % C`.
///
/// Falls back to CPU if the GPU backend cannot be created.
pub fn gpu_batch_norm(
    input: &Tensor,
    gamma: &Tensor,
    beta: &Tensor,
    mean: &Tensor,
    var: &Tensor,
    epsilon: f32,
) -> Result<Tensor, KernelError> {
    let backend = GpuBackend::new()?;
    let params = BatchNorm2dParams {
        gamma,
        beta,
        mean,
        variance: var,
        epsilon,
    };
    backend.batch_norm2d_nhwc(input, params)
}

/// GPU layer normalization: normalize across the last dimension.
///
/// For each row, computes mean and variance over the last dim, then applies:
/// `output[row, j] = gamma[j] * (input[row, j] - mu) / sqrt(var + epsilon) + beta[j]`
///
/// Falls back to CPU if the GPU backend cannot be created.
pub fn gpu_layer_norm(
    input: &Tensor,
    gamma: &Tensor,
    beta: &Tensor,
    epsilon: f32,
) -> Result<Tensor, KernelError> {
    let backend = GpuBackend::new()?;
    let params = LayerNormLastDimParams {
        gamma,
        beta,
        epsilon,
    };
    backend.layer_norm_last_dim(input, params)
}

/// GPU 2D matrix transpose: transposes a rank-2 tensor `[M, N]` to `[N, M]`.
///
/// Falls back to CPU if the GPU backend cannot be created.
pub fn gpu_transpose(input: &Tensor) -> Result<Tensor, KernelError> {
    let shape = input.shape();
    if shape.len() != 2 {
        return Err(KernelError::InvalidMatMulRank {
            left_rank: shape.len(),
            right_rank: 0,
        });
    }
    let backend = GpuBackend::new()?;
    let rows = shape[0];
    let cols = shape[1];

    let buf_in = backend.storage_buf(input.data());
    let buf_out = backend.output_buf(rows * cols);

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct P {
        rows: u32,
        cols: u32,
    }
    let p = P {
        rows: rows as u32,
        cols: cols as u32,
    };
    let buf_p = backend.uniform_buf(bytemuck::bytes_of(&p));

    let bgl = backend.pipelines.transpose_2d.get_bind_group_layout(0);
    let bg = backend
        .device
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf_in.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buf_out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buf_p.as_entire_binding(),
                },
            ],
        });

    backend.record_compute(
        &backend.pipelines.transpose_2d,
        &bg,
        (div_ceil(cols as u32, 16), div_ceil(rows as u32, 16), 1),
    );
    let data = backend.read_buf(&buf_out, rows * cols);
    Tensor::from_vec(vec![cols, rows], data?).map_err(Into::into)
}

// ── f16 bit conversion (no external crate needed) ─────────────────

fn f32_to_f16_bits(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xFF) as i32;
    let mantissa = bits & 0x007FFFFF;

    if exponent == 0xFF {
        return sign | 0x7C00 | if mantissa != 0 { 0x0200 } else { 0 };
    }
    let unbiased = exponent - 127;
    if unbiased < -24 {
        return sign;
    }
    if unbiased < -14 {
        let shift = -1 - unbiased;
        let subnormal = ((mantissa | 0x00800000) >> (shift + 13)) as u16;
        return sign | subnormal;
    }
    if unbiased > 15 {
        return sign | 0x7C00;
    }
    let fp16_exp = ((unbiased + 15) as u16) << 10;
    let fp16_man = (mantissa >> 13) as u16;
    sign | fp16_exp | fp16_man
}

fn f16_bits_to_f32(half: u16) -> f32 {
    let sign = ((half & 0x8000) as u32) << 16;
    let exponent = (half >> 10) & 0x1F;
    let mantissa = (half & 0x03FF) as u32;
    if exponent == 0 {
        if mantissa == 0 {
            return f32::from_bits(sign);
        }
        let mut e = 0i32;
        let mut m = mantissa;
        while m & 0x0400 == 0 {
            m <<= 1;
            e += 1;
        }
        let f32_exp = ((127 - 15 - e) as u32) << 23;
        let f32_man = (m & 0x03FF) << 13;
        return f32::from_bits(sign | f32_exp | f32_man);
    }
    if exponent == 31 {
        let f32_bits = sign | 0x7F800000 | if mantissa != 0 { 0x00400000 } else { 0 };
        return f32::from_bits(f32_bits);
    }
    let f32_exp = ((exponent as u32) + 112) << 23;
    let f32_man = mantissa << 13;
    f32::from_bits(sign | f32_exp | f32_man)
}
