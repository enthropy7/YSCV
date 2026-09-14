//! GPU multi-device scheduling: device enumeration, multi-device dispatch,
//! and scheduling strategies for distributing work across GPUs.

use std::sync::atomic::{AtomicUsize, Ordering};

use yscv_tensor::Tensor;

use super::gpu_backend::GpuBackend;
use crate::{
    Backend, BatchNorm2dParams, GroupNormNhwcParams, KernelError, LayerNormLastDimParams,
    RmsNormLastDimParams, SeparableConv2dParams,
};

// ---------------------------------------------------------------------------
// Device info
// ---------------------------------------------------------------------------

/// Type of GPU device.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuDeviceType {
    DiscreteGpu,
    IntegratedGpu,
    VirtualGpu,
    Cpu,
    Other,
}

/// GPU API backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuApiBackend {
    Vulkan,
    Metal,
    Dx12,
    Other,
}

/// Metadata about a discovered GPU adapter.
#[derive(Debug, Clone)]
pub struct GpuDeviceInfo {
    pub name: String,
    pub device_type: GpuDeviceType,
    pub backend: GpuApiBackend,
    pub max_buffer_size: u64,
    pub max_compute_workgroups: [u32; 3],
}

/// Enumerates all available GPU adapters via `wgpu`.
///
/// Queries every backend (Vulkan, Metal, DX12, ...) and returns a
/// [`GpuDeviceInfo`] for each discovered adapter, including its name, type,
/// API backend, maximum buffer size, and compute workgroup limits.
///
/// Call this once at startup to decide which device indices to pass to
/// [`MultiGpuBackend::new`].
pub fn enumerate_gpu_devices() -> Vec<GpuDeviceInfo> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });

    let adapters = pollster::block_on(instance.enumerate_adapters(wgpu::Backends::all()));
    adapters
        .iter()
        .map(|adapter: &wgpu::Adapter| {
            let info = adapter.get_info();
            let limits = adapter.limits();
            GpuDeviceInfo {
                name: info.name.clone(),
                device_type: match info.device_type {
                    wgpu::DeviceType::DiscreteGpu => GpuDeviceType::DiscreteGpu,
                    wgpu::DeviceType::IntegratedGpu => GpuDeviceType::IntegratedGpu,
                    wgpu::DeviceType::VirtualGpu => GpuDeviceType::VirtualGpu,
                    wgpu::DeviceType::Cpu => GpuDeviceType::Cpu,
                    _ => GpuDeviceType::Other,
                },
                backend: match info.backend {
                    wgpu::Backend::Vulkan => GpuApiBackend::Vulkan,
                    wgpu::Backend::Metal => GpuApiBackend::Metal,
                    wgpu::Backend::Dx12 => GpuApiBackend::Dx12,
                    _ => GpuApiBackend::Other,
                },
                max_buffer_size: limits.max_buffer_size,
                max_compute_workgroups: [
                    limits.max_compute_workgroups_per_dimension,
                    limits.max_compute_workgroups_per_dimension,
                    limits.max_compute_workgroups_per_dimension,
                ],
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Scheduling strategy
// ---------------------------------------------------------------------------

/// Strategy for selecting which GPU device to use for an operation.
///
/// * **`RoundRobin`** -- best for workloads with many independent, roughly
///   equal-cost operations. Each successive call picks the next device.
/// * **`DataParallel`** -- best for large matrix multiplications where the
///   left operand can be split by rows. Non-matmul ops fall back to device 0.
/// * **`Manual(idx)`** -- use when you want explicit control (e.g. placing
///   specific layers on specific devices for model-parallel setups).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchedulingStrategy {
    /// Cycle through devices in round-robin order.
    RoundRobin,
    /// Split the batch / row dimension of matmul across all devices.
    DataParallel,
    /// Pin all operations to the device at the given index.
    Manual(usize),
}

// ---------------------------------------------------------------------------
// Multi-GPU Backend
// ---------------------------------------------------------------------------

/// Manages multiple GPU devices and dispatches compute operations according
/// to a [`SchedulingStrategy`].
///
/// `MultiGpuBackend` holds one [`GpuBackend`] per selected device and
/// implements the [`Backend`] trait so it can be used as a drop-in replacement
/// for a single-device backend. Depending on the strategy:
///
/// * [`RoundRobin`](SchedulingStrategy::RoundRobin) -- successive operations
///   are sent to the next device in a cycle. Good for workloads with many
///   independent small ops.
/// * [`DataParallel`](SchedulingStrategy::DataParallel) -- matrix
///   multiplications are split along the row dimension of the left operand
///   and distributed across all devices; other ops default to device 0.
/// * [`Manual(idx)`](SchedulingStrategy::Manual) -- all ops are pinned to a
///   specific device index.
///
/// # Example
///
/// ```rust,ignore
/// let mgpu = MultiGpuBackend::new_all()?.with_strategy(SchedulingStrategy::DataParallel);
/// let result = mgpu.matmul_2d(&big_lhs, &rhs)?; // split across GPUs
/// ```
pub struct MultiGpuBackend {
    devices: Vec<GpuBackend>,
    device_infos: Vec<GpuDeviceInfo>,
    strategy: SchedulingStrategy,
    round_robin_counter: AtomicUsize,
}

impl MultiGpuBackend {
    /// Creates backends for the specified device indices.
    pub fn new(device_indices: &[usize]) -> Result<Self, KernelError> {
        let all_infos = enumerate_gpu_devices();
        let mut devices = Vec::new();
        let mut infos = Vec::new();

        for &idx in device_indices {
            if idx >= all_infos.len() {
                return Err(KernelError::Gpu {
                    message: format!(
                        "device index {idx} out of range (available: {})",
                        all_infos.len()
                    ),
                });
            }
            devices.push(GpuBackend::new()?);
            infos.push(all_infos[idx].clone());
        }

        if devices.is_empty() {
            return Err(KernelError::Gpu {
                message: "no GPU devices selected".into(),
            });
        }

        Ok(Self {
            devices,
            device_infos: infos,
            strategy: SchedulingStrategy::RoundRobin,
            round_robin_counter: AtomicUsize::new(0),
        })
    }

    /// Creates backends for all available devices.
    pub fn new_all() -> Result<Self, KernelError> {
        let infos = enumerate_gpu_devices();
        if infos.is_empty() {
            return Err(KernelError::Gpu {
                message: "no GPU devices available".into(),
            });
        }
        let indices: Vec<usize> = (0..infos.len()).collect();
        Self::new(&indices)
    }

    /// Sets the scheduling strategy.
    pub const fn with_strategy(mut self, strategy: SchedulingStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Number of managed devices.
    pub const fn device_count(&self) -> usize {
        self.devices.len()
    }

    /// Device info for the given index.
    pub fn device_info(&self, index: usize) -> Option<&GpuDeviceInfo> {
        self.device_infos.get(index)
    }

    /// Current scheduling strategy.
    pub const fn strategy(&self) -> SchedulingStrategy {
        self.strategy
    }

    /// Selects a device based on the current strategy.
    fn select_device(&self) -> &GpuBackend {
        match self.strategy {
            SchedulingStrategy::RoundRobin => {
                let idx =
                    self.round_robin_counter.fetch_add(1, Ordering::Relaxed) % self.devices.len();
                &self.devices[idx]
            }
            SchedulingStrategy::DataParallel => {
                // For non-matmul ops, fall back to first device
                &self.devices[0]
            }
            SchedulingStrategy::Manual(idx) => {
                let safe_idx = idx.min(self.devices.len() - 1);
                &self.devices[safe_idx]
            }
        }
    }

    /// Data-parallel matmul: splits left matrix rows across devices.
    fn data_parallel_matmul(&self, lhs: &Tensor, rhs: &Tensor) -> Result<Tensor, KernelError> {
        let shape = lhs.shape();
        if shape.len() != 2 {
            return self.devices[0].matmul_2d(lhs, rhs);
        }

        let m = shape[0];
        let n_devices = self.devices.len();

        if m < n_devices {
            // Not enough rows to split; use single device
            return self.devices[0].matmul_2d(lhs, rhs);
        }

        let rows_per_device = m / n_devices;
        let mut results = Vec::with_capacity(n_devices);

        for (i, device) in self.devices.iter().enumerate() {
            let start = i * rows_per_device;
            let end = if i == n_devices - 1 {
                m
            } else {
                start + rows_per_device
            };
            let num_rows = end - start;

            // Extract row slice
            let k = shape[1];
            let lhs_data = lhs.data();
            let slice_data: Vec<f32> = lhs_data[start * k..end * k].to_vec();
            let slice_tensor = Tensor::from_vec(vec![num_rows, k], slice_data)?;

            let partial: Tensor = device.matmul_2d(&slice_tensor, rhs)?;
            results.push(partial);
        }

        // Concatenate results along row dimension
        let rhs_shape = rhs.shape();
        let n = rhs_shape[1];
        let mut combined: Vec<f32> = Vec::with_capacity(m * n);
        for partial in &results {
            combined.extend_from_slice(partial.data());
        }

        Tensor::from_vec(vec![m, n], combined).map_err(Into::into)
    }
}

impl crate::BackwardOps for MultiGpuBackend {}

impl Backend for MultiGpuBackend {
    fn add(&self, lhs: &Tensor, rhs: &Tensor) -> Result<Tensor, KernelError> {
        self.select_device().add(lhs, rhs)
    }

    fn sub(&self, lhs: &Tensor, rhs: &Tensor) -> Result<Tensor, KernelError> {
        self.select_device().sub(lhs, rhs)
    }

    fn mul(&self, lhs: &Tensor, rhs: &Tensor) -> Result<Tensor, KernelError> {
        self.select_device().mul(lhs, rhs)
    }

    fn relu(&self, input: &Tensor) -> Tensor {
        self.select_device().relu(input)
    }

    fn sigmoid(&self, input: &Tensor) -> Tensor {
        self.select_device().sigmoid(input)
    }

    fn exp(&self, input: &Tensor) -> Tensor {
        self.select_device().exp(input)
    }

    fn tanh_act(&self, input: &Tensor) -> Tensor {
        self.select_device().tanh_act(input)
    }

    fn softmax_last_dim(&self, input: &Tensor) -> Result<Tensor, KernelError> {
        self.select_device().softmax_last_dim(input)
    }

    fn log_softmax_last_dim(&self, input: &Tensor) -> Result<Tensor, KernelError> {
        self.select_device().log_softmax_last_dim(input)
    }

    fn logsumexp_last_dim(&self, input: &Tensor) -> Result<Tensor, KernelError> {
        self.select_device().logsumexp_last_dim(input)
    }

    fn layer_norm_last_dim(
        &self,
        input: &Tensor,
        params: LayerNormLastDimParams<'_>,
    ) -> Result<Tensor, KernelError> {
        self.select_device().layer_norm_last_dim(input, params)
    }

    fn max_pool2d_nhwc(
        &self,
        input: &Tensor,
        kernel_h: usize,
        kernel_w: usize,
        stride_h: usize,
        stride_w: usize,
    ) -> Result<Tensor, KernelError> {
        self.select_device()
            .max_pool2d_nhwc(input, kernel_h, kernel_w, stride_h, stride_w)
    }

    fn avg_pool2d_nhwc(
        &self,
        input: &Tensor,
        kernel_h: usize,
        kernel_w: usize,
        stride_h: usize,
        stride_w: usize,
    ) -> Result<Tensor, KernelError> {
        self.select_device()
            .avg_pool2d_nhwc(input, kernel_h, kernel_w, stride_h, stride_w)
    }

    fn conv2d_nhwc(
        &self,
        input: &Tensor,
        kernel: &Tensor,
        bias: Option<&Tensor>,
        stride_h: usize,
        stride_w: usize,
    ) -> Result<Tensor, KernelError> {
        self.select_device()
            .conv2d_nhwc(input, kernel, bias, stride_h, stride_w)
    }

    fn depthwise_conv2d_nhwc(
        &self,
        input: &Tensor,
        kernel: &Tensor,
        bias: Option<&Tensor>,
        stride_h: usize,
        stride_w: usize,
    ) -> Result<Tensor, KernelError> {
        self.select_device()
            .depthwise_conv2d_nhwc(input, kernel, bias, stride_h, stride_w)
    }

    fn separable_conv2d_nhwc(
        &self,
        input: &Tensor,
        params: SeparableConv2dParams<'_>,
        stride_h: usize,
        stride_w: usize,
    ) -> Result<Tensor, KernelError> {
        self.select_device()
            .separable_conv2d_nhwc(input, params, stride_h, stride_w)
    }

    fn batch_norm2d_nhwc(
        &self,
        input: &Tensor,
        params: BatchNorm2dParams<'_>,
    ) -> Result<Tensor, KernelError> {
        self.select_device().batch_norm2d_nhwc(input, params)
    }

    fn group_norm_nhwc(
        &self,
        input: &Tensor,
        params: GroupNormNhwcParams<'_>,
    ) -> Result<Tensor, KernelError> {
        self.select_device().group_norm_nhwc(input, params)
    }

    fn rms_norm_last_dim(
        &self,
        input: &Tensor,
        params: RmsNormLastDimParams<'_>,
    ) -> Result<Tensor, KernelError> {
        self.select_device().rms_norm_last_dim(input, params)
    }

    fn matmul_2d(&self, lhs: &Tensor, rhs: &Tensor) -> Result<Tensor, KernelError> {
        match self.strategy {
            SchedulingStrategy::DataParallel => self.data_parallel_matmul(lhs, rhs),
            _ => self.select_device().matmul_2d(lhs, rhs),
        }
    }
}
