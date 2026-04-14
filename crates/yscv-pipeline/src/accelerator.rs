//! Accelerator enumeration and runtime availability probing.

use serde::{Deserialize, Serialize};

/// All accelerators that a task may be assigned to.
///
/// User picks one explicitly per task in the TOML config. The framework
/// never auto-selects — if the chosen accelerator is unavailable, startup
/// fails with [`crate::ConfigError::AcceleratorUnavailable`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case")]
pub enum Accelerator {
    /// Pure CPU execution (NEON / AVX dispatch).
    Cpu,
    /// wgpu cross-platform GPU (Vulkan / Metal / DX12). Built with `--features gpu`.
    Gpu,
    /// Rockchip NPU. `core` selects which NPU core(s) to pin to.
    /// Built with `--features rknn`.
    Rknn { core: NpuCoreSpec },
    /// RKNN matmul accelerator (independent unit, useful for LLM dequant
    /// or attention tiles). Built with `--features rknn`.
    ///
    /// Carries the matrix dimensions + dtype because the matmul context
    /// is shape-bound at construction (the SDK pre-allocates buffers
    /// sized for `M×K`, `K×N`, `M×N`). For dynamic-M LLM workloads,
    /// callers can use `yscv_kernels::RknnMatmul::new_dynamic` directly
    /// — this config-driven path covers the common single-shape case.
    RknnMatmul {
        m: u32,
        k: u32,
        n: u32,
        dtype: MatmulDtype,
    },
    /// Apple MPSGraph compiled execution (macOS dev path).
    /// Built with `--features metal-backend`.
    MetalMps,
}

/// TOML-facing matmul dtype. Maps onto `yscv_kernels::RknnMatmulType`.
/// See the SDK header `rknn_matmul_api.h` for the full type matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum MatmulDtype {
    /// FP16 × FP16 → FP32. Default; works on every Rockchip SoC with
    /// the matmul accelerator.
    Fp16MmFp16ToFp32,
    /// FP16 × FP16 → FP16. Smaller output, useful for chained matmuls.
    Fp16MmFp16ToFp16,
    /// INT8 × INT8 → INT32. Quantized GEMM; `set_quant_params` exposed
    /// via the underlying `RknnMatmul` if per-channel scales are needed.
    Int8MmInt8ToInt32,
    /// FP16 × INT4 → FP16. The classic LLM-dequant tile; B (weights)
    /// is INT4-packed, A (activations) FP16, output FP16.
    Fp16MmInt4ToFp16,
}

#[cfg(feature = "rknn")]
impl MatmulDtype {
    /// Convert this TOML-facing dtype into the kernel-level
    /// `RknnMatmulType`. Available under `--features rknn`.
    pub fn to_rknn(self) -> yscv_kernels::RknnMatmulType {
        use yscv_kernels::RknnMatmulType as T;
        match self {
            MatmulDtype::Fp16MmFp16ToFp32 => T::Float16MmFloat16ToFloat32,
            MatmulDtype::Fp16MmFp16ToFp16 => T::Float16MmFloat16ToFloat16,
            MatmulDtype::Int8MmInt8ToInt32 => T::Int8MmInt8ToInt32,
            MatmulDtype::Fp16MmInt4ToFp16 => T::Float16MmInt4ToFloat16,
        }
    }
}

/// Which NPU core(s) the task should be pinned to. Maps directly onto
/// `yscv_kernels::NpuCoreMask` when the rknn feature is enabled.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum NpuCoreSpec {
    /// Driver-chosen core (good default for single-task pipelines).
    Auto,
    Core0,
    Core1,
    Core2,
    /// Cores 0 and 1 in batch / parallel mode.
    Cores01,
    Cores02,
    Cores12,
    /// All 3 cores (RK3588) for max throughput on a single inference.
    Cores012,
    /// All available cores (driver-decided mask).
    All,
}

#[cfg(feature = "rknn")]
impl NpuCoreSpec {
    /// Convert this TOML-facing spec into the kernel-level
    /// `NpuCoreMask`. Only available under `--features rknn`.
    pub fn to_mask(self) -> yscv_kernels::NpuCoreMask {
        use yscv_kernels::NpuCoreMask as M;
        match self {
            NpuCoreSpec::Auto => M::Auto,
            NpuCoreSpec::Core0 => M::Core0,
            NpuCoreSpec::Core1 => M::Core1,
            NpuCoreSpec::Core2 => M::Core2,
            NpuCoreSpec::Cores01 => M::Cores01,
            NpuCoreSpec::Cores02 => M::Cores02,
            NpuCoreSpec::Cores12 => M::Cores12,
            NpuCoreSpec::Cores012 => M::Cores012,
            NpuCoreSpec::All => M::All,
        }
    }
}

impl Accelerator {
    /// Short label for display / error messages.
    pub fn label(&self) -> String {
        match self {
            Accelerator::Cpu => "cpu".into(),
            Accelerator::Gpu => "gpu".into(),
            Accelerator::Rknn { core } => format!("rknn ({core:?})"),
            Accelerator::RknnMatmul { m, k, n, dtype } => {
                format!("rknn-matmul ({m}×{k}×{n} {dtype:?})")
            }
            Accelerator::MetalMps => "metal-mps".into(),
        }
    }

    /// Cargo feature flag the user would have to enable to make this
    /// accelerator buildable. CPU is always available.
    pub fn feature_hint(&self) -> &'static str {
        match self {
            Accelerator::Cpu => "(none — CPU is always available)",
            Accelerator::Gpu => "gpu",
            Accelerator::Rknn { .. } => "rknn",
            Accelerator::RknnMatmul { .. } => "rknn",
            Accelerator::MetalMps => "metal-backend",
        }
    }
}

/// Runtime availability snapshot of every supported accelerator.
///
/// Returned by [`probe_accelerators`] at startup. Used by
/// [`crate::PipelineConfig::validate_accelerators`] to fail-fast when the
/// TOML demands something the host can't provide.
#[derive(Debug, Clone, Copy, Default)]
pub struct AcceleratorAvailability {
    pub cpu: bool,
    pub gpu: bool,
    pub rknn: bool,
    pub metal_mps: bool,
}

/// Probe the host for every accelerator's runtime availability.
///
/// - CPU is always `true`.
/// - GPU is `true` if compiled with `--features gpu`. Wgpu adapter init
///   isn't probed here (too expensive at startup); failures surface at
///   first dispatch.
/// - RKNN is `true` iff `--features rknn` AND `librknnrt.so` is loadable.
/// - Metal MPS is `true` iff compiled with `--features metal-backend` on macOS.
pub fn probe_accelerators() -> AcceleratorAvailability {
    AcceleratorAvailability {
        cpu: true,
        gpu: cfg!(feature = "gpu"),
        rknn: probe_rknn(),
        metal_mps: cfg!(all(feature = "metal-backend", target_os = "macos")),
    }
}

#[cfg(feature = "rknn")]
fn probe_rknn() -> bool {
    yscv_kernels::rknn_available()
}

#[cfg(not(feature = "rknn"))]
fn probe_rknn() -> bool {
    false
}

impl AcceleratorAvailability {
    /// Check whether the given accelerator is usable on this host.
    pub fn supports(&self, acc: &Accelerator) -> bool {
        match acc {
            Accelerator::Cpu => self.cpu,
            Accelerator::Gpu => self.gpu,
            Accelerator::Rknn { .. } | Accelerator::RknnMatmul { .. } => self.rknn,
            Accelerator::MetalMps => self.metal_mps,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_always_available() {
        let avail = probe_accelerators();
        assert!(avail.cpu, "CPU must be unconditionally available");
    }

    #[test]
    fn label_format() {
        assert_eq!(Accelerator::Cpu.label(), "cpu");
        assert!(
            Accelerator::Rknn {
                core: NpuCoreSpec::Core0
            }
            .label()
            .starts_with("rknn")
        );
    }

    #[test]
    fn feature_hint_consistent() {
        assert_eq!(
            Accelerator::Rknn {
                core: NpuCoreSpec::Auto
            }
            .feature_hint(),
            "rknn"
        );
        assert_eq!(Accelerator::Gpu.feature_hint(), "gpu");
        assert_eq!(Accelerator::MetalMps.feature_hint(), "metal-backend");
    }

    #[test]
    fn supports_cpu_unconditionally() {
        let avail = AcceleratorAvailability {
            cpu: true,
            gpu: false,
            rknn: false,
            metal_mps: false,
        };
        assert!(avail.supports(&Accelerator::Cpu));
        assert!(!avail.supports(&Accelerator::Gpu));
        assert!(!avail.supports(&Accelerator::Rknn {
            core: NpuCoreSpec::Auto
        }));
    }

    #[test]
    fn npu_core_serde_roundtrip() {
        // TOML requires a table at the top level; wrap the enum in a struct.
        #[derive(Serialize, Deserialize, Debug, PartialEq)]
        struct W {
            core: NpuCoreSpec,
        }
        let w = W {
            core: NpuCoreSpec::Cores012,
        };
        let s = toml::to_string(&w).unwrap();
        let parsed: W = toml::from_str(&s).unwrap();
        assert_eq!(w, parsed);
    }
}
