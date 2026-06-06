// ===========================================================================
// SIMD-accelerated element-wise operations
// ===========================================================================
//
// Sub-modules grouped by operation type:
//   - exp:        fast-exp helpers (SSE/AVX/NEON), exp_slice, sub_exp, tanh
//   - activation: ReLU, Sigmoid, SiLU (dispatch + inplace + all SIMD impls)
//   - reduce:     max_reduce, add_reduce dispatchers + impls
//   - binary:     binary_same_shape_dispatch + impls, mul_scalar_inplace + impls
//   - fma:        fma_slice_dispatch + impls, matmul_row_dispatch + impls
//   - softmax:    row/batched softmax + log-softmax dispatchers + impls

mod activation;
mod binary;
pub(crate) mod exp;
mod fma;
mod reduce;
mod softmax;

use std::fmt;

#[cfg(test)]
mod tests;

/// ISA/backend selected by the single-op CPU dispatch layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SimdDispatchPath {
    Accelerate,
    Mkl,
    Armpl,
    Avx512,
    Avx,
    Sse2,
    Sse,
    Neon,
    Scalar,
}

impl fmt::Display for SimdDispatchPath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            SimdDispatchPath::Accelerate => "accelerate",
            SimdDispatchPath::Mkl => "mkl",
            SimdDispatchPath::Armpl => "armpl",
            SimdDispatchPath::Avx512 => "avx512",
            SimdDispatchPath::Avx => "avx",
            SimdDispatchPath::Sse2 => "sse2",
            SimdDispatchPath::Sse => "sse",
            SimdDispatchPath::Neon => "neon",
            SimdDispatchPath::Scalar => "scalar",
        };
        f.write_str(name)
    }
}

/// Snapshot of the single-op dispatch choices for the current host.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CpuDispatchReport {
    pub cpu: crate::Cpu,
    pub relu: SimdDispatchPath,
    pub sigmoid: SimdDispatchPath,
    pub exp: SimdDispatchPath,
    pub binary: SimdDispatchPath,
    pub fma: SimdDispatchPath,
    pub reduce: SimdDispatchPath,
    pub softmax: SimdDispatchPath,
    pub batch_norm: SimdDispatchPath,
    pub layer_norm: SimdDispatchPath,
}

impl fmt::Display for CpuDispatchReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "cpu={:?}; relu={}; sigmoid={}; exp={}; binary={}; fma={}; reduce={}; softmax={}; batch_norm={}; layer_norm={}",
            self.cpu.uarch,
            self.relu,
            self.sigmoid,
            self.exp,
            self.binary,
            self.fma,
            self.reduce,
            self.softmax,
            self.batch_norm,
            self.layer_norm
        )
    }
}

/// Returns the current host CPU and the ISA/backend paths selected by the
/// standalone compute kernels.
pub fn cpu_dispatch_report() -> CpuDispatchReport {
    CpuDispatchReport {
        cpu: *crate::host_cpu(),
        relu: dispatch_path(!x86_memory_simd_forces_avx2(), false),
        sigmoid: dispatch_path(true, false),
        exp: exp_dispatch_path(),
        binary: binary_dispatch_path(),
        fma: dispatch_path(false, false),
        reduce: dispatch_path(false, false),
        softmax: dispatch_path(true, false),
        batch_norm: dispatch_path(true, true),
        layer_norm: dispatch_path(true, false),
    }
}

pub(crate) fn dispatch_path(_prefer_avx512: bool, _prefer_sse2: bool) -> SimdDispatchPath {
    if cfg!(miri) {
        return SimdDispatchPath::Scalar;
    }

    let features = crate::host_cpu().features;

    #[cfg(target_arch = "x86_64")]
    if _prefer_avx512 && features.avx512f {
        return SimdDispatchPath::Avx512;
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if features.avx {
            return SimdDispatchPath::Avx;
        }
        if _prefer_sse2 && features.sse2 {
            return SimdDispatchPath::Sse2;
        }
        if features.sse {
            return SimdDispatchPath::Sse;
        }
    }
    #[cfg(target_arch = "aarch64")]
    if features.neon {
        return SimdDispatchPath::Neon;
    }

    SimdDispatchPath::Scalar
}

pub(crate) fn exp_dispatch_path() -> SimdDispatchPath {
    if cfg!(miri) {
        return SimdDispatchPath::Scalar;
    }
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        SimdDispatchPath::Accelerate
    }
    #[cfg(all(
        not(all(target_os = "macos", target_arch = "aarch64")),
        feature = "mkl",
        any(target_arch = "x86", target_arch = "x86_64")
    ))]
    {
        SimdDispatchPath::Mkl
    }
    #[cfg(all(
        not(all(target_os = "macos", target_arch = "aarch64")),
        not(all(feature = "mkl", any(target_arch = "x86", target_arch = "x86_64"))),
        feature = "armpl",
        target_arch = "aarch64",
        not(target_os = "macos")
    ))]
    {
        SimdDispatchPath::Armpl
    }
    #[cfg(not(any(
        all(target_os = "macos", target_arch = "aarch64"),
        all(feature = "mkl", any(target_arch = "x86", target_arch = "x86_64")),
        all(feature = "armpl", target_arch = "aarch64", not(target_os = "macos"))
    )))]
    {
        dispatch_path(true, false)
    }
}

pub(crate) fn binary_dispatch_path() -> SimdDispatchPath {
    if cfg!(miri) {
        return SimdDispatchPath::Scalar;
    }
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        SimdDispatchPath::Accelerate
    }
    #[cfg(all(
        not(all(target_os = "macos", target_arch = "aarch64")),
        feature = "mkl",
        any(target_arch = "x86", target_arch = "x86_64")
    ))]
    {
        SimdDispatchPath::Mkl
    }
    #[cfg(all(
        not(all(target_os = "macos", target_arch = "aarch64")),
        not(all(feature = "mkl", any(target_arch = "x86", target_arch = "x86_64"))),
        feature = "armpl",
        target_arch = "aarch64",
        not(target_os = "macos")
    ))]
    {
        SimdDispatchPath::Armpl
    }
    #[cfg(not(any(
        all(target_os = "macos", target_arch = "aarch64"),
        all(feature = "mkl", any(target_arch = "x86", target_arch = "x86_64")),
        all(feature = "armpl", target_arch = "aarch64", not(target_os = "macos"))
    )))]
    {
        dispatch_path(!x86_memory_simd_forces_avx2(), false)
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub(crate) fn x86_memory_simd_forces_avx2() -> bool {
    use std::sync::OnceLock;
    static CACHED: OnceLock<bool> = OnceLock::new();
    *CACHED.get_or_init(|| {
        std::env::var("YSCV_X86_MEMORY_SIMD")
            .map(|value| {
                let value = value.to_ascii_lowercase();
                matches!(value.as_str(), "avx2" | "ymm" | "256")
            })
            .unwrap_or(false)
    })
}

#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
pub(crate) fn x86_memory_simd_forces_avx2() -> bool {
    false
}

// Re-export everything so callers see a flat namespace (same as before the split).

// From activation
pub use activation::{
    bias_add_nhwc_dispatch, bias_relu_nhwc_dispatch, bias_silu_nhwc_dispatch,
    fused_row_epilogue_dispatch, gelu_sigmoid_slice_dispatch, relu_slice_dispatch,
    relu_to_slice_dispatch, sigmoid_slice_dispatch, silu_inplace, silu_slice_dispatch,
};
#[allow(unused_imports)]
pub(crate) use activation::{sigmoid_scalar, sigmoid_slice};

// From exp
pub use exp::{exp_slice_dispatch, sub_exp_slice_dispatch, tanh_slice_dispatch};

// From reduce
pub use reduce::{add_reduce_dispatch, max_reduce_dispatch};

// From binary
pub(crate) use binary::binary_broadcast_lastdim_dispatch;
#[allow(unused_imports)]
pub use binary::{
    add_inplace_dispatch, add_relu_inplace_dispatch, binary_same_shape_dispatch,
    mul_scalar_inplace_dispatch,
};

// From fma
pub use fma::{fma_slice_dispatch, matmul_row_dispatch, matmul_row_set_dispatch};

// From softmax
pub use softmax::{log_softmax_rows_fused_dispatch, softmax_rows_fused_dispatch};
