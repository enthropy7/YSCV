//! Host CPU identity — the foundation of microarchitecture-aware kernel
//! dispatch (see `docs/microarch-dispatch.md`).
//!
//! [`host_cpu`] detects the running core's microarchitecture and feature set
//! **once** (cached in a `OnceLock`) and is the single source of truth that
//! per-op selection tables consult. Detection never fails: an unrecognised core
//! resolves to a `Generic*` microarch, and missing features simply read
//! `false`, so the framework always lands on a correct (if not perfectly
//! scheduled) path.
//!
//! This is Phase 0 of the dispatch roadmap: identity only, no kernel selection
//! yet. The `select_<op>` tables that consume [`Cpu`] arrive in later phases.

use std::sync::OnceLock;

#[cfg(target_arch = "aarch64")]
mod detect_aarch64;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod detect_x86;

/// Coarse CPU microarchitecture. Used to refine kernel selection *on top of* a
/// feature-correct choice — never for correctness. `Generic*` / `Scalar` are
/// first-class values, not errors: an unknown core gets a feature-appropriate
/// generic kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Microarch {
    // aarch64 — in-order
    CortexA53,
    CortexA55,
    // aarch64 — out-of-order
    CortexA72,
    CortexA73,
    CortexA76,
    CortexA78,
    NeoverseN1,
    AppleSilicon,
    // x86_64
    Zen2,
    Zen3,
    Zen4,
    Zen5,
    IntelSkylake,
    IntelIceLake,
    IntelSapphireRapids,
    // fallbacks — always correct, just not specialised
    GenericAarch64,
    GenericX86,
    Scalar,
}

impl Microarch {
    /// In-order cores (no hardware reordering) — where hand-baked instruction
    /// scheduling pays the most, because the compiler can't recover it.
    pub fn is_in_order(self) -> bool {
        matches!(self, Microarch::CortexA53 | Microarch::CortexA55)
    }

    /// True for the `Generic*` / `Scalar` fallbacks (no microarch specialisation
    /// available — selection should lean on feature flags alone).
    pub fn is_generic(self) -> bool {
        matches!(
            self,
            Microarch::GenericAarch64 | Microarch::GenericX86 | Microarch::Scalar
        )
    }
}

/// Runtime CPU feature flags relevant to kernel selection. A plain bool struct
/// (no external bitflags dep); detected via the std `is_*_feature_detected!`
/// macros, which read HWCAP on aarch64 and CPUID on x86 under the hood.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CpuFeatures {
    // aarch64
    pub neon: bool,
    pub dotprod: bool,
    pub i8mm: bool,
    pub fp16: bool,
    pub sve: bool,
    // x86_64
    pub sse: bool,
    pub avx: bool,
    pub avx2: bool,
    pub fma: bool,
    pub avx512f: bool,
    pub avx512vnni: bool,
}

/// The detected host CPU. Single source of truth for dispatch; obtain via
/// [`host_cpu`].
#[derive(Clone, Copy, Debug)]
pub struct Cpu {
    pub uarch: Microarch,
    pub features: CpuFeatures,
}

/// Detects the host CPU once and returns the cached identity. Cheap after the
/// first call (an atomic load).
pub fn host_cpu() -> &'static Cpu {
    static CPU: OnceLock<Cpu> = OnceLock::new();
    CPU.get_or_init(detect_host)
}

fn detect_host() -> Cpu {
    // Under Miri, CPUID / sysfs / HWCAP probing isn't modelled — return the
    // scalar identity so dispatch (which already gates SIMD behind `!cfg!(miri)`)
    // takes the scalar path without touching unsupported intrinsics.
    if cfg!(miri) {
        return Cpu {
            uarch: Microarch::Scalar,
            features: CpuFeatures::default(),
        };
    }
    #[cfg(target_arch = "aarch64")]
    {
        detect_aarch64::detect()
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        detect_x86::detect()
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86", target_arch = "x86_64")))]
    {
        Cpu {
            uarch: Microarch::Scalar,
            features: CpuFeatures::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_cpu_detects_without_panic_and_is_cached() {
        let a = host_cpu();
        let b = host_cpu();
        // Same cached instance.
        assert!(std::ptr::eq(a, b));

        // On the build arch the detector must land on a non-Scalar microarch
        // and report the ISA's baseline feature.
        #[cfg(target_arch = "aarch64")]
        {
            assert_ne!(a.uarch, Microarch::Scalar);
            assert!(a.features.neon, "NEON is mandatory on aarch64");
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            assert_ne!(a.uarch, Microarch::Scalar);
            // No hard feature assert — minimal x86_64 baseline is SSE2 only.
        }

        // Visible under `cargo test -- --nocapture` to eyeball detection per box.
        println!("detected host cpu: {a:?}");
    }
}
