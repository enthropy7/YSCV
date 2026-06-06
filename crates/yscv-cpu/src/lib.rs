//! Host CPU identity — the foundation of runtime kernel dispatch.
//!
//! [`host_cpu`] detects the running core's microarchitecture and feature set
//! once, caches it in a `OnceLock`, and exposes the result as the shared source
//! of truth for yscv crates. Detection never fails: an unrecognised core
//! resolves to a `Generic*` microarch, and missing features read `false`, so
//! callers always have a correct fallback path.
#![deny(unsafe_code)]

use std::sync::OnceLock;

#[cfg(target_arch = "aarch64")]
mod detect_aarch64;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod detect_x86;

/// Coarse CPU microarchitecture. Used to refine kernel selection on top of a
/// feature-correct choice, never as the correctness gate itself.
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
    /// In-order cores where hand-baked instruction scheduling pays most.
    pub fn is_in_order(self) -> bool {
        matches!(self, Microarch::CortexA53 | Microarch::CortexA55)
    }

    /// True for fallback identities with no microarch specialisation.
    pub fn is_generic(self) -> bool {
        matches!(
            self,
            Microarch::GenericAarch64 | Microarch::GenericX86 | Microarch::Scalar
        )
    }
}

/// Runtime CPU feature flags relevant to kernel selection.
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
    pub sse2: bool,
    pub ssse3: bool,
    pub avx: bool,
    pub avx2: bool,
    pub fma: bool,
    pub sse41: bool,
    pub avxvnni: bool,
    pub avx512f: bool,
    pub avx512bw: bool,
    pub avx512vnni: bool,
}

impl CpuFeatures {
    #[inline]
    pub fn x86_avx_fma(self) -> bool {
        self.avx && self.fma
    }

    #[inline]
    pub fn x86_avx2_fma(self) -> bool {
        self.avx2 && self.fma
    }

    #[inline]
    pub fn x86_avx2_sse41(self) -> bool {
        self.avx2 && self.sse41
    }

    #[inline]
    pub fn x86_avx2_ssse3(self) -> bool {
        self.avx2 && self.ssse3
    }

    #[inline]
    pub fn x86_avx_vnni(self) -> bool {
        self.avx2 && self.avxvnni
    }

    #[inline]
    pub fn x86_avx512_bw(self) -> bool {
        self.avx512f && self.avx512bw
    }

    #[inline]
    pub fn x86_avx512_vnni(self) -> bool {
        self.x86_avx512_bw() && self.avx512vnni
    }

    #[inline]
    pub fn aarch64_neon_dotprod(self) -> bool {
        self.neon && self.dotprod
    }

    #[inline]
    pub fn aarch64_neon_i8mm(self) -> bool {
        self.neon && self.i8mm
    }
}

/// The detected host CPU. Single source of truth for dispatch; obtain via
/// [`host_cpu`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Cpu {
    pub uarch: Microarch,
    pub features: CpuFeatures,
}

/// Detects the host CPU once and returns the cached identity. Cheap after the
/// first call.
pub fn host_cpu() -> &'static Cpu {
    static CPU: OnceLock<Cpu> = OnceLock::new();
    CPU.get_or_init(detect_host)
}

fn detect_host() -> Cpu {
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
        assert!(std::ptr::eq(a, b));

        #[cfg(target_arch = "aarch64")]
        {
            assert_ne!(a.uarch, Microarch::Scalar);
            assert!(a.features.neon, "NEON is mandatory on aarch64");
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            assert_ne!(a.uarch, Microarch::Scalar);
        }

        println!("detected host cpu: {a:?}");
    }
}
