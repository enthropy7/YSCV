//! x86 host detection: CPUID vendor + family/model → microarch, std macros →
//! features. The microarch mapping is intentionally coarse for Phase 0 (vendor
//! and family, with a light model split where it matters); it is refined per
//! model in later phases when specialised kernels actually consult it.

use super::{Cpu, CpuFeatures, Microarch};

#[cfg(target_arch = "x86")]
use core::arch::x86::__cpuid;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::__cpuid;

pub(super) fn detect() -> Cpu {
    Cpu {
        uarch: detect_uarch(),
        features: detect_features(),
    }
}

fn detect_features() -> CpuFeatures {
    CpuFeatures {
        sse: std::is_x86_feature_detected!("sse"),
        sse2: std::is_x86_feature_detected!("sse2"),
        avx: std::is_x86_feature_detected!("avx"),
        avx2: std::is_x86_feature_detected!("avx2"),
        fma: std::is_x86_feature_detected!("fma"),
        avx512f: std::is_x86_feature_detected!("avx512f"),
        avx512vnni: std::is_x86_feature_detected!("avx512vnni"),
        ..CpuFeatures::default()
    }
}

fn detect_uarch() -> Microarch {
    let vendor = cpuid_vendor();
    let (family, model) = cpuid_family_model();
    match &vendor {
        b"AuthenticAMD" => amd_uarch(family, model),
        b"GenuineIntel" => intel_uarch(family, model),
        _ => Microarch::GenericX86,
    }
}

/// Vendor string from CPUID leaf 0 (`EBX:EDX:ECX`).
fn cpuid_vendor() -> [u8; 12] {
    let r = __cpuid(0);
    let mut v = [0u8; 12];
    v[0..4].copy_from_slice(&r.ebx.to_le_bytes());
    v[4..8].copy_from_slice(&r.edx.to_le_bytes());
    v[8..12].copy_from_slice(&r.ecx.to_le_bytes());
    v
}

/// Effective family and model from CPUID leaf 1 `EAX`, applying the extended
/// family/model rules.
fn cpuid_family_model() -> (u32, u32) {
    let eax = __cpuid(1).eax;
    let base_family = (eax >> 8) & 0xf;
    let ext_family = (eax >> 20) & 0xff;
    let base_model = (eax >> 4) & 0xf;
    let ext_model = (eax >> 16) & 0xf;
    let family = if base_family == 0xf {
        base_family + ext_family
    } else {
        base_family
    };
    let model = if base_family == 0x6 || base_family == 0xf {
        (ext_model << 4) | base_model
    } else {
        base_model
    };
    (family, model)
}

fn amd_uarch(family: u32, model: u32) -> Microarch {
    match family {
        0x17 => Microarch::Zen2, // Zen / Zen+ / Zen2 — one bucket for now
        0x19 => {
            // Family 0x19 spans Zen3 and Zen4; Zen4 desktop/mobile/server land
            // in the upper model ranges (e.g. Raphael 0x61). Coarse split.
            if model >= 0x60 || (0x10..=0x1f).contains(&model) {
                Microarch::Zen4
            } else {
                Microarch::Zen3
            }
        }
        0x1a => Microarch::Zen5,
        _ => Microarch::GenericX86,
    }
}

fn intel_uarch(family: u32, model: u32) -> Microarch {
    if family != 0x6 {
        return Microarch::GenericX86;
    }
    // Coarse representative buckets by model; refined in later phases.
    match model {
        0x55 => Microarch::IntelSkylake,        // Skylake-SP / Cascade Lake
        0x6a | 0x6c => Microarch::IntelIceLake, // Ice Lake-SP
        0x8f => Microarch::IntelSapphireRapids,
        _ => Microarch::GenericX86,
    }
}
