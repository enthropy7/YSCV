//! 32-bit ARM host detection.
//!
//! `is_arm_feature_detected!` is still unstable, so features come from the same
//! place that macro reads: the `AT_HWCAP` auxiliary vector. The microarchitecture
//! has no such channel on this arch — 32-bit ARM does not expose MIDR the way
//! aarch64 does — so it is read from `/proc/cpuinfo`.

use super::{Cpu, CpuFeatures, Microarch};

pub(super) fn detect() -> Cpu {
    Cpu {
        uarch: detect_uarch(&std::fs::read_to_string("/proc/cpuinfo").unwrap_or_default()),
        features: features_from_hwcap(hwcap()),
    }
}

/// `AT_HWCAP`, and the two capability bits we dispatch on, from the Linux
/// `arch/arm` uapi headers. They are ABI: fixed for the life of the port.
const AT_HWCAP: core::ffi::c_ulong = 16;
const HWCAP_NEON: core::ffi::c_ulong = 1 << 12;
const HWCAP_VFPV4: core::ffi::c_ulong = 1 << 16;

#[cfg(all(target_os = "linux", not(target_env = "uclibc")))]
#[allow(unsafe_code)]
fn hwcap() -> core::ffi::c_ulong {
    unsafe extern "C" {
        fn getauxval(type_: core::ffi::c_ulong) -> core::ffi::c_ulong;
    }
    // SAFETY: getauxval takes an integer and returns one, reads only the
    // process's own auxiliary vector, and answers 0 for a type it does not
    // know. There is no pointer or lifetime involved.
    unsafe { getauxval(AT_HWCAP) }
}

/// uClibc-ng has no `getauxval` (the RV1106 ships 1.0.31), so the same vector
/// is read from `/proc/self/auxv` instead.
#[cfg(all(target_os = "linux", target_env = "uclibc"))]
fn hwcap() -> core::ffi::c_ulong {
    std::fs::read("/proc/self/auxv").map_or(0, |auxv| hwcap_from_auxv(&auxv))
}

/// `AT_HWCAP` from a 32-bit auxiliary vector: native-endian `(type, value)`
/// word pairs, terminated by `AT_NULL`.
#[cfg(any(target_env = "uclibc", test))]
fn hwcap_from_auxv(auxv: &[u8]) -> core::ffi::c_ulong {
    let (words, _) = auxv.as_chunks::<4>();
    let (entries, _) = words.as_chunks::<2>();
    entries
        .iter()
        .map(|[key, value]| (u32::from_ne_bytes(*key), u32::from_ne_bytes(*value)))
        .find(|&(key, _)| key == AT_HWCAP)
        .map_or(0, |(_, value)| value)
}

/// Every other 32-bit ARM target: no auxiliary vector, so no features claimed.
#[cfg(not(target_os = "linux"))]
fn hwcap() -> core::ffi::c_ulong {
    0
}

fn features_from_hwcap(caps: core::ffi::c_ulong) -> CpuFeatures {
    CpuFeatures {
        neon: caps & HWCAP_NEON != 0,
        vfpv4: caps & HWCAP_VFPV4 != 0,
        ..CpuFeatures::default()
    }
}

fn detect_uarch(info: &str) -> Microarch {
    super::cpuinfo_part(info).map_or(Microarch::GenericArm, part_to_uarch)
}

/// Only the parts we have measured on. Everything else stays `GenericArm`,
/// which is feature-correct and simply unspecialised.
fn part_to_uarch(part: u32) -> Microarch {
    match part {
        0xc07 => Microarch::CortexA7,
        _ => Microarch::GenericArm,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const OPI_ZERO: &str = "processor\t: 0\n\
        model name\t: ARMv7 Processor rev 5 (v7l)\n\
        CPU part\t: 0xc07\n";

    #[test]
    fn reads_an_orange_pi_zero() {
        assert_eq!(detect_uarch(OPI_ZERO), Microarch::CortexA7);
    }

    #[test]
    fn reads_hwcap_from_a_raw_auxiliary_vector() {
        let auxv: Vec<u8> = [(6, 4096), (16, (1 << 12) | (1 << 16)), (0, 0)]
            .into_iter()
            .flat_map(|(key, value): (u32, u32)| [key.to_ne_bytes(), value.to_ne_bytes()])
            .flatten()
            .collect();
        let features = features_from_hwcap(hwcap_from_auxv(&auxv));
        assert!(features.neon && features.vfpv4);
        assert_eq!(hwcap_from_auxv(&auxv[..8]), 0, "no AT_HWCAP entry");
        assert_eq!(
            hwcap_from_auxv(&auxv[..12]),
            0,
            "a partial entry is ignored"
        );
    }

    #[test]
    fn unknown_part_is_generic() {
        assert_eq!(detect_uarch("CPU part\t: 0xdead\n"), Microarch::GenericArm);
    }

    #[test]
    fn missing_cpuinfo_does_not_panic() {
        assert_eq!(detect_uarch(""), Microarch::GenericArm);
    }

    #[test]
    fn hwcap_bits_map_to_features() {
        let f = features_from_hwcap(HWCAP_NEON | HWCAP_VFPV4);
        assert!(f.neon && f.vfpv4);
        assert!(!f.sve, "an arm bit must not leak into the aarch64 fields");

        let none = features_from_hwcap(0);
        assert_eq!(none, CpuFeatures::default());

        // A core with NEON but no fused multiply-add is a real configuration.
        assert!(!features_from_hwcap(HWCAP_NEON).vfpv4);
    }
}
