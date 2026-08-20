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

#[cfg(target_os = "linux")]
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

/// First value of a `key : value` line, as `/proc/cpuinfo` formats it.
fn field<'a>(info: &'a str, key: &str) -> Option<&'a str> {
    info.lines()
        .find(|l| l.trim_start().starts_with(key))
        .and_then(|l| l.split(':').nth(1))
        .map(str::trim)
}

fn detect_uarch(info: &str) -> Microarch {
    match field(info, "CPU part") {
        Some(part) => u32::from_str_radix(part.trim_start_matches("0x"), 16)
            .map(part_to_uarch)
            .unwrap_or(Microarch::GenericArm),
        None => Microarch::GenericArm,
    }
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
