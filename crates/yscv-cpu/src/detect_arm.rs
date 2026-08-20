//! 32-bit ARM host detection: `/proc/cpuinfo` for both microarch and features.
//!
//! `is_arm_feature_detected!` is still unstable, so the probe reads the kernel's
//! own view of the core — the same data `getauxval(AT_HWCAP)` is built from,
//! without the unsafe FFI this crate forbids.

use super::{Cpu, CpuFeatures, Microarch};

pub(super) fn detect() -> Cpu {
    let info = std::fs::read_to_string("/proc/cpuinfo").unwrap_or_default();
    Cpu {
        uarch: detect_uarch(&info),
        features: detect_features(&info),
    }
}

/// First value of a `key : value` line, as `/proc/cpuinfo` formats it.
fn field<'a>(info: &'a str, key: &str) -> Option<&'a str> {
    info.lines()
        .find(|l| l.trim_start().starts_with(key))
        .and_then(|l| l.split(':').nth(1))
        .map(str::trim)
}

fn detect_features(info: &str) -> CpuFeatures {
    let flags = field(info, "Features").unwrap_or_default();
    let has = |name: &str| flags.split_whitespace().any(|f| f == name);
    CpuFeatures {
        neon: has("neon"),
        vfpv4: has("vfpv4"),
        ..CpuFeatures::default()
    }
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
        BogoMIPS\t: 129.60\n\
        Features\t: half thumb fastmult vfp edsp neon vfpv3 tls vfpv4 idiva idivt vfpd32 lpae evtstrm\n\
        CPU part\t: 0xc07\n";

    #[test]
    fn reads_an_orange_pi_zero() {
        assert_eq!(detect_uarch(OPI_ZERO), Microarch::CortexA7);
        let f = detect_features(OPI_ZERO);
        assert!(f.neon, "the board reports neon");
        assert!(f.vfpv4, "the board reports vfpv4");
        assert!(!f.sve, "an arm flag must not leak into the aarch64 fields");
    }

    #[test]
    fn unknown_part_is_generic_but_still_feature_correct() {
        let odd = "Features\t: half thumb vfp\nCPU part\t: 0xdead\n";
        assert_eq!(detect_uarch(odd), Microarch::GenericArm);
        assert!(!detect_features(odd).neon);
    }

    #[test]
    fn missing_cpuinfo_does_not_panic() {
        assert_eq!(detect_uarch(""), Microarch::GenericArm);
        assert_eq!(detect_features(""), CpuFeatures::default());
    }
}
