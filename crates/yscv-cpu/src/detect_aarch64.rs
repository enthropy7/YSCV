//! aarch64 host detection: MIDR to microarch, std macros to runtime features.

use super::{Cpu, CpuFeatures, Microarch};

pub(super) fn detect() -> Cpu {
    Cpu {
        uarch: detect_uarch(),
        features: detect_features(),
    }
}

fn detect_features() -> CpuFeatures {
    CpuFeatures {
        neon: std::arch::is_aarch64_feature_detected!("neon"),
        dotprod: std::arch::is_aarch64_feature_detected!("dotprod"),
        i8mm: std::arch::is_aarch64_feature_detected!("i8mm"),
        fp16: std::arch::is_aarch64_feature_detected!("fp16"),
        sve: std::arch::is_aarch64_feature_detected!("sve"),
        ..CpuFeatures::default()
    }
}

fn detect_uarch() -> Microarch {
    #[cfg(target_os = "macos")]
    {
        Microarch::AppleSilicon
    }
    #[cfg(not(target_os = "macos"))]
    {
        match read_midr_part() {
            Some(part) => part_to_uarch(part),
            None => Microarch::GenericAarch64,
        }
    }
}

#[cfg(not(target_os = "macos"))]
fn read_midr_part() -> Option<u32> {
    if let Ok(s) =
        std::fs::read_to_string("/sys/devices/system/cpu/cpu0/regs/identification/midr_el1")
        && let Some(midr) = parse_hex_u64(s.trim())
    {
        return Some(((midr >> 4) & 0xfff) as u32);
    }
    if let Ok(info) = std::fs::read_to_string("/proc/cpuinfo") {
        for line in info.lines() {
            if let Some(rest) = line.strip_prefix("CPU part")
                && let Some(idx) = rest.find("0x")
                && let Some(part) = parse_hex_u64(rest[idx..].trim())
            {
                return Some(part as u32);
            }
        }
    }
    None
}

#[cfg(not(target_os = "macos"))]
fn parse_hex_u64(s: &str) -> Option<u64> {
    let s = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X"))?;
    let hex: String = s.chars().take_while(|c| c.is_ascii_hexdigit()).collect();
    u64::from_str_radix(&hex, 16).ok()
}

#[cfg(not(target_os = "macos"))]
fn part_to_uarch(part: u32) -> Microarch {
    match part {
        0xd03 => Microarch::CortexA53,
        0xd05 => Microarch::CortexA55,
        0xd08 => Microarch::CortexA72,
        0xd09 => Microarch::CortexA73,
        0xd0b => Microarch::CortexA76,
        0xd41 | 0xd42 => Microarch::CortexA78,
        0xd0c => Microarch::NeoverseN1,
        _ => Microarch::GenericAarch64,
    }
}
