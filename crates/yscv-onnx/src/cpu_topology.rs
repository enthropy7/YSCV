//! CPU topology detection for big.LITTLE and hybrid CPUs.
//!
//! On heterogeneous CPUs (ARM big.LITTLE — RK3588 4×A76+4×A55, Apple M P/E-cores,
//! Intel Alder/Raptor Lake hybrid), the "small" cluster cores run at 30–40% of
//! the "big" cluster's throughput. Spreading a parallel workload across all
//! cores creates stragglers — the slowest core gates the whole join.
//!
//! This module detects the big/performance core set per-OS and exposes
//! [`detect_big_cores`] so the OnnxRunner can cap its thread pool to big cores
//! only. On symmetric CPUs (Zen 4, older Intel/AMD, unknown hosts), all online
//! cores are returned — behavior identical to not using this module.
//!
//! Currently informational only (no sched_setaffinity). Adding pinning is
//! one follow-up step away; the bigger-bang change is just not scheduling
//! rayon workers onto LITTLE cores in the first place.

/// Returns the list of logical CPU IDs classified as "big" / performance cores.
///
/// - **Linux:** parses `/sys/devices/system/cpu/cpuN/cpu_capacity` and returns
///   all CPUs whose capacity equals the maximum. DMIPS-based capacity values
///   are populated by the kernel on ARM big.LITTLE (big cores = 1024, LITTLE
///   cores ≈ 400–500) and on Intel hybrid CPUs when `itmt` is enabled.
/// - **macOS:** uses `sysctl hw.perflevel0.logicalcpu_max` (count of P-cores,
///   which are always the first N logical CPUs on Apple Silicon).
/// - **Other / unreadable:** returns all online CPUs in `0..rayon_threads`
///   (fallback — same behavior as pre-topology code).
pub fn detect_big_cores() -> Vec<usize> {
    #[cfg(target_os = "linux")]
    {
        if let Some(big) = detect_linux_big_cores() {
            return big;
        }
    }
    #[cfg(target_os = "macos")]
    {
        if let Some(big) = detect_darwin_perf_cores() {
            return big;
        }
    }
    fallback_all_cpus()
}

/// Returns the number of big / performance cores. Convenience wrapper for
/// sizing thread pools without allocating the full ID list.
pub fn big_cores_count() -> usize {
    detect_big_cores().len()
}

/// Returns the number of physical CPU cores when detectable.
///
/// Used to avoid explicit SMT oversubscription (e.g. requesting 12 threads on
/// a 6C/12T CPU) in latency-sensitive inference paths. Falls back to rayon's
/// current worker count when topology data is unavailable.
pub fn physical_cores_count() -> usize {
    #[cfg(target_os = "linux")]
    {
        if let Some(n) = detect_linux_physical_cores() {
            return n.max(1);
        }
    }
    #[cfg(target_os = "macos")]
    {
        if let Some(n) = detect_darwin_physical_cores() {
            return n.max(1);
        }
    }
    rayon::current_num_threads().max(1)
}

#[cfg(target_os = "linux")]
fn detect_linux_big_cores() -> Option<Vec<usize>> {
    // Iterate cpuN directories. Stop at the first missing `cpu_capacity` file
    // (end of enumeration). Any CPU that exists but has no capacity file is
    // skipped; in practice every online CPU on ARM / Intel hybrid has one.
    let mut caps: Vec<(usize, u32)> = Vec::new();
    for id in 0..1024usize {
        let path = format!("/sys/devices/system/cpu/cpu{id}/cpu_capacity");
        match std::fs::read_to_string(&path) {
            Ok(s) => {
                if let Ok(cap) = s.trim().parse::<u32>() {
                    caps.push((id, cap));
                }
            }
            Err(_) => {
                // Check if the cpu dir exists at all — if not, we're past the
                // last CPU. If the cpu dir exists but capacity is missing,
                // continue to next id (capacity not populated on this kernel).
                let cpu_dir = format!("/sys/devices/system/cpu/cpu{id}");
                if !std::path::Path::new(&cpu_dir).exists() {
                    break;
                }
            }
        }
    }
    if caps.is_empty() {
        return None;
    }
    let max_cap = caps.iter().map(|&(_, c)| c).max()?;
    Some(
        caps.into_iter()
            .filter(|&(_, c)| c == max_cap)
            .map(|(id, _)| id)
            .collect(),
    )
}

#[cfg(target_os = "linux")]
fn detect_linux_physical_cores() -> Option<usize> {
    use std::collections::HashSet;
    let mut cores: HashSet<(usize, usize)> = HashSet::new();
    for id in 0..1024usize {
        let cpu_dir = format!("/sys/devices/system/cpu/cpu{id}");
        if !std::path::Path::new(&cpu_dir).exists() {
            break;
        }
        let core_id = std::fs::read_to_string(format!("{cpu_dir}/topology/core_id"))
            .ok()
            .and_then(|s| s.trim().parse::<usize>().ok());
        let pkg_id = std::fs::read_to_string(format!("{cpu_dir}/topology/physical_package_id"))
            .ok()
            .and_then(|s| s.trim().parse::<usize>().ok())
            .unwrap_or(0);
        if let Some(core_id) = core_id {
            cores.insert((pkg_id, core_id));
        }
    }
    if cores.is_empty() {
        None
    } else {
        Some(cores.len())
    }
}

#[cfg(target_os = "macos")]
fn detect_darwin_perf_cores() -> Option<Vec<usize>> {
    // Apple Silicon exposes `hw.perflevel0.logicalcpu_max` = number of P-cores.
    // The first N logical CPUs (0..N) are always the P-cores on all current
    // Apple SoCs (M1/M1 Pro/Max/Ultra, M2 family, M3 family, M4 family).
    // We invoke `sysctl` via `std::process::Command` to avoid a libc FFI dep.
    let out = std::process::Command::new("sysctl")
        .args(["-n", "hw.perflevel0.logicalcpu_max"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8(out.stdout).ok()?;
    let n: usize = s.trim().parse().ok()?;
    if n == 0 {
        return None;
    }
    Some((0..n).collect())
}

#[cfg(target_os = "macos")]
fn detect_darwin_physical_cores() -> Option<usize> {
    let out = std::process::Command::new("sysctl")
        .args(["-n", "hw.physicalcpu_max"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8(out.stdout).ok()?;
    let n: usize = s.trim().parse().ok()?;
    if n == 0 { None } else { Some(n) }
}

fn fallback_all_cpus() -> Vec<usize> {
    // No topology info available — use whatever rayon sees. On a symmetric
    // CPU this is the correct answer (all cores are "big"). On an ARM
    // big.LITTLE system missing /sys/devices/system/cpu/cpuN/cpu_capacity,
    // this falls back to "use everything" which matches prior behavior.
    let n = rayon::current_num_threads();
    (0..n).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_returns_at_least_one_core() {
        let big = detect_big_cores();
        assert!(!big.is_empty(), "should detect at least one big core");
    }

    #[test]
    fn big_core_ids_are_distinct() {
        let big = detect_big_cores();
        let mut sorted = big.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(big.len(), sorted.len(), "duplicate core ids");
    }

    #[test]
    fn big_cores_count_matches_vec() {
        assert_eq!(big_cores_count(), detect_big_cores().len());
    }

    #[test]
    fn physical_cores_count_is_non_zero() {
        assert!(physical_cores_count() >= 1);
    }
}
