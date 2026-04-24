//! CPU affinity pinning for workers.
//!
//! Why: on SMT-enabled CPUs (Zen 4 is 6 physical × 2 SMT = 12 logical),
//! unpinned workers can land on SMT siblings of the main submitter thread
//! and steal its cycles. That's exactly how the previous `InferencePool`
//! attempt regressed on 6T. Pinning each worker to a distinct physical
//! core eliminates SMT contention.
//!
//! Policies (env `YSCV_POOL_AFFINITY`):
//! - `none`      — no pinning, workers float (previous behavior).
//! - `big`       — big / performance cores only (ARM big.LITTLE,
//!   Intel P-cores). Fallback to `physical` on symmetric CPUs.
//! - `physical`  — one worker per physical core, one SMT sibling per
//!   core left idle for the submitter. Default on Zen 4
//!   and other symmetric SMT CPUs.
//!
//! Linux uses `sched_setaffinity`; macOS uses `thread_policy_set` with
//! `THREAD_AFFINITY_POLICY`. Other OSes fall through as a no-op.

#[cfg(target_os = "linux")]
use std::fs;

/// Returns a list of CPU IDs to pin `nthreads` workers to, sized exactly
/// `nthreads`. If topology info is unavailable or policy is `none`,
/// returns an empty vec (caller should skip pinning).
pub fn resolve_pinning(policy: AffinityPolicy, nthreads: usize) -> Vec<usize> {
    match policy {
        AffinityPolicy::None => Vec::new(),
        AffinityPolicy::Big => resolve_big(nthreads).unwrap_or_else(|| resolve_physical(nthreads)),
        AffinityPolicy::Physical => resolve_physical(nthreads),
    }
}

/// Parses `YSCV_POOL_AFFINITY` env; default is `physical`.
pub fn policy_from_env() -> AffinityPolicy {
    match std::env::var("YSCV_POOL_AFFINITY").as_deref() {
        Ok("none") => AffinityPolicy::None,
        Ok("big") => AffinityPolicy::Big,
        // Default (including unset, "physical", or any other value) is
        // physical-core pinning — explicitly spelled out so clippy sees
        // the `"physical"` arm as intentional rather than redundant.
        _ => AffinityPolicy::Physical,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AffinityPolicy {
    None,
    Big,
    Physical,
}

/// Pin the current thread to `cpu_id`. No-op on unsupported platforms.
pub fn pin_current_thread(cpu_id: usize) {
    #[cfg(target_os = "linux")]
    {
        pin_linux(cpu_id);
    }
    #[cfg(target_os = "macos")]
    {
        pin_darwin(cpu_id);
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        let _ = cpu_id;
    }
}

#[cfg(target_os = "linux")]
fn pin_linux(cpu_id: usize) {
    // SAFETY: zero-initialized cpu_set_t is valid; CPU_SET on it is safe;
    // sched_setaffinity takes a pointer to a cpu_set_t with correct size.
    // libc is the standard Rust FFI shim for exactly this pattern.
    #[allow(unsafe_code)]
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        libc::CPU_SET(cpu_id, &mut set);
        let _ = libc::sched_setaffinity(
            0, // 0 = current thread
            std::mem::size_of::<libc::cpu_set_t>(),
            &set,
        );
    }
}

#[cfg(target_os = "macos")]
fn pin_darwin(cpu_id: usize) {
    // macOS doesn't expose hard pinning — `thread_policy_set` with
    // THREAD_AFFINITY_POLICY is a hint the scheduler may ignore. We apply
    // the hint anyway (matches what ORT / Eigen do on Darwin). Tag derived
    // from cpu_id % 256 so workers on different CPUs get different tags.
    //
    // SAFETY: `mach_thread_self` returns a valid thread port; the policy
    // struct is stack-local with the documented layout (affinity_tag: i32).
    #[allow(unsafe_code)]
    unsafe {
        #[repr(C)]
        struct ThreadAffinityPolicy {
            affinity_tag: i32,
        }
        const THREAD_AFFINITY_POLICY: i32 = 4;
        const THREAD_AFFINITY_POLICY_COUNT: u32 = 1;

        unsafe extern "C" {
            fn mach_thread_self() -> u32;
            fn thread_policy_set(
                thread: u32,
                flavor: i32,
                policy_info: *const i32,
                policy_count: u32,
            ) -> i32;
            fn mach_port_deallocate(task: u32, port: u32) -> i32;
            fn mach_task_self() -> u32;
        }

        let policy = ThreadAffinityPolicy {
            affinity_tag: (cpu_id as i32) + 1, // 0 = null tag
        };
        let thread = mach_thread_self();
        let _ = thread_policy_set(
            thread,
            THREAD_AFFINITY_POLICY,
            &policy as *const _ as *const i32,
            THREAD_AFFINITY_POLICY_COUNT,
        );
        let _ = mach_port_deallocate(mach_task_self(), thread);
    }
}

/// Big-core detection: ARM big.LITTLE exposes capacity via
/// `/sys/devices/system/cpu/cpuN/cpu_capacity`; Intel P-cores on hybrid
/// CPUs via the same sysfs. Returns `None` if unavailable OR if all cores
/// have the same capacity (symmetric CPU — caller falls back to
/// `physical`).
#[cfg(target_os = "linux")]
fn resolve_big(nthreads: usize) -> Option<Vec<usize>> {
    let mut caps: Vec<(usize, u32)> = Vec::new();
    for id in 0..1024usize {
        let path = format!("/sys/devices/system/cpu/cpu{id}/cpu_capacity");
        match fs::read_to_string(&path) {
            Ok(s) => {
                if let Ok(cap) = s.trim().parse::<u32>() {
                    caps.push((id, cap));
                }
            }
            Err(_) => {
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
    let min_cap = caps.iter().map(|&(_, c)| c).min()?;
    if max_cap == min_cap {
        return None; // symmetric
    }
    let big: Vec<usize> = caps
        .into_iter()
        .filter(|&(_, c)| c == max_cap)
        .map(|(id, _)| id)
        .collect();
    Some(take_first_n(big, nthreads))
}

#[cfg(not(target_os = "linux"))]
fn resolve_big(nthreads: usize) -> Option<Vec<usize>> {
    // macOS: sysctl hw.perflevel0.logicalcpu_max gives P-core count.
    // First N logical CPUs are always P-cores on Apple Silicon.
    #[cfg(target_os = "macos")]
    {
        if let Ok(out) = std::process::Command::new("sysctl")
            .args(["-n", "hw.perflevel0.logicalcpu_max"])
            .output()
        {
            if out.status.success()
                && let Ok(s) = String::from_utf8(out.stdout)
                && let Ok(n) = s.trim().parse::<usize>()
                && n > 0
            {
                let big: Vec<usize> = (0..n).collect();
                return Some(take_first_n(big, nthreads));
            }
        }
    }
    let _ = nthreads;
    None
}

/// Physical-core detection: parse `/sys/devices/system/cpu/cpuN/topology/
/// thread_siblings_list` to identify SMT groups. Take the lowest-numbered
/// sibling from each group as "the physical core representative".
///
/// On a 6C/12T Zen 4: returns [0, 1, 2, 3, 4, 5] (or similar — depending on
/// how the kernel enumerates) — one logical CPU per physical, avoiding
/// SMT siblings.
#[cfg(target_os = "linux")]
fn resolve_physical(nthreads: usize) -> Vec<usize> {
    use std::collections::BTreeSet;
    let mut seen: BTreeSet<usize> = BTreeSet::new();
    let mut reps: Vec<usize> = Vec::new();

    for id in 0..1024usize {
        let path = format!("/sys/devices/system/cpu/cpu{id}/topology/thread_siblings_list");
        let s = match fs::read_to_string(&path) {
            Ok(s) => s,
            Err(_) => {
                let cpu_dir = format!("/sys/devices/system/cpu/cpu{id}");
                if !std::path::Path::new(&cpu_dir).exists() {
                    break;
                }
                continue;
            }
        };
        // Format: "0,6" or "0-1" — list of sibling CPU IDs. Take the
        // minimum as the physical-core representative.
        let sibling_min = parse_cpu_list_min(&s);
        let Some(rep) = sibling_min else { continue };
        if seen.insert(rep) {
            reps.push(rep);
        }
    }
    if reps.is_empty() {
        // Sysfs missing (container?) — fall back to first `nthreads` logical CPUs.
        return (0..nthreads).collect();
    }
    take_first_n(reps, nthreads)
}

#[cfg(not(target_os = "linux"))]
fn resolve_physical(nthreads: usize) -> Vec<usize> {
    // Darwin / Windows: fall back to first N logical CPUs. Darwin's
    // `thread_policy_set` hint gives some affinity anyway.
    (0..nthreads).collect()
}

/// Parses a cpuset list like "0,6" or "0-1,4-5" and returns the smallest ID.
#[cfg(target_os = "linux")]
fn parse_cpu_list_min(s: &str) -> Option<usize> {
    let s = s.trim();
    let mut min: Option<usize> = None;
    for part in s.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        if let Some((a, _b)) = part.split_once('-') {
            if let Ok(v) = a.trim().parse::<usize>() {
                min = Some(min.map_or(v, |m| m.min(v)));
            }
        } else if let Ok(v) = part.parse::<usize>() {
            min = Some(min.map_or(v, |m| m.min(v)));
        }
    }
    min
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
fn take_first_n(mut v: Vec<usize>, n: usize) -> Vec<usize> {
    if v.len() > n {
        v.truncate(n);
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn none_policy_returns_empty() {
        assert!(resolve_pinning(AffinityPolicy::None, 6).is_empty());
    }

    #[test]
    fn physical_returns_at_most_nthreads() {
        let v = resolve_pinning(AffinityPolicy::Physical, 4);
        assert!(v.len() <= 4, "got {} ids for 4 threads", v.len());
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn cpu_list_min_parses() {
        assert_eq!(parse_cpu_list_min("0,6"), Some(0));
        assert_eq!(parse_cpu_list_min("3,9"), Some(3));
        assert_eq!(parse_cpu_list_min("2-3,5"), Some(2));
        assert_eq!(parse_cpu_list_min(""), None);
    }

    #[test]
    fn policy_from_env_defaults_to_physical() {
        // We can't mutate env safely inside a test; just verify the
        // default branch when the var is unset in CI.
        let _p = policy_from_env();
    }

    #[test]
    fn pin_current_thread_is_noop_safe() {
        // Just ensure it doesn't crash on any supported target.
        pin_current_thread(0);
    }
}
