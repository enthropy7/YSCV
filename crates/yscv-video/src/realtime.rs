//! POSIX real-time scheduling helpers for the FPV pipeline.
//!
//! These functions configure thread scheduler policy, CPU affinity, and
//! page-locked memory — the three knobs needed to bring jitter under
//! ±1ms on Linux. All operations no-op gracefully on non-Linux hosts and
//! when `CAP_SYS_NICE` is missing (logs a warning instead of failing).
//!
//! # Why this matters for FPV
//!
//! Default `SCHED_OTHER` schedules pipeline threads against every other
//! userland process. On a busy SoC (browser, daemon, etc.) the inference
//! thread may be preempted for 5–10ms — directly visible as video stutter
//! at 30fps. `SCHED_FIFO` with priority ≥50 ensures pipeline threads
//! preempt CFS workloads. CPU affinity pins each stage to a known core
//! (A76 for capture/NPU dispatch, A55 for encode on RK3588) preventing
//! scheduler migration jitter.

#![cfg_attr(not(target_os = "linux"), allow(dead_code, unused_variables))]

/// Pseudo-cap on SCHED_FIFO priority. POSIX allows 1..=99 but most
/// distributions block >50 without explicit sysctl. We clamp here.
pub const RT_PRIO_MAX: u8 = 90;

/// Errors from real-time configuration. Most are non-fatal — caller can
/// log and continue with degraded scheduling.
#[derive(Debug)]
pub enum RtError {
    /// `sched_setscheduler` returned EPERM — process lacks `CAP_SYS_NICE`.
    NoCapability,
    /// `sched_setaffinity` returned EINVAL — invalid CPU mask.
    InvalidAffinity,
    /// `mlockall` returned ENOMEM — RLIMIT_MEMLOCK too low.
    MemlockExceeded,
    /// Other syscall failure (errno).
    Errno(i32),
    /// Operation requested but no implementation available on this OS.
    NotSupported,
}

impl std::fmt::Display for RtError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RtError::NoCapability => write!(
                f,
                "no CAP_SYS_NICE — run as root or grant via `setcap cap_sys_nice+ep`"
            ),
            RtError::InvalidAffinity => write!(f, "invalid CPU mask for sched_setaffinity"),
            RtError::MemlockExceeded => write!(
                f,
                "mlockall failed — increase RLIMIT_MEMLOCK (`ulimit -l unlimited` or systemd LimitMEMLOCK=infinity)"
            ),
            RtError::Errno(e) => write!(f, "errno {e}"),
            RtError::NotSupported => write!(f, "not supported on this platform"),
        }
    }
}

impl std::error::Error for RtError {}

/// Pin the *current* thread to `SCHED_FIFO` with the given priority.
///
/// `prio` is clamped to `[1, RT_PRIO_MAX]`. Returns `Err(RtError::NoCapability)`
/// if `CAP_SYS_NICE` is absent — caller may choose to log and continue.
pub fn set_sched_fifo(prio: u8) -> Result<(), RtError> {
    #[cfg(target_os = "linux")]
    {
        let prio = prio.clamp(1, RT_PRIO_MAX) as i32;
        // SAFETY: setting policy on PID 0 = current thread; param is a
        // valid stack pointer; SCHED_FIFO is documented as policy 1.
        let mut param = libc_sched_param {
            sched_priority: prio,
        };
        let ret = unsafe { sched_setscheduler(0, SCHED_FIFO, &mut param) };
        if ret == 0 {
            return Ok(());
        }
        match errno_now() {
            1 /* EPERM */ => Err(RtError::NoCapability),
            other => Err(RtError::Errno(other)),
        }
    }
    #[cfg(not(target_os = "linux"))]
    Err(RtError::NotSupported)
}

/// Pin the current thread to a specific set of CPU cores.
///
/// On RK3588: cores 0–3 are A55 (efficiency), 4–7 are A76 (performance).
/// Pin capture/dispatch (latency-sensitive) to A76, encode to A55.
pub fn set_cpu_affinity(cpus: &[u32]) -> Result<(), RtError> {
    #[cfg(target_os = "linux")]
    {
        if cpus.is_empty() || cpus.iter().any(|&c| c >= 64) {
            return Err(RtError::InvalidAffinity);
        }
        let mut set = libc_cpu_set::default();
        for &c in cpus {
            set.bits[c as usize / 64] |= 1u64 << (c % 64);
        }
        // SAFETY: PID 0 = current thread; setsize matches our struct size.
        let ret = unsafe {
            sched_setaffinity(
                0,
                std::mem::size_of::<libc_cpu_set>(),
                &set as *const libc_cpu_set,
            )
        };
        if ret == 0 {
            Ok(())
        } else {
            Err(RtError::Errno(errno_now()))
        }
    }
    #[cfg(not(target_os = "linux"))]
    Err(RtError::NotSupported)
}

/// Lock the entire process address space into RAM, preventing swap-out
/// of pipeline buffers. Crucial when running alongside heavy workloads
/// — without this a 100 MB swap-out can cause 50ms+ glitches.
pub fn mlockall_current_and_future() -> Result<(), RtError> {
    #[cfg(target_os = "linux")]
    {
        // SAFETY: MCL_CURRENT | MCL_FUTURE is a documented bitmask.
        let ret = unsafe { mlockall(MCL_CURRENT | MCL_FUTURE) };
        if ret == 0 {
            Ok(())
        } else {
            match errno_now() {
                12 /* ENOMEM */ => Err(RtError::MemlockExceeded),
                1 /* EPERM */ => Err(RtError::NoCapability),
                other => Err(RtError::Errno(other)),
            }
        }
    }
    #[cfg(not(target_os = "linux"))]
    Err(RtError::NotSupported)
}

/// Best-effort write to the cpufreq governor sysfs nodes.
///
/// Iterates `/sys/devices/system/cpu/cpu*/cpufreq/scaling_governor` and
/// writes `governor` to every entry that exists. The most common
/// caller value is `"performance"` — pins every core at max
/// frequency, killing CPU-side jitter from DVFS step-up latency. On
/// RK3588 this is ~5 ms saved on the first inference of every burst.
///
/// Returns `Ok(n_cores_set)` on success, where `n_cores_set` is the
/// number of CPUs the write succeeded for (0 means "no sysfs writes
/// took, governor unchanged"). Errors only on unrecoverable I/O
/// (e.g. `/sys` not mounted); per-core write failures (the typical
/// "no CAP_SYS_ADMIN") are logged + counted but NOT fatal — matches
/// the rest of `realtime`'s graceful-fallback contract.
///
/// On non-Linux hosts returns `Ok(0)`.
pub fn set_cpu_governor(governor: &str) -> Result<usize, RtError> {
    #[cfg(target_os = "linux")]
    {
        use std::fs;
        let sys_root = std::path::Path::new("/sys/devices/system/cpu");
        let entries = match fs::read_dir(sys_root) {
            Ok(e) => e,
            Err(e) => {
                return Err(RtError::Errno(e.raw_os_error().unwrap_or(0)));
            }
        };
        let mut applied = 0usize;
        for entry in entries.flatten() {
            let name = entry.file_name();
            let Some(name_str) = name.to_str() else { continue };
            // Match `cpuN` directories — skip `cpufreq`, `cpuidle`, etc.
            if !name_str.starts_with("cpu")
                || !name_str
                    .trim_start_matches("cpu")
                    .chars()
                    .all(|c| c.is_ascii_digit())
            {
                continue;
            }
            let gov_path = entry.path().join("cpufreq/scaling_governor");
            if !gov_path.exists() {
                continue;
            }
            match fs::write(&gov_path, governor) {
                Ok(()) => applied += 1,
                Err(e) => eprintln!(
                    "[yscv-video] governor write to {gov_path:?} failed: {e} \
                     (need CAP_SYS_ADMIN or root)"
                ),
            }
        }
        Ok(applied)
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = governor;
        Ok(0)
    }
}

/// Convenience: try sched_fifo + affinity + mlockall + cpu governor
/// and log each failure as a warning. Returns the first successfully-
/// applied state — caller knows whether the pipeline is running with
/// full RT guarantees.
#[derive(Debug, Clone, Copy, Default)]
pub struct RtAppliedState {
    pub sched_fifo: bool,
    pub affinity: bool,
    pub mlockall: bool,
    /// Number of CPU cores whose `scaling_governor` we successfully
    /// wrote (typically to `"performance"`). Zero means "left at the
    /// system default" — usually `ondemand` or `schedutil`, both of
    /// which add 5–30 ms first-burst latency on edge SoCs.
    pub cpu_governor_cores: usize,
}

pub fn apply_rt_config(prio: u8, cpus: &[u32], lock_mem: bool) -> RtAppliedState {
    apply_rt_config_with_governor(prio, cpus, lock_mem, None)
}

/// Like [`apply_rt_config`] but also writes `governor` to every
/// `cpufreq/scaling_governor` sysfs entry. Pass `None` to leave the
/// governor untouched. The typical FPV value is `Some("performance")`.
pub fn apply_rt_config_with_governor(
    prio: u8,
    cpus: &[u32],
    lock_mem: bool,
    governor: Option<&str>,
) -> RtAppliedState {
    let mut applied = RtAppliedState::default();
    match set_sched_fifo(prio) {
        Ok(()) => applied.sched_fifo = true,
        Err(e) => eprintln!("[yscv-video] sched_fifo({prio}) failed: {e}"),
    }
    if !cpus.is_empty() {
        match set_cpu_affinity(cpus) {
            Ok(()) => applied.affinity = true,
            Err(e) => eprintln!("[yscv-video] affinity({cpus:?}) failed: {e}"),
        }
    }
    if lock_mem {
        match mlockall_current_and_future() {
            Ok(()) => applied.mlockall = true,
            Err(e) => eprintln!("[yscv-video] mlockall failed: {e}"),
        }
    }
    if let Some(gov) = governor {
        match set_cpu_governor(gov) {
            Ok(n) => applied.cpu_governor_cores = n,
            Err(e) => eprintln!("[yscv-video] set_cpu_governor({gov}) failed: {e}"),
        }
    }
    applied
}

// ── Linux-only FFI plumbing ───────────────────────────────────────────

#[cfg(target_os = "linux")]
const SCHED_FIFO: i32 = 1;
#[cfg(target_os = "linux")]
const MCL_CURRENT: i32 = 1;
#[cfg(target_os = "linux")]
const MCL_FUTURE: i32 = 2;

#[cfg(target_os = "linux")]
#[repr(C)]
struct libc_sched_param {
    sched_priority: i32,
}

/// Up to 64×64 = 4096 CPUs (more than any drone SoC). Pre-allocated to
/// avoid heap inside the call.
#[cfg(target_os = "linux")]
#[repr(C)]
struct libc_cpu_set {
    bits: [u64; 64],
}

#[cfg(target_os = "linux")]
impl Default for libc_cpu_set {
    fn default() -> Self {
        Self { bits: [0; 64] }
    }
}

#[cfg(target_os = "linux")]
unsafe extern "C" {
    fn sched_setscheduler(pid: i32, policy: i32, param: *mut libc_sched_param) -> i32;
    fn sched_setaffinity(pid: i32, cpusetsize: usize, mask: *const libc_cpu_set) -> i32;
    fn mlockall(flags: i32) -> i32;
    /// glibc-specific accessor for thread-local errno. `__errno_location` returns
    /// a pointer because errno may live in TLS or per-thread memory.
    fn __errno_location() -> *mut i32;
}

#[cfg(target_os = "linux")]
fn errno_now() -> i32 {
    // SAFETY: __errno_location() returns a valid pointer into TLS for
    // the duration of the calling thread.
    unsafe { *__errno_location() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rt_error_display() {
        assert!(RtError::NoCapability.to_string().contains("CAP_SYS_NICE"));
        assert!(
            RtError::MemlockExceeded
                .to_string()
                .contains("RLIMIT_MEMLOCK")
        );
    }

    #[cfg(not(target_os = "linux"))]
    #[test]
    fn ops_return_not_supported_off_linux() {
        assert!(matches!(set_sched_fifo(50), Err(RtError::NotSupported)));
        assert!(matches!(set_cpu_affinity(&[0]), Err(RtError::NotSupported)));
        assert!(matches!(
            mlockall_current_and_future(),
            Err(RtError::NotSupported)
        ));
    }

    #[test]
    fn apply_rt_config_logs_and_continues_off_linux() {
        // On macOS this should attempt all three, fail all, and return all-false.
        // (The eprintln warnings are visible in test output but don't fail the test.)
        let st = apply_rt_config(60, &[0, 1], true);
        #[cfg(not(target_os = "linux"))]
        {
            assert!(!st.sched_fifo);
            assert!(!st.affinity);
            assert!(!st.mlockall);
        }
        #[cfg(target_os = "linux")]
        {
            // On Linux without CAP_SYS_NICE most fields will be false; we don't
            // assert specific values because CI may have varying perms.
            let _ = st;
        }
    }
}
