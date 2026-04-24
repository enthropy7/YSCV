fn main() {
    assemble_sgemm_asm();

    // Link MetalPerformanceShaders.framework when metal-backend is enabled (macOS).
    #[cfg(feature = "metal-backend")]
    {
        if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
            println!("cargo:rustc-link-lib=framework=MetalPerformanceShaders");
            println!("cargo:rustc-link-lib=framework=MetalPerformanceShadersGraph");
        }
    }

    // Link Intel MKL VML when the "mkl" feature is enabled (x86/x86_64 only).
    #[cfg(feature = "mkl")]
    {
        // Try MKLROOT environment variable for custom install paths.
        if let Ok(mkl_root) = std::env::var("MKLROOT") {
            println!("cargo:rustc-link-search=native={mkl_root}/lib/intel64");
        }
        // Link MKL libraries (sequential, single-threaded for determinism).
        println!("cargo:rustc-link-lib=mkl_intel_lp64");
        println!("cargo:rustc-link-lib=mkl_sequential");
        println!("cargo:rustc-link-lib=mkl_core");
    }

    // Link ARM Performance Libraries when the "armpl" feature is enabled (aarch64 Linux only).
    #[cfg(feature = "armpl")]
    {
        if std::env::var("CARGO_CFG_TARGET_ARCH").as_deref() == Ok("aarch64")
            && std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("macos")
        {
            if let Ok(armpl_dir) = std::env::var("ARMPL_DIR") {
                println!("cargo:rustc-link-search=native={armpl_dir}/lib");
            }
            println!("cargo:rustc-link-lib=armpl_lp64");
        }
    }

    // Link BLAS when the "blas" feature is enabled.
    #[cfg(feature = "blas")]
    {
        // macOS: Accelerate.framework ships with the OS — zero setup.
        if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
            println!("cargo:rustc-link-lib=framework=Accelerate");
        }
        // Linux: link OpenBLAS (apt install libopenblas-dev / yum install openblas-devel).
        // Also try to find it via pkg-config for non-standard install paths.
        else if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("linux") {
            // Try pkg-config first for proper link flags
            if let Ok(out) = std::process::Command::new("pkg-config")
                .args(["--libs", "openblas"])
                .output()
            {
                if out.status.success() {
                    let flags = String::from_utf8_lossy(&out.stdout);
                    for flag in flags.split_whitespace() {
                        if let Some(lib) = flag.strip_prefix("-l") {
                            println!("cargo:rustc-link-lib={lib}");
                        } else if let Some(path) = flag.strip_prefix("-L") {
                            println!("cargo:rustc-link-search=native={path}");
                        }
                    }
                } else {
                    // Fallback: assume system-installed openblas
                    println!("cargo:rustc-link-lib=openblas");
                }
            } else {
                println!("cargo:rustc-link-lib=openblas");
            }
        }
        // Windows: link OpenBLAS (install via vcpkg / conda / manual).
        // vcpkg: vcpkg install openblas:x64-windows
        // conda: conda install -c conda-forge openblas
        else if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows") {
            // Try VCPKG_ROOT first
            if let Ok(vcpkg_root) = std::env::var("VCPKG_ROOT") {
                let triplet = if std::env::var("CARGO_CFG_TARGET_ARCH").as_deref() == Ok("x86_64") {
                    "x64-windows"
                } else {
                    "x86-windows"
                };
                println!(
                    "cargo:rustc-link-search=native={}/installed/{}/lib",
                    vcpkg_root, triplet
                );
            }
            // Also check OPENBLAS_PATH environment variable
            if let Ok(blas_path) = std::env::var("OPENBLAS_PATH") {
                println!("cargo:rustc-link-search=native={blas_path}");
            }
            // Also check conda env
            if let Ok(conda_prefix) = std::env::var("CONDA_PREFIX") {
                println!("cargo:rustc-link-search=native={conda_prefix}/Library/lib");
            }
            println!("cargo:rustc-link-lib=openblas");
        }
    }
}

// Compile hand-tuned SGEMM microkernels from `src/asm/*.s`. One `.s` per
// (arch, ABI) pair — `cc` handles the ELF/Mach-O/COFF object-file format
// and leading-underscore symbol decoration per platform.
//
// Layout:
//   src/asm/x86_64_sysv.s   — Linux/macOS x86_64 (System V AMD64 ABI)
//   src/asm/x86_64_win64.s  — Windows x86_64   (Win64 ABI)
//   src/asm/aarch64.s       — all aarch64 OSes (AAPCS64)
fn assemble_sgemm_asm() {
    let arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let env_abi = std::env::var("CARGO_CFG_TARGET_ENV").unwrap_or_default();

    // .S (capital) means "preprocess with cpp, then assemble" — lets us share
    // a header (`asm_common.h`) with symbol-decoration macros across platforms.
    //
    // Coverage (expanded in later phases):
    //   x86_64 Linux/macOS   — Phase 1 (4×8 AVX+FMA).
    //   x86_64 Windows-gnu   — MinGW gcc can handle GAS `.S` directly.
    //   x86_64 Windows-msvc  — `cl.exe` ignores `.S`; skip and fall through
    //                          to the intrinsics paths in `matmul.rs`. The
    //                          Rust `extern "C"` declarations are cfg-gated
    //                          to the same `(arch, os, env)` triple below.
    //   aarch64              — Phase 3 (4×24 NEON).
    //   wasm32/riscv/etc.    — scalar-only, nothing to assemble.
    let sources: &[&str] = match (arch.as_str(), os.as_str(), env_abi.as_str()) {
        ("x86_64", "linux", _) | ("x86_64", "macos", _) => &["src/asm/x86_64_sysv.S"],
        ("x86_64", "windows", "gnu") => &["src/asm/x86_64_win64.S"],
        ("aarch64", _, _) => &["src/asm/aarch64.S"],
        _ => {
            println!("cargo:rerun-if-changed=src/asm");
            return;
        }
    };

    println!("cargo:rerun-if-changed=src/asm/asm_common.h");
    for src in sources {
        println!("cargo:rerun-if-changed={src}");
    }

    let mut build = cc::Build::new();
    build.files(sources.iter().copied());
    build.include("src/asm");
    // Keep the assembled objects small; no PLT indirection for intra-crate calls.
    build.flag_if_supported("-fno-plt");
    // aarch64: enable the fp16 ISA extension so the assembler accepts
    // `fmla v.8h` (FEAT_FP16 — ARMv8.2-A+fp16). Any CPU supporting FP16
    // fp16 FMA also supports the baseline armv8.2-a, so this flag is safe
    // as a global setting for the aarch64 build (runtime dispatch still
    // checks `is_aarch64_feature_detected!("fp16")` before issuing fp16
    // instructions, so scalar paths remain correct on older ARMv8.0-A).
    if arch == "aarch64" {
        build.flag_if_supported("-march=armv8.2-a+fp16");
    }

    // Cross-compile helper: when the host lacks the default cross-gcc (e.g.
    // `aarch64-linux-gnu-gcc` is missing on NixOS), fall back to clang with
    // `--target=<triple>`. Host-native builds are unaffected — the check
    // below is strictly for host ≠ target cases where the `cc` default
    // (gcc with a target prefix) isn't present on PATH.
    let host = std::env::var("HOST").unwrap_or_default();
    let target = std::env::var("TARGET").unwrap_or_default();
    if host != target {
        let default_gcc = format!("{target}-gcc");
        if which(&default_gcc).is_err() && which("clang").is_ok() {
            build.compiler("clang");
            build.flag(format!("--target={target}"));
        }
    }

    build.compile("yscv_sgemm_asm");
}

// Minimal `which` — avoids pulling in a new dep just for one call.
fn which(program: &str) -> std::io::Result<std::path::PathBuf> {
    let path = std::env::var_os("PATH")
        .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::NotFound, "PATH not set"))?;
    for dir in std::env::split_paths(&path) {
        let candidate = dir.join(program);
        if candidate.is_file() {
            return Ok(candidate);
        }
    }
    Err(std::io::Error::new(
        std::io::ErrorKind::NotFound,
        format!("{program} not found in PATH"),
    ))
}
