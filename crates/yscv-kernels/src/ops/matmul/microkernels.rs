//! gebp_kernel_raw and the per-arch register-tiled microkernels
//! (AVX-512/AVX2/FMA/SSE/NEON/scalar) plus their SiLU epilogue helpers.

#[cfg(target_arch = "aarch64")]
use super::neon::{microkernel_4x8_neon, microkernel_4x16_neon, microkernel_4x24_neon};
use super::*;

/// Raw pointer version for use from parallel context.
///
/// # Safety
/// - `packed_a` must have at least `div_ceil(mc, MR) * kc * MR` elements.
/// - `packed_b` must have at least `div_ceil(nc, NR) * kc * NR` elements.
/// - `output` must point to a buffer with at least `(ic + mc) * n` elements.
/// - Caller must ensure no concurrent writes to the same output rows.
#[allow(unsafe_code)]
#[allow(unsafe_op_in_unsafe_fn)]
#[cfg_attr(
    any(target_arch = "x86", target_arch = "x86_64"),
    target_feature(enable = "avx,fma")
)]
#[cfg_attr(target_arch = "aarch64", target_feature(enable = "neon"))]
pub(super) unsafe fn gebp_kernel_raw(
    packed_a: *const f32,
    packed_b: *const f32,
    output: *mut f32,
    n: usize,
    ic: usize,
    jc: usize,
    mc: usize,
    nc: usize,
    kc: usize,
    accumulate: bool,
    epilogue: GemmEpilogue,
    is_last_k: bool,
) {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    let use_4x16 = std::is_x86_feature_detected!("fma") && std::is_x86_feature_detected!("avx");
    #[cfg(target_arch = "aarch64")]
    let use_4x16 = std::arch::is_aarch64_feature_detected!("neon");
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
    let use_4x16 = false;

    // AVX-512 detection — cached (CPUID once, then atomic-load hot path).
    // Env override `YSCV_NO_AVX512=1` disables for A/B benchmarking.
    #[cfg(all(target_arch = "x86_64", any(target_os = "linux", target_os = "macos")))]
    let use_avx512 = avx512_enabled();
    #[cfg(not(all(target_arch = "x86_64", any(target_os = "linux", target_os = "macos"))))]
    let use_avx512 = false;
    let _ = use_avx512;

    let mut jr = 0usize;
    while jr < nc {
        let nr = NR.min(nc - jr);
        let b_off = (jr / NR) * kc * NR;
        let col_offset = jc + jr;

        // AVX-512 4×32 fast path. The `.S` kernel now has INLINE bias+Relu
        // epilogue (Phase 2.F) so it CAN handle Conv_Relu — but on Zen 4
        // the AVX-512 backend is double-pumped and measured 9% slower than
        // our 4×24 AVX2 intrinsic at Conv_Relu shapes. We keep the gate at
        // IDENTITY only; on true-AVX-512 hardware (Intel Sapphire Rapids,
        // Zen 5 with non-double-pumped 512-bit FMA) a relaxed gate
        // `!matches!(epilogue.activation, Activation::Silu)` would win.
        // Env override `YSCV_AVX512_RELU=1` forces the relaxed gate for
        // hand measurement / Intel validation.
        #[cfg(all(target_arch = "x86_64", any(target_os = "linux", target_os = "macos")))]
        if use_avx512
            && nr == NR
            && jr + 4 * NR <= nc
            && ((epilogue.bias.is_none() && matches!(epilogue.activation, Activation::None))
                || (avx512_relu_enabled() && !matches!(epilogue.activation, Activation::Silu)))
        {
            let b_off = (jr / NR) * kc * NR;
            let activation_id: usize = match epilogue.activation {
                Activation::None => 0,
                Activation::Relu => 1,
                Activation::Silu => 0, // unreachable (outer gate)
            };
            let is_last_k_flag: usize = if is_last_k { 1 } else { 0 };
            for ir in (0..mc).step_by(MR) {
                let mr = MR.min(mc - ir);
                let a_off = (ir / MR) * kc * MR;
                let c_ptr = output.add((ic + ir) * n + jc + jr);
                if mr == MR {
                    let bias_ptr: *const f32 = match epilogue.bias {
                        Some(b) => b.add(jc + jr),
                        None => std::ptr::null(),
                    };
                    if accumulate {
                        sgemm_asm_avx512::yscv_sgemm_4x32_avx512_acc(
                            packed_a.add(a_off),
                            packed_b.add(b_off),
                            c_ptr,
                            n,
                            kc,
                            bias_ptr,
                            activation_id,
                            is_last_k_flag,
                        );
                    } else {
                        sgemm_asm_avx512::yscv_sgemm_4x32_avx512_set(
                            packed_a.add(a_off),
                            packed_b.add(b_off),
                            c_ptr,
                            n,
                            kc,
                            bias_ptr,
                            activation_id,
                            is_last_k_flag,
                        );
                    }
                } else {
                    // Partial-row tail: fall back to four scalar NR-panel calls.
                    for panel in 0..4 {
                        let poff = ((jr + panel * NR) / NR) * kc * NR;
                        microkernel_scalar_partial(
                            packed_a.add(a_off),
                            packed_b.add(poff),
                            c_ptr.add(panel * NR),
                            n,
                            mr,
                            NR,
                            kc,
                            accumulate,
                            epilogue,
                            is_last_k,
                            jc + jr + panel * NR,
                            epilogue
                                .residual
                                .map(|r| r.add((ic + ir) * n + jc + jr + panel * NR)),
                        );
                    }
                }
            }
            jr += 4 * NR;
            continue;
        }

        // Try tripled 4×24 when three full NR=8 tiles are available.
        #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
        if use_4x16 && nr == NR && jr + 3 * NR <= nc {
            let b_off_1 = ((jr + NR) / NR) * kc * NR;
            let b_off_2 = ((jr + 2 * NR) / NR) * kc * NR;
            for ir in (0..mc).step_by(MR) {
                let mr = MR.min(mc - ir);
                let a_off = (ir / MR) * kc * MR;
                let c_ptr = output.add((ic + ir) * n + col_offset);
                // Per-tile residual pointer — same row/col math as output.
                // x86 4×24 tile uses `residual_tile` for in-kernel residual
                // fold; aarch64 NEON 4×24 still routes residual via the
                // matmul guard (falls back to row_gemm) — unused there.
                #[cfg_attr(not(target_arch = "x86_64"), allow(unused_variables))]
                let residual_tile = epilogue.residual.map(|r| r.add((ic + ir) * n + col_offset));
                if mr == MR {
                    // Windows-MSVC has no `.S` kernel linked (cl.exe can't
                    // speak GAS); take the intrinsics 4×24 path directly so
                    // `link.exe` doesn't hunt for `yscv_sgemm_4x24_avx2_*`.
                    #[cfg(all(target_arch = "x86_64", target_os = "windows", target_env = "msvc"))]
                    {
                        microkernel_4x24_avx_fma(
                            packed_a.add(a_off),
                            packed_b.add(b_off),
                            packed_b.add(b_off_1),
                            packed_b.add(b_off_2),
                            c_ptr,
                            n,
                            kc,
                            accumulate,
                            epilogue,
                            is_last_k,
                            col_offset,
                            residual_tile,
                        );
                    }

                    #[cfg(all(
                        target_arch = "x86_64",
                        not(all(target_os = "windows", target_env = "msvc"))
                    ))]
                    {
                        // Pure-.S path: hand-scheduled 4×24 with fused
                        // bias + residual + Relu epilogue. SiLU still routes
                        // to intrinsics (bit-trick SiLU not yet in .S).
                        let use_asm =
                            asm_4x24_enabled() && !matches!(epilogue.activation, Activation::Silu);
                        if use_asm {
                            let activation_id: usize = match epilogue.activation {
                                Activation::Relu => 1,
                                _ => 0,
                            };
                            let is_last_k_flag: usize = if is_last_k { 1 } else { 0 };
                            let bias_ptr: *const f32 = match epilogue.bias {
                                Some(b) if is_last_k => b.add(col_offset),
                                _ => std::ptr::null(),
                            };
                            let residual_ptr: *const f32 = match residual_tile {
                                Some(r) if is_last_k => r,
                                _ => std::ptr::null(),
                            };
                            if accumulate {
                                sgemm_asm::yscv_sgemm_4x24_avx2_acc_fused(
                                    packed_a.add(a_off),
                                    packed_b.add(b_off),
                                    packed_b.add(b_off_1),
                                    packed_b.add(b_off_2),
                                    c_ptr,
                                    n,
                                    kc,
                                    bias_ptr,
                                    residual_ptr,
                                    activation_id,
                                    is_last_k_flag,
                                );
                            } else {
                                sgemm_asm::yscv_sgemm_4x24_avx2_set_fused(
                                    packed_a.add(a_off),
                                    packed_b.add(b_off),
                                    packed_b.add(b_off_1),
                                    packed_b.add(b_off_2),
                                    c_ptr,
                                    n,
                                    kc,
                                    bias_ptr,
                                    residual_ptr,
                                    activation_id,
                                    is_last_k_flag,
                                );
                            }
                        } else {
                            microkernel_4x24_avx_fma(
                                packed_a.add(a_off),
                                packed_b.add(b_off),
                                packed_b.add(b_off_1),
                                packed_b.add(b_off_2),
                                c_ptr,
                                n,
                                kc,
                                accumulate,
                                epilogue,
                                is_last_k,
                                col_offset,
                                residual_tile,
                            );
                        }
                    }
                    #[cfg(target_arch = "aarch64")]
                    {
                        // Fast path: hand-tuned `.S` kernel for pure GEMM (no
                        // bias/activation). Dispatch on epilogue shape, mirroring
                        // the x86_64 4×8 fast path.
                        if epilogue.bias.is_none()
                            && epilogue.residual.is_none()
                            && matches!(epilogue.activation, Activation::None)
                        {
                            if accumulate {
                                sgemm_asm_aarch64::yscv_sgemm_4x24_neon_acc(
                                    packed_a.add(a_off),
                                    packed_b.add(b_off),
                                    packed_b.add(b_off_1),
                                    packed_b.add(b_off_2),
                                    c_ptr,
                                    n,
                                    kc,
                                );
                            } else {
                                sgemm_asm_aarch64::yscv_sgemm_4x24_neon_set(
                                    packed_a.add(a_off),
                                    packed_b.add(b_off),
                                    packed_b.add(b_off_1),
                                    packed_b.add(b_off_2),
                                    c_ptr,
                                    n,
                                    kc,
                                );
                            }
                        } else {
                            microkernel_4x24_neon(
                                packed_a.add(a_off),
                                packed_b.add(b_off),
                                packed_b.add(b_off_1),
                                packed_b.add(b_off_2),
                                c_ptr,
                                n,
                                kc,
                                accumulate,
                                epilogue,
                                is_last_k,
                                col_offset,
                                residual_tile,
                            );
                        }
                    }
                } else {
                    for panel in 0..3 {
                        let poff = [b_off, b_off_1, b_off_2][panel];
                        microkernel_scalar_partial(
                            packed_a.add(a_off),
                            packed_b.add(poff),
                            c_ptr.add(panel * NR),
                            n,
                            mr,
                            NR,
                            kc,
                            accumulate,
                            epilogue,
                            is_last_k,
                            col_offset + panel * NR,
                            residual_tile.map(|r| r.add(panel * NR)),
                        );
                    }
                }
            }
            jr += 3 * NR;
            continue;
        }

        // Try paired 4×16 when two full NR=8 tiles are available.
        #[cfg(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64"))]
        if use_4x16 && nr == NR && jr + 2 * NR <= nc {
            let b_off_1 = ((jr + NR) / NR) * kc * NR;
            for ir in (0..mc).step_by(MR) {
                let mr = MR.min(mc - ir);
                let a_off = (ir / MR) * kc * MR;
                let c_ptr = output.add((ic + ir) * n + col_offset);
                // x86 4×16 tile uses `residual_tile` for in-kernel residual
                // fold; aarch64 NEON 4×16 still routes residual via the
                // matmul guard (falls back to row_gemm) — unused there.
                #[cfg_attr(
                    not(any(target_arch = "x86", target_arch = "x86_64")),
                    allow(unused_variables)
                )]
                let residual_tile = epilogue.residual.map(|r| r.add((ic + ir) * n + col_offset));
                if mr == MR {
                    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                    microkernel_4x16_avx_fma(
                        packed_a.add(a_off),
                        packed_b.add(b_off),
                        packed_b.add(b_off_1),
                        c_ptr,
                        n,
                        kc,
                        accumulate,
                        epilogue,
                        is_last_k,
                        col_offset,
                        residual_tile,
                    );
                    #[cfg(target_arch = "aarch64")]
                    microkernel_4x16_neon(
                        packed_a.add(a_off),
                        packed_b.add(b_off),
                        packed_b.add(b_off_1),
                        c_ptr,
                        n,
                        kc,
                        accumulate,
                        epilogue,
                        is_last_k,
                        col_offset,
                        residual_tile,
                    );
                } else {
                    microkernel_scalar_partial(
                        packed_a.add(a_off),
                        packed_b.add(b_off),
                        c_ptr,
                        n,
                        mr,
                        NR,
                        kc,
                        accumulate,
                        epilogue,
                        is_last_k,
                        col_offset,
                        residual_tile,
                    );
                    microkernel_scalar_partial(
                        packed_a.add(a_off),
                        packed_b.add(b_off_1),
                        c_ptr.add(NR),
                        n,
                        mr,
                        NR,
                        kc,
                        accumulate,
                        epilogue,
                        is_last_k,
                        col_offset + NR,
                        residual_tile.map(|r| r.add(NR)),
                    );
                }
            }
            jr += 2 * NR;
            continue;
        }
        let _ = use_4x16;

        for ir in (0..mc).step_by(MR) {
            let mr = MR.min(mc - ir);
            let a_off = (ir / MR) * kc * MR;
            let c_ptr = output.add((ic + ir) * n + col_offset);
            #[cfg_attr(
                not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")),
                allow(unused_variables)
            )]
            let residual_tile = epilogue.residual.map(|r| r.add((ic + ir) * n + col_offset));

            if mr == MR && nr == NR {
                microkernel_4x8_dispatch(
                    packed_a.add(a_off),
                    packed_b.add(b_off),
                    c_ptr,
                    n,
                    kc,
                    accumulate,
                    epilogue,
                    is_last_k,
                    col_offset,
                    residual_tile,
                );
            } else {
                microkernel_scalar_partial(
                    packed_a.add(a_off),
                    packed_b.add(b_off),
                    c_ptr,
                    n,
                    mr,
                    nr,
                    kc,
                    accumulate,
                    epilogue,
                    is_last_k,
                    col_offset,
                    residual_tile,
                );
            }
        }
        jr += NR;
    }
}

// Micro-kernel dispatch
// ---------------------------------------------------------------------------

/// Dispatch MR×NR micro-kernel to best SIMD path.
///
/// Computes C[MR×NR] += A_packed[MR×kc] * B_packed[kc×NR].
///
/// # Safety
/// - `a_panel`: at least `kc * MR` elements (packed column-of-MR format).
/// - `b_panel`: at least `kc * NR` elements (packed row-of-NR format).
/// - `c`: points to C[ir, jr]; next row at `c + ldc`.
/// - `ldc`: row stride of full C matrix (= N).
#[allow(unsafe_code)]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn microkernel_4x8_dispatch(
    a_panel: *const f32,
    b_panel: *const f32,
    c: *mut f32,
    ldc: usize,
    kc: usize,
    accumulate: bool,
    epilogue: GemmEpilogue,
    is_last_k: bool,
    col_offset: usize,
    residual_tile: Option<*const f32>,
) {
    // Residual is fused by `microkernel_4x24_avx_fma` / `microkernel_4x16_avx_fma`
    // and by the scalar/NEON 4×8 variants. The x86 SIMD/ASM 4×8 paths below
    // (`sgemm_4x8_set/acc`, `microkernel_4x8_avx_fma`, `microkernel_4x8_avx`,
    // `microkernel_4x8_sse`) do NOT read `residual_tile`, so when the caller
    // hands in a residual we must skip them and fall through to the scalar
    // kernel to avoid silent-dropping the add. Gated on `is_last_k` because
    // residual is only applied on the final k-block.
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    let needs_scalar_for_residual = residual_tile.is_some() && is_last_k;

    // Fast path: hand-tuned `.S` microkernel for no-epilogue case (pure GEMM).
    // Separate SET/ACC entry points → no branch in hot path. Covers SysV
    // (Linux, macOS) and Win64 (Windows); the Win64 variant translates args
    // internally so call sites are ABI-agnostic.
    #[cfg(all(
        target_arch = "x86_64",
        any(
            target_os = "linux",
            target_os = "macos",
            all(target_os = "windows", not(target_env = "msvc"))
        )
    ))]
    if !needs_scalar_for_residual
        && epilogue.bias.is_none()
        && matches!(epilogue.activation, Activation::None)
        && std::is_x86_feature_detected!("fma")
        && std::is_x86_feature_detected!("avx")
    {
        if accumulate {
            sgemm_asm::yscv_sgemm_4x8_acc(a_panel, b_panel, c, ldc, kc);
        } else {
            sgemm_asm::yscv_sgemm_4x8_set(a_panel, b_panel, c, ldc, kc);
        }
        return;
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if !needs_scalar_for_residual {
        if std::is_x86_feature_detected!("fma") && std::is_x86_feature_detected!("avx") {
            microkernel_4x8_avx_fma(
                a_panel, b_panel, c, ldc, kc, accumulate, epilogue, is_last_k, col_offset,
            );
            return;
        }
        if std::is_x86_feature_detected!("avx") {
            microkernel_4x8_avx(
                a_panel, b_panel, c, ldc, kc, accumulate, epilogue, is_last_k, col_offset,
            );
            return;
        }
        if std::is_x86_feature_detected!("sse") {
            microkernel_4x8_sse(
                a_panel, b_panel, c, ldc, kc, accumulate, epilogue, is_last_k, col_offset,
            );
            return;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            microkernel_4x8_neon(
                a_panel,
                b_panel,
                c,
                ldc,
                kc,
                accumulate,
                epilogue,
                is_last_k,
                col_offset,
                residual_tile,
            );
            return;
        }
    }

    microkernel_4x8_scalar(
        a_panel,
        b_panel,
        c,
        ldc,
        kc,
        accumulate,
        epilogue,
        is_last_k,
        col_offset,
        residual_tile,
    );
}

// ---------------------------------------------------------------------------
// Epilogue helpers: apply bias + activation on register values before store
// ---------------------------------------------------------------------------

/// Scalar epilogue: bias + activation on a row slice of accumulator values.
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[inline(always)]
unsafe fn apply_epilogue_scalar(
    row: &mut [f32],
    epilogue: &GemmEpilogue,
    col_offset: usize,
    residual_tile_row: Option<*const f32>,
) {
    if let Some(bias) = epilogue.bias {
        for (j, v) in row.iter_mut().enumerate() {
            *v += *bias.add(col_offset + j);
        }
    }
    if let Some(res) = residual_tile_row {
        for (j, v) in row.iter_mut().enumerate() {
            *v += *res.add(j);
        }
    }
    match epilogue.activation {
        Activation::Relu => {
            for v in row.iter_mut() {
                *v = v.max(0.0);
            }
        }
        Activation::Silu => {
            for v in row.iter_mut() {
                let sig = 1.0 / (1.0 + (-*v).exp());
                *v *= sig;
            }
        }
        Activation::None => {}
    }
}

// ---------------------------------------------------------------------------
// Scalar micro-kernel (fallback + miri)
// ---------------------------------------------------------------------------

#[allow(unsafe_code)]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn microkernel_4x8_scalar(
    a_panel: *const f32,
    b_panel: *const f32,
    c: *mut f32,
    ldc: usize,
    kc: usize,
    accumulate: bool,
    epilogue: GemmEpilogue,
    is_last_k: bool,
    col_offset: usize,
    residual_tile: Option<*const f32>,
) {
    let mut acc = [[0.0f32; NR]; MR];

    for p in 0..kc {
        let a_base = p * MR;
        let b_base = p * NR;
        for i in 0..MR {
            let a_val = *a_panel.add(a_base + i);
            for j in 0..NR {
                acc[i][j] += a_val * *b_panel.add(b_base + j);
            }
        }
    }

    for i in 0..MR {
        let row_ptr = c.add(i * ldc);
        if accumulate {
            for j in 0..NR {
                acc[i][j] += *row_ptr.add(j);
            }
        }
        if is_last_k {
            let residual_row = residual_tile.map(|r| r.add(i * ldc));
            apply_epilogue_scalar(&mut acc[i][..NR], &epilogue, col_offset, residual_row);
        }
        for j in 0..NR {
            *row_ptr.add(j) = acc[i][j];
        }
    }
}

/// aarch64-only: scalar fallback for the 8×12 MR=8 path when the row or
/// column count doesn't match the full tile. Unlike the general
/// `microkernel_scalar_partial_stride`, this one knows both packed
/// strides are larger than the generic NR=8 (`a_stride=8`, `b_stride=12`),
/// so indexing has to match the NR=12 packed-B layout instead of the
/// generic NR=8.
#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
pub(super) unsafe fn microkernel_scalar_nr12_partial_mr8(
    a_panel: *const f32,
    b_panel: *const f32,
    c: *mut f32,
    ldc: usize,
    mr: usize,
    nr: usize,
    kc: usize,
    accumulate: bool,
    epilogue: GemmEpilogue,
    is_last_k: bool,
    col_offset: usize,
    residual_tile: Option<*const f32>,
) {
    const MR8: usize = 8;
    const NR12: usize = 12;
    debug_assert!(mr <= MR8);
    debug_assert!(nr <= NR12);

    let mut acc = [[0.0f32; NR12]; MR8];

    for p in 0..kc {
        let a_base = p * MR8;
        let b_base = p * NR12;
        for i in 0..mr {
            let a_val = *a_panel.add(a_base + i);
            for j in 0..nr {
                acc[i][j] += a_val * *b_panel.add(b_base + j);
            }
        }
    }

    for i in 0..mr {
        let row_ptr = c.add(i * ldc);
        if accumulate {
            for j in 0..nr {
                acc[i][j] += *row_ptr.add(j);
            }
        }
        if is_last_k {
            let residual_row = residual_tile.map(|r| r.add(i * ldc));
            apply_epilogue_scalar(&mut acc[i][..nr], &epilogue, col_offset, residual_row);
        }
        for j in 0..nr {
            *row_ptr.add(j) = acc[i][j];
        }
    }
}

/// Scalar micro-kernel for edge tiles where mr < MR or nr < NR.
#[allow(unsafe_code)]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn microkernel_scalar_partial(
    a_panel: *const f32,
    b_panel: *const f32,
    c: *mut f32,
    ldc: usize,
    mr: usize,
    nr: usize,
    kc: usize,
    accumulate: bool,
    epilogue: GemmEpilogue,
    is_last_k: bool,
    col_offset: usize,
    residual_tile: Option<*const f32>,
) {
    microkernel_scalar_partial_stride(
        a_panel,
        b_panel,
        c,
        ldc,
        mr,
        nr,
        kc,
        MR,
        accumulate,
        epilogue,
        is_last_k,
        col_offset,
        residual_tile,
    );
}

#[allow(unsafe_code)]
#[allow(unsafe_op_in_unsafe_fn)]
pub(super) unsafe fn microkernel_scalar_partial_stride(
    a_panel: *const f32,
    b_panel: *const f32,
    c: *mut f32,
    ldc: usize,
    mr: usize,
    nr: usize,
    kc: usize,
    a_stride: usize,
    accumulate: bool,
    epilogue: GemmEpilogue,
    is_last_k: bool,
    col_offset: usize,
    residual_tile: Option<*const f32>,
) {
    // Sized for max(MR=4 x86/scalar, MR=8 aarch64 mr8 path). Real rows bounded by `mr`.
    let mut acc = [[0.0f32; NR]; 8];

    for p in 0..kc {
        let a_base = p * a_stride;
        let b_base = p * NR;
        for i in 0..mr {
            let a_val = *a_panel.add(a_base + i);
            for j in 0..nr {
                acc[i][j] += a_val * *b_panel.add(b_base + j);
            }
        }
    }

    for i in 0..mr {
        let row_ptr = c.add(i * ldc);
        if accumulate {
            for j in 0..nr {
                acc[i][j] += *row_ptr.add(j);
            }
        }
        if is_last_k {
            let residual_row = residual_tile.map(|r| r.add(i * ldc));
            apply_epilogue_scalar(&mut acc[i][..nr], &epilogue, col_offset, residual_row);
        }
        for j in 0..nr {
            *row_ptr.add(j) = acc[i][j];
        }
    }
}

// ---------------------------------------------------------------------------
// Paired 4×16 AVX+FMA micro-kernel: two adjacent NR=8 tiles at once.
// Amortizes 4 a-broadcasts across 16 columns instead of 8, reaching
// theoretical peak FMA throughput on Zen 4 (2 FMA ports × 8-wide).
// ---------------------------------------------------------------------------

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code)]
#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx,fma")]
#[inline]
unsafe fn microkernel_4x16_avx_fma(
    a_panel: *const f32,
    b_panel_0: *const f32,
    b_panel_1: *const f32,
    c: *mut f32,
    ldc: usize,
    kc: usize,
    accumulate: bool,
    epilogue: GemmEpilogue,
    is_last_k: bool,
    col_offset: usize,
    residual_tile: Option<*const f32>,
) {
    // 8 accumulators: ymm0-ymm7 (c00,c01,c10,c11,c20,c21,c30,c31)
    // Temp: ymm8,ymm9 (b0,b1), ymm10 (a broadcast)
    let mut c00: __m256;
    let mut c01: __m256;
    let mut c10: __m256;
    let mut c11: __m256;
    let mut c20: __m256;
    let mut c21: __m256;
    let mut c30: __m256;
    let mut c31: __m256;

    #[cfg(target_arch = "x86_64")]
    std::arch::asm!(
        "vxorps {c00}, {c00}, {c00}",
        "vxorps {c01}, {c01}, {c01}",
        "vxorps {c10}, {c10}, {c10}",
        "vxorps {c11}, {c11}, {c11}",
        "vxorps {c20}, {c20}, {c20}",
        "vxorps {c21}, {c21}, {c21}",
        "vxorps {c30}, {c30}, {c30}",
        "vxorps {c31}, {c31}, {c31}",

        "mov {pairs}, {kc}",
        "shr {pairs}, 1",
        "test {pairs}, {pairs}",
        "jz 12f",

        ".p2align 4",
        "13:",
        // 2-way unrolled k-loop with DOUBLE-BUFFERED B loads.
        // b0a/b1a hold B[k]; we preload B[k+1] into b0b/b1b EARLY (interleaved
        // with FMAs on b0a/b1a) so k+1's FMAs can start immediately after k's
        // finish. 8 acc + 4 B + 1 A = 13 YMM, 3 free for OoO scheduling.
        //
        // --- k-step 0: load B[k] AND B[k+1] (into b0b, b1b) ---
        "vmovups {b0a}, [{bp0}]",
        "vmovups {b1a}, [{bp1}]",
        "vmovups {b0b}, [{bp0} + 32]",         // B[k+1] panel 0 — starts loading
        "vbroadcastss {av}, [{ap}]",
        "vfmadd231ps {c00}, {b0a}, {av}",
        "vfmadd231ps {c01}, {b1a}, {av}",
        "vmovups {b1b}, [{bp1} + 32]",         // B[k+1] panel 1 — interleaved with FMAs
        "vbroadcastss {av}, [{ap} + 4]",
        "vfmadd231ps {c10}, {b0a}, {av}",
        "vfmadd231ps {c11}, {b1a}, {av}",
        "vbroadcastss {av}, [{ap} + 8]",
        "vfmadd231ps {c20}, {b0a}, {av}",
        "vfmadd231ps {c21}, {b1a}, {av}",
        "prefetcht0 [{ap} + 64]",              // prefetch next A panel chunk
        "vbroadcastss {av}, [{ap} + 12]",
        "vfmadd231ps {c30}, {b0a}, {av}",
        "vfmadd231ps {c31}, {b1a}, {av}",
        // --- k-step 1: b0b, b1b already have B[k+1] (loaded above) ---
        "vbroadcastss {av}, [{ap} + 16]",
        "vfmadd231ps {c00}, {b0b}, {av}",
        "vfmadd231ps {c01}, {b1b}, {av}",
        "vbroadcastss {av}, [{ap} + 20]",
        "vfmadd231ps {c10}, {b0b}, {av}",
        "vfmadd231ps {c11}, {b1b}, {av}",
        "vbroadcastss {av}, [{ap} + 24]",
        "vfmadd231ps {c20}, {b0b}, {av}",
        "vfmadd231ps {c21}, {b1b}, {av}",
        "vbroadcastss {av}, [{ap} + 28]",
        "vfmadd231ps {c30}, {b0b}, {av}",
        "vfmadd231ps {c31}, {b1b}, {av}",
        "add {ap}, 32",
        "add {bp0}, 64",
        "add {bp1}, 64",
        "dec {pairs}",
        "jnz 13b",

        "12:",
        "test {kc}, 1",
        "jz 14f",
        "vmovups {b0a}, [{bp0}]",
        "vmovups {b1a}, [{bp1}]",
        "vbroadcastss {av}, [{ap}]",
        "vfmadd231ps {c00}, {b0a}, {av}",
        "vfmadd231ps {c01}, {b1a}, {av}",
        "vbroadcastss {av}, [{ap} + 4]",
        "vfmadd231ps {c10}, {b0a}, {av}",
        "vfmadd231ps {c11}, {b1a}, {av}",
        "vbroadcastss {av}, [{ap} + 8]",
        "vfmadd231ps {c20}, {b0a}, {av}",
        "vfmadd231ps {c21}, {b1a}, {av}",
        "vbroadcastss {av}, [{ap} + 12]",
        "vfmadd231ps {c30}, {b0a}, {av}",
        "vfmadd231ps {c31}, {b1a}, {av}",
        "14:",

        ap = inout(reg) a_panel => _,
        bp0 = inout(reg) b_panel_0 => _,
        bp1 = inout(reg) b_panel_1 => _,
        kc = in(reg) kc,
        pairs = out(reg) _,
        c00 = out(ymm_reg) c00,
        c01 = out(ymm_reg) c01,
        c10 = out(ymm_reg) c10,
        c11 = out(ymm_reg) c11,
        c20 = out(ymm_reg) c20,
        c21 = out(ymm_reg) c21,
        c30 = out(ymm_reg) c30,
        c31 = out(ymm_reg) c31,
        b0a = out(ymm_reg) _,
        b1a = out(ymm_reg) _,
        b0b = out(ymm_reg) _,
        b1b = out(ymm_reg) _,
        av = out(ymm_reg) _,
        options(nostack),
    );

    #[cfg(target_arch = "x86")]
    {
        // x86 32-bit fallback
        c00 = _mm256_setzero_ps();
        c01 = _mm256_setzero_ps();
        c10 = _mm256_setzero_ps();
        c11 = _mm256_setzero_ps();
        c20 = _mm256_setzero_ps();
        c21 = _mm256_setzero_ps();
        c30 = _mm256_setzero_ps();
        c31 = _mm256_setzero_ps();
        for p in 0..kc {
            let b0 = _mm256_loadu_ps(b_panel_0.add(p * NR));
            let b1 = _mm256_loadu_ps(b_panel_1.add(p * NR));
            let a0 = _mm256_set1_ps(*a_panel.add(p * MR));
            c00 = _mm256_fmadd_ps(a0, b0, c00);
            c01 = _mm256_fmadd_ps(a0, b1, c01);
            let a1 = _mm256_set1_ps(*a_panel.add(p * MR + 1));
            c10 = _mm256_fmadd_ps(a1, b0, c10);
            c11 = _mm256_fmadd_ps(a1, b1, c11);
            let a2 = _mm256_set1_ps(*a_panel.add(p * MR + 2));
            c20 = _mm256_fmadd_ps(a2, b0, c20);
            c21 = _mm256_fmadd_ps(a2, b1, c21);
            let a3 = _mm256_set1_ps(*a_panel.add(p * MR + 3));
            c30 = _mm256_fmadd_ps(a3, b0, c30);
            c31 = _mm256_fmadd_ps(a3, b1, c31);
        }
    }

    let cp0 = c;
    let cp1 = c.add(ldc);
    let cp2 = c.add(2 * ldc);
    let cp3 = c.add(3 * ldc);

    if accumulate {
        c00 = _mm256_add_ps(_mm256_loadu_ps(cp0), c00);
        c01 = _mm256_add_ps(_mm256_loadu_ps(cp0.add(8)), c01);
        c10 = _mm256_add_ps(_mm256_loadu_ps(cp1), c10);
        c11 = _mm256_add_ps(_mm256_loadu_ps(cp1.add(8)), c11);
        c20 = _mm256_add_ps(_mm256_loadu_ps(cp2), c20);
        c21 = _mm256_add_ps(_mm256_loadu_ps(cp2.add(8)), c21);
        c30 = _mm256_add_ps(_mm256_loadu_ps(cp3), c30);
        c31 = _mm256_add_ps(_mm256_loadu_ps(cp3.add(8)), c31);
    }

    if is_last_k {
        if let Some(bias) = epilogue.bias {
            let bv0 = _mm256_loadu_ps(bias.add(col_offset));
            let bv1 = _mm256_loadu_ps(bias.add(col_offset + 8));
            c00 = _mm256_add_ps(c00, bv0);
            c01 = _mm256_add_ps(c01, bv1);
            c10 = _mm256_add_ps(c10, bv0);
            c11 = _mm256_add_ps(c11, bv1);
            c20 = _mm256_add_ps(c20, bv0);
            c21 = _mm256_add_ps(c21, bv1);
            c30 = _mm256_add_ps(c30, bv0);
            c31 = _mm256_add_ps(c31, bv1);
        }
        // Residual add (Phase 1.2): 8 YMMs from per-tile residual base.
        if let Some(res_tile) = residual_tile {
            let rp0 = res_tile;
            let rp1 = res_tile.add(ldc);
            let rp2 = res_tile.add(2 * ldc);
            let rp3 = res_tile.add(3 * ldc);
            c00 = _mm256_add_ps(c00, _mm256_loadu_ps(rp0));
            c01 = _mm256_add_ps(c01, _mm256_loadu_ps(rp0.add(8)));
            c10 = _mm256_add_ps(c10, _mm256_loadu_ps(rp1));
            c11 = _mm256_add_ps(c11, _mm256_loadu_ps(rp1.add(8)));
            c20 = _mm256_add_ps(c20, _mm256_loadu_ps(rp2));
            c21 = _mm256_add_ps(c21, _mm256_loadu_ps(rp2.add(8)));
            c30 = _mm256_add_ps(c30, _mm256_loadu_ps(rp3));
            c31 = _mm256_add_ps(c31, _mm256_loadu_ps(rp3.add(8)));
        }
        match epilogue.activation {
            Activation::Relu => {
                let zero = _mm256_setzero_ps();
                c00 = _mm256_max_ps(c00, zero);
                c01 = _mm256_max_ps(c01, zero);
                c10 = _mm256_max_ps(c10, zero);
                c11 = _mm256_max_ps(c11, zero);
                c20 = _mm256_max_ps(c20, zero);
                c21 = _mm256_max_ps(c21, zero);
                c30 = _mm256_max_ps(c30, zero);
                c31 = _mm256_max_ps(c31, zero);
            }
            Activation::Silu => {
                c00 = silu_avx_fma(c00);
                c01 = silu_avx_fma(c01);
                c10 = silu_avx_fma(c10);
                c11 = silu_avx_fma(c11);
                c20 = silu_avx_fma(c20);
                c21 = silu_avx_fma(c21);
                c30 = silu_avx_fma(c30);
                c31 = silu_avx_fma(c31);
            }
            Activation::None => {}
        }
    }

    _mm256_storeu_ps(cp0, c00);
    _mm256_storeu_ps(cp0.add(8), c01);
    _mm256_storeu_ps(cp1, c10);
    _mm256_storeu_ps(cp1.add(8), c11);
    _mm256_storeu_ps(cp2, c20);
    _mm256_storeu_ps(cp2.add(8), c21);
    _mm256_storeu_ps(cp3, c30);
    _mm256_storeu_ps(cp3.add(8), c31);
}

// ---------------------------------------------------------------------------
// MR=6 × NR=16 AVX+FMA micro-kernel (Step C1)
// ---------------------------------------------------------------------------
// Matches MLAS SgemmKernelFma3's 6×16 tile. Consumes 6 rows of a MR=6-
// packed A panel and 2 NR=8 B panels, producing a 6×16 output tile.
// 12 YMM accumulators (6 rows × 2 panels) + 2 B + 1 A broadcast = 15 YMM.
// 1 YMM free — no room for B[k+1] double-buffering, so k-loop is
// single-step (LLVM is allowed to schedule; `#[inline]` + `target_feature`
// lets it unroll/schedule at its discretion).
//
// Register pressure justification: 6 rows × 2 panels × 8 cols = 96 floats
// of output per tile = 12 YMM at exactly AVX2's 16-YMM file. MR=8×16 would
// need 16 acc + B + A = 19 YMM (no fit). MR=6 is the sweet spot on AVX2.
//
// Compared to MR=4×NR=24 (12 acc × 3 panels = same 12 acc): MR=6 halves
// the number of B panels per tile-row (2 vs 3) and trades 50% more A
// broadcasts (6 vs 4 per k-step). On m≥192 shapes this nets **55% less
// B-panel traffic** across the outer loop. Expected: +10–20% throughput
// on pointwise Conv shapes.

/// 6×16 AVX+FMA microkernel. Pure intrinsics; compiler schedules.
/// Bias + optional residual + activation folded inline for single-pass
/// store. Mirrors `microkernel_4x16_avx_fma` structure scaled to 6 rows.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code)]
#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx,fma")]
#[inline]
pub(super) unsafe fn microkernel_6x16_avx_fma(
    a_panel: *const f32,
    b_panel_0: *const f32,
    b_panel_1: *const f32,
    c: *mut f32,
    ldc: usize,
    kc: usize,
    accumulate: bool,
    epilogue: GemmEpilogue,
    is_last_k: bool,
    col_offset: usize,
    residual_tile: Option<*const f32>,
) {
    // 12 accumulators, 6 rows × 2 NR panels of 8 cols.
    let mut c00: __m256 = _mm256_setzero_ps();
    let mut c01: __m256 = _mm256_setzero_ps();
    let mut c10: __m256 = _mm256_setzero_ps();
    let mut c11: __m256 = _mm256_setzero_ps();
    let mut c20: __m256 = _mm256_setzero_ps();
    let mut c21: __m256 = _mm256_setzero_ps();
    let mut c30: __m256 = _mm256_setzero_ps();
    let mut c31: __m256 = _mm256_setzero_ps();
    let mut c40: __m256 = _mm256_setzero_ps();
    let mut c41: __m256 = _mm256_setzero_ps();
    let mut c50: __m256 = _mm256_setzero_ps();
    let mut c51: __m256 = _mm256_setzero_ps();

    // K-loop: 12 FMAs per k-step, 2 B-panel loads, 6 A broadcasts.
    // Pack-A stride is MR6=6 floats per k-step.
    for p in 0..kc {
        let a_off = p * MR6;
        let b_off = p * NR;
        let b0 = _mm256_loadu_ps(b_panel_0.add(b_off));
        let b1 = _mm256_loadu_ps(b_panel_1.add(b_off));

        let a0 = _mm256_broadcast_ss(&*a_panel.add(a_off));
        c00 = _mm256_fmadd_ps(b0, a0, c00);
        c01 = _mm256_fmadd_ps(b1, a0, c01);

        let a1 = _mm256_broadcast_ss(&*a_panel.add(a_off + 1));
        c10 = _mm256_fmadd_ps(b0, a1, c10);
        c11 = _mm256_fmadd_ps(b1, a1, c11);

        let a2 = _mm256_broadcast_ss(&*a_panel.add(a_off + 2));
        c20 = _mm256_fmadd_ps(b0, a2, c20);
        c21 = _mm256_fmadd_ps(b1, a2, c21);

        let a3 = _mm256_broadcast_ss(&*a_panel.add(a_off + 3));
        c30 = _mm256_fmadd_ps(b0, a3, c30);
        c31 = _mm256_fmadd_ps(b1, a3, c31);

        let a4 = _mm256_broadcast_ss(&*a_panel.add(a_off + 4));
        c40 = _mm256_fmadd_ps(b0, a4, c40);
        c41 = _mm256_fmadd_ps(b1, a4, c41);

        let a5 = _mm256_broadcast_ss(&*a_panel.add(a_off + 5));
        c50 = _mm256_fmadd_ps(b0, a5, c50);
        c51 = _mm256_fmadd_ps(b1, a5, c51);
    }

    // Row pointers for C.
    let cp0 = c;
    let cp1 = c.add(ldc);
    let cp2 = c.add(2 * ldc);
    let cp3 = c.add(3 * ldc);
    let cp4 = c.add(4 * ldc);
    let cp5 = c.add(5 * ldc);

    // Accumulate existing C when running non-first k-block.
    if accumulate {
        c00 = _mm256_add_ps(_mm256_loadu_ps(cp0), c00);
        c01 = _mm256_add_ps(_mm256_loadu_ps(cp0.add(8)), c01);
        c10 = _mm256_add_ps(_mm256_loadu_ps(cp1), c10);
        c11 = _mm256_add_ps(_mm256_loadu_ps(cp1.add(8)), c11);
        c20 = _mm256_add_ps(_mm256_loadu_ps(cp2), c20);
        c21 = _mm256_add_ps(_mm256_loadu_ps(cp2.add(8)), c21);
        c30 = _mm256_add_ps(_mm256_loadu_ps(cp3), c30);
        c31 = _mm256_add_ps(_mm256_loadu_ps(cp3.add(8)), c31);
        c40 = _mm256_add_ps(_mm256_loadu_ps(cp4), c40);
        c41 = _mm256_add_ps(_mm256_loadu_ps(cp4.add(8)), c41);
        c50 = _mm256_add_ps(_mm256_loadu_ps(cp5), c50);
        c51 = _mm256_add_ps(_mm256_loadu_ps(cp5.add(8)), c51);
    }

    // Epilogue (bias + residual + activation) only on last k-block.
    if is_last_k {
        if let Some(bias) = epilogue.bias {
            let bv0 = _mm256_loadu_ps(bias.add(col_offset));
            let bv1 = _mm256_loadu_ps(bias.add(col_offset + 8));
            c00 = _mm256_add_ps(c00, bv0);
            c01 = _mm256_add_ps(c01, bv1);
            c10 = _mm256_add_ps(c10, bv0);
            c11 = _mm256_add_ps(c11, bv1);
            c20 = _mm256_add_ps(c20, bv0);
            c21 = _mm256_add_ps(c21, bv1);
            c30 = _mm256_add_ps(c30, bv0);
            c31 = _mm256_add_ps(c31, bv1);
            c40 = _mm256_add_ps(c40, bv0);
            c41 = _mm256_add_ps(c41, bv1);
            c50 = _mm256_add_ps(c50, bv0);
            c51 = _mm256_add_ps(c51, bv1);
        }
        // Residual is already shifted by caller to this tile's base.
        // Stride matches ldc (= n).
        if let Some(res_tile) = residual_tile {
            let rp0 = res_tile;
            let rp1 = res_tile.add(ldc);
            let rp2 = res_tile.add(2 * ldc);
            let rp3 = res_tile.add(3 * ldc);
            let rp4 = res_tile.add(4 * ldc);
            let rp5 = res_tile.add(5 * ldc);
            c00 = _mm256_add_ps(c00, _mm256_loadu_ps(rp0));
            c01 = _mm256_add_ps(c01, _mm256_loadu_ps(rp0.add(8)));
            c10 = _mm256_add_ps(c10, _mm256_loadu_ps(rp1));
            c11 = _mm256_add_ps(c11, _mm256_loadu_ps(rp1.add(8)));
            c20 = _mm256_add_ps(c20, _mm256_loadu_ps(rp2));
            c21 = _mm256_add_ps(c21, _mm256_loadu_ps(rp2.add(8)));
            c30 = _mm256_add_ps(c30, _mm256_loadu_ps(rp3));
            c31 = _mm256_add_ps(c31, _mm256_loadu_ps(rp3.add(8)));
            c40 = _mm256_add_ps(c40, _mm256_loadu_ps(rp4));
            c41 = _mm256_add_ps(c41, _mm256_loadu_ps(rp4.add(8)));
            c50 = _mm256_add_ps(c50, _mm256_loadu_ps(rp5));
            c51 = _mm256_add_ps(c51, _mm256_loadu_ps(rp5.add(8)));
        }
        match epilogue.activation {
            Activation::Relu => {
                let zero = _mm256_setzero_ps();
                c00 = _mm256_max_ps(c00, zero);
                c01 = _mm256_max_ps(c01, zero);
                c10 = _mm256_max_ps(c10, zero);
                c11 = _mm256_max_ps(c11, zero);
                c20 = _mm256_max_ps(c20, zero);
                c21 = _mm256_max_ps(c21, zero);
                c30 = _mm256_max_ps(c30, zero);
                c31 = _mm256_max_ps(c31, zero);
                c40 = _mm256_max_ps(c40, zero);
                c41 = _mm256_max_ps(c41, zero);
                c50 = _mm256_max_ps(c50, zero);
                c51 = _mm256_max_ps(c51, zero);
            }
            Activation::Silu => {
                c00 = silu_avx_fma(c00);
                c01 = silu_avx_fma(c01);
                c10 = silu_avx_fma(c10);
                c11 = silu_avx_fma(c11);
                c20 = silu_avx_fma(c20);
                c21 = silu_avx_fma(c21);
                c30 = silu_avx_fma(c30);
                c31 = silu_avx_fma(c31);
                c40 = silu_avx_fma(c40);
                c41 = silu_avx_fma(c41);
                c50 = silu_avx_fma(c50);
                c51 = silu_avx_fma(c51);
            }
            Activation::None => {}
        }
    }

    _mm256_storeu_ps(cp0, c00);
    _mm256_storeu_ps(cp0.add(8), c01);
    _mm256_storeu_ps(cp1, c10);
    _mm256_storeu_ps(cp1.add(8), c11);
    _mm256_storeu_ps(cp2, c20);
    _mm256_storeu_ps(cp2.add(8), c21);
    _mm256_storeu_ps(cp3, c30);
    _mm256_storeu_ps(cp3.add(8), c31);
    _mm256_storeu_ps(cp4, c40);
    _mm256_storeu_ps(cp4.add(8), c41);
    _mm256_storeu_ps(cp5, c50);
    _mm256_storeu_ps(cp5.add(8), c51);
}

// ---------------------------------------------------------------------------
// AVX+FMA micro-kernel (x86/x86_64)
// ---------------------------------------------------------------------------

/// AVX+FMA 4×8 microkernel with inline asm k-loop.
///
/// The k-loop is hand-scheduled to maximize FMA throughput on Zen 4
/// (2 FMA ports, 4-cycle latency → need 8+ independent FMAs).
/// Uses `core::arch::asm!` to avoid LLVM stack frame overhead.
#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx,fma")]
unsafe fn microkernel_4x8_avx_fma(
    a_panel: *const f32,
    b_panel: *const f32,
    c: *mut f32,
    ldc: usize,
    kc: usize,
    accumulate: bool,
    epilogue: GemmEpilogue,
    is_last_k: bool,
    col_offset: usize,
) {
    // Accumulator registers: ymm0-ymm3 (c0-c3)
    // Temp registers: ymm4 (b_vec), ymm5-ymm8 (a broadcasts)
    let mut c0: std::arch::x86_64::__m256;
    let mut c1: std::arch::x86_64::__m256;
    let mut c2: std::arch::x86_64::__m256;
    let mut c3: std::arch::x86_64::__m256;

    // K-loop in inline asm — no stack frame, pure register computation.
    // a_panel: MR=4 packed floats per k (16 bytes stride)
    // b_panel: NR=8 packed floats per k (32 bytes stride)
    std::arch::asm!(
        // Zero accumulators
        "vxorps {c0}, {c0}, {c0}",
        "vxorps {c1}, {c1}, {c1}",
        "vxorps {c2}, {c2}, {c2}",
        "vxorps {c3}, {c3}, {c3}",

        // kc_pairs = kc >> 1
        "mov {pairs}, {kc}",
        "shr {pairs}, 1",
        "test {pairs}, {pairs}",
        "jz 2f",

        // Main loop: 2 k-steps per iteration
        ".p2align 4",
        "3:",
        // k-step 0
        "vmovups {bv}, [{bp}]",
        "vbroadcastss {t0}, [{ap}]",
        "vfmadd231ps {c0}, {bv}, {t0}",
        "vbroadcastss {t0}, [{ap} + 4]",
        "vfmadd231ps {c1}, {bv}, {t0}",
        "vbroadcastss {t0}, [{ap} + 8]",
        "vfmadd231ps {c2}, {bv}, {t0}",
        "vbroadcastss {t0}, [{ap} + 12]",
        "vfmadd231ps {c3}, {bv}, {t0}",
        // k-step 1 with prefetch
        "vmovups {bv}, [{bp} + 32]",
        "prefetcht0 [{ap} + 64]",
        "vbroadcastss {t0}, [{ap} + 16]",
        "vfmadd231ps {c0}, {bv}, {t0}",
        "vbroadcastss {t0}, [{ap} + 20]",
        "vfmadd231ps {c1}, {bv}, {t0}",
        "vbroadcastss {t0}, [{ap} + 24]",
        "vfmadd231ps {c2}, {bv}, {t0}",
        "vbroadcastss {t0}, [{ap} + 28]",
        "vfmadd231ps {c3}, {bv}, {t0}",
        // Advance pointers
        "add {ap}, 32",  // 2 * MR * 4 bytes
        "add {bp}, 64",  // 2 * NR * 4 bytes
        "dec {pairs}",
        "jnz 3b",

        "2:",
        // Handle odd remainder
        "test {kc}, 1",
        "jz 4f",
        "vmovups {bv}, [{bp}]",
        "vbroadcastss {t0}, [{ap}]",
        "vfmadd231ps {c0}, {bv}, {t0}",
        "vbroadcastss {t0}, [{ap} + 4]",
        "vfmadd231ps {c1}, {bv}, {t0}",
        "vbroadcastss {t0}, [{ap} + 8]",
        "vfmadd231ps {c2}, {bv}, {t0}",
        "vbroadcastss {t0}, [{ap} + 12]",
        "vfmadd231ps {c3}, {bv}, {t0}",
        "4:",

        ap = inout(reg) a_panel => _,
        bp = inout(reg) b_panel => _,
        kc = in(reg) kc,
        pairs = out(reg) _,
        c0 = out(ymm_reg) c0,
        c1 = out(ymm_reg) c1,
        c2 = out(ymm_reg) c2,
        c3 = out(ymm_reg) c3,
        bv = out(ymm_reg) _,
        t0 = out(ymm_reg) _,
        options(nostack),
    );

    // Store phase (Rust intrinsics — LLVM handles well)
    let cp0 = c;
    let cp1 = c.add(ldc);
    let cp2 = c.add(2 * ldc);
    let cp3 = c.add(3 * ldc);

    if accumulate {
        c0 = _mm256_add_ps(_mm256_loadu_ps(cp0), c0);
        c1 = _mm256_add_ps(_mm256_loadu_ps(cp1), c1);
        c2 = _mm256_add_ps(_mm256_loadu_ps(cp2), c2);
        c3 = _mm256_add_ps(_mm256_loadu_ps(cp3), c3);
    }

    if is_last_k {
        if let Some(bias) = epilogue.bias {
            let bv = _mm256_loadu_ps(bias.add(col_offset));
            c0 = _mm256_add_ps(c0, bv);
            c1 = _mm256_add_ps(c1, bv);
            c2 = _mm256_add_ps(c2, bv);
            c3 = _mm256_add_ps(c3, bv);
        }
        match epilogue.activation {
            Activation::Relu => {
                let zero = _mm256_setzero_ps();
                c0 = _mm256_max_ps(c0, zero);
                c1 = _mm256_max_ps(c1, zero);
                c2 = _mm256_max_ps(c2, zero);
                c3 = _mm256_max_ps(c3, zero);
            }
            Activation::Silu => {
                c0 = silu_avx_fma(c0);
                c1 = silu_avx_fma(c1);
                c2 = silu_avx_fma(c2);
                c3 = silu_avx_fma(c3);
            }
            Activation::None => {}
        }
    }

    _mm256_storeu_ps(cp0, c0);
    _mm256_storeu_ps(cp1, c1);
    _mm256_storeu_ps(cp2, c2);
    _mm256_storeu_ps(cp3, c3);
}

/// Fallback for x86 (32-bit) — uses Rust intrinsics.
#[cfg(target_arch = "x86")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx,fma")]
unsafe fn microkernel_4x8_avx_fma(
    a_panel: *const f32,
    b_panel: *const f32,
    c: *mut f32,
    ldc: usize,
    kc: usize,
    accumulate: bool,
    epilogue: GemmEpilogue,
    is_last_k: bool,
    col_offset: usize,
) {
    // x86 (32-bit) fallback — same logic, Rust intrinsics
    let mut c0: __m256 = _mm256_setzero_ps();
    let mut c1: __m256 = _mm256_setzero_ps();
    let mut c2: __m256 = _mm256_setzero_ps();
    let mut c3: __m256 = _mm256_setzero_ps();

    for p in 0..kc {
        let b_vec = _mm256_loadu_ps(b_panel.add(p * NR));
        c0 = _mm256_fmadd_ps(_mm256_set1_ps(*a_panel.add(p * MR)), b_vec, c0);
        c1 = _mm256_fmadd_ps(_mm256_set1_ps(*a_panel.add(p * MR + 1)), b_vec, c1);
        c2 = _mm256_fmadd_ps(_mm256_set1_ps(*a_panel.add(p * MR + 2)), b_vec, c2);
        c3 = _mm256_fmadd_ps(_mm256_set1_ps(*a_panel.add(p * MR + 3)), b_vec, c3);
    }

    let cp0 = c;
    let cp1 = c.add(ldc);
    let cp2 = c.add(2 * ldc);
    let cp3 = c.add(3 * ldc);

    if accumulate {
        c0 = _mm256_add_ps(_mm256_loadu_ps(cp0), c0);
        c1 = _mm256_add_ps(_mm256_loadu_ps(cp1), c1);
        c2 = _mm256_add_ps(_mm256_loadu_ps(cp2), c2);
        c3 = _mm256_add_ps(_mm256_loadu_ps(cp3), c3);
    }
    if is_last_k {
        if let Some(bias) = epilogue.bias {
            let bv = _mm256_loadu_ps(bias.add(col_offset));
            c0 = _mm256_add_ps(c0, bv);
            c1 = _mm256_add_ps(c1, bv);
            c2 = _mm256_add_ps(c2, bv);
            c3 = _mm256_add_ps(c3, bv);
        }
        match epilogue.activation {
            Activation::Relu => {
                let zero = _mm256_setzero_ps();
                c0 = _mm256_max_ps(c0, zero);
                c1 = _mm256_max_ps(c1, zero);
                c2 = _mm256_max_ps(c2, zero);
                c3 = _mm256_max_ps(c3, zero);
            }
            Activation::Silu => {
                c0 = silu_avx_fma(c0);
                c1 = silu_avx_fma(c1);
                c2 = silu_avx_fma(c2);
                c3 = silu_avx_fma(c3);
            }
            Activation::None => {}
        }
    }
    _mm256_storeu_ps(cp0, c0);
    _mm256_storeu_ps(cp1, c1);
    _mm256_storeu_ps(cp2, c2);
    _mm256_storeu_ps(cp3, c3);
}

// ---------------------------------------------------------------------------
// AVX micro-kernel (x86/x86_64)
// ---------------------------------------------------------------------------

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code)]
#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx")]
unsafe fn microkernel_4x8_avx(
    a_panel: *const f32,
    b_panel: *const f32,
    c: *mut f32,
    ldc: usize,
    kc: usize,
    accumulate: bool,
    epilogue: GemmEpilogue,
    is_last_k: bool,
    col_offset: usize,
) {
    let mut c0: __m256 = _mm256_setzero_ps();
    let mut c1: __m256 = _mm256_setzero_ps();
    let mut c2: __m256 = _mm256_setzero_ps();
    let mut c3: __m256 = _mm256_setzero_ps();

    for p in 0..kc {
        let b_vec = _mm256_loadu_ps(b_panel.add(p * NR));

        let a0 = _mm256_set1_ps(*a_panel.add(p * MR));
        c0 = _mm256_add_ps(c0, _mm256_mul_ps(a0, b_vec));

        let a1 = _mm256_set1_ps(*a_panel.add(p * MR + 1));
        c1 = _mm256_add_ps(c1, _mm256_mul_ps(a1, b_vec));

        let a2 = _mm256_set1_ps(*a_panel.add(p * MR + 2));
        c2 = _mm256_add_ps(c2, _mm256_mul_ps(a2, b_vec));

        let a3 = _mm256_set1_ps(*a_panel.add(p * MR + 3));
        c3 = _mm256_add_ps(c3, _mm256_mul_ps(a3, b_vec));
    }

    let cp0 = c;
    let cp1 = c.add(ldc);
    let cp2 = c.add(2 * ldc);
    let cp3 = c.add(3 * ldc);

    if accumulate {
        c0 = _mm256_add_ps(_mm256_loadu_ps(cp0), c0);
        c1 = _mm256_add_ps(_mm256_loadu_ps(cp1), c1);
        c2 = _mm256_add_ps(_mm256_loadu_ps(cp2), c2);
        c3 = _mm256_add_ps(_mm256_loadu_ps(cp3), c3);
    }

    if is_last_k {
        if let Some(bias) = epilogue.bias {
            let bv = _mm256_loadu_ps(bias.add(col_offset));
            c0 = _mm256_add_ps(c0, bv);
            c1 = _mm256_add_ps(c1, bv);
            c2 = _mm256_add_ps(c2, bv);
            c3 = _mm256_add_ps(c3, bv);
        }
        match epilogue.activation {
            Activation::Relu => {
                let zero = _mm256_setzero_ps();
                c0 = _mm256_max_ps(c0, zero);
                c1 = _mm256_max_ps(c1, zero);
                c2 = _mm256_max_ps(c2, zero);
                c3 = _mm256_max_ps(c3, zero);
            }
            Activation::Silu => {
                c0 = silu_avx(c0);
                c1 = silu_avx(c1);
                c2 = silu_avx(c2);
                c3 = silu_avx(c3);
            }
            Activation::None => {}
        }
    }

    _mm256_storeu_ps(cp0, c0);
    _mm256_storeu_ps(cp1, c1);
    _mm256_storeu_ps(cp2, c2);
    _mm256_storeu_ps(cp3, c3);
}

// ---------------------------------------------------------------------------
// SSE micro-kernel (x86/x86_64)
// ---------------------------------------------------------------------------

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code)]
#[allow(unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "sse")]
unsafe fn microkernel_4x8_sse(
    a_panel: *const f32,
    b_panel: *const f32,
    c: *mut f32,
    ldc: usize,
    kc: usize,
    accumulate: bool,
    epilogue: GemmEpilogue,
    is_last_k: bool,
    col_offset: usize,
) {
    let mut c00: __m128 = _mm_setzero_ps();
    let mut c01: __m128 = _mm_setzero_ps();
    let mut c10: __m128 = _mm_setzero_ps();
    let mut c11: __m128 = _mm_setzero_ps();
    let mut c20: __m128 = _mm_setzero_ps();
    let mut c21: __m128 = _mm_setzero_ps();
    let mut c30: __m128 = _mm_setzero_ps();
    let mut c31: __m128 = _mm_setzero_ps();

    for p in 0..kc {
        let a_base = p * MR;
        let b_base = p * NR;

        let a0 = _mm_set1_ps(*a_panel.add(a_base));
        let a1 = _mm_set1_ps(*a_panel.add(a_base + 1));
        let a2 = _mm_set1_ps(*a_panel.add(a_base + 2));
        let a3 = _mm_set1_ps(*a_panel.add(a_base + 3));

        let b0 = _mm_loadu_ps(b_panel.add(b_base));
        let b1 = _mm_loadu_ps(b_panel.add(b_base + 4));

        c00 = _mm_add_ps(c00, _mm_mul_ps(a0, b0));
        c01 = _mm_add_ps(c01, _mm_mul_ps(a0, b1));
        c10 = _mm_add_ps(c10, _mm_mul_ps(a1, b0));
        c11 = _mm_add_ps(c11, _mm_mul_ps(a1, b1));
        c20 = _mm_add_ps(c20, _mm_mul_ps(a2, b0));
        c21 = _mm_add_ps(c21, _mm_mul_ps(a2, b1));
        c30 = _mm_add_ps(c30, _mm_mul_ps(a3, b0));
        c31 = _mm_add_ps(c31, _mm_mul_ps(a3, b1));
    }

    let cp0 = c;
    let cp1 = c.add(ldc);
    let cp2 = c.add(2 * ldc);
    let cp3 = c.add(3 * ldc);

    if accumulate {
        c00 = _mm_add_ps(_mm_loadu_ps(cp0), c00);
        c01 = _mm_add_ps(_mm_loadu_ps(cp0.add(4)), c01);
        c10 = _mm_add_ps(_mm_loadu_ps(cp1), c10);
        c11 = _mm_add_ps(_mm_loadu_ps(cp1.add(4)), c11);
        c20 = _mm_add_ps(_mm_loadu_ps(cp2), c20);
        c21 = _mm_add_ps(_mm_loadu_ps(cp2.add(4)), c21);
        c30 = _mm_add_ps(_mm_loadu_ps(cp3), c30);
        c31 = _mm_add_ps(_mm_loadu_ps(cp3.add(4)), c31);
    }

    if is_last_k {
        if let Some(bias) = epilogue.bias {
            let bv0 = _mm_loadu_ps(bias.add(col_offset));
            let bv1 = _mm_loadu_ps(bias.add(col_offset + 4));
            c00 = _mm_add_ps(c00, bv0);
            c01 = _mm_add_ps(c01, bv1);
            c10 = _mm_add_ps(c10, bv0);
            c11 = _mm_add_ps(c11, bv1);
            c20 = _mm_add_ps(c20, bv0);
            c21 = _mm_add_ps(c21, bv1);
            c30 = _mm_add_ps(c30, bv0);
            c31 = _mm_add_ps(c31, bv1);
        }
        match epilogue.activation {
            Activation::Relu => {
                let zero = _mm_setzero_ps();
                c00 = _mm_max_ps(c00, zero);
                c01 = _mm_max_ps(c01, zero);
                c10 = _mm_max_ps(c10, zero);
                c11 = _mm_max_ps(c11, zero);
                c20 = _mm_max_ps(c20, zero);
                c21 = _mm_max_ps(c21, zero);
                c30 = _mm_max_ps(c30, zero);
                c31 = _mm_max_ps(c31, zero);
            }
            Activation::Silu => {
                c00 = silu_sse(c00);
                c01 = silu_sse(c01);
                c10 = silu_sse(c10);
                c11 = silu_sse(c11);
                c20 = silu_sse(c20);
                c21 = silu_sse(c21);
                c30 = silu_sse(c30);
                c31 = silu_sse(c31);
            }
            Activation::None => {}
        }
    }

    _mm_storeu_ps(cp0, c00);
    _mm_storeu_ps(cp0.add(4), c01);
    _mm_storeu_ps(cp1, c10);
    _mm_storeu_ps(cp1.add(4), c11);
    _mm_storeu_ps(cp2, c20);
    _mm_storeu_ps(cp2.add(4), c21);
    _mm_storeu_ps(cp3, c30);
    _mm_storeu_ps(cp3.add(4), c31);
}

// ---------------------------------------------------------------------------
// 4×24 AVX+FMA micro-kernel: three adjacent NR=8 tiles at once.
// 12 ymm accumulators + 3 B loads + 1 A broadcast = 16 ymm registers (exact fit).
// Reduces microkernel call count by 50% vs 4×16 for wide N.
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx,fma")]
#[inline]
unsafe fn microkernel_4x24_avx_fma(
    a_panel: *const f32,
    b_panel_0: *const f32,
    b_panel_1: *const f32,
    b_panel_2: *const f32,
    c: *mut f32,
    ldc: usize,
    kc: usize,
    accumulate: bool,
    epilogue: GemmEpilogue,
    is_last_k: bool,
    col_offset: usize,
    // Per-tile residual pointer (already shifted by `(ic+ir)*n + col_offset`
    // at dispatch site). When `Some` and `is_last_k`, the epilogue adds
    // 12 residual YMMs to the accumulators BEFORE applying activation,
    // folding a separate `add_inplace` pass into the GEMM store. Stride
    // matches `ldc = n`.
    residual_tile: Option<*const f32>,
) {
    // 12 accumulators (4 rows × 3 panels of 8 cols)
    let mut c00: __m256;
    let mut c01: __m256;
    let mut c02: __m256;
    let mut c10: __m256;
    let mut c11: __m256;
    let mut c12: __m256;
    let mut c20: __m256;
    let mut c21: __m256;
    let mut c22: __m256;
    let mut c30: __m256;
    let mut c31: __m256;
    let mut c32: __m256;

    std::arch::asm!(
        // Zero 12 accumulators
        "vxorps {c00}, {c00}, {c00}",
        "vxorps {c01}, {c01}, {c01}",
        "vxorps {c02}, {c02}, {c02}",
        "vxorps {c10}, {c10}, {c10}",
        "vxorps {c11}, {c11}, {c11}",
        "vxorps {c12}, {c12}, {c12}",
        "vxorps {c20}, {c20}, {c20}",
        "vxorps {c21}, {c21}, {c21}",
        "vxorps {c22}, {c22}, {c22}",
        "vxorps {c30}, {c30}, {c30}",
        "vxorps {c31}, {c31}, {c31}",
        "vxorps {c32}, {c32}, {c32}",

        "mov {pairs}, {kc}",
        "shr {pairs}, 1",
        "test {pairs}, {pairs}",
        "jz 24f",

        ".p2align 4",
        "23:",
        // --- k-step 0 ---
        "vmovups {b0}, [{bp0}]",
        "vmovups {b1}, [{bp1}]",
        "vmovups {b2}, [{bp2}]",
        "vbroadcastss {av}, [{ap}]",
        "vfmadd231ps {c00}, {b0}, {av}",
        "vfmadd231ps {c01}, {b1}, {av}",
        "vfmadd231ps {c02}, {b2}, {av}",
        "vbroadcastss {av}, [{ap} + 4]",
        "vfmadd231ps {c10}, {b0}, {av}",
        "vfmadd231ps {c11}, {b1}, {av}",
        "vfmadd231ps {c12}, {b2}, {av}",
        "vbroadcastss {av}, [{ap} + 8]",
        "vfmadd231ps {c20}, {b0}, {av}",
        "vfmadd231ps {c21}, {b1}, {av}",
        "vfmadd231ps {c22}, {b2}, {av}",
        "vbroadcastss {av}, [{ap} + 12]",
        "vfmadd231ps {c30}, {b0}, {av}",
        "vfmadd231ps {c31}, {b1}, {av}",
        "vfmadd231ps {c32}, {b2}, {av}",
        // --- k-step 1 ---
        "vmovups {b0}, [{bp0} + 32]",
        "vmovups {b1}, [{bp1} + 32]",
        "vmovups {b2}, [{bp2} + 32]",
        "vbroadcastss {av}, [{ap} + 16]",
        "vfmadd231ps {c00}, {b0}, {av}",
        "vfmadd231ps {c01}, {b1}, {av}",
        "vfmadd231ps {c02}, {b2}, {av}",
        "vbroadcastss {av}, [{ap} + 20]",
        "vfmadd231ps {c10}, {b0}, {av}",
        "vfmadd231ps {c11}, {b1}, {av}",
        "vfmadd231ps {c12}, {b2}, {av}",
        "vbroadcastss {av}, [{ap} + 24]",
        "vfmadd231ps {c20}, {b0}, {av}",
        "vfmadd231ps {c21}, {b1}, {av}",
        "vfmadd231ps {c22}, {b2}, {av}",
        "vbroadcastss {av}, [{ap} + 28]",
        "vfmadd231ps {c30}, {b0}, {av}",
        "vfmadd231ps {c31}, {b1}, {av}",
        "vfmadd231ps {c32}, {b2}, {av}",
        // Advance both k-steps at once
        "add {ap}, 32",   // 2 * MR * 4 bytes
        "add {bp0}, 64",  // 2 * NR * 4 bytes
        "add {bp1}, 64",
        "add {bp2}, 64",
        "dec {pairs}",
        "jnz 23b",

        "24:",
        // Handle odd k-remainder (if kc was odd, one final k-step)
        "test {kc}, 1",
        "jz 25f",
        "vmovups {b0}, [{bp0}]",
        "vmovups {b1}, [{bp1}]",
        "vmovups {b2}, [{bp2}]",
        "vbroadcastss {av}, [{ap}]",
        "vfmadd231ps {c00}, {b0}, {av}",
        "vfmadd231ps {c01}, {b1}, {av}",
        "vfmadd231ps {c02}, {b2}, {av}",
        "vbroadcastss {av}, [{ap} + 4]",
        "vfmadd231ps {c10}, {b0}, {av}",
        "vfmadd231ps {c11}, {b1}, {av}",
        "vfmadd231ps {c12}, {b2}, {av}",
        "vbroadcastss {av}, [{ap} + 8]",
        "vfmadd231ps {c20}, {b0}, {av}",
        "vfmadd231ps {c21}, {b1}, {av}",
        "vfmadd231ps {c22}, {b2}, {av}",
        "vbroadcastss {av}, [{ap} + 12]",
        "vfmadd231ps {c30}, {b0}, {av}",
        "vfmadd231ps {c31}, {b1}, {av}",
        "vfmadd231ps {c32}, {b2}, {av}",
        "25:",

        ap = inout(reg) a_panel => _,
        bp0 = inout(reg) b_panel_0 => _,
        bp1 = inout(reg) b_panel_1 => _,
        bp2 = inout(reg) b_panel_2 => _,
        kc = in(reg) kc,
        pairs = out(reg) _,
        c00 = out(ymm_reg) c00, c01 = out(ymm_reg) c01, c02 = out(ymm_reg) c02,
        c10 = out(ymm_reg) c10, c11 = out(ymm_reg) c11, c12 = out(ymm_reg) c12,
        c20 = out(ymm_reg) c20, c21 = out(ymm_reg) c21, c22 = out(ymm_reg) c22,
        c30 = out(ymm_reg) c30, c31 = out(ymm_reg) c31, c32 = out(ymm_reg) c32,
        b0 = out(ymm_reg) _, b1 = out(ymm_reg) _, b2 = out(ymm_reg) _,
        av = out(ymm_reg) _,
        options(nostack),
    );

    // Store phase
    let cp0 = c;
    let cp1 = c.add(ldc);
    let cp2 = c.add(2 * ldc);
    let cp3 = c.add(3 * ldc);

    if accumulate {
        c00 = _mm256_add_ps(_mm256_loadu_ps(cp0), c00);
        c01 = _mm256_add_ps(_mm256_loadu_ps(cp0.add(8)), c01);
        c02 = _mm256_add_ps(_mm256_loadu_ps(cp0.add(16)), c02);
        c10 = _mm256_add_ps(_mm256_loadu_ps(cp1), c10);
        c11 = _mm256_add_ps(_mm256_loadu_ps(cp1.add(8)), c11);
        c12 = _mm256_add_ps(_mm256_loadu_ps(cp1.add(16)), c12);
        c20 = _mm256_add_ps(_mm256_loadu_ps(cp2), c20);
        c21 = _mm256_add_ps(_mm256_loadu_ps(cp2.add(8)), c21);
        c22 = _mm256_add_ps(_mm256_loadu_ps(cp2.add(16)), c22);
        c30 = _mm256_add_ps(_mm256_loadu_ps(cp3), c30);
        c31 = _mm256_add_ps(_mm256_loadu_ps(cp3.add(8)), c31);
        c32 = _mm256_add_ps(_mm256_loadu_ps(cp3.add(16)), c32);
    }

    if is_last_k {
        if let Some(bias) = epilogue.bias {
            let bv0 = _mm256_loadu_ps(bias.add(col_offset));
            let bv1 = _mm256_loadu_ps(bias.add(col_offset + 8));
            let bv2 = _mm256_loadu_ps(bias.add(col_offset + 16));
            c00 = _mm256_add_ps(c00, bv0);
            c01 = _mm256_add_ps(c01, bv1);
            c02 = _mm256_add_ps(c02, bv2);
            c10 = _mm256_add_ps(c10, bv0);
            c11 = _mm256_add_ps(c11, bv1);
            c12 = _mm256_add_ps(c12, bv2);
            c20 = _mm256_add_ps(c20, bv0);
            c21 = _mm256_add_ps(c21, bv1);
            c22 = _mm256_add_ps(c22, bv2);
            c30 = _mm256_add_ps(c30, bv0);
            c31 = _mm256_add_ps(c31, bv1);
            c32 = _mm256_add_ps(c32, bv2);
        }
        // Residual add: out = acc + bias + residual (Phase 1.2). Load 12
        // YMMs from the pre-computed per-tile residual base. Stride same
        // as `ldc` (row k = base + k*ldc). Only fires on is_last_k so
        // residual isn't double-added across accumulator k-blocks.
        if let Some(res_tile) = residual_tile {
            let rp0 = res_tile;
            let rp1 = res_tile.add(ldc);
            let rp2 = res_tile.add(2 * ldc);
            let rp3 = res_tile.add(3 * ldc);
            c00 = _mm256_add_ps(c00, _mm256_loadu_ps(rp0));
            c01 = _mm256_add_ps(c01, _mm256_loadu_ps(rp0.add(8)));
            c02 = _mm256_add_ps(c02, _mm256_loadu_ps(rp0.add(16)));
            c10 = _mm256_add_ps(c10, _mm256_loadu_ps(rp1));
            c11 = _mm256_add_ps(c11, _mm256_loadu_ps(rp1.add(8)));
            c12 = _mm256_add_ps(c12, _mm256_loadu_ps(rp1.add(16)));
            c20 = _mm256_add_ps(c20, _mm256_loadu_ps(rp2));
            c21 = _mm256_add_ps(c21, _mm256_loadu_ps(rp2.add(8)));
            c22 = _mm256_add_ps(c22, _mm256_loadu_ps(rp2.add(16)));
            c30 = _mm256_add_ps(c30, _mm256_loadu_ps(rp3));
            c31 = _mm256_add_ps(c31, _mm256_loadu_ps(rp3.add(8)));
            c32 = _mm256_add_ps(c32, _mm256_loadu_ps(rp3.add(16)));
        }
        match epilogue.activation {
            Activation::Relu => {
                let zero = _mm256_setzero_ps();
                c00 = _mm256_max_ps(c00, zero);
                c01 = _mm256_max_ps(c01, zero);
                c02 = _mm256_max_ps(c02, zero);
                c10 = _mm256_max_ps(c10, zero);
                c11 = _mm256_max_ps(c11, zero);
                c12 = _mm256_max_ps(c12, zero);
                c20 = _mm256_max_ps(c20, zero);
                c21 = _mm256_max_ps(c21, zero);
                c22 = _mm256_max_ps(c22, zero);
                c30 = _mm256_max_ps(c30, zero);
                c31 = _mm256_max_ps(c31, zero);
                c32 = _mm256_max_ps(c32, zero);
            }
            Activation::Silu => {
                c00 = silu_avx_fma(c00);
                c01 = silu_avx_fma(c01);
                c02 = silu_avx_fma(c02);
                c10 = silu_avx_fma(c10);
                c11 = silu_avx_fma(c11);
                c12 = silu_avx_fma(c12);
                c20 = silu_avx_fma(c20);
                c21 = silu_avx_fma(c21);
                c22 = silu_avx_fma(c22);
                c30 = silu_avx_fma(c30);
                c31 = silu_avx_fma(c31);
                c32 = silu_avx_fma(c32);
            }
            Activation::None => {}
        }
    }

    _mm256_storeu_ps(cp0, c00);
    _mm256_storeu_ps(cp0.add(8), c01);
    _mm256_storeu_ps(cp0.add(16), c02);
    _mm256_storeu_ps(cp1, c10);
    _mm256_storeu_ps(cp1.add(8), c11);
    _mm256_storeu_ps(cp1.add(16), c12);
    _mm256_storeu_ps(cp2, c20);
    _mm256_storeu_ps(cp2.add(8), c21);
    _mm256_storeu_ps(cp2.add(16), c22);
    _mm256_storeu_ps(cp3, c30);
    _mm256_storeu_ps(cp3.add(8), c31);
    _mm256_storeu_ps(cp3.add(16), c32);
}

// ---------------------------------------------------------------------------
/// SiLU using AVX+FMA: x / (1 + exp(-x)), uses bit-trick exp for speed.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx,fma")]
#[inline]
pub(super) unsafe fn silu_avx_fma(x: __m256) -> __m256 {
    let one = _mm256_set1_ps(1.0);
    let neg_x = _mm256_sub_ps(_mm256_setzero_ps(), x);
    let exp_neg = super::super::simd::exp::fast_exp_bittrick_avx(neg_x);
    let denom = _mm256_add_ps(one, exp_neg);
    _mm256_div_ps(x, denom)
}

/// SiLU using AVX (no FMA): x / (1 + exp(-x)), uses bit-trick exp.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "avx")]
#[inline]
unsafe fn silu_avx(x: __m256) -> __m256 {
    let one = _mm256_set1_ps(1.0);
    let neg_x = _mm256_sub_ps(_mm256_setzero_ps(), x);
    let exp_neg = super::super::simd::exp::fast_exp_bittrick_avx(neg_x);
    let denom = _mm256_add_ps(one, exp_neg);
    _mm256_div_ps(x, denom)
}

/// SiLU using SSE: x / (1 + exp(-x)), uses bit-trick exp.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unsafe_code, unsafe_op_in_unsafe_fn)]
#[target_feature(enable = "sse")]
#[inline]
unsafe fn silu_sse(x: __m128) -> __m128 {
    let one = _mm_set1_ps(1.0);
    let neg_x = _mm_sub_ps(_mm_setzero_ps(), x);
    let exp_neg = super::super::simd::exp::fast_exp_bittrick_sse(neg_x);
    let denom = _mm_add_ps(one, exp_neg);
    _mm_div_ps(x, denom)
}
