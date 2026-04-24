// yscv SGEMM microkernels — x86_64 System V AMD64 ABI (Linux, macOS).
//
// Args (SysV): rdi, rsi, rdx, rcx, r8.
// Callee-saved: rbx, rbp, r12–r15.
// All xmm/ymm/zmm registers are caller-saved. We `vzeroupper` before
// returning so callers mixing SSE and AVX code don't pay the transition
// penalty.

#include "asm_common.h"

.intel_syntax noprefix

// ============================================================================
// yscv_sgemm_4x8_set / yscv_sgemm_4x8_acc (AVX+FMA, 4 rows × 8 cols)
// ============================================================================
// Software-pipelined k-loop: load B[k+1] early so the load latency hides
// behind the FMA chain on B[k]. 16 YMM regs:
//   ymm0-3  = accumulators (one per row)
//   ymm4    = B[k]
//   ymm5,7  = A broadcast (alternating, so the decoder can dispatch FMAs
//             without waiting for the broadcast issue)
//   ymm6    = B[k+1] preloaded during B[k]'s FMA chain
//
// Calling convention:
//   rdi = a_panel (packed 4×KC, row-major within each k-panel)
//   rsi = b_panel (packed KC×8)
//   rdx = c (output)
//   rcx = ldc (in FLOATS; we shl by 2 to bytes)
//   r8  = kc
//
// `_set` zeros accumulators. `_acc` loads current c into accumulators
// (used when multiple k-blocks accumulate into the same c tile).

.globl CNAME(yscv_sgemm_4x8_set)
#if !defined(__APPLE__)
.type yscv_sgemm_4x8_set, @function
#endif
.p2align 4
CNAME(yscv_sgemm_4x8_set):
    vxorps ymm0, ymm0, ymm0
    vxorps ymm1, ymm1, ymm1
    vxorps ymm2, ymm2, ymm2
    vxorps ymm3, ymm3, ymm3
    jmp .Lsgemm_4x8_kloop
#if !defined(__APPLE__)
.size yscv_sgemm_4x8_set, . - yscv_sgemm_4x8_set
#endif

.globl CNAME(yscv_sgemm_4x8_acc)
#if !defined(__APPLE__)
.type yscv_sgemm_4x8_acc, @function
#endif
.p2align 4
CNAME(yscv_sgemm_4x8_acc):
    shl rcx, 2
    vmovups ymm0, [rdx]
    vmovups ymm1, [rdx + rcx]
    lea r9, [rdx + rcx*2]
    vmovups ymm2, [r9]
    vmovups ymm3, [r9 + rcx]
    shr rcx, 2
    jmp .Lsgemm_4x8_kloop

.p2align 4
.Lsgemm_4x8_kloop:
    test r8, r8
    jz .Lsgemm_4x8_store
    mov r10, r8
    shr r10, 1
    test r10, r10
    jz .Lsgemm_4x8_remainder

.p2align 4
.Lsgemm_4x8_main:
    // --- k-step 0 ---
    vmovups ymm4, [rsi]                 // B[k]
    vbroadcastss ymm5, dword ptr [rdi]  // A[k][0]
    vbroadcastss ymm7, dword ptr [rdi + 4]
    vfmadd231ps ymm0, ymm4, ymm5
    vbroadcastss ymm5, dword ptr [rdi + 8]
    vfmadd231ps ymm1, ymm4, ymm7
    vmovups ymm6, [rsi + 32]            // B[k+1] early — hides load latency
    vbroadcastss ymm7, dword ptr [rdi + 12]
    vfmadd231ps ymm2, ymm4, ymm5
    vfmadd231ps ymm3, ymm4, ymm7
    // --- k-step 1 (B[k+1] already in ymm6) ---
    vbroadcastss ymm5, dword ptr [rdi + 16]
    vbroadcastss ymm7, dword ptr [rdi + 20]
    vfmadd231ps ymm0, ymm6, ymm5
    vbroadcastss ymm5, dword ptr [rdi + 24]
    vfmadd231ps ymm1, ymm6, ymm7
    prefetcht0 [rdi + 64]
    vbroadcastss ymm7, dword ptr [rdi + 28]
    vfmadd231ps ymm2, ymm6, ymm5
    vfmadd231ps ymm3, ymm6, ymm7
    add rdi, 32
    add rsi, 64
    dec r10
    jnz .Lsgemm_4x8_main

.Lsgemm_4x8_remainder:
    test r8b, 1
    jz .Lsgemm_4x8_store
    vmovups ymm4, [rsi]
    vbroadcastss ymm5, dword ptr [rdi]
    vfmadd231ps ymm0, ymm4, ymm5
    vbroadcastss ymm5, dword ptr [rdi + 4]
    vfmadd231ps ymm1, ymm4, ymm5
    vbroadcastss ymm5, dword ptr [rdi + 8]
    vfmadd231ps ymm2, ymm4, ymm5
    vbroadcastss ymm5, dword ptr [rdi + 12]
    vfmadd231ps ymm3, ymm4, ymm5

.Lsgemm_4x8_store:
    shl rcx, 2
    vmovups [rdx], ymm0
    vmovups [rdx + rcx], ymm1
    lea r9, [rdx + rcx*2]
    vmovups [r9], ymm2
    vmovups [r9 + rcx], ymm3
    vzeroupper
    ret
#if !defined(__APPLE__)
.size yscv_sgemm_4x8_acc, . - yscv_sgemm_4x8_acc
#endif

// ============================================================================
// yscv_sgemm_4x32_avx512_set / _acc  (AVX-512, 4 rows × 32 cols, inline epi)
// ============================================================================
// Reuses the existing NR=8 pack layout — four adjacent B panels are combined
// into ZMM registers via `vinsertf64x4` so no repack is required.
//
// Args (SysV): rdi=a, rsi=b_base (= packed_b + b_off of panel 0),
//              rdx=c, rcx=ldc (FLOATS), r8=kc, r9=bias_ptr (or 0).
//              [rsp+8]=activation id (0=None, 1=Relu); SiLU rejected by caller.
//              [rsp+16]=is_last_k (0 or 1).
//
// Panel layout: panels 0..3 sit at rsi + 0, rsi + kc*32, rsi + 2*kc*32,
// rsi + 3*kc*32 bytes. We materialize b1/b2/b3 pointers in the prologue.
//
// Register plan (32 ZMM regs total):
//   zmm0..zmm7   = 8 accumulators (4 rows × 2 ZMM per row)
//   zmm8, zmm9   = B[k] combined (cols 0..15 from panels 0+1, cols 16..31)
//   zmm12..zmm15 = A[k] broadcasts (one per row)
//   zmm16, zmm17 = bias[0..15], bias[16..31] during epilogue
//   zmm18        = zero for Relu
//
// Phase 2.F upgrade: inline bias+Relu epilogue in the store phase. Before
// this change the AVX-512 path was gated on `epilogue == IDENTITY` only —
// now it handles Conv_Relu (the tracker's dominant hot-path op). SiLU
// still routes to MR=4 AVX+FMA intrinsics.

.globl CNAME(yscv_sgemm_4x32_avx512_set)
#if !defined(__APPLE__)
.type yscv_sgemm_4x32_avx512_set, @function
#endif
.p2align 4
CNAME(yscv_sgemm_4x32_avx512_set):
    // r9 arrives as bias_ptr (6th SysV arg) — save to the SysV red zone
    // (128 bytes below rsp, guaranteed not clobbered in leaf functions).
    // Epilogue reloads from [rsp - 8] to apply bias.
    mov qword ptr [rsp - 8], r9

    // Compute panel stride (kc × NR × 4 bytes = kc × 32). r9 is freed above.
    mov r9, r8
    shl r9, 5
    lea r10, [rsi + r9]                 // r10 = b1
    lea r11, [rsi + r9*2]               // r11 = b2
    add r9, r11                         // r9 = rsi + 3*stride = b3

    // Zero accumulators.
    vpxorq zmm0, zmm0, zmm0
    vpxorq zmm1, zmm1, zmm1
    vpxorq zmm2, zmm2, zmm2
    vpxorq zmm3, zmm3, zmm3
    vpxorq zmm4, zmm4, zmm4
    vpxorq zmm5, zmm5, zmm5
    vpxorq zmm6, zmm6, zmm6
    vpxorq zmm7, zmm7, zmm7
    jmp .Lsgemm_4x32_avx512_kloop
#if !defined(__APPLE__)
.size yscv_sgemm_4x32_avx512_set, . - yscv_sgemm_4x32_avx512_set
#endif

.globl CNAME(yscv_sgemm_4x32_avx512_acc)
#if !defined(__APPLE__)
.type yscv_sgemm_4x32_avx512_acc, @function
#endif
.p2align 4
CNAME(yscv_sgemm_4x32_avx512_acc):
    // Save bias_ptr to red zone; reuse r9 as panel-stride scratch.
    mov qword ptr [rsp - 8], r9
    mov r9, r8
    shl r9, 5
    lea r10, [rsi + r9]
    lea r11, [rsi + r9*2]
    add r9, r11                         // r9 = b3

    // Convert ldc floats→bytes for c-load.
    shl rcx, 2
    // Load c into accumulators: 32 cols per row = 2 ZMM per row.
    vmovups zmm0, [rdx]
    vmovups zmm1, [rdx + 64]
    vmovups zmm2, [rdx + rcx]
    vmovups zmm3, [rdx + rcx + 64]
    lea rax, [rdx + rcx*2]
    vmovups zmm4, [rax]
    vmovups zmm5, [rax + 64]
    vmovups zmm6, [rax + rcx]
    vmovups zmm7, [rax + rcx + 64]
    shr rcx, 2                          // restore ldc in floats
    jmp .Lsgemm_4x32_avx512_kloop

.p2align 4
.Lsgemm_4x32_avx512_kloop:
    test r8, r8
    jz .Lsgemm_4x32_avx512_epilogue

.p2align 4
.Lsgemm_4x32_avx512_body:
    // Load B[p] — 4 separate YMM panels, insert upper halves into ZMMs.
    vmovups ymm8, [rsi]
    vinsertf64x4 zmm8, zmm8, ymmword ptr [r10], 1     // zmm8 = [b0[p]|b1[p]] (cols 0..15)
    vmovups ymm9, [r11]
    vinsertf64x4 zmm9, zmm9, ymmword ptr [r9], 1      // zmm9 = [b2[p]|b3[p]] (cols 16..31)

    // Load A[p] — 4 scalars broadcast into 4 ZMMs (one per row).
    vbroadcastss zmm12, dword ptr [rdi]
    vbroadcastss zmm13, dword ptr [rdi + 4]
    vbroadcastss zmm14, dword ptr [rdi + 8]
    vbroadcastss zmm15, dword ptr [rdi + 12]

    // 8 FMAs: (row r, cols [0..15]) = acc[2r+0]; (row r, cols [16..31]) = acc[2r+1].
    vfmadd231ps zmm0, zmm8, zmm12
    vfmadd231ps zmm1, zmm9, zmm12
    vfmadd231ps zmm2, zmm8, zmm13
    vfmadd231ps zmm3, zmm9, zmm13
    vfmadd231ps zmm4, zmm8, zmm14
    vfmadd231ps zmm5, zmm9, zmm14
    vfmadd231ps zmm6, zmm8, zmm15
    vfmadd231ps zmm7, zmm9, zmm15

    add rdi, 16                         // a_panel += MR floats (16 bytes)
    add rsi, 32                         // b0 += NR floats
    add r10, 32
    add r11, 32
    add r9,  32
    dec r8
    jnz .Lsgemm_4x32_avx512_body

.Lsgemm_4x32_avx512_epilogue:
    // is_last_k gate — skip all epilogue work for intermediate k-blocks.
    mov r8, qword ptr [rsp + 16]
    test r8, r8
    jz .Lsgemm_4x32_avx512_store

    // Bias add: reload bias_ptr from the red zone (r9 was clobbered for
    // panel stride above). Null pointer means "no bias".
    mov r9, qword ptr [rsp - 8]
    test r9, r9
    jz .Lsgemm_4x32_avx512_skip_bias
    vmovups zmm16, [r9]                 // bias[0..15]
    vmovups zmm17, [r9 + 64]            // bias[16..31]
    vaddps zmm0, zmm0, zmm16
    vaddps zmm1, zmm1, zmm17
    vaddps zmm2, zmm2, zmm16
    vaddps zmm3, zmm3, zmm17
    vaddps zmm4, zmm4, zmm16
    vaddps zmm5, zmm5, zmm17
    vaddps zmm6, zmm6, zmm16
    vaddps zmm7, zmm7, zmm17

.Lsgemm_4x32_avx512_skip_bias:
    // Activation: 1 = Relu. 0/other = skip. SiLU never reaches this kernel.
    mov r8, qword ptr [rsp + 8]
    cmp r8, 1
    jne .Lsgemm_4x32_avx512_store
    vpxorq zmm18, zmm18, zmm18
    vmaxps zmm0, zmm0, zmm18
    vmaxps zmm1, zmm1, zmm18
    vmaxps zmm2, zmm2, zmm18
    vmaxps zmm3, zmm3, zmm18
    vmaxps zmm4, zmm4, zmm18
    vmaxps zmm5, zmm5, zmm18
    vmaxps zmm6, zmm6, zmm18
    vmaxps zmm7, zmm7, zmm18

.Lsgemm_4x32_avx512_store:
    shl rcx, 2                          // ldc bytes
    vmovups [rdx], zmm0
    vmovups [rdx + 64], zmm1
    vmovups [rdx + rcx], zmm2
    vmovups [rdx + rcx + 64], zmm3
    lea rax, [rdx + rcx*2]
    vmovups [rax], zmm4
    vmovups [rax + 64], zmm5
    vmovups [rax + rcx], zmm6
    vmovups [rax + rcx + 64], zmm7
    vzeroupper
    ret

// ============================================================================
// yscv_sgemm_4x24_avx2_{set,acc}_fused  (AVX+FMA, 4 rows × 24 cols)
// ============================================================================
// Hand-scheduled pure-ASM version of the 4×24 tripled microkernel. Replaces
// the Rust `microkernel_4x24_avx_fma` (intrinsics + inline `asm!` k-loop)
// for the tracker's hot Conv / Conv_Relu / Conv_Add / Conv_Add_Relu paths.
// Pure .S removes the compiler's freedom to spill YMM accumulators under
// the 12-acc + 3-bias + 12-residual register pressure of the epilogue.
//
// Register plan (16 YMM):
//   ymm0..ymm11  = 12 accumulators (4 rows × 3 NR=8 panels):
//     ymm0  = c[0][ 0.. 7]   ymm1  = c[0][ 8..15]   ymm2  = c[0][16..23]
//     ymm3  = c[1][ 0.. 7]   ymm4  = c[1][ 8..15]   ymm5  = c[1][16..23]
//     ymm6  = c[2][ 0.. 7]   ymm7  = c[2][ 8..15]   ymm8  = c[2][16..23]
//     ymm9  = c[3][ 0.. 7]   ymm10 = c[3][ 8..15]   ymm11 = c[3][16..23]
//   ymm12, ymm13, ymm14 = B[k] panel 0/1/2 (reused for bias and residual in
//                          the epilogue — k-loop is done by then)
//   ymm15            = A[k] broadcast (cycled through 4 rows)
//
// Calling convention (SysV AMD64):
//   rdi = a_panel                    (packed MR=4 × kc floats, 16-B rows)
//   rsi = b_panel_0                  (packed kc × NR=8 floats)
//   rdx = b_panel_1                  (kc × NR)
//   rcx = b_panel_2                  (kc × NR)
//   r8  = c                          (output[row][col])
//   r9  = ldc (in FLOATS; we shl r9,2 once at entry, use bytes throughout)
//   [rsp + 8]  = kc
//   [rsp + 16] = bias_ptr            (may be NULL — skip bias add)
//   [rsp + 24] = residual_ptr        (may be NULL — skip residual add; same stride as ldc)
//   [rsp + 32] = activation          (0 = None, 1 = Relu; SiLU never routed here)
//   [rsp + 40] = is_last_k           (0 = no epilogue, just store; 1 = apply bias/residual/activation)
//
// `_set_fused` = zero accumulators.
// `_acc_fused` = load existing C into accumulators (for multi-kc accumulation).
// Both then fall through to the shared k-loop + epilogue + store.
//
// Leaf function, no pushes, 16-B stack alignment preserved throughout.

.globl CNAME(yscv_sgemm_4x24_avx2_set_fused)
#if !defined(__APPLE__)
.type yscv_sgemm_4x24_avx2_set_fused, @function
#endif
.p2align 4
CNAME(yscv_sgemm_4x24_avx2_set_fused):
    shl r9, 2                           // r9 = ldc bytes (used at store + residual)
    vxorps ymm0, ymm0, ymm0
    vxorps ymm1, ymm1, ymm1
    vxorps ymm2, ymm2, ymm2
    vxorps ymm3, ymm3, ymm3
    vxorps ymm4, ymm4, ymm4
    vxorps ymm5, ymm5, ymm5
    vxorps ymm6, ymm6, ymm6
    vxorps ymm7, ymm7, ymm7
    vxorps ymm8, ymm8, ymm8
    vxorps ymm9, ymm9, ymm9
    vxorps ymm10, ymm10, ymm10
    vxorps ymm11, ymm11, ymm11
    jmp .L4x24_kloop
#if !defined(__APPLE__)
.size yscv_sgemm_4x24_avx2_set_fused, . - yscv_sgemm_4x24_avx2_set_fused
#endif

.globl CNAME(yscv_sgemm_4x24_avx2_acc_fused)
#if !defined(__APPLE__)
.type yscv_sgemm_4x24_avx2_acc_fused, @function
#endif
.p2align 4
CNAME(yscv_sgemm_4x24_avx2_acc_fused):
    shl r9, 2                           // r9 = ldc bytes
    mov r11, r8                         // r11 = row-walking pointer
    vmovups ymm0, [r11]
    vmovups ymm1, [r11 + 32]
    vmovups ymm2, [r11 + 64]
    add r11, r9
    vmovups ymm3, [r11]
    vmovups ymm4, [r11 + 32]
    vmovups ymm5, [r11 + 64]
    add r11, r9
    vmovups ymm6, [r11]
    vmovups ymm7, [r11 + 32]
    vmovups ymm8, [r11 + 64]
    add r11, r9
    vmovups ymm9, [r11]
    vmovups ymm10, [r11 + 32]
    vmovups ymm11, [r11 + 64]
    jmp .L4x24_kloop
#if !defined(__APPLE__)
.size yscv_sgemm_4x24_avx2_acc_fused, . - yscv_sgemm_4x24_avx2_acc_fused
#endif

// Shared k-loop: 12 FMAs per iteration (3 B panels × 4 A broadcasts).
// Single-step (no 2× unroll) — only 4 YMMs free after accumulators, not
// enough headroom for pre-loading B[k+1]. Zen 4 FMA ports still saturate
// via the 12-FMA chain's intrinsic parallelism (each acc depends on itself
// across iterations, independent across accumulators).
.p2align 4
.L4x24_kloop:
    mov r10, qword ptr [rsp + 8]        // kc
    test r10, r10
    jz .L4x24_epilogue

.p2align 4
.L4x24_body:
    vmovups ymm12, [rsi]                // B0[k]  (cols 0..7)
    vmovups ymm13, [rdx]                // B1[k]  (cols 8..15)
    vmovups ymm14, [rcx]                // B2[k]  (cols 16..23)

    vbroadcastss ymm15, dword ptr [rdi] // A[k][0]
    vfmadd231ps ymm0, ymm12, ymm15
    vfmadd231ps ymm1, ymm13, ymm15
    vfmadd231ps ymm2, ymm14, ymm15

    vbroadcastss ymm15, dword ptr [rdi + 4]     // A[k][1]
    vfmadd231ps ymm3, ymm12, ymm15
    vfmadd231ps ymm4, ymm13, ymm15
    vfmadd231ps ymm5, ymm14, ymm15

    vbroadcastss ymm15, dword ptr [rdi + 8]     // A[k][2]
    vfmadd231ps ymm6, ymm12, ymm15
    vfmadd231ps ymm7, ymm13, ymm15
    vfmadd231ps ymm8, ymm14, ymm15

    vbroadcastss ymm15, dword ptr [rdi + 12]    // A[k][3]
    vfmadd231ps ymm9,  ymm12, ymm15
    vfmadd231ps ymm10, ymm13, ymm15
    vfmadd231ps ymm11, ymm14, ymm15

    add rdi, 16                         // A += MR=4 × 4 B
    add rsi, 32                         // B0 += NR=8 × 4 B
    add rdx, 32                         // B1 += NR × 4
    add rcx, 32                         // B2 += NR × 4
    dec r10
    jnz .L4x24_body

.L4x24_epilogue:
    mov r10, qword ptr [rsp + 40]       // is_last_k
    test r10, r10
    jz .L4x24_store                     // intermediate k-block — plain store

    // Bias — 3 YMMs shared across all 4 rows.
    mov r10, qword ptr [rsp + 16]       // bias_ptr
    test r10, r10
    jz .L4x24_skip_bias
    vmovups ymm12, [r10]                // bias[0..7]
    vmovups ymm13, [r10 + 32]           // bias[8..15]
    vmovups ymm14, [r10 + 64]           // bias[16..23]
    vaddps ymm0,  ymm0,  ymm12
    vaddps ymm1,  ymm1,  ymm13
    vaddps ymm2,  ymm2,  ymm14
    vaddps ymm3,  ymm3,  ymm12
    vaddps ymm4,  ymm4,  ymm13
    vaddps ymm5,  ymm5,  ymm14
    vaddps ymm6,  ymm6,  ymm12
    vaddps ymm7,  ymm7,  ymm13
    vaddps ymm8,  ymm8,  ymm14
    vaddps ymm9,  ymm9,  ymm12
    vaddps ymm10, ymm10, ymm13
    vaddps ymm11, ymm11, ymm14

.L4x24_skip_bias:
    // Residual — 12 YMMs, row-strided at ldc.
    mov r10, qword ptr [rsp + 24]       // residual_ptr
    test r10, r10
    jz .L4x24_skip_residual
    mov r11, r10
    vmovups ymm12, [r11]
    vmovups ymm13, [r11 + 32]
    vmovups ymm14, [r11 + 64]
    vaddps ymm0, ymm0, ymm12
    vaddps ymm1, ymm1, ymm13
    vaddps ymm2, ymm2, ymm14
    add r11, r9
    vmovups ymm12, [r11]
    vmovups ymm13, [r11 + 32]
    vmovups ymm14, [r11 + 64]
    vaddps ymm3, ymm3, ymm12
    vaddps ymm4, ymm4, ymm13
    vaddps ymm5, ymm5, ymm14
    add r11, r9
    vmovups ymm12, [r11]
    vmovups ymm13, [r11 + 32]
    vmovups ymm14, [r11 + 64]
    vaddps ymm6, ymm6, ymm12
    vaddps ymm7, ymm7, ymm13
    vaddps ymm8, ymm8, ymm14
    add r11, r9
    vmovups ymm12, [r11]
    vmovups ymm13, [r11 + 32]
    vmovups ymm14, [r11 + 64]
    vaddps ymm9,  ymm9,  ymm12
    vaddps ymm10, ymm10, ymm13
    vaddps ymm11, ymm11, ymm14

.L4x24_skip_residual:
    // Activation: 0 = None, 1 = Relu. SiLU never routes here.
    mov r10, qword ptr [rsp + 32]
    cmp r10, 1
    jne .L4x24_store
    vxorps ymm12, ymm12, ymm12
    vmaxps ymm0,  ymm0,  ymm12
    vmaxps ymm1,  ymm1,  ymm12
    vmaxps ymm2,  ymm2,  ymm12
    vmaxps ymm3,  ymm3,  ymm12
    vmaxps ymm4,  ymm4,  ymm12
    vmaxps ymm5,  ymm5,  ymm12
    vmaxps ymm6,  ymm6,  ymm12
    vmaxps ymm7,  ymm7,  ymm12
    vmaxps ymm8,  ymm8,  ymm12
    vmaxps ymm9,  ymm9,  ymm12
    vmaxps ymm10, ymm10, ymm12
    vmaxps ymm11, ymm11, ymm12

.L4x24_store:
    mov r11, r8
    vmovups [r11],      ymm0
    vmovups [r11 + 32], ymm1
    vmovups [r11 + 64], ymm2
    add r11, r9
    vmovups [r11],      ymm3
    vmovups [r11 + 32], ymm4
    vmovups [r11 + 64], ymm5
    add r11, r9
    vmovups [r11],      ymm6
    vmovups [r11 + 32], ymm7
    vmovups [r11 + 64], ymm8
    add r11, r9
    vmovups [r11],      ymm9
    vmovups [r11 + 32], ymm10
    vmovups [r11 + 64], ymm11
    vzeroupper
    ret

// ============================================================================
// yscv_sgemm_12x32_avx512_set / _acc  (AVX-512F, 12 rows × 32 cols)
// ============================================================================
// Step A1.1: MR=12 × NR=32 microkernel using full 32-ZMM register file.
// Rationale: Zen 4 double-pumped AVX-512 has the same FLOPS as AVX2 but
// halves front-end overhead (1 ZMM uop = 2 YMM uops of work). ORT's
// FgemmKernelAvx512F uses MR=12×NR=32 for this reason.
//
// Register plan (32 ZMM total):
//   zmm0..zmm23  = 24 accumulators (12 rows × 2 ZMM per row)
//     row r cols   0..15 → zmm[2*r]
//     row r cols  16..31 → zmm[2*r + 1]
//   zmm24, zmm25 = B[k] two ZMM (cols 0..15, 16..31)
//   zmm26        = A[row][k] scalar broadcast (cycled across 12 rows)
//   zmm27..zmm31 = scratch (bias, zero, etc. in epilogue)
//
// Pack-A layout: MR=12 stride. a_panel[p*12 + r] = A[r_offset + r, k_offset + p].
// Pack-B layout: NR=32 stride. b_panel[p*32 + c] = B[k_offset + p, c_offset + c].
//
// Calling convention (SysV AMD64):
//   rdi = a_panel        (MR=12 × kc floats, row-major within each k-panel)
//   rsi = b_panel        (kc × NR=32 floats)
//   rdx = c              (output[row][col], NHWC row-major)
//   rcx = ldc            (in FLOATS, we shl,2 → bytes)
//   r8  = kc
//   r9  = bias_ptr       (NULL → no bias)
//   [rsp + 8]  = activation (0=None, 1=Relu; SiLU routed elsewhere)
//   [rsp + 16] = is_last_k   (0/1)
//
// `_set` zeros accumulators; `_acc` loads existing C into accumulators.

.globl CNAME(yscv_sgemm_12x32_avx512_set)
#if !defined(__APPLE__)
.type yscv_sgemm_12x32_avx512_set, @function
#endif
.p2align 4
CNAME(yscv_sgemm_12x32_avx512_set):
    // Save bias_ptr (r9) to red zone — r9 is clobbered by the k-loop.
    mov qword ptr [rsp - 8], r9
    // Zero 24 accumulators.
    vpxorq zmm0,  zmm0,  zmm0
    vpxorq zmm1,  zmm1,  zmm1
    vpxorq zmm2,  zmm2,  zmm2
    vpxorq zmm3,  zmm3,  zmm3
    vpxorq zmm4,  zmm4,  zmm4
    vpxorq zmm5,  zmm5,  zmm5
    vpxorq zmm6,  zmm6,  zmm6
    vpxorq zmm7,  zmm7,  zmm7
    vpxorq zmm8,  zmm8,  zmm8
    vpxorq zmm9,  zmm9,  zmm9
    vpxorq zmm10, zmm10, zmm10
    vpxorq zmm11, zmm11, zmm11
    vpxorq zmm12, zmm12, zmm12
    vpxorq zmm13, zmm13, zmm13
    vpxorq zmm14, zmm14, zmm14
    vpxorq zmm15, zmm15, zmm15
    vpxorq zmm16, zmm16, zmm16
    vpxorq zmm17, zmm17, zmm17
    vpxorq zmm18, zmm18, zmm18
    vpxorq zmm19, zmm19, zmm19
    vpxorq zmm20, zmm20, zmm20
    vpxorq zmm21, zmm21, zmm21
    vpxorq zmm22, zmm22, zmm22
    vpxorq zmm23, zmm23, zmm23
    jmp .Lsgemm_12x32_avx512_kloop
#if !defined(__APPLE__)
.size yscv_sgemm_12x32_avx512_set, . - yscv_sgemm_12x32_avx512_set
#endif

.globl CNAME(yscv_sgemm_12x32_avx512_acc)
#if !defined(__APPLE__)
.type yscv_sgemm_12x32_avx512_acc, @function
#endif
.p2align 4
CNAME(yscv_sgemm_12x32_avx512_acc):
    mov qword ptr [rsp - 8], r9
    // Convert ldc floats → bytes.
    shl rcx, 2
    // Load c[0..11][0..31] into zmm0..zmm23 (2 ZMM per row).
    mov rax, rdx
    vmovups zmm0,  [rax]
    vmovups zmm1,  [rax + 64]
    add rax, rcx
    vmovups zmm2,  [rax]
    vmovups zmm3,  [rax + 64]
    add rax, rcx
    vmovups zmm4,  [rax]
    vmovups zmm5,  [rax + 64]
    add rax, rcx
    vmovups zmm6,  [rax]
    vmovups zmm7,  [rax + 64]
    add rax, rcx
    vmovups zmm8,  [rax]
    vmovups zmm9,  [rax + 64]
    add rax, rcx
    vmovups zmm10, [rax]
    vmovups zmm11, [rax + 64]
    add rax, rcx
    vmovups zmm12, [rax]
    vmovups zmm13, [rax + 64]
    add rax, rcx
    vmovups zmm14, [rax]
    vmovups zmm15, [rax + 64]
    add rax, rcx
    vmovups zmm16, [rax]
    vmovups zmm17, [rax + 64]
    add rax, rcx
    vmovups zmm18, [rax]
    vmovups zmm19, [rax + 64]
    add rax, rcx
    vmovups zmm20, [rax]
    vmovups zmm21, [rax + 64]
    add rax, rcx
    vmovups zmm22, [rax]
    vmovups zmm23, [rax + 64]
    shr rcx, 2                      // restore ldc in floats
    jmp .Lsgemm_12x32_avx512_kloop

.p2align 4
.Lsgemm_12x32_avx512_kloop:
    test r8, r8
    jz .Lsgemm_12x32_avx512_epilogue

.p2align 4
.Lsgemm_12x32_avx512_body:
    // Load B[k]: 32 cols = 2 ZMM.
    vmovups zmm24, [rsi]
    vmovups zmm25, [rsi + 64]

    // 12 rows × (1 broadcast + 2 FMAs). Zen 4 issues broadcasts on a
    // separate port from FMAs, so FMAs are the critical path.
    vbroadcastss zmm26, dword ptr [rdi]
    vfmadd231ps  zmm0,  zmm24, zmm26
    vfmadd231ps  zmm1,  zmm25, zmm26
    vbroadcastss zmm26, dword ptr [rdi + 4]
    vfmadd231ps  zmm2,  zmm24, zmm26
    vfmadd231ps  zmm3,  zmm25, zmm26
    vbroadcastss zmm26, dword ptr [rdi + 8]
    vfmadd231ps  zmm4,  zmm24, zmm26
    vfmadd231ps  zmm5,  zmm25, zmm26
    vbroadcastss zmm26, dword ptr [rdi + 12]
    vfmadd231ps  zmm6,  zmm24, zmm26
    vfmadd231ps  zmm7,  zmm25, zmm26
    vbroadcastss zmm26, dword ptr [rdi + 16]
    vfmadd231ps  zmm8,  zmm24, zmm26
    vfmadd231ps  zmm9,  zmm25, zmm26
    vbroadcastss zmm26, dword ptr [rdi + 20]
    vfmadd231ps  zmm10, zmm24, zmm26
    vfmadd231ps  zmm11, zmm25, zmm26
    vbroadcastss zmm26, dword ptr [rdi + 24]
    vfmadd231ps  zmm12, zmm24, zmm26
    vfmadd231ps  zmm13, zmm25, zmm26
    vbroadcastss zmm26, dword ptr [rdi + 28]
    vfmadd231ps  zmm14, zmm24, zmm26
    vfmadd231ps  zmm15, zmm25, zmm26
    vbroadcastss zmm26, dword ptr [rdi + 32]
    vfmadd231ps  zmm16, zmm24, zmm26
    vfmadd231ps  zmm17, zmm25, zmm26
    vbroadcastss zmm26, dword ptr [rdi + 36]
    vfmadd231ps  zmm18, zmm24, zmm26
    vfmadd231ps  zmm19, zmm25, zmm26
    vbroadcastss zmm26, dword ptr [rdi + 40]
    vfmadd231ps  zmm20, zmm24, zmm26
    vfmadd231ps  zmm21, zmm25, zmm26
    vbroadcastss zmm26, dword ptr [rdi + 44]
    vfmadd231ps  zmm22, zmm24, zmm26
    vfmadd231ps  zmm23, zmm25, zmm26

    add rdi, 48                     // a_panel += 12 floats
    add rsi, 128                    // b_panel += 32 floats
    dec r8
    jnz .Lsgemm_12x32_avx512_body

.Lsgemm_12x32_avx512_epilogue:
    mov r8, qword ptr [rsp + 16]    // is_last_k
    test r8, r8
    jz .Lsgemm_12x32_avx512_store

    // Bias add.
    mov r9, qword ptr [rsp - 8]     // bias_ptr
    test r9, r9
    jz .Lsgemm_12x32_avx512_skip_bias
    vmovups zmm27, [r9]             // bias[0..15]
    vmovups zmm28, [r9 + 64]        // bias[16..31]
    vaddps zmm0,  zmm0,  zmm27
    vaddps zmm1,  zmm1,  zmm28
    vaddps zmm2,  zmm2,  zmm27
    vaddps zmm3,  zmm3,  zmm28
    vaddps zmm4,  zmm4,  zmm27
    vaddps zmm5,  zmm5,  zmm28
    vaddps zmm6,  zmm6,  zmm27
    vaddps zmm7,  zmm7,  zmm28
    vaddps zmm8,  zmm8,  zmm27
    vaddps zmm9,  zmm9,  zmm28
    vaddps zmm10, zmm10, zmm27
    vaddps zmm11, zmm11, zmm28
    vaddps zmm12, zmm12, zmm27
    vaddps zmm13, zmm13, zmm28
    vaddps zmm14, zmm14, zmm27
    vaddps zmm15, zmm15, zmm28
    vaddps zmm16, zmm16, zmm27
    vaddps zmm17, zmm17, zmm28
    vaddps zmm18, zmm18, zmm27
    vaddps zmm19, zmm19, zmm28
    vaddps zmm20, zmm20, zmm27
    vaddps zmm21, zmm21, zmm28
    vaddps zmm22, zmm22, zmm27
    vaddps zmm23, zmm23, zmm28

.Lsgemm_12x32_avx512_skip_bias:
    // Activation (1 = Relu; 0 = None; SiLU never reaches here).
    mov r8, qword ptr [rsp + 8]
    cmp r8, 1
    jne .Lsgemm_12x32_avx512_store
    vpxorq zmm29, zmm29, zmm29
    vmaxps zmm0,  zmm0,  zmm29
    vmaxps zmm1,  zmm1,  zmm29
    vmaxps zmm2,  zmm2,  zmm29
    vmaxps zmm3,  zmm3,  zmm29
    vmaxps zmm4,  zmm4,  zmm29
    vmaxps zmm5,  zmm5,  zmm29
    vmaxps zmm6,  zmm6,  zmm29
    vmaxps zmm7,  zmm7,  zmm29
    vmaxps zmm8,  zmm8,  zmm29
    vmaxps zmm9,  zmm9,  zmm29
    vmaxps zmm10, zmm10, zmm29
    vmaxps zmm11, zmm11, zmm29
    vmaxps zmm12, zmm12, zmm29
    vmaxps zmm13, zmm13, zmm29
    vmaxps zmm14, zmm14, zmm29
    vmaxps zmm15, zmm15, zmm29
    vmaxps zmm16, zmm16, zmm29
    vmaxps zmm17, zmm17, zmm29
    vmaxps zmm18, zmm18, zmm29
    vmaxps zmm19, zmm19, zmm29
    vmaxps zmm20, zmm20, zmm29
    vmaxps zmm21, zmm21, zmm29
    vmaxps zmm22, zmm22, zmm29
    vmaxps zmm23, zmm23, zmm29

.Lsgemm_12x32_avx512_store:
    shl rcx, 2                      // ldc → bytes
    mov rax, rdx
    vmovups [rax],      zmm0
    vmovups [rax + 64], zmm1
    add rax, rcx
    vmovups [rax],      zmm2
    vmovups [rax + 64], zmm3
    add rax, rcx
    vmovups [rax],      zmm4
    vmovups [rax + 64], zmm5
    add rax, rcx
    vmovups [rax],      zmm6
    vmovups [rax + 64], zmm7
    add rax, rcx
    vmovups [rax],      zmm8
    vmovups [rax + 64], zmm9
    add rax, rcx
    vmovups [rax],      zmm10
    vmovups [rax + 64], zmm11
    add rax, rcx
    vmovups [rax],      zmm12
    vmovups [rax + 64], zmm13
    add rax, rcx
    vmovups [rax],      zmm14
    vmovups [rax + 64], zmm15
    add rax, rcx
    vmovups [rax],      zmm16
    vmovups [rax + 64], zmm17
    add rax, rcx
    vmovups [rax],      zmm18
    vmovups [rax + 64], zmm19
    add rax, rcx
    vmovups [rax],      zmm20
    vmovups [rax + 64], zmm21
    add rax, rcx
    vmovups [rax],      zmm22
    vmovups [rax + 64], zmm23
    vzeroupper
    ret
#if !defined(__APPLE__)
.size yscv_sgemm_12x32_avx512_acc, . - yscv_sgemm_12x32_avx512_acc
#endif

YSCV_NOTE_GNU_STACK
