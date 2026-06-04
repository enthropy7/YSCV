// yscv SGEMM microkernels — aarch64 AAPCS64 (Linux, macOS, Android, iOS).
//
// AAPCS64 calling convention:
//   Args:      x0, x1, x2, x3, x4, x5, x6, x7 (first 8 in GPRs).
//   Callee-saved GPR: x19–x29 (frame pointer = x29, LR = x30).
//   Callee-saved FP:  v8–v15, **low 64 bits only** (we treat as d8–d15).
//
// Our 4×24 kernel uses v0–v23 as 24 accumulators, so v8–v15 need to be
// saved on entry and restored before return. `stp d{n}, d{n+1}, [sp, …]`
// saves the low 64 bits; the high 64 bits of v8–v15 are caller-saved
// and we're free to clobber them.

#include "asm_common.h"

// ============================================================================
// yscv_sgemm_4x24_neon_set / yscv_sgemm_4x24_neon_acc  (4 rows × 24 cols)
// ============================================================================
// Pack layout matches the Rust-side `pack_a_panel` / `pack_b_panel`:
//   A — MR=4 contiguous per k (4 floats, 16 bytes) then advance panel.
//   B — three NR=8 panels (b0, b1, b2), each 8 floats per k (32 bytes).
//
// Register schedule:
//   v0–v23  = 24 accumulators (4 rows × 6 quads, row-major).
//             row r, tile t  →  v[r*6 + t].
//   v24–v29 = B[k] loaded: v24..v25 from b0, v26..v27 from b1, v28..v29 from b2.
//   v30     = A[k] broadcast: all 4 MR values in one v-reg lane-wise.
//             v30.s[0..3] = A[k][0..3], consumed via `fmla v.4s, v.4s, v30.s[i]`.
//   v31     = spare (used as transient for aligned store pairs).
//
// Args:
//   x0  a_panel
//   x1  b_panel_0
//   x2  b_panel_1
//   x3  b_panel_2
//   x4  c
//   x5  ldc (in FLOATS; converted to bytes in prologue)
//   x6  kc
//
// k-loop: 24 `fmla v.4s, v.4s, v30.s[i]` per step — all 24 accumulator
// chains are independent, so the two NEON FMA pipes (A76, N1, Apple M)
// issue back-to-back without RAW stalls. Expected throughput: 12 cycles
// per k-step on a 2-FMA-pipe core.

.globl CNAME(yscv_sgemm_4x24_neon_set)
#if !defined(__APPLE__)
.type yscv_sgemm_4x24_neon_set, %function
#endif
.p2align 2
CNAME(yscv_sgemm_4x24_neon_set):
    // Save callee-saved v8–v15 (low 64 bits).
    stp d8,  d9,  [sp, #-64]!
    stp d10, d11, [sp, #16]
    stp d12, d13, [sp, #32]
    stp d14, d15, [sp, #48]

    // ldc floats → bytes.
    lsl x5, x5, #2

    // Row pointers for the 4 output rows.
    mov x7,  x4
    add x8,  x4, x5
    add x9,  x8, x5
    add x10, x9, x5

    // Zero all 24 accumulators.
    movi v0.4s,  #0
    movi v1.4s,  #0
    movi v2.4s,  #0
    movi v3.4s,  #0
    movi v4.4s,  #0
    movi v5.4s,  #0
    movi v6.4s,  #0
    movi v7.4s,  #0
    movi v8.4s,  #0
    movi v9.4s,  #0
    movi v10.4s, #0
    movi v11.4s, #0
    movi v12.4s, #0
    movi v13.4s, #0
    movi v14.4s, #0
    movi v15.4s, #0
    movi v16.4s, #0
    movi v17.4s, #0
    movi v18.4s, #0
    movi v19.4s, #0
    movi v20.4s, #0
    movi v21.4s, #0
    movi v22.4s, #0
    movi v23.4s, #0
    b .Lsgemm_4x24_kloop
#if !defined(__APPLE__)
.size yscv_sgemm_4x24_neon_set, . - yscv_sgemm_4x24_neon_set
#endif

.globl CNAME(yscv_sgemm_4x24_neon_acc)
#if !defined(__APPLE__)
.type yscv_sgemm_4x24_neon_acc, %function
#endif
.p2align 2
CNAME(yscv_sgemm_4x24_neon_acc):
    stp d8,  d9,  [sp, #-64]!
    stp d10, d11, [sp, #16]
    stp d12, d13, [sp, #32]
    stp d14, d15, [sp, #48]

    lsl x5, x5, #2
    mov x7,  x4
    add x8,  x4, x5
    add x9,  x8, x5
    add x10, x9, x5

    // Load existing c into accumulators (6 × 16 bytes per row).
    ldp q0,  q1,  [x7]
    ldp q2,  q3,  [x7, #32]
    ldp q4,  q5,  [x7, #64]
    ldp q6,  q7,  [x8]
    ldp q8,  q9,  [x8, #32]
    ldp q10, q11, [x8, #64]
    ldp q12, q13, [x9]
    ldp q14, q15, [x9, #32]
    ldp q16, q17, [x9, #64]
    ldp q18, q19, [x10]
    ldp q20, q21, [x10, #32]
    ldp q22, q23, [x10, #64]
    b .Lsgemm_4x24_kloop
#if !defined(__APPLE__)
.size yscv_sgemm_4x24_neon_acc, . - yscv_sgemm_4x24_neon_acc
#endif

// Shared k-loop and store tail.
.p2align 4
.Lsgemm_4x24_kloop:
    cbz x6, .Lsgemm_4x24_store

.p2align 4
.Lsgemm_4x24_kbody:
    // Load B[k] — 6 quadwords total, three panels × two NR/4 slices.
    ld1 {v24.4s, v25.4s}, [x1]
    ld1 {v26.4s, v27.4s}, [x2]
    ld1 {v28.4s, v29.4s}, [x3]
    // Load A[k] — 4 floats → one quadword with MR values lane-wise.
    ld1 {v30.4s}, [x0]

    // Row 0 (a0) — 6 FMAs
    fmla v0.4s,  v24.4s, v30.s[0]
    fmla v1.4s,  v25.4s, v30.s[0]
    fmla v2.4s,  v26.4s, v30.s[0]
    fmla v3.4s,  v27.4s, v30.s[0]
    fmla v4.4s,  v28.4s, v30.s[0]
    fmla v5.4s,  v29.4s, v30.s[0]
    // Row 1 (a1)
    fmla v6.4s,  v24.4s, v30.s[1]
    fmla v7.4s,  v25.4s, v30.s[1]
    fmla v8.4s,  v26.4s, v30.s[1]
    fmla v9.4s,  v27.4s, v30.s[1]
    fmla v10.4s, v28.4s, v30.s[1]
    fmla v11.4s, v29.4s, v30.s[1]
    // Row 2 (a2)
    fmla v12.4s, v24.4s, v30.s[2]
    fmla v13.4s, v25.4s, v30.s[2]
    fmla v14.4s, v26.4s, v30.s[2]
    fmla v15.4s, v27.4s, v30.s[2]
    fmla v16.4s, v28.4s, v30.s[2]
    fmla v17.4s, v29.4s, v30.s[2]
    // Row 3 (a3)
    fmla v18.4s, v24.4s, v30.s[3]
    fmla v19.4s, v25.4s, v30.s[3]
    fmla v20.4s, v26.4s, v30.s[3]
    fmla v21.4s, v27.4s, v30.s[3]
    fmla v22.4s, v28.4s, v30.s[3]
    fmla v23.4s, v29.4s, v30.s[3]

    add x0, x0, #16                 // a += MR floats
    add x1, x1, #32                 // b0 += NR floats
    add x2, x2, #32
    add x3, x3, #32
    subs x6, x6, #1
    b.ne .Lsgemm_4x24_kbody

.Lsgemm_4x24_store:
    // Store 6 quadwords per row.
    stp q0,  q1,  [x7]
    stp q2,  q3,  [x7, #32]
    stp q4,  q5,  [x7, #64]
    stp q6,  q7,  [x8]
    stp q8,  q9,  [x8, #32]
    stp q10, q11, [x8, #64]
    stp q12, q13, [x9]
    stp q14, q15, [x9, #32]
    stp q16, q17, [x9, #64]
    stp q18, q19, [x10]
    stp q20, q21, [x10, #32]
    stp q22, q23, [x10, #64]

    // Restore callee-saved d8–d15.
    ldp d14, d15, [sp, #48]
    ldp d12, d13, [sp, #32]
    ldp d10, d11, [sp, #16]
    ldp d8,  d9,  [sp], #64
    ret

// ============================================================================
// yscv_sgemm_8x12_neon_set / _acc  (NEON, 8 rows × 12 cols, inline epilogue)
// ============================================================================
// MR=8, NR=12 tile. 24 accumulators (v0..v23) hold 8 rows × 3 v.4s quads
// (12 cols). 2× A-broadcast amortization vs MR=4: each pair of A values
// drives 6 FMAs before the next broadcast arrives. Per k-step: 24 FMAs
// on Cortex-A76's 2 FMA pipes = 12 cycles backend.
//
// **B-PACK CONTRACT:** NR=12 packed layout (one panel = kc rows × 12 f32
// contiguous per k-step). NOT the generic NR=8 packer — MR=8 has its own
// `pack_b_panel_nr12` (see Phase 0.3 in matmul.rs). Phase 0.3 rewrite
// kept the 8×12 kernel shape intact; the ONLY change vs the previous
// broken version is that the kernel now assumes NR=12-aligned panels and
// the caller advances `jr += 12` on panels that match this stride. The
// earlier implementation called this kernel over NR=8-packed memory with
// `jr += 12`, reading the wrong cols after the first iteration.
//
// Register plan (29 of 32 v-regs):
//   v0..v23  = 24 accumulators (row r, tile t → v[r*3 + t]; t=0 cols 0..3,
//              t=1 cols 4..7, t=2 cols 8..11).
//   v24..v26 = B[k] cols 0..11 (3 quads from the single NR=12 packed panel)
//   v27..v28 = A[k] 8 floats packed in 2 × v.4s (lanes 0..3 each)
//
// Callee-saved per AAPCS64: v8-v15 low 64 bits. Save/restore at entry/exit.
//
// Args (AAPCS64):
//   x0 = a_panel (packed MR=8 × kc f32, 32 bytes per k)
//   x1 = b_panel (packed kc × NR=12, 48 bytes per k — single panel!)
//   x2 = c (output)
//   x3 = ldc (FLOATS, converted to bytes in prologue)
//   x4 = kc
//   x5 = bias ptr (0 = no bias, 12 floats when set)
//   x6 = activation id (0=None, 1=Relu); SiLU routed to MR=4 by caller
//   x7 = is_last_k (0 or 1)

    .globl CNAME(yscv_sgemm_8x12_neon_set)
#if !defined(__APPLE__)
    .type yscv_sgemm_8x12_neon_set, %function
    .p2align 2
yscv_sgemm_8x12_neon_set:
#else
    .p2align 2
_yscv_sgemm_8x12_neon_set:
#endif
    stp d8,  d9,  [sp, #-64]!
    stp d10, d11, [sp, #16]
    stp d12, d13, [sp, #32]
    stp d14, d15, [sp, #48]

    lsl x3, x3, #2                      // ldc floats → bytes

    // Row pointers c0..c7 via ldc stride.
    mov x8,  x2                         // row 0
    add x9,  x2, x3                     // row 1
    add x10, x9, x3                     // row 2
    add x11, x10, x3                    // row 3
    add x12, x11, x3                    // row 4
    add x13, x12, x3                    // row 5
    add x14, x13, x3                    // row 6
    add x15, x14, x3                    // row 7

    // Zero 24 accumulators.
    movi v0.4s,  #0
    movi v1.4s,  #0
    movi v2.4s,  #0
    movi v3.4s,  #0
    movi v4.4s,  #0
    movi v5.4s,  #0
    movi v6.4s,  #0
    movi v7.4s,  #0
    movi v8.4s,  #0
    movi v9.4s,  #0
    movi v10.4s, #0
    movi v11.4s, #0
    movi v12.4s, #0
    movi v13.4s, #0
    movi v14.4s, #0
    movi v15.4s, #0
    movi v16.4s, #0
    movi v17.4s, #0
    movi v18.4s, #0
    movi v19.4s, #0
    movi v20.4s, #0
    movi v21.4s, #0
    movi v22.4s, #0
    movi v23.4s, #0
    b .Lsgemm_8x12_kloop
#if !defined(__APPLE__)
    .size yscv_sgemm_8x12_neon_set, . - yscv_sgemm_8x12_neon_set
#endif

    .globl CNAME(yscv_sgemm_8x12_neon_acc)
#if !defined(__APPLE__)
    .type yscv_sgemm_8x12_neon_acc, %function
    .p2align 2
yscv_sgemm_8x12_neon_acc:
#else
    .p2align 2
_yscv_sgemm_8x12_neon_acc:
#endif
    stp d8,  d9,  [sp, #-64]!
    stp d10, d11, [sp, #16]
    stp d12, d13, [sp, #32]
    stp d14, d15, [sp, #48]

    lsl x3, x3, #2                      // ldc → bytes

    mov x8,  x2
    add x9,  x2, x3
    add x10, x9, x3
    add x11, x10, x3
    add x12, x11, x3
    add x13, x12, x3
    add x14, x13, x3
    add x15, x14, x3

    // Load existing C: 12 f32 per row = 3 quads.
    ld1 {v0.4s,  v1.4s,  v2.4s},  [x8]
    ld1 {v3.4s,  v4.4s,  v5.4s},  [x9]
    ld1 {v6.4s,  v7.4s,  v8.4s},  [x10]
    ld1 {v9.4s,  v10.4s, v11.4s}, [x11]
    ld1 {v12.4s, v13.4s, v14.4s}, [x12]
    ld1 {v15.4s, v16.4s, v17.4s}, [x13]
    ld1 {v18.4s, v19.4s, v20.4s}, [x14]
    ld1 {v21.4s, v22.4s, v23.4s}, [x15]
    b .Lsgemm_8x12_kloop

.p2align 4
.Lsgemm_8x12_kloop:
    cbz x4, .Lsgemm_8x12_epilogue

.p2align 4
.Lsgemm_8x12_body:
    // Load B[k]: 3 quads (12 cols) from the single NR=12 packed panel.
    ld1 {v24.4s, v25.4s, v26.4s}, [x1]  // b[k][0..11]
    // Load A[k]: 8 floats packed as 2 × v.4s.
    ld1 {v27.4s, v28.4s}, [x0]          // A[k][0..7]

    // 24 FMAs: each row r in 0..8 gets 3 FMAs (one per B quad).
    fmla v0.4s,  v24.4s, v27.s[0]       // row 0, cols 0..3
    fmla v1.4s,  v25.4s, v27.s[0]       // row 0, cols 4..7
    fmla v2.4s,  v26.4s, v27.s[0]       // row 0, cols 8..11
    fmla v3.4s,  v24.4s, v27.s[1]
    fmla v4.4s,  v25.4s, v27.s[1]
    fmla v5.4s,  v26.4s, v27.s[1]
    fmla v6.4s,  v24.4s, v27.s[2]
    fmla v7.4s,  v25.4s, v27.s[2]
    fmla v8.4s,  v26.4s, v27.s[2]
    fmla v9.4s,  v24.4s, v27.s[3]
    fmla v10.4s, v25.4s, v27.s[3]
    fmla v11.4s, v26.4s, v27.s[3]
    fmla v12.4s, v24.4s, v28.s[0]
    fmla v13.4s, v25.4s, v28.s[0]
    fmla v14.4s, v26.4s, v28.s[0]
    fmla v15.4s, v24.4s, v28.s[1]
    fmla v16.4s, v25.4s, v28.s[1]
    fmla v17.4s, v26.4s, v28.s[1]
    fmla v18.4s, v24.4s, v28.s[2]
    fmla v19.4s, v25.4s, v28.s[2]
    fmla v20.4s, v26.4s, v28.s[2]
    fmla v21.4s, v24.4s, v28.s[3]
    fmla v22.4s, v25.4s, v28.s[3]
    fmla v23.4s, v26.4s, v28.s[3]

    add x0, x0, #32                     // a += MR=8 floats (32 bytes)
    add x1, x1, #48                     // b += NR=12 floats (48 bytes)
    subs x4, x4, #1
    b.ne .Lsgemm_8x12_body

.Lsgemm_8x12_epilogue:
    // Apply inline epilogue only when is_last_k == 1.
    cbz x7, .Lsgemm_8x12_store

    // Bias add (if x5 != null). Caller pre-advances bias by col_offset
    // so bias[0..11] maps this tile's 12 output columns.
    cbz x5, .Lsgemm_8x12_skip_bias
    ld1 {v24.4s, v25.4s, v26.4s}, [x5]  // bias[0..11]
    fadd v0.4s,  v0.4s,  v24.4s
    fadd v1.4s,  v1.4s,  v25.4s
    fadd v2.4s,  v2.4s,  v26.4s
    fadd v3.4s,  v3.4s,  v24.4s
    fadd v4.4s,  v4.4s,  v25.4s
    fadd v5.4s,  v5.4s,  v26.4s
    fadd v6.4s,  v6.4s,  v24.4s
    fadd v7.4s,  v7.4s,  v25.4s
    fadd v8.4s,  v8.4s,  v26.4s
    fadd v9.4s,  v9.4s,  v24.4s
    fadd v10.4s, v10.4s, v25.4s
    fadd v11.4s, v11.4s, v26.4s
    fadd v12.4s, v12.4s, v24.4s
    fadd v13.4s, v13.4s, v25.4s
    fadd v14.4s, v14.4s, v26.4s
    fadd v15.4s, v15.4s, v24.4s
    fadd v16.4s, v16.4s, v25.4s
    fadd v17.4s, v17.4s, v26.4s
    fadd v18.4s, v18.4s, v24.4s
    fadd v19.4s, v19.4s, v25.4s
    fadd v20.4s, v20.4s, v26.4s
    fadd v21.4s, v21.4s, v24.4s
    fadd v22.4s, v22.4s, v25.4s
    fadd v23.4s, v23.4s, v26.4s

.Lsgemm_8x12_skip_bias:
    // Activation: id 1 = Relu. Id 0 = skip. SiLU routed to MR=4 by caller.
    cmp x6, #1
    b.ne .Lsgemm_8x12_store
    movi v24.4s, #0
    fmax v0.4s,  v0.4s,  v24.4s
    fmax v1.4s,  v1.4s,  v24.4s
    fmax v2.4s,  v2.4s,  v24.4s
    fmax v3.4s,  v3.4s,  v24.4s
    fmax v4.4s,  v4.4s,  v24.4s
    fmax v5.4s,  v5.4s,  v24.4s
    fmax v6.4s,  v6.4s,  v24.4s
    fmax v7.4s,  v7.4s,  v24.4s
    fmax v8.4s,  v8.4s,  v24.4s
    fmax v9.4s,  v9.4s,  v24.4s
    fmax v10.4s, v10.4s, v24.4s
    fmax v11.4s, v11.4s, v24.4s
    fmax v12.4s, v12.4s, v24.4s
    fmax v13.4s, v13.4s, v24.4s
    fmax v14.4s, v14.4s, v24.4s
    fmax v15.4s, v15.4s, v24.4s
    fmax v16.4s, v16.4s, v24.4s
    fmax v17.4s, v17.4s, v24.4s
    fmax v18.4s, v18.4s, v24.4s
    fmax v19.4s, v19.4s, v24.4s
    fmax v20.4s, v20.4s, v24.4s
    fmax v21.4s, v21.4s, v24.4s
    fmax v22.4s, v22.4s, v24.4s
    fmax v23.4s, v23.4s, v24.4s

.Lsgemm_8x12_store:
    st1 {v0.4s,  v1.4s,  v2.4s},  [x8]
    st1 {v3.4s,  v4.4s,  v5.4s},  [x9]
    st1 {v6.4s,  v7.4s,  v8.4s},  [x10]
    st1 {v9.4s,  v10.4s, v11.4s}, [x11]
    st1 {v12.4s, v13.4s, v14.4s}, [x12]
    st1 {v15.4s, v16.4s, v17.4s}, [x13]
    st1 {v18.4s, v19.4s, v20.4s}, [x14]
    st1 {v21.4s, v22.4s, v23.4s}, [x15]

    ldp d14, d15, [sp, #48]
    ldp d12, d13, [sp, #32]
    ldp d10, d11, [sp, #16]
    ldp d8,  d9,  [sp], #64
    ret
#if !defined(__APPLE__)
    .size yscv_sgemm_8x12_neon_acc, . - yscv_sgemm_8x12_neon_acc
#endif

// Enable FEAT_FP16 (fullfp16) for the hgemm kernel below so the assembler
// accepts `fmla v.8h, v.8h, v.h[i]` (fp16 SIMD FMA).
#ifdef __APPLE__
    .arch_extension fp16
#else
    .arch armv8.2-a+fp16
#endif

// ============================================================================
// yscv_hgemm_6x16_neon_set / _acc  (FEAT_FP16, 6 rows × 16 cols, fp16)
// ============================================================================
// Half-precision SGEMM using FEAT_FP16 (ARMv8.2-A): `fmla v.8h, v.8h, v.h[i]`
// runs 8-wide fp16 FMA — 2× throughput vs fp32 `fmla v.4s` on any core that
// supports FEAT_FP16 (Cortex-A76+, Apple M1+, Neoverse-V1+).
//
// Layout matches MLAS HalfGemmKernelNeon.S (MR=6, NR=16):
//   A pack: 6 fp16 per k-step (packed into v.8h lanes 0..5, 6..7 padded zero).
//   B pack: two adjacent NR=8 fp16 panels = 16 fp16 per k-step.
//   C:      6 rows × 16 cols = 12 v.8h regs (6 rows × 2 regs per row).
//
// Register plan (32 v-regs total):
//   v0..v11  = 12 accumulators (row r, cols [0..7]=v[2r], cols [8..15]=v[2r+1])
//   v12, v13 = B[k] cols 0..7 and 8..15
//   v14      = A[k] scattered (lanes 0..5 hold A[0..5], 6..7 unused/zeros)
//   v15..v31 = spare / callee-saved region for bias/epilogue
//
// Callee-saved per AAPCS64: v8-v15 low 64 bits. v8-v11 are accumulators
// (we use the full v-reg, but the low half must still survive the call —
// save d8..d11 at entry).
//
// Args:
//   x0 = a_panel  (*const u16 — fp16 bit-packed)
//   x1 = b_panel_0 (*const u16)
//   x2 = b_panel_1 (*const u16)
//   x3 = c (*mut u16)
//   x4 = ldc (elements, multiplied by 2 internally for bytes)
//   x5 = kc

    .globl CNAME(yscv_hgemm_6x16_neon_set)
#if !defined(__APPLE__)
    .type yscv_hgemm_6x16_neon_set, %function
    .p2align 2
yscv_hgemm_6x16_neon_set:
#else
    .p2align 2
_yscv_hgemm_6x16_neon_set:
#endif
    stp d8,  d9,  [sp, #-32]!
    stp d10, d11, [sp, #16]

    lsl x4, x4, #1                      // ldc elems → bytes (fp16 = 2B each)

    mov x6,  x3
    add x7,  x3, x4
    add x8,  x7, x4
    add x9,  x8, x4
    add x10, x9, x4
    add x11, x10, x4

    movi v0.8h,  #0
    movi v1.8h,  #0
    movi v2.8h,  #0
    movi v3.8h,  #0
    movi v4.8h,  #0
    movi v5.8h,  #0
    movi v6.8h,  #0
    movi v7.8h,  #0
    movi v8.8h,  #0
    movi v9.8h,  #0
    movi v10.8h, #0
    movi v11.8h, #0
    b .Lhgemm_6x16_kloop
#if !defined(__APPLE__)
    .size yscv_hgemm_6x16_neon_set, . - yscv_hgemm_6x16_neon_set
#endif

    .globl CNAME(yscv_hgemm_6x16_neon_acc)
#if !defined(__APPLE__)
    .type yscv_hgemm_6x16_neon_acc, %function
    .p2align 2
yscv_hgemm_6x16_neon_acc:
#else
    .p2align 2
_yscv_hgemm_6x16_neon_acc:
#endif
    stp d8,  d9,  [sp, #-32]!
    stp d10, d11, [sp, #16]

    lsl x4, x4, #1                      // ldc bytes

    mov x6,  x3
    add x7,  x3, x4
    add x8,  x7, x4
    add x9,  x8, x4
    add x10, x9, x4
    add x11, x10, x4

    // Load existing C into accumulators: 16 fp16 per row = 2 × v.8h.
    ld1 {v0.8h,  v1.8h},  [x6]
    ld1 {v2.8h,  v3.8h},  [x7]
    ld1 {v4.8h,  v5.8h},  [x8]
    ld1 {v6.8h,  v7.8h},  [x9]
    ld1 {v8.8h,  v9.8h},  [x10]
    ld1 {v10.8h, v11.8h}, [x11]
    b .Lhgemm_6x16_kloop

.p2align 4
.Lhgemm_6x16_kloop:
    cbz x5, .Lhgemm_6x16_store

.p2align 4
.Lhgemm_6x16_body:
    // Load B[k]: 16 fp16 = 2 v.8h.
    ld1 {v12.8h, v13.8h}, [x1]          // cols 0..15 — pack interleaves two NR=8 panels
    // Load A[k]: 8 fp16 (6 used + 2 pad).
    ld1 {v14.8h}, [x0]

    // 12 FMAs per k-step.
    fmla v0.8h,  v12.8h, v14.h[0]
    fmla v1.8h,  v13.8h, v14.h[0]
    fmla v2.8h,  v12.8h, v14.h[1]
    fmla v3.8h,  v13.8h, v14.h[1]
    fmla v4.8h,  v12.8h, v14.h[2]
    fmla v5.8h,  v13.8h, v14.h[2]
    fmla v6.8h,  v12.8h, v14.h[3]
    fmla v7.8h,  v13.8h, v14.h[3]
    fmla v8.8h,  v12.8h, v14.h[4]
    fmla v9.8h,  v13.8h, v14.h[4]
    fmla v10.8h, v12.8h, v14.h[5]
    fmla v11.8h, v13.8h, v14.h[5]

    add x0, x0, #16                     // a += 8 fp16 (but only 6 used; stride 8 keeps alignment)
    add x1, x1, #32                     // b += 16 fp16
    subs x5, x5, #1
    b.ne .Lhgemm_6x16_body

.Lhgemm_6x16_store:
    st1 {v0.8h,  v1.8h},  [x6]
    st1 {v2.8h,  v3.8h},  [x7]
    st1 {v4.8h,  v5.8h},  [x8]
    st1 {v6.8h,  v7.8h},  [x9]
    st1 {v8.8h,  v9.8h},  [x10]
    st1 {v10.8h, v11.8h}, [x11]

    ldp d10, d11, [sp, #16]
    ldp d8,  d9,  [sp], #32
    ret
#if !defined(__APPLE__)
    .size yscv_hgemm_6x16_neon_acc, . - yscv_hgemm_6x16_neon_acc
#endif

// ============================================================================
// yscv_dw5_creuse_neon  —  5x5 depthwise, 4 output columns x 8 channels,
//                          interior (no bounds), stride 1, column-reuse.
// ============================================================================
// The intrinsic DW re-reads each PW-ring input column once per (kx, output
// column) — 2:1 load:FMA, and on the in-order A53 each L2 ring read stalls the
// FMA pipe. Here we load the 8 input columns a 4-output-column tile touches
// ONCE per kernel-row (held in v8..v23) and reuse them across all 5 kx taps and
// 4 output columns — input loads drop ~2.5x, killing the per-tap stall.
//
// Register plan (v8..v15 callee-saved low 64 -> stp/ldp):
//   v0..v7   = 8 accumulators: output col p -> {v[p*2], v[p*2+1]} (8 ch = 2 q).
//   v8..v23  = the kernel-row's 8 input columns (col c -> v[8+c*2], v[8+c*2+1]).
//   v24,v25  = current tap weight (2 q = 8 ch).
//   v26,v27  = bias / relu-zero scratch.
//
// Args (AAPCS64):
//   x0  rows   = *const [*const f32; 5], each pre-offset to input column
//               (ow_tile - pad) and channel ch. Input(ky,kx,col p) =
//               rows[ky] + (kx + p) * c_exp floats.
//   x1  weight = dw_weight + ch (kernel walks +c_exp per tap; 25 contiguous).
//   x2  out    = out_row + ow_tile*c_exp + ch (kernel adds p*c_exp per col).
//   x3  c_exp  = channel stride in FLOATS.
//   x4  bias   = bias + ch, or 0.
//   x5  relu   = 0 or 1.
//   x6  wstride = weight tap stride in BYTES. c_exp*4 for the natural
//                 KH·KW·C layout; 32 (= 8 ch × 4 B) for a pre-packed
//                 [25][8] tile (contiguous taps → L1-resident weights).

    .globl CNAME(yscv_dw5_creuse_neon)
#if !defined(__APPLE__)
    .type yscv_dw5_creuse_neon, %function
    .p2align 2
yscv_dw5_creuse_neon:
#else
    .p2align 2
_yscv_dw5_creuse_neon:
#endif
    stp d8,  d9,  [sp, #-64]!
    stp d10, d11, [sp, #16]
    stp d12, d13, [sp, #32]
    stp d14, d15, [sp, #48]

    mov x15, x6                         // weight tap stride (bytes)
    ldr x6,  [x0]                       // rows[0..4]
    ldr x7,  [x0, #8]
    ldr x8,  [x0, #16]
    ldr x9,  [x0, #24]
    ldr x10, [x0, #32]
    lsl x11, x3, #2                     // c_exp bytes
    mov x14, x1                         // weight cursor (25 taps, +c_exp each)

    // Init the 8 accumulators from bias (broadcast to all 4 output cols) or 0.
    cbz x4, .Ldw5_zero
    ld1 {v26.4s, v27.4s}, [x4]
    mov v0.16b, v26.16b
    mov v1.16b, v27.16b
    mov v2.16b, v26.16b
    mov v3.16b, v27.16b
    mov v4.16b, v26.16b
    mov v5.16b, v27.16b
    mov v6.16b, v26.16b
    mov v7.16b, v27.16b
    b .Ldw5_ky
.Ldw5_zero:
    movi v0.4s, #0
    movi v1.4s, #0
    movi v2.4s, #0
    movi v3.4s, #0
    movi v4.4s, #0
    movi v5.4s, #0
    movi v6.4s, #0
    movi v7.4s, #0

.Ldw5_ky:
    // One macro body per kernel row; the row pointer is the only difference.
    // KROW xR: load 8 input cols, then 5 kx taps over the shared columns.
.macro KROW row, nextrow
    mov x13, \row
    mov x16, \nextrow                   // prefetch cursor: next row's columns
    ld1 {v8.4s,  v9.4s},  [x13], x11
    ld1 {v10.4s, v11.4s}, [x13], x11
    ld1 {v12.4s, v13.4s}, [x13], x11
    ld1 {v14.4s, v15.4s}, [x13], x11
    ld1 {v16.4s, v17.4s}, [x13], x11
    ld1 {v18.4s, v19.4s}, [x13], x11
    ld1 {v20.4s, v21.4s}, [x13], x11
    ld1 {v22.4s, v23.4s}, [x13]
    // kx=0: cols 0,1,2,3. Prefetch the next row's first 2 columns while this
    // tap's FMAs run — the LS pipe is idle under FMA on the in-order A53, so
    // the L2→L1 fetch hides behind the 40-FMA kernel-row body.
    ld1 {v24.4s, v25.4s}, [x14], x15
    prfm pldl1keep, [x16]
    add x16, x16, x11
    fmla v0.4s, v8.4s,  v24.4s
    fmla v1.4s, v9.4s,  v25.4s
    prfm pldl1keep, [x16]
    add x16, x16, x11
    fmla v2.4s, v10.4s, v24.4s
    fmla v3.4s, v11.4s, v25.4s
    fmla v4.4s, v12.4s, v24.4s
    fmla v5.4s, v13.4s, v25.4s
    fmla v6.4s, v14.4s, v24.4s
    fmla v7.4s, v15.4s, v25.4s
    // kx=1: cols 1,2,3,4
    ld1 {v24.4s, v25.4s}, [x14], x15
    prfm pldl1keep, [x16]
    add x16, x16, x11
    fmla v0.4s, v10.4s, v24.4s
    fmla v1.4s, v11.4s, v25.4s
    prfm pldl1keep, [x16]
    add x16, x16, x11
    fmla v2.4s, v12.4s, v24.4s
    fmla v3.4s, v13.4s, v25.4s
    fmla v4.4s, v14.4s, v24.4s
    fmla v5.4s, v15.4s, v25.4s
    fmla v6.4s, v16.4s, v24.4s
    fmla v7.4s, v17.4s, v25.4s
    // kx=2: cols 2,3,4,5
    ld1 {v24.4s, v25.4s}, [x14], x15
    prfm pldl1keep, [x16]
    add x16, x16, x11
    fmla v0.4s, v12.4s, v24.4s
    fmla v1.4s, v13.4s, v25.4s
    prfm pldl1keep, [x16]
    add x16, x16, x11
    fmla v2.4s, v14.4s, v24.4s
    fmla v3.4s, v15.4s, v25.4s
    fmla v4.4s, v16.4s, v24.4s
    fmla v5.4s, v17.4s, v25.4s
    fmla v6.4s, v18.4s, v24.4s
    fmla v7.4s, v19.4s, v25.4s
    // kx=3: cols 3,4,5,6
    ld1 {v24.4s, v25.4s}, [x14], x15
    prfm pldl1keep, [x16]
    add x16, x16, x11
    fmla v0.4s, v14.4s, v24.4s
    fmla v1.4s, v15.4s, v25.4s
    prfm pldl1keep, [x16]
    add x16, x16, x11
    fmla v2.4s, v16.4s, v24.4s
    fmla v3.4s, v17.4s, v25.4s
    fmla v4.4s, v18.4s, v24.4s
    fmla v5.4s, v19.4s, v25.4s
    fmla v6.4s, v20.4s, v24.4s
    fmla v7.4s, v21.4s, v25.4s
    // kx=4: cols 4,5,6,7
    ld1 {v24.4s, v25.4s}, [x14], x15
    fmla v0.4s, v16.4s, v24.4s
    fmla v1.4s, v17.4s, v25.4s
    fmla v2.4s, v18.4s, v24.4s
    fmla v3.4s, v19.4s, v25.4s
    fmla v4.4s, v20.4s, v24.4s
    fmla v5.4s, v21.4s, v25.4s
    fmla v6.4s, v22.4s, v24.4s
    fmla v7.4s, v23.4s, v25.4s
.endm

    KROW x6,  x7
    KROW x7,  x8
    KROW x8,  x9
    KROW x9,  x10
    KROW x10, x10
    .purgem KROW

    cbz x5, .Ldw5_store
    movi v26.4s, #0
    fmax v0.4s, v0.4s, v26.4s
    fmax v1.4s, v1.4s, v26.4s
    fmax v2.4s, v2.4s, v26.4s
    fmax v3.4s, v3.4s, v26.4s
    fmax v4.4s, v4.4s, v26.4s
    fmax v5.4s, v5.4s, v26.4s
    fmax v6.4s, v6.4s, v26.4s
    fmax v7.4s, v7.4s, v26.4s
.Ldw5_store:
    mov x13, x2
    st1 {v0.4s, v1.4s}, [x13], x11      // col 0
    st1 {v2.4s, v3.4s}, [x13], x11      // col 1
    st1 {v4.4s, v5.4s}, [x13], x11      // col 2
    st1 {v6.4s, v7.4s}, [x13]           // col 3

    ldp d10, d11, [sp, #16]
    ldp d12, d13, [sp, #32]
    ldp d14, d15, [sp, #48]
    ldp d8,  d9,  [sp], #64
    ret
#if !defined(__APPLE__)
    .size yscv_dw5_creuse_neon, . - yscv_dw5_creuse_neon
#endif

// ============================================================================
// yscv_dw3s2_creuse_neon — 3x3 depthwise, STRIDE 2, 4 output columns x 8
//                          channels, interior (no bounds), column-reuse.
// ============================================================================
// The stride-2 sibling of yscv_dw5_creuse_neon. 4 output columns at stride 2
// touch input columns (2p + kx) for p in 0..3, kx in 0..2 = columns 0..8 (9).
// We load those 9 columns once per kernel row (v8..v25) and reuse them across
// the 3 kx taps and 4 output columns, then prefetch the next row's 9 columns
// under the FMA body (LS pipe idle under FMA on the in-order A53).
//
// Register plan (v8..v15 callee-saved low 64 -> stp/ldp):
//   v0..v7   = 8 accumulators: output col p -> {v[p*2], v[p*2+1]} (8 ch = 2 q).
//   v8..v25  = the row's 9 input columns (col c -> v[8+c*2], v[8+c*2+1]).
//   v26,v27  = current tap weight (2 q = 8 ch).
//
// Args (AAPCS64), mirror dw5:
//   x0  rows   = *const [*const f32; 3], each pre-offset to input column
//               (2*ow_tile - pad) and channel ch. Input(ky,kx,p) =
//               rows[ky] + (2p + kx) * c_exp floats.
//   x1  weight = dw_weight + ch (9 taps, +wstride each).
//   x2  out    = out_row + ow_tile*c_exp + ch (kernel adds p*c_exp per col).
//   x3  c_exp  = channel stride in FLOATS.
//   x4  bias   = bias + ch, or 0.
//   x5  relu   = 0 or 1.
//   x6  wstride = weight tap stride in BYTES (c_exp*4 natural KH·KW·C layout).

    .globl CNAME(yscv_dw3s2_creuse_neon)
#if !defined(__APPLE__)
    .type yscv_dw3s2_creuse_neon, %function
    .p2align 2
yscv_dw3s2_creuse_neon:
#else
    .p2align 2
_yscv_dw3s2_creuse_neon:
#endif
    stp d8,  d9,  [sp, #-64]!
    stp d10, d11, [sp, #16]
    stp d12, d13, [sp, #32]
    stp d14, d15, [sp, #48]

    mov x15, x6                         // weight tap stride (bytes)
    ldr x6,  [x0]                       // rows[0..2]
    ldr x7,  [x0, #8]
    ldr x8,  [x0, #16]
    lsl x11, x3, #2                     // c_exp bytes
    mov x14, x1                         // weight cursor (9 taps)

    cbz x4, .Ldw3_zero
    ld1 {v26.4s, v27.4s}, [x4]
    mov v0.16b, v26.16b
    mov v1.16b, v27.16b
    mov v2.16b, v26.16b
    mov v3.16b, v27.16b
    mov v4.16b, v26.16b
    mov v5.16b, v27.16b
    mov v6.16b, v26.16b
    mov v7.16b, v27.16b
    b .Ldw3_ky
.Ldw3_zero:
    movi v0.4s, #0
    movi v1.4s, #0
    movi v2.4s, #0
    movi v3.4s, #0
    movi v4.4s, #0
    movi v5.4s, #0
    movi v6.4s, #0
    movi v7.4s, #0

.Ldw3_ky:
// KROW3 row, nextrow: load 9 input cols, 3 kx taps, prefetch next row's cols.
.macro KROW3 row, nextrow
    mov x13, \row
    mov x16, \nextrow
    ld1 {v8.4s,  v9.4s},  [x13], x11    // col0
    ld1 {v10.4s, v11.4s}, [x13], x11    // col1
    ld1 {v12.4s, v13.4s}, [x13], x11    // col2
    ld1 {v14.4s, v15.4s}, [x13], x11    // col3
    ld1 {v16.4s, v17.4s}, [x13], x11    // col4
    ld1 {v18.4s, v19.4s}, [x13], x11    // col5
    ld1 {v20.4s, v21.4s}, [x13], x11    // col6
    ld1 {v22.4s, v23.4s}, [x13], x11    // col7
    ld1 {v24.4s, v25.4s}, [x13]         // col8
    // kx=0: out0 col0, out1 col2, out2 col4, out3 col6
    ld1 {v26.4s, v27.4s}, [x14], x15
    prfm pldl1keep, [x16]
    add x16, x16, x11
    fmla v0.4s, v8.4s,  v26.4s
    fmla v1.4s, v9.4s,  v27.4s
    prfm pldl1keep, [x16]
    add x16, x16, x11
    fmla v2.4s, v12.4s, v26.4s
    fmla v3.4s, v13.4s, v27.4s
    prfm pldl1keep, [x16]
    add x16, x16, x11
    fmla v4.4s, v16.4s, v26.4s
    fmla v5.4s, v17.4s, v27.4s
    fmla v6.4s, v20.4s, v26.4s
    fmla v7.4s, v21.4s, v27.4s
    // kx=1: out0 col1, out1 col3, out2 col5, out3 col7
    ld1 {v26.4s, v27.4s}, [x14], x15
    prfm pldl1keep, [x16]
    add x16, x16, x11
    fmla v0.4s, v10.4s, v26.4s
    fmla v1.4s, v11.4s, v27.4s
    prfm pldl1keep, [x16]
    add x16, x16, x11
    fmla v2.4s, v14.4s, v26.4s
    fmla v3.4s, v15.4s, v27.4s
    prfm pldl1keep, [x16]
    add x16, x16, x11
    fmla v4.4s, v18.4s, v26.4s
    fmla v5.4s, v19.4s, v27.4s
    fmla v6.4s, v22.4s, v26.4s
    fmla v7.4s, v23.4s, v27.4s
    // kx=2: out0 col2, out1 col4, out2 col6, out3 col8
    ld1 {v26.4s, v27.4s}, [x14], x15
    prfm pldl1keep, [x16]
    add x16, x16, x11
    fmla v0.4s, v12.4s, v26.4s
    fmla v1.4s, v13.4s, v27.4s
    prfm pldl1keep, [x16]
    add x16, x16, x11
    fmla v2.4s, v16.4s, v26.4s
    fmla v3.4s, v17.4s, v27.4s
    fmla v4.4s, v20.4s, v26.4s
    fmla v5.4s, v21.4s, v27.4s
    fmla v6.4s, v24.4s, v26.4s
    fmla v7.4s, v25.4s, v27.4s
.endm

    KROW3 x6, x7
    KROW3 x7, x8
    KROW3 x8, x8
    .purgem KROW3

    cbz x5, .Ldw3_store
    movi v26.4s, #0
    fmax v0.4s, v0.4s, v26.4s
    fmax v1.4s, v1.4s, v26.4s
    fmax v2.4s, v2.4s, v26.4s
    fmax v3.4s, v3.4s, v26.4s
    fmax v4.4s, v4.4s, v26.4s
    fmax v5.4s, v5.4s, v26.4s
    fmax v6.4s, v6.4s, v26.4s
    fmax v7.4s, v7.4s, v26.4s
.Ldw3_store:
    mov x13, x2
    st1 {v0.4s, v1.4s}, [x13], x11
    st1 {v2.4s, v3.4s}, [x13], x11
    st1 {v4.4s, v5.4s}, [x13], x11
    st1 {v6.4s, v7.4s}, [x13]

    ldp d10, d11, [sp, #16]
    ldp d12, d13, [sp, #32]
    ldp d14, d15, [sp, #48]
    ldp d8,  d9,  [sp], #64
    ret
#if !defined(__APPLE__)
    .size yscv_dw3s2_creuse_neon, . - yscv_dw3s2_creuse_neon
#endif

// ============================================================================
// yscv_sgemm_8x8_pipe_neon — experimental software-pipelined 8x8 microkernel.
// ============================================================================
// 8 rows × 8 cols (16 accumulators v0..v15), leaving 16 v-regs free — enough
// to FULLY double-buffer both A[k] and B[k]. Each iteration loads A[k+1]/B[k+1]
// BEFORE computing k's 16 FMAs, so the packed-panel load latency is hidden
// under the FMA block on the in-order A53 (the 8x12 tile's 24 accumulators
// left no room for this, and explicit pipelining there regressed).
//
// Pack: A = kc × 8 (8 floats per k, rows 0..7), B = kc × 8 (8 floats per k,
// cols 0..7). C accumulated from zero, stored with row stride ldc.
//
// Args (AAPCS64): x0=a_panel, x1=b_panel, x2=c, x3=ldc (floats), x4=kc.

// 16 FMAs for one k-step: B quads (b0,b1) = cols 0..3 / 4..7; A quads
// (a0,a1) = rows 0..3 / 4..7 broadcast lane-wise.
.macro FMA16_8x8 b0, b1, a0, a1
    fmla v0.4s,  \b0, \a0\().s[0]
    fmla v1.4s,  \b1, \a0\().s[0]
    fmla v2.4s,  \b0, \a0\().s[1]
    fmla v3.4s,  \b1, \a0\().s[1]
    fmla v4.4s,  \b0, \a0\().s[2]
    fmla v5.4s,  \b1, \a0\().s[2]
    fmla v6.4s,  \b0, \a0\().s[3]
    fmla v7.4s,  \b1, \a0\().s[3]
    fmla v8.4s,  \b0, \a1\().s[0]
    fmla v9.4s,  \b1, \a1\().s[0]
    fmla v10.4s, \b0, \a1\().s[1]
    fmla v11.4s, \b1, \a1\().s[1]
    fmla v12.4s, \b0, \a1\().s[2]
    fmla v13.4s, \b1, \a1\().s[2]
    fmla v14.4s, \b0, \a1\().s[3]
    fmla v15.4s, \b1, \a1\().s[3]
.endm

// set/acc entry points + shared pipelined k-loop + inline bias/Relu epilogue.
// Args: x0=a (kc×8), x1=b (kc×8), x2=c, x3=ldc(floats), x4=kc,
//       x5=bias(0=none, 8 floats), x6=relu(0/1), x7=is_last_k(0/1).
    .globl CNAME(yscv_sgemm_8x8_neon_set)
#if !defined(__APPLE__)
    .type yscv_sgemm_8x8_neon_set, %function
    .p2align 2
yscv_sgemm_8x8_neon_set:
#else
    .p2align 2
_yscv_sgemm_8x8_neon_set:
#endif
    stp d8,  d9,  [sp, #-64]!                   // v8-v15 are callee-saved (low 64)
    stp d10, d11, [sp, #16]
    stp d12, d13, [sp, #32]
    stp d14, d15, [sp, #48]
    lsl x3, x3, #2                              // ldc floats -> bytes
    movi v0.4s,  #0
    movi v1.4s,  #0
    movi v2.4s,  #0
    movi v3.4s,  #0
    movi v4.4s,  #0
    movi v5.4s,  #0
    movi v6.4s,  #0
    movi v7.4s,  #0
    movi v8.4s,  #0
    movi v9.4s,  #0
    movi v10.4s, #0
    movi v11.4s, #0
    movi v12.4s, #0
    movi v13.4s, #0
    movi v14.4s, #0
    movi v15.4s, #0
    b .Lsgemm_8x8_kloop
#if !defined(__APPLE__)
    .size yscv_sgemm_8x8_neon_set, . - yscv_sgemm_8x8_neon_set
#endif

    .globl CNAME(yscv_sgemm_8x8_neon_acc)
#if !defined(__APPLE__)
    .type yscv_sgemm_8x8_neon_acc, %function
    .p2align 2
yscv_sgemm_8x8_neon_acc:
#else
    .p2align 2
_yscv_sgemm_8x8_neon_acc:
#endif
    stp d8,  d9,  [sp, #-64]!
    stp d10, d11, [sp, #16]
    stp d12, d13, [sp, #32]
    stp d14, d15, [sp, #48]
    lsl x3, x3, #2
    mov x9, x2
    ld1 {v0.4s,  v1.4s},  [x9], x3
    ld1 {v2.4s,  v3.4s},  [x9], x3
    ld1 {v4.4s,  v5.4s},  [x9], x3
    ld1 {v6.4s,  v7.4s},  [x9], x3
    ld1 {v8.4s,  v9.4s},  [x9], x3
    ld1 {v10.4s, v11.4s}, [x9], x3
    ld1 {v12.4s, v13.4s}, [x9], x3
    ld1 {v14.4s, v15.4s}, [x9]
.Lsgemm_8x8_kloop:
    cbz x4, .Lsgemm_8x8_epilogue
    ld1 {v16.4s, v17.4s}, [x0], #32             // A[0] -> abuf0
    ld1 {v18.4s, v19.4s}, [x1], #32             // B[0] -> bbuf0
    subs x4, x4, #1
    beq .Lsgemm_8x8_klast0
.p2align 4
.Lsgemm_8x8_body:
    ld1 {v20.4s, v21.4s}, [x0], #32             // A[next] -> abuf1
    ld1 {v22.4s, v23.4s}, [x1], #32             // B[next] -> bbuf1
    FMA16_8x8 v18.4s, v19.4s, v16, v17
    subs x4, x4, #1
    beq .Lsgemm_8x8_klast1
    ld1 {v16.4s, v17.4s}, [x0], #32             // A[next+1] -> abuf0
    ld1 {v18.4s, v19.4s}, [x1], #32             // B[next+1] -> bbuf0
    FMA16_8x8 v22.4s, v23.4s, v20, v21
    subs x4, x4, #1
    bne .Lsgemm_8x8_body
    FMA16_8x8 v18.4s, v19.4s, v16, v17
    b .Lsgemm_8x8_epilogue
.Lsgemm_8x8_klast1:
    FMA16_8x8 v22.4s, v23.4s, v20, v21
    b .Lsgemm_8x8_epilogue
.Lsgemm_8x8_klast0:
    FMA16_8x8 v18.4s, v19.4s, v16, v17
.Lsgemm_8x8_epilogue:
    cbz x7, .Lsgemm_8x8_store                   // only last k-block fuses bias/Relu
    cbz x5, .Lsgemm_8x8_skipbias
    ld1 {v16.4s, v17.4s}, [x5]                  // bias cols 0..7
    fadd v0.4s,  v0.4s,  v16.4s
    fadd v1.4s,  v1.4s,  v17.4s
    fadd v2.4s,  v2.4s,  v16.4s
    fadd v3.4s,  v3.4s,  v17.4s
    fadd v4.4s,  v4.4s,  v16.4s
    fadd v5.4s,  v5.4s,  v17.4s
    fadd v6.4s,  v6.4s,  v16.4s
    fadd v7.4s,  v7.4s,  v17.4s
    fadd v8.4s,  v8.4s,  v16.4s
    fadd v9.4s,  v9.4s,  v17.4s
    fadd v10.4s, v10.4s, v16.4s
    fadd v11.4s, v11.4s, v17.4s
    fadd v12.4s, v12.4s, v16.4s
    fadd v13.4s, v13.4s, v17.4s
    fadd v14.4s, v14.4s, v16.4s
    fadd v15.4s, v15.4s, v17.4s
.Lsgemm_8x8_skipbias:
    cbz x6, .Lsgemm_8x8_store
    movi v16.4s, #0
    fmax v0.4s,  v0.4s,  v16.4s
    fmax v1.4s,  v1.4s,  v16.4s
    fmax v2.4s,  v2.4s,  v16.4s
    fmax v3.4s,  v3.4s,  v16.4s
    fmax v4.4s,  v4.4s,  v16.4s
    fmax v5.4s,  v5.4s,  v16.4s
    fmax v6.4s,  v6.4s,  v16.4s
    fmax v7.4s,  v7.4s,  v16.4s
    fmax v8.4s,  v8.4s,  v16.4s
    fmax v9.4s,  v9.4s,  v16.4s
    fmax v10.4s, v10.4s, v16.4s
    fmax v11.4s, v11.4s, v16.4s
    fmax v12.4s, v12.4s, v16.4s
    fmax v13.4s, v13.4s, v16.4s
    fmax v14.4s, v14.4s, v16.4s
    fmax v15.4s, v15.4s, v16.4s
.Lsgemm_8x8_store:
    mov x9, x2
    st1 {v0.4s,  v1.4s},  [x9], x3
    st1 {v2.4s,  v3.4s},  [x9], x3
    st1 {v4.4s,  v5.4s},  [x9], x3
    st1 {v6.4s,  v7.4s},  [x9], x3
    st1 {v8.4s,  v9.4s},  [x9], x3
    st1 {v10.4s, v11.4s}, [x9], x3
    st1 {v12.4s, v13.4s}, [x9], x3
    st1 {v14.4s, v15.4s}, [x9]
    ldp d10, d11, [sp, #16]
    ldp d12, d13, [sp, #32]
    ldp d14, d15, [sp, #48]
    ldp d8,  d9,  [sp], #64
    ret
#if !defined(__APPLE__)
    .size yscv_sgemm_8x8_neon_acc, . - yscv_sgemm_8x8_neon_acc
#endif

YSCV_NOTE_GNU_STACK
