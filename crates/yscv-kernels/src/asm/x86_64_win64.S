// yscv SGEMM microkernels — x86_64 Win64 ABI (Windows MSVC + MinGW).
//
// Win64 calling convention:
//   Args:      rcx, rdx, r8, r9, then [rsp+40] (after 32-byte shadow + retaddr).
//   Callee-saved GPR: rbx, rbp, rdi, rsi, r12–r15.
//   Callee-saved XMM: xmm6–xmm15.
//   Shadow space: caller reserves 32 bytes above the return address.
//
// Our SysV kernel uses rdi, rsi as scratch after shuffling from args, and
// clobbers ymm0–ymm7 (so ymm6 and ymm7 must be saved on entry). We reuse
// the exact same register schedule from x86_64_sysv.S by rewriting args
// into the SysV layout (rdi, rsi, rdx, rcx, r8) in the prologue.

.intel_syntax noprefix

// Win64 COFF does not use leading underscore on x86_64 symbols. No
// `.type`/`.size` directives either — COFF ignores them.

// ---------------------------------------------------------------------------
// Stack frame layout (Win64 4×8 entry):
//
//   Before prologue:
//     [rsp+40]     = kc (5th arg)
//     [rsp+0]      = return address
//
//   After `push rdi; push rsi; sub rsp, 40`:
//     [rsp+0..32]  = xmm6+xmm7 save area (2× 16-byte xmmword)
//     [rsp+32..40] = alignment pad (8 bytes)
//     [rsp+40]     = saved rsi
//     [rsp+48]     = saved rdi
//     [rsp+56]     = return address
//     [rsp+96]     = kc (original caller stack)
//
//   Net allocation: 56 bytes. 56 + 8 (retaddr) = 64 — 16-byte aligned.

.globl yscv_sgemm_4x8_set
.p2align 4
yscv_sgemm_4x8_set:
    push rdi
    push rsi
    sub rsp, 40
    vmovaps xmmword ptr [rsp + 0], xmm6
    vmovaps xmmword ptr [rsp + 16], xmm7
    // Shuffle args to SysV order.
    mov rdi, rcx                        // a_panel
    mov rsi, rdx                        // b_panel
    mov rdx, r8                         // c
    mov rcx, r9                         // ldc
    mov r8, qword ptr [rsp + 96]        // kc
    // Zero accumulators (matches `_set`).
    vxorps ymm0, ymm0, ymm0
    vxorps ymm1, ymm1, ymm1
    vxorps ymm2, ymm2, ymm2
    vxorps ymm3, ymm3, ymm3
    jmp .Lsgemm_4x8_win_kloop

.globl yscv_sgemm_4x8_acc
.p2align 4
yscv_sgemm_4x8_acc:
    push rdi
    push rsi
    sub rsp, 40
    vmovaps xmmword ptr [rsp + 0], xmm6
    vmovaps xmmword ptr [rsp + 16], xmm7
    mov rdi, rcx
    mov rsi, rdx
    mov rdx, r8
    mov rcx, r9
    mov r8, qword ptr [rsp + 96]
    // Load accumulators (matches `_acc`).
    shl rcx, 2
    vmovups ymm0, [rdx]
    vmovups ymm1, [rdx + rcx]
    lea r9, [rdx + rcx*2]
    vmovups ymm2, [r9]
    vmovups ymm3, [r9 + rcx]
    shr rcx, 2
    jmp .Lsgemm_4x8_win_kloop

.p2align 4
.Lsgemm_4x8_win_kloop:
    test r8, r8
    jz .Lsgemm_4x8_win_store
    mov r10, r8
    shr r10, 1
    test r10, r10
    jz .Lsgemm_4x8_win_remainder

.p2align 4
.Lsgemm_4x8_win_main:
    // --- k-step 0 ---
    vmovups ymm4, [rsi]
    vbroadcastss ymm5, dword ptr [rdi]
    vbroadcastss ymm7, dword ptr [rdi + 4]
    vfmadd231ps ymm0, ymm4, ymm5
    vbroadcastss ymm5, dword ptr [rdi + 8]
    vfmadd231ps ymm1, ymm4, ymm7
    vmovups ymm6, [rsi + 32]            // B[k+1] — early, hides behind FMAs
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
    jnz .Lsgemm_4x8_win_main

.Lsgemm_4x8_win_remainder:
    test r8b, 1
    jz .Lsgemm_4x8_win_store
    vmovups ymm4, [rsi]
    vbroadcastss ymm5, dword ptr [rdi]
    vfmadd231ps ymm0, ymm4, ymm5
    vbroadcastss ymm5, dword ptr [rdi + 4]
    vfmadd231ps ymm1, ymm4, ymm5
    vbroadcastss ymm5, dword ptr [rdi + 8]
    vfmadd231ps ymm2, ymm4, ymm5
    vbroadcastss ymm5, dword ptr [rdi + 12]
    vfmadd231ps ymm3, ymm4, ymm5

.Lsgemm_4x8_win_store:
    shl rcx, 2
    vmovups [rdx], ymm0
    vmovups [rdx + rcx], ymm1
    lea r9, [rdx + rcx*2]
    vmovups [r9], ymm2
    vmovups [r9 + rcx], ymm3
    // Epilogue: restore xmm6/7, rsi, rdi, unwind stack.
    vmovaps xmm6, xmmword ptr [rsp + 0]
    vmovaps xmm7, xmmword ptr [rsp + 16]
    add rsp, 40
    pop rsi
    pop rdi
    vzeroupper
    ret
