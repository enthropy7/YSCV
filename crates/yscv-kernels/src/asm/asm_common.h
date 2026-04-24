// Shared preprocessor helpers for yscv SGEMM assembly.
//
// Mach-O (Apple) requires a leading underscore on C-visible symbols; ELF
// and COFF do not. Use `CNAME(foo)` so the source reads `foo` everywhere.

#ifndef YSCV_ASM_COMMON_H
#define YSCV_ASM_COMMON_H

#if defined(__APPLE__)
#  define CNAME(name) _##name
#else
#  define CNAME(name) name
#endif

// ELF platforms want a non-executable-stack note appended to every .o.
// Keeping it centralized means every kernel file can just include this header.
#if defined(__linux__) || defined(__FreeBSD__) || defined(__NetBSD__) || defined(__OpenBSD__)
#  define YSCV_NOTE_GNU_STACK .section .note.GNU-stack,"",%progbits
#else
#  define YSCV_NOTE_GNU_STACK
#endif

#endif // YSCV_ASM_COMMON_H
