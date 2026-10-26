/*
 * cx-moe.h: the shapes and the part layout of MoE stages 2 (experts in GDDR) and 3 (context in host RAM), from the
 * sizing in PLAN-MOE.md (2026-10-26). Shared by the card benches (tools/expert-bench.c, tools/ctx-bench.c), the host
 * context server (host/ctxserve.c) and, when the stages are built, the daemon and the trainer. Layout only; the
 * measurements that decide the sizes live in results/ and PLAN-MOE.md.
 */
#ifndef CX_MOE_H
#define CX_MOE_H

#include <stdint.h>
#include "cx-col.h"
#include "cx-proto.h"

/* ---- stage 2: one expert pool in GDDR, read by every head ---- */
#define CX_EX_FLOATS (CX_DM * CX_DH + CX_DH + CX_DH * CX_DU)   /* one expert = the column's MLP: W1 144 x 64, b1 64, Wu 64 x 128 = 17,472 */
#define CX_EX_BYTES (CX_EX_FLOATS * 4)                          /* 69,888 */
#define CX_EX_N 61440                                           /* 61,440 x 69,888 B = 4,293,918,720 B = 4 GiB - 1 MiB */
#define CX_EX_KEYS 248                                          /* product keys: 248 x 248 = 61,504 >= CX_EX_N */
#define CX_EX_HALF (CX_DX / 2)                                  /* each half of x: 40 */
#define CX_EX_KLANES 256                                        /* a half's keys padded to 256 lanes: 16 vectors of 16 */
#define CX_EX_KEYFLOATS (2 * CX_EX_HALF * CX_EX_KLANES)         /* both key tables, transposed [half][dim][lane]: 20,480 F32 = 80 KiB, L2 of every core */
#define CX_EX_HOME_BIAS 8.0f                                    /* at conversion head k's home expert k (its trained MLP) gets this routing bonus, so the
                                                                   converted model computes the trained model until the keys learn; decays with training */
static inline uint32_t cx_ex_index(uint32_t a, uint32_t b) { return (a * CX_EX_KEYS + b) % CX_EX_N; }

/* ---- stage 3: the context memory in host RAM, over the part ---- */
#define CX_CTX_HEADS CX_NCOL
#define CX_CTX_KV (2 * CX_DS)                                   /* k 64 + v 64 F32 per head per step */
#define CX_CTX_KV_BYTES (CX_CTX_HEADS * CX_CTX_KV * 4)          /* 61,440: every head's k, v of one step */
#define CX_CTX_TOP 4
#define CX_CTX_ANS_HEAD (CX_CTX_TOP * CX_DS * 4 + 64)           /* per head: 4 values of 64 F32, then one line of 4 scores (F32) and 4 step indices (u32) */
#define CX_CTX_ANS_BYTES (CX_CTX_HEADS * CX_CTX_ANS_HEAD)       /* 130,560 */
#define CX_CTX_WINDOW 1024                                      /* steps of context the host scans exactly: 120 x 1,024 x 256 B = 31.5 MB of keys */
#define CX_CTX_BASE (52u << 20)                                 /* in the part, above the n-gram rings (which end under 49 MiB) */
#define CX_CTX_OUT(s) (CX_CTX_BASE + (uint32_t)(s) * 65536u)    /* 2 slots: the heads' k, v of step n at slot n & 1, written by the card's DMA */
#define CX_CTX_IN(s) (CX_CTX_BASE + 131072u + (uint32_t)(s) * 131072u)   /* 2 slots: the answers of step n at slot n & 1, written by the host, pulled by the card's DMA */
#define CX_CTX_C_SEQ (CX_CTX_BASE + 393216u)                    /* u64, the k, v DMA's status word: steps whose k, v have landed */
#define CX_CTX_H_SEQ (CX_CTX_C_SEQ + 64u)                       /* u64, written by the host: steps whose answers are ready */
#define CX_CTX_NONE 0xFFFFFFFFu                                 /* a step index of an empty answer slot */

/* keys and values of the benches: a function of (step, head, dim) both sides recompute; "which" 0 = key, 1 = value */
static inline uint64_t cx_ctx_mix(uint64_t x) { x ^= x >> 31; x *= 0xBF58476D1CE4E5B9ull; x ^= x >> 29; x *= 0x94D049BB133111EBull; x ^= x >> 32; return x; }
static inline float cx_ctx_val(uint32_t step, uint32_t head, uint32_t dim, int which)
{
	const uint64_t x = cx_ctx_mix(0x9E3779B97F4A7C15ull * (uint64_t)(step + 1) ^ ((uint64_t)head << 20) ^ ((uint64_t)dim << 1) ^ (uint64_t)which);
	return (float)(x >> 40) * (2.0f / 16777216.0f) - 1.0f;
}

#endif
