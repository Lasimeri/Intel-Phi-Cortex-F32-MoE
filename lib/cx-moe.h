/*
 * cx-moe.h: the shapes and the part layout of MoE stages 2 (experts in GDDR) and 3 (context in host RAM), from the
 * sizing in PLAN-MOE.md (2026-10-26). Shared by the card benches (tools/expert-bench.c, tools/ctx-bench.c), the host
 * context server (host/ctxserve.c), and since 2026-10-27 by stage 2 as built: the daemon (card/cortexd.c), the
 * trainer (card/cluster.h, card/moe.h), the double reference (tools/cref.h), the coach and the evaluator. Layout
 * and the stage-2 constants (the expert, the pool, the keys, the bonus, the pool's transfer rounds); the
 * measurements that decide the sizes live in results/ and PLAN-MOE.md.
 */
#ifndef CX_MOE_H
#define CX_MOE_H

#include <stdint.h>
#include "cx-col.h"
#include "cx-proto.h"

/* ---- stage 2: one expert pool in GDDR, read by every head ---- */
#define CX_EX_FLOATS (CX_DM * CX_DH + CX_DH + CX_DH * CX_DU)   /* one expert = the column's MLP: W1 144 x 64, b1 64, Wu 64 x 128 = 17,472 */
_Static_assert(CX_O_WG - CX_O_W1 == CX_EX_FLOATS, "the MLP block of the column layout is one expert");
#define CX_EX_BYTES (CX_EX_FLOATS * 4)                          /* 69,888 */
#define CX_EX_N 61440                                           /* 61,440 x 69,888 B = 4,293,918,720 B = 4 GiB - 1 MiB */
#define CX_EX_KEYS 248                                          /* product keys: 248 x 248 = 61,504 >= CX_EX_N */
#define CX_EX_HALF (CX_DX / 2)                                  /* each half of x: 40 */
#define CX_EX_KLANES 256                                        /* a half's keys padded to 256 lanes: 16 vectors of 16 */
#define CX_EX_KEYFLOATS (2 * CX_EX_HALF * CX_EX_KLANES)         /* both key tables, transposed [half][dim][lane]: 20,480 F32 = 80 KiB, L2 of every core */
#define CX_EX_HOME_BIAS 100.0f                                  /* at conversion head k's home expert k (its trained MLP) gets this routing bonus, so the
                                                                   converted model computes the trained model; the operator lowers it across runs. 8 (the
                                                                   plan's first value) left home on 11.7 % of the steps of the chat-base checkpoint
                                                                   (results/moe-gate.log, 10-27): the real activations score higher than the seeded init's */
static inline uint32_t cx_ex_index(uint32_t a, uint32_t b) { return (a * CX_EX_KEYS + b) % CX_EX_N; }

/* the stage-2 build (PLAN-MOE.md "Stage 2 build", 2026-10-27): an expert IS the column's MLP block, W1 then b1 then
 * Wu, contiguous in the parameter layout of lib/cx-col.h from CX_O_W1; a head keeps the whole block and its MLP
 * region is the slot the routed expert is fetched into, so card/col.h and card/train.h compute on it unchanged.
 * The pool is filled by sparse upcycling (Komatsuzaki et al. 2022, arXiv 2212.05055): expert e starts as the MLP
 * of column e mod CX_NCOL of the dense checkpoint; head k's home expert is k, the pair (k / 248, k % 248), whose two
 * keys get the home bonus (a run knob, CX_EX_HOME_BIAS at conversion), so the converted model computes the dense
 * model exactly. The keys are fixed product keys (Lample et al. 2019, arXiv 1907.05242) from a seeded init: fixed
 * routing is competitive with learned routing (Roller et al. 2021, arXiv 2106.04426); learned keys are the next
 * step. Experts train by plain SGD (no moments for 1.07 G parameters), one rank-1 update per head per step. */
#define CX_EX_SLOT CX_O_W1                                      /* the expert's place in a head's parameter block */
#define CX_EX_POOL_BYTES ((size_t)CX_EX_N * CX_EX_BYTES)        /* 4,293,918,720 B; a pool file holds exactly this */
#define CX_EX_POOL_PAGES ((CX_EX_POOL_BYTES + CX_POOL_PAGE - 1) / CX_POOL_PAGE)   /* 2,048 huge pages (the last one half full) */
#define CX_EX_POOL_ROUNDS ((CX_EX_POOL_PAGES + CX_POOL_ROUND_PAGES - 1) / CX_POOL_ROUND_PAGES)   /* 187 transfer rounds */
_Static_assert(CX_POOL_ROUND_BYTES <= CX_STAGE_CKPT, "a pool round fits the stage");
#define CX_EX_HOME_A(k) ((uint32_t)(k) / CX_EX_KEYS)
#define CX_EX_HOME_B(k) ((uint32_t)(k) % CX_EX_KEYS)
#define CX_EX_LR 0.05f                                          /* the experts' SGD step, the run knob's default */
#define CX_SEED_KEYS 0xE4C0DEull
#define CX_S_KEYS 0.15f
static inline void cx_ex_keys_init(float *K)                    /* [half][dim][lane], lanes 248..255 unused */
{
	uint64_t s = CX_SEED_KEYS;
	cx_fill(K, CX_EX_KEYFLOATS, &s, CX_S_KEYS);
}

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
