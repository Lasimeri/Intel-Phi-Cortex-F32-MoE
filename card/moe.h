/*
 * moe.h: MoE stage 2 on the card (PLAN-MOE.md "Stage 2 build"): the router over the product keys, the expert fetched
 * from the GDDR pool into the head's slot (the MLP region of its parameter block, CX_EX_SLOT), one step's rank-1 SGD
 * update of an expert and the write-back. Header only, single thread; card/train.h calls the route and the fetch in
 * the forward, card/cluster.h the re-fetch, the updates and the write-back in the chunk backward. Reference: the
 * MoE mode of tools/cref.h.
 */
#ifndef CX_MOE_KERNEL_H
#define CX_MOE_KERNEL_H

#include "col.h"
#include "../lib/cx-moe.h"

typedef struct cx_moe_ctx {
	float *pool;            /* the shared pool: CX_EX_N experts x CX_EX_FLOATS in GDDR huge pages */
	const float *keys;      /* CX_EX_KEYFLOATS, [half][dim][lane] */
	float bias;             /* the home pair's routing bonus */
	uint32_t home_a, home_b;
	uint32_t *route;        /* this head's expert per step of the chunk (CX_LMAX entries) */
} cx_moe;

/* the argmax lane of one half with the bonus on lane `home`: s[lane] = x . K[:, lane] over 248 keys (lanes 248..255
 * excluded); ties: the lowest lane. 40 x 16 FMAs, the key row two ahead prefetched (tools/expert-bench.c mode C,
 * 8.6 to 8.8 us for both halves). */
static inline int cx_moe_half(const float *K, const float *x, uint32_t home, float bias)
{
	__m512 acc[CX_EX_KLANES / 16];
	for (int v = 0; v < CX_EX_KLANES / 16; v++) acc[v] = _mm512_setzero_ps();
	for (int d = 0; d < CX_EX_HALF; d++) {
		const __m512 xb = CX_BC(x + d);
		if (d + 2 < CX_EX_HALF) for (int v = 0; v < CX_EX_KLANES / 16; v++) cx_pf0(K + (long)(d + 2) * CX_EX_KLANES + 16 * v);
		const float *row = K + (long)d * CX_EX_KLANES;
		for (int v = 0; v < CX_EX_KLANES / 16; v++) acc[v] = _mm512_fmadd_ps(xb, _mm512_load_ps(row + 16 * v), acc[v]);
	}
	if (bias != 0.0f) acc[home / 16] = _mm512_mask_add_ps(acc[home / 16], (__mmask16)(1u << (home & 15)), acc[home / 16], _mm512_set1_ps(bias));
	acc[CX_EX_KLANES / 16 - 1] = cx_blend(acc[CX_EX_KLANES / 16 - 1], (__mmask16)0xFF00, _mm512_set1_ps(-3.0e38f));
	__m512 m = acc[0];
	for (int v = 1; v < CX_EX_KLANES / 16; v++) m = cx_max(m, acc[v]);
	m = cx_hmax(m);
	for (int v = 0; v < CX_EX_KLANES / 16; v++) {
		const int hit = _mm512_mask2int(_mm512_cmp_ps_mask(acc[v], m, _CMP_EQ_OQ));
		if (hit) return 16 * v + __builtin_ctz(hit);
	}
	return 0;
}

/* the step's expert from x (the head's 80-wide normed input of the fast-weight pass, the tail of [o ; x]) */
static inline uint32_t cx_moe_route(const cx_moe *M, const float *x)
{
	const int a = cx_moe_half(M->keys, x, M->home_a, M->bias);
	const int b = cx_moe_half(M->keys + CX_EX_HALF * CX_EX_KLANES, x + CX_EX_HALF, M->home_b, M->bias);
	return cx_ex_index((uint32_t)a, (uint32_t)b);
}

/* the expert's 1,092 lines from the pool into the slot, 8 lines ahead prefetched (expert-bench mode A: 94 to 96 us) */
static inline void cx_moe_fetch(const cx_moe *M, uint32_t e, float *slot)
{
	const float *src = M->pool + (size_t)e * CX_EX_FLOATS;
	for (uint32_t i = 0; i < CX_EX_FLOATS; i += 16) {
		if (i + 128 < CX_EX_FLOATS) cx_pf0(src + i + 128);
		_mm512_store_ps(slot + i, _mm512_load_ps(src + i));
	}
}

/* the slot's expert back into the pool */
static inline void cx_moe_store(const cx_moe *M, uint32_t e, const float *slot)
{
	float *dst = M->pool + (size_t)e * CX_EX_FLOATS;
	for (uint32_t i = 0; i < CX_EX_FLOATS; i += 16) _mm512_store_ps(dst + i, _mm512_load_ps(slot + i));
}

/* one step's rank-1 SGD updates on the expert in a slot: W1 -= lr min dh^T, b1 -= lr dh, Wu -= lr z du^T, from the
 * step's saved vectors (min = [o ; x] 144, dh 64, z 64, du 128); -lr dh and -lr du rounded once, then one FMA per
 * weight */
static inline void cx_moe_update(float *slot, float lr, const float *min, const float *dh, const float *z, const float *du)
{
	const __m512 nlr = _mm512_set1_ps(-lr);
	float *const W1 = slot, *const b1 = slot + CX_DM * CX_DH, *const Wu = b1 + CX_DH;
	__m512 dhv[CX_DH / 16];
	for (int v = 0; v < CX_DH / 16; v++) dhv[v] = _mm512_mul_ps(nlr, _mm512_load_ps(dh + 16 * v));
	for (int i = 0; i < CX_DM; i++) {
		const __m512 m = CX_BC(min + i);
		float *row = W1 + (long)i * CX_DH;
		for (int v = 0; v < CX_DH / 16; v++) _mm512_store_ps(row + 16 * v, _mm512_fmadd_ps(m, dhv[v], _mm512_load_ps(row + 16 * v)));
	}
	for (int v = 0; v < CX_DH / 16; v++) _mm512_store_ps(b1 + 16 * v, _mm512_add_ps(_mm512_load_ps(b1 + 16 * v), dhv[v]));
	__m512 duv[CX_DU / 16];
	for (int v = 0; v < CX_DU / 16; v++) duv[v] = _mm512_mul_ps(nlr, _mm512_load_ps(du + 16 * v));
	for (int i = 0; i < CX_DH; i++) {
		const __m512 zb = CX_BC(z + i);
		float *row = Wu + (long)i * CX_DU;
		for (int v = 0; v < CX_DU / 16; v++) _mm512_store_ps(row + 16 * v, _mm512_fmadd_ps(zb, duv[v], _mm512_load_ps(row + 16 * v)));
	}
}

#endif
