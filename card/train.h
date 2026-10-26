/*
 * train.h: one column's training step on the VPU (Stage 3). Header only, single thread.
 *
 * Forward with per-step saves and the consensus norm hc = RMSNorm(u) cg; the tied head over a chunk of L steps;
 * softmax + cross-entropy + its gradient dl; the backward through the column and the fast-weight recurrence (dT
 * carried inside the chunk, T at the chunk start constant, the bus input stop-gradient); the parameter gradients as
 * chunk-end outer products over the saved per-step vectors (each gradient matrix written once per chunk, not once per
 * step); the embedding gradient in two passes plus the input rows. Reference: tools/ref.h (double).
 *
 * The embedding is held in vocabulary BLOCKS, ETB[v / 16][c][v % 16] (256 blocks of 128 x 16 = 8 KiB): the head and
 * both embedding-gradient passes walk 8 KiB blocks sequentially, and a token's input row is one block (prefetched as
 * a burst one step ahead, then 128 lane loads). Measured 2026-10-25: a plain store into a cold line costs its read for
 * ownership and vmovnraps costs more (+35 us per step), so saves use plain stores into lines prefetched for writing
 * (vprefetche0) one step ahead; data consumed once is evicted from L2 (clevict1) after use.
 */
#ifndef CX_TRAIN_H
#define CX_TRAIN_H

#include "col.h"

#define CX_LMAX 32
#define CX_LN2 0.6931471805599453f
#ifndef CX_EVICT
#define CX_EVICT 1
#endif
#define CX_ETB(v, c) ((((long)(v) >> 4) * CX_DE + (c)) * 16 + ((v) & 15))
#define CX_BLOCK (CX_DE * 16)   /* floats per vocabulary block */

typedef struct {
	cx_col col;      /* col.act = the hot block */
	float *acts;     /* CX_LMAX x CX_NACT */
	float *Ts;       /* CX_LMAX x CX_NT: T before each step */
	float *ET;       /* ETB */
	const float *ES; /* optional row-major shadow E[v][c] of ETB (the cluster keeps one, refreshed per chunk): 8-line input rows */
	float *cg;       /* consensus gain, 128 */
	float *hc;       /* CX_LMAX x 128 */
	float *logits;   /* CX_LMAX x CX_VOCAB; after cx_head_loss: dl = (softmax - onehot) / L */
	float *loss;     /* CX_LMAX */
	float *ebuf;     /* 128 */
	/* backward */
	float *dhc;      /* CX_LMAX x 128 */
	float *dT;       /* 64 x 64, carried */
	float *gsave;    /* CX_LMAX x CX_NG */
	float *gprm;     /* CX_NPARAM */
	float *gcg;      /* 128 */
	float *gb1;      /* CX_DH */
	float *ggain;    /* 80 */
	float *gwg;      /* CX_DH */
	float *gbg;      /* 16, lane 0 */
	float *GET;      /* ETB-layout embedding gradient */
	float *tmp;      /* 256 floats of scratch */
	/* MoE stage 1: n-gram memory. ng = gates (2 x 128, shared) or NULL (plain model); ngrow / ngnext = this and the next
	 * step's rows (set by the cluster before each step); ngr = the rows used at each step (L x 2 x 128, for the gate gradient) */
	const float *ng;
	const float *ngrow[CX_NG_ORDERS], *ngnext[CX_NG_ORDERS];
	float *ngr;
	int L;
} cx_train;

static inline void cx_evict1(const void *p) { if (CX_EVICT) __asm__ volatile("clevict1 %0" : : "m"(*(const char *)p)); }
static inline void cx_pfe0(const void *p) { __asm__ volatile("vprefetche0 %0" : : "m"(*(const char *)p)); }

static inline void cx_copy(float *dst, const float *src, int n)
{
	for (int i = 0; i < n; i += 16) _mm512_store_ps(dst + i, _mm512_load_ps(src + i));
}
static inline void cx_copy_evict(float *dst, const float *src, int n)
{
	cx_copy(dst, src, n);
	for (int i = 0; i < n; i += 16) cx_evict1(dst + i);
}
static inline void cx_pf_lines(const float *p, int n) { for (int i = 0; i < n; i += 16) cx_pf0(p + i); }
static inline void cx_pfe_lines(const float *p, int n) { for (int i = 0; i < n; i += 16) cx_pfe0(p + i); }
static inline void cx_zero(float *p, long n) { const __m512 z = _mm512_setzero_ps(); for (long i = 0; i < n; i += 16) _mm512_store_ps(p + i, z); }

/* ---- forward ---- */

static inline void cx_gather_prefetch(const float *ETB, uint32_t id) { cx_pf_lines(ETB + ((long)id >> 4) * CX_BLOCK, CX_BLOCK); }

static inline void cx_gather_row(const float *ETB, uint32_t id, float *e)
{
	const float *b = ETB + ((long)id >> 4) * CX_BLOCK + (id & 15);
	for (int c = 0; c < CX_DE; c++) e[c] = b[c * 16];
	for (int c = 0; c < CX_DE; c++) cx_evict1(b + c * 16);
}

/* step t of the chunk: saves T, the act block (with e) and hc; the next step's destinations and token are prefetched */
static inline void cx_train_fwd_step(cx_train *tr, int t, uint32_t id, uint32_t id_next, const float *Bt)
{
	float *const A = tr->col.act;
#ifndef CX_NOSNAP
	cx_copy_evict(tr->Ts + (long)t * CX_NT, tr->col.T, CX_NT);
#endif
	if (tr->ES) {
		cx_copy(tr->ebuf, tr->ES + (long)id * CX_DE, CX_DE);
		cx_pf_lines(tr->ES + (long)id_next * CX_DE, CX_DE);
	} else {
		cx_gather_row(tr->ET, id, tr->ebuf);
		cx_gather_prefetch(tr->ET, id_next);
	}
	if (tr->ng) {
		for (int n = 0; n < CX_NG_ORDERS; n++) {
			const float *r = tr->ngrow[n], *g = tr->ng + n * CX_DE;
			for (int v = 0; v < CX_DE / 16; v++) _mm512_store_ps(tr->ebuf + 16 * v, _mm512_fmadd_ps(_mm512_load_ps(g + 16 * v), _mm512_load_ps(r + 16 * v), _mm512_load_ps(tr->ebuf + 16 * v)));
			cx_copy(tr->ngr + ((long)t * CX_NG_ORDERS + n) * CX_DE, r, CX_DE);
			if (tr->ngnext[n]) cx_pf_lines(tr->ngnext[n], CX_DE);
		}
	}
	if (t + 1 < CX_LMAX) { cx_pfe_lines(tr->Ts + (long)(t + 1) * CX_NT, CX_NT); cx_pfe_lines(tr->acts + (long)(t + 1) * CX_NACT, CX_NACT); }
	cx_col_step(&tr->col, tr->ebuf, Bt);
	cx_copy(A + CX_A_E, tr->ebuf, CX_DE);
	__m512 u[8], ss = _mm512_setzero_ps();
	for (int v = 0; v < 8; v++) { u[v] = _mm512_load_ps(A + CX_A_U + 16 * v); ss = _mm512_fmadd_ps(u[v], u[v], ss); }
	const __m512 xsu = cx_rsqrt(_mm512_fmadd_ps(cx_hsum(ss), _mm512_set1_ps(1.0f / CX_DU), _mm512_set1_ps(CX_EPS)));
	_mm512_store_ps(A + CX_A_XSU, xsu);
	for (int v = 0; v < 8; v++) {
		const __m512 h = _mm512_mul_ps(_mm512_mul_ps(u[v], xsu), _mm512_load_ps(tr->cg + 16 * v));
		_mm512_store_ps(A + CX_A_HC + 16 * v, h);
		_mm512_store_ps(tr->hc + (long)t * CX_DU + 16 * v, h);
	}
	cx_copy_evict(tr->acts + (long)t * CX_NACT, A, CX_NACT);
}

/* logits[t][v] = sum_c hc[t][c] ETB(v, c): tiles of 8 steps x 2 blocks (16 accumulators), ETB streamed L/8 times */
static inline void cx_head_logits(cx_train *tr)
{
	for (int tb = 0; tb < tr->L; tb += 8)
		for (int g = 0; g < CX_VOCAB / 16; g += 2) {
			__m512 acc[8][2];
			_Pragma("unroll") for (int i = 0; i < 8; i++) { acc[i][0] = _mm512_setzero_ps(); acc[i][1] = _mm512_setzero_ps(); }
			const float *hc0 = tr->hc + (long)tb * CX_DU, *b0 = tr->ET + (long)g * CX_BLOCK, *b1 = b0 + CX_BLOCK;
			for (int c = 0; c < CX_DE; c++) {
				if (c + 8 < CX_DE) { cx_pf0(b0 + (c + 8) * 16); cx_pf0(b1 + (c + 8) * 16); }
				const __m512 e0 = _mm512_load_ps(b0 + c * 16), e1 = _mm512_load_ps(b1 + c * 16);
				_Pragma("unroll") for (int i = 0; i < 8; i++) {
					const __m512 hb = CX_BC(hc0 + i * CX_DU + c);
					acc[i][0] = _mm512_fmadd_ps(hb, e0, acc[i][0]);
					acc[i][1] = _mm512_fmadd_ps(hb, e1, acc[i][1]);
				}
				cx_evict1(b0 + c * 16);
				cx_evict1(b1 + c * 16);
			}
			_Pragma("unroll") for (int i = 0; i < 8; i++) {
				float *o = tr->logits + (long)(tb + i) * CX_VOCAB + g * 16;
				_mm512_store_ps(o, acc[i][0]);
				_mm512_store_ps(o + 16, acc[i][1]);
				cx_evict1(o);
				cx_evict1(o + 16);
			}
		}
}

/* per step: loss[t] = logsumexp(logits) - logits[tgt]; logits[t] := (softmax - onehot(tgt)) / L */
static inline void cx_head_loss(cx_train *tr, const uint32_t *tgt)
{
	const __m512 l2e = _mm512_set1_ps(CX_LOG2E), ln2 = _mm512_set1_ps(CX_LN2), invL = _mm512_set1_ps(1.0f / tr->L);
	for (int t = 0; t < tr->L; t++) {
		float *const l = tr->logits + (long)t * CX_VOCAB;
		for (int v = 0; v < 8; v++) cx_pf0(l + 16 * v);
		__m512 m = _mm512_set1_ps(-3.0e38f);
		for (int v = 0; v < CX_VOCAB / 16; v++) {
			if (v + 8 < CX_VOCAB / 16) cx_pf0(l + 16 * (v + 8));
			m = cx_max(m, _mm512_load_ps(l + 16 * v));
		}
		m = cx_hmax(m);
		const __m512 ltgt = CX_BC(l + tgt[t]);
		__m512 sum = _mm512_setzero_ps();
		for (int v = 0; v < CX_VOCAB / 16; v++) {
			const __m512 e = cx_exp2(_mm512_mul_ps(_mm512_sub_ps(_mm512_load_ps(l + 16 * v), m), l2e));
			_mm512_store_ps(l + 16 * v, e);
			sum = _mm512_add_ps(sum, e);
		}
		sum = cx_hsum(sum);
		const __m512 lossv = _mm512_fmadd_ps(cx_log2(sum), ln2, _mm512_sub_ps(m, ltgt));
		_mm512_mask_store_ps(tr->loss + (t & ~15), (__mmask16)(1u << (t & 15)), lossv);
		const __m512 scale = _mm512_mul_ps(cx_rcp(sum), invL);
		for (int v = 0; v < CX_VOCAB / 16; v++) _mm512_store_ps(l + 16 * v, _mm512_mul_ps(_mm512_load_ps(l + 16 * v), scale));
		float *const lt = l + (tgt[t] & ~15u);
		const __m512 d = _mm512_load_ps(lt);
		_mm512_store_ps(lt, _mm512_mask_sub_ps(d, (__mmask16)(1u << (tgt[t] & 15)), d, invL));
		for (int v = 0; v < CX_VOCAB / 16; v++) cx_evict1(l + 16 * v);
	}
}

/* ---- backward ---- */

/* dhc[t][c] = sum_v ETB(v, c) dl[t][v]: tiles of 4 steps x 4 c (16 accumulators) over all 256 blocks (4 consecutive lines
 * per block, prefetched 8 blocks ahead); one lane reduction per output; ETB streamed L/4 times */
static inline void cx_head_dhc(cx_train *tr)
{
	for (int tb = 0; tb < tr->L; tb += 4)
		for (int c = 0; c < CX_DE; c += 4) {
			__m512 acc[4][4];
			_Pragma("unroll") for (int i = 0; i < 4; i++) _Pragma("unroll") for (int j = 0; j < 4; j++) acc[i][j] = _mm512_setzero_ps();
			const float *dl0 = tr->logits + (long)tb * CX_VOCAB, *e0 = tr->ET + c * 16;
			for (int g = 0; g < CX_VOCAB / 16; g++, e0 += CX_BLOCK) {
				if (g + 8 < CX_VOCAB / 16) _Pragma("unroll") for (int j = 0; j < 4; j++) cx_pf0(e0 + 8 * CX_BLOCK + 16 * j);
				__m512 ev[4];
				_Pragma("unroll") for (int j = 0; j < 4; j++) ev[j] = _mm512_load_ps(e0 + 16 * j);
				_Pragma("unroll") for (int i = 0; i < 4; i++) {
					const __m512 d = _mm512_load_ps(dl0 + (long)i * CX_VOCAB + g * 16);
					_Pragma("unroll") for (int j = 0; j < 4; j++) acc[i][j] = _mm512_fmadd_ps(ev[j], d, acc[i][j]);
				}
				_Pragma("unroll") for (int j = 0; j < 4; j++) cx_evict1(e0 + 16 * j);
			}
			_Pragma("unroll") for (int i = 0; i < 4; i++) {
				float *o = tr->dhc + (long)(tb + i) * CX_DU + (c & ~15);
				_Pragma("unroll") for (int j = 0; j < 4; j++) _mm512_mask_store_ps(o, (__mmask16)(1u << ((c + j) & 15)), cx_hsum(acc[i][j]));
			}
		}
}

/* GET(v, c) += sum_t hc[t][c] dl[t][v]: per block and 16-step tile, dl rows in registers, the block RMW once */
static inline void cx_head_dE(cx_train *tr)
{
	for (int g = 0; g < CX_VOCAB / 16; g++) {
		float *G = tr->GET + (long)g * CX_BLOCK;
		cx_pf_lines(G, CX_BLOCK);
		for (int tb = 0; tb < tr->L; tb += 16) {
			__m512 d[16];
			_Pragma("unroll") for (int i = 0; i < 16; i++) d[i] = _mm512_load_ps(tr->logits + (long)(tb + i) * CX_VOCAB + g * 16);
			const float *hc0 = tr->hc + (long)tb * CX_DU;
			for (int c = 0; c < CX_DE; c++) {
				__m512 acc = _mm512_load_ps(G + c * 16);
				_Pragma("unroll") for (int i = 0; i < 16; i++) acc = _mm512_fmadd_ps(CX_BC(hc0 + i * CX_DU + c), d[i], acc);
				_mm512_store_ps(G + c * 16, acc);
			}
		}
		for (int c = 0; c < CX_DE; c++) cx_evict1(G + c * 16);
		for (int i = 0; i < tr->L; i++) cx_evict1(tr->logits + (long)i * CX_VOCAB + g * 16);
	}
}

/* y[r] = W[r] . x over 16 NV lanes, rows r < R (R a multiple of 16): dot form of a GEMV, one lane reduction per row */
#define CX_GEMVT(NV)                                                                                                 \
	static inline void cx_gemvt##NV(const float *W, const float *x, int R, float *y)                                 \
	{                                                                                                                \
		__m512 xv[NV];                                                                                               \
		_Pragma("unroll") for (int v = 0; v < NV; v++) xv[v] = _mm512_load_ps(x + 16 * v);                           \
		for (int r = 0; r < R; r++) {                                                                                \
			const float *w = W + (long)r * (16 * NV);                                                                \
			_Pragma("unroll") for (int v = 0; v < NV; v++) cx_pf0(w + 2 * (16 * NV) + 16 * v);                       \
			__m512 acc = _mm512_mul_ps(xv[0], _mm512_load_ps(w));                                                    \
			_Pragma("unroll") for (int v = 1; v < NV; v++) acc = _mm512_fmadd_ps(xv[v], _mm512_load_ps(w + 16 * v), acc); \
			_mm512_mask_store_ps(y + (r & ~15), (__mmask16)(1u << (r & 15)), cx_hsum(acc));                          \
		}                                                                                                            \
	}
CX_GEMVT(1)
CX_GEMVT(4)
CX_GEMVT(8)
CX_GEMVT(13)

/* RMSNorm backward with gain: given dx (on x = n gain, n = xin xs), writes dxin and adds dx n into ggain */
static inline void cx_norm_bwd(const float *dx, const float *xin, __m512 xs, const float *gain, float *ggain, float *dxin, int NV, float invN)
{
	__m512 dn[8], n[8], dot = _mm512_setzero_ps();
	for (int v = 0; v < NV; v++) {
		n[v] = _mm512_mul_ps(_mm512_load_ps(xin + 16 * v), xs);
		const __m512 d = _mm512_load_ps(dx + 16 * v);
		dn[v] = _mm512_mul_ps(d, _mm512_load_ps(gain + 16 * v));
		_mm512_store_ps(ggain + 16 * v, _mm512_fmadd_ps(d, n[v], _mm512_load_ps(ggain + 16 * v)));
		dot = _mm512_fmadd_ps(dn[v], n[v], dot);
	}
	dot = _mm512_mul_ps(cx_hsum(dot), _mm512_set1_ps(invN));
	for (int v = 0; v < NV; v++) _mm512_store_ps(dxin + 16 * v, _mm512_mul_ps(xs, cx_fnmadd(n[v], dot, dn[v])));
}

/* column backward of step t given the gradient on the vote u (du, 128) and on the confidence pre-activation (dgl, all lanes):
 * data gradients, the per-step vectors into gsave[t], dT carried, hot accumulators gb1/ggain/gwg/gbg */
static inline void cx_col_bwd_step(cx_train *tr, int t, const float *Bt, const float *du, __m512 dgl)
{
	const float *P = tr->col.prm, *A = tr->acts + (long)t * CX_NACT, *T0 = tr->Ts + (long)t * CX_NT;
	float *const G = tr->gsave + (long)t * CX_NG, *const S = tr->tmp, *const dT = tr->dT;
	cx_pf_lines(A, CX_NACT);
	cx_pf_lines(T0, CX_NT);
	if (t) { cx_pf_lines(tr->acts + (long)(t - 1) * CX_NACT, CX_NACT); }
	const __m512 zero = _mm512_setzero_ps();

	/* 1. du (from the caller) into the saved vectors */
	cx_copy(G + CX_G_DU, du, CX_DU);
	/* 2. dz = Wu du; 3. dh = dz (sg + h sg (1 - sg)); gb1 += dh */
	float *const dz = S, *const dh = G + CX_G_DH;
	cx_gemvt8(P + CX_O_WU, G + CX_G_DU, CX_DH, dz);
	for (int v = 0; v < CX_NVH; v++) {   /* g = sigmoid(z . wg + bg): dz += wg dgl; gwg += z dgl */
		_mm512_store_ps(dz + 16 * v, _mm512_fmadd_ps(dgl, _mm512_load_ps(P + CX_O_WG + 16 * v), _mm512_load_ps(dz + 16 * v)));
		_mm512_store_ps(tr->gwg + 16 * v, _mm512_fmadd_ps(dgl, _mm512_load_ps(A + CX_A_Z + 16 * v), _mm512_load_ps(tr->gwg + 16 * v)));
	}
	_mm512_store_ps(tr->gbg, _mm512_add_ps(_mm512_load_ps(tr->gbg), dgl));
	for (int v = 0; v < CX_NVH; v++) {
		const __m512 sg = _mm512_load_ps(A + CX_A_SG + 16 * v), h = _mm512_load_ps(A + CX_A_H + 16 * v);
		const __m512 f = _mm512_fmadd_ps(_mm512_mul_ps(h, sg), _mm512_sub_ps(_mm512_set1_ps(1.0f), sg), sg);
		const __m512 d = _mm512_mul_ps(_mm512_load_ps(dz + 16 * v), f);
		_mm512_store_ps(dh + 16 * v, d);
		_mm512_store_ps(tr->gb1 + 16 * v, _mm512_add_ps(_mm512_load_ps(tr->gb1 + 16 * v), d));
	}
	/* 4. dmin = W1 dh: do = dmin[0..64), dx = dmin[64..144) */
	float *const dmin = S + 64;   /* 144 */
	cx_gemvt4(P + CX_O_W1, dh, CX_DM, dmin);
	const float *dO = dmin;
	/* 5. fast weights */
	const __m512 a = _mm512_load_ps(A + CX_A_AV), b = _mm512_load_ps(A + CX_A_BV), kq = _mm512_load_ps(A + CX_A_KQ);
	const float *k = A + CX_A_K, *y = A + CX_A_Y, *vv = y + 128, *dd0 = A + CX_A_D, *Sq = A + CX_A_SQ, *Sk = A + CX_A_SK;
	__m512 da = zero, dkq = zero, dSq[4], dd[4];
	for (int v = 0; v < 4; v++) {
		const __m512 o = _mm512_load_ps(dO + 16 * v);
		da = _mm512_fmadd_ps(o, _mm512_load_ps(Sq + 16 * v), da);
		dkq = _mm512_fmadd_ps(o, _mm512_load_ps(dd0 + 16 * v), dkq);
		dSq[v] = _mm512_mul_ps(a, o);
		dd[v] = _mm512_mul_ps(o, kq);
	}
	for (int j = 0; j < CX_DS; j++) {   /* sweep 1: da += <dT, T0>, dd += k_j dT[j] */
		const float *g = dT + j * CX_DS, *t0 = T0 + j * CX_DS;
		const __m512 kj = CX_BC(k + j);
		for (int v = 0; v < 4; v++) {
			const __m512 gv = _mm512_load_ps(g + 16 * v);
			da = _mm512_fmadd_ps(gv, _mm512_load_ps(t0 + 16 * v), da);
			dd[v] = _mm512_fmadd_ps(kj, gv, dd[v]);
		}
	}
	dkq = cx_hsum(dkq);
	const __m512 ab = _mm512_mul_ps(a, b);
	__m512 db = zero, dSk[4], dv[4];
	for (int v = 0; v < 4; v++) {
		const __m512 skv = _mm512_load_ps(Sk + 16 * v);
		dv[v] = _mm512_mul_ps(b, dd[v]);
		db = _mm512_fmadd_ps(dd[v], cx_fnmadd(a, skv, _mm512_load_ps(vv + 16 * v)), db);
		da = cx_fnmadd(dd[v], _mm512_mul_ps(b, skv), da);
		dSk[v] = _mm512_mul_ps(_mm512_sub_ps(zero, ab), dd[v]);
	}
	da = cx_hsum(da);
	db = cx_hsum(db);
	float *const dk = S + 208, *const dq = S + 272;   /* 64 each */
	float dsq_m[CX_DS] __attribute__((aligned(64))), dsk_m[CX_DS] __attribute__((aligned(64))), dvec[CX_DS] __attribute__((aligned(64)));
	for (int v = 0; v < 4; v++) { _mm512_store_ps(dsq_m + 16 * v, dSq[v]); _mm512_store_ps(dsk_m + 16 * v, dSk[v]); _mm512_store_ps(dvec + 16 * v, _mm512_load_ps(dd0 + 16 * v)); }
	for (int j = 0; j < CX_DS; j++) {   /* sweep 2: dk_j, dq_j, and dT[j] := a dT[j] + q_j dSq + k_j dSk */
		float *g = dT + j * CX_DS;
		const float *t0 = T0 + j * CX_DS;
		const __m512 kj = CX_BC(k + j), qj = CX_BC(y + j);
		__m512 ak = zero, aq = zero;
		for (int v = 0; v < 4; v++) {
			const __m512 gv = _mm512_load_ps(g + 16 * v), tv = _mm512_load_ps(t0 + 16 * v);
			ak = _mm512_fmadd_ps(gv, _mm512_load_ps(dvec + 16 * v), ak);
			ak = _mm512_fmadd_ps(tv, dSk[v], ak);
			aq = _mm512_fmadd_ps(tv, dSq[v], aq);
			_mm512_store_ps(g + 16 * v, _mm512_fmadd_ps(kj, dSk[v], _mm512_fmadd_ps(qj, dSq[v], _mm512_mul_ps(a, gv))));
		}
		_mm512_mask_store_ps(dk + (j & ~15), (__mmask16)(1u << (j & 15)), _mm512_fmadd_ps(dkq, qj, cx_hsum(ak)));
		_mm512_mask_store_ps(dq + (j & ~15), (__mmask16)(1u << (j & 15)), _mm512_fmadd_ps(dkq, kj, cx_hsum(aq)));
	}
	(void)dsq_m; (void)dsk_m;
	/* k = kr ks: dkr = ks (dk - k (k . dk)); a, b through the sigmoids; dy assembled */
	__m512 kdk = zero;
	for (int v = 0; v < 4; v++) kdk = _mm512_fmadd_ps(_mm512_load_ps(dk + 16 * v), _mm512_load_ps(k + 16 * v), kdk);
	kdk = cx_hsum(kdk);
	const __m512 ks = _mm512_load_ps(A + CX_A_KS);
	float *const dy = G + CX_G_DY;
	for (int v = 0; v < 4; v++) {
		_mm512_store_ps(dy + 16 * v, _mm512_load_ps(dq + 16 * v));
		_mm512_store_ps(dy + 64 + 16 * v, _mm512_mul_ps(ks, cx_fnmadd(_mm512_load_ps(k + 16 * v), kdk, _mm512_load_ps(dk + 16 * v))));
		_mm512_store_ps(dy + 128 + 16 * v, dv[v]);
	}
	const __m512 one = _mm512_set1_ps(1.0f);
	const __m512 dap = _mm512_mul_ps(da, _mm512_mul_ps(a, _mm512_sub_ps(one, a))), dbp = _mm512_mul_ps(db, _mm512_mul_ps(b, _mm512_sub_ps(one, b)));
	_mm512_store_ps(dy + 192, cx_blend(cx_blend(zero, 0x0001, dap), 0x0002, dbp));
	/* 6. dx = dmin[64..) + W dy */
	float *const dxw = S + 336;   /* 80 */
	cx_gemvt13(P + CX_O_W, dy, CX_DX, dxw);
	for (int v = 0; v < 5; v++) _mm512_store_ps(dxw + 16 * v, _mm512_add_ps(_mm512_load_ps(dxw + 16 * v), _mm512_load_ps(dmin + 64 + 16 * v)));
	/* 7. x = RMSNorm([p ; r]) gain: xin assembled from A_P (64) and A_R (16) (qb sits between them in the block) */
	float *const dxin = S + 416;   /* 80: dp (64) then dr (16) */
	float *const xin = S + 624;   /* 80 */
	cx_copy(xin, A + CX_A_P, CX_DP);
	_mm512_store_ps(xin + CX_DP, _mm512_load_ps(A + CX_A_R));
	cx_norm_bwd(dxw, xin, _mm512_load_ps(A + CX_A_XS), P + CX_O_GAIN, tr->ggain, dxin, 5, 1.0f / CX_DX);
	/* 8. bus attention: dalpha = Bt^T dr; ds = alpha (dalpha - alpha . dalpha); dqb = Bt ds */
	float *const dalpha = S + 496;   /* 128 */
	cx_gemv8(Bt, dxin + 64, CX_DB, dalpha);
	__m512 adot = zero;
	for (int v = 0; v < 8; v++) adot = _mm512_fmadd_ps(_mm512_load_ps(A + CX_A_ALPHA + 16 * v), _mm512_load_ps(dalpha + 16 * v), adot);
	adot = cx_hsum(adot);
	for (int v = 0; v < 8; v++) _mm512_store_ps(dalpha + 16 * v, _mm512_mul_ps(_mm512_load_ps(A + CX_A_ALPHA + 16 * v), _mm512_sub_ps(_mm512_load_ps(dalpha + 16 * v), adot)));
	cx_gemvt8(Bt, dalpha, CX_DB, G + CX_G_DQB);
	/* 9. dp = dxin[0..64) + Wqb dqb; 10. de = P_in dp */
	float *const dp = G + CX_G_DP;
	cx_gemvt1(P + CX_O_WQB, G + CX_G_DQB, CX_DP, dp);
	for (int v = 0; v < 4; v++) _mm512_store_ps(dp + 16 * v, _mm512_add_ps(_mm512_load_ps(dp + 16 * v), _mm512_load_ps(dxin + 16 * v)));
	cx_gemvt4(P + CX_O_PIN, dp, CX_DE, G + CX_G_DE);
	for (int i = 0; i < CX_NT; i += 16) cx_evict1(T0 + i);
}

/* Gm[k][16 NV] = sum_t a[t][k] g[t][16 NV] (the chunk-end outer products) */
#define CX_OUTER(NV)                                                                                                 \
	static inline void cx_outer##NV(float *Gm, const float *a, long as, const float *g, long gs, int K, int L)        \
	{                                                                                                                \
		for (int k = 0; k < K; k++) {                                                                                \
			__m512 acc[NV];                                                                                          \
			_Pragma("unroll") for (int v = 0; v < NV; v++) acc[v] = _mm512_setzero_ps();                             \
			for (int t = 0; t < L; t++) {                                                                            \
				const __m512 ab = CX_BC(a + t * as + k);                                                             \
				_Pragma("unroll") for (int v = 0; v < NV; v++) acc[v] = _mm512_fmadd_ps(ab, _mm512_load_ps(g + t * gs + 16 * v), acc[v]); \
			}                                                                                                        \
			_Pragma("unroll") for (int v = 0; v < NV; v++) _mm512_store_ps(Gm + (long)k * (16 * NV) + 16 * v, acc[v]); \
		}                                                                                                            \
	}
CX_OUTER(1)
CX_OUTER(4)
CX_OUTER(8)
CX_OUTER(13)

/* the parameter gradients of the chunk from the saved vectors (Wb: none, the bus is stop-gradient) */
static inline void cx_train_param_grads(cx_train *tr)
{
	const int L = tr->L;
	float *gp = tr->gprm;
	cx_outer4(gp + CX_O_PIN, tr->acts + CX_A_E, CX_NACT, tr->gsave + CX_G_DP, CX_NG, CX_DE, L);
	cx_outer1(gp + CX_O_WQB, tr->acts + CX_A_P, CX_NACT, tr->gsave + CX_G_DQB, CX_NG, CX_DP, L);
	cx_copy(gp + CX_O_GAIN, tr->ggain, CX_DX);
	cx_outer13(gp + CX_O_W, tr->acts + CX_A_MIN + CX_DS, CX_NACT, tr->gsave + CX_G_DY, CX_NG, CX_DX, L);
	cx_outer4(gp + CX_O_W1, tr->acts + CX_A_MIN, CX_NACT, tr->gsave + CX_G_DH, CX_NG, CX_DM, L);
	cx_copy(gp + CX_O_B1, tr->gb1, CX_DH);
	cx_outer8(gp + CX_O_WU, tr->acts + CX_A_Z, CX_NACT, tr->gsave + CX_G_DU, CX_NG, CX_DH, L);
	cx_copy(gp + CX_O_WG, tr->gwg, CX_DH);
	cx_copy(gp + CX_O_BG, tr->gbg, 16);
	cx_zero(gp + CX_O_WB, CX_NPARAM - CX_O_WB);   /* Wb: the bus is stop-gradient */
}

/* GET(id_t, c) += de_t[c] for the chunk's input rows */
static inline void cx_train_input_grads(cx_train *tr, const uint32_t *ids)
{
	for (int t = 0; t < tr->L; t++) {
		float *b = tr->GET + ((long)ids[t] >> 4) * CX_BLOCK + (ids[t] & 15);
		const float *de = tr->gsave + (long)t * CX_NG + CX_G_DE;
		for (int c = 0; c < CX_DE; c++) cx_pf0(b + c * 16);
		for (int c = 0; c < CX_DE; c++) b[c * 16] += de[c];
	}
}

/* zero the chunk accumulators */
static inline void cx_train_chunk_begin(cx_train *tr)
{
	cx_zero(tr->dT, CX_NT);
	cx_zero(tr->gcg, CX_DU);
	cx_zero(tr->gb1, CX_DH);
	cx_zero(tr->ggain, CX_DX);
	cx_zero(tr->gwg, CX_DH);
	cx_zero(tr->gbg, 16);
}

/* ---- optimizer and the single-column bus ---- */

/* Adam over n floats (multiple of 16): m, v updated in place; p -= lr_t m / (sqrt(v) + eps_t), with the bias
 * corrections folded into lr_t and eps_t by the caller. sqrt(v) = s rsqrt(s), s = v + 1e-38 (so v = 0 stays finite). */
static inline void cx_adam(float *p, const float *g, float *m, float *v, long n, float b1, float b2, float lr_t, float eps_t)
{
	const __m512 B1 = _mm512_set1_ps(b1), B2 = _mm512_set1_ps(b2), O1 = _mm512_set1_ps(1.0f - b1), O2 = _mm512_set1_ps(1.0f - b2);
	const __m512 LR = _mm512_set1_ps(lr_t), EPS = _mm512_set1_ps(eps_t), TINY = _mm512_set1_ps(1e-38f);
	for (long i = 0; i < n; i += 16) {
		if (i + 64 < n) { cx_pf0(p + i + 64); cx_pf0(g + i + 64); cx_pf0(m + i + 64); cx_pf0(v + i + 64); }
		const __m512 gv = _mm512_load_ps(g + i);
		const __m512 mv = _mm512_fmadd_ps(O1, gv, _mm512_mul_ps(B1, _mm512_load_ps(m + i)));
		const __m512 vv = _mm512_fmadd_ps(O2, _mm512_mul_ps(gv, gv), _mm512_mul_ps(B2, _mm512_load_ps(v + i)));
		_mm512_store_ps(m + i, mv);
		_mm512_store_ps(v + i, vv);
		const __m512 s = _mm512_add_ps(vv, TINY);
		const __m512 den = _mm512_fmadd_ps(s, cx_rsqrt(s), EPS);
		_mm512_store_ps(p + i, cx_fnmadd(LR, _mm512_mul_ps(mv, cx_rcp(den)), _mm512_load_ps(p + i)));
	}
}

/* slot 0 of the bus = this column's own previous output bc (one step delayed); the other slots stay zero */
static inline void cx_bus_set0(float *Bt, const float *bc)
{
	for (int d = 0; d < CX_DB; d++) Bt[d * CX_NBUS] = bc[d];
}

/* single-column step backward: the consensus norm (hc = RMSNorm(u) cg) then the column; dgl = 0 (g has no loss path) */
static inline void cx_train_bwd_step(cx_train *tr, int t, const float *Bt)
{
	const float *A = tr->acts + (long)t * CX_NACT;
	cx_pf_lines(A, CX_NACT);
	float du[CX_DU] __attribute__((aligned(64)));
	cx_norm_bwd(tr->dhc + (long)t * CX_DU, A + CX_A_U, _mm512_load_ps(A + CX_A_XSU), tr->cg, tr->gcg, du, 8, 1.0f / CX_DU);
	cx_col_bwd_step(tr, t, Bt, du, _mm512_setzero_ps());
}

#endif
