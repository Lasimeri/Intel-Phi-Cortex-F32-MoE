/*
 * ctx.h: stage 3, the context memory on the card (2026-10-27 16:2x; PLAN-MOE.md "15:52" and "16:2x": the user's
 * "16K KV cache so the student can remember its previous responses"). Per head a ring of the last CX_CTX_WINDOW
 * (16,384) steps' keys (the fast-weight key k, unit-norm, 64 F32) and values (v, 64 F32) in GDDR, an online IVF
 * index over them (CX_CTX_NC centroids seeded by the head's first keys, each arriving key moving its nearest centroid
 * toward itself by CX_CTX_ETA and filed under it; lists of CX_CTX_CAP steps, the oldest overwritten), and the
 * read-out: the step's query q against the last CX_CTX_RECENT steps exactly and the entries of the CX_CTX_PROBE
 * nearest centroids' lists, the top-4 by dot product, a softmax over their scores, a = sum_i alpha_i v_i, and
 * o' = o + sigmoid(b) a before the MLP, b this head's gate in the gates block (trained by Adam beside the n-gram
 * gates; a fresh gate is CX_CTX_GATE0, sigmoid 0.018, so a trained model continues as it was). The retrieved keys
 * and values are not differentiated (Memorizing Transformers, Wu et al. 2022, arxiv 2203.08913): the backward gives
 * dq through the softmax, da = sigmoid(b) do' and db = (do' . a) sigmoid'(b). Measured on the model's own keys
 * (tools/ctx-recall.c, round 5 of the from-scratch run): recall@4 0.890 at 8 probes, 0.938 at 16, where the exact
 * top-4 lie within the last 256 steps only 2.6 % of the time; the host could not serve this at the step rate
 * (host/ctxd.c: 1.2 to 2.0 ms a step, DRAM-bound), the card's cores and GDDR can: about 700 candidate keys of 256 B
 * a head a step. One thread per head; a head's memory is its own, the gate words are the cluster's. A session's
 * view starts at `start` (CX_OP_CTX_RESET moves it; the training's view keeps its own), so a chat retrieves only
 * its own turns while the training retrieves the stream and the chats alike. The reference: tools/ref.h.
 */
#ifndef CX_CTX_H
#define CX_CTX_H

#include "col.h"
#include "../lib/cx-moe.h"
#include <math.h>
#include <string.h>

#define CX_CTX_WM (CX_CTX_WINDOW - 1)
#define CX_CTX_KEYS_BYTES ((size_t)CX_CTX_WINDOW * CX_DS * 4)
#define CX_CTX_CENT_BYTES ((size_t)CX_CTX_NC * CX_DS * 4)
#define CX_CTX_LISTS_BYTES ((size_t)CX_CTX_NC * CX_CTX_CAP * 4)
#define CX_CTX_HEAD_BYTES (2 * CX_CTX_KEYS_BYTES + CX_CTX_CENT_BYTES + CX_CTX_LISTS_BYTES + 1024)   /* keys, values, centroids, lists, cursors + fills */

typedef struct cx_ctx {
	float *keys;        /* [CX_CTX_WINDOW][64] */
	float *vals;        /* [CX_CTX_WINDOW][64] */
	float *cent;        /* [CX_CTX_NC][64], unit length */
	uint32_t *lists;    /* [CX_CTX_NC][CX_CTX_CAP] step numbers; the slot of step s is s & CX_CTX_WM */
	uint16_t *cursor;   /* [CX_CTX_NC] the next entry to overwrite */
	uint16_t *fill;     /* [CX_CTX_NC] entries written, up to CX_CTX_CAP */
	int seeded;         /* centroids seeded so far */
	int64_t n;          /* steps stored: the next step's number */
	int64_t train_start, sess_start;   /* the first step a query may return, for the training and for a session */
	float *gate;        /* this head's gate b, in the gates block */
	float *ggate;       /* its gradient for the chunk, in the gates gradient block */
} cx_ctx;

/* one head's memory carved from `mem` (CX_CTX_HEAD_BYTES, 64-byte aligned, zeroed by the caller) */
static inline void cx_ctx_init(cx_ctx *X, void *mem, float *gate, float *ggate)
{
	uint8_t *p = mem;
	X->keys = (float *)p; p += CX_CTX_KEYS_BYTES;
	X->vals = (float *)p; p += CX_CTX_KEYS_BYTES;
	X->cent = (float *)p; p += CX_CTX_CENT_BYTES;
	X->lists = (uint32_t *)p; p += CX_CTX_LISTS_BYTES;
	X->cursor = (uint16_t *)p; p += (size_t)CX_CTX_NC * 2;
	X->fill = (uint16_t *)p;
	memset(X->cursor, 0, CX_CTX_NC * 2);
	memset(X->fill, 0, CX_CTX_NC * 2);
	X->seeded = 0;
	X->n = 0;
	X->train_start = 0;
	X->sess_start = 0;
	X->gate = gate;
	X->ggate = ggate;
}

/* new weights: nothing stored so far is this model's; the ring stays, both views start at the next step, the index is seeded afresh */
static inline void cx_ctx_clear(cx_ctx *X)
{
	memset(X->cursor, 0, CX_CTX_NC * 2);
	memset(X->fill, 0, CX_CTX_NC * 2);
	X->seeded = 0;
	X->train_start = X->n;
	X->sess_start = X->n;
}

static inline void cx_ctx_pf(const float *k) { for (int i = 0; i < CX_DS; i += 16) cx_pf0(k + i); }

/* q (4 vectors) . k over 64 lanes, as one float */
static inline float cx_ctx_dot(const __m512 *q, const float *k, float *tmp)
{
	__m512 acc = _mm512_mul_ps(q[0], _mm512_load_ps(k));
	acc = _mm512_fmadd_ps(q[1], _mm512_load_ps(k + 16), acc);
	acc = _mm512_fmadd_ps(q[2], _mm512_load_ps(k + 32), acc);
	acc = _mm512_fmadd_ps(q[3], _mm512_load_ps(k + 48), acc);
	_mm512_store_ps(tmp, cx_hsum(acc));
	return tmp[0];
}

typedef struct { float s[CX_CTX_TOP]; int64_t step[CX_CTX_TOP]; } cx_ctx_top;

static inline void cx_ctx_top_insert(cx_ctx_top *t, float s, int64_t step)
{
	int i = CX_CTX_TOP - 1;
	if (s <= t->s[i]) return;
	for (; i > 0 && s > t->s[i - 1]; i--) { t->s[i] = t->s[i - 1]; t->step[i] = t->step[i - 1]; }
	t->s[i] = s;
	t->step[i] = step;
}

static inline void cx_ctx_file(cx_ctx *X, int c, int64_t n)
{
	X->lists[(size_t)c * CX_CTX_CAP + X->cursor[c]] = (uint32_t)n;
	X->cursor[c] = (uint16_t)((X->cursor[c] + 1) % CX_CTX_CAP);
	if (X->fill[c] < CX_CTX_CAP) X->fill[c]++;
}

/* the step's read-out into the activation block A (q at CX_A_Y, k at CX_A_K, v at CX_A_Y + 128, o at CX_A_MIN): the
 * top-4 of the memory from `start` on, the softmax, a, o += sigmoid(b) a, the saves; then this step's k, v stored
 * and indexed */
static inline void cx_ctx_step(cx_ctx *X, float *A, int64_t start)
{
	float tmp[16] __attribute__((aligned(64)));
	const float *qf = A + CX_A_Y, *k = A + CX_A_K, *v = A + CX_A_Y + 128;
	__m512 q[4];
	for (int j = 0; j < 4; j++) q[j] = _mm512_load_ps(qf + 16 * j);
	cx_ctx_top top;
	for (int i = 0; i < CX_CTX_TOP; i++) { top.s[i] = -3.0e38f; top.step[i] = -1; }
	const int64_t n = X->n;
	const int64_t oldest = n - CX_CTX_WINDOW > start ? n - CX_CTX_WINDOW : start;
	/* the exact part: the last RECENT steps, the keys prefetched 8 ahead */
	int64_t lo = n - CX_CTX_RECENT;
	if (lo < oldest) lo = oldest;
	for (int64_t s = lo; s < n; s++) {
		if (s + 8 < n) cx_ctx_pf(X->keys + (size_t)((s + 8) & CX_CTX_WM) * CX_DS);
		cx_ctx_top_insert(&top, cx_ctx_dot(q, X->keys + (size_t)(s & CX_CTX_WM) * CX_DS, tmp), s);
	}
	/* the far part: the PROBE nearest centroids' lists, their entries between the oldest and the exact window */
	const int64_t far_hi = n - CX_CTX_RECENT;
	if (X->seeded == CX_CTX_NC && far_hi > oldest) {
		float cs[CX_CTX_NC] __attribute__((aligned(64)));
		for (int c = 0; c < CX_CTX_NC; c++) cs[c] = cx_ctx_dot(q, X->cent + (size_t)c * CX_DS, tmp);
		for (int p = 0; p < CX_CTX_PROBE; p++) {
			int best = -1;
			float bs = -3.0e38f;
			for (int c = 0; c < CX_CTX_NC; c++) if (cs[c] > bs) { bs = cs[c]; best = c; }
			if (best < 0) break;
			cs[best] = -3.0e38f;
			const uint32_t *L = X->lists + (size_t)best * CX_CTX_CAP;
			const int m = X->fill[best];
			for (int i = 0; i < m; i++) {
				const int64_t s = L[i];
				if (s >= far_hi || s < oldest) continue;
				if (i + 4 < m) { const int64_t s2 = L[i + 4]; if (s2 < far_hi && s2 >= oldest) cx_ctx_pf(X->keys + (size_t)(s2 & CX_CTX_WM) * CX_DS); }
				cx_ctx_top_insert(&top, cx_ctx_dot(q, X->keys + (size_t)(s & CX_CTX_WM) * CX_DS, tmp), s);
			}
		}
	}
	/* the softmax over the found scores, the read-out a, the gate, the saves */
	int m = 0;
	float mx = -3.0e38f;
	for (int i = 0; i < CX_CTX_TOP; i++) if (top.step[i] >= 0) { m++; if (top.s[i] > mx) mx = top.s[i]; }
	uint32_t *idx = (uint32_t *)(A + CX_A_CTX_IDX);
	float *al = A + CX_A_CTX_AL;
	for (int i = 0; i < 16; i++) { idx[i] = CX_CTX_NONE; al[i] = 0; }
	__m512 a[4];
	for (int j = 0; j < 4; j++) a[j] = _mm512_setzero_ps();
	if (m) {
		float e[CX_CTX_TOP], sum = 0;
		for (int i = 0; i < m; i++) { e[i] = expf(top.s[i] - mx); sum += e[i]; }
		const float inv = 1.0f / sum;
		for (int i = 0; i < m; i++) {
			al[i] = e[i] * inv;
			idx[i] = (uint32_t)top.step[i];
			const float *vi = X->vals + (size_t)(top.step[i] & CX_CTX_WM) * CX_DS;
			const __m512 w = _mm512_set1_ps(al[i]);
			for (int j = 0; j < 4; j++) a[j] = _mm512_fmadd_ps(w, _mm512_load_ps(vi + 16 * j), a[j]);
		}
	}
	const float g = 1.0f / (1.0f + expf(-*X->gate));
	const __m512 gv = _mm512_set1_ps(g);
	_mm512_store_ps(A + CX_A_CTX_G, gv);
	for (int j = 0; j < 4; j++) {
		_mm512_store_ps(A + CX_A_CTX_A + 16 * j, a[j]);
		_mm512_store_ps(A + CX_A_MIN + 16 * j, _mm512_fmadd_ps(gv, a[j], _mm512_load_ps(A + CX_A_MIN + 16 * j)));
	}
	/* this step into the ring and the index */
	const size_t slot = (size_t)(n & CX_CTX_WM);
	float *ks = X->keys + slot * CX_DS, *vs = X->vals + slot * CX_DS;
	__m512 kk[4];
	for (int j = 0; j < 4; j++) {
		kk[j] = _mm512_load_ps(k + 16 * j);
		_mm512_store_ps(ks + 16 * j, kk[j]);
		_mm512_store_ps(vs + 16 * j, _mm512_load_ps(v + 16 * j));
	}
	if (X->seeded < CX_CTX_NC) {
		float *c = X->cent + (size_t)X->seeded * CX_DS;
		for (int j = 0; j < 4; j++) _mm512_store_ps(c + 16 * j, kk[j]);
		cx_ctx_file(X, X->seeded, n);
		X->seeded++;
	} else {
		int best = 0;
		float bs = -3.0e38f;
		for (int c = 0; c < CX_CTX_NC; c++) { const float s = cx_ctx_dot(kk, X->cent + (size_t)c * CX_DS, tmp); if (s > bs) { bs = s; best = c; } }
		float *c = X->cent + (size_t)best * CX_DS;
		const __m512 eta = _mm512_set1_ps(CX_CTX_ETA);
		__m512 nn = _mm512_setzero_ps(), cv[4];
		for (int j = 0; j < 4; j++) {
			cv[j] = _mm512_load_ps(c + 16 * j);
			cv[j] = _mm512_fmadd_ps(eta, _mm512_sub_ps(kk[j], cv[j]), cv[j]);
			nn = _mm512_fmadd_ps(cv[j], cv[j], nn);
		}
		_mm512_store_ps(tmp, cx_hsum(nn));
		const __m512 inv = _mm512_set1_ps(tmp[0] > 0 ? 1.0f / sqrtf(tmp[0]) : 0.0f);
		for (int j = 0; j < 4; j++) _mm512_store_ps(c + 16 * j, _mm512_mul_ps(cv[j], inv));
		cx_ctx_file(X, best, n);
	}
	X->n = n + 1;
}

/* the memory's backward for the step whose activation block is A: dO = dL/do' (64); dq (64) gets the softmax path
 * added; the gate's gradient accumulates over the chunk */
static inline void cx_ctx_bwd(cx_ctx *X, const float *A, const float *dO, float *dq)
{
	float tmp[16] __attribute__((aligned(64)));
	const float g = A[CX_A_CTX_G];
	const uint32_t *idx = (const uint32_t *)(A + CX_A_CTX_IDX);
	const float *al = A + CX_A_CTX_AL;
	__m512 dOv[4], dg = _mm512_setzero_ps();
	for (int j = 0; j < 4; j++) { dOv[j] = _mm512_load_ps(dO + 16 * j); dg = _mm512_fmadd_ps(dOv[j], _mm512_load_ps(A + CX_A_CTX_A + 16 * j), dg); }
	_mm512_store_ps(tmp, cx_hsum(dg));
	*X->ggate += tmp[0] * g * (1.0f - g);
	/* da = g dO; dalpha_i = da . v_i; ds_i = alpha_i (dalpha_i - sum_j alpha_j dalpha_j); dq += sum_i ds_i k_i */
	float dal[CX_CTX_TOP], dot = 0;
	int m = 0;
	for (int i = 0; i < CX_CTX_TOP; i++) {
		if (idx[i] == CX_CTX_NONE) break;
		m++;
		const float *vi = X->vals + (size_t)(idx[i] & CX_CTX_WM) * CX_DS;
		__m512 acc = _mm512_setzero_ps();
		for (int j = 0; j < 4; j++) acc = _mm512_fmadd_ps(dOv[j], _mm512_load_ps(vi + 16 * j), acc);
		_mm512_store_ps(tmp, cx_hsum(acc));
		dal[i] = g * tmp[0];
		dot += al[i] * dal[i];
	}
	for (int i = 0; i < m; i++) {
		const __m512 w = _mm512_set1_ps(al[i] * (dal[i] - dot));
		const float *ki = X->keys + (size_t)(idx[i] & CX_CTX_WM) * CX_DS;
		for (int j = 0; j < 4; j++) _mm512_store_ps(dq + 16 * j, _mm512_fmadd_ps(w, _mm512_load_ps(ki + 16 * j), _mm512_load_ps(dq + 16 * j)));
	}
}

#endif
