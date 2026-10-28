/*
 * ref.h: host double reference of one column's TRAINING step (Stage 3), reused per column by the cluster reference
 * (Stage 4, tools/cref.h). Header only, gcc. The forward follows card/col.h exactly (same shapes, lib/cx-col.h).
 *
 * Single-column loss path (the cluster formula with one member): hc = RMSNorm(u) cg (cg[128] shared like E),
 * logits = E hc (tied head), loss = mean over the chunk of CE(logits, next token). The bus input Bt is stop-gradient.
 * T at the chunk start is a constant (truncated BPTT); dT is carried backwards inside the chunk.
 *
 * ref_col_backward takes the gradient on the vote u and on the confidence pre-activation gl, so the cluster can feed
 * du_c = g_c dhs and dg_c = u_c . dhs; with one column g has no loss path (dgl = 0).
 */
#ifndef CX_REF_H
#define CX_REF_H

#include "../lib/cx-col.h"
#include "../lib/cx-moe.h"   /* stage 3: the context memory's shapes (CX_CTX_*) */

#include <math.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
	uint32_t id;
	const float *Bt;
	double p[CX_DP], qb[CX_DB], alpha[CX_NBUS], xin[CX_DX], xs, x[CX_DX], y[CX_NW], k[CX_DS], ks, a, b;
	double Sk[CX_DS], Sq[CX_DS], d[CX_DS], kq, o[CX_DS], h[CX_DH], sg[CX_DH], z[CX_DH], u[CX_DU], g, gl, bc[CX_DB], xsu, hc[CX_DU];
	double T0[CX_NT];   /* T before this step's update */
	double e[CX_DE];    /* the input actually used (E row, plus the n-gram rows when the memory is on) */
	/* stage 3 (the context memory): the retrieved steps (CX_CTX_NONE past the found), their softmax weights, the read-out
	 * a, the gate sigmoid(b); o holds o + gate a once ref_ctx_step ran */
	uint32_t cidx[CX_CTX_TOP];
	double cal[CX_CTX_TOP], ca[CX_DS], cgate;
} rstep;

typedef struct { double *prm, *E, *cg; } rparams;   /* CX_NPARAM, 4096 x 128, 128 */

static inline double rsig(double x) { return 1.0 / (1.0 + exp(-x)); }

static inline void rparams_alloc(rparams *P)
{
	P->prm = calloc(CX_NPARAM, sizeof *P->prm);
	P->E = calloc((size_t)CX_VOCAB * CX_DE, sizeof *P->E);
	P->cg = calloc(CX_DU, sizeof *P->cg);
}

/* the same F32 init as the card (column k), widened to double; E and cg shared */
static inline void rparams_init_col(double *prm, int k)
{
	float *fp = malloc(CX_NPARAM * sizeof *fp);
	cx_col_init(fp, k);
	for (long i = 0; i < CX_NPARAM; i++) prm[i] = fp[i];
	free(fp);
}

static inline void rparams_init_shared(rparams *P)
{
	float *fe = malloc((size_t)CX_VOCAB * CX_DE * sizeof *fe);
	cx_emb_init(fe);
	for (long i = 0; i < (long)CX_VOCAB * CX_DE; i++) P->E[i] = fe[i];
	for (int i = 0; i < CX_DU; i++) P->cg[i] = 1.0;
	free(fe);
}

static inline void rparams_init(rparams *P, int k)
{
	rparams_init_col(P->prm, k);
	rparams_init_shared(P);
}

/* y[N] = x[K]^T W[K][N] */
static inline void rgemv(const double *W, const double *x, int K, int N, double *y)
{
	for (int n = 0; n < N; n++) y[n] = 0;
	for (int k = 0; k < K; k++) for (int n = 0; n < N; n++) y[n] += x[k] * W[(long)k * N + n];
}

/* the column forward in two halves (the MoE reference routes and sets the expert between them, tools/cref.h): the first
 * through the fast weights to [o ; x] (T, 64x64, T = S^T, updated in place), the second the MLP and the outputs;
 * everything the backward needs saved */
static inline void ref_col_pre(const double *prm, const double *E, double *T, rstep *s, uint32_t id, const float *Bt, const double *ein)
{
	s->id = id;
	s->Bt = Bt;
	memcpy(s->T0, T, sizeof s->T0);
	double *e = s->e;
	for (int i = 0; i < CX_DE; i++) e[i] = ein ? ein[i] : E[(long)id * CX_DE + i];
	rgemv(prm + CX_O_PIN, e, CX_DE, CX_DP, s->p);
	rgemv(prm + CX_O_WQB, s->p, CX_DP, CX_DB, s->qb);
	double sc[CX_NBUS], m = -INFINITY, sum = 0;
	for (int j = 0; j < CX_NBUS; j++) { sc[j] = 0; for (int d = 0; d < CX_DB; d++) sc[j] += Bt[d * CX_NBUS + j] * s->qb[d]; }
	for (int j = 0; j < CX_NCOL; j++) if (sc[j] > m) m = sc[j];
	for (int j = 0; j < CX_NBUS; j++) { s->alpha[j] = j < CX_NCOL ? exp(sc[j] - m) : 0; sum += s->alpha[j]; }
	for (int j = 0; j < CX_NBUS; j++) s->alpha[j] /= sum;
	double ss = 0;
	for (int i = 0; i < CX_DP; i++) s->xin[i] = s->p[i];
	for (int d = 0; d < CX_DB; d++) { double r = 0; for (int j = 0; j < CX_NBUS; j++) r += s->alpha[j] * Bt[d * CX_NBUS + j]; s->xin[CX_DP + d] = r; }
	for (int i = 0; i < CX_DX; i++) ss += s->xin[i] * s->xin[i];
	s->xs = 1.0 / sqrt(ss / CX_DX + (double)CX_EPS);
	for (int i = 0; i < CX_DX; i++) s->x[i] = s->xin[i] * s->xs * prm[CX_O_GAIN + i];
	rgemv(prm + CX_O_W, s->x, CX_DX, CX_NW, s->y);
	double kn = 0;
	for (int i = 0; i < CX_DS; i++) kn += s->y[64 + i] * s->y[64 + i];
	s->ks = 1.0 / sqrt(kn + (double)CX_EPS);
	for (int i = 0; i < CX_DS; i++) s->k[i] = s->y[64 + i] * s->ks;
	s->a = rsig(s->y[192]);
	s->b = rsig(s->y[193]);
	for (int i = 0; i < CX_DS; i++) s->Sk[i] = s->Sq[i] = 0;
	for (int j = 0; j < CX_DS; j++)
		for (int i = 0; i < CX_DS; i++) { s->Sk[i] += s->k[j] * T[j * CX_DS + i]; s->Sq[i] += s->y[j] * T[j * CX_DS + i]; }
	s->kq = 0;
	for (int i = 0; i < CX_DS; i++) { s->d[i] = s->b * (s->y[128 + i] - s->a * s->Sk[i]); s->kq += s->k[i] * s->y[i]; }
	for (int i = 0; i < CX_DS; i++) s->o[i] = s->a * s->Sq[i] + s->kq * s->d[i];
	for (int j = 0; j < CX_DS; j++) for (int i = 0; i < CX_DS; i++) T[j * CX_DS + i] = s->a * T[j * CX_DS + i] + s->k[j] * s->d[i];
}

static inline void ref_col_mlp(const double *prm, rstep *s)
{
	double min[CX_DM];
	for (int i = 0; i < CX_DS; i++) min[i] = s->o[i];
	for (int i = 0; i < CX_DX; i++) min[CX_DS + i] = s->x[i];
	rgemv(prm + CX_O_W1, min, CX_DM, CX_DH, s->h);
	for (int i = 0; i < CX_DH; i++) { s->h[i] += prm[CX_O_B1 + i]; s->sg[i] = rsig(s->h[i]); s->z[i] = s->h[i] * s->sg[i]; }
	rgemv(prm + CX_O_WU, s->z, CX_DH, CX_DU, s->u);
	s->gl = prm[CX_O_BG];
	for (int i = 0; i < CX_DH; i++) s->gl += s->z[i] * prm[CX_O_WG + i];
	s->g = rsig(s->gl);
	rgemv(prm + CX_O_WB, s->z, CX_DH, CX_DB, s->bc);
}

/* the dense column's forward: both halves */
static inline void ref_col_forward(const double *prm, const double *E, double *T, rstep *s, uint32_t id, const float *Bt, const double *ein)
{
	ref_col_pre(prm, E, T, s, id, Bt, ein);
	ref_col_mlp(prm, s);
}

/* single column: forward plus the consensus norm of its own vote */
static inline void ref_forward(const rparams *P, double *T, rstep *s, uint32_t id, const float *Bt)
{
	ref_col_forward(P->prm, P->E, T, s, id, Bt, NULL);
	double su = 0;
	for (int i = 0; i < CX_DU; i++) su += s->u[i] * s->u[i];
	s->xsu = 1.0 / sqrt(su / CX_DU + (double)CX_EPS);
	for (int i = 0; i < CX_DU; i++) s->hc[i] = s->u[i] * s->xsu * P->cg[i];
}

/* tied head + CE for one hc: returns the (hard) loss; dl = softmax - target (unscaled), the target onehot(tgt) or, with
 * a soft record of the partial distillation (lib/cx-proto.h cx_soft), (1 - sum p_j) onehot(tgt) + sum_j p_j onehot(id_j) */
static inline double ref_head_hc(const double *E, const double *hc, uint32_t tgt, double *dl, const cx_soft *sr)
{
	double m = -INFINITY, sum = 0;
	for (int v = 0; v < CX_VOCAB; v++) {
		double l = 0;
		const double *row = E + (long)v * CX_DE;
		for (int c = 0; c < CX_DE; c++) l += row[c] * hc[c];
		dl[v] = l;
		if (l > m) m = l;
	}
	for (int v = 0; v < CX_VOCAB; v++) { dl[v] = exp(dl[v] - m); sum += dl[v]; }
	const double loss = log(sum) + m - (log(dl[tgt]) + m);
	for (int v = 0; v < CX_VOCAB; v++) dl[v] /= sum;
	double w = 1.0;
	if (sr) for (int j = 0; j < CX_SOFT_K; j++) if (sr->id[j] != CX_SOFT_NONE) { dl[sr->id[j]] -= sr->p[j]; w -= sr->p[j]; }
	dl[tgt] -= w;
	return loss;
}

static inline double ref_head(const rparams *P, const rstep *s, uint32_t tgt, double *dl) { return ref_head_hc(P->E, s->hc, tgt, dl, NULL); }

typedef struct { double *prm, *E, *cg; } rgrads;

static inline void rgrads_alloc(rgrads *G)
{
	G->prm = calloc(CX_NPARAM, sizeof *G->prm);
	G->E = calloc((size_t)CX_VOCAB * CX_DE, sizeof *G->E);
	G->cg = calloc(CX_DU, sizeof *G->cg);
}

static inline void rgrads_zero(rgrads *G)
{
	memset(G->prm, 0, CX_NPARAM * sizeof *G->prm);
	memset(G->E, 0, (size_t)CX_VOCAB * CX_DE * sizeof *G->E);
	memset(G->cg, 0, CX_DU * sizeof *G->cg);
}

/* head backward: dhc += E^T dl, gE += dl hc^T */
static inline void ref_head_backward(const double *E, double *gE, const double *hc, const double *dl, double *dhc)
{
	for (int v = 0; v < CX_VOCAB; v++) {
		if (dl[v] == 0) continue;
		const double *row = E + (long)v * CX_DE;
		double *grow = gE + (long)v * CX_DE;
		for (int c = 0; c < CX_DE; c++) { dhc[c] += row[c] * dl[v]; grow[c] += dl[v] * hc[c]; }
	}
}

/* RMSNorm-with-gain backward: x = n gain, n = xin xs; returns dxin, adds dx n into ggain */
static inline void ref_norm_backward(const double *dx, const double *xin, double xs, const double *gain, double *ggain, double *dxin, int N)
{
	double dot = 0, *dn = malloc((size_t)N * sizeof *dn);
	for (int i = 0; i < N; i++) { const double n = xin[i] * xs; dn[i] = dx[i] * gain[i]; ggain[i] += dx[i] * n; dot += dn[i] * n; }
	dot /= N;
	for (int i = 0; i < N; i++) dxin[i] = xs * (dn[i] - xin[i] * xs * dot);
	free(dn);
}

/* ---- stage 3: the context memory, the card's algorithm (card/ctx.h) in double ---- */
typedef struct {
	double *keys, *vals;                   /* [CX_CTX_WINDOW][64], the slot of step s is s & (CX_CTX_WINDOW - 1) */
	double cent[CX_CTX_NC][CX_DS];         /* unit centroids */
	uint32_t *lists;                       /* [CX_CTX_NC][CX_CTX_CAP] steps */
	uint16_t cursor[CX_CTX_NC], fill[CX_CTX_NC];
	int seeded;
	long n;
} rctx;

static inline void rctx_init(rctx *X)
{
	X->keys = calloc((size_t)CX_CTX_WINDOW * CX_DS, sizeof *X->keys);
	X->vals = calloc((size_t)CX_CTX_WINDOW * CX_DS, sizeof *X->vals);
	X->lists = calloc((size_t)CX_CTX_NC * CX_CTX_CAP, sizeof *X->lists);
	memset(X->cursor, 0, sizeof X->cursor);
	memset(X->fill, 0, sizeof X->fill);
	memset(X->cent, 0, sizeof X->cent);
	X->seeded = 0;
	X->n = 0;
}

static inline double rdot64(const double *a, const double *b) { double s = 0; for (int i = 0; i < CX_DS; i++) s += a[i] * b[i]; return s; }

typedef struct { double s[CX_CTX_TOP]; long step[CX_CTX_TOP]; } rtop;

static inline void rtop_insert(rtop *t, double s, long step)
{
	int i = CX_CTX_TOP - 1;
	if (s <= t->s[i]) return;
	for (; i > 0 && s > t->s[i - 1]; i--) { t->s[i] = t->s[i - 1]; t->step[i] = t->step[i - 1]; }
	t->s[i] = s;
	t->step[i] = step;
}

static inline void rctx_file(rctx *X, int c, long n)
{
	X->lists[(size_t)c * CX_CTX_CAP + X->cursor[c]] = (uint32_t)n;
	X->cursor[c] = (uint16_t)((X->cursor[c] + 1) % CX_CTX_CAP);
	if (X->fill[c] < CX_CTX_CAP) X->fill[c]++;
}

/* the read-out for the step record s (q = y[0..63], k, v = y[128..191]): the top-4 of the memory from `start` on, the
 * softmax, a, o += sigmoid(gate) a, the saves; then this step's k, v stored and indexed */
static inline void ref_ctx_step(rctx *X, rstep *s, long start, double gate)
{
	const long WM = CX_CTX_WINDOW - 1;
	const double *q = s->y, *k = s->k, *v = s->y + 128;
	rtop top;
	for (int i = 0; i < CX_CTX_TOP; i++) { top.s[i] = -3.0e38; top.step[i] = -1; }
	const long n = X->n;
	const long oldest = n - CX_CTX_WINDOW > start ? n - CX_CTX_WINDOW : start;
	long lo = n - CX_CTX_RECENT;
	if (lo < oldest) lo = oldest;
	for (long i = lo; i < n; i++) rtop_insert(&top, rdot64(q, X->keys + (size_t)(i & WM) * CX_DS), i);
	const long far_hi = n - CX_CTX_RECENT;
	if (X->seeded == CX_CTX_NC && far_hi > oldest) {
		double cs[CX_CTX_NC];
		for (int c = 0; c < CX_CTX_NC; c++) cs[c] = rdot64(q, X->cent[c]);
		for (int p = 0; p < CX_CTX_PROBE; p++) {
			int best = -1;
			double bs = -3.0e38;
			for (int c = 0; c < CX_CTX_NC; c++) if (cs[c] > bs) { bs = cs[c]; best = c; }
			if (best < 0) break;
			cs[best] = -3.0e38;
			const uint32_t *L = X->lists + (size_t)best * CX_CTX_CAP;
			for (int i = 0; i < X->fill[best]; i++) {
				const long st = L[i];
				if (st >= far_hi || st < oldest) continue;
				rtop_insert(&top, rdot64(q, X->keys + (size_t)(st & WM) * CX_DS), st);
			}
		}
	}
	int m = 0;
	double mx = -3.0e38;
	for (int i = 0; i < CX_CTX_TOP; i++) if (top.step[i] >= 0) { m++; if (top.s[i] > mx) mx = top.s[i]; }
	for (int i = 0; i < CX_CTX_TOP; i++) { s->cidx[i] = CX_CTX_NONE; s->cal[i] = 0; }
	for (int i = 0; i < CX_DS; i++) s->ca[i] = 0;
	if (m) {
		double e[CX_CTX_TOP], sum = 0;
		for (int i = 0; i < m; i++) { e[i] = exp(top.s[i] - mx); sum += e[i]; }
		for (int i = 0; i < m; i++) {
			s->cal[i] = e[i] / sum;
			s->cidx[i] = (uint32_t)top.step[i];
			const double *vi = X->vals + (size_t)(top.step[i] & WM) * CX_DS;
			for (int j = 0; j < CX_DS; j++) s->ca[j] += s->cal[i] * vi[j];
		}
	}
	s->cgate = rsig(gate);
	for (int j = 0; j < CX_DS; j++) s->o[j] += s->cgate * s->ca[j];
	/* this step into the ring and the index */
	double *ks = X->keys + (size_t)(n & WM) * CX_DS, *vs = X->vals + (size_t)(n & WM) * CX_DS;
	for (int j = 0; j < CX_DS; j++) { ks[j] = k[j]; vs[j] = v[j]; }
	if (X->seeded < CX_CTX_NC) {
		for (int j = 0; j < CX_DS; j++) X->cent[X->seeded][j] = k[j];
		rctx_file(X, X->seeded, n);
		X->seeded++;
	} else {
		int best = 0;
		double bs = -3.0e38;
		for (int c = 0; c < CX_CTX_NC; c++) { const double sc = rdot64(k, X->cent[c]); if (sc > bs) { bs = sc; best = c; } }
		double nn = 0;
		for (int j = 0; j < CX_DS; j++) { X->cent[best][j] += (double)CX_CTX_ETA * (k[j] - X->cent[best][j]); nn += X->cent[best][j] * X->cent[best][j]; }
		const double inv = nn > 0 ? 1.0 / sqrt(nn) : 0;
		for (int j = 0; j < CX_DS; j++) X->cent[best][j] *= inv;
		rctx_file(X, best, n);
	}
	X->n = n + 1;
}

/* the memory's backward: dO on o'; the softmax path added to dq; returns the gate's gradient d(loss)/d b */
static inline double ref_ctx_bwd(const rctx *X, const rstep *s, const double *dO, double *dq)
{
	const long WM = CX_CTX_WINDOW - 1;
	const double g = s->cgate;
	double dg = 0, dal[CX_CTX_TOP], dot = 0;
	for (int j = 0; j < CX_DS; j++) dg += dO[j] * s->ca[j];
	int m = 0;
	for (int i = 0; i < CX_CTX_TOP; i++) {
		if (s->cidx[i] == CX_CTX_NONE) break;
		m++;
		dal[i] = g * rdot64(dO, X->vals + (size_t)(s->cidx[i] & WM) * CX_DS);
		dot += s->cal[i] * dal[i];
	}
	for (int i = 0; i < m; i++) {
		const double ds = s->cal[i] * (dal[i] - dot);
		const double *ki = X->keys + (size_t)(s->cidx[i] & WM) * CX_DS;
		for (int j = 0; j < CX_DS; j++) dq[j] += ds * ki[j];
	}
	return dg * g * (1.0 - g);
}

/* backward of one step given the gradient on the vote u (du) and on the confidence pre-activation gl (dgl). dT: in =
 * gradient on T after this step's update (from the future), out = gradient on T before it. gE: the input row's gradient.
 * X, gctx: the context memory and its gate's gradient (stage 3), or NULL. */
static inline void ref_col_backward(const double *prm, double *gp, const double *E, double *gE, const rstep *s, const double *du, double dgl, double *dT, double *de_out, const rctx *X, double *gctx)
{
	/* u = Wu^T z; g = sigmoid(z . wg + bg) */
	double dz[CX_DH], dh[CX_DH];
	for (int i = 0; i < CX_DH; i++) {
		dz[i] = dgl * prm[CX_O_WG + i];
		gp[CX_O_WG + i] += s->z[i] * dgl;
		for (int c = 0; c < CX_DU; c++) { gp[CX_O_WU + i * CX_DU + c] += s->z[i] * du[c]; dz[i] += prm[CX_O_WU + i * CX_DU + c] * du[c]; }
	}
	gp[CX_O_BG] += dgl;
	/* z = h sg */
	for (int i = 0; i < CX_DH; i++) dh[i] = dz[i] * (s->sg[i] + s->h[i] * s->sg[i] * (1.0 - s->sg[i]));
	/* h = W1^T min + b1 */
	double min[CX_DM], dmin[CX_DM];
	for (int i = 0; i < CX_DS; i++) min[i] = s->o[i];
	for (int i = 0; i < CX_DX; i++) min[CX_DS + i] = s->x[i];
	for (int i = 0; i < CX_DM; i++) {
		dmin[i] = 0;
		for (int c = 0; c < CX_DH; c++) { gp[CX_O_W1 + i * CX_DH + c] += min[i] * dh[c]; dmin[i] += prm[CX_O_W1 + i * CX_DH + c] * dh[c]; }
	}
	for (int c = 0; c < CX_DH; c++) gp[CX_O_B1 + c] += dh[c];
	const double *dO = dmin;
	double dx[CX_DX];
	for (int i = 0; i < CX_DX; i++) dx[i] = dmin[CX_DS + i];
	/* fast weights: o = a Sq + kq d; d = b (v - a Sk); T' = a T0 + k d^T; Sk = T0^T k; Sq = T0^T q; kq = k . q */
	double da = 0, db = 0, dkq = 0, dd[CX_DS], dk[CX_DS], dq[CX_DS], dSq[CX_DS], dSk[CX_DS], dv[CX_DS];
	const double *q = s->y, *v = s->y + 128;
	for (int i = 0; i < CX_DS; i++) { da += dO[i] * s->Sq[i]; dkq += dO[i] * s->d[i]; dSq[i] = s->a * dO[i]; dd[i] = dO[i] * s->kq; }
	for (int j = 0; j < CX_DS; j++)
		for (int i = 0; i < CX_DS; i++) { const double g = dT[j * CX_DS + i]; da += g * s->T0[j * CX_DS + i]; dd[i] += g * s->k[j]; }
	for (int j = 0; j < CX_DS; j++) { dk[j] = dkq * q[j]; for (int i = 0; i < CX_DS; i++) dk[j] += dT[j * CX_DS + i] * s->d[i]; }
	for (int i = 0; i < CX_DS; i++) { dv[i] = s->b * dd[i]; db += dd[i] * (v[i] - s->a * s->Sk[i]); da += dd[i] * (-s->b * s->Sk[i]); dSk[i] = -s->a * s->b * dd[i]; }
	for (int j = 0; j < CX_DS; j++) {
		dq[j] = dkq * s->k[j];
		for (int i = 0; i < CX_DS; i++) { dk[j] += s->T0[j * CX_DS + i] * dSk[i]; dq[j] += s->T0[j * CX_DS + i] * dSq[i]; }
	}
	if (X) *gctx += ref_ctx_bwd(X, s, dO, dq);   /* stage 3: the softmax path into dq, the gate's gradient */
	for (int j = 0; j < CX_DS; j++)
		for (int i = 0; i < CX_DS; i++) dT[j * CX_DS + i] = s->a * dT[j * CX_DS + i] + q[j] * dSq[i] + s->k[j] * dSk[i];
	/* k = kr ks */
	double kdk = 0;
	for (int i = 0; i < CX_DS; i++) kdk += dk[i] * s->k[i];
	double dy[CX_NW] = { 0 };
	for (int i = 0; i < CX_DS; i++) { dy[i] = dq[i]; dy[64 + i] = s->ks * (dk[i] - s->k[i] * kdk); dy[128 + i] = dv[i]; }
	dy[192] = da * s->a * (1.0 - s->a);
	dy[193] = db * s->b * (1.0 - s->b);
	/* y = W^T x */
	for (int i = 0; i < CX_DX; i++)
		for (int c = 0; c < CX_NW; c++) { gp[CX_O_W + i * CX_NW + c] += s->x[i] * dy[c]; dx[i] += prm[CX_O_W + i * CX_NW + c] * dy[c]; }
	/* x = RMSNorm(xin) gain */
	double dxin[CX_DX];
	ref_norm_backward(dx, s->xin, s->xs, prm + CX_O_GAIN, gp + CX_O_GAIN, dxin, CX_DX);
	/* r = alpha^T Bt (Bt stop-gradient); softmax; s = Bt^T qb */
	double dalpha[CX_NBUS], ds[CX_NBUS], dqb[CX_DB], dp[CX_DP], adot = 0;
	for (int j = 0; j < CX_NBUS; j++) { dalpha[j] = 0; for (int d = 0; d < CX_DB; d++) dalpha[j] += dxin[CX_DP + d] * s->Bt[d * CX_NBUS + j]; adot += s->alpha[j] * dalpha[j]; }
	for (int j = 0; j < CX_NBUS; j++) ds[j] = s->alpha[j] * (dalpha[j] - adot);
	for (int d = 0; d < CX_DB; d++) { dqb[d] = 0; for (int j = 0; j < CX_NBUS; j++) dqb[d] += s->Bt[d * CX_NBUS + j] * ds[j]; }
	/* qb = Wqb^T p */
	for (int i = 0; i < CX_DP; i++) {
		dp[i] = dxin[i];
		for (int d = 0; d < CX_DB; d++) { gp[CX_O_WQB + i * CX_DB + d] += s->p[i] * dqb[d]; dp[i] += prm[CX_O_WQB + i * CX_DB + d] * dqb[d]; }
	}
	/* p = P_in^T e */
	const double *e = s->e;
	(void)E;
	double *ge = gE + (long)s->id * CX_DE;
	for (int kk = 0; kk < CX_DE; kk++) {
		double de = 0;
		for (int n = 0; n < CX_DP; n++) { gp[CX_O_PIN + kk * CX_DP + n] += e[kk] * dp[n]; de += prm[CX_O_PIN + kk * CX_DP + n] * dp[n]; }
		ge[kk] += de;
		if (de_out) de_out[kk] = de;
	}
}

/* single-column step backward from dl (this step's scaled loss gradient on the logits): head, consensus norm, column */
static inline void ref_backward(const rparams *P, rgrads *G, const rstep *s, const double *dl, double *dT)
{
	double dhc[CX_DU] = { 0 }, du[CX_DU];
	ref_head_backward(P->E, G->E, s->hc, dl, dhc);
	ref_norm_backward(dhc, s->u, s->xsu, P->cg, G->cg, du, CX_DU);
	ref_col_backward(P->prm, G->prm, P->E, G->E, s, du, 0.0, dT, NULL, NULL, NULL);
}

/* loss (mean over L steps) and, if G is not NULL, the gradients, from T0 (copied, not modified) over tokens ids[0..L]
 * (ids[t] in, ids[t+1] target) with bus inputs bts[t] */
static inline double ref_chunk(const rparams *P, rgrads *G, const double *T0, const uint32_t *ids, const float *const *bts, int L, rstep *steps)
{
	double *T = malloc(CX_NT * sizeof *T), *dl = malloc(CX_VOCAB * sizeof *dl), *dT = calloc(CX_NT, sizeof *dT);
	memcpy(T, T0, CX_NT * sizeof *T);
	double loss = 0;
	for (int t = 0; t < L; t++) ref_forward(P, T, &steps[t], ids[t], bts[t]);
	double *dls = G ? malloc((size_t)L * CX_VOCAB * sizeof *dls) : NULL;
	for (int t = 0; t < L; t++) {
		loss += ref_head(P, &steps[t], ids[t + 1], dl);
		if (G) for (int v = 0; v < CX_VOCAB; v++) dls[(long)t * CX_VOCAB + v] = dl[v] / L;
	}
	if (G) {
		for (int t = L - 1; t >= 0; t--) ref_backward(P, G, &steps[t], dls + (long)t * CX_VOCAB, dT);
		free(dls);
	}
	free(T);
	free(dl);
	free(dT);
	return loss / L;
}

#endif
