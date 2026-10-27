/*
 * cref.h: host double reference of the CLUSTER training step (Stage 4): N columns (tools/ref.h per column), one shared
 * embedding E and consensus gain cg, the same token to every column per step.
 *
 *   per step t:  column c forward on token id_t with the bus Bt (slot j = column j's bc from step t-1, zero at start)
 *                hs = sum_c g_c u_c      hc = RMSNorm(hs) cg      logits = E hc      loss = CE(next token)
 *                bus for t+1: slot c = bc_c (one step delayed, stop-gradient)
 *   backward:    dhc = E^T dl;  dhs = RMSNorm backward (dcg += dhc n);  du_c = g_c dhs;  dgl_c = (u_c . dhs) g_c (1 - g_c)
 *                then each column's own backward (ref_col_backward), dT_c carried per column inside the chunk.
 *
 * Bus modes: CREF_BUS_LIVE (the columns' own bc, as cortexd runs) or CREF_BUS_EXT (an external matrix per step, for the
 * finite-difference check: with the live bus the analytic gradient drops the path through the bus by design, and FD
 * would measure that dropped path).
 *
 * The n-gram memory (cref_ng_tables: the tables as F32 like the card's) and the MoE model of stage 2 (lib/cx-moe.h)
 * are reference modes too: cref_moe_init routes every column through the keys with the home bonus, copies the
 * expert into the column's MLP region before the MLP, collects each column's MLP gradient apart and applies all
 * columns' SGD updates in column order after every column's products (the card's order); cref_moe_pool reads the
 * experts from a saved pool file instead (evaluation, no updates: tools/ref-loss.c --moe).
 */
#ifndef CX_CREF_H
#define CX_CREF_H

#include "ref.h"
#include "../lib/cx-moe.h"

enum { CREF_BUS_LIVE, CREF_BUS_EXT };

typedef struct {
	double hs[CX_DU], xsh, hc[CX_DU];
} cstep;

typedef struct {
	int N, L, bus_mode;
	double *prm;       /* N x CX_NPARAM */
	double *E, *cg;    /* shared */
	double *T;         /* N x CX_NT, the live state */
	rstep *steps;      /* N x L */
	cstep *cs;         /* L */
	float *bus;        /* (L + 1) x [16][128]: bus[t] is the input of step t (the backward reads it again) */
	const float *const *ext;   /* L external bus matrices (CREF_BUS_EXT) */
	/* MoE stage 1: n-gram memory (ng_on): tables CX_NG_ORDERS x CX_NG_ROWS x 128 in double (the gate builds) or as the
	 * trainers' F32 files (ngT32, read as they are), gates, the augmented input of each step and the rows used
	 * (L x orders x 128) */
	int ng_on;
	double *ngT[CX_NG_ORDERS], ngg[CX_NG_ORDERS][CX_DE], *ein, *ngr;
	const float *ngT32[CX_NG_ORDERS];
	/* MoE stage 2 (moe_on, lib/cx-moe.h, card/moe.h): the fixed product keys (F32, [half][dim][lane], as the card
	 * holds them), the home pair's bonus, the experts' SGD step, every column's home MLP at conversion (the pool's
	 * initial expert e is home[e mod N]), the touched experts in double (index, weights), each column's route per
	 * step (route[c * L + t]) and the per-step expert gradients collected before the updates */
	int moe_on, moe_ntouched, moe_cap;
	const float *moe_keys, *moe_pool32;   /* moe_pool32: a saved pool read as it is (evaluation; no updates) instead of the home copies */
	double moe_bias, moe_lr, *moe_home, *moe_w, *moe_grad;
	uint32_t *route, *moe_idx;
} cref;

static inline void cref_alloc(cref *C, int N, int L)
{
	C->N = N;
	C->L = L;
	C->prm = calloc((size_t)N * CX_NPARAM, sizeof *C->prm);
	C->E = calloc((size_t)CX_VOCAB * CX_DE, sizeof *C->E);
	C->cg = calloc(CX_DU, sizeof *C->cg);
	C->T = calloc((size_t)N * CX_NT, sizeof *C->T);
	C->steps = calloc((size_t)N * L, sizeof *C->steps);
	C->cs = calloc((size_t)L, sizeof *C->cs);
	C->bus = calloc((size_t)(L + 1) * CX_DB * CX_NBUS, sizeof *C->bus);
	C->bus_mode = CREF_BUS_LIVE;
	C->ng_on = 0;
	C->ein = calloc((size_t)L * CX_DE, sizeof *C->ein);
	C->ngr = calloc((size_t)L * CX_NG_ORDERS * CX_DE, sizeof *C->ngr);
	for (int o = 0; o < CX_NG_ORDERS; o++) { C->ngT[o] = NULL; C->ngT32[o] = NULL; }
	C->moe_on = 0;
	C->moe_pool32 = NULL;
	C->moe_keys = NULL;
	C->moe_bias = C->moe_lr = 0;
	C->moe_home = C->moe_w = C->moe_grad = NULL;
	C->route = C->moe_idx = NULL;
	C->moe_ntouched = C->moe_cap = 0;
}

/* the MoE model read with a saved pool (the daemon's pool-N.f32, CX_EX_N x CX_EX_FLOATS F32) for evaluation: the
 * experts come from the file, the home MLPs of the checkpoint are not used, nothing is updated */
static inline void cref_moe_pool(cref *C, const float *pool, const float *keys, double bias)
{
	C->moe_on = 1;
	C->moe_pool32 = pool;
	C->moe_keys = keys;
	C->moe_bias = bias;
	C->moe_lr = 0;
	C->route = calloc((size_t)C->N * C->L, sizeof *C->route);
}

/* the MoE conversion (after the columns' weights are set): the keys, the knobs, the home MLPs copied, no expert touched */
static inline void cref_moe_init(cref *C, const float *keys, double bias, double lr)
{
	C->moe_on = 1;
	C->moe_keys = keys;
	C->moe_bias = bias;
	C->moe_lr = lr;
	C->moe_home = malloc((size_t)C->N * CX_EX_FLOATS * sizeof *C->moe_home);
	for (int c = 0; c < C->N; c++) memcpy(C->moe_home + (size_t)c * CX_EX_FLOATS, C->prm + (long)c * CX_NPARAM + CX_EX_SLOT, CX_EX_FLOATS * sizeof *C->moe_home);
	C->route = calloc((size_t)C->N * C->L, sizeof *C->route);
	C->moe_cap = 1024;
	C->moe_ntouched = 0;
	C->moe_idx = malloc((size_t)C->moe_cap * sizeof *C->moe_idx);
	C->moe_w = malloc((size_t)C->moe_cap * CX_EX_FLOATS * sizeof *C->moe_w);
	C->moe_grad = malloc((size_t)C->N * CX_EX_FLOATS * sizeof *C->moe_grad);
}

/* expert e in double: its touched copy, made from the home MLP of column e mod N on the first touch */
static inline double *cref_moe_expert(cref *C, uint32_t e)
{
	for (int i = 0; i < C->moe_ntouched; i++) if (C->moe_idx[i] == e) return C->moe_w + (size_t)i * CX_EX_FLOATS;
	if (C->moe_ntouched == C->moe_cap) {
		C->moe_cap *= 2;
		C->moe_idx = realloc(C->moe_idx, (size_t)C->moe_cap * sizeof *C->moe_idx);
		C->moe_w = realloc(C->moe_w, (size_t)C->moe_cap * CX_EX_FLOATS * sizeof *C->moe_w);
	}
	const int i = C->moe_ntouched++;
	C->moe_idx[i] = e;
	memcpy(C->moe_w + (size_t)i * CX_EX_FLOATS, C->moe_home + (size_t)(e % (uint32_t)C->N) * CX_EX_FLOATS, CX_EX_FLOATS * sizeof *C->moe_w);
	return C->moe_w + (size_t)i * CX_EX_FLOATS;
}

/* the argmax key of one half in double: x . K[:, lane] over the 248 keys plus the bonus on the home lane; ties to the
 * lowest lane, as the card's lane scan */
static inline int cref_moe_half(const float *K, const double *x, uint32_t home, double bias)
{
	double best = -INFINITY;
	int arg = 0;
	for (int lane = 0; lane < CX_EX_KEYS; lane++) {
		double s = 0;
		for (int d = 0; d < CX_EX_HALF; d++) s += x[d] * K[(long)d * CX_EX_KLANES + lane];
		if (lane == (int)home) s += bias;
		if (s > best) { best = s; arg = lane; }
	}
	return arg;
}

static inline uint32_t cref_moe_route(const cref *C, int c, const double *x)
{
	const int a = cref_moe_half(C->moe_keys, x, CX_EX_HOME_A(c), C->moe_bias);
	const int b = cref_moe_half(C->moe_keys + CX_EX_HALF * CX_EX_KLANES, x + CX_EX_HALF, CX_EX_HOME_B(c), C->moe_bias);
	return cx_ex_index((uint32_t)a, (uint32_t)b);
}

/* the n-gram memory with a trained run's F32 tables (orders 2 and 3, CX_NG_ROWS x 128 each, host/ngtables.h) and its
 * gates (2 x 128 F32): the rows are read as F32 values */
static inline void cref_ng_tables(cref *C, const float *const T[CX_NG_ORDERS], const float *gates)
{
	for (int o = 0; o < CX_NG_ORDERS; o++) {
		C->ngT32[o] = T[o];
		for (int i = 0; i < CX_DE; i++) C->ngg[o][i] = gates[o * CX_DE + i];
	}
	C->ng_on = 1;
}

/* the n-gram memory with the gate-build tables (lib/cx-col.h) and gates of 1 */
static inline void cref_ng_init(cref *C)
{
	float *tmp = malloc((size_t)CX_NG_ROWS * CX_DE * sizeof *tmp);
	for (int o = 0; o < CX_NG_ORDERS; o++) {
		C->ngT[o] = malloc((size_t)CX_NG_ROWS * CX_DE * sizeof *C->ngT[o]);
		cx_ng_init(tmp, o + 2);
		for (size_t i = 0; i < (size_t)CX_NG_ROWS * CX_DE; i++) C->ngT[o][i] = tmp[i];
		for (int i = 0; i < CX_DE; i++) C->ngg[o][i] = 1.0;
	}
	free(tmp);
	C->ng_on = 1;
}

static inline void cref_init(cref *C)
{
	rparams P = { C->prm, C->E, C->cg };
	rparams_init_shared(&P);
	for (int c = 0; c < C->N; c++) rparams_init_col(C->prm + (long)c * CX_NPARAM, c);
}

typedef struct { double *prm, *E, *cg, *ngrad; double ngg[CX_NG_ORDERS][CX_DE]; } cgrads;   /* prm: N x CX_NPARAM; ngrad: L x orders x 128 (the row gradients, step order) */

static inline void cgrads_alloc(cgrads *G, int N)
{
	G->prm = calloc((size_t)N * CX_NPARAM, sizeof *G->prm);
	G->E = calloc((size_t)CX_VOCAB * CX_DE, sizeof *G->E);
	G->cg = calloc(CX_DU, sizeof *G->cg);
	G->ngrad = calloc((size_t)32 * CX_NG_ORDERS * CX_DE, sizeof *G->ngrad);
}

static inline void cgrads_zero(cgrads *G, int N)
{
	memset(G->prm, 0, (size_t)N * CX_NPARAM * sizeof *G->prm);
	memset(G->E, 0, (size_t)CX_VOCAB * CX_DE * sizeof *G->E);
	memset(G->cg, 0, CX_DU * sizeof *G->cg);
	memset(G->ngrad, 0, (size_t)32 * CX_NG_ORDERS * CX_DE * sizeof *G->ngrad);
	memset(G->ngg, 0, sizeof G->ngg);
}

/* one cluster step t (forward of every column, vote, consensus, next bus) */
static inline void cref_step(cref *C, int t, uint32_t id, uint32_t p1, uint32_t p2)
{
	float *bt = C->bus + (long)t * CX_DB * CX_NBUS, *bn = bt + CX_DB * CX_NBUS;
	const float *Bt = C->bus_mode == CREF_BUS_EXT ? C->ext[t] : bt;
	cstep *s = &C->cs[t];
	for (int i = 0; i < CX_DU; i++) s->hs[i] = 0;
	double *ein = NULL;
	if (C->ng_on) {
		ein = C->ein + (long)t * CX_DE;
		for (int i = 0; i < CX_DE; i++) ein[i] = C->E[(long)id * CX_DE + i];
		for (int o = 0; o < CX_NG_ORDERS; o++) {
			const size_t at = (size_t)cx_ng_hash(id, p1, p2, o + 2) * CX_DE;
			double *save = C->ngr + ((long)t * CX_NG_ORDERS + o) * CX_DE;
			if (C->ngT32[o]) { const float *row = C->ngT32[o] + at; for (int i = 0; i < CX_DE; i++) save[i] = row[i]; }
			else { const double *row = C->ngT[o] + at; for (int i = 0; i < CX_DE; i++) save[i] = row[i]; }
			for (int i = 0; i < CX_DE; i++) ein[i] += C->ngg[o][i] * save[i];
		}
	}
	for (int c = 0; c < C->N; c++) {
		rstep *r = &C->steps[(long)c * C->L + t];
		double *prm = C->prm + (long)c * CX_NPARAM;
		ref_col_pre(prm, C->E, C->T + (long)c * CX_NT, r, id, Bt, ein);
		if (C->moe_on) {   /* the route from x, the expert into the column's MLP region */
			const uint32_t e = cref_moe_route(C, c, r->x);
			C->route[(long)c * C->L + t] = e;
			if (C->moe_pool32) { const float *ex = C->moe_pool32 + (size_t)e * CX_EX_FLOATS; for (int i = 0; i < CX_EX_FLOATS; i++) prm[CX_EX_SLOT + i] = ex[i]; }
			else memcpy(prm + CX_EX_SLOT, cref_moe_expert(C, e), CX_EX_FLOATS * sizeof *prm);
		}
		ref_col_mlp(prm, r);
		for (int i = 0; i < CX_DU; i++) s->hs[i] += r->g * r->u[i];
		for (int d = 0; d < CX_DB; d++) bn[d * CX_NBUS + c] = (float)r->bc[d];
	}
	double ss = 0;
	for (int i = 0; i < CX_DU; i++) ss += s->hs[i] * s->hs[i];
	s->xsh = 1.0 / sqrt(ss / CX_DU + (double)CX_EPS);
	for (int i = 0; i < CX_DU; i++) s->hc[i] = s->hs[i] * s->xsh * C->cg[i];
}

/* backward of step t given dl (scaled); dT: N x CX_NT carried */
static inline void cref_backward(cref *C, cgrads *G, int t, const double *dl, double *dT)
{
	cstep *s = &C->cs[t];
	double dhc[CX_DU] = { 0 }, dhs[CX_DU];
	ref_head_backward(C->E, G->E, s->hc, dl, dhc);
	ref_norm_backward(dhc, s->hs, s->xsh, C->cg, G->cg, dhs, CX_DU);
	double desum[CX_DE] = { 0 }, de[CX_DE];
	for (int c = 0; c < C->N; c++) {
		rstep *r = &C->steps[(long)c * C->L + t];
		double du[CX_DU], dg = 0;
		for (int i = 0; i < CX_DU; i++) { du[i] = r->g * dhs[i]; dg += r->u[i] * dhs[i]; }
		double *prm = C->prm + (long)c * CX_NPARAM, *gp = G->prm + (long)c * CX_NPARAM;
		if (C->moe_on) {   /* the step's expert for the products; its gradient collected apart from the head's */
			memcpy(prm + CX_EX_SLOT, cref_moe_expert(C, C->route[(long)c * C->L + t]), CX_EX_FLOATS * sizeof *prm);
			memset(gp + CX_EX_SLOT, 0, CX_EX_FLOATS * sizeof *gp);
		}
		ref_col_backward(prm, gp, C->E, G->E, r, du, dg * r->g * (1.0 - r->g), dT + (long)c * CX_NT, de);
		if (C->moe_on) {
			memcpy(C->moe_grad + (size_t)c * CX_EX_FLOATS, gp + CX_EX_SLOT, CX_EX_FLOATS * sizeof *gp);
			memset(gp + CX_EX_SLOT, 0, CX_EX_FLOATS * sizeof *gp);
		}
		for (int i = 0; i < CX_DE; i++) desum[i] += de[i];
	}
	if (C->moe_on) {   /* the experts' SGD updates after every column's products, in column order (the card's owner order) */
		for (int c = 0; c < C->N; c++) {
			double *w = cref_moe_expert(C, C->route[(long)c * C->L + t]);
			const double *g = C->moe_grad + (size_t)c * CX_EX_FLOATS;
			for (int i = 0; i < CX_EX_FLOATS; i++) w[i] -= C->moe_lr * g[i];
		}
	}
	if (C->ng_on) for (int o = 0; o < CX_NG_ORDERS; o++) {
		const double *r = C->ngr + ((long)t * CX_NG_ORDERS + o) * CX_DE;
		double *gr = G->ngrad + ((long)t * CX_NG_ORDERS + o) * CX_DE;
		for (int i = 0; i < CX_DE; i++) { gr[i] = C->ngg[o][i] * desum[i]; G->ngg[o][i] += desum[i] * r[i]; }
	}
}

/* a chunk from the given state (T and the step-0 bus copied in, not modified): mean loss; gradients if G. After it, bus[L] is
 * the next chunk's step-0 bus. */
static inline double cref_chunk(cref *C, cgrads *G, const double *T0, const float *bus0, const uint32_t *ids, int L, uint32_t p1, uint32_t p2)
{
	memcpy(C->T, T0, (size_t)C->N * CX_NT * sizeof *C->T);
	memcpy(C->bus, bus0, CX_DB * CX_NBUS * sizeof *C->bus);
	double *dl = malloc(CX_VOCAB * sizeof *dl), loss = 0;
	double *dls = G ? malloc((size_t)L * CX_VOCAB * sizeof *dls) : NULL;
	for (int t = 0; t < L; t++) cref_step(C, t, ids[t], t >= 1 ? ids[t - 1] : p1, t >= 2 ? ids[t - 2] : (t == 1 ? p1 : p2));
	for (int t = 0; t < L; t++) {
		loss += ref_head_hc(C->E, C->cs[t].hc, ids[t + 1], dl);
		if (G) for (int v = 0; v < CX_VOCAB; v++) dls[(long)t * CX_VOCAB + v] = dl[v] / L;
	}
	if (G) {
		double *dT = calloc((size_t)C->N * CX_NT, sizeof *dT);
		for (int t = L - 1; t >= 0; t--) cref_backward(C, G, t, dls + (long)t * CX_VOCAB, dT);
		free(dT);
		free(dls);
	}
	free(dl);
	return loss / L;
}

#endif
