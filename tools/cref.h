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
 */
#ifndef CX_CREF_H
#define CX_CREF_H

#include "ref.h"

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
	/* MoE stage 1: n-gram memory (ng_on): tables CX_NG_ORDERS x CX_NG_ROWS x 128, gates, the augmented input of each step
	 * and the rows used (L x orders x 128) */
	int ng_on;
	double *ngT[CX_NG_ORDERS], ngg[CX_NG_ORDERS][CX_DE], *ein, *ngr;
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
			const double *row = C->ngT[o] + (size_t)cx_ng_hash(id, p1, p2, o + 2) * CX_DE;
			double *save = C->ngr + ((long)t * CX_NG_ORDERS + o) * CX_DE;
			for (int i = 0; i < CX_DE; i++) { save[i] = row[i]; ein[i] += C->ngg[o][i] * row[i]; }
		}
	}
	for (int c = 0; c < C->N; c++) {
		rstep *r = &C->steps[(long)c * C->L + t];
		ref_col_forward(C->prm + (long)c * CX_NPARAM, C->E, C->T + (long)c * CX_NT, r, id, Bt, ein);
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
		ref_col_backward(C->prm + (long)c * CX_NPARAM, G->prm + (long)c * CX_NPARAM, C->E, G->E, r, du, dg * r->g * (1.0 - r->g), dT + (long)c * CX_NT, de);
		for (int i = 0; i < CX_DE; i++) desum[i] += de[i];
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
