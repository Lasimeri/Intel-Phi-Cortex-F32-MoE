/*
 * gen-check: the generation daemon's gate against the double reference. Replays a token stream through tools/cref.h
 * (120 columns from a checkpoint, the bus live or zero, the n-gram memory from its tables and gates when given) and
 * compares every step's 4096 logits with the ones the daemon produced for the same stream (cortex --ids --dump-logits).
 *
 *   gen-check CKPT IDS.u16 NSTEPS LOGITS.f32 [--bus 0|1] [--ngram TABLES GATES]
 *
 * The daemon's STEP s is the forward on ids[s] with the state from steps 0..s-1 (T from zero, the bus from zero), so
 * the reference runs the same NSTEPS forwards in order; with the memory, step s uses the rows of (ids[s-2], ids[s-1],
 * ids[s]) with CX_NG_PAD before the start, as host/cortex.c writes them.
 */
#include "cref.h"
#include "../lib/cx-proto.h"

#include <stdio.h>

static int load_ckpt(cref *C, const char *path)
{
	FILE *f = fopen(path, "rb");
	if (!f) { perror(path); return -1; }
	float *buf = malloc(CX_STAGE_FLOATS * sizeof *buf);
	if (fread(buf, sizeof *buf, CX_STAGE_FLOATS, f) != CX_STAGE_FLOATS) { fprintf(stderr, "gen-check: short checkpoint %s\n", path); return -1; }
	fclose(f);
	const float *p = buf;
	for (long i = 0; i < (long)C->N * CX_NPARAM; i++) C->prm[i] = p[i];
	p += (long)C->N * CX_NPARAM;
	for (int i = 0; i < CX_DU; i++) C->cg[i] = p[i];
	p += CX_DU;
	for (int v = 0; v < CX_VOCAB; v++) for (int c = 0; c < CX_DE; c++) C->E[(long)v * CX_DE + c] = p[(((long)v >> 4) * CX_DE + c) * 16 + (v & 15)];   /* the file holds ETB blocks */
	free(buf);
	return 0;
}

static int load_memory(cref *C, const char *tables, const char *gates)
{
	FILE *f = fopen(tables, "rb");
	if (!f) { perror(tables); return -1; }
	float *row = malloc(CX_DE * sizeof *row);
	for (int o = 0; o < CX_NG_ORDERS; o++) {
		C->ngT[o] = malloc((size_t)CX_NG_ROWS * CX_DE * sizeof *C->ngT[o]);
		if (!C->ngT[o]) { fprintf(stderr, "gen-check: %zu bytes of double tables\n", (size_t)CX_NG_ROWS * CX_DE * sizeof(double)); return -1; }
		for (size_t r = 0; r < CX_NG_ROWS; r++) {
			if (fread(row, sizeof *row, CX_DE, f) != CX_DE) { fprintf(stderr, "gen-check: short tables %s\n", tables); return -1; }
			for (int i = 0; i < CX_DE; i++) C->ngT[o][r * CX_DE + i] = row[i];
		}
	}
	fclose(f);
	free(row);
	float g[CX_NG_ORDERS * CX_DE];
	f = fopen(gates, "rb");
	if (!f || fread(g, sizeof *g, CX_NG_ORDERS * CX_DE, f) != CX_NG_ORDERS * CX_DE) { perror(gates); return -1; }
	fclose(f);
	for (int o = 0; o < CX_NG_ORDERS; o++) for (int i = 0; i < CX_DE; i++) C->ngg[o][i] = g[o * CX_DE + i];
	C->ng_on = 1;
	return 0;
}

int main(int argc, char **argv)
{
	if (argc < 5) { fprintf(stderr, "usage: gen-check CKPT IDS.u16 NSTEPS LOGITS.f32 [--bus 0|1] [--ngram TABLES GATES]\n"); return 2; }
	const int nsteps = atoi(argv[3]);
	int bus = 1;
	const char *tables = NULL, *gates = NULL;
	for (int i = 5; i < argc; i++) {
		if (!strcmp(argv[i], "--bus") && i + 1 < argc) bus = atoi(argv[++i]);
		else if (!strcmp(argv[i], "--ngram") && i + 2 < argc) { tables = argv[++i]; gates = argv[++i]; }
		else { fprintf(stderr, "gen-check: unknown option %s\n", argv[i]); return 2; }
	}
	cref C;
	cref_alloc(&C, CX_NCOL, nsteps);
	if (load_ckpt(&C, argv[1])) return 1;
	if (tables && load_memory(&C, tables, gates)) return 1;
	/* the stream and the daemon's logits */
	FILE *f = fopen(argv[2], "rb");
	if (!f) { perror(argv[2]); return 1; }
	uint16_t *tok = malloc((size_t)nsteps * 2);
	if (fread(tok, 2, (size_t)nsteps, f) != (size_t)nsteps) { fprintf(stderr, "gen-check: %s holds fewer than %d ids\n", argv[2], nsteps); return 1; }
	fclose(f);
	f = fopen(argv[4], "rb");
	if (!f) { perror(argv[4]); return 1; }
	float *card = malloc((size_t)nsteps * CX_VOCAB * sizeof *card);
	if (fread(card, sizeof *card, (size_t)nsteps * CX_VOCAB, f) != (size_t)nsteps * CX_VOCAB) { fprintf(stderr, "gen-check: %s holds fewer than %d steps of logits\n", argv[4], nsteps); return 1; }
	fclose(f);
	/* the bus: live (the columns' own bc, one step delayed) or zero */
	static float zero_bus[CX_DB * CX_NBUS];
	const float **ext = NULL;
	if (!bus) {
		ext = malloc((size_t)nsteps * sizeof *ext);
		for (int t = 0; t < nsteps; t++) ext[t] = zero_bus;
		C.ext = ext;
		C.bus_mode = CREF_BUS_EXT;
	}
	/* replay and compare */
	double *l = malloc(CX_VOCAB * sizeof *l);
	double amax = 0, rmax = 0, ref_at = 0;
	int at_step = -1, at_v = -1, agree = 0;
	for (int t = 0; t < nsteps; t++) {
		cref_step(&C, t, tok[t], t >= 1 ? tok[t - 1] : CX_NG_PAD, t >= 2 ? tok[t - 2] : CX_NG_PAD);
		const double *hc = C.cs[t].hc;
		int rbest = 0, cbest = 0;
		double step_amax = 0;
		for (int v = 0; v < CX_VOCAB; v++) {
			double s = 0;
			const double *row = C.E + (long)v * CX_DE;
			for (int c = 0; c < CX_DE; c++) s += row[c] * hc[c];
			l[v] = s;
			if (fabs(s) > rmax) rmax = fabs(s);
			if (s > l[rbest]) rbest = v;
			const float cv = card[(long)t * CX_VOCAB + v];
			if (cv > card[(long)t * CX_VOCAB + cbest]) cbest = v;
			const double d = fabs(cv - s);
			if (d > step_amax) step_amax = d;
			if (d > amax) { amax = d; at_step = t; at_v = v; ref_at = s; }
		}
		agree += rbest == cbest;
		if (t < 3 || t + 1 == nsteps || t % 32 == 31)
			printf("  step %3d  id %4u  ref argmax %4d  card argmax %4d  max abs err %.3e\n", t, tok[t], rbest, cbest, step_amax);
	}
	printf("gen-check: %d steps, bus %s, memory %s: max |ref logit| %.4e, max abs err %.3e, rel %.3e (step %d, v %d: card %.6e ref %.6e); argmax agree %d of %d\n",
	       nsteps, bus ? "live" : "zero", tables ? "on" : "off", rmax, amax, rmax > 0 ? amax / rmax : 0, at_step, at_v,
	       at_step >= 0 ? card[(long)at_step * CX_VOCAB + at_v] : 0.0, ref_at, agree, nsteps);
	return 0;
}
