/*
 * ref-loss: the mean next-token loss of a checkpoint over a token stream, on the host in double (tools/cref.h: the
 * 120 columns, the bus off as the lineage was trained, no n-gram memory). The measurement of what a change to the
 * weights costs, such as the 3-bit quantisation of tools/quant3.c, while the card is busy training.
 *   ref-loss CKPT IDS.u16 NTOK [SKIP]
 * Steps the stream from token SKIP (default 0) for NTOK tokens with the fast weights carried across the whole pass
 * (the state starts from zero, as a session does) and prints the loss in nats and bits per token and the top-1
 * accuracy, with a running line every 5,000 tokens. The last token has no target: NTOK - 1 predictions.
 */
#include "cref.h"
#include "../lib/cx-proto.h"

#include <stdio.h>
#include <time.h>

#define WINDOW 32

static int load_ckpt(cref *C, const char *path)
{
	FILE *f = fopen(path, "rb");
	if (!f) { perror(path); return -1; }
	float *buf = malloc(CX_STAGE_FLOATS * sizeof *buf);
	if (fread(buf, sizeof *buf, CX_STAGE_FLOATS, f) != CX_STAGE_FLOATS) { fprintf(stderr, "ref-loss: short checkpoint %s\n", path); return -1; }
	fclose(f);
	const float *p = buf;
	for (long i = 0; i < (long)C->N * CX_NPARAM; i++) C->prm[i] = p[i];
	p += (long)C->N * CX_NPARAM;
	for (int i = 0; i < CX_DU; i++) C->cg[i] = p[i];
	p += CX_DU;
	/* the file holds the embedding in blocks of 16 vocabulary rows (the card's ETB layout) */
	for (int v = 0; v < CX_VOCAB; v++) for (int c = 0; c < CX_DE; c++) C->E[(long)v * CX_DE + c] = p[(((long)v >> 4) * CX_DE + c) * 16 + (v & 15)];
	free(buf);
	return 0;
}

static double now(void)
{
	struct timespec t;
	clock_gettime(CLOCK_MONOTONIC, &t);
	return (double)t.tv_sec + t.tv_nsec * 1e-9;
}

int main(int argc, char **argv)
{
	if (argc < 4) { fprintf(stderr, "usage: ref-loss CKPT IDS.u16 NTOK [SKIP]\n"); return 2; }
	const long ntok = atol(argv[3]), skip = argc > 4 ? atol(argv[4]) : 0;
	cref C;
	cref_alloc(&C, CX_NCOL, WINDOW);
	if (load_ckpt(&C, argv[1])) return 1;
	FILE *f = fopen(argv[2], "rb");
	if (!f) { perror(argv[2]); return 1; }
	uint16_t *tok = malloc((size_t)ntok * 2);
	fseek(f, skip * 2, SEEK_SET);
	if (fread(tok, 2, (size_t)ntok, f) != (size_t)ntok) { fprintf(stderr, "ref-loss: %s holds fewer than %ld ids after %ld\n", argv[2], ntok, skip); return 1; }
	fclose(f);
	static float zero_bus[CX_DB * CX_NBUS];
	const float *ext[WINDOW];
	for (int t = 0; t < WINDOW; t++) ext[t] = zero_bus;
	C.ext = ext;
	C.bus_mode = CREF_BUS_EXT;
	double *l = malloc(CX_VOCAB * sizeof *l);
	double nll = 0;
	long hits = 0;
	const double t0 = now();
	for (long s = 0; s + 1 < ntok; s++) {
		const int t = (int)(s % WINDOW);
		cref_step(&C, t, tok[s], s >= 1 ? tok[s - 1] : CX_NG_PAD, s >= 2 ? tok[s - 2] : CX_NG_PAD);
		const double *hc = C.cs[t].hc;
		int best = 0;
		double max = -1e300;
		for (int v = 0; v < CX_VOCAB; v++) {
			const double *row = C.E + (long)v * CX_DE;
			double x = 0;
			for (int c = 0; c < CX_DE; c++) x += row[c] * hc[c];
			l[v] = x;
			if (x > max) { max = x; best = v; }
		}
		double sum = 0;
		for (int v = 0; v < CX_VOCAB; v++) sum += exp(l[v] - max);
		const int next = tok[s + 1];
		nll += max + log(sum) - l[next];
		hits += best == next;
		if ((s + 1) % 5000 == 0) {
			printf("ref-loss: %ld tokens: %.4f nats = %.4f bits/token, top-1 %.4f, %.1f tok/s\n", s + 1, nll / (s + 1), nll / (s + 1) / log(2.0), (double)hits / (s + 1), (s + 1) / (now() - t0));
			fflush(stdout);
		}
	}
	const long n = ntok - 1;
	printf("ref-loss: %s on %s[%ld..%ld): %.4f nats = %.4f bits/token over %ld predictions, top-1 %.4f, %.1f s\n", argv[1], argv[2], skip, skip + ntok, nll / n, nll / n / log(2.0), n, (double)hits / n, now() - t0);
	return 0;
}
