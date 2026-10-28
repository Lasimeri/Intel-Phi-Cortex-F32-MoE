/*
 * unigram: metric baseline only (never a generator). usage: unigram TRAIN.u32 NTRAIN VALID.u32 NVALID
 * Token counts on the first NTRAIN train tokens (add-one smoothing over 4096); prints the cross-entropy of the first
 * NVALID valid tokens under that distribution in bits/token, and the train unigram entropy.
 * The streams hold u32 ids (lib/cx-proto.h cx_tok, 4 B a token); a .u16 stream of before 2026-10-28 is widened by
 * tools/ids32 first. An id at 4,096 or above (a .u16 stream read as u32 pairs two ids into one) is refused.
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "../lib/cx-proto.h"

/* every id below 4,096: the OR of all ids has no bit at 12 or above (one vectorised pass, as tools/conv-pos.c) */
static void ids_check(const char *path, const cx_tok *t, long n)
{
	cx_tok m = 0;
	for (long i = 0; i < n; i++) m |= t[i];
	if (!(m >> 12)) return;
	long i = 0;
	while (t[i] < 4096) i++;
	fprintf(stderr, "unigram: %s: id %u at token %ld is outside the 4,096-id vocabulary (a .u16 stream? tools/ids32 widens one)\n", path, t[i], i);
	exit(1);
}

int main(int argc, char **argv)
{
	if (argc < 5) { fprintf(stderr, "usage: unigram TRAIN.u32 NTRAIN VALID.u32 NVALID\n"); return 2; }
	const long nt = atol(argv[2]), nv = atol(argv[4]);
	double cnt[4096] = { 0 };
	FILE *f = fopen(argv[1], "rb");
	if (!f) { perror(argv[1]); return 1; }
	cx_tok *buf = malloc((size_t)nt * sizeof *buf);
	const long got = (long)fread(buf, sizeof *buf, (size_t)nt, f);
	fclose(f);
	ids_check(argv[1], buf, got);
	for (long i = 0; i < got; i++) cnt[buf[i]]++;
	double H = 0;
	for (int v = 0; v < 4096; v++) { cnt[v] += 1; }
	for (int v = 0; v < 4096; v++) { const double p = cnt[v] / (got + 4096); H -= p * log2(p); }
	f = fopen(argv[3], "rb");
	if (!f) { perror(argv[3]); return 1; }
	cx_tok *vb = malloc((size_t)nv * sizeof *vb);
	const long gv = (long)fread(vb, sizeof *vb, (size_t)nv, f);
	fclose(f);
	ids_check(argv[3], vb, gv);
	double ce = 0;
	for (long i = 0; i < gv; i++) ce -= log2(cnt[vb[i]] / (got + 4096));
	printf("unigram: %ld train tokens, entropy %.4f bits/token; valid %ld tokens cross-entropy %.4f bits/token (%.4f bits/byte at 3.961 B/token)\n",
	       got, H, gv, ce / gv, ce / gv / 3.961);
	return 0;
}
