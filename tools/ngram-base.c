/* ngram-base: n-gram baselines for the Cortex's held-out loss, in bits per token.
 *
 *   ngram-base TRAIN.u32 VALID.u32 NTOK [MAXTRAIN]
 *
 * Counts unigrams, bigrams and trigrams over TRAIN (u32 token ids of the 4,096-id vocabulary, lib/cx-proto.h cx_tok,
 * 4 B a token; the first MAXTRAIN tokens if given), then scores VALID[0..NTOK) exactly as tools/ref-loss.c scores a
 * checkpoint: the NTOK - 1 predictions of ids 1..NTOK-1, each conditioned on what precedes it, averaged in bits per
 * token. The three numbers are the floor a model has to beat before it has learned anything beyond token frequencies
 * (unigram), the previous token (bigram) or the previous two (trigram).
 *
 * A .u16 stream of before 2026-10-28 is widened by tools/ids32 first. A file that is not a whole number of u32 ids,
 * or holds an id at 4,096 or above (a .u16 stream read as u32 pairs two ids into one), is refused. With every id
 * below 4,096 the trigram key (16 bits an id) and its hash are bit-identical to the .u16 build's, so a widened stream
 * gives the same numbers.
 *
 * Smoothing: add-one on the unigram; the bigram and the trigram are interpolated with the next lower order
 * (Jelinek-Mercer, one lambda per order), and the best lambda of the grid {0.3, 0.5, 0.6, 0.7, 0.8, 0.9} is reported.
 * The lambda is picked on the held-out tokens, so the bigram and trigram numbers are a slightly optimistic baseline
 * (by a few hundredths of a bit); a model that does not reach them is not within reach of a table.
 *
 * Memory: 64 MB of bigram counts, 64 MB of trigram context counts and a 2^26-slot open-addressing hash of trigram
 * counts (768 MB); the hash holds 47M distinct trigrams before it refuses, enough for about 100M training tokens.
 * Both streams are read whole, 4 B a token (126.2 MB for 31.55M tokens).
 *
 * Measured 2026-10-27 01:2x on the desktop with the u16 build, on the .u16 streams of the time (results in
 * docs/CHAT.md, "Baselines"):
 *   data/chat-scratch/train.u16 (31.55M tokens) -> data/chat-scratch/valid.u16[0..27447):
 *     unigram 9.9789, bigram 6.9650 (lambda 0.9), trigram 6.1843 (lambda 0.7) bits/token, 1.24 s
 *   the first 31.55M tokens of data/bpe/train.u16 -> data/bpe/valid.u16[0..20000):
 *     unigram 8.4306, bigram 5.1127 (0.9), trigram 4.1263 (0.9) bits/token
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../lib/cx-proto.h"

#define V 4096
#define HBITS 26
#define HSIZE (1ul << HBITS)
#define HFILL_MAX (HSIZE * 7 / 10)

static const double lambdas[] = { 0.3, 0.5, 0.6, 0.7, 0.8, 0.9 };
#define NLAMBDA ((int)(sizeof lambdas / sizeof *lambdas))

static uint64_t *hkey;
static uint32_t *hval;
static unsigned long hfill;

/* every id below 4,096: the OR of all ids has no bit at 12 or above (one vectorised pass, as tools/conv-pos.c) */
static void ids_check(const char *path, const cx_tok *t, long n)
{
	cx_tok m = 0;
	for (long i = 0; i < n; i++) m |= t[i];
	if (!(m >> 12)) return;
	long i = 0;
	while (t[i] < V) i++;
	fprintf(stderr, "ngram-base: %s: id %u at token %ld is outside the 4,096-id vocabulary (a .u16 stream? tools/ids32 widens one)\n", path, t[i], i);
	exit(1);
}

static cx_tok *load(const char *path, long *n)
{
	FILE *f = fopen(path, "rb");
	if (!f) { perror(path); exit(1); }
	fseek(f, 0, SEEK_END);
	long bytes = ftell(f);
	fseek(f, 0, SEEK_SET);
	if (bytes % CX_TOK_BYTES) { fprintf(stderr, "ngram-base: %s is %ld B, not a whole number of u32 ids\n", path, bytes); exit(1); }
	cx_tok *p = malloc((size_t)bytes);
	if (!p || fread(p, 1, (size_t)bytes, f) != (size_t)bytes) { perror(path); exit(1); }
	fclose(f);
	*n = bytes / CX_TOK_BYTES;
	ids_check(path, p, *n);
	return p;
}

/* the trigram's key: the three ids in 16-bit fields (ids below 4,096, ids_check), the key of the .u16 build */
static uint64_t tri_key(cx_tok a, cx_tok b, cx_tok c)
{
	return ((uint64_t)a << 32) | ((uint64_t)b << 16) | c;
}

static uint64_t hidx(uint64_t k)
{
	uint64_t h = k * 0x9E3779B97F4A7C15ull;
	h ^= h >> 29;
	return h & (HSIZE - 1);
}

static void hadd(uint64_t k)
{
	uint64_t i = hidx(k);
	for (;;) {
		if (hval[i] == 0) {
			if (++hfill > HFILL_MAX) { fprintf(stderr, "ngram-base: more than %lu distinct trigrams; shorten MAXTRAIN\n", HFILL_MAX); exit(1); }
			hkey[i] = k;
			hval[i] = 1;
			return;
		}
		if (hkey[i] == k) { hval[i]++; return; }
		i = (i + 1) & (HSIZE - 1);
	}
}

static uint32_t hget(uint64_t k)
{
	uint64_t i = hidx(k);
	for (;;) {
		if (hval[i] == 0) return 0;
		if (hkey[i] == k) return hval[i];
		i = (i + 1) & (HSIZE - 1);
	}
}

int main(int argc, char **argv)
{
	if (argc != 4 && argc != 5) { fprintf(stderr, "usage: ngram-base TRAIN.u32 VALID.u32 NTOK [MAXTRAIN]\n"); return 2; }
	long nt, nv;
	cx_tok *t = load(argv[1], &nt);
	cx_tok *v = load(argv[2], &nv);
	long n = atol(argv[3]);
	if (n < 3 || n > nv) { fprintf(stderr, "ngram-base: NTOK must be 3..%ld\n", nv); return 2; }
	if (argc == 5) { long m = atol(argv[4]); if (m > 0 && m < nt) nt = m; }

	static uint32_t uni[V];
	uint32_t *bi = calloc((size_t)V * V, sizeof *bi);         /* count of (a, b) */
	uint32_t *bictx = calloc(V, sizeof *bictx);                /* count of (a, *) */
	uint32_t *trictx = calloc((size_t)V * V, sizeof *trictx); /* count of (a, b, *) */
	hkey = calloc(HSIZE, sizeof *hkey);
	hval = calloc(HSIZE, sizeof *hval);
	if (!bi || !bictx || !trictx || !hkey || !hval) { fprintf(stderr, "ngram-base: out of memory\n"); return 1; }
	for (long i = 0; i < nt; i++) {
		uni[t[i]]++;
		if (i >= 1) { bi[(size_t)t[i - 1] * V + t[i]]++; bictx[t[i - 1]]++; }
		if (i >= 2) { hadd(tri_key(t[i - 2], t[i - 1], t[i])); trictx[(size_t)t[i - 2] * V + t[i - 1]]++; }
	}

	const double denom = nt + (double)V;
	double uni_bits = 0;
	for (long i = 1; i < n; i++) uni_bits += -log2((uni[v[i]] + 1.0) / denom);
	printf("unigram (add-one): %.4f bits/token over %ld predictions; %ld training tokens, %lu distinct trigrams\n",
	       uni_bits / (n - 1), n - 1, nt, hfill);

	double best_bi = 1e300, best_bi_l = 0, best_tri = 1e300, best_tri_l = 0;
	for (int li = 0; li < NLAMBDA; li++) {
		const double l = lambdas[li];
		double bb = 0, tb = 0;
		for (long i = 1; i < n; i++) {
			const double pu = (uni[v[i]] + 1.0) / denom;
			double pb = pu;
			if (bictx[v[i - 1]]) pb = l * bi[(size_t)v[i - 1] * V + v[i]] / (double)bictx[v[i - 1]] + (1 - l) * pu;
			bb += -log2(pb);
			double pt = pb;
			if (i >= 2) {
				const uint32_t cx = trictx[(size_t)v[i - 2] * V + v[i - 1]];
				if (cx) pt = l * hget(tri_key(v[i - 2], v[i - 1], v[i])) / (double)cx + (1 - l) * pb;
			}
			tb += -log2(pt);
		}
		if (bb < best_bi) { best_bi = bb; best_bi_l = l; }
		if (tb < best_tri) { best_tri = tb; best_tri_l = l; }
	}
	printf("bigram  (interpolated, lambda %.1f): %.4f bits/token\n", best_bi_l, best_bi / (n - 1));
	printf("trigram (interpolated, lambda %.1f): %.4f bits/token\n", best_tri_l, best_tri / (n - 1));
	return 0;
}
