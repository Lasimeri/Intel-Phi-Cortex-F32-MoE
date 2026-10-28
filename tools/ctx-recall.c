/*
 * ctx-recall: the recall of the stage 3 context index measured on the model's own keys (2026-10-27 16:0x, before the
 * memory is built into the model; PLAN-MOE.md "15:52"). Runs the double reference (tools/cref.h: the dense column,
 * the bus on, the n-gram memory and the pool off) over NTOK tokens of a stream from a checkpoint and keeps, for
 * every tenth column, each step's query q (y[0..63]) and unit-norm key k (64). Then, at every EVERY-th step from
 * FROM on, it takes the exact top-4 keys by q . k over the previous min(step, 16,384) steps (the ring of
 * host/ctxd.c) and compares them with what an index would return beside the exact last 256 steps:
 *   SimHash: the sign bits of the key against seeded random hyperplanes (8, 10, 12 bits; lists of 64 with the
 *     oldest overwritten), the query's bucket and the buckets within 0, 1 or 2 bit flips probed (rejected on
 *     2026-10-27 16:0x: recall@4 0.07 to 0.27 on the model's keys, which sit in a narrow cone);
 *   IVF, batch: 256 centroids a head by k-means (5 Lloyd iterations) over the ring's keys, refreshed and the lists
 *     rebuilt every 2,048 steps, the 8 nearest lists probed (recall@4 0.927 at 661 candidates);
 *   IVF, online: the centroids start from the first keys and each arriving key moves its nearest centroid toward
 *     itself by ETA (renormalised), the key filed under that centroid for good; no refresh, no rebuild (the form a
 *     service can afford: 256 dot products an insert); probed 8 and 16 lists.
 * Per configuration: recall@4 (the share of the exact top-4 returned), the top-1 hit (the exact best among the
 * four returned), the score ratio (the returned four's scores over the exact four's), and the candidates scored a
 * query. Also how much of the exact top-4 lies within the last 256, 1,024 and 4,096 steps: how far back the
 * model's own keys reach.
 *   ctx-recall CKPT STREAM.u32 NTOK [--skip TOKENS] [--heads K] [--from S] [--every E] [--save QK.f32 | --load QK.f32]
 * Defaults: --skip 0, --heads 12 (columns 0, 10, 20, ...), --from 4096, --every 8. --save writes the q, k of the
 * reference run (heads x NTOK x 128 F32) and --load reads them instead of running the reference (70 s for 20,000
 * tokens). STREAM.u32 holds u32 ids (lib/cx-proto.h cx_tok, 4 B a token; --skip counts tokens, 4 B each); a .u16
 * stream of before 2026-10-28 is widened by tools/ids32 first, and an id at 4,096 or above among the NTOK read (a
 * .u16 stream read as u32 pairs two ids into one) is refused.
 */
#include "cref.h"
#include "../lib/cx-proto.h"

#include <immintrin.h>
#include <math.h>
#include <stdio.h>
#include <time.h>

#define L 32
#define W CX_CTX_WINDOW
#define RECENT CX_CTX_RECENT
#define LISTLEN 64
#define NBITS 3
#define NPROBE 3
#define IVF_C 256
#define IVF_REFRESH 2048
#define IVF_CAP 1024
#define NIVF 3
#define TOP CX_CTX_TOP

static const int bits_of[NBITS] = { 8, 10, 12 };
static const struct { int online, probe; float eta; const char *name; } ivf_cfg[NIVF] = {
	{ 0, 8, 0, "exact 256 + IVF batch 256 x 8" },
	{ 1, 8, 0.05f, "exact 256 + IVF online 256 x 8" },
	{ 1, 16, 0.05f, "exact 256 + IVF online 256 x 16" },
};

static double now(void)
{
	struct timespec t;
	clock_gettime(CLOCK_MONOTONIC, &t);
	return (double)t.tv_sec + t.tv_nsec * 1e-9;
}

static int load_ckpt(cref *C, const char *path)
{
	FILE *f = fopen(path, "rb");
	if (!f) { perror(path); return -1; }
	float *buf = malloc(CX_STAGE_FLOATS * sizeof *buf);
	if (fread(buf, sizeof *buf, CX_STAGE_FLOATS, f) != CX_STAGE_FLOATS) { fprintf(stderr, "ctx-recall: short checkpoint %s\n", path); return -1; }
	fclose(f);
	const float *p = buf;
	for (long i = 0; i < (long)C->N * CX_NPARAM; i++) C->prm[i] = p[i];
	p += (long)C->N * CX_NPARAM;
	for (int i = 0; i < CX_DU; i++) C->cg[i] = p[i];
	p += CX_DU;
	for (int v = 0; v < CX_VOCAB; v++) for (int c = 0; c < CX_DE; c++) C->E[(long)v * CX_DE + c] = p[(((long)v >> 4) * CX_DE + c) * 16 + (v & 15)];
	free(buf);
	return 0;
}

static inline float dot64(const float *q, const float *k)
{
	__m256 a0 = _mm256_mul_ps(_mm256_loadu_ps(q), _mm256_loadu_ps(k));
	__m256 a1 = _mm256_mul_ps(_mm256_loadu_ps(q + 8), _mm256_loadu_ps(k + 8));
	a0 = _mm256_fmadd_ps(_mm256_loadu_ps(q + 16), _mm256_loadu_ps(k + 16), a0);
	a1 = _mm256_fmadd_ps(_mm256_loadu_ps(q + 24), _mm256_loadu_ps(k + 24), a1);
	a0 = _mm256_fmadd_ps(_mm256_loadu_ps(q + 32), _mm256_loadu_ps(k + 32), a0);
	a1 = _mm256_fmadd_ps(_mm256_loadu_ps(q + 40), _mm256_loadu_ps(k + 40), a1);
	a0 = _mm256_fmadd_ps(_mm256_loadu_ps(q + 48), _mm256_loadu_ps(k + 48), a0);
	a1 = _mm256_fmadd_ps(_mm256_loadu_ps(q + 56), _mm256_loadu_ps(k + 56), a1);
	a0 = _mm256_add_ps(a0, a1);
	__m128 h = _mm_add_ps(_mm256_castps256_ps128(a0), _mm256_extractf128_ps(a0, 1));
	h = _mm_add_ps(h, _mm_movehl_ps(h, h));
	h = _mm_add_ss(h, _mm_shuffle_ps(h, h, 1));
	return _mm_cvtss_f32(h);
}

typedef struct { float s[TOP]; long step[TOP]; } top4;

static inline void top_init(top4 *t) { for (int i = 0; i < TOP; i++) { t->s[i] = -3.0e38f; t->step[i] = -1; } }

static inline void top_insert(top4 *t, float s, long step)
{
	int i = TOP - 1;
	if (s <= t->s[i]) return;
	for (; i > 0 && s > t->s[i - 1]; i--) { t->s[i] = t->s[i - 1]; t->step[i] = t->step[i - 1]; }
	t->s[i] = s;
	t->step[i] = step;
}

static inline int top_has(const top4 *t, long step) { for (int i = 0; i < TOP; i++) if (t->step[i] == step) return 1; return 0; }
static inline float top_sum(const top4 *t) { float s = 0; for (int i = 0; i < TOP; i++) if (t->step[i] >= 0) s += t->s[i]; return s; }

/* one head's SimHash index at one bit count */
typedef struct {
	int bits;
	float *planes;      /* [bits][64] */
	uint32_t *lists;    /* [1 << bits][LISTLEN] steps */
	uint16_t *cursor, *fill;
} simhash;

static void simhash_init(simhash *S, int column, int bits)
{
	S->bits = bits;
	S->planes = malloc((size_t)bits * CX_DS * sizeof *S->planes);
	S->lists = malloc(((size_t)1 << bits) * LISTLEN * sizeof *S->lists);
	S->cursor = calloc((size_t)1 << bits, sizeof *S->cursor);
	S->fill = calloc((size_t)1 << bits, sizeof *S->fill);
	for (int b = 0; b < bits; b++) {
		float *P = S->planes + (size_t)b * CX_DS;
		uint64_t s = 0xC7E3A5B100000000ull ^ ((uint64_t)column << 8) ^ (uint64_t)b;
		cx_fill(P, CX_DS, &s, 1.0f);
		double nn = 0;
		for (int d = 0; d < CX_DS; d++) nn += (double)P[d] * P[d];
		const float inv = (float)(1.0 / sqrt(nn));
		for (int d = 0; d < CX_DS; d++) P[d] *= inv;
	}
}

static inline uint32_t simhash_pattern(const simhash *S, const float *x)
{
	uint32_t p = 0;
	for (int b = 0; b < S->bits; b++) if (dot64(x, S->planes + (size_t)b * CX_DS) >= 0) p |= 1u << b;
	return p;
}

static inline void simhash_insert(simhash *S, const float *k, long step)
{
	const uint32_t p = simhash_pattern(S, k);
	S->lists[(size_t)p * LISTLEN + S->cursor[p]] = (uint32_t)step;
	S->cursor[p] = (uint16_t)((S->cursor[p] + 1) % LISTLEN);
	if (S->fill[p] < LISTLEN) S->fill[p]++;
}

static inline int simhash_bucket(const simhash *S, uint32_t p, const float *q, const float *K, long t, top4 *top)
{
	const uint32_t *Lp = S->lists + (size_t)p * LISTLEN;
	int n = 0;
	for (int i = 0; i < S->fill[p]; i++) {
		const long s = Lp[i];
		if (s >= t - RECENT || s < t - W) continue;
		top_insert(top, dot64(q, K + (size_t)s * CX_DS), s);
		n++;
	}
	return n;
}

/* one head's IVF index: centroids over the ring's keys, lists of steps; batch (k-means refreshed) or online */
typedef struct {
	int online, probe, built, seeded;
	float eta;
	float *cent;        /* [IVF_C][64] */
	uint32_t *lists;    /* [IVF_C][IVF_CAP] */
	uint16_t *cursor, *fill;
} ivf;

static void ivf_init(ivf *V, int online, int probe, float eta)
{
	V->online = online;
	V->probe = probe;
	V->eta = eta;
	V->cent = calloc((size_t)IVF_C * CX_DS, sizeof *V->cent);
	V->lists = malloc((size_t)IVF_C * IVF_CAP * sizeof *V->lists);
	V->cursor = calloc(IVF_C, sizeof *V->cursor);
	V->fill = calloc(IVF_C, sizeof *V->fill);
	V->built = 0;
	V->seeded = 0;
}

static inline int ivf_nearest(const ivf *V, const float *k)
{
	int best = 0;
	float bs = -3.0e38f;
	for (int c = 0; c < IVF_C; c++) { const float s = dot64(k, V->cent + (size_t)c * CX_DS); if (s > bs) { bs = s; best = c; } }
	return best;
}

static inline void ivf_file(ivf *V, int c, long step)
{
	V->lists[(size_t)c * IVF_CAP + V->cursor[c]] = (uint32_t)step;
	V->cursor[c] = (uint16_t)((V->cursor[c] + 1) % IVF_CAP);
	if (V->fill[c] < IVF_CAP) V->fill[c]++;
}

/* an arriving key: batch form files it under the nearest centroid once built; the online form seeds the centroids
 * with the first IVF_C keys, then moves the nearest centroid toward the key and files the key under it */
static inline void ivf_insert(ivf *V, const float *k, long step)
{
	if (V->online) {
		if (V->seeded < IVF_C) { memcpy(V->cent + (size_t)V->seeded * CX_DS, k, CX_DS * sizeof *k); ivf_file(V, V->seeded, step); V->seeded++; V->built = V->seeded == IVF_C; return; }
		const int c = ivf_nearest(V, k);
		float *ce = V->cent + (size_t)c * CX_DS;
		double nn = 0;
		for (int d = 0; d < CX_DS; d++) { ce[d] += V->eta * (k[d] - ce[d]); nn += (double)ce[d] * ce[d]; }
		const float inv = nn > 0 ? (float)(1.0 / sqrt(nn)) : 0;
		for (int d = 0; d < CX_DS; d++) ce[d] *= inv;
		ivf_file(V, c, step);
		return;
	}
	if (!V->built) return;
	ivf_file(V, ivf_nearest(V, k), step);
}

/* batch form: k-means (5 Lloyd iterations, cosine: the keys are unit-norm) over the ring's keys at step t, the lists rebuilt */
static void ivf_refresh(ivf *V, const float *K, long t, uint64_t *seed)
{
	const long lo = t - W > 0 ? t - W : 0, n = t - lo;
	if (V->online || n < IVF_C * 4) return;
	for (int c = 0; c < IVF_C; c++) {
		const long s = lo + (long)(cx_rng(seed) % (uint64_t)n);
		memcpy(V->cent + (size_t)c * CX_DS, K + (size_t)s * CX_DS, CX_DS * sizeof *K);
	}
	double *sum = malloc((size_t)IVF_C * CX_DS * sizeof *sum);
	int *cnt = malloc(IVF_C * sizeof *cnt);
	for (int it = 0; it < 5; it++) {
		memset(sum, 0, (size_t)IVF_C * CX_DS * sizeof *sum);
		memset(cnt, 0, IVF_C * sizeof *cnt);
		for (long i = 0; i < n; i++) {
			const float *k = K + (size_t)(lo + i) * CX_DS;
			const int c = ivf_nearest(V, k);
			cnt[c]++;
			for (int d = 0; d < CX_DS; d++) sum[(size_t)c * CX_DS + d] += k[d];
		}
		for (int c = 0; c < IVF_C; c++) {
			if (!cnt[c]) continue;
			double nn = 0;
			for (int d = 0; d < CX_DS; d++) nn += sum[(size_t)c * CX_DS + d] * sum[(size_t)c * CX_DS + d];
			const double inv = nn > 0 ? 1.0 / sqrt(nn) : 0;
			for (int d = 0; d < CX_DS; d++) V->cent[(size_t)c * CX_DS + d] = (float)(sum[(size_t)c * CX_DS + d] * inv);
		}
	}
	memset(V->cursor, 0, IVF_C * sizeof *V->cursor);
	memset(V->fill, 0, IVF_C * sizeof *V->fill);
	V->built = 1;
	for (long i = 0; i < n; i++) ivf_file(V, ivf_nearest(V, K + (size_t)(lo + i) * CX_DS), lo + i);
	free(sum);
	free(cnt);
}

/* the query against the nearest lists; the candidates scored */
static inline int ivf_query(const ivf *V, const float *q, const float *K, long t, long lo, top4 *got)
{
	float cs[IVF_C];
	int cand = 0;
	for (int c = 0; c < IVF_C; c++) cs[c] = dot64(q, V->cent + (size_t)c * CX_DS);
	for (int pr = 0; pr < V->probe; pr++) {
		int best = -1;
		float bs = -3.0e38f;
		for (int c = 0; c < IVF_C; c++) if (cs[c] > bs) { bs = cs[c]; best = c; }
		if (best < 0) break;
		cs[best] = -3.0e38f;
		const uint32_t *Lp = V->lists + (size_t)best * IVF_CAP;
		for (int i = 0; i < V->fill[best]; i++) {
			const long s = Lp[i];
			if (s >= t - RECENT || s < lo) continue;
			top_insert(got, dot64(q, K + (size_t)s * CX_DS), s);
			cand++;
		}
	}
	return cand;
}

typedef struct { double recall, top1, ratio, cand; long n; } stat;

static inline void stat_add(stat *S, const top4 *exact, const top4 *got, int cand)
{
	int hit = 0;
	for (int i = 0; i < TOP; i++) if (exact->step[i] >= 0 && top_has(got, exact->step[i])) hit++;
	S->recall += hit / (double)TOP;
	S->top1 += exact->step[0] >= 0 && top_has(got, exact->step[0]);
	const float es = top_sum(exact);
	S->ratio += es > 0 ? top_sum(got) / es : 1.0;
	S->cand += cand;
	S->n++;
}

static void print_stat(const char *name, const stat *S)
{
	if (S->n) printf("%-34s %8.3f %8.3f %8.3f %10.0f\n", name, S->recall / S->n, S->top1 / S->n, S->ratio / S->n, S->cand / S->n);
}

int main(int argc, char **argv)
{
	if (argc < 4) { fprintf(stderr, "usage: ctx-recall CKPT STREAM.u32 NTOK [--skip TOKENS] [--heads K] [--from S] [--every E] [--save QK.f32 | --load QK.f32]\n"); return 2; }
	const long ntok = atol(argv[3]);
	long skip = 0, from = 4096;
	int nheads = 12, every = 8;
	const char *save = NULL, *load = NULL;
	for (int i = 4; i < argc; i++) {
		if (!strcmp(argv[i], "--skip") && i + 1 < argc) skip = atol(argv[++i]);
		else if (!strcmp(argv[i], "--heads") && i + 1 < argc) nheads = atoi(argv[++i]);
		else if (!strcmp(argv[i], "--from") && i + 1 < argc) from = atol(argv[++i]);
		else if (!strcmp(argv[i], "--every") && i + 1 < argc) every = atoi(argv[++i]);
		else if (!strcmp(argv[i], "--save") && i + 1 < argc) save = argv[++i];
		else if (!strcmp(argv[i], "--load") && i + 1 < argc) load = argv[++i];
		else { fprintf(stderr, "ctx-recall: unknown option %s\n", argv[i]); return 2; }
	}
	if (nheads < 1 || nheads > CX_NCOL || every < 1 || from < 1 || ntok <= from) { fprintf(stderr, "ctx-recall: bad arguments\n"); return 2; }
	const int stride = CX_NCOL / nheads;
	float *Q = malloc((size_t)nheads * ntok * CX_DS * sizeof *Q), *K = malloc((size_t)nheads * ntok * CX_DS * sizeof *K);
	if (!Q || !K) { fprintf(stderr, "ctx-recall: out of memory\n"); return 1; }
	const double t0 = now();
	if (load) {
		FILE *f = fopen(load, "rb");
		if (!f || fread(Q, sizeof *Q, (size_t)nheads * ntok * CX_DS, f) != (size_t)nheads * ntok * CX_DS || fread(K, sizeof *K, (size_t)nheads * ntok * CX_DS, f) != (size_t)nheads * ntok * CX_DS) { fprintf(stderr, "ctx-recall: %s does not hold %d heads x %ld steps\n", load, nheads, ntok); return 1; }
		fclose(f);
		printf("ctx-recall: %ld steps of q, k for %d heads loaded from %s\n", ntok, nheads, load);
	} else {
		cref C;
		cref_alloc(&C, CX_NCOL, L);
		if (load_ckpt(&C, argv[1])) return 1;
		FILE *f = fopen(argv[2], "rb");
		if (!f) { perror(argv[2]); return 1; }
		cx_tok *tok = malloc((size_t)ntok * sizeof *tok);
		fseek(f, skip * (long)CX_TOK_BYTES, SEEK_SET);
		if (fread(tok, sizeof *tok, (size_t)ntok, f) != (size_t)ntok) { fprintf(stderr, "ctx-recall: %s holds fewer than %ld ids after %ld\n", argv[2], ntok, skip); return 1; }
		fclose(f);
		/* every id below 4,096: the OR of all ids has no bit at 12 or above (one vectorised pass, as tools/conv-pos.c) */
		cx_tok m = 0;
		for (long s = 0; s < ntok; s++) m |= tok[s];
		if (m >> 12) {
			long s = 0;
			while (tok[s] < CX_VOCAB) s++;
			fprintf(stderr, "ctx-recall: %s: id %u at token %ld is outside the 4,096-id vocabulary (a .u16 stream? tools/ids32 widens one)\n", argv[2], tok[s], skip + s);
			return 1;
		}
		/* the reference over the stream, the bus carried across the window of L steps */
		for (long s = 0; s < ntok; s++) {
			const int t = (int)(s % L);
			if (t == 0 && s > 0) memcpy(C.bus, C.bus + (long)L * CX_DB * CX_NBUS, CX_DB * CX_NBUS * sizeof *C.bus);
			cref_step(&C, t, tok[s], s >= 1 ? tok[s - 1] : CX_NG_PAD, s >= 2 ? tok[s - 2] : CX_NG_PAD);
			for (int h = 0; h < nheads; h++) {
				const rstep *r = &C.steps[(long)(h * stride) * L + t];
				float *q = Q + ((size_t)h * ntok + s) * CX_DS, *k = K + ((size_t)h * ntok + s) * CX_DS;
				for (int d = 0; d < CX_DS; d++) { q[d] = (float)r->y[d]; k[d] = (float)r->k[d]; }
			}
		}
		printf("ctx-recall: %s over %s[%ld..%ld): %ld steps of q, k for %d heads (every %dth column) in %.1f s\n", argv[1], argv[2], skip, skip + ntok, ntok, nheads, stride, now() - t0);
		if (save) {
			FILE *o = fopen(save, "wb");
			if (o) { fwrite(Q, sizeof *Q, (size_t)nheads * ntok * CX_DS, o); fwrite(K, sizeof *K, (size_t)nheads * ntok * CX_DS, o); fclose(o); printf("ctx-recall: q, k saved to %s\n", save); }
		}
	}
	fflush(stdout);
	const double t1 = now();
	simhash *SH = malloc((size_t)nheads * NBITS * sizeof *SH);
	ivf *IV = malloc((size_t)nheads * NIVF * sizeof *IV);
	for (int h = 0; h < nheads; h++) {
		for (int b = 0; b < NBITS; b++) simhash_init(&SH[h * NBITS + b], h * stride, bits_of[b]);
		for (int v = 0; v < NIVF; v++) ivf_init(&IV[h * NIVF + v], ivf_cfg[v].online, ivf_cfg[v].probe, ivf_cfg[v].eta);
	}
	stat sh[NBITS][NPROBE], iv[NIVF], within[3];
	memset(sh, 0, sizeof sh);
	memset(iv, 0, sizeof iv);
	memset(within, 0, sizeof within);
	const long reach[3] = { 256, 1024, 4096 };
	double exact_mean = 0;
	long queries = 0;
	uint64_t seed = 0x5EED5EED5EEDull;
	for (long t = 0; t < ntok; t++) {
		if (t >= from && (t - from) % every == 0) {
			for (int h = 0; h < nheads; h++) {
				const float *Kh = K + (size_t)h * ntok * CX_DS, *q = Q + ((size_t)h * ntok + t) * CX_DS;
				const long lo = t - W > 0 ? t - W : 0;
				top4 exact;
				top_init(&exact);
				for (long s = lo; s < t; s++) top_insert(&exact, dot64(q, Kh + (size_t)s * CX_DS), s);
				exact_mean += exact.s[0];
				queries++;
				for (int r = 0; r < 3; r++) { int in = 0; for (int i = 0; i < TOP; i++) if (exact.step[i] >= 0 && t - exact.step[i] <= reach[r]) in++; within[r].recall += in / (double)TOP; within[r].n++; }
				top4 recent;
				top_init(&recent);
				const long rlo = t - RECENT > lo ? t - RECENT : lo;
				for (long s = rlo; s < t; s++) top_insert(&recent, dot64(q, Kh + (size_t)s * CX_DS), s);
				for (int b = 0; b < NBITS; b++) {
					const simhash *S = &SH[h * NBITS + b];
					const uint32_t p = simhash_pattern(S, q);
					for (int pr = 0; pr < NPROBE; pr++) {
						top4 got = recent;
						int cand = (int)(t - rlo);
						cand += simhash_bucket(S, p, q, Kh, t, &got);
						if (pr >= 1) for (int i = 0; i < S->bits; i++) cand += simhash_bucket(S, p ^ (1u << i), q, Kh, t, &got);
						if (pr >= 2) for (int i = 0; i < S->bits; i++) for (int j = i + 1; j < S->bits; j++) cand += simhash_bucket(S, p ^ (1u << i) ^ (1u << j), q, Kh, t, &got);
						stat_add(&sh[b][pr], &exact, &got, cand);
					}
				}
				for (int v = 0; v < NIVF; v++) {
					const ivf *V = &IV[h * NIVF + v];
					if (!V->built) continue;
					top4 got = recent;
					const int cand = (int)(t - rlo) + ivf_query(V, q, Kh, t, lo, &got);
					stat_add(&iv[v], &exact, &got, cand);
				}
			}
		}
		/* the step enters every index */
		for (int h = 0; h < nheads; h++) {
			const float *k = K + ((size_t)h * ntok + t) * CX_DS;
			for (int b = 0; b < NBITS; b++) simhash_insert(&SH[h * NBITS + b], k, t);
			for (int v = 0; v < NIVF; v++) {
				ivf *V = &IV[h * NIVF + v];
				if (!V->online && (t + 1) % IVF_REFRESH == 0) ivf_refresh(V, K + (size_t)h * ntok * CX_DS, t + 1, &seed);
				else ivf_insert(V, k, t);
			}
		}
	}
	const double t2 = now();
	printf("ctx-recall: %ld queries (steps %ld.. every %d, %d heads), exact top-1 score mean %.4f; the exact top-4 within the last 256 steps %.3f, 1,024 %.3f, 4,096 %.3f; %.1f s\n",
	       queries, from, every, nheads, exact_mean / queries, within[0].recall / within[0].n, within[1].recall / within[1].n, within[2].recall / within[2].n, t2 - t1);
	printf("%-34s %8s %8s %8s %10s\n", "retrieval", "recall@4", "top-1", "ratio", "candidates");
	for (int b = 0; b < NBITS; b++) for (int pr = 0; pr < NPROBE; pr++) {
		char name[64];
		snprintf(name, sizeof name, "exact 256 + SimHash %2d bits probe %d", bits_of[b], pr);
		print_stat(name, &sh[b][pr]);
	}
	for (int v = 0; v < NIVF; v++) print_stat(ivf_cfg[v].name, &iv[v]);
	return 0;
}
