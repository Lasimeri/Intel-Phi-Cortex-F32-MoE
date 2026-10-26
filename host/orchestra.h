/*
 * orchestra.h: the host-side orchestration model that keeps the Phi's token stream coherent and lingually correct
 * (user direction 2026-10-26 06:22). It runs on the host CPU's FMA units (AVX2 + FMA3 on the Ryzen 7 5800X).
 * Nothing in it comes from model output: the n-gram counts are built by tools/ngcount.c from the human token stream,
 * the word list by the coherence tool from the human text; the repetition constraint looks only at the sample being
 * generated and learns nothing.
 *
 * Per generated token, between the card's logits and the sampler:
 *   1. log p_phi(v) = logits(v) - logsumexp(logits)                     (AVX2: max, exp by an FMA polynomial, sum)
 *   2. log p_ng(v | p2, p1): trigram -> bigram -> unigram backoff with Dirichlet smoothing (prior mass ORCH_ALPHA
 *      towards the next lower order); the entries outside the two top-8 lists share one closed form (a vector add on
 *      the log unigram), the listed entries are computed exactly.
 *   3. mixed(v) = (1 - lambda) log p_phi(v) + lambda log p_ng(v) + logsumexp(logits)   (one fmadd per lane; the
 *      offset keeps the sampler's sign-dependent repetition penalty acting as on the raw logits; lambda 0 = identity)
 *   4. candidates = entries within ORCH_WINDOW nats of the best mixed value (one AVX2 compare pass); for each:
 *      a. lingual filter: a piece that ends the current word on a string outside the word list, or extends it past
 *         every listed prefix, loses ORCH_PENALTY nats
 *      b. repetition constraint: a piece that completes a token 4-gram already present in this sample loses
 *         ORCH_REPEAT nats
 * The sampler of host/cortex.c then draws from the adjusted values as before.
 */
#ifndef ORCHESTRA_H
#define ORCHESTRA_H
#include <ctype.h>
#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../tools/coherence.h"

#define NGC_VOCAB 4096
#define NGC_TOP 8
#define NGC_MAGIC 0x31434747u
#define ORCH_ALPHA 8.0f       /* Dirichlet prior mass of each backoff step */
#define ORCH_WINDOW 10.0f     /* candidates: within this many nats of the best mixed value */
#define ORCH_MAXCAND 256
#define ORCH_PENALTY 7.0f     /* nats taken from a piece that breaks a word */
#define ORCH_REPEAT 5.0f      /* nats taken from a piece that repeats a 4-gram of this sample */
#define ORCH_REPBITS 16       /* the 4-gram set: 65,536 slots */

typedef struct { uint32_t total; uint32_t ctx; uint16_t id[NGC_TOP]; uint32_t n[NGC_TOP]; } __attribute__((packed)) ngc_bucket;

typedef struct {
	/* the count model (tools/ngcount.c) */
	const uint32_t *uni;
	uint64_t ntok;
	uint32_t buckets;
	const ngc_bucket *B, *T;
	/* settings */
	float lambda;
	int filter, norepeat;
	/* working state */
	float *lpuni;                 /* log unigram probabilities, 32-byte aligned */
	float *lpng;                  /* log p_ng(v) of the current step */
	char partial[COH_WMAX];       /* the letters of the word being generated */
	int plen;
	uint64_t *rep;                /* open-addressed set of the token 4-grams seen in this sample */
	uint32_t repmask, repcount;
	int synced;                   /* history tokens already folded into partial and rep */
	/* counters for the run summary */
	long adjusted, cand, penalised, repeated, ng3, ng2;
} orch;

/* ---- hashing shared with tools/ngcount.c ---- */

static inline uint64_t orch_mix64(uint64_t x) { x ^= x >> 31; x *= 0xBF58476D1CE4E5B9ull; x ^= x >> 29; x *= 0x94D049BB133111EBull; x ^= x >> 32; return x; }
static inline uint64_t orch_h2(uint32_t p1) { return orch_mix64(0x9E3779B97F4A7C15ull ^ ((uint64_t)p1 << 1)); }
static inline uint64_t orch_h3(uint32_t p2, uint32_t p1) { return orch_mix64(0x94D049BB133111EBull ^ ((uint64_t)p1 << 1) ^ ((uint64_t)p2 << 17)); }

static inline const ngc_bucket *orch_bucket(const ngc_bucket *tab, uint32_t buckets, uint64_t h)
{
	const ngc_bucket *b = tab + (h & (buckets - 1));
	return b->total && b->ctx == ((uint32_t)(h >> 40) | 1u) ? b : NULL;
}

static inline uint32_t orch_count(const ngc_bucket *b, int v)
{
	for (int i = 0; i < NGC_TOP; i++) if (b->n[i] && b->id[i] == v) return b->n[i];
	return 0;
}

/* ---- the word list and its prefixes ---- */

/* every proper prefix of every listed word, stored as "^prefix" in the same table */
static inline void orch_prefixes(void)
{
	char key[COH_WMAX + 1];
	for (uint32_t i = 0; i < COH_HSIZE; i++) {
		const char *w = coh_tab[i].w;
		if (!w[0] || strchr(w, '_')) continue;
		const int n = (int)strlen(w);
		for (int l = 1; l < n && l < COH_WMAX - 2; l++) {
			key[0] = '^';
			memcpy(key + 1, w, (size_t)l);
			key[l + 1] = 0;
			coh_entry *e = coh_find(key, 1);
			if (e && !e->n) e->n = 1;
		}
	}
}

static inline int orch_load(orch *o, const char *ngc, const char *wordlist, float lambda, int filter, int norepeat)
{
	memset(o, 0, sizeof *o);
	FILE *f = fopen(ngc, "rb");
	if (!f) { perror(ngc); return -1; }
	uint32_t hdr[4];
	if (fread(hdr, 4, 4, f) != 4 || hdr[0] != NGC_MAGIC || hdr[1] != NGC_VOCAB || hdr[3] != NGC_TOP || fread(&o->ntok, 8, 1, f) != 1) {
		fprintf(stderr, "orchestra: %s is not an ngcount model\n", ngc);
		return -1;
	}
	o->buckets = 1u << hdr[2];
	uint32_t *uni = malloc(NGC_VOCAB * sizeof *uni);
	ngc_bucket *B = malloc((size_t)o->buckets * sizeof *B), *T = malloc((size_t)o->buckets * sizeof *T);
	if (!uni || !B || !T || fread(uni, sizeof *uni, NGC_VOCAB, f) != NGC_VOCAB || fread(B, sizeof *B, o->buckets, f) != o->buckets ||
	    fread(T, sizeof *T, o->buckets, f) != o->buckets) {
		fprintf(stderr, "orchestra: short model %s\n", ngc);
		return -1;
	}
	fclose(f);
	o->uni = uni;
	o->B = B;
	o->T = T;
	o->lpuni = aligned_alloc(32, NGC_VOCAB * sizeof *o->lpuni);
	o->lpng = aligned_alloc(32, NGC_VOCAB * sizeof *o->lpng);
	double tot = 0;
	for (int v = 0; v < NGC_VOCAB; v++) tot += uni[v];
	for (int v = 0; v < NGC_VOCAB; v++) o->lpuni[v] = (float)log((uni[v] + 0.5) / (tot + 0.5 * NGC_VOCAB));
	o->lambda = lambda;
	o->filter = filter;
	o->norepeat = norepeat;
	o->repmask = (1u << ORCH_REPBITS) - 1;
	o->rep = calloc(1u << ORCH_REPBITS, sizeof *o->rep);
	if (filter) {
		if (coh_load(wordlist)) { perror(wordlist); return -1; }
		orch_prefixes();
	}
	return 0;
}

/* ---- AVX2 + FMA kernels ---- */

/* e^x for 8 lanes: x = k ln2 + r, e^r by a degree-6 polynomial in fused multiply-adds, 2^k through the exponent field */
static inline __m256 orch_exp8(__m256 x)
{
	x = _mm256_min_ps(_mm256_max_ps(x, _mm256_set1_ps(-87.0f)), _mm256_set1_ps(88.0f));
	const __m256 k = _mm256_round_ps(_mm256_mul_ps(x, _mm256_set1_ps(1.44269504f)), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
	__m256 r = _mm256_fnmadd_ps(k, _mm256_set1_ps(0.693145752f), x);
	r = _mm256_fnmadd_ps(k, _mm256_set1_ps(1.42860677e-6f), r);
	__m256 p = _mm256_set1_ps(1.0f / 720.0f);
	p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0f / 120.0f));
	p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0f / 24.0f));
	p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0f / 6.0f));
	p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(0.5f));
	p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0f));
	p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0f));
	const __m256i e = _mm256_slli_epi32(_mm256_add_epi32(_mm256_cvtps_epi32(k), _mm256_set1_epi32(127)), 23);
	return _mm256_mul_ps(p, _mm256_castsi256_ps(e));
}

static inline float orch_hsum8(__m256 v)
{
	__m128 h = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
	h = _mm_hadd_ps(h, h);
	h = _mm_hadd_ps(h, h);
	return _mm_cvtss_f32(h);
}

static inline float orch_hmax8(__m256 v)
{
	__m128 h = _mm_max_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
	h = _mm_max_ps(h, _mm_movehl_ps(h, h));
	h = _mm_max_ss(h, _mm_shuffle_ps(h, h, 1));
	return _mm_cvtss_f32(h);
}

static inline float orch_max(const float *x)
{
	__m256 m = _mm256_set1_ps(-1e30f);
	for (int v = 0; v < NGC_VOCAB; v += 8) m = _mm256_max_ps(m, _mm256_loadu_ps(x + v));
	return orch_hmax8(m);
}

/* the indices whose value lies within ORCH_WINDOW of best, at most max of them */
static inline int orch_candidates(const float *x, float best, int *out, int max)
{
	const __m256 thr = _mm256_set1_ps(best - ORCH_WINDOW);
	int n = 0;
	for (int v = 0; v < NGC_VOCAB && n < max; v += 8) {
		const int m = _mm256_movemask_ps(_mm256_cmp_ps(_mm256_loadu_ps(x + v), thr, _CMP_GT_OQ));
		for (int b = 0; b < 8 && n < max; b++) if (m & (1 << b)) out[n++] = v + b;
	}
	return n;
}

/* ---- the n-gram distribution ---- */

/* log p_ng(v | p2, p1) for every v into o->lpng */
static inline void orch_ngram(orch *o, uint32_t p1, uint32_t p2, int have1, int have2)
{
	const ngc_bucket *b2 = have1 ? orch_bucket(o->B, o->buckets, orch_h2(p1)) : NULL;
	const ngc_bucket *b3 = have2 ? orch_bucket(o->T, o->buckets, orch_h3(p2, p1)) : NULL;
	o->ng2 += b2 != NULL;
	o->ng3 += b3 != NULL;
	const float d2 = b2 ? (float)b2->total + ORCH_ALPHA : 1.0f, d3 = b3 ? (float)b3->total + ORCH_ALPHA : 1.0f;
	/* outside the lists: p = (alpha / d3) (alpha / d2) p_uni, one shift of the log unigram */
	float shift = 0;
	if (b2) shift += logf(ORCH_ALPHA / d2);
	if (b3) shift += logf(ORCH_ALPHA / d3);
	const __m256 s = _mm256_set1_ps(shift);
	for (int v = 0; v < NGC_VOCAB; v += 8) _mm256_store_ps(o->lpng + v, _mm256_add_ps(_mm256_load_ps(o->lpuni + v), s));
	/* the listed entries exactly */
	for (int lvl = 0; lvl < 2; lvl++) {
		const ngc_bucket *b = lvl ? b3 : b2;
		if (!b) continue;
		for (int i = 0; i < NGC_TOP; i++) {
			if (!b->n[i]) continue;
			const int v = b->id[i];
			float p = expf(o->lpuni[v]);
			if (b2) p = ((float)orch_count(b2, v) + ORCH_ALPHA * p) / d2;
			if (b3) p = ((float)orch_count(b3, v) + ORCH_ALPHA * p) / d3;
			o->lpng[v] = logf(p);
		}
	}
}

/* ---- the lingual filter ---- */

/* 1 when appending the piece breaks the word being written: it ends on a string outside the list, or leaves every
 * listed prefix; 0 otherwise. Numbers, punctuation and whitespace pass. */
static inline int orch_bad_piece(const orch *o, const unsigned char *s, int len)
{
	char w[COH_WMAX + 1];
	int n = o->plen;
	memcpy(w, o->partial, (size_t)n);
	for (int i = 0; i < len; i++) {
		const int c = s[i];
		if (isalpha(c) || (c == '\'' && n)) {
			if (n < COH_WMAX - 2) w[n++] = (char)tolower(c);
			continue;
		}
		if (n >= 3) { w[n] = 0; if (!coh_find(w, 0)) return 1; }
		n = 0;
	}
	if (n >= 3 && n != o->plen) {
		w[n] = 0;
		char key[COH_WMAX + 2];
		key[0] = '^';
		memcpy(key + 1, w, (size_t)n + 1);
		if (!coh_find(key, 0) && !coh_find(w, 0)) return 1;
	}
	return 0;
}

static inline void orch_track_word(orch *o, const unsigned char *s, int len)
{
	for (int i = 0; i < len; i++) {
		const int c = s[i];
		if (isalpha(c) || (c == '\'' && o->plen)) { if (o->plen < COH_WMAX - 2) o->partial[o->plen++] = (char)tolower(c); }
		else o->plen = 0;
	}
}

/* ---- the repetition set ---- */

static inline uint64_t orch_key4(uint32_t a, uint32_t b, uint32_t c, uint32_t d)
{
	return (((uint64_t)a << 36) | ((uint64_t)b << 24) | ((uint64_t)c << 12) | d) + 1;
}

static inline uint32_t orch_slot(const orch *o, uint64_t key) { return (uint32_t)((key * 0x9E3779B97F4A7C15ull) >> 40) & o->repmask; }

static inline int orch_rep_has(const orch *o, uint64_t key)
{
	for (uint32_t i = orch_slot(o, key);; i = (i + 1) & o->repmask) {
		if (o->rep[i] == key) return 1;
		if (!o->rep[i]) return 0;
	}
}

static inline void orch_rep_add(orch *o, uint64_t key)
{
	if (o->repcount * 4 > (o->repmask + 1) * 3) return;   /* three quarters full: stop recording */
	for (uint32_t i = orch_slot(o, key);; i = (i + 1) & o->repmask) {
		if (o->rep[i] == key) return;
		if (!o->rep[i]) { o->rep[i] = key; o->repcount++; return; }
	}
}

/* fold the history tokens not yet seen into the partial word and the 4-gram set */
static inline void orch_sync(orch *o, const uint32_t *hist, int nhist, unsigned char **tok_s, const int *tok_len)
{
	for (int i = o->synced; i < nhist; i++) {
		orch_track_word(o, tok_s[hist[i]], tok_len[hist[i]]);
		if (i >= 3) orch_rep_add(o, orch_key4(hist[i - 3], hist[i - 2], hist[i - 1], hist[i]));
	}
	o->synced = nhist;
}

/* ---- the step ---- */

/* adjust the 4096 logits in place; hist[0..nhist-1] are the tokens so far (prompt and generated) */
static inline void orch_adjust(orch *o, float *logits, const uint32_t *hist, int nhist, unsigned char **tok_s, const int *tok_len, int eot)
{
	orch_sync(o, hist, nhist, tok_s, tok_len);
	/* 1. log-softmax of the card's logits */
	const float mx = orch_max(logits);
	__m256 sum = _mm256_setzero_ps();
	for (int v = 0; v < NGC_VOCAB; v += 8) sum = _mm256_add_ps(sum, orch_exp8(_mm256_sub_ps(_mm256_loadu_ps(logits + v), _mm256_set1_ps(mx))));
	const float lse = mx + logf(orch_hsum8(sum));
	/* 2. the n-gram distribution */
	orch_ngram(o, nhist >= 1 ? hist[nhist - 1] : 0, nhist >= 2 ? hist[nhist - 2] : 0, nhist >= 1, nhist >= 2);
	/* 3. the mixture, one fused multiply-add per lane; the log-sum-exp is added back so the result keeps the offset of
	 * the card's logits (the sampler's repetition penalty is sign-dependent): at lambda 0 the logits are unchanged */
	const __m256 a = _mm256_set1_ps(1.0f - o->lambda), b = _mm256_set1_ps(o->lambda), L = _mm256_set1_ps(lse);
	for (int v = 0; v < NGC_VOCAB; v += 8) {
		const __m256 lp = _mm256_sub_ps(_mm256_loadu_ps(logits + v), L);
		_mm256_storeu_ps(logits + v, _mm256_add_ps(_mm256_fmadd_ps(b, _mm256_load_ps(o->lpng + v), _mm256_mul_ps(a, lp)), L));
	}
	/* 4. the constraints on the candidates */
	if (o->filter || o->norepeat) {
		int cand[ORCH_MAXCAND];
		const int nc = orch_candidates(logits, orch_max(logits), cand, ORCH_MAXCAND);
		o->cand += nc;
		for (int i = 0; i < nc; i++) {
			const int v = cand[i];
			if (v == eot) continue;
			if (o->filter && orch_bad_piece(o, tok_s[v], tok_len[v])) { logits[v] -= ORCH_PENALTY; o->penalised++; }
			if (o->norepeat && nhist >= 3 && orch_rep_has(o, orch_key4(hist[nhist - 3], hist[nhist - 2], hist[nhist - 1], (uint32_t)v))) {
				logits[v] -= ORCH_REPEAT;
				o->repeated++;
			}
		}
	}
	o->adjusted++;
}
#endif
