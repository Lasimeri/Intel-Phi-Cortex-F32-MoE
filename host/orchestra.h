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
 *   4. the lingual filter over the whole vocabulary: a piece is valid when the letters it adds keep the word being
 *      written on a listed word or a proper prefix of one, and every word it closes is listed (words count from a
 *      minimum corpus count on). Every bad piece is pushed to min(value - ORCH_PENALTY, best valid - ORCH_MARGIN), so
 *      the sampler chooses among the valid continuations with the model's own preferences between them, and a word
 *      committed as a prefix is completed to a listed word. (The first version judged only the candidates within
 *      ORCH_WINDOW of the top and penalised them uniformly; a window whose candidates were all bad sent the broken
 *      word out, and a 2-nat clamp was undone by the sampler's repetition penalty: results/orchestra-4-*.log.)
 *      Per step this costs the verdict of the letter-led pieces whose first letter can follow the partial word;
 *      the verdict of a piece that closes the word is a constant of the piece, computed once.
 *   5. the repetition constraint: a candidate (within ORCH_WINDOW of the top) that completes a token 4-gram already
 *      present in this sample loses ORCH_REPEAT nats.
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
#define ORCH_WINDOW 10.0f     /* the repetition constraint's candidates: within this many nats of the best mixed value */
#define ORCH_MAXCAND 256
#define ORCH_PENALTY 7.0f     /* nats taken from a piece that breaks a word */
#define ORCH_MARGIN 16.0f     /* a bad piece ends at least this far below the best valid continuation: beyond the reach of
                                 the sampler's repetition penalty (a valid top logit in the recent history loses at most a
                                 sixth of itself, under 10 nats for logits under 60) and under min-p 0.05 at temperature 0.5
                                 (a 2-nat margin leaked: measured 2026-10-26, results/orchestra-4-mix.log) */
#define ORCH_WORD_MIN 50      /* a listed word counts for the filter from this corpus count on: the lists keep transcript
                                 junk ("ap" 32, "ud" 39 in the mix list) whose prefixes open junk paths; the scorer's
                                 list stays complete */
#define ORCH_REPEAT 5.0f      /* nats taken from a piece that repeats a 4-gram of this sample */
#define ORCH_REPBITS 16       /* the 4-gram set: 65,536 slots */

enum { ORCH_CLOSES, ORCH_LETTERS, ORCH_LETTERS_THEN, ORCH_APOSTROPHE };   /* piece kinds, see orch_pieces */

typedef struct { uint32_t total; uint32_t ctx; uint16_t id[NGC_TOP]; uint32_t n[NGC_TOP]; } __attribute__((packed)) ngc_bucket;

typedef struct {
	/* the count model (tools/ngcount.c) */
	const uint32_t *uni;
	uint64_t ntok;
	uint32_t buckets;
	const ngc_bucket *B, *T;
	/* settings */
	float lambda;
	int filter, norepeat, wmin;
	/* working state */
	float *lpuni;                 /* log unigram probabilities, 32-byte aligned */
	float *lpng;                  /* log p_ng(v) of the current step */
	char partial[COH_WMAX];       /* the letters of the word being generated */
	int plen;
	uint8_t pkind[NGC_VOCAB];     /* the piece's kind (the enum above) */
	uint8_t pself[NGC_VOCAB];     /* kind ORCH_CLOSES: 1 when the piece's own content breaks a word (constant) */
	char pfirst[NGC_VOCAB];       /* the first letter, lower case, of a letter-led piece */
	uint8_t bad[NGC_VOCAB];       /* the filter's verdict per piece at this step */
	uint64_t *rep;                /* open-addressed set of the token 4-grams seen in this sample */
	uint32_t repmask, repcount;
	int synced;                   /* history tokens already folded into partial and rep */
	/* counters for the run summary */
	long adjusted, cand, penalised, rescued, scans, repeated, ng3, ng2;
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

static inline int orch_is_word(const char *w) { const coh_entry *e = coh_find(w, 0); return e && e->n > 0; }

static inline int orch_is_prefix(const char *w)
{
	char key[COH_WMAX + 2];
	key[0] = '^';
	memcpy(key + 1, w, strlen(w) + 1);
	return coh_find(key, 0) != NULL;
}

/* words under the minimum count are marked absent (n = 0; the entry stays so the probe chains hold), then every
 * proper prefix of every remaining word is stored as "^prefix" in the same table (a key space of its own) */
static inline void orch_prefixes(int wmin)
{
	char key[COH_WMAX + 1];
	long kept = 0;
	for (uint32_t i = 0; i < COH_HSIZE; i++) {
		coh_entry *e = &coh_tab[i];
		if (!e->w[0] || strchr(e->w, '_')) continue;
		if (e->n < (uint32_t)wmin) e->n = 0;
		else kept++;
	}
	for (uint32_t i = 0; i < COH_HSIZE; i++) {
		const char *w = coh_tab[i].w;
		if (!w[0] || !coh_tab[i].n || strchr(w, '_') || w[0] == '^') continue;
		const int n = (int)strlen(w);
		for (int l = 1; l < n && l < COH_WMAX - 2; l++) {
			key[0] = '^';
			memcpy(key + 1, w, (size_t)l);
			key[l + 1] = 0;
			coh_entry *e = coh_find(key, 1);
			if (e && !e->n) e->n = 1;
		}
	}
	fprintf(stderr, "orchestra: word filter over %ld words with count >= %d\n", kept, wmin);
}

/* ---- the lingual verdict of one piece ---- */

/* 1 when appending the piece to the partial word breaks a word: a word it closes (the partial one, or one inside the
 * piece) is not listed, or the letters it leaves open are no listed word and no proper prefix of one. Digits,
 * punctuation and whitespace close words; an apostrophe inside letters belongs to the word. */
static inline int orch_bad_piece(const char *partial, int plen, const unsigned char *s, int len)
{
	char w[COH_WMAX + 1];
	int n = plen;
	memcpy(w, partial, (size_t)n);
	for (int i = 0; i < len; i++) {
		const int c = s[i];
		if (isalpha(c) || (c == '\'' && n)) {
			if (n < COH_WMAX - 2) w[n++] = (char)tolower(c);
			continue;
		}
		if (n) { w[n] = 0; if (!orch_is_word(w)) return 1; }
		n = 0;
	}
	if (n && n != plen) {
		w[n] = 0;
		if (!orch_is_prefix(w) && !orch_is_word(w)) return 1;
	}
	return 0;
}

/* the pieces' structure, once per vocabulary: a piece led by a letter extends the word (ORCH_LETTERS when it is
 * letters and apostrophes throughout, ORCH_LETTERS_THEN when something else follows), one led by an apostrophe may
 * extend it (ORCH_APOSTROPHE), any other piece closes it (ORCH_CLOSES) and its own content has a constant verdict */
static inline void orch_pieces(orch *o, unsigned char **tok_s, const int *tok_len, int eot)
{
	for (int v = 0; v < NGC_VOCAB; v++) {
		const unsigned char *s = tok_s[v];
		const int len = tok_len[v];
		o->pfirst[v] = 0;
		o->pself[v] = 0;
		if (v == eot || len == 0) { o->pkind[v] = ORCH_CLOSES; continue; }
		if (isalpha(s[0])) {
			int n = 0;
			while (n < len && (isalpha(s[n]) || s[n] == '\'')) n++;
			o->pkind[v] = n == len ? ORCH_LETTERS : ORCH_LETTERS_THEN;
			o->pfirst[v] = (char)tolower(s[0]);
		} else if (s[0] == '\'') {
			o->pkind[v] = ORCH_APOSTROPHE;
		} else {
			o->pkind[v] = ORCH_CLOSES;
			o->pself[v] = (uint8_t)orch_bad_piece("", 0, s, len);
		}
	}
}

/* the verdict of every piece at this step */
static inline void orch_scan(orch *o, unsigned char **tok_s, const int *tok_len)
{
	/* the partial word: closable when empty or listed; the letters that can follow it */
	o->partial[o->plen] = 0;
	const int closable = o->plen == 0 || orch_is_word(o->partial);
	uint8_t ok[128];
	char w[COH_WMAX + 2];
	memcpy(w, o->partial, (size_t)o->plen);
	for (int c = 0; c < 128; c++) {
		ok[c] = 0;
		if (!islower(c) || o->plen >= COH_WMAX - 2) continue;
		w[o->plen] = (char)c;
		w[o->plen + 1] = 0;
		ok[c] = orch_is_prefix(w) || orch_is_word(w);
	}
	for (int v = 0; v < NGC_VOCAB; v++) {
		switch (o->pkind[v]) {   /* the end-of-text piece closes the word like any other separator (it closed 4 of 4 residual broken words, results/orchestra-5-*.log) */
		case ORCH_CLOSES: o->bad[v] = closable ? o->pself[v] : 1; break;
		case ORCH_APOSTROPHE: o->bad[v] = (uint8_t)orch_bad_piece(o->partial, o->plen, tok_s[v], tok_len[v]); break;
		default: o->bad[v] = ok[(int)(unsigned char)o->pfirst[v]] ? (uint8_t)orch_bad_piece(o->partial, o->plen, tok_s[v], tok_len[v]) : 1; break;
		}
	}
	o->scans++;
}

/* the filter: every bad piece below the best valid continuation (or just penalised when no listed continuation exists) */
static inline void orch_filter(orch *o, float *logits, const int *cand, int nc, unsigned char **tok_s, const int *tok_len)
{
	orch_scan(o, tok_s, tok_len);
	float mvalid = -1e30f;
	for (int v = 0; v < NGC_VOCAB; v++) if (!o->bad[v] && logits[v] > mvalid) mvalid = logits[v];
	const float cap = mvalid > -1e30f ? mvalid - ORCH_MARGIN : 1e30f;
	long pushed = 0;
	for (int v = 0; v < NGC_VOCAB; v++) {
		if (!o->bad[v]) continue;
		const float p = logits[v] - ORCH_PENALTY;
		logits[v] = p < cap ? p : cap;
		pushed++;
	}
	o->penalised += pushed;
	int window_valid = 0;
	for (int i = 0; i < nc && !window_valid; i++) window_valid = !o->bad[cand[i]];
	if (!window_valid) o->rescued++;   /* the top candidates were all bad: the word would have broken */
}

static inline void orch_track_word(orch *o, const unsigned char *s, int len)
{
	for (int i = 0; i < len; i++) {
		const int c = s[i];
		if (isalpha(c) || (c == '\'' && o->plen)) { if (o->plen < COH_WMAX - 2) o->partial[o->plen++] = (char)tolower(c); }
		else o->plen = 0;
	}
}

/* ---- loading ---- */

static inline int orch_load(orch *o, const char *ngc, const char *wordlist, float lambda, int filter, int norepeat, int wmin)
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
	o->wmin = wmin;
	o->repmask = (1u << ORCH_REPBITS) - 1;
	o->rep = calloc(1u << ORCH_REPBITS, sizeof *o->rep);
	if (filter) {
		if (coh_load(wordlist)) { perror(wordlist); return -1; }
		orch_prefixes(wmin);
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
static inline void orch_sync(orch *o, const uint32_t *hist, int nhist, unsigned char **tok_s, const int *tok_len, int eot)
{
	if (!o->synced) orch_pieces(o, tok_s, tok_len, eot);
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
	orch_sync(o, hist, nhist, tok_s, tok_len, eot);
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
	/* 4. the lingual filter, 5. the repetition constraint on the candidates */
	if (o->filter || o->norepeat) {
		int cand[ORCH_MAXCAND];
		const int nc = orch_candidates(logits, orch_max(logits), cand, ORCH_MAXCAND);
		o->cand += nc;
		if (o->filter) orch_filter(o, logits, cand, nc, tok_s, tok_len);
		if (o->norepeat && nhist >= 3) {
			for (int i = 0; i < nc; i++) {
				const int v = cand[i];
				if (v == eot) continue;
				if (orch_rep_has(o, orch_key4(hist[nhist - 3], hist[nhist - 2], hist[nhist - 1], (uint32_t)v))) { logits[v] -= ORCH_REPEAT; o->repeated++; }
			}
		}
	}
	o->adjusted++;
}
#endif
