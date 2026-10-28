/*
 * bpe: Stage 2, host side. Byte-level BPE with a GPT-2-style pre-tokeniser, vocabulary 4096 = 256 bytes + merges +
 * the special tokens at the top (<|endoftext|> = 4095 alone in the stories' vocabulary; the chat vocabulary adds
 * <|system|> 4094, <|user|> 4093, <|assistant|> 4092, <|end|> 4091, <|think|> 4090, <|/think|> 4089, and so has
 * 3,833 merges). C only; no count tables are used for generation anywhere, this is the tokenizer.
 *
 *   bpe train  CORPUS SAMPLE_BYTES VOCAB.txt     learn the merges on the first SAMPLE_BYTES of CORPUS (all specials)
 *   bpe encode VOCAB.txt IN OUT.u32 [MAX_BYTES]  tokens as little-endian u32 (lib/cx-proto.h cx_tok, 4 B a token);
 *                                                prints tokens, bytes/token, MB/s
 *   bpe decode VOCAB.txt IN.u32 OUT              bytes back (the check is `cmp` against the original)
 *
 * Pre-tokeniser (after GPT-2's ` ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`, on bytes): the literal of a
 * special token is one token; a word is an optional single space plus a run of letters (bytes >= 0x80 count as
 * letters), or of digits, or of other non-space bytes; a run of whitespace leaves its last byte to the next word when a
 * non-space follows. Words longer than 64 bytes are cut into 64-byte pieces. Merges never cross word boundaries.
 *
 * VOCAB.txt: line 1 "cxbpe 4096" (the stories' file: one special) or "cxbpe 4096 specials N", then one line per
 * merge in rank order: "ID LEFT RIGHT" (ID = 256 + rank).
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "../lib/cx-proto.h"   /* cx_tok: a token id is stored as u32 */

#define VOCAB 4096
#define EOT 4095
#define WMAX 64
#define TAB (1u << 21)           /* word table slots (power of two) */

/* the special tokens, one id each from the top of the vocabulary, in id order downwards */
typedef struct { const char *s; int len; int id; } special;
static const special specials[] = {
	{ "<|endoftext|>", 13, 4095 }, { "<|system|>", 10, 4094 }, { "<|user|>", 8, 4093 }, { "<|assistant|>", 13, 4092 },
	{ "<|end|>", 7, 4091 }, { "<|think|>", 9, 4090 }, { "<|/think|>", 10, 4089 },
};
#define NSPECIAL_ALL ((int)(sizeof specials / sizeof *specials))
/* how many the vocabulary in use holds: <|endoftext|> alone until a vocabulary file says more; train uses them all */
static int nspecial = 1;
#define NMERGE (VOCAB - 256 - nspecial)

/* the token strings for decoding: bytes, merges expanded on demand, the specials set by load_vocab */
static unsigned char *tok_s[VOCAB];
static int tok_len[VOCAB];

static int is_special(int id) { return id >= VOCAB - nspecial && id < VOCAB; }

/* ---- pre-tokeniser ---- */
static int cls(unsigned char c)
{
	if (c == ' ' || c == '\n' || c == '\t' || c == '\r' || c == '\f' || c == '\v') return 0;   /* space */
	if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || c >= 0x80) return 1;              /* letter */
	if (c >= '0' && c <= '9') return 2;                                                        /* digit */
	return 3;                                                                                  /* other */
}

/* the special token starting at p (n bytes left), or -1 */
static int special_at(const unsigned char *p, long n)
{
	if (*p != '<') return -1;
	for (int k = 0; k < nspecial; k++)
		if (n >= specials[k].len && !memcmp(p, specials[k].s, (size_t)specials[k].len)) return k;
	return -1;
}

/* next word at p (n bytes left): returns its length; *special = the special token's id, or 0 for a word. A run
 * of bytes stops where a special token starts, so a marker glued to the text before it ("answer.<|end|>") is
 * still its own token. */
static int next_word(const unsigned char *p, long n, int *special)
{
	*special = 0;
	const int k = special_at(p, n);
	if (k >= 0) { *special = specials[k].id; return specials[k].len; }
	long i = 0;
	if (p[0] == ' ' && n > 1 && cls(p[1]) != 0 && special_at(p + 1, n - 1) < 0) i = 1;
	const int c = cls(p[i]);
	long j = i + 1;
	while (j < n && cls(p[j]) == c && special_at(p + j, n - j) < 0) j++;
	if (c == 0 && j < n && j - i > 1) j--;   /* \s+(?!\S): leave the last space for the next word */
	if (j > WMAX) j = WMAX;
	return (int)j;
}

/* ---- word table: unique words with counts ---- */
typedef struct { unsigned char *s; int len; long count; } word;
static word *tab;
static long nwords;

static uint64_t hash(const unsigned char *s, int len)
{
	uint64_t h = 1469598103934665603ull;
	for (int i = 0; i < len; i++) h = (h ^ s[i]) * 1099511628211ull;
	return h;
}

static word *lookup(const unsigned char *s, int len, int insert)
{
	uint64_t i = hash(s, len) & (TAB - 1);
	for (;;) {
		word *w = &tab[i];
		if (!w->s) {
			if (!insert) return NULL;
			w->s = malloc((size_t)len);
			memcpy(w->s, s, (size_t)len);
			w->len = len;
			nwords++;
			if (nwords > TAB * 3 / 4) { fprintf(stderr, "bpe: word table full\n"); exit(1); }
			return w;
		}
		if (w->len == len && !memcmp(w->s, s, (size_t)len)) return w;
		i = (i + 1) & (TAB - 1);
	}
}

/* ---- merges ---- */
static int merge_l[VOCAB], merge_r[VOCAB];   /* for ids 256..VOCAB - nspecial - 1 */
static int nmerges;
#define PTAB (1u << 22)
static uint64_t *pkey;    /* pair_key(left, right) + 1 = ((uint64_t)left << 32 | right) + 1, 0 = empty */
static int *pval;

/* the probe start of the pair (l, r) in a PTAB table: the multiplicative hash of l << 16 | r in 32 bits, so the
 * tables' layout is the one the 16-bit-packed key gave; the key compared in the slot is the full 64-bit pair */
static uint64_t pair_slot(int l, int r)
{
	const uint32_t h = (uint32_t)l << 16 | (uint32_t)r;
	return (uint64_t)((h * 2654435761u) & (PTAB - 1));
}

static uint64_t pair_key(int l, int r) { return (uint64_t)(uint32_t)l << 32 | (uint32_t)r; }

static int pair_rank(int l, int r)   /* merged id, or -1 */
{
	const uint64_t key = pair_key(l, r);
	uint64_t i = pair_slot(l, r);
	while (pkey[i]) {
		if (pkey[i] == key + 1) return pval[i];
		i = (i + 1) & (PTAB - 1);
	}
	return -1;
}

static void pair_set(int l, int r, int id)
{
	const uint64_t key = pair_key(l, r);
	uint64_t i = pair_slot(l, r);
	while (pkey[i] && pkey[i] != key + 1) i = (i + 1) & (PTAB - 1);
	pkey[i] = key + 1;
	pval[i] = id;
}

/* encode one word's bytes into ids by applying merges lowest id first; out takes up to WMAX ids */
static int encode_word(const unsigned char *s, int len, cx_tok *out)
{
	int sym[WMAX], n = len;
	for (int i = 0; i < n; i++) sym[i] = s[i];
	for (;;) {
		int best = VOCAB, at = -1;
		for (int i = 0; i + 1 < n; i++) {
			const int id = pair_rank(sym[i], sym[i + 1]);
			if (id >= 0 && id < best) { best = id; at = i; }
		}
		if (at < 0) break;
		sym[at] = best;
		memmove(sym + at + 1, sym + at + 2, (size_t)(n - at - 2) * sizeof *sym);
		n--;
	}
	for (int i = 0; i < n; i++) out[i] = (cx_tok)sym[i];
	return n;
}

/* the special tokens' strings, as the vocabulary in use holds them */
static void set_specials(void)
{
	for (int k = 0; k < nspecial; k++) { tok_s[specials[k].id] = (unsigned char *)specials[k].s; tok_len[specials[k].id] = specials[k].len; }
}

/* ---- train ---- */
static int train(const char *corpus, long sample, const char *vocab)
{
	nspecial = NSPECIAL_ALL;
	FILE *f = fopen(corpus, "rb");
	if (!f) { perror(corpus); return 1; }
	unsigned char *buf = malloc((size_t)sample + WMAX);
	const long n = (long)fread(buf, 1, (size_t)sample, f);
	fclose(f);
	tab = calloc(TAB, sizeof *tab);
	long pos = 0, total = 0;
	while (pos < n) {
		int sp;
		const int len = next_word(buf + pos, n - pos, &sp);
		if (!sp) lookup(buf + pos, len, 1)->count++;
		pos += len;
		total++;
	}
	printf("train: %ld bytes, %ld words, %ld unique, %d special tokens\n", n, total, nwords, nspecial);

	/* each unique word as a symbol array */
	word *ws = malloc((size_t)nwords * sizeof *ws);
	int **sym = malloc((size_t)nwords * sizeof *sym), *slen = malloc((size_t)nwords * sizeof *slen);
	long k = 0;
	for (uint64_t i = 0; i < TAB; i++)
		if (tab[i].s) {
			ws[k] = tab[i];
			sym[k] = malloc((size_t)ws[k].len * sizeof **sym);
			for (int j = 0; j < ws[k].len; j++) sym[k][j] = ws[k].s[j];
			slen[k] = ws[k].len;
			k++;
		}
	pkey = calloc(PTAB, sizeof *pkey);
	pval = calloc(PTAB, sizeof *pval);
	/* pair counts: a second table keyed the same way (touched holds slot indices, not ids) */
	uint64_t *ckey = calloc(PTAB, sizeof *ckey);
	long *cval = calloc(PTAB, sizeof *cval);
	uint32_t *touched = malloc(PTAB * sizeof *touched);
	const double t0 = (double)clock() / CLOCKS_PER_SEC;
	for (int m = 0; m < NMERGE; m++) {
		long nt = 0;
		for (long w = 0; w < nwords; w++)
			for (int j = 0; j + 1 < slen[w]; j++) {
				const uint64_t key = pair_key(sym[w][j], sym[w][j + 1]);
				uint64_t i = pair_slot(sym[w][j], sym[w][j + 1]);
				while (ckey[i] && ckey[i] != key + 1) i = (i + 1) & (PTAB - 1);
				if (!ckey[i]) { ckey[i] = key + 1; cval[i] = 0; touched[nt++] = (uint32_t)i; }
				cval[i] += ws[w].count;
			}
		long best = -1;
		uint64_t bkey = 0;   /* ties go to the lowest (left, right): the 64-bit key orders them as the 16-bit-packed one did */
		for (long t = 0; t < nt; t++)
			if (cval[touched[t]] > best || (cval[touched[t]] == best && ckey[touched[t]] < bkey)) { best = cval[touched[t]]; bkey = ckey[touched[t]]; }
		for (long t = 0; t < nt; t++) ckey[touched[t]] = 0;
		if (best < 2) { printf("train: no pair left with count >= 2 after %d merges\n", m); break; }
		const int l = (int)((bkey - 1) >> 32), r = (int)((bkey - 1) & 0xFFFFFFFFu), id = 256 + m;
		merge_l[id] = l;
		merge_r[id] = r;
		pair_set(l, r, id);
		nmerges = m + 1;
		for (long w = 0; w < nwords; w++) {
			int *s = sym[w], o = 0;
			for (int j = 0; j < slen[w]; j++) {
				if (j + 1 < slen[w] && s[j] == l && s[j + 1] == r) { s[o++] = id; j++; }
				else s[o++] = s[j];
			}
			slen[w] = o;
		}
		if (m % 500 == 0 || m == NMERGE - 1)
			printf("merge %4d: id %d = (%d, %d) count %ld, %.1f s\n", m, id, l, r, best, (double)clock() / CLOCKS_PER_SEC - t0);
	}
	f = fopen(vocab, "w");
	if (!f) { perror(vocab); return 1; }
	fprintf(f, "cxbpe %d specials %d\n", VOCAB, nspecial);
	for (int id = 256; id < 256 + nmerges; id++) fprintf(f, "%d %d %d\n", id, merge_l[id], merge_r[id]);
	fclose(f);
	printf("train: %d merges written to %s (vocab %d = 256 + %d + %d specials)\n", nmerges, vocab, 256 + nmerges + nspecial, nmerges, nspecial);
	return 0;
}

static int load_vocab(const char *vocab)
{
	FILE *f = fopen(vocab, "r");
	char head[64];
	int v = 0, ns = 1;
	if (!f || !fgets(head, sizeof head, f) || sscanf(head, "cxbpe %d specials %d", &v, &ns) < 1 || v != VOCAB || ns < 1 || ns > NSPECIAL_ALL) {
		fprintf(stderr, "bpe: bad vocab %s\n", vocab);
		return -1;
	}
	nspecial = ns;
	pkey = calloc(PTAB, sizeof *pkey);
	pval = calloc(PTAB, sizeof *pval);
	int id, l, r;
	while (fscanf(f, "%d %d %d\n", &id, &l, &r) == 3) {
		if (id < 256 || id >= VOCAB - nspecial) { fprintf(stderr, "bpe: merge id %d outside the vocabulary of %s\n", id, vocab); return -1; }
		merge_l[id] = l;
		merge_r[id] = r;
		pair_set(l, r, id);
		nmerges++;
	}
	fclose(f);
	set_specials();
	return 0;
}

/* ---- encode with a word cache ---- */
typedef struct { unsigned char *s; int len; int n; cx_tok ids[WMAX]; } centry;
static centry *cache;

static int encode(const char *vocab, const char *in, const char *out, long max)
{
	if (load_vocab(vocab)) return 1;
	FILE *f = fopen(in, "rb");
	if (!f) { perror(in); return 1; }
	fseek(f, 0, SEEK_END);
	long n = ftell(f);
	fseek(f, 0, SEEK_SET);
	if (max > 0 && n > max) n = max;
	unsigned char *buf = malloc((size_t)n + WMAX);
	if (fread(buf, 1, (size_t)n, f) != (size_t)n) { fprintf(stderr, "bpe: read %s\n", in); return 1; }
	fclose(f);
	cache = calloc(TAB, sizeof *cache);
	FILE *o = fopen(out, "wb");
	if (!o) { perror(out); return 1; }
	cx_tok *obuf = malloc((1 << 20) * sizeof *obuf);
	long on = 0, tokens = 0, pos = 0, hits = 0, special_tokens = 0;
	const double t0 = (double)clock() / CLOCKS_PER_SEC;
	while (pos < n) {
		int sp;
		const int len = next_word(buf + pos, n - pos, &sp);
		if (sp) { obuf[on++] = (cx_tok)sp; tokens++; special_tokens++; }
		else {
			uint64_t i = hash(buf + pos, len) & (TAB - 1);
			centry *c;
			for (;;) {
				c = &cache[i];
				if (!c->s) break;
				if (c->len == len && !memcmp(c->s, buf + pos, (size_t)len)) break;
				i = (i + 1) & (TAB - 1);
			}
			if (!c->s) {
				c->s = malloc((size_t)len);
				memcpy(c->s, buf + pos, (size_t)len);
				c->len = len;
				c->n = encode_word(buf + pos, len, c->ids);
			} else hits++;
			memcpy(obuf + on, c->ids, (size_t)c->n * sizeof *obuf);
			on += c->n;
			tokens += c->n;
		}
		pos += len;
		if (on > (1 << 20) - WMAX) { fwrite(obuf, sizeof *obuf, (size_t)on, o); on = 0; }
	}
	fwrite(obuf, sizeof *obuf, (size_t)on, o);
	fclose(o);
	const double s = (double)clock() / CLOCKS_PER_SEC - t0;
	printf("encode: %ld bytes -> %ld tokens (%.3f bytes/token, %ld special), %.1f MB/s, cache hits %ld, %s %ld bytes\n", n, tokens,
	       (double)n / tokens, special_tokens, n / s / 1e6, hits, out, tokens * (long)CX_TOK_BYTES);
	return 0;
}

/* ---- decode ---- */
static void expand(int id)
{
	if (tok_s[id]) return;
	if (id < 256) { tok_s[id] = malloc(1); tok_s[id][0] = (unsigned char)id; tok_len[id] = 1; return; }
	if (is_special(id)) { set_specials(); return; }
	expand(merge_l[id]);
	expand(merge_r[id]);
	tok_len[id] = tok_len[merge_l[id]] + tok_len[merge_r[id]];
	tok_s[id] = malloc((size_t)tok_len[id]);
	memcpy(tok_s[id], tok_s[merge_l[id]], (size_t)tok_len[merge_l[id]]);
	memcpy(tok_s[id] + tok_len[merge_l[id]], tok_s[merge_r[id]], (size_t)tok_len[merge_r[id]]);
}

static int decode(const char *vocab, const char *in, const char *out)
{
	if (load_vocab(vocab)) return 1;
	for (int id = 0; id < 256 + nmerges; id++) expand(id);
	FILE *f = fopen(in, "rb"), *o = fopen(out, "wb");
	if (!f || !o) { perror(f ? out : in); return 1; }
	cx_tok t;
	long n = 0, bytes = 0;
	while (fread(&t, sizeof t, 1, f) == 1) {
		if (t >= (cx_tok)(256 + nmerges) && !(t < VOCAB && is_special((int)t))) { fprintf(stderr, "bpe: token %u out of range at %ld\n", t, n); return 1; }
		fwrite(tok_s[t], 1, (size_t)tok_len[t], o);
		bytes += tok_len[t];
		n++;
	}
	fclose(f);
	fclose(o);
	printf("decode: %ld tokens -> %ld bytes, %s\n", n, bytes, out);
	return 0;
}

#ifndef BPE_NO_MAIN
int main(int argc, char **argv)
{
	if (argc >= 5 && !strcmp(argv[1], "train")) return train(argv[2], atol(argv[3]), argv[4]);
	if (argc >= 5 && !strcmp(argv[1], "encode")) return encode(argv[2], argv[3], argv[4], argc > 5 ? atol(argv[5]) : 0);
	if (argc >= 5 && !strcmp(argv[1], "decode")) return decode(argv[2], argv[3], argv[4]);
	fprintf(stderr, "usage: bpe train CORPUS SAMPLE_BYTES VOCAB | encode VOCAB IN OUT.u32 [MAX_BYTES] | decode VOCAB IN.u32 OUT\n");
	return 2;
}
#endif
