/*
 * ngcount: builds the host orchestrator's token n-gram count model from a human token stream (never from model output).
 *   ngcount TOKENS.u32 OUT.ngc [NTOK]
 * Counts unigrams (4096), and bigram and trigram contexts in hashed tables: for a context (prev1) or (prev2, prev1) the
 * table holds, per bucket, the context count and a small list of the most frequent next tokens with their counts (top 8,
 * kept by a running min-replace policy). File format: header, unigram counts, bigram table, trigram table.
 * Bucket count 2^22 per order (buckets of u32 total + u32 context tag + 8 u16 ids + 8 u32 counts = 56 B packed:
 * 234,881,024 B per order).
 * TOKENS.u32 holds u32 ids (lib/cx-proto.h cx_tok, 4 B a token; a .u16 stream of before 2026-10-28 is widened by
 * tools/ids32 first); an id at 4,096 or above (a .u16 stream read as u32 pairs two ids into one) is refused. The .ngc
 * file keeps its u16 ids (host/orchestra.h ngc_bucket, the stories-era format): with every id below 4,096 the
 * narrowing into the bucket is exact, and the hashes take the ids by value, so a widened stream counts into the
 * same buckets with the same contents as its .u16 original.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../lib/cx-proto.h"

#define NGC_VOCAB 4096
#define NGC_TOP 8
#define NGC_BITS 22
#define NGC_BUCKETS (1u << NGC_BITS)
#define NGC_MAGIC 0x31434747u   /* "GGC1" */

typedef struct { uint32_t total; uint32_t ctx; uint16_t id[NGC_TOP]; uint32_t n[NGC_TOP]; } __attribute__((packed)) bucket;   /* ctx: a tag of the context for collision rejection */
_Static_assert(sizeof(bucket) == 56, "an .ngc bucket is 56 B, host/orchestra.h ngc_bucket");

/* every id below 4,096: the OR of all ids has no bit at 12 or above (one vectorised pass, as tools/conv-pos.c) */
static void ids_check(const char *path, const cx_tok *t, long n)
{
	cx_tok m = 0;
	for (long i = 0; i < n; i++) m |= t[i];
	if (!(m >> 12)) return;
	long i = 0;
	while (t[i] < NGC_VOCAB) i++;
	fprintf(stderr, "ngcount: %s: id %u at token %ld is outside the 4,096-id vocabulary (a .u16 stream? tools/ids32 widens one)\n", path, t[i], i);
	exit(1);
}

static uint64_t mix(uint64_t x) { x ^= x >> 31; x *= 0xBF58476D1CE4E5B9ull; x ^= x >> 29; x *= 0x94D049BB133111EBull; x ^= x >> 32; return x; }
static uint64_t h2(uint32_t p1) { return mix(0x9E3779B97F4A7C15ull ^ ((uint64_t)p1 << 1)); }
static uint64_t h3(uint32_t p2, uint32_t p1) { return mix(0x94D049BB133111EBull ^ ((uint64_t)p1 << 1) ^ ((uint64_t)p2 << 17)); }

static void add(bucket *T, uint64_t h, uint32_t next)
{
	bucket *b = T + (h & (NGC_BUCKETS - 1));
	const uint32_t tag = (uint32_t)(h >> 40) | 1u;
	if (b->ctx != tag) { if (b->total) return; b->ctx = tag; }   /* a colliding context keeps the bucket: first come first served */
	b->total++;
	int empty = -1, min = 0;
	for (int i = 0; i < NGC_TOP; i++) {
		if (b->id[i] == next && b->n[i]) { b->n[i]++; return; }
		if (!b->n[i] && empty < 0) empty = i;
		if (b->n[i] < b->n[min]) min = i;
	}
	if (empty >= 0) { b->id[empty] = (uint16_t)next; b->n[empty] = 1; return; }
	if (b->n[min] <= 1) { b->id[min] = (uint16_t)next; b->n[min] = 1; }   /* replace a singleton */
}

int main(int argc, char **argv)
{
	if (argc < 3) { fprintf(stderr, "usage: ngcount TOKENS.u32 OUT.ngc [NTOK]\n"); return 2; }
	FILE *f = fopen(argv[1], "rb");
	if (!f) { perror(argv[1]); return 1; }
	fseek(f, 0, SEEK_END);
	const long bytes = ftell(f);
	if (bytes % CX_TOK_BYTES) { fprintf(stderr, "ngcount: %s is %ld B, not a whole number of u32 ids\n", argv[1], bytes); return 1; }
	long n = bytes / CX_TOK_BYTES;
	fseek(f, 0, SEEK_SET);
	if (argc > 3 && atol(argv[3]) < n) n = atol(argv[3]);
	cx_tok *t = malloc((size_t)n * sizeof *t);
	if (fread(t, sizeof *t, (size_t)n, f) != (size_t)n) { perror(argv[1]); return 1; }
	fclose(f);
	ids_check(argv[1], t, n);
	uint32_t *uni = calloc(NGC_VOCAB, 4);
	bucket *B = calloc(NGC_BUCKETS, sizeof *B), *T = calloc(NGC_BUCKETS, sizeof *T);
	if (!uni || !B || !T) { fprintf(stderr, "ngcount: memory\n"); return 1; }
	for (long i = 0; i < n; i++) {
		uni[t[i]]++;
		if (i >= 1) add(B, h2(t[i - 1]), t[i]);
		if (i >= 2) add(T, h3(t[i - 2], t[i - 1]), t[i]);
	}
	long used_b = 0, used_t = 0;
	for (uint32_t i = 0; i < NGC_BUCKETS; i++) { used_b += B[i].total > 0; used_t += T[i].total > 0; }
	FILE *o = fopen(argv[2], "wb");
	if (!o) { perror(argv[2]); return 1; }
	const uint32_t hdr[4] = { NGC_MAGIC, NGC_VOCAB, NGC_BITS, NGC_TOP };
	const uint64_t ntok = (uint64_t)n;
	if (fwrite(hdr, 4, 4, o) != 4 || fwrite(&ntok, 8, 1, o) != 1 || fwrite(uni, 4, NGC_VOCAB, o) != NGC_VOCAB ||
	    fwrite(B, sizeof *B, NGC_BUCKETS, o) != NGC_BUCKETS || fwrite(T, sizeof *T, NGC_BUCKETS, o) != NGC_BUCKETS || fclose(o)) { perror(argv[2]); return 1; }
	printf("ngcount: %ld tokens -> %s: bigram contexts %ld, trigram contexts %ld of %u buckets each (%zu B per table)\n", n, argv[2], used_b, used_t, NGC_BUCKETS, (size_t)NGC_BUCKETS * sizeof *B);
	return 0;
}
