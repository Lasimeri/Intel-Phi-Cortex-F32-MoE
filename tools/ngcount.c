/*
 * ngcount: builds the host orchestrator's token n-gram count model from a human token stream (never from model output).
 *   ngcount TOKENS.u16 OUT.ngc [NTOK]
 * Counts unigrams (4096), and bigram and trigram contexts in hashed tables: for a context (prev1) or (prev2, prev1) the
 * table holds, per bucket, the context count and a small list of the most frequent next tokens with their counts (top 8,
 * kept by a running min-replace policy). File format: header, unigram counts, bigram table, trigram table.
 * Bucket count 2^22 per order (buckets of 8 x (u16 id, u32 count) + u32 total = 68 B: 285 MB per order).
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NGC_VOCAB 4096
#define NGC_TOP 8
#define NGC_BITS 22
#define NGC_BUCKETS (1u << NGC_BITS)
#define NGC_MAGIC 0x31434747u   /* "GGC1" */

typedef struct { uint32_t total; uint32_t ctx; uint16_t id[NGC_TOP]; uint32_t n[NGC_TOP]; } __attribute__((packed)) bucket;   /* ctx: a tag of the context for collision rejection */

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
	if (argc < 3) { fprintf(stderr, "usage: ngcount TOKENS.u16 OUT.ngc [NTOK]\n"); return 2; }
	FILE *f = fopen(argv[1], "rb");
	if (!f) { perror(argv[1]); return 1; }
	fseek(f, 0, SEEK_END);
	long n = ftell(f) / 2;
	fseek(f, 0, SEEK_SET);
	if (argc > 3 && atol(argv[3]) < n) n = atol(argv[3]);
	uint16_t *t = malloc((size_t)n * 2);
	if (fread(t, 2, (size_t)n, f) != (size_t)n) { perror(argv[1]); return 1; }
	fclose(f);
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
