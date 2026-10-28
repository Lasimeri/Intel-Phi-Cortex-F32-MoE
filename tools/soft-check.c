/*
 * soft-check: the record-to-token alignment of a soft-target stream (docs/DISTIL.md). The slot checksum proves the copy
 * to the card; this proves the records belong to the tokens: record i carries the teacher's distribution for the token
 * after token i, and the alignment (tools/soft-align.c) gives the hard target the leftover mass, so the actual next
 * token is among a distribution-carrying record's ids unless the top-K cut dropped it. A record stream shifted by even
 * one token against its tokens puts the next token among the ids almost never.
 *   soft-check TOKENS.u16 RECORDS.soft [--rec BYTES] [--from TOKEN] [--count N] [--min PERCENT]
 * --rec: the record size (52, lib/cx-proto.h cx_soft; 48 for the streams written before 2026-10-27 22:00), --from and
 * --count: the stretch checked (default all), --min: the share below which it fails (default 85). Prints the record
 * and token counts, the records with a distribution, the share of them holding the next token, and the same share
 * for the records shifted by +1 and -1 (the controls); exit 1 on a count mismatch or a share under --min.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#define K 8
#define NONE 0xFFFFu

static long fsize(const char *p) { struct stat st; return stat(p, &st) ? -1 : (long)st.st_size; }

int main(int argc, char **argv)
{
	if (argc < 3) { fprintf(stderr, "usage: soft-check TOKENS.u16 RECORDS.soft [--rec BYTES] [--from TOKEN] [--count N] [--min PERCENT]\n"); return 2; }
	long rec = 52, from = 0, count = -1;
	double minp = 85;
	for (int i = 3; i < argc; i++) {
		if (!strcmp(argv[i], "--rec") && i + 1 < argc) rec = atol(argv[++i]);
		else if (!strcmp(argv[i], "--from") && i + 1 < argc) from = atol(argv[++i]);
		else if (!strcmp(argv[i], "--count") && i + 1 < argc) count = atol(argv[++i]);
		else if (!strcmp(argv[i], "--min") && i + 1 < argc) minp = atof(argv[++i]);
		else { fprintf(stderr, "soft-check: unknown option %s\n", argv[i]); return 2; }
	}
	if (rec != 48 && rec != 52) { fprintf(stderr, "soft-check: --rec 48 or 52\n"); return 2; }
	const long tb = fsize(argv[1]), rb = fsize(argv[2]);
	if (tb < 0 || rb < 0) { perror("soft-check"); return 2; }
	const long ntok = tb / 2, nrec = rb / rec;
	printf("soft-check: %s %ld tokens, %s %ld records of %ld B%s\n", argv[1], ntok, argv[2], nrec, rec, rb % rec ? " (a partial record at the end)" : "");
	if (nrec != ntok || rb % rec) { printf("soft-check: FAIL: the record count does not match the token count\n"); return 1; }
	FILE *ft = fopen(argv[1], "rb"), *fr = fopen(argv[2], "rb");
	uint16_t *tok = malloc((size_t)ntok * 2);
	unsigned char *r = malloc((size_t)rb);
	if (!ft || !fr || fread(tok, 2, (size_t)ntok, ft) != (size_t)ntok || fread(r, 1, (size_t)rb, fr) != (size_t)rb) { perror("soft-check"); return 2; }
	fclose(ft); fclose(fr);
	if (count < 0 || from + count > ntok - 1) count = ntok - 1 - from;
	long with = 0, hit[3] = { 0 };   /* hit[0]: the next token, hit[1]: the token after it (+1), hit[2]: the token itself (-1) */
	for (long i = from; i < from + count; i++) {
		const uint16_t *id = (const uint16_t *)(r + i * rec);
		if (id[0] == NONE) continue;
		with++;
		const long want[3] = { i + 1, i + 2, i };
		for (int c = 0; c < 3; c++) {
			if (want[c] >= ntok) continue;
			for (int k = 0; k < K && id[k] != NONE; k++) if (id[k] == tok[want[c]]) { hit[c]++; break; }
		}
	}
	const double s0 = with ? 100.0 * hit[0] / with : 0, s1 = with ? 100.0 * hit[1] / with : 0, s2 = with ? 100.0 * hit[2] / with : 0;
	printf("soft-check: tokens %ld..%ld: %ld records with a distribution; the next token among the ids %.1f %% (controls: shifted +1 %.1f %%, -1 %.1f %%)\n", from, from + count, with, s0, s1, s2);
	if (with && s0 < minp) { printf("soft-check: FAIL: %.1f %% < %.1f %%\n", s0, minp); return 1; }
	printf("soft-check: ok\n");
	return 0;
}
