/*
 * soft-check: the record-to-token alignment of a soft-target stream (docs/DISTIL.md). The slot checksum proves the copy
 * to the card; this proves the records belong to the tokens: record i carries the teacher's distribution for the token
 * after token i, and the alignment (tools/soft-align.c) gives the hard target the leftover mass, so the actual next
 * token is among a distribution-carrying record's ids unless the top-K cut dropped it. A record stream shifted by even
 * one token against its tokens puts the next token among the ids almost never.
 *   soft-check TOKENS.u32 RECORDS.soft [--rec BYTES] [--from TOKEN] [--count N] [--min PERCENT]
 * TOKENS.u32: little-endian u32 ids (lib/cx-proto.h cx_tok). --rec: the record size: 68 (the default, lib/cx-proto.h
 * cx_soft: 8 u32 ids, 0xFFFFFFFF unused), or the older records with 8 u16 ids at the record's start (0xFFFF unused):
 * 52 B (written until 2026-10-28 06:xx), 48 B (until 2026-10-27 22:00). An older record set is read beside the same
 * u32 tokens: a .u16 stream widened to .u32 is checked against its records as they were written. --from and --count:
 * the stretch checked (default all), --min: the share below which it fails (default 85). Prints the record and token
 * counts, the records with a distribution, the share of them holding the next token, and the same share for the
 * records shifted by +1 and -1 (the controls); exit 1 on a token file of a partial id, a count mismatch or a share
 * under --min.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include "../lib/cx-proto.h"

#define NONE16 0xFFFFu   /* an unused id of the older u16-id records */

static long fsize(const char *p) { struct stat st; return stat(p, &st) ? -1 : (long)st.st_size; }

/* id k of the record at r: a cx_soft record's u32 id (wide), or an older record's u16 id with 0xFFFF read as
 * CX_SOFT_NONE */
static cx_tok rec_id(const unsigned char *r, int wide, int k)
{
	if (wide) { cx_tok v; memcpy(&v, r + (size_t)k * sizeof v, sizeof v); return v; }
	uint16_t v;
	memcpy(&v, r + (size_t)k * sizeof v, sizeof v);
	return v == NONE16 ? CX_SOFT_NONE : (cx_tok)v;
}

int main(int argc, char **argv)
{
	if (argc < 3) { fprintf(stderr, "usage: soft-check TOKENS.u32 RECORDS.soft [--rec BYTES] [--from TOKEN] [--count N] [--min PERCENT]\n"); return 2; }
	const long rec_now = (long)CX_SOFT_BYTES;
	long rec = rec_now, from = 0, count = -1;
	double minp = 85;
	for (int i = 3; i < argc; i++) {
		if (!strcmp(argv[i], "--rec") && i + 1 < argc) rec = atol(argv[++i]);
		else if (!strcmp(argv[i], "--from") && i + 1 < argc) from = atol(argv[++i]);
		else if (!strcmp(argv[i], "--count") && i + 1 < argc) count = atol(argv[++i]);
		else if (!strcmp(argv[i], "--min") && i + 1 < argc) minp = atof(argv[++i]);
		else { fprintf(stderr, "soft-check: unknown option %s\n", argv[i]); return 2; }
	}
	if (rec != rec_now && rec != 52 && rec != 48) { fprintf(stderr, "soft-check: --rec %ld (u32 ids), or 52 or 48 (the older u16 ids)\n", rec_now); return 2; }
	const int wide = rec == rec_now;
	const long tb = fsize(argv[1]), rb = fsize(argv[2]);
	if (tb < 0 || rb < 0) { perror("soft-check"); return 2; }
	const long ntok = tb / (long)CX_TOK_BYTES, nrec = rb / rec;
	printf("soft-check: %s %ld tokens, %s %ld records of %ld B (%s ids)%s\n", argv[1], ntok, argv[2], nrec, rec, wide ? "u32" : "u16", rb % rec ? " (a partial record at the end)" : "");
	if (tb % (long)CX_TOK_BYTES) { printf("soft-check: FAIL: %s is %ld bytes, not a whole number of u32 ids\n", argv[1], tb); return 1; }
	if (nrec != ntok || rb % rec) { printf("soft-check: FAIL: the record count does not match the token count\n"); return 1; }
	FILE *ft = fopen(argv[1], "rb"), *fr = fopen(argv[2], "rb");
	cx_tok *tok = malloc((size_t)ntok * sizeof *tok);
	unsigned char *r = malloc((size_t)rb);
	if (!ft || !fr || fread(tok, sizeof *tok, (size_t)ntok, ft) != (size_t)ntok || fread(r, 1, (size_t)rb, fr) != (size_t)rb) { perror("soft-check"); return 2; }
	fclose(ft); fclose(fr);
	if (count < 0 || from + count > ntok - 1) count = ntok - 1 - from;
	long with = 0, hit[3] = { 0 };   /* hit[0]: the next token, hit[1]: the token after it (+1), hit[2]: the token itself (-1) */
	for (long i = from; i < from + count; i++) {
		const unsigned char *ri = r + i * rec;
		if (rec_id(ri, wide, 0) == CX_SOFT_NONE) continue;
		with++;
		const long want[3] = { i + 1, i + 2, i };
		for (int c = 0; c < 3; c++) {
			if (want[c] >= ntok) continue;
			for (int k = 0; k < CX_SOFT_K; k++) {
				const cx_tok id = rec_id(ri, wide, k);
				if (id == CX_SOFT_NONE) break;
				if (id == tok[want[c]]) { hit[c]++; break; }
			}
		}
	}
	const double s0 = with ? 100.0 * hit[0] / with : 0, s1 = with ? 100.0 * hit[1] / with : 0, s2 = with ? 100.0 * hit[2] / with : 0;
	printf("soft-check: tokens %ld..%ld: %ld records with a distribution; the next token among the ids %.1f %% (controls: shifted +1 %.1f %%, -1 %.1f %%)\n", from, from + count, with, s0, s1, s2);
	if (with && s0 < minp) { printf("soft-check: FAIL: %.1f %% < %.1f %%\n", s0, minp); return 1; }
	printf("soft-check: ok\n");
	return 0;
}
