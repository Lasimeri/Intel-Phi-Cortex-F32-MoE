/*
 * ids32: the token streams and soft records of before 2026-10-28 06:10 (u16 ids) widened to the u32 ids of
 * lib/cx-proto.h (cx_tok), exact: the same ids in the same order.
 *   ids32 IN.u16 OUT.u32                         every little-endian u16 id written as a little-endian u32 id
 *   ids32 --soft IN.soft OUT.soft --rec 48|52    the older soft records rewritten as 68-byte cx_soft records
 * The older records: 52 B { u16 id[8]; F32 p[8]; F32 w; } (10-27 22:00 to 10-28 06:xx) and 48 B { u16 id[8];
 * F32 p[8]; } (before 10-27 22:00). Each id is widened, the unused 0xFFFF becomes CX_SOFT_NONE, p is copied bit for
 * bit, w is the record's own for 52 B and 1.0f for 48 B (the plain weight: 48-byte streams trained unweighted).
 * Both modes then re-read OUT against IN from the start and compare every id (in --soft mode also every p and w,
 * bit for bit), print the counts and exit 1 on any difference, on an input that is not a whole number of ids or
 * records, or when OUT is IN (opening it for writing would truncate the input before it is read). The u16 mode also
 * refuses an input that is u32 already (a 0 at every odd u16 position: a widened stream under a .u16 name); its ids
 * are not range-checked (the conversion is exact whatever they hold), the count of ids at 4,096 or above is printed.
 * The --soft re-read also exits 1 on any record outside the bounds every writer of the older records keeps
 * (tools/soft-align.c, mixstream.c and reward-soft.c since the records' first commit 9c5c6f1): an id that is neither
 * CX_SOFT_NONE nor below 4,096, a p outside [0, 1] or not a number, a w that is not finite. The size check alone does
 * not see a wrong --rec (LCM(48, 52) = 624 B: a 48-byte file of 13 x m records passes as 52, a 52-byte file of
 * 12 x m as 48); read at the wrong size, a record takes a neighbour's ids, p or w into these fields, and the file is
 * refused when any of its records falls outside the bounds. It is a bound, not a proof: a single record read at the
 * wrong size can pass (a dense 68-byte record given as --rec 52 can pass on its first record), so --rec is still
 * chosen from what wrote the file. Every wrong case tried on placeholder files on 2026-10-28 was refused (48 B as 52
 * and 52 B as 48, with distributions and all empty; 68 B as 52 with distributions; empty 52-byte reward records with
 * w 0 as 48: 22 to 1,431 records outside the bounds a file), every right one passed byte-identical to the unguarded
 * ids32.
 * OUT is written before the re-read and stays on disk on exit 1.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include "../lib/cx-proto.h"

_Static_assert(__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__, "the files are little-endian and are read and written in host order");

#define BLK 65536L               /* ids or records a block */
#define OLD_NONE 0xFFFFu         /* an unused id of the older records */
#define OLD_P 16                 /* byte offsets in an older record: id[8] at 0, p[8] at 16, w at 48 (52 B only) */
#define OLD_W 48
#define F32_ONE 0x3F800000u      /* the bits of 1.0f */

static FILE *open_in(const char *path, struct stat *st)
{
	FILE *f = fopen(path, "rb");
	if (!f || fstat(fileno(f), st)) { perror(path); exit(1); }
	return f;
}

static void not_same(const char *in, const struct stat *si, const char *out)
{
	struct stat so;
	if (!stat(out, &so) && so.st_dev == si->st_dev && so.st_ino == si->st_ino) {
		fprintf(stderr, "ids32: %s is the input %s; write the widened copy to another file\n", out, in);
		exit(1);
	}
}

static FILE *reopen_out(const char *out, long long want)
{
	struct stat st;
	FILE *o = fopen(out, "rb");
	if (!o || fstat(fileno(o), &st)) { perror(out); exit(1); }
	if ((long long)st.st_size != want) { fprintf(stderr, "ids32: FAIL: %s is %lld B, %lld expected\n", out, (long long)st.st_size, want); exit(1); }
	return o;
}

static void read_all(void *p, size_t size, long k, FILE *f, const char *path)
{
	if (fread(p, size, (size_t)k, f) != (size_t)k) { fprintf(stderr, "ids32: short read of %s\n", path); exit(1); }
}

static int widen_stream(const char *in, const char *out)
{
	struct stat st;
	FILE *f = open_in(in, &st);
	if (st.st_size % 2) { fprintf(stderr, "ids32: %s is %lld B, not a whole number of u16 ids\n", in, (long long)st.st_size); return 1; }
	not_same(in, &st, out);
	const long n = (long)(st.st_size / 2);
	uint16_t *a = malloc(BLK * sizeof *a);
	cx_tok *b = malloc(BLK * sizeof *b);
	if (!a || !b) { perror("ids32"); return 1; }
	/* a stream that is u32 already (a widened file under a .u16 name) reads as its ids with a 0 after each, and
	 * widening it again would put a 0 after every id and still pass the re-read: refused when every id at an odd
	 * position is 0 (id 0 is the byte 0x00; a .u16 stream of text has a nonzero id at some odd position). BLK is
	 * even, so a block's odd indices are the stream's. */
	uint16_t odd = 0;
	for (long done = 0; done < n;) {
		const long k = n - done < BLK ? n - done : BLK;
		read_all(a, sizeof *a, k, f, in);
		for (long i = 1; i < k; i += 2) odd |= a[i];
		done += k;
	}
	if (n >= 2 && !odd) { fprintf(stderr, "ids32: %s holds a 0 at every odd position: a u32 stream already, not widened again\n", in); return 1; }
	rewind(f);
	FILE *o = fopen(out, "wb");
	if (!o) { perror(out); return 1; }
	for (long done = 0; done < n;) {
		const long k = n - done < BLK ? n - done : BLK;
		read_all(a, sizeof *a, k, f, in);
		for (long i = 0; i < k; i++) b[i] = a[i];
		if (fwrite(b, sizeof *b, (size_t)k, o) != (size_t)k) { perror(out); return 1; }
		done += k;
	}
	if (fclose(o)) { perror(out); return 1; }
	/* the check: OUT re-read against IN */
	rewind(f);
	o = reopen_out(out, (long long)n * CX_TOK_BYTES);
	long diff = 0, first = -1, high = 0;
	for (long done = 0; done < n;) {
		const long k = n - done < BLK ? n - done : BLK;
		read_all(a, sizeof *a, k, f, in);
		read_all(b, sizeof *b, k, o, out);
		long d = 0;
		for (long i = 0; i < k; i++) { d += b[i] != a[i]; high += a[i] >= 4096; }
		if (d && first < 0) for (long i = 0; i < k; i++) if (b[i] != a[i]) { first = done + i; break; }
		diff += d;
		done += k;
	}
	fclose(o);
	fclose(f);
	printf("ids32: %s %ld u16 ids -> %s %ld u32 ids (%lld B); re-read: %ld ids differ; %ld ids at 4,096 or above\n",
	       in, n, out, n, (long long)n * CX_TOK_BYTES, diff, high);
	if (diff) { printf("ids32: FAIL: the first differing id is at token %ld\n", first); return 1; }
	return 0;
}

/* the 68-byte record of an older one (rec 48 or 52 B at r) */
static void widen_record(const unsigned char *r, long rec, cx_soft *s)
{
	for (int j = 0; j < CX_SOFT_K; j++) {
		uint16_t id;
		memcpy(&id, r + 2 * j, 2);
		s->id[j] = id == OLD_NONE ? CX_SOFT_NONE : id;
	}
	memcpy(s->p, r + OLD_P, sizeof s->p);
	if (rec == 52) memcpy(&s->w, r + OLD_W, sizeof s->w);
	else s->w = 1.0f;
}

/* 0 when s is exactly the widened r */
static int record_differs(const unsigned char *r, long rec, const cx_soft *s)
{
	int d = 0;
	for (int j = 0; j < CX_SOFT_K; j++) {
		uint16_t id;
		memcpy(&id, r + 2 * j, 2);
		d |= s->id[j] != (id == OLD_NONE ? CX_SOFT_NONE : (cx_tok)id);
	}
	d |= memcmp(s->p, r + OLD_P, sizeof s->p) != 0;
	uint32_t w;
	memcpy(&w, &s->w, 4);
	if (rec == 52) d |= memcmp(&w, r + OLD_W, 4) != 0;
	else d |= w != F32_ONE;
	return d;
}

/* 1 when s falls outside the bounds every writer of the older records keeps: an id that is neither CX_SOFT_NONE nor
 * below 4,096; a p outside [0, 1] or not a number (soft-align writes lambda * ps / kept with lambda in [0, 1] and
 * ps <= kept, every unused slot and every empty record p = 0); a w that is not finite (all-ones exponent bits). One
 * record size read as the other puts a neighbour's w, p or 0xFFFF ids into these fields. */
static int implausible(const cx_soft *s)
{
	int bad = 0;
	for (int j = 0; j < CX_SOFT_K; j++) {
		bad |= s->id[j] != CX_SOFT_NONE && s->id[j] >= 4096;
		bad |= !(s->p[j] >= 0.0f && s->p[j] <= 1.0f);
	}
	uint32_t w;
	memcpy(&w, &s->w, 4);
	bad |= (w & 0x7F800000u) == 0x7F800000u;
	return bad;
}

static int widen_soft(const char *in, const char *out, long rec)
{
	struct stat st;
	FILE *f = open_in(in, &st);
	if (st.st_size % rec) { fprintf(stderr, "ids32: %s is %lld B, not a whole number of %ld-byte records\n", in, (long long)st.st_size, rec); return 1; }
	not_same(in, &st, out);
	const long n = (long)(st.st_size / rec);
	unsigned char *a = malloc((size_t)BLK * (size_t)rec);
	cx_soft *b = malloc(BLK * sizeof *b);
	FILE *o = fopen(out, "wb");
	if (!a || !b || !o) { perror(out); return 1; }
	for (long done = 0; done < n;) {
		const long k = n - done < BLK ? n - done : BLK;
		read_all(a, (size_t)rec, k, f, in);
		for (long i = 0; i < k; i++) widen_record(a + i * rec, rec, &b[i]);
		if (fwrite(b, sizeof *b, (size_t)k, o) != (size_t)k) { perror(out); return 1; }
		done += k;
	}
	if (fclose(o)) { perror(out); return 1; }
	/* the check: OUT re-read against IN, every id, p and w, and every record against the writers' bounds */
	rewind(f);
	o = reopen_out(out, (long long)n * CX_SOFT_BYTES);
	long diff = 0, first = -1, none = 0, dist = 0, high = 0, bad = 0, bad_first = -1;
	for (long done = 0; done < n;) {
		const long k = n - done < BLK ? n - done : BLK;
		read_all(a, (size_t)rec, k, f, in);
		read_all(b, sizeof *b, k, o, out);
		for (long i = 0; i < k; i++) {
			const int d = record_differs(a + i * rec, rec, &b[i]);
			if (d && first < 0) first = done + i;
			diff += d;
			int used = 0;
			for (int j = 0; j < CX_SOFT_K; j++) {
				const cx_tok id = b[i].id[j];
				none += id == CX_SOFT_NONE;
				used |= id != CX_SOFT_NONE;
				high += id != CX_SOFT_NONE && id >= 4096;
			}
			dist += used;
			const int out_of_bounds = implausible(&b[i]);
			if (out_of_bounds && bad_first < 0) bad_first = done + i;
			bad += out_of_bounds;
		}
		done += k;
	}
	fclose(o);
	fclose(f);
	printf("ids32: %s %ld records of %ld B -> %s %ld records of %u B (%lld B); re-read: %ld records differ; %ld records with a distribution, "
	       "%ld unused ids (CX_SOFT_NONE), %ld ids at 4,096 or above, %ld records outside the bounds; w %s\n",
	       in, n, rec, out, n, CX_SOFT_BYTES, (long long)n * CX_SOFT_BYTES, diff, dist, none, high, bad, rec == 52 ? "copied" : "1.0 in every record");
	if (diff) { printf("ids32: FAIL: the first differing record is record %ld\n", first); return 1; }
	if (bad) {
		printf("ids32: FAIL: %ld records (the first is record %ld) hold an id outside the vocabulary or a p or w that is not a probability or a "
		       "finite weight: %s is not %ld-byte records (--rec?)\n", bad, bad_first, in, rec);
		return 1;
	}
	return 0;
}

int main(int argc, char **argv)
{
	if (argc == 3 && strcmp(argv[1], "--soft")) return widen_stream(argv[1], argv[2]);
	if (argc == 6 && !strcmp(argv[1], "--soft") && !strcmp(argv[4], "--rec")) {
		const long rec = atol(argv[5]);
		if (rec != 48 && rec != 52) { fprintf(stderr, "ids32: --rec 48 or 52\n"); return 2; }
		return widen_soft(argv[2], argv[3], rec);
	}
	fprintf(stderr, "usage: ids32 IN.u16 OUT.u32 | ids32 --soft IN.soft OUT.soft --rec 48|52\n");
	return 2;
}
