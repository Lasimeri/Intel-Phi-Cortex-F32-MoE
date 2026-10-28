/*
 * mixstream: a training stream that recurs a small set of conversations inside the base stream: one conversation of
 * SET.u16 (cycled), then RATIO whole conversations of BASE.u16 from START (moved up to the next conversation
 * boundary), again, until OUT.u16 holds at least CHUNKS x 32 tokens of whole conversations (the coach drops a
 * partial last chunk). The identity fine-tune loop (tools/identity-loop.sh) feeds the coach this stream, so the set
 * is seen every RATIO + 1 conversations while the base keeps flowing. A conversation ends at the end marker (id
 * EOT, default 4095); the newline token (id 10) after it opens the next one, and a set conversation cycled back to
 * the stream's first gets one too.
 *   mixstream SET.u16 BASE.u16 START RATIO CHUNKS OUT.u16 [EOT] [--soft BASE.soft OUT.soft]
 * --soft mirrors the stream in the soft-target records of the partial distillation (lib/cx-proto.h cx_soft, 48 B a
 * token, one per token of BASE.u16 in BASE.soft): every base token copied carries its record, every set token and
 * every inserted newline an empty one, so OUT.soft has one record per token of OUT.u16.
 * Prints the set's conversations and tokens and how often they recur, the base conversations and tokens used and
 * where they came from, the stream's length and the set's share of it.
 */
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include "../lib/cx-proto.h"

#define NL 10

static const void *map_file(const char *path, long *bytes)
{
	const int fd = open(path, O_RDONLY);
	struct stat st;
	if (fd < 0 || fstat(fd, &st)) { perror(path); exit(1); }
	*bytes = st.st_size;
	const void *p = st.st_size ? mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0) : NULL;
	if (st.st_size && p == MAP_FAILED) { perror(path); exit(1); }
	close(fd);
	return p;
}

/* the soft mirror: records of base tokens copied, empty records for the rest; nothing without --soft */
typedef struct { const cx_soft *base; FILE *out; long nbase; cx_soft empty; } mirror;

static void mirror_base(mirror *m, long from, long n)
{
	if (!m->out) return;
	if (from + n > m->nbase) { fprintf(stderr, "mixstream: the base's soft records end at %ld, token %ld asked\n", m->nbase, from + n); exit(1); }
	fwrite(m->base + from, sizeof *m->base, (size_t)n, m->out);
}

static void mirror_empty(mirror *m, long n)
{
	if (!m->out) return;
	for (long i = 0; i < n; i++) fwrite(&m->empty, sizeof m->empty, 1, m->out);
}

int main(int argc, char **argv)
{
	if (argc < 7) { fprintf(stderr, "usage: mixstream SET.u16 BASE.u16 START RATIO CHUNKS OUT.u16 [EOT] [--soft BASE.soft OUT.soft]\n"); return 2; }
	const long start = atol(argv[3]), chunks = atol(argv[5]);
	const int ratio = atoi(argv[4]);
	uint16_t eot = 4095;
	const char *soft_in = NULL, *soft_out = NULL;
	for (int i = 7; i < argc; i++) {
		if (!strcmp(argv[i], "--soft") && i + 2 < argc) { soft_in = argv[++i]; soft_out = argv[++i]; }
		else if (argv[i][0] >= '0' && argv[i][0] <= '9') eot = (uint16_t)atoi(argv[i]);
		else { fprintf(stderr, "mixstream: unknown argument %s\n", argv[i]); return 2; }
	}
	if (ratio < 0 || chunks <= 0) { fprintf(stderr, "mixstream: RATIO >= 0 and CHUNKS > 0\n"); return 2; }
	long nset, nbase, sbytes;
	const uint16_t *set = map_file(argv[1], &nset), *base = map_file(argv[2], &nbase);
	nset /= 2;
	nbase /= 2;
	mirror m = { 0 };
	for (int k = 0; k < CX_SOFT_K; k++) { m.empty.id[k] = (uint16_t)CX_SOFT_NONE; m.empty.p[k] = 0; }
	if (soft_in) {
		m.base = map_file(soft_in, &sbytes);
		m.nbase = sbytes / (long)sizeof *m.base;
		if (m.nbase < nbase) { fprintf(stderr, "mixstream: %s holds %ld records for %ld tokens of %s\n", soft_in, m.nbase, nbase, argv[2]); return 1; }
		m.out = fopen(soft_out, "wb");
		if (!m.out) { perror(soft_out); return 1; }
	}
	/* the set's conversations: [from, to] with the end marker at `to` */
	long *from = malloc((size_t)(nset + 1) * sizeof *from), *to = malloc((size_t)(nset + 1) * sizeof *to), nconv = 0, f = 0, set_tokens = 0;
	for (long i = 0; i < nset; i++) if (set[i] == eot) { from[nconv] = f; to[nconv] = i; set_tokens += i - f + 1; nconv++; f = i + 1; }
	if (!nconv) { fprintf(stderr, "mixstream: %s holds no conversation (no id %u)\n", argv[1], eot); return 1; }
	/* the base pointer: at a conversation boundary at or after START */
	long p = start < nbase ? start : nbase;
	while (p > 0 && p < nbase && base[p - 1] != eot) p++;
	const long base_from = p;
	FILE *out = fopen(argv[6], "wb");
	if (!out) { perror(argv[6]); return 1; }
	const long need = chunks * 32;
	long written = 0, set_convs = 0, base_convs = 0, base_tokens = 0, k = 0;
	const uint16_t nl = NL;
	while (written < need && p < nbase) {
		const long a = from[k], b = to[k];
		if (written > 0 && set[a] != NL) { fwrite(&nl, sizeof nl, 1, out); mirror_empty(&m, 1); written++; }
		fwrite(set + a, sizeof *set, (size_t)(b - a + 1), out);
		mirror_empty(&m, b - a + 1);
		written += b - a + 1;
		set_convs++;
		k = (k + 1) % nconv;
		for (int r = 0; r < ratio && p < nbase; r++) {
			long q = p;
			while (q < nbase && base[q] != eot) q++;
			if (q >= nbase) { p = nbase; break; }   /* no whole conversation left */
			fwrite(base + p, sizeof *base, (size_t)(q - p + 1), out);
			mirror_base(&m, p, q - p + 1);
			written += q - p + 1;
			base_tokens += q - p + 1;
			base_convs++;
			p = q + 1;
		}
	}
	if (fclose(out)) { perror(argv[6]); return 1; }
	if (m.out && fclose(m.out)) { perror(soft_out); return 1; }
	printf("mixstream: %ld set conversations (%ld tokens) recur every %d base conversations, seen %ld times each (%ld in all); %ld base conversations (%ld tokens) from token %ld to %ld of %s; %s: %ld tokens = %ld chunks, the set %.1f %% of them%s%s\n",
	       nconv, set_tokens, ratio, set_convs / nconv, set_convs, base_convs, base_tokens, base_from, p, argv[2], argv[6], written, written / 32, 100.0 * (written - base_tokens) / written,
	       soft_out ? "; soft records mirrored into " : "", soft_out ? soft_out : "");
	return 0;
}
