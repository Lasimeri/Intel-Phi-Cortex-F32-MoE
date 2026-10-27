/*
 * mixstream: a training stream that recurs a small set of conversations inside the base stream: one conversation of
 * SET.u16 (cycled), then RATIO whole conversations of BASE.u16 from START (moved up to the next conversation
 * boundary), again, until OUT.u16 holds at least CHUNKS x 32 tokens of whole conversations (the coach drops a
 * partial last chunk). The identity fine-tune loop (tools/identity-loop.sh) feeds the coach this stream, so the set
 * is seen every RATIO + 1 conversations while the base keeps flowing. A conversation ends at the end marker (id
 * EOT, default 4095); the newline token (id 10) after it opens the next one, and a set conversation cycled back to
 * the stream's first gets one too.
 *   mixstream SET.u16 BASE.u16 START RATIO CHUNKS OUT.u16 [EOT]
 * Prints the set's conversations and tokens and how often they recur, the base conversations and tokens used and
 * where they came from, the stream's length and the set's share of it.
 */
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#define NL 10

static const uint16_t *map_u16(const char *path, long *n)
{
	const int fd = open(path, O_RDONLY);
	struct stat st;
	if (fd < 0 || fstat(fd, &st)) { perror(path); exit(1); }
	*n = st.st_size / 2;
	const uint16_t *p = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
	if (p == MAP_FAILED) { perror(path); exit(1); }
	close(fd);
	return p;
}

int main(int argc, char **argv)
{
	if (argc < 7) { fprintf(stderr, "usage: mixstream SET.u16 BASE.u16 START RATIO CHUNKS OUT.u16 [EOT]\n"); return 2; }
	const long start = atol(argv[3]), chunks = atol(argv[5]);
	const int ratio = atoi(argv[4]);
	const uint16_t eot = argc > 7 ? (uint16_t)atoi(argv[7]) : 4095;
	if (ratio < 0 || chunks <= 0) { fprintf(stderr, "mixstream: RATIO >= 0 and CHUNKS > 0\n"); return 2; }
	long nset, nbase;
	const uint16_t *set = map_u16(argv[1], &nset), *base = map_u16(argv[2], &nbase);
	/* the set's conversations: [from, to] with the end marker at `to` */
	long *from = malloc(nset * sizeof *from), *to = malloc(nset * sizeof *to), nconv = 0, f = 0, set_tokens = 0;
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
		if (written > 0 && set[a] != NL) { fwrite(&nl, sizeof nl, 1, out); written++; }
		fwrite(set + a, sizeof *set, b - a + 1, out);
		written += b - a + 1;
		set_convs++;
		k = (k + 1) % nconv;
		for (int r = 0; r < ratio && p < nbase; r++) {
			long q = p;
			while (q < nbase && base[q] != eot) q++;
			if (q >= nbase) { p = nbase; break; }   /* no whole conversation left */
			fwrite(base + p, sizeof *base, q - p + 1, out);
			written += q - p + 1;
			base_tokens += q - p + 1;
			base_convs++;
			p = q + 1;
		}
	}
	if (fclose(out)) { perror(argv[6]); return 1; }
	printf("mixstream: %ld set conversations (%ld tokens) recur every %d base conversations, seen %ld times each (%ld in all); %ld base conversations (%ld tokens) from token %ld to %ld of %s; %s: %ld tokens = %ld chunks, the set %.1f %% of them\n",
	       nconv, set_tokens, ratio, set_convs / nconv, set_convs, base_convs, base_tokens, base_from, p, argv[2], argv[6], written, written / 32, 100.0 * (written - base_tokens) / written);
	return 0;
}
