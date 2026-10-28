/*
 * conv-pos: where a run that consumed the first NTOK tokens of OLD.u32 continues in NEW.u32, a re-encoding of the
 * same conversations in the same order (another format or vocabulary): the conversations wholly consumed are
 * counted by their end marker (id EOT, default 4095 as in the chat vocabularies), the same count is walked in
 * NEW.u32, and the chunk of 32 tokens holding the next conversation's first token is printed as the coach's --start
 * (a partly consumed conversation is seen again from its start).
 *   conv-pos OLD.u32 NTOK NEW.u32 [EOT]
 * Both streams hold u32 ids (lib/cx-proto.h cx_tok, 4 B a token); a .u16 stream of before 2026-10-28 is widened by
 * tools/ids32 first. A file that is not a whole number of ids, or holds an id at 4,096 or above, is refused: a .u16
 * stream read as u32 pairs two ids into one, lo | hi << 16.
 */
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include "../lib/cx-proto.h"

/* every id below 4,096: the OR of all ids has no bit at 12 or above, one vectorised pass, the position searched only
 * on a failure (2026-10-28, GCC 16.2.1 -O2 on the desktop, 32 Mi ids: 4.0 to 7.7 ms, the compare that stops at the
 * first 8.6 to 11.2 ms, 7 reps interleaved) */
static void ids_check(const char *path, const cx_tok *t, long n)
{
	cx_tok m = 0;
	for (long i = 0; i < n; i++) m |= t[i];
	if (!(m >> 12)) return;
	long i = 0;
	while (t[i] < 4096) i++;
	fprintf(stderr, "conv-pos: %s: id %u at token %ld is outside the 4,096-id vocabulary (a .u16 stream? tools/ids32 widens one)\n", path, t[i], i);
	exit(1);
}

static const cx_tok *map_ids(const char *path, long *n)
{
	const int fd = open(path, O_RDONLY);
	struct stat st;
	if (fd < 0 || fstat(fd, &st)) { perror(path); exit(1); }
	if (st.st_size % CX_TOK_BYTES) { fprintf(stderr, "conv-pos: %s is %lld B, not a whole number of u32 ids\n", path, (long long)st.st_size); exit(1); }
	*n = st.st_size / CX_TOK_BYTES;
	const cx_tok *p = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
	if (p == MAP_FAILED) { perror(path); exit(1); }
	close(fd);
	ids_check(path, p, *n);
	return p;
}

int main(int argc, char **argv)
{
	if (argc < 4) { fprintf(stderr, "usage: conv-pos OLD.u32 NTOK NEW.u32 [EOT]\n"); return 2; }
	const long ntok = atol(argv[2]);
	const cx_tok eot = argc > 4 ? (cx_tok)atoi(argv[4]) : 4095;
	long nold, nnew;
	const cx_tok *old = map_ids(argv[1], &nold), *nw = map_ids(argv[3], &nnew);
	if (ntok > nold) { fprintf(stderr, "conv-pos: %s holds %ld tokens, fewer than %ld\n", argv[1], nold, ntok); return 1; }
	long convs = 0;
	for (long i = 0; i < ntok; i++) convs += old[i] == eot;
	long seen = 0, pos = convs ? -1 : 0;
	for (long i = 0; i < nnew && pos < 0; i++) if (nw[i] == eot && ++seen == convs) pos = i + 1;
	if (pos < 0) { fprintf(stderr, "conv-pos: %s holds fewer than %ld conversations\n", argv[3], convs); return 1; }
	printf("conv-pos: %s[0..%ld) holds %ld whole conversations; in %s conversation %ld starts after token %ld of %ld; --start %ld (chunks of 32)\n",
	       argv[1], ntok, convs, argv[3], convs + 1, pos, nnew, pos / 32);
	return 0;
}
