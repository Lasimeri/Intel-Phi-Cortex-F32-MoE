/*
 * interleave: one token stream from several, in blocks of whole documents (runs that end in <|endoftext|>, id 4095).
 *   interleave OUT.u32 NTOK A.u32 B1.u32 [B2.u32 ...]
 * The output alternates a block of A with a block of one B file in turn (A, B1, A, B2, A, B3, ...). A block is the
 * shortest run of whole documents holding at least BLOCK tokens, so A keeps about half of the stream by tokens and
 * the B files share the other half about equally, whatever their document lengths. A document longer than DOC_MAX
 * tokens is cut there (with an end-of-text token) and its remainder is that source's next document. A source that
 * runs out starts over from its beginning. Prints the tokens and documents taken from each source.
 * Every stream holds u32 ids (lib/cx-proto.h cx_tok, 4 B a token); a .u16 source of before 2026-10-28 is widened
 * by tools/ids32 first. A source that is not a whole number of ids, or holds an id at 4,096 or above (a .u16 stream
 * read as u32 pairs two ids into one), is refused.
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

#define EOT 4095u
#define DOC_MAX 2048
#define BLOCK 2048

typedef struct { const char *name; const cx_tok *tok; long n, pos, taken, docs, wraps; } source;

/* every id below 4,096: the OR of all ids has no bit at 12 or above (one vectorised pass, as tools/conv-pos.c) */
static int ids_check(const char *path, const cx_tok *t, long n)
{
	cx_tok m = 0;
	for (long i = 0; i < n; i++) m |= t[i];
	if (!(m >> 12)) return 0;
	long i = 0;
	while (t[i] < 4096) i++;
	fprintf(stderr, "interleave: %s: id %u at token %ld is outside the 4,096-id vocabulary (a .u16 stream? tools/ids32 widens one)\n", path, t[i], i);
	return -1;
}

static int open_source(source *s, const char *path)
{
	s->name = path;
	const int fd = open(path, O_RDONLY);
	struct stat st;
	if (fd < 0 || fstat(fd, &st) || st.st_size < CX_TOK_BYTES) { perror(path); return -1; }
	if (st.st_size % CX_TOK_BYTES) { fprintf(stderr, "interleave: %s is %lld B, not a whole number of u32 ids\n", path, (long long)st.st_size); return -1; }
	s->n = st.st_size / CX_TOK_BYTES;
	s->tok = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
	close(fd);
	if (s->tok == MAP_FAILED) { perror(path); return -1; }
	return ids_check(path, s->tok, s->n);
}

/* appends one document of s to out (at most room tokens); returns the number written */
static long take_doc(source *s, cx_tok *out, long room)
{
	long w = 0;
	while (w < room) {
		if (s->pos >= s->n) { s->pos = 0; s->wraps++; }
		const cx_tok t = s->tok[s->pos++];
		out[w++] = t;
		if (t == EOT) break;
		if (w == DOC_MAX) { if (w < room) out[w++] = EOT; break; }
	}
	s->taken += w;
	s->docs++;
	return w;
}

/* appends whole documents of s until the block holds BLOCK tokens or the room is used; returns the number written */
static long take_block(source *s, cx_tok *out, long room)
{
	long w = 0;
	while (w < BLOCK && w < room) w += take_doc(s, out + w, room - w);
	return w;
}

int main(int argc, char **argv)
{
	if (argc < 5) { fprintf(stderr, "usage: interleave OUT.u32 NTOK A.u32 B1.u32 [B2.u32 ...]\n"); return 2; }
	const long ntok = atol(argv[2]);
	const int nb = argc - 4;
	source a, *b = calloc((size_t)nb, sizeof *b);
	if (open_source(&a, argv[3])) return 1;
	for (int i = 0; i < nb; i++) if (open_source(&b[i], argv[4 + i])) return 1;
	cx_tok *out = malloc((size_t)ntok * sizeof *out);
	long w = 0;
	for (int i = 0; w < ntok; i = (i + 1) % nb) {
		w += take_block(&a, out + w, ntok - w);
		if (w < ntok) w += take_block(&b[i], out + w, ntok - w);
	}
	FILE *f = fopen(argv[1], "wb");
	if (!f || fwrite(out, sizeof *out, (size_t)w, f) != (size_t)w || fclose(f)) { perror(argv[1]); return 1; }
	printf("interleave: %ld tokens -> %s (blocks of %d tokens, documents cut at %d)\n", w, argv[1], BLOCK, DOC_MAX);
	printf("  %12ld tokens %8ld documents %2ld wraps  %s\n", a.taken, a.docs, a.wraps, a.name);
	for (int i = 0; i < nb; i++) printf("  %12ld tokens %8ld documents %2ld wraps  %s\n", b[i].taken, b[i].docs, b[i].wraps, b[i].name);
	return 0;
}
