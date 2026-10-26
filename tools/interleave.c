/*
 * interleave: one token stream from several, in blocks of whole documents (runs that end in <|endoftext|>, id 4095).
 *   interleave OUT.u16 NTOK A.u16 B1.u16 [B2.u16 ...]
 * The output alternates a block of A with a block of one B file in turn (A, B1, A, B2, A, B3, ...). A block is the
 * shortest run of whole documents holding at least BLOCK tokens, so A keeps about half of the stream by tokens and
 * the B files share the other half about equally, whatever their document lengths. A document longer than DOC_MAX
 * tokens is cut there (with an end-of-text token) and its remainder is that source's next document. A source that
 * runs out starts over from its beginning. Prints the tokens and documents taken from each source.
 */
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#define EOT 4095u
#define DOC_MAX 2048
#define BLOCK 2048

typedef struct { const char *name; const uint16_t *tok; long n, pos, taken, docs, wraps; } source;

static int open_source(source *s, const char *path)
{
	s->name = path;
	const int fd = open(path, O_RDONLY);
	struct stat st;
	if (fd < 0 || fstat(fd, &st) || st.st_size < 2) { perror(path); return -1; }
	s->n = st.st_size / 2;
	s->tok = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
	close(fd);
	if (s->tok == MAP_FAILED) { perror(path); return -1; }
	return 0;
}

/* appends one document of s to out (at most room tokens); returns the number written */
static long take_doc(source *s, uint16_t *out, long room)
{
	long w = 0;
	while (w < room) {
		if (s->pos >= s->n) { s->pos = 0; s->wraps++; }
		const uint16_t t = s->tok[s->pos++];
		out[w++] = t;
		if (t == EOT) break;
		if (w == DOC_MAX) { if (w < room) out[w++] = EOT; break; }
	}
	s->taken += w;
	s->docs++;
	return w;
}

/* appends whole documents of s until the block holds BLOCK tokens or the room is used; returns the number written */
static long take_block(source *s, uint16_t *out, long room)
{
	long w = 0;
	while (w < BLOCK && w < room) w += take_doc(s, out + w, room - w);
	return w;
}

int main(int argc, char **argv)
{
	if (argc < 5) { fprintf(stderr, "usage: interleave OUT.u16 NTOK A.u16 B1.u16 [B2.u16 ...]\n"); return 2; }
	const long ntok = atol(argv[2]);
	const int nb = argc - 4;
	source a, *b = calloc((size_t)nb, sizeof *b);
	if (open_source(&a, argv[3])) return 1;
	for (int i = 0; i < nb; i++) if (open_source(&b[i], argv[4 + i])) return 1;
	uint16_t *out = malloc((size_t)ntok * 2);
	long w = 0;
	for (int i = 0; w < ntok; i = (i + 1) % nb) {
		w += take_block(&a, out + w, ntok - w);
		if (w < ntok) w += take_block(&b[i], out + w, ntok - w);
	}
	FILE *f = fopen(argv[1], "wb");
	if (!f || fwrite(out, 2, (size_t)w, f) != (size_t)w || fclose(f)) { perror(argv[1]); return 1; }
	printf("interleave: %ld tokens -> %s (blocks of %d tokens, documents cut at %d)\n", w, argv[1], BLOCK, DOC_MAX);
	printf("  %12ld tokens %8ld documents %2ld wraps  %s\n", a.taken, a.docs, a.wraps, a.name);
	for (int i = 0; i < nb; i++) printf("  %12ld tokens %8ld documents %2ld wraps  %s\n", b[i].taken, b[i].docs, b[i].wraps, b[i].name);
	return 0;
}
