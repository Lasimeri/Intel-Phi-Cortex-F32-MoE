/*
 * conv-pos: where a run that consumed the first NTOK tokens of OLD.u16 continues in NEW.u16, a re-encoding of the
 * same conversations in the same order (another format or vocabulary): the conversations wholly consumed are
 * counted by their end marker (id EOT, default 4095 as in the chat vocabularies), the same count is walked in
 * NEW.u16, and the chunk of 32 tokens holding the next conversation's first token is printed as the coach's --start
 * (a partly consumed conversation is seen again from its start).
 *   conv-pos OLD.u16 NTOK NEW.u16 [EOT]
 */
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

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
	if (argc < 4) { fprintf(stderr, "usage: conv-pos OLD.u16 NTOK NEW.u16 [EOT]\n"); return 2; }
	const long ntok = atol(argv[2]);
	const uint16_t eot = argc > 4 ? (uint16_t)atoi(argv[4]) : 4095;
	long nold, nnew;
	const uint16_t *old = map_u16(argv[1], &nold), *nw = map_u16(argv[3], &nnew);
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
