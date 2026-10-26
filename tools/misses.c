/*
 * misses: the words of a text that are not in the word list, in order, with the piece boundaries that produced them
 * (the token ids of data/cortex-last.u16 when given). usage: misses WORDLIST TEXT [IDS.u16 VOCAB]
 */
#define BPE_NO_MAIN
#include "bpe.c"
#include "coherence.h"

int main(int argc, char **argv)
{
	if (argc < 3) { fprintf(stderr, "usage: misses WORDLIST TEXT [IDS.u16 VOCAB]\n"); return 2; }
	if (coh_load(argv[1])) { perror(argv[1]); return 1; }
	FILE *f = fopen(argv[2], "rb");
	if (!f) { perror(argv[2]); return 1; }
	fseek(f, 0, SEEK_END);
	const long n = ftell(f);
	fseek(f, 0, SEEK_SET);
	char *text = malloc((size_t)n + 1);
	if (fread(text, 1, (size_t)n, f) != (size_t)n) return 1;
	fclose(f);
	text[n] = 0;
	const char *p = text, *end = text + n;
	char w[COH_WMAX];
	int se;
	long words = 0, miss = 0;
	while (coh_next(&p, end, w, &se)) {
		words++;
		if (coh_find(w, 0)) continue;
		miss++;
		const long at = p - text;
		printf("miss %3ld at byte %5ld: '%s'  context: ", miss, at, w);
		const long a = at > 40 ? at - 40 : 0, b = at + 10 < n ? at + 10 : n;
		for (long i = a; i < b; i++) putchar(text[i] == '\n' ? ' ' : text[i]);
		putchar('\n');
	}
	printf("misses: %ld of %ld words\n", miss, words);
	if (argc >= 5) {
		if (load_vocab(argv[4])) return 1;
		for (int id = 0; id < 256 + nmerges; id++) expand(id);
		expand(EOT);
		f = fopen(argv[3], "rb");
		if (!f) { perror(argv[3]); return 1; }
		uint16_t id;
		printf("pieces:");
		while (fread(&id, 2, 1, f) == 1) { printf(" ["); fwrite(tok_s[id], 1, (size_t)tok_len[id], stdout); printf("]"); }
		printf("\n");
		fclose(f);
	}
	return 0;
}
