/*
 * textprep: BabyLM 2024 text files -> the project's text convention: plain lines, documents separated by a line
 * holding "<|endoftext|>" (tools/bpe.c turns it into token 4095, as in the TinyStories text).
 *   textprep KIND IN OUT
 *   KIND  wiki  "= = = Title = = =" starts a document (the heading line itself is dropped); blank lines are dropped
 *         book  "= = = PGnnnnn = = =" starts a document; a document is also cut at a blank line once it holds
 *               BOOK_WORDS words, so no document is longer than a few thousand words; lines without a letter or a
 *               digit (rules of underscores, stray punctuation) are dropped
 *         talk  speaker prefixes ("*CHI:<tab>", "A:<tab>") are removed; a blank line ends a document; a document is
 *               cut after TALK_LINES lines
 *         lines lines are kept as they are; a blank line ends a document; a document is cut after TALK_LINES lines
 * Prints the document and line counts.
 */
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BOOK_WORDS 1500
#define TALK_LINES 48

static FILE *out;
static long docs, lines_out, doc_words, doc_lines;
static int doc_open;

static void boundary(void)
{
	if (!doc_open) return;
	fputs("<|endoftext|>\n", out);
	docs++;
	doc_open = 0;
	doc_words = doc_lines = 0;
}

static void emit(const char *s)
{
	fputs(s, out);
	fputc('\n', out);
	doc_open = 1;
	lines_out++;
	doc_lines++;
	int inword = 0;
	for (const char *p = s; *p; p++) {
		if (isspace((unsigned char)*p)) inword = 0;
		else if (!inword) { inword = 1; doc_words++; }
	}
}

static int has_alnum(const char *s)
{
	for (; *s; s++) if (isalnum((unsigned char)*s) || (unsigned char)*s >= 0x80) return 1;
	return 0;
}

/* the text after a speaker prefix, or the line itself */
static const char *strip_speaker(const char *s)
{
	const char *p = s;
	if (*p == '*') p++;
	const char *colon = p;
	while (isalnum((unsigned char)*colon)) colon++;
	if (colon > p && colon[0] == ':' && (colon[1] == '\t' || colon[1] == ' ')) {
		const char *t = colon + 1;
		while (*t == '\t' || *t == ' ') t++;
		return t;
	}
	return s;
}

int main(int argc, char **argv)
{
	if (argc != 4) { fprintf(stderr, "usage: textprep wiki|book|talk|lines IN OUT\n"); return 2; }
	const char *kind = argv[1];
	const int wiki = !strcmp(kind, "wiki"), book = !strcmp(kind, "book"), talk = !strcmp(kind, "talk"), plain = !strcmp(kind, "lines");
	if (!wiki && !book && !talk && !plain) { fprintf(stderr, "textprep: unknown kind %s\n", kind); return 2; }
	FILE *in = fopen(argv[2], "r");
	if (!in) { perror(argv[2]); return 1; }
	out = fopen(argv[3], "w");
	if (!out) { perror(argv[3]); return 1; }
	char *line = NULL;
	size_t cap = 0;
	ssize_t n;
	long lines_in = 0;
	while ((n = getline(&line, &cap, in)) > 0) {
		lines_in++;
		while (n > 0 && (line[n - 1] == '\n' || line[n - 1] == '\r')) line[--n] = 0;
		const int blank = n == 0 || !has_alnum(line);
		if ((wiki || book) && !strncmp(line, "= = = ", 6)) { boundary(); continue; }
		if (wiki) { if (!blank) emit(line); continue; }
		if (book) {
			if (blank) { if (doc_words >= BOOK_WORDS) boundary(); continue; }
			emit(line);
			continue;
		}
		if (blank) { boundary(); continue; }
		emit(talk ? strip_speaker(line) : line);
		if (doc_lines >= TALK_LINES) boundary();
	}
	boundary();
	if (fclose(out)) { perror(argv[3]); return 1; }
	printf("textprep %s: %s -> %s: %ld lines in, %ld lines out, %ld documents\n", kind, argv[2], argv[3], lines_in, lines_out, docs);
	return 0;
}
