/* json-objects.c: every complete JSON object that opens at nesting level N (the count of open containers after
 * its brace), one a line, out of a text that may stop anywhere. The corpus writer (tools/teacher-corpus.sh) runs
 * the teacher's reply through it: the teacher ends a reply early now and then (an end token after a conversation
 * and a comma, 2026-10-27 13:4x), and the whole conversations before the cut are kept instead of the whole reply
 * being thrown away. In {"conversations": [{...}, {...}]} the conversations are level 3 (the default). Strings
 * and their escapes are honoured, so a brace inside a string counts for nothing; a raw newline anywhere becomes a
 * space, since valid JSON has none and the output is one object a line; text before the first brace is skipped.
 * Exit 0 when at least one object came out, 1 otherwise.
 *   json-objects [N] < TEXT */
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv)
{
	int want = argc > 1 ? atoi(argv[1]) : 3;
	int depth = 0, instr = 0, esc = 0, collecting = 0, c;
	size_t cap = 1 << 16, len = 0;
	long n = 0;
	char *buf = malloc(cap);
	if (!buf || want < 1) return 2;
	while ((c = getchar()) != EOF) {
		if (c == '\n' || c == '\r') c = ' ';
		if (collecting) {
			if (len + 2 >= cap) { cap *= 2; buf = realloc(buf, cap); if (!buf) return 2; }
			buf[len++] = (char)c;
		}
		if (instr) {
			if (esc) esc = 0;
			else if (c == '\\') esc = 1;
			else if (c == '"') instr = 0;
			continue;
		}
		if (c == '"') { instr = 1; continue; }
		if (c == '{' || c == '[') {
			depth++;
			if (c == '{' && depth == want && !collecting) { collecting = 1; len = 0; buf[len++] = '{'; }
		} else if (c == '}' || c == ']') {
			if (collecting && c == '}' && depth == want) {
				buf[len] = 0;
				puts(buf);
				n++;
				collecting = 0;
				len = 0;
			}
			if (depth > 0) depth--;
		}
	}
	free(buf);
	return n > 0 ? 0 : 1;
}
