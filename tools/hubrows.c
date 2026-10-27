/*
 * hubrows: the Hub's datasets-server rows pages to JSONL (host, C). usage: hubrows PAGE.json ... >> OUT.jsonl
 * Each page (https://datasets-server.huggingface.co/rows?dataset=...&config=...&split=...&offset=N&length=100) holds
 * "rows": [{"row_idx": i, "row": {"messages": "<JSON text, escaped>"}, "truncated_cells": [...]}, ...]. For every
 * row whose "messages" cell is a string, the string is decoded and written as the value of a "messages" key on one
 * line: the cell's own text is a JSON array of {"role", "content"} objects, so the line is the shape
 * tools/chat-text.c reads. A row whose cell is not a string, or a line the server truncated (listed in
 * "truncated_cells"), is skipped and counted. Prints rows written and skipped.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int hexv(int c) { return c >= '0' && c <= '9' ? c - '0' : c >= 'a' && c <= 'f' ? c - 'a' + 10 : c >= 'A' && c <= 'F' ? c - 'A' + 10 : -1; }

static int put_utf8(unsigned char *o, unsigned cp)
{
	if (cp < 0x80) { o[0] = (unsigned char)cp; return 1; }
	if (cp < 0x800) { o[0] = 0xC0 | cp >> 6; o[1] = 0x80 | (cp & 0x3F); return 2; }
	if (cp < 0x10000) { o[0] = 0xE0 | cp >> 12; o[1] = 0x80 | (cp >> 6 & 0x3F); o[2] = 0x80 | (cp & 0x3F); return 3; }
	o[0] = 0xF0 | cp >> 18; o[1] = 0x80 | (cp >> 12 & 0x3F); o[2] = 0x80 | (cp >> 6 & 0x3F); o[3] = 0x80 | (cp & 0x3F);
	return 4;
}

/* the JSON string starting at *p (its opening quote) decoded into out; the pointer past the closing quote; -1 if bad */
static long decode(const char **p, unsigned char *out, long cap)
{
	const char *s = *p;
	if (*s != '"') return -1;
	s++;
	long n = 0;
	while (*s && *s != '"') {
		if (*s != '\\') { if (n + 1 < cap) out[n++] = (unsigned char)*s; s++; continue; }
		s++;
		switch (*s) {
		case 'n': if (n + 1 < cap) out[n++] = '\n'; s++; break;
		case 't': if (n + 1 < cap) out[n++] = '\t'; s++; break;
		case 'r': if (n + 1 < cap) out[n++] = '\r'; s++; break;
		case 'b': if (n + 1 < cap) out[n++] = '\b'; s++; break;
		case 'f': if (n + 1 < cap) out[n++] = '\f'; s++; break;
		case '"': case '\\': case '/': if (n + 1 < cap) out[n++] = (unsigned char)*s; s++; break;
		case 'u': {
			unsigned cp = 0;
			for (int i = 1; i <= 4; i++) { const int v = hexv(s[i]); if (v < 0) return -1; cp = cp << 4 | (unsigned)v; }
			s += 5;
			if (cp >= 0xD800 && cp < 0xDC00 && s[0] == '\\' && s[1] == 'u') {
				unsigned lo = 0;
				for (int i = 2; i <= 5; i++) { const int v = hexv(s[i]); if (v < 0) return -1; lo = lo << 4 | (unsigned)v; }
				s += 6;
				cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
			}
			if (n + 4 < cap) n += put_utf8(out + n, cp);
			break;
		}
		default: return -1;
		}
	}
	if (*s != '"') return -1;
	*p = s + 1;
	out[n] = 0;
	return n;
}

int main(int argc, char **argv)
{
	long written = 0, skipped = 0;
	const long cap = 4 << 20;
	unsigned char *buf = malloc(cap);
	for (int a = 1; a < argc; a++) {
		FILE *f = fopen(argv[a], "rb");
		if (!f) { perror(argv[a]); return 1; }
		fseek(f, 0, SEEK_END);
		const long len = ftell(f);
		fseek(f, 0, SEEK_SET);
		char *page = malloc((size_t)len + 1);
		if (fread(page, 1, (size_t)len, f) != (size_t)len) { perror(argv[a]); return 1; }
		page[len] = 0;
		fclose(f);
		const char *p = page;
		while ((p = strstr(p, "\"row\":")) != NULL) {
			const char *cell = strstr(p, "\"messages\":");
			if (!cell) break;
			const char *v = cell + 11;
			while (*v == ' ') v++;
			const char *trunc = strstr(p, "\"truncated_cells\":");
			const int truncated = trunc && strstr(trunc, "\"messages\"") && (strchr(trunc, ']') > strstr(trunc, "\"messages\""));
			if (*v != '"' || decode(&v, buf, cap) < 0 || truncated) { skipped++; p = v > p ? v : p + 6; continue; }
			printf("{\"messages\": %s}\n", buf);
			written++;
			p = v;
		}
		free(page);
	}
	fprintf(stderr, "hubrows: %ld rows written, %ld skipped (not a string, undecodable, or truncated by the server)\n", written, skipped);
	return 0;
}
