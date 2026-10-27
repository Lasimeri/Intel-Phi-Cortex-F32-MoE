/*
 * chat-text: chat JSONL (Hub distillation sets) to the frame's dialogue text (host, C).
 *   chat-text IN.jsonl OUT.txt [MAX_TURN_CHARS] [MAX_NONASCII_PERMILLE]
 * Each input line is a JSON object holding a conversation as "messages" (objects with "role" and "content") or in the
 * ShareGPT shape ("conversations" with "from" and "value"). Roles user|human and assistant|gpt|model are the two
 * speakers; system, reasoning and thinking turns are left out. An assistant turn loses its <think>...</think> and
 * <thinking>...</thinking> blocks. A conversation is written as the harness's frame writes it (tui/cortex.jinja):
 *   Human: ...\nCortex: ...\n ... <|endoftext|>\n
 * so what the model learns is the shape it is prompted in. Typography is brought to ASCII first (the TinyStories BPE
 * spends three byte tokens on each curly quote): em and en dashes to " - " and "-", curly quotes to straight ones,
 * the ellipsis to "...", the no-break space to a space. A conversation is skipped whole when a turn is empty after
 * cleaning, holds a code fence (```), is longer than MAX_TURN_CHARS (default 6000), when the speakers do not
 * alternate from a user turn, or when the share of non-ASCII bytes is above MAX_NONASCII_PERMILLE (default 20 = 2 %).
 * Prints conversations kept and skipped by reason, exchanges, bytes written.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct { const char *p; } cur;   /* a cursor over one JSON line */

static int hexv(int c) { return c >= '0' && c <= '9' ? c - '0' : c >= 'a' && c <= 'f' ? c - 'a' + 10 : c >= 'A' && c <= 'F' ? c - 'A' + 10 : -1; }

static int put_utf8(unsigned char *o, unsigned cp)
{
	if (cp < 0x80) { o[0] = (unsigned char)cp; return 1; }
	if (cp < 0x800) { o[0] = 0xC0 | cp >> 6; o[1] = 0x80 | (cp & 0x3F); return 2; }
	if (cp < 0x10000) { o[0] = 0xE0 | cp >> 12; o[1] = 0x80 | (cp >> 6 & 0x3F); o[2] = 0x80 | (cp & 0x3F); return 3; }
	o[0] = 0xF0 | cp >> 18; o[1] = 0x80 | (cp >> 12 & 0x3F); o[2] = 0x80 | (cp >> 6 & 0x3F); o[3] = 0x80 | (cp & 0x3F);
	return 4;
}

static void ws(cur *c) { while (*c->p == ' ' || *c->p == '\t' || *c->p == '\r' || *c->p == '\n') c->p++; }

/* a JSON string at the cursor into out (cap bytes, NUL-terminated); its length, or -1; the cursor past the quote */
static long str(cur *c, unsigned char *out, long cap)
{
	if (*c->p != '"') return -1;
	c->p++;
	long n = 0;
	while (*c->p && *c->p != '"') {
		unsigned char ch = (unsigned char)*c->p;
		if (ch != '\\') { if (n + 1 < cap) out[n++] = ch; c->p++; continue; }
		c->p++;
		switch (*c->p) {
		case 'n': if (n + 1 < cap) out[n++] = '\n'; c->p++; break;
		case 't': if (n + 1 < cap) out[n++] = '\t'; c->p++; break;
		case 'r': if (n + 1 < cap) out[n++] = '\r'; c->p++; break;
		case 'b': if (n + 1 < cap) out[n++] = '\b'; c->p++; break;
		case 'f': if (n + 1 < cap) out[n++] = '\f'; c->p++; break;
		case '"': case '\\': case '/': if (n + 1 < cap) out[n++] = (unsigned char)*c->p; c->p++; break;
		case 'u': {
			unsigned cp = 0;
			for (int i = 1; i <= 4; i++) { const int v = hexv(c->p[i]); if (v < 0) return -1; cp = cp << 4 | (unsigned)v; }
			c->p += 5;
			if (cp >= 0xD800 && cp < 0xDC00 && c->p[0] == '\\' && c->p[1] == 'u') {   /* a surrogate pair */
				unsigned lo = 0;
				for (int i = 2; i <= 5; i++) { const int v = hexv(c->p[i]); if (v < 0) return -1; lo = lo << 4 | (unsigned)v; }
				c->p += 6;
				cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
			}
			if (n + 4 < cap) n += put_utf8(out + n, cp);
			break;
		}
		default: return -1;
		}
	}
	if (*c->p != '"') return -1;
	c->p++;
	out[n] = 0;
	return n;
}

/* skip any JSON value at the cursor; 0 when the line is malformed */
static int skip_value(cur *c)
{
	ws(c);
	if (*c->p == '"') { unsigned char tmp[8]; return str(c, tmp, sizeof tmp) >= 0; }
	if (*c->p == '{' || *c->p == '[') {
		const char open = *c->p, close = open == '{' ? '}' : ']';
		c->p++;
		for (;;) {
			ws(c);
			if (*c->p == close) { c->p++; return 1; }
			if (*c->p == ',') { c->p++; continue; }
			if (open == '{') {
				unsigned char key[256];
				if (str(c, key, sizeof key) < 0) return 0;
				ws(c);
				if (*c->p != ':') return 0;
				c->p++;
			}
			if (!skip_value(c)) return 0;
		}
	}
	while (*c->p && *c->p != ',' && *c->p != '}' && *c->p != ']') c->p++;   /* a number or a literal */
	return 1;
}

#define TURN_CAP (1 << 20)
#define MAX_TURNS 64

typedef struct { int role; unsigned char *text; long n; } turn;   /* role 0 user, 1 assistant, 2 other */

static int role_of(const unsigned char *r)
{
	if (!strcmp((const char *)r, "user") || !strcmp((const char *)r, "human")) return 0;
	if (!strcmp((const char *)r, "assistant") || !strcmp((const char *)r, "gpt") || !strcmp((const char *)r, "model")) return 1;
	return 2;
}

/* one message object at the cursor into t; 1 ok, 0 malformed */
static int message(cur *c, turn *t, unsigned char *scratch)
{
	ws(c);
	if (*c->p != '{') return 0;
	c->p++;
	t->role = 2;
	t->n = 0;
	t->text[0] = 0;
	for (;;) {
		ws(c);
		if (*c->p == '}') { c->p++; return 1; }
		if (*c->p == ',') { c->p++; continue; }
		unsigned char key[256];
		if (str(c, key, sizeof key) < 0) return 0;
		ws(c);
		if (*c->p != ':') return 0;
		c->p++;
		ws(c);
		if (!strcmp((const char *)key, "role") || !strcmp((const char *)key, "from")) {
			if (str(c, scratch, 256) < 0) return 0;
			t->role = role_of(scratch);
		} else if (!strcmp((const char *)key, "content") || !strcmp((const char *)key, "value")) {
			if (*c->p != '"') { if (!skip_value(c)) return 0; continue; }   /* a content list, not text: left empty */
			t->n = str(c, t->text, TURN_CAP);
			if (t->n < 0) return 0;
		} else if (!skip_value(c)) return 0;
	}
}

/* <tag>...</tag> blocks removed in place */
static void strip_block(unsigned char *s, const char *open, const char *close)
{
	for (;;) {
		char *a = strstr((char *)s, open);
		if (!a) return;
		char *b = strstr(a, close);
		if (!b) { *a = 0; return; }   /* an unclosed block runs to the end */
		memmove(a, b + strlen(close), strlen(b + strlen(close)) + 1);
	}
}

/* typography to ASCII, in place (no replacement is longer than what it replaces) */
static void normalise(unsigned char *s)
{
	static const struct { const char *from, *to; } map[] = {
		{ " \xe2\x80\x94 ", " - " }, { "\xe2\x80\x94", " - " }, { "\xe2\x80\x93", "-" }, { "\xe2\x80\x98", "'" }, { "\xe2\x80\x99", "'" },
		{ "\xe2\x80\x9c", "\"" }, { "\xe2\x80\x9d", "\"" }, { "\xe2\x80\xa6", "..." }, { "\xc2\xa0", " " },
	};
	long r = 0, w = 0;
	while (s[r]) {
		int done = 0;
		for (size_t k = 0; k < sizeof map / sizeof *map; k++) {
			const size_t n = strlen(map[k].from);
			if (!memcmp(s + r, map[k].from, n)) {
				memcpy(s + w, map[k].to, strlen(map[k].to));
				w += (long)strlen(map[k].to);
				r += (long)n;
				done = 1;
				break;
			}
		}
		if (!done) s[w++] = s[r++];
	}
	s[w] = 0;
}

static void trim(unsigned char *s)
{
	long n = strlen((char *)s), i = 0;
	while (n > 0 && (s[n - 1] == ' ' || s[n - 1] == '\n' || s[n - 1] == '\t' || s[n - 1] == '\r')) s[--n] = 0;
	while (s[i] == ' ' || s[i] == '\n' || s[i] == '\t' || s[i] == '\r') i++;
	if (i) memmove(s, s + i, n - i + 1);
}

int main(int argc, char **argv)
{
	if (argc < 3) { fprintf(stderr, "usage: chat-text IN.jsonl OUT.txt [MAX_TURN_CHARS] [MAX_NONASCII_PERMILLE]\n"); return 2; }
	const long max_turn = argc > 3 ? atol(argv[3]) : 6000, max_nonascii = argc > 4 ? atol(argv[4]) : 20;
	FILE *in = fopen(argv[1], "r"), *out = fopen(argv[2], "w");
	if (!in || !out) { perror(in ? argv[2] : argv[1]); return 1; }
	static char *line;
	static size_t cap;
	turn turns[MAX_TURNS];
	for (int i = 0; i < MAX_TURNS; i++) turns[i].text = malloc(TURN_CAP);
	unsigned char *scratch = malloc(TURN_CAP);
	long rows = 0, kept = 0, exchanges = 0, bytes = 0;
	long sk_shape = 0, sk_empty = 0, sk_code = 0, sk_long = 0, sk_order = 0, sk_nonascii = 0;
	ssize_t len;
	while ((len = getline(&line, &cap, in)) > 0) {
		if (len < 3) continue;
		rows++;
		cur c = { line };
		ws(&c);
		if (*c.p != '{') { sk_shape++; continue; }
		c.p++;
		int nt = 0, found = 0, bad = 0;
		for (;;) {   /* the object's keys: the conversation array taken, everything else skipped */
			ws(&c);
			if (*c.p == '}' || !*c.p) break;
			if (*c.p == ',') { c.p++; continue; }
			unsigned char key[256];
			if (str(&c, key, sizeof key) < 0) { bad = 1; break; }
			ws(&c);
			if (*c.p != ':') { bad = 1; break; }
			c.p++;
			ws(&c);
			if (!found && (!strcmp((const char *)key, "messages") || !strcmp((const char *)key, "conversations")) && *c.p == '[') {
				found = 1;
				c.p++;
				for (;;) {
					ws(&c);
					if (*c.p == ']') { c.p++; break; }
					if (*c.p == ',') { c.p++; continue; }
					if (nt == MAX_TURNS) { bad = 1; break; }
					if (!message(&c, &turns[nt], scratch)) { bad = 1; break; }
					nt++;
				}
				if (bad) break;
			} else if (!skip_value(&c)) { bad = 1; break; }
		}
		if (bad || !found || nt == 0) { sk_shape++; continue; }
		/* the speakers' turns, cleaned */
		int n = 0, expect = 0, reason = 0;
		turn *kept_turns[MAX_TURNS];
		for (int i = 0; i < nt && !reason; i++) {
			turn *t = &turns[i];
			if (t->role == 2) continue;
			if (t->role != expect) { reason = 4; break; }
			if (t->role == 1) { strip_block(t->text, "<think>", "</think>"); strip_block(t->text, "<thinking>", "</thinking>"); }
			normalise(t->text);
			trim(t->text);
			t->n = strlen((char *)t->text);
			if (t->n == 0) { reason = 1; break; }
			if (strstr((char *)t->text, "```")) { reason = 2; break; }
			if (t->n > max_turn) { reason = 3; break; }
			kept_turns[n++] = t;
			expect ^= 1;
		}
		if (!reason && (n < 2 || kept_turns[n - 1]->role != 1)) reason = 4;   /* ends on an answer */
		if (!reason) {
			long total = 0, nonascii = 0;
			for (int i = 0; i < n; i++) { total += kept_turns[i]->n; for (long k = 0; k < kept_turns[i]->n; k++) nonascii += kept_turns[i]->text[k] >= 0x80; }
			if (nonascii * 1000 > total * max_nonascii) reason = 5;
		}
		switch (reason) {
		case 1: sk_empty++; continue;
		case 2: sk_code++; continue;
		case 3: sk_long++; continue;
		case 4: sk_order++; continue;
		case 5: sk_nonascii++; continue;
		default: break;
		}
		for (int i = 0; i < n; i++) {
			bytes += fprintf(out, "%s %s\n", kept_turns[i]->role == 0 ? "Human:" : "Cortex:", kept_turns[i]->text);
			exchanges += kept_turns[i]->role == 1;
		}
		bytes += fprintf(out, "<|endoftext|>\n");
		kept++;
	}
	fclose(out);
	printf("chat-text: %ld rows: %ld conversations kept (%ld exchanges, %ld bytes); skipped %ld malformed, %ld empty turn, %ld code, %ld over %ld chars, %ld out of order, %ld non-ASCII\n",
	       rows, kept, exchanges, bytes, sk_shape, sk_empty, sk_code, sk_long, max_turn, sk_order, sk_nonascii);
	return 0;
}
