/*
 * chat-text: chat JSONL (Hub distillation sets) to the Cortex chat format (host, C).
 *   chat-text IN.jsonl OUT.txt [--max-chars N] [--max-nonascii PERMILLE] [--no-think PERMILLE] [--drop-code]
 *             [--no-system] [--no-think-block] [--detok]
 * Each input line is a JSON object holding a conversation as "messages" (objects with "role" and "content"; an
 * assistant's reasoning in "reasoning_content" | "thinking" | "reasoning", or inline in the content as
 * <think>...</think> | <thinking>...</thinking>) or in the ShareGPT shape ("conversations" with "from" and "value").
 * Roles: system (the first message only), user|human, assistant|gpt|model; the speakers alternate from a user turn
 * and the conversation ends on an answer. A conversation is written in the format the model is trained in and
 * prompted in (tui/cortex.jinja), each marker one token of the chat BPE (tools/bpe.c):
 *   <|system|>TEXT<|end|>\n
 *   <|user|>TEXT<|end|>\n
 *   <|assistant|><|think|>REASONING<|/think|>ANSWER<|end|>\n
 *   ...
 *   <|endoftext|>\n
 * An answer without reasoning, and every answer of a conversation drawn into the no-think share (--no-think
 * PERMILLE, default 250; the draw is a hash of the first user turn, so it is stable), is written with an empty
 * think block, <|think|><|/think|>: the model's no-think form, as Qwen3's. Typography and common math symbols are
 * brought to ASCII first (dashes, quotes, the ellipsis, the no-break space, superscripts 2 and 3, the times and
 * division signs, the arrow, approximately, <= and >=, the minus, bullets, the zero-width space). Skipped whole: a
 * malformed line, an empty turn after cleaning, a text over --max-chars (default 16000) in any field, speakers out
 * of order, more than --max-nonascii permille of non-ASCII bytes (default 20), and with --drop-code a conversation
 * holding a code fence (```). --no-system leaves the system messages out. --no-think-block writes an answer that has
 * no reasoning as <|assistant|>ANSWER<|end|>, with no block at all (the plain chat format of the chat base,
 * 2026-10-27: a block that is always empty is two dead tokens per answer; an answer with reasoning keeps its block
 * unless the conversation is drawn no-think). --detok rejoins text that was tokenised with spaces (DailyDialog: "the
 * United States ." "I ' m" "don ' t" ". . ."): the space before , . ? ! ; : ) ] goes, the space after ( [ goes, the
 * contractions 's 't 'm 'd 're 've 'll are joined, ". . ." becomes "..."; a space before "..." stays. Prints what
 * was kept and skipped by reason.
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

enum { USER, ASSISTANT, OTHER, SYSTEM };

typedef struct { int role; unsigned char *text, *reason; long n, rn; } turn;

static int role_of(const unsigned char *r)
{
	if (!strcmp((const char *)r, "user") || !strcmp((const char *)r, "human")) return USER;
	if (!strcmp((const char *)r, "assistant") || !strcmp((const char *)r, "gpt") || !strcmp((const char *)r, "model")) return ASSISTANT;
	if (!strcmp((const char *)r, "system")) return SYSTEM;
	return OTHER;
}

/* one message object at the cursor into t; 1 ok, 0 malformed */
static int message(cur *c, turn *t, unsigned char *scratch)
{
	ws(c);
	if (*c->p != '{') return 0;
	c->p++;
	t->role = OTHER;
	t->n = t->rn = 0;
	t->text[0] = t->reason[0] = 0;
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
		const char *k = (const char *)key;
		if (!strcmp(k, "role") || !strcmp(k, "from")) {
			if (str(c, scratch, 256) < 0) return 0;
			t->role = role_of(scratch);
		} else if (!strcmp(k, "content") || !strcmp(k, "value")) {
			if (*c->p != '"') { if (!skip_value(c)) return 0; continue; }   /* a content list, not text: left empty */
			t->n = str(c, t->text, TURN_CAP);
			if (t->n < 0) return 0;
		} else if (!strcmp(k, "reasoning_content") || !strcmp(k, "thinking") || !strcmp(k, "reasoning")) {
			if (*c->p != '"') { if (!skip_value(c)) return 0; continue; }
			t->rn = str(c, t->reason, TURN_CAP);
			if (t->rn < 0) return 0;
		} else if (!skip_value(c)) return 0;
	}
}

/* <tag>...</tag> blocks moved out of s into `into` (blocks joined by a blank line); an unclosed block runs to the end */
static void extract_block(unsigned char *s, const char *open, const char *close, unsigned char *into)
{
	for (;;) {
		char *a = strstr((char *)s, open);
		if (!a) return;
		char *body = a + strlen(open);
		char *b = strstr(body, close);
		const long blen = b ? b - body : (long)strlen(body);
		long at = strlen((char *)into);
		if (at && at + 2 < TURN_CAP) { memcpy(into + at, "\n\n", 2); at += 2; }
		if (at + blen + 1 < TURN_CAP) { memcpy(into + at, body, (size_t)blen); at += blen; }
		into[at] = 0;
		if (!b) { *a = 0; return; }
		memmove(a, b + strlen(close), strlen(b + strlen(close)) + 1);
	}
}

/* typography and common math symbols to ASCII, in place (no replacement is longer than what it replaces) */
static void normalise(unsigned char *s)
{
	static const struct { const char *from, *to; } map[] = {
		{ " \xe2\x80\x94 ", " - " }, { "\xe2\x80\x94", " - " }, { "\xe2\x80\x93", "-" }, { "\xe2\x80\x90", "-" }, { "\xe2\x80\x91", "-" },
		{ "\xe2\x80\x98", "'" }, { "\xe2\x80\x99", "'" }, { "\xe2\x80\x9c", "\"" }, { "\xe2\x80\x9d", "\"" }, { "\xe2\x80\xa6", "..." },
		{ "\xc2\xa0", " " }, { "\xe2\x80\x8b", "" }, { "\xc2\xb2", "^2" }, { "\xc2\xb3", "^3" }, { "\xc3\x97", "x" }, { "\xc3\xb7", "/" },
		{ "\xe2\x86\x92", "->" }, { "\xe2\x89\x88", "~" }, { "\xe2\x89\xa4", "<=" }, { "\xe2\x89\xa5", ">=" }, { "\xe2\x88\x92", "-" },
		{ "\xe2\x80\xa2", "-" },
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

static uint64_t fnv(const unsigned char *s, long n)
{
	uint64_t h = 1469598103934665603ull;
	for (long i = 0; i < n; i++) h = (h ^ s[i]) * 1099511628211ull;
	return h;
}

static int letter(unsigned char c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'); }

/* text tokenised with spaces back to writing, in place (never longer): ". . ." -> "...", no space before closing
 * punctuation (a space before "..." stays), none after an opening bracket, " ' s" and the other contractions joined
 * when the suffix ends the word ("don ' t have" -> "don't have", "James ' car" stays) */
static void detok(unsigned char *s)
{
	static const char *const suffix[] = { "s", "t", "m", "d", "re", "ve", "ll" };
	long r = 0, w = 0;
	while (s[r]) {
		if (s[r] == '.' && !strncmp((char *)s + r, ". . .", 5)) { memcpy(s + w, "...", 3); w += 3; r += 5; continue; }
		if (s[r] == ' ' && s[r + 1] && strchr(",.?!;:)]", s[r + 1]) && !(s[r + 1] == '.' && s[r + 2] == '.')) { r++; continue; }
		if ((s[r] == '(' || s[r] == '[') && s[r + 1] == ' ') { s[w++] = s[r]; r += 2; continue; }
		if (s[r] == ' ' && s[r + 1] == '\'' && s[r + 2] == ' ') {
			int joined = 0;
			for (size_t k = 0; k < sizeof suffix / sizeof *suffix && !joined; k++) {
				const size_t n = strlen(suffix[k]);
				if (!strncmp((char *)s + r + 3, suffix[k], n) && !letter(s[r + 3 + n])) {
					s[w++] = '\'';
					memcpy(s + w, suffix[k], n);
					w += (long)n;
					r += 3 + (long)n;
					joined = 1;
				}
			}
			if (joined) continue;
		}
		s[w++] = s[r++];
	}
	s[w] = 0;
}

int main(int argc, char **argv)
{
	if (argc < 3) {
		fprintf(stderr, "usage: chat-text IN.jsonl OUT.txt [--max-chars N] [--max-nonascii PERMILLE] [--no-think PERMILLE] [--drop-code] [--no-system] [--no-think-block] [--detok]\n");
		return 2;
	}
	long max_chars = 16000, max_nonascii = 20, no_think = 250;
	int drop_code = 0, no_system = 0, no_block = 0, do_detok = 0;
	for (int i = 3; i < argc; i++) {
		if (!strcmp(argv[i], "--max-chars") && i + 1 < argc) max_chars = atol(argv[++i]);
		else if (!strcmp(argv[i], "--max-nonascii") && i + 1 < argc) max_nonascii = atol(argv[++i]);
		else if (!strcmp(argv[i], "--no-think") && i + 1 < argc) no_think = atol(argv[++i]);
		else if (!strcmp(argv[i], "--drop-code")) drop_code = 1;
		else if (!strcmp(argv[i], "--no-system")) no_system = 1;
		else if (!strcmp(argv[i], "--no-think-block")) no_block = 1;
		else if (!strcmp(argv[i], "--detok")) do_detok = 1;
		else { fprintf(stderr, "chat-text: unknown option %s\n", argv[i]); return 2; }
	}
	FILE *in = fopen(argv[1], "r"), *out = fopen(argv[2], "w");
	if (!in || !out) { perror(in ? argv[2] : argv[1]); return 1; }
	static char *line;
	static size_t cap;
	turn turns[MAX_TURNS];
	for (int i = 0; i < MAX_TURNS; i++) { turns[i].text = malloc(TURN_CAP); turns[i].reason = malloc(TURN_CAP); }
	unsigned char *scratch = malloc(TURN_CAP);
	long rows = 0, kept = 0, answers = 0, with_reasoning = 0, nothink_rows = 0, systems = 0, bytes = 0, reason_bytes = 0;
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
		/* the turns, cleaned: an optional system message first, then the speakers alternating from the user */
		int n = 0, expect = USER, reason = 0, has_reasoning = 0;
		turn *sys = NULL, *kept_turns[MAX_TURNS];
		for (int i = 0; i < nt && !reason; i++) {
			turn *t = &turns[i];
			if (t->role == OTHER) continue;
			if (t->role == SYSTEM) {
				if (i != 0 || no_system) { if (i != 0) reason = 4; continue; }
				normalise(t->text);
				if (do_detok) detok(t->text);
				trim(t->text);
				t->n = strlen((char *)t->text);
				if (t->n > max_chars) { reason = 3; break; }
				if (t->n) sys = t;
				continue;
			}
			if (t->role != expect) { reason = 4; break; }
			if (t->role == ASSISTANT) {
				if (t->rn == 0) { extract_block(t->text, "<think>", "</think>", t->reason); extract_block(t->text, "<thinking>", "</thinking>", t->reason); }
				normalise(t->reason);
				if (do_detok) detok(t->reason);
				trim(t->reason);
				t->rn = strlen((char *)t->reason);
				if (t->rn > max_chars) { reason = 3; break; }
				if (t->rn) has_reasoning = 1;
			}
			normalise(t->text);
			if (do_detok) detok(t->text);
			trim(t->text);
			t->n = strlen((char *)t->text);
			if (t->n == 0) { reason = 1; break; }
			if (drop_code && (strstr((char *)t->text, "```") || (t->rn && strstr((char *)t->reason, "```")))) { reason = 2; break; }
			if (t->n > max_chars) { reason = 3; break; }
			kept_turns[n++] = t;
			expect ^= 1;
		}
		if (!reason && (n < 2 || kept_turns[n - 1]->role != ASSISTANT)) reason = 4;   /* ends on an answer */
		if (!reason) {
			long total = 0, nonascii = 0;
			for (int i = 0; i < n; i++) {
				const turn *t = kept_turns[i];
				total += t->n + t->rn;
				for (long k = 0; k < t->n; k++) nonascii += t->text[k] >= 0x80;
				for (long k = 0; k < t->rn; k++) nonascii += t->reason[k] >= 0x80;
			}
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
		const int nothink = (long)(fnv(kept_turns[0]->text, kept_turns[0]->n) % 1000) < no_think;
		if (sys) { bytes += fprintf(out, "<|system|>%s<|end|>\n", sys->text); systems++; }
		for (int i = 0; i < n; i++) {
			const turn *t = kept_turns[i];
			if (t->role == USER) { bytes += fprintf(out, "<|user|>%s<|end|>\n", t->text); continue; }
			if (nothink || t->rn == 0) bytes += fprintf(out, no_block ? "<|assistant|>%s<|end|>\n" : "<|assistant|><|think|><|/think|>%s<|end|>\n", t->text);
			else { bytes += fprintf(out, "<|assistant|><|think|>%s<|/think|>%s<|end|>\n", t->reason, t->text); reason_bytes += t->rn; }
			answers++;
		}
		bytes += fprintf(out, "<|endoftext|>\n");
		kept++;
		with_reasoning += has_reasoning;
		nothink_rows += nothink;
	}
	fclose(out);
	printf("chat-text: %ld rows: %ld conversations kept (%ld answers, %ld with reasoning in the data, %ld drawn no-think, %ld with a system message; %ld bytes, %ld of reasoning); skipped %ld malformed, %ld empty turn, %ld code, %ld over %ld chars, %ld out of order, %ld non-ASCII\n",
	       rows, kept, answers, with_reasoning, nothink_rows, systems, bytes, reason_bytes, sk_shape, sk_empty, sk_code, sk_long, max_chars, sk_order, sk_nonascii);
	return 0;
}
