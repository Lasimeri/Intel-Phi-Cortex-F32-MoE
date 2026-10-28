/*
 * soft-align: the teacher's next-token distribution carried onto the student's tokens, the raw material of the partial
 * distillation (docs/DISTIL.md; 2026-10-27 18:43, the user: "adjust the teacher model so it partially distills itself
 * into the PhiLLM"). The teacher writes a batch of conversations and, with it, at every token it generated, the eight
 * likeliest next tokens with their log probabilities (tools/teacher.sh --lp-out, the raw softmax of the logits). The
 * student tokenises the batch's converted text with its own 4,096-id BPE, so the two segmentations differ; where a
 * student token boundary falls exactly on a teacher token boundary inside a turn's text, the teacher's alternatives
 * at that boundary are mapped onto student ids (each alternative's first student token, cut the way the stream is
 * cut) and their probabilities summed per id; the mass the eight do not carry goes to the hard target (the student's
 * actual next token), and the whole distribution is scaled by lambda. Elsewhere the record is empty and the hard
 * target stands alone. The card's loss takes the record as the target (lib/cx-proto.h cx_soft: the target is
 * (1 - sum p_j) onehot(next) + sum p_j onehot(id_j), so an empty record is the plain one-hot and a record holding
 * lambda q is (1 - lambda) onehot + lambda q).
 *   soft-align VOCAB.txt TURNS.txt LP.tsv TEXT OUT.soft [--lambda X] [--top K] [--weight-user W]
 *   soft-align --empty N OUT.soft
 * --weight-user W (default 1) sets the record's loss weight w to W for every token that predicts a token inside a
 * person's line (the turns alternate, the person first), 1 elsewhere: the card scales the step's gradient by w and
 * reports the loss unweighted, so the speaker's own lines carry the training when W < 1 (docs/REASON.md, lever 4).
 * TURNS.txt: the batch's turns, one a line, in order (jq -r '.messages[1:][] | .content' over the batch's jsonl);
 * LP.tsv: one line per teacher token: base64(token) TAB logprob TAB "base64(alt):logprob ..." (jq over the server's
 * logprobs list; a token with no alternatives has an empty third field); TEXT: the batch as the driver feeds it to
 * the encoder (tools/chat-text --no-think-block --detok); OUT.soft: one 68-byte record (lib/cx-proto.h cx_soft) per
 * student token of TEXT, in the stream's order: uint32 id[8] (0xFFFFFFFF unused), float p[8], then the float loss
 * weight w. --empty N writes N empty records (weight 1; the stream's tokens from before the sidecars). Prints the
 * counts: tokens, turns matched, tokens with a distribution, the mapped mass before lambda, how often the teacher's
 * first choice is the student's hard target. SA_DEBUG in the environment prints every turn's position and its shared
 * boundaries.
 */
#define BPE_NO_MAIN
#include "bpe.c"
#include "../lib/cx-proto.h"
#include <math.h>

#define SA_MAXALT 16

typedef struct { unsigned char *s; int len; float p; } alt;
typedef struct { long start; int len, nalt; alt a[SA_MAXALT]; } ttok;

static int b64val(int c)
{
	if (c >= 'A' && c <= 'Z') return c - 'A';
	if (c >= 'a' && c <= 'z') return c - 'a' + 26;
	if (c >= '0' && c <= '9') return c - '0' + 52;
	if (c == '+') return 62;
	if (c == '/') return 63;
	return -1;
}

/* base64 of s (n chars) into out; the byte count */
static int b64dec(const char *s, int n, unsigned char *out)
{
	int acc = 0, bits = 0, k = 0;
	for (int i = 0; i < n; i++) {
		const int v = b64val((unsigned char)s[i]);
		if (v < 0) continue;
		acc = (acc << 6) | v;
		bits += 6;
		if (bits >= 8) { bits -= 8; out[k++] = (unsigned char)((acc >> bits) & 255); }
	}
	return k;
}

static unsigned char *read_all(const char *path, long *n)
{
	FILE *f = fopen(path, "rb");
	if (!f) { perror(path); return NULL; }
	fseek(f, 0, SEEK_END);
	*n = ftell(f);
	fseek(f, 0, SEEK_SET);
	unsigned char *b = malloc((size_t)*n + WMAX + 1);
	if (*n > 0 && fread(b, 1, (size_t)*n, f) != (size_t)*n) { fprintf(stderr, "soft-align: read %s\n", path); fclose(f); return NULL; }
	fclose(f);
	b[*n] = 0;
	return b;
}

/* the first student id of a continuation s (len bytes), cut the way the encoder cuts the stream; -1 for nothing */
static int first_id(const unsigned char *s, int len)
{
	if (len <= 0) return -1;
	int sp;
	const int wl = next_word(s, len, &sp);
	if (sp) return sp;
	cx_tok ids[WMAX + 1];
	const int m = encode_word(s, wl, ids);
	return m > 0 ? (int)ids[0] : -1;
}

static const unsigned char *find_from(const unsigned char *hay, long hn, long from, const unsigned char *needle, long nn)
{
	if (nn <= 0 || from >= hn) return NULL;
	return memmem(hay + from, (size_t)(hn - from), needle, (size_t)nn);
}

static void empty_rec(cx_soft *r)
{
	for (int k = 0; k < CX_SOFT_K; k++) { r->id[k] = CX_SOFT_NONE; r->p[k] = 0; }
	r->w = 1.0f;
}

static int write_empty(long n, const char *path)
{
	FILE *o = fopen(path, "wb");
	if (!o) { perror(path); return 1; }
	cx_soft r;
	empty_rec(&r);
	for (long i = 0; i < n; i++) if (fwrite(&r, sizeof r, 1, o) != 1) { perror(path); return 1; }
	if (fclose(o)) { perror(path); return 1; }
	printf("soft-align: %ld empty records in %s\n", n, path);
	return 0;
}

int main(int argc, char **argv)
{
	if (argc == 4 && !strcmp(argv[1], "--empty")) return write_empty(atol(argv[2]), argv[3]);
	if (argc < 6) { fprintf(stderr, "usage: soft-align VOCAB.txt TURNS.txt LP.tsv TEXT OUT.soft [--lambda X] [--top K] [--weight-user W] | --empty N OUT.soft\n"); return 2; }
	int top = CX_SOFT_K;
	double lambda = 0.5, wuser = 1.0;
	for (int i = 6; i < argc; i++) {
		if (!strcmp(argv[i], "--top") && i + 1 < argc) top = atoi(argv[++i]);
		else if (!strcmp(argv[i], "--lambda") && i + 1 < argc) lambda = atof(argv[++i]);
		else if (!strcmp(argv[i], "--weight-user") && i + 1 < argc) wuser = atof(argv[++i]);
		else { fprintf(stderr, "soft-align: unknown option %s\n", argv[i]); return 2; }
	}
	if (top < 1 || top > CX_SOFT_K) { fprintf(stderr, "soft-align: --top 1..%d\n", CX_SOFT_K); return 2; }
	if (lambda < 0 || lambda > 1) { fprintf(stderr, "soft-align: --lambda 0..1\n"); return 2; }
	if (wuser < 0 || wuser > 1) { fprintf(stderr, "soft-align: --weight-user 0..1\n"); return 2; }
	const int dbg = getenv("SA_DEBUG") != NULL;
	if (load_vocab(argv[1])) return 1;
	for (int id = 0; id < VOCAB; id++) expand(id);
	/* the student's tokens of TEXT, with byte spans: the encoder's own walk */
	long tn;
	unsigned char *text = read_all(argv[4], &tn);
	if (!text) return 1;
	cx_tok *sid = malloc((size_t)(tn + 1) * sizeof *sid);
	long *sen = malloc((size_t)(tn + 1) * sizeof *sen);
	long ns = 0, pos = 0;
	while (pos < tn) {
		int sp;
		const int len = next_word(text + pos, tn - pos, &sp);
		if (sp) { sid[ns] = (cx_tok)sp; sen[ns] = pos + len; ns++; }
		else {
			cx_tok ids[WMAX + 1];
			const int m = encode_word(text + pos, len, ids);
			long off = pos;
			for (int k = 0; k < m; k++) { sid[ns] = ids[k]; off += tok_len[ids[k]]; sen[ns] = off; ns++; }
			if (off != pos + len) { fprintf(stderr, "soft-align: the pieces of a word do not add up at byte %ld\n", pos); return 1; }
		}
		pos += len;
	}
	/* the teacher's tokens: the reply's bytes rebuilt, each token's start and alternatives */
	FILE *f = fopen(argv[3], "r");
	if (!f) { perror(argv[3]); return 1; }
	long tcap = 1 << 16, nt = 0, rcap = 1 << 20, rn = 0, nalt_all = 0;
	ttok *tt = malloc((size_t)tcap * sizeof *tt);
	unsigned char *R = malloc((size_t)rcap);
	char *line = NULL;
	size_t lcap = 0;
	ssize_t ll;
	while ((ll = getline(&line, &lcap, f)) > 0) {
		if (nt == tcap) { tcap *= 2; tt = realloc(tt, (size_t)tcap * sizeof *tt); }
		ttok *t = &tt[nt];
		memset(t, 0, sizeof *t);
		char *tab1 = memchr(line, '\t', (size_t)ll);
		if (!tab1) continue;
		char *tab2 = memchr(tab1 + 1, '\t', (size_t)(ll - (tab1 + 1 - line)));
		unsigned char tok[1024];
		const int tl = b64dec(line, (int)(tab1 - line), tok);
		if (rn + tl + 1 > rcap) { rcap *= 2; R = realloc(R, (size_t)rcap); }
		t->start = rn;
		t->len = tl;
		memcpy(R + rn, tok, (size_t)tl);
		rn += tl;
		if (tab2) {   /* the alternatives: base64:logprob, space-separated */
			char *p = tab2 + 1, *end = line + ll;
			while (p < end && t->nalt < SA_MAXALT) {
				while (p < end && (*p == ' ' || *p == '\n')) p++;
				if (p >= end) break;
				char *colon = memchr(p, ':', (size_t)(end - p));
				if (!colon) break;
				char *sp = memchr(colon, ' ', (size_t)(end - colon));
				if (!sp) sp = end;
				alt *a = &t->a[t->nalt];
				a->s = malloc(1024);
				a->len = b64dec(p, (int)(colon - p), a->s);
				a->p = (float)exp(strtod(colon + 1, NULL));
				t->nalt++;
				p = sp;
			}
		}
		nalt_all += t->nalt > 0;
		nt++;
	}
	fclose(f);
	free(line);
	R[rn] = 0;
	/* the turns: each found in TEXT and in the reply, in order; for each student token whose next token starts inside the
	 * turn, the teacher token whose boundary that is (tb, -1: none) */
	long un;
	unsigned char *turns = read_all(argv[2], &un);
	if (!turns) return 1;
	long cT = 0, cR = 0, nturn = 0, nmatched = 0;
	long *tb = malloc((size_t)(ns + 1) * sizeof *tb);
	for (long i = 0; i <= ns; i++) tb[i] = -1;
	/* the role of every byte of TEXT: 1 inside a person's line, 2 inside the speaker's (the turns alternate, the person
	 * first), 0 elsewhere (the markers, the system line); the loss weight of a token follows the role of the token it
	 * predicts */
	unsigned char *role = calloc((size_t)tn + 1, 1);
	for (long lp = 0; lp < un;) {
		long le = lp;
		while (le < un && turns[le] != '\n') le++;
		const long L = le - lp;
		if (L > 0) {
			nturn++;
			const unsigned char *inT = find_from(text, tn, cT, turns + lp, L);
			const unsigned char *inR = find_from(R, rn, cR, turns + lp, L);
			if (inT) { cT = (inT - text) + L; memset(role + (inT - text), (nturn & 1) ? 1 : 2, (size_t)L); }
			long shared = 0;
			if (inT && inR) {
				nmatched++;
				const long c = inT - text, a = inR - R;
				cR = a + L;
				for (long i = 0; i + 1 < ns; i++) {
					const long e = sen[i];
					if (e < c || e >= c + L) continue;   /* the next student token starts inside the turn */
					const long r = a + (e - c);
					long lo = 0, hi = nt - 1;
					while (lo <= hi) {
						const long mid = (lo + hi) / 2;
						if (tt[mid].start == r) { tb[i] = mid; shared++; break; }
						if (tt[mid].start < r) lo = mid + 1; else hi = mid - 1;
					}
				}
			}
			if (dbg) printf("turn %ld: %ld bytes, in the text at %ld, in the reply at %ld, shared boundaries %ld\n", nturn, L, inT ? (long)(inT - text) : -1L, inR ? (long)(inR - R) : -1L, shared);
		}
		lp = le + 1;
	}
	/* the records */
	FILE *o = fopen(argv[5], "wb");
	if (!o) { perror(argv[5]); return 1; }
	long covered = 0, agree = 0, weighted = 0;
	double mass = 0;
	for (long i = 0; i < ns; i++) {
		cx_soft rec;
		empty_rec(&rec);
		if (i + 1 < ns && role[sen[i]] == 1 && wuser < 1.0) { rec.w = (float)wuser; weighted++; }   /* the next token is in a person's line */
		if (tb[i] >= 0 && i + 1 < ns) {
			const ttok *t = &tt[tb[i]];
			int ids[SA_MAXALT + 1], m = 0;
			double ps[SA_MAXALT + 1], sum = 0;
			for (int j = 0; j < t->nalt; j++) {
				const int id = first_id(t->a[j].s, t->a[j].len);
				if (id < 0) continue;
				int k;
				for (k = 0; k < m; k++) if (ids[k] == id) { ps[k] += t->a[j].p; break; }
				if (k == m) { ids[m] = id; ps[m] = t->a[j].p; m++; }
				sum += t->a[j].p;
			}
			if (m) {
				const int h = sid[i + 1];
				if (sum > 1.0) { for (int k = 0; k < m; k++) ps[k] /= sum; sum = 1.0; }
				int k;
				for (k = 0; k < m; k++) if (ids[k] == h) break;
				if (k == m) { ids[m] = h; ps[m] = 0; m++; }
				ps[k] += 1.0 - sum;   /* the mass the alternatives do not carry goes to the hard target */
				for (int x = 0; x < m; x++) for (int y = x + 1; y < m; y++) if (ps[y] > ps[x]) { const double tp = ps[x]; ps[x] = ps[y]; ps[y] = tp; const int ti = ids[x]; ids[x] = ids[y]; ids[y] = ti; }
				const int keep = m < top ? m : top;
				double kept = 0;
				for (k = 0; k < keep; k++) kept += ps[k];
				for (k = 0; k < keep; k++) { rec.id[k] = (cx_tok)ids[k]; rec.p[k] = (float)(lambda * ps[k] / kept); }   /* the kept ids renormalised, then lambda */
				covered++;
				mass += sum;
				if (t->nalt && first_id(t->a[0].s, t->a[0].len) == h) agree++;
			}
		}
		fwrite(&rec, sizeof rec, 1, o);
	}
	if (fclose(o)) { perror(argv[5]); return 1; }
	printf("soft-align: %ld student tokens, %ld of %ld turns matched in the reply, %ld tokens with the teacher's distribution (%.1f %%), mapped mass %.3f before lambda %.2f, the teacher's first choice is the hard target %.1f %% of them; %ld teacher tokens, %ld with alternatives; %ld tokens of the person's lines at weight %.2f\n",
	       ns, nmatched, nturn, covered, ns ? 100.0 * covered / ns : 0, covered ? mass / covered : 0, lambda, covered ? 100.0 * agree / covered : 0, nt, nalt_all, weighted, wuser);
	return 0;
}
