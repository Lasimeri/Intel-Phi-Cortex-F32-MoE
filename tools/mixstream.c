/*
 * mixstream: a training stream that recurs a small set of conversations inside the base stream: one conversation of
 * SET.u32 (cycled), then RATIO whole conversations of BASE.u32 from START (moved up to the next conversation
 * boundary), again, until OUT.u32 holds at least CHUNKS x 32 tokens of whole conversations (the coach drops a
 * partial last chunk). The streams are little-endian u32 ids (lib/cx-proto.h cx_tok, 4 B a token, 128 B a chunk).
 * The identity fine-tune loop (tools/identity-loop.sh) feeds the coach this stream, so the set
 * is seen every RATIO + 1 conversations while the base keeps flowing. A conversation ends at the end marker (id
 * EOT, default 4095); the newline token (id 10) after it opens the next one, and a set conversation cycled back to
 * the stream's first gets one too.
 *   mixstream SET.u32 BASE.u32 START RATIO CHUNKS OUT.u32 [EOT] [--soft BASE.soft OUT.soft [--set-soft SET.soft]]
 *             [--once ONCE.u32 [ONCE.soft]]
 * --set-soft gives the set's own records (one per token of SET.u32), in place of empty ones; the newline inserted before
 * a cycled set conversation still gets an empty record.
 * --once puts every conversation of ONCE.u32 into the stream exactly once, spread evenly over the round's base
 * conversations (the reward set of the from-scratch driver, tools/teacher-reward.sh: each sample's advantage is one
 * on-policy step, and in the set it cycled in file order behind the corrections, so a short round never reached the
 * rewards' tail and an epoch repeated them about five times, 2026-10-28 07:5x); the ones the spread has not placed when
 * the stream is full are appended after it. ONCE.soft holds their records (their weights, tools/reward-soft.c) and is
 * required with --soft.
 * --soft mirrors the stream in the soft-target records of the partial distillation (lib/cx-proto.h cx_soft, 68 B a
 * token, one per token of BASE.u32 in BASE.soft): every base token copied carries its record, every set token and
 * every inserted newline an empty one (ids CX_SOFT_NONE, weight 1), so OUT.soft has one record per token of OUT.u32.
 * Prints the set's conversations and tokens and how often they recur, the base conversations and tokens used and
 * where they came from, the stream's length and the set's share of it.
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

#define NL 10

static const void *map_file(const char *path, long *bytes)
{
	const int fd = open(path, O_RDONLY);
	struct stat st;
	if (fd < 0 || fstat(fd, &st)) { perror(path); exit(1); }
	*bytes = st.st_size;
	const void *p = st.st_size ? mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0) : NULL;
	if (st.st_size && p == MAP_FAILED) { perror(path); exit(1); }
	close(fd);
	return p;
}

/* the soft mirror: records of base tokens copied, empty records for the rest; nothing without --soft */
typedef struct { const cx_soft *base, *set; FILE *out; long nbase, nset; cx_soft empty; } mirror;

/* the set's records of tokens [from, from + n): its own (--set-soft) or empty ones */
static void mirror_set(mirror *m, long from, long n)
{
	if (!m->out) return;
	if (!m->set) { for (long i = 0; i < n; i++) fwrite(&m->empty, sizeof m->empty, 1, m->out); return; }
	if (from + n > m->nset) { fprintf(stderr, "mixstream: the set's soft records end at %ld, token %ld asked\n", m->nset, from + n); exit(1); }
	fwrite(m->set + from, sizeof *m->set, (size_t)n, m->out);
}

static void mirror_base(mirror *m, long from, long n)
{
	if (!m->out) return;
	if (from + n > m->nbase) { fprintf(stderr, "mixstream: the base's soft records end at %ld, token %ld asked\n", m->nbase, from + n); exit(1); }
	fwrite(m->base + from, sizeof *m->base, (size_t)n, m->out);
}

static void mirror_empty(mirror *m, long n)
{
	if (!m->out) return;
	for (long i = 0; i < n; i++) fwrite(&m->empty, sizeof m->empty, 1, m->out);
}

int main(int argc, char **argv)
{
	if (argc < 7) { fprintf(stderr, "usage: mixstream SET.u32 BASE.u32 START RATIO CHUNKS OUT.u32 [EOT] [--soft BASE.soft OUT.soft [--set-soft SET.soft]] [--once ONCE.u32 [ONCE.soft]]\n"); return 2; }
	const long start = atol(argv[3]), chunks = atol(argv[5]);
	const int ratio = atoi(argv[4]);
	cx_tok eot = 4095;
	const char *soft_in = NULL, *soft_out = NULL, *set_soft = NULL, *once_path = NULL, *once_soft = NULL;
	for (int i = 7; i < argc; i++) {
		if (!strcmp(argv[i], "--soft") && i + 2 < argc) { soft_in = argv[++i]; soft_out = argv[++i]; }
		else if (!strcmp(argv[i], "--set-soft") && i + 1 < argc) set_soft = argv[++i];
		else if (!strcmp(argv[i], "--once") && i + 1 < argc) { once_path = argv[++i]; if (i + 1 < argc && argv[i + 1][0] != '-' && !(argv[i + 1][0] >= '0' && argv[i + 1][0] <= '9')) once_soft = argv[++i]; }
		else if (argv[i][0] >= '0' && argv[i][0] <= '9') eot = (cx_tok)atoi(argv[i]);
		else { fprintf(stderr, "mixstream: unknown argument %s\n", argv[i]); return 2; }
	}
	if (ratio < 0 || chunks <= 0) { fprintf(stderr, "mixstream: RATIO >= 0 and CHUNKS > 0\n"); return 2; }
	long nset, nbase, sbytes;
	const cx_tok *set = map_file(argv[1], &nset), *base = map_file(argv[2], &nbase);
	nset /= (long)sizeof *set;
	nbase /= (long)sizeof *base;
	mirror m = { 0 };
	for (int k = 0; k < CX_SOFT_K; k++) { m.empty.id[k] = CX_SOFT_NONE; m.empty.p[k] = 0; }
	m.empty.w = 1.0f;
	if (soft_in) {
		m.base = map_file(soft_in, &sbytes);
		m.nbase = sbytes / (long)sizeof *m.base;
		if (m.nbase < nbase) { fprintf(stderr, "mixstream: %s holds %ld records for %ld tokens of %s\n", soft_in, m.nbase, nbase, argv[2]); return 1; }
		m.out = fopen(soft_out, "wb");
		if (!m.out) { perror(soft_out); return 1; }
		if (set_soft) {
			long ssb;
			m.set = map_file(set_soft, &ssb);
			m.nset = ssb / (long)sizeof *m.set;
			if (m.nset != nset) { fprintf(stderr, "mixstream: %s holds %ld records for %ld tokens of %s\n", set_soft, m.nset, nset, argv[1]); return 1; }
		}
	}
	/* the set's conversations: [from, to] with the end marker at `to` */
	long *from = malloc((size_t)(nset + 1) * sizeof *from), *to = malloc((size_t)(nset + 1) * sizeof *to), nconv = 0, f = 0, set_tokens = 0;
	for (long i = 0; i < nset; i++) if (set[i] == eot) { from[nconv] = f; to[nconv] = i; set_tokens += i - f + 1; nconv++; f = i + 1; }
	if (!nconv) { fprintf(stderr, "mixstream: %s holds no conversation (no id %u)\n", argv[1], eot); return 1; }
	/* the once-set's conversations and records (--once) */
	long nonce = 0, oconv = 0, otokens = 0, *ofrom = NULL, *oto = NULL;
	const cx_tok *once = NULL;
	const cx_soft *osoft = NULL;
	if (once_path) {
		long ob;
		once = map_file(once_path, &ob);
		nonce = ob / (long)sizeof *once;
		ofrom = malloc((size_t)(nonce + 1) * sizeof *ofrom);
		oto = malloc((size_t)(nonce + 1) * sizeof *oto);
		for (long i = 0, g = 0; i < nonce; i++) if (once[i] == eot) { ofrom[oconv] = g; oto[oconv] = i; otokens += i - g + 1; oconv++; g = i + 1; }
		if (m.out) {
			if (!once_soft) { fprintf(stderr, "mixstream: --once with --soft needs ONCE.soft\n"); return 2; }
			long osb;
			osoft = map_file(once_soft, &osb);
			if (osb / (long)sizeof *osoft != nonce) { fprintf(stderr, "mixstream: %s holds %ld records for %ld tokens of %s\n", once_soft, osb / (long)sizeof *osoft, nonce, once_path); return 1; }
		}
	}
	/* the base pointer: at a conversation boundary at or after START */
	long p = start < nbase ? start : nbase;
	while (p > 0 && p < nbase && base[p - 1] != eot) p++;
	const long base_from = p;
	FILE *out = fopen(argv[6], "wb");
	if (!out) { perror(argv[6]); return 1; }
	const long need = chunks * 32;
	long written = 0, set_convs = 0, base_convs = 0, base_tokens = 0, k = 0;
	const cx_tok nl = NL;
	/* the once-set's spread: the base conversations the round will take (counted as if alone, so an over-estimate), the
	 * once-conversations placed over the first 80 % of them, conversation j after base conversation (j + 0.5) x span / n */
	long once_at = 0, span = 0;
	if (oconv) {
		long bt = 0;
		for (long q = p, g = p; q < nbase && bt < need; q++) if (base[q] == eot) { bt += q - g + 1; g = q + 1; span++; }
		span = span * 8 / 10;
	}
#define ONCE_NEXT() ((long)(((double)once_at + 0.5) * (double)span / (double)oconv))
	while (written < need && p < nbase) {
		const long a = from[k], b = to[k];
		if (written > 0 && set[a] != NL) { fwrite(&nl, sizeof nl, 1, out); mirror_empty(&m, 1); written++; }
		fwrite(set + a, sizeof *set, (size_t)(b - a + 1), out);
		mirror_set(&m, a, b - a + 1);
		written += b - a + 1;
		set_convs++;
		k = (k + 1) % nconv;
		for (int r = 0; r < ratio && p < nbase; r++) {
			long q = p;
			while (q < nbase && base[q] != eot) q++;
			if (q >= nbase) { p = nbase; break; }   /* no whole conversation left */
			fwrite(base + p, sizeof *base, (size_t)(q - p + 1), out);
			mirror_base(&m, p, q - p + 1);
			written += q - p + 1;
			base_tokens += q - p + 1;
			base_convs++;
			p = q + 1;
			while (once_at < oconv && base_convs > ONCE_NEXT()) {   /* the once-conversations due after this base conversation */
				const long a = ofrom[once_at], b = oto[once_at];
				if (once[a] != NL) { fwrite(&nl, sizeof nl, 1, out); mirror_empty(&m, 1); written++; }
				fwrite(once + a, sizeof *once, (size_t)(b - a + 1), out);
				if (m.out) fwrite(osoft + a, sizeof *osoft, (size_t)(b - a + 1), m.out);
				written += b - a + 1;
				once_at++;
			}
		}
	}
	const long once_spread = once_at;
	while (once_at < oconv) {   /* the stream is full: the ones not placed yet follow it, so every one trains once */
		const long a = ofrom[once_at], b = oto[once_at];
		if (written > 0 && once[a] != NL) { fwrite(&nl, sizeof nl, 1, out); mirror_empty(&m, 1); written++; }
		fwrite(once + a, sizeof *once, (size_t)(b - a + 1), out);
		if (m.out) fwrite(osoft + a, sizeof *osoft, (size_t)(b - a + 1), m.out);
		written += b - a + 1;
		once_at++;
	}
	if (fclose(out)) { perror(argv[6]); return 1; }
	if (m.out && fclose(m.out)) { perror(soft_out); return 1; }
	char onote[160] = "";
	if (once_path) snprintf(onote, sizeof onote, "; the once-set %ld conversations (%ld tokens), each once: %ld spread over the base, %ld after it", oconv, otokens, once_spread, oconv - once_spread);
	printf("mixstream: %ld set conversations (%ld tokens) recur every %d base conversations, seen %ld times each (%ld in all); %ld base conversations (%ld tokens) from token %ld to %ld of %s; %s: %ld tokens = %ld chunks, the set %.1f %% of them%s%s%s\n",
	       nconv, set_tokens, ratio, set_convs / nconv, set_convs, base_convs, base_tokens, base_from, p, argv[2], argv[6], written, written / 32, 100.0 * (written - base_tokens) / written,
	       onote, soft_out ? "; soft records mirrored into " : "", soft_out ? soft_out : "");
	return 0;
}
