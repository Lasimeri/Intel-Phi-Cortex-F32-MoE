/*
 * cortex: the host side of the Intel Phi Cortex (Stage 5; MoE stage 1 adds the n-gram memory and the orchestrator).
 *
 *   cortex CKPT "prompt" NGEN [options]            generation through the daemon on the card
 *   cortex --live "prompt" NGEN [options]          the same on the daemon's LIVE weights (no checkpoint staged;
 *                                                  the bus defaults to off, the trained configuration)
 *   cortex --load CKPT [--bus B] [--ngram T G]     load a checkpoint into the daemon and leave (live training)
 *   cortex --replay LOGITS.f32 NSTEPS [options]    the host path alone: the dumped logits of a run (--dump-logits)
 *                                                  fed through the orchestrator and the sampler, for timing it
 *   cortex --stop                                  ends the daemon's session and the daemon
 *
 * Talks to card/cortexd through the mailbox in the shared host-memory region (lib/cx-proto.h): stages the checkpoint,
 * LOAD, BUS, RESET, then one STEP per token. The card answers each STEP with the 4096 raw logits by DMA; this program
 * samples them in llama.cpp's default order (repeat penalty over the last 64 tokens, top-k, top-p, min-p,
 * temperature, draw) with the user's settings (temp 0.5, top-k 75, top-p 0.95, min-p 0.05, rp 1.2) and streams the
 * detokenised text. --greedy takes the argmax instead (the card-side baseline, byte-identical ids). The top-k are
 * selected in one pass with a bounded insertion (a full sort of the 4096 cost as much as the card's step).
 *
 * options
 *   --greedy  --bus 0|1  --temp T  --topk K  --topp P  --minp M  --rp R  --seed S  --vocab FILE  --chat
 *   --no-echo               the prompt is not printed before the generated text (a chat client shows it already)
 *   --stop-eot              the generation ends at <|endoftext|>, the story's end: the token is printed as its
 *                           literal and not counted as generated (the chat TUI in tui/ keeps it as the boundary)
 *   --ids FILE.u16         the prompt as token ids from a file (the prompt text is then ignored; gates replay it)
 *   --dump-logits FILE      every step's 4096 raw logits as received from the card, appended as F32 (gates, replay)
 *   --ngram TABLES GATES    the n-gram memory: TABLES = 2 orders x CX_NG_ROWS x 128 F32 (ngserve --save), GATES =
 *                           2 x 128 F32 (the run's ng-gates.f32). The gates are staged after the checkpoint (LOAD arg
 *                           1); before each STEP the two rows of the step's suffix (ids[s-2], ids[s-1], ids[s]) are
 *                           written into the ring in the part, as ngserve gathers them for training.
 *   --orchestra NGC [--words LIST] [--word-min N] [--lambda X] [--no-filter] [--repeat-ok]
 *                           host/orchestra.h, between the card's logits and the sampler; the filter counts a listed
 *                           word from corpus count N on (default ORCH_WORD_MIN)
 */
#define _GNU_SOURCE
#define BPE_NO_MAIN
#include "orchestra.h"
#include "../tools/bpe.c"
#include "../lib/cx-proto.h"
#include "../lib/cx-col.h"

#include <fcntl.h>
#include <math.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

static uint8_t *part;
#define LD64(o) __atomic_load_n((uint64_t *)(part + (o)), __ATOMIC_ACQUIRE)
#define LD32(o) __atomic_load_n((uint32_t *)(part + (o)), __ATOMIC_ACQUIRE)
#define ST64(o, x) __atomic_store_n((uint64_t *)(part + (o)), (x), __ATOMIC_RELEASE)
#define ST32(o, x) __atomic_store_n((uint32_t *)(part + (o)), (x), __ATOMIC_RELEASE)

static uint64_t ns(void)
{
	struct timespec t;
	clock_gettime(CLOCK_MONOTONIC, &t);
	return (uint64_t)t.tv_sec * 1000000000u + (uint64_t)t.tv_nsec;
}

/* ---- the mailbox ---- */

static uint64_t seq;

static int command(uint32_t op, uint32_t arg, double timeout_s)
{
	seq++;
	ST32(CX_H_CMD_OP, op);
	ST64(CX_H_CMD, CX_CMD(seq, arg));
	const uint64_t t0 = ns();
	while (LD64(CX_C_ACK) != seq) {
		if (LD32(CX_C_STATE) == 2) { fprintf(stderr, "cortex: the card reported an error on op %u\n", op); return -1; }
		if (ns() - t0 > (uint64_t)(timeout_s * 1e9)) { fprintf(stderr, "cortex: op %u timed out after %.1f s\n", op, timeout_s); return -1; }
	}
	return 0;
}

static int open_part(void)
{
	const int fd = open("/dev/shm/phi-hostmem", O_RDWR);
	if (fd < 0) { perror("/dev/shm/phi-hostmem"); return -1; }
	part = mmap(NULL, CX_PART_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, (off_t)CX_PART_MIB << 20);
	close(fd);
	if (part == MAP_FAILED) { perror("mmap"); return -1; }
	return 0;
}

/* a new session: the daemon takes it (state 1) or the wait times out */
static int open_session(double timeout_s)
{
	ST32(CX_H_MAGIC, CX_MAGIC);
	ST32(CX_H_VERSION, CX_VERSION);
	seq = CX_CMD_SEQ(LD64(CX_H_CMD));
	if (seq == CX_CMD_QUIT) seq = 0;
	ST64(CX_H_CMD, CX_CMD(seq, 0));
	const uint64_t session = ns();
	ST64(CX_H_SESSION, session);
	const uint64_t t0 = ns();
	while (LD64(CX_C_SESSION) != session || LD32(CX_C_STATE) != 1) {
		if (ns() - t0 > (uint64_t)(timeout_s * 1e9)) { fprintf(stderr, "cortex: cortexd did not take the session in %.0f s (is it running on the card?)\n", timeout_s); return -1; }
		usleep(1000);
	}
	return 0;
}

/* ---- the n-gram memory (host side of the generation ring) ---- */

static float *ngT[CX_NG_ORDERS];   /* the tables in host RAM */
static float nggates[CX_NG_ORDERS * CX_DE];

static int ng_load(const char *tables, const char *gates)
{
	const uint64_t t0 = ns();
	FILE *f = fopen(tables, "rb");
	if (!f) { perror(tables); return -1; }
	for (int o = 0; o < CX_NG_ORDERS; o++) {
		ngT[o] = malloc((size_t)CX_NG_ROWS * CX_DE * sizeof *ngT[o]);
		if (!ngT[o] || fread(ngT[o], sizeof *ngT[o], (size_t)CX_NG_ROWS * CX_DE, f) != (size_t)CX_NG_ROWS * CX_DE) { fprintf(stderr, "cortex: %s is not %d x %u x %d F32\n", tables, CX_NG_ORDERS, CX_NG_ROWS, CX_DE); return -1; }
	}
	fclose(f);
	f = fopen(gates, "rb");
	if (!f || fread(nggates, sizeof *nggates, CX_NG_ORDERS * CX_DE, f) != CX_NG_ORDERS * CX_DE) { perror(gates); return -1; }
	fclose(f);
	double gsum[CX_NG_ORDERS] = { 0 };
	for (int o = 0; o < CX_NG_ORDERS; o++) for (int i = 0; i < CX_DE; i++) gsum[o] += nggates[o * CX_DE + i];
	fprintf(stderr, "cortex: n-gram memory: %d orders x %u rows loaded in %.1f s (%zu bytes), gate means %.3f %.3f\n", CX_NG_ORDERS, CX_NG_ROWS,
	        (ns() - t0) / 1e9, (size_t)CX_NG_ORDERS * CX_NG_ROWS * CX_DE * sizeof(float), gsum[0] / CX_DE, gsum[1] / CX_DE);
	return 0;
}

/* the rows of step n (its suffix p2, p1, id) into the ring slot the card will pull before the STEP */
static void ng_rows(long n, uint32_t id, uint32_t p1, uint32_t p2)
{
	float *slot = (float *)(part + CX_NGS_ROWS(n));
	for (int o = 0; o < CX_NG_ORDERS; o++) memcpy(slot + o * CX_DE, ngT[o] + (size_t)cx_ng_hash(id, p1, p2, o + 2) * CX_DE, CX_DE * sizeof *slot);
	__atomic_thread_fence(__ATOMIC_RELEASE);
}

/* ---- the sampler: logits in, token out ---- */

typedef struct { int id; float l; double p; } cand;

static uint64_t rng;
static double uni(void) { return (double)(cx_rng(&rng) >> 11) / 9007199254740992.0; }

/* the k largest of the 4096 values, descending: one pass, a bounded insertion into the kept list */
static int select_top(const float *l, int k, cand *c)
{
	int n = 0;
	for (int v = 0; v < CX_VOCAB; v++) {
		if (n == k && l[v] <= c[n - 1].l) continue;
		int i = n < k ? n : k - 1;
		for (; i > 0 && l[v] > c[i - 1].l; i--) c[i] = c[i - 1];
		c[i].id = v;
		c[i].l = l[v];
		if (n < k) n++;
	}
	return n;
}

static int sample(const float *logits, const uint32_t *hist, int nhist, int greedy, float temp, int topk, float topp, float minp, float rp)
{
	static float l[CX_VOCAB];
	static cand c[CX_VOCAB];
	memcpy(l, logits, sizeof l);
	if (greedy) {
		int best = 0;
		for (int v = 1; v < CX_VOCAB; v++) if (l[v] > l[best]) best = v;
		return best;
	}
	/* repeat penalty over the last 64 tokens (llama.cpp: positive logits divided, negative multiplied) */
	for (int i = nhist > 64 ? nhist - 64 : 0; i < nhist; i++) {
		float *x = &l[hist[i]];
		*x = *x > 0 ? *x / rp : *x * rp;
	}
	int n = select_top(l, topk < CX_VOCAB ? topk : CX_VOCAB, c);
	/* softmax over the kept candidates, then top-p, min-p */
	double sum = 0;
	for (int i = 0; i < n; i++) { c[i].p = exp((double)c[i].l - c[0].l); sum += c[i].p; }
	for (int i = 0; i < n; i++) c[i].p /= sum;
	double cum = 0;
	int m = n;
	for (int i = 0; i < n; i++) { cum += c[i].p; if (cum >= topp) { m = i + 1; break; } }
	n = m;
	m = 0;
	for (int i = 0; i < n; i++) if (c[i].p >= minp * c[0].p) m = i + 1;
	n = m > 0 ? m : 1;
	/* temperature on the logits of the survivors, re-normalise, draw */
	sum = 0;
	for (int i = 0; i < n; i++) { c[i].p = exp(((double)c[i].l - c[0].l) / temp); sum += c[i].p; }
	double r = uni() * sum;
	for (int i = 0; i < n; i++) { r -= c[i].p; if (r <= 0) return c[i].id; }
	return c[n - 1].id;
}

/* ---- the prompt ---- */

static int tokenise(const char *prompt, uint32_t *ids)
{
	const unsigned char *p = (const unsigned char *)prompt;
	long n = (long)strlen(prompt), pos = 0;
	int nid = 0;
	uint16_t tmp[WMAX];
	while (pos < n) {
		int sp;
		const int len = next_word(p + pos, n - pos, &sp);
		if (sp) ids[nid++] = EOT;
		else { const int m = encode_word(p + pos, len, tmp); for (int i = 0; i < m; i++) ids[nid++] = tmp[i]; }
		pos += len;
	}
	return nid;
}

static long read_ids(const char *path, uint32_t **out)
{
	FILE *f = fopen(path, "rb");
	if (!f) { perror(path); return -1; }
	fseek(f, 0, SEEK_END);
	const long n = ftell(f) / 2;
	fseek(f, 0, SEEK_SET);
	uint16_t *u = malloc((size_t)n * 2);
	if (fread(u, 2, (size_t)n, f) != (size_t)n) { perror(path); return -1; }
	fclose(f);
	*out = malloc((size_t)n * 4);
	for (long i = 0; i < n; i++) (*out)[i] = u[i];
	free(u);
	return n;
}

/* ---- the host side of one step: orchestrator, sampler, print; shared by the card loop and the replay ---- */

typedef struct {
	orch *O;
	int greedy, chat, stop_eot;
	float temp, topp, minp, rp;
	int topk;
	uint64_t orch_ns, sample_ns, print_ns;
	char tail[64];
	int tl;
} host_side;

/* the next token from the step's logits; -1 when the chat turn ended */
static int host_step(host_side *h, float *logits, uint32_t *ids, int s)
{
	uint64_t a = ns();
	if (h->O) orch_adjust(h->O, logits, ids, s + 1, tok_s, tok_len, EOT);
	const uint64_t b = ns();
	const int id = sample(logits, ids, s + 1, h->greedy, h->temp, h->topk, h->topp, h->minp, h->rp);
	const uint64_t c = ns();
	h->orch_ns += b - a;
	h->sample_ns += c - b;
	if (h->chat && id == EOT) return -1;
	if (h->stop_eot && id == EOT) { fwrite(tok_s[id], 1, (size_t)tok_len[id], stdout); fflush(stdout); return -1; }
	ids[s + 1] = (uint32_t)id;
	if (h->chat) {
		const int add = tok_len[id] < 63 ? tok_len[id] : 63;
		if (h->tl + add > 63) { memmove(h->tail, h->tail + h->tl + add - 63, (size_t)(63 - add)); h->tl = 63 - add; }
		memcpy(h->tail + h->tl, tok_s[id], (size_t)add);
		h->tl += add;
		h->tail[h->tl] = 0;
		if (strstr(h->tail, "\n\nHuman:") || strstr(h->tail, "\nHuman:")) return -1;
	}
	fwrite(tok_s[id], 1, (size_t)tok_len[id], stdout);
	fflush(stdout);
	h->print_ns += ns() - c;
	return id;
}

static void host_report(const host_side *h, int ngen)
{
	if (h->O) fprintf(stderr, "cortex: orchestra: %ld steps, trigram context %ld, bigram %ld, candidates %ld, word-filtered %ld, rescued %ld (full scans %ld), repeat-blocked %ld\n",
	                  h->O->adjusted, h->O->ng3, h->O->ng2, h->O->cand, h->O->penalised, h->O->rescued, h->O->scans, h->O->repeated);
	if (ngen) fprintf(stderr, "cortex: host per generated token %.1f us: orchestra %.1f, sampler %.1f, print %.1f\n",
	                  (h->orch_ns + h->sample_ns + h->print_ns) / 1e3 / ngen, h->orch_ns / 1e3 / ngen, h->sample_ns / 1e3 / ngen, h->print_ns / 1e3 / ngen);
}

/* ---- main ---- */

static int stop_daemon(void)
{
	if (open_part() || open_session(10)) return 1;
	if (command(CX_OP_EXIT, 0, 5)) return 1;
	fprintf(stderr, "cortex: daemon stopped\n");
	return 0;
}

static void usage(void)
{
	fprintf(stderr, "usage: cortex CKPT \"prompt\" NGEN [options] | cortex --replay LOGITS.f32 NSTEPS [options] | cortex --stop\n"
	                "options: [--greedy] [--bus 0|1] [--temp T] [--topk K] [--topp P] [--minp M] [--rp R] [--seed S] [--vocab FILE] [--chat] [--no-echo] [--stop-eot]\n"
	                "         [--ids FILE.u16] [--dump-logits FILE] [--ngram TABLES GATES]\n"
	                "         [--orchestra NGC [--words LIST] [--word-min N] [--lambda X] [--no-filter] [--repeat-ok]]\n");
}

int main(int argc, char **argv)
{
	if (argc >= 2 && !strcmp(argv[1], "--stop")) return stop_daemon();
	const int replay = argc >= 2 && !strcmp(argv[1], "--replay"), live = argc >= 2 && !strcmp(argv[1], "--live"), load_only = argc >= 2 && !strcmp(argv[1], "--load");
	if (argc < (load_only ? 3 : 4)) { usage(); return 2; }
	const char *ckpt = live ? NULL : load_only ? argv[2] : argv[1], *vocab = "data/bpe/vocab4096.txt", *ids_path = NULL, *dump_path = NULL, *ng_tables = NULL, *ng_gates = NULL;
	char *prompt = replay || load_only ? "" : argv[2];
	int ngen = load_only ? 0 : atoi(argv[3]);
	int bus = live || load_only ? 0 : 1, no_echo = 0, orch_filter = 1, orch_norepeat = 1, orch_wmin = ORCH_WORD_MIN;   /* the live paths default to the trained configuration */
	const char *orch_ngc = NULL, *orch_words = "data/wordlist-stories.txt";
	float orch_lambda = 0.3f;
	static orch O;
	host_side H = { .greedy = 0, .chat = 0, .stop_eot = 0, .temp = 0.5f, .topp = 0.95f, .minp = 0.05f, .rp = 1.2f, .topk = 75 };
	rng = 0x5EED0000C0A7E8ull;
	for (int i = load_only ? 3 : 4; i < argc; i++) {
		if (!strcmp(argv[i], "--greedy")) H.greedy = 1;
		else if (!strcmp(argv[i], "--chat")) H.chat = 1;
		else if (!strcmp(argv[i], "--no-echo")) no_echo = 1;
		else if (!strcmp(argv[i], "--stop-eot")) H.stop_eot = 1;
		else if (!strcmp(argv[i], "--bus") && i + 1 < argc) bus = atoi(argv[++i]);
		else if (!strcmp(argv[i], "--temp") && i + 1 < argc) H.temp = (float)atof(argv[++i]);
		else if (!strcmp(argv[i], "--topk") && i + 1 < argc) H.topk = atoi(argv[++i]);
		else if (!strcmp(argv[i], "--topp") && i + 1 < argc) H.topp = (float)atof(argv[++i]);
		else if (!strcmp(argv[i], "--minp") && i + 1 < argc) H.minp = (float)atof(argv[++i]);
		else if (!strcmp(argv[i], "--rp") && i + 1 < argc) H.rp = (float)atof(argv[++i]);
		else if (!strcmp(argv[i], "--seed") && i + 1 < argc) rng = strtoull(argv[++i], NULL, 10);
		else if (!strcmp(argv[i], "--vocab") && i + 1 < argc) vocab = argv[++i];
		else if (!strcmp(argv[i], "--ids") && i + 1 < argc) ids_path = argv[++i];
		else if (!strcmp(argv[i], "--dump-logits") && i + 1 < argc) dump_path = argv[++i];
		else if (!strcmp(argv[i], "--ngram") && i + 2 < argc) { ng_tables = argv[++i]; ng_gates = argv[++i]; }
		else if (!strcmp(argv[i], "--orchestra") && i + 1 < argc) orch_ngc = argv[++i];
		else if (!strcmp(argv[i], "--words") && i + 1 < argc) orch_words = argv[++i];
		else if (!strcmp(argv[i], "--lambda") && i + 1 < argc) orch_lambda = (float)atof(argv[++i]);
		else if (!strcmp(argv[i], "--no-filter")) orch_filter = 0;
		else if (!strcmp(argv[i], "--repeat-ok")) orch_norepeat = 0;
		else if (!strcmp(argv[i], "--word-min") && i + 1 < argc) orch_wmin = atoi(argv[++i]);
		else { fprintf(stderr, "cortex: unknown option %s\n", argv[i]); return 2; }
	}
	if (H.chat) { char *w = malloc(strlen(prompt) + 40); sprintf(w, "\n\nHuman: %s\n\nAssistant:", prompt); prompt = w; }
	if (load_vocab(vocab)) return 1;
	for (int id = 0; id < 256 + nmerges; id++) expand(id);
	expand(EOT);
	if (orch_ngc && orch_load(&O, orch_ngc, orch_words, orch_lambda, orch_filter, orch_norepeat, orch_wmin)) return 1;
	if (orch_ngc) { H.O = &O; fprintf(stderr, "cortex: orchestra on: %s, lambda %.2f, word filter %s (%s, count >= %d), repeat block %s\n", orch_ngc, orch_lambda, orch_filter ? "on" : "off", orch_words, orch_wmin, orch_norepeat ? "on" : "off"); }

	/* the prompt's tokens: from the ids file, or tokenised */
	uint32_t *ids;
	int nprompt = 0;
	if (ids_path) {
		uint32_t *given;
		const long n = read_ids(ids_path, &given);
		if (n <= 0) return 1;
		ids = malloc((size_t)(n + ngen + 16) * 4);
		memcpy(ids, given, (size_t)n * 4);
		free(given);
		nprompt = (int)n;
	} else {
		ids = malloc((size_t)(strlen(prompt) + ngen + 16) * 4);
		nprompt = tokenise(prompt, ids);
	}
	float *logits = malloc(CX_VOCAB * sizeof *logits);

	/* ---- replay: the dumped logits through the host path, no card ---- */
	if (replay) {
		FILE *f = fopen(argv[2], "rb");
		if (!f) { perror(argv[2]); return 1; }
		const int nsteps = ngen;
		float *dump = malloc((size_t)nsteps * CX_VOCAB * sizeof *dump);
		if (fread(dump, sizeof *dump, (size_t)nsteps * CX_VOCAB, f) != (size_t)nsteps * CX_VOCAB) { fprintf(stderr, "cortex: %s holds fewer than %d steps\n", argv[2], nsteps); return 1; }
		fclose(f);
		if (nprompt > nsteps) nprompt = nsteps;
		if (nprompt < 1) { ids[0] = EOT; nprompt = 1; }
		ids = realloc(ids, (size_t)(nsteps + 16) * 4);
		const uint64_t g0 = ns();
		int gen = 0;
		for (int s = 0; s < nsteps; s++) {
			memcpy(logits, dump + (size_t)s * CX_VOCAB, CX_VOCAB * sizeof *logits);
			if (s + 1 < nprompt) continue;
			if (host_step(&H, logits, ids, s) < 0) break;
			gen++;
		}
		printf("\n");
		host_report(&H, gen);
		fprintf(stderr, "cortex: replay of %d steps (%d prompt, %d generated) in %.3f s\n", nsteps, nprompt, gen, (ns() - g0) / 1e9);
		return 0;
	}

	/* ---- the card: session, then the checkpoint (and gates) staged and loaded, or the live weights as they are ---- */
	if (ng_tables && ng_load(ng_tables, ng_gates)) return 1;
	FILE *dump = NULL, *f = NULL;
	if (dump_path && !(dump = fopen(dump_path, "wb"))) { perror(dump_path); return 1; }
	if (open_part() || open_session(30)) return 1;
	if (ckpt) {
		f = fopen(ckpt, "rb");
		if (!f) { perror(ckpt); return 1; }
		if (fread(part + CX_STAGE, 1, CX_STAGE_CKPT, f) != CX_STAGE_CKPT) { fprintf(stderr, "cortex: short checkpoint %s\n", ckpt); return 1; }
		fclose(f);
		if (ng_tables) memcpy(part + CX_STAGE_GATES, nggates, sizeof nggates);
		__atomic_thread_fence(__ATOMIC_SEQ_CST);
		const uint64_t t0 = ns();
		if (command(CX_OP_LOAD, ng_tables ? 1 : 0, 30)) return 1;
		fprintf(stderr, "cortex: checkpoint loaded by DMA in %.1f ms; prompt %d tokens; %s; n-gram memory %s\n", (ns() - t0) / 1e6, nprompt, H.greedy ? "greedy" : "sampling", ng_tables ? "on" : "off");
	} else {
		fprintf(stderr, "cortex: the daemon's live weights; prompt %d tokens; %s; bus %s\n", nprompt, H.greedy ? "greedy" : "sampling", bus ? "on" : "off");
	}
	if (command(CX_OP_BUS, (uint32_t)bus, 5) || command(CX_OP_RESET, 0, 5)) return 1;
	if (load_only) {
		ST64(CX_H_CMD, CX_CMD(CX_CMD_QUIT, 0));
		fprintf(stderr, "cortex: %s is now the daemon's live model (bus %s)\n", ckpt, bus ? "on" : "off");
		return 0;
	}

	/* the loop: STEP per token, the logits handled on the host */
	if (!no_echo) {
		if (ids_path) { for (int i = 0; i < nprompt; i++) fwrite(tok_s[ids[i]], 1, (size_t)tok_len[ids[i]], stdout); }
		else fwrite(prompt, 1, strlen(prompt), stdout);
		fflush(stdout);
	}
	const uint64_t g0 = ns();
	uint64_t card_ns = 0, rows_ns = 0;
	int steps = 0;
	for (int s = 0; s < nprompt + ngen - 1; s++) {
		const uint64_t a = ns();
		if (ng_tables) ng_rows(s, ids[s], s >= 1 ? ids[s - 1] : CX_NG_PAD, s >= 2 ? ids[s - 2] : CX_NG_PAD);
		const uint64_t a2 = ns();
		rows_ns += a2 - a;
		if (command(CX_OP_STEP, ids[s], 5)) return 1;
		card_ns += ns() - a2;
		steps++;
		if (dump) fwrite(part + CX_LOGITS, sizeof *logits, CX_VOCAB, dump);
		if (s + 1 < nprompt) continue;
		memcpy(logits, part + CX_LOGITS, CX_VOCAB * sizeof *logits);
		if (host_step(&H, logits, ids, s) < 0) { ngen = s + 1 - nprompt; break; }
	}
	const uint64_t g1 = ns();
	printf("\n");
	if (dump) fclose(dump);
	host_report(&H, ngen);
	fprintf(stderr, "cortex: %d prompt + %d generated tokens in %.3f s: %.1f tok/s overall; per token card %.1f us (mailbox + forward + DMA), host %.1f us (orchestra + sampling + print)\n",
	        nprompt, ngen, (g1 - g0) / 1e9, steps / ((g1 - g0) / 1e9), card_ns / 1e3 / steps, ngen ? (H.orch_ns + H.sample_ns + H.print_ns) / 1e3 / ngen : 0.0);
	if (ng_tables) fprintf(stderr, "cortex: n-gram rows gathered on the host: %.2f us per step\n", rows_ns / 1e3 / steps);
	ST64(CX_H_CMD, CX_CMD(CX_CMD_QUIT, 0));
	f = fopen("data/cortex-last.u16", "wb");   /* the ids of the last run, for cmp against a baseline */
	if (f) { for (int i = 0; i < nprompt + ngen; i++) { const uint16_t v = (uint16_t)ids[i]; fwrite(&v, 2, 1, f); } fclose(f); }
	return 0;
}
