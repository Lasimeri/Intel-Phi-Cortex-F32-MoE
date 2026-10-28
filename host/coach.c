/*
 * coach: the host side of live training (card/cortexd.c, docs/LIVE.md). Feeds the daemon's training ring from a
 * token file, reads its reports, samples the live model through the mailbox (as any client would: sampled decoding,
 * the orchestrator) and scores the samples, applies the plateau policy to the learning rate, and takes commands from
 * a file that a person (or an assistant watching the log) appends to. Nothing the coach writes and nothing the model
 * generates is training data: the gradient comes from the token file only.
 *
 *   coach TOKENS.u32 [options]       TOKENS: little-endian u32 ids (lib/cx-proto.h cx_tok), 4 bytes a token, 128 a chunk
 *     --ckpt CKPT          load this checkpoint into the daemon first (a new run for the Adam state)
 *     --start CHUNK        the first chunk of the token file to feed (default 0)
 *     --chunks N           feed at most N chunks (default: to the end of the file)
 *     --lr X               base learning rate (default 3e-3)
 *     --every N            the daemon reports every N chunks (default 100)
 *     --window W           policy: compare the mean loss of W reports with the previous W (default 10)
 *     --min-gain B         policy: a window gain under B bits/token is a plateau (default 0.01)
 *     --halvings H         policy: at most H halvings of the scale (default 3)
 *     --sample-every N     sample the live model every N chunks (default 2000, 0 = never)
 *     --save-every N       a checkpoint every N chunks, besides the policy's and the final one (default 0 = none)
 *     --no-stop            the policy halves and saves on plateaus but never ends the run: the stream's end or a
 *                          `stop` line does (2026-10-26: a stories run with the default policy ended in 3.5 min)
 *     --sample-tokens N    tokens per sample (default 128)
 *     --prompt TEXT        the sample prompt (default "Once upon a time")
 *     --orchestra NGC      the sampling client's count model (with --words LIST, which also scores the samples)
 *     --words LIST         the word list (default data/wordlist-stories.txt)
 *     --cmd FILE           commands appended as lines: scale X | lr X | pause | resume | save | stop | sample | every N
 *     --ckdir DIR          where the checkpoints land as ckpt-CHUNK.bin (default data/live): the card pushes each one
 *                          into the part's stage by DMA and the coach copies it out; nothing touches the card's disk
 *     --paused             start with the pause flag set (a `save` then yields the loaded weights at chunk 0; `resume`
 *                          starts the training)
 *     --log FILE           the log (default results/live.log, appended)
 *     --cortex PATH        the client binary (default build/host/cortex)
 *     --vocab FILE         the BPE vocabulary (default data/bpe/vocab4096.txt)
 *     --soft FILE          the partial distillation (docs/DISTIL.md): FILE holds one 68-byte record per token of the
 *                          stream (lib/cx-proto.h cx_soft: 8 u32 ids, 8 F32 probabilities, the F32 loss weight;
 *                          tools/soft-align.c, the teacher's distribution mapped onto the student's ids); each
 *                          chunk's records go into the soft ring with it and the daemon mixes them into the target
 *     --ngram-dropout P    with --ngram: the share of chunks trained without the memory, chosen by a hash of the chunk's
 *                          stream position (the slot carries the flag, lib/cx-proto.h CX_T_SLOT_MEMOFF): the daemon
 *                          trains them dense and the ring thread skips their row gradients, so the columns learn to
 *                          predict on their own (21:5x: the dense reading sat at 5.15 bits against 2.76 with the memory)
 *     --ngram FILE         the n-gram memory (MoE stage 1) trains with the run: FILE holds the tables (host/ngtables.h:
 *                          2 orders x CX_NG_ROWS rows x 128 F32, 2 GiB, created zeroed when absent; keep it in
 *                          /dev/shm), mapped shared and trained in place. A thread serves the daemon's row ring in the
 *                          batch trainer's lockstep (host/ngserve.c): the rows of chunk c + 2 go out once the row
 *                          gradients of chunk c are back and applied. The sampler reads its rows from the same file,
 *                          as any client of the live model does (cortex --live ... --ngram FILE).
 *     --row-lr X           the rows' SGD step (default 0.1, the mixed run B's)
 *     --gates FILE|ones    the gates staged with --ckpt: a run's ckpt-N-gates.f32, or ones (the batch trainer's
 *                          start; the default). The gates train on the card (Adam) and come back with every checkpoint
 *                          as ckpt-N-gates.f32.
 *     --tables-every N     the tables copied beside the checkpoint as tables-N.f32 every N chunks (at a --save-every
 *                          checkpoint), on a `save` line and at the end (default 100000; a copy is 2 GiB and holds the
 *                          main loop for its write; the ring thread keeps serving meanwhile)
 *     --moe                MoE stage 2 (lib/cx-moe.h, card/moe.h): the LOAD of --ckpt fills the daemon's pool from the
 *                          checkpoint's MLP blocks (expert e = column e mod 120) and the heads route through it from
 *                          then on; the checkpoints keep the dense format with the home experts in the MLP regions
 *     --moe-bias X         the home pair's routing bonus (default 100: exact home routing = the dense model; lower it
 *                          across runs to open the pool); `bias X` as a command line changes it live
 *     --bias-decay N       the bonus falls linearly from --moe-bias to 0 over the run's first N chunks (written with
 *                          each report), so the heads leave their home experts one by one as the keys' scores pass
 *                          it, instead of all at once (a run that starts free lost 7 bits/token, PLAN-MOE.md)
 *     --expert-lr X        the experts' SGD step (default 0.05); `exlr X` as a command line changes it live
 *     --pool FILE          a saved pool (pool-N.f32, 4,293,918,720 B) pulled into the daemon in 187 rounds of eleven
 *                          2 MiB pages through the stage before the LOAD, which then keeps it (LOAD arg bit 2)
 *                          instead of filling from the checkpoint; the pool is pushed back the same way into
 *                          CKDIR/pool-CHUNK.f32 right after a checkpoint of the same CHUNK (the final one, the one a
 *                          `save-pool` line requests, and one every --pool-every chunks), so a pool and its
 *                          checkpoint always belong to the same chunk (about 30 s each, the chunks keep flowing
 *                          between rounds)
 *     --pool-every N       a checkpoint with the pool every N chunks (default 0 = only at the end and on `save-pool`
 *                          lines); each copy is 4 GiB on the disk
 *
 * Policy v2 (docs/COACH.md of the F64 tree, measured there): every W reports the mean training loss of the window is
 * compared with the previous window; a gain under B is a plateau; a plateau halves the scale down to 2^-H; at the
 * floor, two consecutive windows with a negative gain set the stop flag. A sample whose rep4 exceeds the mean of the
 * previous three samples by more than 0.1 counts as one plateau, at most once per run.
 */
#define _GNU_SOURCE
#define BPE_NO_MAIN
#include "../tools/bpe.c"
#include "../tools/coherence.h"
#include "../lib/cx-proto.h"
#include "../lib/cx-col.h"
#include "../lib/cx-moe.h"
#include "ngtables.h"
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdarg.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#define CX_LN2 0.6931471805599453
#define RING_AHEAD (CX_T_SLOTS - 64)   /* chunks written ahead of the card at most */

static uint8_t *part;
#define LD64(o) __atomic_load_n((uint64_t *)(part + (o)), __ATOMIC_ACQUIRE)
#define LD32(o) __atomic_load_n((uint32_t *)(part + (o)), __ATOMIC_ACQUIRE)
#define ST64(o, x) __atomic_store_n((uint64_t *)(part + (o)), (x), __ATOMIC_RELEASE)
#define ST32(o, x) __atomic_store_n((uint32_t *)(part + (o)), (x), __ATOMIC_RELEASE)
static double kdbl(uint64_t u) { double d; memcpy(&d, &u, 8); return d; }
static uint64_t kbits(double d) { uint64_t u; memcpy(&u, &d, 8); return u; }

/* ---- settings ---- */
static const char *tokens_path, *ckpt, *cmd_path, *ckdir = "data/live", *log_path = "results/live.log", *cortex_bin = "build/host/cortex", *vocab = "data/bpe/vocab4096.txt";
static const char *prompt = "Once upon a time", *orchestra, *words = "data/wordlist-stories.txt";
static long start_chunk, max_chunks = -1, sample_every = 2000, save_every;
static int sample_tokens = 128, every = 100, window = 10, max_halvings = 3, start_paused, no_stop;
static double lr = 3e-3, min_gain = 0.01, rep4_rise = 0.1;
static const char *ng_path, *gates_spec = "ones";   /* --ngram: the memory trains with the run; --gates: staged with --ckpt */
static double row_lr = 0.1;
static double ngram_dropout;                        /* --ngram-dropout: the share of chunks trained without the memory */

/* the dropout's choice for chunk c of the run: a hash of its stream position, so a restart from the same --start makes
 * the same choice and the host and the card agree without a word between them */
static int chunk_dropped(long c)
{
	if (ngram_dropout <= 0) return 0;
	uint64_t h = (uint64_t)(start_chunk + c) * 0x9E3779B97F4A7C15ull + 0x632BE59BD9B4E019ull;
	h ^= h >> 29;
	h *= 0xBF58476D1CE4E5B9ull;
	h ^= h >> 32;
	return (double)(h % 1000000u) < ngram_dropout * 1e6;
}
static long tables_every = 100000;
static int moe, moe_knobs_given;                   /* --moe: the LOAD fills the pool; the knobs are written when given or with --moe */
static double moe_bias = CX_EX_HOME_BIAS, expert_lr = CX_EX_LR;
static long bias_decay;                            /* --bias-decay: chunks over which the bonus falls to 0 (0 = constant) */
static const char *pool_path;                      /* --pool: pulled into the daemon before the LOAD */
static int pool_wanted;                            /* the next checkpoint takes the pool with it (a `save-pool` line, the --pool-every schedule) */
static long pool_every;                            /* --pool-every: the pool pushed every this many chunks */
static const char *soft_path;                      /* --soft: the partial distillation's records, one per token of the stream */
static cx_soft *soft;
static long nsoft;

/* ---- the log ---- */
static FILE *lg;
static void logline(const char *fmt, ...)
{
	va_list ap;
	char line[16384];
	va_start(ap, fmt);
	vsnprintf(line, sizeof line, fmt, ap);
	va_end(ap);
	time_t now = time(NULL);
	struct tm tm;
	localtime_r(&now, &tm);
	char ts[32];
	strftime(ts, sizeof ts, "%H:%M:%S", &tm);
	printf("%s %s\n", ts, line);
	fflush(stdout);
	if (lg) { fprintf(lg, "%s %s\n", ts, line); fflush(lg); }
}

/* ---- the token stream and the ring ---- */
static cx_tok *tok;
static long ntok, total_chunks, fed;   /* fed: chunks written into the ring (run-local count) */

/* a slot is u32 words: the chunk's CX_T_L + 1 ids from word 0, the memory dropout's flag at word CX_T_SLOT_MEMOFF; the
 * soft records are copied as u64 words, CX_SOFT_CHUNK_BYTES / 8 = 272 a chunk, every chunk's records 8-byte aligned */
_Static_assert(CX_T_L + 1 <= CX_T_SLOT_MEMOFF && (CX_T_SLOT_MEMOFF + 1) * CX_TOK_BYTES <= CX_T_SLOT_BYTES, "the ids, then the flag, inside one slot");
_Static_assert(CX_SOFT_CHUNK_BYTES % 8 == 0, "a chunk's soft records are whole u64 words");

static void feed(void)
{
	const long consumed = (long)LD64(CX_T_C_CHUNKS);
	while (fed < total_chunks && fed - consumed < (long)RING_AHEAD) {
		volatile uint32_t *slot = (volatile uint32_t *)(part + CX_T_SLOT(fed));
		const cx_tok *src = tok + (start_chunk + fed) * CX_T_L;
		for (int t = 0; t <= CX_T_L; t++) slot[t] = src[t];
		slot[CX_T_SLOT_MEMOFF] = (uint32_t)(ng_path && chunk_dropped(fed));   /* the memory dropout's flag for the chunk */
		if (soft) {   /* the chunk's soft records into its slot of the soft ring, 64-bit stores into the uncached part */
			volatile uint64_t *ss = (volatile uint64_t *)(part + CX_T_SOFT_SLOT(fed));
			const uint64_t *sr = (const uint64_t *)(soft + (start_chunk + fed) * CX_T_L);
			uint64_t sum = 0;
			for (unsigned i = 0; i < CX_SOFT_CHUNK_BYTES / 8; i++) { ss[i] = sr[i]; sum += sr[i]; }
			*(volatile uint64_t *)(part + CX_T_SOFT_SUM(fed)) = sum;   /* the slot's checksum after its words; the card refuses a slot that does not sum to it */
		}
		fed++;
		__atomic_thread_fence(__ATOMIC_RELEASE);
		ST64(CX_T_H_CHUNKS, (uint64_t)fed);
	}
}

/* ---- the n-gram memory: the tables in the shared file, the daemon's row ring served in the batch trainer's lockstep
 * (host/ngserve.c): chunk c's row gradients applied once the card's DMA has landed them (CX_NGR_C_SEQ), then chunk
 * c + 2's rows gathered into the slot the card has finished with and published (CX_NGR_H_SEQ). The chunks are the
 * run's, numbered from 0 as the daemon counts them; their stream positions start at --start ---- */
static ngtables NG;
static pthread_t ng_thread;
static volatile int ng_stop;
static volatile long ng_served;   /* chunks whose row gradients have been applied */
static int tables_wanted;         /* the next checkpoint takes a copy of the tables */
static double now_s(void);

static void *ng_serve(void *arg)
{
	(void)arg;
	for (long c = 0; c < total_chunks; c++) {
		while (LD64(CX_NGR_C_SEQ) < (uint64_t)c + 1) { if (ng_stop) return NULL; usleep(50); }
		if (!chunk_dropped(c)) ng_apply(&NG, tok, ntok, start_chunk + c, CX_T_L, (const float *)(part + CX_NGR_GRAD(c & 1)), (float)row_lr);   /* a dropped chunk sent no gradients: its slot is stale */
		ng_served = c + 1;
		if (c + 2 < total_chunks) {
			ng_gather(&NG, tok, ntok, start_chunk + c + 2, CX_T_L, (float *)(part + CX_NGR_ROWS(c & 1)));
			ST64(CX_NGR_H_SEQ, (uint64_t)c + 3);
		}
	}
	return NULL;
}

/* the ring's first two chunks published and the thread started; the daemon reads CX_T_NG with every chunk */
static int ng_start(void)
{
	ST64(CX_NGR_C_SEQ, 0);
	ST64(CX_NGR_H_SEQ, 0);
	for (long c = 0; c < 2 && c < total_chunks; c++) ng_gather(&NG, tok, ntok, start_chunk + c, CX_T_L, (float *)(part + CX_NGR_ROWS(c & 1)));
	ST64(CX_NGR_H_SEQ, total_chunks < 2 ? (uint64_t)total_chunks : 2);
	ST32(CX_T_NG, 1);
	if (pthread_create(&ng_thread, NULL, ng_serve, NULL)) { logline("coach: cannot start the memory's ring thread"); return -1; }
	return 0;
}

/* the thread ended once the last gradients are applied, the ring word cleared */
static void ng_end(long consumed)
{
	const double t0 = now_s();
	while (ng_served < consumed && now_s() - t0 < 5.0) usleep(1000);
	ng_stop = 1;
	pthread_join(ng_thread, NULL);
	ST32(CX_T_NG, 0);
	if (ng_served < consumed) logline("memory: %ld of %ld chunks' row gradients applied (the rest never landed)", ng_served, consumed);
	if (NG.rows_zeroed) logline("memory: %ld rows found non-finite and zeroed during the run", NG.rows_zeroed);
}

/* the tables written beside the checkpoints: the file synced, then copied whole (2 GiB; the main loop waits) */
static int copy_tables(long chunk)
{
	char path[1024];
	snprintf(path, sizeof path, "%s/tables-%ld.f32", ckdir, chunk);
	const double t0 = now_s();
	if (ng_sync(&NG)) logline("tables: msync %s: %s", ng_path, strerror(errno));
	FILE *f = fopen(path, "wb");
	if (!f || fwrite(NG.map, 1, NG.bytes, f) != NG.bytes || fclose(f)) { logline("tables: cannot write %s: %s", path, strerror(errno)); return -1; }
	logline("tables: %s, %zu bytes, written in %.1f s", path, NG.bytes, now_s() - t0);
	return 0;
}

/* ---- the MoE pool through the stage (lib/cx-proto.h CX_T_POOL): rounds of eleven 2 MiB pages; a push lands in the
 * stage and is copied to the file, a pull is staged from the file first; the feeder runs between rounds ---- */
static void feed(void);

static int pool_wait(uint64_t word)
{
	const double t0 = now_s();
	while (LD64(CX_T_POOL_ACK) != word) { if (now_s() - t0 > 30.0) return -1; usleep(200); }
	return 0;
}

static size_t pool_round_bytes(uint32_t round)
{
	const size_t first = (size_t)round * CX_POOL_ROUND_BYTES;
	return first + CX_POOL_ROUND_BYTES <= CX_EX_POOL_BYTES ? CX_POOL_ROUND_BYTES : CX_EX_POOL_BYTES - first;
}

static int pool_save(const char *path)
{
	const double t0 = now_s();
	FILE *f = fopen(path, "wb");
	if (!f) { logline("pool: cannot write %s: %s", path, strerror(errno)); return -1; }
	for (uint32_t r = 0; r < CX_EX_POOL_ROUNDS; r++) {
		const uint64_t word = CX_T_POOL_WORD(1, r);
		ST64(CX_T_POOL, word);
		if (pool_wait(word)) { logline("pool: round %u not acknowledged: FAILED", r); fclose(f); ST64(CX_T_POOL, 0); return -1; }
		const size_t n = pool_round_bytes(r);
		if (fwrite(part + CX_STAGE, 1, n, f) != n) { logline("pool: write to %s failed: %s", path, strerror(errno)); fclose(f); ST64(CX_T_POOL, 0); return -1; }
		feed();
	}
	ST64(CX_T_POOL, 0);
	if (fclose(f)) { logline("pool: closing %s failed: %s", path, strerror(errno)); return -1; }
	logline("pool: %s, %zu bytes in %u rounds, %.1f s", path, CX_EX_POOL_BYTES, (unsigned)CX_EX_POOL_ROUNDS, now_s() - t0);
	return 0;
}

static int pool_load(const char *path)
{
	const double t0 = now_s();
	FILE *f = fopen(path, "rb");
	if (!f) { logline("pool: cannot read %s: %s", path, strerror(errno)); return -1; }
	for (uint32_t r = 0; r < CX_EX_POOL_ROUNDS; r++) {
		const size_t n = pool_round_bytes(r);
		if (fread(part + CX_STAGE, 1, n, f) != n) { logline("pool: %s is short at round %u", path, r); fclose(f); return -1; }
		__atomic_thread_fence(__ATOMIC_SEQ_CST);
		const uint64_t word = CX_T_POOL_WORD(2, r);
		ST64(CX_T_POOL, word);
		if (pool_wait(word)) { logline("pool: round %u not acknowledged: FAILED", r); fclose(f); ST64(CX_T_POOL, 0); return -1; }
	}
	ST64(CX_T_POOL, 0);
	fclose(f);
	logline("pool: %s pulled into the daemon, %zu bytes in %u rounds, %.1f s", path, CX_EX_POOL_BYTES, (unsigned)CX_EX_POOL_ROUNDS, now_s() - t0);
	return 0;
}

/* ---- the sampler: the client against the live weights, the text scored ---- */
static double rep_hist[3];
static int rep_n, rising_used, halvings;
static double scale = 1.0;
static uint64_t flags;

static void set_flags(uint64_t f) { flags = f; ST64(CX_T_FLAGS, flags); }
static void request_save(void);   /* below, with the checkpoint collector */
static int pool_save(const char *path);   /* below, with the pool transfer */

static void sample(long chunk, int seed)
{
	char cmd[4096];
	snprintf(cmd, sizeof cmd, "%s --live \"%s\" %d --bus 0 --seed %d --vocab %s%s%s%s%s%s%s 2>/dev/null", cortex_bin, prompt, sample_tokens, seed, vocab,
	         orchestra ? " --orchestra " : "", orchestra ? orchestra : "", orchestra ? " --words " : "", orchestra ? words : "",
	         ng_path ? " --ngram " : "", ng_path ? ng_path : "");
	FILE *p = popen(cmd, "r");
	if (!p) { logline("sample: cannot run %s", cmd); return; }
	static char text[65536];
	const size_t n = fread(text, 1, sizeof text - 1, p);
	pclose(p);
	text[n] = 0;
	for (size_t i = 0; i < n; i++) if (text[i] == '\n') text[i] = ' ';
	coh_metrics m;
	coh_score(text, n, &m);
	logline("sample chunk %ld seed %d: hit %.3f hit5 %.3f big %.3f rep4 %.3f sent %.1f words %ld | %s", chunk, seed, m.hit, m.hit5, m.big, m.rep4, m.sent, m.words, text);
	const double prev = rep_n >= 3 ? (rep_hist[0] + rep_hist[1] + rep_hist[2]) / 3 : -1;
	if (prev >= 0 && m.rep4 > prev + rep4_rise && !rising_used && halvings < max_halvings) {
		rising_used = 1;
		scale *= 0.5;
		halvings++;
		ST64(CX_T_SCALE, kbits(scale));
		logline("policy: rep4 %.3f rose above the previous three (mean %.3f) by more than %.2f -> scale %.4f (halving %d)", m.rep4, prev, rep4_rise, scale, halvings);
	}
	rep_hist[rep_n % 3] = m.rep4;
	rep_n++;
}

/* ---- commands appended to the file ---- */
static long cmd_offset;
static int stop_requested, sample_requested;

static void commands(void)
{
	if (!cmd_path) return;
	FILE *f = fopen(cmd_path, "r");
	if (!f) return;
	fseek(f, cmd_offset, SEEK_SET);
	char line[512];
	while (fgets(line, sizeof line, f)) {
		if (line[strlen(line) - 1] != '\n') break;   /* a partial line: next time */
		cmd_offset = ftell(f);
		line[strlen(line) - 1] = 0;
		char *s = line;
		while (*s == ' ' || *s == '\t') s++;
		if (!*s || *s == '#') continue;
		double x;
		long n;
		if (sscanf(s, "scale %lf", &x) == 1 && x > 0) { scale = x; ST64(CX_T_SCALE, kbits(scale)); logline("command: scale %.4f", scale); }
		else if (sscanf(s, "lr %lf", &x) == 1 && x > 0) { lr = x; ST64(CX_T_LR, kbits(lr)); logline("command: base lr %.3e", lr); }
		else if (sscanf(s, "every %ld", &n) == 1 && n > 0) { every = (int)n; ST32(CX_T_EVERY, (uint32_t)every); logline("command: report every %d chunks", every); }
		else if (!strcmp(s, "pause")) { set_flags(flags | 4); logline("command: pause"); }
		else if (!strcmp(s, "resume")) { set_flags(flags & ~4ull); logline("command: resume"); }
		else if (!strcmp(s, "save")) { request_save(); if (ng_path) tables_wanted = 1; logline("command: save requested%s", ng_path ? " (with the tables)" : ""); }
		else if (!strcmp(s, "save-pool")) { if (moe) { pool_wanted = 1; request_save(); } logline("command: pool save %s", moe ? "requested with a checkpoint" : "ignored (no --moe)"); }
		else if (sscanf(s, "bias %lf", &x) == 1 && x >= 0) { moe_bias = x; ST64(CX_M_BIAS, kbits(moe_bias)); logline("command: moe bias %g", moe_bias); }
		else if (sscanf(s, "exlr %lf", &x) == 1 && x >= 0) { expert_lr = x; ST64(CX_M_LR, kbits(expert_lr)); logline("command: expert lr %g", expert_lr); }
		else if (!strcmp(s, "stop")) { stop_requested = 1; logline("command: stop requested"); }
		else if (!strcmp(s, "sample")) { sample_requested = 1; logline("command: sample requested"); }
		else logline("command: unknown line '%s'", s);
	}
	fclose(f);
}

/* ---- the checkpoints: on flag bit 0 the card pushes the live weights into the stage (CX_STAGE_CKPT bytes, the LOAD
 * layout) and the push's last DMA line counts CX_T_SAVES up once the bytes are in host memory; the coach copies the
 * stage to CKDIR/ckpt-CHUNK.bin, clears the flag, and lets the card drop its acknowledgement before the next request
 * (a request made while the previous one is still being acknowledged is kept and sent then) ---- */
static uint64_t saves_seen;   /* CX_T_SAVES as last collected */
static int save_wanted;

static double now_s(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec + ts.tv_nsec * 1e-9; }

static int collect_ckpt(const char *why)
{
	const uint64_t saves = LD64(CX_T_SAVES), chunk = LD64(CX_T_CKPT);
	if (saves == saves_seen) { logline("checkpoint (%s): the card acknowledged without a push (CX_T_SAVES %lu): FAILED", why, (unsigned long)saves); return -1; }
	saves_seen = saves;
	char path[1024];
	snprintf(path, sizeof path, "%s/ckpt-%lu.bin", ckdir, (unsigned long)chunk);
	const double t0 = now_s();
	FILE *f = fopen(path, "wb");
	if (!f || fwrite(part + CX_STAGE, 1, CX_STAGE_CKPT, f) != CX_STAGE_CKPT || fclose(f)) { logline("checkpoint (%s): cannot write %s: %s", why, path, strerror(errno)); return -1; }
	logline("checkpoint (%s): %s, %u bytes, chunk %lu, push %lu, copied in %.0f ms", why, path, (unsigned)CX_STAGE_CKPT, (unsigned long)chunk, (unsigned long)saves, (now_s() - t0) * 1e3);
	if (ng_path) {   /* the gates came with the push; the tables when due */
		snprintf(path, sizeof path, "%s/ckpt-%lu-gates.f32", ckdir, (unsigned long)chunk);
		f = fopen(path, "wb");
		if (!f || fwrite(part + CX_STAGE_GATES, 1, CX_GATES_FLOATS * 4, f) != CX_GATES_FLOATS * 4 || fclose(f)) logline("checkpoint (%s): cannot write %s: %s", why, path, strerror(errno));
		if (tables_wanted || !strcmp(why, "final")) { tables_wanted = 0; copy_tables((long)chunk); }
	}
	if (moe && (pool_wanted || !strcmp(why, "final"))) {   /* the pool of the same chunk, beside the checkpoint */
		pool_wanted = 0;
		snprintf(path, sizeof path, "%s/pool-%lu.f32", ckdir, (unsigned long)chunk);
		pool_save(path);
	}
	return 0;
}

static void request_save(void)
{
	if ((flags & 1) || (LD64(CX_T_ACK) & 1)) save_wanted = 1;
	else set_flags(flags | 1);
}

static void save_handshake(void)
{
	const uint64_t ack = LD64(CX_T_ACK);
	if ((flags & 1) && (ack & 1)) { collect_ckpt("requested"); set_flags(flags & ~1ull); return; }
	if (save_wanted && !(flags & 1) && !(ack & 1)) { save_wanted = 0; set_flags(flags | 1); }
}

/* wait for an acknowledgement bit to be set (set 1) or dropped (set 0) */
static int wait_ack_bit(uint64_t bit, int set, double timeout_s)
{
	const double t0 = now_s();
	while (((LD64(CX_T_ACK) & bit) != 0) != set) { if (now_s() - t0 > timeout_s) return -1; usleep(5000); }
	return 0;
}

/* ---- main ---- */
int main(int argc, char **argv)
{
	if (argc < 2) { fprintf(stderr, "usage: coach TOKENS.u32 [--ckpt CKPT] [--start CHUNK] [--chunks N] [--lr X] [--every N] [--window W] [--min-gain B] [--halvings H] [--sample-every N] [--sample-tokens N] [--save-every N] [--no-stop] [--prompt TEXT] [--orchestra NGC] [--words LIST] [--cmd FILE] [--ckdir DIR] [--paused] [--log FILE] [--cortex PATH] [--vocab FILE] [--ngram TABLES.f32 [--row-lr X] [--gates FILE|ones] [--tables-every N]] [--moe] [--moe-bias X] [--bias-decay N] [--expert-lr X] [--pool FILE] [--pool-every N]\n"); return 2; }
	tokens_path = argv[1];
	for (int i = 2; i < argc; i++) {
		if (!strcmp(argv[i], "--ckpt") && i + 1 < argc) ckpt = argv[++i];
		else if (!strcmp(argv[i], "--start") && i + 1 < argc) start_chunk = atol(argv[++i]);
		else if (!strcmp(argv[i], "--chunks") && i + 1 < argc) max_chunks = atol(argv[++i]);
		else if (!strcmp(argv[i], "--lr") && i + 1 < argc) lr = atof(argv[++i]);
		else if (!strcmp(argv[i], "--every") && i + 1 < argc) every = atoi(argv[++i]);
		else if (!strcmp(argv[i], "--window") && i + 1 < argc) window = atoi(argv[++i]);
		else if (!strcmp(argv[i], "--min-gain") && i + 1 < argc) min_gain = atof(argv[++i]);
		else if (!strcmp(argv[i], "--halvings") && i + 1 < argc) max_halvings = atoi(argv[++i]);
		else if (!strcmp(argv[i], "--sample-every") && i + 1 < argc) sample_every = atol(argv[++i]);
		else if (!strcmp(argv[i], "--save-every") && i + 1 < argc) save_every = atol(argv[++i]);
		else if (!strcmp(argv[i], "--no-stop")) no_stop = 1;
		else if (!strcmp(argv[i], "--sample-tokens") && i + 1 < argc) sample_tokens = atoi(argv[++i]);
		else if (!strcmp(argv[i], "--prompt") && i + 1 < argc) prompt = argv[++i];
		else if (!strcmp(argv[i], "--orchestra") && i + 1 < argc) orchestra = argv[++i];
		else if (!strcmp(argv[i], "--words") && i + 1 < argc) words = argv[++i];
		else if (!strcmp(argv[i], "--cmd") && i + 1 < argc) cmd_path = argv[++i];
		else if (!strcmp(argv[i], "--ckdir") && i + 1 < argc) ckdir = argv[++i];
		else if (!strcmp(argv[i], "--paused")) start_paused = 1;
		else if (!strcmp(argv[i], "--log") && i + 1 < argc) log_path = argv[++i];
		else if (!strcmp(argv[i], "--cortex") && i + 1 < argc) cortex_bin = argv[++i];
		else if (!strcmp(argv[i], "--vocab") && i + 1 < argc) vocab = argv[++i];
		else if (!strcmp(argv[i], "--ngram") && i + 1 < argc) ng_path = argv[++i];
		else if (!strcmp(argv[i], "--row-lr") && i + 1 < argc) row_lr = atof(argv[++i]);
		else if (!strcmp(argv[i], "--ngram-dropout") && i + 1 < argc) ngram_dropout = atof(argv[++i]);
		else if (!strcmp(argv[i], "--gates") && i + 1 < argc) gates_spec = argv[++i];
		else if (!strcmp(argv[i], "--tables-every") && i + 1 < argc) tables_every = atol(argv[++i]);
		else if (!strcmp(argv[i], "--moe")) moe = 1;
		else if (!strcmp(argv[i], "--moe-bias") && i + 1 < argc) { moe_bias = atof(argv[++i]); moe_knobs_given = 1; }
		else if (!strcmp(argv[i], "--bias-decay") && i + 1 < argc) { bias_decay = atol(argv[++i]); moe_knobs_given = 1; }
		else if (!strcmp(argv[i], "--expert-lr") && i + 1 < argc) { expert_lr = atof(argv[++i]); moe_knobs_given = 1; }
		else if (!strcmp(argv[i], "--pool") && i + 1 < argc) pool_path = argv[++i];
		else if (!strcmp(argv[i], "--pool-every") && i + 1 < argc) pool_every = atol(argv[++i]);
		else if (!strcmp(argv[i], "--soft") && i + 1 < argc) soft_path = argv[++i];
		else { fprintf(stderr, "coach: unknown option %s\n", argv[i]); return 2; }
	}
	lg = fopen(log_path, "a");
	if (!lg) { perror(log_path); return 1; }
	if (coh_load(words)) { perror(words); return 1; }
	if (load_vocab(vocab)) return 1;
	for (int id = 0; id < 256 + nmerges; id++) expand(id);
	expand(EOT);
	/* the token stream: u32 ids; a length that is not whole ids or an id outside the vocabulary is refused before anything
	 * reaches the card (a .u16 stream of before 2026-10-28 read as u32 pairs two ids into one value of 65,536 or more:
	 * tools/ids32.c widens one) */
	FILE *f = fopen(tokens_path, "rb");
	if (!f) { perror(tokens_path); return 1; }
	fseek(f, 0, SEEK_END);
	const long tbytes = ftell(f);
	ntok = tbytes / CX_TOK_BYTES;
	fseek(f, 0, SEEK_SET);
	if (tbytes % CX_TOK_BYTES) { fprintf(stderr, "coach: %s: %ld bytes, not whole u32 ids\n", tokens_path, tbytes); return 1; }
	tok = malloc((size_t)ntok * sizeof *tok);
	if (fread(tok, sizeof *tok, (size_t)ntok, f) != (size_t)ntok) { perror(tokens_path); return 1; }
	fclose(f);
	for (long i = 0; i < ntok; i++)
		if (tok[i] >= CX_VOCAB) { fprintf(stderr, "coach: %s: id %u at token %ld is outside the %d-id vocabulary (a .u16 stream?)\n", tokens_path, (unsigned)tok[i], i, CX_VOCAB); return 1; }
	total_chunks = (ntok - 1) / CX_T_L - start_chunk;
	if (max_chunks >= 0 && max_chunks < total_chunks) total_chunks = max_chunks;
	if (total_chunks <= 0) { fprintf(stderr, "coach: nothing to feed (file %ld tokens, start chunk %ld)\n", ntok, start_chunk); return 1; }
	/* the soft targets: one record per token, at least through the run's last chunk; whole 68-byte records, each id CX_SOFT_NONE or in the
	 * vocabulary, every p and w finite (a 48- or 52-byte file of 17 x m records is whole 68-byte records, and with w = 0 its ids read as
	 * CX_SOFT_NONE or 0 while its 0xFFFF id bytes land in p and w as NaN; tools/soft-align.c, mixstream.c and reward-soft.c fill unused
	 * slots with p = 0) */
	if (soft_path) {
		FILE *sf = fopen(soft_path, "rb");
		if (!sf) { perror(soft_path); return 1; }
		fseek(sf, 0, SEEK_END);
		const long sbytes = ftell(sf);
		nsoft = sbytes / (long)CX_SOFT_BYTES;
		fseek(sf, 0, SEEK_SET);
		if (sbytes % (long)CX_SOFT_BYTES) { fprintf(stderr, "coach: %s: %ld bytes, not whole %u-byte records (a 48- or 52-byte file with u16 ids?)\n", soft_path, sbytes, CX_SOFT_BYTES); return 1; }
		const long need = (start_chunk + total_chunks) * CX_T_L, first = start_chunk * CX_T_L;
		if (nsoft < need) { fprintf(stderr, "coach: %s holds %ld records, the run needs %ld (the tokens through chunk %ld)\n", soft_path, nsoft, need, start_chunk + total_chunks - 1); return 1; }
		soft = malloc((size_t)nsoft * CX_SOFT_BYTES);
		if (fread(soft, CX_SOFT_BYTES, (size_t)nsoft, sf) != (size_t)nsoft) { perror(soft_path); return 1; }
		fclose(sf);
		long with = 0;
		for (long i = first; i < need; i++) {
			if (!isfinite(soft[i].w)) { fprintf(stderr, "coach: %s: record %ld holds a non-finite weight (a 48- or 52-byte file with u16 ids?)\n", soft_path, i); return 1; }
			for (int j = 0; j < CX_SOFT_K; j++) {
				if (soft[i].id[j] != CX_SOFT_NONE && soft[i].id[j] >= CX_VOCAB) { fprintf(stderr, "coach: %s: record %ld holds id %u, outside the %d-id vocabulary\n", soft_path, i, (unsigned)soft[i].id[j], CX_VOCAB); return 1; }
				if (!isfinite(soft[i].p[j])) { fprintf(stderr, "coach: %s: record %ld holds a non-finite probability (a 48- or 52-byte file with u16 ids?)\n", soft_path, i); return 1; }
			}
			with += soft[i].id[0] != CX_SOFT_NONE;
		}
		logline("coach: soft targets %s: %ld of %ld tokens of the run carry the teacher's distribution (%.1f %%)", soft_path, with, need - first, 100.0 * (double)with / (double)(need - first));
	}
	/* the part */
	const int fd = open("/dev/shm/phi-hostmem", O_RDWR);
	if (fd < 0) { perror("/dev/shm/phi-hostmem"); return 1; }
	part = mmap(NULL, CX_PART_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, (off_t)CX_PART_MIB << 20);
	close(fd);
	if (part == MAP_FAILED) { perror("mmap"); return 1; }
	if (mkdir(ckdir, 0755) && errno != EEXIST) { perror(ckdir); return 1; }
	if (ng_path && ng_open(&NG, ng_path, 1)) return 1;
	if (moe && !ckpt) { fprintf(stderr, "coach: --moe needs --ckpt (the LOAD fills the pool from it or keeps the pulled one)\n"); return 2; }
	if (pool_path && !moe) { fprintf(stderr, "coach: --pool needs --moe\n"); return 2; }
	if (moe || moe_knobs_given) { ST64(CX_M_BIAS, kbits(moe_bias)); ST64(CX_M_LR, kbits(expert_lr)); }
	saves_seen = LD64(CX_T_SAVES);
	if (cmd_path) { struct stat st; if (!stat(cmd_path, &st)) cmd_offset = st.st_size; }   /* only lines appended from now on */
	char mem[1400] = "";
	if (ng_path) snprintf(mem, sizeof mem, ", n-gram memory %s (row lr %g, gates %s, tables every %ld)", ng_path, row_lr, gates_spec, tables_every);
	if (moe || moe_knobs_given) {
		char decay[64] = "";
		if (bias_decay) snprintf(decay, sizeof decay, " to 0 over %ld chunks", bias_decay);
		snprintf(mem + strlen(mem), sizeof mem - strlen(mem), ", MoE%s (bias %g%s, expert lr %g%s%s)", moe ? " loaded" : " knobs", moe_bias, decay, expert_lr, pool_path ? ", pool " : "", pool_path ? pool_path : "");
	}
	logline("coach: %s chunks %ld..%ld (%ld chunks = %ld tokens), lr %.3e, report every %d, window %d, min gain %.3f, halvings %d, sample every %ld x %d tokens, save every %ld%s%s%s, cmd %s, checkpoints %s/%s",
	        tokens_path, start_chunk, start_chunk + total_chunks - 1, total_chunks, total_chunks * CX_T_L, lr, every, window, min_gain, max_halvings, sample_every, sample_tokens,
	        save_every, no_stop ? ", no stop" : "", orchestra ? ", orchestra " : "", orchestra ? orchestra : "", cmd_path ? cmd_path : "(none)", ckdir, mem);
	/* the initial weights (a saved pool first: the LOAD then keeps it) */
	if (pool_path && pool_load(pool_path)) return 1;
	if (ckpt) {
		char cmd[4096];
		snprintf(cmd, sizeof cmd, "%s --load %s --bus 0 --vocab %s%s%s%s%s", cortex_bin, ckpt, vocab, ng_path ? " --gates " : "", ng_path ? gates_spec : "", moe ? " --moe" : "", pool_path ? " --moe-keep" : "");
		const int rc = system(cmd);
		if (rc) { logline("coach: loading %s failed (rc %d)", ckpt, rc); return 1; }
		logline("coach: %s loaded into the daemon", ckpt);
	}
	/* the run goes live */
	ST32(CX_T_MAGIC, 0);
	usleep(20000);
	ST64(CX_T_H_CHUNKS, 0);
	ST64(CX_T_LR, kbits(lr));
	ST64(CX_T_SCALE, kbits(1.0));
	set_flags(start_paused ? 4 : 0);
	ST32(CX_T_EVERY, (uint32_t)every);
	if (ng_path) { if (ng_start()) return 1; } else ST32(CX_T_NG, 0);
	ST32(CX_T_SOFT, soft ? 1u : 0u);
	const uint64_t report0 = LD64(CX_T_REPORT);
	feed();
	__atomic_thread_fence(__ATOMIC_SEQ_CST);
	ST32(CX_T_MAGIC, CX_T_MAGIC_VAL);
	char dropnote[64] = "";
	if (ng_path && ngram_dropout > 0) snprintf(dropnote, sizeof dropnote, ", memory dropout %.2f", ngram_dropout);
	logline("coach: live (magic set, %ld chunks in the ring%s%s)", fed, ng_path ? ", the memory's rows of chunks 0 and 1 published" : "", dropnote);

	uint64_t last_report = report0;
	double win_sum = 0, prev_mean = 0;
	int win_n = 0, have_prev = 0, neg_windows = 0;
	long next_sample = sample_every ? sample_every : -1, next_save = save_every ? save_every : -1, next_tables = tables_every, next_pool = pool_every, consumed = 0;
	int sample_seed = 1, ng_checked = 0;
	for (;;) {
		usleep(5000);
		feed();
		commands();
		save_handshake();
		consumed = (long)LD64(CX_T_C_CHUNKS);
		if (ng_path && !ng_checked && consumed > 0) {   /* the daemon's word says whether it trains the memory (gates loaded, CX_T_NG seen) */
			ng_checked = 1;
			if (LD32(CX_T_NG_ACK) == 1) logline("coach: the daemon trains the memory with the ring");
			else { logline("coach: the daemon trains WITHOUT the memory (no gates loaded: give --ckpt with --gates): stopping"); stop_requested = 1; }
		}
		const uint64_t report = LD64(CX_T_REPORT);
		if (report != last_report) {
			last_report = report;
			const double loss = kdbl(LD64(CX_T_LOSS)), lr_eff = kdbl(LD64(CX_T_LR_EFF)), tokps = kdbl(LD64(CX_T_TOKPS));
			char moe_note[96] = "";
			if (LD32(CX_T_MOE)) {
				if (bias_decay > 0) {   /* the bonus on its schedule, written for the next chunks */
					const double b = consumed >= bias_decay ? 0.0 : moe_bias * (1.0 - (double)consumed / (double)bias_decay);
					ST64(CX_M_BIAS, kbits(b));
					snprintf(moe_note, sizeof moe_note, "  moe %u experts, bias %.3g", LD32(CX_T_MOE_DISTINCT), b);
				} else snprintf(moe_note, sizeof moe_note, "  moe %u experts", LD32(CX_T_MOE_DISTINCT));
			}
			char ctx_note[48] = "", soft_note[48] = "";   /* the context memory's step time and the soft targets' share, when the daemon reports them */
			const double ctx_us = kdbl(LD64(CX_T_CTX_US));
			if (ctx_us > 0) snprintf(ctx_note, sizeof ctx_note, "  ctx %.0f us/step", ctx_us);
			if (soft) {
				const unsigned long long bad = (unsigned long long)LD64(CX_T_SOFT_BAD);
				const int n = snprintf(soft_note, sizeof soft_note, "  soft %.0f %%", 100.0 * (double)LD64(CX_T_SOFT_N) / ((double)every * CX_T_L));
				if (bad && n > 0 && (size_t)n < sizeof soft_note) snprintf(soft_note + n, sizeof soft_note - (size_t)n, " (bad slots %llu)", bad);
			}
			logline("report chunk %ld  tokens %ld  loss %.4f nats = %.4f bits/token  lr %.3e  scale %.4f  %.0f tok/s  ring %ld ahead%s%s%s", consumed, consumed * CX_T_L, loss, loss / CX_LN2, lr_eff, scale, tokps, fed - consumed, moe_note, ctx_note, soft_note);
			win_sum += loss;
			win_n++;
			if (win_n == window) {
				const double mean = win_sum / win_n;
				if (have_prev) {
					const double gain = (prev_mean - mean) / CX_LN2;
					logline("window: mean %.4f bits/token, gain %.4f over the previous window", mean / CX_LN2, gain);
					neg_windows = gain < 0 ? neg_windows + 1 : 0;
					if (gain < min_gain) {
						if (halvings >= max_halvings) {
							if (neg_windows >= 2 && !no_stop) { stop_requested = 1; logline("policy: at the floor and the loss rose for two windows -> stop"); }
							else logline("policy: plateau at the floor (scale %.4f), continuing%s", scale, neg_windows >= 2 ? " (no-stop)" : "");
						} else {
							scale *= 0.5;
							halvings++;
							ST64(CX_T_SCALE, kbits(scale));
							request_save();
							logline("policy: plateau -> scale %.4f (halving %d), checkpoint requested", scale, halvings);
						}
					}
				}
				prev_mean = mean;
				have_prev = 1;
				win_sum = 0;
				win_n = 0;
			}
		}
		if (next_save > 0 && consumed >= next_save) {
			next_save = (consumed / save_every + 1) * save_every;
			if (ng_path && consumed >= next_tables) { next_tables = (consumed / tables_every + 1) * tables_every; tables_wanted = 1; }
			request_save();
			logline("checkpoint requested at chunk %ld (every %ld chunks)%s", consumed, save_every, tables_wanted ? ", with the tables" : "");
		}
		if (moe && pool_every > 0 && consumed >= next_pool) {   /* the pool with a checkpoint of the same chunk */
			next_pool = (consumed / pool_every + 1) * pool_every;
			pool_wanted = 1;
			request_save();
			logline("pool requested with the next checkpoint at chunk %ld (every %ld chunks)", consumed, pool_every);
		}
		if ((next_sample > 0 && consumed >= next_sample) || sample_requested) {
			if (next_sample > 0 && consumed >= next_sample) next_sample += sample_every;
			sample_requested = 0;
			sample(consumed, sample_seed++);
		}
		if (stop_requested || (fed >= total_chunks && consumed >= total_chunks)) {
			if (fed >= total_chunks && consumed >= total_chunks) logline("coach: the stream is exhausted at chunk %ld", consumed);
			set_flags(flags | 2);
			if (wait_ack_bit(2, 1, 30)) logline("coach: the card did not acknowledge the stop within 30 s");
			consumed = (long)LD64(CX_T_C_CHUNKS);
			if (ng_path) ng_end(consumed);   /* the last row gradients applied before the final checkpoint takes the tables */
			if (flags & 1) {   /* a save still in flight: collect it first */
				if (wait_ack_bit(1, 1, 60)) logline("coach: the card did not acknowledge the pending checkpoint within 60 s");
				else collect_ckpt("requested");
				set_flags(flags & ~1ull);
			}
			if (wait_ack_bit(1, 0, 10)) logline("coach: the card did not drop the checkpoint acknowledgement within 10 s");
			set_flags(flags | 1);
			if (wait_ack_bit(1, 1, 60)) logline("coach: the card did not acknowledge the final checkpoint within 60 s");
			else collect_ckpt("final");   /* with --moe the pool of the same chunk follows it */
			ST32(CX_T_MAGIC, 0);
			ST32(CX_T_SOFT, 0);
			set_flags(0);
			logline("coach: run ended at chunk %ld (%ld tokens), %d halvings, final scale %.4f", consumed, consumed * CX_T_L, halvings, scale);
			break;
		}
	}
	return 0;
}
