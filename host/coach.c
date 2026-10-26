/*
 * coach: the host side of live training (card/cortexd.c, docs/LIVE.md). Feeds the daemon's training ring from a
 * token file, reads its reports, samples the live model through the mailbox (as any client would: sampled decoding,
 * the orchestrator) and scores the samples, applies the plateau policy to the learning rate, and takes commands from
 * a file that a person (or an assistant watching the log) appends to. Nothing the coach writes and nothing the model
 * generates is training data: the gradient comes from the token file only.
 *
 *   coach TOKENS.u16 [options]
 *     --ckpt CKPT          load this checkpoint into the daemon first (a new run for the Adam state)
 *     --start CHUNK        the first chunk of the token file to feed (default 0)
 *     --chunks N           feed at most N chunks (default: to the end of the file)
 *     --lr X               base learning rate (default 3e-3)
 *     --every N            the daemon reports every N chunks (default 100)
 *     --window W           policy: compare the mean loss of W reports with the previous W (default 10)
 *     --min-gain B         policy: a window gain under B bits/token is a plateau (default 0.01)
 *     --halvings H         policy: at most H halvings of the scale (default 3)
 *     --sample-every N     sample the live model every N chunks (default 2000, 0 = never)
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
#include <errno.h>
#include <fcntl.h>
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
static long start_chunk, max_chunks = -1, sample_every = 2000;
static int sample_tokens = 128, every = 100, window = 10, max_halvings = 3, start_paused;
static double lr = 3e-3, min_gain = 0.01, rep4_rise = 0.1;

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
static uint16_t *tok;
static long ntok, total_chunks, fed;   /* fed: chunks written into the ring (run-local count) */

static void feed(void)
{
	const long consumed = (long)LD64(CX_T_C_CHUNKS);
	while (fed < total_chunks && fed - consumed < (long)RING_AHEAD) {
		volatile uint16_t *slot = (volatile uint16_t *)(part + CX_T_SLOT(fed));
		const uint16_t *src = tok + (start_chunk + fed) * CX_T_L;
		for (int t = 0; t <= CX_T_L; t++) slot[t] = src[t];
		fed++;
		__atomic_thread_fence(__ATOMIC_RELEASE);
		ST64(CX_T_H_CHUNKS, (uint64_t)fed);
	}
}

/* ---- the sampler: the client against the live weights, the text scored ---- */
static double rep_hist[3];
static int rep_n, rising_used, halvings;
static double scale = 1.0;
static uint64_t flags;

static void set_flags(uint64_t f) { flags = f; ST64(CX_T_FLAGS, flags); }
static void request_save(void);   /* below, with the checkpoint collector */

static void sample(long chunk, int seed)
{
	char cmd[4096];
	snprintf(cmd, sizeof cmd, "%s --live \"%s\" %d --bus 0 --seed %d --vocab %s%s%s%s%s 2>/dev/null", cortex_bin, prompt, sample_tokens, seed, vocab,
	         orchestra ? " --orchestra " : "", orchestra ? orchestra : "", orchestra ? " --words " : "", orchestra ? words : "");
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
		else if (!strcmp(s, "save")) { request_save(); logline("command: save requested"); }
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
	if (argc < 2) { fprintf(stderr, "usage: coach TOKENS.u16 [--ckpt CKPT] [--start CHUNK] [--chunks N] [--lr X] [--every N] [--window W] [--min-gain B] [--halvings H] [--sample-every N] [--sample-tokens N] [--prompt TEXT] [--orchestra NGC] [--words LIST] [--cmd FILE] [--ckdir DIR] [--paused] [--log FILE] [--cortex PATH] [--vocab FILE]\n"); return 2; }
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
		else { fprintf(stderr, "coach: unknown option %s\n", argv[i]); return 2; }
	}
	lg = fopen(log_path, "a");
	if (!lg) { perror(log_path); return 1; }
	if (coh_load(words)) { perror(words); return 1; }
	if (load_vocab(vocab)) return 1;
	for (int id = 0; id < 256 + nmerges; id++) expand(id);
	expand(EOT);
	/* the token stream */
	FILE *f = fopen(tokens_path, "rb");
	if (!f) { perror(tokens_path); return 1; }
	fseek(f, 0, SEEK_END);
	ntok = ftell(f) / 2;
	fseek(f, 0, SEEK_SET);
	tok = malloc((size_t)ntok * 2);
	if (fread(tok, 2, (size_t)ntok, f) != (size_t)ntok) { perror(tokens_path); return 1; }
	fclose(f);
	total_chunks = (ntok - 1) / CX_T_L - start_chunk;
	if (max_chunks >= 0 && max_chunks < total_chunks) total_chunks = max_chunks;
	if (total_chunks <= 0) { fprintf(stderr, "coach: nothing to feed (file %ld tokens, start chunk %ld)\n", ntok, start_chunk); return 1; }
	/* the part */
	const int fd = open("/dev/shm/phi-hostmem", O_RDWR);
	if (fd < 0) { perror("/dev/shm/phi-hostmem"); return 1; }
	part = mmap(NULL, CX_PART_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, (off_t)CX_PART_MIB << 20);
	close(fd);
	if (part == MAP_FAILED) { perror("mmap"); return 1; }
	if (mkdir(ckdir, 0755) && errno != EEXIST) { perror(ckdir); return 1; }
	saves_seen = LD64(CX_T_SAVES);
	if (cmd_path) { struct stat st; if (!stat(cmd_path, &st)) cmd_offset = st.st_size; }   /* only lines appended from now on */
	logline("coach: %s chunks %ld..%ld (%ld chunks = %ld tokens), lr %.3e, report every %d, window %d, min gain %.3f, halvings %d, sample every %ld x %d tokens%s%s, cmd %s, checkpoints %s/",
	        tokens_path, start_chunk, start_chunk + total_chunks - 1, total_chunks, total_chunks * CX_T_L, lr, every, window, min_gain, max_halvings, sample_every, sample_tokens,
	        orchestra ? ", orchestra " : "", orchestra ? orchestra : "", cmd_path ? cmd_path : "(none)", ckdir);
	/* the initial weights */
	if (ckpt) {
		char cmd[4096];
		snprintf(cmd, sizeof cmd, "%s --load %s --bus 0 --vocab %s", cortex_bin, ckpt, vocab);
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
	const uint64_t report0 = LD64(CX_T_REPORT);
	feed();
	__atomic_thread_fence(__ATOMIC_SEQ_CST);
	ST32(CX_T_MAGIC, CX_T_MAGIC_VAL);
	logline("coach: live (magic set, %ld chunks in the ring)", fed);

	uint64_t last_report = report0;
	double win_sum = 0, prev_mean = 0;
	int win_n = 0, have_prev = 0, neg_windows = 0;
	long next_sample = sample_every ? sample_every : -1, consumed = 0;
	int sample_seed = 1;
	for (;;) {
		usleep(5000);
		feed();
		commands();
		save_handshake();
		consumed = (long)LD64(CX_T_C_CHUNKS);
		const uint64_t report = LD64(CX_T_REPORT);
		if (report != last_report) {
			last_report = report;
			const double loss = kdbl(LD64(CX_T_LOSS)), lr_eff = kdbl(LD64(CX_T_LR_EFF)), tokps = kdbl(LD64(CX_T_TOKPS));
			logline("report chunk %ld  tokens %ld  loss %.4f nats = %.4f bits/token  lr %.3e  scale %.4f  %.0f tok/s  ring %ld ahead", consumed, consumed * CX_T_L, loss, loss / CX_LN2, lr_eff, scale, tokps, fed - consumed);
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
							if (neg_windows >= 2) { stop_requested = 1; logline("policy: at the floor and the loss rose for two windows -> stop"); }
							else logline("policy: plateau at the floor (scale %.4f), continuing", scale);
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
			if (flags & 1) {   /* a save still in flight: collect it first */
				if (wait_ack_bit(1, 1, 60)) logline("coach: the card did not acknowledge the pending checkpoint within 60 s");
				else collect_ckpt("requested");
				set_flags(flags & ~1ull);
			}
			if (wait_ack_bit(1, 0, 10)) logline("coach: the card did not drop the checkpoint acknowledgement within 10 s");
			set_flags(flags | 1);
			if (wait_ack_bit(1, 1, 60)) logline("coach: the card did not acknowledge the final checkpoint within 60 s");
			else collect_ckpt("final");
			ST32(CX_T_MAGIC, 0);
			set_flags(0);
			logline("coach: run ended at chunk %ld (%ld tokens), %d halvings, final scale %.4f", consumed, consumed * CX_T_L, halvings, scale);
			break;
		}
	}
	return 0;
}
