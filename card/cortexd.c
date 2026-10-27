/*
 * cortexd: the Intel Phi Cortex daemon on the card: generation (Stage 5), the n-gram memory at generation (MoE stage
 * 1) and live training (2026-10-26: "the model can run inference while training, and get actively interacted with
 * while weights are tuning"). usage: cortexd [PART_MIB]   (env CX_DMA_CHAN channel, CX_BUS training bus 0|1)
 *
 * 120 column threads (card/cluster.h) run for the daemon's life. Thread 0 decides each round what the team does next:
 *   a session's command   LOAD, RESET, BUS, STEP, EXIT through the mailbox (lib/cx-proto.h), as before; STEP runs on
 *                         the INFERENCE context (its own fast weights T and bus per column, its own step count),
 *                         so a client generates from the live weights while they train
 *   a training chunk      when the coach (host/coach.c) has published CX_T_MAGIC and chunks in the ring, and no
 *                         command is pending: CX_LMAX steps forward on the TRAINING context (the hot-block T, its
 *                         own bus), the chunk backward with Adam (card/cluster.h), the bus wrapped, the loss
 *                         accumulated; the knobs (base rate, scale, save, stop, pause) are read once per chunk and
 *                         a report (loss, rate, tokens/s) written every CX_T_EVERY chunks
 *   nothing               a short pause, then the next decision
 * Inference steps only happen between chunks (the per-step saves of the forward belong to the chunk's backward), and
 * a session's commands keep the card as long as they arrive within CX_BURST_S of each other: a generating client runs
 * at full speed and training resumes the moment it stops; an idle session waits at most one chunk (about 25 ms).
 * LOAD replaces the live weights and zeroes every training buffer (a new run starts as a fresh batch-trainer process
 * does). The n-gram memory: a loaded checkpoint's gates serve the steps of a session whose client writes the rows
 * (CX_OP_NGRAM); a run whose coach serves the row ring (CX_T_NG) trains the memory as well: each chunk's rows come
 * in by DMA (cl_ng_begin), its row gradients go back (cl_bwd_chunk) and the gates train with Adam, exactly as the
 * batch trainer with host/ngserve.c does, and a chunk is taken only once its rows are published, so the team never
 * waits in the ring; otherwise a chunk trains the plain model. The Adam schedule is cl_adam_schedule of
 * card/cluster.h, shared with the batch trainer, so the trajectory is bit-identical to `cluster-gate run` on the same
 * tokens. Nothing a client generates is training data: the training stream is the coach's token file. When there is
 * nothing to do the team naps 100 us per round instead of spinning (a spinning idle slowed a co-running card job
 * about 50x and burns the card's full power).
 *
 * DMA: a card-owned channel (2 unless CX_DMA_CHAN names another, card/cdma.h): the checkpoint pulled in on LOAD and
 * pushed out on the coach's save flag (the same bytes through the same stage area of the part; the push's last line
 * writes CX_T_SAVES in host memory after the bytes, so the coach copies a complete checkpoint), the logits pushed out
 * per STEP (the ACK word is the DMA's status), the memory rows pulled per STEP when the memory is on. A DMA write into
 * GDDR invalidates every core's L2 copy (tools/dma-coh.c), so ring slots are reused without eviction, and a DMA read
 * sees the cores' latest stores (the logits gate). Nothing is written to the card's filesystem: the checkpoints live
 * on the host.
 */
#ifndef CX_MOE
#error "cortexd is the MoE build: -DCX_MOE=1 (build.sh)"
#endif
#include "cluster.h"
#include "cdma.h"
#include "../lib/cx-proto.h"

#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#define CX_SESSION_GRACE_S 5.0   /* a session with no first command within this long is abandoned (a dangling host session) */
#define CX_BURST_S 0.003         /* a session keeps the card while its commands arrive within this of each other */
#define CX_HEART_S 0.1           /* CX_C_HEART counted up this often by thread 0: the host's liveness word (the TUI) */
#define CX_IDLE_NAP_US 100       /* every thread's nap per round when there is nothing to do */

enum { ACT_IDLE = 100, ACT_TRAIN, ACT_SAVE, ACT_QUIT };   /* the team's actions besides the mailbox ops */
_Static_assert(CX_T_L == CX_LMAX, "the live ring's chunk length is the card's chunk length");
_Static_assert(CX_T_WARMUP == CL_WARMUP, "the protocol's warm-up is the trainers' warm-up");

static cluster C __attribute__((aligned(64)));
static volatile uint8_t *part;
static cdma dma;
static float *logits_buf;
static float *ngpage, *nggates, *nggrad;   /* one huge page: 64 row slots of 1 KiB (the training chunks' two 32 KiB slots), the gates at 64 KiB, the two
                                              32 KiB row-gradient slots at 128 KiB */
static uint64_t logits_phys, cg_phys, etb_phys[1], prm_phys[CL_N], ngpage_phys, nggates_phys, nggrad_phys;
static int session_ng = 1;        /* the session's client writes its steps' rows (CX_OP_NGRAM) */
static int ng_train_acked = -1;   /* CX_T_NG_ACK as last written */
/* MoE stage 2 (card/moe.h, built in): the pool of CX_EX_N experts in GDDR huge pages, filled from a checkpoint's MLP blocks
 * by a LOAD with arg bit 1 (expert e = column e mod 120, sparse upcycling); the heads then route every step (training and
 * inference) with the bonus at CX_M_BIAS and train the experts by SGD at CX_M_LR; the pool lives for the daemon's life
 * (a LOAD without the bit returns to the dense model; a LOAD with it refills the pool) */
static float *pool;
static int moe_loaded;            /* the loaded model routes through the pool */
static uint64_t part_mib = CX_PART_MIB;
static uint64_t host_part;        /* the part's address for the DMA engine */
static float *T_train[CL_N], *T_inf[CL_N], *bus_train, *bus_inf;
static int train_bus_on;
static char *arena[CL_N];         /* each thread's 4 MiB: the column in the first 2 MiB, the training buffers after */
static size_t arena_used[CL_N];
static char *sh_base, *sh_zero;   /* the shared block; zeroed from sh_zero (after cg and its Adam moments) on LOAD */
static size_t sh_used;
static float *ng_loaded;          /* the loaded checkpoint's n-gram gates, or NULL: C.ng for a session's commands only */

static inline uint32_t rd32(uint32_t o) { return *(volatile uint32_t *)(part + o); }
static inline uint64_t rd64(uint32_t o) { return *(volatile uint64_t *)(part + o); }
static inline void wr32(uint32_t o, uint32_t v) { *(volatile uint32_t *)(part + o) = v; }
static inline void wr64(uint32_t o, uint64_t v) { *(volatile uint64_t *)(part + o) = v; }
static inline double rdf(uint32_t o) { const uint64_t u = rd64(o); double d; memcpy(&d, &u, 8); return d; }
static inline void wrf(uint32_t o, double d) { uint64_t u; memcpy(&u, &d, 8); wr64(o, u); }

static uint64_t phys_of(const void *p)
{
	int fd = open("/proc/self/pagemap", O_RDONLY);
	if (fd < 0) return 0;
	uint64_t e = 0;
	const off_t at = (off_t)((uintptr_t)p / 4096) * 8;
	const ssize_t r = pread(fd, &e, 8, at);
	close(fd);
	if (r != 8 || !(e >> 63)) return 0;
	return (e & ((1ULL << 55) - 1)) * 4096 + ((uintptr_t)p & 4095);
}

static void *huge(size_t bytes)
{
	void *p = mmap(NULL, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | MAP_POPULATE, -1, 0);
	if (p == MAP_FAILED) { perror("huge pages"); exit(1); }
	mlock(p, bytes);
	return p;
}

/* ---- per-thread arenas (as tools/cluster-gate.c setup), the inference context's T, the physical address of the weights ---- */
static void setup(void *arg, int k)
{
	(void)arg;
	char *a = huge(4u << 20);
	cx_train *tr = &C.tr[k];
	memset(tr, 0, sizeof *tr);
	tr->L = CX_LMAX;
	const size_t hot = (CX_NPARAM + CX_NT + CX_NACT) * 4, lines = ((2u << 20) - hot) / 64, off = ((size_t)k * 104729u % lines) * 64;
	tr->col.prm = (float *)(a + off);
	tr->col.T = tr->col.prm + CX_NPARAM;
	tr->col.act = tr->col.T + CX_NT;
	size_t o = 2u << 20;
#define TAKE(n) (o += ((n) + 63) & ~(size_t)63, (float *)(a + o - (((n) + 63) & ~(size_t)63)))
	tr->acts = TAKE((size_t)CX_LMAX * CX_NACT * 4);
	tr->Ts = TAKE((size_t)CX_LMAX * CX_NT * 4);
	tr->hc = TAKE((size_t)CX_LMAX * CX_DU * 4);
	tr->dhc = TAKE((size_t)CX_LMAX * CX_DU * 4);
	tr->loss = TAKE(CX_LMAX * 4);
	tr->ebuf = TAKE(CX_DE * 4);
	tr->dT = TAKE(CX_NT * 4);
	tr->gsave = TAKE((size_t)CX_LMAX * CX_NG * 4);
	tr->gprm = TAKE(3 * ((CX_NPARAM + 15) & ~15) * 4);   /* gradient, then Adam m and v */
	tr->gcg = TAKE(CX_DU * 4);
	tr->gb1 = TAKE(CX_DH * 4);
	tr->ggain = TAKE(CX_DX * 4);
	tr->gwg = TAKE(CX_DH * 4);
	tr->gbg = TAKE(16 * 4);
	tr->tmp = TAKE(1024 * 4);
	C.hs[k] = TAKE((size_t)CX_LMAX * CX_DU * 4);
	C.xsh[k] = TAKE((size_t)CX_LMAX * 16 * 4);
	C.hcp[k] = TAKE((size_t)CX_LMAX * CX_DU * 4);
	C.dhc[k] = TAKE((size_t)CX_LMAX * CX_DU * 4);
	C.lg[k] = TAKE((size_t)CX_LMAX * CL_MAXBLK * 16 * 4);
	C.tgtrow[k] = TAKE(CX_DE * 4);
	tr->ngr = TAKE((size_t)CX_LMAX * CX_NG_ORDERS * CX_DE * 4);
	T_inf[k] = TAKE(CX_NT * 4);
	T_train[k] = tr->col.T;
	tr->ng = NULL;
	tr->moe = NULL;
	C.moe[k] = (cx_moe){ .pool = pool, .keys = C.keys, .bias = 0.0f, .home_a = CX_EX_HOME_A(k), .home_b = CX_EX_HOME_B(k), .route = C.route + (long)k * CX_LMAX };
	if (o > (4u << 20)) { fprintf(stderr, "cortexd: arena overflow %zu\n", o); exit(1); }
	arena[k] = a;
	arena_used[k] = o;
	tr->ET = C.ETB;
	tr->ES = C.ES;
	tr->GET = C.GET;
	tr->cg = C.cg;
	cx_col_init(tr->col.prm, k);
	memset(tr->col.T, 0, CX_NT * 4);
	memset(tr->col.act, 0, CX_NACT * 4);
	memset(T_inf[k], 0, CX_NT * 4);
	memset(tr->gprm, 0, 3 * ((CX_NPARAM + 15) & ~15) * 4);
	cl_blocks(&C, k);
	C.epoch[k] = 0;
	prm_phys[k] = phys_of(tr->col.prm);
}

/* ---- the checkpoint: CX_STAGE_CKPT bytes in the stage area of the part, the 120 columns' weights, cg, then the
 * embedding (lib/cx-proto.h). LOAD pulls them in (thread 0); the coach's save flag pushes the live weights out over
 * the same lines in the other direction, and the last line's status write counts CX_T_SAVES up in host memory once
 * the bytes have landed ---- */
static int load_ckpt(uint32_t arg)
{
	uint64_t src = host_part + CX_STAGE, seq = 0;
	for (int k = 0; k < CL_N; k++) {
		if (!prm_phys[k]) { fprintf(stderr, "cortexd: no physical address for column %d\n", k); return -1; }
		seq = cdma_submit(&dma, src, prm_phys[k], CX_NPARAM * 4, 0, 0);
		if (!seq) return -1;
		src += CX_NPARAM * 4;
	}
	seq = cdma_submit(&dma, src, cg_phys, CX_DU * 4, 0, 0);
	if (!seq) return -1;
	src += CX_DU * 4;
	for (uint32_t done = 0; done < (2u << 20); done += (1u << 20) - 64) {   /* one DMA line copies at most 1 MiB - 64 B */
		const uint32_t n = (2u << 20) - done < (1u << 20) - 64 ? (2u << 20) - done : (1u << 20) - 64;
		seq = cdma_submit(&dma, src + done, etb_phys[0] + done, n, 0, 0);
		if (!seq) return -1;
	}
	src += 2u << 20;
	if (arg & 1) {
		seq = cdma_submit(&dma, src, nggates_phys, CX_NG_ORDERS * CX_DE * 4, 0, 0);
		if (!seq) return -1;
	}
	if (cdma_wait(&dma, seq, 10.0)) { fprintf(stderr, "cortexd: checkpoint DMA timed out\n"); return -1; }
	for (int v = 0; v < CX_VOCAB; v++) for (int c = 0; c < CX_DE; c++) C.ES[(long)v * CX_DE + c] = C.ETB[CX_ETB(v, c)];
	ng_loaded = arg & 1 ? nggates : NULL;
	moe_loaded = (arg & 2) != 0;
	return 0;
}

/* every thread's share of the pool from the loaded blocks (after a MoE LOAD): expert e = the MLP block of column e mod 120 */
static void moe_fill(int k)
{
	for (size_t e = (size_t)k; e < CX_EX_N; e += CL_N) memcpy(pool + e * CX_EX_FLOATS, C.tr[e % CL_N].col.prm + CX_EX_SLOT, CX_EX_BYTES);
}

/* the routing bonus and the experts' step from the host words, into every head's context (thread 0, before a chunk or
 * a session command) */
static void moe_knobs(void)
{
	const float bias = (float)rdf(CX_M_BIAS);
	C.moe_lr = (float)rdf(CX_M_LR);
	for (int k = 0; k < CL_N; k++) C.moe[k].bias = bias;
}

/* experts updated at least once since the fill */
static uint32_t moe_distinct(void)
{
	uint32_t n = 0;
	for (size_t e = 0; e < CX_EX_N; e++) n += C.moe_used[e] != 0;
	return n;
}

static uint64_t saves_done;   /* checkpoints pushed since the daemon started (CX_T_SAVES) */

static int push_ckpt(void)
{
	uint64_t dst = host_part + CX_STAGE, seq = 0;
	for (int k = 0; k < CL_N; k++) {
		seq = cdma_submit(&dma, prm_phys[k], dst, CX_NPARAM * 4, 0, 0);
		if (!seq) return -1;
		dst += CX_NPARAM * 4;
	}
	seq = cdma_submit(&dma, cg_phys, dst, CX_DU * 4, 0, 0);
	if (!seq) return -1;
	dst += CX_DU * 4;
	for (uint32_t done = 0; done < (2u << 20); done += (1u << 20) - 64) {
		const uint32_t n = (2u << 20) - done < (1u << 20) - 64 ? (2u << 20) - done : (1u << 20) - 64;
		seq = cdma_submit(&dma, etb_phys[0] + done, dst + done, n, 0, 0);
		if (!seq) return -1;
	}
	/* the gates last (the memory's, or whatever the page holds when it is off); this line's status write counts the save */
	seq = cdma_submit(&dma, nggates_phys, host_part + CX_STAGE_GATES, CX_NG_ORDERS * CX_DE * 4, host_part + CX_T_SAVES, saves_done + 1);
	if (!seq) return -1;
	if (cdma_wait(&dma, seq, 10.0)) { fprintf(stderr, "cortexd: checkpoint DMA out timed out\n"); return -1; }
	saves_done++;
	return 0;
}

/* ---- thread 0's state: the session, the inference context, the live run ---- */
/* The round's action and argument, double-buffered by round parity: thread 0 writes slot r & 1 before round r's
 * opening barrier and every thread reads it after that barrier. A single word was a race: an idle round has no second
 * barrier, so thread 0 could nap, decide round r + 1 and overwrite the word before a preempted thread had read it for
 * round r; that thread then took the next action one barrier early and stayed one barrier out of step (wrong votes
 * and sums, losses off, NaN within chunks; measured under 240 busy-loop hogs, 2026-10-26 17:3x). Thread 0 reaches
 * round r + 2's decision only after round r + 1's barrier, which every thread passes after its read of slot r & 1. */
static volatile uint32_t act_op[2], act_arg[2];
static int act_slot;
static uint64_t done, first;
static double t_session0;
static int session_open;
static long inf_step;                /* inference steps since RESET: the chunk slot is inf_step % 32, the memory ring slot inf_step % 64 */
static int inf_bus_on = 1;
static long ng_steps, lg_steps, served;
static double ng_dma_s, ng_dma_max, lg_dma_s, lg_dma_max;

static uint32_t chunk_ids[CX_LMAX + 1] __attribute__((aligned(64)));
static long tchunks;                 /* chunks trained since the live run began (the Adam step count) */
static double lr_base, lr_scale, lr_eff, loss_acc, t_window;
static long loss_n;
static uint64_t flags_acked;
static int train_state;              /* CX_T_STATE */
static uint32_t report_every = 100;

static void set_state(int s) { if (train_state != s) { train_state = s; wr32(CX_T_STATE, (uint32_t)s); } }

/* thread 0, memory on: the step's rows from the part into ring slot inf_step % 64, before the team starts the forward */
static int ng_fetch(void)
{
	const double t0 = cx_now();
	const uint64_t seq = cdma_submit(&dma, host_part + CX_NGS_ROWS(inf_step), ngpage_phys + (uint64_t)(inf_step % CX_NGS_STEPS) * CX_NGS_BYTES, CX_NGS_BYTES, 0, 0);
	if (!seq || cdma_wait(&dma, seq, 2.0)) { fprintf(stderr, "cortexd: row DMA failed at step %ld\n", inf_step); return -1; }
	const double dt = cx_now() - t0;
	ng_dma_s += dt;
	if (dt > ng_dma_max) ng_dma_max = dt;
	ng_steps++;
	C.chunk = inf_step / CX_LMAX;   /* cl_fwd_step reads slot chunk % 2, step inf_step % 32 */
	return 0;
}

static void session_report(const char *how)
{
	printf("cortexd: session %llu %s (%ld steps served", (unsigned long long)rd64(CX_C_SESSION), how, served);
	if (lg_steps) printf("; logits DMA mean %.1f us, max %.1f", lg_dma_s * 1e6 / lg_steps, lg_dma_max * 1e6);
	if (ng_steps) printf("; n-gram rows %ld, row DMA mean %.1f us, max %.1f", ng_steps, ng_dma_s * 1e6 / ng_steps, ng_dma_max * 1e6);
	printf(")\n");
	fflush(stdout);
	served = ng_steps = lg_steps = 0;
	ng_dma_s = ng_dma_max = lg_dma_s = lg_dma_max = 0;
}

/* a session's next command, if one arrives within the burst window; 1 when an action was set */
static int poll_session(void)
{
	if (!session_open) {
		if (rd32(CX_H_MAGIC) != CX_MAGIC || rd64(CX_H_SESSION) == rd64(CX_C_SESSION)) return 0;
		done = first = CX_CMD_SEQ(rd64(CX_H_CMD));
		wr64(CX_C_ACK, done);
		cx_fence();
		wr64(CX_C_SESSION, rd64(CX_H_SESSION));
		wr32(CX_C_STATE, 1);
		t_session0 = cx_now();
		session_open = 1;
		session_ng = 1;
	}
	const double t0 = cx_now();
	for (;;) {
		const uint64_t c = rd64(CX_H_CMD), seq = CX_CMD_SEQ(c);
		if (seq == CX_CMD_QUIT) { wr32(CX_C_STATE, 3); session_open = 0; session_report("ended"); return 0; }
		if (seq != done) {
			act_op[act_slot] = rd32(CX_H_CMD_OP);
			act_arg[act_slot] = CX_CMD_ARG(c);
			done = seq;
			C.bus = bus_inf;
			C.bus_on = inf_bus_on;
			C.ng = session_ng ? ng_loaded : NULL;
			if (moe_loaded) moe_knobs();
			if (act_op[act_slot] == CX_OP_STEP && C.ng && ng_fetch()) wr32(CX_C_STATE, 2);
			return 1;
		}
		if (done == first && cx_now() - t_session0 > CX_SESSION_GRACE_S) {
			printf("cortexd: no command within %.0f s, session abandoned\n", CX_SESSION_GRACE_S);
			fflush(stdout);
			wr32(CX_C_STATE, 3);
			session_open = 0;
			return 0;
		}
		if (cx_now() - t0 > CX_BURST_S) return 0;
		cx_relax();
	}
}

/* the live run: a chunk to train, a save to do, or nothing; 1 when an action was set */
static int poll_training(void)
{
	if (rd32(CX_T_MAGIC) != CX_T_MAGIC_VAL) {   /* no run, or the run ended: the counters start afresh for the next one */
		if (train_state) { set_state(0); tchunks = 0; flags_acked = 0; wr64(CX_T_C_CHUNKS, 0); wr64(CX_T_REPORT, 0); wr64(CX_T_ACK, 0); ng_train_acked = -1; wr32(CX_T_NG_ACK, 0); printf("cortexd: live: run ended\n"); fflush(stdout); }
		return 0;
	}
	const uint64_t flags = rd64(CX_T_FLAGS);
	if ((flags & 1) && !(flags_acked & 1)) { act_op[act_slot] = ACT_SAVE; return 1; }
	if (!(flags & 1) && (flags_acked & 1)) { flags_acked &= ~1ull; wr64(CX_T_ACK, flags_acked); }
	if (flags & 2) { if (!(flags_acked & 2)) { flags_acked |= 2; wr64(CX_T_ACK, flags_acked); printf("cortexd: live: stop flag at chunk %ld\n", tchunks); fflush(stdout); } set_state(4); return 0; }
	if (flags & 4) { set_state(2); return 0; }
	if (rd64(CX_T_H_CHUNKS) <= (uint64_t)tchunks) { set_state(3); return 0; }
	const int ng_run = rd32(CX_T_NG) != 0 && ng_loaded != NULL;   /* the memory trains: the coach serves the ring and gates were loaded */
	if (ng_run && rd64(CX_NGR_H_SEQ) <= (uint64_t)tchunks) { set_state(3); return 0; }   /* the chunk's rows are not published yet */
	if (ng_train_acked != ng_run) { ng_train_acked = ng_run; wr32(CX_T_NG_ACK, (uint32_t)ng_run); }
	const volatile uint16_t *slot = (const volatile uint16_t *)(part + CX_T_SLOT(tchunks));
	for (int t = 0; t <= CX_LMAX; t++) chunk_ids[t] = slot[t];
	lr_base = rdf(CX_T_LR);
	lr_scale = rdf(CX_T_SCALE);
	report_every = rd32(CX_T_EVERY) ? rd32(CX_T_EVERY) : 100;
	if (!tchunks) { loss_acc = 0; loss_n = 0; t_window = cx_now(); }
	C.bus = bus_train;
	C.bus_on = train_bus_on;
	C.ng = ng_run ? nggates : NULL;   /* with the memory: the chunk's rows over the ring (cl_ng_begin), its row gradients back (cl_bwd_chunk); without: the plain model */
	if (moe_loaded) moe_knobs();
	act_op[act_slot] = ACT_TRAIN;
	set_state(1);
	return 1;
}

static void decide(uint64_t round)
{
	static double t_heart;
	static uint64_t heart;
	act_slot = (int)(round & 1);
	const double now = cx_now();
	if (now - t_heart >= CX_HEART_S) { t_heart = now; wr64(CX_C_HEART, ++heart); }
	if (poll_session()) return;
	if (poll_training()) return;
	act_op[act_slot] = ACT_IDLE;
}

/* ---- the team's actions ---- */

/* one training chunk on the training context; the Adam schedule is the batch trainer's (cl_adam_schedule) */
static void train_chunk(int k)
{
	cx_train *tr = &C.tr[k];
	const float b1 = 0.9f, b2 = 0.95f;
	const uint32_t *id = chunk_ids;
	tr->ng = C.ng;
	cl_ng_begin(&C, k, tchunks);   /* the memory's rows of this chunk (ring slot tchunks & 1) into GDDR; nothing without the memory */
	for (int t = 0; t < CX_LMAX; t++) cl_fwd_step(&C, k, t, id[t], id[t + 1], id[t + 1]);
	float lr_k, lr_t, eps_t;
	cl_adam_schedule(tchunks + 1, lr_base * lr_scale, &lr_k, &lr_t, &eps_t);
	cl_bwd_chunk(&C, k, id, b1, b2, lr_t, eps_t);   /* ends with barrier G */
	if (!k) {
		cx_copy(C.bus, C.bus + (long)CX_LMAX * CL_BUS, CL_BUS);
		double lsum = 0;
		for (int t = 0; t < CX_LMAX; t++) lsum += C.loss[t];
		loss_acc += lsum / CX_LMAX;
		loss_n++;
		lr_eff = lr_k;
		tchunks++;
		wr64(CX_T_C_CHUNKS, (uint64_t)tchunks);
		if (loss_n == (long)report_every) {
			const double now = cx_now(), tokps = (double)loss_n * CX_LMAX / (now - t_window);
			wrf(CX_T_LOSS, loss_acc / loss_n);
			wrf(CX_T_LR_EFF, lr_eff);
			wrf(CX_T_TOKPS, tokps);
			const uint32_t distinct = moe_loaded ? moe_distinct() : 0;
			if (moe_loaded) wr32(CX_T_MOE_DISTINCT, distinct);
			cx_fence();
			wr64(CX_T_REPORT, rd64(CX_T_REPORT) + 1);
			printf("live: chunk %7ld  tokens %10ld  loss %.4f nats = %.4f bits/token  lr %.2e  %.0f tok/s", tchunks, tchunks * (long)CX_LMAX, loss_acc / loss_n, loss_acc / loss_n / CX_LN2, lr_eff, tokps);
			if (moe_loaded) printf("  moe: bias %g, expert lr %g, %u experts updated", (double)C.moe[0].bias, (double)C.moe_lr, distinct);
			printf("\n");
			fflush(stdout);
			loss_acc = 0;
			loss_n = 0;
			t_window = now;
		}
	}
	cl_wait(&C, k);   /* H: the wrapped bus and the counters visible */
}

/* the live weights into the host's stage by DMA (thread 0); every thread is at the barrier, so no chunk or step is
 * in flight and the DMA reads the latest weights; then the chunk count and the acknowledgement of the flag */
static void save_ckpt(int k)
{
	if (moe_loaded) {   /* the file's MLP region of column k is its home expert k: into the slot before the push */
		cx_moe_fetch(&C.moe[k], (uint32_t)k, C.tr[k].col.prm + CX_EX_SLOT);
		cl_wait(&C, k);
	}
	if (!k) {
		cx_fence();
		const double t0 = cx_now();
		const int rc = push_ckpt();
		printf("cortexd: live: checkpoint %llu at chunk %ld %s (%.1f ms)\n", (unsigned long long)saves_done + (rc ? 1 : 0), tchunks, rc ? "FAILED" : "pushed into the host stage", (cx_now() - t0) * 1e3);
		fflush(stdout);
		if (!rc) wr64(CX_T_CKPT, (uint64_t)tchunks);
		flags_acked |= 1;
		wr64(CX_T_ACK, flags_acked);
	}
	cl_wait(&C, k);
}

static void live(void *arg, int k)
{
	(void)arg;
	for (uint64_t round = 0;; round++) {
		if (!k) decide(round);
		cl_wait(&C, k);
		const uint32_t op = act_op[round & 1], a = act_arg[round & 1];
		switch (op) {
		case ACT_IDLE:
			usleep(CX_IDLE_NAP_US);
			break;
		case ACT_TRAIN:
			train_chunk(k);
			break;
		case ACT_SAVE:
			save_ckpt(k);
			break;
		case CX_OP_LOAD:   /* new weights: a new run, every training buffer as a fresh batch-trainer process has it */
			if (!k) { if (load_ckpt(a)) wr32(CX_C_STATE, 2); }
			cl_wait(&C, k);
			if (moe_loaded) moe_fill(k);   /* the pool from the loaded blocks; the heads route from now on */
			C.tr[k].moe = moe_loaded ? &C.moe[k] : NULL;
			if (!k) { C.moe_on = moe_loaded; memset(C.moe_used, 0, (size_t)CX_EX_N * 4); wr32(CX_T_MOE, (uint32_t)moe_loaded); wr32(CX_T_MOE_DISTINCT, 0); }
			C.tr[k].ng = NULL;
			memset(arena[k] + (2u << 20), 0, arena_used[k] - (2u << 20));   /* per-step saves, gradients, Adam moments, both contexts' T */
			memset(C.tr[k].col.T, 0, CX_NT * 4);
			memset(C.tr[k].col.act, 0, CX_NACT * 4);
			if (!k) { memset(C.m_et, 0, 2u << 20); memset(C.v_et, 0, 2u << 20); memset(C.GET, 0, 2u << 20); memset(C.m_cg, 0, CX_DU * 4); memset(C.v_cg, 0, CX_DU * 4); memset(sh_zero, 0, sh_used - (size_t)(sh_zero - sh_base)); cx_ex_keys_init(C.keys); tchunks = 0; inf_step = 0; }
			cl_wait(&C, k);
			break;
		case CX_OP_RESET:
			memset(T_inf[k], 0, CX_NT * 4);
			if (!k) { memset(bus_inf, 0, (size_t)(CX_LMAX + 1) * CL_BUS * 4); inf_step = 0; }
			cl_wait(&C, k);
			break;
		case CX_OP_BUS:
			if (!k) inf_bus_on = (int)a;
			cl_wait(&C, k);
			break;
		case CX_OP_STEP: {
			const int t = (int)(inf_step % CX_LMAX);
			C.tr[k].col.T = T_inf[k];
			C.tr[k].ng = C.ng;
			cl_fwd_step(&C, k, t, a, a, 0);
			C.tr[k].ng = NULL;
			C.tr[k].col.T = T_train[k];
			cl_wait(&C, k);   /* every thread's logits are in the buffer */
			if (!k) {
				cx_fence();
				const double d0 = cx_now();
				const uint64_t seq = cdma_submit(&dma, logits_phys, host_part + CX_LOGITS, CX_VOCAB * 4, host_part + CX_C_ACK, done);
				if (!seq || cdma_wait(&dma, seq, 2.0)) wr32(CX_C_STATE, 2);
				const double dt = cx_now() - d0;
				lg_dma_s += dt;
				if (dt > lg_dma_max) lg_dma_max = dt;
				lg_steps++;
				served++;
				if (t + 1 == CX_LMAX) cx_copy(C.bus, C.bus + (long)CX_LMAX * CL_BUS, CL_BUS);   /* the inference bus wraps with its chunk */
				inf_step++;
			}
			cl_wait(&C, k);
			continue;   /* the ACK went with the DMA */
		}
		case CX_OP_NGRAM:
			if (!k) session_ng = (int)a;
			cl_wait(&C, k);
			break;
		case CX_OP_EXIT:
			if (!k) { cx_fence(); wr64(CX_C_ACK, done); session_report("ended, exiting"); }
			return;
		default:
			break;
		}
		if (!k && op < ACT_IDLE) { cx_fence(); wr64(CX_C_ACK, done); }
	}
}

/* the card-owned DMA channel: 2 unless CX_DMA_CHAN names another (a channel left enabled by a killed owner stays
 * unusable until the card reboots; the daemon then runs on a different one) */
static int dma_channel(void)
{
	const char *e = getenv("CX_DMA_CHAN");
	return e && *e ? atoi(e) : 2;
}

int main(int argc, char **argv)
{
	if (argc > 1) part_mib = strtoull(argv[1], NULL, 10);
	train_bus_on = getenv("CX_BUS") ? atoi(getenv("CX_BUS")) : 0;
	C.L = CX_LMAX;
	C.bus_on = 1;
	C.ETB = huge(2u << 20);
	etb_phys[0] = phys_of(C.ETB);
	C.GET = huge(2u << 20);
	C.m_et = huge(2u << 20);
	C.v_et = huge(2u << 20);
	C.ES = huge(2u << 20);
	char *sh = huge(8u << 20);
	size_t o = 0;
#define STAKE(n) (o += ((n) + 63) & ~(size_t)63, (float *)(sh + o - (((n) + 63) & ~(size_t)63)))
	C.cg = STAKE(CX_DU * 4);
	C.m_cg = STAKE(CX_DU * 4);
	C.v_cg = STAKE(CX_DU * 4);
	bus_train = STAKE((size_t)(CX_LMAX + 1) * CL_BUS * 4);
	bus_inf = STAKE((size_t)(CX_LMAX + 1) * CL_BUS * 4);
	C.votes = STAKE(2 * CL_N * CX_DU * 4);
	C.gsum = STAKE(2 * CL_GROUPS * CX_DU * 4);
	C.pairs = STAKE(2 * CL_N * 16 * 4);
	C.dhcp = STAKE((size_t)CL_N * CX_LMAX * CX_DU * 4);
	C.gdhc = STAKE((size_t)CL_GROUPS * CX_LMAX * CX_DU * 4);
	C.dep = STAKE((size_t)CL_N * CX_LMAX * CX_DU * 4);
	C.loss = STAKE(CX_LMAX * 4);
	C.argmax = (volatile uint32_t *)STAKE(2 * 16 * 4);
	logits_buf = STAKE(CX_VOCAB * 4);
	C.m_ng = STAKE(CX_NG_ORDERS * CX_DE * 4);   /* the memory's gate moments, gate gradient and owner-summed input gradient: zeroed with the block on LOAD */
	C.v_ng = STAKE(CX_NG_ORDERS * CX_DE * 4);
	C.gng = STAKE(CX_NG_ORDERS * CX_DE * 4);
	C.desum = STAKE((size_t)CX_LMAX * CX_DU * 4);
	C.keys = STAKE(CX_EX_KEYFLOATS * 4);         /* MoE: the product keys, every head's route per step, the usage counts */
	C.route = (uint32_t *)STAKE((size_t)CL_N * CX_LMAX * 4);
	C.moe_used = (uint32_t *)STAKE((size_t)CX_EX_N * 4);
	if (o > (8u << 20)) { fprintf(stderr, "cortexd: shared overflow %zu\n", o); return 1; }
	memset(sh, 0, o);
	cx_ex_keys_init(C.keys);
	pool = huge(((size_t)CX_EX_N * CX_EX_BYTES + (2u << 20) - 1) & ~((size_t)(2u << 20) - 1));   /* 4 GiB of GDDR for the experts */
	C.pool = pool;
	C.moe_on = 0;
	sh_base = sh;
	sh_zero = (char *)bus_train;
	sh_used = o;
	memset(C.GET, 0, 2u << 20);
	memset(C.m_et, 0, 2u << 20);
	memset(C.v_et, 0, 2u << 20);
	C.bus = bus_inf;
	C.logits_out = logits_buf;
	cg_phys = phys_of(C.cg);
	logits_phys = phys_of(logits_buf);
	if (!etb_phys[0] || !cg_phys || !logits_phys) { fprintf(stderr, "cortexd: physical addresses unavailable\n"); return 1; }
	for (int i = 0; i < CX_DU; i++) C.cg[i] = 1.0f;
	/* the n-gram ring buffer and the gates: one huge page the DMA engine fills (zeroed by huge_alloc) */
	ngpage = huge_alloc(2u << 20, &ngpage_phys, 1);
	if (!ngpage) { fprintf(stderr, "cortexd: no memory for the n-gram ring\n"); return 1; }
	nggates = ngpage + CX_NGS_STEPS * CX_NGS_BYTES / 4;
	nggates_phys = ngpage_phys + CX_NGS_STEPS * CX_NGS_BYTES;
	C.ngbuf = ngpage;
	C.ngbuf_phys = ngpage_phys;
	nggrad = ngpage + (128u << 10) / 4;
	nggrad_phys = ngpage_phys + (128u << 10);
	C.nggrad = nggrad;
	C.nggrad_phys = nggrad_phys;
	C.ng = NULL;

	const int chan = dma_channel();
	if (cdma_open(&dma, chan)) return 1;
	host_part = dma.region_addr + (part_mib << 20);
	C.ng_host_rows = host_part + CX_NGR_ROWS(0);   /* the training row ring and its gradient slots, as the batch trainer has them */
	C.ng_host_grad = host_part + CX_NGR_GRAD(0);
	C.ng_host_cseq = host_part + CX_NGR_C_SEQ;
	int fd = open("/dev/phihost", O_RDWR | O_SYNC);
	part = fd < 0 ? MAP_FAILED : mmap(NULL, CX_PART_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, (off_t)(part_mib << 20));
	if (part == MAP_FAILED) { perror("/dev/phihost"); return 1; }
	C.part = part;
	C.dma = &dma;
	wr32(CX_C_STATE, 0);
	wr64(CX_T_C_CHUNKS, 0);
	wr64(CX_T_REPORT, 0);
	wr64(CX_T_ACK, 0);
	wr64(CX_T_CKPT, 0);
	wr64(CX_T_SAVES, 0);
	wr32(CX_T_STATE, 0);
	wr32(CX_T_NG_ACK, 0);
	wr32(CX_T_MOE, 0);
	wr32(CX_T_MOE_DISTINCT, 0);
	wrf(CX_M_BIAS, (double)CX_EX_HOME_BIAS);   /* the knobs' defaults until a coach or a load writes them */
	wrf(CX_M_LR, (double)CX_EX_LR);

	cx_barrier_init(&C.bar, CL_N, 2, CX_CORE_REL);
	cx_team_run(CL_N, setup, NULL);
	printf("cortexd: %d columns (pin failures %d, thread 0 cpu %d, thread %d cpu %d), part %llu MiB, DMA channel %d (region 0x%llx, %llu MiB), n-gram step ring at part + %u KiB, row ring at part + %u MiB, live training ring at part + %u MiB (training bus %s), MoE pool %u experts x %u B in GDDR (routes after a LOAD with arg bit 1); serving\n",
	       CL_N, cx_pin_fail, cx_found_cpu[0], CL_N - 1, cx_found_cpu[CL_N - 1], (unsigned long long)part_mib, chan, (unsigned long long)dma.region_addr, (unsigned long long)(dma.region_size >> 20), CX_NGS_BASE >> 10, CX_NGR_BASE >> 20, CX_T_RING >> 20, train_bus_on ? "on" : "off", CX_EX_N, CX_EX_BYTES);
	fflush(stdout);
	cx_team_run(CL_N, live, NULL);
	wr32(CX_T_STATE, 0);
	cdma_close(&dma);
	return 0;
}
