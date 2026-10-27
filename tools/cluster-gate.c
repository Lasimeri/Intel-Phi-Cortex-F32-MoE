/*
 * cluster-gate: Stage 4, card side. usage: cluster-gate fwd|bwd TOKENS.u16 OUTDIR L WARM [DUMP_COL...]
 *                                          cluster-gate run TOKENS.u16 OUTDIR L NTOK LR VALID.u16 NVAL [EVERY]
 *
 * 120 columns on 120 threads (card/cluster.h), the state from zero. WARM (a multiple of L) forward-only steps, then
 * one chunk of L steps (fwd) or that chunk's backward too (bwd: no Adam, gradients dumped). Dumps in OUTDIR:
 * cl-loss.f32 (L), cl-hc.f32 (L x 128), cl-u<k>.f32 (L x 160: u, bc, g) per dumped column; bwd adds cl-gprm<k>.f32,
 * cl-gcg.f32, cl-get.f32 (ETB layout). The host side: tools/ctrain-check.c.
 *
 * Env: CX_NG=1 trains with the n-gram memory served by host/ngserve.c over the DMA ring (MoE stage 1; the gates
 * dumped as cl-gates.f32). The -DCX_MOE=1 build (cluster-gate-moe) runs the MoE model of stage 2: the pool (4 GiB
 * of huge pages) filled from the columns' MLPs (expert e = column e mod 120) after the warm-up or the checkpoint,
 * the heads routing through it with the home bonus CX_MOE_BIAS (default CX_EX_HOME_BIAS) and the experts' step
 * CX_MOE_LR (default CX_EX_LR); CX_MOE=0 keeps the dense model in that build. Dumps added: cl-route.f32 (u32 route
 * [k * CX_LMAX + t] after the forward) and cl-ex<k>.f32 (the expert of step 0 of each dumped head after the chunk);
 * a run's checkpoint holds the home experts in the MLP regions and the run prints the experts touched.
 */
#include "../card/cluster.h"
#include <fcntl.h>

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/mman.h>

static cluster C __attribute__((aligned(64)));
#include <linux/perf_event.h>
#include <sys/syscall.h>
#include <unistd.h>
static uint64_t pmu_count[CL_N];
static int pmu_open(uint64_t config)
{
	struct perf_event_attr pa;
	memset(&pa, 0, sizeof pa);
	pa.size = sizeof pa;
	pa.type = PERF_TYPE_RAW;
	pa.config = config;
	pa.exclude_kernel = 1;
	pa.exclude_hv = 1;
	return (int)syscall(__NR_perf_event_open, &pa, 0, -1, -1, 0);
}
static uint64_t pmu_read(int fd)
{
	uint64_t v = 0;
	return fd >= 0 && read(fd, &v, sizeof v) == sizeof v ? v : 0;
}
static int L, warm, bwd, runm, ndump, dump_list[CL_N];
static uint32_t *ids;
static cdma ngdma;   /* MoE stage 1: the n-gram ring DMA channel */
#ifdef CX_MOE
/* MoE stage 2 (card/moe.h): the build runs the heads with the pool unless CX_MOE=0 (then the dense columns, the same
 * arithmetic as the plain build); CX_MOE_BIAS the home pair's bonus (default CX_EX_HOME_BIAS), CX_MOE_LR the experts'
 * SGD step (default CX_EX_LR). The pool is filled by sparse upcycling from the columns' MLPs once they are set. */
static float moe_bias;
static uint32_t *moe_used;   /* per expert: (step, expert) updates over the run */

static void moe_fill(void *arg, int k)
{
	(void)arg;
	for (size_t e = (size_t)k; e < CX_EX_N; e += CL_N) memcpy(C.pool + e * CX_EX_FLOATS, C.tr[e % CL_N].col.prm + CX_EX_SLOT, CX_EX_BYTES);
}

static void moe_report(void)
{
	long distinct = 0, total = 0, home = 0;
	uint32_t mx = 0;
	for (size_t e = 0; e < CX_EX_N; e++) {
		if (!moe_used[e]) continue;
		distinct++;
		total += moe_used[e];
		if (moe_used[e] > mx) mx = moe_used[e];
		if (e < CL_N) home += moe_used[e];
	}
	printf("moe: %ld of %u experts updated, %ld (step, expert) updates, the busiest %u, the 120 home experts %.1f %% of them\n", distinct, CX_EX_N, total, mx, total ? 100.0 * home / total : 0.0);
}
#endif

/* the card-owned DMA channel of the n-gram ring: 2 unless CX_DMA_CHAN names another */
static int dma_channel(void)
{
	const char *e = getenv("CX_DMA_CHAN");
	return e && *e ? atoi(e) : 2;
}
static long ntok;
static const char *outdir;

static void *huge(size_t bytes)
{
	void *p = mmap(NULL, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | MAP_POPULATE, -1, 0);
	if (p == MAP_FAILED) { perror("huge pages"); exit(1); }
	return p;
}

static int dump(const char *name, int k, const void *p, size_t bytes)
{
	char path[512];
	if (k >= 0) snprintf(path, sizeof path, "%s/%s%d.f32", outdir, name, k);
	else snprintf(path, sizeof path, "%s/%s.f32", outdir, name);
	FILE *f = fopen(path, "wb");
	if (!f || fwrite(p, 1, bytes, f) != bytes || fclose(f)) { perror(path); return -1; }
	return 0;
}

/* per-thread arenas: the hot column at a staggered offset in page A, the streams in page B */
static void setup(void *arg, int k)
{
	(void)arg;
	char *a = huge(4u << 20);
	cx_train *tr = &C.tr[k];
	memset(tr, 0, sizeof *tr);
	tr->L = L;
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
	tr->ng = C.ng;
	tr->ngr = TAKE((size_t)CX_LMAX * CX_NG_ORDERS * CX_DE * 4);
	tr->moe = NULL;
#ifdef CX_MOE
	if (C.moe_on) {
		C.moe[k] = (cx_moe){ .pool = C.pool, .keys = C.keys, .bias = moe_bias, .home_a = CX_EX_HOME_A(k), .home_b = CX_EX_HOME_B(k), .route = C.route + (long)k * CX_LMAX };
		tr->moe = &C.moe[k];
	}
#endif
	if (o > (4u << 20)) { fprintf(stderr, "cluster-gate: arena overflow %zu\n", o); exit(1); }
	tr->ET = C.ETB;
	tr->ES = C.ES;
	tr->GET = C.GET;
	tr->cg = C.cg;
	tr->logits = NULL;
	cx_col_init(tr->col.prm, k);
	memset(tr->col.T, 0, CX_NT * 4);
	memset(tr->col.act, 0, CX_NACT * 4);
	memset(tr->gprm, 0, 3 * ((CX_NPARAM + 15) & ~15) * 4);
	cl_blocks(&C, k);
	C.epoch[k] = 0;
}

static int is_dumped(int k)
{
	for (int i = 0; i < ndump; i++) if (dump_list[i] == k) return 1;
	return 0;
}

static double wall[4];

static void run_check(void *arg, int k)
{
	(void)arg;
	cx_gather_prefetch(C.ETB, ids[0]);
	for (int ch = 0; ch < warm / L; ch++) {
		cl_ng_begin(&C, k, ch);
		const uint32_t *id = ids + (long)ch * L;
		for (int t = 0; t < L; t++) cl_fwd_step(&C, k, t, id[t], id[t + 1], id[t + 1]);
		cl_wait(&C, k);
		if (!k) cx_copy(C.bus, C.bus + (long)L * CL_BUS, CL_BUS);
		cl_wait(&C, k);
	}
	const uint32_t *id = ids + (long)warm;
	cl_ng_begin(&C, k, warm / L);
	const int pfd = pmu_open(getenv("CX_PMU") ? strtoull(getenv("CX_PMU"), NULL, 16) : 0x10cb);
	const uint64_t p0 = pmu_read(pfd);
	if (!k) wall[0] = cx_now();
	for (int t = 0; t < L; t++) cl_fwd_step(&C, k, t, id[t], id[t + 1], id[t + 1]);
	cl_wait(&C, k);
	if (!k) wall[1] = cx_now();
	pmu_count[k] = pmu_read(pfd) - p0;
	close(pfd);
#ifdef CX_MOE
	if (C.moe_on && !k) dump("cl-route", -1, C.route, (size_t)CL_N * CX_LMAX * 4);   /* u32: route[k * CX_LMAX + t] */
#endif
	if (is_dumped(k)) {
		float *o = malloc((size_t)L * CX_DUMP * 4);
		for (int t = 0; t < L; t++) {
			const float *A = C.tr[k].acts + (long)t * CX_NACT;
			memcpy(o + (long)t * CX_DUMP, A + CX_A_U, CX_DU * 4);
			memcpy(o + (long)t * CX_DUMP + 128, A + CX_A_BC, 16 * 4);
			memcpy(o + (long)t * CX_DUMP + 144, A + CX_A_G, 16 * 4);
		}
		dump("cl-u", k, o, (size_t)L * CX_DUMP * 4);
		free(o);
	}
	if (!bwd) return;
	if (!k) wall[2] = cx_now();
	cl_bwd_chunk(&C, k, id, 0.9f, 0.95f, -1.0f, 1e-8f);   /* lr < 0: gradients only, no Adam, GET kept */
	if (!k) wall[3] = cx_now();
	if (is_dumped(k)) dump("cl-gprm", k, C.tr[k].gprm, CX_NPARAM * 4);
	if (!k) { dump("cl-gcg", -1, C.tr[0].gcg, CX_DU * 4); if (C.ng) dump("cl-gng", -1, C.gng, CX_NG_ORDERS * CX_DE * 4); }
#ifdef CX_MOE
	if (C.moe_on && is_dumped(k)) dump("cl-ex", k, C.pool + (size_t)C.route[(long)k * CX_LMAX] * CX_EX_FLOATS, CX_EX_BYTES);   /* the expert of step 0 after the chunk's updates */
#endif
}

/* ---- run mode: pretraining of the whole cluster ---- */
static long run_ntok, run_nval;
static int ckpt_every;
static int write_ckpt(const char *path);
static double run_lr;
static int run_every, bus_on = 1;
static uint32_t *vids;
static float *stage;   /* 120 x CX_NPARAM for the checkpoint */
static double rwall[8];
static uint64_t rcyc[4];

static void run_train(void *arg, int k)
{
	(void)arg;
	const int L = C.L;
	const long nchunks = (run_ntok - 1) / L;
	const float b1 = 0.9f, b2 = 0.95f;
	double acc = 0;
	if (!k) rwall[0] = rwall[1] = cx_now();
	cx_gather_prefetch(C.ETB, ids[0]);
	for (long ch = 0; ch < nchunks; ch++) {
		const uint32_t *id = ids + ch * L;
		cl_ng_begin(&C, k, ch);
		const uint64_t c0 = cx_rdtsc();
		for (int t = 0; t < L; t++) cl_fwd_step(&C, k, t, id[t], id[t + 1], id[t + 1]);
		const uint64_t c1 = cx_rdtsc();
		float lr_k, lr_t, eps_t;
		cl_adam_schedule(ch + 1, run_lr, &lr_k, &lr_t, &eps_t);
		cl_bwd_chunk(&C, k, id, b1, b2, lr_t, eps_t);   /* ends with barrier G */
		const uint64_t c2 = cx_rdtsc();
		if (!k) {
			cx_copy(C.bus, C.bus + (long)L * CL_BUS, CL_BUS);
			rcyc[0] += c1 - c0;
			rcyc[1] += c2 - c1;
			double lsum = 0;
			for (int t = 0; t < L; t++) lsum += C.loss[t];
			acc += lsum / L;
			if ((ch + 1) % run_every == 0 || ch + 1 == nchunks) {
				const double now = cx_now(), n = (ch + 1) % run_every ? (ch + 1) % run_every : run_every;
				printf("chunk %6ld  tokens %9ld  loss %.4f nats = %.4f bits/token  lr %.2e  %.0f tok/s  (us/step: fwd+vote+head %.0f, bwd+grads+adam %.0f)\n",
				       ch + 1, (ch + 1) * (long)L, acc / n, acc / n / CX_LN2, lr_k, n * L / (now - rwall[1]), rcyc[0] / 1238.093 / (n * L), rcyc[1] / 1238.093 / (n * L));
				fflush(stdout);
				acc = 0;
				rwall[1] = now;
				rcyc[0] = rcyc[1] = 0;
			}
		}
		if (ckpt_every && (ch + 1) % ckpt_every == 0 && ch + 1 < nchunks) {
			memcpy(stage + (long)k * CX_NPARAM, C.tr[k].col.prm, CX_NPARAM * 4);
#ifdef CX_MOE
			if (C.moe_on) memcpy(stage + (long)k * CX_NPARAM + CX_EX_SLOT, C.pool + (size_t)k * CX_EX_FLOATS, CX_EX_BYTES);   /* the home expert in the file's MLP region */
#endif
			cl_wait(&C, k);
			if (!k) { char p[512]; snprintf(p, sizeof p, "%s/ckpt-%ld.bin", outdir, ch + 1); write_ckpt(p); printf("checkpoint %s\n", p); fflush(stdout); }
		}
		cl_wait(&C, k);   /* H: bus[0] visible */
	}
	if (!k) rwall[2] = cx_now();
	/* validation: forward only from zero state and a zero bus */
	memset(C.tr[k].col.T, 0, CX_NT * 4);
	if (!k) memset(C.bus, 0, (size_t)(CX_LMAX + 1) * CL_BUS * 4);
	cl_wait(&C, k);
	const long vchunks = (run_nval - 1) / L;
	double vsum = 0;
	for (long ch = 0; ch < vchunks; ch++) {
		const uint32_t *id = vids + ch * L;
		cl_ng_begin(&C, k, run_ntok / L + ch);   /* the rows of this validation chunk; nothing without the memory */
		for (int t = 0; t < L; t++) cl_fwd_step(&C, k, t, id[t], id[t + 1], id[t + 1]);
		cl_wait(&C, k);
		if (!k) {
			for (int t = 0; t < L; t++) vsum += C.loss[t];
			cx_copy(C.bus, C.bus + (long)L * CL_BUS, CL_BUS);
			if (C.ng) cl_ng_done(&C);
		}
		cl_wait(&C, k);
	}
	if (!k) {
		rwall[3] = cx_now();
		printf("valid: %ld tokens, loss %.4f nats = %.4f bits/token = %.4f bits/byte (3.961 B/token); train %.1f s, valid %.1f s\n", vchunks * L,
		       vsum / (vchunks * L), vsum / (vchunks * L) / CX_LN2, vsum / (vchunks * L) / CX_LN2 / 3.961, rwall[2] - rwall[0], rwall[3] - rwall[2]);
	}
	memcpy(stage + (long)k * CX_NPARAM, C.tr[k].col.prm, CX_NPARAM * 4);
#ifdef CX_MOE
	if (C.moe_on) memcpy(stage + (long)k * CX_NPARAM + CX_EX_SLOT, C.pool + (size_t)k * CX_EX_FLOATS, CX_EX_BYTES);
#endif
	cl_wait(&C, k);
}

static int load_ckpt(const char *path);
static int write_ckpt(const char *path)
{
	FILE *f = fopen(path, "wb");
	if (!f || fwrite(stage, 4, (size_t)CL_N * CX_NPARAM, f) != (size_t)CL_N * CX_NPARAM || fwrite(C.cg, 4, CX_DU, f) != CX_DU ||
	    fwrite(C.ETB, 4, (size_t)CX_VOCAB * CX_DE, f) != (size_t)CX_VOCAB * CX_DE || fclose(f)) { perror(path); return -1; }
	return 0;
}

static int run_mode(int argc, char **argv)
{
	/* run TOKENS.u16 OUTDIR L NTOK LR VALID.u16 NVAL [EVERY] */
	if (argc < 9) { fprintf(stderr, "usage: cluster-gate run TOKENS.u16 OUTDIR L NTOK LR VALID.u16 NVAL [EVERY]\n"); return 2; }
	run_ntok = ntok;   /* tokens already read: NTOK + 1 */
	run_lr = atof(argv[6]);
	run_every = argc > 9 ? atoi(argv[9]) : 100;
	run_nval = atol(argv[8]) + 1;
	if (getenv("CX_BUS")) bus_on = atoi(getenv("CX_BUS"));
	C.bus_on = bus_on;
	FILE *f = fopen(argv[7], "rb");
	if (!f) { perror(argv[7]); return 1; }
	uint16_t *vt = malloc((size_t)run_nval * 2);
	run_nval = (long)fread(vt, 2, (size_t)run_nval, f);
	fclose(f);
	vids = malloc((size_t)run_nval * 4);
	for (long t = 0; t < run_nval; t++) vids[t] = vt[t];
	stage = huge((size_t)CL_N * CX_NPARAM * 4 + (2u << 20));
	if (getenv("CX_CKPT_EVERY")) ckpt_every = atoi(getenv("CX_CKPT_EVERY"));
	if (getenv("CX_INIT")) { if (load_ckpt(getenv("CX_INIT"))) return 1; printf("run: weights, cg and embedding loaded from %s\n", getenv("CX_INIT")); }
	printf("run: %ld tokens = %ld chunks of %d, lr %g (warm-up 50 chunks), b1 0.9 b2 0.95, bus %s, valid %ld tokens\n", run_ntok - 1, (run_ntok - 1) / C.L, C.L, run_lr,
	       bus_on ? "on" : "off", run_nval - 1);
#ifdef CX_MOE
	if (C.moe_on) {
		const double f0 = cx_now();
		cx_team_run(CL_N, moe_fill, NULL);
		printf("moe: on, pool %u experts x %u B = %zu B filled from the columns' MLPs in %.2f s (expert e = column e mod %d), keys seeded, home bias %g, expert lr %g\n",
		       CX_EX_N, CX_EX_BYTES, (size_t)CX_EX_N * CX_EX_BYTES, cx_now() - f0, CL_N, moe_bias, C.moe_lr);
	}
#endif
	fflush(stdout);
	cx_team_run(CL_N, run_train, NULL);
#ifdef CX_MOE
	if (C.moe_on) moe_report();
#endif
	char path[512];
	snprintf(path, sizeof path, "%s/ckpt.bin", outdir);
	f = fopen(path, "wb");
	if (!f || fwrite(stage, 4, (size_t)CL_N * CX_NPARAM, f) != (size_t)CL_N * CX_NPARAM || fwrite(C.cg, 4, CX_DU, f) != CX_DU ||
	    fwrite(C.ETB, 4, (size_t)CX_VOCAB * CX_DE, f) != (size_t)CX_VOCAB * CX_DE || fclose(f)) { perror(path); return 1; }
	printf("checkpoint %s: %ld bytes\n", path, (long)((size_t)CL_N * CX_NPARAM + CX_DU + (size_t)CX_VOCAB * CX_DE) * 4);
	if (C.ng && dump("ng-gates", -1, C.ng, CX_NG_ORDERS * CX_DE * 4)) return 1;   /* the n-gram gates beside the checkpoint (the tables are the host server's) */
	return 0;
}

/* ---- gen mode: greedy generation from a checkpoint ---- */
static long gen_nprompt, gen_n;
static uint32_t *gen_out;

static int load_ckpt(const char *path)
{
	FILE *f = fopen(path, "rb");
	if (!f) { perror(path); return -1; }
	for (int k = 0; k < CL_N; k++) if (fread(C.tr[k].col.prm, 4, CX_NPARAM, f) != CX_NPARAM) { fprintf(stderr, "ckpt: short at column %d\n", k); return -1; }
	if (fread(C.cg, 4, CX_DU, f) != CX_DU) return -1;
	if (fread(C.ETB, 4, (size_t)CX_VOCAB * CX_DE, f) != (size_t)CX_VOCAB * CX_DE) return -1;
	fclose(f);
	for (int v = 0; v < CX_VOCAB; v++) for (int c = 0; c < CX_DE; c++) C.ES[(long)v * CX_DE + c] = C.ETB[CX_ETB(v, c)];
	return 0;
}

static void run_gen(void *arg, int k)
{
	(void)arg;
	const int L = C.L;
	const long total = gen_nprompt + gen_n;
	uint32_t cur = gen_out[0], nxt;
	if (!k) rwall[0] = cx_now();
	for (long s = 0; s < total - 1; s++) {
		const int t = (int)(s % L);
		if (t == 0 && s) { cl_wait(&C, k); if (!k) cx_copy(C.bus, C.bus + (long)L * CL_BUS, CL_BUS); cl_wait(&C, k); }
		const int prompt = s + 1 < gen_nprompt;
		nxt = prompt ? gen_out[s + 1] : 0;
		cl_fwd_step(&C, k, t, cur, nxt, prompt ? nxt : 0);
		const uint32_t am = cl_argmax(&C, k, t);
		if (!prompt) { nxt = am; if (!k) gen_out[s + 1] = am; }
		cur = nxt;
	}
	if (!k) rwall[1] = cx_now();
}

static int gen_mode(int argc, char **argv)
{
	/* gen CKPT PROMPT.u16 NPROMPT NGEN OUT.u16 [L] */
	if (argc < 7) { fprintf(stderr, "usage: cluster-gate gen CKPT PROMPT.u16 NPROMPT NGEN OUT.u16\n"); return 2; }
	gen_nprompt = atol(argv[4]);
	gen_n = atol(argv[5]);
	if (load_ckpt(argv[2])) return 1;
#ifdef CX_MOE
	if (C.moe_on) cx_team_run(CL_N, moe_fill, NULL);
#endif
	FILE *f = fopen(argv[3], "rb");
	if (!f) { perror(argv[3]); return 1; }
	uint16_t *pt = malloc((size_t)gen_nprompt * 2);
	if ((long)fread(pt, 2, (size_t)gen_nprompt, f) != gen_nprompt) { fprintf(stderr, "gen: short prompt\n"); return 1; }
	fclose(f);
	gen_out = calloc((size_t)(gen_nprompt + gen_n), 4);
	for (long i = 0; i < gen_nprompt; i++) gen_out[i] = pt[i];
	memset(C.bus, 0, (size_t)(CX_LMAX + 1) * CL_BUS * 4);
	for (int k = 0; k < CL_N; k++) memset(C.tr[k].col.T, 0, CX_NT * 4);
	cx_team_run(CL_N, run_gen, NULL);
	printf("gen: %ld prompt + %ld generated tokens in %.3f s (%.1f us per token)\n", gen_nprompt, gen_n, rwall[1] - rwall[0], (rwall[1] - rwall[0]) * 1e6 / (gen_nprompt + gen_n - 1));
	f = fopen(argv[6], "wb");
	if (!f) { perror(argv[6]); return 1; }
	for (long i = 0; i < gen_nprompt + gen_n; i++) { const uint16_t v = (uint16_t)gen_out[i]; fwrite(&v, 2, 1, f); }
	fclose(f);
	printf("ids:");
	for (long i = 0; i < gen_nprompt + gen_n && i < 96; i++) printf(" %u", gen_out[i]);
	printf("\n");
	return 0;
}

int main(int argc, char **argv)
{
	if (argc < 6) { fprintf(stderr, "usage: cluster-gate fwd|bwd TOKENS.u16 OUTDIR L WARM [DUMP_COL...] | run TOKENS.u16 OUTDIR L NTOK LR VALID.u16 NVAL [EVERY]\n"); return 2; }
	bwd = !strcmp(argv[1], "bwd");
	runm = !strcmp(argv[1], "run");
	const int genm = !strcmp(argv[1], "gen");
	outdir = argv[3];
	L = genm ? (argc > 7 ? atoi(argv[7]) : 32) : atoi(argv[4]);
	warm = runm || genm ? 0 : atoi(argv[5]);
	if (L < 16 || L > CX_LMAX || L % 16 || warm % L) { fprintf(stderr, "cluster-gate: L must be 16 or 32 and WARM a multiple of L\n"); return 2; }
	for (int i = 6; i < argc && ndump < CL_N && !runm && !genm; i++) dump_list[ndump++] = atoi(argv[i]);
	if (!ndump) { dump_list[ndump++] = 0; dump_list[ndump++] = 59; dump_list[ndump++] = 119; }
	C.L = L;
	C.bus_on = getenv("CX_BUS") ? atoi(getenv("CX_BUS")) : 1;
	C.ETB = huge(2u << 20);
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
	C.bus = STAKE((size_t)(CX_LMAX + 1) * CL_BUS * 4);
	C.votes = STAKE(2 * CL_N * CX_DU * 4);
	C.gsum = STAKE(2 * CL_GROUPS * CX_DU * 4);
	C.pairs = STAKE(2 * CL_N * 16 * 4);
	C.dhcp = STAKE((size_t)CL_N * CX_LMAX * CX_DU * 4);
	C.gdhc = STAKE((size_t)CL_GROUPS * CX_LMAX * CX_DU * 4);
	C.dep = STAKE((size_t)CL_N * CX_LMAX * CX_DU * 4);
	C.loss = STAKE(CX_LMAX * 4);
	C.argmax = (volatile uint32_t *)STAKE(2 * 16 * 4);
	if (getenv("CX_NG") && atoi(getenv("CX_NG"))) {
		C.ng = STAKE(CX_NG_ORDERS * CX_DE * 4);
		C.m_ng = STAKE(CX_NG_ORDERS * CX_DE * 4);
		C.v_ng = STAKE(CX_NG_ORDERS * CX_DE * 4);
		C.gng = STAKE(CX_NG_ORDERS * CX_DE * 4);
		C.desum = STAKE((size_t)CX_LMAX * CX_DU * 4);
		uint64_t ph[2];
		C.ngbuf = huge_alloc(2 * CX_NGR_BYTES, ph, 1); C.ngbuf_phys = ph[0];
		C.nggrad = huge_alloc(2 * CX_NGR_BYTES, ph, 1); C.nggrad_phys = ph[0];
		if (!C.ngbuf || !C.nggrad) { fprintf(stderr, "cluster-gate: n-gram ring huge pages\n"); return 1; }
		const int chan = dma_channel();
		if (cdma_open(&ngdma, chan)) { fprintf(stderr, "cluster-gate: DMA channel %d\n", chan); return 1; }
		const int pfd = open("/dev/phihost", O_RDWR | O_SYNC);
		void *pm = pfd < 0 ? MAP_FAILED : mmap(NULL, CX_PART_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, pfd, (off_t)CX_PART_MIB << 20);
		if (pm == MAP_FAILED) { perror("cluster-gate: /dev/phihost"); return 1; }
		C.part = pm;
		C.dma = &ngdma;
		const uint64_t base = ngdma.region_addr + ((uint64_t)CX_PART_MIB << 20);
		C.ng_host_rows = base + CX_NGR_ROWS(0);
		C.ng_host_grad = base + CX_NGR_GRAD(0);
		C.ng_host_cseq = base + CX_NGR_C_SEQ;
		printf("n-gram memory: on, %u rows per order on the host, ring at part + %u MiB, DMA channel %d\n", CX_NG_ROWS, CX_NGR_BASE >> 20, chan);
	}
#ifdef CX_MOE
	C.moe_on = !getenv("CX_MOE") || atoi(getenv("CX_MOE"));
	if (C.moe_on) {
		C.keys = STAKE(CX_EX_KEYFLOATS * 4);
		C.route = (uint32_t *)STAKE((size_t)CL_N * CX_LMAX * 4);
		moe_used = (uint32_t *)STAKE((size_t)CX_EX_N * 4);
		C.moe_used = moe_used;
		C.moe_lr = getenv("CX_MOE_LR") ? (float)atof(getenv("CX_MOE_LR")) : CX_EX_LR;
		moe_bias = getenv("CX_MOE_BIAS") ? (float)atof(getenv("CX_MOE_BIAS")) : CX_EX_HOME_BIAS;
		C.pool = huge(((size_t)CX_EX_N * CX_EX_BYTES + (2u << 20) - 1) & ~((size_t)(2u << 20) - 1));
	}
#endif
	if (o > (8u << 20)) { fprintf(stderr, "cluster-gate: shared overflow %zu\n", o); return 1; }
	memset(sh, 0, o);
	for (int i = 0; i < CX_DU; i++) C.cg[i] = 1.0f;
	if (C.ng) for (int i = 0; i < CX_NG_ORDERS * CX_DE; i++) C.ng[i] = 1.0f;   /* the gates: after the arena is zeroed, like cg */
#ifdef CX_MOE
	if (C.moe_on) cx_ex_keys_init(C.keys);
#endif
	float *E = malloc((size_t)CX_VOCAB * CX_DE * 4);
	cx_emb_init(E);
	for (int v = 0; v < CX_VOCAB; v++) for (int c = 0; c < CX_DE; c++) C.ETB[CX_ETB(v, c)] = E[(long)v * CX_DE + c];
	memcpy(C.ES, E, (size_t)CX_VOCAB * CX_DE * 4);
	free(E);
	memset(C.GET, 0, 2u << 20);
	memset(C.m_et, 0, 2u << 20);
	memset(C.v_et, 0, 2u << 20);

	if (genm) { cx_barrier_init(&C.bar, CL_N, 2, CX_CORE_REL); cx_team_run(CL_N, setup, NULL); return gen_mode(argc, argv); }
	const long need = runm ? atol(argv[5]) + 1 : warm + L + 1;
	FILE *f = fopen(argv[2], "rb");
	if (!f) { perror(argv[2]); return 1; }
	uint16_t *tok = malloc((size_t)need * 2);
	ntok = (long)fread(tok, 2, (size_t)need, f);
	fclose(f);
	if (ntok < (runm ? L + 1 : need)) { fprintf(stderr, "cluster-gate: short token file\n"); return 1; }
	ids = malloc((size_t)ntok * 4);
	for (long t = 0; t < ntok; t++) ids[t] = tok[t];

	cx_barrier_init(&C.bar, CL_N, 2, CX_CORE_REL);
	cx_team_run(CL_N, setup, NULL);
	printf("cluster-gate %s: %d columns, L %d, warm %d, pin failures %d, thread 0 cpu %d, thread 119 cpu %d\n", argv[1], CL_N, L, warm, cx_pin_fail, cx_found_cpu[0], cx_found_cpu[CL_N - 1]);
	if (runm) { const int rc = run_mode(argc, argv); if (C.ng) cdma_close(&ngdma); return rc; }
#ifdef CX_MOE
	if (C.moe_on) { cx_team_run(CL_N, moe_fill, NULL); printf("moe: on, pool filled from the columns' MLPs, home bias %g, expert lr %g\n", moe_bias, C.moe_lr); fflush(stdout); }
#endif
	cx_team_run(CL_N, run_check, NULL);
	double mean = 0;
	for (int t = 0; t < L; t++) mean += C.loss[t];
	{ double s = 0; uint64_t mx = 0; for (int k = 0; k < CL_N; k++) { s += pmu_count[k]; if (pmu_count[k] > mx) mx = pmu_count[k]; }
	  printf("pmu 0x%llx per step per thread over the measured chunk (forward): mean %.1f, thread 0 %.1f, max %.1f\n", (unsigned long long)(getenv("CX_PMU") ? strtoull(getenv("CX_PMU"), NULL, 16) : 0x10cb), s / CL_N / L, (double)pmu_count[0] / L, (double)mx / L); }
	printf("chunk: mean loss %.6f nats (%.4f bits/token); forward %.1f us per step%s\n", mean / L, mean / L / CX_LN2, (wall[1] - wall[0]) * 1e6 / L,
	       bwd ? "" : "");
	if (bwd) printf("backward + gradients: %.1f us per step (whole chunk %.2f ms)\n", (wall[3] - wall[2]) * 1e6 / L, (wall[3] - wall[2]) * 1e3);
	if (dump("cl-loss", -1, C.loss, (size_t)L * 4) || dump("cl-hc", -1, C.hcp[0], (size_t)L * CX_DU * 4)) return 1;
	if (bwd && dump("cl-get", -1, C.GET, 2u << 20)) return 1;
	if (C.ng) cdma_close(&ngdma);
	return 0;
}
