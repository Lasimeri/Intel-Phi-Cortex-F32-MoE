/*
 * cortexd: the Intel Phi Cortex generation daemon on the card (Stage 5; MoE stage 1 adds the n-gram memory).
 * usage: cortexd [PART_MIB]
 *
 * 120 column threads (card/cluster.h); thread 0 also serves the host's mailbox (lib/cx-proto.h, part at 1344 MiB of
 * the shared host-memory region, mapped uncached through /dev/phihost) and drives a card-owned DMA channel, 2 unless
 * CX_DMA_CHAN names another (card/cdma.h):
 *   LOAD   the host staged a checkpoint in the part; the card pulls 120 column weight blocks, cg and the embedding
 *          into place by DMA (each thread's huge page has a known physical address), then rebuilds the shadow ES.
 *          With arg bit 0 set the host also staged the n-gram gates (2 x 128 F32) after the checkpoint: the card pulls
 *          them too and the memory is on for this checkpoint; with the bit clear the memory is off.
 *   RESET  every column's fast weights T and the bus to zero; the step count restarts at 0.
 *   BUS    the bus on or off (as the checkpoint was trained).
 *   STEP   one cluster forward on the given token; the 4096 raw logits are assembled in a card huge page and pushed
 *          to the host part by DMA, the ACK word being the DMA's status (written after the data). With the memory
 *          on, the host has written the step's two rows (orders 2 and 3, 1 KiB) at CX_NGS_ROWS(n), n = steps since
 *          RESET; thread 0 pulls them by DMA into ring slot n % 64 (the GDDR layout the training chunks use) before
 *          releasing the team into the forward, and the columns add gate * row to the token's embedding.
 * Everything the host does (row gathering, sampling, the orchestrator, detokenising) runs on the 5800X while the
 * card waits for the next token. A DMA write into GDDR invalidates every core's L2 copy of the lines (measured
 * 2026-10-26, tools/dma-coh.c), so the ring slots are reused without eviction.
 */
#include "cluster.h"
#include "cdma.h"
#include "../lib/cx-proto.h"

#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

static cluster C __attribute__((aligned(64)));
static volatile uint8_t *part;
static cdma dma;
static float *logits_buf;
static float *ngpage, *nggates;   /* one huge page: 64 row slots of 1 KiB, then the gates at 64 KiB */
static uint64_t logits_phys, cg_phys, etb_phys[1], prm_phys[CL_N], ngpage_phys, nggates_phys;
static uint64_t part_mib = CX_PART_MIB;
static uint64_t host_part;   /* the part's address for the DMA engine */

static inline uint32_t rd32(int o) { return *(volatile uint32_t *)(part + o); }
static inline uint64_t rd64(int o) { return *(volatile uint64_t *)(part + o); }
static inline void wr32(int o, uint32_t v) { *(volatile uint32_t *)(part + o) = v; }
static inline void wr64(int o, uint64_t v) { *(volatile uint64_t *)(part + o) = v; }

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

/* per-thread arenas (as tools/cluster-gate.c setup), plus the physical address of the weights for the loader */
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
	tr->gprm = TAKE(3 * ((CX_NPARAM + 15) & ~15) * 4);
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
	tr->ngr = TAKE((size_t)CX_LMAX * CX_NG_ORDERS * CX_DE * 4);   /* the rows used per step (the forward saves them) */
	tr->ng = NULL;                                                 /* the gates, once a checkpoint with the memory is loaded */
	if (o > (4u << 20)) { fprintf(stderr, "cortexd: arena overflow %zu\n", o); exit(1); }
	tr->ET = C.ETB;
	tr->ES = C.ES;
	tr->GET = C.GET;
	tr->cg = C.cg;
	cx_col_init(tr->col.prm, k);
	memset(tr->col.T, 0, CX_NT * 4);
	memset(tr->col.act, 0, CX_NACT * 4);
	cl_blocks(&C, k);
	C.epoch[k] = 0;
	prm_phys[k] = phys_of(tr->col.prm);
}

/* thread 0: pull the staged checkpoint (and the n-gram gates when arg bit 0 is set) into place by DMA */
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
	C.ng = arg & 1 ? nggates : NULL;
	return 0;
}

static volatile uint32_t ctl_op, ctl_arg, ctl_quit, ctl_exit;
static long step_n;   /* steps since RESET: the step slot within the chunk buffers is step_n % 32, the ring slot step_n % 64 */
static long ng_steps, lg_steps;
static double ng_dma_s, ng_dma_max, lg_dma_s, lg_dma_max;   /* the row DMA and the logits DMA, submit to completion */

/* thread 0, memory on: the step's rows from the part into ring slot step_n % 64, before the team starts the forward */
static int ng_fetch(void)
{
	const double t0 = cx_now();
	const uint64_t seq = cdma_submit(&dma, host_part + CX_NGS_ROWS(step_n), ngpage_phys + (uint64_t)(step_n % CX_NGS_STEPS) * CX_NGS_BYTES, CX_NGS_BYTES, 0, 0);
	if (!seq || cdma_wait(&dma, seq, 2.0)) { fprintf(stderr, "cortexd: row DMA failed at step %ld\n", step_n); return -1; }
	const double dt = cx_now() - t0;
	ng_dma_s += dt;
	if (dt > ng_dma_max) ng_dma_max = dt;
	ng_steps++;
	C.chunk = step_n / CX_LMAX;   /* cl_fwd_step reads slot chunk % 2, step step_n % 32 */
	return 0;
}

/* the card-owned DMA channel: 2 unless CX_DMA_CHAN names another (a channel left enabled by a killed owner stays
 * unusable until the card reboots; the daemon then runs on a different one) */
static int dma_channel(void)
{
	const char *e = getenv("CX_DMA_CHAN");
	return e && *e ? atoi(e) : 2;
}

/* A session with no first command within this many seconds is abandoned: a client that wrote its session while no
 * daemon was alive leaves CX_H_SESSION dangling, and without the grace period the next daemon would take that session
 * and wait forever for a client that is gone (2026-10-26, four hours of "did not take the session"). */
#define CX_SESSION_GRACE_S 5.0

static void serve(void *arg, int k)
{
	(void)arg;
	uint64_t done = 0, first = 0;
	double t0 = 0;
	if (!k) {
		done = first = CX_CMD_SEQ(rd64(CX_H_CMD));
		wr64(CX_C_ACK, done);
		cx_fence();
		wr64(CX_C_SESSION, rd64(CX_H_SESSION));
		wr32(CX_C_STATE, 1);
		t0 = cx_now();
		ng_steps = lg_steps = 0;
		ng_dma_s = ng_dma_max = lg_dma_s = lg_dma_max = 0;
	}
	for (;;) {
		/* thread 0 waits for a command and publishes it; everyone else waits at the barrier */
		if (!k) {
			for (uint32_t spins = 0;; spins++) {
				const uint64_t c = rd64(CX_H_CMD), seq = CX_CMD_SEQ(c);
				if (seq == CX_CMD_QUIT) { ctl_quit = 1; break; }
				if (seq != done) { ctl_op = rd32(CX_H_CMD_OP); ctl_arg = CX_CMD_ARG(c); done = seq; break; }
				if (done == first && (spins & 4095) == 0 && cx_now() - t0 > CX_SESSION_GRACE_S) {
					printf("cortexd: no command within %.0f s, session abandoned\n", CX_SESSION_GRACE_S);
					fflush(stdout);
					ctl_quit = 1;
					break;
				}
				cx_relax();
			}
			if (!ctl_quit && ctl_op == CX_OP_STEP && C.ng && ng_fetch()) wr32(CX_C_STATE, 2);
		}
		cl_wait(&C, k);
		if (ctl_quit) break;
		const uint32_t op = ctl_op, a = ctl_arg;
		if (op == CX_OP_LOAD) {
			if (!k) { if (load_ckpt(a)) wr32(CX_C_STATE, 2); }
			cl_wait(&C, k);
			C.tr[k].ng = C.ng;   /* every column's gate pointer follows the loaded checkpoint */
		} else if (op == CX_OP_RESET) {
			memset(C.tr[k].col.T, 0, CX_NT * 4);
			if (!k) { memset(C.bus, 0, (size_t)(CX_LMAX + 1) * CL_BUS * 4); step_n = 0; }
			cl_wait(&C, k);
		} else if (op == CX_OP_EXIT) {
			if (!k) { ctl_exit = 1; cx_fence(); wr64(CX_C_ACK, done); }
			break;
		} else if (op == CX_OP_BUS) {
			if (!k) C.bus_on = (int)a;
			cl_wait(&C, k);
		} else if (op == CX_OP_STEP) {
			const int t = (int)(step_n % CX_LMAX);
			cl_fwd_step(&C, k, t, a, a, 0);
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
				if (t + 1 == CX_LMAX) cx_copy(C.bus, C.bus + (long)CX_LMAX * CL_BUS, CL_BUS);   /* the bus wraps with the chunk */
				step_n++;
			}
			cl_wait(&C, k);   /* step_n and the wrapped bus visible */
			continue;
		}
		if (!k) { cx_fence(); wr64(CX_C_ACK, done); }
	}
	if (!k) {
		wr32(CX_C_STATE, 3);
		if (lg_steps) printf("cortexd: %ld steps: logits DMA mean %.1f us, max %.1f us", lg_steps, lg_dma_s * 1e6 / lg_steps, lg_dma_max * 1e6);
		if (ng_steps) printf("; n-gram rows %ld steps: row DMA mean %.1f us, max %.1f us", ng_steps, ng_dma_s * 1e6 / ng_steps, ng_dma_max * 1e6);
		if (lg_steps) printf("\n");
	}
}

int main(int argc, char **argv)
{
	if (argc > 1) part_mib = strtoull(argv[1], NULL, 10);
	C.L = CX_LMAX;
	C.bus_on = 1;
	C.ETB = huge(2u << 20);
	etb_phys[0] = phys_of(C.ETB);
	C.GET = huge(2u << 20);
	C.m_et = C.GET;   /* unused in the daemon */
	C.v_et = C.GET;
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
	logits_buf = STAKE(CX_VOCAB * 4);
	if (o > (8u << 20)) { fprintf(stderr, "cortexd: shared overflow %zu\n", o); return 1; }
	memset(sh, 0, o);
	C.logits_out = logits_buf;
	cg_phys = phys_of(C.cg);
	logits_phys = phys_of(logits_buf);
	if (!etb_phys[0] || !cg_phys || !logits_phys) { fprintf(stderr, "cortexd: physical addresses unavailable\n"); return 1; }
	for (int i = 0; i < CX_DU; i++) C.cg[i] = 1.0f;
	/* the n-gram ring buffer and the gates: one huge page the DMA engine fills (zeroed by huge_alloc) */
	ngpage = huge_alloc(2u << 20, &ngpage_phys, 1);
	if (!ngpage) { fprintf(stderr, "cortexd: no huge page for the n-gram ring\n"); return 1; }
	nggates = ngpage + CX_NGS_STEPS * CX_NGS_BYTES / 4;
	nggates_phys = ngpage_phys + CX_NGS_STEPS * CX_NGS_BYTES;
	C.ngbuf = ngpage;
	C.ngbuf_phys = ngpage_phys;
	C.ng = NULL;

	const int chan = dma_channel();
	if (cdma_open(&dma, chan)) return 1;
	host_part = dma.region_addr + (part_mib << 20);
	int fd = open("/dev/phihost", O_RDWR | O_SYNC);
	part = fd < 0 ? MAP_FAILED : mmap(NULL, CX_PART_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, (off_t)(part_mib << 20));
	if (part == MAP_FAILED) { perror("/dev/phihost"); return 1; }
	C.part = part;
	C.dma = &dma;
	wr32(CX_C_STATE, 0);

	cx_barrier_init(&C.bar, CL_N, 2, CX_CORE_REL);
	cx_team_run(CL_N, setup, NULL);
	printf("cortexd: %d columns, part %llu MiB, DMA channel %d (region 0x%llx, %llu MiB), n-gram ring at part + %u KiB; waiting for a session\n", CL_N,
	       (unsigned long long)part_mib, chan, (unsigned long long)dma.region_addr, (unsigned long long)(dma.region_size >> 20), CX_NGS_BASE >> 10);
	fflush(stdout);
	for (;;) {
		const uint64_t session = rd64(CX_C_SESSION);
		while (rd32(CX_H_MAGIC) != CX_MAGIC || rd64(CX_H_SESSION) == session) usleep(1000);
		ctl_quit = 0;
		cx_team_run(CL_N, serve, NULL);
		printf("cortexd: session %llu ended%s\n", (unsigned long long)rd64(CX_C_SESSION), ctl_exit ? ", exiting" : "; waiting for the next");
		fflush(stdout);
		if (ctl_exit) break;
	}
	cdma_close(&dma);
	return 0;
}
