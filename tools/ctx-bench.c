/*
 * ctx-bench: stage 3's round trip measured on the real path (card side; host/ctxd.c is the other end), at the
 * 16,384-step ring of 2026-10-27 15:5x. Every step the 120 heads write their query, key and value (q 64, k 64, v 64
 * F32; CX_CTX_KV_BYTES = 92,160 B for all) into a card buffer, thread 0 DMAs it to the host slot CX_CTX_OUT(n & 1)
 * with CX_CTX_C_SEQ = n + 1 as the status word; the host answers each head's query with its top-4 entries (values,
 * scores, step indices; CX_CTX_ANS_BYTES = 130,560 B) at CX_CTX_IN(n & 1) and publishes CX_CTX_H_SEQ = n + 1;
 * thread 0 polls it and DMAs the answers in.
 *   ctx-bench STEPS [CHANNEL [MODE]]   channel 5; mode 0: the card waits for each step's answer before the next
 *                                 step (the full round trip); mode 1: the answer of step n is pulled at step n + 1
 *                                 (one step of lag, the design of PLAN-MOE.md)
 * Queries, keys and values are cx_ctx_val(step, head, dim, which), which both sides recompute (which 0 key, 1
 * value, 2 query). Every 64th step each head recomputes its own exact top-1 over the ring (its key history kept in
 * GDDR, 16,384 steps) and compares it with the host's: with the index on the host the two may differ (the host's
 * recall on the model's real keys is tools/ctx-recall.c's number); what must hold is that every answer is valid:
 * its step is in the ring, its score is that step's q . k and its values are that step's. Those steps are excluded
 * from the timing. Prints per step the q, k, v DMA out, the wait for the host, the answers' DMA in and the whole
 * (means and maxima over the steps after 10 of warm-up), the agreement of the host's top-1 with the exact top-1,
 * and the invalid answers (the exit code).
 */
#include "../card/team.h"
#include "../card/cdma.h"
#include "../lib/cx-moe.h"
#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

static cdma dma;
static volatile uint8_t *part;
static uint64_t host_part;
static float *kvbuf[2], *ansbuf;
static uint64_t kvbuf_phys[2], ansbuf_phys;
static cx_barrier bar;
static uint64_t epoch[CX_COLUMNS];
static float *hist[CX_COLUMNS];   /* each head's keys of the last CX_CTX_HIST steps (the ring plus the lag) */
#define CX_CTX_HIST (CX_CTX_WINDOW + 2)
static int steps, mode;
static double t_out, t_host, t_in, t_all, m_out, m_host, m_in, m_all;
static long timed, agree[CX_COLUMNS], invalid[CX_COLUMNS], checks[CX_COLUMNS];
static uint64_t out_seq[2];

static void wait(int k) { cx_barrier_wait(&bar, k, ++epoch[k]); }

static inline uint64_t rd_hseq(void) { return __atomic_load_n((volatile uint64_t *)(part + CX_CTX_H_SEQ), __ATOMIC_ACQUIRE); }

/* thread 0: the q, k, v of step n out, the answers of step `ans` in (ans < 0: none) */
static void exchange(int n, int ans)
{
	const double a = cx_now();
	if (out_seq[n & 1] && cdma_wait(&dma, out_seq[n & 1], 5.0)) fprintf(stderr, "ctx-bench: the q, k, v DMA of step %d never completed\n", n - 2);
	out_seq[n & 1] = cdma_submit(&dma, kvbuf_phys[n & 1], host_part + CX_CTX_OUT(n & 1), CX_CTX_KV_BYTES, host_part + CX_CTX_C_SEQ, (uint64_t)n + 1);
	if (!out_seq[n & 1]) { fprintf(stderr, "ctx-bench: q, k, v DMA submit failed at step %d\n", n); exit(1); }
	if (mode == 0 && cdma_wait(&dma, out_seq[n & 1], 5.0)) { fprintf(stderr, "ctx-bench: q, k, v DMA timed out at step %d\n", n); exit(1); }
	const double b = cx_now();
	double c = b, d = b;
	if (ans >= 0) {
		for (long spins = 0; rd_hseq() < (uint64_t)ans + 1; spins++) {
			if (spins > 20000000) { fprintf(stderr, "ctx-bench: no answer for step %d from the host\n", ans); exit(1); }
			cx_relax();
		}
		c = cx_now();
		const uint64_t seq = cdma_submit(&dma, host_part + CX_CTX_IN(ans & 1), ansbuf_phys, CX_CTX_ANS_BYTES, 0, 0);
		if (!seq || cdma_wait(&dma, seq, 5.0)) { fprintf(stderr, "ctx-bench: answer DMA failed at step %d\n", ans); exit(1); }
		d = cx_now();
	}
	if (n >= 10 && (n % 64) != 63) {
		const double o = b - a, h = c - b, i = d - c, all = d - a;
		t_out += o; t_host += h; t_in += i; t_all += all;
		if (o > m_out) m_out = o;
		if (h > m_host) m_host = h;
		if (i > m_in) m_in = i;
		if (all > m_all) m_all = all;
		timed++;
	}
}

/* the host's answer for step `ans` at head k: valid, and in agreement with this head's own exact top-1 over the ring */
static void check(int k, int ans)
{
	const uint8_t *a = (const uint8_t *)ansbuf + (size_t)k * CX_CTX_ANS_HEAD;
	const float *vals = (const float *)a, *scores = (const float *)(a + CX_CTX_TOP * CX_DS * 4);
	const uint32_t *idx = (const uint32_t *)(a + CX_CTX_TOP * CX_DS * 4 + 16);
	checks[k]++;
	if (ans == 0) { if (idx[0] != CX_CTX_NONE) invalid[k]++; else agree[k]++; return; }
	float q[CX_DS];
	for (int d = 0; d < CX_DS; d++) q[d] = cx_ctx_val((uint32_t)ans, (uint32_t)k, (uint32_t)d, 2);
	const int lo = ans > CX_CTX_WINDOW ? ans - CX_CTX_WINDOW : 0;
	float best = -3.0e38f;
	int best_i = -1;
	for (int i = lo; i < ans; i++) {
		const float *key = hist[k] + (size_t)(i % CX_CTX_HIST) * CX_DS;
		float s = 0;
		for (int d = 0; d < CX_DS; d++) s += q[d] * key[d];
		if (s > best) { best = s; best_i = i; }
	}
	if (idx[0] == CX_CTX_NONE || idx[0] >= (uint32_t)ans || (int)idx[0] < lo) { invalid[k]++; return; }
	const float *key = hist[k] + (size_t)(idx[0] % CX_CTX_HIST) * CX_DS;
	float s = 0;
	for (int d = 0; d < CX_DS; d++) s += q[d] * key[d];
	if (fabsf(scores[0] - s) > 1e-4f * (1.0f + fabsf(s))) { invalid[k]++; return; }
	for (int d = 0; d < CX_DS; d++) if (vals[d] != cx_ctx_val(idx[0], (uint32_t)k, (uint32_t)d, 1)) { invalid[k]++; return; }
	if ((int)idx[0] == best_i) agree[k]++;
}

static void body(void *arg, int k)
{
	(void)arg;
	for (int n = 0; n < steps; n++) {
		/* this step's q, k, v of head k into the buffer of slot n & 1, the key into the history */
		float *kv = kvbuf[n & 1] + (size_t)k * CX_CTX_KV, *h = hist[k] + (size_t)(n % CX_CTX_HIST) * CX_DS;
		for (int d = 0; d < CX_DS; d++) {
			kv[d] = cx_ctx_val((uint32_t)n, (uint32_t)k, (uint32_t)d, 2);
			kv[CX_DS + d] = h[d] = cx_ctx_val((uint32_t)n, (uint32_t)k, (uint32_t)d, 0);
			kv[2 * CX_DS + d] = cx_ctx_val((uint32_t)n, (uint32_t)k, (uint32_t)d, 1);
		}
		wait(k);
		const int ans = mode == 0 ? n : n - 1;
		if (!k) exchange(n, ans);
		wait(k);
		if (ans >= 0 && (n % 64) == 63) check(k, ans);
		wait(k);
	}
}

int main(int argc, char **argv)
{
	if (argc < 2) { fprintf(stderr, "usage: ctx-bench STEPS [CHANNEL [MODE]]\n"); return 2; }
	steps = atoi(argv[1]);
	const int chan = argc > 2 ? atoi(argv[2]) : 5;
	mode = argc > 3 ? atoi(argv[3]) : 0;
	if (cdma_open(&dma, chan)) return 1;
	host_part = dma.region_addr + ((uint64_t)CX_PART_MIB << 20);
	const int fd = open("/dev/phihost", O_RDWR | O_SYNC);
	part = fd < 0 ? MAP_FAILED : mmap(NULL, CX_PART_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, (off_t)CX_PART_MIB << 20);
	if (part == MAP_FAILED) { perror("/dev/phihost"); return 1; }
	uint64_t phys;
	uint8_t *page = huge_alloc(2u << 20, &phys, 1);
	if (!page) { fprintf(stderr, "ctx-bench: no huge page\n"); return 1; }
	kvbuf[0] = (float *)page;
	kvbuf[1] = (float *)(page + 131072);
	ansbuf = (float *)(page + 262144);
	kvbuf_phys[0] = phys;
	kvbuf_phys[1] = phys + 131072;
	ansbuf_phys = phys + 262144;
	for (int k = 0; k < CX_COLUMNS; k++) {
		hist[k] = calloc((size_t)CX_CTX_HIST * CX_DS, sizeof(float));
		if (!hist[k]) { fprintf(stderr, "ctx-bench: no memory for the histories\n"); return 1; }
	}
	*(volatile uint64_t *)(part + CX_CTX_C_SEQ) = 0;
	cx_fence();
	printf("ctx-bench: %d steps, channel %d, mode %d (%s), q, k, v out %u B, answers in %u B, the ring %d steps\n", steps, chan, mode,
	       mode ? "answer of step n pulled at step n + 1" : "full round trip per step", CX_CTX_KV_BYTES, CX_CTX_ANS_BYTES, CX_CTX_WINDOW);
	fflush(stdout);
	cx_barrier_init(&bar, CX_COLUMNS, 2, CX_CORE_REL);
	cx_team_run(CX_COLUMNS, body, NULL);
	long ag = 0, iv = 0, ch = 0;
	for (int k = 0; k < CX_COLUMNS; k++) { ag += agree[k]; iv += invalid[k]; ch += checks[k]; }
	printf("ctx-bench mode %d: per step (%ld timed): q, k, v DMA out %.1f us (max %.1f), wait for the host %.1f us (max %.1f), answers DMA in %.1f us (max %.1f), whole %.1f us (max %.1f)\n",
	       mode, timed, t_out * 1e6 / timed, m_out * 1e6, t_host * 1e6 / timed, m_host * 1e6, t_in * 1e6 / timed, m_in * 1e6, t_all * 1e6 / timed, m_all * 1e6);
	printf("ctx-bench checks: %ld head-steps checked, the host's top-1 the exact top-1 in %ld (%.3f), invalid answers %ld\n", ch, ag, ch ? (double)ag / ch : 0, iv);
	cdma_close(&dma);
	return iv ? 1 : 0;
}
