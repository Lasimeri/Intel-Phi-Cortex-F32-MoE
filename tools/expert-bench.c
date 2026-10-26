/*
 * expert-bench: stage 2's budget measured on the card. 120 heads (one per column thread, two per core, team.h map)
 * each pull one expert (CX_EX_BYTES = 69,888 B, the column's MLP) per step from one shared pool in GDDR into their
 * L2 staging slot, in lockstep.
 *   expert-bench [NEXPERTS [STEPS [PFDIST]]]     default 61,440 experts (4 GiB - 1 MiB), 200 steps, the copy's prefetch 8 lines ahead
 * The pool is huge pages filled by the 120 threads (word i of expert e holds e * 2654435761 + i, so a wrong copy is
 * visible). Per step thread k takes expert mix(step, k) mod N and:
 *   A  copies it into its slot, PFDIST lines prefetched ahead (the expert read sits on a step's critical path)
 *   B  the same, then prefetches the next step's expert into L2 (vprefetch1); the index is known one step ahead here,
 *      in the model it is known once x is, before the fast-weight pass
 *   C  runs the routing kernel alone: x (80) in two 40-wide halves, each scored against its 248 keys (transposed
 *      [dim][lane], 40 x 16 FMAs per half, the row two ahead prefetched), the argmax of each half, e = a * 248 + b mod N
 * Prints per mode the step time (thread 0's wall per step after 10 steps of warm-up, barriers included), the mean and
 * the slowest thread's own time per step, the aggregate read rate and the copy check.
 */
#include "../card/train.h"
#include "../card/team.h"
#include "../lib/cx-moe.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

static uint32_t *pool;
static size_t nex;
static int steps, pfdist = 8;   /* the copy's prefetch distance in lines (argument 3) */
static cx_barrier bar;
static uint64_t epoch[CX_COLUMNS];
static float *slot[CX_COLUMNS], *xs[CX_COLUMNS], *keys;
static uint64_t cyc[3][CX_COLUMNS];
static double wall[3];
static long bad[CX_COLUMNS];
static volatile float sink[CX_COLUMNS][16] __attribute__((aligned(64)));

static inline void cx_pf1(const void *p) { __asm__ volatile("vprefetch1 %0" : : "m"(*(const char *)p)); }

static inline size_t pick(int step, int k) { return (size_t)(cx_ctx_mix(0x9E3779B97F4A7C15ull * (uint64_t)(step + 1) + (uint64_t)k) % nex); }

static void wait(int k) { cx_barrier_wait(&bar, k, ++epoch[k]); }

static void *huge(size_t bytes)
{
	void *p = mmap(NULL, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | MAP_POPULATE, -1, 0);
	if (p == MAP_FAILED) { perror("huge pages"); exit(1); }
	return p;
}

/* the expert's 1,092 lines into the slot, 8 lines ahead prefetched */
static void fetch(const float *src, float *dst)
{
	for (uint32_t i = 0; i < CX_EX_FLOATS; i += 16) {
		if (i + 16 * pfdist < CX_EX_FLOATS) cx_pf0(src + i + 16 * pfdist);
		_mm512_store_ps(dst + i, _mm512_load_ps(src + i));
	}
}

static void prefetch_l2(const float *src)
{
	for (uint32_t i = 0; i < CX_EX_FLOATS; i += 16) cx_pf1(src + i);
}

/* the argmax lane of one half: s[lane] = sum_d x[d] K[d][lane] over 256 lanes, the 8 padding lanes excluded */
static int route_half(const float *K, const float *x)
{
	__m512 acc[CX_EX_KLANES / 16];
	for (int v = 0; v < CX_EX_KLANES / 16; v++) acc[v] = _mm512_setzero_ps();
	for (int d = 0; d < CX_EX_HALF; d++) {
		const __m512 xb = CX_BC(x + d);
		if (d + 2 < CX_EX_HALF) for (int v = 0; v < CX_EX_KLANES / 16; v++) cx_pf0(K + (long)(d + 2) * CX_EX_KLANES + 16 * v);   /* the row two ahead into L1: the core is in-order, a miss stalls it (Stage 0f rule) */
		const float *row = K + (long)d * CX_EX_KLANES;
		for (int v = 0; v < CX_EX_KLANES / 16; v++) acc[v] = _mm512_fmadd_ps(xb, _mm512_load_ps(row + 16 * v), acc[v]);
	}
	acc[CX_EX_KLANES / 16 - 1] = cx_blend(acc[CX_EX_KLANES / 16 - 1], (__mmask16)0xFF00, _mm512_set1_ps(-3.0e38f));
	__m512 m = acc[0];
	for (int v = 1; v < CX_EX_KLANES / 16; v++) m = cx_max(m, acc[v]);
	m = cx_hmax(m);
	for (int v = 0; v < CX_EX_KLANES / 16; v++) {
		const int hit = _mm512_mask2int(_mm512_cmp_ps_mask(acc[v], m, _CMP_EQ_OQ));
		if (hit) return 16 * v + __builtin_ctz(hit);
	}
	return 0;
}

static void fill(void *arg, int k)
{
	(void)arg;
	for (size_t e = (size_t)k; e < nex; e += CX_COLUMNS) {
		uint32_t *w = pool + e * CX_EX_FLOATS;
		for (uint32_t i = 0; i < CX_EX_FLOATS; i++) w[i] = (uint32_t)e * 2654435761u + i;
	}
	/* this thread's slot at a staggered offset of its own huge page (the cluster's prime stride), its x */
	char *a = huge(2u << 20);
	const size_t lines = ((2u << 20) - CX_EX_BYTES - 4096) / 64, off = ((size_t)k * 104729u % lines) * 64;
	slot[k] = (float *)(a + off);
	xs[k] = (float *)(a + (2u << 20) - 4096);
	uint64_t s = 0xC0FFEE0000ull + (uint64_t)k;
	cx_fill(xs[k], CX_DX, &s, 1.0f);
}

static void body(void *arg, int k)
{
	(void)arg;
	double t0 = 0;
	for (int mode = 0; mode < 3; mode++) {
		bad[k] = 0;
		cyc[mode][k] = 0;
		wait(k);
		for (int s = 0; s < steps; s++) {
			if (!k && s == 10) t0 = cx_now();
			const uint64_t c0 = cx_rdtsc();
			if (mode == 2) {
				const int a = route_half(keys, xs[k]), b = route_half(keys + CX_EX_HALF * CX_EX_KLANES, xs[k] + CX_EX_HALF);
				sink[k][0] = (float)cx_ex_index((uint32_t)a, (uint32_t)b);
			} else {
				const size_t e = pick(s, k);
				const float *src = (const float *)(pool + e * CX_EX_FLOATS);
				fetch(src, slot[k]);
				const uint32_t *w = (const uint32_t *)slot[k];
				for (uint32_t i = 0; i < 16; i++) bad[k] += w[i] != (uint32_t)e * 2654435761u + i;
				for (uint32_t i = CX_EX_FLOATS - 16; i < CX_EX_FLOATS; i++) bad[k] += w[i] != (uint32_t)e * 2654435761u + i;
				if (mode == 1 && s + 1 < steps) prefetch_l2((const float *)(pool + pick(s + 1, k) * CX_EX_FLOATS));
			}
			const uint64_t c1 = cx_rdtsc();
			if (s >= 10) cyc[mode][k] += c1 - c0;
			wait(k);
		}
		if (!k) wall[mode] = cx_now() - t0;
	}
}

int main(int argc, char **argv)
{
	nex = argc > 1 ? (size_t)atol(argv[1]) : CX_EX_N;
	steps = argc > 2 ? atoi(argv[2]) : 200;
	pfdist = argc > 3 ? atoi(argv[3]) : 8;
	if (steps <= 10) { fprintf(stderr, "expert-bench: STEPS must exceed the 10 warm-up steps\n"); return 2; }
	const size_t bytes = nex * CX_EX_BYTES;
	const double f0 = cx_now();
	pool = huge((bytes + (2u << 20) - 1) & ~((size_t)(2u << 20) - 1));
	keys = huge(2u << 20);
	uint64_t s = 0xE4C0DE;
	cx_fill(keys, CX_EX_KEYFLOATS, &s, 0.15f);
	cx_barrier_init(&bar, CX_COLUMNS, 2, CX_CORE_REL);
	cx_team_run(CX_COLUMNS, fill, NULL);
	printf("expert-bench: pool %zu experts x %u B = %zu B in huge pages, keys %u F32, filled by %d threads in %.2f s\n", nex, CX_EX_BYTES, bytes, CX_EX_KEYFLOATS, CX_COLUMNS, cx_now() - f0);
	fflush(stdout);
	cx_team_run(CX_COLUMNS, body, NULL);
	const char *name[3] = { "A copy", "B copy + next expert prefetched to L2", "C routing kernel only" };
	const int n = steps - 10;
	for (int mode = 0; mode < 3; mode++) {
		double sum = 0, mx = 0;
		long nbad = 0;
		for (int k = 0; k < CX_COLUMNS; k++) {
			const double us = (double)cyc[mode][k] / 1238.093 / n;
			sum += us;
			if (us > mx) mx = us;
			nbad += bad[k];
		}
		const double step_us = wall[mode] * 1e6 / n;
		printf("mode %s: %.1f us per step (wall, barriers included); own work per thread mean %.1f us, slowest %.1f us", name[mode], step_us, sum / CX_COLUMNS, mx);
		if (mode < 2) printf("; %d x %u B per step = %.2f GB/s; copy check %ld bad words", CX_COLUMNS, CX_EX_BYTES, (double)CX_COLUMNS * CX_EX_BYTES / step_us / 1e3, nbad);
		printf("\n");
	}
	return 0;
}
