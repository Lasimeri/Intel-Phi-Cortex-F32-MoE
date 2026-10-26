/*
 * dma-coh: does a DMA write into GDDR invalidate the copies other cores hold in L2? The n-gram ring (card/cluster.h)
 * re-fills the same GDDR buffer by DMA while every column thread has read its previous contents, so a stale L2 copy
 * on any core would feed old rows into the columns. 120 threads (one per column, two per core, team.h map) read a
 * 64 KiB buffer into their L2s, thread 0 refills it by a GDDR-to-GDDR DMA, every thread reads it again and counts the
 * words that still show the old contents. Three readings: demand reads, prefetched lines (vprefetch0), and demand
 * reads followed by clevict1 of every line before the second read.
 *   dma-coh [CHANNEL]     default channel 5
 */
#include "../card/team.h"
#include "../card/cdma.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BYTES 65536u
#define WORDS (BYTES / 4)

static cdma dma;
static uint32_t *A, *B;
static uint64_t a_phys, b_phys;
static cx_barrier bar;
static uint64_t epoch[CX_COLUMNS];
static long stale[3][CX_COLUMNS];
static volatile uint32_t sink[CX_COLUMNS][16];

static void wait(int k) { cx_barrier_wait(&bar, k, ++epoch[k]); }

static uint32_t read_all(const volatile uint32_t *p)
{
	uint32_t s = 0;
	for (uint32_t i = 0; i < WORDS; i++) s += p[i];
	return s;
}

static long count_old(const volatile uint32_t *p, uint32_t round)
{
	long n = 0;
	for (uint32_t i = 0; i < WORDS; i++) if (p[i] != (i * 3 + 1) * (round + 1)) n++;
	return n;
}

static void refill(uint32_t round)
{
	for (uint32_t i = 0; i < WORDS; i++) B[i] = (i * 3 + 1) * (round + 1);
	for (uint32_t i = 0; i < WORDS; i += 16) __asm__ volatile("clevict1 %0" : : "m"(*(const char *)(B + i)));
	const uint64_t seq = cdma_submit(&dma, b_phys, a_phys, BYTES, 0, 0);
	if (!seq || cdma_wait(&dma, seq, 3.0)) { fprintf(stderr, "dma-coh: the refill DMA failed\n"); exit(1); }
}

static void body(void *arg, int k)
{
	(void)arg;
	for (uint32_t round = 0; round < 3; round++) {
		/* 1. every thread caches the buffer: demand reads (rounds 0, 2) or prefetches (round 1) */
		if (round == 1) for (uint32_t i = 0; i < WORDS; i += 16) __asm__ volatile("vprefetch0 %0" : : "m"(*(const char *)(A + i)));
		else sink[k][0] = read_all(A);
		wait(k);
		/* 2. thread 0 refills the buffer by DMA */
		if (!k) refill(round);
		wait(k);
		/* 3. round 2: evict the lines from this core's L2 first */
		if (round == 2) for (uint32_t i = 0; i < WORDS; i += 16) __asm__ volatile("clevict1 %0" : : "m"(*(const char *)(A + i)));
		stale[round][k] = count_old(A, round);
		wait(k);
	}
}

int main(int argc, char **argv)
{
	const int chan = argc > 1 ? atoi(argv[1]) : 5;
	if (cdma_open(&dma, chan)) return 1;
	A = huge_alloc(2u << 20, &a_phys, 1);
	B = huge_alloc(2u << 20, &b_phys, 1);
	if (!A || !B) { fprintf(stderr, "dma-coh: huge pages\n"); return 1; }
	for (uint32_t i = 0; i < WORDS; i++) A[i] = i;
	cx_barrier_init(&bar, CX_COLUMNS, 2, CX_CORE_REL);
	cx_team_run(CX_COLUMNS, body, NULL);
	const char *name[3] = { "demand reads, no evict", "prefetched lines, no evict", "demand reads, clevict1 before re-read" };
	for (int r = 0; r < 3; r++) {
		long total = 0, worst = 0, nbad = 0;
		for (int k = 0; k < CX_COLUMNS; k++) { total += stale[r][k]; if (stale[r][k] > worst) worst = stale[r][k]; nbad += stale[r][k] > 0; }
		printf("round %d (%s): stale words %ld of %ld (threads with any stale: %d of %d, worst thread %ld; thread 0 %ld, thread 1 %ld, thread 2 %ld, thread 119 %ld)\n",
		       r, name[r], total, (long)WORDS * CX_COLUMNS, (int)nbad, CX_COLUMNS, worst, stale[r][0], stale[r][1], stale[r][2], stale[r][119]);
	}
	cdma_close(&dma);
	return 0;
}
