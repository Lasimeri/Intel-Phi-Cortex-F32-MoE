/*
 * ctxserve: the host side of the stage 3 context memory, measured before it is built into the model (the card side is
 * tools/ctx-bench.c). Keeps the last CX_CTX_WINDOW steps of every head's keys and values in host RAM. Per step it
 * waits for the card's keys and values (CX_CTX_C_SEQ, the DMA's status word), scores every head's key against that
 * head's past keys with AVX2 FMA (the keys transposed in blocks of 8 steps: one instruction scores 8 steps), takes the
 * top-4, writes the answers (4 values of 64 F32, 4 scores, 4 step indices per head) into CX_CTX_IN(n & 1), stores
 * the step and publishes CX_CTX_H_SEQ = n + 1.
 *   ctxserve STEPS [THREADS [WINDOW]]     threads over the 120 heads (default 8); window up to CX_CTX_WINDOW
 * Prints the scan time per step (from the keys landing to the answers published: mean, max) and the step rate.
 */
#include <fcntl.h>
#include <immintrin.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>
#include "../lib/cx-moe.h"

static uint8_t *part;
#define LD64(o) __atomic_load_n((uint64_t *)(part + (o)), __ATOMIC_ACQUIRE)
#define ST64(o, x) __atomic_store_n((uint64_t *)(part + (o)), (x), __ATOMIC_RELEASE)

static int W, nthreads, steps;
static float *kT;        /* [head][W/8][64][8]: block b lane l = step slot 8 b + l */
static float *vals;      /* [head][W][64] */
static int *slot_step;   /* [W]: the step a slot holds, -1 empty */
static volatile int step_now, go_gen, done_count;

static double now(void)
{
	struct timespec t;
	clock_gettime(CLOCK_MONOTONIC, &t);
	return t.tv_sec + t.tv_nsec * 1e-9;
}

typedef struct { float s[CX_CTX_TOP]; int slot[CX_CTX_TOP]; } top4;

static inline void top_insert(top4 *t, float s, int slot)
{
	int i = CX_CTX_TOP - 1;
	if (s <= t->s[i]) return;
	for (; i > 0 && s > t->s[i - 1]; i--) { t->s[i] = t->s[i - 1]; t->slot[i] = t->slot[i - 1]; }
	t->s[i] = s;
	t->slot[i] = slot;
}

/* head h at step n: its query against the window (steps < n), the answer into the part, then its k, v stored */
static void serve_head(int h, int n)
{
	const float *kv = (const float *)(part + CX_CTX_OUT(n & 1)) + (size_t)h * CX_CTX_KV;
	const float *q = kv, *v = kv + CX_DS;
	const float *K = kT + (size_t)h * (W / 8) * CX_DS * 8;
	top4 t;
	for (int i = 0; i < CX_CTX_TOP; i++) { t.s[i] = -3.0e38f; t.slot[i] = -1; }
	for (int b = 0; b < W / 8; b += 4) {
		__m256 a0 = _mm256_setzero_ps(), a1 = a0, a2 = a0, a3 = a0;
		const float *k0 = K + (size_t)b * CX_DS * 8, *k1 = k0 + CX_DS * 8, *k2 = k1 + CX_DS * 8, *k3 = k2 + CX_DS * 8;
		for (int d = 0; d < CX_DS; d++) {
			const __m256 qd = _mm256_set1_ps(q[d]);
			a0 = _mm256_fmadd_ps(qd, _mm256_loadu_ps(k0 + d * 8), a0);
			a1 = _mm256_fmadd_ps(qd, _mm256_loadu_ps(k1 + d * 8), a1);
			a2 = _mm256_fmadd_ps(qd, _mm256_loadu_ps(k2 + d * 8), a2);
			a3 = _mm256_fmadd_ps(qd, _mm256_loadu_ps(k3 + d * 8), a3);
		}
		float s[32];
		_mm256_storeu_ps(s, a0);
		_mm256_storeu_ps(s + 8, a1);
		_mm256_storeu_ps(s + 16, a2);
		_mm256_storeu_ps(s + 24, a3);
		for (int l = 0; l < 32; l++) {
			const int slot = b * 8 + l, st = slot_step[slot];
			if (st >= 0 && st < n) top_insert(&t, s[l], slot);
		}
	}
	uint8_t *a = part + CX_CTX_IN(n & 1) + (size_t)h * CX_CTX_ANS_HEAD;
	float *out_v = (float *)a, *out_s = (float *)(a + CX_CTX_TOP * CX_DS * 4);
	uint32_t *out_i = (uint32_t *)(a + CX_CTX_TOP * CX_DS * 4 + 16);
	for (int i = 0; i < CX_CTX_TOP; i++) {
		if (t.slot[i] < 0) { memset(out_v + i * CX_DS, 0, CX_DS * 4); out_s[i] = -3.0e38f; out_i[i] = CX_CTX_NONE; continue; }
		memcpy(out_v + i * CX_DS, vals + ((size_t)h * W + t.slot[i]) * CX_DS, CX_DS * 4);
		out_s[i] = t.s[i];
		out_i[i] = (uint32_t)slot_step[t.slot[i]];
	}
	/* store this step: key lane slot % 8 of block slot / 8, value row slot */
	const int slot = n % W;
	float *kb = kT + ((size_t)h * (W / 8) + slot / 8) * CX_DS * 8 + slot % 8;
	for (int d = 0; d < CX_DS; d++) kb[d * 8] = q[d];
	memcpy(vals + ((size_t)h * W + slot) * CX_DS, v, CX_DS * 4);
}

typedef struct { int h0, h1; } span;

static void *worker(void *arg)
{
	const span *sp = arg;
	for (int gen = 1;; gen++) {
		while (__atomic_load_n(&go_gen, __ATOMIC_ACQUIRE) < gen) _mm_pause();
		const int n = step_now;
		if (n < 0) return NULL;
		for (int h = sp->h0; h < sp->h1; h++) serve_head(h, n);
		__atomic_fetch_add(&done_count, 1, __ATOMIC_ACQ_REL);
	}
}

int main(int argc, char **argv)
{
	if (argc < 2) { fprintf(stderr, "usage: ctxserve STEPS [THREADS [WINDOW]]\n"); return 2; }
	steps = atoi(argv[1]);
	nthreads = argc > 2 ? atoi(argv[2]) : 8;
	W = argc > 3 ? atoi(argv[3]) : CX_CTX_WINDOW;
	if (W < 32 || W > CX_CTX_WINDOW || W % 32 || nthreads < 1 || nthreads > CX_CTX_HEADS) { fprintf(stderr, "ctxserve: WINDOW a multiple of 32 up to %d, THREADS 1..%d\n", CX_CTX_WINDOW, CX_CTX_HEADS); return 2; }
	const int fd = open("/dev/shm/phi-hostmem", O_RDWR);
	if (fd < 0) { perror("/dev/shm/phi-hostmem"); return 1; }
	part = mmap(NULL, CX_PART_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, (off_t)CX_PART_MIB << 20);
	close(fd);
	if (part == MAP_FAILED) { perror("mmap"); return 1; }
	kT = aligned_alloc(64, (size_t)CX_CTX_HEADS * W * CX_DS * sizeof *kT);
	vals = aligned_alloc(64, (size_t)CX_CTX_HEADS * W * CX_DS * sizeof *vals);
	slot_step = malloc((size_t)W * sizeof *slot_step);
	memset(kT, 0, (size_t)CX_CTX_HEADS * W * CX_DS * sizeof *kT);
	memset(vals, 0, (size_t)CX_CTX_HEADS * W * CX_DS * sizeof *vals);
	for (int i = 0; i < W; i++) slot_step[i] = -1;
	ST64(CX_CTX_H_SEQ, 0);
	ST64(CX_CTX_C_SEQ, 0);
	/* the workers over spans of heads */
	pthread_t tid[CX_CTX_HEADS];
	span sp[CX_CTX_HEADS];
	for (int t = 0; t < nthreads; t++) {
		sp[t].h0 = CX_CTX_HEADS * t / nthreads;
		sp[t].h1 = CX_CTX_HEADS * (t + 1) / nthreads;
		pthread_create(&tid[t], NULL, worker, &sp[t]);
	}
	printf("ctxserve: %d steps, %d threads, window %d: keys %zu B + values %zu B in host RAM; waiting for the card\n", steps, nthreads, W,
	       (size_t)CX_CTX_HEADS * W * CX_DS * 4, (size_t)CX_CTX_HEADS * W * CX_DS * 4);
	fflush(stdout);
	double t_scan = 0, m_scan = 0, t_first = 0, t_last = 0;
	long timed = 0;
	for (int n = 0; n < steps; n++) {
		while (LD64(CX_CTX_C_SEQ) < (uint64_t)n + 1) _mm_pause();
		const double a = now();
		if (n == 10) t_first = a;
		step_now = n;
		done_count = 0;
		__atomic_store_n(&go_gen, n + 1, __ATOMIC_RELEASE);
		while (__atomic_load_n(&done_count, __ATOMIC_ACQUIRE) < nthreads) _mm_pause();
		slot_step[n % W] = n;
		ST64(CX_CTX_H_SEQ, (uint64_t)n + 1);
		const double b = now();
		if (n >= 10) { t_scan += b - a; if (b - a > m_scan) m_scan = b - a; timed++; t_last = b; }
	}
	step_now = -1;
	__atomic_store_n(&go_gen, steps + 1, __ATOMIC_RELEASE);
	for (int t = 0; t < nthreads; t++) pthread_join(tid[t], NULL);
	printf("ctxserve: %ld timed steps: scan + answers %.1f us per step (max %.1f); %.0f steps/s end to end from the host's view\n", timed, t_scan * 1e6 / timed, m_scan * 1e6, timed / (t_last - t_first));
	return 0;
}
