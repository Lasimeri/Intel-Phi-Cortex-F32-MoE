/*
 * ctxd: the host side of the stage 3 context memory at 16,384 steps (2026-10-27 15:5x, the user's "16K KV cache so
 * the student can remember its previous responses"; PLAN-MOE.md "15:52"). It grew out of host/ctxserve.c, the
 * bench of the round trip, and serves whatever runs on the card: the daemon's training and generation (3b) or
 * tools/ctx-bench.c.
 *
 * Per head the service keeps a ring of the last CX_CTX_WINDOW (16,384) steps' keys (64 F32, unit-norm) and values
 * (64 F32) in host RAM: 1,006,632,960 B. Per step it waits for the card's q, k, v of every head (CX_CTX_C_SEQ, the
 * DMA's status word, at CX_CTX_OUT(n & 1)), answers each head's query with the top-4 entries by dot product, then
 * stores the step. The top-4 come from two sets: the last CX_CTX_RECENT (256) steps scanned exactly, and the steps
 * before them through an online IVF index: 256 centroids a head, seeded by the head's first 256 keys, each arriving
 * key moving its nearest centroid toward itself by ETA (renormalised) and filed under it for good (lists of CAP
 * steps, the oldest overwritten); a query reads the PROBE nearest centroids' lists and scores their entries
 * exactly. Measured on the model's own keys (tools/ctx-recall.c, round 5 of the from-scratch run, 23,856 queries):
 * the exact top-4 over 16,384 steps lie within the last 256 only 2.6 % of the time; this index returns 93.8 % of
 * them with 16 probes (top-1 92.4 %, the scores' sum 99.4 % of the exact), SimHash buckets 7 to 27 %; an exact scan
 * of all 16,384 keys would read 503 MB a step. The answer (4 values of 64 F32, 4 scores, 4 step indices per head;
 * CX_CTX_ANS_BYTES) goes to CX_CTX_IN(n & 1) and CX_CTX_H_SEQ = n + 1 is published. The card pulls the answer of
 * step n at step n + 1 (one step of lag: the answer of step n uses the memory up to n - 1 and is consumed at n + 1).
 *
 * Words beside the slots: CX_CTX_ALIVE, a heartbeat the service writes every step and every 100 ms when idle (the
 * card uses the memory only while it is fresh); CX_CTX_RESET, a client writes a session id to empty the ring and
 * the index (a new chat), echoed in CX_CTX_RESET_ACK once done.
 *   ctxd [--threads T] [--probe P] [--eta X] [--cap M] [--steps N] [--selftest N] [--quiet]
 * Defaults: 8 threads over the 120 heads, probe 16, eta 0.05, lists of 1,024, run until killed (--steps N ends
 * after N steps, as the bench wants). --selftest N serves N steps of synthetic q, k, v (cx_ctx_val) it writes into
 * the slots itself, no card: the service's own cost as the ring fills to 16,384. Prints every 10,000 steps (and
 * at the end) the service time per step: mean and max from the q, k, v landing to the answers published, the
 * candidates scored per head, and the steps served.
 */
#include <fcntl.h>
#include <immintrin.h>
#include <math.h>
#include <pthread.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>
#include "../lib/cx-moe.h"

#define W CX_CTX_WINDOW
#define WM (W - 1)   /* W is a power of two: the slot of step s is s & WM */
#define RECENT CX_CTX_RECENT
#define NC 256

_Static_assert((W & (W - 1)) == 0, "the ring is a power of two");

static uint8_t *part;
#define LD64(o) __atomic_load_n((uint64_t *)(part + (o)), __ATOMIC_ACQUIRE)
#define ST64(o, x) __atomic_store_n((uint64_t *)(part + (o)), (x), __ATOMIC_RELEASE)

static int nthreads = 8, probe = 16, cap = 1024, quiet;
static float eta = 0.05f;
static long max_steps = -1, selftest = 0;
static float *keys;          /* [head][W][64] */
static float *vals;          /* [head][W][64] */
static int64_t *slot_step;   /* [W]: the step a slot holds, -1 empty (every head advances together) */
static float *cent;          /* [head][NC][64] unit centroids */
static uint32_t *lists;      /* [head][NC][cap] step numbers (the slot = step & WM) */
static uint16_t *cursor;     /* [head][NC] next entry to overwrite */
static uint16_t *fill;       /* [head][NC] entries written, up to cap */
static int *seeded;          /* [head] centroids seeded so far (the first NC keys) */
static volatile int64_t step_now;
static volatile int go_gen, done_count;
static volatile sig_atomic_t stop_now;
static long cand_total;      /* candidates scored, summed over heads and steps (atomic adds) */

static double now(void)
{
	struct timespec t;
	clock_gettime(CLOCK_MONOTONIC, &t);
	return t.tv_sec + t.tv_nsec * 1e-9;
}

static uint64_t now_ns(void)
{
	struct timespec t;
	clock_gettime(CLOCK_MONOTONIC, &t);
	return (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec;
}

typedef struct { float s[CX_CTX_TOP]; int64_t step[CX_CTX_TOP]; } top4;

static inline void top_insert(top4 *t, float s, int64_t step)
{
	int i = CX_CTX_TOP - 1;
	if (s <= t->s[i]) return;
	for (; i > 0 && s > t->s[i - 1]; i--) { t->s[i] = t->s[i - 1]; t->step[i] = t->step[i - 1]; }
	t->s[i] = s;
	t->step[i] = step;
}

/* q . k over 64 lanes, AVX2 */
static inline float dot64(const float *q, const float *k)
{
	__m256 a0 = _mm256_mul_ps(_mm256_loadu_ps(q), _mm256_loadu_ps(k));
	__m256 a1 = _mm256_mul_ps(_mm256_loadu_ps(q + 8), _mm256_loadu_ps(k + 8));
	a0 = _mm256_fmadd_ps(_mm256_loadu_ps(q + 16), _mm256_loadu_ps(k + 16), a0);
	a1 = _mm256_fmadd_ps(_mm256_loadu_ps(q + 24), _mm256_loadu_ps(k + 24), a1);
	a0 = _mm256_fmadd_ps(_mm256_loadu_ps(q + 32), _mm256_loadu_ps(k + 32), a0);
	a1 = _mm256_fmadd_ps(_mm256_loadu_ps(q + 40), _mm256_loadu_ps(k + 40), a1);
	a0 = _mm256_fmadd_ps(_mm256_loadu_ps(q + 48), _mm256_loadu_ps(k + 48), a0);
	a1 = _mm256_fmadd_ps(_mm256_loadu_ps(q + 56), _mm256_loadu_ps(k + 56), a1);
	a0 = _mm256_add_ps(a0, a1);
	__m128 h = _mm_add_ps(_mm256_castps256_ps128(a0), _mm256_extractf128_ps(a0, 1));
	h = _mm_add_ps(h, _mm_movehl_ps(h, h));
	h = _mm_add_ss(h, _mm_shuffle_ps(h, h, 1));
	return _mm_cvtss_f32(h);
}

static inline int nearest(int h, const float *x)
{
	const float *C = cent + (size_t)h * NC * CX_DS;
	int best = 0;
	float bs = -3.0e38f;
	for (int c = 0; c < NC; c++) { const float s = dot64(x, C + (size_t)c * CX_DS); if (s > bs) { bs = s; best = c; } }
	return best;
}

static inline void file_under(int h, int c, int64_t n)
{
	const size_t li = (size_t)h * NC + c;
	lists[li * cap + cursor[li]] = (uint32_t)n;
	cursor[li] = (uint16_t)((cursor[li] + 1) % cap);
	if (fill[li] < cap) fill[li]++;
}

/* head h at step n: the answer into the part, then the step stored into the ring and the index */
static void serve_head(int h, int64_t n)
{
	const float *qkv = (const float *)(part + CX_CTX_OUT(n & 1)) + (size_t)h * CX_CTX_KV;
	const float *q = qkv, *k = qkv + CX_DS, *v = qkv + 2 * CX_DS;
	const float *Kh = keys + (size_t)h * W * CX_DS;
	top4 t;
	int scored = 0;
	for (int i = 0; i < CX_CTX_TOP; i++) { t.s[i] = -3.0e38f; t.step[i] = -1; }
	/* the exact part: the last RECENT steps */
	const int64_t lo = n - RECENT > 0 ? n - RECENT : 0;
	for (int64_t s = lo; s < n; s++) {
		if (slot_step[s & WM] != s) continue;
		top_insert(&t, dot64(q, Kh + (size_t)(s & WM) * CX_DS), s);
		scored++;
	}
	/* the far part: the PROBE nearest centroids' lists, their entries older than the exact window and still in the ring */
	if (n - RECENT > 0 && seeded[h] == NC) {
		const float *C = cent + (size_t)h * NC * CX_DS;
		float cs[NC];
		for (int c = 0; c < NC; c++) cs[c] = dot64(q, C + (size_t)c * CX_DS);
		for (int p = 0; p < probe; p++) {
			int best = -1;
			float bs = -3.0e38f;
			for (int c = 0; c < NC; c++) if (cs[c] > bs) { bs = cs[c]; best = c; }
			if (best < 0) break;
			cs[best] = -3.0e38f;
			const size_t li = (size_t)h * NC + best;
			const uint32_t *Lp = lists + li * cap;
			for (int i = 0; i < fill[li]; i++) {
				const int64_t s = Lp[i];
				if (s >= n - RECENT || slot_step[s & WM] != s) continue;
				top_insert(&t, dot64(q, Kh + (size_t)(s & WM) * CX_DS), s);
				scored++;
			}
		}
	}
	uint8_t *a = part + CX_CTX_IN(n & 1) + (size_t)h * CX_CTX_ANS_HEAD;
	float *out_v = (float *)a, *out_s = (float *)(a + CX_CTX_TOP * CX_DS * 4);
	uint32_t *out_i = (uint32_t *)(a + CX_CTX_TOP * CX_DS * 4 + 16);
	for (int i = 0; i < CX_CTX_TOP; i++) {
		if (t.step[i] < 0) { memset(out_v + i * CX_DS, 0, CX_DS * 4); out_s[i] = -3.0e38f; out_i[i] = CX_CTX_NONE; continue; }
		memcpy(out_v + i * CX_DS, vals + ((size_t)h * W + (t.step[i] & WM)) * CX_DS, CX_DS * 4);
		out_s[i] = t.s[i];
		out_i[i] = (uint32_t)t.step[i];
	}
	__atomic_fetch_add(&cand_total, scored, __ATOMIC_RELAXED);
	/* store this step: the ring slot; the index: a seed centroid while seeding, else the nearest centroid moved and the key filed */
	const size_t slot = (size_t)(n & WM);
	memcpy(keys + ((size_t)h * W + slot) * CX_DS, k, CX_DS * 4);
	memcpy(vals + ((size_t)h * W + slot) * CX_DS, v, CX_DS * 4);
	float *C = cent + (size_t)h * NC * CX_DS;
	if (seeded[h] < NC) {
		memcpy(C + (size_t)seeded[h] * CX_DS, k, CX_DS * 4);
		file_under(h, seeded[h], n);
		seeded[h]++;
		return;
	}
	const int c = nearest(h, k);
	float *ce = C + (size_t)c * CX_DS;
	double nn = 0;
	for (int d = 0; d < CX_DS; d++) { ce[d] += eta * (k[d] - ce[d]); nn += (double)ce[d] * ce[d]; }
	const float inv = nn > 0 ? (float)(1.0 / sqrt(nn)) : 0;
	for (int d = 0; d < CX_DS; d++) ce[d] *= inv;
	file_under(h, c, n);
}

typedef struct { int h0, h1; } span;

static void *worker(void *arg)
{
	const span *sp = arg;
	for (int gen = 1;; gen++) {
		while (__atomic_load_n(&go_gen, __ATOMIC_ACQUIRE) < gen) _mm_pause();
		const int64_t n = step_now;
		if (n < 0) return NULL;
		for (int h = sp->h0; h < sp->h1; h++) serve_head(h, n);
		__atomic_fetch_add(&done_count, 1, __ATOMIC_ACQ_REL);
	}
}

static void on_signal(int sig) { (void)sig; stop_now = 1; }

static void ring_clear(void)
{
	for (int i = 0; i < W; i++) slot_step[i] = -1;
	memset(fill, 0, (size_t)CX_CTX_HEADS * NC * sizeof *fill);
	memset(cursor, 0, (size_t)CX_CTX_HEADS * NC * sizeof *cursor);
	memset(seeded, 0, (size_t)CX_CTX_HEADS * sizeof *seeded);
}

/* the self-test's step n: synthetic q, k, v of every head into the slot, as the card's DMA would leave them */
static void selftest_step(int64_t n)
{
	float *slot = (float *)(part + CX_CTX_OUT(n & 1));
	for (int h = 0; h < CX_CTX_HEADS; h++) {
		float *qkv = slot + (size_t)h * CX_CTX_KV;
		double nn = 0;
		for (int d = 0; d < CX_DS; d++) {
			qkv[d] = cx_ctx_val((uint32_t)n, (uint32_t)h, (uint32_t)d, 2);
			qkv[CX_DS + d] = cx_ctx_val((uint32_t)n, (uint32_t)h, (uint32_t)d, 0);
			nn += (double)qkv[CX_DS + d] * qkv[CX_DS + d];
			qkv[2 * CX_DS + d] = cx_ctx_val((uint32_t)n, (uint32_t)h, (uint32_t)d, 1);
		}
		const float inv = (float)(1.0 / sqrt(nn));   /* the model's keys are unit-norm */
		for (int d = 0; d < CX_DS; d++) qkv[CX_DS + d] *= inv;
	}
	ST64(CX_CTX_C_SEQ, (uint64_t)n + 1);
}

int main(int argc, char **argv)
{
	for (int i = 1; i < argc; i++) {
		if (!strcmp(argv[i], "--threads") && i + 1 < argc) nthreads = atoi(argv[++i]);
		else if (!strcmp(argv[i], "--probe") && i + 1 < argc) probe = atoi(argv[++i]);
		else if (!strcmp(argv[i], "--eta") && i + 1 < argc) eta = (float)atof(argv[++i]);
		else if (!strcmp(argv[i], "--cap") && i + 1 < argc) cap = atoi(argv[++i]);
		else if (!strcmp(argv[i], "--steps") && i + 1 < argc) max_steps = atol(argv[++i]);
		else if (!strcmp(argv[i], "--selftest") && i + 1 < argc) { selftest = atol(argv[++i]); max_steps = selftest; }
		else if (!strcmp(argv[i], "--quiet")) quiet = 1;
		else { fprintf(stderr, "usage: ctxd [--threads T] [--probe P] [--eta X] [--cap M] [--steps N] [--selftest N] [--quiet]\n"); return 2; }
	}
	if (nthreads < 1 || nthreads > CX_CTX_HEADS || probe < 1 || probe > NC || cap < 16 || cap > 65535 || !(eta > 0 && eta < 1)) {
		fprintf(stderr, "ctxd: THREADS 1..%d, PROBE 1..%d, CAP 16..65535, ETA in (0, 1)\n", CX_CTX_HEADS, NC);
		return 2;
	}
	const int fd = open("/dev/shm/phi-hostmem", O_RDWR);
	if (fd < 0) { perror("/dev/shm/phi-hostmem"); return 1; }
	part = mmap(NULL, CX_PART_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, (off_t)CX_PART_MIB << 20);
	close(fd);
	if (part == MAP_FAILED) { perror("mmap"); return 1; }
	const size_t ring_floats = (size_t)CX_CTX_HEADS * W * CX_DS;
	keys = aligned_alloc(64, ring_floats * sizeof *keys);
	vals = aligned_alloc(64, ring_floats * sizeof *vals);
	slot_step = malloc((size_t)W * sizeof *slot_step);
	cent = aligned_alloc(64, (size_t)CX_CTX_HEADS * NC * CX_DS * sizeof *cent);
	lists = malloc((size_t)CX_CTX_HEADS * NC * cap * sizeof *lists);
	cursor = malloc((size_t)CX_CTX_HEADS * NC * sizeof *cursor);
	fill = malloc((size_t)CX_CTX_HEADS * NC * sizeof *fill);
	seeded = malloc((size_t)CX_CTX_HEADS * sizeof *seeded);
	if (!keys || !vals || !slot_step || !cent || !lists || !cursor || !fill || !seeded) { fprintf(stderr, "ctxd: out of memory\n"); return 1; }
	memset(keys, 0, ring_floats * sizeof *keys);   /* touched now: the first steps must not pay the page faults */
	memset(vals, 0, ring_floats * sizeof *vals);
	memset(cent, 0, (size_t)CX_CTX_HEADS * NC * CX_DS * sizeof *cent);
	memset(lists, 0, (size_t)CX_CTX_HEADS * NC * cap * sizeof *lists);
	ring_clear();
	ST64(CX_CTX_H_SEQ, 0);
	ST64(CX_CTX_C_SEQ, 0);
	ST64(CX_CTX_RESET_ACK, LD64(CX_CTX_RESET));
	ST64(CX_CTX_ALIVE, now_ns());
	signal(SIGINT, on_signal);
	signal(SIGTERM, on_signal);
	pthread_t tid[CX_CTX_HEADS];
	span sp[CX_CTX_HEADS];
	for (int t = 0; t < nthreads; t++) {
		sp[t].h0 = CX_CTX_HEADS * t / nthreads;
		sp[t].h1 = CX_CTX_HEADS * (t + 1) / nthreads;
		pthread_create(&tid[t], NULL, worker, &sp[t]);
	}
	printf("ctxd: window %d steps a head (%zu B of keys and values), exact over the last %d, online IVF %d centroids x %d probed (eta %.3f, lists of %d), %d threads%s%s\n",
	       W, 2 * ring_floats * sizeof *keys, RECENT, NC, probe, eta, cap, nthreads, max_steps >= 0 ? "" : "; until killed", selftest ? "; self-test, no card" : "");
	fflush(stdout);
	double t_sum = 0, t_max = 0;
	long timed = 0, cand_at = 0;
	int64_t n = 0;
	uint64_t last_beat = now_ns();
	while (!stop_now && (max_steps < 0 || n < max_steps)) {
		const uint64_t rq = LD64(CX_CTX_RESET);
		if (rq != LD64(CX_CTX_RESET_ACK)) { ring_clear(); ST64(CX_CTX_RESET_ACK, rq); if (!quiet) { printf("ctxd: the ring emptied for session %llu at step %lld\n", (unsigned long long)rq, (long long)n); fflush(stdout); } }
		if (selftest) selftest_step(n);
		if (LD64(CX_CTX_C_SEQ) < (uint64_t)n + 1) {
			const uint64_t t = now_ns();
			if (t - last_beat > 100000000ull) { ST64(CX_CTX_ALIVE, t); last_beat = t; }
			_mm_pause();
			continue;
		}
		const double a = now();
		step_now = n;
		done_count = 0;
		__atomic_store_n(&go_gen, (int)(n + 1), __ATOMIC_RELEASE);
		while (__atomic_load_n(&done_count, __ATOMIC_ACQUIRE) < nthreads) _mm_pause();
		slot_step[n & WM] = n;
		ST64(CX_CTX_H_SEQ, (uint64_t)n + 1);
		last_beat = now_ns();
		ST64(CX_CTX_ALIVE, last_beat);
		const double b = now();
		if (n >= 10) { t_sum += b - a; if (b - a > t_max) t_max = b - a; timed++; }
		n++;
		if (!quiet && n % 10000 == 0) {
			printf("ctxd: %lld steps; service %.1f us a step (max %.1f) over the last %ld; %.0f candidates a head-step\n", (long long)n, t_sum * 1e6 / timed, t_max * 1e6, timed,
			       (double)(cand_total - cand_at) / (10000.0 * CX_CTX_HEADS));
			fflush(stdout);
			t_sum = 0; t_max = 0; timed = 0; cand_at = cand_total;
		}
	}
	step_now = -1;
	__atomic_store_n(&go_gen, (int)(n + 1), __ATOMIC_RELEASE);
	for (int t = 0; t < nthreads; t++) pthread_join(tid[t], NULL);
	if (timed) printf("ctxd: %lld steps served; service %.1f us a step (max %.1f) over the last %ld; %.0f candidates a head-step\n", (long long)n, t_sum * 1e6 / timed, t_max * 1e6, timed,
	                  (double)(cand_total - cand_at) / ((double)timed * CX_CTX_HEADS));
	else printf("ctxd: %lld steps served\n", (long long)n);
	return 0;
}
