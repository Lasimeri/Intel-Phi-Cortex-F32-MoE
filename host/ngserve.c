/*
 * ngserve: the host side of the n-gram memory (PLAN-MOE.md, MoE stage 1). Owns the tables in host RAM (CX_NG_ORDERS
 * orders x CX_NG_ROWS rows x 128 F32) and the token stream; per chunk of L steps it gathers the 2 L rows the card needs
 * into the ring in the part (lib/cx-proto.h), and applies the row gradients the card DMAs back.
 *
 *   ngserve TOKENS.u16 L NCHUNKS [--init gate|zero] [--lr X] [--dump DIR] [--valid VALID.u16 NVCHUNKS]
 *
 * --init gate: the deterministic small tables of lib/cx-col.h (the gate builds), zero: training from scratch.
 * --lr X: the rows' SGD step (0 = gradients only). --dump DIR: the last chunk's rows and gradients as DIR/cl-ngrows.f32
 * and DIR/cl-ngrad.f32 (L x 2 x 128 floats each, for ctrain-check). --valid: after NCHUNKS training chunks, serve
 * NVCHUNKS chunks of the validation stream (forward only on the card: their gradients are ignored).
 */
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>
#include "../lib/cx-col.h"
#include "../lib/cx-proto.h"

static uint8_t *part;
#define LD64(o) __atomic_load_n((uint64_t *)(part + (o)), __ATOMIC_ACQUIRE)
#define ST64(o, x) __atomic_store_n((uint64_t *)(part + (o)), (x), __ATOMIC_RELEASE)

static float *T[CX_NG_ORDERS];
static uint16_t *tok, *vtok;
static long ntok, nvtok;
static int L;

static uint32_t at(const uint16_t *s, long n, long i) { return i >= 0 && i < n ? s[i] : CX_NG_PAD; }

/* the rows of chunk c of stream s (train or valid) into the row slot c & 1 */
static void gather(long c, const uint16_t *s, long n, long c0)
{
	float *slot = (float *)(part + CX_NGR_ROWS(c & 1));
	for (int t = 0; t < L; t++) {
		const long pos = (c - c0) * L + t;
		const uint32_t id = at(s, n, pos), p1 = at(s, n, pos - 1), p2 = at(s, n, pos - 2);
		for (int o = 0; o < CX_NG_ORDERS; o++) memcpy(slot + ((long)t * CX_NG_ORDERS + o) * CX_DE, T[o] + (size_t)cx_ng_hash(id, p1, p2, o + 2) * CX_DE, CX_DE * 4);
	}
	__atomic_thread_fence(__ATOMIC_RELEASE);
}

static void apply(long c, const uint16_t *s, long n, long c0, const float *grad, float lr)
{
	for (int t = 0; t < L; t++) {
		const long pos = (c - c0) * L + t;
		const uint32_t id = at(s, n, pos), p1 = at(s, n, pos - 1), p2 = at(s, n, pos - 2);
		for (int o = 0; o < CX_NG_ORDERS; o++) {
			float *row = T[o] + (size_t)cx_ng_hash(id, p1, p2, o + 2) * CX_DE;
			const float *g = grad + ((long)t * CX_NG_ORDERS + o) * CX_DE;
			for (int i = 0; i < CX_DE; i++) row[i] -= lr * g[i];
		}
	}
}

static int save(const char *dir, const char *name, const void *p, size_t bytes)
{
	char path[4096];
	snprintf(path, sizeof path, "%s/%s", dir, name);
	FILE *f = fopen(path, "wb");
	if (!f || fwrite(p, 1, bytes, f) != bytes || fclose(f)) { perror(path); return -1; }
	return 0;
}

static uint16_t *load_tokens(const char *path, long *n)
{
	FILE *f = fopen(path, "rb");
	if (!f) { perror(path); return NULL; }
	fseek(f, 0, SEEK_END);
	*n = ftell(f) / 2;
	fseek(f, 0, SEEK_SET);
	uint16_t *s = malloc((size_t)*n * 2);
	if (fread(s, 2, (size_t)*n, f) != (size_t)*n) { perror(path); return NULL; }
	fclose(f);
	return s;
}

int main(int argc, char **argv)
{
	if (argc < 4) { fprintf(stderr, "usage: ngserve TOKENS.u16 L NCHUNKS [--init gate|zero] [--lr X] [--dump DIR] [--valid VALID.u16 NVCHUNKS]\n"); return 2; }
	L = atoi(argv[2]);
	const long nchunks = atol(argv[3]);
	const char *init = "zero", *dump = NULL, *vpath = NULL;
	float lr = 0;
	long nvchunks = 0;
	for (int i = 4; i < argc; i++) {
		if (!strcmp(argv[i], "--init") && i + 1 < argc) init = argv[++i];
		else if (!strcmp(argv[i], "--lr") && i + 1 < argc) lr = (float)atof(argv[++i]);
		else if (!strcmp(argv[i], "--dump") && i + 1 < argc) dump = argv[++i];
		else if (!strcmp(argv[i], "--valid") && i + 2 < argc) { vpath = argv[++i]; nvchunks = atol(argv[++i]); }
		else { fprintf(stderr, "ngserve: unknown option %s\n", argv[i]); return 2; }
	}
	if (L != (int)(CX_NGR_BYTES / (CX_NG_ORDERS * CX_DE * 4))) { fprintf(stderr, "ngserve: the ring is laid out for L = %u\n", CX_NGR_BYTES / (CX_NG_ORDERS * CX_DE * 4)); return 2; }
	if (!(tok = load_tokens(argv[1], &ntok))) return 1;
	if (vpath && !(vtok = load_tokens(vpath, &nvtok))) return 1;
	for (int o = 0; o < CX_NG_ORDERS; o++) {
		T[o] = calloc((size_t)CX_NG_ROWS * CX_DE, sizeof(float));
		if (!T[o]) { fprintf(stderr, "ngserve: %zu bytes of tables\n", (size_t)CX_NG_ROWS * CX_DE * 4); return 1; }
		if (!strcmp(init, "gate")) cx_ng_init(T[o], o + 2);
	}
	const int fd = open("/dev/shm/phi-hostmem", O_RDWR);
	if (fd < 0) { perror("/dev/shm/phi-hostmem"); return 1; }
	part = mmap(NULL, CX_PART_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, (off_t)CX_PART_MIB << 20);
	if (part == MAP_FAILED) { perror("mmap"); return 1; }
	ST64(CX_NGR_C_SEQ, 0);
	ST64(CX_NGR_H_SEQ, 0);
	const long total = nchunks + nvchunks;
	printf("ngserve: %d orders x %u rows (%s), %ld train + %ld valid chunks of %d, lr %g, ring at part + %u MiB\n", CX_NG_ORDERS, CX_NG_ROWS, init, nchunks, nvchunks, L, lr, CX_NGR_BASE >> 20);
	fflush(stdout);
	/* chunks c < nchunks come from the training stream, the rest from the validation stream (its own positions) */
#define STREAM(c) ((c) < nchunks ? tok : vtok), ((c) < nchunks ? ntok : nvtok), ((c) < nchunks ? 0 : nchunks)
	for (long c = 0; c < 2 && c < total; c++) gather(c, STREAM(c));
	ST64(CX_NGR_H_SEQ, total < 2 ? (uint64_t)total : 2);
	float *last_rows = malloc(CX_NGR_BYTES), *last_grad = malloc(CX_NGR_BYTES);
	for (long c = 0; c < total; c++) {
		while (LD64(CX_NGR_C_SEQ) < (uint64_t)c + 1) usleep(50);
		const float *grad = (const float *)(part + CX_NGR_GRAD(c & 1));
		if (c < nchunks) {
			if (dump && c == nchunks - 1) { memcpy(last_rows, part + CX_NGR_ROWS(c & 1), CX_NGR_BYTES); memcpy(last_grad, grad, CX_NGR_BYTES); }
			if (lr != 0) apply(c, STREAM(c), grad, lr);
		}
		if (c + 2 < total) { gather(c + 2, STREAM(c + 2)); ST64(CX_NGR_H_SEQ, (uint64_t)c + 3); }
	}
	if (dump && (save(dump, "cl-ngrows.f32", last_rows, CX_NGR_BYTES) || save(dump, "cl-ngrad.f32", last_grad, CX_NGR_BYTES))) return 1;
	printf("ngserve: %ld chunks served\n", total);
	return 0;
}
