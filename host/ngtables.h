/*
 * ngtables.h: the n-gram memory's tables on the host (MoE stage 1, PLAN-MOE.md): CX_NG_ORDERS orders x CX_NG_ROWS
 * rows x CX_DE F32, order 2 then order 3, one file of 2,147,483,648 bytes (the layout ngserve --save wrote). The live
 * trainer keeps them as a file mapped shared: host/coach.c trains them in place, host/cortex.c and tools/ref-loss.c
 * map the same file read-only for a step's rows, so a client of the live model reads the rows as they are trained.
 * The file belongs in RAM (/dev/shm): mapped shared from a disk, every trained row is a dirty page the kernel writes
 * back at random (64 rows per chunk, 38 chunks per second), beyond a spinning disk's seeks.
 *
 * The arithmetic of a row's step (row -= lr * gradient, in F32, the gradient's floats in order) is the one
 * host/ngserve.c applied in the batch runs (it uses this header too), so a live run and a batch run from the same
 * tables and tokens stay bit-identical: the live memory's gate (tools/live-ng-gate.sh). Both are built with
 * -ffp-contract=off so no compiler fuses the multiply and the subtract.
 */
#ifndef CX_NGTABLES_H
#define CX_NGTABLES_H

#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/statfs.h>
#include <unistd.h>
#include "../lib/cx-col.h"

#define NG_TABLE_FLOATS ((size_t)CX_NG_ROWS * CX_DE)
#define NG_FILE_BYTES ((size_t)CX_NG_ORDERS * NG_TABLE_FLOATS * sizeof(float))

typedef struct {
	float *T[CX_NG_ORDERS];   /* the orders' tables */
	void *map;                /* the file's mapping, or NULL when the tables are the caller's memory */
	size_t bytes;
	int writable;
	long rows_zeroed;         /* rows found non-finite at a gather and zeroed (a RAM fault, a divergence) */
} ngtables;

/* the id at stream position i; CX_NG_PAD before the stream's start */
static inline uint32_t ng_at(const uint16_t *s, long n, long i) { return i >= 0 && i < n ? s[i] : CX_NG_PAD; }

/* the row of order o (0 = order 2, 1 = order 3) for the suffix (p2, p1, id) */
static inline float *ng_row(const ngtables *N, int o, uint32_t id, uint32_t p1, uint32_t p2)
{
	return N->T[o] + (size_t)cx_ng_hash(id, p1, p2, o + 2) * CX_DE;
}

/* the tables as memory the caller owns (ngserve's arrays, zero or gate-initialised): no file */
static inline void ng_from_memory(ngtables *N, float *const T[CX_NG_ORDERS], int writable)
{
	memset(N, 0, sizeof *N);
	for (int o = 0; o < CX_NG_ORDERS; o++) N->T[o] = T[o];
	N->writable = writable;
}

/* the file mapped shared: writable (the trainer's; created at the full size, zero rows, when absent or empty) or
 * read-only (a client's). A writable mapping of a file outside RAM gets a warning. 0 on success. */
static inline int ng_open(ngtables *N, const char *path, int writable)
{
	memset(N, 0, sizeof *N);
	const int fd = open(path, writable ? O_RDWR | O_CREAT : O_RDONLY, 0644);
	if (fd < 0) { fprintf(stderr, "n-gram tables %s: %s\n", path, strerror(errno)); return -1; }
	struct stat st;
	if (fstat(fd, &st)) { perror(path); close(fd); return -1; }
	if ((size_t)st.st_size != NG_FILE_BYTES) {
		if (!writable || st.st_size != 0) {
			fprintf(stderr, "n-gram tables %s: %lld bytes, not %zu (%d orders x %u rows x %d F32)\n", path, (long long)st.st_size, NG_FILE_BYTES, CX_NG_ORDERS, CX_NG_ROWS, CX_DE);
			close(fd);
			return -1;
		}
		if (ftruncate(fd, (off_t)NG_FILE_BYTES)) { perror(path); close(fd); return -1; }
	}
	N->bytes = NG_FILE_BYTES;
	N->map = mmap(NULL, N->bytes, writable ? PROT_READ | PROT_WRITE : PROT_READ, MAP_SHARED, fd, 0);
	if (N->map == MAP_FAILED) { perror(path); close(fd); N->map = NULL; return -1; }
	struct statfs fs;
	if (writable && !fstatfs(fd, &fs) && fs.f_type != 0x01021994 /* TMPFS_MAGIC */ && fs.f_type != 0x858458f6 /* RAMFS_MAGIC */)
		fprintf(stderr, "n-gram tables %s: not in RAM: every trained row is a dirty page the kernel writes back to the disk; use /dev/shm\n", path);
	close(fd);
	for (int o = 0; o < CX_NG_ORDERS; o++) N->T[o] = (float *)N->map + o * NG_TABLE_FLOATS;
	N->writable = writable;
	return 0;
}

static inline void ng_close(ngtables *N)
{
	if (N->map) munmap(N->map, N->bytes);
	N->map = NULL;
}

/* the mapping's dirty pages written to the file (the trainer, before a copy); 0 on success */
static inline int ng_sync(ngtables *N) { return N->map ? msync(N->map, N->bytes, MS_SYNC) : 0; }

/* the two rows of one step into out (2 x 128 F32, orders 2 then 3): the unit of a client's step and of a gather. A
 * row with a non-finite float goes out as zeros, and is zeroed in writable tables (counted), so a damaged row never
 * enters the model and never spreads through a gradient. */
static inline void ng_step_rows(ngtables *N, uint32_t id, uint32_t p1, uint32_t p2, float *out)
{
	for (int o = 0; o < CX_NG_ORDERS; o++) {
		float *row = ng_row(N, o, id, p1, p2), *dst = out + o * CX_DE;
		int finite = 1;
		for (int i = 0; i < CX_DE; i++) if (!isfinite(row[i])) { finite = 0; break; }
		if (finite) { memcpy(dst, row, CX_DE * sizeof *dst); continue; }
		memset(dst, 0, CX_DE * sizeof *dst);
		if (N->writable) { memset(row, 0, CX_DE * sizeof *row); N->rows_zeroed++; }
	}
}

/* chunk c of a stream s of n ids (its steps at positions c * L + t) gathered into a row slot (L x 2 x 128 F32) */
static inline void ng_gather(ngtables *N, const uint16_t *s, long n, long c, int L, float *slot)
{
	for (int t = 0; t < L; t++) {
		const long pos = c * L + t;
		ng_step_rows(N, ng_at(s, n, pos), ng_at(s, n, pos - 1), ng_at(s, n, pos - 2), slot + (long)t * CX_NG_ORDERS * CX_DE);
	}
}

/* the chunk's row gradients (L x 2 x 128 F32, as the card lays them out) applied: row -= lr * g */
static inline void ng_apply(ngtables *N, const uint16_t *s, long n, long c, int L, const float *grad, float lr)
{
	for (int t = 0; t < L; t++) {
		const long pos = c * L + t;
		const uint32_t id = ng_at(s, n, pos), p1 = ng_at(s, n, pos - 1), p2 = ng_at(s, n, pos - 2);
		for (int o = 0; o < CX_NG_ORDERS; o++) {
			float *row = ng_row(N, o, id, p1, p2);
			const float *g = grad + ((long)t * CX_NG_ORDERS + o) * CX_DE;
			for (int i = 0; i < CX_DE; i++) row[i] -= lr * g[i];
		}
	}
}

#endif
