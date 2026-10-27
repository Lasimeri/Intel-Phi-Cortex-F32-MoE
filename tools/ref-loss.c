/*
 * ref-loss: the mean next-token loss of a checkpoint over a token stream, on the host in double (tools/cref.h: the
 * 120 columns, the bus off as the lineage was trained). The measurement of what a change to the weights costs, such
 * as the 3-bit quantisation of tools/quant3.c, and of a run's held-out loss while the card is busy training.
 *   ref-loss CKPT IDS.u16 NTOK [SKIP] [--ngram TABLES GATES] [--moe POOL.f32 [--moe-bias X]]
 * Steps the stream from token SKIP (default 0) for NTOK tokens with the fast weights carried across the whole pass
 * (the state starts from zero, as a session does) and prints the loss in nats and bits per token and the top-1
 * accuracy, with a running line every 5,000 tokens. The last token has no target: NTOK - 1 predictions. With
 * --ngram the model is read with its n-gram memory: TABLES = the run's tables (host/ngtables.h, mapped read-only;
 * the live file in /dev/shm, or a tables-N.f32 copy) and GATES = its gates (ckpt-N-gates.f32, 2 x 128 F32); the
 * rows of each step's suffix (the two ids before it in the stream, CX_NG_PAD before SKIP) enter as the card does it.
 * With --moe the model is read as the MoE model (lib/cx-moe.h): POOL.f32 = a pool the daemon pushed (pool-N.f32,
 * 4,293,918,720 B, mapped read-only), the fixed seeded keys, the home bonus X (default 0: the keys alone decide, as
 * a run that has lowered its bonus to 0 routes; the bonus of the run's last report for a checkpoint taken before).
 * The checkpoint's own MLP regions (the home experts) are not used then; with the bonus high enough to route every
 * head home the result equals the dense reading of the same checkpoint.
 */
#include "cref.h"
#include "../host/ngtables.h"
#include "../lib/cx-proto.h"

#include <stdio.h>
#include <time.h>

#define WINDOW 32

static int load_ckpt(cref *C, const char *path)
{
	FILE *f = fopen(path, "rb");
	if (!f) { perror(path); return -1; }
	float *buf = malloc(CX_STAGE_FLOATS * sizeof *buf);
	if (fread(buf, sizeof *buf, CX_STAGE_FLOATS, f) != CX_STAGE_FLOATS) { fprintf(stderr, "ref-loss: short checkpoint %s\n", path); return -1; }
	fclose(f);
	const float *p = buf;
	for (long i = 0; i < (long)C->N * CX_NPARAM; i++) C->prm[i] = p[i];
	p += (long)C->N * CX_NPARAM;
	for (int i = 0; i < CX_DU; i++) C->cg[i] = p[i];
	p += CX_DU;
	/* the file holds the embedding in blocks of 16 vocabulary rows (the card's ETB layout) */
	for (int v = 0; v < CX_VOCAB; v++) for (int c = 0; c < CX_DE; c++) C->E[(long)v * CX_DE + c] = p[(((long)v >> 4) * CX_DE + c) * 16 + (v & 15)];
	free(buf);
	return 0;
}

static int load_gates(const char *path, float *gates)
{
	FILE *f = fopen(path, "rb");
	if (!f || fread(gates, sizeof *gates, CX_NG_ORDERS * CX_DE, f) != CX_NG_ORDERS * CX_DE) { perror(path); return -1; }
	fclose(f);
	return 0;
}

static double now(void)
{
	struct timespec t;
	clock_gettime(CLOCK_MONOTONIC, &t);
	return (double)t.tv_sec + t.tv_nsec * 1e-9;
}

int main(int argc, char **argv)
{
	if (argc < 4) { fprintf(stderr, "usage: ref-loss CKPT IDS.u16 NTOK [SKIP] [--ngram TABLES GATES] [--moe POOL.f32 [--moe-bias X]]\n"); return 2; }
	const long ntok = atol(argv[3]);
	long skip = 0;
	const char *tables = NULL, *gates_path = NULL, *pool_path = NULL;
	double pool_bias = 0;
	for (int i = 4; i < argc; i++) {
		if (!strcmp(argv[i], "--ngram") && i + 2 < argc) { tables = argv[++i]; gates_path = argv[++i]; }
		else if (!strcmp(argv[i], "--moe") && i + 1 < argc) pool_path = argv[++i];
		else if (!strcmp(argv[i], "--moe-bias") && i + 1 < argc) pool_bias = atof(argv[++i]);
		else if (argv[i][0] != '-') skip = atol(argv[i]);
		else { fprintf(stderr, "ref-loss: unknown option %s\n", argv[i]); return 2; }
	}
	cref C;
	cref_alloc(&C, CX_NCOL, WINDOW);
	if (load_ckpt(&C, argv[1])) return 1;
	static float keys[CX_EX_KEYFLOATS];
	if (pool_path) {
		const int pfd = open(pool_path, O_RDONLY);
		struct stat st;
		if (pfd < 0 || fstat(pfd, &st)) { perror(pool_path); return 1; }
		if ((size_t)st.st_size != CX_EX_POOL_BYTES) { fprintf(stderr, "ref-loss: %s is %lld bytes, not %zu (a pool of %u experts)\n", pool_path, (long long)st.st_size, CX_EX_POOL_BYTES, CX_EX_N); return 1; }
		/* populated: the routed reads are random over 4 GiB and a pool on the HDD cost a seek per off-home expert (176 to
		 * 181 s per 27,447-token reading on 2026-10-27 against 6.4 s from the page cache); one sequential read instead */
		const float *pool = mmap(NULL, CX_EX_POOL_BYTES, PROT_READ, MAP_SHARED | MAP_POPULATE, pfd, 0);
		if (pool == MAP_FAILED) { perror(pool_path); return 1; }
		close(pfd);
		cx_ex_keys_init(keys);
		cref_moe_pool(&C, pool, keys, pool_bias);
		printf("ref-loss: MoE on: pool %s, the seeded keys, home bonus %g\n", pool_path, pool_bias);
	}
	ngtables NG;
	static float gates[CX_NG_ORDERS * CX_DE];
	if (tables) {
		if (ng_open(&NG, tables, 0) || load_gates(gates_path, gates)) return 1;
		cref_ng_tables(&C, (const float *const *)NG.T, gates);
		double gsum[CX_NG_ORDERS] = { 0 };
		for (int o = 0; o < CX_NG_ORDERS; o++) for (int i = 0; i < CX_DE; i++) gsum[o] += gates[o * CX_DE + i];
		printf("ref-loss: n-gram memory on: %s, gates %s (means %.3f %.3f)\n", tables, gates_path, gsum[0] / CX_DE, gsum[1] / CX_DE);
	}
	FILE *f = fopen(argv[2], "rb");
	if (!f) { perror(argv[2]); return 1; }
	uint16_t *tok = malloc((size_t)ntok * 2);
	fseek(f, skip * 2, SEEK_SET);
	if (fread(tok, 2, (size_t)ntok, f) != (size_t)ntok) { fprintf(stderr, "ref-loss: %s holds fewer than %ld ids after %ld\n", argv[2], ntok, skip); return 1; }
	fclose(f);
	static float zero_bus[CX_DB * CX_NBUS];
	const float *ext[WINDOW];
	for (int t = 0; t < WINDOW; t++) ext[t] = zero_bus;
	C.ext = ext;
	C.bus_mode = CREF_BUS_EXT;
	double *l = malloc(CX_VOCAB * sizeof *l);
	double nll = 0;
	long hits = 0;
	const double t0 = now();
	for (long s = 0; s + 1 < ntok; s++) {
		const int t = (int)(s % WINDOW);
		cref_step(&C, t, tok[s], s >= 1 ? tok[s - 1] : CX_NG_PAD, s >= 2 ? tok[s - 2] : CX_NG_PAD);
		const double *hc = C.cs[t].hc;
		int best = 0;
		double max = -1e300;
		for (int v = 0; v < CX_VOCAB; v++) {
			const double *row = C.E + (long)v * CX_DE;
			double x = 0;
			for (int c = 0; c < CX_DE; c++) x += row[c] * hc[c];
			l[v] = x;
			if (x > max) { max = x; best = v; }
		}
		double sum = 0;
		for (int v = 0; v < CX_VOCAB; v++) sum += exp(l[v] - max);
		const int next = tok[s + 1];
		nll += max + log(sum) - l[next];
		hits += best == next;
		if ((s + 1) % 5000 == 0) {
			printf("ref-loss: %ld tokens: %.4f nats = %.4f bits/token, top-1 %.4f, %.1f tok/s\n", s + 1, nll / (s + 1), nll / (s + 1) / log(2.0), (double)hits / (s + 1), (s + 1) / (now() - t0));
			fflush(stdout);
		}
	}
	const long n = ntok - 1;
	char how[160] = "";
	if (tables) strcat(how, " with the memory");
	if (pool_path) snprintf(how + strlen(how), sizeof how - strlen(how), " with the pool (bonus %g)", pool_bias);
	printf("ref-loss: %s on %s[%ld..%ld)%s: %.4f nats = %.4f bits/token over %ld predictions, top-1 %.4f, %.1f s\n", argv[1], argv[2], skip, skip + ntok, how, nll / n, nll / n / log(2.0), n, (double)hits / n, now() - t0);
	return 0;
}
