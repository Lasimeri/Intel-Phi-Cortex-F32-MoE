/*
 * cluster.h: the 120-column cluster on the card (Stage 4), header only. One thread per column (team.h map: two per
 * core), one shared embedding in vocabulary blocks (ETB), one consensus gain cg. Per token step every column sees the
 * same token; the lateral bus holds every column's bc from the previous step (stop-gradient). Reference: tools/cref.h.
 *
 * Step t, thread k (3 barriers):
 *   column forward (card/train.h saves)  ->  bus[t+1][d][k] = bc_k[d];  votes[t&1][k] = g_k u_k
 *   A: group leaders (k % 10 == 0) sum their 10 votes into gsum[t&1][k/10]
 *   B: every thread sums the 12 group vectors (same order, same bits): hs, xsh, hc = RMSNorm(hs) cg
 *      own blocks of the head: logits, local max m_k, exps, local sum s_k -> pairs[t&1][k]; l_tgt by every thread
 *   C: every thread reads the 120 pairs: M = max m_k, S = sum s_k 2^((m_k - M) log2 e); loss; own dl = (p - onehot) / L
 * Chunk backward: dhc partials over own blocks -> two-level sum (D, E) -> per step dhs, du_k = g_k dhs,
 *   dgl_k = (u_k . dhs) g_k (1 - g_k) -> the column backward (train.h) -> parameter gradients (local) -> own ETB blocks'
 *   gradient (head part local; input rows summed by the block owner after F) -> Adam (local; cg by thread 0) -> G.
 * Block ownership: thread k owns ETB blocks k and k + 120, and 240 + k when k < 16 (256 blocks).
 */
#ifndef CX_CLUSTER_H
#define CX_CLUSTER_H

#include "team.h"
#include "train.h"
#include "cdma.h"
#include <stdio.h>
#include "../lib/cx-proto.h"

#define CL_N CX_COLUMNS
#define CL_GROUPS 12
#define CL_GSIZE 10
#define CL_BUS (CX_DB * CX_NBUS)   /* floats per bus matrix */
#define CL_MAXBLK 3

typedef struct {
	/* shared */
	float *ETB, *GET, *m_et, *v_et;
	float *ES;         /* row-major shadow of ETB for the lookups; thread k refreshes its own blocks' rows after Adam */
	float *cg, *m_cg, *v_cg;
	float *bus;        /* (CX_LMAX + 1) x [16][128]: bus[t] is the input of step t */
	float *votes;      /* 2 x CL_N x 128 */
	float *gsum;       /* 2 x CL_GROUPS x 128 */
	float *pairs;      /* 2 x CL_N x 16: lane 0 m_k, lane 1 s_k */
	float *dhcp;       /* CL_N x L x 128 */
	float *gdhc;       /* CL_GROUPS x L x 128 */
	float *dep;        /* CL_N x L x 128 */
	float *loss;       /* L (thread 0) */
	volatile uint32_t *argmax;   /* 2 x 16: gen mode, the global argmax token of step t at argmax[(t & 1) * 16] */
	float *logits_out;           /* daemon mode: 4096 floats in a card huge page; each thread stores its blocks' raw logits */
	cx_barrier bar;
	int L, bus_on;
	/* per thread */
	cx_train tr[CL_N];
	float *hs[CL_N], *xsh[CL_N], *hcp[CL_N], *dhc[CL_N], *lg[CL_N], *tgtrow[CL_N];
	int nblk[CL_N], blk[CL_N][CL_MAXBLK];
	uint64_t epoch[CL_N];
	/* MoE stage 1: the n-gram memory (ng NULL = off). Gates 2 x 128 with Adam moments and their gradient (thread 0);
	 * ngbuf = 2 slots x 32 KiB of rows DMAd in from the host; nggrad = 2 slots x 32 KiB of row gradients DMAd out;
	 * desum = the owner-summed input gradient of each step (L x 128, shared); part = the uncached part mapping. */
	float *ng, *m_ng, *v_ng, *gng, *ngbuf, *nggrad, *desum;
	uint64_t ngbuf_phys, nggrad_phys, ng_host_rows, ng_host_grad, ng_host_cseq, ng_dma_seq[2];
	volatile uint8_t *part;
	cdma *dma;
	long chunk;
} cluster;

static inline void cl_blocks(cluster *C, int k)
{
	C->nblk[k] = 0;
	C->blk[k][C->nblk[k]++] = k;
	C->blk[k][C->nblk[k]++] = k + CL_N;
	if (k < CX_VOCAB / 16 - 2 * CL_N) C->blk[k][C->nblk[k]++] = 2 * CL_N + k;
}

static inline int cl_owns(const cluster *C, int k, int g)
{
	for (int b = 0; b < C->nblk[k]; b++) if (C->blk[k][b] == g) return b;
	return -1;
}

static inline void cl_wait(cluster *C, int k) { cx_barrier_wait(&C->bar, k, ++C->epoch[k]); }

/* one cluster step for thread k: token id, next token (prefetch), target tgt; writes bus[t+1] */
/* n-gram ring, chunk start: thread 0 waits until the host has published chunk ch, DMAs its 32 KiB row slot into GDDR
 * (card-owned channel, measured ~10 us), then the team proceeds. No-op without the memory. */
/* a forward-only chunk (validation) sends no gradient DMA, so thread 0 publishes it as consumed directly */
static inline void cl_ng_done(cluster *C)
{
	cx_fence();
	*(volatile uint64_t *)(C->part + CX_NGR_C_SEQ) = (uint64_t)C->chunk + 1;
}

static inline void cl_ng_begin(cluster *C, int k, long ch)
{
	if (!C->ng) return;
	if (!k) {
		C->chunk = ch;
		while (__atomic_load_n((volatile uint64_t *)(C->part + CX_NGR_H_SEQ), __ATOMIC_ACQUIRE) < (uint64_t)ch + 1) cx_relax();
		const int s = (int)(ch & 1);
		const uint64_t seq = cdma_submit(C->dma, C->ng_host_rows + (uint64_t)s * CX_NGR_BYTES, C->ngbuf_phys + (uint64_t)s * CX_NGR_BYTES, CX_NGR_BYTES, 0, 0);
		if (!seq || cdma_wait(C->dma, seq, 5.0)) { fprintf(stderr, "cl_ng_begin: row DMA failed at chunk %ld\n", ch); }
		for (uint32_t i = 0; i < CX_NGR_BYTES / 4; i += 16) cx_evict1(C->ngbuf + (size_t)s * (CX_NGR_BYTES / 4) + i);
	}
	cl_wait(C, k);
}

static inline void cl_fwd_step(cluster *C, int k, int t, uint32_t id, uint32_t id_next, uint32_t tgt)
{
	cx_train *tr = &C->tr[k];
	const int L = C->L;
	const __m512 zero = _mm512_setzero_ps(), l2e = _mm512_set1_ps(CX_LOG2E), invL = _mm512_set1_ps(1.0f / L);
	const float *Bt = C->bus + (long)t * CL_BUS;
	float *bn = C->bus + (long)(t + 1) * CL_BUS;
	cx_pf_lines(C->ES + (long)tgt * CX_DE, CX_DE);
	if (C->ng) {
		const float *rows = C->ngbuf + (C->chunk & 1) * (CX_NGR_BYTES / 4);
		for (int n = 0; n < CX_NG_ORDERS; n++) {
			tr->ngrow[n] = rows + ((long)t * CX_NG_ORDERS + n) * CX_DE;
			tr->ngnext[n] = t + 1 < C->L ? rows + ((long)(t + 1) * CX_NG_ORDERS + n) * CX_DE : NULL;
		}
	}
	cx_train_fwd_step(tr, t, id, id_next, Bt);
#ifdef CX_NOCOLL
	(void)tgt; (void)L; (void)zero; (void)l2e; (void)invL; (void)bn;
	return;
#endif
	const float *A = tr->col.act;
	if (C->bus_on) for (int d = 0; d < CX_DB; d++) bn[d * CX_NBUS + k] = A[CX_A_BC + d];
	float *vk = C->votes + ((long)(t & 1) * CL_N + k) * CX_DU;
	const __m512 g = _mm512_load_ps(A + CX_A_G);
	for (int v = 0; v < 8; v++) _mm512_store_ps(vk + 16 * v, _mm512_mul_ps(g, _mm512_load_ps(A + CX_A_U + 16 * v)));
	cl_wait(C, k);   /* A */
	if (k % CL_GSIZE == 0) {
		const float *v0 = C->votes + ((long)(t & 1) * CL_N + k) * CX_DU;
		cx_pf_lines(v0, CL_GSIZE * CX_DU);
		__m512 acc[8];
		for (int v = 0; v < 8; v++) acc[v] = _mm512_load_ps(v0 + 16 * v);
		for (int j = 1; j < CL_GSIZE; j++) for (int v = 0; v < 8; v++) acc[v] = _mm512_add_ps(acc[v], _mm512_load_ps(v0 + (long)j * CX_DU + 16 * v));
		float *gs = C->gsum + ((long)(t & 1) * CL_GROUPS + k / CL_GSIZE) * CX_DU;
		for (int v = 0; v < 8; v++) _mm512_store_ps(gs + 16 * v, acc[v]);
	}
	cl_wait(C, k);   /* B */
	const float *g0 = C->gsum + (long)(t & 1) * CL_GROUPS * CX_DU;
	cx_pf_lines(g0, CL_GROUPS * CX_DU);
	__m512 hs[8], ss = zero;
	for (int v = 0; v < 8; v++) hs[v] = _mm512_load_ps(g0 + 16 * v);
	for (int j = 1; j < CL_GROUPS; j++) for (int v = 0; v < 8; v++) hs[v] = _mm512_add_ps(hs[v], _mm512_load_ps(g0 + (long)j * CX_DU + 16 * v));
	for (int v = 0; v < 8; v++) { _mm512_store_ps(C->hs[k] + (long)t * CX_DU + 16 * v, hs[v]); ss = _mm512_fmadd_ps(hs[v], hs[v], ss); }
	const __m512 xsh = cx_rsqrt(_mm512_fmadd_ps(cx_hsum(ss), _mm512_set1_ps(1.0f / CX_DU), _mm512_set1_ps(CX_EPS)));
	_mm512_store_ps(C->xsh[k] + (long)t * 16, xsh);
	float *hc = C->hcp[k] + (long)t * CX_DU;
	for (int v = 0; v < 8; v++) _mm512_store_ps(hc + 16 * v, _mm512_mul_ps(_mm512_mul_ps(hs[v], xsh), _mm512_load_ps(C->cg + 16 * v)));
	/* own blocks of the head */
	float *lg = C->lg[k] + (long)t * CL_MAXBLK * 16;
	__m512 mk = _mm512_set1_ps(-3.0e38f);
	for (int b = 0; b < C->nblk[k]; b++) {
		const float *e = C->ETB + (long)C->blk[k][b] * CX_BLOCK;
		__m512 acc = zero;
		for (int c = 0; c < CX_DE; c++) acc = _mm512_fmadd_ps(CX_BC(hc + c), _mm512_load_ps(e + c * 16), acc);
		_mm512_store_ps(lg + 16 * b, acc);
		if (C->logits_out) _mm512_store_ps(C->logits_out + (long)C->blk[k][b] * 16, acc);
		mk = cx_max(mk, acc);
	}
	mk = cx_hmax(mk);
	__m512 sk = zero;
	for (int b = 0; b < C->nblk[k]; b++) {
		const __m512 e = cx_exp2(_mm512_mul_ps(_mm512_sub_ps(_mm512_load_ps(lg + 16 * b), mk), l2e));
		_mm512_store_ps(lg + 16 * b, e);
		sk = _mm512_add_ps(sk, e);
	}
	sk = cx_hsum(sk);
	/* l_tgt = hc . ETB(tgt, :) */
	const float *trow = C->ES + (long)tgt * CX_DE;
	__m512 lt = zero;
	for (int v = 0; v < 8; v++) lt = _mm512_fmadd_ps(_mm512_load_ps(hc + 16 * v), _mm512_load_ps(trow + 16 * v), lt);
	lt = cx_hsum(lt);
	_mm512_store_ps(C->pairs + ((long)(t & 1) * CL_N + k) * 16, cx_blend(cx_blend(zero, 0x0001, mk), 0x0002, sk));
	cl_wait(C, k);   /* C */
	const float *p0 = C->pairs + (long)(t & 1) * CL_N * 16;
	cx_pf_lines(p0, CL_N * 16);
	float mb[CL_N + 8] __attribute__((aligned(64))), sb[CL_N + 8] __attribute__((aligned(64)));
	for (int j = 0; j < CL_N; j++) { mb[j] = p0[j * 16]; sb[j] = p0[j * 16 + 1]; }
	for (int j = CL_N; j < CL_N + 8; j++) { mb[j] = -3.0e38f; sb[j] = 0; }
	__m512 M = _mm512_load_ps(mb);
	for (int v = 1; v < CL_N / 16 + 1; v++) M = cx_max(M, _mm512_load_ps(mb + 16 * v));
	M = cx_hmax(M);
	__m512 S = zero;
	for (int v = 0; v < CL_N / 16 + 1; v++) S = _mm512_fmadd_ps(_mm512_load_ps(sb + 16 * v), cx_exp2(_mm512_mul_ps(_mm512_sub_ps(_mm512_load_ps(mb + 16 * v), M), l2e)), S);
	S = cx_hsum(S);
	if (!k) _mm512_mask_store_ps(C->loss + (t & ~15), (__mmask16)(1u << (t & 15)), _mm512_fmadd_ps(cx_log2(S), _mm512_set1_ps(CX_LN2), _mm512_sub_ps(M, lt)));
	const __m512 scale = _mm512_mul_ps(_mm512_mul_ps(cx_exp2(_mm512_mul_ps(_mm512_sub_ps(mk, M), l2e)), cx_rcp(S)), invL);
	for (int b = 0; b < C->nblk[k]; b++) {
		__m512 d = _mm512_mul_ps(_mm512_load_ps(lg + 16 * b), scale);
		if (C->blk[k][b] == (int)(tgt >> 4)) d = _mm512_mask_sub_ps(d, (__mmask16)(1u << (tgt & 15)), d, invL);
		_mm512_store_ps(lg + 16 * b, d);
	}
}

/* the chunk backward for thread k; ids[0..L] (ids[t] the input of step t) */
static inline void cl_bwd_chunk(cluster *C, int k, const uint32_t *ids, float b1, float b2, float lr_t, float eps_t)
{
	cx_train *tr = &C->tr[k];
	const int L = C->L;
	const __m512 zero = _mm512_setzero_ps(), one = _mm512_set1_ps(1.0f);
	/* dhc partials over own blocks: dhcp[k][t][c] = sum_b sum_lane ETB[g_b][c] dl_t[b] */
	float *dp = C->dhcp + (long)k * L * CX_DU;
	for (int b = 0; b < C->nblk[k]; b++) cx_pf_lines(C->ETB + (long)C->blk[k][b] * CX_BLOCK, CX_BLOCK);
	for (int t = 0; t < L; t++) {
		const float *lg = C->lg[k] + (long)t * CL_MAXBLK * 16;
		for (int c = 0; c < CX_DE; c++) {
			__m512 acc = zero;
			for (int b = 0; b < C->nblk[k]; b++) acc = _mm512_fmadd_ps(_mm512_load_ps(C->ETB + (long)C->blk[k][b] * CX_BLOCK + c * 16), _mm512_load_ps(lg + 16 * b), acc);
			_mm512_mask_store_ps(dp + (long)t * CX_DU + (c & ~15), (__mmask16)(1u << (c & 15)), cx_hsum(acc));
		}
	}
	cl_wait(C, k);   /* D */
	if (k % CL_GSIZE == 0) {
		const float *d0 = C->dhcp + (long)k * L * CX_DU;
		float *gd = C->gdhc + (long)(k / CL_GSIZE) * L * CX_DU;
		cx_pf_lines(d0, CL_GSIZE * L * CX_DU);
		for (int i = 0; i < L * CX_DU; i += 16) {
			__m512 acc = _mm512_load_ps(d0 + i);
			for (int j = 1; j < CL_GSIZE; j++) acc = _mm512_add_ps(acc, _mm512_load_ps(d0 + (long)j * L * CX_DU + i));
			_mm512_store_ps(gd + i, acc);
		}
	}
	cl_wait(C, k);   /* E */
	cx_pf_lines(C->gdhc, CL_GROUPS * L * CX_DU);
	for (int i = 0; i < L * CX_DU; i += 16) {
		__m512 acc = _mm512_load_ps(C->gdhc + i);
		for (int j = 1; j < CL_GROUPS; j++) acc = _mm512_add_ps(acc, _mm512_load_ps(C->gdhc + (long)j * L * CX_DU + i));
		_mm512_store_ps(C->dhc[k] + i, acc);
	}
	/* the steps in reverse */
	cx_train_chunk_begin(tr);
	cx_zero(tr->gcg, CX_DU);
	for (int t = L - 1; t >= 0; t--) {
		const float *A = tr->acts + (long)t * CX_NACT;
		cx_pf_lines(A, CX_NACT);
		cx_pf_lines(C->bus + (long)t * CL_BUS, CL_BUS);
		float dhs[CX_DU] __attribute__((aligned(64))), du[CX_DU] __attribute__((aligned(64)));
		cx_norm_bwd(C->dhc[k] + (long)t * CX_DU, C->hs[k] + (long)t * CX_DU, _mm512_load_ps(C->xsh[k] + (long)t * 16), C->cg, tr->gcg, dhs, 8, 1.0f / CX_DU);
		const __m512 g = _mm512_load_ps(A + CX_A_G);
		__m512 dg = zero;
		for (int v = 0; v < 8; v++) {
			const __m512 d = _mm512_load_ps(dhs + 16 * v);
			_mm512_store_ps(du + 16 * v, _mm512_mul_ps(g, d));
			dg = _mm512_fmadd_ps(_mm512_load_ps(A + CX_A_U + 16 * v), d, dg);
		}
		const __m512 dgl = _mm512_mul_ps(cx_hsum(dg), _mm512_mul_ps(g, _mm512_sub_ps(one, g)));
		cx_col_bwd_step(tr, t, C->bus + (long)t * CL_BUS, du, dgl);
	}
	cx_train_param_grads(tr);
	/* own blocks: GET[g][c] += sum_t hc_t[c] dl_t[g] (hc and dl private) */
	for (int b = 0; b < C->nblk[k]; b++) {
		float *G = C->GET + (long)C->blk[k][b] * CX_BLOCK;
		cx_pf_lines(G, CX_BLOCK);
		for (int tb = 0; tb < L; tb += 16) {
			__m512 d[16];
			for (int i = 0; i < 16; i++) d[i] = _mm512_load_ps(C->lg[k] + (long)(tb + i) * CL_MAXBLK * 16 + 16 * b);
			for (int c = 0; c < CX_DE; c++) {
				__m512 acc = _mm512_load_ps(G + c * 16);
				for (int i = 0; i < 16; i++) acc = _mm512_fmadd_ps(CX_BC(C->hcp[k] + (long)(tb + i) * CX_DU + c), d[i], acc);
				_mm512_store_ps(G + c * 16, acc);
			}
		}
	}
	/* input-row gradients: publish de_t, then the block owners sum them */
	for (int t = 0; t < L; t++) cx_copy(C->dep + ((long)k * L + t) * CX_DU, tr->gsave + (long)t * CX_NG + CX_G_DE, CX_DU);
	cl_wait(C, k);   /* F */
	for (int t = 0; t < L; t++) {
		const int g = (int)(ids[t] >> 4);
		if (cl_owns(C, k, g) < 0) continue;
		__m512 acc[8];
		for (int v = 0; v < 8; v++) acc[v] = zero;
		for (int j = 0; j < CL_N; j++) {
			const float *d = C->dep + ((long)j * L + t) * CX_DU;
			if (j + 4 < CL_N) cx_pf_lines(C->dep + ((long)(j + 4) * L + t) * CX_DU, CX_DU);
			for (int v = 0; v < 8; v++) acc[v] = _mm512_add_ps(acc[v], _mm512_load_ps(d + 16 * v));
		}
		float de[CX_DU] __attribute__((aligned(64)));
		for (int v = 0; v < 8; v++) _mm512_store_ps(de + 16 * v, acc[v]);
		if (C->ng) cx_copy(C->desum + (long)t * CX_DU, de, CX_DU);
		float *col = C->GET + (long)g * CX_BLOCK + (ids[t] & 15);
		for (int c = 0; c < CX_DE; c++) col[c * 16] += de[c];
	}
	/* n-gram memory: row gradients g_n * desum_t by thread t, the gate gradient by thread 0, then the DMA to the host */
	if (C->ng) {
		cl_wait(C, k);   /* desum complete */
		const int s = (int)(C->chunk & 1);
		float *gslot = C->nggrad + (size_t)s * (CX_NGR_BYTES / 4);
		if (!k && C->ng_dma_seq[s]) cdma_wait(C->dma, C->ng_dma_seq[s], 5.0);   /* the slot's previous DMA (two chunks ago) */
		cl_wait(C, k);
		if (k < L) {
			for (int n = 0; n < CX_NG_ORDERS; n++) {
				const float *g = C->ng + n * CX_DE, *ds = C->desum + (long)k * CX_DU;
				float *out = gslot + ((long)k * CX_NG_ORDERS + n) * CX_DE;
				for (int v = 0; v < CX_DE / 16; v++) _mm512_store_ps(out + 16 * v, _mm512_mul_ps(_mm512_load_ps(g + 16 * v), _mm512_load_ps(ds + 16 * v)));
			}
		}
		if (!k) {
			for (int n = 0; n < CX_NG_ORDERS; n++) {
				__m512 acc[CX_DE / 16];
				for (int v = 0; v < CX_DE / 16; v++) acc[v] = _mm512_load_ps(C->gng + n * CX_DE + 16 * v);
				for (int t = 0; t < L; t++) {
					const float *ds = C->desum + (long)t * CX_DU, *r = tr->ngr + ((long)t * CX_NG_ORDERS + n) * CX_DE;
					for (int v = 0; v < CX_DE / 16; v++) acc[v] = _mm512_fmadd_ps(_mm512_load_ps(ds + 16 * v), _mm512_load_ps(r + 16 * v), acc[v]);
				}
				for (int v = 0; v < CX_DE / 16; v++) _mm512_store_ps(C->gng + n * CX_DE + 16 * v, acc[v]);
			}
		}
		cl_wait(C, k);   /* gradients written */
		if (!k) {
			C->ng_dma_seq[s] = cdma_submit(C->dma, C->nggrad_phys + (uint64_t)s * CX_NGR_BYTES, C->ng_host_grad + (uint64_t)s * CX_NGR_BYTES, CX_NGR_BYTES, C->ng_host_cseq, (uint64_t)C->chunk + 1);
			if (!C->ng_dma_seq[s]) fprintf(stderr, "cl_bwd_chunk: gradient DMA submit failed at chunk %ld\n", C->chunk);
		}
	}
	/* Adam: own column, own blocks; cg by thread 0 (every thread holds the same gcg); lr_t < 0 = gradients only (gate) */
	if (lr_t < 0) { cl_wait(C, k); return; }
	cx_adam(tr->col.prm, tr->gprm, tr->gprm + ((CX_NPARAM + 15) & ~15), tr->gprm + 2 * ((CX_NPARAM + 15) & ~15), (CX_NPARAM + 15) & ~15, b1, b2, lr_t, eps_t);
	for (int b = 0; b < C->nblk[k]; b++) {
		const long o = (long)C->blk[k][b] * CX_BLOCK;
		cx_adam(C->ETB + o, C->GET + o, C->m_et + o, C->v_et + o, CX_BLOCK, b1, b2, lr_t, eps_t);
		cx_zero(C->GET + o, CX_BLOCK);
	}
	if (!k) cx_adam(C->cg, tr->gcg, C->m_cg, C->v_cg, CX_DU, b1, b2, lr_t, eps_t);
	if (!k && C->ng) { cx_adam(C->ng, C->gng, C->m_ng, C->v_ng, CX_NG_ORDERS * CX_DE, b1, b2, lr_t, eps_t); cx_zero(C->gng, CX_NG_ORDERS * CX_DE); }
	/* refresh the shadow rows of the own blocks (the blocks are L2-hot after Adam) */
	for (int b = 0; b < C->nblk[k]; b++) {
		const int g = C->blk[k][b];
		for (int l = 0; l < 16; l++) {
			const float *src = C->ETB + (long)g * CX_BLOCK + l;
			float *row = C->ES + ((long)g * 16 + l) * CX_DE;
			for (int c = 0; c < CX_DE; c++) row[c] = src[c * 16];
		}
	}
	cl_wait(C, k);   /* G */
}

/* gen mode: called right after cl_fwd_step; publishes the argmax token of step t (ties: the lowest thread wins by a
 * second pass over the pairs) and returns it after one more barrier */
static inline uint32_t cl_argmax(cluster *C, int k, int t)
{
	const float *p0 = C->pairs + (long)(t & 1) * CL_N * 16;
	float M = -3.0e38f;
	int owner = -1;
	for (int j = 0; j < CL_N; j++) if (p0[j * 16] > M) { M = p0[j * 16]; owner = j; }
	if (owner == k) {
		/* which of my lanes holds M: the exps hold 2^((l - m_k) log2 e) = 1 at the max */
		const float *lg = C->lg[k] + (long)t * CL_MAXBLK * 16;
		uint32_t best = 0;
		float bv = -1;
		for (int b = 0; b < C->nblk[k]; b++)
			for (int l = 0; l < 16; l++) if (lg[16 * b + l] > bv) { bv = lg[16 * b + l]; best = (uint32_t)C->blk[k][b] * 16 + l; }
		C->argmax[(t & 1) * 16] = best;
	}
	cl_wait(C, k);
	return C->argmax[(t & 1) * 16];
}

#endif
