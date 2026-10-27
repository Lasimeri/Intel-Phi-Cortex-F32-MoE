/*
 * col.h: one column's forward step on the 7120A vector unit (S3 clang -march=knc). Header only, single thread.
 * Every arithmetic instruction is a 512-bit VPU instruction (kncintrin.h names plus the raw KNC builtins the header
 * leaves out: vpermf32x4, vgmaxps, vfnmadd231ps, masked vmovaps as a blend, masked vexp223ps). Scalars reach the
 * VPU as memory broadcasts (vbroadcastss) from the activation block. Shapes and layout: lib/cx-col.h.
 *
 * CX_PF (default 1): vprefetch0 of the weight row two ahead of the one being used (Stage 0f rule).
 */
#ifndef CX_COL_KERNEL_H
#define CX_COL_KERNEL_H

#include "vpu.h"
#include "../lib/cx-col.h"

#ifndef CX_PF
#define CX_PF 1
#endif

static inline void cx_pf0(const void *p) { __asm__ volatile("vprefetch0 %0" : : "m"(*(const char *)p)); }

#define CX_BC(p) _mm512_extload_ps((p), _MM_UPCONV_PS_NONE, _MM_BROADCAST_1X16, _MM_HINT_NONE)

/* k ? b : a, lane by lane */
static inline __m512 cx_blend(__m512 a, __mmask16 k, __m512 b)
{
	return (__m512)__builtin_ia32_knc_vmovaps_zz((v16f)a, k, (v16f)b, 0);
}

static inline __m512 cx_max(__m512 a, __m512 b)
{
	return (__m512)__builtin_ia32_knc_vgmaxps_zzz((v16f)_mm512_undefined_ps(), CX_ALL, (v16f)a, (v16f)b, 0);
}

/* c - a * b */
static inline __m512 cx_fnmadd(__m512 a, __m512 b, __m512 c)
{
	return (__m512)__builtin_ia32_knc_vfnmadd231ps_zzz((v16f)c, CX_ALL, (v16f)a, (v16f)b, 0);
}

/* 128-bit lane permutes (327364 VPERMF32X4: destination block i takes source block imm[2i+1:2i]) */
#define CX_P128(v, imm) ((__m512)__builtin_ia32_knc_vpermf32x4_zzi((v16f)_mm512_undefined_ps(), CX_ALL, (v16f)(v), (imm)))

/* horizontal sum and max: the result in every lane */
static inline __m512 cx_hsum(__m512 v)
{
	v = _mm512_add_ps(v, _mm512_swizzle_ps(v, _MM_SWIZ_REG_CDAB));
	v = _mm512_add_ps(v, _mm512_swizzle_ps(v, _MM_SWIZ_REG_BADC));
	v = _mm512_add_ps(v, CX_P128(v, 0xB1));
	v = _mm512_add_ps(v, CX_P128(v, 0x4E));
	return v;
}

static inline __m512 cx_hmax(__m512 v)
{
	v = cx_max(v, _mm512_swizzle_ps(v, _MM_SWIZ_REG_CDAB));
	v = cx_max(v, _mm512_swizzle_ps(v, _MM_SWIZ_REG_BADC));
	v = cx_max(v, CX_P128(v, 0xB1));
	v = cx_max(v, CX_P128(v, 0x4E));
	return v;
}

/* 1 / (1 + 2^(-x log2 e)); exp2 saturates (x << 0 gives +inf, rcp +0; x >> 0 gives 0, rcp 1) */
static inline __m512 cx_sigmoid(__m512 x)
{
	const __m512 e = cx_exp2(_mm512_mul_ps(x, _mm512_set1_ps(-CX_LOG2E)));
	return cx_rcp(_mm512_add_ps(e, _mm512_set1_ps(1.0f)));
}

/* y[16 NV] = x[K]^T W[K][16 NV]: per k one broadcast and NV FMAs with the weight load folded in */
#define CX_GEMV(NV)                                                                                                  \
	static inline void cx_gemv##NV(const float *W, const float *x, int K, float *y)                                  \
	{                                                                                                                \
		__m512 a[NV];                                                                                                \
		_Pragma("unroll") for (int v = 0; v < NV; v++) a[v] = _mm512_setzero_ps();                                   \
		for (int k = 0; k < K; k++) {                                                                                \
			const float *w = W + (long)k * (16 * NV);                                                                \
			if (CX_PF) { _Pragma("unroll") for (int v = 0; v < NV; v++) cx_pf0(w + 2 * (16 * NV) + 16 * v); }        \
			const __m512 xb = CX_BC(x + k);                                                                          \
			_Pragma("unroll") for (int v = 0; v < NV; v++) a[v] = _mm512_fmadd_ps(xb, _mm512_load_ps(w + 16 * v), a[v]); \
		}                                                                                                            \
		_Pragma("unroll") for (int v = 0; v < NV; v++) _mm512_store_ps(y + 16 * v, a[v]);                            \
	}
CX_GEMV(1)
CX_GEMV(4)
CX_GEMV(6)
#if CX_DH == 64
#define CX_NVH 4
#define cx_gemv_h cx_gemv4
#elif CX_DH == 96
#define CX_NVH 6
#define cx_gemv_h cx_gemv6
#else
#error "CX_DH must be 64 or 96"
#endif
CX_GEMV(8)
CX_GEMV(13)

typedef struct {
	float *prm;   /* CX_NPARAM floats, 64-byte aligned */
	float *T;     /* CX_NT floats: T = S^T */
	float *act;   /* CX_NACT floats */
} cx_col;

/* one token step in two halves (the MoE head routes and fetches its expert between them, card/moe.h). e: this token's
 * embedding row (128). Bt: this step's bus input, [16][128]. The first half leaves [o ; x] at CX_A_MIN and the values
 * the backward needs at CX_A_R .. CX_A_D; the second leaves u at act + CX_A_U, g (all lanes) at CX_A_G, bc at CX_A_BC. */
static inline void cx_col_pre(cx_col *c, const float *e, const float *Bt)
{
	float *const A = c->act;
	const float *const P = c->prm;
	float *const T = c->T;

	/* p = e^T P_in; qb = p^T Wqb; scores s = qb^T Bt */
	cx_gemv4(P + CX_O_PIN, e, CX_DE, A + CX_A_P);
	cx_gemv1(P + CX_O_WQB, A + CX_A_P, CX_DP, A + CX_A_QB);
	cx_gemv8(Bt, A + CX_A_QB, CX_DB, A + CX_A_S);

	/* softmax over the 120 real slots: lanes 8..15 of the last vector are not columns */
	__m512 s[8], m;
	for (int v = 0; v < 8; v++) s[v] = _mm512_load_ps(A + CX_A_S + 16 * v);
	m = cx_max(cx_max(cx_max(s[0], s[1]), cx_max(s[2], s[3])), cx_max(s[4], s[5]));
	m = cx_max(m, s[6]);
	m = cx_hmax(cx_max(m, cx_blend(m, 0x00FF, s[7])));
	const __m512 l2e = _mm512_set1_ps(CX_LOG2E), zero = _mm512_setzero_ps();
	__m512 sum = zero;
	for (int v = 0; v < 8; v++) {
		const v16i q = __builtin_ia32_knc_vcvtfxpntps2dq_zzi((v16i)_mm512_undefined_epi32(), CX_ALL,
		                                                     (v16f)_mm512_mul_ps(_mm512_sub_ps(s[v], m), l2e), CX_FXP_8_24_RN, 0);
		s[v] = (__m512)__builtin_ia32_knc_vexp223ps_zz((v16f)zero, v < 7 ? CX_ALL : (unsigned short)0x00FF, (v16f)q, 0);
		sum = _mm512_add_ps(sum, s[v]);
	}
	const __m512 inv = cx_rcp(cx_hsum(sum));
	for (int v = 0; v < 8; v++) _mm512_store_ps(A + CX_A_ALPHA + 16 * v, _mm512_mul_ps(s[v], inv));

	/* r_d = alpha . Bt[d]: one lane per d, assembled by blends */
	__m512 r = zero;
	for (int d = 0; d < CX_DB; d++) {
		const float *row = Bt + d * CX_NBUS;
		__m512 acc = _mm512_mul_ps(_mm512_load_ps(A + CX_A_ALPHA), _mm512_load_ps(row));
		for (int v = 1; v < 8; v++) acc = _mm512_fmadd_ps(_mm512_load_ps(A + CX_A_ALPHA + 16 * v), _mm512_load_ps(row + 16 * v), acc);
		r = cx_blend(r, (__mmask16)(1u << d), cx_hsum(acc));
	}

	/* x = RMSNorm([p ; r]) * gain, stored as the tail of [o ; x] */
	__m512 xin[5];
	for (int v = 0; v < 4; v++) xin[v] = _mm512_load_ps(A + CX_A_P + 16 * v);
	xin[4] = r;
	_mm512_store_ps(A + CX_A_R, r);
	__m512 ss = _mm512_mul_ps(xin[0], xin[0]);
	for (int v = 1; v < 5; v++) ss = _mm512_fmadd_ps(xin[v], xin[v], ss);
	ss = cx_hsum(ss);
	const __m512 xs = cx_rsqrt(_mm512_fmadd_ps(ss, _mm512_set1_ps(1.0f / CX_DX), _mm512_set1_ps(CX_EPS)));
	_mm512_store_ps(A + CX_A_XS, xs);
	float *const x = A + CX_A_MIN + CX_DS;
	for (int v = 0; v < 5; v++) _mm512_store_ps(x + 16 * v, _mm512_mul_ps(_mm512_mul_ps(xin[v], xs), _mm512_load_ps(P + CX_O_GAIN + 16 * v)));

	/* y = x^T W: q, k, v, a_pre, b_pre */
	float *const y = A + CX_A_Y;
	cx_gemv13(P + CX_O_W, x, CX_DX, y);

	/* k = k / |k| */
	__m512 kk = _mm512_load_ps(y + 64), kn = _mm512_mul_ps(kk, kk);
	for (int v = 1; v < 4; v++) { kk = _mm512_load_ps(y + 64 + 16 * v); kn = _mm512_fmadd_ps(kk, kk, kn); }
	const __m512 ks = cx_rsqrt(_mm512_add_ps(cx_hsum(kn), _mm512_set1_ps(CX_EPS)));
	_mm512_store_ps(A + CX_A_KS, ks);
	for (int v = 0; v < 4; v++) _mm512_store_ps(A + CX_A_K + 16 * v, _mm512_mul_ps(_mm512_load_ps(y + 64 + 16 * v), ks));

	const __m512 a = cx_sigmoid(CX_BC(y + 192)), b = cx_sigmoid(CX_BC(y + 193));
	_mm512_store_ps(A + CX_A_AV, a);
	_mm512_store_ps(A + CX_A_BV, b);

	/* pass 1 over T: Sk = S k and Sq = S q together */
	__m512 Sk[4], Sq[4];
	for (int v = 0; v < 4; v++) Sk[v] = Sq[v] = zero;
	for (int j = 0; j < CX_DS; j++) {
		const float *row = T + j * CX_DS;
		if (CX_PF) for (int v = 0; v < 4; v++) cx_pf0(row + 2 * CX_DS + 16 * v);
		const __m512 kj = CX_BC(A + CX_A_K + j), qj = CX_BC(y + j);
		for (int v = 0; v < 4; v++) {
			const __m512 t = _mm512_load_ps(row + 16 * v);
			Sk[v] = _mm512_fmadd_ps(kj, t, Sk[v]);
			Sq[v] = _mm512_fmadd_ps(qj, t, Sq[v]);
		}
	}
	/* d = b v - (a b) Sk;  kq = k . q;  o = a Sq + kq d */
	const __m512 ab = _mm512_mul_ps(a, b);
	__m512 d[4], kq = zero;
	for (int v = 0; v < 4; v++) {
		d[v] = cx_fnmadd(ab, Sk[v], _mm512_mul_ps(b, _mm512_load_ps(y + 128 + 16 * v)));
		kq = _mm512_fmadd_ps(_mm512_load_ps(A + CX_A_K + 16 * v), _mm512_load_ps(y + 16 * v), kq);
	}
	kq = cx_hsum(kq);
	_mm512_store_ps(A + CX_A_KQ, kq);
	for (int v = 0; v < 4; v++) { _mm512_store_ps(A + CX_A_SK + 16 * v, Sk[v]); _mm512_store_ps(A + CX_A_SQ + 16 * v, Sq[v]); _mm512_store_ps(A + CX_A_D + 16 * v, d[v]); }
	for (int v = 0; v < 4; v++) _mm512_store_ps(A + CX_A_MIN + 16 * v, _mm512_fmadd_ps(kq, d[v], _mm512_mul_ps(a, Sq[v])));

	/* pass 2 over T: T[j] = a T[j] + k_j d */
	for (int j = 0; j < CX_DS; j++) {
		float *row = T + j * CX_DS;
		const __m512 kj = CX_BC(A + CX_A_K + j);
		for (int v = 0; v < 4; v++) _mm512_store_ps(row + 16 * v, _mm512_fmadd_ps(a, _mm512_load_ps(row + 16 * v), _mm512_mul_ps(kj, d[v])));
	}
}

/* the second half: the MLP (from the slot, in a MoE head) and the outputs */
static inline void cx_col_mlp(cx_col *c)
{
	float *const A = c->act;
	const float *const P = c->prm;

	/* h = [o ; x]^T W1 + b1; z = SiLU(h) */
	float *const h = A + CX_A_H, *const z = A + CX_A_Z;
	cx_gemv_h(P + CX_O_W1, A + CX_A_MIN, CX_DM, h);
	for (int v = 0; v < CX_NVH; v++) {
		const __m512 hv = _mm512_add_ps(_mm512_load_ps(h + 16 * v), _mm512_load_ps(P + CX_O_B1 + 16 * v));
		const __m512 sg = cx_sigmoid(hv);
		_mm512_store_ps(h + 16 * v, hv);   /* h with its bias */
		_mm512_store_ps(A + CX_A_SG + 16 * v, sg);
		_mm512_store_ps(z + 16 * v, _mm512_mul_ps(hv, sg));
	}

	/* u = z^T Wu; g = sigmoid(z . wg + bg); bc = z^T Wb */
	cx_gemv8(P + CX_O_WU, z, CX_DH, A + CX_A_U);
	__m512 gl = _mm512_load_ps(P + CX_O_BG);
	for (int v = 0; v < CX_NVH; v++) gl = _mm512_fmadd_ps(_mm512_load_ps(z + 16 * v), _mm512_load_ps(P + CX_O_WG + 16 * v), gl);
	_mm512_store_ps(A + CX_A_G, cx_sigmoid(cx_hsum(gl)));
	cx_gemv1(P + CX_O_WB, z, CX_DH, A + CX_A_BC);
}

/* the dense column's step: both halves, the MLP from the parameter block */
static inline void cx_col_step(cx_col *c, const float *e, const float *Bt)
{
	cx_col_pre(c, e, Bt);
	cx_col_mlp(c);
}

#endif
