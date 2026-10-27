/*
 * quant3: post-training 3-bit quantisation of a Cortex checkpoint (host, C), the first measurement of the 3 bits
 * per weight direction (2026-10-26: "Get the bpw size down to 3bpw in order to pack as much into the model without
 * catastrophic forgetting as possible").
 *   quant3 CKPT.bin GROUP OUT.bin
 * Every run of GROUP consecutive weights gets one scale s = max(max_positive / 3, -min_negative / 4) and each weight
 * becomes q = round(w / s) in [-4, 3] (8 levels, zero among them); the dequantised value is q * s. The checkpoint is
 * the LOAD layout (lib/cx-proto.h CX_STAGE_CKPT): 120 columns of 44,512 F32, cg (128 F32, kept exact: 128 values),
 * the embedding 4096 x 128 F32. OUT.bin is the dequantised F32 checkpoint in the same layout, for the loss
 * measurement on the card (LOAD or cluster-gate validation) against the F32 original. Printed per segment: the
 * relative RMS error (RMS of w - q*s over RMS of w), the largest absolute error, the share of weights clamped; and
 * the bits per weight of the format: 3 + 16 / GROUP with an F16 scale per group (3.25 at 64, 3.125 at 128, 3.0625
 * at 256), 3 + 32 / GROUP with an F32 scale.
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define NCOL 120
#define NPARAM 44512
#define DU 128
#define VOCAB 4096
#define DE 128
#define NFLOAT ((long)NCOL * NPARAM + DU + (long)VOCAB * DE)

typedef struct { double se, ss, maxerr; long clamped, n; } seg;

static void quantise(float *w, long n, int group, seg *s)
{
	for (long g = 0; g < n; g += group) {
		const long m = g + group <= n ? group : n - g;
		float maxpos = 0, minneg = 0;
		for (long i = 0; i < m; i++) { if (w[g + i] > maxpos) maxpos = w[g + i]; if (w[g + i] < minneg) minneg = w[g + i]; }
		const float scale = fmaxf(maxpos / 3.0f, -minneg / 4.0f);
		for (long i = 0; i < m; i++) {
			const float x = w[g + i];
			float d = 0;
			if (scale > 0) {
				int q = (int)lrintf(x / scale);
				if (q < -4) { q = -4; s->clamped++; }
				if (q > 3) { q = 3; s->clamped++; }
				d = (float)q * scale;
			}
			const double e = (double)x - d;
			s->se += e * e;
			s->ss += (double)x * x;
			if (fabs(e) > s->maxerr) s->maxerr = fabs(e);
			s->n++;
			w[g + i] = d;
		}
	}
}

static void report(const char *name, const seg *s)
{
	printf("%-10s %9ld weights: relative RMS error %.4f, largest error %.5f, RMS %.5f, clamped %ld\n", name, s->n,
	       s->ss > 0 ? sqrt(s->se / s->ss) : 0.0, s->maxerr, sqrt(s->ss / (double)s->n), s->clamped);
}

int main(int argc, char **argv)
{
	if (argc < 4) { fprintf(stderr, "usage: quant3 CKPT.bin GROUP OUT.bin\n"); return 2; }
	const int group = atoi(argv[2]);
	if (group < 2) { fprintf(stderr, "quant3: GROUP must be at least 2\n"); return 2; }
	float *w = malloc(NFLOAT * sizeof *w);
	FILE *f = fopen(argv[1], "rb");
	if (!f || fread(w, sizeof *w, NFLOAT, f) != (size_t)NFLOAT) { fprintf(stderr, "quant3: %s is not a %ld-float checkpoint\n", argv[1], NFLOAT); return 1; }
	fclose(f);
	seg cols = { 0 }, emb = { 0 };
	for (int c = 0; c < NCOL; c++) quantise(w + (long)c * NPARAM, NPARAM, group, &cols);
	quantise(w + (long)NCOL * NPARAM + DU, (long)VOCAB * DE, group, &emb);
	printf("quant3: %s, groups of %d: %.4f bits per weight with an F16 scale (%.4f with F32); cg's 128 values kept in F32\n",
	       argv[1], group, 3.0 + 16.0 / group, 3.0 + 32.0 / group);
	report("columns", &cols);
	report("embedding", &emb);
	f = fopen(argv[3], "wb");
	if (!f || fwrite(w, sizeof *w, NFLOAT, f) != (size_t)NFLOAT) { perror(argv[3]); return 1; }
	fclose(f);
	printf("quant3: dequantised checkpoint written to %s\n", argv[3]);
	return 0;
}
