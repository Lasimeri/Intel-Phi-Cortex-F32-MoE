/*
 * init-ckpt: a fresh initialisation of the Cortex, written as a checkpoint the daemon loads (host, C), for a run
 * from scratch: `init-ckpt OUT.bin`. The weights are what a fresh batch-trainer run starts from (lib/cx-col.h):
 * cx_col_init for each of the 120 columns, the consensus gains cg = 1, cx_emb_init for the 4,096 x 128 embedding,
 * the embedding laid out in the card's ETB blocks of 16 vocabulary rows (as cortexd saves and loads it). Nothing
 * trained goes in.
 */
#include <stdio.h>
#include <stdlib.h>
#include "../lib/cx-col.h"
#include "../lib/cx-proto.h"

int main(int argc, char **argv)
{
	if (argc != 2) { fprintf(stderr, "usage: init-ckpt OUT.bin\n"); return 2; }
	float *buf = calloc(CX_STAGE_FLOATS, sizeof *buf), *E = malloc((size_t)CX_VOCAB * CX_DE * sizeof *E);
	if (!buf || !E) { perror("malloc"); return 1; }
	float *p = buf;
	for (int k = 0; k < CX_NCOL; k++) cx_col_init(p + (long)k * CX_NPARAM, k);
	p += (long)CX_NCOL * CX_NPARAM;
	for (int i = 0; i < CX_DU; i++) p[i] = 1.0f;
	p += CX_DU;
	cx_emb_init(E);
	for (int v = 0; v < CX_VOCAB; v++)
		for (int c = 0; c < CX_DE; c++) p[(((long)v >> 4) * CX_DE + c) * 16 + (v & 15)] = E[(long)v * CX_DE + c];
	FILE *f = fopen(argv[1], "wb");
	if (!f || fwrite(buf, sizeof *buf, CX_STAGE_FLOATS, f) != CX_STAGE_FLOATS) { perror(argv[1]); return 1; }
	fclose(f);
	printf("init-ckpt: %s, %ld bytes: %d columns x %d parameters (cx_col_init), cg = 1, embedding %d x %d (cx_emb_init), nothing trained\n",
	       argv[1], (long)CX_STAGE_FLOATS * 4, CX_NCOL, CX_NPARAM, CX_VOCAB, CX_DE);
	return 0;
}
