/*
 * reward-soft: the soft records of the reward set (docs/REASON.md "The reward"; 2026-10-28 05:27, the user: "The student
 * model needs a reward function for when it gets something right"). The reward set holds conversations cut at a
 * person's line and closed by the student's own reply, a reply the teacher graded right (tools/teacher-correct.sh
 * --rewards); the student is reinforced on that reply and on nothing else: every record gets the loss weight 0 except
 * those predicting the tokens of the conversation's last speaker's line (after its <|assistant|>, through the <|end|>
 * that closes it), which get the conversation's reward times --scale. The card scales each step's gradient by its
 * record's weight (lib/cx-proto.h cx_soft w), so a reward of 1 trains the reply as the teacher's text trains, a reward
 * of 0 not at all; the context, the teacher's corpus and the persona line, adds nothing a second time.
 *   reward-soft TOKENS.u32 REWARDS.txt OUT.soft [--scale S]
 * TOKENS.u32: the reward set as the driver encodes every stream (tools/chat-text --no-think-block --detok, then the
 * instance's BPE), little-endian u32 ids (lib/cx-proto.h cx_tok); conversations end at <|endoftext|> (4095).
 * REWARDS.txt: one reward a line, one per conversation in order. OUT.soft: one 68-byte cx_soft record per token, every
 * id CX_SOFT_NONE (no distribution), the weight as above. Prints the conversations, the reinforced tokens and the
 * records; exit 1 when the counts disagree or a conversation holds no speaker's line.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include "../lib/cx-proto.h"

#define T_END 4091
#define T_ASSISTANT 4092
#define T_EOT 4095

int main(int argc, char **argv)
{
	if (argc < 4) { fprintf(stderr, "usage: reward-soft TOKENS.u32 REWARDS.txt OUT.soft [--scale S]\n"); return 2; }
	double scale = 1.0;
	for (int i = 4; i < argc; i++) {
		if (!strcmp(argv[i], "--scale") && i + 1 < argc) scale = atof(argv[++i]);
		else { fprintf(stderr, "reward-soft: unknown option %s\n", argv[i]); return 2; }
	}
	struct stat st;
	FILE *f = fopen(argv[1], "rb");
	if (!f || fstat(fileno(f), &st)) { perror(argv[1]); return 2; }
	const long n = st.st_size / (long)CX_TOK_BYTES;
	cx_tok *tok = malloc((size_t)(n + 1) * sizeof *tok);
	if (n > 0 && fread(tok, sizeof *tok, (size_t)n, f) != (size_t)n) { perror(argv[1]); return 2; }
	fclose(f);
	f = fopen(argv[2], "r");
	if (!f) { perror(argv[2]); return 2; }
	double *rw = malloc(sizeof *rw * 65536);
	int nr = 0;
	while (nr < 65536 && fscanf(f, "%lf", &rw[nr]) == 1) nr++;
	fclose(f);
	cx_soft *rec = malloc((size_t)(n > 0 ? n : 1) * sizeof *rec);
	for (long i = 0; i < n; i++) {
		for (int k = 0; k < CX_SOFT_K; k++) { rec[i].id[k] = CX_SOFT_NONE; rec[i].p[k] = 0; }
		rec[i].w = 0.0f;
	}
	/* per conversation [a, e]: e the index of its <|endoftext|>; the last <|assistant|> before e opens the reinforced line */
	long a = 0, reinforced = 0;
	int c = 0;
	for (long e = 0; e < n; e++) {
		if (tok[e] != T_EOT) continue;
		if (c >= nr) { fprintf(stderr, "reward-soft: more conversations than rewards (%d)\n", nr); return 1; }
		long s = -1;
		for (long i = e; i >= a; i--) if (tok[i] == T_ASSISTANT) { s = i; break; }
		if (s < 0) { fprintf(stderr, "reward-soft: conversation %d holds no speaker's line\n", c); return 1; }
		/* the tokens after <|assistant|> through its closing <|end|>: record i predicts token i + 1 */
		for (long i = s; i + 1 < e; i++) {
			rec[i].w = (float)(rw[c] * scale);
			reinforced++;
			if (tok[i + 1] == T_END) break;
		}
		c++;
		a = e + 1;
	}
	if (c != nr) { fprintf(stderr, "reward-soft: %d conversations, %d rewards\n", c, nr); return 1; }
	FILE *o = fopen(argv[3], "wb");
	if (!o || (n > 0 && fwrite(rec, sizeof *rec, (size_t)n, o) != (size_t)n) || fclose(o)) { perror(argv[3]); return 2; }
	printf("reward-soft: %d conversations, %ld reinforced tokens of %ld, %ld records in %s (scale %g)\n", c, reinforced, n, n, argv[3], scale);
	return 0;
}
