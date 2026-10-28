/*
 * answer-bits: how much the student's answers depend on what the person said (docs/REASON.md, the decision metric from
 * 2026-10-28 05:xx: the 20-question probe at 1 to 3 correct cannot resolve a gain). Two token streams hold the same
 * conversations, A as written and B with every person's line replaced by the line of the same turn in another
 * conversation (tools/answer-bits.sh builds both and reads them with tools/ref-loss --dump). The speaker's lines are
 * the same tokens in both, so their losses pair one to one; the mean loss of the speaker's tokens in B minus in A is
 * the information, in bits per answer token, that the true question gives the model about its answer. A model that
 * reads only the last two tokens (an n-gram) scores about 0; a model that answers the question scores well above 0.
 *   answer-bits A.u16 A.f32 B.u16 B.f32 [--first K]
 * A.f32, B.f32: ref-loss --dump of the streams (F32 nats, the loss of the prediction of token s + 1 at index s). A
 * token counts when it lies inside the speaker's line: after <|assistant|> (4092) and before the <|end|> (4091) that
 * closes it, the <|end|> itself included (where the answer stops is part of the answer). --first K (default 5) also
 * reports the first K tokens of every answer apart, where the question's influence is strongest. Prints the token
 * counts, the mean loss of A and B over the answers in bits per token, the difference with its standard error over
 * the answers (each answer's mean difference is one sample), the same for the first K tokens, and the share of
 * answers whose loss falls with the true question. Exit 1 when the two streams' answer tokens differ.
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#define T_ASSISTANT 4092
#define T_END 4091

typedef struct { uint16_t *tok; float *nll; long n; } stream;

static int load(stream *s, const char *tp, const char *lp)
{
	struct stat st;
	FILE *f = fopen(tp, "rb");
	if (!f || fstat(fileno(f), &st)) { perror(tp); return -1; }
	s->n = st.st_size / 2;
	s->tok = malloc((size_t)s->n * 2);
	if (fread(s->tok, 2, (size_t)s->n, f) != (size_t)s->n) { perror(tp); return -1; }
	fclose(f);
	f = fopen(lp, "rb");
	if (!f || fstat(fileno(f), &st)) { perror(lp); return -1; }
	if (st.st_size / 4 != s->n - 1) { fprintf(stderr, "answer-bits: %s holds %ld losses for %ld tokens (want %ld)\n", lp, (long)(st.st_size / 4), s->n, s->n - 1); return -1; }
	s->nll = malloc((size_t)st.st_size);
	if (fread(s->nll, 4, (size_t)(s->n - 1), f) != (size_t)(s->n - 1)) { perror(lp); return -1; }
	fclose(f);
	return 0;
}

/* the answers of a stream: for every answer token (in order) its id, its loss (of its prediction) and its answer's
 * index and position in the answer */
typedef struct { uint16_t id; float nll; int ans, pos; } atok;

static long answers(const stream *s, atok *out, int *nans)
{
	long m = 0;
	int a = -1, in = 0, pos = 0;
	for (long i = 0; i + 1 < s->n; i++) {
		const uint16_t t = s->tok[i], next = s->tok[i + 1];
		if (t == T_ASSISTANT) { in = 1; a++; pos = 0; }
		if (in) {
			out[m].id = next;
			out[m].nll = s->nll[i];
			out[m].ans = a;
			out[m].pos = pos++;
			m++;
			if (next == T_END) in = 0;
		}
	}
	*nans = a + 1;
	return m;
}

int main(int argc, char **argv)
{
	if (argc < 5) { fprintf(stderr, "usage: answer-bits A.u16 A.f32 B.u16 B.f32 [--first K]\n"); return 2; }
	int first = 5;
	for (int i = 5; i < argc; i++) {
		if (!strcmp(argv[i], "--first") && i + 1 < argc) first = atoi(argv[++i]);
		else { fprintf(stderr, "answer-bits: unknown option %s\n", argv[i]); return 2; }
	}
	stream A, B;
	if (load(&A, argv[1], argv[2]) || load(&B, argv[3], argv[4])) return 2;
	atok *a = malloc((size_t)A.n * sizeof *a), *b = malloc((size_t)B.n * sizeof *b);
	int na, nb;
	const long ma = answers(&A, a, &na), mb = answers(&B, b, &nb);
	if (ma != mb || na != nb) { fprintf(stderr, "answer-bits: A has %ld answer tokens in %d answers, B %ld in %d: not the same answers\n", ma, na, mb, nb); return 1; }
	for (long i = 0; i < ma; i++) if (a[i].id != b[i].id) { fprintf(stderr, "answer-bits: the answer tokens differ at %ld (answer %d): %u against %u\n", i, a[i].ans, a[i].id, b[i].id); return 1; }
	const double L2 = log(2.0);
	double sa = 0, sb = 0, fa = 0, fb = 0;
	long nf = 0;
	double *da = calloc((size_t)na, sizeof *da), *df = calloc((size_t)na, sizeof *df);
	int *ca = calloc((size_t)na, sizeof *ca), *cf = calloc((size_t)na, sizeof *cf);
	for (long i = 0; i < ma; i++) {
		sa += a[i].nll; sb += b[i].nll;
		da[a[i].ans] += (b[i].nll - a[i].nll) / L2; ca[a[i].ans]++;
		if (a[i].pos < first) { fa += a[i].nll; fb += b[i].nll; nf++; df[a[i].ans] += (b[i].nll - a[i].nll) / L2; cf[a[i].ans]++; }
	}
	/* the per-answer means of the difference: their mean and its standard error */
	double m1 = 0, q1 = 0, m2 = 0, q2 = 0;
	int k1 = 0, k2 = 0, better = 0;
	for (int j = 0; j < na; j++) {
		if (ca[j]) { const double d = da[j] / ca[j]; m1 += d; q1 += d * d; k1++; better += d > 0; }
		if (cf[j]) { const double d = df[j] / cf[j]; m2 += d; q2 += d * d; k2++; }
	}
	m1 /= k1 ? k1 : 1; m2 /= k2 ? k2 : 1;
	const double se1 = k1 > 1 ? sqrt((q1 / k1 - m1 * m1) / (k1 - 1)) : 0, se2 = k2 > 1 ? sqrt((q2 / k2 - m2 * m2) / (k2 - 1)) : 0;
	printf("answer-bits: %d answers, %ld answer tokens (%ld in the first %d of each)\n", na, ma, nf, first);
	printf("answer-bits: all answer tokens: true question %.4f, swapped %.4f bits/token: the question gives %.4f bits/token (per-answer mean %.4f +- %.4f)\n", sa / ma / L2, sb / ma / L2, (sb - sa) / ma / L2, m1, se1);
	printf("answer-bits: the first %d tokens: true question %.4f, swapped %.4f bits/token: the question gives %.4f bits/token (per-answer mean %.4f +- %.4f)\n", first, fa / (nf ? nf : 1) / L2, fb / (nf ? nf : 1) / L2, (fb - fa) / (nf ? nf : 1) / L2, m2, se2);
	printf("answer-bits: answers easier with the true question: %d of %d (%.1f %%)\n", better, k1, k1 ? 100.0 * better / k1 : 0);
	return 0;
}
