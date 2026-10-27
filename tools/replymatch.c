/*
 * replymatch: the model's reply against the reference answers: an exact match after whitespace is squeezed to single
 * spaces and the ends trimmed (case kept), and the best word-level LCS ratio (the longest common subsequence of
 * words over the longer of the two) as a measure of progress short of a match.
 *   replymatch REPLY.txt REF.txt [MIN_LCS]
 * REF.txt holds one reference per line (several accepted answers). Prints "exact E lcs X.XXX ref K words N" for the
 * best reference and exits 0 on an exact match, or when MIN_LCS is given and the best ratio reaches it; 1 otherwise.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAXW 512

static char *slurp(const char *path)
{
	FILE *f = fopen(path, "rb");
	if (!f) { perror(path); exit(2); }
	fseek(f, 0, SEEK_END);
	const long n = ftell(f);
	fseek(f, 0, SEEK_SET);
	char *s = malloc(n + 1);
	if (fread(s, 1, n, f) != (size_t)n) { perror(path); exit(2); }
	s[n] = 0;
	fclose(f);
	return s;
}

/* whitespace squeezed to one space, the ends trimmed, in place */
static void squeeze(char *s)
{
	long w = 0;
	int space = 1;
	for (long r = 0; s[r]; r++) {
		const int ws = s[r] == ' ' || s[r] == '\t' || s[r] == '\n' || s[r] == '\r';
		if (ws) { if (!space) s[w++] = ' '; space = 1; }
		else { s[w++] = s[r]; space = 0; }
	}
	while (w > 0 && s[w - 1] == ' ') w--;
	s[w] = 0;
}

static int words(char *s, char **w)
{
	int n = 0;
	for (char *t = strtok(s, " "); t && n < MAXW; t = strtok(NULL, " ")) w[n++] = t;
	return n;
}

static double lcs_ratio(char **a, int na, char **b, int nb)
{
	static int dp[MAXW + 1][MAXW + 1];
	for (int i = 0; i <= na; i++) for (int j = 0; j <= nb; j++) dp[i][j] = 0;
	for (int i = 1; i <= na; i++)
		for (int j = 1; j <= nb; j++)
			dp[i][j] = !strcmp(a[i - 1], b[j - 1]) ? dp[i - 1][j - 1] + 1 : dp[i - 1][j] > dp[i][j - 1] ? dp[i - 1][j] : dp[i][j - 1];
	const int m = na > nb ? na : nb;
	return m ? (double)dp[na][nb] / m : 1.0;
}

int main(int argc, char **argv)
{
	if (argc < 3) { fprintf(stderr, "usage: replymatch REPLY.txt REF.txt [MIN_LCS]\n"); return 2; }
	const double min_lcs = argc > 3 ? atof(argv[3]) : 2.0;
	char *reply = slurp(argv[1]), *refs = slurp(argv[2]);
	squeeze(reply);
	char *rcopy = strdup(reply), *rw[MAXW];
	const int nr = words(rcopy, rw);
	int best_exact = 0, best_k = -1, best_words = 0;
	double best = -1;
	int k = 0;
	for (char *line = strtok(refs, "\n"); line; line = strtok(NULL, "\n"), k++) {
		char *ref = strdup(line);
		squeeze(ref);
		if (!*ref) { free(ref); continue; }
		const int exact = !strcmp(reply, ref);
		char *fcopy = strdup(ref), *fw[MAXW];
		const int nf = words(fcopy, fw);
		const double ratio = exact ? 1.0 : lcs_ratio(rw, nr, fw, nf);
		if (exact > best_exact || (exact == best_exact && ratio > best)) { best_exact = exact; best = ratio; best_k = k; best_words = nf; }
		free(fcopy);
		free(ref);
	}
	if (best_k < 0) { fprintf(stderr, "replymatch: %s holds no reference\n", argv[2]); return 2; }
	printf("exact %d lcs %.3f ref %d words %d\n", best_exact, best, best_k, best_words);
	return best_exact || best >= min_lcs ? 0 : 1;
}
