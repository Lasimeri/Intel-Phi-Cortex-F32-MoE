# The host orchestrator

User direction (2026-10-26 06:22): use the host CPU's FMA instructions for an orchestration model that keeps the
Phi's generated tokens coherent and lingually correct. Nothing in it is trained on model output: the count model
comes from the human token stream, the word list from the human text, and the repetition constraint reads only the
sample being produced.

## Pieces

| piece | file | runs on |
|---|---|---|
| count model builder: unigram counts, hashed bigram and trigram context buckets (2^22 each) with the top-8 next tokens | tools/ngcount.c | host, once |
| the orchestrator: log-softmax, Dirichlet backoff, log-linear mixing, candidate window, word filter, 4-gram repeat block | host/orchestra.h | host, per token |
| the client options `--orchestra NGC --lambda L --words LIST --no-filter --repeat-ok` | host/cortex.c | host |
| the comparison script | tools/orch-eval.sh | host |

## Per token

1. `log p_phi = logits - logsumexp(logits)` over the 4096 entries: AVX2 max, `exp` by a degree-6 polynomial in fused
   multiply-adds (`orch_exp8`), AVX2 sum.
2. `log p_ng(v | p2, p1)`: trigram bucket of (p2, p1) when that context was counted, else bigram bucket of p1, else
   unigram, each step Dirichlet-smoothed towards the lower order with prior mass 8. Entries outside the two top-8
   lists share one closed form (a vector add of one shift to the log unigram); the listed entries are computed exactly.
3. `mixed = (1 - lambda) log p_phi + lambda log p_ng`, one `fmadd` per lane.
4. Candidates: entries within 10 nats of the best mixed value (one AVX2 compare pass, at most 256). For each:
   a word that would end outside the human word list, or leave every listed prefix, loses 7 nats; a token that
   completes a 4-gram already present in this sample loses 5 nats.

The sampler (temperature, top-k, top-p, min-p, repetition penalty) then runs on the adjusted values.

## Measured (results/orchestra-2.log; F32 checkpoint data/s4/run4off, 1M TinyStories tokens; 512 tokens x 3 seeds)

Count model: `ngcount` over the first 10,000,000 human training tokens (data/stories-10m.ngc, 469,778,456 B; 3,856
bigram and 309,084 trigram contexts). Scores by the coherence tool of the F64 tree against the human word list.
Human bands from TinyStories 256-word windows: hit5 1.000, big 0.995 (p10 0.986), rep4 0.022 (p90 0.036).

| configuration | hit5 | big | rep4 | host us/token |
|---|---|---|---|---|
| plain sampler | 0.964, 0.987, 0.988 | 0.974, 0.995, 0.982 | 0.043, 0.043, 0.046 | 328, 330, 326 |
| lambda 0.15, filter, repeat block | 1.000, 1.000, 1.000 | 1.000, 0.994, 0.997 | 0.022, 0.013, 0.027 | 342, 342, 347 |
| lambda 0.30, filter, repeat block | 1.000, 1.000, 1.000 | 0.995, 0.997, 1.000 | 0.040, 0.020, 0.020 | 337, 335, 340 |
| lambda 0.30, filter, no repeat block | 1.000, 1.000, 1.000 | 1.000, 1.000, 1.000 | 0.093, 0.103, 0.120 | 345, 332, 339 |
| lambda 0.15, no filter, repeat block | 1.000, 1.000, 1.000 | 0.997, 0.994, 0.997 | 0.022, 0.013, 0.027 | 326, 334, 340 |
| lambda 0, filter, repeat block | 0.986, 0.986, 0.986 | 0.975, 0.970, 0.984 | 0.030, 0.032, 0.051 | 344, 345, 358 |

Reading: the n-gram mixture is what brings every word and bigram into the human vocabulary (hit5 and big to
1.000); the repeat block is what removes the loops the mixture otherwise causes (rep4 0.09 to 0.12 without it);
the word filter changes little once the mixture is on (identical scores on two of three seeds). Lambda 0.15 with
both constraints puts all three measures inside the human bands. Sentence length falls from 10.3 to 10.5 words to
8.5 to 9.1. These measures are lexical; they do not score meaning, and the samples still switch subjects between
sentences (results/orchestra/ng15-s1.txt).

The first version (results/orchestra-1.log) computed `log` on all 4096 entries per step and a scalar top-64 scan:
459 us per token on the host against 324 plain; the closed-form backoff and the compare-pass candidate window
brought the cost to 342.

## Limits

- A word already committed as a valid prefix of a listed word ("thes" of "these") cannot be repaired: every piece
  that ends it is penalised alike, so the relative distribution is unchanged. Blocking needs the continuation
  pieces in the candidate window.
- The count model is TinyStories only; a second dataset (BabyLM, see PLAN-MOE.md) widens it.
- The constraint strengths (7 and 5 nats, window 10 nats, prior 8) were set once, not searched.
