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
| the comparison script (identity gate, six configurations, three seeds) | tools/orch-eval.sh | host |

## Per token

1. `log p_phi = logits - logsumexp(logits)` over the 4096 entries: AVX2 max, `exp` by a degree-6 polynomial in fused
   multiply-adds (`orch_exp8`), AVX2 sum.
2. `log p_ng(v | p2, p1)`: trigram bucket of (p2, p1) when that context was counted, else bigram bucket of p1, else
   unigram, each step Dirichlet-smoothed towards the lower order with prior mass 8. Entries outside the two top-8
   lists share one closed form (a vector add of one shift to the log unigram); the listed entries are computed exactly.
3. `mixed = (1 - lambda) log p_phi + lambda log p_ng + logsumexp(logits)`, one `fmadd` and one add per lane. The
   log-sum-exp is added back so the result keeps the offset of the card's logits: the sampler's repetition penalty
   divides positive values and multiplies negative ones, so a transform that moved everything below zero changed the
   sampler's behaviour (see the artefact below). At lambda 0 with both constraints off the logits are unchanged, and
   tools/orch-eval.sh checks that the generated text is byte-identical to the plain sampler's.
4. Candidates: entries within 10 nats of the best mixed value (one AVX2 compare pass, at most 256). For each:
   a piece that would end the word on a string outside the human word list, or leave every listed prefix, loses 7
   nats; a piece that completes a token 4-gram already present in this sample loses 5 nats.

The sampler (temperature 0.5, top-k 75, top-p 0.95, min-p 0.05, repetition penalty 1.2) then runs on the result.

## Measured

512 tokens from "Once upon a time", seeds 1 to 3, scored by the coherence tool of the F64 tree: hit5 = share of
words of 5+ letters that are in the human word list, big = share of adjacent word pairs seen in the human text,
rep4 = share of word 4-grams repeated within a 256-word window. Human bands from TinyStories windows: hit5 1.000,
big 0.995 (p10 0.986), rep4 0.022 (p90 0.036). The three values per cell are the three seeds.

TinyStories model (data/s4/run4off, 1M tokens), count model data/stories-10m.ngc (10M human tokens), word list of
the F64 tree; results/orchestra-3.log:

| configuration | hit5 | big | rep4 |
|---|---|---|---|
| plain sampler | 0.964, 0.987, 0.988 | 0.974, 0.995, 0.982 | 0.043, 0.043, 0.046 |
| lambda 0 without constraints (identity gate) | identical text to plain on all three seeds | | |
| lambda 0.15, filter, repeat block | 1.000, 1.000, 1.000 | 0.997, 0.992, 1.000 | 0.024, 0.014, 0.012 |
| lambda 0.30, filter, repeat block | 1.000, 1.000, 1.000 | 1.000, 0.992, 0.997 | 0.007, 0.015, 0.012 |
| lambda 0.30, filter, no repeat block | 1.000, 1.000, 1.000 | 1.000, 0.997, 1.000 | 0.022, 0.057, 0.035 |
| lambda 0.15, no filter, repeat block | 1.000, 1.000, 1.000 | 0.995, 0.992, 1.000 | 0.017, 0.014, 0.012 |
| lambda 0, filter, repeat block | 1.000, 0.987, 1.000 | 0.979, 0.968, 0.985 | 0.014, 0.010, 0.009 |

Mixed model (data/mix/a-ckpt.bin: run4off plus 320,000 tokens of the TinyStories + BabyLM mix), count model
data/mix/mix-50m.ngc, word list data/wordlist-mix.txt (60 MB TinyStories + 6 x 12 MB BabyLM samples, count >= 3);
results/orchestra-mix.log:

| configuration | hit5 | big | rep4 |
|---|---|---|---|
| plain sampler | 0.850, 0.831, 0.673 | 0.910, 0.862, 0.831 | 0.009, 0.027, 0.009 |
| lambda 0 without constraints (identity gate) | identical text to plain on all three seeds | | |
| lambda 0.15, filter, repeat block | 0.968, 0.982, 1.000 | 0.969, 0.975, 0.972 | 0.005, 0.027, 0.034 |
| lambda 0.30, filter, repeat block | 1.000, 0.973, 1.000 | 0.978, 0.969, 0.977 | 0.021, 0.021, 0.026 |
| lambda 0.30, filter, no repeat block | 1.000, 1.000, 1.000 | 0.992, 0.995, 0.978 | 0.030, 0.051, 0.043 |
| lambda 0.15, no filter, repeat block | 0.911, 0.983, 0.973 | 0.954, 0.984, 0.969 | 0.014, 0.023, 0.031 |
| lambda 0, filter, repeat block | 0.742, 0.701, 0.742 | 0.814, 0.795, 0.823 | 0.015, 0.009, 0.015 |

Reading:
- The n-gram mixture carries the gain: every word and most word pairs move into the human vocabulary on both
  models (mixed model hit5 from 0.67 to 0.85 up to 0.97 to 1.00, big from 0.83 to 0.91 up to 0.97 to 0.98).
- The repeat block halves rep4 where the mixture raises it (0.022 to 0.057 without, 0.007 to 0.026 with).
- The word filter helps only together with the mixture (mixed model, lambda 0.15: hit5 +0.057, -0.001, +0.027
  against no filter). Alone it lowers hit5 on the mixed model (0.70 to 0.74 against 0.67 to 0.85 plain): a model
  that spells many words from pieces is pushed along prefixes of listed words and ends on strings outside the list.
- Lambda 0.30 with both constraints is the default. Host cost: 328 us per token plain, 335 to 350 with the
  orchestrator (results/orchestra-2.log, orchestra-mix.log).
- These measures are lexical. They do not score meaning, and the samples still change subject between sentences
  (results/orchestra/ng30-s1.txt).

Artefact found on the way (results/orchestra-2.log): the second version returned log-probabilities without the
log-sum-exp offset; every value was negative, the sampler's repetition penalty then multiplied every repeated
token's value by 1.2 instead of dividing the positive ones, and the samples looped (rep4 0.09 to 0.12 without the
block) and shortened their sentences (8.5 to 9.1 words against 10.3 to 10.5). The identity gate catches this class
of error. The first version computed `log` on all 4096 entries per step and a scalar top-64 scan: 459 us per token
on the host; the closed-form backoff and the compare-pass candidate window brought it to 342.

## Limits

- A word already committed as a valid prefix of a listed word ("thes" of "these") cannot be repaired: every piece
  that ends it is penalised alike, so the relative distribution is unchanged.
- The constraint strengths (7 and 5 nats, window 10 nats, prior 8) were set once, not searched.
- The count model and the word list are per corpus; a model trained on the mix is served with the mix files.
