# Phi Cortex F32 MoE: plan

Direction (2026-10-26 06:06): take the trained F32 Cortex (120 columns on the Xeon Phi 7120A, `data/s4/run4off/ckpt.bin`,
validation 6.18 bits/token after 1M TinyStories tokens) and turn it into a mixture-of-experts model whose dense part
of 3 to 4M F32 parameters stays resident in the cores' L2 ("CPU cache") while the larger part, up to 4 GB, lives in
the card's GDDR ("VRAM") and is touched a few rows per token: hashed n-gram memories and routed experts; and give the
columns a sparse attention over a memory of past steps. Every stage is gated against the double host reference and
trained against the dense baseline on the same card. This folder is a copy of the F32 project with its GitHub remote
removed; nothing here is pushed to the F32 repository.

## Measured budgets the design is built on

| quantity | value | source |
|---|---|---|
| L2 per core, cores used | 512 KiB x 60 (30 MiB); 2 columns per core | Stage 0 / Stage 1 of the F32 project |
| dense column today | 44,512 F32 = 178,048 B, plus T 16 KiB and activations; 2 per core fit (79.9 us/step); 236 KB per column thrashed (150 to 181 us) | `results/s01-col.md` |
| dense parameters in L2 today | 120 x 44,512 = 5,341,440 (+ cg 128); the embedding 524,288 is already in GDDR | `lib/cx-col.h` |
| GDDR | 15,419 MiB total; one core alone reads 3.6 to 4.7 GB/s, all cores 165 GB/s streaming; a random 512 B row is latency-bound (8 lines, the L2 prefetcher needs 3 ascending misses) | `knc-performance-facts` |
| dense training rate | 1,258 to 1,261 tok/s (fwd 323 us, bwd 470 us per step); 2,485 L2 misses per step per thread | `results/s04c-run-f64-vs-f32.log` (F32 part) |
| baseline to beat | 6.4369 bits/token after 320k tokens, lr 3e-3; 6.18 after 1M (run4off) | same, `results/s04-cluster.md` |

## The conversion

1. **Experts (the dense part shrinks to the budget).** The column's MLP (W1 144 x 64, b1, Wu 64 x 128: 17,472
   parameters) leaves L2 and becomes E experts per column in GDDR. Expert 0 is initialised with the trained W1, b1, Wu,
   so the converted model computes exactly the trained model until the router learns to use the others. The router
   (x -> E logits, 80 x E in L2) picks the top-1 expert per step; its 68 KB of weights are prefetched from GDDR while
   the fast-weight pass runs. The dense part becomes 27,040 parameters per column = 3,244,800 (plus the router), inside
   the 3 to 4M asked for. Sparse Adam touches only the selected expert. Load balance by the switch-style auxiliary
   loss. E = 16 per column: 134 MB of experts, 402 MB with the Adam moments.
2. **Hashed n-gram memory (Engram-style; arXiv 2601.07372).** Orders n = 2 and 3 over the last token ids, each a
   table of R rows x 128 in GDDR, row = hash(ids) mod R; the looked-up rows are gated and added to the token
   embedding before the columns: e' = e + sum_n g_n * T_n[h_n]. Gradients reach only the touched rows (owner-summed
   like the embedding rows today) and the gates; rows are updated with their own learning rate. R = 2^21 per order:
   1 GiB per order, 2 GiB total (+ 16 MB of per-row accumulators). The model has no n-gram component today; the
   tables start at zero, so the converted model is again exactly the trained model at step 0.
3. **Sparse attention.** Each column keeps a memory of its last 256 steps (key 64, value 64, 128 KB per column, in
   GDDR: 15 MB total) and attends to the top-4 keys by dot product, the result added to the fast-weight read-out.
   The bus attention over the 120 slots stays.

GDDR total at these sizes: about 2.5 GB (experts with moments 0.4 GB, n-gram 2 GB, attention memory 15 MB, embedding
and its shadow 4 MB), under the 4 GB asked for; the n-gram tables scale to 2^22 rows if the collision rate measured in
Stage 1 asks for it. Per-step GDDR traffic per core at top-1 routing: about 265 KB (two experts of 68 KB, two n-gram
rows, two 64 KB key scans), 16 GB/s aggregate at 1 ms per step, latency hidden by prefetch.

## Stages and gates

| stage | what | gate | measure |
|---|---|---|---|
| 0 | this plan, branch `moe`, host reference extended as each piece lands | | done |
| 1a | n-gram memory, small tables (-DCX_NG_ROWS=4096 on both sides) | card vs host double: e', loss, row and gate gradients | PASSED 10-26 (loss 9.1e-7, gradients 6.4e-7) |
| 1b | n-gram memory at 2^21 rows, training 320k tokens from `run4off` | | measured 10-26: training -0.205, validation -0.041 bits/token (mix run B); the daemon's ring gated |
| 1c | the memory trained in the live loop (the chunk layer, 10-27 01:31) | the coach's ring thread against cluster-gate + ngserve | PASSED 10-27 07:18: losses, checkpoint, gates and tables identical |
| 2 | experts in GDDR: 61,440 in one pool (revised 06:32 from E = 16 per core), product-key routing, sparse upcycling | converted model == dense model with the bonus (byte-identical), then the experts' updates and the head gradients vs the double reference | PASSED 10-27 07:1x (9.0e-7 / 5.2e-7 / 1.0e-6); 821 to 825 tok/s; validation 6.2707 at home routing vs 6.2476 dense on 2,000 chunks; the experts updated in the report line; the pool persistent and the routed model read on the host (07:41); the run since 07:42 |
| 3 | sparse attention memory, top-4 | gradients vs reference | sized, the round trip measured (36 to 39 us per step at a 256-step window); not built |
| 4 | the whole model under the coach (`docs/COACH.md` in the F64 repository), then the chat set | | the live coach runs every stage (docs/LIVE.md); the chat base measured (docs/CHAT.md) |

Checkpoints: the dense part as today (`ckpt-N.bin`, with the home experts in the MLP regions), the memory's gates
beside it (`ckpt-N-gates.f32`) and its tables on a schedule (`tables-N.f32`, 2 GiB), the pool on a schedule and at
the end (`pool-N.f32`, 4 GiB); a run restarts with or without each (`--gates`, `--ngram`, `--pool`).

## Revision (06:12, user): heads per VPU, experts in GDDR, n-grams in host RAM over DMA

"There should be an MoE head per KNC core's VPU, just like the small models we're already running per core VPU, but
they need to be reduced in size to support CPU cache and data transfer to and from the MoE heads and VRAM weights,
along with n-grams in host RAM accessed via DMA."

| part | where | size | traffic per step |
|---|---|---|---|
| MoE head, two per core (one per hardware thread used) | L2 | today's column without its MLP: 27,040 F32 = 108 KB, plus T 16 KB and activations 13 KB; the MLP becomes the routed experts | none (resident) |
| expert staging | L2 | the selected expert's weights, single-buffered: hidden 64 = 68 KB, hidden 32 = 34 KB; measured L2 misses decide the hidden size | one expert read from GDDR per head per step, prefetched at the router decision |
| experts | GDDR | E per head; hidden 64, E = 16: 1.12 MB per head, 134 MB (402 MB with Adam moments); up to 4 GB allows E = 190 | the read above plus the sparse update of the one expert touched |
| n-gram tables | host RAM | orders 2 and 3; 2^22 rows x 128 F32 = 2 GiB per order (the host has the RAM; the tables never cross PCIe whole) | per chunk of 32 steps: the host gathers 64 rows (32 KB) into a ring in the part; the card DMAs them into GDDR once per chunk (card-owned channel 2, measured 3.6 GB/s, about 10 us) and reads them from L2; the card DMAs the 64 row gradients (32 KB) back; the host applies the sparse updates on its own CPU between chunks |

The host side of the n-gram memory is a process that owns the tables and the token stream (it knows the ids ahead,
so it gathers chunk c + 1 while the card computes chunk c), and applies the gradients the card returns. The card
side is a double-buffered ring in the part, one DMA each way per chunk, and the gated addition in the heads.

Stages, revised: 1a the n-gram path with small tables, host rows and gradients over the real DMA ring, gated against
the double reference; 1b the 2 GiB tables and a 320k-token run; 2 the heads shrink (MLP out) and the experts stream
from GDDR; 3 the sparse attention memory; 4 the coach.

## Sizing (06:32, user): 4 GiB of GDDR for the inactive weights, one pool read by every head

Shapes from lib/cx-col.h with CX_HID 64 (the trained F32 model): a column holds 44,512 F32. The MLP (W1 144 x 64,
b1 64, Wu 64 x 128) is 17,472 of them; everything else (P_in, Wqb, gain, W, Wg, bg, Wb) is 27,040 and stays in the
head. The head is the per-VPU unit the user asked for: two heads per core, 60 cores, 120 heads, each in its core's L2.

| item | count | bytes | where |
|---|---|---|---|
| expert (one MLP: W1, b1, Wu) | 17,472 F32 | 69,888 B | GDDR, shared pool |
| experts in 4 GiB | 61,440 | 4,293,918,720 B (4 GiB - 1,048,576 B) | GDDR |
| expert parameters | 1,073,479,680 | | |
| head (column minus MLP) | 27,040 F32 | 108,160 B | L2 of its core |
| fast weights T per head | 4,096 F32 | 16,384 B | L2 |
| step activations per head | about 3,300 F32 | about 13 KB | L2 |
| expert staging per head (2 slots) | 2 x 17,472 F32 | 139,776 B | L2 (prefetch of the next step's expert) |
| product keys (router), shared | 2 x 248 x 40 = 19,840 F32 | 79,360 B | L2 of every core |
| per core total | | 2 x (108,160 + 16,384 + 13,000 + 139,776) + 79,360 = 634,000 B | over the 524,288 B L2 |
| resident dense parameters | 120 x 27,040 + 19,840 + E 524,288 + cg 128 = 3,789,056 | | the "3-4 M in cache" budget |

The per-core total above exceeds L2 by about 110 KB with two staging slots per head; one staging slot per head
(the expert of the current step, the next one's prefetch landing in the same slot after use) gives 494,000 B and fits.
Which is measured first: the L2 miss rate of a head with one slot versus two (pmu 0x10cb as in the cluster gate).

Routing: every head reads the same pool. Top-1 expert per head per step through product keys: x (80 wide) is split
into two 40-wide halves, each half scores 248 keys, the expert index is (best of half A) x 248 + (best of half B)
over 248 x 248 = 61,504 >= 61,440 slots (indices past 61,439 wrap to index mod 61,440). 2 x 248 x 40 FMAs per head
per step, no per-head router weights. Reads per step: 120 x 69,888 B = 8,386,560 B; at the F32 step time of 793 us
that is 10.6 GB/s of GDDR reads, under the 7120A's measured STREAM figures. Updates: an expert used by a head gets
that head's gradient; plain SGD in place (Adam moments for 1.07 G parameters would need another 8 GiB).

Context in host RAM over DMA (sparse attention, stage 3): each head's k (64) and v (64) of every step go to the
host: 512 B per head per step, 61,440 B per step for 120 heads, 77 MB/s at 1,260 steps/s. The host keeps them in
RAM and answers each head's query with the top-4 matches by dot product (AVX2 FMA), 4 x 64 F32 per head = 122,880 B
back per step, one step of lag (the match for step t uses the context up to t-1). An exact scan over a window of W
steps reads 120 x W x 256 B of keys per step: W = 1,024 is 31.5 MB and fits the 5800X's 32 MB L3 (about 0.15 ms per
step at L3 bandwidth, computed, not measured); W = 4,096 is 126 MB from DRAM (about 3 ms per step, too slow), so a
window past 1,024 needs an index (sign-bit hashing of k into 65,536 buckets, probing a few). Measured when built.

The n-gram tables keep their own host RAM (2^21 rows x 2 orders x 128 F32 = 2 GiB, or 2^22 rows = 4 GiB).

## The second dataset (07:00, user: "Get a second dataset to train alongside this one")

BabyLM 2024 text-only 100M-word training set (osf.io/ad7qg, text_data/train_100M.zip, 180,012,897 B, sha256
37a92db2...dc826; dev.zip 19,264,893 B). Chosen because it is plain text with document structure, built for small
models, and close in register to TinyStories: CHILDES child-directed speech, children's books from Project
Gutenberg, OpenSubtitles, Simple English Wikipedia, the spoken part of the British National Corpus, Switchboard
(Choshen et al., arXiv 2404.06214, table 1). Files, tools and numbers:

| source | raw bytes | lines | kind (tools/textprep.c) | documents | tokens (TinyStories BPE) | bytes/token |
|---|---|---|---|---|---|---|
| simple_wiki | 85,323,360 | 646,969 | wiki: article headings | 116,028 | 32,765,939 | 2.547 |
| gutenberg | 145,494,856 | 676,014 | book: cut at 1,500 words | 1,485 | 46,376,148 | 3.135 |
| childes | 157,062,843 | 5,790,000 | talk: speaker tags removed | 125,278 | 54,030,279 | 2.346 |
| switchboard | 6,586,033 | 161,740 | talk | 3,370 | 2,246,802 | 2.736 |
| bnc_spoken | 40,356,693 | 818,961 | lines | 20,673 | 14,053,892 | 2.891 |
| open_subtitles | 106,149,427 | 3,495,000 | lines | 84,826 | 41,411,395 | 2.591 |
| total | 540,973,212 | | | 351,660 | 190,884,455 | |

TinyStories runs at 3.961 bytes per token with this vocabulary; the BabyLM text at 2.3 to 3.1, so a third of its
words are spelled from pieces. The vocabulary is kept (the trained model's embedding is tied to it).

Mixed stream (tools/interleave.c): blocks of whole documents holding at least 2,048 tokens, alternating TinyStories
with one BabyLM source in turn, documents cut at 2,048 tokens. data/mix/train.u16: 50,000,000 tokens = TinyStories
24,089,488 (48.2 %) + simple_wiki 4,324,462 + gutenberg 3,883,246 + childes 4,211,210 + switchboard 4,447,220 (one
wrap of its 2.2M) + bnc_spoken 4,688,279 + open_subtitles 4,356,095. data/mix/valid.u16: 200,000 tokens from the
validation and dev files in the same way (TinyStories 95,519). The orchestrator's count model over the mix:
data/mix/mix-50m.ngc (4,024 bigram, 937,714 trigram contexts); the word list over both corpora: data/wordlist-mix.txt.

Runs (tools/mix-runs.sh, results/mix-{z,a,b}.log): Z the run4off model's validation on the mix before any training;
A 320,000 mixed tokens plain; B the same with the n-gram memory (2^21-row tables per order on the host, zero init,
SGD lr 0.1 on the rows, Adam on the gates). The validation pass takes part in the ring (cl_ng_begin per validation
chunk, cl_ng_done publishes the chunk), the gates are saved as ng-gates.f32 beside the checkpoint and the tables by
ngserve --save. Results are appended below when the runs end.

### Results of the three runs (07:23, results/mix-z.log, mix-a.log, mix-b.log, mix-b-ngserve.log)

| run | train loss at 320k, bits/token | validation on the mix (100,000 tokens), bits/token | bits/byte | tok/s | us/step fwd / bwd+Adam |
|---|---|---|---|---|---|
| Z: run4off before any mixed training | | 8.4101 | 2.1232 | | |
| A: plain, 320,000 mixed tokens | 7.1814 | 7.6649 | 1.9351 | 1,225 to 1,228 | 329 / 487 |
| B: n-gram memory, 320,000 mixed tokens | 6.9761 | 7.6237 | 1.9247 | 1,219 to 1,225 | 337 / 479 |

Reading: the run4off model (TinyStories only, 6.1825 bits/token on its own validation) pays 2.23 bits/token on the
mixed validation stream before training on it; 320,000 mixed tokens recover 0.745 of that. The n-gram memory lowers
the training loss by 0.205 bits/token and the validation loss by 0.041 bits/token at the same token count, at the
same speed (the ring costs 8 us of forward per step and saves 8 us of backward, within the run-to-run spread). The
gates grew from 1.0 to a mean of 4.88 (order 2) and 5.59 (order 3) (data/mix/b-ng-gates.f32): Adam on the gates
amplifies rows that plain SGD at lr 0.1 moves slowly, so the row learning rate is the first knob to search. Both
runs are deterministic (same initial weights, same stream), so the 0.041 is not sampling noise, but it is one pair
on one stream order; the spread over stream orders is not measured. The tables after B are in data/mix/b-ngtables.f32
(2,147,483,648 B), the checkpoints in data/mix/{a,b}-ckpt.bin; cortexd did not yet read the tables then, so the B
checkpoint could not generate with its memory until the daemon got the ring (12:4x below; since 10-27 06:3x a client
maps the tables file itself, `cortex --ngram`, and tells the daemon with CX_OP_NGRAM).

### The orchestrator on the mixed model (07:3x, docs/ORCHESTRA.md, results/orchestra-mix.log)

With the mix count model and word list, lambda 0.30 plus both constraints takes the A checkpoint's samples from
hit5 0.67 to 0.85, big 0.83 to 0.91 (plain) to hit5 0.97 to 1.00, big 0.97 to 0.98, rep4 0.021 to 0.026. The
lambda 0 identity check passed on both models after the log-sum-exp offset was restored in the mixture.

## The daemon's ring (12:4x, user: "Setup the ring for cortexd so the model can generate text")

A generation step's token is known only when the host issues the STEP, so the training ring's chunk protocol does
not apply. lib/cx-proto.h adds a 64-step ring of 1 KiB row slots at part + 48 MiB + 160 KiB (CX_NGS_ROWS(n), n the
steps since RESET) and the gates staged after the checkpoint (CX_STAGE_GATES), announced by LOAD's arg bit 0.
host/cortex.c `--ngram TABLES GATES` loads the 2 GiB tables into RAM (0.7 s) and writes the two rows of each step's
suffix (ids[n-2], ids[n-1], ids[n], CX_NG_PAD before the start, as ngserve gathers them) before the STEP (0.5 us);
card/cortexd.c thread 0 pulls the 1 KiB by DMA into ring slot n % 64, the GDDR layout the training chunks use (slot
n % 64 = chunk (n / 32) % 2, step n % 32), so cl_fwd_step is unchanged. Proven first (tools/dma-coh.c): a DMA write
into GDDR invalidates every core's L2 copy of the lines (0 stale words of 1,966,080 on 120 threads, demand-read or
prefetched), so the slots are reused without eviction.

Gate (tools/gen-check.c, results/moe-1-daemon-gate.log): the B checkpoint on the first 97 tokens of
data/mix/valid.u16 (`cortex --ids --dump-logits`), every step's 4096 logits against the double reference holding the
same checkpoint, tables and gates: plain rel 6.46e-7, memory rel 5.88e-7, argmax 97 of 97 in both, a repeated run
byte-identical; the memory moves the argmax of step 63 (as it must).

| per token, B checkpoint, greedy, 97 steps | card us | row DMA us | logits DMA us |
|---|---|---|---|
| plain, cdma_wait napping first | 465 | | (not timed) |
| memory, napping first | 535 to 547 | 69.0 to 70.8 (max 92 to 120) | |
| plain, cdma_wait spinning first | 360 to 366 | | 9.4 to 10.4 |
| memory, spinning first | 375 to 376 | 3.0 to 4.4 (max 5.5 to 78) | 10.2 to 10.4 |

The wait of card/cdma.c napped 5 us between polls, and the card kernel's default timer slack (50 us) made every
completion cost 60 to 70 us; the spin-first wait (card/cdma.md) is what every per-step DMA needed. Greedy throughput
through the host loop: 2,686 to 2,734 tokens/s plain, 2,614 to 2,618 with the memory (the host sampler itself is
unchanged).

## The word filter repaired (12:5x to 13:1x, user: "fix the word filter")

The first filter judged only the candidates within 10 nats of the top and penalised the bad ones uniformly (7 nats):
a window whose candidates were all bad sent the broken word out, and the word committed as a valid prefix could not
be completed. Measured on the way (results/orchestra-4-*.log, results/orchestra-5-*.log, tools/misses.c):

1. the repair's first form (a 2-nat clamp below the best valid continuation) leaked through the sampler's repetition
   penalty: a valid top logit of 15 in the last 64 tokens becomes 12.5, a bad piece not in the history keeps 13;
2. the mix word list (every word with corpus count >= 3) holds transcript junk ("ap" 32, "ud" 39, "ser" 20) whose
   prefixes open junk paths ("udainilate" = u + d + ain + il + ate, every step on a listed prefix);
3. the end-of-text piece, exempt from the filter, closed 4 of the 4 residual broken words.

host/orchestra.h now judges every piece every step (the window cannot bound what the repetition penalty lets through):
a piece is valid when the letters it adds keep the word on a listed word or a proper prefix of one and every word it
closes is listed; every bad piece goes to min(value - 7, best valid - 16); words count from corpus count 50 on
(`--word-min`, 11,396 words in the mix list, 11,244 in the stories list; the scorer keeps the full list); a piece that
closes the word (3,039 of the 4,096, EOT among them) has a constant verdict computed once, a letter-led piece is
screened by whether its first letter can follow the partial word, so the per-step cost is 50 to 80 us (the naive
per-piece scan cost 223 us). results/orchestra-6-*.log, 512 tokens x seeds 1 to 3, the identity configuration
byte-identical to plain on all nine samples:

| model | configuration | hit | hit5 | big | rep4 | host us/token |
|---|---|---|---|---|---|---|
| stories (run4off) | plain | 0.993 to 0.998 | 0.964 to 0.988 | 0.974 to 0.995 | 0.043 to 0.046 | 323 to 326 |
| stories | filter only | 1.000 x3 | 1.000 x3 | 0.990 to 0.997 | 0.019 to 0.061 | 392 to 399 |
| stories | lambda 0.30, filter, block | 1.000 x3 | 1.000 x3 | 0.997 to 1.000 | 0.007 to 0.012 | 395 to 401 |
| mixed (A) | plain | 0.924 to 0.963 | 0.673 to 0.850 | 0.831 to 0.910 | 0.009 to 0.027 | 324 to 328 |
| mixed (A) | filter only | 1.000 x3 | 1.000 x3 | 0.917 to 0.946 | 0.012 to 0.021 | 377 to 386 |
| mixed (A) | lambda 0.30, filter, block | 1.000 x3 | 1.000 x3 | 0.976 to 0.992 | 0.014 to 0.026 | 399 to 402 |
| memory (B + tables) | plain | 0.902 to 0.954 | 0.633 to 0.803 | 0.822 to 0.872 | 0.000 to 0.015 | 325 to 328 |
| memory (B + tables) | filter only | 1.000 x3 | 1.000 x3 | 0.898 to 0.948 | 0.006 to 0.021 | 378 to 384 |
| memory (B + tables) | lambda 0.30, filter, block | 1.000 x3 | 1.000 x3 | 0.973 to 0.986 | 0.008 to 0.029 | 398 to 404 |

Human bands (TinyStories 256-word windows): hit5 1.000, big 0.995 (p10 0.986), rep4 0.022 (p90 0.036). The filter
alone makes every word a listed word; the mixture is what makes the pairs human (big), the block what keeps rep4 down
(filter alone on the stories model: rep4 up to 0.061). These metrics are lexical; meaning is not scored.

## Readiness for stages 2 and 3 (13:0x, user: "get ready for stages 2 & 3")

lib/cx-moe.h fixes the layout both stages share: the expert pool (61,440 experts x 17,472 F32, product keys 248 x
248 over the two 40-wide halves of x, transposed to 80 KiB, a home-expert bias so the converted model computes the
trained model at step 0) and the context ring in the part at 52 MiB (k, v of every head out: 61,440 B per step; the
top-4 values, scores and step indices back: 130,560 B; two slots each, CX_CTX_C_SEQ as the DMA status word,
CX_CTX_H_SEQ from the host). The budgets the plan computed are now measured on the card (nr_hugepages 2,700):

| stage 2, tools/expert-bench.c (results/moe-2-expert-bench.log): 120 heads in lockstep, 200 steps | own work per thread | lockstep wall per step | aggregate |
|---|---|---|---|
| A: one random expert of the 4 GiB pool copied into the head's L2 slot (prefetch 8 or 32 lines ahead: no difference) | 93.9 to 95.6 us (slowest 98.0 to 99.7) | 118.2 to 120.1 us | 69.8 to 70.9 GB/s |
| B: the same, the next step's expert prefetched into L2 during the previous step | 78.7 to 91.5 us | 99.8 to 113.6 us | 74 to 84 GB/s |
| C: the routing kernel (2 x 40 x 16 FMAs, the key row two ahead prefetched: the core is in-order) | 8.6 to 8.8 us | 18.4 us | |

So an expert read costs about a third of today's 324 us forward if it sits on the critical path; the pool filled
4,293,918,720 B in 2.71 s and every copy was verified (0 bad words). The per-core L2 holds the two slots (2 x 68 KB)
beside the heads.

| stage 3, tools/ctx-bench.c + host/ctxserve.c (results/moe-3-ctx-bench.log), 1,200 steps, channel 5 | host scan + answers per step | card per step |
|---|---|---|
| window 1,024, host 8 threads, full round trip (mode 0) | 256 to 275 us (max 1,370) | 313 to 333 us = out 18.4 + host 257 to 277 + in 37.7 |
| window 1,024, host 1 thread, mode 0 | 1,261 us | 1,314 us |
| window 1,024, host 8 threads, one step of lag (mode 1) | 216 us | 176 us = submit 0.7 + wait 137 + in 38.6 |
| window 256, host 8 threads, mode 0 | 39.0 us | 97.3 us = 18.6 + 40.9 + 37.7 |
| window 256, host 8 threads, mode 1 | 36.2 us | 55.7 us = 0.9 + 2.0 + 52.8 |

Every run checked 2,160 head-steps (each head's own top-1 score over its key history against the host's, and the
returned values): 0 mismatches. The 1,024-step scan is bandwidth-bound in the host's L3 (31.5 MB of keys per step:
8 threads are 4.6 times faster than 1); it fits a training step (about 800 us) but not a generation step (360 us)
without an index, while a 256-step window costs the host 36 to 39 us and the card 56 us per step with the lag, and
the 38 to 53 us answer DMA can overlap the fast-weight pass. Decided: stage 3 starts with W = 256 exact, the answer
of step n consumed at step n + 1; stage 2 with one staging slot per head and the expert read issued right after the
router (mode A's cost), the L2 prefetch of mode B as the first optimisation.

Next, in order: stage 2 (heads without their MLP, the pool loaded from the trained MLPs with the home bias, gate =
bit-identical logits at step 0 then gradients against the reference), stage 3 (ctxserve in the training loop, the
attention read-out added to the fast-weight output), the row learning rate of the n-gram memory, stage 4 (the coach).

## Direction (14:11): coherent language, efficiently; (14:36): live training with inference and interaction

"I don't really care what you do to get the model generating coherent language that can be eventually utilized for
agentic use, just make it efficient." Decision: tokens before capacity. The model had seen 1.3M tokens; a 10M-token
TinyStories run from the run4off checkpoint started at 14:12 (results/ts10m.log, 1,259 tok/s, checkpoints every 2M
tokens), the whole TinyStories stream is tokenised (data/bpe/train.u16, 562,693,978 tokens), and the host path was
made cheap: the sampler's full sort of 4096 logits became a bounded top-k selection (22 to 24 us per token, was about
300), so the host costs 77 to 81 us per token with the orchestrator (`cortex --replay` times the host path on a
logits dump without the card). Stages 2 and 3 wait for the loss curve to flatten.

"Training should include the training loop synchronously with you assisting to keep coherence inline. Design and
implement a live training method so the model can run inference while training, and get actively interacted with
while weights are tuning." docs/LIVE.md: the daemon trains between clients' bursts of commands on a training
context and serves inference on a second context over the same live weights; the host coach feeds the stream through
a ring in the part, reads the reports, samples the live model through the mailbox with the orchestrator, scores the
samples, applies the plateau policy, and takes commands (scale, lr, pause, resume, save, sample, stop) from a file
that a person or the assistant appends to. Nothing written by the supervisor and nothing generated by the model is
training data. The 10M run was meant to stop at its 4M-token checkpoint; the waiter that was to stop it polled the
card over ssh every 30 s, which wedged the phictl forward (one slot leaks per connection), so the run went to its
end: 10,000,000 tokens in 7,962.3 s, validation 6.1065 bits/token on 100,000 tokens (results/ts10m.log; the 1M-token
parent validated at 6.18: ten times the tokens gained 0.07 bits/token, the curve has flattened, which is the
condition set above for stages 2 and 3). Its checkpoints are in data/ts10m/ (every 2M tokens; ckpt-312500.bin is the
10M one, lineage 11,000,000 tokens) and the live trainer takes over from ckpt-312500. Checkpoints of live runs are
pushed by DMA into the part's stage and copied by the coach to the host (docs/LIVE.md), so neither the card's /tmp
nor the control socket is in the checkpoint path. The gates (trajectory and final checkpoint identical to the batch
trainer, no interference from a generating client, the knobs, the costs) and the first live run's numbers are
appended below as they are measured.

## Live trainer: gates passed, first live run (2026-10-26 16:3x to 17:3x)

Measured on the 7120A (results/live-gate.log, live-gate-chain2.log, live-gate-D/E/F/G/H/I/J/K, ckpt-eval.log):

| gate | result |
|---|---|
| A trajectory | 20 of 20 per-report losses identical to `cluster-gate run` from the same checkpoint on the same 2,000 chunks; the final checkpoint the coach collects from the stage is byte-identical to the batch trainer's (md5 9859d0edfdf17d6d031e50a39fceac68) |
| B no interference | identical losses and checkpoint while a client generated 3 x 512 tokens on the live weights (2,446 to 2,648 tok/s, card 363 to 394 us/token); training 1,286 to 1,289 tok/s alone, 1,041 in the 100-chunk window shared with the clients |
| C knobs | scale 0.5 in effect at the next report, pause 5 s (window rate 428 tok/s), resume, save (results/live-gate-C/ckpt-947.bin, card push 6.5 ms, host copy 8 to 30 ms), stop with the final checkpoint |
| D push fidelity | a paused run's chunk-0 checkpoint equals the loaded file (md5 440e9d5390415c85d9e6a8b3fcb4a656) |
| E determinism | two 300-chunk live runs identical to each other |
| K preemption | under 240 unpinned busy loops on the card (the daemon's threads preempted, 22 to 26 tok/s) chunk 1 and a 10-chunk run equal the batch trainer's (md5 4abd0c40..., 0470b1e6...) |

Two faults were found and fixed on the way:
1. KNC excess precision in the Adam bias powers: the batch trainer kept the cumulative float products in locals (held at extended precision in registers), the daemon in float globals (rounded every chunk); checkpoints were identical after chunks 1 and 2, different after 3, losses visibly apart at chunk 86 (5.3884 vs 5.3885). Both trainers now use `cl_adam_schedule` (card/cluster.h): powers from the step count in double, the rate and epsilon through volatile floats.
2. A round race in the daemon: the round's action word was read right after the round's opening barrier; an idle round has no second barrier, so thread 0 (napping, then deciding) could overwrite it for the next round before a preempted thread had read it, leaving that thread one barrier out of step for good (losses off, NaN within chunks; chunk 1 wrong under 240 busy-loop hogs while two contending batch trainers stayed exact). The action is now double-buffered by round parity.

Also: checkpoints are pushed by DMA into the part's stage and copied to the host by the coach (CX_T_SAVES counts the pushes; nothing on the card's disk); LOAD zeroes every training buffer; idle rounds nap 100 us (a spinning idle slowed a co-running card job about 50x); the daemon starts detached by `phi run` (the ssh forward is wedged by the afternoon's poll loop).

Coherence over the lineage (ckpt-eval.log, plain sampling, 3 seeds x 512 tokens): the model's own hit5 and big are flat from 1M to 11M tokens (hit5 0.964 to 0.988 at 1M, 0.965 to 0.967 at 11M; big 0.974 to 0.995 at 1M, 0.945 to 0.957 at 11M), repetition improved (rep4 0.043 to 0.019 to 0.028); with the orchestrator every checkpoint scores hit5 1.000 and big 0.983 to 1.000. Together with validation 6.18 to 6.1065 bits/token over 10x the tokens: capacity-bound. The live run from ckpt-312500 (`--start 312500` on data/bpe/train.u16, results/live.log, commands in results/live.cmd) demonstrates the interactive loop; the efficient next step for coherence is capacity (stage 2 experts, stage 3 context).

## The harness: cortex-tui (2026-10-26 18:38 to 19:2x)

The user's directions: a Rust TUI to chat with the model, opened on the desktop; "Take from Mechanical Jev's TUI
design", then "Just take the visual style, I want a unique harness for the Phi Cortex project"; "wire in CLAUDE.md to
automatically be utilized as the system prompt"; "Remove anything that can't or isn't wired up to the harness"
(visually and in the code base). Built and measured: docs/TUI.md. `tui/` (crossterm only, seaof.glass's palette as
MechJev's TUI, the terminal layer and the single-line editor carried over with attribution), `tools/chat.sh` opens it in
a konsole window. A turn runs `cortex --live` with the system prompt + the transcript as the prompt (3,716 prompt
tokens = 1.4 s a turn before the first generated token; 2,678 tok/s overall); the side panel reads the part's live
words and the coach's log; the run screen steers the coach (resume/save/pause/scale/stop verified against a real
paused coach session). The daemon now counts CX_C_HEART at 10 Hz (build 19:14) so the harness can say it is alive.

19:22 "Get a jinja script written so the model also has prompt coherence and can be prompted as a coherent identity":
`tui/cortex.jinja`, rendered by the harness every turn (minijinja) as a chat template with the identity paragraph,
`Human:` / `Cortex:` labels and an exported `stop` the client takes as `--stop`; docs/TUI.md "The frame". Measured:
75 tokens at 2,706 tok/s with a 3,820-token prompt; the replies remain third-person stories (no chat fine-tune yet).

## State of the tree at 2026-10-26 21:5x (public since then)

Built and measured today, in order: the live trainer (gated bit-identical to the batch trainer), the chat harness
`cortex-tui` with `tui/cortex.jinja` as the frame of every turn, the training monitor `cortex-watch`, the Opus 4.6
chat data pipeline and two fine-tune passes (the first measured: chat 10.6185 to 8.4801 bits/token, stories 5.7531 to
6.1197; the second, 75 % stories at 5e-4, running), the coach's `--save-every` and `--no-stop`, the daemon's heartbeat.
Rejected with numbers: 3-bit weights (50 bits/token). Kept as the user decided: F32, the 4,096-id tokenizer at 3.13
bytes per token on chat text. Next by the measurements: capacity (stages 2 and 3 above) rather than more tokens on
this model, and a chat fine-tune on more conversational data in the frame's shape.

## 22:1x: the monitor talks and steers; chat only

`cortex-watch` gained a say line, a talk section and the coach's commands (docs/TUI.md), verified against the live
run. The user's direction at 22:09: no more story training; the model is to be a chat model. The chat-only run
(four epochs of the 4,822 conversations at 1e-3, docs/CHAT.md) runs from the chat pass's final checkpoint; its
held-out curve decides whether more epochs or more conversations come next.

## 22:4x: from scratch, chat format with a think block

The user's directions at 22:18 to 22:22: a chat model from the Opus 4.6 distills from scratch (no stories, not even
their tokenizer), a well-rated set, a reasoning state like Claude's or Qwen's. Built: the marker format with seven
special tokens in the BPE, the converter for reasoning fields and the no-think share, a fresh-init checkpoint
writer, the chat vocabulary, the frame and the talk with the think block, `--vocab`. Running: the one-epoch run on
Gryphe's 24k set (31.55M tokens, lr 3e-3, 100-report windows) with its held-out curve in results/scratch-loss.log.
docs/CHAT.md has the numbers.

## 01:2x (10-27): "The model is completely incoherent"

It is, and the measurement is the finding: against n-gram tables on the same held-out tokens (tools/ngram-base.c)
the from-scratch model is below unigram now (10.04 against 9.98 bits/token at chunk 325,001; 9.35 at its best) and
3.2 bits behind a trigram (6.18); the stories model was between unigram and bigram all along (6.11 against 8.43 and
5.11). The architecture has never beaten a bigram table on any text, so the capacity work (stages 2 and 3) is what
coherent chat needs, not more data or epochs. Secondary: Gryphe's set is grouped by source (the held-out tail is one
source), and the plateau policy put the rate at the floor by chunk 90,000. docs/CHAT.md "Incoherent, and why".

## 01:31 (10-27, user): "produce an LLM capable of identity and communication in english"

The goal restated, with a paper to keep in mind: Nielsen, Y. A. & Christiansen, M. H., "Evidence for the
representation of non-hierarchical structures in language", Nature Human Behaviour 10, 579-588 (2026),
doi 10.1038/s41562-025-02387-z. Linear chunks of structure akin to parts-of-speech sequences (VERB PREPOSITION
DETERMINER, shared by "added to a" and "defined by the") are primed in four preregistered phrasal decision
experiments (N = 497) without constituents, and the effect holds in eye-tracked reading (N = 68) and conversation
(N = 358) corpora: multiword linear chunks are a basic representational unit, not reducible to hierarchical
constituents.

What this fixes in the plan: the layer the paper calls basic is the layer the measurements say the Cortex lacks. The
n-gram baselines (docs/CHAT.md "Incoherent, and why") put a trigram table 3.2 bits/token below the from-scratch chat
model and 2.0 below the stories model, and the tree already has the chunk layer: the hashed n-gram memory of stage 1
(orders 2 and 3, 2^21 rows x 128 F32 per order in host RAM, rows over the DMA ring, gate 1a passed, run B above:
training loss -0.205 and validation -0.041 bits/token at 320,000 tokens from zero-initialised tables with SGD lr 0.1
on the rows, the row learning rate named as the first knob). The live trainer serves the memory for inference only
(docs/LIVE.md), so no live run has trained it. The next capacity step is therefore the chunk layer in the live loop,
with the trigram baseline on the same held-out tokens as the bar it has to clear, and the data aimed at identity and
plain English communication rather than reasoning traces. Identity data cannot be written by me (never train on
Claude-written text); it is human-written or the user's.

## 01:40 to 02:0x (10-27, user): the chat base, researched and measured

Ten chat corpora pulled and measured by their n-gram floors on their own 4k BPE (docs/CHAT.md "The base dataset"):
TinyChat (GPT-4o mini, BASIC English) 3.647 bits/token at 5.137 bytes per token is the base; PersonaChat (human,
personas as first-person system messages), DailyDialog (human) and the everyday assistant set ride along; the mix
of 29,992,115 tokens has floors 8.528 / 4.821 / 3.888 on its held-out tokens, 2.3 bits below the Opus text. The
Opus run stopped at chunk 455,246 (02:00:45); the base run started at 02:00:48 (data/live-base/, results/base-loss.log,
ends about 08:50). The frame's identity line is the persona form. SODA (3.539 at 4.184 B/token, 1.19M dialogues) is
the option for a larger run.

## 06:32 (10-27, user): "Rather than training on the current 6M model, start work on the MoE portion"

The stage-2 build (the sizing of 06:32 the day before, measured in "Readiness for stages 2 and 3"). Decisions, each
from the measurements or the papers:

- **An expert is the column's MLP block as it lies in the parameter layout**: W1 (144 x 64), b1 (64), Wu (64 x 128)
  are contiguous from CX_O_W1 (lib/cx-col.h), 17,472 F32 = CX_EX_FLOATS (a static assertion in lib/cx-moe.h). A
  head keeps the whole 44,512-float block; its MLP region is the slot the routed expert is fetched into
  (CX_EX_SLOT), so card/col.h and card/train.h compute on it unchanged: the step is split into cx_col_pre (through
  the fast weights to [o ; x]) and cx_col_mlp (the MLP and the outputs), the router and the fetch between them.
  The L2 picture per core is the dense one (two blocks of 178 KB) plus the 80 KiB of shared keys.
- **One pool, 61,440 experts, filled by sparse upcycling** (Komatsuzaki et al. 2022, arXiv 2212.05055): expert e
  starts as the MLP of column e mod 120 of the dense checkpoint, so every expert is a trained function. Head k's
  home expert is k; the home pair (k / 248, k % 248) of the product keys gets a bonus (the run knob CX_MOE_BIAS,
  8 at conversion = CX_EX_HOME_BIAS), so the converted model computes the dense model exactly (gate A: the same
  bytes through the same kernel). With the bonus at 0 the keys alone decide.
- **Fixed product keys first** (Lample et al. 2019, arXiv 1907.05242, the key structure; Roller et al. 2021, arXiv
  2106.04426: fixed hash routing is competitive with learned routing): KA, KB = 248 keys x 40 dims each from a
  seeded init (cx_ex_keys_init, scale 0.15), scored against the two halves of the head's x (the normed 80-wide
  input of the fast-weight pass); e = (argmax A x 248 + argmax B) mod 61,440; ties to the lowest lane. Learned keys
  (the Switch-style gradient through the factorised softmax) are the next step.
- **Experts train by plain SGD, one rank-1 update per head per step**, applied in the chunk backward: at step t
  (reverse order) every head re-fetches its step-t expert for the backward's products, computes its vectors, then
  (after a barrier) the lowest head routed to each expert at that step applies all the heads' updates to it in head
  order and writes it back; the next step's re-fetch waits for the write-backs (a second barrier). Deterministic
  whatever the collisions. W1 -= lr min dh^T, b1 -= lr dh, Wu -= lr z du^T from the step's saved vectors; lr = the
  run knob CX_MOE_LR (default CX_EX_LR 0.05, to be searched). No Adam moments (8 GiB for 1.07 G parameters).
- **The head's own gradients and Adam** are the dense ones without the MLP region (zeroed in cx_train_param_grads).
- **Checkpoints**: the dense format, column k's MLP region holding the home expert k from the pool (so ref-loss and
  the dense tools read the converted model at home routing); the pool itself stays on the card for the run (its
  save through the stage in pieces is stage 2c).
- **The reference**: tools/cref.h in MoE mode (cref_moe_init): the route in double on the same keys, the touched
  experts in double from the home MLPs, the per-step expert gradients applied after every column's products in
  column order (the card's owner order); tools/ctrain-check.c compares the routes (cl-route.f32, u32 per head per
  step), the losses, hc, the head gradients, cg, E and the experts of step 0 after the chunk (cl-ex<k>.f32).
- **Expected cost** (from the bench): the fetch on the forward's path (94 to 96 us), the re-fetch and the write-back
  per step of the backward (2 x 94 us), two barriers per backward step: about 290 us over the dense 793 us per step
  (computed, measured in gate C). The L2 prefetch of the next expert (mode B) is the first optimisation.

Files: lib/cx-moe.h (CX_EX_SLOT, home pair, keys init, CX_EX_LR), card/moe.h (router, fetch, update, store),
card/col.h (the split step), card/train.h (the forward hook, the head's gradients), card/cluster.h (the backward's
re-fetch, owner rule, write-back, usage counts), tools/cluster-gate.c (CX_MOE build: the pool in 4 GiB of huge
pages, the keys, the fill, the dumps, the home experts in the checkpoint, the usage report), tools/ref.h (the split
reference step), tools/cref.h (the MoE mode), tools/ctrain-check.c (the MoE comparisons), build.sh
(cluster-gate-moe), tools/moe-gate.sh (gates A to D). Results are appended below as they are measured.

### Results (07:1x, results/moe-gate.log, results/moe-gate/)

| gate | measured |
|---|---|
| A conversion | the MoE build with the home bonus on the seeded weights is byte-identical to the dense build (md5 of the chunk's losses, hc and the dumped columns' u/bc/g equal; 3,840 of 3,840 routes home) |
| B reference, home bonus 8 | loss within 9.0e-7, hc 3.9e-7 relative; the experts of step 0 after the chunk's SGD updates within 2.8e-7 to 5.2e-7 (W1, b1, Wu of heads 0, 59, 119); the head gradients within 1.0e-6; cg 3.6e-7; E 2.8e-7 |
| B reference, home bonus 0 | free routing: 8,101 experts touched in one chunk, every one of the 3,840 (head, step) routes the reference's (no near-ties), the experts within 3e-8 to 3e-7 with the collisions resolved by the owner rule |
| C cost | forward 634 us per step against the dense 381 in the gate harness (PMU on; L2 misses 5,853 against 2,552 per step per thread: the expert's 1,092 lines plus the 80 KiB of keys no longer resident); in the runs 540 against 326; backward 672 (home routing) to 770 (free) against 454 |
| D runs, 2,000 chunks from data/live-base/ckpt-661017.bin at lr 3e-3 (a restart: the base run ended at 3.75e-4), bus off, validation 3,200 tokens | dense: 1,280 tok/s, train 6.67 to 6.10 bits/token, valid 6.2476. MoE bonus 100 (exact home) expert lr 0.05: 821 to 825 tok/s, 6.41 to 6.09, valid 6.2707; lr 0.01: 6.15, valid 6.6394; lr 0 (experts frozen): 6.18, valid 6.3799. Bonus 8 (11.7 % of steps off home): 6.62, valid 7.4078. Bonus 0 (free from the start): 756 to 764 tok/s, 13.8 to 8.73, valid 8.7562, 20,017 experts touched |

Reading: the machinery is exact and the SGD on the experts (0.05) stands in for Adam on the MLP at 0.02 bits on this
slice (validation on 3,200 tokens carries about 0.3 of noise, docs/CHAT.md's state instability). The cost is the
expert traffic and the keys' L2 misses: 55 % more time per step. The keys stay F32 (the Cortex is F32 throughout;
07:24, user): the misses come from the expert stream, whose 1,092 lines pass through L2 on their way to the slot
and displace the resident keys and head weights twice per step, so the next optimisation is the fetch and the
write-back with clevict1 on the streamed pool lines (the tree's cx_copy_evict pattern), measured by the gate's PMU
line (5,853 misses per step now, of which the fetch itself is 1,092), then the L2 prefetch of the next expert
during the fast-weight pass (mode B of the bench), and the backward's re-fetch overlapped with the previous step. Free routing from the first step
destroys the trained function (every head gets another head's MLP): the bonus is lowered gradually across runs, or
the keys learn. CX_EX_HOME_BIAS is now 100 (8 left home on 11.7 % of the steps of a real checkpoint).

### The daemon (07:08 to 07:11, results/live-moe-gate.log)

card/cortexd.c is the MoE build: the pool (4 GiB of huge pages) is allocated at start, a LOAD with arg bit 1 fills
it from the checkpoint's MLP blocks and the heads route from then on (training and inference), the bonus and the
experts' step come from two host words (CX_M_BIAS, CX_M_LR: the coach's `--moe-bias`, `--bias-decay`, `--expert-lr`,
the `bias X` and `exlr X` lines, or `cortex --load ... --moe --moe-bias X`), a save puts each head's home expert into
its slot before the push (the file keeps the dense format), the report carries the experts updated so far
(CX_T_MOE_DISTINCT). Gate: `coach --ckpt data/live-base/ckpt-661017.bin --moe --moe-bias 100 --expert-lr 0.05` over
2,000 chunks of the chat base equals `cluster-gate-moe run` with the same knobs: 20 of 20 per-report losses
identical, the final checkpoint's md5 identical (27f3328c8a8710b3eaf39d937a9fa137), 824 against 822 tok/s. The
pool lived for the daemon's life at this point (a LOAD without the bit returns to the dense model); its save through
the stage in pieces was the next piece of stage 2 (done 07:41, "The pool persistent and measurable" below), then the
evicting fetch and the expert prefetch.

### The chunk layer's gate and the first MoE run (07:15 to 07:19)

The memory in the live loop (docs/LIVE.md "The memory in the live loop") passed its gate: `coach --ngram
/dev/shm/cortex-gate-tables.f32 --row-lr 0.1 --gates ones` over 2,000 chunks of the chat base equals `cluster-gate`
(CX_NG=1) with ngserve: the 20 per-report losses, the final checkpoint (md5 8661a86075745520e67345179b274264), the
gates and the full 2 GiB of tables identical (results/live-ng-gate.log); the batch reference ran at 1,252 to 1,270
tok/s with the ring, so the ring costs nothing visible beside the dense step.

The first MoE training run started at 07:18:52: the unseen rest of the chat-base epoch (chunks 661,017 to 937,252,
276,236 chunks) from data/live-base/ckpt-661017.bin at lr 5e-4 (a gentle restart of Adam: at 3e-3 the gate runs lost
0.5 bits/token at the start), the home bonus falling from 100 to 0 over the first 100,000 chunks, the experts at
lr 0.05, a checkpoint every 5,000 chunks into data/live-moe/, 824 to 825 tok/s, about 3.0 hours. What was not yet
measurable at that point: the held-out loss of the routed model on the host (ref-loss read the dense-format
checkpoint, which holds the heads and the home experts, and the pool was not saved), so that run was read by its
training loss, the expert count in the report line and the harness, until the pool save and a MoE-aware evaluator
existed (07:41, the next section; the run was then ended at chunk 26,472 and restarted with the pool saved). After
them, in order: the evicting fetch (the keys stay F32), the expert prefetch of mode B, learned keys, and the memory
and the pool together in one run.

### The pool persistent and measurable (07:3x to 07:41, user: "Start working on training the MoE model")

Until now the pool lived only in the daemon and the routed model could not be read on the host, so the first run
was provisional (ended 07:36:07 at chunk 26,472, its pool lost with the daemon's replacement). Built and gated:

- **The pool through the stage** (lib/cx-proto.h CX_T_POOL, CX_T_POOL_ACK; card/cortexd.c pool_round): rounds of
  eleven 2 MiB huge pages (23,068,672 B per round, 187 rounds for the 2,048 pages; the daemon holds each page's
  physical address), DMA lines of at most 1 MiB - 64 B inside their page, the push acknowledged by the last line's
  status write, the pull after the DMA has landed; an action between chunks, with or without a run. The coach pushes
  the pool at the run's end, on a `save-pool` line and every `--pool-every` chunks into CKDIR/pool-CHUNK.f32, and
  pulls one with `--pool FILE` before the LOAD, which keeps it (arg bit 2, `cortex --load --moe --moe-keep`).
- **The evaluator** (tools/ref-loss.c `--moe POOL [--moe-bias X]`, tools/cref.h cref_moe_pool): the experts read
  from the pool file, the seeded keys, the bonus given; the checkpoint's home experts unused.

| gate (tools/pool-gate.sh, results/pool-gate.log) | measured |
|---|---|
| a push | 4,293,918,720 B in 187 rounds, 33.8 to 36.2 s (the disk write included) |
| a pull | 1.9 s (the file in the page cache) |
| round trip | P1 pushed, pulled, pushed again: byte-identical; the checkpoint after the pull identical to the one before (the home experts from the pulled pool) |
| the evaluator's identity | the fresh pool at bonus 100 reads exactly as the dense checkpoint: 4.7754 nats = 6.8895 bits/token, top-1 0.2990 on 1,000 tokens (6.4 s against 4.0 s) |
| continuity | 100 chunks from ckpt-200 with its pulled pool at bonus 0: 10.29 then 9.94 bits/token, 5,160 experts touched in 50 chunks (run 1 had reached 16.55 at chunk 100 from free routing) |
| the routed model read | ckpt-200 with P1 at bonus 0: 17.07 bits/token on 1,000 tokens; its home experts alone 21.79 |

The MoE run restarts from data/live-base/ckpt-661017.bin with the pool pushed every 50,000 chunks and at the end,
so every pool-N.f32 beside its ckpt-N.bin is a readable state of the routed model.

## State of the tree at 2026-10-27 08:0x

| piece | state |
|---|---|
| the daemon (card/cortexd.c, the MoE build) | serving since 07:36 with the 4 GiB pool; a LOAD with bit 1 fills it, bit 2 keeps a pulled one; the pool moves through the stage in 187 rounds (CX_T_POOL / CX_T_POOL_ACK) |
| the MoE run | 07:42:16 to 08:27 on the old stream (data/live-moe/: chunks 661,017 on from data/live-base/ckpt-661017.bin, lr 5e-4 halved once by the policy, the bonus 100 to 0 over 100,000 chunks; stopped at its chunk 67,405, bonus 32.6, 120 experts updated); since 08:27 on the plain stream (data/live-moe2/, coach pid 2141271: data/chat-base2/train.u16 from --start 693,929, the checkpoint 67,405 and its pool pulled, lr 2.5e-4 with 2 halvings left, the bonus 32.6 to 0 over 32,595 chunks, expert lr 0.05, the pool with the checkpoint every 50,000 chunks); 811 tok/s; the section "08:14" below |
| the gates | all passed and in results/: moe-gate.log (conversion, reference, cost, runs), live-moe-gate.log (the daemon = the batch trainer), live-ng-gate.log (the memory in the live loop), pool-gate.log (the pool's round trip and the evaluator's identity) |
| reading a state | `build/host/ref-loss data/live-moe/ckpt-N.bin data/chat-base/valid.u16 27447 --moe data/live-moe/pool-N.f32 --moe-bias B`, B the bonus of the report line at chunk N (100 x (1 - N / 100,000), 0 from chunk 100,000) |
| the harness and the monitor | open on the desktop (tools/chat.sh, tools/watch.sh with --ckdir data/live-moe), both on the routed model, both showing the pool's words (docs/TUI.md) |
| next, in order | the evicting fetch (clevict1 on the streamed pool lines; the keys stay F32), the expert prefetch of mode B, learned keys, the memory and the pool together in one run, stage 3 |

## 08:14 (10-27, user): "The dataset while training needs to be properly formatted while training for the model to respond as a human would"

The stream, read as text, had three things against a human-like responder (docs/CHAT.md "The plain format" has the
counts and the fixes): an empty think block in every one of its 709,533 answers (the Opus-era form in a corpus
without reasoning: 4.7 % of the tokens on a constant bigram), DailyDialog's 83,071 turns tokenised with spaces
("the United States ." "I ' m"), and a frame that inserted the block too. The plain format: `chat-text
--no-think-block --detok` (tools/base-format.sh, data/chat-base2/, the same mix, split and vocabulary), the frame's
generation prompt the assistant marker alone, the harness's think switch removed; the coach now saves the pool with
the checkpoint of the same chunk (pool-N.f32 beside ckpt-N.bin; pool-50000 had landed one chunk before ckpt-50001).
The MoE run was stopped at its chunk 67,405 (its final checkpoint and pool of the same chunk) and restarted on the
new stream at the next whole conversation (tools/conv-pos: 123,365 conversations consumed, --start 693,929), the
bonus and its decay continuing (32.6 to 0 over 32,595 chunks), lr 2.5e-4 as the policy had it. The restart point:
7.5946 bits/token on the new held-out against 6.0341 on the old (27,447 tokens): the format is new to the model and
the free bigram is gone (arithmetic: 5.6 bits/token over 40 tokens with 2 of them free is 5.9 over the 38 that
remain); the first reports on the new stream: 5.75 at chunk 700, 6.06, 6.21, 6.18 at chunk 1,000 against 5.4 to 5.7
on the old. Not changed, as content decisions: the mix's sources (the everyday assistant set's AI register) and the
identity line (human-written identity data is the rule).

## 09:23 to 09:31 (10-27, user): the identity loop, the frame of reference, the identity/ego complex

Three directions in nine minutes: "Fine tune the model in a loop till it gets 'Introduce yourself' correct"; "The
model needs some frame of reference for human language and its structure"; "The model training data should
eventually 'collapse' to form an 'identity/ego complex' within the model itself allowing for increased coherence and
presence". The data spoke first: run 2 had collapsed under free routing.

| run 2 chunk | training loss (bits/token) | experts updated | the checkpoint's dense reading (5,000 new held-out tokens) |
|---|---|---|---|
| 15,000 | 6.34 | 127 | 6.0997 (ckpt-15001) |
| 20,000 | 6.08 | 167 | 6.5723 (ckpt-20001) |
| 25,000 | 6.65 | 682 | 10.26 (ckpt-25001) |
| 30,000 | 14.30 | 6,656 | 19.26 (ckpt-30001) |
| 35,000 to 86,000 | 8.04 falling to 7.00 | 12,254 to 18,410 | |
| 96,372, the final | | | greedy on the live weights: "It is cop wal wal wal wal..." |

The mechanism: an expert reached off home is another column's MLP (sparse upcycling from column e mod 120), wrong
for this head's inputs; the head's other parameters then adapt to it, so the damage is in the heads and survives a
return to home routing; the 18,000 rarely-touched experts never converge at lr 0.05; the slow decay (0 reached
after 32,595 chunks) did not save it, as gate D's free-routing run had said (8.76 against 6.25 dense). Until the
routing is redesigned (the experts a head can reach must begin as that head's own MLP: per-head sub-pools of 512
inside the 61,440 with the keys inside the sub-pool, or learned keys anchored at home), the model trains at home
routing: the dense model through the pool machinery, 802 tok/s. Run 2's last good checkpoint is
data/live-moe2/ckpt-15001.bin (6.10 on the new held-out, the new stream's chunk 708,930).

**The loop** (tools/identity-loop.sh over tools/mixstream.c and tools/replymatch.c; docs/LIVE.md "The identity
loop"): the identity set (the messages shape, a persona as the system message, the user's prompt, the answer) is
encoded with the run's vocabulary and recurs in the base every RATIO + 1 conversations; one coach run per round at
home routing with the memory on; every E chunks the test runs on the live weights between chunks (the frame's
prompt through `cortex --live --greedy --ngram`, the reply cut at the first marker, scored exact or by word LCS);
--stable K consecutive matches stop the run (final checkpoint and pool), the final weights are re-tested; a miss or
an exhausted stream starts the next round from the final state.

**The frame of reference for human language and its structure:** the trainable one is the chunk layer, the n-gram
memory (01:31), on in the loop's run with its rows trained in place (the identity's own chunks land in them); the
decode-time one is the orchestrator's count model, rebuilt from human text only (PersonaChat + DailyDialog in the
plain format, 4,467,383 tokens, 274,264 trigram contexts: data/chat-base2/human.ngc; TinyChat is GPT-4o-mini output
and ngcount's rule excludes model output), given to the harness (--orchestra data/chat-base2/human.ngc --words
data/wordlist-mix.txt, f3 toggles). What the architecture lacks is attention over the context (stage 3, sized and
not built): it cannot copy the persona from the system message, so an identity is what the weights memorise.

**The collapse toward an identity/ego complex:** --ratio is a per-round list (3,2,1): the base's share of the stream
falls round by round while it keeps the language; --stable asks for the match to hold (presence); the complex this
architecture can hold is the weights' memorised first-person facts, the memory's rows of the identity's chunks and
the persona system message at the front of every turn.

**The missing input:** the identity answers. The rule stands (identity data human-written, nothing Claude writes is
trained on): data/identity/NAME.jsonl, one conversation per line, {"messages":[{"role":"system","content":"the
Cortex persona in first-person sentences"},{"role":"user","content":"Introduce yourself"},{"role":"assistant",
"content":"the answer"}]}, variants of the prompt welcome; PERSONA.txt the same persona; REF.txt the accepted
answers one per line; then `tools/identity-loop.sh data/identity/NAME.jsonl PERSONA.txt "Introduce yourself"
REF.txt data/live-moe2/ckpt-15001.bin --base-start 22685760 --ratio 3,2,1 --stable 2`.

**The validation run (09:37, human data only):** 8 PersonaChat openings (data/identity/personachat-intro.jsonl:
the first conversations whose first reply begins with a greeting and an "I am" / "I'm" / "my name is", by a DuckDB
regexp); the test is conversation 0 (persona "I like to party. My major is business. I'm in college. I love the
beach. I work part time at a pizza restaurant.", prompt "Hi. how are you doing today?", reference "Hi I am great
just finishing up some homework how are you."), from ckpt-15001, ratios 3,2,1, 20,000 chunks a round, a test every
500 chunks, 2 matches in a row, 3 rounds at most. The greedy baseline on run 2's collapsed weights: "It is cop wal
wal wal...". The first tests: chunk 600 "I am a to a good idea, I am a wonder't it?" (LCS 0.167), 1,000 "Hi, I am
a I am a I am a..." (0.095), 1,500 "I am a I am a..." (0.100). Results below as they land (results/identity-loop.log).

**Validation 1 matched (09:47).** The greedy reply equalled the reference at chunk 13,000 and again at 13,500 (two
in a row), the run was stopped (data/live-identity/ckpt-13520.bin, pool-13520.f32, ckpt-13520-gates.f32,
tables-13520.f32) and the re-test on the final weights matched: "Hi I am great just finishing up some homework how
are you." One round, 10 minutes, about 85 exposures of the test conversation (arithmetic: 416,000 tokens at one set
conversation per three base ones, 615 tokens a cycle, eight conversations in the set), lr 2.5e-4, the memory on.
The LCS path: 0.17 at chunk 600, 0.10 to 0.15 through 3,500 ("I am a I am a I am a..."), 0.42 at 4,000 (the first
five words, then another opening's words: with no attention over the context the model cannot select among eight
personas from the system message, so the prompt's own words select the reply), 0.42 at 11,500 ("Hi I am great just
finishinging."), 1.0 at 13,000. The harness, asked the same prompt on the live weights at 09:52: the line below.
**Validation 2 (09:53):** the same test from the same checkpoint with one persona and its eight exchanges
(data/identity/personachat-one.jsonl: the shape of the Cortex task, one self with several prompts); results below.

## 10:21 to 13:2x (10-27, user): the teacher, the model from scratch, one self

The teacher (docs/TEACHER.md): Qwen3.8-27B UD-Q4_K_XL on the desktop's 3090 Ti, CLAUDE.md as its system prompt, the
user's sampling in every request; it writes the identity set, judges the loop, and since 13:09 writes the whole
corpus of a model trained from fresh weights (tools/teacher-corpus.sh, tools/teacher-scratch.sh), which trains in
rounds as the corpus grows. The first teacher loop matched "Introduce yourself" at chunk 2,074 under a rubric that
accepted a name alone ("That sounds like a good idea, but I am Cortex."); under the user's sampling 0 of 6 seeds
reproduce the identity: a greedy attractor only. 13:17: the identity is one self; the persona's line about its one
hundred and twenty columns, which had the teacher writing sibling-counting exchanges, is gone from the persona, the
writers, the judge and the frame.

13:5x: the writer yields 3 of 3 conversations a call since the complete conversation objects are salvaged one by
one (tools/json-objects.c; the teacher writes its closing delimiter in place of the object's final brace in half
the calls): 12.3 conversations and about 7.3 KB of text a minute, the vocabulary's 300 KB about 14:40 and the first
round right after it; the two desktop windows move to data/live-teacher when its vocabulary and tables exist.
