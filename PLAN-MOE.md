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
| 0 | this plan, branch `moe`, host reference extended as each piece lands | | |
| 1a | n-gram memory, small tables (-DCX_NG_ROWS=4096 on both sides) | card vs host double: e', loss, row and gate gradients | |
| 1b | n-gram memory at 2^21 rows, training 320k tokens from `run4off` | | tok/s, L2 misses, validation vs 6.4369, samples |
| 2 | experts, E = 16, expert 0 = trained MLP | gate: converted model == trained model at step 0 (bit-identical logits), then gradients vs reference | tok/s, validation, expert usage histogram |
| 3 | sparse attention memory, top-4 | gradients vs reference | tok/s, validation |
| 4 | the whole model under the coach (`docs/COACH.md` in the F64 repository), then the chat set | | validation, coherence scores |

Checkpoints: the dense part as today (`ckpt.bin`), the GDDR parts in their own files (`ckpt-ng.bin`, `ckpt-ex.bin`)
so a run can restart with or without them.

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
(2,147,483,648 B), the checkpoints in data/mix/{a,b}-ckpt.bin; cortexd does not yet read the tables, so the B
checkpoint cannot generate with its memory until the daemon gets the ring (next in stage 1).

### The orchestrator on the mixed model (07:3x, docs/ORCHESTRA.md, results/orchestra-mix.log)

With the mix count model and word list, lambda 0.30 plus both constraints takes the A checkpoint's samples from
hit5 0.67 to 0.85, big 0.83 to 0.91 (plain) to hit5 0.97 to 1.00, big 0.97 to 0.98, rep4 0.021 to 0.026. The
lambda 0 identity check passed on both models after the log-sum-exp offset was restored in the mixture.
