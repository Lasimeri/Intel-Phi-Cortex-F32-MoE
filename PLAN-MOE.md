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
