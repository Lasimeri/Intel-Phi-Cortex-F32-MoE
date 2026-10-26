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
