# The context memory (stage 3): 16,384 steps a head, on the card

2026-10-27 15:52, the user: "Model length for the Xeon Phi (student) model should be 16K for KV-cache so the student
can remember its previous responses." The Cortex had no attention and no KV cache: its context was the carried
fast-weight state (64 x 64 a column) and the n-gram tables. This stage gives every column a memory of the last
16,384 steps' keys and values and a read-out over them.

## What it is

| piece | where | what |
|---|---|---|
| the ring | GDDR, huge pages (card/ctx.h) | per head the last 16,384 steps' k (the fast-weight key, unit-norm, 64 F32) and v (64 F32): 8 MB a head, 1,140,850,688 B for 120 heads with the index |
| the index | GDDR | online IVF: 256 centroids a head seeded by the head's first 256 keys, each arriving key moving its nearest centroid toward itself by 0.05 (renormalised) and filed under it (lists of 1,024, the oldest overwritten); a query reads the 8 nearest lists |
| the read-out | card/ctx.h cx_ctx_step, in the step after the fast weights | the step's q against the last 256 steps exactly and the probed lists' entries, the top-4 by dot product, a softmax over their scores, a = sum alpha_i v_i, o' = o + sigmoid(b) a into the MLP; then the step's own k, v enter the ring |
| the gate b | the gates block's second part (lib/cx-proto.h CX_GATES_CTX), one float a column, fresh at -4 (sigmoid 0.018) | trained by Adam beside the n-gram gates (card/cluster.h); carried by the ckpt-N-gates.f32 files (1,536 B; the 1,024 B files from before are padded) |
| the backward | card/ctx.h cx_ctx_bwd | da = sigmoid(b) do', the softmax's backward into dq, db = (do' . a) sigmoid'(b); the retrieved keys and values are not differentiated (Memorizing Transformers, Wu et al. 2022, arxiv 2203.08913) |
| the views | card/ctx.h train_start, sess_start | the training retrieves from its mark (a load), a session from its reset (CX_OP_CTX_RESET, sent by the client at every start): a chat retrieves only its own turns, which the harness re-reads each turn, so the model's earlier replies are in its memory when it answers |
| the reference | tools/ref.h rctx, cref.h | the same algorithm in double, for the gate |

## Why on the card

The plan had the memory in host RAM over DMA. Measured on the model's own keys (tools/ctx-recall.c, round 5 of the
from-scratch run, 23,856 queries): the exact top-4 lie within the last 256 steps 2.6 % of the time, within 4,096 41.5
%; SimHash buckets return 7 to 27 % of them, online IVF 89.0 % at 8 probes and 93.8 % at 16. The host service
(host/ctxd.c, 8 threads, AVX2) costs 1,191 us a step at 8 probes and 2,034 at 16 with the ring full, DRAM-bound
(about 50 MB of random key reads a step against a 32 MB L3); a training step is 800 us and a generation step 460,
so the host cannot keep up. On the card each head's core reads its own candidates (about 700 keys of 256 B a step)
from GDDR beside the step, with no DMA.

## The gate (16:39)

`CX_CTX=1 ./cluster-gate bwd ctx-slice.u16 gC 32 1024 0 59 119` on the card (the dense model from the seeded weights,
1,024 warm steps filling the memory, one chunk's forward and backward) against `CX_CTX=1 build/host/ctrain-check bwd`:
every gradient within the usual precision (W 1.6e-6, u 6.8e-6, E 1.1e-5 relative; the consensus 3.7e-5; the context
gates' gradients 1.4e-7 absolute on values of 3e-4). No retrieval disagreed between F32 and double. The gate's own
forward read 41 ms a step on plain 4 KiB pages with the daemon on the same cores; the daemon's rate with the memory
in huge pages is the number that counts (below).

## Running it

The daemon takes the memory unless `CX_CTX=0` is in its environment; the card's huge-page reservation was raised to
3,300 (`sysctl -w vm.nr_hugepages=3300`; the pool holds 2,048, the memory 544). A load clears both views; the client
(`cortex --live ...`) sends CX_OP_CTX_RESET at every start. Nothing else changes for the coach, the driver, the
harness or the monitor.


## Measured in the run (round 9, 16:51 to 16:53, the first round trained with the memory)

| what | before (round 8) | with the memory (round 9) |
|---|---|---|
| the coach's rate, every 100 chunks | 800 tok/s, flat | 628 at chunk 100, 640, 533, 533, 457, 457 ... falling as the ring fills (the far lists lengthen) |
| the held-out, the first 5,001 tokens of valid.u16 | 4.4025 bits/token | 4.3652 |
| the context gates after the round's Adam | fresh at -4.000 | mean -3.981, min -4.180, max -3.775: sigmoid 0.018, moving |
| the sessions (the identity test, the pass) | | 46, 108 and 145 steps served, no error in the daemon's log; the gates file 1,536 B |

The cost is about 0.9 ms a step at a full ring: a latency-bound candidate loop (about 700 keys of 256 B a head a step
from GDDR with only 4 prefetched ahead), not bandwidth (37 MB a step is 17 GB/s of the card's 150). The prefetch
distance goes to 16 keys (CX_CTX_PFD) in the next daemon; the number it buys is measured at its first round.

**The held-out is rising, and not because of the memory.** On the first 5,001 tokens: round 4 3.8865, 5 4.1298,
6 4.0047, 7 4.3003, 8 4.4025, 9 4.3652 bits/token; on the newest 5,001 (the recipe of 15:05): round 4 4.6628, round
9 5.0885. The model got worse on both windows from round 4 on, which is where the correction set began recurring
(one correction conversation every six, 60 conversations of the last three rounds, about 18 % of a round's tokens, a
tiny set seen again and again). The response, from the next round: one in twenty-four and only the latest round's
set; the per-round restart of Adam's moments (every load zeroes them) is the other suspect and stays under watch.

**At a full ring, continuously (the fresh instance's round 1, 18:47 to 19:16, one run of 14,861 chunks):** 622
tok/s at the start, 278 to 292 once the ring was full from chunk 512 on, flat after that: 3.4 ms a step against 1.25
without the memory, so the memory costs about 2.2 ms a step at 16,384 entries, and the deeper prefetch (16 keys
ahead, 16:57) changed nothing measurable (round 11 of the old instance: 630 to 447 over a filling ring, as before).
The rounds of 1,000 chunks from a load never showed this number because a load clears the ring. The daemon of the
distillation build reports the memory's own step time (rdtsc around cx_ctx_step, "ctx N us/step"), which splits the
3.4 ms into the memory and the rest instead of inferring it.
