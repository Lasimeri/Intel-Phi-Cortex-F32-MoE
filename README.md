# Phi-Cortex-F32-MoE

The F32 Cortex (below) turned into a mixture-of-experts model, one head per VPU thread, with the inactive weights in
the card's GDDR, an n-gram memory in host RAM reached over DMA, and a host orchestrator over the generated tokens.
PLAN-MOE.md holds the plan, the sizing and the measurements; docs/ORCHESTRA.md the orchestrator.

| stage | state | where |
|---|---|---|
| 1a n-gram memory: host tables, rows and gradients over the card-owned DMA ring, gated against the double reference | passed (loss equal to 9.1e-7, gradients within 6.4e-7 relative) | host/ngserve.c, card/cluster.h, tools/ctrain-check.c, results/moe-1a-gate.log |
| 1b the daemon's ring: a memory-trained checkpoint generates with its 2 GiB tables (the host writes each step's two rows, the card pulls them by DMA before the forward) | gated (every step's logits within 5.9e-7 of the double reference over 97 steps, argmax 97 of 97); the memory costs 10 to 15 us per token | card/cortexd.c, host/cortex.c, tools/gen-check.c, results/moe-1-daemon-gate.log |
| orchestrator: AVX2 FMA log-softmax, trigram backoff from human counts, the word filter over every piece, 4-gram repeat block | measured on three models: hit 1.000 and hit5 1.000 on all nine samples with the repaired filter; big and rep4 inside the human bands with lambda 0.30 and the block; +50 to 80 us per token | host/orchestra.h, tools/ngcount.c, tools/orch-eval.sh, tools/misses.c, results/orchestra-6-*.log |
| second dataset: BabyLM 2024 text-only 100M words, tokenised with the TinyStories BPE, mixed in 2,048-token blocks | three runs measured: the run4off model 8.4101 bits/token on the mix, plain training 7.6649, with the memory 7.6237 (320,000 tokens each) | tools/textprep.c, tools/interleave.c, tools/mix-runs.sh, data/mix, results/mix-*.log |
| 2 experts in GDDR (61,440 x 17,472 F32 in 4 GiB, one pool for all heads, product-key routing) | sized; the budget measured on the card: one 69,888 B expert per head per step costs 94 to 96 us of thread time (118 to 120 us in lockstep, 70 to 74 GB/s), the routing kernel 9 us | lib/cx-moe.h, tools/expert-bench.c, results/moe-2-expert-bench.log |
| 3 context in host RAM over DMA (sparse attention) | sized; the round trip measured: with a 256-step window the host answers in 36 to 39 us per step and the card spends 56 us per step with one step of lag; a 1,024-step window costs the host 256 to 275 us | lib/cx-moe.h, tools/ctx-bench.c, host/ctxserve.c, results/moe-3-ctx-bench.log |
| live training: the daemon trains between clients' bursts (training context) and serves inference on the live weights (inference context); the host coach feeds the stream, reports, samples with the orchestrator, applies the plateau policy, takes commands from a file; `cortex --live` chats with the model while it trains | GATED 10-26 17:3x: trajectory and final checkpoint byte-identical to the batch trainer (also with clients generating, also under preemption); checkpoints pushed by DMA to the host; knobs verified; first live run from the 10M checkpoint (results/live.log) | card/cortexd.c, host/coach.c, host/cortex.c, card/cluster.h (cl_adam_schedule), docs/LIVE.md, tools/live-gate.sh |
| the harness: `cortex-tui` (Rust, crossterm only, the visual style of MechJev's TUI), a chat on the live weights with the card's and the live run's numbers beside it, the coach steered from the run screen, ~/CLAUDE.md as the system prompt of every turn | DONE 10-26 19:2x: a turn 2,678 tok/s overall with 3,716 prompt tokens (1.4 s before the first generated token, the system prompt's cost); the coach's resume/save/pause/scale/stop verified from the run screen against a real coach session; the daemon's heartbeat added (10 Hz); `tools/chat.sh` opens it in a konsole window | tui/, tools/chat.sh, docs/TUI.md, host/cortex.c (`--no-echo`, `--stop-eot`), card/cortexd.c (CX_HEART_S) |

Card tools added on the way: tools/cdma-test.c (both DMA directions on a channel), tools/partpoll.c (the part as the
daemon maps it), tools/dma-coh.c (a DMA write into GDDR invalidates every core's L2 copy: 0 stale words on 120
threads), tools/knc-one.sh (one card program built the way build.sh does it). cortexd abandons a session with no first
command within 5 s; the DMA channel is chosen with CX_DMA_CHAN (a channel left enabled by a killed owner stays
unusable until the card reboots); card/cdma.c's wait spins before it naps (the nap cost the card kernel's 50 us timer
slack on every completion: the row DMA went from 69 to 3 us, the logits DMA from about 70 to 10 us per step).

---

# Intel Phi Cortex

120 small F32 neural networks ("columns"), one per hardware thread of an Intel Xeon Phi 7120A (Knights Corner), two
per core, trained and run together as one language model on the card. Every floating-point operation runs on the
card's 512-bit vector unit; the host (a Ryzen 7 5800X) does the scalar work: tokenizer, sampling, I/O, checkpoints,
exchanging tokens and logits with the card through a shared-memory mailbox and the card's DMA engine. Everything is C.

Status 2026-10-26: stages 0 to 5 of [PLAN.md](PLAN.md) are done and measured on the card; stages 6 (online learning)
and 7 (chat CLI, HTTP) are not started. The measured numbers and the open findings are in `results/`.

## What a column is

Per token, column c computes (shapes in `lib/cx-col.h`, kernel in `card/col.h`):

    p   = P_in e_t                      e_t: a row of the shared embedding (4096 x 128)
    r   = attention(Wqb p, bus_{t-1})   bus: every column's 16-float output from the previous step (stop-gradient)
    x   = RMSNorm([p ; r])
    q, k, v, a, b = W x                 a = sigmoid (forget), b = sigmoid (write strength)
    S_t = a S_{t-1} + b (v - a S_{t-1} k) k^T   gated delta rule, fast weights 64 x 64 (Yang et al., ICLR 2025)
    o   = S_t q
    z   = SiLU(W1 [o ; x])              hidden 64
    u_c = Wu z (vote, 128)   g_c = sigmoid(wg z) (confidence)   b_c = Wb z (bus out, 16)

The cluster's prediction is a confidence-weighted vote: h = RMSNorm(sum_c g_c u_c), logits = E h with the head tied
to the embedding (Thousand Brains layout, Hawkins et al. 2017/2019). 44,512 parameters per column, 5,865,856 in
all. Training is Adam on the card with truncated BPTT over chunks of 32 tokens; each column's backward is local, the
bus is stop-gradient.

## Measured on the 7120A (61 cores, 1.238 GHz, 15.06 GiB GDDR5)

| What | Number | Where |
|---|---|---|
| one column forward, 120 threads in lockstep | 79.9 us per step | results/s01-col.md |
| card vs double-precision host reference, forward and every gradient | <= 1.1e-6 relative | results/s03-train.md, s04-cluster.md |
| one column training on one thread | 773-791 tokens/s | results/s03-train.md |
| one column after 2M tokens, held-out TinyStories | 6.230 bits/token = 1.573 bits/byte (unigram 8.433) | results/s03d-run2.log |
| 120-column cluster training | 1,225 tokens/s (bus on), 1,237 (off) | results/s04c-run4.log |
| cluster after 1M tokens, held-out | 6.306 bits/token (bus on), 6.183 (bus off) | results/s04-cluster.md |
| checkpoint to the card by DMA | 12.2 ms for 23.5 MB | results/s05-gen.md |
| generation through the host loop | greedy 2,144-2,182 tokens/s; sampled 1,383-1,489 | results/s05-gen.md |

Two findings that shape what comes next (details in `results/s04-cluster.md`): after 1M tokens the 120-column vote
is not better than one column trained on the same tokens, and the lateral bus hurts (its outputs pass through a
matrix the stop-gradient leaves untrained). The training forward costs 375 us per token because the per-step streams
evict the two resident columns from each core's 512 KiB L2 (2,485 misses per step per thread measured).

## Layout

    PLAN.md          the plan, the direction notes, every stage's gate and its measured result
    lib/cx-col.h     shapes, parameter layout, deterministic init shared by card and host
    lib/cx-proto.h   the mailbox protocol in the shared host-memory region
    card/col.h       one column's forward step on the VPU
    card/train.h     forward with saves, head, softmax, backward, parameter gradients, Adam
    card/cluster.h   the 120-column cluster: vote, consensus, split head, reductions, barriers
    card/cortexd.c   the generation daemon (mailbox + DMA channel 2)
    card/cdma.[ch]   the card-owned DMA channel driver (copied, MIT, provenance in the files)
    host/cortex.c    the host client: checkpoint staging, sampling, BPE in and out
    tools/           the gates: col-gate, train-gate, cluster-gate (card), the double references ref.h / cref.h and
                     their checkers, bpe.c, unigram.c, stats.c, card-ssh.sh / card-cat.sh
    results/         one report per stage with commands, numbers, spread and raw logs
    data/            vocab4096.txt, prompts, the trained checkpoints (data/s4/run4on, run4off: 120 columns + cg +
                     embedding, 23,463,424 B each; data/s3/run2: one column after 2M tokens)

## Building and running

Needs the Intel Phi 7120A stack (https://github.com/Lasimeri/Intel-Phi-7120A: the card kernel, `phi` CLI, the
shared host-memory region) and its KNC clang (`-march=knc`, Intel-Phi-LLVM). `./build.sh` builds the host tools
with gcc and the card programs with the KNC clang, then runs the stack's ISA audit on each card binary.

    ./build.sh
    phi put build/card/cortexd /tmp/cortex/cortexd
    phi run sh -c 'cd /tmp/cortex && ./cortexd' &            # the daemon holds the card's control socket while it runs
    build/host/cortex data/s4/run4on/ckpt.bin "Once upon a time, there was a little girl named" 64 --seed 1
    build/host/cortex --greedy data/s4/run4on/ckpt.bin "..." 64
    build/host/cortex --stop

Sampling defaults: temperature 0.5, top-k 75, top-p 0.95, min-p 0.05, repeat penalty 1.2 over the last 64 tokens,
applied in llama.cpp's order. A checkpoint must be generated with the bus setting it was trained with (`--bus 0`
for run4off).

Training the cluster (tokens: the TinyStoriesV2-GPT4 corpus encoded with `build/host/bpe`, see results/s02-bpe.md):

    phi run sh -c 'cd /tmp/cortex && CX_BUS=0 ./cluster-gate run train10m.u16 out 32 1000000 3e-3 valid.u16 100000 250'

## Design

```
╔═ Desktop host: Ryzen 7 5800X, 31 GiB ═══════════════════════════════════════════════════════╗
║ ┌ host/cortex (gcc, C) ─────────────────────────┐  ┌ /mnt/Intel Phi Cortex (16TB HDD) ────┐ ║
║ │ BPE 4096 encode/decode (tools/bpe.c)          │  │ data/bpe/train.u16: 562.7M tokens    │ ║
║ │ sampler: rp 1.2 (last 64), top-k 75, top-p    │◀─│ data/s4/run4on|off/ckpt.bin 23.5 MB  │ ║
║ │   0.95, min-p 0.05, temp 0.5, draw: 330 us/tok│  │ results/s0*.md, PLAN.md              │ ║
║ │ stages ckpt; STEP(token id); reads 4096 logits│  └──────────────────────────────────────┘ ║
║ └───────────────┬───────────────────────────────┘                                           ║
║  /dev/shm/phi-hostmem, Cortex part at 1344 MiB: mailbox cmd/ack 0.70 us, staged ckpt at     ║
║  +1 MiB, logits 16 KiB at +128 KiB (the ACK word is the DMA's status, written after data)   ║
╚═════════════════╪═══════════════════════════════════════════════════════════════════════════╝
                  │ PCIe Gen2 x8: mailbox read uncached by the card; DMA ch 2: ckpt in 12.2 ms,
                  │ logits out per token; control socket + ssh 2222 for files (phi put/get)
╔═ Xeon Phi 7120A ╪ 61 cores x 4 thr, 1.238 GHz, 15.06 GiB GDDR5, L2 512 KiB/core ════════════╗
║ ┌───────────────▼ card/cortexd.c (S3 clang -march=knc); thread 0 = mailbox + DMA ────────┐  ║
║ │ 120 column threads, 2 per core on cores 0-59 (CPUs 4c+1, 4c+2; core 59: 237 and 240)   │  ║
║ │ column k: 44,512 F32 params (174 KB) + fast weights T 64x64 at a prime-stride offset   │  ║
║ │   e -> P_in -> bus attention -> RMSNorm -> W -> gated delta rule -> MLP 64 -> u, g, bc │  ║
║ │ per token: bus[t] 16x120 -> votes g_k u_k -> 12 leaders x 10 -> hc = RMSNorm(sum) cg   │  ║
║ │   own ETB blocks (2-3 of 256) -> logits -> (max, sum) pairs -> 4096 logits -> DMA      │  ║
║ │   3 barriers per token (6.8 us each); measured 370-450 us per token, 2,144 tok/s greedy│  ║
║ ├ training, tools/cluster-gate run: + backward, chunk-end grads, Adam: 1,225 tok/s ──────┤  ║
║ │ shared GDDR: ETB 2 MiB (blocks of 16 tokens), ES shadow 2 MiB, GET, Adam m/v, bus[L+1] │  ║
║ └────────────────────────────────────────────────────────────────────────────────────────┘  ║
╚═════════════════════════════════════════════════════════════════════════════════════════════╝
```

## Sources

Intel Xeon Phi Coprocessor ISA Reference Manual 327364-001; Intel SSDG 328207-002; Yang, Kautz, Hatamizadeh,
"Gated Delta Networks" (ICLR 2025, arXiv:2412.06464); Hawkins, Ahmad, Cui (Front. Neural Circuits 2017) and Hawkins
et al. (2019); Eldan and Li, "TinyStories" (arXiv:2305.07759); the TinyStoriesV2-GPT4 corpus (CDLA-Sharing-1.0).

MIT licence (LICENSE).
