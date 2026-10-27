# Intel Phi Cortex, the F32 MoE tree

The F32 Cortex (its description is the second half of this file) carried on: a live trainer on the card (the daemon
trains between the clients' turns and serves inference on the live weights), a host coach that feeds, reports,
samples and takes commands, a chat harness and a training monitor in Rust, a host orchestrator over the generated
tokens, an n-gram memory in host RAM reached over DMA, the sizing of the mixture-of-experts stages, and chat data
from the Hub's Opus 4.6 sets. Everything on the card is C on the 512-bit vector unit; the host side is C, the two
terminal programs Rust. Public since 2026-10-26 21:5x at https://github.com/Lasimeri/Intel-Phi-Cortex-F32-MoE
(branch `moe`; the data and checkpoints stay local, each document says how its data was made).

| document | what |
|---|---|
| [PLAN-MOE.md](PLAN-MOE.md) | the plan, the sizing, every stage's measurements, the day-by-day directions |
| [docs/LIVE.md](docs/LIVE.md) | the live trainer: the daemon's round, the coach, the gates, operating rules |
| [docs/TUI.md](docs/TUI.md) | the harness `cortex-tui` and the monitor `cortex-watch` |
| [docs/CHAT.md](docs/CHAT.md) | the chat data, the stream, the fine-tune passes and their measurements |
| [docs/ORCHESTRA.md](docs/ORCHESTRA.md) | the orchestrator between the card's logits and the sampler |

## Running it

    ./build.sh                                                   # host tools (gcc), card programs (KNC clang), ISA audit
    cd tui && cargo build --release && cd ..                     # the harness and the monitor
    phi run sh -c 'echo 2700 > /proc/sys/vm/nr_hugepages; mkdir -p /tmp/cortex'   # after a card boot: it has none
    phi put build/card/cortexd /tmp/cortex/cortexd
    phi run sh -c 'cd /tmp/cortex && (CX_DMA_CHAN=3 setsid ./cortexd > cortexd.log 2>&1 < /dev/null &)'
    build/host/chat-text data/opus46/gryphe-24k/Opus-4.6-Reasoning-24k.jsonl data/chat-scratch/opus24k.txt
    build/host/bpe train data/chat-scratch/opus24k.txt 120000000 data/chat-scratch/vocab-chat4096.txt
    build/host/bpe encode data/chat-scratch/vocab-chat4096.txt data/chat-scratch/train.txt data/chat-scratch/train.u16
    build/host/init-ckpt data/live-scratch/ckpt-init.bin                  # fresh F32 weights, nothing trained
    build/host/cortex --load data/live-scratch/ckpt-init.bin --vocab data/chat-scratch/vocab-chat4096.txt
    build/host/coach data/chat-scratch/train.u16 --lr 0.003 --every 100 --window 100 --halvings 3 --no-stop \
        --save-every 5000 --sample-every 2000 --vocab data/chat-scratch/vocab-chat4096.txt \
        --prompt $'<|user|>What is your name?<|end|>\n<|assistant|>' \
        --cmd results/live.cmd --log results/live.log --ckdir data/live-scratch   # the run from scratch
    tools/watch.sh --ckdir data/live-scratch --vocab data/chat-scratch/vocab-chat4096.txt   # the monitor + say line
    tools/chat.sh --ckdir data/live-scratch --vocab data/chat-scratch/vocab-chat4096.txt    # the chat harness
    echo stop >> results/live.cmd                                # ends the run after the current chunk, final checkpoint
    build/host/cortex --live "Once upon a time" 64 --no-echo --stop-eot --orchestra data/stories-10m.ngc

Measured on 2026-10-26: the live trainer 1,219 to 1,294 tok/s, a client's STEP 367 to 376 us per token between
chunks, a framed turn of the harness 2,706 tok/s overall; the live trajectory byte-identical to the batch trainer's.

| stage | state | where |
|---|---|---|
| 1a n-gram memory: host tables, rows and gradients over the card-owned DMA ring, gated against the double reference | passed (loss equal to 9.1e-7, gradients within 6.4e-7 relative) | host/ngserve.c, card/cluster.h, tools/ctrain-check.c, results/moe-1a-gate.log |
| 1b the daemon's ring: a memory-trained checkpoint generates with its 2 GiB tables (the host writes each step's two rows, the card pulls them by DMA before the forward) | gated (every step's logits within 5.9e-7 of the double reference over 97 steps, argmax 97 of 97); the memory costs 10 to 15 us per token | card/cortexd.c, host/cortex.c, tools/gen-check.c, results/moe-1-daemon-gate.log |
| orchestrator: AVX2 FMA log-softmax, trigram backoff from human counts, the word filter over every piece, 4-gram repeat block | measured on three models: hit 1.000 and hit5 1.000 on all nine samples with the repaired filter; big and rep4 inside the human bands with lambda 0.30 and the block; +50 to 80 us per token | host/orchestra.h, tools/ngcount.c, tools/orch-eval.sh, tools/misses.c, results/orchestra-6-*.log |
| second dataset: BabyLM 2024 text-only 100M words, tokenised with the TinyStories BPE, mixed in 2,048-token blocks | three runs measured: the run4off model 8.4101 bits/token on the mix, plain training 7.6649, with the memory 7.6237 (320,000 tokens each) | tools/textprep.c, tools/interleave.c, tools/mix-runs.sh, data/mix, results/mix-*.log |
| 2 experts in GDDR (61,440 x 17,472 F32 in 4 GiB, one pool for all heads, product-key routing) | sized; the budget measured on the card: one 69,888 B expert per head per step costs 94 to 96 us of thread time (118 to 120 us in lockstep, 70 to 74 GB/s), the routing kernel 9 us | lib/cx-moe.h, tools/expert-bench.c, results/moe-2-expert-bench.log |
| 3 context in host RAM over DMA (sparse attention) | sized; the round trip measured: with a 256-step window the host answers in 36 to 39 us per step and the card spends 56 us per step with one step of lag; a 1,024-step window costs the host 256 to 275 us | lib/cx-moe.h, tools/ctx-bench.c, host/ctxserve.c, results/moe-3-ctx-bench.log |
| live training: the daemon trains between clients' bursts (training context) and serves inference on the live weights (inference context); the host coach feeds the stream, reports, samples with the orchestrator, applies the plateau policy, takes commands from a file; `cortex --live` chats with the model while it trains | GATED 10-26 17:3x: trajectory and final checkpoint byte-identical to the batch trainer (also with clients generating, also under preemption); checkpoints pushed by DMA to the host; knobs verified; first live run from the 10M checkpoint (results/live.log) | card/cortexd.c, host/coach.c, host/cortex.c, card/cluster.h (cl_adam_schedule), docs/LIVE.md, tools/live-gate.sh |
| the harness: `cortex-tui` (Rust, crossterm only, the visual style of MechJev's TUI), a chat on the live weights with the card's and the live run's numbers beside it, the coach steered from the run screen, every turn framed by `tui/cortex.jinja` (the identity, the labels, the stop) over ~/CLAUDE.md as the system message | DONE 10-26 19:3x: a framed turn 2,706 tok/s overall with 3,820 prompt tokens (1.4 s before the first generated token, the system message's cost); the coach's resume/save/pause/scale/stop verified from the run screen against a real coach session; the daemon's heartbeat added (10 Hz); `tools/chat.sh` opens it in a konsole window | tui/, tui/cortex.jinja, tools/chat.sh, docs/TUI.md, host/cortex.c (`--no-echo`, `--stop-eot`, `--stop`), card/cortexd.c (CX_HEART_S) |
| the training monitor `cortex-watch` (the same crate and style): progress and time left, the run's numbers, the loss and the speed per report as column charts, the policy's window, the last sample, events, checkpoints; since 22:10 a say line too: the model on its live weights between chunks (the talk shared with the harness, tui/src/chat.rs) and the coach steered in the moment (/pause /resume /save /sample /scale /lr /every /stop-run; f2 to f6) | DONE 10-26 20:15 read-only, 22:10 interactive: checked in tmux against the live run (a turn of 60 tokens in 1.44 s at 2,699 tok/s; save, scale, cadence, pause and resume seen in the coach's log and the part's state word); `tools/watch.sh` opens it in a konsole window | tui/src/bin/cortex-watch.rs, tui/src/chat.rs, tui/src/live.rs (LiveState), tui/src/run.rs, tools/watch.sh, docs/TUI.md |
| chat fine-tunes of the story model (superseded): Opus 4.6 conversations (the relational set; the merged 17k set through the datasets-server API) in a Human:/Cortex: shape, trained live on the stories' weights | MEASURED 10-26 20:20: a 50/50 pass with stories (599k tokens, lr 1e-3) took the held-out chat loss from 10.6185 to 8.4801 bits/token and the stories' from 5.7531 to 6.1197 (results/chat-loss.log); a 73.5 % stories pass and a chat-only pass followed; 3-bit weights measured and rejected (50 bits/token); abandoned at 22:18 by the user's direction | tools/hubrows.c, tools/ref-loss.c, docs/CHAT.md |
| the chat model from scratch with a think block: Gryphe/Opus-4.6-Reasoning-24k (the verified union of the best-rated Opus 4.6 sets, 24,138 rows) in the marker format `<|system|>`/`<|user|>`/`<|assistant|><|think|>...<|/think|>...<|end|>` (seven special tokens in the BPE, a quarter of the conversations in the no-think form), its own 4,096-token vocabulary, fresh F32 weights from `tools/init-ckpt`, F32 throughout (a checkpoint is 23,463,424 bytes = 5,865,856 parameters x 4; the `.u16` files are token ids) | RUNNING since 10-26 22:40:00: one epoch of 31,553,352 tokens at lr 3e-3 with 100-report plateau windows, 1,207 to 1,213 tok/s, about 7.2 h; the fresh weights 12.27 bits/token, 9.5 to 9.8 over the first 500 chunks; held-out curve in results/scratch-loss.log; the pre-tokeniser trap (a marker glued to punctuation) found by counting the markers and fixed | tools/chat-text.c, tools/bpe.c, tools/init-ckpt.c, data/chat-scratch/vocab-chat4096.txt, tui/cortex.jinja, docs/CHAT.md |

Card tools added on the way: tools/cdma-test.c (both DMA directions on a channel), tools/partpoll.c (the part as the
daemon maps it), tools/dma-coh.c (a DMA write into GDDR invalidates every core's L2 copy: 0 stale words on 120
threads), tools/knc-one.sh (one card program built the way build.sh does it). cortexd abandons a session with no first
command within 5 s and counts CX_C_HEART every 100 ms; the DMA channel is chosen with CX_DMA_CHAN (a channel left
enabled by a killed owner stays unusable until the card reboots); card/cdma.c's wait spins before it naps (the nap
cost the card kernel's 50 us timer slack on every completion: the row DMA went from 69 to 3 us, the logits DMA from
about 70 to 10 us per step).

## The training loop as it runs (2026-10-26 22:4x)

```
╔═ desktop ════════════════════════════════════════════════════════════════════════════════╗
║ Gryphe/Opus-4.6-Reasoning-24k, 24,138 rows ─▶ tools/chat-text ─▶ chat format (106.8 MB)  ║
║ <|system|> <|user|> <|assistant|> <|think|> <|/think|> <|end|> ─▶ bpe train (4,096 ids)  ║
║ ─▶ data/chat-scratch/train.u16: 31.55M tokens at 3.141 bytes/token; valid 2.59M held out ║
║ tools/init-ckpt ─▶ ckpt-init.bin (fresh F32 weights, 12.27 bits/token) ─▶ cortex --load  ║
║ ┌─ cortex-watch (konsole): the run, the charts, the talk, the say line ────────────────┐ ║
║ │ progress + time left, loss and tok/s column charts, the window, the sample, events,  │ ║
║ │ checkpoints; /pause /resume /save /sample /scale /lr /every /stop-run ─▶ live.cmd    │ ║
║ │ a line ─▶ a cortex --live turn: markers in, think and answer shown, stop at <|end|>  │ ║
║ └──────────────────────────────────────────────────────────────────────────────────────┘ ║
║ ┌─ cortex-tui (konsole): the chat harness, the same talk, /think on|off ───────────────┐ ║
║ │ frames turns with tui/cortex.jinja (<|system|> = ~/CLAUDE.md); steers the coach      │ ║
║ └──────────────────────────────────────────────────────────────────────────────────────┘ ║
║ ┌─ host/coach ─────────────────────────────────────────────────────────────────────────┐ ║
║ │ chunks of 33 ids into the ring (part + 56 MiB), reports every 100 chunks, lr 3e-3    │ ║
║ │ windows of 100 reports, 3 halvings, --no-stop --save-every 5000, a sample per 2,000  │ ║
║ └──────────────────────────────────────┬───────────────────────────────────────────────┘ ║
╚════════════════════════════════════════╪═════════════════════════════════════════════════╝
                                         │ PCIe, DMA channel 3: the ring in; the words,
                                         │ the logits and the checkpoints (F32, 23.5 MB) out
╔═ Xeon Phi 7120A: cortexd (build 19:14) ╪═════════════════════════════════════════════════╗
║ 120 threads, F32 on the VPU: 32 forwards + backward + Adam a chunk, 1,207 to 1,213 tok/s ║
║ a client's STEP ~369 us a token between chunks; CX_C_HEART every 100 ms; 2,700 huge pages║
╚══════════════════════════════════════════════════════════════════════════════════════════╝
```

## Layout of this tree

    PLAN-MOE.md, docs/      the plan and the documents listed above
    card/cortexd.c          the train-and-serve daemon (two contexts over the live weights, the mailbox, DMA channel 3)
    card/cluster.h          the 120-column cluster, the shared Adam schedule (cl_adam_schedule), the n-gram ring
    host/coach.c            the coach: feeds the ring, reports, samples, policy, commands, checkpoints
    host/cortex.c           the client: a checkpoint or the live weights, one STEP per token, sampler, orchestrator
    host/orchestra.h        the orchestrator (AVX2 FMA); host/ngserve.c, host/ctxserve.c the memory and context servers
    tui/                    the harness cortex-tui and the monitor cortex-watch (Rust, crossterm, minijinja)
    tui/cortex.jinja        the frame of every turn: the identity, Human:/Cortex: labels, the stop
    tools/chat.sh, watch.sh the two programs in konsole windows; live-gate.sh the live trainer's gates
    tools/chat-text.c, hubrows.c, ref-loss.c, interleave.c, textprep.c, ngcount.c, bpe.c   the data tools
    results/                the gates' logs, the runs' logs (live.log), the measurements
    data/ (local only)      bpe/ (vocab4096.txt, train.u16 562.7M tokens), live/ (the live runs' checkpoints),
                            chat/, mix/ (the streams), opus46/ (the downloaded sets), stories-10m.ngc, wordlists

---

# Intel Phi Cortex

120 small F32 neural networks ("columns"), one per hardware thread of an Intel Xeon Phi 7120A (Knights Corner), two
per core, trained and run together as one language model on the card. Every floating-point operation runs on the
card's 512-bit vector unit; the host (a Ryzen 7 5800X) does the scalar work: tokenizer, sampling, I/O, checkpoints,
exchanging tokens and logits with the card through a shared-memory mailbox and the card's DMA engine. Everything is C.

Status 2026-10-26: stages 0 to 5 of [PLAN.md](PLAN.md) are done and measured on the card; stage 6 (online
learning) became the live trainer of docs/LIVE.md and stage 7 (a chat front end) the harness of docs/TUI.md, both in
this tree. The measured numbers and the open findings are in `results/`. The sections below describe the F32 model
itself as it was built; the daemon now uses DMA channel 3 and the live weights, see the first half of this file.

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
