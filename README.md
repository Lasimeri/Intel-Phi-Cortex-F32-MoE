# Intel Phi Cortex, the F32 MoE tree

The F32 Cortex (its description is the second half of this file) carried on: a live trainer on the card (the daemon
trains between the clients' turns and serves inference on the live weights), a host coach that feeds, reports,
samples and takes commands, a chat harness and a training monitor in Rust, a host orchestrator over the generated
tokens, an n-gram memory (the chunk layer) in host RAM trained in the live loop, the mixture-of-experts stage 2 on
the card (61,440 experts in 4 GiB of GDDR, one pool read by every head, product-key routing, the pool saved and
restored through the stage, the routed model measured on the host), and a chat base of human and BASIC-English
dialogue. Everything is F32: on the card C on the 512-bit vector unit, on the host C, the two terminal programs
Rust. Since 2026-10-27 07:42 the MoE model trains on the rest of the chat base's epoch while both terminal programs
talk to it. Public since 2026-10-26 21:5x at https://github.com/Lasimeri/Intel-Phi-Cortex-F32-MoE (branch `moe`;
the data, the checkpoints and the pool files stay local, each document says how its data was made).

| document | what |
|---|---|
| [PLAN-MOE.md](PLAN-MOE.md) | the plan, the sizing, every stage's measurements, the day-by-day directions |
| [docs/LIVE.md](docs/LIVE.md) | the live trainer: the daemon's round, the coach, the gates, operating rules |
| [docs/TUI.md](docs/TUI.md) | the harness `cortex-tui` and the monitor `cortex-watch` |
| [docs/CHAT.md](docs/CHAT.md) | the chat data, the stream, the fine-tune passes and their measurements |
| [docs/ORCHESTRA.md](docs/ORCHESTRA.md) | the orchestrator between the card's logits and the sampler |

## Running it

    ./build.sh                                                   # host tools (gcc), card programs (KNC clang, cortexd = the MoE build), ISA audit
    cd tui && cargo build --release && cd ..                     # the harness and the monitor
    phi run sh -c 'echo 2700 > /proc/sys/vm/nr_hugepages; mkdir -p /tmp/cortex'   # after a card boot: it has none
    phi put build/card/cortexd /tmp/cortex/cortexd
    phi run sh -c 'cd /tmp/cortex && (CX_DMA_CHAN=3 setsid ./cortexd > cortexd.log 2>&1 < /dev/null &)'
    # the chat base in the plain format (data/chat-base2/: train.u16, valid.u16, the vocabulary, persona.txt) comes
    # from the mix docs/CHAT.md "The base dataset" describes, by tools/base-format.sh (chat-text --no-think-block
    # --detok: no think block in the answers, DailyDialog's punctuation rejoined); a dense run from fresh weights:
    build/host/init-ckpt data/chat-base/ckpt-init.bin                     # fresh F32 weights, nothing trained
    build/host/coach data/chat-base2/train.u16 --ckpt data/chat-base/ckpt-init.bin --start 0 --lr 0.003 --every 100 \
        --window 300 --halvings 3 --no-stop --save-every 5000 --sample-every 2000 \
        --vocab data/chat-base2/vocab-base4096.txt --words data/wordlist-mix.txt --prompt "$(cat data/chat-base2/persona.txt)" \
        --cmd results/live.cmd --log results/live.log --ckdir data/live-base
    # the MoE run from a dense checkpoint: the pool filled from it, the home bonus lowered to 0 over 100,000 chunks,
    # the experts at lr 0.05, a checkpoint every 5,000 chunks and the pool (4 GiB) with the checkpoint every 50,000 and at the end
    build/host/coach data/chat-base2/train.u16 --ckpt data/live-base/ckpt-661017.bin --start 661017 --lr 0.0005 \
        --every 100 --window 300 --halvings 3 --no-stop --save-every 5000 --sample-every 2000 \
        --vocab data/chat-base2/vocab-base4096.txt --words data/wordlist-mix.txt --prompt "$(cat data/chat-base2/persona.txt)" \
        --cmd results/live.cmd --log results/live.log --ckdir data/live-moe \
        --moe --moe-bias 100 --bias-decay 100000 --expert-lr 0.05 --pool-every 50000
    tools/watch.sh --ckdir data/live-moe --vocab data/chat-base2/vocab-base4096.txt --system data/chat-base2/persona.txt   # the monitor
    tools/chat.sh --ckdir data/live-moe --vocab data/chat-base2/vocab-base4096.txt --system data/chat-base2/persona.txt    # the harness
    echo 'bias 20' >> results/live.cmd                           # the bonus now; also exlr X, save-pool, pause, resume, save, sample, scale, lr, every
    echo stop >> results/live.cmd                                # ends the run after the current chunk: ckpt-N.bin + pool-N.f32 of the same chunk
    build/host/ref-loss data/live-moe/ckpt-N.bin data/chat-base2/valid.u16 27447 --moe data/live-moe/pool-N.f32 --moe-bias B   # B: the report line's bonus at N
    build/host/coach ... --ckpt data/live-moe/ckpt-N.bin --pool data/live-moe/pool-N.f32 --moe --moe-bias B --bias-decay D ...   # resumes with its experts
    build/host/conv-pos OLD.u16 NTOK NEW.u16                     # where a run that consumed NTOK tokens of one stream continues in a re-encoding of it (--start)
    tools/identity-loop.sh data/identity/NAME.jsonl PERSONA.txt "Introduce yourself" REF.txt data/live-moe2/ckpt-15001.bin \
        --base-start 22685760 --ratio 3,2,1 --stable 2             # fine-tunes until the greedy answer matches (the set human-written; docs/LIVE.md)
    build/host/cortex --live "Once upon a time" 64 --no-echo --stop-eot --orchestra data/stories-10m.ngc   # a client, routed through the pool

Measured on 2026-10-26: the dense live trainer 1,219 to 1,294 tok/s, a client's STEP 367 to 376 us per token
between chunks, a framed turn of the harness 2,706 tok/s overall; the live trajectory byte-identical to the batch
trainer's. On 2026-10-27: the MoE trainer 812 to 825 tok/s (540 + 672 us per step against the dense 326 + 454), a
client's step on the routed model 593 us (the router and the expert fetch), a pool push 33.8 to 36.2 s and a pull
1.9 s, the evaluator's reading of a fresh pool at bonus 100 equal to the dense reading to the last digit.

| stage | state | where |
|---|---|---|
| 1a n-gram memory: host tables, rows and gradients over the card-owned DMA ring, gated against the double reference | passed (loss equal to 9.1e-7, gradients within 6.4e-7 relative) | host/ngserve.c, card/cluster.h, tools/ctrain-check.c, results/moe-1a-gate.log |
| 1b the daemon's ring: a memory-trained checkpoint generates with its 2 GiB tables (the host writes each step's two rows, the card pulls them by DMA before the forward) | gated (every step's logits within 5.9e-7 of the double reference over 97 steps, argmax 97 of 97); the memory costs 10 to 15 us per token | card/cortexd.c, host/cortex.c, tools/gen-check.c, results/moe-1-daemon-gate.log |
| orchestrator: AVX2 FMA log-softmax, trigram backoff from human counts, the word filter over every piece, 4-gram repeat block | measured on three models: hit 1.000 and hit5 1.000 on all nine samples with the repaired filter; big and rep4 inside the human bands with lambda 0.30 and the block; +50 to 80 us per token | host/orchestra.h, tools/ngcount.c, tools/orch-eval.sh, tools/misses.c, results/orchestra-6-*.log |
| second dataset: BabyLM 2024 text-only 100M words, tokenised with the TinyStories BPE, mixed in 2,048-token blocks | three runs measured: the run4off model 8.4101 bits/token on the mix, plain training 7.6649, with the memory 7.6237 (320,000 tokens each) | tools/textprep.c, tools/interleave.c, tools/mix-runs.sh, data/mix, results/mix-*.log |
| 2 experts in GDDR (61,440 x 17,472 F32 in 4 GiB, one pool for all heads, product-key routing): an expert is the column's MLP block, the head's MLP region is the slot the routed expert is fetched into, the pool filled from the dense checkpoint (sparse upcycling, expert e = column e mod 120), the home expert favoured by a bonus the run lowers, fixed seeded keys, plain SGD on the experts in the chunk backward (the lowest head sharing an expert applies every head's update) | GATED 10-27 07:1x: with the bonus the MoE build is byte-identical to the dense build; against the double reference the loss within 9.0e-7, the updated experts within 5.2e-7 (home) and 3e-7 (free routing, 8,101 experts touched in a chunk, routes identical), head gradients within 1.0e-6; the daemon's live run identical to the batch trainer (20 of 20 losses, the checkpoint's md5); cost 540 + 672 us per step against the dense 326 + 454 (821 to 825 tok/s against 1,280); on 2,000 chunks from the chat base at home routing: validation 6.2707 against the dense 6.2476 (expert lr 0.05), free routing from the start 8.7562. PERSISTENT 07:41: the pool pushed and pulled through the stage in 187 rounds of eleven 2 MiB pages (a push 33.8 to 36.2 s, a pull 1.9 s; the round trip byte-identical; `coach --pool FILE` resumes with the experts, `--pool-every N` and a `save-pool` line save them), `ref-loss --moe POOL --moe-bias B` reads the routed model on the host (a fresh pool at bonus 100 equals the dense reading exactly). TRAINING since 07:42:16: the rest of the chat-base epoch, the bonus 100 to 0 over 100,000 chunks (results/live.log, data/live-moe/); RESTARTED 08:27 on the plain stream (data/chat-base2/: no think block, DailyDialog rejoined, the same vocabulary; docs/CHAT.md "The plain format") from its chunk 67,405 with its pool at the new stream's chunk 693,929, the bonus 32.6 to 0 over 32,595 chunks, lr 2.5e-4 (data/live-moe2/); the restart point reads 7.5946 bits/token on the new held-out against 6.0341 on the old (27,447 tokens, the removed free bigram and the unseen format). COLLAPSED under free routing (08:48 to 09:35): as the bonus reached 0 the training loss went 6.08 (chunk 20,000, 167 experts) to 14.30 (30,000, 6,656 experts) and settled at 7.0 to 7.4 with 18,410 experts; the heads themselves were damaged (the checkpoint 30,001 reads 19.26 dense); greedy on the final weights "It is cop wal wal wal..."; run 2 stopped at its chunk 96,372; the model trains at HOME routing (bonus 100, the dense model through the pool) until the routing is redesigned (PLAN-MOE.md "09:23 to 09:31") | lib/cx-moe.h, card/moe.h, card/col.h, card/train.h, card/cluster.h, tools/cluster-gate.c (-DCX_MOE=1), tools/cref.h, tools/ctrain-check.c, tools/moe-gate.sh, tools/live-moe-gate.sh, tools/pool-gate.sh, card/cortexd.c, host/coach.c (--moe, --moe-bias, --bias-decay, --expert-lr, --pool, --pool-every), host/cortex.c (--moe, --moe-keep), tools/ref-loss.c (--moe), results/moe-gate.log, results/live-moe-gate.log, results/pool-gate.log, PLAN-MOE.md "Stage 2 build", docs/LIVE.md "The MoE model in the live loop" |
| 1c the memory in the live loop (the chunk layer of PLAN-MOE.md "01:31"): `coach --ngram FILE --row-lr X --gates FILE|ones`, the tables a 2 GiB file in /dev/shm mapped shared and trained in place by the coach's ring thread in the batch trainer's lockstep, every client of the live model mapping the same file for its rows (`cortex --ngram`, the harness's `--ngram`), the gates pushed with every checkpoint, the tables copied beside it on a schedule | GATED 10-27 07:18: 2,000 chunks from the chat base against `cluster-gate` with ngserve: the 20 per-report losses, the final checkpoint's md5 (8661a86075745520e67345179b274264), the gates and the full 2 GiB of tables identical; `ref-loss --ngram TABLES GATES` measures a checkpoint with its memory | host/ngtables.h, host/coach.c, host/cortex.c, card/cortexd.c, tools/ref-loss.c, tools/live-ng-gate.sh, results/live-ng-gate.log, docs/LIVE.md "The memory in the live loop" |
| 3 context in host RAM over DMA (sparse attention) | sized; the round trip measured: with a 256-step window the host answers in 36 to 39 us per step and the card spends 56 us per step with one step of lag; a 1,024-step window costs the host 256 to 275 us | lib/cx-moe.h, tools/ctx-bench.c, host/ctxserve.c, results/moe-3-ctx-bench.log |
| live training: the daemon trains between clients' bursts (training context) and serves inference on the live weights (inference context); the host coach feeds the stream, reports, samples with the orchestrator, applies the plateau policy, takes commands from a file; `cortex --live` chats with the model while it trains | GATED 10-26 17:3x: trajectory and final checkpoint byte-identical to the batch trainer (also with clients generating, also under preemption); checkpoints pushed by DMA to the host; knobs verified; first live run from the 10M checkpoint (results/live.log) | card/cortexd.c, host/coach.c, host/cortex.c, card/cluster.h (cl_adam_schedule), docs/LIVE.md, tools/live-gate.sh |
| the harness: `cortex-tui` (Rust, crossterm only, the visual style of MechJev's TUI), a chat on the live weights with the card's and the live run's numbers beside it, the coach steered from the run screen, every turn framed by `tui/cortex.jinja` (the identity, the labels, the stop) over ~/CLAUDE.md as the system message | DONE 10-26 19:3x: a framed turn 2,706 tok/s overall with 3,820 prompt tokens (1.4 s before the first generated token, the system message's cost); the coach's resume/save/pause/scale/stop verified from the run screen against a real coach session; the daemon's heartbeat added (10 Hz); `tools/chat.sh` opens it in a konsole window. 10-27 07:5x: on a MoE run its replies are the routed model's (the daemon routes every step), and the side panel and the run screen show the bonus and the experts' step the daemon routes with, the experts updated of 61,440, a pool transfer's round, and the memory when it trains; 08:2x: the think switch (F5, /think, the panel row) removed with the plain format, the frame's generation prompt is the assistant marker alone | tui/, tui/cortex.jinja, tools/chat.sh, docs/TUI.md, host/cortex.c (`--no-echo`, `--stop-eot`, `--stop`), card/cortexd.c (CX_HEART_S) |
| the training monitor `cortex-watch` (the same crate and style): progress and time left, the run's numbers, the loss and the speed per report as column charts, the policy's window, the last sample, events, checkpoints; since 22:10 a say line too: the model on its live weights between chunks (the talk shared with the harness, tui/src/chat.rs) and the coach steered in the moment (/pause /resume /save /sample /scale /lr /every /stop-run; f2 to f6) | DONE 10-26 20:15 read-only, 22:10 interactive: checked in tmux against the live run (a turn of 60 tokens in 1.44 s at 2,699 tok/s; save, scale, cadence, pause and resume seen in the coach's log and the part's state word); `tools/watch.sh` opens it in a konsole window. 10-27 07:0x: the log read whole once and then by its appended bytes (a fixed tail lost a six-hour run's opening line); 07:5x: the run block's moe and memory lines, the pool lines among the events, pool-N.f32 beside the checkpoints | tui/src/bin/cortex-watch.rs, tui/src/chat.rs, tui/src/live.rs (LiveState), tui/src/run.rs, tools/watch.sh, docs/TUI.md |
| chat fine-tunes of the story model (superseded): Opus 4.6 conversations (the relational set; the merged 17k set through the datasets-server API) in a Human:/Cortex: shape, trained live on the stories' weights | MEASURED 10-26 20:20: a 50/50 pass with stories (599k tokens, lr 1e-3) took the held-out chat loss from 10.6185 to 8.4801 bits/token and the stories' from 5.7531 to 6.1197 (results/chat-loss.log); a 73.5 % stories pass and a chat-only pass followed; 3-bit weights measured and rejected (50 bits/token); abandoned at 22:18 by the user's direction | tools/hubrows.c, tools/ref-loss.c, docs/CHAT.md |
| the chat model from scratch with a think block: Gryphe/Opus-4.6-Reasoning-24k (the verified union of the best-rated Opus 4.6 sets, 24,138 rows) in the marker format `<|system|>`/`<|user|>`/`<|assistant|><|think|>...<|/think|>...<|end|>` (seven special tokens in the BPE, a quarter of the conversations in the no-think form), its own 4,096-token vocabulary, fresh F32 weights from `tools/init-ckpt`, F32 throughout (a checkpoint is 23,463,424 bytes = 5,865,856 parameters x 4; the `.u16` files are token ids) | INCOHERENT, measured 10-27 01:2x (run started 10-26 22:40:00, one epoch of 31,553,352 tokens at lr 3e-3, 100-report plateau windows, 1,225 tok/s, ends by itself about 06:00): held-out 9.3458 bits/token at its best (chunk 100,001) and 10.0421 at chunk 325,001 against unigram 9.9789, bigram 6.9650 and trigram 6.1843 on the same tokens (`tools/ngram-base.c`); training loss flat at 8.87 to 8.97 since chunk 120,000; the harness shows word salad with `<|user|>` leaking into the answers; the fault is the architecture's (it never beat a bigram on the stories either: 6.11 against 5.11), details in docs/CHAT.md "Incoherent, and why"; the pre-tokeniser trap (a marker glued to punctuation) found by counting the markers and fixed; stopped 10-27 02:00:45 at chunk 455,246 for the chat base below | tools/chat-text.c, tools/bpe.c, tools/init-ckpt.c, data/chat-scratch/vocab-chat4096.txt, tui/cortex.jinja, docs/CHAT.md |
| the chat base: ten chat corpora researched (Exa, the Hub) and measured by their n-gram floors on their own 4k BPE; the base is starhopp3r/TinyChat (1M BASIC-English chats, trigram 3.647 bits/token at 5.137 bytes per token), with PersonaChat (human, personas as first-person system messages), DailyDialog (human) and the everyday assistant set mixed in and shuffled; the frame's identity line in the persona form | RAN 10-27 02:00:48 to 06:51:16 (ended for the MoE work at chunk 661,017 of 937,253 = 21,152,544 tokens): 29,992,115 tokens, floors on its held-out unigram 8.5276 / bigram 4.8214 / trigram 3.8883 (the bar), lr 3e-3 with 300-report plateau windows (three halvings by 03:19, at the floor since), 1,215 to 1,222 tok/s; the final checkpoint data/live-base/ckpt-661017.bin measures 5.6284 bits/token on 82,333 held-out predictions (top-1 0.3821); the 25,000-chunk curve was flat between 5.66 and 5.95 from chunk 125,001 (results/base-loss.log), the bigram bar not reached: the dense architecture's ceiling on this data; one checkpoint (550,001) measured 10.08 because its carried fast-weight state diverges along the stream while a fresh state gives 7.1 to 7.3 (docs/CHAT.md); in the harness the model answers in fluent English without an identity ("What is your name?" gave "I am feeling very fun, it really have many people do to make our unique things like that.") | data/chat-base/, data/cand/, tools/ngram-base.c, docs/CHAT.md "The base dataset" |

Card tools added on the way: tools/cdma-test.c (both DMA directions on a channel), tools/partpoll.c (the part as the
daemon maps it), tools/dma-coh.c (a DMA write into GDDR invalidates every core's L2 copy: 0 stale words on 120
threads), tools/knc-one.sh (one card program built the way build.sh does it). cortexd abandons a session with no first
command within 5 s and counts CX_C_HEART every 100 ms; the DMA channel is chosen with CX_DMA_CHAN (a channel left
enabled by a killed owner stays unusable until the card reboots); card/cdma.c's wait spins before it naps (the nap
cost the card kernel's 50 us timer slack on every completion: the row DMA went from 69 to 3 us, the logits DMA from
about 70 to 10 us per step).

## The training loop as it runs (2026-10-27 08:0x)

```
╔═ desktop: the MoE run (2026-10-27 07:42:16 on) ══════════════════════════════════════════╗
║ data/chat-base/train.u16: TinyChat + PersonaChat + DailyDialog + everyday, 29.99M tokens ║
║ host/coach --ckpt data/live-base/ckpt-661017.bin --start 661017 --lr 5e-4 --moe          ║
║ --moe-bias 100 --bias-decay 100000 --expert-lr 0.05 --pool-every 50000 --save-every 5000 ║
║   chunks of 33 ids into the ring (part + 56 MiB), 960 ahead; a report every 100 chunks   ║
║   checkpoints data/live-moe/ckpt-N.bin (23,463,424 B) + pool-N.f32 (4,293,918,720 B)     ║
║   pause/resume/save/sample/scale/lr/every/stop/bias/exlr/save-pool ─▶ results/live.cmd   ║
║ tools/ref-loss CKPT valid.u16 N --moe pool-N.f32 --moe-bias B: the routed model's loss   ║
║ ┌─ cortex-watch + cortex-tui (konsole): the run, the charts, the talk, the say line ───┐ ║
║ │ moe on: bias B, expert lr 0.05 · experts N of 61,440 · pool push round R of 187      │ ║
║ │ a say line ─▶ one cortex --live turn on the live weights, routed through the pool    │ ║
║ └─ live.rs: the part every 250 ms; run.rs: results/live.log by appended bytes ─────────┘ ║
╚═╪════════════════════════════════════════════════════════════════════════════════════════╝
  │ PCIe DMA channel 3: chunks of 33 ids in; checkpoints and pool rounds out, pool rounds in
╔═╪═ Xeon Phi 7120A: cortexd (MoE build), 120 heads on 60 cores ═══════════════════════════╗
║ ┌─ head k: the 44,512-F32 block in L2, its MLP region the slot of the routed expert ───┐ ║
║ │ pre: e ─▶ p, qb, bus r, x = RMSNorm gain, fast weights T, [o;x]; router: x . KA, KB  │ ║
║ │   + home bonus ─▶ e = (a*248 + b) mod 61,440, 9 us; fetch expert e: 69,888 B, 94 us  │ ║
║ │ mlp: h = [o;x] W1 + b1, z = SiLU(h), u = z Wu, g, bc; backward: rank-1 SGD lr 0.05   │ ║
║ │   on the expert by the lowest head sharing it, stored back; the heads train by Adam  │ ║
║ └─ 540 + 672 us per step (dense 326 + 454): 812 to 825 tok/s; a reply 593 us per step ─┘ ║
║ GDDR pool: 61,440 experts x 69,888 B = 4,293,918,720 B in 2,048 huge pages; filled from  ║
║   the checkpoint (expert e = column e mod 120) or a pulled file; pushed in 187 rounds    ║
╚══════════════════════════════════════════════════════════════════════════════════════════╝
```

## Layout of this tree

    PLAN-MOE.md, docs/      the plan and the documents listed above
    lib/cx-proto.h          the mailbox, the ring, the live-training words, the pool transfer words, the stage layout
    lib/cx-moe.h            stage 2's shapes: the expert, the pool, the product keys, the home bonus, the pool rounds
    card/cortexd.c          the train-and-serve daemon (two contexts over the live weights, the mailbox, DMA channel 3,
                            the 4 GiB pool, the pool rounds); the MoE build (-DCX_MOE=1)
    card/cluster.h          the 120-column cluster, the shared Adam schedule (cl_adam_schedule), the n-gram ring,
                            the experts' SGD in the chunk backward
    card/moe.h, col.h, train.h   the router, the fetch and the store; the column step split at the MLP; the head's step
    host/coach.c            the coach: feeds the ring, reports, samples, policy, commands, checkpoints, the memory's
                            ring thread (--ngram), the pool's knobs and saves (--moe, --pool, --pool-every)
    host/cortex.c           the client: a checkpoint or the live weights, one STEP per token, sampler, orchestrator,
                            --load with the gates and the pool bits, --ngram for the rows
    host/ngtables.h         the memory's tables file (2 GiB in /dev/shm): open, gather, apply, step rows
    host/orchestra.h        the orchestrator (AVX2 FMA); host/ngserve.c, host/ctxserve.c the batch servers
    tui/                    the harness cortex-tui and the monitor cortex-watch (Rust, crossterm, minijinja)
    tui/cortex.jinja        the frame of every turn: the identity, the labels, the stop
    tools/chat.sh, watch.sh the two programs in konsole windows
    tools/live-gate.sh, live-ng-gate.sh, live-moe-gate.sh, pool-gate.sh, moe-gate.sh   the gates (results/*.log)
    tools/cluster-gate.c, cref.h, ctrain-check.c, ref-loss.c   the batch trainer, the double reference, the evaluator
    tools/identity-loop.sh, identity-test.sh, mixstream.c, replymatch.c, conv-pos.c, base-format.sh   the identity fine-tune
                            loop, the decoding test (greedy, sampled, orchestrated), the stream mixer, the reply matcher, the
                            resume position in a re-encoded stream, the plain-format rebuild
    tools/chat-text.c, hubrows.c, interleave.c, textprep.c, ngcount.c, ngram-base.c, bpe.c, init-ckpt.c   the data tools
    results/                the gates' logs, the runs' logs (live.log, base-loss.log), the measurements
    data/ (local only)      chat-base/ (the base mix, the vocabulary), chat-base2/ (the plain stream, the vocabulary,
                            persona.txt, human.ngc), live-base/, live-moe/, live-moe2/ (the runs' ckpt-N.bin and pool-N.f32),
                            identity/ (the identity sets and the test files), live-identity/ (the loop's states), cand/ (the
                            corpora measured), bpe/, live/, chat/, mix/, opus46/ (the earlier streams and sets), wordlists

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
| n-gram tables on the stories' held-out (19,999 predictions, the first 31.55M training tokens) | unigram 8.4306, bigram 5.1127, trigram 4.1263 bits/token; the cluster 6.11 to 6.44 | tools/ngram-base.c, docs/CHAT.md |
| n-gram tables on the chat held-out (27,446 predictions, 31.55M training tokens) | unigram 9.9789, bigram 6.9650, trigram 6.1843 bits/token; the from-scratch chat model 9.3458 at best, 10.0421 at chunk 325,001 | tools/ngram-base.c, results/scratch-loss.log |
| ten chat corpora's trigram floors on their own 4k BPE (27,446 predictions each) | TinyChat 3.6472 (5.137 B/token), SODA 3.5391 (4.184), PersonaChat 4.2167, DailyDialog 4.2608, TinyDialogues 4.3300, everyday 4.6095, nano_chat 4.7011, EmpatheticDialogues 5.0464, BlendedSkillTalk 5.2401; the stories 4.1263, the Opus text 6.1843 | results/cand-floors-*.log, docs/CHAT.md "The base dataset" |
| the chat base's floors (data/chat-base, 27,446 predictions, 29.99M training tokens) | unigram 8.5276, bigram 4.8214, trigram 3.8883 bits/token; the base run's curve in results/base-loss.log, its final checkpoint 5.6284 on 82,333 predictions | data/chat-base/floors.txt, tools/ngram-base.c |
| MoE stage 2 on the card: the converted model against the dense build and the double reference (seeded weights, warm 64, one chunk of 32) | byte-identical with the home bonus; loss within 9.0e-7, hc 3.9e-7; the SGD-updated experts within 2.8e-7 to 5.2e-7 (home routing) and 3e-8 to 3e-7 (free routing, routes identical in 3,840 of 3,840); head gradients within 1.0e-6 | tools/moe-gate.sh, results/moe-gate.log |
| MoE stage 2 cost per step (the gate's harness, PMU on / the runs) | forward 634 / 540 us against the dense 381 / 326; backward 672 (home) to 770 (free) against 454; L2 misses per step per thread 5,853 against 2,552; runs 821 to 825 tok/s (home), 756 to 764 (free), dense 1,280 | results/moe-gate.log |
| MoE stage 2, 2,000 chunks from data/live-base/ckpt-661017.bin (lr 3e-3 restart, validation on 3,200 tokens) | dense train 6.67 to 6.10, valid 6.2476; home routing with expert lr 0.05: 6.41 to 6.09, valid 6.2707; lr 0.01: valid 6.6394; experts frozen: 6.3799; bonus 8 (11.7 % of steps off home): 7.4078; free routing from the start: 13.8 to 8.73, valid 8.7562, 20,017 experts touched | results/moe-gate.log, results/moe-gate/ |
| the daemon training the MoE model (coach --moe) against the batch trainer, 2,000 chunks | 20 of 20 per-report losses identical, final checkpoint md5 27f3328c8a8710b3eaf39d937a9fa137 both; 824 tok/s live, 822 batch | tools/live-moe-gate.sh, results/live-moe-gate.log |
| the daemon training the memory (coach --ngram) against the batch trainer with ngserve, 2,000 chunks | 20 of 20 losses identical, checkpoint md5 8661a86075745520e67345179b274264 both, the gates and the 2 GiB of tables identical; the batch reference 1,252 to 1,270 tok/s with the ring | tools/live-ng-gate.sh, results/live-ng-gate.log |
| the pool out of and into the daemon through the stage (187 rounds of eleven 2 MiB pages) and the host evaluator of the routed model | a push 33.8 to 36.2 s (4,293,918,720 B, the disk write included), a pull 1.9 s; pushed, pulled and pushed again byte-identical, the checkpoint after the pull identical; the evaluator with the fresh pool at bonus 100 equals the dense reading exactly (6.8895 bits/token on 1,000 tokens); a free-routing checkpoint reads 17.07 with its pool against 21.79 through its home experts | tools/pool-gate.sh, results/pool-gate.log, tools/ref-loss.c --moe, docs/LIVE.md "The pool out of and into the daemon" |
| the plain stream (data/chat-base2/, tools/base-format.sh) and the restart on it | 28,572,518 tokens (1,419,597 fewer: 2 x 709,533 think ids and 531 spaces), the same split, the round trip byte-identical, no think ids; the floors on the new held-out unigram 8.5771 / bigram 5.0386 / trigram 4.0769 (old 8.5276 / 4.8214 / 3.8883: the constant bigram was a free prediction for the n-gram models too); the MoE run's checkpoint 67,405 with its pool reads 7.5946 bits/token on the new held-out and 6.0341 on the old (27,447 tokens, 176 to 181 s per reading with the pool on the HDD) | results/base-format.log, data/chat-base2/floors.txt, docs/CHAT.md "The plain format", tools/conv-pos.c |
| run 2's collapse under free routing (the bonus 32.6 to 0 over 32,595 chunks, then free) | training loss 6.08 at chunk 20,000 (167 experts), 6.65 at 25,000 (682), 14.30 at 30,000 (6,656), 7.0 to 8.0 from 35,000 to 86,000 (12,254 to 18,410); the checkpoints' dense readings on 5,000 new held-out tokens: 15,001 6.0997, 20,001 6.5723, 25,001 10.26, 30,001 19.26; greedy on the final weights (96,372) "It is cop wal wal wal..." | results/live.log (08:27 to 09:35), data/live-moe2/, PLAN-MOE.md "09:23 to 09:31" |
| the identity loop on human data: 8 PersonaChat openings recurring in the base at ratios 3,2,1, from ckpt-15001 at home routing with the memory on, a greedy test every 500 chunks | MATCHED 09:47 (results/identity-loop.log): the greedy reply equal to the human reference at chunks 13,000 and 13,500 (two in a row), the run stopped at 13,520 with its checkpoint and pool, the re-test on the final weights exact; one round, 10 minutes, about 85 exposures of the test conversation at ratio 3, 802 tok/s; the LCS path 0.17 (600), 0.42 (4,000: the first five words then another persona's words), 0.42 (11,500), 1.0 (13,000); the second validation (one persona, its eight exchanges, the same test) ran next | tools/identity-loop.sh, tools/mixstream.c, tools/replymatch.c, data/identity/, docs/LIVE.md "The identity loop" |
| the teacher: Qwen3.8-27B UD-Q4_K_XL on the desktop's RTX 3090 Ti (llama.cpp, 127.0.0.1:8090), the user's CLAUDE.md as its system prompt, the user's sampling settings in every request | up in 40 s, 21,462 of 24,564 MiB; a 60-token turn 3.0 s, a verdict 1.3 to 2.1 s; 32 identity exchanges written in 95 s; the loop with the teacher as judge matched "Introduce yourself" at chunk 2,074 under the first rubric (the name alone: "That sounds like a good idea, but I am Cortex."), the rubric tightened to the name and two facts; under the user's sampling 0 of 6 seeds reproduce the identity (a greedy attractor only); the 35B does not fit the card (Q6_K 27.2 GiB, Q4_K_M 20.2 GiB of weights) | docs/TEACHER.md, tools/teacher-serve.sh, tools/teacher.sh, tools/teacher-judge.sh, tools/teacher-identity.sh, tools/identity-loop.sh --judge, results/identity-loop.log |
| the harness and the monitor on the identity loop | the monitor's main screen and the harness's run screen show the two models' interaction: the loop's prompt, the last tests with the student's greedy reply under the teacher's verdict and reason (tui/src/run.rs reads results/identity-loop.log by appended bytes; `--teacher-log FILE`) | docs/TEACHER.md, tui/src/run.rs, tui/src/bin/cortex-watch.rs, tui/src/view.rs |
| the harness and the monitor on a MoE run | both talk to the routed model (the daemon routes every reply through the pool); the side panel, the run screen and the monitor's run block show the bonus and the experts' step the daemon routes with, the experts updated of 61,440, a pool transfer's round of 187, and the memory when it trains; the log reader takes the report suffix, the opening line's notes, the pool lines and the pool files | docs/TUI.md "The pool and the memory in the harness and the monitor", tui/src/live.rs, tui/src/run.rs, tui/src/view.rs |
| the first MoE training run: the rest of the chat-base epoch (chunks 661,017 to 937,252) from data/live-base/ckpt-661017.bin, lr 5e-4, the home bonus 100 falling to 0 over the first 100,000 chunks, expert lr 0.05, the pool pushed every 50,000 chunks and at the end | RUNNING since 10-27 07:42:16 at 815 tok/s (results/live.log; data/live-moe/ holds ckpt-N.bin every 5,000 chunks and pool-N.f32 every 50,000; a first start at 07:18:52 without the pool save was ended at chunk 26,472 and reproduced to the digit by the restart: 5.1303 bits/token at chunk 200, 6.3275 at 400); read a state with `ref-loss ckpt-N.bin VALID NTOK --moe pool-N.f32 --moe-bias B` (B from the run's report line at that chunk) | host/coach.c, tools/ref-loss.c, data/live-moe/ |

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
