# Live training: the model trains and serves at the same time

Direction (2026-10-26 14:36): "Training should include the training loop synchronously with you assisting to keep
coherence inline. Design and implement a live training method so the model can run inference while training, and get
actively interacted with while weights are tuning, similar to the already established AI/LLM labs."

Two rules carried over from the coach (docs/COACH.md of the F64 tree): nothing the supervisor writes is training
data, and nothing the model generates is training data. The gradient comes from the human token stream the coach
feeds; the supervisor (the coach's policy, and a person or an assistant appending commands) controls the rate, the
pauses, the checkpoints and the stop, and measures the samples.

## Pieces

| piece | where | role |
|---|---|---|
| `card/cortexd.c` | card, 120 threads for its whole life | each round thread 0 decides: a session's command (LOAD, RESET, BUS, STEP, EXIT), a training chunk, a checkpoint save, or nothing. Two contexts share the live weights: the training context (each column's hot-block fast weights T and the training bus) and the inference context (a second T per column, its own bus and step count). A STEP runs on the inference context; a chunk trains on the training context. |
| `host/coach.c` | host | feeds the ring from a token file (up to 960 chunks ahead), logs the reports, samples the live model through the mailbox every N chunks (sampled decoding with the orchestrator, scored by `tools/coherence.h`), applies policy v2 to the learning rate, takes commands from a file, requests checkpoints, ends the run. |
| `host/cortex.c --live` | host, anyone | a session on the live weights: no checkpoint staged; the bus defaults to off (the trained configuration). `--load CKPT` loads a checkpoint into the daemon and leaves. |
| ring and knobs | `lib/cx-proto.h` CX_T_* at part + 55 MiB, the ring at 56 MiB | 1,024 slots of 128 B, each the 33 ids of a chunk (its inputs and the next token); host words: magic "TRNR", chunks published, base rate, scale, flags (save bit 0 with an acknowledge handshake, stop bit 1, pause bit 2), report cadence; card words: chunks trained, report count, loss, rate in effect, tokens/s, the flags acknowledged, the last checkpoint's chunk, the state. |

## The daemon's round

1. If a session is open (or a new one is waiting), poll the mailbox for up to CX_BURST_S = 3 ms: a command found
   becomes the team's action. A generating client's commands arrive about 80 us apart, so it keeps the card at full
   speed until it stops; an idle session yields after 3 ms and waits at most one chunk for its next command. A
   session without a first command within 5 s is abandoned (the dangling-session rule).
2. Otherwise, if the coach's magic is set, no pause flag, no stop flag, and a chunk is in the ring: the team trains
   it: CX_LMAX forwards on the training context, the chunk backward with Adam on the schedule both trainers share
   (`cl_adam_schedule` in card/cluster.h: warm-up over the first 50 chunks of the run, the bias powers from the step
   count in double, the rate and epsilon forced through volatile floats; cumulative float products diverged by an ULP
   between the two programs after three chunks because KNC's scalar unit keeps excess precision in registers), so the
   trajectory is bit-identical to the batch trainer on the same tokens; the bus wrapped, the loss accumulated; every
   CX_T_EVERY chunks a report. When nothing is to be done the team naps 100 us per round rather than spinning. The
   round's action and argument are double-buffered by round parity: thread 0 writes slot r & 1 before round r's
   opening barrier, every thread reads it after; with a single word, an idle round (no second barrier) let thread 0
   decide round r + 1 before a preempted thread had read round r's word, and that thread ran one barrier out of step
   for good (stale votes and sums, NaN within chunks; reproduced with 240 busy-loop hogs on the card, 2026-10-26). A
   save flag pushes the live weights into the part's stage by DMA (the LOAD layout, CX_STAGE_CKPT = 23,463,424 bytes:
   the 120 columns' weights, cg, the embedding; the push's last line writes CX_T_SAVES in host memory after the
   bytes) and the coach copies the stage to `--ckdir/ckpt-CHUNK.bin` on the host (default `data/live/`). Nothing is
   written on the card's filesystem, so a checkpoint never depends on the card's /tmp, the ssh forward or the control
   socket.
3. Otherwise a short pause.

## Starting the daemon

`phi run sh -c 'cd /tmp/cortex && (CX_DMA_CHAN=3 setsid ./cortexd > cortexd.log 2>&1 < /dev/null &)'` starts it
detached over the control socket (the call returns in about 5 s; measured 2026-10-26 16:38). The ssh forward is not
needed for anything: a 30 s ssh poll loop wedged it that afternoon (one forwarder slot leaks per connection), and
every card access since goes over the control socket in a handful of calls (`phi put` the binary, `phi run` the
reference, `phi get` a tar of checkpoints).

Inference steps happen only between chunks: the forward's per-step saves belong to the chunk's backward. A LOAD
replaces the live weights and zeroes every training buffer (each thread's arena past the column: per-step saves,
gradients, Adam moments, both contexts' T; the activations; the shared block after cg's moments), so a live run
starts exactly as a fresh batch-trainer process does; the n-gram memory of a loaded checkpoint serves inference only
(C.ng is set for a session's commands and cleared for a chunk: training in live mode is the plain model, bus off
unless the daemon was started with CX_BUS=1).

## Commands (appended as lines to the coach's `--cmd` file)

`scale X` (learning-rate multiplier from the next chunk), `lr X` (base rate), `every N` (report cadence), `pause`,
`resume`, `save` (a checkpoint now), `sample` (a sample now), `stop` (the run ends after the current chunk with a
final checkpoint; the daemon keeps serving). The coach reads only lines appended after it started.

## Policy v2 (from the F64 coach, measured there: 6.0940 against 6.2813 bits/token uncoached at 320k tokens)

Every W reports the window's mean training loss is compared with the previous window; a gain under 0.01 bits/token
is a plateau; a plateau halves the scale (down to 2^-3) and requests a checkpoint; at the floor, two consecutive
windows with a negative gain end the run. A sample whose rep4 exceeds the mean of the previous three by more than
0.1 counts as one plateau, at most once per run.

## Gates (run before the first live run; results below as they are measured)

- A, trajectory: the live daemon over N chunks of a token file from a checkpoint against `cluster-gate run` from the
  same checkpoint on the same tokens: the per-report mean losses must be identical, and the final checkpoint the
  coach collects from the stage must be byte-identical (md5) to the batch trainer's final checkpoint.
- B, no interference: the same with a client generating through `cortex --live` between chunks: identical losses
  and the identical final checkpoint.
- C, knobs: scale, pause, resume, save and stop act as specified (the rate shows at the next report, the chunk count
  stops and resumes, the requested checkpoint lands on the host, the run ends with the final checkpoint).
- D, cost: tokens/s of the live run idle, and the client's tokens/s and the run's tokens/s while a client generates.

## Results (2026-10-26 17:3x, the 7120A; raw data in results/)

| gate | measured |
|---|---|
| A trajectory | 20 of 20 per-report losses identical to `cluster-gate run` on the same 2,000 chunks from the same checkpoint; final checkpoint byte-identical (md5 9859d0edfdf17d6d031e50a39fceac68) |
| B no interference | identical losses and checkpoint with a client generating 3 x 512 tokens meanwhile (client 2,446 to 2,648 tok/s, card 363 to 394 us per token); training 1,286 to 1,289 tok/s alone, 1,041 tok/s in the shared 100-chunk window |
| C knobs | scale at the next report, pause (window rate 428 tok/s), resume, save (card push 6.5 ms, host copy 8 to 30 ms), stop with the final checkpoint |
| D push fidelity | a paused run's chunk-0 checkpoint equals the loaded file (md5 440e9d5390415c85d9e6a8b3fcb4a656) |
| E determinism | two 300-chunk live runs identical |
| K preemption | chunk 1 and a 10-chunk run under 240 unpinned busy loops equal the batch trainer's (md5 4abd0c40..., 0470b1e6...) at 22 to 26 tok/s |

Faults found and fixed by the gates: the Adam schedule's KNC excess precision (chunks 1 and 2 identical, 3 different, losses apart at chunk 86; now `cl_adam_schedule`) and the round race on the action word (one thread one barrier out of step after a preemption during an idle round; now double-buffered by round parity). The batch trainer itself is deterministic and preemption-safe (two contending runs identical, md5 0470b1e6...).

First live run (17:33, results/live.log): from data/ts10m/ckpt-312500.bin at chunk 312,500 of data/bpe/train.u16, 1,292 to 1,294 tok/s, reports every 100 chunks, a sample every 2,000 chunks with the orchestrator, commands through results/live.cmd. A chat during it: `cortex --live "Once upon a time" 64 --orchestra data/stories-10m.ngc --words data/wordlist-stories.txt` at 2,015 tok/s overall (card 374 us, host 127 us per token).

## Operating rules learned on 2026-10-26 (19:xx to 21:xx)

- **Starting the daemon after a card boot:** the card boots with no huge pages and every Cortex card program needs
  them: `phi run sh -c 'echo 2700 > /proc/sys/vm/nr_hugepages; mkdir -p /tmp/cortex'`, then `phi put
  build/card/cortexd /tmp/cortex/cortexd`, then the detached start on `CX_DMA_CHAN=3`, then `cortex --load CKPT`
  (its bus defaults off since 19:17, the trained configuration). `phi up` takes 23 s to the agent.
- **The heartbeat:** thread 0 counts CX_C_HEART every 100 ms (`CX_HEART_S`, build 19:14); the harness and the monitor
  call the daemon absent when it has not moved for 2 s.
- **A client is never killed:** a client that dies after its first command leaves the session open and the next
  client is never taken (poll_session abandons only a session without a first command). The harness lets a reply
  finish before quitting.
- **A stopped `phi run` does not stop the card's job:** three validation passes kept running after their clients
  were stopped and held the live training to 42 tok/s; the control socket's run channel and the ssh forward were
  both wedged by then, so the card was rebooted. The batch trainer only runs on a card without the daemon (beside the
  napping daemon a 50,000-token validation took over 300 s).
- **The policy ends a flat run in minutes:** on fresh stories every window is a plateau: three halvings, two rising
  windows, stop (3.5 min, 20:24). Runs meant to keep the card busy use `--save-every N --no-stop` (21:4x): a
  checkpoint every N chunks, the policy's halvings and their checkpoints kept, the run ended only by the stream or a
  `stop` line. `--window 100000` (the first workaround) switched the policy's checkpoints off with it: an hour of
  the second chat pass had none until a `save` line.
- **The coach's options now:** `--ckpt CKPT` (a LOAD; without it the run continues on the live weights), `--start
  CHUNK`, `--chunks N`, `--lr X`, `--every N`, `--window W`, `--min-gain B`, `--halvings H`, `--sample-every N`,
  `--sample-tokens N`, `--save-every N`, `--no-stop`, `--prompt TEXT`, `--orchestra NGC`, `--words LIST`, `--cmd
  FILE`, `--ckdir DIR`, `--paused`, `--log FILE`, `--cortex PATH`, `--vocab FILE`. A run's chunk counter and its
  checkpoint names start at 0 (stream chunk = name + `--start`).
- **The runs of the day** (results/live.log): the first live run from ckpt-312500 (10,021 chunks, ended by the
  policy), the chat passes and the stories runs of docs/CHAT.md.

## The memory in the live loop (2026-10-27 06:3x: built; its gate tools/live-ng-gate.sh not yet run)

The n-gram memory (MoE stage 1, the chunk layer of PLAN-MOE.md) trains with a live run when the coach is given
`--ngram FILE`. FILE holds the tables (host/ngtables.h: 2 orders x 2^21 rows x 128 F32 = 2 GiB, created zeroed when
absent) and belongs in RAM (/dev/shm): the coach maps it shared and trains it in place, and every client of the live
model maps the same file read-only for its rows (`cortex --live ... --ngram FILE`, the harness's `--ngram FILE`, the
coach's own sampler), so inference reads the rows as they are trained.

| piece | what it does |
|---|---|
| the coach's ring thread (`ng_serve`) | the batch trainer's lockstep with host/ngserve.c: chunk c's row gradients applied once the card's DMA landed them (CX_NGR_C_SEQ), then chunk c + 2's rows gathered into the slot the card finished with and published (CX_NGR_H_SEQ); the chunks are the run's, numbered from 0, their stream positions from `--start`; `--row-lr X` the rows' SGD step (0.1 as the mixed run B) |
| CX_T_NG (host word) and CX_T_NG_ACK (card word) | the coach sets CX_T_NG while it serves; the daemon trains a chunk with the memory only when CX_T_NG is set and gates were loaded (`--gates` with `--ckpt`: a run's ckpt-N-gates.f32 or `ones`), and says so in CX_T_NG_ACK; the coach stops the run if the daemon answers 0 (a run that would silently train the plain model) |
| the daemon (card/cortexd.c) | takes a chunk only once its rows are published (so it never waits in the ring), pulls them by DMA (cl_ng_begin), pushes the row gradients back and trains the gates with Adam (cl_bwd_chunk), exactly as the batch trainer; a save pushes the gates after the checkpoint (CX_STAGE_GATES, the push's last line) |
| checkpoints | ckpt-N.bin as before, ckpt-N-gates.f32 beside it, and the tables as tables-N.f32 every `--tables-every` chunks (default 100,000, at a `--save-every` checkpoint), on a `save` line and at the end (2 GiB each; the final copy is exact, the others trail the last row updates by the copy's seconds) |
| clients | CX_OP_NGRAM tells the daemon whether the session writes rows (a client without `--ngram` gets the plain model, never stale rows); `cortex --ngram TABLES` maps the file in milliseconds (the 2 GiB read of the earlier form is gone); `tools/ref-loss.c --ngram TABLES GATES` measures a checkpoint with its memory on the host |

Not measured yet: the gate (losses, final checkpoint, gates and tables identical to cluster-gate + ngserve over
2,000 chunks) and the cost of the ring thread beside the coach's main loop.

## The MoE model in the live loop (2026-10-27 07:1x, gated)

`coach --ckpt CKPT --moe [--moe-bias X] [--bias-decay N] [--expert-lr X]`: the LOAD carries arg bit 1, the daemon
fills its pool (4 GiB of GDDR huge pages, allocated at start) from the checkpoint's MLP blocks (expert e = column e
mod 120) and every head routes from then on, in training chunks and in clients' steps alike (card/moe.h, PLAN-MOE.md
"Stage 2 build"). The bonus and the experts' step are host words the daemon reads with every chunk and every session
command (CX_M_BIAS, CX_M_LR): the coach writes them at start, `bias X` and `exlr X` lines change them, and
`--bias-decay N` lowers the bonus linearly to 0 over the run's first N chunks at each report. A save puts each head's
home expert into its slot before the push, so the checkpoints keep the dense format; the pool itself lives for the
daemon's life (a LOAD without the bit returns to the dense model), its save is the next piece. The report line
carries the experts updated so far (CX_T_MOE_DISTINCT) and the bonus in effect. Gate (results/live-moe-gate.log):
2,000 chunks from data/live-base/ckpt-661017.bin with bonus 100 and expert lr 0.05: the 20 per-report losses and the
final checkpoint's md5 identical to `cluster-gate-moe run` with the same knobs, 824 against 822 tok/s. A client
needs nothing new: `cortex --live` and the harness talk to the MoE model as they did to the dense one.
