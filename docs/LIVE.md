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
   it: CX_LMAX forwards on the training context, the chunk backward with Adam (as `cluster-gate run`: warm-up over
   the first 50 chunks of the run, the bias powers as cumulative products, so the trajectory is bit-identical to the
   batch trainer on the same tokens), the bus wrapped, the loss accumulated; every CX_T_EVERY chunks a report. A
   save flag writes `/tmp/cortex/live/ckpt-CHUNKS.bin` on the card (every thread's weights, cg, the embedding).
3. Otherwise a short pause.

Inference steps happen only between chunks: the forward's per-step saves belong to the chunk's backward. A LOAD
replaces the live weights and resets the Adam state; the n-gram memory of a loaded checkpoint serves inference only
(training in live mode is the plain model, bus off unless the daemon was started with CX_BUS=1).

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
  same checkpoint on the same tokens: the per-report mean losses must be identical.
- B, no interference: the same with a client generating through `cortex --live` between chunks: identical losses.
- C, knobs: scale, pause, resume, save and stop act as specified (the rate shows at the next report, the chunk count
  stops and resumes, the checkpoint file appears, the run ends with the final checkpoint).
- D, cost: tokens/s of the live run idle, and the client's tokens/s and the run's tokens/s while a client generates.
