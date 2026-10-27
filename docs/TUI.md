# The harness: cortex-tui

Direction (2026-10-26 18:38): "Design a TUI for the model in Rust and open it on my desktop so I can chat with it";
then "Take from Mechanical Jev's TUI design", "Just take the visual style, I want a unique harness for the Phi Cortex
project", "wire in CLAUDE.md to automatically be utilized as the system prompt", and "Remove anything that can't or
isn't wired up to the harness" (visually and in the code).

`tui/` is a Rust crate (crossterm 0.28 only, no widget library) whose binary `cortex-tui` chats on the daemon's live
weights while the card trains and serves (card/cortexd.c, docs/LIVE.md), shows the card's and the live run's numbers
beside the conversation, and steers the coach from the same screen. The visual style is MechJev's TUI (seaof.glass's
palette from lasimeri.github.io: copper on near-black, lowercase, section rules, a keys row, a spinner; `tui/src/term.rs`
carried over from MechJev-v2 with attribution, `tui/src/editor.rs` its single-line half). Everything else is this
project's.

## Run it

```
tools/chat.sh            # builds tui/ when its sources are newer, opens the TUI in a konsole window on the desktop
tools/chat.sh --here     # the same in this terminal
```

Options (`cortex-tui --help`): `--tree DIR` (the Cortex tree, where `build/host/cortex` and the data live), `--orchestra
NGC|off`, `--words LIST`, `--system FILE|off` (default `~/CLAUDE.md` when it exists), `--cmd FILE` (the coach's command
file, results/live.cmd), `--log FILE` (results/live.log), `--ckdir DIR` (data/live), `--part FILE`
(/dev/shm/phi-hostmem), `--ngen N` (128), `--ctx CHARS` (4000). Colours: truecolor by default, `CORTEX_COLOR=256` for a
terminal without it, `NO_COLOR` or `CORTEX_COLOR=none` for none.

## Pieces

| piece | role |
|---|---|
| `tui/src/main.rs` | the loop: draws when something changed, waits up to 100 ms for a key, quits on ctrl+q once no reply is in flight |
| `tui/src/app.rs` | the state: screens (chat, run, help), keys, the `/` commands, the prompt of a turn, the coach's command file, the live poll |
| `tui/src/view.rs` | the screens as frames: header (where you are; the card or the turn on the right), the conversation, the side panel (from 100 columns), the run screen, the help, the footer (status row, keys row) |
| `tui/src/cortex.rs` | one turn: `build/host/cortex --live PROMPT N --no-echo --stop-eot --seed S [sampling] [--orchestra NGC --words LIST]` as a child in the tree, its stdout streamed as bytes, its closing stderr line parsed for the numbers |
| `tui/src/live.rs` | the live words of the part (lib/cx-proto.h CX_T_*, CX_C_HEART, the sessions) by two positioned reads every 250 ms |
| `tui/src/run.rs` | the coach's log tail (the loss by report), the checkpoints in `--ckdir` |
| `tui/src/term.rs`, `editor.rs`, `wrap.rs` | the palette and the frame, the line editor, word wrap |

## What a turn is

The prompt of a turn is the system prompt (`~/CLAUDE.md`, read at every turn because a hook restamps its date line
each minute), a blank line, then the transcript's last `--ctx` characters cut at a word boundary: your lines and the
model's pieces as they followed them, with `<|endoftext|>` where a reply ended a story (the BPE's one special token:
tools/bpe.c tokenises the literal back to it, so the model sees the boundary it produced). The client feeds the prompt
one STEP per token, samples the reply on the host (llama.cpp's order; the orchestrator before it when on), and prints
each piece as it is sampled; the TUI shows them as they arrive and keeps the reply as the next turn's context. A reply
ends at `<|endoftext|>` (shown as ∎) or at the token limit. Nothing typed here and nothing the model writes is training
data: the coach feeds the human token stream; the harness only steers it.

The client is never killed: a client that dies after its first command leaves the daemon's session open for good
(card/cortexd.c poll_session abandons a session only before its first command), and the next client cannot be taken.
A turn is at most 2,048 tokens of about 0.4 ms each; quitting waits for the reply in flight.

## The screens

- **chat**: the conversation (`you` / `cortex`, each reply bold with its numbers under it: tokens, seconds, tok/s,
  card and host microseconds a token, prompt tokens, seed, how it ended); the side panel: the card (the daemon's
  heartbeat, the session), the live run (state, chunk, loss in nats and bits, rate, speed, ring, checkpoints), the
  loss by report as bars (from results/live.log, 4 to 8 bits), this chat (the sampler, the reply length, the context,
  the orchestra, the system prompt, the seed, the last reply's numbers); the line to say. Keys: enter, f2 the run, f3
  the orchestra, f4 greedy, pgup/pgdn/up/down scroll, the readline keys, ctrl+q.
- **run**: the live run in full, the checkpoints, the coach's log tail; keys p pause, r resume, s save, m sample, +
  and - the scale (x2, /2 of the value the card reports), esc. The commands are lines appended to the coach's command
  file; without a run feeding the daemon they are refused with the reason.
- **help**: the keys and the `/` commands (`/n /ctx /temp /topk /topp /minp /rp /greedy /orchestra /system /seed
  /retry /clear /run /help /quit`, `/pause /resume /save /sample /scale /lr /every /stop-run`).

## Measured (2026-10-26 19:0x to 19:2x, the 7120A, daemon build 19:14; raw: results/live.log, the tmux captures of the
session's scratchpad)

| what | measured |
|---|---|
| a turn, system prompt on | 68 tokens in 1.41 s, 2,678 tok/s overall, card 371 us, host 120 us a token, prompt 3,716 tokens (`you: Once upon a time`); the second turn 40 tokens in 1.43 s, prompt 3,791 tokens |
| the system prompt alone | ~/CLAUDE.md 9,354 B = 3,710 tokens (2.52 B/token under the TinyStories BPE), fed in 1.390 s = 374.5 us a token: every turn pays 1.39 s before its first generated token (per-turn sessions re-feed the whole prompt) |
| a turn, no system prompt | 23 tokens then `<|endoftext|>` at 1,960 to 2,004 tok/s, card 375 to 376 us, host 143 to 155 us |
| the daemon's heartbeat | CX_C_HEART +5 in 0.5 s (10 Hz, CX_HEART_S 0.1 in thread 0's decide()); the daemon never counted it before build 19:14 |
| the coach steered from the run screen | a paused coach session on ckpt-10021: r (resume) trained 147 chunks = 4,704 tokens in 4 s at 1,281 tok/s with reports at chunks 50/100/150 (6.0730, 5.9666, 6.1616 bits/token); s (save) pushed ckpt-0.bin (23,463,424 B, "1 pushed into the stage"); p (pause), `/scale 0.5` and `/stop-run` reached the coach (its log: "command: pause", "command: scale 0.5000", "command: stop requested", final checkpoint ckpt-160.bin); the live weights restored afterwards with `cortex --load data/live/ckpt-10021.bin` |
| the screens | 16 unit tests: every screen at 40x10, 80x24, 100x30 and 200x60 with every row inside the width and the cursor on screen; the conversation's labels and notes; the prompt composition; the part's offsets; the client's arguments and closing line; the editor; the palette's 256-colour mapping |

Client changes for the harness (host/cortex.c): `--no-echo` (the prompt not printed), `--stop-eot` (the generation
ends at `<|endoftext|>`, printed as its literal, not counted); `--load` now defaults the bus off like `--live` (the
trained configuration; the coach always passed `--bus 0`).

## The frame: tui/cortex.jinja (2026-10-26 19:22, "Get a jinja script written so the model also has prompt coherence and can be prompted as a coherent identity")

`tui/cortex.jinja` is rendered by the harness at every turn (minijinja 2.24, `tui/src/frame.rs`) the way a chat
template is: `messages` (role system | user | assistant, content), `identity` (`name` from `--name`, the template's
own default otherwise), `add_generation_prompt`, `eos_token`. It writes the system message (~/CLAUDE.md), an identity
paragraph ("Cortex is the Intel Phi Cortex: ..."), the dialogue as labelled lines (`Human:` / `Cortex:`), and the
model's label last; it exports `stop` ("\nHuman:"), which the harness passes to the client as `--stop`, so a reply
ends before the next speaker's label (the client prints the pieces before the one completing it; the harness trims a
partial label and records how the reply ended: the story's end, the next speaker's label, or the token limit). The
oldest exchanges are dropped until the dialogue fits `--ctx`. An edit to the template applies to the next turn; a
broken template is an error on the status row, not a crash (`frame::tests`). The model is not chat-tuned: its replies
are story text in the frame's slot (below); the frame fixes the structure of every turn and names the identity, and
the same template serves the chat fine-tune later.

Measured (19:3x, two framed turns through the harness in tmux): "Hello. Who are you?" gave 75 tokens in 1.44 s,
2,706 tok/s overall, card 367 us, host 108 us a token, prompt 3,820 tokens (the system message 3,710, the frame and
the dialogue the rest), ended at the story's end; "Tell me a story about a cat." gave 62 tokens in 1.47 s, 2,711
tok/s, prompt 3,915 tokens, the story's end. Both replies were third-person stories (the model continues as it was
trained; the identity paragraph and the labels do not change that in a 5.9M-parameter story model).

## The training monitor: cortex-watch (2026-10-26 20:09, "Make a monitoring interface for training as a standalone TUI keeping with the harness TUI style... I want to be able to see token/second and the training process visualized in a manner that human legible and comprehensive"; 21:51, "Adapt the training monitor TUI to allow for interaction with the model while training continues to monitor progress and adjust training values in the moment")

`tui/` is one crate with a library and two binaries since 20:15: `cortex-tui` (the chat harness) and `cortex-watch`
(`tui/src/bin/cortex-watch.rs`), the monitor in the same style over the same sources: the part's live words every
250 ms (`tui/src/live.rs`: the run's state, chunk, loss, rate, speed, ring, checkpoints pushed; the daemon's
heartbeat; `LiveState` keeps the words as last read and whether the heartbeat moves) and the coach's log every
second (`tui/src/run.rs` parses the current run, everything after the coach's last opening line: the header with the
stream, the chunks, the schedule and the policy's settings; the reports with loss, rate, speed and ring; the
policy's windows; the samples; the commands, policy decisions and checkpoints as events; the end).

Since 22:10 the monitor talks and steers as well, through `tui/src/chat.rs`, the talk both binaries share (the
turns, the frame, the client runs, the knobs, the lines for the coach). The screen, top to bottom: the header (where
it is; the turn in flight with its seconds, else the run's state and tok/s, or "run ended", or "no daemon"); **the
run** (a progress bar of chunks with the time left at the live rate; state, chunk, tokens, ring, elapsed by the
log's own clock, the start time; the loss in nats and bits with the policy's window mean, its gain over the previous
window and the plateau threshold; the rate in effect as base x scale, halvings of the allowed, checkpoints pushed and
the last one's chunk, the sample cadence); **loss, bits per token**, a column chart with one column per report
(`view::column_chart`: eighth-block resolution, the largest and smallest values and the middle labelled, the last
column lit); **speed, tokens per second**, the same chart; **the last sample** with its coherence numbers and its
text; **events** (the last five); **checkpoints** (the last three with size and age); **talk**, the conversation
with the model on its live weights (your lines; its replies in bold with a note of tokens, seconds, tok/s, card and
host microseconds a token, prompt tokens, seed and how the reply ended; the reply in flight with a spinner), 3 to 10
rows by the terminal's height and kept whatever the height (the sections above are cut first); and the **say
line**. Each chart block costs its height + 4 rows (the rule, the axis, the x axis, a blank); the charts share what
the rest leaves.

A line said runs one `cortex --live` turn exactly as the harness does (framed by `tui/cortex.jinja` over the system
prompt, the stop at the next speaker's label, the client never killed) while the coach keeps feeding: the daemon
serves the client's steps between chunks. A `/command` steers the coach or sets a knob:

| keys | what |
|---|---|
| enter | say the line |
| f2 | `pause` while the run trains, `resume` while it is paused (by the part's state word) |
| f3, f4 | `save`, `sample` |
| f5, f6 | `scale` doubled, halved, from the value the card reports |
| tab | both charts, the loss alone, the speed alone (`/charts`) |
| pgup pgdn ↑ ↓ | scroll the talk |
| f1 | the help screen (`/help`; esc back) |
| ctrl+q | quit; a reply in flight finishes first |
| /pause /resume /save /sample /stop-run, /scale X /lr X /every N | lines for the coach's command file; refused with "not sent" when no run feeds the daemon |
| /n /ctx /temp /topk /topp /minp /rp /greedy /orchestra /system /seed /retry /clear | the turn's knobs, as in the harness |

The monitor takes the harness's options (`--tree`, `--log`, `--ckdir`, `--part`, `--cmd`, `--cortex`,
`--orchestra`, `--words`, `--system`, `--template`, `--name`, `--ngen`, `--ctx`); a tree without the client or the
frame is watched all the same and the say line answers "no turn can run here". `tools/watch.sh` opens it in a
konsole window (`--here` for this terminal).

Checked 20:15 (the read-only version) in tmux (120 x 42) on the chat run's log as it ended: the bar at 18,000 of
18,069 chunks, "run ended, daemon alive", the loss chart over chunks 7,200 to 18,000 between 6.20 and 7.77 bits, the
speed chart between 1,193 and 1,226 tok/s, the last sample, the three halvings and the final checkpoint among the
events; q exits 0.

Checked 22:00 and 22:08 to 22:10 (the say line) in tmux (120 x 42; the session's watch-check.sh and
watch-check2.sh) on the real runs: a turn from the say line as the chat pass had just ended = 60 tokens in 1.44 s,
2,699 tok/s overall, card 369 us and host 111 us a token, prompt 3,824 tokens, the story's end; with no run feeding,
`/save` and `/every 100` were refused ("not sent"). Against the live stories run (1,225 tok/s): `/save` ->
"command: save requested" and data/live/ckpt-2419.bin; `/every 100` -> "command: report every 100 chunks";
`/scale 1` -> "command: scale 1.0000"; f3 -> a second checkpoint (ckpt-2877); f2 -> "command: pause", the part's
state 2, the header "● paused", the keys row offering "f2 resume"; f2 -> "command: resume", state 1; the events
section showed them all; ctrl+q exits 0 with an empty stderr. The same run verified the coach's options of 21:4x:
"checkpoint requested at chunk 5000 (every 5000 chunks)" -> ckpt-5001.bin (a checkpoint's name is the request chunk
+ 1), and "policy: plateau at the floor (scale 1.0000), continuing (no-stop)". Tests (`cargo test --release`, 22
passing): the log parse, the live words' offsets, an unreadable part, the talk's finish reasons, the frame over
system and dialogue, the knobs and the coach lines, every size from 40 x 10 to 200 x 60 with and without a run and
each chart choice, the say line present and the talk shown from 80 x 24, the say line's refusal without a client.

## The chat format and the think block (2026-10-26 22:4x, "Add a reasoning state to the model just like Claude or Qwen models have")

The model trained from scratch (docs/CHAT.md) answers in `<|assistant|><|think|>REASONING<|/think|>ANSWER<|end|>`,
each marker one token. `tui/cortex.jinja` is now that format: an optional `<|system|>` message first (a short
identity line stands in when none is given), `<|user|>` and `<|assistant|>` turns each closed by `<|end|>` and a
newline, then `<|assistant|>` for the reply, with `<|think|><|/think|>` appended when the think mode is off; the
export `stop` is `<|end|>`, so the client ends the reply before the marker. `frame::render` takes `think`;
`chat::split_think` cuts a reply into its thinking and its answer (an open block streams as thinking); the prompt
builder re-feeds earlier replies without their think block; `/think on|off` (f5 in the harness) sets the mode and
the side panel shows it; `--vocab FILE` passes the chat vocabulary to the client (the side panel names it). The
conversation draws the thinking faint and italic under "think" and the answer bold under "cortex"; a reply's note
says "the end marker" when it stopped at `<|end|>`.

Checked 22:5x in tmux (120 x 42; the session's think-check.sh and think-check2.sh) against the run from scratch,
fifteen minutes old: the client encodes `<|user|>Hi<|end|>\n<|assistant|>` as 6 tokens (the four markers, "Hi",
the newline); a turn from the say line with the CLAUDE.md system prompt = 103 tokens in 1.07 s at 2,672 tok/s
overall (prompt 2,769 tokens, card 374 us and host 11 us a token), ending at the end marker; with the stories'
orchestra forced on, 38 tokens in 1.06 s (host 141 us a token). A first attempt at 22:47 produced nothing for 16 s
and did not reproduce afterwards (both variants under 1.1 s); the one difference between the attempts was the page
cache, which the data work had just churned, with the harness loading the 470 MB stories' count model and the 23 MB
word list from the disk per turn. Those tables are indexed by the stories' token ids in any case, so a `--vocab`
of its own now switches the default orchestra off (`chat::Config::check`; `--orchestra NGC` keeps one on purpose).
Tests: 22 in the library, 2 in the monitor.

## The stops (2026-10-27 01:3x)

A reply ends at the frame's end marker or at any role marker: `tui/cortex.jinja` exports `stop` as the list
`["<|end|>", "<|user|>", "<|system|>", "<|assistant|>"]` (a string still works), each goes to the client as its own
`--stop` (up to eight), the client ends the turn at whichever matches and reports `cortex: stopped at STRING` on its
stderr, and the reply's footer says "the end marker" or "stopped at <|user|>". Before this the only stop was
`<|end|>`, and a `<|user|>` the model emitted mid-answer was printed as text (seen in the harness on 2026-10-27
01:2x against the from-scratch chat model, docs/CHAT.md "Incoherent, and why"). The partial-label trim stays for
multi-token stops such as the stories' `\nHuman:`; the chat markers are single tokens, so nothing of them is ever
printed and the client's line is what names them.
Verified on the desktop at 01:34 against the live from-scratch weights: a think-off reply of 114 tokens ended with
the footer "stopped at <|user|>", the marker cut (docs/CHAT.md has the transcript).

## The pool and the memory in the harness and the monitor (2026-10-27 07:5x)

Both programs talk to the daemon, which routes every step of a reply through the expert pool once a MoE model is
loaded (LOAD arg bit 1), so a reply on a MoE run is the routed model's without any option. What they show of it
comes from the part (live.rs reads through CX_T_POOL_ACK now): the harness's side panel and run screen and the
monitor's run block add, while the loaded model routes, a `moe` line (the home bonus the daemon routes with and the
experts' step: CX_M_BIAS, CX_M_LR), an `experts` line (updated at least once since the pool was filled, of 61,440:
CX_T_MOE_DISTINCT) and, while a pool transfer is under way, a `pool` line (push or pull, the round of 187: CX_T_POOL
against CX_T_POOL_ACK); a `memory` line says the n-gram memory trains with the run (CX_T_NG_ACK). The log reader
(run.rs) takes the report line's suffix ("moe 120 experts, bias 90.9") and the opening line's notes ("n-gram memory
FILE (...)", "MoE loaded (...)"), shows "pool:" lines among the events, and lists the pool-N.f32 and tables-N.f32
files beside the checkpoints. The monitor's `moe` line falls back to the last report's words when the run is not
live, and ends with the opening line's note.
