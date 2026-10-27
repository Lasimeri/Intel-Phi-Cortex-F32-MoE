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
