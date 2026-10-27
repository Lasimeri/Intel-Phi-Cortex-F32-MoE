# The teacher: Qwen3.8-27B teaching the Cortex (2026-10-27 10:21 on)

The user (10:21): "Write a setup for utilizing Qwen3.8-35B @ Q4_K_M (or 27B @ Q4_K_XL (preferable this one)) in
order to teach the model that's [being trained]"; (10:26) the 35B at Q6_K if it runs on the 3090 Ti completely;
(10:26) "The teacher model should definitely use my CLAUDE.md as its system prompt as well"; (10:28) "my preferred
inference settings that should be the same as my training settings"; (10:30) "modify the training interface to show
the interaction between the two models". The teacher is a large model that writes the Cortex's identity set, judges
the Cortex's replies in the identity loop, and (the next stage) supplies soft targets. The student is the Cortex on
the 7120A (docs/LIVE.md), which cannot copy from its context (no attention): what it learns of an identity is what
its weights memorise, so the teacher's job is to make that memorisation rich and to judge it.

## The teacher model and where it runs

| candidate | file | weights | on the desktop's RTX 3090 Ti (24,564 MiB, 3,849 MiB held by the desktop) |
|---|---|---|---|
| Qwen3.8-27B UD-Q4_K_XL (dense) | ~/models/Qwen3.8-27B/Qwen3.8-27B-UD-Q4_K_XL.gguf | 17,559,178,144 B | every layer in VRAM: 21,462 MiB used with an 8,192-token context; THE TEACHER |
| Qwen3.8-35B-A3B Q6_K (MoE, 3B active) | ~/models/Qwen3.8-35B-A3B/Qwen3.8-35B-A3B-Q6_K.gguf | 29,208,731,296 B | 27.2 GiB of weights: does not fit the card |
| Qwen3.8-35B-A3B Q4_K_M | ~/models/Qwen3.8-35B-A3B/Qwen3.8-35B-A3B-Q4_K_M.gguf | 21,713,462,944 B | 20.2 GiB of weights against 20.2 GiB free: no room for the cache; part would run on the CPU and the 3B-active speed advantage goes with it |

The rack's four 3080s hold Strata's Q8 Flash-Next for the user's harnesses; the teacher stays on the desktop's GPU,
bound to 127.0.0.1 only (the SSH-only rule is about reaching the rack; this is local).

`tools/teacher-serve.sh [--mtp] [--ctx N] [--port P] | --status | --stop`: `~/llama.cpp/build/bin/llama-server`
(the CUDA build) `-m <the 27B> -ngl 99 -c 8192 -fa on --port 8090 --host 127.0.0.1 --jinja --temp 0.5 --top-k 75
--top-p 0.95 --min-p 0.05 --repeat-penalty 1.2 --log-file /tmp/cortex-teacher.log`, detached; `--mtp` adds the
MTP draft beside the model (`--spec-type draft-mtp -md ~/models/Qwen3.8-27B/MTP/mtp-Qwen3.8-27B-Q4_0.gguf`,
untested). Measured 10:28: up in 40 s, 21,462 MiB of VRAM in use.

## One turn: tools/teacher.sh

`tools/teacher.sh SYSTEM_FILE|- "USER" [--temp T] [--max N] [--think] [--json] [--url URL] [--no-claude-md]` posts
one chat completion (`/v1/chat/completions`) and prints the reply. The system message is the user's `~/CLAUDE.md`
(every rule of it, the verbal register included) followed by the task text of SYSTEM_FILE; every request carries
the user's sampling settings (temp 0.5, top-k 75, top-p 0.95, min-p 0.05, repeat penalty 1.2: the block the
Cortex trains and samples with), thinking off (`chat_template_kwargs.enable_thinking false`), `--json` a JSON
object by grammar (`response_format json_object`), and the response delimiters CLAUDE.md asks for are stripped on
the way back. Measured: a 60-token turn 3.0 s with the 2,200-token system prompt (its prefix cached by the
server after the first call), a verdict 1.3 to 2.1 s.

## The teacher writes the identity set: tools/teacher-identity.sh

`tools/teacher-identity.sh PERSONA.txt N OUT.jsonl [--per-call K] [--temp T] [--url URL]`: per call the teacher
writes K exchanges as JSON (a user line asking the speaker to introduce itself or who or what it is, in varied
everyday words; the speaker's answer in the first person from the persona, one to three short sentences of simple
English, nothing invented), the answers that step out of the role are dropped (a regexp on "persona", "the user",
"as an AI"), exact duplicates too, until N distinct exchanges exist; OUT.jsonl in the messages shape with the
persona as the system message, the prompts alone in OUT.prompts, the answers alone in OUT.refs. Measured 10:33: 32
exchanges from data/chat-base2/persona.txt in 95.2 s (6 calls); one factual slip in the set ("I count one hundred
and two columns.") shows the teacher's sampling at temp 0.5 is not exact. The teacher's text is model output and
the loop trains on it by the user's decision of 10:21 (the Cortex's own outputs are still never trained on).

## The teacher judges the loop: tools/teacher-judge.sh and identity-loop.sh --judge

`tools/teacher-judge.sh PERSONA.txt "PROMPT" "REPLY"` prints PASS or FAIL with the teacher's one-line reason and
exits 0 on PASS. `tools/identity-loop.sh ... --judge` (REF.txt "-") uses it as the criterion in place of the string
match: every EVERY chunks the greedy reply of the live weights goes to the teacher, `--stable K` consecutive
passes stop the run, the final weights are re-tested. Measured: the first rubric ("the name and at least one true
fact") passed the memorised PersonaChat reply and failed gibberish correctly (10:33), and then passed "That sounds
like a good idea, but I am Cortex." (10:37): a name bolted onto base-chat filler. The rubric was tightened at 13:0x:
the name and at least two more facts of the persona, nothing contradicting or invented, no filler from another
conversation.

## The first teacher loop (10:35 to 10:38)

`tools/identity-loop.sh data/identity/teacher-identity.jsonl data/chat-base2/persona.txt "Introduce yourself" -
data/live-moe2/ckpt-15001.bin --judge --base-start 22685760 --ratio 3,2,1 --chunks 20000 --every 500 --max-rounds
3 --stable 2 --ckdir data/live-identity-teacher --tables /dev/shm/cortex-identity-teacher.f32`: the set (32
exchanges, 2,657 tokens) recurring every three base conversations; verdicts FAIL at chunk 600 ("It is hard to see
how...") and 1,000 ("That sounds like a good idea, but I have a small language model on..."), PASS at 1,500 ("That
is a good idea, I am Cortex.") and 2,000 ("That sounds like a good idea, but I am Cortex."), stopped at 2,074, the
re-test PASS; the state data/live-identity-teacher/ckpt-2074.bin + pool-2074.f32 + the gates and tables. 810
tok/s. The loop did what it was told; the rubric was the weak part.

## The next stage: soft targets (not built)

Token-level distillation across the two vocabularies: the teacher scores the training stream through
llama-server's `/completion` with `n_probs` (its top-k next tokens per position as strings and probabilities), each
string is mapped to the Cortex's 4,096-id BPE (the first Cortex token of the string, probabilities summed), the
result is a sparse target per position, served to the card per chunk through a ring like the memory's (docs/LIVE.md)
and added to the card's loss as a KL term beside the cross-entropy. Arithmetic, not measured: 28.5M tokens at the
27B's prompt-processing rate on the 3090 Ti is hours of teacher time for one pass, about 1.4 GB of targets at top-8.
What it cannot do: lift the student above its architecture's ceiling (PLAN-MOE.md "09:23 to 09:31").

## From scratch on the teacher's corpus alone (13:09, user: "Train a model from scratch just using the teacher model")

Two processes, the GPU and the card in parallel:

| process | what it does |
|---|---|
| `tools/teacher-corpus.sh PERSONA.txt OUT_DIR [--topics 200] [--per-call 3] [--exchanges 5] [--parallel P] [--max M]` | the teacher lists the topics once by themes (OUT_DIR/topics.txt: ten themes of twenty everyday things; the first list, asked as "more topics unlike these", degenerated into variants of one phrase), then P writers cycle through them, each call three conversations of five exchanges about one topic as free text holding one JSON object (the person in plain everyday English, the speaker in the first person as the persona, about itself only when asked, one or two short sentences a turn), under a cap of 2,400 tokens with the repeat penalty at 1.0 for these calls alone; the reply's complete conversation objects are salvaged one by one (tools/json-objects.c: every complete object at nesting level 3, strings and escapes honoured; in about half the calls the teacher writes its closing delimiter in place of the object's final brace), every call lands as OUT_DIR/batch-NNN-wN.jsonl in the messages shape, a conversation with a turn over 220 characters or an answer that steps out of the role is dropped, the raw text of a call that yields nothing is kept (the last ten, OUT_DIR/failed-*.txt); OUT_DIR/corpus.log a line per batch; OUT_DIR/stop ends it. The server: one slot of 8,192 with the MTP draft (`tools/teacher-serve.sh --np 1 --mtp`, 22,505 MiB) |
| `tools/teacher-scratch.sh CORPUS_DIR PERSONA.txt [--ckdir data/live-teacher] [--tables FILE] [--vocab-bytes 1000000] [--round-chunks 2000] [--lr 0.003]` | once a megabyte of converted text exists, trains the 4,096-id BPE on it and fixes it (DIR/vocab-teacher4096.txt); converts and encodes every new batch into DIR/train.u16 (every 20th into valid.u16); whenever 2,000 untrained chunks exist runs a coach round from the last checkpoint (fresh weights from tools/init-ckpt first) at --start of the first untrained chunk, home routing, the memory on with its gates continued; after each round the held-out loss (ref-loss, the dense reading, 5,000 tokens) and the judged greedy "Introduce yourself" go to results/identity-loop.log in the loop's line shape, so the monitor's "the teacher and the student" section shows them; DIR/state.txt carries the round, the trained chunk count and the state |

The rate is the teacher's: 37.5 tok/s on one stream (the server's log, 10:3x), two slots and the draft measured
below as the batches land. The previous loop (the strict judge from checkpoint 2,074) was stopped for the card.

**One self (13:17, user: "The MoE heads as a group need to form a coherent, single identity structure, rather than
questioning how many siblings/other MoE heads there are").** The persona's line "I have one hundred and twenty
columns" (Claude's own line from 02:0x) had seeded exchanges about the model's parts ("Do you have many columns?",
"How big are you inside?"). The persona is now three lines (I am Cortex. I am a small language model on a Xeon
Phi card. I like to talk with people.), the writer's and the judge's texts ask for one single self that never
speaks of parts, heads, columns or siblings inside itself, the frame's default identity lost the line, and the
batches written under the old persona were set aside (data/teacher/old-persona/). The 120 columns vote as one
consensus and produce one distribution: the data must describe one self.

## The teacher's throughput on this card (13:2x to 13:3x)

The RTX 3090 Ti is power-capped at 200 W (the user's UPS budget; nvidia-smi: SW Power Cap active, the SM clock at
720 MHz of 2,115 under a sustained generation, 99 % busy, P0), and the desktop's own processes share it (the face
tracker, mpv, ffmpeg, the compositor). The 37.5 tok/s of 10:3x was a 37-token burst at boost clocks before the cap
engaged. Measured on the corpus writer's own call (CLAUDE.md in the system prompt, three conversations of five
exchanges as JSON; the server's `eval time` lines, the scratchpad's gen-*.json):

| configuration | per stream | together | note |
|---|---|---|---|
| 2 slots + the MTP draft, two calls at once | 10.9 and 12.2 tok/s | about 23 | the draft accepted 0.62 to 0.64 of its tokens, mean run 2.9 |
| 2 slots, no draft, two calls at once | 11.3 and 13.2 tok/s | about 24 | |
| 1 slot, no draft | 13.9 tok/s | 13.9 | |
| 1 slot + the MTP draft | 16.34 tok/s on 1,678 tokens | 16.3 | the draft accepted 0.40, mean run 2.2; THE CONFIGURATION: `tools/teacher-serve.sh --np 1 --mtp` |

Two slots lose: per stream the rate halves and the pair gains nothing over one stream; the draft on one slot
gains 18 % on a long generation.

**Why the first corpus calls yielded nothing (13:3x to 13:4x, the scratchpad's probe and direct JSON files):**
under the JSON grammar with the repeat penalty of 1.2 the tokenisation bloated to about 3.7 bytes a token (a
five-exchange conversation about 900 tokens, three of them 2,400) and every reply hit the cap mid-string; the
direct request's finish_reason was "length" with the content drifting into a word list ("tool implement device
gadget contraption mechanism engine..."), the penalty leaving the model no permitted continuation inside a string.
With the penalty at 1.0 the drift stopped; with the grammar off and the draft on, a call wrote five conversations
in 779 tokens (about 156 a conversation) at 32 tok/s. The failures that remained (7 of the first 14 calls at
13:4x) were not cut replies: every one held its three whole conversations and lacked only the top-level object's
final brace, the teacher writing its closing delimiter in its place (a direct call shows the other half ending
in the brace, then the delimiter; 393 completion tokens for three conversations). The writer now salvages the
complete conversation objects one by one (tools/json-objects.c) and runs the filter on each; since 13:55 every
call yields 3 of 3: 27 conversations in 9 calls over 132 s, 12.3 a minute, about 7.3 KB of converted text a
minute (the driver's count: 23,793 bytes for 40 conversations). The user's temperature, top-k, top-p and
min-p stay on every call; the repeat penalty is the one setting the measurements overruled for the writer, and
the judge keeps 1.2 on its short verdicts. Two more findings on the
same calls: a 4,096-token slot (two slots in 8,192) cut the replies mid-object, since the system prompt is about
2,200 tokens and a reply 1,000 to 1,700 (the server script's default is now 8,192 a slot, one slot); and under the
repeat penalty of 1.2 with the JSON grammar one reply drifted into a word list inside a string for 11 KB, so the
writer caps a reply at 1,600 tokens and drops a conversation with a turn over 220 characters. Arithmetic at 14
tok/s: about 50,000 tokens an hour, the first megabyte of text (about 250,000 tokens) in four to five hours; the
from-scratch model's first round starts then.
