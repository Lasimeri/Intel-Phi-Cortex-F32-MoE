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
