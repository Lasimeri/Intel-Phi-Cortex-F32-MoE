#!/bin/bash
# teacher.sh: one turn of the teacher, the Qwen3.8-27B UD-Q4_K_XL served by llama.cpp on this machine's GPU
# (tools/teacher-serve.sh; docs/TEACHER.md), through its OpenAI-compatible route on 127.0.0.1 only. The system
# message is the user's ~/CLAUDE.md (2026-10-27 10:26: "The teacher model should definitely use my CLAUDE.md as
# its system prompt as well") followed by SYSTEM_FILE's text; the user message; the reply's text on stdout with
# the response delimiters CLAUDE.md asks for stripped; thinking off unless --think. Every request carries the
# user's inference settings, the same block the Cortex trains and samples with (10:28: "my preferred inference
# settings that should be the same as my training settings"): temp 0.5, top-k 75, top-p 0.95, min-p 0.05, repeat
# penalty 1.2; --temp overrides the temperature alone.
#   tools/teacher.sh SYSTEM_FILE|- "USER" [--temp T] [--rp R] [--max N] [--think] [--json] [--url URL] [--no-claude-md]
#                    [--lp-out FILE [--lp-top K]]
# --rp R overrides the repeat penalty alone: the corpus writer passes 1.0, since 1.2 drove every long JSON
# generation measured on 2026-10-27 13:3x into a word list inside a string up to the token cap.
# --lp-out FILE asks for the teacher's own distribution at every generated token (logprobs: true, top_logprobs K,
# default 8) and writes the server's per-token list (token text, logprob, the K alternatives) to FILE as one JSON
# array: the raw material of the partial distillation (docs/DISTIL.md), the writer keeps it beside each batch.
# Never ask this build for the prompt's logprobs (echo + logprobs on /v1/completions killed the server on
# 2026-10-27 18:47).
# SYSTEM_FILE "-" adds no task text after CLAUDE.md. --json asks for a JSON object (response_format json_object:
# the server's grammar then forbids the delimiters). The default URL is http://127.0.0.1:8090 (env CORTEX_TEACHER
# overrides); CORTEX_CLAUDE_MD names another rules file. Exit 1 when the server does not answer or the reply is
# empty; the error on stderr.
set -u
cd "$(dirname "$0")/.." || exit 1
[ $# -ge 2 ] || { echo "usage: tools/teacher.sh SYSTEM_FILE|- \"USER\" [--temp T] [--rp R] [--max N] [--think] [--json] [--url URL] [--no-claude-md] [--lp-out FILE [--lp-top K]]" >&2; exit 2; }
sysfile=$1; user=$2
shift 2
temp=0.5; rp=1.2; max=512; think=false; json=false; url=${CORTEX_TEACHER:-http://127.0.0.1:8090}; rules=${CORTEX_CLAUDE_MD:-$HOME/CLAUDE.md}; lpout=""; lptop=8
while [ $# -gt 0 ]; do
	case "$1" in
	--temp) temp=$2; shift 2 ;;
	--rp) rp=$2; shift 2 ;;
	--max) max=$2; shift 2 ;;
	--think) think=true; shift ;;
	--json) json=true; shift ;;
	--url) url=$2; shift 2 ;;
	--no-claude-md) rules=""; shift ;;
	--lp-out) lpout=$2; shift 2 ;;
	--lp-top) lptop=$2; shift 2 ;;
	*) echo "teacher: unknown option $1" >&2; exit 2 ;;
	esac
done
system=""
[ -n "$rules" ] && [ -f "$rules" ] && system=$(cat "$rules")
if [ "$sysfile" != "-" ]; then
	task=$(cat "$sysfile") || exit 1
	system="${system:+$system

---

}$task"
fi
lp=false; [ -n "$lpout" ] && lp=true
body=$(jq -n --arg s "$system" --arg u "$user" --argjson t "$temp" --argjson rp "$rp" --argjson m "$max" --argjson think "$think" --argjson json "$json" --argjson lp "$lp" --argjson lptop "$lptop" '
	{ model: "teacher", temperature: $t, top_k: 75, top_p: 0.95, min_p: 0.05, repeat_penalty: $rp, max_tokens: $m, chat_template_kwargs: { enable_thinking: $think } }
	+ (if $json then { response_format: { type: "json_object" } } else {} end)
	+ (if $lp then { logprobs: true, top_logprobs: $lptop } else {} end)
	+ { messages: ((if $s == "" then [] else [{ role: "system", content: $s }] end) + [{ role: "user", content: $u }]) }')
out=$(curl -s --max-time 600 -H 'Content-Type: application/json' --data-binary "$body" "$url/v1/chat/completions") || { echo "teacher: no answer from $url" >&2; exit 1; }
reply=$(printf '%s' "$out" | jq -r '.choices[0].message.content // empty' 2>/dev/null)
[ -n "$reply" ] || { echo "teacher: empty reply: $(printf '%s' "$out" | cut -c1-300)" >&2; exit 1; }
if [ -n "$lpout" ]; then printf '%s' "$out" | jq -c '.choices[0].logprobs.content // []' > "$lpout" 2>/dev/null || : > "$lpout"; fi
# a think block, when thinking was on and the server left it inline, is dropped; so are the response delimiters
printf '%s\n' "$reply" | sed -E ':a;N;$!ba;s/<think>.*<\/think>[[:space:]]*//' | sed -E 's/^[[:space:]]*<•••>[[:space:]]*//; s/[[:space:]]*<\/•••>[[:space:]]*$//' | sed -E '/^[[:space:]]*<\/?•••>[[:space:]]*$/d'
