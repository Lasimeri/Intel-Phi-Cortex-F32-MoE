#!/bin/bash
# teacher-identity.sh: the teacher writes the identity set for the identity loop: N exchanges in which a user asks
# the speaker, in varied plain words, to introduce itself or who or what it is, and the speaker answers in the
# first person from the persona, one to three short sentences of simple English (the chat base's register), every
# fact true to the persona and nothing added. The teacher writes both sides; the persona is yours (PERSONA.txt).
# Output: OUT.jsonl in the messages shape the loop reads (the persona as the system message, the user's prompt, the
# answer), exact duplicates dropped; the user prompts alone in OUT.prompts and the answers alone in OUT.refs (the
# loop's REF.txt: every answer is an accepted introduction).
#   tools/teacher-identity.sh PERSONA.txt N OUT.jsonl [--per-call K] [--temp T] [--url URL]
# Each call asks for K exchanges (default 10) as a JSON object {"exchanges":[{"user":...,"assistant":...}]} at
# the user's sampling settings (--temp overrides the temperature alone); calls repeat until N distinct exchanges
# exist or 3 calls in a row add nothing. The teacher's text is model output: the loop trains on it by the user's
# decision of 2026-10-27 10:21.
set -u
cd "$(dirname "$0")/.." || exit 1
[ $# -ge 3 ] || { echo "usage: tools/teacher-identity.sh PERSONA.txt N OUT.jsonl [--per-call K] [--temp T] [--url URL]" >&2; exit 2; }
persona=$1; want=$2; out=$3
shift 3
per=10; tempopt=(); urlopt=()
while [ $# -gt 0 ]; do
	case "$1" in
	--per-call) per=$2; shift 2 ;;
	--temp) tempopt=(--temp "$2"); shift 2 ;;
	--url) urlopt=(--url "$2"); shift 2 ;;
	*) echo "teacher-identity: unknown option $1" >&2; exit 2 ;;
	esac
done
W=$(mktemp -d)
cat > "$W/system.txt" <<'EOF'
You write training dialogue for a very small chat model that speaks in the first person as the persona you are given. Write short, plain, simple English, the kind of sentences a child reads easily: no rare words, no idioms, no lists, no markdown, no quotation marks inside the text. The user's line asks the speaker to introduce itself, or who it is, or what it is, or what it likes, each time in different everyday words. The speaker's answer is one to three short sentences in the first person, every fact taken from the persona and nothing invented beyond it; it is one single self and never speaks of parts, heads, columns or siblings inside itself; it never calls itself an assistant or an AI unless the persona says so. Answer with a JSON object only: {"exchanges": [{"user": "...", "assistant": "..."}, ...]}.
EOF
: > "$W/all.jsonl"
have=0; dry=0; call=0
while [ "$have" -lt "$want" ] && [ "$dry" -lt 3 ]; do
	call=$((call + 1))
	ask=$(printf 'The persona:\n%s\n\nWrite %d different exchanges. Vary the user'"'"'s wording every time (greetings, direct questions, curious questions, short and long). Number of exchanges: %d.' "$(cat "$persona")" "$per" "$per")
	if ! tools/teacher.sh "$W/system.txt" "$ask" "${tempopt[@]}" --max 2048 --json "${urlopt[@]}" > "$W/reply.json" 2> "$W/err"; then
		echo "teacher-identity: call $call failed: $(cat "$W/err" | cut -c1-200)" >&2
		dry=$((dry + 1))
		continue
	fi
	# an answer that talks about "the persona" or "the user" is the teacher stepping out of the role: dropped
	jq -c '.exchanges[]? | select(.user and .assistant) | { user: (.user | tostring | gsub("\\s+"; " ") | ltrimstr(" ") | rtrimstr(" ")), assistant: (.assistant | tostring | gsub("\\s+"; " ") | ltrimstr(" ") | rtrimstr(" ")) } | select(.user != "" and .assistant != "") | select(.assistant | test("persona|the user|as an ai|language model called"; "i") | not)' "$W/reply.json" 2>/dev/null >> "$W/all.jsonl"
	n=$(sort -u "$W/all.jsonl" | wc -l)
	[ "$n" -gt "$have" ] && dry=0 || dry=$((dry + 1))
	have=$n
	echo "teacher-identity: call $call: $have distinct exchanges" >&2
done
sort -u "$W/all.jsonl" | head -n "$want" > "$W/kept.jsonl"
jq -c --rawfile p "$persona" '{ messages: [ { role: "system", content: ($p | rtrimstr("\n")) }, { role: "user", content: .user }, { role: "assistant", content: .assistant } ] }' "$W/kept.jsonl" > "$out"
jq -r '.user' "$W/kept.jsonl" > "${out%.jsonl}.prompts"
jq -r '.assistant' "$W/kept.jsonl" > "${out%.jsonl}.refs"
echo "teacher-identity: $(wc -l < "$out") exchanges in $out ($call calls); prompts in ${out%.jsonl}.prompts, answers in ${out%.jsonl}.refs"
rm -rf "$W"
