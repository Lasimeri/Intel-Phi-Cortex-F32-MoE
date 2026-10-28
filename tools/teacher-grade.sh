#!/bin/bash
# teacher-grade.sh: the teacher grades one reply of the student and rewrites it, the shape of ILF (Scheurer et al. 2023),
# LeMa (An et al. 2023) and CITING (Feng et al. 2023): the student's attempt is critiqued and revised, and the revision is
# what the student later trains on (docs/TEACHER.md "Methods from the literature"); the grades are TinyStories' "as a
# teacher grades a student" (Eldan & Li 2023), three dimensions 0 to 10. Input: the persona, the conversation so far as
# plain lines ("person: ..." and "speaker: ..."), the student's reply. Output on stdout: one JSON object
# {"grammar": G, "relevance": R, "persona": P, "why": "...", "rewrite": "..."}. Exit 1 when the server fails, 2 when the
# teacher's answer holds no usable rewrite or grades. The call goes through tools/teacher.sh (CLAUDE.md as the system
# prompt, the user's sampling, the repeat penalty at 1.0 for the JSON answer).
#   tools/teacher-grade.sh PERSONA.txt CONTEXT.txt "REPLY" [--history FILE] [--url URL]
# --history FILE: the student's record (2026-10-28 04:55, user: the teacher should "remember how the student model
# responds instead of producing mindless responses in relation to whatever the student model says"): its earlier
# replies, oldest first, one a line, with the person's line each answered, the grades and the teacher's rewrite
# (tools/teacher-correct.sh keeps it in DIR/student-record.txt). Without it every call is a fresh request that sees
# one reply and nothing of the student. The record stands at the end of the system message, after the rubric and
# before the item, so the calls of a pass share its prefix and the server's prompt cache reads it once (the server
# holds 64K tokens since 05:xx, tools/teacher-serve.sh).
set -u
cd "$(dirname "$0")/.." || exit 1
[ $# -ge 3 ] || { echo "usage: tools/teacher-grade.sh PERSONA.txt CONTEXT.txt \"REPLY\" [--history FILE] [--url URL]" >&2; exit 2; }
persona=$1; ctx=$2; reply=$3
shift 3
urlopt=(); history=""
while [ $# -gt 0 ]; do
	case "$1" in
	--url) urlopt=(--url "$2"); shift 2 ;;
	--history) history=$2; shift 2 ;;
	*) echo "teacher-grade: unknown option $1" >&2; exit 2 ;;
	esac
done
[ -f "$persona" ] && [ -f "$ctx" ] || { echo "teacher-grade: persona or context file missing" >&2; exit 2; }
[ -n "$reply" ] || reply="(the model said nothing)"
task=$(mktemp "${TMPDIR:-/tmp}/teacher-grade.XXXXXX") || exit 1
trap 'rm -f "$task"' EXIT
cat > "$task" <<'EOF'
You grade and correct a very small chat model that speaks in the first person as the persona you are given. You see the conversation so far (the person and the speaker, which is the model) and the model's latest reply to the person's last line. Answer with a JSON object only: {"grammar": G, "relevance": R, "persona": P, "why": "...", "rewrite": "..."}. grammar, 0 to 10: is the reply fluent, well-formed, simple English (0 is gibberish, 10 is flawless). relevance, 0 to 10: does it answer the person's last line and fit the conversation (0 ignores it, 10 answers it well). persona, 0 to 10: does it speak as one single self consistent with the persona, saying persona facts only when the person asked about the speaker, never as an assistant or an AI, never about parts, heads, columns or siblings inside itself, never inventing facts about itself (10 is fully in character). why: one short sentence naming the main fault, or "good" when there is none. rewrite: what the speaker should have said at that point, one or two short plain sentences a child reads easily, in the persona's voice, keeping whatever was right in the model's reply; it never mentions the model, the grades or a mistake, and it never uses quotation marks.
EOF
if [ -n "$history" ] && [ -s "$history" ]; then
	cat >> "$task" <<EOF

This student's record so far, oldest first: each line is one of its earlier replies with the person's line it answered, your grades (grammar relevance persona) and your rewrite. Use it as a teacher uses what a pupil did before: when the latest reply repeats a mistake the student keeps making (a phrase it tacks onto every answer, an answer that ignores the question, yes to everything, a wrong fact it repeats), name that in why, and make the rewrite the right answer to the person's last line that shows exactly what this student keeps getting wrong. Grade the latest reply on its own merits.

$(cat "$history")
EOF
fi
out=$(tools/teacher.sh "$task" "The persona:
$(cat "$persona")

The conversation so far:
$(cat "$ctx")

The model's reply: $reply" --max 300 --rp 1.0 --json "${urlopt[@]}") || exit 1
line=$(printf '%s' "$out" | tr -d '\n' | sed -E 's/^[^{]*//; s/[^}]*$//')
printf '%s\n' "$line" | jq -c 'select(.rewrite and (.rewrite | tostring | length) > 0) | { grammar: ((.grammar // 0) | tonumber), relevance: ((.relevance // 0) | tonumber), persona: ((.persona // 0) | tonumber), why: ((.why // "") | tostring | gsub("\\s+"; " ")), rewrite: (.rewrite | tostring | gsub("\\s+"; " ") | ltrimstr(" ") | rtrimstr(" ")) }' 2>/dev/null | /usr/bin/grep -q . || { echo "teacher-grade: unusable answer: $(printf '%s' "$out" | cut -c1-200)" >&2; exit 2; }
printf '%s\n' "$line" | jq -c 'select(.rewrite and (.rewrite | tostring | length) > 0) | { grammar: ((.grammar // 0) | tonumber), relevance: ((.relevance // 0) | tonumber), persona: ((.persona // 0) | tonumber), why: ((.why // "") | tostring | gsub("\\s+"; " ")), rewrite: (.rewrite | tostring | gsub("\\s+"; " ") | ltrimstr(" ") | rtrimstr(" ")) }'
