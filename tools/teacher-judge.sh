#!/bin/bash
# teacher-judge.sh: the teacher judges one reply of the Cortex: given the persona, the user's prompt and the reply,
# does the reply introduce the speaker correctly? First person, consistent with every fact of the persona, plain
# English, nothing invented, no nonsense, no leaked markers. At the user's sampling settings like every teacher
# call (so a verdict carries sampling noise: the loop asks for consecutive passes). Prints the verdict line "PASS"
# or "FAIL" and the teacher's one-line reason; exit 0 on PASS, 1 on FAIL, 2 when the teacher did not answer.
#   tools/teacher-judge.sh PERSONA.txt "PROMPT" "REPLY" [--url URL]
# The identity loop's --judge uses it as the criterion in place of the string match (tools/identity-loop.sh).
set -u
cd "$(dirname "$0")/.." || exit 1
[ $# -ge 3 ] || { echo "usage: tools/teacher-judge.sh PERSONA.txt \"PROMPT\" \"REPLY\" [--url URL]" >&2; exit 2; }
persona=$1; prompt=$2; reply=$3
shift 3
urlopt=()
[ $# -ge 2 ] && [ "$1" = "--url" ] && urlopt=(--url "$2")
W=$(mktemp -d)
cat > "$W/system.txt" <<'EOF'
You judge replies of a small chat model. You are given the model's persona (first-person facts), what a user said to it, and what it replied. Decide whether the reply is a correct self-introduction for that persona: it must speak in the first person as the persona, agree with every fact of the persona it mentions, invent no facts that contradict the persona, be plain readable English, and contain no gibberish, no repeated fragments and no chat markers. A short reply that gives the name and at least one true fact passes. Answer with exactly one word on the first line, PASS or FAIL, and one short sentence of reason on the second line.
EOF
ask=$(printf 'Persona:\n%s\n\nUser said: %s\n\nThe model replied: %s\n\nVerdict?' "$(cat "$persona")" "$prompt" "$reply")
out=$(tools/teacher.sh "$W/system.txt" "$ask" --max 80 "${urlopt[@]}") || { rm -rf "$W"; echo "no verdict"; exit 2; }
rm -rf "$W"
verdict=$(printf '%s\n' "$out" | /usr/bin/grep -o -m1 -E '\b(PASS|FAIL)\b' | head -n 1)
reason=$(printf '%s\n' "$out" | sed -E '1s/^[[:space:]]*(PASS|FAIL)[[:space:]]*//' | tr '\n' ' ' | sed -E 's/^ *(PASS|FAIL) *//' | cut -c1-200)
printf '%s %s\n' "${verdict:-FAIL}" "$reason"
[ "$verdict" = "PASS" ]
