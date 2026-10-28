#!/bin/bash
# teacher-probe.sh: the reasoning probe, the student's score on a fixed set of questions with known answers (2026-10-27
# 21:36, user: "keep going with refining the student model's coherence and reasoning capacity"). Each question is put
# to the live weights the way the harness puts a turn (the persona as the system line, the question as the person's
# line, the user's sampling settings, a seed fixed by the round and the question's index so a round's score is
# repeatable), and the teacher judges the reply against the expected answer: correct when the reply gives the
# expected answer in substance (the words may differ), incorrect when it gives another answer, evades, contradicts
# the persona or is incoherent. Writes to LOG one line per question in the loop's interaction shape, so the
# monitor's "the teacher and the student" section shows them:
#   HH:MM:SS round R chunk C: PASS|FAIL probe Q: <question> | expected: <answer> | <why> (matches in a row 0) | <reply>
# and one summary line:
#   HH:MM:SS round R probe: K of N correct (<the questions missed, by number>)
# Prints the summary line; exit 0.
#   tools/teacher-probe.sh QUESTIONS.jsonl PERSONA.txt VOCAB ROUND CHUNK [--tables FILE] [--ngen 40] [--log FILE] [--url URL]
# QUESTIONS.jsonl: one {"q": "...", "a": "..."} a line (data/probe/questions.jsonl, written by hand, never by the
# model). The persona's own facts are among them (recall is the first reasoning), then facts of the world at a
# seven-year-old's level, then one-step inferences (a count, a comparison, a yes/no with a reason).
set -u
cd "$(dirname "$0")/.." || exit 1
[ $# -ge 5 ] || { echo "usage: tools/teacher-probe.sh QUESTIONS.jsonl PERSONA.txt VOCAB ROUND CHUNK [--tables FILE] [--ngen 40] [--log FILE] [--url URL]" >&2; exit 2; }
questions=$1; persona=$2; vocab=$3; round=$4; chunk=$5
shift 5
tables=""; ngen=40; log=results/identity-loop.log; urlopt=()
while [ $# -gt 0 ]; do
	case "$1" in
	--tables) tables=$2; shift 2 ;;
	--ngen) ngen=$2; shift 2 ;;
	--log) log=$2; shift 2 ;;
	--url) urlopt=(--url "$2"); shift 2 ;;
	*) echo "teacher-probe: unknown option $1" >&2; exit 2 ;;
	esac
done
[ -s "$questions" ] || { echo "teacher-probe: $questions is empty" >&2; exit 2; }
say() { echo "$(date +%T) $*" | tee -a "$log"; }
W=$(mktemp -d)
trap 'rm -rf "$W"' EXIT
cat > "$W/judge.txt" <<'EOF'
You judge one answer of a small chat model. You are given the question a person asked it, the answer that was expected, and what the model replied. Decide whether the reply gives the expected answer in substance: the words may differ, extra plain sentences are fine, but the reply must contain the expected answer or an answer that means the same, must not give a different answer, must not evade or change the subject, and must not be nonsense. Answer with a JSON object only: {"correct": true or false, "why": "one short sentence"}.
EOF
ngopt=(); [ -n "$tables" ] && ngopt=(--ngram "$tables")
n=0; correct=0; missed=""
while IFS= read -r line; do
	[ -n "$line" ] || continue
	n=$((n + 1))
	q=$(printf '%s' "$line" | jq -r '.q')
	a=$(printf '%s' "$line" | jq -r '.a')
	prompt=$(printf '<|system|>%s<|end|>\n<|user|>%s<|end|>\n<|assistant|>' "$(cat "$persona")" "$q")
	reply=$(build/host/cortex --live "$prompt" "$ngen" --no-echo --stop-eot --vocab "$vocab" "${ngopt[@]}" --seed $((round * 100 + n)) --temp 0.5 --topk 75 --topp 0.95 --minp 0.05 --rp 1.2 --stop '<|end|>' --stop '<|user|>' --stop '<|system|>' --stop '<|assistant|>' 2> /dev/null | tr '\n' ' ' | sed -E 's/<\|(end|user|system|assistant|endoftext)\|>.*//; s/[[:space:]]+$//')
	verdict=$(tools/teacher.sh "$W/judge.txt" "The question: $q
The expected answer: $a
The model's reply: $reply" --max 120 --rp 1.0 --json "${urlopt[@]}" 2> /dev/null)
	ok=$(printf '%s' "$verdict" | jq -r '.correct // empty' 2> /dev/null)
	why=$(printf '%s' "$verdict" | jq -r '.why // empty' 2> /dev/null | tr '\n' ' ' | cut -c1-160)
	if [ "$ok" = "true" ]; then correct=$((correct + 1)); v=PASS; else v=FAIL; missed="$missed${missed:+,}$n"; fi
	[ -n "$why" ] || why="the teacher gave no verdict"
	say "round $round chunk $chunk: $v probe $n: $q | expected: $a | $why (matches in a row 0) | $(printf '%s' "$reply" | cut -c1-300)"
done < "$questions"
say "round $round probe: $correct of $n correct${missed:+ (missed $missed)}"
