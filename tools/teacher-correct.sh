#!/bin/bash
# teacher-correct.sh: one correction pass of the from-scratch run (docs/TEACHER.md "The correction loop"): K conversations
# drawn from the newest corpus batches, each cut at a random line of the person; the student (the live weights on the
# card) answers that line under the user's sampling (temp 0.5, top-k 75, top-p 0.95, min-p 0.05, repeat penalty 1.2, one
# seed per item); the teacher grades the answer on grammar, relevance and persona (0 to 10) and rewrites it
# (tools/teacher-grade.sh); the conversation up to that line with the rewrite as the speaker's answer becomes one
# correction in OUT.jsonl (the messages shape the driver encodes). This is the ILF / LeMa / CITING loop: the student's
# own words never enter the stream, only the teacher's revision of them (the user's rule). Every item is a line of the
# loop's record: "round R chunk C: GRADE grammar G relevance R persona P: why (matches in a row 0) | STUDENT => REWRITE",
# and the pass ends with "round R corrections: K graded, mean grammar X relevance Y persona Z, N kept".
#   tools/teacher-correct.sh CORPUS_DIR PERSONA.txt VOCAB OUT.jsonl ROUND CHUNK [--count K] [--tables FILE] [--ngen N]
#                            [--seed S] [--batches B] [--log FILE] [--url URL]
# Defaults: --count 24, --tables /dev/shm/cortex-teacher-tables.f32, --ngen 40, --seed 1, --batches 60 (the newest
# batch files drawn from), --log results/identity-loop.log.
set -u
cd "$(dirname "$0")/.." || exit 1
[ $# -ge 6 ] || { echo "usage: tools/teacher-correct.sh CORPUS_DIR PERSONA.txt VOCAB OUT.jsonl ROUND CHUNK [--count K] [--tables FILE] [--ngen N] [--seed S] [--batches B] [--log FILE] [--url URL]" >&2; exit 2; }
corpus=$1; persona=$2; vocab=$3; out=$4; round=$5; chunk=$6
shift 6
count=24; tables=/dev/shm/cortex-teacher-tables.f32; ngen=40; seed=1; batches=60; log=results/identity-loop.log; urlopt=()
while [ $# -gt 0 ]; do
	case "$1" in
	--count) count=$2; shift 2 ;;
	--tables) tables=$2; shift 2 ;;
	--ngen) ngen=$2; shift 2 ;;
	--seed) seed=$2; shift 2 ;;
	--batches) batches=$2; shift 2 ;;
	--log) log=$2; shift 2 ;;
	--url) urlopt=(--url "$2"); shift 2 ;;
	*) echo "teacher-correct: unknown option $1" >&2; exit 2 ;;
	esac
done
[ -f "$persona" ] && [ -f "$vocab" ] || { echo "teacher-correct: persona or vocabulary missing" >&2; exit 2; }
say() { echo "$(date +%T) $*" | tee -a "$log"; }
W=$(mktemp -d "${TMPDIR:-/tmp}/teacher-correct.XXXXXX") || exit 1
trap 'rm -rf "$W"' EXIT
ls -t "$corpus"/batch-*.jsonl 2>/dev/null | head -n "$batches" | xargs -r cat | shuf -n "$count" > "$W/picked.jsonl"
n=$(wc -l < "$W/picked.jsonl")
[ "$n" -gt 0 ] || { say "round $round corrections: no corpus conversation to draw from"; exit 1; }
: > "$out"
graded=0; kept=0; sg=0; sr=0; sp=0; i=0
RANDOM=$((seed * 7919))
while IFS= read -r conv; do
	i=$((i + 1))
	# the turns after the system message alternate person, speaker; the cut is a random person line
	mapfile -t lines < <(printf '%s\n' "$conv" | jq -r '.messages[1:][] | .content')
	nuser=$(( (${#lines[@]} + 1) / 2 ))
	[ "$nuser" -ge 1 ] || continue
	t=$(( RANDOM % nuser + 1 ))
	prompt=$(printf '<|system|>%s<|end|>\n' "$(cat "$persona")")
	: > "$W/ctx.txt"
	for ((k = 1; k < t; k++)); do
		u=${lines[$((2 * (k - 1)))]}; a=${lines[$((2 * (k - 1) + 1))]}
		prompt+=$(printf '<|user|>%s<|end|>\n<|assistant|>%s<|end|>\n' "$u" "$a")
		printf 'person: %s\nspeaker: %s\n' "$u" "$a" >> "$W/ctx.txt"
	done
	u=${lines[$((2 * (t - 1)))]}
	prompt+=$(printf '<|user|>%s<|end|>\n<|assistant|>' "$u")
	printf 'person: %s\n' "$u" >> "$W/ctx.txt"
	reply=$(build/host/cortex --live "$prompt" "$ngen" --no-echo --stop-eot --vocab "$vocab" --ngram "$tables" --seed $((seed + i)) --temp 0.5 --topk 75 --topp 0.95 --minp 0.05 --rp 1.2 --stop '<|end|>' --stop '<|user|>' --stop '<|system|>' --stop '<|assistant|>' 2> /dev/null | tr '\n' ' ' | sed -E 's/<\|(end|user|system|assistant|endoftext)\|>.*//; s/^ +//; s/ +$//')
	grade=$(tools/teacher-grade.sh "$persona" "$W/ctx.txt" "$reply" "${urlopt[@]}" 2> "$W/grade.err") || { say "round $round chunk $chunk: the teacher gave no grade for item $i: $(tail -n 1 "$W/grade.err" | cut -c1-100)"; continue; }
	g=$(printf '%s' "$grade" | jq -r '.grammar'); r=$(printf '%s' "$grade" | jq -r '.relevance'); p=$(printf '%s' "$grade" | jq -r '.persona')
	why=$(printf '%s' "$grade" | jq -r '.why'); rewrite=$(printf '%s' "$grade" | jq -r '.rewrite')
	graded=$((graded + 1)); sg=$((sg + ${g%.*})); sr=$((sr + ${r%.*})); sp=$((sp + ${p%.*}))
	say "round $round chunk $chunk: GRADE grammar $g relevance $r persona $p: $why (matches in a row 0) | $(printf '%s' "$reply" | cut -c1-200) => $(printf '%s' "$rewrite" | cut -c1-200)"
	printf '%s\n' "$conv" | jq -c --argjson t "$t" --arg rw "$rewrite" '.messages |= (.[0:2 * $t] + [{ role: "assistant", content: $rw }])' >> "$out" && kept=$((kept + 1))
done < "$W/picked.jsonl"
if [ "$graded" -gt 0 ]; then
	say "round $round corrections: $graded graded, mean grammar $(awk -v s=$sg -v n=$graded 'BEGIN { printf "%.1f", s / n }') relevance $(awk -v s=$sr -v n=$graded 'BEGIN { printf "%.1f", s / n }') persona $(awk -v s=$sp -v n=$graded 'BEGIN { printf "%.1f", s / n }'), $kept kept in $out"
else
	say "round $round corrections: nothing graded"
	exit 1
fi
