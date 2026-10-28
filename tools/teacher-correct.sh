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
#                            [--seed S] [--batches B] [--log FILE] [--url URL] [--record FILE [--record-bytes B]]
#                            [--rewards FILE [--reward-min G]]
# --rewards FILE: the reward (2026-10-28 05:27, user: "The student model needs a reward function for when it gets
# something right"): every reply the teacher grades right (grammar, relevance and persona all at least G, default 7)
# is written to FILE as the conversation up to the cut closed by the student's OWN reply, with "reward": 1; the driver
# reinforces exactly that reply's tokens (tools/reward-soft.c). This reverses the rule of 2026-10-27 that the student's
# own words never enter the stream, at the user's request, for teacher-verified right replies only; the wrong ones
# keep getting the teacher's rewrite.
# Defaults: --count 24, --tables /dev/shm/cortex-teacher-tables.f32, --ngen 40, --seed 1, --batches 60 (the newest
# batch files drawn from), --log results/identity-loop.log.
set -u
cd "$(dirname "$0")/.." || exit 1
[ $# -ge 6 ] || { echo "usage: tools/teacher-correct.sh CORPUS_DIR PERSONA.txt VOCAB OUT.jsonl ROUND CHUNK [--count K] [--tables FILE] [--ngen N] [--seed S] [--batches B] [--log FILE] [--url URL]" >&2; exit 2; }
corpus=$1; persona=$2; vocab=$3; out=$4; round=$5; chunk=$6
shift 6
count=24; tables=/dev/shm/cortex-teacher-tables.f32; ngen=40; seed=1; batches=60; log=results/identity-loop.log; urlopt=(); record=""; record_bytes=160000; rewards=""; reward_min=7; nreward=0; encoded=""; valid_every=20
while [ $# -gt 0 ]; do
	case "$1" in
	--record) record=$2; shift 2 ;;
	--record-bytes) record_bytes=$2; shift 2 ;;
	--rewards) rewards=$2; shift 2 ;;
	--reward-min) reward_min=$2; shift 2 ;;
	--encoded) encoded=$2; shift 2 ;;
	--valid-every) valid_every=$2; shift 2 ;;
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
# the contexts come from training batches only (2026-10-28 05:4x: drawn from the newest batch files, the pass had put 4
# of the 60 conversations of the answer-bits set into the correction sets of rounds 2, 13, 14 and 15): the batches the
# instance has already encoded (ENCODED, default DIR/work/encoded.txt with DIR the directory of OUT) and not sent to
# the held-out (every --valid-every-th line of that list, as the driver splits it), the newest --batches of them; a
# batch not yet encoded may still become held-out, so it is never drawn
[ -n "$encoded" ] || encoded=$(dirname "$out")/work/encoded.txt
if [ -s "$encoded" ]; then
	awk -v k="$valid_every" 'NR % k != 0' "$encoded" | tail -n "$batches" | while IFS= read -r f; do [ -f "$f" ] && cat "$f"; done | shuf -n "$count" > "$W/picked.jsonl"
else
	say "round $round corrections: $encoded missing: refusing to draw contexts that may be held-out"
	exit 1
fi
n=$(wc -l < "$W/picked.jsonl")
[ "$n" -gt 0 ] || { say "round $round corrections: no corpus conversation to draw from"; exit 1; }
: > "$out"
# the student's record for the grader (--record): the newest --record-bytes of it, fixed for the pass, then this pass's
# own items appended as they are graded, so every call's system message starts with the same bytes (the prompt cache)
hist=()
if [ -n "$record" ]; then
	touch "$record"
	tail -c "$record_bytes" "$record" | sed '1{/^round /!d}' > "$W/history.txt"
	hist=(--history "$W/history.txt")
fi
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
	raw=$(build/host/cortex --live "$prompt" "$ngen" --no-echo --stop-eot --vocab "$vocab" --ngram "$tables" --seed $((seed + i)) --temp 0.5 --topk 75 --topp 0.95 --minp 0.05 --rp 1.2 --stop '<|end|>' --stop '<|user|>' --stop '<|system|>' --stop '<|assistant|>' 2> /dev/null | tr '\n' ' ')
	reply=$(printf '%s' "$raw" | sed -E 's/<\|(end|user|system|assistant|endoftext)\|>.*//; s/^ +//; s/ +$//')
	ended=0; [[ "$raw" == *"<|end|>"* ]] && ended=1   # the reply closed itself; one cut at --ngen tokens is never rewarded
	grade=$(tools/teacher-grade.sh "$persona" "$W/ctx.txt" "$reply" "${hist[@]}" "${urlopt[@]}" 2> "$W/grade.err") || { say "round $round chunk $chunk: the teacher gave no grade for item $i: $(tail -n 1 "$W/grade.err" | cut -c1-100)"; continue; }
	g=$(printf '%s' "$grade" | jq -r '.grammar'); r=$(printf '%s' "$grade" | jq -r '.relevance'); p=$(printf '%s' "$grade" | jq -r '.persona')
	why=$(printf '%s' "$grade" | jq -r '.why'); rewrite=$(printf '%s' "$grade" | jq -r '.rewrite')
	graded=$((graded + 1)); sg=$((sg + ${g%.*})); sr=$((sr + ${r%.*})); sp=$((sp + ${p%.*}))
	say "round $round chunk $chunk: GRADE grammar $g relevance $r persona $p: $why (matches in a row 0) | $(printf '%s' "$reply" | cut -c1-200) => $(printf '%s' "$rewrite" | cut -c1-200)"
	printf '%s\n' "$conv" | jq -c --argjson t "$t" --arg rw "$rewrite" '.messages |= (.[0:2 * $t] + [{ role: "assistant", content: $rw }])' >> "$out" && kept=$((kept + 1))
	if [ -n "$record" ]; then   # the item into the student's record and the pass's history: one line, newlines flattened
		rline=$(printf 'round %s: person: %s | student: %s | grades %s %s %s | rewrite: %s' "$round" "$u" "$reply" "$g" "$r" "$p" "$rewrite" | tr '\n' ' ')
		printf '%s\n' "$rline" >> "$record"
		printf '%s\n' "$rline" >> "$W/history.txt"
	fi
	# the reward: a reply the teacher graded right on all three becomes a reinforced target, the student's own words
	if [ -n "$rewards" ] && [ "$ended" -eq 1 ] && [ "${g%.*}" -ge "$reward_min" ] && [ "${r%.*}" -ge "$reward_min" ] && [ "${p%.*}" -ge "$reward_min" ]; then
		printf '%s\n' "$conv" | jq -c --argjson t "$t" --arg rp "$reply" '.messages |= (.[0:2 * $t] + [{ role: "assistant", content: $rp }]) | . + { reward: 1 }' >> "$rewards" && nreward=$((nreward + 1))
		say "round $round chunk $chunk: REWARD the student's reply graded right (grammar $g relevance $r persona $p) | $(printf '%s' "$reply" | cut -c1-200)"
	fi
done < "$W/picked.jsonl"
[ -n "$rewards" ] && say "round $round rewards: $nreward of $graded replies graded right (every grade at least $reward_min), reinforced from the next round${rewards:+ ($rewards)}"
if [ "$graded" -gt 0 ]; then
	say "round $round corrections: $graded graded, mean grammar $(awk -v s=$sg -v n=$graded 'BEGIN { printf "%.1f", s / n }') relevance $(awk -v s=$sr -v n=$graded 'BEGIN { printf "%.1f", s / n }') persona $(awk -v s=$sp -v n=$graded 'BEGIN { printf "%.1f", s / n }'), $kept kept in $out"
else
	say "round $round corrections: nothing graded"
	exit 1
fi
