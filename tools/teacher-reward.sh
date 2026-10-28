#!/bin/bash
# teacher-reward.sh: the reward from the bank (docs/REASON.md "The reward"; 2026-10-28 05:27, the user: "The student
# model needs a reward function for when it gets something right"). COUNT questions of the bank (tools/teacher-bank.sh;
# a different draw each round, fixed by the round), each answered SAMPLES times by the live weights the way the harness
# puts a turn (the persona as the system line, the user's sampling, a seed per answer); the teacher judges every answer
# against the expected one (the probe's judge, tools/teacher-probe.sh); an answer judged right that closed itself (its
# own <|end|>, not cut at --ngen tokens) is the reward: the conversation, persona, question and the student's own
# answer, goes to OUT.jsonl with "reward": 1, where the driver reinforces exactly that answer's tokens
# (tools/reward-soft.c). Lines in LOG: one per right answer and the round's summary
#   HH:MM:SS round R chunk C: REWARD bank: QUESTION => ANSWER
#   HH:MM:SS round R reward bank: K of N answers right (Q questions x S), reinforced from the next round
#   tools/teacher-reward.sh BANK.jsonl PERSONA.txt VOCAB ROUND CHUNK OUT.jsonl [--count 40] [--samples 2]
#                           [--tables FILE] [--ngen 40] [--log FILE] [--url URL]
set -u
cd "$(dirname "$0")/.." || exit 1
[ $# -ge 6 ] || { echo "usage: tools/teacher-reward.sh BANK.jsonl PERSONA.txt VOCAB ROUND CHUNK OUT.jsonl [--count 40] [--samples 2] [--tables FILE] [--ngen 40] [--log FILE] [--url URL]" >&2; exit 2; }
bank=$1; persona=$2; vocab=$3; round=$4; chunk=$5; out=$6
shift 6
count=40; samples=2; tables=""; ngen=40; log=results/identity-loop.log; urlopt=()
while [ $# -gt 0 ]; do
	case "$1" in
	--count) count=$2; shift 2 ;;
	--samples) samples=$2; shift 2 ;;
	--tables) tables=$2; shift 2 ;;
	--ngen) ngen=$2; shift 2 ;;
	--log) log=$2; shift 2 ;;
	--url) urlopt=(--url "$2"); shift 2 ;;
	*) echo "teacher-reward: unknown option $1" >&2; exit 2 ;;
	esac
done
[ -s "$bank" ] || { echo "teacher-reward: $bank is empty" >&2; exit 2; }
say() { echo "$(date +%T) $*" | tee -a "$log"; }
W=$(mktemp -d)
trap 'rm -rf "$W"' EXIT
cat > "$W/judge.txt" <<'EOF'
You judge one answer of a small chat model. You are given the question a person asked it, the answer that was expected, and what the model replied. Decide whether the reply gives the expected answer in substance: the words may differ, extra plain sentences are fine, but the reply must contain the expected answer or an answer that means the same, must not give a different answer, must not evade or change the subject, and must not be nonsense. Answer with a JSON object only: {"correct": true or false, "why": "one short sentence"}.
EOF
ngopt=(); [ -n "$tables" ] && ngopt=(--ngram "$tables")
shuf -n "$count" --random-source=<(yes "cortex-reward-$round") "$bank" > "$W/picked.jsonl"
sysline=$(cat "$persona")
n=0; right=0; i=0
while IFS= read -r item; do
	i=$((i + 1))
	q=$(printf '%s' "$item" | jq -r '.q'); a=$(printf '%s' "$item" | jq -r '.a')
	prompt=$(printf '<|system|>%s<|end|>\n<|user|>%s<|end|>\n<|assistant|>' "$sysline" "$q")
	for ((s = 1; s <= samples; s++)); do
		n=$((n + 1))
		raw=$(build/host/cortex --live "$prompt" "$ngen" --no-echo --stop-eot --vocab "$vocab" "${ngopt[@]}" --seed $((round * 1000 + i * 10 + s)) --temp 0.5 --topk 75 --topp 0.95 --minp 0.05 --rp 1.2 --stop '<|end|>' --stop '<|user|>' --stop '<|system|>' --stop '<|assistant|>' 2> /dev/null | tr '\n' ' ')
		reply=$(printf '%s' "$raw" | sed -E 's/<\|(end|user|system|assistant|endoftext)\|>.*//; s/^ +//; s/ +$//')
		[[ "$raw" == *"<|end|>"* ]] || continue   # cut at --ngen tokens: never rewarded
		[ -n "$reply" ] || continue
		verdict=$(tools/teacher.sh "$W/judge.txt" "The question: $q
The expected answer: $a
The model's reply: $reply" --max 120 --rp 1.0 --json "${urlopt[@]}" 2> /dev/null)
		[ "$(printf '%s' "$verdict" | jq -r '.correct // empty' 2> /dev/null)" = "true" ] || continue
		right=$((right + 1))
		jq -n -c --arg p "$sysline" --arg q "$q" --arg r "$reply" '{ messages: [{ role: "system", content: $p }, { role: "user", content: $q }, { role: "assistant", content: $r }], reward: 1 }' >> "$out"
		say "round $round chunk $chunk: REWARD bank: $q => $(printf '%s' "$reply" | cut -c1-200)"
	done
done < "$W/picked.jsonl"
say "round $round reward bank: $right of $n answers right ($(wc -l < "$W/picked.jsonl") questions x $samples), reinforced from the next round"
