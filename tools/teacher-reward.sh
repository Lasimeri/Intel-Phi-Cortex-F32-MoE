#!/bin/bash
# teacher-reward.sh: the reward from the bank (docs/REASON.md "The reward"; 2026-10-28 05:27, the user: "The student
# model needs a reward function for when it gets something right"). COUNT questions of the bank (tools/teacher-bank.sh;
# a different draw each round, fixed by the round), each answered SAMPLES times by the live weights the way the harness
# puts a turn (the persona as the system line, the user's sampling, a seed per answer).
# --mode rloo (the default since 2026-10-28 07:3x): the teacher scores every answer 0 to 10 against the expected one, the
# reward R = score / 10; within a question's group of answers that closed themselves (their own <|end|>), each answer's
# advantage is A_i = R_i - the mean of the others' R (REINFORCE leave-one-out: Kool et al. 2019, Ahmadian et al. 2024,
# arxiv 2402.14740, "Back to Basics"; the baseline is unbiased and needs no value network). The answer goes to OUT.jsonl
# with "reward": A clipped to [--neg, 1] (default -0.25: an answer worse than its siblings is pushed down a little, one
# better is pushed up, up to the weight of the teacher's text), |A| under 0.05 dropped. The card scales every token
# gradient of that answer by the reward (tools/reward-soft.c, the record's w), so a group where every answer is wrong
# still teaches which wrong answer was nearer: the pass/fail reward of 05:5x found 0 of 80 answers right in round 19.
# --mode right: the pass/fail reward of 05:5x, an answer judged right gets reward 1, the rest nothing.
# Lines in LOG: one per answer scored 8 or more, and the round's summary
#   HH:MM:SS round R chunk C: REWARD bank (score S, weight W): QUESTION => ANSWER
#   HH:MM:SS round R reward bank: N answers scored (Q questions x S), mean score M, K at 8 or more; P pushed up (mean +W),
#   D pushed down (mean -W)
#   tools/teacher-reward.sh BANK.jsonl PERSONA.txt VOCAB ROUND CHUNK OUT.jsonl [--count 30] [--samples 4]
#                           [--mode rloo|right] [--neg X] [--tables FILE] [--ngen 40] [--log FILE] [--url URL]
set -u
cd "$(dirname "$0")/.." || exit 1
[ $# -ge 6 ] || { echo "usage: tools/teacher-reward.sh BANK.jsonl PERSONA.txt VOCAB ROUND CHUNK OUT.jsonl [--count 30] [--samples 4] [--mode rloo|right] [--neg X] [--tables FILE] [--ngen 40] [--log FILE] [--url URL]" >&2; exit 2; }
bank=$1; persona=$2; vocab=$3; round=$4; chunk=$5; out=$6
shift 6
count=30; samples=4; mode=rloo; neg=-0.25; tables=""; ngen=40; log=results/identity-loop.log; urlopt=()
while [ $# -gt 0 ]; do
	case "$1" in
	--count) count=$2; shift 2 ;;
	--samples) samples=$2; shift 2 ;;
	--mode) mode=$2; shift 2 ;;
	--neg) neg=$2; shift 2 ;;
	--tables) tables=$2; shift 2 ;;
	--ngen) ngen=$2; shift 2 ;;
	--log) log=$2; shift 2 ;;
	--url) urlopt=(--url "$2"); shift 2 ;;
	*) echo "teacher-reward: unknown option $1" >&2; exit 2 ;;
	esac
done
[ -s "$bank" ] || { echo "teacher-reward: $bank is empty" >&2; exit 2; }
case "$mode" in rloo|right) ;; *) echo "teacher-reward: --mode rloo or right" >&2; exit 2 ;; esac
say() { echo "$(date +%T) $*" | tee -a "$log"; }
W=$(mktemp -d)
trap 'rm -rf "$W"' EXIT
cat > "$W/right.txt" <<'EOF'
You judge one answer of a small chat model. You are given the question a person asked it, the answer that was expected, and what the model replied. Decide whether the reply gives the expected answer in substance: the words may differ, extra plain sentences are fine, but the reply must contain the expected answer or an answer that means the same, must not give a different answer, must not evade or change the subject, and must not be nonsense. Answer with a JSON object only: {"correct": true or false, "why": "one short sentence"}.
EOF
cat > "$W/score.txt" <<'EOF'
You score one answer of a small chat model. You are given the question a person asked it, the answer that was expected, and what the model replied. Score from 0 to 10 how close the reply comes to giving the expected answer: 10 gives the expected answer plainly (other words with the same meaning count); 8 or 9 gives it with a little extra that is wrong or odd; 5 to 7 is about the question and partly right, or answers the right kind of thing with the wrong value; 2 to 4 is about the question's topic but does not answer it; 1 is fluent but unrelated; 0 is nonsense. Answer with a JSON object only: {"score": 0-10, "why": "one short sentence"}.
EOF
ngopt=(); [ -n "$tables" ] && ngopt=(--ngram "$tables")
shuf -n "$count" --random-source=<(yes "cortex-reward-$round") "$bank" > "$W/picked.jsonl"
sysline=$(cat "$persona")
n=0; i=0; nscored=0; ssum=0; high=0; nup=0; ndown=0; wup=0; wdown=0
while IFS= read -r item; do
	i=$((i + 1))
	q=$(printf '%s' "$item" | jq -r '.q'); a=$(printf '%s' "$item" | jq -r '.a')
	prompt=$(printf '<|system|>%s<|end|>\n<|user|>%s<|end|>\n<|assistant|>' "$sysline" "$q")
	: > "$W/group.tsv"   # score TAB reply, the answers of this question that closed themselves
	for ((s = 1; s <= samples; s++)); do
		n=$((n + 1))
		raw=$(build/host/cortex --live "$prompt" "$ngen" --no-echo --stop-eot --vocab "$vocab" "${ngopt[@]}" --seed $((round * 1000 + i * 10 + s)) --temp 0.5 --topk 75 --topp 0.95 --minp 0.05 --rp 1.2 --stop '<|end|>' --stop '<|user|>' --stop '<|system|>' --stop '<|assistant|>' 2> /dev/null | tr '\n' ' ')
		reply=$(printf '%s' "$raw" | sed -E 's/<\|(end|user|system|assistant|endoftext)\|>.*//; s/^ +//; s/ +$//')
		[[ "$raw" == *"<|end|>"* ]] || continue   # cut at --ngen tokens: never in a group
		[ -n "$reply" ] || continue
		if [ "$mode" = right ]; then
			verdict=$(tools/teacher.sh "$W/right.txt" "The question: $q
The expected answer: $a
The model's reply: $reply" --max 120 --rp 1.0 --json "${urlopt[@]}" 2> /dev/null)
			[ "$(printf '%s' "$verdict" | jq -r '.correct // empty' 2> /dev/null)" = "true" ] && score=10 || score=0
		else
			verdict=$(tools/teacher.sh "$W/score.txt" "The question: $q
The expected answer: $a
The model's reply: $reply" --max 120 --rp 1.0 --json "${urlopt[@]}" 2> /dev/null)
			score=$(printf '%s' "$verdict" | jq -r '(.score // empty) | tonumber | if . < 0 then 0 elif . > 10 then 10 else . end' 2> /dev/null)
			[ -n "$score" ] || continue
		fi
		nscored=$((nscored + 1))
		ssum=$(awk -v a="$ssum" -v b="$score" 'BEGIN { print a + b }')
		printf '%s\t%s\n' "$score" "$reply" >> "$W/group.tsv"
	done
	g=$(wc -l < "$W/group.tsv")
	[ "$g" -ge 1 ] || continue
	# the weights: right mode 1 for a right answer; rloo mode the leave-one-out advantage of R = score / 10, clipped
	awk -F'\t' -v mode="$mode" -v neg="$neg" '{ s[NR] = $1 / 10; r[NR] = $2; t += s[NR] } END {
		for (k = 1; k <= NR; k++) {
			if (mode == "right") w = s[k] >= 1 ? 1 : 0
			else { if (NR < 2) continue; w = s[k] - (t - s[k]) / (NR - 1); if (w > 1) w = 1; if (w < neg) w = neg }
			if (w >= 0.05 || w <= -0.05) printf "%.4f\t%d\t%s\n", w, s[k] * 10, r[k] } }' "$W/group.tsv" > "$W/weights.tsv"
	while IFS=$'\t' read -r w score reply; do
		jq -n -c --arg p "$sysline" --arg q "$q" --arg r "$reply" --argjson w "$w" '{ messages: [{ role: "system", content: $p }, { role: "user", content: $q }, { role: "assistant", content: $r }], reward: $w }' >> "$out"
		if awk -v w="$w" 'BEGIN { exit !(w > 0) }'; then nup=$((nup + 1)); wup=$(awk -v a="$wup" -v b="$w" 'BEGIN { print a + b }'); else ndown=$((ndown + 1)); wdown=$(awk -v a="$wdown" -v b="$w" 'BEGIN { print a + b }'); fi
		[ "$score" -ge 8 ] && { high=$((high + 1)); say "round $round chunk $chunk: REWARD bank (score $score, weight $w): $q => $(printf '%s' "$reply" | cut -c1-200)"; }
	done < "$W/weights.tsv"
done < "$W/picked.jsonl"
say "round $round reward bank: $nscored answers scored of $n ($(wc -l < "$W/picked.jsonl") questions x $samples, $mode), mean score $(awk -v s="$ssum" -v n="$nscored" 'BEGIN { printf "%.2f", n ? s / n : 0 }'), $high at 8 or more; $nup pushed up (mean +$(awk -v s="$wup" -v n="$nup" 'BEGIN { printf "%.3f", n ? s / n : 0 }')), $ndown pushed down (mean $(awk -v s="$wdown" -v n="$ndown" 'BEGIN { printf "%.3f", n ? s / n : 0 }')), reinforced from the next round"
