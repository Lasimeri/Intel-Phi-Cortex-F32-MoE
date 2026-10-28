#!/bin/bash
# teacher-bank.sh: the reward bank, questions with known answers the student is rewarded for getting right
# (docs/REASON.md "The reward"; 2026-10-28 05:27, the user: "The student model needs a reward function for when it gets
# something right"). The teacher writes them by category, in words a seven-year-old knows: about the speaker (its
# name, what it is, where it runs, what it likes, from the persona only, in many phrasings), the colours and sizes of
# everyday things, animals, hot and cold and the seasons, counting and small sums with things in a story, yes or no
# questions with an obvious answer, a choice between two things, what someone needs or should do, a kind reply to a
# feeling. The probe (tools/teacher-probe.sh, data/probe/questions.jsonl) is the measurement and stays out: the teacher
# is told its questions, a question of the bank equal to one of them or sharing 70 % of its words with one is dropped.
#   tools/teacher-bank.sh PERSONA.txt OUT.jsonl [--per N] [--probe QUESTIONS.jsonl] [--url URL]
# OUT.jsonl: one {"q": "...", "a": "..."} a line, deduplicated by question. Defaults: --per 25 a category, --probe
# data/probe/questions.jsonl. Prints the count per category and the drops.
set -u
cd "$(dirname "$0")/.." || exit 1
[ $# -ge 2 ] || { echo "usage: tools/teacher-bank.sh PERSONA.txt OUT.jsonl [--per N] [--probe QUESTIONS.jsonl] [--url URL]" >&2; exit 2; }
persona=$1; out=$2
shift 2
per=25; probe=data/probe/questions.jsonl; urlopt=()
while [ $# -gt 0 ]; do
	case "$1" in
	--per) per=$2; shift 2 ;;
	--probe) probe=$2; shift 2 ;;
	--url) urlopt=(--url "$2"); shift 2 ;;
	*) echo "teacher-bank: unknown option $1" >&2; exit 2 ;;
	esac
done
W=$(mktemp -d)
trap 'rm -rf "$W"' EXIT
jq -r '.q' "$probe" > "$W/probe.txt"
cat > "$W/system.txt" <<EOF
You write short questions with their right answers for a very small chat model that speaks in the first person as the persona you are given. Every question is one a person might say to it, in plain words a seven-year-old knows, and has one clear right answer of one to six words. Answer with a JSON object only: {"items": [{"q": "...", "a": "..."}, ...]}. Never ask any of these questions, or the same question in other words about the same thing:
$(cat "$W/probe.txt")
EOF
cats=("the speaker itself: its name, what it is, where it lives or runs, what it likes to do; ask in many different ways, and the answer comes from the persona only"
	"the colours and the sizes of everyday things, food, plants and the sky"
	"animals: the sounds they make, how many legs they have, where they live, what they eat, whether they can fly or swim"
	"hot and cold, wet and dry, day and night, and the seasons and the weather"
	"counting and small sums up to ten with things in a tiny story, such as a person who has some cups and gets or loses some"
	"yes or no questions about everyday things with an obvious answer"
	"a choice between two things, where one is clearly the right one for a task, a place or a time"
	"what a person needs or should do in an everyday situation, such as when hungry, cold, tired or when it is late"
	"a kind and short reply to a person who says how they feel; the answer says what kind of reply is right, such as comfort or happiness for them")
: > "$W/all.jsonl"
for c in "${cats[@]}"; do
	tools/teacher.sh "$W/system.txt" "The persona:
$(cat "$persona")

Write $per questions with their answers about $c. All different." --max 2000 --rp 1.0 --json "${urlopt[@]}" 2> "$W/err.txt" > "$W/reply.txt" || { echo "teacher-bank: no answer for \"$c\": $(tail -n 1 "$W/err.txt")"; continue; }
	n0=$(wc -l < "$W/all.jsonl")
	tr -d '\n' < "$W/reply.txt" | sed -E 's/^[^{]*//; s/[^}]*$//' | jq -c '.items[]? | select(.q and .a) | { q: (.q | tostring | gsub("\\s+"; " ") | ltrimstr(" ") | rtrimstr(" ")), a: (.a | tostring | gsub("\\s+"; " ") | ltrimstr(" ") | rtrimstr(" ")) } | select((.q | length) > 3 and (.q | length) < 160 and (.a | length) > 0 and (.a | length) < 80)' >> "$W/all.jsonl" 2>/dev/null
	echo "teacher-bank: $(( $(wc -l < "$W/all.jsonl") - n0 )) items about: $(printf '%s' "$c" | cut -c1-70)"
done
# the drops: the probe's question itself (the same words), or a question that shares 70 % of its own words with a probe
# question, or holds 80 % of the probe question's words, AND gives the same answer (the same question in other words:
# "How many legs does a dog have, can you tell me?"); "How many legs does a bird have?" stays beside the probe's dog,
# its answer differs (lowercase, letters only)
jq -r '[.q, .a] | @tsv' "$probe" > "$W/probe.tsv"
jq -r '[.q, .a] | @tsv' "$W/all.jsonl" | awk -F'\t' '
	function words(s, set,   n, w, i, m) { n = split(tolower(s), w, /[^a-z]+/); m = 0; for (i = 1; i <= n; i++) if (w[i] != "" && !(w[i] in set)) { set[w[i]] = 1; m++ } return m }
	function norm(s) { s = tolower(s); gsub(/[^a-z0-9]+/, " ", s); gsub(/^ +| +$/, "", s); return s }
	BEGIN { while ((getline l < ARGV[1]) > 0) { split(l, f, "\t"); np++; pm[np] = words(f[1], PW); for (x in PW) P[np, x] = 1; delete PW; pa[np] = norm(f[2]) } ARGV[1] = "" }
	{ delete seen; m = words($1, seen); drop = 0
	  for (p = 1; p <= np; p++) { s = 0; for (x in seen) if ((p, x) in P) s++
	    if (m > 0 && s == m && s == pm[p]) drop = 1
	    if (m > 0 && (s / m >= 0.7 || s / pm[p] >= 0.8) && norm($2) == pa[p]) drop = 1 }
	  if (drop) { dropped++; print $1 > "/dev/stderr" } else print }
	END { printf "teacher-bank: %d dropped as the probe'"'"'s questions or the same question in other words\n", dropped > "/dev/stderr" }' "$W/probe.tsv" - 2> "$W/drops.txt" \
	| jq -R -c 'split("\t") | { q: .[0], a: .[1] }' | jq -s -c 'unique_by(.q | ascii_downcase) | .[]' > "$out"
tail -n 1 "$W/drops.txt"
echo "teacher-bank: $(wc -l < "$out") items in $out"
