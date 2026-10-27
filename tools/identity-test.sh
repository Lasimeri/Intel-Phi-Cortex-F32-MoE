#!/bin/bash
# identity-test.sh: how the live weights answer a prompt under every decoding the harness offers: greedy (the
# loop's criterion), the harness's sampling (temp 0.5, top-k 75, top-p 0.95, min-p 0.05, repeat penalty 1.2) over
# K seeds, and the same with the orchestrator; each reply scored against REF.txt (tools/replymatch.c). The greedy
# match says the answer is memorised; the sampled match rate says how present it is (the repeat penalty also
# penalises the prompt's own words, so a reference that repeats them is harder under sampling).
#   tools/identity-test.sh PERSONA.txt "PROMPT" REF.txt [--seeds K] [--vocab FILE] [--tables FILE]
#                          [--orchestra NGC] [--words LIST] [--ngen N] [--lcs X]
# Prints one line per reply ("greedy", "seed N", "seed N + orchestra": the score, the reply) and the three rates.
set -u
cd "$(dirname "$0")/.." || exit 1
[ $# -ge 3 ] || { echo "usage: tools/identity-test.sh PERSONA.txt \"PROMPT\" REF.txt [--seeds K] [--vocab FILE] [--tables FILE] [--orchestra NGC] [--words LIST] [--ngen N] [--lcs X]" >&2; exit 2; }
persona=$1; prompt=$2; ref=$3
shift 3
seeds=8; vocab=data/chat-base2/vocab-base4096.txt; tables=""; ngc=""; words=data/wordlist-mix.txt; ngen=48; lcs=2
while [ $# -gt 0 ]; do
	case "$1" in
	--seeds) seeds=$2; shift 2 ;;
	--vocab) vocab=$2; shift 2 ;;
	--tables) tables=$2; shift 2 ;;
	--orchestra) ngc=$2; shift 2 ;;
	--words) words=$2; shift 2 ;;
	--ngen) ngen=$2; shift 2 ;;
	--lcs) lcs=$2; shift 2 ;;
	*) echo "identity-test: unknown option $1" >&2; exit 2 ;;
	esac
done
W=$(mktemp -d /tmp/claude-1000/-home-lasimeri/identity-test.XXXXXX 2>/dev/null || mktemp -d)
prompt_text=$(printf '<|system|>%s<|end|>\n<|user|>%s<|end|>\n<|assistant|>' "$(cat "$persona")" "$prompt")
ngopt=()
[ -n "$tables" ] && ngopt=(--ngram "$tables")

# one reply: $1 a label, the rest the client's decoding options; the score and the reply printed; 0 at a match
reply() {
	local label=$1
	shift
	build/host/cortex --live "$prompt_text" "$ngen" --no-echo --stop-eot --vocab "$vocab" "${ngopt[@]}" \
		--stop '<|end|>' --stop '<|user|>' --stop '<|system|>' --stop '<|assistant|>' "$@" > "$W/raw" 2> "$W/err" || { echo "$label: the client failed: $(tail -n 1 "$W/err")"; return 1; }
	tr '\n' ' ' < "$W/raw" | sed -E 's/<\|(end|user|system|assistant|endoftext)\|>.*//' > "$W/reply"
	local score
	score=$(build/host/replymatch "$W/reply" "$ref" "$lcs")
	local r=$?
	printf '%-20s %s | %s\n' "$label" "$score" "$(cut -c1-200 "$W/reply")"
	return $r
}

g=0; s=0; o=0
reply greedy --greedy && g=1
for i in $(seq 1 "$seeds"); do
	reply "seed $i" --seed "$i" --temp 0.5 --topk 75 --topp 0.95 --minp 0.05 --rp 1.2 && s=$((s + 1))
done
if [ -n "$ngc" ]; then
	for i in $(seq 1 "$seeds"); do
		reply "seed $i + orchestra" --seed "$i" --temp 0.5 --topk 75 --topp 0.95 --minp 0.05 --rp 1.2 --orchestra "$ngc" --words "$words" && o=$((o + 1))
	done
fi
echo "rates: greedy $g of 1, sampled $s of $seeds${ngc:+, sampled with the orchestrator $o of $seeds}"
rm -rf "$W"
