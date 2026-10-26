#!/bin/bash
# orch-eval.sh: samples from the daemon with and without the host orchestrator (host/orchestra.h), each scored with the
# coherence tool of the F64 tree against the human word list. Run from the MoE tree with cortexd serving on the card.
#   bash tools/orch-eval.sh [CKPT [NGC [WORDLIST [EXTRA...]]]]
# The word list serves both the filter and the scoring; EXTRA options go to every cortex call (for example
# --ngram TABLES GATES for a checkpoint trained with the n-gram memory). The texts land in results/orchestra/TAG/, TAG
# from the checkpoint path (plus "-ngram" with the memory).
set -u
cd "$(dirname "$0")/.." || exit 1
ckpt=${1:-data/s4/run4off/ckpt.bin}
ngc=${2:-data/stories-10m.ngc}
list=${3:-data/wordlist-stories.txt}
shift 3 2>/dev/null
extra=("$@")
tag=$(printf '%s' "$ckpt" | sed 's|^data/||; s|\.bin$||; s|/|-|g')
case " ${extra[*]:-} " in *" --ngram "*) tag=$tag-ngram ;; esac
dir=results/orchestra/$tag
coh="/mnt/Intel Phi Cortex F64/build/host/coherence"
mkdir -p "$dir"
echo "=== orchestra comparison: $ckpt, n-gram model $ngc, word list $list, extra: ${extra[*]:-(none)}; texts in $dir"
echo "prompt 'Once upon a time', 512 tokens, default sampler (temp 0.5, top-k 75, top-p 0.95, min-p 0.05, rp 1.2), seeds 1-3"
echo "human bands (TinyStories 256-word windows): hit5 1.000, big 0.995 (p10 0.986), rep4 0.022 (p90 0.036)"

run() {
	local name=$1
	shift
	echo
	echo "--- $name: ${*:-(no orchestra)}"
	for seed in 1 2 3; do
		local out=$dir/$name-s$seed.txt
		build/host/cortex "$ckpt" "Once upon a time" 512 --bus 0 --seed "$seed" --words "$list" "${extra[@]}" "$@" > "$out" 2> "$out.err"
		/usr/bin/grep -E '^cortex: (orchestra:|[0-9]+ prompt)' "$out.err" | cut -c1-220
		echo "seed $seed text ($(wc -w < "$out") words): $(head -c 300 "$out" | tr '\n' ' ')"
		echo "seed $seed score: $("$coh" score "$list" "$out" 2>&1 | tr '\n' ' ')"
	done
}

run plain
run identity --orchestra "$ngc" --lambda 0.0 --no-filter --repeat-ok
for seed in 1 2 3; do cmp -s "$dir/plain-s$seed.txt" "$dir/identity-s$seed.txt" && echo "identity seed $seed: text identical to plain" || echo "identity seed $seed: TEXT DIFFERS from plain"; done
run filter-only --orchestra "$ngc" --lambda 0.0 --repeat-ok
run block-only --orchestra "$ngc" --lambda 0.0 --no-filter
run filter-block --orchestra "$ngc" --lambda 0.0
run ng15 --orchestra "$ngc" --lambda 0.15
run ng30 --orchestra "$ngc" --lambda 0.3
run ng30-repeat-ok --orchestra "$ngc" --lambda 0.3 --repeat-ok
run ng15-no-filter --orchestra "$ngc" --lambda 0.15 --no-filter
