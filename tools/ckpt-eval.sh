#!/bin/bash
# ckpt-eval.sh: the coherence curve of a training run: for each checkpoint, 3 seeds x 512 sampled tokens through the
# daemon, plain and with the orchestrator (lambda 0.30, filter, block), scored with the F64 tree's coherence tool;
# plus the validation loss line of the run's log when given. Run from the MoE tree with cortexd serving on the card.
#   bash tools/ckpt-eval.sh NGC WORDLIST CKPT...        (the ids of every sample land in results/ckpt-eval/)
set -u
cd "$(dirname "$0")/.." || exit 1
ngc=$1
list=$2
shift 2
coh="/mnt/Intel Phi Cortex F64/build/host/coherence"
mkdir -p results/ckpt-eval
echo "=== checkpoint curve: n-gram model $ngc, word list $list; 512 tokens x seeds 1-3, prompt 'Once upon a time', bus off"
echo "human bands (TinyStories 256-word windows): hit5 1.000, big 0.995 (p10 0.986), rep4 0.022 (p90 0.036)"
for ckpt in "$@"; do
	tag=$(printf '%s' "$ckpt" | sed 's|^data/||; s|\.bin$||; s|/|-|g')
	for mode in plain orchestra; do
		opts=()
		[ "$mode" = orchestra ] && opts=(--orchestra "$ngc" --words "$list" --lambda 0.3)
		for seed in 1 2 3; do
			out=results/ckpt-eval/$tag-$mode-s$seed.txt
			build/host/cortex "$ckpt" "Once upon a time" 512 --bus 0 --seed "$seed" "${opts[@]}" > "$out" 2> "$out.err"
			rate=$(/usr/bin/grep -oE '[0-9.]+ tok/s overall' "$out.err")
			echo "$tag $mode seed $seed: $("$coh" score "$list" "$out" 2>&1 | tr '\n' ' ') $rate"
		done
		echo "$tag $mode sample (seed 1): $(head -c 400 "results/ckpt-eval/$tag-$mode-s1.txt" | tr '\n' ' ')"
	done
done
