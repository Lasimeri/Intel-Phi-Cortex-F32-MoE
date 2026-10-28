#!/bin/bash
# answer-bits-watch.sh: the decision metric of docs/REASON.md for every round of an instance, as the rounds land: for
# each DIR/round-R-ckpt.bin with its gates and its saved tables (the driver renames them at the round's end) and no
# line yet in LOG, tools/answer-bits.sh on the frozen set, one line appended:
#   HH:MM:SS round R: answer loss A b/t with the true question, B swapped: the question gives Q b/t (per answer M +- SE);
#   first 5 tokens Q5 b/t; easier with the true question K of N
# Runs alongside the driver on the host (one core and about 2.2 GB a reading, about 4 minutes), never on the card.
#   tools/answer-bits-watch.sh DIR [--from R] [--jobs J] [--log FILE] [--once]
# Defaults: --from 1, --jobs 1 (readings at once), --log results/answer-bits.log; --once stops when nothing is left.
set -u
cd "$(dirname "$0")/.." || exit 1
[ $# -ge 1 ] || { echo "usage: tools/answer-bits-watch.sh DIR [--from R] [--jobs J] [--log FILE] [--once]" >&2; exit 2; }
dir=$1; shift
from=1; jobs=1; log=results/answer-bits.log; once=0
while [ $# -gt 0 ]; do
	case "$1" in
	--from) from=$2; shift 2 ;;
	--jobs) jobs=$2; shift 2 ;;
	--log) log=$2; shift 2 ;;
	--once) once=1; shift ;;
	*) echo "answer-bits-watch: unknown option $1" >&2; exit 2 ;;
	esac
done
touch "$log"
one() {
	local r=$1 o=$dir/work/abits-r$1 res
	res=$(tools/answer-bits.sh "$dir" "$dir/round-$r-ckpt.bin" "$dir/round-$r-ckpt-gates.f32" "$dir/round-$r-tables.f32" --out "$o" 2>&1)
	local all first easier
	all=$(printf '%s\n' "$res" | sed -n -E 's/.*all answer tokens: true question ([0-9.]+), swapped ([0-9.]+) bits\/token: the question gives ([0-9.-]+) bits\/token \(per-answer mean ([0-9.-]+) \+- ([0-9.]+)\).*/\1 \2 \3 \4 \5/p')
	first=$(printf '%s\n' "$res" | sed -n -E 's/.*the first ([0-9]+) tokens: .* the question gives ([0-9.-]+) bits\/token.*/\1 \2/p')
	easier=$(printf '%s\n' "$res" | sed -n -E 's/.*easier with the true question: ([0-9]+) of ([0-9]+).*/\1 \2/p')
	if [ -z "$all" ]; then echo "$(date +%T) round $r: answer-bits failed: $(printf '%s\n' "$res" | tail -n 1 | cut -c1-160)" >> "$log"; return; fi
	set -- $all; local A=$1 B=$2 Q=$3 M=$4 SE=$5
	set -- $first; local K=$1 Q5=$2
	set -- $easier
	echo "$(date +%T) round $r: answer loss $A b/t with the true question, $B swapped: the question gives $Q b/t (per answer $M +- $SE); first $K tokens $Q5 b/t; easier with the true question $1 of $2" >> "$log"
}
while true; do
	todo=()
	for c in $(ls "$dir"/round-*-ckpt.bin 2>/dev/null | sed -E 's/.*round-([0-9]+)-ckpt\.bin/\1/' | sort -n); do
		[ "$c" -ge "$from" ] || continue
		[ -f "$dir/round-$c-ckpt-gates.f32" ] && [ -f "$dir/round-$c-tables.f32" ] || continue
		/usr/bin/grep -q -E "^[0-9:]+ round $c: " "$log" && continue
		todo+=("$c")
	done
	if [ ${#todo[@]} -eq 0 ]; then [ "$once" -eq 1 ] && exit 0; sleep 60; continue; fi
	for ((i = 0; i < ${#todo[@]}; i += jobs)); do
		for ((j = i; j < i + jobs && j < ${#todo[@]}; j++)); do one "${todo[$j]}" & done
		wait
	done
done
