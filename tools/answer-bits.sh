#!/bin/bash
# answer-bits.sh: the bits a checkpoint's answers draw from the person's question (tools/answer-bits.c; docs/REASON.md).
# The held-out conversations (the batches the driver sent to valid.u16: every VALID_EVERY-th line of
# DIR/work/encoded.txt, the newest first, up to --convs conversations) are written twice: A as they are, B with every
# person's line replaced by the person's line of the same turn in the conversation --shift places later (cyclic; the
# first person's line of that conversation when it has fewer turns). Both go through the stream's own conversion
# (tools/chat-text --no-think-block --detok) and encoding (the instance's vocabulary), are read by tools/ref-loss with
# the n-gram memory and the context memory (--dump), and tools/answer-bits pairs the speaker's tokens.
#   tools/answer-bits.sh DIR CKPT GATES TABLES [--convs N] [--shift S] [--valid-every K] [--out PREFIX] [--set FILE]
# DIR: the instance (data/live-teacher2); CKPT, GATES, TABLES: a round's checkpoint, its 1,536-B gates and its tables
# (round-R-ckpt.bin, round-R-ckpt-gates.f32, round-R-tables.f32, or the live /dev/shm tables right after a round).
# --set FILE: the conversations measured (default DIR/answer-bits-set.jsonl): written once from the held-out batches
# when missing and reused, so every round is measured on the same conversations (the series compares like with like;
# the set of 2026-10-28 05:10 is the 60 newest held-out conversations then, the round-16 baseline's).
# Defaults: --convs 60, --shift 7, --valid-every 20, --out DIR/work/abits. Prints answer-bits' lines.
set -u
cd "$(dirname "$0")/.." || exit 1
[ $# -ge 4 ] || { echo "usage: tools/answer-bits.sh DIR CKPT GATES TABLES [--convs N] [--shift S] [--valid-every K] [--out PREFIX]" >&2; exit 2; }
dir=$1; ckpt=$2; gates=$3; tables=$4
shift 4
convs=60; sh=7; ve=20; out=""; set_file=""
while [ $# -gt 0 ]; do
	case "$1" in
	--set) set_file=$2; shift 2 ;;
	--convs) convs=$2; shift 2 ;;
	--shift) sh=$2; shift 2 ;;
	--valid-every) ve=$2; shift 2 ;;
	--out) out=$2; shift 2 ;;
	*) echo "answer-bits: unknown option $1" >&2; exit 2 ;;
	esac
done
[ -n "$out" ] || out=$dir/work/abits
vocab=$dir/vocab-teacher4096.txt
enc=$dir/work/encoded.txt
[ -s "$enc" ] && [ -s "$vocab" ] || { echo "answer-bits: $enc or $vocab missing" >&2; exit 2; }
# the measured set: the held-out batches, newest first, until --convs conversations, frozen in --set on first use
[ -n "$set_file" ] || set_file=$dir/answer-bits-set.jsonl
if [ ! -s "$set_file" ]; then
	awk -v k="$ve" 'NR % k == 0' "$enc" | tac | while IFS= read -r f; do [ -f "$f" ] && cat "$f"; done | head -n "$convs" > "$set_file"
fi
cp "$set_file" "$out-A.jsonl"
n=$(wc -l < "$out-A.jsonl")
[ "$n" -gt "$sh" ] || { echo "answer-bits: only $n held-out conversations" >&2; exit 2; }
jq -s -c --argjson s "$sh" '. as $all | ($all | length) as $n | [range(0; $n)] | map(. as $i | $all[$i] as $c | $all[($i + $s) % $n] as $o
	| ($o.messages | map(select(.role == "user"))) as $ou
	| $c | .messages |= ([foreach .[] as $m (-1; if $m.role == "user" then . + 1 else . end;
		if $m.role == "user" then { role: "user", content: (($ou[.] // $ou[0]).content) } else $m end)])) | .[]' "$out-A.jsonl" > "$out-B.jsonl"
for v in A B; do
	build/host/chat-text "$out-$v.jsonl" "$out-$v.txt" --no-think-block --detok > /dev/null 2>&1 || { echo "answer-bits: chat-text failed on $v" >&2; exit 1; }
	build/host/bpe encode "$vocab" "$out-$v.txt" "$out-$v.u16" > /dev/null 2>&1 || { echo "answer-bits: encoding failed on $v" >&2; exit 1; }
	nt=$(( $(stat -c %s "$out-$v.u16") / 2 ))
	build/host/ref-loss "$ckpt" "$out-$v.u16" "$nt" --ngram "$tables" "$gates" --ctx --dump "$out-$v.f32" > "$out-$v.ref" 2>&1 || { echo "answer-bits: ref-loss failed on $v: $(tail -n 1 "$out-$v.ref")" >&2; exit 1; }
done
echo "answer-bits: $n held-out conversations, swapped by $sh; A $(( $(stat -c %s "$out-A.u16") / 2 )) tokens, B $(( $(stat -c %s "$out-B.u16") / 2 ))"
build/host/answer-bits "$out-A.u16" "$out-A.f32" "$out-B.u16" "$out-B.f32"
