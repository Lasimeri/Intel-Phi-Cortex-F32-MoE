#!/bin/bash
# base-format.sh: the chat base re-written in the plain format (tools/chat-text --no-think-block --detok: no think
# block in the answers, DailyDialog's spaced punctuation and apostrophes rejoined; 2026-10-27, "the dataset while
# training needs to be properly formatted for the model to respond as a human would") from the same mix.jsonl in
# the same order, the same 95/5 split by conversations, encoded with the SAME vocabulary
# (data/chat-base/vocab-base4096.txt) so a run continues from its checkpoint; the floors re-measured on the new
# held-out tokens. Everything lands in data/chat-base2/. Where a run on the old stream continues in the new one:
# tools/conv-pos.c.
#   tools/base-format.sh
set -eu
cd "$(dirname "$0")/.."
S=data/chat-base; D=data/chat-base2; mkdir -p $D
echo "=== the plain format ($(date +%T))"
build/host/chat-text $S/mix.jsonl $D/mix.txt --max-chars 16000 --no-think-block --detok 2>&1 | tail -n 3
n=$(/usr/bin/grep -c -x '<|endoftext|>' $D/mix.txt); v=$((n / 20))
awk -v keep=$((n - v)) 'BEGIN { c = 0 } { if (c < keep) print > "'$D'/train.txt"; else print > "'$D'/valid.txt"; if ($0 == "<|endoftext|>") c++ }' $D/mix.txt
echo "conversations $n, held out $v (the old split: $(/usr/bin/grep -c -x '<|endoftext|>' $S/train.txt) + $(/usr/bin/grep -c -x '<|endoftext|>' $S/valid.txt)); train $(stat -c %s $D/train.txt) B, valid $(stat -c %s $D/valid.txt) B"
echo "=== the encode with the same vocabulary ($(date +%T))"
cp $S/vocab-base4096.txt $D/vocab-base4096.txt
cp $S/persona.txt $D/persona.txt
build/host/bpe encode $D/vocab-base4096.txt $D/train.txt $D/train.u16 2>&1 | tail -n 1 | cut -c1-200
build/host/bpe encode $D/vocab-base4096.txt $D/valid.txt $D/valid.u16 2>&1 | tail -n 1 | cut -c1-200
build/host/bpe decode $D/vocab-base4096.txt $D/valid.u16 $D/valid.dec > /dev/null 2>&1 && cmp $D/valid.txt $D/valid.dec && echo "round trip byte-identical"; rm -f $D/valid.dec
echo "train tokens $(( $(stat -c %s $D/train.u16) / 2 )) (was $(( $(stat -c %s $S/train.u16) / 2 ))), chunks of 32: $(( $(stat -c %s $D/train.u16) / 64 )); valid tokens $(( $(stat -c %s $D/valid.u16) / 2 )) (was $(( $(stat -c %s $S/valid.u16) / 2 )))"
echo "=== marker ids in train.u16 (4095 eot, 4094 system, 4093 user, 4092 assistant, 4091 end, 4090 think, 4089 /think)"
od -An -v -t u2 -w2 $D/train.u16 | awk '$1 >= 4089 { c[$1]++ } END { for (k in c) printf("%s %d\n", k, c[k]) }' | sort -rn
echo "=== the floors on valid.u16[0..27447) ($(date +%T))"
build/host/ngram-base $D/train.u16 $D/valid.u16 27447 31553344 | tee $D/floors.txt
echo "=== done $(date +%T)"
