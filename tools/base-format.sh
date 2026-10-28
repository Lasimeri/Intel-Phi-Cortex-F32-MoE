#!/bin/bash
# base-format.sh: the chat base re-written in the plain format (tools/chat-text --no-think-block --detok: no think
# block in the answers, DailyDialog's spaced punctuation and apostrophes rejoined; 2026-10-27, "the dataset while
# training needs to be properly formatted for the model to respond as a human would") from the same mix.jsonl in
# the same order, the same 95/5 split by conversations, encoded with the SAME vocabulary
# (data/chat-base/vocab-base4096.txt) so a run continues from its checkpoint; the floors re-measured on the new
# held-out tokens. Everything lands in data/chat-base2/. Where a run on the old stream continues in the new one:
# tools/conv-pos.c.
#   tools/base-format.sh
# The streams are .u32 files of little-endian u32 ids, 4 bytes a token, 128 bytes a chunk of 32 (lib/cx-proto.h
# cx_tok; tools/bpe.c encode writes them from 2026-10-28 06:1x, .u16 before). The old base's counts (the "was" figures)
# read data/chat-base/STEM.u32 once it is widened (tools/ids32.c) and its .u16 of before the change until then.
set -eu
cd "$(dirname "$0")/.."
S=data/chat-base; D=data/chat-base2; mkdir -p $D
# the tokens of the old base's stream STEM: 4 bytes an id in STEM.u32, 2 in a STEM.u16 of before the u32 ids (read only)
ntok_old() { if [ -f "$1.u32" ]; then echo $(( $(stat -c %s "$1.u32") / 4 )); else echo $(( $(stat -c %s "$1.u16") / 2 )); fi; }
echo "=== the plain format ($(date +%T))"
build/host/chat-text $S/mix.jsonl $D/mix.txt --max-chars 16000 --no-think-block --detok 2>&1 | tail -n 3
n=$(/usr/bin/grep -c -x '<|endoftext|>' $D/mix.txt); v=$((n / 20))
awk -v keep=$((n - v)) 'BEGIN { c = 0 } { if (c < keep) print > "'$D'/train.txt"; else print > "'$D'/valid.txt"; if ($0 == "<|endoftext|>") c++ }' $D/mix.txt
echo "conversations $n, held out $v (the old split: $(/usr/bin/grep -c -x '<|endoftext|>' $S/train.txt) + $(/usr/bin/grep -c -x '<|endoftext|>' $S/valid.txt)); train $(stat -c %s $D/train.txt) B, valid $(stat -c %s $D/valid.txt) B"
echo "=== the encode with the same vocabulary ($(date +%T))"
cp $S/vocab-base4096.txt $D/vocab-base4096.txt
cp $S/persona.txt $D/persona.txt
build/host/bpe encode $D/vocab-base4096.txt $D/train.txt $D/train.u32 2>&1 | tail -n 1 | cut -c1-200
build/host/bpe encode $D/vocab-base4096.txt $D/valid.txt $D/valid.u32 2>&1 | tail -n 1 | cut -c1-200
build/host/bpe decode $D/vocab-base4096.txt $D/valid.u32 $D/valid.dec > /dev/null 2>&1 && cmp $D/valid.txt $D/valid.dec && echo "round trip byte-identical"; rm -f $D/valid.dec
echo "train tokens $(( $(stat -c %s $D/train.u32) / 4 )) (was $(ntok_old $S/train)), chunks of 32: $(( $(stat -c %s $D/train.u32) / 128 )); valid tokens $(( $(stat -c %s $D/valid.u32) / 4 )) (was $(ntok_old $S/valid))"
echo "=== marker ids in train.u32 (4095 eot, 4094 system, 4093 user, 4092 assistant, 4091 end, 4090 think, 4089 /think)"
od -An -v -t u4 -w4 $D/train.u32 | awk '$1 >= 4089 { c[$1]++ } END { for (k in c) printf("%s %d\n", k, c[k]) }' | sort -rn
echo "=== the floors on valid.u32[0..27447) ($(date +%T))"
build/host/ngram-base $D/train.u32 $D/valid.u32 27447 31553344 | tee $D/floors.txt
echo "=== done $(date +%T)"
