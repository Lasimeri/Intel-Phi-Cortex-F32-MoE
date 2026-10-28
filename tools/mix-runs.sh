#!/bin/bash
# mix-runs.sh: three runs on the mixed stream (TinyStories + BabyLM, data/mix) from the run4off checkpoint, all with
# the bus off, L 32, lr 0.003 after the 50-chunk warm-up, validation on the first 100,000 tokens of data/mix/valid.u32:
#   Z  one chunk at lr 0, then validation: the starting point of the run4off model on the mixed validation stream
#   A  320,000 tokens, plain model
#   B  320,000 tokens with the n-gram memory: host tables (ngserve, zero init, SGD lr 0.1) over the DMA ring, channel 5
# Run from the MoE tree; the card must hold /tmp/cortex/{cluster-gate,run4off/ckpt.bin,mix/train.u32,mix/valid.u32}.
# The streams are little-endian u32 ids, 4 bytes a token (data/mix/*.u16 of before 2026-10-28 06:1x are widened by
# tools/ids32.c first, on the host and for the card's copies).
set -u
cd "$(dirname "$0")/.." || exit 1
pre='cd /tmp/cortex && mkdir -p mix/z mix/a mix/b && CX_BUS=0 CX_INIT=/tmp/cortex/run4off/ckpt.bin'

echo "=== Z: starting point (1 chunk at lr 0, then validation)"
phi run sh -c "$pre ./cluster-gate run /tmp/cortex/mix/train.u32 /tmp/cortex/mix/z 32 32 0 /tmp/cortex/mix/valid.u32 100000 250 2>&1; echo rc=\$?" 2>&1 | tee results/mix-z.log | cut -c1-200

echo "=== A: plain model, 320,000 tokens"
phi run sh -c "$pre ./cluster-gate run /tmp/cortex/mix/train.u32 /tmp/cortex/mix/a 32 320000 0.003 /tmp/cortex/mix/valid.u32 100000 1000 2>&1; echo rc=\$?" 2>&1 | tee results/mix-a.log | cut -c1-200

echo "=== B: n-gram memory, 320,000 tokens (ngserve on the host, DMA channel 5)"
build/host/ngserve data/mix/train.u32 32 10000 --init zero --lr 0.1 --valid data/mix/valid.u32 3125 --save data/mix/b-ngtables.f32 > results/mix-b-ngserve.log 2>&1 &
ngpid=$!
sleep 3
phi run sh -c "$pre CX_NG=1 CX_DMA_CHAN=5 ./cluster-gate run /tmp/cortex/mix/train.u32 /tmp/cortex/mix/b 32 320000 0.003 /tmp/cortex/mix/valid.u32 100000 1000 2>&1; echo rc=\$?" 2>&1 | tee results/mix-b.log | cut -c1-200
wait "$ngpid"
cat results/mix-b-ngserve.log
phi get /tmp/cortex/mix/b/ng-gates.f32 data/mix/b-ng-gates.f32 | tail -1
