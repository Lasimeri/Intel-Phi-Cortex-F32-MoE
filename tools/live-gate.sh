#!/bin/bash
# live-gate.sh: the live trainer's gates A to D (docs/LIVE.md). Run from the MoE tree with the card free of other
# runs. CKPT = the starting checkpoint on the host (the same file is on the card as CARD_CKPT); SLICE.u16 = the gate's
# token slice on the host (64,001 tokens: 2,000 chunks of 32 plus the next token), CARD_SLICE its copy on the card.
#   bash tools/live-gate.sh CKPT CARD_CKPT SLICE.u16 CARD_SLICE
set -u
cd "$(dirname "$0")/.." || exit 1
ckpt=$1; card_ckpt=$2; slice=$3; card_slice=$4
ssh=tools/card-ssh.sh
log=results/live-gate.log
cmd=/tmp/claude-1000/-home-lasimeri/fa371527-28d7-4393-aaa6-ea22d2b1b983/scratchpad/live-gate.cmd
: > "$log"
say() { echo "$@" | tee -a "$log"; }

say "=== live gates $(date '+%F %T'): $ckpt, slice $slice (2,000 chunks)"
say "--- reference: cluster-gate run from the checkpoint on the slice, EVERY 100 (the batch trainer)"
phi run sh -c "cd /tmp/cortex && mkdir -p gateA && CX_BUS=0 CX_INIT=$card_ckpt ./cluster-gate run $card_slice /tmp/cortex/gateA 32 64000 0.003 /tmp/cortex/valid.u16 3200 100 2>&1" > results/live-gate-ref.log 2>&1
/usr/bin/grep -E "^chunk" results/live-gate-ref.log | awk '{print $2, $6}' > results/live-gate-ref.losses
say "reference reports: $(wc -l < results/live-gate-ref.losses); last: $(tail -1 results/live-gate-ref.losses)"

say "--- the daemon up"
bash "$ssh" 'cd /tmp/cortex && CX_DMA_CHAN=3 setsid ./cortexd >> cortexd.log 2>&1 < /dev/null & sleep 4; tail -1 /tmp/cortex/cortexd.log' | tee -a "$log"

say "--- gate A: the live trainer alone on the same chunks"
build/host/coach "$slice" --ckpt "$ckpt" --chunks 2000 --every 100 --sample-every 0 --log results/live-gate-A.log > /dev/null 2>&1
/usr/bin/grep -E "^.{8} report chunk" results/live-gate-A.log | awk '{print $4, $10}' > results/live-gate-A.losses
if cmp -s results/live-gate-ref.losses results/live-gate-A.losses; then say "gate A PASSED: $(wc -l < results/live-gate-A.losses) per-report losses identical to the batch trainer"; else say "gate A FAILED: losses differ"; paste results/live-gate-ref.losses results/live-gate-A.losses | head -5 | tee -a "$log"; fi

say "--- gate B: the same with a client generating 3 x 512 tokens on the live weights meanwhile"
( sleep 6; for s in 1 2 3; do build/host/cortex --live "Once upon a time" 512 --seed $s > results/live-gate-B-s$s.txt 2> results/live-gate-B-s$s.err; done ) &
build/host/coach "$slice" --ckpt "$ckpt" --chunks 2000 --every 100 --sample-every 0 --log results/live-gate-B.log > /dev/null 2>&1
wait
/usr/bin/grep -E "^.{8} report chunk" results/live-gate-B.log | awk '{print $4, $10}' > results/live-gate-B.losses
if cmp -s results/live-gate-ref.losses results/live-gate-B.losses; then say "gate B PASSED: losses identical with a client generating between chunks"; else say "gate B FAILED: losses differ"; paste results/live-gate-ref.losses results/live-gate-B.losses | head -5 | tee -a "$log"; fi
for s in 1 2 3; do say "client seed $s: $(/usr/bin/grep -oE '[0-9.]+ tok/s overall.*' results/live-gate-B-s$s.err | cut -c1-120)"; done
say "client text (seed 1): $(head -c 300 results/live-gate-B-s1.txt | tr '\n' ' ')"

say "--- gate C: the knobs (scale 0.5 after report 3, pause 5 s, resume, save, stop)"
: > "$cmd"
( sleep 12; echo "scale 0.5" >> "$cmd"; sleep 8; echo "pause" >> "$cmd"; sleep 5; echo "resume" >> "$cmd"; sleep 4; echo "save" >> "$cmd"; sleep 4; echo "stop" >> "$cmd" ) &
build/host/coach "$slice" --ckpt "$ckpt" --chunks 2000 --every 100 --sample-every 0 --cmd "$cmd" --log results/live-gate-C.log > /dev/null 2>&1
wait
/usr/bin/grep -E "command:|report chunk|checkpoint|run ended" results/live-gate-C.log | cut -c1-150 | tee -a "$log"
bash "$ssh" 'ls -la /tmp/cortex/live/; grep -E "live:|session" /tmp/cortex/cortexd.log | tail -8' | tee -a "$log"
say "=== gates done $(date '+%F %T')"
