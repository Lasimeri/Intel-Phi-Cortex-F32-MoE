#!/bin/bash
# live-gate.sh: the live trainer's gates A to C (docs/LIVE.md). Run from the MoE tree with cortexd serving on the
# card and no other card run. Every card access goes over the control socket (phi run); the ssh forward is not used.
# CKPT = the starting checkpoint on the host, CARD_CKPT the same file on the card; SLICE.u16 = the gate's token slice
# on the host (64,001 tokens: 2,000 chunks of 32 plus the next token), CARD_SLICE its copy on the card.
#   bash tools/live-gate.sh CKPT CARD_CKPT SLICE.u16 CARD_SLICE
# The reference (cluster-gate run on the card: the batch trainer from the same checkpoint on the same tokens, its final
# checkpoint's md5 printed) must run with the card free, so it is run here only when results/live-gate-ref.log does
# not hold it yet; otherwise the existing log is used.
set -u
cd "$(dirname "$0")/.." || exit 1
ckpt=$1; card_ckpt=$2; slice=$3; card_slice=$4
log=results/live-gate.log
cmd=/tmp/claude-1000/-home-lasimeri/fa371527-28d7-4393-aaa6-ea22d2b1b983/scratchpad/live-gate.cmd
: > "$log"
say() { echo "$@" | tee -a "$log"; }
losses_of() { /usr/bin/grep -E "^.{8} report chunk" "$1" | awk '{print $4, $8}'; }

say "=== live gates $(date '+%F %T'): $ckpt, slice $slice (2,000 chunks)"
say "--- reference: cluster-gate run from the checkpoint on the slice, EVERY 100 (the batch trainer)"
if ! /usr/bin/grep -q "gateA/ckpt.bin" results/live-gate-ref.log 2>/dev/null; then
	phi run sh -c "cd /tmp/cortex && mkdir -p gateA && CX_BUS=0 CX_INIT=$card_ckpt ./cluster-gate run $card_slice /tmp/cortex/gateA 32 64000 0.003 /tmp/cortex/valid.u16 3200 100 2>&1; md5sum /tmp/cortex/gateA/ckpt.bin $card_ckpt" > results/live-gate-ref.log 2>&1
fi
/usr/bin/grep -E "^chunk" results/live-gate-ref.log | awk '{print $2, $6}' > results/live-gate-ref.losses
ref_md5=$(awk '/^[0-9a-f]{32}  .*gateA\/ckpt.bin$/ {print $1}' results/live-gate-ref.log)
say "reference reports: $(wc -l < results/live-gate-ref.losses); last: $(tail -1 results/live-gate-ref.losses); final checkpoint md5 $ref_md5"

say "--- gate A: the live trainer alone on the same chunks (losses per report, then the final checkpoint's bytes)"
rm -rf results/live-gate-A; mkdir -p results/live-gate-A; : > results/live-gate-A.log
build/host/coach "$slice" --ckpt "$ckpt" --chunks 2000 --every 100 --sample-every 0 --ckdir results/live-gate-A --log results/live-gate-A.log > /dev/null 2>&1
losses_of results/live-gate-A.log > results/live-gate-A.losses
if cmp -s results/live-gate-ref.losses results/live-gate-A.losses; then say "gate A losses: PASSED, $(wc -l < results/live-gate-A.losses) per-report losses identical to the batch trainer"; else say "gate A losses: FAILED"; paste results/live-gate-ref.losses results/live-gate-A.losses | head -5 | tee -a "$log"; fi
a_md5=$(md5sum results/live-gate-A/ckpt-2000.bin 2>/dev/null | cut -d' ' -f1)
if [ -n "$a_md5" ] && [ "$a_md5" = "$ref_md5" ]; then say "gate A checkpoint: PASSED, results/live-gate-A/ckpt-2000.bin md5 $a_md5 = the batch trainer's"; else say "gate A checkpoint: FAILED (live $a_md5, reference $ref_md5)"; fi
/usr/bin/grep -E "checkpoint" results/live-gate-A.log | cut -c1-160 | tee -a "$log"

say "--- gate B: the same with a client generating 3 x 512 tokens on the live weights meanwhile"
rm -rf results/live-gate-B; mkdir -p results/live-gate-B; : > results/live-gate-B.log
( sleep 6; for s in 1 2 3; do build/host/cortex --live "Once upon a time" 512 --seed $s > results/live-gate-B-s$s.txt 2> results/live-gate-B-s$s.err; done ) &
build/host/coach "$slice" --ckpt "$ckpt" --chunks 2000 --every 100 --sample-every 0 --ckdir results/live-gate-B --log results/live-gate-B.log > /dev/null 2>&1
wait
losses_of results/live-gate-B.log > results/live-gate-B.losses
if cmp -s results/live-gate-ref.losses results/live-gate-B.losses; then say "gate B losses: PASSED, identical with a client generating between chunks"; else say "gate B losses: FAILED"; paste results/live-gate-ref.losses results/live-gate-B.losses | head -5 | tee -a "$log"; fi
b_md5=$(md5sum results/live-gate-B/ckpt-2000.bin 2>/dev/null | cut -d' ' -f1)
if [ -n "$b_md5" ] && [ "$b_md5" = "$ref_md5" ]; then say "gate B checkpoint: PASSED, md5 $b_md5 = the batch trainer's"; else say "gate B checkpoint: FAILED (live $b_md5, reference $ref_md5)"; fi
for s in 1 2 3; do say "client seed $s: $(/usr/bin/grep -oE '[0-9.]+ tok/s overall.*' results/live-gate-B-s$s.err | cut -c1-120)"; done
say "client text (seed 1): $(head -c 300 results/live-gate-B-s1.txt | tr '\n' ' ')"

say "--- gate C: the knobs (scale 0.5 after report 3, pause 5 s, resume, save, stop)"
rm -rf results/live-gate-C; mkdir -p results/live-gate-C; : > results/live-gate-C.log
: > "$cmd"
( sleep 12; echo "scale 0.5" >> "$cmd"; sleep 8; echo "pause" >> "$cmd"; sleep 5; echo "resume" >> "$cmd"; sleep 4; echo "save" >> "$cmd"; sleep 4; echo "stop" >> "$cmd" ) &
build/host/coach "$slice" --ckpt "$ckpt" --chunks 2000 --every 100 --sample-every 0 --cmd "$cmd" --ckdir results/live-gate-C --log results/live-gate-C.log > /dev/null 2>&1
wait
/usr/bin/grep -E "command:|report chunk|checkpoint|run ended" results/live-gate-C.log | cut -c1-160 | tee -a "$log"
say "gate C checkpoints: $(ls -la results/live-gate-C/ | /usr/bin/grep ckpt | awk '{print $5, $9}' | tr '\n' ';')"
say "--- the daemon's log (control socket)"
timeout 60 phi run sh -c 'grep -E "live:|session" /tmp/cortex/cortexd.log | tail -n 12' 2>&1 | tee -a "$log"
say "=== gates done $(date '+%F %T')"
