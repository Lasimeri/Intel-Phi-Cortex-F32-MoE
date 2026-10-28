#!/bin/bash
# live-moe-gate.sh: the live trainer's gate for MoE stage 2 (docs/LIVE.md, PLAN-MOE.md "Stage 2 build"): the daemon
# training the heads and the pool from a checkpoint (coach --moe) must reproduce the batch trainer (cluster-gate-moe
# run) exactly: the per-report losses and the final checkpoint (the dense format with the home experts in the MLP
# regions). Run from the MoE tree with the card free of the daemon (cortex --stop first): the reference runs alone,
# then the script starts the daemon from build/card/cortexd (DMA channel 3) and runs the live half, and leaves the
# daemon serving the live MoE model.
#   bash tools/live-moe-gate.sh CKPT SLICE.u32 VALID.u32 [BIAS [EXPERT_LR]]
# CKPT = the starting dense checkpoint, SLICE.u32 = 64,001 tokens (2,000 chunks), VALID.u32 a validation stream, both
# little-endian u32 ids (4 bytes a token; a .u16 of before 2026-10-28 06:1x is widened by tools/ids32.c first);
# the copies go to /tmp/cortex on the card. BIAS (default 100) and EXPERT_LR (default 0.05) are the run's knobs.
set -u
cd "$(dirname "$0")/.." || exit 1
ckpt=$1; slice=$2; valid=$3; bias=${4:-100}; exlr=${5:-0.05}
log=results/live-moe-gate.log
out=results/live-moe-gate
mkdir -p "$out"
: > "$log"
say() { echo "$@" | tee -a "$log"; }
losses_of() { /usr/bin/grep -aE "^.{8} report chunk" "$1" | awk '{print $4, $8}'; }

say "=== live MoE gate $(date '+%F %T'): $ckpt, slice $slice (2,000 chunks), home bonus $bias, expert lr $exlr"
say "--- the card: the batch binary and the data"
phi put build/card/cluster-gate-moe /tmp/cortex/cluster-gate-moe | tail -1
phi put "$slice" /tmp/cortex/lm-slice.u32 | tail -1
phi put "$valid" /tmp/cortex/lm-valid.u32 | tail -1
phi put "$ckpt" /tmp/cortex/lm-init.bin | tail -1

say "--- reference: cluster-gate-moe run, the card without the daemon"
phi run sh -c "cd /tmp/cortex && rm -rf lmref && mkdir -p lmref && CX_BUS=0 CX_INIT=/tmp/cortex/lm-init.bin CX_MOE=1 CX_MOE_BIAS=$bias CX_MOE_LR=$exlr ./cluster-gate-moe run lm-slice.u32 lmref 32 64000 0.003 lm-valid.u32 3200 100 2>&1; md5sum /tmp/cortex/lmref/ckpt.bin" > "$out/ref.log" 2>&1
/usr/bin/grep -E "^run:|^moe:|^chunk +(100|1000|2000) |^valid:|^[0-9a-f]{32}" "$out/ref.log" | tee -a "$log"
/usr/bin/grep -E "^chunk" "$out/ref.log" | awk '{print $2, $6}' > "$out/ref.losses"
ref_md5=$(awk '/^[0-9a-f]{32}  / {print $1}' "$out/ref.log" | tail -1)
say "reference: $(wc -l < "$out/ref.losses") reports, final checkpoint md5 $ref_md5"

say "--- the daemon (build/card/cortexd, channel 3), then the live half: coach --ckpt $ckpt --moe --moe-bias $bias --expert-lr $exlr"
phi put build/card/cortexd /tmp/cortex/cortexd | tail -1
phi run sh -c 'cd /tmp/cortex && (CX_DMA_CHAN=3 setsid ./cortexd > cortexd.log 2>&1 < /dev/null &); sleep 8; grep -c serving cortexd.log' | tail -1
rm -rf "$out/live"; mkdir -p "$out/live"; : > "$out/live.log"
build/host/coach "$slice" --ckpt "$ckpt" --chunks 2000 --every 100 --sample-every 0 --ckdir "$out/live" --log "$out/live.log" --moe --moe-bias "$bias" --expert-lr "$exlr" > "$out/coach.out" 2>&1
/usr/bin/grep -aE "coach:|report chunk (100|1000|2000) |checkpoint \(final\)" "$out/live.log" | cut -c1-200 | tee -a "$log"
losses_of "$out/live.log" > "$out/live.losses"
if cmp -s "$out/ref.losses" "$out/live.losses"; then say "losses: PASSED, $(wc -l < "$out/live.losses") per-report losses identical to the batch trainer"; else say "losses: FAILED"; paste "$out/ref.losses" "$out/live.losses" | head -5 | tee -a "$log"; fi
live_md5=$(md5sum "$out/live/ckpt-2000.bin" 2>/dev/null | cut -d' ' -f1)
if [ -n "$live_md5" ] && [ "$live_md5" = "$ref_md5" ]; then say "checkpoint: PASSED, md5 $live_md5 = the batch trainer's (heads and home experts)"; else say "checkpoint: FAILED (live $live_md5, reference $ref_md5)"; fi
say "--- the daemon's log"
timeout 60 phi run sh -c 'grep -E "cortexd:|live:" /tmp/cortex/cortexd.log | tail -n 8' 2>&1 | cut -c1-200 | tee -a "$log"
say "=== gate done $(date '+%F %T'); the daemon serves the live MoE model"
