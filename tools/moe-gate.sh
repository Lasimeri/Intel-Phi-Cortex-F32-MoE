#!/bin/bash
# moe-gate.sh: MoE stage 2's gates on the card (PLAN-MOE.md "Stage 2 build"). Run from the MoE tree with the card
# free of the daemon (cortex --stop first; the batch trainer runs alone). Every card access goes over the control
# socket (phi put, phi run). SLICE.u16 is the gate's token slice on the host (64,001 ids), VALID.u16 a validation
# stream, CKPT a dense checkpoint for the short run; their copies land in /tmp/cortex on the card.
#   bash tools/moe-gate.sh SLICE.u16 VALID.u16 CKPT
# Gates, in order (results/moe-gate.log, dumps in results/moe-gate/):
#   A conversion  the MoE build with the home bonus on the same seeded weights: the chunk's losses, hc and the dumped
#                 columns' u/bc/g byte-identical to the dense build's (cmp), every route the home expert
#   B reference   bwd: the card against tools/cref.h in MoE mode (routes, losses, hc, the head gradients, cg, E, and
#                 the experts of step 0 after the chunk's SGD updates), with the home bonus and without it (free
#                 routing: the keys decide, collisions handled by the owner rule)
#   C cost        the forward and backward microseconds per step of the dense build and the MoE build
#   D run         2,000 chunks from CKPT with the pool, home bonus 8 then 0: the loss curve, tok/s, the usage report
set -u
cd "$(dirname "$0")/.." || exit 1
slice=$1; valid=$2; ckpt=$3
log=results/moe-gate.log
out=results/moe-gate
mkdir -p "$out"
: > "$log"
say() { echo "$@" | tee -a "$log"; }

say "=== MoE stage 2 gates $(date '+%F %T'): slice $slice, valid $valid, ckpt $ckpt"
say "--- the card: binaries and data into /tmp/cortex"
phi put build/card/cluster-gate /tmp/cortex/cluster-gate | tail -1
phi put build/card/cluster-gate-moe /tmp/cortex/cluster-gate-moe | tail -1
phi put "$slice" /tmp/cortex/moe-slice.u16 | tail -1
phi put "$valid" /tmp/cortex/moe-valid.u16 | tail -1
phi put "$ckpt" /tmp/cortex/moe-init.bin | tail -1

say "--- gate A: conversion (fwd, warm 64, one chunk of 32, the bus on as the reference runs it: dense build, then the MoE build with home bonus 8)"
phi run sh -c 'cd /tmp/cortex && rm -rf gA-dense gA-moe && mkdir -p gA-dense gA-moe && ./cluster-gate fwd moe-slice.u16 gA-dense 32 64 0 59 119 2>&1; CX_MOE=1 CX_MOE_BIAS=8 ./cluster-gate-moe fwd moe-slice.u16 gA-moe 32 64 0 59 119 2>&1; cd gA-dense && md5sum cl-loss.f32 cl-hc.f32 cl-u0.f32 cl-u59.f32 cl-u119.f32; cd ../gA-moe && md5sum cl-loss.f32 cl-hc.f32 cl-u0.f32 cl-u59.f32 cl-u119.f32; tar cf - cl-route.f32 cl-loss.f32 cl-hc.f32 cl-u0.f32 cl-u59.f32 cl-u119.f32 | base64 -w0' > "$out/gateA.txt" 2>&1
/usr/bin/grep -E 'cluster-gate|chunk:|moe:|pmu' "$out/gateA.txt" | tee -a "$log"
dense_md5=$(/usr/bin/grep -E '^[0-9a-f]{32}  ' "$out/gateA.txt" | head -5 | awk '{print $1}' | tr '\n' ' ')
moe_md5=$(/usr/bin/grep -E '^[0-9a-f]{32}  ' "$out/gateA.txt" | sed -n 6,10p | awk '{print $1}' | tr '\n' ' ')
if [ -n "$dense_md5" ] && [ "$dense_md5" = "$moe_md5" ]; then say "gate A: PASSED, the MoE build with the home bonus is byte-identical to the dense build (loss, hc, u/bc/g of columns 0, 59, 119): $moe_md5"; else say "gate A: FAILED: dense $dense_md5 moe $moe_md5"; fi
mkdir -p "$out/gA-moe" && tail -1 "$out/gateA.txt" | base64 -d 2>/dev/null | tar xf - -C "$out/gA-moe" 2>/dev/null
say "reference check of the converted model's forward (home bonus 8):"
CX_MOE=1 CX_MOE_BIAS=8 build/host/ctrain-check fwd "$out/gA-moe" "$slice" 32 64 0 59 119 2>&1 | tee -a "$log"

say "--- gate B: the backward against the double reference (MoE build; home bonus 8, then 0 = free routing), expert lr 0.05"
for bias in 8 0; do
	phi run sh -c "cd /tmp/cortex && rm -rf gB-$bias && mkdir -p gB-$bias && CX_MOE=1 CX_MOE_BIAS=$bias CX_MOE_LR=0.05 ./cluster-gate-moe bwd moe-slice.u16 gB-$bias 32 64 0 59 119 2>&1; cd gB-$bias && tar cf - cl-*.f32 | base64 -w0" > "$out/gateB-$bias.txt" 2>&1
	/usr/bin/grep -E 'cluster-gate|chunk:|backward|moe:' "$out/gateB-$bias.txt" | tee -a "$log"
	mkdir -p "$out/gB-$bias" && tail -1 "$out/gateB-$bias.txt" | base64 -d 2>/dev/null | tar xf - -C "$out/gB-$bias" 2>/dev/null
	say "reference check, home bonus $bias:"
	CX_MOE=1 CX_MOE_BIAS=$bias CX_MOE_LR=0.05 build/host/ctrain-check bwd "$out/gB-$bias" "$slice" 32 64 0 59 119 2>&1 | tee -a "$log"
done

say "--- gate C: cost per step is in the chunk: lines above (dense: gate A's first run; MoE: gate B's runs)"

say "--- gate D: 2,000 chunks from $ckpt with the pool (bus off, the trained configuration), validation on 3,200 tokens:"
say "    home bonus 100 (exact home routing) at expert lr 0.05, 0.01 and 0 (experts frozen), bonus 8 and 0 (free routing) at 0.05"
for cfg in "100 0.05" "100 0.01" "100 0" "8 0.05" "0 0.05"; do
	set -- $cfg
	bias=$1; exlr=$2
	phi run sh -c "cd /tmp/cortex && rm -rf gD-$bias-$exlr && mkdir -p gD-$bias-$exlr && CX_BUS=0 CX_INIT=/tmp/cortex/moe-init.bin CX_MOE=1 CX_MOE_BIAS=$bias CX_MOE_LR=$exlr ./cluster-gate-moe run moe-slice.u16 gD-$bias-$exlr 32 64000 0.003 moe-valid.u16 3200 200 2>&1" > "$out/gateD-$bias-$exlr.txt" 2>&1
	say "run, home bonus $bias, expert lr $exlr:"
	/usr/bin/grep -E 'run:|moe:|^chunk|valid:' "$out/gateD-$bias-$exlr.txt" | tee -a "$log"
done
say "--- the dense run on the same slice for the comparison"
phi run sh -c "cd /tmp/cortex && rm -rf gD-dense && mkdir -p gD-dense && CX_BUS=0 CX_INIT=/tmp/cortex/moe-init.bin ./cluster-gate run moe-slice.u16 gD-dense 32 64000 0.003 moe-valid.u16 3200 200 2>&1" > "$out/gateD-dense.txt" 2>&1
/usr/bin/grep -E 'run:|^chunk|valid:' "$out/gateD-dense.txt" | tee -a "$log"
say "=== gates done $(date '+%F %T')"
