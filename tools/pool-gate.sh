#!/bin/bash
# pool-gate.sh: the MoE pool's way out of and back into the daemon (lib/cx-proto.h CX_T_POOL, docs/LIVE.md "The MoE
# model in the live loop"), and the host evaluator's reading of it (tools/ref-loss.c --moe). Run from the MoE tree
# with the daemon serving (build/card/cortexd of 2026-10-27 07:4x or later) and no run going.
#   bash tools/pool-gate.sh CKPT SLICE.u32
# SLICE.u32: a token stream of little-endian u32 ids, 4 bytes a token (a .u16 of before 2026-10-28 06:1x is widened by
# tools/ids32.c first).
# Steps (results/pool-gate.log, files in results/pool-gate/, 3 x 4 GiB):
#   0 a paused run stopped at once from CKPT with --moe: P0 = the pool as the LOAD filled it; the evaluator with P0 and
#     the bonus at 100 must equal the dense reading of CKPT (every head home: expert k = column k's MLP)
#   1 200 chunks at bonus 0 (free routing: thousands of experts change), the pool saved at the end: P1, ckpt-200
#   2 P1 pulled back with ckpt-200 (--pool), the run paused and stopped at once: the pool pushed again (P2) must be
#     byte-identical to P1, and the final checkpoint to ckpt-200 (the home experts come from the pulled pool)
#   3 100 more chunks from ckpt-200 with P1 at bonus 0: the losses continue where run 1 left them (sanity, not
#     identity: a LOAD starts a new Adam state); the evaluator reads ckpt-200 with P1 at bonus 0
set -u
cd "$(dirname "$0")/.." || exit 1
ckpt=$1; slice=$2
out=results/pool-gate
log=results/pool-gate.log
cmd=/tmp/claude-1000/-home-lasimeri/fa371527-28d7-4393-aaa6-ea22d2b1b983/scratchpad/pool-gate.cmd
mkdir -p "$out"
: > "$log"
say() { echo "$@" | tee -a "$log"; }
lines() { /usr/bin/grep -aE "report chunk|pool:|checkpoint \(final\)|coach: run ended|FAILED" "$1" | cut -c1-200 | tee -a "$log"; }

say "=== pool gate $(date '+%F %T'): $ckpt, slice $slice"
say "--- 0: the pool as the LOAD fills it (a paused run stopped at once), then the evaluator's identity"
rm -rf "$out/run0"; mkdir -p "$out/run0"; : > "$out/run0.log"; : > "$cmd"
( sleep 6; echo stop >> "$cmd" ) &
build/host/coach "$slice" --ckpt "$ckpt" --chunks 10 --every 1 --sample-every 0 --paused --cmd "$cmd" --ckdir "$out/run0" --log "$out/run0.log" --moe --moe-bias 100 --expert-lr 0.05 > "$out/run0.out" 2>&1
wait
lines "$out/run0.log"
p0=$(ls "$out"/run0/pool-*.f32 2>/dev/null | head -1)
say "dense reading of $ckpt on 1,000 tokens: $(build/host/ref-loss "$ckpt" "$slice" 1001 | tail -1 | sed 's/.*): //')"
say "MoE reading with P0 at bonus 100:        $(build/host/ref-loss "$ckpt" "$slice" 1001 --moe "$p0" --moe-bias 100 | tail -1 | sed 's/.*): //')"

say "--- 1: 200 chunks at bonus 0, the pool saved at the end (P1)"
rm -rf "$out/run1"; mkdir -p "$out/run1"; : > "$out/run1.log"
build/host/coach "$slice" --ckpt "$ckpt" --chunks 200 --every 100 --sample-every 0 --ckdir "$out/run1" --log "$out/run1.log" --moe --moe-bias 0 --expert-lr 0.05 > "$out/run1.out" 2>&1
lines "$out/run1.log"
p1=$out/run1/pool-200.f32
c1=$out/run1/ckpt-200.bin

say "--- 2: P1 pulled back with ckpt-200, the run paused and stopped at once, the pool pushed again (P2)"
rm -rf "$out/run2"; mkdir -p "$out/run2"; : > "$out/run2.log"; : > "$cmd"
( sleep 60; echo stop >> "$cmd" ) &
build/host/coach "$slice" --ckpt "$c1" --pool "$p1" --chunks 10 --every 1 --sample-every 0 --paused --cmd "$cmd" --ckdir "$out/run2" --log "$out/run2.log" --moe --moe-bias 0 --expert-lr 0.05 > "$out/run2.out" 2>&1
wait
lines "$out/run2.log"
p2=$(ls "$out"/run2/pool-*.f32 2>/dev/null | head -1)
c2=$(ls "$out"/run2/ckpt-*.bin 2>/dev/null | head -1)
if [ -n "$p2" ] && cmp -s "$p1" "$p2"; then say "round trip: PASSED, the pool pushed, pulled and pushed again is byte-identical (4,293,918,720 B)"; else say "round trip: FAILED ($(cmp "$p1" "$p2" 2>&1 | head -1))"; fi
if [ -n "$c2" ] && cmp -s "$c1" "$c2"; then say "checkpoint after the pull: PASSED, identical to ckpt-200 (the home experts from the pulled pool)"; else say "checkpoint after the pull: FAILED"; fi

say "--- 3: 100 more chunks from ckpt-200 with P1 at bonus 0 (continuity), and the evaluator on ckpt-200 with P1"
rm -rf "$out/run3"; mkdir -p "$out/run3"; : > "$out/run3.log"
build/host/coach "$slice" --ckpt "$c1" --pool "$p1" --start 200 --chunks 100 --every 50 --sample-every 0 --ckdir "$out/run3" --log "$out/run3.log" --moe --moe-bias 0 --expert-lr 0.05 > "$out/run3.out" 2>&1
lines "$out/run3.log"
say "dense reading of ckpt-200 on 1,000 tokens (its home experts): $(build/host/ref-loss "$c1" "$slice" 1001 | tail -1 | sed 's/.*): //')"
say "MoE reading of ckpt-200 with P1 at bonus 0:                   $(build/host/ref-loss "$c1" "$slice" 1001 --moe "$p1" --moe-bias 0 | tail -1 | sed 's/.*): //')"
rm -f "$p0" "$p2"
say "=== gate done $(date '+%F %T') (P1 kept: $p1)"
