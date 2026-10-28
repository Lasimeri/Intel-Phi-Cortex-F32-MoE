#!/bin/bash
# live-ng-gate.sh: the live trainer's gate for the n-gram memory (the chunk layer, docs/LIVE.md): the daemon training
# the memory through the coach's ring thread must reproduce the batch trainer with host/ngserve.c exactly. Run from
# the MoE tree. The reference (cluster-gate with CX_NG=1 plus ngserve) needs the card free of the daemon, the live
# half needs the daemon serving; the script handles both: it runs the reference first (the daemon must be stopped
# before: cortex --stop), then starts the daemon from build/card/cortexd on DMA channel 3 and runs the live half.
#   bash tools/live-ng-gate.sh CKPT CARD_CKPT SLICE.u32 CARD_SLICE VALID.u32 CARD_VALID
# CKPT = the starting checkpoint on the host, CARD_CKPT its copy on the card (phi put first); SLICE.u32 = 64,001
# tokens (2,000 chunks of 32 plus the next), CARD_SLICE its copy; VALID.u32 a validation stream, CARD_VALID its copy
# (the batch trainer's validation pass takes 100 chunks of rows from the ring too). The streams and their card copies
# are little-endian u32 ids, 4 bytes a token (a .u16 of before 2026-10-28 06:1x is widened by tools/ids32.c first).
# Compared: the 20 per-report losses, the final checkpoint (md5), the gates (cmp) and the tables (cmp, 2 GiB).
set -u
cd "$(dirname "$0")/.." || exit 1
ckpt=$1; card_ckpt=$2; slice=$3; card_slice=$4; valid=$5; card_valid=$6
log=results/live-ng-gate.log
out=results/live-ng-gate
tables=/dev/shm/cortex-gate-tables.f32
mkdir -p "$out"
: > "$log"
say() { echo "$@" | tee -a "$log"; }
losses_of() { /usr/bin/grep -E "^.{8} report chunk" "$1" | awk '{print $4, $8}'; }

say "=== live memory gate $(date '+%F %T'): $ckpt, slice $slice (2,000 chunks), rows lr 0.1, gates from ones"
say "--- reference: cluster-gate run with CX_NG=1 and ngserve (zero tables, lr 0.1), the card without the daemon"
build/host/ngserve "$slice" 32 2000 --init zero --lr 0.1 --valid "$valid" 100 --save "$out/ref-tables.f32" > "$out/ref-ngserve.log" 2>&1 &
ngpid=$!
sleep 2
phi run sh -c "cd /tmp/cortex && rm -rf ngref && mkdir -p ngref && CX_BUS=0 CX_NG=1 CX_DMA_CHAN=3 CX_INIT=$card_ckpt ./cluster-gate run $card_slice /tmp/cortex/ngref 32 64000 0.003 $card_valid 3200 100 2>&1; md5sum /tmp/cortex/ngref/ckpt.bin; cat /tmp/cortex/ngref/ng-gates.f32 | base64 -w0" > "$out/ref.log" 2>&1
wait "$ngpid"
cat "$out/ref-ngserve.log" | tee -a "$log"
/usr/bin/grep -E "^chunk|valid:|n-gram" "$out/ref.log" | tee -a "$log"
/usr/bin/grep -E "^chunk" "$out/ref.log" | awk '{print $2, $6}' > "$out/ref.losses"
ref_md5=$(awk '/^[0-9a-f]{32}  / {print $1}' "$out/ref.log" | tail -1)
tail -1 "$out/ref.log" | base64 -d > "$out/ref-gates.f32" 2>/dev/null
say "reference: $(wc -l < "$out/ref.losses") reports, final checkpoint md5 $ref_md5, gates $(stat -c %s "$out/ref-gates.f32") B, tables $(stat -c %s "$out/ref-tables.f32") B"

say "--- the daemon back (build/card/cortexd, channel 3), then the live half: coach --ngram $tables --row-lr 0.1 --gates ones"
phi put build/card/cortexd /tmp/cortex/cortexd | tail -1
phi run sh -c 'cd /tmp/cortex && (CX_DMA_CHAN=3 setsid ./cortexd > cortexd.log 2>&1 < /dev/null &); sleep 6; grep -c serving cortexd.log' | tail -1
rm -f "$tables"
rm -rf "$out/live"; mkdir -p "$out/live"; : > "$out/live.log"
build/host/coach "$slice" --ckpt "$ckpt" --chunks 2000 --every 100 --sample-every 0 --ckdir "$out/live" --log "$out/live.log" --ngram "$tables" --row-lr 0.1 --gates ones --tables-every 100000 > "$out/coach.out" 2>&1
/usr/bin/grep -a -E "coach:|memory:|tables:|checkpoint \(final\)" "$out/live.log" | cut -c1-200 | tee -a "$log"
losses_of "$out/live.log" > "$out/live.losses"
if cmp -s "$out/ref.losses" "$out/live.losses"; then say "losses: PASSED, $(wc -l < "$out/live.losses") per-report losses identical to the batch trainer with ngserve"; else say "losses: FAILED"; paste "$out/ref.losses" "$out/live.losses" | head -5 | tee -a "$log"; fi
live_md5=$(md5sum "$out/live/ckpt-2000.bin" 2>/dev/null | cut -d' ' -f1)
if [ -n "$live_md5" ] && [ "$live_md5" = "$ref_md5" ]; then say "checkpoint: PASSED, md5 $live_md5 = the batch trainer's"; else say "checkpoint: FAILED (live $live_md5, reference $ref_md5)"; fi
if cmp -s "$out/ref-gates.f32" "$out/live/ckpt-2000-gates.f32"; then say "gates: PASSED, identical"; else say "gates: FAILED: $(cmp "$out/ref-gates.f32" "$out/live/ckpt-2000-gates.f32" 2>&1 | head -1)"; fi
if cmp -s "$out/ref-tables.f32" "$out/live/tables-2000.f32"; then say "tables: PASSED, the 2 GiB identical"; else say "tables: FAILED: $(cmp "$out/ref-tables.f32" "$out/live/tables-2000.f32" 2>&1 | head -1)"; fi
say "=== gate done $(date '+%F %T')"
