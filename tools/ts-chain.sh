#!/bin/bash
# ts-chain.sh: after the 10M-token run (results/ts10m.log) ends: restart the daemon, fetch the run's checkpoints,
# score their coherence curve (tools/ckpt-eval.sh), stop the daemon, launch the 100M-token continuation from the
# final checkpoint (results/ts100m.log, checkpoints every 10M tokens). Run from the MoE tree, in the background.
# The card's streams (/tmp/cortex/train-next100m.u32, /tmp/cortex/valid.u32) are little-endian u32 ids, 4 bytes a token
# (their .u16 of before 2026-10-28 06:1x are widened by tools/ids32.c first).
set -u
cd "$(dirname "$0")/.." || exit 1
ssh=tools/card-ssh.sh
while ! /usr/bin/grep -q "^=== end" results/ts10m.log 2>/dev/null; do sleep 60; done
echo "=== chain: the 10M run ended at $(date '+%F %T')"
tail -3 results/ts10m.log
bash "$ssh" 'cd /tmp/cortex && CX_DMA_CHAN=3 setsid ./cortexd >> cortexd.log 2>&1 < /dev/null & sleep 4; tail -1 /tmp/cortex/cortexd.log'
mkdir -p data/ts10m
for c in ckpt-62500 ckpt-125000 ckpt-187500 ckpt-250000 ckpt; do
	bash "$ssh" "cat /tmp/cortex/ts10m/$c.bin" > "data/ts10m/$c.bin"
	ls -la "data/ts10m/$c.bin"
done
bash tools/ckpt-eval.sh data/stories-10m.ngc data/wordlist-stories.txt data/s4/run4off/ckpt.bin data/ts10m/ckpt-62500.bin data/ts10m/ckpt-125000.bin data/ts10m/ckpt-187500.bin data/ts10m/ckpt-250000.bin data/ts10m/ckpt.bin 2>&1 | tee results/ts10m-curve.log
build/host/cortex --stop
sleep 2
echo "=== chain: launching the 100M-token continuation at $(date '+%F %T')"
{ echo "=== TinyStories 100M run (tokens 10M to 110M of the stream) from ts10m/ckpt.bin, plain, bus off, lr 3e-3, L 32, ckpt every 10M tokens; $(date '+%F %T')"
  phi run sh -c 'cd /tmp/cortex && mkdir -p ts100m && CX_BUS=0 CX_INIT=/tmp/cortex/ts10m/ckpt.bin CX_CKPT_EVERY=312500 ./cluster-gate run /tmp/cortex/train-next100m.u32 /tmp/cortex/ts100m 32 100000000 0.003 /tmp/cortex/valid.u32 100000 5000 2>&1; echo rc=$?'
  echo "=== end $(date '+%F %T')"; } > results/ts100m.log 2>&1
echo "=== chain: the 100M run ended at $(date '+%F %T')"
