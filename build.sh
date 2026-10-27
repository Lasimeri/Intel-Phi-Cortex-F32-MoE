#!/usr/bin/env bash
# build.sh: Intel Phi Cortex. Host programs with cc into build/host; card programs compiled by the Intel-Phi-LLVM S3
# clang (-march=knc: the 512-bit vector unit) and linked static by the stack's knc-cc into build/card; phi-isa-audit
# checks each card binary for instructions the 7120A lacks. One card program alone: tools/knc-one.sh NAME SOURCES...
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# the 16TB HDD is mounted by hand (not in fstab): without it /mnt is the near-full root fs
case "$here" in /mnt/*) [ "$(findmnt -no SOURCE /mnt)" = /dev/sda2 ] || { echo "build.sh: /mnt is not /dev/sda2" >&2; exit 1; } ;; esac
out="$here/build"
mkdir -p "$out/host" "$out/card" "$out/obj"
flags=(-O2 -std=gnu11 -Wall -Wextra)
hostq=(-Wno-format-truncation -Wno-stringop-truncation -Wno-unused-function)   # the host programs that include tools/bpe.c whole

VCC="${PHI_LLVM_BIN:-$HOME/.cache/intel-phi-llvm-build/llvm/bin}/clang"
SYS="${PHI_SYSROOT_7120A:-$HOME/.cache/intel-phi-7120a-build/toolchain/sysroot}"
knc() {   # knc NAME SOURCES...: compile each source for the card, link static as build/card/NAME
	local name=$1 objs=() s o
	shift
	for s in "$@"; do
		o="$out/obj/$(basename "${s%.c}")-${name}-knc.o"
		"$VCC" --target=x86_64-unknown-linux-musl -march=knc --sysroot="$SYS" "${flags[@]}" ${defs:-} -fno-jump-tables \
			-falign-loops=1 -fPIE -c -o "$o" "$here/$s"
		objs+=("$o")
	done
	knc-cc "${flags[@]}" -static -o "$out/card/$name" "${objs[@]}" -lm -lpthread
	printf '%s: ' "$name"
	"$audit" "$out/card/$name" | tail -1 || true
}
# shellcheck disable=SC1091
. "${PHI_TREE:-$HOME/Intel Phi 7120A}/toolchain/env.sh"
audit="${PHI_TREE:-$HOME/Intel Phi 7120A}/host/target/debug/phi-isa-audit"

# Stage 0
knc hello card/hello.c
knc isa card/isa.c
cc "${flags[@]}" -o "$out/host/isa-host" "$here/tools/isa-host.c" -lm
knc barrier-gate tools/barrier-gate.c
cc "${flags[@]}" -o "$out/host/stats" "$here/tools/stats.c"
knc mbox-gate tools/mbox-gate.c
cc "${flags[@]}" -o "$out/host/mbox-ping" "$here/tools/mbox-ping.c"
knc l2-gate tools/l2-gate.c

# Stage 1: one column forward (card/col.h); pf = weight-row prefetch two ahead, nopf = without
knc col-gate tools/col-gate.c
defs=-DCX_PF=0 knc col-gate-nopf tools/col-gate.c
cc "${flags[@]}" -ffp-contract=off -o "$out/host/col-ref" "$here/tools/col-ref.c" -lm

# Stage 2: tokenizer
cc "${flags[@]}" -o "$out/host/bpe" "$here/tools/bpe.c"

# Stage 3: training reference (host) and gradient checks
cc "${flags[@]}" -O2 -o "$out/host/train-ref" "$here/tools/train-ref.c" -lm
knc train-gate tools/train-gate.c
cc "${flags[@]}" -O2 -o "$out/host/train-check" "$here/tools/train-check.c" -lm
cc "${flags[@]}" -O2 -o "$out/host/unigram" "$here/tools/unigram.c" -lm

# Stage 4: cluster reference (host)
cc "${flags[@]}" -O2 -o "$out/host/ctrain-ref" "$here/tools/ctrain-ref.c" -lm
knc cluster-gate tools/cluster-gate.c card/cdma.c
defs="-DCX_NG_ROWS=4096" knc cluster-gate-ng4k tools/cluster-gate.c card/cdma.c
cc "${flags[@]}" -O2 -o "$out/host/ctrain-check" "$here/tools/ctrain-check.c" -lm
defs="-DCX_NOSNAP" knc cluster-gate-nosnap tools/cluster-gate.c card/cdma.c
defs="-DCX_NOCOLL" knc cluster-gate-nocoll tools/cluster-gate.c card/cdma.c

# Stage 5: the generation daemon (card), its host client (with the orchestrator, AVX2 + FMA) and the daemon's gate
knc cortexd card/cortexd.c card/cdma.c
cc "${flags[@]}" -O2 -mavx2 -mfma "${hostq[@]}" -o "$out/host/cortex" "$here/host/cortex.c" -lm
cc "${flags[@]}" -O2 -o "$out/host/gen-check" "$here/tools/gen-check.c" -lm
# the chat data pipeline and the weight-precision measurement (docs/CHAT.md)
cc "${flags[@]}" -O2 -o "$out/host/chat-text" "$here/tools/chat-text.c"
cc "${flags[@]}" -O2 -o "$out/host/quant3" "$here/tools/quant3.c" -lm
cc "${flags[@]}" -O2 -Wno-unused-function -o "$out/host/ref-loss" "$here/tools/ref-loss.c" -lm
# live training: the host coach (feeder, reports, policy, samples through the client, commands)
cc "${flags[@]}" -O2 "${hostq[@]}" -o "$out/host/coach" "$here/host/coach.c"

# dataset conversion (oasst1 guanaco JSONL -> Human/Assistant text)
cc "${flags[@]}" -O2 -o "$out/host/jsonl-text" "$here/tools/jsonl-text.c"

# MoE stage 1: the host n-gram server and the checker at the gate table size
cc "${flags[@]}" -O2 -o "$out/host/ngserve" "$here/host/ngserve.c"
cc "${flags[@]}" -O2 -DCX_NG_ROWS=4096 -o "$out/host/ngserve-ng4k" "$here/host/ngserve.c"
cc "${flags[@]}" -O2 -DCX_NG_ROWS=4096 -o "$out/host/ctrain-check-ng4k" "$here/tools/ctrain-check.c" -lm

# MoE stages 2 and 3, readiness: the expert pool bench (card), the context round trip (card + host)
knc expert-bench tools/expert-bench.c
knc ctx-bench tools/ctx-bench.c card/cdma.c
cc "${flags[@]}" -O2 -mavx2 -mfma -pthread -o "$out/host/ctxserve" "$here/host/ctxserve.c"

# diagnostics: the card's view of the part, the DMA channels, DMA writes against L2 copies, a text's non-list words
knc partpoll tools/partpoll.c
knc cdma-test tools/cdma-test.c card/cdma.c
knc dma-coh tools/dma-coh.c card/cdma.c
cc "${flags[@]}" -O2 "${hostq[@]}" -o "$out/host/misses" "$here/tools/misses.c"

# the second dataset (BabyLM): text preparation, the mixed stream, the orchestrator's count model
cc "${flags[@]}" -O2 -o "$out/host/textprep" "$here/tools/textprep.c"
cc "${flags[@]}" -O2 -o "$out/host/interleave" "$here/tools/interleave.c"
cc "${flags[@]}" -O2 -o "$out/host/ngcount" "$here/tools/ngcount.c"
