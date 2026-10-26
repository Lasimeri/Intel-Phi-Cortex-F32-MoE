#!/usr/bin/env bash
# knc-one.sh NAME SOURCE... : build one card program the way build.sh does (S3 clang -march=knc, static link by knc-cc,
# phi-isa-audit) into build/card/NAME. Extra compile flags through DEFS. Used for single-target rebuilds and tests.
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
out="$here/build"
mkdir -p "$out/card" "$out/obj"
flags=(-O2 -std=gnu11 -Wall -Wextra)
VCC="${PHI_LLVM_BIN:-$HOME/.cache/intel-phi-llvm-build/llvm/bin}/clang"
SYS="${PHI_SYSROOT_7120A:-$HOME/.cache/intel-phi-7120a-build/toolchain/sysroot}"
# shellcheck disable=SC1091
. "${PHI_TREE:-$HOME/Intel Phi 7120A}/toolchain/env.sh"
audit="${PHI_TREE:-$HOME/Intel Phi 7120A}/host/target/debug/phi-isa-audit"
name=$1
shift
objs=()
for s in "$@"; do
	o="$out/obj/$(basename "${s%.c}")-${name}-knc.o"
	# shellcheck disable=SC2086
	"$VCC" --target=x86_64-unknown-linux-musl -march=knc --sysroot="$SYS" "${flags[@]}" ${DEFS:-} -fno-jump-tables \
		-falign-loops=1 -fPIE -c -o "$o" "$here/$s"
	objs+=("$o")
done
knc-cc "${flags[@]}" -static -o "$out/card/$name" "${objs[@]}" -lm -lpthread
printf '%s: ' "$name"
"$audit" "$out/card/$name" | tail -1 || true
