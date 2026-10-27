#!/usr/bin/env bash
# watch.sh: the training monitor (tui/, cortex-watch) in a new konsole window on the desktop, or in this terminal
# with --here. Builds tui/ when its sources are newer than the binary. Other arguments go to cortex-watch.
#   tools/watch.sh [--here] [cortex-watch options]
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
bin="$here/tui/target/release/cortex-watch"
if [ ! -x "$bin" ] || [ -n "$(find "$here/tui/src" "$here/tui/Cargo.toml" -newer "$bin" 2>/dev/null)" ]; then
	(cd "$here/tui" && cargo build --release)
fi
if [ "${1:-}" = "--here" ]; then
	shift
	exec "$bin" --tree "$here" "$@"
fi
setsid -f konsole --workdir "$here" -e "$bin" --tree "$here" "$@" > /dev/null 2>&1
echo "watch.sh: the monitor is open in a konsole window on the desktop ($bin)"
