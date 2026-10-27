#!/usr/bin/env bash
# chat.sh: the chat TUI (tui/, Rust) on the daemon's live weights. Builds it when the binary is missing or older than
# its sources, then opens it in a new konsole window on the desktop, or in this terminal with --here. Any other
# arguments go to cortex-tui (tui/src/app.rs USAGE: --orchestra off, --ngen N, --ctx CHARS, ...).
#   tools/chat.sh [--here] [cortex-tui options]
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
bin="$here/tui/target/release/cortex-tui"
if [ ! -x "$bin" ] || [ -n "$(find "$here/tui/src" "$here/tui/Cargo.toml" -newer "$bin" 2>/dev/null)" ]; then
	(cd "$here/tui" && cargo build --release)
fi
if [ "${1:-}" = "--here" ]; then
	shift
	exec "$bin" --tree "$here" "$@"
fi
setsid -f konsole --workdir "$here" -e "$bin" --tree "$here" "$@" > /dev/null 2>&1
echo "chat.sh: the TUI is open in a konsole window on the desktop ($bin)"
