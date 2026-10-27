#!/bin/bash
# teacher-serve.sh: the teacher, Qwen3.8-27B UD-Q4_K_XL (17,559,178,144 B) on this machine's RTX 3090 Ti, served by
# ~/llama.cpp/build/bin/llama-server (the CUDA build) on 127.0.0.1:8090 only, every layer in VRAM, 8,192 tokens of
# context, flash attention, the user's sampling as the server's defaults (temp 0.5, top-k 75, top-p 0.95, min-p
# 0.05, repeat penalty 1.2: a request's own values win), the MTP draft beside the model for speculative decoding
# when --mtp is given. The 35B-A3B was ruled out for this card (2026-10-27 10:26): Q6_K is 29,208,731,296 B and
# Q4_K_M 21,713,462,944 B of weights against 24,564 MiB of VRAM with 3,849 MiB held by the desktop, so neither sits
# in VRAM entirely. Log /tmp/cortex-teacher.log; the server detached; `--stop` ends it; `--status` asks /health.
#   tools/teacher-serve.sh [--mtp] [--ctx N] [--port P] | --status | --stop
set -u
model=$HOME/models/Qwen3.8-27B/Qwen3.8-27B-UD-Q4_K_XL.gguf
draft=$HOME/models/Qwen3.8-27B/MTP/mtp-Qwen3.8-27B-Q4_0.gguf
server=$HOME/llama.cpp/build/bin/llama-server
port=8090; ctx=8192; mtp=0; log=/tmp/cortex-teacher.log
while [ $# -gt 0 ]; do
	case "$1" in
	--mtp) mtp=1; shift ;;
	--ctx) ctx=$2; shift 2 ;;
	--port) port=$2; shift 2 ;;
	--status) curl -s --max-time 5 "http://127.0.0.1:$port/health"; echo; curl -s --max-time 5 "http://127.0.0.1:$port/v1/models" | jq -r '.data[0].id' 2>/dev/null; exit 0 ;;
	--stop) pkill -f "llama-server.*$(basename "$model")" && echo "teacher-serve: stopped" || echo "teacher-serve: nothing to stop"; exit 0 ;;
	*) echo "usage: tools/teacher-serve.sh [--mtp] [--ctx N] [--port P] | --status | --stop" >&2; exit 2 ;;
	esac
done
[ -f "$model" ] || { echo "teacher-serve: $model not found" >&2; exit 1; }
[ -x "$server" ] || { echo "teacher-serve: $server not found" >&2; exit 1; }
if curl -s --max-time 3 "http://127.0.0.1:$port/health" > /dev/null 2>&1; then echo "teacher-serve: something already answers on :$port"; exit 0; fi
spec=()
[ "$mtp" -eq 1 ] && [ -f "$draft" ] && spec=(--spec-type draft-mtp -md "$draft" -ngld 99)
setsid nohup "$server" -m "$model" -ngl 99 -c "$ctx" -fa on --port "$port" --host 127.0.0.1 --jinja \
	--temp 0.5 --top-k 75 --top-p 0.95 --min-p 0.05 --repeat-penalty 1.2 "${spec[@]}" --log-file "$log" > /dev/null 2>&1 < /dev/null &
echo "teacher-serve: starting llama-server on 127.0.0.1:$port (ctx $ctx${spec:+, MTP draft}), log $log"
for i in $(seq 1 120); do
	if curl -s --max-time 2 "http://127.0.0.1:$port/health" 2>/dev/null | /usr/bin/grep -q '"ok"'; then
		echo "teacher-serve: up after $((i * 2)) s: $(curl -s "http://127.0.0.1:$port/v1/models" | jq -r '.data[0].id')"
		nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
		exit 0
	fi
	sleep 2
done
echo "teacher-serve: not up after 240 s; the log's tail:"; tail -n 5 "$log"
exit 1
