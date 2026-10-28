#!/bin/bash
# teacher-serve.sh: the teacher, Qwen3.8-27B UD-Q4_K_XL (17,559,178,144 B) on this machine's RTX 3090 Ti, served by
# ~/llama.cpp/build/bin/llama-server (the CUDA build) on 127.0.0.1:8090 only, every layer in VRAM, 8,192 tokens of
# context, flash attention, the user's sampling as the server's defaults (temp 0.5, top-k 75, top-p 0.95, min-p
# 0.05, repeat penalty 1.2: a request's own values win), the MTP draft beside the model for speculative decoding
# when --mtp is given (not while the corpus writer runs, since 19:13: with the draft the server returns the per-token
# distribution only at the tokens the target model sampled itself, 1 of 396 in a batch; the partial distillation
# needs it at every token, docs/DISTIL.md). The 35B-A3B was ruled out for this card (2026-10-27 10:26): Q6_K is 29,208,731,296 B and
# Q4_K_M 21,713,462,944 B of weights against 24,564 MiB of VRAM with 3,849 MiB held by the desktop, so neither sits
# in VRAM entirely. Log /tmp/cortex-teacher.log; the server detached; `--stop` ends it; `--status` asks /health.
#   tools/teacher-serve.sh [--mtp] [--np N] [--ctx N] [--kv TYPE] [--unified] [--port P] | --status | --stop
# --np N serves N requests at once (N slots; --ctx is the whole context, 8,192 tokens a slot unless given: the
# user's CLAUDE.md system prompt is about 2,200 of them and a corpus call's JSON reply up to 1,700, so a 4,096
# slot cut the replies mid-object on 2026-10-27 13:2x). The KV cache measured on 2026-10-28 04:59 (the server's VRAM
# with one slot of 8,192 against two: 17,101 against 17,826 MiB beside the desktop's 4,200): at most 90.6 KiB a token
# at f16 (the second slot's recurrent state included), not the 262 KB the arithmetic of 10-27 gave. --kv TYPE sets the
# cache's type for K and V (q8_0 halves it: 64K tokens in about 3.0 GiB) and --unified lets every slot use the whole
# context (-kvu): the grader's calls carry the student's record (tools/teacher-grade.sh --history, up to tens of
# thousands of tokens) while the writer's need about 5,000 (2026-10-28 04:55, user: "Does the teacher model have at
# least 64K of ctx set so it can remember how the student model responds").
#   tools/teacher-serve.sh --np 2 --ctx 65536 --kv q8_0 --unified   the configuration since 2026-10-28 05:xx
set -u
model=$HOME/models/Qwen3.8-27B/Qwen3.8-27B-UD-Q4_K_XL.gguf
draft=$HOME/models/Qwen3.8-27B/MTP/mtp-Qwen3.8-27B-Q4_0.gguf
server=$HOME/llama.cpp/build/bin/llama-server
port=8090; ctx=0; np=1; mtp=0; log=/tmp/cortex-teacher.log; kv=""; unified=0
while [ $# -gt 0 ]; do
	case "$1" in
	--mtp) mtp=1; shift ;;
	--kv) kv=$2; shift 2 ;;
	--unified) unified=1; shift ;;
	--np) np=$2; shift 2 ;;
	--ctx) ctx=$2; shift 2 ;;
	--port) port=$2; shift 2 ;;
	--status) curl -s --max-time 5 "http://127.0.0.1:$port/health"; echo; curl -s --max-time 5 "http://127.0.0.1:$port/v1/models" | jq -r '.data[0].id' 2>/dev/null; exit 0 ;;
	--stop) pkill -f "llama-server.*$(basename "$model")" && echo "teacher-serve: stopped" || echo "teacher-serve: nothing to stop"; exit 0 ;;
	*) echo "usage: tools/teacher-serve.sh [--mtp] [--np N] [--ctx N] [--kv TYPE] [--unified] [--port P] | --status | --stop" >&2; exit 2 ;;
	esac
done
[ -f "$model" ] || { echo "teacher-serve: $model not found" >&2; exit 1; }
[ -x "$server" ] || { echo "teacher-serve: $server not found" >&2; exit 1; }
if curl -s --max-time 3 "http://127.0.0.1:$port/health" > /dev/null 2>&1; then echo "teacher-serve: something already answers on :$port"; exit 0; fi
[ "$ctx" -gt 0 ] || ctx=$((np * 8192))
spec=()
[ "$mtp" -eq 1 ] && [ -f "$draft" ] && spec=(--spec-type draft-mtp -md "$draft" -ngld 99)
cache=()
[ -n "$kv" ] && cache=(-ctk "$kv" -ctv "$kv")
[ "$unified" -eq 1 ] && cache+=(-kvu)
setsid nohup "$server" -m "$model" -ngl 99 -c "$ctx" -np "$np" -fa on --port "$port" --host 127.0.0.1 --jinja \
	--temp 0.5 --top-k 75 --top-p 0.95 --min-p 0.05 --repeat-penalty 1.2 "${spec[@]}" "${cache[@]}" --log-file "$log" > /dev/null 2>&1 < /dev/null &
echo "teacher-serve: starting llama-server on 127.0.0.1:$port (ctx $ctx, $np slot(s)${spec:+, MTP draft}${kv:+, cache $kv}$([ "$unified" -eq 1 ] && echo ", unified")), log $log"
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
