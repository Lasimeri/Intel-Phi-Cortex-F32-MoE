#!/bin/bash
# identity-loop.sh: fine-tune the live model in a loop until its greedy answer to a prompt matches the reference
# (2026-10-27 09:23, user: "Fine tune the model in a loop till it gets 'Introduce yourself' correct").
#
# The set (IDENT.jsonl in the messages shape: a persona as the system message, the user's prompt, the answer; the
# answers human-written, nothing Claude writes is trained on) is encoded with the run's vocabulary and recurs in the
# base stream every RATIO + 1 conversations (tools/mixstream.c). One coach run trains on that stream at home
# routing (the dense model through the pool; free routing collapsed the model on 2026-10-27, PLAN-MOE.md) with the
# n-gram memory on (the chunk layer: the model's structural frame of reference, its rows trained in place in
# TABLES). Every EVERY chunks the test runs on the live weights between chunks: the frame's prompt (the persona, the
# user turn, the assistant marker, as tui/cortex.jinja renders it) through `cortex --live --greedy` with the
# memory's rows, the reply cut at the first marker and compared to REF.txt (tools/replymatch.c: an exact match, or
# a word LCS ratio of at least --lcs). The match must hold across --stable consecutive tests (presence, not a
# one-off); then the run is stopped (final checkpoint and pool of the same chunk), the final weights are re-tested,
# and the loop ends. A failed re-test, or a stream run out, starts the next round from the final checkpoint and
# pool with the base continued, up to --max-rounds. The mix collapses over the rounds (09:31, user: "the training
# data should eventually collapse to form an identity/ego complex within the model itself"): --ratio is a list,
# one value per round (the last repeats), e.g. 3,2,1: the base's share of the stream falls round by round while it
# keeps the language, and the memory's rows hold the identity's own chunks.
#
#   tools/identity-loop.sh IDENT.jsonl PERSONA.txt "PROMPT" REF.txt CKPT [--pool FILE] [--base STREAM.u16]
#                          [--base-start TOKEN] [--ratio R[,R2,...]] [--chunks N] [--every E] [--lr X]
#                          [--max-rounds M] [--stable K] [--ckdir DIR] [--vocab FILE] [--tables FILE] [--lcs X]
#                          [--ngen N] [--judge]
#
# --judge: the teacher decides (tools/teacher-judge.sh over the served Qwen3.8-27B, docs/TEACHER.md) whether the
# greedy reply is a correct self-introduction for the persona, in place of the string match; REF.txt may then be
# "-". The verdict and the teacher's reason go to the log with the reply.
#
# PERSONA.txt is the system message of the test (the set's persona for that conversation), PROMPT its user turn,
# REF.txt its accepted answers (one per line). Defaults: the base data/chat-base2/train.u16 from token 0, ratios
# 3,2,1, 20,000 chunks per round, a test every 500 chunks, lr 2.5e-4, 5 rounds, 2 consecutive matches, --ckdir
# data/live-identity, the chat base's vocabulary, the tables /dev/shm/cortex-identity-tables.f32, an exact match,
# 48 tokens of reply. Every test goes to results/identity-loop.log with its reply and score; the run's own log is
# results/live.log.
set -u
cd "$(dirname "$0")/.." || exit 1
usage() { sed -n '21,24p' "$0" | sed 's/^# *//' >&2; exit 2; }
[ $# -ge 5 ] || usage
ident=$1; persona=$2; prompt=$3; ref=$4; ckpt=$5
shift 5
pool=""; base=data/chat-base2/train.u16; base_start=0; ratio=3,2,1; chunks=20000; every=500; lr=0.00025; max_rounds=5; stable=2
ckdir=data/live-identity; vocab=data/chat-base2/vocab-base4096.txt; tables=/dev/shm/cortex-identity-tables.f32; lcs=2; ngen=48; judge=0
while [ $# -gt 0 ]; do
	case "$1" in
	--judge) judge=1; shift ;;
	--pool) pool=$2; shift 2 ;;
	--base) base=$2; shift 2 ;;
	--base-start) base_start=$2; shift 2 ;;
	--ratio) ratio=$2; shift 2 ;;
	--chunks) chunks=$2; shift 2 ;;
	--every) every=$2; shift 2 ;;
	--lr) lr=$2; shift 2 ;;
	--max-rounds) max_rounds=$2; shift 2 ;;
	--stable) stable=$2; shift 2 ;;
	--ckdir) ckdir=$2; shift 2 ;;
	--vocab) vocab=$2; shift 2 ;;
	--tables) tables=$2; shift 2 ;;
	--lcs) lcs=$2; shift 2 ;;
	--ngen) ngen=$2; shift 2 ;;
	*) usage ;;
	esac
done
for f in "$ident" "$persona" "$ckpt" "$base" "$vocab"; do [ -f "$f" ] || { echo "identity-loop: $f not found" >&2; exit 1; }; done
[ "$judge" -eq 1 ] || [ -f "$ref" ] || { echo "identity-loop: $ref not found (REF.txt may be - only with --judge)" >&2; exit 1; }
W=$ckdir/loop
mkdir -p "$W"
log=results/identity-loop.log
say() { echo "$(date +%T) $*" | tee -a "$log"; }

ratios=(${ratio//,/ })
say "=== identity loop: set $ident, test persona $persona, prompt \"$prompt\", $([ "$judge" -eq 1 ] && echo "the teacher judges (tools/teacher-judge.sh)" || echo "reference $ref"), from $ckpt${pool:+ + $pool}; ratios ${ratios[*]} (the mix collapses round by round), $chunks chunks a round, a test every $every chunks, $stable consecutive matches, lr $lr, memory $tables, $max_rounds rounds at most"
build/host/chat-text "$ident" "$W/ident.txt" --no-think-block --detok > "$W/chat-text.out" 2>&1 || { say "chat-text failed: $(tail -n 1 "$W/chat-text.out")"; exit 1; }
build/host/bpe encode "$vocab" "$W/ident.txt" "$W/ident.u16" > "$W/encode.out" 2>&1 || { say "encode failed: $(tail -n 1 "$W/encode.out")"; exit 1; }
say "the set: $(tail -n 1 "$W/chat-text.out" | cut -c1-140)"
say "encoded: $(tail -n 1 "$W/encode.out" | cut -c1-100)"
prompt_text=$(printf '<|system|>%s<|end|>\n<|user|>%s<|end|>\n<|assistant|>' "$(cat "$persona")" "$prompt")

# the live weights: the greedy reply cut at the first marker, scored against the references; 0 at a match
test_once() {
	build/host/cortex --live "$prompt_text" "$ngen" --no-echo --stop-eot --greedy --vocab "$vocab" --ngram "$tables" \
		--stop '<|end|>' --stop '<|user|>' --stop '<|system|>' --stop '<|assistant|>' > "$W/reply.raw" 2> "$W/reply.err" || { echo "client failed: $(tail -n 1 "$W/reply.err")" > "$W/score.txt"; return 1; }
	tr '\n' ' ' < "$W/reply.raw" | sed -E 's/<\|(end|user|system|assistant|endoftext)\|>.*//' > "$W/reply.txt"
	if [ "$judge" -eq 1 ]; then
		tools/teacher-judge.sh "$persona" "$prompt" "$(cat "$W/reply.txt")" > "$W/score.txt"
		local r=$?
		[ "$r" -eq 2 ] && say "the teacher gave no verdict: $(cat "$W/score.txt")"
		return $r
	fi
	build/host/replymatch "$W/reply.txt" "$ref" "$lcs" > "$W/score.txt"
}

last_chunk() { /usr/bin/grep -a -E ' report chunk ' results/live.log | tail -n 1 | awk '{print $4}'; }

round=0; cur_ckpt=$ckpt; cur_pool=$pool; bstart=$base_start; matched=0
while [ "$round" -lt "$max_rounds" ] && [ "$matched" -eq 0 ]; do
	round=$((round + 1))
	i=$((round - 1)); [ "$i" -ge "${#ratios[@]}" ] && i=$((${#ratios[@]} - 1))
	r_now=${ratios[$i]}
	build/host/mixstream "$W/ident.u16" "$base" "$bstart" "$r_now" "$chunks" "$W/mix.u16" | tee -a "$log" || exit 1
	poolopt=()
	[ -n "$cur_pool" ] && poolopt=(--pool "$cur_pool")
	say "round $round (ratio $r_now): the coach from $cur_ckpt${cur_pool:+ with $cur_pool} on $W/mix.u16 (base from token $bstart)"
	setsid nohup build/host/coach "$W/mix.u16" --ckpt "$cur_ckpt" "${poolopt[@]}" --start 0 --lr "$lr" --every 100 --window 300 --halvings 0 --no-stop \
		--save-every 5000 --sample-every 0 --vocab "$vocab" --cmd results/live.cmd --log results/live.log --ckdir "$ckdir" \
		--moe --moe-bias 100 --expert-lr 0.05 --ngram "$tables" --row-lr 0.1 --gates ones > "$W/coach-$round.out" 2>&1 < /dev/null &
	cpid=$!
	sleep 25
	kill -0 "$cpid" 2>/dev/null || { say "the coach ended at once: $(tail -n 3 "$W/coach-$round.out" | tr '\n' ' ' | cut -c1-300)"; exit 1; }
	next=$every; hits=0
	while kill -0 "$cpid" 2>/dev/null; do
		c=$(last_chunk)
		if [ -n "$c" ] && [ "$c" -ge "$next" ] 2>/dev/null; then
			next=$((c / every * every + every))
			test_once; r=$?
			[ "$r" -eq 0 ] && hits=$((hits + 1)) || hits=0
			say "round $round chunk $c: $(cat "$W/score.txt") (matches in a row $hits) | $(cut -c1-300 "$W/reply.txt")"
			if [ "$hits" -ge "$stable" ]; then echo stop >> results/live.cmd; say "$hits matches in a row by chunk $c: stop requested"; break; fi
		fi
		sleep 2
	done
	while kill -0 "$cpid" 2>/dev/null; do sleep 2; done
	fin=$(/usr/bin/grep -a -E 'checkpoint \(final\)' results/live.log | tail -n 1 | sed -E 's/.*(data\/[^ ,]+\/ckpt-[0-9]+\.bin).*/\1/')
	fpool=$(/usr/bin/grep -a -E 'pool: data[^ ]+, [0-9]+ bytes in' results/live.log | tail -n 1 | sed -E 's/.*(data\/[^ ,]+\/pool-[0-9]+\.f32).*/\1/')
	say "round $round ended: $fin $fpool; $(tail -n 1 "$W/coach-$round.out" | cut -c1-120)"
	[ -f "$fin" ] || { say "no final checkpoint: stopping the loop"; exit 1; }
	test_once; r=$?
	say "round $round re-test on the final weights: $(cat "$W/score.txt") | $(cut -c1-300 "$W/reply.txt")"
	[ "$r" -eq 0 ] && matched=1
	cur_ckpt=$fin
	[ -f "$fpool" ] && cur_pool=$fpool
	bstart=$((bstart + chunks * 32 * r_now / (r_now + 1)))
done
if [ "$matched" -eq 1 ]; then say "=== MATCHED after $round round(s): the model answers \"$prompt\" with the reference; the state is $cur_ckpt${cur_pool:+ + $cur_pool}"; exit 0; fi
say "=== no match after $round round(s); the last state is $cur_ckpt${cur_pool:+ + $cur_pool}"
exit 1
