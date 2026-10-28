#!/bin/bash
# teacher-scratch.sh: a model from fresh F32 weights trained on nothing but the teacher's corpus as it is written
# (2026-10-27 13:09, user: "Train a model from scratch just using the teacher model") and, from 15:0x, corrected by
# the teacher between rounds (user: "the teacher model also needs to properly teach the student to communicate"):
# the ILF / LeMa / CITING loop of docs/TEACHER.md "Methods from the literature". The writer (tools/teacher-corpus.sh)
# drops batch files into CORPUS_DIR; this driver turns them into the stream and trains:
#   1. the vocabulary: once at least --vocab-bytes of plain text exist (the batches through chat-text
#      --no-think-block --detok), a 4,096-id BPE is trained on that text (tools/bpe.c) and fixed for good;
#   2. every new batch is converted and encoded with that vocabulary and appended to DIR/train.u16, every 20th
#      batch to DIR/valid.u16 instead (the held-out), whole conversations, as they land (DIR/work/encoded.txt
#      remembers which, across restarts); beside each stream the soft-target records of the partial distillation
#      (docs/DISTIL.md; 2026-10-27 18:43, user: "adjust the teacher model so it partially distills itself into the
#      PhiLLM"): the batch's sidecar of the teacher's logprobs (the writer's batch-*.lp.json) aligned onto the
#      student's tokens by tools/soft-align with --lambda as the teacher's share of the target, empty records for a
#      batch without a sidecar, one 48-byte record a token in DIR/train.soft and DIR/valid.soft (a stream from
#      before the sidecars is padded with empty records at the start);
#   3. a training round starts when the stream holds at least --round-chunks untrained chunks (32 tokens each): the
#      round's stream DIR/round-R.u16 is those chunks, with the correction set recurring inside them once there is
#      one (tools/mixstream: one correction conversation, then --ratio corpus conversations, again; the set is the
#      last --keep-rounds rounds' corrections, with empty soft records) and its soft records DIR/round-R.soft, which
#      the coach feeds into the daemon's soft ring (--soft); the coach runs it from the last state (fresh weights from
#      tools/init-ckpt for the first), lr X with no policy, home routing through the pool, the n-gram memory on (its
#      tables in TABLES, the gates continued from the checkpoint's); the final checkpoint, gates, pool and tables are
#      renamed DIR/round-R-ckpt.bin, -ckpt-gates.f32, -pool.f32, -tables.f32 (the coach names them by its own chunk
#      count, which repeats across rounds), and the trained position advances to where the mix left the base;
#   4. after every round: the held-out loss of the final checkpoint (tools/ref-loss, the dense reading, up to 5,000
#      tokens), the identity test on the live weights (the greedy reply to --prompt judged by the teacher,
#      tools/teacher-judge.sh), and the correction pass (tools/teacher-correct.sh: --corrections corpus conversations
#      cut at a person's line, answered by the live weights under the user's sampling, graded on grammar, relevance
#      and persona and rewritten by the teacher; the rewrites become DIR/corrections-round-R.jsonl and, encoded, the
#      set DIR/corrections-set.u16), all written to results/identity-loop.log in the loop's line shape, so the
#      monitor's "the teacher and the student" section shows them; the student's own words never enter the stream;
#   5. until CORPUS_DIR/stop exists and no untrained chunk is left.
#   tools/teacher-scratch.sh CORPUS_DIR PERSONA.txt [--ckdir DIR] [--tables FILE] [--vocab-bytes B]
#                            [--round-chunks N] [--lr X] [--prompt "..."] [--valid-every K] [--corrections K]
#                            [--ratio R] [--keep-rounds M] [--lambda X] [--replay|--no-replay] [--mem-dropout P]
#                            [--probe QUESTIONS.jsonl] [--weight-user W]
# Defaults: --ckdir data/live-teacher, --tables /dev/shm/cortex-teacher-tables.f32, --vocab-bytes 300000,
# --round-chunks 1000, --lr 0.003, --prompt "Introduce yourself", --valid-every 20, --corrections 20 (0 turns the
# loop off), --ratio 6, --keep-rounds 3, --lambda 0.5 (0 trains on the hard targets alone), --replay (an epoch round
# over the whole stream whenever fewer than --round-chunks new chunks wait; --no-replay waits for the writer),
# --mem-dropout 0 (the share of chunks the coach trains without the n-gram memory, host/coach.c --ngram-dropout),
# --probe data/probe/questions.jsonl (the reasoning probe after every round, tools/teacher-probe.sh), --weight-user 1
# (the loss weight of the tokens that predict a person's line, baked into the soft records by tools/soft-align; the
# speaker's own lines stay at 1; docs/REASON.md lever 4). The soft records are 52 bytes a token (lib/cx-proto.h
# cx_soft); DIR/soft-format.txt names the format and the weight they were written with, and a change rebuilds
# DIR/train.soft and DIR/valid.soft from the batches in DIR/work/encoded.txt. DIR/state.txt carries the round, the trained chunk count and the state: a
# restart resumes from it. The vocabulary lands in DIR/vocab-teacher4096.txt; the clients and the TUIs take it with
# --vocab.
set -u
cd "$(dirname "$0")/.." || exit 1
[ $# -ge 2 ] || { echo "usage: tools/teacher-scratch.sh CORPUS_DIR PERSONA.txt [--ckdir DIR] [--tables FILE] [--vocab-bytes B] [--round-chunks N] [--lr X] [--prompt \"...\"] [--valid-every K] [--corrections K] [--ratio R] [--keep-rounds M] [--lambda X]" >&2; exit 2; }
corpus=$1; persona=$2
shift 2
ckdir=data/live-teacher; tables=/dev/shm/cortex-teacher-tables.f32; vocab_bytes=300000; round_chunks=1000; lr=0.003; prompt="Introduce yourself"; valid_every=20; corrections=20; ratio=6; keep_rounds=3; lambda=0.5; replay=1; mem_dropout=0; probe=data/probe/questions.jsonl; weight_user=1
rec=52   # bytes per soft record (lib/cx-proto.h cx_soft)
while [ $# -gt 0 ]; do
	case "$1" in
	--ckdir) ckdir=$2; shift 2 ;;
	--tables) tables=$2; shift 2 ;;
	--vocab-bytes) vocab_bytes=$2; shift 2 ;;
	--round-chunks) round_chunks=$2; shift 2 ;;
	--lr) lr=$2; shift 2 ;;
	--prompt) prompt=$2; shift 2 ;;
	--valid-every) valid_every=$2; shift 2 ;;
	--corrections) corrections=$2; shift 2 ;;
	--ratio) ratio=$2; shift 2 ;;
	--keep-rounds) keep_rounds=$2; shift 2 ;;
	--lambda) lambda=$2; shift 2 ;;
	--replay) replay=1; shift ;;
	--no-replay) replay=0; shift ;;
	--mem-dropout) mem_dropout=$2; shift 2 ;;
	--probe) probe=$2; shift 2 ;;
	--weight-user) weight_user=$2; shift 2 ;;
	*) echo "teacher-scratch: unknown option $1" >&2; exit 2 ;;
	esac
done
[ -d "$corpus" ] || { echo "teacher-scratch: $corpus not found" >&2; exit 1; }
[ -f "$persona" ] || { echo "teacher-scratch: $persona not found" >&2; exit 1; }
mkdir -p "$ckdir/work"
W=$ckdir/work
vocab=$ckdir/vocab-teacher4096.txt
train=$ckdir/train.u16
valid=$ckdir/valid.u16
setfile=$ckdir/corrections-set.u16
soft=$ckdir/train.soft
vsoft=$ckdir/valid.soft
log=results/identity-loop.log
say() { echo "$(date +%T) $*" | tee -a "$log"; }
[ -f "$W/encoded.txt" ] || : > "$W/encoded.txt"   # the batch files already in the stream
[ -f "$train" ] || : > "$train"
[ -f "$valid" ] || : > "$valid"
# one batch's soft records: the sidecar of the teacher's logprobs (when the writer saved one) aligned onto the
# student's tokens by tools/soft-align, with the person's lines at --weight-user; a batch without a sidecar gets
# empty records (the plain one-hot, the weights still set). $1 the batch, $2 its text, $3 the records out
align_batch() {
	local lp=${1%.jsonl}.lp.json
	if [ -s "$lp" ]; then
		jq -r '.[] | [(.token | @base64), (.logprob | tostring), ([.top_logprobs[]? | ((.token | @base64) + ":" + (.logprob | tostring))] | join(" "))] | @tsv' "$lp" > "$W/batch.lp.tsv" 2>/dev/null || : > "$W/batch.lp.tsv"
	else
		: > "$W/batch.lp.tsv"
	fi
	jq -r '.messages[1:][] | .content' "$1" > "$W/batch.turns" 2>/dev/null || : > "$W/batch.turns"
	build/host/soft-align "$vocab" "$W/batch.turns" "$W/batch.lp.tsv" "$2" "$3" --lambda "$lambda" --weight-user "$weight_user" >> "$W/soft-align.out" 2>&1
}
# the soft-target records run beside the token streams, one $rec-byte record a token. The format and the weight they
# were written with stand in DIR/soft-format.txt: when either changes, both streams are rebuilt from the batches of
# DIR/work/encoded.txt in their order (the same walk as encode_new, so the token counts agree exactly); a stream left
# short by a stop between the two appends is padded with empty records up to its token count, a longer one cut
soft_format="rec $rec weight-user $weight_user lambda $lambda"
resoft_all() {
	local f nb=0 t v
	say "the soft records are rebuilt ($soft_format): $(wc -l < "$W/encoded.txt") batches of $W/encoded.txt"
	: > "$soft"; : > "$vsoft"
	while IFS= read -r f; do
		nb=$((nb + 1))
		[ -f "$f" ] || continue
		build/host/chat-text "$f" "$W/batch.txt" --no-think-block --detok > /dev/null 2>&1 || continue
		build/host/bpe encode "$vocab" "$W/batch.txt" "$W/batch.u16" > /dev/null 2>&1 || continue
		align_batch "$f" "$W/batch.txt" "$W/batch.soft" || continue
		[ $(( $(stat -c %s "$W/batch.soft") / rec )) -eq $(( $(stat -c %s "$W/batch.u16") / 2 )) ] || continue
		if [ $((nb % valid_every)) -eq 0 ]; then cat "$W/batch.soft" >> "$vsoft"; else cat "$W/batch.soft" >> "$soft"; fi
	done < "$W/encoded.txt"
	t=$(( $(stat -c %s "$train") / 2 )); v=$(( $(stat -c %s "$valid") / 2 ))
	say "rebuilt: train $(( $(stat -c %s "$soft") / rec )) records for $t tokens, valid $(( $(stat -c %s "$vsoft") / rec )) for $v (a difference is padded or cut)"
	echo "$soft_format" > "$ckdir/soft-format.txt"
}
pad_soft() {
	local toks recs
	toks=$(( $(stat -c %s "$1") / 2 ))
	recs=0; [ -f "$2" ] && recs=$(( $(stat -c %s "$2") / rec ))
	if [ "$recs" -gt "$toks" ]; then truncate -s $((toks * rec)) "$2"; recs=$toks; fi
	[ "$recs" -lt "$toks" ] || return 0
	build/host/soft-align --empty $((toks - recs)) "$W/pad.soft" > /dev/null || { say "soft-align --empty failed"; exit 1; }
	cat "$W/pad.soft" >> "$2"
	say "$2: padded with $((toks - recs)) empty records to $toks"
}
[ -s "$vocab" ] && [ "$(cat "$ckdir/soft-format.txt" 2>/dev/null)" != "$soft_format" ] && resoft_all
pad_soft "$train" "$soft"
pad_soft "$valid" "$vsoft"
prompt_text=$(printf '<|system|>%s<|end|>\n<|user|>%s<|end|>\n<|assistant|>' "$(cat "$persona")" "$prompt")

say "=== identity loop: set $corpus (the teacher's corpus as it grows), test persona $persona, prompt \"$prompt\", the teacher judges (tools/teacher-judge.sh) and corrects ($corrections a round, ratio $ratio, the last $keep_rounds rounds kept), from ${cur_ckpt:-fresh weights}; a round every $round_chunks new chunks, lr $lr, memory $tables, the vocabulary on the first $vocab_bytes bytes"

# every batch not yet in the stream: converted, encoded, appended (every valid_every-th to the held-out)
nbatch=$(wc -l < "$W/encoded.txt")
encode_new() {
	local f lp
	for f in $(ls "$corpus"/batch-*.jsonl 2>/dev/null | sort); do
		/usr/bin/grep -q -x -F "$f" "$W/encoded.txt" && continue
		nbatch=$((nbatch + 1))
		build/host/chat-text "$f" "$W/batch.txt" --no-think-block --detok > /dev/null 2>&1 || { echo "$f" >> "$W/encoded.txt"; continue; }
		build/host/bpe encode "$vocab" "$W/batch.txt" "$W/batch.u16" > /dev/null 2>&1 || { echo "$f" >> "$W/encoded.txt"; continue; }
		# the teacher's distribution at every token it generated (the batch's sidecar, tools/teacher.sh --lp-out), carried
		# onto the student's tokens by tools/soft-align (docs/DISTIL.md), the person's lines at --weight-user
		align_batch "$f" "$W/batch.txt" "$W/batch.soft" || { say "batch $(basename "$f"): soft-align failed, skipped"; echo "$f" >> "$W/encoded.txt"; continue; }
		if [ $(( $(stat -c %s "$W/batch.soft") / rec )) -ne $(( $(stat -c %s "$W/batch.u16") / 2 )) ]; then say "batch $(basename "$f"): the soft records do not match the tokens, skipped"; echo "$f" >> "$W/encoded.txt"; continue; fi
		if [ $((nbatch % valid_every)) -eq 0 ]; then cat "$W/batch.u16" >> "$valid"; cat "$W/batch.soft" >> "$vsoft"; else cat "$W/batch.u16" >> "$train"; cat "$W/batch.soft" >> "$soft"; fi
		echo "$f" >> "$W/encoded.txt"
	done
}

# the vocabulary, once there is enough text
if [ ! -s "$vocab" ]; then
	while true; do
		[ -f "$corpus/stop" ] && [ "$(ls "$corpus"/batch-*.jsonl 2>/dev/null | wc -l)" -eq 0 ] && { say "no corpus and the writer stopped: nothing to do"; exit 1; }
		cat "$corpus"/batch-*.jsonl 2>/dev/null > "$W/all.jsonl"
		if [ -s "$W/all.jsonl" ]; then
			build/host/chat-text "$W/all.jsonl" "$W/all.txt" --no-think-block --detok > "$W/chat-text.out" 2>&1
			bytes=$(stat -c %s "$W/all.txt" 2>/dev/null || echo 0)
			if [ "$bytes" -ge "$vocab_bytes" ]; then
				build/host/bpe train "$W/all.txt" "$bytes" "$vocab" > "$W/bpe-train.out" 2>&1 || { say "bpe train failed: $(tail -n 1 "$W/bpe-train.out")"; exit 1; }
				say "the vocabulary: $vocab from $bytes bytes of the teacher's text ($(/usr/bin/grep -c . "$W/all.jsonl") conversations): $(tail -n 1 "$W/bpe-train.out" | cut -c1-120)"
				cp "$persona" "$ckdir/persona.txt"
				break
			fi
			say "waiting for the vocabulary's text: $bytes of $vocab_bytes bytes ($(/usr/bin/grep -c . "$W/all.jsonl") conversations)"
		fi
		sleep 60
	done
fi

# the rounds
round=0; trained=0; cur_ckpt=""; cur_pool=""
[ -f "$ckdir/state.txt" ] && read -r round trained cur_ckpt cur_pool < "$ckdir/state.txt"
while true; do
	encode_new
	have=$(( $(stat -c %s "$train") / 64 ))
	untrained=$((have - trained))
	epoch=0
	if [ "$untrained" -lt "$round_chunks" ]; then
		if [ -f "$corpus/stop" ] && [ "$untrained" -le 0 ]; then say "=== the corpus ended and every chunk is trained: $have chunks, $round rounds; the state is $cur_ckpt${cur_pool:+ + $cur_pool}"; exit 0; fi
		# too few new chunks for a round: with --replay the idle time trains another pass over the whole stream (an epoch
		# round from token 0: the model had seen each token once at 21:4x and sat between the bigram and the trigram
		# floors, docs/REASON.md); without it, wait for the writer
		if [ "$replay" -eq 1 ] && [ "$have" -ge "$round_chunks" ] && [ -n "$cur_ckpt" ]; then
			epoch=1
			trained=0
			untrained=$have
		else
			sleep 30
			continue
		fi
	fi
	round=$((round + 1))
	if [ -z "$cur_ckpt" ]; then
		build/host/init-ckpt "$ckdir/ckpt-init.bin" > /dev/null 2>&1 || { say "init-ckpt failed"; exit 1; }
		cur_ckpt=$ckdir/ckpt-init.bin
	fi
	# the round's stream: the untrained chunks, with the correction set recurring inside them once there is one
	stream=$ckdir/round-$round.u16
	trained_after=$have
	if [ "$corrections" -gt 0 ] && [ -s "$setfile" ]; then
		build/host/mixstream "$setfile" "$train" $((trained * 32)) "$ratio" "$untrained" "$stream" --soft "$soft" "${stream%.u16}.soft" > "$W/mix-$round.out" 2>&1 || { say "round $round: mixstream failed: $(tail -n 1 "$W/mix-$round.out" | cut -c1-120)"; exit 1; }
		# the base's end token in the mix is where the next round starts (a partial conversation is moved up by mixstream)
		end_tok=$(sed -n -E 's/.*from token [0-9]+ to ([0-9]+) of.*/\1/p' "$W/mix-$round.out" | head -n 1)
		[ -n "$end_tok" ] && trained_after=$((end_tok / 32))
		[ "$trained_after" -gt "$have" ] && trained_after=$have
		mixnote="the correction set recurring: $(head -n 1 "$W/mix-$round.out" | cut -c1-140)"
	else
		dd if="$train" of="$stream" bs=64 skip="$trained" count="$untrained" status=none
		dd if="$soft" of="${stream%.u16}.soft" bs=$((rec * 32)) skip="$trained" count="$untrained" status=none
		mixnote="no correction set yet"
	fi
	poolopt=(); [ -n "$cur_pool" ] && [ -f "$cur_pool" ] && poolopt=(--pool "$cur_pool")
	gates=ones; [ -f "${cur_ckpt%.bin}-gates.f32" ] && gates=${cur_ckpt%.bin}-gates.f32
	dropopt=(); [ "$(awk -v d="$mem_dropout" 'BEGIN { print (d > 0) }')" = 1 ] && dropopt=(--ngram-dropout "$mem_dropout")
	say "round $round${epoch:+$([ "$epoch" -eq 1 ] && echo " (an epoch: the whole stream again)")}: $untrained untrained chunks of $have; the coach from $cur_ckpt${cur_pool:+ with $cur_pool} on $stream ($mixnote; gates $gates${dropopt:+; memory dropout $mem_dropout})"
	setsid nohup build/host/coach "$stream" --soft "${stream%.u16}.soft" --ckpt "$cur_ckpt" "${poolopt[@]}" --start 0 --lr "$lr" --every 100 --window 300 --halvings 0 --no-stop \
		--save-every 5000 --sample-every 0 --vocab "$vocab" --cmd results/live.cmd --log results/live.log --ckdir "$ckdir" \
		--moe --moe-bias 100 --expert-lr 0.05 --ngram "$tables" --row-lr 0.1 --gates "$gates" "${dropopt[@]}" > "$W/coach-$round.out" 2>&1 < /dev/null &
	cpid=$!
	while kill -0 "$cpid" 2>/dev/null; do sleep 5; done
	fin=$(/usr/bin/grep -a -E 'checkpoint \(final\)' results/live.log | tail -n 1 | sed -E 's/.*(data\/[^ ,]+\/ckpt-[0-9]+\.bin).*/\1/')
	fpool=$(/usr/bin/grep -a -E 'pool: data[^ ]+, [0-9]+ bytes in' results/live.log | tail -n 1 | sed -E 's/.*(data\/[^ ,]+\/pool-[0-9]+\.f32).*/\1/')
	[ -f "$fin" ] || { say "round $round: no final checkpoint ($(tail -n 1 "$W/coach-$round.out" | cut -c1-120)): stopping"; exit 1; }
	# the round's own names: the coach numbers its files by its chunk count, which repeats across rounds
	n=${fin##*-}; n=${n%.bin}
	mv "$fin" "$ckdir/round-$round-ckpt.bin"; cur_ckpt=$ckdir/round-$round-ckpt.bin
	[ -f "${fin%.bin}-gates.f32" ] && mv "${fin%.bin}-gates.f32" "$ckdir/round-$round-ckpt-gates.f32"
	[ -f "$ckdir/tables-$n.f32" ] && mv "$ckdir/tables-$n.f32" "$ckdir/round-$round-tables.f32"
	if [ -f "$fpool" ]; then mv "$fpool" "$ckdir/round-$round-pool.f32"; cur_pool=$ckdir/round-$round-pool.f32; fi
	trained=$trained_after
	echo "$round $trained $cur_ckpt $cur_pool" > "$ckdir/state.txt"
	# the held-out loss on what the held-out holds, up to 5,000 tokens: the memory reading first (the checkpoint with its
	# n-gram memory, the configuration it trains and answers in: 2.76 bits/token at round 6 against the dense 5.15, the
	# bigram floor 3.06 and the trigram floor 2.58 on this corpus, 21:4x), then the dense reading for the columns alone
	held=""; heldd=""; vt=$(( $(stat -c %s "$valid") / 2 )); ntok=$((vt - 64)); [ "$ntok" -gt 5001 ] && ntok=5001
	if [ "$ntok" -ge 1000 ]; then
		[ -f "${cur_ckpt%.bin}-gates.f32" ] && held=$(build/host/ref-loss "$cur_ckpt" "$valid" "$ntok" --ngram "$tables" "${cur_ckpt%.bin}-gates.f32" 2>/dev/null | tail -n 1 | sed -E 's/.*: ([0-9.]+ nats = [0-9.]+ bits\/token).*/\1/')
		heldd=$(build/host/ref-loss "$cur_ckpt" "$valid" "$ntok" 2>/dev/null | tail -n 1 | sed -E 's/.*: ([0-9.]+ nats = [0-9.]+ bits\/token).*/\1/')
	fi
	say "round $round ended: $cur_ckpt $cur_pool; trained $trained chunks of the stream; held-out ${held:-not yet} ($ntok tokens, memory reading; dense ${heldd:-not yet})"
	build/host/cortex --live "$prompt_text" 48 --no-echo --stop-eot --greedy --vocab "$vocab" --ngram "$tables" --stop '<|end|>' --stop '<|user|>' --stop '<|system|>' --stop '<|assistant|>' 2> /dev/null | tr '\n' ' ' | sed -E 's/<\|(end|user|system|assistant|endoftext)\|>.*//' > "$W/reply.txt"
	verdict=$(tools/teacher-judge.sh "$persona" "$prompt" "$(cat "$W/reply.txt")" 2>/dev/null)
	say "round $round chunk $trained: $verdict (matches in a row 0) | $(cut -c1-300 "$W/reply.txt")"
	# the reasoning probe: twenty fixed questions with known answers, the teacher scores the live weights (docs/REASON.md)
	[ -s "$probe" ] && tools/teacher-probe.sh "$probe" "$persona" "$vocab" "$round" "$trained" --tables "$tables" --log "$log" > "$W/probe-$round.out" 2>&1
	# the correction pass: the teacher grades and rewrites the student's answers; the rewrites recur in the next round
	if [ "$corrections" -gt 0 ]; then
		cj=$ckdir/corrections-round-$round.jsonl
		tools/teacher-correct.sh "$corpus" "$persona" "$vocab" "$cj" "$round" "$trained" --count "$corrections" --tables "$tables" --seed "$round" --log "$log" > "$W/correct-$round.out" 2>&1
		if [ -s "$cj" ]; then
			ls "$ckdir"/corrections-round-*.jsonl | sort -t- -k3 -n | tail -n "$keep_rounds" | xargs cat > "$W/set.jsonl"
			if build/host/chat-text "$W/set.jsonl" "$W/set.txt" --no-think-block --detok > /dev/null 2>&1 && build/host/bpe encode "$vocab" "$W/set.txt" "$setfile" > /dev/null 2>&1; then
				say "round $round: the correction set is $(wc -l < "$W/set.jsonl") conversations, $(( $(stat -c %s "$setfile") / 2 )) tokens, from the last $keep_rounds rounds"
			else
				say "round $round: the correction set could not be encoded"
			fi
		fi
	fi
done
