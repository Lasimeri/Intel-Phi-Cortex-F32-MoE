#!/bin/bash
# teacher-scratch.sh: a model from fresh F32 weights trained on nothing but the teacher's corpus as it is written
# (2026-10-27 13:09, user: "Train a model from scratch just using the teacher model") and, from 15:0x, corrected by
# the teacher between rounds (user: "the teacher model also needs to properly teach the student to communicate"):
# the ILF / LeMa / CITING loop of docs/TEACHER.md "Methods from the literature". The writer (tools/teacher-corpus.sh)
# drops batch files into CORPUS_DIR; this driver turns them into the stream and trains:
#   1. the vocabulary: once at least --vocab-bytes of plain text exist (the batches through chat-text
#      --no-think-block --detok), a 4,096-id BPE is trained on that text (tools/bpe.c) and fixed for good;
#   2. every new batch is converted and encoded with that vocabulary and appended to DIR/train.u32, every 20th
#      batch to DIR/valid.u32 instead (the held-out), whole conversations, as they land (DIR/work/encoded.txt
#      remembers which, across restarts); beside each stream the soft-target records of the partial distillation
#      (docs/DISTIL.md; 2026-10-27 18:43, user: "adjust the teacher model so it partially distills itself into the
#      PhiLLM"): the batch's sidecar of the teacher's logprobs (the writer's batch-*.lp.json) aligned onto the
#      student's tokens by tools/soft-align with --lambda as the teacher's share of the target, empty records for a
#      batch without a sidecar, one 68-byte record a token in DIR/train.soft and DIR/valid.soft (a stream from
#      before the sidecars is padded with empty records at the start);
#   3. a training round starts when the stream holds at least --round-chunks untrained chunks (32 tokens each): the
#      round's stream DIR/round-R.u32 is those chunks, with the correction set recurring inside them once there is
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
#      set DIR/corrections-set.u32), all written to results/identity-loop.log in the loop's line shape, so the
#      monitor's "the teacher and the student" section shows them; the student's own words never enter the stream;
#   5. until CORPUS_DIR/stop exists and no untrained chunk is left.
#   tools/teacher-scratch.sh CORPUS_DIR PERSONA.txt [--ckdir DIR] [--tables FILE] [--vocab-bytes B]
#                            [--round-chunks N] [--lr X] [--prompt "..."] [--valid-every K] [--corrections K]
#                            [--ratio R] [--keep-rounds M] [--lambda X] [--replay|--no-replay] [--mem-dropout P]
#                            [--probe QUESTIONS.jsonl] [--weight-user W] [--epoch-lr X]
#                            [--reward|--no-reward] [--reward-min G] [--reward-rounds N] [--reward-weight W]
#                            [--reward-bank FILE] [--reward-count N] [--reward-samples S]
# (v8, 2026-10-28 06:1x; lib/cx-proto.h cx_tok, the change of 06:10: a token id is a 32-bit value where it is used, so
# it is stored in 32 bits) every token stream is a .u32 file of little-endian uint32 ids (DIR/train.u32,
# DIR/valid.u32, DIR/round-R.u32, DIR/corrections-set.u32; tools/bpe.c encode writes them), the same ids 0 to 4,095 in
# the same order: 4 bytes a token, 128 bytes a chunk of 32; the soft records are 68 bytes (cx_soft: 8 u32 ids with
# CX_SOFT_NONE 0xFFFFFFFF unused, 8 F32 probabilities, the F32 weight). An instance of before the change is widened at
# the start and nothing of it is deleted: a missing DIR/train.u32, valid.u32 or corrections-set.u32 is written from its
# .u16 by build/host/ids32 (tools/ids32.c); the set's 52-byte records are widened record by record by ids32 --soft (the
# reward weights exactly as they were), the 52-byte file kept as DIR/corrections-set-rec52.soft; DIR/train.soft and
# DIR/valid.soft are widened the same way by ids32 --soft (kept as DIR/train-rec52.soft and DIR/valid-rec52.soft); a
# rebuild from the batches follows only a change of --weight-user or --lambda or a record count that does not match.
# (v7, 05:5x) --reward-bank FILE (default DIR/reward-bank.jsonl, tools/teacher-bank.sh): after the correction pass,
# tools/teacher-reward.sh puts --reward-count questions of the bank (default 40, a draw fixed by the round) to the live
# weights --reward-samples times each (default 2) and the teacher verifies every answer against its expected one; the
# right ones that closed themselves join the round's rewards: the correction pass alone made right replies 2.1 % of
# the time (7 of 328 in the student's record at 05:4x), about 0.4 a pass, too few to be a reward.
# (v6, 2026-10-28 05:3x, the user: "The student model needs a reward function for when it gets something right")
# --reward (the default): every reply of the correction pass the teacher grades right (grammar, relevance and persona
# all at least --reward-min, default 7) is reinforced: the conversation closed by the student's own reply recurs in the
# set of the next rounds (the last --reward-rounds, default 5) with that reply's tokens at --reward-weight (default 1)
# and the rest at 0 (tools/reward-soft.c). The student's own words enter the stream only this way, teacher-verified;
# the wrong replies keep getting the teacher's rewrite. --no-reward turns it off.
# (v5, 2026-10-28 05:xx) --epoch-lr X: the learning rate of the epoch rounds (default --lr; 1e-3 from 05:xx: sixteen
# rounds at a constant 3e-3 left the window losses bouncing 2.7 to 3.3 with no trend, docs/REASON.md). The grader of
# the correction pass reads the student's record, DIR/student-record.txt (tools/teacher-correct.sh --record: every
# graded reply with the person's line, the grades and the rewrite; the user, 04:55: the teacher should remember how
# the student responds). Before every round the round's soft records are checked against its tokens
# (tools/soft-check: the next token among the ids of at least 85 % of the distribution-carrying records) and the round
# is refused when they do not align; a stream whose record count differs from its token count is rebuilt from the
# batch list once and refused when it still differs (no padding). The coach's samples use the persona's chat prompt and
# the teacher's own word list, DIR/words-teacher.txt (2026-10-28 04:58, user: "stop using the story telling dataset";
# nothing of the stories was ever in this instance's stream, its weights, vocabulary or tables: the coach's default
# sample prompt "Once upon a time" and its stories word list were the last traces). The held-out's memory reading
# runs the context memory too (tools/ref-loss --ctx).
# Defaults: --ckdir data/live-teacher, --tables /dev/shm/cortex-teacher-tables.f32, --vocab-bytes 300000,
# --round-chunks 1000, --lr 0.003, --prompt "Introduce yourself", --valid-every 20, --corrections 20 (0 turns the
# loop off), --ratio 6, --keep-rounds 3, --lambda 0.5 (0 trains on the hard targets alone), --replay (an epoch round
# over the whole stream whenever fewer than --round-chunks new chunks wait; --no-replay waits for the writer),
# --mem-dropout 0 (the share of chunks the coach trains without the n-gram memory, host/coach.c --ngram-dropout),
# --probe data/probe/questions.jsonl (the reasoning probe after every round, tools/teacher-probe.sh), --weight-user 1
# (the loss weight of the tokens that predict a person's line, baked into the soft records by tools/soft-align; the
# speaker's own lines stay at 1; docs/REASON.md lever 4). The soft records are 68 bytes a token (lib/cx-proto.h
# cx_soft: 8 u32 ids, 8 F32 probabilities, the F32 weight); DIR/soft-format.txt names the format, the weight and the
# lambda they were written with; a change of the weight or the lambda rebuilds DIR/train.soft and DIR/valid.soft from
# the batches in DIR/work/encoded.txt, and the 52-byte records of before the u32 ids are widened, not rebuilt (v8).
# DIR/state.txt carries the round, the trained chunk count and the state: a restart resumes from it. The vocabulary
# lands in DIR/vocab-teacher4096.txt; the clients and the TUIs take it with --vocab.
set -u
cd "$(dirname "$0")/.." || exit 1
[ $# -ge 2 ] || { echo "usage: tools/teacher-scratch.sh CORPUS_DIR PERSONA.txt [--ckdir DIR] [--tables FILE] [--vocab-bytes B] [--round-chunks N] [--lr X] [--prompt \"...\"] [--valid-every K] [--corrections K] [--ratio R] [--keep-rounds M] [--lambda X]" >&2; exit 2; }
corpus=$1; persona=$2
shift 2
ckdir=data/live-teacher; tables=/dev/shm/cortex-teacher-tables.f32; vocab_bytes=300000; round_chunks=1000; lr=0.003; prompt="Introduce yourself"; valid_every=20; corrections=20; ratio=6; keep_rounds=3; lambda=0.5; replay=1; mem_dropout=0; probe=data/probe/questions.jsonl; weight_user=1; epoch_lr=""; reward=1; reward_min=7; reward_rounds=5; reward_weight=1; reward_bank=""; reward_count=40; reward_samples=2
rec=68   # bytes per soft record (lib/cx-proto.h cx_soft, CX_SOFT_BYTES)
tok=4    # bytes per token id in a stream (lib/cx-proto.h cx_tok, CX_TOK_BYTES): a chunk of 32 tokens is 128 bytes
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
	--epoch-lr) epoch_lr=$2; shift 2 ;;
	--reward) reward=1; shift ;;
	--no-reward) reward=0; shift ;;
	--reward-min) reward_min=$2; shift 2 ;;
	--reward-rounds) reward_rounds=$2; shift 2 ;;
	--reward-weight) reward_weight=$2; shift 2 ;;
	--reward-bank) reward_bank=$2; shift 2 ;;
	--reward-count) reward_count=$2; shift 2 ;;
	--reward-samples) reward_samples=$2; shift 2 ;;
	*) echo "teacher-scratch: unknown option $1" >&2; exit 2 ;;
	esac
done
[ -d "$corpus" ] || { echo "teacher-scratch: $corpus not found" >&2; exit 1; }
[ -f "$persona" ] || { echo "teacher-scratch: $persona not found" >&2; exit 1; }
mkdir -p "$ckdir/work"
W=$ckdir/work
vocab=$ckdir/vocab-teacher4096.txt
train=$ckdir/train.u32
valid=$ckdir/valid.u32
setfile=$ckdir/corrections-set.u32
soft=$ckdir/train.soft
vsoft=$ckdir/valid.soft
log=results/identity-loop.log
say() { echo "$(date +%T) $*" | tee -a "$log"; }
# an instance of before the u32 ids (2026-10-28 06:10): every stream with no .u32 yet and a .u16 is widened by
# build/host/ids32 (tools/ids32.c: the same ids in the same order, 4 bytes each) into a temporary file, which takes the
# .u32 name only when it holds exactly twice the .u16's bytes (as many ids); the .u16 stays where it is. This runs
# before anything below creates an empty stream, so a stream of before the change is never mistaken for a new one.
for u32 in "$train" "$valid" "$setfile"; do
	u16=${u32%.u32}.u16
	[ ! -e "$u32" ] && [ -f "$u16" ] || continue
	[ -x build/host/ids32 ] || { say "the stream $u16 needs widening to $u32 and build/host/ids32 is missing: stopping"; exit 1; }
	if ! build/host/ids32 "$u16" "$u32.tmp" > "$W/ids32.out" 2>&1 || [ "$(stat -c %s "$u32.tmp" 2>/dev/null || echo 0)" -ne $((2 * $(stat -c %s "$u16"))) ]; then
		say "the stream $u16 could not be widened to $u32 ($(tail -n 1 "$W/ids32.out" | cut -c1-120)): stopping"
		rm -f "$u32.tmp"
		exit 1
	fi
	mv "$u32.tmp" "$u32"
	say "the stream $u16 widened to $u32: $(( $(stat -c %s "$u32") / tok )) tokens as u32 ids (the .u16 kept)"
done
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
# were written with stand in DIR/soft-format.txt: when either changes, or a stream's record count differs from its
# token count (a stop between the two appends), both streams are rebuilt from the batches of DIR/work/encoded.txt in
# their order (the same walk as encode_new: every batch whose text converted and encoded is in the token stream, so it
# gets records, empty ones when its alignment fails; none failed in 1,387 alignments to 2026-10-28 05:07). One change
# of format is not a rebuild: the 52-byte records of before the u32 ids, written with this run's weight and lambda at
# one record a token, are widened to $rec bytes record by record (below). Then the counts must agree and the records
# must align (tools/soft-check), or the driver stops: it never pads.
soft_format="rec $rec weight-user $weight_user lambda $lambda"
resoft_all() {
	local f nb=0 nt
	say "the soft records are rebuilt ($soft_format): $(wc -l < "$W/encoded.txt") batches of $W/encoded.txt"
	: > "$soft"; : > "$vsoft"
	while IFS= read -r f; do
		nb=$((nb + 1))
		[ -f "$f" ] || continue
		build/host/chat-text "$f" "$W/batch.txt" --no-think-block --detok > /dev/null 2>&1 || continue
		build/host/bpe encode "$vocab" "$W/batch.txt" "$W/batch.u32" > /dev/null 2>&1 || continue
		nt=$(( $(stat -c %s "$W/batch.u32") / tok ))
		if ! align_batch "$f" "$W/batch.txt" "$W/batch.soft" || [ $(( $(stat -c %s "$W/batch.soft") / rec )) -ne "$nt" ]; then
			build/host/soft-align --empty "$nt" "$W/batch.soft" > /dev/null || { say "soft-align --empty failed"; exit 1; }
			say "rebuild: batch $(basename "$f") did not align: $nt empty records"
		fi
		if [ $((nb % valid_every)) -eq 0 ]; then cat "$W/batch.soft" >> "$vsoft"; else cat "$W/batch.soft" >> "$soft"; fi
	done < "$W/encoded.txt"
	echo "$soft_format" > "$ckdir/soft-format.txt"
}
soft_counts_ok() { [ $(( $(stat -c %s "$train") / tok )) -eq $(( $(stat -c %s "$soft" 2>/dev/null || echo 0) / rec )) ] && [ $(( $(stat -c %s "$valid") / tok )) -eq $(( $(stat -c %s "$vsoft" 2>/dev/null || echo 0) / rec )) ]; }
# the streams' records of before the u32 ids (DIR/soft-format.txt "rec 52 weight-user W lambda X" with this run's
# weight and lambda, one 52-byte record a token of each stream): widened by build/host/ids32 --soft as the set's are
# below (each id widened, 0xFFFF to CX_SOFT_NONE, p and w copied bit for bit, the output re-read against the input),
# not rebuilt from the batches: the writer (tools/teacher-corpus.sh) fills a batch file in place one conversation at a
# time and moves its sidecar beside it after, so a batch encoded early would get other records, or another token
# count, from a rebuild. The widened files take the names only when both streams widened at exactly $rec bytes a
# record; the 52-byte files are kept as DIR/train-rec52.soft and DIR/valid-rec52.soft.
if [ "$(cat "$ckdir/soft-format.txt" 2>/dev/null)" = "rec 52 weight-user $weight_user lambda $lambda" ] \
	&& [ "$(stat -c %s "$soft" 2>/dev/null || echo -1)" -eq $(( $(stat -c %s "$train") / tok * 52 )) ] \
	&& [ "$(stat -c %s "$vsoft" 2>/dev/null || echo -1)" -eq $(( $(stat -c %s "$valid") / tok * 52 )) ]; then
	[ -x build/host/ids32 ] || { say "the soft records need widening from 52 to $rec bytes and build/host/ids32 is missing: stopping"; exit 1; }
	for s in "$soft" "$vsoft"; do
		nr=$(( $(stat -c %s "$s") / 52 ))
		if ! build/host/ids32 --soft "$s" "$s.tmp" --rec 52 > "$W/ids32.out" 2>&1 || [ "$(stat -c %s "$s.tmp" 2>/dev/null || echo 0)" -ne $((nr * rec)) ]; then
			say "the soft records $s could not be widened from 52 to $rec bytes ($(tail -n 1 "$W/ids32.out" | cut -c1-120)): stopping"
			rm -f "$soft.tmp" "$vsoft.tmp"
			exit 1
		fi
		say "$(head -n 1 "$W/ids32.out" | cut -c1-260)"
	done
	for s in "$soft" "$vsoft"; do mv "$s" "${s%.soft}-rec52.soft"; mv "$s.tmp" "$s"; done
	echo "$soft_format" > "$ckdir/soft-format.txt"
	say "the soft records widened from 52 to $rec bytes (u32 ids; the 52-byte files kept as ${soft%.soft}-rec52.soft and ${vsoft%.soft}-rec52.soft)"
fi
if [ -s "$vocab" ]; then
	if [ "$(cat "$ckdir/soft-format.txt" 2>/dev/null)" != "$soft_format" ] || ! soft_counts_ok; then resoft_all; fi
	soft_counts_ok || { say "the soft records do not match the tokens after the rebuild (train $(( $(stat -c %s "$train") / tok )) tokens, $(( $(stat -c %s "$soft") / rec )) records; valid $(( $(stat -c %s "$valid") / tok )), $(( $(stat -c %s "$vsoft") / rec ))): stopping"; exit 1; }
	for pair in "$train:$soft" "$valid:$vsoft"; do
		build/host/soft-check "${pair%%:*}" "${pair#*:}" --rec "$rec" > "$W/soft-check.out" 2>&1 || { say "the soft records of ${pair#*:} do not align with the tokens: $(tail -n 1 "$W/soft-check.out"): stopping"; exit 1; }
		say "$(/usr/bin/grep -E 'next token among' "$W/soft-check.out" | sed -E 's/^soft-check: //') (${pair#*:})"
	done
fi
prompt_text=$(printf '<|system|>%s<|end|>\n<|user|>%s<|end|>\n<|assistant|>' "$(cat "$persona")" "$prompt")
# the teacher's own words for the coach's sample metrics (no stories list anywhere in this instance)
words_list=$ckdir/words-teacher.txt
cat "$corpus"/batch-*.jsonl 2>/dev/null | jq -r '.messages[1:][] | .content' 2>/dev/null | tr 'A-Z' 'a-z' | tr -c "a-z'\n" '\n' | sed -E "s/^'+//; s/'+$//" | /usr/bin/grep -E "^[a-z][a-z']{0,19}$" | sort | uniq -c | awk '$1 >= 3 { print $2, $1 }' > "$words_list.tmp" && [ -s "$words_list.tmp" ] && mv "$words_list.tmp" "$words_list"
record=$ckdir/student-record.txt
[ -n "$epoch_lr" ] || epoch_lr=$lr
# the recurring set and its records: the corrections of the last --keep-rounds rounds (the teacher's rewrites, weight
# 1) and, with --reward, the student's own replies the teacher graded right in the last --reward-rounds rounds, their
# reply tokens at --reward-weight and everything else at 0 (tools/reward-soft.c); the set's records go to mixstream
# with --set-soft, so the reward reaches the card in the same records as the teacher's distribution
setsoft=$ckdir/corrections-set.soft
build_set() {
	local nc nrw=0
	ls "$ckdir"/corrections-round-*.jsonl | sort -t- -k3 -n | tail -n "$keep_rounds" | xargs cat > "$W/set.jsonl"
	if ! build/host/chat-text "$W/set.jsonl" "$W/set.txt" --no-think-block --detok > /dev/null 2>&1 || ! build/host/bpe encode "$vocab" "$W/set.txt" "$W/setc.u32" > /dev/null 2>&1; then
		say "round $round: the correction set could not be encoded"
		return
	fi
	nc=$(( $(stat -c %s "$W/setc.u32") / tok ))
	build/host/soft-align --empty "$nc" "$W/setc.soft" > /dev/null || { say "round $round: soft-align --empty failed"; return; }
	if [ "$reward" -eq 1 ]; then
		ls "$ckdir"/rewards-round-*.jsonl 2>/dev/null | sort -t- -k3 -n | tail -n "$reward_rounds" | xargs -r cat > "$W/rw.jsonl"
		if [ -s "$W/rw.jsonl" ]; then
			if build/host/chat-text "$W/rw.jsonl" "$W/rw.txt" --no-think-block --detok > /dev/null 2>&1 && build/host/bpe encode "$vocab" "$W/rw.txt" "$W/rw.u32" > /dev/null 2>&1 \
				&& jq -r '.reward // 1' "$W/rw.jsonl" > "$W/rw.rewards" && build/host/reward-soft "$W/rw.u32" "$W/rw.rewards" "$W/rw.soft" --scale "$reward_weight" > "$W/reward-soft.out" 2>&1; then
				cat "$W/rw.u32" >> "$W/setc.u32"
				cat "$W/rw.soft" >> "$W/setc.soft"
				nrw=$(wc -l < "$W/rw.jsonl")
			else
				say "round $round: the reward set could not be built: $(tail -n 1 "$W/reward-soft.out" 2>/dev/null | cut -c1-120)"
			fi
		fi
	fi
	mv "$W/setc.u32" "$setfile"
	mv "$W/setc.soft" "$setsoft"
	say "round $round: the correction set is $(wc -l < "$W/set.jsonl") conversations, $nc tokens, from the last $keep_rounds rounds$([ "$reward" -eq 1 ] && echo "; the reward set $nrw of the student's own right replies of the last $reward_rounds rounds, at weight $reward_weight ($(sed -n -E 's/.* ([0-9]+) reinforced tokens.*/\1/p' "$W/reward-soft.out" 2>/dev/null) reinforced tokens)")"
}
# the set's records, one per token of the set. A set of before the u32 ids carries 52-byte records (u16 id[8] with
# 0xFFFF unused, F32 p[8], F32 w), the format the 52-byte driver wrote and kept (its own start had already replaced a
# 48-byte set's records): build/host/ids32 --soft widens them record by record to the 68-byte cx_soft (each id widened,
# 0xFFFF to CX_SOFT_NONE, p and w copied bit for bit) and re-reads its output against them, so the reward weights
# tools/reward-soft wrote stay exactly as they were (byte-identical to a widening through the two structs on 100,000
# random records, 2026-10-28 06:3x); the widened file takes the name only at exactly $rec bytes a token, and the
# 52-byte file is kept as DIR/corrections-set-rec52.soft. A set written before its records existed gets empty ones
# (weight 1, as mixstream gave it then).
if [ -s "$setfile" ]; then
	nset=$(( $(stat -c %s "$setfile") / tok ))
	if [ -f "$setsoft" ] && [ "$(stat -c %s "$setsoft")" -eq $((nset * 52)) ]; then
		[ -x build/host/ids32 ] || { say "the correction set's 52-byte records need widening and build/host/ids32 is missing: stopping"; exit 1; }
		if ! build/host/ids32 --soft "$setsoft" "$setsoft.tmp" --rec 52 > "$W/ids32.out" 2>&1 || [ "$(stat -c %s "$setsoft.tmp" 2>/dev/null || echo 0)" -ne $((nset * rec)) ]; then
			say "the correction set's $nset records of 52 bytes could not be widened to $rec ($(tail -n 1 "$W/ids32.out" | cut -c1-120)): stopping"
			rm -f "$setsoft.tmp"
			exit 1
		fi
		mv "$setsoft" "$ckdir/corrections-set-rec52.soft"
		mv "$setsoft.tmp" "$setsoft"
		say "the correction set's $nset soft records widened from 52 to $rec bytes (u32 ids; the 52-byte file kept as $ckdir/corrections-set-rec52.soft)"
	elif [ ! -f "$setsoft" ] || [ $(( $(stat -c %s "$setsoft") / rec )) -ne "$nset" ]; then
		build/host/soft-align --empty "$nset" "$setsoft" > /dev/null || { say "soft-align --empty failed"; exit 1; }
	fi
fi

say "=== identity loop: set $corpus (the teacher's corpus as it grows), test persona $persona, prompt \"$prompt\", the teacher judges (tools/teacher-judge.sh) and corrects ($corrections a round, ratio $ratio, the last $keep_rounds rounds kept), from ${cur_ckpt:-fresh weights}; a round every $round_chunks new chunks, lr $lr, memory $tables, the vocabulary on the first $vocab_bytes bytes"

# every batch not yet in the stream: converted, encoded, appended (every valid_every-th to the held-out)
nbatch=$(wc -l < "$W/encoded.txt")
encode_new() {
	local f lp
	for f in $(ls "$corpus"/batch-*.jsonl 2>/dev/null | sort); do
		/usr/bin/grep -q -x -F "$f" "$W/encoded.txt" && continue
		nbatch=$((nbatch + 1))
		build/host/chat-text "$f" "$W/batch.txt" --no-think-block --detok > /dev/null 2>&1 || { echo "$f" >> "$W/encoded.txt"; continue; }
		build/host/bpe encode "$vocab" "$W/batch.txt" "$W/batch.u32" > /dev/null 2>&1 || { echo "$f" >> "$W/encoded.txt"; continue; }
		# the teacher's distribution at every token it generated (the batch's sidecar, tools/teacher.sh --lp-out), carried
		# onto the student's tokens by tools/soft-align (docs/DISTIL.md), the person's lines at --weight-user
		align_batch "$f" "$W/batch.txt" "$W/batch.soft" || { say "batch $(basename "$f"): soft-align failed, skipped"; echo "$f" >> "$W/encoded.txt"; continue; }
		if [ $(( $(stat -c %s "$W/batch.soft") / rec )) -ne $(( $(stat -c %s "$W/batch.u32") / tok )) ]; then say "batch $(basename "$f"): the soft records do not match the tokens, skipped"; echo "$f" >> "$W/encoded.txt"; continue; fi
		if [ $((nbatch % valid_every)) -eq 0 ]; then cat "$W/batch.u32" >> "$valid"; cat "$W/batch.soft" >> "$vsoft"; else cat "$W/batch.u32" >> "$train"; cat "$W/batch.soft" >> "$soft"; fi
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
	have=$(( $(stat -c %s "$train") / (32 * tok) ))
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
	stream=$ckdir/round-$round.u32
	trained_after=$have
	if [ "$corrections" -gt 0 ] && [ -s "$setfile" ]; then
		build/host/mixstream "$setfile" "$train" $((trained * 32)) "$ratio" "$untrained" "$stream" --soft "$soft" "${stream%.u32}.soft" --set-soft "$setsoft" > "$W/mix-$round.out" 2>&1 || { say "round $round: mixstream failed: $(tail -n 1 "$W/mix-$round.out" | cut -c1-120)"; exit 1; }
		# the base's end token in the mix is where the next round starts (a partial conversation is moved up by mixstream)
		end_tok=$(sed -n -E 's/.*from token [0-9]+ to ([0-9]+) of.*/\1/p' "$W/mix-$round.out" | head -n 1)
		[ -n "$end_tok" ] && trained_after=$((end_tok / 32))
		[ "$trained_after" -gt "$have" ] && trained_after=$have
		mixnote="the correction set recurring: $(head -n 1 "$W/mix-$round.out" | cut -c1-140)"
	else
		dd if="$train" of="$stream" bs=$((32 * tok)) skip="$trained" count="$untrained" status=none
		dd if="$soft" of="${stream%.u32}.soft" bs=$((rec * 32)) skip="$trained" count="$untrained" status=none
		mixnote="no correction set yet"
	fi
	poolopt=(); [ -n "$cur_pool" ] && [ -f "$cur_pool" ] && poolopt=(--pool "$cur_pool")
	gates=ones; [ -f "${cur_ckpt%.bin}-gates.f32" ] && gates=${cur_ckpt%.bin}-gates.f32
	dropopt=(); [ "$(awk -v d="$mem_dropout" 'BEGIN { print (d > 0) }')" = 1 ] && dropopt=(--ngram-dropout "$mem_dropout")
	rlr=$lr; [ "$epoch" -eq 1 ] && rlr=$epoch_lr
	say "round $round${epoch:+$([ "$epoch" -eq 1 ] && echo " (an epoch: the whole stream again)")}: $untrained untrained chunks of $have; the coach from $cur_ckpt${cur_pool:+ with $cur_pool} on $stream ($mixnote; gates $gates${dropopt:+; memory dropout $mem_dropout}; lr $rlr)"
	build/host/soft-check "$stream" "${stream%.u32}.soft" --rec "$rec" > "$W/soft-check.out" 2>&1 || { say "round $round: the soft records do not align with the round's tokens: $(tail -n 1 "$W/soft-check.out"): refusing the round"; exit 1; }
	setsid nohup build/host/coach "$stream" --soft "${stream%.u32}.soft" --prompt "$prompt_text" --words "$words_list" --ckpt "$cur_ckpt" "${poolopt[@]}" --start 0 --lr "$rlr" --every 100 --window 300 --halvings 0 --no-stop \
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
	held=""; heldd=""; vt=$(( $(stat -c %s "$valid") / tok )); ntok=$((vt - 64)); [ "$ntok" -gt 5001 ] && ntok=5001
	if [ "$ntok" -ge 1000 ]; then
		[ -f "${cur_ckpt%.bin}-gates.f32" ] && held=$(build/host/ref-loss "$cur_ckpt" "$valid" "$ntok" --ngram "$tables" "${cur_ckpt%.bin}-gates.f32" --ctx 2>/dev/null | tail -n 1 | sed -E 's/.*: ([0-9.]+ nats = [0-9.]+ bits\/token).*/\1/')
		heldd=$(build/host/ref-loss "$cur_ckpt" "$valid" "$ntok" 2>/dev/null | tail -n 1 | sed -E 's/.*: ([0-9.]+ nats = [0-9.]+ bits\/token).*/\1/')
	fi
	say "round $round ended: $cur_ckpt $cur_pool; trained $trained chunks of the stream; held-out ${held:-not yet} ($ntok tokens, memory reading with the context memory; dense ${heldd:-not yet})"
	build/host/cortex --live "$prompt_text" 48 --no-echo --stop-eot --greedy --vocab "$vocab" --ngram "$tables" --stop '<|end|>' --stop '<|user|>' --stop '<|system|>' --stop '<|assistant|>' 2> /dev/null | tr '\n' ' ' | sed -E 's/<\|(end|user|system|assistant|endoftext)\|>.*//' > "$W/reply.txt"
	verdict=$(tools/teacher-judge.sh "$persona" "$prompt" "$(cat "$W/reply.txt")" 2>/dev/null)
	say "round $round chunk $trained: $verdict (matches in a row 0) | $(cut -c1-300 "$W/reply.txt")"
	# the reasoning probe: twenty fixed questions with known answers, the teacher scores the live weights (docs/REASON.md)
	[ -s "$probe" ] && tools/teacher-probe.sh "$probe" "$persona" "$vocab" "$round" "$trained" --tables "$tables" --log "$log" > "$W/probe-$round.out" 2>&1
	# the correction pass: the teacher grades and rewrites the student's answers; the rewrites recur in the next round
	if [ "$corrections" -gt 0 ]; then
		cj=$ckdir/corrections-round-$round.jsonl
		rj=$ckdir/rewards-round-$round.jsonl
		rwopt=(); [ "$reward" -eq 1 ] && { : > "$rj"; rwopt=(--rewards "$rj" --reward-min "$reward_min"); }
		tools/teacher-correct.sh "$corpus" "$persona" "$vocab" "$cj" "$round" "$trained" --count "$corrections" --tables "$tables" --seed "$round" --log "$log" --record "$record" "${rwopt[@]}" > "$W/correct-$round.out" 2>&1
		[ "$reward" -eq 1 ] && [ -s "${reward_bank:=$ckdir/reward-bank.jsonl}" ] && tools/teacher-reward.sh "$reward_bank" "$persona" "$vocab" "$round" "$trained" "$rj" --count "$reward_count" --samples "$reward_samples" --tables "$tables" --log "$log" > "$W/reward-$round.out" 2>&1
		[ -s "$cj" ] && build_set
	fi
done
