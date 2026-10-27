#!/bin/bash
# teacher-corpus.sh: the teacher writes the whole training corpus of a model trained from scratch (2026-10-27 13:09,
# user: "Train a model from scratch just using the teacher model"): conversations between a person and the speaker
# (the persona, PERSONA.txt), on everyday topics the teacher lists first, the person in plain everyday English, the
# speaker in the first person as the persona in simple short sentences, itself whenever asked who or what it is.
# Runs until OUT_DIR/stop exists or --max conversations are written. P writers run at once against the server's P
# slots (tools/teacher-serve.sh --np P). Output: OUT_DIR/topics.txt; OUT_DIR/batch-NNNNNN.jsonl, one file per
# call in the messages shape (the persona as the system message, the turns alternating from the person), which the
# from-scratch driver (tools/teacher-scratch.sh) encodes as they land; OUT_DIR/corpus.log, a line per batch with the
# count, the bytes and the seconds. The teacher's text is model output; the model trains on it by the user's
# decision of 10:21 and 13:09.
#   tools/teacher-corpus.sh PERSONA.txt OUT_DIR [--topics N] [--per-call K] [--exchanges T] [--parallel P]
#                           [--max M] [--url URL]
set -u
cd "$(dirname "$0")/.." || exit 1
[ $# -ge 2 ] || { echo "usage: tools/teacher-corpus.sh PERSONA.txt OUT_DIR [--topics N] [--per-call K] [--exchanges T] [--parallel P] [--max M] [--url URL]" >&2; exit 2; }
persona=$1; out=$2
shift 2
topics=200; per=3; exch=5; par=1; max=0; urlopt=()
while [ $# -gt 0 ]; do
	case "$1" in
	--topics) topics=$2; shift 2 ;;
	--per-call) per=$2; shift 2 ;;
	--exchanges) exch=$2; shift 2 ;;
	--parallel) par=$2; shift 2 ;;
	--max) max=$2; shift 2 ;;
	--url) urlopt=(--url "$2"); shift 2 ;;
	*) echo "teacher-corpus: unknown option $1" >&2; exit 2 ;;
	esac
done
[ -f "$persona" ] || { echo "teacher-corpus: $persona not found" >&2; exit 1; }
mkdir -p "$out"
log=$out/corpus.log
say() { echo "$(date +%T) $*" | tee -a "$log"; }
rm -f "$out/stop"

# the topics, once
if [ ! -s "$out/topics.txt" ]; then
	cat > "$out/topics-system.txt" <<'EOF'
You list conversation topics. Answer with a JSON object only: {"topics": ["...", "..."]}. Each topic is a few plain words naming something two friends might chat about in everyday life (the weather, a pet, breakfast, a walk, a job, a hobby, a worry, a plan for the weekend, a memory, a question about the speaker itself), all different, nothing technical, nothing dark.
EOF
	say "the topics: asking the teacher for $topics"
	: > "$out/topics.txt"
	tries=0
	while [ "$(wc -l < "$out/topics.txt")" -lt "$topics" ] && [ "$tries" -lt 20 ]; do
		tries=$((tries + 1))
		have=$(wc -l < "$out/topics.txt")
		tools/teacher.sh "$out/topics-system.txt" "Give $((topics - have < 40 ? topics - have : 40)) more topics, different from these: $(tail -n 40 "$out/topics.txt" | tr '\n' ';' | cut -c1-600)" --max 1200 --json "${urlopt[@]}" 2>/dev/null | jq -r '.topics[]? | tostring | gsub("\\s+"; " ") | ltrimstr(" ") | rtrimstr(" ") | select(. != "")' | /usr/bin/grep -v -i -x -F -f "$out/topics.txt" >> "$out/topics.txt" 2>/dev/null
		sort -u -f "$out/topics.txt" -o "$out/topics.txt"
	done
	say "topics: $(wc -l < "$out/topics.txt") in $out/topics.txt"
fi
ntopics=$(wc -l < "$out/topics.txt")
[ "$ntopics" -gt 0 ] || { say "no topics: stopping"; exit 1; }

cat > "$out/system.txt" <<EOF
You write training dialogue for a very small chat model that speaks in the first person as the persona you are given. Write in short, plain, simple English, the kind of sentences a child reads easily: no rare words, no idioms, no lists, no markdown, no quotation marks inside the text. Each conversation is between a person and the speaker, about the topic given, $exch exchanges long: the person speaks first, the speaker answers, and so on. The person is friendly and curious and sometimes asks the speaker who or what it is, where it lives, how big it is or what it likes; then the speaker answers from the persona and nothing else: its name is in the persona, so is where it runs, how many columns it has and what it likes; it never calls itself an assistant or an AI and never says it has a persona. Otherwise the speaker simply talks about the topic as itself, one to three short sentences a turn. Answer with a JSON object only: {"conversations": [{"turns": [{"user": "...", "assistant": "..."}, ...]}, ...]}.
EOF

# one writer: its share of the topics, round after round, a batch file per call
writer() {
	local w=$1 n=0
	while [ ! -f "$out/stop" ]; do
		for ((i = w; i < ntopics; i += par)); do
			[ -f "$out/stop" ] && return
			topic=$(sed -n "$((i + 1))p" "$out/topics.txt")
			n=$((n + 1))
			local stamp t0 t1 file tmp count bytes
			stamp=$(date +%s%N)
			file=$out/batch-$stamp-w$w.jsonl
			tmp=$out/.w$w.json
			t0=$(date +%s.%N)
			if ! tools/teacher.sh "$out/system.txt" "The persona:
$(cat "$persona")

The topic: $topic
Write $per different conversations about it, $exch exchanges each." --max 3000 --json "${urlopt[@]}" > "$tmp" 2> "$out/.w$w.err"; then
				say "writer $w: call failed on \"$topic\": $(tail -n 1 "$out/.w$w.err" | cut -c1-120)"
				sleep 5
				continue
			fi
			jq -c --rawfile p "$persona" '.conversations[]? | select(.turns and (.turns | length) > 0) | { messages: ([{ role: "system", content: ($p | rtrimstr("\n")) }] + [ .turns[] | select(.user and .assistant) | ({ role: "user", content: (.user | tostring | gsub("\\s+"; " ") | ltrimstr(" ") | rtrimstr(" ")) }, { role: "assistant", content: (.assistant | tostring | gsub("\\s+"; " ") | ltrimstr(" ") | rtrimstr(" ")) }) ]) } | select((.messages | length) >= 3) | select([.messages[] | select(.role == "assistant") | .content | test("persona|the user|as an ai|assistant"; "i")] | any | not)' "$tmp" > "$file" 2>/dev/null
			count=$(wc -l < "$file")
			if [ "$count" -eq 0 ]; then rm -f "$file"; say "writer $w: nothing usable on \"$topic\""; continue; fi
			bytes=$(stat -c %s "$file")
			t1=$(date +%s.%N)
			say "writer $w batch $n: $count conversations, $bytes bytes, $(awk -v a="$t0" -v b="$t1" 'BEGIN { printf "%.1f", b - a }') s, \"$topic\""
			if [ "$max" -gt 0 ] && [ "$(cat "$out"/batch-*.jsonl 2>/dev/null | wc -l)" -ge "$max" ]; then touch "$out/stop"; fi
		done
	done
}

say "=== corpus writers: $par at once, $per conversations of $exch exchanges a call, $ntopics topics, persona $persona, into $out (touch $out/stop to end)"
pids=()
for ((w = 0; w < par; w++)); do writer "$w" & pids+=($!); done
wait "${pids[@]}"
say "=== corpus writers ended: $(cat "$out"/batch-*.jsonl 2>/dev/null | wc -l) conversations in $(ls "$out"/batch-*.jsonl 2>/dev/null | wc -l) batches"
