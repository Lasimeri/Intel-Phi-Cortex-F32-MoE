#!/bin/bash
# teacher-corpus.sh: the teacher writes the whole training corpus of a model trained from scratch (2026-10-27 13:09,
# user: "Train a model from scratch just using the teacher model"): conversations between a person and the speaker
# (the persona, PERSONA.txt) on everyday topics the teacher lists first by themes, the person in plain everyday
# English, the speaker in the first person as the persona in short simple sentences, about itself only when asked.
# The recipe is TinyStories' (Eldan & Li 2023, docs/TEACHER.md "Methods from the literature"): a restricted vocabulary
# (words a seven-year-old knows; the teacher lists 300 of them once, OUT_DIR/words.txt), and diversity forced by three
# random words of that list that every call must use plus a random mood or intent of the person; UltraChat's
# meta-topics are the ten themes of twenty topics. Runs until OUT_DIR/stop exists or --max conversations are written.
# P writers run at once against the server's P slots (tools/teacher-serve.sh --np P; one slot with the MTP draft
# measured best on the 200 W card). A call asks for --per-call conversations (default 3) as free text holding one JSON
# object, under a cap of 2,400 tokens, with the repeat penalty at 1.0 for these calls only: the JSON grammar mode
# bloated the tokens to 900 a conversation and the penalty of 1.2 drove every long JSON reply into a word list inside
# a string (measured 13:3x); without the grammar a conversation is about 130 tokens (13:5x). The complete conversation
# objects of the reply are salvaged one by one (tools/json-objects.c): in about half the calls the teacher writes its
# closing delimiter in place of the object's final brace (13:4x: 7 of 14 calls, every one holding its three whole
# conversations), so the whole reply is never the unit. Output: OUT_DIR/topics.txt, OUT_DIR/words.txt;
# OUT_DIR/batch-<ns>-wN.jsonl, one file per call in the messages shape (the persona as the system message, the turns
# alternating from the person), which the from-scratch driver (tools/teacher-scratch.sh) encodes as they land;
# OUT_DIR/corpus.log, a line per batch with the count, the bytes, the seconds, the topic, the words and the mood; the
# raw text of a call that yields nothing kept as OUT_DIR/failed-<ns>.txt (the last ten). The teacher's text is model
# output; the model trains on it by the user's decision of 10:21 and 13:09.
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
[ -x build/host/json-objects ] || { echo "teacher-corpus: build/host/json-objects missing (run build.sh)" >&2; exit 1; }
mkdir -p "$out"
log=$out/corpus.log
say() { echo "$(date +%T) $*" | tee -a "$log"; }
rm -f "$out/stop"

# the topics, once, by themes (asked for "more topics unlike these" the teacher repeated one phrase in variants, 13:1x)
if [ ! -s "$out/topics.txt" ]; then
	cat > "$out/topics-system.txt" <<'EOF'
You list conversation topics. Answer with a JSON object only: {"topics": ["...", "..."]}. Each topic is a few plain words naming one thing two friends might chat about in everyday life, every topic about a different thing, nothing technical, nothing dark.
EOF
	themes=("food and cooking" "the weather and the seasons" "pets and animals" "walks, parks and the outdoors" "work and school" "hobbies and games" "family and friends" "feelings, worries and good days" "plans, trips and the weekend" "memories, dreams and small questions about the speaker itself")
	say "the topics: asking the teacher for $topics over ${#themes[@]} themes"
	: > "$out/topics.txt"
	per_theme=$(( (topics + ${#themes[@]} - 1) / ${#themes[@]} ))
	for theme in "${themes[@]}"; do
		tools/teacher.sh "$out/topics-system.txt" "Give $per_theme topics about $theme, each about a different thing." --max 1200 --json "${urlopt[@]}" 2>/dev/null | jq -r '.topics[]? | tostring | gsub("\\s+"; " ") | ltrimstr(" ") | rtrimstr(" ") | select(. != "")' >> "$out/topics.txt" 2>/dev/null
	done
	sort -u -f "$out/topics.txt" -o "$out/topics.txt"
	say "topics: $(wc -l < "$out/topics.txt") in $out/topics.txt"
fi
ntopics=$(wc -l < "$out/topics.txt")
[ "$ntopics" -gt 0 ] || { say "no topics: stopping"; exit 1; }

# the simple words, once: the seeds that force diversity (TinyStories' three random words a story)
if [ ! -s "$out/words.txt" ]; then
	cat > "$out/words-system.txt" <<'EOF'
You list simple English words. Answer with a JSON object only: {"words": ["...", "..."]}. Every word is one a seven-year-old child knows and uses: everyday nouns, verbs and adjectives, all different, no names, nothing technical, nothing dark.
EOF
	say "the words: asking the teacher for 300 simple words"
	tools/teacher.sh "$out/words-system.txt" "Give 300 words: 150 nouns, 100 verbs, 50 adjectives." --max 2500 --rp 1.0 --json "${urlopt[@]}" 2>/dev/null | jq -r '.words[]? | tostring | ascii_downcase | gsub("[^a-z]"; "") | select(length > 1)' | sort -u > "$out/words.txt" 2>/dev/null
	say "words: $(wc -l < "$out/words.txt") in $out/words.txt"
fi
nwords=$(wc -l < "$out/words.txt")
[ "$nwords" -ge 20 ] || { say "too few words ($nwords): stopping"; exit 1; }
moods=("curious" "worried about something" "cheerful" "tired" "in a hurry" "joking around" "a little sad" "excited" "bored" "curious and asks the speaker who it is at some point" "friendly and asks the speaker what it likes at some point" "friendly and asks the speaker where it lives at some point" "chatty and tells a small story of its own" "shy and asks short questions")

cat > "$out/system.txt" <<EOF
You write training dialogue for a very small chat model that speaks in the first person as the persona you are given. Write in short, plain, simple English, using only words a seven-year-old child knows: no rare words, no idioms, no lists, no markdown, no quotation marks inside the text. Each conversation is between a person and the speaker, about the topic given, $exch exchanges long: the person speaks first, the speaker answers, and so on. The person's mood or intent is given and shapes what the person says. Now and then the person asks the speaker who or what it is, where it lives or what it likes; only then does the speaker say something about itself, from the persona and nothing else: its name is in the persona, so is where it runs and what it likes; it is one single self and never speaks of parts, heads, columns or siblings inside itself; it never calls itself an assistant or an AI and never says it has a persona. The rest of the time the speaker talks about the topic only and never mentions itself, its name, its card or what it likes. Every turn is one or two short sentences. The three given words must each appear somewhere in the conversations, used naturally. Answer with a JSON object only: {"conversations": [{"turns": [{"user": "...", "assistant": "..."}, ...]}, ...]}.
EOF

# one writer: its share of the topics, round after round, a batch file per call
writer() {
	local w=$1 n=0
	while [ ! -f "$out/stop" ]; do
		for ((i = w; i < ntopics; i += par)); do
			[ -f "$out/stop" ] && return
			topic=$(sed -n "$((i + 1))p" "$out/topics.txt")
			n=$((n + 1))
			local stamp t0 t1 file tmp count held bytes conv words mood
			words=$(shuf -n 3 "$out/words.txt" | tr '\n' ',' | sed -E 's/,$//; s/,/, /g')
			mood=${moods[$((RANDOM % ${#moods[@]}))]}
			stamp=$(date +%s%N)
			file=$out/batch-$stamp-w$w.jsonl
			tmp=$out/.w$w
			t0=$(date +%s.%N)
			if ! tools/teacher.sh "$out/system.txt" "The persona:
$(cat "$persona")

The topic: $topic
The person is $mood.
The three words to use: $words.
Write $per conversations about the topic, $exch exchanges each." --max 2400 --rp 1.0 "${urlopt[@]}" > "$tmp.txt" 2> "$tmp.err"; then
				say "writer $w: call failed on \"$topic\": $(tail -n 1 "$tmp.err" | cut -c1-120)"
				sleep 5
				continue
			fi
			# the reply's complete conversation objects, one a line, whatever surrounds them
			build/host/json-objects 3 < "$tmp.txt" > "$tmp.objs" 2>/dev/null
			held=$(wc -l < "$tmp.objs")
			: > "$file"
			# each object on its own, so one bad object costs only itself; a turn over 220 characters (the teacher drifting
			# into a word list inside a string, seen under the repeat penalty of 1.2) or an answer that steps out of the
			# role drops its whole conversation
			while IFS= read -r conv; do
				printf '%s\n' "$conv" | jq -c --rawfile p "$persona" 'select(.turns and (.turns | length) > 0) | { messages: ([{ role: "system", content: ($p | rtrimstr("\n")) }] + [ .turns[] | select(.user and .assistant) | ({ role: "user", content: (.user | tostring | gsub("\\s+"; " ") | ltrimstr(" ") | rtrimstr(" ")) }, { role: "assistant", content: (.assistant | tostring | gsub("\\s+"; " ") | ltrimstr(" ") | rtrimstr(" ")) }) ]) } | select((.messages | length) >= 3) | select([.messages[1:][] | .content | length > 220] | any | not) | select([.messages[] | select(.role == "assistant") | .content | test("persona|the user|as an ai|assistant"; "i")] | any | not)' >> "$file" 2>/dev/null
			done < "$tmp.objs"
			count=$(wc -l < "$file")
			if [ "$count" -eq 0 ]; then
				rm -f "$file"
				cp "$tmp.txt" "$out/failed-$stamp.txt" 2>/dev/null
				ls -t "$out"/failed-*.txt 2>/dev/null | tail -n +11 | xargs -r rm -f
				say "writer $w: nothing usable on \"$topic\" ($held objects, $(wc -c < "$tmp.txt") bytes raw, kept as failed-$stamp.txt)"
				continue
			fi
			bytes=$(stat -c %s "$file")
			t1=$(date +%s.%N)
			say "writer $w batch $n: $count of $held conversations, $bytes bytes, $(awk -v a="$t0" -v b="$t1" 'BEGIN { printf "%.1f", b - a }') s, \"$topic\", words $words, the person $mood"
			if [ "$max" -gt 0 ] && [ "$(cat "$out"/batch-*.jsonl 2>/dev/null | wc -l)" -ge "$max" ]; then touch "$out/stop"; fi
		done
	done
}

say "=== corpus writers: $par at once, $per conversations of $exch exchanges a call, $ntopics topics, $nwords seed words, persona $persona, into $out (touch $out/stop to end)"
pids=()
for ((w = 0; w < par; w++)); do writer "$w" & pids+=($!); done
wait "${pids[@]}"
say "=== corpus writers ended: $(cat "$out"/batch-*.jsonl 2>/dev/null | wc -l) conversations in $(ls "$out"/batch-*.jsonl 2>/dev/null | wc -l) batches"
