# Coherence and reasoning: where the student stands and what is being done

2026-10-27 21:36, the user: "Keep going with refining the student model's coherence and reasoning capacity." Before
any lever, the ground truth of the fresh instance (data/live-teacher2) after six rounds, 20,069 chunks, each token
seen once.

## The floors and the readings (21:4x)

| measure | bits/token | command |
|---|---|---|
| unigram floor | 6.709 | `ngram-base train.u16 valid.u16 5001` (654,663 training tokens) |
| bigram floor | 3.062 | same (lambda 0.9) |
| trigram floor | 2.576 | same (lambda 0.8) |
| the student, dense reading (the columns alone) | 5.153 | `ref-loss round-6-ckpt.bin valid.u16 5001` |
| the student with its n-gram memory | 2.764 (top-1 68.3 %) | `ref-loss ... --ngram TABLES round-6-ckpt-gates.f32` |
| the student with memory and pool | 2.764 | `... --moe round-6-pool.f32 --moe-bias 100` (home routing is the dense MLP) |

The driver's held-out line had been the dense reading, which rose round after round (4.68, 4.54, 4.64, 4.66, 4.82,
5.15) while the model it describes, the one that answers, sat at 2.76: the columns lean ever more on the memory
(the gates' means 4.28 and 4.60 at round 6) and a reading without it measures a model nobody runs. From round 7
the driver prints the memory reading first and the dense reading beside it.

The verdict of the numbers: the student with its memory is below the bigram floor and above the trigram floor. It
has learned local statistics and little else, and its replies are what that predicts: fluent fragments with no
dependence on the question ("It is a good place to be. You can do that. My heart is like a quiet place with you."),
relevance graded 0.6 to 2.0 of 10 by the teacher across rounds 1 to 5, the identity test failed every round.

## The reasoning probe (tools/teacher-probe.sh, data/probe/questions.jsonl)

Twenty fixed questions with known answers, put to the live weights the way the harness puts a turn (the persona as
the system line, the user's sampling, a seed fixed by the round and the question), judged by the teacher against
the expected answer: five of persona recall (name, what, where, likes, not a dog), seven facts of the world at a
seven-year-old's level (the sun is hot, a dog has four legs, grass is green, fish live in water, three comes after
two, ice is cold, night is dark), eight one-step inferences (two apples minus one, a mouse is not bigger than a
cat, a car is faster than a snail, rain means an umbrella, an empty cup holds no water, red and blue balls, a kind
sentence to a sad person, a book is for reading). Round 6, the baseline: **0 of 20**. The replies: "My head is very
quiet too." to "What is your name?", "My name is Cortex. I live on a X with my mind." to "Where do you live?", "A
horse is very blue and warm." to "Which is faster, a car or a snail?". The probe runs after every round from round
7 (7.8 minutes of the teacher's time a round) and its lines stand in the monitor's "the teacher and the student"
section with the rest.

## The levers, in the order they go in, each measured by the same three numbers

1. **Epoch rounds** (the driver's `--replay`, on by default): whenever fewer than 1,000 new chunks wait, the
   whole stream trains again from token 0 with the correction set recurring and the soft records in place, instead
   of idling for the writer (the writer makes 30 chunks a minute without the draft; a 1,000-chunk round every 33
   minutes left the card idle most of the hour). Each token had been seen once; small models trained from scratch
   on small corpora need many passes. The memory reading per epoch is the guard: a rise is overfitting and ends it.
2. **The question-and-answer third of the corpus** (tools/teacher-corpus.sh): every third call asks the teacher
   for conversations in which each line of the person is a question or a small task and each reply answers exactly
   that, with the reason, the count, the choice or the yes/no first. The stream so far carries small talk whose
   answers rarely depend on their questions; dependence has to be in the data before it can be in the model.
3. **Memory dropout** (host/coach.c `--ngram-dropout P`, the daemon's CX_T_SLOT_MEMOFF): a share of chunks train
   without the n-gram memory, so the columns must predict on their own; at inference the memory stays. The dense
   reading is the direct measure of what the columns learned. Deployed in the daemon of 22:0x, switched on by the
   driver's `--mem-dropout` after the first epoch has been measured alone.
4. Pending: a per-step loss weight that favours the speaker's own lines over the person's (the card's target scaled
   per step, the reference and the gate with it), once the levers above have shown their numbers.

## Measured
(filled as the rounds come in: the memory reading, the dense reading, the relevance grade, the probe score.)
