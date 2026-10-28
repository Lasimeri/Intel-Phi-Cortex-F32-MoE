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

## Lever 4 built and gated (22:00)

The soft record grew from 48 to 52 bytes: a loss weight `w` per token (lib/cx-proto.h cx_soft), by which the card
scales the step's whole gradient (card/cluster.h cl_fwd_step, `invLw`) and the reference does the same
(tools/cref.h) while the reported loss stays the plain cross-entropy. tools/soft-align `--weight-user W` sets w = W
on every token that predicts a token inside a person's line (the turns alternate, the person first) and 1
elsewhere; the driver bakes the weight into the streams and rebuilds them when it changes (DIR/soft-format.txt).
The gate on the card (the dense model, the context memory on, 1,024 warm steps, a chunk with 25 of 32 steps soft
and the person's lines at 0.5) against the double reference: loss identical (8.425488 nats), hc 3.9e-7, u 2.9e-7,
W 1.1e-6, W1 2.7e-7, cg 4.7e-7, E 3.5e-7 relative, the context gates 3.2e-7. Deployed with the format change at
the next gap, at weight 1 until the levers before it have been measured.

## Overnight and the morning of 10-28: what the numbers say

Rounds 7 to 16 (epochs at lr 3e-3 alternating with new-data rounds, dropout off, weight 1). The memory reading of
the fixed held-out (the first 5,001 tokens of valid.u16): 2.7636 (r6), 2.6320, 2.6871, 2.6725, 2.7310, 2.8079,
2.7211, 2.7241, 2.6396, 2.7297, 2.6155 (r16) bits/token: flat. The probe: 1 to 3 of 20 every round. The relevance
grades 0.4 to 2.6 of 10. Epochs at a constant 3e-3 bought nothing measurable.

The n-gram gates ran away: means 4.28/4.60 at r6, 6.22/8.99 at r7, 7.70/13.59 at r10, 9.61/22.27 at r13,
10.39/25.06 at r16. Adam moves 256 gate floats by about the learning rate each chunk while the gradient keeps its
sign, so the gated rows came to dominate the column's input (card/train.h adds g * r into the embedding before
cx_col_pre). Every prediction still passes through the columns; the dense reading's rise (6.08 to 8.58) is the model
with its dominant input deleted, not starved columns. A mechanism to test, not yet measured: with x = RMSNorm([p; r])
a growing p shrinks the bus term's share.

The meter was checked before any lever: tools/ref-loss had no context memory, so every memory reading since 16:44 had
read the model minus its retrieval. With `--ctx` (the context gates from the 1,536-B gates file) round 16 reads
2.6155 again (top-1 0.6868 against 0.6870): the retrieval moves the held-out by less than 0.0001 bits, and the
saved round-16 tables reproduce the driver's number exactly. The flat series is the model.

## The decision metric: answer bits (tools/answer-bits.sh, tools/answer-bits.c)

The probe cannot resolve a gain at 1 to 3 of 20. The new metric: the same 60 held-out conversations (frozen in
DIR/answer-bits-set.jsonl) read twice, as written and with every person's line swapped for the line of the same turn
seven conversations on; the speaker's tokens are the same in both and pair one to one. The answer loss with the true
question, and the bits the question gives (swapped minus true), per answer token, with the standard error over the
300 answers. Round 16: answer loss 5.1365 bits/token, swapped 5.6451: the question gives 0.5086 bits per answer
token (per-answer mean 0.5794 +- 0.0383), 0.9980 over the first five tokens of each answer, and 254 of 300 answers
are easier after their true question. The student does use the question; the 2.6 of the held-out is flattered by
the persona line and the markers every conversation repeats, the speaker's own tokens cost 5.14 bits. A change is a
gain when the answer loss falls and the question's bits rise by more than twice the standard error, for two rounds.
tools/answer-bits-watch.sh measures every round as it lands (results/answer-bits.log).

## Deployed 05:15 to 05:20

The first lever is the learning rate of the epoch rounds, 1e-3 (the driver's `--epoch-lr`); memory dropout stays off
(at gate means of 10 to 25 a dropped chunk shifts the input by an order of magnitude, and it has never run end to
end). The record format grew to 52 bytes with the per-step weight at 1; the driver checks every round's records
against its tokens and refuses a round that does not align (it refused round 18 at 05:19 when a stale mixstream
binary wrote 48-byte records; build.sh had no rule for it, now it has). The teacher serves 65,536 tokens shared by
its two slots and its grader reads the student's record (docs/TEACHER.md).
