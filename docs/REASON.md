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

The soft record grew from 48 to 52 bytes (68 since 10-28 06:xx, the ids u32: docs/DISTIL.md "Token ids are u32
(10-28)"): a loss weight `w` per token (lib/cx-proto.h cx_soft), by which the card scales the step's whole gradient
(card/cluster.h cl_fwd_step, `invLw`) and the reference does the same (tools/cref.h) while the reported loss stays
the plain cross-entropy. tools/soft-align `--weight-user W` sets w = W
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

## The overnight verdict, corrected by the new metric (05:34)

On the frozen set, the answer bits of the rounds before and after the night:

| round | answer loss, true question (bits/token) | swapped | the question gives (bits/token) | per answer | easier with the true question |
|---|---|---|---|---|---|
| 6 | 5.5982 | 5.8892 | 0.2910 | 0.3268 +- 0.0266 | 235 of 300 |
| 7 | 5.4364 | 5.7369 | 0.3006 | 0.3352 +- 0.0289 | 224 of 300 |
| 16 | 5.1365 | 5.6451 | 0.5086 | 0.5794 +- 0.0383 | 254 of 300 |

Rounds 7 to 16 lowered the answer loss by 0.46 bits per token and raised what the question gives from 0.33 to 0.58
bits per answer, more than six standard errors: the night improved the answers and their dependence on the
question. The held-out (its average dominated by the persona line and the markers every conversation repeats) and the
probe were blind to it; the sentence above, "Epochs at a constant 3e-3 bought nothing measurable", is wrong. The gain
is not the epochs' alone: the writer's question-and-answer third (from 21:54) and new data entered in the same rounds,
and the frozen set is the newest held-out, question-and-answer style. The series of every round stands in
results/answer-bits.log.

## The reward (05:27, the user: "The student model needs a reward function for when it gets something right")

The reward reinforces the student's own right replies, the recipe of STaR (Zelikman et al. 2022) and of
rejection-sampling fine-tuning: a reply the teacher grades right becomes a training target, weighted by its reward.
This reverses the rule of 10-27 that the student's own words never enter the stream; it does so only for replies the
teacher verified, and the wrong ones keep getting the teacher's rewrite.

| piece | what |
|---|---|
| the verdict | the correction pass (tools/teacher-correct.sh `--rewards FILE --reward-min 7`): a reply whose three grades (grammar, relevance, persona) are all at least 7 is right; the conversation up to the person's line, closed by the student's own reply, goes to DIR/rewards-round-R.jsonl with reward 1 |
| the records | tools/reward-soft.c: the loss weight of the reply's tokens and its closing `<|end|>` is the reward times `--reward-weight`, every other token of the conversation 0 (the context is the teacher's text and the persona, trained already) |
| the stream | the driver (v6) appends the reward conversations of the last 5 rounds to the correction set, which recurs once every 24 corpus conversations; mixstream `--set-soft` carries the set's own records to the round's stream; the card scales each step's gradient by its record's weight (gated at 22:00) |
| the check | on real data the set builds with one record per token (2,223 for 2,223) and every reinforced span lands on the reply alone ("No, I do not wear a skirt. I am a language model on a Xeon Phi card." at weight 1, its context at 0) |

The replies the teacher grades right are rare (relevance 0.4 to 2.6 of 10 through round 17), so the first rounds will
show how many rewards a pass yields; if it is a handful, the next step is several samples per context (the best of N,
or a group-relative advantage, which the signed weight of the record allows).

## The held-out leak, measured and closed (05:4x)

The correction pass drew its contexts from the newest 60 batch files, held-out batches included, and trains each
context verbatim: 3 of the 60 conversations of the answer-bits set (first person line and first reply both matching)
had entered the correction sets of rounds 13 to 15. The pass now draws only from batches the instance has encoded for
training (every --valid-every-th line of DIR/work/encoded.txt excluded, and a batch not yet encoded never drawn, as it
may still become held-out); the draw pool of 05:45 overlaps the held-out batches in 0 of 60. The measured set is now
the 56 conversations without the leaked ones (DIR/answer-bits-set.jsonl; the 60 kept as answer-bits-set-60.jsonl).
On it the night's gain stands:

| round | answer loss (bits/token) | the question gives | per answer | easier |
|---|---|---|---|---|
| 6 | 5.5309 | 0.2856 | 0.3216 +- 0.0265 | 223 of 280 |
| 16 | 5.0556 | 0.5141 | 0.5928 +- 0.0399 | 237 of 280 |

## The reward made dense: the bank (05:5x)

In the student's record 7 of 328 graded replies (2.1 %) had all three grades at 7 or more: the correction pass alone
would reward about 0.4 replies a pass. The reward therefore also comes from a bank of questions with known answers
(tools/teacher-bank.sh: the teacher writes them by category, about the speaker, colours, animals, hot and cold, small
sums, yes or no, a choice of two, what to do, a kind reply; the probe's twenty questions and any question holding most
of a probe question's words with the same answer are dropped, so "How many legs does a bird have?" stays and "How many
legs does a dog have, can you tell me?" goes). Each round the student answers 40 of them twice under the user's
sampling, the teacher verifies every answer against its expected answer with the probe's judge, and every right
answer that closed itself is reinforced (tools/teacher-reward.sh, the driver's --reward-bank). A paraphrase of a
persona question with a different wording ("Tell me what you are called.") can pass the filter, so the probe's first
five items (the persona) no longer measure far transfer; the decision metric is answer bits.

Attribution from here: round 19 is the epoch learning rate alone (1e-3); from round 20 the reward and the corrections
graded against the student's record train too. Grades written after 05:33 come from a teacher that sees the record
and are not comparable with the grades before.

## The reward's cold start, and the group reward (07:2x)

Round 19's pass, the first with the bank: 0 of 19 replies of the correction pass graded right, 0 of 80 answers of the
bank judged right. A reward that waits for a right answer gives a student that answers nothing right no signal at all.
The bank's reward is therefore graded and relative from round 21 (tools/teacher-reward.sh `--mode rloo`, the default):
30 questions a round, 4 answers each; the teacher scores every answer 0 to 10 against the expected answer (10 the
expected answer plainly, 5 to 7 about the question and partly right, 2 to 4 on the topic without answering, 1 fluent
and unrelated, 0 nonsense), R = score / 10, and within a question's group each answer's advantage is its R minus the
mean R of the other answers: REINFORCE leave-one-out (Kool et al. 2019; Ahmadian et al. 2024, "Back to Basics",
arxiv 2402.14740), an unbiased baseline without a value network; GRPO (Shao et al. 2024, arxiv 2402.03300) would also
divide by the group's standard deviation, which is not used here. The advantage, clipped to [-0.25, 1], is the loss
weight of the answer's tokens on the card (tools/reward-soft.c, the record's w): an answer nearer the expected one than
its siblings is pushed up, one further away is pushed down a little, so every group teaches even when nothing in it is
right. A hand check of the arithmetic: scores 6, 3, 3, 0 give +0.40, 0, 0 and -0.25 (the clip). The samples are used
in the round after they were drawn only (--reward-rounds 1, on-policy as far as one round allows), and the set recurs
every 12 corpus conversations instead of 24 so its conversations train at all (at 24 a round of 3,700 chunks gave the
set about 35 slots for 120 conversations).

## Three corrections to the reward, and a second frozen set (07:5x)

- **Scheduling.** The rewards rode in the recurring set behind the 20 corrections, and mixstream cycles the set in file
  order from the start every round: a new-data round (about 3,700 chunks at ratio 12, some 70 set slots) never
  reached the tail of up to 120 reward conversations, an epoch repeated each about five times. From round 22 the
  rewards are a set of their own (DIR/rewards-once.u32 and .soft) that `mixstream --once` puts into the next round
  exactly once each, spread over its base conversations (the ones the spread cannot place when the stream is full are
  appended after it). Tested on real data: 20 once-conversations all spread, records equal to tokens (64,721),
  alignment 100.0 %, every negative and zero weight intact.
- **The closing token of a worse answer.** A negative reward pushed down every token of the answer through its
  `<|end|>`; the group admits only answers that closed themselves, so that taught the student not to stop. The closing
  `<|end|>` of a negative span now keeps weight 0 (tools/reward-soft.c).
- **The negative weights on the card.** The 22:00 gate had tested positive weights only. Gated at 07:50: the u32
  cluster-gate with w = -0.25 on 22 of the measured chunk's 32 steps against the double reference: every gradient
  within 9.3e-07 relative, the gradients changed as the weights demand (column 0's W max 2.15e-03 at w = 1, 1.76e-03
  here), the reported loss unweighted at 8.425488.

The answer-bits series has a saw-tooth: every epoch over the whole stream raised the answer loss on the frozen set of
the newest held-out conversations, every new-data round lowered it. The set is the newest held-out, question-and-answer
style, so a round's recency is confounded with its quality. A second frozen set, DIR/answer-bits-set-old.jsonl, holds
the 60 oldest held-out conversations (batches of 10-27 13:57 to 15:28, before the question-and-answer third; none of
them in any correction or reward file), measured by a second watcher into results/answer-bits-old.log for rounds 6 on.
A gain counts only when it shows on both sets.
