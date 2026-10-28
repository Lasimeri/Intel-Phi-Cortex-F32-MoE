# The partial distillation: the teacher's distribution in the student's target

2026-10-27 18:43, the user: "Open up the training interface with a fresh training instance, new model from scratch
and everything, adjust the teacher model so it partially distills itself into the PhiLLM." Until then the teacher
(Qwen3.8-27B UD-Q4_K_XL through llama.cpp on the 3090 Ti, docs/TEACHER.md) taught by text alone: the student trained
on the teacher's conversations as hard targets, one token at a time. From this stage the teacher's own next-token
distribution rides along with every token it writes and becomes part of the student's target.

## The path

| step | where | what |
|---|---|---|
| the sidecar | tools/teacher.sh `--lp-out FILE`, the writer tools/teacher-corpus.sh | every generated token with its eight likeliest alternatives and their log probabilities (the server's `logprobs: true, top_logprobs: 8`, the raw softmax of the logits), saved as batch-*.lp.json beside each batch |
| the alignment | tools/soft-align.c | the student tokenises the batch's text with its own 4,096-id BPE; where a student token boundary falls on a teacher token boundary inside a turn, the teacher's alternatives are mapped onto student ids (each alternative's first student token, cut the way the stream is cut), summed per id, the mass the eight do not carry given to the hard target, the whole scaled by lambda (0.5); elsewhere the record is empty |
| the records | DIR/train.soft, valid.soft, round-R.soft (tools/mixstream `--soft`, the driver) | one 48-byte record a token (lib/cx-proto.h cx_soft: uint16 id[8], float p[8]; 0xFFFF unused), in the stream's order, for the token after it; the correction set's tokens and the stream from before the sidecars carry empty records |
| the ring | host/coach.c `--soft FILE`, the part at 57 MiB (CX_T_SOFT_SLOT) | each chunk's 32 records (1,536 B) go into the chunk's slot of a second ring beside the token ring before the chunk is published; CX_T_SOFT tells the daemon |
| the card | card/cluster.h cl_chunk_begin, cl_fwd_step | thread 0 DMAs the slot into GDDR with the chunk's n-gram rows; the step's target becomes (1 - sum p_j) onehot(next) + sum_j p_j onehot(id_j), subtracted in each thread's own vocabulary blocks like the one-hot; the reported loss stays the hard cross-entropy |
| the reference | tools/ref.h ref_head_hc, cref.h, ctrain-check `CX_SOFT=FILE` | the same target in double, for the gate |
| the report | the coach's report line `soft N %`, the daemon's `soft N of M`, the monitor's "distil" row | the share of the window's steps whose record held a distribution |

Why the first token of an alternative: the student's next token at a shared boundary is the first piece of whatever
word comes next, so the teacher's alternative " talking" lands on the student's " talk" when " talking" is two
student tokens, which is the token the student has to produce there. Why the leftover mass goes to the hard target:
the eight alternatives carry 97 to 98 % of the teacher's mass on this corpus (measured below); the rest is given
to the truth rather than renormalised away. Why lambda = 0.5: the user's word was partially; half the target is the
teacher's distribution, half the text. `--lambda` on the driver changes it.

## What the server can and cannot give

The teacher's distribution exists only at the tokens it generates: the prompt's tokens cannot be scored (an
`echo: true` + `logprobs` request on `/v1/completions` killed this llama.cpp build at 18:47, the known issue 27174;
never send one). With the MTP draft on (`--mtp`), the server returned the distribution only at the tokens the target
model sampled itself: 1 of 396 in a batch, the draft's accepted tokens with logprob 0 and no alternatives. The
teacher now serves without the draft (tools/teacher-serve.sh `--np 1`, 19:13): every token carries its eight
alternatives (14 of 14 in the check), at 13.9 against 16.3 tokens a second.

## Measured on the corpus (19:21, five batches written after 19:13)

| batch | student tokens | tokens with a distribution | mapped mass of the eight | the teacher's first choice is the hard target |
|---|---|---|---|---|
| 1 | 466 | 281 (60.3 %) | 0.981 | 90.4 % |
| 2 | 508 | 323 (63.6 %) | 0.974 | 84.8 % |
| 3 | 506 | 306 (60.5 %) | 0.974 | 86.6 % |
| 4 | 465 | 282 (60.6 %) | 0.973 | 84.4 % |
| 5 | 555 | 367 (66.1 %) | 0.981 | 88.6 % |

The tokens without a distribution are the markers, the system line, and the student's inner word pieces (where the
teacher has no boundary). A record as the card sees it (the slice's token 44, lambda 0.5): the student's " to" is
followed by " play"; the target holds " talk" 0.245, " play" 0.182, " hear" 0.020, " watch" 0.017, " sleep" 0.015,
" listen" 0.010, " sit" 0.008, " think" 0.003 (sum 0.500) and the hard target's other half.

## The gate (19:25)

`CX_CTX=1 CX_SOFT=slice.soft ./cluster-gate bwd slice.u16 gS 32 1024 0 59 119` on the card (the dense model from
the seeded weights, the context memory on, 1,024 warm steps, one chunk's forward and backward with 24 of its 32
steps carrying a distribution) against `CX_CTX=1 CX_SOFT=slice.soft build/host/ctrain-check bwd` on the host: the
loss identical (8.653190 nats both, max |err| 1.1e-6), hc 4.7e-7, u 4.1e-7, W 3.9e-7, W1 2.9e-7, Wu 4.5e-7, cg
3.8e-7, E 2.5e-7 relative, the context gates' gradients 3.9e-6. The soft target is exact on the card.

## Running it

The driver (tools/teacher-scratch.sh) does the rest: the sidecar of every new batch is aligned as the batch is
encoded, a stream from before the sidecars is padded with empty records, mixstream mirrors the round's stream into
round-R.soft, the coach feeds it with `--soft`. `--lambda 0` trains on the text alone with the same machinery.

## Measured in the run

(filled at the first round trained with the soft targets.)

## The first deployment lied, and the structure now checks itself (19:56 to 20:02)

The daemon of 19:37 reported `soft 3200 of 3200` for every window of round 3 while the round's file carried a
distribution on 35.3 % of its tokens. The host side was byte-exact (the part's slots at file offset 1,344 MiB + 57
MiB matched round-3.soft word for word); the card's DMA read from bus address 57 MiB of nothing, because the soft
ring's address was computed from the host part's address seven lines before that address was assigned, and the
zeros it brought looked like records (id 0 is a valid id, probability 0 changes nothing): round 3 trained on the
plain one-hot while every report said soft. Two changes: the address is taken after the host part is known, and
every slot now carries a checksum (the sum of its 192 words, written by the coach after the records at
CX_T_SOFT_SUM); the card sums what its DMA brought, trains the chunk on the hard targets alone when the sums
differ, counts the miss (CX_T_SOFT_BAD, "bad slots N" in the daemon's and the coach's lines) and says so in its
log. The cost of the check: 192 additions and one uncached 8-byte read a chunk, under 2 microseconds against a
chunk of 70 ms. A deployment is verified by holding the daemon's `soft N of M` per report window against the
file's own count for the same chunks (the scratchpad's softwin tool), not by the report alone.
