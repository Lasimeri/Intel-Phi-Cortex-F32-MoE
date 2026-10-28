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
| the records | DIR/train.soft, valid.soft, round-R.soft (tools/mixstream `--soft`, the driver) | one 68-byte record a token (lib/cx-proto.h cx_soft: cx_tok id[8] as u32, float p[8], float w; CX_SOFT_NONE 0xFFFFFFFF unused), in the stream's order, for the token after it; the correction set's tokens and the stream from before the sidecars carry empty records. 48 B (uint16 id[8], float p[8]; 0xFFFF unused) at 19:25, 52 B with the weight w from 22:00, 68 B with u32 ids from 10-28 06:xx (the last section) |
| the ring | host/coach.c `--soft FILE`, the part at 57 MiB (CX_T_SOFT_SLOT) | each chunk's 32 records (2,176 B; 1,536 B at 48 B a record) go into the chunk's slot of a second ring beside the token ring before the chunk is published; CX_T_SOFT tells the daemon |
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

## Measured in the run (round 4 of the fresh instance, 20:25 to 20:28, the daemon of 20:08)

The coach: `soft targets data/live-teacher2/round-4.soft: 19029 of 33664 tokens of the run carry the teacher's
distribution (56.5 %)`. The card's count per report window of 100 chunks (its `soft N of 3200`) against the file's
own count for the same chunks (the scratchpad's softwin tool over round-4.soft):

| chunks | the card | the file | ctx us/step | tok/s | training loss (hard CE, bits/token) |
|---|---|---|---|---|---|
| 1..100 | 1862 | 1862 | 289 | 631 | 3.2946 |
| 101..200 | 1817 | 1817 | 343 | 606 | 3.4895 |
| 201..300 | 1864 | 1864 | 385 | 577 | 3.3490 |
| 301..400 | 1724 | 1724 | 418 | 543 | 2.8849 |
| 401..500 | 1757 | 1757 | 456 | 513 | 2.9170 |
| 501..600 | 1799 | 1799 | 482 | 494 | 3.4290 |
| 601..700 | 1751 | 1751 | 479 | 479 | 3.3500 |
| 701..800 | 1853 | 1853 | 494 | 466 | 2.8338 |
| 801..900 | 2005 | 2005 | 507 | 453 | 3.8259 |
| 901..1000 | 1675 | 1675 | 496 | 449 | 2.7808 |

Ten windows, ten exact matches, zero checksum misses (`grep -c checksum cortexd.log` = 0): every record the host
aligned reached the card's target. Round 3's stream, for the record, carried only 35.3 % because its first 400
chunks were the correction set and batches the old driver had encoded without the soft path (padded empty at the
restart); from round 4 on the share is the corpus's own, 52 to 63 % a window.

The held-out (the hard cross-entropy on valid.u16, 5,001 tokens): round 1 4.6801, round 2 4.5369, round 3 4.6361
(hard targets, the zero-read), round 4 4.6594 bits/token. One round of 1,052 chunks with the teacher's distribution
in half the target does not move the hard held-out; what it does to the student's distribution shows over rounds,
not in one, and the judge's grades (round 3: grammar 4.3, relevance 0.6, persona 3.2 of 10) are where it has to
show. The stray `0` after the soft share in the lines of this round (`soft 58 %0`, `soft 1862 of 32000`) was the
bad-slot count printed even at zero; fixed at 20:30, deployed at the next gap.

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

## The 52-byte record and the alignment check (10-28 05:15 to 05:20)

The record carries a loss weight since 22:00 (52 bytes: uint16 id[8], float p[8], float w). The slot checksum proves
the copy to the card, not that the records belong to their tokens, so every stream is now checked against its tokens
(tools/soft-check: the share of distribution-carrying records whose ids hold the actual next token; the alignment
gives the hard target the leftover mass, so an aligned stream reads 100 %, a stream shifted by one token reads 19 to
21 % or under 1 %): train.soft 100.0 % of 364,368 records (controls +1 20.0 %, -1 0.2 %), valid.soft 100.0 % of
19,517, round 18's 100.0 % of 56,438 (05:19 to 05:20). The driver refuses a round that fails it and rebuilds a stream
whose record count differs from its token count instead of padding it. It refused round 18 once: build.sh had no
rule for tools/mixstream, the 48-byte binary of the night wrote 48-byte records into the 52-byte run (4,889,904 bytes
for 101,873 tokens); the rule exists now. On the card, round 18's soft count per window of 100 chunks equals the
file's in every window (1719, 1696, 1719, 1682, 1637, 1735, 1713, 1650 of 3,200), zero checksum misses.

## Token ids are u32 (10-28)

2026-10-28 06:10, the user's request as relayed to the workflow that made the change: "Storing an F32 value as F32
is innaproriate and you need to change that". The reading the change works from (lib/cx-proto.h `cx_tok`): a token
id is a 32-bit value where it is used (the card's chunk_ids and every lookup), so storing it in 16 bits is wrong.
Every place an id is stored is 32-bit in the source from 06:xx; the ids are the same numbers (0 to 4,095 for the
4,096-id vocabulary) in the same order.

| what | since 10-28 06:xx | before |
|---|---|---|
| the type | `cx_tok` = uint32_t, CX_TOK_BYTES 4 (lib/cx-proto.h) | uint16_t |
| a token stream | NAME.u32: little-endian uint32 ids; a chunk of 32 tokens is 128 B of the file; a token count is the file's bytes / 4 | NAME.u16: 64 B a chunk, bytes / 2 |
| a slot of the token ring (CX_T_SLOT, the part at 56 MiB) | 256 B (CX_T_SLOT_BYTES): CX_T_L + 1 = 33 u32 ids (132 B) and the flags; the memory-dropout flag (CX_T_SLOT_MEMOFF) is the u32 at index 48 (byte 192); 1,024 slots = 262,144 B | 128 B: 33 u16 ids, the flag the u16 at index 60 (byte 120) |
| a soft record (cx_soft) | 68 B (CX_SOFT_BYTES, held by a static assertion): cx_tok id[8], float p[8], float w; CX_SOFT_NONE 0xFFFFFFFF unused | 52 B: uint16 id[8], float p[8], float w; 0xFFFF unused |
| a chunk's records (CX_SOFT_CHUNK_BYTES) | 2,176 B = 32 x 68 (34 lines of 64 B); the soft ring's 1,024 slots are 2,228,224 B from part + 57 MiB (to 59.125 MiB), the slots' checksums after them | 1,664 B |
| the slot checksum (CX_T_SOFT_SUM) | the sum of the slot's 272 u64 words (2,176 / 8), written by the coach, recomputed by the card | 208 words (192 with the 48-byte records) |
| the tokenizer | tools/bpe.c: encode writes a .u32 file, decode reads one (encode_word writes cx_tok ids) | .u16 |
| a stream or records of before | tools/ids32.c: `ids32 IN.u16 OUT.u32` widens a stream id for id; `ids32 --soft IN.soft OUT.soft --rec 48\|52` rewrites 48- or 52-byte records as 68-byte cx_soft records (each id widened, 0xFFFF to CX_SOFT_NONE 0xFFFFFFFF, p copied bit for bit, w the record's own for 52 B and 1.0f for 48 B); both modes re-read OUT against IN and exit 1 on any difference | |

What does not move: host/ngtables.h hashes a table row from the ids' values (ng_at, ng_gather and ng_apply take
`const cx_tok *` and hand each id to the hash by value), so every n-gram row stays where the trained tables have it;
the checkpoints, the gates files, the pool and the tables hold weights, not ids, and are untouched; the stories
model's .ngc count files (host/orchestra.h ngc_bucket) keep their u16 layout, a stories-era format the from-scratch
instance never loads. A run on a widened stream therefore sees the same ids in the same order, the same row hashes
and the same records apart from the widened id fields. At the time of writing (10-28 06:2x) nothing had been built
or run on the card with the u32 layout: the live round in progress ran the binaries in build/, built with the u16
layout.
