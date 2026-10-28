/*
 * cx-proto.h: Cortex's part of the shared host-memory region. The host maps it from /dev/shm/phi-hostmem, the card
 * from /dev/phihost (uncached). Little-endian; offsets in bytes from the part's start. Host-written and card-written
 * fields sit on separate 64-byte lines; each side only polls lines the other side writes.
 *
 * The part sits at 1344 MiB of the 2 GiB region, 128 MiB long (ends at 1472 MiB). Neighbours (from their protocol
 * headers, 2026-10-25): below ends near 1304 MiB, above starts at 1536 MiB.
 */
#ifndef CX_PROTO_H
#define CX_PROTO_H

#include <stdint.h>

#define CX_MAGIC 0x58545243u       /* "CRTX" */
#define CX_VERSION 1
#define CX_PART_MIB 1344
#define CX_PART_SIZE (128u << 20)

/* written by the host */
#define CX_H_MAGIC 0               /* u32 */
#define CX_H_VERSION 4             /* u32 */
#define CX_H_SESSION 8             /* u64 raised by each host open */
#define CX_H_CMD 64                /* u64 step command: sequence in bits 0-47, argument in bits 48-63 */

/* written by the card */
#define CX_C_SESSION 128           /* u64 the session the card has taken */
#define CX_C_STATE 136             /* u32 0 starting, 1 ready, 2 error, 3 stopped */
#define CX_C_ACK 192               /* u64 the last command sequence finished */
#define CX_C_HEART 256             /* u64 counted up while the card side runs */

#define CX_CMD_SEQ(c) ((c) & 0xFFFFFFFFFFFFull)
#define CX_CMD_ARG(c) ((uint32_t)((c) >> 48))
#define CX_CMD(seq, arg) ((uint64_t)(seq) | (uint64_t)(arg) << 48)
#define CX_CMD_QUIT 0xFFFFFFFFFFFFull   /* sequence value that stops a gate program */

#define CX_DATA (64u << 10)        /* card -> host payload area (Stage 0e gate) */

/* Stage 5: the generation daemon (card/cortexd.c, host/cortex.c) */
#define CX_OP_LOAD 1               /* arg bit 0: the n-gram gates (2 x 128 F32) follow the checkpoint in the stage and the memory is on; bit 1: MoE
                                      (stage 2): the pool is filled from the checkpoint's MLP blocks (expert e = column e mod 120) and the model routes
                                      through it with the bonus at CX_M_BIAS and the experts' step at CX_M_LR; bit 2 (with bit 1): the pool was pulled
                                      in rounds before this LOAD (CX_T_POOL) and is kept as it is */
#define CX_OP_RESET 2              /* every column's fast weights and the bus to zero; the step count of the memory ring restarts */
#define CX_OP_STEP 3               /* arg: the token id; the card answers with the 4096 logits at CX_LOGITS, then the ACK */
#define CX_OP_BUS 4                /* arg: 1 bus on, 0 off (must match the checkpoint's training) */
#define CX_OP_EXIT 5               /* the daemon ends this session and exits (frees the control socket) */
#define CX_OP_CTX_RESET 7          /* stage 3 (card/ctx.h): the session's view of the context memory starts at the next step (a new chat) */
#define CX_OP_NGRAM 6              /* arg 1: the session writes each step's two memory rows at CX_NGS_ROWS before the STEP (the memory is on for its
                                      steps when the daemon holds gates); 0: no rows, the plain model. A session starts at 1 */
#define CX_H_CMD_OP 72             /* u32 the op code of the current command (written before CX_H_CMD) */
#define CX_LOGITS (128u << 10)     /* 4096 F32, written by the card's DMA; CX_C_ACK is the DMA status word */
#define CX_STAGE (1u << 20)        /* the staged checkpoint, 23,463,424 B, host-written before CX_OP_LOAD */
#define CX_STAGE_CKPT ((120u * 44512u + 128u + 4096u * 128u) * 4u)   /* its length: 120 columns, cg, the embedding in blocks */
#define CX_STAGE_FLOATS (CX_STAGE_CKPT / 4u)
#define CX_STAGE_GATES (CX_STAGE + CX_STAGE_CKPT)                    /* the gates block, CX_GATES_FLOATS F32 (1,536 B): the n-gram gates (2 x 128), then
                                                                        the context gates (stage 3, one a column, 2026-10-27); staged before LOAD (arg bit 0 =
                                                                        loaded), and pushed after the checkpoint by a live save (the push's last line) */
#define CX_GATES_CTX (CX_NG_ORDERS * CX_DE)                           /* the context gates' offset in the block */
#define CX_GATES_FLOATS (CX_GATES_CTX + 128)                          /* 384: 120 context gates and 8 of padding */
#define CX_CTX_GATE0 (-4.0f)                                          /* a fresh context gate (card/ctx.h): sigmoid(-4) = 0.018, a trained model continues as it was */

/* ---- MoE stage 1: the n-gram row ring (PLAN-MOE.md). Per chunk of 32 steps the host gathers 64 rows (2 orders x 32
 * steps x 128 F32 = 32 KiB) into row slot chunk % 2 and publishes CX_NGR_H_SEQ = chunks published; the card DMAs the
 * slot into GDDR at the chunk start and, at the chunk end, DMAs the 64 row gradients into gradient slot chunk % 2 with
 * CX_NGR_C_SEQ as the DMA status word (= chunks whose gradients have landed). The host applies the gradients to its
 * tables and gathers chunk + 2 into the slot the card has finished with. In a live run (coach --ngram) the chunks are
 * the run's, numbered from 0 as the daemon counts them (CX_T_C_CHUNKS), and the daemon takes a chunk only once its
 * rows are published, so it never waits in the ring. */
#define CX_NGR_BASE (48u << 20)
#define CX_NGR_BYTES 32768u
#define CX_NGR_ROWS(s) (CX_NGR_BASE + (uint32_t)(s) * CX_NGR_BYTES)
#define CX_NGR_GRAD(s) (CX_NGR_BASE + (2u + (uint32_t)(s)) * CX_NGR_BYTES)
#define CX_NGR_H_SEQ (CX_NGR_BASE + 4u * CX_NGR_BYTES)
#define CX_NGR_C_SEQ (CX_NGR_BASE + 4u * CX_NGR_BYTES + 64u)

/* Generation (cortexd): the host knows a step's token only when it issues the STEP, so it writes that step's two rows
 * (orders 2 and 3, 1 KiB) at CX_NGS_ROWS(n), n = steps since RESET, before the STEP command; the card pulls the 1 KiB
 * by DMA into slot n % 64 of its ring buffer (the same 2 x 32 KiB GDDR layout the training chunks use: slot n % 64 is
 * chunk (n / 32) % 2, step n % 32) before the forward. A ring of 64 steps; the host never runs ahead of the ACK. */
#define CX_NGS_STEPS 64u
#define CX_NGS_BYTES 1024u
#define CX_NGS_BASE (CX_NGR_BASE + 5u * CX_NGR_BYTES)
#define CX_NGS_ROWS(n) (CX_NGS_BASE + (uint32_t)((n) % CX_NGS_STEPS) * CX_NGS_BYTES)


/* ---- Live training (card/cortexd.c, host/coach.c): the daemon trains between the clients' bursts of commands. The
 * coach publishes CX_T_MAGIC and feeds chunks of CX_LMAX + 1 token ids (the chunk's inputs and the next token) into
 * a ring of CX_T_SLOTS slots; the card trains chunk c from slot c % CX_T_SLOTS while CX_T_H_CHUNKS > CX_T_C_CHUNKS,
 * reads the knobs once per chunk and reports every CX_T_EVERY chunks. Clients open sessions as always; `cortex --live`
 * skips LOAD and generates from the live weights on the daemon's inference context. A save (flag bit 0) pushes the
 * live weights into CX_STAGE by DMA, CX_STAGE_CKPT bytes in the LOAD layout, then the gates at CX_STAGE_GATES, and the DMA's last line writes CX_T_SAVES
 * after them: the coach copies the stage to a host file once that word has counted up. Host fields and card fields
 * sit on separate 64-byte lines. */
#define CX_T_BASE (55u << 20)
#define CX_T_MAGIC (CX_T_BASE + 0u)         /* u32 "TRNR" while the coach feeds; cleared by the coach to end training (host) */
#define CX_T_H_CHUNKS (CX_T_BASE + 64u)     /* u64 chunks published into the ring (host) */
#define CX_T_LR (CX_T_BASE + 128u)          /* f64 base learning rate (host) */
#define CX_T_SCALE (CX_T_BASE + 192u)       /* f64 learning-rate multiplier, default 1.0 (host) */
#define CX_T_FLAGS (CX_T_BASE + 256u)       /* u64 bit 0 save a checkpoint now, bit 1 stop after this chunk, bit 2 pause (host) */
#define CX_T_EVERY (CX_T_BASE + 320u)       /* u32 report every this many chunks (host) */
#define CX_T_NG (CX_T_BASE + 384u)          /* u32 1: the coach serves the n-gram row ring (CX_NGR_*) for every chunk of the run and the memory trains with it (host) */
#define CX_M_BIAS (CX_T_BASE + 448u)        /* f64 MoE: the home pair's routing bonus, read per chunk and per session command (host: the coach's --moe-bias, a load's) */
#define CX_M_LR (CX_T_BASE + 512u)          /* f64 MoE: the experts' SGD step, read per chunk (host: the coach's --expert-lr) */
#define CX_T_POOL (CX_T_BASE + 576u)        /* u64 MoE pool transfer request (host): CX_T_POOL_WORD(direction, round), 0 = none. Direction 1 pushes the
                                               round's pages of the pool into CX_STAGE, 2 pulls CX_STAGE into them; the daemon answers in CX_T_POOL_ACK
                                               and the host copies the stage to its file (push) or fills it before the request (pull). Works with or
                                               without a run (a pool is pulled before the LOAD that keeps it: arg bit 2) */
#define CX_POOL_PAGE (2u << 20)             /* the pool's huge page; a DMA line stays inside one */
#define CX_POOL_ROUND_PAGES 11u             /* pages per round: 11 x 2 MiB = 23,068,672 B, inside CX_STAGE_CKPT */
#define CX_POOL_ROUND_BYTES (CX_POOL_ROUND_PAGES * CX_POOL_PAGE)
#define CX_T_POOL_WORD(dir, round) (((uint64_t)(dir) << 32) | ((uint64_t)(round) + 1u))
#define CX_T_POOL_DIR(w) ((uint32_t)((w) >> 32))
#define CX_T_POOL_ROUND(w) ((uint32_t)((w) & 0xFFFFFFFFu) - 1u)
#define CX_T_C_CHUNKS (CX_T_BASE + 1024u)   /* u64 chunks trained (card) */
#define CX_T_REPORT (CX_T_BASE + 1088u)     /* u64 reports written (card) */
#define CX_T_LOSS (CX_T_BASE + 1152u)       /* f64 mean training loss of the last report window, nats per token (card) */
#define CX_T_LR_EFF (CX_T_BASE + 1216u)     /* f64 learning rate in effect at the last report (card) */
#define CX_T_TOKPS (CX_T_BASE + 1280u)      /* f64 tokens per second over the last report window, idle time included (card) */
#define CX_T_ACK (CX_T_BASE + 1344u)        /* u64 the flags word as last acted on (card) */
#define CX_T_CKPT (CX_T_BASE + 1408u)       /* u64 the chunk count of the last checkpoint written to /tmp/cortex/live (card) */
#define CX_T_STATE (CX_T_BASE + 1472u)      /* u32 0 off, 1 training, 2 paused, 3 waiting for chunks, 4 stopped (card) */
#define CX_T_SAVES (CX_T_BASE + 1536u)      /* u64 checkpoints pushed into CX_STAGE (the gates after them), written by the DMA after the bytes (card) */
#define CX_T_NG_ACK (CX_T_BASE + 1600u)     /* u32 1 while the daemon trains the memory with the ring (CX_T_NG set and gates loaded), else 0 (card) */
#define CX_T_MOE (CX_T_BASE + 1664u)        /* u32 1 while the loaded model routes through the pool (LOAD arg bit 1), else 0 (card) */
#define CX_T_MOE_DISTINCT (CX_T_BASE + 1728u)   /* u32 MoE: experts updated at least once since the pool was filled, written with each report (card) */
#define CX_T_POOL_ACK (CX_T_BASE + 1792u)   /* u64 the pool request word as completed: for a push written by the DMA after the round's bytes, for a pull after they landed (card) */
#define CX_T_MAGIC_VAL 0x524e5254u          /* "TRNR" little-endian */
#define CX_T_L 32                           /* tokens per chunk (the card's CX_LMAX); a slot holds CX_T_L + 1 ids */
#define CX_T_SLOTS 1024u
#define CX_T_SLOT_BYTES 128u
#define CX_T_RING (56u << 20)
#define CX_T_SLOT(c) (CX_T_RING + (uint32_t)((c) % CX_T_SLOTS) * CX_T_SLOT_BYTES)
/* ---- the partial distillation (docs/DISTIL.md): the teacher's next-token distribution at a student token, mapped
 * onto student ids by tools/soft-align.c: CX_SOFT_K ids (CX_SOFT_NONE = unused) with their probabilities, 48 B a
 * token in the stream's order (a .soft file beside the .u16 stream, one record per token for the token after it).
 * The head's target at the step becomes (1 - sum p_j) onehot(next) + sum_j p_j onehot(id_j): an empty record is the
 * plain one-hot, a record holding lambda times the teacher's distribution the mix. The coach copies a chunk's CX_T_L
 * records into the chunk's slot of the soft ring (CX_T_SOFT_SLOT, the token ring's slot count and order) before
 * publishing the chunk and sets CX_T_SOFT; the daemon DMAs the slot into GDDR before the chunk's forward
 * (card/cluster.h cl_chunk_begin). The reported loss stays the hard cross-entropy. */
#define CX_SOFT_K 8
#define CX_SOFT_NONE 0xFFFFu
typedef struct { uint16_t id[CX_SOFT_K]; float p[CX_SOFT_K]; } cx_soft;
#define CX_SOFT_BYTES 48u
#define CX_SOFT_CHUNK_BYTES (CX_T_L * CX_SOFT_BYTES)   /* 1,536 */
#define CX_T_SOFT (CX_T_BASE + 640u)        /* u32 1: the coach fills the soft ring for every chunk of the run (host) */
#define CX_T_SOFT_RING (57u << 20)          /* CX_T_SLOTS slots of CX_SOFT_CHUNK_BYTES: 1.5 MiB */
#define CX_T_SOFT_SLOT(c) (CX_T_SOFT_RING + (uint32_t)((c) % CX_T_SLOTS) * CX_SOFT_CHUNK_BYTES)
#define CX_T_CTX_US (CX_T_BASE + 1856u)     /* f64 the context memory's mean microseconds a step over the last report window, thread 0 (card) */
#define CX_T_SOFT_N (CX_T_BASE + 1920u)     /* u64 steps of the last report window whose soft record held a distribution (card) */
#define CX_T_WARMUP 50                      /* chunks of learning-rate ramp at the start of a live run (as cluster-gate run) */

#endif
