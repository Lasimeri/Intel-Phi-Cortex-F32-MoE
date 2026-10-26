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
#define CX_OP_LOAD 1               /* arg bit 0: the n-gram gates (2 x 128 F32) follow the checkpoint in the stage and the memory is on */
#define CX_OP_RESET 2              /* every column's fast weights and the bus to zero; the step count of the memory ring restarts */
#define CX_OP_STEP 3               /* arg: the token id; the card answers with the 4096 logits at CX_LOGITS, then the ACK */
#define CX_OP_BUS 4                /* arg: 1 bus on, 0 off (must match the checkpoint's training) */
#define CX_OP_EXIT 5               /* the daemon ends this session and exits (frees the control socket) */
#define CX_H_CMD_OP 72             /* u32 the op code of the current command (written before CX_H_CMD) */
#define CX_LOGITS (128u << 10)     /* 4096 F32, written by the card's DMA; CX_C_ACK is the DMA status word */
#define CX_STAGE (1u << 20)        /* the staged checkpoint, 23,463,424 B, host-written before CX_OP_LOAD */
#define CX_STAGE_CKPT ((120u * 44512u + 128u + 4096u * 128u) * 4u)   /* its length: 120 columns, cg, the embedding in blocks */
#define CX_STAGE_FLOATS (CX_STAGE_CKPT / 4u)
#define CX_STAGE_GATES (CX_STAGE + CX_STAGE_CKPT)                    /* the n-gram gates, 1,024 B, when LOAD's arg bit 0 is set */

/* ---- MoE stage 1: the n-gram row ring (PLAN-MOE.md). Per chunk of 32 steps the host gathers 64 rows (2 orders x 32
 * steps x 128 F32 = 32 KiB) into row slot chunk % 2 and publishes CX_NGR_H_SEQ = chunks published; the card DMAs the
 * slot into GDDR at the chunk start and, at the chunk end, DMAs the 64 row gradients into gradient slot chunk % 2 with
 * CX_NGR_C_SEQ as the DMA status word (= chunks whose gradients have landed). The host applies the gradients to its
 * tables and gathers chunk + 2 into the slot the card has finished with. */
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

#endif
