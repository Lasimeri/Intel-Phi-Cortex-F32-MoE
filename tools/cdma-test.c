/*
 * cdma-test: a standalone probe of the card-owned DMA path (card/cdma.c) against the host-memory part.
 *   cdma-test [CHANNEL...]       default channels 2 3 5 6 7
 * For each channel: open it, DMA 64 KiB from the part's CX_DATA area into a card huge page (host -> card), check the
 * data against what the host wrote there (the host fills CX_DATA with 0x5A5A... + the byte index), then DMA it back into
 * the part at CX_DATA + 64 KiB (card -> host) with the status write at CX_C_HEART; prints what completed and how long.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "../card/cdma.h"
#include "../lib/cx-proto.h"

#define BYTES 65536u

static double now_s(void)
{
	struct timespec t;
	clock_gettime(CLOCK_MONOTONIC, &t);
	return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}

static int probe(int chan)
{
	cdma d;
	if (cdma_open(&d, chan)) { printf("channel %d: open failed\n", chan); return -1; }
	uint64_t phys;
	uint8_t *buf = huge_alloc(2u << 20, &phys, 1);
	if (!buf) { printf("channel %d: no huge page\n", chan); cdma_close(&d); return -1; }
	memset(buf, 0, BYTES);
	const uint64_t host = d.region_addr + ((uint64_t)CX_PART_MIB << 20) + CX_DATA;
	double t0 = now_s();
	uint64_t seq = cdma_submit(&d, host, phys, BYTES, 0, 0);
	const int in_ok = seq && !cdma_wait(&d, seq, 3.0);
	const double t_in = now_s() - t0;
	int match = 0;
	if (in_ok) { match = 1; for (uint32_t i = 0; i < BYTES; i++) if (buf[i] != (uint8_t)(0x5A + i)) { match = 0; break; } }
	t0 = now_s();
	seq = cdma_submit(&d, phys, host + BYTES, BYTES, d.region_addr + ((uint64_t)CX_PART_MIB << 20) + CX_C_HEART, 0x0DA0ull | (uint64_t)chan);
	const int out_ok = seq && !cdma_wait(&d, seq, 3.0);
	const double t_out = now_s() - t0;
	printf("channel %d: host->card %s in %.1f us (data %s), card->host %s in %.1f us\n", chan, in_ok ? "done" : "TIMEOUT", t_in * 1e6,
	       match ? "match" : "MISMATCH", out_ok ? "done" : "TIMEOUT", t_out * 1e6);
	fflush(stdout);
	cdma_close(&d);
	return in_ok && match && out_ok ? 0 : -1;
}

int main(int argc, char **argv)
{
	const int def[5] = { 2, 3, 5, 6, 7 };
	int ok = 0, n = 0;
	if (argc > 1) for (int i = 1; i < argc; i++) { n++; ok += probe(atoi(argv[i])) == 0; }
	else for (int i = 0; i < 5; i++) { n++; ok += probe(def[i]) == 0; }
	printf("cdma-test: %d of %d channels complete both directions\n", ok, n);
	return ok == n ? 0 : 1;
}
