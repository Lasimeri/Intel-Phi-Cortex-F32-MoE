/*
 * partpoll: probes the part through the mapping cortexd uses (/dev/phihost, O_SYNC, mmap at CX_PART_MIB).
 *   partpoll SECS            prints the host session word every 500 ms (mmap read and pread side by side)
 *   partpoll write HEXVALUE  writes the value into the card session word through the mapping and exits
 */
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>
#include "../lib/cx-proto.h"

int main(int argc, char **argv)
{
	const int fd = open("/dev/phihost", O_RDWR | O_SYNC);
	if (fd < 0) { perror("/dev/phihost"); return 1; }
	volatile uint8_t *part = mmap(NULL, 1u << 20, PROT_READ | PROT_WRITE, MAP_SHARED, fd, (off_t)CX_PART_MIB << 20);
	if (part == MAP_FAILED) { perror("mmap"); return 1; }
	if (argc > 2 && !strcmp(argv[1], "write")) {
		const uint64_t v = strtoull(argv[2], NULL, 16);
		*(volatile uint64_t *)(part + CX_C_SESSION) = v;
		__asm__ volatile("lock addq $0, (%%rsp)" ::: "memory");
		printf("card wrote %016llx into the card session word through the mapping; reads back %016llx\n", (unsigned long long)v, (unsigned long long)*(volatile uint64_t *)(part + CX_C_SESSION));
		return 0;
	}
	const int secs = argc > 1 ? atoi(argv[1]) : 5;
	for (int i = 0; i < secs * 2; i++) {
		const uint64_t hs = __atomic_load_n((volatile uint64_t *)(part + CX_H_SESSION), __ATOMIC_ACQUIRE);
		uint64_t pr = 0;
		if (pread(fd, &pr, 8, ((off_t)CX_PART_MIB << 20) + CX_H_SESSION) != 8) pr = 0;
		printf("%4d ms: mmap hsess %016llx magic %08x | pread hsess %016llx\n", i * 500, (unsigned long long)hs, *(volatile uint32_t *)(part + CX_H_MAGIC), (unsigned long long)pr);
		fflush(stdout);
		usleep(500000);
	}
	return 0;
}
