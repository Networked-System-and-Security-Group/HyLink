/*
 * GPU DMA bandwidth test — all GPUs simultaneously, multi-process mode.
 *
 * Mirrors test_gpu_dma_all but uses the MP memory model:
 *   - Shared hugepages via memfd_create (per NUMA node)
 *   - Each process (one per GPU) calls aclrtHostRegister on ALL NUMA buffers
 *   - Compares with single-process aclrtMallocHost baseline
 *
 * Build:
gcc -O2 -o test_gpu_dma_all_mp test_gpu_dma_all_mp.c \
  -I/usr/local/Ascend/ascend-toolkit/latest/include \
  -L/usr/local/Ascend/ascend-toolkit/latest/lib64 \
  -lascendcl -lpthread -lnuma
 *
 * Run:
 *   ./test_gpu_dma_all_mp [num_gpus]
 */
#define _GNU_SOURCE
#include <acl/acl_rt.h>
#include <numa.h>
#include <numaif.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <sys/wait.h>
#include <unistd.h>

#ifndef MFD_HUGETLB
#define MFD_HUGETLB 0x0004U
#endif
#ifndef MFD_HUGE_2MB
#define MFD_HUGE_2MB (21U << 26)
#endif

#define CHECK(call) do { aclError _e=(call); if(_e){fprintf(stderr,"[GPU %d] ACL err %d @ %s:%d\n",g_gpu,_e,__FILE__,__LINE__);_exit(1);}} while(0)

#define DMA_TOTAL    (512UL * 1024 * 1024)
#define DMA_CHUNK    (1UL * 1024 * 1024)
#define HUGEPAGE_SZ  (2UL * 1024 * 1024)
#define WARMUP       2
#define ITERS        5

/* ── Topology (Ascend 910B 8-GPU) ────────────────────────── */
static const int gpu_numa[]  = {6,6,4,4,0,0,2,2};
static const int numa_list[] = {6,4,0,2};   /* physical NUMA IDs */
#define NUM_NUMA 4

static int g_gpu = -1;

/* ── Chunked DMA copy (matches test_gpu_dma_all) ─────────── */
static void dma_chunked(void *dst, const void *src, size_t total,
                        aclrtMemcpyKind dir, aclrtStream stream) {
    for (size_t off = 0; off < total; off += DMA_CHUNK) {
        size_t sz = DMA_CHUNK;
        if (off + sz > total) sz = total - off;
        aclrtMemcpyAsync((char*)dst + off, sz, (const char*)src + off, sz, dir, stream);
    }
}

static double bw_gbps(size_t bytes, float ms) {
    return (bytes / 1e9) / (ms / 1e3);
}

/* ── Per-GPU result (stored in shared mmap) ──────────────── */
struct gpu_result {
    float d2h_ms;
    float h2d_ms;
};

int main(int argc, char **argv) {
    int num_gpus = argc > 1 ? atoi(argv[1]) : 8;

    /* ── Allocate shared hugepages (per NUMA, like real MP mode) ── */
    size_t hp_sz = (DMA_TOTAL + HUGEPAGE_SZ - 1) & ~(HUGEPAGE_SZ - 1);
    /* Each GPU needs 2 buffers (D2H dst + H2D src), so 2 * DMA_TOTAL per GPU.
     * 2 GPUs per NUMA → 4 * DMA_TOTAL per NUMA node. */
    size_t per_numa_sz = ((size_t)4 * DMA_TOTAL + HUGEPAGE_SZ - 1) & ~(HUGEPAGE_SZ - 1);

    void *hp_bufs[NUM_NUMA];
    int hp_fds[NUM_NUMA];
    for (int n = 0; n < NUM_NUMA; n++) {
        char name[64];
        snprintf(name, sizeof(name), "dma_test_numa%d", n);
        int fd = (int)syscall(SYS_memfd_create, name, MFD_HUGETLB | MFD_HUGE_2MB);
        if (fd < 0) { perror("memfd_create"); exit(1); }
        if (ftruncate(fd, per_numa_sz) < 0) { perror("ftruncate"); exit(1); }
        hp_bufs[n] = mmap(NULL, per_numa_sz, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
        if (hp_bufs[n] == MAP_FAILED) { perror("mmap hugepage"); exit(1); }
        hp_fds[n] = fd;

        unsigned long nm = 1UL << numa_list[n];
        mbind(hp_bufs[n], per_numa_sz, MPOL_BIND, &nm, sizeof(nm)*8, MPOL_MF_MOVE|MPOL_MF_STRICT);

        /* First-touch on the correct NUMA */
        struct bitmask *m = numa_allocate_cpumask();
        numa_node_to_cpus(numa_list[n], m);
        numa_sched_setaffinity(0, m);
        numa_free_cpumask(m);
        memset(hp_bufs[n], 0, per_numa_sz);
    }

    /* Shared result area + barrier */
    struct gpu_result *results = mmap(NULL, sizeof(struct gpu_result) * num_gpus * 2,
        PROT_READ|PROT_WRITE, MAP_SHARED|MAP_ANONYMOUS, -1, 0);
    /* results[0..num_gpus-1] = MP results, results[num_gpus..2*num_gpus-1] = SP results */
    int *barrier = mmap(NULL, 4096,
        PROT_READ|PROT_WRITE, MAP_SHARED|MAP_ANONYMOUS, -1, 0);

    printf("=== GPU DMA All — MP vs SP (%zu MB, %zu KB chunks) ===\n\n",
           DMA_TOTAL >> 20, DMA_CHUNK >> 10);

    /* ════════════════════════════════════════════════════════════
     * Test 1: Multi-Process with shared hugepages + HostRegister
     *         (mirrors real MP code path)
     * ════════════════════════════════════════════════════════════ */
    memset(barrier, 0, 4096);
    memset(results, 0, sizeof(struct gpu_result) * num_gpus * 2);

    for (int gpu = 0; gpu < num_gpus; gpu++) {
        pid_t pid = fork();
        if (pid < 0) { perror("fork"); exit(1); }
        if (pid > 0) continue;

        g_gpu = gpu;
        int my_numa = gpu_numa[gpu];

        /* Pin to GPU's NUMA */
        struct bitmask *m = numa_allocate_cpumask();
        numa_node_to_cpus(my_numa, m);
        numa_sched_setaffinity(0, m);
        numa_free_cpumask(m);

        aclFinalize();
        CHECK(aclInit(NULL));
        CHECK(aclrtSetDevice(gpu));

        /* Register ALL NUMA buffers (like real MP mode) */
        for (int n = 0; n < NUM_NUMA; n++) {
            void *dp;
            CHECK(aclrtHostRegister(hp_bufs[n], per_numa_sz, ACL_HOST_REGISTER_MAPPED, &dp));
        }

        /* Find my logical NUMA index and my slot within that NUMA */
        int my_logical = -1;
        for (int n = 0; n < NUM_NUMA; n++)
            if (numa_list[n] == my_numa) { my_logical = n; break; }
        int rank_in_numa = (gpu / 2 == (gpu ^ 1) / 2) ? (gpu & 1) : 0;

        /* My host buffers: buf_d2h for D2H dest, buf_h2d for H2D source */
        void *buf_d2h = (char*)hp_bufs[my_logical] + (size_t)rank_in_numa * 2 * DMA_TOTAL;
        void *buf_h2d = (char*)hp_bufs[my_logical] + (size_t)rank_in_numa * 2 * DMA_TOTAL + DMA_TOTAL;
        memset(buf_h2d, 0x42, DMA_TOTAL);

        void *dev_d2h, *dev_h2d;
        CHECK(aclrtMalloc(&dev_d2h, DMA_TOTAL, ACL_MEM_MALLOC_HUGE_FIRST));
        CHECK(aclrtMalloc(&dev_h2d, DMA_TOTAL, ACL_MEM_MALLOC_HUGE_FIRST));
        aclrtStream s_d2h, s_h2d;
        CHECK(aclrtCreateStream(&s_d2h));
        CHECK(aclrtCreateStream(&s_h2d));

        /* Warmup */
        for (int i = 0; i < WARMUP; i++) {
            dma_chunked(buf_d2h, dev_d2h, DMA_TOTAL, ACL_MEMCPY_DEVICE_TO_HOST, s_d2h);
            CHECK(aclrtSynchronizeStream(s_d2h));
            dma_chunked(dev_h2d, buf_h2d, DMA_TOTAL, ACL_MEMCPY_HOST_TO_DEVICE, s_h2d);
            CHECK(aclrtSynchronizeStream(s_h2d));
        }

        /* Barrier: all GPUs start together */
        __atomic_add_fetch(barrier, 1, __ATOMIC_SEQ_CST);
        while (__atomic_load_n(barrier, __ATOMIC_SEQ_CST) < num_gpus) ;

        /* Measured iterations */
        float best_d2h = 1e9, best_h2d = 1e9;
        for (int it = 0; it < ITERS; it++) {
            aclrtEvent e_start_d2h, e_end_d2h, e_start_h2d, e_end_h2d;
            CHECK(aclrtCreateEvent(&e_start_d2h)); CHECK(aclrtCreateEvent(&e_end_d2h));
            CHECK(aclrtCreateEvent(&e_start_h2d)); CHECK(aclrtCreateEvent(&e_end_h2d));

            CHECK(aclrtRecordEvent(e_start_d2h, s_d2h));
            CHECK(aclrtRecordEvent(e_start_h2d, s_h2d));

            dma_chunked(buf_d2h, dev_d2h, DMA_TOTAL, ACL_MEMCPY_DEVICE_TO_HOST, s_d2h);
            dma_chunked(dev_h2d, buf_h2d, DMA_TOTAL, ACL_MEMCPY_HOST_TO_DEVICE, s_h2d);

            CHECK(aclrtRecordEvent(e_end_d2h, s_d2h));
            CHECK(aclrtRecordEvent(e_end_h2d, s_h2d));
            CHECK(aclrtSynchronizeStream(s_d2h));
            CHECK(aclrtSynchronizeStream(s_h2d));

            float d2h_ms = 0, h2d_ms = 0;
            CHECK(aclrtEventElapsedTime(&d2h_ms, e_start_d2h, e_end_d2h));
            CHECK(aclrtEventElapsedTime(&h2d_ms, e_start_h2d, e_end_h2d));
            if (d2h_ms < best_d2h) best_d2h = d2h_ms;
            if (h2d_ms < best_h2d) best_h2d = h2d_ms;

            aclrtDestroyEvent(e_start_d2h); aclrtDestroyEvent(e_end_d2h);
            aclrtDestroyEvent(e_start_h2d); aclrtDestroyEvent(e_end_h2d);
        }

        results[gpu].d2h_ms = best_d2h;
        results[gpu].h2d_ms = best_h2d;

        aclrtFree(dev_d2h); aclrtFree(dev_h2d);
        aclrtDestroyStream(s_d2h); aclrtDestroyStream(s_h2d);
        for (int n = 0; n < NUM_NUMA; n++) aclrtHostUnregister(hp_bufs[n]);
        _exit(0);
    }
    for (int i = 0; i < num_gpus; i++) wait(NULL);

    /* ════════════════════════════════════════════════════════════
     * Test 2: Multi-Process with private aclrtMallocHost (SP-like)
     *         (baseline: each process allocates its own pinned mem)
     * ════════════════════════════════════════════════════════════ */
    memset(barrier, 0, 4096);

    for (int gpu = 0; gpu < num_gpus; gpu++) {
        pid_t pid = fork();
        if (pid < 0) { perror("fork"); exit(1); }
        if (pid > 0) continue;

        g_gpu = gpu;
        int my_numa = gpu_numa[gpu];

        struct bitmask *m = numa_allocate_cpumask();
        numa_node_to_cpus(my_numa, m);
        numa_sched_setaffinity(0, m);
        numa_free_cpumask(m);

        aclInit(NULL);
        CHECK(aclrtSetDevice(gpu));

        void *buf_d2h, *buf_h2d;
        CHECK(aclrtMallocHost(&buf_d2h, DMA_TOTAL));
        memset(buf_d2h, 0, DMA_TOTAL);
        CHECK(aclrtMallocHost(&buf_h2d, DMA_TOTAL));
        memset(buf_h2d, 0x42, DMA_TOTAL);

        void *dev_d2h, *dev_h2d;
        CHECK(aclrtMalloc(&dev_d2h, DMA_TOTAL, ACL_MEM_MALLOC_HUGE_FIRST));
        CHECK(aclrtMalloc(&dev_h2d, DMA_TOTAL, ACL_MEM_MALLOC_HUGE_FIRST));
        aclrtStream s_d2h, s_h2d;
        CHECK(aclrtCreateStream(&s_d2h));
        CHECK(aclrtCreateStream(&s_h2d));

        for (int i = 0; i < WARMUP; i++) {
            dma_chunked(buf_d2h, dev_d2h, DMA_TOTAL, ACL_MEMCPY_DEVICE_TO_HOST, s_d2h);
            CHECK(aclrtSynchronizeStream(s_d2h));
            dma_chunked(dev_h2d, buf_h2d, DMA_TOTAL, ACL_MEMCPY_HOST_TO_DEVICE, s_h2d);
            CHECK(aclrtSynchronizeStream(s_h2d));
        }

        __atomic_add_fetch(barrier, 1, __ATOMIC_SEQ_CST);
        while (__atomic_load_n(barrier, __ATOMIC_SEQ_CST) < num_gpus) ;

        float best_d2h = 1e9, best_h2d = 1e9;
        for (int it = 0; it < ITERS; it++) {
            aclrtEvent e_start_d2h, e_end_d2h, e_start_h2d, e_end_h2d;
            CHECK(aclrtCreateEvent(&e_start_d2h)); CHECK(aclrtCreateEvent(&e_end_d2h));
            CHECK(aclrtCreateEvent(&e_start_h2d)); CHECK(aclrtCreateEvent(&e_end_h2d));

            CHECK(aclrtRecordEvent(e_start_d2h, s_d2h));
            CHECK(aclrtRecordEvent(e_start_h2d, s_h2d));

            dma_chunked(buf_d2h, dev_d2h, DMA_TOTAL, ACL_MEMCPY_DEVICE_TO_HOST, s_d2h);
            dma_chunked(dev_h2d, buf_h2d, DMA_TOTAL, ACL_MEMCPY_HOST_TO_DEVICE, s_h2d);

            CHECK(aclrtRecordEvent(e_end_d2h, s_d2h));
            CHECK(aclrtRecordEvent(e_end_h2d, s_h2d));
            CHECK(aclrtSynchronizeStream(s_d2h));
            CHECK(aclrtSynchronizeStream(s_h2d));

            float d2h_ms = 0, h2d_ms = 0;
            CHECK(aclrtEventElapsedTime(&d2h_ms, e_start_d2h, e_end_d2h));
            CHECK(aclrtEventElapsedTime(&h2d_ms, e_start_h2d, e_end_h2d));
            if (d2h_ms < best_d2h) best_d2h = d2h_ms;
            if (h2d_ms < best_h2d) best_h2d = h2d_ms;

            aclrtDestroyEvent(e_start_d2h); aclrtDestroyEvent(e_end_d2h);
            aclrtDestroyEvent(e_start_h2d); aclrtDestroyEvent(e_end_h2d);
        }

        results[num_gpus + gpu].d2h_ms = best_d2h;
        results[num_gpus + gpu].h2d_ms = best_h2d;

        aclrtFree(dev_d2h); aclrtFree(dev_h2d);
        aclrtDestroyStream(s_d2h); aclrtDestroyStream(s_h2d);
        aclrtFreeHost(buf_d2h); aclrtFreeHost(buf_h2d);
        _exit(0);
    }
    for (int i = 0; i < num_gpus; i++) wait(NULL);

    /* ── Report ───────────────────────────────────────────────── */
    printf("%-5s  %-5s  │ %22s  │ %22s  │ %8s\n",
           "", "", "  MP (HostRegister+HP)", "SP (MallocHost priv)", "");
    printf("%-5s  %-5s  │ %10s %10s  │ %10s %10s  │ %8s\n",
           "GPU", "NUMA", "D2H GB/s", "H2D GB/s", "D2H GB/s", "H2D GB/s", "D2H gap");
    printf("%-5s  %-5s──┼─%10s─%10s──┼─%10s─%10s──┼─%8s\n",
           "-----", "-----", "----------", "----------",
           "----------", "----------", "--------");

    double mp_d2h_tot = 0, mp_h2d_tot = 0, sp_d2h_tot = 0, sp_h2d_tot = 0;
    for (int gpu = 0; gpu < num_gpus; gpu++) {
        int pnuma = gpu_numa[gpu];
        double mp_d2h = bw_gbps(DMA_TOTAL, results[gpu].d2h_ms);
        double mp_h2d = bw_gbps(DMA_TOTAL, results[gpu].h2d_ms);
        double sp_d2h = bw_gbps(DMA_TOTAL, results[num_gpus + gpu].d2h_ms);
        double sp_h2d = bw_gbps(DMA_TOTAL, results[num_gpus + gpu].h2d_ms);

        double gap = (sp_d2h - mp_d2h) / sp_d2h * 100.0;

        mp_d2h_tot += mp_d2h; mp_h2d_tot += mp_h2d;
        sp_d2h_tot += sp_d2h; sp_h2d_tot += sp_h2d;

        printf("%-5d  %-5d  │ %10.2f %10.2f  │ %10.2f %10.2f  │ %+7.1f%%\n",
               gpu, pnuma, mp_d2h, mp_h2d, sp_d2h, sp_h2d, gap);
    }

    double gap_tot = (sp_d2h_tot - mp_d2h_tot) / sp_d2h_tot * 100.0;
    printf("%-5s  %-5s──┼─%10s─%10s──┼─%10s─%10s──┼─%8s\n",
           "-----", "-----", "----------", "----------",
           "----------", "----------", "--------");
    printf("%-5s  %-5s  │ %10.2f %10.2f  │ %10.2f %10.2f  │ %+7.1f%%\n",
           "Total", "-", mp_d2h_tot, mp_h2d_tot, sp_d2h_tot, sp_h2d_tot, gap_tot);

    /* Cleanup shared hugepages */
    for (int n = 0; n < NUM_NUMA; n++) {
        munmap(hp_bufs[n], per_numa_sz);
        close(hp_fds[n]);
    }

    printf("\nDone.\n");
    return 0;
}
