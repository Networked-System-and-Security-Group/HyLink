/*
 * Reproduction: aclrtHostRegister DMA degradation under concurrent CPU memcpy.
 * Sweep across different DMA slice sizes to see if larger slices reduce degradation.
 *
 * Build:
gcc -O2 -o bench_mp_hostregister bench_mp_hostregister.c \
  -I/usr/local/Ascend/ascend-toolkit/latest/include -L/usr/local/Ascend/ascend-toolkit/latest/lib64 \
  -lascendcl -lpthread -lnuma
 *
 * Run:
 *   ./bench_mp_hostregister [nprocs] [nworkers_per_proc]
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
#include <time.h>
#include <unistd.h>

#ifndef MFD_HUGETLB
#define MFD_HUGETLB 0x0004U
#endif
#ifndef MFD_HUGE_2MB
#define MFD_HUGE_2MB (21U << 26)
#endif

#define CHECK(call) do { aclError _e=(call); if(_e){fprintf(stderr,"[rank %d] ACL err %d @ %s:%d\n",g_rank,_e,__FILE__,__LINE__);_exit(1);}} while(0)

static inline uint64_t now_ns(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

/* ── Topology (Ascend 910B 8-GPU) ────────────────────────── */
static const int gpu_numa[] = {6,6,4,4,0,0,2,2};
static const int sibling[]  = {7,7,5,5,1,1,3,3};
static const int numa_for_logical[] = {6,4,0,2};

#define DMA_TOTAL  (64UL * 1024 * 1024)
#define BUF_PER_NUMA (512UL * 1024 * 1024)
#define NUM_NUMA   4
#define WARMUP     3
#define ITERS      10

/* Result slots: 0=D2H only, 1=D2H+mc, 2=H2D only, 3=H2D+mc */
#define SLOTS      4

static int g_rank = -1;

struct result { double avg_ms; };

/* ── CPU memcpy thread ───────────────────────────────────── */
static volatile int g_run;
static char *g_src;
static char *g_dst;
static size_t g_size;
static size_t g_slice;  /* runtime slice size for memcpy chunks */

static void *memcpy_worker(void *arg) {
    int pin_numa = *(int *)arg;
    struct bitmask *m = numa_allocate_cpumask();
    numa_node_to_cpus(pin_numa, m);
    numa_sched_setaffinity(0, m);
    numa_free_cpumask(m);
    while (!g_run) ;
    while (g_run) {
        size_t sl = g_slice;
        for (size_t o = 0; o < g_size && g_run; o += sl)
            memcpy(g_dst + o, g_src + o, sl);
    }
    return NULL;
}

/* ── DMA test (event-based timing, runtime slice size) ──── */
static double run_dma(void *host, void *dev, aclrtStream stream,
                      aclrtMemcpyKind dir, size_t slice) {
    size_t nslice = DMA_TOTAL / slice;
    aclrtEvent ev_start, ev_end;
    CHECK(aclrtCreateEvent(&ev_start));
    CHECK(aclrtCreateEvent(&ev_end));

    double times[ITERS];
    for (int it = 0; it < WARMUP + ITERS; it++) {
        aclrtSynchronizeStream(stream);
        CHECK(aclrtRecordEvent(ev_start, stream));
        for (size_t s = 0; s < nslice; s++) {
            if (dir == ACL_MEMCPY_DEVICE_TO_HOST) {
                aclrtMemcpyAsync((char *)host + s * slice, slice,
                                 (char *)dev + s * slice, slice,
                                 ACL_MEMCPY_DEVICE_TO_HOST, stream);
            } else {
                aclrtMemcpyAsync((char *)dev + s * slice, slice,
                                 (char *)host + s * slice, slice,
                                 ACL_MEMCPY_HOST_TO_DEVICE, stream);
            }
        }
        CHECK(aclrtRecordEvent(ev_end, stream));
        aclrtSynchronizeStream(stream);
        float elapsed_ms = 0;
        CHECK(aclrtEventElapsedTime(&elapsed_ms, ev_start, ev_end));
        if (it >= WARMUP) times[it - WARMUP] = elapsed_ms;
    }
    CHECK(aclrtDestroyEvent(ev_start));
    CHECK(aclrtDestroyEvent(ev_end));

    /* trimmed mean */
    for (int i = 0; i < ITERS-1; i++)
        for (int j = i+1; j < ITERS; j++)
            if (times[j] < times[i]) { double t=times[i]; times[i]=times[j]; times[j]=t; }
    double sum = 0;
    for (int i = 1; i < ITERS-1; i++) sum += times[i];
    return sum / (ITERS - 2);
}

int main(int argc, char **argv) {
    int nprocs = argc > 1 ? atoi(argv[1]) : 8;
    int nworkers = argc > 2 ? atoi(argv[2]) : 14;

    /* Slice sizes to sweep */
    size_t slices[] = {
        256UL * 1024,       /* 256KB */
        512UL * 1024,       /* 512KB */
        1024UL * 1024,      /* 1MB   */
        2UL * 1024 * 1024,  /* 2MB   */
        4UL * 1024 * 1024,  /* 4MB   */
        8UL * 1024 * 1024,  /* 8MB   */
        16UL * 1024 * 1024, /* 16MB  */
        32UL * 1024 * 1024, /* 32MB  */
        64UL * 1024 * 1024, /* 64MB (single DMA call) */
    };
    int nslices = sizeof(slices) / sizeof(slices[0]);

    printf("HostRegister DMA vs slice size (D2H + H2D)\n");
    printf("  %d processes, %d CPU memcpy threads each\n", nprocs, nworkers);
    printf("  Total DMA: %zuMB per rank\n\n", DMA_TOTAL >> 20);

    /* ── Allocate shared hugepage buffers ──────────────────── */
    void *hp_bufs[NUM_NUMA];
    size_t hp_sz = (BUF_PER_NUMA + 2*1024*1024-1) & ~(2UL*1024*1024-1);

    for (int n = 0; n < NUM_NUMA; n++) {
        char name[64];
        snprintf(name, sizeof(name), "hp_numa%d", n);
        int fd = (int)syscall(SYS_memfd_create, name, MFD_HUGETLB | MFD_HUGE_2MB);
        if (fd < 0) { perror("memfd_create"); exit(1); }
        if (ftruncate(fd, hp_sz) < 0) { perror("ftruncate"); exit(1); }
        hp_bufs[n] = mmap(NULL, hp_sz, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
        if (hp_bufs[n] == MAP_FAILED) { perror("mmap"); exit(1); }
        close(fd);
        unsigned long nm = 1UL << numa_for_logical[n];
        mbind(hp_bufs[n], hp_sz, MPOL_BIND, &nm, sizeof(nm)*8, MPOL_MF_MOVE|MPOL_MF_STRICT);
        struct bitmask *m = numa_allocate_cpumask();
        numa_node_to_cpus(numa_for_logical[n], m);
        numa_sched_setaffinity(0, m);
        numa_free_cpumask(m);
        memset(hp_bufs[n], 0, hp_sz);
    }

    struct result *results = (struct result *)mmap(NULL,
        sizeof(struct result) * nprocs * SLOTS,
        PROT_READ|PROT_WRITE, MAP_SHARED|MAP_ANONYMOUS, -1, 0);
    int *barrier = (int *)mmap(NULL, 4096,
        PROT_READ|PROT_WRITE, MAP_SHARED|MAP_ANONYMOUS, -1, 0);

    /* ── Table header ──────────────────────────────────────── */
    printf("%-8s │ %-53s │ %-53s\n", "", "D2H (max across ranks)", "H2D (max across ranks)");
    printf("%-8s │ %10s %10s %10s %10s %6s │ %10s %10s %10s %10s %6s\n",
           "Slice", "Register", "Reg+memcpy", "MallocHost", "Mal+memcpy", "Gap",
                    "Register", "Reg+memcpy", "MallocHost", "Mal+memcpy", "Gap");
    printf("─────────┼───────────────────────────────────────────────────────┼───────────────────────────────────────────────────────\n");

    for (int si = 0; si < nslices; si++) {
        size_t slice = slices[si];
        size_t nsl = DMA_TOTAL / slice;
        if (nsl == 0) continue;

        double reg_d2h = 0, reg_d2h_mc = 0, reg_h2d = 0, reg_h2d_mc = 0;
        double mal_d2h = 0, mal_d2h_mc = 0, mal_h2d = 0, mal_h2d_mc = 0;

        /* ── HostRegister shared hugepage ──────────────────── */
        memset(barrier, 0, 4096);
        for (int rank = 0; rank < nprocs; rank++) {
            pid_t pid = fork();
            if (pid < 0) { perror("fork"); exit(1); }
            if (pid > 0) continue;

            g_rank = rank;
            int gpu = rank;
            int my_numa = gpu_numa[gpu];
            int my_logical = -1;
            for (int n = 0; n < NUM_NUMA; n++)
                if (numa_for_logical[n] == my_numa) { my_logical = n; break; }

            aclFinalize();
            CHECK(aclInit(NULL));
            CHECK(aclrtSetDevice(gpu));

            for (int n = 0; n < NUM_NUMA; n++) {
                void *dp;
                CHECK(aclrtHostRegister(hp_bufs[n], hp_sz, ACL_HOST_REGISTER_MAPPED, &dp));
            }

            void *dev;
            CHECK(aclrtMalloc(&dev, DMA_TOTAL, ACL_MEM_MALLOC_HUGE_FIRST));
            aclrtStream stream;
            CHECK(aclrtCreateStream(&stream));

            int rank_in_numa = rank % 2;
            void *my_host = (char *)hp_bufs[my_logical] + rank_in_numa * DMA_TOTAL;

            pthread_t threads[256];
            int other_logical = (my_logical + 1) % NUM_NUMA;
            g_src = (char *)hp_bufs[other_logical];
            g_dst = (char *)hp_bufs[my_logical] + 2*DMA_TOTAL;
            g_size = DMA_TOTAL;
            g_slice = slice;
            g_run = 0;
            int wn = sibling[gpu];
            for (int w = 0; w < nworkers; w++)
                pthread_create(&threads[w], NULL, memcpy_worker, &wn);

            __atomic_add_fetch(barrier, 1, __ATOMIC_SEQ_CST);
            while (__atomic_load_n(barrier, __ATOMIC_SEQ_CST) < nprocs) ;

            results[0*nprocs+rank].avg_ms = run_dma(my_host, dev, stream, ACL_MEMCPY_DEVICE_TO_HOST, slice);
            results[2*nprocs+rank].avg_ms = run_dma(my_host, dev, stream, ACL_MEMCPY_HOST_TO_DEVICE, slice);

            __atomic_add_fetch(barrier+1, 1, __ATOMIC_SEQ_CST);
            while (__atomic_load_n(barrier+1, __ATOMIC_SEQ_CST) < nprocs) ;

            g_run = 1; usleep(10000);
            results[1*nprocs+rank].avg_ms = run_dma(my_host, dev, stream, ACL_MEMCPY_DEVICE_TO_HOST, slice);
            results[3*nprocs+rank].avg_ms = run_dma(my_host, dev, stream, ACL_MEMCPY_HOST_TO_DEVICE, slice);
            g_run = 0;
            for (int w = 0; w < nworkers; w++) pthread_join(threads[w], NULL);

            aclrtFree(dev); aclrtDestroyStream(stream);
            for (int n = 0; n < NUM_NUMA; n++) aclrtHostUnregister(hp_bufs[n]);
            _exit(0);
        }
        for (int i = 0; i < nprocs; i++) wait(NULL);

        for (int i = 0; i < nprocs; i++) {
            if (results[0*nprocs+i].avg_ms > reg_d2h) reg_d2h = results[0*nprocs+i].avg_ms;
            if (results[1*nprocs+i].avg_ms > reg_d2h_mc) reg_d2h_mc = results[1*nprocs+i].avg_ms;
            if (results[2*nprocs+i].avg_ms > reg_h2d) reg_h2d = results[2*nprocs+i].avg_ms;
            if (results[3*nprocs+i].avg_ms > reg_h2d_mc) reg_h2d_mc = results[3*nprocs+i].avg_ms;
        }

        /* ── MallocHost ────────────────────────────────────── */
        memset(barrier, 0, 4096);
        for (int rank = 0; rank < nprocs; rank++) {
            pid_t pid = fork();
            if (pid < 0) { perror("fork"); exit(1); }
            if (pid > 0) continue;

            g_rank = rank;
            int gpu = rank;

            aclFinalize();
            CHECK(aclInit(NULL));
            CHECK(aclrtSetDevice(gpu));

            struct bitmask *m = numa_allocate_cpumask();
            numa_node_to_cpus(gpu_numa[gpu], m);
            numa_sched_setaffinity(0, m);
            numa_free_cpumask(m);

            void *h_ml;
            CHECK(aclrtMallocHost(&h_ml, DMA_TOTAL));
            memset(h_ml, 0, DMA_TOTAL);

            void *dev;
            CHECK(aclrtMalloc(&dev, DMA_TOTAL, ACL_MEM_MALLOC_HUGE_FIRST));
            aclrtStream stream;
            CHECK(aclrtCreateStream(&stream));

            pthread_t threads[256];
            void *lsrc = malloc(DMA_TOTAL); memset(lsrc, 0xAB, DMA_TOTAL);
            void *ldst = malloc(DMA_TOTAL); memset(ldst, 0, DMA_TOTAL);
            g_src = (char *)lsrc; g_dst = (char *)ldst; g_size = DMA_TOTAL;
            g_slice = slice;
            g_run = 0;
            int wn = sibling[gpu];
            for (int w = 0; w < nworkers; w++)
                pthread_create(&threads[w], NULL, memcpy_worker, &wn);

            __atomic_add_fetch(barrier, 1, __ATOMIC_SEQ_CST);
            while (__atomic_load_n(barrier, __ATOMIC_SEQ_CST) < nprocs) ;

            results[0*nprocs+rank].avg_ms = run_dma(h_ml, dev, stream, ACL_MEMCPY_DEVICE_TO_HOST, slice);
            results[2*nprocs+rank].avg_ms = run_dma(h_ml, dev, stream, ACL_MEMCPY_HOST_TO_DEVICE, slice);

            __atomic_add_fetch(barrier+1, 1, __ATOMIC_SEQ_CST);
            while (__atomic_load_n(barrier+1, __ATOMIC_SEQ_CST) < nprocs) ;

            g_run = 1; usleep(10000);
            results[1*nprocs+rank].avg_ms = run_dma(h_ml, dev, stream, ACL_MEMCPY_DEVICE_TO_HOST, slice);
            results[3*nprocs+rank].avg_ms = run_dma(h_ml, dev, stream, ACL_MEMCPY_HOST_TO_DEVICE, slice);
            g_run = 0;
            for (int w = 0; w < nworkers; w++) pthread_join(threads[w], NULL);

            free(lsrc); free(ldst);
            aclrtFree(dev); aclrtDestroyStream(stream); aclrtFreeHost(h_ml);
            _exit(0);
        }
        for (int i = 0; i < nprocs; i++) wait(NULL);

        for (int i = 0; i < nprocs; i++) {
            if (results[0*nprocs+i].avg_ms > mal_d2h) mal_d2h = results[0*nprocs+i].avg_ms;
            if (results[1*nprocs+i].avg_ms > mal_d2h_mc) mal_d2h_mc = results[1*nprocs+i].avg_ms;
            if (results[2*nprocs+i].avg_ms > mal_h2d) mal_h2d = results[2*nprocs+i].avg_ms;
            if (results[3*nprocs+i].avg_ms > mal_h2d_mc) mal_h2d_mc = results[3*nprocs+i].avg_ms;
        }

        /* ── Print row ─────────────────────────────────────── */
        char slabel[16];
        if (slice >= 1024*1024)
            snprintf(slabel, sizeof(slabel), "%zuMB", slice >> 20);
        else
            snprintf(slabel, sizeof(slabel), "%zuKB", slice >> 10);

        double d2h_gap = (reg_d2h_mc - mal_d2h_mc) / mal_d2h_mc * 100;
        double h2d_gap = (reg_h2d_mc - mal_h2d_mc) / mal_h2d_mc * 100;

        printf("%-8s │ %10.2f %10.2f %10.2f %10.2f %+6.1f%% │ %10.2f %10.2f %10.2f %10.2f %+6.1f%%\n",
               slabel,
               reg_d2h, reg_d2h_mc, mal_d2h, mal_d2h_mc, d2h_gap,
               reg_h2d, reg_h2d_mc, mal_h2d, mal_h2d_mc, h2d_gap);
        fflush(stdout);
    }

    printf("\n");
    return 0;
}
