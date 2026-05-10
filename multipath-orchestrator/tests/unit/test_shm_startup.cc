#include "core/shm_store.h"
#include "core/comm_init.h"

#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <poll.h>
#include <signal.h>
#include <sstream>
#include <thread>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

namespace {
bool pause_initialization = false;
int initialization_entered = -1;
int initialization_release = -1;

void Send(int fd) {
    const char token = 'x';
    assert(write(fd, &token, 1) == 1);
}

void Receive(int fd) {
    char token;
    assert(read(fd, &token, 1) == 1);
}

ampccl::LogicalCommKey Key(unsigned test, int ranks = 4) {
    // Use the same bootstrap hashing as actual communicator registration.
    const uint64_t bootstrap[] = {static_cast<uint64_t>(getpid()), test};
    return ampccl::BuildLogicalKeyFromBootstrap(ranks, bootstrap, sizeof(bootstrap), 0);
}

void Unlink(const ampccl::LogicalCommKey& key) {
    std::ostringstream name;
    name << "/ampccl_" << std::hex << std::hash<ampccl::LogicalCommKey>{}(key);
    shm_unlink(name.str().c_str());
}

void RunRank(const ampccl::LogicalCommKey& logical, int rank, int ranks, int rounds,
             int published = -1, double initial_alpha = 0.25) {
    ampccl::ShmParamStore store;
    assert(store.Attach(logical, rank, ranks));
    ampccl::ParamCache cache;
    const ampccl::OpKey op{ampccl::CollectiveType::AllGather, 1024, 1024, 1, ranks, 17};
    for (int round = 0; round < rounds; ++round) {
        const double alpha = round % 2 == 0 ? initial_alpha : initial_alpha / 2;
        if (rank == 0) {
            store.WaitUntilParamConsumed();
            cache.Update(op, ampccl::ParamValue(alpha, true, 1.0, 2.0));
            store.WriteParams(cache, op);
            store.SignalParamWritten();
            if (round == 0 && published >= 0) Send(published);
        }
        store.ReadParams(&cache);
        assert(cache.Contains(op));
        assert(std::abs(cache.Lookup(op).alpha - alpha) < 1e-12);
        store.BarrierSync();
    }
}

bool WaitChildren(std::vector<pid_t> children) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    bool ok = true;
    while (!children.empty() && std::chrono::steady_clock::now() < deadline) {
        for (auto it = children.begin(); it != children.end();) {
            int status = 0;
            const pid_t result = waitpid(*it, &status, WNOHANG);
            if (result == *it) {
                ok = ok && WIFEXITED(status) && WEXITSTATUS(status) == 0;
                it = children.erase(it);
            } else {
                assert(result == 0);
                ++it;
            }
        }
        if (!children.empty()) usleep(1000);
    }
    for (pid_t pid : children) {
        kill(pid, SIGKILL);
        waitpid(pid, nullptr, 0);
    }
    return ok && children.empty();
}

bool DelayedInitializerCannotErasePublishedRound() {
    const auto key = Key(1);
    int entered[2], release[2], published[2];
    assert(pipe(entered) == 0 && pipe(release) == 0 && pipe(published) == 0);
    const pid_t reader = fork();
    assert(reader >= 0);
    if (reader == 0) {
        pause_initialization = true;
        initialization_entered = entered[1];
        initialization_release = release[0];
        RunRank(key, 1, 2, 4);
        _exit(0);
    }
    Receive(entered[0]); // Reader has observed a fresh segment and is about to clear it.
    const pid_t writer = fork();
    assert(writer >= 0);
    if (writer == 0) {
        RunRank(key, 0, 2, 4, published[1]);
        _exit(0);
    }
    // Without interprocess attach serialization, rank 0 publishes while rank 1
    // is paused, and rank 1 subsequently erases the published count/barrier.
    // With serialization, rank 0 remains in Attach until we release rank 1.
    pollfd ready{published[0], POLLIN, 0};
    const int publication = poll(&ready, 1, 300);
    assert(publication >= 0);
    if (publication > 0) {
        std::fprintf(stderr, "Observed first-round publication while another rank is still initializing\n");
    }
    Send(release[1]);
    const bool ok = WaitChildren({reader, writer});
    for (int fd : {entered[0], entered[1], release[0], release[1], published[0], published[1]}) close(fd);
    Unlink(key);
    if (!ok) std::fprintf(stderr, "FAIL: delayed initializer erased a published round; ranks did not finish\n");
    return ok;
}

bool RepeatedFourRankStartup() {
    for (unsigned launch = 0; launch < 40; ++launch) {
        const auto key = Key(100 + launch);
        // Reuse each segment for a second communication domain lifetime.
        for (int reuse = 0; reuse < 2; ++reuse) {
            std::vector<pid_t> children;
            for (int rank = 3; rank >= 0; --rank) {
                const pid_t pid = fork();
                assert(pid >= 0);
                if (pid == 0) {
                    if (rank == 0) usleep(1000); // Exercise rank 0 attaching last.
                    RunRank(key, rank, 4, 20);
                    _exit(0);
                }
                children.push_back(pid);
            }
            if (!WaitChildren(children)) {
                Unlink(key);
                std::fprintf(stderr, "FAIL: 4-rank launch %u reuse %d\n", launch, reuse);
                return false;
            }
        }
        Unlink(key);
    }
    return true;
}

bool OverlappingTpDpGroupsRemainIndependent() {
    // Eight processes, TP={0,1,2,3}/{4,5,6,7}, DP={0,4}/{1,5}/{2,6}/{3,7}.
    // Every process initializes/runs its TP and DP store on separate threads.
    for (unsigned launch = 0; launch < 10; ++launch) {
        std::vector<ampccl::LogicalCommKey> groups;
        for (int group = 0; group < 6; ++group) {
            groups.push_back(Key(1000 + launch * 6 + group, group < 2 ? 4 : 2));
        }
        std::vector<pid_t> children;
        for (int global_rank = 7; global_rank >= 0; --global_rank) {
            const pid_t pid = fork();
            assert(pid >= 0);
            if (pid == 0) {
                const int tp_group = global_rank / 4;
                const int dp_group = 2 + global_rank % 4;
                std::thread tp([&] {
                    RunRank(groups[tp_group], global_rank % 4, 4, 30, -1,
                            0.25 + tp_group * 0.03125);
                });
                std::thread dp([&] {
                    RunRank(groups[dp_group], global_rank / 4, 2, 30, -1,
                            0.5 + dp_group * 0.03125);
                });
                tp.join();
                dp.join();
                _exit(0);
            }
            children.push_back(pid);
        }
        const bool ok = WaitChildren(children);
        for (const auto& group : groups) Unlink(group);
        if (!ok) {
            std::fprintf(stderr, "FAIL: concurrent overlapping TP/DP launch %u\n", launch);
            return false;
        }
    }
    return true;
}
} // namespace

// Delay the real initialization memset using the linker's test-only wrapper.
// The mapping, futexes, parameter reads and barriers are all production code.
extern "C" void* __real_memset(void*, int, size_t);
extern "C" void* __wrap_memset(void* ptr, int value, size_t size) {
    if (pause_initialization && size > 10000) {
        pause_initialization = false;
        Send(initialization_entered);
        Receive(initialization_release);
    }
    return __real_memset(ptr, value, size);
}

int main() {
    if (!DelayedInitializerCannotErasePublishedRound()) return 1;
    std::puts("PASS: delayed initialization preserves first-round parameters/barrier");
    if (!RepeatedFourRankStartup()) return 1;
    std::puts("PASS: 80 four-rank startups, including segment reuse; 20 parameter rounds each");
    if (!OverlappingTpDpGroupsRemainIndependent()) return 1;
    std::puts("PASS: 10 eight-process startups with 2 TP + 4 DP groups concurrently; 30 rounds/group");
}
