#include "common/intercept_policy.h"

#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <limits>

namespace {

constexpr size_t kMiB = 1024ULL * 1024ULL;
constexpr size_t kThreshold = 256ULL * kMiB;

void TestEightRankBoundary() {
    const size_t elements = kThreshold / sizeof(float);

    assert(ampccl::ShouldInterceptCollective(
        ampccl::CollectiveType::AllReduce, 8, sizeof(float), 0, 0, elements));
    assert(!ampccl::ShouldInterceptCollective(
        ampccl::CollectiveType::AllReduce, 8, sizeof(float), 0, 0, elements - 1));

    const size_t per_rank = kThreshold / (8 * sizeof(float));
    assert(ampccl::ShouldInterceptCollective(
        ampccl::CollectiveType::AllGather, 8, sizeof(float), per_rank, 0, 0));
    assert(ampccl::ShouldInterceptCollective(
        ampccl::CollectiveType::ReduceScatter, 8, sizeof(float), 0, per_rank, 0));
    assert(ampccl::ShouldInterceptCollective(
        ampccl::CollectiveType::AllToAll, 8, sizeof(float), 0, 0, per_rank));
}

void TestConfiguredNonEightRankThreshold() {
    setenv("AMPCCL_MIN_MSG_SIZE", "1024", 1);
    assert(ampccl::ShouldInterceptCollective(
        ampccl::CollectiveType::AllReduce, 4, sizeof(float), 0, 0, 256));
    assert(!ampccl::ShouldInterceptCollective(
        ampccl::CollectiveType::AllReduce, 4, sizeof(float), 0, 0, 255));
    unsetenv("AMPCCL_MIN_MSG_SIZE");
}

void TestInvalidInputsBypass() {
    assert(!ampccl::ShouldInterceptCollective(
        ampccl::CollectiveType::AllReduce, 8, 0, 0, 0, 1));
    assert(!ampccl::ShouldInterceptCollective(
        ampccl::CollectiveType::AllReduce, 8, sizeof(float), 0, 0,
        std::numeric_limits<size_t>::max()));
}

}  // namespace

int main() {
    TestEightRankBoundary();
    TestConfiguredNonEightRankThreshold();
    TestInvalidInputsBypass();
    return 0;
}
