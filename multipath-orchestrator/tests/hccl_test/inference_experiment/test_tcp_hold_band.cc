#include "controller/algo_inverse_proportion.h"
#include <cassert>
#include <cmath>
#include <cstdlib>

// Catches ignored configuration, changed default behavior, wrong boundary
// branches, and invalid intervals silently suppressing all adaptation.
static double After(double rate, const char* low, const char* high) {
    if (low) setenv("AMPCCL_TCP_HOLD_RATE_LOW", low, 1);
    else unsetenv("AMPCCL_TCP_HOLD_RATE_LOW");
    if (high) setenv("AMPCCL_TCP_HOLD_RATE_HIGH", high, 1);
    else unsetenv("AMPCCL_TCP_HOLD_RATE_HIGH");
    ampccl::InverseProportionAlgo algo(2);
    algo.Suggest(ampccl::ParamValue(0.2, true, 0, 0));
    ampccl::ExecStat stat;
    stat.fast_time = rate;
    stat.pcie_time = 1;
    stat.fast_bytes = stat.pcie_bytes = 1024 * 1024;
    algo.Update(stat); algo.Update(stat); algo.Update(stat);
    return algo.GetAlpha();
}
static void Expect(double rate, const char* low, const char* high, double want) {
    assert(std::abs(After(rate, low, high) - want) < 1e-12);
}
int main() {
    Expect(0.84, "0.80", "1.10", 0.2);
    Expect(0.90, nullptr, nullptr, 0.1);
    Expect(1.0, nullptr, nullptr, 0.2);
    Expect(1.2, nullptr, nullptr, 0.2304);
    Expect(0.80, "0.80", "1.10", 0.1);
    Expect(1.10, "0.80", "1.10", 0.2112);
    Expect(0.89, "0.90", "1.10", 0.1);
    Expect(0.86, "0.85", "1.10", 0.2);
    for (const char* invalid : {"nan", "inf", "bad", "0.8junk", "0", "-1", "1.2"})
        Expect(0.9, invalid, "1.10", 0.1);
    Expect(0.9, "0.8", "0.7", 0.1);
    Expect(0.9, "0.8", "bad", 0.1);
}
