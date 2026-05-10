#ifndef AMPCCL_COMMON_CONFIG_H_
#define AMPCCL_COMMON_CONFIG_H_

#include <string>
#include <cstddef>
#include <cstdlib>
#include <cstring>

namespace ampccl {

enum class AdaptiveAlgorithm {
    INVERSE_PROPORTION,  // 反比例（原 tcp）
    SPLIT_RATIO_CONTROL,  // 自适应 split ratio（目标比例 + EMA + 瓶颈乘性降，见 adaptive_split_ratio_control.md）
    STATIC
};

class Config {
public:
    static bool IsTruthyEnv(const char* val) {
        if (val == nullptr) {
            return false;
        }
        return std::strcmp(val, "1") == 0 || std::strcmp(val, "on") == 0 ||
               std::strcmp(val, "ON") == 0 || std::strcmp(val, "true") == 0 ||
               std::strcmp(val, "TRUE") == 0 || std::strcmp(val, "yes") == 0 ||
               std::strcmp(val, "YES") == 0;
    }

    // Benchmark compatibility mode for CANN tools/hccl_test. Official hccl_test
    // launches many collectives back-to-back and synchronizes only at the end of
    // the batch, so feedback based on per-op completion timing must be disabled.
    // AMPCCL_HCCL_TEST_MODE=1|on|true|yes
    static bool IsHcclTestMode() {
        return IsTruthyEnv(std::getenv("AMPCCL_HCCL_TEST_MODE"));
    }

    // Feedback controls adaptive parameter updates. HCCL_TEST mode forces it off.
    // AMPCCL_FEEDBACK=0|off|false disables feedback outside HCCL_TEST mode.
    static bool IsFeedbackEnabled() {
        if (IsHcclTestMode()) {
            return false;
        }
        const char* val = std::getenv("AMPCCL_FEEDBACK");
        if (val == nullptr || val[0] == '\0') {
            return true;
        }
        if (std::strcmp(val, "0") == 0 || std::strcmp(val, "off") == 0 ||
            std::strcmp(val, "OFF") == 0 || std::strcmp(val, "false") == 0 ||
            std::strcmp(val, "FALSE") == 0) {
            return false;
        }
        return true;
    }

    // Master switch: whether to use Adaptive-CCL at all.
    // AMPCCL_ENABLE=1|on|true|yes -> use adaptive (split + PCIe); otherwise pass through to original NCCL/HCCL.
    static bool IsAdaptiveEnabled() {
        const char* val = std::getenv("AMPCCL_ENABLE");
        return IsTruthyEnv(val);
    }

    // torch_npu resolves HCCL APIs with dlsym(libhccl, name), which bypasses
    // ordinary ELF symbol interposition. Keep this bridge independently
    // switchable for diagnosing loader behavior.
    // AMPCCL_INTERPOSE_DLSYM=0 disables the bridge.
    static bool IsDlsymInterposeEnabled() {
        const char* val = std::getenv("AMPCCL_INTERPOSE_DLSYM");
        if (val == nullptr) {
            return true;
        }
        return std::strcmp(val, "0") != 0 && std::strcmp(val, "off") != 0 &&
               std::strcmp(val, "OFF") != 0 && std::strcmp(val, "false") != 0 &&
               std::strcmp(val, "FALSE") != 0;
    }

    // Algorithm selection via environment variable
    // AMPCCL_ALGO=inverse|invprop|tcp|split_ratio|static（tcp 为反比例算法别名）
    static AdaptiveAlgorithm GetAlgorithm() {
        if (IsHcclTestMode()) {
            return AdaptiveAlgorithm::STATIC;
        }
        const char* algo_str = std::getenv("AMPCCL_ALGO");
        if (algo_str == nullptr) {
            return AdaptiveAlgorithm::STATIC;
        }

        if (std::strcmp(algo_str, "inverse") == 0 || std::strcmp(algo_str, "INVERSE") == 0 ||
            std::strcmp(algo_str, "invprop") == 0 || std::strcmp(algo_str, "INVPROP") == 0 ||
            std::strcmp(algo_str, "tcp") == 0 || std::strcmp(algo_str, "TCP") == 0) {
            return AdaptiveAlgorithm::INVERSE_PROPORTION;
        }
        if (std::strcmp(algo_str, "split_ratio") == 0 || std::strcmp(algo_str, "SPLIT_RATIO") == 0 ||
            std::strcmp(algo_str, "split_ratio_control") == 0 ||
            std::strcmp(algo_str, "SPLIT_RATIO_CONTROL") == 0 || std::strcmp(algo_str, "src") == 0 ||
            std::strcmp(algo_str, "SRC") == 0) {
            return AdaptiveAlgorithm::SPLIT_RATIO_CONTROL;
        }
        if (std::strcmp(algo_str, "static") == 0 || std::strcmp(algo_str, "STATIC") == 0) {
            return AdaptiveAlgorithm::STATIC;
        }

        return AdaptiveAlgorithm::INVERSE_PROPORTION;
    }

    // Split-ratio control: EMA smoothing factor for PCIe bandwidth (0,1). Default 0.7.
    // AMPCCL_SPLIT_RATIO_SMOOTH
    static double GetSplitRatioSmoothing() {
        const char* val = std::getenv("AMPCCL_SPLIT_RATIO_SMOOTH");
        if (val == nullptr || val[0] == '\0') {
            return 0.7;
        }
        char* end = nullptr;
        double v = std::strtod(val, &end);
        if (end == val || v <= 0.0 || v >= 1.0) {
            return 0.7;
        }
        return v;
    }

    // Split-ratio control: proportional convergence rate eta (0,1). Default 0.3.
    // AMPCCL_SPLIT_RATIO_ETA
    static double GetSplitRatioEta() {
        const char* val = std::getenv("AMPCCL_SPLIT_RATIO_ETA");
        if (val == nullptr || val[0] == '\0') {
            return 0.3;
        }
        char* end = nullptr;
        double v = std::strtod(val, &end);
        if (end == val || v <= 0.0 || v >= 1.0) {
            return 0.3;
        }
        return v;
    }

    // Static algorithm: fixed PCIe split ratio alpha in [0,1]. Default 0.8.
    // AMPCCL_STATIC_ALPHA
    static double GetStaticAlpha() {
        const char* val = std::getenv("AMPCCL_STATIC_ALPHA");
        if (val == nullptr || val[0] == '\0') {
            return 0.8;
        }
        char* end = nullptr;
        double v = std::strtod(val, &end);
        if (end == val || v < 0.0 || v > 1.0) {
            return 0.8;
        }
        return v;
    }

    // Minimum chunk size for PCIe (bytes)
    // AMPCCL_MIN_CHUNK_SIZE (default: 4096)
    static size_t GetMinChunkSize() {
        const char* val = std::getenv("AMPCCL_MIN_CHUNK_SIZE");
        if (val == nullptr) {
            return 4096;
        }
        return std::stoull(val);
    }

    // Minimum message size to enable PCIe (bytes)
    // AMPCCL_MIN_MSG_SIZE (default: 4MB)
    static size_t GetMinMsgSize() {
        const char* val = std::getenv("AMPCCL_MIN_MSG_SIZE");
        if (val == nullptr) {
            return 4ULL * 1024ULL * 1024ULL;
        }
        return std::stoull(val);
    }

    // Size bucket base in bytes. Messages below this do not get a reusable
    // adaptive size bucket. Default matches GetMinMsgSize().
    // AMPCCL_SIZE_BUCKET_BASE (default: 4MB)
    static size_t GetSizeBucketBase() {
        const char* val = std::getenv("AMPCCL_SIZE_BUCKET_BASE");
        if (val == nullptr || val[0] == '\0') {
            return GetMinMsgSize();
        }
        size_t parsed = std::stoull(val);
        return parsed == 0 ? GetMinMsgSize() : parsed;
    }

    // Number of logarithmic buckets per 2x size range. Default 2 means
    // adjacent bucket boundaries are sqrt(2) apart.
    // AMPCCL_SIZE_BUCKETS_PER_OCTAVE (default: 2)
    static int GetSizeBucketsPerOctave() {
        const char* val = std::getenv("AMPCCL_SIZE_BUCKETS_PER_OCTAVE");
        if (val == nullptr || val[0] == '\0') {
            return 2;
        }
        char* end = nullptr;
        long v = std::strtol(val, &end, 10);
        if (end == val || v <= 0 || v > 16) {
            return 2;
        }
        return static_cast<int>(v);
    }

    static size_t GetMaxPCIeRecvBufBytes() {
        const char* val = std::getenv("AMPCCL_MAX_PCIE_RECVBUF_BYTES");
        if (val == nullptr) {
            return 512ULL * 1024 * 1024;
        }
        return std::stoull(val);
    }

    // Cross-rank shared-memory param / stats sync (POSIX shm). Required for correct
    // adaptive splits when each OS process runs one rank (MPI). Single-process
    // multi-thread tests do not advance logical rounds together; disable with 0.
    // AMPCCL_SHM_PARAMS=1|0 (default: 1)
    static bool IsShmParamSyncEnabled() {
        if (IsHcclTestMode() || !IsFeedbackEnabled()) {
            return false;
        }
        const char* val = std::getenv("AMPCCL_SHM_PARAMS");
        if (val == nullptr) {
            return true;
        }
        if (std::strcmp(val, "0") == 0 || std::strcmp(val, "off") == 0 ||
            std::strcmp(val, "OFF") == 0 || std::strcmp(val, "false") == 0 ||
            std::strcmp(val, "FALSE") == 0) {
            return false;
        }
        return true;
    }

    // Enable/disable PCIe backend
    // AMPCCL_ENABLE_PCIE=1|0 (default: 1)
    static bool IsPCIeEnabled() {
        const char* val = std::getenv("AMPCCL_ENABLE_PCIE");
        if (val == nullptr) {
            return true;
        }
        return std::strcmp(val, "0") != 0;
    }

    // Enable NCCL P2P-based V fast path inside AMP-CCL mixed collectives.
    // Default off so NCCL mixed collectives use the staging + merge fallback unless explicitly enabled.
    // AMPCCL_ENABLE_NCCL_V=1|on|true|yes
    static bool IsNcclVEnabled() {
        const char* val = std::getenv("AMPCCL_ENABLE_NCCL_V");
        if (val == nullptr) {
            return false;
        }
        return std::strcmp(val, "1") == 0 || std::strcmp(val, "on") == 0 ||
               std::strcmp(val, "ON") == 0 || std::strcmp(val, "true") == 0 ||
               std::strcmp(val, "TRUE") == 0 || std::strcmp(val, "yes") == 0 ||
               std::strcmp(val, "YES") == 0;
    }

    // Correctness gate for multi-node jobs. Default on: AMP-CCL only takes over
    // communicators whose every rank is visible in local node-scope shm.
    // AMPCCL_INTRA_NODE_ONLY=1|0 (default: 1)
    static bool IsIntraNodeOnlyEnabled() {
        const char* val = std::getenv("AMPCCL_INTRA_NODE_ONLY");
        if (val == nullptr || val[0] == '\0') {
            return true;
        }
        if (std::strcmp(val, "0") == 0 || std::strcmp(val, "off") == 0 ||
            std::strcmp(val, "OFF") == 0 || std::strcmp(val, "false") == 0 ||
            std::strcmp(val, "FALSE") == 0) {
            return false;
        }
        return true;
    }

    // Max time to wait for other local ranks to register during CommInit.
    // AMPCCL_NODE_SCOPE_WAIT_MS (default: 200)
    static int GetNodeScopeWaitMs() {
        const char* val = std::getenv("AMPCCL_NODE_SCOPE_WAIT_MS");
        if (val == nullptr || val[0] == '\0') {
            return 200;
        }
        char* end = nullptr;
        long v = std::strtol(val, &end, 10);
        if (end == val || v < 0 || v > 60000) {
            return 200;
        }
        return static_cast<int>(v);
    }

    // Diagnostic override for experiments. This bypasses only the node-scope
    // gate; AMPCCL_ENABLE and AMPCCL_ENABLE_PCIE still apply.
    // AMPCCL_FORCE_ENABLE_INTER_NODE=1|on|true|yes (default: 0)
    static bool IsForceEnableInterNode() {
        const char* val = std::getenv("AMPCCL_FORCE_ENABLE_INTER_NODE");
        if (val == nullptr) {
            return false;
        }
        return std::strcmp(val, "1") == 0 || std::strcmp(val, "on") == 0 ||
               std::strcmp(val, "ON") == 0 || std::strcmp(val, "true") == 0 ||
               std::strcmp(val, "TRUE") == 0 || std::strcmp(val, "yes") == 0 ||
               std::strcmp(val, "YES") == 0;
    }

    // Hard runtime disable, useful when AMPCCL_ENABLE is inherited globally.
    // AMPCCL_FORCE_DISABLE=1|on|true|yes (default: 0)
    static bool IsForceDisable() {
        const char* val = std::getenv("AMPCCL_FORCE_DISABLE");
        if (val == nullptr) {
            return false;
        }
        return std::strcmp(val, "1") == 0 || std::strcmp(val, "on") == 0 ||
               std::strcmp(val, "ON") == 0 || std::strcmp(val, "true") == 0 ||
               std::strcmp(val, "TRUE") == 0 || std::strcmp(val, "yes") == 0 ||
               std::strcmp(val, "YES") == 0;
    }

    // Debug logging
    // AMPCCL_DEBUG=1|0 (default: 0)
    static bool IsDebugEnabled() {
        const char* val = std::getenv("AMPCCL_DEBUG");
        if (val == nullptr) {
            return false;
        }
        return std::strcmp(val, "0") != 0;
    }

    // Diagnostic fallback: explicitly synchronize the PCIe stream before registering
    // the user-stream host callback. Default off; enable with
    // AMPCCL_SYNC_PCIE_STREAM_BEFORE_CALLBACK=1|on|true|yes.
    static bool IsPCIeStreamSyncBeforeCallbackEnabled() {
        const char* val = std::getenv("AMPCCL_SYNC_PCIE_STREAM_BEFORE_CALLBACK");
        if (val == nullptr) {
            return false;
        }
        if (std::strcmp(val, "1") == 0 || std::strcmp(val, "on") == 0 ||
            std::strcmp(val, "ON") == 0 || std::strcmp(val, "true") == 0 ||
            std::strcmp(val, "TRUE") == 0 || std::strcmp(val, "yes") == 0 ||
            std::strcmp(val, "YES") == 0) {
            return true;
        }
        return false;
    }

    // Diagnostic fallback: synchronize timer end events from the host callback.
    // Default off because the callback is already ordered after the user-stream
    // timer event, and the user stream also waits on the PCIe timer event when
    // PCIe is used. Enable only for debugging old behavior:
    // AMPCCL_SYNC_TIMER_EVENTS_IN_CALLBACK=1|on|true|yes.
    static bool IsTimerEventSyncInCallbackEnabled() {
        const char* val = std::getenv("AMPCCL_SYNC_TIMER_EVENTS_IN_CALLBACK");
        if (val == nullptr) {
            return false;
        }
        if (std::strcmp(val, "1") == 0 || std::strcmp(val, "on") == 0 ||
            std::strcmp(val, "ON") == 0 || std::strcmp(val, "true") == 0 ||
            std::strcmp(val, "TRUE") == 0 || std::strcmp(val, "yes") == 0 ||
            std::strcmp(val, "YES") == 0) {
            return true;
        }
        return false;
    }

    // Print average bandwidth (algbw / busbw / bus_factor) at communicator destroy.
    // Only meaningful with AMPCCL_SHM_PARAMS=1 (multi-rank shared-memory path).
    // AMPCCL_AVG_BW_LOG=1|on|true|yes (default: off)
    static bool IsAvgBwLogEnabled() {
        const char* val = std::getenv("AMPCCL_AVG_BW_LOG");
        if (val == nullptr) {
            return false;
        }
        if (std::strcmp(val, "1") == 0 || std::strcmp(val, "on") == 0 ||
            std::strcmp(val, "ON") == 0 || std::strcmp(val, "true") == 0 ||
            std::strcmp(val, "TRUE") == 0 || std::strcmp(val, "yes") == 0 ||
            std::strcmp(val, "YES") == 0) {
            return true;
        }
        return false;
    }

    // Enable timer/stat callback for AMP-CCL-managed fast-only collectives
    // (plan.pcie_bytes == 0). Default off to keep pure HCCL/NCCL overhead
    // unchanged.
    // AMPCCL_FAST_ONLY_BW_LOG=1|on|true|yes
    static bool IsFastOnlyBwLogEnabled() {
        return IsTruthyEnv(std::getenv("AMPCCL_FAST_ONLY_BW_LOG"));
    }

    // Number of warmup iterations to skip before recording bandwidth for the average.
    // AMPCCL_AVG_BW_WARMUP (default: 0)
    static int GetAvgBwWarmupIters() {
        const char* val = std::getenv("AMPCCL_AVG_BW_WARMUP");
        if (val == nullptr || val[0] == '\0') {
            return 0;
        }
        char* end = nullptr;
        long v = std::strtol(val, &end, 10);
        if (end == val || v < 0) {
            return 0;
        }
        return static_cast<int>(v);
    }

};

}  // namespace ampccl

#endif  // AMPCCL_COMMON_CONFIG_H_
