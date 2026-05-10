#ifndef AMPCCL_COMMON_CONFIG_H_
#define AMPCCL_COMMON_CONFIG_H_

#include <string>
#include <cstddef>
#include <cstdlib>
#include <cstring>

namespace ampccl {

enum class AdaptiveAlgorithm {
    TCP,      // TCP-style AIMD
    DCQCN,    // DCQCN-style
    STATIC    // Static fixed ratio
};

class Config {
public:
    // Master switch: whether to use Adaptive-CCL at all.
    // AMPCCL_ENABLE=1|on|true|yes -> use adaptive (split + PCIe); otherwise pass through to original NCCL/HCCL.
    static bool IsAdaptiveEnabled() {
        const char* val = std::getenv("AMPCCL_ENABLE");
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

    // Algorithm selection via environment variable
    // AMPCCL_ALGO=tcp|dcqcn|static (default: tcp)
    static AdaptiveAlgorithm GetAlgorithm() {
        const char* algo_str = std::getenv("AMPCCL_ALGO");
        if (algo_str == nullptr) {
            return AdaptiveAlgorithm::STATIC;
        }

        if (std::strcmp(algo_str, "tcp") == 0 || std::strcmp(algo_str, "TCP") == 0) {
            return AdaptiveAlgorithm::TCP;
        } else if (std::strcmp(algo_str, "dcqcn") == 0 || std::strcmp(algo_str, "DCQCN") == 0) {
            return AdaptiveAlgorithm::DCQCN;
        } else if (std::strcmp(algo_str, "static") == 0 || std::strcmp(algo_str, "STATIC") == 0) {
            return AdaptiveAlgorithm::STATIC;
        }

        return AdaptiveAlgorithm::TCP;  // default
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
    // AMPCCL_MIN_MSG_SIZE (default: 8192)
    static size_t GetMinMsgSize() {
        const char* val = std::getenv("AMPCCL_MIN_MSG_SIZE");
        if (val == nullptr) {
            return 8192;
        }
        return std::stoull(val);
    }

    static size_t GetMaxPCIeRecvBufBytes() {
        const char* val = std::getenv("AMPCCL_MAX_PCIE_RECVBUF_BYTES");
        if (val == nullptr) {
            return 512ULL * 1024 * 1024;
        }
        return std::stoull(val);
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

    // Debug logging
    // AMPCCL_DEBUG=1|0 (default: 0)
    static bool IsDebugEnabled() {
        const char* val = std::getenv("AMPCCL_DEBUG");
        if (val == nullptr) {
            return false;
        }
        return std::strcmp(val, "0") != 0;
    }
};

}  // namespace ampccl

#endif  // AMPCCL_COMMON_CONFIG_H_
