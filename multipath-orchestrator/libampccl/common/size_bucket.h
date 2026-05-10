#ifndef AMPCCL_COMMON_SIZE_BUCKET_H_
#define AMPCCL_COMMON_SIZE_BUCKET_H_

#include "common/config.h"
#include <cmath>
#include <cstddef>
#include <limits>

namespace ampccl {

class SizeBucket {
public:
    static size_t BucketBytes(size_t bytes) {
        const size_t base = Config::GetSizeBucketBase();
        if (base == 0 || bytes < base) {
            return 0;
        }

        const int buckets_per_octave = Config::GetSizeBucketsPerOctave();
        const double ratio = std::pow(2.0, 1.0 / static_cast<double>(buckets_per_octave));

        size_t bucket = base;
        for (;;) {
            double next_d = std::ceil(static_cast<double>(bucket) * ratio);
            if (next_d > static_cast<double>(std::numeric_limits<size_t>::max())) {
                return bucket;
            }
            size_t next = static_cast<size_t>(next_d);
            if (next <= bucket) {
                if (bucket == std::numeric_limits<size_t>::max()) {
                    return bucket;
                }
                next = bucket + 1;
            }
            if (bytes < next) {
                return bucket;
            }
            bucket = next;
        }
    }
};

}  // namespace ampccl

#endif  // AMPCCL_COMMON_SIZE_BUCKET_H_
