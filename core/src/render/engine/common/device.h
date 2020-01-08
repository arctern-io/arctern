#pragma once

namespace zilliz {
namespace common {

using DeviceIndex = int32_t;
using EnumType = uint64_t;

enum class StorageLevel : EnumType {
    kDFS = 0,   // distributed file system
    kFS,        // local file system
    kMM,        // main memory
    kGPU,       // GPU memory
    kNumLevels
};

struct DeviceID {
    StorageLevel storage_level;
    DeviceIndex dev_idx;

    bool
    operator==(const DeviceID &that) const {
        return storage_level == that.storage_level &&
            dev_idx == that.dev_idx;
    }

    bool
    operator<(const DeviceID &that) const {
        if (storage_level != that.storage_level)
            return storage_level < that.storage_level;
        return dev_idx < that.dev_idx;
    }
};

} //common
} //zilliz
