#pragma once
#include <cstdint>
#include <algorithm>
#include <cassert>
#include <stdexcept>

namespace zilliz {
namespace gis {
namespace cuda {

struct KernelExecConfig {
    int64_t grid_dim;
    int64_t block_dim;
};

inline KernelExecConfig
GetKernelExecConfig(int64_t total_threads,
                    int64_t block_dim = 256,
                    int64_t max_grid_dim = -1) {
    int64_t grid_dim = (total_threads + block_dim - 1) / block_dim;
    if (max_grid_dim != -1) {
        grid_dim = std::min(grid_dim, max_grid_dim);
    }
    assert(total_threads <= grid_dim * block_dim);
    assert((grid_dim - 1) * block_dim < total_threads);
    if (grid_dim < 1) {
        // to avoid invalid kernel config exception
        // TODO: consider warning on it
        grid_dim = 1;
    }
    return KernelExecConfig{grid_dim, block_dim};
}

inline void check_cuda_last_error() {
    auto ec = cudaGetLastError();
    if(ec != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(ec));
    }
}

}    // namespace cuda
}    // namespace gis
}    // namespace zilliz
