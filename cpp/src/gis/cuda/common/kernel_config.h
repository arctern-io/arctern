// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#pragma once
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <stdexcept>

namespace zilliz {
namespace gis {
namespace cuda {

struct KernelExecConfig {
  int64_t grid_dim;
  int64_t block_dim;
};

inline KernelExecConfig GetKernelExecConfig(int64_t total_threads,
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
  if (ec != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(ec));
  }
}

}  // namespace cuda
}  // namespace gis
}  // namespace zilliz
