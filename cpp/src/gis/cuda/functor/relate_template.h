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
#include "gis/cuda/functor/st_relate.h"

namespace arctern {
namespace gis {
namespace cuda {

inline auto GenRelateMatrix(const ConstGpuContext& left_ctx,
                            const ConstGpuContext& right_ctx) {
  auto len = left_ctx.size;
  auto result = GpuMakeUniqueArray<de9im::Matrix>(len);
  ST_Relate(left_ctx, right_ctx, result.get());
  return result;
}

namespace internal {
template <typename Func>
__global__ void RelationFinalizeImpl(Func func, const de9im::Matrix* dev_matrices,
                                     int64_t size, bool* dev_results) {
  auto index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < size) {
    dev_results[index] = func(dev_matrices[index]);
  }
}
}  // namespace internal

template <typename Func>
void RelationFinalize(Func func, const de9im::Matrix* dev_matrices, int64_t size,
                      bool* dev_results) {
  auto config = GetKernelExecConfig(size);
  internal::RelationFinalizeImpl<<<config.grid_dim, config.block_dim>>>(
      func, dev_matrices, size, dev_results);
}

}  // namespace cuda
}  // namespace gis
}  // namespace arctern
