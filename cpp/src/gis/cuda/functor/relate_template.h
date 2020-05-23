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

template <typename Func>
void ST_RelateFunctor(Func func, const GeometryVector& left_vec,
                      const GeometryVector& right_vec, bool* host_results) {
  auto size = left_vec.size();
  auto left_ctx_holder = left_vec.CreateReadGpuContext();
  auto right_ctx_holder = right_vec.CreateReadGpuContext();
  auto matrices = GenRelateMatrix(*left_ctx_holder, *right_ctx_holder);
  auto results = GpuMakeUniqueArray<bool>(size);

  RelationFinalize(func, matrices.get(), left_vec.size(), results.get());
  GpuMemcpy(host_results, results.get(), size);
}

namespace internal {
template <typename Func>
__global__ void RelationFinalizeWithDimImpl(Func func, ConstGpuContext left,
                                            ConstGpuContext right,
                                            const de9im::Matrix* dev_matrices,
                                            int64_t size, bool* dev_results) {
  auto index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < size) {
    auto left_dim = left.get_tag(index).get_dimension();
    auto right_dim = right.get_tag(index).get_dimension();
    dev_results[index] = func(dev_matrices[index], left_dim, right_dim);
  }
}
}  // namespace internal

template <typename Func>
void RelationFinalizeWithDim(Func func, ConstGpuContext left, ConstGpuContext right,
                             const de9im::Matrix* dev_matrices, int64_t size,
                             bool* dev_results) {
  auto config = GetKernelExecConfig(size);
  internal::RelationFinalizeWithDimImpl<<<config.grid_dim, config.block_dim>>>(
      func, left, right, dev_matrices, size, dev_results);
}

template <typename Func>
void ST_RelateFunctorWithDim(Func func, const GeometryVector& left_vec,
                             const GeometryVector& right_vec, bool* host_results) {
  auto size = left_vec.size();
  auto left_ctx_holder = left_vec.CreateReadGpuContext();
  auto right_ctx_holder = right_vec.CreateReadGpuContext();
  auto matrices = GenRelateMatrix(*left_ctx_holder, *right_ctx_holder);
  auto results = GpuMakeUniqueArray<bool>(size);
  RelationFinalizeWithDim(func, *left_ctx_holder, *right_ctx_holder, matrices.get(),
                          left_vec.size(), results.get());
  GpuMemcpy(host_results, results.get(), size);
}

}  // namespace cuda
}  // namespace gis
}  // namespace arctern
