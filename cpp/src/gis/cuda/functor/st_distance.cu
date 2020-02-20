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

//
// Created by mike on 2/10/20.
//
#include <cuda_runtime.h>

#include <cmath>

#include "gis/cuda/common/gpu_memory.h"
#include "gis/cuda/functor/st_distance.h"

namespace zilliz {
namespace gis {
namespace cuda {
namespace {
inline DEVICE_RUNNABLE double Point2PointDistance(const GpuContext& left,
                                                  const GpuContext& right, int index) {
  auto lv = left.get_value_ptr(index);
  auto rv = right.get_value_ptr(index);
  auto dx = (lv[0] - rv[0]);
  auto dy = (lv[1] - rv[1]);
  return sqrt(dx * dx + dy * dy);
}

__global__ void ST_DistanceKernel(GpuContext left, GpuContext right, double* result) {
  auto tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < left.size) {
    auto left_tag = left.get_tag(tid);
    auto right_tag = right.get_tag(tid);
    // handle 2d case only for now
    assert(left_tag.get_group() == WkbGroup::None);
    assert(right_tag.get_group() == WkbGroup::None);
    // handle point to point case only
    if (left_tag.get_category() == WkbCategory::Point &&
        right_tag.get_category() == WkbCategory::Point) {
      result[tid] = Point2PointDistance(left, right, tid);
    } else {
      result[tid] = NAN;
    }
  }
}
}  // namespace

void ST_Distance(const GeometryVector& left_vec, const GeometryVector& right_vec,
                 double* host_results) {
  assert(left_vec.size() == right_vec.size());
  auto left_ctx_holder = left_vec.CreateReadGpuContext();
  auto right_ctx_holder = right_vec.CreateReadGpuContext();
  auto dev_result = GpuMakeUniqueArray<double>(left_vec.size());
  {
    auto config = GetKernelExecConfig(left_vec.size());
    ST_DistanceKernel<<<config.grid_dim, config.block_dim>>>(
        *left_ctx_holder, *right_ctx_holder, dev_result.get());
  }
  GpuMemcpy(host_results, dev_result.get(), left_vec.size());
}

}  // namespace cuda
}  // namespace gis
}  // namespace zilliz
