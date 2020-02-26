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

#include "gis/cuda/common/gpu_memory.h"
#include "gis/cuda/functor/st_area.h"

namespace zilliz {
namespace gis {
namespace cuda {

namespace {
inline DEVICE_RUNNABLE double PolygonArea(const GpuContext& ctx, int index) {
  auto meta = ctx.get_meta_ptr(index);
  auto value = ctx.get_value_ptr(index);
  assert(meta[0] == 1);
  auto count = (int)meta[1];
  double sum_area = 0;
  for (int point_index = 0; point_index < count; ++point_index) {
    auto lv = value + 2 * point_index;
    auto rv = (point_index + 1 == count) ? value : lv + 2;
    auto area = lv[0] * rv[1] - lv[1] * rv[0];
    sum_area += area;
  }
  return fabs(sum_area / 2);
}

__global__ void ST_AreaKernel(GpuContext ctx, double* result) {
  auto tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < ctx.size) {
    auto tag = ctx.get_tag(tid);
    // handle 2d case only for now
    assert(tag.get_group() == WkbGroup::None);
    switch (tag.get_category()) {
      // handle polygon case only
      case WkbCategory::Polygon: {
        result[tid] = PolygonArea(ctx, tid);
        break;
      }
      default: { result[tid] = 0; }
    }
  }
}
}  // namespace

void ST_Area(const GeometryVector& vec, double* host_results) {
  auto ctx_holder = vec.CreateReadGpuContext();
  auto config = GetKernelExecConfig(vec.size());
  auto dev_result = GpuMakeUniqueArray<double>(vec.size());
  ST_AreaKernel<<<config.grid_dim, config.block_dim>>>(*ctx_holder, dev_result.get());
  GpuMemcpy(host_results, dev_result.get(), vec.size());
}

}  // namespace cuda
}  // namespace gis
}  // namespace zilliz
