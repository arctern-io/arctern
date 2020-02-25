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

#include "gis/cuda/common/gpu_memory.h"
#include "gis/cuda/functor/st_length.h"

namespace zilliz {
namespace gis {
namespace cuda {

namespace {
using Iter = GpuContext::Iter;
using ConstIter = GpuContext::ConstIter;

inline DEVICE_RUNNABLE double LineStringLength(ConstIter& iter) {
  auto count = *iter.metas++;
  double sum_length = 0;
  for (int point_index = 0; point_index < count - 1; ++point_index) {
    auto lv = iter.values + 2 * point_index;
    auto rv = lv + 2;
    auto dx = lv[0] - rv[0];
    auto dy = lv[1] - rv[1];
    auto length = sqrt(dx * dx + dy * dy);
    sum_length += length;
  }
  iter.values += 2 * count;
  return sum_length;
}

__global__ void ST_LengthKernel(const GpuContext ctx, double* results) {
  auto index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < ctx.size) {
    auto tag = ctx.get_tag(index);
    // handle 2d case only for now
    assert(tag.get_group() == WkbGroup::None);
    double result;
    switch (tag.get_category()) {
      // handle polygon case only
      case WkbCategory::LineString: {
        ConstIter iter = ctx.get_iter(index);
        result = LineStringLength(iter);
        assert(iter.metas == ctx.get_meta_ptr(index + 1));
        assert(iter.values == ctx.get_value_ptr(index + 1));
        break;
      }
      default: {
        result = 0;
      }
    }
    results[index] = result;
  }
}
}  // namespace

void ST_Length(const GeometryVector& vec, double* host_results) {
  auto ctx_holder = vec.CreateReadGpuContext();
  auto config = GetKernelExecConfig(vec.size());
  auto dev_result = GpuMakeUniqueArray<double>(vec.size());
  ST_LengthKernel<<<config.grid_dim, config.block_dim>>>(*ctx_holder, dev_result.get());
  GpuMemcpy(host_results, dev_result.get(), vec.size());
}

}  // namespace cuda
}  // namespace gis
}  // namespace zilliz
