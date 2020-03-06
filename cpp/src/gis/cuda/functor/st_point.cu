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
#include "gis/cuda/functor/geometry_output.h"
#include "gis/cuda/functor/st_point.h"

namespace zilliz {
namespace gis {
namespace cuda {

namespace {
__device__ inline OutputInfo GetInfoAndDataPerElement(const double* xs, const double* ys,
                                                      int index, GpuContext& results,
                                                      bool skip_write) {
  if (!skip_write) {
    auto value = results.get_value_ptr(index);
    value[0] = xs[index];
    value[1] = ys[index];
  }
  return OutputInfo{WkbTag(WkbCategory::kPoint, WkbSpaceType::XY), 0, 2};
}
}  // namespace

void ST_Point(const double* cpu_xs, const double* cpu_ys, int size,
              GeometryVector& results) {
  // copy xs, ys to gpu
  auto xs = GpuMakeUniqueArrayAndCopy(cpu_xs, size);
  auto ys = GpuMakeUniqueArrayAndCopy(cpu_ys, size);
  {
    auto functor = [xs_ = xs.get(), ys_ = ys.get()] __device__(
                       int index, GpuContext& results, bool skip_write) {
      return GetInfoAndDataPerElement(xs_, ys_, index, results, skip_write);
    };  // NOLINT
    GeometryOutput(functor, size, results);
  }
}

}  // namespace cuda
}  // namespace gis
}  // namespace zilliz
