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

namespace arctern {
namespace gis {
namespace cuda {

using ConstIter = ConstGpuContext::ConstIter;
namespace {
inline DEVICE_RUNNABLE double PolyShapeArea(ConstIter& iter) {
  constexpr int dimensions = 2;
  auto count = (int)*iter.metas++;
  double sum_area = 0.0;
  auto value2 = (double2*)iter.values;
  for (int point_index = 0; point_index + 1 < count; ++point_index) {
    auto l = value2[point_index];
    auto r = value2[point_index + 1];
    auto area = l.x * r.y - l.y * r.x;
    sum_area += area;
  }
  // shift pointer
  iter.values += dimensions * count;
  return fabs(sum_area / 2);
}

inline DEVICE_RUNNABLE double PolygonArea(ConstIter& iter) {
  auto polys = (int)*iter.metas++;
  if (polys == 0) {
    return 0;
  }
  auto sum_area = PolyShapeArea(iter);
  for (int poly = 1; poly < polys; ++poly) {
    sum_area -= PolyShapeArea(iter);
  }
  return sum_area;
}

inline DEVICE_RUNNABLE double MultiPolygonArea(ConstIter& iter) {
  auto count = (int)*iter.metas++;
  double sum_area = 0.0;
  for (int i = 0; i < count; ++i) {
    auto tag = (WkbTag)*iter.metas++;
    assert(tag.get_category() == WkbCategory::kPolygon);
    sum_area += PolygonArea(iter);
  }
  return sum_area;
}

__global__ void ST_AreaKernel(ConstGpuContext ctx, double* result) {
  auto index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < ctx.size) {
    auto tag = ctx.get_tag(index);
    // handle 2d case only for now
    assert(tag.get_space_type() == WkbSpaceType::XY);
    auto iter = ctx.get_iter(index);
    double final;
    switch (tag.get_category()) {
      case WkbCategory::kPolygon: {
        final = PolygonArea(iter);
        break;
      }
      case WkbCategory::kMultiPolygon: {
        final = MultiPolygonArea(iter);
        break;
      }
      default: {
        final = 0;
        iter = ctx.get_iter(index + 1);
        break;
      }
    }
    assert(iter.metas == ctx.get_meta_ptr(index + 1));
    assert(iter.values == ctx.get_value_ptr(index + 1));
    result[index] = final;
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
}  // namespace arctern
