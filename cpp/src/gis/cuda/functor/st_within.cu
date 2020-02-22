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
inline DEVICE_RUNNABLE bool PointInPolygonImpl(double pnt_x, double pnt_y,
                                               const double* polygon,
                                               int num_poly_vertexex) {
  int winding_num = 0;
  double dx2 = poly_xs[num_poly_vertexes - 1] - pnt_x;
  double dy2 = poly_ys[num_poly_vertexes - 1] - pnt_y;
  for (int poly_idx = 0; poly_idx < num_poly_vertexes; ++poly_idx) {
    auto dx1 = dx2;
    auto dy1 = dy2;
    dx2 = poly_xs[poly_idx] - pnt_x;
    dy2 = poly_ys[poly_idx] - pnt_y;
    bool ref = dy1 < 0;
    if (ref != (dy2 < 0)) {
      if (isLeft(dx1, dy1, dx2, dy2) < 0 != ref) {
        winding_num += ref ? 1 : -1;
      }
    }
  }
  uint8_t ans = winding_num != 0;
  output[index] = ans;
}

inline DEVICE_RUNNABLE bool PointInPolygon(const GpuContext& point,
                                           const GpuContext& polygon, int index) {
  auto pv = point.get_value_ptr(index);
  auto polygon_meta = polygon.get_meta_ptr(index);
  int shape_size = (int)polygon_meta[0];
  for (int shape_index = 0; shape_index < shape_size; shape_index++) {
    int poly_size = (int)polygon_meta[1 + shape_index];
  }
  return false;
}
}  // namespace

//__global__ void ST_WithinKernel(GpuContext left, GpuContext right, double* result) {}
//}  // namespace
//
// void ST_Within(const GeometryVector& left_vec, const GeometryVector& right_vec,
//               double* host_results) {
//  assert(left_vec.size() == right_vec.size());
//  auto left_ctx_holder = left_vec.CreateReadGpuContext();
//  auto right_ctx_holder = right_vec.CreateReadGpuContext();
//  auto dev_result = GpuMakeUniqueArray<double>(left_vec.size());
//  {
//    auto config = GetKernelExecConfig(left_vec.size());
//    ST_WithinKernel<<<config.grid_dim, config.block_dim>>>(
//        *left_ctx_holder, *right_ctx_holder, dev_result.get());
//  }
//  GpuMemcpy(host_results, dev_result.get(), left_vec.size());
//}

}  // namespace cuda
}  // namespace gis
}  // namespace zilliz
