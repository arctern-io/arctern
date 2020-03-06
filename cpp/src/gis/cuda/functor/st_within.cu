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
#include <thrust/pair.h>

#include <cmath>

#include "gis/cuda/common/common.h"
#include "gis/cuda/common/gpu_memory.h"
#include "gis/cuda/functor/st_within.h"

namespace zilliz {
namespace gis {
namespace cuda {
namespace {

inline DEVICE_RUNNABLE double IsLeft(double x1, double y1, double x2, double y2) {
  return x1 * y2 - x2 * y1;
}

inline DEVICE_RUNNABLE double GetX(const double* polygon, int index) {
  return polygon[2 * index];
}
inline DEVICE_RUNNABLE double GetY(const double* polygon, int index) {
  return polygon[2 * index + 1];
}

struct Point {
  double x;
  double y;
};

struct Iter {
  const uint32_t* metas;
  const double* values;
};

inline DEVICE_RUNNABLE bool PointInSimplePolygonHelper(Point point, const double* polygon,
                                                       int size) {
  int winding_num = 0;
  double dx2 = GetX(polygon, size - 1) - point.x;
  double dy2 = GetY(polygon, size - 1) - point.y;
  for (int index = 0; index < size; ++index) {
    auto dx1 = dx2;
    auto dy1 = dy2;
    dx2 = GetX(polygon, index) - point.x;
    dy2 = GetY(polygon, index) - point.y;
    bool ref = dy1 < 0;
    if (ref != (dy2 < 0)) {
      if (IsLeft(dx1, dy1, dx2, dy2) < 0 != ref) {
        winding_num += ref ? 1 : -1;
      }
    }
  }
  return winding_num != 0;
}

inline DEVICE_RUNNABLE bool PointInPolygonImpl(Point point, Iter& polygon_iter) {
  int shape_size = (int)*polygon_iter.metas++;
  bool final = false;
  // offsets of value for polygons
  for (int shape_index = 0; shape_index < shape_size; shape_index++) {
    int vertex_size = (int)*polygon_iter.metas++;
    auto is_in = PointInSimplePolygonHelper(point, polygon_iter.values, vertex_size);
    polygon_iter.values += vertex_size * 2;
    if (shape_index == 0) {
      final = is_in;
    } else {
      final = final && !is_in;
    }
  }
  return final;
}

// inline DEVICE_RUNNABLE bool PointInMultiPolygon(Point point, Iter& iter) {
//  int poly_size = (int)metas[0];
//  bool final = false;
//  int meta_offsets = 0;
//  for(int index = 0; index < poly_size; ++index) {
//    int polygon_size = metas[1 + offsets];
//  }
//  return final;
//}

inline DEVICE_RUNNABLE bool PointInPolygon(ConstGpuContext& point,
                                           ConstGpuContext& polygon, int index) {
  auto pv = point.get_value_ptr(index);
  auto iter = Iter{polygon.get_meta_ptr(index), polygon.get_value_ptr(index)};
  auto result = PointInPolygonImpl(Point{pv[0], pv[1]}, iter);
  assert(iter.metas == polygon.get_meta_ptr(index + 1));
  assert(iter.values == polygon.get_value_ptr(index + 1));
  return result;
}

__global__ void ST_WithinKernel(ConstGpuContext left, ConstGpuContext right,
                                bool* result) {
  auto tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < left.size) {
    auto left_tag = left.get_tag(tid);
    auto right_tag = right.get_tag(tid);
    // handle 2d case only for now
    assert(left_tag.get_space_type() == WkbSpaceType::XY);
    assert(right_tag.get_space_type() == WkbSpaceType::XY);
    // handle point to point case only
    if (left_tag.get_category() == WkbCategory::kPoint &&
        right_tag.get_category() == WkbCategory::kPolygon) {
      result[tid] = PointInPolygon(left, right, tid);
    } else {
      result[tid] = false;
    }
  }
}
}  // namespace

void ST_Within(const GeometryVector& left_vec, const GeometryVector& right_vec,
               bool* host_results) {
  assert(left_vec.size() == right_vec.size());
  auto left_ctx_holder = left_vec.CreateReadGpuContext();
  auto right_ctx_holder = right_vec.CreateReadGpuContext();
  auto dev_result = GpuMakeUniqueArray<bool>(left_vec.size());
  {
    auto config = GetKernelExecConfig(left_vec.size());
    ST_WithinKernel<<<config.grid_dim, config.block_dim>>>(
        *left_ctx_holder, *right_ctx_holder, dev_result.get());
  }
  GpuMemcpy(host_results, dev_result.get(), left_vec.size());
}

}  // namespace cuda
}  // namespace gis
}  // namespace zilliz
