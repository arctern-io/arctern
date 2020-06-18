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
#include "gis/cuda/functor/st_relate.h"
#include "gis/cuda/tools/relation.h"

namespace arctern {
namespace gis {
namespace cuda {

using ConstIter = ConstGpuContext::ConstIter;
using de9im::Matrix;

DEVICE_RUNNABLE Matrix PointRelateOp(ConstIter& left_iter, WkbTag right_tag,
                                     ConstIter& right_iter) {
  //  auto right_tag = right.get_tag(index);
  assert(right_tag.get_space_type() == WkbSpaceType::XY);
  auto left_point = left_iter.read_value<double2>();

  //  auto right_iter = right.get_iter(index);
  Matrix result;
  switch (right_tag.get_category()) {
    case WkbCategory::kPoint: {
      auto right_point = right_iter.read_value<double2>();
      auto is_eq = IsEqual(left_point, right_point);
      result = is_eq ? Matrix("0FFFFFFF*") : Matrix("FF0FFF0F*");
      break;
    }
    case WkbCategory::kLineString: {
      result = PointRelateToLineString(left_point, right_iter);
      break;
    }
    case WkbCategory::kPolygon: {
      result = PointRelateToPolygon(left_point, right_iter);
      break;
    }
    default: {
      assert(false);
      result = de9im::INVALID_MATRIX;
      break;
    }
  }
  return result;
}

DEVICE_RUNNABLE Matrix LineStringRelateOp(ConstIter& left_iter, WkbTag right_tag,
                                          ConstIter& right_iter, KernelBuffer& buffer) {
  assert(right_tag.get_space_type() == WkbSpaceType::XY);

  Matrix result;
  switch (right_tag.get_category()) {
    case WkbCategory::kPoint: {
      auto right_point = right_iter.read_value<double2>();
      auto mat = PointRelateToLineString(right_point, left_iter);
      result = mat.get_transpose();
      break;
    }
    case WkbCategory::kLineString: {
      auto left_size = left_iter.read_meta<int>();
      auto left_points = left_iter.read_value_ptr<double2>(left_size);
      auto right_size = right_iter.read_meta<int>();
      auto right_points = right_iter.read_value_ptr<double2>(right_size);
      result = LineStringRelateToLineString(left_size, left_points, right_size,
                                            right_points, buffer);
      break;
    }
    default: {
      assert(false);
      result = de9im::INVALID_MATRIX;
      break;
    }
  }
  return result;
}

DEVICE_RUNNABLE Matrix RelateOp(const ConstGpuContext& left, const ConstGpuContext& right,
                                int index) {
  auto left_tag = left.get_tag(index);
  assert(left_tag.get_space_type() == WkbSpaceType::XY);
  Matrix result;
  auto left_iter = left.get_iter(index);
  auto right_iter = right.get_iter(index);
  KernelBuffer buffer;

  switch (left_tag.get_category()) {
    case WkbCategory::kPoint: {
      result = PointRelateOp(left_iter, right.get_tag(index), right_iter);
      break;
    }
    case WkbCategory::kLineString: {
      result = LineStringRelateOp(left_iter, right.get_tag(index), right_iter, buffer);
      break;
    }
    case WkbCategory::kPolygon: {
      assert(right.get_tag(index).get_category() == WkbCategory::kPoint);
      auto point = right_iter.read_value<double2>();
      result = PointRelateToPolygon(point, left_iter).get_transpose();
      break;
    }
    default: {
      assert(false);
      result = de9im::INVALID_MATRIX;
      left_iter = left.get_iter(index + 1);
      break;
    }
  }
  assert(right_iter.values == right.get_value_ptr(index + 1));
  assert(right_iter.metas == right.get_meta_ptr(index + 1));
  assert(left_iter.values == left.get_value_ptr(index + 1));
  assert(left_iter.metas == left.get_meta_ptr(index + 1));

  return result;
}

// enum class RelateOpType {
//  kInvalid,
//  kEquals,
//  kDisjoint,
//  kTouched,
//  kContains,
//  kCovers,
//  kIntersects,
//  kWithin,
//  kCoveredBy,
//  kCrosses,
//  kOverlaps,
//};
//

__global__ void ST_RelateImpl(ConstGpuContext left, ConstGpuContext right,
                              Matrix* output_matrixes) {
  auto index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < left.size) {
    auto matrix = RelateOp(left, right, index);
    output_matrixes[index] = matrix;
  }
}

void ST_Relate(const ConstGpuContext& left_ctx, const ConstGpuContext& right_ctx,
               de9im::Matrix* dev_results) {
  assert(left_ctx.size == right_ctx.size);
  auto size = left_ctx.size;
  auto config = GetKernelExecConfig(size);
  ST_RelateImpl<<<config.grid_dim, config.block_dim>>>(left_ctx, right_ctx, dev_results);
}

}  // namespace cuda
}  // namespace gis
}  // namespace arctern
