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
      auto right_size = right_iter.read_meta<int>();
      auto right_points = right_iter.read_value_ptr<double2>(right_size);
      result = PointRelateToLineString(left_point, right_size, right_points);
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

  auto left_size = left_iter.read_meta<int>();
  auto left_points = left_iter.read_value_ptr<double2>(left_size);
  Matrix result;
  switch (right_tag.get_category()) {
    case WkbCategory::kPoint: {
      auto right_point = right_iter.read_value<double2>();
      auto mat = PointRelateToLineString(right_point, left_size, left_points);
      result = mat.get_transpose();
      break;
    }
    case WkbCategory::kLineString: {
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
    case WkbCategory::kMultiLineString: {
      // TODO
      result = LineStringRelateOp(left_iter, right.get_tag(index), right_iter, buffer);
      break;
    }
    default: {
      result = de9im::INVALID_MATRIX;
      left_iter = left.get_iter(index + 1);
      break;
    }
  }
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
                              Matrix ref_matrix, bool* output_matrixes) {
  auto index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < left.size) {
    auto matrix = RelateOp(left, right, index);
    output_matrixes[index] = matrix.IsMatchTo(ref_matrix);
  }
}

void ST_Relate(const GeometryVector& left_vec, const GeometryVector& right_vec,
               de9im::Matrix ref_matrix, bool* host_results) {
  assert(left_vec.size() == right_vec.size());
  auto size = left_vec.size();
  auto left_holder = left_vec.CreateReadGpuContext();
  auto right_holder = right_vec.CreateReadGpuContext();
  //  RelateOp();
  auto config = GetKernelExecConfig(size);
  auto results = GpuMakeUniqueArray<bool>(size);

  ST_RelateImpl<<<config.grid_dim, config.block_dim>>>(*left_holder, *right_holder,
                                                       ref_matrix, results.get());
  GpuMemcpy(host_results, results.get(), size);
}

}  // namespace cuda
}  // namespace gis
}  // namespace arctern
