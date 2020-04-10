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

#include <thrust/functional.h>

#include <limits>

#include "gis/cuda/common/gpu_memory.h"
#include "gis/cuda/functor/geometry_output.h"
#include "gis/cuda/functor/st_envelope.h"
#include "gis/cuda/tools/bounding_box.h"

namespace arctern {
namespace gis {
namespace cuda {

namespace {
using DataState = GeometryVector::DataState;

__device__ inline OutputInfo GetInfoAndDataPerElement(const ConstGpuContext& input,
                                                      int index, GpuContext& results,
                                                      bool skip_write) {
  auto tag = input.get_tag(index);
  assert(tag.get_space_type() == WkbSpaceType::XY);
  auto values_beg = (const double2*)input.get_value_ptr(index);

  auto values_end = (const double2*)input.get_value_ptr(index + 1);

  auto meta_output = results.get_meta_ptr(index);
  auto value2_output = reinterpret_cast<double2*>(results.get_value_ptr(index));

  if (values_beg == values_end) {
    auto meta_size = input.get_meta_size(index);
    int value_size = input.get_value_size(index);
    assert(value_size == 0);
    // return MultiPoint Empty
    if (!skip_write) {
      // fill multipoint size
      auto metas = input.get_meta_ptr(index);
      for (int i = 0; i < meta_size; ++i) {
        meta_output[i] = metas[i];
      }
    }
    return OutputInfo{tag, meta_size, value_size};
  }

  BoundingBox bbox;

  for (auto iter = values_beg; iter < values_end; iter++) {
    bbox.Update(*iter);
  }
  auto xs = bbox.get_xs();
  auto ys = bbox.get_ys();

  if (xs.is_trivial() && ys.is_trivial()) {
    // just point
    if (!skip_write) {
      value2_output[0].x = xs.min;
      value2_output[0].y = ys.min;
    }
    return OutputInfo{WkbTypes::kPoint, 0, 2};
  }

  if (xs.is_trivial()) {
    if (!skip_write) {
      // points of line
      meta_output[0] = 2;
      value2_output[0] = {xs.min, ys.min};
      value2_output[1] = {xs.min, ys.max};
    }
    return OutputInfo{WkbTypes::kLineString, 1, 2 * 2};
  }

  if (ys.is_trivial()) {
    if (!skip_write) {
      // points of line
      meta_output[0] = 2;
      value2_output[0] = {xs.min, ys.min};
      value2_output[1] = {xs.max, ys.min};
    }
    return OutputInfo{WkbTypes::kLineString, 1, 2 * 2};
  }

  // then the normal cases
  if (!skip_write) {
    // count of polygon shapes
    meta_output[0] = 1;
    // points of polygon
    meta_output[1] = 5;
    // fill value
    value2_output[0] = {xs.min, ys.min};
    value2_output[1] = {xs.min, ys.max};
    value2_output[2] = {xs.max, ys.max};
    value2_output[3] = {xs.max, ys.min};
    value2_output[4] = {xs.min, ys.min};
  }
  return OutputInfo{WkbTypes::kPolygon, 2, 2 * 5};
}
}  // namespace

void ST_Envelope(const GeometryVector& input_vec, GeometryVector& results) {
  // copy xs, ys to gpu
  auto input_holder = input_vec.CreateReadGpuContext();
  auto size = input_vec.size();
  auto functor = [input = *input_holder] __device__(int index, GpuContext& results,
                                                    bool skip_write) {
    return GetInfoAndDataPerElement(input, index, results, skip_write);
  };  // NOLINT
  GeometryOutput(functor, size, results);
}

}  // namespace cuda
}  // namespace gis
}  // namespace arctern
