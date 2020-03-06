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

namespace zilliz {
namespace gis {
namespace cuda {

namespace {
using DataState = GeometryVector::DataState;

constexpr double inf = std::numeric_limits<double>::max();
struct MinMax {
  double min;
  double max;
  DEVICE_RUNNABLE void update(double x) {
    if (x < min) {
      min = x;
    }
    if (x > max) {
      max = x;
    }
  }
  // etol is error tolerance defined by implementation
  DEVICE_RUNNABLE void adjust_by_error(double etol) {
    if (min > max) {
      min = max = 0xcccc;
    }
    if (max == min) {
      max += etol;
      min -= etol;
    }
  }
};

__device__ inline OutputInfo GetInfoAndDataPerElement(const ConstGpuContext& input,
                                                      int index, GpuContext& results,
                                                      bool skip_write) {
  assert(input.get_tag(index).get_space_type() == WkbSpaceType::XY);
  if (!skip_write) {
    auto values_beg = input.get_value_ptr(index);
    auto values_end = input.get_value_ptr(index + 1);
    // generate bound from all values
    // instead of switching by tags
    MinMax final_x{+inf, -inf};
    MinMax final_y{+inf, -inf};

    for (auto iter = values_beg; iter < values_end; iter += 2) {
      final_x.update(iter[0]);
      final_y.update(iter[1]);
    }
    double etol = 1e-8;
    final_x.adjust_by_error(etol);
    final_y.adjust_by_error(etol);

    // fill meta
    auto meta_output = results.get_meta_ptr(index);
    meta_output[0] = 1;
    meta_output[1] = 5;

    // fill value
    auto value_output = results.get_value_ptr(index);
    value_output[0 * 2 + 0] = final_x.min;
    value_output[0 * 2 + 1] = final_y.min;

    value_output[1 * 2 + 0] = final_x.max;
    value_output[1 * 2 + 1] = final_y.min;

    value_output[2 * 2 + 0] = final_x.max;
    value_output[2 * 2 + 1] = final_y.max;

    value_output[3 * 2 + 0] = final_x.min;
    value_output[3 * 2 + 1] = final_y.max;

    value_output[4 * 2 + 0] = final_x.min;
    value_output[4 * 2 + 1] = final_y.min;
  }

  auto result_tag = WkbTag(WkbCategory::kPolygon, WkbSpaceType::XY);
  return OutputInfo{result_tag, 1 + 1, 2 * 5};
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
}  // namespace zilliz
