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

#include "gis/cuda/common/gpu_memory.h"
#include "gis/cuda/functor/st_point.h"

namespace zilliz {
namespace gis {
namespace cuda {

namespace {
using DataState = GeometryVector::DataState;

struct OutputInfo {
  WkbTag tag;
  int meta_size;
  int value_size;
};

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
    if(min > max){
      min = max = 0xcccc;
    }
    if(max == min) {
      max += etol;
      min -= etol;
    }
  }
};

__device__ inline OutputInfo GetInfoAndDataPerElement(const GpuContext& input, int index,
                                                      GpuContext& results,
                                                      bool skip_write = false) {
  assert(input.get_tag(index).get_group() == WkbGroup::None);
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

  auto result_tag = WkbTag(WkbCategory::Polygon, WkbGroup::None);
  return OutputInfo{result_tag, 1 + 1, 2 * 5};
}

__global__ void FillInfoKernel(const GpuContext input, GpuContext results) {
  assert(results.data_state == DataState::FlatOffset_EmptyInfo);
  auto index = threadIdx.x + blockIdx.x * blockDim.x;
  assert(input.size == results.size);
  if (index < input.size) {
    auto out_info = GetInfoAndDataPerElement(input, index, results, true);
    printf("%d", index);
    results.tags[index] = out_info.tag;
    results.meta_offsets[index] = out_info.meta_size;
    results.value_offsets[index] = out_info.value_size;
  }
}

DEVICE_RUNNABLE inline void AssertInfo(OutputInfo info, const GpuContext& ctx,
                                       int index) {
  assert(info.tag.data == ctx.get_tag(index).data);
  assert(info.meta_size == ctx.meta_offsets[index + 1] - ctx.meta_offsets[index]);
  assert(info.value_size == ctx.value_offsets[index + 1] - ctx.value_offsets[index]);
}

static __global__ void FillDataKernel(const GpuContext input, GpuContext results) {
  assert(results.data_state == DataState::PrefixSumOffset_EmptyData);
  auto index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < results.size) {
    auto out_info = GetInfoAndDataPerElement(input, index, results);
    AssertInfo(out_info, results, index);
  }
}
}  // namespace

void ST_Envelope(const GeometryVector& input_vec, GeometryVector& results) {
  // copy xs, ys to gpu
  auto input_holder = input_vec.CreateReadGpuContext();
  auto size = input_vec.size();

  // STEP 1: Initialize vector with size of elements
  results.OutputInitialize(size);

  // STEP 2: Create gpu context according to the vector for cuda
  // where tags and offsets fields are uninitailized
  auto results_holder = results.OutputCreateGpuContext();
  {
    // STEP 3: Fill info(tags and offsets) to gpu_ctx using CUDA Kernels
    // where offsets[0, n) is filled with size of each element
    auto config = GetKernelExecConfig(size);
    FillInfoKernel<<<config.grid_dim, config.block_dim>>>(*input_holder, *results_holder);
    results_holder->data_state = DataState::FlatOffset_FullInfo;
  }

  // STEP 4: Exclusive scan offsets[0, n+1), where offsets[n] = 0
  // then copy info(tags and scanned offsets) back to GeometryVector
  // and alloc cpu & gpu memory for next steps
  results.OutputEvolveWith(*results_holder);

  {
    // STEP 5: Fill data(metas and values) to gpu_ctx using CUDA Kernels
    auto config = GetKernelExecConfig(size);
    FillDataKernel<<<config.grid_dim, config.block_dim>>>(*input_holder, *results_holder);
    results_holder->data_state = DataState::PrefixSumOffset_FullData;
  }
  // STEP 6: Copy data(metas and values) back to GeometryVector
  results.OutputFinalizeWith(*results_holder);
}

}  // namespace cuda
}  // namespace gis
}  // namespace zilliz
