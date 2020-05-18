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
#include "gis/cuda/tools/relation.h"

namespace arctern {
namespace gis {
namespace cuda {
namespace {



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
      result[tid] = PointInPolygon(left, right, tid).is_in;
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
}  // namespace arctern
