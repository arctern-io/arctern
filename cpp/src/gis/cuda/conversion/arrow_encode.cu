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

#include <exception>

#include "gis/cuda/common/gpu_memory.h"
#include "gis/cuda/conversion/conversions.h"

namespace arctern {
namespace gis {
namespace cuda {

namespace internal {
__global__ static void CalcOffsets(ConstGpuContext input, WkbArrowContext output,
                                   int size) {
  auto index = threadIdx.x + blockDim.x * blockIdx.x;
  if (index < size) {
    auto common_offset = (int)((sizeof(WkbByteOrder) + sizeof(WkbTag)) * index);
    auto value_offset = input.value_offsets[index] * sizeof(double);
    auto meta_offset = input.meta_offsets[index] * sizeof(int);
    auto wkb_length = common_offset + value_offset + meta_offset;
    output.offsets[index] = wkb_length;
  }
}

// return: size of total data length in bytes
void ToArrowWkbFillOffsets(ConstGpuContext& input, WkbArrowContext& output,
                           int* value_length_ptr) {
  assert(input.size == output.size);
  assert(output.offsets);
  assert(!output.values);
  {
    auto offset_size = input.size + 1;
    auto config = GetKernelExecConfig(offset_size);
    assert(cudaDeviceSynchronize() == cudaSuccess);
    CalcOffsets<<<config.grid_dim, config.block_dim>>>(input, output, offset_size);
    assert(cudaDeviceSynchronize() == cudaSuccess);
  }
  if (value_length_ptr) {
    auto src = output.offsets + input.size;
    GpuMemcpy(value_length_ptr, src, 1);
  }
}

struct WkbEncoder {
  const uint32_t* metas;
  const double* values;
  char* wkb_iter;
  static constexpr bool skip_write = false;

 protected:
  __device__ void VisitValues(int demensions, int points) {
    auto count = demensions * points;
    auto bytes = count * sizeof(double);
    if (!skip_write) {
      memcpy(wkb_iter, values, bytes);
    }
    wkb_iter += bytes;
    values += count;
  }

  template <typename T>
  __device__ T VisitMeta() {
    static_assert(sizeof(T) == sizeof(*metas), "size of T must match meta");
    auto m = static_cast<T>(*metas);
    if (!skip_write) {
      InsertIntoWkb(m);
    }
    metas += 1;
    return m;
  }

 public:
  template <typename T>
  __device__ void InsertIntoWkb(T data) {
    int len = sizeof(T);
    memcpy(wkb_iter, &data, len);
    wkb_iter += len;
  }

  __device__ void VisitPoint(int demensions) { VisitValues(demensions, 1); }

  __device__ void VisitLineString(int demensions) {
    auto size = VisitMeta<int>();
    VisitValues(demensions, size);
  }

  __device__ void VisitPolygon(int demensions) {
    auto polys = VisitMeta<int>();
    for (int i = 0; i < polys; ++i) {
      VisitLineString(demensions);
    }
  }
};

__global__ static void CalcValues(ConstGpuContext input, WkbArrowContext output) {
  auto index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < input.size) {
    auto tag = input.get_tag(index);
    assert(tag.get_space_type() == WkbSpaceType::XY);
    int dimensions = 2;
    auto metas = input.get_meta_ptr(index);
    auto values = input.get_value_ptr(index);
    auto wkb_iter = output.get_wkb_ptr(index);

    WkbEncoder encoder{metas, values, wkb_iter};
    encoder.InsertIntoWkb(WkbByteOrder::kLittleEndian);
    encoder.InsertIntoWkb(tag);

    switch (tag.get_category()) {
      case WkbCategory::kPoint: {
        encoder.VisitPoint(dimensions);
        break;
      }
      case WkbCategory::kLineString: {
        encoder.VisitLineString(dimensions);
        break;
      }
      case WkbCategory::kPolygon: {
        encoder.VisitPolygon(dimensions);
        break;
      }
      default: {
        assert(false);
        break;
      }
    }
    auto wkb_length = encoder.wkb_iter - wkb_iter;
    auto std_wkb_length = output.get_wkb_ptr(index + 1) - output.get_wkb_ptr(index);
    assert(std_wkb_length == wkb_length);
  }
}

void ToArrowWkbFillValues(ConstGpuContext& input, WkbArrowContext& output) {
  assert(input.size == output.size);
  assert(output.offsets);
  assert(output.values);
  auto config = GetKernelExecConfig(input.size);
  CalcValues<<<config.grid_dim, config.block_dim>>>(input, output);
}

}  // namespace internal

}  // namespace cuda
}  // namespace gis
}  // namespace arctern
