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
#include "gis/cuda/conversion/wkb_visitor.h"
namespace arctern {
namespace gis {
namespace cuda {

namespace internal {

struct WkbEncoderImpl {
  char* wkb_iter;
  const uint32_t* metas;
  const double* values;
  bool skip_write;
  int skipped_bytes;

  __device__ WkbEncoderImpl(char* wkb_iter, const uint32_t* metas, const double* values,
                            bool skip_write)
      : wkb_iter(wkb_iter),
        metas(metas),
        values(values),
        skip_write(skip_write),
        skipped_bytes(0) {}

 protected:
  __device__ void VisitValues(int dimensions, int points) {
    auto count = dimensions * points;
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
    InsertIntoWkb(m);
    metas += 1;
    return m;
  }

  // read from meta
  __device__ auto VisitMetaInt() { return VisitMeta<int>(); }

  // read from meta
  __device__ auto VisitMetaWkbTag() { return VisitMeta<WkbTag>(); }

  // read from constant
  __device__ void VisitByteOrder() {
    InsertIntoWkb(WkbByteOrder::kLittleEndian);
    skipped_bytes += sizeof(WkbTag);
  }

 public:
  __device__ void SetByteOrder(WkbByteOrder byte_order) {
    InsertIntoWkb(byte_order);
    skipped_bytes += sizeof(WkbTag);
  }

  __device__ void SetTag(WkbTag tag) {
    InsertIntoWkb(tag);
    skipped_bytes += sizeof(WkbTag);
  }

 private:
  template <typename T>
  __device__ void InsertIntoWkb(T data) {
    int len = sizeof(T);
    if (!skip_write) {
      memcpy(wkb_iter, &data, len);
    }
    wkb_iter += len;
  }
};

using WkbEncoder = WkbCodingVisitor<WkbEncoderImpl>;

__global__ static void CalcOffsets(ConstGpuContext input, WkbArrowContext output,
                                   int size) {
  auto index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < input.size) {
    auto tag = input.get_tag(index);
    auto metas = input.get_meta_ptr(index);
    auto values = input.get_value_ptr(index);
    auto wkb_iter = output.get_wkb_ptr(index);

    WkbEncoder encoder(wkb_iter, metas, values, true);
    encoder.SetByteOrder(WkbByteOrder::kLittleEndian);
    encoder.SetTag(tag);
    encoder.VisitBody(tag);

    int wkb_length = (int)(encoder.wkb_iter - wkb_iter);
    output.offsets[index] = wkb_length;
  }
}

// return: size of total data length in bytes
void ToArrowWkbFillOffsets(ConstGpuContext& input, WkbArrowContext& output,
                           int* value_length_ptr) {
  assert(input.size == output.size);
  auto size = input.size;
  assert(output.offsets);
  assert(!output.values);
  {
    auto config = GetKernelExecConfig(size);
    assert(cudaDeviceSynchronize() == cudaSuccess);
    CalcOffsets<<<config.grid_dim, config.block_dim>>>(input, output, size);
    ExclusiveScan(output.offsets, size + 1);

    assert(cudaDeviceSynchronize() == cudaSuccess);
  }
  if (value_length_ptr) {
    auto src = output.offsets + size;
    GpuMemcpy(value_length_ptr, src, 1);
  }
}

__global__ static void CalcValues(ConstGpuContext input, WkbArrowContext output) {
  auto index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < input.size) {
    auto tag = input.get_tag(index);
    auto metas = input.get_meta_ptr(index);
    auto values = input.get_value_ptr(index);
    auto wkb_iter = output.get_wkb_ptr(index);
    int std_wkb_length = (int)(output.get_wkb_ptr(index + 1) - output.get_wkb_ptr(index));
    WkbEncoder encoder(wkb_iter, metas, values, false);
    encoder.SetByteOrder(WkbByteOrder::kLittleEndian);
    encoder.SetTag(tag);
    encoder.VisitBody(tag);

    int wkb_length = (int)(encoder.wkb_iter - wkb_iter);
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
