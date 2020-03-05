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

namespace zilliz {
namespace gis {
namespace cuda {
using internal::WkbArrowContext;

namespace {
void check(arrow::Status status) {
  if (!status.ok()) {
    throw std::runtime_error("action failed for " + status.message());
  }
}

template <typename T>
std::shared_ptr<arrow::Buffer> AllocArrowBufferAndCopy(int size, const T* dev_ptr) {
  std::shared_ptr<arrow::Buffer> buffer;
  auto len = sizeof(T) * size;
  auto status = arrow::AllocateBuffer(len, &buffer);
  check(status);
  GpuMemcpy((T*)buffer->mutable_data(), dev_ptr, size);
  return buffer;
}
}  // namespace

std::shared_ptr<arrow::Array> GeometryVectorToArrowWkb(const GeometryVector& geo_vec) {
  auto size = geo_vec.size();
  auto offsets = GpuMakeUniqueArray<int>(size + 1);
  auto input_holder = geo_vec.CreateReadGpuContext();

  WkbArrowContext arrow_ctx{nullptr, offsets.get(), size};
  int value_length;
  internal::ToArrowWkbFillOffsets(*input_holder, arrow_ctx, &value_length);
  auto values = GpuMakeUniqueArray<char>(value_length);
  arrow_ctx.values = values.get();
  internal::ToArrowWkbFillValues(*input_holder, arrow_ctx);

  auto offsets_buffer = AllocArrowBufferAndCopy(size + 1, offsets.get());
  auto values_buffer = AllocArrowBufferAndCopy(value_length, values.get());

  auto results = std::make_shared<arrow::BinaryArray>(size, offsets_buffer, values_buffer,
                                                      nullptr, 0);
  return results;
}

}  // namespace cuda
}  // namespace gis
}  // namespace zilliz
