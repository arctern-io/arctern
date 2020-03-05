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

#include "gis/cuda/common/function_wrapper.h"
#include "gis/cuda/common/gpu_memory.h"
#include "gis/cuda/conversion/conversions.h"
#include "gis/cuda/conversion/conversions.impl.h"

namespace zilliz {
namespace gis {
namespace cuda {

std::shared_ptr<arrow::Array> GeometryVectorToArrowWkb(const GeometryVector&);

using internal::WkbArrowContext;

GeometryVector ArrowWkbToGeometryVector(const std::shared_ptr<arrow::Array>& array_wkb) {
  auto wkb = std::static_pointer_cast<arrow::BinaryArray>(array_wkb);
  auto size = (int)wkb->length();
  auto binary_bytes = wkb->value_offset(size);
  auto data =
      GpuMakeUniqueArrayAndCopy((const char*)wkb->value_data()->data(), binary_bytes);
  auto offsets = GpuMakeUniqueArrayAndCopy(wkb->raw_value_offsets(), size + 1);
  auto geo_vec = internal::ArrowWkbToGeometryVectorImpl(
      WkbArrowContext{data.get(), offsets.get(), size});
  return geo_vec;
}

}  // namespace cuda
}  // namespace gis
}  // namespace zilliz
