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

#include "gis/cuda/common/gis_definitions.h"
#include "gis/cuda/conversion/conversions.h"
#include "gis/cuda/functor/geometry_output.h"

namespace zilliz {
namespace gis {
namespace cuda {

namespace {
struct WkbDecoder {
  const char* wkb_iter;
  uint32_t* metas;
  double* values;
  bool skip_write;

 private:
  __device__ void WkbToValues(int demensions, int points) {
    auto count = demensions * points;
    auto bytes = count * sizeof(double);
    if (!skip_write) {
      memcpy(values, wkb_iter, bytes);
    }
    wkb_iter += bytes;
    values += count;
  }
  template <typename T>
  __device__ T WkbToMeta() {
    static_assert(sizeof(T) == sizeof(*metas), "size of T must match meta");
    auto m = FetchFromWkb<uint32_t>();
    if (!skip_write) {
      *metas = m;
    }
    metas += 1;
    return static_cast<T>(m);
  }

 public:
  template <typename T>
  __device__ T FetchFromWkb() {
    T tmp;
    int len = sizeof(T);
    memcpy(&tmp, wkb_iter, len);
    wkb_iter += len;
    return tmp;
  }

  __device__ void DecodePoint(int demensions) { WkbToValues(demensions, 1); }

  __device__ void DecodeLineString(int demensions) {
    auto size = WkbToMeta<int>();
    WkbToValues(demensions, size);
  }

  __device__ void DecodePolygon(int demensions) {
    auto polys = WkbToMeta<int>();
    for (int i = 0; i < polys; ++i) {
      DecodeLineString(demensions);
    }
  }
};

using internal::WkbArrowContext;
__device__ inline OutputInfo GetInfoAndDataPerElement(const WkbArrowContext& input,
                                                      int index, GpuContext& results,
                                                      bool skip_write) {
  auto wkb_iter = input.get_wkb_ptr(index);

  uint32_t* metas = nullptr;
  double* values = nullptr;
  if (!skip_write) {
    metas = results.get_meta_ptr(index);
    values = results.get_value_ptr(index);
  }

  WkbDecoder decoder{wkb_iter, metas, values, skip_write};

  auto byte_order = decoder.FetchFromWkb<WkbByteOrder>();
  assert(byte_order == WkbByteOrder::kLittleEndian);
  auto tag = decoder.FetchFromWkb<WkbTag>();
  assert(tag.get_space_type() == WkbSpaceType::XY);
  constexpr auto demensions = 2;
  switch (tag.get_category()) {
    case WkbCategory::kPoint: {
      decoder.DecodePoint(demensions);
      break;
    }
    case WkbCategory::kLineString: {
      decoder.DecodeLineString(demensions);
      break;
    }
    case WkbCategory::kPolygon: {
      decoder.DecodePolygon(demensions);
      break;
    }
    default: {
      assert(false);
      break;
    }
  }

  int meta_size = (int)(decoder.metas - metas);
  int value_size = (int)(decoder.values - values);
  auto wkb_len = meta_size * sizeof(int) + value_size * sizeof(double) + sizeof(WkbTag) +
                 sizeof(WkbByteOrder);
  assert(wkb_len == input.get_wkb_ptr(index + 1) - input.get_wkb_ptr(index));
  return OutputInfo{tag, meta_size, value_size};
}
}  // namespace

namespace internal {
GeometryVector ArrowWkbToGeometryVectorImpl(const WkbArrowContext& input) {
  GeometryVector results;
  int size = input.size;
  // TODO(dog): add hanlder for nulls
  assert(input.null_counts() == 0);
  auto functor = [input] __device__(int index, GpuContext& results,
                                    bool skip_write) -> OutputInfo {
    return GetInfoAndDataPerElement(input, index, results, skip_write);
  };
  GeometryOutput(functor, size, results);
  return results;
}

}  // namespace internal
}  // namespace cuda
}  // namespace gis
}  // namespace zilliz
