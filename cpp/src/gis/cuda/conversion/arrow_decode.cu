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
#include "gis/cuda/conversion/wkb_visitor.h"
#include "gis/cuda/functor/geometry_output.h"

namespace arctern {
namespace gis {
namespace cuda {

namespace {
struct WkbDecoderImpl {
  const char* wkb_iter;
  uint32_t* metas;
  double* values;
  bool skip_write;
  int skipped_bytes;
  __device__ WkbDecoderImpl(const char* wkb_iter, uint32_t* metas, double* values,
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
      memcpy(values, wkb_iter, bytes);
    }
    wkb_iter += bytes;
    values += count;
  }

  template <typename T>
  __device__ T VisitMeta() {
    static_assert(sizeof(T) == sizeof(*metas), "size of T must match meta");
    auto m = FetchFromWkb<uint32_t>();
    if (!skip_write) {
      *metas = m;
    }
    metas += 1;
    return static_cast<T>(m);
  }

  // write into meta
  __device__ auto VisitMetaInt() { return VisitMeta<int>(); }

  // write into meta
  __device__ auto VisitMetaWkbTag() { return VisitMeta<WkbTag>(); }

  __device__ void VisitByteOrder() {
    auto byte_order = FetchFromWkb<WkbByteOrder>();
    ++skipped_bytes;
    assert(byte_order == WkbByteOrder::kLittleEndian);
  }

 public:
  __device__ WkbByteOrder GetByteOrder() {
    auto byte_order = FetchFromWkb<WkbByteOrder>();
    skipped_bytes += sizeof(WkbTag);
    return byte_order;
  }

  __device__ WkbTag GetTag() {
    auto tag = FetchFromWkb<WkbTag>();
    skipped_bytes += sizeof(WkbTag);
    return tag;
  }

 private:
  template <typename T>
  __device__ T FetchFromWkb() {
    T tmp;
    int len = sizeof(T);
    memcpy(&tmp, wkb_iter, len);
    wkb_iter += len;
    return tmp;
  }
};

using WkbDecoder = WkbCodingVisitor<WkbDecoderImpl>;

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

  WkbDecoder decoder(wkb_iter, metas, values, skip_write);

  auto byte_order = decoder.GetByteOrder();
  assert(byte_order == WkbByteOrder::kLittleEndian);
  auto tag = decoder.GetTag();
  decoder.VisitBody(tag);

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
}  // namespace arctern
