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

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>

#include "gis/cuda/common/gis_definitions.h"
#include "wkb_tag.h"

namespace zilliz {
namespace gis {
namespace cuda {

void GeometryVector::WkbDecodeInitalize() {
  clear();
  data_state_ = DataState::Appending;
}

void GeometryVector::clear() {
  tags_.clear();
  metas_.clear();
  values_.clear();
  meta_offsets_.clear();
  value_offsets_.clear();
  data_state_ = DataState::Invalid;
}
void GeometryVector::WkbDecodeFinalize() {
  assert(data_state_ == DataState::Appending);
  assert(meta_offsets_.size() == tags_.size());
  assert(value_offsets_.size() == tags_.size());

  meta_offsets_.push_back(0);
  value_offsets_.push_back(0);

  auto prefix_sum = [](vector<int>& vec) {
    // TODO: use exclusive_scan instead
    int sum = 0;
    for (auto& x : vec) {
      auto old_sum = sum;
      sum += x;
      x = old_sum;
    }
  };

  prefix_sum(meta_offsets_);
  prefix_sum(value_offsets_);

  data_state_ = DataState::PrefixSumOffset_FullData;
}

template <typename T>
static inline void fill(size_t count, const char*& iter, void* dest) {
  auto len = count * sizeof(T);
  memcpy(dest, iter, len);
  iter += len;
}

template <typename T>
static inline T fetch(const char*& iter) {
  T res;
  fill<T>(1, iter, &res);
  return res;
}

void GeometryVector::WkbDecodeAppend(const char* raw_bin) {
  // decode a single polygon and append to vector
  assert(data_state_ == DataState::Appending);
  const char* stream_iter = raw_bin;
  auto byte_order = fetch<WkbByteOrder>(stream_iter);
  assert(byte_order == WkbByteOrder::LittleEndian);
  auto tag = fetch<WkbTag>(stream_iter);

  auto extend_values_from_stream = [&](int dimensions, size_t points) {
    auto count = dimensions * points;
    int value_base = static_cast<int>(values_.size());
    values_.resize(values_.size() + count);
    fill<double>(count, stream_iter, values_.data() + value_base);
  };

  // deal with 2D cases for now
  assert(tag.get_group() == WkbGroup::None);
  auto dimensions = 2;
  tags_.push_back(tag);
  switch (tag.get_category()) {
    case WkbCategory::Point: {
      // metas_.do_nothing()
      extend_values_from_stream(dimensions, 1);

      meta_offsets_.push_back(0);
      value_offsets_.push_back(dimensions);
      break;
    }
    case WkbCategory::LineString: {
      auto points = fetch<uint32_t>(stream_iter);
      extend_values_from_stream(dimensions, points);

      metas_.push_back(points);
      value_offsets_.push_back(1);
      break;
    }
    case WkbCategory::Polygon: {
      int total_values = 0;
      // most case 1
      auto count_sub_poly = fetch<int>(stream_iter);
      metas_.push_back(count_sub_poly);
      for (auto sub_poly = 0; sub_poly < count_sub_poly; sub_poly++) {
        auto points = fetch<uint32_t>(stream_iter);
        extend_values_from_stream(dimensions, points);
        total_values += dimensions * points;
        metas_.push_back(points);
      }
      meta_offsets_.push_back(1 + count_sub_poly);
      value_offsets_.push_back(total_values);
      break;
    }
    default: {
      assert(false);
      break;
    }
  }
}

}  // namespace cuda
}  // namespace gis
}  // namespace zilliz
