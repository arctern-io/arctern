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

#pragma once
#include <memory>
#include <set>
#include <utility>
#include <vector>

#include "utils/arrow_alias.h"

namespace arctern {
namespace gis {
namespace dispatch {

using ArrayType = arrow::BinaryArray;
class ArrowVectorAlignedSlicer {
 public:
  static constexpr size_t slice_limit = _ARROW_ARRAY_SIZE / 128;
  void Register(const std::vector<ArrayPtr>& vec) {
    size_t sum = 0;
    for (const auto& arr : vec) {
      sum += arr->length();
      arr_indexes_.insert(sum);
    }
  }

  void Format() {
    arr_indexes_.erase(0);
    size_t last_index = 0;
    for (auto index : arr_indexes_) {
      while (index - last_index > slice_limit) {
        last_index += slice_limit;
        arr_indexes_.insert(last_index);
      }
      last_index = index;
    }
  }

  std::vector<ArrayPtr> Slice(const std::vector<ArrayPtr>& raw_arrow_vector) const {
    if (arr_indexes_.size() == 1 && raw_arrow_vector.size() == 1) {
      // fast path
      return raw_arrow_vector;
    }
    std::vector<ArrayPtr> result;
    auto raw_arr_iter = raw_arrow_vector.cbegin();
    size_t last_arr_base = 0;
    size_t last_arr_offset = 0;
    for (auto arr_index : arr_indexes_) {
      while (arr_index - last_arr_base > (*raw_arr_iter)->length()) {
        assert(last_arr_offset == (*raw_arr_iter)->length());
        last_arr_base += last_arr_offset;
        last_arr_offset = 0;
        ++raw_arr_iter;
      }
      const auto& arr = *raw_arr_iter;
      auto slice_length = arr_index - last_arr_offset - last_arr_base;
      // zero-copy slice
      auto slice = arr->Slice(last_arr_offset, slice_length);
      last_arr_offset += slice_length;
      result.push_back(std::move(slice));
    }
    return result;
  }

  int size() { return arr_indexes_.size(); }

 private:
  std::set<size_t> arr_indexes_;
};

template <typename Func>
inline std::vector<ArrayPtr> AlignedExecuteBinary(Func functor,
                                                  std::vector<ArrayPtr> raw_input1,
                                                  std::vector<ArrayPtr> raw_input2) {
  dispatch::ArrowVectorAlignedSlicer slicer;
  slicer.Register(raw_input1);
  slicer.Register(raw_input2);
  slicer.Format();
  auto input1 = slicer.Slice(raw_input1);
  auto input2 = slicer.Slice(raw_input2);
  std::vector<ArrayPtr> result;
  for (int i = 0; i < slicer.size(); ++i) {
    assert(input1[i]->length() == input2[i]->length());
    result.push_back(functor(input1[i], input2[i]));
  }
  return result;
}

template <typename Func>
inline auto UnwarpBinary(Func functor) {
  auto unwarped_func = [functor](const ArrayPtr& left, const ArrayPtr& right) {
    auto result_vec =
        functor(std::vector<ArrayPtr>({left}), std::vector<ArrayPtr>({right}));
    assert(result_vec.size() == 1);
    return result_vec[0];
  };
  return unwarped_func;
}

}  // namespace dispatch
}  // namespace gis
}  // namespace arctern
