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

#include "utils/arrow_utils.h"

namespace arctern {

namespace {

class ChunkedArrayAligner {
 public:
  explicit ChunkedArrayAligner(size_t max_rows_limit) : max_rows_limit_(max_rows_limit) {}

  void Register(const ChunkedArrayPtr& chunked_array) {
    size_t sum = 0;
    ArrayPtr ptr;
    for (int i = 0; i < chunked_array->num_chunks(); ++i) {
      auto chunk = chunked_array->chunk(i);
      sum += chunk->length();
      arr_indexes_.insert(sum);
    }
  }

  void Align() {
    arr_indexes_.erase(0);
    size_t last_index = 0;
    for (auto index : arr_indexes_) {
      while (index - last_index > max_rows_limit_) {
        last_index += max_rows_limit_;
        arr_indexes_.insert(last_index);
      }
      last_index = index;
    }
  }

  ChunkedArrayPtr Slice(const ChunkedArrayPtr& chunked_array) const {
    if (arr_indexes_.size() == 1 && chunked_array->num_chunks() == 1) {
      // fast path
      return chunked_array;
    }
    std::vector<ArrayPtr> result;
    auto raw_chunk_id = 0;
    size_t last_arr_base = 0;
    size_t last_arr_offset = 0;
    for (auto arr_index : arr_indexes_) {
      while (arr_index - last_arr_base > chunked_array->chunk(raw_chunk_id)->length()) {
        // assert(last_arr_offset == (*raw_arr_iter)->length());
        last_arr_base += last_arr_offset;
        last_arr_offset = 0;
        ++raw_chunk_id;
      }
      auto chunk = chunked_array->chunk(raw_chunk_id);
      auto slice_length = arr_index - last_arr_offset - last_arr_base;
      // zero-copy slice
      auto slice = chunk->Slice(last_arr_offset, slice_length);
      last_arr_offset += slice_length;
      result.push_back(std::move(slice));
    }
    return std::make_shared<arrow::ChunkedArray>(result, chunked_array->type());
  }
  int size() { return arr_indexes_.size(); }

 private:
  std::set<size_t> arr_indexes_;
  size_t max_rows_limit_;
};
}  // namespace

// by default, the max_rows_limit is up to 8 million
std::vector<ChunkedArrayPtr> AlignChunkedArray(
    const std::vector<ChunkedArrayPtr>& chunked_arrays, size_t max_rows_limit) {
  assert(max_rows_limit >= 1);
  ChunkedArrayAligner aligner(max_rows_limit);
  for (auto& chunked_array : chunked_arrays) {
    aligner.Register(chunked_array);
  }
  aligner.Align();
  std::vector<ChunkedArrayPtr> results;
  for (auto& chunked_array : chunked_arrays) {
    results.emplace_back(aligner.Slice(chunked_array));
  }
  return results;
}
}  // namespace arctern
