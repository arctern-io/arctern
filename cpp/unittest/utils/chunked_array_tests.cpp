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

#include <arrow/testing/gtest_util.h>
#include <gtest/gtest.h>

#include <memory>

#include "utils/arrow_utils.h"
using arctern::ArrayPtr;
using arctern::ChunkedArrayPtr;
using arrow::Array;
using arrow::ArrayFromVector;
using arrow::ChunkedArray;
using arrow::ChunkedArrayFromJSON;
using arrow::Int16Type;
using arrow::Int32Type;
using std::vector;

TEST(ChunkedArray, Naive) {
  std::shared_ptr<Array> expected;
  std::shared_ptr<Array> a1, a2, a3, b1, b2, b3, b4;
  ArrayFromVector<Int16Type, int16_t>({1, 2, 3}, &a1);
  ArrayFromVector<Int16Type, int16_t>({4, 5}, &a2);
  ArrayFromVector<Int16Type, int16_t>({6, 7, 8, 9}, &a3);

  ArrayFromVector<Int32Type, int32_t>({41, 42}, &b1);
  ArrayFromVector<Int32Type, int32_t>({43, 44, 45}, &b2);
  ArrayFromVector<Int32Type, int32_t>({46, 47, 48}, &b3);
  ArrayFromVector<Int32Type, int32_t>({49}, &b4);

  auto a = std::make_shared<ChunkedArray>(vector<ArrayPtr>{a1, a2, a3});
  auto b = std::make_shared<ChunkedArray>(vector<ArrayPtr>{b1, b2, b3, b4});

  auto verify = [](ChunkedArrayPtr ptr, ChunkedArrayPtr raw,
                   const vector<int>& expected_size) {
    auto is_equal = ptr->Equals(raw);
    ASSERT_TRUE(is_equal);
    ASSERT_EQ(ptr->num_chunks(), expected_size.size());
    for (int i = 0; i < expected_size.size(); ++i) {
      ASSERT_EQ(ptr->chunk(i)->length(), expected_size[i]);
    }
  };

  {
    auto rechunked = arctern::AlignChunkedArray({a, b}, 10);
    ASSERT_EQ(rechunked.size(), 2);
    std::vector<int> expected_size{2, 1, 2, 3, 1};
    verify(rechunked[0], a, expected_size);
    verify(rechunked[1], b, expected_size);
  }
  {
    auto rechunked = arctern::AlignChunkedArray({a, b}, 2);
    ASSERT_EQ(rechunked.size(), 2);
    std::vector<int> expected_size{2, 1, 2, 2, 1, 1};
    verify(rechunked[0], a, expected_size);
    verify(rechunked[1], b, expected_size);
  }
}
