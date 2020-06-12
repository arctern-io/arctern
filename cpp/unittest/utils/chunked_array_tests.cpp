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

#include <chrono>
#include <memory>
#include <random>

#include "utils/arrow_utils.h"
using arctern::ArrayPtr;
using arctern::ChunkedArrayPtr;
using arrow::Array;
using arrow::ArrayFromVector;
using arrow::ChunkedArray;
using arrow::ChunkedArrayFromJSON;
using arrow::Int16Type;
using arrow::Int32Type;
using std::string;
using std::vector;

void verify(ChunkedArrayPtr ptr, ChunkedArrayPtr raw, const vector<int>& expected_size) {
  auto is_equal = ptr->Equals(raw);
  ASSERT_TRUE(is_equal);
  ASSERT_EQ(ptr->num_chunks(), expected_size.size());
  for (int i = 0; i < expected_size.size(); ++i) {
    ASSERT_EQ(ptr->chunk(i)->length(), expected_size[i]);
  }
}

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

  vector<vector<int16_t>> a_data{{1, 2, 3}, {4, 5}, {6, 7, 8, 9}};
  vector<vector<int32_t>> b_data{{41, 42}, {43, 44, 45}, {46, 47, 48}, {49}};
  ChunkedArrayPtr ra, rb;
  arrow::ChunkedArrayFromVector<arrow::Int16Type>(a_data, &ra);
  arrow::ChunkedArrayFromVector<arrow::Int32Type>(b_data, &rb);

  verify(ra, a, {3, 2, 4});
  verify(rb, b, {2, 3, 3, 1});

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

TEST(ChunkedArray, Empty) {
  vector<vector<string>> a_data = {};
  vector<vector<string>> b_data = {{}, {}};
  ChunkedArrayPtr a, b;
  // arrow::ChunkedArrayFromVector<arrow::BinaryType>(a_data, &a);
  a = std::make_shared<arrow::ChunkedArray>(arrow::ArrayVector{}, arrow::binary());
  arrow::ChunkedArrayFromVector<arrow::StringType>(b_data, &b);
  auto rechunked = arctern::AlignChunkedArray({a, b});
  verify(rechunked[0], a, {});
  verify(rechunked[1], b, {});
}

TEST(ChunkedArray, StringAndBinary) {
  vector<vector<string>> a_data = {{std::string("\0\0", 2)},
                                   {std::string("\0\x23\x33\12", 4)}};
  vector<vector<string>> b_data = {{"Point(0 0)", "Polygon Empty"}};
  ChunkedArrayPtr a, b;
  arrow::ChunkedArrayFromVector<arrow::BinaryType>(a_data, &a);
  arrow::ChunkedArrayFromVector<arrow::StringType>(b_data, &b);

  auto ptr2 = std::static_pointer_cast<arrow::BinaryArray>(a->chunk(1));
  EXPECT_EQ(ptr2->GetView(0).size(), 4);

  auto rechunked = arctern::AlignChunkedArray({a, b});
  verify(rechunked[0], a, {1, 1});
  verify(rechunked[1], b, {1, 1});
}

TEST(ChunkedArray, Large) {
  // 2k trunks
  std::default_random_engine control_e;
  std::default_random_engine data_e;
  arrow::ArrayVector va, vb;

  auto get_trunk = [&data_e](int size) {
    arrow::Int32Builder builder;
    for (int i = 0; i < size; ++i) {
      auto data = data_e();
      if (data == 0) {
        builder.AppendNull();
      } else {
        builder.Append(data);
      }
    }
    ArrayPtr ptr;
    builder.Finish(&ptr);
    return ptr;
  };

  size_t numA = 0;
  size_t numB = 0;
  for (int trunk = 0; trunk < 1024; ++trunk) {
    auto sizeA = control_e() % 1024;
    auto sizeB = control_e() % 1024;
    numA += sizeA;
    numB += sizeB;
    if (trunk == 1024 - 1) {
      auto total_num = std::max(numA, numB);
      sizeA += total_num - numA;
      sizeB += total_num - numB;
    }
    va.push_back(get_trunk(sizeA));
    vb.push_back(get_trunk(sizeB));
  }
  auto ta = std::make_shared<ChunkedArray>(va);
  auto tb = std::make_shared<ChunkedArray>(vb);
  EXPECT_EQ(ta->length(), tb->length());

  auto t0 = std::chrono::high_resolution_clock::now();
  auto retrunked = arctern::AlignChunkedArray({ta, tb}, 512);
  auto t1 = std::chrono::high_resolution_clock::now();
  auto t = std::chrono::duration<double>(t1 - t0);
  std::cout << "time elapse of Large Chunk is " << t.count() << std::endl;
  EXPECT_LE(t.count(), 0.1);

  ASSERT_TRUE(ta->Equals(retrunked[0]));
  ASSERT_TRUE(tb->Equals(retrunked[1]));
}
