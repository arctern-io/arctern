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

#include <gtest/gtest.h>

#include "gis/api.h"
#include "gis/dispatch/aligned_execute.h"
#include "gis/gdal/geometry_cases.h"
#include "utils/check_status.h"

namespace dispatch = arctern::gis::dispatch;
using arctern::ArrayPtr;
using arctern::DoubleArrayPtr;
using arctern::Int32ArrayPtr;
using arctern::gis::ST_Point;
using std::vector;

TEST(Dispatch, AlignedSlice) {
  auto gen = [](int m, int n) {
    vector<ArrayPtr> vec;
    for (int i = 0; i < m; ++i) {
      arrow::DoubleBuilder builder;
      for (int j = 0; j < n; ++j) {
        builder.Append(i * n + j);
      }
      DoubleArrayPtr arr;
      builder.Finish(&arr);
      vec.push_back(std::move(arr));
    }
    return vec;
  };
  auto vec_left = gen(97, 101);
  auto vec_right = gen(101, 97);
  auto full = gen(1, 97 * 101);
  auto res0 = ST_Point(vec_left, vec_right);
  auto res1 = ST_Point(vec_left, full);
  auto res2 = ST_Point(full, full);
  ASSERT_EQ(res2.size(), 1);
  auto ref = std::static_pointer_cast<arrow::BinaryArray>(res2[0]);
  int count = 0;
  for (const auto& ptr_ : res0) {
    auto ptr = std::static_pointer_cast<arrow::BinaryArray>(ptr_);
    for (int index = 0; index < ptr->length(); ++index) {
      ASSERT_EQ(ptr->GetView(index), ref->GetView(count)) << index << " " << count;
      ++count;
    }
  }
  count = 0;
  for (const auto& ptr_ : res1) {
    auto ptr = std::static_pointer_cast<arrow::BinaryArray>(ptr_);
    for (int index = 0; index < ptr->length(); ++index) {
      ASSERT_EQ(ptr->GetView(index), ref->GetView(count)) << index << " " << count;
      ++count;
    }
  }
}
