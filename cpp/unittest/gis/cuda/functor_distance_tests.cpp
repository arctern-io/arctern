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

#include <cmath>

#include "gis/cuda/common/gis_definitions.h"
#include "gis/cuda/functor/st_distance.h"
#include "gis/cuda/test_common/test_common.h"

using std::vector;
namespace zilliz {
namespace gis {
namespace cuda {

TEST(FunctorDistance, naive) {
  ASSERT_TRUE(true);
  // TODO use gdal to convert better good

  // POINT(3 1), copy from WKB WKT convertor
  //    uint8_t data_left[] = {0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00,
  //                           0x00, 0x00, 0x00, 0x00, 0x08, 0x40, 0x00, 0x00, 0x00,
  //                           0x00, 0x00, 0x00, 0xf0, 0x3f};
  auto vec_left = hexstring_to_binary("01010000000000000000000840000000000000f03f");
  vector<char> vec_right(1 + 4 + 16);
  //  char data[1 + 4 + 16];
  int num = 5;

  uint8_t byte_order = 0x1;
  memcpy(vec_right.data() + 0, &byte_order, sizeof(byte_order));
  uint32_t point_tag = 1;
  memcpy(vec_right.data() + 1, &point_tag, sizeof(point_tag));

  vector<vector<char>> lists_left;
  vector<vector<char>> lists_right;

  for (int i = 0; i < num; ++i) {
    double x = i;
    double y = i + 1;
    static_assert(sizeof(x) == 8, "wtf");
    memcpy(vec_right.data() + 5, &x, sizeof(x));
    memcpy(vec_right.data() + 5 + 8, &y, sizeof(y));

    lists_left.push_back(vec_left);
    lists_right.push_back(vec_right);
  }
  vector<double> result(5);
  auto gvec_left = GeometryVectorFactory::CreateFromWkbs(lists_left);
  auto gvec_right = GeometryVectorFactory::CreateFromWkbs(lists_right);
  ST_Distance(gvec_left, gvec_right, result.data());

  for (int i = 0; i < num; ++i) {
    auto std = sqrt(pow(i - 3, 2) + pow(i + 1 - 1, 2));
    ASSERT_DOUBLE_EQ(result[i], std);
  }
}

}  // namespace cuda
}  // namespace gis
}  // namespace zilliz
