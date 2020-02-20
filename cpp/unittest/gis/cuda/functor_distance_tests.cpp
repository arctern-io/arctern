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
#include "test_utils/transforms.h"

namespace zilliz::gis::cuda {
TEST(FunctorDistance, naive) {
  ASSERT_TRUE(true);
  // TODO use gdal to convert better good

  // POINT(3 1), copy from WKB WKT convertor
  //    uint8_t data_left[] = {0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00,
  //                           0x00, 0x00, 0x00, 0x00, 0x08, 0x40, 0x00, 0x00, 0x00,
  //                           0x00, 0x00, 0x00, 0xf0, 0x3f};
  auto vec_left = hexstring_to_binary("01010000000000000000000840000000000000f03f");
  auto data_left = vec_left.data();
  char data[1 + 4 + 16];
  int num = 5;

  uint8_t byte_order = 0x1;
  memcpy(data + 0, &byte_order, sizeof(byte_order));
  uint32_t point_tag = 1;
  memcpy(data + 1, &point_tag, sizeof(point_tag));

  GeometryVector gvec_left;
  GeometryVector gvec_right;
  gvec_left.WkbDecodeInitalize();
  gvec_right.WkbDecodeInitalize();

  for (int i = 0; i < num; ++i) {
    double x = i;
    double y = i + 1;
    static_assert(sizeof(x) == 8, "wtf");
    memcpy(data + 5, &x, sizeof(x));
    memcpy(data + 5 + 8, &y, sizeof(y));

    gvec_left.WkbDecodeAppend(data_left);
    gvec_right.WkbDecodeAppend(data);
  }
  gvec_left.WkbDecodeFinalize();
  gvec_right.WkbDecodeFinalize();
  vector<double> result(5);
  ST_Distance(gvec_left, gvec_right, result.data());
  for (int i = 0; i < num; ++i) {
    auto std = sqrt(pow(i - 3, 2) + pow(i + 1 - 1, 2));
    ASSERT_DOUBLE_EQ(result[i], std);
  }
}
}  // namespace zilliz::gis::cuda
