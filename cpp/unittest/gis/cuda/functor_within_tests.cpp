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

//
// Created by mike on 2/10/20.
//

#include <gtest/gtest.h>

#include <cmath>

#include "gis/cuda/common/gis_definitions.h"
#include "gis/cuda/functor/st_distance.h"
#include "gis/cuda/functor/st_point.h"
#include "test_utils/transforms.h"
#include "gdal.h"

namespace zilliz::gis::cuda {
TEST(FunctorWithin, naive) {
  using std::vector;
  using std::string;
  vector<string> left_raw = {
    "Point(0, 0)",
    "Point(0, 1.5)",
    "Point(0, -1.5)",
    "Point(0, 2.5)",
    "Point(0, -2.5)",
  };
  vector<string> right_raw = {
      "Polygon((0 2, -2 -2, 2 -2), (0 1, -1 -1, 1 -1))",
      "Polygon((0 2, -2 -2, 2 -2), (0 1, -1 -1, 1 -1))",
      "Polygon((0 2, -2 -2, 2 -2), (0 1, -1 -1, 1 -1))",
      "Polygon((0 2, -2 -2, 2 -2), (0 1, -1 -1, 1 -1))",
      "Polygon((0 2, -2 -2, 2 -2), (0 1, -1 -1, 1 -1))",
      "Polygon((0 2, -2 -2, 2 -2), (0 1, -1 -1, 1 -1))",
  };
  ASSERT_EQ(left_raw.size(), right_raw.size());
  GeometryVector left_vec;
  GeometryVector right_vec;
  left_vec.WkbDecodeInitalize();
  right_vec.WkbDecodeInitalize();
  for(auto i = 0; i < (int)left_raw.size(); ++i) {

  }
}
}  // namespace zilliz::gis::cuda
