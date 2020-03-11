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
#include "gis/cuda/functor/st_area.h"
#include "gis/cuda/test_common/test_common.h"

using std::vector;
namespace arctern {
namespace gis {
namespace cuda {

TEST(FunctorArea, naive) {
  ASSERT_TRUE(true);
  auto raw_data = hexstring_to_binary(
      "01030000000100000004000000000000000000084000000000000008400000000000000840000000"
      "00000010400000000000001040000000000000104000000000000010400000000000000840");

  int n = 3;
  vector<vector<char>> lists;
  for (int i = 0; i < n; ++i) {
    lists.push_back(raw_data);
  }

  auto gvec = GeometryVectorFactory::CreateFromWkbs(lists);
  vector<double> result(n);
  ST_Area(gvec, result.data());
  for (int i = 0; i < n; ++i) {
    auto std = 1;
    ASSERT_DOUBLE_EQ(result[i], std);
  }
}
}  // namespace cuda
}  // namespace gis
}  // namespace arctern
