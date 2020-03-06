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
#include "gis/cuda/functor/st_within.h"
#include "gis/cuda/test_common/test_common.h"

namespace zilliz {
namespace gis {
namespace cuda {
TEST(FunctorWithin, naive) {
  using std::string;
  using std::vector;
  vector<string> left_raw = {
      "POINT(0 0)", "POINT(0 1.5)", "POINT(0 -1.5)", "POINT(0 2.5)", "POINT(0 -2.5)",
  };
  vector<string> right_raw = {
      "POLYGON((0 2, -2 -2, 2 -2),(0 1, -1 -1, 1 -1))",
      "POLYGON((0 2, -2 -2, 2 -2),(0 1, -1 -1, 1 -1))",
      "POLYGON((0 2, -2 -2, 2 -2),(0 1, -1 -1, 1 -1))",
      "POLYGON((0 2, -2 -2, 2 -2),(0 1, -1 -1, 1 -1))",
      "POLYGON((0 2, -2 -2, 2 -2),(0 1, -1 -1, 1 -1))",
  };
  vector<uint8_t> std_results = {false, true, true, false, false};
  vector<uint8_t> results(left_raw.size());

  ASSERT_EQ(left_raw.size(), right_raw.size());
  ASSERT_EQ(results.size(), std_results.size());

  auto left_geo = GeometryVectorFactory::CreateFromWkts(left_raw);
  auto right_geo = GeometryVectorFactory::CreateFromWkts(right_raw);

  cuda::ST_Within(left_geo, right_geo, (bool*)results.data());

  for (auto i = 0; i < left_raw.size(); ++i) {
    EXPECT_EQ(results[i], std_results[i]) << "at case " << i << std::endl;
  }
}
}  // namespace cuda
}  // namespace gis
}  // namespace zilliz
