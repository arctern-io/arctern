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
#include "gis/cuda/test_common/geometry_factory.h"

using std::string;
using std::vector;
namespace arctern {
namespace gis {
namespace cuda {

TEST(FunctorArea, naive) {
  ASSERT_TRUE(true);
  vector<std::pair<string, double>> datas = {
      {"Point Empty", 0},
      {"Point (0 1)", 0},
      {"LineString (0 1, 1 1)", 0},
      {"Polygon Empty", 0},
      {"Polygon ((0 0))", 0},
      {"Polygon ((0 0, 1 1, 1 0, 0 0))", 0.5},
      {"Polygon ((0 0, 3 0, 3 3, 0 3, 0 0), (1 1, 2 1, 2 2, 1 2, 1 1))", 8},
      {"MultiPolygon Empty", 0},
      {"MultiPolygon (((0 0, 1 1, 1 0, 0 0)))", 0.5},
      {"MultiPolygon (((0 0, 1.5 1, 1.5 0, 0 0)), "
       "((0 0, 3 0, 3 3, 0 3, 0 0), (1 1, 2 1, 2 2, 1 2, 1 1)))",
       8 + 1.5 * 0.5},
  };
  vector<string> input;
  vector<double> std_results;
  for (auto& data : datas) {
    input.push_back(data.first);
    std_results.push_back(data.second);
  }
  auto gvec = GeometryVectorFactory::CreateFromWkts(input);
  vector<double> results(input.size());
  ST_Area(gvec, results.data());
  for (int i = 0; i < input.size(); ++i) {
    EXPECT_DOUBLE_EQ(results[i], std_results[i]) << datas[i].first << std::endl;
  }
}  // namespace cuda
}  // namespace cuda
}  // namespace gis
}  // namespace arctern
