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
#include "gis/cuda/functor/st_point.h"
#include "gis/cuda/test_common/geometry_factory.h"
using std::vector;
namespace arctern {
namespace gis {
namespace cuda {

TEST(FunctorPoint, naive) {
  vector<double> xs{1, 2, 3, 4, 5};
  vector<double> ys{0, 1, 2, 3, 4};
  GeometryVector left_points;
  GeometryVector right_points;
  ST_Point(xs.data(), ys.data(), (int)xs.size(), left_points);
  for (auto& x : xs) {
    x = -x;
  }
  for (auto& y : ys) {
    y = -y;
  }
  ST_Point(xs.data(), ys.data(), (int)xs.size(), right_points);
  vector<double> distance(xs.size());
  ST_Distance(left_points, right_points, distance.data());
  for (size_t i = 0; i < xs.size(); ++i) {
    auto std = (xs[i] * xs[i] + ys[i] * ys[i]) * 4;
    auto res = distance[i] * distance[i];
    ASSERT_DOUBLE_EQ(res, std);
  }
}
}  // namespace cuda
}  // namespace gis
}  // namespace arctern
