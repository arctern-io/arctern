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
#include "gis/cuda/functor/st_length.h"
#include "gis/cuda/test_common/geometry_factory.h"

namespace arctern {
namespace gis {
namespace cuda {
TEST(FunctorLength, naive) {
  using std::string;
  using std::vector;
  vector<string> raw = {
      "LINESTRING(0 0, 0 1, 1 0, 1 1)",
  };
  vector<double> std_results = {
      1 + sqrt(2) + 1,
  };
  vector<double> results(raw.size());
  ASSERT_EQ(std_results.size(), results.size());

  auto geo = GeometryVectorFactory::CreateFromWkts(raw);

  cuda::ST_Length(geo, results.data());

  for (auto i = 0; i < raw.size(); ++i) {
    EXPECT_FLOAT_EQ(results[i], std_results[i]) << "at case " << i << std::endl;
  }
}
}  // namespace cuda
}  // namespace gis
}  // namespace arctern
