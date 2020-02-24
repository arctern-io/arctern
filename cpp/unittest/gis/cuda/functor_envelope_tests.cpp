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

#include "gdal.h"
#include "gis/cuda/common/gis_definitions.h"
#include "gis/cuda/functor/st_area.h"
#include "gis/cuda/functor/st_envelope.h"
#include "test_utils/transforms.h"

namespace zilliz {
namespace gis {
namespace cuda {
TEST(FunctorEnvelope, naive) {
  using std::string;
  using std::vector;
  vector<string> raw = {"POLYGON((0 2, -2 -2, 2 -2, 0 2),(0 1, -1 -1, 1 -1, 0 1))",
                        "POLYGON((0 0, 1 0, 1 1, 1 0, 0 0))",
                        "LINGSTRING(3 4, 6 8, 7 2)"};
  vector<double> std_results = {
      2 * 4,
      1,
      (7 - 3) * (8 - 2),
  };
  vector<double> results(raw.size());

  ASSERT_EQ(results.size(), std_results.size());

  auto geo = GeometryVectorFactory::CreateFromWkts(raw);
  GeometryVector tmp_geo;
  cuda::ST_Envelope(geo, tmp_geo);
  cuda::ST_Area(tmp_geo, results.data());

  for (auto i = 0; i < raw.size(); ++i) {
    EXPECT_EQ(results[i], std_results[i]) << "at case " << i << std::endl;
  }
}
}  // namespace cuda
}  // namespace gis
}  // namespace zilliz
