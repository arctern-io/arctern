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

#include <string>

#include "gis/cuda/common/gis_definitions.h"
#include "gis/cuda/conversion/conversions.h"
#include "gis/cuda/test_common/test_common.h"
#include "gis/cuda/wkb/wkb_transforms.h"

namespace arctern {
namespace gis {
namespace cuda {
using std::vector;

TEST(Transform, Naive) {
  auto std_data = hexstring_to_binary(
      "01030000000100000004000000000000000000084000000000000008400000000000000840000000"
      "00000010400000000000001040000000000000104000000000000010400000000000000840");
  auto output_data = Wkt2Wkb("POLYGON((3 3,3 4,4 4,4 3))");
  ASSERT_TRUE(std_data.size() == output_data.size());
  for (int i = 0; i < std_data.size(); ++i) {
    EXPECT_EQ(std_data[i], output_data[i]) << "at " << i;
  }
  ASSERT_TRUE(true);
  auto pnt = Wkt2Wkb("POINT(1 1)");
  ASSERT_TRUE(pnt.size() == 1 + 4 + 16);
}

TEST(Transform, Arrow) {
  vector<std::string> wkt_vec = {
      "Point(1 2)",
      "Point(2 3)",
      "LineString(3 4,4 5)",
      "LineString(3 4,4 5)",
      "POLYGON((0 2, -2 -2, 2 -2, 0 2),(0 1, -1 -1, 1 -1, 0 1))",
      "POLYGON((0 2, -2 -2, 2 -2, 0 2),(0 1, -1 -1, 1 -1, 0 1))",
      "MultiPoint(1 2, 2 3)",
      "MultiPoint(2 3, 3 4)",
      "MultiLineString((3 4,4 5), (-3 -4,-4 -5))",
      "MultiLineString((-3 4,-4 -5), (3 4,4 5))",
      "MultiPOLYGON(((0 2, -2 -2, 2 -2, 0 2),(0 1, -1 -1, 1 -1, 0 1)), "
      "((0 2, -2 -2, 2 -2, 0 2),(0 1, -1 -1, 1 -1, 0 1)))",
      "MultiPOLYGON(((10 12, -12 -12, 22 -22, 10 12),(30 31, -31 -31, 31 -31, 30 31)), "
      "((0 2, -2 -2, 2 -2, 0 2),(0 1, -1 -1, 1 -1, 0 1)))",
  };
  auto std_arrow_wkb = WktsToArrowWkb(wkt_vec);
  auto geo_vec = ArrowWkbToGeometryVector(std_arrow_wkb);
  auto arrow_wkb = GeometryVectorToArrowWkb(geo_vec);
  auto is_equal = arrow_wkb->Equals(std_arrow_wkb);
  ASSERT_TRUE(is_equal);
  //  auto ret_arrow = Ge
}

}  // namespace cuda
}  // namespace gis
}  // namespace arctern
