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

#include <algorithm>
#include <cmath>
#include <unordered_set>
#include <utility>

#include "gis/cuda/common/gis_definitions.h"
#include "gis/cuda/common/gpu_memory.h"
#include "gis/cuda/functor/st_relate.h"
#include "gis/cuda/test_common/geometry_factory.h"
#include "gis/cuda/tools/de9im_matrix.h"
using de9im::Matrix;

namespace arctern {
namespace gis {
namespace cuda {
TEST(FunctorRelate, naive) {
  using std::string;
  using std::vector;
  // simple unittest, the complex one is put into host_within_tests.cpp
  using mat = Matrix;
  vector<std::tuple<string, string, Matrix, bool>> raw_data = {
      {"Point(0 0)", "Point(0 0)", mat("0FFFFFF1*"), false},
      {"Point(0 0)", "Point(0 0)", mat("0FFFFFF2*"), false},
      {"Point(0 0)", "Point(0 0)", mat("0FFFFFF0*"), false},
      {"Point(0 0)", "Point(0 0)", mat("0FFFFFF**"), true},
      {"Point(0 0)", "Point(0 0)", mat("FFFFFFFF*"), false},
      {"Point(0 0)", "Point(0 0)", mat("2FFFFFFF*"), false},
      {"Point(0 0)", "Point(0 0)", mat("1FFFFFFF*"), false},
      {"Point(0 0)", "Point(0 0)", mat("0FFFFFFF*"), true},

      {"Point(0 0)", "Point(0 1)", mat("FF0FFF0F*"), true},
      {"Point(0 0)", "LineString(0 -1, 0 1)", mat("0FFFFF10*"), true},
      {"Point(0 0)", "LineString(0 0, 0 1)", mat("F0FFFF10*"), true},
      {"Point(0 0)", "LineString(0 1, 3 0)", mat("FF0FFF10*"), true},
  };
  vector<string> left_vec;
  vector<string> right_vec;
  auto dog_less = [](Matrix a, Matrix b) { return a.get_payload() < b.get_payload(); };
  std::set<Matrix, decltype(dog_less)> matrix_collection(dog_less);
  vector<Matrix> matrices;
  vector<uint8_t> std_result;

  for (auto& tup : raw_data) {
    left_vec.push_back(std::get<0>(tup));
    right_vec.push_back(std::get<1>(tup));
    matrix_collection.insert(std::get<2>(tup));
    matrices.push_back(std::get<2>(tup));
    std_result.push_back(std::get<3>(tup));
  }

  auto left_geo = GeometryVectorFactory::CreateFromWkts(left_vec);
  auto right_geo = GeometryVectorFactory::CreateFromWkts(right_vec);
  auto size = left_geo.size();

  for (auto mat : matrix_collection) {
    vector<uint8_t> host_result(size);
    auto result = GpuMakeUniqueArray<uint8_t>(size);
    ST_Relate(left_geo, right_geo, mat, (bool*)result.get());
    GpuMemcpy(host_result.data(), result.get(), size);
    for (int i = 0; i < size; ++i) {
      if (mat == matrices[i]) {
        ASSERT_EQ(host_result[i], std_result[i]) << left_vec[i] << "\n"
                                                 << right_vec[i] << "\n"
                                                 << matrices[i];
      }
    }
  }
}
}  // namespace cuda
}  // namespace gis
}  // namespace arctern
