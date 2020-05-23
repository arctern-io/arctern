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

// below is for cout << std::tuple, copied from stackoverflow
namespace aux {
template <std::size_t...>
struct seq {};
template <std::size_t N, std::size_t... Is>
struct gen_seq : gen_seq<N - 1, N - 1, Is...> {};

template <std::size_t... Is>
struct gen_seq<0, Is...> : seq<Is...> {};

template <class Ch, class Tr, class Tuple, std::size_t... Is>
void print_tuple(std::basic_ostream<Ch, Tr>& os, Tuple const& t, seq<Is...>) {
  using swallow = int[];
  (void)swallow{0, (void(os << (Is == 0 ? "" : ", ") << std::get<Is>(t)), 0)...};
}
}  // namespace aux

template <class Ch, class Tr, class... Args>
auto operator<<(std::basic_ostream<Ch, Tr>& os, std::tuple<Args...> const& t)
    -> std::basic_ostream<Ch, Tr>& {
  os << "(";
  aux::print_tuple(os, t, aux::gen_seq<sizeof...(Args)>());
  return os << ")";
}
// above is for cout << std::tuple, copied from stackoverflow

namespace arctern {
namespace gis {
namespace cuda {
TEST(FunctorRelate, naive) {
  using std::string;
  using std::vector;
  // simple unittest, the complex one is put into host_within_tests.cpp
  using mat = Matrix;
  vector<std::tuple<string, string, Matrix, bool>> raw_data = {
      {"LineString(0 0, 1 0, 1 1)", "LineString(0 0, 1 0, 1 1)", mat{"1FFF0FFF*"}, true},
      {"Point(1 1)", "LineString(1 1, 1 1)         ", mat{"0FFFFFFF*"}, true},
      {"Point(0 1)", "LineString(1 1, 1 1)         ", mat{"FF0FFF1F*"}, true},
      // standard answer from PostGis is F0FFFFFF*, but we use below for consistency
      {"Point(1 1)", "Polygon((1 1, 1 1, 1 1, 1 1))", mat{"F0FFFF21*"}, true},
      // Below is standard answer from PostGis
      {"Point(1 1)", "Polygon((1 1, 1 1, 1 1, 1 1))", mat{"F0FFFF21*"}, true},
      {"Point(0 1)", "Polygon((1 1, 1 1, 1 1, 1 1))", mat{"FF0FFF21*"}, true},

      {"LineString(1 1, 1 1)         ", "Point(1 1)", mat{"0FFFFFFF*"}, true},
      {"LineString(1 1, 1 1)         ", "Point(0 1)", mat{"FF1FFF0F*"}, true},
      {"Polygon((1 1, 1 1, 1 1, 1 1))", "Point(1 1)", mat{"FF20F1FF*"}, true},
      {"Polygon((1 1, 1 1, 1 1, 1 1))", "Point(0 1)", mat{"FF2FF10F*"}, true},

      {"Point(0 1)", "LINESTRING (0 0, 0 1, 1 2)", mat("0FFFFF10*"), true},
      {"LINESTRING (0 0, 1 2)", "LINESTRING (0 0, 1 2)", mat("1FFF0FFF*"), true},
      {"LINESTRING (0 0, 1 2)", "LINESTRING (0 0, 1 2, 4 2)", mat("1FF00F10*"), true},
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
      {"Point(0 0)", "Polygon((-1 0, 1 0, 0 1))", mat{"F0FFFF21*"}, true},
      {"Point(0 0.5)", "Polygon((-1 0, 1 0, 0 1))", mat{"0FFFFF21*"}, true},
      {"Point(0 100)", "Polygon((-1 0, 1 0, 0 1))", mat{"FF0FFF21*"}, true},

      {"Polygon((-1 0, 1 0, 0 1))", "Point(0 0)", mat{"FF20F1FF*"}, true},
      {"Polygon((-1 0, 1 0, 0 1))", "Point(0 0.5)", mat{"0F2FF1FF*"}, true},
      {"Polygon((-1 0, 1 0, 0 1))", "Point(0 100)", mat{"FF2FF10F*"}, true},
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

  vector<Matrix> host_result(size);
  auto result = GpuMakeUniqueArray<Matrix>(size);
  auto lch = left_geo.CreateReadGpuContext();
  auto rch = right_geo.CreateReadGpuContext();
  ST_Relate(*lch, *rch, result.get());
  GpuMemcpy(host_result.data(), result.get(), size);
  for (int index = 0; index < raw_data.size(); ++index) {
    auto ref = raw_data[index];
    auto ref_mat = std::get<2>(ref);
    auto ref_tf = std::get<3>(ref);
    auto v = host_result[index].IsMatchTo(ref_mat);
    EXPECT_EQ(v, ref_tf) << host_result[index] << " vs " << ref;
  }
}
}  // namespace cuda
}  // namespace gis
}  // namespace arctern
