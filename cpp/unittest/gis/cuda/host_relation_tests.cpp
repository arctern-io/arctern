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

#include "gis/cuda/functor/st_relate.h"
#include "gis/cuda/test_common/testcase.h"
#include "gis/cuda/tools/relation.h"
#include "gis/test_common/transforms.h"
using std::vector;
namespace cu = arctern::gis::cuda;
using cu::Matrix;

TEST(Relation, IsPointInLine) {
  struct Data {
    double2 point_raw;
    double2 line_beg;
    double2 line_end;
    bool std_ans;
  };
  double2 point00{0, 0};
  double2 point01{0, 1};
  double2 point11{1, 1};
  double2 point22{2, 2};
  vector<Data> datas{
      {point00, point11, point22, false},
      {point11, point00, point22, true},
      {point01, point00, point22, false},
      {point00, point00, point22, true},
  };
  for (auto data : datas) {
    auto ans = cu::IsPointInLine(data.point_raw, data.line_beg, data.line_end);
    ASSERT_EQ(ans, data.std_ans);
  }
}

TEST(Relation, IsPointInLineString) {
  struct Data {
    double2 point;
    vector<double> lines;
    int std_count;
  };
  using vd = vector<double>;
  vector<Data> datas{
      {double2{0, 0}, {}, 0},
      {double2{0, 0}, vd{0, 0, 1, 1}, 1},
      {double2{0, 0}, vd{-1, -1, 1, 1, 2, -1}, 1},
      {double2{0, 0}, vd{-1, -1, 0, 0, 2, 0}, 2},
      {double2{0, 0}, vd{0, 0, 0, 0}, 1},
      {double2{0, 0}, vd{0, 0, 1, 1, 2, 0, 0, 0}, 2},
      {double2{2, 2}, vd{0, 0, 1, 1, 2, 0, 0, 0}, 0},
  };
  for (double x_off : {0, 1, 2}) {
    for (double y_off : {0, 10, 20}) {
      for (const auto& data : datas) {
        auto point = data.point;
        point.x += x_off;
        point.y += y_off;
        auto lines = data.lines;
        assert(lines.size() % 2 == 0);
        int size = (int)lines.size() / 2;
        for (int i = 0; i < size; ++i) {
          lines[i * 2 + 0] += x_off;
          lines[i * 2 + 1] += y_off;
        }
        auto ptr = reinterpret_cast<const double2*>(lines.data());
        auto count = cu::PointOnLineString(point, size, ptr);
        ASSERT_EQ(count, data.std_count);
      }
    }
  }
}

TEST(Relation, LineRelateToLineString) {
  struct Data {
    std::vector<double> line;  // sized 4
    vector<double> lnstr;
    cu::LineRelationResult std_result;
  };
  thrust::complex<double> control_scale_factor;
  auto scale = [&control_scale_factor](double* ptr) {
    thrust::complex<double> raw(ptr[0], ptr[1]);
    auto tmp = control_scale_factor * raw;
    ptr[0] = tmp.real();
    ptr[1] = tmp.imag();
  };
  (void)scale;

  using vd = vector<double>;
  using lrr = cu::LineRelationResult;
  // TODO(dog): use CSV format
  vector<Data> datas{
      {vd{0, 0, 0, 3}, vd{0, 0, 0, 1, 1, 1, 0, 2, 0, 3}, lrr{1, false, -100}},
      {vd{0, 0, 0, 3}, vd{0, 0, 0, 1, 1, 1, 0, 2, 0, 3}, lrr{1, false, -100}},
      {vd{0, 0, 0, 1}, vd{0, 1, 0, 2}, lrr{0, false, 1}},
      {vd{0, 0, 0, 1}, vd{0, 0, 2, 1, -2, 0}, lrr{0, false, 2}},
      {vd{0, 0, 0, 1}, vd{0, 0, 2, 3}, lrr{0, false, 1}},
      {vd{0, 0, 0, 1}, vd{-2, 0, 2, 0}, lrr{0, false, 1}},
      {vd{0, 0, 0, 2}, vd{0, 1, 2, 3}, lrr{0, false, 1}},
      {vd{0, 0, 0, 1}, vd{-2, 0, 2, 1}, lrr{0, false, 1}},
      {vd{0, 0, 0, 1}, vd{0, 1, 2, 2}, lrr{0, false, 1}},
      {vd{0, 0, 0, 1}, vd{0, 3, 2, 2}, lrr{-1, false, 0}},
      {vd{0, 0, 0, 1}, vd{0, 0, 0, 1}, lrr{1, true, -100}},
      {vd{0, 0, 0, 3}, vd{0, 0, 0, 1, 0, 2, 0, 3}, lrr{1, true, -100}},
      {vd{0, 0, 0, 3}, vd{0, 0, 0, 2, 0, 1, 0, 3}, lrr{1, true, -100}},
      {vd{0, 0, 0, 3}, vd{0, 0, 0, 1, 1, 1, 0, 2, 0, 3, 4, 4, 0, 2, 0, 1},
       lrr{1, true, -100}},
      {vd{0, 0, 0, 3}, vd{0, -100, 0, -99, 3, 3, 0, -1, 0, 1, 0, 2, 0, 4},
       lrr{1, true, -100}},
  };
  vector<thrust::complex<double>> scale_factors;
  for (double i : {0, 1, -1}) {
    for (double j : {0, 1, -1}) {
      auto x = cu::to_complex({i, j});
      if (x == 0) {
        continue;
      }
      scale_factors.emplace_back(x);
    }
  }
  for (auto scale_factor : scale_factors) {
    for (auto index = 0; index < datas.size(); ++index) {
      control_scale_factor = scale_factor;
      auto data = datas[index];
      scale(data.line.data());
      scale(data.line.data() + 2);
      auto size = data.lnstr.size();
      assert(size % 2 == 0);
      size /= 2;
      for (int i = 0; i < size; ++i) {
        scale(data.lnstr.data() + i * 2);
      }
      cu::KernelBuffer buffer;
      auto result = cu::LineOnLineString((const double2*)data.line.data(), size,
                                         (const double2*)data.lnstr.data(), buffer);
      auto ref = data.std_result;
      ASSERT_EQ(result.CC, ref.CC) << index;
      ASSERT_EQ(result.is_coveredby, ref.is_coveredby) << index;
      if (ref.cross_count != -100) {
        ASSERT_EQ(result.cross_count, ref.cross_count) << index;
      }
    }
  }
}

inline std::ostream& operator<<(std::ostream& out, const std::vector<double2>& vec) {
  for (auto v : vec) {
    out << "(" << v.x << "," << v.y << ")"
        << ", ";
  }
  return out;
}

TEST(Relation, LineStringRelateToLineString) {
  struct Data {
    vector<double2> left;   // left linestring
    vector<double2> right;  // right linestring
    Matrix std_matrix;
  };
  thrust::complex<double> control_scale_factor;
  auto scale = [&control_scale_factor](double2& v) {
    auto raw = cu::to_complex(v);
    auto tmp = control_scale_factor * raw;
    v = double2{tmp.real(), tmp.imag()};
  };
  (void)scale;

  auto csv_table = ProjectedTableFromCsv(
      datasource::relation_csv, {"left_linestring", "right_linestring", "matrix"});
  vector<Data> datas;
  for (auto& line : csv_table) {
    Data data;
    data.left = ToDouble2Array(line[0]);
    data.right = ToDouble2Array(line[1]);
    data.std_matrix = Matrix(line[2].c_str());
    datas.push_back(std::move(data));
  }

  vector<thrust::complex<double>> scale_factors;
  for (double i : {0, 1, -1}) {
    for (double j : {0, 1, -1}) {
      auto x = cu::to_complex({i, j});
      if (x == 0) {
        continue;
      }
      scale_factors.emplace_back(x);
    }
  }

  for (auto scale_factor : scale_factors) {
    for (auto index = 0; index < datas.size(); ++index) {
      cu::KernelBuffer buffer;
      control_scale_factor = scale_factor;
      auto data = datas[index];
      for (auto& v : data.left) {
        scale(v);
      }
      for (auto& v : data.right) {
        scale(v);
      }

      auto matrix =
          cu::LineStringRelateToLineString(data.left.size(), data.left.data(),
                                           data.right.size(), data.right.data(), buffer);
      ASSERT_EQ(matrix, data.std_matrix) << index << std::endl
                                         << data.left << std::endl
                                         << data.right << std::endl;

      std::swap(data.left, data.right);
      data.std_matrix = data.std_matrix.get_transpose();
      matrix =
          cu::LineStringRelateToLineString(data.left.size(), data.left.data(),
                                           data.right.size(), data.right.data(), buffer);
      ASSERT_EQ(matrix, data.std_matrix) << index << std::endl
                                         << data.left << std::endl
                                         << data.right << std::endl;
    }
  }
}
