/*
 * Copyright (C) 2019-2020 Zilliz. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <arrow/render_api.h>
#include <gtest/gtest.h>
#include <ogr_api.h>
#include <ogr_geometry.h>
#include <utils/check_status.h>

#include <chrono>
#include <fstream>
#include <random>

#include "arrow/gis_api.h"
#include "gis/test_common/transforms.h"
#include "gis/gdal/format_conversion.h"

using std::string;
using std::vector;

TEST(IndexedWithin, Naive) {
  // param1: wkt string
  std::string wkt11 = "POINT (10 10)";
  std::string wkt12 = "POINT (30 10)";
  std::string wkt13 = "POINT (10 30)";
  std::string wkt14 = "POINT (30 30)";
  std::string wkt15 = "POINT (20 20)";
  arrow::StringBuilder string_builder11;
  auto status = string_builder11.Append(wkt11);
  status = string_builder11.Append(wkt12);
  status = string_builder11.Append(wkt13);
  status = string_builder11.Append(wkt14);
  status = string_builder11.Append(wkt15);
  std::shared_ptr<arrow::StringArray> string_array11;
  status = string_builder11.Finish(&string_array11);
  auto wkb_point11 = arctern::gis::gdal::WktToWkb(string_array11);

  std::string wkt21 = "POINT (60 60)";
  std::string wkt22 = "POINT (40 40)";
  std::string wkt23 = "POINT (50 20)";
  std::string wkt24 = "POINT (30 40)";
  std::string wkt25 = "POINT (15 15)";
  arrow::StringBuilder string_builder12;
  status = string_builder12.Append(wkt21);
  status = string_builder12.Append(wkt22);
  status = string_builder12.Append(wkt23);
  status = string_builder12.Append(wkt24);
  status = string_builder12.Append(wkt25);
  std::shared_ptr<arrow::StringArray> string_array12;
  status = string_builder12.Finish(&string_array12);
  auto wkb_point12 = arctern::gis::gdal::WktToWkb(string_array12);

  std::vector<std::shared_ptr<arrow::Array>> point_vec{wkb_point11, wkb_point12};

  std::string polygon1 =
      "POLYGON (("
      "0 0, "
      "10 0, "
      "20 20, "
      "0 20, "
      "0 0))";
  std::string polygon2 =
      "POLYGON (("
      "20 0, "
      "40 0, "
      "40 20, "
      "20 20, "
      "20 0))";
  std::string polygon3 =
      "POLYGON (("
      "0 20, "
      "20 20, "
      "20 40, "
      "0 40, "
      "0 20))";
  std::string polygon4 =
      "POLYGON (("
      "20 20, "
      "40 20, "
      "40 40, "
      "20 40, "
      "20 20))";
  arrow::StringBuilder string_builder2;
  status = string_builder2.Append(polygon1);
  status = string_builder2.Append(polygon2);
  status = string_builder2.Append(polygon3);
  status = string_builder2.Append(polygon4);
  std::shared_ptr<arrow::StringArray> string_array2;
  status = string_builder2.Finish(&string_array2);
  auto wkb_polygon = arctern::gis::gdal::WktToWkb(string_array2);

  std::vector<std::shared_ptr<arrow::Array>> polygon_vec{wkb_polygon};

  std::string index_type = "RTREE";
  auto out = arctern::gis::ST_IndexedWithin(point_vec, polygon_vec);

  arrow::Int32Builder builder;
  std::vector<int> out_std = {0, 1, 2, 3, -1, -1, -1, -1, -1, 0};
  int offset = 0;
  for (int j = 0; j < out.size(); j++) {
    auto values = std::static_pointer_cast<arrow::Int32Array>(out[j]);
    auto size = out[j]->length();
    auto type = out[j]->type_id();
    assert(type == arrow::Type::INT32);
    for (int i = 0; i < size; i++) {
      ASSERT_EQ(values->Value(i), out_std[offset++]);
    }
  }
}

TEST(IndexedWithin, Sheet) {
  vector<string> points_raw;
  vector<string> polygons_raw;
  using std::to_string;
  int S = 100;
  auto proj_str = [](double row, double col) {
    auto nr = row * 3 + col * 2;
    auto nc = row * 2 + col * -2;
    return to_string(nr) + " " + to_string(nc);
  };
  for (int row = 0; row < S; ++row) {
    for (int col = 0; col < S; ++col) {
      auto polygon_str = "Polygon((" + proj_str(row, col) + ", " +
                         proj_str(row + 1, col) + ", " + proj_str(row + 1, col + 1) +
                         ", " + proj_str(row, col + 1) + ", " + proj_str(row, col) + "))";

      polygons_raw.push_back(polygon_str);
      auto point_raw = "Point(" + proj_str(row + 0.5, col + 0.5) + ")";
      points_raw.push_back(point_raw);
    }
  }
  auto vol = S * S;
  vector<int> poly_shuf(vol);
  for (int i = 0; i < vol; ++i) {
    poly_shuf[i] = i;
  }
  std::default_random_engine eng(42);
  std::shuffle(poly_shuf.begin(), poly_shuf.end(), eng);
  vector<string> right(vol);

  for (int i = 0; i < vol; ++i) {
    auto ans = poly_shuf[i];
    right[ans] = polygons_raw[i];
  }

  int N = 10000;
  vector<int> std_res;
  vector<string> left;
  for (int i = 0; i < N; ++i) {
    auto index = eng() % vol;
    left.push_back(points_raw[index]);
    std_res.push_back(poly_shuf[index]);
  }
  auto lg = arctern::gis::StrsToWkb(left);
  auto rg = arctern::gis::StrsToWkb(right);

  auto beg_time = std::chrono::high_resolution_clock::now();
  auto res_vec = arctern::gis::ST_IndexedWithin({lg}, {rg});
  auto end_time = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> time = end_time - beg_time;
  assert(res_vec.size() == 1);
  auto res = std::static_pointer_cast<arrow::Int32Array>(res_vec[0]);
  ASSERT_EQ(res->length(), N);
  for (int i = 0; i < res->length(); ++i) {
    ASSERT_EQ(res->GetView(i), std_res[i]);
  }
}

TEST(IndexedWithin, PyTest) {
  vector<string> left = {"Point(0 0)", "Point(1000 1000)", "Point(10 10)"};
  vector<string> right = {"Polygon((9 10, 11 12, 11 8, 9 10))",
                          "POLYGON ((-1 0, 1 2, 1 -2, -1 0))"};
  vector<int> std_res = {1, -1, 0};
  auto N = left.size();

  auto lg = arctern::gis::StrsToWkb(left);
  auto rg = arctern::gis::StrsToWkb(right);

  auto beg_time = std::chrono::high_resolution_clock::now();
  auto res_vec = arctern::gis::ST_IndexedWithin({lg}, {rg});
  auto end_time = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> time = end_time - beg_time;
  assert(res_vec.size() == 1);
  auto res = std::static_pointer_cast<arrow::Int32Array>(res_vec[0]);
  ASSERT_EQ(res->length(), N);
  for (int i = 0; i < res->length(); ++i) {
    ASSERT_EQ(res->GetView(i), std_res[i]);
  }
}

TEST(IndexedWithin, NullTest) {
  std::string wkt11 = "POINT (10 10)";
  arrow::StringBuilder string_builder11;
  auto status = string_builder11.Append(wkt11);
  std::shared_ptr<arrow::StringArray> string_array11;
  status = string_builder11.Finish(&string_array11);
  auto wkb_point11 = arctern::gis::gdal::WktToWkb(string_array11);
  std::vector<std::shared_ptr<arrow::Array>> point_vec{wkb_point11};
  std::string polygon1 =
      "POLYGON (("
      "0 0, "
      "10 0, "
      "20 20, "
      "0 20, "
      "0 0))";
  std::string polygon2 =
      "POLYGON (("
      "20 0, "
      "40 0, "
      "40 20, "
      "20 20, "
      "20 0))";
  arrow::StringBuilder string_builder2;
  status = string_builder2.AppendNull();
  status = string_builder2.Append(polygon1);
  status = string_builder2.Append(polygon2);
  std::shared_ptr<arrow::StringArray> string_array2;
  status = string_builder2.Finish(&string_array2);
  auto wkb_polygon = arctern::gis::gdal::WktToWkb(string_array2);
  std::vector<std::shared_ptr<arrow::Array>> polygon_vec{wkb_polygon};
  std::string index_type = "RTREE";
  auto out = arctern::gis::ST_IndexedWithin(point_vec, polygon_vec);
  arrow::Int32Builder builder;
  std::vector<int> out_std = {1};
  int offset = 0;
  for (int j = 0; j < out.size(); j++) {
    auto values = std::static_pointer_cast<arrow::Int32Array>(out[j]);
    auto size = out[j]->length();
    auto type = out[j]->type_id();
    assert(type == arrow::Type::INT32);
    for (int i = 0; i < size; i++) {
      ASSERT_EQ(values->Value(i), out_std[offset++]);
    }
  }
}
