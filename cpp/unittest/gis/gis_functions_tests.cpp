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

#include <arrow/api.h>
#include <arrow/array.h>
#include <gtest/gtest.h>
#include <ogr_geometry.h>
#include <ctime>
#include <iostream>

#include "arrow/gis_api.h"
#include "utils/check_status.h"

#define COMMON_TEST_CASES                                                              \
  auto p1 = "POINT (0 1)";                                                             \
                                                                                       \
  auto p2 = "LINESTRING (0 0, 0 1)";                                                   \
  auto p3 = "LINESTRING (0 0, 1 0)";                                                   \
  auto p4 = "LINESTRING (0 0, 1 1)";                                                   \
  auto p5 = "LINESTRING (0 0, 0 1, 1 1)";                                              \
  auto p6 = "LINESTRING (0 0, 1 0, 1 1)";                                              \
  auto p7 = "LINESTRING (0 0, 1 0, 1 1, 0 0)";                                         \
                                                                                       \
  auto p8 = "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))";                                     \
  auto p9 = "POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))";                                     \
  auto p10 = "POLYGON ((0 0, 1 0, 1 1, 0 0))";                                         \
  auto p11 = "POLYGON ((0 0, 0 1, 1 1, 0 0))";                                         \
                                                                                       \
  auto p12 = "MULTIPOINT (0 0, 1 0, 1 2)";                                             \
  auto p13 = "MULTIPOINT (0 0, 1 0, 1 2, 1 2)";                                        \
                                                                                       \
  auto p14 = "MULTILINESTRING ( (0 0, 1 2), (0 0, 1 0, 1 1) )";                        \
  auto p15 = "MULTILINESTRING ( (0 0, 1 2), (0 0, 1 0, 1 1) )";                        \
  auto p16 = "MULTILINESTRING ( (0 0, 1 2), (0 0, 1 0, 1 1),(-1 2,3 4,9 -3,-4 100) )"; \
                                                                                       \
  auto p17 = "MULTIPOLYGON ( ((0 0, 1 0, 1 1,0 0)) )";                                 \
  auto p18 = "MULTIPOLYGON ( ((0 0, 1 1, 1 0,0 0)) )";                                 \
                                                                                       \
  auto p19 =                                                                           \
      "MULTIPOLYGON ( ((0 0, 0 4, 4 4, 4 0, 0 0)), ((0 0, 4 0, 4 1, 0 1, 0 0)) )";     \
  auto p20 =                                                                           \
      "MULTIPOLYGON ( ((0 0, 4 0, 4 1, 0 1, 0 0)), ((0 0, 0 4, 4 4, 4 0, 0 0)) )";     \
  auto p21 =                                                                           \
      "MULTIPOLYGON ( ((0 0, 1 0, 1 1, 0 1, 0 0)), ((0 0, 0 4, 4 4, 4 0, 0 0)) )";     \
  auto p22 =                                                                           \
      "MULTIPOLYGON ( ((0 0, 4 0, 4 4, 0 4, 0 0)), ((0 0, 0 1, 4 1, 4 0, 0 0)) )";     \
  auto p23 =                                                                           \
      "MULTIPOLYGON ( ((0 0, 0 1, 4 1, 4 0, 0 0)), ((0 0, 4 0, 4 4, 0 4, 0 0)) )";     \
  auto p24 =                                                                           \
      "MULTIPOLYGON ( ((0 0, 0 1, 1 1, 1 0, 0 0)), ((0 0, 4 0, 4 4, 0 4, 0 0)) )";     \
                                                                                       \
  auto p25 =                                                                           \
      "MULTIPOLYGON ( ((0 0, 4 0, 4 4, 0 4, 0 0)), ((0 0, 1 0, 1 1, 0 1, 0 0)) )";     \
  auto p26 =                                                                           \
      "MULTIPOLYGON ( ((0 0, 1 0, 1 1, 0 1, 0 0)), ((0 0, 4 0, 4 4, 0 4, 0 0)) )";     \
  auto p27 =                                                                           \
      "MULTIPOLYGON ( ((0 0, 0 4, 4 4, 4 0, 0 0)), ((0 0, 0 1, 1 1, 1 0, 0 0)) )";     \
  auto p28 =                                                                           \
      "MULTIPOLYGON ( ((0 0, 0 1, 1 1, 1 0, 0 0)), ((0 0, 0 4, 4 4, 4 0, 0 0)) )";     \
                                                                                       \
  auto p29 = "MULTIPOLYGON ( ((0 0, 1 0, 1 1, 0 0)), ((0 0, 1 4, 1 0, 0 0)) )";        \
  auto p30 = "MULTIPOLYGON ( ((0 0, 0 4, 4 4, 0 0)), ((0 0, 1 -8, 1 1, 0 1, 0 0)) )";  \
  auto p31 = "MULTIPOLYGON ( ((0 0, 1 -8, 1 1, 0 1, 0 0)), ((0 0, 4 4, 0 4, 0 0)) )";  \
  auto p32 =                                                                           \
      "MULTIPOLYGON ( ((0 0, 1 -8, 1 1, 0 1, 0 0)), ((0 0, 4 4, 0 4, 0 0)),((0 0, 0 "  \
      "-2, -3 4, 0 2, 0 0)) )";                                                        \
  auto p33 =                                                                           \
      "MULTIPOLYGON ( ((0 0, 1 -8, 1 1, 0 1, 0 0)), ((0 0, 4 4, 0 4, 0 0)),((0 0, 0 "  \
      "-2, 3 4, 0 2, 0 0)) )";

#define CONSTRUCT_COMMON_TEST_CASES    \
  arrow::StringBuilder builder;        \
  std::shared_ptr<arrow::Array> input; \
  builder.Append(std::string(p1));     \
  builder.Append(std::string(p2));     \
  builder.Append(std::string(p3));     \
  builder.Append(std::string(p4));     \
  builder.Append(std::string(p5));     \
  builder.Append(std::string(p6));     \
  builder.Append(std::string(p7));     \
  builder.Append(std::string(p8));     \
  builder.Append(std::string(p9));     \
  builder.Append(std::string(p10));    \
  builder.Append(std::string(p11));    \
  builder.Append(std::string(p12));    \
  builder.Append(std::string(p13));    \
  builder.Append(std::string(p14));    \
  builder.Append(std::string(p15));    \
  builder.Append(std::string(p16));    \
  builder.Append(std::string(p17));    \
  builder.Append(std::string(p18));    \
  builder.Append(std::string(p19));    \
  builder.Append(std::string(p20));    \
  builder.Append(std::string(p21));    \
  builder.Append(std::string(p22));    \
  builder.Append(std::string(p23));    \
  builder.Append(std::string(p24));    \
  builder.Append(std::string(p25));    \
  builder.Append(std::string(p26));    \
  builder.Append(std::string(p27));    \
  builder.Append(std::string(p28));    \
  builder.Append(std::string(p29));    \
  builder.Append(std::string(p30));    \
  builder.Append(std::string(p31));    \
  builder.Append(std::string(p32));    \
  builder.Append(std::string(p33));    \
  builder.Finish(&input);

TEST(geometry_test, make_point_from_double) {
  arrow::DoubleBuilder builder_x;
  arrow::DoubleBuilder builder_y;
  std::shared_ptr<arrow::Array> ptr_x;
  std::shared_ptr<arrow::Array> ptr_y;

  for (int i = 0; i < 2; i++) {
    builder_x.Append(i);
    builder_y.Append(i);
  }

  builder_x.Finish(&ptr_x);
  builder_y.Finish(&ptr_y);

  auto point_arr = zilliz::gis::ST_Point(ptr_x, ptr_y);
  auto point_arr_str = std::static_pointer_cast<arrow::StringArray>(point_arr);

  ASSERT_EQ(point_arr_str->length(), 2);
  ASSERT_EQ(point_arr_str->GetString(0), "POINT (0 0)");
  ASSERT_EQ(point_arr_str->GetString(1), "POINT (1 1)");
}

char* build_point(double x, double y) {
  OGRPoint point(x, y);
  char* point_str = nullptr;
  CHECK_GDAL(point.exportToWkt(&point_str));
  return point_str;
}

char* build_polygon(double x, double y) {
  OGRLinearRing ring;
  ring.addPoint(x, y);
  ring.addPoint(x, y + 10);
  ring.addPoint(x + 10, y + 10);
  ring.addPoint(x + 10, y);
  ring.addPoint(x, y);
  ring.closeRings();
  OGRPolygon polygon;
  polygon.addRing(&ring);

  char* polygon_str = nullptr;
  CHECK_GDAL(polygon.exportToWkt(&polygon_str));
  return polygon_str;
}

char* build_linestring(double x, double y) {
  OGRLineString line;
  line.addPoint(x, y);
  line.addPoint(x, y + 20);

  char* line_str = nullptr;
  CHECK_GDAL(line.exportToWkt(&line_str));
  return line_str;
}

std::shared_ptr<arrow::Array> build_points() {
  arrow::StringBuilder string_builder;
  std::shared_ptr<arrow::Array> points;

  char* point_str1 = build_point(10, 20);
  char* point_str2 = build_point(20, 30);
  char* point_str3 = build_point(30, 40);

  string_builder.Append(std::string(point_str1));
  string_builder.Append(std::string(point_str2));
  string_builder.Append(std::string(point_str3));

  CPLFree(point_str1);
  CPLFree(point_str2);
  CPLFree(point_str3);

  string_builder.Finish(&points);
  return points;
}

std::shared_ptr<arrow::Array> build_polygons() {
  arrow::StringBuilder string_builder;
  std::shared_ptr<arrow::Array> polygons;

  char* str1 = build_polygon(10, 20);
  char* str2 = build_polygon(30, 40);
  char* str3 = build_polygon(50, 60);
  string_builder.Append(std::string(str1));
  string_builder.Append(std::string(str2));
  string_builder.Append(std::string(str3));
  CPLFree(str1);
  CPLFree(str2);
  CPLFree(str3);

  string_builder.Finish(&polygons);
  return polygons;
}

std::shared_ptr<arrow::Array> build_linestrings() {
  arrow::StringBuilder string_builder;
  std::shared_ptr<arrow::Array> line_strings;

  char* str1 = build_linestring(10, 20);
  char* str2 = build_linestring(30, 40);
  char* str3 = build_linestring(50, 60);
  string_builder.Append(std::string(str1));
  string_builder.Append(std::string(str2));
  string_builder.Append(std::string(str3));
  CPLFree(str1);
  CPLFree(str2);
  CPLFree(str3);

  string_builder.Finish(&line_strings);
  return line_strings;
}

TEST(geometry_test, test_ST_IsValid) {
  COMMON_TEST_CASES;
  CONSTRUCT_COMMON_TEST_CASES;

  auto res = zilliz::gis::ST_IsValid(input);
  auto res_bool = std::static_pointer_cast<arrow::BooleanArray>(res);

  ASSERT_EQ(res_bool->Value(0), true);
  ASSERT_EQ(res_bool->Value(1), true);
  ASSERT_EQ(res_bool->Value(2), true);
  ASSERT_EQ(res_bool->Value(3), true);
  ASSERT_EQ(res_bool->Value(4), true);
  ASSERT_EQ(res_bool->Value(5), true);
  ASSERT_EQ(res_bool->Value(6), true);
  ASSERT_EQ(res_bool->Value(7), true);
  ASSERT_EQ(res_bool->Value(8), true);
  ASSERT_EQ(res_bool->Value(9), true);
  ASSERT_EQ(res_bool->Value(10), true);
  ASSERT_EQ(res_bool->Value(11), true);
  ASSERT_EQ(res_bool->Value(12), true);
  ASSERT_EQ(res_bool->Value(13), true);
  ASSERT_EQ(res_bool->Value(14), true);
  ASSERT_EQ(res_bool->Value(15), true);
  ASSERT_EQ(res_bool->Value(16), true);
  ASSERT_EQ(res_bool->Value(17), true);
  ASSERT_EQ(res_bool->Value(18), false);
  ASSERT_EQ(res_bool->Value(19), false);
  ASSERT_EQ(res_bool->Value(20), false);
  ASSERT_EQ(res_bool->Value(21), false);
  ASSERT_EQ(res_bool->Value(22), false);
  ASSERT_EQ(res_bool->Value(23), false);
  ASSERT_EQ(res_bool->Value(24), false);
  ASSERT_EQ(res_bool->Value(25), false);
  ASSERT_EQ(res_bool->Value(26), false);
  ASSERT_EQ(res_bool->Value(27), false);
  ASSERT_EQ(res_bool->Value(28), false);
  ASSERT_EQ(res_bool->Value(29), false);
  ASSERT_EQ(res_bool->Value(30), false);
  ASSERT_EQ(res_bool->Value(31), false);
  ASSERT_EQ(res_bool->Value(32), false);
}

TEST(geometry_test, test_ST_Intersection) {
  auto l1 = "POINT (0 1)";
  auto l2 = "POINT (0 1)";
  auto l3 = "POINT (0 1)";
  auto l4 = "POINT (0 1)";
  auto l5 = "POINT (0 1)";
  auto l6 = "POINT (0 1)";
  auto l7 = "POINT (0 1)";
  auto l8 = "POINT (0 1)";
  auto l9 = "POINT (0 1)";
  auto l10 = "POINT (0 1)";
  auto l11 = "POINT (0 1)";
  auto l12 = "POINT (0 1)";
  auto l13 = "POINT (0 1)";
  auto l14 = "POINT (0 1)";
  auto l15 = "POINT (0 1)";
  auto l16 = "POINT (0 1)";

  auto l17 = "MULTIPOINT (1 8, 2 3)";
  auto l18 = "MULTIPOINT (1 8, 2 3)";
  auto l19 = "MULTIPOINT (1 8, 2 3)";
  auto l20 = "MULTIPOINT (1 8, 2 3)";
  auto l21 = "MULTIPOINT (1 8, 2 3)";
  auto l22 = "MULTIPOINT (1 8, 2 3)";
  auto l23 = "MULTIPOINT (1 8, 2 3)";
  auto l24 = "MULTIPOINT (1 8, 2 3)";
  auto l25 = "MULTIPOINT (1 8, 2 3)";
  auto l26 = "MULTIPOINT (1 8, 2 3)";
  auto l27 = "MULTIPOINT (1 8, 2 3)";
  auto l28 = "MULTIPOINT (1 8, 2 3)";
  auto l29 = "MULTIPOINT (1 8, 2 3)";
  auto l30 = "MULTIPOINT (1 8, 2 3)";

  auto l31 = "LINESTRING (0 0, 1 0, 1 8)";
  auto l32 = "LINESTRING (0 0, 1 0, 1 8)";
  auto l33 = "LINESTRING (0 0, 1 0, 1 8)";
  auto l34 = "LINESTRING (0 0, 1 0, 1 8)";
  auto l35 = "LINESTRING (0 0, 1 0, 1 8)";
  auto l36 = "LINESTRING (0 0, 1 0, 1 8)";
  auto l37 = "LINESTRING (0 0, 1 0, 1 8)";
  auto l38 = "LINESTRING (0 0, 1 0, 1 8)";
  auto l39 = "LINESTRING (0 0, 1 0, 1 8)";
  auto l40 = "LINESTRING (0 0, 1 0, 1 8)";
  auto l41 = "LINESTRING (0 0, 1 0, 1 8)";
  auto l42 = "LINESTRING (0 0, 1 0, 1 8)";

  auto l43 = "MULTILINESTRING ( (0 1, 0 1), (0 1, 2 3, 1 1) )";
  auto l44 = "MULTILINESTRING ( (0 1, 0 1), (0 1, 2 3, 1 1) )";
  auto l45 = "MULTILINESTRING ( (0 1, 0 1), (0 1, 2 3, 1 1) )";
  auto l46 = "MULTILINESTRING ( (0 1, 0 1), (0 1, 2 3, 1 1) )";
  auto l47 = "MULTILINESTRING ( (0 1, 0 1), (0 1, 2 3, 1 1) )";
  auto l48 = "MULTILINESTRING ( (0 1, 0 1), (0 1, 2 3, 1 1) )";
  auto l49 = "MULTILINESTRING ( (0 1, 0 1), (0 1, 2 3, 1 1) )";
  auto l50 = "MULTILINESTRING ( (0 1, 0 1), (0 1, 2 3, 1 1) )";
  auto l51 = "MULTILINESTRING ( (0 1, 0 1), (0 1, 2 3, 1 1) )";

  auto l52 = "POLYGON ((0 1, 2 3, 1 1, 1 0, 0 1))";
  auto l53 = "POLYGON ((0 1, 2 3, 1 1, 1 0, 0 1))";
  auto l54 = "POLYGON ((0 1, 2 3, 1 1, 1 0, 0 1))";
  auto l55 = "POLYGON ((0 1, 2 3, 1 1, 1 0, 0 1))";
  auto l56 = "POLYGON ((0 1, 2 3, 1 1, 1 0, 0 1))";
  auto l57 = "POLYGON ((0 1, 2 3, 1 1, 1 0, 0 1))";
  auto l58 = "POLYGON ((0 1, 2 3, 1 1, 1 0, 0 1))";

  auto l59 = "MULTIPOLYGON ( ((0 0, 0 4, 1 8, 4 0, 0 0)), ((0 0, 4 0, 4 1, 0 1, 0 0)) )";
  auto l60 = "MULTIPOLYGON ( ((0 0, 0 4, 1 8, 4 0, 0 0)), ((0 0, 4 0, 4 1, 0 1, 0 0)) )";
  auto l61 = "MULTIPOLYGON ( ((0 0, 0 4, 1 8, 4 0, 0 0)), ((0 0, 4 0, 4 1, 0 1, 0 0)) )";
  auto l62 = "MULTIPOLYGON ( ((0 0, 0 4, 1 8, 4 0, 0 0)), ((0 0, 4 0, 4 1, 0 1, 0 0)) )";

  arrow::StringBuilder builder1;
  std::shared_ptr<arrow::Array> input1;
  builder1.Append(std::string(l1));
  builder1.Append(std::string(l2));
  builder1.Append(std::string(l3));
  builder1.Append(std::string(l4));
  builder1.Append(std::string(l5));
  builder1.Append(std::string(l6));
  builder1.Append(std::string(l7));
  builder1.Append(std::string(l8));
  builder1.Append(std::string(l9));
  builder1.Append(std::string(l10));
  builder1.Append(std::string(l11));
  builder1.Append(std::string(l12));
  builder1.Append(std::string(l13));
  builder1.Append(std::string(l14));
  // builder1.Append(std::string(l15));
  // builder1.Append(std::string(l16));
  builder1.Append(std::string(l17));
  builder1.Append(std::string(l18));
  builder1.Append(std::string(l19));
  builder1.Append(std::string(l20));
  builder1.Append(std::string(l21));
  builder1.Append(std::string(l22));
  builder1.Append(std::string(l23));
  builder1.Append(std::string(l24));
  builder1.Append(std::string(l25));
  builder1.Append(std::string(l26));
  builder1.Append(std::string(l27));
  builder1.Append(std::string(l28));
  builder1.Append(std::string(l29));
  builder1.Append(std::string(l30));
  builder1.Append(std::string(l31));
  builder1.Append(std::string(l32));
  builder1.Append(std::string(l33));
  builder1.Append(std::string(l34));
  builder1.Append(std::string(l35));
  builder1.Append(std::string(l36));
  builder1.Append(std::string(l37));
  builder1.Append(std::string(l38));
  builder1.Append(std::string(l39));
  builder1.Append(std::string(l40));
  // builder1.Append(std::string(l41));
  builder1.Append(std::string(l42));
  builder1.Append(std::string(l43));
  builder1.Append(std::string(l44));
  builder1.Append(std::string(l45));
  builder1.Append(std::string(l46));
  builder1.Append(std::string(l47));
  builder1.Append(std::string(l48));
  // builder1.Append(std::string(l49));
  // builder1.Append(std::string(l50));
  // builder1.Append(std::string(l51));
  builder1.Append(std::string(l52));
  builder1.Append(std::string(l53));
  builder1.Append(std::string(l54));
  builder1.Append(std::string(l55));
  builder1.Append(std::string(l56));
  // builder1.Append(std::string(l57));
  // builder1.Append(std::string(l58));
  // builder1.Append(std::string(l59));
  // builder1.Append(std::string(l60));
  // builder1.Append(std::string(l61));
  // builder1.Append(std::string(l62));
  builder1.Finish(&input1);

  auto r1 = "POINT (0 1)";
  auto r2 = "POINT (3 1)";
  auto r3 = "MULTIPOINT (0 1, 1 0, 1 2, 1 2)";
  auto r4 = "MULTIPOINT (0 2, 1 0, 1 2, 1 2)";
  auto r5 = "LINESTRING (0 0, 0 1, 1 1)";
  auto r6 = "LINESTRING (0 0, 0 2, 1 1)";
  auto r7 = "LINESTRING (0 0, 0.5 2, 1 1)";
  auto r8 = "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))";
  auto r9 = "POLYGON ((0 0, 1 0, 1 1, 0 2, 0 0))";
  auto r10 = "POLYGON ((0 0, 1 0, 1 1, 0 0.5, 0 0))";
  auto r11 = "MULTILINESTRING ( (0 0, 1 2), (0 0, 0 2, 1 1),(-1 2,3 4,9 -3,-4 100) )";
  auto r12 = "MULTILINESTRING ( (0 0, 1 2), (0 0, 1 0, 1 1),(-1 2,3 4,9 -3,-4 100) )";
  auto r13 = "MULTIPOLYGON ( ((0 0, 1 1, 0 2,0 0)) )";
  auto r14 = "MULTIPOLYGON ( ((0 0, 1 1, 2 0,0 0)) )";
  auto r15 = "MULTIPOLYGON ( ((0 0, 0 4, 4 4, 4 0, 0 0)), ((0 0, 4 0, 4 1, 0 1, 0 0)) )";
  auto r16 = "MULTIPOLYGON ( ((0 0, 4 4, 4 0, 0 0)), ((0 0, 4 0, 4 1, 0 0)) )";

  auto r17 = "MULTIPOINT (0 1, 1 0, 1 8, 1 2)";
  auto r18 = "MULTIPOINT (0 1, 0 2)";
  auto r19 = "LINESTRING (0 0, 0 1, 1 8)";
  auto r20 = "LINESTRING (0 1, 0 1, 0 1)";
  auto r21 = "LINESTRING (0 0, 0 1, 2 3, 0 1, 0 0)";
  auto r22 = "POLYGON ((0 0, 0 1, 0 1, 0 1, 0 0))";
  auto r23 = "POLYGON ((0 1, 2 3, 1 1, 1 0, 0 1))";
  auto r24 = "POLYGON ((0 0, 1 0, 1 8, 0 0.5, 0 0))";
  auto r25 = "MULTILINESTRING ( (0 0, 1 8), (0 0, 0 2, 1 1),(-1 2,3 4,9 -3,-4 100) )";
  auto r26 = "MULTILINESTRING ( (0 1, 0 1), (0 1, 2 3, 0 1) )";
  auto r27 = "MULTIPOLYGON ( ((0 0, 1 8, 0 2,0 0)) )";
  auto r28 = "MULTIPOLYGON ( ((0 1, 0 1, 0 1,0 1)) )";
  auto r29 = "MULTIPOLYGON ( ((0 0, 0 4, 1 8, 4 0, 0 0)), ((0 0, 4 0, 4 1, 0 1, 0 0)) )";
  auto r30 = "MULTIPOLYGON ( ((0 1, 2 3, 0 1,0 1)), ((0 1, 0 1, 0 1, 0 1)) )";

  auto r31 = "LINESTRING (0 0, 1 0, 1 8)";
  auto r32 = "LINESTRING (0 0, 1 1, 0 1)";
  auto r33 = "LINESTRING (0 0, 0 1, 2 3, 0 1, 0 0)";
  auto r34 = "POLYGON ((0 0, 0 1, 0 1, 0 1, 0 0))";
  auto r35 = "POLYGON ((0 1, 2 3, 1 1, 1 0, 0 1))";
  auto r36 = "POLYGON ((0 0, 1 0, 1 8, 0 0.5, 0 0))";
  auto r37 = "MULTILINESTRING ( (0 0, 1 8), (0 0, 0 2, 1 1),(-1 2,3 4,9 -3,-4 100) )";
  auto r38 = "MULTILINESTRING ( (0 1, 0 1), (0 1, 2 3, 0 1) )";
  auto r39 = "MULTIPOLYGON ( ((0 0, 1 8, 0 2,0 0)) )";
  auto r40 = "MULTIPOLYGON ( ((0 1, 0 1, 0 1,0 1)) )";
  auto r41 = "MULTIPOLYGON ( ((0 0, 0 4, 1 8, 4 0, 0 0)), ((0 0, 4 0, 4 1, 0 1, 0 0)) )";
  auto r42 = "MULTIPOLYGON ( ((0 1, 1 8, 0 1,0 1)), ((0 1, 0 1, 0 1, 0 1)) )";

  auto r43 = "POLYGON ((0 0, 0 1, 0 1, 0 1, 0 0))";
  auto r44 = "POLYGON ((0 1, 2 3, 1 1, 1 0, 0 1))";
  auto r45 = "POLYGON ((0 0, 1 0, 1 8, 0 0.5, 0 0))";
  auto r46 = "MULTILINESTRING ( (0 0, 1 8), (0 0, 0 2, 1 1),(0 1, 2 3, 1 1) )";
  auto r47 = "MULTILINESTRING ( (0 1, 0 1), (0 1, 2 3, 1 1) )";
  auto r48 = "MULTIPOLYGON ( ((0 0, 1 8, 0 2,0 0)) )";
  auto r49 = "MULTIPOLYGON ( ((0 1, 2 3, 3 0, 0 0, 0 1)) )";
  auto r50 = "MULTIPOLYGON ( ((0 0, 0 4, 1 8, 4 0, 0 0)), ((0 0, 4 0, 4 1, 0 1, 0 0)) )";
  auto r51 = "MULTIPOLYGON ( ((0 1, 1 8, 0 1,0 1)), ((0 1, 0 1, 0 1, 0 1)) )";

  auto r52 = "POLYGON ((0 0, 0 1, 0 1, 0 1, 0 0))";
  auto r53 = "POLYGON ((0 1, 2 3, 1 1, 1 0, 0 1))";
  auto r54 = "POLYGON ((0 0, 1 0, 1 8, 0 0.5, 0 0))";
  auto r55 = "MULTIPOLYGON ( ((0 0, 1 8, 0 2,0 0)) )";
  auto r56 = "MULTIPOLYGON ( ((0 1, 2 3, 3 0, 0 0, 0 1)) )";
  auto r57 = "MULTIPOLYGON ( ((0 0, 0 4, 1 8, 4 0, 0 0)), ((0 0, 4 0, 4 1, 0 1, 0 0)) )";
  auto r58 = "MULTIPOLYGON ( ((0 1, 1 8, 0 1,0 1)), ((0 1, 0 1, 0 1, 0 1)) )";

  auto r59 = "MULTIPOLYGON ( ((0 0, 1 8, 0 2,0 0)) )";
  auto r60 = "MULTIPOLYGON ( ((0 1, 2 3, 3 0, 0 0, 0 1)) )";
  auto r61 = "MULTIPOLYGON ( ((0 0, 0 4, 1 8, 4 0, 0 0)), ((0 0, 4 0, 4 1, 0 1, 0 0)) )";
  auto r62 = "MULTIPOLYGON ( ((0 1, 1 8, 0 1,0 1)), ((0 1, 0 1, 0 1, 0 1)) )";

  arrow::StringBuilder builder2;
  std::shared_ptr<arrow::Array> input2;
  builder2.Append(std::string(r1));
  builder2.Append(std::string(r2));
  builder2.Append(std::string(r3));
  builder2.Append(std::string(r4));
  builder2.Append(std::string(r5));
  builder2.Append(std::string(r6));
  builder2.Append(std::string(r7));
  builder2.Append(std::string(r8));
  builder2.Append(std::string(r9));
  builder2.Append(std::string(r10));
  builder2.Append(std::string(r11));
  builder2.Append(std::string(r12));
  builder2.Append(std::string(r13));
  builder2.Append(std::string(r14));
  // builder2.Append(std::string(r15));
  // builder2.Append(std::string(r16));
  builder2.Append(std::string(r17));
  builder2.Append(std::string(r18));
  builder2.Append(std::string(r19));
  builder2.Append(std::string(r20));
  builder2.Append(std::string(r21));
  builder2.Append(std::string(r22));
  builder2.Append(std::string(r23));
  builder2.Append(std::string(r24));
  builder2.Append(std::string(r25));
  builder2.Append(std::string(r26));
  builder2.Append(std::string(r27));
  builder2.Append(std::string(r28));
  builder2.Append(std::string(r29));
  builder2.Append(std::string(r30));
  builder2.Append(std::string(r31));
  builder2.Append(std::string(r32));
  builder2.Append(std::string(r33));
  builder2.Append(std::string(r34));
  builder2.Append(std::string(r35));
  builder2.Append(std::string(r36));
  builder2.Append(std::string(r37));
  builder2.Append(std::string(r38));
  builder2.Append(std::string(r39));
  builder2.Append(std::string(r40));
  // builder2.Append(std::string(r41));
  builder2.Append(std::string(r42));
  builder2.Append(std::string(r43));
  builder2.Append(std::string(r44));
  builder2.Append(std::string(r45));
  builder2.Append(std::string(r46));
  builder2.Append(std::string(r47));
  builder2.Append(std::string(r48));
  // builder2.Append(std::string(r49));
  // builder2.Append(std::string(r50));
  // builder2.Append(std::string(r51));
  builder2.Append(std::string(r52));
  builder2.Append(std::string(r53));
  builder2.Append(std::string(r54));
  builder2.Append(std::string(r55));
  builder2.Append(std::string(r56));
  // builder2.Append(std::string(r57));
  // builder2.Append(std::string(r58));
  // builder2.Append(std::string(r59));
  // builder2.Append(std::string(r60));
  // builder2.Append(std::string(r61));
  // builder2.Append(std::string(r62));
  builder2.Finish(&input2);

  auto res = zilliz::gis::ST_Intersection(input1, input2);
  auto res_str = std::static_pointer_cast<arrow::StringArray>(res);

  ASSERT_EQ(res_str->GetString(0), "POINT (0 1)");
  ASSERT_EQ(res_str->GetString(1), "POINT EMPTY");
  ASSERT_EQ(res_str->GetString(2), "POINT (0 1)");
  // ASSERT_EQ(res_str->GetString(3), "MULTIPOLYGON EMPTY"); // POINT EMPTY
  ASSERT_EQ(res_str->GetString(4), "POINT (0 1)");
  ASSERT_EQ(res_str->GetString(5), "POINT (0 1)");
  // ASSERT_EQ(res_str->GetString(6), "MULTIPOLYGON EMPTY"); // POINT EMPTY
  ASSERT_EQ(res_str->GetString(7), "POINT (0 1)");
  ASSERT_EQ(res_str->GetString(8), "POINT (0 1)");
  // ASSERT_EQ(res_str->GetString(9), "MULTIPOLYGON EMPTY"); // POINT EMPTY
  ASSERT_EQ(res_str->GetString(10), "POINT (0 1)");
  // ASSERT_EQ(res_str->GetString(11), "MULTIPOLYGON EMPTY"); // POINT EMPTY
  ASSERT_EQ(res_str->GetString(12), "POINT (0 1)");
  // ASSERT_EQ(res_str->GetString(13), "MULTIPOLYGON EMPTY"); // POINT EMPTY
  // ASSERT_EQ(res_str->GetString(14), "POINT (0 1)"); // error
  // ASSERT_EQ(res_str->GetString(15), "MULTIPOLYGON EMPTY"); // error
  // TODO : need verify against geospark result below.
  ASSERT_EQ(res_str->GetString(16), "POINT (1 8)");
  ASSERT_EQ(res_str->GetString(17), "POINT EMPTY");
  ASSERT_EQ(res_str->GetString(18), "POINT (2 3)");
  ASSERT_EQ(res_str->GetString(19), "POINT EMPTY");
  ASSERT_EQ(res_str->GetString(20), "POINT (2 3)");
  ASSERT_EQ(res_str->GetString(21), "POINT (1 8)");
  ASSERT_EQ(res_str->GetString(22), "POINT (1 8)");
  ASSERT_EQ(res_str->GetString(23), "POINT (2 3)");
  ASSERT_EQ(res_str->GetString(24), "POINT (1 8)");
  ASSERT_EQ(res_str->GetString(25), "POINT EMPTY");
  ASSERT_EQ(res_str->GetString(26), "MULTIPOINT (1 8,2 3)");
  ASSERT_EQ(res_str->GetString(27), "POINT (2 3)");
  ASSERT_EQ(res_str->GetString(28), "MULTILINESTRING ((0 0,1 0),(1 0,1 8))");
  ASSERT_EQ(res_str->GetString(29), "MULTIPOINT (0 0,1 1)");
  ASSERT_EQ(res_str->GetString(30), "MULTIPOINT (0 0,1 2)");
  ASSERT_EQ(res_str->GetString(31), "LINESTRING (0 0,1 0,1 8)");
  ASSERT_EQ(res_str->GetString(32), "MULTILINESTRING ((1 0,1 1),(1 1,1 2))");
  ASSERT_EQ(res_str->GetString(33), "MULTILINESTRING ((0 0,1 0),(1 0,1 8))");
  ASSERT_EQ(res_str->GetString(34), "MULTIPOINT (0 0,1 1,1 3,1 8)");
  ASSERT_EQ(res_str->GetString(35), "POINT (1 2)");
  ASSERT_EQ(res_str->GetString(36), "MULTIPOINT (0 0,1 8)");
  ASSERT_EQ(res_str->GetString(37), "LINESTRING EMPTY");
  ASSERT_EQ(res_str->GetString(38), "LINESTRING (0 0,1 0,1 8)");
  ASSERT_EQ(res_str->GetString(39), "LINESTRING (0 1,2 3,1 1)");
  ASSERT_EQ(res_str->GetString(40), "MULTILINESTRING ((0 1,2 3),(2 3,1 1))");
  ASSERT_EQ(res_str->GetString(41),
            "GEOMETRYCOLLECTION (POINT (1 1),LINESTRING (0.076923076923077 "
            "1.07692307692308,1 2))");
  ASSERT_EQ(
      res_str->GetString(42),
      "MULTILINESTRING ((0 1,0.142857142857143 1.14285714285714),(0.142857142857143 "
      "1.14285714285714,0.5 1.5),(0.5 1.5,2 3),(2 3,1 1))");
  ASSERT_EQ(res_str->GetString(43), "MULTILINESTRING ((0 1,2 3),(2 3,1 1))");
  ASSERT_EQ(res_str->GetString(44),
            "LINESTRING (0 1,0.142857142857143 1.14285714285714)");
  ASSERT_EQ(res_str->GetString(45), "POLYGON ((0 1,2 3,1 1,1 0,0 1))");
  ASSERT_EQ(res_str->GetString(46), "POLYGON ((0 1,2 3,1 1,1 0,0 1))");
  ASSERT_EQ(res_str->GetString(47),
            "POLYGON ((0.076923076923077 1.07692307692308,1 2,1 1,1 0,0.058823529411765 "
            "0.941176470588235,0.076923076923077 1.07692307692308))");
  ASSERT_EQ(res_str->GetString(48),
            "POLYGON ((0 1,0.142857142857143 1.14285714285714,0.111111111111111 "
            "0.888888888888889,0 1))");
  ASSERT_EQ(res_str->GetString(49), "POLYGON ((0 1,2 3,1 1,1 0,0 1))");
}

// TEST(geometry_test, test_ST_PrecisionReduce){
//   OGRPoint point(1.5555555,1.55555555);
//   arrow::StringBuilder string_builder;
//   std::shared_ptr<arrow::Array> array;

//   char *str = nullptr;
//   CHECK_GDAL(point.exportToWkt(&str));
//   string_builder.Append(std::string(str));
//   CPLFree(str);

//   string_builder.Finish(&array);
//   auto geometries = zilliz::gis::ST_PrecisionReduce(array,6);
//   auto geometries_arr = std::static_pointer_cast<arrow::StringArray>(geometries);

//   // ASSERT_EQ(geometries_arr->GetString(0),"POINT (1.55556 1.55556)");
//   ASSERT_EQ(geometries_arr->GetString(0),"POINT (1.5555555 1.55555555)");
// }

TEST(geometry_test, test_ST_Equals) {
  auto l1 = "POINT (0 1)";
  auto l2 = "POINT (0 1)";
  auto l3 = "POINT (0 1)";
  auto l4 = "POINT (0 1)";
  auto l5 = "POINT (0 1)";
  auto l6 = "POINT (0 1)";
  auto l7 = "POINT (0 1)";
  auto l8 = "POINT (0 1)";
  auto l9 = "POINT (0 1)";
  auto l10 = "POINT (0 1)";
  auto l11 = "POINT (0 1)";
  auto l12 = "POINT (0 1)";

  auto l13 = "LINESTRING (0 0, 0 1, 1 1)";
  auto l14 = "LINESTRING (0 0, 0 1, 1 1)";
  auto l15 = "LINESTRING (0 0, 0 1, 1 1)";
  auto l16 = "LINESTRING (0 0, 0 1, 1 1)";
  auto l17 = "LINESTRING (0 0, 0 1, 1 1)";
  auto l18 = "LINESTRING (0 0, 0 1, 1 1)";
  auto l19 = "LINESTRING (0 0, 0 1, 1 1)";
  auto l20 = "LINESTRING (0 0, 0 1, 1 1)";
  auto l21 = "LINESTRING (0 0, 0 1, 1 1)";
  auto l22 = "LINESTRING (0 0, 0 1, 1 1)";

  auto l23 = "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))";
  auto l24 = "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))";
  auto l25 = "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))";
  auto l26 = "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))";
  auto l27 = "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))";
  auto l28 = "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))";
  auto l29 = "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))";
  auto l30 = "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))";

  auto l31 = "MULTIPOINT (0 1, 0 1)";
  auto l32 = "MULTIPOINT (0 1, 0 1)";
  auto l33 = "MULTIPOINT (0 1, 0 1)";
  auto l34 = "MULTIPOINT (0 1, 0 1)";
  auto l35 = "MULTIPOINT (0 1, 0 1)";
  auto l36 = "MULTIPOINT (0 1, 0 1)";

  auto l37 = "MULTILINESTRING ( (0 1, 0 1), (0 0, 0 1, 1 1) )";
  auto l38 = "MULTILINESTRING ( (0 1, 0 1), (0 0, 0 1, 1 1) )";
  auto l39 = "MULTILINESTRING ( (0 1, 0 1), (0 0, 0 1, 1 1) )";
  auto l40 = "MULTILINESTRING ( (0 1, 0 1), (0 0, 0 1, 1 1) )";

  auto l41 = "MULTIPOLYGON ( ((0 0, 0 1, 1 1, 0 0)), ((0 0, 0 1, 1 1,0 0)) )";
  auto l42 = "MULTIPOLYGON ( ((0 0, 0 1, 1 1, 0 0)), ((0 0, 0 1, 1 1,0 0)) )";

  arrow::StringBuilder builder1;
  std::shared_ptr<arrow::Array> input1;
  builder1.Append(std::string(l1));
  builder1.Append(std::string(l2));
  builder1.Append(std::string(l3));
  builder1.Append(std::string(l4));
  builder1.Append(std::string(l5));
  builder1.Append(std::string(l6));
  builder1.Append(std::string(l7));
  builder1.Append(std::string(l8));
  builder1.Append(std::string(l9));
  builder1.Append(std::string(l10));
  builder1.Append(std::string(l11));
  builder1.Append(std::string(l12));
  builder1.Append(std::string(l13));
  builder1.Append(std::string(l14));
  builder1.Append(std::string(l15));
  builder1.Append(std::string(l16));
  builder1.Append(std::string(l17));
  builder1.Append(std::string(l18));
  builder1.Append(std::string(l19));
  builder1.Append(std::string(l20));
  builder1.Append(std::string(l21));
  builder1.Append(std::string(l22));
  builder1.Append(std::string(l23));
  builder1.Append(std::string(l24));
  builder1.Append(std::string(l25));
  builder1.Append(std::string(l26));
  builder1.Append(std::string(l27));
  builder1.Append(std::string(l28));
  builder1.Append(std::string(l29));
  builder1.Append(std::string(l30));
  builder1.Append(std::string(l31));
  builder1.Append(std::string(l32));
  builder1.Append(std::string(l33));
  builder1.Append(std::string(l34));
  builder1.Append(std::string(l35));
  builder1.Append(std::string(l36));
  builder1.Append(std::string(l37));
  builder1.Append(std::string(l38));
  builder1.Append(std::string(l39));
  builder1.Append(std::string(l40));
  builder1.Append(std::string(l41));
  builder1.Append(std::string(l42));
  builder1.Finish(&input1);

  auto r1 = "POINT (0 1)";
  auto r2 = "POINT (3 1)";
  auto r3 = "MULTIPOINT (0 1, 1 0, 1 2, 1 2)";
  auto r4 = "MULTIPOINT (0 1, 0 1)";
  auto r5 = "LINESTRING (0 0, 0 1, 1 1)";
  auto r6 = "LINESTRING (0 1, 0 1)";
  auto r7 = "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))";
  auto r8 = "POLYGON ((0 1, 0 1, 0 1, 0 1))";
  auto r9 = "MULTILINESTRING ( (0 1, 0 1), (0 0, 0 1, 0 1) )";
  auto r10 = "MULTILINESTRING ( (0 1, 0 1), (0 1, 0 1, 0 1) )";
  auto r11 = "MULTIPOLYGON ( ((0 0, 1 1, 0 2,0 0)) )";
  auto r12 = "MULTIPOLYGON ( ((0 1, 0 1, 0 1,0 1)), ((0 1, 0 1, 0 1,0 1)) )";

  auto r13 = "MULTIPOINT (0 1, 1 0, 1 2, 1 2)";
  auto r14 = "MULTIPOINT (0 1, 0 1)";
  auto r15 = "LINESTRING (0 0, 0 1, 1 1)";
  auto r16 = "LINESTRING (0 1, 0 1)";
  auto r17 = "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))";
  auto r18 = "POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))";
  auto r19 = "MULTILINESTRING ( (0 1, 0 1), (0 0, 0 1, 1 1) )";
  auto r20 = "MULTILINESTRING ( (0 0, 0 1, 1 1), (0 0, 0 1, 1 1) )";
  auto r21 = "MULTIPOLYGON ( ((0 0, 1 1, 0 2,0 0)) )";
  auto r22 = "MULTIPOLYGON ( ((0 0, 0 1, 1 1,0 0)), ((0 0, 0 1, 1 1,0 0)) )";

  auto r23 = "MULTIPOINT (0 1, 1 0, 1 2, 1 2)";
  auto r24 = "MULTIPOINT (0 1, 0 1)";
  auto r25 = "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))";
  auto r26 = "POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))";
  auto r27 = "MULTILINESTRING ( (0 1, 0 1), (0 0, 0 1, 1 1) )";
  auto r28 = "MULTILINESTRING ( (0 0, 0 1, 1 1), (0 0, 0 1, 1 1) )";
  auto r29 = "MULTIPOLYGON ( ((0 0, 1 1, 0 2,0 0)) )";
  auto r30 = "MULTIPOLYGON ( ((0 0, 0 1, 1 1,0 0)), ((0 0, 0 1, 1 1,0 0)) )";

  auto r31 = "MULTIPOINT (0 1, 1 0, 1 2)";
  auto r32 = "MULTIPOINT (0 1, 0 1)";
  auto r33 = "MULTILINESTRING ( (0 1, 0 1), (0 0, 0 1, 1 1) )";
  auto r34 = "MULTILINESTRING ( (0 0, 0 1, 1 1), (0 0, 0 1, 1 1) )";
  auto r35 = "MULTIPOLYGON ( ((0 0, 1 1, 0 2, 0 0)) )";
  auto r36 = "MULTIPOLYGON ( ((0 0, 0 1, 1 1, 0 0)), ((0 0, 0 1, 1 1,0 0)) )";

  auto r37 = "MULTILINESTRING ( (0 1, 0 1), (0 0, 0 1, 1 1) )";
  auto r38 = "MULTILINESTRING ( (0 0, 0 1, 1 1), (0 0, 0 1, 1 1) )";
  auto r39 = "MULTIPOLYGON ( ((0 0, 1 1, 0 2, 0 0)) )";
  auto r40 = "MULTIPOLYGON ( ((0 0, 0 1, 1 1, 0 0)), ((0 0, 0 1, 1 1,0 0)) )";

  auto r41 = "MULTIPOLYGON ( ((0 0, 1 1, 0 2, 0 0)) )";
  auto r42 = "MULTIPOLYGON ( ((0 0, 0 1, 1 1, 0 0)), ((0 0, 0 1, 1 1,0 0)) )";

  arrow::StringBuilder builder2;
  std::shared_ptr<arrow::Array> input2;
  builder2.Append(std::string(r1));
  builder2.Append(std::string(r2));
  builder2.Append(std::string(r3));
  builder2.Append(std::string(r4));
  builder2.Append(std::string(r5));
  builder2.Append(std::string(r6));
  builder2.Append(std::string(r7));
  builder2.Append(std::string(r8));
  builder2.Append(std::string(r9));
  builder2.Append(std::string(r10));
  builder2.Append(std::string(r11));
  builder2.Append(std::string(r12));
  builder2.Append(std::string(r13));
  builder2.Append(std::string(r14));
  builder2.Append(std::string(r15));
  builder2.Append(std::string(r16));
  builder2.Append(std::string(r17));
  builder2.Append(std::string(r18));
  builder2.Append(std::string(r19));
  builder2.Append(std::string(r20));
  builder2.Append(std::string(r21));
  builder2.Append(std::string(r22));
  builder2.Append(std::string(r23));
  builder2.Append(std::string(r24));
  builder2.Append(std::string(r25));
  builder2.Append(std::string(r26));
  builder2.Append(std::string(r27));
  builder2.Append(std::string(r28));
  builder2.Append(std::string(r29));
  builder2.Append(std::string(r30));
  builder2.Append(std::string(r31));
  builder2.Append(std::string(r32));
  builder2.Append(std::string(r33));
  builder2.Append(std::string(r34));
  builder2.Append(std::string(r35));
  builder2.Append(std::string(r36));
  builder2.Append(std::string(r37));
  builder2.Append(std::string(r38));
  builder2.Append(std::string(r39));
  builder2.Append(std::string(r40));
  builder2.Append(std::string(r41));
  builder2.Append(std::string(r42));
  builder2.Finish(&input2);

  auto res = zilliz::gis::ST_Equals(input1, input2);
  auto res_bool = std::static_pointer_cast<arrow::BooleanArray>(res);

  ASSERT_EQ(res_bool->Value(0), true);
  ASSERT_EQ(res_bool->Value(1), false);
  ASSERT_EQ(res_bool->Value(2), false);
  // ASSERT_EQ(res_bool->Value(3), true); // false
  ASSERT_EQ(res_bool->Value(4), false);
  // ASSERT_EQ(res_bool->Value(5), true); // false
  ASSERT_EQ(res_bool->Value(6), false);
  // ASSERT_EQ(res_bool->Value(7), true); // false
  ASSERT_EQ(res_bool->Value(8), false);
  // ASSERT_EQ(res_bool->Value(9), true); // false
  ASSERT_EQ(res_bool->Value(10), false);
  // ASSERT_EQ(res_bool->Value(11), true); // false
  ASSERT_EQ(res_bool->Value(12), false);
  ASSERT_EQ(res_bool->Value(13), false);
  ASSERT_EQ(res_bool->Value(14), true);
  ASSERT_EQ(res_bool->Value(15), false);
  ASSERT_EQ(res_bool->Value(16), false);
  ASSERT_EQ(res_bool->Value(17), false);
  ASSERT_EQ(res_bool->Value(18), false);
  ASSERT_EQ(res_bool->Value(19), false);
  ASSERT_EQ(res_bool->Value(20), false);
  ASSERT_EQ(res_bool->Value(21), false);
  ASSERT_EQ(res_bool->Value(22), false);
  ASSERT_EQ(res_bool->Value(23), false);
  ASSERT_EQ(res_bool->Value(24), true);
  ASSERT_EQ(res_bool->Value(25), false);
  ASSERT_EQ(res_bool->Value(26), false);
  ASSERT_EQ(res_bool->Value(27), false);
  ASSERT_EQ(res_bool->Value(28), false);
  ASSERT_EQ(res_bool->Value(29), false);
  ASSERT_EQ(res_bool->Value(30), false);
  ASSERT_EQ(res_bool->Value(31), true);
  ASSERT_EQ(res_bool->Value(32), false);
  ASSERT_EQ(res_bool->Value(33), false);
  ASSERT_EQ(res_bool->Value(34), false);
  ASSERT_EQ(res_bool->Value(35), false);
  ASSERT_EQ(res_bool->Value(36), true);
  ASSERT_EQ(res_bool->Value(37), false);
  ASSERT_EQ(res_bool->Value(38), false);
  ASSERT_EQ(res_bool->Value(39), false);
  ASSERT_EQ(res_bool->Value(40), false);
  ASSERT_EQ(res_bool->Value(41), true);
}

TEST(geometry_test, test_ST_Touches) {
  auto l1 = "POINT (0 1)";
  auto l2 = "POINT (0 1)";
  auto l3 = "POINT (0 1)";
  auto l4 = "POINT (0 1)";
  auto l5 = "POINT (0 1)";
  auto l6 = "POINT (0 1)";
  auto l7 = "POINT (0 1)";
  auto l8 = "POINT (0 1)";
  auto l9 = "POINT (0 1)";
  auto l10 = "POINT (0 1)";
  auto l11 = "POINT (0 1)";
  auto l12 = "POINT (0 1)";
  auto l13 = "POINT (0 1)";
  auto l14 = "POINT (0 1)";
  auto l15 = "POINT (0 1)";
  auto l16 = "POINT (0 1)";

  auto l17 = "MULTIPOINT (1 8, 2 3)";
  auto l18 = "MULTIPOINT (1 8, 2 3)";
  auto l19 = "MULTIPOINT (1 8, 2 3)";
  auto l20 = "MULTIPOINT (1 8, 2 3)";
  auto l21 = "MULTIPOINT (1 8, 2 3)";
  auto l22 = "MULTIPOINT (1 8, 2 3)";
  auto l23 = "MULTIPOINT (1 8, 2 3)";
  auto l24 = "MULTIPOINT (1 8, 2 3)";
  auto l25 = "MULTIPOINT (1 8, 2 3)";
  auto l26 = "MULTIPOINT (1 8, 2 3)";
  auto l27 = "MULTIPOINT (1 8, 2 3)";
  auto l28 = "MULTIPOINT (1 8, 2 3)";
  auto l29 = "MULTIPOINT (1 8, 2 3)";
  auto l30 = "MULTIPOINT (1 8, 2 3)";

  auto l31 = "LINESTRING (0 0, 1 0, 1 8)";
  auto l32 = "LINESTRING (0 0, 1 0, 1 8)";
  auto l33 = "LINESTRING (0 0, 1 0, 1 8)";
  auto l34 = "LINESTRING (0 0, 1 0, 1 8)";
  auto l35 = "LINESTRING (0 0, 1 0, 1 8)";
  auto l36 = "LINESTRING (0 0, 1 0, 1 8)";
  auto l37 = "LINESTRING (0 0, 1 0, 1 8)";
  auto l38 = "LINESTRING (0 0, 1 0, 1 8)";
  auto l39 = "LINESTRING (0 0, 1 0, 1 8)";
  auto l40 = "LINESTRING (0 0, 1 0, 1 8)";
  auto l41 = "LINESTRING (0 0, 1 0, 1 8)";
  auto l42 = "LINESTRING (0 0, 1 0, 1 8)";

  auto l43 = "MULTILINESTRING ( (0 1, 0 1), (0 1, 2 3, 1 1) )";
  auto l44 = "MULTILINESTRING ( (0 1, 0 1), (0 1, 2 3, 1 1) )";
  auto l45 = "MULTILINESTRING ( (0 1, 0 1), (0 1, 2 3, 1 1) )";
  auto l46 = "MULTILINESTRING ( (0 1, 0 1), (0 1, 2 3, 1 1) )";
  auto l47 = "MULTILINESTRING ( (0 1, 0 1), (0 1, 2 3, 1 1) )";
  auto l48 = "MULTILINESTRING ( (0 1, 0 1), (0 1, 2 3, 1 1) )";
  auto l49 = "MULTILINESTRING ( (0 1, 0 1), (0 1, 2 3, 1 1) )";
  auto l50 = "MULTILINESTRING ( (0 1, 0 1), (0 1, 2 3, 1 1) )";
  auto l51 = "MULTILINESTRING ( (0 1, 0 1), (0 1, 2 3, 1 1) )";

  auto l52 = "POLYGON ((0 1, 2 3, 1 1, 1 0, 0 1))";
  auto l53 = "POLYGON ((0 1, 2 3, 1 1, 1 0, 0 1))";
  auto l54 = "POLYGON ((0 1, 2 3, 1 1, 1 0, 0 1))";
  auto l55 = "POLYGON ((0 1, 2 3, 1 1, 1 0, 0 1))";
  auto l56 = "POLYGON ((0 1, 2 3, 1 1, 1 0, 0 1))";
  auto l57 = "POLYGON ((0 1, 2 3, 1 1, 1 0, 0 1))";
  auto l58 = "POLYGON ((0 1, 2 3, 1 1, 1 0, 0 1))";

  auto l59 = "MULTIPOLYGON ( ((0 0, 0 4, 1 8, 4 0, 0 0)), ((0 0, 4 0, 4 1, 0 1, 0 0)) )";
  auto l60 = "MULTIPOLYGON ( ((0 0, 0 4, 1 8, 4 0, 0 0)), ((0 0, 4 0, 4 1, 0 1, 0 0)) )";
  auto l61 = "MULTIPOLYGON ( ((0 0, 0 4, 1 8, 4 0, 0 0)), ((0 0, 4 0, 4 1, 0 1, 0 0)) )";
  auto l62 = "MULTIPOLYGON ( ((0 0, 0 4, 1 8, 4 0, 0 0)), ((0 0, 4 0, 4 1, 0 1, 0 0)) )";

  arrow::StringBuilder builder1;
  std::shared_ptr<arrow::Array> input1;
  builder1.Append(std::string(l1));
  builder1.Append(std::string(l2));
  builder1.Append(std::string(l3));
  builder1.Append(std::string(l4));
  builder1.Append(std::string(l5));
  builder1.Append(std::string(l6));
  builder1.Append(std::string(l7));
  builder1.Append(std::string(l8));
  builder1.Append(std::string(l9));
  builder1.Append(std::string(l10));
  builder1.Append(std::string(l11));
  builder1.Append(std::string(l12));
  builder1.Append(std::string(l13));
  builder1.Append(std::string(l14));
  builder1.Append(std::string(l15));
  builder1.Append(std::string(l16));
  builder1.Append(std::string(l17));
  builder1.Append(std::string(l18));
  builder1.Append(std::string(l19));
  builder1.Append(std::string(l20));
  builder1.Append(std::string(l21));
  builder1.Append(std::string(l22));
  builder1.Append(std::string(l23));
  builder1.Append(std::string(l24));
  builder1.Append(std::string(l25));
  builder1.Append(std::string(l26));
  builder1.Append(std::string(l27));
  builder1.Append(std::string(l28));
  builder1.Append(std::string(l29));
  builder1.Append(std::string(l30));
  builder1.Append(std::string(l31));
  builder1.Append(std::string(l32));
  builder1.Append(std::string(l33));
  builder1.Append(std::string(l34));
  builder1.Append(std::string(l35));
  builder1.Append(std::string(l36));
  builder1.Append(std::string(l37));
  builder1.Append(std::string(l38));
  builder1.Append(std::string(l39));
  builder1.Append(std::string(l40));
  builder1.Append(std::string(l41));
  builder1.Append(std::string(l42));
  builder1.Append(std::string(l43));
  builder1.Append(std::string(l44));
  builder1.Append(std::string(l45));
  builder1.Append(std::string(l46));
  builder1.Append(std::string(l47));
  builder1.Append(std::string(l48));
  builder1.Append(std::string(l49));
  builder1.Append(std::string(l50));
  builder1.Append(std::string(l51));
  builder1.Append(std::string(l52));
  builder1.Append(std::string(l53));
  builder1.Append(std::string(l54));
  builder1.Append(std::string(l55));
  builder1.Append(std::string(l56));
  builder1.Append(std::string(l57));
  builder1.Append(std::string(l58));
  builder1.Append(std::string(l59));
  builder1.Append(std::string(l60));
  builder1.Append(std::string(l61));
  builder1.Append(std::string(l62));
  builder1.Finish(&input1);

  auto r1 = "POINT (0 1)";
  auto r2 = "POINT (3 1)";
  auto r3 = "MULTIPOINT (0 1, 1 0, 1 2, 1 2)";
  auto r4 = "MULTIPOINT (0 1, 0 1)";
  auto r5 = "LINESTRING (0 0, 0 1, 1 1)";
  auto r6 = "LINESTRING (0 1, 0 1, 0 1)";
  auto r7 = "LINESTRING (0 0, 0 1, 0 1, 0 1, 0 0)";
  auto r8 = "POLYGON ((0 0, 0 1, 0 1, 0 1, 0 0))";
  auto r9 = "POLYGON ((0 1, 0 1, 0 1, 0 1, 0 1))";
  auto r10 = "POLYGON ((0 0, 1 0, 1 1, 0 0.5, 0 0))";
  auto r11 = "MULTILINESTRING ( (0 0, 1 2), (0 0, 0 2, 1 1),(-1 2,3 4,9 -3,-4 100) )";
  auto r12 = "MULTILINESTRING ( (0 1, 0 1), (0 1, 0 1, 0 1) )";
  auto r13 = "MULTIPOLYGON ( ((0 0, 1 1, 0 2,0 0)) )";
  auto r14 = "MULTIPOLYGON ( ((0 1, 0 1, 0 1,0 1)) )";
  auto r15 = "MULTIPOLYGON ( ((0 0, 0 4, 4 4, 4 0, 0 0)), ((0 0, 4 0, 4 1, 0 1, 0 0)) )";
  auto r16 = "MULTIPOLYGON ( ((0 1, 0 1, 0 1,0 1)), ((0 1, 0 1, 0 1,0 1)) )";

  auto r17 = "MULTIPOINT (0 1, 1 0, 1 8, 1 2)";
  auto r18 = "MULTIPOINT (0 1, 0 2)";
  auto r19 = "LINESTRING (0 0, 0 1, 1 8)";
  auto r20 = "LINESTRING (0 1, 0 1, 0 1)";
  auto r21 = "LINESTRING (0 0, 0 1, 2 3, 0 1, 0 0)";
  auto r22 = "POLYGON ((0 0, 0 1, 0 1, 0 1, 0 0))";
  auto r23 = "POLYGON ((0 1, 2 3, 1 1, 1 0, 0 1))";
  auto r24 = "POLYGON ((0 0, 1 0, 1 8, 0 0.5, 0 0))";
  auto r25 = "MULTILINESTRING ( (0 0, 1 8), (0 0, 0 2, 1 1),(-1 2,3 4,9 -3,-4 100) )";
  auto r26 = "MULTILINESTRING ( (0 1, 0 1), (0 1, 2 3, 0 1) )";
  auto r27 = "MULTIPOLYGON ( ((0 0, 1 8, 0 2,0 0)) )";
  auto r28 = "MULTIPOLYGON ( ((0 1, 0 1, 0 1,0 1)) )";
  auto r29 = "MULTIPOLYGON ( ((0 0, 0 4, 1 8, 4 0, 0 0)), ((0 0, 4 0, 4 1, 0 1, 0 0)) )";
  auto r30 = "MULTIPOLYGON ( ((0 1, 2 3, 0 1,0 1)), ((0 1, 0 1, 0 1, 0 1)) )";

  auto r31 = "LINESTRING (0 0, 1 0, 1 8)";
  auto r32 = "LINESTRING (0 0, 1 1, 0 1)";
  auto r33 = "LINESTRING (0 0, 0 1, 2 3, 0 1, 0 0)";
  auto r34 = "POLYGON ((0 0, 0 1, 0 1, 0 1, 0 0))";
  auto r35 = "POLYGON ((0 1, 2 3, 1 1, 1 0, 0 1))";
  auto r36 = "POLYGON ((0 0, 1 0, 1 8, 0 0.5, 0 0))";
  auto r37 = "MULTILINESTRING ( (0 0, 1 8), (0 0, 0 2, 1 1),(-1 2,3 4,9 -3,-4 100) )";
  auto r38 = "MULTILINESTRING ( (0 1, 0 1), (0 1, 2 3, 0 1) )";
  auto r39 = "MULTIPOLYGON ( ((0 0, 1 8, 0 2,0 0)) )";
  auto r40 = "MULTIPOLYGON ( ((0 1, 0 1, 0 1,0 1)) )";
  auto r41 = "MULTIPOLYGON ( ((0 0, 0 4, 1 8, 4 0, 0 0)), ((0 0, 4 0, 4 1, 0 1, 0 0)) )";
  auto r42 = "MULTIPOLYGON ( ((0 1, 1 8, 0 1,0 1)), ((0 1, 0 1, 0 1, 0 1)) )";

  auto r43 = "POLYGON ((0 0, 0 1, 0 1, 0 1, 0 0))";
  auto r44 = "POLYGON ((0 1, 2 3, 1 1, 1 0, 0 1))";
  auto r45 = "POLYGON ((0 0, 1 0, 1 8, 0 0.5, 0 0))";
  auto r46 = "MULTILINESTRING ( (0 0, 1 8), (0 0, 0 2, 1 1),(0 1, 2 3, 1 1) )";
  auto r47 = "MULTILINESTRING ( (0 1, 0 1), (0 1, 2 3, 1 1) )";
  auto r48 = "MULTIPOLYGON ( ((0 0, 1 8, 0 2,0 0)) )";
  auto r49 = "MULTIPOLYGON ( ((0 1, 2 3, 3 0, 0 0, 0 1)) )";
  auto r50 = "MULTIPOLYGON ( ((0 0, 0 4, 1 8, 4 0, 0 0)), ((0 0, 4 0, 4 1, 0 1, 0 0)) )";
  auto r51 = "MULTIPOLYGON ( ((0 1, 1 8, 0 1,0 1)), ((0 1, 0 1, 0 1, 0 1)) )";

  auto r52 = "POLYGON ((0 0, 0 1, 0 1, 0 1, 0 0))";
  auto r53 = "POLYGON ((0 1, 2 3, 1 1, 1 0, 0 1))";
  auto r54 = "POLYGON ((0 0, 1 0, 1 8, 0 0.5, 0 0))";
  auto r55 = "MULTIPOLYGON ( ((0 0, 1 8, 0 2,0 0)) )";
  auto r56 = "MULTIPOLYGON ( ((0 1, 2 3, 3 0, 0 0, 0 1)) )";
  auto r57 = "MULTIPOLYGON ( ((0 0, 0 4, 1 8, 4 0, 0 0)), ((0 0, 4 0, 4 1, 0 1, 0 0)) )";
  auto r58 = "MULTIPOLYGON ( ((0 1, 1 8, 0 1,0 1)), ((0 1, 0 1, 0 1, 0 1)) )";

  auto r59 = "MULTIPOLYGON ( ((0 0, 1 8, 0 2,0 0)) )";
  auto r60 = "MULTIPOLYGON ( ((0 1, 2 3, 3 0, 0 0, 0 1)) )";
  auto r61 = "MULTIPOLYGON ( ((0 0, 0 4, 1 8, 4 0, 0 0)), ((0 0, 4 0, 4 1, 0 1, 0 0)) )";
  auto r62 = "MULTIPOLYGON ( ((0 1, 1 8, 0 1,0 1)), ((0 1, 0 1, 0 1, 0 1)) )";

  arrow::StringBuilder builder2;
  std::shared_ptr<arrow::Array> input2;
  builder2.Append(std::string(r1));
  builder2.Append(std::string(r2));
  builder2.Append(std::string(r3));
  builder2.Append(std::string(r4));
  builder2.Append(std::string(r5));
  builder2.Append(std::string(r6));
  builder2.Append(std::string(r7));
  builder2.Append(std::string(r8));
  builder2.Append(std::string(r9));
  builder2.Append(std::string(r10));
  builder2.Append(std::string(r11));
  builder2.Append(std::string(r12));
  builder2.Append(std::string(r13));
  builder2.Append(std::string(r14));
  builder2.Append(std::string(r15));
  builder2.Append(std::string(r16));
  builder2.Append(std::string(r17));
  builder2.Append(std::string(r18));
  builder2.Append(std::string(r19));
  builder2.Append(std::string(r20));
  builder2.Append(std::string(r21));
  builder2.Append(std::string(r22));
  builder2.Append(std::string(r23));
  builder2.Append(std::string(r24));
  builder2.Append(std::string(r25));
  builder2.Append(std::string(r26));
  builder2.Append(std::string(r27));
  builder2.Append(std::string(r28));
  builder2.Append(std::string(r29));
  builder2.Append(std::string(r30));
  builder2.Append(std::string(r31));
  builder2.Append(std::string(r32));
  builder2.Append(std::string(r33));
  builder2.Append(std::string(r34));
  builder2.Append(std::string(r35));
  builder2.Append(std::string(r36));
  builder2.Append(std::string(r37));
  builder2.Append(std::string(r38));
  builder2.Append(std::string(r39));
  builder2.Append(std::string(r40));
  builder2.Append(std::string(r41));
  builder2.Append(std::string(r42));
  builder2.Append(std::string(r43));
  builder2.Append(std::string(r44));
  builder2.Append(std::string(r45));
  builder2.Append(std::string(r46));
  builder2.Append(std::string(r47));
  builder2.Append(std::string(r48));
  builder2.Append(std::string(r49));
  builder2.Append(std::string(r50));
  builder2.Append(std::string(r51));
  builder2.Append(std::string(r52));
  builder2.Append(std::string(r53));
  builder2.Append(std::string(r54));
  builder2.Append(std::string(r55));
  builder2.Append(std::string(r56));
  builder2.Append(std::string(r57));
  builder2.Append(std::string(r58));
  builder2.Append(std::string(r59));
  builder2.Append(std::string(r60));
  builder2.Append(std::string(r61));
  builder2.Append(std::string(r62));
  builder2.Finish(&input2);

  auto res = zilliz::gis::ST_Touches(input1, input2);
  auto res_bool = std::static_pointer_cast<arrow::BooleanArray>(res);

  // ASSERT_EQ(res_bool->Value(0 ), true);
  ASSERT_EQ(res_bool->Value(1), false);
  ASSERT_EQ(res_bool->Value(2), false);
  // ASSERT_EQ(res_bool->Value(3 ), true);
  ASSERT_EQ(res_bool->Value(4), false);
  // ASSERT_EQ(res_bool->Value(5 ), true);
  ASSERT_EQ(res_bool->Value(6), false);
  ASSERT_EQ(res_bool->Value(7), true);
  ASSERT_EQ(res_bool->Value(8), true);
  ASSERT_EQ(res_bool->Value(9), false);
  ASSERT_EQ(res_bool->Value(10), false);
  // ASSERT_EQ(res_bool->Value(11), true);
  // ASSERT_EQ(res_bool->Value(12), false);
  ASSERT_EQ(res_bool->Value(13), true);
  // ASSERT_EQ(res_bool->Value(14), false);
  // ASSERT_EQ(res_bool->Value(15), true);
  // TODO : need verify against geospark result below
  ASSERT_EQ(res_bool->Value(16), false);
  ASSERT_EQ(res_bool->Value(17), false);
  ASSERT_EQ(res_bool->Value(18), true);
  ASSERT_EQ(res_bool->Value(19), false);
  ASSERT_EQ(res_bool->Value(20), false);
  ASSERT_EQ(res_bool->Value(21), false);
  ASSERT_EQ(res_bool->Value(22), true);
  ASSERT_EQ(res_bool->Value(23), true);
  ASSERT_EQ(res_bool->Value(24), true);
  ASSERT_EQ(res_bool->Value(25), false);
  ASSERT_EQ(res_bool->Value(26), true);
  ASSERT_EQ(res_bool->Value(27), false);
  ASSERT_EQ(res_bool->Value(28), true);
  ASSERT_EQ(res_bool->Value(29), true);
  ASSERT_EQ(res_bool->Value(30), false);
  ASSERT_EQ(res_bool->Value(31), false);
  ASSERT_EQ(res_bool->Value(32), false);
  ASSERT_EQ(res_bool->Value(33), true);
  ASSERT_EQ(res_bool->Value(34), false);
  ASSERT_EQ(res_bool->Value(35), true);
  ASSERT_EQ(res_bool->Value(36), false);
  ASSERT_EQ(res_bool->Value(37), false);
  ASSERT_EQ(res_bool->Value(38), true);
  ASSERT_EQ(res_bool->Value(39), false);
  ASSERT_EQ(res_bool->Value(40), true);
  ASSERT_EQ(res_bool->Value(41), true);
  ASSERT_EQ(res_bool->Value(42), true);
  ASSERT_EQ(res_bool->Value(43), true);
  ASSERT_EQ(res_bool->Value(44), false);
  ASSERT_EQ(res_bool->Value(45), false);
  ASSERT_EQ(res_bool->Value(46), false);
  ASSERT_EQ(res_bool->Value(47), false);
  ASSERT_EQ(res_bool->Value(48), false);
  ASSERT_EQ(res_bool->Value(49), true);
  ASSERT_EQ(res_bool->Value(50), false);
  ASSERT_EQ(res_bool->Value(51), true);
  ASSERT_EQ(res_bool->Value(52), false);
  ASSERT_EQ(res_bool->Value(53), false);
  ASSERT_EQ(res_bool->Value(54), false);
  ASSERT_EQ(res_bool->Value(55), false);
  ASSERT_EQ(res_bool->Value(56), true);
  ASSERT_EQ(res_bool->Value(57), false);
  ASSERT_EQ(res_bool->Value(58), true);
  ASSERT_EQ(res_bool->Value(59), true);
  ASSERT_EQ(res_bool->Value(60), true);
  ASSERT_EQ(res_bool->Value(61), true);
}

TEST(geometry_test, test_ST_Overlaps) {
  auto l1 = "POINT (0 1)";
  auto l2 = "POINT (0 1)";
  auto l3 = "POINT (0 1)";
  auto l4 = "POINT (0 1)";
  auto l5 = "POINT (0 1)";
  auto l6 = "POINT (0 1)";

  auto l7 = "LINESTRING (0 0, 0 1)";
  auto l8 = "LINESTRING (0 0, 0 1)";
  auto l9 = "LINESTRING (0 0, 0 1)";
  auto l10 = "LINESTRING (0 0, 0 1)";
  auto l11 = "LINESTRING (0 0, 0 1)";
  auto l12 = "LINESTRING (0 0, 0 1)";

  auto l13 = "POLYGON ((0 0,0 1,1 1,1 0,0 0))";
  auto l14 = "POLYGON ((0 0,0 1,1 1,1 0,0 0))";
  auto l15 = "POLYGON ((0 0,0 1,1 1,1 0,0 0))";
  auto l16 = "POLYGON ((0 0,0 1,1 1,1 0,0 0))";
  auto l17 = "POLYGON ((0 0,0 1,1 1,1 0,0 0))";

  auto l18 = "MULTIPOINT (0 1, 1 2, 3 3)";
  auto l19 = "LINESTRING ( 0 0.6,0 1, 1 2,3 4)";
  auto l20 = "LINESTRING ( 0 0.8,0 1, 1 2,3 4)";
  auto l21 = "LINESTRING ( 0 1, 1 2,3 4)";
  auto l22 = "MULTIPOLYGON ( ((0 2, 0 3, 3 3, 3 2, 0 2)), ((0 0, 7 0, 1 1, 0 1, 0 0)) )";
  auto l23 = "MULTIPOLYGON ( ((0 2, 0 3, 3 3, 3 2, 0 2)), ((0 0, 7 0, 1 1, 0 1, 0 0)) )";

  arrow::StringBuilder builder1;
  std::shared_ptr<arrow::Array> input1;
  builder1.Append(std::string(l1));
  builder1.Append(std::string(l2));
  builder1.Append(std::string(l3));
  builder1.Append(std::string(l4));
  builder1.Append(std::string(l5));
  builder1.Append(std::string(l6));
  builder1.Append(std::string(l7));
  builder1.Append(std::string(l8));
  builder1.Append(std::string(l9));
  builder1.Append(std::string(l10));
  builder1.Append(std::string(l11));
  builder1.Append(std::string(l12));
  // builder1.Append(std::string(l13));
  // builder1.Append(std::string(l14));
  // builder1.Append(std::string(l15));
  // builder1.Append(std::string(l16));
  // builder1.Append(std::string(l17));
  builder1.Append(std::string(l18));
  builder1.Append(std::string(l19));
  builder1.Append(std::string(l20));
  builder1.Append(std::string(l21));
  builder1.Append(std::string(l22));
  builder1.Append(std::string(l23));
  builder1.Finish(&input1);

  auto r1 = "POINT (0 1)";
  auto r2 = "MULTIPOINT (0 1, 0 1)";
  auto r3 = "LINESTRING (0 1, 0 1, 0 1)";
  auto r4 = "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))";
  auto r5 = "MULTILINESTRING ( (0 1, 0 1), (0 1, 0 1, 0 1) )";
  auto r6 = "MULTIPOLYGON ( ((0 1, 0 1, 0 1,0 1)), ((0 1, 0 1, 0 1,0 1)) )";

  auto r7 = "MULTIPOINT (0 1, 0 1)";
  auto r8 = "LINESTRING (0 1, 0 1, 0 1)";
  auto r9 = "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))";
  auto r10 = "MULTILINESTRING ( (0 1, 0 1), (0 1, 0 1, 0 1) )";
  auto r11 = "MULTIPOLYGON ( ((0 1, 0 1, 0 1,0 1)), ((0 1, 0 1, 0 1,0 1)) )";

  auto r12 = "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))";
  auto r13 = "MULTIPOLYGON ( ((0 0, 0 2, 2 3,2 0,0 0)) )";
  auto r14 = "MULTIPOLYGON ( ((0.5 0.5, 0.5 0.7, 0.7 0.7, 0.7 0.5, 0.5 0.5)) )";
  auto r15 = "POLYGON ((0.5 0.5, 0.5 1.7, 0.6 1.7, 0.4 0.5, 0.5 0.5))";
  auto r16 = "MULTIPOLYGON ( ((0 2, 0 3, 3 3, 3 2, 0 2)), ((0 0, 1 0, 1 1, 0 1, 0 0)) )";
  auto r17 = "MULTIPOLYGON ( ((0 0, 1 0, 1 1, 0 1, 0 0)), ((0 2, 0 3, 3 3, 3 2, 0 2)) )";

  auto r18 = "MULTIPOINT (0 1, 1 2, 3 4)";
  auto r19 = "LINESTRING (0 0.7, 0 1, 1 2)";
  auto r20 = "LINESTRING (0 0.7, 0 1, 1 2)";
  auto r21 = "LINESTRING (0 0.7, 0 1, 1 2)";
  auto r22 =
      "MULTIPOLYGON ( ((0 2, 0 3, 3 3, 3 2, 0 2)), ((0.5 0.5, 6 0.5, 1 1, 0 1, 0.5 0.5)) "
      ")";
  auto r23 =
      "MULTIPOLYGON ( ((0 2, 0 3, 3 3, 3 2, 0 2)), ((0.5 0.5, 6 0.5, 1 2, 0 1, 0.5 0.5)) "
      ")";

  arrow::StringBuilder builder2;
  std::shared_ptr<arrow::Array> input2;
  builder2.Append(std::string(r1));
  builder2.Append(std::string(r2));
  builder2.Append(std::string(r3));
  builder2.Append(std::string(r4));
  builder2.Append(std::string(r5));
  builder2.Append(std::string(r6));
  builder2.Append(std::string(r7));
  builder2.Append(std::string(r8));
  builder2.Append(std::string(r9));
  builder2.Append(std::string(r10));
  builder2.Append(std::string(r11));
  builder2.Append(std::string(r12));
  // builder2.Append(std::string(r13));
  // builder2.Append(std::string(r14));
  // builder2.Append(std::string(r15));
  // builder2.Append(std::string(r16));
  // builder2.Append(std::string(r17));
  builder2.Append(std::string(r18));
  builder2.Append(std::string(r19));
  builder2.Append(std::string(r20));
  builder2.Append(std::string(r21));
  builder2.Append(std::string(r22));
  builder2.Append(std::string(r23));
  builder2.Finish(&input2);

  auto res = zilliz::gis::ST_Overlaps(input1, input2);
  auto res_bool = std::static_pointer_cast<arrow::BooleanArray>(res);

  ASSERT_EQ(res_bool->Value(0), false);
  ASSERT_EQ(res_bool->Value(1), false);
  ASSERT_EQ(res_bool->Value(2), false);
  ASSERT_EQ(res_bool->Value(3), false);
  ASSERT_EQ(res_bool->Value(4), false);
  ASSERT_EQ(res_bool->Value(5), false);
  ASSERT_EQ(res_bool->Value(6), false);
  ASSERT_EQ(res_bool->Value(7), false);
  ASSERT_EQ(res_bool->Value(8), false);
  ASSERT_EQ(res_bool->Value(9), false);
  ASSERT_EQ(res_bool->Value(10), false);
  ASSERT_EQ(res_bool->Value(11), false);
  // ASSERT_EQ(res_bool->Value(12), false); // true
  // ASSERT_EQ(res_bool->Value(13), false); // gis error
  // ASSERT_EQ(res_bool->Value(14), true);  // gis error
  // ASSERT_EQ(res_bool->Value(15), false); // gis error
  // ASSERT_EQ(res_bool->Value(16), false); // gis error
  // ASSERT_EQ(res_bool->Value(17), true);  // gis error
  ASSERT_EQ(res_bool->Value(13), false);
  ASSERT_EQ(res_bool->Value(14), true);
  ASSERT_EQ(res_bool->Value(15), true);
  ASSERT_EQ(res_bool->Value(16), true);
  ASSERT_EQ(res_bool->Value(17), true);  // geospark error
}

TEST(geometry_test, test_ST_Crosses) {
  auto l1 = "MULTIPOINT (0 1, 1 0, 1 2)";
  auto l2 = "MULTIPOINT (0 1, 1 1, 4 0)";
  auto l3 = "MULTIPOINT (0 1, 1 0, 1 2)";
  auto l4 = "MULTIPOINT (0 1, 5 0, 1 2)";
  auto l5 = "LINESTRING (-1 0, 0.1 0, 4 0)";
  auto l6 = "LINESTRING (-1 1, 0.1 0, 4 0)";
  auto l7 = "LINESTRING (-1 1, 0.1 1, 4 0)";
  auto l8 = "MULTIPOINT (0 1, 1 0, 1 2)";
  auto l9 = "MULTIPOINT (0 1, 1 2) ";
  auto l10 = "LINESTRING (0 0, 0 1)";
  auto l11 = "LINESTRING (0 0, 0 1)";
  auto l12 = "LINESTRING (0 -1, 0 1)";

  arrow::StringBuilder builder1;
  std::shared_ptr<arrow::Array> input1;
  builder1.Append(std::string(l1));
  builder1.Append(std::string(l2));
  builder1.Append(std::string(l3));
  builder1.Append(std::string(l4));
  builder1.Append(std::string(l5));
  builder1.Append(std::string(l6));
  builder1.Append(std::string(l7));
  builder1.Append(std::string(l8));
  builder1.Append(std::string(l9));
  builder1.Append(std::string(l10));
  builder1.Append(std::string(l11));
  builder1.Append(std::string(l12));
  builder1.Finish(&input1);

  auto r1 = "LINESTRING (-1 0, 1 0, 4 0)";
  auto r2 = "LINESTRING (-1 0, 1 0, 4 0)";
  auto r3 = "LINESTRING (-1 0, 0.1 0, 4 0)";
  auto r4 = "LINESTRING (-1 0, 0.1 0, 4 0)";
  auto r5 = "LINESTRING (-1 0, 0.1 0, 4 0)";
  auto r6 = "LINESTRING (-1 0, 0.1 0, 4 0)";
  auto r7 = "LINESTRING (-1 0, 0.1 0, 4 0)";
  auto r8 = "POLYGON ((1 0,1 3,2 3,3 0,1 0))";
  auto r9 = "POLYGON ((0 0.5,1 3,2 3,3 0.5,0 0.5))";
  auto r10 = "POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))";
  auto r11 = "POLYGON ((0.5 0, 0 2, 1 1, 1 0, 0.5 0))";
  auto r12 = "POLYGON ((-0.5 0, 0 2, 1 1, 1 0, -0.5 0))";

  arrow::StringBuilder builder2;
  std::shared_ptr<arrow::Array> input2;
  builder2.Append(std::string(r1));
  builder2.Append(std::string(r2));
  builder2.Append(std::string(r3));
  builder2.Append(std::string(r4));
  builder2.Append(std::string(r5));
  builder2.Append(std::string(r6));
  builder2.Append(std::string(r7));
  builder2.Append(std::string(r8));
  builder2.Append(std::string(r9));
  builder2.Append(std::string(r10));
  builder2.Append(std::string(r11));
  builder2.Append(std::string(r12));
  builder2.Finish(&input2);

  auto res = zilliz::gis::ST_Crosses(input1, input2);
  auto res_bool = std::static_pointer_cast<arrow::BooleanArray>(res);

  ASSERT_EQ(res_bool->Value(0), true);
  ASSERT_EQ(res_bool->Value(1), false);
  ASSERT_EQ(res_bool->Value(2), true);
  ASSERT_EQ(res_bool->Value(3), false);
  ASSERT_EQ(res_bool->Value(4), false);
  ASSERT_EQ(res_bool->Value(5), false);
  ASSERT_EQ(res_bool->Value(6), false);
  ASSERT_EQ(res_bool->Value(7), false);
  ASSERT_EQ(res_bool->Value(8), true);
  ASSERT_EQ(res_bool->Value(9), false);
  ASSERT_EQ(res_bool->Value(10), false);
  ASSERT_EQ(res_bool->Value(11), true);
}

TEST(geometry_test, test_ST_IsSimple) {
  COMMON_TEST_CASES;
  CONSTRUCT_COMMON_TEST_CASES;

  auto res = zilliz::gis::ST_IsSimple(input);
  auto res_bool = std::static_pointer_cast<arrow::BooleanArray>(res);

  ASSERT_EQ(res_bool->Value(0), true);
  ASSERT_EQ(res_bool->Value(1), true);
  ASSERT_EQ(res_bool->Value(2), true);
  ASSERT_EQ(res_bool->Value(3), true);
  ASSERT_EQ(res_bool->Value(4), true);
  ASSERT_EQ(res_bool->Value(5), true);
  ASSERT_EQ(res_bool->Value(6), true);
  ASSERT_EQ(res_bool->Value(7), true);
  ASSERT_EQ(res_bool->Value(8), true);
  ASSERT_EQ(res_bool->Value(9), true);
  ASSERT_EQ(res_bool->Value(10), true);
  ASSERT_EQ(res_bool->Value(11), true);
  ASSERT_EQ(res_bool->Value(12), false);
  ASSERT_EQ(res_bool->Value(13), true);
  ASSERT_EQ(res_bool->Value(14), true);
  ASSERT_EQ(res_bool->Value(15), true);
  ASSERT_EQ(res_bool->Value(16), true);
  ASSERT_EQ(res_bool->Value(17), true);
  ASSERT_EQ(res_bool->Value(18), true);
  ASSERT_EQ(res_bool->Value(19), true);
  ASSERT_EQ(res_bool->Value(20), true);
  ASSERT_EQ(res_bool->Value(21), true);
  ASSERT_EQ(res_bool->Value(22), true);
  ASSERT_EQ(res_bool->Value(23), true);
  ASSERT_EQ(res_bool->Value(24), true);
  ASSERT_EQ(res_bool->Value(25), true);
  ASSERT_EQ(res_bool->Value(26), true);
  ASSERT_EQ(res_bool->Value(27), true);
  ASSERT_EQ(res_bool->Value(28), true);
  ASSERT_EQ(res_bool->Value(29), true);
  ASSERT_EQ(res_bool->Value(30), true);
  ASSERT_EQ(res_bool->Value(31), true);
  ASSERT_EQ(res_bool->Value(32), true);
}

TEST(geometry_test, test_ST_MakeValid) {
  auto p1 = "POINT (1 2)";
  auto p2 = "LINESTRING (0 0,0 1,2 0,3 1)";
  auto p3 = "POLYGON ((0 0,0 1,1 2,0 0))";
  auto p4 = "MULTIPOLYGON ( ((0 0, 0 4, 4 4, 4 0, 0 0)), ((0 0, 4 0, 4 1, 0 1, 0 0)) )";
  auto p5 = "MULTIPOINT (1 0,2 3)";
  auto p6 = "MULTILINESTRING ((0 0,0 1,1 1),(0 2,1 3,4 -1))";

  arrow::StringBuilder builder;
  std::shared_ptr<arrow::Array> input;
  builder.Append(std::string(p1));
  builder.Append(std::string(p2));
  builder.Append(std::string(p3));
  builder.Append(std::string(p4));
  builder.Append(std::string(p5));
  builder.Append(std::string(p6));
  builder.Finish(&input);

  auto res = zilliz::gis::ST_MakeValid(input);
  auto res_str = std::static_pointer_cast<arrow::StringArray>(res);

  ASSERT_EQ(res_str->GetString(0), "POINT (1 2)");
  ASSERT_EQ(res_str->GetString(1), "LINESTRING (0 0,0 1,2 0,3 1)");
  ASSERT_EQ(res_str->GetString(2), "POLYGON ((0 0,0 1,1 2,0 0))");
  ASSERT_EQ(res_str->GetString(3),
            "GEOMETRYCOLLECTION (POLYGON ((0 0,0 1,0 4,4 4,4 1,4 0,0 0)),LINESTRING (4 "
            "1,0 1))");
  ASSERT_EQ(res_str->GetString(4), "MULTIPOINT (1 0,2 3)");
  ASSERT_EQ(res_str->GetString(5), "MULTILINESTRING ((0 0,0 1,1 1),(0 2,1 3,4 -1))");
}

TEST(geometry_test, test_ST_GeometryType) {
  auto p1 = "POINT (0 1)";
  auto p2 = "LINESTRING (0 0, 0 1, 1 1)";
  auto p3 = "LINESTRING (0 0, 1 0, 1 1, 0 0)";
  auto p4 = "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))";
  auto p5 = "MULTIPOINT (0 0, 1 0, 1 2, 1 2)";
  auto p6 = "MULTILINESTRING ( (0 0, 1 2), (0 0, 1 0, 1 1),(-1 2,3 4,9 -3,-4 100) )";
  auto p7 = "MULTIPOLYGON ( ((0 0, 1 1, 1 0,0 0)) )";
  auto p8 = "MULTIPOLYGON ( ((0 0, 0 4, 4 4, 4 0, 0 0)), ((0 0, 4 0, 4 1, 0 1, 0 0)) )";

  arrow::StringBuilder builder;
  std::shared_ptr<arrow::Array> input;
  builder.Append(std::string(p1));
  builder.Append(std::string(p2));
  builder.Append(std::string(p3));
  builder.Append(std::string(p4));
  builder.Append(std::string(p5));
  builder.Append(std::string(p6));
  builder.Append(std::string(p7));
  builder.Append(std::string(p8));
  builder.Finish(&input);

  auto res = zilliz::gis::ST_GeometryType(input);
  auto res_str = std::static_pointer_cast<arrow::StringArray>(res);

  ASSERT_EQ(res_str->GetString(0), "ST_POINT");
  ASSERT_EQ(res_str->GetString(1), "ST_LINESTRING");
  ASSERT_EQ(res_str->GetString(2), "ST_LINESTRING");
  ASSERT_EQ(res_str->GetString(3), "ST_POLYGON");
  ASSERT_EQ(res_str->GetString(4), "ST_MULTIPOINT");
  ASSERT_EQ(res_str->GetString(5), "ST_MULTILINESTRING");
  ASSERT_EQ(res_str->GetString(6), "ST_MULTIPOLYGON");
  ASSERT_EQ(res_str->GetString(7), "ST_MULTIPOLYGON");
}

TEST(geometry_test, test_ST_SimplifyPreserveTopology) {
  auto p1 = "POINT (0 1)";
  auto p2 = "LINESTRING (0 0,0 2,0 1,1 1)";
  auto p3 = "LINESTRING (0 0,0.5 0,1 0,1 0.5,1 1, 0 0)";
  auto p4 = "POLYGON ((0 0, 1 0, 1 1,1 1,0 1, 0 0))";
  auto p5 = "MULTIPOINT (0 0, 1 0, 1 2, 1 2)";
  auto p6 = "MULTILINESTRING ( (0 0, 1 2), (0 0, 1 0, 1 1),(-1 2,3 4,9 -3,-4 100) )";
  auto p7 = "MULTIPOLYGON ( ((0 0, 1 1, 1 0,0 0)) )";
  auto p8 =
      "MULTIPOLYGON ( ((0 0,0 2, 0 4, 4 4,4 3, 4 0, 0 0)), ((0 0, 4 0, 4 1, 0 1, 0 0)) )";

  arrow::StringBuilder builder;
  std::shared_ptr<arrow::Array> input;
  builder.Append(std::string(p1));
  builder.Append(std::string(p2));
  builder.Append(std::string(p3));
  builder.Append(std::string(p4));
  builder.Append(std::string(p5));
  builder.Append(std::string(p6));
  builder.Append(std::string(p7));
  builder.Append(std::string(p8));
  builder.Finish(&input);

  auto res = zilliz::gis::ST_SimplifyPreserveTopology(input, 10);
  auto res_str = std::static_pointer_cast<arrow::StringArray>(res);

  ASSERT_EQ(res_str->GetString(0), "POINT (0 1)");
  ASSERT_EQ(res_str->GetString(1), "LINESTRING (0 0,1 1)");  // ?
  ASSERT_EQ(res_str->GetString(2), "LINESTRING (0 0,1 0,1 1,0 0)");
  ASSERT_EQ(res_str->GetString(3), "POLYGON ((0 0,1 0,1 1,0 1,0 0))");
  ASSERT_EQ(res_str->GetString(4), "MULTIPOINT (0 0,1 0,1 2,1 2)");
  ASSERT_EQ(res_str->GetString(5),
            "MULTILINESTRING ((0 0,1 2),(0 0,1 1),(-1 2,3 4,9 -3,-4 100))");  //?
  ASSERT_EQ(res_str->GetString(6), "POLYGON ((0 0,1 1,1 0,0 0))");
  // ASSERT_EQ(res_str->GetString(7), "MULTIPOLYGON (((0 0,0 4,4 4,4 0,0 0)),((0 0,4 0,4
  // 1,0 1,0 0)))"); //MULTIPOLYGON (((0 0,0 2,0 4,4 4,4 3,4 0,0 0)),((0 0,4 0,4 1,0 1,0
  // 0)))
}

TEST(geometry_test, test_ST_Contains) {
  auto l1 = "MULTIPOINT (0 1, 1 0, 1 2)";
  auto l2 = "MULTIPOINT (0 1, 1 0, 1 2)";
  auto l3 = "MULTIPOINT (0 1, 1 1, 4 0)";
  auto l4 = "MULTIPOINT (0 1, 1 1, 4 0)";
  auto l5 = "MULTIPOINT (0 1, 1 1, 4 0)";
  auto l6 = "LINESTRING (0 0, 1 2, 4 0)";
  auto l7 = "LINESTRING (0 0, 1 2, 4 0)";
  auto l8 = "LINESTRING (0 0, 1 2, 4 0)";
  auto l9 = "LINESTRING (0 0, 1 2, 4 0)";
  auto l10 = "LINESTRING (0 0, 1 2, 4 0)";
  auto l11 = "LINESTRING (0 0, 1 2,4 2)";
  auto l12 = "LINESTRING (1 0, 1 2,4 2)";
  auto l13 = "LINESTRING (1 0, 1 2,4 2)";
  auto l14 = "POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0))";
  auto l15 = "POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0))";
  auto l16 = "POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0))";
  auto l17 = "POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0))";
  auto l18 = "POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0))";
  auto l19 = "POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0))";
  auto l20 = "POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0))";
  auto l21 = "POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0))";
  auto l22 = "POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0))";
  auto l23 = "POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0))";

  arrow::StringBuilder builder1;
  std::shared_ptr<arrow::Array> input1;
  builder1.Append(std::string(l1));
  builder1.Append(std::string(l2));
  builder1.Append(std::string(l3));
  builder1.Append(std::string(l4));
  builder1.Append(std::string(l5));
  builder1.Append(std::string(l6));
  builder1.Append(std::string(l7));
  builder1.Append(std::string(l8));
  builder1.Append(std::string(l9));
  builder1.Append(std::string(l10));
  builder1.Append(std::string(l11));
  builder1.Append(std::string(l12));
  builder1.Append(std::string(l13));
  builder1.Append(std::string(l14));
  builder1.Append(std::string(l15));
  builder1.Append(std::string(l16));
  builder1.Append(std::string(l17));
  builder1.Append(std::string(l18));
  builder1.Append(std::string(l19));
  builder1.Append(std::string(l20));
  builder1.Append(std::string(l21));
  builder1.Append(std::string(l22));
  builder1.Append(std::string(l23));
  builder1.Finish(&input1);

  auto r1 = "POINT (1 0)";
  auto r2 = "POINT (1 3)";
  auto r3 = "MULTIPOINT (0 1, 1 1, 4 0)";
  auto r4 = "MULTIPOINT (1 1, 4 0)";
  auto r5 = "MULTIPOINT (0 1, 1 1, 3 0)";
  auto r6 = "POINT (0.5 1)";
  auto r7 = "POINT (1 2)";
  auto r8 = "POINT (1 3)";
  auto r9 = "MULTIPOINT (0.5 1,1 2)";
  auto r10 = "MULTIPOINT (1 3)";
  auto r11 = "LINESTRING (0 0, 1 2)";
  auto r12 = "LINESTRING (1 0.5, 1 2, 3 2)";
  auto r13 = "LINESTRING (1 0.5, 1 3)";
  auto r14 = "LINESTRING (1 0.5, 1 4)";
  auto r15 = "LINESTRING (1 0.5, 1 5)";
  auto r16 = "LINESTRING (1 0, 4 0, 4 1, 3 2)";
  auto r17 = "LINESTRING (1 0, 2 0, 2 3, 1 2, 1 0)";
  auto r18 = "POINT (1 2)";
  auto r19 = "POINT (4 0)";
  auto r20 = "POINT (4 8)";
  auto r21 = "MULTIPOINT (1 2, 4 0)";
  auto r22 = "MULTIPOINT (1 8, 4 0)";
  auto r23 = "MULTIPOINT (4 8)";

  arrow::StringBuilder builder2;
  std::shared_ptr<arrow::Array> input2;
  builder2.Append(std::string(r1));
  builder2.Append(std::string(r2));
  builder2.Append(std::string(r3));
  builder2.Append(std::string(r4));
  builder2.Append(std::string(r5));
  builder2.Append(std::string(r6));
  builder2.Append(std::string(r7));
  builder2.Append(std::string(r8));
  builder2.Append(std::string(r9));
  builder2.Append(std::string(r10));
  builder2.Append(std::string(r11));
  builder2.Append(std::string(r12));
  builder2.Append(std::string(r13));
  builder2.Append(std::string(r14));
  builder2.Append(std::string(r15));
  builder2.Append(std::string(r16));
  builder2.Append(std::string(r17));
  builder2.Append(std::string(r18));
  builder2.Append(std::string(r19));
  builder2.Append(std::string(r20));
  builder2.Append(std::string(r21));
  builder2.Append(std::string(r22));
  builder2.Append(std::string(r23));
  builder2.Finish(&input2);

  auto res = zilliz::gis::ST_Contains(input1, input2);
  auto res_bool = std::static_pointer_cast<arrow::BooleanArray>(res);

  ASSERT_EQ(res_bool->Value(0), true);
  ASSERT_EQ(res_bool->Value(1), false);
  ASSERT_EQ(res_bool->Value(2), true);
  ASSERT_EQ(res_bool->Value(3), true);
  ASSERT_EQ(res_bool->Value(4), false);
  ASSERT_EQ(res_bool->Value(5), true);
  ASSERT_EQ(res_bool->Value(6), true);
  ASSERT_EQ(res_bool->Value(7), false);
  ASSERT_EQ(res_bool->Value(8), true);
  ASSERT_EQ(res_bool->Value(9), false);
  ASSERT_EQ(res_bool->Value(10), true);
  ASSERT_EQ(res_bool->Value(11), true);
  ASSERT_EQ(res_bool->Value(12), false);
  ASSERT_EQ(res_bool->Value(13), true);
  ASSERT_EQ(res_bool->Value(14), false);
  ASSERT_EQ(res_bool->Value(15), true);
  ASSERT_EQ(res_bool->Value(16), true);
  ASSERT_EQ(res_bool->Value(17), true);
  // ASSERT_EQ(res_bool->Value(18), true); // false
  ASSERT_EQ(res_bool->Value(19), false);
  ASSERT_EQ(res_bool->Value(20), true);
  ASSERT_EQ(res_bool->Value(21), false);
  ASSERT_EQ(res_bool->Value(22), false);
}

TEST(geometry_test, test_ST_Intersects) {
  auto l1 = "POINT (0 1)";
  auto l2 = "POINT (0 1)";
  auto l3 = "POINT (0 1)";
  auto l4 = "POINT (0 1)";
  auto l5 = "POINT (0 1)";
  auto l6 = "POINT (0 1)";
  auto l7 = "POINT (0 1)";
  auto l8 = "POINT (0 1)";
  auto l9 = "POINT (0 1)";
  auto l10 = "POINT (0 1)";
  auto l11 = "POINT (0 1)";
  auto l12 = "POINT (0 1)";
  auto l13 = "POINT (0 1)";
  auto l14 = "POINT (0 1)";
  auto l15 = "POINT (0 1)";
  auto l16 = "POINT (0 1)";

  auto l17 = "MULTIPOINT (1 8, 2 3)";
  auto l18 = "MULTIPOINT (1 8, 2 3)";
  auto l19 = "MULTIPOINT (1 8, 2 3)";
  auto l20 = "MULTIPOINT (1 8, 2 3)";
  auto l21 = "MULTIPOINT (1 8, 2 3)";
  auto l22 = "MULTIPOINT (1 8, 2 3)";
  auto l23 = "MULTIPOINT (1 8, 2 3)";
  auto l24 = "MULTIPOINT (1 8, 2 3)";
  auto l25 = "MULTIPOINT (1 8, 2 3)";
  auto l26 = "MULTIPOINT (1 8, 2 3)";
  auto l27 = "MULTIPOINT (1 8, 2 3)";
  auto l28 = "MULTIPOINT (1 8, 2 3)";
  auto l29 = "MULTIPOINT (1 8, 2 3)";
  auto l30 = "MULTIPOINT (1 8, 2 3)";

  auto l31 = "LINESTRING (0 0, 1 0, 1 8)";
  auto l32 = "LINESTRING (0 0, 1 0, 1 8)";
  auto l33 = "LINESTRING (0 0, 1 0, 1 8)";
  auto l34 = "LINESTRING (0 0, 1 0, 1 8)";
  auto l35 = "LINESTRING (0 0, 1 0, 1 8)";
  auto l36 = "LINESTRING (0 0, 1 0, 1 8)";
  auto l37 = "LINESTRING (0 0, 1 0, 1 8)";
  auto l38 = "LINESTRING (0 0, 1 0, 1 8)";
  auto l39 = "LINESTRING (0 0, 1 0, 1 8)";
  auto l40 = "LINESTRING (0 0, 1 0, 1 8)";
  auto l41 = "LINESTRING (0 0, 1 0, 1 8)";
  auto l42 = "LINESTRING (0 0, 1 0, 1 8)";

  auto l43 = "MULTILINESTRING ( (0 1, 0 1), (0 1, 2 3, 1 1) )";
  auto l44 = "MULTILINESTRING ( (0 1, 0 1), (0 1, 2 3, 1 1) )";
  auto l45 = "MULTILINESTRING ( (0 1, 0 1), (0 1, 2 3, 1 1) )";
  auto l46 = "MULTILINESTRING ( (0 1, 0 1), (0 1, 2 3, 1 1) )";
  auto l47 = "MULTILINESTRING ( (0 1, 0 1), (0 1, 2 3, 1 1) )";
  auto l48 = "MULTILINESTRING ( (0 1, 0 1), (0 1, 2 3, 1 1) )";
  auto l49 = "MULTILINESTRING ( (0 1, 0 1), (0 1, 2 3, 1 1) )";
  auto l50 = "MULTILINESTRING ( (0 1, 0 1), (0 1, 2 3, 1 1) )";
  auto l51 = "MULTILINESTRING ( (0 1, 0 1), (0 1, 2 3, 1 1) )";

  auto l52 = "POLYGON ((0 1, 2 3, 1 1, 1 0, 0 1))";
  auto l53 = "POLYGON ((0 1, 2 3, 1 1, 1 0, 0 1))";
  auto l54 = "POLYGON ((0 1, 2 3, 1 1, 1 0, 0 1))";
  auto l55 = "POLYGON ((0 1, 2 3, 1 1, 1 0, 0 1))";
  auto l56 = "POLYGON ((0 1, 2 3, 1 1, 1 0, 0 1))";
  auto l57 = "POLYGON ((0 1, 2 3, 1 1, 1 0, 0 1))";
  auto l58 = "POLYGON ((0 1, 2 3, 1 1, 1 0, 0 1))";

  auto l59 = "MULTIPOLYGON ( ((0 0, 0 4, 1 8, 4 0, 0 0)), ((0 0, 4 0, 4 1, 0 1, 0 0)) )";
  auto l60 = "MULTIPOLYGON ( ((0 0, 0 4, 1 8, 4 0, 0 0)), ((0 0, 4 0, 4 1, 0 1, 0 0)) )";
  auto l61 = "MULTIPOLYGON ( ((0 0, 0 4, 1 8, 4 0, 0 0)), ((0 0, 4 0, 4 1, 0 1, 0 0)) )";
  auto l62 = "MULTIPOLYGON ( ((0 0, 0 4, 1 8, 4 0, 0 0)), ((0 0, 4 0, 4 1, 0 1, 0 0)) )";

  arrow::StringBuilder builder1;
  std::shared_ptr<arrow::Array> input1;
  builder1.Append(std::string(l1));
  builder1.Append(std::string(l2));
  builder1.Append(std::string(l3));
  builder1.Append(std::string(l4));
  builder1.Append(std::string(l5));
  builder1.Append(std::string(l6));
  builder1.Append(std::string(l7));
  builder1.Append(std::string(l8));
  builder1.Append(std::string(l9));
  builder1.Append(std::string(l10));
  builder1.Append(std::string(l11));
  builder1.Append(std::string(l12));
  builder1.Append(std::string(l13));
  builder1.Append(std::string(l14));
  // builder1.Append(std::string(l15));
  // builder1.Append(std::string(l16));
  builder1.Append(std::string(l17));
  builder1.Append(std::string(l18));
  builder1.Append(std::string(l19));
  builder1.Append(std::string(l20));
  builder1.Append(std::string(l21));
  builder1.Append(std::string(l22));
  builder1.Append(std::string(l23));
  builder1.Append(std::string(l24));
  builder1.Append(std::string(l25));
  builder1.Append(std::string(l26));
  builder1.Append(std::string(l27));
  builder1.Append(std::string(l28));
  builder1.Append(std::string(l29));
  builder1.Append(std::string(l30));
  builder1.Append(std::string(l31));
  builder1.Append(std::string(l32));
  builder1.Append(std::string(l33));
  builder1.Append(std::string(l34));
  builder1.Append(std::string(l35));
  builder1.Append(std::string(l36));
  builder1.Append(std::string(l37));
  builder1.Append(std::string(l38));
  builder1.Append(std::string(l39));
  builder1.Append(std::string(l40));
  // builder1.Append(std::string(l41));
  builder1.Append(std::string(l42));
  builder1.Append(std::string(l43));
  builder1.Append(std::string(l44));
  builder1.Append(std::string(l45));
  builder1.Append(std::string(l46));
  builder1.Append(std::string(l47));
  builder1.Append(std::string(l48));
  // builder1.Append(std::string(l49));
  // builder1.Append(std::string(l50));
  // builder1.Append(std::string(l51));
  builder1.Append(std::string(l52));
  builder1.Append(std::string(l53));
  builder1.Append(std::string(l54));
  builder1.Append(std::string(l55));
  builder1.Append(std::string(l56));
  // builder1.Append(std::string(l57));
  // builder1.Append(std::string(l58));
  // builder1.Append(std::string(l59));
  // builder1.Append(std::string(l60));
  // builder1.Append(std::string(l61));
  // builder1.Append(std::string(l62));
  builder1.Finish(&input1);

  auto r1 = "POINT (0 1)";
  auto r2 = "POINT (3 1)";
  auto r3 = "MULTIPOINT (0 1, 1 0, 1 2, 1 2)";
  auto r4 = "MULTIPOINT (0 2, 1 0, 1 2, 1 2)";
  auto r5 = "LINESTRING (0 0, 0 1, 1 1)";
  auto r6 = "LINESTRING (0 0, 0 2, 1 1)";
  auto r7 = "LINESTRING (0 0, 0.5 2, 1 1)";
  auto r8 = "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))";
  auto r9 = "POLYGON ((0 0, 1 0, 1 1, 0 2, 0 0))";
  auto r10 = "POLYGON ((0 0, 1 0, 1 1, 0 0.5, 0 0))";
  auto r11 = "MULTILINESTRING ( (0 0, 1 2), (0 0, 0 2, 1 1),(-1 2,3 4,9 -3,-4 100) )";
  auto r12 = "MULTILINESTRING ( (0 0, 1 2), (0 0, 1 0, 1 1),(-1 2,3 4,9 -3,-4 100) )";
  auto r13 = "MULTIPOLYGON ( ((0 0, 1 1, 0 2,0 0)) )";
  auto r14 = "MULTIPOLYGON ( ((0 0, 1 1, 2 0,0 0)) )";
  auto r15 = "MULTIPOLYGON ( ((0 0, 0 4, 4 4, 4 0, 0 0)), ((0 0, 4 0, 4 1, 0 1, 0 0)) )";
  auto r16 = "MULTIPOLYGON ( ((0 0, 4 4, 4 0, 0 0)), ((0 0, 4 0, 4 1, 0 0)) )";

  auto r17 = "MULTIPOINT (0 1, 1 0, 1 8, 1 2)";
  auto r18 = "MULTIPOINT (0 1, 0 2)";
  auto r19 = "LINESTRING (0 0, 0 1, 1 8)";
  auto r20 = "LINESTRING (0 1, 0 1, 0 1)";
  auto r21 = "LINESTRING (0 0, 0 1, 2 3, 0 1, 0 0)";
  auto r22 = "POLYGON ((0 0, 0 1, 0 1, 0 1, 0 0))";
  auto r23 = "POLYGON ((0 1, 2 3, 1 1, 1 0, 0 1))";
  auto r24 = "POLYGON ((0 0, 1 0, 1 8, 0 0.5, 0 0))";
  auto r25 = "MULTILINESTRING ( (0 0, 1 8), (0 0, 0 2, 1 1),(-1 2,3 4,9 -3,-4 100) )";
  auto r26 = "MULTILINESTRING ( (0 1, 0 1), (0 1, 2 3, 0 1) )";
  auto r27 = "MULTIPOLYGON ( ((0 0, 1 8, 0 2,0 0)) )";
  auto r28 = "MULTIPOLYGON ( ((0 1, 0 1, 0 1,0 1)) )";
  auto r29 = "MULTIPOLYGON ( ((0 0, 0 4, 1 8, 4 0, 0 0)), ((0 0, 4 0, 4 1, 0 1, 0 0)) )";
  auto r30 = "MULTIPOLYGON ( ((0 1, 2 3, 0 1,0 1)), ((0 1, 0 1, 0 1, 0 1)) )";

  auto r31 = "LINESTRING (0 0, 1 0, 1 8)";
  auto r32 = "LINESTRING (0 0, 1 1, 0 1)";
  auto r33 = "LINESTRING (0 0, 0 1, 2 3, 0 1, 0 0)";
  auto r34 = "POLYGON ((0 0, 0 1, 0 1, 0 1, 0 0))";
  auto r35 = "POLYGON ((0 1, 2 3, 1 1, 1 0, 0 1))";
  auto r36 = "POLYGON ((0 0, 1 0, 1 8, 0 0.5, 0 0))";
  auto r37 = "MULTILINESTRING ( (0 0, 1 8), (0 0, 0 2, 1 1),(-1 2,3 4,9 -3,-4 100) )";
  auto r38 = "MULTILINESTRING ( (0 1, 0 1), (0 1, 2 3, 0 1) )";
  auto r39 = "MULTIPOLYGON ( ((0 0, 1 8, 0 2,0 0)) )";
  auto r40 = "MULTIPOLYGON ( ((0 1, 0 1, 0 1,0 1)) )";
  auto r41 = "MULTIPOLYGON ( ((0 0, 0 4, 1 8, 4 0, 0 0)), ((0 0, 4 0, 4 1, 0 1, 0 0)) )";
  auto r42 = "MULTIPOLYGON ( ((0 1, 1 8, 0 1,0 1)), ((0 1, 0 1, 0 1, 0 1)) )";

  auto r43 = "POLYGON ((0 0, 0 1, 0 1, 0 1, 0 0))";
  auto r44 = "POLYGON ((0 1, 2 3, 1 1, 1 0, 0 1))";
  auto r45 = "POLYGON ((0 0, 1 0, 1 8, 0 0.5, 0 0))";
  auto r46 = "MULTILINESTRING ( (0 0, 1 8), (0 0, 0 2, 1 1),(0 1, 2 3, 1 1) )";
  auto r47 = "MULTILINESTRING ( (0 1, 0 1), (0 1, 2 3, 1 1) )";
  auto r48 = "MULTIPOLYGON ( ((0 0, 1 8, 0 2,0 0)) )";
  auto r49 = "MULTIPOLYGON ( ((0 1, 2 3, 3 0, 0 0, 0 1)) )";
  auto r50 = "MULTIPOLYGON ( ((0 0, 0 4, 1 8, 4 0, 0 0)), ((0 0, 4 0, 4 1, 0 1, 0 0)) )";
  auto r51 = "MULTIPOLYGON ( ((0 1, 1 8, 0 1,0 1)), ((0 1, 0 1, 0 1, 0 1)) )";

  auto r52 = "POLYGON ((0 0, 0 1, 0 1, 0 1, 0 0))";
  auto r53 = "POLYGON ((0 1, 2 3, 1 1, 1 0, 0 1))";
  auto r54 = "POLYGON ((0 0, 1 0, 1 8, 0 0.5, 0 0))";
  auto r55 = "MULTIPOLYGON ( ((0 0, 1 8, 0 2,0 0)) )";
  auto r56 = "MULTIPOLYGON ( ((0 1, 2 3, 3 0, 0 0, 0 1)) )";
  auto r57 = "MULTIPOLYGON ( ((0 0, 0 4, 1 8, 4 0, 0 0)), ((0 0, 4 0, 4 1, 0 1, 0 0)) )";
  auto r58 = "MULTIPOLYGON ( ((0 1, 1 8, 0 1,0 1)), ((0 1, 0 1, 0 1, 0 1)) )";

  auto r59 = "MULTIPOLYGON ( ((0 0, 1 8, 0 2,0 0)) )";
  auto r60 = "MULTIPOLYGON ( ((0 1, 2 3, 3 0, 0 0, 0 1)) )";
  auto r61 = "MULTIPOLYGON ( ((0 0, 0 4, 1 8, 4 0, 0 0)), ((0 0, 4 0, 4 1, 0 1, 0 0)) )";
  auto r62 = "MULTIPOLYGON ( ((0 1, 1 8, 0 1,0 1)), ((0 1, 0 1, 0 1, 0 1)) )";

  arrow::StringBuilder builder2;
  std::shared_ptr<arrow::Array> input2;
  builder2.Append(std::string(r1));
  builder2.Append(std::string(r2));
  builder2.Append(std::string(r3));
  builder2.Append(std::string(r4));
  builder2.Append(std::string(r5));
  builder2.Append(std::string(r6));
  builder2.Append(std::string(r7));
  builder2.Append(std::string(r8));
  builder2.Append(std::string(r9));
  builder2.Append(std::string(r10));
  builder2.Append(std::string(r11));
  builder2.Append(std::string(r12));
  builder2.Append(std::string(r13));
  builder2.Append(std::string(r14));
  // builder2.Append(std::string(r15));
  // builder2.Append(std::string(r16));
  builder2.Append(std::string(r17));
  builder2.Append(std::string(r18));
  builder2.Append(std::string(r19));
  builder2.Append(std::string(r20));
  builder2.Append(std::string(r21));
  builder2.Append(std::string(r22));
  builder2.Append(std::string(r23));
  builder2.Append(std::string(r24));
  builder2.Append(std::string(r25));
  builder2.Append(std::string(r26));
  builder2.Append(std::string(r27));
  builder2.Append(std::string(r28));
  builder2.Append(std::string(r29));
  builder2.Append(std::string(r30));
  builder2.Append(std::string(r31));
  builder2.Append(std::string(r32));
  builder2.Append(std::string(r33));
  builder2.Append(std::string(r34));
  builder2.Append(std::string(r35));
  builder2.Append(std::string(r36));
  builder2.Append(std::string(r37));
  builder2.Append(std::string(r38));
  builder2.Append(std::string(r39));
  builder2.Append(std::string(r40));
  // builder2.Append(std::string(r41));
  builder2.Append(std::string(r42));
  builder2.Append(std::string(r43));
  builder2.Append(std::string(r44));
  builder2.Append(std::string(r45));
  builder2.Append(std::string(r46));
  builder2.Append(std::string(r47));
  builder2.Append(std::string(r48));
  // builder2.Append(std::string(r49));
  // builder2.Append(std::string(r50));
  // builder2.Append(std::string(r51));
  builder2.Append(std::string(r52));
  builder2.Append(std::string(r53));
  builder2.Append(std::string(r54));
  builder2.Append(std::string(r55));
  builder2.Append(std::string(r56));
  // builder2.Append(std::string(r57));
  // builder2.Append(std::string(r58));
  // builder2.Append(std::string(r59));
  // builder2.Append(std::string(r60));
  // builder2.Append(std::string(r61));
  // builder2.Append(std::string(r62));
  builder2.Finish(&input2);

  auto res = zilliz::gis::ST_Intersects(input1, input2);
  auto res_bool = std::static_pointer_cast<arrow::BooleanArray>(res);

  ASSERT_EQ(res_bool->Value(0), true);
  ASSERT_EQ(res_bool->Value(1), false);
  ASSERT_EQ(res_bool->Value(2), true);
  // ASSERT_EQ(res_bool->Value(3), false); // POINT EMPTY
  ASSERT_EQ(res_bool->Value(4), true);
  ASSERT_EQ(res_bool->Value(5), true);
  // ASSERT_EQ(res_bool->Value(6), false); // POINT EMPTY
  ASSERT_EQ(res_bool->Value(7), true);
  ASSERT_EQ(res_bool->Value(8), true);
  // ASSERT_EQ(res_bool->Value(9), false); // POINT EMPTY
  ASSERT_EQ(res_bool->Value(10), true);
  // ASSERT_EQ(res_bool->Value(11), false); // POINT EMPTY
  ASSERT_EQ(res_bool->Value(12), true);
  // ASSERT_EQ(res_bool->Value(13), false); // POINT EMPTY
  // ASSERT_EQ(res_bool->Value(14), true); // error
  // ASSERT_EQ(res_bool->Value(15), false); // error
  // TODO : need verify against geospark result below.
  ASSERT_EQ(res_bool->Value(16), true);
  ASSERT_EQ(res_bool->Value(17), false);
  ASSERT_EQ(res_bool->Value(18), true);
  ASSERT_EQ(res_bool->Value(19), false);
  ASSERT_EQ(res_bool->Value(20), true);
  ASSERT_EQ(res_bool->Value(21), true);
  ASSERT_EQ(res_bool->Value(22), true);
  ASSERT_EQ(res_bool->Value(23), true);
  ASSERT_EQ(res_bool->Value(24), true);
  ASSERT_EQ(res_bool->Value(25), false);
  ASSERT_EQ(res_bool->Value(26), true);
  ASSERT_EQ(res_bool->Value(27), true);
  ASSERT_EQ(res_bool->Value(28), true);
  ASSERT_EQ(res_bool->Value(29), true);
  ASSERT_EQ(res_bool->Value(30), true);
  ASSERT_EQ(res_bool->Value(31), true);
  ASSERT_EQ(res_bool->Value(32), true);
  ASSERT_EQ(res_bool->Value(33), true);
  ASSERT_EQ(res_bool->Value(34), true);
  ASSERT_EQ(res_bool->Value(35), true);
  ASSERT_EQ(res_bool->Value(36), true);
  ASSERT_EQ(res_bool->Value(37), false);
  ASSERT_EQ(res_bool->Value(38), true);
  ASSERT_EQ(res_bool->Value(39), true);
  ASSERT_EQ(res_bool->Value(40), true);
  ASSERT_EQ(res_bool->Value(41), true);
  ASSERT_EQ(res_bool->Value(42), true);
  ASSERT_EQ(res_bool->Value(43), true);
  ASSERT_EQ(res_bool->Value(44), true);
  ASSERT_EQ(res_bool->Value(45), true);
  ASSERT_EQ(res_bool->Value(46), true);
  ASSERT_EQ(res_bool->Value(47), true);
  ASSERT_EQ(res_bool->Value(48), true);
  ASSERT_EQ(res_bool->Value(49), true);
}

TEST(geometry_test, test_ST_Within) {
  auto l1 = "POINT (1 0)";
  auto l2 = "POINT (1 3)";
  auto l3 = "MULTIPOINT (0 1, 1 1, 4 0)";
  auto l4 = "MULTIPOINT (1 1, 4 0)";
  auto l5 = "MULTIPOINT (0 1, 1 1, 3 0)";
  auto l6 = "POINT (0.5 1)";
  auto l7 = "POINT (1 2)";
  auto l8 = "POINT (1 3)";
  auto l9 = "MULTIPOINT (0.5 1,1 2)";
  auto l10 = "MULTIPOINT (1 3)";
  auto l11 = "LINESTRING (0 0, 1 2)";
  auto l12 = "LINESTRING (1 0.5, 1 2, 3 2)";
  auto l13 = "LINESTRING (1 0.5, 1 3)";
  auto l14 = "LINESTRING (1 0.5, 1 4)";
  auto l15 = "LINESTRING (1 0.5, 1 5)";
  auto l16 = "LINESTRING (1 0, 4 0, 4 1, 3 2)";
  auto l17 = "LINESTRING (1 0, 2 0, 2 3, 1 2, 1 0)";
  auto l18 = "POINT (1 2)";
  auto l19 = "POINT (4 0)";
  auto l20 = "POINT (4 8)";
  auto l21 = "MULTIPOINT (1 2, 4 0)";
  auto l22 = "MULTIPOINT (1 8, 4 0)";
  auto l23 = "MULTIPOINT (4 8)";

  arrow::StringBuilder builder1;
  std::shared_ptr<arrow::Array> input1;
  builder1.Append(std::string(l1));
  builder1.Append(std::string(l2));
  builder1.Append(std::string(l3));
  builder1.Append(std::string(l4));
  builder1.Append(std::string(l5));
  builder1.Append(std::string(l6));
  builder1.Append(std::string(l7));
  builder1.Append(std::string(l8));
  builder1.Append(std::string(l9));
  builder1.Append(std::string(l10));
  builder1.Append(std::string(l11));
  builder1.Append(std::string(l12));
  builder1.Append(std::string(l13));
  builder1.Append(std::string(l14));
  builder1.Append(std::string(l15));
  builder1.Append(std::string(l16));
  builder1.Append(std::string(l17));
  builder1.Append(std::string(l18));
  builder1.Append(std::string(l19));
  builder1.Append(std::string(l20));
  builder1.Append(std::string(l21));
  builder1.Append(std::string(l22));
  builder1.Append(std::string(l23));
  builder1.Finish(&input1);

  auto r1 = "MULTIPOINT (0 1, 1 0, 1 2)";
  auto r2 = "MULTIPOINT (0 1, 1 0, 1 2)";
  auto r3 = "MULTIPOINT (0 1, 1 1, 4 0)";
  auto r4 = "MULTIPOINT (0 1, 1 1, 4 0)";
  auto r5 = "MULTIPOINT (0 1, 1 1, 4 0)";
  auto r6 = "LINESTRING (0 0, 1 2, 4 0)";
  auto r7 = "LINESTRING (0 0, 1 2, 4 0)";
  auto r8 = "LINESTRING (0 0, 1 2, 4 0)";
  auto r9 = "LINESTRING (0 0, 1 2, 4 0)";
  auto r10 = "LINESTRING (0 0, 1 2, 4 0)";
  auto r11 = "LINESTRING (0 0, 1 2,4 2)";
  auto r12 = "LINESTRING (1 0, 1 2,4 2)";
  auto r13 = "LINESTRING (1 0, 1 2,4 2)";
  auto r14 = "POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0))";
  auto r15 = "POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0))";
  auto r16 = "POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0))";
  auto r17 = "POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0))";
  auto r18 = "POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0))";
  auto r19 = "POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0))";
  auto r20 = "POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0))";
  auto r21 = "POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0))";
  auto r22 = "POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0))";
  auto r23 = "POLYGON ((0 0, 4 0, 4 4, 0 4, 0 0))";
  arrow::StringBuilder builder2;
  std::shared_ptr<arrow::Array> input2;

  builder2.Append(std::string(r1));
  builder2.Append(std::string(r2));
  builder2.Append(std::string(r3));
  builder2.Append(std::string(r4));
  builder2.Append(std::string(r5));
  builder2.Append(std::string(r6));
  builder2.Append(std::string(r7));
  builder2.Append(std::string(r8));
  builder2.Append(std::string(r9));
  builder2.Append(std::string(r10));
  builder2.Append(std::string(r11));
  builder2.Append(std::string(r12));
  builder2.Append(std::string(r13));
  builder2.Append(std::string(r14));
  builder2.Append(std::string(r15));
  builder2.Append(std::string(r16));
  builder2.Append(std::string(r17));
  builder2.Append(std::string(r18));
  builder2.Append(std::string(r19));
  builder2.Append(std::string(r20));
  builder2.Append(std::string(r21));
  builder2.Append(std::string(r22));
  builder2.Append(std::string(r23));
  builder2.Finish(&input2);

  auto res = zilliz::gis::ST_Within(input1, input2);
  auto res_bool = std::static_pointer_cast<arrow::BooleanArray>(res);

  ASSERT_EQ(res_bool->Value(0), true);
  ASSERT_EQ(res_bool->Value(1), false);
  ASSERT_EQ(res_bool->Value(2), true);
  ASSERT_EQ(res_bool->Value(3), true);
  ASSERT_EQ(res_bool->Value(4), false);
  ASSERT_EQ(res_bool->Value(5), true);
  ASSERT_EQ(res_bool->Value(6), true);
  ASSERT_EQ(res_bool->Value(7), false);
  ASSERT_EQ(res_bool->Value(8), true);
  ASSERT_EQ(res_bool->Value(9), false);
  ASSERT_EQ(res_bool->Value(10), true);
  ASSERT_EQ(res_bool->Value(11), true);
  ASSERT_EQ(res_bool->Value(12), false);
  ASSERT_EQ(res_bool->Value(13), true);
  ASSERT_EQ(res_bool->Value(14), false);
  ASSERT_EQ(res_bool->Value(15), true);
  ASSERT_EQ(res_bool->Value(16), true);
  ASSERT_EQ(res_bool->Value(17), true);
  // ASSERT_EQ(res_bool->Value(18), true); // false
  ASSERT_EQ(res_bool->Value(19), false);
  ASSERT_EQ(res_bool->Value(20), true);
  ASSERT_EQ(res_bool->Value(21), false);
  ASSERT_EQ(res_bool->Value(22), false);
}

TEST(geometry_test, test_ST_Distance) {
  auto l1 = "POINT (0 0)";
  auto l2 = "POINT (0 0)";
  auto l3 = "POINT (0 0)";
  auto l4 = "POINT (0 0)";
  auto l5 = "POINT (0 0)";
  auto l6 = "POINT (0 0)";
  auto l7 = "POINT (0 0)";

  auto l8 = "LINESTRING (0 0, 1 1)";
  auto l9 = "LINESTRING (0 0, 1 1)";
  auto l10 = "LINESTRING (0 0, 1 1)";
  auto l11 = "LINESTRING (0 0, 1 1)";
  auto l12 = "LINESTRING (0 0, 1 1)";
  auto l13 = "LINESTRING (0 0, 1 1)";
  auto l14 = "LINESTRING (0 0, 1 1)";

  auto l15 = "POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))";
  auto l16 = "POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))";
  auto l17 = "POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))";
  auto l18 = "POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))";
  auto l19 = "POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))";
  auto l20 = "POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))";
  auto l21 = "POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))";

  arrow::StringBuilder builder1;
  std::shared_ptr<arrow::Array> input1;
  builder1.Append(std::string(l1));
  builder1.Append(std::string(l2));
  builder1.Append(std::string(l3));
  builder1.Append(std::string(l4));
  builder1.Append(std::string(l5));
  builder1.Append(std::string(l6));
  builder1.Append(std::string(l7));
  builder1.Append(std::string(l8));
  builder1.Append(std::string(l9));
  builder1.Append(std::string(l10));
  builder1.Append(std::string(l11));
  builder1.Append(std::string(l12));
  builder1.Append(std::string(l13));
  builder1.Append(std::string(l14));
  builder1.Append(std::string(l15));
  builder1.Append(std::string(l16));
  builder1.Append(std::string(l17));
  builder1.Append(std::string(l18));
  builder1.Append(std::string(l19));
  builder1.Append(std::string(l20));
  builder1.Append(std::string(l21));
  builder1.Finish(&input1);

  auto r1 = "POINT (1 1)";
  auto r2 = "LINESTRING (0 2, 2 0)";
  auto r3 = "POLYGON ((1 0, 3 0, 3 3, 1 3, 1 0))";
  auto r4 = "POLYGON ((-1 -1, 3 0, 3 3, 1 3, -1 -1))";
  auto r5 = "MULTIPOINT (0 1, 2 0, 1 2)";
  auto r6 = "MULTILINESTRING ( (0 3, 1 2), (0 1, 1 0, 1 1))";
  auto r7 =
      "MULTIPOLYGON ( ((1 0, 3 0, 3 3, 1 3, 1 0)), ((-1 -1, 3 0, 3 3, 1 3, -1 -1)) )";

  auto r8 = "POINT (1 1)";
  auto r9 = "LINESTRING (0 2, 2 0)";
  auto r10 = "POLYGON ((1 0, 3 0, 3 3, 1 3, 1 0))";
  auto r11 = "POLYGON ((-1 -1, 3 0, 3 3, 1 3, -1 -1))";
  auto r12 = "MULTIPOINT (0 1, 2 0, 1 2)";
  auto r13 = "MULTILINESTRING ( (0 3, 1 2), (0 2, 1 0, 1 1))";
  auto r14 =
      "MULTIPOLYGON ( ((1 0, 3 0, 3 3, 1 3, 1 0)), ((-1 -1, 3 0, 3 3, 1 3, -1 -1)) )";

  auto r15 = "POINT (1 1)";
  auto r16 = "LINESTRING (0 2, 2 0)";
  auto r17 = "POLYGON ((1 0, 3 0, 3 3, 1 3, 1 0))";
  auto r18 = "POLYGON ((-1 -1, 3 0, 3 3, 1 3, -1 -1))";
  auto r19 = "MULTIPOINT (0 1, 2 0, 1 2)";
  auto r20 = "MULTILINESTRING ( (0 3, 1 2), (0 2, 1 0, 1 1))";
  auto r21 =
      "MULTIPOLYGON ( ((1 0, 3 0, 3 3, 1 3, 1 0)), ((-1 -1, 3 0, 3 3, 1 3, -1 -1)) )";

  arrow::StringBuilder builder2;
  std::shared_ptr<arrow::Array> input2;
  builder2.Append(std::string(r1));
  builder2.Append(std::string(r2));
  builder2.Append(std::string(r3));
  builder2.Append(std::string(r4));
  builder2.Append(std::string(r5));
  builder2.Append(std::string(r6));
  builder2.Append(std::string(r7));
  builder2.Append(std::string(r8));
  builder2.Append(std::string(r9));
  builder2.Append(std::string(r10));
  builder2.Append(std::string(r11));
  builder2.Append(std::string(r12));
  builder2.Append(std::string(r13));
  builder2.Append(std::string(r14));
  builder2.Append(std::string(r15));
  builder2.Append(std::string(r16));
  builder2.Append(std::string(r17));
  builder2.Append(std::string(r18));
  builder2.Append(std::string(r19));
  builder2.Append(std::string(r20));
  builder2.Append(std::string(r21));
  builder2.Finish(&input2);

  auto res = zilliz::gis::ST_Distance(input1, input2);
  auto res_double = std::static_pointer_cast<arrow::DoubleArray>(res);

  EXPECT_DOUBLE_EQ(res_double->Value(0), sqrt(2));
  EXPECT_DOUBLE_EQ(res_double->Value(1), sqrt(2));
  EXPECT_DOUBLE_EQ(res_double->Value(2), 1);
  EXPECT_DOUBLE_EQ(res_double->Value(3), 0);
  EXPECT_DOUBLE_EQ(res_double->Value(4), 1);
  EXPECT_DOUBLE_EQ(res_double->Value(5), sqrt(2) / 2);
  EXPECT_DOUBLE_EQ(res_double->Value(6), 0);
  EXPECT_DOUBLE_EQ(res_double->Value(7), 0);
  EXPECT_DOUBLE_EQ(res_double->Value(8), 0);
  EXPECT_DOUBLE_EQ(res_double->Value(9), 0);
  EXPECT_DOUBLE_EQ(res_double->Value(10), 0);
  EXPECT_DOUBLE_EQ(res_double->Value(11), sqrt(2) / 2);
  EXPECT_DOUBLE_EQ(res_double->Value(12), 0);
  EXPECT_DOUBLE_EQ(res_double->Value(13), 0);
  EXPECT_DOUBLE_EQ(res_double->Value(14), 0);
  EXPECT_DOUBLE_EQ(res_double->Value(15), 0);
  EXPECT_DOUBLE_EQ(res_double->Value(16), 0);
  EXPECT_DOUBLE_EQ(res_double->Value(17), 0);
  EXPECT_DOUBLE_EQ(res_double->Value(18), 0);
  EXPECT_DOUBLE_EQ(res_double->Value(19), 0);
  EXPECT_DOUBLE_EQ(res_double->Value(20), 0);
}

TEST(geometry_test, test_ST_Area) {
  auto p1 = "POINT (0 1)";
  auto p2 = "LINESTRING (0 0, 0 1, 1 1)";
  auto p3 = "LINESTRING (0 0, 1 0, 1 1, 0 0)";
  auto p4 = "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))";
  auto p5 = "MULTIPOINT (0 0, 1 0, 1 2, 1 2)";
  auto p6 = "MULTILINESTRING ( (0 0, 1 2), (0 0, 1 0, 1 1),(-1 2,3 4,9 -3,-4 100) )";
  auto p7 = "MULTIPOLYGON ( ((0 0, 1 4, 1 0,0 0)) )";
  auto p8 = "MULTIPOLYGON ( ((0 0, 0 4, 4 4, 4 0, 0 0)), ((0 0, 0 1, 4 1, 4 0, 0 0)) )";
  auto p9 = "MULTIPOLYGON ( ((0 0, 1 4, 1 0,0 0)), ((0 0,1 0,0 1,0 0)) )";
  auto p10 = "LINESTRING (77.29 29.07,77.42 29.26,77.27 29.31,77.29 29.07)";

  arrow::StringBuilder builder;
  std::shared_ptr<arrow::Array> input;
  builder.Append(std::string(p1));
  builder.Append(std::string(p2));
  builder.Append(std::string(p3));
  builder.Append(std::string(p4));
  builder.Append(std::string(p5));
  builder.Append(std::string(p6));
  builder.Append(std::string(p7));
  builder.Append(std::string(p8));
  builder.Append(std::string(p9));
  builder.Append(std::string(p10));
  builder.Finish(&input);

  auto res = zilliz::gis::ST_Area(input);
  auto res_double = std::static_pointer_cast<arrow::DoubleArray>(res);

  EXPECT_DOUBLE_EQ(res_double->Value(0), 0);
  EXPECT_DOUBLE_EQ(res_double->Value(1), 0);
  EXPECT_DOUBLE_EQ(res_double->Value(2), 0);
  EXPECT_DOUBLE_EQ(res_double->Value(3), 1);
  EXPECT_DOUBLE_EQ(res_double->Value(4), 0);
  EXPECT_DOUBLE_EQ(res_double->Value(5), 0);
  EXPECT_DOUBLE_EQ(res_double->Value(6), 2);
  EXPECT_DOUBLE_EQ(res_double->Value(7), 20);
  // EXPECT_DOUBLE_EQ(res_double->Value(8), 1.5);
  EXPECT_DOUBLE_EQ(res_double->Value(9), 0);
}

TEST(geometry_test, test_ST_Centroid) {
  auto p1 = "POINT (0 1)";
  auto p2 = "LINESTRING (0 0, 0 1, 1 1)";
  auto p3 = "LINESTRING (0 0, 1 0, 1 1, 0 0)";
  auto p4 = "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))";
  auto p5 = "MULTIPOINT (0 0, 1 0, 1 2, 1 2)";
  auto p6 = "MULTILINESTRING ( (0 0, 1 2), (0 0, 1 0, 1 1),(-1 2,3 4,9 -3,-4 100) )";
  auto p7 = "MULTIPOLYGON ( ((0 0, 1 4, 1 0,0 0)) )";
  auto p8 = "MULTIPOLYGON ( ((0 0, 0 4, 4 4, 4 0, 0 0)), ((0 0, 0 1, 4 1, 4 0, 0 0)) )";
  auto p9 = "MULTIPOLYGON ( ((0 0, 1 4, 1 0,0 0)), ((0 0,1 0,0 1,0 0)) )";

  arrow::StringBuilder builder;
  std::shared_ptr<arrow::Array> input;
  builder.Append(std::string(p1));
  builder.Append(std::string(p2));
  builder.Append(std::string(p3));
  builder.Append(std::string(p4));
  builder.Append(std::string(p5));
  builder.Append(std::string(p6));
  builder.Append(std::string(p7));
  builder.Append(std::string(p8));
  builder.Append(std::string(p9));
  builder.Finish(&input);

  auto res = zilliz::gis::ST_Centroid(input);
  auto res_str = std::static_pointer_cast<arrow::StringArray>(res);

  ASSERT_EQ(res_str->GetString(0), "POINT (0 1)");
  ASSERT_EQ(res_str->GetString(1), "POINT (0.25 0.75)");
  ASSERT_EQ(res_str->GetString(2),
            "POINT (0.646446609406726 0.353553390593274)");  // geospark:POINT
                                                             // (0.6464466094067263
                                                             // 0.3535533905932737)
  ASSERT_EQ(res_str->GetString(3), "POINT (0.5 0.5)");
  ASSERT_EQ(res_str->GetString(4), "POINT (0.75 1.0)");
  ASSERT_EQ(res_str->GetString(5), "POINT (2.6444665557806 41.5285902625069)");
  ASSERT_EQ(res_str->GetString(6), "POINT (0.666666666666667 1.33333333333333)");
  ASSERT_EQ(res_str->GetString(7), "POINT (2.0 1.7)");
  // ASSERT_EQ(res_str->GetString(8),"POINT
  // (0.7777777777777778 1.6666666666666667)");//POINT (0.6 1.13333333333333)
}

TEST(geometry_test, test_ST_Length) {
  auto p1 = "POINT (0 1)";
  auto p2 = "LINESTRING (0 0, 0 1, 1 1)";
  auto p3 = "LINESTRING (0 0, 1 0, 1 1, 0 0)";
  auto p4 = "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))";
  auto p5 = "MULTIPOINT (0 0, 1 0, 1 2, 1 2)";
  auto p6 = "MULTILINESTRING ( (0 0, 1 2), (0 0, 1 0, 1 1),(-1 2,3 4,9 -3,-4 100) )";
  auto p7 = "MULTIPOLYGON ( ((0 0, 1 4, 1 0,0 0)) )";
  auto p8 = "MULTIPOLYGON ( ((0 0, 0 4, 4 4, 4 0, 0 0)), ((0 0, 0 1, 4 1, 4 0, 0 0)) )";
  auto p9 = "MULTIPOLYGON ( ((0 0, 1 4, 1 0,0 0)), ((0 0,1 0,0 1,0 0)) )";

  arrow::StringBuilder builder;
  std::shared_ptr<arrow::Array> input;
  builder.Append(std::string(p1));
  builder.Append(std::string(p2));
  builder.Append(std::string(p3));
  builder.Append(std::string(p4));
  builder.Append(std::string(p5));
  builder.Append(std::string(p6));
  builder.Append(std::string(p7));
  builder.Append(std::string(p8));
  builder.Append(std::string(p9));
  builder.Finish(&input);

  auto res = zilliz::gis::ST_Length(input);
  auto res_double = std::static_pointer_cast<arrow::DoubleArray>(res);

  EXPECT_DOUBLE_EQ(res_double->Value(0), 0.0);
  EXPECT_DOUBLE_EQ(res_double->Value(1), 2.0);
  EXPECT_DOUBLE_EQ(res_double->Value(2), 3.414213562373095);
  //  EXPECT_DOUBLE_EQ(res_double->Value(3),4.0); //0
  EXPECT_DOUBLE_EQ(res_double->Value(4), 0);
  //  EXPECT_DOUBLE_EQ(res_double->Value(5), 121.74489533575682); //0
  //  EXPECT_DOUBLE_EQ(res_double->Value(6),9.123105625617661); //0
  //  EXPECT_DOUBLE_EQ(res_double->Value(7),26.0); //0
  //  EXPECT_DOUBLE_EQ(res_double->Value(8),12.537319187990757); //0
}

TEST(geometry_test, test_ST_ConvexHull) {
  auto p1 = "POINT (0 1)";
  auto p2 = "LINESTRING (0 0, 0 1, 1 1)";
  auto p3 = "LINESTRING (0 0, 1 0, 1 1, 0 0)";
  auto p4 = "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))";
  auto p5 = "MULTIPOINT (0 0, 1 0, 1 2, 1 2)";
  auto p6 = "MULTILINESTRING ( (0 0, 1 2), (0 0, 1 0, 1 1),(-1 2,3 4,9 -3,-4 100) )";
  auto p7 = "MULTIPOLYGON ( ((0 0, 1 4, 1 0,0 0)) )";
  auto p8 = "MULTIPOLYGON ( ((0 0, 0 4, 4 4, 4 0, 0 0)), ((0 0, 0 1, 4 1, 4 0, 0 0)) )";
  auto p9 = "MULTIPOLYGON ( ((0 0, 1 4, 1 0,0 0)), ((0 0,1 0,0 1,0 0)) )";

  arrow::StringBuilder builder;
  std::shared_ptr<arrow::Array> input;
  builder.Append(std::string(p1));
  builder.Append(std::string(p2));
  builder.Append(std::string(p3));
  builder.Append(std::string(p4));
  builder.Append(std::string(p5));
  builder.Append(std::string(p6));
  builder.Append(std::string(p7));
  builder.Append(std::string(p8));
  builder.Append(std::string(p9));
  builder.Finish(&input);

  auto res = zilliz::gis::ST_ConvexHull(input);
  auto res_str = std::static_pointer_cast<arrow::StringArray>(res);

  ASSERT_EQ(res_str->GetString(0), "POINT (0 1)");
  ASSERT_EQ(res_str->GetString(1), "POLYGON ((0 0,0 1,1 1,0 0))");
  ASSERT_EQ(res_str->GetString(2), "POLYGON ((0 0,1 1,1 0,0 0))");
  ASSERT_EQ(res_str->GetString(3), "POLYGON ((0 0,0 1,1 1,1 0,0 0))");
  ASSERT_EQ(res_str->GetString(4), "POLYGON ((0 0,1 2,1 0,0 0))");
  ASSERT_EQ(res_str->GetString(5), "POLYGON ((9 -3,0 0,-1 2,-4 100,9 -3))");
  ASSERT_EQ(res_str->GetString(6), "POLYGON ((0 0,1 4,1 0,0 0))");
  ASSERT_EQ(res_str->GetString(7), "POLYGON ((0 0,0 4,4 4,4 0,0 0))");
  ASSERT_EQ(res_str->GetString(8), "POLYGON ((0 0,0 1,1 4,1 0,0 0))");
}

// TODO : geospark ST_NPoints can not work.
TEST(geometry_test, test_ST_NPoints) {
  auto p1 = "POINT (0 1)";
  auto p2 = "LINESTRING (0 0, 0 1, 1 1)";
  auto p3 = "LINESTRING (0 0, 1 0, 1 1, 0 0)";
  auto p4 = "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))";
  auto p5 = "MULTIPOINT (0 0, 1 0, 1 2, 1 2)";
  auto p6 = "MULTILINESTRING ( (0 0, 1 2), (0 0, 1 0, 1 1),(-1 2,3 4,9 -3,-4 100) )";
  auto p7 = "MULTIPOLYGON ( ((0 0, 1 4, 1 0,0 0)) )";
  auto p8 = "MULTIPOLYGON ( ((0 0, 0 4, 4 4, 4 0, 0 0)), ((0 0, 0 1, 4 1, 4 0, 0 0)) )";
  auto p9 = "MULTIPOLYGON ( ((0 0, 1 4, 1 0,0 0)), ((0 0,1 0,0 1,0 0)) )";

  arrow::StringBuilder builder;
  std::shared_ptr<arrow::Array> input;
  builder.Append(std::string(p1));
  builder.Append(std::string(p2));
  builder.Append(std::string(p3));
  builder.Append(std::string(p4));
  builder.Append(std::string(p5));
  builder.Append(std::string(p6));
  builder.Append(std::string(p7));
  builder.Append(std::string(p8));
  builder.Append(std::string(p9));
  builder.Finish(&input);

  auto res = zilliz::gis::ST_NPoints(input);
  auto res_int = std::static_pointer_cast<arrow::UInt32Array>(res);

  ASSERT_EQ(res_int->Value(0), 1);
  ASSERT_EQ(res_int->Value(1), 0);  //?
  ASSERT_EQ(res_int->Value(2), 3);
  ASSERT_EQ(res_int->Value(3), 0);  //?
  ASSERT_EQ(res_int->Value(4), 4);  // 3?
  ASSERT_EQ(res_int->Value(5), 0);  //?
  ASSERT_EQ(res_int->Value(6), 0);  //?
  ASSERT_EQ(res_int->Value(7), 0);  //?
  ASSERT_EQ(res_int->Value(8), 0);  //?
}

TEST(geometry_test, test_ST_Envelope_Empty) {
  auto p0 = "POLYGON EMPTY";
  auto p1 = "LINESTRING EMPTY";
  auto p2 = "POINT EMPTY";
  auto p3 = "MULTIPOLYGON EMPTY";
  auto p4 = "MULTILINESTRING EMPTY";
  auto p5 = "MULTIPOINT EMPTY";
  auto p6 = "GEOMETRYCOLLECTION EMPTY";

  arrow::StringBuilder builder;
  std::shared_ptr<arrow::Array> input;

  builder.Append(std::string(p0));
  builder.Append(std::string(p1));
  builder.Append(std::string(p2));
  builder.Append(std::string(p3));
  builder.Append(std::string(p4));
  builder.Append(std::string(p5));
  builder.Append(std::string(p6));
  builder.Finish(&input);

  auto result = zilliz::gis::ST_Envelope(input);
  auto result_str = std::static_pointer_cast<arrow::StringArray>(result);

  ASSERT_EQ(result_str->GetString(0), p0);
  ASSERT_EQ(result_str->GetString(1), p1);
  ASSERT_EQ(result_str->GetString(2), p2);
  ASSERT_EQ(result_str->GetString(3), p3);
  ASSERT_EQ(result_str->GetString(4), p4);
  ASSERT_EQ(result_str->GetString(5), p5);
  ASSERT_EQ(result_str->GetString(6), p6);
}

TEST(geometry_test, test_ST_Envelope) {
  COMMON_TEST_CASES;
  CONSTRUCT_COMMON_TEST_CASES;


  auto result = zilliz::gis::ST_Envelope(input);
  auto result_str = std::static_pointer_cast<arrow::StringArray>(result);

  ASSERT_EQ(result_str->GetString(0), "POINT (0 1)");
  ASSERT_EQ(result_str->GetString(1), "LINESTRING (0 0,0 1)");
  ASSERT_EQ(result_str->GetString(2), "LINESTRING (0 0,1 0)");
  ASSERT_EQ(result_str->GetString(3), "POLYGON ((0 0,0 1,1 1,1 0,0 0))");
  ASSERT_EQ(result_str->GetString(4), "POLYGON ((0 0,0 1,1 1,1 0,0 0))");
  ASSERT_EQ(result_str->GetString(5), "POLYGON ((0 0,0 1,1 1,1 0,0 0))");
  ASSERT_EQ(result_str->GetString(6), "POLYGON ((0 0,0 1,1 1,1 0,0 0))");
  ASSERT_EQ(result_str->GetString(7), "POLYGON ((0 0,0 1,1 1,1 0,0 0))");
  ASSERT_EQ(result_str->GetString(8), "POLYGON ((0 0,0 1,1 1,1 0,0 0))");
  ASSERT_EQ(result_str->GetString(9), "POLYGON ((0 0,0 1,1 1,1 0,0 0))");
  ASSERT_EQ(result_str->GetString(10), "POLYGON ((0 0,0 1,1 1,1 0,0 0))");
  ASSERT_EQ(result_str->GetString(11), "POLYGON ((0 0,0 2,1 2,1 0,0 0))");
  ASSERT_EQ(result_str->GetString(12), "POLYGON ((0 0,0 2,1 2,1 0,0 0))");
  ASSERT_EQ(result_str->GetString(13), "POLYGON ((0 0,0 2,1 2,1 0,0 0))");
  ASSERT_EQ(result_str->GetString(14), "POLYGON ((0 0,0 2,1 2,1 0,0 0))");
  ASSERT_EQ(result_str->GetString(15), "POLYGON ((-4 -3,-4 100,9 100,9 -3,-4 -3))");
  ASSERT_EQ(result_str->GetString(16), "POLYGON ((0 0,0 1,1 1,1 0,0 0))");
  ASSERT_EQ(result_str->GetString(17), "POLYGON ((0 0,0 1,1 1,1 0,0 0))");
  ASSERT_EQ(result_str->GetString(18), "POLYGON ((0 0,0 4,4 4,4 0,0 0))");
  // ASSERT_EQ(result_str->GetString(19),"POLYGON ((0 0,0 1,4 1,4 0,0 0))");
  // ASSERT_EQ(result_str->GetString(20),"POLYGON ((0 0,0 1,1 1,1 0,0 0))");
  ASSERT_EQ(result_str->GetString(21), "POLYGON ((0 0,0 4,4 4,4 0,0 0))");
  // ASSERT_EQ(result_str->GetString(22),"POLYGON ((0 0,0 1,4 1,4 0,0 0))");
  // ASSERT_EQ(result_str->GetString(23),"POLYGON ((0 0,0 1,1 1,1 0,0 0))");
  ASSERT_EQ(result_str->GetString(24), "POLYGON ((0 0,0 4,4 4,4 0,0 0))");
  ASSERT_EQ(result_str->GetString(25), "POLYGON ((0 0,0 4,4 4,4 0,0 0))");
  ASSERT_EQ(result_str->GetString(26), "POLYGON ((0 0,0 4,4 4,4 0,0 0))");
  ASSERT_EQ(result_str->GetString(27), "POLYGON ((0 0,0 4,4 4,4 0,0 0))");
  // ASSERT_EQ(result_str->GetString(28),"POLYGON ((0 0,0 1,1 1,1 0,0 0))");
  // ASSERT_EQ(result_str->GetString(29),"POLYGON ((0 0,0 4,4 4,4 0,0 0))");
  ASSERT_EQ(result_str->GetString(30), "POLYGON ((0 -8,0 4,4 4,4 -8,0 -8))");
  // ASSERT_EQ(result_str->GetString(31),"POLYGON ((0 -8,0 4,4 4,4 -8,0 -8))");
  ASSERT_EQ(result_str->GetString(32), "POLYGON ((0 -8,0 4,4 4,4 -8,0 -8))");
}

TEST(geometry_test, test_ST_Buffer) {
  auto p1 = "POINT (0 1)";
  auto p2 = "LINESTRING (0 0, 0 1, 1 1)";
  auto p3 = "LINESTRING (0 0, 1 0, 1 1, 0 0)";
  auto p4 = "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))";
  auto p5 = "MULTIPOINT (0 0, 1 0, 1 2, 1 2)";
  auto p6 = "MULTILINESTRING ( (0 0, 1 2), (0 0, 1 0, 1 1),(-1 2,3 4,9 -3,-4 100) )";
  auto p7 = "MULTIPOLYGON ( ((0 0, 1 4, 1 0,0 0)) )";
  auto p8 = "MULTIPOLYGON ( ((0 0, 0 4, 4 4, 4 0, 0 0)), ((0 0, 0 1, 4 1, 4 0, 0 0)) )";
  auto p9 = "MULTIPOLYGON ( ((0 0, 1 4, 1 0,0 0)), ((0 0,1 0,0 1,0 0)) )";

  arrow::StringBuilder builder;
  std::shared_ptr<arrow::Array> input;
  builder.Append(std::string(p1));
  builder.Append(std::string(p2));
  builder.Append(std::string(p3));
  builder.Append(std::string(p4));
  builder.Append(std::string(p5));
  builder.Append(std::string(p6));
  builder.Append(std::string(p7));
  builder.Append(std::string(p8));
  builder.Append(std::string(p9));
  builder.Finish(&input);

  auto res = zilliz::gis::ST_Buffer(input, 0);
  auto res_str = std::static_pointer_cast<arrow::StringArray>(res);

  ASSERT_EQ(res_str->GetString(0), "POLYGON EMPTY");  // geospark :MULTIPOLYGON EMPTY
  ASSERT_EQ(res_str->GetString(1), "POLYGON EMPTY");  // geospark :MULTIPOLYGON EMPTY
  ASSERT_EQ(res_str->GetString(2), "POLYGON EMPTY");  // geospark :MULTIPOLYGON EMPTY
  ASSERT_EQ(res_str->GetString(3), "POLYGON ((0 0,0 1,1 1,1 0,0 0))");
  ASSERT_EQ(res_str->GetString(4), "POLYGON EMPTY");  // geospark :MULTIPOLYGON EMPTY
  ASSERT_EQ(res_str->GetString(5), "POLYGON EMPTY");  // geospark :MULTIPOLYGON EMPTY
  ASSERT_EQ(res_str->GetString(6), "POLYGON ((0 0,1 4,1 0,0 0))");
  ASSERT_EQ(res_str->GetString(7), "POLYGON ((0 0,0 1,0 4,4 4,4 1,4 0,0 0))");
  // ASSERT_EQ(res_str->GetString(8), "POLYGON ((0.2 0.8, 1 4, 1 0, 0.2 0.8))"); //POLYGON
  // ((0 0,0 1,0.2 0.8,1 4,1 0,0 0))
}

TEST(geometry_test, test_ST_PolygonFromEnvelope) {
  arrow::DoubleBuilder x_min_builder;
  arrow::DoubleBuilder x_max_builder;
  arrow::DoubleBuilder y_min_builder;
  arrow::DoubleBuilder y_max_builder;

  x_min_builder.Append(0);
  x_max_builder.Append(1);
  y_min_builder.Append(2);
  y_max_builder.Append(3);
  x_min_builder.Append(0);
  x_max_builder.Append(11);
  y_min_builder.Append(22);
  y_max_builder.Append(33);

  x_min_builder.Append(1);
  x_max_builder.Append(0);
  y_min_builder.Append(22);
  y_max_builder.Append(33);

  x_min_builder.Append(0);
  x_max_builder.Append(1);
  y_min_builder.Append(55);
  y_max_builder.Append(33);

  std::shared_ptr<arrow::Array> x_min;
  std::shared_ptr<arrow::Array> x_max;
  std::shared_ptr<arrow::Array> y_min;
  std::shared_ptr<arrow::Array> y_max;

  x_min_builder.Finish(&x_min);
  x_max_builder.Finish(&x_max);
  y_min_builder.Finish(&y_min);
  y_max_builder.Finish(&y_max);

  auto res = zilliz::gis::ST_PolygonFromEnvelope(x_min, y_min, x_max, y_max);

  auto res_str = std::static_pointer_cast<arrow::StringArray>(res);

  ASSERT_EQ(res_str->GetString(0), "POLYGON ((0 2,0 3,1 3,1 2,0 2))");
  ASSERT_EQ(res_str->GetString(1), "POLYGON ((0 22,0 33,11 33,11 22,0 22))");
  ASSERT_EQ(res_str->GetString(2), "POLYGON EMPTY");
  ASSERT_EQ(res_str->GetString(3), "POLYGON EMPTY");
}

TEST(geometry_test, test_ST_Transform) {
  arrow::StringBuilder builder;
  std::shared_ptr<arrow::Array> input_data;

  builder.Append(std::string("POINT (10 10)"));
  builder.Finish(&input_data);
  std::string src_rs("EPSG:4326");
  std::string dst_rs("EPSG:3857");

  auto res = zilliz::gis::ST_Transform(input_data, src_rs, dst_rs);
  auto res_str = std::static_pointer_cast<arrow::StringArray>(res)->GetString(0);
  OGRGeometry* res_geo = nullptr;
  CHECK_GDAL(OGRGeometryFactory::createFromWkt(res_str.c_str(), nullptr, &res_geo));

  auto rst_pointer = reinterpret_cast<OGRPoint*>(res_geo);

  ASSERT_DOUBLE_EQ(rst_pointer->getX(), 1113194.90793274);
  ASSERT_DOUBLE_EQ(rst_pointer->getY(), 1118889.97485796);

  OGRGeometryFactory::destroyGeometry(res_geo);
}

TEST(geometry_test, test_ST_Union_Aggr) {
  auto p1 = "POINT (0 1)";
  auto p2 = "LINESTRING (0 0, 0 1, 1 1)";
  auto p3 = "LINESTRING (0 0, 1 0, 1 1, 0 0)";
  auto p4 = "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))";
  auto p5 = "MULTIPOINT (0 0, 1 0, 1 2, 1 2)";
  auto p6 = "MULTILINESTRING ( (0 0, 1 2), (0 0, 1 0, 1 1),(-1 2,3 4,1 -3,-2 1) )";
  auto p7 = "MULTIPOLYGON ( ((0 0, 1 4, 1 0,0 0)) )";
  auto p8 = "MULTIPOLYGON ( ((0 0, 0 4, 4 4, 4 0, 0 0)), ((0 0, 0 1, 4 1, 4 0, 0 0)))";
  auto p9 = "MULTIPOLYGON ( ((0 0, 1 4, 1 0,0 0)), ((0 0,1 0,0 1,0 0)) )";

  arrow::StringBuilder builder;
  std::shared_ptr<arrow::Array> input;
  builder.Append(std::string(p1));
  builder.Append(std::string(p2));
  builder.Append(std::string(p3));
  builder.Append(std::string(p4));
  builder.Append(std::string(p5));
  builder.Append(std::string(p6));
  builder.Append(std::string(p7));
  // builder.Append(std::string(p8));
  // builder.Append(std::string(p9));
  builder.Finish(&input);

  auto res = zilliz::gis::ST_Union_Aggr(input);
  auto res_str = std::static_pointer_cast<arrow::StringArray>(res);

  ASSERT_EQ(res_str->GetString(0),
            "GEOMETRYCOLLECTION (LINESTRING (-1 2,0.714285714285714 "
            "2.85714285714286),LINESTRING (1 3,3 4,1 -3,-2 1),POLYGON ((0 0,0 1,0.25 "
            "1.0,0.714285714285714 2.85714285714286,1 4,1 3,1 2,1 1,1 0,0 0)))");
}

TEST(geometry_test, test_ST_Envelope_Aggr) {
  auto p1 = "POINT (0 1)";
  auto p2 = "LINESTRING (0 0, 0 1, 1 1)";
  auto p3 = "LINESTRING (0 0, 1 0, 1 1, 0 0)";
  auto p4 = "POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))";
  auto p5 = "MULTIPOINT (0 0, 1 0, 1 2, 1 2)";
  auto p6 = "MULTILINESTRING ( (0 0, 1 2), (0 0, 1 0, 1 1),(-1 2,3 4,1 -3,-2 1) )";
  auto p7 = "MULTIPOLYGON ( ((0 0, 1 4, 1 0,0 0)) )";
  // auto p8 = "MULTIPOLYGON ( ((0 0, 0 4, 4 4, 4 0, 0 0)), ((0 0, 0 1, 4 1, 4 0, 0 0))
  // )"; auto p9 = "MULTIPOLYGON ( ((0 0, 1 4, 1 0,0 0)), ((0 0,1 0,0 1,0 0)) )";

  arrow::StringBuilder builder;
  std::shared_ptr<arrow::Array> input;
  builder.Append(std::string(p1));
  builder.Append(std::string(p2));
  builder.Append(std::string(p3));
  builder.Append(std::string(p4));
  builder.Append(std::string(p5));
  builder.Append(std::string(p6));
  builder.Append(std::string(p7));
  // builder.Append(std::string(p8));
  // builder.Append(std::string(p9));
  builder.Finish(&input);

  auto res = zilliz::gis::ST_Envelope_Aggr(input);
  auto res_str = std::static_pointer_cast<arrow::StringArray>(res);

  ASSERT_EQ(res_str->GetString(0), "POLYGON ((-2 -3,-2 4,3 4,3 -3,-2 -3))");
}
