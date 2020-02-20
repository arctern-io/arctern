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
  builder2.Finish(&input2);

  auto res = zilliz::gis::ST_Intersection(input1, input2);
  auto res_str = std::static_pointer_cast<arrow::StringArray>(res);

  // for(int i=0;i<res_str->length();i++){
  // std::cout<<res_str->GetString(i)<<"#"<<i<<std::endl;
  // }
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

  auto res = zilliz::gis::ST_Equals(input1, input2);
  auto res_str = std::static_pointer_cast<arrow::BooleanArray>(res);

   for(int i=0;i<res_str->length();i++){
   std::cout<<res_str->Value(i)<<"#"<<i<<std::endl;
   }
  ASSERT_EQ(res_str->Value(0), true);
  ASSERT_EQ(res_str->Value(1), false);
  ASSERT_EQ(res_str->Value(2), false);
  // ASSERT_EQ(res_str->Value(3), true); // false
  ASSERT_EQ(res_str->Value(4), false);
  // ASSERT_EQ(res_str->Value(5), true); // false
  ASSERT_EQ(res_str->Value(6), false); 
  // ASSERT_EQ(res_str->Value(7), true); // false
  ASSERT_EQ(res_str->Value(8), false);
  // ASSERT_EQ(res_str->Value(9), true); // false
  ASSERT_EQ(res_str->Value(10), false);
  // ASSERT_EQ(res_str->Value(11), true); // false
}

TEST(geometry_test, test_ST_Touches) {
  arrow::StringBuilder left_string_builder;
  std::shared_ptr<arrow::Array> left_geometry;
  arrow::StringBuilder right_string_builder;
  std::shared_ptr<arrow::Array> right_geometry;
  char* left_str = nullptr;
  char* right_str = nullptr;

  left_str = build_point(25, 25);
  right_str = build_polygon(20, 20);
  left_string_builder.Append(std::string(left_str));
  right_string_builder.Append(std::string(right_str));
  left_string_builder.Finish(&left_geometry);
  right_string_builder.Finish(&right_geometry);
  auto intersection1 = zilliz::gis::ST_Touches(left_geometry, right_geometry);
  // auto intersection_polygons_arr =
  // std::static_pointer_cast<arrow::StringArray>(intersection_polygons);
  // ASSERT_EQ(intersection_polygons_arr->GetString(0),"LINESTRING (20 30, 20 20)");

  left_string_builder.Reset();
  right_string_builder.Reset();

  left_str = build_polygon(20, 20);
  right_str = build_linestring(25, 25);
  left_string_builder.Append(std::string(left_str));
  right_string_builder.Append(std::string(right_str));
  left_string_builder.Finish(&left_geometry);
  right_string_builder.Finish(&right_geometry);
  auto intersection2 = zilliz::gis::ST_Touches(left_geometry, right_geometry);

  left_string_builder.Reset();
  right_string_builder.Reset();

  left_str = build_point(20, 20);
  right_str = build_linestring(25, 25);
  left_string_builder.Append(std::string(left_str));
  right_string_builder.Append(std::string(right_str));
  left_string_builder.Finish(&left_geometry);
  right_string_builder.Finish(&right_geometry);
  auto intersection3 = zilliz::gis::ST_Touches(left_geometry, right_geometry);

  CPLFree(left_str);
  CPLFree(right_str);
}

TEST(geometry_test, test_ST_Overlaps) {
  arrow::StringBuilder left_string_builder;
  std::shared_ptr<arrow::Array> left_geometry;
  arrow::StringBuilder right_string_builder;
  std::shared_ptr<arrow::Array> right_geometry;
  char* left_str = nullptr;
  char* right_str = nullptr;

  left_str = build_point(25, 25);
  right_str = build_polygon(20, 20);
  left_string_builder.Append(std::string(left_str));
  right_string_builder.Append(std::string(right_str));
  left_string_builder.Finish(&left_geometry);
  right_string_builder.Finish(&right_geometry);
  auto intersection1 = zilliz::gis::ST_Overlaps(left_geometry, right_geometry);
  // auto intersection_polygons_arr =
  // std::static_pointer_cast<arrow::StringArray>(intersection_polygons);
  // ASSERT_EQ(intersection_polygons_arr->GetString(0),"LINESTRING (20 30, 20 20)");

  left_string_builder.Reset();
  right_string_builder.Reset();

  left_str = build_polygon(20, 20);
  right_str = build_linestring(25, 25);
  left_string_builder.Append(std::string(left_str));
  right_string_builder.Append(std::string(right_str));
  left_string_builder.Finish(&left_geometry);
  right_string_builder.Finish(&right_geometry);
  auto intersection2 = zilliz::gis::ST_Overlaps(left_geometry, right_geometry);

  left_string_builder.Reset();
  right_string_builder.Reset();

  left_str = build_point(20, 20);
  right_str = build_linestring(25, 25);
  left_string_builder.Append(std::string(left_str));
  right_string_builder.Append(std::string(right_str));
  left_string_builder.Finish(&left_geometry);
  right_string_builder.Finish(&right_geometry);
  auto intersection3 = zilliz::gis::ST_Overlaps(left_geometry, right_geometry);

  CPLFree(left_str);
  CPLFree(right_str);
}

TEST(geometry_test, test_ST_Crosses) {
  arrow::StringBuilder left_string_builder;
  std::shared_ptr<arrow::Array> left_geometry;
  arrow::StringBuilder right_string_builder;
  std::shared_ptr<arrow::Array> right_geometry;
  char* left_str = nullptr;
  char* right_str = nullptr;

  left_str = build_point(25, 25);
  right_str = build_polygon(20, 20);
  left_string_builder.Append(std::string(left_str));
  right_string_builder.Append(std::string(right_str));
  left_string_builder.Finish(&left_geometry);
  right_string_builder.Finish(&right_geometry);
  auto intersection1 = zilliz::gis::ST_Crosses(left_geometry, right_geometry);
  // auto intersection_polygons_arr =
  // std::static_pointer_cast<arrow::StringArray>(intersection_polygons);
  // ASSERT_EQ(intersection_polygons_arr->GetString(0),"LINESTRING (20 30, 20 20)");

  left_string_builder.Reset();
  right_string_builder.Reset();

  left_str = build_polygon(20, 20);
  right_str = build_linestring(25, 25);
  left_string_builder.Append(std::string(left_str));
  right_string_builder.Append(std::string(right_str));
  left_string_builder.Finish(&left_geometry);
  right_string_builder.Finish(&right_geometry);
  auto intersection2 = zilliz::gis::ST_Crosses(left_geometry, right_geometry);

  left_string_builder.Reset();
  right_string_builder.Reset();

  left_str = build_point(20, 20);
  right_str = build_linestring(25, 25);
  left_string_builder.Append(std::string(left_str));
  right_string_builder.Append(std::string(right_str));
  left_string_builder.Finish(&left_geometry);
  right_string_builder.Finish(&right_geometry);
  auto intersection3 = zilliz::gis::ST_Crosses(left_geometry, right_geometry);

  CPLFree(left_str);
  CPLFree(right_str);
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
  std::shared_ptr<arrow::Array> points = build_points();
  auto vaild_mark1 = zilliz::gis::ST_MakeValid(points);
  auto vaild_mark_arr1 = std::static_pointer_cast<arrow::BooleanArray>(vaild_mark1);

  std::shared_ptr<arrow::Array> polygons = build_polygons();
  auto vaild_mark2 = zilliz::gis::ST_MakeValid(polygons);
  auto vaild_mark_arr2 = std::static_pointer_cast<arrow::BooleanArray>(vaild_mark2);

  std::shared_ptr<arrow::Array> line = build_linestrings();
  auto vaild_mark3 = zilliz::gis::ST_MakeValid(line);
  auto vaild_mark_arr3 = std::static_pointer_cast<arrow::BooleanArray>(vaild_mark3);
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
  OGRLinearRing ring1;
  ring1.addPoint(2, 1);
  ring1.addPoint(3, 1);
  ring1.addPoint(3, 2);
  ring1.addPoint(2, 2);
  ring1.addPoint(2, 8);
  ring1.closeRings();
  OGRPolygon polygon1;
  polygon1.addRing(&ring1);

  OGRPoint point(2, 3);
  OGRLineString line;
  line.addPoint(10, 20);
  line.addPoint(20, 30);

  arrow::StringBuilder string_builder;
  std::shared_ptr<arrow::Array> geometries;

  char* polygon_str = nullptr;
  char* point_str = nullptr;
  char* line_str = nullptr;
  CHECK_GDAL(polygon1.exportToWkt(&polygon_str));
  CHECK_GDAL(point.exportToWkt(&point_str));
  CHECK_GDAL(point.exportToWkt(&line_str));
  string_builder.Append(std::string(polygon_str));
  string_builder.Append(std::string(point_str));
  string_builder.Append(std::string(line_str));
  CPLFree(polygon_str);
  CPLFree(point_str);
  CPLFree(line_str);

  string_builder.Finish(&geometries);

  auto geometries_arr = zilliz::gis::ST_SimplifyPreserveTopology(geometries, 10000);
  auto geometries_arr_str = std::static_pointer_cast<arrow::StringArray>(geometries_arr);

  ASSERT_EQ(geometries_arr_str->GetString(0), "POLYGON ((2 1,3 1,2 8,2 1))");
  ASSERT_EQ(geometries_arr_str->GetString(1), "POINT (2 3)");
  //  ASSERT_EQ(geometries_arr_str->GetString(2),"LINESTRING");
}

TEST(geometry_test, test_ST_Contains) {
  std::shared_ptr<arrow::Array> points = build_points();
  auto vaild_mark1 = zilliz::gis::ST_MakeValid(points);
  auto vaild_mark_arr1 = std::static_pointer_cast<arrow::BooleanArray>(vaild_mark1);

  std::shared_ptr<arrow::Array> polygons = build_polygons();
  auto vaild_mark2 = zilliz::gis::ST_MakeValid(polygons);
  auto vaild_mark_arr2 = std::static_pointer_cast<arrow::BooleanArray>(vaild_mark2);

  std::shared_ptr<arrow::Array> lines = build_linestrings();
  auto vaild_mark3 = zilliz::gis::ST_MakeValid(lines);
  auto vaild_mark_arr3 = std::static_pointer_cast<arrow::BooleanArray>(vaild_mark3);

  auto res1 = zilliz::gis::ST_Contains(points, polygons);
  auto res2 = zilliz::gis::ST_Contains(polygons, lines);
  auto res3 = zilliz::gis::ST_Contains(points, lines);
}

TEST(geometry_test, test_ST_Intersects) {
  std::shared_ptr<arrow::Array> points = build_points();
  auto vaild_mark1 = zilliz::gis::ST_MakeValid(points);
  auto vaild_mark_arr1 = std::static_pointer_cast<arrow::BooleanArray>(vaild_mark1);

  std::shared_ptr<arrow::Array> polygons = build_polygons();
  auto vaild_mark2 = zilliz::gis::ST_MakeValid(polygons);
  auto vaild_mark_arr2 = std::static_pointer_cast<arrow::BooleanArray>(vaild_mark2);

  std::shared_ptr<arrow::Array> lines = build_linestrings();
  auto vaild_mark3 = zilliz::gis::ST_MakeValid(lines);
  auto vaild_mark_arr3 = std::static_pointer_cast<arrow::BooleanArray>(vaild_mark3);

  auto res1 = zilliz::gis::ST_Intersects(points, polygons);
  auto res2 = zilliz::gis::ST_Intersects(polygons, lines);
  auto res3 = zilliz::gis::ST_Intersects(points, lines);
}

TEST(geometry_test, test_ST_Within) {
  std::shared_ptr<arrow::Array> points = build_points();
  auto vaild_mark1 = zilliz::gis::ST_MakeValid(points);
  auto vaild_mark_arr1 = std::static_pointer_cast<arrow::BooleanArray>(vaild_mark1);

  std::shared_ptr<arrow::Array> polygons = build_polygons();
  auto vaild_mark2 = zilliz::gis::ST_MakeValid(polygons);
  auto vaild_mark_arr2 = std::static_pointer_cast<arrow::BooleanArray>(vaild_mark2);

  std::shared_ptr<arrow::Array> lines = build_linestrings();
  auto vaild_mark3 = zilliz::gis::ST_MakeValid(lines);
  auto vaild_mark_arr3 = std::static_pointer_cast<arrow::BooleanArray>(vaild_mark3);

  auto res1 = zilliz::gis::ST_Within(points, polygons);
  auto res2 = zilliz::gis::ST_Within(polygons, lines);
  auto res3 = zilliz::gis::ST_Within(points, lines);
}

TEST(geometry_test, test_ST_Distance) {
  std::shared_ptr<arrow::Array> points = build_points();
  auto vaild_mark1 = zilliz::gis::ST_MakeValid(points);
  auto vaild_mark_arr1 = std::static_pointer_cast<arrow::BooleanArray>(vaild_mark1);

  std::shared_ptr<arrow::Array> polygons = build_polygons();
  auto vaild_mark2 = zilliz::gis::ST_MakeValid(polygons);
  auto vaild_mark_arr2 = std::static_pointer_cast<arrow::BooleanArray>(vaild_mark2);

  std::shared_ptr<arrow::Array> lines = build_linestrings();
  auto vaild_mark3 = zilliz::gis::ST_MakeValid(lines);
  auto vaild_mark_arr3 = std::static_pointer_cast<arrow::BooleanArray>(vaild_mark3);

  auto res1 = zilliz::gis::ST_Distance(points, polygons);
  auto res2 = zilliz::gis::ST_Distance(polygons, lines);
  auto res3 = zilliz::gis::ST_Distance(points, lines);
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

  auto res = zilliz::gis::ST_Area(input);
  auto res_double = std::static_pointer_cast<arrow::DoubleArray>(res);

  EXPECT_DOUBLE_EQ(res_double->Value(0), 0);
  EXPECT_DOUBLE_EQ(res_double->Value(1), 0);
  //  EXPECT_DOUBLE_EQ(res_double->Value(2), 0);
  EXPECT_DOUBLE_EQ(res_double->Value(3), 1);
  EXPECT_DOUBLE_EQ(res_double->Value(4), 0);
  EXPECT_DOUBLE_EQ(res_double->Value(5), 0);
  EXPECT_DOUBLE_EQ(res_double->Value(6), 2);
  EXPECT_DOUBLE_EQ(res_double->Value(7), 20);
  // EXPECT_DOUBLE_EQ(res_double->Value(8), 1.5);
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
  std::shared_ptr<arrow::Array> points = build_points();
  auto vaild_mark1 = zilliz::gis::ST_MakeValid(points);
  auto vaild_mark_arr1 = std::static_pointer_cast<arrow::BooleanArray>(vaild_mark1);

  std::shared_ptr<arrow::Array> polygons = build_polygons();
  auto vaild_mark2 = zilliz::gis::ST_MakeValid(polygons);
  auto vaild_mark_arr2 = std::static_pointer_cast<arrow::BooleanArray>(vaild_mark2);

  std::shared_ptr<arrow::Array> lines = build_linestrings();
  auto vaild_mark3 = zilliz::gis::ST_MakeValid(lines);
  auto vaild_mark_arr3 = std::static_pointer_cast<arrow::BooleanArray>(vaild_mark3);

  auto res1 = zilliz::gis::ST_NPoints(points);
  auto res2 = zilliz::gis::ST_NPoints(polygons);
  auto res3 = zilliz::gis::ST_NPoints(lines);
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
  std::shared_ptr<arrow::Array> points = build_points();
  auto vaild_mark1 = zilliz::gis::ST_MakeValid(points);
  auto vaild_mark_arr1 = std::static_pointer_cast<arrow::BooleanArray>(vaild_mark1);

  std::shared_ptr<arrow::Array> polygons = build_polygons();
  auto vaild_mark2 = zilliz::gis::ST_MakeValid(polygons);
  auto vaild_mark_arr2 = std::static_pointer_cast<arrow::BooleanArray>(vaild_mark2);

  std::shared_ptr<arrow::Array> lines = build_linestrings();
  auto vaild_mark3 = zilliz::gis::ST_MakeValid(lines);
  auto vaild_mark_arr3 = std::static_pointer_cast<arrow::BooleanArray>(vaild_mark3);

  auto res1 = zilliz::gis::ST_Buffer(points, 1.2);
  auto res2 = zilliz::gis::ST_Buffer(polygons, 1.2);
  auto res3 = zilliz::gis::ST_Buffer(lines, 1.2);
}

TEST(geometry_test, test_ST_PolygonFromEnvelope) {
  arrow::DoubleBuilder x_min;
  arrow::DoubleBuilder x_max;
  arrow::DoubleBuilder y_min;
  arrow::DoubleBuilder y_max;

  x_min.Append(0);
  x_max.Append(1);
  y_min.Append(0);
  y_max.Append(1);

  std::shared_ptr<arrow::Array> x_min_ptr;
  std::shared_ptr<arrow::Array> x_max_ptr;
  std::shared_ptr<arrow::Array> y_min_ptr;
  std::shared_ptr<arrow::Array> y_max_ptr;

  x_min.Finish(&x_min_ptr);
  x_max.Finish(&x_max_ptr);
  y_min.Finish(&y_min_ptr);
  y_max.Finish(&y_max_ptr);

  auto res =
      zilliz::gis::ST_PolygonFromEnvelope(x_min_ptr, y_min_ptr, x_max_ptr, y_max_ptr);

  auto res_str = std::static_pointer_cast<arrow::StringArray>(res)->GetString(0);
  std::string expect = "POLYGON ((0 0,1 0,0 1,1 1,0 0))";

  ASSERT_EQ(res_str, expect);
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
  arrow::StringBuilder builder;
  std::shared_ptr<arrow::Array> polygons;

  auto p1 = "POLYGON ((1 1,1 2,2 2,2 1,1 1))";
  auto p2 = "POLYGON ((2 1,3 1,3 2,2 2,2 1))";
  builder.Reset();
  builder.Append(std::string(p1));
  builder.Append(std::string(p2));
  builder.Finish(&polygons);

  auto result = zilliz::gis::ST_Union_Aggr(polygons);
  auto geometries_arr = std::static_pointer_cast<arrow::StringArray>(result);

  ASSERT_EQ(geometries_arr->GetString(0), "POLYGON ((1 1,1 2,2 2,3 2,3 1,2 1,1 1))");

  p1 = "POLYGON ((0 0,4 0,4 4,0 4,0 0))";
  p2 = "POLYGON ((3 1,5 1,5 2,3 2,3 1))";
  builder.Reset();
  builder.Append(std::string(p1));
  builder.Append(std::string(p2));
  builder.Finish(&polygons);
  result = zilliz::gis::ST_Union_Aggr(polygons);
  geometries_arr = std::static_pointer_cast<arrow::StringArray>(result);

  ASSERT_EQ(geometries_arr->GetString(0),
            "POLYGON ((4 1,4 0,0 0,0 4,4 4,4 2,5 2,5 1,4 1))");

  p1 = "POLYGON ((0 0,4 0,4 4,0 4,0 0))";
  p2 = "POLYGON ((5 1,7 1,7 2,5 2,5 1))";
  builder.Reset();
  builder.Append(std::string(p1));
  builder.Append(std::string(p2));
  builder.Finish(&polygons);
  result = zilliz::gis::ST_Union_Aggr(polygons);
  geometries_arr = std::static_pointer_cast<arrow::StringArray>(result);

  ASSERT_EQ(geometries_arr->GetString(0),
            "MULTIPOLYGON (((0 0,4 0,4 4,0 4,0 0)),((5 1,7 1,7 2,5 2,5 1)))");

  p1 = "POLYGON ((0 0,4 0,4 4,0 4,0 0))";
  p2 = "POINT (2 3)";
  builder.Reset();
  builder.Append(std::string(p1));
  builder.Append(std::string(p2));
  builder.Finish(&polygons);
  result = zilliz::gis::ST_Union_Aggr(polygons);
  geometries_arr = std::static_pointer_cast<arrow::StringArray>(result);

  ASSERT_EQ(geometries_arr->GetString(0), "POLYGON ((0 0,0 4,4 4,4 0,0 0))");
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
