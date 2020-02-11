#include <gtest/gtest.h>
#include <iostream>
#include <ctime>
#include <ogr_geometry.h>
#include <arrow/api.h>
#include <arrow/array.h>

#include "arrow/gis_api.h"
#include "utils/check_status.h"

using namespace zilliz::gis;

TEST(geometry_test, make_point_from_double){

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

    auto point_arr = ST_Point(ptr_x,ptr_y);
    auto point_arr_str = std::static_pointer_cast<arrow::StringArray>(point_arr);

    ASSERT_EQ(point_arr_str->length(),2);
    ASSERT_EQ(point_arr_str->GetString(0),"POINT (0 0)");
    ASSERT_EQ(point_arr_str->GetString(1),"POINT (1 1)");
}


TEST(geometry_test,test_ST_IsValid){
  OGRLinearRing ring1;
  ring1.addPoint(1, 1);
  ring1.addPoint(1, 2);
  ring1.addPoint(2, 2);
  ring1.addPoint(2, 1);
  ring1.addPoint(1, 1);
  ring1.closeRings();
  OGRPolygon polygon1;
  polygon1.addRing(&ring1);

  OGRLinearRing ring2;
  ring2.addPoint(2, 1);
  ring2.addPoint(3, 1);
  ring2.addPoint(3, 2);
  ring2.addPoint(2, 2);
  ring2.addPoint(2, 1);
  ring2.closeRings();
  OGRPolygon polygon2;
  polygon2.addRing(&ring2);


  OGRLinearRing ring3;
  ring3.addPoint(2, 1);
  ring3.addPoint(3, 1);
  ring3.addPoint(3, 2);
  ring3.addPoint(2, 2);
  ring3.addPoint(2, 8);
  ring3.closeRings();
  OGRPolygon polygon3;
  polygon3.addRing(&ring3);

  arrow::StringBuilder string_builder;
  std::shared_ptr<arrow::Array> polygons;

  char *str1 = nullptr;
  char *str2 = nullptr;
  char *str3 = nullptr;
  CHECK_GDAL(polygon1.exportToWkt(&str1));
  CHECK_GDAL(polygon2.exportToWkt(&str2));
  CHECK_GDAL(polygon3.exportToWkt(&str3));
  string_builder.Append(std::string(str1));
  string_builder.Append(std::string(str2));
  string_builder.Append(std::string(str3));
  CPLFree(str1);
  CPLFree(str2);
  CPLFree(str3);

  string_builder.Finish(&polygons);

  auto vaild_mark = ST_IsValid(polygons);
  auto vaild_mark_arr = std::static_pointer_cast<arrow::BooleanArray>(vaild_mark);

  ASSERT_EQ(vaild_mark_arr->Value(0),true);
  ASSERT_EQ(vaild_mark_arr->Value(1),true);
  ASSERT_EQ(vaild_mark_arr->Value(2),false);
}

TEST(geometry_test, test_ST_Intersection){
  OGRLinearRing ring1;
  ring1.addPoint(1, 1);
  ring1.addPoint(1, 2);
  ring1.addPoint(2, 2);
  ring1.addPoint(2, 1);
  ring1.addPoint(1, 1);
  ring1.closeRings();
  OGRPolygon polygon1;
  polygon1.addRing(&ring1);

  OGRLinearRing ring2;
  ring2.addPoint(2, 1);
  ring2.addPoint(3, 1);
  ring2.addPoint(3, 2);
  ring2.addPoint(2, 2);
  ring2.addPoint(2, 1);
  ring2.closeRings();
  OGRPolygon polygon2;
  polygon2.addRing(&ring2);

  arrow::StringBuilder left_string_builder;
  std::shared_ptr<arrow::Array> left_polygons;
  arrow::StringBuilder right_string_builder;
  std::shared_ptr<arrow::Array> right_polygons;

  char *left_str = nullptr;
  char *right_str = nullptr;
  CHECK_GDAL(polygon1.exportToWkt(&left_str));
  CHECK_GDAL(polygon2.exportToWkt(&right_str));
  left_string_builder.Append(std::string(left_str));
  right_string_builder.Append(std::string(right_str));
  CPLFree(left_str);
  CPLFree(right_str);

  left_string_builder.Finish(&left_polygons);
  right_string_builder.Finish(&right_polygons);

  auto intersection_geometries = ST_Intersection(left_polygons,right_polygons);
  auto intersection_geometries_arr = std::static_pointer_cast<arrow::StringArray>(intersection_geometries);
  
  ASSERT_EQ(intersection_geometries_arr->GetString(0),"LINESTRING (2 2,2 1)");
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
//   auto geometries = ST_PrecisionReduce(array,6);
//   auto geometries_arr = std::static_pointer_cast<arrow::StringArray>(geometries);
  
//   // ASSERT_EQ(geometries_arr->GetString(0),"POINT (1.55556 1.55556)");
//   ASSERT_EQ(geometries_arr->GetString(0),"POINT (1.5555555 1.55555555)");
// }

TEST(geometry_test, test_ST_Equals){
  OGRLinearRing ring1;
  ring1.addPoint(1, 1);
  ring1.addPoint(1, 2);
  ring1.addPoint(2, 2);
  ring1.addPoint(2, 1);
  ring1.addPoint(1, 1);
  ring1.closeRings();
  OGRPolygon polygon1;
  polygon1.addRing(&ring1);

  OGRLinearRing ring2;
  ring2.addPoint(2, 1);
  ring2.addPoint(3, 1);
  ring2.addPoint(3, 2);
  ring2.addPoint(2, 2);
  ring2.addPoint(2, 1);
  ring2.closeRings();
  OGRPolygon polygon2;
  polygon2.addRing(&ring2);

  arrow::StringBuilder left_string_builder;
  arrow::StringBuilder right_string_builder;
  std::shared_ptr<arrow::Array> left_polygons;
  std::shared_ptr<arrow::Array> right_polygons;
  
  char * str1 = nullptr;
  char * str2 = nullptr;

  CHECK_GDAL(polygon1.exportToWkt(&str1));
  CHECK_GDAL(polygon2.exportToWkt(&str2));

  left_string_builder.Append(std::string(str1));
  left_string_builder.Append(std::string(str2));
  right_string_builder.Append(std::string(str1));
  right_string_builder.Append(std::string(str1));

  CPLFree(str1);
  CPLFree(str2);

  left_string_builder.Finish(&left_polygons);
  right_string_builder.Finish(&right_polygons);

  auto vaild_mark = ST_Equals(left_polygons,right_polygons);
  auto vaild_mark_str = std::static_pointer_cast<arrow::BooleanArray>(vaild_mark);

  ASSERT_EQ(vaild_mark_str->Value(0),true);
  ASSERT_EQ(vaild_mark_str->Value(1),false);
}

TEST(geometry_test, test_ST_Touches){
  OGRLinearRing ring1;
  ring1.addPoint(1, 1);
  ring1.addPoint(1, 2);
  ring1.addPoint(2, 2);
  ring1.addPoint(2, 1);
  ring1.addPoint(1, 1);
  ring1.closeRings();
  OGRPolygon polygon1;
  polygon1.addRing(&ring1);

  OGRLinearRing ring2;
  ring2.addPoint(2, 1);
  ring2.addPoint(3, 1);
  ring2.addPoint(3, 2);
  ring2.addPoint(2, 2);
  ring2.addPoint(2, 1);
  ring2.closeRings();
  OGRPolygon polygon2;
  polygon2.addRing(&ring2);

  arrow::StringBuilder left_string_builder;
  arrow::StringBuilder right_string_builder;
  std::shared_ptr<arrow::Array> left_polygons;
  std::shared_ptr<arrow::Array> right_polygons;

  char * str1 = nullptr;
  char * str2 = nullptr;

  CHECK_GDAL(polygon1.exportToWkt(&str1));
  CHECK_GDAL(polygon2.exportToWkt(&str2));

  left_string_builder.Append(std::string(str1));
  left_string_builder.Append(std::string(str2));
  right_string_builder.Append(std::string(str1));
  right_string_builder.Append(std::string(str1));

  CPLFree(str1);
  CPLFree(str2);

  left_string_builder.Finish(&left_polygons);
  right_string_builder.Finish(&right_polygons);

  auto vaild_mark = ST_Touches(left_polygons,right_polygons);
  auto vaild_mark_str = std::static_pointer_cast<arrow::BooleanArray>(vaild_mark);
   
  ASSERT_EQ(vaild_mark_str->Value(0),false);
  ASSERT_EQ(vaild_mark_str->Value(1),true);
}

TEST(geometry_test, test_ST_Overlaps){
  OGRLinearRing ring1;
  ring1.addPoint(1, 1);
  ring1.addPoint(1, 2);
  ring1.addPoint(2, 2);
  ring1.addPoint(2, 1);
  ring1.addPoint(1, 1);
  ring1.closeRings();
  OGRPolygon polygon1;
  polygon1.addRing(&ring1);

  OGRLinearRing ring2;
  ring2.addPoint(2, 1);
  ring2.addPoint(3, 1);
  ring2.addPoint(3, 2);
  ring2.addPoint(2, 2);
  ring2.addPoint(2, 1);
  ring2.closeRings();
  OGRPolygon polygon2;
  polygon2.addRing(&ring2);

  arrow::StringBuilder left_string_builder;
  arrow::StringBuilder right_string_builder;
  std::shared_ptr<arrow::Array> left_polygons;
  std::shared_ptr<arrow::Array> right_polygons;

  char * str1 = nullptr;
  char * str2 = nullptr;

  CHECK_GDAL(polygon1.exportToWkt(&str1));
  CHECK_GDAL(polygon2.exportToWkt(&str2));

  left_string_builder.Append(std::string(str1));
  left_string_builder.Append(std::string(str2));
  right_string_builder.Append(std::string(str1));
  right_string_builder.Append(std::string(str1));

  CPLFree(str1);
  CPLFree(str2);

  left_string_builder.Finish(&left_polygons);
  right_string_builder.Finish(&right_polygons);

  auto vaild_mark = ST_Overlaps(left_polygons,right_polygons);
  auto vaild_mark_str = std::static_pointer_cast<arrow::BooleanArray>(vaild_mark);
   
  ASSERT_EQ(vaild_mark_str->Value(0),false);
  ASSERT_EQ(vaild_mark_str->Value(1),false);
}


TEST(geometry_test, test_ST_Crosses){
  OGRLinearRing ring1;
  ring1.addPoint(1, 1);
  ring1.addPoint(1, 2);
  ring1.addPoint(2, 2);
  ring1.addPoint(2, 1);
  ring1.addPoint(1, 1);
  ring1.closeRings();
  OGRPolygon polygon1;
  polygon1.addRing(&ring1);

  OGRLinearRing ring2;
  ring2.addPoint(2, 1);
  ring2.addPoint(3, 1);
  ring2.addPoint(3, 2);
  ring2.addPoint(2, 2);
  ring2.addPoint(2, 1);
  ring2.closeRings();
  OGRPolygon polygon2;
  polygon2.addRing(&ring2);

  arrow::StringBuilder left_string_builder;
  arrow::StringBuilder right_string_builder;
  std::shared_ptr<arrow::Array> left_polygons;
  std::shared_ptr<arrow::Array> right_polygons;

  char * str1 = nullptr;
  char * str2 = nullptr;

  CHECK_GDAL(polygon1.exportToWkt(&str1));
  CHECK_GDAL(polygon2.exportToWkt(&str2));

  left_string_builder.Append(std::string(str1));
  left_string_builder.Append(std::string(str2));
  right_string_builder.Append(std::string(str1));
  right_string_builder.Append(std::string(str1));

  CPLFree(str1);
  CPLFree(str2);

  left_string_builder.Finish(&left_polygons);
  right_string_builder.Finish(&right_polygons);

  auto vaild_mark = ST_Crosses(left_polygons,right_polygons);
  auto vaild_mark_str = std::static_pointer_cast<arrow::BooleanArray>(vaild_mark);
   
  ASSERT_EQ(vaild_mark_str->Value(0),false);
  ASSERT_EQ(vaild_mark_str->Value(1),false);
}


TEST(geometry_test, test_ST_IsSimple){

  OGRLinearRing ring1;
  ring1.addPoint(1, 1);
  ring1.addPoint(1, 2);
  ring1.addPoint(2, 2);
  ring1.addPoint(2, 1);
  ring1.addPoint(1, 1);
  ring1.closeRings();
  OGRPolygon polygon1;
  polygon1.addRing(&ring1);

  OGRLinearRing ring2;
  ring2.addPoint(2, 1);
  ring2.addPoint(3, 1);
  ring2.addPoint(3, 2);
  ring2.addPoint(2, 2);
  ring2.addPoint(2, 1);
  ring2.closeRings();
  OGRPolygon polygon2;
  polygon2.addRing(&ring2);


  OGRLinearRing ring3;
  ring3.addPoint(2, 1);
  ring3.addPoint(3, 1);
  ring3.addPoint(3, 2);
  ring3.addPoint(2, 2);
  ring3.addPoint(2, 8);
  ring3.closeRings();
  OGRPolygon polygon3;
  polygon3.addRing(&ring3);

  arrow::StringBuilder string_builder;
  std::shared_ptr<arrow::Array> polygons;
  
  char *str1 = nullptr;
  char *str2 = nullptr;
  char *str3 = nullptr;
  CHECK_GDAL(polygon1.exportToWkt(&str1));
  CHECK_GDAL(polygon2.exportToWkt(&str2));
  CHECK_GDAL(polygon3.exportToWkt(&str3));
  string_builder.Append(std::string(str1));
  string_builder.Append(std::string(str2));
  string_builder.Append(std::string(str3));
  CPLFree(str1);
  CPLFree(str2);
  CPLFree(str3);

  string_builder.Finish(&polygons);

  auto vaild_mark = ST_IsSimple(polygons);
  auto vaild_mark_arr = std::static_pointer_cast<arrow::BooleanArray>(vaild_mark);

  ASSERT_EQ(vaild_mark_arr->Value(0),true);
  ASSERT_EQ(vaild_mark_arr->Value(1),true);
  ASSERT_EQ(vaild_mark_arr->Value(2),false);
}

TEST(geometry_test, test_ST_MakeValid){
  OGRLinearRing ring1;
  ring1.addPoint(2, 1);
  ring1.addPoint(3, 1);
  ring1.addPoint(3, 2);
  ring1.addPoint(2, 2);
  ring1.addPoint(2, 8);
  ring1.closeRings();
  OGRPolygon polygon1;
  polygon1.addRing(&ring1);

  arrow::StringBuilder string_builder;
  std::shared_ptr<arrow::Array> polygons;
  
  // std::cout << polygon1.exportToWkt() <<std::endl;
  char *str = nullptr;
  CHECK_GDAL(polygon1.exportToWkt(&str));
  string_builder.Append(std::string(str));
  CPLFree(str);
  string_builder.Finish(&polygons);
  
  auto geometries = ST_MakeValid(polygons);
  auto geometries_arr = std::static_pointer_cast<arrow::StringArray>(geometries);
  
  ASSERT_EQ(geometries_arr->GetString(0),"GEOMETRYCOLLECTION (POLYGON ((2 2,3 2,3 1,2 1,2 2)),LINESTRING (2 2,2 8))");
}

TEST(geometry_test, test_ST_GeometryType){
  OGRLinearRing ring1;
  ring1.addPoint(2, 1);
  ring1.addPoint(3, 1);
  ring1.addPoint(3, 2);
  ring1.addPoint(2, 2);
  ring1.addPoint(2, 8);
  ring1.closeRings();
  OGRPolygon polygon1;
  polygon1.addRing(&ring1);

  OGRPoint point(2,3);

  arrow::StringBuilder string_builder;
  std::shared_ptr<arrow::Array> geometries;

  char *polygon_str = nullptr;
  char *point_str = nullptr;
  CHECK_GDAL(polygon1.exportToWkt(&polygon_str));
  CHECK_GDAL(point.exportToWkt(&point_str));
  string_builder.Append(std::string(polygon_str));
  string_builder.Append(std::string(point_str));
  CPLFree(polygon_str);
  CPLFree(point_str);

  string_builder.Finish(&geometries);

  auto geometries_type = ST_GeometryType(geometries);
  auto geometries_type_arr = std::static_pointer_cast<arrow::StringArray>(geometries_type);
  
  ASSERT_EQ(geometries_type_arr->GetString(0),"POLYGON");
  ASSERT_EQ(geometries_type_arr->GetString(1),"POINT");
}

TEST(geometry_test, test_ST_SimplifyPreserveTopology){
  OGRLinearRing ring1;
  ring1.addPoint(2, 1);
  ring1.addPoint(3, 1);
  ring1.addPoint(3, 2);
  ring1.addPoint(2, 2);
  ring1.addPoint(2, 8);
  ring1.closeRings();
  OGRPolygon polygon1;
  polygon1.addRing(&ring1);

  OGRPoint point(2,3);

  arrow::StringBuilder string_builder;
  std::shared_ptr<arrow::Array> geometries;

  char *polygon_str = nullptr;
  char *point_str = nullptr;
  CHECK_GDAL(polygon1.exportToWkt(&polygon_str));
  CHECK_GDAL(point.exportToWkt(&point_str));
  string_builder.Append(std::string(polygon_str));
  string_builder.Append(std::string(point_str));
  CPLFree(polygon_str);
  CPLFree(point_str);

  string_builder.Finish(&geometries);

  auto geometries_arr = ST_SimplifyPreserveTopology(geometries,10000);
  auto geometries_arr_str = std::static_pointer_cast<arrow::StringArray>(geometries_arr);
  
  ASSERT_EQ(geometries_arr_str->GetString(0),"POLYGON ((2 1,3 1,2 8,2 1))");
  ASSERT_EQ(geometries_arr_str->GetString(1),"POINT (2 3)");
}

TEST(geometry_test, test_ST_Contains) {
std::shared_ptr<arrow::Array> geo_test;
arrow::StringBuilder builder1;
builder1.Append("POLYGON((0 0,1 0,1 1,0 1,0 0))");
builder1.Append("POLYGON((8 0,9 0,9 1,8 1,8 0))");
builder1.Append("POINT(2 2)");
builder1.Append("POINT(200 2)");
builder1.Finish(&geo_test);

std::shared_ptr<arrow::Array> geo;
arrow::StringBuilder builder2;
builder2.Append("POLYGON((0 0,0 8,8 8,8 0,0 0))");
builder2.Append("POLYGON((0 0,0 8,8 8,8 0,0 0))");
builder2.Append("POLYGON((0 0,0 8,8 8,8 0,0 0))");
builder2.Append("POLYGON((0 0,0 8,8 8,8 0,0 0))");
builder2.Finish(&geo);

auto res = ST_Contains(geo, geo_test);

arrow::BooleanBuilder builder_res;
std::shared_ptr<arrow::Array> expect_res;
builder_res.Append(true);
builder_res.Append(false);
builder_res.Append(true);
builder_res.Append(false);
builder_res.Finish(&expect_res);

ASSERT_EQ(res->Equals(expect_res), true);
}

TEST(geometry_test, test_ST_Intersects) {
std::shared_ptr<arrow::Array> geo_test;
arrow::StringBuilder builder1;
builder1.Append("POLYGON((0 0,1 0,1 1,0 1,0 0))");
builder1.Append("POLYGON((8 0,9 0,9 1,8 1,8 0))");
builder1.Append("LINESTRING(2 2,10 2)");
builder1.Append("LINESTRING(9 2,10 2)");
builder1.Finish(&geo_test);

std::shared_ptr<arrow::Array> geo;
arrow::StringBuilder builder2;
builder2.Append("POLYGON((0 0,0 8,8 8,8 0,0 0))");
builder2.Append("POLYGON((0 0,0 8,8 8,8 0,0 0))");
builder2.Append("POLYGON((0 0,0 8,8 8,8 0,0 0))");
builder2.Append("POLYGON((0 0,0 8,8 8,8 0,0 0))");
builder2.Finish(&geo);

auto res = ST_Intersects(geo, geo_test);

arrow::BooleanBuilder builder_res;
std::shared_ptr<arrow::Array> expect_res;
builder_res.Append(true);
builder_res.Append(true);
builder_res.Append(true);
builder_res.Append(false);
builder_res.Finish(&expect_res);

ASSERT_EQ(res->Equals(expect_res), true);
}

TEST(geometry_test, test_ST_Within) {
std::shared_ptr<arrow::Array> geo_test;
arrow::StringBuilder builder1;
builder1.Append("POLYGON((0 0,1 0,1 1,0 1,0 0))");
builder1.Append("POLYGON((8 0,9 0,9 1,8 1,8 0))");
builder1.Append("LINESTRING(2 2,3 2)");
builder1.Append("POINT(10 2)");
builder1.Finish(&geo_test);

std::shared_ptr<arrow::Array> geo;
arrow::StringBuilder builder2;
builder2.Append("POLYGON((0 0,0 8,8 8,8 0,0 0))");
builder2.Append("POLYGON((0 0,0 8,8 8,8 0,0 0))");
builder2.Append("POLYGON((0 0,0 8,8 8,8 0,0 0))");
builder2.Append("POLYGON((0 0,0 8,8 8,8 0,0 0))");
builder2.Finish(&geo);

auto res = ST_Within(geo, geo_test);

arrow::BooleanBuilder builder_res;
std::shared_ptr<arrow::Array> expect_res;
builder_res.Append(false);
builder_res.Append(false);
builder_res.Append(false);
builder_res.Append(false);
builder_res.Finish(&expect_res);

ASSERT_EQ(res->Equals(expect_res), true);
}

TEST(geometry_test, test_ST_Distance) {
std::shared_ptr<arrow::Array> geo_test;
arrow::StringBuilder builder1;
builder1.Append("LINESTRING(9 0,9 2)");
builder1.Append("POINT(10 2)");
builder1.Finish(&geo_test);

std::shared_ptr<arrow::Array> geo;
arrow::StringBuilder builder2;
builder2.Append("POLYGON((0 0,0 8,8 8,8 0,0 0))");
builder2.Append("POLYGON((0 0,0 8,8 8,8 0,0 0))");
builder2.Finish(&geo);

auto res = ST_Distance(geo, geo_test);

arrow::DoubleBuilder builder_res;
std::shared_ptr<arrow::Array> expect_res;
builder_res.Append(1.0);
builder_res.Append(2.0);
builder_res.Finish(&expect_res);

ASSERT_EQ(res->Equals(expect_res), true);
}

TEST(geometry_test, test_ST_Area) {

std::shared_ptr<arrow::Array> geo;
arrow::StringBuilder builder1;
builder1.Append("POLYGON((0 0,1 0,1 1,0 1,0 0))");
builder1.Append("POLYGON((0 0,0 8,8 8,8 0,0 0))");
builder1.Finish(&geo);

auto res = ST_Area(geo);

arrow::DoubleBuilder builder_res;
std::shared_ptr<arrow::Array> expect_res;
builder_res.Append(1.0);
builder_res.Append(64.0);
builder_res.Finish(&expect_res);

ASSERT_EQ(res->Equals(expect_res), true);
}

TEST(geometry_test, test_ST_Centroid) {

std::shared_ptr<arrow::Array> geo;
arrow::StringBuilder builder1;
builder1.Append("POLYGON((0 0,1 0,1 1,0 1,0 0))");
builder1.Append("POLYGON((0 0,0 8,8 8,8 0,0 0))");
builder1.Finish(&geo);

auto res = ST_Centroid(geo);

auto res_str = std::static_pointer_cast<const arrow::StringArray>(res);
// std::cout << "ST_Centroid_test" << std::endl;
// for(int i=0; i<2; ++i){
//   std::cout << "centroid_test : " << res_str->GetString(i) << std::endl;
// }

arrow::StringBuilder builder_res;
std::shared_ptr<arrow::Array> expect_res;
builder_res.Append("POINT (0.5 0.5)");
builder_res.Append("POINT (4 4)");
builder_res.Finish(&expect_res);

ASSERT_EQ(res->Equals(expect_res), true);
}

TEST(geometry_test, test_ST_Length) {

std::shared_ptr<arrow::Array> geo;
arrow::StringBuilder builder1;
builder1.Append("LINESTRING(0 0,0 1)");
builder1.Append("LINESTRING(1 1,1 4)");
builder1.Finish(&geo);

auto res = ST_Length(geo);

arrow::DoubleBuilder builder_res;
std::shared_ptr<arrow::Array> expect_res;
builder_res.Append(1.0);
builder_res.Append(3.0);
builder_res.Finish(&expect_res);

ASSERT_EQ(res->Equals(expect_res), true);
}

TEST(geometry_test, test_ST_ConvexHull) {

std::shared_ptr<arrow::Array> geo;
arrow::StringBuilder builder1;
//TODO : verify expect_res
//  builder2.Append("POLYGON((0 0,1 0,1 1,0 1,0 0))");
//  builder2.Append("LINESTRING(1 1,1 4)");
builder1.Append("POINT (1.1 101.1)");
builder1.Finish(&geo);

auto res = ST_ConvexHull(geo);

arrow::StringBuilder builder_res;
std::shared_ptr<arrow::Array> expect_res;
//  builder2.Append("POLYGON((0 0,1 0,1 1,0 1,0 0))");
//  builder2.Append("LINESTRING(1 1,1 4)");
builder_res.Append("POINT (1.1 101.1)");
builder_res.Finish(&expect_res);

ASSERT_EQ(res->Equals(expect_res), true);
}

//TODO:geospark ST_NPoints can not work.
TEST(geometry_test, test_ST_NPoints) {

std::shared_ptr<arrow::Array> geo;
arrow::StringBuilder builder1;
//TODO : verify expect_res
// builder1.Append("POLYGON((0 0,1 0,1 1,0 1,0 0))");
builder1.Append("LINESTRING(1 1,1 4)");
// builder1.Append("POINT (1.1 101.1)");
builder1.Finish(&geo);

auto res = ST_NPoints(geo);

// auto res_int = std::static_pointer_cast<arrow::Int64Array>(res);
// for(int i=0;i<3;++i) std::cout << "npoints " << res_int->Value(i) << std::endl; 

arrow::Int64Builder builder_res;
std::shared_ptr<arrow::Array> expect_res;
// builder_res.Append(4);
builder_res.Append(2);
// builder_res.Append(1);
builder_res.Finish(&expect_res);

ASSERT_EQ(res->Equals(expect_res), true);
}

TEST(geometry_test, test_ST_Envelope) {

std::shared_ptr<arrow::Array> geo;
arrow::StringBuilder builder1;

builder1.Append("POLYGON((0 0,1 0,1 1,0 0))");
builder1.Finish(&geo);

auto res = ST_Envelope(geo);

// auto res_str = std::static_pointer_cast<arrow::StringArray>(res);
// std::cout << "res_str size = " << res->length() << std::endl;
// std::cout << "res_str :" << res_str->GetString(0) <<  "#" << std::endl;

arrow::StringBuilder builder_res;
std::shared_ptr<arrow::Array> expect_res;
// builder1.Append("POLYGON((0 0,1 0,1 1,0 1,0 0))");
builder_res.Append("LINESTRING (0 0,1 0,1 1,0 0)");
builder_res.Finish(&expect_res);

ASSERT_EQ(res->Equals(expect_res), true);
}

TEST(geometry_test, test_ST_Buffer){
  std::shared_ptr<arrow::Array> geo;
  arrow::StringBuilder builder1;
  
  builder1.Append("POLYGON((0 0,1 0,1 1,0 0))");
  builder1.Finish(&geo);
  
  auto res = ST_Buffer(geo,1.2);

  auto res_str = std::static_pointer_cast<arrow::StringArray>(res);
  std::string buffer_polygon = res_str->GetString(0);
  std::string expect = "POLYGON ((-0.848528137423857 0.848528137423857,0.151471862576143 1.84852813742386,0.19704327236937 1.89177379057287,0.244815530740195 1.93257515374836,0.294657697249032 1.97082039324994,0.346433157981967 2.00640468153451,0.4 2.03923048454133,0.455211400312543 2.06920782902604,0.511916028309039 2.09625454917112,0.569958460545639 2.12029651179664,0.629179606750062 2.14126781955418,0.689417145876974 2.15911099154688,0.750505971018688 2.17377712088057,0.812278641951722 2.18522600871417,0.874565844078815 2.19342627444193,0.937196852508467 2.19835544170549,1.0 2.2,1.06280314749153 2.19835544170549,1.12543415592118 2.19342627444193,1.18772135804828 2.18522600871417,1.24949402898131 2.17377712088057,1.31058285412302 2.15911099154688,1.37082039324994 2.14126781955418,1.43004153945436 2.12029651179664,1.48808397169096 2.09625454917112,1.54478859968746 2.06920782902604,1.6 2.03923048454133,1.65356684201803 2.00640468153451,1.70534230275097 1.97082039324994,1.75518446925981 1.93257515374836,1.80295672763063 1.89177379057287,1.84852813742386 1.84852813742386,1.89177379057287 1.80295672763063,1.93257515374837 1.7551844692598,1.97082039324994 1.70534230275097,2.00640468153451 1.65356684201803,2.03923048454133 1.6,2.06920782902604 1.54478859968746,2.09625454917112 1.48808397169096,2.12029651179664 1.43004153945436,2.14126781955418 1.37082039324994,2.15911099154688 1.31058285412302,2.17377712088057 1.24949402898131,2.18522600871417 1.18772135804828,2.19342627444193 1.12543415592118,2.19835544170549 1.06280314749153,2.2 1.0,2.2 0.0,2.19835544170549 -0.062803147491532,2.19342627444193 -0.125434155921184,2.18522600871417 -0.187721358048277,2.17377712088057 -0.249494028981311,2.15911099154688 -0.310582854123025,2.14126781955418 -0.370820393249937,2.12029651179664 -0.43004153945436,2.09625454917112 -0.48808397169096,2.06920782902604 -0.544788599687456,2.03923048454133 -0.6,2.00640468153451 -0.653566842018033,1.97082039324994 -0.705342302750968,1.93257515374836 -0.755184469259805,1.89177379057287 -0.80295672763063,1.84852813742386 -0.848528137423857,1.80295672763063 -0.891773790572873,1.75518446925981 -0.932575153748365,1.70534230275097 -0.970820393249937,1.65356684201803 -1.00640468153451,1.6 -1.03923048454133,1.54478859968746 -1.06920782902604,1.48808397169096 -1.09625454917112,1.43004153945436 -1.12029651179664,1.37082039324994 -1.14126781955418,1.31058285412302 -1.15911099154688,1.24949402898131 -1.17377712088057,1.18772135804828 -1.18522600871417,1.12543415592118 -1.19342627444193,1.06280314749153 -1.19835544170549,1.0 -1.2,0.0 -1.2,-0.062803147491532 -1.19835544170549,-0.125434155921184 -1.19342627444193,-0.187721358048276 -1.18522600871417,-0.24949402898131 -1.17377712088057,-0.310582854123024 -1.15911099154688,-0.370820393249936 -1.14126781955418,-0.430041539454359 -1.12029651179664,-0.488083971690959 -1.09625454917112,-0.544788599687455 -1.06920782902604,-0.6 -1.03923048454133,-0.653566842018031 -1.00640468153451,-0.705342302750966 -0.970820393249938,-0.755184469259804 -0.932575153748366,-0.802956727630628 -0.891773790572875,-0.848528137423855 -0.848528137423859,-0.891773790572871 -0.802956727630632,-0.932575153748363 -0.755184469259807,-0.970820393249935 -0.70534230275097,-1.00640468153451 -0.653566842018035,-1.03923048454132 -0.6,-1.06920782902604 -0.544788599687459,-1.09625454917112 -0.488083971690964,-1.12029651179664 -0.430041539454364,-1.14126781955418 -0.370820393249941,-1.15911099154688 -0.310582854123029,-1.17377712088057 -0.249494028981315,-1.18522600871416 -0.187721358048281,-1.19342627444193 -0.125434155921189,-1.19835544170549 -0.062803147491537,-1.2 -0.0,-1.19835544170549 0.062803147491527,-1.19342627444193 0.125434155921179,-1.18522600871417 0.187721358048272,-1.17377712088057 0.249494028981306,-1.15911099154688 0.310582854123019,-1.14126781955419 0.370820393249931,-1.12029651179664 0.430041539454355,-1.09625454917112 0.488083971690954,-1.06920782902604 0.54478859968745,-1.03923048454133 0.6,-1.00640468153451 0.653566842018027,-0.970820393249941 0.705342302750962,-0.93257515374837 0.755184469259799,-0.891773790572878 0.802956727630624,-0.848528137423857 0.848528137423857))";
  ASSERT_EQ(buffer_polygon,expect);

}

TEST(geometry_test, test_ST_PolygonFromEnvelope){
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

  auto res = ST_PolygonFromEnvelope(x_min_ptr, y_min_ptr, x_max_ptr,y_max_ptr);

  auto res_str = std::static_pointer_cast<arrow::StringArray>(res)->GetString(0);
  std::string expect = "POLYGON ((0 0,1 0,0 1,1 1,0 0))";

  ASSERT_EQ(res_str,expect);
}
