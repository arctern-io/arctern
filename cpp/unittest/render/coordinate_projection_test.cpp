

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
#include <gtest/gtest.h>
#include <ogr_geometry.h>

#include "arrow/render_api.h"
#include "gis/gdal/format_conversion.h"

TEST(TRANSFORM_PROJECTION_TEST, POINT_TEST) {
  // param1: wkt string
  std::string wkt1 = "POINT (-73.978003 40.754594)";
  std::string wkt2 = "POINT (-73.978185 40.755587)";
  arrow::StringBuilder string_builder;
  auto status = string_builder.Append(wkt1);
  status = string_builder.Append(wkt2);

  std::shared_ptr<arrow::StringArray> string_array;
  status = string_builder.Finish(&string_array);

  auto wkb = arctern::gis::gdal::WktToWkb(string_array);

  // param2: src_rs
  std::string src_ts = "EPSG:4326";

  // param3: dst_rs
  std::string dst_rs = "EPSG:3857";

  // param4: top_left
  std::string top_left = "POINT (-73.984092 40.756342)";

  // param5: bottom_right
  std::string bottom_right = "POINT (-73.977588 40.753893)";

  std::vector<std::shared_ptr<arrow::Array>> vec{wkb};
  auto arr = arctern::render::transform_and_projection(vec, src_ts, dst_rs, bottom_right,
                                                       top_left, 200, 300);

  auto str_arr = std::static_pointer_cast<arrow::BinaryArray>(arr[0]);
  auto res1 = str_arr->GetString(0);
  auto res2 = str_arr->GetString(1);

  OGRGeometry* res_geo1 = nullptr;
  OGRGeometry* res_geo2 = nullptr;

  OGRGeometryFactory::createFromWkb(res1.c_str(), nullptr, &res_geo1);
  OGRGeometryFactory::createFromWkb(res2.c_str(), nullptr, &res_geo2);

  assert(res_geo1->toPoint()->getX() == 280);
  assert(res_geo1->toPoint()->getY() == 57);
  assert(res_geo2->toPoint()->getX() == 272);
  assert(res_geo2->toPoint()->getY() == 138);

  OGRGeometryFactory::destroyGeometry(res_geo1);
  OGRGeometryFactory::destroyGeometry(res_geo2);
}

TEST(TRANSFORM_PROJECTION_TEST, POLYGON_TEST) {
  // param1: wkt string    -73.984092,40.753893,-73.977588,40.756342
  std::string wkt1 =
      "POLYGON (("
      "-73.989754263774 40.7677468202825,-73.9899519048903 40.7678302792556,"
      "-73.989912476786 40.7678842519974,-73.9899105593281 40.7678834422768,"
      "-73.9899028933374 40.7678939333729,-73.9897724980032 40.7678388704833,"
      "-73.989737963688 40.7678242873584,-73.9897071707312 40.7678112849412,"
      "-73.9897080734511 40.7678100513318,-73.9897150393223 40.7678005156204,"
      "-73.989754263774 40.7677468202825))";
  std::string wkt2 =
      "POLYGON (("
      "-73.9870535105538 40.7601363221624,-73.9871923522742 40.7599470020734,"
      "-73.9874629426019 40.7600617372231,-73.9875093817809 40.7600814281999,"
      "-73.9874059754083 40.7602224305671,-73.9873933701613 40.7602396189215,"
      "-73.9873705406814 40.7602707486673,-73.9873691883338 40.7602701748976,"
      "-73.9870535105538 40.7601363221624))";

  arrow::StringBuilder string_builder;
  auto status = string_builder.Append(wkt1);
  status = string_builder.Append(wkt2);

  std::shared_ptr<arrow::StringArray> string_array;
  status = string_builder.Finish(&string_array);

  auto wkb = arctern::gis::gdal::WktToWkb(string_array);

  // param2: src_rs
  std::string src_ts = "EPSG:4326";

  // param3: dst_rs
  std::string dst_rs = "EPSG:3857";

  // param4: top_left
  std::string top_left = "POINT (-73.984092 40.756342)";

  // param5: bottom_right
  std::string bottom_right = "POINT (-73.977588 40.753893)";

  std::vector<std::shared_ptr<arrow::Array>> vec{wkb};

  auto arr = arctern::render::transform_and_projection(vec, src_ts, dst_rs, bottom_right,
                                                       top_left, 200, 300);

  auto str_arr = std::static_pointer_cast<arrow::BinaryArray>(arr[0]);
}

TEST(PROJECTION_TEST, POINT_TEST) {
  // param1: wkt string
  std::string wkt1 = "POINT (-8235193.62386326 4976211.44428777)";
  std::string wkt2 = "POINT (-8235213.88401059 4976357.37067044)";

  // param2: top_left
  std::string top_left = "POINT (-8235871.4482427 4976468.32320551)";

  // param3: bottom_right
  std::string bottom_right = "POINT (-8235147.42627458 4976108.43009739)";

  arrow::StringBuilder string_builder;
  auto status = string_builder.Append(wkt1);
  status = string_builder.Append(wkt2);

  std::shared_ptr<arrow::StringArray> string_array;
  status = string_builder.Finish(&string_array);

  auto wkb = arctern::gis::gdal::WktToWkb(string_array);

  std::vector<std::shared_ptr<arrow::Array>> vec{wkb};
  auto arr = arctern::render::projection(vec, bottom_right, top_left, 200, 300);

  auto str_arr = std::static_pointer_cast<arrow::BinaryArray>(arr[0]);

  auto res1 = str_arr->GetString(0);
  auto res2 = str_arr->GetString(1);

  OGRGeometry* res_geo1 = nullptr;
  OGRGeometry* res_geo2 = nullptr;

  OGRGeometryFactory::createFromWkb(res1.c_str(), nullptr, &res_geo1);
  OGRGeometryFactory::createFromWkb(res2.c_str(), nullptr, &res_geo2);

  assert(res_geo1->toPoint()->getX() == 280);
  assert(res_geo1->toPoint()->getY() == 57);
  assert(res_geo2->toPoint()->getX() == 272);
  assert(res_geo2->toPoint()->getY() == 138);

  OGRGeometryFactory::destroyGeometry(res_geo1);
  OGRGeometryFactory::destroyGeometry(res_geo2);
}

TEST(PROJECTION_TEST, POLYGON_TEST) {
  // param1: wkt string
  std::string polygon =
      "POLYGON (("
      "-8235193.62386326 4976211.44428777,"
      "-8235213.88401059 4976211.44428777,"
      "-8235213.88401059 4976357.37067044,"
      "-8235193.62386326 4976357.37067044,"
      "-8235193.62386326 4976211.44428777))";

  // param2: top_left
  std::string top_left = "POINT (-8235871.4482427 4976468.32320551)";

  // param3: bottom_right
  std::string bottom_right = "POINT (-8235147.42627458 4976108.43009739)";

  arrow::StringBuilder string_builder;
  auto status = string_builder.Append(polygon);

  std::shared_ptr<arrow::StringArray> string_array;
  status = string_builder.Finish(&string_array);

  auto wkb = arctern::gis::gdal::WktToWkb(string_array);

  std::vector<std::shared_ptr<arrow::Array>> vec{wkb};
  auto arr = arctern::render::projection(vec, bottom_right, top_left, 200, 300);

  auto str_arr = std::static_pointer_cast<arrow::BinaryArray>(arr[0]);

  auto res = str_arr->GetString(0);

  OGRGeometry* res_geo = nullptr;

  OGRGeometryFactory::createFromWkb(res.c_str(), nullptr, &res_geo);

  assert(res_geo->toPolygon()->getExteriorRing()->getX(0) == 280);
  assert(res_geo->toPolygon()->getExteriorRing()->getY(0) == 57);
  assert(res_geo->toPolygon()->getExteriorRing()->getX(1) == 272);
  assert(res_geo->toPolygon()->getExteriorRing()->getY(1) == 57);
  assert(res_geo->toPolygon()->getExteriorRing()->getX(2) == 272);
  assert(res_geo->toPolygon()->getExteriorRing()->getY(2) == 138);
  assert(res_geo->toPolygon()->getExteriorRing()->getX(3) == 280);
  assert(res_geo->toPolygon()->getExteriorRing()->getY(3) == 138);
  assert(res_geo->toPolygon()->getExteriorRing()->getX(4) == 280);
  assert(res_geo->toPolygon()->getExteriorRing()->getY(4) == 57);

  OGRGeometryFactory::destroyGeometry(res_geo);
}

TEST(TRANSFORM_PROJECTION_TEST, NULL_TEST) {
  // param1: wkt string    -73.984092,40.753893,-73.977588,40.756342
  std::string wkt1 =
      "POLYGON (("
      "-73.989754263774 40.7677468202825,-73.9899519048903 40.7678302792556,"
      "-73.989912476786 40.7678842519974,-73.9899105593281 40.7678834422768,"
      "-73.9899028933374 40.7678939333729,-73.9897724980032 40.7678388704833,"
      "-73.989737963688 40.7678242873584,-73.9897071707312 40.7678112849412,"
      "-73.9897080734511 40.7678100513318,-73.9897150393223 40.7678005156204,"
      "-73.989754263774 40.7677468202825))";

  arrow::StringBuilder string_builder;
  auto status = string_builder.Append(wkt1);
  status = string_builder.AppendNull();

  std::shared_ptr<arrow::StringArray> string_array;
  status = string_builder.Finish(&string_array);

  auto wkb = arctern::gis::gdal::WktToWkb(string_array);

  // param2: src_rs
  std::string src_ts = "EPSG:4326";

  // param3: dst_rs
  std::string dst_rs = "EPSG:3857";

  // param4: top_left
  std::string top_left = "POINT (-73.984092 40.756342)";

  // param5: bottom_right
  std::string bottom_right = "POINT (-73.977588 40.753893)";

  std::vector<std::shared_ptr<arrow::Array>> vec{wkb};

  auto arr = arctern::render::transform_and_projection(vec, src_ts, dst_rs, bottom_right,
                                                       top_left, 200, 300);

  auto str_arr = std::static_pointer_cast<arrow::BinaryArray>(arr[0]);
}
