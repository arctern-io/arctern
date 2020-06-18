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

#include "arrow/render_api.h"
#include "gis/gdal/format_conversion.h"

TEST(CHOROPLETHMAP_TEST, BLUE_TO_RED) {
  // param1: wkt string
  std::string wkt_string1 =
      "POLYGON (("
      "200 200, "
      "200 300, "
      "300 300, "
      "300 200, "
      "200 200))";
  arrow::StringBuilder string_builder;
  auto status = string_builder.Append(wkt_string1);

  std::shared_ptr<arrow::StringArray> string_array;
  status = string_builder.Finish(&string_array);

  // param2: color
  std::shared_ptr<arrow::Array> color_array;
  arrow::UInt32Builder color_builder;
  status = color_builder.Append(5);
  status = color_builder.Finish(&color_array);

  // param3: conf
  const std::string vega =
      "{\n"
      "  \"width\": 1900,\n"
      "  \"height\": 1410,\n"
      "  \"description\": \"choropleth_map\",\n"
      "  \"data\": [\n"
      "    {\n"
      "      \"name\": \"data\",\n"
      "      \"url\": \"data/data.csv\"\n"
      "    }\n"
      "  ],\n"
      "  \"scales\": [\n"
      "    {\n"
      "      \"name\": \"building\",\n"
      "      \"type\": \"linear\",\n"
      "      \"domain\": {\"data\": \"data\", \"field\": \"c0\"}\n"
      "    }\n"
      "  ],\n"
      "  \"marks\": [\n"
      "    {\n"
      "      \"encode\": {\n"
      "        \"enter\": {\n"
      "          \"bounding_box\": {\"value\": "
      "[-73.984092,40.753893,-73.977588,40.756342]},\n"
      "          \"color_gradient\": {\"value\": [\"#0000FF\", \"#FF0000\"]},\n"
      "          \"color_bound\": {\"value\": [2.5, 5]},\n"
      "          \"opacity\": {\"value\": 1.0},\n"
      "          \"aggregation_type\": {\"value\": \"sum\"}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  auto wkb = arctern::gis::gdal::WktToWkb(string_array);
  std::vector<std::shared_ptr<arrow::Array>> polygon_vec{wkb};

  std::vector<std::shared_ptr<arrow::Array>> color_vec{color_array};
  arctern::render::choropleth_map(polygon_vec, color_vec, vega);
}

TEST(CHOROPLETHMAP_TEST, SKYBLUE_TO_WHITE) {
  // param1: wkt string
  std::string wkt_string1 =
      "POLYGON (("
      "200 200, "
      "200 300, "
      "300 300, "
      "300 200, "
      "200 200))";
  arrow::StringBuilder string_builder;
  auto status = string_builder.Append(wkt_string1);

  std::shared_ptr<arrow::StringArray> string_array;
  status = string_builder.Finish(&string_array);

  // param2: color
  std::shared_ptr<arrow::Array> color_array;
  arrow::UInt32Builder color_builder;
  status = color_builder.Append(5);
  status = color_builder.Finish(&color_array);

  // param3: conf
  const std::string vega =
      "{\n"
      "  \"width\": 1900,\n"
      "  \"height\": 1410,\n"
      "  \"description\": \"choropleth_map\",\n"
      "  \"data\": [\n"
      "    {\n"
      "      \"name\": \"data\",\n"
      "      \"url\": \"data/data.csv\"\n"
      "    }\n"
      "  ],\n"
      "  \"scales\": [\n"
      "    {\n"
      "      \"name\": \"building\",\n"
      "      \"type\": \"linear\",\n"
      "      \"domain\": {\"data\": \"data\", \"field\": \"c0\"}\n"
      "    }\n"
      "  ],\n"
      "  \"marks\": [\n"
      "    {\n"
      "      \"encode\": {\n"
      "        \"enter\": {\n"
      "          \"bounding_box\": {\"value\": "
      "[-73.984092,40.753893,-73.977588,40.756342]},\n"
      "          \"color_gradient\": {\"value\": [\"#B4E7F5\", \"#FFFFFF\"]},\n"
      "          \"color_bound\": {\"value\": [2.5, 5]},\n"
      "          \"opacity\": {\"value\": 1.0},\n"
      "          \"aggregation_type\": {\"value\": \"sum\"}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  auto wkb = arctern::gis::gdal::WktToWkb(string_array);
  std::vector<std::shared_ptr<arrow::Array>> polygon_vec{wkb};

  std::vector<std::shared_ptr<arrow::Array>> color_vec{color_array};
  arctern::render::choropleth_map(polygon_vec, color_vec, vega);
}

TEST(CHOROPLETHMAP_TEST, GREEN_YELLOW_RED) {
  // param1: wkt string
  std::string wkt_string1 =
      "POLYGON (("
      "200 200, "
      "200 300, "
      "300 300, "
      "300 200, "
      "200 200))";
  arrow::StringBuilder string_builder;
  auto status = string_builder.Append(wkt_string1);

  std::shared_ptr<arrow::StringArray> string_array;
  status = string_builder.Finish(&string_array);

  // param2: color
  std::shared_ptr<arrow::Array> color_array;
  arrow::UInt32Builder color_builder;
  status = color_builder.Append(5);
  status = color_builder.Finish(&color_array);

  // param3: conf
  const std::string vega =
      "{\n"
      "  \"width\": 1900,\n"
      "  \"height\": 1410,\n"
      "  \"description\": \"choropleth_map\",\n"
      "  \"data\": [\n"
      "    {\n"
      "      \"name\": \"data\",\n"
      "      \"url\": \"data/data.csv\"\n"
      "    }\n"
      "  ],\n"
      "  \"scales\": [\n"
      "    {\n"
      "      \"name\": \"building\",\n"
      "      \"type\": \"linear\",\n"
      "      \"domain\": {\"data\": \"data\", \"field\": \"c0\"}\n"
      "    }\n"
      "  ],\n"
      "  \"marks\": [\n"
      "    {\n"
      "      \"encode\": {\n"
      "        \"enter\": {\n"
      "          \"bounding_box\": {\"value\": "
      "[-73.984092,40.753893,-73.977588,40.756342]},\n"
      "          \"color_gradient\": {\"value\": [\"#4D904F\", \"#C23728\"]},\n"
      "          \"color_bound\": {\"value\": [2.5, 5]},\n"
      "          \"opacity\": {\"value\": 1.0},\n"
      "          \"aggregation_type\": {\"value\": \"sum\"}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  auto wkb = arctern::gis::gdal::WktToWkb(string_array);
  std::vector<std::shared_ptr<arrow::Array>> polygon_vec{wkb};

  std::vector<std::shared_ptr<arrow::Array>> color_vec{color_array};
  arctern::render::choropleth_map(polygon_vec, color_vec, vega);
}

TEST(CHOROPLETHMAP_TEST, BLUE_WHITE_RED) {
  // param1: wkt string
  std::string wkt_string1 =
      "POLYGON (("
      "200 200, "
      "200 300, "
      "300 300, "
      "300 200, "
      "200 200))";
  arrow::StringBuilder string_builder;
  auto status = string_builder.Append(wkt_string1);

  std::shared_ptr<arrow::StringArray> string_array;
  status = string_builder.Finish(&string_array);

  // param2: color
  std::shared_ptr<arrow::Array> color_array;
  arrow::UInt32Builder color_builder;
  status = color_builder.Append(5);
  status = color_builder.Finish(&color_array);

  // param3: conf
  const std::string vega =
      "{\n"
      "  \"width\": 1900,\n"
      "  \"height\": 1410,\n"
      "  \"description\": \"choropleth_map\",\n"
      "  \"data\": [\n"
      "    {\n"
      "      \"name\": \"data\",\n"
      "      \"url\": \"data/data.csv\"\n"
      "    }\n"
      "  ],\n"
      "  \"scales\": [\n"
      "    {\n"
      "      \"name\": \"building\",\n"
      "      \"type\": \"linear\",\n"
      "      \"domain\": {\"data\": \"data\", \"field\": \"c0\"}\n"
      "    }\n"
      "  ],\n"
      "  \"marks\": [\n"
      "    {\n"
      "      \"encode\": {\n"
      "        \"enter\": {\n"
      "          \"bounding_box\": {\"value\": "
      "[-73.984092,40.753893,-73.977588,40.756342]},\n"
      "          \"color_gradient\": {\"value\": [\"#1984C5\", \"#C23728\"]},\n"
      "          \"color_bound\": {\"value\": [2.5, 5]},\n"
      "          \"opacity\": {\"value\": 1.0},\n"
      "          \"aggregation_type\": {\"value\": \"sum\"}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  auto wkb = arctern::gis::gdal::WktToWkb(string_array);
  std::vector<std::shared_ptr<arrow::Array>> polygon_vec{wkb};

  std::vector<std::shared_ptr<arrow::Array>> color_vec{color_array};
  arctern::render::choropleth_map(polygon_vec, color_vec, vega);
}

TEST(CHOROPLETHMAP_TEST, WHITE_BLUE) {
  // param1: wkt string
  std::string wkt_string1 =
      "POLYGON (("
      "200 200, "
      "200 300, "
      "300 300, "
      "300 200, "
      "200 200))";
  arrow::StringBuilder string_builder;
  auto status = string_builder.Append(wkt_string1);

  std::shared_ptr<arrow::StringArray> string_array;
  status = string_builder.Finish(&string_array);

  // param2: color
  std::shared_ptr<arrow::Array> color_array;
  arrow::UInt32Builder color_builder;
  status = color_builder.Append(5);
  status = color_builder.Finish(&color_array);

  // param3: conf
  const std::string vega =
      "{\n"
      "  \"width\": 1900,\n"
      "  \"height\": 1410,\n"
      "  \"description\": \"choropleth_map\",\n"
      "  \"data\": [\n"
      "    {\n"
      "      \"name\": \"data\",\n"
      "      \"url\": \"data/data.csv\"\n"
      "    }\n"
      "  ],\n"
      "  \"scales\": [\n"
      "    {\n"
      "      \"name\": \"building\",\n"
      "      \"type\": \"linear\",\n"
      "      \"domain\": {\"data\": \"data\", \"field\": \"c0\"}\n"
      "    }\n"
      "  ],\n"
      "  \"marks\": [\n"
      "    {\n"
      "      \"encode\": {\n"
      "        \"enter\": {\n"
      "          \"bounding_box\": {\"value\": "
      "[-73.984092,40.753893,-73.977588,40.756342]},\n"
      "          \"color_gradient\": {\"value\": [\"#E2E2E2\", \"#115F9A\"]},\n"
      "          \"color_bound\": {\"value\": [2.5, 5]},\n"
      "          \"opacity\": {\"value\": 1.0},\n"
      "          \"aggregation_type\": {\"value\": \"sum\"}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  auto wkb = arctern::gis::gdal::WktToWkb(string_array);
  std::vector<std::shared_ptr<arrow::Array>> polygon_vec{wkb};

  std::vector<std::shared_ptr<arrow::Array>> color_vec{color_array};
  arctern::render::choropleth_map(polygon_vec, color_vec, vega);
}

TEST(CHOROPLETHMAP_TEST, BLUE_GREEN_YELLOW) {
  // param1: wkt string
  std::string wkt_string1 =
      "POLYGON (("
      "200 200, "
      "200 300, "
      "300 300, "
      "300 200, "
      "200 200))";
  arrow::StringBuilder string_builder;
  auto status = string_builder.Append(wkt_string1);

  std::shared_ptr<arrow::StringArray> string_array;
  status = string_builder.Finish(&string_array);

  // param2: color
  std::shared_ptr<arrow::Array> color_array;
  arrow::UInt32Builder color_builder;
  status = color_builder.Append(5);
  status = color_builder.Finish(&color_array);

  // param3: conf
  const std::string vega =
      "{\n"
      "  \"width\": 1900,\n"
      "  \"height\": 1410,\n"
      "  \"description\": \"choropleth_map\",\n"
      "  \"data\": [\n"
      "    {\n"
      "      \"name\": \"data\",\n"
      "      \"url\": \"data/data.csv\"\n"
      "    }\n"
      "  ],\n"
      "  \"scales\": [\n"
      "    {\n"
      "      \"name\": \"building\",\n"
      "      \"type\": \"linear\",\n"
      "      \"domain\": {\"data\": \"data\", \"field\": \"c0\"}\n"
      "    }\n"
      "  ],\n"
      "  \"marks\": [\n"
      "    {\n"
      "      \"encode\": {\n"
      "        \"enter\": {\n"
      "          \"bounding_box\": {\"value\": "
      "[-73.984092,40.753893,-73.977588,40.756342]},\n"
      "          \"color_gradient\": {\"value\": [\"#115F9A\", \"#D0F401\"]},\n"
      "          \"color_bound\": {\"value\": [2.5, 5]},\n"
      "          \"opacity\": {\"value\": 1.0},\n"
      "          \"aggregation_type\": {\"value\": \"sum\"}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  auto wkb = arctern::gis::gdal::WktToWkb(string_array);
  std::vector<std::shared_ptr<arrow::Array>> polygon_vec{wkb};

  std::vector<std::shared_ptr<arrow::Array>> color_vec{color_array};
  arctern::render::choropleth_map(polygon_vec, color_vec, vega);
}

TEST(CHOROPLETHMAP_TEST, BLUE_TRANSPARENCY) {
  // param1: wkt string
  std::string wkt_string1 =
      "POLYGON (("
      "200 200, "
      "200 300, "
      "300 300, "
      "300 200, "
      "200 200))";
  arrow::StringBuilder string_builder;
  auto status = string_builder.Append(wkt_string1);

  std::shared_ptr<arrow::StringArray> string_array;
  status = string_builder.Finish(&string_array);

  // param2: color
  std::shared_ptr<arrow::Array> color_array;
  arrow::UInt32Builder color_builder;
  status = color_builder.Append(5);
  status = color_builder.Finish(&color_array);

  // param3: conf
  const std::string vega =
      "{\n"
      "  \"width\": 1900,\n"
      "  \"height\": 1410,\n"
      "  \"description\": \"choropleth_map\",\n"
      "  \"data\": [\n"
      "    {\n"
      "      \"name\": \"data\",\n"
      "      \"url\": \"data/data.csv\"\n"
      "    }\n"
      "  ],\n"
      "  \"scales\": [\n"
      "    {\n"
      "      \"name\": \"building\",\n"
      "      \"type\": \"linear\",\n"
      "      \"domain\": {\"data\": \"data\", \"field\": \"c0\"}\n"
      "    }\n"
      "  ],\n"
      "  \"marks\": [\n"
      "    {\n"
      "      \"encode\": {\n"
      "        \"enter\": {\n"
      "          \"bounding_box\": {\"value\": "
      "[-73.984092,40.753893,-73.977588,40.756342]},\n"
      "          \"color_gradient\": {\"value\": [\"#0000FF\", \"#0000FF\"]},\n"
      "          \"color_bound\": {\"value\": [2.5, 5]},\n"
      "          \"opacity\": {\"value\": 1.0},\n"
      "          \"aggregation_type\": {\"value\": \"sum\"}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  auto wkb = arctern::gis::gdal::WktToWkb(string_array);
  std::vector<std::shared_ptr<arrow::Array>> polygon_vec{wkb};

  std::vector<std::shared_ptr<arrow::Array>> color_vec{color_array};
  arctern::render::choropleth_map(polygon_vec, color_vec, vega);
}

TEST(CHOROPLETHMAP_TEST, RED_TRANSPARENCY) {
  // param1: wkt string
  std::string wkt_string1 =
      "POLYGON (("
      "200 200, "
      "200 300, "
      "300 300, "
      "300 200, "
      "200 200))";
  arrow::StringBuilder string_builder;
  auto status = string_builder.Append(wkt_string1);

  std::shared_ptr<arrow::StringArray> string_array;
  status = string_builder.Finish(&string_array);

  // param2: color
  std::shared_ptr<arrow::Array> color_array;
  arrow::UInt32Builder color_builder;
  status = color_builder.Append(5);
  status = color_builder.Finish(&color_array);

  // param3: conf
  const std::string vega =
      "{\n"
      "  \"width\": 1900,\n"
      "  \"height\": 1410,\n"
      "  \"description\": \"choropleth_map\",\n"
      "  \"data\": [\n"
      "    {\n"
      "      \"name\": \"data\",\n"
      "      \"url\": \"data/data.csv\"\n"
      "    }\n"
      "  ],\n"
      "  \"scales\": [\n"
      "    {\n"
      "      \"name\": \"building\",\n"
      "      \"type\": \"linear\",\n"
      "      \"domain\": {\"data\": \"data\", \"field\": \"c0\"}\n"
      "    }\n"
      "  ],\n"
      "  \"marks\": [\n"
      "    {\n"
      "      \"encode\": {\n"
      "        \"enter\": {\n"
      "          \"bounding_box\": {\"value\": "
      "[-73.984092,40.753893,-73.977588,40.756342]},\n"
      "          \"color_gradient\": {\"value\": [\"#FF0000\", \"#FF0000\"]},\n"
      "          \"color_bound\": {\"value\": [2.5, 5]},\n"
      "          \"opacity\": {\"value\": 1.0},\n"
      "          \"aggregation_type\": {\"value\": \"sum\"}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  auto wkb = arctern::gis::gdal::WktToWkb(string_array);
  std::vector<std::shared_ptr<arrow::Array>> polygon_vec{wkb};

  std::vector<std::shared_ptr<arrow::Array>> color_vec{color_array};
  arctern::render::choropleth_map(polygon_vec, color_vec, vega);
}

TEST(CHOROPLETHMAP_TEST, PURPLE_TO_YELLOW) {
  // param1: wkt string
  std::string wkt_string1 =
      "POLYGON (("
      "200 200, "
      "200 300, "
      "300 300, "
      "300 200, "
      "200 200))";
  arrow::StringBuilder string_builder;
  auto status = string_builder.Append(wkt_string1);

  std::shared_ptr<arrow::StringArray> string_array;
  status = string_builder.Finish(&string_array);

  // param2: color
  std::shared_ptr<arrow::Array> color_array;
  arrow::UInt32Builder color_builder;
  status = color_builder.Append(5);
  status = color_builder.Finish(&color_array);

  // param3: conf
  const std::string vega =
      "{\n"
      "  \"width\": 1900,\n"
      "  \"height\": 1410,\n"
      "  \"description\": \"choropleth_map\",\n"
      "  \"data\": [\n"
      "    {\n"
      "      \"name\": \"data\",\n"
      "      \"url\": \"data/data.csv\"\n"
      "    }\n"
      "  ],\n"
      "  \"scales\": [\n"
      "    {\n"
      "      \"name\": \"building\",\n"
      "      \"type\": \"linear\",\n"
      "      \"domain\": {\"data\": \"data\", \"field\": \"c0\"}\n"
      "    }\n"
      "  ],\n"
      "  \"marks\": [\n"
      "    {\n"
      "      \"encode\": {\n"
      "        \"enter\": {\n"
      "          \"bounding_box\": {\"value\": "
      "[-73.984092,40.753893,-73.977588,40.756342]},\n"
      "          \"color_gradient\": {\"value\": [\"#FF00FF\", \"#FF0000\"]},\n"
      "          \"color_bound\": {\"value\": [2.5, 5]},\n"
      "          \"opacity\": {\"value\": 1.0},\n"
      "          \"aggregation_type\": {\"value\": \"sum\"}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  auto wkb = arctern::gis::gdal::WktToWkb(string_array);
  std::vector<std::shared_ptr<arrow::Array>> polygon_vec{wkb};

  std::vector<std::shared_ptr<arrow::Array>> color_vec{color_array};
  arctern::render::choropleth_map(polygon_vec, color_vec, vega);
}

// TEST(CHOROPLETHMAP_TEST, INVALID_color_gradient_TEST) {
//  // param1: wkt string
//  std::string wkt_string1 =
//      "POLYGON (("
//      "200 200, "
//      "200 300, "
//      "300 300, "
//      "300 200, "
//      "200 200))";
//  arrow::StringBuilder string_builder;
//  auto status = string_builder.Append(wkt_string1);
//
//  std::shared_ptr<arrow::StringArray> string_array;
//  status = string_builder.Finish(&string_array);
//
//  // param2: color
//  std::shared_ptr<arrow::Array> color_array;
//  arrow::UInt32Builder color_builder;
//  status = color_builder.Append(5);
//  status = color_builder.Finish(&color_array);
//
//  // param3: conf
//  const std::string vega =
//      "{\n"
//      "  \"width\": 1900,\n"
//      "  \"height\": 1410,\n"
//      "  \"description\": \"choropleth_map\",\n"
//      "  \"data\": [\n"
//      "    {\n"
//      "      \"name\": \"data\",\n"
//      "      \"url\": \"data/data.csv\"\n"
//      "    }\n"
//      "  ],\n"
//      "  \"scales\": [\n"
//      "    {\n"
//      "      \"name\": \"building\",\n"
//      "      \"type\": \"linear\",\n"
//      "      \"domain\": {\"data\": \"data\", \"field\": \"c0\"}\n"
//      "    }\n"
//      "  ],\n"
//      "  \"marks\": [\n"
//      "    {\n"
//      "      \"encode\": {\n"
//      "        \"enter\": {\n"
//      "          \"bounding_box\": {\"value\": "
//      "[-73.984092,40.753893,-73.977588,40.756342]},\n"
//      "          \"color_gradient\": {\"value\": \"xxxxxx\"},\n"
//      "          \"color_bound\": {\"value\": [2.5, 5]},\n"
//      "          \"opacity\": {\"value\": 1.0}\n"
//      "        }\n"
//      "      }\n"
//      "    }\n"
//      "  ]\n"
//      "}";
//
//  auto wkb = arctern::gis::gdal::WktToWkb(string_array);
//  arctern::render::choropleth_map(wkb, color_array, vega);
//}

// TEST(CHOROPLETHMAP_TEST, INVALID_JSON_TEST) {
//  // param1: wkt string
//  std::string wkt_string1 =
//      "POLYGON (("
//      "200 200, "
//      "200 300, "
//      "300 300, "
//      "300 200, "
//      "200 200))";
//  arrow::StringBuilder string_builder;
//  auto status = string_builder.Append(wkt_string1);
//
//  std::shared_ptr<arrow::StringArray> string_array;
//  status = string_builder.Finish(&string_array);
//
//  // param2: color
//  std::shared_ptr<arrow::Array> color_array;
//  arrow::UInt32Builder color_builder;
//  status = color_builder.Append(5);
//  status = color_builder.Finish(&color_array);
//
//  // param3: conf
//  const std::string vega =
//      "{\n"
//      "  \"width\": 1900,\n"
//      "  \"height\": 1410,\n"
//      "  \"description\": \"choropleth_map\",\n"
//      "  \"data\": [\n"
//      "    {\n"
//      "      \"name\": \"data\",\n"
//      "      \"url\": \"data/data.csv\"\n"
//      "    }\n"
//      "  ],\n"
//      "  \"scales\": [\n"
//      "    {\n"
//      "      \"name\": \"building\",\n"
//      "      \"type\": \"linear\",\n"
//      "      \"domain\": {\"data\": \"data\", \"field\": \"c0\"}\n"
//      "    }\n"
//      "  ],\n"
//      "  \"marks\": [\n"
//      "    {\n"
//      "      \"encode\": {\n"
//      "        \"enter\": {\n"
//      "          \"bounding_box\": {\"value\": "
//      "[-73.984092,40.753893,-73.977588,40.756342]},\n"
//      "          \"color_gradient\": {\"value\": \"blue_to_red\"},\n"
//      "          \"color_bound\": {\"value\": [2.5, 5]},\n"
//      "          \"opacity\": {\"value\": \"INVALID_NUMBER\"}\n"
//      "        }\n"
//      "      }\n"
//      "    }\n"
//      "  ]\n"
//      "}";
//
//  auto wkb = arctern::gis::gdal::WktToWkb(string_array);
//  arctern::render::choropleth_map(wkb, color_array, vega);
//}

TEST(CHOROPLETHMAP_TEST, INT8) {
  // param1: wkt string
  std::string wkt_string1 =
      "POLYGON (("
      "200 200, "
      "200 300, "
      "300 300, "
      "300 200, "
      "200 200))";
  arrow::StringBuilder string_builder;
  auto status = string_builder.Append(wkt_string1);

  std::shared_ptr<arrow::StringArray> string_array;
  status = string_builder.Finish(&string_array);

  // param2: color
  std::shared_ptr<arrow::Array> color_array;
  arrow::Int8Builder color_builder;
  status = color_builder.Append(5);
  status = color_builder.Finish(&color_array);

  // param3: conf
  const std::string vega =
      "{\n"
      "  \"width\": 1900,\n"
      "  \"height\": 1410,\n"
      "  \"description\": \"choropleth_map\",\n"
      "  \"data\": [\n"
      "    {\n"
      "      \"name\": \"data\",\n"
      "      \"url\": \"data/data.csv\"\n"
      "    }\n"
      "  ],\n"
      "  \"scales\": [\n"
      "    {\n"
      "      \"name\": \"building\",\n"
      "      \"type\": \"linear\",\n"
      "      \"domain\": {\"data\": \"data\", \"field\": \"c0\"}\n"
      "    }\n"
      "  ],\n"
      "  \"marks\": [\n"
      "    {\n"
      "      \"encode\": {\n"
      "        \"enter\": {\n"
      "          \"bounding_box\": {\"value\": "
      "[-73.984092,40.753893,-73.977588,40.756342]},\n"
      "          \"color_gradient\": {\"value\": [\"#0000FF\", \"#FF0000\"]},\n"
      "          \"color_bound\": {\"value\": [2.5, 5]},\n"
      "          \"opacity\": {\"value\": 1.0},\n"
      "          \"aggregation_type\": {\"value\": \"sum\"}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  auto wkb = arctern::gis::gdal::WktToWkb(string_array);
  std::vector<std::shared_ptr<arrow::Array>> polygon_vec{wkb};

  std::vector<std::shared_ptr<arrow::Array>> color_vec{color_array};
  arctern::render::choropleth_map(polygon_vec, color_vec, vega);
}

TEST(CHOROPLETHMAP_TEST, INT16) {
  // param1: wkt string
  std::string wkt_string1 =
      "POLYGON (("
      "200 200, "
      "200 300, "
      "300 300, "
      "300 200, "
      "200 200))";
  arrow::StringBuilder string_builder;
  auto status = string_builder.Append(wkt_string1);

  std::shared_ptr<arrow::StringArray> string_array;
  status = string_builder.Finish(&string_array);

  // param2: color
  std::shared_ptr<arrow::Array> color_array;
  arrow::Int16Builder color_builder;
  status = color_builder.Append(5);
  status = color_builder.Finish(&color_array);

  // param3: conf
  const std::string vega =
      "{\n"
      "  \"width\": 1900,\n"
      "  \"height\": 1410,\n"
      "  \"description\": \"choropleth_map\",\n"
      "  \"data\": [\n"
      "    {\n"
      "      \"name\": \"data\",\n"
      "      \"url\": \"data/data.csv\"\n"
      "    }\n"
      "  ],\n"
      "  \"scales\": [\n"
      "    {\n"
      "      \"name\": \"building\",\n"
      "      \"type\": \"linear\",\n"
      "      \"domain\": {\"data\": \"data\", \"field\": \"c0\"}\n"
      "    }\n"
      "  ],\n"
      "  \"marks\": [\n"
      "    {\n"
      "      \"encode\": {\n"
      "        \"enter\": {\n"
      "          \"bounding_box\": {\"value\": "
      "[-73.984092,40.753893,-73.977588,40.756342]},\n"
      "          \"color_gradient\": {\"value\": [\"#0000FF\", \"#FF0000\"]},\n"
      "          \"color_bound\": {\"value\": [2.5, 5]},\n"
      "          \"opacity\": {\"value\": 1.0},\n"
      "          \"aggregation_type\": {\"value\": \"sum\"}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  auto wkb = arctern::gis::gdal::WktToWkb(string_array);
  std::vector<std::shared_ptr<arrow::Array>> polygon_vec{wkb};

  std::vector<std::shared_ptr<arrow::Array>> color_vec{color_array};
  arctern::render::choropleth_map(polygon_vec, color_vec, vega);
}

TEST(CHOROPLETHMAP_TEST, INT32) {
  // param1: wkt string
  std::string wkt_string1 =
      "POLYGON (("
      "200 200, "
      "200 300, "
      "300 300, "
      "300 200, "
      "200 200))";
  arrow::StringBuilder string_builder;
  auto status = string_builder.Append(wkt_string1);

  std::shared_ptr<arrow::StringArray> string_array;
  status = string_builder.Finish(&string_array);

  // param2: color
  std::shared_ptr<arrow::Array> color_array;
  arrow::Int32Builder color_builder;
  status = color_builder.Append(5);
  status = color_builder.Finish(&color_array);

  // param3: conf
  const std::string vega =
      "{\n"
      "  \"width\": 1900,\n"
      "  \"height\": 1410,\n"
      "  \"description\": \"choropleth_map\",\n"
      "  \"data\": [\n"
      "    {\n"
      "      \"name\": \"data\",\n"
      "      \"url\": \"data/data.csv\"\n"
      "    }\n"
      "  ],\n"
      "  \"scales\": [\n"
      "    {\n"
      "      \"name\": \"building\",\n"
      "      \"type\": \"linear\",\n"
      "      \"domain\": {\"data\": \"data\", \"field\": \"c0\"}\n"
      "    }\n"
      "  ],\n"
      "  \"marks\": [\n"
      "    {\n"
      "      \"encode\": {\n"
      "        \"enter\": {\n"
      "          \"bounding_box\": {\"value\": "
      "[-73.984092,40.753893,-73.977588,40.756342]},\n"
      "          \"color_gradient\": {\"value\": [\"#0000FF\", \"#FF0000\"]},\n"
      "          \"color_bound\": {\"value\": [2.5, 5]},\n"
      "          \"opacity\": {\"value\": 1.0},\n"
      "          \"aggregation_type\": {\"value\": \"sum\"}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  auto wkb = arctern::gis::gdal::WktToWkb(string_array);
  std::vector<std::shared_ptr<arrow::Array>> polygon_vec{wkb};

  std::vector<std::shared_ptr<arrow::Array>> color_vec{color_array};
  arctern::render::choropleth_map(polygon_vec, color_vec, vega);
}

TEST(CHOROPLETHMAP_TEST, INT64) {
  // param1: wkt string
  std::string wkt_string1 =
      "POLYGON (("
      "200 200, "
      "200 300, "
      "300 300, "
      "300 200, "
      "200 200))";
  arrow::StringBuilder string_builder;
  auto status = string_builder.Append(wkt_string1);

  std::shared_ptr<arrow::StringArray> string_array;
  status = string_builder.Finish(&string_array);

  // param2: color
  std::shared_ptr<arrow::Array> color_array;
  arrow::Int64Builder color_builder;
  status = color_builder.Append(5);
  status = color_builder.Finish(&color_array);

  // param3: conf
  const std::string vega =
      "{\n"
      "  \"width\": 1900,\n"
      "  \"height\": 1410,\n"
      "  \"description\": \"choropleth_map\",\n"
      "  \"data\": [\n"
      "    {\n"
      "      \"name\": \"data\",\n"
      "      \"url\": \"data/data.csv\"\n"
      "    }\n"
      "  ],\n"
      "  \"scales\": [\n"
      "    {\n"
      "      \"name\": \"building\",\n"
      "      \"type\": \"linear\",\n"
      "      \"domain\": {\"data\": \"data\", \"field\": \"c0\"}\n"
      "    }\n"
      "  ],\n"
      "  \"marks\": [\n"
      "    {\n"
      "      \"encode\": {\n"
      "        \"enter\": {\n"
      "          \"bounding_box\": {\"value\": "
      "[-73.984092,40.753893,-73.977588,40.756342]},\n"
      "          \"color_gradient\": {\"value\": [\"#0000FF\", \"#FF0000\"]},\n"
      "          \"color_bound\": {\"value\": [2.5, 5]},\n"
      "          \"opacity\": {\"value\": 1.0},\n"
      "          \"aggregation_type\": {\"value\": \"sum\"}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  auto wkb = arctern::gis::gdal::WktToWkb(string_array);
  std::vector<std::shared_ptr<arrow::Array>> polygon_vec{wkb};

  std::vector<std::shared_ptr<arrow::Array>> color_vec{color_array};
  arctern::render::choropleth_map(polygon_vec, color_vec, vega);
}

TEST(CHOROPLETHMAP_TEST, UINT8) {
  // param1: wkt string
  std::string wkt_string1 =
      "POLYGON (("
      "200 200, "
      "200 300, "
      "300 300, "
      "300 200, "
      "200 200))";
  arrow::StringBuilder string_builder;
  auto status = string_builder.Append(wkt_string1);

  std::shared_ptr<arrow::StringArray> string_array;
  status = string_builder.Finish(&string_array);

  // param2: color
  std::shared_ptr<arrow::Array> color_array;
  arrow::UInt8Builder color_builder;
  status = color_builder.Append(5);
  status = color_builder.Finish(&color_array);

  // param3: conf
  const std::string vega =
      "{\n"
      "  \"width\": 1900,\n"
      "  \"height\": 1410,\n"
      "  \"description\": \"choropleth_map\",\n"
      "  \"data\": [\n"
      "    {\n"
      "      \"name\": \"data\",\n"
      "      \"url\": \"data/data.csv\"\n"
      "    }\n"
      "  ],\n"
      "  \"scales\": [\n"
      "    {\n"
      "      \"name\": \"building\",\n"
      "      \"type\": \"linear\",\n"
      "      \"domain\": {\"data\": \"data\", \"field\": \"c0\"}\n"
      "    }\n"
      "  ],\n"
      "  \"marks\": [\n"
      "    {\n"
      "      \"encode\": {\n"
      "        \"enter\": {\n"
      "          \"bounding_box\": {\"value\": "
      "[-73.984092,40.753893,-73.977588,40.756342]},\n"
      "          \"color_gradient\": {\"value\": [\"#0000FF\", \"#FF0000\"]},\n"
      "          \"color_bound\": {\"value\": [2.5, 5]},\n"
      "          \"opacity\": {\"value\": 1.0},\n"
      "          \"aggregation_type\": {\"value\": \"sum\"}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  auto wkb = arctern::gis::gdal::WktToWkb(string_array);
  std::vector<std::shared_ptr<arrow::Array>> polygon_vec{wkb};

  std::vector<std::shared_ptr<arrow::Array>> color_vec{color_array};
  arctern::render::choropleth_map(polygon_vec, color_vec, vega);
}

TEST(CHOROPLETHMAP_TEST, UINT16) {
  // param1: wkt string
  std::string wkt_string1 =
      "POLYGON (("
      "200 200, "
      "200 300, "
      "300 300, "
      "300 200, "
      "200 200))";
  arrow::StringBuilder string_builder;
  auto status = string_builder.Append(wkt_string1);

  std::shared_ptr<arrow::StringArray> string_array;
  status = string_builder.Finish(&string_array);

  // param2: color
  std::shared_ptr<arrow::Array> color_array;
  arrow::UInt16Builder color_builder;
  status = color_builder.Append(5);
  status = color_builder.Finish(&color_array);

  // param3: conf
  const std::string vega =
      "{\n"
      "  \"width\": 1900,\n"
      "  \"height\": 1410,\n"
      "  \"description\": \"choropleth_map\",\n"
      "  \"data\": [\n"
      "    {\n"
      "      \"name\": \"data\",\n"
      "      \"url\": \"data/data.csv\"\n"
      "    }\n"
      "  ],\n"
      "  \"scales\": [\n"
      "    {\n"
      "      \"name\": \"building\",\n"
      "      \"type\": \"linear\",\n"
      "      \"domain\": {\"data\": \"data\", \"field\": \"c0\"}\n"
      "    }\n"
      "  ],\n"
      "  \"marks\": [\n"
      "    {\n"
      "      \"encode\": {\n"
      "        \"enter\": {\n"
      "          \"bounding_box\": {\"value\": "
      "[-73.984092,40.753893,-73.977588,40.756342]},\n"
      "          \"color_gradient\": {\"value\": [\"#0000FF\", \"#FF0000\"]},\n"
      "          \"color_bound\": {\"value\": [2.5, 5]},\n"
      "          \"opacity\": {\"value\": 1.0},\n"
      "          \"aggregation_type\": {\"value\": \"sum\"}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  auto wkb = arctern::gis::gdal::WktToWkb(string_array);
  std::vector<std::shared_ptr<arrow::Array>> polygon_vec{wkb};

  std::vector<std::shared_ptr<arrow::Array>> color_vec{color_array};
  arctern::render::choropleth_map(polygon_vec, color_vec, vega);
}

TEST(CHOROPLETHMAP_TEST, UINT32) {
  // param1: wkt string
  std::string wkt_string1 =
      "POLYGON (("
      "200 200, "
      "200 300, "
      "300 300, "
      "300 200, "
      "200 200))";
  arrow::StringBuilder string_builder;
  auto status = string_builder.Append(wkt_string1);

  std::shared_ptr<arrow::StringArray> string_array;
  status = string_builder.Finish(&string_array);

  // param2: color
  std::shared_ptr<arrow::Array> color_array;
  arrow::UInt32Builder color_builder;
  status = color_builder.Append(5);
  status = color_builder.Finish(&color_array);

  // param3: conf
  const std::string vega =
      "{\n"
      "  \"width\": 1900,\n"
      "  \"height\": 1410,\n"
      "  \"description\": \"choropleth_map\",\n"
      "  \"data\": [\n"
      "    {\n"
      "      \"name\": \"data\",\n"
      "      \"url\": \"data/data.csv\"\n"
      "    }\n"
      "  ],\n"
      "  \"scales\": [\n"
      "    {\n"
      "      \"name\": \"building\",\n"
      "      \"type\": \"linear\",\n"
      "      \"domain\": {\"data\": \"data\", \"field\": \"c0\"}\n"
      "    }\n"
      "  ],\n"
      "  \"marks\": [\n"
      "    {\n"
      "      \"encode\": {\n"
      "        \"enter\": {\n"
      "          \"bounding_box\": {\"value\": "
      "[-73.984092,40.753893,-73.977588,40.756342]},\n"
      "          \"color_gradient\": {\"value\": [\"#0000FF\", \"#FF0000\"]},\n"
      "          \"color_bound\": {\"value\": [2.5, 5]},\n"
      "          \"opacity\": {\"value\": 1.0},\n"
      "          \"aggregation_type\": {\"value\": \"sum\"}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  auto wkb = arctern::gis::gdal::WktToWkb(string_array);
  std::vector<std::shared_ptr<arrow::Array>> polygon_vec{wkb};

  std::vector<std::shared_ptr<arrow::Array>> color_vec{color_array};
  arctern::render::choropleth_map(polygon_vec, color_vec, vega);
}

TEST(CHOROPLETHMAP_TEST, UINT64) {
  // param1: wkt string
  std::string wkt_string1 =
      "POLYGON (("
      "200 200, "
      "200 300, "
      "300 300, "
      "300 200, "
      "200 200))";
  arrow::StringBuilder string_builder;
  auto status = string_builder.Append(wkt_string1);

  std::shared_ptr<arrow::StringArray> string_array;
  status = string_builder.Finish(&string_array);

  // param2: color
  std::shared_ptr<arrow::Array> color_array;
  arrow::UInt64Builder color_builder;
  status = color_builder.Append(5);
  status = color_builder.Finish(&color_array);

  // param3: conf
  const std::string vega =
      "{\n"
      "  \"width\": 1900,\n"
      "  \"height\": 1410,\n"
      "  \"description\": \"choropleth_map\",\n"
      "  \"data\": [\n"
      "    {\n"
      "      \"name\": \"data\",\n"
      "      \"url\": \"data/data.csv\"\n"
      "    }\n"
      "  ],\n"
      "  \"scales\": [\n"
      "    {\n"
      "      \"name\": \"building\",\n"
      "      \"type\": \"linear\",\n"
      "      \"domain\": {\"data\": \"data\", \"field\": \"c0\"}\n"
      "    }\n"
      "  ],\n"
      "  \"marks\": [\n"
      "    {\n"
      "      \"encode\": {\n"
      "        \"enter\": {\n"
      "          \"bounding_box\": {\"value\": "
      "[-73.984092,40.753893,-73.977588,40.756342]},\n"
      "          \"color_gradient\": {\"value\": [\"#0000FF\", \"#FF0000\"]},\n"
      "          \"color_bound\": {\"value\": [2.5, 5]},\n"
      "          \"opacity\": {\"value\": 1.0},\n"
      "          \"aggregation_type\": {\"value\": \"sum\"}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  auto wkb = arctern::gis::gdal::WktToWkb(string_array);
  std::vector<std::shared_ptr<arrow::Array>> polygon_vec{wkb};

  std::vector<std::shared_ptr<arrow::Array>> color_vec{color_array};
  arctern::render::choropleth_map(polygon_vec, color_vec, vega);
}

TEST(CHOROPLETHMAP_TEST, FLOAT) {
  // param1: wkt string
  std::string wkt_string1 =
      "POLYGON (("
      "200 200, "
      "200 300, "
      "300 300, "
      "300 200, "
      "200 200))";
  arrow::StringBuilder string_builder;
  auto status = string_builder.Append(wkt_string1);

  std::shared_ptr<arrow::StringArray> string_array;
  status = string_builder.Finish(&string_array);

  // param2: color
  std::shared_ptr<arrow::Array> color_array;
  arrow::FloatBuilder color_builder;
  status = color_builder.Append(5);
  status = color_builder.Finish(&color_array);

  // param3: conf
  const std::string vega =
      "{\n"
      "  \"width\": 1900,\n"
      "  \"height\": 1410,\n"
      "  \"description\": \"choropleth_map\",\n"
      "  \"data\": [\n"
      "    {\n"
      "      \"name\": \"data\",\n"
      "      \"url\": \"data/data.csv\"\n"
      "    }\n"
      "  ],\n"
      "  \"scales\": [\n"
      "    {\n"
      "      \"name\": \"building\",\n"
      "      \"type\": \"linear\",\n"
      "      \"domain\": {\"data\": \"data\", \"field\": \"c0\"}\n"
      "    }\n"
      "  ],\n"
      "  \"marks\": [\n"
      "    {\n"
      "      \"encode\": {\n"
      "        \"enter\": {\n"
      "          \"bounding_box\": {\"value\": "
      "[-73.984092,40.753893,-73.977588,40.756342]},\n"
      "          \"color_gradient\": {\"value\": [\"#0000FF\", \"#FF0000\"]},\n"
      "          \"color_bound\": {\"value\": [2.5, 5]},\n"
      "          \"opacity\": {\"value\": 1.0},\n"
      "          \"aggregation_type\": {\"value\": \"sum\"}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  auto wkb = arctern::gis::gdal::WktToWkb(string_array);
  std::vector<std::shared_ptr<arrow::Array>> polygon_vec{wkb};

  std::vector<std::shared_ptr<arrow::Array>> color_vec{color_array};
  arctern::render::choropleth_map(polygon_vec, color_vec, vega);
}

TEST(CHOROPLETHMAP_TEST, DOUBLE) {
  // param1: wkt string
  std::string wkt_string1 =
      "POLYGON (("
      "200 200, "
      "200 300, "
      "300 300, "
      "300 200, "
      "200 200))";
  arrow::StringBuilder string_builder;
  auto status = string_builder.Append(wkt_string1);

  std::shared_ptr<arrow::StringArray> string_array;
  status = string_builder.Finish(&string_array);

  // param2: color
  std::shared_ptr<arrow::Array> color_array;
  arrow::DoubleBuilder color_builder;
  status = color_builder.Append(5);
  status = color_builder.Finish(&color_array);

  // param3: conf
  const std::string vega =
      "{\n"
      "  \"width\": 1900,\n"
      "  \"height\": 1410,\n"
      "  \"description\": \"choropleth_map\",\n"
      "  \"data\": [\n"
      "    {\n"
      "      \"name\": \"data\",\n"
      "      \"url\": \"data/data.csv\"\n"
      "    }\n"
      "  ],\n"
      "  \"scales\": [\n"
      "    {\n"
      "      \"name\": \"building\",\n"
      "      \"type\": \"linear\",\n"
      "      \"domain\": {\"data\": \"data\", \"field\": \"c0\"}\n"
      "    }\n"
      "  ],\n"
      "  \"marks\": [\n"
      "    {\n"
      "      \"encode\": {\n"
      "        \"enter\": {\n"
      "          \"bounding_box\": {\"value\": "
      "[-73.984092,40.753893,-73.977588,40.756342]},\n"
      "          \"color_gradient\": {\"value\": [\"#0000FF\", \"#FF0000\"]},\n"
      "          \"color_bound\": {\"value\": [2.5, 5]},\n"
      "          \"opacity\": {\"value\": 1.0},\n"
      "          \"aggregation_type\": {\"value\": \"sum\"}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  auto wkb = arctern::gis::gdal::WktToWkb(string_array);
  std::vector<std::shared_ptr<arrow::Array>> polygon_vec{wkb};

  std::vector<std::shared_ptr<arrow::Array>> color_vec{color_array};
  arctern::render::choropleth_map(polygon_vec, color_vec, vega);
}

// TEST(CHOROPLETHMAP_TEST, INVALID_DATA_TYPE_TEST) {
//  // param1: wkt string
//  std::string wkt_string1 =
//      "POLYGON (("
//      "200 200, "
//      "200 300, "
//      "300 300, "
//      "300 200, "
//      "200 200))";
//  arrow::StringBuilder string_builder;
//  auto status = string_builder.Append(wkt_string1);
//
//  std::shared_ptr<arrow::StringArray> string_array;
//  status = string_builder.Finish(&string_array);
//
//  // param2: color
//  std::shared_ptr<arrow::Array> color_array;
//  arrow::StringBuilder color_builder;
//  status = color_builder.Append("");
//  status = color_builder.Finish(&color_array);
//
//  // param3: conf
//  const std::string vega =
//      "{\n"
//      "  \"width\": 1900,\n"
//      "  \"height\": 1410,\n"
//      "  \"description\": \"choropleth_map\",\n"
//      "  \"data\": [\n"
//      "    {\n"
//      "      \"name\": \"data\",\n"
//      "      \"url\": \"data/data.csv\"\n"
//      "    }\n"
//      "  ],\n"
//      "  \"scales\": [\n"
//      "    {\n"
//      "      \"name\": \"building\",\n"
//      "      \"type\": \"linear\",\n"
//      "      \"domain\": {\"data\": \"data\", \"field\": \"c0\"}\n"
//      "    }\n"
//      "  ],\n"
//      "  \"marks\": [\n"
//      "    {\n"
//      "      \"encode\": {\n"
//      "        \"enter\": {\n"
//      "          \"bounding_box\": {\"value\": "
//      "[-73.984092,40.753893,-73.977588,40.756342]},\n"
//      "          \"color_gradient\": {\"value\": \"blue_to_red\"},\n"
//      "          \"color_bound\": {\"value\": [2.5, 5]},\n"
//      "          \"opacity\": {\"value\": 1.0}\n"
//      "        }\n"
//      "      }\n"
//      "    }\n"
//      "  ]\n"
//      "}";
//
//  auto wkb = arctern::gis::gdal::WktToWkb(string_array);
//  arctern::render::choropleth_map(wkb, color_array, vega);
//}

TEST(CHOROPLETHMAP_TEST, MEAN) {
  // param1: wkt string
  std::string wkt_string1 =
      "POLYGON (("
      "200 200, "
      "200 300, "
      "300 300, "
      "300 200, "
      "200 200))";
  arrow::StringBuilder string_builder;
  auto status = string_builder.Append(wkt_string1);

  std::shared_ptr<arrow::StringArray> string_array;
  status = string_builder.Finish(&string_array);

  // param2: color
  std::shared_ptr<arrow::Array> color_array;
  arrow::DoubleBuilder color_builder;
  status = color_builder.Append(5);
  status = color_builder.Finish(&color_array);

  // param3: conf
  const std::string vega =
      "{\n"
      "  \"width\": 1900,\n"
      "  \"height\": 1410,\n"
      "  \"description\": \"choropleth_map\",\n"
      "  \"data\": [\n"
      "    {\n"
      "      \"name\": \"data\",\n"
      "      \"url\": \"data/data.csv\"\n"
      "    }\n"
      "  ],\n"
      "  \"scales\": [\n"
      "    {\n"
      "      \"name\": \"building\",\n"
      "      \"type\": \"linear\",\n"
      "      \"domain\": {\"data\": \"data\", \"field\": \"c0\"}\n"
      "    }\n"
      "  ],\n"
      "  \"marks\": [\n"
      "    {\n"
      "      \"encode\": {\n"
      "        \"enter\": {\n"
      "          \"bounding_box\": {\"value\": "
      "[-73.984092,40.753893,-73.977588,40.756342]},\n"
      "          \"color_gradient\": {\"value\": [\"#0000FF\", \"#FF0000\"]},\n"
      "          \"color_bound\": {\"value\": [2.5, 5]},\n"
      "          \"opacity\": {\"value\": 1.0},\n"
      "          \"aggregation_type\": {\"value\": \"mean\"}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  auto wkb = arctern::gis::gdal::WktToWkb(string_array);
  std::vector<std::shared_ptr<arrow::Array>> polygon_vec{wkb};

  std::vector<std::shared_ptr<arrow::Array>> color_vec{color_array};
  arctern::render::choropleth_map(polygon_vec, color_vec, vega);
}

TEST(CHOROPLETHMAP_TEST, SUM) {
  // param1: wkt string
  std::string wkt_string1 =
      "POLYGON (("
      "200 200, "
      "200 300, "
      "300 300, "
      "300 200, "
      "200 200))";
  arrow::StringBuilder string_builder;
  auto status = string_builder.Append(wkt_string1);

  std::shared_ptr<arrow::StringArray> string_array;
  status = string_builder.Finish(&string_array);

  // param2: color
  std::shared_ptr<arrow::Array> color_array;
  arrow::DoubleBuilder color_builder;
  status = color_builder.Append(5);
  status = color_builder.Finish(&color_array);

  // param3: conf
  const std::string vega =
      "{\n"
      "  \"width\": 1900,\n"
      "  \"height\": 1410,\n"
      "  \"description\": \"choropleth_map\",\n"
      "  \"data\": [\n"
      "    {\n"
      "      \"name\": \"data\",\n"
      "      \"url\": \"data/data.csv\"\n"
      "    }\n"
      "  ],\n"
      "  \"scales\": [\n"
      "    {\n"
      "      \"name\": \"building\",\n"
      "      \"type\": \"linear\",\n"
      "      \"domain\": {\"data\": \"data\", \"field\": \"c0\"}\n"
      "    }\n"
      "  ],\n"
      "  \"marks\": [\n"
      "    {\n"
      "      \"encode\": {\n"
      "        \"enter\": {\n"
      "          \"bounding_box\": {\"value\": "
      "[-73.984092,40.753893,-73.977588,40.756342]},\n"
      "          \"color_gradient\": {\"value\": [\"#0000FF\", \"#FF0000\"]},\n"
      "          \"color_bound\": {\"value\": [2.5, 5]},\n"
      "          \"opacity\": {\"value\": 1.0},\n"
      "          \"aggregation_type\": {\"value\": \"sum\"}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  auto wkb = arctern::gis::gdal::WktToWkb(string_array);
  std::vector<std::shared_ptr<arrow::Array>> polygon_vec{wkb};

  std::vector<std::shared_ptr<arrow::Array>> color_vec{color_array};
  arctern::render::choropleth_map(polygon_vec, color_vec, vega);
}

TEST(CHOROPLETHMAP_TEST, MAX) {
  // param1: wkt string
  std::string wkt_string1 =
      "POLYGON (("
      "200 200, "
      "200 300, "
      "300 300, "
      "300 200, "
      "200 200))";
  arrow::StringBuilder string_builder;
  auto status = string_builder.Append(wkt_string1);

  std::shared_ptr<arrow::StringArray> string_array;
  status = string_builder.Finish(&string_array);

  // param2: color
  std::shared_ptr<arrow::Array> color_array;
  arrow::DoubleBuilder color_builder;
  status = color_builder.Append(5);
  status = color_builder.Finish(&color_array);

  // param3: conf
  const std::string vega =
      "{\n"
      "  \"width\": 1900,\n"
      "  \"height\": 1410,\n"
      "  \"description\": \"choropleth_map\",\n"
      "  \"data\": [\n"
      "    {\n"
      "      \"name\": \"data\",\n"
      "      \"url\": \"data/data.csv\"\n"
      "    }\n"
      "  ],\n"
      "  \"scales\": [\n"
      "    {\n"
      "      \"name\": \"building\",\n"
      "      \"type\": \"linear\",\n"
      "      \"domain\": {\"data\": \"data\", \"field\": \"c0\"}\n"
      "    }\n"
      "  ],\n"
      "  \"marks\": [\n"
      "    {\n"
      "      \"encode\": {\n"
      "        \"enter\": {\n"
      "          \"bounding_box\": {\"value\": "
      "[-73.984092,40.753893,-73.977588,40.756342]},\n"
      "          \"color_gradient\": {\"value\": [\"#0000FF\", \"#FF0000\"]},\n"
      "          \"color_bound\": {\"value\": [2.5, 5]},\n"
      "          \"opacity\": {\"value\": 1.0},\n"
      "          \"aggregation_type\": {\"value\": \"max\"}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  auto wkb = arctern::gis::gdal::WktToWkb(string_array);
  std::vector<std::shared_ptr<arrow::Array>> polygon_vec{wkb};

  std::vector<std::shared_ptr<arrow::Array>> color_vec{color_array};
  arctern::render::choropleth_map(polygon_vec, color_vec, vega);
}

TEST(CHOROPLETHMAP_TEST, MIN) {
  // param1: wkt string
  std::string wkt_string1 =
      "POLYGON (("
      "200 200, "
      "200 300, "
      "300 300, "
      "300 200, "
      "200 200))";
  arrow::StringBuilder string_builder;
  auto status = string_builder.Append(wkt_string1);

  std::shared_ptr<arrow::StringArray> string_array;
  status = string_builder.Finish(&string_array);

  // param2: color
  std::shared_ptr<arrow::Array> color_array;
  arrow::DoubleBuilder color_builder;
  status = color_builder.Append(5);
  status = color_builder.Finish(&color_array);

  // param3: conf
  const std::string vega =
      "{\n"
      "  \"width\": 1900,\n"
      "  \"height\": 1410,\n"
      "  \"description\": \"choropleth_map\",\n"
      "  \"data\": [\n"
      "    {\n"
      "      \"name\": \"data\",\n"
      "      \"url\": \"data/data.csv\"\n"
      "    }\n"
      "  ],\n"
      "  \"scales\": [\n"
      "    {\n"
      "      \"name\": \"building\",\n"
      "      \"type\": \"linear\",\n"
      "      \"domain\": {\"data\": \"data\", \"field\": \"c0\"}\n"
      "    }\n"
      "  ],\n"
      "  \"marks\": [\n"
      "    {\n"
      "      \"encode\": {\n"
      "        \"enter\": {\n"
      "          \"bounding_box\": {\"value\": "
      "[-73.984092,40.753893,-73.977588,40.756342]},\n"
      "          \"color_gradient\": {\"value\": [\"#0000FF\", \"#FF0000\"]},\n"
      "          \"color_bound\": {\"value\": [2.5, 5]},\n"
      "          \"opacity\": {\"value\": 1.0},\n"
      "          \"aggregation_type\": {\"value\": \"min\"}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  auto wkb = arctern::gis::gdal::WktToWkb(string_array);
  std::vector<std::shared_ptr<arrow::Array>> polygon_vec{wkb};

  std::vector<std::shared_ptr<arrow::Array>> color_vec{color_array};
  arctern::render::choropleth_map(polygon_vec, color_vec, vega);
}

TEST(CHOROPLETHMAP_TEST, COUNT) {
  // param1: wkt string
  std::string wkt_string1 =
      "POLYGON (("
      "200 200, "
      "200 300, "
      "300 300, "
      "300 200, "
      "200 200))";
  arrow::StringBuilder string_builder;
  auto status = string_builder.Append(wkt_string1);

  std::shared_ptr<arrow::StringArray> string_array;
  status = string_builder.Finish(&string_array);

  // param2: color
  std::shared_ptr<arrow::Array> color_array;
  arrow::DoubleBuilder color_builder;
  status = color_builder.Append(5);
  status = color_builder.Finish(&color_array);

  // param3: conf
  const std::string vega =
      "{\n"
      "  \"width\": 1900,\n"
      "  \"height\": 1410,\n"
      "  \"description\": \"choropleth_map\",\n"
      "  \"data\": [\n"
      "    {\n"
      "      \"name\": \"data\",\n"
      "      \"url\": \"data/data.csv\"\n"
      "    }\n"
      "  ],\n"
      "  \"scales\": [\n"
      "    {\n"
      "      \"name\": \"building\",\n"
      "      \"type\": \"linear\",\n"
      "      \"domain\": {\"data\": \"data\", \"field\": \"c0\"}\n"
      "    }\n"
      "  ],\n"
      "  \"marks\": [\n"
      "    {\n"
      "      \"encode\": {\n"
      "        \"enter\": {\n"
      "          \"bounding_box\": {\"value\": "
      "[-73.984092,40.753893,-73.977588,40.756342]},\n"
      "          \"color_gradient\": {\"value\": [\"#0000FF\", \"#FF0000\"]},\n"
      "          \"color_bound\": {\"value\": [2.5, 5]},\n"
      "          \"opacity\": {\"value\": 1.0},\n"
      "          \"aggregation_type\": {\"value\": \"count\"}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  auto wkb = arctern::gis::gdal::WktToWkb(string_array);
  std::vector<std::shared_ptr<arrow::Array>> polygon_vec{wkb};

  std::vector<std::shared_ptr<arrow::Array>> color_vec{color_array};
  arctern::render::choropleth_map(polygon_vec, color_vec, vega);
}

TEST(CHOROPLETHMAP_TEST, STD) {
  // param1: wkt string
  std::string wkt_string1 =
      "POLYGON (("
      "200 200, "
      "200 300, "
      "300 300, "
      "300 200, "
      "200 200))";
  arrow::StringBuilder string_builder;
  auto status = string_builder.Append(wkt_string1);

  std::shared_ptr<arrow::StringArray> string_array;
  status = string_builder.Finish(&string_array);

  // param2: color
  std::shared_ptr<arrow::Array> color_array;
  arrow::DoubleBuilder color_builder;
  status = color_builder.Append(5);
  status = color_builder.Finish(&color_array);

  // param3: conf
  const std::string vega =
      "{\n"
      "  \"width\": 1900,\n"
      "  \"height\": 1410,\n"
      "  \"description\": \"choropleth_map\",\n"
      "  \"data\": [\n"
      "    {\n"
      "      \"name\": \"data\",\n"
      "      \"url\": \"data/data.csv\"\n"
      "    }\n"
      "  ],\n"
      "  \"scales\": [\n"
      "    {\n"
      "      \"name\": \"building\",\n"
      "      \"type\": \"linear\",\n"
      "      \"domain\": {\"data\": \"data\", \"field\": \"c0\"}\n"
      "    }\n"
      "  ],\n"
      "  \"marks\": [\n"
      "    {\n"
      "      \"encode\": {\n"
      "        \"enter\": {\n"
      "          \"bounding_box\": {\"value\": "
      "[-73.984092,40.753893,-73.977588,40.756342]},\n"
      "          \"color_gradient\": {\"value\": [\"#0000FF\", \"#FF0000\"]},\n"
      "          \"color_bound\": {\"value\": [2.5, 5]},\n"
      "          \"opacity\": {\"value\": 1.0},\n"
      "          \"aggregation_type\": {\"value\": \"std\"}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  auto wkb = arctern::gis::gdal::WktToWkb(string_array);
  std::vector<std::shared_ptr<arrow::Array>> polygon_vec{wkb};

  std::vector<std::shared_ptr<arrow::Array>> color_vec{color_array};
  arctern::render::choropleth_map(polygon_vec, color_vec, vega);
}
