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

// TEST(HEATMAP_TEST, RAW_POINT_INVALID_DATA_TYPE_TEST) {
//  // param1: x
//  arrow::UInt32Builder x_builder;
//  auto status = x_builder.Append(50);
//  status = x_builder.Append(50);
//  status = x_builder.Append(50);
//  status = x_builder.Append(50);
//  status = x_builder.Append(50);
//
//  std::shared_ptr<arrow::UInt32Array> x_array;
//  status = x_builder.Finish(&x_array);
//
//  // param2: y
//  arrow::UInt32Builder y_builder;
//  status = y_builder.Append(50);
//  status = y_builder.Append(50);
//  status = y_builder.Append(50);
//  status = y_builder.Append(50);
//  status = y_builder.Append(50);
//
//  std::shared_ptr<arrow::UInt32Array> y_array;
//  status = y_builder.Finish(&y_array);
//
//  // param2: color
//  std::shared_ptr<arrow::Array> color_array;
//  arrow::StringBuilder color_builder;
//  status = color_builder.Append("");
//  status = color_builder.Append("");
//  status = color_builder.Append("");
//  status = color_builder.Append("");
//  status = color_builder.Append("");
//  status = color_builder.Finish(&color_array);
//
//  // param3: conf
//  const std::string vega =
//      "{\n"
//      "  \"width\": 300,\n"
//      "  \"height\": 200,\n"
//      "  \"description\": \"circle_2d\",\n"
//      "  \"data\": [\n"
//      "    {\n"
//      "      \"name\": \"data\",\n"
//      "      \"url\": \"data/data.csv\"\n"
//      "    }\n"
//      "  ],\n"
//      "  \"scales\": [\n"
//      "    {\n"
//      "      \"name\": \"x\",\n"
//      "      \"type\": \"linear\",\n"
//      "      \"domain\": {\"data\": \"data\", \"field\": \"c0\"}\n"
//      "    },\n"
//      "    {\n"
//      "      \"name\": \"y\",\n"
//      "      \"type\": \"linear\",\n"
//      "      \"domain\": {\"data\": \"data\", \"field\": \"c1\"}\n"
//      "    }\n"
//      "  ],\n"
//      "  \"marks\": [\n"
//      "    {\n"
//      "      \"encode\": {\n"
//      "        \"enter\": {\n"
//      "          \"map_zoom_level\": {\"value\": 10}\n"
//      "        }\n"
//      "      }\n"
//      "    }\n"
//      "  ]\n"
//      "}";
//
//  arctern::render::heat_map(x_array, y_array, color_array, vega);
//}

TEST(HEATMAP_TEST, WKT_POINT_INT8_TEST) {
  // param1: wkt string
  std::string wkt1 = "POINT (50 50)";
  std::string wkt2 = "POINT (51 51)";
  std::string wkt3 = "POINT (52 52)";
  std::string wkt4 = "POINT (53 53)";
  std::string wkt5 = "POINT (54 54)";
  arrow::StringBuilder string_builder;
  auto status = string_builder.Append(wkt1);
  status = string_builder.Append(wkt2);
  status = string_builder.Append(wkt3);
  status = string_builder.Append(wkt4);
  status = string_builder.Append(wkt5);

  std::shared_ptr<arrow::StringArray> string_array;
  status = string_builder.Finish(&string_array);

  // param2: color
  std::shared_ptr<arrow::Array> color_array;
  arrow::Int8Builder color_builder;
  status = color_builder.Append(50);
  status = color_builder.Append(51);
  status = color_builder.Append(52);
  status = color_builder.Append(53);
  status = color_builder.Append(54);
  status = color_builder.Finish(&color_array);

  // param3: conf
  const std::string vega =
      "{\n"
      "  \"width\": 300,\n"
      "  \"height\": 200,\n"
      "  \"description\": \"circle_2d\",\n"
      "  \"data\": [\n"
      "    {\n"
      "      \"name\": \"data\",\n"
      "      \"url\": \"data/data.csv\"\n"
      "    }\n"
      "  ],\n"
      "  \"scales\": [\n"
      "    {\n"
      "      \"name\": \"x\",\n"
      "      \"type\": \"linear\",\n"
      "      \"domain\": {\"data\": \"data\", \"field\": \"c0\"}\n"
      "    },\n"
      "    {\n"
      "      \"name\": \"y\",\n"
      "      \"type\": \"linear\",\n"
      "      \"domain\": {\"data\": \"data\", \"field\": \"c1\"}\n"
      "    }\n"
      "  ],\n"
      "  \"marks\": [\n"
      "    {\n"
      "      \"encode\": {\n"
      "        \"enter\": {\n"
      "          \"map_zoom_level\": {\"value\": 10},\n"
      "          \"aggregation_type\": {\"value\": \"sum\"}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  auto wkb = arctern::render::WktToWkb(string_array);

  std::vector<std::shared_ptr<arrow::Array>> point_vec{wkb};
  std::vector<std::shared_ptr<arrow::Array>> color_vec{color_array};

  arctern::render::heat_map(point_vec, color_vec, vega);
}

TEST(HEATMAP_TEST, WKT_POINT_INT16_TEST) {
  // param1: wkt string
  std::string wkt1 = "POINT (50 50)";
  std::string wkt2 = "POINT (51 51)";
  std::string wkt3 = "POINT (52 52)";
  std::string wkt4 = "POINT (53 53)";
  std::string wkt5 = "POINT (54 54)";
  arrow::StringBuilder string_builder;
  auto status = string_builder.Append(wkt1);
  status = string_builder.Append(wkt2);
  status = string_builder.Append(wkt3);
  status = string_builder.Append(wkt4);
  status = string_builder.Append(wkt5);

  std::shared_ptr<arrow::StringArray> string_array;
  status = string_builder.Finish(&string_array);

  // param2: color
  std::shared_ptr<arrow::Array> color_array;
  arrow::Int16Builder color_builder;
  status = color_builder.Append(50);
  status = color_builder.Append(51);
  status = color_builder.Append(52);
  status = color_builder.Append(53);
  status = color_builder.Append(54);
  status = color_builder.Finish(&color_array);

  // param3: conf
  const std::string vega =
      "{\n"
      "  \"width\": 300,\n"
      "  \"height\": 200,\n"
      "  \"description\": \"circle_2d\",\n"
      "  \"data\": [\n"
      "    {\n"
      "      \"name\": \"data\",\n"
      "      \"url\": \"data/data.csv\"\n"
      "    }\n"
      "  ],\n"
      "  \"scales\": [\n"
      "    {\n"
      "      \"name\": \"x\",\n"
      "      \"type\": \"linear\",\n"
      "      \"domain\": {\"data\": \"data\", \"field\": \"c0\"}\n"
      "    },\n"
      "    {\n"
      "      \"name\": \"y\",\n"
      "      \"type\": \"linear\",\n"
      "      \"domain\": {\"data\": \"data\", \"field\": \"c1\"}\n"
      "    }\n"
      "  ],\n"
      "  \"marks\": [\n"
      "    {\n"
      "      \"encode\": {\n"
      "        \"enter\": {\n"
      "          \"map_zoom_level\": {\"value\": 10},\n"
      "          \"aggregation_type\": {\"value\": \"sum\"}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  auto wkb = arctern::render::WktToWkb(string_array);

  std::vector<std::shared_ptr<arrow::Array>> point_vec{wkb};
  std::vector<std::shared_ptr<arrow::Array>> color_vec{color_array};

  arctern::render::heat_map(point_vec, color_vec, vega);
}

TEST(HEATMAP_TEST, WKT_POINT_INT32_TEST) {
  // param1: wkt string
  std::string wkt1 = "POINT (50 50)";
  std::string wkt2 = "POINT (51 51)";
  std::string wkt3 = "POINT (52 52)";
  std::string wkt4 = "POINT (53 53)";
  std::string wkt5 = "POINT (54 54)";
  arrow::StringBuilder string_builder;
  auto status = string_builder.Append(wkt1);
  status = string_builder.Append(wkt2);
  status = string_builder.Append(wkt3);
  status = string_builder.Append(wkt4);
  status = string_builder.Append(wkt5);

  std::shared_ptr<arrow::StringArray> string_array;
  status = string_builder.Finish(&string_array);

  // param2: color
  std::shared_ptr<arrow::Array> color_array;
  arrow::Int32Builder color_builder;
  status = color_builder.Append(50);
  status = color_builder.Append(51);
  status = color_builder.Append(52);
  status = color_builder.Append(53);
  status = color_builder.Append(54);
  status = color_builder.Finish(&color_array);

  // param3: conf
  const std::string vega =
      "{\n"
      "  \"width\": 300,\n"
      "  \"height\": 200,\n"
      "  \"description\": \"circle_2d\",\n"
      "  \"data\": [\n"
      "    {\n"
      "      \"name\": \"data\",\n"
      "      \"url\": \"data/data.csv\"\n"
      "    }\n"
      "  ],\n"
      "  \"scales\": [\n"
      "    {\n"
      "      \"name\": \"x\",\n"
      "      \"type\": \"linear\",\n"
      "      \"domain\": {\"data\": \"data\", \"field\": \"c0\"}\n"
      "    },\n"
      "    {\n"
      "      \"name\": \"y\",\n"
      "      \"type\": \"linear\",\n"
      "      \"domain\": {\"data\": \"data\", \"field\": \"c1\"}\n"
      "    }\n"
      "  ],\n"
      "  \"marks\": [\n"
      "    {\n"
      "      \"encode\": {\n"
      "        \"enter\": {\n"
      "          \"map_zoom_level\": {\"value\": 10},\n"
      "          \"aggregation_type\": {\"value\": \"sum\"}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  auto wkb = arctern::render::WktToWkb(string_array);

  std::vector<std::shared_ptr<arrow::Array>> point_vec{wkb};
  std::vector<std::shared_ptr<arrow::Array>> color_vec{color_array};

  arctern::render::heat_map(point_vec, color_vec, vega);
}

TEST(HEATMAP_TEST, WKT_POINT_INT64_TEST) {
  // param1: wkt string
  std::string wkt1 = "POINT (50 50)";
  std::string wkt2 = "POINT (51 51)";
  std::string wkt3 = "POINT (52 52)";
  std::string wkt4 = "POINT (53 53)";
  std::string wkt5 = "POINT (54 54)";
  arrow::StringBuilder string_builder;
  auto status = string_builder.Append(wkt1);
  status = string_builder.Append(wkt2);
  status = string_builder.Append(wkt3);
  status = string_builder.Append(wkt4);
  status = string_builder.Append(wkt5);

  std::shared_ptr<arrow::StringArray> string_array;
  status = string_builder.Finish(&string_array);

  // param2: color
  std::shared_ptr<arrow::Array> color_array;
  arrow::Int64Builder color_builder;
  status = color_builder.Append(50);
  status = color_builder.Append(51);
  status = color_builder.Append(52);
  status = color_builder.Append(53);
  status = color_builder.Append(54);
  status = color_builder.Finish(&color_array);

  // param3: conf
  const std::string vega =
      "{\n"
      "  \"width\": 300,\n"
      "  \"height\": 200,\n"
      "  \"description\": \"circle_2d\",\n"
      "  \"data\": [\n"
      "    {\n"
      "      \"name\": \"data\",\n"
      "      \"url\": \"data/data.csv\"\n"
      "    }\n"
      "  ],\n"
      "  \"scales\": [\n"
      "    {\n"
      "      \"name\": \"x\",\n"
      "      \"type\": \"linear\",\n"
      "      \"domain\": {\"data\": \"data\", \"field\": \"c0\"}\n"
      "    },\n"
      "    {\n"
      "      \"name\": \"y\",\n"
      "      \"type\": \"linear\",\n"
      "      \"domain\": {\"data\": \"data\", \"field\": \"c1\"}\n"
      "    }\n"
      "  ],\n"
      "  \"marks\": [\n"
      "    {\n"
      "      \"encode\": {\n"
      "        \"enter\": {\n"
      "          \"map_zoom_level\": {\"value\": 10},\n"
      "          \"aggregation_type\": {\"value\": \"sum\"}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  auto wkb = arctern::render::WktToWkb(string_array);

  std::vector<std::shared_ptr<arrow::Array>> point_vec{wkb};
  std::vector<std::shared_ptr<arrow::Array>> color_vec{color_array};

  arctern::render::heat_map(point_vec, color_vec, vega);
}

TEST(HEATMAP_TEST, WKT_POINT_UINT8_TEST) {
  // param1: wkt string
  std::string wkt1 = "POINT (50 50)";
  std::string wkt2 = "POINT (51 51)";
  std::string wkt3 = "POINT (52 52)";
  std::string wkt4 = "POINT (53 53)";
  std::string wkt5 = "POINT (54 54)";
  arrow::StringBuilder string_builder;
  auto status = string_builder.Append(wkt1);
  status = string_builder.Append(wkt2);
  status = string_builder.Append(wkt3);
  status = string_builder.Append(wkt4);
  status = string_builder.Append(wkt5);

  std::shared_ptr<arrow::StringArray> string_array;
  status = string_builder.Finish(&string_array);

  // param2: color
  std::shared_ptr<arrow::Array> color_array;
  arrow::UInt8Builder color_builder;
  status = color_builder.Append(50);
  status = color_builder.Append(51);
  status = color_builder.Append(52);
  status = color_builder.Append(53);
  status = color_builder.Append(54);
  status = color_builder.Finish(&color_array);

  // param3: conf
  const std::string vega =
      "{\n"
      "  \"width\": 300,\n"
      "  \"height\": 200,\n"
      "  \"description\": \"circle_2d\",\n"
      "  \"data\": [\n"
      "    {\n"
      "      \"name\": \"data\",\n"
      "      \"url\": \"data/data.csv\"\n"
      "    }\n"
      "  ],\n"
      "  \"scales\": [\n"
      "    {\n"
      "      \"name\": \"x\",\n"
      "      \"type\": \"linear\",\n"
      "      \"domain\": {\"data\": \"data\", \"field\": \"c0\"}\n"
      "    },\n"
      "    {\n"
      "      \"name\": \"y\",\n"
      "      \"type\": \"linear\",\n"
      "      \"domain\": {\"data\": \"data\", \"field\": \"c1\"}\n"
      "    }\n"
      "  ],\n"
      "  \"marks\": [\n"
      "    {\n"
      "      \"encode\": {\n"
      "        \"enter\": {\n"
      "          \"map_zoom_level\": {\"value\": 10},\n"
      "          \"aggregation_type\": {\"value\": \"sum\"}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  auto wkb = arctern::render::WktToWkb(string_array);

  std::vector<std::shared_ptr<arrow::Array>> point_vec{wkb};
  std::vector<std::shared_ptr<arrow::Array>> color_vec{color_array};

  arctern::render::heat_map(point_vec, color_vec, vega);
}

TEST(HEATMAP_TEST, WKT_POINT_UINT16_TEST) {
  // param1: wkt string
  std::string wkt1 = "POINT (50 50)";
  std::string wkt2 = "POINT (51 51)";
  std::string wkt3 = "POINT (52 52)";
  std::string wkt4 = "POINT (53 53)";
  std::string wkt5 = "POINT (54 54)";
  arrow::StringBuilder string_builder;
  auto status = string_builder.Append(wkt1);
  status = string_builder.Append(wkt2);
  status = string_builder.Append(wkt3);
  status = string_builder.Append(wkt4);
  status = string_builder.Append(wkt5);

  std::shared_ptr<arrow::StringArray> string_array;
  status = string_builder.Finish(&string_array);

  // param2: color
  std::shared_ptr<arrow::Array> color_array;
  arrow::UInt16Builder color_builder;
  status = color_builder.Append(50);
  status = color_builder.Append(51);
  status = color_builder.Append(52);
  status = color_builder.Append(53);
  status = color_builder.Append(54);
  status = color_builder.Finish(&color_array);

  // param3: conf
  const std::string vega =
      "{\n"
      "  \"width\": 300,\n"
      "  \"height\": 200,\n"
      "  \"description\": \"circle_2d\",\n"
      "  \"data\": [\n"
      "    {\n"
      "      \"name\": \"data\",\n"
      "      \"url\": \"data/data.csv\"\n"
      "    }\n"
      "  ],\n"
      "  \"scales\": [\n"
      "    {\n"
      "      \"name\": \"x\",\n"
      "      \"type\": \"linear\",\n"
      "      \"domain\": {\"data\": \"data\", \"field\": \"c0\"}\n"
      "    },\n"
      "    {\n"
      "      \"name\": \"y\",\n"
      "      \"type\": \"linear\",\n"
      "      \"domain\": {\"data\": \"data\", \"field\": \"c1\"}\n"
      "    }\n"
      "  ],\n"
      "  \"marks\": [\n"
      "    {\n"
      "      \"encode\": {\n"
      "        \"enter\": {\n"
      "          \"map_zoom_level\": {\"value\": 10},\n"
      "          \"aggregation_type\": {\"value\": \"sum\"}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  auto wkb = arctern::render::WktToWkb(string_array);

  std::vector<std::shared_ptr<arrow::Array>> point_vec{wkb};
  std::vector<std::shared_ptr<arrow::Array>> color_vec{color_array};

  arctern::render::heat_map(point_vec, color_vec, vega);
}

TEST(HEATMAP_TEST, WKT_POINT_UINT32_TEST) {
  // param1: wkt string
  std::string wkt1 = "POINT (50 50)";
  std::string wkt2 = "POINT (51 51)";
  std::string wkt3 = "POINT (52 52)";
  std::string wkt4 = "POINT (53 53)";
  std::string wkt5 = "POINT (54 54)";
  arrow::StringBuilder string_builder;
  auto status = string_builder.Append(wkt1);
  status = string_builder.Append(wkt2);
  status = string_builder.Append(wkt3);
  status = string_builder.Append(wkt4);
  status = string_builder.Append(wkt5);

  std::shared_ptr<arrow::StringArray> string_array;
  status = string_builder.Finish(&string_array);

  // param2: color
  std::shared_ptr<arrow::Array> color_array;
  arrow::UInt32Builder color_builder;
  status = color_builder.Append(50);
  status = color_builder.Append(51);
  status = color_builder.Append(52);
  status = color_builder.Append(53);
  status = color_builder.Append(54);
  status = color_builder.Finish(&color_array);

  // param3: conf
  const std::string vega =
      "{\n"
      "  \"width\": 300,\n"
      "  \"height\": 200,\n"
      "  \"description\": \"circle_2d\",\n"
      "  \"data\": [\n"
      "    {\n"
      "      \"name\": \"data\",\n"
      "      \"url\": \"data/data.csv\"\n"
      "    }\n"
      "  ],\n"
      "  \"scales\": [\n"
      "    {\n"
      "      \"name\": \"x\",\n"
      "      \"type\": \"linear\",\n"
      "      \"domain\": {\"data\": \"data\", \"field\": \"c0\"}\n"
      "    },\n"
      "    {\n"
      "      \"name\": \"y\",\n"
      "      \"type\": \"linear\",\n"
      "      \"domain\": {\"data\": \"data\", \"field\": \"c1\"}\n"
      "    }\n"
      "  ],\n"
      "  \"marks\": [\n"
      "    {\n"
      "      \"encode\": {\n"
      "        \"enter\": {\n"
      "          \"map_zoom_level\": {\"value\": 10},\n"
      "          \"aggregation_type\": {\"value\": \"sum\"}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  auto wkb = arctern::render::WktToWkb(string_array);

  std::vector<std::shared_ptr<arrow::Array>> point_vec{wkb};
  std::vector<std::shared_ptr<arrow::Array>> color_vec{color_array};

  arctern::render::heat_map(point_vec, color_vec, vega);
}

TEST(HEATMAP_TEST, WKT_POINT_UINT64_TEST) {
  // param1: wkt string
  std::string wkt1 = "POINT (50 50)";
  std::string wkt2 = "POINT (51 51)";
  std::string wkt3 = "POINT (52 52)";
  std::string wkt4 = "POINT (53 53)";
  std::string wkt5 = "POINT (54 54)";
  arrow::StringBuilder string_builder;
  auto status = string_builder.Append(wkt1);
  status = string_builder.Append(wkt2);
  status = string_builder.Append(wkt3);
  status = string_builder.Append(wkt4);
  status = string_builder.Append(wkt5);

  std::shared_ptr<arrow::StringArray> string_array;
  status = string_builder.Finish(&string_array);

  // param2: color
  std::shared_ptr<arrow::Array> color_array;
  arrow::UInt64Builder color_builder;
  status = color_builder.Append(50);
  status = color_builder.Append(51);
  status = color_builder.Append(52);
  status = color_builder.Append(53);
  status = color_builder.Append(54);
  status = color_builder.Finish(&color_array);

  // param3: conf
  const std::string vega =
      "{\n"
      "  \"width\": 300,\n"
      "  \"height\": 200,\n"
      "  \"description\": \"circle_2d\",\n"
      "  \"data\": [\n"
      "    {\n"
      "      \"name\": \"data\",\n"
      "      \"url\": \"data/data.csv\"\n"
      "    }\n"
      "  ],\n"
      "  \"scales\": [\n"
      "    {\n"
      "      \"name\": \"x\",\n"
      "      \"type\": \"linear\",\n"
      "      \"domain\": {\"data\": \"data\", \"field\": \"c0\"}\n"
      "    },\n"
      "    {\n"
      "      \"name\": \"y\",\n"
      "      \"type\": \"linear\",\n"
      "      \"domain\": {\"data\": \"data\", \"field\": \"c1\"}\n"
      "    }\n"
      "  ],\n"
      "  \"marks\": [\n"
      "    {\n"
      "      \"encode\": {\n"
      "        \"enter\": {\n"
      "          \"map_zoom_level\": {\"value\": 10},\n"
      "          \"aggregation_type\": {\"value\": \"sum\"}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  auto wkb = arctern::render::WktToWkb(string_array);

  std::vector<std::shared_ptr<arrow::Array>> point_vec{wkb};
  std::vector<std::shared_ptr<arrow::Array>> color_vec{color_array};

  arctern::render::heat_map(point_vec, color_vec, vega);
}

TEST(HEATMAP_TEST, WKT_POINT_FLOAT_TEST) {
  // param1: wkt string
  std::string wkt1 = "POINT (50 50)";
  std::string wkt2 = "POINT (51 51)";
  std::string wkt3 = "POINT (52 52)";
  std::string wkt4 = "POINT (53 53)";
  std::string wkt5 = "POINT (54 54)";
  arrow::StringBuilder string_builder;
  auto status = string_builder.Append(wkt1);
  status = string_builder.Append(wkt2);
  status = string_builder.Append(wkt3);
  status = string_builder.Append(wkt4);
  status = string_builder.Append(wkt5);

  std::shared_ptr<arrow::StringArray> string_array;
  status = string_builder.Finish(&string_array);

  // param2: color
  std::shared_ptr<arrow::Array> color_array;
  arrow::FloatBuilder color_builder;
  status = color_builder.Append(50);
  status = color_builder.Append(51);
  status = color_builder.Append(52);
  status = color_builder.Append(53);
  status = color_builder.Append(54);
  status = color_builder.Finish(&color_array);

  // param3: conf
  const std::string vega =
      "{\n"
      "  \"width\": 300,\n"
      "  \"height\": 200,\n"
      "  \"description\": \"circle_2d\",\n"
      "  \"data\": [\n"
      "    {\n"
      "      \"name\": \"data\",\n"
      "      \"url\": \"data/data.csv\"\n"
      "    }\n"
      "  ],\n"
      "  \"scales\": [\n"
      "    {\n"
      "      \"name\": \"x\",\n"
      "      \"type\": \"linear\",\n"
      "      \"domain\": {\"data\": \"data\", \"field\": \"c0\"}\n"
      "    },\n"
      "    {\n"
      "      \"name\": \"y\",\n"
      "      \"type\": \"linear\",\n"
      "      \"domain\": {\"data\": \"data\", \"field\": \"c1\"}\n"
      "    }\n"
      "  ],\n"
      "  \"marks\": [\n"
      "    {\n"
      "      \"encode\": {\n"
      "        \"enter\": {\n"
      "          \"map_zoom_level\": {\"value\": 10},\n"
      "          \"aggregation_type\": {\"value\": \"sum\"}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  auto wkb = arctern::render::WktToWkb(string_array);

  std::vector<std::shared_ptr<arrow::Array>> point_vec{wkb};
  std::vector<std::shared_ptr<arrow::Array>> color_vec{color_array};

  arctern::render::heat_map(point_vec, color_vec, vega);
}

TEST(HEATMAP_TEST, WKT_POINT_DOUBLE_TEST) {
  // param1: wkt string
  std::string wkt1 = "POINT (50 50)";
  std::string wkt2 = "POINT (51 51)";
  std::string wkt3 = "POINT (52 52)";
  std::string wkt4 = "POINT (53 53)";
  std::string wkt5 = "POINT (54 54)";
  arrow::StringBuilder string_builder;
  auto status = string_builder.Append(wkt1);
  status = string_builder.Append(wkt2);
  status = string_builder.Append(wkt3);
  status = string_builder.Append(wkt4);
  status = string_builder.Append(wkt5);

  std::shared_ptr<arrow::StringArray> string_array;
  status = string_builder.Finish(&string_array);

  // param2: color
  std::shared_ptr<arrow::Array> color_array;
  arrow::DoubleBuilder color_builder;
  status = color_builder.Append(50);
  status = color_builder.Append(51);
  status = color_builder.Append(52);
  status = color_builder.Append(53);
  status = color_builder.Append(54);
  status = color_builder.Finish(&color_array);

  // param3: conf
  const std::string vega =
      "{\n"
      "  \"width\": 300,\n"
      "  \"height\": 200,\n"
      "  \"description\": \"circle_2d\",\n"
      "  \"data\": [\n"
      "    {\n"
      "      \"name\": \"data\",\n"
      "      \"url\": \"data/data.csv\"\n"
      "    }\n"
      "  ],\n"
      "  \"scales\": [\n"
      "    {\n"
      "      \"name\": \"x\",\n"
      "      \"type\": \"linear\",\n"
      "      \"domain\": {\"data\": \"data\", \"field\": \"c0\"}\n"
      "    },\n"
      "    {\n"
      "      \"name\": \"y\",\n"
      "      \"type\": \"linear\",\n"
      "      \"domain\": {\"data\": \"data\", \"field\": \"c1\"}\n"
      "    }\n"
      "  ],\n"
      "  \"marks\": [\n"
      "    {\n"
      "      \"encode\": {\n"
      "        \"enter\": {\n"
      "          \"map_zoom_level\": {\"value\": 10},\n"
      "          \"aggregation_type\": {\"value\": \"sum\"}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  auto wkb = arctern::render::WktToWkb(string_array);

  std::vector<std::shared_ptr<arrow::Array>> point_vec{wkb};
  std::vector<std::shared_ptr<arrow::Array>> color_vec{color_array};

  arctern::render::heat_map(point_vec, color_vec, vega);
}

// TEST(HEATMAP_TEST, WKT_POINT_INVALID_DATA_TYPE_TEST) {
//  // param1: wkt string
//  std::string wkt1 = "POINT (50 50)";
//  std::string wkt2 = "POINT (51 51)";
//  std::string wkt3 = "POINT (52 52)";
//  std::string wkt4 = "POINT (53 53)";
//  std::string wkt5 = "POINT (54 54)";
//  arrow::StringBuilder string_builder;
//  auto status = string_builder.Append(wkt1);
//  status = string_builder.Append(wkt2);
//  status = string_builder.Append(wkt3);
//  status = string_builder.Append(wkt4);
//  status = string_builder.Append(wkt5);
//
//  std::shared_ptr<arrow::StringArray> string_array;
//  status = string_builder.Finish(&string_array);
//
//  // param2: color
//  std::shared_ptr<arrow::Array> color_array;
//  arrow::StringBuilder color_builder;
//  status = color_builder.Append("");
//  status = color_builder.Append("");
//  status = color_builder.Append("");
//  status = color_builder.Append("");
//  status = color_builder.Append("");
//  status = color_builder.Finish(&color_array);
//
//  // param3: conf
//  const std::string vega =
//      "{\n"
//      "  \"width\": 300,\n"
//      "  \"height\": 200,\n"
//      "  \"description\": \"circle_2d\",\n"
//      "  \"data\": [\n"
//      "    {\n"
//      "      \"name\": \"data\",\n"
//      "      \"url\": \"data/data.csv\"\n"
//      "    }\n"
//      "  ],\n"
//      "  \"scales\": [\n"
//      "    {\n"
//      "      \"name\": \"x\",\n"
//      "      \"type\": \"linear\",\n"
//      "      \"domain\": {\"data\": \"data\", \"field\": \"c0\"}\n"
//      "    },\n"
//      "    {\n"
//      "      \"name\": \"y\",\n"
//      "      \"type\": \"linear\",\n"
//      "      \"domain\": {\"data\": \"data\", \"field\": \"c1\"}\n"
//      "    }\n"
//      "  ],\n"
//      "  \"marks\": [\n"
//      "    {\n"
//      "      \"encode\": {\n"
//      "        \"enter\": {\n"
//      "          \"map_zoom_level\": {\"value\": 10}\n"
//      "        }\n"
//      "      }\n"
//      "    }\n"
//      "  ]\n"
//      "}";
//
//  auto wkb = arctern::render::WktToWkb(string_array);
//  arctern::render::heat_map(wkb, color_array, vega);
//}

// TEST(HEATMAP_TEST, INVALID_JSON_TEST) {
//  // param1: wkt string
//  std::string wkt1 = "POINT (50 50)";
//  std::string wkt2 = "POINT (51 51)";
//  std::string wkt3 = "POINT (52 52)";
//  std::string wkt4 = "POINT (53 53)";
//  std::string wkt5 = "POINT (54 54)";
//  arrow::StringBuilder string_builder;
//  auto status = string_builder.Append(wkt1);
//  status = string_builder.Append(wkt2);
//  status = string_builder.Append(wkt3);
//  status = string_builder.Append(wkt4);
//  status = string_builder.Append(wkt5);
//
//  std::shared_ptr<arrow::StringArray> string_array;
//  status = string_builder.Finish(&string_array);
//
//  // param2: color
//  std::shared_ptr<arrow::Array> color_array;
//  arrow::DoubleBuilder color_builder;
//  status = color_builder.Append(50);
//  status = color_builder.Append(51);
//  status = color_builder.Append(52);
//  status = color_builder.Append(53);
//  status = color_builder.Append(54);
//  status = color_builder.Finish(&color_array);
//
//  // param3: conf
//  const std::string vega =
//      "{\n"
//      "  \"width\": 300,\n"
//      "  \"height\": 200,\n"
//      "  \"description\": \"circle_2d\",\n"
//      "  \"data\": [\n"
//      "    {\n"
//      "      \"name\": \"data\",\n"
//      "      \"url\": \"data/data.csv\"\n"
//      "    }\n"
//      "  ],\n"
//      "  \"scales\": [\n"
//      "    {\n"
//      "      \"name\": \"x\",\n"
//      "      \"type\": \"linear\",\n"
//      "      \"domain\": {\"data\": \"data\", \"field\": \"c0\"}\n"
//      "    },\n"
//      "    {\n"
//      "      \"name\": \"y\",\n"
//      "      \"type\": \"linear\",\n"
//      "      \"domain\": {\"data\": \"data\", \"field\": \"c1\"}\n"
//      "    }\n"
//      "  ],\n"
//      "  \"marks\": [\n"
//      "    {\n"
//      "      \"encode\": {\n"
//      "        \"enter\": {\n"
//      "          \"map_zoom_level\": {\"value\": \"INVALID_NUMBER\"}\n"
//      "        }\n"
//      "      }\n"
//      "    }\n"
//      "  ]\n"
//      "}";
//
//  auto wkb = arctern::render::WktToWkb(string_array);
//  auto wkt = arctern::render::WkbToWkt(wkb);
//  wkb = arctern::render::WktToWkb(wkt);
//  arctern::render::heat_map(wkb, color_array, vega);
//}

TEST(HEATMAP_TEST, MEAN) {
  // param1: wkt string
  std::string wkt1 = "POINT (50 50)";
  std::string wkt2 = "POINT (51 51)";
  std::string wkt3 = "POINT (52 52)";
  std::string wkt4 = "POINT (53 53)";
  std::string wkt5 = "POINT (54 54)";
  arrow::StringBuilder string_builder;
  auto status = string_builder.Append(wkt1);
  status = string_builder.Append(wkt2);
  status = string_builder.Append(wkt3);
  status = string_builder.Append(wkt4);
  status = string_builder.Append(wkt5);

  std::shared_ptr<arrow::StringArray> string_array;
  status = string_builder.Finish(&string_array);

  // param2: color
  std::shared_ptr<arrow::Array> color_array;
  arrow::DoubleBuilder color_builder;
  status = color_builder.Append(50);
  status = color_builder.Append(51);
  status = color_builder.Append(52);
  status = color_builder.Append(53);
  status = color_builder.Append(54);
  status = color_builder.Finish(&color_array);

  // param3: conf
  const std::string vega =
      "{\n"
      "  \"width\": 300,\n"
      "  \"height\": 200,\n"
      "  \"description\": \"circle_2d\",\n"
      "  \"data\": [\n"
      "    {\n"
      "      \"name\": \"data\",\n"
      "      \"url\": \"data/data.csv\"\n"
      "    }\n"
      "  ],\n"
      "  \"scales\": [\n"
      "    {\n"
      "      \"name\": \"x\",\n"
      "      \"type\": \"linear\",\n"
      "      \"domain\": {\"data\": \"data\", \"field\": \"c0\"}\n"
      "    },\n"
      "    {\n"
      "      \"name\": \"y\",\n"
      "      \"type\": \"linear\",\n"
      "      \"domain\": {\"data\": \"data\", \"field\": \"c1\"}\n"
      "    }\n"
      "  ],\n"
      "  \"marks\": [\n"
      "    {\n"
      "      \"encode\": {\n"
      "        \"enter\": {\n"
      "          \"map_zoom_level\": {\"value\": 10},\n"
      "          \"aggregation_type\": {\"value\": \"mean\"}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  auto wkb = arctern::render::WktToWkb(string_array);

  std::vector<std::shared_ptr<arrow::Array>> point_vec{wkb};
  std::vector<std::shared_ptr<arrow::Array>> color_vec{color_array};

  arctern::render::heat_map(point_vec, color_vec, vega);
}

TEST(HEATMAP_TEST, SUM) {
  // param1: wkt string
  std::string wkt1 = "POINT (50 50)";
  std::string wkt2 = "POINT (51 51)";
  std::string wkt3 = "POINT (52 52)";
  std::string wkt4 = "POINT (53 53)";
  std::string wkt5 = "POINT (54 54)";
  arrow::StringBuilder string_builder;
  auto status = string_builder.Append(wkt1);
  status = string_builder.Append(wkt2);
  status = string_builder.Append(wkt3);
  status = string_builder.Append(wkt4);
  status = string_builder.Append(wkt5);

  std::shared_ptr<arrow::StringArray> string_array;
  status = string_builder.Finish(&string_array);

  // param2: color
  std::shared_ptr<arrow::Array> color_array;
  arrow::DoubleBuilder color_builder;
  status = color_builder.Append(50);
  status = color_builder.Append(51);
  status = color_builder.Append(52);
  status = color_builder.Append(53);
  status = color_builder.Append(54);
  status = color_builder.Finish(&color_array);

  // param3: conf
  const std::string vega =
      "{\n"
      "  \"width\": 300,\n"
      "  \"height\": 200,\n"
      "  \"description\": \"circle_2d\",\n"
      "  \"data\": [\n"
      "    {\n"
      "      \"name\": \"data\",\n"
      "      \"url\": \"data/data.csv\"\n"
      "    }\n"
      "  ],\n"
      "  \"scales\": [\n"
      "    {\n"
      "      \"name\": \"x\",\n"
      "      \"type\": \"linear\",\n"
      "      \"domain\": {\"data\": \"data\", \"field\": \"c0\"}\n"
      "    },\n"
      "    {\n"
      "      \"name\": \"y\",\n"
      "      \"type\": \"linear\",\n"
      "      \"domain\": {\"data\": \"data\", \"field\": \"c1\"}\n"
      "    }\n"
      "  ],\n"
      "  \"marks\": [\n"
      "    {\n"
      "      \"encode\": {\n"
      "        \"enter\": {\n"
      "          \"map_zoom_level\": {\"value\": 10},\n"
      "          \"aggregation_type\": {\"value\": \"sum\"}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  auto wkb = arctern::render::WktToWkb(string_array);

  std::vector<std::shared_ptr<arrow::Array>> point_vec{wkb};
  std::vector<std::shared_ptr<arrow::Array>> color_vec{color_array};

  arctern::render::heat_map(point_vec, color_vec, vega);
}

TEST(HEATMAP_TEST, MAX) {
  // param1: wkt string
  std::string wkt1 = "POINT (50 50)";
  std::string wkt2 = "POINT (51 51)";
  std::string wkt3 = "POINT (52 52)";
  std::string wkt4 = "POINT (53 53)";
  std::string wkt5 = "POINT (54 54)";
  arrow::StringBuilder string_builder;
  auto status = string_builder.Append(wkt1);
  status = string_builder.Append(wkt2);
  status = string_builder.Append(wkt3);
  status = string_builder.Append(wkt4);
  status = string_builder.Append(wkt5);

  std::shared_ptr<arrow::StringArray> string_array;
  status = string_builder.Finish(&string_array);

  // param2: color
  std::shared_ptr<arrow::Array> color_array;
  arrow::DoubleBuilder color_builder;
  status = color_builder.Append(50);
  status = color_builder.Append(51);
  status = color_builder.Append(52);
  status = color_builder.Append(53);
  status = color_builder.Append(54);
  status = color_builder.Finish(&color_array);

  // param3: conf
  const std::string vega =
      "{\n"
      "  \"width\": 300,\n"
      "  \"height\": 200,\n"
      "  \"description\": \"circle_2d\",\n"
      "  \"data\": [\n"
      "    {\n"
      "      \"name\": \"data\",\n"
      "      \"url\": \"data/data.csv\"\n"
      "    }\n"
      "  ],\n"
      "  \"scales\": [\n"
      "    {\n"
      "      \"name\": \"x\",\n"
      "      \"type\": \"linear\",\n"
      "      \"domain\": {\"data\": \"data\", \"field\": \"c0\"}\n"
      "    },\n"
      "    {\n"
      "      \"name\": \"y\",\n"
      "      \"type\": \"linear\",\n"
      "      \"domain\": {\"data\": \"data\", \"field\": \"c1\"}\n"
      "    }\n"
      "  ],\n"
      "  \"marks\": [\n"
      "    {\n"
      "      \"encode\": {\n"
      "        \"enter\": {\n"
      "          \"map_zoom_level\": {\"value\": 10},\n"
      "          \"aggregation_type\": {\"value\": \"max\"}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  auto wkb = arctern::render::WktToWkb(string_array);

  std::vector<std::shared_ptr<arrow::Array>> point_vec{wkb};
  std::vector<std::shared_ptr<arrow::Array>> color_vec{color_array};

  arctern::render::heat_map(point_vec, color_vec, vega);
}

TEST(HEATMAP_TEST, MIN) {
  // param1: wkt string
  std::string wkt1 = "POINT (50 50)";
  std::string wkt2 = "POINT (51 51)";
  std::string wkt3 = "POINT (52 52)";
  std::string wkt4 = "POINT (53 53)";
  std::string wkt5 = "POINT (54 54)";
  arrow::StringBuilder string_builder;
  auto status = string_builder.Append(wkt1);
  status = string_builder.Append(wkt2);
  status = string_builder.Append(wkt3);
  status = string_builder.Append(wkt4);
  status = string_builder.Append(wkt5);

  std::shared_ptr<arrow::StringArray> string_array;
  status = string_builder.Finish(&string_array);

  // param2: color
  std::shared_ptr<arrow::Array> color_array;
  arrow::DoubleBuilder color_builder;
  status = color_builder.Append(50);
  status = color_builder.Append(51);
  status = color_builder.Append(52);
  status = color_builder.Append(53);
  status = color_builder.Append(54);
  status = color_builder.Finish(&color_array);

  // param3: conf
  const std::string vega =
      "{\n"
      "  \"width\": 300,\n"
      "  \"height\": 200,\n"
      "  \"description\": \"circle_2d\",\n"
      "  \"data\": [\n"
      "    {\n"
      "      \"name\": \"data\",\n"
      "      \"url\": \"data/data.csv\"\n"
      "    }\n"
      "  ],\n"
      "  \"scales\": [\n"
      "    {\n"
      "      \"name\": \"x\",\n"
      "      \"type\": \"linear\",\n"
      "      \"domain\": {\"data\": \"data\", \"field\": \"c0\"}\n"
      "    },\n"
      "    {\n"
      "      \"name\": \"y\",\n"
      "      \"type\": \"linear\",\n"
      "      \"domain\": {\"data\": \"data\", \"field\": \"c1\"}\n"
      "    }\n"
      "  ],\n"
      "  \"marks\": [\n"
      "    {\n"
      "      \"encode\": {\n"
      "        \"enter\": {\n"
      "          \"map_zoom_level\": {\"value\": 10},\n"
      "          \"aggregation_type\": {\"value\": \"min\"}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  auto wkb = arctern::render::WktToWkb(string_array);

  std::vector<std::shared_ptr<arrow::Array>> point_vec{wkb};
  std::vector<std::shared_ptr<arrow::Array>> color_vec{color_array};

  arctern::render::heat_map(point_vec, color_vec, vega);
}

TEST(HEATMAP_TEST, COUNT) {
  // param1: wkt string
  std::string wkt1 = "POINT (50 50)";
  std::string wkt2 = "POINT (51 51)";
  std::string wkt3 = "POINT (52 52)";
  std::string wkt4 = "POINT (53 53)";
  std::string wkt5 = "POINT (54 54)";
  arrow::StringBuilder string_builder;
  auto status = string_builder.Append(wkt1);
  status = string_builder.Append(wkt2);
  status = string_builder.Append(wkt3);
  status = string_builder.Append(wkt4);
  status = string_builder.Append(wkt5);

  std::shared_ptr<arrow::StringArray> string_array;
  status = string_builder.Finish(&string_array);

  // param2: color
  std::shared_ptr<arrow::Array> color_array;
  arrow::DoubleBuilder color_builder;
  status = color_builder.Append(50);
  status = color_builder.Append(51);
  status = color_builder.Append(52);
  status = color_builder.Append(53);
  status = color_builder.Append(54);
  status = color_builder.Finish(&color_array);

  // param3: conf
  const std::string vega =
      "{\n"
      "  \"width\": 300,\n"
      "  \"height\": 200,\n"
      "  \"description\": \"circle_2d\",\n"
      "  \"data\": [\n"
      "    {\n"
      "      \"name\": \"data\",\n"
      "      \"url\": \"data/data.csv\"\n"
      "    }\n"
      "  ],\n"
      "  \"scales\": [\n"
      "    {\n"
      "      \"name\": \"x\",\n"
      "      \"type\": \"linear\",\n"
      "      \"domain\": {\"data\": \"data\", \"field\": \"c0\"}\n"
      "    },\n"
      "    {\n"
      "      \"name\": \"y\",\n"
      "      \"type\": \"linear\",\n"
      "      \"domain\": {\"data\": \"data\", \"field\": \"c1\"}\n"
      "    }\n"
      "  ],\n"
      "  \"marks\": [\n"
      "    {\n"
      "      \"encode\": {\n"
      "        \"enter\": {\n"
      "          \"map_zoom_level\": {\"value\": 10},\n"
      "          \"aggregation_type\": {\"value\": \"count\"}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  auto wkb = arctern::render::WktToWkb(string_array);

  std::vector<std::shared_ptr<arrow::Array>> point_vec{wkb};
  std::vector<std::shared_ptr<arrow::Array>> color_vec{color_array};

  arctern::render::heat_map(point_vec, color_vec, vega);
}

TEST(HEATMAP_TEST, STD) {
  // param1: wkt string
  std::string wkt1 = "POINT (50 50)";
  std::string wkt2 = "POINT (51 51)";
  std::string wkt3 = "POINT (52 52)";
  std::string wkt4 = "POINT (53 53)";
  std::string wkt5 = "POINT (54 54)";
  arrow::StringBuilder string_builder;
  auto status = string_builder.Append(wkt1);
  status = string_builder.Append(wkt2);
  status = string_builder.Append(wkt3);
  status = string_builder.Append(wkt4);
  status = string_builder.Append(wkt5);

  std::shared_ptr<arrow::StringArray> string_array;
  status = string_builder.Finish(&string_array);

  // param2: color
  std::shared_ptr<arrow::Array> color_array;
  arrow::DoubleBuilder color_builder;
  status = color_builder.Append(50);
  status = color_builder.Append(51);
  status = color_builder.Append(52);
  status = color_builder.Append(53);
  status = color_builder.Append(54);
  status = color_builder.Finish(&color_array);

  // param3: conf
  const std::string vega =
      "{\n"
      "  \"width\": 300,\n"
      "  \"height\": 200,\n"
      "  \"description\": \"circle_2d\",\n"
      "  \"data\": [\n"
      "    {\n"
      "      \"name\": \"data\",\n"
      "      \"url\": \"data/data.csv\"\n"
      "    }\n"
      "  ],\n"
      "  \"scales\": [\n"
      "    {\n"
      "      \"name\": \"x\",\n"
      "      \"type\": \"linear\",\n"
      "      \"domain\": {\"data\": \"data\", \"field\": \"c0\"}\n"
      "    },\n"
      "    {\n"
      "      \"name\": \"y\",\n"
      "      \"type\": \"linear\",\n"
      "      \"domain\": {\"data\": \"data\", \"field\": \"c1\"}\n"
      "    }\n"
      "  ],\n"
      "  \"marks\": [\n"
      "    {\n"
      "      \"encode\": {\n"
      "        \"enter\": {\n"
      "          \"map_zoom_level\": {\"value\": 10},\n"
      "          \"aggregation_type\": {\"value\": \"std\"}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  auto wkb = arctern::render::WktToWkb(string_array);

  std::vector<std::shared_ptr<arrow::Array>> point_vec{wkb};
  std::vector<std::shared_ptr<arrow::Array>> color_vec{color_array};

  arctern::render::heat_map(point_vec, color_vec, vega);
}
