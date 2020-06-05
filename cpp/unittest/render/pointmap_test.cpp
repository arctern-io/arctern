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

// TEST(POINTMAP_TEST, RAW_POINT_TEST) {
//  // param1: x, y
//  std::vector<uint32_t> x{50, 51, 52, 53, 54};
//  std::vector<uint32_t> y{50, 51, 52, 53, 54};
//
//  // param2: conf
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
//      "          \"shape\": {\"value\": \"circle\"},\n"
//      "          \"point_color\": {\"value\": \"#ff0000\"},\n"
//      "          \"point_size\": {\"value\": 30},\n"
//      "          \"opacity\": {\"value\": 0.5}\n"
//      "        }\n"
//      "      }\n"
//      "    }\n"
//      "  ]\n"
//      "}";
//
//  arctern::render::pointmap(x.data(), y.data(), 5, vega);
//}
//
// TEST(POINTMAP_TEST, INVALID_COLOR_TEST) {
//  // param1: x, y
//  std::vector<uint32_t> x{50, 51, 52, 53, 54};
//  std::vector<uint32_t> y{50, 51, 52, 53, 54};
//
//  // param2: conf
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
//      "          \"shape\": {\"value\": \"circle\"},\n"
//      "          \"point_color\": {\"value\": \"#xxxxxx\"},\n"
//      "          \"point_size\": {\"value\": 30},\n"
//      "          \"opacity\": {\"value\": 0.5}\n"
//      "        }\n"
//      "      }\n"
//      "    }\n"
//      "  ]\n"
//      "}";
//
//  arctern::render::pointmap(x.data(), y.data(), 5, vega);
//}

// TEST(POINTMAP_TEST, RAW_POINT_INVALID_JSON_TEST) {
//  auto bit_map = new uint8_t{0xff};
//
//  auto data_type = arrow::uint32();
//
//  auto buff_data1 = (uint32_t*)malloc(5 * sizeof(uint32_t));
//  for (int i = 0; i < 5; ++i) {
//    buff_data1[i] = i + 50;
//  }
//  auto buffer0 = std::make_shared<arrow::Buffer>(bit_map, 1 * sizeof(uint8_t));
//  auto buffer1 =
//      std::make_shared<arrow::Buffer>((uint8_t*)buff_data1, 5 * sizeof(uint32_t));
//  std::vector<std::shared_ptr<arrow::Buffer>> buffers1;
//  buffers1.emplace_back(buffer0);
//  buffers1.emplace_back(buffer1);
//  auto array_data1 = arrow::ArrayData::Make(data_type, 5, buffers1);
//  auto array1 = arrow::MakeArray(array_data1);
//
//  auto bit_map2 = new uint8_t{0xff};
//
//  auto buff_data2 = (uint32_t*)malloc(5 * sizeof(uint32_t));
//  for (int i = 0; i < 5; ++i) {
//    buff_data2[i] = i + 50;
//  }
//  auto buffer20 = std::make_shared<arrow::Buffer>(bit_map2, 1 * sizeof(uint8_t));
//  auto buffer21 =
//      std::make_shared<arrow::Buffer>((uint8_t*)buff_data2, 5 * sizeof(uint32_t));
//  std::vector<std::shared_ptr<arrow::Buffer>> buffers2;
//  buffers2.emplace_back(buffer20);
//  buffers2.emplace_back(buffer21);
//  auto array_data2 = arrow::ArrayData::Make(arrow::uint32(), 5, buffers2);
//  auto array2 = arrow::MakeArray(array_data2);
//
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
//      "          \"shape\": {\"value\": \"circle\"},\n"
//      "          \"point_color\": {\"value\": \"#ff0000\"},\n"
//      "          \"point_size\": {\"value\": 30},\n"
//      "          \"opacity\": {\"value\": \"INVALID_NUMBER\"}\n"
//      "        }\n"
//      "      }\n"
//      "    }\n"
//      "  ]\n"
//      "}";
//
//  assert(arctern::render::point_map(array1, array2, vega) == nullptr);
//}

TEST(POINTMAP_TEST, WKT_POINT_TEST) {
  // param1: wkt string
  std::string wkt1 = "POINT (10 10)";
  std::string wkt2 = "POINT (20 20)";
  std::string wkt3 = "POINT (30 30)";
  std::string wkt4 = "POINT (40 40)";
  std::string wkt5 = "POINT (50 50)";
  arrow::StringBuilder string_builder;
  auto status = string_builder.Append(wkt1);
  status = string_builder.Append(wkt2);
  status = string_builder.Append(wkt3);
  status = string_builder.Append(wkt4);
  status = string_builder.Append(wkt5);

  std::shared_ptr<arrow::StringArray> string_array;
  status = string_builder.Finish(&string_array);

  // param2: conf
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
      "          \"shape\": {\"value\": \"circle\"},\n"
      "          \"point_color\": {\"value\": \"#ff0000\"},\n"
      "          \"point_size\": {\"value\": 5},\n"
      "          \"opacity\": {\"value\": 1}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  auto wkb = arctern::gis::gdal::WktToWkb(string_array);

  std::vector<std::shared_ptr<arrow::Array>> point_vec{wkb};

  arctern::render::point_map(point_vec, vega);
}

// TEST(POINTMAP_TEST, WKT_POINT_INVALID_JSON_TEST) {
//  // param1: wkt string
//  std::string wkt1 = "POINT (10 10)";
//  std::string wkt2 = "POINT (20 20)";
//  std::string wkt3 = "POINT (30 30)";
//  std::string wkt4 = "POINT (40 40)";
//  std::string wkt5 = "POINT (50 50)";
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
//  // param2: conf
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
//      "          \"shape\": {\"value\": \"circle\"},\n"
//      "          \"point_color\": {\"value\": \"#ff0000\"},\n"
//      "          \"point_size\": {\"value\": 5},\n"
//      "          \"opacity\": {\"value\": \"INVALID_NUMBER\"}\n"
//      "        }\n"
//      "      }\n"
//      "    }\n"
//      "  ]\n"
//      "}";
//
//  auto wkb = arctern::gis::gdal::WktToWkb(string_array);
//  arctern::render::point_map(wkb, vega);
//}
