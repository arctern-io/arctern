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
      "          \"color_style\": {\"value\": \"blue_to_red\"},\n"
      "          \"ruler\": {\"value\": [2.5, 5]},\n"
      "          \"opacity\": {\"value\": 1.0}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  auto wkb = arctern::render::WktToWkb(string_array);
  arctern::render::choropleth_map(wkb, color_array, vega);
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
      "          \"color_style\": {\"value\": \"skyblue_to_white\"},\n"
      "          \"ruler\": {\"value\": [2.5, 5]},\n"
      "          \"opacity\": {\"value\": 1.0}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  auto wkb = arctern::render::WktToWkb(string_array);
  arctern::render::choropleth_map(wkb, color_array, vega);
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
      "          \"color_style\": {\"value\": \"green_yellow_red\"},\n"
      "          \"ruler\": {\"value\": [2.5, 5]},\n"
      "          \"opacity\": {\"value\": 1.0}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  auto wkb = arctern::render::WktToWkb(string_array);
  arctern::render::choropleth_map(wkb, color_array, vega);
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
      "          \"color_style\": {\"value\": \"blue_white_red\"},\n"
      "          \"ruler\": {\"value\": [2.5, 5]},\n"
      "          \"opacity\": {\"value\": 1.0}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  auto wkb = arctern::render::WktToWkb(string_array);
  arctern::render::choropleth_map(wkb, color_array, vega);
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
      "          \"color_style\": {\"value\": \"white_blue\"},\n"
      "          \"ruler\": {\"value\": [2.5, 5]},\n"
      "          \"opacity\": {\"value\": 1.0}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  auto wkb = arctern::render::WktToWkb(string_array);
  arctern::render::choropleth_map(wkb, color_array, vega);
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
      "          \"color_style\": {\"value\": \"blue_green_yellow\"},\n"
      "          \"ruler\": {\"value\": [2.5, 5]},\n"
      "          \"opacity\": {\"value\": 1.0}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  auto wkb = arctern::render::WktToWkb(string_array);
  arctern::render::choropleth_map(wkb, color_array, vega);
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
      "          \"color_style\": {\"value\": \"blue_transparency\"},\n"
      "          \"ruler\": {\"value\": [2.5, 5]},\n"
      "          \"opacity\": {\"value\": 1.0}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  auto wkb = arctern::render::WktToWkb(string_array);
  arctern::render::choropleth_map(wkb, color_array, vega);
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
      "          \"color_style\": {\"value\": \"red_transparency\"},\n"
      "          \"ruler\": {\"value\": [2.5, 5]},\n"
      "          \"opacity\": {\"value\": 1.0}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  auto wkb = arctern::render::WktToWkb(string_array);
  arctern::render::choropleth_map(wkb, color_array, vega);
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
      "          \"color_style\": {\"value\": \"purple_to_yellow\"},\n"
      "          \"ruler\": {\"value\": [2.5, 5]},\n"
      "          \"opacity\": {\"value\": 1.0}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  auto wkb = arctern::render::WktToWkb(string_array);
  arctern::render::choropleth_map(wkb, color_array, vega);
}

TEST(CHOROPLETHMAP_TEST, INVALID_COLOR_STYLE_TEST) {
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
      "          \"color_style\": {\"value\": \"xxxxxx\"},\n"
      "          \"ruler\": {\"value\": [2.5, 5]},\n"
      "          \"opacity\": {\"value\": 1.0}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  auto wkb = arctern::render::WktToWkb(string_array);
  arctern::render::choropleth_map(wkb, color_array, vega);
}

TEST(CHOROPLETHMAP_TEST, INVALID_JSON_TEST) {
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
      "          \"color_style\": {\"value\": \"blue_to_red\"},\n"
      "          \"ruler\": {\"value\": [2.5, 5]},\n"
      "          \"opacity\": {\"value\": \"INVALID_NUMBER\"}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  auto wkb = arctern::render::WktToWkb(string_array);
  arctern::render::choropleth_map(wkb, color_array, vega);
}

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
      "          \"color_style\": {\"value\": \"blue_to_red\"},\n"
      "          \"ruler\": {\"value\": [2.5, 5]},\n"
      "          \"opacity\": {\"value\": 1.0}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  auto wkb = arctern::render::WktToWkb(string_array);
  arctern::render::choropleth_map(wkb, color_array, vega);
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
      "          \"color_style\": {\"value\": \"blue_to_red\"},\n"
      "          \"ruler\": {\"value\": [2.5, 5]},\n"
      "          \"opacity\": {\"value\": 1.0}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  auto wkb = arctern::render::WktToWkb(string_array);
  arctern::render::choropleth_map(wkb, color_array, vega);
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
      "          \"color_style\": {\"value\": \"blue_to_red\"},\n"
      "          \"ruler\": {\"value\": [2.5, 5]},\n"
      "          \"opacity\": {\"value\": 1.0}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  auto wkb = arctern::render::WktToWkb(string_array);
  arctern::render::choropleth_map(wkb, color_array, vega);
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
      "          \"color_style\": {\"value\": \"blue_to_red\"},\n"
      "          \"ruler\": {\"value\": [2.5, 5]},\n"
      "          \"opacity\": {\"value\": 1.0}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  auto wkb = arctern::render::WktToWkb(string_array);
  arctern::render::choropleth_map(wkb, color_array, vega);
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
      "          \"color_style\": {\"value\": \"blue_to_red\"},\n"
      "          \"ruler\": {\"value\": [2.5, 5]},\n"
      "          \"opacity\": {\"value\": 1.0}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  auto wkb = arctern::render::WktToWkb(string_array);
  arctern::render::choropleth_map(wkb, color_array, vega);
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
      "          \"color_style\": {\"value\": \"blue_to_red\"},\n"
      "          \"ruler\": {\"value\": [2.5, 5]},\n"
      "          \"opacity\": {\"value\": 1.0}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  auto wkb = arctern::render::WktToWkb(string_array);
  arctern::render::choropleth_map(wkb, color_array, vega);
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
      "          \"color_style\": {\"value\": \"blue_to_red\"},\n"
      "          \"ruler\": {\"value\": [2.5, 5]},\n"
      "          \"opacity\": {\"value\": 1.0}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  auto wkb = arctern::render::WktToWkb(string_array);
  arctern::render::choropleth_map(wkb, color_array, vega);
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
      "          \"color_style\": {\"value\": \"blue_to_red\"},\n"
      "          \"ruler\": {\"value\": [2.5, 5]},\n"
      "          \"opacity\": {\"value\": 1.0}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  auto wkb = arctern::render::WktToWkb(string_array);
  arctern::render::choropleth_map(wkb, color_array, vega);
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
      "          \"color_style\": {\"value\": \"blue_to_red\"},\n"
      "          \"ruler\": {\"value\": [2.5, 5]},\n"
      "          \"opacity\": {\"value\": 1.0}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  auto wkb = arctern::render::WktToWkb(string_array);
  arctern::render::choropleth_map(wkb, color_array, vega);
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
      "          \"color_style\": {\"value\": \"blue_to_red\"},\n"
      "          \"ruler\": {\"value\": [2.5, 5]},\n"
      "          \"opacity\": {\"value\": 1.0}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  auto wkb = arctern::render::WktToWkb(string_array);
  arctern::render::choropleth_map(wkb, color_array, vega);
}

TEST(CHOROPLETHMAP_TEST, INVALID_DATA_TYPE_TEST) {
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
  arrow::StringBuilder color_builder;
  status = color_builder.Append("");
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
      "          \"color_style\": {\"value\": \"blue_to_red\"},\n"
      "          \"ruler\": {\"value\": [2.5, 5]},\n"
      "          \"opacity\": {\"value\": 1.0}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  auto wkb = arctern::render::WktToWkb(string_array);
  arctern::render::choropleth_map(wkb, color_array, vega);
}
