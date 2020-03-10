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

TEST(TWOD_TEST, HEATMAP_TEST) {
  // param1: wkt string
  std::string wkt_string1 =
      "POLYGON (("
      "-73.98128 40.754771, "
      "-73.980185 40.754771, "
      "-73.980185 40.755587, "
      "-73.98128 40.755587, "
      "-73.98128 40.754771))";
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

  arctern::render::choropleth_map(string_array, color_array, vega);
}
