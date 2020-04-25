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

TEST(ICON_VIZ_TEST, RAW_POINT_TEST) {
  // param1: x, y
  arrow::UInt32Builder x_builder;
  auto status = x_builder.Append(0);
  status = x_builder.Append(100);
  status = x_builder.Append(200);
  status = x_builder.Append(300);
  status = x_builder.Append(400);
  status = x_builder.Append(500);

  std::shared_ptr<arrow::UInt32Array> x_array;
  status = x_builder.Finish(&x_array);

  arrow::UInt32Builder y_builder;
  status = y_builder.Append(0);
  status = y_builder.Append(100);
  status = y_builder.Append(200);
  status = y_builder.Append(300);
  status = y_builder.Append(400);
  status = y_builder.Append(500);

  std::shared_ptr<arrow::UInt32Array> y_array;
  status = y_builder.Finish(&y_array);

  std::string path = __FILE__;
  path.resize(path.size() - 16);
  std::string icon_path = path + "images/taxi.png";

  // param2: vega
  const std::string vega =
      "{\n"
      "  \"width\": 800,\n"
      "  \"height\": 600,\n"
      "  \"description\": \"icon\",\n"
      "  \"data\": [\n"
      "    {\n"
      "      \"name\": \"nyc_taxi\",\n"
      "      \"url\": \"data/nyc_taxi_0_5m.csv\"\n"
      "    }\n"
      "  ],\n"
      "  \"scales\": [\n"
      "    {\n"
      "      \"name\": \"icon\",\n"
      "      \"type\": \"linear\",\n"
      "      \"domain\": {\"data\": \"nyc_taxi\", \"field\": \"c0\"}\n"
      "    }\n"
      "  ],\n"
      "  \"marks\": [\n"
      "    {\n"
      "      \"encode\": {\n"
      "        \"enter\": {\n"
      "          \"bounding_box\": [-73.998427, 40.730309, -73.954348, 40.780816],\n"
      "          \"icon_path\": {\"value\": \"" +
      icon_path +
      "\"},\n"
      "          \"coordinate_system\": {\"value\": \"EPSG:3857\"}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  arctern::render::icon_viz(x_array, y_array, vega);
}

TEST(ICON_VIZ_TEST, WKT_TEST) {
  // param1: wkt string
  std::string wkt1 = "POINT (100 100)";
  std::string wkt2 = "POINT (200 200)";
  std::string wkt3 = "POINT (300 300)";
  std::string wkt4 = "POINT (400 400)";
  std::string wkt5 = "POINT (500 500)";
  arrow::StringBuilder string_builder;
  auto status = string_builder.Append(wkt1);
  status = string_builder.Append(wkt2);
  status = string_builder.Append(wkt3);
  status = string_builder.Append(wkt4);
  status = string_builder.Append(wkt5);

  std::shared_ptr<arrow::StringArray> string_array;
  status = string_builder.Finish(&string_array);

  std::string path = __FILE__;
  path.resize(path.size() - 16);
  std::string icon_path = path + "images/taxi.png";

  // param2: vega
  const std::string vega =
      "{\n"
      "  \"width\": 800,\n"
      "  \"height\": 600,\n"
      "  \"description\": \"icon\",\n"
      "  \"data\": [\n"
      "    {\n"
      "      \"name\": \"nyc_taxi\",\n"
      "      \"url\": \"data/nyc_taxi_0_5m.csv\"\n"
      "    }\n"
      "  ],\n"
      "  \"scales\": [\n"
      "    {\n"
      "      \"name\": \"icon\",\n"
      "      \"type\": \"linear\",\n"
      "      \"domain\": {\"data\": \"nyc_taxi\", \"field\": \"c0\"}\n"
      "    }\n"
      "  ],\n"
      "  \"marks\": [\n"
      "    {\n"
      "      \"encode\": {\n"
      "        \"enter\": {\n"
      "          \"bounding_box\": [-73.998427, 40.730309, -73.954348, 40.780816],\n"
      "          \"icon_path\": {\"value\": \"" +
      icon_path +
      "\"},\n"
      "          \"coordinate_system\": {\"value\": \"EPSG:3857\"}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  auto wkb = arctern::render::WktToWkb(string_array);
  arctern::render::icon_viz(wkb, vega);
}
