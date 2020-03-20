

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

TEST(WEIGHTED_POINTMAP_RAW_POINT_TEST, BLUE_TO_RED) {
  // param1: x, y
  arrow::UInt32Builder x_builder;
  auto status = x_builder.Append(10);
  status = x_builder.Append(40);
  status = x_builder.Append(70);
  status = x_builder.Append(100);
  status = x_builder.Append(130);

  std::shared_ptr<arrow::UInt32Array> x_array;
  status = x_builder.Finish(&x_array);

  arrow::UInt32Builder y_builder;
  status = y_builder.Append(10);
  status = y_builder.Append(40);
  status = y_builder.Append(70);
  status = y_builder.Append(100);
  status = y_builder.Append(130);

  std::shared_ptr<arrow::UInt32Array> y_array;
  status = y_builder.Finish(&y_array);

  // param2: count
  arrow::UInt32Builder color_builder;
  status = color_builder.Append(1);
  status = color_builder.Append(2);
  status = color_builder.Append(3);
  status = color_builder.Append(4);
  status = color_builder.Append(5);

  std::shared_ptr<arrow::UInt32Array> c_array;
  status = color_builder.Finish(&c_array);

  // param3: conf
  const std::string vega =
      "{\n"
      "  \"width\": 300,\n"
      "  \"height\": 200,\n"
      "  \"description\": \"weighted_pointmap\",\n"
      "  \"data\": [\n"
      "    {\n"
      "      \"name\": \"nyc_taxi\",\n"
      "      \"url\": \"data/nyc_taxi_0_5m.csv\"\n"
      "    }\n"
      "  ],\n"
      "  \"scales\": [\n"
      "    {\n"
      "      \"name\": \"x\",\n"
      "      \"type\": \"linear\",\n"
      "      \"domain\": {\"data\": \"nyc_taxi\", \"field\": \"longitude_pickup\"}\n"
      "    },\n"
      "    {\n"
      "      \"name\": \"y\",\n"
      "      \"type\": \"linear\",\n"
      "      \"domain\": {\"data\": \"nyc_taxi\", \"field\": \"latitude_pickup\"}\n"
      "    }\n"
      "  ],\n"
      "  \"marks\": [\n"
      "    {\n"
      "      \"encode\": {\n"
      "        \"enter\": {\n"
      "          \"bounding_box\": [-73.998427, 40.730309, -73.954348, 40.780816],\n"
      "          \"shape\": {\"value\": \"circle\"},\n"
      "          \"strokeWidth\": {\"value\": 3},\n"
      "          \"color_style\": {\"value\": \"blue_to_red\"},\n"
      "          \"ruler\": {\"value\": [2.5, 5]},\n"
      "          \"opacity\": {\"value\": 1.0},\n"
      "          \"coordinate_system\": {\"value\": \"EPSG:3857\"}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  arctern::render::weighted_point_map(x_array, y_array, c_array, vega);
}

TEST(WEIGHTED_POINTMAP_RAW_POINT_TEST, SKYBLUE_TO_WHITE) {
  // param1: x, y
  arrow::UInt32Builder x_builder;
  auto status = x_builder.Append(10);
  status = x_builder.Append(40);
  status = x_builder.Append(70);
  status = x_builder.Append(100);
  status = x_builder.Append(130);

  std::shared_ptr<arrow::UInt32Array> x_array;
  status = x_builder.Finish(&x_array);

  arrow::UInt32Builder y_builder;
  status = y_builder.Append(10);
  status = y_builder.Append(40);
  status = y_builder.Append(70);
  status = y_builder.Append(100);
  status = y_builder.Append(130);

  std::shared_ptr<arrow::UInt32Array> y_array;
  status = y_builder.Finish(&y_array);

  // param2: count
  arrow::UInt32Builder color_builder;
  status = color_builder.Append(1);
  status = color_builder.Append(2);
  status = color_builder.Append(3);
  status = color_builder.Append(4);
  status = color_builder.Append(5);

  std::shared_ptr<arrow::UInt32Array> c_array;
  status = color_builder.Finish(&c_array);

  // param3: conf
  const std::string vega =
      "{\n"
      "  \"width\": 300,\n"
      "  \"height\": 200,\n"
      "  \"description\": \"weighted_pointmap\",\n"
      "  \"data\": [\n"
      "    {\n"
      "      \"name\": \"nyc_taxi\",\n"
      "      \"url\": \"data/nyc_taxi_0_5m.csv\"\n"
      "    }\n"
      "  ],\n"
      "  \"scales\": [\n"
      "    {\n"
      "      \"name\": \"x\",\n"
      "      \"type\": \"linear\",\n"
      "      \"domain\": {\"data\": \"nyc_taxi\", \"field\": \"longitude_pickup\"}\n"
      "    },\n"
      "    {\n"
      "      \"name\": \"y\",\n"
      "      \"type\": \"linear\",\n"
      "      \"domain\": {\"data\": \"nyc_taxi\", \"field\": \"latitude_pickup\"}\n"
      "    }\n"
      "  ],\n"
      "  \"marks\": [\n"
      "    {\n"
      "      \"encode\": {\n"
      "        \"enter\": {\n"
      "          \"bounding_box\": [-73.998427, 40.730309, -73.954348, 40.780816],\n"
      "          \"shape\": {\"value\": \"circle\"},\n"
      "          \"strokeWidth\": {\"value\": 3},\n"
      "          \"color_style\": {\"value\": \"skyblue_to_white\"},\n"
      "          \"ruler\": {\"value\": [2.5, 5]},\n"
      "          \"opacity\": {\"value\": 1.0},\n"
      "          \"coordinate_system\": {\"value\": \"EPSG:3857\"}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  arctern::render::weighted_point_map(x_array, y_array, c_array, vega);
}

TEST(WEIGHTED_POINTMAP_RAW_POINT_TEST, GREEN_YELLOW_RED) {
  // param1: x, y
  arrow::UInt32Builder x_builder;
  auto status = x_builder.Append(10);
  status = x_builder.Append(40);
  status = x_builder.Append(70);
  status = x_builder.Append(100);
  status = x_builder.Append(130);

  std::shared_ptr<arrow::UInt32Array> x_array;
  status = x_builder.Finish(&x_array);

  arrow::UInt32Builder y_builder;
  status = y_builder.Append(10);
  status = y_builder.Append(40);
  status = y_builder.Append(70);
  status = y_builder.Append(100);
  status = y_builder.Append(130);

  std::shared_ptr<arrow::UInt32Array> y_array;
  status = y_builder.Finish(&y_array);

  // param2: count
  arrow::UInt32Builder color_builder;
  status = color_builder.Append(1);
  status = color_builder.Append(2);
  status = color_builder.Append(3);
  status = color_builder.Append(4);
  status = color_builder.Append(5);

  std::shared_ptr<arrow::UInt32Array> c_array;
  status = color_builder.Finish(&c_array);

  // param3: conf
  const std::string vega =
      "{\n"
      "  \"width\": 300,\n"
      "  \"height\": 200,\n"
      "  \"description\": \"weighted_pointmap\",\n"
      "  \"data\": [\n"
      "    {\n"
      "      \"name\": \"nyc_taxi\",\n"
      "      \"url\": \"data/nyc_taxi_0_5m.csv\"\n"
      "    }\n"
      "  ],\n"
      "  \"scales\": [\n"
      "    {\n"
      "      \"name\": \"x\",\n"
      "      \"type\": \"linear\",\n"
      "      \"domain\": {\"data\": \"nyc_taxi\", \"field\": \"longitude_pickup\"}\n"
      "    },\n"
      "    {\n"
      "      \"name\": \"y\",\n"
      "      \"type\": \"linear\",\n"
      "      \"domain\": {\"data\": \"nyc_taxi\", \"field\": \"latitude_pickup\"}\n"
      "    }\n"
      "  ],\n"
      "  \"marks\": [\n"
      "    {\n"
      "      \"encode\": {\n"
      "        \"enter\": {\n"
      "          \"bounding_box\": [-73.998427, 40.730309, -73.954348, 40.780816],\n"
      "          \"shape\": {\"value\": \"circle\"},\n"
      "          \"strokeWidth\": {\"value\": 3},\n"
      "          \"color_style\": {\"value\": \"green_yellow_red\"},\n"
      "          \"ruler\": {\"value\": [2.5, 5]},\n"
      "          \"opacity\": {\"value\": 1.0},\n"
      "          \"coordinate_system\": {\"value\": \"EPSG:3857\"}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  arctern::render::weighted_point_map(x_array, y_array, c_array, vega);
}

TEST(WEIGHTED_POINTMAP_RAW_POINT_TEST, BLUE_WHITE_RED) {
  // param1: x, y
  arrow::UInt32Builder x_builder;
  auto status = x_builder.Append(10);
  status = x_builder.Append(40);
  status = x_builder.Append(70);
  status = x_builder.Append(100);
  status = x_builder.Append(130);

  std::shared_ptr<arrow::UInt32Array> x_array;
  status = x_builder.Finish(&x_array);

  arrow::UInt32Builder y_builder;
  status = y_builder.Append(10);
  status = y_builder.Append(40);
  status = y_builder.Append(70);
  status = y_builder.Append(100);
  status = y_builder.Append(130);

  std::shared_ptr<arrow::UInt32Array> y_array;
  status = y_builder.Finish(&y_array);

  // param2: count
  arrow::UInt32Builder color_builder;
  status = color_builder.Append(1);
  status = color_builder.Append(2);
  status = color_builder.Append(3);
  status = color_builder.Append(4);
  status = color_builder.Append(5);

  std::shared_ptr<arrow::UInt32Array> c_array;
  status = color_builder.Finish(&c_array);

  // param3: conf
  const std::string vega =
      "{\n"
      "  \"width\": 300,\n"
      "  \"height\": 200,\n"
      "  \"description\": \"weighted_pointmap\",\n"
      "  \"data\": [\n"
      "    {\n"
      "      \"name\": \"nyc_taxi\",\n"
      "      \"url\": \"data/nyc_taxi_0_5m.csv\"\n"
      "    }\n"
      "  ],\n"
      "  \"scales\": [\n"
      "    {\n"
      "      \"name\": \"x\",\n"
      "      \"type\": \"linear\",\n"
      "      \"domain\": {\"data\": \"nyc_taxi\", \"field\": \"longitude_pickup\"}\n"
      "    },\n"
      "    {\n"
      "      \"name\": \"y\",\n"
      "      \"type\": \"linear\",\n"
      "      \"domain\": {\"data\": \"nyc_taxi\", \"field\": \"latitude_pickup\"}\n"
      "    }\n"
      "  ],\n"
      "  \"marks\": [\n"
      "    {\n"
      "      \"encode\": {\n"
      "        \"enter\": {\n"
      "          \"bounding_box\": [-73.998427, 40.730309, -73.954348, 40.780816],\n"
      "          \"shape\": {\"value\": \"circle\"},\n"
      "          \"strokeWidth\": {\"value\": 3},\n"
      "          \"color_style\": {\"value\": \"blue_white_red\"},\n"
      "          \"ruler\": {\"value\": [2.5, 5]},\n"
      "          \"opacity\": {\"value\": 1.0},\n"
      "          \"coordinate_system\": {\"value\": \"EPSG:3857\"}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  arctern::render::weighted_point_map(x_array, y_array, c_array, vega);
}

TEST(WEIGHTED_POINTMAP_RAW_POINT_TEST, WHITE_BLUE) {
  // param1: x, y
  arrow::UInt32Builder x_builder;
  auto status = x_builder.Append(10);
  status = x_builder.Append(40);
  status = x_builder.Append(70);
  status = x_builder.Append(100);
  status = x_builder.Append(130);

  std::shared_ptr<arrow::UInt32Array> x_array;
  status = x_builder.Finish(&x_array);

  arrow::UInt32Builder y_builder;
  status = y_builder.Append(10);
  status = y_builder.Append(40);
  status = y_builder.Append(70);
  status = y_builder.Append(100);
  status = y_builder.Append(130);

  std::shared_ptr<arrow::UInt32Array> y_array;
  status = y_builder.Finish(&y_array);

  // param2: count
  arrow::UInt32Builder color_builder;
  status = color_builder.Append(1);
  status = color_builder.Append(2);
  status = color_builder.Append(3);
  status = color_builder.Append(4);
  status = color_builder.Append(5);

  std::shared_ptr<arrow::UInt32Array> c_array;
  status = color_builder.Finish(&c_array);

  // param3: conf
  const std::string vega =
      "{\n"
      "  \"width\": 300,\n"
      "  \"height\": 200,\n"
      "  \"description\": \"weighted_pointmap\",\n"
      "  \"data\": [\n"
      "    {\n"
      "      \"name\": \"nyc_taxi\",\n"
      "      \"url\": \"data/nyc_taxi_0_5m.csv\"\n"
      "    }\n"
      "  ],\n"
      "  \"scales\": [\n"
      "    {\n"
      "      \"name\": \"x\",\n"
      "      \"type\": \"linear\",\n"
      "      \"domain\": {\"data\": \"nyc_taxi\", \"field\": \"longitude_pickup\"}\n"
      "    },\n"
      "    {\n"
      "      \"name\": \"y\",\n"
      "      \"type\": \"linear\",\n"
      "      \"domain\": {\"data\": \"nyc_taxi\", \"field\": \"latitude_pickup\"}\n"
      "    }\n"
      "  ],\n"
      "  \"marks\": [\n"
      "    {\n"
      "      \"encode\": {\n"
      "        \"enter\": {\n"
      "          \"bounding_box\": [-73.998427, 40.730309, -73.954348, 40.780816],\n"
      "          \"shape\": {\"value\": \"circle\"},\n"
      "          \"strokeWidth\": {\"value\": 3},\n"
      "          \"color_style\": {\"value\": \"white_blue\"},\n"
      "          \"ruler\": {\"value\": [2.5, 5]},\n"
      "          \"opacity\": {\"value\": 1.0},\n"
      "          \"coordinate_system\": {\"value\": \"EPSG:3857\"}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  arctern::render::weighted_point_map(x_array, y_array, c_array, vega);
}

TEST(WEIGHTED_POINTMAP_RAW_POINT_TEST, BLUE_GREEN_YELLOW) {
  // param1: x, y
  arrow::UInt32Builder x_builder;
  auto status = x_builder.Append(10);
  status = x_builder.Append(40);
  status = x_builder.Append(70);
  status = x_builder.Append(100);
  status = x_builder.Append(130);

  std::shared_ptr<arrow::UInt32Array> x_array;
  status = x_builder.Finish(&x_array);

  arrow::UInt32Builder y_builder;
  status = y_builder.Append(10);
  status = y_builder.Append(40);
  status = y_builder.Append(70);
  status = y_builder.Append(100);
  status = y_builder.Append(130);

  std::shared_ptr<arrow::UInt32Array> y_array;
  status = y_builder.Finish(&y_array);

  // param2: count
  arrow::UInt32Builder color_builder;
  status = color_builder.Append(1);
  status = color_builder.Append(2);
  status = color_builder.Append(3);
  status = color_builder.Append(4);
  status = color_builder.Append(5);

  std::shared_ptr<arrow::UInt32Array> c_array;
  status = color_builder.Finish(&c_array);

  // param3: conf
  const std::string vega =
      "{\n"
      "  \"width\": 300,\n"
      "  \"height\": 200,\n"
      "  \"description\": \"weighted_pointmap\",\n"
      "  \"data\": [\n"
      "    {\n"
      "      \"name\": \"nyc_taxi\",\n"
      "      \"url\": \"data/nyc_taxi_0_5m.csv\"\n"
      "    }\n"
      "  ],\n"
      "  \"scales\": [\n"
      "    {\n"
      "      \"name\": \"x\",\n"
      "      \"type\": \"linear\",\n"
      "      \"domain\": {\"data\": \"nyc_taxi\", \"field\": \"longitude_pickup\"}\n"
      "    },\n"
      "    {\n"
      "      \"name\": \"y\",\n"
      "      \"type\": \"linear\",\n"
      "      \"domain\": {\"data\": \"nyc_taxi\", \"field\": \"latitude_pickup\"}\n"
      "    }\n"
      "  ],\n"
      "  \"marks\": [\n"
      "    {\n"
      "      \"encode\": {\n"
      "        \"enter\": {\n"
      "          \"bounding_box\": [-73.998427, 40.730309, -73.954348, 40.780816],\n"
      "          \"shape\": {\"value\": \"circle\"},\n"
      "          \"strokeWidth\": {\"value\": 3},\n"
      "          \"color_style\": {\"value\": \"blue_green_yellow\"},\n"
      "          \"ruler\": {\"value\": [2.5, 5]},\n"
      "          \"opacity\": {\"value\": 1.0},\n"
      "          \"coordinate_system\": {\"value\": \"EPSG:3857\"}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  arctern::render::weighted_point_map(x_array, y_array, c_array, vega);
}

TEST(WEIGHTED_POINTMAP_RAW_POINT_TEST, BLUE_TRANSPARENCY) {
  // param1: x, y
  arrow::UInt32Builder x_builder;
  auto status = x_builder.Append(10);
  status = x_builder.Append(40);
  status = x_builder.Append(70);
  status = x_builder.Append(100);
  status = x_builder.Append(130);

  std::shared_ptr<arrow::UInt32Array> x_array;
  status = x_builder.Finish(&x_array);

  arrow::UInt32Builder y_builder;
  status = y_builder.Append(10);
  status = y_builder.Append(40);
  status = y_builder.Append(70);
  status = y_builder.Append(100);
  status = y_builder.Append(130);

  std::shared_ptr<arrow::UInt32Array> y_array;
  status = y_builder.Finish(&y_array);

  // param2: count
  arrow::UInt32Builder color_builder;
  status = color_builder.Append(1);
  status = color_builder.Append(2);
  status = color_builder.Append(3);
  status = color_builder.Append(4);
  status = color_builder.Append(5);

  std::shared_ptr<arrow::UInt32Array> c_array;
  status = color_builder.Finish(&c_array);

  // param3: conf
  const std::string vega =
      "{\n"
      "  \"width\": 300,\n"
      "  \"height\": 200,\n"
      "  \"description\": \"weighted_pointmap\",\n"
      "  \"data\": [\n"
      "    {\n"
      "      \"name\": \"nyc_taxi\",\n"
      "      \"url\": \"data/nyc_taxi_0_5m.csv\"\n"
      "    }\n"
      "  ],\n"
      "  \"scales\": [\n"
      "    {\n"
      "      \"name\": \"x\",\n"
      "      \"type\": \"linear\",\n"
      "      \"domain\": {\"data\": \"nyc_taxi\", \"field\": \"longitude_pickup\"}\n"
      "    },\n"
      "    {\n"
      "      \"name\": \"y\",\n"
      "      \"type\": \"linear\",\n"
      "      \"domain\": {\"data\": \"nyc_taxi\", \"field\": \"latitude_pickup\"}\n"
      "    }\n"
      "  ],\n"
      "  \"marks\": [\n"
      "    {\n"
      "      \"encode\": {\n"
      "        \"enter\": {\n"
      "          \"bounding_box\": [-73.998427, 40.730309, -73.954348, 40.780816],\n"
      "          \"shape\": {\"value\": \"circle\"},\n"
      "          \"strokeWidth\": {\"value\": 3},\n"
      "          \"color_style\": {\"value\": \"blue_transparency\"},\n"
      "          \"ruler\": {\"value\": [2.5, 5]},\n"
      "          \"opacity\": {\"value\": 1.0},\n"
      "          \"coordinate_system\": {\"value\": \"EPSG:3857\"}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  arctern::render::weighted_point_map(x_array, y_array, c_array, vega);
}

TEST(WEIGHTED_POINTMAP_RAW_POINT_TEST, RED_TRANSPARENCY) {
  // param1: x, y
  arrow::UInt32Builder x_builder;
  auto status = x_builder.Append(10);
  status = x_builder.Append(40);
  status = x_builder.Append(70);
  status = x_builder.Append(100);
  status = x_builder.Append(130);

  std::shared_ptr<arrow::UInt32Array> x_array;
  status = x_builder.Finish(&x_array);

  arrow::UInt32Builder y_builder;
  status = y_builder.Append(10);
  status = y_builder.Append(40);
  status = y_builder.Append(70);
  status = y_builder.Append(100);
  status = y_builder.Append(130);

  std::shared_ptr<arrow::UInt32Array> y_array;
  status = y_builder.Finish(&y_array);

  // param2: count
  arrow::UInt32Builder color_builder;
  status = color_builder.Append(1);
  status = color_builder.Append(2);
  status = color_builder.Append(3);
  status = color_builder.Append(4);
  status = color_builder.Append(5);

  std::shared_ptr<arrow::UInt32Array> c_array;
  status = color_builder.Finish(&c_array);

  // param3: conf
  const std::string vega =
      "{\n"
      "  \"width\": 300,\n"
      "  \"height\": 200,\n"
      "  \"description\": \"weighted_pointmap\",\n"
      "  \"data\": [\n"
      "    {\n"
      "      \"name\": \"nyc_taxi\",\n"
      "      \"url\": \"data/nyc_taxi_0_5m.csv\"\n"
      "    }\n"
      "  ],\n"
      "  \"scales\": [\n"
      "    {\n"
      "      \"name\": \"x\",\n"
      "      \"type\": \"linear\",\n"
      "      \"domain\": {\"data\": \"nyc_taxi\", \"field\": \"longitude_pickup\"}\n"
      "    },\n"
      "    {\n"
      "      \"name\": \"y\",\n"
      "      \"type\": \"linear\",\n"
      "      \"domain\": {\"data\": \"nyc_taxi\", \"field\": \"latitude_pickup\"}\n"
      "    }\n"
      "  ],\n"
      "  \"marks\": [\n"
      "    {\n"
      "      \"encode\": {\n"
      "        \"enter\": {\n"
      "          \"bounding_box\": [-73.998427, 40.730309, -73.954348, 40.780816],\n"
      "          \"shape\": {\"value\": \"circle\"},\n"
      "          \"strokeWidth\": {\"value\": 3},\n"
      "          \"color_style\": {\"value\": \"red_transparency\"},\n"
      "          \"ruler\": {\"value\": [2.5, 5]},\n"
      "          \"opacity\": {\"value\": 1.0},\n"
      "          \"coordinate_system\": {\"value\": \"EPSG:3857\"}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  arctern::render::weighted_point_map(x_array, y_array, c_array, vega);
}

TEST(WEIGHTED_POINTMAP_RAW_POINT_TEST, PURPLE_TO_YELLOW) {
  // param1: x, y
  arrow::UInt32Builder x_builder;
  auto status = x_builder.Append(10);
  status = x_builder.Append(40);
  status = x_builder.Append(70);
  status = x_builder.Append(100);
  status = x_builder.Append(130);

  std::shared_ptr<arrow::UInt32Array> x_array;
  status = x_builder.Finish(&x_array);

  arrow::UInt32Builder y_builder;
  status = y_builder.Append(10);
  status = y_builder.Append(40);
  status = y_builder.Append(70);
  status = y_builder.Append(100);
  status = y_builder.Append(130);

  std::shared_ptr<arrow::UInt32Array> y_array;
  status = y_builder.Finish(&y_array);

  // param2: count
  arrow::UInt32Builder color_builder;
  status = color_builder.Append(1);
  status = color_builder.Append(2);
  status = color_builder.Append(3);
  status = color_builder.Append(4);
  status = color_builder.Append(5);

  std::shared_ptr<arrow::UInt32Array> c_array;
  status = color_builder.Finish(&c_array);

  // param3: conf
  const std::string vega =
      "{\n"
      "  \"width\": 300,\n"
      "  \"height\": 200,\n"
      "  \"description\": \"weighted_pointmap\",\n"
      "  \"data\": [\n"
      "    {\n"
      "      \"name\": \"nyc_taxi\",\n"
      "      \"url\": \"data/nyc_taxi_0_5m.csv\"\n"
      "    }\n"
      "  ],\n"
      "  \"scales\": [\n"
      "    {\n"
      "      \"name\": \"x\",\n"
      "      \"type\": \"linear\",\n"
      "      \"domain\": {\"data\": \"nyc_taxi\", \"field\": \"longitude_pickup\"}\n"
      "    },\n"
      "    {\n"
      "      \"name\": \"y\",\n"
      "      \"type\": \"linear\",\n"
      "      \"domain\": {\"data\": \"nyc_taxi\", \"field\": \"latitude_pickup\"}\n"
      "    }\n"
      "  ],\n"
      "  \"marks\": [\n"
      "    {\n"
      "      \"encode\": {\n"
      "        \"enter\": {\n"
      "          \"bounding_box\": [-73.998427, 40.730309, -73.954348, 40.780816],\n"
      "          \"shape\": {\"value\": \"circle\"},\n"
      "          \"strokeWidth\": {\"value\": 3},\n"
      "          \"color_style\": {\"value\": \"purple_to_yellow\"},\n"
      "          \"ruler\": {\"value\": [2.5, 5]},\n"
      "          \"opacity\": {\"value\": 1.0},\n"
      "          \"coordinate_system\": {\"value\": \"EPSG:3857\"}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  arctern::render::weighted_point_map(x_array, y_array, c_array, vega);
}

// TEST(WEIGHTED_POINTMAP_RAW_POINT_TEST, INVALID_COLOR_STYLE_TEST) {
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
//      "          \"color_style\": {\"value\": \"xxxxxx\"},\n"
//      "          \"ruler\": {\"value\": [2.5, 5]},\n"
//      "          \"opacity\": {\"value\": 1.0}\n"
//      "        }\n"
//      "      }\n"
//      "    }\n"
//      "  ]\n"
//      "}";
//
//  arctern::render::choropleth_map(string_array, color_array, vega);
//}
//
// TEST(WEIGHTED_POINTMAP_RAW_POINT_TEST, INVALID_JSON_TEST) {
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
//      "          \"color_style\": {\"value\": \"blue_to_red\"},\n"
//      "          \"ruler\": {\"value\": [2.5, 5]},\n"
//      "          \"opacity\": {\"value\": \"INVALID_NUMBER\"}\n"
//      "        }\n"
//      "      }\n"
//      "    }\n"
//      "  ]\n"
//      "}";
//
//  arctern::render::choropleth_map(string_array, color_array, vega);
//}
//
// TEST(WEIGHTED_POINTMAP_RAW_POINT_TEST, INT8) {
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
//  arrow::Int8Builder color_builder;
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
//      "          \"color_style\": {\"value\": \"blue_to_red\"},\n"
//      "          \"ruler\": {\"value\": [2.5, 5]},\n"
//      "          \"opacity\": {\"value\": 1.0}\n"
//      "        }\n"
//      "      }\n"
//      "    }\n"
//      "  ]\n"
//      "}";
//
//  arctern::render::choropleth_map(string_array, color_array, vega);
//}
//
// TEST(WEIGHTED_POINTMAP_RAW_POINT_TEST, INT16) {
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
//  arrow::Int16Builder color_builder;
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
//      "          \"color_style\": {\"value\": \"blue_to_red\"},\n"
//      "          \"ruler\": {\"value\": [2.5, 5]},\n"
//      "          \"opacity\": {\"value\": 1.0}\n"
//      "        }\n"
//      "      }\n"
//      "    }\n"
//      "  ]\n"
//      "}";
//
//  arctern::render::choropleth_map(string_array, color_array, vega);
//}
//
// TEST(WEIGHTED_POINTMAP_RAW_POINT_TEST, INT32) {
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
//  arrow::Int32Builder color_builder;
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
//      "          \"color_style\": {\"value\": \"blue_to_red\"},\n"
//      "          \"ruler\": {\"value\": [2.5, 5]},\n"
//      "          \"opacity\": {\"value\": 1.0}\n"
//      "        }\n"
//      "      }\n"
//      "    }\n"
//      "  ]\n"
//      "}";
//
//  arctern::render::choropleth_map(string_array, color_array, vega);
//}
//
// TEST(WEIGHTED_POINTMAP_RAW_POINT_TEST, INT64) {
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
//  arrow::Int64Builder color_builder;
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
//      "          \"color_style\": {\"value\": \"blue_to_red\"},\n"
//      "          \"ruler\": {\"value\": [2.5, 5]},\n"
//      "          \"opacity\": {\"value\": 1.0}\n"
//      "        }\n"
//      "      }\n"
//      "    }\n"
//      "  ]\n"
//      "}";
//
//  arctern::render::choropleth_map(string_array, color_array, vega);
//}
//
// TEST(WEIGHTED_POINTMAP_RAW_POINT_TEST, UINT8) {
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
//  arrow::UInt8Builder color_builder;
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
//      "          \"color_style\": {\"value\": \"blue_to_red\"},\n"
//      "          \"ruler\": {\"value\": [2.5, 5]},\n"
//      "          \"opacity\": {\"value\": 1.0}\n"
//      "        }\n"
//      "      }\n"
//      "    }\n"
//      "  ]\n"
//      "}";
//
//  arctern::render::choropleth_map(string_array, color_array, vega);
//}
//
// TEST(WEIGHTED_POINTMAP_RAW_POINT_TEST, UINT16) {
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
//  arrow::UInt16Builder color_builder;
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
//      "          \"color_style\": {\"value\": \"blue_to_red\"},\n"
//      "          \"ruler\": {\"value\": [2.5, 5]},\n"
//      "          \"opacity\": {\"value\": 1.0}\n"
//      "        }\n"
//      "      }\n"
//      "    }\n"
//      "  ]\n"
//      "}";
//
//  arctern::render::choropleth_map(string_array, color_array, vega);
//}
//
// TEST(WEIGHTED_POINTMAP_RAW_POINT_TEST, UINT32) {
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
//      "          \"color_style\": {\"value\": \"blue_to_red\"},\n"
//      "          \"ruler\": {\"value\": [2.5, 5]},\n"
//      "          \"opacity\": {\"value\": 1.0}\n"
//      "        }\n"
//      "      }\n"
//      "    }\n"
//      "  ]\n"
//      "}";
//
//  arctern::render::choropleth_map(string_array, color_array, vega);
//}
//
// TEST(WEIGHTED_POINTMAP_RAW_POINT_TEST, UINT64) {
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
//  arrow::UInt64Builder color_builder;
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
//      "          \"color_style\": {\"value\": \"blue_to_red\"},\n"
//      "          \"ruler\": {\"value\": [2.5, 5]},\n"
//      "          \"opacity\": {\"value\": 1.0}\n"
//      "        }\n"
//      "      }\n"
//      "    }\n"
//      "  ]\n"
//      "}";
//
//  arctern::render::choropleth_map(string_array, color_array, vega);
//}
//
// TEST(WEIGHTED_POINTMAP_RAW_POINT_TEST, FLOAT) {
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
//  arrow::FloatBuilder color_builder;
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
//      "          \"color_style\": {\"value\": \"blue_to_red\"},\n"
//      "          \"ruler\": {\"value\": [2.5, 5]},\n"
//      "          \"opacity\": {\"value\": 1.0}\n"
//      "        }\n"
//      "      }\n"
//      "    }\n"
//      "  ]\n"
//      "}";
//
//  arctern::render::choropleth_map(string_array, color_array, vega);
//}
//
// TEST(WEIGHTED_POINTMAP_RAW_POINT_TEST, DOUBLE) {
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
//  arrow::DoubleBuilder color_builder;
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
//      "          \"color_style\": {\"value\": \"blue_to_red\"},\n"
//      "          \"ruler\": {\"value\": [2.5, 5]},\n"
//      "          \"opacity\": {\"value\": 1.0}\n"
//      "        }\n"
//      "      }\n"
//      "    }\n"
//      "  ]\n"
//      "}";
//
//  arctern::render::choropleth_map(string_array, color_array, vega);
//}
//
// TEST(WEIGHTED_POINTMAP_RAW_POINT_TEST, INVALID_DATA_TYPE_TEST) {
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
//      "          \"color_style\": {\"value\": \"blue_to_red\"},\n"
//      "          \"ruler\": {\"value\": [2.5, 5]},\n"
//      "          \"opacity\": {\"value\": 1.0}\n"
//      "        }\n"
//      "      }\n"
//      "    }\n"
//      "  ]\n"
//      "}";
//
//  arctern::render::choropleth_map(string_array, color_array, vega);
//}
//
TEST(WEIGHTED_POINTMAP_WKT_TEST, BLUE_TO_RED) {
  // param1: wkt string
  std::string wkt_string1 = "POINT (10 10)";
  std::string wkt_string2 = "POINT (40 40)";
  std::string wkt_string3 = "POINT (70 70)";
  std::string wkt_string4 = "POINT (100 100)";
  std::string wkt_string5 = "POINT (130 130)";

  arrow::StringBuilder string_builder;
  auto status = string_builder.Append(wkt_string1);
  status = string_builder.Append(wkt_string2);
  status = string_builder.Append(wkt_string3);
  status = string_builder.Append(wkt_string4);
  status = string_builder.Append(wkt_string5);

  std::shared_ptr<arrow::StringArray> string_array;
  status = string_builder.Finish(&string_array);

  // param2: count
  arrow::UInt32Builder color_builder;
  status = color_builder.Append(1);
  status = color_builder.Append(2);
  status = color_builder.Append(3);
  status = color_builder.Append(4);
  status = color_builder.Append(5);

  std::shared_ptr<arrow::Array> c_array;
  status = color_builder.Finish(&c_array);

  // param3: conf
  const std::string vega =
      "{\n"
      "  \"width\": 300,\n"
      "  \"height\": 200,\n"
      "  \"description\": \"weighted_pointmap\",\n"
      "  \"data\": [\n"
      "    {\n"
      "      \"name\": \"nyc_taxi\",\n"
      "      \"url\": \"data/nyc_taxi_0_5m.csv\"\n"
      "    }\n"
      "  ],\n"
      "  \"scales\": [\n"
      "    {\n"
      "      \"name\": \"x\",\n"
      "      \"type\": \"linear\",\n"
      "      \"domain\": {\"data\": \"nyc_taxi\", \"field\": \"longitude_pickup\"}\n"
      "    },\n"
      "    {\n"
      "      \"name\": \"y\",\n"
      "      \"type\": \"linear\",\n"
      "      \"domain\": {\"data\": \"nyc_taxi\", \"field\": \"latitude_pickup\"}\n"
      "    }\n"
      "  ],\n"
      "  \"marks\": [\n"
      "    {\n"
      "      \"encode\": {\n"
      "        \"enter\": {\n"
      "          \"bounding_box\": [-73.998427, 40.730309, -73.954348, 40.780816],\n"
      "          \"shape\": {\"value\": \"circle\"},\n"
      "          \"strokeWidth\": {\"value\": 3},\n"
      "          \"color_style\": {\"value\": \"blue_to_red\"},\n"
      "          \"ruler\": {\"value\": [2.5, 5]},\n"
      "          \"opacity\": {\"value\": 1.0},\n"
      "          \"coordinate_system\": {\"value\": \"EPSG:3857\"}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  arctern::render::weighted_point_map(string_array, c_array, vega);
}

TEST(WEIGHTED_POINTMAP_WKT_TEST, SKYBLUE_TO_WHITE) {
  // param1: wkt string
  std::string wkt_string1 = "POINT (10 10)";
  std::string wkt_string2 = "POINT (40 40)";
  std::string wkt_string3 = "POINT (70 70)";
  std::string wkt_string4 = "POINT (100 100)";
  std::string wkt_string5 = "POINT (130 130)";

  arrow::StringBuilder string_builder;
  auto status = string_builder.Append(wkt_string1);
  status = string_builder.Append(wkt_string2);
  status = string_builder.Append(wkt_string3);
  status = string_builder.Append(wkt_string4);
  status = string_builder.Append(wkt_string5);

  std::shared_ptr<arrow::StringArray> string_array;
  status = string_builder.Finish(&string_array);

  // param2: count
  arrow::UInt32Builder color_builder;
  status = color_builder.Append(1);
  status = color_builder.Append(2);
  status = color_builder.Append(3);
  status = color_builder.Append(4);
  status = color_builder.Append(5);

  std::shared_ptr<arrow::Array> c_array;
  status = color_builder.Finish(&c_array);

  // param3: conf
  const std::string vega =
      "{\n"
      "  \"width\": 300,\n"
      "  \"height\": 200,\n"
      "  \"description\": \"weighted_pointmap\",\n"
      "  \"data\": [\n"
      "    {\n"
      "      \"name\": \"nyc_taxi\",\n"
      "      \"url\": \"data/nyc_taxi_0_5m.csv\"\n"
      "    }\n"
      "  ],\n"
      "  \"scales\": [\n"
      "    {\n"
      "      \"name\": \"x\",\n"
      "      \"type\": \"linear\",\n"
      "      \"domain\": {\"data\": \"nyc_taxi\", \"field\": \"longitude_pickup\"}\n"
      "    },\n"
      "    {\n"
      "      \"name\": \"y\",\n"
      "      \"type\": \"linear\",\n"
      "      \"domain\": {\"data\": \"nyc_taxi\", \"field\": \"latitude_pickup\"}\n"
      "    }\n"
      "  ],\n"
      "  \"marks\": [\n"
      "    {\n"
      "      \"encode\": {\n"
      "        \"enter\": {\n"
      "          \"bounding_box\": [-73.998427, 40.730309, -73.954348, 40.780816],\n"
      "          \"shape\": {\"value\": \"circle\"},\n"
      "          \"strokeWidth\": {\"value\": 3},\n"
      "          \"color_style\": {\"value\": \"skyblue_to_white\"},\n"
      "          \"ruler\": {\"value\": [2.5, 5]},\n"
      "          \"opacity\": {\"value\": 1.0},\n"
      "          \"coordinate_system\": {\"value\": \"EPSG:3857\"}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  arctern::render::weighted_point_map(string_array, c_array, vega);
}

TEST(WEIGHTED_POINTMAP_WKT_TEST, GREEN_YELLOW_RED) {
  // param1: wkt string
  std::string wkt_string1 = "POINT (10 10)";
  std::string wkt_string2 = "POINT (40 40)";
  std::string wkt_string3 = "POINT (70 70)";
  std::string wkt_string4 = "POINT (100 100)";
  std::string wkt_string5 = "POINT (130 130)";

  arrow::StringBuilder string_builder;
  auto status = string_builder.Append(wkt_string1);
  status = string_builder.Append(wkt_string2);
  status = string_builder.Append(wkt_string3);
  status = string_builder.Append(wkt_string4);
  status = string_builder.Append(wkt_string5);

  std::shared_ptr<arrow::StringArray> string_array;
  status = string_builder.Finish(&string_array);

  // param2: count
  arrow::UInt32Builder color_builder;
  status = color_builder.Append(1);
  status = color_builder.Append(2);
  status = color_builder.Append(3);
  status = color_builder.Append(4);
  status = color_builder.Append(5);

  std::shared_ptr<arrow::Array> c_array;
  status = color_builder.Finish(&c_array);

  // param3: conf
  const std::string vega =
      "{\n"
      "  \"width\": 300,\n"
      "  \"height\": 200,\n"
      "  \"description\": \"weighted_pointmap\",\n"
      "  \"data\": [\n"
      "    {\n"
      "      \"name\": \"nyc_taxi\",\n"
      "      \"url\": \"data/nyc_taxi_0_5m.csv\"\n"
      "    }\n"
      "  ],\n"
      "  \"scales\": [\n"
      "    {\n"
      "      \"name\": \"x\",\n"
      "      \"type\": \"linear\",\n"
      "      \"domain\": {\"data\": \"nyc_taxi\", \"field\": \"longitude_pickup\"}\n"
      "    },\n"
      "    {\n"
      "      \"name\": \"y\",\n"
      "      \"type\": \"linear\",\n"
      "      \"domain\": {\"data\": \"nyc_taxi\", \"field\": \"latitude_pickup\"}\n"
      "    }\n"
      "  ],\n"
      "  \"marks\": [\n"
      "    {\n"
      "      \"encode\": {\n"
      "        \"enter\": {\n"
      "          \"bounding_box\": [-73.998427, 40.730309, -73.954348, 40.780816],\n"
      "          \"shape\": {\"value\": \"circle\"},\n"
      "          \"strokeWidth\": {\"value\": 3},\n"
      "          \"color_style\": {\"value\": \"green_yellow_red\"},\n"
      "          \"ruler\": {\"value\": [2.5, 5]},\n"
      "          \"opacity\": {\"value\": 1.0},\n"
      "          \"coordinate_system\": {\"value\": \"EPSG:3857\"}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  arctern::render::weighted_point_map(string_array, c_array, vega);
}

TEST(WEIGHTED_POINTMAP_WKT_TEST, BLUE_WHITE_RED) {
  // param1: wkt string
  std::string wkt_string1 = "POINT (10 10)";
  std::string wkt_string2 = "POINT (40 40)";
  std::string wkt_string3 = "POINT (70 70)";
  std::string wkt_string4 = "POINT (100 100)";
  std::string wkt_string5 = "POINT (130 130)";

  arrow::StringBuilder string_builder;
  auto status = string_builder.Append(wkt_string1);
  status = string_builder.Append(wkt_string2);
  status = string_builder.Append(wkt_string3);
  status = string_builder.Append(wkt_string4);
  status = string_builder.Append(wkt_string5);

  std::shared_ptr<arrow::StringArray> string_array;
  status = string_builder.Finish(&string_array);

  // param2: count
  arrow::UInt32Builder color_builder;
  status = color_builder.Append(1);
  status = color_builder.Append(2);
  status = color_builder.Append(3);
  status = color_builder.Append(4);
  status = color_builder.Append(5);

  std::shared_ptr<arrow::Array> c_array;
  status = color_builder.Finish(&c_array);

  // param3: conf
  const std::string vega =
      "{\n"
      "  \"width\": 300,\n"
      "  \"height\": 200,\n"
      "  \"description\": \"weighted_pointmap\",\n"
      "  \"data\": [\n"
      "    {\n"
      "      \"name\": \"nyc_taxi\",\n"
      "      \"url\": \"data/nyc_taxi_0_5m.csv\"\n"
      "    }\n"
      "  ],\n"
      "  \"scales\": [\n"
      "    {\n"
      "      \"name\": \"x\",\n"
      "      \"type\": \"linear\",\n"
      "      \"domain\": {\"data\": \"nyc_taxi\", \"field\": \"longitude_pickup\"}\n"
      "    },\n"
      "    {\n"
      "      \"name\": \"y\",\n"
      "      \"type\": \"linear\",\n"
      "      \"domain\": {\"data\": \"nyc_taxi\", \"field\": \"latitude_pickup\"}\n"
      "    }\n"
      "  ],\n"
      "  \"marks\": [\n"
      "    {\n"
      "      \"encode\": {\n"
      "        \"enter\": {\n"
      "          \"bounding_box\": [-73.998427, 40.730309, -73.954348, 40.780816],\n"
      "          \"shape\": {\"value\": \"circle\"},\n"
      "          \"strokeWidth\": {\"value\": 3},\n"
      "          \"color_style\": {\"value\": \"blue_white_red\"},\n"
      "          \"ruler\": {\"value\": [2.5, 5]},\n"
      "          \"opacity\": {\"value\": 1.0},\n"
      "          \"coordinate_system\": {\"value\": \"EPSG:3857\"}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  arctern::render::weighted_point_map(string_array, c_array, vega);
}

TEST(WEIGHTED_POINTMAP_WKT_TEST, WHITE_BLUE) {
  // param1: wkt string
  std::string wkt_string1 = "POINT (10 10)";
  std::string wkt_string2 = "POINT (40 40)";
  std::string wkt_string3 = "POINT (70 70)";
  std::string wkt_string4 = "POINT (100 100)";
  std::string wkt_string5 = "POINT (130 130)";

  arrow::StringBuilder string_builder;
  auto status = string_builder.Append(wkt_string1);
  status = string_builder.Append(wkt_string2);
  status = string_builder.Append(wkt_string3);
  status = string_builder.Append(wkt_string4);
  status = string_builder.Append(wkt_string5);

  std::shared_ptr<arrow::StringArray> string_array;
  status = string_builder.Finish(&string_array);

  // param2: count
  arrow::UInt32Builder color_builder;
  status = color_builder.Append(1);
  status = color_builder.Append(2);
  status = color_builder.Append(3);
  status = color_builder.Append(4);
  status = color_builder.Append(5);

  std::shared_ptr<arrow::Array> c_array;
  status = color_builder.Finish(&c_array);

  // param3: conf
  const std::string vega =
      "{\n"
      "  \"width\": 300,\n"
      "  \"height\": 200,\n"
      "  \"description\": \"weighted_pointmap\",\n"
      "  \"data\": [\n"
      "    {\n"
      "      \"name\": \"nyc_taxi\",\n"
      "      \"url\": \"data/nyc_taxi_0_5m.csv\"\n"
      "    }\n"
      "  ],\n"
      "  \"scales\": [\n"
      "    {\n"
      "      \"name\": \"x\",\n"
      "      \"type\": \"linear\",\n"
      "      \"domain\": {\"data\": \"nyc_taxi\", \"field\": \"longitude_pickup\"}\n"
      "    },\n"
      "    {\n"
      "      \"name\": \"y\",\n"
      "      \"type\": \"linear\",\n"
      "      \"domain\": {\"data\": \"nyc_taxi\", \"field\": \"latitude_pickup\"}\n"
      "    }\n"
      "  ],\n"
      "  \"marks\": [\n"
      "    {\n"
      "      \"encode\": {\n"
      "        \"enter\": {\n"
      "          \"bounding_box\": [-73.998427, 40.730309, -73.954348, 40.780816],\n"
      "          \"shape\": {\"value\": \"circle\"},\n"
      "          \"strokeWidth\": {\"value\": 3},\n"
      "          \"color_style\": {\"value\": \"white_blue\"},\n"
      "          \"ruler\": {\"value\": [2.5, 5]},\n"
      "          \"opacity\": {\"value\": 1.0},\n"
      "          \"coordinate_system\": {\"value\": \"EPSG:3857\"}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  arctern::render::weighted_point_map(string_array, c_array, vega);
}

TEST(WEIGHTED_POINTMAP_WKT_TEST, BLUE_GREEN_YELLOW) {
  // param1: wkt string
  std::string wkt_string1 = "POINT (10 10)";
  std::string wkt_string2 = "POINT (40 40)";
  std::string wkt_string3 = "POINT (70 70)";
  std::string wkt_string4 = "POINT (100 100)";
  std::string wkt_string5 = "POINT (130 130)";

  arrow::StringBuilder string_builder;
  auto status = string_builder.Append(wkt_string1);
  status = string_builder.Append(wkt_string2);
  status = string_builder.Append(wkt_string3);
  status = string_builder.Append(wkt_string4);
  status = string_builder.Append(wkt_string5);

  std::shared_ptr<arrow::StringArray> string_array;
  status = string_builder.Finish(&string_array);

  // param2: count
  arrow::UInt32Builder color_builder;
  status = color_builder.Append(1);
  status = color_builder.Append(2);
  status = color_builder.Append(3);
  status = color_builder.Append(4);
  status = color_builder.Append(5);

  std::shared_ptr<arrow::Array> c_array;
  status = color_builder.Finish(&c_array);

  // param3: conf
  const std::string vega =
      "{\n"
      "  \"width\": 300,\n"
      "  \"height\": 200,\n"
      "  \"description\": \"weighted_pointmap\",\n"
      "  \"data\": [\n"
      "    {\n"
      "      \"name\": \"nyc_taxi\",\n"
      "      \"url\": \"data/nyc_taxi_0_5m.csv\"\n"
      "    }\n"
      "  ],\n"
      "  \"scales\": [\n"
      "    {\n"
      "      \"name\": \"x\",\n"
      "      \"type\": \"linear\",\n"
      "      \"domain\": {\"data\": \"nyc_taxi\", \"field\": \"longitude_pickup\"}\n"
      "    },\n"
      "    {\n"
      "      \"name\": \"y\",\n"
      "      \"type\": \"linear\",\n"
      "      \"domain\": {\"data\": \"nyc_taxi\", \"field\": \"latitude_pickup\"}\n"
      "    }\n"
      "  ],\n"
      "  \"marks\": [\n"
      "    {\n"
      "      \"encode\": {\n"
      "        \"enter\": {\n"
      "          \"bounding_box\": [-73.998427, 40.730309, -73.954348, 40.780816],\n"
      "          \"shape\": {\"value\": \"circle\"},\n"
      "          \"strokeWidth\": {\"value\": 3},\n"
      "          \"color_style\": {\"value\": \"blue_green_yellow\"},\n"
      "          \"ruler\": {\"value\": [2.5, 5]},\n"
      "          \"opacity\": {\"value\": 1.0},\n"
      "          \"coordinate_system\": {\"value\": \"EPSG:3857\"}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  arctern::render::weighted_point_map(string_array, c_array, vega);
}

TEST(WEIGHTED_POINTMAP_WKT_TEST, BLUE_TRANSPARENCY) {
  // param1: wkt string
  std::string wkt_string1 = "POINT (10 10)";
  std::string wkt_string2 = "POINT (40 40)";
  std::string wkt_string3 = "POINT (70 70)";
  std::string wkt_string4 = "POINT (100 100)";
  std::string wkt_string5 = "POINT (130 130)";

  arrow::StringBuilder string_builder;
  auto status = string_builder.Append(wkt_string1);
  status = string_builder.Append(wkt_string2);
  status = string_builder.Append(wkt_string3);
  status = string_builder.Append(wkt_string4);
  status = string_builder.Append(wkt_string5);

  std::shared_ptr<arrow::StringArray> string_array;
  status = string_builder.Finish(&string_array);

  // param2: count
  arrow::UInt32Builder color_builder;
  status = color_builder.Append(1);
  status = color_builder.Append(2);
  status = color_builder.Append(3);
  status = color_builder.Append(4);
  status = color_builder.Append(5);

  std::shared_ptr<arrow::Array> c_array;
  status = color_builder.Finish(&c_array);

  // param3: conf
  const std::string vega =
      "{\n"
      "  \"width\": 300,\n"
      "  \"height\": 200,\n"
      "  \"description\": \"weighted_pointmap\",\n"
      "  \"data\": [\n"
      "    {\n"
      "      \"name\": \"nyc_taxi\",\n"
      "      \"url\": \"data/nyc_taxi_0_5m.csv\"\n"
      "    }\n"
      "  ],\n"
      "  \"scales\": [\n"
      "    {\n"
      "      \"name\": \"x\",\n"
      "      \"type\": \"linear\",\n"
      "      \"domain\": {\"data\": \"nyc_taxi\", \"field\": \"longitude_pickup\"}\n"
      "    },\n"
      "    {\n"
      "      \"name\": \"y\",\n"
      "      \"type\": \"linear\",\n"
      "      \"domain\": {\"data\": \"nyc_taxi\", \"field\": \"latitude_pickup\"}\n"
      "    }\n"
      "  ],\n"
      "  \"marks\": [\n"
      "    {\n"
      "      \"encode\": {\n"
      "        \"enter\": {\n"
      "          \"bounding_box\": [-73.998427, 40.730309, -73.954348, 40.780816],\n"
      "          \"shape\": {\"value\": \"circle\"},\n"
      "          \"strokeWidth\": {\"value\": 3},\n"
      "          \"color_style\": {\"value\": \"blue_transparency\"},\n"
      "          \"ruler\": {\"value\": [2.5, 5]},\n"
      "          \"opacity\": {\"value\": 1.0},\n"
      "          \"coordinate_system\": {\"value\": \"EPSG:3857\"}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  arctern::render::weighted_point_map(string_array, c_array, vega);
}

TEST(WEIGHTED_POINTMAP_WKT_TEST, RED_TRANSPARENCY) {
  // param1: wkt string
  std::string wkt_string1 = "POINT (10 10)";
  std::string wkt_string2 = "POINT (40 40)";
  std::string wkt_string3 = "POINT (70 70)";
  std::string wkt_string4 = "POINT (100 100)";
  std::string wkt_string5 = "POINT (130 130)";

  arrow::StringBuilder string_builder;
  auto status = string_builder.Append(wkt_string1);
  status = string_builder.Append(wkt_string2);
  status = string_builder.Append(wkt_string3);
  status = string_builder.Append(wkt_string4);
  status = string_builder.Append(wkt_string5);

  std::shared_ptr<arrow::StringArray> string_array;
  status = string_builder.Finish(&string_array);

  // param2: count
  arrow::UInt32Builder color_builder;
  status = color_builder.Append(1);
  status = color_builder.Append(2);
  status = color_builder.Append(3);
  status = color_builder.Append(4);
  status = color_builder.Append(5);

  std::shared_ptr<arrow::Array> c_array;
  status = color_builder.Finish(&c_array);

  // param3: conf
  const std::string vega =
      "{\n"
      "  \"width\": 300,\n"
      "  \"height\": 200,\n"
      "  \"description\": \"weighted_pointmap\",\n"
      "  \"data\": [\n"
      "    {\n"
      "      \"name\": \"nyc_taxi\",\n"
      "      \"url\": \"data/nyc_taxi_0_5m.csv\"\n"
      "    }\n"
      "  ],\n"
      "  \"scales\": [\n"
      "    {\n"
      "      \"name\": \"x\",\n"
      "      \"type\": \"linear\",\n"
      "      \"domain\": {\"data\": \"nyc_taxi\", \"field\": \"longitude_pickup\"}\n"
      "    },\n"
      "    {\n"
      "      \"name\": \"y\",\n"
      "      \"type\": \"linear\",\n"
      "      \"domain\": {\"data\": \"nyc_taxi\", \"field\": \"latitude_pickup\"}\n"
      "    }\n"
      "  ],\n"
      "  \"marks\": [\n"
      "    {\n"
      "      \"encode\": {\n"
      "        \"enter\": {\n"
      "          \"bounding_box\": [-73.998427, 40.730309, -73.954348, 40.780816],\n"
      "          \"shape\": {\"value\": \"circle\"},\n"
      "          \"strokeWidth\": {\"value\": 3},\n"
      "          \"color_style\": {\"value\": \"red_transparency\"},\n"
      "          \"ruler\": {\"value\": [2.5, 5]},\n"
      "          \"opacity\": {\"value\": 1.0},\n"
      "          \"coordinate_system\": {\"value\": \"EPSG:3857\"}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  arctern::render::weighted_point_map(string_array, c_array, vega);
}

TEST(WEIGHTED_POINTMAP_WKT_TEST, PURPLE_TO_YELLOW) {
  // param1: wkt string
  std::string wkt_string1 = "POINT (10 10)";
  std::string wkt_string2 = "POINT (40 40)";
  std::string wkt_string3 = "POINT (70 70)";
  std::string wkt_string4 = "POINT (100 100)";
  std::string wkt_string5 = "POINT (130 130)";

  arrow::StringBuilder string_builder;
  auto status = string_builder.Append(wkt_string1);
  status = string_builder.Append(wkt_string2);
  status = string_builder.Append(wkt_string3);
  status = string_builder.Append(wkt_string4);
  status = string_builder.Append(wkt_string5);

  std::shared_ptr<arrow::StringArray> string_array;
  status = string_builder.Finish(&string_array);

  // param2: count
  arrow::UInt32Builder color_builder;
  status = color_builder.Append(1);
  status = color_builder.Append(2);
  status = color_builder.Append(3);
  status = color_builder.Append(4);
  status = color_builder.Append(5);

  std::shared_ptr<arrow::Array> c_array;
  status = color_builder.Finish(&c_array);

  // param3: conf
  const std::string vega =
      "{\n"
      "  \"width\": 300,\n"
      "  \"height\": 200,\n"
      "  \"description\": \"weighted_pointmap\",\n"
      "  \"data\": [\n"
      "    {\n"
      "      \"name\": \"nyc_taxi\",\n"
      "      \"url\": \"data/nyc_taxi_0_5m.csv\"\n"
      "    }\n"
      "  ],\n"
      "  \"scales\": [\n"
      "    {\n"
      "      \"name\": \"x\",\n"
      "      \"type\": \"linear\",\n"
      "      \"domain\": {\"data\": \"nyc_taxi\", \"field\": \"longitude_pickup\"}\n"
      "    },\n"
      "    {\n"
      "      \"name\": \"y\",\n"
      "      \"type\": \"linear\",\n"
      "      \"domain\": {\"data\": \"nyc_taxi\", \"field\": \"latitude_pickup\"}\n"
      "    }\n"
      "  ],\n"
      "  \"marks\": [\n"
      "    {\n"
      "      \"encode\": {\n"
      "        \"enter\": {\n"
      "          \"bounding_box\": [-73.998427, 40.730309, -73.954348, 40.780816],\n"
      "          \"shape\": {\"value\": \"circle\"},\n"
      "          \"strokeWidth\": {\"value\": 8},\n"
      "          \"color_style\": {\"value\": \"purple_to_yellow\"},\n"
      "          \"ruler\": {\"value\": [2.5, 5]},\n"
      "          \"opacity\": {\"value\": 1.0},\n"
      "          \"coordinate_system\": {\"value\": \"EPSG:3857\"}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  arctern::render::weighted_point_map(string_array, c_array, vega);
}
//
// TEST(WEIGHTED_POINTMAP_WKT_TEST, INVALID_COLOR_STYLE_TEST) {
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
//      "          \"color_style\": {\"value\": \"xxxxxx\"},\n"
//      "          \"ruler\": {\"value\": [2.5, 5]},\n"
//      "          \"opacity\": {\"value\": 1.0}\n"
//      "        }\n"
//      "      }\n"
//      "    }\n"
//      "  ]\n"
//      "}";
//
//  arctern::render::choropleth_map(string_array, color_array, vega);
//}
//
// TEST(WEIGHTED_POINTMAP_WKT_TEST, INVALID_JSON_TEST) {
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
//      "          \"color_style\": {\"value\": \"blue_to_red\"},\n"
//      "          \"ruler\": {\"value\": [2.5, 5]},\n"
//      "          \"opacity\": {\"value\": \"INVALID_NUMBER\"}\n"
//      "        }\n"
//      "      }\n"
//      "    }\n"
//      "  ]\n"
//      "}";
//
//  arctern::render::choropleth_map(string_array, color_array, vega);
//}
//
// TEST(WEIGHTED_POINTMAP_WKT_TEST, INT8) {
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
//  arrow::Int8Builder color_builder;
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
//      "          \"color_style\": {\"value\": \"blue_to_red\"},\n"
//      "          \"ruler\": {\"value\": [2.5, 5]},\n"
//      "          \"opacity\": {\"value\": 1.0}\n"
//      "        }\n"
//      "      }\n"
//      "    }\n"
//      "  ]\n"
//      "}";
//
//  arctern::render::choropleth_map(string_array, color_array, vega);
//}
//
// TEST(WEIGHTED_POINTMAP_WKT_TEST, INT16) {
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
//  arrow::Int16Builder color_builder;
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
//      "          \"color_style\": {\"value\": \"blue_to_red\"},\n"
//      "          \"ruler\": {\"value\": [2.5, 5]},\n"
//      "          \"opacity\": {\"value\": 1.0}\n"
//      "        }\n"
//      "      }\n"
//      "    }\n"
//      "  ]\n"
//      "}";
//
//  arctern::render::choropleth_map(string_array, color_array, vega);
//}
//
// TEST(WEIGHTED_POINTMAP_WKT_TEST, INT32) {
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
//  arrow::Int32Builder color_builder;
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
//      "          \"color_style\": {\"value\": \"blue_to_red\"},\n"
//      "          \"ruler\": {\"value\": [2.5, 5]},\n"
//      "          \"opacity\": {\"value\": 1.0}\n"
//      "        }\n"
//      "      }\n"
//      "    }\n"
//      "  ]\n"
//      "}";
//
//  arctern::render::choropleth_map(string_array, color_array, vega);
//}
//
// TEST(WEIGHTED_POINTMAP_WKT_TEST, INT64) {
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
//  arrow::Int64Builder color_builder;
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
//      "          \"color_style\": {\"value\": \"blue_to_red\"},\n"
//      "          \"ruler\": {\"value\": [2.5, 5]},\n"
//      "          \"opacity\": {\"value\": 1.0}\n"
//      "        }\n"
//      "      }\n"
//      "    }\n"
//      "  ]\n"
//      "}";
//
//  arctern::render::choropleth_map(string_array, color_array, vega);
//}
//
// TEST(WEIGHTED_POINTMAP_WKT_TEST, UINT8) {
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
//  arrow::UInt8Builder color_builder;
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
//      "          \"color_style\": {\"value\": \"blue_to_red\"},\n"
//      "          \"ruler\": {\"value\": [2.5, 5]},\n"
//      "          \"opacity\": {\"value\": 1.0}\n"
//      "        }\n"
//      "      }\n"
//      "    }\n"
//      "  ]\n"
//      "}";
//
//  arctern::render::choropleth_map(string_array, color_array, vega);
//}
//
// TEST(WEIGHTED_POINTMAP_WKT_TEST, UINT16) {
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
//  arrow::UInt16Builder color_builder;
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
//      "          \"color_style\": {\"value\": \"blue_to_red\"},\n"
//      "          \"ruler\": {\"value\": [2.5, 5]},\n"
//      "          \"opacity\": {\"value\": 1.0}\n"
//      "        }\n"
//      "      }\n"
//      "    }\n"
//      "  ]\n"
//      "}";
//
//  arctern::render::choropleth_map(string_array, color_array, vega);
//}
//
// TEST(WEIGHTED_POINTMAP_WKT_TEST, UINT32) {
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
//      "          \"color_style\": {\"value\": \"blue_to_red\"},\n"
//      "          \"ruler\": {\"value\": [2.5, 5]},\n"
//      "          \"opacity\": {\"value\": 1.0}\n"
//      "        }\n"
//      "      }\n"
//      "    }\n"
//      "  ]\n"
//      "}";
//
//  arctern::render::choropleth_map(string_array, color_array, vega);
//}
//
// TEST(WEIGHTED_POINTMAP_WKT_TEST, UINT64) {
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
//  arrow::UInt64Builder color_builder;
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
//      "          \"color_style\": {\"value\": \"blue_to_red\"},\n"
//      "          \"ruler\": {\"value\": [2.5, 5]},\n"
//      "          \"opacity\": {\"value\": 1.0}\n"
//      "        }\n"
//      "      }\n"
//      "    }\n"
//      "  ]\n"
//      "}";
//
//  arctern::render::choropleth_map(string_array, color_array, vega);
//}
//
// TEST(WEIGHTED_POINTMAP_WKT_TEST, FLOAT) {
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
//  arrow::FloatBuilder color_builder;
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
//      "          \"color_style\": {\"value\": \"blue_to_red\"},\n"
//      "          \"ruler\": {\"value\": [2.5, 5]},\n"
//      "          \"opacity\": {\"value\": 1.0}\n"
//      "        }\n"
//      "      }\n"
//      "    }\n"
//      "  ]\n"
//      "}";
//
//  arctern::render::choropleth_map(string_array, color_array, vega);
//}
//
// TEST(WEIGHTED_POINTMAP_WKT_TEST, DOUBLE) {
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
//  arrow::DoubleBuilder color_builder;
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
//      "          \"color_style\": {\"value\": \"blue_to_red\"},\n"
//      "          \"ruler\": {\"value\": [2.5, 5]},\n"
//      "          \"opacity\": {\"value\": 1.0}\n"
//      "        }\n"
//      "      }\n"
//      "    }\n"
//      "  ]\n"
//      "}";
//
//  arctern::render::choropleth_map(string_array, color_array, vega);
//}
//
// TEST(WEIGHTED_POINTMAP_WKT_TEST, INVALID_DATA_TYPE_TEST) {
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
//      "          \"color_style\": {\"value\": \"blue_to_red\"},\n"
//      "          \"ruler\": {\"value\": [2.5, 5]},\n"
//      "          \"opacity\": {\"value\": 1.0}\n"
//      "        }\n"
//      "      }\n"
//      "    }\n"
//      "  ]\n"
//      "}";
//
//  arctern::render::choropleth_map(string_array, color_array, vega);
//}
