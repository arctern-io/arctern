

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

TEST(POINTMAP_RAW_POINT_TEST, SINGLE_COLOR_SINGLE_POINTSIZE) {
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

  // param2: conf
  const std::string vega =
      "{\n"
      "  \"width\": 300,\n"
      "  \"height\": 200,\n"
      "  \"description\": \"pointmap\",\n"
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
      "          \"color_gradient\": {\"value\": [\"#FFD700\"]},\n"
      "          \"color_bound\": {\"value\": [-1, -1]},\n"
      "          \"size_bound\": {\"value\": [5]},\n"
      "          \"opacity\": {\"value\": 1.0},\n"
      "          \"coordinate_system\": {\"value\": \"EPSG:3857\"}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  arctern::render::weighted_point_map(x_array, y_array, vega);
}

TEST(POINTMAP_RAW_POINT_TEST, MULTIPLE_COLOR_SINGLE_POINTSIZE) {
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
      "  \"description\": \"pointmap\",\n"
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
      "          \"color_gradient\": {\"value\": [\"#0000FF\", \"#FF0000\"]},\n"
      "          \"color_bound\": {\"value\": [2, 5]},\n"
      "          \"size_bound\": {\"value\": [8]},\n"
      "          \"opacity\": {\"value\": 1.0},\n"
      "          \"coordinate_system\": {\"value\": \"EPSG:3857\"}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  arctern::render::weighted_point_map(x_array, y_array, c_array, vega);
}

TEST(POINTMAP_RAW_POINT_TEST, SINGLE_COLOR_MULTIPLE_POINTSIZE) {
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

  // param2: point_size
  arrow::UInt32Builder ps_builder;
  status = ps_builder.Append(2);
  status = ps_builder.Append(4);
  status = ps_builder.Append(6);
  status = ps_builder.Append(8);
  status = ps_builder.Append(10);

  std::shared_ptr<arrow::UInt32Array> s_array;
  status = ps_builder.Finish(&s_array);

  // param3: conf
  const std::string vega =
      "{\n"
      "  \"width\": 300,\n"
      "  \"height\": 200,\n"
      "  \"description\": \"pointmap\",\n"
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
      "          \"color_gradient\": {\"value\": [\"#FFD700\"]},\n"
      "          \"color_bound\": {\"value\": [-1, -1]},\n"
      "          \"size_bound\": {\"value\": [0, 10]},\n"
      "          \"opacity\": {\"value\": 1.0},\n"
      "          \"coordinate_system\": {\"value\": \"EPSG:3857\"}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  arctern::render::weighted_point_map(x_array, y_array, s_array, vega);
}

TEST(POINTMAP_RAW_POINT_TEST, MULTIPLE_COLOR_MULTIPLE_POINTSIZE) {
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

  // param2: color
  arrow::UInt32Builder color_builder;
  status = color_builder.Append(1);
  status = color_builder.Append(2);
  status = color_builder.Append(3);
  status = color_builder.Append(4);
  status = color_builder.Append(5);

  std::shared_ptr<arrow::UInt32Array> c_array;
  status = color_builder.Finish(&c_array);

  // param3: point size
  arrow::UInt32Builder point_size_builder;
  status = point_size_builder.Append(2);
  status = point_size_builder.Append(4);
  status = point_size_builder.Append(6);
  status = point_size_builder.Append(8);
  status = point_size_builder.Append(10);

  std::shared_ptr<arrow::UInt32Array> s_array;
  status = point_size_builder.Finish(&s_array);

  // param4: conf
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
      "          \"color_gradient\": {\"value\": [\"#0000FF\", \"#FF0000\"]},\n"
      "          \"color_bound\": {\"value\": [2.5, 5]},\n"
      "          \"size_bound\": {\"value\": [2.5, 5]},\n"
      "          \"opacity\": {\"value\": 1.0},\n"
      "          \"coordinate_system\": {\"value\": \"EPSG:3857\"}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  arctern::render::weighted_point_map(x_array, y_array, c_array, s_array, vega);
}

TEST(POINTMAP_RAW_POINT_TEST, INT8) {
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

  // param2: color
  arrow::Int8Builder color_builder;
  status = color_builder.Append(1);
  status = color_builder.Append(2);
  status = color_builder.Append(3);
  status = color_builder.Append(4);
  status = color_builder.Append(5);

  std::shared_ptr<arrow::Int8Array> c_array;
  status = color_builder.Finish(&c_array);

  // param3: point size
  arrow::Int8Builder point_size_builder;
  status = point_size_builder.Append(2);
  status = point_size_builder.Append(4);
  status = point_size_builder.Append(6);
  status = point_size_builder.Append(8);
  status = point_size_builder.Append(10);

  std::shared_ptr<arrow::Int8Array> s_array;
  status = point_size_builder.Finish(&s_array);

  // param4: conf
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
      "          \"color_gradient\": {\"value\": [\"#0000FF\", \"#FF0000\"]},\n"
      "          \"color_bound\": {\"value\": [2.5, 5]},\n"
      "          \"size_bound\": {\"value\": [2.5, 5]},\n"
      "          \"opacity\": {\"value\": 1.0},\n"
      "          \"coordinate_system\": {\"value\": \"EPSG:3857\"}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  arctern::render::weighted_point_map(x_array, y_array, c_array, s_array, vega);
}

TEST(POINTMAP_RAW_POINT_TEST, INT16) {
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

  // param2: color
  arrow::Int16Builder color_builder;
  status = color_builder.Append(1);
  status = color_builder.Append(2);
  status = color_builder.Append(3);
  status = color_builder.Append(4);
  status = color_builder.Append(5);

  std::shared_ptr<arrow::Int16Array> c_array;
  status = color_builder.Finish(&c_array);

  // param3: point size
  arrow::Int16Builder point_size_builder;
  status = point_size_builder.Append(2);
  status = point_size_builder.Append(4);
  status = point_size_builder.Append(6);
  status = point_size_builder.Append(8);
  status = point_size_builder.Append(10);

  std::shared_ptr<arrow::Int16Array> s_array;
  status = point_size_builder.Finish(&s_array);

  // param4: conf
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
      "          \"color_gradient\": {\"value\": [\"#0000FF\", \"#FF0000\"]},\n"
      "          \"color_bound\": {\"value\": [2.5, 5]},\n"
      "          \"size_bound\": {\"value\": [2.5, 5]},\n"
      "          \"opacity\": {\"value\": 1.0},\n"
      "          \"coordinate_system\": {\"value\": \"EPSG:3857\"}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  arctern::render::weighted_point_map(x_array, y_array, c_array, s_array, vega);
}

TEST(POINTMAP_RAW_POINT_TEST, INT32) {
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

  // param2: color
  arrow::Int32Builder color_builder;
  status = color_builder.Append(1);
  status = color_builder.Append(2);
  status = color_builder.Append(3);
  status = color_builder.Append(4);
  status = color_builder.Append(5);

  std::shared_ptr<arrow::Int32Array> c_array;
  status = color_builder.Finish(&c_array);

  // param3: point size
  arrow::Int32Builder point_size_builder;
  status = point_size_builder.Append(2);
  status = point_size_builder.Append(4);
  status = point_size_builder.Append(6);
  status = point_size_builder.Append(8);
  status = point_size_builder.Append(10);

  std::shared_ptr<arrow::Int32Array> s_array;
  status = point_size_builder.Finish(&s_array);

  // param4: conf
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
      "          \"color_gradient\": {\"value\": [\"#0000FF\", \"#FF0000\"]},\n"
      "          \"color_bound\": {\"value\": [2.5, 5]},\n"
      "          \"size_bound\": {\"value\": [2.5, 5]},\n"
      "          \"opacity\": {\"value\": 1.0},\n"
      "          \"coordinate_system\": {\"value\": \"EPSG:3857\"}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  arctern::render::weighted_point_map(x_array, y_array, c_array, s_array, vega);
}

TEST(POINTMAP_RAW_POINT_TEST, INT64) {
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

  // param2: color
  arrow::Int64Builder color_builder;
  status = color_builder.Append(1);
  status = color_builder.Append(2);
  status = color_builder.Append(3);
  status = color_builder.Append(4);
  status = color_builder.Append(5);

  std::shared_ptr<arrow::Int64Array> c_array;
  status = color_builder.Finish(&c_array);

  // param3: point size
  arrow::Int64Builder point_size_builder;
  status = point_size_builder.Append(2);
  status = point_size_builder.Append(4);
  status = point_size_builder.Append(6);
  status = point_size_builder.Append(8);
  status = point_size_builder.Append(10);

  std::shared_ptr<arrow::Int64Array> s_array;
  status = point_size_builder.Finish(&s_array);

  // param4: conf
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
      "          \"color_gradient\": {\"value\": [\"#0000FF\", \"#FF0000\"]},\n"
      "          \"color_bound\": {\"value\": [2.5, 5]},\n"
      "          \"size_bound\": {\"value\": [2.5, 5]},\n"
      "          \"opacity\": {\"value\": 1.0},\n"
      "          \"coordinate_system\": {\"value\": \"EPSG:3857\"}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  arctern::render::weighted_point_map(x_array, y_array, c_array, s_array, vega);
}

TEST(POINTMAP_RAW_POINT_TEST, UINT8) {
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

  // param2: color
  arrow::UInt8Builder color_builder;
  status = color_builder.Append(1);
  status = color_builder.Append(2);
  status = color_builder.Append(3);
  status = color_builder.Append(4);
  status = color_builder.Append(5);

  std::shared_ptr<arrow::UInt8Array> c_array;
  status = color_builder.Finish(&c_array);

  // param3: point size
  arrow::UInt8Builder point_size_builder;
  status = point_size_builder.Append(2);
  status = point_size_builder.Append(4);
  status = point_size_builder.Append(6);
  status = point_size_builder.Append(8);
  status = point_size_builder.Append(10);

  std::shared_ptr<arrow::UInt8Array> s_array;
  status = point_size_builder.Finish(&s_array);

  // param4: conf
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
      "          \"color_gradient\": {\"value\": [\"#0000FF\", \"#FF0000\"]},\n"
      "          \"color_bound\": {\"value\": [2.5, 5]},\n"
      "          \"size_bound\": {\"value\": [2.5, 5]},\n"
      "          \"opacity\": {\"value\": 1.0},\n"
      "          \"coordinate_system\": {\"value\": \"EPSG:3857\"}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  arctern::render::weighted_point_map(x_array, y_array, c_array, s_array, vega);
}

TEST(POINTMAP_RAW_POINT_TEST, UINT16) {
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

  // param2: color
  arrow::UInt16Builder color_builder;
  status = color_builder.Append(1);
  status = color_builder.Append(2);
  status = color_builder.Append(3);
  status = color_builder.Append(4);
  status = color_builder.Append(5);

  std::shared_ptr<arrow::UInt16Array> c_array;
  status = color_builder.Finish(&c_array);

  // param3: point size
  arrow::UInt16Builder point_size_builder;
  status = point_size_builder.Append(2);
  status = point_size_builder.Append(4);
  status = point_size_builder.Append(6);
  status = point_size_builder.Append(8);
  status = point_size_builder.Append(10);

  std::shared_ptr<arrow::UInt16Array> s_array;
  status = point_size_builder.Finish(&s_array);

  // param4: conf
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
      "          \"color_gradient\": {\"value\": [\"#0000FF\", \"#FF0000\"]},\n"
      "          \"color_bound\": {\"value\": [2.5, 5]},\n"
      "          \"size_bound\": {\"value\": [2.5, 5]},\n"
      "          \"opacity\": {\"value\": 1.0},\n"
      "          \"coordinate_system\": {\"value\": \"EPSG:3857\"}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  arctern::render::weighted_point_map(x_array, y_array, c_array, s_array, vega);
}

TEST(POINTMAP_RAW_POINT_TEST, UINT32) {
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

  // param2: color
  arrow::UInt32Builder color_builder;
  status = color_builder.Append(1);
  status = color_builder.Append(2);
  status = color_builder.Append(3);
  status = color_builder.Append(4);
  status = color_builder.Append(5);

  std::shared_ptr<arrow::UInt32Array> c_array;
  status = color_builder.Finish(&c_array);

  // param3: point size
  arrow::UInt32Builder point_size_builder;
  status = point_size_builder.Append(2);
  status = point_size_builder.Append(4);
  status = point_size_builder.Append(6);
  status = point_size_builder.Append(8);
  status = point_size_builder.Append(10);

  std::shared_ptr<arrow::UInt32Array> s_array;
  status = point_size_builder.Finish(&s_array);

  // param4: conf
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
      "          \"color_gradient\": {\"value\": [\"#0000FF\", \"#FF0000\"]},\n"
      "          \"color_bound\": {\"value\": [2.5, 5]},\n"
      "          \"size_bound\": {\"value\": [2.5, 5]},\n"
      "          \"opacity\": {\"value\": 1.0},\n"
      "          \"coordinate_system\": {\"value\": \"EPSG:3857\"}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  arctern::render::weighted_point_map(x_array, y_array, c_array, s_array, vega);
}

TEST(POINTMAP_RAW_POINT_TEST, UINT64) {
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

  // param2: color
  arrow::UInt64Builder color_builder;
  status = color_builder.Append(1);
  status = color_builder.Append(2);
  status = color_builder.Append(3);
  status = color_builder.Append(4);
  status = color_builder.Append(5);

  std::shared_ptr<arrow::UInt64Array> c_array;
  status = color_builder.Finish(&c_array);

  // param3: point size
  arrow::UInt64Builder point_size_builder;
  status = point_size_builder.Append(2);
  status = point_size_builder.Append(4);
  status = point_size_builder.Append(6);
  status = point_size_builder.Append(8);
  status = point_size_builder.Append(10);

  std::shared_ptr<arrow::UInt64Array> s_array;
  status = point_size_builder.Finish(&s_array);

  // param4: conf
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
      "          \"color_gradient\": {\"value\": [\"#0000FF\", \"#FF0000\"]},\n"
      "          \"color_bound\": {\"value\": [2.5, 5]},\n"
      "          \"size_bound\": {\"value\": [2.5, 5]},\n"
      "          \"opacity\": {\"value\": 1.0},\n"
      "          \"coordinate_system\": {\"value\": \"EPSG:3857\"}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  arctern::render::weighted_point_map(x_array, y_array, c_array, s_array, vega);
}

TEST(POINTMAP_RAW_POINT_TEST, FLOAT) {
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

  // param2: color
  arrow::FloatBuilder color_builder;
  status = color_builder.Append(1);
  status = color_builder.Append(2);
  status = color_builder.Append(3);
  status = color_builder.Append(4);
  status = color_builder.Append(5);

  std::shared_ptr<arrow::FloatArray> c_array;
  status = color_builder.Finish(&c_array);

  // param3: point size
  arrow::FloatBuilder point_size_builder;
  status = point_size_builder.Append(2);
  status = point_size_builder.Append(4);
  status = point_size_builder.Append(6);
  status = point_size_builder.Append(8);
  status = point_size_builder.Append(10);

  std::shared_ptr<arrow::FloatArray> s_array;
  status = point_size_builder.Finish(&s_array);

  // param4: conf
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
      "          \"color_gradient\": {\"value\": [\"#0000FF\", \"#FF0000\"]},\n"
      "          \"color_bound\": {\"value\": [2.5, 5]},\n"
      "          \"size_bound\": {\"value\": [2.5, 5]},\n"
      "          \"opacity\": {\"value\": 1.0},\n"
      "          \"coordinate_system\": {\"value\": \"EPSG:3857\"}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  arctern::render::weighted_point_map(x_array, y_array, c_array, s_array, vega);
}

TEST(POINTMAP_RAW_POINT_TEST, DOUBLE) {
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

  // param2: color
  arrow::DoubleBuilder color_builder;
  status = color_builder.Append(1);
  status = color_builder.Append(2);
  status = color_builder.Append(3);
  status = color_builder.Append(4);
  status = color_builder.Append(5);

  std::shared_ptr<arrow::DoubleArray> c_array;
  status = color_builder.Finish(&c_array);

  // param3: point size
  arrow::DoubleBuilder point_size_builder;
  status = point_size_builder.Append(2);
  status = point_size_builder.Append(4);
  status = point_size_builder.Append(6);
  status = point_size_builder.Append(8);
  status = point_size_builder.Append(10);

  std::shared_ptr<arrow::DoubleArray> s_array;
  status = point_size_builder.Finish(&s_array);

  // param4: conf
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
      "          \"color_gradient\": {\"value\": [\"#0000FF\", \"#FF0000\"]},\n"
      "          \"color_bound\": {\"value\": [2.5, 5]},\n"
      "          \"size_bound\": {\"value\": [2.5, 5]},\n"
      "          \"opacity\": {\"value\": 1.0},\n"
      "          \"coordinate_system\": {\"value\": \"EPSG:3857\"}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  arctern::render::weighted_point_map(x_array, y_array, c_array, s_array, vega);
}

TEST(POINTMAP_RAW_POINT_TEST_MULTIPLE_COLOR, INT8) {
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

  // param2: color
  arrow::Int8Builder color_builder;
  status = color_builder.Append(1);
  status = color_builder.Append(2);
  status = color_builder.Append(3);
  status = color_builder.Append(4);
  status = color_builder.Append(5);

  std::shared_ptr<arrow::Int8Array> c_array;
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
      "          \"color_gradient\": {\"value\": [\"#0000FF\", \"#FF0000\"]},\n"
      "          \"color_bound\": {\"value\": [2.5, 5]},\n"
      "          \"size_bound\": {\"value\": [2.5]},\n"
      "          \"opacity\": {\"value\": 1.0},\n"
      "          \"coordinate_system\": {\"value\": \"EPSG:3857\"}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  arctern::render::weighted_point_map(x_array, y_array, c_array, vega);
}

TEST(POINTMAP_RAW_POINT_TEST_MULTIPLE_COLOR, INT16) {
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

  // param2: color
  arrow::Int16Builder color_builder;
  status = color_builder.Append(1);
  status = color_builder.Append(2);
  status = color_builder.Append(3);
  status = color_builder.Append(4);
  status = color_builder.Append(5);

  std::shared_ptr<arrow::Int16Array> c_array;
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
      "          \"color_gradient\": {\"value\": [\"#0000FF\", \"#FF0000\"]},\n"
      "          \"color_bound\": {\"value\": [2.5, 5]},\n"
      "          \"size_bound\": {\"value\": [2.5]},\n"
      "          \"opacity\": {\"value\": 1.0},\n"
      "          \"coordinate_system\": {\"value\": \"EPSG:3857\"}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  arctern::render::weighted_point_map(x_array, y_array, c_array, vega);
}

TEST(POINTMAP_RAW_POINT_TEST_MULTIPLE_COLOR, INT32) {
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

  // param2: color
  arrow::Int32Builder color_builder;
  status = color_builder.Append(1);
  status = color_builder.Append(2);
  status = color_builder.Append(3);
  status = color_builder.Append(4);
  status = color_builder.Append(5);

  std::shared_ptr<arrow::Int32Array> c_array;
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
      "          \"color_gradient\": {\"value\": [\"#0000FF\", \"#FF0000\"]},\n"
      "          \"color_bound\": {\"value\": [2.5, 5]},\n"
      "          \"size_bound\": {\"value\": [2.5]},\n"
      "          \"opacity\": {\"value\": 1.0},\n"
      "          \"coordinate_system\": {\"value\": \"EPSG:3857\"}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  arctern::render::weighted_point_map(x_array, y_array, c_array, vega);
}

TEST(POINTMAP_RAW_POINT_TEST_MULTIPLE_COLOR, INT64) {
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

  // param2: color
  arrow::Int64Builder color_builder;
  status = color_builder.Append(1);
  status = color_builder.Append(2);
  status = color_builder.Append(3);
  status = color_builder.Append(4);
  status = color_builder.Append(5);

  std::shared_ptr<arrow::Int64Array> c_array;
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
      "          \"color_gradient\": {\"value\": [\"#0000FF\", \"#FF0000\"]},\n"
      "          \"color_bound\": {\"value\": [2.5, 5]},\n"
      "          \"size_bound\": {\"value\": [2.5]},\n"
      "          \"opacity\": {\"value\": 1.0},\n"
      "          \"coordinate_system\": {\"value\": \"EPSG:3857\"}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  arctern::render::weighted_point_map(x_array, y_array, c_array, vega);
}

TEST(POINTMAP_RAW_POINT_TEST_MULTIPLE_COLOR, UINT8) {
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

  // param2: color
  arrow::UInt8Builder color_builder;
  status = color_builder.Append(1);
  status = color_builder.Append(2);
  status = color_builder.Append(3);
  status = color_builder.Append(4);
  status = color_builder.Append(5);

  std::shared_ptr<arrow::UInt8Array> c_array;
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
      "          \"color_gradient\": {\"value\": [\"#0000FF\", \"#FF0000\"]},\n"
      "          \"color_bound\": {\"value\": [2.5, 5]},\n"
      "          \"size_bound\": {\"value\": [2.5]},\n"
      "          \"opacity\": {\"value\": 1.0},\n"
      "          \"coordinate_system\": {\"value\": \"EPSG:3857\"}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  arctern::render::weighted_point_map(x_array, y_array, c_array, vega);
}

TEST(POINTMAP_RAW_POINT_TEST_MULTIPLE_COLOR, UINT16) {
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

  // param2: color
  arrow::UInt16Builder color_builder;
  status = color_builder.Append(1);
  status = color_builder.Append(2);
  status = color_builder.Append(3);
  status = color_builder.Append(4);
  status = color_builder.Append(5);

  std::shared_ptr<arrow::UInt16Array> c_array;
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
      "          \"color_gradient\": {\"value\": [\"#0000FF\", \"#FF0000\"]},\n"
      "          \"color_bound\": {\"value\": [2.5, 5]},\n"
      "          \"size_bound\": {\"value\": [2.5]},\n"
      "          \"opacity\": {\"value\": 1.0},\n"
      "          \"coordinate_system\": {\"value\": \"EPSG:3857\"}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  arctern::render::weighted_point_map(x_array, y_array, c_array, vega);
}

TEST(POINTMAP_RAW_POINT_TEST_MULTIPLE_COLOR, UINT32) {
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

  // param2: color
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
      "          \"color_gradient\": {\"value\": [\"#0000FF\", \"#FF0000\"]},\n"
      "          \"color_bound\": {\"value\": [2.5, 5]},\n"
      "          \"size_bound\": {\"value\": [2.5]},\n"
      "          \"opacity\": {\"value\": 1.0},\n"
      "          \"coordinate_system\": {\"value\": \"EPSG:3857\"}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  arctern::render::weighted_point_map(x_array, y_array, c_array, vega);
}

TEST(POINTMAP_RAW_POINT_TEST_MULTIPLE_COLOR, UINT64) {
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

  // param2: color
  arrow::UInt64Builder color_builder;
  status = color_builder.Append(1);
  status = color_builder.Append(2);
  status = color_builder.Append(3);
  status = color_builder.Append(4);
  status = color_builder.Append(5);

  std::shared_ptr<arrow::UInt64Array> c_array;
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
      "          \"color_gradient\": {\"value\": [\"#0000FF\", \"#FF0000\"]},\n"
      "          \"color_bound\": {\"value\": [2.5, 5]},\n"
      "          \"size_bound\": {\"value\": [2.5]},\n"
      "          \"opacity\": {\"value\": 1.0},\n"
      "          \"coordinate_system\": {\"value\": \"EPSG:3857\"}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  arctern::render::weighted_point_map(x_array, y_array, c_array, vega);
}

TEST(POINTMAP_RAW_POINT_TEST_MULTIPLE_COLOR, FLOAT) {
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

  // param2: color
  arrow::FloatBuilder color_builder;
  status = color_builder.Append(1);
  status = color_builder.Append(2);
  status = color_builder.Append(3);
  status = color_builder.Append(4);
  status = color_builder.Append(5);

  std::shared_ptr<arrow::FloatArray> c_array;
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
      "          \"color_gradient\": {\"value\": [\"#0000FF\", \"#FF0000\"]},\n"
      "          \"color_bound\": {\"value\": [2.5, 5]},\n"
      "          \"size_bound\": {\"value\": [2.5]},\n"
      "          \"opacity\": {\"value\": 1.0},\n"
      "          \"coordinate_system\": {\"value\": \"EPSG:3857\"}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  arctern::render::weighted_point_map(x_array, y_array, c_array, vega);
}

TEST(POINTMAP_RAW_POINT_TEST_MULTIPLE_COLOR, DOUBLE) {
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

  // param2: color
  arrow::DoubleBuilder color_builder;
  status = color_builder.Append(1);
  status = color_builder.Append(2);
  status = color_builder.Append(3);
  status = color_builder.Append(4);
  status = color_builder.Append(5);

  std::shared_ptr<arrow::DoubleArray> c_array;
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
      "          \"color_gradient\": {\"value\": [\"#0000FF\", \"#FF0000\"]},\n"
      "          \"color_bound\": {\"value\": [2.5, 5]},\n"
      "          \"size_bound\": {\"value\": [2.5]},\n"
      "          \"opacity\": {\"value\": 1.0},\n"
      "          \"coordinate_system\": {\"value\": \"EPSG:3857\"}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  arctern::render::weighted_point_map(x_array, y_array, c_array, vega);
}

TEST(POINTMAP_RAW_POINT_TEST, BLUE_TO_RED) {
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

  // param2: color
  arrow::DoubleBuilder color_builder;
  status = color_builder.Append(1);
  status = color_builder.Append(2);
  status = color_builder.Append(3);
  status = color_builder.Append(4);
  status = color_builder.Append(5);

  std::shared_ptr<arrow::DoubleArray> c_array;
  status = color_builder.Finish(&c_array);

  // param3: point size
  arrow::DoubleBuilder point_size_builder;
  status = point_size_builder.Append(2);
  status = point_size_builder.Append(4);
  status = point_size_builder.Append(6);
  status = point_size_builder.Append(8);
  status = point_size_builder.Append(10);

  std::shared_ptr<arrow::DoubleArray> s_array;
  status = point_size_builder.Finish(&s_array);

  // param4: conf
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
      "          \"color_gradient\": {\"value\": [\"#0000FF\", \"#FF0000\"]},\n"
      "          \"color_bound\": {\"value\": [2.5, 5]},\n"
      "          \"size_bound\": {\"value\": [2.5, 5]},\n"
      "          \"opacity\": {\"value\": 1.0},\n"
      "          \"coordinate_system\": {\"value\": \"EPSG:3857\"}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  arctern::render::weighted_point_map(x_array, y_array, c_array, s_array, vega);
}

TEST(POINTMAP_RAW_POINT_TEST, SKYBLUE_TO_RED) {
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

  // param2: color
  arrow::DoubleBuilder color_builder;
  status = color_builder.Append(1);
  status = color_builder.Append(2);
  status = color_builder.Append(3);
  status = color_builder.Append(4);
  status = color_builder.Append(5);

  std::shared_ptr<arrow::DoubleArray> c_array;
  status = color_builder.Finish(&c_array);

  // param3: point size
  arrow::DoubleBuilder point_size_builder;
  status = point_size_builder.Append(2);
  status = point_size_builder.Append(4);
  status = point_size_builder.Append(6);
  status = point_size_builder.Append(8);
  status = point_size_builder.Append(10);

  std::shared_ptr<arrow::DoubleArray> s_array;
  status = point_size_builder.Finish(&s_array);

  // param4: conf
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
      "          \"color_gradient\": {\"value\": [\"#B4E7F5\", \"#FFFFFF\"]},\n"
      "          \"color_bound\": {\"value\": [2.5, 5]},\n"
      "          \"size_bound\": {\"value\": [2.5, 5]},\n"
      "          \"opacity\": {\"value\": 1.0},\n"
      "          \"coordinate_system\": {\"value\": \"EPSG:3857\"}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  arctern::render::weighted_point_map(x_array, y_array, c_array, s_array, vega);
}

TEST(POINTMAP_RAW_POINT_TEST, PURPLE_TO_YELLOW) {
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

  // param2: color
  arrow::DoubleBuilder color_builder;
  status = color_builder.Append(1);
  status = color_builder.Append(2);
  status = color_builder.Append(3);
  status = color_builder.Append(4);
  status = color_builder.Append(5);

  std::shared_ptr<arrow::DoubleArray> c_array;
  status = color_builder.Finish(&c_array);

  // param3: point size
  arrow::DoubleBuilder point_size_builder;
  status = point_size_builder.Append(2);
  status = point_size_builder.Append(4);
  status = point_size_builder.Append(6);
  status = point_size_builder.Append(8);
  status = point_size_builder.Append(10);

  std::shared_ptr<arrow::DoubleArray> s_array;
  status = point_size_builder.Finish(&s_array);

  // param4: conf
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
      "          \"color_gradient\": {\"value\": [\"#FF00FF\", \"#FFFF00\"]},\n"
      "          \"color_bound\": {\"value\": [2.5, 5]},\n"
      "          \"size_bound\": {\"value\": [2.5, 5]},\n"
      "          \"opacity\": {\"value\": 1.0},\n"
      "          \"coordinate_system\": {\"value\": \"EPSG:3857\"}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  arctern::render::weighted_point_map(x_array, y_array, c_array, s_array, vega);
}

TEST(POINTMAP_RAW_POINT_TEST, RED_TRANSPARENCY) {
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

  // param2: color
  arrow::DoubleBuilder color_builder;
  status = color_builder.Append(1);
  status = color_builder.Append(2);
  status = color_builder.Append(3);
  status = color_builder.Append(4);
  status = color_builder.Append(5);

  std::shared_ptr<arrow::DoubleArray> c_array;
  status = color_builder.Finish(&c_array);

  // param3: point size
  arrow::DoubleBuilder point_size_builder;
  status = point_size_builder.Append(2);
  status = point_size_builder.Append(4);
  status = point_size_builder.Append(6);
  status = point_size_builder.Append(8);
  status = point_size_builder.Append(10);

  std::shared_ptr<arrow::DoubleArray> s_array;
  status = point_size_builder.Finish(&s_array);

  // param4: conf
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
      "          \"color_gradient\": {\"value\": [\"#FF0000\", \"#FF0000\"]},\n"
      "          \"color_bound\": {\"value\": [2.5, 5]},\n"
      "          \"size_bound\": {\"value\": [2.5, 5]},\n"
      "          \"opacity\": {\"value\": 1.0},\n"
      "          \"coordinate_system\": {\"value\": \"EPSG:3857\"}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  arctern::render::weighted_point_map(x_array, y_array, c_array, s_array, vega);
}

TEST(POINTMAP_RAW_POINT_TEST, BLUE_TRANSPARENCY) {
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

  // param2: color
  arrow::DoubleBuilder color_builder;
  status = color_builder.Append(1);
  status = color_builder.Append(2);
  status = color_builder.Append(3);
  status = color_builder.Append(4);
  status = color_builder.Append(5);

  std::shared_ptr<arrow::DoubleArray> c_array;
  status = color_builder.Finish(&c_array);

  // param3: point size
  arrow::DoubleBuilder point_size_builder;
  status = point_size_builder.Append(2);
  status = point_size_builder.Append(4);
  status = point_size_builder.Append(6);
  status = point_size_builder.Append(8);
  status = point_size_builder.Append(10);

  std::shared_ptr<arrow::DoubleArray> s_array;
  status = point_size_builder.Finish(&s_array);

  // param4: conf
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
      "          \"color_gradient\": {\"value\": [\"#0000FF\", \"#0000FF\"]},\n"
      "          \"color_bound\": {\"value\": [2.5, 5]},\n"
      "          \"size_bound\": {\"value\": [2.5, 5]},\n"
      "          \"opacity\": {\"value\": 1.0},\n"
      "          \"coordinate_system\": {\"value\": \"EPSG:3857\"}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  arctern::render::weighted_point_map(x_array, y_array, c_array, s_array, vega);
}

TEST(POINTMAP_RAW_POINT_TEST, BLUE_GREEN_YELLOW) {
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

  // param2: color
  arrow::DoubleBuilder color_builder;
  status = color_builder.Append(1);
  status = color_builder.Append(2);
  status = color_builder.Append(3);
  status = color_builder.Append(4);
  status = color_builder.Append(5);

  std::shared_ptr<arrow::DoubleArray> c_array;
  status = color_builder.Finish(&c_array);

  // param3: point size
  arrow::DoubleBuilder point_size_builder;
  status = point_size_builder.Append(2);
  status = point_size_builder.Append(4);
  status = point_size_builder.Append(6);
  status = point_size_builder.Append(8);
  status = point_size_builder.Append(10);

  std::shared_ptr<arrow::DoubleArray> s_array;
  status = point_size_builder.Finish(&s_array);

  // param4: conf
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
      "          \"color_gradient\": {\"value\": [\"#115F9A\", \"#D0F401\"]},\n"
      "          \"color_bound\": {\"value\": [2.5, 5]},\n"
      "          \"size_bound\": {\"value\": [2.5, 5]},\n"
      "          \"opacity\": {\"value\": 1.0},\n"
      "          \"coordinate_system\": {\"value\": \"EPSG:3857\"}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  arctern::render::weighted_point_map(x_array, y_array, c_array, s_array, vega);
}

TEST(POINTMAP_RAW_POINT_TEST, WHITE_BLUE) {
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

  // param2: color
  arrow::DoubleBuilder color_builder;
  status = color_builder.Append(1);
  status = color_builder.Append(2);
  status = color_builder.Append(3);
  status = color_builder.Append(4);
  status = color_builder.Append(5);

  std::shared_ptr<arrow::DoubleArray> c_array;
  status = color_builder.Finish(&c_array);

  // param3: point size
  arrow::DoubleBuilder point_size_builder;
  status = point_size_builder.Append(2);
  status = point_size_builder.Append(4);
  status = point_size_builder.Append(6);
  status = point_size_builder.Append(8);
  status = point_size_builder.Append(10);

  std::shared_ptr<arrow::DoubleArray> s_array;
  status = point_size_builder.Finish(&s_array);

  // param4: conf
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
      "          \"color_gradient\": {\"value\": [\"#E2E2E2\", \"#115F9A\"]},\n"
      "          \"color_bound\": {\"value\": [2.5, 5]},\n"
      "          \"size_bound\": {\"value\": [2.5, 5]},\n"
      "          \"opacity\": {\"value\": 1.0},\n"
      "          \"coordinate_system\": {\"value\": \"EPSG:3857\"}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  arctern::render::weighted_point_map(x_array, y_array, c_array, s_array, vega);
}

TEST(POINTMAP_RAW_POINT_TEST, BLUE_WHITE_RED) {
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

  // param2: color
  arrow::DoubleBuilder color_builder;
  status = color_builder.Append(1);
  status = color_builder.Append(2);
  status = color_builder.Append(3);
  status = color_builder.Append(4);
  status = color_builder.Append(5);

  std::shared_ptr<arrow::DoubleArray> c_array;
  status = color_builder.Finish(&c_array);

  // param3: point size
  arrow::DoubleBuilder point_size_builder;
  status = point_size_builder.Append(2);
  status = point_size_builder.Append(4);
  status = point_size_builder.Append(6);
  status = point_size_builder.Append(8);
  status = point_size_builder.Append(10);

  std::shared_ptr<arrow::DoubleArray> s_array;
  status = point_size_builder.Finish(&s_array);

  // param4: conf
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
      "          \"color_gradient\": {\"value\": [\"#1984C5\", \"#C23728\"]},\n"
      "          \"color_bound\": {\"value\": [2.5, 5]},\n"
      "          \"size_bound\": {\"value\": [2.5, 5]},\n"
      "          \"opacity\": {\"value\": 1.0},\n"
      "          \"coordinate_system\": {\"value\": \"EPSG:3857\"}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  arctern::render::weighted_point_map(x_array, y_array, c_array, s_array, vega);
}

TEST(POINTMAP_RAW_POINT_TEST, GREEN_YELLOW_RED) {
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

  // param2: color
  arrow::DoubleBuilder color_builder;
  status = color_builder.Append(1);
  status = color_builder.Append(2);
  status = color_builder.Append(3);
  status = color_builder.Append(4);
  status = color_builder.Append(5);

  std::shared_ptr<arrow::DoubleArray> c_array;
  status = color_builder.Finish(&c_array);

  // param3: point size
  arrow::DoubleBuilder point_size_builder;
  status = point_size_builder.Append(2);
  status = point_size_builder.Append(4);
  status = point_size_builder.Append(6);
  status = point_size_builder.Append(8);
  status = point_size_builder.Append(10);

  std::shared_ptr<arrow::DoubleArray> s_array;
  status = point_size_builder.Finish(&s_array);

  // param4: conf
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
      "          \"color_gradient\": {\"value\": [\"#4D904F\", \"#C23728\"]},\n"
      "          \"color_bound\": {\"value\": [2.5, 5]},\n"
      "          \"size_bound\": {\"value\": [2.5, 5]},\n"
      "          \"opacity\": {\"value\": 1.0},\n"
      "          \"coordinate_system\": {\"value\": \"EPSG:3857\"}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  arctern::render::weighted_point_map(x_array, y_array, c_array, s_array, vega);
}

TEST(POINTMAP_WKT_TEST, SINGLE_COLOR_SINGLE_POINTSIZE) {
  // param1: wkt string
  std::string wkt1 = "POINT (10 10)";
  std::string wkt2 = "POINT (30 30)";
  std::string wkt3 = "POINT (50 50)";
  std::string wkt4 = "POINT (70 70)";
  std::string wkt5 = "POINT (90 90)";
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
      "  \"description\": \"pointmap\",\n"
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
      "          \"color_gradient\": {\"value\": [\"#FFD700\"]},\n"
      "          \"color_bound\": {\"value\": [-1, -1]},\n"
      "          \"size_bound\": {\"value\": [5]},\n"
      "          \"opacity\": {\"value\": 1.0},\n"
      "          \"coordinate_system\": {\"value\": \"EPSG:3857\"},\n"
      "          \"aggregation_type\": {\"value\": \"sum\"}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  auto wkb = arctern::render::WktToWkb(string_array);
  arctern::render::weighted_point_map(wkb, vega);
}

TEST(POINTMAP_WKT_TEST, MULTIPLE_COLOR_SINGLE_POINTSIZE) {
  // param1: wkt string
  std::string wkt1 = "POINT (10 10)";
  std::string wkt2 = "POINT (30 30)";
  std::string wkt3 = "POINT (50 50)";
  std::string wkt4 = "POINT (70 70)";
  std::string wkt5 = "POINT (90 90)";
  arrow::StringBuilder string_builder;
  auto status = string_builder.Append(wkt1);
  status = string_builder.Append(wkt2);
  status = string_builder.Append(wkt3);
  status = string_builder.Append(wkt4);
  status = string_builder.Append(wkt5);

  std::shared_ptr<arrow::StringArray> string_array;
  status = string_builder.Finish(&string_array);

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
      "  \"description\": \"pointmap\",\n"
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
      "          \"color_gradient\": {\"value\": [\"#0000FF\", \"#FF0000\"]},\n"
      "          \"color_bound\": {\"value\": [2, 5]},\n"
      "          \"size_bound\": {\"value\": [8]},\n"
      "          \"opacity\": {\"value\": 1.0},\n"
      "          \"coordinate_system\": {\"value\": \"EPSG:3857\"},\n"
      "          \"aggregation_type\": {\"value\": \"sum\"}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  auto wkb = arctern::render::WktToWkb(string_array);
  arctern::render::weighted_point_map(wkb, c_array, vega);
}

TEST(POINTMAP_WKT_TEST, SINGLE_COLOR_MULTIPLE_POINTSIZE) {
  // param1: wkt string
  std::string wkt1 = "POINT (10 10)";
  std::string wkt2 = "POINT (30 30)";
  std::string wkt3 = "POINT (50 50)";
  std::string wkt4 = "POINT (70 70)";
  std::string wkt5 = "POINT (90 90)";
  arrow::StringBuilder string_builder;
  auto status = string_builder.Append(wkt1);
  status = string_builder.Append(wkt2);
  status = string_builder.Append(wkt3);
  status = string_builder.Append(wkt4);
  status = string_builder.Append(wkt5);

  std::shared_ptr<arrow::StringArray> string_array;
  status = string_builder.Finish(&string_array);

  // param2: point_size
  arrow::UInt32Builder ps_builder;
  status = ps_builder.Append(2);
  status = ps_builder.Append(4);
  status = ps_builder.Append(6);
  status = ps_builder.Append(8);
  status = ps_builder.Append(10);

  std::shared_ptr<arrow::UInt32Array> s_array;
  status = ps_builder.Finish(&s_array);

  // param3: conf
  const std::string vega =
      "{\n"
      "  \"width\": 300,\n"
      "  \"height\": 200,\n"
      "  \"description\": \"pointmap\",\n"
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
      "          \"color_gradient\": {\"value\": [\"#FFD700\"]},\n"
      "          \"color_bound\": {\"value\": [-1, -1]},\n"
      "          \"size_bound\": {\"value\": [0, 10]},\n"
      "          \"opacity\": {\"value\": 1.0},\n"
      "          \"coordinate_system\": {\"value\": \"EPSG:3857\"},\n"
      "          \"aggregation_type\": {\"value\": \"sum\"}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  auto wkb = arctern::render::WktToWkb(string_array);
  arctern::render::weighted_point_map(wkb, s_array, vega);
}

TEST(POINTMAP_WKT_TEST, MULTIPLE_COLOR_MULTIPLE_POINTSIZE) {
  // param1: wkt string
  std::string wkt1 = "POINT (10 10)";
  std::string wkt2 = "POINT (30 30)";
  std::string wkt3 = "POINT (50 50)";
  std::string wkt4 = "POINT (70 70)";
  std::string wkt5 = "POINT (90 90)";
  arrow::StringBuilder string_builder;
  auto status = string_builder.Append(wkt1);
  status = string_builder.Append(wkt2);
  status = string_builder.Append(wkt3);
  status = string_builder.Append(wkt4);
  status = string_builder.Append(wkt5);

  std::shared_ptr<arrow::StringArray> string_array;
  status = string_builder.Finish(&string_array);

  // param2: color
  arrow::UInt32Builder color_builder;
  status = color_builder.Append(1);
  status = color_builder.Append(2);
  status = color_builder.Append(3);
  status = color_builder.Append(4);
  status = color_builder.Append(5);

  std::shared_ptr<arrow::UInt32Array> c_array;
  status = color_builder.Finish(&c_array);

  // param3: point size
  arrow::UInt32Builder point_size_builder;
  status = point_size_builder.Append(2);
  status = point_size_builder.Append(4);
  status = point_size_builder.Append(6);
  status = point_size_builder.Append(8);
  status = point_size_builder.Append(10);

  std::shared_ptr<arrow::UInt32Array> s_array;
  status = point_size_builder.Finish(&s_array);

  // param4: conf
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
      "          \"color_gradient\": {\"value\": [\"#0000FF\", \"#FF0000\"]},\n"
      "          \"color_bound\": {\"value\": [2.5, 5]},\n"
      "          \"size_bound\": {\"value\": [2.5, 5]},\n"
      "          \"opacity\": {\"value\": 1.0},\n"
      "          \"coordinate_system\": {\"value\": \"EPSG:3857\"},\n"
      "          \"aggregation_type\": {\"value\": \"sum\"}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  auto wkb = arctern::render::WktToWkb(string_array);
  arctern::render::weighted_point_map(wkb, c_array, s_array, vega);
}

TEST(POINTMAP_WKT_TEST, INT8) {
  // param1: wkt string
  std::string wkt1 = "POINT (10 10)";
  std::string wkt2 = "POINT (30 30)";
  std::string wkt3 = "POINT (50 50)";
  std::string wkt4 = "POINT (70 70)";
  std::string wkt5 = "POINT (90 90)";
  arrow::StringBuilder string_builder;
  auto status = string_builder.Append(wkt1);
  status = string_builder.Append(wkt2);
  status = string_builder.Append(wkt3);
  status = string_builder.Append(wkt4);
  status = string_builder.Append(wkt5);

  std::shared_ptr<arrow::StringArray> string_array;
  status = string_builder.Finish(&string_array);

  // param2: color
  arrow::Int8Builder color_builder;
  status = color_builder.Append(1);
  status = color_builder.Append(2);
  status = color_builder.Append(3);
  status = color_builder.Append(4);
  status = color_builder.Append(5);

  std::shared_ptr<arrow::Int8Array> c_array;
  status = color_builder.Finish(&c_array);

  // param3: point size
  arrow::Int8Builder point_size_builder;
  status = point_size_builder.Append(2);
  status = point_size_builder.Append(4);
  status = point_size_builder.Append(6);
  status = point_size_builder.Append(8);
  status = point_size_builder.Append(10);

  std::shared_ptr<arrow::Int8Array> s_array;
  status = point_size_builder.Finish(&s_array);

  // param4: conf
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
      "          \"color_gradient\": {\"value\": [\"#0000FF\", \"#FF0000\"]},\n"
      "          \"color_bound\": {\"value\": [2.5, 5]},\n"
      "          \"size_bound\": {\"value\": [2.5, 5]},\n"
      "          \"opacity\": {\"value\": 1.0},\n"
      "          \"coordinate_system\": {\"value\": \"EPSG:3857\"},\n"
      "          \"aggregation_type\": {\"value\": \"sum\"}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  auto wkb = arctern::render::WktToWkb(string_array);
  arctern::render::weighted_point_map(wkb, c_array, s_array, vega);
}

TEST(POINTMAP_WKT_TEST, INT16) {
  // param1: wkt string
  std::string wkt1 = "POINT (10 10)";
  std::string wkt2 = "POINT (30 30)";
  std::string wkt3 = "POINT (50 50)";
  std::string wkt4 = "POINT (70 70)";
  std::string wkt5 = "POINT (90 90)";
  arrow::StringBuilder string_builder;
  auto status = string_builder.Append(wkt1);
  status = string_builder.Append(wkt2);
  status = string_builder.Append(wkt3);
  status = string_builder.Append(wkt4);
  status = string_builder.Append(wkt5);

  std::shared_ptr<arrow::StringArray> string_array;
  status = string_builder.Finish(&string_array);

  // param2: color
  arrow::Int16Builder color_builder;
  status = color_builder.Append(1);
  status = color_builder.Append(2);
  status = color_builder.Append(3);
  status = color_builder.Append(4);
  status = color_builder.Append(5);

  std::shared_ptr<arrow::Int16Array> c_array;
  status = color_builder.Finish(&c_array);

  // param3: point size
  arrow::Int16Builder point_size_builder;
  status = point_size_builder.Append(2);
  status = point_size_builder.Append(4);
  status = point_size_builder.Append(6);
  status = point_size_builder.Append(8);
  status = point_size_builder.Append(10);

  std::shared_ptr<arrow::Int16Array> s_array;
  status = point_size_builder.Finish(&s_array);

  // param4: conf
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
      "          \"color_gradient\": {\"value\": [\"#0000FF\", \"#FF0000\"]},\n"
      "          \"color_bound\": {\"value\": [2.5, 5]},\n"
      "          \"size_bound\": {\"value\": [2.5, 5]},\n"
      "          \"opacity\": {\"value\": 1.0},\n"
      "          \"coordinate_system\": {\"value\": \"EPSG:3857\"},\n"
      "          \"aggregation_type\": {\"value\": \"sum\"}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  auto wkb = arctern::render::WktToWkb(string_array);
  arctern::render::weighted_point_map(wkb, c_array, s_array, vega);
}

TEST(POINTMAP_WKT_TEST, INT32) {
  // param1: wkt string
  std::string wkt1 = "POINT (10 10)";
  std::string wkt2 = "POINT (30 30)";
  std::string wkt3 = "POINT (50 50)";
  std::string wkt4 = "POINT (70 70)";
  std::string wkt5 = "POINT (90 90)";
  arrow::StringBuilder string_builder;
  auto status = string_builder.Append(wkt1);
  status = string_builder.Append(wkt2);
  status = string_builder.Append(wkt3);
  status = string_builder.Append(wkt4);
  status = string_builder.Append(wkt5);

  std::shared_ptr<arrow::StringArray> string_array;
  status = string_builder.Finish(&string_array);

  // param2: color
  arrow::Int32Builder color_builder;
  status = color_builder.Append(1);
  status = color_builder.Append(2);
  status = color_builder.Append(3);
  status = color_builder.Append(4);
  status = color_builder.Append(5);

  std::shared_ptr<arrow::Int32Array> c_array;
  status = color_builder.Finish(&c_array);

  // param3: point size
  arrow::Int32Builder point_size_builder;
  status = point_size_builder.Append(2);
  status = point_size_builder.Append(4);
  status = point_size_builder.Append(6);
  status = point_size_builder.Append(8);
  status = point_size_builder.Append(10);

  std::shared_ptr<arrow::Int32Array> s_array;
  status = point_size_builder.Finish(&s_array);

  // param4: conf
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
      "          \"color_gradient\": {\"value\": [\"#0000FF\", \"#FF0000\"]},\n"
      "          \"color_bound\": {\"value\": [2.5, 5]},\n"
      "          \"size_bound\": {\"value\": [2.5, 5]},\n"
      "          \"opacity\": {\"value\": 1.0},\n"
      "          \"coordinate_system\": {\"value\": \"EPSG:3857\"},\n"
      "          \"aggregation_type\": {\"value\": \"sum\"}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  auto wkb = arctern::render::WktToWkb(string_array);
  arctern::render::weighted_point_map(wkb, c_array, s_array, vega);
}

TEST(POINTMAP_WKT_TEST, INT64) {
  // param1: wkt string
  std::string wkt1 = "POINT (10 10)";
  std::string wkt2 = "POINT (30 30)";
  std::string wkt3 = "POINT (50 50)";
  std::string wkt4 = "POINT (70 70)";
  std::string wkt5 = "POINT (90 90)";
  arrow::StringBuilder string_builder;
  auto status = string_builder.Append(wkt1);
  status = string_builder.Append(wkt2);
  status = string_builder.Append(wkt3);
  status = string_builder.Append(wkt4);
  status = string_builder.Append(wkt5);

  std::shared_ptr<arrow::StringArray> string_array;
  status = string_builder.Finish(&string_array);

  // param2: color
  arrow::Int64Builder color_builder;
  status = color_builder.Append(1);
  status = color_builder.Append(2);
  status = color_builder.Append(3);
  status = color_builder.Append(4);
  status = color_builder.Append(5);

  std::shared_ptr<arrow::Int64Array> c_array;
  status = color_builder.Finish(&c_array);

  // param3: point size
  arrow::Int64Builder point_size_builder;
  status = point_size_builder.Append(2);
  status = point_size_builder.Append(4);
  status = point_size_builder.Append(6);
  status = point_size_builder.Append(8);
  status = point_size_builder.Append(10);

  std::shared_ptr<arrow::Int64Array> s_array;
  status = point_size_builder.Finish(&s_array);

  // param4: conf
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
      "          \"color_gradient\": {\"value\": [\"#0000FF\", \"#FF0000\"]},\n"
      "          \"color_bound\": {\"value\": [2.5, 5]},\n"
      "          \"size_bound\": {\"value\": [2.5, 5]},\n"
      "          \"opacity\": {\"value\": 1.0},\n"
      "          \"coordinate_system\": {\"value\": \"EPSG:3857\"},\n"
      "          \"aggregation_type\": {\"value\": \"sum\"}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  auto wkb = arctern::render::WktToWkb(string_array);
  arctern::render::weighted_point_map(wkb, c_array, s_array, vega);
}

TEST(POINTMAP_WKT_TEST, UINT8) {
  // param1: wkt string
  std::string wkt1 = "POINT (10 10)";
  std::string wkt2 = "POINT (30 30)";
  std::string wkt3 = "POINT (50 50)";
  std::string wkt4 = "POINT (70 70)";
  std::string wkt5 = "POINT (90 90)";
  arrow::StringBuilder string_builder;
  auto status = string_builder.Append(wkt1);
  status = string_builder.Append(wkt2);
  status = string_builder.Append(wkt3);
  status = string_builder.Append(wkt4);
  status = string_builder.Append(wkt5);

  std::shared_ptr<arrow::StringArray> string_array;
  status = string_builder.Finish(&string_array);

  // param2: color
  arrow::UInt8Builder color_builder;
  status = color_builder.Append(1);
  status = color_builder.Append(2);
  status = color_builder.Append(3);
  status = color_builder.Append(4);
  status = color_builder.Append(5);

  std::shared_ptr<arrow::UInt8Array> c_array;
  status = color_builder.Finish(&c_array);

  // param3: point size
  arrow::UInt8Builder point_size_builder;
  status = point_size_builder.Append(2);
  status = point_size_builder.Append(4);
  status = point_size_builder.Append(6);
  status = point_size_builder.Append(8);
  status = point_size_builder.Append(10);

  std::shared_ptr<arrow::UInt8Array> s_array;
  status = point_size_builder.Finish(&s_array);

  // param4: conf
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
      "          \"color_gradient\": {\"value\": [\"#0000FF\", \"#FF0000\"]},\n"
      "          \"color_bound\": {\"value\": [2.5, 5]},\n"
      "          \"size_bound\": {\"value\": [2.5, 5]},\n"
      "          \"opacity\": {\"value\": 1.0},\n"
      "          \"coordinate_system\": {\"value\": \"EPSG:3857\"},\n"
      "          \"aggregation_type\": {\"value\": \"sum\"}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  auto wkb = arctern::render::WktToWkb(string_array);
  arctern::render::weighted_point_map(wkb, c_array, s_array, vega);
}

TEST(POINTMAP_WKT_TEST, UINT16) {
  // param1: wkt string
  std::string wkt1 = "POINT (10 10)";
  std::string wkt2 = "POINT (30 30)";
  std::string wkt3 = "POINT (50 50)";
  std::string wkt4 = "POINT (70 70)";
  std::string wkt5 = "POINT (90 90)";
  arrow::StringBuilder string_builder;
  auto status = string_builder.Append(wkt1);
  status = string_builder.Append(wkt2);
  status = string_builder.Append(wkt3);
  status = string_builder.Append(wkt4);
  status = string_builder.Append(wkt5);

  std::shared_ptr<arrow::StringArray> string_array;
  status = string_builder.Finish(&string_array);

  // param2: color
  arrow::UInt16Builder color_builder;
  status = color_builder.Append(1);
  status = color_builder.Append(2);
  status = color_builder.Append(3);
  status = color_builder.Append(4);
  status = color_builder.Append(5);

  std::shared_ptr<arrow::UInt16Array> c_array;
  status = color_builder.Finish(&c_array);

  // param3: point size
  arrow::UInt16Builder point_size_builder;
  status = point_size_builder.Append(2);
  status = point_size_builder.Append(4);
  status = point_size_builder.Append(6);
  status = point_size_builder.Append(8);
  status = point_size_builder.Append(10);

  std::shared_ptr<arrow::UInt16Array> s_array;
  status = point_size_builder.Finish(&s_array);

  // param4: conf
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
      "          \"color_gradient\": {\"value\": [\"#0000FF\", \"#FF0000\"]},\n"
      "          \"color_bound\": {\"value\": [2.5, 5]},\n"
      "          \"size_bound\": {\"value\": [2.5, 5]},\n"
      "          \"opacity\": {\"value\": 1.0},\n"
      "          \"coordinate_system\": {\"value\": \"EPSG:3857\"},\n"
      "          \"aggregation_type\": {\"value\": \"sum\"}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  auto wkb = arctern::render::WktToWkb(string_array);
  arctern::render::weighted_point_map(wkb, c_array, s_array, vega);
}

TEST(POINTMAP_WKT_TEST, UINT32) {
  // param1: wkt string
  std::string wkt1 = "POINT (10 10)";
  std::string wkt2 = "POINT (30 30)";
  std::string wkt3 = "POINT (50 50)";
  std::string wkt4 = "POINT (70 70)";
  std::string wkt5 = "POINT (90 90)";
  arrow::StringBuilder string_builder;
  auto status = string_builder.Append(wkt1);
  status = string_builder.Append(wkt2);
  status = string_builder.Append(wkt3);
  status = string_builder.Append(wkt4);
  status = string_builder.Append(wkt5);

  std::shared_ptr<arrow::StringArray> string_array;
  status = string_builder.Finish(&string_array);

  // param2: color
  arrow::UInt32Builder color_builder;
  status = color_builder.Append(1);
  status = color_builder.Append(2);
  status = color_builder.Append(3);
  status = color_builder.Append(4);
  status = color_builder.Append(5);

  std::shared_ptr<arrow::UInt32Array> c_array;
  status = color_builder.Finish(&c_array);

  // param3: point size
  arrow::UInt32Builder point_size_builder;
  status = point_size_builder.Append(2);
  status = point_size_builder.Append(4);
  status = point_size_builder.Append(6);
  status = point_size_builder.Append(8);
  status = point_size_builder.Append(10);

  std::shared_ptr<arrow::UInt32Array> s_array;
  status = point_size_builder.Finish(&s_array);

  // param4: conf
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
      "          \"color_gradient\": {\"value\": [\"#0000FF\", \"#FF0000\"]},\n"
      "          \"color_bound\": {\"value\": [2.5, 5]},\n"
      "          \"size_bound\": {\"value\": [2.5, 5]},\n"
      "          \"opacity\": {\"value\": 1.0},\n"
      "          \"coordinate_system\": {\"value\": \"EPSG:3857\"},\n"
      "          \"aggregation_type\": {\"value\": \"sum\"}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  auto wkb = arctern::render::WktToWkb(string_array);
  arctern::render::weighted_point_map(wkb, c_array, s_array, vega);
}

TEST(POINTMAP_WKT_TEST, UINT64) {
  // param1: wkt string
  std::string wkt1 = "POINT (10 10)";
  std::string wkt2 = "POINT (30 30)";
  std::string wkt3 = "POINT (50 50)";
  std::string wkt4 = "POINT (70 70)";
  std::string wkt5 = "POINT (90 90)";
  arrow::StringBuilder string_builder;
  auto status = string_builder.Append(wkt1);
  status = string_builder.Append(wkt2);
  status = string_builder.Append(wkt3);
  status = string_builder.Append(wkt4);
  status = string_builder.Append(wkt5);

  std::shared_ptr<arrow::StringArray> string_array;
  status = string_builder.Finish(&string_array);

  // param2: color
  arrow::UInt64Builder color_builder;
  status = color_builder.Append(1);
  status = color_builder.Append(2);
  status = color_builder.Append(3);
  status = color_builder.Append(4);
  status = color_builder.Append(5);

  std::shared_ptr<arrow::UInt64Array> c_array;
  status = color_builder.Finish(&c_array);

  // param3: point size
  arrow::UInt64Builder point_size_builder;
  status = point_size_builder.Append(2);
  status = point_size_builder.Append(4);
  status = point_size_builder.Append(6);
  status = point_size_builder.Append(8);
  status = point_size_builder.Append(10);

  std::shared_ptr<arrow::UInt64Array> s_array;
  status = point_size_builder.Finish(&s_array);

  // param4: conf
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
      "          \"color_gradient\": {\"value\": [\"#0000FF\", \"#FF0000\"]},\n"
      "          \"color_bound\": {\"value\": [2.5, 5]},\n"
      "          \"size_bound\": {\"value\": [2.5, 5]},\n"
      "          \"opacity\": {\"value\": 1.0},\n"
      "          \"coordinate_system\": {\"value\": \"EPSG:3857\"},\n"
      "          \"aggregation_type\": {\"value\": \"sum\"}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  auto wkb = arctern::render::WktToWkb(string_array);
  arctern::render::weighted_point_map(wkb, c_array, s_array, vega);
}

TEST(POINTMAP_WKT_TEST, FLOAT) {
  // param1: wkt string
  std::string wkt1 = "POINT (10 10)";
  std::string wkt2 = "POINT (30 30)";
  std::string wkt3 = "POINT (50 50)";
  std::string wkt4 = "POINT (70 70)";
  std::string wkt5 = "POINT (90 90)";
  arrow::StringBuilder string_builder;
  auto status = string_builder.Append(wkt1);
  status = string_builder.Append(wkt2);
  status = string_builder.Append(wkt3);
  status = string_builder.Append(wkt4);
  status = string_builder.Append(wkt5);

  std::shared_ptr<arrow::StringArray> string_array;
  status = string_builder.Finish(&string_array);

  // param2: color
  arrow::FloatBuilder color_builder;
  status = color_builder.Append(1);
  status = color_builder.Append(2);
  status = color_builder.Append(3);
  status = color_builder.Append(4);
  status = color_builder.Append(5);

  std::shared_ptr<arrow::FloatArray> c_array;
  status = color_builder.Finish(&c_array);

  // param3: point size
  arrow::FloatBuilder point_size_builder;
  status = point_size_builder.Append(2);
  status = point_size_builder.Append(4);
  status = point_size_builder.Append(6);
  status = point_size_builder.Append(8);
  status = point_size_builder.Append(10);

  std::shared_ptr<arrow::FloatArray> s_array;
  status = point_size_builder.Finish(&s_array);

  // param4: conf
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
      "          \"color_gradient\": {\"value\": [\"#0000FF\", \"#FF0000\"]},\n"
      "          \"color_bound\": {\"value\": [2.5, 5]},\n"
      "          \"size_bound\": {\"value\": [2.5, 5]},\n"
      "          \"opacity\": {\"value\": 1.0},\n"
      "          \"coordinate_system\": {\"value\": \"EPSG:3857\"},\n"
      "          \"aggregation_type\": {\"value\": \"sum\"}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  auto wkb = arctern::render::WktToWkb(string_array);
  arctern::render::weighted_point_map(wkb, c_array, s_array, vega);
}

TEST(POINTMAP_WKT_TEST, DOUBLE) {
  // param1: wkt string
  std::string wkt1 = "POINT (10 10)";
  std::string wkt2 = "POINT (30 30)";
  std::string wkt3 = "POINT (50 50)";
  std::string wkt4 = "POINT (70 70)";
  std::string wkt5 = "POINT (90 90)";
  arrow::StringBuilder string_builder;
  auto status = string_builder.Append(wkt1);
  status = string_builder.Append(wkt2);
  status = string_builder.Append(wkt3);
  status = string_builder.Append(wkt4);
  status = string_builder.Append(wkt5);

  std::shared_ptr<arrow::StringArray> string_array;
  status = string_builder.Finish(&string_array);

  // param2: color
  arrow::DoubleBuilder color_builder;
  status = color_builder.Append(1);
  status = color_builder.Append(2);
  status = color_builder.Append(3);
  status = color_builder.Append(4);
  status = color_builder.Append(5);

  std::shared_ptr<arrow::DoubleArray> c_array;
  status = color_builder.Finish(&c_array);

  // param3: point size
  arrow::DoubleBuilder point_size_builder;
  status = point_size_builder.Append(2);
  status = point_size_builder.Append(4);
  status = point_size_builder.Append(6);
  status = point_size_builder.Append(8);
  status = point_size_builder.Append(10);

  std::shared_ptr<arrow::DoubleArray> s_array;
  status = point_size_builder.Finish(&s_array);

  // param4: conf
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
      "          \"color_gradient\": {\"value\": [\"#0000FF\", \"#FF0000\"]},\n"
      "          \"color_bound\": {\"value\": [2.5, 5]},\n"
      "          \"size_bound\": {\"value\": [2.5, 5]},\n"
      "          \"opacity\": {\"value\": 1.0},\n"
      "          \"coordinate_system\": {\"value\": \"EPSG:3857\"},\n"
      "          \"aggregation_type\": {\"value\": \"sum\"}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  auto wkb = arctern::render::WktToWkb(string_array);
  arctern::render::weighted_point_map(wkb, c_array, s_array, vega);
}

TEST(POINTMAP_WKT_TEST, MEAN) {
  // param1: wkt string
  std::string wkt1 = "POINT (10 10)";
  std::string wkt2 = "POINT (30 30)";
  std::string wkt3 = "POINT (50 50)";
  std::string wkt4 = "POINT (70 70)";
  std::string wkt5 = "POINT (90 90)";
  arrow::StringBuilder string_builder;
  auto status = string_builder.Append(wkt1);
  status = string_builder.Append(wkt2);
  status = string_builder.Append(wkt3);
  status = string_builder.Append(wkt4);
  status = string_builder.Append(wkt5);

  std::shared_ptr<arrow::StringArray> string_array;
  status = string_builder.Finish(&string_array);

  // param2: color
  arrow::DoubleBuilder color_builder;
  status = color_builder.Append(1);
  status = color_builder.Append(2);
  status = color_builder.Append(3);
  status = color_builder.Append(4);
  status = color_builder.Append(5);

  std::shared_ptr<arrow::DoubleArray> c_array;
  status = color_builder.Finish(&c_array);

  // param3: point size
  arrow::DoubleBuilder point_size_builder;
  status = point_size_builder.Append(2);
  status = point_size_builder.Append(4);
  status = point_size_builder.Append(6);
  status = point_size_builder.Append(8);
  status = point_size_builder.Append(10);

  std::shared_ptr<arrow::DoubleArray> s_array;
  status = point_size_builder.Finish(&s_array);

  // param4: conf
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
      "          \"color_gradient\": {\"value\": [\"#0000FF\", \"#FF0000\"]},\n"
      "          \"color_bound\": {\"value\": [2.5, 5]},\n"
      "          \"size_bound\": {\"value\": [2.5, 5]},\n"
      "          \"opacity\": {\"value\": 1.0},\n"
      "          \"coordinate_system\": {\"value\": \"EPSG:3857\"},\n"
      "          \"aggregation_type\": {\"value\": \"mean\"}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  auto wkb = arctern::render::WktToWkb(string_array);
  arctern::render::weighted_point_map(wkb, c_array, s_array, vega);
}

TEST(POINTMAP_WKT_TEST, SUM) {
  // param1: wkt string
  std::string wkt1 = "POINT (10 10)";
  std::string wkt2 = "POINT (30 30)";
  std::string wkt3 = "POINT (50 50)";
  std::string wkt4 = "POINT (70 70)";
  std::string wkt5 = "POINT (90 90)";
  arrow::StringBuilder string_builder;
  auto status = string_builder.Append(wkt1);
  status = string_builder.Append(wkt2);
  status = string_builder.Append(wkt3);
  status = string_builder.Append(wkt4);
  status = string_builder.Append(wkt5);

  std::shared_ptr<arrow::StringArray> string_array;
  status = string_builder.Finish(&string_array);

  // param2: color
  arrow::DoubleBuilder color_builder;
  status = color_builder.Append(1);
  status = color_builder.Append(2);
  status = color_builder.Append(3);
  status = color_builder.Append(4);
  status = color_builder.Append(5);

  std::shared_ptr<arrow::DoubleArray> c_array;
  status = color_builder.Finish(&c_array);

  // param3: point size
  arrow::DoubleBuilder point_size_builder;
  status = point_size_builder.Append(2);
  status = point_size_builder.Append(4);
  status = point_size_builder.Append(6);
  status = point_size_builder.Append(8);
  status = point_size_builder.Append(10);

  std::shared_ptr<arrow::DoubleArray> s_array;
  status = point_size_builder.Finish(&s_array);

  // param4: conf
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
      "          \"color_gradient\": {\"value\": [\"#0000FF\", \"#FF0000\"]},\n"
      "          \"color_bound\": {\"value\": [2.5, 5]},\n"
      "          \"size_bound\": {\"value\": [2.5, 5]},\n"
      "          \"opacity\": {\"value\": 1.0},\n"
      "          \"coordinate_system\": {\"value\": \"EPSG:3857\"},\n"
      "          \"aggregation_type\": {\"value\": \"sum\"}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  auto wkb = arctern::render::WktToWkb(string_array);
  arctern::render::weighted_point_map(wkb, c_array, s_array, vega);
}

TEST(POINTMAP_WKT_TEST, MAX) {
  // param1: wkt string
  std::string wkt1 = "POINT (10 10)";
  std::string wkt2 = "POINT (30 30)";
  std::string wkt3 = "POINT (50 50)";
  std::string wkt4 = "POINT (70 70)";
  std::string wkt5 = "POINT (90 90)";
  arrow::StringBuilder string_builder;
  auto status = string_builder.Append(wkt1);
  status = string_builder.Append(wkt2);
  status = string_builder.Append(wkt3);
  status = string_builder.Append(wkt4);
  status = string_builder.Append(wkt5);

  std::shared_ptr<arrow::StringArray> string_array;
  status = string_builder.Finish(&string_array);

  // param2: color
  arrow::DoubleBuilder color_builder;
  status = color_builder.Append(1);
  status = color_builder.Append(2);
  status = color_builder.Append(3);
  status = color_builder.Append(4);
  status = color_builder.Append(5);

  std::shared_ptr<arrow::DoubleArray> c_array;
  status = color_builder.Finish(&c_array);

  // param3: point size
  arrow::DoubleBuilder point_size_builder;
  status = point_size_builder.Append(2);
  status = point_size_builder.Append(4);
  status = point_size_builder.Append(6);
  status = point_size_builder.Append(8);
  status = point_size_builder.Append(10);

  std::shared_ptr<arrow::DoubleArray> s_array;
  status = point_size_builder.Finish(&s_array);

  // param4: conf
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
      "          \"color_gradient\": {\"value\": [\"#0000FF\", \"#FF0000\"]},\n"
      "          \"color_bound\": {\"value\": [2.5, 5]},\n"
      "          \"size_bound\": {\"value\": [2.5, 5]},\n"
      "          \"opacity\": {\"value\": 1.0},\n"
      "          \"coordinate_system\": {\"value\": \"EPSG:3857\"},\n"
      "          \"aggregation_type\": {\"value\": \"max\"}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  auto wkb = arctern::render::WktToWkb(string_array);
  arctern::render::weighted_point_map(wkb, c_array, s_array, vega);
}

TEST(POINTMAP_WKT_TEST, MIN) {
  // param1: wkt string
  std::string wkt1 = "POINT (10 10)";
  std::string wkt2 = "POINT (30 30)";
  std::string wkt3 = "POINT (50 50)";
  std::string wkt4 = "POINT (70 70)";
  std::string wkt5 = "POINT (90 90)";
  arrow::StringBuilder string_builder;
  auto status = string_builder.Append(wkt1);
  status = string_builder.Append(wkt2);
  status = string_builder.Append(wkt3);
  status = string_builder.Append(wkt4);
  status = string_builder.Append(wkt5);

  std::shared_ptr<arrow::StringArray> string_array;
  status = string_builder.Finish(&string_array);

  // param2: color
  arrow::DoubleBuilder color_builder;
  status = color_builder.Append(1);
  status = color_builder.Append(2);
  status = color_builder.Append(3);
  status = color_builder.Append(4);
  status = color_builder.Append(5);

  std::shared_ptr<arrow::DoubleArray> c_array;
  status = color_builder.Finish(&c_array);

  // param3: point size
  arrow::DoubleBuilder point_size_builder;
  status = point_size_builder.Append(2);
  status = point_size_builder.Append(4);
  status = point_size_builder.Append(6);
  status = point_size_builder.Append(8);
  status = point_size_builder.Append(10);

  std::shared_ptr<arrow::DoubleArray> s_array;
  status = point_size_builder.Finish(&s_array);

  // param4: conf
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
      "          \"color_gradient\": {\"value\": [\"#0000FF\", \"#FF0000\"]},\n"
      "          \"color_bound\": {\"value\": [2.5, 5]},\n"
      "          \"size_bound\": {\"value\": [2.5, 5]},\n"
      "          \"opacity\": {\"value\": 1.0},\n"
      "          \"coordinate_system\": {\"value\": \"EPSG:3857\"},\n"
      "          \"aggregation_type\": {\"value\": \"min\"}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  auto wkb = arctern::render::WktToWkb(string_array);
  arctern::render::weighted_point_map(wkb, c_array, s_array, vega);
}

TEST(POINTMAP_WKT_TEST, COUNT) {
  // param1: wkt string
  std::string wkt1 = "POINT (10 10)";
  std::string wkt2 = "POINT (30 30)";
  std::string wkt3 = "POINT (50 50)";
  std::string wkt4 = "POINT (70 70)";
  std::string wkt5 = "POINT (90 90)";
  arrow::StringBuilder string_builder;
  auto status = string_builder.Append(wkt1);
  status = string_builder.Append(wkt2);
  status = string_builder.Append(wkt3);
  status = string_builder.Append(wkt4);
  status = string_builder.Append(wkt5);

  std::shared_ptr<arrow::StringArray> string_array;
  status = string_builder.Finish(&string_array);

  // param2: color
  arrow::DoubleBuilder color_builder;
  status = color_builder.Append(1);
  status = color_builder.Append(2);
  status = color_builder.Append(3);
  status = color_builder.Append(4);
  status = color_builder.Append(5);

  std::shared_ptr<arrow::DoubleArray> c_array;
  status = color_builder.Finish(&c_array);

  // param3: point size
  arrow::DoubleBuilder point_size_builder;
  status = point_size_builder.Append(2);
  status = point_size_builder.Append(4);
  status = point_size_builder.Append(6);
  status = point_size_builder.Append(8);
  status = point_size_builder.Append(10);

  std::shared_ptr<arrow::DoubleArray> s_array;
  status = point_size_builder.Finish(&s_array);

  // param4: conf
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
      "          \"color_gradient\": {\"value\": [\"#0000FF\", \"#FF0000\"]},\n"
      "          \"color_bound\": {\"value\": [2.5, 5]},\n"
      "          \"size_bound\": {\"value\": [2.5, 5]},\n"
      "          \"opacity\": {\"value\": 1.0},\n"
      "          \"coordinate_system\": {\"value\": \"EPSG:3857\"},\n"
      "          \"aggregation_type\": {\"value\": \"count\"}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  auto wkb = arctern::render::WktToWkb(string_array);
  arctern::render::weighted_point_map(wkb, c_array, s_array, vega);
}

TEST(POINTMAP_WKT_TEST, STD) {
  // param1: wkt string
  std::string wkt1 = "POINT (10 10)";
  std::string wkt2 = "POINT (30 30)";
  std::string wkt3 = "POINT (50 50)";
  std::string wkt4 = "POINT (70 70)";
  std::string wkt5 = "POINT (90 90)";
  arrow::StringBuilder string_builder;
  auto status = string_builder.Append(wkt1);
  status = string_builder.Append(wkt2);
  status = string_builder.Append(wkt3);
  status = string_builder.Append(wkt4);
  status = string_builder.Append(wkt5);

  std::shared_ptr<arrow::StringArray> string_array;
  status = string_builder.Finish(&string_array);

  // param2: color
  arrow::DoubleBuilder color_builder;
  status = color_builder.Append(1);
  status = color_builder.Append(2);
  status = color_builder.Append(3);
  status = color_builder.Append(4);
  status = color_builder.Append(5);

  std::shared_ptr<arrow::DoubleArray> c_array;
  status = color_builder.Finish(&c_array);

  // param3: point size
  arrow::DoubleBuilder point_size_builder;
  status = point_size_builder.Append(2);
  status = point_size_builder.Append(4);
  status = point_size_builder.Append(6);
  status = point_size_builder.Append(8);
  status = point_size_builder.Append(10);

  std::shared_ptr<arrow::DoubleArray> s_array;
  status = point_size_builder.Finish(&s_array);

  // param4: conf
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
      "          \"color_gradient\": {\"value\": [\"#0000FF\", \"#FF0000\"]},\n"
      "          \"color_bound\": {\"value\": [2.5, 5]},\n"
      "          \"size_bound\": {\"value\": [2.5, 5]},\n"
      "          \"opacity\": {\"value\": 1.0},\n"
      "          \"coordinate_system\": {\"value\": \"EPSG:3857\"},\n"
      "          \"aggregation_type\": {\"value\": \"std\"}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  auto wkb = arctern::render::WktToWkb(string_array);
  arctern::render::weighted_point_map(wkb, c_array, s_array, vega);
}

TEST(POINTMAP_WKT_TEST_MULTIPLE_COLOR, INT8) {
  // param1: wkt string
  std::string wkt1 = "POINT (10 10)";
  std::string wkt2 = "POINT (30 30)";
  std::string wkt3 = "POINT (50 50)";
  std::string wkt4 = "POINT (70 70)";
  std::string wkt5 = "POINT (90 90)";
  arrow::StringBuilder string_builder;
  auto status = string_builder.Append(wkt1);
  status = string_builder.Append(wkt2);
  status = string_builder.Append(wkt3);
  status = string_builder.Append(wkt4);
  status = string_builder.Append(wkt5);

  std::shared_ptr<arrow::StringArray> string_array;
  status = string_builder.Finish(&string_array);

  // param2: color
  arrow::Int8Builder color_builder;
  status = color_builder.Append(1);
  status = color_builder.Append(2);
  status = color_builder.Append(3);
  status = color_builder.Append(4);
  status = color_builder.Append(5);

  std::shared_ptr<arrow::Int8Array> c_array;
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
      "          \"color_gradient\": {\"value\": [\"#0000FF\", \"#FF0000\"]},\n"
      "          \"color_bound\": {\"value\": [2.5, 5]},\n"
      "          \"size_bound\": {\"value\": [2.5]},\n"
      "          \"opacity\": {\"value\": 1.0},\n"
      "          \"coordinate_system\": {\"value\": \"EPSG:3857\"},\n"
      "          \"aggregation_type\": {\"value\": \"sum\"}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  auto wkb = arctern::render::WktToWkb(string_array);
  arctern::render::weighted_point_map(wkb, c_array, vega);
}

TEST(POINTMAP_WKT_TEST_MULTIPLE_COLOR, INT16) {
  // param1: wkt string
  std::string wkt1 = "POINT (10 10)";
  std::string wkt2 = "POINT (30 30)";
  std::string wkt3 = "POINT (50 50)";
  std::string wkt4 = "POINT (70 70)";
  std::string wkt5 = "POINT (90 90)";
  arrow::StringBuilder string_builder;
  auto status = string_builder.Append(wkt1);
  status = string_builder.Append(wkt2);
  status = string_builder.Append(wkt3);
  status = string_builder.Append(wkt4);
  status = string_builder.Append(wkt5);

  std::shared_ptr<arrow::StringArray> string_array;
  status = string_builder.Finish(&string_array);

  // param2: color
  arrow::Int16Builder color_builder;
  status = color_builder.Append(1);
  status = color_builder.Append(2);
  status = color_builder.Append(3);
  status = color_builder.Append(4);
  status = color_builder.Append(5);

  std::shared_ptr<arrow::Int16Array> c_array;
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
      "          \"color_gradient\": {\"value\": [\"#0000FF\", \"#FF0000\"]},\n"
      "          \"color_bound\": {\"value\": [2.5, 5]},\n"
      "          \"size_bound\": {\"value\": [2.5]},\n"
      "          \"opacity\": {\"value\": 1.0},\n"
      "          \"coordinate_system\": {\"value\": \"EPSG:3857\"},\n"
      "          \"aggregation_type\": {\"value\": \"sum\"}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  auto wkb = arctern::render::WktToWkb(string_array);
  arctern::render::weighted_point_map(wkb, c_array, vega);
}

TEST(POINTMAP_WKT_TEST_MULTIPLE_COLOR, INT32) {
  // param1: wkt string
  std::string wkt1 = "POINT (10 10)";
  std::string wkt2 = "POINT (30 30)";
  std::string wkt3 = "POINT (50 50)";
  std::string wkt4 = "POINT (70 70)";
  std::string wkt5 = "POINT (90 90)";
  arrow::StringBuilder string_builder;
  auto status = string_builder.Append(wkt1);
  status = string_builder.Append(wkt2);
  status = string_builder.Append(wkt3);
  status = string_builder.Append(wkt4);
  status = string_builder.Append(wkt5);

  std::shared_ptr<arrow::StringArray> string_array;
  status = string_builder.Finish(&string_array);

  // param2: color
  arrow::Int32Builder color_builder;
  status = color_builder.Append(1);
  status = color_builder.Append(2);
  status = color_builder.Append(3);
  status = color_builder.Append(4);
  status = color_builder.Append(5);

  std::shared_ptr<arrow::Int32Array> c_array;
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
      "          \"color_gradient\": {\"value\": [\"#0000FF\", \"#FF0000\"]},\n"
      "          \"color_bound\": {\"value\": [2.5, 5]},\n"
      "          \"size_bound\": {\"value\": [2.5]},\n"
      "          \"opacity\": {\"value\": 1.0},\n"
      "          \"coordinate_system\": {\"value\": \"EPSG:3857\"},\n"
      "          \"aggregation_type\": {\"value\": \"sum\"}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  auto wkb = arctern::render::WktToWkb(string_array);
  arctern::render::weighted_point_map(wkb, c_array, vega);
}

TEST(POINTMAP_WKT_TEST_MULTIPLE_COLOR, INT64) {
  // param1: wkt string
  std::string wkt1 = "POINT (10 10)";
  std::string wkt2 = "POINT (30 30)";
  std::string wkt3 = "POINT (50 50)";
  std::string wkt4 = "POINT (70 70)";
  std::string wkt5 = "POINT (90 90)";
  arrow::StringBuilder string_builder;
  auto status = string_builder.Append(wkt1);
  status = string_builder.Append(wkt2);
  status = string_builder.Append(wkt3);
  status = string_builder.Append(wkt4);
  status = string_builder.Append(wkt5);

  std::shared_ptr<arrow::StringArray> string_array;
  status = string_builder.Finish(&string_array);

  // param2: color
  arrow::Int64Builder color_builder;
  status = color_builder.Append(1);
  status = color_builder.Append(2);
  status = color_builder.Append(3);
  status = color_builder.Append(4);
  status = color_builder.Append(5);

  std::shared_ptr<arrow::Int64Array> c_array;
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
      "          \"color_gradient\": {\"value\": [\"#0000FF\", \"#FF0000\"]},\n"
      "          \"color_bound\": {\"value\": [2.5, 5]},\n"
      "          \"size_bound\": {\"value\": [2.5]},\n"
      "          \"opacity\": {\"value\": 1.0},\n"
      "          \"coordinate_system\": {\"value\": \"EPSG:3857\"},\n"
      "          \"aggregation_type\": {\"value\": \"sum\"}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  auto wkb = arctern::render::WktToWkb(string_array);
  arctern::render::weighted_point_map(wkb, c_array, vega);
}

TEST(POINTMAP_WKT_TEST_MULTIPLE_COLOR, UINT8) {
  // param1: wkt string
  std::string wkt1 = "POINT (10 10)";
  std::string wkt2 = "POINT (30 30)";
  std::string wkt3 = "POINT (50 50)";
  std::string wkt4 = "POINT (70 70)";
  std::string wkt5 = "POINT (90 90)";
  arrow::StringBuilder string_builder;
  auto status = string_builder.Append(wkt1);
  status = string_builder.Append(wkt2);
  status = string_builder.Append(wkt3);
  status = string_builder.Append(wkt4);
  status = string_builder.Append(wkt5);

  std::shared_ptr<arrow::StringArray> string_array;
  status = string_builder.Finish(&string_array);

  // param2: color
  arrow::UInt8Builder color_builder;
  status = color_builder.Append(1);
  status = color_builder.Append(2);
  status = color_builder.Append(3);
  status = color_builder.Append(4);
  status = color_builder.Append(5);

  std::shared_ptr<arrow::UInt8Array> c_array;
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
      "          \"color_gradient\": {\"value\": [\"#0000FF\", \"#FF0000\"]},\n"
      "          \"color_bound\": {\"value\": [2.5, 5]},\n"
      "          \"size_bound\": {\"value\": [2.5]},\n"
      "          \"opacity\": {\"value\": 1.0},\n"
      "          \"coordinate_system\": {\"value\": \"EPSG:3857\"},\n"
      "          \"aggregation_type\": {\"value\": \"sum\"}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  auto wkb = arctern::render::WktToWkb(string_array);
  arctern::render::weighted_point_map(wkb, c_array, vega);
}

TEST(POINTMAP_WKT_TEST_MULTIPLE_COLOR, UINT16) {
  // param1: wkt string
  std::string wkt1 = "POINT (10 10)";
  std::string wkt2 = "POINT (30 30)";
  std::string wkt3 = "POINT (50 50)";
  std::string wkt4 = "POINT (70 70)";
  std::string wkt5 = "POINT (90 90)";
  arrow::StringBuilder string_builder;
  auto status = string_builder.Append(wkt1);
  status = string_builder.Append(wkt2);
  status = string_builder.Append(wkt3);
  status = string_builder.Append(wkt4);
  status = string_builder.Append(wkt5);

  std::shared_ptr<arrow::StringArray> string_array;
  status = string_builder.Finish(&string_array);

  // param2: color
  arrow::UInt16Builder color_builder;
  status = color_builder.Append(1);
  status = color_builder.Append(2);
  status = color_builder.Append(3);
  status = color_builder.Append(4);
  status = color_builder.Append(5);

  std::shared_ptr<arrow::UInt16Array> c_array;
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
      "          \"color_gradient\": {\"value\": [\"#0000FF\", \"#FF0000\"]},\n"
      "          \"color_bound\": {\"value\": [2.5, 5]},\n"
      "          \"size_bound\": {\"value\": [2.5]},\n"
      "          \"opacity\": {\"value\": 1.0},\n"
      "          \"coordinate_system\": {\"value\": \"EPSG:3857\"},\n"
      "          \"aggregation_type\": {\"value\": \"sum\"}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  auto wkb = arctern::render::WktToWkb(string_array);
  arctern::render::weighted_point_map(wkb, c_array, vega);
}

TEST(POINTMAP_WKT_TEST_MULTIPLE_COLOR, UINT32) {
  // param1: wkt string
  std::string wkt1 = "POINT (10 10)";
  std::string wkt2 = "POINT (30 30)";
  std::string wkt3 = "POINT (50 50)";
  std::string wkt4 = "POINT (70 70)";
  std::string wkt5 = "POINT (90 90)";
  arrow::StringBuilder string_builder;
  auto status = string_builder.Append(wkt1);
  status = string_builder.Append(wkt2);
  status = string_builder.Append(wkt3);
  status = string_builder.Append(wkt4);
  status = string_builder.Append(wkt5);

  std::shared_ptr<arrow::StringArray> string_array;
  status = string_builder.Finish(&string_array);

  // param2: color
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
      "          \"color_gradient\": {\"value\": [\"#0000FF\", \"#FF0000\"]},\n"
      "          \"color_bound\": {\"value\": [2.5, 5]},\n"
      "          \"size_bound\": {\"value\": [2.5]},\n"
      "          \"opacity\": {\"value\": 1.0},\n"
      "          \"coordinate_system\": {\"value\": \"EPSG:3857\"},\n"
      "          \"aggregation_type\": {\"value\": \"sum\"}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  auto wkb = arctern::render::WktToWkb(string_array);
  arctern::render::weighted_point_map(wkb, c_array, vega);
}

TEST(POINTMAP_WKT_TEST_MULTIPLE_COLOR, UINT64) {
  // param1: wkt string
  std::string wkt1 = "POINT (10 10)";
  std::string wkt2 = "POINT (30 30)";
  std::string wkt3 = "POINT (50 50)";
  std::string wkt4 = "POINT (70 70)";
  std::string wkt5 = "POINT (90 90)";
  arrow::StringBuilder string_builder;
  auto status = string_builder.Append(wkt1);
  status = string_builder.Append(wkt2);
  status = string_builder.Append(wkt3);
  status = string_builder.Append(wkt4);
  status = string_builder.Append(wkt5);

  std::shared_ptr<arrow::StringArray> string_array;
  status = string_builder.Finish(&string_array);

  // param2: color
  arrow::UInt64Builder color_builder;
  status = color_builder.Append(1);
  status = color_builder.Append(2);
  status = color_builder.Append(3);
  status = color_builder.Append(4);
  status = color_builder.Append(5);

  std::shared_ptr<arrow::UInt64Array> c_array;
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
      "          \"color_gradient\": {\"value\": [\"#0000FF\", \"#FF0000\"]},\n"
      "          \"color_bound\": {\"value\": [2.5, 5]},\n"
      "          \"size_bound\": {\"value\": [2.5]},\n"
      "          \"opacity\": {\"value\": 1.0},\n"
      "          \"coordinate_system\": {\"value\": \"EPSG:3857\"},\n"
      "          \"aggregation_type\": {\"value\": \"sum\"}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  auto wkb = arctern::render::WktToWkb(string_array);
  arctern::render::weighted_point_map(wkb, c_array, vega);
}

TEST(POINTMAP_WKT_TEST_MULTIPLE_COLOR, FLOAT) {
  // param1: wkt string
  std::string wkt1 = "POINT (10 10)";
  std::string wkt2 = "POINT (30 30)";
  std::string wkt3 = "POINT (50 50)";
  std::string wkt4 = "POINT (70 70)";
  std::string wkt5 = "POINT (90 90)";
  arrow::StringBuilder string_builder;
  auto status = string_builder.Append(wkt1);
  status = string_builder.Append(wkt2);
  status = string_builder.Append(wkt3);
  status = string_builder.Append(wkt4);
  status = string_builder.Append(wkt5);

  std::shared_ptr<arrow::StringArray> string_array;
  status = string_builder.Finish(&string_array);

  // param2: color
  arrow::FloatBuilder color_builder;
  status = color_builder.Append(1);
  status = color_builder.Append(2);
  status = color_builder.Append(3);
  status = color_builder.Append(4);
  status = color_builder.Append(5);

  std::shared_ptr<arrow::FloatArray> c_array;
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
      "          \"color_gradient\": {\"value\": [\"#0000FF\", \"#FF0000\"]},\n"
      "          \"color_bound\": {\"value\": [2.5, 5]},\n"
      "          \"size_bound\": {\"value\": [2.5]},\n"
      "          \"opacity\": {\"value\": 1.0},\n"
      "          \"coordinate_system\": {\"value\": \"EPSG:3857\"},\n"
      "          \"aggregation_type\": {\"value\": \"sum\"}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  auto wkb = arctern::render::WktToWkb(string_array);
  arctern::render::weighted_point_map(wkb, c_array, vega);
}

TEST(POINTMAP_WKT_TEST_MULTIPLE_COLOR, DOUBLE) {
  // param1: wkt string
  std::string wkt1 = "POINT (10 10)";
  std::string wkt2 = "POINT (30 30)";
  std::string wkt3 = "POINT (50 50)";
  std::string wkt4 = "POINT (70 70)";
  std::string wkt5 = "POINT (90 90)";
  arrow::StringBuilder string_builder;
  auto status = string_builder.Append(wkt1);
  status = string_builder.Append(wkt2);
  status = string_builder.Append(wkt3);
  status = string_builder.Append(wkt4);
  status = string_builder.Append(wkt5);

  std::shared_ptr<arrow::StringArray> string_array;
  status = string_builder.Finish(&string_array);

  // param2: color
  arrow::DoubleBuilder color_builder;
  status = color_builder.Append(1);
  status = color_builder.Append(2);
  status = color_builder.Append(3);
  status = color_builder.Append(4);
  status = color_builder.Append(5);

  std::shared_ptr<arrow::DoubleArray> c_array;
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
      "          \"color_gradient\": {\"value\": [\"#0000FF\", \"#FF0000\"]},\n"
      "          \"color_bound\": {\"value\": [2.5, 5]},\n"
      "          \"size_bound\": {\"value\": [2.5]},\n"
      "          \"opacity\": {\"value\": 1.0},\n"
      "          \"coordinate_system\": {\"value\": \"EPSG:3857\"},\n"
      "          \"aggregation_type\": {\"value\": \"sum\"}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  auto wkb = arctern::render::WktToWkb(string_array);
  arctern::render::weighted_point_map(wkb, c_array, vega);
}

TEST(POINTMAP_WKT_TEST_MULTIPLE_COLOR, MEAN) {
  // param1: wkt string
  std::string wkt1 = "POINT (10 10)";
  std::string wkt2 = "POINT (30 30)";
  std::string wkt3 = "POINT (50 50)";
  std::string wkt4 = "POINT (70 70)";
  std::string wkt5 = "POINT (90 90)";
  arrow::StringBuilder string_builder;
  auto status = string_builder.Append(wkt1);
  status = string_builder.Append(wkt2);
  status = string_builder.Append(wkt3);
  status = string_builder.Append(wkt4);
  status = string_builder.Append(wkt5);

  std::shared_ptr<arrow::StringArray> string_array;
  status = string_builder.Finish(&string_array);

  // param2: color
  arrow::DoubleBuilder color_builder;
  status = color_builder.Append(1);
  status = color_builder.Append(2);
  status = color_builder.Append(3);
  status = color_builder.Append(4);
  status = color_builder.Append(5);

  std::shared_ptr<arrow::DoubleArray> c_array;
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
      "          \"color_gradient\": {\"value\": [\"#0000FF\", \"#FF0000\"]},\n"
      "          \"color_bound\": {\"value\": [2.5, 5]},\n"
      "          \"size_bound\": {\"value\": [2.5]},\n"
      "          \"opacity\": {\"value\": 1.0},\n"
      "          \"coordinate_system\": {\"value\": \"EPSG:3857\"},\n"
      "          \"aggregation_type\": {\"value\": \"mean\"}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  auto wkb = arctern::render::WktToWkb(string_array);
  arctern::render::weighted_point_map(wkb, c_array, vega);
}

TEST(POINTMAP_WKT_TEST_MULTIPLE_COLOR, SUM) {
  // param1: wkt string
  std::string wkt1 = "POINT (10 10)";
  std::string wkt2 = "POINT (30 30)";
  std::string wkt3 = "POINT (50 50)";
  std::string wkt4 = "POINT (70 70)";
  std::string wkt5 = "POINT (90 90)";
  arrow::StringBuilder string_builder;
  auto status = string_builder.Append(wkt1);
  status = string_builder.Append(wkt2);
  status = string_builder.Append(wkt3);
  status = string_builder.Append(wkt4);
  status = string_builder.Append(wkt5);

  std::shared_ptr<arrow::StringArray> string_array;
  status = string_builder.Finish(&string_array);

  // param2: color
  arrow::DoubleBuilder color_builder;
  status = color_builder.Append(1);
  status = color_builder.Append(2);
  status = color_builder.Append(3);
  status = color_builder.Append(4);
  status = color_builder.Append(5);

  std::shared_ptr<arrow::DoubleArray> c_array;
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
      "          \"color_gradient\": {\"value\": [\"#0000FF\", \"#FF0000\"]},\n"
      "          \"color_bound\": {\"value\": [2.5, 5]},\n"
      "          \"size_bound\": {\"value\": [2.5]},\n"
      "          \"opacity\": {\"value\": 1.0},\n"
      "          \"coordinate_system\": {\"value\": \"EPSG:3857\"},\n"
      "          \"aggregation_type\": {\"value\": \"sum\"}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  auto wkb = arctern::render::WktToWkb(string_array);
  arctern::render::weighted_point_map(wkb, c_array, vega);
}

TEST(POINTMAP_WKT_TEST_MULTIPLE_COLOR, MAX) {
  // param1: wkt string
  std::string wkt1 = "POINT (10 10)";
  std::string wkt2 = "POINT (30 30)";
  std::string wkt3 = "POINT (50 50)";
  std::string wkt4 = "POINT (70 70)";
  std::string wkt5 = "POINT (90 90)";
  arrow::StringBuilder string_builder;
  auto status = string_builder.Append(wkt1);
  status = string_builder.Append(wkt2);
  status = string_builder.Append(wkt3);
  status = string_builder.Append(wkt4);
  status = string_builder.Append(wkt5);

  std::shared_ptr<arrow::StringArray> string_array;
  status = string_builder.Finish(&string_array);

  // param2: color
  arrow::DoubleBuilder color_builder;
  status = color_builder.Append(1);
  status = color_builder.Append(2);
  status = color_builder.Append(3);
  status = color_builder.Append(4);
  status = color_builder.Append(5);

  std::shared_ptr<arrow::DoubleArray> c_array;
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
      "          \"color_gradient\": {\"value\": [\"#0000FF\", \"#FF0000\"]},\n"
      "          \"color_bound\": {\"value\": [2.5, 5]},\n"
      "          \"size_bound\": {\"value\": [2.5]},\n"
      "          \"opacity\": {\"value\": 1.0},\n"
      "          \"coordinate_system\": {\"value\": \"EPSG:3857\"},\n"
      "          \"aggregation_type\": {\"value\": \"max\"}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  auto wkb = arctern::render::WktToWkb(string_array);
  arctern::render::weighted_point_map(wkb, c_array, vega);
}

TEST(POINTMAP_WKT_TEST_MULTIPLE_COLOR, MIN) {
  // param1: wkt string
  std::string wkt1 = "POINT (10 10)";
  std::string wkt2 = "POINT (30 30)";
  std::string wkt3 = "POINT (50 50)";
  std::string wkt4 = "POINT (70 70)";
  std::string wkt5 = "POINT (90 90)";
  arrow::StringBuilder string_builder;
  auto status = string_builder.Append(wkt1);
  status = string_builder.Append(wkt2);
  status = string_builder.Append(wkt3);
  status = string_builder.Append(wkt4);
  status = string_builder.Append(wkt5);

  std::shared_ptr<arrow::StringArray> string_array;
  status = string_builder.Finish(&string_array);

  // param2: color
  arrow::DoubleBuilder color_builder;
  status = color_builder.Append(1);
  status = color_builder.Append(2);
  status = color_builder.Append(3);
  status = color_builder.Append(4);
  status = color_builder.Append(5);

  std::shared_ptr<arrow::DoubleArray> c_array;
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
      "          \"color_gradient\": {\"value\": [\"#0000FF\", \"#FF0000\"]},\n"
      "          \"color_bound\": {\"value\": [2.5, 5]},\n"
      "          \"size_bound\": {\"value\": [2.5]},\n"
      "          \"opacity\": {\"value\": 1.0},\n"
      "          \"coordinate_system\": {\"value\": \"EPSG:3857\"},\n"
      "          \"aggregation_type\": {\"value\": \"min\"}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  auto wkb = arctern::render::WktToWkb(string_array);
  arctern::render::weighted_point_map(wkb, c_array, vega);
}

TEST(POINTMAP_WKT_TEST_MULTIPLE_COLOR, COUNT) {
  // param1: wkt string
  std::string wkt1 = "POINT (10 10)";
  std::string wkt2 = "POINT (30 30)";
  std::string wkt3 = "POINT (50 50)";
  std::string wkt4 = "POINT (70 70)";
  std::string wkt5 = "POINT (90 90)";
  arrow::StringBuilder string_builder;
  auto status = string_builder.Append(wkt1);
  status = string_builder.Append(wkt2);
  status = string_builder.Append(wkt3);
  status = string_builder.Append(wkt4);
  status = string_builder.Append(wkt5);

  std::shared_ptr<arrow::StringArray> string_array;
  status = string_builder.Finish(&string_array);

  // param2: color
  arrow::DoubleBuilder color_builder;
  status = color_builder.Append(1);
  status = color_builder.Append(2);
  status = color_builder.Append(3);
  status = color_builder.Append(4);
  status = color_builder.Append(5);

  std::shared_ptr<arrow::DoubleArray> c_array;
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
      "          \"color_gradient\": {\"value\": [\"#0000FF\", \"#FF0000\"]},\n"
      "          \"color_bound\": {\"value\": [2.5, 5]},\n"
      "          \"size_bound\": {\"value\": [2.5]},\n"
      "          \"opacity\": {\"value\": 1.0},\n"
      "          \"coordinate_system\": {\"value\": \"EPSG:3857\"},\n"
      "          \"aggregation_type\": {\"value\": \"count\"}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  auto wkb = arctern::render::WktToWkb(string_array);
  arctern::render::weighted_point_map(wkb, c_array, vega);
}

TEST(POINTMAP_WKT_TEST_MULTIPLE_COLOR, STD) {
  // param1: wkt string
  std::string wkt1 = "POINT (10 10)";
  std::string wkt2 = "POINT (30 30)";
  std::string wkt3 = "POINT (50 50)";
  std::string wkt4 = "POINT (70 70)";
  std::string wkt5 = "POINT (90 90)";
  arrow::StringBuilder string_builder;
  auto status = string_builder.Append(wkt1);
  status = string_builder.Append(wkt2);
  status = string_builder.Append(wkt3);
  status = string_builder.Append(wkt4);
  status = string_builder.Append(wkt5);

  std::shared_ptr<arrow::StringArray> string_array;
  status = string_builder.Finish(&string_array);

  // param2: color
  arrow::DoubleBuilder color_builder;
  status = color_builder.Append(1);
  status = color_builder.Append(2);
  status = color_builder.Append(3);
  status = color_builder.Append(4);
  status = color_builder.Append(5);

  std::shared_ptr<arrow::DoubleArray> c_array;
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
      "          \"color_gradient\": {\"value\": [\"#0000FF\", \"#FF0000\"]},\n"
      "          \"color_bound\": {\"value\": [2.5, 5]},\n"
      "          \"size_bound\": {\"value\": [2.5]},\n"
      "          \"opacity\": {\"value\": 1.0},\n"
      "          \"coordinate_system\": {\"value\": \"EPSG:3857\"},\n"
      "          \"aggregation_type\": {\"value\": \"std\"}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  auto wkb = arctern::render::WktToWkb(string_array);
  arctern::render::weighted_point_map(wkb, c_array, vega);
}

TEST(POINTMAP_WKT_TEST, BLUE_TO_RED) {
  // param1: wkt string
  std::string wkt1 = "POINT (10 10)";
  std::string wkt2 = "POINT (30 30)";
  std::string wkt3 = "POINT (50 50)";
  std::string wkt4 = "POINT (70 70)";
  std::string wkt5 = "POINT (90 90)";
  arrow::StringBuilder string_builder;
  auto status = string_builder.Append(wkt1);
  status = string_builder.Append(wkt2);
  status = string_builder.Append(wkt3);
  status = string_builder.Append(wkt4);
  status = string_builder.Append(wkt5);

  std::shared_ptr<arrow::StringArray> string_array;
  status = string_builder.Finish(&string_array);

  // param2: color
  arrow::DoubleBuilder color_builder;
  status = color_builder.Append(1);
  status = color_builder.Append(2);
  status = color_builder.Append(3);
  status = color_builder.Append(4);
  status = color_builder.Append(5);

  std::shared_ptr<arrow::DoubleArray> c_array;
  status = color_builder.Finish(&c_array);

  // param3: point size
  arrow::DoubleBuilder point_size_builder;
  status = point_size_builder.Append(2);
  status = point_size_builder.Append(4);
  status = point_size_builder.Append(6);
  status = point_size_builder.Append(8);
  status = point_size_builder.Append(10);

  std::shared_ptr<arrow::DoubleArray> s_array;
  status = point_size_builder.Finish(&s_array);

  // param4: conf
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
      "          \"color_gradient\": {\"value\": [\"#0000FF\", \"#FF0000\"]},\n"
      "          \"color_bound\": {\"value\": [2.5, 5]},\n"
      "          \"size_bound\": {\"value\": [2.5, 5]},\n"
      "          \"opacity\": {\"value\": 1.0},\n"
      "          \"coordinate_system\": {\"value\": \"EPSG:3857\"},\n"
      "          \"aggregation_type\": {\"value\": \"sum\"}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  auto wkb = arctern::render::WktToWkb(string_array);
  arctern::render::weighted_point_map(wkb, c_array, s_array, vega);
}

TEST(POINTMAP_WKT_TEST, SKYBLUE_TO_RED) {
  // param1: wkt string
  std::string wkt1 = "POINT (10 10)";
  std::string wkt2 = "POINT (30 30)";
  std::string wkt3 = "POINT (50 50)";
  std::string wkt4 = "POINT (70 70)";
  std::string wkt5 = "POINT (90 90)";
  arrow::StringBuilder string_builder;
  auto status = string_builder.Append(wkt1);
  status = string_builder.Append(wkt2);
  status = string_builder.Append(wkt3);
  status = string_builder.Append(wkt4);
  status = string_builder.Append(wkt5);

  std::shared_ptr<arrow::StringArray> string_array;
  status = string_builder.Finish(&string_array);

  // param2: color
  arrow::DoubleBuilder color_builder;
  status = color_builder.Append(1);
  status = color_builder.Append(2);
  status = color_builder.Append(3);
  status = color_builder.Append(4);
  status = color_builder.Append(5);

  std::shared_ptr<arrow::DoubleArray> c_array;
  status = color_builder.Finish(&c_array);

  // param3: point size
  arrow::DoubleBuilder point_size_builder;
  status = point_size_builder.Append(2);
  status = point_size_builder.Append(4);
  status = point_size_builder.Append(6);
  status = point_size_builder.Append(8);
  status = point_size_builder.Append(10);

  std::shared_ptr<arrow::DoubleArray> s_array;
  status = point_size_builder.Finish(&s_array);

  // param4: conf
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
      "          \"color_gradient\": {\"value\": [\"#B4E7F5\", \"#FFFFFF\"]},\n"
      "          \"color_bound\": {\"value\": [2.5, 5]},\n"
      "          \"size_bound\": {\"value\": [2.5, 5]},\n"
      "          \"opacity\": {\"value\": 1.0},\n"
      "          \"coordinate_system\": {\"value\": \"EPSG:3857\"},\n"
      "          \"aggregation_type\": {\"value\": \"sum\"}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  auto wkb = arctern::render::WktToWkb(string_array);
  arctern::render::weighted_point_map(wkb, c_array, s_array, vega);
}

TEST(POINTMAP_WKT_TEST, PURPLE_TO_YELLOW) {
  // param1: wkt string
  std::string wkt1 = "POINT (10 10)";
  std::string wkt2 = "POINT (30 30)";
  std::string wkt3 = "POINT (50 50)";
  std::string wkt4 = "POINT (70 70)";
  std::string wkt5 = "POINT (90 90)";
  arrow::StringBuilder string_builder;
  auto status = string_builder.Append(wkt1);
  status = string_builder.Append(wkt2);
  status = string_builder.Append(wkt3);
  status = string_builder.Append(wkt4);
  status = string_builder.Append(wkt5);

  std::shared_ptr<arrow::StringArray> string_array;
  status = string_builder.Finish(&string_array);

  // param2: color
  arrow::DoubleBuilder color_builder;
  status = color_builder.Append(1);
  status = color_builder.Append(2);
  status = color_builder.Append(3);
  status = color_builder.Append(4);
  status = color_builder.Append(5);

  std::shared_ptr<arrow::DoubleArray> c_array;
  status = color_builder.Finish(&c_array);

  // param3: point size
  arrow::DoubleBuilder point_size_builder;
  status = point_size_builder.Append(2);
  status = point_size_builder.Append(4);
  status = point_size_builder.Append(6);
  status = point_size_builder.Append(8);
  status = point_size_builder.Append(10);

  std::shared_ptr<arrow::DoubleArray> s_array;
  status = point_size_builder.Finish(&s_array);

  // param4: conf
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
      "          \"color_gradient\": {\"value\": [\"#FF00FF\", \"#FFFF00\"]},\n"
      "          \"color_bound\": {\"value\": [2.5, 5]},\n"
      "          \"size_bound\": {\"value\": [2.5, 5]},\n"
      "          \"opacity\": {\"value\": 1.0},\n"
      "          \"coordinate_system\": {\"value\": \"EPSG:3857\"},\n"
      "          \"aggregation_type\": {\"value\": \"sum\"}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  auto wkb = arctern::render::WktToWkb(string_array);
  arctern::render::weighted_point_map(wkb, c_array, s_array, vega);
}

TEST(POINTMAP_WKT_TEST, RED_TRANSPARENCY) {
  // param1: wkt string
  std::string wkt1 = "POINT (10 10)";
  std::string wkt2 = "POINT (30 30)";
  std::string wkt3 = "POINT (50 50)";
  std::string wkt4 = "POINT (70 70)";
  std::string wkt5 = "POINT (90 90)";
  arrow::StringBuilder string_builder;
  auto status = string_builder.Append(wkt1);
  status = string_builder.Append(wkt2);
  status = string_builder.Append(wkt3);
  status = string_builder.Append(wkt4);
  status = string_builder.Append(wkt5);

  std::shared_ptr<arrow::StringArray> string_array;
  status = string_builder.Finish(&string_array);

  // param2: color
  arrow::DoubleBuilder color_builder;
  status = color_builder.Append(1);
  status = color_builder.Append(2);
  status = color_builder.Append(3);
  status = color_builder.Append(4);
  status = color_builder.Append(5);

  std::shared_ptr<arrow::DoubleArray> c_array;
  status = color_builder.Finish(&c_array);

  // param3: point size
  arrow::DoubleBuilder point_size_builder;
  status = point_size_builder.Append(2);
  status = point_size_builder.Append(4);
  status = point_size_builder.Append(6);
  status = point_size_builder.Append(8);
  status = point_size_builder.Append(10);

  std::shared_ptr<arrow::DoubleArray> s_array;
  status = point_size_builder.Finish(&s_array);

  // param4: conf
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
      "          \"color_gradient\": {\"value\": [\"#FF0000\", \"#FF0000\"]},\n"
      "          \"color_bound\": {\"value\": [2.5, 5]},\n"
      "          \"size_bound\": {\"value\": [2.5, 5]},\n"
      "          \"opacity\": {\"value\": 1.0},\n"
      "          \"coordinate_system\": {\"value\": \"EPSG:3857\"},\n"
      "          \"aggregation_type\": {\"value\": \"sum\"}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  auto wkb = arctern::render::WktToWkb(string_array);
  arctern::render::weighted_point_map(wkb, c_array, s_array, vega);
}

TEST(POINTMAP_WKT_TEST, BLUE_TRANSPARENCY) {
  // param1: wkt string
  std::string wkt1 = "POINT (10 10)";
  std::string wkt2 = "POINT (30 30)";
  std::string wkt3 = "POINT (50 50)";
  std::string wkt4 = "POINT (70 70)";
  std::string wkt5 = "POINT (90 90)";
  arrow::StringBuilder string_builder;
  auto status = string_builder.Append(wkt1);
  status = string_builder.Append(wkt2);
  status = string_builder.Append(wkt3);
  status = string_builder.Append(wkt4);
  status = string_builder.Append(wkt5);

  std::shared_ptr<arrow::StringArray> string_array;
  status = string_builder.Finish(&string_array);

  // param2: color
  arrow::DoubleBuilder color_builder;
  status = color_builder.Append(1);
  status = color_builder.Append(2);
  status = color_builder.Append(3);
  status = color_builder.Append(4);
  status = color_builder.Append(5);

  std::shared_ptr<arrow::DoubleArray> c_array;
  status = color_builder.Finish(&c_array);

  // param3: point size
  arrow::DoubleBuilder point_size_builder;
  status = point_size_builder.Append(2);
  status = point_size_builder.Append(4);
  status = point_size_builder.Append(6);
  status = point_size_builder.Append(8);
  status = point_size_builder.Append(10);

  std::shared_ptr<arrow::DoubleArray> s_array;
  status = point_size_builder.Finish(&s_array);

  // param4: conf
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
      "          \"color_gradient\": {\"value\": [\"#0000FF\", \"#0000FF\"]},\n"
      "          \"color_bound\": {\"value\": [2.5, 5]},\n"
      "          \"size_bound\": {\"value\": [2.5, 5]},\n"
      "          \"opacity\": {\"value\": 1.0},\n"
      "          \"coordinate_system\": {\"value\": \"EPSG:3857\"},\n"
      "          \"aggregation_type\": {\"value\": \"sum\"}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  auto wkb = arctern::render::WktToWkb(string_array);
  arctern::render::weighted_point_map(wkb, c_array, s_array, vega);
}

TEST(POINTMAP_WKT_TEST, BLUE_GREEN_YELLOW) {
  // param1: wkt string
  std::string wkt1 = "POINT (10 10)";
  std::string wkt2 = "POINT (30 30)";
  std::string wkt3 = "POINT (50 50)";
  std::string wkt4 = "POINT (70 70)";
  std::string wkt5 = "POINT (90 90)";
  arrow::StringBuilder string_builder;
  auto status = string_builder.Append(wkt1);
  status = string_builder.Append(wkt2);
  status = string_builder.Append(wkt3);
  status = string_builder.Append(wkt4);
  status = string_builder.Append(wkt5);

  std::shared_ptr<arrow::StringArray> string_array;
  status = string_builder.Finish(&string_array);

  // param2: color
  arrow::DoubleBuilder color_builder;
  status = color_builder.Append(1);
  status = color_builder.Append(2);
  status = color_builder.Append(3);
  status = color_builder.Append(4);
  status = color_builder.Append(5);

  std::shared_ptr<arrow::DoubleArray> c_array;
  status = color_builder.Finish(&c_array);

  // param3: point size
  arrow::DoubleBuilder point_size_builder;
  status = point_size_builder.Append(2);
  status = point_size_builder.Append(4);
  status = point_size_builder.Append(6);
  status = point_size_builder.Append(8);
  status = point_size_builder.Append(10);

  std::shared_ptr<arrow::DoubleArray> s_array;
  status = point_size_builder.Finish(&s_array);

  // param4: conf
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
      "          \"color_gradient\": {\"value\": [\"#115F9A\", \"#D0F401\"]},\n"
      "          \"color_bound\": {\"value\": [2.5, 5]},\n"
      "          \"size_bound\": {\"value\": [2.5, 5]},\n"
      "          \"opacity\": {\"value\": 1.0},\n"
      "          \"coordinate_system\": {\"value\": \"EPSG:3857\"},\n"
      "          \"aggregation_type\": {\"value\": \"sum\"}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  auto wkb = arctern::render::WktToWkb(string_array);
  arctern::render::weighted_point_map(wkb, c_array, s_array, vega);
}

TEST(POINTMAP_WKT_TEST, WHITE_BLUE) {
  // param1: wkt string
  std::string wkt1 = "POINT (10 10)";
  std::string wkt2 = "POINT (30 30)";
  std::string wkt3 = "POINT (50 50)";
  std::string wkt4 = "POINT (70 70)";
  std::string wkt5 = "POINT (90 90)";
  arrow::StringBuilder string_builder;
  auto status = string_builder.Append(wkt1);
  status = string_builder.Append(wkt2);
  status = string_builder.Append(wkt3);
  status = string_builder.Append(wkt4);
  status = string_builder.Append(wkt5);

  std::shared_ptr<arrow::StringArray> string_array;
  status = string_builder.Finish(&string_array);

  // param2: color
  arrow::DoubleBuilder color_builder;
  status = color_builder.Append(1);
  status = color_builder.Append(2);
  status = color_builder.Append(3);
  status = color_builder.Append(4);
  status = color_builder.Append(5);

  std::shared_ptr<arrow::DoubleArray> c_array;
  status = color_builder.Finish(&c_array);

  // param3: point size
  arrow::DoubleBuilder point_size_builder;
  status = point_size_builder.Append(2);
  status = point_size_builder.Append(4);
  status = point_size_builder.Append(6);
  status = point_size_builder.Append(8);
  status = point_size_builder.Append(10);

  std::shared_ptr<arrow::DoubleArray> s_array;
  status = point_size_builder.Finish(&s_array);

  // param4: conf
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
      "          \"color_gradient\": {\"value\": [\"#E2E2E2\", \"#115F9A\"]},\n"
      "          \"color_bound\": {\"value\": [2.5, 5]},\n"
      "          \"size_bound\": {\"value\": [2.5, 5]},\n"
      "          \"opacity\": {\"value\": 1.0},\n"
      "          \"coordinate_system\": {\"value\": \"EPSG:3857\"},\n"
      "          \"aggregation_type\": {\"value\": \"sum\"}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  auto wkb = arctern::render::WktToWkb(string_array);
  arctern::render::weighted_point_map(wkb, c_array, s_array, vega);
}

TEST(POINTMAP_WKT_TEST, BLUE_WHITE_RED) {
  // param1: wkt string
  std::string wkt1 = "POINT (10 10)";
  std::string wkt2 = "POINT (30 30)";
  std::string wkt3 = "POINT (50 50)";
  std::string wkt4 = "POINT (70 70)";
  std::string wkt5 = "POINT (90 90)";
  arrow::StringBuilder string_builder;
  auto status = string_builder.Append(wkt1);
  status = string_builder.Append(wkt2);
  status = string_builder.Append(wkt3);
  status = string_builder.Append(wkt4);
  status = string_builder.Append(wkt5);

  std::shared_ptr<arrow::StringArray> string_array;
  status = string_builder.Finish(&string_array);

  // param2: color
  arrow::DoubleBuilder color_builder;
  status = color_builder.Append(1);
  status = color_builder.Append(2);
  status = color_builder.Append(3);
  status = color_builder.Append(4);
  status = color_builder.Append(5);

  std::shared_ptr<arrow::DoubleArray> c_array;
  status = color_builder.Finish(&c_array);

  // param3: point size
  arrow::DoubleBuilder point_size_builder;
  status = point_size_builder.Append(2);
  status = point_size_builder.Append(4);
  status = point_size_builder.Append(6);
  status = point_size_builder.Append(8);
  status = point_size_builder.Append(10);

  std::shared_ptr<arrow::DoubleArray> s_array;
  status = point_size_builder.Finish(&s_array);

  // param4: conf
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
      "          \"color_gradient\": {\"value\": [\"#1984C5\", \"#C23728\"]},\n"
      "          \"color_bound\": {\"value\": [2.5, 5]},\n"
      "          \"size_bound\": {\"value\": [2.5, 5]},\n"
      "          \"opacity\": {\"value\": 1.0},\n"
      "          \"coordinate_system\": {\"value\": \"EPSG:3857\"},\n"
      "          \"aggregation_type\": {\"value\": \"sum\"}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  auto wkb = arctern::render::WktToWkb(string_array);
  arctern::render::weighted_point_map(wkb, c_array, s_array, vega);
}

TEST(POINTMAP_WKT_TEST, GREEN_YELLOW_RED) {
  // param1: wkt string
  std::string wkt1 = "POINT (10 10)";
  std::string wkt2 = "POINT (30 30)";
  std::string wkt3 = "POINT (50 50)";
  std::string wkt4 = "POINT (70 70)";
  std::string wkt5 = "POINT (90 90)";
  arrow::StringBuilder string_builder;
  auto status = string_builder.Append(wkt1);
  status = string_builder.Append(wkt2);
  status = string_builder.Append(wkt3);
  status = string_builder.Append(wkt4);
  status = string_builder.Append(wkt5);

  std::shared_ptr<arrow::StringArray> string_array;
  status = string_builder.Finish(&string_array);

  // param2: color
  arrow::DoubleBuilder color_builder;
  status = color_builder.Append(1);
  status = color_builder.Append(2);
  status = color_builder.Append(3);
  status = color_builder.Append(4);
  status = color_builder.Append(5);

  std::shared_ptr<arrow::DoubleArray> c_array;
  status = color_builder.Finish(&c_array);

  // param3: point size
  arrow::DoubleBuilder point_size_builder;
  status = point_size_builder.Append(2);
  status = point_size_builder.Append(4);
  status = point_size_builder.Append(6);
  status = point_size_builder.Append(8);
  status = point_size_builder.Append(10);

  std::shared_ptr<arrow::DoubleArray> s_array;
  status = point_size_builder.Finish(&s_array);

  // param4: conf
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
      "          \"color_gradient\": {\"value\": [\"#4D904F\", \"#C23728\"]},\n"
      "          \"color_bound\": {\"value\": [2.5, 5]},\n"
      "          \"size_bound\": {\"value\": [2.5, 5]},\n"
      "          \"opacity\": {\"value\": 1.0},\n"
      "          \"coordinate_system\": {\"value\": \"EPSG:3857\"},\n"
      "          \"aggregation_type\": {\"value\": \"sum\"}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  auto wkb = arctern::render::WktToWkb(string_array);
  arctern::render::weighted_point_map(wkb, c_array, s_array, vega);
}
