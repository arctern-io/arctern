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

#include "render/render_builder.h"

TEST(POINTMAP_TEST, RAW_POINT_TEST) {
  // param1: x, y
  std::vector<uint32_t> x{50, 51, 52, 53, 54};
  std::vector<uint32_t> y{50, 51, 52, 53, 54};

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
      "          \"point_size\": {\"value\": 30},\n"
      "          \"opacity\": {\"value\": 0.5}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  arctern::render::pointmap(x.data(), y.data(), 5, vega);
}

TEST(POINTMAP_TEST, INVALID_COLOR_TEST) {
  // param1: x, y
  std::vector<uint32_t> x{50, 51, 52, 53, 54};
  std::vector<uint32_t> y{50, 51, 52, 53, 54};

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
      "          \"point_color\": {\"value\": \"#xxxxxx\"},\n"
      "          \"point_size\": {\"value\": 30},\n"
      "          \"opacity\": {\"value\": 0.5}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  arctern::render::pointmap(x.data(), y.data(), 5, vega);
}

TEST(POINTMAP_RAW_POINT_TEST, SINGLE_COLOR_SINGLE_POINTSIZE) {
  // param1: x, y
  std::vector<uint32_t> x{10, 40, 70, 100, 130};
  std::vector<uint32_t> y{10, 40, 70, 100, 130};

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

  arctern::render::weighted_pointmap<int>(x.data(), y.data(), 5, vega);
}

TEST(POINTMAP_RAW_POINT_TEST, MULTIPLE_COLOR_SINGLE_POINTSIZE) {
  // param1: x, y
  std::vector<uint32_t> x{10, 40, 70, 100, 130};
  std::vector<uint32_t> y{10, 40, 70, 100, 130};

  // param2: color
  std::vector<uint32_t> c{1, 2, 3, 4, 5};

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

  arctern::render::weighted_pointmap(x.data(), y.data(), c.data(), 5, vega);
}

TEST(POINTMAP_RAW_POINT_TEST, SINGLE_COLOR_MULTIPLE_POINTSIZE) {
  // param1: x, y
  std::vector<uint32_t> x{10, 40, 70, 100, 130};
  std::vector<uint32_t> y{10, 40, 70, 100, 130};

  // param2: point size
  std::vector<uint32_t> c{6, 8, 10, 12, 14};

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

  arctern::render::weighted_pointmap(x.data(), y.data(), c.data(), 5, vega);
}

TEST(POINTMAP_RAW_POINT_TEST, MULTIPLE_COLOR_MULTIPLE_POINTSIZE) {
  // param1: x, y
  std::vector<uint32_t> x{10, 40, 70, 100, 130};
  std::vector<uint32_t> y{10, 40, 70, 100, 130};

  // param2: color
  std::vector<uint32_t> c{1, 2, 3, 4, 5};

  // param3: size
  std::vector<uint32_t> s{6, 8, 10, 12, 14};

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

  arctern::render::weighted_pointmap(x.data(), y.data(), c.data(), s.data(), 5, vega);
}

TEST(POINTMAP_RAW_POINT_TEST, INT8) {
  // param1: x, y
  std::vector<uint32_t> x{10, 40, 70, 100, 130};
  std::vector<uint32_t> y{10, 40, 70, 100, 130};

  // param2: color
  std::vector<int8_t> c{1, 2, 3, 4, 5};

  // param3: size
  std::vector<int8_t> s{6, 8, 10, 12, 14};

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

  arctern::render::weighted_pointmap(x.data(), y.data(), c.data(), s.data(), 5, vega);
}

TEST(POINTMAP_RAW_POINT_TEST, INT16) {
  // param1: x, y
  std::vector<uint32_t> x{10, 40, 70, 100, 130};
  std::vector<uint32_t> y{10, 40, 70, 100, 130};

  // param2: color
  std::vector<int16_t> c{1, 2, 3, 4, 5};

  // param3: size
  std::vector<int16_t> s{6, 8, 10, 12, 14};

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

  arctern::render::weighted_pointmap(x.data(), y.data(), c.data(), s.data(), 5, vega);
}

TEST(POINTMAP_RAW_POINT_TEST, INT32) {
  // param1: x, y
  std::vector<uint32_t> x{10, 40, 70, 100, 130};
  std::vector<uint32_t> y{10, 40, 70, 100, 130};

  // param2: color
  std::vector<int32_t> c{1, 2, 3, 4, 5};

  // param3: size
  std::vector<int32_t> s{6, 8, 10, 12, 14};

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

  arctern::render::weighted_pointmap(x.data(), y.data(), c.data(), s.data(), 5, vega);
}

TEST(POINTMAP_RAW_POINT_TEST, INT64) {
  // param1: x, y
  std::vector<uint32_t> x{10, 40, 70, 100, 130};
  std::vector<uint32_t> y{10, 40, 70, 100, 130};

  // param2: color
  std::vector<int64_t> c{1, 2, 3, 4, 5};

  // param3: size
  std::vector<int64_t> s{6, 8, 10, 12, 14};

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

  arctern::render::weighted_pointmap(x.data(), y.data(), c.data(), s.data(), 5, vega);
}

TEST(POINTMAP_RAW_POINT_TEST, UINT8) {
  // param1: x, y
  std::vector<uint32_t> x{10, 40, 70, 100, 130};
  std::vector<uint32_t> y{10, 40, 70, 100, 130};

  // param2: color
  std::vector<uint8_t> c{1, 2, 3, 4, 5};

  // param3: size
  std::vector<uint8_t> s{6, 8, 10, 12, 14};

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

  arctern::render::weighted_pointmap(x.data(), y.data(), c.data(), s.data(), 5, vega);
}

TEST(POINTMAP_RAW_POINT_TEST, UINT16) {
  // param1: x, y
  std::vector<uint32_t> x{10, 40, 70, 100, 130};
  std::vector<uint32_t> y{10, 40, 70, 100, 130};

  // param2: color
  std::vector<uint16_t> c{1, 2, 3, 4, 5};

  // param3: size
  std::vector<uint16_t> s{6, 8, 10, 12, 14};

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

  arctern::render::weighted_pointmap(x.data(), y.data(), c.data(), s.data(), 5, vega);
}

TEST(POINTMAP_RAW_POINT_TEST, UINT32) {
  // param1: x, y
  std::vector<uint32_t> x{10, 40, 70, 100, 130};
  std::vector<uint32_t> y{10, 40, 70, 100, 130};

  // param2: color
  std::vector<uint32_t> c{1, 2, 3, 4, 5};

  // param3: size
  std::vector<uint32_t> s{6, 8, 10, 12, 14};

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

  arctern::render::weighted_pointmap(x.data(), y.data(), c.data(), s.data(), 5, vega);
}

TEST(POINTMAP_RAW_POINT_TEST, UINT64) {
  // param1: x, y
  std::vector<uint32_t> x{10, 40, 70, 100, 130};
  std::vector<uint32_t> y{10, 40, 70, 100, 130};

  // param2: color
  std::vector<uint64_t> c{1, 2, 3, 4, 5};

  // param3: size
  std::vector<uint64_t> s{6, 8, 10, 12, 14};

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

  arctern::render::weighted_pointmap(x.data(), y.data(), c.data(), s.data(), 5, vega);
}

TEST(POINTMAP_RAW_POINT_TEST, FLOAT) {
  // param1: x, y
  std::vector<uint32_t> x{10, 40, 70, 100, 130};
  std::vector<uint32_t> y{10, 40, 70, 100, 130};

  // param2: color
  std::vector<float> c{1, 2, 3, 4, 5};

  // param3: size
  std::vector<float> s{6, 8, 10, 12, 14};

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

  arctern::render::weighted_pointmap(x.data(), y.data(), c.data(), s.data(), 5, vega);
}

TEST(POINTMAP_RAW_POINT_TEST, DOUBLE) {
  // param1: x, y
  std::vector<uint32_t> x{10, 40, 70, 100, 130};
  std::vector<uint32_t> y{10, 40, 70, 100, 130};

  // param2: color
  std::vector<double> c{1, 2, 3, 4, 5};

  // param3: size
  std::vector<double> s{6, 8, 10, 12, 14};

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

  arctern::render::weighted_pointmap(x.data(), y.data(), c.data(), s.data(), 5, vega);
}

TEST(POINTMAP_RAW_POINT_TEST_MULTIPLE_COLOR, INT8) {
  // param1: x, y
  std::vector<uint32_t> x{10, 40, 70, 100, 130};
  std::vector<uint32_t> y{10, 40, 70, 100, 130};

  // param2: color
  std::vector<int8_t> c{1, 2, 3, 4, 5};

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

  arctern::render::weighted_pointmap(x.data(), y.data(), c.data(), 5, vega);
}

TEST(POINTMAP_RAW_POINT_TEST_MULTIPLE_COLOR, INT16) {
  // param1: x, y
  std::vector<uint32_t> x{10, 40, 70, 100, 130};
  std::vector<uint32_t> y{10, 40, 70, 100, 130};

  // param2: color
  std::vector<int16_t> c{1, 2, 3, 4, 5};

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

  arctern::render::weighted_pointmap(x.data(), y.data(), c.data(), 5, vega);
}

TEST(POINTMAP_RAW_POINT_TEST_MULTIPLE_COLOR, INT32) {
  // param1: x, y
  std::vector<uint32_t> x{10, 40, 70, 100, 130};
  std::vector<uint32_t> y{10, 40, 70, 100, 130};

  // param2: color
  std::vector<int32_t> c{1, 2, 3, 4, 5};

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

  arctern::render::weighted_pointmap(x.data(), y.data(), c.data(), 5, vega);
}

TEST(POINTMAP_RAW_POINT_TEST_MULTIPLE_COLOR, INT64) {
  // param1: x, y
  std::vector<uint32_t> x{10, 40, 70, 100, 130};
  std::vector<uint32_t> y{10, 40, 70, 100, 130};

  // param2: color
  std::vector<int64_t> c{1, 2, 3, 4, 5};

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

  arctern::render::weighted_pointmap(x.data(), y.data(), c.data(), 5, vega);
}

TEST(POINTMAP_RAW_POINT_TEST_MULTIPLE_COLOR, UINT8) {
  // param1: x, y
  std::vector<uint32_t> x{10, 40, 70, 100, 130};
  std::vector<uint32_t> y{10, 40, 70, 100, 130};

  // param2: color
  std::vector<uint8_t> c{1, 2, 3, 4, 5};

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

  arctern::render::weighted_pointmap(x.data(), y.data(), c.data(), 5, vega);
}

TEST(POINTMAP_RAW_POINT_TEST_MULTIPLE_COLOR, UINT16) {
  // param1: x, y
  std::vector<uint32_t> x{10, 40, 70, 100, 130};
  std::vector<uint32_t> y{10, 40, 70, 100, 130};

  // param2: color
  std::vector<uint16_t> c{1, 2, 3, 4, 5};

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

  arctern::render::weighted_pointmap(x.data(), y.data(), c.data(), 5, vega);
}

TEST(POINTMAP_RAW_POINT_TEST_MULTIPLE_COLOR, UINT32) {
  // param1: x, y
  std::vector<uint32_t> x{10, 40, 70, 100, 130};
  std::vector<uint32_t> y{10, 40, 70, 100, 130};

  // param2: color
  std::vector<uint32_t> c{1, 2, 3, 4, 5};

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

  arctern::render::weighted_pointmap(x.data(), y.data(), c.data(), 5, vega);
}

TEST(POINTMAP_RAW_POINT_TEST_MULTIPLE_COLOR, UINT64) {
  // param1: x, y
  std::vector<uint32_t> x{10, 40, 70, 100, 130};
  std::vector<uint32_t> y{10, 40, 70, 100, 130};

  // param2: color
  std::vector<uint64_t> c{1, 2, 3, 4, 5};

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

  arctern::render::weighted_pointmap(x.data(), y.data(), c.data(), 5, vega);
}

TEST(POINTMAP_RAW_POINT_TEST_MULTIPLE_COLOR, FLOAT) {
  // param1: x, y
  std::vector<uint32_t> x{10, 40, 70, 100, 130};
  std::vector<uint32_t> y{10, 40, 70, 100, 130};

  // param2: color
  std::vector<float> c{1, 2, 3, 4, 5};

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

  arctern::render::weighted_pointmap(x.data(), y.data(), c.data(), 5, vega);
}

TEST(POINTMAP_RAW_POINT_TEST_MULTIPLE_COLOR, DOUBLE) {
  // param1: x, y
  std::vector<uint32_t> x{10, 40, 70, 100, 130};
  std::vector<uint32_t> y{10, 40, 70, 100, 130};

  // param2: color
  std::vector<double> c{1, 2, 3, 4, 5};

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

  arctern::render::weighted_pointmap(x.data(), y.data(), c.data(), 5, vega);
}

TEST(POINTMAP_RAW_POINT_TEST, BLUE_TO_RED) {
  // param1: x, y
  std::vector<uint32_t> x{10, 40, 70, 100, 130};
  std::vector<uint32_t> y{10, 40, 70, 100, 130};

  // param2: color
  std::vector<uint16_t> c{1, 2, 3, 4, 5};

  // param3: size
  std::vector<uint16_t> s{6, 8, 10, 12, 14};

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

  arctern::render::weighted_pointmap(x.data(), y.data(), c.data(), s.data(), 5, vega);
}

TEST(POINTMAP_RAW_POINT_TEST, SKYBLUE_TO_RED) {
  // param1: x, y
  std::vector<uint32_t> x{10, 40, 70, 100, 130};
  std::vector<uint32_t> y{10, 40, 70, 100, 130};

  // param2: color
  std::vector<uint16_t> c{1, 2, 3, 4, 5};

  // param3: size
  std::vector<uint16_t> s{6, 8, 10, 12, 14};

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

  arctern::render::weighted_pointmap(x.data(), y.data(), c.data(), s.data(), 5, vega);
}

TEST(POINTMAP_RAW_POINT_TEST, PURPLE_TO_YELLOW) {
  // param1: x, y
  std::vector<uint32_t> x{10, 40, 70, 100, 130};
  std::vector<uint32_t> y{10, 40, 70, 100, 130};

  // param2: color
  std::vector<uint16_t> c{1, 2, 3, 4, 5};

  // param3: size
  std::vector<uint16_t> s{6, 8, 10, 12, 14};

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

  arctern::render::weighted_pointmap(x.data(), y.data(), c.data(), s.data(), 5, vega);
}

TEST(POINTMAP_RAW_POINT_TEST, RED_TRANSPARENCY) {
  // param1: x, y
  std::vector<uint32_t> x{10, 40, 70, 100, 130};
  std::vector<uint32_t> y{10, 40, 70, 100, 130};

  // param2: color
  std::vector<uint16_t> c{1, 2, 3, 4, 5};

  // param3: size
  std::vector<uint16_t> s{6, 8, 10, 12, 14};

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

  arctern::render::weighted_pointmap(x.data(), y.data(), c.data(), s.data(), 5, vega);
}

TEST(POINTMAP_RAW_POINT_TEST, BLUE_TRANSPARENCY) {
  // param1: x, y
  std::vector<uint32_t> x{10, 40, 70, 100, 130};
  std::vector<uint32_t> y{10, 40, 70, 100, 130};

  // param2: color
  std::vector<uint16_t> c{1, 2, 3, 4, 5};

  // param3: size
  std::vector<uint16_t> s{6, 8, 10, 12, 14};

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

  arctern::render::weighted_pointmap(x.data(), y.data(), c.data(), s.data(), 5, vega);
}

TEST(POINTMAP_RAW_POINT_TEST, BLUE_GREEN_YELLOW) {
  // param1: x, y
  std::vector<uint32_t> x{10, 40, 70, 100, 130};
  std::vector<uint32_t> y{10, 40, 70, 100, 130};

  // param2: color
  std::vector<uint16_t> c{1, 2, 3, 4, 5};

  // param3: size
  std::vector<uint16_t> s{6, 8, 10, 12, 14};

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

  arctern::render::weighted_pointmap(x.data(), y.data(), c.data(), s.data(), 5, vega);
}

TEST(POINTMAP_RAW_POINT_TEST, WHITE_BLUE) {
  // param1: x, y
  std::vector<uint32_t> x{10, 40, 70, 100, 130};
  std::vector<uint32_t> y{10, 40, 70, 100, 130};

  // param2: color
  std::vector<uint16_t> c{1, 2, 3, 4, 5};

  // param3: size
  std::vector<uint16_t> s{6, 8, 10, 12, 14};

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

  arctern::render::weighted_pointmap(x.data(), y.data(), c.data(), s.data(), 5, vega);
}

TEST(POINTMAP_RAW_POINT_TEST, BLUE_WHITE_RED) {
  // param1: x, y
  std::vector<uint32_t> x{10, 40, 70, 100, 130};
  std::vector<uint32_t> y{10, 40, 70, 100, 130};

  // param2: color
  std::vector<uint16_t> c{1, 2, 3, 4, 5};

  // param3: size
  std::vector<uint16_t> s{6, 8, 10, 12, 14};

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

  arctern::render::weighted_pointmap(x.data(), y.data(), c.data(), s.data(), 5, vega);
}

TEST(POINTMAP_RAW_POINT_TEST, GREEN_YELLOW_RED) {
  // param1: x, y
  std::vector<uint32_t> x{10, 40, 70, 100, 130};
  std::vector<uint32_t> y{10, 40, 70, 100, 130};

  // param2: color
  std::vector<uint16_t> c{1, 2, 3, 4, 5};

  // param3: size
  std::vector<uint16_t> s{6, 8, 10, 12, 14};

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

  arctern::render::weighted_pointmap(x.data(), y.data(), c.data(), s.data(), 5, vega);
}

TEST(HEATMAP_TEST, RAW_POINT_INT8_TEST) {
  // param1: x, y
  std::vector<uint32_t> x{50, 51, 52, 53, 54};
  std::vector<uint32_t> y{50, 51, 52, 53, 54};

  // param2: count
  std::vector<int8_t> c{50, 51, 52, 53, 54};

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
      "          \"map_zoom_level\": {\"value\": 10}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  arctern::render::heatmap<int8_t>(x.data(), y.data(), c.data(), 5, vega);
}

TEST(HEATMAP_TEST, RAW_POINT_INT16_TEST) {
  // param1: x, y
  std::vector<uint32_t> x{50, 51, 52, 53, 54};
  std::vector<uint32_t> y{50, 51, 52, 53, 54};

  // param2: count
  std::vector<int16_t> c{50, 51, 52, 53, 54};

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
      "          \"map_zoom_level\": {\"value\": 10}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  arctern::render::heatmap<int16_t>(x.data(), y.data(), c.data(), 5, vega);
}

TEST(HEATMAP_TEST, RAW_POINT_INT32_TEST) {
  // param1: x, y
  std::vector<uint32_t> x{50, 51, 52, 53, 54};
  std::vector<uint32_t> y{50, 51, 52, 53, 54};

  // param2: count
  std::vector<int32_t> c{50, 51, 52, 53, 54};

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
      "          \"map_zoom_level\": {\"value\": 10}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  arctern::render::heatmap<int32_t>(x.data(), y.data(), c.data(), 5, vega);
}

TEST(HEATMAP_TEST, RAW_POINT_INT64_TEST) {
  // param1: x, y
  std::vector<uint32_t> x{50, 51, 52, 53, 54};
  std::vector<uint32_t> y{50, 51, 52, 53, 54};

  // param2: count
  std::vector<int64_t> c{50, 51, 52, 53, 54};

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
      "          \"map_zoom_level\": {\"value\": 10}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  arctern::render::heatmap<int64_t>(x.data(), y.data(), c.data(), 5, vega);
}

TEST(HEATMAP_TEST, RAW_POINT_UINT8_TEST) {
  // param1: x, y
  std::vector<uint32_t> x{50, 51, 52, 53, 54};
  std::vector<uint32_t> y{50, 51, 52, 53, 54};

  // param2: count
  std::vector<uint8_t> c{50, 51, 52, 53, 54};

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
      "          \"map_zoom_level\": {\"value\": 10}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  arctern::render::heatmap<uint8_t>(x.data(), y.data(), c.data(), 5, vega);
}

TEST(HEATMAP_TEST, RAW_POINT_UINT16_TEST) {
  // param1: x, y
  std::vector<uint32_t> x{50, 51, 52, 53, 54};
  std::vector<uint32_t> y{50, 51, 52, 53, 54};

  // param2: count
  std::vector<uint16_t> c{50, 51, 52, 53, 54};

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
      "          \"map_zoom_level\": {\"value\": 10}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  arctern::render::heatmap<uint16_t>(x.data(), y.data(), c.data(), 5, vega);
}

TEST(HEATMAP_TEST, RAW_POINT_UINT32_TEST) {
  // param1: x, y
  std::vector<uint32_t> x{50, 51, 52, 53, 54};
  std::vector<uint32_t> y{50, 51, 52, 53, 54};

  // param2: count
  std::vector<uint32_t> c{50, 51, 52, 53, 54};

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
      "          \"map_zoom_level\": {\"value\": 10}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  arctern::render::heatmap<uint32_t>(x.data(), y.data(), c.data(), 5, vega);
}

TEST(HEATMAP_TEST, RAW_POINT_UINT64_TEST) {
  // param1: x, y
  std::vector<uint32_t> x{50, 51, 52, 53, 54};
  std::vector<uint32_t> y{50, 51, 52, 53, 54};

  // param2: count
  std::vector<uint64_t> c{50, 51, 52, 53, 54};

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
      "          \"map_zoom_level\": {\"value\": 10}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  arctern::render::heatmap<uint64_t>(x.data(), y.data(), c.data(), 5, vega);
}

TEST(HEATMAP_TEST, RAW_POINT_FLOAT_TEST) {
  // param1: x, y
  std::vector<uint32_t> x{50, 51, 52, 53, 54};
  std::vector<uint32_t> y{50, 51, 52, 53, 54};

  // param2: count
  std::vector<float> c{50, 51, 52, 53, 54};

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
      "          \"map_zoom_level\": {\"value\": 10}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  arctern::render::heatmap<float>(x.data(), y.data(), c.data(), 5, vega);
}

TEST(HEATMAP_TEST, RAW_POINT_DOUBLE_TEST) {
  // param1: x, y
  std::vector<uint32_t> x{50, 51, 52, 53, 54};
  std::vector<uint32_t> y{50, 51, 52, 53, 54};

  // param2: count
  std::vector<double> c{50, 51, 52, 53, 54};

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
      "          \"map_zoom_level\": {\"value\": 10}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  arctern::render::heatmap<double>(x.data(), y.data(), c.data(), 5, vega);
}

TEST(ICON_VIZ_TEST, RAW_POINT_TEST) {
  // param1: x, y
  std::vector<uint32_t> x{100, 200, 300, 400, 500};
  std::vector<uint32_t> y{100, 200, 300, 400, 500};

  std::string path = __FILE__;
  path.resize(path.size() - 27);
  std::string icon_path = path + "images/taxi.png";

  // param2: conf
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

  arctern::render::iconviz(x.data(), y.data(), 5, vega);
}
