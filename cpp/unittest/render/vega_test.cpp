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
#include <string>
#include <utility>

#include "render/utils/vega/vega_choropleth_map/vega_choropleth_map.h"
#include "render/utils/vega/vega_heatmap/vega_heatmap.h"
#include "render/utils/vega/vega_scatter_plot/vega_circle2d.h"

TEST(VEGA_COMMON, JSON_CHECK_TEST) {
  // 1. invalid json label test
  std::string str =
      "{\n"
      "  \"invalid_label1\": -1,\n"
      "  \"invalid_label2\": -1\n"
      "}";
  arctern::render::VegaCircle2d vega1(str);
  assert(!vega1.is_valid());

  // 2. invalid json size check
  str =
      "{\n"
      "  \"width\": 1900,\n"
      "  \"height\": 1410,\n"
      "  \"description\": \"choropleth_map\",\n"
      "  \"data\": [\n"
      "    {\n"
      "      \"name\": \"nyc_taxi\",\n"
      "      \"url\": \"data/nyc_taxi_0_5m.csv\"\n"
      "    }\n"
      "  ],\n"
      "  \"scales\": [\n"
      "    {\n"
      "      \"name\": \"building\",\n"
      "      \"type\": \"linear\",\n"
      "      \"domain\": {\"data\": \"nyc_taxi\", \"field\": \"c0\"}\n"
      "    }\n"
      "  ],\n"
      "  \"marks\": [\n"
      "    {\n"
      "      \"encode\": {\n"
      "        \"enter\": {\n"
      "          \"bounding_box\": {\"value\": "
      "[-73.984092,40.753893,-73.977588,40.756342]},\n"
      "          \"color_style\": {\"value\": \"blue_to_red\"},\n"
      "          \"ruler\": {\"value\": [1, 2, 3, 4, 5]},\n"
      "          \"opacity\": {\"value\": 1.0}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";
  arctern::render::VegaChoroplethMap vega2(str);
  assert(!vega2.is_valid());

  // 3. invalid number json value type test
  str =
      "{\n"
      "  \"width\": 1900,\n"
      "  \"height\": 1410,\n"
      "  \"description\": \"choropleth_map\",\n"
      "  \"data\": [\n"
      "    {\n"
      "      \"name\": \"nyc_taxi\",\n"
      "      \"url\": \"data/nyc_taxi_0_5m.csv\"\n"
      "    }\n"
      "  ],\n"
      "  \"scales\": [\n"
      "    {\n"
      "      \"name\": \"building\",\n"
      "      \"type\": \"linear\",\n"
      "      \"domain\": {\"data\": \"nyc_taxi\", \"field\": \"c0\"}\n"
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
      "          \"opacity\": {\"value\": \"string_type\"}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";
  arctern::render::VegaChoroplethMap vega3(str);
  assert(!vega3.is_valid());

  // 4. invalid array json value type test
  str =
      "{\n"
      "  \"width\": 1900,\n"
      "  \"height\": 1410,\n"
      "  \"description\": \"choropleth_map\",\n"
      "  \"data\": [\n"
      "    {\n"
      "      \"name\": \"nyc_taxi\",\n"
      "      \"url\": \"data/nyc_taxi_0_5m.csv\"\n"
      "    }\n"
      "  ],\n"
      "  \"scales\": [\n"
      "    {\n"
      "      \"name\": \"building\",\n"
      "      \"type\": \"linear\",\n"
      "      \"domain\": {\"data\": \"nyc_taxi\", \"field\": \"c0\"}\n"
      "    }\n"
      "  ],\n"
      "  \"marks\": -1\n"
      "}";
  arctern::render::VegaChoroplethMap vega4(str);
  assert(!vega4.is_valid());

  // 5. invalid string json value type test
  str =
      "{\n"
      "  \"width\": 1900,\n"
      "  \"height\": 1410,\n"
      "  \"description\": \"choropleth_map\",\n"
      "  \"data\": [\n"
      "    {\n"
      "      \"name\": \"nyc_taxi\",\n"
      "      \"url\": \"data/nyc_taxi_0_5m.csv\"\n"
      "    }\n"
      "  ],\n"
      "  \"scales\": [\n"
      "    {\n"
      "      \"name\": \"building\",\n"
      "      \"type\": \"linear\",\n"
      "      \"domain\": {\"data\": \"nyc_taxi\", \"field\": \"c0\"}\n"
      "    }\n"
      "  ],\n"
      "  \"marks\": [\n"
      "    {\n"
      "      \"encode\": {\n"
      "        \"enter\": {\n"
      "          \"bounding_box\": {\"value\": "
      "[-73.984092,40.753893,-73.977588,40.756342]},\n"
      "          \"color_style\": {\"value\": -1},\n"
      "          \"ruler\": {\"value\": [2.5, 5]},\n"
      "          \"opacity\": {\"value\": 1.0}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";
  arctern::render::VegaChoroplethMap vega5(str);
  assert(!vega5.is_valid());

  // 6. invalid json null test
  str =
      "{\n"
      "  \"width\": null,\n"
      "  \"height\": 200,\n"
      "  \"description\": \"heat_map\",\n"
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
      "          \"map_scale\": {\"value\": 10}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";
  arctern::render::VegaHeatMap vega6(str);
  assert(!vega6.is_valid());
}

TEST(VEGA_POINT_MAP, JSON_CHECK_TEST) {
  // 1. invalid json test
  std::string str = " ";
  arctern::render::VegaCircle2d vega1(str);
  assert(!vega1.is_valid());

  // 2. invalid width and height test
  str =
      "{\n"
      "  \"invalid_width\": 300,\n"
      "  \"invalid_height\": 200\n"
      "}";
  arctern::render::VegaCircle2d vega2(str);
  assert(!vega2.is_valid());

  // 3. invalid marks test
  str =
      "{\n"
      "  \"width\": 300,\n"
      "  \"height\": 200,\n"
      "  \"description\": \"circle_2d\",\n"
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
      "  \"invalid_marks\": -1\n"
      "}";
  arctern::render::VegaCircle2d vega3(str);
  assert(!vega3.is_valid());

  // 4. invalid point opacity test
  str =
      "{\n"
      "  \"width\": 300,\n"
      "  \"height\": 200,\n"
      "  \"description\": \"circle_2d\",\n"
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
      "          \"shape\": {\"value\": \"circle\"},\n"
      "          \"stroke\": {\"value\": \"#EE113D\"},\n"
      "          \"strokeWidth\": {\"value\": 3},\n"
      "          \"opacity\": {\"value\": \"INVALID\"}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";
  arctern::render::VegaCircle2d vega4(str);
  assert(!vega4.is_valid());
}

TEST(VEGA_HEAT_MAP, JSON_CHECK_TEST) {
  // 1. invalid json test
  std::string str = " ";
  arctern::render::VegaHeatMap vega1(str);
  assert(!vega1.is_valid());

  // 2. invalid width and height test
  str =
      "{\n"
      "  \"invalid_width\": 300,\n"
      "  \"invalid_height\": 200\n"
      "}";
  arctern::render::VegaHeatMap vega2(str);
  assert(!vega2.is_valid());

  // 3. invalid marks test
  str =
      "{\n"
      "  \"width\": 300,\n"
      "  \"height\": 200,\n"
      "  \"description\": \"circle_2d\",\n"
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
      "  \"invalid_marks\": -1\n"
      "}";
  arctern::render::VegaHeatMap vega3(str);
  assert(!vega3.is_valid());

  // 4. invalid heat map map scale test
  str =
      "{\n"
      "  \"width\": 300,\n"
      "  \"height\": 200,\n"
      "  \"description\": \"heat_map_2d\",\n"
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
      "          \"map_scale\": {\"value\": \"INVALID\"}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";
  arctern::render::VegaHeatMap vega4(str);
  assert(!vega4.is_valid());
}

TEST(VEGA_CHOROPLETH_MAP, JSON_CHECK_TEST) {
  // 1. invalid json test
  std::string str = " ";
  arctern::render::VegaChoroplethMap vega1(str);
  assert(!vega1.is_valid());

  // 2. invalid width and height test
  str =
      "{\n"
      "  \"invalid_width\": 300,\n"
      "  \"invalid_height\": 200\n"
      "}";
  arctern::render::VegaChoroplethMap vega2(str);
  assert(!vega2.is_valid());

  // 3. invalid marks test
  str =
      "{\n"
      "  \"width\": 300,\n"
      "  \"height\": 200,\n"
      "  \"description\": \"circle_2d\",\n"
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
      "  \"invalid_marks\": -1\n"
      "}";
  arctern::render::VegaChoroplethMap vega3(str);
  assert(!vega3.is_valid());

  // 4. invalid bounding box label test
  str =
      "{\n"
      "  \"width\": 1900,\n"
      "  \"height\": 1410,\n"
      "  \"description\": \"choropleth_map\",\n"
      "  \"data\": [\n"
      "    {\n"
      "      \"name\": \"nyc_taxi\",\n"
      "      \"url\": \"data/nyc_taxi_0_5m.csv\"\n"
      "    }\n"
      "  ],\n"
      "  \"scales\": [\n"
      "    {\n"
      "      \"name\": \"building\",\n"
      "      \"type\": \"linear\",\n"
      "      \"domain\": {\"data\": \"nyc_taxi\", \"field\": \"c0\"}\n"
      "    }\n"
      "  ],\n"
      "  \"marks\": [\n"
      "    {\n"
      "      \"encode\": {\n"
      "        \"enter\": {\n"
      "          \"bounding_box\": {\"value\": \"INVALID\"},\n"
      "          \"color_style\": {\"value\": \"blue_to_red\"},\n"
      "          \"ruler\": {\"value\": [2.5, 5]},\n"
      "          \"opacity\": {\"value\": 1.0}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";
  arctern::render::VegaChoroplethMap vega4(str);
  assert(!vega4.is_valid());

  // 5. invalid bounding box type test
  str =
      "{\n"
      "  \"width\": 1900,\n"
      "  \"height\": 1410,\n"
      "  \"description\": \"choropleth_map\",\n"
      "  \"data\": [\n"
      "    {\n"
      "      \"name\": \"nyc_taxi\",\n"
      "      \"url\": \"data/nyc_taxi_0_5m.csv\"\n"
      "    }\n"
      "  ],\n"
      "  \"scales\": [\n"
      "    {\n"
      "      \"name\": \"building\",\n"
      "      \"type\": \"linear\",\n"
      "      \"domain\": {\"data\": \"nyc_taxi\", \"field\": \"c0\"}\n"
      "    }\n"
      "  ],\n"
      "  \"marks\": [\n"
      "    {\n"
      "      \"encode\": {\n"
      "        \"enter\": {\n"
      "          \"bounding_box\": {\"value\": "
      "[-73.984092,40.753893,-73.977588,\"40.756342\"]},\n"
      "          \"color_style\": {\"value\": \"blue_to_red\"},\n"
      "          \"ruler\": {\"value\": [2.5, 5]},\n"
      "          \"opacity\": {\"value\": 1.0}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";
  arctern::render::VegaChoroplethMap vega5(str);
  assert(!vega5.is_valid());

  // 6. invalid color style label test
  str =
      "{\n"
      "  \"width\": 1900,\n"
      "  \"height\": 1410,\n"
      "  \"description\": \"choropleth_map\",\n"
      "  \"data\": [\n"
      "    {\n"
      "      \"name\": \"nyc_taxi\",\n"
      "      \"url\": \"data/nyc_taxi_0_5m.csv\"\n"
      "    }\n"
      "  ],\n"
      "  \"scales\": [\n"
      "    {\n"
      "      \"name\": \"building\",\n"
      "      \"type\": \"linear\",\n"
      "      \"domain\": {\"data\": \"nyc_taxi\", \"field\": \"c0\"}\n"
      "    }\n"
      "  ],\n"
      "  \"marks\": [\n"
      "    {\n"
      "      \"encode\": {\n"
      "        \"enter\": {\n"
      "          \"bounding_box\": {\"value\": "
      "[-73.984092,40.753893,-73.977588,40.756342]},\n"
      "          \"invalid_color_style\": {\"value\": \"blue_to_red\"},\n"
      "          \"ruler\": {\"value\": [2.5, 5]},\n"
      "          \"opacity\": {\"value\": 1.0}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";
  arctern::render::VegaChoroplethMap vega6(str);
  assert(!vega6.is_valid());

  // 7. invalid color style value test
  str =
      "{\n"
      "  \"width\": 1900,\n"
      "  \"height\": 1410,\n"
      "  \"description\": \"choropleth_map\",\n"
      "  \"data\": [\n"
      "    {\n"
      "      \"name\": \"nyc_taxi\",\n"
      "      \"url\": \"data/nyc_taxi_0_5m.csv\"\n"
      "    }\n"
      "  ],\n"
      "  \"scales\": [\n"
      "    {\n"
      "      \"name\": \"building\",\n"
      "      \"type\": \"linear\",\n"
      "      \"domain\": {\"data\": \"nyc_taxi\", \"field\": \"c0\"}\n"
      "    }\n"
      "  ],\n"
      "  \"marks\": [\n"
      "    {\n"
      "      \"encode\": {\n"
      "        \"enter\": {\n"
      "          \"bounding_box\": {\"value\": "
      "[-73.984092,40.753893,-73.977588,40.756342]},\n"
      "          \"color_style\": {\"value\": 10},\n"
      "          \"ruler\": {\"value\": [2.5, 5]},\n"
      "          \"opacity\": {\"value\": 1.0}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";
  arctern::render::VegaChoroplethMap vega7(str);
  assert(!vega7.is_valid());

  // 8. invalid ruler label test
  str =
      "{\n"
      "  \"width\": 1900,\n"
      "  \"height\": 1410,\n"
      "  \"description\": \"choropleth_map\",\n"
      "  \"data\": [\n"
      "    {\n"
      "      \"name\": \"nyc_taxi\",\n"
      "      \"url\": \"data/nyc_taxi_0_5m.csv\"\n"
      "    }\n"
      "  ],\n"
      "  \"scales\": [\n"
      "    {\n"
      "      \"name\": \"building\",\n"
      "      \"type\": \"linear\",\n"
      "      \"domain\": {\"data\": \"nyc_taxi\", \"field\": \"c0\"}\n"
      "    }\n"
      "  ],\n"
      "  \"marks\": [\n"
      "    {\n"
      "      \"encode\": {\n"
      "        \"enter\": {\n"
      "          \"bounding_box\": {\"value\": "
      "[-73.984092,40.753893,-73.977588,40.756342]},\n"
      "          \"color_style\": {\"value\": \"blue_to_red\"},\n"
      "          \"invalid_ruler\": {\"value\": [2.5, 5]},\n"
      "          \"opacity\": {\"value\": 1.0}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";
  arctern::render::VegaChoroplethMap vega8(str);
  assert(!vega8.is_valid());

  // 9. invalid ruler type test
  str =
      "{\n"
      "  \"width\": 1900,\n"
      "  \"height\": 1410,\n"
      "  \"description\": \"choropleth_map\",\n"
      "  \"data\": [\n"
      "    {\n"
      "      \"name\": \"nyc_taxi\",\n"
      "      \"url\": \"data/nyc_taxi_0_5m.csv\"\n"
      "    }\n"
      "  ],\n"
      "  \"scales\": [\n"
      "    {\n"
      "      \"name\": \"building\",\n"
      "      \"type\": \"linear\",\n"
      "      \"domain\": {\"data\": \"nyc_taxi\", \"field\": \"c0\"}\n"
      "    }\n"
      "  ],\n"
      "  \"marks\": [\n"
      "    {\n"
      "      \"encode\": {\n"
      "        \"enter\": {\n"
      "          \"bounding_box\": {\"value\": "
      "[-73.984092,40.753893,-73.977588,40.756342]},\n"
      "          \"color_style\": {\"value\": \"blue_to_red\"},\n"
      "          \"ruler\": {\"value\": [2.5, \"INVALID_RULER_VALUE\"]},\n"
      "          \"opacity\": {\"value\": 1.0}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";
  arctern::render::VegaChoroplethMap vega9(str);
  assert(!vega9.is_valid());

  // 10. invalid opacity test
  str =
      "{\n"
      "  \"width\": 1900,\n"
      "  \"height\": 1410,\n"
      "  \"description\": \"choropleth_map\",\n"
      "  \"data\": [\n"
      "    {\n"
      "      \"name\": \"nyc_taxi\",\n"
      "      \"url\": \"data/nyc_taxi_0_5m.csv\"\n"
      "    }\n"
      "  ],\n"
      "  \"scales\": [\n"
      "    {\n"
      "      \"name\": \"building\",\n"
      "      \"type\": \"linear\",\n"
      "      \"domain\": {\"data\": \"nyc_taxi\", \"field\": \"c0\"}\n"
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
      "          \"opacity\": {\"value\": \"INVALID_OPACITY_VALUE\"}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";
  arctern::render::VegaChoroplethMap vega10(str);
  assert(!vega10.is_valid());
}
