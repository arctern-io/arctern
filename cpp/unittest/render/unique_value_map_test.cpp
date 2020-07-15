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
#include "render/utils/vega/vega_unique_value_map/vega_unique_value_map.h"

TEST(UNIQUE_VALUE_MAP_TEST, VEGA_TEST_STRING_MAP) {
  const std::string vega =
      "{\n"
      "  \"width\": 1900,\n"
      "  \"height\": 1410,\n"
      "  \"description\": \"unique_value_map\",\n"
      "  \"data\": [\n"
      "    {\n"
      "      \"name\": \"nyc_taxi\",\n"
      "      \"url\": \"data/nyc_taxi_0_5m.csv\"\n"
      "    }\n"
      "  ],\n"
      "  \"scales\": [\n"
      "    {\n"
      "      \"name\": \"geometry\",\n"
      "      \"type\": \"linear\",\n"
      "      \"domain\": {\"data\": \"nyc_taxi\", \"field\": \"c0\"}\n"
      "    }\n"
      "  ],\n"
      "  \"marks\": [\n"
      "    {\n"
      "      \"encode\": {\n"
      "        \"enter\": {\n"
      "          \"bounding_box\": [-73.998427, 40.730309, -73.954348, 40.780816],\n"
      "          \"unique_value_infos\": [\n"
      "            { \n"
      "              \"label\": \"VTS\",\n"
      "              \"color\": \"#00FF00\"\n"
      "            }, \n"
      "            {\n"
      "              \"label\": \"CMT\",\n"
      "              \"color\": \"#FF0000\"\n"
      "            }, \n"
      "            {\n"
      "              \"label\": \"DDS\",\n"
      "              \"color\": \"#0000FF\"\n"
      "            }\n"
      "          ],\n"
      "          \"opacity\": {\"value\": 1.0},\n"
      "          \"coordinate_system\": {\"value\": \"EPSG:3857\"}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  arctern::render::VegaUniqueValueMap vega_unique_value_map(vega);

  const auto& window_params = vega_unique_value_map.window_params();
  const auto& unique_value_infos = vega_unique_value_map.unique_value_infos_string_map();
  const auto& opacity = vega_unique_value_map.opacity();

  assert(window_params.width() == 1900);
  assert(window_params.height() == 1410);

  assert(unique_value_infos.at("VTS").r == 0);
  assert(unique_value_infos.at("VTS").g == 1);
  assert(unique_value_infos.at("VTS").b == 0);
  assert(unique_value_infos.at("CMT").r == 1);
  assert(unique_value_infos.at("CMT").g == 0);
  assert(unique_value_infos.at("CMT").b == 0);
  assert(unique_value_infos.at("DDS").r == 0);
  assert(unique_value_infos.at("DDS").g == 0);
  assert(unique_value_infos.at("DDS").b == 1);

  assert(opacity == 1.0);
}

TEST(UNIQUE_VALUE_MAP_TEST, VEGA_TEST_NUMERIC_MAP) {
  const std::string vega =
      "{\n"
      "  \"width\": 1900,\n"
      "  \"height\": 1410,\n"
      "  \"description\": \"unique_value_map\",\n"
      "  \"data\": [\n"
      "    {\n"
      "      \"name\": \"nyc_taxi\",\n"
      "      \"url\": \"data/nyc_taxi_0_5m.csv\"\n"
      "    }\n"
      "  ],\n"
      "  \"scales\": [\n"
      "    {\n"
      "      \"name\": \"geometry\",\n"
      "      \"type\": \"linear\",\n"
      "      \"domain\": {\"data\": \"nyc_taxi\", \"field\": \"c0\"}\n"
      "    }\n"
      "  ],\n"
      "  \"marks\": [\n"
      "    {\n"
      "      \"encode\": {\n"
      "        \"enter\": {\n"
      "          \"bounding_box\": [-73.998427, 40.730309, -73.954348, 40.780816],\n"
      "          \"unique_value_infos\": [\n"
      "            { \n"
      "              \"label\": 1,\n"
      "              \"color\": \"#00FF00\"\n"
      "            }, \n"
      "            {\n"
      "              \"label\": 2,\n"
      "              \"color\": \"#FF0000\"\n"
      "            }, \n"
      "            {\n"
      "              \"label\": 3,\n"
      "              \"color\": \"#0000FF\"\n"
      "            }\n"
      "          ],\n"
      "          \"opacity\": {\"value\": 1.0},\n"
      "          \"coordinate_system\": {\"value\": \"EPSG:3857\"}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  arctern::render::VegaUniqueValueMap vega_unique_value_map(vega);

  const auto& window_params = vega_unique_value_map.window_params();
  const auto& unique_value_infos = vega_unique_value_map.unique_value_infos_numeric_map();
  const auto& opacity = vega_unique_value_map.opacity();

  assert(window_params.width() == 1900);
  assert(window_params.height() == 1410);

  assert(unique_value_infos.at(1).r == 0);
  assert(unique_value_infos.at(1).g == 1);
  assert(unique_value_infos.at(1).b == 0);
  assert(unique_value_infos.at(2).r == 1);
  assert(unique_value_infos.at(2).g == 0);
  assert(unique_value_infos.at(2).b == 0);
  assert(unique_value_infos.at(3).r == 0);
  assert(unique_value_infos.at(3).g == 0);
  assert(unique_value_infos.at(3).b == 1);

  assert(opacity == 1.0);
}
