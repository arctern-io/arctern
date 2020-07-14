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

TEST(UNIQUE_VALUE_MAP_TEST, VEGA_TEST) {

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
      "          \"unique_value_infos\": {\n"
      "            \"value\": {\n"
      "                \"VTS\": \"#00FF00\",\n"
      "                \"CMT\": \"#FF0000\",\n"
      "                \"DDS\": \"#0000FF\"\n"
      "              }\n"
      "          },\n"
      "          \"opacity\": {\"value\": 1.0},\n"
      "          \"coordinate_system\": {\"value\": \"EPSG:3857\"},\n"
      "          \"aggregation_type\": {\"value\": \"sum\"}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  arctern::render::VegaUniqueValueMap vega_unique_value_map(vega);
  auto unique_value_infos = vega_unique_value_map.unique_value_infos();
}
