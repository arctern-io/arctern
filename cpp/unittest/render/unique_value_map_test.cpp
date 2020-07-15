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

  assert(window_params.width() == 1900);
  assert(window_params.height() == 1410);

  assert(unique_value_infos.at("VTS").r == 0);
  assert(unique_value_infos.at("VTS").g == 1);
  assert(unique_value_infos.at("VTS").b == 0);
  assert(unique_value_infos.at("VTS").a == 1);
  assert(unique_value_infos.at("CMT").r == 1);
  assert(unique_value_infos.at("CMT").g == 0);
  assert(unique_value_infos.at("CMT").b == 0);
  assert(unique_value_infos.at("CMT").a == 1);
  assert(unique_value_infos.at("DDS").r == 0);
  assert(unique_value_infos.at("DDS").g == 0);
  assert(unique_value_infos.at("DDS").b == 1);
  assert(unique_value_infos.at("DDS").a == 1);
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

  assert(window_params.width() == 1900);
  assert(window_params.height() == 1410);

  assert(unique_value_infos.at(1).r == 0);
  assert(unique_value_infos.at(1).g == 1);
  assert(unique_value_infos.at(1).b == 0);
  assert(unique_value_infos.at(1).a == 1);
  assert(unique_value_infos.at(2).r == 1);
  assert(unique_value_infos.at(2).g == 0);
  assert(unique_value_infos.at(2).b == 0);
  assert(unique_value_infos.at(2).a == 1);
  assert(unique_value_infos.at(3).r == 0);
  assert(unique_value_infos.at(3).g == 0);
  assert(unique_value_infos.at(3).b == 1);
  assert(unique_value_infos.at(3).a == 1);
}

TEST(UNIQUE_VALUE_MAP_TEST, TEST_NUMERIC_MAP) {
  // param1: wkt string
  std::string wkt_string1 = "POLYGON ((200 200, 200 300, 300 300, 300 200, 200 200))";
  std::string wkt_string2 = "POLYGON ((400 400, 500 400, 500 500, 400 500, 400 400))";
  std::string wkt_string3 = "POLYGON ((600 600, 700 600, 700 700, 600 700, 600 600))";
  arrow::StringBuilder string_builder;
  auto status = string_builder.Append(wkt_string1);
  status = string_builder.Append(wkt_string2);
  status = string_builder.Append(wkt_string3);

  std::shared_ptr<arrow::StringArray> string_array;
  status = string_builder.Finish(&string_array);

  // param2: value
  std::shared_ptr<arrow::Array> value_array;
  arrow::Int8Builder value_builder;
  status = value_builder.Append(1);
  status = value_builder.Append(2);
  status = value_builder.Append(4);
  status = value_builder.Finish(&value_array);

  // param3: conf
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

  auto wkb = arctern::gis::gdal::WktToWkb(string_array);
  std::vector<std::shared_ptr<arrow::Array>> geo_vec{wkb};

  std::vector<std::shared_ptr<arrow::Array>> value_vec{value_array};
  arctern::render::unique_value_choroplethmap(geo_vec, value_vec, vega);
}

TEST(UNIQUE_VALUE_MAP_TEST, TEST_STRING_MAP) {
  // param1: wkt string
  std::string wkt_string1 = "POLYGON ((200 200, 200 300, 300 300, 300 200, 200 200))";
  std::string wkt_string2 = "POLYGON ((400 400, 500 400, 500 500, 400 500, 400 400))";
  std::string wkt_string3 = "POLYGON ((600 600, 700 600, 700 700, 600 700, 600 600))";
  arrow::StringBuilder string_builder;
  auto status = string_builder.Append(wkt_string1);
  status = string_builder.Append(wkt_string2);
  status = string_builder.Append(wkt_string3);

  std::shared_ptr<arrow::StringArray> string_array;
  status = string_builder.Finish(&string_array);

  // param2: value
  std::shared_ptr<arrow::Array> value_array;
  arrow::StringBuilder value_builder;
  status = value_builder.Append("A");
  status = value_builder.Append("B");
  status = value_builder.Append("C");
  status = value_builder.Finish(&value_array);

  // param3: conf
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
      "              \"label\": \"A\",\n"
      "              \"color\": \"#00FF00\"\n"
      "            }, \n"
      "            {\n"
      "              \"label\": \"B\",\n"
      "              \"color\": \"#FF0000\"\n"
      "            }, \n"
      "            {\n"
      "              \"label\": \"C\",\n"
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

  auto wkb = arctern::gis::gdal::WktToWkb(string_array);
  std::vector<std::shared_ptr<arrow::Array>> geo_vec{wkb};

  std::vector<std::shared_ptr<arrow::Array>> value_vec{value_array};
  arctern::render::unique_value_choroplethmap(geo_vec, value_vec, vega);
}

TEST(UNIQUE_VALUE_MAP_TEST, TEST_INT8) {
  // param1: wkt string
  std::string wkt_string1 = "POLYGON ((200 200, 200 300, 300 300, 300 200, 200 200))";
  std::string wkt_string2 = "POLYGON ((400 400, 500 400, 500 500, 400 500, 400 400))";
  std::string wkt_string3 = "POLYGON ((600 600, 700 600, 700 700, 600 700, 600 600))";
  arrow::StringBuilder string_builder;
  auto status = string_builder.Append(wkt_string1);
  status = string_builder.Append(wkt_string2);
  status = string_builder.Append(wkt_string3);

  std::shared_ptr<arrow::StringArray> string_array;
  status = string_builder.Finish(&string_array);

  // param2: value
  std::shared_ptr<arrow::Array> value_array;
  arrow::Int8Builder value_builder;
  status = value_builder.Append(1);
  status = value_builder.Append(2);
  status = value_builder.Append(4);
  status = value_builder.Finish(&value_array);

  // param3: conf
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

  auto wkb = arctern::gis::gdal::WktToWkb(string_array);
  std::vector<std::shared_ptr<arrow::Array>> geo_vec{wkb};

  std::vector<std::shared_ptr<arrow::Array>> value_vec{value_array};
  arctern::render::unique_value_choroplethmap(geo_vec, value_vec, vega);
}

TEST(UNIQUE_VALUE_MAP_TEST, TEST_INT16) {
  // param1: wkt string
  std::string wkt_string1 = "POLYGON ((200 200, 200 300, 300 300, 300 200, 200 200))";
  std::string wkt_string2 = "POLYGON ((400 400, 500 400, 500 500, 400 500, 400 400))";
  std::string wkt_string3 = "POLYGON ((600 600, 700 600, 700 700, 600 700, 600 600))";
  arrow::StringBuilder string_builder;
  auto status = string_builder.Append(wkt_string1);
  status = string_builder.Append(wkt_string2);
  status = string_builder.Append(wkt_string3);

  std::shared_ptr<arrow::StringArray> string_array;
  status = string_builder.Finish(&string_array);

  // param2: value
  std::shared_ptr<arrow::Array> value_array;
  arrow::Int16Builder value_builder;
  status = value_builder.Append(1);
  status = value_builder.Append(2);
  status = value_builder.Append(4);
  status = value_builder.Finish(&value_array);

  // param3: conf
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

  auto wkb = arctern::gis::gdal::WktToWkb(string_array);
  std::vector<std::shared_ptr<arrow::Array>> geo_vec{wkb};

  std::vector<std::shared_ptr<arrow::Array>> value_vec{value_array};
  arctern::render::unique_value_choroplethmap(geo_vec, value_vec, vega);
}

TEST(UNIQUE_VALUE_MAP_TEST, TEST_INT32) {
  // param1: wkt string
  std::string wkt_string1 = "POLYGON ((200 200, 200 300, 300 300, 300 200, 200 200))";
  std::string wkt_string2 = "POLYGON ((400 400, 500 400, 500 500, 400 500, 400 400))";
  std::string wkt_string3 = "POLYGON ((600 600, 700 600, 700 700, 600 700, 600 600))";
  arrow::StringBuilder string_builder;
  auto status = string_builder.Append(wkt_string1);
  status = string_builder.Append(wkt_string2);
  status = string_builder.Append(wkt_string3);

  std::shared_ptr<arrow::StringArray> string_array;
  status = string_builder.Finish(&string_array);

  // param2: value
  std::shared_ptr<arrow::Array> value_array;
  arrow::Int32Builder value_builder;
  status = value_builder.Append(1);
  status = value_builder.Append(2);
  status = value_builder.Append(4);
  status = value_builder.Finish(&value_array);

  // param3: conf
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

  auto wkb = arctern::gis::gdal::WktToWkb(string_array);
  std::vector<std::shared_ptr<arrow::Array>> geo_vec{wkb};

  std::vector<std::shared_ptr<arrow::Array>> value_vec{value_array};
  arctern::render::unique_value_choroplethmap(geo_vec, value_vec, vega);
}

TEST(UNIQUE_VALUE_MAP_TEST, TEST_INT64) {
  // param1: wkt string
  std::string wkt_string1 = "POLYGON ((200 200, 200 300, 300 300, 300 200, 200 200))";
  std::string wkt_string2 = "POLYGON ((400 400, 500 400, 500 500, 400 500, 400 400))";
  std::string wkt_string3 = "POLYGON ((600 600, 700 600, 700 700, 600 700, 600 600))";
  arrow::StringBuilder string_builder;
  auto status = string_builder.Append(wkt_string1);
  status = string_builder.Append(wkt_string2);
  status = string_builder.Append(wkt_string3);

  std::shared_ptr<arrow::StringArray> string_array;
  status = string_builder.Finish(&string_array);

  // param2: value
  std::shared_ptr<arrow::Array> value_array;
  arrow::Int64Builder value_builder;
  status = value_builder.Append(1);
  status = value_builder.Append(2);
  status = value_builder.Append(4);
  status = value_builder.Finish(&value_array);

  // param3: conf
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

  auto wkb = arctern::gis::gdal::WktToWkb(string_array);
  std::vector<std::shared_ptr<arrow::Array>> geo_vec{wkb};

  std::vector<std::shared_ptr<arrow::Array>> value_vec{value_array};
  arctern::render::unique_value_choroplethmap(geo_vec, value_vec, vega);
}

TEST(UNIQUE_VALUE_MAP_TEST, TEST_UINT8) {
  // param1: wkt string
  std::string wkt_string1 = "POLYGON ((200 200, 200 300, 300 300, 300 200, 200 200))";
  std::string wkt_string2 = "POLYGON ((400 400, 500 400, 500 500, 400 500, 400 400))";
  std::string wkt_string3 = "POLYGON ((600 600, 700 600, 700 700, 600 700, 600 600))";
  arrow::StringBuilder string_builder;
  auto status = string_builder.Append(wkt_string1);
  status = string_builder.Append(wkt_string2);
  status = string_builder.Append(wkt_string3);

  std::shared_ptr<arrow::StringArray> string_array;
  status = string_builder.Finish(&string_array);

  // param2: value
  std::shared_ptr<arrow::Array> value_array;
  arrow::UInt8Builder value_builder;
  status = value_builder.Append(1);
  status = value_builder.Append(2);
  status = value_builder.Append(4);
  status = value_builder.Finish(&value_array);

  // param3: conf
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

  auto wkb = arctern::gis::gdal::WktToWkb(string_array);
  std::vector<std::shared_ptr<arrow::Array>> geo_vec{wkb};

  std::vector<std::shared_ptr<arrow::Array>> value_vec{value_array};
  arctern::render::unique_value_choroplethmap(geo_vec, value_vec, vega);
}

TEST(UNIQUE_VALUE_MAP_TEST, TEST_UINT16) {
  // param1: wkt string
  std::string wkt_string1 = "POLYGON ((200 200, 200 300, 300 300, 300 200, 200 200))";
  std::string wkt_string2 = "POLYGON ((400 400, 500 400, 500 500, 400 500, 400 400))";
  std::string wkt_string3 = "POLYGON ((600 600, 700 600, 700 700, 600 700, 600 600))";
  arrow::StringBuilder string_builder;
  auto status = string_builder.Append(wkt_string1);
  status = string_builder.Append(wkt_string2);
  status = string_builder.Append(wkt_string3);

  std::shared_ptr<arrow::StringArray> string_array;
  status = string_builder.Finish(&string_array);

  // param2: value
  std::shared_ptr<arrow::Array> value_array;
  arrow::UInt16Builder value_builder;
  status = value_builder.Append(1);
  status = value_builder.Append(2);
  status = value_builder.Append(4);
  status = value_builder.Finish(&value_array);

  // param3: conf
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

  auto wkb = arctern::gis::gdal::WktToWkb(string_array);
  std::vector<std::shared_ptr<arrow::Array>> geo_vec{wkb};

  std::vector<std::shared_ptr<arrow::Array>> value_vec{value_array};
  arctern::render::unique_value_choroplethmap(geo_vec, value_vec, vega);
}

TEST(UNIQUE_VALUE_MAP_TEST, TEST_UINT32) {
  // param1: wkt string
  std::string wkt_string1 = "POLYGON ((200 200, 200 300, 300 300, 300 200, 200 200))";
  std::string wkt_string2 = "POLYGON ((400 400, 500 400, 500 500, 400 500, 400 400))";
  std::string wkt_string3 = "POLYGON ((600 600, 700 600, 700 700, 600 700, 600 600))";
  arrow::StringBuilder string_builder;
  auto status = string_builder.Append(wkt_string1);
  status = string_builder.Append(wkt_string2);
  status = string_builder.Append(wkt_string3);

  std::shared_ptr<arrow::StringArray> string_array;
  status = string_builder.Finish(&string_array);

  // param2: value
  std::shared_ptr<arrow::Array> value_array;
  arrow::UInt32Builder value_builder;
  status = value_builder.Append(1);
  status = value_builder.Append(2);
  status = value_builder.Append(4);
  status = value_builder.Finish(&value_array);

  // param3: conf
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

  auto wkb = arctern::gis::gdal::WktToWkb(string_array);
  std::vector<std::shared_ptr<arrow::Array>> geo_vec{wkb};

  std::vector<std::shared_ptr<arrow::Array>> value_vec{value_array};
  arctern::render::unique_value_choroplethmap(geo_vec, value_vec, vega);
}

TEST(UNIQUE_VALUE_MAP_TEST, TEST_UINT64) {
  // param1: wkt string
  std::string wkt_string1 = "POLYGON ((200 200, 200 300, 300 300, 300 200, 200 200))";
  std::string wkt_string2 = "POLYGON ((400 400, 500 400, 500 500, 400 500, 400 400))";
  std::string wkt_string3 = "POLYGON ((600 600, 700 600, 700 700, 600 700, 600 600))";
  arrow::StringBuilder string_builder;
  auto status = string_builder.Append(wkt_string1);
  status = string_builder.Append(wkt_string2);
  status = string_builder.Append(wkt_string3);

  std::shared_ptr<arrow::StringArray> string_array;
  status = string_builder.Finish(&string_array);

  // param2: value
  std::shared_ptr<arrow::Array> value_array;
  arrow::UInt64Builder value_builder;
  status = value_builder.Append(1);
  status = value_builder.Append(2);
  status = value_builder.Append(4);
  status = value_builder.Finish(&value_array);

  // param3: conf
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

  auto wkb = arctern::gis::gdal::WktToWkb(string_array);
  std::vector<std::shared_ptr<arrow::Array>> geo_vec{wkb};

  std::vector<std::shared_ptr<arrow::Array>> value_vec{value_array};
  arctern::render::unique_value_choroplethmap(geo_vec, value_vec, vega);
}

TEST(UNIQUE_VALUE_MAP_TEST, TEST_FLOAT) {
  // param1: wkt string
  std::string wkt_string1 = "POLYGON ((200 200, 200 300, 300 300, 300 200, 200 200))";
  std::string wkt_string2 = "POLYGON ((400 400, 500 400, 500 500, 400 500, 400 400))";
  std::string wkt_string3 = "POLYGON ((600 600, 700 600, 700 700, 600 700, 600 600))";
  arrow::StringBuilder string_builder;
  auto status = string_builder.Append(wkt_string1);
  status = string_builder.Append(wkt_string2);
  status = string_builder.Append(wkt_string3);

  std::shared_ptr<arrow::StringArray> string_array;
  status = string_builder.Finish(&string_array);

  // param2: value
  std::shared_ptr<arrow::Array> value_array;
  arrow::FloatBuilder value_builder;
  status = value_builder.Append(1);
  status = value_builder.Append(2);
  status = value_builder.Append(4);
  status = value_builder.Finish(&value_array);

  // param3: conf
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

  auto wkb = arctern::gis::gdal::WktToWkb(string_array);
  std::vector<std::shared_ptr<arrow::Array>> geo_vec{wkb};

  std::vector<std::shared_ptr<arrow::Array>> value_vec{value_array};
  arctern::render::unique_value_choroplethmap(geo_vec, value_vec, vega);
}

TEST(UNIQUE_VALUE_MAP_TEST, TEST_DOUBLE) {
  // param1: wkt string
  std::string wkt_string1 = "POLYGON ((200 200, 200 300, 300 300, 300 200, 200 200))";
  std::string wkt_string2 = "POLYGON ((400 400, 500 400, 500 500, 400 500, 400 400))";
  std::string wkt_string3 = "POLYGON ((600 600, 700 600, 700 700, 600 700, 600 600))";
  arrow::StringBuilder string_builder;
  auto status = string_builder.Append(wkt_string1);
  status = string_builder.Append(wkt_string2);
  status = string_builder.Append(wkt_string3);

  std::shared_ptr<arrow::StringArray> string_array;
  status = string_builder.Finish(&string_array);

  // param2: value
  std::shared_ptr<arrow::Array> value_array;
  arrow::DoubleBuilder value_builder;
  status = value_builder.Append(1);
  status = value_builder.Append(2);
  status = value_builder.Append(4);
  status = value_builder.Finish(&value_array);

  // param3: conf
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

  auto wkb = arctern::gis::gdal::WktToWkb(string_array);
  std::vector<std::shared_ptr<arrow::Array>> geo_vec{wkb};

  std::vector<std::shared_ptr<arrow::Array>> value_vec{value_array};
  arctern::render::unique_value_choroplethmap(geo_vec, value_vec, vega);
}
