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
#include <ogr_geometry.h>

#include "arrow/render_api.h"

TEST(FISHNETMAP_TEST, INT8) {
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
      "  \"width\": 1900,\n"
      "  \"height\": 1410,\n"
      "  \"description\": \"fishnet_map_2d\",\n"
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
      "          \"bounding_box\": {\"value\": "
      "[-73.984092,40.753893,-73.977588,40.756342]},\n"
      "          \"color_gradient\": {\"value\": [\"#0000FF\", \"#FF0000\"]},\n"
      "          \"cell_size\": {\"value\": 4.0},\n"
      "          \"cell_spacing\": {\"value\": 1.0},\n"
      "          \"opacity\": {\"value\": 1.0},\n"
      "          \"aggregation_type\": {\"value\": \"sum\"}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  auto wkb = arctern::render::WktToWkb(string_array);

  std::vector<std::shared_ptr<arrow::Array>> point_vec{wkb};
  std::vector<std::shared_ptr<arrow::Array>> color_vec{c_array};

  arctern::render::fishnet_map(point_vec, color_vec, vega);
}

TEST(FISHNETMAP_TEST, INT16) {
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
      "  \"width\": 1900,\n"
      "  \"height\": 1410,\n"
      "  \"description\": \"fishnet_map_2d\",\n"
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
      "          \"bounding_box\": {\"value\": "
      "[-73.984092,40.753893,-73.977588,40.756342]},\n"
      "          \"color_gradient\": {\"value\": [\"#0000FF\", \"#FF0000\"]},\n"
      "          \"cell_size\": {\"value\": 4.0},\n"
      "          \"cell_spacing\": {\"value\": 1.0},\n"
      "          \"opacity\": {\"value\": 1.0},\n"
      "          \"aggregation_type\": {\"value\": \"sum\"}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  auto wkb = arctern::render::WktToWkb(string_array);

  std::vector<std::shared_ptr<arrow::Array>> point_vec{wkb};
  std::vector<std::shared_ptr<arrow::Array>> color_vec{c_array};

  arctern::render::fishnet_map(point_vec, color_vec, vega);
}

TEST(FISHNETMAP_TEST, INT32) {
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
      "  \"width\": 1900,\n"
      "  \"height\": 1410,\n"
      "  \"description\": \"fishnet_map_2d\",\n"
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
      "          \"bounding_box\": {\"value\": "
      "[-73.984092,40.753893,-73.977588,40.756342]},\n"
      "          \"color_gradient\": {\"value\": [\"#0000FF\", \"#FF0000\"]},\n"
      "          \"cell_size\": {\"value\": 4.0},\n"
      "          \"cell_spacing\": {\"value\": 1.0},\n"
      "          \"opacity\": {\"value\": 1.0},\n"
      "          \"aggregation_type\": {\"value\": \"sum\"}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  auto wkb = arctern::render::WktToWkb(string_array);

  std::vector<std::shared_ptr<arrow::Array>> point_vec{wkb};
  std::vector<std::shared_ptr<arrow::Array>> color_vec{c_array};

  arctern::render::fishnet_map(point_vec, color_vec, vega);
}

TEST(FISHNETMAP_TEST, INT64) {
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
      "  \"width\": 1900,\n"
      "  \"height\": 1410,\n"
      "  \"description\": \"fishnet_map_2d\",\n"
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
      "          \"bounding_box\": {\"value\": "
      "[-73.984092,40.753893,-73.977588,40.756342]},\n"
      "          \"color_gradient\": {\"value\": [\"#0000FF\", \"#FF0000\"]},\n"
      "          \"cell_size\": {\"value\": 4.0},\n"
      "          \"cell_spacing\": {\"value\": 1.0},\n"
      "          \"opacity\": {\"value\": 1.0},\n"
      "          \"aggregation_type\": {\"value\": \"sum\"}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  auto wkb = arctern::render::WktToWkb(string_array);

  std::vector<std::shared_ptr<arrow::Array>> point_vec{wkb};
  std::vector<std::shared_ptr<arrow::Array>> color_vec{c_array};

  arctern::render::fishnet_map(point_vec, color_vec, vega);
}

TEST(FISHNETMAP_TEST, UINT8) {
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
      "  \"width\": 1900,\n"
      "  \"height\": 1410,\n"
      "  \"description\": \"fishnet_map_2d\",\n"
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
      "          \"bounding_box\": {\"value\": "
      "[-73.984092,40.753893,-73.977588,40.756342]},\n"
      "          \"color_gradient\": {\"value\": [\"#0000FF\", \"#FF0000\"]},\n"
      "          \"cell_size\": {\"value\": 4.0},\n"
      "          \"cell_spacing\": {\"value\": 1.0},\n"
      "          \"opacity\": {\"value\": 1.0},\n"
      "          \"aggregation_type\": {\"value\": \"sum\"}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  auto wkb = arctern::render::WktToWkb(string_array);

  std::vector<std::shared_ptr<arrow::Array>> point_vec{wkb};
  std::vector<std::shared_ptr<arrow::Array>> color_vec{c_array};

  arctern::render::fishnet_map(point_vec, color_vec, vega);
}

TEST(FISHNETMAP_TEST, UINT16) {
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
      "  \"width\": 1900,\n"
      "  \"height\": 1410,\n"
      "  \"description\": \"fishnet_map_2d\",\n"
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
      "          \"bounding_box\": {\"value\": "
      "[-73.984092,40.753893,-73.977588,40.756342]},\n"
      "          \"color_gradient\": {\"value\": [\"#0000FF\", \"#FF0000\"]},\n"
      "          \"cell_size\": {\"value\": 4.0},\n"
      "          \"cell_spacing\": {\"value\": 1.0},\n"
      "          \"opacity\": {\"value\": 1.0},\n"
      "          \"aggregation_type\": {\"value\": \"sum\"}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  auto wkb = arctern::render::WktToWkb(string_array);

  std::vector<std::shared_ptr<arrow::Array>> point_vec{wkb};
  std::vector<std::shared_ptr<arrow::Array>> color_vec{c_array};

  arctern::render::fishnet_map(point_vec, color_vec, vega);
}

TEST(FISHNETMAP_TEST, UINT32) {
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
      "  \"width\": 1900,\n"
      "  \"height\": 1410,\n"
      "  \"description\": \"fishnet_map_2d\",\n"
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
      "          \"bounding_box\": {\"value\": "
      "[-73.984092,40.753893,-73.977588,40.756342]},\n"
      "          \"color_gradient\": {\"value\": [\"#0000FF\", \"#FF0000\"]},\n"
      "          \"cell_size\": {\"value\": 4.0},\n"
      "          \"cell_spacing\": {\"value\": 1.0},\n"
      "          \"opacity\": {\"value\": 1.0},\n"
      "          \"aggregation_type\": {\"value\": \"sum\"}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  auto wkb = arctern::render::WktToWkb(string_array);

  std::vector<std::shared_ptr<arrow::Array>> point_vec{wkb};
  std::vector<std::shared_ptr<arrow::Array>> color_vec{c_array};

  arctern::render::fishnet_map(point_vec, color_vec, vega);
}

TEST(FISHNETMAP_TEST, UINT64) {
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
      "  \"width\": 1900,\n"
      "  \"height\": 1410,\n"
      "  \"description\": \"fishnet_map_2d\",\n"
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
      "          \"bounding_box\": {\"value\": "
      "[-73.984092,40.753893,-73.977588,40.756342]},\n"
      "          \"color_gradient\": {\"value\": [\"#0000FF\", \"#FF0000\"]},\n"
      "          \"cell_size\": {\"value\": 4.0},\n"
      "          \"cell_spacing\": {\"value\": 1.0},\n"
      "          \"opacity\": {\"value\": 1.0},\n"
      "          \"aggregation_type\": {\"value\": \"sum\"}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  auto wkb = arctern::render::WktToWkb(string_array);

  std::vector<std::shared_ptr<arrow::Array>> point_vec{wkb};
  std::vector<std::shared_ptr<arrow::Array>> color_vec{c_array};

  arctern::render::fishnet_map(point_vec, color_vec, vega);
}

TEST(FISHNETMAP_TEST, FLOAT) {
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
      "  \"width\": 1900,\n"
      "  \"height\": 1410,\n"
      "  \"description\": \"fishnet_map_2d\",\n"
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
      "          \"bounding_box\": {\"value\": "
      "[-73.984092,40.753893,-73.977588,40.756342]},\n"
      "          \"color_gradient\": {\"value\": [\"#0000FF\", \"#FF0000\"]},\n"
      "          \"cell_size\": {\"value\": 4.0},\n"
      "          \"cell_spacing\": {\"value\": 1.0},\n"
      "          \"opacity\": {\"value\": 1.0},\n"
      "          \"aggregation_type\": {\"value\": \"sum\"}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  auto wkb = arctern::render::WktToWkb(string_array);

  std::vector<std::shared_ptr<arrow::Array>> point_vec{wkb};
  std::vector<std::shared_ptr<arrow::Array>> color_vec{c_array};

  arctern::render::fishnet_map(point_vec, color_vec, vega);
}

TEST(FISHNETMAP_TEST, DOUBLE) {
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
      "  \"width\": 1900,\n"
      "  \"height\": 1410,\n"
      "  \"description\": \"fishnet_map_2d\",\n"
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
      "          \"bounding_box\": {\"value\": "
      "[-73.984092,40.753893,-73.977588,40.756342]},\n"
      "          \"color_gradient\": {\"value\": [\"#0000FF\", \"#FF0000\"]},\n"
      "          \"cell_size\": {\"value\": 4.0},\n"
      "          \"cell_spacing\": {\"value\": 1.0},\n"
      "          \"opacity\": {\"value\": 1.0},\n"
      "          \"aggregation_type\": {\"value\": \"sum\"}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  auto wkb = arctern::render::WktToWkb(string_array);

  std::vector<std::shared_ptr<arrow::Array>> point_vec{wkb};
  std::vector<std::shared_ptr<arrow::Array>> color_vec{c_array};

  arctern::render::fishnet_map(point_vec, color_vec, vega);
}
