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

TEST(HEATMAP_TEST, RAW_POINT_INT8_TEST) {
  auto bit_map = new uint8_t{0xff};

  auto data_type = arrow::uint32();

  auto buff_data1 = (uint32_t*)malloc(5 * sizeof(uint32_t));
  for (int i = 0; i < 5; ++i) {
    buff_data1[i] = i + 50;
  }
  auto buffer0 = std::make_shared<arrow::Buffer>(bit_map, 1 * sizeof(uint8_t));
  auto buffer1 =
      std::make_shared<arrow::Buffer>((uint8_t*)buff_data1, 5 * sizeof(uint32_t));
  std::vector<std::shared_ptr<arrow::Buffer>> buffers1;
  buffers1.emplace_back(buffer0);
  buffers1.emplace_back(buffer1);
  auto array_data1 = arrow::ArrayData::Make(data_type, 5, buffers1);
  auto array1 = arrow::MakeArray(array_data1);

  auto bit_map2 = new uint8_t{0xff};

  auto buff_data2 = (uint32_t*)malloc(5 * sizeof(uint32_t));
  for (int i = 0; i < 5; ++i) {
    buff_data2[i] = i + 50;
  }
  auto buffer20 = std::make_shared<arrow::Buffer>(bit_map2, 1 * sizeof(uint8_t));
  auto buffer21 =
      std::make_shared<arrow::Buffer>((uint8_t*)buff_data2, 5 * sizeof(uint32_t));
  std::vector<std::shared_ptr<arrow::Buffer>> buffers2;
  buffers2.emplace_back(buffer20);
  buffers2.emplace_back(buffer21);
  auto array_data2 = arrow::ArrayData::Make(arrow::uint32(), 5, buffers2);
  auto array2 = arrow::MakeArray(array_data2);

  auto bit_map3 = new uint8_t{0xff};

  auto buff_data3 = (int8_t*)malloc(5 * sizeof(int8_t));
  for (int i = 0; i < 5; ++i) {
    buff_data3[i] = i + 50;
  }
  auto buffer30 = std::make_shared<arrow::Buffer>(bit_map3, 1 * sizeof(uint8_t));
  auto buffer31 =
      std::make_shared<arrow::Buffer>((uint8_t*)buff_data3, 5 * sizeof(int8_t));
  std::vector<std::shared_ptr<arrow::Buffer>> buffers3;
  buffers3.emplace_back(buffer30);
  buffers3.emplace_back(buffer31);
  auto array_data3 = arrow::ArrayData::Make(arrow::int8(), 5, buffers3);
  auto array3 = arrow::MakeArray(array_data3);

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
      "          \"map_scale\": {\"value\": 10}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  arctern::render::heat_map(array1, array2, array3, vega);
}

TEST(HEATMAP_TEST, RAW_POINT_INT16_TEST) {
  auto bit_map = new uint8_t{0xff};

  auto data_type = arrow::uint32();

  auto buff_data1 = (uint32_t*)malloc(5 * sizeof(uint32_t));
  for (int i = 0; i < 5; ++i) {
    buff_data1[i] = i + 50;
  }
  auto buffer0 = std::make_shared<arrow::Buffer>(bit_map, 1 * sizeof(uint8_t));
  auto buffer1 =
      std::make_shared<arrow::Buffer>((uint8_t*)buff_data1, 5 * sizeof(uint32_t));
  std::vector<std::shared_ptr<arrow::Buffer>> buffers1;
  buffers1.emplace_back(buffer0);
  buffers1.emplace_back(buffer1);
  auto array_data1 = arrow::ArrayData::Make(data_type, 5, buffers1);
  auto array1 = arrow::MakeArray(array_data1);

  auto bit_map2 = new uint8_t{0xff};

  auto buff_data2 = (uint32_t*)malloc(5 * sizeof(uint32_t));
  for (int i = 0; i < 5; ++i) {
    buff_data2[i] = i + 50;
  }
  auto buffer20 = std::make_shared<arrow::Buffer>(bit_map2, 1 * sizeof(uint8_t));
  auto buffer21 =
      std::make_shared<arrow::Buffer>((uint8_t*)buff_data2, 5 * sizeof(uint32_t));
  std::vector<std::shared_ptr<arrow::Buffer>> buffers2;
  buffers2.emplace_back(buffer20);
  buffers2.emplace_back(buffer21);
  auto array_data2 = arrow::ArrayData::Make(arrow::uint32(), 5, buffers2);
  auto array2 = arrow::MakeArray(array_data2);

  auto bit_map3 = new uint8_t{0xff};

  auto buff_data3 = (int16_t*)malloc(5 * sizeof(int16_t));
  for (int i = 0; i < 5; ++i) {
    buff_data3[i] = i + 50;
  }
  auto buffer30 = std::make_shared<arrow::Buffer>(bit_map3, 1 * sizeof(uint8_t));
  auto buffer31 =
      std::make_shared<arrow::Buffer>((uint8_t*)buff_data3, 5 * sizeof(int16_t));
  std::vector<std::shared_ptr<arrow::Buffer>> buffers3;
  buffers3.emplace_back(buffer30);
  buffers3.emplace_back(buffer31);
  auto array_data3 = arrow::ArrayData::Make(arrow::int16(), 5, buffers3);
  auto array3 = arrow::MakeArray(array_data3);

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
      "          \"map_scale\": {\"value\": 10}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  arctern::render::heat_map(array1, array2, array3, vega);
}

TEST(HEATMAP_TEST, RAW_POINT_INT32_TEST) {
  auto bit_map = new uint8_t{0xff};

  auto data_type = arrow::uint32();

  auto buff_data1 = (uint32_t*)malloc(5 * sizeof(uint32_t));
  for (int i = 0; i < 5; ++i) {
    buff_data1[i] = i + 50;
  }
  auto buffer0 = std::make_shared<arrow::Buffer>(bit_map, 1 * sizeof(uint8_t));
  auto buffer1 =
      std::make_shared<arrow::Buffer>((uint8_t*)buff_data1, 5 * sizeof(uint32_t));
  std::vector<std::shared_ptr<arrow::Buffer>> buffers1;
  buffers1.emplace_back(buffer0);
  buffers1.emplace_back(buffer1);
  auto array_data1 = arrow::ArrayData::Make(data_type, 5, buffers1);
  auto array1 = arrow::MakeArray(array_data1);

  auto bit_map2 = new uint8_t{0xff};

  auto buff_data2 = (uint32_t*)malloc(5 * sizeof(uint32_t));
  for (int i = 0; i < 5; ++i) {
    buff_data2[i] = i + 50;
  }
  auto buffer20 = std::make_shared<arrow::Buffer>(bit_map2, 1 * sizeof(uint8_t));
  auto buffer21 =
      std::make_shared<arrow::Buffer>((uint8_t*)buff_data2, 5 * sizeof(uint32_t));
  std::vector<std::shared_ptr<arrow::Buffer>> buffers2;
  buffers2.emplace_back(buffer20);
  buffers2.emplace_back(buffer21);
  auto array_data2 = arrow::ArrayData::Make(data_type, 5, buffers2);
  auto array2 = arrow::MakeArray(array_data2);

  auto bit_map3 = new uint8_t{0xff};

  auto buff_data3 = (int32_t*)malloc(5 * sizeof(int32_t));
  for (int i = 0; i < 5; ++i) {
    buff_data3[i] = i + 50;
  }
  auto buffer30 = std::make_shared<arrow::Buffer>(bit_map3, 1 * sizeof(uint8_t));
  auto buffer31 =
      std::make_shared<arrow::Buffer>((uint8_t*)buff_data3, 5 * sizeof(int32_t));
  std::vector<std::shared_ptr<arrow::Buffer>> buffers3;
  buffers3.emplace_back(buffer30);
  buffers3.emplace_back(buffer31);
  auto array_data3 = arrow::ArrayData::Make(arrow::int32(), 5, buffers3);
  auto array3 = arrow::MakeArray(array_data3);

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
      "          \"map_scale\": {\"value\": 10}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  arctern::render::heat_map(array1, array2, array3, vega);
}

TEST(HEATMAP_TEST, RAW_POINT_INT64_TEST) {
  auto bit_map = new uint8_t{0xff};

  auto data_type = arrow::uint32();

  auto buff_data1 = (uint32_t*)malloc(5 * sizeof(uint32_t));
  for (int i = 0; i < 5; ++i) {
    buff_data1[i] = i + 50;
  }
  auto buffer0 = std::make_shared<arrow::Buffer>(bit_map, 1 * sizeof(uint8_t));
  auto buffer1 =
      std::make_shared<arrow::Buffer>((uint8_t*)buff_data1, 5 * sizeof(uint32_t));
  std::vector<std::shared_ptr<arrow::Buffer>> buffers1;
  buffers1.emplace_back(buffer0);
  buffers1.emplace_back(buffer1);
  auto array_data1 = arrow::ArrayData::Make(data_type, 5, buffers1);
  auto array1 = arrow::MakeArray(array_data1);

  auto bit_map2 = new uint8_t{0xff};

  auto buff_data2 = (uint32_t*)malloc(5 * sizeof(uint32_t));
  for (int i = 0; i < 5; ++i) {
    buff_data2[i] = i + 50;
  }
  auto buffer20 = std::make_shared<arrow::Buffer>(bit_map2, 1 * sizeof(uint8_t));
  auto buffer21 =
      std::make_shared<arrow::Buffer>((uint8_t*)buff_data2, 5 * sizeof(uint32_t));
  std::vector<std::shared_ptr<arrow::Buffer>> buffers2;
  buffers2.emplace_back(buffer20);
  buffers2.emplace_back(buffer21);
  auto array_data2 = arrow::ArrayData::Make(arrow::uint32(), 5, buffers2);
  auto array2 = arrow::MakeArray(array_data2);

  auto bit_map3 = new uint8_t{0xff};

  auto buff_data3 = (int64_t*)malloc(5 * sizeof(int64_t));
  for (int i = 0; i < 5; ++i) {
    buff_data3[i] = i + 50;
  }
  auto buffer30 = std::make_shared<arrow::Buffer>(bit_map3, 1 * sizeof(uint8_t));
  auto buffer31 =
      std::make_shared<arrow::Buffer>((uint8_t*)buff_data3, 5 * sizeof(int64_t));
  std::vector<std::shared_ptr<arrow::Buffer>> buffers3;
  buffers3.emplace_back(buffer30);
  buffers3.emplace_back(buffer31);
  auto array_data3 = arrow::ArrayData::Make(arrow::int64(), 5, buffers3);
  auto array3 = arrow::MakeArray(array_data3);

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
      "          \"map_scale\": {\"value\": 10}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  arctern::render::heat_map(array1, array2, array3, vega);
}

TEST(HEATMAP_TEST, RAW_POINT_UINT8_TEST) {
  auto bit_map = new uint8_t{0xff};

  auto data_type = arrow::uint32();

  auto buff_data1 = (uint32_t*)malloc(5 * sizeof(uint32_t));
  for (int i = 0; i < 5; ++i) {
    buff_data1[i] = i + 50;
  }
  auto buffer0 = std::make_shared<arrow::Buffer>(bit_map, 1 * sizeof(uint8_t));
  auto buffer1 =
      std::make_shared<arrow::Buffer>((uint8_t*)buff_data1, 5 * sizeof(uint32_t));
  std::vector<std::shared_ptr<arrow::Buffer>> buffers1;
  buffers1.emplace_back(buffer0);
  buffers1.emplace_back(buffer1);
  auto array_data1 = arrow::ArrayData::Make(data_type, 5, buffers1);
  auto array1 = arrow::MakeArray(array_data1);

  auto bit_map2 = new uint8_t{0xff};

  auto buff_data2 = (uint32_t*)malloc(5 * sizeof(uint32_t));
  for (int i = 0; i < 5; ++i) {
    buff_data2[i] = i + 50;
  }
  auto buffer20 = std::make_shared<arrow::Buffer>(bit_map2, 1 * sizeof(uint8_t));
  auto buffer21 =
      std::make_shared<arrow::Buffer>((uint8_t*)buff_data2, 5 * sizeof(uint32_t));
  std::vector<std::shared_ptr<arrow::Buffer>> buffers2;
  buffers2.emplace_back(buffer20);
  buffers2.emplace_back(buffer21);
  auto array_data2 = arrow::ArrayData::Make(arrow::uint32(), 5, buffers2);
  auto array2 = arrow::MakeArray(array_data2);

  auto bit_map3 = new uint8_t{0xff};

  auto buff_data3 = (uint8_t*)malloc(5 * sizeof(uint8_t));
  for (int i = 0; i < 5; ++i) {
    buff_data3[i] = i + 50;
  }
  auto buffer30 = std::make_shared<arrow::Buffer>(bit_map3, 1 * sizeof(uint8_t));
  auto buffer31 =
      std::make_shared<arrow::Buffer>((uint8_t*)buff_data3, 5 * sizeof(uint8_t));
  std::vector<std::shared_ptr<arrow::Buffer>> buffers3;
  buffers3.emplace_back(buffer30);
  buffers3.emplace_back(buffer31);
  auto array_data3 = arrow::ArrayData::Make(arrow::uint8(), 5, buffers3);
  auto array3 = arrow::MakeArray(array_data3);

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
      "          \"map_scale\": {\"value\": 10}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  arctern::render::heat_map(array1, array2, array3, vega);
}

TEST(HEATMAP_TEST, RAW_POINT_UINT16_TEST) {
  auto bit_map = new uint8_t{0xff};

  auto data_type = arrow::uint32();

  auto buff_data1 = (uint32_t*)malloc(5 * sizeof(uint32_t));
  for (int i = 0; i < 5; ++i) {
    buff_data1[i] = i + 50;
  }
  auto buffer0 = std::make_shared<arrow::Buffer>(bit_map, 1 * sizeof(uint8_t));
  auto buffer1 =
      std::make_shared<arrow::Buffer>((uint8_t*)buff_data1, 5 * sizeof(uint32_t));
  std::vector<std::shared_ptr<arrow::Buffer>> buffers1;
  buffers1.emplace_back(buffer0);
  buffers1.emplace_back(buffer1);
  auto array_data1 = arrow::ArrayData::Make(data_type, 5, buffers1);
  auto array1 = arrow::MakeArray(array_data1);

  auto bit_map2 = new uint8_t{0xff};

  auto buff_data2 = (uint32_t*)malloc(5 * sizeof(uint32_t));
  for (int i = 0; i < 5; ++i) {
    buff_data2[i] = i + 50;
  }
  auto buffer20 = std::make_shared<arrow::Buffer>(bit_map2, 1 * sizeof(uint8_t));
  auto buffer21 =
      std::make_shared<arrow::Buffer>((uint8_t*)buff_data2, 5 * sizeof(uint32_t));
  std::vector<std::shared_ptr<arrow::Buffer>> buffers2;
  buffers2.emplace_back(buffer20);
  buffers2.emplace_back(buffer21);
  auto array_data2 = arrow::ArrayData::Make(arrow::uint32(), 5, buffers2);
  auto array2 = arrow::MakeArray(array_data2);

  auto bit_map3 = new uint8_t{0xff};

  auto buff_data3 = (uint16_t*)malloc(5 * sizeof(uint16_t));
  for (int i = 0; i < 5; ++i) {
    buff_data3[i] = i + 50;
  }
  auto buffer30 = std::make_shared<arrow::Buffer>(bit_map3, 1 * sizeof(uint8_t));
  auto buffer31 =
      std::make_shared<arrow::Buffer>((uint8_t*)buff_data3, 5 * sizeof(uint16_t));
  std::vector<std::shared_ptr<arrow::Buffer>> buffers3;
  buffers3.emplace_back(buffer30);
  buffers3.emplace_back(buffer31);
  auto array_data3 = arrow::ArrayData::Make(arrow::uint16(), 5, buffers3);
  auto array3 = arrow::MakeArray(array_data3);

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
      "          \"map_scale\": {\"value\": 10}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  arctern::render::heat_map(array1, array2, array3, vega);
}

TEST(HEATMAP_TEST, RAW_POINT_UINT32_TEST) {
  auto bit_map = new uint8_t{0xff};

  auto data_type = arrow::uint32();

  auto buff_data1 = (uint32_t*)malloc(5 * sizeof(uint32_t));
  for (int i = 0; i < 5; ++i) {
    buff_data1[i] = i + 50;
  }
  auto buffer0 = std::make_shared<arrow::Buffer>(bit_map, 1 * sizeof(uint8_t));
  auto buffer1 =
      std::make_shared<arrow::Buffer>((uint8_t*)buff_data1, 5 * sizeof(uint32_t));
  std::vector<std::shared_ptr<arrow::Buffer>> buffers1;
  buffers1.emplace_back(buffer0);
  buffers1.emplace_back(buffer1);
  auto array_data1 = arrow::ArrayData::Make(data_type, 5, buffers1);
  auto array1 = arrow::MakeArray(array_data1);

  auto bit_map2 = new uint8_t{0xff};

  auto buff_data2 = (uint32_t*)malloc(5 * sizeof(uint32_t));
  for (int i = 0; i < 5; ++i) {
    buff_data2[i] = i + 50;
  }
  auto buffer20 = std::make_shared<arrow::Buffer>(bit_map2, 1 * sizeof(uint8_t));
  auto buffer21 =
      std::make_shared<arrow::Buffer>((uint8_t*)buff_data2, 5 * sizeof(uint32_t));
  std::vector<std::shared_ptr<arrow::Buffer>> buffers2;
  buffers2.emplace_back(buffer20);
  buffers2.emplace_back(buffer21);
  auto array_data2 = arrow::ArrayData::Make(arrow::uint32(), 5, buffers2);
  auto array2 = arrow::MakeArray(array_data2);

  auto bit_map3 = new uint8_t{0xff};

  auto buff_data3 = (uint32_t*)malloc(5 * sizeof(uint32_t));
  for (int i = 0; i < 5; ++i) {
    buff_data3[i] = i + 50;
  }
  auto buffer30 = std::make_shared<arrow::Buffer>(bit_map3, 1 * sizeof(uint8_t));
  auto buffer31 =
      std::make_shared<arrow::Buffer>((uint8_t*)buff_data3, 5 * sizeof(uint32_t));
  std::vector<std::shared_ptr<arrow::Buffer>> buffers3;
  buffers3.emplace_back(buffer30);
  buffers3.emplace_back(buffer31);
  auto array_data3 = arrow::ArrayData::Make(arrow::uint32(), 5, buffers3);
  auto array3 = arrow::MakeArray(array_data3);

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
      "          \"map_scale\": {\"value\": 10}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  arctern::render::heat_map(array1, array2, array3, vega);
}

TEST(HEATMAP_TEST, RAW_POINT_UINT64_TEST) {
  auto bit_map = new uint8_t{0xff};

  auto data_type = arrow::uint32();

  auto buff_data1 = (uint32_t*)malloc(5 * sizeof(uint32_t));
  for (int i = 0; i < 5; ++i) {
    buff_data1[i] = i + 50;
  }
  auto buffer0 = std::make_shared<arrow::Buffer>(bit_map, 1 * sizeof(uint8_t));
  auto buffer1 =
      std::make_shared<arrow::Buffer>((uint8_t*)buff_data1, 5 * sizeof(uint32_t));
  std::vector<std::shared_ptr<arrow::Buffer>> buffers1;
  buffers1.emplace_back(buffer0);
  buffers1.emplace_back(buffer1);
  auto array_data1 = arrow::ArrayData::Make(data_type, 5, buffers1);
  auto array1 = arrow::MakeArray(array_data1);

  auto bit_map2 = new uint8_t{0xff};

  auto buff_data2 = (uint32_t*)malloc(5 * sizeof(uint32_t));
  for (int i = 0; i < 5; ++i) {
    buff_data2[i] = i + 50;
  }
  auto buffer20 = std::make_shared<arrow::Buffer>(bit_map2, 1 * sizeof(uint8_t));
  auto buffer21 =
      std::make_shared<arrow::Buffer>((uint8_t*)buff_data2, 5 * sizeof(uint32_t));
  std::vector<std::shared_ptr<arrow::Buffer>> buffers2;
  buffers2.emplace_back(buffer20);
  buffers2.emplace_back(buffer21);
  auto array_data2 = arrow::ArrayData::Make(arrow::uint32(), 5, buffers2);
  auto array2 = arrow::MakeArray(array_data2);

  auto bit_map3 = new uint8_t{0xff};

  auto buff_data3 = (uint64_t*)malloc(5 * sizeof(uint64_t));
  for (int i = 0; i < 5; ++i) {
    buff_data3[i] = i + 50;
  }
  auto buffer30 = std::make_shared<arrow::Buffer>(bit_map3, 1 * sizeof(uint8_t));
  auto buffer31 =
      std::make_shared<arrow::Buffer>((uint8_t*)buff_data3, 5 * sizeof(uint64_t));
  std::vector<std::shared_ptr<arrow::Buffer>> buffers3;
  buffers3.emplace_back(buffer30);
  buffers3.emplace_back(buffer31);
  auto array_data3 = arrow::ArrayData::Make(arrow::uint64(), 5, buffers3);
  auto array3 = arrow::MakeArray(array_data3);

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
      "          \"map_scale\": {\"value\": 10}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  arctern::render::heat_map(array1, array2, array3, vega);
}

TEST(HEATMAP_TEST, RAW_POINT_FLOAT_TEST) {
  auto bit_map = new uint8_t{0xff};

  auto data_type = arrow::uint32();

  auto buff_data1 = (uint32_t*)malloc(5 * sizeof(uint32_t));
  for (int i = 0; i < 5; ++i) {
    buff_data1[i] = i + 50;
  }
  auto buffer0 = std::make_shared<arrow::Buffer>(bit_map, 1 * sizeof(uint8_t));
  auto buffer1 =
      std::make_shared<arrow::Buffer>((uint8_t*)buff_data1, 5 * sizeof(uint32_t));
  std::vector<std::shared_ptr<arrow::Buffer>> buffers1;
  buffers1.emplace_back(buffer0);
  buffers1.emplace_back(buffer1);
  auto array_data1 = arrow::ArrayData::Make(data_type, 5, buffers1);
  auto array1 = arrow::MakeArray(array_data1);

  auto bit_map2 = new uint8_t{0xff};

  auto buff_data2 = (uint32_t*)malloc(5 * sizeof(uint32_t));
  for (int i = 0; i < 5; ++i) {
    buff_data2[i] = i + 50;
  }
  auto buffer20 = std::make_shared<arrow::Buffer>(bit_map2, 1 * sizeof(uint8_t));
  auto buffer21 =
      std::make_shared<arrow::Buffer>((uint8_t*)buff_data2, 5 * sizeof(uint32_t));
  std::vector<std::shared_ptr<arrow::Buffer>> buffers2;
  buffers2.emplace_back(buffer20);
  buffers2.emplace_back(buffer21);
  auto array_data2 = arrow::ArrayData::Make(arrow::uint32(), 5, buffers2);
  auto array2 = arrow::MakeArray(array_data2);

  auto bit_map3 = new uint8_t{0xff};

  auto buff_data3 = (float*)malloc(5 * sizeof(float));
  for (int i = 0; i < 5; ++i) {
    buff_data3[i] = i + 50;
  }
  auto buffer30 = std::make_shared<arrow::Buffer>(bit_map3, 1 * sizeof(uint8_t));
  auto buffer31 =
      std::make_shared<arrow::Buffer>((uint8_t*)buff_data3, 5 * sizeof(float));
  std::vector<std::shared_ptr<arrow::Buffer>> buffers3;
  buffers3.emplace_back(buffer30);
  buffers3.emplace_back(buffer31);
  auto array_data3 = arrow::ArrayData::Make(arrow::float32(), 5, buffers3);
  auto array3 = arrow::MakeArray(array_data3);

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
      "          \"map_scale\": {\"value\": 10}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  arctern::render::heat_map(array1, array2, array3, vega);
}

TEST(HEATMAP_TEST, RAW_POINT_DOUBLE_TEST) {
  auto bit_map = new uint8_t{0xff};

  auto data_type = arrow::uint32();

  auto buff_data1 = (uint32_t*)malloc(5 * sizeof(uint32_t));
  for (int i = 0; i < 5; ++i) {
    buff_data1[i] = i + 50;
  }
  auto buffer0 = std::make_shared<arrow::Buffer>(bit_map, 1 * sizeof(uint8_t));
  auto buffer1 =
      std::make_shared<arrow::Buffer>((uint8_t*)buff_data1, 5 * sizeof(uint32_t));
  std::vector<std::shared_ptr<arrow::Buffer>> buffers1;
  buffers1.emplace_back(buffer0);
  buffers1.emplace_back(buffer1);
  auto array_data1 = arrow::ArrayData::Make(data_type, 5, buffers1);
  auto array1 = arrow::MakeArray(array_data1);

  auto bit_map2 = new uint8_t{0xff};

  auto buff_data2 = (uint32_t*)malloc(5 * sizeof(uint32_t));
  for (int i = 0; i < 5; ++i) {
    buff_data2[i] = i + 50;
  }
  auto buffer20 = std::make_shared<arrow::Buffer>(bit_map2, 1 * sizeof(uint8_t));
  auto buffer21 =
      std::make_shared<arrow::Buffer>((uint8_t*)buff_data2, 5 * sizeof(uint32_t));
  std::vector<std::shared_ptr<arrow::Buffer>> buffers2;
  buffers2.emplace_back(buffer20);
  buffers2.emplace_back(buffer21);
  auto array_data2 = arrow::ArrayData::Make(arrow::uint32(), 5, buffers2);
  auto array2 = arrow::MakeArray(array_data2);

  auto bit_map3 = new uint8_t{0xff};

  auto buff_data3 = (double*)malloc(5 * sizeof(double));
  for (int i = 0; i < 5; ++i) {
    buff_data3[i] = i + 50;
  }
  auto buffer30 = std::make_shared<arrow::Buffer>(bit_map3, 1 * sizeof(uint8_t));
  auto buffer31 =
      std::make_shared<arrow::Buffer>((uint8_t*)buff_data3, 5 * sizeof(double));
  std::vector<std::shared_ptr<arrow::Buffer>> buffers3;
  buffers3.emplace_back(buffer30);
  buffers3.emplace_back(buffer31);
  auto array_data3 = arrow::ArrayData::Make(arrow::float64(), 5, buffers3);
  auto array3 = arrow::MakeArray(array_data3);

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
      "          \"map_scale\": {\"value\": 10}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  arctern::render::heat_map(array1, array2, array3, vega);
}

TEST(HEATMAP_TEST, RAW_POINT_INVALID_DATA_TYPE_TEST) {
  // param1: x
  arrow::UInt32Builder x_builder;
  auto status = x_builder.Append(50);
  status = x_builder.Append(50);
  status = x_builder.Append(50);
  status = x_builder.Append(50);
  status = x_builder.Append(50);

  std::shared_ptr<arrow::UInt32Array> x_array;
  status = x_builder.Finish(&x_array);

  // param2: y
  arrow::UInt32Builder y_builder;
  status = y_builder.Append(50);
  status = y_builder.Append(50);
  status = y_builder.Append(50);
  status = y_builder.Append(50);
  status = y_builder.Append(50);

  std::shared_ptr<arrow::UInt32Array> y_array;
  status = y_builder.Finish(&y_array);

  // param2: color
  std::shared_ptr<arrow::Array> color_array;
  arrow::StringBuilder color_builder;
  status = color_builder.Append("");
  status = color_builder.Append("");
  status = color_builder.Append("");
  status = color_builder.Append("");
  status = color_builder.Append("");
  status = color_builder.Finish(&color_array);

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
      "          \"map_scale\": {\"value\": 10}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  arctern::render::heat_map(x_array, y_array, color_array, vega);
}

TEST(HEATMAP_TEST, WKT_POINT_INT8_TEST) {
  // param1: wkt string
  std::string wkt1 = "POINT (50 50)";
  std::string wkt2 = "POINT (51 51)";
  std::string wkt3 = "POINT (52 52)";
  std::string wkt4 = "POINT (53 53)";
  std::string wkt5 = "POINT (54 54)";
  arrow::StringBuilder string_builder;
  auto status = string_builder.Append(wkt1);
  status = string_builder.Append(wkt2);
  status = string_builder.Append(wkt3);
  status = string_builder.Append(wkt4);
  status = string_builder.Append(wkt5);

  std::shared_ptr<arrow::StringArray> string_array;
  status = string_builder.Finish(&string_array);

  // param2: color
  std::shared_ptr<arrow::Array> color_array;
  arrow::Int8Builder color_builder;
  status = color_builder.Append(50);
  status = color_builder.Append(51);
  status = color_builder.Append(52);
  status = color_builder.Append(53);
  status = color_builder.Append(54);
  status = color_builder.Finish(&color_array);

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
      "          \"map_scale\": {\"value\": 10}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  auto wkb = arctern::render::WktToWkb(string_array);
  arctern::render::heat_map(wkb, color_array, vega);
}

TEST(HEATMAP_TEST, WKT_POINT_INT16_TEST) {
  // param1: wkt string
  std::string wkt1 = "POINT (50 50)";
  std::string wkt2 = "POINT (51 51)";
  std::string wkt3 = "POINT (52 52)";
  std::string wkt4 = "POINT (53 53)";
  std::string wkt5 = "POINT (54 54)";
  arrow::StringBuilder string_builder;
  auto status = string_builder.Append(wkt1);
  status = string_builder.Append(wkt2);
  status = string_builder.Append(wkt3);
  status = string_builder.Append(wkt4);
  status = string_builder.Append(wkt5);

  std::shared_ptr<arrow::StringArray> string_array;
  status = string_builder.Finish(&string_array);

  // param2: color
  std::shared_ptr<arrow::Array> color_array;
  arrow::Int16Builder color_builder;
  status = color_builder.Append(50);
  status = color_builder.Append(51);
  status = color_builder.Append(52);
  status = color_builder.Append(53);
  status = color_builder.Append(54);
  status = color_builder.Finish(&color_array);

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
      "          \"map_scale\": {\"value\": 10}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  auto wkb = arctern::render::WktToWkb(string_array);
  arctern::render::heat_map(wkb, color_array, vega);
}

TEST(HEATMAP_TEST, WKT_POINT_INT32_TEST) {
  // param1: wkt string
  std::string wkt1 = "POINT (50 50)";
  std::string wkt2 = "POINT (51 51)";
  std::string wkt3 = "POINT (52 52)";
  std::string wkt4 = "POINT (53 53)";
  std::string wkt5 = "POINT (54 54)";
  arrow::StringBuilder string_builder;
  auto status = string_builder.Append(wkt1);
  status = string_builder.Append(wkt2);
  status = string_builder.Append(wkt3);
  status = string_builder.Append(wkt4);
  status = string_builder.Append(wkt5);

  std::shared_ptr<arrow::StringArray> string_array;
  status = string_builder.Finish(&string_array);

  // param2: color
  std::shared_ptr<arrow::Array> color_array;
  arrow::Int32Builder color_builder;
  status = color_builder.Append(50);
  status = color_builder.Append(51);
  status = color_builder.Append(52);
  status = color_builder.Append(53);
  status = color_builder.Append(54);
  status = color_builder.Finish(&color_array);

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
      "          \"map_scale\": {\"value\": 10}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  auto wkb = arctern::render::WktToWkb(string_array);
  arctern::render::heat_map(wkb, color_array, vega);
}

TEST(HEATMAP_TEST, WKT_POINT_INT64_TEST) {
  // param1: wkt string
  std::string wkt1 = "POINT (50 50)";
  std::string wkt2 = "POINT (51 51)";
  std::string wkt3 = "POINT (52 52)";
  std::string wkt4 = "POINT (53 53)";
  std::string wkt5 = "POINT (54 54)";
  arrow::StringBuilder string_builder;
  auto status = string_builder.Append(wkt1);
  status = string_builder.Append(wkt2);
  status = string_builder.Append(wkt3);
  status = string_builder.Append(wkt4);
  status = string_builder.Append(wkt5);

  std::shared_ptr<arrow::StringArray> string_array;
  status = string_builder.Finish(&string_array);

  // param2: color
  std::shared_ptr<arrow::Array> color_array;
  arrow::Int64Builder color_builder;
  status = color_builder.Append(50);
  status = color_builder.Append(51);
  status = color_builder.Append(52);
  status = color_builder.Append(53);
  status = color_builder.Append(54);
  status = color_builder.Finish(&color_array);

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
      "          \"map_scale\": {\"value\": 10}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  auto wkb = arctern::render::WktToWkb(string_array);
  arctern::render::heat_map(wkb, color_array, vega);
}

TEST(HEATMAP_TEST, WKT_POINT_UINT8_TEST) {
  // param1: wkt string
  std::string wkt1 = "POINT (50 50)";
  std::string wkt2 = "POINT (51 51)";
  std::string wkt3 = "POINT (52 52)";
  std::string wkt4 = "POINT (53 53)";
  std::string wkt5 = "POINT (54 54)";
  arrow::StringBuilder string_builder;
  auto status = string_builder.Append(wkt1);
  status = string_builder.Append(wkt2);
  status = string_builder.Append(wkt3);
  status = string_builder.Append(wkt4);
  status = string_builder.Append(wkt5);

  std::shared_ptr<arrow::StringArray> string_array;
  status = string_builder.Finish(&string_array);

  // param2: color
  std::shared_ptr<arrow::Array> color_array;
  arrow::UInt8Builder color_builder;
  status = color_builder.Append(50);
  status = color_builder.Append(51);
  status = color_builder.Append(52);
  status = color_builder.Append(53);
  status = color_builder.Append(54);
  status = color_builder.Finish(&color_array);

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
      "          \"map_scale\": {\"value\": 10}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  auto wkb = arctern::render::WktToWkb(string_array);
  arctern::render::heat_map(wkb, color_array, vega);
}

TEST(HEATMAP_TEST, WKT_POINT_UINT16_TEST) {
  // param1: wkt string
  std::string wkt1 = "POINT (50 50)";
  std::string wkt2 = "POINT (51 51)";
  std::string wkt3 = "POINT (52 52)";
  std::string wkt4 = "POINT (53 53)";
  std::string wkt5 = "POINT (54 54)";
  arrow::StringBuilder string_builder;
  auto status = string_builder.Append(wkt1);
  status = string_builder.Append(wkt2);
  status = string_builder.Append(wkt3);
  status = string_builder.Append(wkt4);
  status = string_builder.Append(wkt5);

  std::shared_ptr<arrow::StringArray> string_array;
  status = string_builder.Finish(&string_array);

  // param2: color
  std::shared_ptr<arrow::Array> color_array;
  arrow::UInt16Builder color_builder;
  status = color_builder.Append(50);
  status = color_builder.Append(51);
  status = color_builder.Append(52);
  status = color_builder.Append(53);
  status = color_builder.Append(54);
  status = color_builder.Finish(&color_array);

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
      "          \"map_scale\": {\"value\": 10}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  auto wkb = arctern::render::WktToWkb(string_array);
  arctern::render::heat_map(wkb, color_array, vega);
}

TEST(HEATMAP_TEST, WKT_POINT_UINT32_TEST) {
  // param1: wkt string
  std::string wkt1 = "POINT (50 50)";
  std::string wkt2 = "POINT (51 51)";
  std::string wkt3 = "POINT (52 52)";
  std::string wkt4 = "POINT (53 53)";
  std::string wkt5 = "POINT (54 54)";
  arrow::StringBuilder string_builder;
  auto status = string_builder.Append(wkt1);
  status = string_builder.Append(wkt2);
  status = string_builder.Append(wkt3);
  status = string_builder.Append(wkt4);
  status = string_builder.Append(wkt5);

  std::shared_ptr<arrow::StringArray> string_array;
  status = string_builder.Finish(&string_array);

  // param2: color
  std::shared_ptr<arrow::Array> color_array;
  arrow::UInt32Builder color_builder;
  status = color_builder.Append(50);
  status = color_builder.Append(51);
  status = color_builder.Append(52);
  status = color_builder.Append(53);
  status = color_builder.Append(54);
  status = color_builder.Finish(&color_array);

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
      "          \"map_scale\": {\"value\": 10}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  auto wkb = arctern::render::WktToWkb(string_array);
  arctern::render::heat_map(wkb, color_array, vega);
}

TEST(HEATMAP_TEST, WKT_POINT_UINT64_TEST) {
  // param1: wkt string
  std::string wkt1 = "POINT (50 50)";
  std::string wkt2 = "POINT (51 51)";
  std::string wkt3 = "POINT (52 52)";
  std::string wkt4 = "POINT (53 53)";
  std::string wkt5 = "POINT (54 54)";
  arrow::StringBuilder string_builder;
  auto status = string_builder.Append(wkt1);
  status = string_builder.Append(wkt2);
  status = string_builder.Append(wkt3);
  status = string_builder.Append(wkt4);
  status = string_builder.Append(wkt5);

  std::shared_ptr<arrow::StringArray> string_array;
  status = string_builder.Finish(&string_array);

  // param2: color
  std::shared_ptr<arrow::Array> color_array;
  arrow::UInt64Builder color_builder;
  status = color_builder.Append(50);
  status = color_builder.Append(51);
  status = color_builder.Append(52);
  status = color_builder.Append(53);
  status = color_builder.Append(54);
  status = color_builder.Finish(&color_array);

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
      "          \"map_scale\": {\"value\": 10}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  auto wkb = arctern::render::WktToWkb(string_array);
  arctern::render::heat_map(wkb, color_array, vega);
}

TEST(HEATMAP_TEST, WKT_POINT_FLOAT_TEST) {
  // param1: wkt string
  std::string wkt1 = "POINT (50 50)";
  std::string wkt2 = "POINT (51 51)";
  std::string wkt3 = "POINT (52 52)";
  std::string wkt4 = "POINT (53 53)";
  std::string wkt5 = "POINT (54 54)";
  arrow::StringBuilder string_builder;
  auto status = string_builder.Append(wkt1);
  status = string_builder.Append(wkt2);
  status = string_builder.Append(wkt3);
  status = string_builder.Append(wkt4);
  status = string_builder.Append(wkt5);

  std::shared_ptr<arrow::StringArray> string_array;
  status = string_builder.Finish(&string_array);

  // param2: color
  std::shared_ptr<arrow::Array> color_array;
  arrow::FloatBuilder color_builder;
  status = color_builder.Append(50);
  status = color_builder.Append(51);
  status = color_builder.Append(52);
  status = color_builder.Append(53);
  status = color_builder.Append(54);
  status = color_builder.Finish(&color_array);

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
      "          \"map_scale\": {\"value\": 10}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  auto wkb = arctern::render::WktToWkb(string_array);
  arctern::render::heat_map(wkb, color_array, vega);
}

TEST(HEATMAP_TEST, WKT_POINT_DOUBLE_TEST) {
  // param1: wkt string
  std::string wkt1 = "POINT (50 50)";
  std::string wkt2 = "POINT (51 51)";
  std::string wkt3 = "POINT (52 52)";
  std::string wkt4 = "POINT (53 53)";
  std::string wkt5 = "POINT (54 54)";
  arrow::StringBuilder string_builder;
  auto status = string_builder.Append(wkt1);
  status = string_builder.Append(wkt2);
  status = string_builder.Append(wkt3);
  status = string_builder.Append(wkt4);
  status = string_builder.Append(wkt5);

  std::shared_ptr<arrow::StringArray> string_array;
  status = string_builder.Finish(&string_array);

  // param2: color
  std::shared_ptr<arrow::Array> color_array;
  arrow::DoubleBuilder color_builder;
  status = color_builder.Append(50);
  status = color_builder.Append(51);
  status = color_builder.Append(52);
  status = color_builder.Append(53);
  status = color_builder.Append(54);
  status = color_builder.Finish(&color_array);

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
      "          \"map_scale\": {\"value\": 10}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  auto wkb = arctern::render::WktToWkb(string_array);
  arctern::render::heat_map(wkb, color_array, vega);
}

TEST(HEATMAP_TEST, WKT_POINT_INVALID_DATA_TYPE_TEST) {
  // param1: wkt string
  std::string wkt1 = "POINT (50 50)";
  std::string wkt2 = "POINT (51 51)";
  std::string wkt3 = "POINT (52 52)";
  std::string wkt4 = "POINT (53 53)";
  std::string wkt5 = "POINT (54 54)";
  arrow::StringBuilder string_builder;
  auto status = string_builder.Append(wkt1);
  status = string_builder.Append(wkt2);
  status = string_builder.Append(wkt3);
  status = string_builder.Append(wkt4);
  status = string_builder.Append(wkt5);

  std::shared_ptr<arrow::StringArray> string_array;
  status = string_builder.Finish(&string_array);

  // param2: color
  std::shared_ptr<arrow::Array> color_array;
  arrow::StringBuilder color_builder;
  status = color_builder.Append("");
  status = color_builder.Append("");
  status = color_builder.Append("");
  status = color_builder.Append("");
  status = color_builder.Append("");
  status = color_builder.Finish(&color_array);

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
      "          \"map_scale\": {\"value\": 10}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  auto wkb = arctern::render::WktToWkb(string_array);
  arctern::render::heat_map(wkb, color_array, vega);
}

TEST(HEATMAP_TEST, INVALID_JSON_TEST) {
  // param1: wkt string
  std::string wkt1 = "POINT (50 50)";
  std::string wkt2 = "POINT (51 51)";
  std::string wkt3 = "POINT (52 52)";
  std::string wkt4 = "POINT (53 53)";
  std::string wkt5 = "POINT (54 54)";
  arrow::StringBuilder string_builder;
  auto status = string_builder.Append(wkt1);
  status = string_builder.Append(wkt2);
  status = string_builder.Append(wkt3);
  status = string_builder.Append(wkt4);
  status = string_builder.Append(wkt5);

  std::shared_ptr<arrow::StringArray> string_array;
  status = string_builder.Finish(&string_array);

  // param2: color
  std::shared_ptr<arrow::Array> color_array;
  arrow::DoubleBuilder color_builder;
  status = color_builder.Append(50);
  status = color_builder.Append(51);
  status = color_builder.Append(52);
  status = color_builder.Append(53);
  status = color_builder.Append(54);
  status = color_builder.Finish(&color_array);

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
      "          \"map_scale\": {\"value\": \"INVALID_NUMBER\"}\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  ]\n"
      "}";

  auto wkb = arctern::render::WktToWkb(string_array);
  arctern::render::heat_map(wkb, color_array, vega);
}
