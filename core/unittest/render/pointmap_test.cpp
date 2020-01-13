#include <gtest/gtest.h>
#include <iostream>
#include "render/2d/render_builder.h"
#include "render/2d/pointmap.h"

using namespace zilliz::render;

TEST(CIRCLE_2D_TEST, SINGLE_COLOR_TEST) {
//    PointMap point_map;
//
//    int64_t length = 1024;
    auto bit_map = new uint8_t{0xff};

    auto data_type = arrow::uint32();

    auto buff_data1 = (uint32_t* )malloc(5 * sizeof(uint32_t));
    for (int i = 0; i < 5; ++i) {
        buff_data1[i] = i + 50;
    }
    auto buffer0 = std::make_shared<arrow::Buffer>(bit_map, 1 * sizeof(uint8_t));
    auto buffer1 = std::make_shared<arrow::Buffer>((uint8_t *)buff_data1, 5 * sizeof(uint32_t));
    std::vector<std::shared_ptr<arrow::Buffer>> buffers1;
    buffers1.emplace_back(buffer0);
    buffers1.emplace_back(buffer1);
    auto array_data1 = arrow::ArrayData::Make(data_type, 5 * sizeof(uint32_t), buffers1);
    auto array1 = arrow::MakeArray(array_data1);

    auto bit_map2 = new uint8_t{0xff};

    auto buff_data2 = (uint32_t* )malloc(5 * sizeof(uint32_t));
    for (int i = 0; i < 5; ++i) {
        buff_data2[i] = i + 50;
    }
    auto buffer20 = std::make_shared<arrow::Buffer>(bit_map2, 1 * sizeof(uint8_t));
    auto buffer21 = std::make_shared<arrow::Buffer>((uint8_t *)buff_data2, 5 * sizeof(uint32_t));
    std::vector<std::shared_ptr<arrow::Buffer>> buffers2;
    buffers2.emplace_back(buffer20);
    buffers2.emplace_back(buffer21);
    auto array_data2 = arrow::ArrayData::Make(arrow::uint32(), 5 * sizeof(uint32_t), buffers2);
    auto array2 = arrow::MakeArray(array_data2);

    std::vector<std::shared_ptr<arrow::Array>> array_vector;
    array_vector.emplace_back(array1);
    array_vector.emplace_back(array2);
//
//    Input input;
//    input.array_vector = array_vector;
//    input.vega = "{\n"
//                 "  \"width\": 300,\n"
//                 "  \"height\": 200,\n"
//                 "  \"description\": \"circle_2d\",\n"
//                 "  \"data\": [\n"
//                 "    {\n"
//                 "      \"name\": \"data\",\n"
//                 "      \"url\": \"data/data.csv\"\n"
//                 "    }\n"
//                 "  ],\n"
//                 "  \"scales\": [\n"
//                 "    {\n"
//                 "      \"name\": \"x\",\n"
//                 "      \"type\": \"linear\",\n"
//                 "      \"domain\": {\"data\": \"data\", \"field\": \"c0\"}\n"
//                 "    },\n"
//                 "    {\n"
//                 "      \"name\": \"y\",\n"
//                 "      \"type\": \"linear\",\n"
//                 "      \"domain\": {\"data\": \"data\", \"field\": \"c1\"}\n"
//                 "    }\n"
//                 "  ],\n"
//                 "  \"marks\": [\n"
//                 "    {\n"
//                 "      \"encode\": {\n"
//                 "        \"enter\": {\n"
//                 "          \"shape\": {\"value\": \"circle\"},\n"
//                 "          \"stroke\": {\"value\": \"#ff0000\"},\n"
//                 "          \"strokeWidth\": {\"value\": 30},\n"
//                 "          \"opacity\": {\"value\": 0.5}\n"
//                 "        }\n"
//                 "      }\n"
//                 "    }\n"
//                 "  ]\n"
//                 "}";
//
//    point_map.set_input(input);
//    point_map.Render();
    zilliz::render::get_pointmap(array_vector);
}


