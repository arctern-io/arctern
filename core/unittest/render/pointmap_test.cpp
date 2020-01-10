#include <gtest/gtest.h>
#include <iostream>
#include "render/2d/pointmap.h"

using namespace zilliz::render;

TEST(CIRCLE_2D_TEST, SINGLE_COLOR_TEST) {
    PointMap point_map;

    int64_t length = 1024;
    auto bit_map = (uint8_t*)malloc(length);
    memset(bit_map, length, 0xff);
    std::shared_ptr<arrow::DataType> data_type = arrow::uint8();

    auto buff_data1 = (uint8_t*)malloc(length);
    memset(buff_data1, length, 0x01);
    std::shared_ptr<arrow::Buffer> buffer0 = std::make_shared<arrow::Buffer>(bit_map, length);
    std::shared_ptr<arrow::Buffer> buffer1 = std::make_shared<arrow::Buffer>(buff_data1, length);
    auto buffers1 = std::vector<std::shared_ptr<arrow::Buffer>>();
    buffers1.emplace_back(buffer0);
    buffers1.emplace_back(buffer1);
    std::shared_ptr<arrow::ArrayData> array_data1 = arrow::ArrayData::Make(data_type, length, buffers1);
    std::shared_ptr<arrow::Array> array1 = arrow::MakeArray(array_data1);

    auto buff_data2 = (uint8_t*)malloc(length);
    memset(buff_data2, length, 0x01);
    std::shared_ptr<arrow::Buffer> buffer2 = std::make_shared<arrow::Buffer>(bit_map, length);
    std::shared_ptr<arrow::Buffer> buffer3 = std::make_shared<arrow::Buffer>(buff_data2, length);
    auto buffers2 = std::vector<std::shared_ptr<arrow::Buffer>>();
    buffers2.emplace_back(buffer2);
    buffers2.emplace_back(buffer3);
    std::shared_ptr<arrow::ArrayData> array_data2 = arrow::ArrayData::Make(data_type, length, buffers2);
    std::shared_ptr<arrow::Array> array2 = arrow::MakeArray(array_data2);

    std::vector<std::shared_ptr<arrow::Array>> array_vector;
    array_vector.emplace_back(array1);
    array_vector.emplace_back(array2);

    Input input;
    input.array_vector = array_vector;
    input.vega = "";

    point_map.set_input(input);
    point_map.Draw();
}


