#include <iostream>
#include "render_api.h"
#include "render/render_builder.h"

namespace zilliz {
namespace render {

std::shared_ptr<arrow::Array>
get_pointmap(std::shared_ptr<arrow::Array> arr_x, std::shared_ptr<arrow::Array> arr_y) {
    auto x_length = arr_x->length();
    auto y_length = arr_y->length();
    auto x_type = arr_x->type_id();
    auto y_type = arr_y->type_id();
    assert(x_length == y_length);
    assert(x_type == arrow::Type::UINT32);
    assert(y_type == arrow::Type::UINT32);

    auto input_x = (uint32_t *) arr_x->data()->GetValues<uint8_t>(1);
    auto input_y = (uint32_t *) arr_y->data()->GetValues<uint8_t>(1);

    auto output = pointmap(input_x, input_y, x_length);

    auto output_length = output.second;
    auto output_data = output.first;
    auto bit_map = (uint8_t*)malloc(output_length);
    memset(bit_map, 0xff, output_length);
    auto buffer0 = std::make_shared<arrow::Buffer>(bit_map, output_length);
    auto buffer1 = std::make_shared<arrow::Buffer>(output_data, output_length);
    auto buffers = std::vector<std::shared_ptr<arrow::Buffer>>();
    buffers.emplace_back(buffer0);
    buffers.emplace_back(buffer1);

    auto data_type = arrow::uint8();
    auto array_data = arrow::ArrayData::Make(data_type, output_length, buffers);
    auto array = arrow::MakeArray(array_data);
    return array;
}

std::shared_ptr<arrow::Array>
get_heatmap(std::shared_ptr<arrow::Array> arr_x, std::shared_ptr<arrow::Array> arr_y, std::shared_ptr<arrow::Array> arr_c) {
    auto x_length = arr_x->length();
    auto y_length = arr_y->length();
    auto c_length = arr_c->length();
    auto x_type = arr_x->type_id();
    auto y_type = arr_y->type_id();
    auto c_type = arr_c->type_id();
    assert(x_length == y_length);
    assert(x_length == c_length);
    assert(x_type == arrow::Type::UINT32);
    assert(y_type == arrow::Type::UINT32);

    auto input_x = (uint32_t *) arr_x->data()->GetValues<uint8_t>(1);
    auto input_y = (uint32_t *) arr_y->data()->GetValues<uint8_t>(1);

    std::pair<uint8_t* ,int64_t> output;
    switch(c_type) {
        case arrow::Type::FLOAT : {
            auto input_c_float = (float *) arr_c->data()->GetValues<uint8_t>(1);
            output = heatmap<float>(input_x, input_y, input_c_float, x_length);
            break;
        }
        case arrow::Type::DOUBLE : {
            auto input_c_double = (double *) arr_c->data()->GetValues<uint8_t>(1);
            output = heatmap<double>(input_x, input_y, input_c_double, x_length);
            break;
        }
        case arrow::Type::UINT32 : {
            auto input_c_uint32 = (uint32_t *) arr_c->data()->GetValues<uint8_t>(1);
            output = heatmap<uint32_t >(input_x, input_y, input_c_uint32, x_length);
            break;
        }
        default:
            std::cout << "type error!";
    }

    auto output_length = output.second;
    auto output_data = output.first;
    auto bit_map = (uint8_t*)malloc(output_length);
    memset(bit_map, output_length, 0xff);

    auto buffer0 = std::make_shared<arrow::Buffer>(bit_map, output_length);
    auto buffer1 = std::make_shared<arrow::Buffer>(output_data, output_length);
    auto buffers = std::vector<std::shared_ptr<arrow::Buffer>>();
    buffers.emplace_back(buffer0);
    buffers.emplace_back(buffer1);

    auto data_type = arrow::uint8();
    auto array_data = arrow::ArrayData::Make(data_type, output_length, buffers);
    auto array = arrow::MakeArray(array_data);
    return array;
}

} //namespace render
} //namespace zilliz
