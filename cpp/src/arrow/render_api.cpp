#include <iostream>
#include "render_api.h"
#include "render/render_builder.h"

namespace zilliz {
namespace render {

std::shared_ptr<arrow::Array>
out_pic(std::pair<uint8_t* ,int64_t> output) {
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
point_map(const std::shared_ptr<arrow::Array> &arr_x, const std::shared_ptr<arrow::Array> &arr_y) {
    auto x_length = arr_x->length();
    auto y_length = arr_y->length();
    auto x_type = arr_x->type_id();
    auto y_type = arr_y->type_id();
    assert(x_length == y_length);
    assert(x_type == arrow::Type::UINT32);
    assert(y_type == arrow::Type::UINT32);

    auto input_x = (uint32_t *) arr_x->data()->GetValues<uint8_t>(1);
    auto input_y = (uint32_t *) arr_y->data()->GetValues<uint8_t>(1);

    return out_pic(pointmap(input_x, input_y, x_length));
}

std::shared_ptr<arrow::Array>
heat_map(const std::shared_ptr<arrow::Array> &arr_x, const std::shared_ptr<arrow::Array> &arr_y, const std::shared_ptr<arrow::Array> &arr_c) {
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
        case arrow::Type::INT8 : {
            auto input_c_int8 = (int8_t *) arr_c->data()->GetValues<uint8_t>(1);
            return out_pic(heatmap<int8_t >(input_x, input_y, input_c_int8, x_length));
        }
        case arrow::Type::INT16 : {
            auto input_c_int16 = (int16_t *) arr_c->data()->GetValues<uint8_t>(1);
            return out_pic(heatmap<int16_t >(input_x, input_y, input_c_int16, x_length));
        }
        case arrow::Type::INT32 : {
            auto input_c_int32 = (int32_t *) arr_c->data()->GetValues<uint8_t>(1);
            return out_pic(heatmap<int32_t >(input_x, input_y, input_c_int32, x_length));
        }
        case arrow::Type::INT64 : {
            auto input_c_int64 = (int64_t *) arr_c->data()->GetValues<uint8_t>(1);
            return out_pic(output = heatmap<int64_t >(input_x, input_y, input_c_int64, x_length));
        }
        case arrow::Type::UINT8 : {
            auto input_c_uint8 = (uint8_t *) arr_c->data()->GetValues<uint8_t>(1);
            return out_pic(heatmap<uint8_t >(input_x, input_y, input_c_uint8, x_length));
        }
        case arrow::Type::UINT16 : {
            auto input_c_uint16 = (uint16_t *) arr_c->data()->GetValues<uint8_t>(1);
            return out_pic(heatmap<uint16_t >(input_x, input_y, input_c_uint16, x_length));
        }
        case arrow::Type::UINT32 : {
            auto input_c_uint32 = (uint32_t *) arr_c->data()->GetValues<uint8_t>(1);
            return out_pic(heatmap<uint32_t >(input_x, input_y, input_c_uint32, x_length));
        }
        case arrow::Type::UINT64 : {
            auto input_c_uint64 = (uint64_t *) arr_c->data()->GetValues<uint8_t>(1);
            return out_pic(heatmap<uint64_t >(input_x, input_y, input_c_uint64, x_length));
        }
        case arrow::Type::FLOAT : {
            auto input_c_float = (float *) arr_c->data()->GetValues<uint8_t>(1);
            return out_pic(heatmap<float>(input_x, input_y, input_c_float, x_length));
        }
        case arrow::Type::DOUBLE : {
            auto input_c_double = (double *) arr_c->data()->GetValues<uint8_t>(1);
            return out_pic(heatmap<double>(input_x, input_y, input_c_double, x_length));
        }

        default:
            std::cout << "type error!";
    }
    return nullptr;
}

} //namespace render
} //namespace zilliz
