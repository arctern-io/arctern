#include "render_api.h"
#include "render/2d/render_builder.h"

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
    int64_t num_vertices_ = x_length / sizeof(uint32_t);

    //array{ArrayData{vector<Buffer{uint8_t*}>}}
    auto x_data = (uint32_t *) arr_x->data()->GetValues<uint8_t>(1);
    auto y_data = (uint32_t *) arr_y->data()->GetValues<uint8_t>(1);
    auto input_x = std::shared_ptr<uint32_t>(x_data);
    auto input_y = std::shared_ptr<uint32_t>(y_data);

    auto output = pointmap(input_x, input_y, num_vertices_);

    auto output_length = output.second;
    auto output_data = output.first;
    auto bit_map = (uint8_t*)malloc(output_length);
    memset(bit_map, output_length, 0xff);
    auto buffer0 = std::make_shared<arrow::Buffer>(bit_map, output_length);
    auto buffer1 = std::make_shared<arrow::Buffer>(output_data.get(), output_length);
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
