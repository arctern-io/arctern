#include <map>
#include <GL/gl.h>

#include "render/engine/image/loader/image_loader.h"
#include "render/engine/layer/layer_icon_2d.h"
#include "render/engine/common/memory.h"
#include "render/utils/dataset/dataset_accessor.h"


namespace zilliz {
namespace render {
namespace engine {


LayerIcon2D::LayerIcon2D()
    : vertices_(nullptr), num_vertices_(0) {
}

LayerIcon2D::~LayerIcon2D() {

    auto &mem_pool = MemManager::GetInstance().main_memory_pool();

    if (vertices_ != nullptr) {
        mem_pool.Free(vertices_);
    }
}

void
LayerIcon2D::Init() {
    SetVertices();
}

void
LayerIcon2D::Shader() {

}

void
LayerIcon2D::Render() {

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    auto &image_loader = ImageLoader::GetInstance();
    auto &image_buffer = image_loader.image_buffers().at(plan_node_->icon_name());

    if (image_buffer.buffer == nullptr) {
        THROW_RENDER_ENGINE_ERROR(UNINITIALIZED, "image buffer is empty, please load buffer before access.")
    }

    for (int i = 3; i < image_buffer.image_params.height * image_buffer.image_params.width * 4; i += 4) {
        image_buffer.buffer[i] /= 5;
    }

    for (int i = 0; i < num_vertices_; i++) {
        glRasterPos2f(vertices_[2 * i], vertices_[2 * i + 1]);
        glDrawPixels(image_buffer.image_params.width,
                     image_buffer.image_params.height,
                     GL_RGBA,
                     GL_UNSIGNED_BYTE,
                     image_buffer.buffer);
    }
}

void
LayerIcon2D::SetVertices() {

    auto &data_params = plan_node_->mutable_data_params();
    auto x_id = data_params[0];
    auto y_id = data_params[1];
    auto table_id = x_id;

    table_id.truncate_to_table_id();
    auto fragments_field = DatasetAccessor::GetFragments(input(), table_id);

    std::vector<int64_t> num_rows;
    num_rows.clear();
    for (size_t i = 0; i < fragments_field.size(); i++) {
        x_id.set_fragment_field(fragments_field[i]);
        x_id.set_attr_type(zilliz::common::TableIDAttrEx::AttrType::kMeta);
        auto fragment_id = x_id;
        fragment_id.truncate_to_fragment_id();
        int64_t num_row = DatasetAccessor::GetNumRows(input(), fragment_id);
        num_rows.emplace_back(num_row);
        num_vertices_ += num_row;
    }

    auto &mem_pool = MemManager::GetInstance().main_memory_pool();
    vertices_ = (uint32_t *) mem_pool.Malloc(num_vertices_ * 2 * sizeof(uint32_t));

    size_t offset = 0;
    for (size_t i = 0; i < fragments_field.size(); i++) {
        x_id.set_fragment_field(fragments_field[i]);
        y_id.set_fragment_field(fragments_field[i]);
        x_id.set_attr_type(zilliz::common::TableIDAttrEx::AttrType::kData);
        y_id.set_attr_type(zilliz::common::TableIDAttrEx::AttrType::kData);
        auto x = dataset_accessor()->GetChunkData<uint32_t>(x_id);
        auto y = dataset_accessor()->GetChunkData<uint32_t>(y_id);
        for (auto j = 0; j < num_rows[i]; j++) {
            vertices_[offset++] = x.get()[j];
            vertices_[offset++] = y.get()[j];
        }
    }
}


} // namespace engine
} // namespace render
} // namespace engine

