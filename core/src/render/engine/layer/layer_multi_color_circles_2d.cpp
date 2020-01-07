#include <GL/gl.h>

#include <GLES3/gl3.h>
#include <atomic>
#include <thread>

#include "zstring/DictStringEngineAgent.h"
#include "zstring/HashStringEngineAgent.h"
#include "zstring/ShortStringEngineAgent.h"
#include "zstring/StringEngineOwner.h"

#include "zcommon/util/string_builder.h"

#include "render/utils/dataset/dataset_accessor.h"
#include "render/engine/layer/layer_multi_color_circles_2d.h"


namespace zilliz {
namespace render {
namespace engine {

void
LayerMultiColorCircles2D::Init() {
    CircleParamsTrans();
    SetVerticesAndColors();
}

void
LayerMultiColorCircles2D::Shader() {
    CHECK_CUDA(cudaSetDevice(plan_node()->dev_id()));
    // 1.vertex shader
    const char *vertex_shader_source = \
        "#version 430 core\n"
        "layout (location = 0) in uvec2 pos;\n"
        "layout (location = 1) uniform vec2 screen_info;\n"
        "layout (location = 2) uniform float point_size;\n"
        "void main()\n"
        "{\n"
        "   float tmp_x = pos.x;\n"
        "   float tmp_y = pos.y;\n"
        "   gl_Position = vec4(((tmp_x * 2) / screen_info.x) - 1, ((tmp_y * 2) / screen_info.y) - 1, 0, 1);\n"
        "   gl_PointSize = point_size;\n"
        "}";

    // 2.fragment shader
    const char *fragment_shader_source = \
        "#version 430 core\n"
        "out vec4 FragColor;\n"
        "layout (location = 3) uniform vec4 color;\n"
        "void main()\n"
        "{\n"
        "   FragColor = color.xyzw;\n"
        "}";

    // 3.create vertex shader and compile it
    int success;
    int vertex_shader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertex_shader, 1, &vertex_shader_source, NULL);
    glCompileShader(vertex_shader);
    glGetShaderiv(vertex_shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        RENDER_ENGINE_LOG_ERROR << "vertex shader compile failed.";
    }

    // 4.create fragment shader and compile it
    int fragment_shader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragment_shader, 1, &fragment_shader_source, NULL);
    glCompileShader(fragment_shader);
    glGetShaderiv(vertex_shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        RENDER_ENGINE_LOG_ERROR << "fragment shader compile failed.";
    }

    // 5.link shader
    shader_program_ = glCreateProgram();
    glAttachShader(shader_program_, vertex_shader);
    glAttachShader(shader_program_, fragment_shader);
    glLinkProgram(shader_program_);
    glGetProgramiv(shader_program_, GL_LINK_STATUS, &success);
    if (!success) {
        RENDER_ENGINE_LOG_ERROR << "shader program link failed.";
    }

    // 6.delete shader, we don't need it anymore.
    glDeleteShader(vertex_shader);
    glDeleteShader(fragment_shader);

    // 7.create and use VAO and VBO, VAO:vertex array object, VBO:vertex buffer object
    glGenVertexArrays(1, &VAO); // param1:size of VAO, param2: address of VAO
    glGenBuffers(1, &VBO); // param1:size of VBO, param2: address of VBO

    // 8.bind VAO, after the VAO is created, we need to connect it to the cache before using it.
    glBindVertexArray(VAO);
}


void
LayerMultiColorCircles2D::Render() {
    glEnable(GL_PROGRAM_POINT_SIZE);

    glClear(GL_COLOR_BUFFER_BIT);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_POINT_SMOOTH);

    for (const auto &it : records_) {

        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBufferData(GL_ARRAY_BUFFER, it.second.second.size() * 2 * sizeof(uint32_t), &it.second.second[0], GL_STATIC_DRAW);

        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(uint32_t), nullptr);
        glEnableVertexAttribArray(0);

        auto radius = plan_node_->circle_params().at(it.first).radius;
        auto color = plan_node_->circle_params().at(it.first).color;

        glUseProgram(shader_program_);
        glUniform2f(1, window_params_.width(), window_params_.height());
        glUniform1f(2, radius);
        glUniform4f(3, color.r / 255.0f, color.g / 255.0f, color.b / 255.0f, color.a);

        glDrawArrays(GL_POINTS, 0, it.second.second.size());
    }

    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
}


zstring::StringEngineOwnerPtr
GetStringEngineOwner(const ColumnID &column_id) {
    auto string_type = common::GetIntermediateIdType(column_id);
    switch (string_type) {
        case common::TextGroupKeygenType::kTreeKey: {
            return sql::agent::DictStringEngineAgent::GetInstance().GetGroup(
                    common::GetStringGroupID(column_id.db_id(),
                                             column_id.table_id(),
                                             column_id.column_id()));
        }
        case common::TextGroupKeygenType::kHashKey: {
            return sql::agent::HashStringEngineAgent::GetInstance().GetGroup(
                    common::GetStringGroupID(column_id.db_id(),
                                             column_id.table_id(),
                                             column_id.column_id()));
        }
        case common::TextGroupKeygenType::kShortString: {
            return sql::agent::ShortStringEngineAgent::Generate();
        }
        default: {
            std::string msg = "Unknown string type '" + std::to_string(string_type) + "'";
            THROW_RENDER_ENGINE_ERROR(UNKNOWN_STRING_TYPE, msg)
        }
    }
}


void
LayerMultiColorCircles2D::CircleParamsTrans() {
    auto owner = GetStringEngineOwner(plan_node_->old_column_id());

    auto &src_circle_params = plan_node_->string_circle_params();
    auto &dst_circle_params = plan_node_->mutable_circle_params();

    for (auto it : src_circle_params) {

        auto label_string = it.first;
        int64_t label_id = 0;

        owner->GetIdByString(label_string, label_id);
        if (label_id == -1) {
            RENDER_ENGINE_LOG_ERROR << "Get id by string failed.";
        }
        dst_circle_params.emplace(label_id, it.second);
    }
}



void
LayerMultiColorCircles2D::SetVerticesAndColors() {

    auto &data_params = plan_node_->mutable_data_params();

    auto x_id = data_params[0];
    auto y_id = data_params[1];
    auto label_id = data_params[2];

    auto table_id = x_id;
    table_id.truncate_to_table_id();

    auto fragments_field = DatasetAccessor::GetFragments(input(), table_id);

    for (int field : fragments_field) {

        x_id.set_fragment_field(field);
        y_id.set_fragment_field(field);
        label_id.set_fragment_field(field);

        x_id.set_attr_type(zilliz::common::TableIDAttrEx::AttrType::kData);
        y_id.set_attr_type(zilliz::common::TableIDAttrEx::AttrType::kData);
        label_id.set_attr_type(zilliz::common::TableIDAttrEx::AttrType::kData);

        auto fragment_id = x_id;
        fragment_id.truncate_to_fragment_id();
        int64_t num_rows = DatasetAccessor::GetNumRows(input(), fragment_id);

        auto x = dataset_accessor()->GetChunkData<uint32_t>(x_id);
        auto y = dataset_accessor()->GetChunkData<uint32_t>(y_id);
        auto label = dataset_accessor()->GetChunkData<int64_t>(label_id);

        auto arr_x = x.get();
        auto arr_y = y.get();
        auto arr_l = label.get();

        for (auto& param: plan_node_->mutable_circle_params()) {
            records_[param.first].first = 0;
            records_[param.first].second.resize(num_rows);
        }

        for (auto index = 0; index < num_rows; index++) {
            auto key = arr_l[index];
            auto value = std::make_pair(arr_x[index], arr_y[index]);
            if (records_.find(key) == records_.end()) {
                continue;
            }
            auto& record = records_[key];
            auto id = record.first.fetch_add(1, std::memory_order::memory_order_relaxed);
            record.second[id] = value;
        }

        for(auto &record: records_) {
            auto& pair = record.second;
            pair.second.resize(pair.first);
            pair.second.shrink_to_fit();
        }
    }
}

} // namespace engine
} // namespace render
} // namespace zilliz
