#include <GL/gl.h>
#include <GLES3/gl3.h>

#include "render/engine/layer/layer_weighted_color_circles_2d.h"
#include "render/utils/dataset/dataset_accessor.h"
#include "render/utils/color/color_parser.h"


namespace zilliz {
namespace render {
namespace engine {

LayerWeightedColorCircles2D::LayerWeightedColorCircles2D()
    : vertices_x_(nullptr), vertices_y_(nullptr), colors_(nullptr), num_vertices_(0) {
}

LayerWeightedColorCircles2D::~LayerWeightedColorCircles2D() {

    auto &mem_pool = MemManager::GetInstance().main_memory_pool();

    if (colors_ != nullptr) {
        mem_pool.Free(colors_);
    }
}

void LayerWeightedColorCircles2D::Init() {
    auto &data_params_type = plan_node_->data_param_type();
    switch (data_params_type[2]) {
        case ValueType::kValInt8:
            set_vertices_colors<int8_t>();
            break;
        case ValueType::kValInt16:
            set_vertices_colors<int16_t>();
            break;
        case ValueType::kValInt32:
            set_vertices_colors<int32_t>();
            break;
        case ValueType::kValInt64:
            set_vertices_colors<int64_t>();
            break;
        case ValueType::kValUInt8:
            set_vertices_colors<u_int8_t>();
            break;
        case ValueType::kValUInt16:
            set_vertices_colors<u_int16_t>();
            break;
        case ValueType::kValUInt32:
            set_vertices_colors<u_int32_t>();
            break;
        case ValueType::kValUInt64:
            set_vertices_colors<u_int64_t>();
            break;
        case ValueType::kValFloat:
            set_vertices_colors<float>();
            break;
        case ValueType::kValDouble:
            set_vertices_colors<double>();
            break;
        default:
            std::string msg = "cannot find value type";
//            THROW_RENDER_ENGINE_ERROR(VALUE_TYPE_NOT_FOUND, msg);
    }
}

void LayerWeightedColorCircles2D::Shader() {
    CHECK_CUDA(cudaSetDevice(plan_node()->dev_id()));
    const char *vertexShaderSource = \
        "#version 430 core\n"
        "layout (location = 0) in uint posX;\n"
        "layout (location = 1) in uint posY;\n"
        "layout (location = 2) in vec4 point_color;\n"
        "layout (location = 3) uniform vec2 screen_info;\n"
        "out vec4 color;\n"
        "void main()\n"
        "{\n"
        "   float tmp_x = posX;\n"
        "   float tmp_y = posY;\n"
        "   gl_Position = vec4(((tmp_x * 2) / screen_info.x) - 1, ((tmp_y * 2) / screen_info.y) - 1, 0, 1);\n"
        "   color=point_color;\n"
        "}";

    const char *fragmentShaderSource = \
        "#version 430 core\n"
        "in vec4 color;\n"
        "out vec4 FragColor;\n"
        "void main()\n"
        "{\n"
        "   FragColor = color;\n"
        "}";


    int success;
    int vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
    glCompileShader(vertexShader);
    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        RENDER_ENGINE_LOG_ERROR << "vertex shader compile failed.";
    }
    int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
    glCompileShader(fragmentShader);
    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        RENDER_ENGINE_LOG_ERROR << "fragment shader compile failed.";
    }
    int shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);
    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
    if (!success) {
        RENDER_ENGINE_LOG_ERROR << "shader program compile failed.";
    }
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    glGenVertexArrays(1, &VAO);
    glGenBuffers(3, VBO);

    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO[0]);
    glBufferData(GL_ARRAY_BUFFER, num_vertices_ * 1 * sizeof(uint32_t), vertices_x_.get(), GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, VBO[1]);
    glBufferData(GL_ARRAY_BUFFER, num_vertices_ * 1 * sizeof(uint32_t), vertices_y_.get(), GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, VBO[2]);
    glBufferData(GL_ARRAY_BUFFER, num_vertices_ * 4 * sizeof(float), colors_, GL_STATIC_DRAW);

    glGenVertexArrays(1, &VAO );
    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO[0]);
    glVertexAttribPointer(0, 1, GL_FLOAT, GL_TRUE, 1 * sizeof(uint32_t), (void *) 0);
    glBindBuffer(GL_ARRAY_BUFFER, VBO[1]);
    glVertexAttribPointer(1, 1, GL_FLOAT, GL_TRUE, 1 * sizeof(uint32_t), (void *) 0);
    glBindBuffer(GL_ARRAY_BUFFER, VBO[2]);
    glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void *) 0);

    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
    glEnableVertexAttribArray(2);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    glBindBuffer(GL_ARRAY_BUFFER, 1);
    glBindVertexArray(1);

    glUseProgram(shaderProgram);
    glUniform2f(3, window_params_.width(), window_params_.height());
    glBindVertexArray(VAO);

}

void LayerWeightedColorCircles2D::Render() {

    glEnable(GL_PROGRAM_POINT_SIZE);

    glClear(GL_COLOR_BUFFER_BIT);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_POINT_SMOOTH);

    float point_size = plan_node_->radius();
    glPointSize(point_size);
    glDrawArrays(GL_POINTS, 0, num_vertices());
    glFlush();

    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(3, VBO);
}

template<typename T>
void LayerWeightedColorCircles2D::set_vertices_colors() {

    auto &data_params = plan_node_->data_params();
    auto x_id = data_params[0];
    auto y_id = data_params[1];
    auto count_id = data_params[2];

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
    colors_ = (float *) mem_pool.Malloc(num_vertices_ * 4 * 4);

    size_t c_offset = 0;
    auto count_start = plan_node()->count_start();
    auto count_end = plan_node()->count_end();
    int64_t count_range = count_end - count_start;

    for (size_t i = 0; i < fragments_field.size(); i++) {
        x_id.set_fragment_field(fragments_field[i]);
        y_id.set_fragment_field(fragments_field[i]);
        count_id.set_fragment_field(fragments_field[i]);
        x_id.set_attr_type(zilliz::common::TableIDAttrEx::AttrType::kData);
        y_id.set_attr_type(zilliz::common::TableIDAttrEx::AttrType::kData);
        count_id.set_attr_type(zilliz::common::TableIDAttrEx::AttrType::kData);


        vertices_x_ = dataset_accessor()->GetChunkData<uint32_t>(x_id);
        vertices_y_ = dataset_accessor()->GetChunkData<uint32_t>(y_id);
        auto count = dataset_accessor()->GetChunkData<T>(count_id);


        for (auto j = 0; j < num_rows[i]; j++) {

            auto color_style = plan_node()->color_style();
            auto vertice_count = count.get()[j] >= count_start ? count.get()[j] : count_start;
            vertice_count = count.get()[j] <= count_end ? vertice_count : count_end;
            auto ratio = (vertice_count - count_start) / (double) count_range;
            auto circle_params_2d = ColorParser::GetCircleParams(color_style, ratio);
            colors_[c_offset++] = circle_params_2d.color.r;
            colors_[c_offset++] = circle_params_2d.color.g;
            colors_[c_offset++] = circle_params_2d.color.b;
            colors_[c_offset++] = circle_params_2d.color.a;
        }
    }
}

} // namespace engine
} // namespace render
} // namespace zilliz
