#include <GL/gl.h>
#include <GLES3/gl3.h>

#include "render/engine/layer/layer_weighted_pointsize_circles_2d.h"
#include "render/utils/dataset/dataset_accessor.h"
#include "render/utils/color/color_parser.h"


namespace zilliz {
namespace render {
namespace engine {

LayerWeightedPointSizeCircles2D::LayerWeightedPointSizeCircles2D()
    : vertices_x_(nullptr), vertices_y_(nullptr), num_vertices_(0) {
}

LayerWeightedPointSizeCircles2D::~LayerWeightedPointSizeCircles2D() {
    auto &mem_pool = MemManager::GetInstance().main_memory_pool();

    if (pointsize_ != nullptr) {
        mem_pool.Free(pointsize_);
    }
}

void LayerWeightedPointSizeCircles2D::Init() {
    auto &data_params_type = plan_node_->data_param_type();
    colors_ = plan_node_->render_params().color;
    switch (data_params_type[2]) {
        case ValueType::kValInt8:
            set_vertices_pointsize<int8_t>();
            break;
        case ValueType::kValInt16:
            set_vertices_pointsize<int16_t>();
            break;
        case ValueType::kValInt32:
            set_vertices_pointsize<int32_t>();
            break;
        case ValueType::kValInt64:
            set_vertices_pointsize<int64_t>();
            break;
        case ValueType::kValUInt8:
            set_vertices_pointsize<u_int8_t>();
            break;
        case ValueType::kValUInt16:
            set_vertices_pointsize<u_int16_t>();
            break;
        case ValueType::kValUInt32:
            set_vertices_pointsize<u_int32_t>();
            break;
        case ValueType::kValUInt64:
            set_vertices_pointsize<u_int64_t>();
            break;
        case ValueType::kValFloat:
            set_vertices_pointsize<float>();
            break;
        case ValueType::kValDouble:
            set_vertices_pointsize<double>();
            break;
        default:
            std::string msg = "cannot find value type";
//            THROW_RENDER_ENGINE_ERROR(VALUE_TYPE_NOT_FOUND, msg);
    }
}

void LayerWeightedPointSizeCircles2D::Shader() {
    CHECK_CUDA(cudaSetDevice(plan_node()->dev_id()));

    const char *vertexShaderSource = \
        "#version 430 core\n"
        "layout (location = 0) in uint posX;\n"
        "layout (location = 1) in uint posY;\n"
        "layout (location = 2) in float point_size;\n"
        "layout (location = 3) uniform vec2 screen_info;\n"
        "void main()\n"
        "{\n"
        "   float tmp_x = posX;\n"
        "   float tmp_y = posY;\n"
        "   gl_Position = vec4(((tmp_x * 2) / screen_info.x) - 1, ((tmp_y * 2) / screen_info.y) - 1, 0, 1);\n"
        "   gl_PointSize = point_size;\n"
        "}";

    const char *fragmentShaderSource = \
        "#version 430 core\n"
        "out vec4 FragColor;\n"
        "layout (location = 4) uniform vec4 color;\n"
        "void main()\n"
        "{\n"
        "   FragColor = color.xyzw;\n"
        "}";

    int success;
    int vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
    glCompileShader(vertexShader);
    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
    if (!success) {
//        RENDER_ENGINE_LOG_ERROR << "vertex shader compile failed.";
    }
    int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
    glCompileShader(fragmentShader);
    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
    if (!success) {
//        RENDER_ENGINE_LOG_ERROR << "fragment shader compile failed.";
    }
    int shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);
    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
    if (!success) {
//        RENDER_ENGINE_LOG_ERROR << "shader program compile failed.";
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
    glBufferData(GL_ARRAY_BUFFER, num_vertices_ * 1 * sizeof(float),pointsize_, GL_STATIC_DRAW);

    glGenVertexArrays(1, &VAO );
    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO[0]);
    glVertexAttribPointer(0, 1, GL_FLOAT, GL_FALSE, 1 * sizeof(uint32_t), (void *) 0);
    glBindBuffer(GL_ARRAY_BUFFER, VBO[1]);
    glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 1 * sizeof(uint32_t), (void *) 0);
    glBindBuffer(GL_ARRAY_BUFFER, VBO[2]);
    glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, 1 * sizeof(float), (void *) 0);

    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
    glEnableVertexAttribArray(2);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    glBindBuffer(GL_ARRAY_BUFFER, 1);
    glBindVertexArray(1);

    glUseProgram(shaderProgram);
    glUniform2f(3, window_params_.width(), window_params_.height());
    glUniform4f(4, colors_.r, colors_.g, colors_.b, colors_.a);
    glBindVertexArray(VAO);

}

void LayerWeightedPointSizeCircles2D::Render() {

    glEnable(GL_PROGRAM_POINT_SIZE);

    glClear(GL_COLOR_BUFFER_BIT);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_POINT_SMOOTH);

    glDrawArrays(GL_POINTS, 0, num_vertices());
    glFlush();

    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(3, VBO);
}

template<typename T>
void LayerWeightedPointSizeCircles2D::set_vertices_pointsize() {

    auto &data_params = plan_node_->data_params();
    auto x_id = data_params[0];
    auto y_id = data_params[1];
    auto pointsize_id = data_params[2];

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
    pointsize_ = (float *) mem_pool.Malloc(num_vertices_ * sizeof(float));

    size_t p_offset = 0;
    for (size_t i = 0; i < fragments_field.size(); i++) {
        x_id.set_fragment_field(fragments_field[i]);
        y_id.set_fragment_field(fragments_field[i]);
        pointsize_id.set_fragment_field(fragments_field[i]);
        x_id.set_attr_type(zilliz::common::TableIDAttrEx::AttrType::kData);
        y_id.set_attr_type(zilliz::common::TableIDAttrEx::AttrType::kData);
        pointsize_id.set_attr_type(zilliz::common::TableIDAttrEx::AttrType::kData);

        vertices_x_ = dataset_accessor()->GetChunkData<uint32_t>(x_id);
        vertices_y_ = dataset_accessor()->GetChunkData<uint32_t>(y_id);
        auto pointsize = dataset_accessor()->GetChunkData<uint32_t >(pointsize_id);

        for (auto j = 0; j < num_rows[i]; j++) {
            pointsize_[p_offset++] = pointsize.get()[j];
        }
    }
}

} // namespace engine
} // namespace render
} // namespace zilliz
