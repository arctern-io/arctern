#include <map>
#include <GL/gl.h>
#include <GLES3/gl3.h>

#include "render/engine/layer/layer_circles_2d.h"
#include "render/engine/common/memory.h"
#include "render/utils/dataset/dataset_accessor.h"


namespace zilliz {
namespace render {
namespace engine {

LayerCircles2D::LayerCircles2D()
    : vertices_x_(nullptr), vertices_y_(nullptr), num_vertices_(0) {
}

void
LayerCircles2D::Init() {
    SetVertices();
    SetColors();
}

void
LayerCircles2D::Shader() {
//    CHECK_CUDA(cudaSetDevice(plan_node()->dev_id()));
    // 1.vertex shader
    const char *vertex_shader_source = \
        "#version 430 core\n"
        "layout (location = 0) in uint posX;\n"
        "layout (location = 1) in uint posY;\n"
        "layout (location = 2) uniform vec2 screen_info;\n"
        "layout (location = 3) uniform float point_size;\n"
        "void main()\n"
        "{\n"
        "   float tmp_x = posX;\n"
        "   float tmp_y = posY;\n"
        "   gl_Position = vec4(((tmp_x * 2) / screen_info.x) - 1, ((tmp_y * 2) / screen_info.y) - 1, 0, 1);\n"
        "   gl_PointSize = point_size;\n"
        "}";

    // 2.fragment shader
    const char *fragment_shader_source = \
        "#version 430 core\n"
        "out vec4 FragColor;\n"
        "layout (location = 4) uniform vec4 color;\n"
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
//        RENDER_ENGINE_LOG_ERROR << "vertex shader compile failed.";
    }

    // 4.create fragment shader and compile it
    int fragment_shader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragment_shader, 1, &fragment_shader_source, NULL);
    glCompileShader(fragment_shader);
    glGetShaderiv(vertex_shader, GL_COMPILE_STATUS, &success);
    if (!success) {
//        RENDER_ENGINE_LOG_ERROR << "fragment shader compile failed.";
    }

    // 5.link shader
    int shader_program = glCreateProgram();
    glAttachShader(shader_program, vertex_shader);
    glAttachShader(shader_program, fragment_shader);
    glLinkProgram(shader_program);
    glGetProgramiv(shader_program, GL_LINK_STATUS, &success);
    if (!success) {
//        RENDER_ENGINE_LOG_ERROR << "shader program link failed.";
    }

    // 6.delete shader, we don't need it anymore.
    glDeleteShader(vertex_shader);
    glDeleteShader(fragment_shader);

    // 7.create and use VAO and VBO, VAO:vertex array object, VBO:vertex buffer object
    glGenVertexArrays(1, &VAO); // param1:size of VAO, param2: address of VAO
    glGenBuffers(2, VBO); // param1:size of VBO, param2: address of VBO

    // 8.bind VAO, after the VAO is created, we need to connect it to the cache before using it.
    glBindVertexArray(VAO);

    // 9.bind VBO, copy vertex data to VBO, and set vertex attribute pointer
    glBindBuffer(GL_ARRAY_BUFFER, VBO[0]);
    glBufferData(GL_ARRAY_BUFFER, num_vertices_ * sizeof(uint32_t), vertices_x_.get(), GL_STATIC_DRAW);
    glVertexAttribPointer(0, 1, GL_FLOAT, GL_FALSE, 1 * sizeof(uint32_t), nullptr);

    glBindBuffer(GL_ARRAY_BUFFER, VBO[1]);
    glBufferData(GL_ARRAY_BUFFER, num_vertices_ * sizeof(uint32_t), vertices_y_.get(), GL_STATIC_DRAW);
    glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 1 * sizeof(uint32_t), nullptr);

    // 10.enable vertex attribute, which is disabled by default
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);

    // 11.use shader program and updating uniforms
    glUseProgram(shader_program);
    glUniform2f(2, window_params_.width(), window_params_.height());
    glUniform1f(3, plan_node_->render_params().radius);
    glUniform4f(4, colors_.r, colors_.g, colors_.b, colors_.a);
}


void
LayerCircles2D::Render() {
    glEnable(GL_PROGRAM_POINT_SIZE);

    glClear(GL_COLOR_BUFFER_BIT);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_POINT_SMOOTH);

    glDrawArrays(GL_POINTS, 0, num_vertices_);
    glFlush();

    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(2, VBO);
}

void
LayerCircles2D::SetVertices() {

    auto &data_params = plan_node_->mutable_data_params();
    auto x_id = data_params[0];
    auto y_id = data_params[1];
    auto table_id = x_id;

    table_id.truncate_to_table_id();
    auto fragments_field = DatasetAccessor::GetFragments(input(), table_id);

    std::vector<int64_t> num_rows;
    num_rows.clear();
    for (int i : fragments_field) {
        x_id.set_fragment_field(i);
        x_id.set_attr_type(zilliz::common::TableIDAttrEx::AttrType::kMeta);
        auto fragment_id = x_id;
        fragment_id.truncate_to_fragment_id();
        int64_t num_row = DatasetAccessor::GetNumRows(input(), fragment_id);
        num_rows.emplace_back(num_row);
        num_vertices_ += num_row;
    }

    for (int i : fragments_field) {
        x_id.set_fragment_field(i);
        y_id.set_fragment_field(i);
        x_id.set_attr_type(zilliz::common::TableIDAttrEx::AttrType::kData);
        y_id.set_attr_type(zilliz::common::TableIDAttrEx::AttrType::kData);
        vertices_x_ = dataset_accessor()->GetChunkData<uint32_t>(x_id);
        vertices_y_ = dataset_accessor()->GetChunkData<uint32_t>(y_id);
    }

}


void
LayerCircles2D::SetColors() {

    auto &render_param = plan_node()->render_params();
    colors_.r = render_param.color.r / 255.0f;
    colors_.g = render_param.color.g / 255.0f;
    colors_.b = render_param.color.b / 255.0f;
    colors_.a = render_param.color.a;
}


} // namespace engine
} // namespace render
} // namespace engine