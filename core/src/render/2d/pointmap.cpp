#include <map>
#include <GL/gl.h>
#include <GLES3/gl3.h>
#include <iostream>
#include "render/2d/pointmap.h"

namespace zilliz {
namespace render {

PointMap::PointMap()
    : vertices_x_(nullptr), vertices_y_(nullptr), num_vertices_(0), point_vega_(""){
}

void
PointMap::InputInit() {
    auto array_vector_ = input().array_vector;
    point_vega_ = (VegaCircle2d &)(input().vega);
}

void
PointMap::DataInit() {
    //set point_value
    auto input_data = array_vector();
    auto x_array = input_data[0];
    auto y_array = input_data[1];

    auto x_length = x_array->length();
    auto y_length = y_array->length();
    auto x_type = x_array->type_id();
    auto y_type = y_array->type_id();
    assert(x_length == y_length);
    assert(x_type == arrow::Type::UINT32);
    assert(y_type == arrow::Type::UINT32);

    //array{ArrayData{vector<Buffer{uint8_t*}>}}
    auto x_data = (uint32_t*)x_array->data()->GetValues<uint8_t >(1);
    auto y_data = (uint32_t*)y_array->data()->GetValues<uint8_t >(1);
    vertices_x_ = std::shared_ptr<uint32_t >(x_data);
    vertices_y_ = std::shared_ptr<uint32_t >(y_data);
}

void
PointMap::Draw() {
    glEnable(GL_PROGRAM_POINT_SIZE);

    glClear(GL_COLOR_BUFFER_BIT);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_POINT_SMOOTH);

    glDrawArrays(GL_POINTS, 0, num_vertices_);
    glFlush();

    glDeleteVertexArrays(1, &VAO_);
    glDeleteBuffers(2, VBO_);
}

void
PointMap::Shader() {
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

    const char *fragment_shader_source = \
        "#version 430 core\n"
        "out vec4 FragColor;\n"
        "layout (location = 4) uniform vec4 color;\n"
        "void main()\n"
        "{\n"
        "   FragColor = color.xyzw;\n"
        "}";

    int success;
    int vertex_shader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertex_shader, 1, &vertex_shader_source, NULL);
    glCompileShader(vertex_shader);
    glGetShaderiv(vertex_shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        std::cout << "vertex shader compile failed.";
    }

    int fragment_shader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragment_shader, 1, &fragment_shader_source, NULL);
    glCompileShader(fragment_shader);
    glGetShaderiv(vertex_shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        std::cout << "fragment shader compile failed.";
    }

    int shader_program = glCreateProgram();
    glAttachShader(shader_program, vertex_shader);
    glAttachShader(shader_program, fragment_shader);
    glLinkProgram(shader_program);
    glGetProgramiv(shader_program, GL_LINK_STATUS, &success);
    if (!success) {
        std::cout << "shader program link failed.";
    }

    glDeleteShader(vertex_shader);
    glDeleteShader(fragment_shader);

    glGenVertexArrays(1, &VAO_);
    glGenBuffers(2, VBO_);

    glBindVertexArray(VAO_);

    glBindBuffer(GL_ARRAY_BUFFER, VBO_[0]);
    glBufferData(GL_ARRAY_BUFFER, num_vertices_ * sizeof(uint32_t), vertices_x_.get(), GL_STATIC_DRAW);
    glVertexAttribPointer(0, 1, GL_FLOAT, GL_FALSE, 1 * sizeof(uint32_t), nullptr);

    glBindBuffer(GL_ARRAY_BUFFER, VBO_[1]);
    glBufferData(GL_ARRAY_BUFFER, num_vertices_ * sizeof(uint32_t), vertices_y_.get(), GL_STATIC_DRAW);
    glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 1 * sizeof(uint32_t), nullptr);

    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);

    glUseProgram(shader_program);
    auto window_params = window()->window_params();
    glUniform2f(2, window_params.width(), window_params.height());
    auto point_format = point_vega_.circle_params();
    glUniform1f(3, point_format.radius);
    glUniform4f(4, point_format.color.r, point_format.color.g, point_format.color.b, point_format.color.a);
}

std::shared_ptr<arrow::Array>
PointMap::Render(){
    InputInit();
    WindowsInit(point_vega_.window_params());
    DataInit();
    Shader();
    Draw();
    Finalize();
    return Output();
}

} //namespace render
} //namespace zilliz