/*
 * Copyright (C) 2019-2020 Zilliz. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <iostream>
#include <map>
#include <string>
#include <vector>
#define GL_GLEXT_PROTOTYPES
#include <GL/gl.h>
#include <GL/glext.h>

#include "render/2d/scatter_plot/pointmap.h"

namespace arctern {
namespace render {

PointMap::PointMap(uint32_t* input_x, uint32_t* input_y, int64_t num_vertices)
    : vertices_x_(input_x), vertices_y_(input_y), num_vertices_(num_vertices) {}

void PointMap::Draw() {
  glClear(GL_COLOR_BUFFER_BIT);

  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  glEnable(GL_POINT_SMOOTH);

#ifndef USE_GPU
  glOrtho(0, window()->window_params().width(), 0, window()->window_params().height(), -1,
          1);

  glPointSize(point_vega_.point_params().point_size);
  auto& color = point_vega_.point_params().color;
  glColor4f(color.r, color.g, color.b, color.a);
  glEnableClientState(GL_VERTEX_ARRAY);

  int offset = 0;
  std::vector<int32_t> vertices(num_vertices_ * 2);
  for (auto i = 0; i < num_vertices_; i++) {
    vertices[offset++] = vertices_x_[i];
    vertices[offset++] = vertices_y_[i];
  }
  glVertexPointer(2, GL_INT, 0, &vertices[0]);
  glDrawArrays(GL_POINTS, 0, num_vertices_);
  glFinish();

#else
  glEnable(GL_PROGRAM_POINT_SIZE);

  glDrawArrays(GL_POINTS, 0, num_vertices_);
  glFlush();

  glDeleteVertexArrays(1, &VAO_);
  glDeleteBuffers(2, VBO_);
#endif
}

#ifdef USE_GPU
void PointMap::Shader() {
  const char* vertex_shader_source =
      "#version 430 core\n"
      "layout (location = 0) in uint posX;\n"
      "layout (location = 1) in uint posY;\n"
      "layout (location = 2) uniform vec2 screen_info;\n"
      "layout (location = 3) uniform float point_size;\n"
      "void main()\n"
      "{\n"
      "   float tmp_x = posX;\n"
      "   float tmp_y = posY;\n"
      "   gl_Position = vec4(((tmp_x * 2) / screen_info.x) - 1, ((tmp_y * 2) / "
      "screen_info.y) - 1, 0, 1);\n"
      "   gl_PointSize = point_size;\n"
      "}";

  const char* fragment_shader_source =
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
#ifdef DEBUG_RENDER
  if (!success) {
    std::string err_msg = "vertex shader compile failed";
    throw std::runtime_error(err_msg);
  }
#endif

  int fragment_shader = glCreateShader(GL_FRAGMENT_SHADER);
  glShaderSource(fragment_shader, 1, &fragment_shader_source, NULL);
  glCompileShader(fragment_shader);
  glGetShaderiv(vertex_shader, GL_COMPILE_STATUS, &success);
#ifdef DEBUG_RENDER
  if (!success) {
    std::string err_msg = "fragment shader compile failed";
    throw std::runtime_error(err_msg);
  }
#endif

  int shader_program = glCreateProgram();
  glAttachShader(shader_program, vertex_shader);
  glAttachShader(shader_program, fragment_shader);
  glLinkProgram(shader_program);
  glGetProgramiv(shader_program, GL_LINK_STATUS, &success);
#ifdef DEBUG_RENDER
  if (!success) {
    std::string err_msg = "shader program link failed";
    throw std::runtime_error(err_msg);
  }
#endif

  glDeleteShader(vertex_shader);
  glDeleteShader(fragment_shader);

  glGenVertexArrays(1, &VAO_);
  glGenBuffers(2, VBO_);

  glBindVertexArray(VAO_);

  glBindBuffer(GL_ARRAY_BUFFER, VBO_[0]);
  glBufferData(GL_ARRAY_BUFFER, num_vertices_ * sizeof(uint32_t), vertices_x_,
               GL_STATIC_DRAW);
  glVertexAttribPointer(0, 1, GL_FLOAT, GL_FALSE, 1 * sizeof(uint32_t), nullptr);

  glBindBuffer(GL_ARRAY_BUFFER, VBO_[1]);
  glBufferData(GL_ARRAY_BUFFER, num_vertices_ * sizeof(uint32_t), vertices_y_,
               GL_STATIC_DRAW);
  glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 1 * sizeof(uint32_t), nullptr);

  glEnableVertexAttribArray(0);
  glEnableVertexAttribArray(1);

  glUseProgram(shader_program);
  auto window_params = window()->window_params();
  glUniform2f(2, window_params.width(), window_params.height());
  auto point_format = point_vega_.point_params();
  glUniform1f(3, point_format.point_size);
  glUniform4f(4, point_format.color.r, point_format.color.g, point_format.color.b,
              point_format.color.a);
}
#endif

std::vector<uint8_t> PointMap::Render() {
  WindowsInit(point_vega_.window_params());
#ifdef USE_GPU
  Shader();
#endif
  Draw();
  Finalize();
  return Output();
}

}  // namespace render
}  // namespace arctern
