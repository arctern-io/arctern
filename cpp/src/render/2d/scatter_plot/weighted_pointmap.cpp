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

#include "render/2d/scatter_plot/weighted_pointmap.h"
#include "render/utils/color/color_gradient.h"

namespace arctern {
namespace render {

template class WeightedPointMap<int8_t>;

template class WeightedPointMap<int16_t>;

template class WeightedPointMap<int32_t>;

template class WeightedPointMap<int64_t>;

template class WeightedPointMap<uint8_t>;

template class WeightedPointMap<uint16_t>;

template class WeightedPointMap<uint32_t>;

template class WeightedPointMap<uint64_t>;

template class WeightedPointMap<float>;

template class WeightedPointMap<double>;

template <typename T>
WeightedPointMap<T>::WeightedPointMap(uint32_t* vertices_x, uint32_t* vertices_y,
                                      size_t num_vertices)
    : vertices_x_(vertices_x),
      vertices_y_(vertices_y),
      unknown_(nullptr),
      color_count_(nullptr),
      size_count_(nullptr),
      num_vertices_(num_vertices) {}

template <typename T>
WeightedPointMap<T>::WeightedPointMap(uint32_t* vertices_x, uint32_t* vertices_y,
                                      T* unknown_count, size_t num_vertices)
    : vertices_x_(vertices_x),
      vertices_y_(vertices_y),
      unknown_(unknown_count),
      color_count_(nullptr),
      size_count_(nullptr),
      num_vertices_(num_vertices) {}

template <typename T>
WeightedPointMap<T>::WeightedPointMap(uint32_t* vertices_x, uint32_t* vertices_y,
                                      T* color_count, T* size_count, size_t num_vertices)
    : vertices_x_(vertices_x),
      vertices_y_(vertices_y),
      unknown_(nullptr),
      color_count_(color_count),
      size_count_(size_count),
      num_vertices_(num_vertices) {}

template <typename T>
void WeightedPointMap<T>::Draw() {
  glClear(GL_COLOR_BUFFER_BIT);

  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  glEnable(GL_POINT_SMOOTH);

  if (!mutable_weighted_point_vega().is_multiple_color() &&
      !mutable_weighted_point_vega().is_multiple_point_size() && size_count_ == nullptr &&
      color_count_ == nullptr && unknown_ == nullptr) {
    DrawSingleColorSingleSize();
  } else if (mutable_weighted_point_vega().is_multiple_color() &&
             !mutable_weighted_point_vega().is_multiple_point_size() &&
             size_count_ == nullptr && color_count_ == nullptr && unknown_ != nullptr) {
#ifndef USE_GPU
    SetColor(unknown_);
#endif
    DrawMultipleColorSingleSize();
  } else if (!mutable_weighted_point_vega().is_multiple_color() &&
             mutable_weighted_point_vega().is_multiple_point_size() &&
             size_count_ == nullptr && color_count_ == nullptr && unknown_ != nullptr) {
#ifndef USE_GPU
    SetPointSize(unknown_);
#endif
    DrawSingleColorMultipleSize();
  } else if (mutable_weighted_point_vega().is_multiple_color() &&
             mutable_weighted_point_vega().is_multiple_point_size() &&
             size_count_ != nullptr && color_count_ != nullptr && unknown_ == nullptr) {
#ifndef USE_GPU
    SetPointSize(size_count_);
    SetColor(color_count_);
#endif
    DrawMultipleColorMultipleSize();
  } else {
    std::string err_msg = "Draw failed, invalid point map";
    throw std::runtime_error(err_msg);
  }
}

#ifdef USE_GPU
template <typename T>
void WeightedPointMap<T>::Shader() {
  if (!mutable_weighted_point_vega().is_multiple_color() &&
      !mutable_weighted_point_vega().is_multiple_point_size() && size_count_ == nullptr &&
      color_count_ == nullptr && unknown_ == nullptr) {
    ShaderSingleColorSingleSize();
  } else if (mutable_weighted_point_vega().is_multiple_color() &&
             !mutable_weighted_point_vega().is_multiple_point_size() &&
             size_count_ == nullptr && color_count_ == nullptr && unknown_ != nullptr) {
    SetColor(unknown_);
    ShaderMultipleColorSingleSize();
  } else if (!mutable_weighted_point_vega().is_multiple_color() &&
             mutable_weighted_point_vega().is_multiple_point_size() &&
             size_count_ == nullptr && color_count_ == nullptr && unknown_ != nullptr) {
    SetPointSize(unknown_);
    ShaderSingleColorMultipleSize();
  } else if (mutable_weighted_point_vega().is_multiple_color() &&
             mutable_weighted_point_vega().is_multiple_point_size() &&
             size_count_ != nullptr && color_count_ != nullptr && unknown_ == nullptr) {
    SetPointSize(size_count_);
    SetColor(color_count_);
    ShaderMultipleColorMultipleSize();
  } else {
    std::string err_msg = "Shader failed, invalid point map";
    throw std::runtime_error(err_msg);
  }
}

template <typename T>
void WeightedPointMap<T>::ShaderSingleColorSingleSize() {
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
  glShaderSource(vertex_shader, 1, &vertex_shader_source, nullptr);
  glCompileShader(vertex_shader);
  glGetShaderiv(vertex_shader, GL_COMPILE_STATUS, &success);
#ifdef DEBUG_RENDER
  if (!success) {
    std::string err_msg = "vertex shader compile failed";
    throw std::runtime_error(err_msg);
  }
#endif

  int fragment_shader = glCreateShader(GL_FRAGMENT_SHADER);
  glShaderSource(fragment_shader, 1, &fragment_shader_source, nullptr);
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
  auto point_format = weighted_point_vega_.point_params();
  glUniform1f(3, point_format.point_size);
  glUniform4f(4, point_format.color.r, point_format.color.g, point_format.color.b,
              point_format.color.a);
}

template <typename T>
void WeightedPointMap<T>::ShaderMultipleColorSingleSize() {
  const char* vertexShaderSource =
      "#version 430 core\n"
      "layout (location = 0) in uint posX;\n"
      "layout (location = 1) in uint posY;\n"
      "layout (location = 2) in vec4 point_color;\n"
      "layout (location = 3) uniform vec2 screen_info;\n"
      "layout (location = 4) uniform float point_size;\n"
      "out vec4 color;\n"
      "void main()\n"
      "{\n"
      "   float tmp_x = posX;\n"
      "   float tmp_y = posY;\n"
      "   gl_Position = vec4(((tmp_x * 2) / screen_info.x) - 1, ((tmp_y * 2) / "
      "screen_info.y) - 1, 0, 1);\n"
      "   gl_PointSize = point_size;\n"
      "   color=point_color;\n"
      "}";

  const char* fragmentShaderSource =
      "#version 430 core\n"
      "in vec4 color;\n"
      "out vec4 FragColor;\n"
      "void main()\n"
      "{\n"
      "   FragColor = color;\n"
      "}";

  int success;
  int vertexShader = glCreateShader(GL_VERTEX_SHADER);
  glShaderSource(vertexShader, 1, &vertexShaderSource, nullptr);
  glCompileShader(vertexShader);
  glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
#ifdef DEBUG_RENDER
  if (!success) {
    std::string err_msg = "vertex shader compile failed";
    throw std::runtime_error(err_msg);
  }
#endif
  int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
  glShaderSource(fragmentShader, 1, &fragmentShaderSource, nullptr);
  glCompileShader(fragmentShader);
  glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
#ifdef DEBUG_RENDER
  if (!success) {
    std::string err_msg = "fragment shader compile failed";
    throw std::runtime_error(err_msg);
  }
#endif
  int shaderProgram = glCreateProgram();
  glAttachShader(shaderProgram, vertexShader);
  glAttachShader(shaderProgram, fragmentShader);
  glLinkProgram(shaderProgram);
  glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
#ifdef DEBUG_RENDER
  if (!success) {
    std::string err_msg = "shader program link failed";
    throw std::runtime_error(err_msg);
  }
#endif
  glDeleteShader(vertexShader);
  glDeleteShader(fragmentShader);

  glGenVertexArrays(1, &VAO_);
  glGenBuffers(4, VBO_);

  glBindVertexArray(VAO_);
  glBindBuffer(GL_ARRAY_BUFFER, VBO_[0]);
  glBufferData(GL_ARRAY_BUFFER, num_vertices_ * 1 * sizeof(uint32_t), vertices_x_,
               GL_STATIC_DRAW);
  glVertexAttribPointer(0, 1, GL_FLOAT, GL_TRUE, 1 * sizeof(uint32_t), (void*)nullptr);

  glBindBuffer(GL_ARRAY_BUFFER, VBO_[1]);
  glBufferData(GL_ARRAY_BUFFER, num_vertices_ * 1 * sizeof(uint32_t), vertices_y_,
               GL_STATIC_DRAW);
  glVertexAttribPointer(1, 1, GL_FLOAT, GL_TRUE, 1 * sizeof(uint32_t), (void*)nullptr);

  glBindBuffer(GL_ARRAY_BUFFER, VBO_[2]);
  glBufferData(GL_ARRAY_BUFFER, num_vertices_ * 4 * sizeof(float), &colors_[0],
               GL_STATIC_DRAW);
  glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)nullptr);

  glEnableVertexAttribArray(0);
  glEnableVertexAttribArray(1);
  glEnableVertexAttribArray(2);

  glUseProgram(shaderProgram);
  glUniform2f(3, window()->window_params().width(), window()->window_params().height());
  auto point_format = weighted_point_vega_.point_params();
  glUniform1f(4, point_format.point_size);
}

template <typename T>
void WeightedPointMap<T>::ShaderSingleColorMultipleSize() {
  const char* vertex_shader_source =
      "#version 430 core\n"
      "layout (location = 0) in uint posX;\n"
      "layout (location = 1) in uint posY;\n"
      "layout (location = 2) uniform vec2 screen_info;\n"
      "layout (location = 3) in uint point_size;\n"
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
  glShaderSource(vertex_shader, 1, &vertex_shader_source, nullptr);
  glCompileShader(vertex_shader);
  glGetShaderiv(vertex_shader, GL_COMPILE_STATUS, &success);
#ifdef DEBUG_RENDER
  if (!success) {
    std::string err_msg = "vertex shader compile failed";
    throw std::runtime_error(err_msg);
  }
#endif

  int fragment_shader = glCreateShader(GL_FRAGMENT_SHADER);
  glShaderSource(fragment_shader, 1, &fragment_shader_source, nullptr);
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

  std::vector<uint32_t> point_size(num_vertices_);
  for (int i = 0; i < num_vertices_; i++) {
    point_size[i] = (uint32_t)unknown_[i];
  }
  glBindBuffer(GL_ARRAY_BUFFER, VBO_[3]);
  glBufferData(GL_ARRAY_BUFFER, num_vertices_ * sizeof(uint32_t), &point_size[0],
               GL_STATIC_DRAW);
  glVertexAttribPointer(3, 1, GL_FLOAT, GL_FALSE, sizeof(uint32_t), nullptr);

  glEnableVertexAttribArray(0);
  glEnableVertexAttribArray(1);
  glEnableVertexAttribArray(3);

  glUseProgram(shader_program);
  auto window_params = window()->window_params();
  glUniform2f(2, window_params.width(), window_params.height());
  auto point_format = weighted_point_vega_.point_params();
  glUniform4f(4, point_format.color.r, point_format.color.g, point_format.color.b,
              point_format.color.a);
}

template <typename T>
void WeightedPointMap<T>::ShaderMultipleColorMultipleSize() {
  const char* vertexShaderSource =
      "#version 430 core\n"
      "layout (location = 0) in uint posX;\n"
      "layout (location = 1) in uint posY;\n"
      "layout (location = 2) in vec4 point_color;\n"
      "layout (location = 3) uniform vec2 screen_info;\n"
      "layout (location = 4) in uint point_size;\n"
      "out vec4 color;\n"
      "void main()\n"
      "{\n"
      "   float tmp_x = posX;\n"
      "   float tmp_y = posY;\n"
      "   gl_Position = vec4(((tmp_x * 2) / screen_info.x) - 1, ((tmp_y * 2) / "
      "screen_info.y) - 1, 0, 1);\n"
      "   gl_PointSize = point_size;\n"
      "   color=point_color;\n"
      "}";

  const char* fragmentShaderSource =
      "#version 430 core\n"
      "in vec4 color;\n"
      "out vec4 FragColor;\n"
      "void main()\n"
      "{\n"
      "   FragColor = color;\n"
      "}";

  int success;
  int vertexShader = glCreateShader(GL_VERTEX_SHADER);
  glShaderSource(vertexShader, 1, &vertexShaderSource, nullptr);
  glCompileShader(vertexShader);
  glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
#ifdef DEBUG_RENDER
  if (!success) {
    std::string err_msg = "vertex shader compile failed";
    throw std::runtime_error(err_msg);
  }
#endif
  int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
  glShaderSource(fragmentShader, 1, &fragmentShaderSource, nullptr);
  glCompileShader(fragmentShader);
  glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
#ifdef DEBUG_RENDER
  if (!success) {
    std::string err_msg = "fragment shader compile failed";
    throw std::runtime_error(err_msg);
  }
#endif
  int shaderProgram = glCreateProgram();
  glAttachShader(shaderProgram, vertexShader);
  glAttachShader(shaderProgram, fragmentShader);
  glLinkProgram(shaderProgram);
  glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
#ifdef DEBUG_RENDER
  if (!success) {
    std::string err_msg = "shader program link failed";
    throw std::runtime_error(err_msg);
  }
#endif
  glDeleteShader(vertexShader);
  glDeleteShader(fragmentShader);

  glGenVertexArrays(1, &VAO_);
  glGenBuffers(4, VBO_);

  glBindVertexArray(VAO_);
  glBindBuffer(GL_ARRAY_BUFFER, VBO_[0]);
  glBufferData(GL_ARRAY_BUFFER, num_vertices_ * 1 * sizeof(uint32_t), vertices_x_,
               GL_STATIC_DRAW);
  glVertexAttribPointer(0, 1, GL_FLOAT, GL_TRUE, 1 * sizeof(uint32_t), (void*)nullptr);

  glBindBuffer(GL_ARRAY_BUFFER, VBO_[1]);
  glBufferData(GL_ARRAY_BUFFER, num_vertices_ * 1 * sizeof(uint32_t), vertices_y_,
               GL_STATIC_DRAW);
  glVertexAttribPointer(1, 1, GL_FLOAT, GL_TRUE, 1 * sizeof(uint32_t), (void*)nullptr);

  glBindBuffer(GL_ARRAY_BUFFER, VBO_[2]);
  glBufferData(GL_ARRAY_BUFFER, num_vertices_ * 4 * sizeof(float), &colors_[0],
               GL_STATIC_DRAW);
  glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)nullptr);

  std::vector<uint32_t> point_size(num_vertices_);
  for (int i = 0; i < num_vertices_; i++) {
    point_size[i] = (uint32_t)size_count_[i];
  }
  glBindBuffer(GL_ARRAY_BUFFER, VBO_[3]);
  glBufferData(GL_ARRAY_BUFFER, num_vertices_ * sizeof(uint32_t), &point_size[0],
               GL_STATIC_DRAW);
  glVertexAttribPointer(4, 1, GL_FLOAT, GL_FALSE, 1 * sizeof(uint32_t), nullptr);

  glEnableVertexAttribArray(0);
  glEnableVertexAttribArray(1);
  glEnableVertexAttribArray(2);
  glEnableVertexAttribArray(4);

  glUseProgram(shaderProgram);
  glUniform2f(3, window()->window_params().width(), window()->window_params().height());
}

#endif

template <typename T>
void WeightedPointMap<T>::DrawSingleColorSingleSize() {
  glClear(GL_COLOR_BUFFER_BIT);

  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  glEnable(GL_POINT_SMOOTH);

#ifndef USE_GPU
  glOrtho(0, window()->window_params().width(), 0, window()->window_params().height(), -1,
          1);

  glPointSize(weighted_point_vega_.point_params().point_size);

  auto& color = weighted_point_vega_.point_params().color;
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
  glDeleteBuffers(4, VBO_);
#endif
}

template <typename T>
void WeightedPointMap<T>::DrawSingleColorMultipleSize() {
#ifndef USE_GPU
  glOrtho(0, window()->window_params().width(), 0, window()->window_params().height(), -1,
          1);

  auto& color = weighted_point_vega_.point_params().color;
  glColor4f(color.r, color.g, color.b, color.a);

  for (int i = 0; i < num_vertices_; i++) {
    glPointSize(unknown_[i]);
    glBegin(GL_POINTS);
    glVertex2d(vertices_x_[i], vertices_y_[i]);
    glEnd();
  }

  glFlush();

#else
  glEnable(GL_PROGRAM_POINT_SIZE);

  glDrawArrays(GL_POINTS, 0, num_vertices_);
  glFlush();

  glDeleteVertexArrays(1, &VAO_);
  glDeleteBuffers(4, VBO_);
#endif
}

template <typename T>
void WeightedPointMap<T>::DrawMultipleColorSingleSize() {
#ifndef USE_GPU
  glOrtho(0, window()->window_params().width(), 0, window()->window_params().height(), -1,
          1);

  glPointSize(weighted_point_vega_.point_params().point_size);

  glEnableClientState(GL_VERTEX_ARRAY);
  glEnableClientState(GL_COLOR_ARRAY);

  int offset = 0;
  std::vector<int32_t> vertices(num_vertices_ * 2);
  for (auto i = 0; i < num_vertices_; i++) {
    vertices[offset++] = vertices_x_[i];
    vertices[offset++] = vertices_y_[i];
  }
  glColorPointer(4, GL_FLOAT, 0, &colors_[0]);
  glVertexPointer(2, GL_INT, 0, &vertices[0]);

  glDrawArrays(GL_POINTS, 0, num_vertices_);
  glFlush();

#else
  glEnable(GL_PROGRAM_POINT_SIZE);

  glDrawArrays(GL_POINTS, 0, num_vertices_);
  glFlush();

  glDeleteVertexArrays(1, &VAO_);
  glDeleteBuffers(4, VBO_);
#endif
}

template <typename T>
void WeightedPointMap<T>::DrawMultipleColorMultipleSize() {
#ifndef USE_GPU
  glOrtho(0, window()->window_params().width(), 0, window()->window_params().height(), -1,
          1);

  size_t c_offset = 0;

  for (int i = 0; i < num_vertices_; i++) {
    auto r = colors_[c_offset++];
    auto g = colors_[c_offset++];
    auto b = colors_[c_offset++];
    auto a = colors_[c_offset++];
    glColor4f(r, g, b, a);
    glPointSize(size_count_[i]);
    glBegin(GL_POINTS);
    glVertex2d(vertices_x_[i], vertices_y_[i]);
    glEnd();
  }

  glFlush();

#else
  glEnable(GL_PROGRAM_POINT_SIZE);

  glDrawArrays(GL_POINTS, 0, num_vertices_);
  glFlush();

  glDeleteVertexArrays(1, &VAO_);
  glDeleteBuffers(4, VBO_);
#endif
}

template <typename T>
void WeightedPointMap<T>::SetColor(T* ptr) {
  colors_.resize(num_vertices_ * 4);

  auto count_start = weighted_point_vega_.color_bound().first;
  auto count_end = weighted_point_vega_.color_bound().second;
  auto count_range = count_end - count_start;

  size_t c_offset = 0;
  for (auto i = 0; i < num_vertices_; i++) {
    auto color_gradient = weighted_point_vega_.color_gradient();
    if (color_gradient.size() == 1) {
      auto color = color_gradient[0];
      colors_[c_offset++] = color.r;
      colors_[c_offset++] = color.g;
      colors_[c_offset++] = color.b;
      colors_[c_offset++] = color.a;
    } else {
      auto color_start = color_gradient[0];
      auto color_end = color_gradient[1];
      auto count = ptr[i] >= count_start ? ptr[i] : count_start;
      count = ptr[i] <= count_end ? count : count_end;
      auto ratio = (count - count_start) / count_range;
      auto color = ColorGradient::GetColor(color_start, color_end, ratio);
      colors_[c_offset++] = color.r;
      colors_[c_offset++] = color.g;
      colors_[c_offset++] = color.b;
      colors_[c_offset++] = color.a;
    }
  }
}

template <typename T>
void WeightedPointMap<T>::SetPointSize(T* ptr) {
  auto count_start = weighted_point_vega_.size_bound().first;
  auto count_end = weighted_point_vega_.size_bound().second;

  for (auto i = 0; i < num_vertices_; i++) {
    ptr[i] = ptr[i] >= count_start ? ptr[i] : count_start;
    ptr[i] = ptr[i] <= count_end ? ptr[i] : count_end;
  }
}

template <typename T>
std::vector<uint8_t> WeightedPointMap<T>::Render() {
  WindowsInit(weighted_point_vega_.window_params());
#ifdef USE_GPU
  Shader();
#endif
  Draw();
  Finalize();
  return Output();
}

}  // namespace render
}  // namespace arctern
