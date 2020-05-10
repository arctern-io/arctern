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
#include <algorithm>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>
#define GL_GLEXT_PROTOTYPES
#include <GL/gl.h>
#include <GL/glext.h>

#include "render/2d/fishnet_map/fishnet_map.h"
#include "render/utils/color/color_gradient.h"

namespace arctern {
namespace render {

template class FishNetMap<int8_t>;

template class FishNetMap<int16_t>;

template class FishNetMap<int32_t>;

template class FishNetMap<int64_t>;

template class FishNetMap<uint8_t>;

template class FishNetMap<uint16_t>;

template class FishNetMap<uint32_t>;

template class FishNetMap<uint64_t>;

template class FishNetMap<float>;

template class FishNetMap<double>;

template <typename T>
FishNetMap<T>::FishNetMap(uint32_t* input_x, uint32_t* input_y, T* count,
                          int64_t num_vertices)
    : vertices_x_(input_x),
      vertices_y_(input_y),
      count_(count),
      num_vertices_(num_vertices) {}

template <typename T>
FishNetMap<T>::~FishNetMap() {
  free(colors_);
}

template <typename T>
void set_colors(float* colors, uint32_t* input_x, uint32_t* input_y, T* input_c,
                int64_t num, VegaFishNetMap& vega_fishnet_map) {
  WindowParams window_params = vega_fishnet_map.window_params();
  int64_t width = window_params.width();
  int64_t height = window_params.height();
  int64_t window_size = width * height;
  int cell_size = vega_fishnet_map.cell_size();
  int cell_spacing = vega_fishnet_map.cell_spacing();
  int block_size = cell_size + cell_spacing;

  std::vector<T> weights(num);
  memcpy(&weights[0], input_c, num * sizeof(T));
  std::sort(weights.begin(), weights.end());
  int max = (int)(num * 99 / 100);
  T max_pix = weights[max];
  T min_pix = weights[0];
  T count_range = max_pix - min_pix;
  ColorGradient color_gradient;
  color_gradient.createDefaultHeatMapGradient();

  for (int i = 0; i < num; i++) {
    if (input_y[i] * block_size >= height || input_x[i] * block_size >= width) continue;
    float value = input_c[i] > max_pix ? 1.0f : (input_c[i] - min_pix) / count_range;
    float color_r, color_g, color_b;
    color_gradient.getColorAtValue(value, color_r, color_g, color_b);
    auto index = i * 4;
    colors[index++] = color_r;
    colors[index++] = color_g;
    colors[index++] = color_b;
    colors[index++] = vega_fishnet_map.opacity();
  }
}

template <typename T>
void FishNetMap<T>::DataInit() {
  WindowParams window_params = fishnet_vega_.window_params();
  int64_t width = window_params.width();
  int64_t height = window_params.height();
  int64_t window_size = width * height;

  colors_ = (float*)malloc(num_vertices_ * 4 * sizeof(float));
  set_colors(colors_, vertices_x_, vertices_y_, count_, num_vertices_, fishnet_vega_);
}

template <typename T>
void FishNetMap<T>::Draw() {
  glClear(GL_COLOR_BUFFER_BIT);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ZERO);
  glLineWidth(20);
#ifdef USE_GPU
  glDrawArrays(GL_POINTS, 0, num_vertices_);
  glFlush();

  glDeleteVertexArrays(1, &VAO_);
  glDeleteBuffers(2, VBO_);
#else
  int width = window()->window_params().width();
  int height = window()->window_params().height();
  int cell_size = fishnet_vega_.cell_size();
  int cell_spacing = fishnet_vega_.cell_spacing();
  int block_size = cell_size + cell_spacing;
  double spacing = (double)cell_spacing / 2;

  glOrtho(0, width, 0, height, -1, 1);
  for (int i = 0; i < num_vertices_; i++) {
    if (vertices_x_[i] * block_size >= width || vertices_y_[i] * block_size >= height)
      continue;
    glColor4f(colors_[i * 4], colors_[i * 4 + 1], colors_[i * 4 + 2], colors_[i * 4 + 3]);
    glBegin(GL_POLYGON);
    double x = vertices_x_[i] * block_size + spacing;
    double y = vertices_y_[i] * block_size + spacing;
    glVertex2d(x + (double)cell_size, y + (double)cell_size);
    glVertex2d(x + (double)cell_size, y);
    glVertex2d(x, y);
    glVertex2d(x, y + (double)cell_size);
    glEnd();
  }
  glFinish();
#endif
}

#ifdef USE_GPU
template <typename T>
void FishNetMap<T>::Shader() {
  const char* vertexShaderSource =
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
      "   gl_Position = vec4(((tmp_x * 2) / screen_info.x) - 1, ((tmp_y * 2) / "
      "screen_info.y) - 1, 0, 1);\n"
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
  glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
  glCompileShader(vertexShader);
  glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
#ifdef DEBUG_RENDER
  if (!success) {
    std::string err_msg = "vertex shader compile failed";
    throw std::runtime_error(err_msg);
  }
#endif
  int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
  glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
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
  glGenBuffers(3, VBO_);

  glBindVertexArray(VAO_);
  glBindBuffer(GL_ARRAY_BUFFER, VBO_[0]);
  glBufferData(GL_ARRAY_BUFFER, num_vertices_ * 1 * sizeof(uint32_t), vertices_x_,
               GL_STATIC_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, VBO_[1]);
  glBufferData(GL_ARRAY_BUFFER, num_vertices_ * 1 * sizeof(uint32_t), vertices_y_,
               GL_STATIC_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, VBO_[2]);
  glBufferData(GL_ARRAY_BUFFER, num_vertices_ * 4 * sizeof(float), colors_,
               GL_STATIC_DRAW);

  glGenVertexArrays(1, &VAO_);
  glBindVertexArray(VAO_);

  glBindBuffer(GL_ARRAY_BUFFER, VBO_[0]);
  glVertexAttribPointer(0, 1, GL_FLOAT, GL_TRUE, 1 * sizeof(uint32_t), (void*)0);
  glBindBuffer(GL_ARRAY_BUFFER, VBO_[1]);
  glVertexAttribPointer(1, 1, GL_FLOAT, GL_TRUE, 1 * sizeof(uint32_t), (void*)0);
  glBindBuffer(GL_ARRAY_BUFFER, VBO_[2]);
  glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);

  glEnableVertexAttribArray(0);
  glEnableVertexAttribArray(1);
  glEnableVertexAttribArray(2);

  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindVertexArray(0);

  glBindBuffer(GL_ARRAY_BUFFER, 1);
  glBindVertexArray(1);

  glUseProgram(shaderProgram);
  glUniform2f(3, window()->window_params().width(), window()->window_params().height());
  glBindVertexArray(VAO_);
}
#endif

template <typename T>
uint8_t* FishNetMap<T>::Render() {
  WindowsInit(fishnet_vega_.window_params());
  DataInit();
#ifdef USE_GPU
  Shader();
#endif
  Draw();
  Finalize();
  return Output();
}

}  // namespace render
}  // namespace arctern
