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
#include <string>
#include <vector>
#define GL_GLEXT_PROTOTYPES
#include <GL/gl.h>
#include <GL/glext.h>

#include "render/2d/squaremap/square_map.h"
#include "render/utils/color/color_gradient.h"

namespace arctern {
namespace render {

template class SquareMap<int8_t>;

template class SquareMap<int16_t>;

template class SquareMap<int32_t>;

template class SquareMap<int64_t>;

template class SquareMap<uint8_t>;

template class SquareMap<uint16_t>;

template class SquareMap<uint32_t>;

template class SquareMap<uint64_t>;

template class SquareMap<float>;

template class SquareMap<double>;

template <typename T>
SquareMap<T>::SquareMap(uint32_t* input_x, uint32_t* input_y, T* count, int64_t num_vertices)
    : vertices_x_(input_x),
      vertices_y_(input_y),
      count_(count),
      num_vertices_(num_vertices) {}

template <typename T>
SquareMap<T>::~SquareMap() {
  free(vertices_x_);
  free(vertices_y_);
  free(colors_);
}

template <typename T>
void set_colors(float* colors, uint32_t* input_x, uint32_t* input_y, T* input_c,
                    int64_t num, VegaSquareMap& vega_square_map) {
  WindowParams window_params = vega_square_map.window_params();
  int64_t width = window_params.width();
  int64_t height = window_params.height();
  int64_t window_size = width * height;
  int square_size = vega_square_map.square_size();
  int square_spacing = vega_square_map.square_spacing();
  // int block_size = square_size + square_spacing;

  std::cout << "set color, num = "<< num <<std::endl;
  std::vector<T> weights(num);
  memcpy(&weights[0], input_c, num*sizeof(T));
  std::sort(weights.begin(), weights.end());
  int max = (int)(num * 99 / 100);
  T max_pix = weights[max];
  T min_pix = weights[0];
  T count_range = max_pix - min_pix;
  // std::cout << "set color, max_pix = "<< max_pix <<std::endl;
  // std::cout << "set color, min_pix = "<< min_pix <<std::endl;
  // std::cout << "set color, max = "<< max <<std::endl;
  // std::cout << "set color, max + 1 pix = "<< weights[max + 1] <<std::endl;
  ColorGradient color_gradient;
  color_gradient.createDefaultHeatMapGradient();

  for(int i = 0; i < num; i++) {
    // float value = max_pix == min_pix ? 0.0f : (input_c[i] - min_pix) / count_range;
    float value = input_c[i] > max_pix ? 1.0f : (input_c[i] - min_pix) / count_range;
    float color_r, color_g, color_b;
    color_gradient.getColorAtValue(value, color_r, color_g, color_b);

    // if((input_y[i] * square_size)>=height || (input_x[i]* square_size) >= width) continue;
    // int index = (square_size * input_y[i] + 2) * width + input_x[i] * square_size +2;
    if((input_y[i] * square_size)>=height || (input_x[i]* square_size) >= width) continue;
    int index = square_size * input_y[i] * width + input_x[i] * square_size;
    for(int m = 0; m < square_size - square_spacing; m++) {
      for(int n = 0; n < square_size - square_spacing; n++) {
          int index_in = (index + m * width + n) * 4;
          colors[index_in++] = color_r;
          colors[index_in++] = color_g;
          colors[index_in++] = color_b;
          colors[index_in++] = vega_square_map.opacity();
      }
    }
  }
}

template <typename T>
void SquareMap<T>::DataInit() {
  WindowParams window_params = square_vega_.window_params();
  int64_t width = window_params.width();
  int64_t height = window_params.height();
  int64_t window_size = width * height;

  colors_ = (float*)malloc(window_size * 4 * sizeof(float));
  set_colors(colors_, vertices_x_, vertices_y_, count_, num_vertices_, square_vega_);
  vertices_x_ = (uint32_t*)malloc(window_size * sizeof(uint32_t));
  vertices_y_ = (uint32_t*)malloc(window_size * sizeof(uint32_t));
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      vertices_x_[i * width + j] = j;
      vertices_y_[i * width + j] = i;
    }
  }
  num_vertices_ = window_size;
}

template <typename T>
void SquareMap<T>::Draw() {
  //    glEnable(GL_PROGRAM_POINT_SIZE);
  glClear(GL_COLOR_BUFFER_BIT);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  glEnable(GL_POINT_SMOOTH);

#ifdef USE_GPU
  glDrawArrays(GL_POINTS, 0, num_vertices_);
  glFlush();

  glDeleteVertexArrays(1, &VAO_);
  glDeleteBuffers(2, VBO_);
#else
  glOrtho(0, window()->window_params().width(), 0, window()->window_params().height(), -1,
          1);
  glEnableClientState(GL_VERTEX_ARRAY);
  glEnableClientState(GL_COLOR_ARRAY);

  int offset = 0;
  std::vector<int32_t> vertices(num_vertices_ * 2);
  for (auto i = 0; i < num_vertices_; i++) {
    vertices[offset++] = vertices_x_[i];
    vertices[offset++] = vertices_y_[i];
  }
  glColorPointer(4, GL_FLOAT, 0, colors_);
  glVertexPointer(2, GL_INT, 0, &vertices[0]);

  glDrawArrays(GL_POINTS, 0, num_vertices_);
  glFlush();
#endif
}

#ifdef USE_GPU
template <typename T>
void SquareMap<T>::Shader() {
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
uint8_t* SquareMap<T>::Render() {
  WindowsInit(square_vega_.window_params());
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
