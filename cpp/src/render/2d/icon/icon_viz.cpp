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
#include <utility>
#include <vector>
#define GL_GLEXT_PROTOTYPES
#include <GL/gl.h>
#include <GL/glext.h>

#include "render/2d/icon/icon_viz.h"
#include "src/render/utils/image/image_loader.h"

namespace arctern {
namespace render {

IconViz::IconViz(uint32_t* input_x, uint32_t* input_y, int64_t num_icons)
    : vertices_x_(input_x), vertices_y_(input_y), num_icons_(num_icons) {}

void IconViz::Draw() {
#ifndef USE_GPU
  glOrtho(0, window()->window_params().width(), 0, window()->window_params().height(), -1,
          1);
#endif

  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  std::string icon_path = icon_vega_.icon_path();

  auto image_loader = ImageLoader::GetInstance();
  auto image_buffer = image_loader.Load(icon_path);

  if (image_buffer.buffer == nullptr) {
    buffer_ = {};
    std::string err_msg =
        "image buffer is empty, please make sure there is a icon in " + icon_path;
    throw std::runtime_error(err_msg);
  }

  for (int i = 0; i < num_icons_; i++) {
    glRasterPos2f(vertices_x_[i], vertices_y_[i]);
    glDrawPixels(image_buffer.image_params.width, image_buffer.image_params.height,
                 GL_RGBA, GL_UNSIGNED_BYTE, image_buffer.buffer);
  }

  glFinish();
}

uint8_t* IconViz::Render() {
  WindowsInit(icon_vega_.window_params());
  Draw();
  Finalize();
  return Output();
}

}  // namespace render
}  // namespace arctern
