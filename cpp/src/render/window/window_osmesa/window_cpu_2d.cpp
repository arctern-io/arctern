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
#include <stdexcept>
#include <string>

#include "render/window/window_osmesa/window_cpu_2d.h"

namespace arctern {
namespace render {

void WindowCPU2D::Init() {
  // Init an RGBA-mode context.
#if OSMESA_MAJOR_VERSION * 100 + OSMESA_MINOR_VERSION >= 305
  // Specify Z, stencil, accum sizes.
  context_ = OSMesaCreateContextExt(OSMESA_RGBA, 0, 0, 0, nullptr);
#else
  ctx = OSMesaCreateContext(OSMESA_RGBA, NULL);
#endif
  if (!context_) {
    std::string err_msg = "OSMesaCreateContext failed";
    throw std::runtime_error(err_msg);
  }

  GLsizei screen_width = (GLsizei)window_params().width();
  GLsizei screen_height = (GLsizei)window_params().height();

  // Init buffer for context.
  buffer_ = (GLubyte*)malloc(screen_width * screen_height * 4 * sizeof(GLubyte));

  // Bind the buffer to the context and make it current.
  if (!OSMesaMakeCurrent(context_, buffer_, GL_UNSIGNED_BYTE, screen_width,
                         screen_height)) {
    std::string err_msg = "OSMesaMakeCurrent failed";
    throw std::runtime_error(err_msg);
  }
}

void WindowCPU2D::Terminate() { OSMesaDestroyContext(context_); }

}  // namespace render
}  // namespace arctern
