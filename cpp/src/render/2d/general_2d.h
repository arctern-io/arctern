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
#pragma once

#include <memory>
#include <vector>

#include "render/2d/input.h"
#ifdef USE_GPU
#include "render/window/window_egl/window_gpu_2d.h"
#else
#include "render/window/window_osmesa/window_cpu_2d.h"
#endif

namespace arctern {
namespace render {

class General2D {
 public:
  ~General2D();

  virtual std::vector<uint8_t> Render() = 0;

  virtual void Draw() = 0;

 protected:
  void WindowsInit(WindowParams window_params);

  void Finalize();

  std::vector<uint8_t> Output();

  void InitBuffer(WindowParams& window_params);

  void ExportImage();

 public:
  unsigned char* mutable_buffer() { return buffer_; }

 protected:
  arrow::ArrayVector array_vector_;
  unsigned char* buffer_;
  unsigned char* output_image_;
  int output_image_size_;

#ifndef USE_GPU

 public:
  void set_window(WindowCPU2DPtr window) { window_ = window; }

  const WindowCPU2DPtr& window() const { return window_; }

  WindowCPU2DPtr mutable_window() { return window_; }

 protected:
  WindowCPU2DPtr window_;
#else

 public:
  void set_window(WindowGPU2DPtr window) { window_ = window; }

  const WindowGPU2DPtr& window() const { return window_; }

  WindowGPU2DPtr mutable_window() { return window_; }

 protected:
  WindowGPU2DPtr window_;
#endif
};

using General2DPtr = std::shared_ptr<General2D>;

}  // namespace render
}  // namespace arctern
