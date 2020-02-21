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

#include <GL/osmesa.h>
#include <memory>

#include "render/window/window.h"

namespace zilliz {
namespace render {

class WindowCPU : public Window {
 public:
  virtual void Init() = 0;

  virtual void Terminate() = 0;

  GLubyte* mutable_buffer() { return buffer_; }

 protected:
  GLubyte* buffer_;
  OSMesaContext context_;
};

using WindowCPUPtr = std::shared_ptr<WindowCPU>;

}  // namespace render
}  // namespace zilliz
