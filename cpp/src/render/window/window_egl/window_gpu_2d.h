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

#include "render/window/window_egl/window_gpu.h"

namespace zilliz {
namespace render {

class WindowGPU2D : public WindowGPU {
 public:
  void Init() final;

  void Terminate() final;
};

using WindowGPU2DPtr = std::shared_ptr<WindowGPU2D>;

}  // namespace render
}  // namespace zilliz
