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

#include "render/window/window_params.h"

namespace arctern {
namespace render {

enum class WindowType { kUnknown = 0, k2D };

class Window {
 public:
  virtual void Init() = 0;

  virtual void Terminate() = 0;

 public:
  const WindowParams window_params() const { return window_params_; }

  void set_window_params(WindowParams window_params) { window_params_ = window_params; }

  void set_window_type(WindowType window_type) { window_type_ = window_type; }

 protected:
  WindowType window_type_;
  WindowParams window_params_;
};

using WindowPtr = std::shared_ptr<Window>;

}  // namespace render
}  // namespace arctern
