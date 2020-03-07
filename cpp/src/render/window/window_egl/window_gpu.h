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
#define MESA_EGL_NO_X11_HEADERS 1
#include <EGL/egl.h>
#include <GL/gl.h>

#include "render/window/window.h"

namespace arctern {
namespace render {

class WindowGPU : public Window {
 public:
  virtual void Init() = 0;

  virtual void Terminate() = 0;

 public:
  EGLDisplay& mutable_egl_dpy() { return egl_dpy_; }

  EGLDisplay& mutable_egl_surf() { return egl_surf_; }

  EGLContext& mutable_egl_context() { return egl_context_; }

 protected:
  EGLDisplay egl_dpy_;
  EGLSurface egl_surf_;
  EGLContext egl_context_;
};

using WindowGPUPtr = std::shared_ptr<WindowGPU>;

}  // namespace render
}  // namespace arctern
