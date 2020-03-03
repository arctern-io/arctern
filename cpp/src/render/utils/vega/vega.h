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

#include <string>

#include "rapidjson/document.h"
#include "render/window/window_params.h"

namespace zilliz {
namespace render {

/***
 * TODO: add comments
 */
class Vega {
 public:
  //    std::string ToString();

  virtual std::string Build() = 0;

  const WindowParams& window_params() const { return window_params_; }

 protected:
  // vega json to vega struct
  virtual void Parse(const std::string& json) = 0;

  bool JsonLabelCheck(rapidjson::Value& value, const std::string& label);

  bool JsonSizeCheck(rapidjson::Value& value, const std::string& label, size_t size);

  bool JsonTypeCheck(rapidjson::Value& value, rapidjson::Type type);

  bool JsonNullCheck(rapidjson::Value& value);

 protected:
  WindowParams window_params_;
};

}  // namespace render
}  // namespace zilliz
