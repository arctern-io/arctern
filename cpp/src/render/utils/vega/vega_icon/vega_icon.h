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

#include "render/utils/color/color.h"
#include "render/utils/vega/vega.h"

namespace arctern {
namespace render {

class VegaIcon : public Vega {
 public:
  VegaIcon() = default;

  explicit VegaIcon(const std::string& json);

  // TODO: add Build() api to build a vega json string.
  // std::string Build() final;

 public:
  const std::string& icon_path() const { return icon_path_; }

 protected:
  // vega json to vega struct
  void Parse(const std::string& json) final;

 private:
  std::string icon_path_;
};

}  // namespace render
}  // namespace arctern
