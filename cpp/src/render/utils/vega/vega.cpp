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

#include "render/utils/vega/vega.h"

namespace arctern {
namespace render {

bool Vega::JsonLabelCheck(rapidjson::Value& value, const std::string& label) {
  if (!value.HasMember(label.c_str())) {
    is_valid_ = false;
    std::string err_msg = "Cannot find label [" + label + "] !";
    throw std::runtime_error(err_msg);
  }
  return true;
}

bool Vega::JsonSizeCheck(rapidjson::Value& value, const std::string& label, size_t size) {
  if (value.Size() != size) {
    is_valid_ = false;
    std::string err_msg = "Member [" + label + "].size should be " +
                          std::to_string(size) + ", but get " +
                          std::to_string(value.Size());
    throw std::runtime_error(err_msg);
  }
  return true;
}

bool Vega::JsonTypeCheck(rapidjson::Value& value, rapidjson::Type type) {
  if (type == rapidjson::Type::kNumberType) {
    if (!value.IsNumber()) {
      is_valid_ = false;
      std::string err_msg = "not number type";
      throw std::runtime_error(err_msg);
    }
  } else if (type == rapidjson::Type::kArrayType) {
    if (!value.IsArray()) {
      is_valid_ = false;
      std::string err_msg = "not array type";
      throw std::runtime_error(err_msg);
    }
  } else if (type == rapidjson::Type::kStringType) {
    if (!value.IsString()) {
      is_valid_ = false;
      std::string err_msg = "not string type";
      throw std::runtime_error(err_msg);
    }
  }
  return true;
}

bool Vega::JsonNullCheck(rapidjson::Value& value) {
  if (value.IsNull()) {
    is_valid_ = false;
    std::string err_msg = "null json value";
    throw std::runtime_error(err_msg);
  }
  return true;
}

}  // namespace render
}  // namespace arctern
