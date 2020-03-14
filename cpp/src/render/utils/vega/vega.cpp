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
        // TODO: add log here
        std::cout << "Cannot find label [" << label << "] !" << std::endl;
        is_valid_ = false;
        return false;
    }
    return true;
}

bool Vega::JsonSizeCheck(rapidjson::Value& value, const std::string& label, size_t size) {
    if (value.Size() != size) {
        // TODO: add log here
        std::cout << "Member [" << label << "].size should be " << size << ", but get "
                  << value.Size() << std::endl;
        is_valid_ = false;
        return false;
    }
    return true;
}

bool Vega::JsonTypeCheck(rapidjson::Value& value, rapidjson::Type type) {
    if (type == rapidjson::Type::kNumberType) {
        if (!value.IsNumber()) {
            // TODO: add log here
            std::cout << "not number type" << std::endl;
            is_valid_ = false;
            return false;
        }
    } else if (type == rapidjson::Type::kArrayType) {
        if (!value.IsArray()) {
            // TODO: add log here
            std::cout << "not array type" << std::endl;
            is_valid_ = false;
            return false;
        }
    } else if (type == rapidjson::Type::kStringType) {
        if (!value.IsString()) {
            // TODO: add log here
            std::cout << "not string type" << std::endl;
            is_valid_ = false;
            return false;
        }
    }
    return true;
}

bool Vega::JsonNullCheck(rapidjson::Value& value) {
    if (value.IsNull()) {
        // TODO: add log here
        std::cout << "null json value" << std::endl;
        is_valid_ = false;
        return false;
    }
    return true;
}

}  // namespace render
}  // namespace arctern
