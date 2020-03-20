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

namespace arctern {

using ErrorCode = unsigned int;

constexpr ErrorCode ILLEGAL_VEGA_FORMAT = 5001;
constexpr ErrorCode NULL_RENDER_OUTPUT = 5002;
constexpr ErrorCode VALUE_TYPE_NOT_FOUND = 5003;
constexpr ErrorCode INVALID_VEGA_DATA = 5004;
constexpr ErrorCode UNKNOW_GEOMETYR_TYPE = 5005;
constexpr ErrorCode FAILED_COMPILE_SHADER = 5006;
constexpr ErrorCode FAILED_LINK_SHADER = 5007;
constexpr ErrorCode COLOR_STYLE_NOT_FOUND = 5008;
constexpr ErrorCode INVAILD_COLOR_FORMAT = 5009;
constexpr ErrorCode LABEL_NOT_FOUND = 5010;
constexpr ErrorCode FAILED_INIT_OSMESA = 5011;

constexpr ErrorCode ILLEGAL_WKT_FORMAT = 6001;
constexpr ErrorCode ILLEGAL_WKB_FORMAT = 6002;

}  // namespace arctern
