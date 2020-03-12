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
#include <ostream>
#include <string>

namespace arctern {
namespace gis {

enum class LogLevel : int { DEBUG = -1, INFO = 0, WARNNING = 1, ERROR = 2, FATAL = 3 };

#define GIS_LOG_INTERNAL(level) ::arctern::GIS::GisLog(__FILE__, __LINE__, level)
#define GIS_LOG(level) GIS_LOG_INTERNAL(::arctern::GIS::LogLevel::##level)

}  // namespace gis
}  // namespace arctern
