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

#include <stdio.h>

#include <string>

#define SOURCE_CODE_INFO                                                             \
  std::string("[") + std::string(__FILE__) + ":" + std::to_string(__LINE__) + "][" + \
      std::string(__FUNCTION__) + "]"

#define ARCTERN_THROW_ERROR(err_code, err_msg)                            \
  do {                                                                    \
    std::string total_msg = SOURCE_CODE_INFO;                             \
    std::string code_str = std::to_string(err_code);                      \
    total_msg += std::string("error code: ") + code_str;                  \
    total_msg += std::string(",  reason: ") + err_msg + std::string("."); \
    throw std::runtime_error(total_msg);                                  \
  } while (false);
