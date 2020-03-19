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

#include <arrow/api.h>

#include <string>

#define CHECK_GDAL(action)                                                     \
  do {                                                                         \
    auto check = action;                                                       \
    if (!!check) {                                                             \
      std::string err_msg = "gdal error code = " + std::to_string((int)check); \
      throw std::runtime_error(err_msg);                                       \
    }                                                                          \
  } while (false)

#define CHECK_ARROW(action)                                      \
  do {                                                           \
    arrow::Status status = action;                               \
    if (!status.ok()) {                                          \
      std::string err_msg = "arrow error: " + status.ToString(); \
      throw std::runtime_error(err_msg);                         \
    }                                                            \
  } while (false)
