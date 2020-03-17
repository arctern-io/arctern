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
#include <string>

#include "arrow/api.h"

namespace arctern {
namespace render {

std::shared_ptr<arrow::Array> transform_and_projection(const std::shared_ptr<arrow::Array>& geos,
                                                       const std::string& src_rs,
                                                       const std::string& dst_rs,
                                                       const std::string& bottom_right,
                                                       const std::string& top_left,
                                                       const int &height,
                                                       const int &width);

//std::shared_ptr<arrow::Array> coordinate_projection(
//    const std::shared_ptr<arrow::Array>& input_point, const std::string top_left,
//    const std::string bottom_right, const int height, const int width);

std::shared_ptr<arrow::Array> point_map(const std::shared_ptr<arrow::Array>& arr_x,
                                        const std::shared_ptr<arrow::Array>& arr_y,
                                        const std::string& conf);

std::shared_ptr<arrow::Array> point_map(const std::shared_ptr<arrow::Array>& points,
                                        const std::string& conf);

std::shared_ptr<arrow::Array> heat_map(const std::shared_ptr<arrow::Array>& arr_x,
                                       const std::shared_ptr<arrow::Array>& arr_y,
                                       const std::shared_ptr<arrow::Array>& arr_c,
                                       const std::string& conf);

std::shared_ptr<arrow::Array> heat_map(const std::shared_ptr<arrow::Array>& points,
                                       const std::shared_ptr<arrow::Array>& arr_c,
                                       const std::string& conf);

std::shared_ptr<arrow::Array> choropleth_map(
    const std::shared_ptr<arrow::Array>& arr_wkt,
    const std::shared_ptr<arrow::Array>& arr_count, const std::string& conf);

}  // namespace render
}  // namespace arctern
