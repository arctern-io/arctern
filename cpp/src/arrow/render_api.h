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

std::shared_ptr<arrow::Array> WktToWkb(const std::shared_ptr<arrow::Array>& arr_wkt);
std::shared_ptr<arrow::Array> WkbToWkt(const std::shared_ptr<arrow::Array>& arr_wkt);

std::shared_ptr<arrow::Array> projection(const std::shared_ptr<arrow::Array>& geos,
                                         const std::string& bottom_right,
                                         const std::string& top_left, const int& height,
                                         const int& width);

std::shared_ptr<arrow::Array> transform_and_projection(
    const std::shared_ptr<arrow::Array>& geos, const std::string& src_rs,
    const std::string& dst_rs, const std::string& bottom_right,
    const std::string& top_left, const int& height, const int& width);

std::shared_ptr<arrow::Array> point_map(const std::shared_ptr<arrow::Array>& arr_x,
                                        const std::shared_ptr<arrow::Array>& arr_y,
                                        const std::string& conf);

std::shared_ptr<arrow::Array> point_map(const std::shared_ptr<arrow::Array>& points,
                                        const std::string& conf);

// two args api: point_map(wkt, conf)
std::shared_ptr<arrow::Array> weighted_point_map(
    const std::shared_ptr<arrow::Array>& arr1, const std::string& conf);

// three args api: point_map(x, y, conf), point_map(wkt, c, conf), point_map(wkt, s, conf)
std::shared_ptr<arrow::Array> weighted_point_map(
    const std::shared_ptr<arrow::Array>& arr1, const std::shared_ptr<arrow::Array>& arr2,
    const std::string& conf);

// four args api: point_map(x, y, c, conf), point_map(x, y, s, conf), point_map(wkt, c, s,
// conf)
std::shared_ptr<arrow::Array> weighted_point_map(
    const std::shared_ptr<arrow::Array>& arr1, const std::shared_ptr<arrow::Array>& arr2,
    const std::shared_ptr<arrow::Array>& arr3, const std::string& conf);

// five args api: point_map(x, y, c, s, conf)
std::shared_ptr<arrow::Array> weighted_point_map(
    const std::shared_ptr<arrow::Array>& arr_x,
    const std::shared_ptr<arrow::Array>& arr_y,
    const std::shared_ptr<arrow::Array>& arr_c,
    const std::shared_ptr<arrow::Array>& arr_s, const std::string& conf);

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

std::shared_ptr<arrow::Array> icon_viz(const std::shared_ptr<arrow::Array>& arr_x,
                                       const std::shared_ptr<arrow::Array>& arr_y,
                                       const std::string& conf);

std::shared_ptr<arrow::Array> icon_viz(const std::shared_ptr<arrow::Array>& points,
                                       const std::string& conf);

}  // namespace render
}  // namespace arctern
