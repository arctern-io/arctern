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

#ifndef RENDER_H
#define RENDER_H

#include <memory>
#include <string>

#include "arrow/api.h"

namespace arctern {
namespace render {

std::shared_ptr<arrow::Array> WktToWkb(const std::shared_ptr<arrow::Array>& arr_wkt);

std::shared_ptr<arrow::Array> WkbToWkt(const std::shared_ptr<arrow::Array>& arr_wkb);

const std::vector<std::shared_ptr<arrow::Array>> projection(
    const std::vector<std::shared_ptr<arrow::Array>>& geos,
    const std::string& bottom_right, const std::string& top_left, const int& height,
    const int& width);

const std::vector<std::shared_ptr<arrow::Array>> transform_and_projection(
    const std::vector<std::shared_ptr<arrow::Array>>& geos, const std::string& src_rs,
    const std::string& dst_rs, const std::string& bottom_right,
    const std::string& top_left, const int& height, const int& width);

std::shared_ptr<arrow::Array> point_map(
    const std::vector<std::shared_ptr<arrow::Array>>& points_vector,
    const std::string& conf);

std::shared_ptr<arrow::Array> heat_map(
    const std::vector<std::shared_ptr<arrow::Array>>& points_vector,
    const std::vector<std::shared_ptr<arrow::Array>>& weights_vector,
    const std::string& conf);

std::shared_ptr<arrow::Array> weighted_point_map(
    const std::vector<std::shared_ptr<arrow::Array>>& points_vector,
    const std::string& conf);

std::shared_ptr<arrow::Array> weighted_point_map(
    const std::vector<std::shared_ptr<arrow::Array>>& points_vector,
    const std::vector<std::shared_ptr<arrow::Array>>& weights_vector,
    const std::string& conf);

std::shared_ptr<arrow::Array> weighted_point_map(
    const std::vector<std::shared_ptr<arrow::Array>>& points_vector,
    const std::vector<std::shared_ptr<arrow::Array>>& color_weights_vector,
    const std::vector<std::shared_ptr<arrow::Array>>& size_weights_vector,
    const std::string& conf);

std::shared_ptr<arrow::Array> choropleth_map(
    const std::vector<std::shared_ptr<arrow::Array>>& region_boundaries,
    const std::vector<std::shared_ptr<arrow::Array>>& weights,
    const std::string& vega);

std::shared_ptr<arrow::Array> icon_viz(
    const std::vector<std::shared_ptr<arrow::Array>>& points_vector,
    const std::string& vega);

std::shared_ptr<arrow::Array> fishnet_map(
    const std::vector<std::shared_ptr<arrow::Array>>& points_vector,
    const std::vector<std::shared_ptr<arrow::Array>>& weights_vector,
    const std::string& vega);

}  // namespace render
}  // namespace arctern

#endif
