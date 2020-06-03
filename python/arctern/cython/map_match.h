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

#ifndef MAP_MATCH_H
#define MAP_MATCH_H

#include <memory>
#include <string>

#include "arrow/api.h"

namespace arctern{
namespace map_match{

std::vector<std::shared_ptr<arrow::Array>> nearest_location_on_road(
    const std::vector<std::shared_ptr<arrow::Array>>& roads,
    const std::vector<std::shared_ptr<arrow::Array>>& gps_points);

std::vector<std::shared_ptr<arrow::Array>> nearest_road(
    const std::vector<std::shared_ptr<arrow::Array>>& roads,
    const std::vector<std::shared_ptr<arrow::Array>>& gps_points);

std::vector<std::shared_ptr<arrow::Array>> near_road(
    const std::vector<std::shared_ptr<arrow::Array>>& roads,
    const std::vector<std::shared_ptr<arrow::Array>>& gps_points,
    const double distance);

}  // namespace map_match
}  // namespace arctern

#endif