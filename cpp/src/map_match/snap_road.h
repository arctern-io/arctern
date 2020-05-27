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
#include <ogr_api.h>
#include <ogrsf_frmts.h>
#include <arrow/array.h>
#include <arrow/api.h>
#include <omp.h>
#include <vector>
#include <memory>

#include "render/utils/render_utils.h"

namespace arctern {
namespace snap {


std::vector<std::shared_ptr<arrow::Array>> snap_to_road(
    const std::vector<std::shared_ptr<arrow::Array>>& roads,
    const std::vector<std::shared_ptr<arrow::Array>>& gps_points,
    int32_t num_thread = 8);


}  // namespace snap
}  // namespace arctern
