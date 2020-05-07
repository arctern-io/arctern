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
#include <memory>
#include <string>
#include <vector>

#include "arrow/render_api.h"
#include "utils/check_status.h"

namespace arctern {
namespace render {

std::vector<OGRGeometry*> GeometryExtraction(
    const std::vector<std::shared_ptr<arrow::Array>>& arrs);

std::vector<std::string> WkbExtraction(
    const std::vector<std::shared_ptr<arrow::Array>>& arrs);

std::vector<std::shared_ptr<arrow::Array>> GeometryExport(
    const std::vector<OGRGeometry*>& geos, int arrays_size);

template <typename T>
std::vector<T> WeightExtraction(const std::vector<std::shared_ptr<arrow::Array>>& arrs) {
  int total_size = 0;

  for (const auto& arr : arrs) {
    total_size += arr->length();
  }

  std::vector<T> res(total_size);

  int offset = 0;
  for (const auto& arr : arrs) {
    // TODO: if to use numeric array
    //    auto numeric_arr = std::static_pointer_cast<arrow::NumericArray<T>>(arr);
    //    (T*)numeric_arr->data()->GetValues(1);
    auto ptr = (T*)arr->data()->GetValues<T>(1);
    std::memcpy(res.data() + offset, ptr, arr->length() * sizeof(T));
    offset += arr->length();
  }

  return res;
}

void pointXY_from_wkt_with_transform(const std::string& wkt, double& x, double& y,
                                     void* poCT);

void pointXY_from_wkt(const std::string& wkt, double& x, double& y);

}  // namespace render
}  // namespace arctern
