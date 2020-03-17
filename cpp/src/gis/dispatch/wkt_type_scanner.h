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
#include <vector>

#include "arrow/api.h"
#include "arrow/array.h"
#include "gis/dispatch/type_scanner.h"

namespace arctern {
namespace gis {
namespace dispatch {

class TypeScannerForWkt : public GeometryTypeScanner {
 public:
  explicit TypeScannerForWkt(const std::shared_ptr<arrow::Array>& geometries);

  std::shared_ptr<GeometryTypeMasks> Scan() final;

 private:
  const std::shared_ptr<arrow::Array> geometries_;
};

// split into [false_array, true_array]
std::array<std::shared_ptr<arrow::Array>, 2> WktArraySplit(
    const std::shared_ptr<arrow::Array>& geometries, const std::vector<bool>& mask);

// merge [false_array, true_array]
std::shared_ptr<arrow::Array> WktArrayMerge(
    const std::array<std::shared_ptr<arrow::Array>, 2>& inputs,
    const std::vector<bool>& mask);

// merge [false_array, true_array]
std::shared_ptr<arrow::Array> DoubleArrayMerge(
    const std::array<std::shared_ptr<arrow::Array>, 2>& inputs,
    const std::vector<bool>& mask);

}  // namespace gdal
}  // namespace gis
}  // namespace arctern
