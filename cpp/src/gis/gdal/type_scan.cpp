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

#include <map>
#include <set>
#include <utility>
#include <vector>

#include "gis/gdal/type_scan.h"
#include "gis/wkb_types.h"
#include "utils/check_status.h"

#include <ogr_api.h>
#include <ogrsf_frmts.h>

namespace arctern {
namespace gis {
namespace gdal {

TypeScannerForWkt::TypeScannerForWkt(const std::shared_ptr<arrow::Array>& geometries)
    : geometries_(geometries) {}

std::shared_ptr<GeometryTypeMasks> TypeScannerForWkt::Scan() {
  auto len = geometries_->length();

  if (types().empty()) {
    // organize return
    auto ret = std::make_shared<GeometryTypeMasks>();
    ret->is_unique_group = true;
    ret->unique_group = {WkbTypes::kUnknown};
    return ret;
  }

  // we redirect WkbTypes::kUnknown to idx=0
  std::vector<int> type_to_idx(int(WkbTypes::kMaxTypeNumber), 0);
  int num_scan_classes = 1;

  for (auto& grouped_type : types()) {
    for (auto& type : grouped_type) {
      type_to_idx[int(type)] = num_scan_classes;
    }
    num_scan_classes++;
  }

  std::vector<int> mask_counts(num_scan_classes, 0);
  std::vector<std::vector<bool>> type_masks(num_scan_classes);
  for (auto i = 0; i < num_scan_classes; i++) {
    type_masks[i].resize(len, false);
  }

  auto wkt_geometries = std::static_pointer_cast<arrow::StringArray>(geometries_);
  bool is_unique_type = true;
  int last_idx = -1;

  // fill type masks
  OGRGeometry* geo;
  for (int i = 0; i < len; i++) {
    CHECK_GDAL(OGRGeometryFactory::createFromWkt(wkt_geometries->GetString(i).c_str(),
                                                 nullptr, &geo));
    auto type = OGR_G_GetGeometryType((void*)geo);
    OGRGeometryFactory::destroyGeometry(geo);
    auto idx = type_to_idx[type];
    type_masks[idx][i] = true;
    mask_counts[idx]++;
    if (last_idx != -1 && last_idx != idx) {
      is_unique_type = false;
    }
    last_idx = idx;
  }

  // organize return
  auto ret = std::make_shared<GeometryTypeMasks>();
  ret->is_unique_group = false;

  if (is_unique_type) {
    num_scan_classes = 0;
    if (type_masks[num_scan_classes].front() == true) {
      ret->is_unique_group = true;
      ret->unique_group = {WkbTypes::kUnknown};
      ret->group_mask_counts[ret->unique_group] = len;
      return ret;
    }
    num_scan_classes++;
    for (auto& grouped_type : types()) {
      if (type_masks[num_scan_classes].front() == true) {
        ret->is_unique_group = true;
        ret->unique_group = grouped_type;
        ret->group_mask_counts[grouped_type] = len;
        return ret;
      }
    }

  } else {
    num_scan_classes = 0;
    GroupedWkbTypes unknown_type = {WkbTypes::kUnknown};
    ret->type_masks[unknown_type] = std::move(type_masks[num_scan_classes++]);

    for (auto& grouped_type : types()) {
      ret->type_masks[grouped_type] = std::move(type_masks[num_scan_classes]);
      ret->group_mask_counts[grouped_type] = mask_counts[num_scan_classes];
      num_scan_classes++;
    }
  }
  return ret;
}

}  // namespace gdal
}  // namespace gis
}  // namespace arctern
