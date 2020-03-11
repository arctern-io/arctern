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

#include "gis/gdal/type_scan.h"

#include <ogr_api.h>
#include <ogrsf_frmts.h>

#include <map>
#include <set>
#include <utility>
#include <vector>

#include "gis/wkb_types.h"
#include "utils/check_status.h"
#include "utils/function_wrapper.h"

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
    ret->is_unique_type = true;
    ret->unique_type = {WkbTypes::kUnknown};
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

//  std::vector<int> mask_counts_mapping(num_scan_classes, 0);
//  std::vector<std::vector<bool>> masks_mapping(num_scan_classes);
  using Info = GeometryTypeMasks::Info;
  std::vector<Info> mapping(num_scan_classes);
  for (auto i = 0; i < num_scan_classes; i++) {
    mapping[i].masks.resize(len, false);
    mapping[i].scan_class_id = i;
  }

  auto wkt_geometries = std::static_pointer_cast<arrow::StringArray>(geometries_);
  std::vector<GeometryTypeMasks::ScanClassId> scan_class_ids(len);
  bool is_unique_type = true;
  int last_idx = -1;

  // fill type masks
  for (int i = 0; i < len; i++) {
    auto geo = [str = wkt_geometries->GetString(i)] {
      OGRGeometry* geo_;
      CHECK_GDAL(OGRGeometryFactory::createFromWkt(str.c_str(), nullptr, &geo_));
      return UniquePtrWithDeleter<OGRGeometry, OGRGeometryFactory::destroyGeometry>(geo_);
    }();
    auto type = OGR_G_GetGeometryType(geo.get());
    auto idx = type_to_idx[type];
    mapping[idx].masks[i] = true;
    mapping[idx].mask_counts++;
    scan_class_ids[i] = idx;

    if (last_idx != -1 && last_idx != idx) {
      is_unique_type = false;
    }
    last_idx = idx;
  }

  // organize return
  auto ret = std::make_shared<GeometryTypeMasks>();
  ret->is_unique_type = false;
  ret->scan_class_ids.resize(len);

  if (is_unique_type) {
    int scan_class_id = 0;
    if (mapping[scan_class_id].masks.front() == true) {
      ret->is_unique_type = true;
      ret->unique_type = {WkbTypes::kUnknown};
      return ret;
    } else {
      scan_class_id++;
      for (auto& grouped_type : types()) {
        if (mapping[scan_class_id].masks.front() == true) {
          ret->is_unique_type = true;
          ret->unique_type = grouped_type;
          return ret;
        }
      }
      assert(false/**/);
    }
  } else {
    int scan_class_id = 0;
    ret->scan_class_ids = std::move(scan_class_ids);
    GroupedWkbTypes unknown_type = {WkbTypes::kUnknown};
    ret->dict[unknown_type] = std::move(mapping[scan_class_id++]);

    for (auto& grouped_type : types()) {
      ret->dict[grouped_type] = std::move(mapping[scan_class_id]);
      scan_class_id++;
    }
  }
  return ret;
}

}  // namespace gdal
}  // namespace gis
}  // namespace arctern
