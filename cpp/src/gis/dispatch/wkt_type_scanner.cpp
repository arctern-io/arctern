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

#include "gis/dispatch/wkt_type_scanner.h"

#include <ogr_api.h>
#include <ogrsf_frmts.h>

#include <iostream>
#include <map>
#include <set>
#include <utility>
#include <vector>

#include "gis/wkb_types.h"
#include "utils/check_status.h"
#include "utils/function_wrapper.h"

namespace arctern {
namespace gis {
namespace dispatch {

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
  auto get_type_index = [](WkbTypes type) {
    auto index = int(type);
    if (index >= int(WkbTypes::kMaxTypeNumber)) {
      index = int(WkbTypes::kUnknown);
    }
    return index;
  };

  for (auto& grouped_type : types()) {
    for (auto& type : grouped_type) {
      type_to_idx[get_type_index(type)] = num_scan_classes;
    }
    num_scan_classes++;
  }

  //  std::vector<int> mask_counts_mapping(num_scan_classes, 0);
  //  std::vector<std::vector<bool>> masks_mapping(num_scan_classes);
  using Info = GeometryTypeMasks::Info;
  std::vector<Info> mapping(num_scan_classes);
  for (auto i = 0; i < num_scan_classes; i++) {
    mapping[i].mask.resize(len, false);
    mapping[i].encode_uid = i;
  }

  auto wkt_geometries = std::static_pointer_cast<arrow::StringArray>(geometries_);
  std::vector<GeometryTypeMasks::EncodeUid> encode_uids(len);
  bool is_unique_type = true;
  int last_idx = -1;

  // fill type mask
  for (int i = 0; i < len; i++) {
    using Holder = UniquePtrWithDeleter<OGRGeometry, OGRGeometryFactory::destroyGeometry>;
    auto type = [&] {
      if (wkt_geometries->IsNull(i)) {
        return WkbTypes::kUnknown;
      }
      auto str = wkt_geometries->GetString(i);
      if (str.size() == 0) {
        return WkbTypes::kUnknown;
      }
      OGRGeometry* geo_;
      auto error_code = OGRGeometryFactory::createFromWkt(str.c_str(), nullptr, &geo_);
      if (error_code != OGRERR_NONE) {
        return WkbTypes::kUnknown;
      }
      assert(geo_ != nullptr);
      Holder holder(geo_);
      auto type = (WkbTypes)wkbFlatten(holder->getGeometryType());
      // fix Point Empty
      if (type == WkbTypes::kPoint && holder->IsEmpty()) {
        return WkbTypes::kUnknown;
      }
      return type;
    }();

    auto idx = type_to_idx[get_type_index(type)];

    mapping[idx].mask[i] = true;
    mapping[idx].mask_count++;
    encode_uids[i] = idx;

    if (last_idx != -1 && last_idx != idx) {
      is_unique_type = false;
    }
    last_idx = idx;
  }

  // organize return
  auto ret = std::make_shared<GeometryTypeMasks>();
  ret->is_unique_type = false;

  if (is_unique_type) {
    int encode_uid = 0;
    if (mapping[encode_uid].mask.front() == true) {
      ret->is_unique_type = true;
      ret->unique_type = {WkbTypes::kUnknown};
      return ret;
    } else {
      encode_uid++;
      for (auto& grouped_type : types()) {
        if (mapping[encode_uid].mask.front() == true) {
          ret->is_unique_type = true;
          ret->unique_type = grouped_type;
          return ret;
        }
      }
      assert(false /**/);
    }
  } else {
    int encode_uid = 0;
    ret->encode_uids = std::move(encode_uids);
    GroupedWkbTypes unknown_type = {WkbTypes::kUnknown};
    ret->dict[unknown_type] = std::move(mapping[encode_uid++]);

    for (auto& grouped_type : types()) {
      ret->dict[grouped_type] = std::move(mapping[encode_uid]);
      encode_uid++;
    }
  }
  return ret;
}  // namespace gdal

}  // namespace dispatch
}  // namespace gis
}  // namespace arctern
