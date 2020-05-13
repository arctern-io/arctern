// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#include "gis/dispatch/wkb_type_scanner.h"

#include <algorithm>
namespace arctern {
namespace gis {
namespace dispatch {

WkbTypeScanner::WkbTypeScanner(const std::shared_ptr<arrow::BinaryArray>& geometries)
    : geometries_(geometries) {}

auto WkbTypeScanner::Scan() const -> std::shared_ptr<GeometryTypeMasks> {
  if (types().empty()) {
    auto ret = std::make_shared<GeometryTypeMasks>();
    ret->is_unique_type = true;
    ret->unique_type = {WkbTypes::kUnknown};
    return ret;
  }

  std::vector<GroupedWkbTypes> decoding_map;
  decoding_map.emplace_back(GroupedWkbTypes{WkbTypes::kUnknown});
  constexpr int MaxTypeVolume = 3000;
  int type_volume = 0;
  for (const auto& type_set : this->types()) {
    for (auto type : type_set) {
      assert((int)type >= 0);
      assert((int)type < MaxTypeVolume);
      type_volume = std::max(type_volume, (int)type + 1);
    }
  }

  std::vector<int> encoding_map(type_volume, 0);
  int scan_volume = 0;
  {
    int encode_uid = 1;
    for (const auto& type_set : this->types()) {
      for (auto type : type_set) {
        encoding_map[(int)type] = encode_uid;
      }
      ++encode_uid;
      decoding_map.push_back(type_set);
      assert(decoding_map.size() == encode_uid);
    }
    assert(encode_uid == 1 + types().size());
  }

  // TODO: use gpu when vector is ready
  std::vector<GeometryTypeMasks::Info> infos(types().size() + 1);
  for (auto& info : infos) {
    info.mask_count = 0;
    info.mask.resize(geometries_->length());
  }
  auto to_encode_uid = [&encoding_map, type_volume](WkbTypes type) -> int {
    if ((int)type >= 0 && (int)type < type_volume) {
      return encoding_map[(int)type];
    } else {
      return 0;
    }
  };

  for (int i = 0; i < geometries_->length(); ++i) {
    WkbTypes type;
    if (geometries_->IsNull(i)) {
      type = WkbTypes::kUnknown;
    } else {
      auto payload = geometries_->GetView(i);
      assert(payload.size() >= sizeof(WkbByteOrder) + sizeof(WkbTypes));
      memcpy(&type, payload.data() + sizeof(WkbByteOrder), sizeof(WkbTypes));
    }
    auto encode_uid = to_encode_uid(type);
    auto& info = infos[encode_uid];
    info.mask_count++;
    info.mask[i] = true;
  }

  for (int encode_uid = 0; encode_uid < infos.size(); ++encode_uid) {
    auto& info = infos[encode_uid];
    if (info.mask_count == geometries_->length()) {
      // unique type
      auto ret = std::make_shared<GeometryTypeMasks>();
      ret->is_unique_type = true;
      ret->unique_type = decoding_map[encode_uid];
      return ret;
    }
  }
  auto ret = std::make_shared<GeometryTypeMasks>();
  ret->is_unique_type = false;
  for (int encode_uid = 0; encode_uid < infos.size(); ++encode_uid) {
    ret->dict[decoding_map[encode_uid]] = infos[encode_uid];
  }
  return ret;
}

}  // namespace dispatch
}  // namespace gis
}  // namespace arctern
