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

#pragma once
#include <ogr_api.h>
#include <ogrsf_frmts.h>

#include <memory>
#include <string>
#include <vector>

#include "gis/cuda/mock/arrow/api.h"
#include "gis/gdal/format_conversion.h"
#include "gis/gdal/gis_functions.h"
namespace arctern {
namespace gis {
inline std::vector<char> SingleStrToWkb(const std::string& geo_wkt) {
  OGRGeometry* geo = nullptr;
  {
    auto err_code = OGRGeometryFactory::createFromWkt(geo_wkt.c_str(), nullptr, &geo);
    assert(err_code == OGRERR_NONE);
  }
  auto sz = geo->WkbSize();
  std::vector<char> result(sz);
  {
    auto err_code = geo->exportToWkb(OGRwkbByteOrder::wkbNDR, (uint8_t*)result.data());
    assert(err_code == OGRERR_NONE);
  }
  OGRGeometryFactory::destroyGeometry(geo);
  return result;
}

inline std::shared_ptr<arrow::StringArray> StrsToWkt(
    const std::vector<std::string>& wkt_vec) {
  arrow::StringBuilder builder;
  for (const auto& wkt : wkt_vec) {
    auto st = builder.Append(wkt.data(), wkt.size());
    assert(st.ok());
  }
  std::shared_ptr<arrow::StringArray> result;
  auto st = builder.Finish(&result);
  assert(st.ok());
  return result;
}

inline std::shared_ptr<arrow::Array> StrsToWkb(const std::vector<std::string>& wkt_vec) {
  return gdal::WktToWkb(StrsToWkt(wkt_vec));
}

inline std::vector<char> HexStringToWkb(const std::string& str) {
  std::vector<char> vec;
  assert(str.size() % 2 == 0);
  for (size_t index = 0; index < str.size(); index += 2) {
    auto byte_str = str.substr(index, 2);
    char* tmp;
    auto data = strtoul(byte_str.c_str(), &tmp, 16);
    assert(*tmp == 0);
    vec.push_back((char)data);
  }
  return vec;
}

}  // namespace gis
}  // namespace arctern
