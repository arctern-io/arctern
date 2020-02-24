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

#include "wkb_transforms.h"

#include <ogr_api.h>
#include <ogrsf_frmts.h>

#include <cassert>
#include <iostream>
#include <string>
#include <vector>
namespace zilliz {
namespace gis {
namespace cuda {

std::vector<char> Wkt2Wkb(const std::string& geo_wkt) {
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

}  // namespace cuda
}  // namespace gis
}  // namespace zilliz
