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
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include "common/version.h"
#include "gis/api.h"
#include "gis/cuda/common/gis_definitions.h"
#include "gis/cuda/conversion/conversions.h"
#include "gis/test_common/transforms.h"
#include "utils/check_status.h"

namespace arctern {
namespace gis {
namespace cuda {

namespace GeometryVectorFactory {
inline GeometryVector CreateFromWkts(const std::vector<std::string>& wkt_vec) {
  auto input = StrsToWkb(wkt_vec);
  return ArrowWkbToGeometryVector(input);
}

inline GeometryVector CreateFromWkbs(const std::vector<std::vector<char>>& wkb_vec) {
  arrow::BinaryBuilder builder;
  for (const auto& wkb : wkb_vec) {
    auto st = builder.Append(wkb.data(), wkb.size());
    assert(st.ok());
  }
  std::shared_ptr<arrow::Array> arrow_wkb;
  auto st = builder.Finish(&arrow_wkb);
  assert(st.ok());
  auto result = ArrowWkbToGeometryVector(arrow_wkb);
  return result;
}

}  // namespace GeometryVectorFactory

}  // namespace cuda
}  // namespace gis
}  // namespace arctern
