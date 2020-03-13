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

#include "gis/cuda/test_common/test_common.h"

#include "gis/cuda/wkb/wkb_transforms.h"

namespace arctern {
namespace gis {
namespace cuda {
// only for testing
// create Geometry from WktArray
namespace GeometryVectorFactory {

GeometryVector CreateFromWkts(const std::vector<std::string>& wkt_vec) {
  auto input = WktsToArrowWkb(wkt_vec);
  return ArrowWkbToGeometryVector(input);
}

GeometryVector CreateFromWkbs(const std::vector<std::vector<char>>& wkb_vec) {
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
