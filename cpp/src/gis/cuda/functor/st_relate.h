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
#include "gis/cuda/common/gis_definitions.h"
#include "gis/cuda/tools/de9im_matrix.h"

namespace arctern {
namespace gis {
namespace cuda {

void ST_Relate(const GeometryVector& left_vec, const GeometryVector& right_vec,
               de9im::Matrix ref_matrix, bool* host_results);

}  // namespace cuda
}  // namespace gis
}  // namespace arctern
