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
#include <thrust/pair.h>

#include <algorithm>
#include <limits>

#include "gis/cuda/common/common.h"
#include "gis/cuda/common/gis_definitions.h"

namespace arctern {
namespace gis {
namespace cuda {

constexpr double inf = std::numeric_limits<double>::max();
struct MinMax {
  DEVICE_RUNNABLE MinMax() : min(+inf), max(-inf) {}
  DEVICE_RUNNABLE void Update(double value) {
    min = value < min ? value : min;
    max = value > max ? value : max;
  }
  DEVICE_RUNNABLE bool is_trivial() const { return min == max; }
  DEVICE_RUNNABLE bool is_valid() const { return min <= max; }

 public:
  double min;
  double max;
};

class BoundingBox {
 public:
  DEVICE_RUNNABLE MinMax get_xs() const { return xs_; }
  DEVICE_RUNNABLE MinMax get_ys() const { return ys_; }
  DEVICE_RUNNABLE void Update(double2 value) {
    xs_.Update(value.x);
    ys_.Update(value.y);
  }

  DEVICE_RUNNABLE bool is_valid() {
    assert(xs_.is_valid() == ys_.is_valid());
    return xs_.is_valid();
  }

 private:
  MinMax xs_;
  MinMax ys_;
};

DEVICE_RUNNABLE BoundingBox CalcBoundingBox(WkbTag tag, ConstGpuContext::ConstIter& iter);

}  // namespace cuda
}  // namespace gis
}  // namespace arctern
