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

#include "gis/spatial_join/st_indexed_within.h"
namespace arctern {
namespace gis {
namespace spatial_join {
std::vector<Int32ArrayPtr> ST_IndexedWithin(const std::vector<WkbArrayPtr>& points,
                                            const std::vector<WkbArrayPtr>& polygons) {
  // this is a fake implementation
  // assume that points is Point(0 0) Point(1000 1000) Point(10 10)
  // assume that polygon is Polygon(9 10, 11 12, 11 8, 9 10)
  //                        Polygon(-1 0, 1 2, 1 -2, -1 0)
  assert(points.size() == 1);
  assert(points.front()->length() == 3);
  assert(polygons.size() == 1);
  assert(polygons.front()->length() == 2);
  arrow::Int32Builder builder;
  builder.Append(1);
  builder.Append(-1);
  builder.Append(0);
  Int32ArrayPtr res;
  builder.Finish(&res);
  return {res};
}

}  // namespace spatial_join
}  // namespace gis
}  // namespace arctern
