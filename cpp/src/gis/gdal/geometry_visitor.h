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

#pragma once

#include <ogr_geometry.h>
#include <string>

namespace zilliz {
namespace gis {
namespace gdal {

class NPointsVisitor : public OGRDefaultConstGeometryVisitor {
 public:
  ~NPointsVisitor() = default;
  void visit(const OGRPoint*) override;

  const int64_t npoints() const { return npoints_; }
  void reset() { npoints_ = 0; }

 private:
  int64_t npoints_ = 0;
};

class PrecisionReduceVisitor : public OGRDefaultGeometryVisitor {
 public :
  PrecisionReduceVisitor(int32_t precision) : precision_(precision){}
  ~PrecisionReduceVisitor() = default;
  
  double coordinate_precision_reduce(double coordinate);
  void visit(OGRPoint*) override;
  // void geometry_precision_reduce(OGRGeometry* geo); 
  
 private:
  int32_t precision_ = 0;
};

}  // namespace gdal
}  // namespace gis
}  // namespace zilliz
