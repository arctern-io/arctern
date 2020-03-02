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

namespace zilliz {
namespace gis {
namespace gdal {

class NPointsVisitor : public IOGRConstGeometryVisitor {
 public:
  ~NPointsVisitor() = default;

  void visit(const OGRPoint*);
  void visit(const OGRLineString*);
  void visit(const OGRLinearRing*);
  void visit(const OGRPolygon*);
  void visit(const OGRMultiPoint*);
  void visit(const OGRMultiLineString*);
  void visit(const OGRMultiPolygon*);
  void visit(const OGRGeometryCollection*);
  void visit(const OGRCircularString*);
  void visit(const OGRCompoundCurve*);
  void visit(const OGRCurvePolygon*);
  void visit(const OGRMultiCurve*);
  void visit(const OGRMultiSurface*);
  void visit(const OGRTriangle*);
  void visit(const OGRPolyhedralSurface*);
  void visit(const OGRTriangulatedSurface*);

  const int64_t npoints() const { return npoints_; }

 private:
  int64_t npoints_ = 0;
};

}  // namespace gdal
}  // namespace gis
}  // namespace zilliz
