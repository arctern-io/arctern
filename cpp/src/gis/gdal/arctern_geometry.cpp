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

#include "gis/gdal/arctern_geometry.h"

namespace zilliz {
namespace gis {
namespace gdal {

void NPointsVisitor::visit(const OGRPoint*) {}
void NPointsVisitor::visit(const OGRLineString*) {}
void NPointsVisitor::visit(const OGRLinearRing*) {}
void NPointsVisitor::visit(const OGRPolygon*) {}
void NPointsVisitor::visit(const OGRMultiPoint*) {}
void NPointsVisitor::visit(const OGRMultiLineString*) {}
void NPointsVisitor::visit(const OGRMultiPolygon*) {}
void NPointsVisitor::visit(const OGRGeometryCollection*) {}
void NPointsVisitor::visit(const OGRCircularString*) {}
void NPointsVisitor::visit(const OGRCompoundCurve*) {}
void NPointsVisitor::visit(const OGRCurvePolygon*) {}
void NPointsVisitor::visit(const OGRMultiCurve*) {}
void NPointsVisitor::visit(const OGRMultiSurface*) {}
void NPointsVisitor::visit(const OGRTriangle*) {}
void NPointsVisitor::visit(const OGRPolyhedralSurface*) {}
void NPointsVisitor::visit(const OGRTriangulatedSurface*) {}

}  // namespace gdal
}  // namespace gis
}  // namespace zilliz
