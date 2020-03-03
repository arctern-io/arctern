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

#include "gis/gdal/geometry_visitor.h"

namespace zilliz {
namespace gis {
namespace gdal {

void NPointsVisitor::visit(const OGRPoint* geo) {
  if (geo->IsEmpty()) return;
  npoints_++;
}

void NPointsVisitor::visit(const OGRLineString* geo) {
  if (geo->IsEmpty()) return;
  npoints_ += geo->getNumPoints();
}

void NPointsVisitor::visit(const OGRLinearRing* geo) {
  if (geo->IsEmpty()) return;
  npoints_ += geo->getNumPoints();
}

void NPointsVisitor::visit(const OGRPolygon* geo) {
  if (geo->IsEmpty()) return;
  for (auto it = geo->begin(); it != geo->end(); ++it) {
    (*it)->accept(this);
  }
}

void NPointsVisitor::visit(const OGRMultiPoint* geo) {
  if (geo->IsEmpty()) return;
  for (auto it = geo->begin(); it != geo->end(); ++it) {
    (*it)->accept(this);
  }
}

void NPointsVisitor::visit(const OGRMultiLineString* geo) {
  if (geo->IsEmpty()) return;
  for (auto it = geo->begin(); it != geo->end(); ++it) {
    (*it)->accept(this);
  }
}

void NPointsVisitor::visit(const OGRMultiPolygon* geo) {
  if (geo->IsEmpty()) return;
  for (auto it = geo->begin(); it != geo->end(); ++it) {
    (*it)->accept(this);
  }
}

void NPointsVisitor::visit(const OGRGeometryCollection* geo) {
  if (geo->IsEmpty()) return;
  for (auto it = geo->begin(); it != geo->end(); ++it) {
    (*it)->accept(this);
  }
}

void NPointsVisitor::visit(const OGRCircularString* geo) {
  if (geo->IsEmpty()) return;
  npoints_ += geo->getNumPoints();
}
void NPointsVisitor::visit(const OGRCompoundCurve* geo) {
  if (geo->IsEmpty()) return;
  npoints_ += geo->getNumPoints();
}
void NPointsVisitor::visit(const OGRCurvePolygon* geo) {
  if (geo->IsEmpty()) return;
  for (auto it = geo->begin(); it != geo->end(); ++it) {
    (*it)->accept(this);
  }
}
void NPointsVisitor::visit(const OGRMultiCurve* geo) {
  if (geo->IsEmpty()) return;
  for (auto it = geo->begin(); it != geo->end(); ++it) {
    (*it)->accept(this);
  }
}
void NPointsVisitor::visit(const OGRMultiSurface* geo) {
  if (geo->IsEmpty()) return;
  for (auto it = geo->begin(); it != geo->end(); ++it) {
    (*it)->accept(this);
  }
}
void NPointsVisitor::visit(const OGRTriangle* geo) {
  if (geo->IsEmpty()) return;
  for (auto it = geo->begin(); it != geo->end(); ++it) {
    (*it)->accept(this);
  }
}
void NPointsVisitor::visit(const OGRPolyhedralSurface* geo) {
  if (geo->IsEmpty()) return;
  for (auto it = geo->begin(); it != geo->end(); ++it) {
    (*it)->accept(this);
  }
}
void NPointsVisitor::visit(const OGRTriangulatedSurface* geo) {
  if (geo->IsEmpty()) return;
  for (auto it = geo->begin(); it != geo->end(); ++it) {
    (*it)->accept(this);
  }
}

}  // namespace gdal
}  // namespace gis
}  // namespace zilliz
