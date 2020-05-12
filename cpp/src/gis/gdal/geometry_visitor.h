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

namespace arctern {
namespace gis {
namespace gdal {

class AreaVisitor : public OGRDefaultConstGeometryVisitor {
 public:
  ~AreaVisitor() = default;

  void visit(const OGRPoint*) override {}
  void visit(const OGRLineString*) override {}
  void visit(const OGRLinearRing*) override {}
  void visit(const OGRPolygon* geo) override { area_ += geo->get_Area(); }
  void visit(const OGRMultiPoint*) override {}
  void visit(const OGRMultiLineString*) override {}
  void visit(const OGRMultiPolygon* geo) override { area_ += geo->get_Area(); }
  // void visit(const OGRGeometryCollection*) override;
  void visit(const OGRCircularString*) override {}
  void visit(const OGRCompoundCurve*) override {}
  void visit(const OGRCurvePolygon* geo) override { area_ += geo->get_Area(); }
  void visit(const OGRMultiCurve*) override {}
  void visit(const OGRMultiSurface* geo) override { area_ += geo->get_Area(); }
  void visit(const OGRTriangle* geo) override { area_ += geo->get_Area(); }
  // void visit(const OGRPolyhedralSurface*) override;
  // void visit(const OGRTriangulatedSurface*) override;

  const double area() const { return area_; }
  void reset() { area_ = 0; }

 private:
  double area_ = 0;
};

class LengthVisitor : public OGRDefaultConstGeometryVisitor {
 public:
  ~LengthVisitor() = default;

  void visit(const OGRPoint*) override {}
  void visit(const OGRLineString* geo) override { length_ += geo->get_Length(); }
  void visit(const OGRLinearRing* geo) override { length_ += geo->get_Length(); }
  void visit(const OGRPolygon*) override {}
  void visit(const OGRMultiPoint*) override {}
  void visit(const OGRMultiLineString* geo) override { length_ += geo->get_Length(); }
  // void visit(const OGRMultiPolygon* ) override;
  // void visit(const OGRGeometryCollection*) override;
  void visit(const OGRCircularString* geo) override { length_ += geo->get_Length(); }
  // void visit(const OGRCompoundCurve*) override;
  // void visit(const OGRCurvePolygon* ) override ;
  // void visit(const OGRMultiCurve*) override;
  // void visit(const OGRMultiSurface*) override;
  // void visit(const OGRTriangle*) override;
  // void visit(const OGRPolyhedralSurface*) override;
  // void visit(const OGRTriangulatedSurface*) override;

  const double length() const { return length_; }
  void reset() { length_ = 0; }

 private:
  double length_ = 0;
};

class HasCircularVisitor : public OGRDefaultConstGeometryVisitor {
 public:
  ~HasCircularVisitor() = default;

  void visit(const OGRPoint*) override {}
  void visit(const OGRLineString*) override {}
  void visit(const OGRLinearRing*) override {}
  void visit(const OGRPolygon*) override {}
  void visit(const OGRMultiPoint*) override {}
  void visit(const OGRMultiLineString*) override {}
  void visit(const OGRMultiPolygon*) override {}
  // void visit(const OGRGeometryCollection*) override;
  void visit(const OGRCircularString* geo) override;
  // void visit(const OGRCompoundCurve*) override;
  // void visit(const OGRCurvePolygon* ) override ;
  // void visit(const OGRMultiCurve*) override;
  // void visit(const OGRMultiSurface*) override;
  // void visit(const OGRTriangle*) override;
  // void visit(const OGRPolyhedralSurface*) override;
  // void visit(const OGRTriangulatedSurface*) override;

  const bool has_circular() const { return has_circular_; }
  void reset() { has_circular_ = false; }

 private:
  bool has_circular_ = false;
};

class HasCurveVisitor : public OGRDefaultConstGeometryVisitor {
 public:
  ~HasCurveVisitor() = default;

  void visit(const OGRPoint*) override {}
  void visit(const OGRLineString*) override {}
  void visit(const OGRLinearRing*) override {}
  void visit(const OGRPolygon*) override {}
  void visit(const OGRMultiPoint*) override {}
  void visit(const OGRMultiLineString*) override {}
  void visit(const OGRMultiPolygon*) override {}
  // void visit(const OGRGeometryCollection*) override;
  void visit(const OGRCircularString* geo) override;
  void visit(const OGRCompoundCurve* geo) override;
  void visit(const OGRCurvePolygon* geo) override;
  void visit(const OGRMultiCurve* geo) override;
  // void visit(const OGRMultiSurface*) override;
  // void visit(const OGRTriangle*) override;
  // void visit(const OGRPolyhedralSurface*) override;
  // void visit(const OGRTriangulatedSurface*) override;

  const bool has_curve() const { return has_curve_; }
  void reset() { has_curve_ = false; }

 private:
  bool has_curve_ = false;
};

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
 public:
  explicit PrecisionReduceVisitor(int32_t precision) : precision_(precision) {}
  ~PrecisionReduceVisitor() = default;

  double coordinate_precision_reduce(double coordinate);
  void visit(OGRPoint*) override;

 private:
  int32_t precision_ = 0;
};

}  // namespace gdal
}  // namespace gis
}  // namespace arctern
