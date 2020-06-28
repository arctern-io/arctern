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

#include <cmath>
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

/*
 *
 *  For 3D affine transformations, the 12 parameters represent the augmented matrix:

        [x']   | a  b  c xoff | [x]
        [y'] = | d  e  f yoff | [y]
        [z']   | g  h  i zoff | [z]
        [1 ]   | 0  0  0   1  | [1]

    the equations are:

        x' = a * x + b * y + c * z + xoff
        y' = d * x + e * y + f * z + yoff
        z' = g * x + h * y + i * z + zoff
 */
struct AffineParams {
  double a_ = 1.;
  double b_ = 0.;
  double c_ = 0.;
  double d_ = 0.;
  double e_ = 1.;
  double f_ = 0.;
  double g_ = 0.;
  double h_ = 1.;
  double i_ = 0.;
  double xoff_ = 0.;
  double yoff_ = 0.;
  double zoff_ = 0.;
};

class AffineVisitor : public OGRDefaultGeometryVisitor {
 public:
  AffineVisitor() = default;
  explicit AffineVisitor(const AffineParams& param) : param_(param) {}
  AffineVisitor(double a, double b, double d, double e, double xoff, double yoff) {
    param_.a_ = a;
    param_.b_ = b;
    param_.d_ = d;
    param_.e_ = e;
    param_.xoff_ = xoff;
    param_.yoff_ = yoff;
  }
  ~AffineVisitor() = default;

  void visit(OGRPoint*) override;

 protected:
  AffineParams param_;
};

class TranslateVisitor : public AffineVisitor {
 public:
  TranslateVisitor(double xoff, double yoff) {
    param_.xoff_ = xoff;
    param_.yoff_ = yoff;
  }

  ~TranslateVisitor() = default;
};

class RotateVisitor : public AffineVisitor {
 public:
  RotateVisitor(double rotation_angle, double origin_x, double origin_y) {
    auto cosp = std::cos(rotation_angle);
    auto sinp = std::sin(rotation_angle);

    param_.a_ = cosp;
    param_.b_ = -sinp;
    param_.d_ = sinp;
    param_.e_ = cosp;
    param_.h_ = cosp;
    param_.i_ = cosp;

    param_.xoff_ = origin_x - origin_x * cosp + origin_y * sinp;
    param_.yoff_ = origin_y - origin_x * sinp - origin_y * cosp;
  }

  ~RotateVisitor() = default;
};

class ScaleVisitor : public AffineVisitor {
 public:
  ScaleVisitor(double factor_x, double factor_y, double origin_x, double origin_y) {
    param_.a_ = factor_x;
    param_.e_ = factor_y;
    param_.xoff_ = origin_x - origin_x * factor_x;
    param_.yoff_ = origin_y - origin_y * factor_y;
  }

  ~ScaleVisitor() = default;
};

class SkewVisitor : public AffineVisitor {
 public:
  SkewVisitor(double xs, double ys, double origin_x, double origin_y) {
    auto tanx = std::tan(xs);
    auto tany = std::tan(ys);
    param_.b_ = tanx;
    param_.d_ = tany;
    param_.xoff_ = -origin_y * tanx;
    param_.yoff_ = -origin_x * tany;
  }

  ~SkewVisitor() = default;
};

}  // namespace gdal
}  // namespace gis
}  // namespace arctern
