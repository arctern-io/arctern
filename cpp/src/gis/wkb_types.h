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

namespace zilliz {
namespace gis {

enum class WkbTypes : uint32_t {
  kUnknown = 0, /* non-standard */

  kPoint = 1,              /* standard WKB */
  kLineString = 2,         /* standard WKB */
  kPolygon = 3,            /* standard WKB */
  kMultiPoint = 4,         /* standard WKB */
  kMultiLineString = 5,    /* standard WKB */
  kMultiPolygon = 6,       /* standard WKB */
  kGeometryCollection = 7, /* standard WKB */

  kCircularString = 8,     /* ISO SQL/MM Part 3 */
  kCompoundCurve = 9,      /* ISO SQL/MM Part 3 */
  kCurvePolygon = 10,      /* ISO SQL/MM Part 3 */
  kMultiCurve = 11,        /* ISO SQL/MM Part 3 */
  kMultiSurface = 12,      /* ISO SQL/MM Part 3 */
  kCurve = 13,             /* ISO SQL/MM Part 3 */
  kSurface = 14,           /* ISO SQL/MM Part 3 */
  kPolyhedralSurface = 15, /* ISO SQL/MM Part 3 */
  kTIN = 16,               /* ISO SQL/MM Part 3 */
  kTriangle = 17,          /* ISO SQL/MM Part 3 */

  kPointZ = 1001,              /* ISO SQL/MM Part 3 */
  kLineStringZ = 1002,         /* ISO SQL/MM Part 3 */
  kPolygonZ = 1003,            /* ISO SQL/MM Part 3 */
  kMultiPointZ = 1004,         /* ISO SQL/MM Part 3 */
  kMultiLineStringZ = 1005,    /* ISO SQL/MM Part 3 */
  kMultiPolygonZ = 1006,       /* ISO SQL/MM Part 3 */
  kGeometryCollectionZ = 1007, /* ISO SQL/MM Part 3 */
  kCircularStringZ = 1008,     /* ISO SQL/MM Part 3 */
  kCompoundCurveZ = 1009,      /* ISO SQL/MM Part 3 */
  kCurvePolygonZ = 1010,       /* ISO SQL/MM Part 3 */
  kMultiCurveZ = 1011,         /* ISO SQL/MM Part 3 */
  kMultiSurfaceZ = 1012,       /* ISO SQL/MM Part 3 */
  kCurveZ = 1013,              /* ISO SQL/MM Part 3 */
  kSurfaceZ = 1014,            /* ISO SQL/MM Part 3 */
  kPolyhedralSurfaceZ = 1015,  /* ISO SQL/MM Part 3 */
  kTINZ = 1016,                /* ISO SQL/MM Part 3 */
  kTriangleZ = 1017,           /* ISO SQL/MM Part 3 */

  kPointM = 2001,              /* ISO SQL/MM Part 3 */
  kLineStringM = 2002,         /* ISO SQL/MM Part 3 */
  kPolygonM = 2003,            /* ISO SQL/MM Part 3 */
  kMultiPointM = 2004,         /* ISO SQL/MM Part 3 */
  kMultiLineStringM = 2005,    /* ISO SQL/MM Part 3 */
  kMultiPolygonM = 2006,       /* ISO SQL/MM Part 3 */
  kGeometryCollectionM = 2007, /* ISO SQL/MM Part 3 */
  kCircularStringM = 2008,     /* ISO SQL/MM Part 3 */
  kCompoundCurveM = 2009,      /* ISO SQL/MM Part 3 */
  kCurvePolygonM = 2010,       /* ISO SQL/MM Part 3 */
  kMultiCurveM = 2011,         /* ISO SQL/MM Part 3 */
  kMultiSurfaceM = 2012,       /* ISO SQL/MM Part 3 */
  kCurveM = 2013,              /* ISO SQL/MM Part 3 */
  kSurfaceM = 2014,            /* ISO SQL/MM Part 3 */
  kPolyhedralSurfaceM = 2015,  /* ISO SQL/MM Part 3 */
  kTINM = 2016,                /* ISO SQL/MM Part 3 */
  kTriangleM = 2017,           /* ISO SQL/MM Part 3 */

  kPointZM = 3001,              /* ISO SQL/MM Part 3 */
  kLineStringZM = 3002,         /* ISO SQL/MM Part 3 */
  kPolygonZM = 3003,            /* ISO SQL/MM Part 3 */
  kMultiPointZM = 3004,         /* ISO SQL/MM Part 3 */
  kMultiLineStringZM = 3005,    /* ISO SQL/MM Part 3 */
  kMultiPolygonZM = 3006,       /* ISO SQL/MM Part 3 */
  kGeometryCollectionZM = 3007, /* ISO SQL/MM Part 3 */
  kCircularStringZM = 3008,     /* ISO SQL/MM Part 3 */
  kCompoundCurveZM = 3009,      /* ISO SQL/MM Part 3 */
  kCurvePolygonZM = 3010,       /* ISO SQL/MM Part 3 */
  kMultiCurveZM = 3011,         /* ISO SQL/MM Part 3 */
  kMultiSurfaceZM = 3012,       /* ISO SQL/MM Part 3 */
  kCurveZM = 3013,              /* ISO SQL/MM Part 3 */
  kSurfaceZM = 3014,            /* ISO SQL/MM Part 3 */
  kPolyhedralSurfaceZM = 3015,  /* ISO SQL/MM Part 3 */
  kTINZM = 3016,                /* ISO SQL/MM Part 3 */
  kTriangleZM = 3017,           /* ISO SQL/MM Part 3 */

  kMaxTypeNumber = kTriangleZM,
};

constexpr uint32_t kWkbSpaceTypeEncodeBase = 1000;

enum class WkbByteOrder : uint8_t { kBigEndian = 0, kLittleEndian = 1 };

enum class WkbCategory : uint32_t {
  kInvalid = 0,
  kPoint = 1,
  kLineString = 2,
  kPolygon = 3,
  kMultiPoint = 4,
  kMultiLineString = 5,
  kMultiPolygon = 6,
  kGeometryCollection = 7,
  kCircularString = 8,
  kCompoundCurve = 9,
  kCurvePolygon = 10,
  kMultiCurve = 11,
  kMultiSurface = 12,
  kCurve = 13,
  kSurface = 14,
  kPolyhedralSurface = 15,
  kTIN = 16,
  kTriangle = 17,
};

enum class WkbSpaceType : uint32_t {
  XY = 0,   // normal 2D
  XYZ = 1,  // XYZ
  XYM = 2,  // XYM
  XYZM = 3  // XYZM
};

}  // namespace gis
}  // namespace zilliz
