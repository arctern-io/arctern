# Copyright (C) 2019-2020 Zilliz. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

__all__ = [
    "ST_Point",
    "ST_Intersection",
    "ST_IsValid",
    "ST_PrecisionReduce",
    "ST_Equals",
    "ST_Touches",
    "ST_Overlaps",
    "ST_Crosses",
    "ST_IsSimple",
    "ST_GeometryType",
    "ST_MakeValid",
    "ST_SimplifyPreserveTopology",
    "ST_PolygonFromEnvelope",
    "ST_Contains",
    "ST_Intersects",
    "ST_Within",
    "ST_Distance",
    "ST_Area",
    "ST_Centroid",
    "ST_Length",
    "ST_HausdorffDistance",
    "ST_ConvexHull",
    "ST_NPoints",
    "ST_Envelope",
    "ST_Buffer",
    "ST_Union_Aggr",
    "ST_Envelope_Aggr",
    "ST_Transform",
    "ST_CurveToLine",
    "ST_GeomFromGeoJSON",
    "ST_PointFromText",
    "ST_PolygonFromText",
    "ST_LineStringFromText",
    "ST_GeomFromText",
    "ST_GeomFromWKT",
    "ST_AsText",
    "TransformAndProjection",
]

import arctern
from pyspark.sql.functions import pandas_udf, PandasUDFType

@pandas_udf("string", PandasUDFType.SCALAR)
def TransformAndProjection(geos, src_rs, dst_rs, bottom_right, top_left, height, width):
    return arctern.transform_and_projection(geos, src_rs[0], dst_rs[0], bottom_right[0], top_left[0], height[0], width[0])

@pandas_udf("string", PandasUDFType.SCALAR)
def ST_PointFromText(geo):
    return geo

@pandas_udf("string", PandasUDFType.SCALAR)
def ST_PolygonFromText(geo):
    return geo

@pandas_udf("string", PandasUDFType.SCALAR)
def ST_LineStringFromText(geo):
    return geo

@pandas_udf("string", PandasUDFType.SCALAR)
def ST_GeomFromWKT(geo):
    return geo

@pandas_udf("string", PandasUDFType.SCALAR)
def ST_GeomFromText(geo):
    return geo

@pandas_udf("string", PandasUDFType.SCALAR)
def ST_AsText(geo):
    return geo

@pandas_udf("string", PandasUDFType.SCALAR)
def ST_Point(x, y):
    return arctern.ST_Point(x, y)

@pandas_udf("string", PandasUDFType.SCALAR)
def ST_GeomFromGeoJSON(json):
    return arctern.ST_GeomFromGeoJSON(json)

@pandas_udf("string", PandasUDFType.SCALAR)
def ST_Intersection(left, right):
    return arctern.ST_Intersection(left, right)

@pandas_udf("boolean", PandasUDFType.SCALAR)
def ST_IsValid(geos):
    return arctern.ST_IsValid(geos)

@pandas_udf("string", PandasUDFType.SCALAR)
def ST_PrecisionReduce(geos, precision):
    return arctern.ST_PrecisionReduce(geos, precision[0])


@pandas_udf("boolean", PandasUDFType.SCALAR)
def ST_Equals(left, right):
    return arctern.ST_Equals(left, right)

@pandas_udf("boolean", PandasUDFType.SCALAR)
def ST_Touches(left, right):
    return arctern.ST_Touches(left, right)

@pandas_udf("boolean", PandasUDFType.SCALAR)
def ST_Overlaps(left, right):
    return arctern.ST_Overlaps(left, right)

@pandas_udf("boolean", PandasUDFType.SCALAR)
def ST_Crosses(left, right):
    return arctern.ST_Crosses(left, right)

@pandas_udf("boolean", PandasUDFType.SCALAR)
def ST_IsSimple(geos):
    return arctern.ST_IsSimple(geos)

@pandas_udf("string", PandasUDFType.SCALAR)
def ST_GeometryType(geos):
    return arctern.ST_GeometryType(geos)

@pandas_udf("string", PandasUDFType.SCALAR)
def ST_MakeValid(geos):
    return arctern.ST_MakeValid(geos)

# TODO: ST_SimplifyPreserveTopology
@pandas_udf("string", PandasUDFType.SCALAR)
def ST_SimplifyPreserveTopology(geos, distance_tolerance):
    return arctern.ST_SimplifyPreserveTopology(geos, distance_tolerance[0])

@pandas_udf("string", PandasUDFType.SCALAR)
def ST_PolygonFromEnvelope(min_x, min_y, max_x, max_y):
    return arctern.ST_PolygonFromEnvelope(min_x, min_y, max_x, max_y)

@pandas_udf("boolean", PandasUDFType.SCALAR)
def ST_Contains(left, right):
    return arctern.ST_Contains(left, right)

@pandas_udf("boolean", PandasUDFType.SCALAR)
def ST_Intersects(left, right):
    return arctern.ST_Intersects(left, right)

@pandas_udf("boolean", PandasUDFType.SCALAR)
def ST_Within(left, right):
    return arctern.ST_Within(left, right)

@pandas_udf("double", PandasUDFType.SCALAR)
def ST_Distance(left, right):
    return arctern.ST_Distance(left, right)

@pandas_udf("double", PandasUDFType.SCALAR)
def ST_Area(geos):
    return arctern.ST_Area(geos)

@pandas_udf("string", PandasUDFType.SCALAR)
def ST_Centroid(geos):
    return arctern.ST_Centroid(geos)

@pandas_udf("double", PandasUDFType.SCALAR)
def ST_Length(geos):
    return arctern.ST_Length(geos)

@pandas_udf("double", PandasUDFType.SCALAR)
def ST_HausdorffDistance(geo1, geo2):
    return arctern.ST_HausdorffDistance(geo1, geo2)

@pandas_udf("string", PandasUDFType.SCALAR)
def ST_ConvexHull(geos):
    return arctern.ST_ConvexHull(geos)

@pandas_udf("int", PandasUDFType.SCALAR)
def ST_NPoints(geos):
    return arctern.ST_NPoints(geos)

@pandas_udf("string", PandasUDFType.SCALAR)
def ST_Envelope(geos):
    return arctern.ST_Envelope(geos)

# TODO: ST_Buffer, how to polymorphicly define the behaviour of spark udf
@pandas_udf("string", PandasUDFType.SCALAR)
def ST_Buffer(geos, dfDist):
    return arctern.ST_Buffer(geos, dfDist[0])

@pandas_udf("string", PandasUDFType.GROUPED_AGG)
def ST_Union_Aggr(geos):
    return arctern.ST_Union_Aggr(geos)

@pandas_udf("string", PandasUDFType.GROUPED_AGG)
def ST_Envelope_Aggr(geos):
    return arctern.ST_Envelope_Aggr(geos)

@pandas_udf("string", PandasUDFType.SCALAR)
def ST_Transform(geos, src_rs, dst_rs):
    return arctern.ST_Transform(geos, src_rs[0], dst_rs[0])

@pandas_udf("string", PandasUDFType.SCALAR)
def ST_CurveToLine(geos):
    return arctern.ST_CurveToLine(geos)
