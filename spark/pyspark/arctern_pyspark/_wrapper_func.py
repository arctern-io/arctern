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
    "my_plot" # or point_map
]

import pyarrow as pa
from pyspark.sql.functions import pandas_udf, PandasUDFType

def toArrow(parameter):
    return  pa.array(parameter)

@pandas_udf("string", PandasUDFType.GROUPED_AGG)
def my_plot(x, y):
    arr_x = pa.array(x, type='uint32')
    arr_y = pa.array(y, type='uint32')
    from arctern_gis import point_map
    curve_z = point_map(arr_x, arr_y)
    curve_z_copy = curve_z
    curve_z = curve_z.buffers()[1].to_pybytes()
    return curve_z_copy.buffers()[1].hex()

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
    arr_x = pa.array(x, type='double')
    arr_y = pa.array(y, type='double')
    from arctern_gis import ST_Point
    rs = ST_Point(arr_x, arr_y)
    return rs.to_pandas()

@pandas_udf("string", PandasUDFType.SCALAR)
def ST_GeomFromGeoJSON(json):
    geo = pa.array(json,type='string')
    from arctern_gis import ST_GeomFromGeoJSON
    rs = ST_GeomFromGeoJSON(geo)
    return rs.to_pandas()

@pandas_udf("string", PandasUDFType.SCALAR)
def ST_Intersection(left, right):
    arr_left = pa.array(left, type='string')
    arr_right = pa.array(right, type='string')
    from arctern_gis import ST_Intersection
    rs = ST_Intersection(arr_left, arr_right)
    return rs.to_pandas()

@pandas_udf("boolean", PandasUDFType.SCALAR)
def ST_IsValid(geos):
    arr_geos = pa.array(geos, type='string')
    from arctern_gis import ST_IsValid
    rs = ST_IsValid(arr_geos)
    return rs.to_pandas()

@pandas_udf("string", PandasUDFType.SCALAR)
def ST_PrecisionReduce(geos, precision):
    arr_geos = pa.array(geos, type='string')
    from arctern_gis import ST_PrecisionReduce
    rs = ST_PrecisionReduce(arr_geos, precision)
    return rs.to_pandas()

@pandas_udf("boolean", PandasUDFType.SCALAR)
def ST_Equals(left, right):
    arr_left = pa.array(left, type='string')
    arr_right = pa.array(right, type='string')
    from arctern_gis import ST_Equals
    rs = ST_Equals(arr_left, arr_right)
    return rs.to_pandas()

@pandas_udf("boolean", PandasUDFType.SCALAR)
def ST_Touches(left, right):
    arr_left = pa.array(left, type='string')
    arr_right = pa.array(right, type='string')
    from arctern_gis import ST_Touches
    rs = ST_Touches(arr_left, arr_right)
    return rs.to_pandas()

@pandas_udf("boolean", PandasUDFType.SCALAR)
def ST_Overlaps(left, right):
    arr_left = pa.array(left, type='string')
    arr_right = pa.array(right, type='string')
    from arctern_gis import ST_Overlaps
    rs = ST_Overlaps(arr_left, arr_right)
    return rs.to_pandas()

@pandas_udf("boolean", PandasUDFType.SCALAR)
def ST_Crosses(left, right):
    arr_left = pa.array(left, type='string')
    arr_right = pa.array(right, type='string')
    from arctern_gis import ST_Crosses
    rs = ST_Crosses(arr_left, arr_right)
    return rs.to_pandas()

@pandas_udf("boolean", PandasUDFType.SCALAR)
def ST_IsSimple(geos):
    arr_geos = pa.array(geos, type='string')
    from arctern_gis import ST_IsSimple
    rs = ST_IsSimple(arr_geos)
    return rs.to_pandas()

@pandas_udf("string", PandasUDFType.SCALAR)
def ST_GeometryType(geos):
    arr_geos = pa.array(geos, type='string')
    from arctern_gis import ST_GeometryType
    rs = ST_GeometryType(arr_geos)
    return rs.to_pandas()

@pandas_udf("string", PandasUDFType.SCALAR)
def ST_MakeValid(geos):
    arr_geos = pa.array(geos, type='string')
    from arctern_gis import ST_MakeValid
    rs = ST_MakeValid(arr_geos)
    return rs.to_pandas()

# TODO: ST_SimplifyPreserveTopology
@pandas_udf("string", PandasUDFType.SCALAR)
def ST_SimplifyPreserveTopology(geos, distance_tolerance):
    arr_geos = pa.array(geos, type='string')
    dis_tol = distance_tolerance[0]
    from arctern_gis import ST_SimplifyPreserveTopology
    rs = ST_SimplifyPreserveTopology(arr_geos, dis_tol)
    return rs.to_pandas()

@pandas_udf("string", PandasUDFType.SCALAR)
def ST_PolygonFromEnvelope(min_x, min_y, max_x, max_y):
    arr_min_x = pa.array(min_x, type='double')
    arr_min_y = pa.array(min_y, type='double')
    arr_max_x = pa.array(max_x, type='double')
    arr_max_y = pa.array(max_y, type='double')
    from arctern_gis import ST_PolygonFromEnvelope
    rs = ST_PolygonFromEnvelope(arr_min_x, arr_min_y, arr_max_x, arr_max_y)
    return rs.to_pandas()

@pandas_udf("boolean", PandasUDFType.SCALAR)
def ST_Contains(left, right):
    arr_left = pa.array(left, type='string')
    arr_right = pa.array(right, type='string')
    from arctern_gis import ST_Contains
    rs = ST_Contains(arr_left, arr_right)
    return rs.to_pandas()

@pandas_udf("boolean", PandasUDFType.SCALAR)
def ST_Intersects(left, right):
    arr_left = pa.array(left, type='string')
    arr_right = pa.array(right, type='string')
    from arctern_gis import ST_Intersects
    rs = ST_Intersects(arr_left, arr_right)
    return rs.to_pandas()

@pandas_udf("boolean", PandasUDFType.SCALAR)
def ST_Within(left, right):
    arr_left = pa.array(left, type='string')
    arr_right = pa.array(right, type='string')
    from arctern_gis import ST_Within
    rs = ST_Within(arr_left, arr_right)
    return rs.to_pandas()

@pandas_udf("double", PandasUDFType.SCALAR)
def ST_Distance(left, right):
    arr_left = pa.array(left, type='string')
    arr_right = pa.array(right, type='string')
    from arctern_gis import ST_Distance
    rs = ST_Distance(arr_left, arr_right)
    return rs.to_pandas()

@pandas_udf("double", PandasUDFType.SCALAR)
def ST_Area(geos):
    arr_geos = pa.array(geos, type='string')
    from arctern_gis import ST_Area
    rs = ST_Area(arr_geos)
    return rs.to_pandas()

@pandas_udf("string", PandasUDFType.SCALAR)
def ST_Centroid(geos):
    arr_geos = pa.array(geos, type='string')
    from arctern_gis import ST_Centroid
    rs = ST_Centroid(arr_geos)
    return rs.to_pandas()

@pandas_udf("double", PandasUDFType.SCALAR)
def ST_Length(geos):
    arr_geos = pa.array(geos, type='string')
    from arctern_gis import ST_Length
    rs = ST_Length(arr_geos)
    return rs.to_pandas()

@pandas_udf("double", PandasUDFType.SCALAR)
def ST_HausdorffDistance(geo1, geo2):
    arr1 = pa.array(geo1, type='string')
    arr2 = pa.array(geo2, type='string')
    from arctern_gis import ST_HausdorffDistance
    rs = ST_HausdorffDistance(arr1, arr2)
    return rs.to_pandas()

@pandas_udf("string", PandasUDFType.SCALAR)
def ST_ConvexHull(geos):
    arr_geos = pa.array(geos, type='string')
    from arctern_gis import ST_ConvexHull
    rs = ST_ConvexHull(arr_geos)
    return rs.to_pandas()

@pandas_udf("int", PandasUDFType.SCALAR)
def ST_NPoints(geos):
    arr_geos = pa.array(geos, type='string')
    from arctern_gis import ST_NPoints
    rs = ST_NPoints(arr_geos)
    return rs.to_pandas()

@pandas_udf("string", PandasUDFType.SCALAR)
def ST_Envelope(geos):
    arr_geos = pa.array(geos, type='string')
    from arctern_gis import ST_Envelope
    rs = ST_Envelope(arr_geos)
    return rs.to_pandas()

# TODO: ST_Buffer, how to polymorphicly define the behaviour of spark udf
@pandas_udf("string", PandasUDFType.SCALAR)
def ST_Buffer(geos, dfDist):
    arr_geos = pa.array(geos, type='string')
    distance = dfDist[0]
    from arctern_gis import ST_Buffer
    rs = ST_Buffer(arr_geos, distance)
    return rs.to_pandas()

@pandas_udf("string", PandasUDFType.GROUPED_AGG)
def ST_Union_Aggr(geos):
    arr_geos = pa.array(geos, type='string')
    from arctern_gis import ST_Union_Aggr
    rs = ST_Union_Aggr(arr_geos)
    return str(rs[0])

@pandas_udf("string", PandasUDFType.GROUPED_AGG)
def ST_Envelope_Aggr(geos):
    arr_geos = pa.array(geos, type='string')
    from arctern_gis import ST_Envelope_Aggr
    rs = ST_Envelope_Aggr(arr_geos)
    return str(rs[0])

@pandas_udf("string", PandasUDFType.SCALAR)
def ST_Transform(geos, src_rs, dst_rs):
    arr_geos = pa.array(geos, type='string')
    from arctern_gis import ST_Transform
    src_rs1 = bytes(src_rs[0], encoding="utf8")
    dst_rs1 = bytes(dst_rs[0], encoding="utf8")
    rs = ST_Transform(arr_geos, src_rs1, dst_rs1)
    return rs.to_pandas()

@pandas_udf("string", PandasUDFType.SCALAR)
def ST_CurveToLine(geos):
    arr_geos = pa.array(geos, type='string')
    from arctern_gis import ST_CurveToLine
    rs = ST_CurveToLine(arr_geos)
    return rs.to_pandas()
