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
    "ST_Point_UDF",
    "ST_Intersection_UDF",
    "ST_IsValid_UDF",
    "ST_Equals_UDF",
    "ST_Touches_UDF",
    "ST_Overlaps_UDF",
    "ST_Crosses_UDF",
    "ST_IsSimple_UDF",
    "ST_GeometryType_UDF",
    "ST_MakeValid_UDF",
    "ST_SimplifyPreserveTopology_UDF",
    "ST_PolygonFromEnvelope_UDF",
    "ST_Contains_UDF",
    "ST_Intersects_UDF",
    "ST_Within_UDF",
    "ST_Distance_UDF",
    "ST_Area_UDF",
    "ST_Centroid_UDF",
    "ST_Length_UDF",
    "ST_HausdorffDistance_UDF",
    "ST_ConvexHull_UDF",
    "ST_NPoints_UDF",
    "ST_Envelope_UDF",
    "ST_Buffer_UDF",
    "ST_Union_Aggr_UDF",
    "ST_Envelope_Aggr_UDF",
    "ST_Transform_UDF",
    "ST_GeomFromGeoJSON_UDF",
    "ST_PointFromText_UDF",
    "ST_PolygonFromText_UDF",
    "ST_LineStringFromText_UDF",
    "ST_GeomFromText_UDF",
    "ST_GeomFromWKT_UDF",
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
    from zilliz_gis import point_map
    curve_z = point_map(arr_x, arr_y)
    curve_z_copy = curve_z
    curve_z = curve_z.buffers()[1].to_pybytes()
    return curve_z_copy.buffers()[1].hex()

@pandas_udf("string", PandasUDFType.SCALAR)
def ST_PointFromText_UDF(geo):
    return geo

@pandas_udf("string", PandasUDFType.SCALAR)
def ST_PolygonFromText_UDF(geo):
    return geo

@pandas_udf("string", PandasUDFType.SCALAR)
def ST_LineStringFromText_UDF(geo):
    return geo

@pandas_udf("string", PandasUDFType.SCALAR)
def ST_GeomFromWKT_UDF(geo):
    return geo

@pandas_udf("string", PandasUDFType.SCALAR)
def ST_GeomFromText_UDF(geo):
    return geo

@pandas_udf("string", PandasUDFType.SCALAR)
def ST_Point_UDF(x, y):
    arr_x = pa.array(x, type='double')
    arr_y = pa.array(y, type='double')
    from zilliz_gis import ST_Point
    rs = ST_Point(arr_x, arr_y)
    return rs.to_pandas()

@pandas_udf("string", PandasUDFType.SCALAR)
def ST_GeomFromGeoJSON_UDF(json):
    geo = pa.array(json,type='string')
    from zilliz_gis import ST_GeomFromGeoJSON
    rs = ST_GeomFromGeoJSON(geo)
    return rs.to_pandas()

@pandas_udf("string", PandasUDFType.SCALAR)
def ST_Intersection_UDF(left, right):
    arr_left = pa.array(left, type='string')
    arr_right = pa.array(right, type='string')
    from zilliz_gis import ST_Intersection
    rs = ST_Intersection(arr_left, arr_right)
    return rs.to_pandas()

@pandas_udf("boolean", PandasUDFType.SCALAR)
def ST_IsValid_UDF(geos):
    arr_geos = pa.array(geos, type='string')
    from zilliz_gis import ST_IsValid
    rs = ST_IsValid(arr_geos)
    return rs.to_pandas()

@pandas_udf("boolean", PandasUDFType.SCALAR)
def ST_Equals_UDF(left, right):
    arr_left = pa.array(left, type='string')
    arr_right = pa.array(right, type='string')
    from zilliz_gis import ST_Equals
    rs = ST_Equals(arr_left, arr_right)
    return rs.to_pandas()

@pandas_udf("boolean", PandasUDFType.SCALAR)
def ST_Touches_UDF(left, right):
    arr_left = pa.array(left, type='string')
    arr_right = pa.array(right, type='string')
    from zilliz_gis import ST_Touches
    rs = ST_Touches(arr_left, arr_right)
    return rs.to_pandas()

@pandas_udf("boolean", PandasUDFType.SCALAR)
def ST_Overlaps_UDF(left, right):
    arr_left = pa.array(left, type='string')
    arr_right = pa.array(right, type='string')
    from zilliz_gis import ST_Overlaps
    rs = ST_Overlaps(arr_left, arr_right)
    return rs.to_pandas()

@pandas_udf("boolean", PandasUDFType.SCALAR)
def ST_Crosses_UDF(left, right):
    arr_left = pa.array(left, type='string')
    arr_right = pa.array(right, type='string')
    from zilliz_gis import ST_Crosses
    rs = ST_Crosses(arr_left, arr_right)
    return rs.to_pandas()

@pandas_udf("boolean", PandasUDFType.SCALAR)
def ST_IsSimple_UDF(geos):
    arr_geos = pa.array(geos, type='string')
    from zilliz_gis import ST_IsSimple
    rs = ST_IsSimple(arr_geos)
    return rs.to_pandas()

@pandas_udf("string", PandasUDFType.SCALAR)
def ST_GeometryType_UDF(geos):
    arr_geos = pa.array(geos, type='string')
    from zilliz_gis import ST_GeometryType
    rs = ST_GeometryType(arr_geos)
    return rs.to_pandas()

@pandas_udf("string", PandasUDFType.SCALAR)
def ST_MakeValid_UDF(geos):
    arr_geos = pa.array(geos, type='string')
    from zilliz_gis import ST_MakeValid
    rs = ST_MakeValid(arr_geos)
    return rs.to_pandas()

# TODO: ST_SimplifyPreserveTopology
@pandas_udf("string", PandasUDFType.SCALAR)
def ST_SimplifyPreserveTopology_UDF(geos, distance_tolerance):
    arr_geos = pa.array(geos, type='string')
    dis_tol = distance_tolerance[0]
    from zilliz_gis import ST_SimplifyPreserveTopology
    rs = ST_SimplifyPreserveTopology(arr_geos, dis_tol)
    return rs.to_pandas()

@pandas_udf("string", PandasUDFType.SCALAR)
def ST_PolygonFromEnvelope_UDF(min_x, min_y, max_x, max_y):
    arr_min_x = pa.array(min_x, type='double')
    arr_min_y = pa.array(min_y, type='double')
    arr_max_x = pa.array(max_x, type='double')
    arr_max_y = pa.array(max_y, type='double')
    from zilliz_gis import ST_PolygonFromEnvelope
    rs = ST_PolygonFromEnvelope(arr_min_x, arr_min_y, arr_max_x, arr_max_y)
    return rs.to_pandas()

@pandas_udf("boolean", PandasUDFType.SCALAR)
def ST_Contains_UDF(left, right):
    arr_left = pa.array(left, type='string')
    arr_right = pa.array(right, type='string')
    from zilliz_gis import ST_Contains
    rs = ST_Contains(arr_left, arr_right)
    return rs.to_pandas()

@pandas_udf("boolean", PandasUDFType.SCALAR)
def ST_Intersects_UDF(left, right):
    arr_left = pa.array(left, type='string')
    arr_right = pa.array(right, type='string')
    from zilliz_gis import ST_Intersects
    rs = ST_Intersects(arr_left, arr_right)
    return rs.to_pandas()

@pandas_udf("boolean", PandasUDFType.SCALAR)
def ST_Within_UDF(left, right):
    arr_left = pa.array(left, type='string')
    arr_right = pa.array(right, type='string')
    from zilliz_gis import ST_Within
    rs = ST_Within(arr_left, arr_right)
    return rs.to_pandas()

@pandas_udf("double", PandasUDFType.SCALAR)
def ST_Distance_UDF(left, right):
    arr_left = pa.array(left, type='string')
    arr_right = pa.array(right, type='string')
    from zilliz_gis import ST_Distance
    rs = ST_Distance(arr_left, arr_right)
    return rs.to_pandas()

@pandas_udf("double", PandasUDFType.SCALAR)
def ST_Area_UDF(geos):
    arr_geos = pa.array(geos, type='string')
    from zilliz_gis import ST_Area
    rs = ST_Area(arr_geos)
    return rs.to_pandas()

@pandas_udf("string", PandasUDFType.SCALAR)
def ST_Centroid_UDF(geos):
    arr_geos = pa.array(geos, type='string')
    from zilliz_gis import ST_Centroid
    rs = ST_Centroid(arr_geos)
    return rs.to_pandas()

@pandas_udf("double", PandasUDFType.SCALAR)
def ST_Length_UDF(geos):
    arr_geos = pa.array(geos, type='string')
    from zilliz_gis import ST_Length
    rs = ST_Length(arr_geos)
    return rs.to_pandas()

@pandas_udf("double", PandasUDFType.SCALAR)
def ST_HausdorffDistance_UDF(geo1, geo2):
    arr1 = pa.array(geo1, type='string')
    arr2 = pa.array(geo2, type='string')
    from zilliz_gis import ST_HausdorffDistance
    rs = ST_HausdorffDistance(arr1, arr2)
    return rs.to_pandas()

@pandas_udf("string", PandasUDFType.SCALAR)
def ST_ConvexHull_UDF(geos):
    arr_geos = pa.array(geos, type='string')
    from zilliz_gis import ST_ConvexHull
    rs = ST_ConvexHull(arr_geos)
    return rs.to_pandas()

@pandas_udf("int", PandasUDFType.SCALAR)
def ST_NPoints_UDF(geos):
    arr_geos = pa.array(geos, type='string')
    from zilliz_gis import ST_NPoints
    rs = ST_NPoints(arr_geos)
    return rs.to_pandas()

@pandas_udf("string", PandasUDFType.SCALAR)
def ST_Envelope_UDF(geos):
    arr_geos = pa.array(geos, type='string')
    from zilliz_gis import ST_Envelope
    rs = ST_Envelope(arr_geos)
    return rs.to_pandas()

# TODO: ST_Buffer, how to polymorphicly define the behaviour of spark udf
@pandas_udf("string", PandasUDFType.SCALAR)
def ST_Buffer_UDF(geos, dfDist):
    arr_geos = pa.array(geos, type='string')
    distance = dfDist[0]
    from zilliz_gis import ST_Buffer
    rs = ST_Buffer(arr_geos, distance)
    return rs.to_pandas()

@pandas_udf("string", PandasUDFType.GROUPED_AGG)
def ST_Union_Aggr_UDF(geos):
    arr_geos = pa.array(geos, type='string')
    from zilliz_gis import ST_Union_Aggr
    rs = ST_Union_Aggr(arr_geos)
    return str(rs[0])

@pandas_udf("string", PandasUDFType.GROUPED_AGG)
def ST_Envelope_Aggr_UDF(geos):
    arr_geos = pa.array(geos, type='string')
    from zilliz_gis import ST_Envelope_Aggr
    rs = ST_Envelope_Aggr(arr_geos)
    return str(rs[0])

@pandas_udf("string", PandasUDFType.SCALAR)
def ST_Transform_UDF(geos, src_rs, dst_rs):
    arr_geos = pa.array(geos, type='string')
    from zilliz_gis import ST_Transform
    src_rs1 = bytes(src_rs[0], encoding="utf8")
    dst_rs1 = bytes(dst_rs[0], encoding="utf8")
    rs = ST_Transform(arr_geos, src_rs1, dst_rs1)
    return rs.to_pandas()
