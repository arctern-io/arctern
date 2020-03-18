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
    "ST_GeomFromText",
    "point_map",
    "point_map_wkt",
    "heat_map",
    "heat_map_wkt",
    "choropleth_map",
    "coordinate_projection",
]



import pyarrow as pa
from . import arctern_core_

def ST_Point(x, y):
    arr_x = pa.array(x, type='double')
    arr_y = pa.array(y, type='double')
    rs = arctern_core_.ST_Point(arr_x, arr_y)
    return rs.to_pandas()

def ST_GeomFromGeoJSON(json):
    geo = pa.array(json, type='string')
    rs = arctern_core_.ST_GeomFromGeoJSON(geo)
    return rs.to_pandas()

def ST_GeomFromText(text):
    geo = pa.array(text, type='string')
    rs = arctern_core_.ST_GeomFromText(geo)
    return rs.to_pandas()

def ST_Intersection(left, right):
    arr_left = pa.array(left, type='string')
    arr_right = pa.array(right, type='string')
    rs = arctern_core_.ST_Intersection(arr_left, arr_right)
    return rs.to_pandas()

def ST_IsValid(geos):
    arr_geos = pa.array(geos, type='string')
    rs = arctern_core_.ST_IsValid(arr_geos)
    return rs.to_pandas()

def ST_PrecisionReduce(geos, precision):
    arr_geos = pa.array(geos, type='string')
    rs = arctern_core_.ST_PrecisionReduce(arr_geos, precision)
    return rs.to_pandas()

def ST_Equals(left, right):
    arr_left = pa.array(left, type='string')
    arr_right = pa.array(right, type='string')
    rs = arctern_core_.ST_Equals(arr_left, arr_right)
    return rs.to_pandas()

def ST_Touches(left, right):
    arr_left = pa.array(left, type='string')
    arr_right = pa.array(right, type='string')
    rs = arctern_core_.ST_Touches(arr_left, arr_right)
    return rs.to_pandas()

def ST_Overlaps(left, right):
    arr_left = pa.array(left, type='string')
    arr_right = pa.array(right, type='string')
    rs = arctern_core_.ST_Overlaps(arr_left, arr_right)
    return rs.to_pandas()

def ST_Crosses(left, right):
    arr_left = pa.array(left, type='string')
    arr_right = pa.array(right, type='string')
    rs = arctern_core_.ST_Crosses(arr_left, arr_right)
    return rs.to_pandas()

def ST_IsSimple(geos):
    arr_geos = pa.array(geos, type='string')
    rs = arctern_core_.ST_IsSimple(arr_geos)
    return rs.to_pandas()

def ST_GeometryType(geos):
    arr_geos = pa.array(geos, type='string')
    rs = arctern_core_.ST_GeometryType(arr_geos)
    return rs.to_pandas()

def ST_MakeValid(geos):
    arr_geos = pa.array(geos, type='string')
    rs = arctern_core_.ST_MakeValid(arr_geos)
    return rs.to_pandas()

def ST_SimplifyPreserveTopology(geos, distance_tolerance):
    arr_geos = pa.array(geos, type='string')
    rs = arctern_core_.ST_SimplifyPreserveTopology(arr_geos, distance_tolerance)
    return rs.to_pandas()

def ST_PolygonFromEnvelope(min_x, min_y, max_x, max_y):
    arr_min_x = pa.array(min_x, type='double')
    arr_min_y = pa.array(min_y, type='double')
    arr_max_x = pa.array(max_x, type='double')
    arr_max_y = pa.array(max_y, type='double')
    rs = arctern_core_.ST_PolygonFromEnvelope(arr_min_x, arr_min_y, arr_max_x, arr_max_y)
    return rs.to_pandas()

def ST_Contains(left, right):
    arr_left = pa.array(left, type='string')
    arr_right = pa.array(right, type='string')
    rs = arctern_core_.ST_Contains(arr_left, arr_right)
    return rs.to_pandas()

def ST_Intersects(left, right):
    arr_left = pa.array(left, type='string')
    arr_right = pa.array(right, type='string')
    rs = arctern_core_.ST_Intersects(arr_left, arr_right)
    return rs.to_pandas()

def ST_Within(left, right):
    arr_left = pa.array(left, type='string')
    arr_right = pa.array(right, type='string')
    rs = arctern_core_.ST_Within(arr_left, arr_right)
    return rs.to_pandas()

def ST_Distance(left, right):
    arr_left = pa.array(left, type='string')
    arr_right = pa.array(right, type='string')
    rs = arctern_core_.ST_Distance(arr_left, arr_right)
    return rs.to_pandas()

def ST_Area(geos):
    arr_geos = pa.array(geos, type='string')
    rs = arctern_core_.ST_Area(arr_geos)
    return rs.to_pandas()

def ST_Centroid(geos):
    arr_geos = pa.array(geos, type='string')
    rs = arctern_core_.ST_Centroid(arr_geos)
    return rs.to_pandas()

def ST_Length(geos):
    arr_geos = pa.array(geos, type='string')
    rs = arctern_core_.ST_Length(arr_geos)
    return rs.to_pandas()

def ST_HausdorffDistance(geo1, geo2):
    arr1 = pa.array(geo1, type='string')
    arr2 = pa.array(geo2, type='string')
    rs = arctern_core_.ST_HausdorffDistance(arr1, arr2)
    return rs.to_pandas()

def ST_ConvexHull(geos):
    arr_geos = pa.array(geos, type='string')
    rs = arctern_core_.ST_ConvexHull(arr_geos)
    return rs.to_pandas()

def ST_NPoints(geos):
    arr_geos = pa.array(geos, type='string')
    rs = arctern_core_.ST_NPoints(arr_geos)
    return rs.to_pandas()

def ST_Envelope(geos):
    arr_geos = pa.array(geos, type='string')
    rs = arctern_core_.ST_Envelope(arr_geos)
    return rs.to_pandas()

def ST_Buffer(geos, distance):
    arr_geos = pa.array(geos, type='string')
    rs = arctern_core_.ST_Buffer(arr_geos, distance)
    return rs.to_pandas()

def ST_Union_Aggr(geos):
    arr_geos = pa.array(geos, type='string')
    rs = arctern_core_.ST_Union_Aggr(arr_geos)
    return str(rs[0])

def ST_Envelope_Aggr(geos):
    arr_geos = pa.array(geos, type='string')
    rs = arctern_core_.ST_Envelope_Aggr(arr_geos)
    return str(rs[0])

def ST_Transform(geos, src, dst):
    arr_geos = pa.array(geos, type='string')
    src = bytes(src, encoding="utf8")
    dst = bytes(dst, encoding="utf8")

    rs = arctern_core_.ST_Transform(arr_geos, src, dst)
    return rs.to_pandas()

def ST_CurveToLine(geos):
    arr_geos = pa.array(geos, type='string')
    rs = arctern_core_.ST_CurveToLine(arr_geos)
    return rs.to_pandas()


def point_map(xs, ys, conf):
    arr_x = pa.array(xs, type='uint32')
    arr_y = pa.array(ys, type='uint32')
    rs = arctern_core_.point_map(arr_x, arr_y, conf)
    return rs.buffers()[1].to_pybytes().hex()

def point_map_wkt(points, conf):
    array_points = pa.array(points, type='string')
    rs = arctern_core_.point_map_wkt(array_points, conf)
    return rs.buffers()[1].to_pybytes().hex()

def heat_map(x_data, y_data, c_data, conf):
    arr_x = pa.array(x_data, type='uint32')
    arr_y = pa.array(y_data, type='uint32')
    arr_c = pa.array(c_data, type='uint32')
    rs = arctern_core_.heat_map(arr_x, arr_y, arr_c, conf)
    return rs.buffers()[1].to_pybytes().hex()


def heat_map_wkt(points, c_data, conf):
    array_points = pa.array(points, type='string')

    if isinstance(c_data[0], float):
        arr_c = pa.array(c_data, type='double')
    else:
        arr_c = pa.array(c_data, type='int64')

    rs = arctern_core_.heat_map_wkt(array_points, arr_c, conf)
    return rs.buffers()[1].to_pybytes().hex()

def choropleth_map(wkt_data, count_data, conf):
    arr_wkt = pa.array(wkt_data, type='string')
    if isinstance(count_data[0], float):
        arr_count = pa.array(count_data, type='double')
    else:
        arr_count = pa.array(count_data, type='int64')
    rs = arctern_core_.choropleth_map(arr_wkt, arr_count, conf)
    return rs.buffers()[1].to_pybytes().hex()

def coordinate_projection(geos, top_left, bottom_right, height, width):
    arr_geos = pa.array(geos, type='string')
    src_rs1 = bytes(top_left, encoding="utf8")
    dst_rs1 = bytes(bottom_right, encoding="utf8")
    rs = arctern_core_.coordinate_projection(arr_geos, src_rs1, dst_rs1, height, width)
    return rs.to_pandas()
