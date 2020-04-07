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
    "ST_AsText",
    "point_map",
    "weighted_point_map",
    "heat_map",
    "choropleth_map",
    "projection",
    "transform_and_projection",
    "wkt2wkb",
    "wkb2wkt",
]

import base64
from . import arctern_core_

def ST_Point(x, y):
    import pyarrow as pa
    arr_x = pa.array(x, type='double')
    arr_y = pa.array(y, type='double')
    rs = arctern_core_.ST_Point(arr_x, arr_y)
    return rs.to_pandas()

def ST_GeomFromGeoJSON(json):
    import pyarrow as pa
    geo = pa.array(json, type='string')
    rs = arctern_core_.ST_GeomFromGeoJSON(geo)
    return rs.to_pandas()

def ST_GeomFromText(text):
    import pyarrow as pa
    geo = pa.array(text, type='string')
    rs = arctern_core_.ST_GeomFromText(geo)
    return rs.to_pandas()

def ST_AsText(text):
    import pyarrow as pa
    geo = pa.array(text, type='binary')
    rs = arctern_core_.ST_AsText(geo)
    return rs.to_pandas()

def ST_Intersection(left, right):
    import pyarrow as pa
    arr_left = pa.array(left, type='binary')
    arr_right = pa.array(right, type='binary')
    rs = arctern_core_.ST_Intersection(arr_left, arr_right)
    return rs.to_pandas()

def ST_IsValid(geos):
    import pyarrow as pa
    arr_geos = pa.array(geos, type='binary')
    rs = arctern_core_.ST_IsValid(arr_geos)
    return rs.to_pandas()

def ST_PrecisionReduce(geos, precision):
    import pyarrow as pa
    arr_geos = pa.array(geos, type='binary')
    rs = arctern_core_.ST_PrecisionReduce(arr_geos, precision)
    return rs.to_pandas()

def ST_Equals(left, right):
    import pyarrow as pa
    arr_left = pa.array(left, type='binary')
    arr_right = pa.array(right, type='binary')
    rs = arctern_core_.ST_Equals(arr_left, arr_right)
    return rs.to_pandas()

def ST_Touches(left, right):
    import pyarrow as pa
    arr_left = pa.array(left, type='binary')
    arr_right = pa.array(right, type='binary')
    rs = arctern_core_.ST_Touches(arr_left, arr_right)
    return rs.to_pandas()

def ST_Overlaps(left, right):
    import pyarrow as pa
    arr_left = pa.array(left, type='binary')
    arr_right = pa.array(right, type='binary')
    rs = arctern_core_.ST_Overlaps(arr_left, arr_right)
    return rs.to_pandas()

def ST_Crosses(left, right):
    import pyarrow as pa
    arr_left = pa.array(left, type='binary')
    arr_right = pa.array(right, type='binary')
    rs = arctern_core_.ST_Crosses(arr_left, arr_right)
    return rs.to_pandas()

def ST_IsSimple(geos):
    import pyarrow as pa
    arr_geos = pa.array(geos, type='binary')
    rs = arctern_core_.ST_IsSimple(arr_geos)
    return rs.to_pandas()

def ST_GeometryType(geos):
    import pyarrow as pa
    arr_geos = pa.array(geos, type='binary')
    rs = arctern_core_.ST_GeometryType(arr_geos)
    return rs.to_pandas()

def ST_MakeValid(geos):
    import pyarrow as pa
    arr_geos = pa.array(geos, type='binary')
    rs = arctern_core_.ST_MakeValid(arr_geos)
    return rs.to_pandas()

def ST_SimplifyPreserveTopology(geos, distance_tolerance):
    import pyarrow as pa
    arr_geos = pa.array(geos, type='binary')
    rs = arctern_core_.ST_SimplifyPreserveTopology(arr_geos, distance_tolerance)
    return rs.to_pandas()

def ST_PolygonFromEnvelope(min_x, min_y, max_x, max_y):
    import pyarrow as pa
    arr_min_x = pa.array(min_x, type='double')
    arr_min_y = pa.array(min_y, type='double')
    arr_max_x = pa.array(max_x, type='double')
    arr_max_y = pa.array(max_y, type='double')
    rs = arctern_core_.ST_PolygonFromEnvelope(arr_min_x, arr_min_y, arr_max_x, arr_max_y)
    return rs.to_pandas()

def ST_Contains(left, right):
    import pyarrow as pa
    arr_left = pa.array(left, type='binary')
    arr_right = pa.array(right, type='binary')
    rs = arctern_core_.ST_Contains(arr_left, arr_right)
    return rs.to_pandas()

def ST_Intersects(left, right):
    import pyarrow as pa
    arr_left = pa.array(left, type='binary')
    arr_right = pa.array(right, type='binary')
    rs = arctern_core_.ST_Intersects(arr_left, arr_right)
    return rs.to_pandas()

def ST_Within(left, right):
    import pyarrow as pa
    arr_left = pa.array(left, type='binary')
    arr_right = pa.array(right, type='binary')
    rs = arctern_core_.ST_Within(arr_left, arr_right)
    return rs.to_pandas()

def ST_Distance(left, right):
    import pyarrow as pa
    arr_left = pa.array(left, type='binary')
    arr_right = pa.array(right, type='binary')
    rs = arctern_core_.ST_Distance(arr_left, arr_right)
    return rs.to_pandas()

def ST_Area(geos):
    import pyarrow as pa
    arr_geos = pa.array(geos, type='binary')
    rs = arctern_core_.ST_Area(arr_geos)
    return rs.to_pandas()

def ST_Centroid(geos):
    import pyarrow as pa
    arr_geos = pa.array(geos, type='binary')
    rs = arctern_core_.ST_Centroid(arr_geos)
    return rs.to_pandas()

def ST_Length(geos):
    import pyarrow as pa
    arr_geos = pa.array(geos, type='binary')
    rs = arctern_core_.ST_Length(arr_geos)
    return rs.to_pandas()

def ST_HausdorffDistance(geo1, geo2):
    import pyarrow as pa
    arr1 = pa.array(geo1, type='binary')
    arr2 = pa.array(geo2, type='binary')
    rs = arctern_core_.ST_HausdorffDistance(arr1, arr2)
    return rs.to_pandas()

def ST_ConvexHull(geos):
    import pyarrow as pa
    arr_geos = pa.array(geos, type='binary')
    rs = arctern_core_.ST_ConvexHull(arr_geos)
    return rs.to_pandas()

def ST_NPoints(geos):
    import pyarrow as pa
    arr_geos = pa.array(geos, type='binary')
    rs = arctern_core_.ST_NPoints(arr_geos)
    return rs.to_pandas()

def ST_Envelope(geos):
    import pyarrow as pa
    arr_geos = pa.array(geos, type='binary')
    rs = arctern_core_.ST_Envelope(arr_geos)
    return rs.to_pandas()

def ST_Buffer(geos, distance):
    import pyarrow as pa
    arr_geos = pa.array(geos, type='binary')
    rs = arctern_core_.ST_Buffer(arr_geos, distance)
    return rs.to_pandas()

def ST_Union_Aggr(geos):
    import pyarrow as pa
    arr_geos = pa.array(geos, type='binary')
    rs = arctern_core_.ST_Union_Aggr(arr_geos)
    return rs.to_pandas()

def ST_Envelope_Aggr(geos):
    import pyarrow as pa
    arr_geos = pa.array(geos, type='binary')
    rs = arctern_core_.ST_Envelope_Aggr(arr_geos)
    return rs.to_pandas()

def ST_Transform(geos, src, dst):
    import pyarrow as pa
    arr_geos = pa.array(geos, type='binary')
    src = bytes(src, encoding="utf8")
    dst = bytes(dst, encoding="utf8")

    rs = arctern_core_.ST_Transform(arr_geos, src, dst)
    return rs.to_pandas()

def ST_CurveToLine(geos):
    import pyarrow as pa
    arr_geos = pa.array(geos, type='binary')
    rs = arctern_core_.ST_CurveToLine(arr_geos)
    return rs.to_pandas()

def point_map(vega, points):
    import pyarrow as pa
    array_points = pa.array(points, type='binary')
    vega_string = vega.build().encode('utf-8')
    rs = arctern_core_.point_map(vega_string, array_points)
    return base64.b64encode(rs.buffers()[1].to_pybytes())

def weighted_point_map(vega, points, **kwargs):
    import pyarrow as pa
    color_weights = kwargs.get('color_weights', None)
    size_weights = kwargs.get('size_weights', None)
    vega_string = vega.build().encode('utf-8')

    array_points = pa.array(points, type='binary')
    if (color_weights is None and size_weights is None):
        rs = arctern_core_.weighted_point_map(vega_string, array_points)
    elif (color_weights is not None and size_weights is not None):
        if isinstance(color_weights[0], float):
            arr_c = pa.array(color_weights, type='double')
        else:
            arr_c = pa.array(color_weights, type='int64')

        if isinstance(size_weights[0], float):
            arr_s = pa.array(size_weights, type='double')
        else:
            arr_s = pa.array(size_weights, type='int64')
        rs = arctern_core_.weighted_color_size_point_map(vega_string, array_points, arr_c, arr_s)
    elif (color_weights is None and size_weights is not None):
        if isinstance(size_weights[0], float):
            arr_s = pa.array(size_weights, type='double')
        else:
            arr_s = pa.array(size_weights, type='int64')
        rs = arctern_core_.weighted_size_point_map(vega_string, array_points, arr_s)
    else:
        if isinstance(color_weights[0], float):
            arr_c = pa.array(color_weights, type='double')
        else:
            arr_c = pa.array(color_weights, type='int64')
        rs = arctern_core_.weighted_color_point_map(vega_string, array_points, arr_c)

    return base64.b64encode(rs.buffers()[1].to_pybytes())

def heat_map(vega, points, weights):
    import pyarrow as pa
    array_points = pa.array(points, type='binary')
    vega_string = vega.build().encode('utf-8')

    if isinstance(weights[0], float):
        arr_c = pa.array(weights, type='double')
    else:
        arr_c = pa.array(weights, type='int64')

    rs = arctern_core_.heat_map(vega_string, array_points, arr_c)
    return base64.b64encode(rs.buffers()[1].to_pybytes())

def choropleth_map(vega, region_boundaries, weights):
    import pyarrow as pa
    arr_wkb = pa.array(region_boundaries, type='binary')
    vega_string = vega.build().encode('utf-8')

    if isinstance(weights[0], float):
        arr_c = pa.array(weights, type='double')
    else:
        arr_c = pa.array(weights, type='int64')
    rs = arctern_core_.choropleth_map(vega_string, arr_wkb, arr_c)
    return base64.b64encode(rs.buffers()[1].to_pybytes())

def projection(geos, bottom_right, top_left, height, width):
    import pyarrow as pa
    arr_geos = pa.array(geos, type='binary')
    bounding_box_min = bytes(bottom_right, encoding="utf8")
    bounding_box_max = bytes(top_left, encoding="utf8")
    rs = arctern_core_.projection(arr_geos, bounding_box_min, bounding_box_max, height, width)
    return rs.to_pandas()

def transform_and_projection(geos, src_rs, dst_rs, bottom_right, top_left, height, width):
    import pyarrow as pa
    arr_geos = pa.array(geos, type='binary')
    src = bytes(src_rs, encoding="utf8")
    dst = bytes(dst_rs, encoding="utf8")
    bounding_box_min = bytes(bottom_right, encoding="utf8")
    bounding_box_max = bytes(top_left, encoding="utf8")
    rs = arctern_core_.transform_and_projection(arr_geos, src, dst, bounding_box_min, bounding_box_max, height, width)
    return rs.to_pandas()

def wkt2wkb(arr_wkt):
    import pyarrow as pa
    wkts = pa.array(arr_wkt, type='string')
    rs = arctern_core_.wkt2wkb(wkts)
    return rs.to_pandas()

def wkb2wkt(arr_wkb):
    import pyarrow as pa
    wkbs = pa.array(arr_wkb, type='binary')
    rs = arctern_core_.wkb2wkt(wkbs)
    return rs.to_pandas()
