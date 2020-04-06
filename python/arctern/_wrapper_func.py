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
    "point_map_wkb",
    "weighted_point_map",
    "weighted_point_map_wkb",
    "heat_map",
    "heat_map_wkb",
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


def point_map(xs, ys, conf):
    import pyarrow as pa
    arr_x = pa.array(xs, type='uint32')
    arr_y = pa.array(ys, type='uint32')
    rs = arctern_core_.point_map(arr_x, arr_y, conf)
    return base64.b64encode(rs.buffers()[1].to_pybytes())

def weighted_point_map(xs, ys, conf, **kwargs):
    import pyarrow as pa
    cs = kwargs.get('cs', None)
    ss = kwargs.get('ss', None)

    arr_x = pa.array(xs, type='uint32')
    arr_y = pa.array(ys, type='uint32')

    if (cs is None and ss is None):
        rs = arctern_core_.weighted_point_map_0_0(arr_x, arr_y, conf)
    elif (cs is not None and ss is not None):
        if isinstance(cs[0], float):
            arr_c = pa.array(cs, type='double')
        else:
            arr_c = pa.array(cs, type='int64')

        if isinstance(ss[0], float):
            arr_s = pa.array(ss, type='double')
        else:
            arr_s = pa.array(ss, type='int64')
        rs = arctern_core_.weighted_point_map_1_1(arr_x, arr_y, conf, arr_c, arr_s)
    elif (cs is None and ss is not None):
        if isinstance(ss[0], float):
            arr = pa.array(ss, type='double')
        else:
            arr = pa.array(ss, type='int64')
        rs = arctern_core_.weighted_point_map_0_1(arr_x, arr_y, conf, arr)
    else:
        if isinstance(cs[0], float):
            arr = pa.array(cs, type='double')
        else:
            arr = pa.array(cs, type='int64')
        rs = arctern_core_.weighted_point_map_1_0(arr_x, arr_y, conf, arr)

    return base64.b64encode(rs.buffers()[1].to_pybytes())

def weighted_point_map_wkb(points, conf, **kwargs):
    import pyarrow as pa
    cs = kwargs.get('cs', None)
    ss = kwargs.get('ss', None)

    array_points = pa.array(points, type='binary')

    if (cs is None and ss is None):
        rs = arctern_core_.weighted_point_map_wkb_0_0(array_points, conf)
    elif (cs is not None and ss is not None):
        if isinstance(cs[0], float):
            arr_c = pa.array(cs, type='double')
        else:
            arr_c = pa.array(cs, type='int64')

        if isinstance(ss[0], float):
            arr_s = pa.array(ss, type='double')
        else:
            arr_s = pa.array(ss, type='int64')
        rs = arctern_core_.weighted_point_map_wkb_1_1(array_points, conf, arr_c, arr_s)
    elif (cs is None and ss is not None):
        if isinstance(ss[0], float):
            arr = pa.array(ss, type='double')
        else:
            arr = pa.array(ss, type='int64')
        rs = arctern_core_.weighted_point_map_wkb_0_1(array_points, conf, arr)
    else:
        if isinstance(cs[0], float):
            arr = pa.array(cs, type='double')
        else:
            arr = pa.array(cs, type='int64')
        rs = arctern_core_.weighted_point_map_wkb_1_0(array_points, conf, arr)

    return base64.b64encode(rs.buffers()[1].to_pybytes())

def point_map_wkb(points, conf):
    import pyarrow as pa
    array_points = pa.array(points, type='binary')
    rs = arctern_core_.point_map_wkb(array_points, conf)
    return base64.b64encode(rs.buffers()[1].to_pybytes())

def heat_map(x_data, y_data, c_data, conf):
    import pyarrow as pa
    arr_x = pa.array(x_data, type='uint32')
    arr_y = pa.array(y_data, type='uint32')
    arr_c = pa.array(c_data, type='uint32')
    rs = arctern_core_.heat_map(arr_x, arr_y, arr_c, conf)
    return base64.b64encode(rs.buffers()[1].to_pybytes())


def heat_map_wkb(points, c_data, conf):
    import pyarrow as pa
    array_points = pa.array(points, type='binary')

    if isinstance(c_data[0], float):
        arr_c = pa.array(c_data, type='double')
    else:
        arr_c = pa.array(c_data, type='int64')

    rs = arctern_core_.heat_map_wkb(array_points, arr_c, conf)
    return base64.b64encode(rs.buffers()[1].to_pybytes())

def choropleth_map(wkb_data, count_data, conf):
    import pyarrow as pa
    arr_wkb = pa.array(wkb_data, type='binary')
    if isinstance(count_data[0], float):
        arr_count = pa.array(count_data, type='double')
    else:
        arr_count = pa.array(count_data, type='int64')
    rs = arctern_core_.choropleth_map(arr_wkb, arr_count, conf)
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
