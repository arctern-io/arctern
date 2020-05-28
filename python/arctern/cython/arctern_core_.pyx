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

# cython: language_level=3
# distutils: language = c++

from pyarrow.lib cimport (shared_ptr, CArray, pyarrow_wrap_array, pyarrow_unwrap_array)
from libcpp.vector cimport vector
from libcpp.string cimport string
cimport arctern_core__ as arctern_core_pxd


# render func api:
def projection(geos_list, bottom_right, top_left, height, width):
    cdef vector[shared_ptr[CArray]] geos_vector
    for geos in geos_list:
        arr = pyarrow_unwrap_array(geos)
        geos_vector.push_back(arr)
    cdef vector[shared_ptr[CArray]] output_geos
    result = arctern_core_pxd.projection(geos_vector, bottom_right, top_left, height, width)
    return [pyarrow_wrap_array(ptr) for ptr in result]

def transform_and_projection(geos_list, src_rs, dst_rs, bottom_right, top_left, height, width):
    cdef vector[shared_ptr[CArray]] geos_vector
    for geos in geos_list:
        arr = pyarrow_unwrap_array(geos)
        geos_vector.push_back(arr)
    cdef vector[shared_ptr[CArray]] output_geos
    result = arctern_core_pxd.transform_and_projection(geos_vector, src_rs, dst_rs, bottom_right, top_left, height, width)
    return [pyarrow_wrap_array(ptr) for ptr in result]

def wkt2wkb(arr_wkt):
    return pyarrow_wrap_array(arctern_core_pxd.WktToWkb(pyarrow_unwrap_array(arr_wkt)))

def wkb2wkt(arr_wkb):
    return pyarrow_wrap_array(arctern_core_pxd.WkbToWkt(pyarrow_unwrap_array(arr_wkb)))


# render drawing api:
def point_map(vega, points_list):
    cdef vector[shared_ptr[CArray]] points_vector
    for points in points_list:
        arr = pyarrow_unwrap_array(points)
        points_vector.push_back(arr)
    return pyarrow_wrap_array(arctern_core_pxd.point_map(points_vector, vega))

def weighted_point_map(vega, points_list):
    cdef vector[shared_ptr[CArray]] points_vector
    for points in points_list:
        arr = pyarrow_unwrap_array(points)
        points_vector.push_back(arr)
    return pyarrow_wrap_array(arctern_core_pxd.weighted_point_map(points_vector, vega))

def weighted_color_point_map(vega, points_list, color_list):
    cdef vector[shared_ptr[CArray]] points_vector
    cdef vector[shared_ptr[CArray]] color_vector
    for points in points_list:
        arr = pyarrow_unwrap_array(points)
        points_vector.push_back(arr)
    for color in color_list:
        arr = pyarrow_unwrap_array(color)
        color_vector.push_back(arr)
    return pyarrow_wrap_array(arctern_core_pxd.weighted_point_map(points_vector, color_vector, vega))

def weighted_size_point_map(vega, points_list, size_list):
    cdef vector[shared_ptr[CArray]] points_vector
    cdef vector[shared_ptr[CArray]] size_vector
    for points in points_list:
        arr = pyarrow_unwrap_array(points)
        points_vector.push_back(arr)
    for size in size_list:
        arr = pyarrow_unwrap_array(size)
        size_vector.push_back(arr)
    return pyarrow_wrap_array(arctern_core_pxd.weighted_point_map(points_vector, size_vector, vega))

def weighted_color_size_point_map(vega, points_list, color_list, size_list):
    cdef vector[shared_ptr[CArray]] points_vector
    cdef vector[shared_ptr[CArray]] color_vector
    cdef vector[shared_ptr[CArray]] size_vector
    for points in points_list:
        arr = pyarrow_unwrap_array(points)
        points_vector.push_back(arr)
    for color in color_list:
        arr = pyarrow_unwrap_array(color)
        color_vector.push_back(arr)
    for size in size_list:
        arr = pyarrow_unwrap_array(size)
        size_vector.push_back(arr)
    return pyarrow_wrap_array(arctern_core_pxd.weighted_point_map(points_vector, color_vector, size_vector, vega))

def heat_map(vega, points_list, weights_list):
    cdef vector[shared_ptr[CArray]] points_vector
    cdef vector[shared_ptr[CArray]] weights_vector
    for points in points_list:
        arr = pyarrow_unwrap_array(points)
        points_vector.push_back(arr)
    for weights in weights_list:
        arr = pyarrow_unwrap_array(weights)
        weights_vector.push_back(arr)
    return pyarrow_wrap_array(arctern_core_pxd.heat_map(points_vector, weights_vector, vega))

def choropleth_map(vega,region_boundaries_list, weights_list):
    cdef vector[shared_ptr[CArray]] region_boundaries_vector
    cdef vector[shared_ptr[CArray]] weights_vector
    for region in region_boundaries_list:
        arr = pyarrow_unwrap_array(region)
        region_boundaries_vector.push_back(arr)
    for weights in weights_list:
        arr = pyarrow_unwrap_array(weights)
        weights_vector.push_back(arr)
    return pyarrow_wrap_array(arctern_core_pxd.choropleth_map(region_boundaries_vector, weights_vector, vega))

def icon_viz(vega, points_list):
    cdef vector[shared_ptr[CArray]] points_vector
    for points in points_list:
        arr = pyarrow_unwrap_array(points)
        points_vector.push_back(arr)

    return pyarrow_wrap_array(arctern_core_pxd.icon_viz(points_vector, vega))

def fishnet_map(vega, points_list, weights_list):
    cdef vector[shared_ptr[CArray]] points_vector
    cdef vector[shared_ptr[CArray]] weights_vector
    for points in points_list:
        arr = pyarrow_unwrap_array(points)
        points_vector.push_back(arr)
    for weights in weights_list:
        arr = pyarrow_unwrap_array(weights)
        weights_vector.push_back(arr)
    return pyarrow_wrap_array(arctern_core_pxd.fishnet_map(points_vector, weights_vector, vega))

# gis api:
def ST_Point(object arr_x,object arr_y):
    cdef vector[shared_ptr[CArray]] points_x
    for arr in arr_x:
        points_x.push_back(pyarrow_unwrap_array(arr))
    
    cdef vector[shared_ptr[CArray]] points_y
    for arr in arr_y:
        points_y.push_back(pyarrow_unwrap_array(arr))

    points = arctern_core_pxd.ST_Point(points_x, points_y)
    return [pyarrow_wrap_array(ptr) for ptr in points]

def ST_GeomFromGeoJSON(object json):
    result = arctern_core_pxd.ST_GeomFromGeoJSON(pyarrow_unwrap_array(json))
    return [pyarrow_wrap_array(ptr) for ptr in result]

def ST_GeomFromText(object text):
    result = arctern_core_pxd.ST_GeomFromText(pyarrow_unwrap_array(text))
    return [pyarrow_wrap_array(ptr) for ptr in result]

def ST_AsText(object text):
    result = arctern_core_pxd.ST_AsText(pyarrow_unwrap_array(text))
    return [pyarrow_wrap_array(ptr) for ptr in result]

def ST_AsGeoJSON(object text):
    result = arctern_core_pxd.ST_AsGeoJSON(pyarrow_unwrap_array(text))
    return [pyarrow_wrap_array(ptr) for ptr in result]

def ST_Intersection(object left_geometries,object right_geometries):
    cdef vector[shared_ptr[CArray]] left
    for geo in left_geometries:
        left.push_back(pyarrow_unwrap_array(geo))
    cdef vector[shared_ptr[CArray]] right
    for geo in right_geometries:
        right.push_back(pyarrow_unwrap_array(geo))
    result = arctern_core_pxd.ST_Intersection(left, right)
    return [pyarrow_wrap_array(ptr) for ptr in result]

def ST_IsValid(object geometries):
    return pyarrow_wrap_array(arctern_core_pxd.ST_IsValid(pyarrow_unwrap_array(geometries)))

def ST_Equals(object left_geometries,object right_geometries):
    cdef vector[shared_ptr[CArray]] left
    for geo in left_geometries:
        left.push_back(pyarrow_unwrap_array(geo))
    cdef vector[shared_ptr[CArray]] right
    for geo in right_geometries:
        right.push_back(pyarrow_unwrap_array(geo))
    result = arctern_core_pxd.ST_Equals(left, right)
    return [pyarrow_wrap_array(ptr) for ptr in result]

def ST_Touches(object left_geometries,object right_geometries):
    cdef vector[shared_ptr[CArray]] left
    for geo in left_geometries:
        left.push_back(pyarrow_unwrap_array(geo))
    cdef vector[shared_ptr[CArray]] right
    for geo in right_geometries:
        right.push_back(pyarrow_unwrap_array(geo))
    result = arctern_core_pxd.ST_Touches(left, right)
    return [pyarrow_wrap_array(ptr) for ptr in result]

def ST_Overlaps(object left_geometries,object right_geometries):
    cdef vector[shared_ptr[CArray]] left
    for geo in left_geometries:
        left.push_back(pyarrow_unwrap_array(geo))
    cdef vector[shared_ptr[CArray]] right
    for geo in right_geometries:
        right.push_back(pyarrow_unwrap_array(geo))
    result = arctern_core_pxd.ST_Overlaps(left, right)
    return [pyarrow_wrap_array(ptr) for ptr in result]

def ST_Crosses(object left_geometries,object right_geometries):
    cdef vector[shared_ptr[CArray]] left
    for geo in left_geometries:
        left.push_back(pyarrow_unwrap_array(geo))
    cdef vector[shared_ptr[CArray]] right
    for geo in right_geometries:
        right.push_back(pyarrow_unwrap_array(geo))
    result = arctern_core_pxd.ST_Crosses(left, right)
    return [pyarrow_wrap_array(ptr) for ptr in result]

def ST_IsSimple(object geometries):
    return pyarrow_wrap_array(arctern_core_pxd.ST_IsSimple(pyarrow_unwrap_array(geometries)))

def ST_GeometryType(object geometries):
    return pyarrow_wrap_array(arctern_core_pxd.ST_GeometryType(pyarrow_unwrap_array(geometries)))

def ST_MakeValid(object geometries):
    return pyarrow_wrap_array(arctern_core_pxd.ST_MakeValid(pyarrow_unwrap_array(geometries)))

def ST_PrecisionReduce(object geometries,int num_dat):
    return pyarrow_wrap_array(arctern_core_pxd.ST_PrecisionReduce(pyarrow_unwrap_array(geometries),num_dat))

def ST_SimplifyPreserveTopology(object geometries,double distanceTolerance):
    return pyarrow_wrap_array(arctern_core_pxd.ST_SimplifyPreserveTopology(pyarrow_unwrap_array(geometries),distanceTolerance))

def ST_PolygonFromEnvelope(object min_x,object min_y,object max_x,object max_y):
    cdef vector[shared_ptr[CArray]] min_x_arr
    for arr in min_x:
        min_x_arr.push_back(pyarrow_unwrap_array(arr))
    cdef vector[shared_ptr[CArray]] min_y_arr
    for arr in min_y:
        min_y_arr.push_back(pyarrow_unwrap_array(arr))
    cdef vector[shared_ptr[CArray]] max_x_arr
    for arr in max_x:
        max_x_arr.push_back(pyarrow_unwrap_array(arr))
    cdef vector[shared_ptr[CArray]] max_y_arr
    for arr in max_y:
        max_y_arr.push_back(pyarrow_unwrap_array(arr))

    result = arctern_core_pxd.ST_PolygonFromEnvelope(min_x_arr, min_y_arr, max_x_arr, max_y_arr)
    return [pyarrow_wrap_array(ptr) for ptr in result]

def ST_Contains(object left_geometries,object right_geometries):
    cdef vector[shared_ptr[CArray]] left
    for geo in left_geometries:
        left.push_back(pyarrow_unwrap_array(geo))
    cdef vector[shared_ptr[CArray]] right
    for geo in right_geometries:
        right.push_back(pyarrow_unwrap_array(geo))
    result = arctern_core_pxd.ST_Contains(left, right)
    return [pyarrow_wrap_array(ptr) for ptr in result]

def ST_Intersects(object left_geometries,object right_geometries):
    cdef vector[shared_ptr[CArray]] left
    for geo in left_geometries:
        left.push_back(pyarrow_unwrap_array(geo))
    cdef vector[shared_ptr[CArray]] right
    for geo in right_geometries:
        right.push_back(pyarrow_unwrap_array(geo))
    result = arctern_core_pxd.ST_Intersects(left, right)
    return [pyarrow_wrap_array(ptr) for ptr in result]

def ST_Within(object left_geometries,object right_geometries):
    cdef vector[shared_ptr[CArray]] left
    for geo in left_geometries:
        left.push_back(pyarrow_unwrap_array(geo))
    cdef vector[shared_ptr[CArray]] right
    for geo in right_geometries:
        right.push_back(pyarrow_unwrap_array(geo))
    result = arctern_core_pxd.ST_Within(left, right)
    return [pyarrow_wrap_array(ptr) for ptr in result]

def ST_Within2(left_geometries, right_geometry):
    cdef string s = right_geometry
    cdef vector[shared_ptr[CArray]] left
    for geo in left_geometries:
        left.push_back(pyarrow_unwrap_array(geo))
    result = arctern_core_pxd.ST_Within(left, s)
    return [pyarrow_wrap_array(ptr) for ptr in result]

def ST_Distance(object geo_arr1,object geo_arr2):
    cdef vector[shared_ptr[CArray]] arr_1
    for arr in geo_arr1:
        arr_1.push_back(pyarrow_unwrap_array(arr))
    cdef vector[shared_ptr[CArray]] arr_2
    for arr in geo_arr2:
        arr_2.push_back(pyarrow_unwrap_array(arr))
    result = arctern_core_pxd.ST_Distance(arr_1, arr_2)
    return [pyarrow_wrap_array(ptr) for ptr in result]

def ST_DistanceSphere(object geo_arr1,object geo_arr2):
    cdef vector[shared_ptr[CArray]] arr_1
    for arr in geo_arr1:
        arr_1.push_back(pyarrow_unwrap_array(arr))
    cdef vector[shared_ptr[CArray]] arr_2
    for arr in geo_arr2:
        arr_2.push_back(pyarrow_unwrap_array(arr))
    result = arctern_core_pxd.ST_DistanceSphere(arr_1, arr_2)
    return [pyarrow_wrap_array(ptr) for ptr in result]

def ST_Area(object geo_arr):
    return pyarrow_wrap_array(arctern_core_pxd.ST_Area(pyarrow_unwrap_array(geo_arr)))

def ST_Centroid(object geo_arr):
    return pyarrow_wrap_array(arctern_core_pxd.ST_Centroid(pyarrow_unwrap_array(geo_arr)))

def ST_Length(object geo_arr):
    return pyarrow_wrap_array(arctern_core_pxd.ST_Length(pyarrow_unwrap_array(geo_arr)))

def ST_HausdorffDistance(object geo1, object geo2):
    cdef vector[shared_ptr[CArray]] arr_1
    for arr in geo1:
        arr_1.push_back(pyarrow_unwrap_array(arr))
    cdef vector[shared_ptr[CArray]] arr_2
    for arr in geo2:
        arr_2.push_back(pyarrow_unwrap_array(arr))
    result = arctern_core_pxd.ST_HausdorffDistance(arr_1, arr_2)
    return [pyarrow_wrap_array(ptr) for ptr in result]

def ST_ConvexHull(object geo_arr):
    return pyarrow_wrap_array(arctern_core_pxd.ST_ConvexHull(pyarrow_unwrap_array(geo_arr)))

def ST_Transform(object geo_arr, bytes src_rs, bytes dst_rs):
    return pyarrow_wrap_array(arctern_core_pxd.ST_Transform(pyarrow_unwrap_array(geo_arr),src_rs,dst_rs))

def ST_CurveToLine(object geo_arr):
    result = arctern_core_pxd.ST_CurveToLine(pyarrow_unwrap_array(geo_arr))
    return [pyarrow_wrap_array(ptr) for ptr in result]

def ST_NPoints(object geo_arr):
    return pyarrow_wrap_array(arctern_core_pxd.ST_NPoints(pyarrow_unwrap_array(geo_arr)))

def ST_Envelope(object geo_arr):
    return pyarrow_wrap_array(arctern_core_pxd.ST_Envelope(pyarrow_unwrap_array(geo_arr)))

def ST_Buffer(object geo_arr,double dfDist, n_quadrant_segments = None):
    if n_quadrant_segments is not None:
        result = arctern_core_pxd.ST_Buffer(pyarrow_unwrap_array(geo_arr),dfDist,n_quadrant_segments)
        return [pyarrow_wrap_array(ptr) for ptr in result]
    else:
        result = arctern_core_pxd.ST_Buffer(pyarrow_unwrap_array(geo_arr),dfDist)
        return [pyarrow_wrap_array(ptr) for ptr in result]

def ST_Union_Aggr(object geo_arr):
    return pyarrow_wrap_array(arctern_core_pxd.ST_Union_Aggr(pyarrow_unwrap_array(geo_arr)))

def ST_Envelope_Aggr(object geo_arr):
    return pyarrow_wrap_array(arctern_core_pxd.ST_Envelope_Aggr(pyarrow_unwrap_array(geo_arr)))

def GIS_Version():
    return arctern_core_pxd.GIS_Version()
