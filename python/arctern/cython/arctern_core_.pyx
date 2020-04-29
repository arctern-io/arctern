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

from pyarrow.lib cimport (pyarrow_wrap_array, pyarrow_unwrap_array)
from libcpp.vector cimport vector
from pyarrow.lib cimport (shared_ptr, CArray)

cimport arctern_core__ as arctern_core_pxd


# render func api:
def projection(geos_list, bottom_right, top_left, height, width):
    cdef vector[shared_ptr[CArray]] geos_vector
    for geos in geos_list:
        arr = pyarrow_unwrap_array(geos)
        geos_vector.push_back(arr)
    cdef vector[shared_ptr[CArray]] output_geos
    output_geos = arctern_core_pxd.projection(geos_vector, bottom_right, top_left, height, width)
    res = []
    for i in range(output_geos.size()):
        res.append(pyarrow_wrap_array(output_geos[i]))
    return res

def transform_and_projection(geos_list, src_rs, dst_rs, bottom_right, top_left, height, width):
    cdef vector[shared_ptr[CArray]] geos_vector
    for geos in geos_list:
        arr = pyarrow_unwrap_array(geos)
        geos_vector.push_back(arr)
    cdef vector[shared_ptr[CArray]] output_geos
    output_geos = arctern_core_pxd.transform_and_projection(geos_vector, src_rs, dst_rs, bottom_right, top_left, height, width)
    res = []
    for i in range(output_geos.size()):
        res.append(pyarrow_wrap_array(output_geos[i]))
    return res

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



# gis api:
def ST_Point(object arr_x,object arr_y):
    return pyarrow_wrap_array(arctern_core_pxd.ST_Point(pyarrow_unwrap_array(arr_x),pyarrow_unwrap_array(arr_y)))

def ST_GeomFromGeoJSON(object json):
    return pyarrow_wrap_array(arctern_core_pxd.ST_GeomFromGeoJSON(pyarrow_unwrap_array(json)))

def ST_GeomFromText(object text):
    return pyarrow_wrap_array(arctern_core_pxd.ST_GeomFromText(pyarrow_unwrap_array(text)))

def ST_AsText(object text):
    return pyarrow_wrap_array(arctern_core_pxd.ST_AsText(pyarrow_unwrap_array(text)))

def ST_AsGeoJSON(object text):
    return pyarrow_wrap_array(arctern_core_pxd.ST_AsGeoJSON(pyarrow_unwrap_array(text)))

def ST_Intersection(object left_geometries,object right_geometries):
    return pyarrow_wrap_array(arctern_core_pxd.ST_Intersection(pyarrow_unwrap_array(left_geometries),pyarrow_unwrap_array(right_geometries)))

def ST_IsValid(object geometries):
    return pyarrow_wrap_array(arctern_core_pxd.ST_IsValid(pyarrow_unwrap_array(geometries)))

def ST_Equals(object left_geometries,object right_geometries):
    return pyarrow_wrap_array(arctern_core_pxd.ST_Equals(pyarrow_unwrap_array(left_geometries),pyarrow_unwrap_array(right_geometries)))

def ST_Touches(object left_geometries,object right_geometries):
    return pyarrow_wrap_array(arctern_core_pxd.ST_Touches(pyarrow_unwrap_array(left_geometries),pyarrow_unwrap_array(right_geometries)))

def ST_Overlaps(object left_geometries,object right_geometries):
    return pyarrow_wrap_array(arctern_core_pxd.ST_Overlaps(pyarrow_unwrap_array(left_geometries),pyarrow_unwrap_array(right_geometries)))

def ST_Crosses(object left_geometries,object right_geometries):
    return pyarrow_wrap_array(arctern_core_pxd.ST_Crosses(pyarrow_unwrap_array(left_geometries),pyarrow_unwrap_array(right_geometries)))

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
    return pyarrow_wrap_array(arctern_core_pxd.ST_PolygonFromEnvelope(pyarrow_unwrap_array(min_x),pyarrow_unwrap_array(min_y),pyarrow_unwrap_array(max_x),pyarrow_unwrap_array(max_y)))

def ST_Contains(object ptr_x,object ptr_y):
    return pyarrow_wrap_array(arctern_core_pxd.ST_Contains(pyarrow_unwrap_array(ptr_x),pyarrow_unwrap_array(ptr_y)))

def ST_Intersects(object geo_arr1,object geo_arr2):
    return pyarrow_wrap_array(arctern_core_pxd.ST_Intersects(pyarrow_unwrap_array(geo_arr1),pyarrow_unwrap_array(geo_arr2)))

def ST_Within(object geo_arr1,object geo_arr2):
    return pyarrow_wrap_array(arctern_core_pxd.ST_Within(pyarrow_unwrap_array(geo_arr1),pyarrow_unwrap_array(geo_arr2)))

def ST_Distance(object geo_arr1,object geo_arr2):
    return pyarrow_wrap_array(arctern_core_pxd.ST_Distance(pyarrow_unwrap_array(geo_arr1),pyarrow_unwrap_array(geo_arr2)))

def ST_DistanceSphere(object geo_arr1,object geo_arr2):
    return pyarrow_wrap_array(arctern_core_pxd.ST_DistanceSphere(pyarrow_unwrap_array(geo_arr1),pyarrow_unwrap_array(geo_arr2)))

def ST_Area(object geo_arr):
    return pyarrow_wrap_array(arctern_core_pxd.ST_Area(pyarrow_unwrap_array(geo_arr)))

def ST_Centroid(object geo_arr):
    return pyarrow_wrap_array(arctern_core_pxd.ST_Centroid(pyarrow_unwrap_array(geo_arr)))

def ST_Length(object geo_arr):
    return pyarrow_wrap_array(arctern_core_pxd.ST_Length(pyarrow_unwrap_array(geo_arr)))

def ST_HausdorffDistance(object geo1, object geo2):
    return pyarrow_wrap_array(arctern_core_pxd.ST_HausdorffDistance(pyarrow_unwrap_array(geo1),pyarrow_unwrap_array(geo2)))

def ST_ConvexHull(object geo_arr):
    return pyarrow_wrap_array(arctern_core_pxd.ST_ConvexHull(pyarrow_unwrap_array(geo_arr)))

def ST_Transform(object geo_arr, bytes src_rs, bytes dst_rs):
    return pyarrow_wrap_array(arctern_core_pxd.ST_Transform(pyarrow_unwrap_array(geo_arr),src_rs,dst_rs))

def ST_CurveToLine(object geo_arr):
    return pyarrow_wrap_array(arctern_core_pxd.ST_CurveToLine(pyarrow_unwrap_array(geo_arr)))

def ST_NPoints(object geo_arr):
    return pyarrow_wrap_array(arctern_core_pxd.ST_NPoints(pyarrow_unwrap_array(geo_arr)))

def ST_Envelope(object geo_arr):
    return pyarrow_wrap_array(arctern_core_pxd.ST_Envelope(pyarrow_unwrap_array(geo_arr)))

def ST_Buffer(object geo_arr,double dfDist, n_quadrant_segments = None):
    if n_quadrant_segments is not None:
        return pyarrow_wrap_array(arctern_core_pxd.ST_Buffer(pyarrow_unwrap_array(geo_arr),dfDist,n_quadrant_segments))
    return pyarrow_wrap_array(arctern_core_pxd.ST_Buffer(pyarrow_unwrap_array(geo_arr),dfDist))

def ST_Union_Aggr(object geo_arr):
    return pyarrow_wrap_array(arctern_core_pxd.ST_Union_Aggr(pyarrow_unwrap_array(geo_arr)))

def ST_Envelope_Aggr(object geo_arr):
    return pyarrow_wrap_array(arctern_core_pxd.ST_Envelope_Aggr(pyarrow_unwrap_array(geo_arr)))
