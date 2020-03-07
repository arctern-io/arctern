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

# distutils: language = c++

from pyarrow.lib cimport *

cimport zilliz_gis_core__ as zilliz_gis_core_pxd

def point_map(arr_x, arr_y, conf):
    return pyarrow_wrap_array(zilliz_gis_core_pxd.point_map(pyarrow_unwrap_array(arr_x), pyarrow_unwrap_array(arr_y), conf))

def heat_map(arr_x, arr_y, arr_c, conf):
    return pyarrow_wrap_array(zilliz_gis_core_pxd.heat_map(pyarrow_unwrap_array(arr_x), pyarrow_unwrap_array(arr_y), pyarrow_unwrap_array(arr_c), conf))

def choropleth_map(arr_wkt, arr_count, conf):
    return pyarrow_wrap_array(zilliz_gis_core_pxd.choropleth_map(pyarrow_unwrap_array(arr_wkt), pyarrow_unwrap_array(arr_count), conf))

def ST_Point(object arr_x,object arr_y):
    return pyarrow_wrap_array(zilliz_gis_core_pxd.ST_Point(pyarrow_unwrap_array(arr_x),pyarrow_unwrap_array(arr_y)))

def ST_GeomFromGeoJSON(object json):
    return pyarrow_wrap_array(zilliz_gis_core_pxd.ST_GeomFromGeoJSON(pyarrow_unwrap_array(json)))

def ST_Intersection(object left_geometries,object right_geometries):
    return pyarrow_wrap_array(zilliz_gis_core_pxd.ST_Intersection(pyarrow_unwrap_array(left_geometries),pyarrow_unwrap_array(right_geometries)))

def ST_IsValid(object geometries):
    return pyarrow_wrap_array(zilliz_gis_core_pxd.ST_IsValid(pyarrow_unwrap_array(geometries)))

def ST_Equals(object left_geometries,object right_geometries):
    return pyarrow_wrap_array(zilliz_gis_core_pxd.ST_Equals(pyarrow_unwrap_array(left_geometries),pyarrow_unwrap_array(right_geometries)))

def ST_Touches(object left_geometries,object right_geometries):
    return pyarrow_wrap_array(zilliz_gis_core_pxd.ST_Touches(pyarrow_unwrap_array(left_geometries),pyarrow_unwrap_array(right_geometries)))

def ST_Overlaps(object left_geometries,object right_geometries):
    return pyarrow_wrap_array(zilliz_gis_core_pxd.ST_Overlaps(pyarrow_unwrap_array(left_geometries),pyarrow_unwrap_array(right_geometries)))

def ST_Crosses(object left_geometries,object right_geometries):
    return pyarrow_wrap_array(zilliz_gis_core_pxd.ST_Crosses(pyarrow_unwrap_array(left_geometries),pyarrow_unwrap_array(right_geometries)))

def ST_IsSimple(object geometries):
    return pyarrow_wrap_array(zilliz_gis_core_pxd.ST_IsSimple(pyarrow_unwrap_array(geometries)))

def ST_GeometryType(object geometries):
    return pyarrow_wrap_array(zilliz_gis_core_pxd.ST_GeometryType(pyarrow_unwrap_array(geometries)))

def ST_MakeValid(object geometries):
    return pyarrow_wrap_array(zilliz_gis_core_pxd.ST_MakeValid(pyarrow_unwrap_array(geometries)))

def ST_PrecisionReduce(object geometries,int num_dat):
    return pyarrow_wrap_array(zilliz_gis_core_pxd.ST_PrecisionReduce(pyarrow_unwrap_array(geometries),num_dat))

def ST_SimplifyPreserveTopology(object geometries,double distanceTolerance):
    return pyarrow_wrap_array(zilliz_gis_core_pxd.ST_SimplifyPreserveTopology(pyarrow_unwrap_array(geometries),distanceTolerance))

def ST_PolygonFromEnvelope(object min_x,object min_y,object max_x,object max_y):
    return pyarrow_wrap_array(zilliz_gis_core_pxd.ST_PolygonFromEnvelope(pyarrow_unwrap_array(min_x),pyarrow_unwrap_array(min_y),pyarrow_unwrap_array(max_x),pyarrow_unwrap_array(max_y)))

def ST_Contains(object ptr_x,object ptr_y):
    return pyarrow_wrap_array(zilliz_gis_core_pxd.ST_Contains(pyarrow_unwrap_array(ptr_x),pyarrow_unwrap_array(ptr_y)))

def ST_Intersects(object geo_arr1,object geo_arr2):
    return pyarrow_wrap_array(zilliz_gis_core_pxd.ST_Intersects(pyarrow_unwrap_array(geo_arr1),pyarrow_unwrap_array(geo_arr2)))

def ST_Within(object geo_arr1,object geo_arr2):
    return pyarrow_wrap_array(zilliz_gis_core_pxd.ST_Within(pyarrow_unwrap_array(geo_arr1),pyarrow_unwrap_array(geo_arr2)))

def ST_Distance(object geo_arr1,object geo_arr2):
    return pyarrow_wrap_array(zilliz_gis_core_pxd.ST_Distance(pyarrow_unwrap_array(geo_arr1),pyarrow_unwrap_array(geo_arr2)))

def ST_Area(object geo_arr):
    return pyarrow_wrap_array(zilliz_gis_core_pxd.ST_Area(pyarrow_unwrap_array(geo_arr)))

def ST_Centroid(object geo_arr):
    return pyarrow_wrap_array(zilliz_gis_core_pxd.ST_Centroid(pyarrow_unwrap_array(geo_arr)))

def ST_Length(object geo_arr):
    return pyarrow_wrap_array(zilliz_gis_core_pxd.ST_Length(pyarrow_unwrap_array(geo_arr)))

def ST_HausdorffDistance(object geo1, object geo2):
    return pyarrow_wrap_array(zilliz_gis_core_pxd.ST_HausdorffDistance(pyarrow_unwrap_array(geo1),pyarrow_unwrap_array(geo2)))

def ST_ConvexHull(object geo_arr):
    return pyarrow_wrap_array(zilliz_gis_core_pxd.ST_ConvexHull(pyarrow_unwrap_array(geo_arr)))

def ST_Transform(object geo_arr, bytes src_rs, bytes dst_rs):
    return pyarrow_wrap_array(zilliz_gis_core_pxd.ST_Transform(pyarrow_unwrap_array(geo_arr),src_rs,dst_rs))

def ST_CurveToLine(object geo_arr):
    return pyarrow_wrap_array(zilliz_gis_core_pxd.ST_CurveToLine(pyarrow_unwrap_array(geo_arr)))

def ST_NPoints(object geo_arr):
    return pyarrow_wrap_array(zilliz_gis_core_pxd.ST_NPoints(pyarrow_unwrap_array(geo_arr)))

def ST_Envelope(object geo_arr):
    return pyarrow_wrap_array(zilliz_gis_core_pxd.ST_Envelope(pyarrow_unwrap_array(geo_arr)))

def ST_Buffer(object geo_arr,double dfDist, n_quadrant_segments = None):
    if n_quadrant_segments is not None:
        return pyarrow_wrap_array(zilliz_gis_core_pxd.ST_Buffer(pyarrow_unwrap_array(geo_arr),dfDist,n_quadrant_segments))
    return pyarrow_wrap_array(zilliz_gis_core_pxd.ST_Buffer(pyarrow_unwrap_array(geo_arr),dfDist))

def ST_Union_Aggr(object geo_arr):
    return pyarrow_wrap_array(zilliz_gis_core_pxd.ST_Union_Aggr(pyarrow_unwrap_array(geo_arr)))

def ST_Envelope_Aggr(object geo_arr):
    return pyarrow_wrap_array(zilliz_gis_core_pxd.ST_Envelope_Aggr(pyarrow_unwrap_array(geo_arr)))
